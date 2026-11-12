// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Factorization kernels for the block-sparse cores: `my_qc`/`my_cq`, `qc`/`cq`, rank-preserving
//! `qr`/`lq`, `svd_horizontal`/`svd_vertical`, the `chop` truncation rules, `reduce_ranks`, and the
//! [`Sketch`] factor object with its `lmul`/`rmul`/`*` folds, plus the horizontal/vertical
//! unfolding assemblers.

mod chop;
mod coreops;
mod dense_solver_kernels;
mod sketch;

pub(crate) use chop::chop_global;
pub use chop::{chop_energy, chop_rank};
#[cfg(test)]
pub(crate) use coreops::thin_svd_qr;
pub use coreops::{my_cq, svd_uvt_svals};
pub(crate) use coreops::{my_qc, stewart_frame, svals_qr, thin_q};
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use dense_solver_kernels::{SignedRitzPath, signed_ritz};
pub(crate) use dense_solver_kernels::{cluster_null_space, dense_lstsq_colpiv};
pub use dense_solver_kernels::{
    dense_eigen_general, dense_self_adjoint_eigen, dense_self_adjoint_smallest, tridiag_exp_e1,
};
pub(crate) use sketch::core_times_adjoint_core_seq;
pub use sketch::{
    Sketch, adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
};

use std::sync::Once;

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};

use crate::scalar::Scalar;

fn ensure_seq() {
    static PIN: Once = Once::new();
    PIN.call_once(|| faer::set_global_parallelism(Par::Seq));
}

pub(crate) fn gemm<T: Scalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let mut c = Mat::<T>::zeros(a.nrows(), b.ncols());
    if a.ncols() > 0 && a.nrows() > 0 && b.ncols() > 0 {
        matmul(c.as_mut(), Accum::Replace, a, b, T::one(), Par::Seq);
    }
    c
}

const GEMM_NT_ROW_BLOCK: usize = 64;

pub(crate) fn gemm_nt_rowblocks(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    use rayon::prelude::*;
    assert!(
        a.len() == m * k && b.len() == n * k,
        "gemm_nt_rowblocks: a holds {} entries (m·k = {}), b holds {} (n·k = {})",
        a.len(),
        m * k,
        b.len(),
        n * k
    );
    let mut out = vec![0.0f64; m * n];
    if m == 0 || n == 0 || k == 0 {
        return out;
    }
    out.par_chunks_mut(GEMM_NT_ROW_BLOCK * n)
        .enumerate()
        .for_each(|(blk, chunk)| {
            let rows = chunk.len() / n;
            let r0 = blk * GEMM_NT_ROW_BLOCK;
            let ab = MatRef::from_row_major_slice(&a[r0 * k..(r0 + rows) * k], rows, k);
            let bt = MatRef::from_row_major_slice(b, n, k).transpose();
            let dst = MatMut::from_row_major_slice_mut(chunk, rows, n);
            matmul(dst, Accum::Replace, ab, bt, 1.0, Par::Seq);
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(len: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..len)
            .map(|f| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                if f % 37 == 5 {
                    -0.0
                } else if f % 41 == 7 {
                    0.0
                } else {
                    2.0 * ((s >> 11) as f64) / ((1u64 << 53) as f64) - 1.0
                }
            })
            .collect()
    }

    fn pool(n: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon pool")
    }

    #[test]
    fn gemm_nt_rowblocks_matches_naive_and_is_pool_invariant() {
        for &(m, n, k) in &[
            (150usize, 37usize, 29usize),
            (64, 5, 7),
            (65, 5, 7),
            (25, 37, 29),
            (1, 1, 1),
            (0, 4, 3),
            (4, 0, 3),
            (4, 3, 0),
        ] {
            let what = format!("m = {m}, n = {n}, k = {k}");
            let a = data(m * k, 0xA11C_E000 + m as u64);
            let b = data(n * k, 0xB0B0_0000 + n as u64);
            let c1 = pool(1).install(|| gemm_nt_rowblocks(&a, &b, m, n, k));
            let c4 = pool(4).install(|| gemm_nt_rowblocks(&a, &b, m, n, k));
            assert_eq!(c1.len(), m * n, "{what}: output length");
            for (f, (x, y)) in c1.iter().zip(&c4).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{what}: C[{f}] pool 1 {x:e} vs pool 4 {y:e}"
                );
            }
            for i in 0..m {
                for j in 0..n {
                    let (mut s, mut mag) = (0.0f64, 0.0f64);
                    for l in 0..k {
                        s += a[i * k + l] * b[j * k + l];
                        mag += (a[i * k + l] * b[j * k + l]).abs();
                    }
                    let got = c1[i * n + j];
                    assert!(
                        (got - s).abs() <= 1e-13 * mag.max(f64::MIN_POSITIVE),
                        "{what}: C[{i},{j}] = {got:e} vs naive {s:e}"
                    );
                    if k == 0 {
                        assert_eq!(
                            got.to_bits(),
                            0.0f64.to_bits(),
                            "{what}: k = 0 must give +0.0"
                        );
                    }
                }
            }
        }

        let (m, n, k) = (150usize, 37usize, 29usize);
        let a = data(m * k, 0xA11C_E000 + m as u64);
        let b = data(n * k, 0xB0B0_0000 + n as u64);
        let full = gemm_nt_rowblocks(&a, &b, m, n, k);
        for (r0, r1) in [(64usize, 128usize), (128, 150), (0, 64)] {
            let rows = r1 - r0;
            let part = gemm_nt_rowblocks(&a[r0 * k..r1 * k], &b, rows, n, k);
            for (f, (x, y)) in full[r0 * n..r1 * n].iter().zip(&part).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "rows [{r0}, {r1}): entry {f} of the full product {x:e} vs the block alone {y:e}"
                );
            }
        }
    }
}
