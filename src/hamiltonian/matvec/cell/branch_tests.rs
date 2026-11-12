// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;
use crate::hamiltonian::csx::{CooCell, csx_pair};
use crate::scalar::gemm_par_hook::{Hook, pool};
use faer::Par;

fn fill(r: usize, c: usize, seed: u64) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(r, c);
    let mut s = seed | 1;
    for j in 0..c {
        for i in 0..r {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *m.get_mut(i, j) = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
        }
    }
    m
}

fn ranges(n: usize, w: usize) -> Vec<Range<usize>> {
    (0..n).map(|i| (i * w)..((i + 1) * w)).collect()
}

fn coo_of(rows: &[i32], cols: &[i32]) -> CooCell<f64> {
    let vals = (0..rows.len()).map(|t| 0.5 + 0.13 * t as f64).collect();
    CooCell {
        rows: rows.to_vec(),
        cols: cols.to_vec(),
        vals,
    }
}

fn max_close(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut e = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            e = e.max((*a.get(i, j) - *b.get(i, j)).abs());
        }
    }
    e
}

fn branch_id(coo: &CooCell<f64>) -> u8 {
    let (csr, csc) = csx_pair(coo);
    let (nnz, nrow, ncol) = (csr.nnz(), csr.span_len(), csc.span_len());
    if ncol < nnz.min(nrow) {
        2
    } else if nrow < nnz.min(ncol) {
        1
    } else {
        0
    }
}

const COO_ROWS: [i32; 3] = [0, 1, 2];
const COO_COLS: [i32; 3] = [0, 1, 2];
const CSC_ROWS: [i32; 6] = [0, 1, 2, 3, 4, 5];
const CSC_COLS: [i32; 6] = [0, 0, 0, 1, 1, 1];
const CSR_ROWS: [i32; 6] = [0, 0, 0, 1, 1, 1];
const CSR_COLS: [i32; 6] = [0, 1, 2, 3, 4, 5];

#[test]
fn cell_apply_left_all_branches() {
    let (xr, xc, fr) = (2usize, 3usize, 4usize);
    for (rows, cols, n_i, n_j, want) in [
        (&COO_ROWS[..], &COO_COLS[..], 3, 3, 0u8),
        (&CSR_ROWS[..], &CSR_COLS[..], 2, 6, 1u8),
        (&CSC_ROWS[..], &CSC_COLS[..], 6, 2, 2u8),
    ] {
        let coo = coo_of(rows, cols);
        assert_eq!(
            branch_id(&coo),
            want,
            "pattern did not force the intended branch"
        );
        let (csr, csc) = csx_pair(&coo);
        let i_ranges = ranges(n_i, xr);
        let j_ranges = ranges(n_j, xc);
        let f = fill(fr, n_i * xr, 11);
        let x = fill(xr, xc, 22);

        let mut yb = Mat::<f64>::zeros(fr, n_j * xc);
        cell_apply_left(
            yb.as_mut(),
            f.as_ref(),
            x.as_ref(),
            &i_ranges,
            &j_ranges,
            &csr,
            &csc,
            ColShard::ALL,
        );

        let mut yref = Mat::<f64>::zeros(fr, n_j * xc);
        for t in 0..coo.rows.len() {
            let ir = &i_ranges[coo.rows[t] as usize];
            let jr = &j_ranges[coo.cols[t] as usize];
            let fsub = f.submatrix(0, ir.start, fr, ir.len());
            let dst = yref.as_mut().submatrix_mut(0, jr.start, fr, jr.len());
            matmul(dst, Accum::Add, fsub, x.as_ref(), coo.vals[t], Par::Seq);
        }
        assert!(max_close(&yb, &yref) < 1e-13, "left branch {want} mismatch");
    }
}

#[test]
fn cell_apply_right_all_branches() {
    let (xr, xc, fc) = (2usize, 3usize, 4usize);
    for (rows, cols, n_i, n_j, want) in [
        (&COO_ROWS[..], &COO_COLS[..], 3, 3, 0u8),
        (&CSR_ROWS[..], &CSR_COLS[..], 2, 6, 1u8),
        (&CSC_ROWS[..], &CSC_COLS[..], 6, 2, 2u8),
    ] {
        let coo = coo_of(rows, cols);
        assert_eq!(
            branch_id(&coo),
            want,
            "pattern did not force the intended branch"
        );
        let (csr, csc) = csx_pair(&coo);
        let i_ranges = ranges(n_i, xr);
        let j_ranges = ranges(n_j, xc);
        let f = fill(n_j * xc, fc, 33);
        let x = fill(xr, xc, 44);

        let mut yb = Mat::<f64>::zeros(n_i * xr, fc);
        cell_apply_right(
            yb.as_mut(),
            x.as_ref(),
            f.as_ref(),
            &i_ranges,
            &j_ranges,
            &csr,
            &csc,
        );

        let mut yref = Mat::<f64>::zeros(n_i * xr, fc);
        for t in 0..coo.rows.len() {
            let ir = &i_ranges[coo.rows[t] as usize];
            let jr = &j_ranges[coo.cols[t] as usize];
            let fsub = f.submatrix(jr.start, 0, jr.len(), fc);
            let dst = yref.as_mut().submatrix_mut(ir.start, 0, ir.len(), fc);
            matmul(dst, Accum::Add, x.as_ref(), fsub, coo.vals[t], Par::Seq);
        }
        assert!(
            max_close(&yb, &yref) < 1e-13,
            "right branch {want} mismatch"
        );
    }
}

fn apply_both(
    rows: &[i32],
    cols: &[i32],
    n_i: usize,
    n_j: usize,
    (xr, xc, fd): (usize, usize, usize),
) -> (Mat<f64>, Mat<f64>) {
    let coo = coo_of(rows, cols);
    let (csr, csc) = csx_pair(&coo);
    let (i_ranges, j_ranges) = (ranges(n_i, xr), ranges(n_j, xc));
    let x = fill(xr, xc, 22);
    let fl = fill(fd, n_i * xr, 11);
    let mut yl = Mat::<f64>::zeros(fd, n_j * xc);
    cell_apply_left(
        yl.as_mut(),
        fl.as_ref(),
        x.as_ref(),
        &i_ranges,
        &j_ranges,
        &csr,
        &csc,
        ColShard::ALL,
    );
    let fr = fill(n_j * xc, fd, 33);
    let mut yr = Mat::<f64>::zeros(n_i * xr, fd);
    cell_apply_right(
        yr.as_mut(),
        x.as_ref(),
        fr.as_ref(),
        &i_ranges,
        &j_ranges,
        &csr,
        &csc,
    );
    (yl, yr)
}

fn frob(a: &Mat<f64>) -> f64 {
    a.norm_l2()
}

fn assert_bits(a: &Mat<f64>, b: &Mat<f64>, what: &str) {
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_eq!(
                a.get(i, j).to_bits(),
                b.get(i, j).to_bits(),
                "{what}: ({i},{j})"
            );
        }
    }
}

#[test]
fn cell_apply_gated_matches_seq_all_branches() {
    let dims = (40usize, 48usize, 64usize);
    for (rows, cols, n_i, n_j, want) in [
        (&COO_ROWS[..], &COO_COLS[..], 3, 3, 0u8),
        (&CSR_ROWS[..], &CSR_COLS[..], 2, 6, 1u8),
        (&CSC_ROWS[..], &CSC_COLS[..], 6, 2, 2u8),
    ] {
        assert_eq!(branch_id(&coo_of(rows, cols)), want);
        let one = Hook::forced();
        let (l1, r1) = pool(1, &one).install(|| apply_both(rows, cols, n_i, n_j, dims));
        let (l1b, r1b) = pool(1, &one).install(|| apply_both(rows, cols, n_i, n_j, dims));
        assert_eq!(
            one.taken(),
            0,
            "branch {want}: a 1-thread pool threaded a GEMM"
        );
        assert_bits(&l1, &l1b, &format!("left branch {want}, 1-thread re-run"));
        assert_bits(&r1, &r1b, &format!("right branch {want}, 1-thread re-run"));
        for nt in [8usize, 18] {
            let hook = Hook::forced();
            let (ln, rn) = pool(nt, &hook).install(|| apply_both(rows, cols, n_i, n_j, dims));
            assert!(
                hook.taken() > 0,
                "branch {want}, {nt} threads: no gated GEMM threaded ({} of {} calls)",
                hook.taken(),
                hook.calls()
            );
            for (side, y1, yn) in [("left", &l1, &ln), ("right", &r1, &rn)] {
                let mut d = yn.clone();
                d -= y1;
                let rel = frob(&d) / frob(y1);
                println!("{side} branch {want}, 1 vs {nt} threads (forced): ‖Δ‖/‖y‖ = {rel:.3e}");
                assert!(
                    rel <= 1e-12,
                    "{side} branch {want}, {nt} threads: {rel:.3e}"
                );
            }
        }
    }
}

#[test]
fn cell_apply_small_blocks_never_gated() {
    for (rows, cols, n_i, n_j, want) in [
        (&COO_ROWS[..], &COO_COLS[..], 3, 3, 0u8),
        (&CSR_ROWS[..], &CSR_COLS[..], 2, 6, 1u8),
        (&CSC_ROWS[..], &CSC_COLS[..], 6, 2, 2u8),
    ] {
        let dims = (40usize, 48usize, 64usize);
        let (l1, r1) =
            pool(1, &Hook::observed()).install(|| apply_both(rows, cols, n_i, n_j, dims));
        let hook = Hook::observed();
        let (ln, rn) = pool(18, &hook).install(|| apply_both(rows, cols, n_i, n_j, dims));
        assert!(hook.calls() > 0);
        assert_eq!(hook.taken(), 0, "branch {want}: a sub-floor GEMM threaded");
        assert_bits(&l1, &ln, &format!("left branch {want}, 1 vs 18 threads"));
        assert_bits(&r1, &rn, &format!("right branch {want}, 1 vs 18 threads"));
    }
}
