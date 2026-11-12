// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::MatRef;
use qntt::coeff::CoeffTensor;
use qntt::solvers::InnerProductVector;
use qntt::solvers::davidson::effective_diagonal_bruteforce;
use qntt::su2::core::{Family as Su2Family, Su2Core};
use qntt::su2::coreops::{adjoint_core_times_core, rproject, rproject_plain};
use qntt::su2::expand::expand;
use qntt::su2::frame::Su2Sketch;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hamiltonian::matvec::{
    Su2OneSiteWorkspace, h_sketch_mul, sketch_h_mul, su2_onesite_workspace,
};
use qntt::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use qntt::su2::twosite::{Su2ContractedCores, contract, factor_qc, factor_svd};
use qntt::su2::vector::Su2TtVector;
use qntt::{C64, Scalar};
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn block_diff_and_norm<T: Scalar>(want: MatRef<'_, T>, got: MatRef<'_, T>) -> (f64, f64) {
    assert_eq!(
        (want.nrows(), want.ncols()),
        (got.nrows(), got.ncols()),
        "block shape mismatch"
    );
    let mut diff = 0.0f64;
    let mut refn = 0.0f64;
    for j in 0..want.ncols() {
        for i in 0..want.nrows() {
            let d = *want.get(i, j) - *got.get(i, j);
            diff += d.abs_s() * d.abs_s();
            refn += want.get(i, j).abs_s() * want.get(i, j).abs_s();
        }
    }
    (diff.sqrt(), refn.sqrt())
}

fn dense_norm<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt()
}

fn dense_diff_norm<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x - *y;
            d.abs_s() * d.abs_s()
        })
        .sum::<f64>()
        .sqrt()
}

fn expand_dense<T: Scalar>(x: &Su2TtVector<T>) -> Vec<T> {
    expand(x, x.space.two_s as i64).to_dense()
}

fn ha_tensor<T: Scalar>(d: usize, bonds: &[(usize, usize, f64)]) -> CoeffTensor<T> {
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for &(i, j, val) in bonds {
        t.set(&[i, j], T::from_f64(val));
        t.set(&[j, i], T::from_f64(val));
    }
    t
}

fn zero_v<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 4)
}

fn hb_tensor<T: Scalar>(d: usize, entries: &[(usize, usize, usize, usize, f64)]) -> CoeffTensor<T> {
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for &(i, j, k, l, val) in entries {
        v.set(&[i, j, k, l], T::from_f64(val));
    }
    v
}

fn right_to_left_chain<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Vec<Su2Sketch<T>> {
    let d = x.space.d;
    let mut chain: Vec<Su2Sketch<T>> = vec![Su2Sketch::identity(x.space, d, 1)];
    for k in (0..d).rev() {
        let ws: Su2OneSiteWorkspace = su2_onesite_workspace(h, x.core(k));
        let prev = chain.last().expect("chain non-empty");
        let c = h.sketching_step_left(x, prev, &ws);
        chain.push(c);
    }
    chain.reverse();
    chain
}

fn right_to_left_chain_projected(
    h: &Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    plain: bool,
) -> Vec<Su2Sketch<f64>> {
    let d = x.space.d;
    let mut chain: Vec<Su2Sketch<f64>> = vec![Su2Sketch::identity(x.space, d, 1)];
    for k in (0..d).rev() {
        let xk = x.core(k);
        let prev = chain.last().expect("chain non-empty");
        let y = h_sketch_mul(h, xk, prev);
        let c = if plain {
            rproject_plain(&y, xk)
        } else {
            rproject(&y, xk)
        };
        chain.push(c);
    }
    chain.reverse();
    chain
}

fn right_to_left_chain_plain(
    h: &Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
) -> Vec<Su2Sketch<f64>> {
    right_to_left_chain_projected(h, x, true)
}

fn left_to_right_chain<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Vec<Su2Sketch<T>> {
    let d = x.space.d;
    let mut chain: Vec<Su2Sketch<T>> = vec![Su2Sketch::identity(x.space, 0, 1)];
    for k in 0..d {
        let ws: Su2OneSiteWorkspace = su2_onesite_workspace(h, x.core(k));
        let prev = chain.last().expect("chain non-empty");
        let c = h.sketching_step_right(x, prev, &ws);
        chain.push(c);
    }
    chain
}

fn non_vacuous_theta_count(space: &Su2Space, k: usize, m: Su2Sector) -> usize {
    let mut n = 0usize;
    for f in Su2Family::ALL {
        if let Some(r) = f.apply(m, space.pg.irrep_of(k + 1))
            && space.contains(k + 2, r)
            && (qntt::su2::theta(m.two_s, r.two_s) - 1.0).abs() > 1e-12
        {
            n += 1;
        }
    }
    n
}

mod algebra;
mod anti_oracles;
mod hooks;
mod twosite;
