// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::qn::QnSpace;
use qntt::solvers::{InnerProductVector, TtHamiltonian, TtState};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hamiltonian::matvec::su2_onesite_workspace;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::als::{AlsOptions, als, boundary_idsketch, right_to_left_sketching};
use qntt::su2::solvers::mals::{MalsOptions, mals};
use qntt::su2::solvers::tdvp::{RealtimeOptions, TdvpOptions, tdvp, tdvp_realtime};
use qntt::su2::vector::Su2TtVector;
use qntt::vector::TtVector;
use qntt::{C64, Scalar};
use qntt_testkit::randomized_state;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
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

fn zero_t<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 2)
}

fn hb_tensor<T: Scalar>(d: usize, entries: &[(usize, usize, usize, usize, f64)]) -> CoeffTensor<T> {
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for &(i, j, k, l, val) in entries {
        v.set(&[i, j, k, l], T::from_f64(val));
    }
    v
}

fn ha_su2(space: Su2Space, i: usize, j: usize) -> Su2SparseHamiltonian<f64> {
    let t = ha_tensor::<f64>(space.d, &[(i, j, -1.0)]);
    Su2SparseHamiltonian::new(space, &t, &zero_v(space.d), None, 1e-12)
}

fn ha_abelian(space: QnSpace, i: usize, j: usize) -> SparseHamiltonian<f64> {
    let t = ha_tensor::<f64>(space.d, &[(i, j, -1.0)]);
    SparseHamiltonian::new(space, &t, &zero_v(space.d), None, 1e-12)
}

fn ha_su2_c64(space: Su2Space, i: usize, j: usize, phase: f64) -> Su2SparseHamiltonian<C64> {
    let mut t = CoeffTensor::<C64>::zeros(space.d, 2);
    t.set(&[i, j], -C64::new(phase.cos(), phase.sin()));
    t.set(&[j, i], -C64::new(phase.cos(), -phase.sin()));
    Su2SparseHamiltonian::new(space, &t, &zero_v(space.d), None, 1e-12)
}

fn hb_su2(space: Su2Space) -> Su2SparseHamiltonian<f64> {
    let v = hb_tensor::<f64>(space.d, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)]);
    Su2SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
}

fn hb_abelian(space: QnSpace) -> SparseHamiltonian<f64> {
    let v = hb_tensor::<f64>(space.d, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)]);
    SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
}

fn hb_polarized_su2(
    space: Su2Space,
    a: usize,
    b: usize,
    c: usize,
    e: usize,
) -> Su2SparseHamiltonian<f64> {
    let v = hb_tensor::<f64>(space.d, &[(b, a, c, e, -1.0), (a, b, e, c, -1.0)]);
    Su2SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
}

fn global_eigen_residual(h: &Su2SparseHamiltonian<f64>, x: &Su2TtVector<f64>, lambda: f64) -> f64 {
    let hx = TtHamiltonian::matvec(h, x);
    let r = TtState::sub(&hx, &TtState::scaled(x, lambda));
    TtState::norm(&r) / TtState::norm(x).max(1e-30)
}

fn global_eigen_residual_abelian(
    h: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    lambda: f64,
) -> f64 {
    let hx = h * x;
    let r = &hx - &(x * lambda);
    r.norm_lr(false) / x.norm_lr(false).max(1e-30)
}

fn max_rank_of<T: Scalar>(x: &Su2TtVector<T>) -> usize {
    x.ranks
        .iter()
        .flat_map(|g| g.iter().map(|(_, &r)| r))
        .max()
        .unwrap()
}

fn count_theta_cw<T: Scalar>(x: &Su2TtVector<T>) -> (usize, usize) {
    let space = x.space;
    let d = space.d;
    let mut n_theta = 0usize;
    let mut n_cw = 0usize;
    for b in 0..d {
        for m in space.bond_sectors(b) {
            if x.rank(b, m) == 0 {
                continue;
            }
            for succ in space.bond_sectors(b + 1) {
                if x.rank(b + 1, succ) == 0 {
                    continue;
                }
                if (qntt::su2::theta(m.two_s, succ.two_s) - 1.0).abs() > 1e-12 {
                    n_theta += 1;
                }
            }
        }
    }
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            if x.rank(b, s) == 0 {
                continue;
            }
            if (qntt::su2::centerweight(s.two_s, space.two_s) - 1.0).abs() > 1e-12 {
                n_cw += 1;
            }
        }
    }
    (n_theta, n_cw)
}

mod als;
mod cross_symmetry;
mod mals;
mod tdvp;
