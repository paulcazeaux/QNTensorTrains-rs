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
use qntt::round::RoundMode;
use qntt::solvers::{TtHamiltonian, TtState};
use qntt::su2::core::{Family, Su2Core};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;
use qntt::vector::TtVector;
use qntt::{C64, Scalar};
pub(crate) use qntt_testkit::randomized_state;
pub(crate) use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub(crate) fn ha_tensor<T: Scalar>(d: usize, bonds: &[(usize, usize, f64)]) -> CoeffTensor<T> {
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for &(i, j, val) in bonds {
        t.set(&[i, j], T::from_f64(val));
        t.set(&[j, i], T::from_f64(val));
    }
    t
}

pub(crate) fn zero_v<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 4)
}

pub(crate) fn zero_t<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 2)
}

pub(crate) fn hb_tensor<T: Scalar>(
    d: usize,
    entries: &[(usize, usize, usize, usize, f64)],
) -> CoeffTensor<T> {
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for &(i, j, k, l, val) in entries {
        v.set(&[i, j, k, l], T::from_f64(val));
    }
    v
}

pub(crate) fn ha_su2(space: Su2Space, i: usize, j: usize) -> Su2SparseHamiltonian<f64> {
    let t = ha_tensor::<f64>(space.d, &[(i, j, -1.0)]);
    Su2SparseHamiltonian::new(space, &t, &zero_v(space.d), None, 1e-12)
}

pub(crate) fn ha_abelian(space: QnSpace, i: usize, j: usize) -> SparseHamiltonian<f64> {
    let t = ha_tensor::<f64>(space.d, &[(i, j, -1.0)]);
    SparseHamiltonian::new(space, &t, &zero_v(space.d), None, 1e-12)
}

pub(crate) fn hb_v_tensor<T: Scalar>(d: usize) -> CoeffTensor<T> {
    hb_tensor::<T>(d, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)])
}

pub(crate) fn two_body_su2(
    space: Su2Space,
    a: usize,
    b: usize,
    c: usize,
    e: usize,
) -> Su2SparseHamiltonian<f64> {
    let v = hb_tensor::<f64>(space.d, &[(c, a, e, b, -1.0), (b, e, a, c, -1.0)]);
    Su2SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
}

pub(crate) fn global_eigen_residual(
    h: &Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    lambda: f64,
) -> f64 {
    let hx = TtHamiltonian::matvec(h, x);
    let r = TtState::sub(&hx, &TtState::scaled(x, lambda));
    TtState::norm(&r) / TtState::norm(x).max(1e-30)
}

pub(crate) fn global_eigen_residual_abelian(
    h: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    lambda: f64,
) -> f64 {
    let hx = h * x;
    let r = &hx - &(x * lambda);
    r.norm_lr(false) / x.norm_lr(false).max(1e-30)
}

pub(crate) fn count_theta_cw<T: Scalar>(x: &Su2TtVector<T>) -> (usize, usize) {
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

pub(crate) fn phase_gauge(
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    theta: &[f64],
) -> (CoeffTensor<C64>, CoeffTensor<C64>) {
    let d = theta.len();
    let cis = |th: f64| C64::new(th.cos(), th.sin());
    let mut t2 = CoeffTensor::<C64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            let tv = t.get(&[i, j]);
            if tv == 0.0 {
                continue;
            }
            let val = C64::from_f64(tv) * cis(theta[i]).conj_s() * cis(theta[j]);
            t2.set(&[i, j], val);
        }
    }
    let mut v2 = CoeffTensor::<C64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let vv = v.get(&[i, j, k, l]);
                    if vv == 0.0 {
                        continue;
                    }
                    let val = C64::from_f64(vv)
                        * cis(theta[i]).conj_s()
                        * cis(theta[k]).conj_s()
                        * cis(theta[l])
                        * cis(theta[j]);
                    v2.set(&[i, j, k, l], val);
                }
            }
        }
    }
    (t2, v2)
}

pub(crate) fn dense_rayleigh(h_ab: &SparseHamiltonian<f64>, e: &TtVector<f64>) -> f64 {
    let hy = (h_ab * e).to_dense();
    let ee = e.to_dense();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..ee.len() {
        num += ee[i] * hy[i];
        den += ee[i] * ee[i];
    }
    num / den
}

#[allow(clippy::type_complexity)]
pub(crate) fn build_tangent_pieces(
    space: Su2Space,
    seed: u64,
) -> (
    Su2TtVector<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
) {
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let nx = x.norm_lr(false);
    x.scale_mut(1.0 / nx);
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    let mut dir1: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    dir1.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let n1 = dir1.norm_lr(false);
    dir1.scale_mut(1.0 / n1);
    let mut dir2: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    dir2.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let n2 = dir2.norm_lr(false);
    dir2.scale_mut(1.0 / n2);
    (x, base_l, base_r, dir1, dir2)
}

#[allow(clippy::type_complexity)]
pub(crate) fn tangent_setup(
    seed: u64,
) -> (
    Su2Space,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
) {
    let space = Su2Space::new(6, 2, 2);
    let (x, base_l, base_r, dir1, dir2) = build_tangent_pieces(space, seed);
    (space, x, base_l, base_r, dir1, dir2)
}

pub(crate) fn flat_component_norm<T: Scalar>(comp: &Su2Core<T>) -> f64 {
    let mut acc = 0.0f64;
    for f in Family::ALL {
        for l in comp.transitions(f).collect::<Vec<_>>() {
            let blk = comp.block(f, l).expect("transition present");
            for j in 0..blk.ncols() {
                for i in 0..blk.nrows() {
                    let v = *blk.get(i, j);
                    acc += v.abs_s() * v.abs_s();
                }
            }
        }
    }
    acc
}

pub(crate) fn gauge_residual_norm(l: faer::MatRef<'_, f64>, c: faer::MatRef<'_, f64>) -> f64 {
    let mut n2 = 0.0f64;
    for j in 0..c.ncols() {
        for i in 0..l.ncols() {
            let mut acc = 0.0f64;
            for kk in 0..l.nrows() {
                acc += *l.get(kk, i) * *c.get(kk, j);
            }
            n2 += acc * acc;
        }
    }
    n2.sqrt()
}
