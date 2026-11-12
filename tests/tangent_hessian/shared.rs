// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::Scalar;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt_testkit::gad_mol::su2_grad_exact_at;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn su2_minimizer<T: Scalar>(
    ham: Su2SparseHamiltonian<T>,
    space: Su2Space,
    seed: u64,
    lo: usize,
    hi: usize,
) -> (
    Su2SparseHamiltonian<T>,
    Su2TtVector<T>,
    Su2TtVector<T>,
    Su2TtVector<T>,
) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x0: Su2TtVector<T> = randomized_su2_state(space, lo, hi, &mut rng);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
    let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
    for _ in 0..8 {
        if su2_grad_exact_at(&ham, &x).norm() < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(&ham, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    x.check();
    let gnorm = su2_grad_exact_at(&ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "SU(2) fixture not a Riemannian critical point (‖grad‖={gnorm:.3e})"
    );
    let (base_l, base_r) = tangent_bases(&x);
    (ham, x, base_l, base_r)
}

pub(crate) fn tangent_ortho_basis(
    base_l: &Su2TtVector<f64>,
    base_r: &Su2TtVector<f64>,
    space: Su2Space,
    seed: u64,
    max_dirs: usize,
) -> Vec<Su2TtTangent<f64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut basis: Vec<Su2TtTangent<f64>> = Vec::new();
    for _ in 0..max_dirs {
        let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let mut u = Su2TtTangent::project(base_l, base_r, &dir);
        for b in &basis {
            let c = b.dot(&u);
            u = u.sub(&b.scale(c));
        }
        let nrm = u.norm();
        if nrm > 1e-8 {
            u.scale_mut(1.0 / nrm);
            basis.push(u);
        }
    }
    basis
}

pub(crate) fn dedup_sorted(vals: &[f64], rel_tol: f64) -> Vec<f64> {
    let scale = vals.last().copied().unwrap_or(1.0).abs().max(1.0);
    let mut out: Vec<f64> = Vec::new();
    for &v in vals {
        if out.last().is_none_or(|&l| v - l > rel_tol * scale) {
            out.push(v);
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn sym_eigvals_jacobi(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _sweep in 0..200 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-15 {
                    continue;
                }
                let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut v: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v
}
