// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::CoeffTensor;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh};
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::{PointGroup, Scalar};
pub(crate) use qntt_testkit::pg::c2v_table;
use qntt_testkit::{randomized_su2_state, seeded};

pub(crate) fn su2_minimizer(
    h: &Su2SparseHamiltonian<f64>,
    space: Su2Space,
    seed: u64,
    lo: usize,
    hi: usize,
) -> Su2TtVector<f64> {
    let mut rng = seeded(seed);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let (_l, mut x) = riemannian_cg(h, &x0, &opts);
    for _ in 0..8 {
        if su2_grad_norm(h, &x) < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(h, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    assert!(x.check(), "su2_minimizer: fixture failed check()");
    let gnorm = su2_grad_norm(h, &x);
    assert!(
        gnorm < 1e-9,
        "su2 fixture not critical (‖grad‖={gnorm:.3e})"
    );
    x
}

pub(crate) fn su2_grad_norm(h: &Su2SparseHamiltonian<f64>, x: &Su2TtVector<f64>) -> f64 {
    let lam = su2_rayleigh(h, x).re_s();
    let hx = h.apply(x);
    let res = &hx - &(x * lam);
    let (bl, br) = tangent_bases(x);
    Su2TtTangent::project(&bl, &br, &res).norm()
}

pub(crate) fn spin_free_tv_symmetric(
    d: usize,
    pg: PointGroup,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        t.set(&[p, p], 0.15 * (p as f64 - 0.5 * d as f64));
    }
    for p in 0..d {
        for q in (p + 1)..d {
            if pg.irrep_of(p) == pg.irrep_of(q) {
                t.set(&[p, q], -1.0);
                t.set(&[q, p], -1.0);
            }
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    (t, v)
}
