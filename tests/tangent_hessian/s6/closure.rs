// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::C64;
use qntt::Scalar;
use qntt::coeff::CoeffTensor;
use qntt::solvers::{TtHamiltonian, TtState};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::als::{AlsOptions, als};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_zeros as su2_tt_zeros};

use crate::s6::util::{
    count_theta_cw, flat_component_norm, phase_gauge, randomized_su2_state, seeded, tangent_setup,
    zero_v,
};

#[test]
fn s6_e1_unweighted_metric_anti_oracle() {
    let (space, _x, base_l, base_r, dir1, _dir2) = tangent_setup(0x6e10);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
    let n_cw = dx.norm();

    let mut n_flat_sq = 0.0f64;
    for k in 0..space.d {
        n_flat_sq += flat_component_norm(dx.component(k));
    }
    let n_flat = n_flat_sq.sqrt();

    let (n_theta, n_cw_count) = count_theta_cw(&dx.to_vector());
    assert!(
        n_theta >= 2 && n_cw_count >= 2,
        "vacuous: theta/centerweight trivial (n_theta={n_theta}, n_cw={n_cw_count})"
    );

    assert!(
        (n_cw - n_flat).abs() > 1e-3,
        "anti-oracle did not bite: centerweighted norm {n_cw} == flat norm {n_flat}"
    );
}

#[test]
fn s6_g1_complex_hermitian_als_phase_gauge() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[0, 1], -1.0);
    t.set(&[1, 0], -1.0);
    let v = zero_v::<f64>(d);
    let theta: Vec<f64> = (0..d).map(|k| 0.3 * (k as f64 + 1.0)).collect();
    let (t2, v2) = phase_gauge(&t, &v, &theta);
    let h = Su2SparseHamiltonian::new(space, &t2, &v2, None, 1e-12);

    #[allow(unused_assignments)]
    let mut lambda = f64::INFINITY;
    let mut xa = su2_tt_zeros::<C64>(space);
    for trial in 0..4u64 {
        let mut rng = seeded(0x6910 + trial);
        let x0: Su2TtVector<C64> = randomized_su2_state(space, 2, 4, &mut rng);
        let (l, xt) = als(
            &h,
            &x0,
            &AlsOptions {
                eps: 1e-6,
                ..Default::default()
            },
        );
        lambda = l.re_s();
        xa = xt;
        if (lambda + 2.0).abs() < 1e-4 {
            break;
        }
    }
    assert!(xa.check());
    let rq = qntt::su2::hamiltonian::rayleigh_quotient(&h, &xa);
    assert!(
        (rq.re_s() - (-2.0)).abs() <= 1e-4,
        "als phase-gauge Re(RQ) {}",
        rq.re_s()
    );
    assert!(
        rq.im_s().abs() <= 1e-6,
        "als phase-gauge Im(RQ) {}",
        rq.im_s()
    );

    let hx = TtHamiltonian::matvec(&h, &xa);
    let r = TtState::sub(&hx, &TtState::scaled(&xa, C64::from_f64(rq.re_s())));
    let resid = TtState::norm(&r) / TtState::norm(&xa).max(1e-30);
    assert!(resid <= 1e-4, "als phase-gauge residual {resid}");
}
