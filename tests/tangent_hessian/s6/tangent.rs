// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::Scalar;
use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::qn::QnSpace;
use qntt::round::RoundMode;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::tangent_bases;
use qntt::su2::tangent::{Su2TtTangent, projective_transport};
use qntt::su2::vector::Su2TtVector;
use qntt::su2::{expand, ops::dot as su2_dot};

use crate::s6::util::{
    build_tangent_pieces, count_theta_cw, dense_rayleigh, gauge_residual_norm, hb_v_tensor,
    randomized_su2_state, seeded, tangent_setup, zero_t, zero_v,
};

#[test]
fn s6_a1_metric_lock() {
    for (idx, &(d, n, two_s)) in [(6usize, 2usize, 2usize), (5, 3, 1)].iter().enumerate() {
        let space = Su2Space::new(d, n, two_s);
        let (_x, base_l, base_r, dir1, dir2) = build_tangent_pieces(space, 0x6a10 + idx as u64);
        let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
        let dy = Su2TtTangent::project(&base_l, &base_r, &dir2);
        let v = dx.to_vector();
        let dxn = dx.norm();

        assert!(
            (dxn - v.norm_lr(false)).abs() <= 1e-12 * dxn.max(1.0),
            "metric lock (tangent norm vs embedded norm_lr) at {space:?}: {dxn} vs {}",
            v.norm_lr(false)
        );

        let mut tsz = -(two_s as i64);
        while tsz <= two_s as i64 {
            let ev = expand(&v, tsz).norm();
            assert!(
                (dxn - ev).abs() <= 1e-12 * dxn.max(1.0),
                "metric lock (norm-through-expand) at {space:?}, tsz={tsz}: {dxn} vs {ev}"
            );
            tsz += 2;
        }

        let (n_theta, n_cw) = count_theta_cw(&v);
        assert!(
            n_theta >= 2 && n_cw >= 2,
            "vacuous: theta/centerweight trivial at {space:?} (n_theta={n_theta}, n_cw={n_cw})"
        );

        assert!(
            (dx.dot(&dy) - su2_dot(&dx.to_vector(), &dy.to_vector())).abs() < 1e-12,
            "metric lock (tangent dot vs embedded dot) at {space:?}"
        );
        assert!(
            (dx.dot(&dx) - dxn * dxn).abs() < 1e-12 * dxn.max(1.0).powi(2),
            "dot(dx,dx) == norm(dx)^2 at {space:?}"
        );
    }
}

#[test]
fn s6_a2_gauge_condition() {
    let (space, _x, base_l, base_r, dir1, _dir2) = tangent_setup(0x6a20);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
    for k in 0..space.d - 1 {
        for s in space.bond_sectors(k + 1) {
            let l = base_l.core(k).vertical_unfolding(s);
            let c = dx.component(k).vertical_unfolding(s);
            let n = gauge_residual_norm(l.as_ref(), c.as_ref());
            assert!(
                n < 1e-12,
                "gauge condition at bond {}, sector {s:?}: {n}",
                k + 1
            );
        }
    }
}

#[test]
fn s6_a3_projection_linearity() {
    let (_space, _x, base_l, base_r, dir1, dir2) = tangent_setup(0x6a30);
    let combo = &(&dir1 * 0.7) - &(&dir2 * 1.3);
    let dxy = Su2TtTangent::project(&base_l, &base_r, &combo);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
    let dy = Su2TtTangent::project(&base_l, &base_r, &dir2);
    let want = dx.scale(0.7).sub(&dy.scale(1.3));
    let diff = (&dxy.to_vector() - &want.to_vector()).norm_lr(false);
    assert!(diff < 1e-12, "projection linearity: {diff}");
}

#[test]
fn s6_a4_projection_idempotence() {
    let (_space, _x, base_l, base_r, dir1, _dir2) = tangent_setup(0x6a40);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
    let dx2 = Su2TtTangent::project(&base_l, &base_r, &dx.to_vector());
    let diff = (&dx.to_vector() - &dx2.to_vector()).norm_lr(false);
    assert!(diff < 1e-12, "projection idempotence: {diff}");
}

#[test]
fn s6_a5_vector_space_embedding() {
    let (_space, _x, base_l, base_r, dir1, dir2) = tangent_setup(0x6a50);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
    let dy = Su2TtTangent::project(&base_l, &base_r, &dir2);

    let sum_v = &dx.to_vector() + &dy.to_vector();
    let v_sum = dx.add(&dy).to_vector();
    assert!((&sum_v - &v_sum).norm_lr(false) < 1e-12, "to_vector(dx+dy)");

    let diff_v = &dx.to_vector() - &dy.to_vector();
    let v_diff = dx.sub(&dy).to_vector();
    assert!(
        (&diff_v - &v_diff).norm_lr(false) < 1e-12,
        "to_vector(dx-dy)"
    );

    let scaled = &dx.to_vector() * 2.5;
    let v_scaled = dx.scale(2.5).to_vector();
    assert!(
        (&scaled - &v_scaled).norm_lr(false) < 1e-12,
        "to_vector(2.5*dx)"
    );

    let mut dz = dx.clone();
    dz.axpy(0.3, &dy);
    let want_axpy = &dx.to_vector() + &(&dy.to_vector() * 0.3);
    assert!(
        (&dz.to_vector() - &want_axpy).norm_lr(false) < 1e-12,
        "axpy!(0.3, dy, dz)"
    );

    let mut dw = dx.clone();
    dw.axpby(0.3, &dy, -1.7);
    let want_axpby = &(&dy.to_vector() * 0.3) + &(&dx.to_vector() * (-1.7));
    assert!(
        (&dw.to_vector() - &want_axpby).norm_lr(false) < 1e-12,
        "axpby!(0.3, dy, -1.7, dw)"
    );

    let mut dv = dx.clone();
    dv.scale_mut(-0.6);
    let want_scale_mut = &dx.to_vector() * (-0.6);
    assert!(
        (&dv.to_vector() - &want_scale_mut).norm_lr(false) < 1e-12,
        "scale_mut(-0.6, dx)"
    );
}

#[test]
fn s6_a6_retraction() {
    let (_space, x, base_l, base_r, dir1, _dir2) = tangent_setup(0x6a60);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);

    let r0 = dx.retract_hosvd(0.0);
    assert!((&r0 - &x).norm_lr(false) < 1e-12, "retract at alpha=0 == x");

    let alpha = 1e-3;
    let lin_at = |a: f64| -> f64 {
        let ra = dx.retract_hosvd(a);
        let linear = &x + &(&dx.to_vector() * a);
        (&ra - &linear).norm_lr(false)
    };
    let e1 = lin_at(alpha);
    let e2 = lin_at(2.0 * alpha);
    assert!(
        e1 < 10.0 * alpha * alpha,
        "first-order retraction error e1={e1}"
    );
    assert!(
        (e2 / e1 - 4.0).abs() < 1.0,
        "retraction ratio e2/e1 = {}",
        e2 / e1
    );
}

#[test]
fn s6_a7_first_order_fd() {
    let (x, base_l, base_r, dir1, _dir2) = build_tangent_pieces(Su2Space::new(8, 4, 2), 0x6a70);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
    let dxv = dx.to_vector();

    let fd_at = |eps: f64| -> Su2TtVector<f64> {
        let mut p = &x + &(&dir1 * eps);
        p.round_ranks_mut(&x.ranks);
        &(&p - &x) * (1.0 / eps)
    };
    let e1 = (&dxv - &fd_at(1e-6)).norm_lr(false);
    let e2 = (&dxv - &fd_at(2e-6)).norm_lr(false);
    assert!((e2 / e1 - 2.0).abs() < 0.1, "FD ratio e2/e1 = {}", e2 / e1);
}

#[test]
fn s6_a8_transport() {
    let (space, x, base_l, base_r, dir1, _dir2) = tangent_setup(0x6a80);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir1);
    let self_transport = projective_transport(&x, &dx);
    let diff = (&self_transport.to_vector() - &dx.to_vector()).norm_lr(false);
    assert!(diff < 1e-12, "self-transport is identity: {diff}");

    let mut rng_y = seeded(0x6a81);
    let mut y: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng_y);
    y.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let ny = y.norm_lr(false);
    y.scale_mut(1.0 / ny);
    let mut y_base_l = y.clone();
    y_base_l.left_orthogonalize(true);
    let transported = projective_transport(&y, &dx);
    for k in 0..space.d - 1 {
        for s in space.bond_sectors(k + 1) {
            let l = y_base_l.core(k).vertical_unfolding(s);
            let c = transported.component(k).vertical_unfolding(s);
            let n = gauge_residual_norm(l.as_ref(), c.as_ref());
            assert!(
                n < 1e-12,
                "transported gauge condition at bond {}: {n}",
                k + 1
            );
        }
    }
}

#[test]
fn s6_b1_metric_fd_through_expand() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[0, 1], -0.8);
    t.set(&[1, 0], -0.8);
    t.set(&[2, 3], -0.5);
    t.set(&[3, 2], -0.5);
    t.set(&[0, 0], 0.3);
    t.set(&[2, 2], -0.4);
    let v = zero_v::<f64>(d);
    let h_su2 = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);

    let mut rng = seeded(0x6b10);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let nx = x.norm_lr(false);
    x.scale_mut(1.0 / nx);
    let (base_l, base_r) = tangent_bases(&x);

    let mut dir: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    dir.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let nd = dir.norm_lr(false);
    dir.scale_mut(1.0 / nd);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir);

    let lambda = qntt::su2::hamiltonian::rayleigh_quotient(&h_su2, &x).re_s();
    let hx = h_su2.apply(&x);
    let resid = &hx - &(&x * lambda);
    let g = Su2TtTangent::project(&base_l, &base_r, &resid);

    let slope_ana = 2.0 * g.dot(&dx);

    let t_sz = space.two_s as i64;
    let ab_space = QnSpace::from_n_two_sz(d, space.n, t_sz);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-12);
    let energy = |y: &Su2TtVector<f64>| -> f64 { dense_rayleigh(&h_ab, &expand(y, t_sz)) };

    let eps = 1e-5;
    let y_plus = dx.retract_hosvd(eps);
    let y_minus = dx.retract_hosvd(-eps);
    let slope_fd = (energy(&y_plus) - energy(&y_minus)) / (2.0 * eps);

    assert!(
        slope_ana.abs() > 0.1,
        "non-vacuity: slope_ana = {slope_ana}"
    );
    let (n_theta, n_cw) = count_theta_cw(&dx.to_vector());
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight trivial (n_theta={n_theta}, n_cw={n_cw})"
    );

    assert!(
        (slope_fd - slope_ana).abs() < 1e-4 * slope_ana.abs().max(1.0),
        "B1 metric linchpin: slope_fd={slope_fd} vs slope_ana={slope_ana}"
    );
}

#[test]
fn s6_b2_gradient_fd_relation() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let t = zero_t::<f64>(d);
    let v = hb_v_tensor::<f64>(d);
    let h_su2 = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);

    let t_sz = space.two_s as i64;
    let ab_space = QnSpace::from_n_two_sz(d, space.n, t_sz);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-12);
    let energy = |y: &Su2TtVector<f64>| -> f64 { dense_rayleigh(&h_ab, &expand(y, t_sz)) };

    let mut rng = seeded(0x6b23);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let nx = x.norm_lr(false);
    x.scale_mut(1.0 / nx);
    let (base_l, base_r) = tangent_bases(&x);

    let mut dir: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    dir.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let nd = dir.norm_lr(false);
    dir.scale_mut(1.0 / nd);
    let dx = Su2TtTangent::project(&base_l, &base_r, &dir);

    let lambda = qntt::su2::hamiltonian::rayleigh_quotient(&h_su2, &x).re_s();
    let hx = h_su2.apply(&x);
    let resid = &hx - &(&x * lambda);
    let g = Su2TtTangent::project(&base_l, &base_r, &resid);
    let slope_ana = 2.0 * g.dot(&dx);

    let eps = 1e-5;
    let slope_fd = (energy(&dx.retract_hosvd(eps)) - energy(&dx.retract_hosvd(-eps))) / (2.0 * eps);

    assert!(
        slope_ana.abs() > 0.1,
        "non-vacuity: slope_ana = {slope_ana}"
    );
    let (n_theta, n_cw) = count_theta_cw(&dx.to_vector());
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight trivial (n_theta={n_theta}, n_cw={n_cw})"
    );
    assert!(
        (slope_fd - slope_ana).abs() < 1e-4 * slope_ana.abs().max(1.0),
        "B2 two-body metric cross-check: slope_fd={slope_fd} vs slope_ana={slope_ana}"
    );
}
