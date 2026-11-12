// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg3::*;

#[test]
fn pg3_dot_norm_axpby_against_dense() {
    let (pg, space) = pg3_space_d6();
    let x: TtVector<f64> = capped_randn(space, 1, 3, 303_040);
    let y: TtVector<f64> = capped_randn(space, 1, 3, 303_041);
    let (alpha, beta) = (0.37, -1.8);

    let want = dense_dot(&x.to_dense(), &y.to_dense());
    assert!(want.abs() > 0.0, "vacuous");
    let got = dot(&x, &y);
    assert!((got - want).abs() <= 1e-8 * want.abs());

    let n_dense = dense_norm(&x.to_dense());
    assert!((x.norm() - n_dense).abs() <= 1e-14 * n_dense);

    let dxx = dot(&x, &x);
    assert!((x.norm() * x.norm() - dxx).abs() <= 1e-12 * dxx.abs().max(1.0));

    let ax = alpha * &x;
    let by = beta * &y;
    let lhs = &ax + &by;
    let got_lin = dot(&lhs, &y);
    let want_lin = alpha * dot(&x, &y) + beta * dot(&y, &y);
    assert!(
        (got_lin - want_lin).abs() <= 1e-10 * want_lin.abs().max(1.0),
        "left linearity"
    );

    let az = alpha * &y;
    let bxv = beta * &x;
    let rhs = &az + &bxv;
    let got_lin2 = dot(&x, &rhs);
    let want_lin2 = alpha * dot(&x, &y) + beta * dot(&x, &x);
    assert!(
        (got_lin2 - want_lin2).abs() <= 1e-10 * want_lin2.abs().max(1.0),
        "right linearity"
    );

    let z = &(alpha * &x) + &(beta * &y);
    assert!(z.check());
    let (xd, yd) = (x.to_dense(), y.to_dense());
    let want_z: Vec<f64> = xd
        .iter()
        .zip(&yd)
        .map(|(&a, &b)| alpha * a + beta * b)
        .collect();
    assert!(rel_err(&z.to_dense(), &want_z) < 50.0 * 8.0 * f64::EPSILON);
    assert_pure(&pg, &space, &z);

    let w = &(alpha * &x) - &(beta * &y);
    assert!(w.check());
    let want_w: Vec<f64> = xd
        .iter()
        .zip(&yd)
        .map(|(&a, &b)| alpha * a - beta * b)
        .collect();
    assert!(rel_err(&w.to_dense(), &want_w) < 50.0 * 8.0 * f64::EPSILON);
    assert_pure(&pg, &space, &w);

    let c = std::f64::consts::PI;
    let left = c * &x;
    let right = &x * c;
    let want_scaled: Vec<f64> = xd.iter().map(|&a| c * a).collect();
    let bound = 50.0 * 8.0 * f64::EPSILON;
    assert!(rel_err(&left.to_dense(), &want_scaled) < bound, "c*x");
    assert!(rel_err(&right.to_dense(), &want_scaled) < bound, "x*c");
    assert_pure(&pg, &space, &left);
    assert_pure(&pg, &space, &right);
    let back = &left / c;
    assert!(rel_err(&back.to_dense(), &xd) < bound, "(c*x)/c");

    let h = hadamard(&x, &y);
    assert!(h.check());
    let want_h: Vec<f64> = xd.iter().zip(&yd).map(|(&a, &b)| a * b).collect();
    assert!(rel_err(&h.to_dense(), &want_h) < bound);
    assert_pure(&pg, &space, &h);

    let zero = &x - &x;
    assert!(zero.norm() < 1e-14);
}

#[test]
fn pg3_dot_of_determinants_is_kronecker_delta() {
    let (pg, space) = pg3_space_d6();
    let dets = determinants_with_irrep(&pg, space.d, space.nup, space.ndn, space.target);
    assert!(
        dets.len() >= 3,
        "vacuous: need >= 3 distinct determinants of the fixed target irrep"
    );

    let states: Vec<TtVector<f64>> = dets[..3]
        .iter()
        .map(|(occ_up, occ_dn)| tt_state_in::<f64>(space, occ_up, occ_dn))
        .collect();
    for s in &states {
        assert!(s.check());
        assert_eq!(s.space, space);
        assert_pure(&pg, &space, s);
    }
    assert_ne!(states[0].to_dense(), states[1].to_dense());
    assert_ne!(states[0].to_dense(), states[2].to_dense());

    for s in &states {
        assert!((dot(s, s) - 1.0).abs() < 1e-12);
    }
    assert!(dot(&states[0], &states[1]).abs() < 1e-12);
    assert!(dot(&states[0], &states[2]).abs() < 1e-12);
    assert!(dot(&states[1], &states[2]).abs() < 1e-12);
    assert!((dot(&states[0], &states[1]) - dot(&states[1], &states[0])).abs() < 1e-12);
}

#[test]
fn pg3_add_scalar_is_target_indicator_offset() {
    let (pg, space) = pg3_space_d6();
    let x: TtVector<f64> = capped_randn(space, 1, 2, 303_050);
    let ones: TtVector<f64> = qntt::tt_ones(space);
    let xd = x.to_dense();
    let onesd = ones.to_dense();
    let c = 0.7;

    let y = x.add_scalar(c);
    assert!(y.check());
    let want: Vec<f64> = xd.iter().zip(&onesd).map(|(&a, &o)| a + c * o).collect();
    let yd = y.to_dense();
    for (idx, (&g, &w)) in yd.iter().zip(&want).enumerate() {
        assert!((g - w).abs() < 1e-12, "config {idx}: add_scalar linearity");
    }
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);

    let z = (-&x).add_scalar(c);
    let want_m: Vec<f64> = xd.iter().zip(&onesd).map(|(&a, &o)| c * o - a).collect();
    let zd = z.to_dense();
    for (idx, (&g, &w)) in zd.iter().zip(&want_m).enumerate() {
        assert!((g - w).abs() < 1e-12, "config {idx}: mirror form");
    }
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &zd);
}
