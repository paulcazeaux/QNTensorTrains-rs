// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase4::*;

#[test]
fn t1_elem_1body_i_lt_j() {
    let d = 6;
    let a: TtVector<f64> = state_from_sites(d, &[1], &[3]);
    let b: TtVector<f64> = state_from_sites(d, &[1], &[4]);
    let x0 = &a + &b;
    let i = Orbital::new(0, Spin::Up);
    let j = Orbital::new(1, Spin::Up);
    let x = adag_i_a_j(&x0, i, j, 1.0);
    let ra: TtVector<f64> = state_from_sites(d, &[0], &[3]);
    let rb: TtVector<f64> = state_from_sites(d, &[0], &[4]);
    let refr = &ra + &rb;
    let diff = &x - &refr;
    assert!(
        diff.norm() < 1e-12,
        "AdagᵢAⱼ up-hop 2→1 mismatch: ‖x-ref‖={}",
        diff.norm()
    );
}

#[test]
fn t1_elem_1body_i_gt_j() {
    let d = 6;
    let a: TtVector<f64> = state_from_sites(d, &[5], &[3]);
    let b: TtVector<f64> = state_from_sites(d, &[5], &[4]);
    let c: TtVector<f64> = state_from_sites(d, &[2], &[1]);
    let x0 = &(&a + &b) + &c;
    let i = Orbital::new(4, Spin::Dn);
    let j = Orbital::new(3, Spin::Dn);
    let x = adag_i_a_j(&x0, i, j, 1.0);
    let refr: TtVector<f64> = state_from_sites(d, &[5], &[4]);
    let diff = &x - &refr;
    assert!(diff.norm() < 1e-12, "AdagᵢAⱼ dn-hop 4→5 (i>j) mismatch");
}

#[test]
fn t1_elem_density_lr() {
    let d = 5;
    let a: TtVector<f64> = state_from_sites(d, &[0], &[3]);
    let b: TtVector<f64> = state_from_sites(d, &[0], &[4]);
    let c: TtVector<f64> = state_from_sites(d, &[2], &[3]);
    let x0 = &(&a + &b) + &c;
    let inner = adag_i_a_j(
        &x0,
        Orbital::new(1, Spin::Up),
        Orbital::new(0, Spin::Up),
        1.0,
    );
    let x = adag_i_a_j(
        &inner,
        Orbital::new(3, Spin::Dn),
        Orbital::new(3, Spin::Dn),
        1.0,
    );
    let refr: TtVector<f64> = state_from_sites(d, &[1], &[3]);
    let diff = &x - &refr;
    assert!(
        diff.norm_lr(false) < 1e-12,
        "composed hop + N(Dn) density mismatch"
    );
}

#[test]
fn t1_elem_1body_dense_oracle() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE11);
    let x0: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let i = Orbital::new(1, Spin::Up);
    let j = Orbital::new(2, Spin::Up);
    let coeff = 0.7_f64;
    let x = adag_i_a_j(&x0, i, j, coeff);

    let x0d = x0.to_dense();
    let refr: Vec<f64> = dense_apply_string(d, &[(i, true), (j, false)], &x0d)
        .into_iter()
        .map(|v| v * coeff)
        .collect();
    let got = x.to_dense();
    let tol = 50.0 * d as f64 * f64::EPSILON;
    assert!(
        rel_err(&got, &refr) < tol,
        "1-body dense oracle mismatch: rel_err={}",
        rel_err(&got, &refr)
    );
}
