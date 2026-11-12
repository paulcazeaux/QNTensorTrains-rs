// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::coreops::{rproject, rproject_plain};
use qntt::su2::expand::expand;
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;

use qntt::{C64, RoundMode};

use crate::s2::util::{
    assert_dense_close, close_rel_abs, dense_diff_norm, dense_norm, gram_row_dev,
    pick_open_shell_core, randomized_su2_state, rproject_plain_transpose, seeded, sketch_frob_diff,
    su2_left_canonicity_dev, su2_right_canonicity_dev,
};

#[test]
fn o1a_rproject_anti_oracle_plain_adjoint_must_fail() {
    let mut rng = seeded(20_401);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let k = pick_open_shell_core(&x);
    let a = x.core(k).clone();

    let (q, c_true, _row_ranks) = a.cq();

    let c_rproj = rproject(&a, &q);
    let dev = sketch_frob_diff(&c_rproj, &c_true);
    assert!(
        dev < 1e-11,
        "rproject failed to reconstruct C_true: dev={dev}"
    );

    let c_plain = rproject_plain(&a, &q);
    let dev_plain = sketch_frob_diff(&c_plain, &c_true);
    assert!(
        dev_plain > 1e-3,
        "plain-adjoint anti-oracle did NOT fail (dev={dev_plain}) — the anti-oracle is vacuous"
    );
}

#[test]
fn o1b_rproject_anti_oracle_plain_adjoint_must_fail_second_size() {
    let mut rng = seeded(20_402);
    let space = Su2Space::new(5, 3, 1);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let k = pick_open_shell_core(&x);
    let a = x.core(k).clone();

    let (q, c_true, _row_ranks) = a.cq();
    let c_rproj = rproject(&a, &q);
    assert!(sketch_frob_diff(&c_rproj, &c_true) < 1e-11);

    let c_plain = rproject_plain(&a, &q);
    assert!(sketch_frob_diff(&c_plain, &c_true) > 1e-3);
}

#[test]
fn o1_complex_rproject_anti_oracle_and_plain_transpose_also_fails() {
    let mut rng = seeded(20_403);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
    let k = pick_open_shell_core(&x);
    let a = x.core(k).clone();

    let (q, c_true, _row_ranks) = a.cq();

    let c_rproj = rproject(&a, &q);
    assert!(sketch_frob_diff(&c_rproj, &c_true) < 1e-10);

    let c_plain = rproject_plain(&a, &q);
    assert!(sketch_frob_diff(&c_plain, &c_true) > 1e-3);

    let c_plain_t = rproject_plain_transpose(&a, &q);
    assert!(sketch_frob_diff(&c_plain_t, &c_true) > 1e-3);
}

#[test]
#[allow(clippy::type_complexity)]
fn o2_orthogonalization_exact_gauge_transform_every_two_sz() {
    for &(d, n, two_s) in &[(6usize, 4usize, 2usize), (7, 5, 1), (5, 4, 0)] {
        let mut rng = seeded(20_500 + two_s as u64 + d as u64);
        let space = Su2Space::new(d, n, two_s);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let ts_list = crate::s2::util::all_two_sz(two_s);
        let snapshots: Vec<Vec<f64>> = ts_list
            .iter()
            .map(|&ts| expand(&x0, ts).to_dense())
            .collect();

        let positions = [0usize, d / 2, d - 1];
        let sweeps: Vec<(Box<dyn Fn(&mut Su2TtVector<f64>)>, usize)> = vec![
            (
                Box::new(|x: &mut Su2TtVector<f64>| x.left_orthogonalize(false)),
                d - 1,
            ),
            (
                Box::new(|x: &mut Su2TtVector<f64>| x.right_orthogonalize(false)),
                0,
            ),
            (
                Box::new(move |x: &mut Su2TtVector<f64>| x.move_core(positions[0], false)),
                positions[0],
            ),
            (
                Box::new(move |x: &mut Su2TtVector<f64>| x.move_core(positions[1], false)),
                positions[1],
            ),
            (
                Box::new(move |x: &mut Su2TtVector<f64>| x.move_core(positions[2], false)),
                positions[2],
            ),
        ];

        for (sweep, center) in &sweeps {
            let mut x = x0.clone();
            sweep(&mut x);
            for (i, &ts) in ts_list.iter().enumerate() {
                let got = expand(&x, ts).to_dense();
                assert_dense_close(&got, &snapshots[i], 1e-13);
            }
            let ld = su2_left_canonicity_dev(&x, 0..*center);
            assert!(
                ld < 1e-13,
                "d={d} two_s={two_s} center={center}: left dev {ld}"
            );
            let rd = su2_right_canonicity_dev(&x, (*center + 1)..d);
            assert!(
                rd < 1e-13,
                "d={d} two_s={two_s} center={center}: right dev {rd}"
            );
        }
    }
}

#[test]
#[allow(clippy::type_complexity)]
fn o2_complex_gauge_transform_every_two_sz() {
    let mut rng = seeded(20_501);
    let space = Su2Space::new(6, 4, 2);
    let x0: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
    let ts_list = crate::s2::util::all_two_sz(space.two_s);
    let snapshots: Vec<Vec<C64>> = ts_list
        .iter()
        .map(|&ts| expand(&x0, ts).to_dense())
        .collect();

    let mid = space.d / 2;
    let sweeps: Vec<Box<dyn Fn(&mut Su2TtVector<C64>)>> = vec![
        Box::new(|x: &mut Su2TtVector<C64>| x.left_orthogonalize(false)),
        Box::new(|x: &mut Su2TtVector<C64>| x.right_orthogonalize(false)),
        Box::new(move |x: &mut Su2TtVector<C64>| x.move_core(mid, false)),
    ];
    for sweep in &sweeps {
        let mut x = x0.clone();
        sweep(&mut x);
        for (i, &ts) in ts_list.iter().enumerate() {
            let got = expand(&x, ts).to_dense();
            assert_dense_close(&got, &snapshots[i], 1e-11);
        }
    }
}

#[test]
fn o3_su2_right_canonicity_vs_documented_abelian_negative() {
    let mut rng = seeded(20_600);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    x.right_orthogonalize(false);

    let dev = su2_right_canonicity_dev(&x, 1..space.d);
    assert!(dev < 1e-13, "su2-side right-canonicity dev {dev}");

    let e = expand(&x, 0);
    let mut worst = 0.0f64;
    let mut saw_nonsquare = false;
    for k in 1..space.d {
        let core = e.core(k);
        for l in e.space.bond_sectors(k) {
            if core.row_ranks[l] == 0 {
                continue;
            }
            let h = core.horizontal_unfolding(l);
            if h.ncols() > h.nrows() {
                saw_nonsquare = true;
            }
            worst = worst.max(gram_row_dev(h.as_ref()));
        }
    }
    assert!(
        saw_nonsquare,
        "O3 non-vacuity guard: no non-square abelian horizontal unfolding seen"
    );
    assert!(
        worst > 0.1,
        "expanded cores were unexpectedly abelian-right-canonical: worst={worst}"
    );
}

#[test]
fn o4_weighted_round_bound_and_eps_to_zero_exactness() {
    let mut rng = seeded(20_700);
    let space = Su2Space::new(7, 5, 1);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 5, &mut rng);
    let ex = expand(&x, space.two_s as i64);
    let ex_dense = ex.to_dense();
    let ex_norm = dense_norm(&ex_dense);

    for &eps in &[1e-1_f64, 1e-2, 1e-4] {
        let y = x.round(eps, qntt::Pooled::NONE, RoundMode::Rel);
        let err = dense_diff_norm(&ex_dense, &expand(&y, space.two_s as i64).to_dense());
        assert!(
            err <= eps * ex_norm,
            "rel bound: err={err} > eps*||ex||={}",
            eps * ex_norm
        );

        let y_abs = x.round(eps, qntt::Pooled::NONE, RoundMode::Abs);
        let err_abs = dense_diff_norm(&ex_dense, &expand(&y_abs, space.two_s as i64).to_dense());
        assert!(err_abs <= eps, "abs bound: err={err_abs} > eps={eps}");
    }

    let mut x_lo = x.clone();
    x_lo.left_orthogonalize(false);
    let y0 = x.round(0.0, qntt::Pooled::NONE, RoundMode::Abs);
    let err0 = dense_diff_norm(&ex_dense, &expand(&y0, space.two_s as i64).to_dense());
    assert!(
        err0 <= 1e-12 * ex_norm.max(1.0),
        "eps=0 should be lossless: err0={err0}"
    );
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert!(
                y0.ranks[b][s] <= x_lo.ranks[b][s],
                "eps=0 must not inflate ranks past the left-canonical form: \
                 bond {b} sector {s:?}: {} > {}",
                y0.ranks[b][s],
                x_lo.ranks[b][s]
            );
        }
    }

    let y = x.round(1e-2, qntt::Pooled::NONE, RoundMode::Abs);
    let err_ab = dense_diff_norm(&ex_dense, &expand(&y, space.two_s as i64).to_dense());
    let err_su2 = (&x - &y).norm();
    assert!(
        close_rel_abs(err_su2, err_ab, 1e-6, 10.0 * f64::EPSILON.sqrt() * ex_norm),
        "SU2/abelian error-norm disagreement: err_su2={err_su2}, err_ab={err_ab}"
    );
}

#[test]
fn o4_complex_weighted_round_bound() {
    let mut rng = seeded(20_701);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
    let ex = expand(&x, space.two_s as i64);
    let ex_dense = ex.to_dense();
    let ex_norm = dense_norm(&ex_dense);

    let eps = 1e-2;
    let y = x.round(eps, qntt::Pooled::NONE, RoundMode::Rel);
    let err = dense_diff_norm(&ex_dense, &expand(&y, space.two_s as i64).to_dense());
    assert!(
        err <= eps * ex_norm + 1e-11,
        "rel bound: err={err} > eps*||ex||={}",
        eps * ex_norm
    );
}
