// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::centerweight;
use qntt::su2::expand::expand;
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;

use qntt::{RoundMode, Scalar};

use crate::s2::util::{
    assert_dense_close, close_rel_abs, dense_diff_norm, dense_norm, gram_col_dev, random_csf,
    randomized_su2_state, seeded, su2_left_canonicity_dev, su2_right_canonicity_dev,
};

#[test]
fn r1_left_orthogonalize_invariance_and_left_canonicity() {
    let mut rng = seeded(20_101);
    let space = Su2Space::new(7, 5, 1);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 0, 4, &mut rng);
    let want = expand(&x0, space.two_s as i64).to_dense();

    let mut x = x0.clone();
    x.left_orthogonalize(false);

    let got = expand(&x, space.two_s as i64).to_dense();
    assert_dense_close(&got, &want, 1e-12);

    let dev = su2_left_canonicity_dev(&x, 0..(space.d - 1));
    assert!(dev < 1e-13, "su2 left-canonicity dev {dev}");

    let e = expand(&x, space.two_s as i64);
    for k in 0..(space.d - 1) {
        let core = e.core(k);
        for s in e.space.bond_sectors(k + 1) {
            if core.col_ranks[s] == 0 {
                continue;
            }
            let v = core.vertical_unfolding(s);
            assert!(gram_col_dev(v.as_ref()) < 1e-13);
        }
    }
}

#[test]
fn r2_left_orthogonalize_keep_rank_preserves_ranks() {
    let mut rng = seeded(20_102);
    let space = Su2Space::new(7, 4, 2);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 0, 4, &mut rng);
    let want = expand(&x0, space.two_s as i64).to_dense();
    let r_before = x0.ranks.clone();

    let mut x = x0.clone();
    x.left_orthogonalize(true);

    let got = expand(&x, space.two_s as i64).to_dense();
    assert_dense_close(&got, &want, 1e-12);
    assert_eq!(
        x.ranks, r_before,
        "keepRank must preserve every bond rank grid exactly"
    );
}

#[test]
fn r3_right_orthogonalize_invariance_and_theta_weighted_canonicity() {
    let mut rng = seeded(20_103);
    let space = Su2Space::new(7, 4, 2);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 0, 4, &mut rng);
    let want = expand(&x0, space.two_s as i64).to_dense();

    let mut x = x0.clone();
    x.right_orthogonalize(false);

    let got = expand(&x, space.two_s as i64).to_dense();
    assert_dense_close(&got, &want, 1e-12);

    let dev = su2_right_canonicity_dev(&x, 1..space.d);
    assert!(dev < 1e-13, "su2 theta-weighted right-canonicity dev {dev}");
}

#[test]
fn r4_right_orthogonalize_keep_rank_rank_preservation_only() {
    let mut rng = seeded(20_104);
    let space = Su2Space::new(7, 4, 0);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 0, 4, &mut rng);
    let want = expand(&x0, space.two_s as i64).to_dense();
    let r_before = x0.ranks.clone();

    let mut x = x0.clone();
    x.right_orthogonalize(true);

    let got = expand(&x, space.two_s as i64).to_dense();
    assert_dense_close(&got, &want, 1e-12);
    assert_eq!(
        x.ranks, r_before,
        "keepRank must preserve every bond rank grid exactly"
    );
}

#[test]
fn r5_move_core_multiple_positions_canonicity_ranges() {
    let mut rng = seeded(20_105);
    let space = Su2Space::new(6, 4, 2);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 4, &mut rng);
    let want = expand(&x0, space.two_s as i64).to_dense();

    for &pos in &[2usize, 0, space.d - 1, 1] {
        let mut x = x0.clone();
        x.move_core(pos, false);

        let got = expand(&x, space.two_s as i64).to_dense();
        assert_dense_close(&got, &want, 1e-12);

        let left_dev = su2_left_canonicity_dev(&x, 0..pos);
        assert!(left_dev < 1e-13, "pos={pos}: left dev {left_dev}");
        let right_dev = su2_right_canonicity_dev(&x, (pos + 1)..space.d);
        assert!(right_dev < 1e-13, "pos={pos}: right dev {right_dev}");
    }
}

#[test]
fn r6_lossless_round() {
    let mut rng = seeded(20_106);
    let space = Su2Space::new(7, 5, 3);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 4, &mut rng);
    let want = expand(&x, space.two_s as i64).to_dense();

    x.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);

    let got = expand(&x, space.two_s as i64).to_dense();
    assert_dense_close(&got, &want, 1e-12);
}

#[test]
fn r9_round_global_bound() {
    let mut rng = seeded(20_109);
    let space = Su2Space::new(6, 4, 0);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let ex = expand(&x, space.two_s as i64);
    let tol = 1e-2;

    let y = x.round_global(tol, qntt::PerSector::NONE, RoundMode::Abs);
    y.check();
    let ey = expand(&y, space.two_s as i64);
    let err = dense_diff_norm(&ex.to_dense(), &ey.to_dense());
    assert!(
        err <= 2.0 * tol,
        "global round bound violated: err={err} > 2*tol={}",
        2.0 * tol
    );
}

#[test]
fn r11_x_plus_neg_x_rounds_to_zero() {
    let mut rng = seeded(20_111);
    let x: Su2TtVector<f64> = random_csf(6, 4, 2, &mut rng);
    let neg_x = &x * (-1.0);
    let mut z = &x + &neg_x;
    z.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Abs);
    assert!(z.norm() < 1e-12, "norm(z)={} not ~0", z.norm());

    for b in 1..z.space.d {
        let total: usize = z.space.bond_sectors(b).iter().map(|&s| z.ranks[b][s]).sum();
        assert!(
            total <= 1,
            "bond {b}: keep-one must collapse to total rank ≤ 1, got {total}"
        );
    }
}

#[test]
fn r12_svdvals_weighted_sigma2_sum_equals_squared_norm() {
    let mut rng = seeded(20_112);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 4, &mut rng);
    let nrm2 = dense_norm(&expand(&x, space.two_s as i64).to_dense()).powi(2);

    let sv = x.svdvals();
    for k in 1..space.d {
        let sk = &sv[k - 1];
        let mut s2 = 0.0f64;
        for sec in sk.sectors() {
            let Some(b) = sk.block(sec) else { continue };
            let cw = centerweight(sec.two_s, space.two_s);
            let n = b.nrows().min(b.ncols());
            for i in 0..n {
                let v = b.get(i, i).abs_s();
                s2 += cw * v * v;
            }
        }
        assert!(
            close_rel_abs(s2, nrm2, 1e-10, 0.0),
            "bond {k}: weighted sigma^2 sum {s2} != ||x||^2 {nrm2}"
        );
    }
}
