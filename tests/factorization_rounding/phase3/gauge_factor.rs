// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase3::*;

#[test]
fn round_abs_rel_equivalence() {
    let space = QnSpace::from_n_two_sz(8, 4, 0);
    let eps = 1e-4;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_007);

    let xs: Vec<TtVector<f64>> = (1..=5)
        .map(|i| {
            let xi: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
            let scale = 10f64.powi(1 - i) / xi.norm();
            &xi * scale
        })
        .collect();
    let mut x = xs[0].clone();
    for xi in &xs[1..] {
        x = &x + xi;
    }
    x = (1e3 / x.norm()) * &x;
    let nx = x.norm();
    let xd = x.to_dense();

    let y_rel = x.round(eps, qntt::Pooled::NONE, RoundMode::Rel);
    let y_abs = x.round(eps * nx, qntt::Pooled::NONE, RoundMode::Abs);
    let g_rel = x.round_global(eps, qntt::PerSector::NONE, RoundMode::Rel);

    for (tag, y) in [("rel", &y_rel), ("abs", &y_abs), ("global", &g_rel)] {
        assert!(
            dense_diff_norm(&y.to_dense(), &xd) <= eps * nx * (1.0 + 1e-6),
            "{tag}: ‖· − x‖ ≤ ε·‖x‖"
        );
    }
    assert_eq!(
        sum_interior_rank(&y_rel),
        sum_interior_rank(&y_abs),
        ":rel(ε) and :abs(ε·‖x‖) give identical total interior rank"
    );
}

#[test]
fn qr_lr_rl_boundary_factor() {
    let space = QnSpace::from_n_two_sz(5, 4, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(301_010);
    let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let xd = x.to_dense();

    let (y, r) = x.qr_lr();
    y.check();
    assert_eq!((r.nrows(), r.ncols()), (1, 1), "R is 1×1");
    let r00 = *r.get(0, 0);
    let recon: Vec<f64> = y.to_dense().iter().map(|&v| v * r00).collect();
    assert!(
        dense_diff_norm(&recon, &xd) <= 1e-10 * dense_norm(&xd),
        "‖y·R − x‖ ≤ 1e-10‖x‖"
    );

    let (y2, l) = x.qr_rl();
    y2.check();
    assert_eq!((l.nrows(), l.ncols()), (1, 1), "L is 1×1");
    let l00 = *l.get(0, 0);
    let recon2: Vec<f64> = y2.to_dense().iter().map(|&v| l00 * v).collect();
    assert!(
        dense_diff_norm(&recon2, &xd) <= 1e-10 * dense_norm(&xd),
        "‖L·y − x‖ ≤ 1e-10‖x‖"
    );
}

#[test]
fn svd_vertical_reconstructs_blocks() {
    let space = QnSpace::from_n_two_sz(5, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(301_012);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let a = &x.cores[2];

    let (u, s, vt) = a.svd_vertical();
    for &cs in &a.col_qn {
        if a.col_ranks[cs] == 0 {
            continue;
        }
        let av = a.vertical_unfolding(cs);
        let ub = u[cs].as_ref().expect("U block present");
        let sb = s.block(cs).expect("S block present");
        let vtb = vt.block(cs).expect("Vt block present");
        let recon = mat_mul(mat_mul(ub.as_ref(), sb).as_ref(), vtb);
        assert!(
            frob_diff(recon.as_ref(), av.as_ref()) <= 1e-12 * frob(av.as_ref()).max(1.0),
            "U·S·Vt reconstructs the vertical unfolding at {cs:?}"
        );
    }
}

#[test]
fn svd_horizontal_reconstructs_blocks() {
    let space = QnSpace::from_n_two_sz(5, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(301_013);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let a = &x.cores[2];

    let (u, s, vt) = a.svd_horizontal();
    for &rs in &a.row_qn {
        if a.row_ranks[rs] == 0 {
            continue;
        }
        let ah = a.horizontal_unfolding(rs);
        let ub = u.block(rs).expect("U block present");
        let sb = s.block(rs).expect("S block present");
        let vtb = vt[rs].as_ref().expect("Vt block present");
        let recon = mat_mul(mat_mul(ub, sb).as_ref(), vtb.as_ref());
        assert!(
            frob_diff(recon.as_ref(), ah.as_ref()) <= 1e-12 * frob(ah.as_ref()).max(1.0),
            "U·S·Vt reconstructs the horizontal unfolding at {rs:?}"
        );
    }
}
