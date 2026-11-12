// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg3::*;

#[test]
fn pg3_round_abs_rel_equivalence() {
    let (pg, space) = pg3_space_d6();
    let eps = 1e-4;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_012);

    let xs: Vec<TtVector<f64>> = (1..=5)
        .map(|i| {
            let ranks = capped_ranks(&space, 2, 3, &mut rng);
            let xi: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
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
        assert!(y.check(), "{tag}: check()");
        assert_pure(&pg, &space, y);
    }
    assert_eq!(
        sum_interior_rank(&y_rel),
        sum_interior_rank(&y_abs),
        ":rel(ε) and :abs(ε·‖x‖) give identical total interior rank"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn pg3_round_ranks_clamps_and_truncates() {
    let (pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_014);
    let ranks0 = capped_ranks(&space, 2, 3, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks0, &mut rng);
    assert!(x.check());
    let xd = x.to_dense();
    let d = space.d;

    let over: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d { 1 } else { 1_000 };
            }
            r
        })
        .collect();
    let mut y_full = x.clone();
    y_full.round_ranks_mut(&over);
    assert!(y_full.check());
    assert_eq!(y_full.ortho, Some(0), "round_ranks is right-canonical");
    assert!(
        rel_err(&y_full.to_dense(), &xd) < 1e-12,
        "over-request keeps all singular values (clamp, no OOB)"
    );
    assert_pure(&pg, &space, &y_full);

    let caps = max_ranks(&space);
    let mut any_zero_cap = false;
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            if caps[b][s] == 0 {
                any_zero_cap = true;
                assert_eq!(
                    y_full.ranks[b][s], 0,
                    "bond {b} {s:?}: cap-0 sector acquired rank"
                );
            }
        }
    }
    assert!(any_zero_cap, "vacuous: no cap-0 sector present");

    let top = d - 1;
    let mut t = over.clone();
    for (j, s) in space.bond_sectors(top).into_iter().enumerate() {
        t[top][s] = if j % 2 == 0 { 1 } else { 1_000 };
    }
    let mut y = x.clone();
    y.round_ranks_mut(&t);
    assert!(y.check());
    for s in space.bond_sectors(top) {
        let expected = t[top][s].min(y_full.ranks[top][s]);
        assert_eq!(
            y.ranks[top][s], expected,
            "bond {top} {s:?}: rank == min(target, available)"
        );
    }
    assert_pure(&pg, &space, &y);

    let ones: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = 1;
            }
            r
        })
        .collect();
    let mut z = x.clone();
    z.round_ranks_mut(&ones);
    assert!(z.check());
    assert_eq!(z.ortho, Some(0));
    for b in 1..d {
        for s in space.bond_sectors(b) {
            assert!(z.ranks[b][s] <= 1, "bond {b} {s:?} truncated to ≤ 1");
        }
    }
    assert!(sum_interior_rank(&z) < sum_interior_rank(&y_full));
    assert!(rel_err(&z.to_dense(), &xd).is_finite());
    assert_pure(&pg, &space, &z);
}

#[test]
fn pg3_round_rank_cap_honored() {
    let (pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_015);
    let ranks = capped_ranks(&space, 2, 3, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let xd = x.to_dense();
    let d = space.d;

    let y = x.round(1e-14, qntt::Pooled(2), RoundMode::Abs);
    assert!(y.check());
    for b in 1..d {
        assert!(
            y.ranks[b].total() <= 2,
            "bond {b} total rank ≤ rmax=2 (maxRank arm)"
        );
    }
    assert_pure(&pg, &space, &y);

    let z = x.round(0.0, qntt::Pooled(2), RoundMode::Abs);
    assert!(z.check());
    for b in 1..d {
        assert!(
            z.ranks[b].total() <= 2,
            "bond {b} total rank ≤ rmax=2 (chop_rank)"
        );
    }
    assert_pure(&pg, &space, &z);

    let z4 = x.round(0.0, qntt::Pooled(4), RoundMode::Abs);
    assert!(z4.check());
    for b in 1..d {
        assert!(z4.ranks[b].total() <= 4);
    }
    assert!(sum_interior_rank(&z4) >= sum_interior_rank(&z));
    assert!(rel_err(&z4.to_dense(), &xd) <= rel_err(&z.to_dense(), &xd) + 1e-12);
    assert_pure(&pg, &space, &z4);
}

#[test]
fn pg3_round_keep_one_fallback_nonzero() {
    let (pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_016);
    let ranks = capped_ranks(&space, 1, 2, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let nx = x.norm();

    let y = x.round(10.0 * nx, qntt::Pooled::NONE, RoundMode::Abs);
    assert!(y.check());
    for b in 1..space.d {
        assert!(
            y.ranks[b].total() <= 1,
            "bond {b} keeps at most one direction"
        );
    }
    assert!(dense_diff_norm(&y.to_dense(), &x.to_dense()) <= 10.0 * nx);
    assert_pure(&pg, &space, &y);
}
