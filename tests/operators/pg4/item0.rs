// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg4::util::*;
use qntt::{Irrep, QnSpace, TtVector, max_ranks, tt_ones};

#[test]
#[allow(clippy::needless_range_loop)]
fn pg4_tt_ones_is_target_irrep_indicator_ga() {
    let (pg, space) = g_a();
    let x: TtVector<f64> = tt_ones(space);
    assert!(x.check(), "tt_ones output must pass check()");
    let dense = x.to_dense();

    let mut n_ones = 0usize;
    let mut n_block = 0usize;
    for (idx, &val) in dense.iter().enumerate() {
        let dig = digits(idx, space.d);
        let in_particle_block = config_counts(&dig) == (space.nup, space.ndn);
        let want = if in_particle_block && config_irrep_from_digits(&pg, &dig) == space.target {
            1.0
        } else {
            0.0
        };
        assert_eq!(
            val, want,
            "config {idx}: tt_ones must be the EXACT indicator"
        );
        if in_particle_block {
            n_block += 1;
        }
        if want == 1.0 {
            n_ones += 1;
        }
    }
    assert!(
        0 < n_ones && n_ones < n_block,
        "vacuous fixture: indicator is empty or the whole particle block ({n_ones} of {n_block})"
    );

    let caps = max_ranks(&space);
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert_eq!(
                x.rank(b, s),
                usize::from(caps[b][s] > 0),
                "bond {b} sector {s:?}: tt_ones rank must be 1 iff reachable, 0 otherwise"
            );
        }
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn pg4_tt_ones_is_target_irrep_indicator_gb() {
    let (pg, space) = g_b();
    let x: TtVector<f64> = tt_ones(space);
    assert!(x.check());
    let dense = x.to_dense();

    let mut n_ones = 0usize;
    let mut n_block = 0usize;
    for (idx, &val) in dense.iter().enumerate() {
        let dig = digits(idx, space.d);
        let in_particle_block = config_counts(&dig) == (space.nup, space.ndn);
        let want = if in_particle_block && config_irrep_from_digits(&pg, &dig) == space.target {
            1.0
        } else {
            0.0
        };
        assert_eq!(
            val, want,
            "config {idx}: tt_ones must be the EXACT indicator"
        );
        if in_particle_block {
            n_block += 1;
        }
        if want == 1.0 {
            n_ones += 1;
        }
    }
    assert!(0 < n_ones && n_ones < n_block, "vacuous G-B fixture");

    let caps = max_ranks(&space);
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert_eq!(x.rank(b, s), usize::from(caps[b][s] > 0));
        }
    }
}

#[test]
fn pg4_tt_ones_at_identity_target_nontrivial_group_gc() {
    let (pg, space) = g_c();
    assert!(pg.order() > 1);
    assert_eq!(space.target, Irrep::IDENTITY);

    let x: TtVector<f64> = tt_ones(space);
    assert!(x.check());
    let dense = x.to_dense();

    let mut n_ones = 0usize;
    let mut n_block = 0usize;
    for (idx, &val) in dense.iter().enumerate() {
        let dig = digits(idx, space.d);
        let in_particle_block = config_counts(&dig) == (space.nup, space.ndn);
        let want = if in_particle_block && config_irrep_from_digits(&pg, &dig) == Irrep::IDENTITY {
            1.0
        } else {
            0.0
        };
        assert_eq!(val, want, "config {idx}");
        if in_particle_block {
            n_block += 1;
        }
        if want == 1.0 {
            n_ones += 1;
        }
    }
    assert!(
        0 < n_ones && n_ones < n_block,
        "G-C: the identity-target indicator must be a PROPER, non-empty subset ({n_ones} of \
         {n_block}) — a vacuous split here means the group was not actually irrep-live"
    );
}

#[test]
fn pg4_tt_ones_trivial_group_matches_pre_pg4_grid() {
    let space = QnSpace::from_n_two_sz(6, 5, 1);
    assert!(space.pg.is_trivial());
    let x: TtVector<f64> = tt_ones(space);
    assert!(x.check());
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert_eq!(
                x.rank(b, s),
                1,
                "bond {b} sector {s:?}: |G|=1 must allocate rank 1 everywhere"
            );
        }
    }
    let dense = x.to_dense();
    for (idx, &val) in dense.iter().enumerate() {
        let dig = digits(idx, space.d);
        let want = if config_counts(&dig) == (space.nup, space.ndn) {
            1.0
        } else {
            0.0
        };
        assert_eq!(val, want, "config {idx}");
    }
}

#[test]
fn pg4_add_scalar_dense_oracle_linearity_at_nontrivial_group() {
    let (pg, space) = g_a();
    let x: TtVector<f64> = capped_randn(space, 1, 2, 40_401);
    let ones: TtVector<f64> = tt_ones(space);
    let xd = x.to_dense();
    let onesd = ones.to_dense();
    let c = 0.7_f64;

    let y = x.add_scalar(c);
    assert!(y.check(), "add_scalar output must pass check()");
    let want: Vec<f64> = xd.iter().zip(&onesd).map(|(&a, &o)| a + c * o).collect();
    let yd = y.to_dense();
    for (idx, (&g, &w)) in yd.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() < 1e-12,
            "config {idx}: add_scalar linearity, got {g} want {w}"
        );
    }
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);

    let z = (-&x).add_scalar(c);
    let want_m: Vec<f64> = xd.iter().zip(&onesd).map(|(&a, &o)| c * o - a).collect();
    let zd = z.to_dense();
    for (idx, (&g, &w)) in zd.iter().zip(&want_m).enumerate() {
        assert!((g - w).abs() < 1e-12, "config {idx}: mirror form");
    }
}

#[test]
fn pg4_add_scalar_dense_oracle_linearity_gb() {
    let (pg, space) = g_b();
    let x: TtVector<f64> = capped_randn(space, 1, 2, 40_402);
    let ones: TtVector<f64> = tt_ones(space);
    let xd = x.to_dense();
    let onesd = ones.to_dense();
    let c = -1.3_f64;

    let y = x.add_scalar(c);
    assert!(y.check());
    let want: Vec<f64> = xd.iter().zip(&onesd).map(|(&a, &o)| a + c * o).collect();
    let yd = y.to_dense();
    for (g, w) in yd.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12);
    }
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
}
