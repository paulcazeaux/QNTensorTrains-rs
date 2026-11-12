// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::core::{Family, Su2Core};
use qntt::su2::expand::{ab_layout, expand};
use qntt::su2::frame::Su2Sketch;
use qntt::su2::ops::dot as su2_dot;
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks};
use qntt::su2::vector::{Su2TtVector, tt_csf_with_point_group, tt_randn, tt_zeros};
use qntt::{C64, Irrep, PointGroup, QnSpace, Scalar, TtVector, dot as ab_dot};
use qntt_testkit::pg::assert_irrep_split;
use qntt_testkit::pg::c2v_table;
use qntt_testkit::pg::config_counts;
use qntt_testkit::pg::config_irrep_from_digits;
use qntt_testkit::pg::digits;
use qntt_testkit::pg::su2_capped_ranks;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn build_pg(nbits: u8, d: usize) -> PointGroup {
    let order = 1u8 << nbits;
    let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k as u8) % order)).collect();
    PointGroup::new(nbits, &irreps)
}

fn find_nonidentity_reachable_target(
    pg: PointGroup,
    n: usize,
    two_s: usize,
    d: usize,
) -> Option<Irrep> {
    (1..pg.order()).map(Irrep::new).find(|&ir| {
        let space = Su2Space::with_point_group(d, n, two_s, pg, ir);
        let caps = max_ranks(&space);
        caps[d][Su2Sector::with_irrep(n, two_s, ir)] > 0
    })
}

fn codes_for_two_s(d: usize, want_two_s: usize) -> Vec<Family> {
    assert!(
        want_two_s <= d,
        "fixture bug: d={d} too small for 2s={want_two_s}"
    );
    let mut codes = Vec::with_capacity(d);
    for _ in 0..want_two_s {
        codes.push(Family::Up);
    }
    while codes.len() < d {
        if d - codes.len() >= 2 {
            codes.push(Family::Up);
            codes.push(Family::Dn);
        } else {
            codes.push(Family::Empty);
        }
    }
    codes
}

fn assert_purity<T: Scalar>(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    target: Irrep,
    dense: &[T],
) {
    assert_eq!(dense.len(), 1usize << (2 * d));
    let mut any_nonzero_in_target = false;
    for (idx, v) in dense.iter().enumerate() {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            assert!(
                *v == T::zero(),
                "d={d}: config {idx} outside (Nup={nup},Ndn={ndn}) must be exactly zero, got {v:?}"
            );
            continue;
        }
        let irrep = config_irrep_from_digits(pg, &dig);
        if irrep == target {
            if *v != T::zero() {
                any_nonzero_in_target = true;
            }
        } else {
            assert!(
                *v == T::zero(),
                "THE campaign oracle: d={d} config {idx} has irrep {irrep:?} != target \
                 {target:?} but amplitude is {v:?} (not exactly zero) — a hand-rolled inversion, \
                 an irrep_of(k)/irrep_of(k+1) swap, or a Family::flux/qn::ODD_ELECTRON \
                 divergence would show up here"
            );
        }
    }
    assert!(
        any_nonzero_in_target,
        "d={d}: oracle vacuous — no nonzero amplitude found inside the target irrep {target:?}"
    );
}

fn assert_expand_purity_chain<T: Scalar>(x: &Su2TtVector<T>, pg: &PointGroup) {
    let d = x.space.d;
    let two_s = x.space.two_s as i64;
    let target = x.space.target;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let nup = ((x.space.n as i64 + two_sz) / 2) as usize;
            let ndn = ((x.space.n as i64 - two_sz) / 2) as usize;
            let y: TtVector<T> = expand(x, two_sz);
            assert_eq!(
                y.space,
                QnSpace::with_point_group(d, nup, ndn, x.space.pg, target),
                "expand(x, two_sz={two_sz}): abelian space mismatch"
            );
            assert!(
                y.check(),
                "expand(x, two_sz={two_sz}): abelian check() failed"
            );
            assert_irrep_split(pg, d, nup, ndn);
            assert_purity(pg, d, nup, ndn, target, &y.to_dense());
        }
        two_sz += 1;
    }
}

fn csf_purity_case<T: Scalar>(d: usize, want_two_s: usize) {
    let pg = c2v_table(d);
    let codes = codes_for_two_s(d, want_two_s);
    let x: Su2TtVector<T> = tt_csf_with_point_group(&codes, pg);
    assert!(x.check());
    assert_expand_purity_chain(&x, &pg);
}

fn randn_purity_case<T: Scalar>(d: usize, want_two_s: usize, target: Irrep, seed: u64) {
    let pg = c2v_table(d);
    let n = want_two_s + 2;
    let space = Su2Space::with_point_group(d, n, want_two_s, pg, target);
    let ranks = su2_capped_ranks(&space, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x: Su2TtVector<T> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    assert_expand_purity_chain(&x, &pg);
}

#[test]
fn spg2_purity_oracle_tt_csf_with_point_group_f64() {
    for d in [4usize, 5, 7] {
        for want_two_s in [0usize, 1, 2] {
            csf_purity_case::<f64>(d, want_two_s);
        }
    }
}

#[test]
fn spg2_purity_oracle_tt_csf_with_point_group_complex() {
    csf_purity_case::<C64>(5, 1);
    csf_purity_case::<C64>(6, 2);
}

#[test]
fn spg2_purity_oracle_tt_randn_f64() {
    for d in [4usize, 6] {
        for want_two_s in [0usize, 1, 2] {
            let pg = c2v_table(d);
            randn_purity_case::<f64>(
                d,
                want_two_s,
                Irrep::IDENTITY,
                1000 + d as u64 + want_two_s as u64,
            );
            if let Some(target) =
                find_nonidentity_reachable_target(pg, want_two_s + 2, want_two_s, d)
            {
                randn_purity_case::<f64>(
                    d,
                    want_two_s,
                    target,
                    2000 + d as u64 + want_two_s as u64,
                );
            }
        }
    }
}

#[test]
fn spg2_purity_oracle_tt_randn_complex() {
    let pg = c2v_table(6);
    if let Some(target) = find_nonidentity_reachable_target(pg, 3, 1, 6) {
        randn_purity_case::<C64>(6, 1, target, 4242);
    }
    randn_purity_case::<C64>(6, 1, Irrep::IDENTITY, 4343);
}

#[test]
fn spg2_r4_pin_two_site_hand_computed_dense() {
    let pg = PointGroup::new(1, &[Irrep::new(0), Irrep::new(1)]);
    let codes = [Family::Up, Family::Dn];
    let x: Su2TtVector<f64> = tt_csf_with_point_group(&codes, pg);
    assert_eq!(x.space.n, 2);
    assert_eq!(x.space.two_s, 0);
    assert_eq!(x.space.target, Irrep::new(1));
    assert!(x.check());

    let y: TtVector<f64> = expand(&x, 0);
    assert!(y.check());
    let dense = y.to_dense();
    assert_eq!(dense.len(), 16);

    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut expected = [0.0f64; 16];
    expected[1 + 2 * 4] = inv_sqrt2;
    expected[2 + 4] = -inv_sqrt2;

    for (idx, (&got, &want)) in dense.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "R4 pin: dense[{idx}] = {got} but hand-derivation predicts {want}"
        );
    }

    assert_purity(&pg, 2, 1, 1, Irrep::new(1), &dense);
}

#[test]
fn spg2_check_invariants_passes_on_correct_core_at_nontrivial_group() {
    let d = 4;
    let pg = build_pg(2, d);
    let n = 3;
    let two_s = 1;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or(Irrep::IDENTITY);
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let ranks = su2_capped_ranks(&space, 2);
    let core0 = Su2Core::<f64>::zeros(space, 0, &ranks[0], &ranks[1]);
    assert!(core0.check_invariants());
}

#[test]
#[should_panic(expected = "has wrong shape")]
fn spg2_check_invariants_fails_on_corrupted_column_rank() {
    let d = 4;
    let pg = build_pg(1, d);
    let n = 3;
    let two_s = 1;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or(Irrep::IDENTITY);
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let ranks = su2_capped_ranks(&space, 3);
    let mut core0 = Su2Core::<f64>::zeros(space, 0, &ranks[0], &ranks[1]);
    assert!(core0.check_invariants(), "fixture bug: must start valid");

    let g0 = pg.irrep_of(0);
    let s = core0
        .transitions(Family::Up)
        .next()
        .expect("fixture bug: need a populated Up transition at site 0");
    let t = Family::Up
        .apply(s, g0)
        .expect("transition admissible by construction");

    let wrong_irrep = Irrep::new((t.irrep.index() + 1) % pg.order());
    let wrong_t = Su2Sector::with_irrep(t.n, t.two_s, wrong_irrep);
    let real = core0.col_ranks[t];
    let mut corrupt = core0.col_ranks[wrong_t];
    if corrupt == real {
        corrupt += 1;
    }
    core0.col_ranks[t] = corrupt;

    core0.check_invariants();
}

#[test]
fn spg2_tt_zeros_boundary_sectors_at_nontrivial_target() {
    let d = 4;
    let pg = build_pg(2, d);
    let n = 3;
    let two_s = 1;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let x: Su2TtVector<f64> = tt_zeros(space);
    assert!(x.check());

    assert_eq!(x.ranks[0][Su2Sector::new(0, 0)], 1);
    assert_eq!(x.ranks[d][Su2Sector::with_irrep(n, two_s, target)], 1);
    assert_eq!(
        x.ranks[d][Su2Sector::with_irrep(n, two_s, Irrep::IDENTITY)],
        0
    );

    for (s, &r) in x.ranks[0].iter() {
        assert!(
            r == 0 || s == Su2Sector::new(0, 0),
            "bond 0: stray rank at {s:?}"
        );
    }
    for (s, &r) in x.ranks[d].iter() {
        assert!(
            r == 0 || s == Su2Sector::with_irrep(n, two_s, target),
            "bond {d}: stray rank at {s:?}"
        );
    }
}

#[test]
#[should_panic(expected = "right boundary sector")]
fn spg2_check_rejects_wrong_right_boundary_irrep() {
    let d = 4;
    let pg = build_pg(2, d);
    let n = 3;
    let two_s = 1;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let space0 = Su2Space::with_point_group(d, n, two_s, pg, Irrep::IDENTITY);
    let ranks = su2_capped_ranks(&space0, 2);

    let cores: Vec<Su2Core<f64>> = (0..d)
        .map(|k| Su2Core::zeros(space, k, &ranks[k], &ranks[k + 1]))
        .collect();
    let x: Su2TtVector<f64> = Su2TtVector::new(space, cores);
    x.check();
}

#[test]
fn spg2_sketch_identity_gated_at_nontrivial_group() {
    let d = 4;
    let pg = build_pg(2, d);
    let n = 3;
    let two_s = 1;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let caps = max_ranks(&space);

    for k in [0usize, 1, d] {
        let r = 3usize;
        let sketch = Su2Sketch::<f64>::identity(space, k, r);
        let mut any_gated = false;
        let mut any_full = false;
        for s in space.bond_sectors(k) {
            let block = sketch
                .block(s)
                .expect("presence must be contains-driven (Some)");
            let expect_n = if caps[k][s] > 0 { r } else { 0 };
            assert_eq!(
                (block.nrows(), block.ncols()),
                (expect_n, expect_n),
                "bond {k} sector {s:?}: identity extent must be gated by max_ranks reachability"
            );
            if expect_n == 0 {
                any_gated = true;
            } else {
                any_full = true;
                for i in 0..expect_n {
                    assert_eq!(*block.get(i, i), 1.0);
                }
            }
        }
        assert!(
            any_full,
            "bond {k}: no reachable sector found (vacuous fixture)"
        );
        if k == 0 || k == d {
            assert!(
                any_gated,
                "bond {k}: expected at least one gated (0x0) sector"
            );
        }
    }
}

#[test]
fn spg2_ab_layout_partitions_by_irrep_rather_than_dropping_rank() {
    let d = 5;
    let pg = build_pg(2, d);
    let n = 3;
    let two_s = 1;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or(Irrep::IDENTITY);
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let all_ranks = max_ranks(&space);
    let ranks = &all_ranks[2];

    for nn in 0..=n {
        for twom in -(nn as i64)..=(nn as i64) {
            let expected: usize = space
                .bond_sectors(2)
                .into_iter()
                .filter(|s| s.n == nn && (s.two_s as i64) >= twom.abs())
                .map(|s| ranks[s])
                .sum();
            let partitioned: usize = (0..pg.order())
                .map(|ir| ab_layout(&space, 2, ranks, nn, twom, Irrep::new(ir)).2)
                .sum();
            assert_eq!(
                partitioned, expected,
                "nn={nn} twom={twom}: per-lambda ab_layout totals must sum to the irrep-blind \
                 total (partition, not drop)"
            );
        }
    }
}

#[test]
fn spg2_dot_no_panic_and_matches_expand_dot_at_nontrivial_group() {
    let d = 5;
    let pg = build_pg(2, d);
    let n = 4;
    let two_s = 0;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let ranks = su2_capped_ranks(&space, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0D0_1234);
    let x: Su2TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());

    let via_su2 = su2_dot(&x, &x);
    assert!(via_su2 > 0.0, "vacuous: x is the zero vector");

    let y: TtVector<f64> = expand(&x, 0);
    let via_ab = ab_dot(&y, &y);
    assert!(
        (via_su2 - via_ab).abs() < 1e-12 * (1.0 + via_ab.abs()),
        "su2 dot={via_su2} but abelian dot(expand(x,0),expand(x,0))={via_ab}"
    );
}

#[test]
fn spg2_check_invariants_passes_on_every_transition_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let n = 4;
    let two_s = 2;
    let target = find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or(Irrep::IDENTITY);
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let ranks = su2_capped_ranks(&space, 3);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xABCD_1234);
    let x: Su2TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
}
