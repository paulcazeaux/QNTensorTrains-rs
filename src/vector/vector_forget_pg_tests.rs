// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::{TtVector, apply_group_element, symmetry_defect};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{QnSpace, Sector};
use crate::round::RoundMode;
use crate::test_util::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const D: usize = 4;
const NUP: usize = 2;
const NDN: usize = 1;

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
    )
}

fn typed_state(target: Irrep, seed: u64) -> (QnSpace, TtVector<f64>) {
    let space = QnSpace::with_point_group(D, NUP, NDN, fx_pg(), target);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-10, crate::qn::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    (space, x)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().map(|x| x.abs()).fold(0.0f64, f64::max)
}

#[test]
fn vfp_forget_is_dense_identical_at_pooled_ranks() {
    let pg = fx_pg();
    let untyped = QnSpace::new(D, NUP, NDN);
    for g in 0..pg.order() {
        let target = Irrep::new(g);
        let (_space, x) = typed_state(target, 0xFA0 + g as u64);
        let f = x.forget_point_group();

        assert_eq!(f.space, untyped, "target {g}: wrong output space");
        assert!(f.check(), "target {g}: the embedding is not a legal train");
        assert_eq!(f.ortho, x.ortho, "target {g}: `ortho` must survive");

        for b in 0..=D {
            for s in untyped.bond_sectors(b) {
                let pooled: usize = (0..pg.order())
                    .map(|ir| x.ranks[b][Sector::with_irrep(s.nup, s.ndn, Irrep::new(ir))])
                    .sum();
                assert_eq!(
                    f.ranks[b][s], pooled,
                    "target {g}, bond {b}, sector {s:?}: rank {} != pooled {pooled}",
                    f.ranks[b][s]
                );
            }
        }

        let xd = x.to_dense();
        let fd = f.to_dense();
        let scale = max_abs(&xd);
        assert!(
            scale > 1e-3,
            "target {g}: anti-vacuity — the typed fixture has no amplitude ({scale:e})"
        );
        let err = max_abs_diff(&xd, &fd);
        println!("vfp target {g}: max |typed - forgotten| = {err:e} (scale {scale:e})");
        assert!(
            err <= 1e-14 * scale,
            "target {g}: forget_point_group changed the coefficients (max diff {err:e}, \
             scale {scale:e}) — a wrong per-irrep block offset looks exactly like this"
        );
    }
}

#[test]
fn vfp_forget_keeps_the_reference_character() {
    let pg = fx_pg();
    for g in 1..pg.order() {
        let target = Irrep::new(g);
        let (_space, x) = typed_state(target, 0xFB0 + g as u64);
        let f = x.forget_point_group();
        let fd = f.to_dense();
        let scale = max_abs(&fd);
        assert!(scale > 1e-3, "target {g}: anti-vacuity ({scale:e})");

        let mut saw_minus = false;
        for h in 0..pg.order() {
            let hh = Irrep::new(h);
            let chi = target.character(hh);
            if chi < 0.0 {
                saw_minus = true;
            }
            let ud: Vec<f64> = apply_group_element(&f, pg, hh).to_dense();
            let want: Vec<f64> = fd.iter().map(|v| chi * v).collect();
            let err = max_abs_diff(&ud, &want);
            assert!(
                err <= 1e-14 * scale,
                "target {g}, h = {h}: U_h forget(x) != chi({chi}) forget(x), max diff {err:e}"
            );
        }
        assert!(
            saw_minus,
            "target {g}: anti-vacuity — every character was +1, so this arm discriminated nothing"
        );

        let (gamma, defects) = symmetry_defect(&f, pg);
        let worst = defects.iter().cloned().fold(0.0f64, f64::max);
        println!("vfp target {g}: symmetry_defect gamma={gamma:?} worst={worst:e}");
        assert_eq!(gamma, target, "target {g}: wrong irrep read back");
        assert!(
            worst <= 1e-7,
            "target {g}: defect {worst:e} above the floor"
        );
    }
}

#[test]
fn vfp_forget_on_a_trivial_group_is_a_clone() {
    let space = QnSpace::new(D, NUP, NDN);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFC0);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-10, crate::qn::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let f = x.forget_point_group();
    assert_eq!(f.space, x.space);
    assert_eq!(f.ranks, x.ranks);
    assert_eq!(f.ortho, x.ortho);
    let (a, b) = (x.to_dense(), f.to_dense());
    assert!(
        max_abs(&a) > 1e-3,
        "anti-vacuity: the fixture has no amplitude"
    );
    assert_eq!(a, b, "the trivial-group path must be an exact clone");
}
