// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::irrep_weights;
use crate::C64;
use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;
use crate::su2::expand::expand;
use crate::su2::qn::{Su2Sector, Su2Space, max_ranks};
use crate::su2::vector::{Su2TtVector, tt_randn};
use crate::test_util::{randomized_su2_state, seeded, su2_capped_grids};

const D: usize = 5;

const NS: [(usize, usize); 3] = [(4, 0), (3, 1), (4, 2)];

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[
            Irrep::new(0),
            Irrep::new(1),
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
        ],
    )
}

fn typed_space(n: usize, two_s: usize, target: Irrep) -> Su2Space {
    Su2Space::with_point_group(D, n, two_s, fx_pg(), target)
}

fn reachable_targets(n: usize, two_s: usize) -> Vec<Irrep> {
    let out: Vec<Irrep> = (0..fx_pg().order())
        .map(Irrep::new)
        .filter(|&g| {
            let sp = typed_space(n, two_s, g);
            max_ranks(&sp)[D][Su2Sector::with_irrep(n, two_s, g)] > 0
        })
        .collect();
    assert!(
        out.len() >= 2,
        "fixture: (N, 2S) = ({n}, {two_s}) reaches only {} target(s)",
        out.len()
    );
    out
}

fn typed_state<T: Scalar>(n: usize, two_s: usize, target: Irrep, seed: u64) -> Su2TtVector<T> {
    let space = typed_space(n, two_s, target);
    tt_randn::<T, _>(space, &su2_capped_grids(&space, 2), &mut seeded(seed))
}

fn members(two_s: usize) -> Vec<i64> {
    let t = two_s as i64;
    (0..=t).map(|i| -t + 2 * i).collect()
}

fn det_irrep(pg: PointGroup, idx: usize) -> Irrep {
    (0..D).fold(Irrep::IDENTITY, |acc, k| match (idx >> (2 * k)) & 3 {
        1 | 2 => acc.fuse(pg.irrep_of(k)),
        _ => acc,
    })
}

fn dense_decomposition<T: Scalar>(v: &[T], pg: PointGroup) -> (Vec<f64>, Vec<f64>) {
    let order = usize::from(pg.order());
    let mut chars = vec![0.0f64; order];
    let mut weights = vec![0.0f64; order];
    let mut total = 0.0f64;
    for (idx, c) in v.iter().enumerate() {
        let p = c.abs_s().powi(2);
        let g = det_irrep(pg, idx);
        total += p;
        weights[usize::from(g.index())] += p;
        for (h, ch) in chars.iter_mut().enumerate() {
            *ch += g.character(Irrep::new(h as u8)) * p;
        }
    }
    assert!(total > 1e-6, "dense oracle: anti-vacuity, the member is ~0");
    (
        chars.iter().map(|c| c / total).collect(),
        weights.iter().map(|w| w / total).collect(),
    )
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn characters(w: &[f64]) -> Vec<f64> {
    (0..w.len())
        .map(|h| {
            w.iter()
                .enumerate()
                .map(|(g, &wg)| Irrep::new(g as u8).character(Irrep::new(h as u8)) * wg)
                .sum()
        })
        .collect()
}

fn assert_partition(w: &[f64], label: &str) {
    let sum: f64 = w.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-14,
        "{label}: weights sum to {sum} (|Δ| = {:e})",
        (sum - 1.0).abs()
    );
    assert!(
        w.iter().all(|&v| (-1e-14..=1.0 + 1e-14).contains(&v)),
        "{label}: a weight outside [0, 1]: {w:?}"
    );
}

fn assert_members_match<T: Scalar>(x: &Su2TtVector<T>, w: &[f64], label: &str) -> f64 {
    let pg = fx_pg();
    let chars = characters(w);
    let mut worst = 0.0f64;
    for m in members(x.space.two_s) {
        let (dc, dw) = dense_decomposition(&expand(x, m).to_dense(), pg);
        let (ec, ew) = (max_abs_diff(&chars, &dc), max_abs_diff(w, &dw));
        assert!(
            ec <= 1e-13,
            "{label}, 2Sz = {m}: characters {chars:?} vs dense {dc:?}"
        );
        assert!(
            ew <= 1e-13,
            "{label}, 2Sz = {m}: weights {w:?} vs dense {dw:?}"
        );
        worst = worst.max(ec).max(ew);
    }
    worst
}

fn assert_matches_abelian<T: Scalar>(x: &Su2TtVector<T>, w: &[f64], label: &str) -> f64 {
    let pg = fx_pg();
    let mut worst = 0.0f64;
    for m in members(x.space.two_s) {
        let wa = crate::ops::irrep_weights(&expand(x, m), pg);
        let e = max_abs_diff(w, &wa);
        assert!(
            e <= 1e-14,
            "{label}, 2Sz = {m}: SU(2) weights {w:?} vs abelian member {wa:?}"
        );
        worst = worst.max(e);
    }
    worst
}

#[test]
fn siw_eigenstate_reads_its_irrep() {
    let pg = fx_pg();
    let mut saw_minus = false;
    for (n, two_s) in NS {
        for target in reachable_targets(n, two_s) {
            let g = target.index();
            let x: Su2TtVector<f64> = typed_state(n, two_s, target, 0x2A0 + u64::from(g));
            let f = x.forget_point_group();
            let want_c: Vec<f64> = (0..pg.order())
                .map(|h| target.character(Irrep::new(h)))
                .collect();
            let want_w: Vec<f64> = (0..pg.order())
                .map(|h| if h == g { 1.0 } else { 0.0 })
                .collect();
            saw_minus |= want_c.contains(&-1.0);
            for (what, w) in [
                ("embedded", irrep_weights(&f, pg)),
                ("typed", irrep_weights(&x, pg)),
            ] {
                let label = format!("(N, 2S) = ({n}, {two_s}) target {g} {what}");
                assert_partition(&w, &label);
                let (ec, ew) = (
                    max_abs_diff(&characters(&w), &want_c),
                    max_abs_diff(&w, &want_w),
                );
                println!("siw {label}: |Δχ| = {ec:e}, |Δw| = {ew:e}");
                assert!(ew <= 1e-14, "{label}: weights {w:?}");
                assert!(ec <= 1e-14, "{label}: characters {:?}", characters(&w));
            }
        }
    }
    assert!(saw_minus, "anti-vacuity: no target had a −1 character");
}

fn superposition(n: usize, two_s: usize) -> (Su2TtVector<f64>, Vec<f64>) {
    let ts = reachable_targets(n, two_s);
    let (g1, g2) = (ts[0], ts[1]);
    let x1: Su2TtVector<f64> = typed_state(n, two_s, g1, 0x2B1);
    let x2: Su2TtVector<f64> = typed_state(n, two_s, g2, 0x2B2);
    let (a, b) = (0.6, -1.3);
    let (n1, n2) = (x1.norm().powi(2), x2.norm().powi(2));
    let s = &(&x1.forget_point_group() * a) + &(&x2.forget_point_group() * b);
    let den = a * a * n1 + b * b * n2;
    let mut closed = vec![0.0; 4];
    closed[usize::from(g1.index())] = a * a * n1 / den;
    closed[usize::from(g2.index())] = b * b * n2 / den;
    assert!(
        closed.iter().filter(|&&w| w > 0.05).count() == 2,
        "anti-vacuity: one component is negligible: {closed:?}"
    );
    (s, closed)
}

fn mixture(n: usize, two_s: usize) -> Su2TtVector<f64> {
    randomized_su2_state(Su2Space::new(D, n, two_s), 1, 2, &mut seeded(0x2B3))
}

#[test]
fn siw_superposition_and_mixture_match_the_dense_projector() {
    let pg = fx_pg();
    for (n, two_s) in [(3, 1), (4, 2)] {
        let (s, closed) = superposition(n, two_s);
        let m = mixture(n, two_s);
        for (what, x, closed) in [("superposition", &s, Some(&closed)), ("mixture", &m, None)] {
            let label = format!("(N, 2S) = ({n}, {two_s}) {what}");
            let w = irrep_weights(x, pg);
            assert_partition(&w, &label);
            let worst = assert_members_match(x, &w, &label);
            println!("siw {label}: weights {w:?}; worst member |Δ| = {worst:e}");
            if let Some(c) = closed {
                let e = max_abs_diff(&w, c);
                assert!(e <= 1e-13, "{label}: weights {w:?} vs closed {c:?}");
            } else {
                let mixed = w.iter().filter(|&&v| v > 1e-3).count();
                assert!(
                    mixed >= 3,
                    "{label}: anti-vacuity: the random state is not a mixture: {w:?}"
                );
            }
        }
    }
}

#[test]
fn siw_matches_the_abelian_readout_on_every_member() {
    let pg = fx_pg();
    let mut worst = 0.0f64;
    let mut cases = 0usize;
    for (n, two_s) in NS {
        let mut states: Vec<(String, Su2TtVector<f64>)> = Vec::new();
        for target in reachable_targets(n, two_s) {
            let g = target.index();
            let x: Su2TtVector<f64> = typed_state(n, two_s, target, 0x2E0 + u64::from(g));
            states.push((format!("target {g} embedded"), x.forget_point_group()));
            states.push((format!("target {g} typed"), x));
        }
        if two_s >= 1 {
            states.push(("superposition".into(), superposition(n, two_s).0));
            states.push(("mixture".into(), mixture(n, two_s)));
        }
        for (what, x) in &states {
            let label = format!("(N, 2S) = ({n}, {two_s}) {what}");
            let w = irrep_weights(x, pg);
            worst = worst.max(assert_matches_abelian(x, &w, &label));
            assert_eq!(
                irrep_weights(x, PointGroup::TRIVIAL),
                vec![1.0],
                "{label}: the trivial group must return [1.0], as the abelian readout does"
            );
            cases += 1;
        }
    }
    println!("siw cross-layer: {cases} states, worst |Δ| = {worst:e}");
    assert!(cases >= 12, "anti-vacuity: only {cases} states compared");
}

#[test]
fn siw_c64_superposition() {
    let pg = fx_pg();
    let (n, two_s) = (3, 1);
    let ts = reachable_targets(n, two_s);
    let (g1, g2) = (ts[ts.len() - 1], ts[0]);
    let unit = |x: Su2TtVector<f64>| {
        let nx = x.norm();
        (&x * (1.0 / nx)).forget_point_group().to_c64()
    };
    let x1 = unit(typed_state(n, two_s, g1, 0x2C1));
    let x2 = unit(typed_state(n, two_s, g2, 0x2C2));
    let (al, be) = (C64::new(0.6, 0.3), C64::new(-0.2, 1.1));
    let s = &(al * &x1) + &(be * &x2);
    let (n1, n2) = (x1.norm().powi(2), x2.norm().powi(2));
    let den = al.norm_sqr() * n1 + be.norm_sqr() * n2;
    let mut closed = vec![0.0; 4];
    closed[usize::from(g1.index())] = al.norm_sqr() * n1 / den;
    closed[usize::from(g2.index())] = be.norm_sqr() * n2 / den;
    assert!(
        closed.iter().filter(|&&w| w > 0.05).count() == 2,
        "anti-vacuity: one component is negligible: {closed:?}"
    );

    let w = irrep_weights(&s, pg);
    assert_partition(&w, "c64");
    let worst = assert_members_match(&s, &w, "c64");
    let worst_ab = assert_matches_abelian(&s, &w, "c64");
    let e = max_abs_diff(&w, &closed);
    println!(
        "siw c64: worst member |Δ| = {worst:e}, abelian |Δ| = {worst_ab:e}, closed form |Δ| = {e:e}"
    );
    assert!(e <= 1e-13, "c64: weights {w:?} vs closed {closed:?}");
}
