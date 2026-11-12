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
use crate::qn::QnSpace;
use crate::scalar::Scalar;
use crate::test_util::{randomized_state, seeded};
use crate::vector::TtVector;

const D: usize = 4;
const NUP: usize = 2;
const NDN: usize = 1;

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
    )
}

fn typed_state(target: Irrep, seed: u64) -> TtVector<f64> {
    let space = QnSpace::with_point_group(D, NUP, NDN, fx_pg(), target);
    randomized_state(space, 1, 2, &mut seeded(seed))
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
    assert!(total > 1e-6, "dense oracle: anti-vacuity, the state is ~0");
    (
        chars.iter().map(|c| c / total).collect(),
        weights.iter().map(|w| w / total).collect(),
    )
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

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
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

#[test]
fn oiw_eigenstate_reads_its_irrep() {
    let pg = fx_pg();
    let mut saw_minus = false;
    for g in 0..pg.order() {
        let target = Irrep::new(g);
        let x = typed_state(target, 0x1A0 + u64::from(g));
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
            let label = format!("target {g} {what}");
            assert_partition(&w, &label);
            let (ec, ew) = (
                max_abs_diff(&characters(&w), &want_c),
                max_abs_diff(&w, &want_w),
            );
            println!("oiw {label}: |Δχ| = {ec:e}, |Δw| = {ew:e}");
            assert!(ew <= 1e-14, "{label}: weights {w:?}");
            assert!(ec <= 1e-14, "{label}: characters {:?}", characters(&w));
        }
    }
    assert!(saw_minus, "anti-vacuity: no target had a −1 character");
}

#[test]
fn oiw_superposition_and_mixture_match_the_dense_projector() {
    let pg = fx_pg();
    let (g1, g2) = (Irrep::new(1), Irrep::new(2));
    let x1 = typed_state(g1, 0x1B1);
    let x2 = typed_state(g2, 0x1B2);
    let (a, b) = (0.6, -1.3);
    let (n1, n2) = (x1.norm().powi(2), x2.norm().powi(2));
    let s = &(&x1.forget_point_group() * a) + &(&x2.forget_point_group() * b);
    let den = a * a * n1 + b * b * n2;
    let mut closed = vec![0.0; 4];
    closed[1] = a * a * n1 / den;
    closed[2] = b * b * n2 / den;
    assert!(
        closed[1].min(closed[2]) > 0.05,
        "anti-vacuity: one component is negligible: {closed:?}"
    );

    let untyped = QnSpace::new(D, NUP, NDN);
    let m: TtVector<f64> = randomized_state(untyped, 1, 2, &mut seeded(0x1B3));

    for (label, x, closed) in [("superposition", &s, Some(&closed)), ("mixture", &m, None)] {
        let w = irrep_weights(x, pg);
        assert_partition(&w, label);
        let (dc, dw) = dense_decomposition(&x.to_dense(), pg);
        let (ec, ew) = (max_abs_diff(&characters(&w), &dc), max_abs_diff(&w, &dw));
        println!("oiw {label}: weights {w:?}; dense |Δχ| = {ec:e}, |Δw| = {ew:e}");
        assert!(ew <= 1e-13, "{label}: weights {w:?} vs dense {dw:?}");
        assert!(
            ec <= 1e-13,
            "{label}: characters {:?} vs dense {dc:?}",
            characters(&w)
        );
        if let Some(c) = closed {
            let e = max_abs_diff(&w, c);
            assert!(e <= 1e-13, "{label}: weights {w:?} vs closed form {c:?}");
        } else {
            let mixed = w.iter().filter(|&&v| v > 1e-3).count();
            assert!(
                mixed >= 3,
                "{label}: anti-vacuity: the random state is not a mixture: {w:?}"
            );
        }
    }
}

#[test]
fn oiw_c64_superposition() {
    let pg = fx_pg();
    let (g1, g3) = (Irrep::new(1), Irrep::new(3));
    let unit = |x: TtVector<f64>| {
        let n = x.norm();
        (&x * (1.0 / n)).forget_point_group().to_c64()
    };
    let x1 = unit(typed_state(g1, 0x1C1));
    let x3 = unit(typed_state(g3, 0x1C3));
    let (al, be) = (C64::new(0.6, 0.3), C64::new(-0.2, 1.1));
    let s = &(al * &x1) + &(be * &x3);
    let (n1, n3) = (x1.norm().powi(2), x3.norm().powi(2));
    let den = al.norm_sqr() * n1 + be.norm_sqr() * n3;
    let mut closed = vec![0.0; 4];
    closed[1] = al.norm_sqr() * n1 / den;
    closed[3] = be.norm_sqr() * n3 / den;
    assert!(
        closed[1].min(closed[3]) > 0.05,
        "anti-vacuity: one component is negligible: {closed:?}"
    );

    let w = irrep_weights(&s, pg);
    assert_partition(&w, "c64");
    let (dc, dw) = dense_decomposition(&s.to_dense(), pg);
    let (ec, ew, ef) = (
        max_abs_diff(&characters(&w), &dc),
        max_abs_diff(&w, &dw),
        max_abs_diff(&w, &closed),
    );
    println!("oiw c64: dense |Δχ| = {ec:e}, |Δw| = {ew:e}; closed form |Δ| = {ef:e}");
    assert!(ew <= 1e-13, "c64: weights {w:?} vs dense {dw:?}");
    assert!(
        ec <= 1e-13,
        "c64: characters {:?} vs dense {dc:?}",
        characters(&w)
    );
    assert!(ef <= 1e-13, "c64: weights {w:?} vs closed {closed:?}");
}
