// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use rand::RngExt;

use crate::scalar::Scalar;
use crate::su2::expand::expand;
use crate::su2::ops::{dot, dot_block};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::vector::{Su2TtVector, tt_randn};
use crate::test_util::{dense_dot, seeded};
use crate::{C64, PointGroup};

pub(crate) fn su2_open_block_with<T: Scalar>(
    space: Su2Space,
    rho: usize,
    seed: u64,
    mut interior: impl FnMut() -> usize,
) -> Su2TtVector<T> {
    let d = space.d;
    let last = Su2Sector::with_irrep(space.n, space.two_s, space.target);
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut g = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                g[s] = if b == 0 {
                    1
                } else if b == d {
                    if s == last { rho } else { 0 }
                } else {
                    interior()
                };
            }
            g
        })
        .collect();
    let x = tt_randn(space, &ranks, &mut seeded(seed));
    assert_eq!(x.rank(d, last), rho);
    x
}

pub(crate) fn su2_open_block<T: Scalar>(
    space: Su2Space,
    hi: usize,
    rho: usize,
    seed: u64,
) -> Su2TtVector<T> {
    let mut pick = seeded(seed ^ 0x5EED);
    su2_open_block_with(space, rho, seed, || pick.random_range(1..=hi))
}

pub(crate) fn su2_dense_columns<T: Scalar>(x: &Su2TtVector<T>, two_sz: i64) -> Vec<Vec<T>> {
    (0..x.boundary_width())
        .map(|j| expand(&x.column(j), two_sz).to_dense())
        .collect()
}

fn gram_matches_dense<T: Scalar>(space: Su2Space) {
    let x: Su2TtVector<T> = su2_open_block(space, 3, 3, 11);
    let y: Su2TtVector<T> = su2_open_block(space, 4, 2, 12);
    let g = dot_block(&x, &y);
    assert_eq!((g.nrows(), g.ncols()), (3, 2));
    let two_s = space.two_s as i64;
    for two_sz in (-two_s..=two_s).step_by(2) {
        let (xd, yd) = (su2_dense_columns(&x, two_sz), su2_dense_columns(&y, two_sz));
        for i in 0..3 {
            for j in 0..2 {
                let want = dense_dot(&xd[i], &yd[j]);
                assert!(
                    (*g.get(i, j) - want).abs_s() <= 1e-12 * want.abs_s().max(1.0),
                    "2Sz = {two_sz}: G[{i},{j}] = {:?}, dense {:?}",
                    g.get(i, j),
                    want
                );
            }
        }
    }
    for i in 0..3 {
        for j in 0..2 {
            let via_dot = dot(&x.column(i), &y.column(j));
            assert!((*g.get(i, j) - via_dot).abs_s() <= 1e-12 * via_dot.abs_s().max(1.0));
        }
    }
    let gt = dot_block(&y, &x);
    for i in 0..3 {
        for j in 0..2 {
            assert!((*gt.get(j, i) - g.get(i, j).conj_s()).abs_s() <= 1e-12);
        }
    }
}

#[test]
fn su2_dot_block_matches_the_dense_gram_doublet() {
    gram_matches_dense::<f64>(Su2Space::new(5, 3, 1));
}

#[test]
fn su2_dot_block_matches_the_dense_gram_triplet_c64() {
    gram_matches_dense::<C64>(Su2Space::new(4, 4, 2));
}

#[test]
fn su2_dot_block_matches_the_dense_gram_quartet() {
    gram_matches_dense::<f64>(Su2Space::new(5, 5, 3));
}

#[test]
fn su2_dot_block_matches_the_dense_gram_singlet_control() {
    gram_matches_dense::<f64>(Su2Space::new(4, 4, 0));
}

#[test]
fn su2_dot_block_matches_the_dense_gram_at_nontrivial_point_group() {
    let pg: PointGroup = crate::test_util::c2v_pg(5);
    let target = (0..4u8)
        .map(crate::Irrep::new)
        .find(|&t| {
            let sp = Su2Space::with_point_group(5, 3, 1, pg, t);
            crate::su2::qn::max_ranks(&sp)[5][Su2Sector::with_irrep(3, 1, t)] > 0
        })
        .expect("some irrep is reachable");
    gram_matches_dense::<f64>(Su2Space::with_point_group(5, 3, 1, pg, target));
}

#[test]
fn su2_dot_block_at_unit_boundary_is_dot_bit_for_bit() {
    let space = Su2Space::new(5, 3, 1);
    let x: Su2TtVector<f64> = su2_open_block(space, 3, 1, 21);
    let y: Su2TtVector<f64> = su2_open_block(space, 3, 1, 22);
    let g = dot_block(&x, &y);
    assert_eq!((g.nrows(), g.ncols()), (1, 1));
    assert_eq!(*g.get(0, 0), dot(&x, &y));
}

#[test]
fn su2_norm_of_an_open_block_is_sqrt_trace_of_the_gram() {
    let space = Su2Space::new(5, 3, 1);
    let x: Su2TtVector<f64> = su2_open_block(space, 3, 3, 31);
    let cols = su2_dense_columns(&x, 1);
    let want = cols.iter().map(|c| dense_dot(c, c)).sum::<f64>().sqrt();
    let g = dot_block(&x, &x);
    let mut entry_sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            entry_sum += *g.get(i, j);
        }
    }
    let wrong = entry_sum.abs().sqrt();
    assert!(
        (wrong - want).abs() > 1e-3 * want,
        "fixture does not discriminate: √tr = {want}, √Σ = {wrong}"
    );
    assert!(
        (x.norm() - want).abs() <= 1e-12 * want,
        "{} vs {want}",
        x.norm()
    );

    let mut l = x.clone();
    l.left_orthogonalize(false);
    assert_eq!(l.ortho, Some(space.d - 1));
    assert!(
        (l.norm() - want).abs() <= 1e-12 * want,
        "left centre: {} vs {want}",
        l.norm()
    );
    let mut r = x.clone();
    r.right_orthogonalize(false);
    assert_eq!(r.ortho, Some(0));
    assert!(
        (r.norm() - want).abs() <= 1e-12 * want,
        "right centre: {} vs {want}",
        r.norm()
    );
    for (j, c) in cols.iter().enumerate() {
        for t in [&l, &r] {
            let got = expand(&t.column(j), 1).to_dense();
            let err: f64 = got
                .iter()
                .zip(c)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(err <= 1e-12, "column {j} moved by {err}");
        }
    }
}
