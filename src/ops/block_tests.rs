// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use rand::RngExt;

use crate::qn::{QnSpace, RankGrid, Sector};
use crate::scalar::Scalar;
use crate::test_util::{dense_dot, pg3_space_d6, seeded};
use crate::{C64, TtVector, dot, dot_block, tt_randn};

pub(crate) fn open_block<T: Scalar>(
    space: QnSpace,
    hi: usize,
    rho: usize,
    seed: u64,
) -> TtVector<T> {
    let mut rng = seeded(seed);
    let mut pick = seeded(seed ^ 0x5EED);
    open_block_with(space, rho, &mut rng, |_, _| pick.random_range(1..=hi))
}

pub(crate) fn open_block_uniform<T: Scalar>(
    space: QnSpace,
    r: usize,
    rho: usize,
    seed: u64,
) -> TtVector<T> {
    open_block_with(space, rho, &mut seeded(seed), |_, _| r)
}

fn open_block_with<T: Scalar>(
    space: QnSpace,
    rho: usize,
    rng: &mut rand_xoshiro::Xoshiro256PlusPlus,
    mut interior: impl FnMut(usize, Sector) -> usize,
) -> TtVector<T> {
    let d = space.d;
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut g = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                g[s] = if b == 0 {
                    1
                } else if b == d {
                    if s == last { rho } else { 0 }
                } else {
                    interior(b, s)
                };
            }
            g
        })
        .collect();
    let x = tt_randn(space, &ranks, rng);
    assert_eq!(x.rank(d, last), rho);
    x
}

fn dense_columns<T: Scalar>(x: &TtVector<T>) -> Vec<Vec<T>> {
    (0..x.boundary_width())
        .map(|j| x.column(j).to_dense())
        .collect()
}

fn gram_matches_dense<T: Scalar>(space: QnSpace, tol: f64) {
    let x: TtVector<T> = open_block(space, 3, 3, 11);
    let y: TtVector<T> = open_block(space, 4, 2, 12);
    let g = dot_block(&x, &y);
    assert_eq!((g.nrows(), g.ncols()), (3, 2));
    let (xd, yd) = (dense_columns(&x), dense_columns(&y));
    for i in 0..3 {
        for j in 0..2 {
            let want = dense_dot(&xd[i], &yd[j]);
            assert!(
                (*g.get(i, j) - want).abs_s() <= tol * want.abs_s().max(1.0),
                "G[{i},{j}] = {:?}, dense {:?}",
                g.get(i, j),
                want
            );
            let via_dot = dot(&x.column(i), &y.column(j));
            assert!((*g.get(i, j) - via_dot).abs_s() <= tol * want.abs_s().max(1.0));
        }
    }
    let gt = dot_block(&y, &x);
    for i in 0..3 {
        for j in 0..2 {
            assert!((*gt.get(j, i) - g.get(i, j).conj_s()).abs_s() <= tol);
        }
    }
}

#[test]
fn dot_block_matches_the_dense_gram_f64() {
    gram_matches_dense::<f64>(QnSpace::new(4, 2, 1), 1e-12);
}

#[test]
fn dot_block_matches_the_dense_gram_c64() {
    gram_matches_dense::<C64>(QnSpace::new(4, 2, 1), 1e-12);
}

#[test]
fn dot_block_matches_the_dense_gram_at_nontrivial_point_group() {
    let (_pg, space) = pg3_space_d6();
    gram_matches_dense::<f64>(space, 1e-12);
}

#[test]
fn dot_block_at_unit_boundary_is_dot_bit_for_bit() {
    let space = QnSpace::new(5, 2, 2);
    let x: TtVector<f64> = open_block(space, 3, 1, 21);
    let y: TtVector<f64> = open_block(space, 3, 1, 22);
    let g = dot_block(&x, &y);
    assert_eq!((g.nrows(), g.ncols()), (1, 1));
    assert_eq!(*g.get(0, 0), dot(&x, &y));
}

#[test]
fn norm_of_an_open_block_is_sqrt_trace_of_the_gram() {
    let space = QnSpace::new(4, 2, 1);
    let x: TtVector<f64> = open_block(space, 3, 3, 31);
    let cols = dense_columns(&x);
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
    assert!((l.norm() - want).abs() <= 1e-12 * want);
    let mut r = x.clone();
    r.right_orthogonalize(false);
    assert_eq!(r.ortho, Some(0));
    assert!((r.norm() - want).abs() <= 1e-12 * want);
    for (j, c) in cols.iter().enumerate() {
        for t in [&l, &r] {
            let got = t.column(j).to_dense();
            let err: f64 = got
                .iter()
                .zip(c)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(err <= 1e-12, "column {j} moved by {err}");
        }
    }
}
