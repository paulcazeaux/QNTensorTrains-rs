// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use super::MultiTt;
use crate::qn::QnSpace;
use crate::scalar::Scalar;
use crate::test_util::{dense_dot, randomized_state, seeded};
use crate::{C64, TtVector};

fn container<T: Scalar>(space: QnSpace, pos: usize, n: usize, seed: u64) -> MultiTt<T> {
    container_gauged(space, pos, n, seed, true)
}

fn container_gauged<T: Scalar>(
    space: QnSpace,
    pos: usize,
    n: usize,
    seed: u64,
    keep_rank: bool,
) -> MultiTt<T> {
    let mut rng = seeded(seed);
    let mut base: TtVector<T> = randomized_state(space, 2, 3, &mut rng);
    base.move_core(pos, keep_rank);
    let c0 = base.core(pos).clone();
    let mut centers = vec![c0.clone()];
    for _ in 1..n {
        let mut c = c0.clone();
        c.for_each_coeff_mut(|z| *z = T::sample_standard(&mut rng));
        centers.push(c);
    }
    let m = MultiTt {
        base,
        centers,
        weights: vec![1.0 / n as f64; n],
        core_position: pos,
    };
    m.check();
    m
}

fn gram_matches_dense<T: Scalar>(a: &MultiTt<T>, b: &MultiTt<T>) {
    let g = a.gram(b);
    assert_eq!((g.nrows(), g.ncols()), (a.nstates(), b.nstates()));
    let ad: Vec<Vec<T>> = a.states().iter().map(TtVector::to_dense).collect();
    let bd: Vec<Vec<T>> = b.states().iter().map(TtVector::to_dense).collect();
    for i in 0..a.nstates() {
        for j in 0..b.nstates() {
            let want = dense_dot(&ad[i], &bd[j]);
            assert!(
                (*g.get(i, j) - want).abs_s() <= 1e-12 * want.abs_s().max(1.0),
                "G[{i},{j}] = {:?}, dense {:?}",
                g.get(i, j),
                want
            );
        }
    }
}

#[test]
fn multitt_gram_matches_dense_for_every_ordering_of_the_open_sites() {
    let space = QnSpace::new(5, 2, 2);
    let a: MultiTt<f64> = container(space, 1, 3, 1);
    let b: MultiTt<f64> = container(space, 3, 2, 2);
    gram_matches_dense(&a, &b);
    gram_matches_dense(&b, &a);
    gram_matches_dense(&a, &a);
    let e0: MultiTt<f64> = container(space, 0, 2, 3);
    let e4: MultiTt<f64> = container(space, 4, 4, 4);
    gram_matches_dense(&e0, &e4);
    gram_matches_dense(&e4, &e4);
}

#[test]
fn multitt_gram_matches_dense_c64() {
    let space = QnSpace::new(4, 2, 1);
    let a: MultiTt<C64> = container(space, 2, 3, 5);
    let b: MultiTt<C64> = container(space, 1, 2, 6);
    gram_matches_dense(&a, &b);
    gram_matches_dense(&a, &a);
}

#[test]
fn multitt_self_gram_is_the_centre_gram() {
    let space = QnSpace::new(5, 2, 2);
    let a: MultiTt<f64> = container_gauged(space, 2, 3, 7, false);
    let g = a.gram(&a);
    for i in 0..3 {
        for j in 0..3 {
            let (mut u, mut v) = (Vec::new(), Vec::new());
            a.centers[i].for_each_coeff(|z| u.push(z));
            a.centers[j].for_each_coeff(|z| v.push(z));
            let want = dense_dot(&u, &v);
            assert!((*g.get(i, j) - want).abs() <= 1e-12 * want.abs().max(1.0));
        }
    }
}

#[test]
fn multitt_gram_matches_dense_at_nontrivial_point_group() {
    let (_pg, space) = crate::test_util::pg3_space_d6();
    let a: MultiTt<f64> = container(space, 1, 3, 8);
    let b: MultiTt<f64> = container(space, 4, 2, 9);
    gram_matches_dense(&a, &b);
    gram_matches_dense(&b, &a);
    gram_matches_dense(&a, &a);
}
