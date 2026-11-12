// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use super::MultiTt;
use crate::scalar::Scalar;
use crate::su2::expand::expand;
use crate::su2::qn::Su2Space;
use crate::su2::vector::Su2TtVector;
use crate::test_util::{dense_dot, randomized_su2_state, seeded};

fn container(space: Su2Space, pos: usize, n: usize, seed: u64) -> MultiTt<f64> {
    let mut rng = seeded(seed);
    let mut base: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    base.move_core(pos, true);
    let c0 = base.core(pos).clone();
    let mut centers = vec![c0.clone()];
    for _ in 1..n {
        let mut c = c0.clone();
        c.for_each_coeff_mut(|z| *z = f64::sample_standard(&mut rng));
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

fn gram_matches_dense(a: &MultiTt<f64>, b: &MultiTt<f64>) {
    let g = a.gram(b);
    assert_eq!((g.nrows(), g.ncols()), (a.nstates(), b.nstates()));
    let two_s = a.base.space.two_s as i64;
    for two_sz in (-two_s..=two_s).step_by(2) {
        let dense = |m: &MultiTt<f64>| -> Vec<Vec<f64>> {
            m.states()
                .iter()
                .map(|x| expand(x, two_sz).to_dense())
                .collect()
        };
        let (ad, bd) = (dense(a), dense(b));
        for i in 0..a.nstates() {
            for j in 0..b.nstates() {
                let want = dense_dot(&ad[i], &bd[j]);
                assert!(
                    (*g.get(i, j) - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "2Sz = {two_sz}: G[{i},{j}] = {}, dense {want}",
                    g.get(i, j)
                );
            }
        }
    }
}

fn every_ordering(space: Su2Space) {
    let d = space.d;
    let a = container(space, 1, 3, 1);
    let b = container(space, d - 2, 2, 2);
    gram_matches_dense(&a, &b);
    gram_matches_dense(&b, &a);
    gram_matches_dense(&a, &a);
    let e0 = container(space, 0, 2, 3);
    let el = container(space, d - 1, 3, 4);
    gram_matches_dense(&e0, &el);
    gram_matches_dense(&el, &el);
}

#[test]
fn su2_multitt_gram_matches_dense_doublet() {
    every_ordering(Su2Space::new(5, 3, 1));
}

#[test]
fn su2_multitt_gram_matches_dense_triplet() {
    every_ordering(Su2Space::new(5, 4, 2));
}

#[test]
fn su2_multitt_gram_matches_dense_singlet_control() {
    every_ordering(Su2Space::new(4, 4, 0));
}

#[test]
fn su2_multitt_gram_matches_dense_at_nontrivial_point_group() {
    let (_pg, space) = crate::test_util::su2_pg_space(5, 3, 1);
    every_ordering(space);
}
