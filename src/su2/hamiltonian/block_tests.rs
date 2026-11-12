// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use crate::su2::expand::expand;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient, xt_h_y, xt_h_y_block};
use crate::su2::ops::block_tests::su2_open_block;
use crate::su2::qn::Su2Sector;
use crate::su2::qn::Su2Space;
use crate::su2::vector::Su2TtVector;
use crate::test_util::{c2v_pg, dense_dot, generic_su2_h, pg_symmetric_tv};

fn operator_gram_matches_dense(space: Su2Space) {
    operator_gram_matches_dense_with(space, &generic_su2_h::<f64>(space));
}

fn operator_gram_matches_dense_with(space: Su2Space, h: &Su2SparseHamiltonian<f64>) {
    let x: Su2TtVector<f64> = su2_open_block(space, 3, 3, 41);
    let y: Su2TtVector<f64> = su2_open_block(space, 4, 2, 42);
    let g = xt_h_y_block(&x, h, &y);
    assert_eq!((g.nrows(), g.ncols()), (3, 2));
    let hy: Vec<Su2TtVector<f64>> = (0..2).map(|j| h.apply(&y.column(j))).collect();
    let two_s = space.two_s as i64;
    let mut scale = 0.0f64;
    for two_sz in (-two_s..=two_s).step_by(2) {
        for i in 0..3 {
            let xd = expand(&x.column(i), two_sz).to_dense();
            for j in 0..2 {
                let want = dense_dot(&xd, &expand(&hy[j], two_sz).to_dense());
                scale = scale.max(want.abs());
                assert!(
                    (*g.get(i, j) - want).abs() <= 1e-11 * want.abs().max(1.0),
                    "2Sz = {two_sz}: G[{i},{j}] = {}, dense {want}",
                    g.get(i, j)
                );
            }
        }
    }
    assert!(scale > 1e-3, "degenerate fixture");
}

#[test]
fn su2_xt_h_y_block_matches_the_dense_operator_gram_doublet() {
    operator_gram_matches_dense(Su2Space::new(5, 3, 1));
}

#[test]
fn su2_xt_h_y_block_matches_the_dense_operator_gram_triplet() {
    operator_gram_matches_dense(Su2Space::new(5, 4, 2));
}

#[test]
fn su2_xt_h_y_block_matches_the_dense_operator_gram_at_nontrivial_point_group() {
    let (d, n, two_s) = (5, 3, 1);
    let pg = c2v_pg(d);
    let target = (1..pg.order())
        .map(crate::Irrep::new)
        .find(|&ir| {
            let sp = Su2Space::with_point_group(d, n, two_s, pg, ir);
            crate::su2::qn::max_ranks(&sp)[d][Su2Sector::with_irrep(n, two_s, ir)] > 0
        })
        .expect("a reachable non-identity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let (t, v) = pg_symmetric_tv(&pg, d, 0xB10C);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    operator_gram_matches_dense_with(space, &h);
}

#[test]
fn su2_xt_h_y_block_matches_the_dense_operator_gram_singlet_control() {
    operator_gram_matches_dense(Su2Space::new(4, 4, 0));
}

#[test]
fn su2_xt_h_y_block_at_unit_boundary_is_xt_h_y_bit_for_bit() {
    let space = Su2Space::new(5, 3, 1);
    let h = generic_su2_h::<f64>(space);
    let x: Su2TtVector<f64> = su2_open_block(space, 3, 1, 43);
    let y: Su2TtVector<f64> = su2_open_block(space, 3, 1, 44);
    let g = xt_h_y_block(&x, &h, &y);
    assert_eq!((g.nrows(), g.ncols()), (1, 1));
    assert_eq!(*g.get(0, 0), xt_h_y(&x, &h, &y));
}

#[test]
#[should_panic(expected = "xt_h_y_block")]
fn su2_xt_h_y_refuses_an_open_block() {
    let space = Su2Space::new(5, 3, 1);
    let h = generic_su2_h::<f64>(space);
    let x: Su2TtVector<f64> = su2_open_block(space, 2, 2, 45);
    let _ = xt_h_y(&x, &h, &x);
}

#[test]
#[should_panic(expected = "xt_h_y_block")]
fn su2_rayleigh_quotient_refuses_an_open_block() {
    let space = Su2Space::new(5, 3, 1);
    let h = generic_su2_h::<f64>(space);
    let x: Su2TtVector<f64> = su2_open_block(space, 2, 2, 46);
    let _ = rayleigh_quotient(&h, &x);
}
