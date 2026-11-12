// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use crate::hamiltonian::{rayleigh_quotient, xt_h_y, xt_h_y_block};
use crate::ops::block_tests::open_block;
use crate::qn::QnSpace;
use crate::scalar::Scalar;
use crate::test_util::{dense_dot, generic_truncated_h, pg_symmetric_tv, pg3_space_d6};
use crate::{C64, SparseHamiltonian, TtVector};

fn operator_gram_matches_dense<T: Scalar>(space: QnSpace) {
    operator_gram_matches_dense_with(space, &generic_truncated_h::<T>(space));
}

fn operator_gram_matches_dense_with<T: Scalar>(space: QnSpace, h: &SparseHamiltonian<T>) {
    let x: TtVector<T> = open_block(space, 3, 3, 41);
    let y: TtVector<T> = open_block(space, 4, 2, 42);
    let g = xt_h_y_block(&x, h, &y);
    assert_eq!((g.nrows(), g.ncols()), (3, 2));
    let xd: Vec<Vec<T>> = (0..3).map(|i| x.column(i).to_dense()).collect();
    let hyd: Vec<Vec<T>> = (0..2).map(|j| h.apply(&y.column(j)).to_dense()).collect();
    let mut off_diag_scale = 0.0f64;
    for i in 0..3 {
        for j in 0..2 {
            let want = dense_dot(&xd[i], &hyd[j]);
            off_diag_scale = off_diag_scale.max(want.abs_s());
            assert!(
                (*g.get(i, j) - want).abs_s() <= 1e-11 * want.abs_s().max(1.0),
                "G[{i},{j}] = {:?}, dense {:?}",
                g.get(i, j),
                want
            );
        }
    }
    assert!(off_diag_scale > 1e-3, "degenerate fixture");
}

#[test]
fn xt_h_y_block_matches_the_dense_operator_gram_f64() {
    operator_gram_matches_dense::<f64>(QnSpace::new(5, 2, 2));
}

#[test]
fn xt_h_y_block_matches_the_dense_operator_gram_c64() {
    operator_gram_matches_dense::<C64>(QnSpace::new(4, 2, 1));
}

#[test]
fn xt_h_y_block_matches_the_dense_operator_gram_at_nontrivial_point_group() {
    let (pg, space) = pg3_space_d6();
    let (t, v) = pg_symmetric_tv(&pg, space.d, 0xB10C);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    operator_gram_matches_dense_with(space, &h);
}

#[test]
fn xt_h_y_block_at_unit_boundary_is_xt_h_y_bit_for_bit() {
    let space = QnSpace::new(5, 2, 2);
    let h = generic_truncated_h::<f64>(space);
    let x: TtVector<f64> = open_block(space, 3, 1, 43);
    let y: TtVector<f64> = open_block(space, 3, 1, 44);
    let g = xt_h_y_block(&x, &h, &y);
    assert_eq!((g.nrows(), g.ncols()), (1, 1));
    assert_eq!(*g.get(0, 0), xt_h_y(&x, &h, &y));
}

#[test]
#[should_panic(expected = "xt_h_y_block")]
fn xt_h_y_refuses_an_open_block() {
    let space = QnSpace::new(4, 2, 1);
    let h = generic_truncated_h::<f64>(space);
    let x: TtVector<f64> = open_block(space, 2, 2, 45);
    let _ = xt_h_y(&x, &h, &x);
}

#[test]
#[should_panic(expected = "xt_h_y_block")]
fn rayleigh_quotient_refuses_an_open_block() {
    let space = QnSpace::new(4, 2, 1);
    let h = generic_truncated_h::<f64>(space);
    let x: TtVector<f64> = open_block(space, 2, 2, 46);
    let _ = rayleigh_quotient(&h, &x);
}
