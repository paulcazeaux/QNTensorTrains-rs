// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use qntt::{C64, QnSpace, RoundMode, Scalar, TtVector};
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn dense_norm<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt()
}

fn dense_diff_norm<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x - *y;
            d.abs_s() * d.abs_s()
        })
        .sum::<f64>()
        .sqrt()
}

fn rel_err<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    dense_diff_norm(a, b) / dense_norm(b)
}

fn mat_mul<T: Scalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let mut c = Mat::<T>::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, T::one(), Par::Seq);
    c
}

fn frob<T: Scalar>(a: MatRef<'_, T>) -> f64 {
    let mut s = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let v = *a.get(i, j);
            s += v.abs_s() * v.abs_s();
        }
    }
    s.sqrt()
}

fn frob_diff<T: Scalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> f64 {
    let mut s = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let d = *a.get(i, j) - *b.get(i, j);
            s += d.abs_s() * d.abs_s();
        }
    }
    s.sqrt()
}

fn gram_row_dev<T: Scalar>(a: MatRef<'_, T>) -> f64 {
    let n = a.nrows();
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = T::zero();
            for c in 0..a.ncols() {
                acc += *a.get(i, c) * a.get(j, c).conj_s();
            }
            let target = if i == j { T::one() } else { T::zero() };
            worst = worst.max((acc - target).abs_s());
        }
    }
    worst
}

fn core_right_orthonormality_dev<T: Scalar>(tt: &TtVector<T>) -> f64 {
    let d = tt.space.d;
    let mut worst = 0.0f64;
    for k in 1..d {
        let core = &tt.cores[k];
        for &rs in &core.row_qn {
            if core.row_ranks[rs] == 0 {
                continue;
            }
            let a = core.horizontal_unfolding(rs);
            if a.ncols() == 0 {
                continue;
            }
            worst = worst.max(gram_row_dev(a.as_ref()));
        }
    }
    worst
}

fn sum_interior_rank<T: Scalar>(tt: &TtVector<T>) -> usize {
    (1..tt.space.d).map(|b| tt.ranks[b].total()).sum()
}

mod complex_coverage;
mod gauge_factor;
