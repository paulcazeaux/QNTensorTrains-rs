// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatRef, Par};

use qntt::su2::core::{Family as Su2Family, Su2Core};
use qntt::su2::frame::Su2Sketch;
use qntt::su2::qn::{Su2Sector, Su2Space};
use qntt::su2::theta;
use qntt::su2::vector::Su2TtVector;

use qntt::Scalar;
pub(crate) use qntt_testkit::randomized_su2_state;
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn random_csf<T: Scalar, R: Rng + ?Sized>(
    d: usize,
    n: usize,
    two_s: usize,
    rng: &mut R,
) -> Su2TtVector<T> {
    let space = Su2Space::new(d, n, two_s);
    let deltas: [(Su2Family, i64, i64); 4] = [
        (Su2Family::Empty, 0, 0),
        (Su2Family::Up, 1, 1),
        (Su2Family::Dn, 1, -1),
        (Su2Family::Occ, 2, 0),
    ];
    let mut codes = Vec::with_capacity(d);
    let (mut cn, mut ct): (i64, i64) = (0, 0);
    for k in 0..d {
        let mut moves = Vec::new();
        for &(f, dn, dt) in &deltas {
            let nn = cn + dn;
            let nt = ct + dt;
            if nt < 0 {
                continue;
            }
            if space.contains(k + 1, Su2Sector::new(nn as usize, nt as usize)) {
                moves.push((f, nn, nt));
            }
        }
        assert!(
            !moves.is_empty(),
            "random_csf: no admissible move at site {k}"
        );
        let idx = rng.random_range(0..moves.len());
        let (f, nn, nt) = moves[idx];
        codes.push(f);
        cn = nn;
        ct = nt;
    }
    qntt::su2::tt_csf(&codes)
}

pub(crate) fn dense_norm<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt()
}

pub(crate) fn dense_diff_norm<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x - *y;
            d.abs_s() * d.abs_s()
        })
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn assert_dense_close<T: Scalar>(got: &[T], want: &[T], tol: f64) {
    let err = dense_diff_norm(got, want);
    let bound = tol * dense_norm(want).max(1.0);
    assert!(
        err <= bound,
        "dense mismatch: err={err} > bound={bound} (tol={tol})"
    );
}

pub(crate) fn close_rel_abs(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

pub(crate) fn gram_col_dev<T: Scalar>(q: MatRef<'_, T>) -> f64 {
    let n = q.ncols();
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut acc = T::zero();
            for r in 0..q.nrows() {
                acc += q.get(r, i).conj_s() * *q.get(r, j);
            }
            let target = if i == j { T::one() } else { T::zero() };
            worst = worst.max((acc - target).abs_s());
        }
    }
    worst
}

pub(crate) fn gram_row_dev<T: Scalar>(a: MatRef<'_, T>) -> f64 {
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

pub(crate) fn frob<T: Scalar>(a: MatRef<'_, T>) -> f64 {
    let mut s = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let v = *a.get(i, j);
            s += v.abs_s() * v.abs_s();
        }
    }
    s.sqrt()
}

pub(crate) fn frob_diff<T: Scalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> f64 {
    let mut s = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let d = *a.get(i, j) - *b.get(i, j);
            s += d.abs_s() * d.abs_s();
        }
    }
    s.sqrt()
}

pub(crate) fn su2_left_canonicity_dev<T: Scalar>(
    x: &Su2TtVector<T>,
    range: std::ops::Range<usize>,
) -> f64 {
    let mut maxdev = 0.0f64;
    for k in range {
        let core = x.core(k);
        for r in core.col_qn.iter().copied() {
            if core.col_ranks[r] == 0 {
                continue;
            }
            let v = core.vertical_unfolding(r);
            maxdev = maxdev.max(gram_col_dev(v.as_ref()));
        }
    }
    maxdev
}

pub(crate) fn accumulate_weighted_bbh<T: Scalar>(g: &mut Mat<T>, b: MatRef<'_, T>, weight: f64) {
    let w = T::from_f64(weight);
    for i in 0..b.nrows() {
        for j in 0..b.nrows() {
            let mut acc = T::zero();
            for c in 0..b.ncols() {
                acc += *b.get(i, c) * b.get(j, c).conj_s();
            }
            *g.get_mut(i, j) += acc * w;
        }
    }
}

pub(crate) fn max_dev_identity<T: Scalar>(g: MatRef<'_, T>) -> f64 {
    let n = g.nrows();
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { T::one() } else { T::zero() };
            worst = worst.max((*g.get(i, j) - target).abs_s());
        }
    }
    worst
}

pub(crate) fn su2_right_canonicity_dev<T: Scalar>(
    x: &Su2TtVector<T>,
    range: std::ops::Range<usize>,
) -> f64 {
    let mut maxdev = 0.0f64;
    for k in range {
        let core = x.core(k);
        for l in core.row_qn.iter().copied() {
            let r = core.row_ranks[l];
            if r == 0 {
                continue;
            }
            let mut g = Mat::<T>::zeros(r, r);
            let weights: [(Su2Family, f64); 4] = [
                (Su2Family::Empty, 1.0),
                (
                    Su2Family::Up,
                    (l.two_s as f64 + 2.0) / (l.two_s as f64 + 1.0),
                ),
                (Su2Family::Dn, l.two_s as f64 / (l.two_s as f64 + 1.0)),
                (Su2Family::Occ, 1.0),
            ];
            for (f, w) in weights {
                if let Some(b) = core.block(f, l) {
                    accumulate_weighted_bbh(&mut g, b, w);
                }
            }
            maxdev = maxdev.max(max_dev_identity(g.as_ref()));
        }
    }
    maxdev
}

#[test]
fn helper_right_canonicity_weights_match_theta_squared() {
    for two_s in 0usize..6 {
        let up = (two_s as f64 + 2.0) / (two_s as f64 + 1.0);
        let dn = two_s as f64 / (two_s as f64 + 1.0);
        assert!((up - theta(two_s, two_s + 1).powi(2)).abs() < 1e-14);
        if two_s >= 1 {
            assert!((dn - theta(two_s, two_s - 1).powi(2)).abs() < 1e-14);
        }
    }
}

pub(crate) fn all_two_sz(two_s: usize) -> Vec<i64> {
    let t = two_s as i64;
    (-t..=t).step_by(2).collect()
}

pub(crate) fn pick_open_shell_core<T: Scalar>(x: &Su2TtVector<T>) -> usize {
    let d = x.space.d;
    for k in 0..d.saturating_sub(1).max(1) {
        let core = x.core(k);
        let has_branch = core.row_qn.iter().copied().any(|s| {
            core.row_ranks[s] > 0
                && [Su2Family::Up, Su2Family::Dn].iter().any(|&f| {
                    f.apply(s, core.space.pg.irrep_of(k))
                        .is_some_and(|t| core.space.contains(k + 1, t) && core.col_ranks[t] > 0)
                })
        });
        if has_branch {
            return k;
        }
    }
    panic!("pick_open_shell_core: no interior core with an up/dn branch (bad test sizing)");
}

pub(crate) fn sketch_frob_diff<T: Scalar>(a: &Su2Sketch<T>, b: &Su2Sketch<T>) -> f64 {
    let mut s = 0.0f64;
    for sec in a.sectors() {
        if let (Some(ab), Some(bb)) = (a.block(sec), b.block(sec)) {
            s += frob_diff(ab, bb).powi(2);
        }
    }
    s.sqrt()
}

pub(crate) fn rproject_plain_transpose<T: Scalar>(a: &Su2Core<T>, q: &Su2Core<T>) -> Su2Sketch<T> {
    let space = a.space;
    let k = a.k;
    let mut c = Su2Sketch::zeros(space, k, &a.row_ranks, &q.row_ranks);
    for l in space.bond_sectors(k) {
        let Some(mut dst) = c.block_mut(l) else {
            continue;
        };
        for f in Su2Family::ALL {
            let Some(t) = f.apply(l, space.pg.irrep_of(k)) else {
                continue;
            };
            if !space.contains(k + 1, t) {
                continue;
            }
            let (Some(ab), Some(qb)) = (a.block(f, l), q.block(f, l)) else {
                continue;
            };
            matmul(
                dst.rb_mut(),
                Accum::Add,
                ab,
                qb.transpose(),
                T::one(),
                Par::Seq,
            );
        }
    }
    c
}

pub(crate) fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}
