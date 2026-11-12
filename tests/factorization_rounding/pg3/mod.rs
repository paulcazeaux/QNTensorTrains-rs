// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use qntt::core::factor::{adjoint_core_times_core, core_times_adjoint_core};
use qntt::core::tangent::{TtTangent, projective_transport};
use qntt::core::twosite::{contract, factor_qc};
use qntt::{
    C64, Irrep, PointGroup, PointGroupKind, QnSpace, RankGrid, RoundMode, Scalar, Sector, TtVector,
    dot, dot_orth, hadamard, max_ranks, tt_randn, tt_state_in,
};
use qntt_testkit::pg::capped_ranks;
use qntt_testkit::pg::config_counts;
use qntt_testkit::pg::config_irrep_from_digits;
use qntt_testkit::pg::digits;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn c2v_pg(d: usize) -> PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| [2u8, 3, 4, 1][k % 4]).collect();
    PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}

fn pg3_space_d6() -> (PointGroup, QnSpace) {
    let pg = c2v_pg(6);
    (pg, QnSpace::with_point_group(6, 3, 3, pg, Irrep::new(1)))
}

fn naive_ranks(
    space: &QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<RankGrid> {
    let d = space.d;
    (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect()
}

fn capped_randn<T: Scalar>(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks = capped_ranks(&space, lo, hi, &mut rng);
    let x: TtVector<T> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    x
}

fn config_occ(dig: &[usize]) -> (Vec<bool>, Vec<bool>) {
    let occ_up = dig.iter().map(|&x| x == 1 || x == 3).collect();
    let occ_dn = dig.iter().map(|&x| x == 2 || x == 3).collect();
    (occ_up, occ_dn)
}

#[allow(dead_code)]
fn config_irrep(pg: &PointGroup, occ_up: &[bool], occ_dn: &[bool]) -> Irrep {
    let mut irrep = Irrep::IDENTITY;
    for (k, (&u, &v)) in occ_up.iter().zip(occ_dn.iter()).enumerate() {
        if u {
            irrep = irrep.fuse(pg.irrep_of(k));
        }
        if v {
            irrep = irrep.fuse(pg.irrep_of(k));
        }
    }
    irrep
}

fn determinants_with_irrep(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    irrep: Irrep,
) -> Vec<(Vec<bool>, Vec<bool>)> {
    let mut out = Vec::new();
    for idx in 0..(1usize << (2 * d)) {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            continue;
        }
        if config_irrep_from_digits(pg, &dig) == irrep {
            out.push(config_occ(&dig));
        }
    }
    out
}

#[allow(dead_code)]
fn pick_nontrivial_target(
    census: &HashMap<(usize, usize, Irrep), u64>,
    d: usize,
) -> (usize, usize, Irrep) {
    census
        .iter()
        .filter(|&(&(_, _, ir), &c)| ir != Irrep::IDENTITY && c > 0)
        .map(|(&(u, v, ir), _)| (u, v, ir))
        .min_by_key(|&(u, v, ir)| (u, v, ir.index()))
        .unwrap_or_else(|| panic!("d={d}: fixture has no reachable nonidentity-irrep determinant"))
}

fn assert_purity<T: Scalar>(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    target: Irrep,
    dense: &[T],
) {
    assert_eq!(dense.len(), 1usize << (2 * d));
    let mut any_nonzero_in_target = false;
    for (idx, v) in dense.iter().enumerate() {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            assert!(
                *v == T::zero(),
                "d={d}: config {idx} outside (Nup={nup},Ndn={ndn}) must be exactly zero, got {v:?}"
            );
            continue;
        }
        let irrep = config_irrep_from_digits(pg, &dig);
        if irrep == target {
            if *v != T::zero() {
                any_nonzero_in_target = true;
            }
        } else {
            assert!(
                *v == T::zero(),
                "THE campaign oracle: d={d} config {idx} has irrep {irrep:?} != target \
                 {target:?} but amplitude is {v:?} (not exactly zero)"
            );
        }
    }
    assert!(
        any_nonzero_in_target,
        "d={d}: oracle vacuous — no nonzero amplitude found inside the target irrep {target:?}"
    );
}

fn assert_pure<T: Scalar>(pg: &PointGroup, space: &QnSpace, x: &TtVector<T>) {
    assert_purity(
        pg,
        space.d,
        space.nup,
        space.ndn,
        space.target,
        &x.to_dense(),
    );
}

fn dense_dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (a, b) in x.iter().zip(y) {
        s += a.conj_s() * *b;
    }
    s
}

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

fn core_right_orthonormality_dev<T: Scalar>(tt: &TtVector<T>) -> (f64, usize) {
    let d = tt.space.d;
    let mut worst = 0.0f64;
    let mut checked = 0usize;
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
            checked += 1;
        }
    }
    (worst, checked)
}

fn sum_interior_rank<T: Scalar>(tt: &TtVector<T>) -> usize {
    (1..tt.space.d).map(|b| tt.ranks[b].total()).sum()
}

fn gauge_residual(l: MatRef<'_, f64>, c: MatRef<'_, f64>) -> f64 {
    let mut n2 = 0.0f64;
    for j in 0..c.ncols() {
        for i in 0..l.ncols() {
            let mut acc = 0.0f64;
            for kk in 0..l.nrows() {
                acc += *l.get(kk, i) * *c.get(kk, j);
            }
            n2 += acc * acc;
        }
    }
    n2.sqrt()
}

mod algebra;
mod factorization;
mod fixpin;
mod orthogonalize;
mod rounding;
mod tangent;
