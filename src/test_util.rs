// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::qn::{QnSpace, RankGrid};
use crate::scalar::Scalar;
use crate::su2::qn::{Su2RankGrid, Su2Space};
use crate::su2::vector::Su2TtVector;
use crate::vector::TtVector;
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub(crate) fn randomized_state<T: Scalar, R: Rng + ?Sized>(
    space: QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut R,
) -> TtVector<T> {
    let d = space.d;
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    crate::vector::tt_randn(space, &ranks, rng)
}

pub(crate) fn randomized_su2_state<T: Scalar, R: Rng + ?Sized>(
    space: Su2Space,
    lo: usize,
    hi: usize,
    rng: &mut R,
) -> Su2TtVector<T> {
    let d = space.d;
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    crate::su2::vector::tt_randn(space, &ranks, rng)
}

use crate::C64;
use crate::coeff::CoeffTensor;
use crate::core::tangent::TtTangent;
use crate::hamiltonian::SparseHamiltonian;
use crate::pointgroup::{Irrep, PointGroup, PointGroupKind};
use crate::su2::expand::expand;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::tangent::Su2TtTangent;
use faer::MatRef;
use std::collections::HashMap;

pub(crate) fn build_h_matvec(space: QnSpace) -> SparseHamiltonian<f64> {
    let d = space.d;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            if i != j {
                t.set(&[i, j], 1.0 / (i as f64 - j as f64).abs());
            }
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let denom = (i as f64 + k as f64) - (j as f64 + l as f64);
                    if denom != 0.0 {
                        v.set(&[i, j, k, l], 2.0 / denom.abs());
                    }
                }
            }
        }
    }
    SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}
pub(crate) fn pg3_space_d8() -> (PointGroup, QnSpace) {
    let pg = c2v_pg(8);
    (pg, QnSpace::with_point_group(8, 4, 4, pg, Irrep::new(1)))
}
pub(crate) fn pg3_space_d6() -> (PointGroup, QnSpace) {
    let pg = c2v_pg(6);
    (pg, QnSpace::with_point_group(6, 3, 3, pg, Irrep::new(1)))
}
pub(crate) fn assert_pure<T: Scalar>(pg: &PointGroup, space: &QnSpace, x: &TtVector<T>) {
    assert_purity(
        pg,
        space.d,
        space.nup,
        space.ndn,
        space.target,
        &x.to_dense(),
    );
}

pub(crate) fn c2v_pg(d: usize) -> PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| [2u8, 3, 4, 1][k % 4]).collect();
    PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}
pub(crate) fn naive_ranks(
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

pub(crate) fn find_determinant(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    irrep: Irrep,
) -> (Vec<bool>, Vec<bool>) {
    for idx in 0..(1usize << (2 * d)) {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            continue;
        }
        if config_irrep_from_digits(pg, &dig) == irrep {
            return config_occ(&dig);
        }
    }
    panic!("d={d}: no determinant with (nup={nup}, ndn={ndn}, irrep={irrep:?})");
}
pub(crate) fn assert_purity<T: Scalar>(
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

pub(crate) fn digits(idx: usize, d: usize) -> Vec<usize> {
    (0..d).map(|k| (idx >> (2 * k)) & 3).collect()
}
pub(crate) fn config_counts(dig: &[usize]) -> (usize, usize) {
    let nup = dig.iter().filter(|&&x| x == 1 || x == 3).count();
    let ndn = dig.iter().filter(|&&x| x == 2 || x == 3).count();
    (nup, ndn)
}
pub(crate) fn config_irrep_from_digits(pg: &PointGroup, dig: &[usize]) -> Irrep {
    let mut irrep = Irrep::IDENTITY;
    for (k, &x) in dig.iter().enumerate() {
        if x == 1 || x == 3 {
            irrep = irrep.fuse(pg.irrep_of(k));
        }
        if x == 2 || x == 3 {
            irrep = irrep.fuse(pg.irrep_of(k));
        }
    }
    irrep
}
pub(crate) fn config_occ(dig: &[usize]) -> (Vec<bool>, Vec<bool>) {
    let occ_up = dig.iter().map(|&x| x == 1 || x == 3).collect();
    let occ_dn = dig.iter().map(|&x| x == 2 || x == 3).collect();
    (occ_up, occ_dn)
}

pub(crate) fn su2_capped_grids(space: &Su2Space, m: usize) -> Vec<Su2RankGrid> {
    crate::su2::qn::max_ranks(space)
        .into_iter()
        .map(|g| {
            let mut out = Su2RankGrid::zeros(space);
            for (s, &c) in g.iter() {
                out[s] = c.min(m);
            }
            out
        })
        .collect()
}

pub(crate) fn su2_coeffs<T: Scalar>(c: &crate::su2::core::Su2Core<T>) -> Vec<T> {
    let mut out = Vec::with_capacity(c.num_coeffs());
    c.for_each_coeff(|v| out.push(v));
    out
}

pub(crate) fn randn_matrix<T: Scalar, R: Rng + ?Sized>(d: usize, rng: &mut R) -> CoeffTensor<T> {
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], T::from_f64(rng.sample(rand_distr::StandardNormal)));
        }
    }
    t
}

pub(crate) fn randn_4tensor<T: Scalar, R: Rng + ?Sized>(d: usize, rng: &mut R) -> CoeffTensor<T> {
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    v.set(
                        &[i, j, k, l],
                        T::from_f64(rng.sample(rand_distr::StandardNormal)),
                    );
                }
            }
        }
    }
    v
}

pub(crate) fn zeros_matrix<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 2)
}

pub(crate) fn zeros_4tensor<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 4)
}

pub(crate) fn su2_hcheck<T: Scalar>(
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    x: &Su2TtVector<T>,
) -> f64 {
    let space = x.space;
    let h = Su2SparseHamiltonian::new(space, t, v, None, f64::EPSILON);
    let y = h.apply(x);
    assert!(y.check(), "su2_hcheck: H*x fails check()");
    max_err_over_two_sz(x, t, v, None, &y, space)
}

pub(crate) fn su2_hcheck3<T: Scalar>(
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: &dyn crate::ThreeBody<T>,
    x: &Su2TtVector<T>,
) -> f64 {
    let space = x.space;
    let h = Su2SparseHamiltonian::new(space, t, v, Some(w), f64::EPSILON);
    let y = h.apply(x);
    assert!(y.check(), "su2_hcheck3: H*x fails check()");
    max_err_over_two_sz(x, t, v, Some(w), &y, space)
}

pub(crate) fn max_err_over_two_sz<T: Scalar>(
    x: &Su2TtVector<T>,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn crate::ThreeBody<T>>,
    y: &Su2TtVector<T>,
    space: Su2Space,
) -> f64 {
    let mut max_err = 0.0f64;
    let two_s = space.two_s as i64;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        let ab_space = QnSpace::from_n_two_sz(space.d, space.n, two_sz);
        let h_ab = SparseHamiltonian::new(ab_space, t, v, w, f64::EPSILON);
        let x_exp: TtVector<T> = expand(x, two_sz);
        let yref = &h_ab * &x_exp;
        let y_exp: TtVector<T> = expand(y, two_sz);
        let got = y_exp.to_dense();
        let want = yref.to_dense();
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| {
                let d = *g - *w;
                d.abs_s() * d.abs_s()
            })
            .sum::<f64>()
            .sqrt();
        let norm_ref: f64 = want
            .iter()
            .map(|w| w.abs_s() * w.abs_s())
            .sum::<f64>()
            .sqrt();
        let rel = err / norm_ref.max(1.0);
        if rel > max_err {
            max_err = rel;
        }
        two_sz += 2;
    }
    max_err
}

pub(crate) fn c2v_table(d: usize) -> PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| ((k % 4) + 1) as u8).collect();
    PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}

pub(crate) fn config_irrep(pg: &PointGroup, occ_up: &[bool], occ_dn: &[bool]) -> Irrep {
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

pub(crate) fn irrep_census(pg: &PointGroup, d: usize) -> HashMap<(usize, usize, Irrep), u64> {
    let mut counts = HashMap::new();
    for idx in 0..(1usize << (2 * d)) {
        let dig = digits(idx, d);
        let (nup, ndn) = config_counts(&dig);
        let irrep = config_irrep_from_digits(pg, &dig);
        *counts.entry((nup, ndn, irrep)).or_insert(0u64) += 1;
    }
    counts
}

pub(crate) fn pick_nontrivial_target(
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

pub(crate) fn pg3_space_d4() -> (PointGroup, QnSpace) {
    let pg = c2v_pg(4);
    (pg, QnSpace::with_point_group(4, 2, 2, pg, Irrep::new(2)))
}

pub(crate) fn dense_dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (a, b) in x.iter().zip(y) {
        s += a.conj_s() * *b;
    }
    s
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

pub(crate) fn rel_err<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    dense_diff_norm(a, b) / dense_norm(b)
}

pub(crate) fn rel_lr_c(z: &TtVector<C64>, x: &TtVector<C64>) -> f64 {
    let diff = z - x;
    diff.norm_lr(false) / x.norm_lr(false)
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

pub(crate) fn core_left_orthonormality_dev<T: Scalar>(tt: &TtVector<T>) -> (f64, usize) {
    let d = tt.space.d;
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for k in 0..d.saturating_sub(1) {
        let core = &tt.cores[k];
        for &cs in &core.col_qn {
            if core.col_ranks[cs] == 0 {
                continue;
            }
            let q = core.vertical_unfolding(cs);
            if q.nrows() == 0 {
                continue;
            }
            worst = worst.max(gram_col_dev(q.as_ref()));
            checked += 1;
        }
    }
    (worst, checked)
}

pub(crate) fn core_right_orthonormality_dev<T: Scalar>(tt: &TtVector<T>) -> (f64, usize) {
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

pub(crate) fn sum_interior_rank<T: Scalar>(tt: &TtVector<T>) -> usize {
    (1..tt.space.d).map(|b| tt.ranks[b].total()).sum()
}

pub(crate) fn generic_truncated_h<T: Scalar>(space: QnSpace) -> SparseHamiltonian<T> {
    let (t, v) = generic_truncated_tv::<T>(space.d);
    SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub(crate) fn generic_truncated_tv<T: Scalar>(d: usize) -> (CoeffTensor<T>, CoeffTensor<T>) {
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for p in 0..d - 1 {
        t.set(&[p, p + 1], T::from_f64(-1.0));
        t.set(&[p + 1, p], T::from_f64(-1.0));
    }
    for p in 0..d {
        t.set(&[p, p], T::from_f64(0.15 * (p as f64 - 0.5 * d as f64)));
    }
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(
                    &[p, p, q, q],
                    T::from_f64(0.5 / (1.0 + (p as f64 - q as f64).abs())),
                );
            }
        }
    }
    (t, v)
}

pub(crate) fn generic_su2_h<T: Scalar>(space: Su2Space) -> Su2SparseHamiltonian<T> {
    let d = space.d;
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for p in 0..d - 1 {
        t.set(&[p, p + 1], T::from_f64(-1.0));
        t.set(&[p + 1, p], T::from_f64(-1.0));
    }
    for p in 0..d {
        t.set(&[p, p], T::from_f64(0.15 * (p as f64 - 0.5 * d as f64)));
    }
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(
                    &[p, p, q, q],
                    T::from_f64(0.5 / (1.0 + (p as f64 - q as f64).abs())),
                );
            }
        }
    }
    Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub(crate) fn pg_symmetric_tv(
    pg: &PointGroup,
    d: usize,
    seed: u64,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut rng = seeded(seed);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in i..d {
            if pg.irrep_of(i) == pg.irrep_of(j) {
                let val = f64::sample_standard(&mut rng);
                t.set(&[i, j], val);
                t.set(&[j, i], val);
            }
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let fused = pg
                        .irrep_of(i)
                        .fuse(pg.irrep_of(j))
                        .fuse(pg.irrep_of(k))
                        .fuse(pg.irrep_of(l));
                    if fused != Irrep::IDENTITY || v.get(&[i, j, k, l]) != 0.0 {
                        continue;
                    }
                    let val = f64::sample_standard(&mut rng);
                    for idx in [
                        [i, j, k, l],
                        [j, i, k, l],
                        [i, j, l, k],
                        [j, i, l, k],
                        [k, l, i, j],
                        [l, k, i, j],
                        [k, l, j, i],
                        [l, k, j, i],
                    ] {
                        v.set(&idx, val);
                    }
                }
            }
        }
    }
    (t, v)
}

pub(crate) fn su2_pg_space(d: usize, n: usize, two_s: usize) -> (PointGroup, Su2Space) {
    let pg = c2v_pg(d);
    let target = (1..pg.order())
        .map(Irrep::new)
        .find(|&ir| {
            let sp = Su2Space::with_point_group(d, n, two_s, pg, ir);
            crate::su2::qn::max_ranks(&sp)[d][crate::su2::qn::Su2Sector::with_irrep(n, two_s, ir)]
                > 0
        })
        .expect("a reachable non-identity target");
    (pg, Su2Space::with_point_group(d, n, two_s, pg, target))
}

pub(crate) fn pg4_curved_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(3), Irrep::new(2), Irrep::new(1), Irrep::new(0)],
    )
}

pub(crate) fn pg4_curved_state(seed: u64) -> TtVector<f64> {
    let space = QnSpace::with_point_group(4, 2, 2, pg4_curved_pg(), Irrep::IDENTITY);
    let mut rng = seeded(seed);
    let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    x.round_mut(1e-8, crate::qn::Pooled::NONE, crate::round::RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    x
}

pub(crate) fn schmidt_floor_candidates(values: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values
        .iter()
        .copied()
        .filter(|s| s.is_finite() && *s > 0.0)
        .collect();
    v.sort_by(f64::total_cmp);
    v.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * b.abs());
    let mut out = Vec::new();
    for w in v.windows(2) {
        if (w[1] - w[0]) / w[1] >= 1e-3 {
            let f = (w[0] * w[1]).sqrt();
            assert!(
                values.iter().all(|s| (s - f).abs() > 1e-6 * f),
                "floor rule: a Schmidt value lies within 1e-6 of the floor {f:e}"
            );
            out.push(f);
        }
    }
    out
}

pub(crate) fn pooled_schmidt(gauges: &[crate::core::weingarten::BondGauge<f64>]) -> Vec<f64> {
    gauges
        .iter()
        .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
        .collect()
}

pub(crate) fn relabel_sensitive_floor(
    bl: &TtVector<f64>,
    br: &TtVector<f64>,
    gauges: &[crate::core::weingarten::BondGauge<f64>],
    phi: Irrep,
) -> Option<(f64, Vec<(usize, crate::qn::Sector)>)> {
    use crate::qn::Sector;
    let space = bl.space;
    let space_out = space.with_target(space.target.fuse(phi));
    let basis = crate::core::tangent::tangent_basis_shifted(bl, br, phi);
    let template = basis.first()?;
    let shift = |t: Sector| Sector::with_irrep(t.nup, t.ndn, t.irrep.fuse(phi));
    let sigma = |b: usize, t: Sector| -> Vec<f64> {
        gauges
            .iter()
            .filter(|g| g.k == b)
            .flat_map(|g| g.svals().iter())
            .filter(|(q, _)| *q == t)
            .flat_map(|(_, v)| v.iter().copied())
            .collect()
    };
    for f in schmidt_floor_candidates(&pooled_schmidt(gauges)) {
        let mut blocks = Vec::new();
        for b in 1..space.d {
            for t in space.bond_sectors(b) {
                let r_t = bl.ranks[b][t];
                if r_t == 0 || !sigma(b, t).iter().any(|&s| s < f) {
                    continue;
                }
                let s = shift(t);
                if !space_out.contains(b, s) {
                    continue;
                }
                let r_s = bl.ranks[b][s];
                let uncut = sigma(b, s).iter().all(|&v| v >= f);
                let rows = template.component(b - 1).vertical_unfolding(s).nrows();
                let room = rows.saturating_sub(r_s) * r_t;
                if uncut && r_t != r_s && room > 0 {
                    blocks.push((b, t));
                }
            }
        }
        if !blocks.is_empty() {
            return Some((f, blocks));
        }
    }
    None
}

pub(crate) fn in_pool<R: Send>(nt: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build()
        .unwrap()
        .install(f)
}

pub(crate) fn set8(v: &mut CoeffTensor<f64>, i: usize, j: usize, k: usize, l: usize, val: f64) {
    for idx in [
        [i, j, k, l],
        [j, i, k, l],
        [i, j, l, k],
        [j, i, l, k],
        [k, l, i, j],
        [l, k, i, j],
        [k, l, j, i],
        [l, k, j, i],
    ] {
        v.set(&idx, val);
    }
}

pub(crate) fn tangent_bits(v: &TtTangent<f64>) -> Vec<u64> {
    (0..v.num_coeffs()).map(|i| v.coeff(i).to_bits()).collect()
}

pub(crate) fn su2_tangent_bits(v: &Su2TtTangent<f64>) -> Vec<u64> {
    (0..v.num_coeffs()).map(|i| v.coeff(i).to_bits()).collect()
}

pub(crate) fn flat_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[cfg(test)]
mod parity_pins {
    use super::*;

    #[test]
    fn abelian_draw_matches_testkit() {
        let mine: TtVector<f64> =
            randomized_state(QnSpace::new(4, 2, 1), 1, 3, &mut seeded(0x71717));
        let theirs = qntt_testkit::randomized_state::<f64, _>(
            qntt::QnSpace::new(4, 2, 1),
            1,
            3,
            &mut seeded(0x71717),
        );
        assert_eq!(
            mine.to_dense(),
            theirs.to_dense(),
            "the in-crate randomized_state twin drifted from qntt_testkit's"
        );
    }

    #[test]
    fn generic_truncated_tv_matches_testkit() {
        let d = 5;
        let (t, v) = generic_truncated_tv::<f64>(d);
        let (tk_t, tk_v) = qntt_testkit::generic_truncated_tv::<f64>(d);
        for p in 0..d {
            for q in 0..d {
                assert_eq!(t.get(&[p, q]), tk_t.get(&[p, q]));
                for r in 0..d {
                    for s in 0..d {
                        assert_eq!(v.get(&[p, q, r, s]), tk_v.get(&[p, q, r, s]));
                    }
                }
            }
        }
    }

    #[test]
    fn su2_draw_matches_testkit() {
        let mine: Su2TtVector<f64> =
            randomized_su2_state(Su2Space::new(4, 3, 1), 1, 2, &mut seeded(0x71718));
        let theirs = qntt_testkit::randomized_su2_state::<f64, _>(
            qntt::su2::qn::Su2Space::new(4, 3, 1),
            1,
            2,
            &mut seeded(0x71718),
        );
        let mine_d: Vec<f64> = crate::su2::expand::expand(&mine, 1).to_dense();
        let theirs_d: Vec<f64> = qntt::su2::expand::expand(&theirs, 1).to_dense();
        assert_eq!(
            mine_d, theirs_d,
            "the in-crate randomized_su2_state twin drifted from qntt_testkit's"
        );
    }
}
