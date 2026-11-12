// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::su2::expand::expand;
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks};
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::vector::{TtVector, tt_randn};
use qntt::{Irrep, PointGroup, QnSpace, RankGrid, Scalar, ThreeBody};

pub(crate) use qntt_testkit::pg::assert_irrep_split;
pub(crate) use qntt_testkit::pg::assert_purity;
pub(crate) use qntt_testkit::pg::c2v_coeffs;
pub(crate) use qntt_testkit::pg::c2v_table;
pub(crate) use qntt_testkit::pg::su2_capped_ranks;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn find_nonidentity_reachable_target(
    pg: PointGroup,
    n: usize,
    two_s: usize,
    d: usize,
) -> Option<Irrep> {
    (1..pg.order()).map(Irrep::new).find(|&ir| {
        let space = Su2Space::with_point_group(d, n, two_s, pg, ir);
        let caps = max_ranks(&space);
        caps[d][Su2Sector::with_irrep(n, two_s, ir)] > 0
    })
}

pub(crate) fn assert_expand_purity_chain<T: Scalar>(x: &Su2TtVector<T>, pg: &PointGroup) {
    let d = x.space.d;
    let two_s = x.space.two_s as i64;
    let target = x.space.target;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let nup = ((x.space.n as i64 + two_sz) / 2) as usize;
            let ndn = ((x.space.n as i64 - two_sz) / 2) as usize;
            let y: TtVector<T> = expand(x, two_sz);
            assert_eq!(
                y.space,
                QnSpace::with_point_group(d, nup, ndn, x.space.pg, target),
                "expand(x, two_sz={two_sz}): abelian space mismatch"
            );
            assert!(
                y.check(),
                "expand(x, two_sz={two_sz}): abelian check() failed"
            );
            assert_irrep_split(pg, d, nup, ndn);
            assert_purity(pg, d, nup, ndn, target, &y.to_dense());
        }
        two_sz += 1;
    }
}

pub(crate) fn assert_su2core_exact_eq<T: Scalar>(
    a: &qntt::su2::core::Su2Core<T>,
    b: &qntt::su2::core::Su2Core<T>,
    label: &str,
) {
    assert_eq!(
        a.num_coeffs(),
        b.num_coeffs(),
        "{label}: num_coeffs differs"
    );
    for i in 0..a.num_coeffs() {
        assert_eq!(a.coeff(i), b.coeff(i), "{label}: coeff {i} differs");
    }
}

pub(crate) fn assert_su2tt_exact_eq<T: Scalar>(
    a: &Su2TtVector<T>,
    b: &Su2TtVector<T>,
    label: &str,
) {
    assert_eq!(a.space.d, b.space.d, "{label}: d differs");
    for k in 0..a.space.d {
        assert_su2core_exact_eq(a.core(k), b.core(k), &format!("{label} core {k}"));
    }
}

pub(crate) fn build_pg_state<T: Scalar>(
    d: usize,
    n: usize,
    two_s: usize,
    pg: PointGroup,
    target: Irrep,
    rmax: usize,
    seed: u64,
) -> Su2TtVector<T> {
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let ranks = su2_capped_ranks(&space, rmax);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x: Su2TtVector<T> = su2_tt_randn(space, &ranks, &mut rng);
    assert!(x.check(), "build_pg_state: fixture failed check()");
    x
}

pub(crate) fn assert_c2v_coeffs_nonvacuous(pg: &PointGroup, d: usize, t: &CoeffTensor<f64>) {
    assert!(pg.order() > 1, "fixture group must be nontrivial");
    assert!(
        (1..d).any(|k| pg.irrep_of(k) != pg.irrep_of(0)),
        "fixture group must be inhomogeneous across sites"
    );
    assert!(
        (0..d).any(|i| (0..d).any(|j| i != j && t.get(&[i, j]) != 0.0)),
        "vacuous: t has no allowed off-diagonal nonzero despite a same-irrep pair existing"
    );
    assert!(
        (0..d).any(|i| (0..d).any(|j| pg.irrep_of(i) != pg.irrep_of(j) && t.get(&[i, j]) == 0.0)),
        "vacuous: t has no symmetry-forbidden (necessarily-zero) entry"
    );
}

pub(crate) fn c2v_w(
    pg: &PointGroup,
    d: usize,
    count: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> CoeffTensor<f64> {
    let mut w = CoeffTensor::<f64>::zeros(d, 6);
    let mut placed = 0usize;
    let mut tries = 0usize;
    while placed < count && tries < count * 200 {
        tries += 1;
        let idx = [
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
        ];
        let tot = idx
            .iter()
            .fold(Irrep::IDENTITY, |acc, &s| acc.fuse(pg.irrep_of(s)));
        if tot == Irrep::IDENTITY && w.get(&idx) == 0.0 {
            w.set(&idx, rng.sample::<f64, _>(StandardNormal));
            placed += 1;
        }
    }
    assert!(placed > 0, "vacuous: c2v_w placed zero entries");
    w
}

pub(crate) fn find_same_nonidentity_irrep_pair(pg: &PointGroup, d: usize) -> (usize, usize) {
    for i in 0..d {
        for j in 0..d {
            if i != j && pg.irrep_of(i) == pg.irrep_of(j) && pg.irrep_of(i) != Irrep::IDENTITY {
                return (i, j);
            }
        }
    }
    panic!("fixture bug: no same-nonidentity-irrep pair found");
}

pub(crate) fn find_different_irrep_pair(pg: &PointGroup, d: usize) -> (usize, usize) {
    for i in 0..d {
        for j in 0..d {
            if i != j && pg.irrep_of(i) != pg.irrep_of(j) {
                return (i, j);
            }
        }
    }
    panic!("fixture bug: no differing-irrep pair found");
}

pub(crate) fn capped_ranks_ab(
    space: &QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<RankGrid> {
    let d = space.d;
    let caps = qntt::max_ranks(space);
    (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(space);
            for s in space.bond_sectors(b) {
                let draw = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
                r[s] = draw.min(caps[b][s]);
            }
            r
        })
        .collect()
}

pub(crate) fn capped_randn_ab<T: Scalar>(
    space: QnSpace,
    lo: usize,
    hi: usize,
    seed: u64,
) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks = capped_ranks_ab(&space, lo, hi, &mut rng);
    let x: TtVector<T> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    x
}

pub(crate) fn randn_coeffs(
    d: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], rng.sample::<f64, _>(StandardNormal));
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    v.set(&[i, j, k, l], rng.sample::<f64, _>(StandardNormal));
                }
            }
        }
    }
    (t, v)
}

pub(crate) fn max_err_over_two_sz_pg<T: Scalar>(
    x: &Su2TtVector<T>,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn ThreeBody<T>>,
    y: &Su2TtVector<T>,
    space: Su2Space,
) -> f64 {
    let mut max_err = 0.0f64;
    let two_s = space.two_s as i64;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        let ab_space = QnSpace::from_n_two_sz_with_point_group(
            space.d,
            space.n,
            two_sz,
            space.pg,
            space.target,
        );
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
