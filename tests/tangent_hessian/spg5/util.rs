// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::su2::core::Su2Core;
use qntt::su2::expand::expand;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks};
use qntt::su2::riemannian::normalize_mut;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::vector::{TtVector, tt_randn};
use qntt::{Irrep, PointGroup, QnSpace, RankGrid, Scalar};

pub(crate) use crate::shared::dedup_sorted;
pub(crate) use crate::shared::su2_minimizer;
pub(crate) use crate::shared::tangent_ortho_basis;
pub(crate) use qntt_testkit::gad_mol::su2_grad_exact_at as grad_exact_at;
pub(crate) use qntt_testkit::pg::assert_irrep_split;
pub(crate) use qntt_testkit::pg::assert_purity;
pub(crate) use qntt_testkit::pg::c2v_coeffs;
pub(crate) use qntt_testkit::pg::c2v_table;
pub(crate) use qntt_testkit::pg::su2_capped_ranks;
pub(crate) use qntt_testkit::randomized_su2_state;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

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

pub(crate) fn assert_su2core_exact_eq<T: Scalar>(a: &Su2Core<T>, b: &Su2Core<T>, label: &str) {
    assert_eq!(
        a.num_coeffs(),
        b.num_coeffs(),
        "{label}: num_coeffs differs"
    );
    for i in 0..a.num_coeffs() {
        assert_eq!(a.coeff(i), b.coeff(i), "{label}: coeff {i} differs");
    }
}

pub(crate) fn assert_expand_matches<T: Scalar>(y: &Su2TtVector<T>, x: &Su2TtVector<T>, tol: f64) {
    assert_eq!(y.space, x.space, "assert_expand_matches: space mismatch");
    let two_s = y.space.two_s as i64;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let ye: TtVector<T> = expand(y, two_sz);
            let xe: TtVector<T> = expand(x, two_sz);
            let yd = ye.to_dense();
            let xd = xe.to_dense();
            let diff: f64 = yd
                .iter()
                .zip(&xd)
                .map(|(a, b)| {
                    let d = *a - *b;
                    d.abs_s() * d.abs_s()
                })
                .sum::<f64>()
                .sqrt();
            let refn: f64 = xd.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt();
            assert!(
                diff <= tol * refn.max(1.0),
                "expand mismatch at two_sz={two_sz}: diff={diff} refn={refn}"
            );
        }
        two_sz += 1;
    }
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

pub(crate) fn hess_fd_central<T: Scalar>(
    ham: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    xi: &Su2TtTangent<T>,
    h: f64,
) -> Su2TtTangent<T> {
    let mut xp = xi.retract_hosvd(T::from_f64(h));
    normalize_mut(&mut xp);
    let mut xm = xi.retract_hosvd(T::from_f64(-h));
    normalize_mut(&mut xm);
    let gp = qntt::su2::tangent::projective_transport(x, &grad_exact_at(ham, &xp));
    let gm = qntt::su2::tangent::projective_transport(x, &grad_exact_at(ham, &xm));
    gp.sub(&gm).scale(T::from_f64(1.0 / (2.0 * h)))
}

pub(crate) fn gauge_residual_norm(l: faer::MatRef<'_, f64>, c: faer::MatRef<'_, f64>) -> f64 {
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

pub(crate) fn retag_core<T: Scalar>(mut c: Su2Core<T>, space: Su2Space) -> Su2Core<T> {
    c.space = space;
    c
}
