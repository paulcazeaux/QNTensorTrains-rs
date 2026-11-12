// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::core::Su2Core;
use qntt::su2::expand::expand;
use qntt::su2::frame::Su2Sketch;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hamiltonian::matvec::{Su2OneSiteWorkspace, su2_onesite_workspace};
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks};
use qntt::su2::twosite::Su2ContractedCores;
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::vector::{TtVector, tt_randn};
use qntt::{Irrep, PointGroup, QnSpace, RankGrid, Scalar};

pub(crate) use qntt_testkit::pg::assert_irrep_split;
pub(crate) use qntt_testkit::pg::assert_purity;
pub(crate) use qntt_testkit::pg::c2v_coeffs;
pub(crate) use qntt_testkit::pg::c2v_table;
pub(crate) use qntt_testkit::pg::su2_capped_ranks;
use rand::{RngExt, SeedableRng};
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

pub(crate) fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub(crate) fn right_to_left_chain<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Vec<Su2Sketch<T>> {
    let d = x.space.d;
    let mut chain: Vec<Su2Sketch<T>> = vec![Su2Sketch::identity(x.space, d, 1)];
    for k in (0..d).rev() {
        let ws: Su2OneSiteWorkspace = su2_onesite_workspace(h, x.core(k));
        let prev = chain.last().expect("chain non-empty");
        let c = h.sketching_step_left(x, prev, &ws);
        chain.push(c);
    }
    chain.reverse();
    chain
}

pub(crate) fn left_to_right_chain<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Vec<Su2Sketch<T>> {
    let d = x.space.d;
    let mut chain: Vec<Su2Sketch<T>> = vec![Su2Sketch::identity(x.space, 0, 1)];
    for k in 0..d {
        let ws: Su2OneSiteWorkspace = su2_onesite_workspace(h, x.core(k));
        let prev = chain.last().expect("chain non-empty");
        let c = h.sketching_step_right(x, prev, &ws);
        chain.push(c);
    }
    chain
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

pub(crate) fn assert_su2contracted_exact_eq<T: Scalar>(
    a: &Su2ContractedCores<T>,
    b: &Su2ContractedCores<T>,
    label: &str,
) {
    assert_eq!(a.mid_qn, b.mid_qn, "{label}: mid_qn differs");
    for &m in &a.mid_qn {
        match (a.block(m), b.block(m)) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(
                    (x.nrows(), x.ncols()),
                    (y.nrows(), y.ncols()),
                    "{label}: block {m:?} shape differs"
                );
                for j in 0..x.ncols() {
                    for i in 0..x.nrows() {
                        assert_eq!(
                            *x.get(i, j),
                            *y.get(i, j),
                            "{label}: block {m:?} entry ({i},{j}) differs"
                        );
                    }
                }
            }
            _ => panic!("{label}: block {m:?} presence differs"),
        }
    }
}

pub(crate) fn retag_core<T: Scalar>(mut c: Su2Core<T>, space: Su2Space) -> Su2Core<T> {
    c.space = space;
    c
}
