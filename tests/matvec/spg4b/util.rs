// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::su2::core::{Family, Su2Core};
use qntt::su2::expand::expand;
use qntt::su2::frame::Su2Sketch;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hamiltonian::matvec::{Su2OneSiteWorkspace, su2_onesite_workspace};
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks};
use qntt::su2::twosite::Su2ContractedCores;
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::vector::TtVector;
use qntt::{Irrep, PointGroup, QnSpace, Scalar};

pub(crate) use qntt_testkit::pg::assert_irrep_split;
pub(crate) use qntt_testkit::pg::assert_purity;
pub(crate) use qntt_testkit::pg::c2v_coeffs;
pub(crate) use qntt_testkit::pg::c2v_table;
pub(crate) use qntt_testkit::pg::su2_capped_ranks;
use rand::SeedableRng;
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

pub(crate) fn max_err_over_two_sz_pg<T: Scalar>(
    x: &Su2TtVector<T>,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
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
        let h_ab = SparseHamiltonian::new(ab_space, t, v, None, f64::EPSILON);
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

pub(crate) fn window_rows(
    space: &Su2Space,
    k: usize,
    m: Su2Sector,
    row_g: Irrep,
) -> Vec<(Family, Su2Sector)> {
    let mut out = Vec::new();
    for f in Family::ALL {
        let Some(l) = f.unapply(m, row_g) else {
            continue;
        };
        if !space.contains(k, l) {
            continue;
        }
        out.push((f, l));
    }
    out
}

pub(crate) fn window_cols(
    space: &Su2Space,
    k: usize,
    m: Su2Sector,
    col_g: Irrep,
) -> Vec<(Family, Su2Sector)> {
    Family::ALL
        .iter()
        .filter_map(|&f| {
            f.apply(m, col_g)
                .filter(|&r| space.contains(k + 2, r))
                .map(|r| (f, r))
        })
        .collect()
}

pub(crate) fn rebuild_two_site<T: Scalar>(
    x: &Su2TtVector<T>,
    k: usize,
    xk: Su2Core<T>,
    s: &Su2Sketch<T>,
    mut xk1: Su2Core<T>,
) -> Su2TtVector<T> {
    xk1.fold_sketch_left_mut(s);
    let mut cores = x.cores.clone();
    cores[k] = xk;
    cores[k + 1] = xk1;
    Su2TtVector::new(x.space, cores)
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
