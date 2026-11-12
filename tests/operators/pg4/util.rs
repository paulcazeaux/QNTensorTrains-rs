// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::Mat;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::{CoeffTensor, Irrep, PointGroup, PointGroupKind, QnSpace, Scalar, TtVector, tt_randn};
pub(crate) use qntt_testkit::dense_apply_string;
pub(crate) use qntt_testkit::pg::c2v_coeffs;
pub(crate) use qntt_testkit::pg::capped_ranks;
pub(crate) use qntt_testkit::pg::config_counts;
pub(crate) use qntt_testkit::pg::config_irrep_from_digits;
pub(crate) use qntt_testkit::pg::digits;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn c2v_pg(d: usize) -> PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| [2u8, 3, 4, 1][k % 4]).collect();
    PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}

pub(crate) fn config_occ(dig: &[usize]) -> (Vec<bool>, Vec<bool>) {
    let occ_up = dig.iter().map(|&x| x == 1 || x == 3).collect();
    let occ_dn = dig.iter().map(|&x| x == 2 || x == 3).collect();
    (occ_up, occ_dn)
}

pub(crate) fn determinants_with_irrep(
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

pub(crate) fn capped_randn<T: Scalar>(
    space: QnSpace,
    lo: usize,
    hi: usize,
    seed: u64,
) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks = capped_ranks(&space, lo, hi, &mut rng);
    let x: TtVector<T> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    x
}

pub(crate) fn dense_apply_hamiltonian_12<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    vec: &[T],
) -> Vec<T> {
    let len = vec.len();
    let mut out = vec![T::zero(); len];
    let spins = [qntt::Spin::Up, qntt::Spin::Dn];
    let accumulate = |ops: &[(qntt::Orbital, bool)], coeff: T, out: &mut [T]| {
        let term = dense_apply_string(d, ops, vec);
        for (o, tm) in out.iter_mut().zip(term) {
            *o += tm * coeff;
        }
    };
    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c == T::zero() {
                continue;
            }
            for &s in &spins {
                accumulate(
                    &[
                        (qntt::Orbital::new(i, s), true),
                        (qntt::Orbital::new(j, s), false),
                    ],
                    c,
                    &mut out,
                );
            }
        }
    }
    let half = T::from_f64(0.5);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c == T::zero() {
                        continue;
                    }
                    for &s1 in &spins {
                        for &s2 in &spins {
                            let (oi, oj) = (qntt::Orbital::new(i, s1), qntt::Orbital::new(j, s1));
                            let (ok, ol) = (qntt::Orbital::new(k, s2), qntt::Orbital::new(l, s2));
                            if oi != ok && oj != ol {
                                accumulate(
                                    &[(oi, true), (ok, true), (ol, false), (oj, false)],
                                    c * half,
                                    &mut out,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub(crate) fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in a.iter().zip(b) {
        num += (x - y) * (x - y);
        den += y * y;
    }
    (num / den.max(1e-300)).sqrt()
}

pub(crate) fn basis_index(occ_up: &[bool], occ_dn: &[bool]) -> usize {
    let d = occ_up.len();
    let mut idx = 0usize;
    for k in (0..d).rev() {
        let fam = (occ_up[k] as usize) | ((occ_dn[k] as usize) << 1);
        idx = (idx << 2) | fam;
    }
    idx
}

pub(crate) fn assert_c2v_coeffs_nonvacuous(pg: &PointGroup, d: usize, t: &CoeffTensor<f64>) {
    assert!(pg.order() > 1, "fixture group must be nontrivial");
    assert!(
        (1..d).any(|k| pg.irrep_of(k) != pg.irrep_of(0)),
        "fixture group must be inhomogeneous across sites"
    );
    let any_shared_pair =
        (0..d).any(|i| (0..d).any(|j| i != j && pg.irrep_of(i) == pg.irrep_of(j)));
    if any_shared_pair {
        assert!(
            (0..d).any(|i| (0..d).any(|j| i != j && t.get(&[i, j]) != 0.0)),
            "vacuous: t has no allowed off-diagonal nonzero despite a same-irrep pair existing"
        );
    }
    assert!(
        (0..d).any(|i| (0..d).any(|j| pg.irrep_of(i) != pg.irrep_of(j) && t.get(&[i, j]) == 0.0)),
        "vacuous: t has no symmetry-forbidden (necessarily-zero) entry"
    );
}

#[allow(clippy::type_complexity)]
pub(crate) fn dense_target_block_ground_state(
    pg: &PointGroup,
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    nup: usize,
    ndn: usize,
    target: Irrep,
) -> (Vec<(Vec<bool>, Vec<bool>)>, f64) {
    let dets = determinants_with_irrep(pg, d, nup, ndn, target);
    let n = dets.len();
    assert!(n > 0, "empty target-irrep determinant block");

    let n_particle_block = (0..(1usize << (2 * d)))
        .filter(|&idx| config_counts(&digits(idx, d)) == (nup, ndn))
        .count();
    assert!(
        n < n_particle_block,
        "vacuous: target-irrep block ({n}) is the WHOLE particle block ({n_particle_block}) — \
         the group did not actually refine anything"
    );

    let len = 1usize << (2 * d);
    let mut hblock = Mat::<f64>::zeros(n, n);
    for (col, (ou, od)) in dets.iter().enumerate() {
        let mut e = vec![0.0f64; len];
        e[basis_index(ou, od)] = 1.0;
        let he = dense_apply_hamiltonian_12(d, t, v, &e);
        for (row, (ru, rd)) in dets.iter().enumerate() {
            *hblock.get_mut(row, col) = he[basis_index(ru, rd)];
        }
    }
    let mut asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            asym = asym.max((hblock.get(i, j) - hblock.get(j, i)).abs());
        }
    }
    assert!(
        asym < 1e-9,
        "fixture H_block is not Hermitian to 1e-9 (max asymmetry {asym}) — a c2v_coeffs draw bug"
    );

    let (evals, _u) = dense_self_adjoint_eigen(hblock.as_ref());
    (dets, evals[0])
}

pub(crate) fn g_a() -> (PointGroup, QnSpace) {
    let d = 6;
    let pg = c2v_pg(d);
    (pg, QnSpace::with_point_group(d, 2, 2, pg, Irrep::new(1)))
}

pub(crate) fn g_b() -> (PointGroup, QnSpace) {
    let d = 4;
    let pg = c2v_pg(d);
    (pg, QnSpace::with_point_group(d, 2, 2, pg, Irrep::new(2)))
}

pub(crate) fn g_c() -> (PointGroup, QnSpace) {
    let d = 6;
    let pg = c2v_pg(d);
    (pg, QnSpace::with_point_group(d, 2, 2, pg, Irrep::IDENTITY))
}
