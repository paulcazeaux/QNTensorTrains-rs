// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg6::util::*;
use faer::Mat;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::{CoeffTensor, Irrep, PointGroup, PointGroupKind, Scalar};
use qntt_testkit::dense_apply_string;
use qntt_testkit::pg::c2v_coeffs;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn c2v_pg(d: usize) -> PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| [2u8, 3, 4, 1][k % 4]).collect();
    PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}
fn dense_apply_hamiltonian_12<T: Scalar>(
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
#[allow(clippy::type_complexity)]
fn dense_target_block_ground_state(
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
        "vacuous: target-irrep block is the WHOLE particle block"
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
        "fixture H_block is not Hermitian to 1e-9 (max asymmetry {asym})"
    );

    let (evals, _u) = dense_self_adjoint_eigen(hblock.as_ref());
    (dets, evals[0])
}

#[test]
fn pg6_sparse_oracle_matches_the_pg5_dense_oracle() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
    let pg = c2v_pg(d);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(76_100);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);

    let (dets_sparse, lam_sparse) =
        target_block_ground_state_sparse(&pg, d, &t, &v, nup, ndn, target);
    let (dets_dense, lam_dense) = dense_target_block_ground_state(&pg, d, &t, &v, nup, ndn, target);

    assert_eq!(
        dets_sparse, dets_dense,
        "the sparse and dense oracles must enumerate the IDENTICAL determinant set in the SAME \
         order (both delegate to `determinants_with_irrep`)"
    );
    assert!(
        (lam_sparse - lam_dense).abs() <= 1e-12 * lam_dense.abs().max(1.0),
        "the §4 sparse-oracle refactor must reproduce PG5's dense oracle: {lam_sparse} vs \
         {lam_dense}"
    );
}
