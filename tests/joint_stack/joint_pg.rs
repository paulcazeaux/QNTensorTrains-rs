// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::util::capped_ranks;
use crate::util::load_h2o;
use qntt::rdm::transition_1rdm_bruteforce;
use qntt::rotation::{
    PairMask, expm_antisym, expm_kappa_blocked, g_pq, kappa_apply, kappa_operator,
    orbital_gradient, orbital_pairs, rotate_integrals,
};
use qntt::{
    CoeffTensor, Irrep, PointGroup, QnSpace, Scalar, SparseHamiltonian, TtVector,
    rayleigh_quotient, tt_randn,
};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

fn assert_close(got: f64, want: f64, tol: f64, msg: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{msg}: got {got:e}, want {want:e}, |diff|={:e} > tol {tol:e}",
        (got - want).abs()
    );
}

fn transpose_flat(v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = v[i * n + j];
        }
    }
    out
}

fn capped_randn<T: Scalar>(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks = capped_ranks(&space, lo, hi, &mut rng);
    let x: TtVector<T> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    x
}

fn same_irrep_kappa(pg: PointGroup, d: usize, seed: u64) -> CoeffTensor<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q) in &orbital_pairs(pg, d) {
        let v: f64 = rng.sample::<f64, _>(StandardNormal);
        k.set(&[p, q], v);
        k.set(&[q, p], -v);
    }
    k
}

#[test]
fn tjp_pairs_trivial_and_h2o_c2v() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let d = load.ints.norb;
    assert!(
        pg.order() > 1,
        "h2o_c2v must resolve to a nontrivial point group"
    );

    assert_eq!(
        orbital_pairs(PointGroup::TRIVIAL, d),
        PairMask::all(d).keep,
        "orbital_pairs at TRIVIAL must be bit-identical to the unfiltered strict lower triangle"
    );

    let pairs_c2v = orbital_pairs(pg, d);
    assert!(
        pairs_c2v.len() < PairMask::all(d).keep.len(),
        "h2o_c2v's C2v group must genuinely filter some pairs (non-vacuity guard)"
    );
    for &(p, q) in &pairs_c2v {
        assert_eq!(
            pg.irrep_of(p),
            pg.irrep_of(q),
            "orbital_pairs kept a cross-irrep pair ({p},{q})"
        );
    }

    let mut counts = std::collections::BTreeMap::<u8, usize>::new();
    for k in 0..d {
        *counts.entry(pg.irrep_of(k).index()).or_insert(0) += 1;
    }
    let analytic: usize = counts.values().map(|&n| n * n.saturating_sub(1) / 2).sum();
    assert_eq!(
        pairs_c2v.len(),
        analytic,
        "orbital_pairs count must equal Σ_Γ d_Γ(d_Γ-1)/2"
    );
}

#[test]
fn tjp_kappa_builds_same_irrep() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let space = load.space;
    let d = load.ints.norb;
    let kappa = same_irrep_kappa(pg, d, 0xB100);

    let op = kappa_operator(space, &kappa);
    let x0: TtVector<f64> = capped_randn(space, 1, 2, 0xB101);
    let y = &op * &x0;
    assert!(y.check());

    let mut reference: Option<TtVector<f64>> = None;
    for &(p, q) in &orbital_pairs(pg, d) {
        let kpq = kappa.get(&[p, q]);
        if kpq == 0.0 {
            continue;
        }
        let term = &g_pq(&x0, p, q) * kpq;
        reference = Some(match reference {
            None => term,
            Some(acc) => &acc + &term,
        });
    }
    let reference = reference.expect("nonzero same-irrep kappa draw");
    let y2 = kappa_apply(&kappa, &x0);
    let mut diff = &y2 - &reference;
    diff.round_mut(1e-14, qntt::Pooled::NONE, qntt::round::RoundMode::Rel);

    assert!(
        diff.norm() <= 1e-11 * reference.norm().max(1e-8),
        "kappa_apply vs Σ κ[p,q]·g_pq(x,p,q) over orbital_pairs(pg,d) under C2v: ‖diff‖={} ‖ref‖={}",
        diff.norm(),
        reference.norm()
    );
}

#[test]
#[should_panic(expected = "totally symmetric")]
fn tjp_kappa_cross_irrep_pair_panics_at_construction() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let space = load.space;
    let d = load.ints.norb;
    let (p, q) = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .find(|&(i, j)| pg.irrep_of(i) != pg.irrep_of(j))
        .expect("a |G| > 1 fixture has at least one cross-irrep pair");

    let mut kappa = CoeffTensor::<f64>::zeros(d, 2);
    kappa.set(&[p, q], 1.0);
    kappa.set(&[q, p], -1.0);
    let _ = kappa_operator(space, &kappa);
}

#[test]
#[should_panic(expected = "nonzero across irreps")]
fn tjp_expm_kappa_blocked_cross_irrep_kappa_panics() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let d = load.ints.norb;
    let (p, q) = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .find(|&(i, j)| pg.irrep_of(i) != pg.irrep_of(j))
        .expect("a |G| > 1 fixture has at least one cross-irrep pair");

    let mut kappa = CoeffTensor::<f64>::zeros(d, 2);
    kappa.set(&[p, q], 1.0);
    kappa.set(&[q, p], -1.0);
    let _ = expm_kappa_blocked(&kappa, pg);
}

#[test]
fn tjp_orbital_gradient_over_h2o_c2v_skips_cross_irrep_and_is_nonvacuous() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let space = load.space;
    let d = load.ints.norb;
    let ham = SparseHamiltonian::new(space, &load.ints.t, &load.ints.v, None, f64::EPSILON);
    let x0: TtVector<f64> = capped_randn(space, 1, 2, 0xB400);

    let g = orbital_gradient(&ham, &x0);
    let hx = ham.apply(&x0);
    let gamma = transition_1rdm_bruteforce(&x0, &hx);

    let mut max_cross_g = 0.0f64;
    let mut max_cross_gamma = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                max_cross_g = max_cross_g.max(g.get(&[p, q]).abs());
                max_cross_gamma = max_cross_gamma.max(gamma.get(&[p, q]).abs());
            }
        }
    }
    assert_eq!(
        max_cross_g, 0.0,
        "orbital_gradient must leave cross-irrep entries EXACTLY 0.0 under C2v"
    );
    assert_eq!(
        max_cross_gamma, 0.0,
        "transition_1rdm_bruteforce must leave cross-irrep entries EXACTLY 0.0 under C2v"
    );

    let mut max_g_same_irrep = 0.0f64;
    for &(p, q) in &orbital_pairs(pg, d) {
        max_g_same_irrep = max_g_same_irrep.max(g.get(&[p, q]).abs());
    }
    assert!(
        max_g_same_irrep > 1e-6,
        "anti-vacuity: orbital_gradient must be genuinely nonzero over orbital_pairs(pg,d) — got \
         max|g|={max_g_same_irrep:e}"
    );
}

#[test]
fn tjp_zero_cells_bitwise_and_anti_arm() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let space = load.space;
    let d = load.ints.norb;
    let kappa = same_irrep_kappa(pg, d, 0xB200);

    let u_blocked = expm_kappa_blocked(&kappa, pg);
    let rotated = rotate_integrals(&load.ints, &u_blocked);

    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of(i) != pg.irrep_of(j) {
                assert_eq!(
                    rotated.t.get(&[i, j]),
                    0.0,
                    "forbidden t' cell not bitwise zero: ({i},{j})"
                );
            }
        }
    }
    let mut forbidden_v_count = 0usize;
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                        forbidden_v_count += 1;
                        assert_eq!(
                            rotated.v.get(&[i, j, k, l]),
                            0.0,
                            "forbidden v' cell not bitwise zero: ({i},{j},{k},{l})"
                        );
                    }
                }
            }
        }
    }
    assert!(
        forbidden_v_count > 0,
        "vacuous: h2o_c2v carries no forbidden v cells"
    );

    let h = SparseHamiltonian::new(space, &rotated.t, &rotated.v, None, f64::EPSILON);

    let x0: TtVector<f64> = capped_randn(space, 1, 2, 0xB201);
    let h_before = SparseHamiltonian::new(space, &load.ints.t, &load.ints.v, None, f64::EPSILON);
    let rq_before = rayleigh_quotient(&h_before, &x0);
    let rq_rotated = rayleigh_quotient(&h, &x0);
    let u_back = transpose_flat(&u_blocked, d);
    let rotated_back = rotate_integrals(&rotated, &u_back);
    let h_back =
        SparseHamiltonian::new(space, &rotated_back.t, &rotated_back.v, None, f64::EPSILON);
    let rq_back = rayleigh_quotient(&h_back, &x0);
    assert_close(
        rq_back,
        rq_before,
        1e-8 * rq_before.abs().max(1.0),
        "rotate-then-rotate-back must be the identity on the Rayleigh quotient",
    );
    let _ = rq_rotated;

    let u_full = expm_antisym(&kappa.data, d);
    let rotated_full = rotate_integrals(&load.ints, &u_full);
    let mut max_forbidden = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of(i) != pg.irrep_of(j) {
                max_forbidden = max_forbidden.max(rotated_full.t.get(&[i, j]).abs());
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                        max_forbidden = max_forbidden.max(rotated_full.v.get(&[i, j, k, l]).abs());
                    }
                }
            }
        }
    }
    eprintln!(
        "tjp_zero_cells anti-arm: full-matrix expm_antisym route left max forbidden-cell residue \
         {max_forbidden:e} (blocked route leaves exactly 0.0 in the same cells)"
    );
    assert!(
        max_forbidden > 0.0,
        "anti-arm vacuous: the full-matrix route left EXACTLY zero residue in forbidden cells \
         (unexpected — this was supposed to demonstrate the hazard expm_kappa_blocked removes)"
    );
}

#[test]
fn tjp_blocked_eq_expm_antisym_on_same_irrep_kappa() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let d = load.ints.norb;
    let kappa = same_irrep_kappa(pg, d, 0xB300);

    let u_blocked = expm_kappa_blocked(&kappa, pg);
    let u_full = expm_antisym(&kappa.data, d);
    let maxdiff = u_blocked
        .iter()
        .zip(&u_full)
        .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
    assert!(
        maxdiff <= 1e-9,
        "expm_kappa_blocked vs expm_antisym on a same-irrep kappa must agree closely (the block \
         structure changes nothing mathematically): max|diff|={maxdiff:e}"
    );
}
