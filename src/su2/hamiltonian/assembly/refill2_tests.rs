// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::cell::Cell;
use std::sync::Arc;

use rand::RngExt;

use super::refill_tests::{
    assert_no_zeros, assert_recorded, assert_same_operator, bits, dense, full_class_count,
    h2o_space, h2o_typed, n_slots, testdata,
};
use super::{
    SRC_NONE, Su2Factor, Su2PatternMask, Su2TermScreen, alpha, for_each_term, median_weight,
};
use crate::C64;
use crate::coeff::{CoeffTensor, ThreeBody};
use crate::integrals::{Integrals, IntegralsOptions, load_fcidump, load_fixture_dir};
use crate::pointgroup::PointGroup;
use crate::rotation::{
    compose_rotations, expm_kappa_blocked, givens, identity_matrix, rotate_integrals,
};
use crate::scalar::Scalar;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::Su2Space;
use crate::test_util::{in_pool, seeded};

const EPS: f64 = f64::EPSILON;

const UNSPLIT: usize = 1 << 30;

fn sa_trivial() -> Su2TermScreen {
    Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL)
}

fn ints_of<T: Scalar>((t, v): (CoeffTensor<T>, CoeffTensor<T>)) -> Integrals<T> {
    let d = t.d;
    Integrals {
        t,
        v,
        e0: T::zero(),
        norb: d,
        nelec: 0,
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    }
}

fn lih_ints() -> Integrals<f64> {
    load_fixture_dir::<f64>(&testdata("lih_sto3g"), &IntegralsOptions::default())
        .expect("lih_sto3g loads")
        .ints
}

fn generic_u(d: usize, pg: PointGroup, scale: f64, seed: u64) -> Vec<f64> {
    let mut rng = seeded(seed);
    let mut kappa = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in 0..p {
            if pg.irrep_of(p) == pg.irrep_of(q) {
                let x: f64 = scale * rng.random_range(-1.0..1.0);
                kappa.set(&[p, q], x);
                kappa.set(&[q, p], -x);
            }
        }
    }
    expm_kappa_blocked(&kappa, pg)
}

fn mint<T: Scalar>(
    space: Su2Space,
    ints: &Integrals<T>,
    screen: &Su2TermScreen,
) -> Su2SparseHamiltonian<T> {
    Su2SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, screen.clone())
}

fn token_stream<T: Scalar>(
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> Vec<Vec<(usize, bool)>> {
    let mut out = Vec::new();
    for_each_term(t.d, t, v, None, screen, |tokens, _| {
        out.push(tokens.to_vec())
    });
    out
}

fn gamma_bits<T: Scalar>(h: &Su2SparseHamiltonian<T>, what: &str) -> Vec<(u64, u64)> {
    h.refill_gamma()
        .unwrap_or_else(|| panic!("{what}: a refillable build carries gamma"))
        .iter()
        .map(|&g| bits(g))
        .collect()
}

fn value_bits<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> Vec<(u64, u64)> {
    h.nests
        .iter()
        .flat_map(|n| n.cells.iter())
        .flat_map(|c| c.csr.vals.iter().chain(c.csc.vals.iter()))
        .map(|&x| bits(x))
        .collect()
}

fn assert_same_provenance<T: Scalar>(
    a: &Su2SparseHamiltonian<T>,
    b: &Su2SparseHamiltonian<T>,
    what: &str,
) {
    assert_eq!(a.n_terms(), b.n_terms(), "{what}: n_terms");
    let key = |f: &Su2Factor| (f.cw.to_bits(), f.fr.to_bits(), f.neg, f.transport);
    let (fa, fb) = (
        a.factors().expect("refillable: factors"),
        b.factors().expect("refillable: factors"),
    );
    assert!(
        fa.len() == fb.len() && fa.iter().map(key).eq(fb.iter().map(key)),
        "{what}: factor tables differ"
    );
    assert_eq!(a.nests.len(), b.nests.len(), "{what}: nest counts");
    for (k, (na, nb)) in a.nests.iter().zip(&b.nests).enumerate() {
        assert_eq!(na.cells.len(), nb.cells.len(), "{what}: site {k} cells");
        for (c, (ca, cb)) in na.cells.iter().zip(&nb.cells).enumerate() {
            let sa = ca.src.as_ref().expect("refillable: cell provenance");
            let sb = cb.src.as_ref().expect("refillable: cell provenance");
            assert_eq!(
                sa.slots(),
                sb.slots(),
                "{what}: site {k} cell {c} slot counts"
            );
            for p in 0..sa.slots() {
                assert!(
                    sa.run(p).eq(sb.run(p)),
                    "{what}: site {k} cell {c} slot {p}: runs {:?} vs {:?}",
                    sa.run(p).collect::<Vec<_>>(),
                    sb.run(p).collect::<Vec<_>>()
                );
            }
            for q in 0..sa.slots() {
                assert_eq!(
                    sa.csc_to_csr(q),
                    sb.csc_to_csr(q),
                    "{what}: site {k} cell {c} CSC slot {q} → CSR slot"
                );
            }
            assert_eq!(sa.lf, sb.lf, "{what}: site {k} cell {c} lf bytes");
            assert_eq!(sa.t16, sb.t16, "{what}: site {k} cell {c} term offsets");
            assert_eq!(
                sa.perm16, sb.perm16,
                "{what}: site {k} cell {c} u16 CSC permutation"
            );
            assert_eq!(
                sa.aux, sb.aux,
                "{what}: site {k} cell {c} u32 sections (block words, escape bitmap, fid words, \
                 u32 permutation, escape terms, spill index)"
            );
            assert_eq!(sa.spill, sb.spill, "{what}: site {k} cell {c} spill words");
            assert_eq!(
                (sa.n_fids, sa.spill_at),
                (sb.n_fids, sb.spill_at),
                "{what}: site {k} cell {c} layout header (fid table length, spill offset)"
            );
        }
    }
}

fn assert_refilled_is_fresh<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    fresh: &Su2SparseHamiltonian<T>,
    screen: &Su2TermScreen,
    what: &str,
) {
    assert!(
        h.is_refillable(),
        "{what}: the refill dropped refillability"
    );
    assert_eq!(
        h.term_screen(),
        Some(screen.clone()),
        "{what}: term_screen moved"
    );
    assert_recorded(h, what);
    assert_same_operator(h, fresh, what);
    assert_eq!(
        h.artifact_hash(),
        fresh.artifact_hash(),
        "{what}: artifact_hash ≠ the fresh build's"
    );
    assert_eq!(
        h.skeleton_hash(),
        fresh.skeleton_hash(),
        "{what}: skeleton_hash ≠ the fresh build's"
    );
    assert_same_provenance(h, fresh, what);
    assert_eq!(
        gamma_bits(h, what),
        gamma_bits(fresh, what),
        "{what}: the stored gamma is not the fresh build's Phase-A stream"
    );
    assert_eq!(h.src_hash(), fresh.src_hash(), "{what}: src_hash");
    assert_eq!(
        h.refill_census(),
        fresh.refill_census(),
        "{what}: census (debris_drops must survive; zero_values reads the new values)"
    );
}

fn expected_stats<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    old: &[(u64, u64)],
    new: &[(u64, u64)],
) -> (usize, usize) {
    assert_eq!(old.len(), new.len(), "expected_stats: stream lengths");
    let changed: Vec<bool> = old.iter().zip(new).map(|(a, b)| a != b).collect();
    let mut runs = 0usize;
    for c in h.nests.iter().flat_map(|n| n.cells.iter()) {
        let src = c.src.as_ref().expect("refillable: cell provenance");
        for p in 0..c.csr.nnz() {
            runs += usize::from(
                src.run(p)
                    .any(|(term, _)| term != SRC_NONE && changed[term as usize]),
            );
        }
    }
    (changed.iter().filter(|&&c| c).count(), 2 * runs)
}

fn run_kinds<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> (usize, usize) {
    let (mut gamma, mut transport) = (0usize, 0usize);
    for c in h.nests.iter().flat_map(|n| n.cells.iter()) {
        let src = c.src.as_ref().expect("refillable: cell provenance");
        for p in 0..c.csr.nnz() {
            if src.run(p).all(|(term, _)| term == SRC_NONE) {
                transport += 1;
            } else {
                gamma += 1;
            }
        }
    }
    (gamma, transport)
}

fn plus0_start_sensitive<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> usize {
    let gamma = h.refill_gamma().expect("refillable: gamma");
    let factors = h.factors().expect("refillable: factors");
    let mut n = 0usize;
    for c in h.nests.iter().flat_map(|n| n.cells.iter()) {
        let src = c.src.as_ref().expect("refillable: cell provenance");
        for p in 0..c.csr.nnz() {
            let alphas: Vec<T> = src
                .run(p)
                .map(|(term, fid)| {
                    let f = factors[fid as usize];
                    if term == SRC_NONE {
                        alpha(T::one(), f.neg, f.fr)
                    } else {
                        alpha(median_weight(gamma[term as usize], f.cw), f.neg, f.fr)
                    }
                })
                .collect();
            let mut s0 = T::zero();
            for &a in &alphas {
                s0 += a;
            }
            let mut s1 = alphas[0];
            for &a in &alphas[1..] {
                s1 += a;
            }
            n += usize::from(bits(s0) != bits(s1));
        }
    }
    n
}

fn assert_identity_refill<T: Scalar>(
    h: &mut Su2SparseHamiltonian<T>,
    ints: &Integrals<T>,
    what: &str,
) {
    let (vals, art, g) = (value_bits(h), h.artifact_hash(), gamma_bits(h, what));
    let st = h.refill_chunked(&ints.t, &ints.v, None, UNSPLIT);
    assert_eq!(
        (st.changed_terms, st.slots_written),
        (0, 0),
        "{what}: an identity refill must change no term and write no slot"
    );
    assert_eq!(
        value_bits(h),
        vals,
        "{what}: an identity refill changed a bit"
    );
    assert_eq!(h.artifact_hash(), art, "{what}: identity refill artifact");
    assert_eq!(gamma_bits(h, what), g, "{what}: identity refill gamma");
}

fn o1_arm<T: Scalar>(
    what: &str,
    space: Su2Space,
    from: &Integrals<T>,
    to: &Integrals<T>,
    screen: &Su2TermScreen,
) {
    let mut h = mint(space, from, screen);
    let (skel, art0, g0) = (h.skeleton_hash(), h.artifact_hash(), gamma_bits(&h, what));
    h.refill(&to.t, &to.v, None);
    let fresh = mint(space, to, screen);
    let changed = g0
        .iter()
        .zip(&gamma_bits(&fresh, what))
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "{what}: {changed} of {} terms changed; {} stored slots",
        g0.len(),
        n_slots(&h)
    );
    assert!(changed > 0, "{what}: vacuous — no term changed");
    assert_ne!(
        h.artifact_hash(),
        art0,
        "{what}: vacuous — the refill changed no value"
    );
    assert_eq!(
        h.skeleton_hash(),
        skel,
        "{what}: the refill moved the skeleton"
    );
    assert_refilled_is_fresh(&h, &fresh, screen, what);
}

#[test]
fn r2_o1_dense_refill_is_bitwise_a_fresh_build() {
    let d = 5;
    let src_f = ints_of(dense::<f64>(d, 0x520A_0001));
    let unrel_f = ints_of(dense::<f64>(d, 0x520A_0002));
    let src_c = ints_of(dense::<C64>(d, 0x520A_0003));
    let unrel_c = ints_of(dense::<C64>(d, 0x520A_0004));
    let u = generic_u(d, PointGroup::TRIVIAL, 0.3, 0x520A_0005);
    let rot_f = rotate_integrals(&src_f, &u);
    let rot_c = rotate_integrals(&src_c, &u);
    for (name, i) in [
        ("src", &src_f),
        ("unrelated", &unrel_f),
        ("rotated", &rot_f),
    ] {
        assert_no_zeros(&i.t, &i.v, &format!("f64 {name}"));
    }
    for (name, i) in [
        ("src", &src_c),
        ("unrelated", &unrel_c),
        ("rotated", &rot_c),
    ] {
        assert_no_zeros(&i.t, &i.v, &format!("C64 {name}"));
    }
    let mag = Su2TermScreen::Magnitude(EPS);
    let full = token_stream(&src_f.t, &src_f.v, &mag);
    assert_eq!(
        full.len(),
        full_class_count(d),
        "dense: every class admitted"
    );
    for (name, t, v) in [
        ("f64 rotated", &rot_f.t, &rot_f.v),
        ("f64 unrelated", &unrel_f.t, &unrel_f.v),
    ] {
        assert_eq!(token_stream(t, v, &mag), full, "{name}: Magnitude term set");
    }
    for (name, t, v) in [
        ("C64 src", &src_c.t, &src_c.v),
        ("C64 rotated", &rot_c.t, &rot_c.v),
        ("C64 unrelated", &unrel_c.t, &unrel_c.v),
    ] {
        assert_eq!(token_stream(t, v, &mag), full, "{name}: Magnitude term set");
    }
    for screen in [sa_trivial(), mag] {
        for (n, two_s) in [(3usize, 1usize), (4, 2)] {
            let space = Su2Space::new(d, n, two_s);
            for (target, to_f, to_c) in [
                ("rotated", &rot_f, &rot_c),
                ("unrelated", &unrel_f, &unrel_c),
            ] {
                let tag = format!("d=5 ({n},{two_s}) {screen:?} → {target}");
                o1_arm(&format!("f64 {tag}"), space, &src_f, to_f, &screen);
                o1_arm(&format!("C64 {tag}"), space, &src_c, to_c, &screen);
            }
        }
    }
}

#[test]
fn r2_o1_lih_refill_is_bitwise_a_fresh_build() {
    let lih = lih_ints();
    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x520B_0001));
    let sa = sa_trivial();
    let zeros = |t: &CoeffTensor<f64>, v: &CoeffTensor<f64>| {
        let mut z = 0usize;
        for_each_term(6, t, v, None, &sa, |_, w| z += usize::from(w == 0.0));
        z
    };
    assert_eq!(
        zeros(&lih.t, &lih.v),
        438,
        "lih: 438 zero classes (R0 census)"
    );
    assert_eq!(zeros(&rot.t, &rot.v), 0, "rotated lih: no zero class");
    for (n, two_s) in [(4usize, 0usize), (3, 1), (4, 2)] {
        o1_arm(
            &format!("lih (6,{n},{two_s})"),
            Su2Space::new(6, n, two_s),
            &lih,
            &rot,
            &sa,
        );
    }
}

#[test]
fn r2_o1_typed_h2o_refill_is_bitwise_a_fresh_build() {
    let (ints, pgd) = h2o_typed();
    let (pg, d) = (pgd.pg, ints.norb);
    let u = generic_u(d, pg, 0.1, 0x520C_0001);
    let mut mix = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert!(u[p * d + q] == 0.0, "u[{p}][{q}] crosses irreps");
            } else if p != q {
                mix = mix.max(u[p * d + q].abs());
            }
        }
    }
    assert!(
        mix > 1e-3,
        "vacuous: the blocked rotation does not mix ({mix:e})"
    );
    let rot = rotate_integrals(&ints, &u);
    let screen = Su2TermScreen::SymmetryAllowed(pg);
    for two_s in [0usize, 2] {
        o1_arm(
            &format!("h2o C2v 2S={two_s}"),
            h2o_space(&ints, &pgd, two_s),
            &ints,
            &rot,
            &screen,
        );
    }
}

fn chunk_pool_arm<T: Scalar>(what: &str, space: Su2Space, from: &Integrals<T>, to: &Integrals<T>) {
    let screen = sa_trivial();
    let fresh = mint(space, to, &screen);
    let mut reference = mint(space, from, &screen);
    reference.refill(&to.t, &to.v, None);
    assert_refilled_is_fresh(&reference, &fresh, &screen, &format!("{what} default"));

    let mut h = mint(space, from, &screen);
    let st = in_pool(1, || h.refill_chunked(&to.t, &to.v, None, UNSPLIT));
    let unsplit = (st.changed_terms, st.slots_written);
    assert_same_operator(&h, &reference, &format!("{what} unsplit, 1 thread"));
    assert!(unsplit.1 > 0, "{what}: vacuous — nothing written");

    let cells = n_cells(&reference);
    for chunk in [1usize, 3] {
        let tag = format!("{what} chunk {chunk}, 4 threads");
        let mut h = mint(space, from, &screen);
        let split = h
            .nests
            .iter()
            .flat_map(|n| n.cells.iter())
            .filter(|c| c.csr.nnz() > chunk)
            .count();
        println!("{tag}: {split} of {cells} cells split");
        assert!(
            split > 0,
            "{tag}: no cell splits — the split path is not exercised"
        );
        let st = in_pool(4, || h.refill_chunked(&to.t, &to.v, None, chunk));
        assert_same_operator(&h, &reference, &tag);
        assert_eq!(h.artifact_hash(), reference.artifact_hash(), "{tag}");
        assert_eq!(gamma_bits(&h, &tag), gamma_bits(&reference, &tag), "{tag}");
        assert_eq!(
            (st.changed_terms, st.slots_written),
            unsplit,
            "{tag}: the stats depend on the chunk / pool"
        );
    }
    for threads in [1usize, 4] {
        let tag = format!("{what} refill in a {threads}-thread pool");
        let mut h = mint(space, from, &screen);
        in_pool(threads, || h.refill(&to.t, &to.v, None));
        assert_same_operator(&h, &reference, &tag);
        assert_eq!(h.artifact_hash(), reference.artifact_hash(), "{tag}");
    }
}

fn n_cells<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> usize {
    h.nests.iter().map(|n| n.cells.len()).sum()
}

#[test]
fn r2_chunked_and_pooled_refills_are_bitwise_the_default() {
    let lih = lih_ints();
    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x520D_0001));
    chunk_pool_arm("lih (6,4,0)", Su2Space::new(6, 4, 0), &lih, &rot);
    let d = 5;
    let from = ints_of(dense::<C64>(d, 0x520D_0002));
    let to = ints_of(dense::<C64>(d, 0x520D_0003));
    chunk_pool_arm("C64 d=5 (4,2)", Su2Space::new(d, 4, 2), &from, &to);
}

#[test]
fn r2_chunked_partial_refill_is_bitwise_the_unsplit_one() {
    let lih = lih_ints();
    let space = Su2Space::new(6, 4, 0);
    let screen = sa_trivial();
    let rot = rotate_integrals(&lih, &givens(6, 2, 0.3));
    let what = "lih (6,4,0) Givens (2,3)";
    let fresh = mint(space, &rot, &screen);
    let mut reference = mint(space, &lih, &screen);
    let n_terms = reference.n_terms().expect("refillable: n_terms");
    let (old, new) = (gamma_bits(&reference, what), gamma_bits(&fresh, what));
    let changed: Vec<bool> = old.iter().zip(&new).map(|(a, b)| a != b).collect();
    let want = expected_stats(&reference, &old, &new);
    assert!(
        0 < want.0 && want.0 < n_terms,
        "{what}: fixture — a strict subset of the terms must move ({} of {n_terms})",
        want.0
    );
    reference.refill(&rot.t, &rot.v, None);
    assert_refilled_is_fresh(&reference, &fresh, &screen, &format!("{what} default"));

    let mut h = mint(space, &lih, &screen);
    let st = in_pool(1, || h.refill_chunked(&rot.t, &rot.v, None, UNSPLIT));
    let unsplit = (st.changed_terms, st.slots_written);
    assert_eq!(
        unsplit, want,
        "{what}: unsplit stats ≠ the independent count"
    );
    assert_same_operator(&h, &reference, &format!("{what} unsplit, 1 thread"));

    for chunk in [1usize, 3] {
        let tag = format!("{what} chunk {chunk}, 4 threads");
        let mut h = mint(space, &lih, &screen);
        let mixed = h
            .nests
            .iter()
            .flat_map(|n| n.cells.iter())
            .filter(|c| {
                let src = c.src.as_ref().expect("refillable: cell provenance");
                let written = |p: usize| {
                    src.run(p)
                        .any(|(term, _)| term != SRC_NONE && changed[term as usize])
                };
                let n_written = (0..c.csr.nnz()).filter(|&p| written(p)).count();
                c.csr.nnz() > chunk && 0 < n_written && n_written < c.csr.nnz()
            })
            .count();
        println!("{tag}: {mixed} split cells mix written and kept slots");
        assert!(
            mixed > 0,
            "{tag}: vacuous — no split cell mixes written and kept slots"
        );
        let st = in_pool(4, || h.refill_chunked(&rot.t, &rot.v, None, chunk));
        assert_same_operator(&h, &reference, &tag);
        assert_eq!(h.artifact_hash(), reference.artifact_hash(), "{tag}");
        assert_eq!(gamma_bits(&h, &tag), gamma_bits(&reference, &tag), "{tag}");
        assert_eq!(
            (st.changed_terms, st.slots_written),
            unsplit,
            "{tag}: the stats depend on the chunk / pool"
        );
    }
}

fn partial_arm<T: Scalar>(
    what: &str,
    space: Su2Space,
    ints: &Integrals<T>,
    pairs: &[usize],
    all_change: &Integrals<T>,
) {
    let d = space.d;
    let screen = sa_trivial();
    let mut h = mint(space, ints, &screen);
    let n_terms = h.n_terms().expect("refillable: n_terms");
    let total = 2 * n_slots(&h);
    let (gamma_runs, transport_runs) = run_kinds(&h);
    println!(
        "{what}: {n_terms} terms; {} stored runs ({gamma_runs} γ-bearing, {transport_runs} \
         transport-only); {total} value slots (CSR + CSC)",
        n_slots(&h)
    );
    assert!(
        transport_runs > 0,
        "{what}: vacuous — no transport-only run (\"never touched\" untested)"
    );
    assert_identity_refill(&mut h, ints, &format!("{what} at the mint"));

    let mut u = identity_matrix(d);
    for (step, &k) in pairs.iter().enumerate() {
        let tag = format!("{what} step {step}: Givens on pair ({k},{})", k + 1);
        u = compose_rotations(&u, &givens(d, k, 0.3 + 0.07 * step as f64), d);
        let rot = rotate_integrals(ints, &u);
        let fresh = mint(space, &rot, &screen);
        let want = expected_stats(&h, &gamma_bits(&h, &tag), &gamma_bits(&fresh, &tag));
        let st = h.refill_chunked(&rot.t, &rot.v, None, UNSPLIT);
        println!(
            "{tag}: changed terms {} of {n_terms}; slots written {} of {total}",
            st.changed_terms, st.slots_written
        );
        assert!(
            0 < st.changed_terms && st.changed_terms < n_terms,
            "{tag}: a one-pair rotation must move a STRICT subset of the terms ({} of {n_terms})",
            st.changed_terms
        );
        assert!(
            0 < st.slots_written && st.slots_written < total,
            "{tag}: a one-pair rotation must rewrite part of the slots ({} of {total})",
            st.slots_written
        );
        assert_eq!(
            (st.changed_terms, st.slots_written),
            want,
            "{tag}: (changed_terms, slots_written) ≠ (the bitwise gamma diff, 2 × the stored runs \
             referencing a changed term)"
        );
        assert_refilled_is_fresh(&h, &fresh, &screen, &tag);
        assert_identity_refill(&mut h, &rot, &tag);
    }

    let tag = format!("{what} all-change target");
    let fresh = mint(space, all_change, &screen);
    let want = expected_stats(&h, &gamma_bits(&h, &tag), &gamma_bits(&fresh, &tag));
    assert_eq!(
        want,
        (n_terms, 2 * gamma_runs),
        "{tag}: fixture — the target must move every term"
    );
    let st = h.refill_chunked(&all_change.t, &all_change.v, None, UNSPLIT);
    println!(
        "{tag}: changed terms {} of {n_terms}; slots written {} of {total}",
        st.changed_terms, st.slots_written
    );
    assert_eq!(
        (st.changed_terms, st.slots_written),
        (n_terms, 2 * gamma_runs),
        "{tag}: every term moved, so every γ-bearing run and no transport-only run is rewritten"
    );
    assert_refilled_is_fresh(&h, &fresh, &screen, &tag);
}

#[test]
fn r2_partial_refill_rewrites_exactly_the_runs_of_changed_terms() {
    let lih = lih_ints();
    let all = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x520E_0001));
    partial_arm(
        "lih (6,4,0)",
        Su2Space::new(6, 4, 0),
        &lih,
        &[2, 4, 0, 3, 1, 2],
        &all,
    );
    let d = 5;
    let src = ints_of(dense::<C64>(d, 0x520E_0002));
    let all = ints_of(dense::<C64>(d, 0x520E_0003));
    partial_arm(
        "C64 d=5 (4,2)",
        Su2Space::new(d, 4, 2),
        &src,
        &[2, 3, 1, 2, 0],
        &all,
    );
}

fn runs_with_term_not_first<T: Scalar>(h: &Su2SparseHamiltonian<T>, o: u32) -> usize {
    let mut n = 0usize;
    for c in h.nests.iter().flat_map(|n| n.cells.iter()) {
        let src = c.src.as_ref().expect("refillable: cell provenance");
        for p in 0..c.csr.nnz() {
            let run: Vec<(u32, u32)> = src.run(p).collect();
            n += usize::from(run[0].0 != o && run.iter().any(|&(term, _)| term == o));
        }
    }
    n
}

fn single_term_step<T: Scalar>(
    what: &str,
    space: Su2Space,
    h: &mut Su2SparseHamiltonian<T>,
    cur: &mut Integrals<T>,
    o: u32,
    edit: impl FnOnce(&mut Integrals<T>),
) {
    let screen = sa_trivial();
    let old = gamma_bits(h, what);
    edit(cur);
    let fresh = mint(space, cur, &screen);
    let new = gamma_bits(&fresh, what);
    assert!(
        old[o as usize] != new[o as usize],
        "{what}: fixture — term {o} did not move"
    );
    let want = expected_stats(h, &old, &new);
    assert_eq!(want.0, 1, "{what}: fixture — exactly one term must move");
    let st = h.refill_chunked(&cur.t, &cur.v, None, UNSPLIT);
    println!(
        "{what}: term {o}: changed terms {}, slots written {} (expected {want:?})",
        st.changed_terms, st.slots_written
    );
    assert_eq!(
        (st.changed_terms, st.slots_written),
        want,
        "{what}: (changed_terms, slots_written) ≠ (1, 2 × the stored runs referencing term {o})"
    );
    assert_refilled_is_fresh(h, &fresh, &screen, what);
}

#[test]
fn r2_single_term_refill_rewrites_every_run_that_references_it() {
    fn arm<T: Scalar>(what: &str, space: Su2Space, ints: Integrals<T>) {
        let d = space.d;
        let k = d - 1;
        let mut cur = ints;
        let mut h = mint(space, &cur, &sa_trivial());
        let o_t = (d * d - 1) as u32;
        let o_v = (full_class_count(d) - 1) as u32;
        let tokens = token_stream(&cur.t, &cur.v, &sa_trivial());
        let o_mid = tokens
            .iter()
            .position(|tok| tok[..] == [(0, true), (2, true), (3, false), (1, false)])
            .expect("the class (0,1),(2,3) is emitted") as u32;
        assert_eq!(tokens[o_t as usize], [(k, true), (k, false)], "{what}: o_t");
        assert_eq!(
            tokens[o_v as usize],
            [(k, true), (k, true), (k, false), (k, false)],
            "{what}: o_v"
        );
        let not_first = runs_with_term_not_first(&h, o_v);
        println!("{what}: {not_first} stored runs reference term {o_v} after another term");
        assert!(
            not_first > 0,
            "{what}: vacuous — term {o_v} always heads its runs"
        );
        let bump = T::from_f64(0.25);
        single_term_step(
            &format!("{what} v[k,k,k,k]"),
            space,
            &mut h,
            &mut cur,
            o_v,
            |i| i.v.set(&[k, k, k, k], i.v.get(&[k, k, k, k]) + bump),
        );
        single_term_step(
            &format!("{what} t[k,k]"),
            space,
            &mut h,
            &mut cur,
            o_t,
            |i| i.t.set(&[k, k], i.t.get(&[k, k]) + bump),
        );
        single_term_step(
            &format!("{what} v[0,1,2,3]"),
            space,
            &mut h,
            &mut cur,
            o_mid,
            |i| i.v.set(&[0, 1, 2, 3], i.v.get(&[0, 1, 2, 3]) + bump),
        );
    }
    let d = 5;
    arm(
        "f64 d=5 (3,1)",
        Su2Space::new(d, 3, 1),
        ints_of(dense::<f64>(d, 0x5211_0001)),
    );
    arm(
        "C64 d=5 (4,2)",
        Su2Space::new(d, 4, 2),
        ints_of(dense::<C64>(d, 0x5211_0002)),
    );
}

fn round_trip_arm<T: Scalar>(
    what: &str,
    space: Su2Space,
    ints: &Integrals<T>,
    rot: &Integrals<T>,
) -> usize {
    let screen = sa_trivial();
    let minted = mint(space, ints, &screen);
    let sensitive = plus0_start_sensitive(&minted);
    let mut h = mint(space, ints, &screen);
    h.refill(&rot.t, &rot.v, None);
    assert_ne!(
        h.artifact_hash(),
        minted.artifact_hash(),
        "{what}: vacuous — refill(Ũ) changed no value"
    );
    h.refill(&ints.t, &ints.v, None);
    println!("{what}: {sensitive} stored runs whose value depends on the +0 start");
    assert_refilled_is_fresh(&h, &minted, &screen, &format!("{what} round trip"));
    sensitive
}

#[test]
fn r2_round_trip_refill_restores_the_minted_operator() {
    let lih = lih_ints();
    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x520F_0001));
    for (n, two_s) in [(3usize, 1usize), (4, 0)] {
        let what = format!("lih (6,{n},{two_s})");
        let sensitive = round_trip_arm(&what, Su2Space::new(6, n, two_s), &lih, &rot);
        assert!(
            sensitive > 0,
            "{what}: vacuous — no stored run depends on the +0 start"
        );
    }
    let d = 5;
    let src = ints_of(dense::<C64>(d, 0x520F_0002));
    let rot = rotate_integrals(&src, &generic_u(d, PointGroup::TRIVIAL, 0.3, 0x520F_0003));
    round_trip_arm("C64 d=5 (4,2)", Su2Space::new(d, 4, 2), &src, &rot);
}

fn digest_arm<T: Scalar>(
    what: &str,
    space: Su2Space,
    from: &Integrals<T>,
    to: &Integrals<T>,
    screen: &Su2TermScreen,
) {
    let row = |build: &str, h: &Su2SparseHamiltonian<T>| {
        let src = h
            .src_hash()
            .map_or_else(|| "-".repeat(18), |x| format!("{x:#018x}"));
        println!(
            "r1c0-digest | {what:<22} | {build:<10} | src {src} | art {:#018x} | skel {:#018x}",
            h.artifact_hash(),
            h.skeleton_hash()
        );
    };
    let plain = Su2SparseHamiltonian::with_screen(space, &from.t, &from.v, None, screen.clone());
    row("plain", &plain);
    let mut h = mint(space, from, screen);
    row("mint", &h);
    let (src0, art0) = (h.src_hash(), h.artifact_hash());
    assert!(src0.is_some(), "{what}: a refillable build has a src_hash");
    h.refill(&to.t, &to.v, None);
    row("refill", &h);
    let fresh = mint(space, to, screen);
    row("fresh", &fresh);
    assert_eq!(
        h.src_hash(),
        fresh.src_hash(),
        "{what}: refilled src_hash ≠ the fresh mint's"
    );
    assert_eq!(
        h.artifact_hash(),
        fresh.artifact_hash(),
        "{what}: refilled artifact_hash ≠ the fresh mint's"
    );
    assert_ne!(h.src_hash(), src0, "{what}: vacuous — gamma did not move");
    assert_ne!(h.artifact_hash(), art0, "{what}: vacuous — no value moved");
}

#[test]
fn r1c0_logical_digest_table() {
    let sa = sa_trivial();
    let h4 = load_fcidump::<f64>(&testdata("h4.FCIDUMP"), None, &IntegralsOptions::default())
        .expect("h4.FCIDUMP loads")
        .ints;
    let h4_rot = rotate_integrals(&h4, &generic_u(4, PointGroup::TRIVIAL, 0.3, 0x521C_0001));
    for two_s in [0usize, 2] {
        digest_arm(
            &format!("h4 (4,4,{two_s})"),
            Su2Space::new(4, 4, two_s),
            &h4,
            &h4_rot,
            &sa,
        );
    }

    let lih = lih_ints();
    let lih_rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x521C_0002));
    for (n, two_s) in [(4usize, 0usize), (3, 1), (4, 2)] {
        digest_arm(
            &format!("lih (6,{n},{two_s})"),
            Su2Space::new(6, n, two_s),
            &lih,
            &lih_rot,
            &sa,
        );
    }

    let (h2o, pgd) = h2o_typed();
    let pg = pgd.pg;
    let h2o_rot = rotate_integrals(&h2o, &generic_u(h2o.norb, pg, 0.1, 0x521C_0003));
    let typed = Su2TermScreen::SymmetryAllowed(pg);
    for two_s in [0usize, 2] {
        digest_arm(
            &format!("h2o C2v 2S={two_s}"),
            h2o_space(&h2o, &pgd, two_s),
            &h2o,
            &h2o_rot,
            &typed,
        );
    }

    let d = 5;
    let u = generic_u(d, PointGroup::TRIVIAL, 0.3, 0x521C_0004);
    let src_f = ints_of(dense::<f64>(d, 0x521C_0005));
    let src_c = ints_of(dense::<C64>(d, 0x521C_0006));
    let (rot_f, rot_c) = (rotate_integrals(&src_f, &u), rotate_integrals(&src_c, &u));
    digest_arm(
        "dense f64 d=5 (3,1)",
        Su2Space::new(d, 3, 1),
        &src_f,
        &rot_f,
        &sa,
    );
    digest_arm(
        "dense C64 d=5 (4,2)",
        Su2Space::new(d, 4, 2),
        &src_c,
        &rot_c,
        &sa,
    );
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::refill: not a refillable build")]
fn r2_refusal_plain_build() {
    let d = 5;
    let (t, v) = dense::<f64>(d, 0x5210_0001);
    let mut h = Su2SparseHamiltonian::new(Su2Space::new(d, 3, 1), &t, &v, None, EPS);
    assert!(!h.is_refillable());
    h.refill(&t, &v, None);
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::refill: not a refillable build")]
fn r2_refusal_value_rule_recording_build() {
    let d = 5;
    let (t, v) = dense::<f64>(d, 0x5210_0002);
    let (t1, v1) = dense::<f64>(d, 0x5210_0003);
    let mut h = Su2SparseHamiltonian::with_screen_recorded(
        Su2Space::new(d, 3, 1),
        &t,
        &v,
        None,
        sa_trivial(),
    );
    assert!(
        !h.is_refillable() && h.refill_gamma().is_some(),
        "fixture: a recording, non-refillable build"
    );
    h.refill(&t1, &v1, None);
}

fn refill_with_three_body(screen: Su2TermScreen) {
    let d = 5;
    let (t, v) = dense::<f64>(d, 0x5210_0004);
    let w = CoeffTensor::<f64>::zeros(d, 6);
    let mut h = Su2SparseHamiltonian::new_refillable(Su2Space::new(d, 3, 1), &t, &v, None, screen);
    assert!(h.is_refillable());
    h.refill(&t, &v, Some(&w as &dyn ThreeBody<f64>));
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::refill: refuses a three-body channel")]
fn r2_refusal_three_body_under_symmetry_allowed() {
    refill_with_three_body(sa_trivial());
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::refill: refuses a three-body channel")]
fn r2_refusal_three_body_under_magnitude() {
    refill_with_three_body(Su2TermScreen::Magnitude(EPS));
}

#[test]
#[should_panic(expected = "one-body tensor must be rank-2 over d=")]
fn r2_refusal_one_body_shape_mismatch() {
    let d = 5;
    let (t, v) = dense::<f64>(d, 0x5210_0005);
    let (t_big, _) = dense::<f64>(d + 1, 0x5210_0006);
    let mut h =
        Su2SparseHamiltonian::new_refillable(Su2Space::new(d, 3, 1), &t, &v, None, sa_trivial());
    h.refill(&t_big, &v, None);
}

#[test]
#[should_panic(expected = "two-body tensor must be rank-4 over d=")]
fn r2_refusal_two_body_shape_mismatch() {
    let d = 5;
    let (t, v) = dense::<f64>(d, 0x5210_0007);
    let (_, v_big) = dense::<f64>(d + 1, 0x5210_0008);
    let mut h =
        Su2SparseHamiltonian::new_refillable(Su2Space::new(d, 3, 1), &t, &v, None, sa_trivial());
    h.refill(&t, &v_big, None);
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::refill: the Phase-A term count changed")]
fn r2_refusal_magnitude_term_count_change() {
    let d = 5;
    let (t1, v) = dense::<f64>(d, 0x5210_0009);
    let mut t0 = t1.clone();
    t0.set(&[1, 0], 0.0);
    let mag = Su2TermScreen::Magnitude(EPS);
    let (n0, n1) = (
        token_stream(&t0, &v, &mag).len(),
        token_stream(&t1, &v, &mag).len(),
    );
    assert_eq!(
        (n0, n1),
        (full_class_count(d) - 1, full_class_count(d)),
        "fixture: the count must grow by one"
    );
    let mut h = Su2SparseHamiltonian::new_refillable(Su2Space::new(d, 3, 1), &t0, &v, None, mag);
    h.refill(&t1, &v, None);
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::refill: the Phase-A term set changed")]
fn r2_refusal_magnitude_term_set_change_at_equal_count() {
    let d = 5;
    let (mut t0, v) = dense::<f64>(d, 0x5210_000A);
    t0.set(&[1, 0], 0.0);
    let mut t1 = t0.clone();
    t1.set(&[0, 1], 0.0);
    t1.set(&[1, 0], 0.25);
    let mag = Su2TermScreen::Magnitude(EPS);
    let (s0, s1) = (token_stream(&t0, &v, &mag), token_stream(&t1, &v, &mag));
    assert_eq!(
        s0.len(),
        full_class_count(d) - 1,
        "fixture: one class screened"
    );
    assert_eq!(
        s1.len(),
        s0.len(),
        "anti-vacuity: the term COUNT must not change (only the token-hash guard may fire)"
    );
    assert_ne!(s1, s0, "anti-vacuity: the term SET must change");
    let mut h = Su2SparseHamiltonian::new_refillable(Su2Space::new(d, 3, 1), &t0, &v, None, mag);
    assert!(h.is_refillable());
    h.refill(&t1, &v, None);
}

fn assert_refused_refill_is_inert(
    what: &str,
    h: &mut Su2SparseHamiltonian<f64>,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    expected: &str,
) {
    let (art, src, g) = (h.artifact_hash(), h.src_hash(), gamma_bits(h, what));
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| h.refill(t, v, None)))
        .expect_err(&format!("{what}: the refill was not refused"));
    let msg = err
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic payload>");
    assert!(
        msg.contains(expected),
        "{what}: refused by another guard: {msg}"
    );
    assert_eq!(
        h.artifact_hash(),
        art,
        "{what}: a refused refill moved the operator"
    );
    assert_eq!(
        h.src_hash(),
        src,
        "{what}: a refused refill moved the provenance"
    );
    assert_eq!(
        gamma_bits(h, what),
        g,
        "{what}: a refused refill moved the stored gamma"
    );
}

#[test]
fn r2_refused_refill_leaves_the_operator_untouched() {
    let d = 5;
    let space = Su2Space::new(d, 3, 1);
    let mag = Su2TermScreen::Magnitude(EPS);

    let (t1, v) = dense::<f64>(d, 0x5210_000B);
    let mut t0 = t1.clone();
    t0.set(&[1, 0], 0.0);
    assert_eq!(
        (
            token_stream(&t0, &v, &mag).len(),
            token_stream(&t1, &v, &mag).len()
        ),
        (full_class_count(d) - 1, full_class_count(d)),
        "fixture: the count must grow by one"
    );
    let mut h = Su2SparseHamiltonian::new_refillable(space, &t0, &v, None, mag.clone());
    assert_refused_refill_is_inert(
        "count guard",
        &mut h,
        &t1,
        &v,
        "Su2SparseHamiltonian::refill: the Phase-A term count changed",
    );

    let (mut t0, v) = dense::<f64>(d, 0x5210_000C);
    t0.set(&[1, 0], 0.0);
    let mut t1 = t0.clone();
    t1.set(&[0, 1], 0.0);
    t1.set(&[1, 0], 0.25);
    let (s0, s1) = (token_stream(&t0, &v, &mag), token_stream(&t1, &v, &mag));
    assert_eq!(
        s1.len(),
        s0.len(),
        "anti-vacuity: the term COUNT must not change"
    );
    assert_ne!(s1, s0, "anti-vacuity: the term SET must change");
    let mut h = Su2SparseHamiltonian::new_refillable(space, &t0, &v, None, mag);
    assert_refused_refill_is_inert(
        "token-hash guard",
        &mut h,
        &t1,
        &v,
        "Su2SparseHamiltonian::refill: the Phase-A term set changed",
    );
}

fn token_hashes_during<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let n0 = super::TOKEN_HASHES.with(Cell::get);
    let r = f();
    (r, super::TOKEN_HASHES.with(Cell::get) - n0)
}

fn r1ch_arm(
    what: &str,
    space: Su2Space,
    from: &Integrals<f64>,
    to: &Integrals<f64>,
    screen: &Su2TermScreen,
    hashed: bool,
) {
    let want = usize::from(hashed);
    let (mut h, at_mint) = token_hashes_during(|| mint(space, from, screen));
    assert_eq!(at_mint, want, "{what}: token hashes at mint");
    assert_eq!(
        h.skeleton.as_ref().expect("refillable").term_hash.is_some(),
        hashed,
        "{what}: the stored token hash"
    );
    let art0 = h.artifact_hash();
    let ((), at_refill) = token_hashes_during(|| h.refill(&to.t, &to.v, None));
    assert_eq!(at_refill, want, "{what}: token hashes in the refill");
    assert_ne!(
        h.artifact_hash(),
        art0,
        "{what}: vacuous — the refill changed no value"
    );
    assert_refilled_is_fresh(&h, &mint(space, to, screen), screen, what);
}

#[test]
fn r1ch_token_hash_is_computed_under_magnitude_only() {
    let d = 5;
    let space = Su2Space::new(d, 3, 1);
    let src = ints_of(dense::<f64>(d, 0x51C8_0001));
    let rot = rotate_integrals(&src, &generic_u(d, PointGroup::TRIVIAL, 0.3, 0x51C8_0002));
    assert_no_zeros(&src.t, &src.v, "src");
    assert_no_zeros(&rot.t, &rot.v, "rotated");

    r1ch_arm("SymmetryAllowed", space, &src, &rot, &sa_trivial(), false);

    let mut t0 = src.t.clone();
    t0.set(&[1, 0], 0.0);
    let src0 = ints_of((t0, src.v.clone()));
    let m = Su2PatternMask::from_magnitude(&src0.t, &src0.v, EPS, PointGroup::TRIVIAL);
    assert_eq!(
        m.n_classes(),
        full_class_count(d) - 1,
        "fixture: the mask must be a proper subset"
    );
    let mask = Su2TermScreen::Mask(Arc::new(m));
    r1ch_arm("Mask", space, &src0, &rot, &mask, false);

    let mag = Su2TermScreen::Magnitude(EPS);
    assert_eq!(
        token_stream(&rot.t, &rot.v, &mag),
        token_stream(&src.t, &src.v, &mag),
        "fixture: the Magnitude term set must not move (else a guard refuses the refill)"
    );
    r1ch_arm("Magnitude", space, &src, &rot, &mag, true);
}
