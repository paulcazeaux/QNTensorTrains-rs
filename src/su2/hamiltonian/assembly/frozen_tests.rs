// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use rand::RngExt;

use super::refill_tests::{
    assert_no_zeros, assert_plain, assert_recorded, assert_same_operator, bits, dense,
    full_class_count, h2o_space, h2o_typed, n_slots, recompute_all, testdata,
};
use super::{
    SRC_NONE, Su2Factor, Su2TermScreen, alpha, for_each_term, frozen_run_cancels, median_weight,
};
use crate::C64;
use crate::coeff::{CoeffTensor, ThreeBody};
use crate::integrals::{Integrals, IntegralsOptions, load_fixture_dir};
use crate::mem::Footprint;
use crate::pointgroup::PointGroup;
use crate::rotation::{expm_kappa_blocked, rotate_integrals};
use crate::scalar::Scalar;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, Ts};
use crate::su2::qn::{Su2Sector, Su2Space};
use crate::su2::vector::Su2TtVector;
use crate::test_util::{max_err_over_two_sz, randomized_su2_state, seeded};

const EPS: f64 = f64::EPSILON;

fn sa_trivial() -> Su2TermScreen {
    Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL)
}

fn refillable<T: Scalar>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> Su2SparseHamiltonian<T> {
    Su2SparseHamiltonian::new_refillable(space, t, v, None, screen.clone())
}

fn value_build<T: Scalar>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> Su2SparseHamiltonian<T> {
    Su2SparseHamiltonian::with_screen(space, t, v, None, screen.clone())
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

fn phase_a_census<T: Scalar>(
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> (usize, usize) {
    let (mut n, mut z) = (0usize, 0usize);
    for_each_term(t.d, t, v, None, screen, |_, w| {
        n += 1;
        z += usize::from(w.abs_s() == 0.0);
    });
    (n, z)
}

fn census<T: Scalar>(h: &Su2SparseHamiltonian<T>, what: &str) -> [usize; 5] {
    let c = h
        .refill_census()
        .unwrap_or_else(|| panic!("{what}: a refillable build must carry its census"));
    [
        c.multi_contrib_runs,
        c.multi_term_runs,
        c.transport_term_runs,
        c.debris_drops,
        c.factors,
    ]
}

fn print_census<T: Scalar>(h: &Su2SparseHamiltonian<T>, what: &str) -> [usize; 5] {
    let c = census(h, what);
    println!(
        "{what}: census — runs > 1 contribution {}, runs > 1 term {}, transport+term runs {}, \
         debris drops {}, factors {}, stored +0 values {}; stored slots {}",
        c[0],
        c[1],
        c[2],
        c[3],
        c[4],
        zero_values(h, what).0,
        n_slots(h)
    );
    c
}

fn zero_values<T: Scalar>(h: &Su2SparseHamiltonian<T>, what: &str) -> (usize, usize) {
    let census = h
        .refill_census()
        .unwrap_or_else(|| panic!("{what}: a refillable build must carry its census"))
        .zero_values;
    let plus0 = bits(T::zero());
    let recount = h
        .nests
        .iter()
        .flat_map(|n| n.cells.iter())
        .flat_map(|c| c.csr.vals.iter())
        .filter(|&&x| bits(x) == plus0)
        .count();
    (census, recount)
}

fn assert_census_matches(
    c: [usize; 5],
    zv: (usize, usize),
    cen: &super::refill_tests::Census,
    what: &str,
) {
    assert_eq!(
        zv.0, zv.1,
        "{what}: census zero_values ≠ the stored values exactly +0"
    );
    assert_eq!(
        c[0], cen.spilled,
        "{what}: census multi_contrib_runs ≠ the runs of > 1 contribution"
    );
    assert_eq!(
        c[1], cen.multi_term,
        "{what}: census multi_term_runs ≠ the runs of > 1 term"
    );
    assert_eq!(
        c[2], cen.transport_and_term,
        "{what}: census transport_term_runs ≠ the transport+term runs"
    );
    assert_eq!(
        c[4], cen.factors,
        "{what}: census factors ≠ factors().len()"
    );
}

fn prov_structure_hash<T: Scalar>(h: &Su2SparseHamiltonian<T>, what: &str) -> u64 {
    let mut s = rustc_hash::FxHasher::default();
    h.n_terms()
        .unwrap_or_else(|| panic!("{what}: no n_terms"))
        .hash(&mut s);
    for f in h
        .factors()
        .unwrap_or_else(|| panic!("{what}: no factor table"))
    {
        f.cw.to_bits().hash(&mut s);
        f.fr.to_bits().hash(&mut s);
        f.neg.hash(&mut s);
        f.transport.hash(&mut s);
    }
    for n in &h.nests {
        n.cells.len().hash(&mut s);
        for c in &n.cells {
            c.src
                .as_ref()
                .unwrap_or_else(|| panic!("{what}: a refillable cell without provenance"))
                .hash_into(&mut s);
        }
    }
    s.finish()
}

type EntryKey = (usize, usize, u8, u32, i32, i32);

fn entry_map<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> BTreeMap<EntryKey, (u64, u64)> {
    let mut m = BTreeMap::new();
    for (k, n) in h.nests.iter().enumerate() {
        for o in 0..n.offsets.len() - 1 {
            let (lo, hi) = (n.offsets[o] as usize, n.offsets[o + 1] as usize);
            for c in &n.cells[lo..hi] {
                for (r, col, x) in c.csr.entries() {
                    let prev = m.insert((k, o, c.x_occ, c.m_ord, r, col), bits(x));
                    assert!(prev.is_none(), "duplicate entry at site {k} slot {o}");
                }
            }
        }
    }
    m
}

#[derive(Debug, Default)]
struct Agreement {
    common: usize,
    extras: usize,
    frozen_only_nonzero: Vec<EntryKey>,
    value_only: Vec<EntryKey>,
}

impl Agreement {
    fn disagreements(&self) -> usize {
        self.frozen_only_nonzero.len() + self.value_only.len()
    }
}

fn compare_with_value_build<T: Scalar>(
    frozen: &Su2SparseHamiltonian<T>,
    value: &Su2SparseHamiltonian<T>,
    what: &str,
) -> Agreement {
    assert!(frozen.space == value.space, "{what}: spaces differ");
    assert!(
        frozen.states == value.states,
        "{what}: the refillable and value builds' vertex lists differ (Phases A/B read no values)"
    );
    assert!(
        frozen.state_qns == value.state_qns,
        "{what}: the refillable and value builds' coupled x-sector lists differ"
    );
    assert_eq!(frozen.nests.len(), value.nests.len(), "{what}: nest counts");
    for (k, (a, b)) in frozen.nests.iter().zip(&value.nests).enumerate() {
        assert_eq!(a.sectors, b.sectors, "{what}: site {k} sector lists");
        assert_eq!(
            a.offsets.len(),
            b.offsets.len(),
            "{what}: site {k} slot counts"
        );
    }
    let (fm, vm) = (entry_map(frozen), entry_map(value));
    let plus0 = bits(T::zero());
    let mut ag = Agreement::default();
    for (key, fb) in &fm {
        match vm.get(key) {
            Some(vb) => {
                assert_eq!(
                    fb, vb,
                    "{what}: entry {key:?} present in both builds with different bits"
                );
                ag.common += 1;
            }
            None if *fb == plus0 => ag.extras += 1,
            None => ag.frozen_only_nonzero.push(*key),
        }
    }
    ag.value_only = vm.keys().filter(|k| !fm.contains_key(k)).copied().collect();
    println!(
        "{what}: vs the value build — common {}, extras (exactly +0) {}, disagreements {} \
         (frozen-only nonzero {}, value-only {}); first: {:?} / {:?}",
        ag.common,
        ag.extras,
        ag.disagreements(),
        ag.frozen_only_nonzero.len(),
        ag.value_only.len(),
        ag.frozen_only_nonzero.first(),
        ag.value_only.first()
    );
    ag
}

fn zero_weight_storage<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> (usize, usize) {
    let gamma = h.refill_gamma().expect("refillable: gamma");
    let plus0 = bits(T::zero());
    let (mut zero_contribs, mut plus0_vals) = (0usize, 0usize);
    for n in &h.nests {
        for c in &n.cells {
            let src = c.src.as_ref().expect("refillable: cell provenance");
            for p in 0..c.csr.nnz() {
                for (term, _) in src.run(p) {
                    if term != SRC_NONE && gamma[term as usize].abs_s() == 0.0 {
                        zero_contribs += 1;
                    }
                }
                plus0_vals += usize::from(bits(c.csr.vals[p]) == plus0);
            }
        }
    }
    (zero_contribs, plus0_vals)
}

fn artifact_arm<T: Scalar>(what: &str, space: Su2Space, t: &CoeffTensor<T>, v: &CoeffTensor<T>) {
    let plain = Su2SparseHamiltonian::new(space, t, v, None, EPS);
    assert_plain(&plain, what);
    assert!(!plain.is_refillable(), "{what}: new() reports refillable");
    assert_eq!(plain.term_screen(), None, "{what}: new() reports a screen");
    assert!(
        plain.refill_census().is_none(),
        "{what}: new() carries a refill census"
    );
    assert!(
        n_slots(&plain) > 0,
        "{what}: empty operator (vacuous comparison)"
    );
    let screens = [sa_trivial(), Su2TermScreen::Magnitude(EPS)];
    for screen in &screens {
        let tag = format!("{what} new_refillable({screen:?})");
        let h = refillable(space, t, v, screen);
        assert!(h.is_refillable(), "{tag}: not refillable");
        assert_eq!(h.term_screen(), Some(screen.clone()), "{tag}: term_screen");
        assert_recorded(&h, &tag);
        assert_eq!(
            h.n_terms(),
            Some(full_class_count(space.d)),
            "{tag}: the dense fixture must admit every class under both screens"
        );
        print_census(&h, &tag);
        assert_eq!(
            zero_values(&h, &tag),
            (0, 0),
            "{tag}: a dense build stores no +0 value (the plain build never stores a zero)"
        );
        assert_same_operator(&h, &plain, &tag);
        assert_eq!(
            h.artifact_hash(),
            plain.artifact_hash(),
            "{tag}: artifact_hash ≠ new(EPS)"
        );
        assert_eq!(
            h.skeleton_hash(),
            plain.skeleton_hash(),
            "{tag}: skeleton_hash ≠ new(EPS)"
        );
    }
}

#[test]
fn r1b_a_refillable_builds_are_the_plain_build_on_dense_values() {
    let d = 5;
    let (tf, vf) = dense::<f64>(d, 0x51B1_A001);
    let (tc, vc) = dense::<C64>(d, 0x51B1_A002);
    assert_no_zeros(&tf, &vf, "f64 d=5");
    assert_no_zeros(&tc, &vc, "C64 d=5");
    for (n, two_s) in [(4usize, 0usize), (3, 1), (4, 2)] {
        let space = Su2Space::new(d, n, two_s);
        artifact_arm(&format!("f64 d=5 ({n},{two_s})"), space, &tf, &vf);
        artifact_arm(&format!("C64 d=5 ({n},{two_s})"), space, &tc, &vc);
    }
}

#[derive(Debug, PartialEq)]
struct Structure {
    skeleton: u64,
    provenance: u64,
    census: [usize; 5],
    slots: usize,
}

fn structure<T: Scalar>(
    what: &str,
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
) -> Structure {
    let h = refillable(space, t, v, &sa_trivial());
    assert_recorded(&h, what);
    let census = print_census(&h, what);
    Structure {
        skeleton: h.skeleton_hash(),
        provenance: prov_structure_hash(&h, what),
        census,
        slots: n_slots(&h),
    }
}

#[test]
fn r1b_b_skeleton_hash_is_value_independent() {
    let d = 6;
    let (tf, vf) = dense::<f64>(d, 0x51B1_B101);
    let (tc, vc) = dense::<C64>(d, 0x51B1_B102);
    assert_no_zeros(&tf, &vf, "f64 d=6");
    assert_no_zeros(&tc, &vc, "C64 d=6");
    let lih = lih_ints();
    let rot = rotate_integrals(&lih, &generic_u(d, PointGroup::TRIVIAL, 0.3, 0x51B1_B103));
    let sa = sa_trivial();
    assert_eq!(
        phase_a_census(&lih.t, &lih.v, &sa),
        (full_class_count(d), 438),
        "lih: 702 classes, 438 at weight exactly 0 (R0 census)"
    );
    assert_eq!(
        phase_a_census(&rot.t, &rot.v, &sa),
        (full_class_count(d), 0),
        "rotated lih: a generic rotation leaves no zero class"
    );
    for (n, two_s) in [(4usize, 0usize), (3, 1)] {
        let space = Su2Space::new(d, n, two_s);
        let tag = format!("d=6 ({n},{two_s})");
        let reference = structure(&format!("{tag} dense f64"), space, &tf, &vf);
        let others = [
            (
                "dense C64",
                structure(&format!("{tag} dense C64"), space, &tc, &vc),
            ),
            (
                "lih",
                structure(&format!("{tag} lih"), space, &lih.t, &lih.v),
            ),
            (
                "lih rotated",
                structure(&format!("{tag} lih rotated"), space, &rot.t, &rot.v),
            ),
        ];
        for (name, s) in &others {
            assert_eq!(
                s, &reference,
                "{tag} {name}: the refillable build's structure moved with the values \
                 (skeleton / provenance / census / slots)"
            );
        }
        let vd = value_build(space, &tf, &vf, &sa);
        let vl = value_build(space, &lih.t, &lih.v, &sa);
        println!(
            "{tag}: value build slots dense {} / lih {}; frozen {}",
            n_slots(&vd),
            n_slots(&vl),
            reference.slots
        );
        assert_ne!(
            vd.skeleton_hash(),
            vl.skeleton_hash(),
            "{tag}: vacuous — the value build's skeleton does not move between dense and lih"
        );
        assert!(
            n_slots(&vl) < n_slots(&vd),
            "{tag}: the value build of lih must store fewer entries (its α = ±0 runs skipped)"
        );
        assert_eq!(
            reference.skeleton,
            vd.skeleton_hash(),
            "{tag}: on dense values the frozen skeleton must be the value build's"
        );
    }
}

type LocatedRun = (usize, usize, Vec<(u32, u32)>);

fn merged_runs<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    k: usize,
    o_t: u32,
    o_v: u32,
) -> Vec<LocatedRun> {
    let mut out = Vec::new();
    for (ci, c) in h.nests[k].cells.iter().enumerate() {
        let src = c.src.as_ref().expect("refillable: cell provenance");
        for p in 0..c.csr.nnz() {
            let run: Vec<(u32, u32)> = src.run(p).collect();
            let mut terms: Vec<u32> = run.iter().map(|&(term, _)| term).collect();
            terms.sort_unstable();
            terms.dedup();
            if terms == [o_t, o_v] {
                out.push((ci, p, run));
            }
        }
    }
    out
}

fn group_sum(run: &[(u32, u32)], o: u32, factors: &[Su2Factor]) -> f64 {
    run.iter()
        .filter(|&&(term, _)| term == o)
        .map(|&(_, fid)| {
            let f = factors[fid as usize];
            (f.cw * if f.neg { -1.0 } else { 1.0 }) * f.fr
        })
        .sum()
}

fn run_value<T: Scalar>(h: &Su2SparseHamiltonian<T>, run: &[(u32, u32)]) -> (T, f64) {
    let gamma = h.refill_gamma().expect("refillable: gamma");
    let factors = h.factors().expect("refillable: factors");
    let (mut s, mut runmax) = (T::zero(), 0.0f64);
    for &(term, fid) in run {
        let f = factors[fid as usize];
        let a = if term == SRC_NONE {
            alpha(T::one(), f.neg, f.fr)
        } else {
            alpha(median_weight(gamma[term as usize], f.cw), f.neg, f.fr)
        };
        runmax = runmax.max(a.abs_s());
        s += a;
    }
    (s, runmax)
}

fn tuned_arm<T: Scalar>(what: &str, space: Su2Space, t: &CoeffTensor<T>, v: &CoeffTensor<T>) {
    let d = space.d;
    let k = d - 1;
    let sa = sa_trivial();
    let o_t = (d * d - 1) as u32;
    let o_v = (full_class_count(d) - 1) as u32;
    let half = T::from_f64(0.5);
    let vw = v.get(&[k, k, k, k]) * half;

    let f0 = refillable(space, t, v, &sa);
    let gamma = f0.refill_gamma().expect("refillable: gamma");
    assert_eq!(gamma.len(), full_class_count(d), "{what}: n_terms");
    assert_eq!(
        bits(gamma[o_t as usize]),
        bits(t.get(&[k, k])),
        "{what}: term {o_t} is not t[d-1,d-1]"
    );
    assert_eq!(
        bits(gamma[o_v as usize]),
        bits(vw),
        "{what}: term {o_v} is not v[d-1,d-1,d-1,d-1]/2"
    );
    let merged = merged_runs(&f0, k, o_t, o_v);
    assert_eq!(
        merged.len(),
        1,
        "{what}: expected exactly one stored run at site {k} merging terms {o_t} and {o_v} \
         (the bond-d collapse); found {merged:?}"
    );
    let (ci, p, run) = merged[0].clone();
    let factors = f0.factors().expect("refillable: factors");
    let (a, b) = (group_sum(&run, o_t, factors), group_sum(&run, o_v, factors));
    println!("{what}: merged run at site {k} cell {ci} slot {p}: {run:?}; groups A = {a}, B = {b}");
    assert!(
        a != 0.0 && b != 0.0,
        "{what}: a term group of the merged run cancels structurally (A = {a}, B = {b})"
    );
    assert!(
        !frozen_run_cancels(&run, factors),
        "{what}: the frozen V3 predicate drops the merged run"
    );

    let mut tt = t.clone();
    tt.set(&[k, k], vw * T::from_f64(-b / a));

    let f1 = refillable(space, &tt, v, &sa);
    let merged1 = merged_runs(&f1, k, o_t, o_v);
    assert_eq!(
        merged1.len(),
        1,
        "{what} tuned: the frozen build must still store the merged run"
    );
    assert_eq!(
        (merged1[0].0, merged1[0].1),
        (ci, p),
        "{what} tuned: the merged run moved"
    );
    let (s, runmax) = run_value(&f1, &merged1[0].2);
    let stored = f1.nests[k].cells[ci].csr.vals[p];
    println!(
        "{what} tuned: stored ({:e}, {:e}), |s| = {:e} vs the value rule's 16·eps·runmax = {:e}",
        stored.re_s(),
        stored.im_s(),
        s.abs_s(),
        16.0 * EPS * runmax
    );
    assert_eq!(
        bits(stored),
        bits(s),
        "{what} tuned: the stored value is not its run's recomputation"
    );
    assert!(
        s.abs_s() <= 16.0 * EPS * runmax && runmax > 0.0,
        "{what} tuned: the tuning does not cancel the merged run (|s| = {:e}, runmax {runmax:e})",
        s.abs_s()
    );

    let v0 = value_build(space, t, v, &sa);
    let v1 = value_build(space, &tt, v, &sa);
    let site_slots = |h: &Su2SparseHamiltonian<T>| -> usize {
        h.nests[k].cells.iter().map(|c| c.csr.nnz()).sum()
    };
    println!(
        "{what}: value build slots {} → {} (site {k}: {} → {})",
        n_slots(&v0),
        n_slots(&v1),
        site_slots(&v0),
        site_slots(&v1)
    );
    assert_ne!(
        v1.skeleton_hash(),
        v0.skeleton_hash(),
        "{what}: vacuous — the value build's skeleton did not move under the tuning"
    );
    assert_eq!(
        (n_slots(&v1) + 1, site_slots(&v1) + 1),
        (n_slots(&v0), site_slots(&v0)),
        "{what}: the value build must lose exactly the merged run at site {k}"
    );

    assert_eq!(
        f1.skeleton_hash(),
        f0.skeleton_hash(),
        "{what}: the refillable skeleton moved under the tuning"
    );
    assert_eq!(
        prov_structure_hash(&f1, what),
        prov_structure_hash(&f0, what),
        "{what}: the refillable provenance structure moved under the tuning"
    );
    assert_eq!(
        census(&f1, what),
        census(&f0, what),
        "{what}: the census moved under the tuning"
    );

    let ag0 = compare_with_value_build(&f0, &v0, &format!("{what} untuned"));
    let ag1 = compare_with_value_build(&f1, &v1, &format!("{what} tuned"));
    assert!(
        ag0.value_only.is_empty() && ag1.value_only.is_empty(),
        "{what}: the value build stores an entry the refillable build does not"
    );
    assert_eq!(
        ag1.extras + ag1.frozen_only_nonzero.len(),
        ag0.extras + ag0.frozen_only_nonzero.len() + 1,
        "{what}: the tuning must add exactly one refillable-only entry"
    );
}

#[test]
fn r1b_b_tuned_cancellation_moves_the_value_skeleton_not_the_frozen_one() {
    let d = 6;
    let (tf, vf) = dense::<f64>(d, 0x51B1_B001);
    let (tc, vc) = dense::<C64>(d, 0x51B1_B002);
    for (n, two_s) in [(4usize, 0usize), (3, 1)] {
        let space = Su2Space::new(d, n, two_s);
        tuned_arm(&format!("f64 d=6 ({n},{two_s})"), space, &tf, &vf);
        tuned_arm(&format!("C64 d=6 ({n},{two_s})"), space, &tc, &vc);
    }
}

#[test]
fn r1b_v3_frozen_run_cancels_groups_by_term() {
    let med = |cw: f64, neg: bool, fr: f64| Su2Factor {
        cw,
        fr,
        neg,
        transport: false,
    };
    let tr = |neg: bool, fr: f64| Su2Factor {
        cw: 0.0,
        fr,
        neg,
        transport: true,
    };
    let factors = [
        med(0.5, false, 0.8),
        med(0.25, true, 1.6),
        med(1.0, false, 0.4),
        tr(false, 0.5),
        tr(true, 0.5),
        med(1.0, false, 0.3),
        med(1.0, true, 0.3 * (1.0 + 4.0 * EPS)),
        med(1.0, true, 0.3 * (1.0 + 1e-9)),
        tr(true, 0.4),
    ];
    let n = SRC_NONE;
    type Case<'a> = (&'a str, &'a [(u32, u32)], bool);
    let cases: &[Case] = &[
        ("single median", &[(5, 0)], false),
        ("single transport", &[(n, 3)], false),
        (
            "one term, cancelling through cw and jw",
            &[(5, 0), (5, 1)],
            true,
        ),
        ("two terms, Φ₁ = −Φ₂", &[(5, 0), (9, 1)], false),
        ("median vs transport, opposite p", &[(5, 2), (n, 8)], false),
        (
            "two interleaved terms, both cancel",
            &[(5, 0), (9, 2), (5, 1), (9, 1)],
            true,
        ),
        (
            "two interleaved terms, one cancels",
            &[(5, 0), (9, 2), (5, 1)],
            false,
        ),
        (
            "transports cancel, the median does not",
            &[(n, 3), (5, 2), (n, 4)],
            false,
        ),
        (
            "transports cancel, the median group cancels",
            &[(n, 3), (5, 0), (n, 4), (5, 1)],
            true,
        ),
        ("transport-only, cancelling", &[(n, 3), (n, 4)], true),
        ("residue ≈ 4 ulp (within 16 eps)", &[(7, 5), (7, 6)], true),
        ("residue 1e-9 (outside 16 eps)", &[(7, 5), (7, 7)], false),
    ];
    for &(what, run, drops) in cases {
        assert_eq!(
            frozen_run_cancels(run, &factors),
            drops,
            "{what}: run {run:?} — expected drop = {drops}"
        );
    }
}

#[test]
fn r1b_c_every_stored_value_of_a_refillable_build_is_recomputable() {
    let d = 5;
    let (tf, vf) = dense::<f64>(d, 0x51B1_C001);
    let (tc, vc) = dense::<C64>(d, 0x51B1_C002);
    assert_no_zeros(&tc, &vc, "C64 d=5");
    for screen in [sa_trivial(), Su2TermScreen::Magnitude(EPS)] {
        let tag = format!("{screen:?}");
        let hf = refillable(Su2Space::new(d, 3, 1), &tf, &vf, &screen);
        let hc = refillable(Su2Space::new(d, 4, 2), &tc, &vc, &screen);
        for (what, cen, c, zv) in [
            (
                format!("f64 d=5 (3,1) {tag}"),
                recompute_all(&format!("f64 d=5 (3,1) {tag}"), &hf),
                print_census(&hf, &format!("f64 d=5 (3,1) {tag}")),
                zero_values(&hf, "f64 d=5 (3,1)"),
            ),
            (
                format!("C64 d=5 (4,2) {tag}"),
                recompute_all(&format!("C64 d=5 (4,2) {tag}"), &hc),
                print_census(&hc, &format!("C64 d=5 (4,2) {tag}")),
                zero_values(&hc, "C64 d=5 (4,2)"),
            ),
        ] {
            assert_census_matches(c, zv, &cen, &what);
            assert!(cen.spilled >= 1, "{what}: no spilled run (vacuous)");
            assert!(
                cen.multi_term_sites.contains(&(d - 1)),
                "{what}: no multi-term run at site d−1 (sites {:?})",
                cen.multi_term_sites
            );
        }
    }

    let lih = lih_ints();
    let what = "lih (6,3,1) SymmetryAllowed(TRIVIAL)";
    let h = refillable(Su2Space::new(6, 3, 1), &lih.t, &lih.v, &sa_trivial());
    let cen = recompute_all(what, &h);
    assert_census_matches(print_census(&h, what), zero_values(&h, what), &cen, what);
    let (zero_contribs, plus0_vals) = zero_weight_storage(&h);
    println!(
        "{what}: {zero_contribs} stored contributions of a zero-weight term, {plus0_vals} stored \
         values exactly +0"
    );
    assert!(
        zero_contribs > 0 && plus0_vals > 0,
        "{what}: vacuous — no α = ±0 contribution / +0 value stored (frozen V2 not exercised)"
    );

    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x51B1_C003));
    let what = "lih rotated (6,4,0) SymmetryAllowed(TRIVIAL)";
    let h = refillable(Su2Space::new(6, 4, 0), &rot.t, &rot.v, &sa_trivial());
    assert_census_matches(
        print_census(&h, what),
        zero_values(&h, what),
        &recompute_all(what, &h),
        what,
    );

    let (ints, pgd) = h2o_typed();
    let what = "h2o C2v 2S=2 SymmetryAllowed(pg)";
    let h = refillable(
        h2o_space(&ints, &pgd, 2),
        &ints.t,
        &ints.v,
        &Su2TermScreen::SymmetryAllowed(pgd.pg),
    );
    assert_census_matches(
        print_census(&h, what),
        zero_values(&h, what),
        &recompute_all(what, &h),
        what,
    );
}

#[test]
fn r1b_d_refillable_build_extends_the_value_build_by_exact_zeros() {
    fn arm<T: Scalar>(
        what: &str,
        space: Su2Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        screen: &Su2TermScreen,
    ) -> Agreement {
        let f = refillable(space, t, v, screen);
        let val = value_build(space, t, v, screen);
        print_census(&f, what);
        let ag = compare_with_value_build(&f, &val, what);
        assert!(ag.common > 0, "{what}: no common entry (vacuous)");
        let zv = zero_values(&f, what);
        assert_eq!(
            zv,
            (ag.extras, ag.extras),
            "{what}: census zero_values / stored +0 recount ≠ the +0 extras over the value build"
        );
        assert_eq!(
            ag.disagreements(),
            0,
            "{what}: {} frozen-vs-value disagreements (frozen-only nonzero {:?}, value-only {:?})",
            ag.disagreements(),
            ag.frozen_only_nonzero,
            ag.value_only
        );
        assert_eq!(
            n_slots(&f),
            n_slots(&val) + ag.extras,
            "{what}: slot counts do not add up"
        );
        ag
    }

    let d = 5;
    let (tf, vf) = dense::<f64>(d, 0x51B1_D001);
    let (tc, vc) = dense::<C64>(d, 0x51B1_D002);
    let sa = sa_trivial();
    arm("f64 d=5 (3,1)", Su2Space::new(d, 3, 1), &tf, &vf, &sa);
    arm("C64 d=5 (4,2)", Su2Space::new(d, 4, 2), &tc, &vc, &sa);

    let lih = lih_ints();
    for (n, two_s) in [(4usize, 0usize), (3, 1)] {
        let what = format!("lih (6,{n},{two_s})");
        let ag = arm(&what, Su2Space::new(6, n, two_s), &lih.t, &lih.v, &sa);
        assert!(
            ag.extras > 0,
            "{what}: vacuous — no +0 extra (the zero-weight classes must add entries)"
        );
    }
    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x51B1_D003));
    arm(
        "lih rotated (6,4,0)",
        Su2Space::new(6, 4, 0),
        &rot.t,
        &rot.v,
        &sa,
    );

    let (ints, pgd) = h2o_typed();
    let typed = Su2TermScreen::SymmetryAllowed(pgd.pg);
    for two_s in [0usize, 2] {
        arm(
            &format!("h2o C2v 2S={two_s}"),
            h2o_space(&ints, &pgd, two_s),
            &ints.t,
            &ints.v,
            &typed,
        );
    }
}

#[test]
#[should_panic(expected = "new_refillable refuses a three-body channel")]
fn r1b_e_refuses_three_body_under_symmetry_allowed() {
    let d = 4;
    let w = CoeffTensor::<f64>::zeros(d, 6);
    let (t, v) = dense::<f64>(d, 0x51B1_E001);
    let _ = Su2SparseHamiltonian::new_refillable(
        Su2Space::new(d, 4, 0),
        &t,
        &v,
        Some(&w as &dyn ThreeBody<f64>),
        sa_trivial(),
    );
}

#[test]
#[should_panic(expected = "new_refillable refuses a three-body channel")]
fn r1b_e_refuses_three_body_under_magnitude() {
    let d = 4;
    let w = CoeffTensor::<f64>::zeros(d, 6);
    let (t, v) = dense::<f64>(d, 0x51B1_E002);
    let _ = Su2SparseHamiltonian::new_refillable(
        Su2Space::new(d, 4, 0),
        &t,
        &v,
        Some(&w as &dyn ThreeBody<f64>),
        Su2TermScreen::Magnitude(EPS),
    );
}

#[test]
#[should_panic(expected = "point group must be the space's")]
fn r1b_e_refuses_a_trivial_screen_on_a_typed_space() {
    let (ints, pgd) = h2o_typed();
    let _ = Su2SparseHamiltonian::new_refillable(
        h2o_space(&ints, &pgd, 0),
        &ints.t,
        &ints.v,
        None,
        sa_trivial(),
    );
}

#[test]
#[should_panic(expected = "point group must be the space's")]
fn r1b_e_refuses_a_typed_screen_on_an_untyped_space() {
    let (ints, pgd) = h2o_typed();
    assert!(!pgd.pg.is_trivial());
    let _ = Su2SparseHamiltonian::new_refillable(
        Su2Space::new(ints.norb, ints.nelec, 0),
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::SymmetryAllowed(pgd.pg),
    );
}

#[test]
fn r1b_o3_refillable_rotated_lih_passes_the_cross_layer_check() {
    let lih = lih_ints();
    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x51B1_F001));
    for (n, two_s, seed) in [(4usize, 0usize, 0x51B1_F002u64), (4, 2, 0x51B1_F003)] {
        let space = Su2Space::new(6, n, two_s);
        let h = refillable(space, &rot.t, &rot.v, &sa_trivial());
        assert!(h.is_refillable());
        let mut rng = seeded(seed);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let y = h.apply(&x);
        assert!(y.check(), "(6,{n},{two_s}): H·x fails check()");
        let ny = y.norm();
        let err = max_err_over_two_sz(&x, &rot.t, &rot.v, None, &y, space);
        println!("rotated lih (6,{n},{two_s}): ‖Hx‖ = {ny:e}, cross-layer err {err:e}");
        assert!(ny > 0.1, "(6,{n},{two_s}): vacuous — ‖Hx‖ = {ny:e}");
        assert!(
            err <= 1e-11,
            "(6,{n},{two_s}): refillable rotated-lih cross-layer err {err:e}"
        );
    }
}

fn operator_bytes<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> u64 {
    let blocks: u64 = h
        .nests
        .iter()
        .flat_map(|n| n.cells.iter())
        .map(|c| c.csr.bytes() + c.csc.bytes())
        .sum();
    let states: u64 = h
        .states
        .iter()
        .flat_map(|g| g.iter())
        .map(|(_, v)| (v.capacity() * std::mem::size_of::<Ts>()) as u64)
        .sum();
    let state_qns: u64 = h
        .state_qns
        .iter()
        .flat_map(|g| g.iter())
        .map(|(_, v)| (v.capacity() * std::mem::size_of::<Su2Sector>()) as u64)
        .sum();
    blocks + states + state_qns
}

fn provenance_floor<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> u64 {
    let cells: usize = h
        .nests
        .iter()
        .flat_map(|n| n.cells.iter())
        .map(|c| {
            c.src
                .as_ref()
                .expect("refillable: cell provenance")
                .payload_len_bytes()
        })
        .sum();
    let gamma = std::mem::size_of_val(h.refill_gamma().expect("gamma"));
    let factors = std::mem::size_of_val(h.factors().expect("factors"));
    (cells + gamma + factors) as u64
}

#[test]
fn r1b_footprint_counts_the_provenance() {
    fn arm<T: Scalar>(what: &str, space: Su2Space, t: &CoeffTensor<T>, v: &CoeffTensor<T>) {
        let f = refillable(space, t, v, &sa_trivial());
        let p = value_build(space, t, v, &sa_trivial());
        let (op_f, op_p, floor) = (operator_bytes(&f), operator_bytes(&p), provenance_floor(&f));
        println!(
            "{what}: Footprint refillable {} (operator {op_f}, provenance ≥ {floor}, ratio ≥ {:.3}); \
             plain {} (operator {op_p}); census {:?}",
            f.bytes(),
            floor as f64 / op_f as f64,
            p.bytes(),
            f.refill_census().expect("refillable: census")
        );
        assert!(floor > 0, "{what}: no provenance (vacuous)");
        assert_eq!(
            p.bytes(),
            op_p,
            "{what}: a plain build's Footprint is the operator"
        );
        assert!(
            f.bytes() >= op_f + floor,
            "{what}: Footprint {} omits provenance (operator {op_f} + floor {floor})",
            f.bytes()
        );
    }
    let d = 5;
    let (tf, vf) = dense::<f64>(d, 0x51B1_9001);
    let (tc, vc) = dense::<C64>(d, 0x51B1_9002);
    arm("f64 d=5 (3,1)", Su2Space::new(d, 3, 1), &tf, &vf);
    arm("C64 d=5 (4,2)", Su2Space::new(d, 4, 2), &tc, &vc);
    let lih = lih_ints();
    arm("lih (6,3,1)", Su2Space::new(6, 3, 1), &lih.t, &lih.v);
}
