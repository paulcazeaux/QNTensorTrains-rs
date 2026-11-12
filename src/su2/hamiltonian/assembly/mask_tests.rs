// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::collections::BTreeSet;
use std::sync::Arc;

use super::refill_tests::{
    assert_recorded, assert_same_operator, bits, dense, full_class_count, h2o_space, h2o_typed,
    n_slots, testdata,
};
use super::{for_each_term, phase_a_token_stream};
use crate::C64;
use crate::coeff::{CoeffTensor, ThreeBody};
use crate::integrals::{Integrals, IntegralsOptions, load_fcidump, load_fixture_dir};
use crate::mem::Footprint;
use crate::pointgroup::{Irrep, PointGroup};
use crate::rotation::{expm_kappa_blocked, givens, rotate_integrals};
use crate::scalar::Scalar;
use crate::su2::hamiltonian::{Su2PatternMask, Su2SparseHamiltonian, Su2TermScreen};
use crate::su2::qn::Su2Space;
use crate::su2::vector::Su2TtVector;
use crate::test_util::{randomized_su2_state, seeded};
use rand::RngExt;

const EPS: f64 = f64::EPSILON;

const EPS_LEVELS: [f64; 3] = [0.0, EPS, 1e-6];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Class {
    One(usize, usize),
    Two(usize, usize, usize, usize),
}

impl Class {
    fn sites(self) -> Vec<usize> {
        match self {
            Class::One(i, j) => vec![i, j],
            Class::Two(i, j, k, l) => vec![i, j, k, l],
        }
    }

    fn of_tokens(tokens: &[(usize, bool)]) -> Class {
        match *tokens {
            [(i, true), (j, false)] => Class::One(i, j),
            [(i, true), (k, true), (l, false), (j, false)] => Class::Two(i, j, k, l),
            _ => panic!("not a one-/two-body token pattern: {tokens:?}"),
        }
    }
}

fn allowed(pg: PointGroup, c: Class) -> bool {
    pg.is_trivial() || pg.irrep_of_sites(c.sites()) == Irrep::IDENTITY
}

fn brute_weights<T: Scalar>(t: &CoeffTensor<T>, v: &CoeffTensor<T>) -> Vec<(Class, T)> {
    let d = t.d;
    let half = T::from_f64(0.5);
    let mut out = Vec::new();
    for i in 0..d {
        for j in 0..d {
            out.push((Class::One(i, j), t.get(&[i, j])));
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if (i, j) > (k, l) {
                        continue;
                    }
                    let w = if (i, j) == (k, l) {
                        v.get(&[i, j, k, l]) * half
                    } else {
                        (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])) * half
                    };
                    out.push((Class::Two(i, j, k, l), w));
                }
            }
        }
    }
    out
}

fn brute_classes<T: Scalar>(
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    eps: f64,
    pg: PointGroup,
) -> Vec<Class> {
    brute_weights(t, v)
        .into_iter()
        .filter(|&(c, w)| w.abs_s() > eps && allowed(pg, c))
        .map(|(c, _)| c)
        .collect()
}

fn brute_leak<T: Scalar>(
    mask: &[Class],
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    eps: f64,
    pg: PointGroup,
) -> usize {
    let set: BTreeSet<Class> = mask.iter().copied().collect();
    brute_classes(t, v, eps, pg)
        .into_iter()
        .filter(|c| !set.contains(c))
        .count()
}

fn stream<T: Scalar>(
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> Vec<(Class, (u64, u64))> {
    let mut out = Vec::new();
    for_each_term(t.d, t, v, None, screen, |tokens, w| {
        out.push((Class::of_tokens(tokens), bits(w)))
    });
    out
}

fn classes_of(s: &[(Class, (u64, u64))]) -> Vec<Class> {
    s.iter().map(|&(c, _)| c).collect()
}

fn assert_oracle_is_phase_a<T: Scalar>(t: &CoeffTensor<T>, v: &CoeffTensor<T>, what: &str) {
    let oracle: Vec<(Class, (u64, u64))> = brute_weights(t, v)
        .into_iter()
        .map(|(c, w)| (c, bits(w)))
        .collect();
    assert_eq!(
        oracle.len(),
        full_class_count(t.d),
        "{what}: oracle class count"
    );
    assert_eq!(
        oracle,
        stream(t, v, &Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL)),
        "{what}: the restated weights ≠ Phase A's (the oracle drifted from the convention)"
    );
}

fn mask_screen(m: &Arc<Su2PatternMask>) -> Su2TermScreen {
    Su2TermScreen::Mask(Arc::clone(m))
}

fn minted<T: Scalar>(
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    eps: f64,
    pg: PointGroup,
) -> Arc<Su2PatternMask> {
    Arc::new(Su2PatternMask::from_magnitude(t, v, eps, pg))
}

fn planted<T: Scalar>(seed: u64) -> (CoeffTensor<T>, CoeffTensor<T>) {
    let (mut t, mut v) = dense::<T>(5, seed);
    let re = |x: f64| T::from_f64(x);
    t.set(&[0, 1], re(0.0));
    t.set(&[1, 2], re(1e-17));
    t.set(&[2, 3], re(1e-9));
    v.set(&[0, 1, 2, 3], re(0.7));
    v.set(&[2, 3, 0, 1], re(-0.7));
    v.set(&[1, 2, 1, 2], re(3e-16));
    v.set(&[0, 2, 1, 3], re(3e-16));
    v.set(&[1, 3, 0, 2], re(1e-17));
    v.set(&[1, 1, 3, 4], re(0.0));
    v.set(&[3, 4, 1, 1], re(0.4));
    v.set(&[2, 2, 4, 4], re(1e-9));
    v.set(&[4, 4, 2, 2], re(1e-9));
    if T::IS_COMPLEX {
        t.set(&[3, 4], T::from_complex_f64(0.0, 0.5));
        v.set(&[0, 3, 2, 4], T::from_complex_f64(0.0, 0.3));
        v.set(&[2, 4, 0, 3], T::from_complex_f64(0.0, 0.3));
        t.set(&[4, 0], T::from_complex_f64(0.0, 1e-9));
    }
    (t, v)
}

fn planted_cases<T: Scalar>() -> Vec<(&'static str, Class, [bool; 3])> {
    let mut cases = vec![
        ("P1 exact zero", Class::One(0, 1), [false, false, false]),
        ("P2 (0, EPS]", Class::One(1, 2), [true, false, false]),
        ("P3 (EPS, 1e-6]", Class::One(2, 3), [true, true, false]),
        (
            "P4 symmetrization cancels",
            Class::Two(0, 1, 2, 3),
            [false, false, false],
        ),
        (
            "P5 diagonal ½",
            Class::Two(1, 2, 1, 2),
            [true, false, false],
        ),
        (
            "P6 off-diagonal ½",
            Class::Two(0, 2, 1, 3),
            [true, false, false],
        ),
        (
            "P7 partner only",
            Class::Two(1, 1, 3, 4),
            [true, true, true],
        ),
        (
            "P8 two-body (EPS, 1e-6]",
            Class::Two(2, 2, 4, 4),
            [true, true, false],
        ),
    ];
    if T::IS_COMPLEX {
        cases.push(("P9 imaginary 0.5i", Class::One(3, 4), [true, true, true]));
        cases.push((
            "P10 imaginary 0.3i",
            Class::Two(0, 3, 2, 4),
            [true, true, true],
        ));
        cases.push((
            "P11 imaginary (EPS, 1e-6]",
            Class::One(4, 0),
            [true, true, false],
        ));
    }
    cases
}

fn lih_ints() -> Integrals<f64> {
    load_fixture_dir::<f64>(&testdata("lih_sto3g"), &IntegralsOptions::default())
        .expect("lih_sto3g loads")
        .ints
}

fn h4_ints() -> Integrals<f64> {
    load_fcidump::<f64>(&testdata("h4.FCIDUMP"), None, &IntegralsOptions::default())
        .expect("h4.FCIDUMP loads")
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

fn h2o_with_forbidden_residue(
    ints: &Integrals<f64>,
    pg: PointGroup,
) -> (CoeffTensor<f64>, CoeffTensor<f64>, [Class; 2]) {
    let ws = brute_weights(&ints.t, &ints.v);
    let first = |two: bool| {
        ws.iter()
            .find(|&&(c, _)| matches!(c, Class::Two(..)) == two && !allowed(pg, c))
            .copied()
            .expect("h2o C2v has forbidden classes of both arities")
    };
    let (c1, w1) = first(false);
    let (c2, w2) = first(true);
    assert!(
        w1 == 0.0 && w2 == 0.0,
        "fixture: the shipped dump's forbidden cells are exact zeros ({w1:e}, {w2:e})"
    );
    let (mut t, mut v) = (ints.t.clone(), ints.v.clone());
    let Class::One(i, j) = c1 else { unreachable!() };
    t.set(&[i, j], 1e-14);
    let Class::Two(i, j, k, l) = c2 else {
        unreachable!()
    };
    v.set(&[i, j, k, l], 2e-3);
    (t, v, [c1, c2])
}

fn rel_err<T: Scalar>(a: &Su2TtVector<T>, b: &Su2TtVector<T>) -> (f64, f64) {
    let mut z = a - b;
    z.left_orthogonalize(true);
    let nb = b.norm();
    (z.norm() / nb.max(1e-300), nb)
}

fn m1_arm<T: Scalar>(
    what: &str,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    pg: PointGroup,
) -> [usize; 3] {
    assert_oracle_is_phase_a(t, v, what);
    let sa = stream(t, v, &Su2TermScreen::SymmetryAllowed(pg));
    let mut counts = [0usize; 3];
    for (n, eps) in EPS_LEVELS.into_iter().enumerate() {
        let tag = format!("{what} eps={eps:e}");
        let m = minted(t, v, eps, pg);
        let want = brute_classes(t, v, eps, pg);
        assert_eq!(
            m.n_classes(),
            want.len(),
            "{tag}: n_classes ≠ the brute-force count"
        );
        assert_eq!(
            *m,
            Su2PatternMask::from_magnitude(t, v, eps, pg),
            "{tag}: from_magnitude is not deterministic"
        );
        let got = stream(t, v, &mask_screen(&m));
        assert_eq!(
            classes_of(&got),
            want,
            "{tag}: the Mask stream's classes ≠ the brute-force |w| > eps ∧ allowed list (order \
             included)"
        );
        let keep: BTreeSet<Class> = want.iter().copied().collect();
        let sa_kept: Vec<_> = sa
            .iter()
            .filter(|(c, _)| keep.contains(c))
            .copied()
            .collect();
        assert_eq!(
            got, sa_kept,
            "{tag}: the Mask stream's weights ≠ Phase A's on the same classes"
        );
        if pg.is_trivial() {
            assert_eq!(
                got,
                stream(t, v, &Su2TermScreen::Magnitude(eps)),
                "{tag}: at |G| = 1 the minting-point mask stream must be the Magnitude(eps) stream"
            );
        }
        assert_eq!(
            m.leak(t, v, eps, pg),
            0,
            "{tag}: a mask leaks against its own minting integrals"
        );
        counts[n] = want.len();
    }
    println!("{what}: n_classes at eps = [0, EPS, 1e-6]: {counts:?}");
    counts
}

#[test]
fn r3_m1_mask_is_the_brute_force_class_set() {
    let (tf, vf) = planted::<f64>(0x5300_0001);
    let counts = m1_arm("planted f64", &tf, &vf, PointGroup::TRIVIAL);
    planted_membership("planted f64", &tf, &vf, counts);
    let (tc, vc) = planted::<C64>(0x5300_0002);
    let counts = m1_arm("planted C64", &tc, &vc, PointGroup::TRIVIAL);
    planted_membership("planted C64", &tc, &vc, counts);

    let h4 = h4_ints();
    assert_eq!(
        m1_arm("h4", &h4.t, &h4.v, PointGroup::TRIVIAL),
        [152, 114, 80],
        "h4: the eps-band census"
    );

    let lih = lih_ints();
    let c = m1_arm("lih", &lih.t, &lih.v, PointGroup::TRIVIAL);
    assert_eq!(c, [264, 264, 264], "lih: the R0 census (438 exact zeros)");
    assert!(
        c[1] < full_class_count(6),
        "lih: the mask must admit strictly fewer classes than SymmetryAllowed"
    );

    let (ints, pgd) = h2o_typed();
    let pg = pgd.pg;
    let (t, v, planted_forbidden) = h2o_with_forbidden_residue(&ints, pg);
    let c = m1_arm("h2o C2v + forbidden residue", &t, &v, pg);
    assert_eq!(
        c,
        [434, 434, 434],
        "h2o C2v: every allowed class (and no forbidden one)"
    );
    for (n, eps) in EPS_LEVELS.into_iter().enumerate() {
        let mag = classes_of(&stream(&t, &v, &Su2TermScreen::Magnitude(eps)));
        let in_mag = planted_forbidden.iter().filter(|c| mag.contains(c)).count();
        assert_eq!(
            in_mag,
            [2, 2, 1][n],
            "h2o eps={eps:e}: vacuous — the planted forbidden classes must pass the magnitude \
             screen, so only the point group can exclude them"
        );
    }
}

fn planted_membership<T: Scalar>(
    what: &str,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    counts: [usize; 3],
) {
    for (n, eps) in EPS_LEVELS.into_iter().enumerate() {
        let set: BTreeSet<Class> = brute_classes(t, v, eps, PointGroup::TRIVIAL)
            .into_iter()
            .collect();
        for (case, c, admitted) in planted_cases::<T>() {
            assert_eq!(
                set.contains(&c),
                admitted[n],
                "{what} eps={eps:e}: planted {case} {c:?}"
            );
        }
    }
    let band2 = if T::IS_COMPLEX { 3 } else { 2 };
    assert_eq!(
        counts,
        [
            full_class_count(5) - 2,
            full_class_count(5) - 5,
            full_class_count(5) - 5 - band2
        ],
        "{what}: the planted count deltas (a random class at |w| ≤ 1e-6 would shift them)"
    );
}

fn m2_arm<T: Scalar>(
    what: &str,
    (t0, v0): (&CoeffTensor<T>, &CoeffTensor<T>),
    targets: &[(&str, &CoeffTensor<T>, &CoeffTensor<T>)],
    pg: PointGroup,
) -> Vec<usize> {
    let m = minted(t0, v0, EPS, pg);
    let want = brute_classes(t0, v0, EPS, pg);
    let keep: BTreeSet<Class> = want.iter().copied().collect();
    let mut adds = Vec::new();
    for &(name, t1, v1) in targets {
        let tag = format!("{what} → {name}");
        let got = stream(t1, v1, &mask_screen(&m));
        assert_eq!(
            classes_of(&got),
            want,
            "{tag}: the Mask stream moved with the values (it must be the minted class set)"
        );
        let sa: Vec<_> = stream(t1, v1, &Su2TermScreen::SymmetryAllowed(pg))
            .into_iter()
            .filter(|(c, _)| keep.contains(c))
            .collect();
        assert_eq!(
            got, sa,
            "{tag}: the Mask stream's weights are not the target's"
        );
        let live: BTreeSet<Class> = brute_classes(t1, v1, EPS, pg).into_iter().collect();
        let drops = want.iter().filter(|c| !live.contains(c)).count();
        let add = live.iter().filter(|c| !keep.contains(c)).count();
        println!(
            "{tag}: {} mask classes; a value screen would drop {drops} and add {add}",
            want.len()
        );
        adds.push(add);
    }
    adds
}

#[test]
fn r3_m2_mask_screen_admits_its_classes_regardless_of_value() {
    fn planted_arm<T: Scalar>(what: &str, seed: (u64, u64)) {
        let (t0, v0) = planted::<T>(seed.0);
        let (t1, v1) = dense::<T>(5, seed.1);
        let (tz, vz) = (CoeffTensor::<T>::zeros(5, 2), CoeffTensor::<T>::zeros(5, 4));
        let adds = m2_arm(
            what,
            (&t0, &v0),
            &[("zeros", &tz, &vz), ("unrelated dense", &t1, &v1)],
            PointGroup::TRIVIAL,
        );
        assert!(
            adds[1] > 0,
            "{what}: vacuous — the unrelated draw must make an excluded class live"
        );
        let z = stream(
            &tz,
            &vz,
            &mask_screen(&minted(&t0, &v0, EPS, PointGroup::TRIVIAL)),
        );
        assert!(
            !z.is_empty() && z.iter().all(|&(_, b)| b == bits(T::zero())),
            "{what}: on zero integrals every masked class is emitted at +0"
        );
    }
    planted_arm::<f64>("planted f64", (0x5300_0101, 0x5300_0102));
    planted_arm::<C64>("planted C64", (0x5300_0103, 0x5300_0104));

    let lih = lih_ints();
    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x5300_0105));
    let (tz, vz) = (
        CoeffTensor::<f64>::zeros(6, 2),
        CoeffTensor::<f64>::zeros(6, 4),
    );
    let adds = m2_arm(
        "lih",
        (&lih.t, &lih.v),
        &[("zeros", &tz, &vz), ("generic rotation", &rot.t, &rot.v)],
        PointGroup::TRIVIAL,
    );
    assert_eq!(
        adds[1],
        702 - 264,
        "lih: a generic rotation makes every one of the 438 zero classes live"
    );
}

fn mint_arm<T: Scalar>(
    what: &str,
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
) -> Su2SparseHamiltonian<T> {
    let m = minted(t, v, EPS, space.pg);
    let screen = mask_screen(&m);
    let h = Su2SparseHamiltonian::new_refillable(space, t, v, None, screen.clone());
    assert!(h.is_refillable(), "{what}: not refillable");
    assert_eq!(h.term_screen(), Some(screen.clone()), "{what}: term_screen");
    assert_recorded(&h, what);
    assert_eq!(
        h.n_terms(),
        Some(m.n_classes()),
        "{what}: n_terms ≠ n_classes"
    );
    let mag = Su2TermScreen::Magnitude(EPS);
    let (g_mag, h_mag) = phase_a_token_stream(space.d, t, v, &mag);
    let (g_mask, h_mask) = phase_a_token_stream(space.d, t, v, &screen);
    assert_eq!(
        h_mask, h_mag,
        "{what}: the mask's token sequence ≠ Magnitude(EPS)'s"
    );
    let gb = |g: &[T]| g.iter().map(|&x| bits(x)).collect::<Vec<_>>();
    assert_eq!(gb(&g_mask), gb(&g_mag), "{what}: weight stream");
    assert_eq!(
        gb(h.refill_gamma().expect("refillable: gamma")),
        gb(&g_mag),
        "{what}: the stored gamma ≠ the Magnitude(EPS) stream"
    );
    let plain = Su2SparseHamiltonian::new(space, t, v, None, EPS);
    let census = h.refill_census().expect("refillable: census");
    println!(
        "{what}: {} classes; debris drops {}, stored +0 values {}, stored slots {} (plain {})",
        m.n_classes(),
        census.debris_drops,
        census.zero_values,
        n_slots(&h),
        n_slots(&plain)
    );
    assert_eq!(
        census.zero_values, 0,
        "{what}: a mask mint stores no +0 (it carries no zero-weight class)"
    );
    assert_same_operator(&h, &plain, &format!("{what} vs new(EPS)"));
    assert_eq!(
        h.artifact_hash(),
        plain.artifact_hash(),
        "{what}: artifact_hash ≠ new(EPS)"
    );
    assert_eq!(
        h.skeleton_hash(),
        plain.skeleton_hash(),
        "{what}: skeleton_hash ≠ new(EPS)"
    );
    let hm = Su2SparseHamiltonian::new_refillable(space, t, v, None, mag);
    assert_eq!(
        h.artifact_hash(),
        hm.artifact_hash(),
        "{what}: ≠ the Magnitude refillable"
    );
    assert_eq!(
        h.src_hash(),
        hm.src_hash(),
        "{what}: provenance ≠ the Magnitude refillable's"
    );
    assert_eq!(
        h.refill_census(),
        hm.refill_census(),
        "{what}: census ≠ the Magnitude refillable's"
    );
    assert_eq!(
        h.pattern_leak(t, v, EPS),
        0,
        "{what}: a mint leaks against itself"
    );
    h
}

#[test]
fn r3_m2_mask_mint_is_the_plain_build() {
    let sa = Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL);
    let lih = lih_ints();
    for two_s in [0usize, 2] {
        let space = Su2Space::new(6, 4, two_s);
        let what = format!("lih dump (6,4,{two_s})");
        let h = mint_arm(&what, space, &lih.t, &lih.v);
        let hs = Su2SparseHamiltonian::new_refillable(space, &lih.t, &lih.v, None, sa.clone());
        assert!(
            h.n_terms() < hs.n_terms() && n_slots(&h) < n_slots(&hs) && h.bytes() < hs.bytes(),
            "{what}: the mask build must be strictly smaller than SymmetryAllowed's (terms {:?} vs \
             {:?}, slots {} vs {}, bytes {} vs {})",
            h.n_terms(),
            hs.n_terms(),
            n_slots(&h),
            n_slots(&hs),
            h.bytes(),
            hs.bytes()
        );
    }

    let g2 = rotate_integrals(&lih, &givens(6, 2, 0.3));
    let space = Su2Space::new(6, 4, 0);
    let h = mint_arm("lih Givens(2,3) θ=0.3 (6,4,0)", space, &g2.t, &g2.v);
    let n = h.n_terms().expect("n_terms");
    assert!(
        264 < n && n < 702,
        "lih Givens(2,3): fixture — the mask must sit strictly between the dump's and the full set \
         ({n})"
    );

    let rot = rotate_integrals(&lih, &generic_u(6, PointGroup::TRIVIAL, 0.3, 0x5300_0201));
    for two_s in [0usize, 2] {
        let space = Su2Space::new(6, 4, two_s);
        let what = format!("lih generic rotation (6,4,{two_s})");
        let h = mint_arm(&what, space, &rot.t, &rot.v);
        assert_eq!(h.n_terms(), Some(702), "{what}: fixture — every class live");
        let hs = Su2SparseHamiltonian::new_refillable(space, &rot.t, &rot.v, None, sa.clone());
        assert_eq!(
            h.artifact_hash(),
            hs.artifact_hash(),
            "{what}: an all-live mask build ≠ the SymmetryAllowed build"
        );
    }

    let (tf, vf) = planted::<f64>(0x5300_0202);
    mint_arm("planted f64 (5,3,1)", Su2Space::new(5, 3, 1), &tf, &vf);
    let (tc, vc) = planted::<C64>(0x5300_0203);
    mint_arm("planted C64 (5,4,2)", Su2Space::new(5, 4, 2), &tc, &vc);

    let (ints, pgd) = h2o_typed();
    let space = h2o_space(&ints, &pgd, 0);
    let h = mint_arm("h2o C2v 2S=0", space, &ints.t, &ints.v);
    let hs = Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::SymmetryAllowed(pgd.pg),
    );
    assert_eq!(
        h.artifact_hash(),
        hs.artifact_hash(),
        "h2o C2v: the mask (every allowed class live) ≠ the SymmetryAllowed(pg) build"
    );
}

fn lih_raised(
    lih: &Integrals<f64>,
) -> (CoeffTensor<f64>, CoeffTensor<f64>, [Class; 2], Class, Class) {
    let ws = brute_weights(&lih.t, &lih.v);
    let find = |p: &dyn Fn(Class, f64) -> bool| {
        ws.iter()
            .find(|&&(c, w)| p(c, w))
            .map(|&(c, _)| c)
            .expect("lih fixture class")
    };
    let zero_one = find(&|c, w| matches!(c, Class::One(..)) && w == 0.0);
    let zero_two =
        find(&|c, w| matches!(c, Class::Two(i, j, k, l) if (i, j) != (k, l)) && w == 0.0);
    let live_one = find(&|c, w| matches!(c, Class::One(..)) && w != 0.0);
    let live_two = find(&|c, w| matches!(c, Class::Two(..)) && w != 0.0);
    let (mut t, mut v) = (lih.t.clone(), lih.v.clone());
    let Class::One(i, j) = zero_one else {
        unreachable!()
    };
    t.set(&[i, j], 0.1);
    let Class::Two(i, j, k, l) = zero_two else {
        unreachable!()
    };
    v.set(&[i, j, k, l], 0.2);
    let Class::One(i, j) = live_one else {
        unreachable!()
    };
    t.set(&[i, j], 0.0);
    let Class::Two(i, j, k, l) = live_two else {
        unreachable!()
    };
    v.set(&[i, j, k, l], 2.0 * lih.v.get(&[i, j, k, l]));
    if (i, j) != (k, l) {
        v.set(&[k, l, i, j], 2.0 * lih.v.get(&[k, l, i, j]));
    }
    (t, v, [zero_one, zero_two], live_one, live_two)
}

#[test]
fn r3_m1_leak_counts_exactly_the_added_classes() {
    let tr = PointGroup::TRIVIAL;
    let lih = lih_ints();
    let m = minted(&lih.t, &lih.v, EPS, tr);
    let mask0 = brute_classes(&lih.t, &lih.v, EPS, tr);
    let (t1, v1, raised, lowered, changed) = lih_raised(&lih);
    let live1 = brute_classes(&t1, &v1, EPS, tr);
    for c in raised {
        assert!(
            !mask0.contains(&c) && live1.contains(&c),
            "lih: {c:?} is not a raised screened class"
        );
    }
    assert!(
        mask0.contains(&lowered) && !live1.contains(&lowered),
        "lih: lowered class"
    );
    assert!(
        mask0.contains(&changed) && live1.contains(&changed),
        "lih: changed class"
    );
    assert_eq!(brute_leak(&mask0, &t1, &v1, EPS, tr), 2, "lih: brute leak");
    assert_eq!(
        m.leak(&t1, &v1, EPS, tr),
        2,
        "lih: leak ≠ the two raised classes"
    );

    let h4 = h4_ints();
    let masks: Vec<(f64, Arc<Su2PatternMask>, Vec<Class>)> = EPS_LEVELS
        .into_iter()
        .map(|e| {
            (
                e,
                minted(&h4.t, &h4.v, e, tr),
                brute_classes(&h4.t, &h4.v, e, tr),
            )
        })
        .collect();
    let want = [[0usize, 0, 0], [38, 0, 0], [72, 34, 0]];
    for (a, (me, m, mc)) in masks.iter().enumerate() {
        for (b, e) in EPS_LEVELS.into_iter().enumerate() {
            let tag = format!("h4 mask@{me:e} leak@{e:e}");
            assert_eq!(
                brute_leak(mc, &h4.t, &h4.v, e, tr),
                want[a][b],
                "{tag}: brute"
            );
            assert_eq!(m.leak(&h4.t, &h4.v, e, tr), want[a][b], "{tag}");
        }
    }

    fn planted_raise<T: Scalar>(what: &str, seed: u64) {
        let tr = PointGroup::TRIVIAL;
        let (t0, v0) = planted::<T>(seed);
        let m = minted(&t0, &v0, EPS, tr);
        let mask0 = brute_classes(&t0, &v0, EPS, tr);
        let (mut t1, mut v1) = (t0.clone(), v0.clone());
        t1.set(
            &[0, 1],
            T::from_complex_f64(if T::IS_COMPLEX { 0.0 } else { 0.1 }, 0.1),
        );
        v1.set(&[2, 3, 0, 1], T::from_f64(-0.5));
        v1.set(&[1, 2, 1, 2], T::from_f64(0.2));
        assert_eq!(brute_leak(&mask0, &t1, &v1, EPS, tr), 3, "{what}: brute");
        assert_eq!(
            m.leak(&t1, &v1, EPS, tr),
            3,
            "{what}: leak ≠ the three raised classes"
        );
    }
    planted_raise::<f64>("planted f64 raised", 0x5300_0301);
    planted_raise::<C64>("planted C64 raised", 0x5300_0302);

    let (ints, pgd) = h2o_typed();
    let pg = pgd.pg;
    let m = minted(&ints.t, &ints.v, EPS, pg);
    let (t1, v1, _) = h2o_with_forbidden_residue(&ints, pg);
    let mask0 = brute_classes(&ints.t, &ints.v, EPS, pg);
    let blind = brute_classes(&t1, &v1, EPS, tr)
        .into_iter()
        .filter(|c| !mask0.contains(c))
        .count();
    assert_eq!(
        blind, 2,
        "h2o: vacuous — a point-group-blind leak must see the residue"
    );
    assert_eq!(
        m.leak(&t1, &v1, EPS, pg),
        0,
        "h2o: the forbidden residue leaked"
    );
    let mut t0 = ints.t.clone();
    t0.set(&[0, 0], 0.0);
    assert!(allowed(pg, Class::One(0, 0)) && ints.t.get(&[0, 0]) != 0.0);
    let m0 = minted(&t0, &ints.v, EPS, pg);
    assert_eq!(
        m0.n_classes(),
        m.n_classes() - 1,
        "h2o: one allowed class zeroed"
    );
    assert_eq!(
        m0.leak(&ints.t, &ints.v, EPS, pg),
        1,
        "h2o: the zeroed allowed class leaks back"
    );

    let (t0, v0) = planted::<f64>(0x5300_0303);
    let (t1, v1) = dense::<f64>(5, 0x5300_0304);
    assert_eq!(
        brute_classes(&t1, &v1, EPS, tr).len(),
        full_class_count(5),
        "fixture: every class of the dense draw is live"
    );
    let m = minted(&t0, &v0, EPS, tr);
    assert_eq!(
        m.leak(&t1, &v1, EPS, tr),
        full_class_count(5) - m.n_classes(),
        "planted → dense: leak ≠ every excluded class"
    );
}

#[test]
fn r3_m3_pattern_leak_dispatches_on_the_screen() {
    let tr = PointGroup::TRIVIAL;
    let lih = lih_ints();
    let (t1, v1, ..) = lih_raised(&lih);
    let space = Su2Space::new(6, 4, 0);
    let m = minted(&lih.t, &lih.v, EPS, tr);
    let h = Su2SparseHamiltonian::new_refillable(space, &lih.t, &lih.v, None, mask_screen(&m));
    assert_eq!(
        h.pattern_leak(&lih.t, &lih.v, EPS),
        0,
        "lih mask: at the mint"
    );
    assert_eq!(
        h.pattern_leak(&t1, &v1, EPS),
        m.leak(&t1, &v1, EPS, tr),
        "lih mask"
    );
    assert_eq!(h.pattern_leak(&t1, &v1, EPS), 2, "lih mask: raised");
    let hs = Su2SparseHamiltonian::new_refillable(
        space,
        &lih.t,
        &lih.v,
        None,
        Su2TermScreen::SymmetryAllowed(tr),
    );
    for e in EPS_LEVELS {
        assert_eq!(
            hs.pattern_leak(&t1, &v1, e),
            0,
            "SymmetryAllowed at eps={e:e}"
        );
    }

    let h4 = h4_ints();
    let m = minted(&h4.t, &h4.v, EPS, tr);
    let h = Su2SparseHamiltonian::new_refillable(
        Su2Space::new(4, 4, 0),
        &h4.t,
        &h4.v,
        None,
        mask_screen(&m),
    );
    assert_eq!(
        h.pattern_leak(&h4.t, &h4.v, 0.0),
        38,
        "h4 EPS mask at eps = 0"
    );

    let (ints, pgd) = h2o_typed();
    let pg = pgd.pg;
    let space = h2o_space(&ints, &pgd, 0);
    let m = minted(&ints.t, &ints.v, EPS, pg);
    let h = Su2SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, mask_screen(&m));
    let (tf, vf, _) = h2o_with_forbidden_residue(&ints, pg);
    assert_eq!(
        h.pattern_leak(&tf, &vf, EPS),
        0,
        "h2o C2v: the forbidden residue"
    );
    let mut t0 = ints.t.clone();
    t0.set(&[0, 0], 0.0);
    let m0 = minted(&t0, &ints.v, EPS, pg);
    let h0 = Su2SparseHamiltonian::new_refillable(space, &t0, &ints.v, None, mask_screen(&m0));
    assert_eq!(
        h0.pattern_leak(&ints.t, &ints.v, EPS),
        1,
        "h2o C2v: the zeroed class"
    );
}

fn refill_arm<T: Scalar>(
    what: &str,
    space: Su2Space,
    (t0, v0): (&CoeffTensor<T>, &CoeffTensor<T>),
    (t1, v1): (&CoeffTensor<T>, &CoeffTensor<T>),
    seed: u64,
) -> (Su2SparseHamiltonian<T>, usize) {
    let pg = space.pg;
    let m = minted(t0, v0, EPS, pg);
    let screen = mask_screen(&m);
    let mask0 = brute_classes(t0, v0, EPS, pg);
    let leak = brute_leak(&mask0, t1, v1, EPS, pg);
    let live1: BTreeSet<Class> = brute_classes(t1, v1, EPS, pg).into_iter().collect();
    let lowered = mask0.iter().filter(|c| !live1.contains(c)).count();

    let mut h = Su2SparseHamiltonian::new_refillable(space, t0, v0, None, screen.clone());
    let (skel, art0) = (h.skeleton_hash(), h.artifact_hash());
    assert_eq!(
        h.pattern_leak(t1, v1, EPS),
        leak,
        "{what}: pattern_leak ≠ brute"
    );
    h.refill(t1, v1, None);
    let fresh = Su2SparseHamiltonian::new_refillable(space, t1, v1, None, screen.clone());
    assert!(h.is_refillable(), "{what}: refill dropped refillability");
    assert_eq!(h.term_screen(), Some(screen), "{what}: term_screen moved");
    assert_eq!(
        h.skeleton_hash(),
        skel,
        "{what}: the refill moved the skeleton"
    );
    assert_ne!(
        h.artifact_hash(),
        art0,
        "{what}: vacuous — no value changed"
    );
    assert_same_operator(&h, &fresh, &format!("{what} vs a fresh Mask build"));
    assert_eq!(h.artifact_hash(), fresh.artifact_hash(), "{what}: artifact");
    assert_eq!(h.src_hash(), fresh.src_hash(), "{what}: provenance + gamma");
    assert_eq!(h.refill_census(), fresh.refill_census(), "{what}: census");

    let plain = Su2SparseHamiltonian::new(space, t1, v1, None, EPS);
    let mut rng = seeded(seed);
    let x: Su2TtVector<T> = randomized_su2_state(space, 1, 3, &mut rng);
    let (rel, norm) = rel_err(&h.apply(&x), &plain.apply(&x));
    let zero_values = h.refill_census().expect("census").zero_values;
    println!(
        "{what}: leak {leak}, lowered {lowered}, stored +0 values {zero_values}, \
         ‖H_refill x − H_new x‖/‖H_new x‖ = {rel:e} (‖H_new x‖ = {norm:e})"
    );
    assert!(norm > 1e-8, "{what}: the reference action vanishes");
    if leak == 0 {
        assert!(rel <= 1e-12, "{what}: leak 0 but apply rel err {rel:e}");
        if lowered == 0 {
            assert_same_operator(&h, &plain, &format!("{what} vs new(EPS)"));
            assert_eq!(
                h.artifact_hash(),
                plain.artifact_hash(),
                "{what}: ≠ new(EPS)"
            );
        }
    } else {
        assert!(
            rel > 1e-8,
            "{what}: the leaked classes must be missing from the refilled action ({rel:e})"
        );
    }
    (h, leak)
}

#[test]
fn r3_refill_under_a_mask_is_bitwise_a_fresh_mask_build() {
    let lih = lih_ints();
    let g1 = rotate_integrals(&lih, &givens(6, 1, 0.3));
    for (two_s, seed) in [(0usize, 0x5300_0401u64), (2, 0x5300_0402)] {
        let (_, leak) = refill_arm(
            &format!("lih (6,4,{two_s}) → Givens(1,2)"),
            Su2Space::new(6, 4, two_s),
            (&lih.t, &lih.v),
            (&g1.t, &g1.v),
            seed,
        );
        assert_eq!(leak, 0, "fixture: Givens(1,2) keeps lih's zero pattern");
    }
    assert_eq!(
        brute_classes(&g1.t, &g1.v, EPS, PointGroup::TRIVIAL),
        brute_classes(&lih.t, &lih.v, EPS, PointGroup::TRIVIAL),
        "fixture: Givens(1,2) lowers no mask class either"
    );

    let g2 = rotate_integrals(&lih, &givens(6, 2, 0.3));
    let space = Su2Space::new(6, 4, 0);
    let (mut h, leak) = refill_arm(
        "lih (6,4,0) → Givens(2,3)",
        space,
        (&lih.t, &lih.v),
        (&g2.t, &g2.v),
        0x5300_0403,
    );
    assert!(leak > 0, "fixture: Givens(2,3) must grow lih's pattern");
    let minted_art = Su2SparseHamiltonian::new_refillable(
        space,
        &lih.t,
        &lih.v,
        None,
        mask_screen(&minted(&lih.t, &lih.v, EPS, PointGroup::TRIVIAL)),
    )
    .artifact_hash();
    h.refill(&lih.t, &lih.v, None);
    assert_eq!(
        h.artifact_hash(),
        minted_art,
        "lih: the round trip to the dump"
    );

    let (t0, v0) = planted::<f64>(0x5300_0404);
    let (t1, v1) = planted::<f64>(0x5300_0405);
    let (_, leak) = refill_arm(
        "planted f64 (5,3,1)",
        Su2Space::new(5, 3, 1),
        (&t0, &v0),
        (&t1, &v1),
        0x5300_0406,
    );
    assert_eq!(leak, 0, "fixture: the same planting is the same pattern");

    let (t0, v0) = planted::<C64>(0x5300_0407);
    let (mut t1, mut v1) = planted::<C64>(0x5300_0408);
    t1.set(&[3, 3], C64::new(0.0, 0.0));
    v1.set(&[0, 0, 1, 1], C64::new(0.0, 0.0));
    v1.set(&[1, 1, 0, 0], C64::new(0.0, 0.0));
    let lowered: Vec<Class> = brute_classes(&t0, &v0, EPS, PointGroup::TRIVIAL)
        .into_iter()
        .filter(|c| !brute_classes(&t1, &v1, EPS, PointGroup::TRIVIAL).contains(c))
        .collect();
    assert_eq!(
        lowered,
        vec![Class::One(3, 3), Class::Two(0, 0, 1, 1)],
        "fixture: exactly the two zeroed mask classes are lowered"
    );
    let (_, leak) = refill_arm(
        "planted C64 (5,4,2) lowered",
        Su2Space::new(5, 4, 2),
        (&t0, &v0),
        (&t1, &v1),
        0x5300_0409,
    );
    assert_eq!(leak, 0, "fixture: lowering is not a leak");
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::pattern_leak: not a refillable build")]
fn r3_refusal_pattern_leak_on_a_plain_build() {
    let (t, v) = dense::<f64>(5, 0x5300_0501);
    let h = Su2SparseHamiltonian::new(Su2Space::new(5, 3, 1), &t, &v, None, EPS);
    assert!(!h.is_refillable());
    let _ = h.pattern_leak(&t, &v, EPS);
}

#[test]
#[should_panic(expected = "Su2SparseHamiltonian::pattern_leak: not a refillable build")]
fn r3_refusal_pattern_leak_on_a_value_rule_recording_build() {
    let (t, v) = dense::<f64>(5, 0x5300_0502);
    let h = Su2SparseHamiltonian::with_screen_recorded(
        Su2Space::new(5, 3, 1),
        &t,
        &v,
        None,
        Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL),
    );
    assert!(!h.is_refillable() && h.refill_gamma().is_some());
    let _ = h.pattern_leak(&t, &v, EPS);
}

#[test]
#[should_panic(
    expected = "Su2SparseHamiltonian::pattern_leak: a Magnitude skeleton has no frozen class set"
)]
fn r3_refusal_pattern_leak_on_a_magnitude_skeleton() {
    let (t, v) = dense::<f64>(5, 0x5300_0503);
    let h = Su2SparseHamiltonian::new_refillable(
        Su2Space::new(5, 3, 1),
        &t,
        &v,
        None,
        Su2TermScreen::Magnitude(EPS),
    );
    assert!(h.is_refillable());
    let _ = h.pattern_leak(&t, &v, EPS);
}

#[test]
#[should_panic(expected = "Su2PatternMask::leak: foreign point group")]
fn r3_refusal_leak_with_a_foreign_point_group() {
    let (ints, pgd) = h2o_typed();
    let m = Su2PatternMask::from_magnitude(&ints.t, &ints.v, EPS, pgd.pg);
    let _ = m.leak(&ints.t, &ints.v, EPS, PointGroup::TRIVIAL);
}

#[test]
#[should_panic(expected = "Su2PatternMask::leak: foreign point group")]
fn r3_refusal_leak_with_a_foreign_point_group_converse() {
    let (ints, pgd) = h2o_typed();
    assert!(!pgd.pg.is_trivial());
    let m = Su2PatternMask::from_magnitude(&ints.t, &ints.v, EPS, PointGroup::TRIVIAL);
    let _ = m.leak(&ints.t, &ints.v, EPS, pgd.pg);
}

#[test]
#[should_panic(expected = "Su2PatternMask::leak: d mismatch")]
fn r3_refusal_leak_d_mismatch() {
    let (t0, v0) = dense::<f64>(5, 0x5300_0504);
    let (t1, v1) = dense::<f64>(6, 0x5300_0505);
    let m = Su2PatternMask::from_magnitude(&t0, &v0, EPS, PointGroup::TRIVIAL);
    let _ = m.leak(&t1, &v1, EPS, PointGroup::TRIVIAL);
}

#[test]
#[should_panic(expected = "Su2PatternMask: t must be d×d, v d⁴")]
fn r3_refusal_from_magnitude_v_of_another_d() {
    let (t, _) = dense::<f64>(5, 0x5300_0509);
    let (_, v) = dense::<f64>(6, 0x5300_050a);
    let _ = Su2PatternMask::from_magnitude(&t, &v, EPS, PointGroup::TRIVIAL);
}

#[test]
#[should_panic(expected = "Su2PatternMask: t must be d×d, v d⁴")]
fn r3_refusal_from_magnitude_t_not_rank_2() {
    let (_, v) = dense::<f64>(5, 0x5300_050b);
    let _ = Su2PatternMask::from_magnitude(&v, &v, EPS, PointGroup::TRIVIAL);
}

#[test]
#[should_panic(expected = "Su2PatternMask: t must be d×d, v d⁴")]
fn r3_refusal_from_magnitude_v_not_rank_4() {
    let (t, _) = dense::<f64>(5, 0x5300_050c);
    let _ = Su2PatternMask::from_magnitude(&t, &t, EPS, PointGroup::TRIVIAL);
}

#[test]
#[should_panic(expected = "for_each_term: the Mask's d (5) is not the enumeration's (6)")]
fn r3_refusal_mask_of_another_d_in_phase_a() {
    let (t5, v5) = dense::<f64>(5, 0x5300_050d);
    let (t6, v6) = dense::<f64>(6, 0x5300_050e);
    let m = minted(&t5, &v5, EPS, PointGroup::TRIVIAL);
    for_each_term(6, &t6, &v6, None, &mask_screen(&m), |_, _| {
        panic!("a refused screen emitted a term")
    });
}

#[test]
#[should_panic(expected = "point group must be the space's")]
fn r3_refusal_new_refillable_mask_of_a_foreign_point_group() {
    let (ints, pgd) = h2o_typed();
    let m = minted(&ints.t, &ints.v, EPS, PointGroup::TRIVIAL);
    let _ = Su2SparseHamiltonian::new_refillable(
        h2o_space(&ints, &pgd, 0),
        &ints.t,
        &ints.v,
        None,
        mask_screen(&m),
    );
}

#[test]
#[should_panic(expected = "point group must be the space's")]
fn r3_refusal_new_refillable_mask_of_a_foreign_point_group_converse() {
    let (ints, pgd) = h2o_typed();
    assert!(!pgd.pg.is_trivial());
    let m = minted(&ints.t, &ints.v, EPS, pgd.pg);
    let _ = Su2SparseHamiltonian::new_refillable(
        Su2Space::new(ints.norb, ints.nelec, 0),
        &ints.t,
        &ints.v,
        None,
        mask_screen(&m),
    );
}

#[test]
#[should_panic(expected = "new_refillable: the Mask's d must be the space's")]
fn r3_refusal_new_refillable_mask_of_another_d() {
    let lih = lih_ints();
    let m = minted(&lih.t, &lih.v, EPS, PointGroup::TRIVIAL);
    let (t, v) = dense::<f64>(5, 0x5300_0506);
    let _ =
        Su2SparseHamiltonian::new_refillable(Su2Space::new(5, 3, 1), &t, &v, None, mask_screen(&m));
}

#[test]
#[should_panic(expected = "does not cover the three-body channel")]
fn r3_refusal_three_body_under_a_mask_in_phase_a() {
    let (t, v) = dense::<f64>(4, 0x5300_0507);
    let w = CoeffTensor::<f64>::zeros(4, 6);
    let m = minted(&t, &v, EPS, PointGroup::TRIVIAL);
    for_each_term(
        4,
        &t,
        &v,
        Some(&w as &dyn ThreeBody<f64>),
        &mask_screen(&m),
        |_, _| panic!("a refused screen emitted a term"),
    );
}

#[test]
#[should_panic(expected = "new_refillable refuses a three-body channel")]
fn r3_refusal_three_body_under_a_mask_in_new_refillable() {
    let (t, v) = dense::<f64>(4, 0x5300_0508);
    let w = CoeffTensor::<f64>::zeros(4, 6);
    let m = minted(&t, &v, EPS, PointGroup::TRIVIAL);
    let _ = Su2SparseHamiltonian::new_refillable(
        Su2Space::new(4, 4, 0),
        &t,
        &v,
        Some(&w as &dyn ThreeBody<f64>),
        mask_screen(&m),
    );
}
