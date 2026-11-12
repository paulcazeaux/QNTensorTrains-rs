// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use qntt::coeff::CoeffTensor;
use qntt::integrals::{
    Integrals, IntegralsOptions, PointGroupPolicy, load_fcidump, load_fixture_dir,
};
use qntt::mem::Footprint;
use qntt::rotation::{
    compose_rotations, expm_kappa_blocked, givens, identity_matrix, rotate_integrals,
};
use qntt::su2::fcidump::su2_space;
use qntt::su2::hamiltonian::{Su2PatternMask, Su2SparseHamiltonian, Su2TermScreen};
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;
use qntt::{Irrep, PointGroup};
use qntt_testkit::randomized_su2_state;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

const EPS: f64 = f64::EPSILON;

fn generic_rotation(
    d: usize,
    pg: PointGroup,
    scale: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<f64> {
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

fn su2_classes_at_or_below_eps(ints: &Integrals<f64>, pg: PointGroup, eps: f64) -> usize {
    let d = ints.norb;
    let (t, v) = (&ints.t, &ints.v);
    let mut n = 0usize;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of_sites([i, j]) == Irrep::IDENTITY && t.get(&[i, j]).abs() <= eps {
                n += 1;
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if [i, j] > [k, l] || pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                        continue;
                    }
                    let vw = if [i, j] == [k, l] {
                        v.get(&[i, j, k, l]) * 0.5
                    } else {
                        (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])) * 0.5
                    };
                    n += usize::from(vw.abs() <= eps);
                }
            }
        }
    }
    n
}

fn forbidden_nonzero_cells(ints: &Integrals<f64>, pg: PointGroup) -> usize {
    let d = ints.norb;
    let mut n = 0usize;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of_sites([i, j]) != Irrep::IDENTITY && ints.t.get(&[i, j]) != 0.0 {
                n += 1;
            }
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY
                        && ints.v.get(&[i, j, k, l]) != 0.0
                    {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}

fn rel_err(a: &Su2TtVector<f64>, b: &Su2TtVector<f64>) -> (f64, f64) {
    let mut z = a - b;
    z.left_orthogonalize(true);
    let nb = b.norm();
    (z.norm() / nb.max(1e-300), nb)
}

fn o2_battery(
    what: &str,
    ints: &Integrals<f64>,
    space: Su2Space,
    u: &[f64],
    seed: u64,
    plain_skeleton_moves: bool,
) -> usize {
    let pg = space.pg;
    let screen = Su2TermScreen::SymmetryAllowed(pg);
    let rot = rotate_integrals(ints, u);
    let n_small = su2_classes_at_or_below_eps(&rot, pg, EPS);
    println!("{what}: rotated allowed classes with |w'| <= eps: {n_small}");

    let h_i = Su2SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, screen.clone());
    let h_u = Su2SparseHamiltonian::new_refillable(space, &rot.t, &rot.v, None, screen.clone());
    for (tag, h) in [("U = I", &h_i), ("U~", &h_u)] {
        assert!(h.is_refillable(), "{what} {tag}: not refillable");
        assert_eq!(
            h.term_screen(),
            Some(screen.clone()),
            "{what} {tag}: term_screen"
        );
    }
    assert_eq!(
        h_i.skeleton_hash(),
        h_u.skeleton_hash(),
        "{what}: the refillable skeleton minted at the dump ≠ the one minted at the rotated dump"
    );
    assert_ne!(
        h_i.artifact_hash(),
        h_u.artifact_hash(),
        "{what}: vacuous — the rotation did not change the operator's values"
    );

    let p_i = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, EPS);
    let p_u = Su2SparseHamiltonian::new(space, &rot.t, &rot.v, None, EPS);
    assert!(
        !p_u.is_refillable() && p_u.term_screen().is_none(),
        "{what}: new() reports refillable"
    );
    let plain_moved = p_i.skeleton_hash() != p_u.skeleton_hash();
    println!("{what}: the plain new(EPS) skeleton moves under the rotation: {plain_moved}");
    if plain_skeleton_moves {
        assert!(
            plain_moved,
            "{what}: vacuous — the plain skeleton must move between the dump and the rotation"
        );
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    for (tag, h, p) in [("U = I", &h_i, &p_i), ("U~", &h_u, &p_u)] {
        let (rel, norm) = rel_err(&h.apply(&x), &p.apply(&x));
        println!("{what} {tag}: ‖H_refill x − H_new x‖/‖H_new x‖ = {rel:e} (‖H_new x‖ = {norm:e})");
        assert!(norm > 1e-8, "{what} {tag}: the reference action vanishes");
        assert!(rel <= 1e-12, "{what} {tag}: rel err {rel:e}");
    }
    if n_small == 0 {
        assert_eq!(
            h_u.artifact_hash(),
            p_u.artifact_hash(),
            "{what}: no admitted class at |w'| <= eps, so new_refillable(U~) must be new(U~, EPS)"
        );
    }
    n_small
}

#[test]
fn su2_refill_o2_rotated_h4() {
    let load = load_fcidump::<f64>(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP"),
        None,
        &IntegralsOptions::default(),
    )
    .expect("load h4.FCIDUMP");
    let ints = load.ints;
    assert_eq!((ints.norb, ints.nelec), (4, 4));
    let u = generic_rotation(
        4,
        PointGroup::TRIVIAL,
        0.3,
        &mut Xoshiro256PlusPlus::seed_from_u64(0x51B1_0201),
    );
    for (two_s, seed) in [(0usize, 0x51B1_0202u64), (2, 0x51B1_0203)] {
        let space = Su2Space::new(4, 4, two_s);
        let n_small = o2_battery(&format!("h4 (4,4,{two_s})"), &ints, space, &u, seed, true);
        assert_eq!(n_small, 0, "h4 (4,4,{two_s}): the plain-build pin must run");
    }
}

#[test]
fn su2_refill_o2_rotated_lih_sto3g() {
    let load = load_fixture_dir::<f64>(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g");
    let ints = load.ints;
    assert_eq!((ints.norb, ints.nelec), (6, 4));
    let u = generic_rotation(
        6,
        PointGroup::TRIVIAL,
        0.3,
        &mut Xoshiro256PlusPlus::seed_from_u64(0x51B1_0301),
    );
    for (two_s, seed) in [(0usize, 0x51B1_0302u64), (2, 0x51B1_0303)] {
        let space = Su2Space::new(6, 4, two_s);
        let n_small = o2_battery(
            &format!("lih_sto3g (6,4,{two_s})"),
            &ints,
            space,
            &u,
            seed,
            true,
        );
        assert_eq!(
            n_small, 0,
            "lih (6,4,{two_s}): the plain-build pin must run"
        );
    }
}

#[test]
fn su2_refill_o2_typed_h2o_sto3g_irrep_block_diagonal() {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h2o_sto3g"),
        &opts,
    )
    .expect("load h2o_sto3g typed");
    let (ints, pgd) = (load.ints, load.pg);
    let (pg, d) = (pgd.pg, ints.norb);
    assert!(!pg.is_trivial(), "h2o_sto3g must resolve C2v");
    let u = generic_rotation(
        d,
        pg,
        0.1,
        &mut Xoshiro256PlusPlus::seed_from_u64(0x51B1_0401),
    );
    let mut mix = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert!(
                    u[p * d + q] == 0.0,
                    "u[{p}][{q}] = {:e} crosses irreps",
                    u[p * d + q]
                );
            } else if p != q {
                mix = mix.max(u[p * d + q].abs());
            }
        }
    }
    assert!(
        mix > 1e-3,
        "vacuous: the blocked rotation does not mix ({mix:e})"
    );
    assert_eq!(forbidden_nonzero_cells(&ints, pg), 0, "the screened dump");
    assert_eq!(
        forbidden_nonzero_cells(&rotate_integrals(&ints, &u), pg),
        0,
        "a blocked rotation must keep every forbidden cell an exact zero"
    );
    for (two_s, seed) in [(0usize, 0x51B1_0402u64), (2, 0x51B1_0403)] {
        let space = su2_space(&ints, &pgd, Some(two_s)).expect("h2o SU(2) space");
        assert_eq!(space.pg, pg);
        let n_small = o2_battery(
            &format!("h2o_sto3g C2v 2S={two_s}"),
            &ints,
            space,
            &u,
            seed,
            false,
        );
        assert_eq!(
            n_small, 0,
            "h2o C2v 2S={two_s}: the plain-build pin must run on the typed arm"
        );
    }
}

fn refill_battery(what: &str, ints: &Integrals<f64>, space: Su2Space, u: &[f64], seed: u64) {
    let screen = Su2TermScreen::SymmetryAllowed(space.pg);
    let rot = rotate_integrals(ints, u);
    let n_small = su2_classes_at_or_below_eps(&rot, space.pg, EPS);
    assert_eq!(
        n_small, 0,
        "{what}: fixture — the rotated dump must admit no class at |w'| <= EPS"
    );

    let mut h = Su2SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, screen.clone());
    let (minted, skel) = (h.artifact_hash(), h.skeleton_hash());
    h.refill(&rot.t, &rot.v, None);
    assert!(
        h.is_refillable(),
        "{what}: the refill dropped refillability"
    );
    assert_eq!(h.term_screen(), Some(screen.clone()), "{what}: term_screen");
    assert_eq!(
        h.skeleton_hash(),
        skel,
        "{what}: the refill moved the skeleton"
    );
    assert_ne!(
        h.artifact_hash(),
        minted,
        "{what}: vacuous — the refill changed no value"
    );
    let fresh = Su2SparseHamiltonian::new_refillable(space, &rot.t, &rot.v, None, screen.clone());
    assert_eq!(
        h.artifact_hash(),
        fresh.artifact_hash(),
        "{what}: refill(U~) ≠ a fresh new_refillable(U~)"
    );

    let plain = Su2SparseHamiltonian::new(space, &rot.t, &rot.v, None, EPS);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let (rel, norm) = rel_err(&h.apply(&x), &plain.apply(&x));
    println!("{what}: ‖H_refilled x − H_new(U~) x‖/‖H_new x‖ = {rel:e} (‖H_new x‖ = {norm:e})");
    assert!(norm > 1e-8, "{what}: the reference action vanishes");
    assert!(rel <= 1e-12, "{what}: refilled apply rel err {rel:e}");
    assert_eq!(
        h.artifact_hash(),
        plain.artifact_hash(),
        "{what}: no admitted class at |w'| <= EPS, so refill(U~) must be new(U~, EPS)"
    );

    h.refill(&ints.t, &ints.v, None);
    assert_eq!(
        h.artifact_hash(),
        minted,
        "{what}: the round trip back to the dump"
    );
}

#[test]
fn su2_refill_r2_refill_h4() {
    let load = load_fcidump::<f64>(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP"),
        None,
        &IntegralsOptions::default(),
    )
    .expect("load h4.FCIDUMP");
    let ints = load.ints;
    assert_eq!((ints.norb, ints.nelec), (4, 4));
    let u = generic_rotation(
        4,
        PointGroup::TRIVIAL,
        0.3,
        &mut Xoshiro256PlusPlus::seed_from_u64(0x51B1_0201),
    );
    for (two_s, seed) in [(0usize, 0x5220_0001u64), (2, 0x5220_0002)] {
        refill_battery(
            &format!("h4 (4,4,{two_s})"),
            &ints,
            Su2Space::new(4, 4, two_s),
            &u,
            seed,
        );
    }
}

#[test]
fn su2_refill_r2_refill_lih_sto3g() {
    let load = load_fixture_dir::<f64>(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g");
    let ints = load.ints;
    assert_eq!((ints.norb, ints.nelec), (6, 4));
    let u = generic_rotation(
        6,
        PointGroup::TRIVIAL,
        0.3,
        &mut Xoshiro256PlusPlus::seed_from_u64(0x51B1_0301),
    );
    for (two_s, seed) in [(0usize, 0x5220_0003u64), (2, 0x5220_0004)] {
        refill_battery(
            &format!("lih_sto3g (6,4,{two_s})"),
            &ints,
            Su2Space::new(6, 4, two_s),
            &u,
            seed,
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Su2Class {
    One(usize, usize),
    Two(usize, usize, usize, usize),
}

fn su2_class_weights(ints: &Integrals<f64>) -> Vec<(Su2Class, f64)> {
    let d = ints.norb;
    let (t, v) = (&ints.t, &ints.v);
    let mut out = Vec::new();
    for i in 0..d {
        for j in 0..d {
            out.push((Su2Class::One(i, j), t.get(&[i, j])));
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if [i, j] > [k, l] {
                        continue;
                    }
                    let vw = if [i, j] == [k, l] {
                        v.get(&[i, j, k, l]) * 0.5
                    } else {
                        (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])) * 0.5
                    };
                    out.push((Su2Class::Two(i, j, k, l), vw));
                }
            }
        }
    }
    out
}

fn su2_class_set(ints: &Integrals<f64>, eps: f64) -> BTreeSet<Su2Class> {
    su2_class_weights(ints)
        .into_iter()
        .filter(|&(_, w)| w.abs() > eps)
        .map(|(c, _)| c)
        .collect()
}

fn su2_eps_margin(ints: &Integrals<f64>) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
    for (_, w) in su2_class_weights(ints) {
        let a = w.abs();
        if a > EPS {
            lo = lo.min(a);
        } else if a > 0.0 {
            hi = hi.max(a);
        }
    }
    (lo, hi)
}

fn load_lih() -> Integrals<f64> {
    load_fixture_dir::<f64>(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g")
    .ints
}

const R3_STEPS: usize = 12;

fn r3_trajectory(
    what: &str,
    ints: &Integrals<f64>,
    space: Su2Space,
    x_seed: u64,
) -> (usize, usize) {
    let d = space.d;
    let pg = space.pg;
    let full = su2_class_weights(ints).len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5230_0001);
    let x: Su2TtVector<f64> =
        randomized_su2_state(space, 1, 3, &mut Xoshiro256PlusPlus::seed_from_u64(x_seed));
    let mint = |rot: &Integrals<f64>, live: &BTreeSet<Su2Class>, tag: &str| {
        let m = Arc::new(Su2PatternMask::from_magnitude(&rot.t, &rot.v, EPS, pg));
        assert_eq!(
            m.n_classes(),
            live.len(),
            "{tag}: n_classes ≠ the brute-force |w| > EPS count"
        );
        let screen = Su2TermScreen::Mask(Arc::clone(&m));
        let h = Su2SparseHamiltonian::new_refillable(space, &rot.t, &rot.v, None, screen.clone());
        assert!(h.is_refillable(), "{tag}: not refillable");
        assert_eq!(h.term_screen(), Some(screen), "{tag}: term_screen");
        assert_eq!(
            h.pattern_leak(&rot.t, &rot.v, EPS),
            0,
            "{tag}: a fresh mint leaks against its own integrals"
        );
        (m, live.clone(), h)
    };

    let mut u = identity_matrix(d);
    let mut cur: Option<(
        Arc<Su2PatternMask>,
        BTreeSet<Su2Class>,
        Su2SparseHamiltonian<f64>,
    )> = None;
    let (mut refills, mut remints, mut bitwise) = (0usize, 0usize, 0usize);
    for step in 0..R3_STEPS {
        let bond = rng.random_range(0..d - 1);
        let mag: f64 = rng.random_range(0.1..0.5);
        let theta = if rng.random_range(0..2) == 0 {
            mag
        } else {
            -mag
        };
        u = compose_rotations(&u, &givens(d, bond, theta), d);
        let rot = rotate_integrals(ints, &u);
        let live = su2_class_set(&rot, EPS);
        let tag = format!("{what} step {step}");
        let (action, leak, (m, set, h)) = match cur.take() {
            None => {
                let minted = mint(&rot, &live, &tag);
                assert!(
                    minted.0.n_classes() < full,
                    "{tag}: vacuous — the first-step mask admits every class ({full})"
                );
                let sa = Su2SparseHamiltonian::new_refillable(
                    space,
                    &rot.t,
                    &rot.v,
                    None,
                    Su2TermScreen::SymmetryAllowed(pg),
                );
                println!(
                    "{tag}: mask {} of {full} classes; Footprint mask {} B vs SymmetryAllowed {} B",
                    minted.0.n_classes(),
                    minted.2.bytes(),
                    sa.bytes()
                );
                assert!(
                    minted.2.bytes() < sa.bytes(),
                    "{tag}: the mask build must be smaller than the SymmetryAllowed build"
                );
                ("mint", 0, minted)
            }
            Some((m, set, mut h)) => {
                let leak = h.pattern_leak(&rot.t, &rot.v, EPS);
                assert_eq!(
                    leak,
                    live.difference(&set).count(),
                    "{tag}: pattern_leak ≠ the brute-force leak"
                );
                assert_eq!(
                    leak,
                    m.leak(&rot.t, &rot.v, EPS, pg),
                    "{tag}: pattern_leak ≠ the mask's leak"
                );
                if leak == 0 {
                    let (skel, art0) = (h.skeleton_hash(), h.artifact_hash());
                    h.refill(&rot.t, &rot.v, None);
                    assert_eq!(
                        h.skeleton_hash(),
                        skel,
                        "{tag}: the refill moved the skeleton"
                    );
                    assert_ne!(
                        h.artifact_hash(),
                        art0,
                        "{tag}: vacuous — the step moved no value"
                    );
                    let fresh = Su2SparseHamiltonian::new_refillable(
                        space,
                        &rot.t,
                        &rot.v,
                        None,
                        Su2TermScreen::Mask(Arc::clone(&m)),
                    );
                    assert_eq!(
                        h.artifact_hash(),
                        fresh.artifact_hash(),
                        "{tag}: refill ≠ a fresh new_refillable(Mask(m), Ũ)"
                    );
                    refills += 1;
                    ("refill", 0, (m, set, h))
                } else {
                    remints += 1;
                    ("re-mint", leak, mint(&rot, &live, &tag))
                }
            }
        };
        let lowered = set.difference(&live).count();
        let plain = Su2SparseHamiltonian::new(space, &rot.t, &rot.v, None, EPS);
        let same = h.artifact_hash() == plain.artifact_hash();
        bitwise += usize::from(same);
        let (rel, norm) = rel_err(&h.apply(&x), &plain.apply(&x));
        let (lo, hi) = su2_eps_margin(&rot);
        println!(
            "{what} step {step:2}: bond ({bond},{}) θ = {theta:+.3}  leak {leak:3}  {action:7}  \
             n_classes {:3}  lowered {lowered}  bitwise new(Ũ,EPS) {same}  rel {rel:.2e}  \
             min |w|>EPS {lo:.1e}  max |w| in (0,EPS] {hi:.1e}",
            bond + 1,
            m.n_classes()
        );
        assert!(norm > 1e-8, "{tag}: the reference action vanishes");
        assert!(
            rel <= 1e-12,
            "{tag}: ‖H x − H_new(Ũ, EPS) x‖/‖H_new x‖ = {rel:e}"
        );
        cur = Some((m, set, h));
    }
    println!(
        "{what}: {R3_STEPS} steps — 1 mint, {remints} re-mint(s), {refills} refill(s); bitwise \
         new(Ũ, EPS) at {bitwise}/{R3_STEPS} steps (printed, not asserted: cancellation debris \
         may legitimately split the frozen and value rules)"
    );
    assert_eq!(refills + remints, R3_STEPS - 1, "{what}: step accounting");
    (refills, remints)
}

#[test]
fn su2_refill_r3_gate_leak_along_a_rotated_lih_trajectory() {
    let ints = load_lih();
    assert_eq!((ints.norb, ints.nelec), (6, 4));
    for (two_s, seed) in [(0usize, 0x5230_0002u64), (2, 0x5230_0003)] {
        let what = format!("lih_sto3g (6,4,{two_s})");
        let (refills, remints) = r3_trajectory(&what, &ints, Su2Space::new(6, 4, two_s), seed);
        assert!(refills >= 1, "{what}: vacuous — no step refilled");
        assert!(remints >= 1, "{what}: vacuous — the re-mint path never ran");
    }
}

#[test]
fn su2_refill_r3_mask_mint_and_leak_on_h4_eps_bands() {
    let ints = load_fcidump::<f64>(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP"),
        None,
        &IntegralsOptions::default(),
    )
    .expect("load h4.FCIDUMP")
    .ints;
    assert_eq!((ints.norb, ints.nelec), (4, 4));
    let levels = [0.0, EPS, 1e-6];
    let masks: Vec<Arc<Su2PatternMask>> = levels
        .iter()
        .map(|&e| {
            Arc::new(Su2PatternMask::from_magnitude(
                &ints.t,
                &ints.v,
                e,
                PointGroup::TRIVIAL,
            ))
        })
        .collect();
    let counts: Vec<usize> = masks.iter().map(|m| m.n_classes()).collect();
    let brute: Vec<usize> = levels
        .iter()
        .map(|&e| su2_class_set(&ints, e).len())
        .collect();
    println!("h4: n_classes at eps = [0, EPS, 1e-6]: {counts:?} (brute force {brute:?})");
    assert_eq!(counts, brute, "h4: n_classes ≠ the brute force");
    assert_eq!(counts, vec![152, 114, 80], "h4: the eps-band census");
    let want = [[0usize, 0, 0], [38, 0, 0], [72, 34, 0]];
    for two_s in [0usize, 2] {
        let space = Su2Space::new(4, 4, two_s);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5240_0001 + two_s as u64);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let sa = Su2SparseHamiltonian::new_refillable(
            space,
            &ints.t,
            &ints.v,
            None,
            Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL),
        );
        let plain_eps = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, EPS);
        for (a, (&e, m)) in levels.iter().zip(&masks).enumerate() {
            let tag = format!("h4 (4,4,{two_s}) mask@{e:e}");
            let h = Su2SparseHamiltonian::new_refillable(
                space,
                &ints.t,
                &ints.v,
                None,
                Su2TermScreen::Mask(Arc::clone(m)),
            );
            let plain = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, e);
            assert_eq!(
                h.artifact_hash(),
                plain.artifact_hash(),
                "{tag}: the mask mint ≠ new(h4, eps)"
            );
            for (b, &le) in levels.iter().enumerate() {
                assert_eq!(
                    h.pattern_leak(&ints.t, &ints.v, le),
                    want[a][b],
                    "{tag}: pattern_leak at {le:e}"
                );
                assert_eq!(sa.pattern_leak(&ints.t, &ints.v, le), 0, "SymmetryAllowed");
            }
            if a == 0 {
                assert_eq!(
                    h.artifact_hash(),
                    sa.artifact_hash(),
                    "{tag}: the all-class mask build ≠ the SymmetryAllowed build"
                );
                assert_ne!(
                    h.artifact_hash(),
                    plain_eps.artifact_hash(),
                    "{tag}: vacuous — the 38 band classes must be stored"
                );
                let (rel, norm) = rel_err(&h.apply(&x), &plain_eps.apply(&x));
                println!("{tag}: ‖H x − H_new(EPS) x‖/‖H_new x‖ = {rel:e} (‖H_new x‖ = {norm:e})");
                assert!(norm > 1e-8 && rel <= 1e-12, "{tag}: rel err {rel:e}");
            }
        }
    }
}
