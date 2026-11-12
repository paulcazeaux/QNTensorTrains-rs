// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

use super::{Assembled, Su2TermScreen, assemble, assemble_screened, for_each_term};
use crate::coeff::{CoeffTensor, ThreeBody};
use crate::fcidump::{PointGroupData, TwoBodySymmetry, read_fcidump, screen_point_group_forbidden};
use crate::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::Su2Space;
use crate::su2::spinalgebra::SpinAlgebra;
use crate::su2::vector::Su2TtVector;
use crate::test_util::{
    randn_4tensor, randn_matrix, randomized_su2_state, seeded, set8, zeros_4tensor, zeros_matrix,
};

type Term = (Vec<(usize, bool)>, f64);
type Stream = Vec<Term>;

fn stream(
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    w: Option<&dyn ThreeBody<f64>>,
    screen: &Su2TermScreen,
) -> Stream {
    let mut out = Stream::new();
    for_each_term(t.d, t, v, w, screen, |tok, wt| out.push((tok.to_vec(), wt)));
    out
}

fn tokens(s: &Stream) -> Vec<Vec<(usize, bool)>> {
    s.iter().map(|(tok, _)| tok.clone()).collect()
}

fn assert_streams_identical(a: &Stream, b: &Stream, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: sink-call counts differ");
    for (n, ((ta, wa), (tb, wb))) in a.iter().zip(b).enumerate() {
        assert!(
            ta == tb && wa.to_bits() == wb.to_bits(),
            "{what}: term #{n} differs: {ta:?} w={wa:e} vs {tb:?} w={wb:e}"
        );
    }
}

fn pre_r0_magnitude_stream(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    w: Option<&dyn ThreeBody<f64>>,
    eps: f64,
) -> Stream {
    let mut out = Stream::new();
    let mut sink = |tok: &[(usize, bool)], wt: f64| out.push((tok.to_vec(), wt));
    for i in 0..d {
        for j in 0..d {
            let tij = t.get(&[i, j]);
            if tij.abs_s() > eps {
                sink(&[(i, true), (j, false)], tij);
            }
        }
    }
    let half = f64::from_f64(0.5);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if [i, j] > [k, l] {
                        continue;
                    }
                    let vw = if [i, j] == [k, l] {
                        v.get(&[i, j, k, l]) * half
                    } else {
                        (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])) * half
                    };
                    if vw.abs_s() > eps {
                        sink(&[(i, true), (k, true), (l, false), (j, false)], vw);
                    }
                }
            }
        }
    }
    if let Some(w) = w {
        let sixth = f64::from_f64(1.0 / 6.0);
        let mut nz3 = w.support_above(6.0 * eps);
        nz3.sort_unstable();
        for idx in nz3 {
            let [i, j, k, l, m, n] = idx;
            let cw = w.entry(&idx) * sixth;
            if cw.abs_s() > eps {
                sink(
                    &[
                        (i, true),
                        (k, true),
                        (m, true),
                        (n, false),
                        (l, false),
                        (j, false),
                    ],
                    cw,
                );
            }
        }
    }
    out
}

fn fused(pg: PointGroup, sites: &[usize]) -> Irrep {
    let mut acc = Irrep::IDENTITY;
    for &s in sites {
        acc = acc.fuse(pg.irrep_of(s));
    }
    acc
}

fn phase_a_classes(
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    keep: impl Fn(&[usize]) -> bool,
) -> Stream {
    let d = t.d;
    let mut out = Stream::new();
    for i in 0..d {
        for j in 0..d {
            if keep(&[i, j]) {
                out.push((vec![(i, true), (j, false)], t.get(&[i, j])));
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if [i, j] > [k, l] || !keep(&[i, j, k, l]) {
                        continue;
                    }
                    let vw = if [i, j] == [k, l] {
                        v.get(&[i, j, k, l]) * 0.5
                    } else {
                        (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])) * 0.5
                    };
                    out.push((vec![(i, true), (k, true), (l, false), (j, false)], vw));
                }
            }
        }
    }
    out
}

fn allowed_classes(pg: PointGroup, t: &CoeffTensor<f64>, v: &CoeffTensor<f64>) -> Stream {
    phase_a_classes(t, v, |s| fused(pg, s) == Irrep::IDENTITY)
}

fn full_class_count(d: usize) -> usize {
    d * d + d * d * (d * d + 1) / 2
}

fn assert_assembled_identical(a: Assembled<f64>, b: Assembled<f64>, what: &str) {
    assert!(a.states == b.states, "{what}: vertex lists (states) differ");
    assert!(
        a.state_qns == b.state_qns,
        "{what}: coupled x-sector lists (state_qns) differ"
    );
    assert_eq!(a.coo.len(), b.coo.len(), "{what}: nest counts differ");
    for (k, (na, nb)) in a.coo.into_iter().zip(b.coo).enumerate() {
        assert_eq!(
            na.offsets(),
            nb.offsets(),
            "{what}: site {k} offsets differ"
        );
        assert_eq!(
            na.sectors(),
            nb.sectors(),
            "{what}: site {k} sectors differ"
        );
        let ca: Vec<_> = na.into_cells().collect();
        let cb: Vec<_> = nb.into_cells().collect();
        assert_eq!(ca.len(), cb.len(), "{what}: site {k} populated-cell counts");
        for (c, (ea, eb)) in ca.iter().zip(&cb).enumerate() {
            assert_eq!(
                (ea.x_occ, ea.m_ord),
                (eb.x_occ, eb.m_ord),
                "{what}: site {k} cell {c} key"
            );
            assert_eq!(ea.cell.rows, eb.cell.rows, "{what}: site {k} cell {c} rows");
            assert_eq!(ea.cell.cols, eb.cell.cols, "{what}: site {k} cell {c} cols");
            let ba: Vec<u64> = ea.cell.vals.iter().map(|x| x.to_bits()).collect();
            let bb: Vec<u64> = eb.cell.vals.iter().map(|x| x.to_bits()).collect();
            assert_eq!(ba, bb, "{what}: site {k} cell {c} values");
        }
    }
}

fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

fn h4_tv() -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let data = read_fcidump::<f64>(&testdata("h4.FCIDUMP"), TwoBodySymmetry::Eightfold)
        .expect("read h4.FCIDUMP");
    (data.t, data.v)
}

fn lih_tv() -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let load = load_fixture_dir::<f64>(&testdata("lih_sto3g"), &IntegralsOptions::default())
        .expect("lih_sto3g loads");
    (load.ints.t, load.ints.v)
}

fn h2o_typed() -> (Integrals<f64>, PointGroupData) {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: false,
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(&testdata("h2o_sto3g"), &opts).expect("h2o_sto3g loads");
    assert!(
        !load.pg.pg.is_trivial(),
        "h2o_sto3g must resolve to a nontrivial group, or test 3 is not on a typed space"
    );
    (load.ints, load.pg)
}

fn h2o_space(ints: &Integrals<f64>, pg: &PointGroupData, two_s: usize) -> Su2Space {
    crate::su2::fcidump::su2_space(ints, pg, Some(two_s)).expect("h2o SU(2) space")
}

fn screened(
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    pg: PointGroup,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let (mut ts, mut vs) = (t.clone(), v.clone());
    screen_point_group_forbidden(&mut ts, &mut vs, pg).expect("screen h2o");
    (ts, vs)
}

fn rel_apply_err(
    ha: &Su2SparseHamiltonian<f64>,
    hb: &Su2SparseHamiltonian<f64>,
    space: Su2Space,
    seed: u64,
) -> f64 {
    let mut rng = seeded(seed);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let ya = ha.apply(&x);
    let yb = hb.apply(&x);
    let nb = yb.norm();
    assert!(nb > 1e-8, "reference action vanishes (‖Hx‖ = {nb:e})");
    (&ya - &yb).norm() / nb
}

fn random_d5_with_edges() -> (CoeffTensor<f64>, CoeffTensor<f64>, CoeffTensor<f64>) {
    let d = 5;
    let mut rng = seeded(0x5230_0001);
    let mut t = randn_matrix::<f64, _>(d, &mut rng);
    let mut v = randn_4tensor::<f64, _>(d, &mut rng);
    t.set(&[0, 1], 0.0);
    t.set(&[1, 2], 1e-6);
    t.set(&[2, 3], 1e-9);
    v.set(&[1, 2, 1, 2], 2e-6);
    v.set(&[0, 3, 4, 1], 1e-9);
    v.set(&[4, 1, 0, 3], 0.0);
    let mut w = CoeffTensor::<f64>::zeros(d, 6);
    w.set(&[0, 1, 2, 2, 3, 4], 0.7);
    w.set(&[1, 1, 2, 2, 3, 3], -0.4);
    w.set(&[0, 4, 1, 3, 2, 2], 3e-9);
    (t, v, w)
}

const EPS_GRID: [f64; 3] = [0.0, f64::EPSILON, 1e-6];

type IntegralArm<'a> = (
    &'a str,
    &'a CoeffTensor<f64>,
    &'a CoeffTensor<f64>,
    Option<&'a dyn ThreeBody<f64>>,
);
type SpaceArm<'a> = (
    &'a str,
    Su2Space,
    &'a CoeffTensor<f64>,
    &'a CoeffTensor<f64>,
    Option<&'a dyn ThreeBody<f64>>,
);

#[test]
fn r0_t1_magnitude_stream_is_the_pre_r0_stream_bitwise() {
    let (tr, vr, wr) = random_d5_with_edges();
    let (th, vh) = h4_tv();
    let (tl, vl) = lih_tv();
    let arms: [IntegralArm; 3] = [
        ("random d=5 + w", &tr, &vr, Some(&wr as &dyn ThreeBody<f64>)),
        ("h4", &th, &vh, None),
        ("lih_sto3g", &tl, &vl, None),
    ];
    for (name, t, v, w) in arms {
        let mut counts = Vec::new();
        for eps in EPS_GRID {
            let got = stream(t, v, w, &Su2TermScreen::Magnitude(eps));
            let want = pre_r0_magnitude_stream(t.d, t, v, w, eps);
            assert_streams_identical(&got, &want, &format!("{name} Magnitude({eps:e})"));
            counts.push(got.len());
        }
        println!("{name}: sink calls at eps = 0 / EPS / 1e-6: {counts:?}");
        match name {
            "random d=5 + w" => {
                assert!(
                    counts[0] < full_class_count(5) + 3,
                    "eps = 0 must drop the exact zero (strict `>`): {counts:?}"
                );
                assert!(
                    counts[2] + 5 <= counts[1],
                    "1e-6 must drop the edge terms: {counts:?}"
                );
                for eps in EPS_GRID {
                    let n3 = stream(t, v, w, &Su2TermScreen::Magnitude(eps))
                        .iter()
                        .filter(|(tok, _)| tok.len() == 6)
                        .count();
                    assert!(n3 >= 2, "three-body terms missing at eps = {eps:e} ({n3})");
                }
            }
            "h4" => assert!(
                counts[0] > counts[1] && counts[1] > counts[2],
                "h4 must discriminate the eps grid: {counts:?}"
            ),
            _ => {}
        }
    }
}

#[test]
fn r0_t1_assemble_is_assemble_screened_magnitude_bitwise() {
    let (tr, vr, wr) = random_d5_with_edges();
    let (th, vh) = h4_tv();
    let (tl, vl) = lih_tv();
    let arms: [SpaceArm; 3] = [
        (
            "random d=5 (3,1) + w",
            Su2Space::new(5, 3, 1),
            &tr,
            &vr,
            Some(&wr as &dyn ThreeBody<f64>),
        ),
        ("lih_sto3g (6,4,0)", Su2Space::new(6, 4, 0), &tl, &vl, None),
        ("h4 (4,4,2)", Su2Space::new(4, 4, 2), &th, &vh, None),
    ];
    for (name, space, t, v, w) in arms {
        for eps in EPS_GRID {
            let a = assemble(space, t, v, w, eps, &mut SpinAlgebra::new());
            let b = assemble_screened(
                space,
                t,
                v,
                w,
                &Su2TermScreen::Magnitude(eps),
                &mut SpinAlgebra::new(),
            );
            assert_assembled_identical(a, b, &format!("{name} eps={eps:e}"));
        }
    }
}

#[test]
fn r0_t2_symmetry_allowed_trivial_admits_every_class_value_blind() {
    let (th, vh) = h4_tv();
    let (tl, vl) = lih_tv();
    let trivial = Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL);
    for (name, t, v) in [("h4", &th, &vh), ("lih_sto3g", &tl, &vl)] {
        let d = t.d;
        let sa = stream(t, v, None, &trivial);
        assert_eq!(
            sa.len(),
            full_class_count(d),
            "{name}: d² + #{{(i,j) ≤ (k,l)}}"
        );
        assert_streams_identical(
            &sa,
            &allowed_classes(PointGroup::TRIVIAL, t, v),
            &format!("{name} SymmetryAllowed(TRIVIAL) vs brute force"),
        );
        let blind = stream(&zeros_matrix(d), &zeros_4tensor(d), None, &trivial);
        assert!(
            tokens(&blind) == tokens(&sa),
            "{name}: the SymmetryAllowed term set moved with the integral values"
        );
        let mag0 = stream(t, v, None, &Su2TermScreen::Magnitude(0.0));
        let nonzero: Stream = sa.iter().filter(|(_, w)| *w != 0.0).cloned().collect();
        assert_streams_identical(
            &nonzero,
            &mag0,
            &format!("{name}: SymmetryAllowed minus its exact zeros vs Magnitude(0.0)"),
        );
        let extra = sa.len() - mag0.len();
        println!(
            "{name}: SymmetryAllowed(TRIVIAL) {} sink calls, Magnitude(0.0) {}, extra exact zeros {extra}",
            sa.len(),
            mag0.len()
        );
        if name == "lih_sto3g" {
            assert!(
                extra > 0,
                "lih must carry exact-zero classes (anti-vacuity)"
            );
        }
    }
}

#[test]
fn r0_t2_symmetry_allowed_trivial_apply_matches_magnitude_zero() {
    let (th, vh) = h4_tv();
    let (tl, vl) = lih_tv();
    let arms = [
        (
            "h4 (4,4,0)",
            Su2Space::new(4, 4, 0),
            &th,
            &vh,
            0x5230_2001u64,
        ),
        ("h4 (4,4,2)", Su2Space::new(4, 4, 2), &th, &vh, 0x5230_2002),
        (
            "lih_sto3g (6,4,0)",
            Su2Space::new(6, 4, 0),
            &tl,
            &vl,
            0x5230_2003,
        ),
        (
            "lih_sto3g (6,3,1)",
            Su2Space::new(6, 3, 1),
            &tl,
            &vl,
            0x5230_2004,
        ),
    ];
    for (name, space, t, v, seed) in arms {
        let ha = Su2SparseHamiltonian::with_screen(
            space,
            t,
            v,
            None,
            Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL),
        );
        let hb =
            Su2SparseHamiltonian::with_screen(space, t, v, None, Su2TermScreen::Magnitude(0.0));
        let rel = rel_apply_err(&ha, &hb, space, seed);
        println!("{name}: ‖H_SA x − H_mag0 x‖/‖H_mag0 x‖ = {rel:e}");
        assert!(
            rel <= 1e-12,
            "{name}: SymmetryAllowed(TRIVIAL) apply rel err {rel:e}"
        );
    }
}

#[test]
fn r0_t3_symmetry_allowed_typed_admits_exactly_the_allowed_classes() {
    let (ints, pgd) = h2o_typed();
    let pg = pgd.pg;
    let d = ints.norb;
    let screen = Su2TermScreen::SymmetryAllowed(pg);
    let sa = stream(&ints.t, &ints.v, None, &screen);
    let brute = allowed_classes(pg, &ints.t, &ints.v);
    assert_streams_identical(&sa, &brute, "h2o SymmetryAllowed(C2v) vs brute force");
    let n1 = sa.iter().filter(|(tok, _)| tok.len() == 2).count();
    let n2 = sa.iter().filter(|(tok, _)| tok.len() == 4).count();
    println!(
        "h2o_sto3g C2v: {} admitted ({n1} one-body + {n2} two-body) of {} classes",
        sa.len(),
        full_class_count(d)
    );
    assert!(
        sa.len() < full_class_count(d) && n1 > 0 && n2 > 0,
        "the C2v table must exclude classes and admit both arities"
    );
    let blind = stream(&zeros_matrix(d), &zeros_4tensor(d), None, &screen);
    assert!(
        tokens(&blind) == tokens(&sa),
        "h2o: the typed SymmetryAllowed term set moved with the integral values"
    );
}

#[test]
fn r0_t3_symmetry_allowed_typed_apply_matches_screened_magnitude() {
    let (ints, pgd) = h2o_typed();
    let pg = pgd.pg;
    let (ts, vs) = screened(&ints.t, &ints.v, pg);

    let (mut tz, mut vz) = (ts.clone(), vs.clone());
    tz.set(&[0, 1], 0.0);
    tz.set(&[1, 0], 0.0);
    set8(&mut vz, 0, 1, 3, 5, 0.0);
    set8(&mut vz, 2, 6, 4, 4, 0.0);
    let zeroed: [&[usize]; 3] = [&[0, 1], &[0, 1, 3, 5], &[2, 6, 4, 4]];
    for sites in zeroed {
        assert_eq!(
            fused(pg, sites),
            Irrep::IDENTITY,
            "{sites:?} must be an allowed class"
        );
    }
    let zero_allowed = allowed_classes(pg, &tz, &vz)
        .iter()
        .filter(|(_, w)| *w == 0.0)
        .count();
    let extra = stream(&tz, &vz, None, &Su2TermScreen::SymmetryAllowed(pg)).len()
        - stream(&tz, &vz, None, &Su2TermScreen::Magnitude(f64::EPSILON)).len();
    println!("h2o zeroed-allowed arm: {zero_allowed} exact-zero allowed classes, extra {extra}");
    assert!(
        zero_allowed > 0,
        "the zeroed arm must carry exact-zero allowed classes"
    );
    assert_eq!(
        extra, zero_allowed,
        "SymmetryAllowed − Magnitude(EPS) = the exact-zero classes"
    );

    for (two_s, seed) in [(0usize, 0x5230_3001u64), (2, 0x5230_3002)] {
        let space = h2o_space(&ints, &pgd, two_s);
        let ha = Su2SparseHamiltonian::with_screen(
            space,
            &ints.t,
            &ints.v,
            None,
            Su2TermScreen::SymmetryAllowed(pg),
        );
        let hb = Su2SparseHamiltonian::new(space, &ts, &vs, None, f64::EPSILON);
        let rel = rel_apply_err(&ha, &hb, space, seed);
        println!(
            "h2o 2S={two_s}: SymmetryAllowed(unscreened) vs Magnitude(EPS, screened): {rel:e}"
        );
        assert!(rel <= 1e-12, "h2o 2S={two_s}: rel err {rel:e}");

        let ha = Su2SparseHamiltonian::with_screen(
            space,
            &tz,
            &vz,
            None,
            Su2TermScreen::SymmetryAllowed(pg),
        );
        let hb = Su2SparseHamiltonian::new(space, &tz, &vz, None, f64::EPSILON);
        let rel = rel_apply_err(&ha, &hb, space, seed ^ 0xFF);
        println!("h2o 2S={two_s} zeroed-allowed: {rel:e}");
        assert!(
            rel <= 1e-12,
            "h2o 2S={two_s} zeroed-allowed: rel err {rel:e}"
        );
    }
}

#[test]
fn r0_t3_symmetry_allowed_typed_excludes_forbidden_residue() {
    let (ints, pgd) = h2o_typed();
    let pg = pgd.pg;
    let (ts, vs) = screened(&ints.t, &ints.v, pg);

    let residue = 1e-14;
    let (mut tr, mut vr) = (ts.clone(), vs.clone());
    tr.set(&[0, 2], residue);
    tr.set(&[2, 0], residue);
    set8(&mut vr, 0, 2, 0, 0, residue);
    let injected: [&[usize]; 2] = [&[0, 2], &[0, 2, 0, 0]];
    for sites in injected {
        assert_ne!(
            fused(pg, sites),
            Irrep::IDENTITY,
            "{sites:?} must be a forbidden class"
        );
    }

    let screen = Su2TermScreen::SymmetryAllowed(pg);
    assert_streams_identical(
        &stream(&tr, &vr, None, &screen),
        &stream(&ts, &vs, None, &screen),
        "h2o SymmetryAllowed: residue vs screened",
    );
    for two_s in [0usize, 2] {
        let space = h2o_space(&ints, &pgd, two_s);
        let a = assemble_screened(space, &tr, &vr, None, &screen, &mut SpinAlgebra::new());
        let b = assemble_screened(space, &ts, &vs, None, &screen, &mut SpinAlgebra::new());
        assert_assembled_identical(a, b, &format!("h2o 2S={two_s} SymmetryAllowed residue"));
    }

    for eps in [0.0, f64::EPSILON] {
        let forbidden_admitted = phase_a_classes(&tr, &vr, |s| fused(pg, s) != Irrep::IDENTITY)
            .iter()
            .filter(|(_, w)| w.abs() > eps)
            .count();
        let mag = Su2TermScreen::Magnitude(eps);
        let with_res = stream(&tr, &vr, None, &mag).len();
        let clean = stream(&ts, &vs, None, &mag).len();
        println!(
            "h2o Magnitude({eps:e}): {with_res} sink calls with the residue, {clean} screened, \
             {forbidden_admitted} forbidden classes above eps"
        );
        assert!(
            forbidden_admitted > 0,
            "the residue must sit above eps = {eps:e}"
        );
        assert_eq!(
            with_res - clean,
            forbidden_admitted,
            "Magnitude({eps:e}) must admit exactly the residue's forbidden classes"
        );
    }
}

#[test]
#[should_panic(expected = "does not cover the three-body channel")]
fn r0_t4_for_each_term_refuses_three_body_under_symmetry_allowed() {
    let d = 4;
    let w = CoeffTensor::<f64>::zeros(d, 6);
    for_each_term(
        d,
        &zeros_matrix::<f64>(d),
        &zeros_4tensor::<f64>(d),
        Some(&w as &dyn ThreeBody<f64>),
        &Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL),
        |_, _| {},
    );
}

#[test]
#[should_panic(expected = "does not cover the three-body channel")]
fn r0_t4_with_screen_refuses_three_body_under_symmetry_allowed() {
    let d = 4;
    let w = CoeffTensor::<f64>::zeros(d, 6);
    let _ = Su2SparseHamiltonian::with_screen(
        Su2Space::new(d, 4, 0),
        &zeros_matrix::<f64>(d),
        &zeros_4tensor::<f64>(d),
        Some(&w as &dyn ThreeBody<f64>),
        Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL),
    );
}

#[test]
fn r0_t5_with_screen_refuses_a_foreign_point_group() {
    let d = 4;
    let irs: Vec<Irrep> = [0u8, 1, 2, 3].iter().map(|&i| Irrep::new(i)).collect();
    let pg = PointGroup::new(2, &irs);
    assert_ne!(pg.irrep_of(0), pg.irrep_of(1), "vacuous: (0, 1) is allowed");
    let typed = Su2Space::with_point_group(d, 4, 0, pg, Irrep::IDENTITY);
    let trivial = Su2Space::new(d, 4, 0);
    for (space, screen) in [
        (trivial, Su2TermScreen::SymmetryAllowed(pg)),
        (typed, Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL)),
    ] {
        let err = std::panic::catch_unwind(|| {
            Su2SparseHamiltonian::with_screen(
                space,
                &zeros_matrix::<f64>(d),
                &zeros_4tensor::<f64>(d),
                None,
                screen.clone(),
            )
        })
        .map(|_| ())
        .expect_err("a foreign SymmetryAllowed point group must be refused");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains("point group must be the space's"),
            "{screen:?} on a space with pg {:?}: wrong panic {msg:?}",
            space.pg
        );
    }
}
