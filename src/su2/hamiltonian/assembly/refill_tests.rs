// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::path::{Path, PathBuf};

use rand::RngExt;

use super::automaton::set_edges_rec;
use super::fill::{finalize, freeze_cell_src};
use super::provenance::{BLOCK, FID_TRANSPORT};
use super::{
    Automaton, BuildRecorder, Recorder, SRC_NONE, Src, Su2TermScreen, alpha, for_each_term,
    freeze_cell, median_weight,
};
use crate::C64;
use crate::coeff::{CoeffTensor, ThreeBody};
use crate::fcidump::PointGroupData;
use crate::hamiltonian::csx::CsxBlock;
use crate::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use crate::pointgroup::PointGroup;
use crate::scalar::Scalar;
use crate::su2::hamiltonian::{StateT, Su2SparseHamiltonian, Ts};
use crate::su2::qn::{Su2Sector, Su2Space};
use crate::su2::spinalgebra::{SiteOp, SpinAlgebra};
use crate::test_util::seeded;

const EPS: f64 = f64::EPSILON;

pub(super) fn dense<T: Scalar>(d: usize, seed: u64) -> (CoeffTensor<T>, CoeffTensor<T>) {
    let mut rng = seeded(seed);
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for x in t.data.iter_mut().chain(v.data.iter_mut()) {
        let re: f64 = rng.random_range(-1.0..1.0);
        let im: f64 = rng.random_range(-1.0..1.0);
        *x = T::from_complex_f64(re, im);
    }
    (t, v)
}

pub(super) fn assert_no_zeros<T: Scalar>(t: &CoeffTensor<T>, v: &CoeffTensor<T>, what: &str) {
    let ok = t
        .data
        .iter()
        .chain(&v.data)
        .all(|x| x.re_s() != 0.0 && (!T::IS_COMPLEX || x.im_s() != 0.0));
    assert!(ok, "{what}: the dense fixture carries an exact zero");
}

pub(super) fn full_class_count(d: usize) -> usize {
    d * d + d * d * (d * d + 1) / 2
}

pub(super) fn testdata(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

pub(super) fn lih_tv() -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let load = load_fixture_dir::<f64>(&testdata("lih_sto3g"), &IntegralsOptions::default())
        .expect("lih_sto3g loads");
    (load.ints.t, load.ints.v)
}

pub(super) fn h2o_typed() -> (Integrals<f64>, PointGroupData) {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: false,
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(&testdata("h2o_sto3g"), &opts).expect("h2o_sto3g loads");
    assert!(
        !load.pg.pg.is_trivial(),
        "h2o_sto3g must resolve to a nontrivial group, or its arms are not on a typed space"
    );
    (load.ints, load.pg)
}

pub(super) fn h2o_space(ints: &Integrals<f64>, pg: &PointGroupData, two_s: usize) -> Su2Space {
    crate::su2::fcidump::su2_space(ints, pg, Some(two_s)).expect("h2o SU(2) space")
}

fn trivial_screens() -> [Su2TermScreen; 2] {
    [
        Su2TermScreen::Magnitude(EPS),
        Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL),
    ]
}

fn plain<T: Scalar>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> Su2SparseHamiltonian<T> {
    match screen {
        Su2TermScreen::Magnitude(eps) => Su2SparseHamiltonian::new(space, t, v, None, *eps),
        _ => Su2SparseHamiltonian::with_screen(space, t, v, None, screen.clone()),
    }
}

fn recorded<T: Scalar>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> Su2SparseHamiltonian<T> {
    Su2SparseHamiltonian::with_screen_recorded(space, t, v, None, screen.clone())
}

pub(super) fn bits<T: Scalar>(x: T) -> (u64, u64) {
    (x.re_s().to_bits(), x.im_s().to_bits())
}

fn val_bits<T: Scalar>(b: &CsxBlock<T>) -> Vec<(u64, u64)> {
    b.vals.iter().map(|&x| bits(x)).collect()
}

pub(super) fn coords<T: Scalar>(b: &CsxBlock<T>) -> Vec<(i32, i32)> {
    b.entries().map(|(maj, min, _)| (maj, min)).collect()
}

pub(super) fn n_slots<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> usize {
    h.nests
        .iter()
        .flat_map(|n| n.cells.iter())
        .map(|c| c.csr.nnz())
        .sum()
}

pub(super) fn assert_same_operator<T: Scalar>(
    a: &Su2SparseHamiltonian<T>,
    b: &Su2SparseHamiltonian<T>,
    what: &str,
) {
    assert!(a.space == b.space, "{what}: spaces differ");
    assert!(a.states == b.states, "{what}: vertex lists (states) differ");
    assert!(
        a.state_qns == b.state_qns,
        "{what}: coupled x-sector lists (state_qns) differ"
    );
    assert_eq!(a.nests.len(), b.nests.len(), "{what}: nest counts differ");
    for (k, (na, nb)) in a.nests.iter().zip(&b.nests).enumerate() {
        assert_eq!(na.offsets, nb.offsets, "{what}: site {k} offsets");
        assert_eq!(na.diag, nb.diag, "{what}: site {k} diagonal index");
        assert_eq!(na.sectors, nb.sectors, "{what}: site {k} sectors");
        assert_eq!(
            na.cells.len(),
            nb.cells.len(),
            "{what}: site {k} cell counts"
        );
        for (c, (ca, cb)) in na.cells.iter().zip(&nb.cells).enumerate() {
            assert_eq!(
                (ca.x_occ, ca.m_ord),
                (cb.x_occ, cb.m_ord),
                "{what}: site {k} cell {c} key"
            );
            for (blk, xa, xb) in [("CSR", &ca.csr, &cb.csr), ("CSC", &ca.csc, &cb.csc)] {
                assert_eq!(xa.ids, xb.ids, "{what}: site {k} cell {c} {blk} ids");
                assert_eq!(
                    xa.starts, xb.starts,
                    "{what}: site {k} cell {c} {blk} starts"
                );
                assert_eq!(xa.index, xb.index, "{what}: site {k} cell {c} {blk} index");
                assert_eq!(
                    val_bits(xa),
                    val_bits(xb),
                    "{what}: site {k} cell {c} {blk} value bits"
                );
            }
        }
    }
}

pub(super) fn assert_plain<T: Scalar>(h: &Su2SparseHamiltonian<T>, what: &str) {
    assert!(
        h.refill_gamma().is_none()
            && h.n_terms().is_none()
            && h.factors().is_none()
            && h.src_hash().is_none(),
        "{what}: a plain build must carry no provenance"
    );
    assert!(
        h.nests
            .iter()
            .flat_map(|n| n.cells.iter())
            .all(|c| c.src.is_none()),
        "{what}: a plain build's cells must carry no provenance"
    );
}

pub(super) fn assert_recorded<T: Scalar>(h: &Su2SparseHamiltonian<T>, what: &str) {
    assert!(
        h.refill_gamma().is_some()
            && h.n_terms().is_some()
            && h.factors().is_some()
            && h.src_hash().is_some(),
        "{what}: a recording build must carry gamma, n_terms, factors and a src_hash"
    );
    assert!(
        h.nests
            .iter()
            .flat_map(|n| n.cells.iter())
            .all(|c| c.src.is_some()),
        "{what}: every cell of a recording build must carry provenance"
    );
}

fn assert_recording_invisible<T: Scalar>(
    what: &str,
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) {
    let p = plain(space, t, v, screen);
    let r = recorded(space, t, v, screen);
    assert_plain(&p, what);
    assert_recorded(&r, what);
    assert!(
        n_slots(&p) > 0,
        "{what}: empty operator (vacuous comparison)"
    );
    assert_same_operator(&r, &p, what);
    assert_eq!(
        r.artifact_hash(),
        p.artifact_hash(),
        "{what}: artifact_hash differs"
    );
    assert_eq!(
        r.skeleton_hash(),
        p.skeleton_hash(),
        "{what}: skeleton_hash differs"
    );
}

#[test]
fn r1a_t1_recording_is_invisible() {
    let d = 5;
    let (tf, vf) = dense::<f64>(d, 0x51A1_1001);
    let (tc, vc) = dense::<C64>(d, 0x51A1_1002);
    assert_no_zeros(&tc, &vc, "C64 d=5");
    let (tl, vl) = lih_tv();
    for screen in &trivial_screens() {
        for (n, two_s) in [(3usize, 1usize), (4, 2)] {
            let space = Su2Space::new(d, n, two_s);
            let tag = format!("random d=5 ({n},{two_s}) {screen:?}");
            assert_recording_invisible(&format!("f64 {tag}"), space, &tf, &vf, screen);
            assert_recording_invisible(&format!("C64 {tag}"), space, &tc, &vc, screen);
        }
        for (n, two_s) in [(4usize, 0usize), (3, 1)] {
            let space = Su2Space::new(6, n, two_s);
            let tag = format!("lih_sto3g (6,{n},{two_s}) {screen:?}");
            assert_recording_invisible(&tag, space, &tl, &vl, screen);
        }
    }
    let (ints, pgd) = h2o_typed();
    let screen = Su2TermScreen::SymmetryAllowed(pgd.pg);
    for two_s in [0usize, 2] {
        let space = h2o_space(&ints, &pgd, two_s);
        let tag = format!("h2o_sto3g C2v 2S={two_s} SymmetryAllowed(pg)");
        assert_recording_invisible(&tag, space, &ints.t, &ints.v, &screen);
    }
}

#[test]
#[should_panic(expected = "recording refuses a three-body channel")]
fn r1a_t1_recording_refuses_three_body() {
    let d = 4;
    let w = CoeffTensor::<f64>::zeros(d, 6);
    let (t, v) = dense::<f64>(d, 0x51A1_1003);
    let _ = Su2SparseHamiltonian::with_screen_recorded(
        Su2Space::new(d, 4, 0),
        &t,
        &v,
        Some(&w as &dyn ThreeBody<f64>),
        Su2TermScreen::Magnitude(EPS),
    );
}

fn assert_gamma_is_stream<T: Scalar>(
    what: &str,
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
    census: usize,
) -> Vec<T> {
    let mut stream: Vec<T> = Vec::new();
    for_each_term(space.d, t, v, None, screen, |_, w| stream.push(w));
    assert_eq!(
        stream.len(),
        census,
        "{what}: sink count ≠ the census class count"
    );
    let h = recorded(space, t, v, screen);
    assert_eq!(
        h.n_terms(),
        Some(stream.len()),
        "{what}: n_terms ≠ the sink count"
    );
    let gamma = h.refill_gamma().expect("a recording build carries gamma");
    assert_eq!(gamma.len(), stream.len(), "{what}: gamma length");
    for (o, (g, w)) in gamma.iter().zip(&stream).enumerate() {
        assert_eq!(
            bits(*g),
            bits(*w),
            "{what}: gamma[{o}] = ({:e}, {:e}) but term #{o} was emitted at ({:e}, {:e})",
            g.re_s(),
            g.im_s(),
            w.re_s(),
            w.im_s()
        );
    }
    gamma.to_vec()
}

#[test]
fn r1a_t2_refill_gamma_is_the_term_stream() {
    let d = 5;
    let (tf, vf) = dense::<f64>(d, 0x51A1_2001);
    let (tc, vc) = dense::<C64>(d, 0x51A1_2002);
    for screen in &trivial_screens() {
        let tag = format!("{screen:?}");
        assert_gamma_is_stream(
            &format!("f64 d=5 (3,1) {tag}"),
            Su2Space::new(d, 3, 1),
            &tf,
            &vf,
            screen,
            full_class_count(d),
        );
        let gc = assert_gamma_is_stream(
            &format!("C64 d=5 (4,2) {tag}"),
            Su2Space::new(d, 4, 2),
            &tc,
            &vc,
            screen,
            full_class_count(d),
        );
        assert!(
            gc.iter().filter(|g| g.im_s() != 0.0).count() > 0,
            "C64 {tag}: gamma has no imaginary part (vacuous complex arm)"
        );
    }
    let (tl, vl) = lih_tv();
    let space = Su2Space::new(6, 3, 1);
    let g_sa = assert_gamma_is_stream(
        "lih (6,3,1) SymmetryAllowed(TRIVIAL)",
        space,
        &tl,
        &vl,
        &Su2TermScreen::SymmetryAllowed(PointGroup::TRIVIAL),
        702,
    );
    let zeros = g_sa.iter().filter(|g| **g == 0.0).count();
    println!(
        "lih SymmetryAllowed(TRIVIAL): gamma {} entries, {zeros} exact zeros",
        g_sa.len()
    );
    assert_eq!(
        zeros,
        702 - 264,
        "lih: gamma must carry the 438 exact-zero terms"
    );
    assert_gamma_is_stream(
        "lih (6,3,1) Magnitude(EPS)",
        space,
        &tl,
        &vl,
        &Su2TermScreen::Magnitude(EPS),
        264,
    );
    let (ints, pgd) = h2o_typed();
    assert_gamma_is_stream(
        "h2o C2v 2S=2 SymmetryAllowed(pg)",
        h2o_space(&ints, &pgd, 2),
        &ints.t,
        &ints.v,
        &Su2TermScreen::SymmetryAllowed(pgd.pg),
        434,
    );
}

#[derive(Debug, Default)]
pub(super) struct Census {
    pub(super) cells: usize,
    pub(super) slots: usize,
    pub(super) spilled: usize,
    pub(super) multi_term: usize,
    pub(super) multi_term_sites: Vec<usize>,
    pub(super) transport_and_term: usize,
    pub(super) transport_contribs: usize,
    pub(super) terms_referenced: usize,
    pub(super) n_terms: usize,
    pub(super) factors: usize,
    pub(super) neg_factors: usize,
    pub(super) imag_vals: usize,
}

pub(super) fn recompute_all<T: Scalar>(what: &str, h: &Su2SparseHamiltonian<T>) -> Census {
    let gamma = h.refill_gamma().expect("recorded: gamma");
    let factors = h.factors().expect("recorded: factors");
    let n_terms = h.n_terms().expect("recorded: n_terms");
    assert_eq!(gamma.len(), n_terms, "{what}: gamma length ≠ n_terms");

    let mut keys: Vec<(Option<u64>, bool, u64)> = factors
        .iter()
        .map(|f| {
            (
                (!f.transport).then_some(f.cw.to_bits()),
                f.neg,
                f.fr.to_bits(),
            )
        })
        .collect();
    keys.sort_unstable();
    let before = keys.len();
    keys.dedup();
    assert_eq!(
        keys.len(),
        before,
        "{what}: the factor table has duplicate entries (not interned)"
    );

    let mut cen = Census {
        n_terms,
        factors: factors.len(),
        neg_factors: factors.iter().filter(|f| f.neg).count(),
        ..Census::default()
    };
    let mut referenced = vec![false; n_terms];
    for (k, nest) in h.nests.iter().enumerate() {
        for (ci, cell) in nest.cells.iter().enumerate() {
            cen.cells += 1;
            let src = cell
                .src
                .as_ref()
                .unwrap_or_else(|| panic!("{what}: site {k} cell {ci} carries no provenance"));
            let nnz = cell.csr.nnz();
            assert_eq!(
                cell.csc.nnz(),
                nnz,
                "{what}: site {k} cell {ci} CSR/CSC nnz"
            );
            let mut want: Vec<T> = Vec::with_capacity(nnz);
            for p in 0..nnz {
                let run: Vec<(u32, u32)> = src.run(p).collect();
                assert!(
                    !run.is_empty(),
                    "{what}: site {k} cell {ci} CSR slot {p}: a stored value with no contribution"
                );
                let mut s = T::zero();
                let mut terms: Vec<u32> = Vec::new();
                let mut n_transport = 0usize;
                for &(term, fid) in &run {
                    let f = factors.get(fid as usize).unwrap_or_else(|| {
                        panic!("{what}: site {k} cell {ci} slot {p}: fid {fid} out of range")
                    });
                    let a = if term == SRC_NONE {
                        assert!(
                            f.transport,
                            "{what}: site {k} cell {ci} slot {p}: SRC_NONE on a median factor"
                        );
                        n_transport += 1;
                        alpha(T::one(), f.neg, f.fr)
                    } else {
                        assert!(
                            !f.transport,
                            "{what}: site {k} cell {ci} slot {p}: term {term} on a transport factor"
                        );
                        let g = *gamma.get(term as usize).unwrap_or_else(|| {
                            panic!("{what}: site {k} cell {ci} slot {p}: term {term} ≥ n_terms")
                        });
                        referenced[term as usize] = true;
                        if !terms.contains(&term) {
                            terms.push(term);
                        }
                        alpha(median_weight(g, f.cw), f.neg, f.fr)
                    };
                    s += a;
                }
                let stored = cell.csr.vals[p];
                assert_eq!(
                    bits(s),
                    bits(stored),
                    "{what}: site {k} cell {ci} CSR slot {p}, run {run:?}: recomputed \
                     ({:e}, {:e}) ≠ stored ({:e}, {:e})",
                    s.re_s(),
                    s.im_s(),
                    stored.re_s(),
                    stored.im_s()
                );
                cen.slots += 1;
                cen.transport_contribs += n_transport;
                if run.len() > 1 {
                    cen.spilled += 1;
                }
                if terms.len() > 1 {
                    cen.multi_term += 1;
                    cen.multi_term_sites.push(k);
                }
                if n_transport > 0 && !terms.is_empty() {
                    cen.transport_and_term += 1;
                }
                if stored.im_s() != 0.0 {
                    cen.imag_vals += 1;
                }
                want.push(s);
            }
            let csr_at = coords(&cell.csr);
            let csc_at = coords(&cell.csc);
            let mut seen = vec![false; nnz];
            for q in 0..nnz {
                let p = src.csc_to_csr(q);
                assert!(
                    p < nnz && !seen[p],
                    "{what}: site {k} cell {ci}: csc_to_csr is not a permutation (q {q} → {p})"
                );
                seen[p] = true;
                let (col, row) = csc_at[q];
                assert_eq!(
                    (row, col),
                    csr_at[p],
                    "{what}: site {k} cell {ci}: CSC slot {q} and CSR slot {p} are different entries"
                );
                assert_eq!(
                    bits(cell.csc.vals[q]),
                    bits(want[p]),
                    "{what}: site {k} cell {ci} CSC slot {q} (CSR slot {p}): stored ≠ recomputed"
                );
            }
        }
    }
    cen.terms_referenced = referenced.iter().filter(|&&r| r).count();
    println!(
        "{what}: {} cells, {} slots; runs > 1 contribution (spilled) {}, runs > 1 term {} at sites \
         {:?}, transport+term runs {}, transport contributions {}; terms referenced {}/{}; \
         factors {} ({} with the JW sign); stored values with im ≠ 0: {}",
        cen.cells,
        cen.slots,
        cen.spilled,
        cen.multi_term,
        cen.multi_term_sites,
        cen.transport_and_term,
        cen.transport_contribs,
        cen.terms_referenced,
        cen.n_terms,
        cen.factors,
        cen.neg_factors,
        cen.imag_vals
    );
    assert!(cen.slots > 0, "{what}: empty operator (vacuous)");
    assert!(
        cen.transport_contribs > 0,
        "{what}: no transport contribution recomputed"
    );
    assert!(
        cen.terms_referenced > 0,
        "{what}: no median contribution recomputed"
    );
    assert!(
        cen.neg_factors > 0,
        "{what}: no JW-signed factor recomputed"
    );
    cen
}

#[test]
fn r1a_t3_every_stored_value_is_recomputable_bitwise() {
    let d = 5;
    let (tf, vf) = dense::<f64>(d, 0x51A1_3001);
    let (tc, vc) = dense::<C64>(d, 0x51A1_3002);
    assert_no_zeros(&tc, &vc, "C64 d=5");
    for screen in &trivial_screens() {
        let tag = format!("{screen:?}");
        let cf = recompute_all(
            &format!("f64 d=5 (3,1) {tag}"),
            &recorded(Su2Space::new(d, 3, 1), &tf, &vf, screen),
        );
        let cc = recompute_all(
            &format!("C64 d=5 (4,2) {tag}"),
            &recorded(Su2Space::new(d, 4, 2), &tc, &vc, screen),
        );
        for (what, cen) in [("f64 (3,1)", &cf), ("C64 (4,2)", &cc)] {
            assert!(
                cen.spilled >= 1,
                "{what} {tag}: no spilled run (vacuous spill path)"
            );
            assert!(
                cen.multi_term >= 1 && cen.multi_term_sites.contains(&(d - 1)),
                "{what} {tag}: no multi-term run at site d−1 = {} (sites {:?})",
                d - 1,
                cen.multi_term_sites
            );
        }
        assert!(
            cc.imag_vals > 0,
            "C64 {tag}: no stored value with a nonzero imaginary part (vacuous complex arm)"
        );
    }
    let (tl, vl) = lih_tv();
    for screen in &trivial_screens() {
        recompute_all(
            &format!("lih (6,3,1) {screen:?}"),
            &recorded(Su2Space::new(6, 3, 1), &tl, &vl, screen),
        );
    }
    let (ints, pgd) = h2o_typed();
    recompute_all(
        "h2o C2v 2S=2 SymmetryAllowed(pg)",
        &recorded(
            h2o_space(&ints, &pgd, 2),
            &ints.t,
            &ints.v,
            &Su2TermScreen::SymmetryAllowed(pgd.pg),
        ),
    );
}

#[test]
fn r1a_t3_helpers_are_the_literal_expressions() {
    fn literal_median<T: Scalar>(w: T, cw: f64) -> T {
        w * T::from_f64(cw)
    }
    fn literal_alpha<T: Scalar>(w: T, neg: bool, fr: f64) -> T {
        let jw: T = if neg { T::from_f64(-1.0) } else { T::one() };
        w * jw * T::from_f64(fr)
    }
    let cws = [
        0.5,
        -(2.0f64 / 3.0).sqrt(),
        1.0 / 3.0,
        std::f64::consts::FRAC_1_SQRT_2,
    ];
    let frs = [0.75, -(1.0f64 / 6.0).sqrt(), 1.0 / 3.0, 1.0 / 7.0, 1e-300];
    let probes_c: [C64; 8] = [
        C64::new(-1.5, 0.0),
        C64::new(1.5, -0.0),
        C64::new(0.0, 2.0),
        C64::new(-0.0, -0.3),
        C64::new(-0.7, 0.3),
        C64::new(0.1, 0.2),
        C64::new(0.0, 0.0),
        C64::new(3e-300, -7e-301),
    ];
    let probes_f: Vec<f64> = probes_c.iter().map(|z| z.re).collect();

    fn grid<T: Scalar>(probes: &[T], cws: &[f64], frs: &[f64]) -> (usize, usize, usize) {
        let (mut checked, mut jw_fold_moves, mut cw_fold_moves) = (0, 0, 0);
        for &w in probes {
            for &cw in cws {
                assert_eq!(
                    bits(median_weight(w, cw)),
                    bits(literal_median(w, cw)),
                    "median_weight({:e}+{:e}i, {cw:e}) is not w * T::from_f64(cw)",
                    w.re_s(),
                    w.im_s()
                );
                for neg in [false, true] {
                    for &fr in frs {
                        let got = alpha(w, neg, fr);
                        let lit = literal_alpha(w, neg, fr);
                        assert_eq!(
                            bits(got),
                            bits(lit),
                            "alpha({:e}+{:e}i, {neg}, {fr:e}) is not w * jw * T::from_f64(fr)",
                            w.re_s(),
                            w.im_s()
                        );
                        checked += 1;
                        let jw_fold = w * T::from_f64(if neg { -fr } else { fr });
                        jw_fold_moves += usize::from(bits(jw_fold) != bits(lit));
                        let staged = literal_alpha(literal_median(w, cw), neg, fr);
                        let cw_fold = literal_alpha(w, neg, cw * fr);
                        cw_fold_moves += usize::from(bits(cw_fold) != bits(staged));
                    }
                }
            }
        }
        (checked, jw_fold_moves, cw_fold_moves)
    }

    let (nf, jf, cf) = grid::<f64>(&probes_f, &cws, &frs);
    let (nc, jc, cc) = grid::<C64>(&probes_c, &cws, &frs);
    println!(
        "helpers: {nf} f64 / {nc} C64 probes; the jw-into-fr fold moves {jf} / {jc}, the \
         cw-into-fr fold moves {cf} / {cc}"
    );
    assert!(
        jc > 0,
        "no C64 probe discriminates the jw-into-fr fold (vacuous grid)"
    );
    assert!(
        cf + cc > 0,
        "no probe discriminates the cw-into-fr fold (vacuous grid)"
    );
}

fn assert_hash_arm<T: Scalar>(
    what: &str,
    space: Su2Space,
    screen: &Su2TermScreen,
    seeds: [u64; 2],
) {
    let (t0, v0) = dense::<T>(space.d, seeds[0]);
    let (t1, v1) = dense::<T>(space.d, seeds[1]);
    assert_no_zeros(&t0, &v0, what);
    assert_no_zeros(&t1, &v1, what);
    let h0 = recorded(space, &t0, &v0, screen);
    let h0b = recorded(space, &t0, &v0, screen);
    let h1 = recorded(space, &t1, &v1, screen);
    assert!(
        h0.src_hash().is_some(),
        "{what}: a recording build has a src_hash"
    );
    assert_eq!(
        h0.n_terms(),
        Some(full_class_count(space.d)),
        "{what}: dense values must admit every class"
    );
    assert_eq!(
        h0.n_terms(),
        h1.n_terms(),
        "{what}: the two value sets admit different terms"
    );
    assert_eq!(
        h0.skeleton_hash(),
        h1.skeleton_hash(),
        "{what}: skeleton_hash moved with the values"
    );
    assert_ne!(
        h0.artifact_hash(),
        h1.artifact_hash(),
        "{what}: artifact_hash blind to the values"
    );
    assert_ne!(
        h0.src_hash(),
        h1.src_hash(),
        "{what}: src_hash blind to gamma"
    );
    assert_eq!(
        h0.artifact_hash(),
        h0b.artifact_hash(),
        "{what}: artifact_hash not deterministic"
    );
    assert_eq!(
        h0.skeleton_hash(),
        h0b.skeleton_hash(),
        "{what}: skeleton_hash not deterministic"
    );
    assert_eq!(
        h0.src_hash(),
        h0b.src_hash(),
        "{what}: src_hash not deterministic"
    );
}

#[test]
fn r1a_t4_skeleton_hash_value_blind_artifact_hash_value_bearing() {
    for screen in &trivial_screens() {
        let tag = format!("{screen:?}");
        assert_hash_arm::<f64>(
            &format!("f64 d=5 (3,1) {tag}"),
            Su2Space::new(5, 3, 1),
            screen,
            [0x51A1_4001, 0x51A1_4002],
        );
        assert_hash_arm::<C64>(
            &format!("C64 d=5 (4,2) {tag}"),
            Su2Space::new(5, 4, 2),
            screen,
            [0x51A1_4003, 0x51A1_4004],
        );
    }

    let space = Su2Space::new(5, 4, 2);
    let (t0, v0) = dense::<C64>(5, 0x51A1_4005);
    let (t1, v1) = {
        let (mut t1, mut v1) = (t0.clone(), v0.clone());
        let mut rng = seeded(0x51A1_4006);
        for x in t1.data.iter_mut().chain(v1.data.iter_mut()) {
            x.im = rng.random_range(-1.0..1.0);
        }
        (t1, v1)
    };
    assert_no_zeros(&t1, &v1, "imaginary-only arm");
    let screen = Su2TermScreen::Magnitude(EPS);
    let h0 = recorded(space, &t0, &v0, &screen);
    let h1 = recorded(space, &t1, &v1, &screen);
    let re_bits = |h: &Su2SparseHamiltonian<C64>| -> Vec<u64> {
        h.nests
            .iter()
            .flat_map(|n| n.cells.iter())
            .flat_map(|c| c.csr.vals.iter().chain(&c.csc.vals))
            .map(|x| x.re.to_bits())
            .collect()
    };
    assert_eq!(
        re_bits(&h0),
        re_bits(&h1),
        "imaginary-only arm: a real part moved (the arm no longer isolates the imaginary bits)"
    );
    assert_eq!(
        h0.skeleton_hash(),
        h1.skeleton_hash(),
        "imaginary-only arm: skeleton moved"
    );
    assert_ne!(
        h0.artifact_hash(),
        h1.artifact_hash(),
        "artifact_hash is blind to the imaginary parts of the values"
    );
}

type Poke = fn(&mut Su2SparseHamiltonian<C64>);

fn first_cell(h: &Su2SparseHamiltonian<C64>) -> (usize, usize) {
    let k = (0..h.nests.len())
        .find(|&k| h.nests[k].sectors.len() >= 2 && !h.nests[k].cells.is_empty())
        .expect("a site with two sectors and a populated cell");
    (k, 0)
}

fn first_state(h: &Su2SparseHamiltonian<C64>) -> (usize, Su2Sector) {
    (1..h.space.d)
        .flat_map(|b| h.space.bond_sectors(b).into_iter().map(move |s| (b, s)))
        .find(|&(b, s)| !h.states[b][s].is_empty())
        .expect("an interior bond with vertices")
}

fn flip_re(x: &mut C64) {
    *x = C64::new(-x.re, x.im);
}

fn flip_im(x: &mut C64) {
    *x = C64::new(x.re, -x.im);
}

#[test]
fn r1a_t4_hashes_cover_every_listed_component() {
    let space = Su2Space::new(5, 4, 2);
    let (t, v) = dense::<C64>(5, 0x51A1_4101);
    let build = || recorded(space, &t, &v, &Su2TermScreen::Magnitude(EPS));
    let base = build();
    let (s0, a0) = (base.skeleton_hash(), base.artifact_hash());
    let structural: [(&str, Poke); 13] = [
        ("states", |h| {
            let (b, s) = first_state(h);
            h.states[b][s][0].1[0] ^= 1;
        }),
        ("state_qns", |h| {
            let (b, s) = first_state(h);
            h.state_qns[b][s][0].two_s += 2;
        }),
        ("nest offsets", |h| {
            let (k, _) = first_cell(h);
            *h.nests[k].offsets.last_mut().unwrap() += 1;
        }),
        ("nest diag", |h| {
            let (k, _) = first_cell(h);
            h.nests[k].diag[0] = h.nests[k].diag[0].wrapping_add(1);
        }),
        ("nest sectors", |h| {
            let (k, _) = first_cell(h);
            h.nests[k].sectors.swap(0, 1);
        }),
        ("cell x_occ", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].x_occ ^= 1;
        }),
        ("cell m_ord", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].m_ord ^= 1;
        }),
        ("CSR ids", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].csr.ids[0] += 1;
        }),
        ("CSR starts", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].csr.starts[0] += 1;
        }),
        ("CSR index", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].csr.index[0] += 1;
        }),
        ("CSC ids", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].csc.ids[0] += 1;
        }),
        ("CSC starts", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].csc.starts[0] += 1;
        }),
        ("CSC index", |h| {
            let (k, c) = first_cell(h);
            h.nests[k].cells[c].csc.index[0] += 1;
        }),
    ];
    let values: [(&str, Poke); 4] = [
        ("CSR value re", |h| {
            let (k, c) = first_cell(h);
            flip_re(&mut h.nests[k].cells[c].csr.vals[0]);
        }),
        ("CSR value im", |h| {
            let (k, c) = first_cell(h);
            flip_im(&mut h.nests[k].cells[c].csr.vals[0]);
        }),
        ("CSC value re", |h| {
            let (k, c) = first_cell(h);
            flip_re(&mut h.nests[k].cells[c].csc.vals[0]);
        }),
        ("CSC value im", |h| {
            let (k, c) = first_cell(h);
            flip_im(&mut h.nests[k].cells[c].csc.vals[0]);
        }),
    ];
    for (name, poke) in structural {
        let mut h = build();
        poke(&mut h);
        assert_ne!(h.skeleton_hash(), s0, "skeleton_hash ignores {name}");
        assert_ne!(h.artifact_hash(), a0, "artifact_hash ignores {name}");
    }
    for (name, poke) in values {
        let mut h = build();
        poke(&mut h);
        assert_eq!(h.skeleton_hash(), s0, "skeleton_hash reads {name}");
        assert_ne!(h.artifact_hash(), a0, "artifact_hash ignores {name}");
    }
}

type SlotAt = (usize, usize, usize);

fn run_at(h: &Su2SparseHamiltonian<C64>, (k, c, p): SlotAt) -> Vec<(u32, u32)> {
    h.nests[k].cells[c]
        .src
        .as_ref()
        .expect("a recorded cell carries provenance")
        .run(p)
        .collect()
}

fn first_slot(
    h: &Su2SparseHamiltonian<C64>,
    pred: fn(&[(u32, u32)]) -> bool,
    what: &str,
) -> SlotAt {
    for (k, n) in h.nests.iter().enumerate() {
        for (c, cell) in n.cells.iter().enumerate() {
            let s = cell
                .src
                .as_ref()
                .expect("a recorded cell carries provenance");
            for p in 0..s.slots() {
                if pred(&s.run(p).collect::<Vec<_>>()) {
                    return (k, c, p);
                }
            }
        }
    }
    panic!("no recorded slot holds {what}")
}

fn first_median_single(h: &Su2SparseHamiltonian<C64>) -> SlotAt {
    first_slot(
        h,
        |r| r.len() == 1 && r[0].0 != SRC_NONE,
        "a single median contribution",
    )
}

fn first_spilled_slot(h: &Su2SparseHamiltonian<C64>) -> SlotAt {
    first_slot(h, |r| r.len() > 1, "a run of more than one contribution")
}

fn median_single_beside_another_median_fid(h: &Su2SparseHamiltonian<C64>) -> (SlotAt, usize) {
    for (k, n) in h.nests.iter().enumerate() {
        for (c, cell) in n.cells.iter().enumerate() {
            let s = cell
                .src
                .as_ref()
                .expect("a recorded cell carries provenance");
            for p in 0..s.slots() {
                let run: Vec<(u32, u32)> = s.run(p).collect();
                if run.len() != 1 || run[0].0 == SRC_NONE {
                    continue;
                }
                let other = (0..)
                    .map_while(|i| s.fid_entry_at(i).map(|a| (i, s.aux[a])))
                    .find(|&(i, w)| i != usize::from(s.lf[p]) && w & FID_TRANSPORT == 0);
                if let Some((i, _)) = other {
                    return ((k, c, p), i);
                }
            }
        }
    }
    panic!("no median single shares its cell's local fid table with another median fid")
}

fn flip_lsb(x: &mut f64) {
    *x = f64::from_bits(x.to_bits() ^ 1);
}

#[test]
fn r1a_t4_src_hash_covers_every_provenance_component() {
    let space = Su2Space::new(5, 4, 2);
    let (t, v) = dense::<C64>(5, 0x51A1_4101);
    let build = || recorded(space, &t, &v, &Su2TermScreen::Magnitude(EPS));
    let base = build();
    let (s0, a0, h0) = (base.skeleton_hash(), base.artifact_hash(), base.src_hash());
    assert!(h0.is_some(), "a recording build has a src_hash");
    let pokes: [(&str, Poke); 12] = [
        ("factor cw", |h| {
            flip_lsb(&mut h.prov.as_mut().unwrap().factors[0].cw)
        }),
        ("factor fr", |h| {
            flip_lsb(&mut h.prov.as_mut().unwrap().factors[0].fr)
        }),
        ("factor neg", |h| {
            let f = &mut h.prov.as_mut().unwrap().factors[0];
            f.neg = !f.neg;
        }),
        ("factor transport", |h| {
            let f = &mut h.prov.as_mut().unwrap().factors[0];
            f.transport = !f.transport;
        }),
        ("gamma entry", |h| {
            flip_lsb(&mut h.prov.as_mut().unwrap().refill_gamma[0].re)
        }),
        ("median slot term", |h| {
            let at @ (k, c, p) = first_median_single(h);
            let (t, f) = run_at(h, at)[0];
            let s = h.nests[k].cells[c].src.as_mut().unwrap();
            let want = if s.is_escape_block(p / BLOCK) {
                let i = s.escape_word_at(p);
                s.aux[i] ^= 1;
                t ^ 1
            } else {
                s.t16[p] ^= 1;
                s.aux[s.block_word_at(p / BLOCK)] + u32::from(s.t16[p])
            };
            assert_eq!(
                run_at(h, at),
                [(want, f)],
                "the poke missed the slot's term"
            );
        }),
        ("median slot lf byte", |h| {
            let (at @ (k, c, p), j) = median_single_beside_another_median_fid(h);
            let (t, f) = run_at(h, at)[0];
            let s = h.nests[k].cells[c].src.as_mut().unwrap();
            let g = s.aux[s.fid_entry_at(j).expect("a local fid table entry")];
            s.lf[p] = u8::try_from(j).expect("a local fid index");
            assert_ne!(g, f, "the poke target is the slot's own fid");
            assert_eq!(run_at(h, at), [(t, g)], "the poke missed the slot's fid");
        }),
        ("local fid table entry", |h| {
            let at @ (k, c, p) = first_median_single(h);
            let (t, f) = run_at(h, at)[0];
            let s = h.nests[k].cells[c].src.as_mut().unwrap();
            let i = s
                .fid_entry_at(usize::from(s.lf[p]))
                .expect("the fixture's first median single indexes a local fid table");
            s.aux[i] ^= 1;
            assert_eq!(
                run_at(h, at),
                [(t, f ^ 1)],
                "the poke missed the slot's table entry"
            );
        }),
        ("block base", |h| {
            let at @ (k, c, p) = first_median_single(h);
            let (t, f) = run_at(h, at)[0];
            let s = h.nests[k].cells[c].src.as_mut().unwrap();
            let b = p / BLOCK;
            assert!(
                !s.is_escape_block(b),
                "the fixture's first median single sits in an escape block (no base to poke)"
            );
            let i = s.block_word_at(b);
            let base = s.aux[i];
            s.aux[i] ^= 1;
            assert_eq!(
                run_at(h, at),
                [(t - base + (base ^ 1), f)],
                "the poke missed the block's base"
            );
        }),
        ("spill word term", |h| {
            let at @ (k, c, _) = first_spilled_slot(h);
            let mut want = run_at(h, at);
            want[0].0 ^= 1;
            h.nests[k].cells[c].src.as_mut().unwrap().spill[0].0 ^= 1;
            assert_eq!(run_at(h, at), want, "the poke missed the run's first term");
        }),
        ("spill word fid", |h| {
            let at @ (k, c, _) = first_spilled_slot(h);
            let mut want = run_at(h, at);
            want[0].1 ^= 1;
            h.nests[k].cells[c].src.as_mut().unwrap().spill[0].1 ^= 1;
            assert_eq!(run_at(h, at), want, "the poke missed the run's first fid");
        }),
        ("csc_perm entry", |h| {
            let (k, c) = first_cell(h);
            let s = h.nests[k].cells[c].src.as_mut().unwrap();
            let p = s.csc_to_csr(0);
            assert_eq!(s.perm16.len(), s.slots(), "the first cell's map is not u16");
            s.perm16[0] ^= 1;
            assert_eq!(s.csc_to_csr(0), p ^ 1, "the poke missed CSC slot 0's map");
        }),
    ];
    for (name, poke) in pokes {
        let mut h = build();
        poke(&mut h);
        assert_ne!(h.src_hash(), h0, "src_hash ignores {name}");
        assert_eq!(h.skeleton_hash(), s0, "skeleton_hash reads {name}");
        assert_eq!(h.artifact_hash(), a0, "artifact_hash reads {name}");
    }
}

#[test]
fn r1a_t5_freeze_cell_src_records_the_summation_order() {
    let factors = [(1.0, 1.0, false, false), (0.0, 1.0, true, true)];
    let gamma = [1e16, 3.0, 1.0, 1.0, 0.5, 0.25, -0.25];
    let alpha_of = |(term, fid): (u32, u32)| -> f64 {
        let (cw, fr, neg, transport) = factors[fid as usize];
        if transport {
            assert_eq!(term, SRC_NONE, "a transport word carries no term");
            alpha(1.0, neg, fr)
        } else {
            alpha(median_weight(gamma[term as usize], cw), neg, fr)
        }
    };
    let staged: [((i32, i32), (u32, u32)); 8] = [
        ((0, 1), (0, 0)),
        ((1, 0), (5, 0)),
        ((0, 0), (1, 0)),
        ((0, 1), (2, 0)),
        ((0, 0), (SRC_NONE, 1)),
        ((1, 0), (6, 0)),
        ((0, 1), (3, 0)),
        ((0, 0), (4, 0)),
    ];
    let mut cell: Vec<(i32, i32, f64)> = staged
        .iter()
        .map(|&((i, j), word)| (i, j, alpha_of(word)))
        .collect();
    let src: Vec<(u32, u32)> = staged.iter().map(|&(_, word)| word).collect();
    let mut plain_cell = cell.clone();
    let plain = freeze_cell(&mut plain_cell);
    let (coo, runs) = freeze_cell_src(&mut cell, &src);

    let tbits = |c: &[(i32, i32, f64)]| -> Vec<(i32, i32, u64)> {
        c.iter().map(|&(i, j, x)| (i, j, x.to_bits())).collect()
    };
    assert_eq!(
        tbits(&cell),
        tbits(&plain_cell),
        "freeze_cell_src sorted the cell differently from freeze_cell"
    );
    assert_eq!(
        (&coo.rows, &coo.cols),
        (&plain.rows, &plain.cols),
        "frozen keys differ from freeze_cell's"
    );
    assert_eq!(
        coo.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        plain.vals.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "frozen values differ from freeze_cell's"
    );
    assert_eq!(
        (coo.rows.clone(), coo.cols.clone()),
        (vec![0, 0], vec![0, 1]),
        "the kept entries are (0,0) and (0,1), the (1,0) debris dropped"
    );
    assert_eq!(runs.slots(), 2, "a dropped run must record nothing");

    let want: [&[(u32, u32)]; 2] = [&[(1, 0), (SRC_NONE, 1), (4, 0)], &[(0, 0), (2, 0), (3, 0)]];
    for p in 0..2 {
        assert_eq!(
            runs.run(p),
            want[p],
            "slot {p}: recorded contributions are not the staging (summation) order"
        );
        let mut s = 0.0f64;
        for &word in runs.run(p) {
            s += alpha_of(word);
        }
        assert_eq!(
            s.to_bits(),
            coo.vals[p].to_bits(),
            "slot {p}: recomputed {s:e} ≠ stored {:e}",
            coo.vals[p]
        );
    }
    let mut rev = 0.0f64;
    for &word in runs.run(1).iter().rev() {
        rev += alpha_of(word);
    }
    println!(
        "run (0,1): stored {:e} (summation order), reversed {rev:e}",
        coo.vals[1]
    );
    assert_ne!(
        rev.to_bits(),
        coo.vals[1].to_bits(),
        "run (0,1) is not order-dependent (vacuous order pin)"
    );
}

#[test]
fn r1a_t5_recorder_lists_are_the_automaton_lists() {
    let space = Su2Space::new(2, 2, 0);
    let mut auto = Automaton::<f64>::new();
    let mut rec = BuildRecorder::<f64>::new();
    let (g0, g1, cw0, cw1) = (4.0, 12.0, 0.5, 0.25);
    let med0 = Src::Median {
        term: rec.term(g0),
        cw: cw0,
    };
    let med1 = Src::Median {
        term: rec.term(g1),
        cw: cw1,
    };
    let id = SiteOp {
        two_k: 0,
        flux: 0,
        odd: false,
        red: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let num = SiteOp {
        red: [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        ..id.clone()
    };
    let st: Ts = (StateT::ZERO, [0, 0]);
    let calls: [(Option<&SiteOp>, Option<f64>, Src); 5] = [
        (None, Some(median_weight(g0, cw0)), med0),
        (None, None, Src::Transport),
        (None, Some(median_weight(g1, cw1)), med1),
        (Some(&num), None, Src::Transport),
        (Some(&num), None, Src::Transport),
    ];
    for (op, w, src) in calls {
        set_edges_rec(&mut auto, &mut rec, &space, None, 0, op, w, src, st, st);
    }
    assert_eq!(auto.edges.len(), 3, "the three bond-0 edges");
    for (&key, head) in &auto.edges {
        let mut list: Vec<(&SiteOp, f64)> = vec![(&head.0, head.1)];
        list.extend(
            auto.edges_ovf
                .get(&key)
                .into_iter()
                .flatten()
                .map(|(o, w)| (o, *w)),
        );
        assert!(
            list.len() == 3
                && *list[0].0 == id
                && *list[1].0 == id
                && *list[2].0 == num
                && [list[0].1, list[1].1, list[2].1] == [2.0, 3.0, 1.0],
            "edge {key:#x}: the automaton holds {list:?}, not [id·2, id·3, num·1]"
        );
        assert_eq!(
            rec.contributions(key),
            3,
            "edge {key:#x}: recorded contribution count"
        );
        assert_eq!(
            [rec.src(key, 0), rec.src(key, 1), rec.src(key, 2)],
            [med0, med1, Src::Transport],
            "edge {key:#x}: recorded provenance list"
        );
    }

    let mut spin = SpinAlgebra::new();
    let asm = finalize(auto, space, &mut spin, &mut rec);
    let (gamma, factors, sites) = rec.into_parts();
    assert_eq!(
        sites.len(),
        asm.coo.len(),
        "one run-provenance site per nest"
    );
    let (mut slots, mut mixed) = (0usize, 0usize);
    for (k, (nest, runs)) in asm.coo.iter().zip(&sites).enumerate() {
        assert_eq!(runs.len(), nest.cells.len(), "site {k}: runs per cell");
        for (c, (e, r)) in nest.cells.iter().zip(runs).enumerate() {
            assert_eq!(r.slots(), e.cell.vals.len(), "site {k} cell {c}: slots");
            for p in 0..r.slots() {
                let mut s = 0.0f64;
                let (mut med, mut tra) = (false, false);
                for &(term, fid) in r.run(p) {
                    let f = factors[fid as usize];
                    s += if term == SRC_NONE {
                        assert!(f.transport, "SRC_NONE on a median factor");
                        tra = true;
                        alpha(1.0, f.neg, f.fr)
                    } else {
                        assert!(!f.transport, "a term on a transport factor");
                        med = true;
                        alpha(median_weight(gamma[term as usize], f.cw), f.neg, f.fr)
                    };
                }
                assert_eq!(
                    s.to_bits(),
                    e.cell.vals[p].to_bits(),
                    "site {k} cell {c} slot {p}: recomputed {s:e} ≠ stored {:e}",
                    e.cell.vals[p]
                );
                slots += 1;
                mixed += usize::from(med && tra);
            }
        }
    }
    println!("hand-built automaton: {slots} stored values, {mixed} transport+median runs");
    assert!(
        mixed > 0,
        "no stored run mixes a transport and a median (vacuous)"
    );
}
