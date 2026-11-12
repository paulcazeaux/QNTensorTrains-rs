// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::coeff::Sparse3Body;
use crate::orbital::Orbital;
use crate::qn::QnSpace;
use rand::RngExt;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use super::Assembled;
use super::automaton::{Automaton, finalize};
use super::enumerate::{
    TermCollector, TermSink, cand_mixed, cand_same, enumerate_3body, enumerate_3body_naive,
};

#[derive(Default)]
struct Recorder {
    terms: Vec<(Vec<Orbital>, Vec<Orbital>, f64)>,
}
impl TermSink<f64> for Recorder {
    fn term(&mut self, cre: &[Orbital], ann: &[Orbital], w: f64) {
        self.terms.push((cre.to_vec(), ann.to_vec(), w));
    }
}

fn assemble_3body_only(
    space: QnSpace,
    w: &dyn crate::coeff::ThreeBody<f64>,
    eps: f64,
    naive: bool,
) -> Assembled<f64> {
    let d = space.d;
    let hd = (d as i64 - 1) / 2;
    let mut auto = Automaton::<f64> {
        space,
        hd,
        vlabels: Vec::new(),
        fams: Vec::new(),
        fam_of: rustc_hash::FxHashMap::default(),
        n_slots: space.pg.order() as usize * (space.nup + 1) * (space.ndn + 1),
        caps: (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space)),
        edge_keys: Vec::new(),
        edge_data: Vec::new(),
        edge_site: Vec::new(),
        fam_edge_seen: rustc_hash::FxHashMap::default(),
        qn_cache: rustc_hash::FxHashMap::default(),
        switches: None,
        early_break: true,
        cur_term: 0,
        record_src: false,
    };
    if naive {
        enumerate_3body_naive(&mut auto, w, eps);
    } else {
        enumerate_3body(&mut auto, w, eps);
    }
    finalize(auto)
}

fn assemble_3body_only_bipartite(
    space: QnSpace,
    w: &dyn crate::coeff::ThreeBody<f64>,
    eps: f64,
    naive: bool,
) -> Assembled<f64> {
    let mut sink = TermCollector::<f64> { terms: Vec::new() };
    if naive {
        enumerate_3body_naive(&mut sink, w, eps);
    } else {
        enumerate_3body(&mut sink, w, eps);
    }
    crate::hamiltonian::bipartite::assemble_bipartite(space, sink.terms, false)
}

fn seeded_w(d: usize, count: usize, seed: u64) -> Sparse3Body<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut w = Sparse3Body::<f64>::zeros(d);
    for _ in 0..count {
        let idx = [
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
        ];
        let x: f64 = rng.sample(StandardNormal);
        w.set(&idx, x);
    }
    w
}

#[test]
fn t7b_cand_same_pin() {
    let s = cand_same(&[[0, 3, 1, 4, 2, 5]]);
    assert_eq!(s.len(), 1);
    assert!(s.contains(&[0, 1, 2, 3, 4, 5]));
    let s2 = cand_same(&[[0, 3, 0, 4, 2, 5]]);
    assert!(s2.is_empty());
}

#[test]
fn t7b_cand_mixed_pin() {
    let s = cand_mixed(&[[0, 3, 1, 4, 2, 5]]);
    assert_eq!(s.len(), 3);
    assert!(s.contains(&[0, 1, 3, 4, 2, 5]));
    assert!(s.contains(&[0, 2, 3, 5, 1, 4]));
    assert!(s.contains(&[1, 2, 4, 5, 0, 3]));
    let s2 = cand_mixed(&[[0, 3, 0, 4, 2, 5]]);
    assert_eq!(s2.len(), 2);
    assert!(s2.contains(&[0, 2, 3, 5, 0, 4]));
    assert!(s2.contains(&[0, 2, 4, 5, 0, 3]));
}

fn seeded_w_mixed(d: usize, count: usize, seed: u64, eps: f64) -> Sparse3Body<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut w = Sparse3Body::<f64>::zeros(d);
    for c in 0..count {
        let idx = [
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
        ];
        let mag = match c % 3 {
            0 => eps / 12.0,
            1 => eps / 3.0,
            _ => 1.0,
        };
        let sign = if rng.random_range(0..2) == 0 {
            1.0
        } else {
            -1.0
        };
        w.set(&idx, sign * mag);
    }
    w
}

#[test]
fn t7b_screen_tau_filter_fires() {
    for seed in [0x7A5EED_u64, 0x0FF1CE, 0xB16B00B5] {
        let eps = 1e-2;
        let tau = eps / 6.0;
        let w = seeded_w_mixed(6, 30, seed, eps);
        let total = w.nnz();
        let kept = crate::coeff::ThreeBody::<f64>::support_above(&w, tau).len();
        assert!(
            kept < total,
            "seed {seed:#x}: support_above dropped nothing ({kept}/{total}) — filter vacuous"
        );
        assert!(kept > 0, "seed {seed:#x}: support_above dropped everything");

        let mut naive = Recorder::default();
        enumerate_3body_naive(&mut naive, &w as &dyn crate::coeff::ThreeBody<f64>, eps);
        let mut screened = Recorder::default();
        enumerate_3body(&mut screened, &w, eps);
        assert!(
            naive.terms == screened.terms,
            "seed {seed:#x}: screened diverged from naive with a firing tau filter \
             (naive {} terms, screened {} terms)",
            naive.terms.len(),
            screened.terms.len()
        );
    }
}

#[test]
fn t7b_screen_sequence_certificate() {
    for seed in [0xC0FFEE_u64, 0x1234_5678, 0xDEAD_BEEF] {
        let w = seeded_w(6, 24, seed);
        let eps = 1e-9;
        let mut naive = Recorder::default();
        enumerate_3body_naive(&mut naive, &w as &dyn crate::coeff::ThreeBody<f64>, eps);
        let mut screened = Recorder::default();
        enumerate_3body(&mut screened, &w, eps);
        assert!(
            naive.terms == screened.terms,
            "screened enumeration diverged from naive at seed {seed:#x} \
             (naive {} terms, screened {} terms)",
            naive.terms.len(),
            screened.terms.len()
        );
        assert!(
            !naive.terms.is_empty(),
            "vacuous certificate at seed {seed:#x}"
        );
    }
}

#[test]
fn t7b_screen_whole_artifact_certificate() {
    let space = QnSpace::new(6, 3, 3);
    for seed in [0xABCD_u64, 0x5555_AAAA] {
        let w = seeded_w(6, 24, seed);
        let eps = 1e-9;
        let a = assemble_3body_only(space, &w, eps, true);
        let b = assemble_3body_only(space, &w, eps, false);
        assert_eq!(a.coo.len(), b.coo.len());
        for k in 0..a.coo.len() {
            let (ga, gb) = (a.coo[k].iter_grid_order(), b.coo[k].iter_grid_order());
            assert_eq!(ga.len(), gb.len(), "cell count mismatch site {k}");
            for (ea, eb) in ga.iter().zip(&gb) {
                assert_eq!(
                    (ea.0, ea.1, ea.2, ea.3),
                    (eb.0, eb.1, eb.2, eb.3),
                    "cell key mismatch site {k} seed {seed:#x}"
                );
                assert_eq!(
                    ea.4.rows, eb.4.rows,
                    "row mismatch site {k} yf {} xf {} seed {seed:#x}",
                    ea.0, ea.2
                );
                assert_eq!(ea.4.cols, eb.4.cols, "col mismatch");
                assert_eq!(ea.4.vals, eb.4.vals, "val mismatch");
            }
        }
    }
}

#[test]
fn t7b_screen_whole_artifact_certificate_bipartite() {
    let space = QnSpace::new(6, 3, 3);
    for seed in [0xABCD_u64, 0x5555_AAAA] {
        let w = seeded_w(6, 24, seed);
        let eps = 1e-9;
        let a = assemble_3body_only_bipartite(space, &w, eps, true);
        let b = assemble_3body_only_bipartite(space, &w, eps, false);
        assert_eq!(a.coo.len(), b.coo.len());
        for k in 0..a.coo.len() {
            let (ga, gb) = (a.coo[k].iter_grid_order(), b.coo[k].iter_grid_order());
            assert_eq!(ga.len(), gb.len(), "cell count mismatch site {k}");
            for (ea, eb) in ga.iter().zip(&gb) {
                assert_eq!(
                    (ea.0, ea.1, ea.2, ea.3),
                    (eb.0, eb.1, eb.2, eb.3),
                    "cell key mismatch site {k} seed {seed:#x}"
                );
                assert_eq!(
                    ea.4.rows, eb.4.rows,
                    "row mismatch site {k} yf {} xf {} seed {seed:#x}",
                    ea.0, ea.2
                );
                assert_eq!(ea.4.cols, eb.4.cols, "col mismatch");
                assert_eq!(ea.4.vals, eb.4.vals, "val mismatch");
            }
        }
        let mut nnz = 0usize;
        for nest in b.coo.iter() {
            for (_, _, _, _, cell) in nest.iter_grid_order() {
                nnz += cell.rows.len();
            }
        }
        assert!(
            nnz > 0,
            "vacuous bipartite 3-body artifact at seed {seed:#x}"
        );
    }
}
