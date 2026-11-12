// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The operator-action automaton assembler: the Phase-A/B/C machinery behind the
//! `SparseHamiltonian` constructor, with the interning closures as explicit methods on an
//! [`automaton::Automaton`] struct. The parallel Phase-A partition-merge is
//! deferred (the serial path is byte-identical); this stage builds the serial interner + the
//! Phase-B flatten + the Phase-C COO fill (COO only — the CSR/CSC per-site nest is a
//! matvec-kernel concern).

mod automaton;
#[cfg(test)]
mod automaton_tests;
mod enumerate;
#[cfg(test)]
mod enumerate_tests;
mod kernels;
mod types;

pub(crate) use automaton::weighted_edge;
pub(crate) use kernels::{
    NestBuilder, empty_nest, jordanwigner_sign, jw_apply, lr_sectors, op_delta_cands,
    op_irrep_flux, push_coo_cell, src_tag,
};
pub use types::{
    Assembled, ChannelDesc, Construction, CooEntry, CooNest, DEFAULT_CONSTRUCTION, Flux, OpenIdx,
    SpinShift, StateShift,
};
pub(crate) use types::{Op, SRC_NEG, SRC_NONE};

pub use enumerate::{PatternMask, TermScreen};
pub(crate) use enumerate::{collect_terms, collect_terms_screened};

pub use super::bipartite::{
    BlockStat, CompChannelStat, TermSwitch, bipartite_probe, switch_profile,
};

use rustc_hash::FxHashMap;

use crate::qn::QnSpace;
use crate::scalar::Scalar;

use super::construct_probe as probe;
use automaton::Automaton;
use enumerate::{enumerate_1body_screened, enumerate_2body_screened, enumerate_3body};

/// Assemble with the default construction (see [`DEFAULT_CONSTRUCTION`]).
pub fn assemble<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    eps: f64,
) -> Assembled<T> {
    assemble_with(space, t, v, w, eps, DEFAULT_CONSTRUCTION)
}

/// Build the operator-action automaton for a screened one-/two-/three-body Hamiltonian with an
/// explicit [`Construction`] selection. `w = None` omits the three-body channel entirely. Both
/// arms share the Phase-A enumeration verbatim (only the terminal sink differs: `run_term`
/// κ-sweeps for `Median`, `SopTerm` recording for `Bipartite`).
pub fn assemble_with<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    eps: f64,
    construction: Construction,
) -> Assembled<T> {
    assemble_screened(
        space,
        t,
        v,
        w,
        TermScreen::Magnitude(eps),
        construction,
        false,
    )
}

/// [`assemble_with`] under an explicit [`TermScreen`], optionally recording per-value provenance
/// (`record_src`, the refillable build: every COO triple learns which term's `γ` it
/// holds and with which Jordan-Wigner sign, or that it is a transport constant).
/// `Magnitude(eps)` without provenance is exactly [`assemble_with`].
pub fn assemble_screened<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    screen: TermScreen,
    construction: Construction,
    record_src: bool,
) -> Assembled<T> {
    let d = space.d;
    assert!(d <= 126, "d ≤ 126 required by the 7-bit label packing");
    let probe_t0 = std::time::Instant::now();
    let out = assemble_dispatch(space, t, v, w, &screen, construction, record_src);
    probe::set_total(probe_t0.elapsed().as_nanos() as u64);
    out
}

pub(crate) fn gamma_stream<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    screen: &TermScreen,
    construction: Construction,
) -> Vec<T> {
    match construction {
        Construction::Median | Construction::MedianCoverSwitch => {
            enumerate::gamma_stream_emission_order(space, t, v, w, screen)
        }
        Construction::Bipartite
        | Construction::BipartitePerSector
        | Construction::BipartiteMedianSwitch => {
            let mut terms = collect_terms_screened(space, t, v, w, screen);
            super::bipartite::sort_terms(&mut terms);
            terms.into_iter().map(|t| t.gamma).collect()
        }
    }
}

fn assemble_dispatch<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    screen: &TermScreen,
    construction: Construction,
    record_src: bool,
) -> Assembled<T> {
    let d = space.d;
    match construction {
        Construction::Median | Construction::MedianCoverSwitch => {
            let switches = if construction == Construction::MedianCoverSwitch {
                let mut terms = collect_terms_screened(space, t, v, w, screen);
                Some(super::bipartite::cover_switch_bonds(
                    space, &mut terms, true,
                ))
            } else {
                None
            };
            let hd = (d as i64 - 1) / 2;
            let mut auto = Automaton::<T> {
                space,
                hd,
                vlabels: Vec::new(),
                fams: Vec::new(),
                fam_of: FxHashMap::default(),
                n_slots: space.pg.order() as usize * (space.nup + 1) * (space.ndn + 1),
                caps: (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space)),
                edge_keys: Vec::new(),
                edge_data: Vec::new(),
                edge_site: Vec::new(),
                fam_edge_seen: FxHashMap::default(),
                qn_cache: FxHashMap::default(),
                switches,
                early_break: true,
                cur_term: 0,
                record_src,
            };
            {
                let _t = probe::timer(probe::Stage::PhaseA);
                enumerate_1body_screened(&mut auto, t, screen, d);
                enumerate_2body_screened(&mut auto, v, screen, d);
                if let Some(w) = w {
                    enumerate_3body(&mut auto, w, screen.three_body_eps());
                }
            }
            probe::bump(probe::Counter::Vertices, auto.vlabels.len() as u64);
            probe::bump(probe::Counter::Edges, auto.edge_keys.len() as u64);
            automaton::finalize(auto)
        }
        Construction::Bipartite => {
            let terms = {
                let _t = probe::timer(probe::Stage::PhaseA);
                collect_terms_screened(space, t, v, w, screen)
            };
            super::bipartite::assemble_bipartite(space, terms, record_src)
        }
        Construction::BipartitePerSector => {
            let terms = {
                let _t = probe::timer(probe::Stage::PhaseA);
                collect_terms_screened(space, t, v, w, screen)
            };
            super::bipartite::assemble_bipartite_persector(space, terms, record_src)
        }
        Construction::BipartiteMedianSwitch => {
            let terms = {
                let _t = probe::timer(probe::Stage::PhaseA);
                collect_terms_screened(space, t, v, w, screen)
            };
            super::bipartite::assemble_bipartite_median_switch(space, terms, record_src)
        }
    }
}

#[cfg(test)]
mod phase4_tests {
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::test_util::randomized_state;
    use crate::{
        Orbital, Spin, TtVector, adag_i_a_j, adag_i_adag_k_a_l_a_j,
        adag_i_adag_k_adag_m_a_n_a_l_a_j, tt_zeros,
    };
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t1_ham_1body_curated() {
        let d = 10;
        let space = QnSpace::from_n_two_sz(d, 6, 2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1B1);
        let x0: TtVector<f64> = randomized_state(space, 1, 6, &mut rng);

        let pairs_1based = [
            (1, 1),
            (5, 5),
            (1, 2),
            (2, 1),
            (1, d),
            (d, 1),
            (3, 7),
            (7, 3),
            (4, 9),
            (6, 3),
        ];
        for &(i1, j1) in &pairs_1based {
            let (i, j) = (i1 - 1, j1 - 1);
            let tij: f64 = rng.random_range(-1.0..1.0);
            let x_up = adag_i_a_j(
                &x0,
                Orbital::new(i, Spin::Up),
                Orbital::new(j, Spin::Up),
                tij,
            );
            let x_dn = adag_i_a_j(
                &x0,
                Orbital::new(i, Spin::Dn),
                Orbital::new(j, Spin::Dn),
                tij,
            );
            let x = &x_up + &x_dn;

            let mut t = CoeffTensor::<f64>::zeros(d, 2);
            t.set(&[i, j], tij);
            let v = CoeffTensor::<f64>::zeros(d, 4);
            let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
            let y = &h * &x0;

            let diff = &x - &y;
            assert!(
                diff.norm_lr(false) < x.norm_lr(false) * 1e-12,
                "curated (i,j)=({i1},{j1}) mismatch"
            );
        }
    }

    #[test]
    fn t1_ham_1body_dense() {
        let d = 4;
        let space = QnSpace::from_n_two_sz(d, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1B2);
        let x0: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);

        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        let mut x = tt_zeros::<f64>(space);
        for i in 0..d {
            for j in 0..d {
                let tij: f64 = rng.random_range(-1.0..1.0);
                t.set(&[i, j], tij);
                let x_up = adag_i_a_j(
                    &x0,
                    Orbital::new(i, Spin::Up),
                    Orbital::new(j, Spin::Up),
                    tij,
                );
                let x_dn = adag_i_a_j(
                    &x0,
                    Orbital::new(i, Spin::Dn),
                    Orbital::new(j, Spin::Dn),
                    tij,
                );
                x = &(&x + &x_up) + &x_dn;
            }
        }
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let y = &h * &x0;

        let diff = &x - &y;
        assert!(
            diff.norm_lr(false) < x.norm_lr(false) * 1e-12,
            "dense one-body mismatch"
        );
    }

    const FIXED2: [(usize, usize, usize, usize); 35] = [
        (1, 2, 3, 4),
        (1, 3, 2, 4),
        (1, 4, 2, 3),
        (2, 1, 4, 3),
        (3, 1, 4, 2),
        (4, 3, 2, 1),
        (2, 4, 1, 3),
        (1, 6, 3, 4),
        (1, 1, 2, 3),
        (1, 2, 1, 3),
        (1, 2, 3, 1),
        (2, 1, 1, 3),
        (2, 1, 3, 1),
        (2, 3, 1, 1),
        (1, 2, 2, 3),
        (2, 1, 2, 3),
        (2, 2, 1, 3),
        (1, 2, 3, 2),
        (2, 1, 3, 2),
        (1, 3, 2, 2),
        (1, 2, 3, 3),
        (3, 1, 2, 3),
        (3, 3, 1, 2),
        (1, 3, 2, 3),
        (3, 1, 3, 2),
        (3, 2, 3, 1),
        (1, 1, 2, 2),
        (1, 2, 2, 1),
        (1, 2, 1, 2),
        (2, 1, 2, 1),
        (1, 1, 1, 2),
        (1, 2, 1, 1),
        (1, 2, 2, 2),
        (2, 2, 1, 2),
        (2, 2, 2, 2),
    ];

    #[test]
    fn t2_ham_2body() {
        let d = 6usize;
        let space = QnSpace::from_n_two_sz(d, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x2B0);
        let x0: TtVector<f64> = randomized_state(space, 1, 6, &mut rng);

        for &(i1, j1, k1, l1) in FIXED2.iter() {
            let (i, j, k, l) = (i1 - 1, j1 - 1, k1 - 1, l1 - 1);
            let wijkl: f64 = rng.random_range(0.1..3.0);
            let mut x = tt_zeros::<f64>(space);
            for &s1 in &[Spin::Up, Spin::Dn] {
                for &s2 in &[Spin::Up, Spin::Dn] {
                    let oi = Orbital::new(i, s1);
                    let oj = Orbital::new(j, s1);
                    let ok = Orbital::new(k, s2);
                    let ol = Orbital::new(l, s2);
                    if oi != ok && oj != ol {
                        let term = adag_i_adag_k_a_l_a_j(&x0, oi, oj, ok, ol, 0.5 * wijkl);
                        x = &x + &term;
                    }
                }
            }

            let t = CoeffTensor::<f64>::zeros(d, 2);
            let mut v = CoeffTensor::<f64>::zeros(d, 4);
            v.set(&[i, j, k, l], wijkl);
            let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
            let y = &h * &x0;
            let diff = &x - &y;
            assert!(
                diff.norm_lr(false) < x.norm_lr(false) * 1e-12,
                "fixed2 tuple (i,j,k,l)=({i1},{j1},{k1},{l1}) mismatch"
            );
        }
    }

    const FIXED3: [(usize, usize, usize, usize, usize, usize); 8] = [
        (1, 1, 2, 2, 3, 3),
        (1, 2, 2, 1, 3, 3),
        (1, 2, 1, 2, 3, 4),
        (2, 3, 3, 2, 4, 4),
        (3, 3, 2, 4, 1, 5),
        (1, 6, 2, 5, 3, 4),
        (1, 2, 1, 3, 1, 4),
        (2, 1, 3, 1, 4, 1),
    ];

    #[test]
    fn t3_ham_3body() {
        let d = 6usize;
        let space = QnSpace::from_n_two_sz(d, 6, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x3B0);
        let x0: TtVector<f64> = randomized_state(space, 1, 6, &mut rng);

        for &(i1, j1, k1, l1, m1, n1) in FIXED3.iter() {
            let (i, j, k, l, m, n) = (i1 - 1, j1 - 1, k1 - 1, l1 - 1, m1 - 1, n1 - 1);
            let wijklmn: f64 = rng.random_range(0.1..3.0);
            let mut x = tt_zeros::<f64>(space);
            for &s1 in &[Spin::Up, Spin::Dn] {
                for &s2 in &[Spin::Up, Spin::Dn] {
                    for &s3 in &[Spin::Up, Spin::Dn] {
                        let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                        let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                        let (om, on) = (Orbital::new(m, s3), Orbital::new(n, s3));
                        let creations_distinct = oi != ok && oi != om && ok != om;
                        let annihilations_distinct = oj != ol && oj != on && ol != on;
                        if creations_distinct && annihilations_distinct {
                            let term = adag_i_adag_k_adag_m_a_n_a_l_a_j(
                                &x0,
                                oi,
                                oj,
                                ok,
                                ol,
                                om,
                                on,
                                wijklmn / 6.0,
                            );
                            x = &x + &term;
                        }
                    }
                }
            }

            let t = CoeffTensor::<f64>::zeros(d, 2);
            let v = CoeffTensor::<f64>::zeros(d, 4);
            let mut w = CoeffTensor::<f64>::zeros(d, 6);
            w.set(&[i, j, k, l, m, n], wijklmn);
            let h = SparseHamiltonian::new(space, &t, &v, Some(&w), f64::EPSILON);
            let y = &h * &x0;
            let diff = &x - &y;
            assert!(
                diff.norm_lr(false) <= x.norm_lr(false) * 1e-12,
                "fixed3 tuple ({i1},{j1},{k1},{l1},{m1},{n1}) mismatch"
            );
        }
    }
}
