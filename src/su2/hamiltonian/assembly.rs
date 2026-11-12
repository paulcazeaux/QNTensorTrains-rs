// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The SU(2) Hamiltonian automaton: interner, spin-free term enumeration, the parity-preserving
//! x-spin window, `blockclass`, the Phase-B x-spin flatten, the Phase-C recoupling COO fill, and
//! the freeze. SERIAL ONLY — a threaded fill/freeze/CSX path is a benchmark-milestone deferral.

mod automaton;
mod fill;
mod provenance;

pub use automaton::{
    Assembled, Automaton, EdgeKey, VertexId, VertexLabel, add_term, blockclass, chain_state,
    jl_window, set_edges,
};
pub(crate) use automaton::{WireGate, jl_window_gated};
#[allow(unused_imports)]
pub(crate) use fill::Su2NestBuilder;
pub use fill::freeze_cell;
pub(crate) use provenance::{
    BuildRecorder, Recorder, SRC_NONE, Src, Su2CellSrc, Su2Factor, Su2RunSrc, alpha,
    frozen_run_cancels, median_weight,
};

use std::sync::Arc;

use crate::coeff::{CoeffTensor, ThreeBody};
use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;
use crate::su2::qn::{Su2RankGrid, Su2Space};
use crate::su2::spinalgebra::SpinAlgebra;

/// Which spin-free term classes Phase A ([`for_each_term`]) admits — the SU(2) twin of the abelian
/// [`crate::hamiltonian::TermScreen`].
#[derive(Clone, Debug, PartialEq)]
pub enum Su2TermScreen {
    /// `|w| > eps` on the class's weight.
    Magnitude(f64),
    /// Every class the point group allows, regardless of value.
    SymmetryAllowed(PointGroup),
    /// Exactly the classes of a frozen [`Su2PatternMask`], regardless of value.
    Mask(Arc<Su2PatternMask>),
}

impl Su2TermScreen {
    #[inline]
    fn admits<T: Scalar>(&self, w: T, c: TermClass) -> bool {
        match self {
            Su2TermScreen::Magnitude(eps) => w.abs_s() > *eps,
            Su2TermScreen::SymmetryAllowed(pg) => c.allowed(*pg),
            Su2TermScreen::Mask(m) => m.contains(c),
        }
    }

    fn three_body_eps(&self) -> f64 {
        match self {
            Su2TermScreen::Magnitude(eps) => *eps,
            Su2TermScreen::SymmetryAllowed(_) | Su2TermScreen::Mask(_) => panic!(
                "a pattern Su2TermScreen (SymmetryAllowed / Mask) does not cover the three-body \
                 channel: its candidate screen is value-driven, so a pattern screen must pass \
                 `w = None`"
            ),
        }
    }
}

#[derive(Clone, Copy)]
enum TermClass {
    One(usize, usize),
    Two(usize, usize, usize, usize),
}

impl TermClass {
    #[inline]
    fn allowed(self, pg: PointGroup) -> bool {
        pg.is_trivial()
            || match self {
                TermClass::One(i, j) => pg.irrep_of_sites([i, j]),
                TermClass::Two(i, j, k, l) => pg.irrep_of_sites([i, j, k, l]),
            } == Irrep::IDENTITY
    }
}

/// A frozen Phase-A class set (orbital-rotation R3, the SU(2) twin of the abelian
/// [`crate::hamiltonian::PatternMask`]): one bit per one-body cell `(i, j)` over the `d²`
/// grid and per two-body class `(i, j, k, l)`, `(i, j) ≤ (k, l)`, over the `d⁴` grid (the bits
/// with `(i, j) > (k, l)` stay clear; 77 KB at `d = 28`). Minted from the magnitude pattern of one
/// integral set ([`Self::from_magnitude`]) and consumed by [`Su2TermScreen::Mask`];
/// [`Self::leak`] counts the classes a later integral set would ADD under the same screen — the
/// one thing a frozen skeleton cannot represent, and the signal to re-mint it. The mask carries
/// its point group, so it is never read under another.
#[derive(Clone, Debug, PartialEq)]
pub struct Su2PatternMask {
    d: usize,
    pg: PointGroup,
    one: Vec<u64>,
    two: Vec<u64>,
    n_classes: usize,
}

impl Su2PatternMask {
    /// The classes [`for_each_term`] enumerates whose Phase-A weight — its own expressions, read
    /// through the same class scan — has `|w| > eps` (strict: `eps = 0` admits every nonzero
    /// class, never an exact zero) AND whose irrep product is `IDENTITY` under `pg`.
    ///
    /// # Panics
    ///
    /// Unless `t` is `d×d` and `v` is `d⁴` over one `d`.
    pub fn from_magnitude<T: Scalar>(
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        eps: f64,
        pg: PointGroup,
    ) -> Self {
        let d = t.d;
        assert!(
            t.rank == 2 && v.rank == 4 && v.d == d,
            "Su2PatternMask: t must be d×d, v d⁴"
        );
        let mut m = Su2PatternMask {
            d,
            pg,
            one: vec![0; (d * d).div_ceil(64)],
            two: vec![0; (d * d * d * d).div_ceil(64)],
            n_classes: 0,
        };
        let (mag, sym) = (
            Su2TermScreen::Magnitude(eps),
            Su2TermScreen::SymmetryAllowed(pg),
        );
        for_each_class(d, t, v, |c, w| {
            if sym.admits(w, c) && mag.admits(w, c) {
                let (bits, idx) = m.slot_mut(c);
                bits[idx >> 6] |= 1u64 << (idx & 63);
                m.n_classes += 1;
            }
        });
        m
    }

    /// Classes of `(t, v)` passing the minting screen (`|w| > eps` and allowed under `pg`) that
    /// are NOT in the mask — terms a fresh `new(t, v, eps)` would carry and a skeleton minted on
    /// the mask cannot. Additions only: a mask class whose weight dropped is not counted. `eps`
    /// need not be the mint's.
    ///
    /// # Panics
    ///
    /// If `pg` is not the mask's point group, or `(t, v)` are not over the mask's `d`.
    pub fn leak<T: Scalar>(
        &self,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        eps: f64,
        pg: PointGroup,
    ) -> usize {
        assert!(
            pg == self.pg,
            "Su2PatternMask::leak: foreign point group ({pg:?}; the mask was minted under {:?})",
            self.pg
        );
        assert!(
            t.rank == 2 && v.rank == 4,
            "Su2PatternMask::leak: t must be d×d, v d⁴"
        );
        assert!(
            t.d == self.d && v.d == self.d,
            "Su2PatternMask::leak: d mismatch (t over {}, v over {}, the mask over {})",
            t.d,
            v.d,
            self.d
        );
        let (mag, sym) = (
            Su2TermScreen::Magnitude(eps),
            Su2TermScreen::SymmetryAllowed(pg),
        );
        let mut n = 0usize;
        for_each_class(self.d, t, v, |c, w| {
            if sym.admits(w, c) && mag.admits(w, c) && !self.contains(c) {
                n += 1;
            }
        });
        n
    }

    /// Admitted class count (bits set).
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub(super) fn d(&self) -> usize {
        self.d
    }

    pub(super) fn pg(&self) -> PointGroup {
        self.pg
    }

    #[inline]
    fn slot(&self, c: TermClass) -> (&[u64], usize) {
        let d = self.d;
        match c {
            TermClass::One(i, j) => (&self.one, i * d + j),
            TermClass::Two(i, j, k, l) => (&self.two, ((i * d + j) * d + k) * d + l),
        }
    }

    #[inline]
    fn slot_mut(&mut self, c: TermClass) -> (&mut [u64], usize) {
        let d = self.d;
        match c {
            TermClass::One(i, j) => (&mut self.one, i * d + j),
            TermClass::Two(i, j, k, l) => (&mut self.two, ((i * d + j) * d + k) * d + l),
        }
    }

    #[inline]
    fn contains(&self, c: TermClass) -> bool {
        let (bits, idx) = self.slot(c);
        bits[idx >> 6] & (1u64 << (idx & 63)) != 0
    }
}

/// Run the full staged pipeline: interner setup, Phase A spin-free term enumeration (1-/2-/
/// 3-body), Phase B x-spin flatten, Phase C recoupling COO fill, and the freeze. SERIAL; a
/// threaded restatement is a benchmark-milestone deferral (module docs).
pub fn assemble<T: Scalar>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn ThreeBody<T>>,
    eps: f64,
    spin: &mut SpinAlgebra,
) -> Assembled<T> {
    assemble_screened(space, t, v, w, &Su2TermScreen::Magnitude(eps), spin)
}

pub(crate) fn assemble_screened<T: Scalar>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn ThreeBody<T>>,
    screen: &Su2TermScreen,
    spin: &mut SpinAlgebra,
) -> Assembled<T> {
    assemble_with(space, t, v, w, screen, spin, &mut ())
}

pub(crate) fn assemble_recorded<T: Scalar>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
    frozen: bool,
    spin: &mut SpinAlgebra,
) -> (Assembled<T>, BuildRecorder<T>) {
    let mut rec = if frozen {
        BuildRecorder::new_frozen()
    } else {
        BuildRecorder::new()
    };
    let asm = assemble_with(space, t, v, None, screen, spin, &mut rec);
    (asm, rec)
}

fn assemble_with<T: Scalar, R: Recorder<T>>(
    space: Su2Space,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn ThreeBody<T>>,
    screen: &Su2TermScreen,
    spin: &mut SpinAlgebra,
    rec: &mut R,
) -> Assembled<T> {
    match screen {
        Su2TermScreen::SymmetryAllowed(pg) => assert!(
            *pg == space.pg,
            "assemble_screened: the SymmetryAllowed screen's point group must be the space's"
        ),
        Su2TermScreen::Mask(m) => assert!(
            m.pg() == space.pg,
            "assemble_screened: the Mask's point group must be the space's"
        ),
        Su2TermScreen::Magnitude(_) => {}
    }
    let d = space.d;
    assert!(
        d <= 126,
        "d ≤ 126: the assembler's construction-domain cap. Nothing here packs the site index — \
         vertices are keyed on the VertexLabel struct — but the cap is kept: it bounds every \
         downstream byte-packed orbital key, and sits far above any reachable d"
    );
    let hd = (d - 1) / 2;

    let caps_owned: Option<Vec<Su2RankGrid>> =
        (!space.pg.is_trivial()).then(|| crate::su2::qn::max_ranks(&space));
    let caps = caps_owned.as_deref();

    let mut auto = automaton::Automaton::<T>::new();
    for_each_term(d, t, v, w, screen, |tokens, wt| {
        let term = rec.term(wt);
        automaton::add_term_rec(&mut auto, rec, spin, &space, caps, tokens, wt, hd, term);
    });

    fill::finalize(auto, space, spin, rec)
}

pub(crate) fn for_each_term<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn ThreeBody<T>>,
    screen: &Su2TermScreen,
    mut sink: impl FnMut(&[(usize, bool)], T),
) {
    let w = w.map(|w| (w, screen.three_body_eps()));
    if let Su2TermScreen::Mask(m) = screen {
        assert!(
            m.d() == d,
            "for_each_term: the Mask's d ({}) is not the enumeration's ({d})",
            m.d()
        );
    }

    for_each_class(d, t, v, |c, wt| {
        if screen.admits(wt, c) {
            match c {
                TermClass::One(i, j) => sink(&[(i, true), (j, false)], wt),
                TermClass::Two(i, j, k, l) => {
                    sink(&[(i, true), (k, true), (l, false), (j, false)], wt)
                }
            }
        }
    });

    if let Some((w, eps)) = w {
        let sixth = T::from_f64(1.0 / 6.0);
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
}

#[inline]
fn for_each_class<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    mut f: impl FnMut(TermClass, T),
) {
    for i in 0..d {
        for j in 0..d {
            f(TermClass::One(i, j), t.get(&[i, j]));
        }
    }
    let half = T::from_f64(0.5);
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
                    f(TermClass::Two(i, j, k, l), vw);
                }
            }
        }
    }
}

pub(crate) fn phase_a_stream<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> (Vec<T>, Option<u64>) {
    match screen {
        Su2TermScreen::Magnitude(_) => {
            let (gamma, hash) = phase_a_token_stream(d, t, v, screen);
            (gamma, Some(hash))
        }
        Su2TermScreen::SymmetryAllowed(_) | Su2TermScreen::Mask(_) => {
            let mut gamma = Vec::new();
            for_each_term(d, t, v, None, screen, |_, w| gamma.push(w));
            (gamma, None)
        }
    }
}

pub(crate) fn phase_a_token_stream<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    screen: &Su2TermScreen,
) -> (Vec<T>, u64) {
    use std::hash::{Hash, Hasher};
    #[cfg(test)]
    TOKEN_HASHES.with(|n| n.set(n.get() + 1));
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let mut gamma = Vec::new();
    for_each_term(d, t, v, None, screen, |tokens, w| {
        tokens.hash(&mut h);
        gamma.push(w);
    });
    gamma.len().hash(&mut h);
    (gamma, h.finish())
}

#[cfg(test)]
thread_local! {
    static TOKEN_HASHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod screen_tests;

#[cfg(test)]
mod refill_tests;

#[cfg(test)]
mod frozen_tests;

#[cfg(test)]
mod refill2_tests;

#[cfg(test)]
mod mask_tests;

#[cfg(test)]
mod compact_tests;

#[cfg(test)]
mod tests {
    use crate::C64;
    use crate::coeff::CoeffTensor;
    use crate::su2::expand::expand;
    use crate::su2::hamiltonian::Su2SparseHamiltonian;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{
        randn_4tensor, randn_matrix, randomized_su2_state, seeded, su2_hcheck, su2_hcheck3,
        zeros_4tensor, zeros_matrix,
    };
    use rand::RngExt;

    #[test]
    fn s2_b1_gate_g1_one_body_multiple_sectors() {
        for &(d, n, two_s) in &[(6usize, 4usize, 2usize), (6, 3, 1), (5, 4, 0)] {
            let space = Su2Space::new(d, n, two_s);
            let mut rng = seeded(20901 + two_s as u64);
            let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
            for &(i, j) in &[
                (0usize, 0usize),
                (1, 4),
                (4, 1),
                (0, d - 1),
                (d - 1, 0),
                (2, 3),
                (3, 2),
            ] {
                let mut t = zeros_matrix::<f64>(d);
                t.set(&[i, j], rng.sample(rand_distr::StandardNormal));
                let v = zeros_4tensor::<f64>(d);
                let err = su2_hcheck(&t, &v, &x);
                assert!(err <= 1e-11, "G1 single-entry ({i},{j}) err={err}");
            }
            let t = randn_matrix::<f64, _>(d, &mut rng);
            let v = zeros_4tensor::<f64>(d);
            let err = su2_hcheck(&t, &v, &x);
            assert!(err <= 1e-11, "G1 dense t err={err}");
        }
    }

    #[test]
    fn s2_b6_eps_screening_idempotence() {
        let d = 5;
        let n = 3;
        let two_s = 1;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(556);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let t = randn_matrix::<f64, _>(d, &mut rng);
        let v = randn_4tensor::<f64, _>(d, &mut rng);
        let h1 = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let h2 = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-300);
        let y1 = h1.apply(&x);
        let y2 = h2.apply(&x);
        let diff = (&y1 - &y2).norm();
        let norm1 = y1.norm();
        assert!(diff <= 1e-12 * norm1.max(1.0));
    }

    #[test]
    fn s3_construction_determinism_bit_identical() {
        let d = 5;
        let n = 3;
        let two_s = 1;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(20917);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let t = randn_matrix::<f64, _>(d, &mut rng);
        let v = randn_4tensor::<f64, _>(d, &mut rng);
        let mut w = CoeffTensor::<f64>::zeros(d, 6);
        w.set(&[0, 1, 2, 2, 3, 4], rng.sample(rand_distr::StandardNormal));
        let h1 = Su2SparseHamiltonian::new(space, &t, &v, Some(&w), f64::EPSILON);
        let h2 = Su2SparseHamiltonian::new(space, &t, &v, Some(&w), f64::EPSILON);
        let y1 = h1.apply(&x);
        let y2 = h2.apply(&x);
        let a = expand(&y1, two_s as i64).to_dense();
        let b = expand(&y2, two_s as i64).to_dense();
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(&b).all(|(p, q)| p == q),
            "repeated construction is not bit-identical"
        );
    }

    #[test]
    fn s5_e1_complex_cross_symmetry_gate() {
        let d = 5;
        let n = 4;
        let two_s = 2;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(20907);
        let x: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
        let t = randn_matrix::<C64, _>(d, &mut rng);
        let v = randn_4tensor::<C64, _>(d, &mut rng);
        let err = su2_hcheck(&t, &v, &x);
        assert!(err <= 1e-10, "E1 err={err}");
    }

    #[test]
    fn s3_b2_gate_g2_two_body_coincidence_classes() {
        let d = 6;
        let n = 4;
        let two_s = 2;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(20902);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let patterns: [(usize, usize, usize, usize); 16] = [
            (0, 1, 2, 3),
            (1, 0, 3, 2),
            (0, 2, 1, 3),
            (0, 5, 2, 3),
            (0, 0, 1, 2),
            (0, 1, 0, 2),
            (0, 1, 2, 0),
            (1, 0, 0, 2),
            (1, 0, 2, 0),
            (1, 2, 0, 0),
            (0, 0, 1, 1),
            (0, 1, 1, 0),
            (0, 1, 0, 1),
            (0, 0, 0, 1),
            (0, 1, 1, 1),
            (1, 1, 1, 1),
        ];
        for &(i, j, k, l) in &patterns {
            let mut v = zeros_4tensor::<f64>(d);
            v.set(&[i, j, k, l], rng.sample(rand_distr::StandardNormal));
            let t = zeros_matrix::<f64>(d);
            let err = su2_hcheck(&t, &v, &x);
            assert!(err <= 1e-11, "G2 pattern ({i},{j},{k},{l}) err={err}");
        }
    }

    #[test]
    fn s3_b3_gate_g3_dense_random_all_sz() {
        for &(d, n, two_s, seed) in &[(5usize, 4usize, 2usize, 20903u64), (6, 4, 0, 20904)] {
            let space = Su2Space::new(d, n, two_s);
            let mut rng = seeded(seed);
            let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
            let t = randn_matrix::<f64, _>(d, &mut rng);
            let v = randn_4tensor::<f64, _>(d, &mut rng);
            let err = su2_hcheck(&t, &v, &x);
            assert!(err <= 1e-11, "G3 ({d},{n},{two_s}) err={err}");
        }
    }

    #[test]
    fn s4_b7_gate_g4_three_body_patterns() {
        let d = 5;
        let n = 3;
        let two_s = 1;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(20905);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let patterns: [(usize, usize, usize, usize, usize, usize); 5] = [
            (1, 1, 2, 2, 3, 3),
            (0, 1, 2, 2, 3, 4),
            (0, 0, 1, 3, 2, 4),
            (1, 2, 2, 1, 3, 3),
            (0, 1, 1, 0, 2, 2),
        ];
        for &p in &patterns {
            let mut w = CoeffTensor::<f64>::zeros(d, 6);
            w.set(
                &[p.0, p.1, p.2, p.3, p.4, p.5],
                rng.sample(rand_distr::StandardNormal),
            );
            let t = zeros_matrix::<f64>(d);
            let v = zeros_4tensor::<f64>(d);
            let err = su2_hcheck3(&t, &v, &w, &x);
            assert!(err <= 1e-10, "G4 pattern {p:?} err={err}");
        }
        let t0 = randn_matrix::<f64, _>(d, &mut rng);
        let mut t = zeros_matrix::<f64>(d);
        for a in 0..d {
            for b in 0..d {
                t.set(&[a, b], (t0.get(&[a, b]) + t0.get(&[b, a])) * 0.5);
            }
        }
        let mut v = zeros_4tensor::<f64>(d);
        v.set(&[0, 1, 2, 3], 0.3);
        v.set(&[3, 2, 1, 0], 0.3);
        let mut w = CoeffTensor::<f64>::zeros(d, 6);
        w.set(&[1, 1, 2, 2, 3, 3], 0.7);
        w.set(&[0, 1, 2, 2, 3, 4], -0.4);
        let err = su2_hcheck3(&t, &v, &w, &x);
        assert!(err <= 1e-10, "G4 combined t+v+w err={err}");
    }

    #[test]
    fn s4_b7_gate_g4b_rank_three_half_fully_distinct() {
        let d = 6;
        let n = 3;
        let two_s = 1;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(20906);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let patterns: [(usize, usize, usize, usize, usize, usize); 7] = [
            (0, 3, 1, 4, 2, 5),
            (0, 5, 1, 4, 2, 3),
            (1, 3, 0, 4, 2, 5),
            (0, 3, 2, 5, 1, 4),
            (0, 4, 1, 5, 2, 3),
            (1, 5, 0, 3, 2, 4),
            (0, 3, 1, 5, 2, 4),
        ];
        for &p in &patterns {
            let mut sites = [p.0, p.1, p.2, p.3, p.4, p.5];
            sites.sort_unstable();
            assert!(
                sites.windows(2).all(|w| w[0] != w[1]),
                "G4b pattern {p:?} is not fully site-distinct"
            );
            let mut w = CoeffTensor::<f64>::zeros(d, 6);
            w.set(
                &[p.0, p.1, p.2, p.3, p.4, p.5],
                rng.sample(rand_distr::StandardNormal),
            );
            let t = zeros_matrix::<f64>(d);
            let v = zeros_4tensor::<f64>(d);
            let err = su2_hcheck3(&t, &v, &w, &x);
            assert!(err <= 1e-9, "G4b pattern {p:?} err={err}");
        }
    }

    #[test]
    fn s4_b7_gate_g4b_singlet_rank_three_half() {
        let d = 6;
        let n = 4;
        let two_s = 0;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(20916);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let patterns: [(usize, usize, usize, usize, usize, usize); 2] =
            [(0, 3, 1, 4, 2, 5), (0, 5, 1, 4, 2, 3)];
        for &p in &patterns {
            let mut w = CoeffTensor::<f64>::zeros(d, 6);
            w.set(
                &[p.0, p.1, p.2, p.3, p.4, p.5],
                rng.sample(rand_distr::StandardNormal),
            );
            let t = zeros_matrix::<f64>(d);
            let v = zeros_4tensor::<f64>(d);
            let err = su2_hcheck3(&t, &v, &w, &x);
            assert!(err <= 1e-9, "G4b singlet pattern {p:?} err={err}");
        }
    }
}
