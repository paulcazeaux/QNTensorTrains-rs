// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::coeff::CoeffTensor;
use crate::orbital::{Orbital, Spin, fermionic_sign};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::QnSpace;
use crate::scalar::Scalar;

/// Which canonical term classes Phase A admits.
#[derive(Clone, Debug, PartialEq)]
pub enum TermScreen {
    /// `|W| > eps` on the class's signed combination.
    Magnitude(f64),
    /// Every class the point group allows, regardless of value.
    SymmetryAllowed(PointGroup),
    /// Exactly the classes of a frozen [`PatternMask`], regardless of value.
    Mask(Arc<PatternMask>),
}

impl TermScreen {
    #[inline]
    fn allowed(pg: PointGroup, sites: impl IntoIterator<Item = usize>) -> bool {
        pg.is_trivial() || pg.irrep_of_sites(sites) == Irrep::IDENTITY
    }

    #[inline]
    fn admits_1<T: Scalar>(&self, w: T, i: usize, j: usize) -> bool {
        match self {
            TermScreen::Magnitude(eps) => w.abs_s() > *eps,
            TermScreen::SymmetryAllowed(pg) => Self::allowed(*pg, [i, j]),
            TermScreen::Mask(m) => m.has_one(i, j),
        }
    }

    #[inline]
    fn admits_same<T: Scalar>(&self, w: T, i: usize, j: usize, k: usize, l: usize) -> bool {
        match self {
            TermScreen::Magnitude(eps) => w.abs_s() > *eps,
            TermScreen::SymmetryAllowed(pg) => Self::allowed(*pg, [i, j, k, l]),
            TermScreen::Mask(m) => m.has_same(i, j, k, l),
        }
    }

    #[inline]
    fn admits_mixed<T: Scalar>(&self, w: T, i: usize, j: usize, k: usize, l: usize) -> bool {
        match self {
            TermScreen::Magnitude(eps) => w.abs_s() > *eps,
            TermScreen::SymmetryAllowed(pg) => Self::allowed(*pg, [i, j, k, l]),
            TermScreen::Mask(m) => m.has_mixed(i, j, k, l),
        }
    }

    pub(super) fn three_body_eps(&self) -> f64 {
        match self {
            TermScreen::Magnitude(eps) => *eps,
            TermScreen::SymmetryAllowed(_) | TermScreen::Mask(_) => panic!(
                "a pattern TermScreen (SymmetryAllowed / Mask) does not cover the three-body \
                 channel: its candidate screen is value-driven, so a pattern-frozen (refillable) \
                 build must pass `w = None`"
            ),
        }
    }
}

/// A frozen Phase-A class set: one bit per one-body cell `(i, j)`, per same-spin canonical class
/// `(i<k, l<j)` and per mixed-spin class `(i, j, k, l)`, over the full `d²`/`d⁴` index grids
/// (77 KB per two-body grid at `d = 28`). Built from the magnitude pattern of one integral set
/// ([`Self::from_magnitude`]) and consumed by [`TermScreen::Mask`]; [`Self::leak`] counts the
/// classes a later integral set would ADD under the same magnitude screen — the one thing a
/// frozen skeleton cannot represent, and the signal to re-mint it.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternMask {
    d: usize,
    one: Vec<u64>,
    same: Vec<u64>,
    mixed: Vec<u64>,
    n_classes: usize,
}

#[inline]
fn bit_get(bits: &[u64], idx: usize) -> bool {
    bits[idx >> 6] & (1u64 << (idx & 63)) != 0
}

#[inline]
fn bit_set(bits: &mut [u64], idx: usize) {
    bits[idx >> 6] |= 1u64 << (idx & 63);
}

impl PatternMask {
    /// The classes surviving `|W| > eps` (Phase A's own combinations) AND the point group, on
    /// `(t, v)`.
    pub fn from_magnitude<T: Scalar>(
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        eps: f64,
        pg: PointGroup,
    ) -> Self {
        let d = t.d;
        assert!(
            t.rank == 2 && v.rank == 4 && v.d == d,
            "PatternMask: t must be d×d, v d⁴"
        );
        let words2 = d * d / 64 + 1;
        let words4 = d * d * d * d / 64 + 1;
        let mut m = PatternMask {
            d,
            one: vec![0; words2],
            same: vec![0; words4],
            mixed: vec![0; words4],
            n_classes: 0,
        };
        let screen = TermScreen::Magnitude(eps);
        Self::scan(t, v, pg, |kind, idx, w| {
            let keep = match kind {
                Kind::One(i, j) => screen.admits_1(w, i, j),
                Kind::Same(i, j, k, l) => screen.admits_same(w, i, j, k, l),
                Kind::Mixed(i, j, k, l) => screen.admits_mixed(w, i, j, k, l),
            };
            if keep {
                m.n_classes += 1;
                match kind {
                    Kind::One(..) => bit_set(&mut m.one, idx),
                    Kind::Same(..) => bit_set(&mut m.same, idx),
                    Kind::Mixed(..) => bit_set(&mut m.mixed, idx),
                }
            }
        });
        m
    }

    /// Classes of `(t, v)` that pass `|W| > eps` (and the point group) but are NOT in the mask —
    /// terms a fresh magnitude build would carry and a refill of this skeleton cannot.
    pub fn leak<T: Scalar>(
        &self,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        eps: f64,
        pg: PointGroup,
    ) -> usize {
        assert_eq!(t.d, self.d, "PatternMask::leak: d mismatch");
        let mut n = 0usize;
        Self::scan(t, v, pg, |kind, idx, w| {
            if w.abs_s() > eps {
                let inside = match kind {
                    Kind::One(..) => bit_get(&self.one, idx),
                    Kind::Same(..) => bit_get(&self.same, idx),
                    Kind::Mixed(..) => bit_get(&self.mixed, idx),
                };
                if !inside {
                    n += 1;
                }
            }
        });
        n
    }

    /// Admitted class count.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    #[inline]
    fn has_one(&self, i: usize, j: usize) -> bool {
        bit_get(&self.one, i * self.d + j)
    }
    #[inline]
    fn has_same(&self, i: usize, j: usize, k: usize, l: usize) -> bool {
        bit_get(&self.same, ((i * self.d + j) * self.d + k) * self.d + l)
    }
    #[inline]
    fn has_mixed(&self, i: usize, j: usize, k: usize, l: usize) -> bool {
        bit_get(&self.mixed, ((i * self.d + j) * self.d + k) * self.d + l)
    }

    fn scan<T: Scalar>(
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        pg: PointGroup,
        mut f: impl FnMut(Kind, usize, T),
    ) {
        let d = t.d;
        let half = T::from_f64(0.5);
        for i in 0..d {
            for j in 0..d {
                if TermScreen::allowed(pg, [i, j]) {
                    f(Kind::One(i, j), i * d + j, t.get(&[i, j]));
                }
            }
        }
        if d >= 2 {
            for i in 0..d - 1 {
                for k in i + 1..d {
                    for l in 0..d - 1 {
                        for j in l + 1..d {
                            if !TermScreen::allowed(pg, [i, j, k, l]) {
                                continue;
                            }
                            let w = half
                                * (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])
                                    - v.get(&[i, l, k, j])
                                    - v.get(&[k, j, i, l]));
                            f(Kind::Same(i, j, k, l), ((i * d + j) * d + k) * d + l, w);
                        }
                    }
                }
            }
        }
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        if !TermScreen::allowed(pg, [i, j, k, l]) {
                            continue;
                        }
                        let w = half * (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j]));
                        f(Kind::Mixed(i, j, k, l), ((i * d + j) * d + k) * d + l, w);
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    One(usize, usize),
    Same(usize, usize, usize, usize),
    Mixed(usize, usize, usize, usize),
}

pub(super) fn assert_term_symmetric(space: &QnSpace, cre: &[Orbital], ann: &[Orbital]) {
    if space.pg.is_trivial() {
        return;
    }
    let flux = space
        .pg
        .irrep_of_sites(cre.iter().chain(ann.iter()).map(|o| o.site));
    assert_eq!(
        flux,
        Irrep::IDENTITY,
        "symmetry-forbidden term (cre {cre:?}, ann {ann:?}) with total irrep flux \
         Γ = {flux:?} ≠ IDENTITY survived the ϵ screen: the assembled Hamiltonian would not be \
         totally symmetric. Screen the integrals first \
         (`fcidump::screen_point_group_forbidden`, on by default on the loader paths) or raise \
         `eps` above the dump's forbidden-integral noise floor (~1e-15 for a produced FCIDUMP)."
    );
}

struct SymmetryGated<'a, S> {
    inner: &'a mut S,
    space: &'a QnSpace,
}

impl<T: Scalar, S: TermSink<T>> TermSink<T> for SymmetryGated<'_, S> {
    fn term(&mut self, cre: &[Orbital], ann: &[Orbital], w: T) {
        assert_term_symmetric(self.space, cre, ann);
        self.inner.term(cre, ann, w);
    }
}

pub(super) trait TermSink<T: Scalar> {
    fn term(&mut self, cre: &[Orbital], ann: &[Orbital], w: T);
}

#[cfg(test)]
pub(super) fn enumerate_1body<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    t: &crate::coeff::CoeffTensor<T>,
    eps: f64,
    d: usize,
) {
    enumerate_1body_screened(sink, t, &TermScreen::Magnitude(eps), d)
}

pub(super) fn enumerate_1body_screened<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    t: &crate::coeff::CoeffTensor<T>,
    screen: &TermScreen,
    d: usize,
) {
    for i in 0..d {
        for j in 0..d {
            let tij = t.get(&[i, j]);
            if screen.admits_1(tij, i, j) {
                for spin in [Spin::Up, Spin::Dn] {
                    let io = Orbital::new(i, spin);
                    let jo = Orbital::new(j, spin);
                    let sign = crate::orbital::fermionic_sign(&[io, jo]);
                    let weight = T::from_f64(sign as f64) * tij;
                    sink.term(&[io], &[jo], weight);
                }
            }
        }
    }
}

pub(super) struct TermCollector<T: Scalar> {
    pub(super) terms: Vec<super::super::bipartite::SopTerm<T>>,
}

impl<T: Scalar> TermSink<T> for TermCollector<T> {
    fn term(&mut self, cre: &[Orbital], ann: &[Orbital], w: T) {
        self.terms
            .push(super::super::bipartite::SopTerm::new(cre, ann, w));
    }
}

pub(crate) fn collect_terms<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    eps: f64,
) -> Vec<super::super::bipartite::SopTerm<T>> {
    collect_terms_screened(space, t, v, w, &TermScreen::Magnitude(eps))
}

pub(crate) fn collect_terms_screened<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    screen: &TermScreen,
) -> Vec<super::super::bipartite::SopTerm<T>> {
    let mut sink = TermCollector { terms: Vec::new() };
    enumerate_all_gated(&mut sink, space, t, v, w, screen);
    sink.terms
}

fn enumerate_all_gated<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    screen: &TermScreen,
) {
    let d = space.d;
    let mut gated = SymmetryGated {
        inner: sink,
        space: &space,
    };
    enumerate_1body_screened(&mut gated, t, screen, d);
    enumerate_2body_screened(&mut gated, v, screen, d);
    if let Some(w) = w {
        enumerate_3body(&mut gated, w, screen.three_body_eps());
    }
}

struct GammaCollector<T> {
    gammas: Vec<T>,
}

impl<T: Scalar> TermSink<T> for GammaCollector<T> {
    fn term(&mut self, _cre: &[Orbital], _ann: &[Orbital], w: T) {
        self.gammas.push(w);
    }
}

pub(crate) fn gamma_stream_emission_order<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    w: Option<&dyn crate::coeff::ThreeBody<T>>,
    screen: &TermScreen,
) -> Vec<T> {
    use rayon::prelude::*;
    let d = space.d;
    let mut sink = GammaCollector { gammas: Vec::new() };
    enumerate_1body_screened(
        &mut SymmetryGated {
            inner: &mut sink,
            space: &space,
        },
        t,
        screen,
        d,
    );
    let blocks: Vec<(bool, usize)> = (0..d.saturating_sub(1))
        .map(|i| (true, i))
        .chain((0..d).map(|i| (false, i)))
        .collect();
    let two_body: Vec<Vec<T>> = blocks
        .par_iter()
        .map(|&(same_spin, i)| {
            let mut block = GammaCollector { gammas: Vec::new() };
            let mut gated = SymmetryGated {
                inner: &mut block,
                space: &space,
            };
            if same_spin {
                enumerate_2body_same_spin_i(&mut gated, v, screen, d, i);
            } else {
                enumerate_2body_mixed_spin_i(&mut gated, v, screen, d, i);
            }
            block.gammas
        })
        .collect();
    sink.gammas.reserve(two_body.iter().map(Vec::len).sum());
    for block in two_body {
        sink.gammas.extend(block);
    }
    if let Some(w) = w {
        enumerate_3body(
            &mut SymmetryGated {
                inner: &mut sink,
                space: &space,
            },
            w,
            screen.three_body_eps(),
        );
    }
    sink.gammas
}

#[cfg(test)]
pub(super) fn enumerate_2body<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    v: &crate::coeff::CoeffTensor<T>,
    eps: f64,
    d: usize,
) {
    enumerate_2body_screened(sink, v, &TermScreen::Magnitude(eps), d)
}

pub(super) fn enumerate_2body_screened<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    v: &crate::coeff::CoeffTensor<T>,
    screen: &TermScreen,
    d: usize,
) {
    if d >= 2 {
        for i in 0..d - 1 {
            enumerate_2body_same_spin_i(sink, v, screen, d, i);
        }
    }
    for i in 0..d {
        enumerate_2body_mixed_spin_i(sink, v, screen, d, i);
    }
}

fn enumerate_2body_same_spin_i<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    v: &crate::coeff::CoeffTensor<T>,
    screen: &TermScreen,
    d: usize,
    i: usize,
) {
    let half = T::from_f64(0.5);

    for k in i + 1..d {
        for l in 0..d - 1 {
            for j in l + 1..d {
                let w = half
                    * (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])
                        - v.get(&[i, l, k, j])
                        - v.get(&[k, j, i, l]));
                if screen.admits_same(w, i, j, k, l) {
                    for spin in [Spin::Up, Spin::Dn] {
                        let io = Orbital::new(i, spin);
                        let jo = Orbital::new(j, spin);
                        let ko = Orbital::new(k, spin);
                        let lo = Orbital::new(l, spin);
                        let sign = crate::orbital::fermionic_sign(&[io, ko, lo, jo]);
                        let weight = T::from_f64(sign as f64) * w;
                        sink.term(&[io, ko], &[lo, jo], weight);
                    }
                }
            }
        }
    }
}

fn enumerate_2body_mixed_spin_i<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    v: &crate::coeff::CoeffTensor<T>,
    screen: &TermScreen,
    d: usize,
    i: usize,
) {
    let half = T::from_f64(0.5);

    for j in 0..d {
        for k in 0..d {
            for l in 0..d {
                let w = half * (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j]));
                if screen.admits_mixed(w, i, j, k, l) {
                    let io = Orbital::new(i, Spin::Up);
                    let jo = Orbital::new(j, Spin::Up);
                    let ko = Orbital::new(k, Spin::Dn);
                    let lo = Orbital::new(l, Spin::Dn);
                    let sign = crate::orbital::fermionic_sign(&[io, ko, lo, jo]);
                    let weight = T::from_f64(sign as f64) * w;
                    sink.term(&[io, ko], &[lo, jo], weight);
                }
            }
        }
    }
}

const P3: [([usize; 3], i8); 6] = [
    ([0, 1, 2], 1),
    ([1, 2, 0], 1),
    ([2, 0, 1], 1),
    ([1, 0, 2], -1),
    ([0, 2, 1], -1),
    ([2, 1, 0], -1),
];

fn w3_same<T: Scalar>(w: &dyn crate::coeff::ThreeBody<T>, c: [usize; 3], a: [usize; 3]) -> T {
    let mut acc = T::zero();
    for (pa, sa) in P3 {
        for (pb, sb) in P3 {
            let term = w.entry(&[c[pa[0]], a[pb[0]], c[pa[1]], a[pb[1]], c[pa[2]], a[pb[2]]]);
            acc += T::from_f64((sa * sb) as f64) * term;
        }
    }
    acc * T::from_f64(1.0 / 6.0)
}

#[allow(clippy::too_many_arguments)]
fn w3_ud<T: Scalar>(
    w: &dyn crate::coeff::ThreeBody<T>,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    m: usize,
    n: usize,
) -> T {
    let g = |idx: [usize; 6]| w.entry(&idx);
    let s = g([i, j, k, l, m, n]) + g([k, l, i, j, m, n])
        - g([i, l, k, j, m, n])
        - g([k, j, i, l, m, n])
        + g([i, j, m, n, k, l])
        + g([k, l, m, n, i, j])
        - g([i, l, m, n, k, j])
        - g([k, j, m, n, i, l])
        + g([m, n, i, j, k, l])
        + g([m, n, k, l, i, j])
        - g([m, n, i, l, k, j])
        - g([m, n, k, j, i, l]);
    s * T::from_f64(1.0 / 6.0)
}

#[allow(clippy::too_many_arguments)]
fn w3_du<T: Scalar>(
    w: &dyn crate::coeff::ThreeBody<T>,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    m: usize,
    n: usize,
) -> T {
    let g = |idx: [usize; 6]| w.entry(&idx);
    let s = g([i, j, k, l, m, n]) + g([i, j, m, n, k, l])
        - g([i, j, k, n, m, l])
        - g([i, j, m, l, k, n])
        + g([k, l, i, j, m, n])
        + g([m, n, i, j, k, l])
        - g([k, n, i, j, m, l])
        - g([m, l, i, j, k, n])
        + g([k, l, m, n, i, j])
        + g([m, n, k, l, i, j])
        - g([k, n, m, l, i, j])
        - g([m, l, k, n, i, j]);
    s * T::from_f64(1.0 / 6.0)
}

#[inline]
fn sort3(a: usize, b: usize, c: usize) -> [usize; 3] {
    let (mut a, mut b, mut c) = (a, b, c);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if b > c {
        std::mem::swap(&mut b, &mut c);
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    [a, b, c]
}

fn emit_same<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    w: &dyn crate::coeff::ThreeBody<T>,
    class: [usize; 6],
    eps: f64,
) {
    let [i, k, m, n, l, j] = class;
    let ww = w3_same(w, [i, k, m], [j, l, n]);
    if ww.abs_s() > eps {
        for spin in [Spin::Up, Spin::Dn] {
            let io = Orbital::new(i, spin);
            let jo = Orbital::new(j, spin);
            let ko = Orbital::new(k, spin);
            let lo = Orbital::new(l, spin);
            let mo = Orbital::new(m, spin);
            let no = Orbital::new(n, spin);
            let sign = fermionic_sign(&[io, ko, mo, no, lo, jo]);
            let weight = T::from_f64(sign as f64) * ww;
            sink.term(&[io, ko, mo], &[no, lo, jo], weight);
        }
    }
}

fn emit_ud<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    w: &dyn crate::coeff::ThreeBody<T>,
    class: [usize; 6],
    eps: f64,
) {
    let [i, k, l, j, m, n] = class;
    let ww = w3_ud(w, i, j, k, l, m, n);
    if ww.abs_s() > eps {
        let io = Orbital::new(i, Spin::Up);
        let jo = Orbital::new(j, Spin::Up);
        let ko = Orbital::new(k, Spin::Up);
        let lo = Orbital::new(l, Spin::Up);
        let mo = Orbital::new(m, Spin::Dn);
        let no = Orbital::new(n, Spin::Dn);
        let sign = fermionic_sign(&[io, ko, mo, no, lo, jo]);
        let weight = T::from_f64(sign as f64) * ww;
        sink.term(&[io, ko, mo], &[no, lo, jo], weight);
    }
}

fn emit_du<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    w: &dyn crate::coeff::ThreeBody<T>,
    class: [usize; 6],
    eps: f64,
) {
    let [k, m, n, l, i, j] = class;
    let ww = w3_du(w, i, j, k, l, m, n);
    if ww.abs_s() > eps {
        let io = Orbital::new(i, Spin::Up);
        let jo = Orbital::new(j, Spin::Up);
        let ko = Orbital::new(k, Spin::Dn);
        let lo = Orbital::new(l, Spin::Dn);
        let mo = Orbital::new(m, Spin::Dn);
        let no = Orbital::new(n, Spin::Dn);
        let sign = fermionic_sign(&[io, ko, mo, no, lo, jo]);
        let weight = T::from_f64(sign as f64) * ww;
        sink.term(&[io, ko, mo], &[no, lo, jo], weight);
    }
}

pub(super) fn cand_same(nz: &[[usize; 6]]) -> BTreeSet<[usize; 6]> {
    let mut set = BTreeSet::new();
    for &[a, b, c, e, f, g] in nz {
        if a == c || a == f || c == f || b == e || b == g || e == g {
            continue;
        }
        let [i, k, m] = sort3(a, c, f);
        let [n, l, j] = sort3(b, e, g);
        set.insert([i, k, m, n, l, j]);
    }
    set
}

pub(super) fn cand_mixed(nz: &[[usize; 6]]) -> BTreeSet<[usize; 6]> {
    let mut set = BTreeSet::new();
    for &e in nz {
        let p = [[e[0], e[1]], [e[2], e[3]], [e[4], e[5]]];
        for free in 0..3 {
            let (a, b) = match free {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            let [u1, v1] = p[a];
            let [u2, v2] = p[b];
            if u1 == u2 || v1 == v2 {
                continue;
            }
            set.insert([
                u1.min(u2),
                u1.max(u2),
                v1.min(v2),
                v1.max(v2),
                p[free][0],
                p[free][1],
            ]);
        }
    }
    set
}

pub(super) fn enumerate_3body<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    w: &dyn crate::coeff::ThreeBody<T>,
    eps: f64,
) {
    let tau = eps / 6.0;
    let nz = w.support_above(tau);

    for &class in &cand_same(&nz) {
        emit_same(sink, w, class, eps);
    }
    let mixed = cand_mixed(&nz);
    for &class in &mixed {
        emit_ud(sink, w, class, eps);
    }
    for &class in &mixed {
        emit_du(sink, w, class, eps);
    }
}

#[cfg(test)]
pub(super) fn enumerate_3body_naive<T: Scalar, S: TermSink<T>>(
    sink: &mut S,
    w: &dyn crate::coeff::ThreeBody<T>,
    eps: f64,
) {
    let d = w.norb();
    for i in 0..d {
        for k in i + 1..d {
            for m in k + 1..d {
                for n in 0..d {
                    for l in n + 1..d {
                        for j in l + 1..d {
                            emit_same(sink, w, [i, k, m, n, l, j], eps);
                        }
                    }
                }
            }
        }
    }
    for i in 0..d {
        for k in i + 1..d {
            for l in 0..d {
                for j in l + 1..d {
                    for m in 0..d {
                        for n in 0..d {
                            emit_ud(sink, w, [i, k, l, j, m, n], eps);
                        }
                    }
                }
            }
        }
    }
    for k in 0..d {
        for m in k + 1..d {
            for n in 0..d {
                for l in n + 1..d {
                    for i in 0..d {
                        for j in 0..d {
                            emit_du(sink, w, [k, m, n, l, i, j], eps);
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod gamma_stream_tests {
    use super::*;
    use crate::coeff::CoeffTensor;

    #[test]
    fn threaded_gamma_stream_is_bitwise_the_serial_stream() {
        let d = 6;
        let space = QnSpace::from_n_two_sz(d, 4, 0);
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for x in t.data.iter_mut().chain(v.data.iter_mut()) {
            *x = next();
        }
        for screen in [
            TermScreen::SymmetryAllowed(space.pg),
            TermScreen::Magnitude(0.2),
        ] {
            let mut serial = GammaCollector { gammas: Vec::new() };
            enumerate_all_gated(&mut serial, space, &t, &v, None, &screen);
            let threaded = gamma_stream_emission_order(space, &t, &v, None, &screen);
            assert_eq!(threaded.len(), serial.gammas.len(), "{screen:?}: length");
            assert!(
                threaded
                    .iter()
                    .zip(&serial.gammas)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{screen:?}: not bitwise the serial stream"
            );
            assert!(!threaded.is_empty());
        }
    }
}
