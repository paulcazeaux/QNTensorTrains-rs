// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Operator-channel semantics read off the assembled automaton, and the precondition of the
//! environment gauge (`matvec::env_gauge`).

use std::collections::HashMap;

use crate::orbital::{Orbital, Spin};
use crate::qn::{QnSpace, Sector};
use crate::scalar::Scalar;

use super::SparseHamiltonian;
use super::assembly::{ChannelDesc, Construction, OpenIdx, TermScreen, weighted_edge};
use super::bipartite::{ChanKind, PackedOps, key_kind, key_ops};

/// Which side of a bond a channel's coefficient sits on — see the module doc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelPhase {
    /// Applied string carried; `γ` still to the RIGHT.
    Normal,
    /// Remainder string carried; `γ` already deposited to the LEFT.
    Comp,
}

/// A channel's open operator string: creations then annihilations, canonical spin-block order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenOps {
    pub cre: Vec<Orbital>,
    pub ann: Vec<Orbital>,
}

impl OpenOps {
    /// Does the string act on orbital `i` or `i + 1`?
    pub fn touches_pair(&self, i: usize) -> bool {
        self.cre
            .iter()
            .chain(&self.ann)
            .any(|o| o.site == i || o.site == i + 1)
    }
}

pub(crate) fn channel_info(desc: &ChannelDesc, b: usize) -> (ChannelPhase, OpenOps) {
    match desc {
        ChannelDesc::Median(state, open) => {
            let open: Vec<Orbital> = open.iter().flatten().copied().collect();
            let phase = match open.first() {
                Some(o) if o.site < b => ChannelPhase::Normal,
                Some(_) => ChannelPhase::Comp,
                None if state.up.nr + state.dn.nr > 0 => ChannelPhase::Normal,
                None => ChannelPhase::Comp,
            };
            let ncre = match phase {
                ChannelPhase::Normal => (state.up.nl + state.dn.nl) as usize,
                ChannelPhase::Comp => (state.up.nr + state.dn.nr) as usize,
            };
            let (cre, ann) = open.split_at(ncre.min(open.len()));
            (
                phase,
                OpenOps {
                    cre: cre.to_vec(),
                    ann: ann.to_vec(),
                },
            )
        }
        ChannelDesc::Bipartite(key) => {
            let phase = match key_kind(*key) {
                ChanKind::Normal => ChannelPhase::Normal,
                ChanKind::Comp => ChannelPhase::Comp,
            };
            let (cre, ann) = key_ops(*key).orbitals();
            (phase, OpenOps { cre, ann })
        }
    }
}

impl<T: Scalar> SparseHamiltonian<T> {
    /// The phase of channel `c` of sector `s` at bond `b` (see the module doc).
    pub fn channel_phase(&self, b: usize, s: Sector, c: usize) -> ChannelPhase {
        channel_info(&self.channel_descs(b, s)[c], b).0
    }

    /// The open operator string of channel `c` of sector `s` at bond `b`: the applied string for
    /// a Normal channel, the remainder for a Comp one.
    pub fn channel_open(&self, b: usize, s: Sector, c: usize) -> OpenOps {
        channel_info(&self.channel_descs(b, s)[c], b).1
    }
}

fn canonical_key(o: &Orbital, first: Spin) -> (u8, usize) {
    ((o.spin != first) as u8, o.site)
}

fn canonicalize(ops: &mut [Orbital], first: Spin) -> Option<f64> {
    let n = ops.len();
    let mut sign = 1.0f64;
    for a in 1..n {
        let mut b = a;
        while b > 0 && canonical_key(&ops[b - 1], first) > canonical_key(&ops[b], first) {
            ops.swap(b - 1, b);
            sign = -sign;
            b -= 1;
        }
    }
    for w in ops.windows(2) {
        if w[0] == w[1] {
            return None;
        }
    }
    Some(sign)
}

pub(crate) fn expand_under_pair_rotation(
    ops: &OpenOps,
    i: usize,
    g: [f64; 4],
) -> Vec<(OpenOps, f64)> {
    let mut ops = OpenOps {
        cre: ops.cre.clone(),
        ann: ops.ann.clone(),
    };
    let sc0 = canonicalize(&mut ops.cre, Spin::Up).expect("source string has no Pauli duplicate");
    let sa0 = canonicalize(&mut ops.ann, Spin::Dn).expect("source string has no Pauli duplicate");
    let base_sign = sc0 * sa0 * {
        let full: Vec<Orbital> = ops.cre.iter().chain(&ops.ann).copied().collect();
        crate::orbital::fermionic_sign(&full) as f64
    };
    let ops = &ops;
    let slots: Vec<(bool, usize)> = ops
        .cre
        .iter()
        .enumerate()
        .filter(|(_, o)| o.site == i || o.site == i + 1)
        .map(|(k, _)| (true, k))
        .chain(
            ops.ann
                .iter()
                .enumerate()
                .filter(|(_, o)| o.site == i || o.site == i + 1)
                .map(|(k, _)| (false, k)),
        )
        .collect();
    let mut out: Vec<(OpenOps, f64)> = Vec::new();
    for mask in 0..(1usize << slots.len()) {
        let mut cre = ops.cre.clone();
        let mut ann = ops.ann.clone();
        let mut coef = 1.0f64;
        for (bit, &(is_cre, k)) in slots.iter().enumerate() {
            let o = if is_cre { &mut cre[k] } else { &mut ann[k] };
            let old_local = o.site - i;
            let new_local = (mask >> bit) & 1;
            coef *= g[old_local * 2 + new_local];
            o.site = i + new_local;
        }
        if coef == 0.0 {
            continue;
        }
        let Some(sc) = canonicalize(&mut cre, Spin::Up) else {
            continue;
        };
        let Some(sa) = canonicalize(&mut ann, Spin::Dn) else {
            continue;
        };
        let own_sign = {
            let full: Vec<Orbital> = cre.iter().chain(&ann).copied().collect();
            crate::orbital::fermionic_sign(&full) as f64
        };
        let coef = coef * sc * sa * base_sign * own_sign;
        let key = OpenOps { cre, ann };
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some((_, c)) => *c += coef,
            None => out.push((key, coef)),
        }
    }
    out
}

const MAX_PAIR_SLOTS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OrbitMono(u16);

impl OrbitMono {
    fn new(idx: &[u8], neg: bool) -> Self {
        assert!(idx.len() <= MAX_PAIR_SLOTS, "more than six pair slots");
        let mut w = 0u16;
        for (k, &x) in idx.iter().enumerate() {
            w |= (x as u16) << (2 * k);
        }
        OrbitMono(w | ((idx.len() as u16) << 12) | ((neg as u16) << 15))
    }

    #[inline]
    pub(crate) fn eval(self, g: &[f64; 4]) -> Option<f64> {
        let n = ((self.0 >> 12) & 0x7) as usize;
        let mut coef = 1.0f64;
        for k in 0..n {
            coef *= g[((self.0 >> (2 * k)) & 0x3) as usize];
        }
        if coef == 0.0 {
            return None;
        }
        Some(if self.0 >> 15 != 0 { -coef } else { coef })
    }
}

#[inline]
pub(crate) fn orbit_coef(monos: &[OrbitMono], g: &[f64; 4]) -> Option<f64> {
    let mut acc: Option<f64> = None;
    for m in monos {
        if let Some(c) = m.eval(g) {
            acc = Some(match acc {
                None => c,
                Some(a) => a + c,
            });
        }
    }
    acc
}

pub(crate) fn orbit_structure(ops: &OpenOps, i: usize) -> Vec<(OpenOps, Vec<OrbitMono>)> {
    let mut ops = OpenOps {
        cre: ops.cre.clone(),
        ann: ops.ann.clone(),
    };
    let sc0 = canonicalize(&mut ops.cre, Spin::Up).expect("source string has no Pauli duplicate");
    let sa0 = canonicalize(&mut ops.ann, Spin::Dn).expect("source string has no Pauli duplicate");
    let base_sign = sc0 * sa0 * {
        let full: Vec<Orbital> = ops.cre.iter().chain(&ops.ann).copied().collect();
        crate::orbital::fermionic_sign(&full) as f64
    };
    let ops = &ops;
    let slots: Vec<(bool, usize)> = ops
        .cre
        .iter()
        .enumerate()
        .filter(|(_, o)| o.site == i || o.site == i + 1)
        .map(|(k, _)| (true, k))
        .chain(
            ops.ann
                .iter()
                .enumerate()
                .filter(|(_, o)| o.site == i || o.site == i + 1)
                .map(|(k, _)| (false, k)),
        )
        .collect();
    let mut out: Vec<(OpenOps, Vec<OrbitMono>)> = Vec::new();
    let mut idx: Vec<u8> = Vec::with_capacity(slots.len());
    for mask in 0..(1usize << slots.len()) {
        let mut cre = ops.cre.clone();
        let mut ann = ops.ann.clone();
        idx.clear();
        for (bit, &(is_cre, k)) in slots.iter().enumerate() {
            let o = if is_cre { &mut cre[k] } else { &mut ann[k] };
            let old_local = o.site - i;
            let new_local = (mask >> bit) & 1;
            idx.push((old_local * 2 + new_local) as u8);
            o.site = i + new_local;
        }
        let Some(sc) = canonicalize(&mut cre, Spin::Up) else {
            continue;
        };
        let Some(sa) = canonicalize(&mut ann, Spin::Dn) else {
            continue;
        };
        let own_sign = {
            let full: Vec<Orbital> = cre.iter().chain(&ann).copied().collect();
            crate::orbital::fermionic_sign(&full) as f64
        };
        let mono = OrbitMono::new(&idx, sc * sa * base_sign * own_sign < 0.0);
        let key = OpenOps { cre, ann };
        match out.iter_mut().find(|(k, _)| *k == key) {
            Some((_, m)) => m.push(mono),
            None => out.push((key, vec![mono])),
        }
    }
    out
}

/// Why a skeleton fails the gauge precondition.
#[derive(Clone, Debug)]
pub struct OrbitViolation {
    /// The rotated pair `(i, i+1)`.
    pub pair: usize,
    /// The term whose relabelling orbit is not homogeneous (creations, annihilations).
    pub term: (Vec<Orbital>, Vec<Orbital>),
    pub reason: String,
}

impl std::fmt::Display for OrbitViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pair ({}, {}), term cre {:?} ann {:?}: {}",
            self.pair,
            self.pair + 1,
            self.term.0,
            self.term.1,
            self.reason
        )
    }
}

/// The gauge precondition (see the module doc): for every adjacent pair `(i, i+1)` and every term
/// touching it, every canonical relabelling of the term's pair slots is itself a term of the
/// skeleton, and all of them deposit on the same side of every bond outside `[i, i+1]`
/// (`min(switch, i)` and `max(switch, i+1)` agree across the orbit). Returns the number of
/// non-trivial orbits checked.
pub fn switch_orbits_homogeneous<T: Scalar>(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<T>,
    v: &crate::coeff::CoeffTensor<T>,
    screen: &TermScreen,
    construction: Construction,
) -> Result<usize, OrbitViolation> {
    let d = space.d;
    let mut terms = super::assembly::collect_terms_screened(space, t, v, None, screen);
    let switches: HashMap<PackedOps, usize> = match construction {
        Construction::Median => {
            let hd = (d as i64 - 1) / 2;
            terms
                .iter()
                .map(|term| {
                    let (cre, ann) = term.ops.orbitals();
                    (term.ops, weighted_edge(hd, &cre, &ann))
                })
                .collect()
        }
        Construction::Bipartite => super::bipartite::cover_switch_bonds(space, &mut terms, false)
            .into_iter()
            .collect(),
        other => panic!("switch_orbits_homogeneous: construction {other:?} not supported"),
    };
    let generic = [1.0, 1.0, 1.0, 1.0];
    let mut checked = 0usize;
    for term in &terms {
        let (cre, ann) = term.ops.orbitals();
        let ops = OpenOps {
            cre: cre.clone(),
            ann: ann.clone(),
        };
        for i in 0..d - 1 {
            if !ops.touches_pair(i) || space.pg.irrep_of(i) != space.pg.irrep_of(i + 1) {
                continue;
            }
            let orbit = expand_under_pair_rotation(&ops, i, generic);
            if orbit.len() < 2 {
                continue;
            }
            checked += 1;
            let mut lo: Option<usize> = None;
            let mut hi: Option<usize> = None;
            for (member, _) in &orbit {
                let key = PackedOps::new(&member.cre, &member.ann);
                let Some(&sw) = switches.get(&key) else {
                    return Err(OrbitViolation {
                        pair: i,
                        term: (cre, ann),
                        reason: format!(
                            "relabelled string cre {:?} ann {:?} is not a term of the skeleton",
                            member.cre, member.ann
                        ),
                    });
                };
                let (l, h) = (sw.min(i), sw.max(i + 1));
                if *lo.get_or_insert(l) != l || *hi.get_or_insert(h) != h {
                    return Err(OrbitViolation {
                        pair: i,
                        term: (cre, ann),
                        reason: format!(
                            "orbit deposits on both sides of a bond outside the pair (switch {sw} \
                             vs an earlier member's)"
                        ),
                    });
                }
            }
        }
    }
    Ok(checked)
}

pub(crate) fn open_idx_of(ops: &OpenOps) -> OpenIdx {
    let mut out: OpenIdx = [None; 6];
    for (slot, o) in out.iter_mut().zip(ops.cre.iter().chain(&ops.ann)) {
        *slot = Some(*o);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn o(site: usize, up: bool) -> Orbital {
        Orbital::new(site, if up { Spin::Up } else { Spin::Dn })
    }

    #[test]
    fn expand_same_spin_pair_is_det_invariant() {
        let (c, s) = (0.6f64, 0.8f64);
        let g = [c, s, -s, c];
        let ops = OpenOps {
            cre: vec![o(2, true), o(3, true)],
            ann: Vec::new(),
        };
        let out = expand_under_pair_rotation(&ops, 2, g);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].0, ops);
        assert!((out[0].1 - 1.0).abs() < 1e-15, "coef {}", out[0].1);
    }

    #[test]
    fn expand_hop_spreads_over_the_four_classes_with_the_parity_ratio() {
        let g = [0.6, 0.8, -0.8, 0.6];
        let ops = OpenOps {
            cre: vec![o(0, true)],
            ann: vec![o(1, true)],
        };
        let mut out = expand_under_pair_rotation(&ops, 0, g);
        out.sort_by(|a, b| {
            (a.0.cre[0].site, a.0.ann[0].site).cmp(&(b.0.cre[0].site, b.0.ann[0].site))
        });
        assert_eq!(out.len(), 4);
        let base = crate::orbital::fermionic_sign(&[o(0, true), o(1, true)]) as f64;
        for (m, cf) in &out {
            let (p, q) = (m.cre[0].site, m.ann[0].site);
            let bare = g[p] * g[2 + q];
            let own = crate::orbital::fermionic_sign(&[o(p, true), o(q, true)]) as f64;
            let want = bare * base * own;
            assert!((cf - want).abs() < 1e-15, "({p}, {q}): {cf} vs {want}");
        }
        let rev = out
            .iter()
            .find(|(m, _)| m.cre[0].site == 1 && m.ann[0].site == 0)
            .unwrap();
        assert!((rev.1 - 0.64).abs() < 1e-15, "reversed hop: {}", rev.1);
    }

    #[test]
    fn single_slot_relabelling_never_flips_sign() {
        let g = [0.6, 0.8, -0.8, 0.6];
        let ops = OpenOps {
            cre: vec![o(1, true), o(4, true), o(2, false)],
            ann: vec![o(0, false), o(3, true)],
        };
        for i in 0..4 {
            for (member, coef) in expand_under_pair_rotation(&ops, i, g) {
                let mut bare = 1.0;
                for (old, new) in ops.cre.iter().zip(&member.cre) {
                    if old.site == i || old.site == i + 1 {
                        bare *= g[(old.site - i) * 2 + (new.site - i)];
                    }
                }
                for (old, new) in ops.ann.iter().zip(&member.ann) {
                    if old.site == i || old.site == i + 1 {
                        bare *= g[(old.site - i) * 2 + (new.site - i)];
                    }
                }
                assert!(
                    (coef - bare).abs() < 1e-15 || (coef + bare).abs() < 1e-15,
                    "pair {i}: {member:?} coef {coef} bare {bare}"
                );
            }
        }
    }

    #[test]
    fn orbit_structure_matches_the_expansion_bitwise() {
        let d = 4;
        let orbs: Vec<Orbital> = (0..d).flat_map(|s| [o(s, true), o(s, false)]).collect();
        let mut lists: Vec<Vec<Orbital>> = vec![Vec::new()];
        for a in 0..orbs.len() {
            lists.push(vec![orbs[a]]);
            for b in 0..orbs.len() {
                if a != b {
                    lists.push(vec![orbs[a], orbs[b]]);
                }
            }
        }
        let (c, s) = (0.37f64.cos(), 0.37f64.sin());
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let gs = [
            [c, s, -s, c],
            [h, h, -h, h],
            [0.0, 1.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.3, -0.7, 1.1, 0.2],
        ];
        let (mut checked, mut vanished) = (0usize, 0usize);
        for cre in &lists {
            for ann in &lists {
                let ops = OpenOps {
                    cre: cre.clone(),
                    ann: ann.clone(),
                };
                for i in 0..d - 1 {
                    if !ops.touches_pair(i) {
                        continue;
                    }
                    let structure = orbit_structure(&ops, i);
                    for g in &gs {
                        let want = expand_under_pair_rotation(&ops, i, *g);
                        let got: Vec<(OpenOps, f64)> = structure
                            .iter()
                            .filter_map(|(m, monos)| orbit_coef(monos, g).map(|v| (m.clone(), v)))
                            .collect();
                        vanished += structure.len() - got.len();
                        assert_eq!(got.len(), want.len(), "{ops:?} pair {i} g {g:?}");
                        for (m, v) in &want {
                            let (_, w) = got.iter().find(|(k, _)| k == m).unwrap_or_else(|| {
                                panic!("{ops:?} pair {i}: member {m:?} missing")
                            });
                            assert_eq!(
                                w.to_bits(),
                                v.to_bits(),
                                "{ops:?} pair {i} g {g:?} member {m:?}: {w:e} vs {v:e}"
                            );
                        }
                        checked += 1;
                    }
                }
            }
        }
        println!(
            "checked {checked} (string, pair, g) cases; {vanished} members vanished at exact zeros"
        );
        assert!(vanished > 0, "no case exercised a vanishing member");
    }
}
