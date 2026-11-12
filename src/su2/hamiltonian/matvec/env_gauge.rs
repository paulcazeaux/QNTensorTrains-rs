// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::borrow::Cow;
use std::ops::Range;

use rayon::prelude::*;

use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::channel::Su2TouchIndex;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, Su2TermScreen};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid};
use crate::su2::vector::Su2TtVector;

use super::sketching::{Along, mix_bands};
use super::{Su2OneSiteWorkspace, su2_onesite_workspace};

struct CachedWorkspace {
    row_ranks: Su2RankGrid,
    col_ranks: Su2RankGrid,
    ws: Su2OneSiteWorkspace,
}

pub(crate) struct Su2GaugeCache {
    slots: Vec<Option<CachedWorkspace>>,
    builds: usize,
}

impl Su2GaugeCache {
    pub(crate) fn new(d: usize) -> Self {
        Su2GaugeCache {
            slots: (0..d).map(|_| None).collect(),
            builds: 0,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn builds(&self) -> usize {
        self.builds
    }
}

fn workspace<'s, T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    xk: &Su2Core<T>,
    slot: &'s mut Option<CachedWorkspace>,
    builds: &mut usize,
) -> &'s Su2OneSiteWorkspace {
    let hit = slot
        .as_ref()
        .is_some_and(|c| c.row_ranks == xk.row_ranks && c.col_ranks == xk.col_ranks);
    if !hit {
        *builds += 1;
        *slot = Some(CachedWorkspace {
            row_ranks: xk.row_ranks.clone(),
            col_ranks: xk.col_ranks.clone(),
            ws: su2_onesite_workspace(h, xk),
        });
    }
    &slot.as_ref().expect("filled above").ws
}

fn keeps_nothing(keep: &[(Su2Sector, Vec<usize>)]) -> bool {
    keep.iter().all(|(_, cs)| cs.is_empty())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Su2GaugeMode {
    Recontract,
    MixA,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(crate) enum Su2GaugeFault {
    Intact,
    SwapOutputs,
    Unmixed(Vec<(usize, Su2Sector, u32)>),
    NoMix,
    SkipRecontract,
}

type Pairs = [(Su2Sector, Vec<[u32; 2]>)];

#[cfg(not(test))]
#[derive(Clone, Copy)]
struct Inject<'f>(std::marker::PhantomData<&'f ()>);

#[cfg(not(test))]
impl Inject<'_> {
    fn none() -> Self {
        Inject(std::marker::PhantomData)
    }

    fn mixes(self) -> bool {
        true
    }

    fn recontracts(self) -> bool {
        true
    }

    fn pairs(self, _b: usize, pairs: &Pairs) -> Cow<'_, Pairs> {
        Cow::Borrowed(pairs)
    }

    fn mix<T: Scalar>(
        self,
        dst: &mut Su2Sketch<T>,
        pairs: &Pairs,
        tab: &Su2SectorGrid<Vec<Range<usize>>>,
        along: Along,
        (c, s): (f64, f64),
    ) {
        mix_bands(dst, pairs, tab, along, c, s);
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
struct Inject<'f>(&'f Su2GaugeFault);

#[cfg(test)]
impl Inject<'_> {
    fn none() -> Self {
        Inject(&Su2GaugeFault::Intact)
    }

    fn mixes(self) -> bool {
        !matches!(self.0, Su2GaugeFault::NoMix)
    }

    fn recontracts(self) -> bool {
        !matches!(self.0, Su2GaugeFault::SkipRecontract)
    }

    fn pairs(self, b: usize, pairs: &Pairs) -> Cow<'_, Pairs> {
        let Su2GaugeFault::Unmixed(gone) = self.0 else {
            return Cow::Borrowed(pairs);
        };
        Cow::Owned(
            pairs
                .iter()
                .map(|(s, ps)| {
                    let kept = ps
                        .iter()
                        .copied()
                        .filter(|&[lo, _]| !gone.contains(&(b, *s, lo)))
                        .collect();
                    (*s, kept)
                })
                .collect(),
        )
    }

    fn mix<T: Scalar>(
        self,
        dst: &mut Su2Sketch<T>,
        pairs: &Pairs,
        tab: &Su2SectorGrid<Vec<Range<usize>>>,
        along: Along,
        (c, s): (f64, f64),
    ) {
        if matches!(self.0, Su2GaugeFault::SwapOutputs) {
            super::sketching::mix_bands_swapped(dst, pairs, tab, along, c, s);
        } else {
            mix_bands(dst, pairs, tab, along, c, s);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gauge_after_pair_rotation<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    i: usize,
    theta: f64,
    idx: &Su2TouchIndex,
    fl: &mut [Su2Sketch<T>],
    fr: &mut [Su2Sketch<T>],
    cache: &mut Su2GaugeCache,
    mode: Su2GaugeMode,
) {
    match mode {
        Su2GaugeMode::Recontract => {
            check_domains(h, x, i, fl, fr, cache);
            chains(h, x, i, idx, fl, fr, cache, Plan::Recontract);
        }
        Su2GaugeMode::MixA => mix_a(h, x, i, theta, idx, fl, fr, cache, Inject::none()),
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gauge_after_pair_rotation_faulty<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    i: usize,
    theta: f64,
    idx: &Su2TouchIndex,
    fl: &mut [Su2Sketch<T>],
    fr: &mut [Su2Sketch<T>],
    cache: &mut Su2GaugeCache,
    fault: &Su2GaugeFault,
) {
    mix_a(h, x, i, theta, idx, fl, fr, cache, Inject(fault));
}

fn check_domains<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    i: usize,
    fl: &[Su2Sketch<T>],
    fr: &[Su2Sketch<T>],
    cache: &Su2GaugeCache,
) {
    assert!(
        h.is_refillable(),
        "gauge_after_pair_rotation refuses a non-refillable operator (a three-body label lists \
         only part of a side, so the touched set can miss stale bands) — build it with \
         `new_refillable`"
    );
    let d = x.space.d;
    assert_eq!(
        h.space.d, d,
        "gauge_after_pair_rotation: operator and state chain lengths"
    );
    assert!(
        i + 1 < d,
        "gauge_after_pair_rotation: the pair ({i}, {}) is not in a chain of {d} sites",
        i + 1
    );
    assert_eq!(
        (fl.len(), fr.len()),
        (i + 1, d - i - 1),
        "gauge_after_pair_rotation: the domains of the pair ({i}, {}) are Fᴸ bonds 0..={i} and Fᴿ \
         slots {}..{d} (lengths {}, {})",
        i + 1,
        i + 1,
        i + 1,
        d - i - 1
    );
    assert_eq!(
        cache.slots.len(),
        d,
        "gauge_after_pair_rotation: the workspace cache must have d = {d} slots"
    );
}

#[allow(clippy::too_many_arguments)]
fn mix_a<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    i: usize,
    theta: f64,
    idx: &Su2TouchIndex,
    fl: &mut [Su2Sketch<T>],
    fr: &mut [Su2Sketch<T>],
    cache: &mut Su2GaugeCache,
    inject: Inject<'_>,
) {
    check_domains(h, x, i, fl, fr, cache);
    let pg = h.space.pg;
    assert!(
        h.term_screen() == Some(Su2TermScreen::SymmetryAllowed(pg)),
        "gauge_after_pair_rotation: MixA requires a SymmetryAllowed(space.pg) skeleton (under a \
         Mask or Magnitude screen a twin label can exist through another term while the \
         relabelled one was screened out, so the mix is not exact) — use Su2GaugeMode::Recontract"
    );
    assert!(
        pg.irrep_of(i) == pg.irrep_of(i + 1),
        "gauge_after_pair_rotation: MixA refuses the cross-irrep pair ({i}, {}) of a typed space \
         (a rotation that mixes two irreps has no representation there)",
        i + 1
    );
    if theta == 0.0 {
        return;
    }
    let (s, c) = theta.sin_cos();
    chains(h, x, i, idx, fl, fr, cache, Plan::MixA { c, s, inject });
}

#[derive(Clone, Copy)]
enum Plan<'f> {
    Recontract,
    MixA { c: f64, s: f64, inject: Inject<'f> },
}

impl Plan<'_> {
    fn keep(self, idx: &Su2TouchIndex, b: usize, i: usize) -> Option<&[(Su2Sector, Vec<usize>)]> {
        match self {
            Plan::Recontract => Some(idx.touched_ref(b, i)),
            Plan::MixA { inject, .. } => inject.recontracts().then(|| idx.recontract_ref(b, i)),
        }
    }
}

struct MixBond<'a, T: Scalar> {
    b: usize,
    core: &'a Su2Core<T>,
    slot: &'a mut Option<CachedWorkspace>,
    dst: &'a mut Su2Sketch<T>,
}

fn mix_chain<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    idx: &Su2TouchIndex,
    i: usize,
    bonds: Vec<MixBond<'_, T>>,
    along: Along,
    (c, s, inject): (f64, f64, Inject<'_>),
) -> usize {
    let one = |mb: MixBond<'_, T>| {
        let mut builds = 0;
        let ws = workspace(h, mb.core, mb.slot, &mut builds);
        let tab = match along {
            Along::Cols => &ws.j_tab,
            Along::Rows => &ws.i_tab,
        };
        let pairs = inject.pairs(mb.b, idx.mix_ref(mb.b, i));
        inject.mix(mb.dst, &pairs, tab, along, (c, s));
        builds
    };
    if bonds.len() > 1 {
        bonds.into_par_iter().map(one).sum()
    } else {
        bonds.into_iter().map(one).sum()
    }
}

#[allow(clippy::too_many_arguments)]
fn chains<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    i: usize,
    idx: &Su2TouchIndex,
    fl: &mut [Su2Sketch<T>],
    fr: &mut [Su2Sketch<T>],
    cache: &mut Su2GaugeCache,
    plan: Plan<'_>,
) {
    let d = x.space.d;
    let has_pairs = |b: usize| idx.mix_ref(b, i).iter().any(|(_, ps)| !ps.is_empty());
    let (left_slots, rest) = cache.slots.split_at_mut(i);
    let right_slots = &mut rest[2..];
    let (left_builds, right_builds) = rayon::join(
        || {
            let mut builds = 0;
            if let Plan::MixA { c, s, inject } = plan
                && inject.mixes()
            {
                let bonds = fl[1..]
                    .iter_mut()
                    .zip(left_slots.iter_mut())
                    .enumerate()
                    .map(|(o, (dst, slot))| MixBond {
                        b: o + 1,
                        core: x.core(o),
                        slot,
                        dst,
                    })
                    .filter(|mb| has_pairs(mb.b))
                    .collect();
                builds += mix_chain(h, idx, i, bonds, Along::Cols, (c, s, inject));
            }
            for b in 1..=i {
                let Some(keep) = plan.keep(idx, b, i) else {
                    break;
                };
                if keeps_nothing(keep) {
                    continue;
                }
                let xk = x.core(b - 1);
                let ws = workspace(h, xk, &mut left_slots[b - 1], &mut builds);
                let (prev, cur) = fl.split_at_mut(b);
                h.recontract_left_cols(xk, &prev[b - 1], keep, ws, &mut cur[0]);
            }
            builds
        },
        || {
            let mut builds = 0;
            if let Plan::MixA { c, s, inject } = plan
                && inject.mixes()
            {
                let bonds = fr
                    .iter_mut()
                    .zip(right_slots.iter_mut())
                    .enumerate()
                    .map(|(o, (dst, slot))| MixBond {
                        b: i + 2 + o,
                        core: x.core(i + 2 + o),
                        slot,
                        dst,
                    })
                    .filter(|mb| has_pairs(mb.b))
                    .collect();
                builds += mix_chain(h, idx, i, bonds, Along::Rows, (c, s, inject));
            }
            for j in (i + 2..d).rev() {
                let Some(keep) = plan.keep(idx, j, i) else {
                    break;
                };
                if keeps_nothing(keep) {
                    continue;
                }
                let xk = x.core(j);
                let ws = workspace(h, xk, &mut right_slots[j - i - 2], &mut builds);
                let (dst, next) = fr.split_at_mut(j - i - 1);
                h.recontract_right_rows(xk, &next[0], keep, ws, &mut dst[j - i - 2]);
            }
            builds
        },
    );
    cache.builds += left_builds + right_builds;
}

#[cfg(test)]
#[path = "env_gauge_tests.rs"]
mod tests;
