// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;

use crate::qn::{QnSpace, SectorGrid};
use crate::scalar::Scalar;

use super::assembly::{Assembled, empty_nest};
use super::construct_probe as probe;

mod bitset;
mod chan_key;
mod matching;
mod packed_ops;
mod stats;
mod sweep;
mod sweep_site;

pub(crate) use chan_key::{ChanKind, key_flux, key_kind, key_ops, key_with_ops};
pub(crate) use packed_ops::{PackedOps, SopTerm, sort_terms};
pub use stats::{BlockStat, CompChannelStat, TermSwitch, bipartite_probe, switch_profile};
use sweep::Sweep;

pub(crate) fn assemble_bipartite<T: Scalar>(
    space: QnSpace,
    mut terms: Vec<SopTerm<T>>,
    record_src: bool,
) -> Assembled<T> {
    assemble_bipartite_impl(space, &mut terms, true, false, record_src)
}

pub(crate) fn cover_switch_bonds<T: Scalar>(
    space: QnSpace,
    terms: &mut [SopTerm<T>],
    refine: bool,
) -> FxHashMap<PackedOps, usize> {
    let mut out: FxHashMap<PackedOps, usize> = FxHashMap::default();
    sort_terms(terms);
    if terms.is_empty() {
        return out;
    }
    let mut sweep = Sweep::new_with_switch(space, terms, true, false, refine, true);
    for k in 0..space.d {
        sweep.site(k);
        let _t = probe::timer(probe::Stage::SwitchScan);
        for ride in &sweep.rides {
            if key_kind(sweep.chans[ride.chan as usize]) == ChanKind::Comp {
                out.entry(terms[ride.term as usize].ops).or_insert(k);
            }
        }
    }
    out
}

pub(crate) fn assemble_bipartite_persector<T: Scalar>(
    space: QnSpace,
    mut terms: Vec<SopTerm<T>>,
    record_src: bool,
) -> Assembled<T> {
    assemble_bipartite_impl(space, &mut terms, false, false, record_src)
}

pub(crate) fn assemble_bipartite_median_switch<T: Scalar>(
    space: QnSpace,
    mut terms: Vec<SopTerm<T>>,
    record_src: bool,
) -> Assembled<T> {
    assemble_bipartite_impl(space, &mut terms, true, true, record_src)
}

fn assemble_bipartite_impl<T: Scalar>(
    space: QnSpace,
    terms: &mut [SopTerm<T>],
    per_flux: bool,
    median_switch: bool,
    record_src: bool,
) -> Assembled<T> {
    sort_terms(terms);
    if terms.is_empty() {
        return Assembled {
            states: (0..=space.d)
                .map(|_| SectorGrid::filled(&space, Vec::new()))
                .collect(),
            state_qns: (0..=space.d)
                .map(|_| SectorGrid::filled(&space, Vec::new()))
                .collect(),
            coo: (0..space.d).map(|k| empty_nest(&space, k)).collect(),
            n_terms: 0,
        };
    }
    let mut sweep = Sweep::new_with_switch(space, terms, per_flux, median_switch, false, false);
    sweep.record_src = record_src;
    for k in 0..space.d {
        sweep.site(k);
    }
    sweep.finish()
}
