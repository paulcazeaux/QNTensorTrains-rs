// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

use super::super::assembly::{Assembled, ChannelDesc, CooNest, Flux, StateShift};
use super::bitset::AdmMask;
use super::chan_key::{ChanKind, chan_key_refined, identity_key, key_flux, key_kind};
use super::packed_ops::{PackedOps, SopTerm, term_state, unpack_site};

pub(super) struct Ride {
    pub(super) term: u32,
    pub(super) chan: u32,
    pub(super) secs: Vec<u16>,
}

pub(super) type CompMap = BTreeMap<(usize, (usize, usize, u8), Flux), BTreeMap<u64, CompAgg>>;

#[derive(Default)]
pub(super) struct CompAgg {
    pub(super) terms: usize,
    pub(super) nl_up: BTreeSet<i8>,
    pub(super) nl_dn: BTreeSet<i8>,
}

pub(super) struct Sweep<'a, T: Scalar> {
    pub(super) space: QnSpace,
    pub(super) terms: &'a [SopTerm<T>],
    pub(super) per_flux: bool,
    median_switch: Vec<usize>,
    pub(super) refine: bool,
    pub(super) switch_only: bool,
    pub(super) record_src: bool,
    pub(super) adm_cache: FxHashMap<(usize, StateShift), Rc<AdmMask>>,
    pub(super) caps: Option<Vec<RankGrid>>,
    pub(super) chans: Vec<u64>,
    pub(super) rides: Vec<Ride>,
    pub(super) secs: Vec<Sector>,
    pub(super) pos: Vec<i32>,
    pub(super) pos_stride: usize,
    pub(super) states: Vec<SectorGrid<Vec<ChannelDesc>>>,
    pub(super) state_qns: Vec<SectorGrid<Vec<Sector>>>,
    pub(super) coo: Vec<CooNest<T>>,
}

impl<'a, T: Scalar> Sweep<'a, T> {
    pub(super) fn new(space: QnSpace, terms: &'a [SopTerm<T>], per_flux: bool) -> Self {
        Self::new_with_switch(space, terms, per_flux, false, false, false)
    }
    pub(super) fn new_with_switch(
        space: QnSpace,
        terms: &'a [SopTerm<T>],
        per_flux: bool,
        median_switch: bool,
        refine: bool,
        switch_only: bool,
    ) -> Self {
        let d = space.d;
        let mut states: Vec<SectorGrid<Vec<ChannelDesc>>> = (0..=d)
            .map(|_| SectorGrid::filled(&space, Vec::new()))
            .collect();
        let mut state_qns: Vec<SectorGrid<Vec<Sector>>> = (0..=d)
            .map(|_| SectorGrid::filled(&space, Vec::new()))
            .collect();
        let coo: Vec<CooNest<T>> = Vec::with_capacity(if switch_only { 0 } else { d });

        let s0 = Sector::new(0, 0);
        let pos_stride = space.bond_sectors(0).len();
        let mut chans: Vec<u64> = Vec::new();
        let mut rides: Vec<Ride> = Vec::with_capacity(terms.len());
        let pos: Vec<i32> = {
            let mut ord: BTreeMap<u64, u32> = BTreeMap::new();
            for (ti, term) in terms.iter().enumerate() {
                let key = if refine {
                    let st = term_state(&space.pg, &term.ops, 0);
                    chan_key_refined(
                        0,
                        0,
                        Irrep::IDENTITY,
                        ChanKind::Normal,
                        PackedOps::default(),
                        (st.up.nr, st.dn.nr),
                    )
                } else {
                    identity_key()
                };
                let c = *ord.entry(key).or_insert_with(|| {
                    chans.push(key);
                    (chans.len() - 1) as u32
                });
                rides.push(Ride {
                    term: ti as u32,
                    chan: c,
                    secs: vec![0],
                });
            }
            let s0_ord = 0usize;
            debug_assert_eq!(space.bond_sectors(0)[s0_ord], s0);
            let mut pos = vec![-1i32; chans.len() * pos_stride];
            for (p, (_, &c)) in ord.iter().enumerate() {
                pos[c as usize * pos_stride + s0_ord] = p as i32;
                states[0][s0].push(ChannelDesc::Bipartite(chans[c as usize]));
                state_qns[0][s0].push(s0);
            }
            pos
        };
        rides.sort_unstable_by_key(|r| (r.term, r.chan));
        let median_switch = if median_switch {
            let hd = (d as i64 - 1) / 2;
            terms
                .iter()
                .map(|t| {
                    let nbody = t.ops.cre.iter().filter(|&&b| b != 0).count();
                    let mut arr: Vec<i64> = t
                        .ops
                        .cre
                        .iter()
                        .chain(t.ops.ann.iter())
                        .filter(|&&b| b != 0)
                        .map(|&b| unpack_site(b) as i64)
                        .collect();
                    arr.push(hd);
                    arr.sort_unstable();
                    arr[nbody] as usize
                })
                .collect()
        } else {
            Vec::new()
        };
        Sweep {
            space,
            terms,
            per_flux,
            median_switch,
            refine,
            switch_only,
            record_src: false,
            adm_cache: FxHashMap::default(),
            caps: (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space)),
            chans,
            rides,
            secs: space.bond_sectors(0),
            pos,
            pos_stride,
            states,
            state_qns,
            coo,
        }
    }
    pub(super) fn uses_median_switch(&self) -> bool {
        !self.median_switch.is_empty()
    }

    pub(super) fn median_switch_of(&self, rides: &[u32]) -> usize {
        let sw = self.median_switch[self.rides[rides[0] as usize].term as usize];
        debug_assert!(
            rides
                .iter()
                .all(|&ri| self.median_switch[self.rides[ri as usize].term as usize] == sw),
            "a Normal-channel edge carried terms with different median switch bonds"
        );
        sw
    }
    pub(super) fn record_comp_stats(&self, k: usize, comp: &mut CompMap) {
        for ride in &self.rides {
            let key = self.chans[ride.chan as usize];
            if key_kind(key) != ChanKind::Comp {
                continue;
            }
            let st = term_state(&self.space.pg, &self.terms[ride.term as usize].ops, k);
            let flux = key_flux(key);
            for &so in &ride.secs {
                let s = self.secs[so as usize];
                let agg = comp
                    .entry((k, (s.nup, s.ndn, s.irrep.index()), flux))
                    .or_default()
                    .entry(key)
                    .or_default();
                agg.terms += 1;
                agg.nl_up.insert(st.up.nl);
                agg.nl_dn.insert(st.dn.nl);
            }
        }
    }
    pub(super) fn finish(self) -> Assembled<T> {
        debug_assert!(
            self.refine || {
                let distinct: std::collections::BTreeSet<_> =
                    self.chans.iter().map(|&c| key_flux(c).irrep).collect();
                self.chans.len() <= distinct.len()
            }
        );
        debug_assert!(
            self.chans.iter().all(|&c| key_kind(c) == ChanKind::Comp
                && key_flux(c).up == 0
                && key_flux(c).dn == 0)
        );
        Assembled {
            states: self.states,
            state_qns: self.state_qns,
            coo: self.coo,
            n_terms: self.terms.len(),
        }
    }
}
