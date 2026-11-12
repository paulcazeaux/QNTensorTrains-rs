// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::hamiltonian::ChannelPhase;
use crate::scalar::Scalar;
use crate::su2::qn::{Su2RankGrid, Su2Sector};

use super::{Su2SparseHamiltonian, Ts};

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn su2_channel_info(ts: &Ts, b: usize) -> (ChannelPhase, [Option<usize>; 2]) {
    let sites = ts.1.map(|raw| raw.checked_sub(1).map(usize::from));
    let phase = match sites.iter().flatten().next() {
        Some(&p) if p < b => ChannelPhase::Normal,
        Some(_) => ChannelPhase::Comp,
        None if ts.0.mask == 0 => ChannelPhase::Normal,
        None => ChannelPhase::Comp,
    };
    (phase, sites)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Touch {
    None,
    A,
    B,
}

pub(crate) fn touch(ts: &Ts, i: usize) -> Touch {
    let on =
        ts.1.iter()
            .filter(|&&r| usize::from(r) == i + 1 || usize::from(r) == i + 2)
            .count();
    match on {
        0 => Touch::None,
        1 => Touch::A,
        _ => Touch::B,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Su2TouchCensus {
    pub(crate) channels: usize,
    pub(crate) touched: usize,
    pub(crate) class_a: usize,
    pub(crate) class_b: usize,
}

#[derive(Clone, Debug, Default)]
struct PairTouch {
    lists: Vec<(Su2Sector, Vec<usize>)>,
    classes: Vec<(usize, Vec<bool>)>,
    mix: Vec<(Su2Sector, Vec<[u32; 2]>)>,
    recontract: Vec<(Su2Sector, Vec<usize>)>,
}

fn push_in<V>(list: &mut Vec<(Su2Sector, Vec<V>)>, s: Su2Sector, v: V) {
    match list.last_mut() {
        Some((t, vs)) if *t == s => vs.push(v),
        _ => list.push((s, vec![v])),
    }
}

const AMBIGUOUS: u32 = u32::MAX;

#[derive(Clone, Debug)]
struct BondTouch {
    qns: Vec<Vec<Su2Sector>>,
    pairs: Vec<PairTouch>,
}

pub(crate) struct Su2TouchIndex {
    d: usize,
    bonds: Vec<BondTouch>,
    unpaired: usize,
}

impl Su2TouchIndex {
    pub(crate) fn new<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> Self {
        assert!(
            h.is_refillable(),
            "Su2TouchIndex::new refuses a non-refillable operator (a three-body label lists only \
             part of a side) — build it with `new_refillable`"
        );
        let space = h.space;
        let (d, pg) = (space.d, space.pg);
        let npairs = d.saturating_sub(1);
        let mut unpaired = 0usize;
        let bonds = (1..d)
            .map(|b| {
                let sectors = space.bond_sectors(b);
                let mut qns = Vec::with_capacity(sectors.len());
                let mut pairs = vec![PairTouch::default(); npairs];
                for (k, &s) in sectors.iter().enumerate() {
                    let descs = h.channel_descs(b, s);
                    let m = h.state_qns_at(b, s);
                    assert_eq!(
                        descs.len(),
                        m.len(),
                        "bond {b} sector {s:?}: labels and x-sectors are not index-aligned"
                    );
                    let mut twins: Option<FxHashMap<(Ts, Su2Sector), u32>> = None;
                    for (c, ts) in descs.iter().enumerate() {
                        let mut seen = [usize::MAX; 4];
                        let mut n = 0;
                        for p in ts.1.iter().filter_map(|&r| r.checked_sub(1)) {
                            let p = usize::from(p);
                            for i in [p.wrapping_sub(1), p] {
                                if i < npairs && !seen[..n].contains(&i) {
                                    seen[n] = i;
                                    n += 1;
                                }
                            }
                        }
                        for &i in &seen[..n] {
                            let class = touch(ts, i);
                            debug_assert_ne!(class, Touch::None);
                            let pt = &mut pairs[i];
                            if pt.classes.last().is_none_or(|&(o, _)| o != k) {
                                pt.lists.push((s, Vec::new()));
                                pt.classes.push((k, Vec::new()));
                            }
                            pt.lists.last_mut().expect("pushed above").1.push(c);
                            pt.classes
                                .last_mut()
                                .expect("pushed above")
                                .1
                                .push(class == Touch::B);
                            if b == i + 1 || pg.irrep_of(i) != pg.irrep_of(i + 1) {
                                continue;
                            }
                            if class == Touch::B {
                                push_in(&mut pt.recontract, s, c);
                                continue;
                            }
                            let map = twins.get_or_insert_with(|| twin_map(descs, m));
                            match class_a_twin(map, ts, m[c], c, i) {
                                Twin::Lo(hi) => push_in(&mut pt.mix, s, [c as u32, hi]),
                                Twin::Hi => {}
                                Twin::Absent => {
                                    push_in(&mut pt.recontract, s, c);
                                    unpaired += 1;
                                }
                            }
                        }
                    }
                    qns.push(m.to_vec());
                }
                BondTouch { qns, pairs }
            })
            .collect();
        Su2TouchIndex { d, bonds, unpaired }
    }

    fn bond(&self, b: usize) -> &BondTouch {
        assert!(
            (1..self.d).contains(&b),
            "Su2TouchIndex: bond {b} is not an interior bond (1..={})",
            self.d - 1
        );
        &self.bonds[b - 1]
    }

    fn pair<'a>(&self, bond: &'a BondTouch, i: usize) -> &'a PairTouch {
        assert!(
            i + 1 < self.d,
            "Su2TouchIndex: the pair ({i}, {}) is not in a chain of {} sites",
            i + 1,
            self.d
        );
        &bond.pairs[i]
    }

    pub(crate) fn touched_ref(&self, b: usize, i: usize) -> &[(Su2Sector, Vec<usize>)] {
        &self.pair(self.bond(b), i).lists
    }

    pub(crate) fn mix_ref(&self, b: usize, i: usize) -> &[(Su2Sector, Vec<[u32; 2]>)] {
        &self.pair(self.bond(b), i).mix
    }

    pub(crate) fn recontract_ref(&self, b: usize, i: usize) -> &[(Su2Sector, Vec<usize>)] {
        &self.pair(self.bond(b), i).recontract
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn unpaired(&self) -> usize {
        self.unpaired
    }

    pub(crate) fn bytes(&self) -> usize {
        use std::mem::size_of;
        self.bonds
            .iter()
            .map(|bond| {
                let qns: usize = bond
                    .qns
                    .iter()
                    .map(|q| q.len() * size_of::<Su2Sector>())
                    .sum();
                let pairs: usize = bond
                    .pairs
                    .iter()
                    .map(|pt| {
                        let lists: usize = pt
                            .lists
                            .iter()
                            .map(|(_, cs)| size_of::<Su2Sector>() + cs.len() * size_of::<usize>())
                            .sum();
                        let classes: usize = pt
                            .classes
                            .iter()
                            .map(|(_, ks)| size_of::<usize>() + ks.len() * size_of::<bool>())
                            .sum();
                        let mix: usize = pt
                            .mix
                            .iter()
                            .map(|(_, ps)| {
                                size_of::<Su2Sector>() + ps.len() * size_of::<[u32; 2]>()
                            })
                            .sum();
                        let recontract: usize = pt
                            .recontract
                            .iter()
                            .map(|(_, cs)| size_of::<Su2Sector>() + cs.len() * size_of::<usize>())
                            .sum();
                        lists + classes + mix + recontract
                    })
                    .sum();
                qns + pairs
            })
            .sum()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn touched(&self, b: usize, i: usize) -> Vec<(Su2Sector, Vec<usize>)> {
        self.touched_ref(b, i).to_vec()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn census(&self, b: usize, i: usize, ranks: Option<&Su2RankGrid>) -> Su2TouchCensus {
        let bond = self.bond(b);
        let weight = |m: Su2Sector| ranks.map_or(1, |r| r[m]);
        let channels = bond.qns.iter().flatten().map(|&m| weight(m)).sum();
        let (mut class_a, mut class_b) = (0usize, 0usize);
        let pt = self.pair(bond, i);
        for ((_, chans), (k, kinds)) in pt.lists.iter().zip(&pt.classes) {
            for (&c, &is_b) in chans.iter().zip(kinds) {
                let w = weight(bond.qns[*k][c]);
                if is_b {
                    class_b += w;
                } else {
                    class_a += w;
                }
            }
        }
        Su2TouchCensus {
            channels,
            touched: class_a + class_b,
            class_a,
            class_b,
        }
    }
}

fn twin_map(descs: &[Ts], qns: &[Su2Sector]) -> FxHashMap<(Ts, Su2Sector), u32> {
    assert!(
        u32::try_from(descs.len()).is_ok_and(|n| n < AMBIGUOUS),
        "Su2TouchIndex: {} channels in one sector overflow the mixing plan's u32 indices",
        descs.len()
    );
    let mut map = FxHashMap::with_capacity_and_hasher(descs.len(), Default::default());
    for (c, (&ts, &m)) in descs.iter().zip(qns).enumerate() {
        map.entry((ts, m))
            .and_modify(|v| *v = AMBIGUOUS)
            .or_insert(c as u32);
    }
    map
}

enum Twin {
    Lo(u32),
    Hi,
    Absent,
}

fn class_a_twin(
    map: &FxHashMap<(Ts, Su2Sector), u32>,
    ts: &Ts,
    m: Su2Sector,
    c: usize,
    i: usize,
) -> Twin {
    let (lo, hi) = (i + 1, i + 2);
    let slot =
        ts.1.iter()
            .position(|&r| usize::from(r) == lo || usize::from(r) == hi)
            .expect("a class-A label has one slot on the pair");
    let on_lo = usize::from(ts.1[slot]) == lo;
    let mut twin = *ts;
    twin.1[slot] = if on_lo { hi } else { lo } as u16;
    let own = map.get(&(*ts, m)).copied();
    debug_assert!(own.is_some(), "a channel's own key is in its sector's map");
    match (own, map.get(&(twin, m)).copied()) {
        (Some(o), Some(t)) if o as usize == c && t != AMBIGUOUS => {
            if on_lo {
                Twin::Lo(t)
            } else {
                Twin::Hi
            }
        }
        _ => Twin::Absent,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn su2_env_gauge_closed<T: Scalar>(h: &Su2SparseHamiltonian<T>) -> Result<(), String> {
    assert!(
        h.is_refillable(),
        "su2_env_gauge_closed refuses a non-refillable operator (a three-body label lists only \
         part of a side) — build it with `new_refillable`"
    );
    let space = h.space;
    let d = space.d;
    let pg = space.pg;
    for b in 1..d {
        for s in space.bond_sectors(b) {
            let descs = h.channel_descs(b, s);
            if descs.is_empty() {
                continue;
            }
            let qns = h.state_qns_at(b, s);
            assert_eq!(
                descs.len(),
                qns.len(),
                "bond {b} sector {s:?}: labels and x-sectors are not index-aligned"
            );
            let lookup: HashMap<(Ts, Su2Sector), usize> = descs
                .iter()
                .zip(qns)
                .enumerate()
                .map(|(c, (&ts, &m))| ((ts, m), c))
                .collect();
            for (c, (ts, &m)) in descs.iter().zip(qns).enumerate() {
                for (slot, &raw) in ts.1.iter().enumerate() {
                    let Some(p) = raw.checked_sub(1).map(usize::from) else {
                        continue;
                    };
                    let pairs = [Some((p, p + 1)), p.checked_sub(1).map(|q| (q, q))];
                    for (i, to) in pairs.into_iter().flatten() {
                        if i + 1 >= d
                            || !(b <= i || b >= i + 2)
                            || pg.irrep_of(i) != pg.irrep_of(i + 1)
                            || touch(ts, i) != Touch::A
                        {
                            continue;
                        }
                        let mut twin = *ts;
                        twin.1[slot] = (to + 1) as u16;
                        if !lookup.contains_key(&(twin, m)) {
                            return Err(format!(
                                "bond {b} sector {s:?}: class-A channel {c} {:?} open {:?} (m = \
                                 {m:?}) relabelled on pair ({i}, {}) to open {:?} has no channel",
                                ts.0,
                                su2_channel_info(ts, b).1,
                                i + 1,
                                su2_channel_info(&twin, b).1
                            ));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "channel_tests.rs"]
pub(super) mod tests;
