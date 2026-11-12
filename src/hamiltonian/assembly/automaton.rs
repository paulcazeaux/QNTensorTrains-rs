// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::rc::Rc;

use crate::core::Family;
use crate::elementaryops::shift_qn;
use crate::orbital::{Orbital, Spin};
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, Sector, SectorGrid};
use crate::scalar::Scalar;

use super::super::construct_probe as probe;
use super::enumerate::TermSink;
use super::kernels::{
    NestBuilder, jordanwigner_sign, jw_apply, lr_sectors, occupation, op_delta_cands,
    op_irrep_flux, push_coo_cell, src_tag,
};
use super::types::{
    Assembled, ChannelDesc, EdgeData, Flux, Label, OpenIdx, SRC_NEG, SRC_NONE, SpinShift,
    StateShift,
};

pub(super) struct Automaton<T: Scalar> {
    pub(super) space: QnSpace,
    pub(super) hd: i64,
    pub(super) vlabels: Vec<Label>,
    pub(super) fams: Vec<VertexFamily>,
    pub(super) fam_of: FxHashMap<u128, u32>,
    pub(super) n_slots: usize,
    pub(super) caps: Option<Vec<crate::qn::RankGrid>>,
    pub(super) edge_keys: Vec<u64>,
    pub(super) edge_data: Vec<EdgeData<T>>,
    pub(super) edge_site: Vec<u8>,
    pub(super) fam_edge_seen: FxHashMap<(u32, u32), T>,
    pub(super) switches: Option<FxHashMap<super::super::bipartite::PackedOps, usize>>,
    pub(super) early_break: bool,
    pub(super) qn_cache: FxHashMap<(usize, StateShift), Rc<Vec<Sector>>>,
    pub(super) cur_term: u32,
    pub(super) record_src: bool,
}

pub(super) struct VertexFamily {
    bond: usize,
    state: StateShift,
    open: OpenIdx,
    ids: Vec<i32>,
}

fn pack_orb(o: Option<Orbital>) -> u64 {
    match o {
        None => 0,
        Some(orb) => {
            let spin = if orb.spin == Spin::Up { 0 } else { 1 };
            ((orb.site as u64 + 1) << 1) | spin
        }
    }
}

fn pack_family(bond: usize, state: &StateShift, open: &OpenIdx) -> u128 {
    debug_assert!(
        state.delta.index() < 8,
        "irrep index exceeds pack_family's 3-bit Δ field"
    );
    let mut key = bond as u128;
    key = (key << 3) | (state.up.flux + 3) as u128;
    key = (key << 2) | state.up.nl as u128;
    key = (key << 2) | state.up.nr as u128;
    key = (key << 3) | (state.dn.flux + 3) as u128;
    key = (key << 2) | state.dn.nl as u128;
    key = (key << 2) | state.dn.nr as u128;
    key = (key << 3) | state.delta.index() as u128;
    for &o in open {
        key = (key << 8) | pack_orb(o) as u128;
    }
    key
}

impl<T: Scalar> Automaton<T> {
    fn shifted_qn(&mut self, bond: usize, s: StateShift) -> Rc<Vec<Sector>> {
        if let Some(v) = self.qn_cache.get(&(bond, s)) {
            return Rc::clone(v);
        }
        let v = Rc::new(shift_qn(
            &self.space,
            self.caps.as_deref(),
            bond,
            s.up.flux as i64,
            s.up.nl as i64,
            s.up.nr as i64,
            s.dn.flux as i64,
            s.dn.nl as i64,
            s.dn.nr as i64,
            s.delta,
        ));
        self.qn_cache.insert((bond, s), Rc::clone(&v));
        v
    }

    fn family(&mut self, bond: usize, state: StateShift, open: OpenIdx) -> u32 {
        let key = pack_family(bond, &state, &open);
        if let Some(&f) = self.fam_of.get(&key) {
            return f;
        }
        let f = self.fams.len() as u32;
        self.fams.push(VertexFamily {
            bond,
            state,
            open,
            ids: vec![-1; self.n_slots],
        });
        self.fam_of.insert(key, f);
        f
    }

    #[inline]
    fn vid_of(&mut self, fam: u32, sector: Sector) -> (usize, bool) {
        let plane = (self.space.nup + 1) * (self.space.ndn + 1);
        let slot = (sector.irrep.index() as usize) * plane
            + sector.nup * (self.space.ndn + 1)
            + sector.ndn;
        let cur = self.fams[fam as usize].ids[slot];
        if cur >= 0 {
            return (cur as usize, false);
        }
        let id = self.vlabels.len();
        let (bond, state, open) = {
            let f = &self.fams[fam as usize];
            (f.bond, f.state, f.open)
        };
        self.vlabels.push(Label {
            bond,
            sector,
            state,
            open,
        });
        self.fams[fam as usize].ids[slot] = id as i32;
        (id, true)
    }

    fn emit(
        &mut self,
        fam_m: u32,
        n_minus: Sector,
        fam_p: u32,
        n_plus: Sector,
        data: EdgeData<T>,
    ) -> bool {
        let (im, new_m) = self.vid_of(fam_m, n_minus);
        let (ip, new_p) = self.vid_of(fam_p, n_plus);
        assert!(
            im <= u32::MAX as usize && ip <= u32::MAX as usize,
            "vertex id overflows the 32-bit edge-key field"
        );
        assert!(
            self.edge_keys.len() < u32::MAX as usize,
            "edge count overflows the 32-bit edge index"
        );
        let ekey = ((im as u64) << 32) | ip as u64;
        self.edge_keys.push(ekey);
        self.edge_data.push(data);
        self.edge_site.push(self.fams[fam_m as usize].bond as u8);
        new_m || new_p
    }

    fn family_pair_seen(&mut self, fam_m: u32, fam_p: u32, w: T) -> bool {
        match self.fam_edge_seen.entry((fam_m, fam_p)) {
            std::collections::hash_map::Entry::Occupied(e) => {
                debug_assert!(
                    (*e.get() - w).abs_s() <= 1e-12 * (1.0 + e.get().abs_s()),
                    "family pair ({fam_m}, {fam_p}) re-emitted with a different coefficient: \
                     {} vs {}",
                    e.get().abs_s(),
                    w.abs_s()
                );
                true
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(w);
                false
            }
        }
    }
}

pub(super) fn vertex(
    pg: &crate::pointgroup::PointGroup,
    bond: usize,
    switch: usize,
    cre: &[Orbital],
    ann: &[Orbital],
) -> (StateShift, OpenIdx) {
    let b = bond as i64;
    let (mut fu, mut nlu, mut nru) = (0i64, 0i64, 0i64);
    let (mut fd, mut nld, mut nrd) = (0i64, 0i64, 0i64);
    let mut delta = Irrep::IDENTITY;
    for o in cre {
        let applied = (o.site as i64) < b;
        if applied {
            delta = delta.fuse(pg.irrep_of(o.site));
        }
        if o.spin == Spin::Up {
            if applied {
                fu += 1;
                nlu += 1;
            } else {
                nru += 1;
            }
        } else if applied {
            fd += 1;
            nld += 1;
        } else {
            nrd += 1;
        }
    }
    for o in ann {
        if (o.site as i64) < b {
            delta = delta.fuse(pg.irrep_of(o.site));
            if o.spin == Spin::Up {
                fu -= 1;
            } else {
                fd -= 1;
            }
        }
    }
    let state = StateShift {
        up: SpinShift {
            flux: fu as i8,
            nl: nlu as i8,
            nr: nru as i8,
        },
        dn: SpinShift {
            flux: fd as i8,
            nl: nld as i8,
            nr: nrd as i8,
        },
        delta,
    };

    let keep_applied = bond <= switch;
    let mut open: OpenIdx = [None; 6];
    let mut n = 0usize;
    for o in cre.iter().chain(ann) {
        if ((o.site as i64) < b) == keep_applied {
            open[n] = Some(*o);
            n += 1;
        }
    }
    debug_assert_eq!(
        delta.fuse(
            open.iter()
                .flatten()
                .fold(Irrep::IDENTITY, |a, o| a.fuse(pg.irrep_of(o.site)))
        ),
        if keep_applied {
            Irrep::IDENTITY
        } else {
            cre.iter()
                .chain(ann)
                .fold(Irrep::IDENTITY, |a, o| a.fuse(pg.irrep_of(o.site)))
        },
        "vertex Δ disagrees with the irrep flux derivable from its open-orbital list"
    );
    (state, open)
}

pub(crate) fn weighted_edge(hd: i64, cre: &[Orbital], ann: &[Orbital]) -> usize {
    assert!(cre.len() <= 3 && ann.len() <= 3, "operator string too long");
    let nhalf = cre.len();
    let mut buf = [0i64; 7];
    buf[0] = hd;
    let mut n = 1;
    for o in cre.iter().chain(ann) {
        buf[n] = o.site as i64;
        n += 1;
    }
    let arr = &mut buf[..n];
    arr.sort_unstable();
    arr[nhalf] as usize
}

fn spin_op(site: usize, spin: Spin, cre: &[Orbital], ann: &[Orbital]) -> super::types::Op {
    use super::types::Op;
    let in_cre = cre.iter().any(|o| o.site == site && o.spin == spin);
    let in_ann = ann.iter().any(|o| o.site == site && o.spin == spin);
    if in_cre && in_ann {
        Op::N
    } else if in_cre {
        Op::Adag
    } else if in_ann {
        Op::A
    } else {
        Op::Id
    }
}

fn set_edges<T: Scalar>(
    auto: &mut Automaton<T>,
    bond: usize,
    cre: &[Orbital],
    ann: &[Orbital],
    w: T,
    switch: usize,
) -> bool {
    probe::bump(probe::Counter::SetEdgesCalls, 1);
    let (s_minus, idx_minus) = vertex(&auto.space.pg, bond, switch, cre, ann);
    let (s_plus, idx_plus) = vertex(&auto.space.pg, bond + 1, switch, cre, ann);

    let op_up = spin_op(bond, Spin::Up, cre, ann);
    let op_dn = spin_op(bond, Spin::Dn, cre, ann);

    let fam_m = auto.family(bond, s_minus, idx_minus);
    let fam_p = auto.family(bond + 1, s_plus, idx_plus);
    if auto.family_pair_seen(fam_m, fam_p, w) {
        return false;
    }
    let qn_minus = auto.shifted_qn(bond, s_minus);
    let qn_plus = auto.shifted_qn(bond + 1, s_plus);

    let mut cont = false;
    for &n_minus in qn_minus.iter() {
        let up_cands = op_delta_cands(op_up);
        let dn_cands = op_delta_cands(op_dn);
        for &du in up_cands {
            for &dd in dn_cands {
                let y_occ = Family::from_occupation(du == 1, dd == 1);
                let n_plus = y_occ.apply(n_minus, auto.space.pg.irrep_of(bond));
                if !qn_plus.contains(&n_plus) {
                    continue;
                }
                let (l, r) = lr_sectors(
                    n_minus,
                    n_plus,
                    Flux {
                        up: s_minus.up.flux,
                        dn: s_minus.dn.flux,
                        irrep: s_minus.delta,
                    },
                    Flux {
                        up: s_plus.up.flux,
                        dn: s_plus.dn.flux,
                        irrep: s_plus.delta,
                    },
                );
                debug_assert_eq!(
                    occupation(l, r).flux(auto.space.pg.irrep_of(bond)),
                    y_occ.flux(auto.space.pg.irrep_of(bond)).fuse(op_irrep_flux(
                        op_up,
                        op_dn,
                        auto.space.pg.irrep_of(bond)
                    )),
                    "operator/state irrep bookkeeping disagree at site {bond}"
                );
                let (occ, neg) = jordanwigner_sign(l, r, (op_up, op_dn));
                let alpha = jw_apply(w, neg);
                probe::bump(probe::Counter::Emits, 1);
                let src = if bond == switch {
                    probe::bump(probe::Counter::EmitEntriesWeighted, 1);
                    probe::bump(probe::Counter::CooTriplesWeighted, 1);
                    src_tag(auto.cur_term, neg)
                } else {
                    SRC_NONE
                };
                let new_v = auto.emit(
                    fam_m,
                    n_minus,
                    fam_p,
                    n_plus,
                    EdgeData {
                        m: l,
                        x_occ: occ,
                        alpha,
                        src,
                    },
                );
                cont = new_v || cont;
            }
        }
    }
    cont
}

fn run_term<T: Scalar>(
    auto: &mut Automaton<T>,
    cre: &[Orbital],
    ann: &[Orbital],
    w: T,
    switch: usize,
    early_break: bool,
) {
    let d = auto.space.d;
    let bm = switch;
    set_edges(auto, bm, cre, ann, w, switch);
    let mut b = bm + 1;
    while b < d {
        if !set_edges(auto, b, cre, ann, T::one(), switch) && early_break {
            break;
        }
        b += 1;
    }
    let mut b = bm;
    while b > 0 {
        b -= 1;
        if !set_edges(auto, b, cre, ann, T::one(), switch) && early_break {
            break;
        }
    }
}

impl<T: Scalar> TermSink<T> for Automaton<T> {
    fn term(&mut self, cre: &[Orbital], ann: &[Orbital], w: T) {
        super::enumerate::assert_term_symmetric(&self.space, cre, ann);
        let _t = probe::timer(probe::Stage::SetEdges);
        probe::bump(probe::Counter::Terms, 1);
        match &self.switches {
            None => {
                let bm = weighted_edge(self.hd, cre, ann);
                let eb = self.early_break;
                run_term(self, cre, ann, w, bm, eb);
            }
            Some(map) => {
                let key = super::super::bipartite::PackedOps::new(cre, ann);
                let bm = map
                    .get(&key)
                    .copied()
                    .unwrap_or_else(|| weighted_edge(self.hd, cre, ann));
                let eb = self.early_break;
                run_term(self, cre, ann, w, bm, eb);
            }
        }
        self.cur_term += 1;
        assert!(
            self.cur_term < SRC_NEG,
            "term count overflows the 31-bit provenance ordinal"
        );
    }
}

pub(super) fn finalize<T: Scalar>(auto: Automaton<T>) -> Assembled<T> {
    let space = auto.space;
    let d = space.d;

    let all_sectors: Vec<Sector> = (0..space.pg.order())
        .flat_map(|ir| {
            (0..=space.nup).flat_map(move |nu| {
                (0..=space.ndn).map(move |nd| Sector::with_irrep(nu, nd, Irrep::new(ir)))
            })
        })
        .collect();

    let mut buckets: Vec<SectorGrid<Vec<usize>>> = (0..=d)
        .map(|_| SectorGrid::filled(&space, Vec::<usize>::new()))
        .collect();
    for (id, lab) in auto.vlabels.iter().enumerate() {
        buckets[lab.bond][lab.sector].push(id);
    }

    let mut vid_to_i = vec![0usize; auto.vlabels.len()];
    let mut states: Vec<SectorGrid<Vec<ChannelDesc>>> = (0..=d)
        .map(|_| SectorGrid::filled(&space, Vec::new()))
        .collect();
    let mut state_qns: Vec<SectorGrid<Vec<Sector>>> = (0..=d)
        .map(|_| SectorGrid::filled(&space, Vec::new()))
        .collect();

    let _t_sort = probe::timer(probe::Stage::FinalizeSort);
    for b in 0..=d {
        for &s in &all_sectors {
            buckets[b][s].sort_by(|&a, &c| {
                (auto.vlabels[a].state, auto.vlabels[a].open)
                    .cmp(&(auto.vlabels[c].state, auto.vlabels[c].open))
            });
            let bucket = &buckets[b][s];
            for (i, &id) in bucket.iter().enumerate() {
                vid_to_i[id] = i;
            }
            let st_list: Vec<ChannelDesc> = bucket
                .iter()
                .map(|&id| ChannelDesc::Median(auto.vlabels[id].state, auto.vlabels[id].open))
                .collect();
            let sq_list: Vec<Sector> = bucket
                .iter()
                .map(|&id| {
                    let st = auto.vlabels[id].state;
                    Sector::with_irrep(
                        (s.nup as i64 - st.up.flux as i64) as usize,
                        (s.ndn as i64 - st.dn.flux as i64) as usize,
                        s.irrep.fuse(st.delta.inverse()),
                    )
                })
                .collect();
            states[b][s] = st_list;
            state_qns[b][s] = sq_list;
        }
    }

    for &s in &all_sectors {
        for &id in &buckets[d][s] {
            let dl = auto.vlabels[id].state.delta;
            assert_eq!(
                dl,
                Irrep::IDENTITY,
                "bond-{d} automaton vertex carries total irrep flux Δ = {dl:?} ≠ IDENTITY: the \
                 assembled Hamiltonian is not totally symmetric. A symmetry-forbidden integral \
                 survived the ϵ screen — screen the integrals first \
                 (`fcidump::screen_point_group_forbidden`, on by default on the loader paths) or \
                 raise `eps` above the dump's forbidden-integral noise floor (~1e-15 for a \
                 produced FCIDUMP)."
            );
        }
    }

    for &id in &buckets[0][Sector::new(0, 0)] {
        vid_to_i[id] = 0;
    }
    for &id in &buckets[d][Sector::with_irrep(space.nup, space.ndn, space.target)] {
        vid_to_i[id] = 0;
    }

    assert!(
        vid_to_i.iter().all(|&i| i <= i32::MAX as usize),
        "flattened state index overflows Int32 COO storage"
    );

    drop(_t_sort);
    let _t_coo = probe::timer(probe::Stage::FinalizeCoo);
    let mut ekeys: Vec<(u8, u64, u32)> = auto
        .edge_keys
        .iter()
        .zip(&auto.edge_site)
        .enumerate()
        .map(|(i, (&k, &b))| (b, k, i as u32))
        .collect();
    {
        let _t = probe::timer(probe::Stage::EdgeKeySort);
        ekeys.par_sort_unstable_by_key(|&(b, k, _)| (b, k));
    }
    let mut site_start = vec![0usize; d + 2];
    for &(b, _, _) in &ekeys {
        site_start[(b as usize).min(d) + 1] += 1;
    }
    for i in 0..=d {
        site_start[i + 1] += site_start[i];
    }
    let (vlabels, edge_data, vid_to_i_ref) = (&auto.vlabels, &auto.edge_data, &vid_to_i);
    let coo: Vec<super::types::CooNest<T>> = (0..d)
        .into_par_iter()
        .map(|site| {
            let mut nest = NestBuilder::new_with_src(&space, site, auto.record_src);
            for &(_, ekey, slot) in &ekeys[site_start[site]..site_start[site + 1]] {
                let data = &edge_data[slot as usize];
                let id_minus = (ekey >> 32) as usize;
                let id_plus = (ekey & 0xffff_ffff) as usize;
                let n1 = vlabels[id_minus].sector;
                let n2 = vlabels[id_plus].sector;
                let row = vid_to_i_ref[id_minus] as i32;
                let col = vid_to_i_ref[id_plus] as i32;
                push_coo_cell(
                    &mut nest,
                    n1,
                    n2,
                    data.x_occ,
                    data.m,
                    (row, col),
                    data.alpha,
                    data.src,
                );
            }
            nest.freeze()
        })
        .collect();

    Assembled {
        states,
        state_qns,
        coo,
        n_terms: auto.cur_term as usize,
    }
}
