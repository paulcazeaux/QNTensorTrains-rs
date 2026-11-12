// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::core::Family;
use crate::qn::Sector;
use crate::scalar::Scalar;

use super::super::assembly::{
    ChannelDesc, Flux, NestBuilder, Op, SRC_NONE, jordanwigner_sign, jw_apply, lr_sectors,
    op_delta_cands, op_irrep_flux, push_coo_cell, src_tag,
};
use super::super::construct_probe as probe;
use super::bitset::{
    AdmMask, adm_mask, bit_slot, bits_for_each, bits_or, bits_set, bits_shift_and, plane_words,
};
use super::chan_key::{
    ChanKind, chan_key, chan_key_refined, check_edge_collision, identity_key, key_flux, key_kind,
    key_ops,
};
use super::matching::{hopcroft_karp, koenig_cover};
use super::packed_ops::{PackedOps, Z_ID, extend_applied, site_ops, term_state, z_pack, z_unpack};
use super::sweep::{Ride, Sweep};

type EmitKey = (u16, u32, u32, u8);
type EmitVal<T> = (T, u8, u32);
type EmitList<T> = Vec<(EmitKey, EmitVal<T>)>;

type BlockMap = BTreeMap<(u16, Flux), Vec<(u32, u8)>>;

impl<'a, T: Scalar> Sweep<'a, T> {
    pub(super) fn site(&mut self, k: usize) {
        let space = self.space;
        let d = space.d;
        let per_flux = self.per_flux;
        let secs1 = space.bond_sectors(k + 1);
        let nsec1 = secs1.len();
        let pw = plane_words(&space);
        let nwords = space.pg.order() as usize * pw;
        let mut ord1 = vec![u16::MAX; nwords * 64];
        for (i, s) in secs1.iter().enumerate() {
            ord1[bit_slot(&space, *s)] = i as u16;
        }
        let mut ord0 = vec![u16::MAX; nwords * 64];
        for (i, s) in self.secs.iter().enumerate() {
            ord0[bit_slot(&space, *s)] = i as u16;
        }
        let sec_ord1 = |s: Sector| -> u16 {
            if s.nup > space.nup || s.ndn > space.ndn {
                u16::MAX
            } else {
                ord1[bit_slot(&space, s)]
            }
        };

        let mut src_mask = [vec![!0u64; pw], vec![0u64; pw]];
        for nu in 0..=space.nup {
            for nd in 0..space.ndn {
                bits_set(&mut src_mask[1], nu * (space.ndn + 1) + nd);
            }
        }
        let chan_bits: Vec<u64> = if per_flux {
            let mut cb = vec![0u64; self.chans.len() * nwords];
            for c in 0..self.chans.len() {
                let base = c * nwords;
                for (so, &p) in self.pos[c * self.pos_stride..(c + 1) * self.pos_stride]
                    .iter()
                    .enumerate()
                {
                    if p >= 0 {
                        let s = self.secs[so];
                        bits_set(&mut cb[base..base + nwords], bit_slot(&space, s));
                    }
                }
            }
            cb
        } else {
            Vec::new()
        };

        struct Edge<T> {
            chan: u32,
            zu: Op,
            zdn: Op,
            z: u8,
            rem: PackedOps,
            flux_out: Flux,
            refine_normal: (i8, i8),
            refine_comp: (i8, i8),
            val: T,
            rides: Vec<u32>,
        }
        let chans_in = &self.chans;
        let rides_in = &self.rides;
        let deposit_src = |e: &Edge<T>| -> u32 {
            if key_kind(chans_in[e.chan as usize]) == ChanKind::Normal {
                assert!(
                    e.rides.len() == 1,
                    "γ-deposit edge at site {k} pools {} terms — the single-source provenance \
                     invariant is violated",
                    e.rides.len()
                );
                rides_in[e.rides[0] as usize].term
            } else {
                SRC_NONE
            }
        };
        let _t_graph = probe::timer(probe::Stage::GraphBuild);
        probe::bump(probe::Counter::Rides, self.rides.len() as u64);
        let mut edges: Vec<Edge<T>> = Vec::new();
        let mut edge_ord: FxHashMap<(u32, u8, PackedOps), u32> = FxHashMap::default();
        let mut ride_edge: Vec<u32> = Vec::with_capacity(self.rides.len());
        let mut ride_adm: Vec<Rc<AdmMask>> = Vec::with_capacity(self.rides.len());
        for (ri, ride) in self.rides.iter().enumerate() {
            let term = &self.terms[ride.term as usize];
            let (zu, zdn) = site_ops(&term.ops, k);
            let z = z_pack(zu, zdn);
            let rem = term.ops.split_remainder(k + 1);
            let ckey = self.chans[ride.chan as usize];
            let val = if key_kind(ckey) == ChanKind::Normal {
                term.gamma
            } else {
                T::one()
            };
            debug_assert_eq!(
                key_ops(ckey),
                match key_kind(ckey) {
                    ChanKind::Normal => term.ops.split_applied(k),
                    ChanKind::Comp => term.ops.split_remainder(k),
                },
                "channel opstring diverged from the riding term at site {k}"
            );
            debug_assert_eq!(
                key_flux(ckey),
                {
                    let st = term_state(&space.pg, &term.ops, k);
                    Flux {
                        up: st.up.flux,
                        dn: st.dn.flux,
                        irrep: st.delta,
                    }
                },
                "channel flux diverged from the riding term at site {k}"
            );
            let st1 = term_state(&space.pg, &term.ops, k + 1);
            ride_adm.push(adm_mask(
                &mut self.adm_cache,
                &space,
                self.caps.as_deref(),
                k + 1,
                st1,
            ));
            let flux_out = Flux {
                up: st1.up.flux,
                dn: st1.dn.flux,
                irrep: st1.delta,
            };
            let (refine_normal, refine_comp) = if self.refine {
                ((st1.up.nr, st1.dn.nr), (st1.up.nl, st1.dn.nl))
            } else {
                ((0, 0), (0, 0))
            };
            match edge_ord.entry((ride.chan, z, rem)) {
                std::collections::hash_map::Entry::Occupied(e) => {
                    let ei = *e.get();
                    check_edge_collision(edges[ei as usize].val, val);
                    debug_assert!(
                        edges[ei as usize].refine_normal == refine_normal
                            && edges[ei as usize].refine_comp == refine_comp,
                        "median-identity refinement not single-valued on an edge at site {k}"
                    );
                    edges[ei as usize].rides.push(ri as u32);
                    ride_edge.push(ei);
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    let ei = edges.len() as u32;
                    slot.insert(ei);
                    edges.push(Edge {
                        chan: ride.chan,
                        zu,
                        zdn,
                        z,
                        rem,
                        flux_out,
                        refine_normal,
                        refine_comp,
                        val,
                        rides: vec![ri as u32],
                    });
                    ride_edge.push(ei);
                }
            }
        }

        drop(_t_graph);
        probe::bump(probe::Counter::GraphEdges, edges.len() as u64);
        let _t_block = probe::timer(probe::Stage::Blocking);
        let mut blocks: BlockMap = BlockMap::new();
        let mut edge_adm: Vec<u64> = if per_flux {
            vec![0u64; edges.len() * nwords]
        } else {
            Vec::new()
        };
        let mut edge_live: Vec<bool> = vec![false; if per_flux { edges.len() } else { 0 }];
        let mut mask_by_sec = vec![0u8; nsec1];
        let mut touched: Vec<u16> = Vec::new();
        let mut scratch = vec![0u64; nwords];
        for (ei, e) in edges.iter().enumerate() {
            if per_flux {
                let (lo, hi) = (ei * nwords, (ei + 1) * nwords);
                for &ri in &e.rides {
                    let src = &ride_adm[ri as usize].bits;
                    bits_or(&mut edge_adm[lo..hi], src);
                }
                let mut live = false;
                for &du in op_delta_cands(e.zu) {
                    for &dd in op_delta_cands(e.zdn) {
                        let dg = Family::from_occupation(du == 1, dd == 1)
                            .flux(space.pg.irrep_of(k))
                            .index();
                        live |= bits_shift_and(
                            &mut scratch,
                            &chan_bits[e.chan as usize * nwords..(e.chan as usize + 1) * nwords],
                            &src_mask[dd],
                            du * (space.ndn + 1) + dd,
                            &edge_adm[lo..hi],
                            pw,
                            dg,
                        );
                    }
                }
                edge_live[ei] = live;
                continue;
            }
            touched.clear();
            for &ri in &e.rides {
                let adm = &ride_adm[ri as usize].ord;
                for &so in &self.rides[ri as usize].secs {
                    let s = self.secs[so as usize];
                    for &du in op_delta_cands(e.zu) {
                        for &dd in op_delta_cands(e.zdn) {
                            let y_occ = Family::from_occupation(du == 1, dd == 1);
                            let o = sec_ord1(y_occ.apply(s, space.pg.irrep_of(k)));
                            if o == u16::MAX || !adm[o as usize] {
                                continue;
                            }
                            if mask_by_sec[o as usize] == 0 {
                                touched.push(o);
                            }
                            mask_by_sec[o as usize] |= 1 << (du * 2 + dd);
                        }
                    }
                }
            }
            touched.sort_unstable();
            for &o in &touched {
                blocks
                    .entry((o, e.flux_out))
                    .or_default()
                    .push((ei as u32, mask_by_sec[o as usize]));
                mask_by_sec[o as usize] = 0;
            }
        }

        drop(_t_block);
        let mut chans1: Vec<u64> = Vec::new();
        let mut chan_ord1: FxHashMap<u64, u32> = FxHashMap::default();
        let intern1 = |key: u64, chans1: &mut Vec<u64>, m: &mut FxHashMap<u64, u32>| -> u32 {
            *m.entry(key).or_insert_with(|| {
                chans1.push(key);
                (chans1.len() - 1) as u32
            })
        };
        let mut inst_pairs: Vec<(u16, u32)> = Vec::new();
        let mut route_map: FxHashMap<(u32, u16), u32> = FxHashMap::default();
        let mut emits: EmitList<T> = EmitList::new();
        let mut edge_target: FxHashMap<u32, u32> = FxHashMap::default();
        if self.per_flux {
            let mut flux_edges: BTreeMap<Flux, Vec<u32>> = BTreeMap::new();
            for (ei, e) in edges.iter().enumerate() {
                if edge_live[ei] {
                    flux_edges.entry(e.flux_out).or_default().push(ei as u32);
                }
            }
            let mut per_sec = vec![0u8; nsec1];
            let mut touch: Vec<u16> = Vec::new();
            for (&flux, fedges) in &flux_edges {
                let _t_prep = probe::timer(probe::Stage::CoverPrep);
                debug_assert!(
                    fedges.iter().all(|&ei| edges[ei as usize].flux_out == flux),
                    "per-flux cover block mixed fluxes at site {k}"
                );
                let mut ukeys: Vec<(u64, u8)> = fedges
                    .iter()
                    .map(|&ei| {
                        let e = &edges[ei as usize];
                        (self.chans[e.chan as usize], e.z)
                    })
                    .collect();
                ukeys.sort_unstable();
                ukeys.dedup();
                let mut vkeys: Vec<(PackedOps, (i8, i8))> = fedges
                    .iter()
                    .map(|&ei| {
                        let e = &edges[ei as usize];
                        (e.rem, e.refine_comp)
                    })
                    .collect();
                vkeys.sort_unstable();
                vkeys.dedup();
                let (nu, nv) = (ukeys.len(), vkeys.len());
                let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nu];
                let mut uv: Vec<(usize, usize)> = Vec::with_capacity(fedges.len());
                for &ei in fedges {
                    let e = &edges[ei as usize];
                    let u = ukeys
                        .binary_search(&(self.chans[e.chan as usize], e.z))
                        .expect("U key present");
                    let v = vkeys
                        .binary_search(&(e.rem, e.refine_comp))
                        .expect("V key present");
                    adj[u].push(v);
                    uv.push((u, v));
                }
                for a in &mut adj {
                    a.sort_unstable();
                }
                drop(_t_prep);
                probe::bump(probe::Counter::CoverBlocks, 1);
                probe::bump(probe::Counter::CoverU, nu as u64);
                probe::bump(probe::Counter::CoverV, nv as u64);
                let _t_hk = probe::timer(probe::Stage::CoverHk);
                let (msize, mu, mv) = hopcroft_karp(nu, nv, &adj);
                let (mut cov_u, mut cov_v) = koenig_cover(nu, nv, &adj, &mu, &mv);
                drop(_t_hk);
                let _t_route = probe::timer(probe::Stage::Route);
                debug_assert_eq!(
                    cov_u.iter().filter(|&&c| c).count() + cov_v.iter().filter(|&&c| c).count(),
                    msize,
                    "koenig_cover non-minimum (decoupled) at site {k}, flux {flux:?}"
                );
                let median_switch_arm = self.uses_median_switch();
                if median_switch_arm {
                    cov_u.iter_mut().for_each(|c| *c = true);
                    cov_v.iter_mut().for_each(|c| *c = true);
                }
                let allow_u: Vec<bool> = if median_switch_arm {
                    fedges
                        .iter()
                        .map(|&ei| {
                            let e = &edges[ei as usize];
                            key_kind(self.chans[e.chan as usize]) == ChanKind::Comp
                                || self.median_switch_of(&e.rides) != k
                        })
                        .collect()
                } else {
                    vec![true; fedges.len()]
                };
                if k == d - 1 {
                    debug_assert!(
                        flux.up == 0
                            && flux.dn == 0
                            && (self.refine
                                || (nv == 1
                                    && vkeys
                                        .binary_search(&(PackedOps::default(), (0, 0)))
                                        .is_ok()))
                    );
                    cov_u.iter_mut().for_each(|c| *c = false);
                    cov_v.iter_mut().for_each(|c| *c = true);
                } else if flux == Flux::default() {
                    if self.refine {
                        for (ui, &(key, z)) in ukeys.iter().enumerate() {
                            if z == Z_ID
                                && key_kind(key) == ChanKind::Normal
                                && key_ops(key) == PackedOps::default()
                            {
                                cov_u[ui] = true;
                            }
                        }
                    } else if let Ok(ui) = ukeys.binary_search(&(identity_key(), Z_ID)) {
                        cov_u[ui] = true;
                    }
                }
                let mut taken = vec![false; fedges.len()];
                let mut u_start = vec![0u32; nu + 1];
                let mut v_start = vec![0u32; nv + 1];
                for &(u, v) in &uv {
                    u_start[u + 1] += 1;
                    v_start[v + 1] += 1;
                }
                for i in 0..nu {
                    u_start[i + 1] += u_start[i];
                }
                for i in 0..nv {
                    v_start[i + 1] += v_start[i];
                }
                let mut u_idx = vec![0u32; uv.len()];
                let mut v_idx = vec![0u32; uv.len()];
                {
                    let mut uc = u_start.clone();
                    let mut vc = v_start.clone();
                    for (idx, &(u, v)) in uv.iter().enumerate() {
                        u_idx[uc[u] as usize] = idx as u32;
                        uc[u] += 1;
                        v_idx[vc[v] as usize] = idx as u32;
                        vc[v] += 1;
                    }
                }
                macro_rules! collect_edge {
                    ($ei:expr, $touch:expr, $per_sec:expr) => {{
                        let e = &edges[$ei as usize];
                        let (lo, hi) = ($ei as usize * nwords, ($ei as usize + 1) * nwords);
                        for &du in op_delta_cands(e.zu) {
                            for &dd in op_delta_cands(e.zdn) {
                                let dg = Family::from_occupation(du == 1, dd == 1)
                                    .flux(space.pg.irrep_of(k))
                                    .index();
                                if bits_shift_and(
                                    &mut scratch,
                                    &chan_bits
                                        [e.chan as usize * nwords..(e.chan as usize + 1) * nwords],
                                    &src_mask[dd],
                                    du * (space.ndn + 1) + dd,
                                    &edge_adm[lo..hi],
                                    pw,
                                    dg,
                                ) {
                                    let bit = 1u8 << (du * 2 + dd);
                                    bits_for_each(&scratch, |dense| {
                                        let o = ord1[dense];
                                        debug_assert_ne!(o, u16::MAX);
                                        if $per_sec[o as usize] == 0 {
                                            $touch.push(o);
                                        }
                                        $per_sec[o as usize] |= bit;
                                    });
                                }
                            }
                        }
                    }};
                }
                macro_rules! emit_target {
                    ($cin:expr, $z:expr, $cout:expr, $alpha:expr, $src:expr, $touch:expr, $per_sec:expr) => {{
                        $touch.sort_unstable();
                        for &bo in $touch.iter() {
                            let mask = std::mem::replace(&mut $per_sec[bo as usize], 0);
                            emits.push(((bo, $cin, $cout, $z), ($alpha, mask, $src)));
                        }
                        $touch.clear();
                    }};
                }

                let normal_target = |e: &Edge<T>| -> u64 {
                    let cin_key = self.chans[e.chan as usize];
                    if key_kind(cin_key) == ChanKind::Normal {
                        chan_key_refined(
                            flux.up,
                            flux.dn,
                            flux.irrep,
                            ChanKind::Normal,
                            extend_applied(key_ops(cin_key), k, e.zu, e.zdn),
                            e.refine_normal,
                        )
                    } else {
                        chan_key_refined(
                            flux.up,
                            flux.dn,
                            flux.irrep,
                            ChanKind::Comp,
                            e.rem,
                            e.refine_comp,
                        )
                    }
                };
                for u in 0..nu {
                    if !cov_u[u] {
                        continue;
                    }
                    let mut cout: Option<u32> = None;
                    for &idx in &u_idx[u_start[u] as usize..u_start[u + 1] as usize] {
                        let idx = idx as usize;
                        if taken[idx] || (median_switch_arm && !allow_u[idx]) {
                            continue;
                        }
                        taken[idx] = true;
                        let ei = fedges[idx];
                        let e = &edges[ei as usize];
                        let c = *cout.get_or_insert_with(|| {
                            intern1(normal_target(e), &mut chans1, &mut chan_ord1)
                        });
                        debug_assert_eq!(
                            c,
                            chan_ord1[&normal_target(e)],
                            "U-vertex edges disagreed on their routing target at site {k}"
                        );
                        edge_target.insert(ei, c);
                        collect_edge!(ei, touch, per_sec);
                    }
                    if let Some(c) = cout {
                        let e0 = &edges[fedges[u_idx[u_start[u] as usize] as usize] as usize];
                        emit_target!(e0.chan, e0.z, c, T::one(), SRC_NONE, touch, per_sec);
                    }
                }
                for v in 0..nv {
                    if !cov_v[v] {
                        continue;
                    }
                    for &idx in &v_idx[v_start[v] as usize..v_start[v + 1] as usize] {
                        let idx = idx as usize;
                        if taken[idx] {
                            continue;
                        }
                        taken[idx] = true;
                        let ei = fedges[idx];
                        let e = &edges[ei as usize];
                        let out_key = chan_key_refined(
                            flux.up,
                            flux.dn,
                            flux.irrep,
                            ChanKind::Comp,
                            e.rem,
                            e.refine_comp,
                        );
                        let cout = intern1(out_key, &mut chans1, &mut chan_ord1);
                        edge_target.insert(ei, cout);
                        collect_edge!(ei, touch, per_sec);
                        let src = deposit_src(e);
                        emit_target!(e.chan, e.z, cout, e.val, src, touch, per_sec);
                    }
                }
                debug_assert!(
                    taken.iter().all(|&t| t),
                    "vertex cover left an edge unrouted (decoupled) at site {k}"
                );
                drop(_t_route);
            }
        } else {
            for (&(bo, flux), elist) in &blocks {
                let mut ukeys: Vec<(u64, u8)> = elist
                    .iter()
                    .map(|&(ei, _)| {
                        let e = &edges[ei as usize];
                        (self.chans[e.chan as usize], e.z)
                    })
                    .collect();
                ukeys.sort_unstable();
                ukeys.dedup();
                let mut vkeys: Vec<PackedOps> = elist
                    .iter()
                    .map(|&(ei, _)| edges[ei as usize].rem)
                    .collect();
                vkeys.sort_unstable();
                vkeys.dedup();
                let (nu, nv) = (ukeys.len(), vkeys.len());
                let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nu];
                let mut uv: Vec<(usize, usize)> = Vec::with_capacity(elist.len());
                for &(ei, _) in elist {
                    let e = &edges[ei as usize];
                    let u = ukeys
                        .binary_search(&(self.chans[e.chan as usize], e.z))
                        .expect("U key present");
                    let v = vkeys.binary_search(&e.rem).expect("V key present");
                    adj[u].push(v);
                    uv.push((u, v));
                }
                for a in &mut adj {
                    a.sort_unstable();
                }
                let (msize, mu, mv) = hopcroft_karp(nu, nv, &adj);
                let (mut cov_u, mut cov_v) = koenig_cover(nu, nv, &adj, &mu, &mv);
                debug_assert_eq!(
                    cov_u.iter().filter(|&&c| c).count() + cov_v.iter().filter(|&&c| c).count(),
                    msize,
                    "koenig_cover returned a non-minimum cover ({} vertices, matching size {}) at \
                 site {k}, block (sector-ordinal {bo}, flux {flux:?})",
                    cov_u.iter().filter(|&&c| c).count() + cov_v.iter().filter(|&&c| c).count(),
                    msize
                );
                if k == d - 1 {
                    debug_assert!(
                        flux.up == 0
                            && flux.dn == 0
                            && nv == 1
                            && vkeys.binary_search(&PackedOps::default()).is_ok()
                    );
                    cov_u.iter_mut().for_each(|c| *c = false);
                    cov_v.iter_mut().for_each(|c| *c = true);
                } else if flux == Flux::default()
                    && let Ok(ui) = ukeys.binary_search(&(identity_key(), Z_ID))
                {
                    cov_u[ui] = true;
                }
                let mut taken = vec![false; elist.len()];
                let mut route = |idx: usize, routed_u: bool| {
                    let (ei, mask) = elist[idx];
                    let e = &edges[ei as usize];
                    let cin_key = self.chans[e.chan as usize];
                    let out_key = if routed_u && key_kind(cin_key) == ChanKind::Normal {
                        chan_key(
                            flux.up,
                            flux.dn,
                            flux.irrep,
                            ChanKind::Normal,
                            extend_applied(key_ops(cin_key), k, e.zu, e.zdn),
                        )
                    } else {
                        chan_key(flux.up, flux.dn, flux.irrep, ChanKind::Comp, e.rem)
                    };
                    let cout = intern1(out_key, &mut chans1, &mut chan_ord1);
                    route_map.insert((ei, bo), cout);
                    let (alpha, src) = if routed_u {
                        (T::one(), SRC_NONE)
                    } else {
                        (e.val, deposit_src(e))
                    };
                    emits.push(((bo, e.chan, cout, e.z), (alpha, mask, src)));
                };
                for (u, &covered) in cov_u.iter().enumerate() {
                    if !covered {
                        continue;
                    }
                    for (idx, &(u2, _)) in uv.iter().enumerate() {
                        if u2 == u && !taken[idx] {
                            taken[idx] = true;
                            route(idx, true);
                        }
                    }
                }
                for (v, &covered) in cov_v.iter().enumerate() {
                    if !covered {
                        continue;
                    }
                    for (idx, &(_, v2)) in uv.iter().enumerate() {
                        if v2 == v && !taken[idx] {
                            taken[idx] = true;
                            route(idx, false);
                        }
                    }
                }
                debug_assert!(
                    taken.iter().all(|&t| t),
                    "vertex cover left an edge unrouted at site {k}"
                );
            }
        }
        drop(blocks);

        let _t_carrier = probe::timer(probe::Stage::Carrier);
        let mut rides1: Vec<Ride> = if per_flux {
            let mut r: Vec<Ride> = Vec::with_capacity(self.rides.len());
            for (ri, ride) in self.rides.iter().enumerate() {
                if let Some(&cout) = edge_target.get(&ride_edge[ri]) {
                    r.push(Ride {
                        term: ride.term,
                        chan: cout,
                        secs: Vec::new(),
                    });
                }
            }
            debug_assert!(
                {
                    let mut keys: Vec<(u32, u32)> = r.iter().map(|x| (x.term, x.chan)).collect();
                    keys.sort_unstable();
                    keys.windows(2).all(|w| w[0] != w[1])
                },
                "decoupled carrier produced duplicate (term, chan) rides at site {k}"
            );
            r
        } else {
            let mut new_rides: FxHashMap<(u32, u32), Vec<u16>> = FxHashMap::default();
            for (ri, ride) in self.rides.iter().enumerate() {
                let ei = ride_edge[ri];
                let e = &edges[ei as usize];
                let adm = &ride_adm[ri].ord;
                let mut local: Vec<(u32, Vec<u16>)> = Vec::new();
                for &so in &ride.secs {
                    let s = self.secs[so as usize];
                    for &du in op_delta_cands(e.zu) {
                        for &dd in op_delta_cands(e.zdn) {
                            let y_occ = Family::from_occupation(du == 1, dd == 1);
                            let o = sec_ord1(y_occ.apply(s, space.pg.irrep_of(k)));
                            if o == u16::MAX || !adm[o as usize] {
                                continue;
                            }
                            let cout = route_map[&(ei, o)];
                            match local.iter_mut().find(|(c, _)| *c == cout) {
                                Some((_, secs)) => secs.push(o),
                                None => local.push((cout, vec![o])),
                            }
                        }
                    }
                }
                for (cout, secs) in local {
                    new_rides.entry((ride.term, cout)).or_default().extend(secs);
                }
            }
            new_rides
                .into_iter()
                .map(|((term, chan), mut secs)| {
                    secs.sort_unstable();
                    secs.dedup();
                    Ride { term, chan, secs }
                })
                .collect()
        };
        rides1.sort_unstable_by_key(|r| (r.term, r.chan));
        drop(_t_carrier);

        let _t_instantiate = probe::timer(probe::Stage::Instantiate);
        inst_pairs.reserve(emits.len());
        for &((bo, _cin, cout, _z), _) in emits.iter() {
            inst_pairs.push((bo, cout));
        }
        inst_pairs.sort_unstable_by_key(|&(bo, c)| (bo, chans1[c as usize]));
        inst_pairs.dedup();
        let mut pos1: Vec<i32> = vec![-1i32; chans1.len() * nsec1];
        {
            let mut i = 0usize;
            while i < inst_pairs.len() {
                let bo = inst_pairs[i].0;
                let s = secs1[bo as usize];
                let mut j = i;
                let mut prev_key = 0u64;
                while j < inst_pairs.len() && inst_pairs[j].0 == bo {
                    let c = inst_pairs[j].1;
                    let key = chans1[c as usize];
                    debug_assert!(j == i || prev_key < key, "channel keys not sorted");
                    prev_key = key;
                    let p = j - i;
                    assert!(
                        p <= i32::MAX as usize,
                        "channel position overflows Int32 COO"
                    );
                    pos1[c as usize * nsec1 + bo as usize] = p as i32;
                    probe::bump(probe::Counter::Vertices, 1);
                    if !self.switch_only {
                        let fl = key_flux(key);
                        self.states[k + 1][s].push(ChannelDesc::Bipartite(key));
                        self.state_qns[k + 1][s].push(Sector::with_irrep(
                            (s.nup as i64 - fl.up as i64) as usize,
                            (s.ndn as i64 - fl.dn as i64) as usize,
                            s.irrep.fuse(fl.irrep.inverse()),
                        ));
                    }
                    j += 1;
                }
                i = j;
            }
        }

        drop(_t_instantiate);
        let _t_emit = probe::timer(probe::Stage::CooEmit);
        probe::bump(probe::Counter::EmitEntries, emits.len() as u64);
        if self.switch_only {
            self.chans = chans1;
            self.rides = rides1;
            self.secs = secs1;
            self.pos = pos1;
            self.pos_stride = nsec1;
            return;
        }
        debug_assert_eq!(
            self.coo.len(),
            k,
            "sweep visits sites left to right, once each"
        );
        let mut nest = NestBuilder::new_with_src(&space, k, self.record_src);
        let mut elist = emits;
        elist.sort_unstable_by_key(|&(key, _): &(EmitKey, EmitVal<T>)| key);
        if self.per_flux {
            assert!(
                elist.windows(2).all(|w| w[0].0 != w[1].0),
                "duplicate routing entry at site {k} — the per-U-vertex union produced two entries \
                 for one (sector, c_in, c_out, z)"
            );
        } else {
            let mut w = 0usize;
            for r in 0..elist.len() {
                if w > 0 && elist[w - 1].0 == elist[r].0 {
                    check_edge_collision(elist[w - 1].1.0, elist[r].1.0);
                    assert_eq!(
                        elist[w - 1].1.2,
                        elist[r].1.2,
                        "coalesced routing entries at site {k} carry different provenance"
                    );
                    elist[w - 1].1.1 |= elist[r].1.1;
                } else {
                    elist[w] = elist[r];
                    w += 1;
                }
            }
            elist.truncate(w);
        }
        for ((bo, cin, cout, z), (alpha, mask, src)) in elist {
            let n_plus = secs1[bo as usize];
            let fl_i = key_flux(self.chans[cin as usize]);
            let fl_o = key_flux(chans1[cout as usize]);
            let weighted = key_kind(self.chans[cin as usize]) == ChanKind::Normal
                && key_kind(chans1[cout as usize]) == ChanKind::Comp;
            if weighted {
                probe::bump(probe::Counter::EmitEntriesWeighted, 1);
            }
            let ops = z_unpack(z);
            let col = pos1[cout as usize * nsec1 + bo as usize];
            for du in 0..=1usize {
                for dd in 0..=1usize {
                    if mask & (1 << (du * 2 + dd)) == 0 {
                        continue;
                    }
                    let y_occ = Family::from_occupation(du == 1, dd == 1);
                    let Some(n_minus) = y_occ.unapply(n_plus, space.pg.irrep_of(k)) else {
                        continue;
                    };
                    let (l, r) = lr_sectors(n_minus, n_plus, fl_i, fl_o);
                    let (occ, neg) = jordanwigner_sign(l, r, ops);
                    let alpha2 = jw_apply(alpha, neg);
                    debug_assert_eq!(
                        occ.flux(space.pg.irrep_of(k)),
                        y_occ.flux(space.pg.irrep_of(k)).fuse(op_irrep_flux(
                            ops.0,
                            ops.1,
                            space.pg.irrep_of(k)
                        )),
                        "operator/state irrep bookkeeping disagree at site {k}"
                    );
                    let mo = ord0[bit_slot(&space, n_minus)];
                    debug_assert_ne!(mo, u16::MAX, "emitted sector admissible at bond k");
                    let row = self.pos[cin as usize * self.pos_stride + mo as usize];
                    debug_assert!(row >= 0, "row channel not instantiated at bond k");
                    if weighted {
                        probe::bump(probe::Counter::CooTriplesWeighted, 1);
                    }
                    push_coo_cell(
                        &mut nest,
                        n_minus,
                        n_plus,
                        occ,
                        l,
                        (row, col),
                        alpha2,
                        src_tag(src, neg),
                    );
                }
            }
        }
        self.coo.push(nest.freeze());

        drop(_t_emit);
        self.chans = chans1;
        self.rides = rides1;
        self.secs = secs1;
        self.pos = pos1;
        self.pos_stride = nsec1;
    }
}
