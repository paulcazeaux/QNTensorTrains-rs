// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;

use crate::hamiltonian::csx::CooCell;
use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::su2::core::Family;
use crate::su2::hamiltonian::{Su2CooEntry, Su2CooNest, Su2GridGeom, Ts};
use crate::su2::qn::{Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::spinalgebra::{SiteOp, SpinAlgebra};

use super::automaton::{
    Assembled, Automaton, EdgeKey, VertexId, blockclass, jl_window, op_irrep_flux,
};
use super::{Recorder, Src, Su2Factor, Su2RunSrc, alpha, frozen_run_cancels};

/// Sort-then-run-merge one accumulator cell into a frozen COO triple, dropping numerically-zero
/// cancellation debris. **KEEP the comment
/// verbatim**: a legitimately small LONE contribution is its own run's magnitude and is NEVER
/// dropped; only a run whose signed sum cancels to `<= 16*eps*runmax` is debris. Stable sort by
/// `(i1,i2)` preserves within-key edge order (load-bearing for the exact- zero cancellations the
/// rank-3/2 gate relies on).
pub fn freeze_cell<T: Scalar>(cell: &mut [(i32, i32, T)]) -> CooCell<T> {
    cell.sort_by_key(|a| (a.0, a.1));
    freeze_runs(cell, |_, _, s, runmax| value_keeps(s, runmax))
}

#[inline]
fn value_keeps<T: Scalar>(s: T, runmax: f64) -> bool {
    s.abs_s() > 16.0 * f64::EPSILON * runmax
}

pub(crate) fn freeze_cell_src<T: Scalar>(
    cell: &mut [(i32, i32, T)],
    src: &[(u32, u32)],
) -> (CooCell<T>, Su2RunSrc) {
    freeze_cell_src_with(cell, src, |_, s, runmax| value_keeps(s, runmax))
}

pub(crate) fn freeze_cell_frozen<T: Scalar>(
    cell: &mut [(i32, i32, T)],
    src: &[(u32, u32)],
    factors: &[Su2Factor],
) -> (CooCell<T>, Su2RunSrc, usize) {
    let mut debris = 0usize;
    let (coo, runs) = freeze_cell_src_with(cell, src, |run, _, _| {
        let keep = !frozen_run_cancels(run, factors);
        debris += usize::from(!keep);
        keep
    });
    (coo, runs, debris)
}

fn freeze_cell_src_with<T: Scalar>(
    cell: &mut [(i32, i32, T)],
    src: &[(u32, u32)],
    mut keep: impl FnMut(&[(u32, u32)], T, f64) -> bool,
) -> (CooCell<T>, Su2RunSrc) {
    assert_eq!(
        cell.len(),
        src.len(),
        "freeze_cell_src: provenance not parallel to the staged contributions"
    );
    let mut perm: Vec<usize> = (0..cell.len()).collect();
    perm.sort_by_key(|&p| (cell[p].0, cell[p].1));
    let sorted: Vec<(i32, i32, T)> = perm.iter().map(|&p| cell[p]).collect();
    cell.copy_from_slice(&sorted);
    let src: Vec<(u32, u32)> = perm.iter().map(|&p| src[p]).collect();
    let mut runs = Su2RunSrc::default();
    let coo = freeze_runs(cell, |lo, hi, s, runmax| {
        let kept = keep(&src[lo..hi], s, runmax);
        if kept {
            runs.push_run(&src[lo..hi]);
        }
        kept
    });
    (coo, runs)
}

#[inline]
fn freeze_runs<T: Scalar>(
    cell: &[(i32, i32, T)],
    mut keep: impl FnMut(usize, usize, T, f64) -> bool,
) -> CooCell<T> {
    let mut out = CooCell::empty();
    let n = cell.len();
    let mut p = 0usize;
    while p < n {
        let (i, j) = (cell[p].0, cell[p].1);
        let lo = p;
        let mut s = T::zero();
        let mut runmax = 0.0f64;
        while p < n && cell[p].0 == i && cell[p].1 == j {
            s += cell[p].2;
            runmax = runmax.max(cell[p].2.abs_s());
            p += 1;
        }
        if keep(lo, p, s, runmax) {
            out.rows.push(i);
            out.cols.push(j);
            out.vals.push(s);
        }
    }
    out
}

struct Su2AccEntry<T: Scalar> {
    x_occ: u8,
    m_ord: u32,
    acc: Vec<(i32, i32, T)>,
    src: Vec<(u32, u32)>,
}

pub(crate) struct Su2NestBuilder<T: Scalar> {
    geom: Su2GridGeom,
    sectors: Vec<Su2Sector>,
    mord: Vec<u32>,
    index: Vec<u32>,
    staged: Vec<Su2AccEntry<T>>,
}

impl<T: Scalar> Su2NestBuilder<T> {
    pub(crate) fn new(space: &Su2Space, k: usize) -> Self {
        let geom = Su2GridGeom::of(space);
        let sectors = space.bond_sectors(k);
        let mut mord = vec![u32::MAX; geom.nsec()];
        for (o, &s) in sectors.iter().enumerate() {
            mord[geom.idx(s)] = o as u32;
        }
        let index = vec![u32::MAX; 16 * geom.nsec() * sectors.len()];
        Su2NestBuilder {
            geom,
            sectors,
            mord,
            index,
            staged: Vec::new(),
        }
    }

    #[inline]
    fn acc_mut(
        &mut self,
        y_occ: Family,
        n1: Su2Sector,
        x_occ: Family,
        m: Su2Sector,
    ) -> &mut Su2AccEntry<T> {
        let ns_k = self.sectors.len();
        let mo = self.mord[self.geom.idx(m)];
        debug_assert!(
            mo != u32::MAX,
            "COO x-row sector is admissible at this bond"
        );
        let oslot = (y_occ as usize) * self.geom.nsec() + self.geom.idx(n1);
        let ck = (oslot * 4 + x_occ as usize) * ns_k + mo as usize;
        let mut i = self.index[ck];
        if i == u32::MAX {
            i = self.staged.len() as u32;
            assert!(i != u32::MAX, "populated COO cells overflow the u32 index");
            self.index[ck] = i;
            self.staged.push(Su2AccEntry {
                x_occ: x_occ as u8,
                m_ord: mo,
                acc: Vec::new(),
                src: Vec::new(),
            });
        }
        &mut self.staged[i as usize]
    }

    pub(crate) fn freeze<R: Recorder<T>>(mut self, rec: &mut R) -> Su2CooNest<T> {
        let (nsec, ns_k) = (self.geom.nsec(), self.sectors.len());
        let staged_n = self.staged.len();
        let mut visited = 0usize;
        let mut debris = 0usize;
        let mut offsets = Vec::with_capacity(4 * nsec + 1);
        let mut cells: Vec<Su2CooEntry<T>> = Vec::with_capacity(staged_n);
        let mut runs: Vec<Su2RunSrc> = Vec::new();
        for y_occ in 0..4usize {
            for n1g in 0..nsec {
                offsets.push(cells.len() as u32);
                let oslot = y_occ * nsec + n1g;
                for x_occ in 0..4usize {
                    let base = (oslot * 4 + x_occ) * ns_k;
                    for mo in 0..ns_k {
                        let i = self.index[base + mo];
                        if i == u32::MAX {
                            continue;
                        }
                        visited += 1;
                        let e = &mut self.staged[i as usize];
                        let (cell, run) = if R::ON {
                            let (cell, run) = match rec.frozen_factors() {
                                None => freeze_cell_src(&mut e.acc, &e.src),
                                Some(factors) => {
                                    let (cell, run, dropped) =
                                        freeze_cell_frozen(&mut e.acc, &e.src, factors);
                                    debris += dropped;
                                    (cell, run)
                                }
                            };
                            (cell, Some(run))
                        } else {
                            (freeze_cell(&mut e.acc), None)
                        };
                        if cell.rows.is_empty() {
                            continue;
                        }
                        runs.extend(run);
                        cells.push(Su2CooEntry {
                            x_occ: e.x_occ,
                            m_ord: e.m_ord,
                            cell,
                        });
                    }
                }
            }
        }
        offsets.push(cells.len() as u32);
        assert_eq!(
            visited,
            staged_n,
            "COO freeze lost {} staged cell(s) — an x-sector outside bond_sectors(k)",
            staged_n - visited
        );
        if R::ON {
            rec.site(runs, debris);
        }
        Su2CooNest::from_parts(offsets, cells, self.geom, self.sectors)
    }
}

pub(super) fn finalize<T: Scalar, R: Recorder<T>>(
    auto: Automaton<T>,
    space: Su2Space,
    spin: &mut SpinAlgebra,
    rec: &mut R,
) -> Assembled<T> {
    let d = space.d;

    let mut buckets: Vec<Su2SectorGrid<Vec<VertexId>>> = (0..=d)
        .map(|_| Su2SectorGrid::filled(&space, Vec::new()))
        .collect();
    for (id, lab) in auto.vlabels.iter().enumerate() {
        buckets[lab.bond][lab.sector].push(id);
    }

    let mut states: Vec<Su2SectorGrid<Vec<Ts>>> = (0..=d)
        .map(|_| Su2SectorGrid::filled(&space, Vec::new()))
        .collect();
    let mut state_qns: Vec<Su2SectorGrid<Vec<Su2Sector>>> = (0..=d)
        .map(|_| Su2SectorGrid::filled(&space, Vec::new()))
        .collect();
    let mut vid_jl_to_i: FxHashMap<(VertexId, i64), usize> = FxHashMap::default();

    for b in 0..=d {
        for s in space.bond_sectors(b) {
            buckets[b][s].sort_by(|&a, &c| {
                (auto.vlabels[a].state, auto.vlabels[a].open)
                    .cmp(&(auto.vlabels[c].state, auto.vlabels[c].open))
            });
            let mut flat: Vec<Ts> = Vec::new();
            let mut sq: Vec<Su2Sector> = Vec::new();
            for &id in &buckets[b][s] {
                let st = auto.vlabels[id].state;
                let open = auto.vlabels[id].open;
                let ml = s.n as i64 - st.flux as i64;
                if ml < 0 {
                    continue;
                }
                let x_ir = s.irrep.fuse(st.delta.inverse());
                for two_j in jl_window(s.two_s as i64, st.twok as i64) {
                    let m = Su2Sector::with_irrep(ml as usize, two_j as usize, x_ir);
                    if space.contains(b, m) {
                        vid_jl_to_i.insert((id, two_j), flat.len());
                        flat.push((st, open));
                        sq.push(m);
                    }
                }
            }
            states[b][s] = flat;
            state_qns[b][s] = sq;
        }
    }

    for s in space.bond_sectors(d) {
        for &id in &buckets[d][s] {
            let dl = auto.vlabels[id].state.delta;
            assert_eq!(
                dl,
                Irrep::IDENTITY,
                "bond-{d} automaton vertex carries total irrep flux Δ = {dl:?} ≠ IDENTITY: the \
                 assembled Hamiltonian is not totally symmetric. A symmetry-forbidden integral \
                 survived the ϵ screen — raise `eps` above the dump's forbidden-integral noise \
                 floor (~1e-15 for a produced FCIDUMP)."
            );
        }
    }

    let s0 = Su2Sector::new(0, 0);
    for &id in &buckets[0][s0] {
        let st = auto.vlabels[id].state;
        for two_j in jl_window(s0.two_s as i64, st.twok as i64) {
            if vid_jl_to_i.contains_key(&(id, two_j)) {
                vid_jl_to_i.insert((id, two_j), 0);
            }
        }
    }
    let sd = Su2Sector::with_irrep(space.n, space.two_s, space.target);
    for &id in &buckets[d][sd] {
        let st = auto.vlabels[id].state;
        for two_j in jl_window(sd.two_s as i64, st.twok as i64) {
            if vid_jl_to_i.contains_key(&(id, two_j)) {
                vid_jl_to_i.insert((id, two_j), 0);
            }
        }
    }
    if !state_qns[0][s0].is_empty() {
        state_qns[0][s0] = vec![s0];
    }
    if !state_qns[d][sd].is_empty() {
        state_qns[d][sd] = vec![sd];
    }

    assert!(
        vid_jl_to_i.values().all(|&i| i <= i32::MAX as usize),
        "flattened state index overflows Int32 COO storage"
    );

    let mut ekeys: Vec<EdgeKey> = auto.edges.keys().copied().collect();
    ekeys.sort_unstable();
    let frozen = R::ON && rec.frozen_factors().is_some();

    let mut by_site: Vec<Vec<EdgeKey>> = vec![Vec::new(); d];
    for &ekey in &ekeys {
        let k1 = auto.vlabels[(ekey >> 32) as VertexId].bond;
        if k1 < d {
            by_site[k1].push(ekey);
        }
    }

    let mut coo: Vec<Su2CooNest<T>> = Vec::with_capacity(d);
    for (k, site_keys) in by_site.iter().enumerate() {
        let mut nb = Su2NestBuilder::<T>::new(&space, k);
        for &ekey in site_keys {
            let (head_op, head_w) = &auto.edges[&ekey];
            let id_m = (ekey >> 32) as VertexId;
            let id_p = (ekey & 0xffff_ffff) as VertexId;
            let lm = auto.vlabels[id_m];
            let lp = auto.vlabels[id_p];
            let k1 = lm.bond;
            debug_assert_eq!(k1, k, "edge bucketed under the wrong output site");
            let g_k1 = space.pg.irrep_of(k1);
            let x_ir_m = lm.sector.irrep.fuse(lm.state.delta.inverse());
            let x_ir_p = lp.sector.irrep.fuse(lp.state.delta.inverse());
            let nl_m = lm.sector.n as i64;
            let sl_m = lm.sector.two_s as i64;
            let nl_p = lp.sector.n as i64;
            let sl_p = lp.sector.two_s as i64;
            let (occ_n, occ_y, twosy) = match blockclass(nl_p - nl_m, sl_p - sl_m) {
                Some(x) => x,
                None => continue,
            };
            let ml_m = nl_m - lm.state.flux as i64;
            let ml_p = nl_p - lp.state.flux as i64;
            if ml_m < 0 || ml_p < 0 {
                continue;
            }

            let ovf = auto.edges_ovf.get(&ekey);
            let n_extra = ovf.map(|v| v.len()).unwrap_or(0);
            if R::ON {
                debug_assert_eq!(
                    rec.contributions(ekey),
                    1 + n_extra,
                    "recording: edge {ekey:#x} recorded a different contribution count than the \
                     automaton appended"
                );
            }
            let win_m = jl_window(sl_m, lm.state.twok as i64);
            let win_p = jl_window(sl_p, lp.state.twok as i64);
            for ci in 0..=n_extra {
                let (op, w): (&SiteOp, &T) = if ci == 0 {
                    (head_op, head_w)
                } else {
                    let e = &ovf.unwrap()[ci - 1];
                    (&e.0, &e.1)
                };
                let src = if R::ON {
                    rec.src(ekey, ci)
                } else {
                    Src::Transport
                };
                let dm = ml_p - ml_m;
                if dm != (nl_p - nl_m) - op.flux as i64 {
                    continue;
                }
                let neg = op.odd && ml_m.rem_euclid(2) == 1;
                for &two_j_m in &win_m {
                    if !space.contains(
                        k1,
                        Su2Sector::with_irrep(ml_m as usize, two_j_m as usize, x_ir_m),
                    ) {
                        continue;
                    }
                    let i1 = match vid_jl_to_i.get(&(id_m, two_j_m)) {
                        Some(&i) => i,
                        None => continue,
                    };
                    let jps_two = [two_j_m - 1, two_j_m + 1];
                    let jps_one = [two_j_m];
                    let jps: &[i64] = if dm == 1 { &jps_two } else { &jps_one };
                    for &two_j_p in jps {
                        if two_j_p < 0 {
                            continue;
                        }
                        if !space.contains(
                            k1 + 1,
                            Su2Sector::with_irrep(ml_p as usize, two_j_p as usize, x_ir_p),
                        ) {
                            continue;
                        }
                        if !win_p.contains(&two_j_p) {
                            continue;
                        }
                        let (occ_m, occ_x, twosx) = match blockclass(dm, two_j_p - two_j_m) {
                            Some(x) => x,
                            None => continue,
                        };
                        debug_assert_eq!(
                            occ_m.flux(g_k1),
                            occ_n.flux(g_k1).fuse(op_irrep_flux(op, g_k1)),
                            "operator/state irrep bookkeeping disagree at site {k1}"
                        );
                        debug_assert_eq!(
                            x_ir_p.fuse(x_ir_m.inverse()),
                            occ_n.flux(g_k1).fuse(op_irrep_flux(op, g_k1)),
                            "edge endpoints' x-sector irreps disagree with the site-{k1} transition"
                        );
                        if op.red[occ_x][occ_y] == 0.0 {
                            continue;
                        }
                        let i2 = match vid_jl_to_i.get(&(id_p, two_j_p)) {
                            Some(&i) => i,
                            None => continue,
                        };
                        let f = spin.recouple(
                            two_j_m as i32,
                            two_j_p as i32,
                            twosx as i32,
                            lm.state.twok as i32,
                            lp.state.twok as i32,
                            op.two_k,
                            sl_m as i32,
                            sl_p as i32,
                            twosy as i32,
                        );
                        let fr = f * op.red[occ_x][occ_y];
                        let a = alpha(*w, neg, fr);
                        let skip = if frozen { fr == 0.0 } else { a.abs_s() == 0.0 };
                        if skip {
                            continue;
                        }
                        let e = nb.acc_mut(
                            Family::from_physical_index(occ_n as usize),
                            lm.sector,
                            Family::from_physical_index(occ_m as usize),
                            Su2Sector::with_irrep(ml_m as usize, two_j_m as usize, x_ir_m),
                        );
                        e.acc.push((i1 as i32, i2 as i32, a));
                        if R::ON {
                            e.src.push(rec.word(src, neg, fr));
                        }
                    }
                }
            }
        }
        coo.push(nb.freeze(rec));
    }

    Assembled {
        states,
        state_qns,
        coo,
    }
}
