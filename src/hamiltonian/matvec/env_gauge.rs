// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;

use faer::{Mat, unzip, zip};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::factor::Sketch;
use crate::core::{Family, SparseCore};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::assembly::ChannelDesc;
use crate::hamiltonian::bipartite::PackedOps;
use crate::hamiltonian::bipartite::key_with_ops;
use crate::hamiltonian::channel::{
    ChannelPhase, OpenOps, OrbitMono, channel_info, expand_under_pair_rotation, open_idx_of,
    orbit_coef, orbit_structure,
};
use crate::qn::Sector;
use crate::scalar::Scalar;

impl<T: Scalar> SparseHamiltonian<T> {
    /// Is every sector's channel list closed under the relabelling `i ↔ i+1` of every adjacent
    /// pair, for the channels the gauge mixes (Normal channels at bonds `≥ i + 2`, Comp channels at
    /// bonds `≤ i`), up to dead ends? A relabelling with no channel is admitted iff the channel
    /// relabelled is a DEAD END — no edge on the side its environment contracts, so its band is
    /// identically zero (module doc). `Err` names the first other missing channel. The median
    /// construction passes (its only misses are dead ends, which appear on typed spaces); the
    /// bipartite cover's Pauli-exact per-sector pruning fails. On a typed space only the SAME-irrep
    /// pairs are checked: a cross-irrep pair is never rotated, hence never gauged, and its
    /// relabelling need not close (module doc).
    pub fn env_gauge_closed(&self) -> Result<(), String> {
        let d = self.space.d;
        let pg = self.space.pg;
        let generic = [1.0, 1.0, 1.0, 1.0];
        for b in 1..d {
            for q in self.space.bond_sectors(b) {
                let descs = self.channel_descs(b, q);
                if descs.is_empty() {
                    continue;
                }
                let idx: HashMap<ChannelDesc, usize> =
                    descs.iter().enumerate().map(|(c, d)| (*d, c)).collect();
                let mut side = SideEdges::default();
                for (c, desc) in descs.iter().enumerate() {
                    let (phase, ops) = channel_info(desc, b);
                    for i in 0..d - 1 {
                        if pg.irrep_of(i) != pg.irrep_of(i + 1) {
                            continue;
                        }
                        let mixes = match phase {
                            ChannelPhase::Normal => b >= i + 2,
                            ChannelPhase::Comp => b <= i,
                        };
                        if !mixes || !ops.touches_pair(i) {
                            continue;
                        }
                        for (member, _) in expand_under_pair_rotation(&ops, i, generic) {
                            let target = match desc {
                                ChannelDesc::Median(state, _) => {
                                    ChannelDesc::Median(*state, open_idx_of(&member))
                                }
                                ChannelDesc::Bipartite(key) => ChannelDesc::Bipartite(
                                    key_with_ops(*key, PackedOps::new(&member.cre, &member.ann)),
                                ),
                            };
                            if !idx.contains_key(&target) && side.has_edge(self, b, q, phase, c) {
                                return Err(format!(
                                    "bond {b} sector {q:?}: {phase:?} channel {:?}/{:?} relabelled \
                                     on pair ({i}, {}) to {:?}/{:?} has no channel{NOT_DEAD_END}",
                                    ops.cre,
                                    ops.ann,
                                    i + 1,
                                    member.cre,
                                    member.ann
                                ));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Default)]
struct SideEdges {
    normal: Option<Vec<bool>>,
    comp: Option<Vec<bool>>,
}

impl SideEdges {
    fn has_edge<T: Scalar>(
        &mut self,
        h: &SparseHamiltonian<T>,
        b: usize,
        q: Sector,
        phase: ChannelPhase,
        c: usize,
    ) -> bool {
        let slot = match phase {
            ChannelPhase::Normal => &mut self.normal,
            ChannelPhase::Comp => &mut self.comp,
        };
        slot.get_or_insert_with(|| {
            let mut has = vec![false; h.channel_descs(b, q).len()];
            match phase {
                ChannelPhase::Normal => {
                    for y in Family::ALL {
                        for cell in h.nests[b].slot(y, q) {
                            for &r in &cell.csr.ids {
                                has[r as usize] = true;
                            }
                        }
                    }
                }
                ChannelPhase::Comp => {
                    let g = h.space.pg.irrep_of(b - 1);
                    for y in Family::ALL {
                        let Some(n1) = y.unapply(q, g) else { continue };
                        for cell in h.nests[b - 1].slot(y, n1) {
                            for &col in &cell.csc.ids {
                                has[col as usize] = true;
                            }
                        }
                    }
                }
            }
            has
        })[c]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Axis {
    Rows,
    Cols,
}

#[derive(Clone, Copy, Debug)]
struct Target {
    dst: u32,
    src: u32,
    m0: u32,
    m1: u32,
}

#[derive(Clone, Debug)]
struct SectorPlan {
    q: Sector,
    n_channels: usize,
    targets: Vec<Target>,
}

#[derive(Clone, Debug, Default)]
struct PairPlan {
    sectors: Vec<SectorPlan>,
    monos: Vec<OrbitMono>,
}

impl PairPlan {
    fn bytes(&self) -> usize {
        self.monos.len() * std::mem::size_of::<OrbitMono>()
            + self
                .sectors
                .iter()
                .map(|s| {
                    std::mem::size_of::<SectorPlan>()
                        + s.targets.len() * std::mem::size_of::<Target>()
                })
                .sum::<usize>()
    }
}

struct SectorChannels {
    q: Sector,
    info: Vec<(ChannelPhase, OpenOps)>,
    index: FxHashMap<ChannelDesc, u32>,
}

pub(crate) struct EnvGaugePlan {
    d: usize,
    plans: Vec<PairPlan>,
}

impl EnvGaugePlan {
    fn get(&self, b: usize, i: usize) -> &PairPlan {
        &self.plans[b * (self.d - 1) + i]
    }

    pub(crate) fn bytes(&self) -> usize {
        self.plans.iter().map(PairPlan::bytes).sum()
    }
}

impl<T: Scalar> SparseHamiltonian<T> {
    pub(crate) fn env_gauge_plan(&self) -> Result<EnvGaugePlan, String> {
        let d = self.space.d;
        let pg = self.space.pg;
        assert!(d >= 2, "env_gauge_plan: needs at least two sites");
        let per_bond: Vec<Result<Vec<(usize, PairPlan)>, String>> = (1..d)
            .into_par_iter()
            .map(|b| {
                let chans = self.decode_bond_channels(b);
                let mut out = Vec::new();
                for i in 0..d - 1 {
                    if pg.irrep_of(i) != pg.irrep_of(i + 1) {
                        continue;
                    }
                    let mix = if b >= i + 2 {
                        ChannelPhase::Normal
                    } else if b <= i {
                        ChannelPhase::Comp
                    } else {
                        continue;
                    };
                    out.push((i, self.pair_plan(b, i, mix, &chans)?));
                }
                Ok(out)
            })
            .collect();
        let mut plans = vec![PairPlan::default(); (d + 1) * (d - 1)];
        for (b, r) in (1..d).zip(per_bond) {
            for (i, p) in r? {
                plans[b * (d - 1) + i] = p;
            }
        }
        Ok(EnvGaugePlan { d, plans })
    }

    fn decode_bond_channels(&self, b: usize) -> Vec<SectorChannels> {
        self.space
            .bond_sectors(b)
            .into_iter()
            .filter_map(|q| {
                let descs = self.channel_descs(b, q);
                if descs.is_empty() {
                    return None;
                }
                Some(SectorChannels {
                    q,
                    info: descs.iter().map(|desc| channel_info(desc, b)).collect(),
                    index: descs
                        .iter()
                        .enumerate()
                        .map(|(c, desc)| (*desc, c as u32))
                        .collect(),
                })
            })
            .collect()
    }

    fn pair_plan(
        &self,
        b: usize,
        i: usize,
        mix: ChannelPhase,
        chans: &[SectorChannels],
    ) -> Result<PairPlan, String> {
        let mut plan = PairPlan::default();
        for sc in chans {
            let descs = self.channel_descs(b, sc.q);
            let mut targets: Vec<Target> = Vec::new();
            let mut side = SideEdges::default();
            for (c, (phase, ops)) in sc.info.iter().enumerate() {
                if *phase != mix || !ops.touches_pair(i) {
                    continue;
                }
                for (member, monos) in orbit_structure(ops, i) {
                    let target = match &descs[c] {
                        ChannelDesc::Median(state, _) => {
                            ChannelDesc::Median(*state, open_idx_of(&member))
                        }
                        ChannelDesc::Bipartite(key) => ChannelDesc::Bipartite(key_with_ops(
                            *key,
                            PackedOps::new(&member.cre, &member.ann),
                        )),
                    };
                    let Some(&dst) = sc.index.get(&target) else {
                        if !side.has_edge(self, b, sc.q, *phase, c) {
                            continue;
                        }
                        return Err(format!(
                            "bond {b} sector {:?}: {phase:?} channel {:?}/{:?} relabelled on pair \
                             ({i}, {}) to {:?}/{:?} has no channel{NOT_DEAD_END}",
                            sc.q,
                            ops.cre,
                            ops.ann,
                            i + 1,
                            member.cre,
                            member.ann
                        ));
                    };
                    let m0 = plan.monos.len() as u32;
                    plan.monos.extend(monos);
                    targets.push(Target {
                        dst,
                        src: c as u32,
                        m0,
                        m1: plan.monos.len() as u32,
                    });
                }
            }
            if targets.is_empty() {
                continue;
            }
            targets.sort_by_key(|t| (t.dst, t.src));
            plan.sectors.push(SectorPlan {
                q: sc.q,
                n_channels: descs.len(),
                targets,
            });
        }
        Ok(plan)
    }

    /// Gauge a RIGHT environment at bond `j = f.site() ≥ i + 2` under the rotation of orbitals
    /// `(i, i+1)` with `2×2` block `g` (`u[old][new]`, row-major, as `rotate_integrals` takes it).
    /// `ket_j` is the ket core at site `j` (its row bond is `j`) — only its ranks are read.
    /// Builds this `(bond, pair)`'s channel bookkeeping on the fly; the mode-rotating sweep builds
    /// it once for every `(bond, pair)` instead (bitwise the same result).
    pub fn rotate_right_env_adjacent(
        &self,
        ket_j: &SparseCore<T>,
        i: usize,
        g: [f64; 4],
        f: &mut Sketch<T>,
    ) {
        let j = self.check_right_env(ket_j, i, f);
        let plan = self
            .pair_plan(j, i, ChannelPhase::Normal, &self.decode_bond_channels(j))
            .unwrap_or_else(|e| panic!("gauge_env: {e}{NOT_CLOSED}"));
        self.apply_pair_plan(&plan, j, Axis::Rows, &self.row_ranges(ket_j), g, f);
    }

    /// Gauge a LEFT environment at bond `k = f.site() ≤ i` under the rotation of orbitals
    /// `(i, i+1)`; `ket_km1` is the ket core at site `k − 1` (its column bond is `k`). Plan built
    /// on the fly, as [`Self::rotate_right_env_adjacent`].
    pub fn rotate_left_env_adjacent(
        &self,
        ket_km1: &SparseCore<T>,
        i: usize,
        g: [f64; 4],
        f: &mut Sketch<T>,
    ) {
        let k = self.check_left_env(ket_km1, i, f);
        let plan = self
            .pair_plan(k, i, ChannelPhase::Comp, &self.decode_bond_channels(k))
            .unwrap_or_else(|e| panic!("gauge_env: {e}{NOT_CLOSED}"));
        self.apply_pair_plan(&plan, k, Axis::Cols, &self.col_ranges(ket_km1), g, f);
    }

    pub(crate) fn rotate_right_env_planned(
        &self,
        plan: &EnvGaugePlan,
        ket_j: &SparseCore<T>,
        i: usize,
        g: [f64; 4],
        f: &mut Sketch<T>,
    ) {
        let j = self.check_right_env(ket_j, i, f);
        assert_eq!(plan.d, self.space.d, "EnvGaugePlan built for another space");
        self.apply_pair_plan(plan.get(j, i), j, Axis::Rows, &self.row_ranges(ket_j), g, f);
    }

    pub(crate) fn rotate_left_env_planned(
        &self,
        plan: &EnvGaugePlan,
        ket_km1: &SparseCore<T>,
        i: usize,
        g: [f64; 4],
        f: &mut Sketch<T>,
    ) {
        let k = self.check_left_env(ket_km1, i, f);
        assert_eq!(plan.d, self.space.d, "EnvGaugePlan built for another space");
        self.apply_pair_plan(
            plan.get(k, i),
            k,
            Axis::Cols,
            &self.col_ranges(ket_km1),
            g,
            f,
        );
    }

    fn check_same_irrep(&self, i: usize) {
        let pg = self.space.pg;
        assert!(
            pg.irrep_of(i) == pg.irrep_of(i + 1),
            "gauge_env: pair ({i}, {}) spans two irreps — a cross-irrep rotation has no \
             representation on a typed space and is never gauged",
            i + 1
        );
    }

    fn check_right_env(&self, ket_j: &SparseCore<T>, i: usize, f: &Sketch<T>) -> usize {
        self.check_same_irrep(i);
        let j = f.site();
        assert!(
            j >= i + 2,
            "rotate_right_env_adjacent: bond {j} is not outside the pair ({i}, {}) — a right \
             environment must sit at bond ≥ i + 2",
            i + 1
        );
        assert_eq!(
            ket_j.k, j,
            "rotate_right_env_adjacent: ket core must be site {j}"
        );
        j
    }

    fn check_left_env(&self, ket_km1: &SparseCore<T>, i: usize, f: &Sketch<T>) -> usize {
        self.check_same_irrep(i);
        let k = f.site();
        assert!(
            k <= i && k >= 1,
            "rotate_left_env_adjacent: bond {k} is not outside the pair ({i}, {}) — a left \
             environment must sit at bond in 1..=i",
            i + 1
        );
        assert_eq!(
            ket_km1.k + 1,
            k,
            "rotate_left_env_adjacent: ket core must be site {}",
            k - 1
        );
        k
    }

    fn apply_pair_plan(
        &self,
        plan: &PairPlan,
        b: usize,
        axis: Axis,
        ranges: &crate::qn::SectorGrid<Vec<std::ops::Range<usize>>>,
        g: [f64; 4],
        f: &mut Sketch<T>,
    ) {
        let mut coefs: Vec<Option<f64>> = Vec::new();
        let (mut srcs, mut offs): (Vec<u32>, Vec<usize>) = (Vec::new(), Vec::new());
        for sp in &plan.sectors {
            let q = sp.q;
            assert_eq!(
                self.channel_descs(b, q).len(),
                sp.n_channels,
                "gauge plan built for another skeleton (bond {b} sector {q:?})"
            );
            let Some(old) = f.block(q) else { continue };
            let rq = &ranges[q];
            assert_eq!(
                rq.len(),
                sp.n_channels,
                "gauge_env: channel ranges disagree with the channel list at bond {b}"
            );
            let extent = rq.last().map_or(0, |r| r.end);
            let dim = match axis {
                Axis::Rows => old.nrows(),
                Axis::Cols => old.ncols(),
            };
            assert_eq!(
                extent, dim,
                "gauge_env: channel axis extent mismatch at bond {b}"
            );
            coefs.clear();
            coefs.extend(
                sp.targets
                    .iter()
                    .map(|t| orbit_coef(&plan.monos[t.m0 as usize..t.m1 as usize], &g)),
            );
            if coefs.iter().all(Option::is_none) {
                continue;
            }
            srcs.clear();
            srcs.extend(
                sp.targets
                    .iter()
                    .zip(&coefs)
                    .filter(|(_, c)| c.is_some())
                    .map(|(t, _)| t.src),
            );
            srcs.sort_unstable();
            srcs.dedup();
            offs.clear();
            let mut total = 0usize;
            for &c in &srcs {
                offs.push(total);
                total += rq[c as usize].len();
            }
            let saved: Mat<T> = match axis {
                Axis::Rows => {
                    let mut m = Mat::zeros(total, old.ncols());
                    for (&c, &o) in srcs.iter().zip(&offs) {
                        let r = &rq[c as usize];
                        m.subrows_mut(o, r.len())
                            .copy_from(old.subrows(r.start, r.len()));
                    }
                    m
                }
                Axis::Cols => {
                    let mut m = Mat::zeros(old.nrows(), total);
                    for (&c, &o) in srcs.iter().zip(&offs) {
                        let r = &rq[c as usize];
                        m.subcols_mut(o, r.len())
                            .copy_from(old.subcols(r.start, r.len()));
                    }
                    m
                }
            };
            let mut blk = f.block_mut(q).expect("block present");
            let mut last = u32::MAX;
            for (t, c) in sp.targets.iter().zip(&coefs) {
                if c.is_none() || t.dst == last {
                    continue;
                }
                last = t.dst;
                let r = &rq[t.dst as usize];
                match axis {
                    Axis::Rows => blk.as_mut().subrows_mut(r.start, r.len()).fill(T::zero()),
                    Axis::Cols => blk.as_mut().subcols_mut(r.start, r.len()).fill(T::zero()),
                }
            }
            for (t, c) in sp.targets.iter().zip(&coefs) {
                let Some(coef) = *c else { continue };
                let rd = &rq[t.dst as usize];
                let len = rq[t.src as usize].len();
                let o = offs[srcs.binary_search(&t.src).expect("source saved")];
                let a = T::from_f64(coef);
                match axis {
                    Axis::Rows => {
                        let nb = blk.as_mut().subrows_mut(rd.start, rd.len());
                        let ob = saved.subrows(o, len);
                        zip!(nb, ob).for_each(|unzip!(n, o)| *n += a * *o);
                    }
                    Axis::Cols => {
                        let nb = blk.as_mut().subcols_mut(rd.start, rd.len());
                        let ob = saved.subcols(o, len);
                        zip!(nb, ob).for_each(|unzip!(n, o)| *n += a * *o);
                    }
                }
            }
        }
    }
}

const NOT_CLOSED: &str = " — the operator's per-sector channel sets are not closed under the \
                          pair relabelling (`env_gauge_closed` refuses this skeleton; the \
                          bipartite cover's Pauli-exact pruning does this, the median \
                          construction does not)";

const NOT_DEAD_END: &str = ", and the channel is not a dead end (it has an edge on the side its \
                            environment contracts)";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::{Construction, TermScreen, switch_orbits_homogeneous};
    use crate::integrals::Integrals;
    use crate::qn::QnSpace;
    use crate::rotation::{givens, rotate_integrals};
    use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::test_util::randomized_state;
    use crate::vector::TtVector;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn dense_ints(d: usize, rng: &mut Xoshiro256PlusPlus) -> Integrals<f64> {
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for x in t.data.iter_mut().chain(v.data.iter_mut()) {
            *x = rng.random_range(-1.0..1.0);
        }
        Integrals {
            t,
            v,
            e0: 0.0,
            norb: d,
            nelec: 0,
            ms2: 0,
            orbsym: Vec::new(),
            isym: None,
        }
    }

    fn sketch_rel_diff<T: Scalar>(a: &Sketch<T>, b: &Sketch<T>) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for s in a.sectors() {
            match (a.block(s), b.block(s)) {
                (Some(x), Some(y)) => {
                    assert_eq!(
                        (x.nrows(), x.ncols()),
                        (y.nrows(), y.ncols()),
                        "shape at {s:?}"
                    );
                    for r in 0..x.nrows() {
                        for c in 0..x.ncols() {
                            let dd = x[(r, c)] - y[(r, c)];
                            num += dd.abs_s() * dd.abs_s();
                            den += y[(r, c)].abs_s() * y[(r, c)].abs_s();
                        }
                    }
                }
                (None, None) => {}
                _ => panic!("block presence differs at {s:?}"),
            }
        }
        (num / den.max(1e-300)).sqrt()
    }

    #[test]
    fn t9_env_gauge_matches_recontraction() {
        let d = 6;
        let space = QnSpace::from_n_two_sz(d, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6A_06E);
        let ints = dense_ints(d, &mut rng);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let screen = TermScreen::SymmetryAllowed(space.pg);
        for c in [Construction::Median, Construction::Bipartite] {
            let verdict = switch_orbits_homogeneous(space, &ints.t, &ints.v, &screen, c);
            println!("{c:?}: switch_orbits_homogeneous = {verdict:?}");
            assert!(
                verdict.is_ok(),
                "{c:?}: the switch rule must be orbit-homogeneous"
            );
            let h =
                SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, screen.clone(), c);
            let closed = h.env_gauge_closed();
            println!("{c:?}: env_gauge_closed = {closed:?}");
            match c {
                Construction::Median => assert!(closed.is_ok(), "median must be closed"),
                _ => {
                    assert!(
                        closed.is_err(),
                        "bipartite closure: if this now holds, enable it"
                    );
                    continue;
                }
            }
            let fr = right_to_left_sketching(&h, &x);
            let fl = left_to_right_sketching(&h, &x);
            for i in 0..d - 1 {
                let theta = 0.37 + 0.1 * i as f64;
                let u = givens(d, i, theta);
                let g = [
                    u[i * d + i],
                    u[i * d + i + 1],
                    u[(i + 1) * d + i],
                    u[(i + 1) * d + i + 1],
                ];
                let rot = rotate_integrals(&ints, &u);
                let h2 = SparseHamiltonian::new_refillable(
                    space,
                    &rot.t,
                    &rot.v,
                    None,
                    screen.clone(),
                    c,
                );
                assert_eq!(
                    h.skeleton_hash(),
                    h2.skeleton_hash(),
                    "{c:?}: skeleton must not move"
                );
                let fr2 = right_to_left_sketching(&h2, &x);
                let fl2 = left_to_right_sketching(&h2, &x);
                for j in i + 2..d {
                    let mut f = fr[j - 1].clone();
                    assert_eq!(f.site(), j);
                    h.rotate_right_env_adjacent(x.core(j), i, g, &mut f);
                    let rel = sketch_rel_diff(&f, &fr2[j - 1]);
                    assert!(
                        rel < 1e-12,
                        "{c:?}: right env pair {i} bond {j}: rel {rel:.3e}"
                    );
                }
                for k in 1..=i {
                    let mut f = fl[k].clone();
                    assert_eq!(f.site(), k);
                    h.rotate_left_env_adjacent(x.core(k - 1), i, g, &mut f);
                    let rel = sketch_rel_diff(&f, &fl2[k]);
                    assert!(
                        rel < 1e-12,
                        "{c:?}: left env pair {i} bond {k}: rel {rel:.3e}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "not outside the pair")]
    fn t9_right_env_inside_the_pair_is_refused() {
        let d = 5;
        let space = QnSpace::from_n_two_sz(d, 2, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6A_06F);
        let ints = dense_ints(d, &mut rng);
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let h = SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
        let fr = right_to_left_sketching(&h, &x);
        let mut f = fr[1].clone();
        h.rotate_right_env_adjacent(x.core(2), 1, [1.0, 0.0, 0.0, 1.0], &mut f);
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_gauge_env<T: Scalar>(
        h: &SparseHamiltonian<T>,
        b: usize,
        mix: ChannelPhase,
        axis: Axis,
        ranges: &crate::qn::SectorGrid<Vec<std::ops::Range<usize>>>,
        i: usize,
        g: [f64; 4],
        f: &mut Sketch<T>,
    ) {
        for q in h.space.bond_sectors(b) {
            let descs = h.channel_descs(b, q);
            if descs.is_empty() {
                continue;
            }
            let Some(old) = f.block(q) else { continue };
            let rq = &ranges[q];
            assert_eq!(
                rq.len(),
                descs.len(),
                "gauge_env: channel ranges disagree with the channel list at bond {b}"
            );
            let extent = rq.last().map_or(0, |r| r.end);
            let dim = match axis {
                Axis::Rows => old.nrows(),
                Axis::Cols => old.ncols(),
            };
            assert_eq!(
                extent, dim,
                "gauge_env: channel axis extent mismatch at bond {b}"
            );
            let mut index: Option<HashMap<ChannelDesc, usize>> = None;
            let mut targets: Vec<(usize, usize, f64)> = Vec::new();
            for (c, desc) in descs.iter().enumerate() {
                let (phase, ops) = channel_info(desc, b);
                if phase != mix || !ops.touches_pair(i) {
                    continue;
                }
                let idx = index.get_or_insert_with(|| {
                    descs.iter().enumerate().map(|(c, d)| (*d, c)).collect()
                });
                for (member, coef) in expand_under_pair_rotation(&ops, i, g) {
                    let target = match desc {
                        ChannelDesc::Median(state, _) => {
                            ChannelDesc::Median(*state, open_idx_of(&member))
                        }
                        ChannelDesc::Bipartite(key) => ChannelDesc::Bipartite(key_with_ops(
                            *key,
                            PackedOps::new(&member.cre, &member.ann),
                        )),
                    };
                    let Some(&dst) = idx.get(&target) else {
                        panic!(
                            "gauge_env: relabelled string {:?}/{:?} has no channel at bond {b} \
                             sector {q:?} — the operator's per-sector channel sets are not closed \
                             under the pair relabelling (`env_gauge_closed` refuses this skeleton; \
                             the bipartite cover's Pauli-exact pruning does this, the median \
                             construction does not)",
                            member.cre, member.ann
                        );
                    };
                    targets.push((dst, c, coef));
                }
            }
            if targets.is_empty() {
                continue;
            }
            let old: Mat<T> = old.to_owned();
            let mut new = old.clone();
            let mut dst_seen: Vec<usize> = targets.iter().map(|t| t.0).collect();
            dst_seen.sort_unstable();
            dst_seen.dedup();
            for &dst in &dst_seen {
                let r = &rq[dst];
                match axis {
                    Axis::Rows => new.subrows_mut(r.start, r.len()).fill(T::zero()),
                    Axis::Cols => new.subcols_mut(r.start, r.len()).fill(T::zero()),
                }
            }
            targets.sort_by_key(|&(dst, src, _)| (dst, src));
            for &(dst, src, coef) in &targets {
                let (rd, rs) = (&rq[dst], &rq[src]);
                let a = T::from_f64(coef);
                match axis {
                    Axis::Rows => {
                        let mut nb = new.subrows_mut(rd.start, rd.len());
                        let ob = old.subrows(rs.start, rs.len());
                        zip!(nb.as_mut(), ob).for_each(|unzip!(n, o)| *n += a * *o);
                    }
                    Axis::Cols => {
                        let mut nb = new.subcols_mut(rd.start, rd.len());
                        let ob = old.subcols(rs.start, rs.len());
                        zip!(nb.as_mut(), ob).for_each(|unzip!(n, o)| *n += a * *o);
                    }
                }
            }
            let mut blk = f.block_mut(q).expect("block present");
            blk.copy_from(&new);
        }
    }

    fn sketch_bits<T: Scalar>(f: &Sketch<T>) -> Vec<(crate::qn::Sector, Vec<u64>)> {
        f.sectors()
            .into_iter()
            .filter_map(|s| {
                f.block(s).map(|b| {
                    let mut v = Vec::with_capacity(2 * b.nrows() * b.ncols());
                    for c in 0..b.ncols() {
                        for r in 0..b.nrows() {
                            v.push(b[(r, c)].re_s().to_bits());
                            v.push(b[(r, c)].im_s().to_bits());
                        }
                    }
                    (s, v)
                })
            })
            .collect()
    }

    #[test]
    fn t9_planned_gauge_is_bitwise_the_reference() {
        let d = 6;
        let space = QnSpace::from_n_two_sz(d, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6A_070);
        let ints = dense_ints(d, &mut rng);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let screen = TermScreen::SymmetryAllowed(space.pg);
        let h = SparseHamiltonian::new_refillable(
            space,
            &ints.t,
            &ints.v,
            None,
            screen,
            Construction::Median,
        );
        let plan = h.env_gauge_plan().expect("median skeleton is gauge-closed");
        assert!(plan.bytes() > 0);
        let fr = right_to_left_sketching(&h, &x);
        let fl = left_to_right_sketching(&h, &x);
        let hh = std::f64::consts::FRAC_1_SQRT_2;
        let (mut n_cases, mut n_moved) = (0usize, 0usize);
        for i in 0..d - 1 {
            let (c, s) = ((0.37 + 0.1 * i as f64).cos(), (0.37 + 0.1 * i as f64).sin());
            for g in [[c, s, -s, c], [hh, hh, -hh, hh], [0.0, 1.0, -1.0, 0.0]] {
                for j in i + 2..d {
                    let f0 = fr[j - 1].clone();
                    let mut want = f0.clone();
                    let ranges = h.row_ranges(x.core(j));
                    reference_gauge_env(
                        &h,
                        j,
                        ChannelPhase::Normal,
                        Axis::Rows,
                        &ranges,
                        i,
                        g,
                        &mut want,
                    );
                    let mut planned = f0.clone();
                    h.rotate_right_env_planned(&plan, x.core(j), i, g, &mut planned);
                    let mut adjacent = f0.clone();
                    h.rotate_right_env_adjacent(x.core(j), i, g, &mut adjacent);
                    let w = sketch_bits(&want);
                    assert!(
                        w == sketch_bits(&planned),
                        "right env pair {i} bond {j} g {g:?}: planned"
                    );
                    assert!(
                        w == sketch_bits(&adjacent),
                        "right env pair {i} bond {j} g {g:?}: adjacent"
                    );
                    n_moved += (w != sketch_bits(&f0)) as usize;
                    n_cases += 1;
                }
                for (k, fk) in fl.iter().enumerate().take(i + 1).skip(1) {
                    let f0 = fk.clone();
                    let mut want = f0.clone();
                    let ranges = h.col_ranges(x.core(k - 1));
                    reference_gauge_env(
                        &h,
                        k,
                        ChannelPhase::Comp,
                        Axis::Cols,
                        &ranges,
                        i,
                        g,
                        &mut want,
                    );
                    let mut planned = f0.clone();
                    h.rotate_left_env_planned(&plan, x.core(k - 1), i, g, &mut planned);
                    let mut adjacent = f0.clone();
                    h.rotate_left_env_adjacent(x.core(k - 1), i, g, &mut adjacent);
                    let w = sketch_bits(&want);
                    assert!(
                        w == sketch_bits(&planned),
                        "left env pair {i} bond {k} g {g:?}: planned"
                    );
                    assert!(
                        w == sketch_bits(&adjacent),
                        "left env pair {i} bond {k} g {g:?}: adjacent"
                    );
                    n_moved += (w != sketch_bits(&f0)) as usize;
                    n_cases += 1;
                }
            }
        }
        println!("{n_cases} (pair, bond, g) cases, {n_moved} moved by the gauge");
        assert!(
            n_moved * 2 > n_cases,
            "the reference barely moves: the pin is near-vacuous"
        );
    }

    #[allow(clippy::type_complexity)]
    fn typed_twin() -> (
        QnSpace,
        Integrals<f64>,
        TtVector<f64>,
        SparseHamiltonian<f64>,
    ) {
        typed_dense(&[0, 0, 0, 2, 2, 3], (2, 2), 0, 0x6A_072)
    }

    #[allow(clippy::type_complexity)]
    fn typed_dense(
        table: &[u8],
        (nup, ndn): (usize, usize),
        target: u8,
        seed: u64,
    ) -> (
        QnSpace,
        Integrals<f64>,
        TtVector<f64>,
        SparseHamiltonian<f64>,
    ) {
        use crate::pointgroup::{Irrep, PointGroup};
        let d = table.len();
        let irs: Vec<Irrep> = table.iter().map(|&i| Irrep::new(i)).collect();
        let pg = PointGroup::new(2, &irs);
        let space = QnSpace::with_point_group(d, nup, ndn, pg, Irrep::new(target));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut ints = dense_ints(d, &mut rng);
        for i in 0..d {
            for j in 0..d {
                if pg.irrep_of_sites([i, j]) != Irrep::IDENTITY {
                    ints.t.set(&[i, j], 0.0);
                }
                for k in 0..d {
                    for l in 0..d {
                        if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                            ints.v.set(&[i, j, k, l], 0.0);
                        }
                    }
                }
            }
        }
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let h = SparseHamiltonian::new_refillable(
            space,
            &ints.t,
            &ints.v,
            None,
            TermScreen::SymmetryAllowed(pg),
            Construction::Median,
        );
        (space, ints, x, h)
    }

    #[test]
    fn t9_typed_plan_same_irrep_pairs_only() {
        let (space, ints, x, h) = typed_twin();
        let (d, pg) = (space.d, space.pg);
        let (same, cross): (Vec<usize>, Vec<usize>) =
            (0..d - 1).partition(|&i| pg.irrep_of(i) == pg.irrep_of(i + 1));
        assert_eq!((same.clone(), cross.clone()), (vec![0, 1, 3], vec![2, 4]));
        let screen = TermScreen::SymmetryAllowed(pg);
        let closed = h.env_gauge_closed();
        assert!(closed.is_ok(), "typed median skeleton: {closed:?}");
        let plan = h.env_gauge_plan().expect("typed median plan");
        for &i in &cross {
            for b in 1..d {
                let p = plan.get(b, i);
                assert!(
                    p.sectors.is_empty() && p.monos.is_empty(),
                    "cross-irrep pair {i} planned at bond {b}"
                );
            }
        }
        assert!(plan.bytes() > 0, "the same-irrep pairs must be planned");
        let fr = right_to_left_sketching(&h, &x);
        let fl = left_to_right_sketching(&h, &x);
        let (mut moved_r, mut moved_l, mut n_r, mut n_l) = (0usize, 0usize, 0usize, 0usize);
        for &i in &same {
            let theta = 0.41 + 0.1 * i as f64;
            let u = givens(d, i, theta);
            let g = [
                u[i * d + i],
                u[i * d + i + 1],
                u[(i + 1) * d + i],
                u[(i + 1) * d + i + 1],
            ];
            let rot = rotate_integrals(&ints, &u);
            let h2 = SparseHamiltonian::new_refillable(
                space,
                &rot.t,
                &rot.v,
                None,
                screen.clone(),
                Construction::Median,
            );
            assert_eq!(
                h.skeleton_hash(),
                h2.skeleton_hash(),
                "pair {i}: skeleton moved"
            );
            let fr2 = right_to_left_sketching(&h2, &x);
            let fl2 = left_to_right_sketching(&h2, &x);
            for j in i + 2..d {
                let f0 = fr[j - 1].clone();
                let mut want = f0.clone();
                let ranges = h.row_ranges(x.core(j));
                reference_gauge_env(
                    &h,
                    j,
                    ChannelPhase::Normal,
                    Axis::Rows,
                    &ranges,
                    i,
                    g,
                    &mut want,
                );
                let mut planned = f0.clone();
                h.rotate_right_env_planned(&plan, x.core(j), i, g, &mut planned);
                let mut adjacent = f0.clone();
                h.rotate_right_env_adjacent(x.core(j), i, g, &mut adjacent);
                let w = sketch_bits(&want);
                assert!(
                    w == sketch_bits(&planned),
                    "right pair {i} bond {j}: planned"
                );
                assert!(
                    w == sketch_bits(&adjacent),
                    "right pair {i} bond {j}: adjacent"
                );
                let rel = sketch_rel_diff(&planned, &fr2[j - 1]);
                assert!(rel < 1e-12, "right pair {i} bond {j}: rel {rel:.3e}");
                moved_r += (w != sketch_bits(&f0)) as usize;
                n_r += 1;
            }
            for k in 1..=i {
                let f0 = fl[k].clone();
                let mut want = f0.clone();
                let ranges = h.col_ranges(x.core(k - 1));
                reference_gauge_env(
                    &h,
                    k,
                    ChannelPhase::Comp,
                    Axis::Cols,
                    &ranges,
                    i,
                    g,
                    &mut want,
                );
                let mut planned = f0.clone();
                h.rotate_left_env_planned(&plan, x.core(k - 1), i, g, &mut planned);
                let mut adjacent = f0.clone();
                h.rotate_left_env_adjacent(x.core(k - 1), i, g, &mut adjacent);
                let w = sketch_bits(&want);
                assert!(
                    w == sketch_bits(&planned),
                    "left pair {i} bond {k}: planned"
                );
                assert!(
                    w == sketch_bits(&adjacent),
                    "left pair {i} bond {k}: adjacent"
                );
                let rel = sketch_rel_diff(&planned, &fl2[k]);
                assert!(rel < 1e-12, "left pair {i} bond {k}: rel {rel:.3e}");
                moved_l += (w != sketch_bits(&f0)) as usize;
                n_l += 1;
            }
        }
        println!("right {n_r} cases ({moved_r} moved), left {n_l} cases ({moved_l} moved)");
        assert!(
            moved_r > 0 && moved_l > 0,
            "the gauge moved nothing on a stack"
        );
    }

    #[test]
    fn t9_typed_gauge_refuses_a_cross_irrep_pair() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let (space, _ints, x, h) = typed_twin();
        let pg = space.pg;
        let plan = h.env_gauge_plan().expect("typed median plan");
        let fr = right_to_left_sketching(&h, &x);
        let fl = left_to_right_sketching(&h, &x);
        let g = [0.8, -0.6, 0.6, 0.8];
        let refused = |label: &str, f0: &Sketch<f64>, call: &dyn Fn(&mut Sketch<f64>)| {
            let mut f = f0.clone();
            let err = catch_unwind(AssertUnwindSafe(|| call(&mut f)))
                .expect_err(&format!("{label}: a cross-irrep pair was gauged"));
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default();
            assert!(
                msg.contains("spans two irreps"),
                "{label}: wrong refusal: {msg}"
            );
            assert!(
                sketch_bits(&f) == sketch_bits(f0),
                "{label}: f mutated before the refusal"
            );
        };
        let mut n = 0;
        for i in [2usize, 4] {
            assert_ne!(pg.irrep_of(i), pg.irrep_of(i + 1), "pair {i} must be cross");
            for j in i + 2..space.d {
                let f0 = &fr[j - 1];
                refused(&format!("right planned {i}/{j}"), f0, &|f| {
                    h.rotate_right_env_planned(&plan, x.core(j), i, g, f)
                });
                refused(&format!("right adjacent {i}/{j}"), f0, &|f| {
                    h.rotate_right_env_adjacent(x.core(j), i, g, f)
                });
                n += 2;
            }
            for (k, f0) in fl.iter().enumerate().take(i + 1).skip(1) {
                refused(&format!("left planned {i}/{k}"), f0, &|f| {
                    h.rotate_left_env_planned(&plan, x.core(k - 1), i, g, f)
                });
                refused(&format!("left adjacent {i}/{k}"), f0, &|f| {
                    h.rotate_left_env_adjacent(x.core(k - 1), i, g, f)
                });
                n += 2;
            }
        }
        assert_eq!(n, 16, "two entries per stack per (pair, bond)");
    }

    #[test]
    fn t9_env_gauge_plan_refuses_the_bipartite_cover() {
        let d = 6;
        let space = QnSpace::from_n_two_sz(d, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6A_071);
        let ints = dense_ints(d, &mut rng);
        let screen = TermScreen::SymmetryAllowed(space.pg);
        let h = SparseHamiltonian::new_refillable(
            space,
            &ints.t,
            &ints.v,
            None,
            screen,
            Construction::Bipartite,
        );
        assert!(h.env_gauge_closed().is_err());
        let e = h
            .env_gauge_plan()
            .err()
            .expect("bipartite plan must be refused");
        assert!(e.contains("has no channel"), "{e}");
    }

    struct StrictMiss {
        b: usize,
        q: Sector,
        c: usize,
        phase: ChannelPhase,
    }

    fn strict_misses(h: &SparseHamiltonian<f64>) -> Vec<StrictMiss> {
        let (d, pg) = (h.space.d, h.space.pg);
        let mut out = Vec::new();
        for b in 1..d {
            for q in h.space.bond_sectors(b) {
                let descs = h.channel_descs(b, q);
                let idx: HashMap<ChannelDesc, usize> =
                    descs.iter().enumerate().map(|(c, d)| (*d, c)).collect();
                for (c, desc) in descs.iter().enumerate() {
                    let ChannelDesc::Median(state, _) = desc else {
                        panic!("strict_misses: a median skeleton only")
                    };
                    let (phase, ops) = channel_info(desc, b);
                    for i in (0..d - 1).filter(|&i| pg.irrep_of(i) == pg.irrep_of(i + 1)) {
                        let mixes = match phase {
                            ChannelPhase::Normal => b >= i + 2,
                            ChannelPhase::Comp => b <= i,
                        };
                        if !mixes || !ops.touches_pair(i) {
                            continue;
                        }
                        for (member, _) in expand_under_pair_rotation(&ops, i, [1.0; 4]) {
                            let target = ChannelDesc::Median(*state, open_idx_of(&member));
                            if !idx.contains_key(&target) {
                                out.push(StrictMiss { b, q, c, phase });
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn t9_typed_dead_end_misses_gauge_exactly() {
        type Case = (
            &'static str,
            &'static [u8],
            (usize, usize),
            u8,
            u64,
            [bool; 4],
        );
        let cases: [Case; 3] = [
            (
                "A",
                &[0, 1, 1, 1, 2],
                (2, 2),
                2,
                0x6A_0A1,
                [true, false, true, false],
            ),
            (
                "B",
                &[0, 0, 1, 2, 2, 3],
                (3, 3),
                0,
                0x6A_0B1,
                [false, false, true, true],
            ),
            (
                "C",
                &[0, 1, 1, 2, 3, 3, 3],
                (4, 3),
                1,
                0x6A_0C1,
                [true, true, false, false],
            ),
        ];
        for (label, table, n, target, seed, want) in cases {
            let (space, ints, x, h) = typed_dense(table, n, target, seed);
            let (d, pg) = (space.d, space.pg);
            let misses = strict_misses(&h);
            assert!(
                !misses.is_empty(),
                "{label}: vacuous — the strict rule closes"
            );
            let closed = h.env_gauge_closed();
            assert!(closed.is_ok(), "{label}: {closed:?}");
            let plan = h.env_gauge_plan().expect("dead-end skeleton plan");
            let fr = right_to_left_sketching(&h, &x);
            let fl = left_to_right_sketching(&h, &x);
            let mut seen = [false; 4];
            for m in &misses {
                let (f, ranges, axis) = match m.phase {
                    ChannelPhase::Normal => (&fr[m.b - 1], h.row_ranges(x.core(m.b)), Axis::Rows),
                    ChannelPhase::Comp => (&fl[m.b], h.col_ranges(x.core(m.b - 1)), Axis::Cols),
                };
                let band = ranges[m.q][m.c].clone();
                let nonempty = !band.is_empty() && f.block(m.q).is_some();
                if let Some(blk) = f.block(m.q) {
                    let zero = match axis {
                        Axis::Rows => band
                            .clone()
                            .all(|r| (0..blk.ncols()).all(|c| blk[(r, c)] == 0.0)),
                        Axis::Cols => band
                            .clone()
                            .all(|c| (0..blk.nrows()).all(|r| blk[(r, c)] == 0.0)),
                    };
                    assert!(
                        zero,
                        "{label}: miss source band not zero (bond {}, {:?}, channel {})",
                        m.b, m.q, m.c
                    );
                }
                let k = 2 * (m.phase == ChannelPhase::Comp) as usize + nonempty as usize;
                seen[k] = true;
            }
            println!("{label}: {} strict misses, kinds {seen:?}", misses.len());
            for (k, (&w, &s)) in want.iter().zip(&seen).enumerate() {
                assert!(!w || s, "{label}: miss kind {k} not realized: {seen:?}");
            }
            let (mut moved_r, mut moved_l, mut max_rel) = (0usize, 0usize, 0.0f64);
            for i in (0..d - 1).filter(|&i| pg.irrep_of(i) == pg.irrep_of(i + 1)) {
                let u = givens(d, i, 0.43 + 0.1 * i as f64);
                let g = [
                    u[i * d + i],
                    u[i * d + i + 1],
                    u[(i + 1) * d + i],
                    u[(i + 1) * d + i + 1],
                ];
                let rot = rotate_integrals(&ints, &u);
                let h2 = SparseHamiltonian::new_refillable(
                    space,
                    &rot.t,
                    &rot.v,
                    None,
                    TermScreen::SymmetryAllowed(pg),
                    Construction::Median,
                );
                assert_eq!(
                    h.skeleton_hash(),
                    h2.skeleton_hash(),
                    "{label}: skeleton moved"
                );
                let fr2 = right_to_left_sketching(&h2, &x);
                let fl2 = left_to_right_sketching(&h2, &x);
                for j in i + 2..d {
                    let f0 = fr[j - 1].clone();
                    let mut planned = f0.clone();
                    h.rotate_right_env_planned(&plan, x.core(j), i, g, &mut planned);
                    let mut adjacent = f0.clone();
                    h.rotate_right_env_adjacent(x.core(j), i, g, &mut adjacent);
                    let p = sketch_bits(&planned);
                    assert!(
                        p == sketch_bits(&adjacent),
                        "{label}: right {i}/{j}: adjacent"
                    );
                    let rel = sketch_rel_diff(&planned, &fr2[j - 1]);
                    assert!(
                        rel < 1e-12,
                        "{label}: right pair {i} bond {j}: rel {rel:.3e}"
                    );
                    moved_r += (p != sketch_bits(&f0)) as usize;
                    max_rel = max_rel.max(rel);
                }
                for k in 1..=i {
                    let f0 = fl[k].clone();
                    let mut planned = f0.clone();
                    h.rotate_left_env_planned(&plan, x.core(k - 1), i, g, &mut planned);
                    let mut adjacent = f0.clone();
                    h.rotate_left_env_adjacent(x.core(k - 1), i, g, &mut adjacent);
                    let p = sketch_bits(&planned);
                    assert!(
                        p == sketch_bits(&adjacent),
                        "{label}: left {i}/{k}: adjacent"
                    );
                    let rel = sketch_rel_diff(&planned, &fl2[k]);
                    assert!(
                        rel < 1e-12,
                        "{label}: left pair {i} bond {k}: rel {rel:.3e}"
                    );
                    moved_l += (p != sketch_bits(&f0)) as usize;
                    max_rel = max_rel.max(rel);
                }
            }
            println!("{label}: gauge moved right {moved_r}, left {moved_l}; max rel {max_rel:.1e}");
            assert!(
                moved_r > 0 && moved_l > 0,
                "{label}: the gauge moved nothing on a stack"
            );
        }
    }
}
