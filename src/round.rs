// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Orthogonalization sweeps, rounding, boundary QR, norm-orthonormalization, and the real/imag
//! embedding.

use faer::Mat;
use faer::prelude::ReborrowMut;

use crate::core::factor::{Sketch, chop_energy, chop_global, mul_sketch_left, mul_sketch_right};
use crate::core::{Family, SparseCore};
use crate::ops::dot;
use crate::qn::{PerSector, Pooled, RankGrid, Sector};
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// What a Schmidt-floor trim ([`TtVector::sigma_floor_trim_mut`] and its SU(2) twin) removed.
/// `n_dropped == 0` means the point was not touched at all.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SigmaTrim {
    /// Singular values dropped, pooled over bonds and sectors (net of the keep-one fallback).
    pub n_dropped: usize,
    /// The largest dropped value (in the layer's threshold units).
    pub sigma_dropped_max: f64,
    /// Max over bonds of the dropped Schmidt mass `Σσ²` on that bond.
    pub dropped_mass_max: f64,
}

/// Rounding tolerance interpretation: `Abs` treats `ϵ` as an absolute error budget, `Rel`
/// pre-scales it by `‖tt‖`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RoundMode {
    /// Absolute error budget: `‖tt − round(tt)‖ ≤ ϵ`.
    Abs,
    /// Relative error budget: `‖tt − round(tt)‖ ≤ ϵ·‖tt‖`.
    Rel,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Re,
    Im,
}

fn scale_sketch<T: Scalar>(sk: &mut Sketch<T>, factor: f64) {
    let f = T::from_f64(factor);
    for s in sk.sectors() {
        if let Some(mut b) = sk.block_mut(s) {
            let (m, n) = (b.nrows(), b.ncols());
            for j in 0..n {
                for i in 0..m {
                    *b.rb_mut().get_mut(i, j) *= f;
                }
            }
        }
    }
}

impl<T: Scalar> TtVector<T> {
    /// Install a neighbor pair of cores with an explicit rank-grid copy + consistency asserts.
    pub fn set_cores(&mut self, first: SparseCore<T>, second: SparseCore<T>) {
        let k = first.k;
        debug_assert_eq!(second.k, k + 1, "set_cores: cores not adjacent");
        debug_assert_eq!(
            first.col_ranks, second.row_ranks,
            "set_cores: bond rank mismatch"
        );
        self.ranks[k + 1] = first.col_ranks.clone();
        self.cores[k] = first;
        self.cores[k + 1] = second;
    }

    /// Replace a single core in place, updating both incident bond rank grids.
    pub fn set_core(&mut self, core: SparseCore<T>) {
        let k = core.k;
        self.ranks[k] = core.row_ranks.clone();
        self.ranks[k + 1] = core.col_ranks.clone();
        self.cores[k] = core;
    }

    fn step_core_right(&mut self, k: usize, keep_rank: bool) {
        if keep_rank {
            let r = self.cores[k].qr_fixed();
            self.cores[k + 1].fold_sketch_left_mut(&r);
        } else {
            let (qk, c, _) = self.cores[k].qc();
            let new_next = mul_sketch_left(&c, &self.cores[k + 1]);
            self.set_cores(qk, new_next);
        }
        if self.ortho == Some(k) {
            self.ortho = Some(k + 1);
        }
    }

    fn step_core_left(&mut self, k: usize, keep_rank: bool) {
        if keep_rank {
            let l = self.cores[k].lq_fixed();
            self.cores[k - 1].fold_sketch_right_mut(&l);
        } else {
            let (qk, c, _) = self.cores[k].cq();
            let new_prev = mul_sketch_right(&self.cores[k - 1], &c);
            self.set_cores(new_prev, qk);
        }
        if self.ortho == Some(k) {
            self.ortho = Some(k - 1);
        }
    }

    /// Move the orthogonality center to `pos`. Already- orthogonal trains sweep from the current
    /// center; non-orthogonal trains canonicalize fully (left sweep, then right-to-left sweep).
    /// Reverse loop order is load-bearing.
    pub fn move_core(&mut self, pos: usize, keep_rank: bool) {
        let d = self.space.d;
        debug_assert!(pos < d, "move_core: center {pos} out of range 0..{d}");
        if let Some(cur) = self.ortho {
            if cur > pos {
                for k in ((pos + 1)..=cur).rev() {
                    self.step_core_left(k, keep_rank);
                }
            } else if cur < pos {
                for k in cur..pos {
                    self.step_core_right(k, keep_rank);
                }
            }
        } else {
            for k in 0..pos {
                self.step_core_right(k, keep_rank);
            }
            if d >= 2 {
                for k in ((pos + 1)..=(d - 1)).rev() {
                    self.step_core_left(k, keep_rank);
                }
            }
            self.ortho = Some(pos);
        }
    }

    /// Canonicalize left-to-right, center at the last core.
    pub fn left_orthogonalize(&mut self, keep_rank: bool) {
        self.move_core(self.space.d - 1, keep_rank);
    }

    /// Canonicalize right-to-left, center at the first core.
    pub fn right_orthogonalize(&mut self, keep_rank: bool) {
        self.move_core(0, keep_rank);
    }

    fn bond_rmax(&self, maxr: &[RankGrid], bond: usize, chi: Pooled) -> Pooled {
        if chi.is_none() {
            return Pooled::NONE;
        }
        let cap = self
            .space
            .bond_sectors(bond)
            .iter()
            .fold(0u64, |a, &s| a.saturating_add(maxr[bond][s] as u64));
        Pooled(usize::try_from((chi.0 as u64).min(cap)).unwrap_or(usize::MAX))
    }

    fn round_horizontal_sweep<F>(&mut self, mut decide: F)
    where
        F: FnMut(usize, &Sketch<T>) -> RankGrid,
    {
        let space = self.space;
        let d = space.d;
        for k in (1..=(d - 1)).rev() {
            let (mut u, sval, vt) = self.cores[k].svd_horizontal();
            let ranks = decide(k, &sval);
            u.rmul_sketch_mut(&sval);

            let mut xk = SparseCore::zeros(space, k, &ranks, &self.ranks[k + 1]);
            let urr = u.row_ranks();
            let mut c = Sketch::zeros(space, k, &urr, &ranks);
            for s in space.bond_sectors(k) {
                let rk = ranks[s];
                if rk > 0
                    && let Some(vtb) = vt[s].as_ref()
                {
                    xk.scatter_horizontal(s, vtb.as_ref().subrows(0, rk));
                }
                if let (Some(ub), Some(mut cb)) = (u.block(s), c.block_mut(s)) {
                    let rows = ub.nrows();
                    for j in 0..rk {
                        for i in 0..rows {
                            *cb.rb_mut().get_mut(i, j) = *ub.get(i, j);
                        }
                    }
                }
            }
            let new_prev = mul_sketch_right(&self.cores[k - 1], &c);
            self.set_cores(new_prev, xk);
        }
        self.ortho = Some(0);
    }

    /// Rounded copy. Right-canonical result (center `0`). `chi` is a POOLED cap on each bond's
    /// TOTAL dimension `Σ_q r[q]` ([`Pooled::NONE`] = uncapped).
    pub fn round(&self, eps: f64, chi: Pooled, mode: RoundMode) -> Self {
        let mut tt = self.clone();
        tt.round_mut(eps, chi, mode);
        tt
    }

    /// In-place round: left-orthogonalize, then a right-to-left `svd_horizontal` + energy-`chop`
    /// sweep with per-bond budget `ep`. `chi` is a POOLED cap on each bond's TOTAL dimension
    /// `Σ_q r[q]` ([`Pooled::NONE`] = uncapped).
    pub fn round_mut(&mut self, eps: f64, chi: Pooled, mode: RoundMode) {
        let d = self.space.d;
        self.left_orthogonalize(false);
        if d > 1 {
            let base = match mode {
                RoundMode::Rel => self.norm(),
                RoundMode::Abs => 1.0,
            };
            let ep = base / ((d - 1) as f64).sqrt() * eps;
            let chis: Vec<Pooled> = if chi.is_none() {
                vec![Pooled::NONE; d]
            } else {
                let maxr = crate::qn::max_ranks_open(&self.space, self.boundary_width());
                (0..d).map(|b| self.bond_rmax(&maxr, b, chi)).collect()
            };
            self.round_horizontal_sweep(|k, sval| chop_energy(sval, ep, chis[k]));
        }
        self.ortho = Some(0);
        debug_assert!(self.check());
    }

    /// Rounded copy via one global energy threshold.
    /// Result is NOT flagged orthogonal.
    /// `rmax` is a PER-SECTOR cap (`r[q] ≤ rmax` for every sector `q`), NOT a pooled one.
    pub fn round_global(&self, eps: f64, rmax: PerSector, mode: RoundMode) -> Self {
        let mut tt = self.clone();
        tt.round_global_mut(eps, rmax, mode);
        tt
    }

    /// In-place global round: `svdvals` + a pooled global energy chop + `reduce_ranks`, updating
    /// both `self.ranks` and the cores explicitly. `rmax` is a PER-SECTOR cap (`r[q] ≤ rmax` for
    /// every sector `q`) applied after the global chop, NOT a pooled one.
    pub fn round_global_mut(&mut self, eps: f64, rmax: PerSector, mode: RoundMode) {
        let d = self.space.d;
        let svals = self.svdvals();
        let base = match mode {
            RoundMode::Rel => self.norm(),
            RoundMode::Abs => 1.0,
        };
        if d > 1 {
            let ranks_g = chop_global(&svals, base * eps);
            let maxr = crate::qn::max_ranks_open(&self.space, self.boundary_width());
            for b in 1..=(d - 1) {
                let mut rk = ranks_g[b - 1].clone();
                for s in self.space.bond_sectors(b) {
                    let mr = maxr[b][s] as u64;
                    let cap = if !rmax.is_none() {
                        (rmax.0 as u64).min(mr)
                    } else {
                        mr
                    };
                    rk[s] = rk[s].min(usize::try_from(cap).unwrap_or(usize::MAX));
                }
                self.ranks[b] = rk;
            }
        }
        for k in 0..d {
            let nc = self.cores[k].reduce_ranks(&self.ranks[k], &self.ranks[k + 1]);
            self.set_core(nc);
        }
        self.ortho = None;
        debug_assert!(self.check());
    }

    /// Left-to-right SVD rounding sweep. Asserts right-canonical input (center `0`); produces a
    /// left-canonical result (center `d-1`).
    /// `chi` is a POOLED cap on each bond's TOTAL dimension `Σ_q r[q]` ([`Pooled::NONE`] =
    /// uncapped).
    pub fn round_lr_mut(&mut self, eps: f64, chi: Pooled, mode: RoundMode) {
        let d = self.space.d;
        assert_eq!(
            self.ortho,
            Some(0),
            "round_lr requires a right-canonical (center 0) train"
        );
        if d == 1 {
            return;
        }
        let space = self.space;
        let base = match mode {
            RoundMode::Rel => self.norm(),
            RoundMode::Abs => 1.0,
        };
        let ep = base / ((d - 1) as f64).sqrt() * eps;
        let chis: Vec<Pooled> = if chi.is_none() {
            vec![Pooled::NONE; d + 1]
        } else {
            let maxr = crate::qn::max_ranks_open(&self.space, self.boundary_width());
            (0..=d).map(|b| self.bond_rmax(&maxr, b, chi)).collect()
        };
        for k in 0..=(d - 2) {
            let (u, sval, mut vt) = self.cores[k].svd_vertical();
            let ranks = chop_energy(&sval, ep, chis[k + 1]);
            for r in space.bond_sectors(k + 1) {
                if let (Some(sb), Some(mut vtb)) = (sval.block(r), vt.block_mut(r)) {
                    let (m, n) = (vtb.nrows(), vtb.ncols());
                    for i in 0..m {
                        let sc = *sb.get(i, i);
                        for j in 0..n {
                            *vtb.rb_mut().get_mut(i, j) *= sc;
                        }
                    }
                }
            }
            let mut xk = SparseCore::zeros(space, k, &self.ranks[k], &ranks);
            let mut c = Sketch::zeros(space, k + 1, &ranks, &self.ranks[k + 1]);
            for r in space.bond_sectors(k + 1) {
                let rk = ranks[r];
                if rk > 0
                    && let Some(ub) = u[r].as_ref()
                {
                    xk.scatter_vertical(r, ub.as_ref().subcols(0, rk));
                }
                if let (Some(vtb), Some(mut cb)) = (vt.block(r), c.block_mut(r)) {
                    let cols = vtb.ncols();
                    for j in 0..cols {
                        for i in 0..rk {
                            *cb.rb_mut().get_mut(i, j) = *vtb.get(i, j);
                        }
                    }
                }
            }
            let new_next = mul_sketch_left(&c, &self.cores[k + 1]);
            self.set_cores(xk, new_next);
        }
        self.ortho = Some(d - 1);
        debug_assert!(self.check());
    }

    /// Fixed-rank round: truncate to the given per-bond (0-based) rank grids (`len == d+1`), each
    /// entry clamped against the available singular count.
    pub fn round_ranks_mut(&mut self, ranks: &[RankGrid]) {
        let d = self.space.d;
        assert_eq!(ranks.len(), d + 1, "round_ranks: need d+1 rank grids");
        debug_assert!(
            (0..=d).all(|b| ranks[b]
                .iter()
                .all(|(s, &v)| v == 0 || self.space.contains(b, s))),
            "round_ranks: positive requested ranks must be on admissible sectors"
        );
        self.left_orthogonalize(false);
        if d == 1 {
            self.ortho = Some(0);
            return;
        }
        let targets: Vec<RankGrid> = ranks.to_vec();
        self.round_horizontal_sweep(|k, sval| {
            let avail = sval.row_ranks();
            let mut r = targets[k].clone();
            for (s, &v) in avail.iter() {
                r[s] = r[s].min(v);
            }
            r
        });
        debug_assert!(self.check());
    }

    /// Per-value Schmidt-floor trim of the POINT: on every interior bond drop the singular values
    /// `σ ≤ floor` (per sector; the survivors are the leading ones, which is what
    /// [`Self::round_ranks_mut`] keeps) and re-round to the surviving counts. This moves a
    /// manifold-BORDER point onto the lower-rank manifold where it is interior — the mitigation
    /// `hess.rs` ranks first ("trim the border ranks — moves the POINT off the border rather than
    /// altering the operator") and the same rule `cone_restored`'s σ-floor trim applies. (A GAD
    /// stop test built on it was measured a no-op and removed; the primitive stays.)
    pub fn sigma_floor_trim_mut(&mut self, floor: f64) -> SigmaTrim {
        let d = self.space.d;
        if d < 2 || !floor.is_finite() || floor <= 0.0 {
            return SigmaTrim::default();
        }
        let svals = self.clone().svdvals();
        let mut targets: Vec<RankGrid> = self.ranks.clone();
        let mut rep = SigmaTrim::default();
        let mut changed = false;
        for b in 1..d {
            let sv = &svals[b - 1];
            let mut grid = RankGrid::zeros(&self.space);
            let mut dropped_sq = 0.0f64;
            let mut kept = 0usize;
            let mut best: Option<(Sector, f64)> = None;
            for s in self.space.bond_sectors(b) {
                let Some(blk) = sv.block(s) else { continue };
                let len = blk.nrows().min(blk.ncols());
                if len == 0 {
                    continue;
                }
                let lead = blk.get(0, 0).re_s();
                if best.is_none_or(|(_, v)| v < lead) {
                    best = Some((s, lead));
                }
                let mut keep = 0usize;
                for i in 0..len {
                    let v = blk.get(i, i).re_s();
                    if v > floor {
                        keep += 1;
                    } else {
                        dropped_sq += v * v;
                        rep.n_dropped += 1;
                        rep.sigma_dropped_max = rep.sigma_dropped_max.max(v);
                    }
                }
                grid[s] = keep;
                kept += keep;
            }
            if kept == 0
                && let Some((s0, v0)) = best
            {
                grid[s0] = 1;
                dropped_sq -= v0 * v0;
                rep.n_dropped -= 1;
            }
            if grid != targets[b] {
                changed = true;
            }
            rep.dropped_mass_max = rep.dropped_mass_max.max(dropped_sq.max(0.0));
            targets[b] = grid;
        }
        if changed {
            self.round_ranks_mut(&targets);
        } else {
            rep.n_dropped = 0;
        }
        rep
    }

    /// Bond singular values: left-orthogonalize, then a right-to- left `svd_horizontal_to_core`
    /// sweep. Returns the `d-1` interior-bond diagonal sketches (index `i` is bond `i+1`). Leaves
    /// `self` right-canonical (center `0`).
    pub fn svdvals(&mut self) -> Vec<Sketch<T>> {
        let d = self.space.d;
        self.left_orthogonalize(false);
        if d == 1 {
            return Vec::new();
        }
        let mut svals: Vec<Option<Sketch<T>>> = (0..d - 1).map(|_| None).collect();
        for k in (1..=(d - 1)).rev() {
            let (mut u, s, ck) = self.cores[k].svd_horizontal_to_core();
            u.rmul_sketch_mut(&s);
            svals[k - 1] = Some(s);
            let new_prev = mul_sketch_right(&self.cores[k - 1], &u);
            self.set_cores(new_prev, ck);
        }
        self.ortho = Some(0);
        svals.into_iter().map(|s| s.expect("bond sketch")).collect()
    }

    /// Left-canonicalize (keeping ranks) and return the `1×1` boundary factor `R`: `Array(y)·R[0,0]
    /// ≈ x`.
    pub fn qr_lr(&self) -> (Self, Mat<T>) {
        let mut tt = self.clone();
        tt.left_orthogonalize(true);
        let d = tt.space.d;
        let r = tt.cores[d - 1].qr_fixed();
        let last = Sector::with_irrep(tt.space.nup, tt.space.ndn, tt.space.target);
        let rblock = r.block(last).expect("boundary R block").to_owned();
        debug_assert!(tt.check());
        (tt, rblock)
    }

    /// Right-canonicalize (keeping ranks) and return the `1×1` boundary factor `L`:
    /// `L[0,0]·Array(y) ≈ x`.
    pub fn qr_rl(&self) -> (Self, Mat<T>) {
        let mut tt = self.clone();
        tt.right_orthogonalize(true);
        let l = tt.cores[0].lq_fixed();
        let first = Sector::new(0, 0);
        let lblock = l.block(first).expect("boundary L block").to_owned();
        debug_assert!(tt.check());
        (tt, lblock)
    }

    fn step_core_right_norm(&mut self, k: usize, keep_rank: bool) -> f64 {
        let nrm;
        if keep_rank {
            let mut r = self.cores[k].qr_fixed();
            nrm = r.norm();
            if nrm != 0.0 {
                scale_sketch(&mut r, 1.0 / nrm);
            }
            self.cores[k + 1].fold_sketch_left_mut(&r);
        } else {
            let (qk, mut c, _) = self.cores[k].qc();
            nrm = c.norm();
            if nrm != 0.0 {
                scale_sketch(&mut c, 1.0 / nrm);
            }
            let new_next = mul_sketch_left(&c, &self.cores[k + 1]);
            self.set_cores(qk, new_next);
        }
        if self.ortho == Some(k) {
            self.ortho = Some(k + 1);
        }
        nrm
    }

    fn step_core_left_norm(&mut self, k: usize, keep_rank: bool) -> f64 {
        let nrm;
        if keep_rank {
            let mut l = self.cores[k].lq_fixed();
            nrm = l.norm();
            if nrm != 0.0 {
                scale_sketch(&mut l, 1.0 / nrm);
            }
            self.cores[k - 1].fold_sketch_right_mut(&l);
        } else {
            let (qk, mut c, _) = self.cores[k].cq();
            nrm = c.norm();
            if nrm != 0.0 {
                scale_sketch(&mut c, 1.0 / nrm);
            }
            let new_prev = mul_sketch_right(&self.cores[k - 1], &c);
            self.set_cores(new_prev, qk);
        }
        if self.ortho == Some(k) {
            self.ortho = Some(k - 1);
        }
        nrm
    }

    /// Left-orthonormalize in place, returning the per-step factor norms (`len == d`); their
    /// product is `‖self‖`. The terminal-core division guards the terminal core against a zero
    /// norm.
    pub fn left_orthonormalize(&mut self, keep_rank: bool) -> Vec<f64> {
        let d = self.space.d;
        let mut nrm = vec![1.0f64; d];
        if d == 1 {
            let n = self.cores[0].norm();
            nrm[0] = n;
            if n != 0.0 {
                self.cores[0].scale_mut(T::from_f64(1.0 / n));
            }
            self.ortho = Some(0);
            return nrm;
        }
        let start = self.ortho.unwrap_or(0);
        #[allow(clippy::needless_range_loop)]
        for k in start..=(d - 2) {
            nrm[k] = self.step_core_right_norm(k, keep_rank);
        }
        let n = self.cores[d - 1].norm();
        nrm[d - 1] = n;
        if n != 0.0 {
            self.cores[d - 1].scale_mut(T::from_f64(1.0 / n));
        }
        self.ortho = Some(d - 1);
        nrm
    }

    fn right_orthonormalize(&mut self, keep_rank: bool) -> Vec<f64> {
        let d = self.space.d;
        let mut nrm = vec![1.0f64; d];
        if d == 1 {
            let n = self.cores[0].norm();
            nrm[0] = n;
            if n != 0.0 {
                self.cores[0].scale_mut(T::from_f64(1.0 / n));
            }
            self.ortho = Some(0);
            return nrm;
        }
        let start = self.ortho.unwrap_or(d - 1);
        for k in (1..=start).rev() {
            nrm[k] = self.step_core_left_norm(k, keep_rank);
        }
        let n = self.cores[0].norm();
        nrm[0] = n;
        if n != 0.0 {
            self.cores[0].scale_mut(T::from_f64(1.0 / n));
        }
        self.ortho = Some(0);
        nrm
    }

    /// Norm via a left-to-right orthonormalization sweep = `exp(Σ ln nrm)`. CLONES internally
    /// (value semantics). `keep_rank` selects `qr_fixed` over `qc`.
    pub fn norm_lr(&self, keep_rank: bool) -> f64 {
        if let Some(c) = self.ortho {
            return self.cores[c].norm();
        }
        let mut tt = self.clone();
        let nrm = tt.left_orthonormalize(keep_rank);
        nrm.iter().map(|&x| x.ln()).sum::<f64>().exp()
    }

    /// Norm via a right-to-left orthonormalization sweep. CLONES internally.
    pub fn norm_rl(&self, keep_rank: bool) -> f64 {
        let mut tt = self.clone();
        let nrm = tt.right_orthonormalize(keep_rank);
        nrm.iter().map(|&x| x.ln()).sum::<f64>().exp()
    }

    fn real_embedding(&self, which: Part) -> Self {
        let space = self.space;
        let d = space.d;
        let mut cores = Vec::with_capacity(d);
        for k in 0..d {
            let row_mul = if k == 0 { 1 } else { 2 };
            let col_mul = if k == d - 1 { 1 } else { 2 };
            let mut rr = RankGrid::zeros(&space);
            let mut cr = RankGrid::zeros(&space);
            for s in space.bond_sectors(k) {
                rr[s] = row_mul * self.ranks[k][s];
            }
            for s in space.bond_sectors(k + 1) {
                cr[s] = col_mul * self.ranks[k + 1][s];
            }
            let mut core = SparseCore::zeros(space, k, &rr, &cr);
            let src = &self.cores[k];
            for f in Family::ALL {
                for s in src.transitions(f).collect::<Vec<_>>() {
                    let b = src.block(f, s).expect("transition present");
                    let (m, n) = (b.nrows(), b.ncols());
                    let mut dst = core.block_mut(f, s).expect("transition present");
                    for j in 0..n {
                        for i in 0..m {
                            let z = *b.get(i, j);
                            let (re, im) = (z.re_s(), z.im_s());
                            if k == 0 {
                                *dst.rb_mut().get_mut(i, j) = T::from_f64(re);
                                *dst.rb_mut().get_mut(i, n + j) = T::from_f64(im);
                            } else if k == d - 1 {
                                let (top, bot) = match which {
                                    Part::Re => (re, -im),
                                    Part::Im => (im, re),
                                };
                                *dst.rb_mut().get_mut(i, j) = T::from_f64(top);
                                *dst.rb_mut().get_mut(m + i, j) = T::from_f64(bot);
                            } else {
                                *dst.rb_mut().get_mut(i, j) = T::from_f64(re);
                                *dst.rb_mut().get_mut(i, n + j) = T::from_f64(im);
                                *dst.rb_mut().get_mut(m + i, j) = T::from_f64(-im);
                                *dst.rb_mut().get_mut(m + i, n + j) = T::from_f64(re);
                            }
                        }
                    }
                }
            }
            cores.push(core);
        }
        TtVector::new(space, cores)
    }

    fn real_imag_d1(&self, which: Part) -> Self {
        let space = self.space;
        let src = &self.cores[0];
        let mut core = SparseCore::zeros(space, 0, &self.ranks[0], &self.ranks[1]);
        for f in Family::ALL {
            for s in src.transitions(f).collect::<Vec<_>>() {
                let b = src.block(f, s).expect("transition present");
                let (m, n) = (b.nrows(), b.ncols());
                let mut dst = core.block_mut(f, s).expect("transition present");
                for j in 0..n {
                    for i in 0..m {
                        let z = *b.get(i, j);
                        let v = match which {
                            Part::Re => z.re_s(),
                            Part::Im => z.im_s(),
                        };
                        *dst.rb_mut().get_mut(i, j) = T::from_f64(v);
                    }
                }
            }
        }
        TtVector::new(space, vec![core])
    }

    /// Real part of a complex train, same output type `T`, interior bond ranks doubled.
    pub fn real_part(&self) -> Self {
        if self.space.d == 1 {
            self.real_imag_d1(Part::Re)
        } else {
            self.real_embedding(Part::Re)
        }
    }

    /// Imaginary part.
    pub fn imag_part(&self) -> Self {
        if self.space.d == 1 {
            self.real_imag_d1(Part::Im)
        } else {
            self.real_embedding(Part::Im)
        }
    }

    /// Both parts at once.
    pub fn realimag(&self) -> (Self, Self) {
        (self.real_part(), self.imag_part())
    }
}

/// Inner product with a pre-orthogonalization pass: right-orthogonalizes both operands, then
/// calls [`dot`]. CLONES internally (value semantics), so neither argument is mutated.
pub fn dot_orth<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>) -> T {
    let mut xc = x.clone();
    let mut yc = y.clone();
    xc.right_orthogonalize(false);
    yc.right_orthogonalize(false);
    dot(&xc, &yc)
}

/// Deterministic binary-tree weighted sum-then-round (`solvers::lanczos`'s Krylov-vector
/// recombination), composed entirely from `Mul<T>`/`Add`/`round`.
pub fn round_sum<T: Scalar>(alpha: &[T], summands: &[TtVector<T>], tol: f64) -> TtVector<T> {
    assert_eq!(
        alpha.len(),
        summands.len(),
        "round_sum: alpha/summands length mismatch"
    );
    assert!(!summands.is_empty(), "round_sum: need at least one summand");
    let n = summands.len();
    if n == 1 {
        let scaled = &summands[0] * alpha[0];
        scaled.round(tol, Pooled::NONE, RoundMode::Abs)
    } else if n == 2 {
        let x = &summands[0] * alpha[0];
        let y = &summands[1] * alpha[1];
        (&x + &y).round(tol, Pooled::NONE, RoundMode::Abs)
    } else {
        let mid = n / 2;
        let x = round_sum(&alpha[..mid], &summands[..mid], tol / 3.0);
        let y = round_sum(&alpha[mid..], &summands[mid..], tol / 3.0);
        (&x + &y).round(tol / 3.0, Pooled::NONE, RoundMode::Abs)
    }
}

#[cfg(test)]
mod pg3_tests {
    use crate::round::RoundMode;
    use crate::scalar::Scalar;
    use crate::test_util::{
        assert_pure, core_left_orthonormality_dev, core_right_orthonormality_dev, find_determinant,
        naive_ranks, pg3_space_d4, pg3_space_d6, pg3_space_d8, rel_err, rel_lr_c,
        sum_interior_rank,
    };
    use crate::{C64, TtVector, tt_randn, tt_state_in};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn pg3_round_relative_error_bound() {
        let (pg, space) = pg3_space_d8();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_010);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let xd = x.to_dense();

        let y = x.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
        assert!(y.check());

        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "‖x − round(x)‖ ≤ ε·‖x‖"
        );
        assert_eq!(y.ortho, Some(0), "round is right-canonical");
        for b in 0..=space.d {
            for (s, &ry) in y.ranks[b].iter() {
                assert!(ry <= x.ranks[b][s], "rank(y) ≤ rank(x) at bond {b} {s:?}");
            }
        }
        assert!(
            sum_interior_rank(&y) < sum_interior_rank(&x),
            "vacuous: round did not truncate anything"
        );
        assert_pure(&pg, &space, &y);
    }

    #[test]
    fn pg3_round_global_error_bound() {
        let (pg, space) = pg3_space_d6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_011);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let xd = x.to_dense();

        let y = x.round_global(1e-8, crate::qn::PerSector::NONE, RoundMode::Rel);
        assert!(y.check());

        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "global round error bound"
        );
        assert_eq!(y.ortho, None, "global round is not flagged orthogonal");
        assert_pure(&pg, &space, &y);
    }

    #[test]
    fn pg3_round_zero_vector_is_exactly_zero() {
        let (pg, space) = pg3_space_d6();
        let (occ_up, occ_dn) = find_determinant(&pg, space.d, space.nup, space.ndn, space.target);
        let x: TtVector<f64> = tt_state_in(space, &occ_up, &occ_dn);
        assert!(x.check());
        let neg = (-1.0) * &x;
        let z = &x + &neg;
        assert!(z.check());
        assert!(z.norm() < 1e-14, "z is (numerically) the zero vector");

        assert_eq!(
            z.round(1e-14, crate::qn::Pooled::NONE, RoundMode::Abs)
                .norm(),
            0.0,
            "default"
        );
        assert_eq!(
            z.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel)
                .norm(),
            0.0,
            ":rel"
        );
        assert_eq!(
            z.round_global(1e-8, crate::qn::PerSector::NONE, RoundMode::Abs)
                .norm(),
            0.0,
            "Global"
        );
        assert_eq!(
            z.round_global(1e-8, crate::qn::PerSector::NONE, RoundMode::Rel)
                .norm(),
            0.0,
            "Global + :rel"
        );
    }

    #[test]
    fn pg3_round_lr_forward_sweep_error_bound() {
        let (pg, space) = pg3_space_d8();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_013);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let xd = x.to_dense();
        let d = space.d;

        let mut y = x.clone();
        y.right_orthogonalize(false);
        y.round_lr_mut(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
        assert!(y.check());

        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "‖x − round_lr(x)‖ ≤ ε·‖x‖"
        );
        assert_eq!(y.ortho, Some(d - 1), "round_lr is left-canonical");
        let (dev, checked) = core_left_orthonormality_dev(&y);
        assert!(
            dev < 1e-10,
            "cores 0..=d-2 left-orthonormal after round_lr, dev={dev}"
        );
        assert!(checked > 0);
        assert_pure(&pg, &space, &y);
    }

    #[test]
    fn pg3_round_error_bound_c64() {
        let (pg, space) = pg3_space_d6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_303_017);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<C64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let xd = x.to_dense();

        let y = x.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
        assert!(y.check());
        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "complex round error bound"
        );
        assert_eq!(y.ortho, Some(0));
        assert_pure(&pg, &space, &y);
    }

    #[test]
    fn pg3_round_is_deterministic() {
        let (_pg, space) = pg3_space_d6();
        let build = || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_018);
            let ranks = naive_ranks(&space, 0, 3, &mut rng);
            let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
            x.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel)
        };
        let a = build();
        let b = build();
        assert_eq!(a.to_dense(), b.to_dense(), "round is deterministic");
    }

    #[test]
    fn pg3_left_orthogonalize_gauge_and_purity() {
        let (pg, space) = pg3_space_d8();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_001);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let xd = x.to_dense();

        let mut y = x.clone();
        y.left_orthogonalize(false);
        assert!(y.check());

        assert!(rel_err(&y.to_dense(), &xd) < 1e-13, "gauge dense equality");
        assert_eq!(y.ortho, Some(space.d - 1), "center at last core");
        let (dev, checked) = core_left_orthonormality_dev(&y);
        assert!(dev < 1e-12, "cores 0..=d-2 left-orthonormal, dev={dev}");
        assert!(
            checked > 0,
            "vacuous: no positive-rank column sector checked"
        );
        assert_pure(&pg, &space, &y);
    }

    #[test]
    fn pg3_right_orthogonalize_gauge_and_purity() {
        let (pg, space) = pg3_space_d8();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_003);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let xd = x.to_dense();

        let mut y = x.clone();
        y.right_orthogonalize(false);
        assert!(y.check());

        assert!(rel_err(&y.to_dense(), &xd) < 1e-13, "gauge dense equality");
        assert_eq!(y.ortho, Some(0), "center at first core");
        let (dev, checked) = core_right_orthonormality_dev(&y);
        assert!(dev < 1e-12, "cores 1..=d-1 right-orthonormal, dev={dev}");
        assert!(checked > 0, "vacuous: no positive-rank row sector checked");
        assert_pure(&pg, &space, &y);
    }

    #[test]
    fn pg3_left_orthogonalize_gauge_c64() {
        let (pg, space) = pg3_space_d6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_303_005);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<C64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let xd = x.to_dense();

        let mut y = x.clone();
        y.left_orthogonalize(false);
        assert!(y.check());

        assert!(
            rel_err(&y.to_dense(), &xd) < 1e-13,
            "complex gauge equality"
        );
        assert_eq!(y.ortho, Some(space.d - 1));
        let (dev, checked) = core_left_orthonormality_dev(&y);
        assert!(dev < 1e-12, "complex left-orthonormal (QᴴQ=I), dev={dev}");
        assert!(checked > 0);
        assert_pure(&pg, &space, &y);
    }

    #[test]
    fn pg3_real_imag_embedding_purity() {
        let (pg, space) = pg3_space_d6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_303_033);
        let ranks = naive_ranks(&space, 0, 2, &mut rng);
        let x: TtVector<C64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());

        let re = x.real_part();
        let im = x.imag_part();
        assert!(re.check(), "real_part boundary law");
        assert!(im.check(), "imag_part boundary law");
        let recomb = &re + &(C64::i() * &im);
        assert!(rel_lr_c(&recomb, &x) < 1e-10, "real(x) + i·imag(x) ≈ x");

        let (re2, im2) = x.realimag();
        assert!(re2.check());
        assert!(im2.check());
        let recomb2 = &re2 + &(C64::i() * &im2);
        assert!(rel_lr_c(&recomb2, &x) < 1e-10, "recombine(realimag(x)) ≈ x");

        let im_of_re = x.real_part().imag_part();
        assert!(
            im_of_re.norm_lr(false) / x.norm_lr(false) < 1e-12,
            "imag(real(x)) ≈ 0"
        );

        for part in [&re, &im, &re2, &im2] {
            let dense = part.to_dense();
            for v in &dense {
                assert!(
                    v.im_s().abs() < 1e-10,
                    "real/imag output must be real-valued"
                );
            }
            assert_pure(&pg, &space, part);
        }
    }

    #[test]
    fn pg3_norm_lr_is_real_positive_c64() {
        let (_pg, space) = pg3_space_d4();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_303_034);
        let ranks = naive_ranks(&space, 1, 2, &mut rng);
        let x: TtVector<C64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let nx = x.norm_lr(false);
        assert!(nx.is_finite() && nx > 0.0, "norm(:LR) real positive: {nx}");
    }
}

#[cfg(test)]
mod phase3_tests {
    use crate::qn::{QnSpace, RankGrid};
    use crate::round::RoundMode;
    use crate::test_util::{
        core_left_orthonormality_dev, core_right_orthonormality_dev, dense_dot, randomized_state,
        rel_err, rel_lr_c, sum_interior_rank,
    };
    use crate::{C64, TtVector, dot_orth, tt_state};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn complex_state(
        space: QnSpace,
        lo: usize,
        hi: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> TtVector<C64> {
        let mut x = randomized_state::<C64, _>(space, lo, hi, rng);
        x.round_mut(1e-14, crate::qn::Pooled::NONE, RoundMode::Abs);
        x
    }

    #[test]
    fn left_orthogonalize_is_exact_gauge() {
        let space = QnSpace::from_n_two_sz(8, 3, -1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_001);
        let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();

        let mut y = x.clone();
        y.left_orthogonalize(false);
        y.check();

        assert!(rel_err(&y.to_dense(), &xd) < 1e-13, "gauge dense equality");
        assert_eq!(y.ortho, Some(space.d - 1), "center at last core");
        assert!(
            core_left_orthonormality_dev(&y).0 < 1e-12,
            "cores 0..=d-2 left-orthonormal"
        );
    }

    #[test]
    fn left_orthogonalize_keeprank_conserves_ranks() {
        let space = QnSpace::from_n_two_sz(8, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_002);
        let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();
        let n0 = x.norm();
        let r0: Vec<RankGrid> = x.ranks.clone();

        let mut y = x.clone();
        y.left_orthogonalize(true);
        y.check();

        assert!(
            rel_err(&y.to_dense(), &xd) < 1e-13,
            "keeprank gauge equality"
        );
        assert!(
            (y.norm() - n0).abs() <= 1e-10 * n0,
            "ortho-fast-path norm exact after keeprank left-orthogonalize"
        );
        for (b, (yr, r0r)) in y.ranks.iter().zip(&r0).enumerate() {
            assert_eq!(yr, r0r, "bond {b} ranks conserved");
        }
    }

    #[test]
    fn right_orthogonalize_is_exact_gauge() {
        let space = QnSpace::from_n_two_sz(8, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_003);
        let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();

        let mut y = x.clone();
        y.right_orthogonalize(false);
        y.check();

        assert!(rel_err(&y.to_dense(), &xd) < 1e-13, "gauge dense equality");
        assert_eq!(y.ortho, Some(0), "center at first core");
        assert!(
            core_right_orthonormality_dev(&y).0 < 1e-12,
            "cores 1..=d-1 right-orthonormal"
        );
    }

    #[test]
    fn right_orthogonalize_keeprank_conserves_ranks() {
        let space = QnSpace::from_n_two_sz(8, 4, -2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_004);
        let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();
        let n0 = x.norm();
        let r0: Vec<RankGrid> = x.ranks.clone();

        let mut y = x.clone();
        y.right_orthogonalize(true);
        y.check();

        assert!(
            rel_err(&y.to_dense(), &xd) < 1e-13,
            "keeprank gauge equality"
        );
        assert!(
            (y.norm() - n0).abs() <= 1e-10 * n0,
            "ortho-fast-path norm exact after keeprank right-orthogonalize"
        );
        for (b, (yr, r0r)) in y.ranks.iter().zip(&r0).enumerate() {
            assert_eq!(yr, r0r, "bond {b} ranks conserved");
        }
    }

    #[test]
    fn round_relative_error_bound() {
        let space = QnSpace::from_n_two_sz(8, 7, -1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_005);
        let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();

        let y = x.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
        y.check();

        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "‖x − round(x)‖ ≤ ε·‖x‖"
        );
        assert_eq!(y.ortho, Some(0), "round is right-canonical (center 0)");
        for b in 0..=space.d {
            for (s, &ry) in y.ranks[b].iter() {
                assert!(ry <= x.ranks[b][s], "rank(y) ≤ rank(x) at bond {b} {s:?}");
            }
        }
    }

    #[test]
    fn round_global_error_bound() {
        let space = QnSpace::from_n_two_sz(8, 6, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_006);
        let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();

        let y = x.round_global(1e-8, crate::qn::PerSector::NONE, RoundMode::Rel);
        y.check();

        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "global round error bound"
        );
        assert_eq!(y.ortho, None, "global round is not flagged orthogonal");
    }

    #[test]
    fn round_zero_vector_is_exactly_zero() {
        let occ_up = [false, true, false, false, true, false, true, false];
        let occ_dn = [false; 8];
        let x = tt_state::<f64>(&occ_up, &occ_dn);
        let neg = (-1.0) * &x;
        let z = &x + &neg;
        z.check();
        assert!(z.norm() < 1e-14, "z is (numerically) the zero vector");

        assert_eq!(
            z.round(1e-14, crate::qn::Pooled::NONE, RoundMode::Abs)
                .norm(),
            0.0,
            "default"
        );
        assert_eq!(
            z.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel)
                .norm(),
            0.0,
            ":rel"
        );
        assert_eq!(
            z.round_global(1e-8, crate::qn::PerSector::NONE, RoundMode::Abs)
                .norm(),
            0.0,
            "Global"
        );
        assert_eq!(
            z.round_global(1e-8, crate::qn::PerSector::NONE, RoundMode::Rel)
                .norm(),
            0.0,
            "Global + :rel"
        );
    }

    #[test]
    fn dot_orthogonalize_matches_dense() {
        let space = QnSpace::from_n_two_sz(8, 5, -1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(104_009);
        let x: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        let y: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

        let got = dot_orth(&x, &y);
        let want = dense_dot(&x.to_dense(), &y.to_dense());
        assert!(
            (got - want).abs() <= 1e-8 * want.abs(),
            "dot_orth {got} vs dense {want}"
        );
    }

    #[test]
    fn norm_lr_rl_keeprank_matches_dot_norm() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(301_011);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let n0 = x.norm();

        assert!(
            (x.norm_rl(true) - n0).abs() <= 1e-10 * n0,
            "norm_rl(keep_rank)"
        );
        assert!(
            (x.norm_lr(true) - n0).abs() <= 1e-10 * n0,
            "norm_lr(keep_rank)"
        );
        assert!(
            (x.norm_lr(false) - n0).abs() <= 1e-10 * n0,
            "norm(:LR, revealing)"
        );
        assert!(
            (x.norm_rl(false) - n0).abs() <= 1e-10 * n0,
            "norm(:RL, revealing)"
        );
    }

    #[test]
    fn move_core_roundtrip_positions() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_014);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let xd = x.to_dense();
        let d = space.d;

        for pos in 0..d {
            let mut y = x.clone();
            y.move_core(pos, false);
            y.check();
            assert_eq!(y.ortho, Some(pos), "center at {pos}");
            assert!(
                rel_err(&y.to_dense(), &xd) < 1e-12,
                "gauge invariance at center {pos}"
            );
        }

        let mut y = x.clone();
        for pos in 0..d {
            y.move_core(pos, false);
            assert!(rel_err(&y.to_dense(), &xd) < 1e-12, "up-sweep to {pos}");
        }
        for pos in (0..d).rev() {
            y.move_core(pos, false);
            assert!(rel_err(&y.to_dense(), &xd) < 1e-12, "down-sweep to {pos}");
        }
    }

    #[test]
    fn left_orthogonalize_is_exact_gauge_c64() {
        let space = QnSpace::from_n_two_sz(8, 3, -1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_015);
        let x: TtVector<C64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();

        let mut y = x.clone();
        y.left_orthogonalize(false);
        y.check();

        assert!(
            rel_err(&y.to_dense(), &xd) < 1e-13,
            "complex gauge equality"
        );
        assert_eq!(y.ortho, Some(space.d - 1));
        assert!(
            core_left_orthonormality_dev(&y).0 < 1e-12,
            "complex left-orthonormal (QᴴQ = I)"
        );
    }

    #[test]
    fn right_orthogonalize_keeprank_c64() {
        let space = QnSpace::from_n_two_sz(8, 4, -2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_016);
        let x: TtVector<C64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();
        let n0 = x.norm();
        let r0: Vec<RankGrid> = x.ranks.clone();

        let mut y = x.clone();
        y.right_orthogonalize(true);
        y.check();

        assert!(
            rel_err(&y.to_dense(), &xd) < 1e-13,
            "complex keeprank equality"
        );
        assert!(
            (y.norm() - n0).abs() <= 1e-10 * n0,
            "ortho-fast-path norm exact after complex keeprank right-orthogonalize"
        );
        for (b, (yr, r0r)) in y.ranks.iter().zip(&r0).enumerate() {
            assert_eq!(yr, r0r, "bond {b} ranks conserved");
        }
    }

    #[test]
    fn round_roundtrip_c64() {
        let space = QnSpace::new(5, 1, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_017);
        let x = complex_state(space, 2, 4, &mut rng);

        let mut xr = x.clone();
        xr.round_mut(1e-14, crate::qn::Pooled::NONE, RoundMode::Abs);
        assert!(rel_lr_c(&xr, &x) < 1e-10, "round is idempotent up to ε");
    }

    #[test]
    fn norm_lr_is_real_c64() {
        let space = QnSpace::new(6, 1, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_018);
        let x = complex_state(space, 2, 4, &mut rng);

        let nx = x.norm_lr(false);
        assert!(nx.is_finite() && nx > 0.0, "norm(:LR) real positive: {nx}");
    }

    #[test]
    fn conj_real_imag_recombine_c64() {
        let space = QnSpace::new(5, 1, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_019);
        let x = complex_state(space, 2, 4, &mut rng);

        let cc = x.conj().conj();
        assert!(rel_lr_c(&cc, &x) < 1e-12, "conj(conj(x)) ≈ x");

        let re = x.real_part();
        let im = x.imag_part();
        let recomb = &re + &(C64::i() * &im);
        assert!(rel_lr_c(&recomb, &x) < 1e-10, "real(x) + i·imag(x) ≈ x");

        let (re2, im2) = x.realimag();
        let recomb2 = &re2 + &(C64::i() * &im2);
        assert!(rel_lr_c(&recomb2, &x) < 1e-10, "recombine(realimag(x)) ≈ x");

        let im_of_re = x.real_part().imag_part();
        assert!(
            im_of_re.norm_lr(false) / x.norm_lr(false) < 1e-12,
            "imag(real(x)) ≈ 0"
        );
    }

    #[test]
    fn round_error_bound_c64() {
        let space = QnSpace::from_n_two_sz(8, 7, -1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_020);
        let x: TtVector<C64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();

        let y = x.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
        y.check();
        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "complex round error bound"
        );
        assert_eq!(y.ortho, Some(0));
    }

    #[test]
    fn round_is_deterministic() {
        let space = QnSpace::from_n_two_sz(8, 4, 0);
        let build = || {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_021);
            let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
            x.round(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel)
        };
        let a = build();
        let b = build();
        assert_eq!(a.to_dense(), b.to_dense(), "round is deterministic");
    }

    #[test]
    fn round_lr_forward_sweep_error_bound() {
        let space = QnSpace::from_n_two_sz(8, 7, -1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_032);
        let x: TtVector<f64> = randomized_state(space, 0, 6, &mut rng);
        let xd = x.to_dense();
        let d = space.d;

        let mut y = x.clone();
        y.right_orthogonalize(false);
        y.round_lr_mut(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
        y.check();

        assert!(
            rel_err(&y.to_dense(), &xd) <= 1e-8 * (1.0 + 1e-6),
            "‖x − round_lr(x)‖ ≤ ε·‖x‖"
        );
        assert_eq!(
            y.ortho,
            Some(d - 1),
            "round_lr is left-canonical (center d-1)"
        );
        assert!(
            core_left_orthonormality_dev(&y).0 < 1e-10,
            "cores 0..=d-2 left-orthonormal after round_lr"
        );
    }

    #[test]
    fn round_ranks_clamps_and_truncates() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_030);
        let x: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
        let xd = x.to_dense();
        let d = space.d;

        let over: Vec<RankGrid> = (0..=d)
            .map(|b| {
                let mut r = RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    r[s] = if b == 0 || b == d { 1 } else { 1_000 };
                }
                r
            })
            .collect();
        let mut y_full = x.clone();
        y_full.round_ranks_mut(&over);
        y_full.check();
        assert_eq!(y_full.ortho, Some(0), "round_ranks is right-canonical");
        assert!(
            rel_err(&y_full.to_dense(), &xd) < 1e-12,
            "over-request keeps all singular values → exact reconstruction (clamp, no OOB)"
        );

        let top = d - 1;
        let mut t = over.clone();
        for (j, s) in space.bond_sectors(top).into_iter().enumerate() {
            t[top][s] = if j % 2 == 0 { 1 } else { 1_000 };
        }
        let mut y = x.clone();
        y.round_ranks_mut(&t);
        y.check();
        for s in space.bond_sectors(top) {
            let expected = t[top][s].min(y_full.ranks[top][s]);
            assert_eq!(
                y.ranks[top][s], expected,
                "bond {top} {s:?}: rank == min(target, available)"
            );
        }

        let ones: Vec<RankGrid> = (0..=d)
            .map(|b| {
                let mut r = RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    r[s] = 1;
                }
                r
            })
            .collect();
        let mut z = x.clone();
        z.round_ranks_mut(&ones);
        z.check();
        assert_eq!(z.ortho, Some(0));
        for b in 1..d {
            for s in space.bond_sectors(b) {
                assert!(z.ranks[b][s] <= 1, "bond {b} {s:?} truncated to ≤ 1");
            }
        }
        assert!(
            sum_interior_rank(&z) < sum_interior_rank(&y_full),
            "per-sector rank-1 target genuinely reduced the total rank"
        );
        assert!(
            rel_err(&z.to_dense(), &xd).is_finite(),
            "truncated train reconstructs finitely"
        );
    }

    #[test]
    fn round_rank_cap_honored() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_031);
        let x: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
        let xd = x.to_dense();
        let d = space.d;

        let y = x.round(1e-14, crate::qn::Pooled(2), RoundMode::Abs);
        y.check();
        for b in 1..d {
            assert!(
                y.ranks[b].total() <= 2,
                "bond {b} total rank ≤ rmax=2 (maxRank arm)"
            );
        }

        let z = x.round(0.0, crate::qn::Pooled(2), RoundMode::Abs);
        z.check();
        for b in 1..d {
            assert!(
                z.ranks[b].total() <= 2,
                "bond {b} total rank ≤ rmax=2 (chop_rank)"
            );
        }

        let z4 = x.round(0.0, crate::qn::Pooled(4), RoundMode::Abs);
        z4.check();
        for b in 1..d {
            assert!(z4.ranks[b].total() <= 4, "bond {b} total rank ≤ rmax=4");
        }
        assert!(
            sum_interior_rank(&z4) >= sum_interior_rank(&z),
            "rmax=4 keeps ≥ as many directions as rmax=2"
        );
        assert!(
            rel_err(&z4.to_dense(), &xd) <= rel_err(&z.to_dense(), &xd) + 1e-12,
            "rmax=4 approximates ≥ as well as rmax=2"
        );
    }
}

#[cfg(test)]
mod phase7b_tests {
    use crate::qn::QnSpace;
    use crate::round::round_sum;
    use crate::test_util::randomized_state;
    use crate::{TtVector, tt_zeros};
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t7a_round_sum_bound() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A90);
        let summands: Vec<TtVector<f64>> = (0..5)
            .map(|_| {
                let s = randomized_state::<f64, _>(space, 2, 4, &mut rng);
                let n = s.norm();
                &s * (1.0 / n)
            })
            .collect();
        let alpha: Vec<f64> = (0..5)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        for &nn in &[1usize, 2, 5] {
            let a = &alpha[..nn];
            let xs = &summands[..nn];
            let mut ex = tt_zeros::<f64>(space);
            for (aa, s) in a.iter().zip(xs) {
                ex = &ex + &(s * *aa);
            }
            for &tol in &[1e-2, 1e-6] {
                let r = round_sum(a, xs, tol);
                r.check();
                let err = (&r - &ex).norm();
                assert!(
                    err <= tol + 1e-7,
                    "n={nn} tol={tol}: round_sum abs error {err} > tol"
                );
            }
            let r0 = round_sum(a, xs, 1e-14);
            let err0 = (&r0 - &ex).norm();
            assert!(err0 <= 1e-6, "n={nn}: round_sum(tol→0) not exact ({err0})");
        }
    }
}

#[cfg(test)]
mod sigma_trim_tests {
    use crate::round::RoundMode;
    use crate::test_util::randomized_state;
    use crate::{TtVector, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn sigma_floor_trim_drops_dead_values_and_is_a_strict_noop_otherwise() {
        let space = crate::qn::QnSpace::new(8, 4, 4);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x51F7);
        let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
        x.round_mut(1e-14, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let pooled = |v: &TtVector<f64>| -> Vec<usize> {
            (0..=space.d)
                .map(|b| {
                    space
                        .bond_sectors(b)
                        .into_iter()
                        .map(|s| v.ranks[b][s])
                        .sum()
                })
                .collect()
        };
        let base = pooled(&x);

        let mut x_same = x.clone();
        let rep = x_same.sigma_floor_trim_mut(1e-8);
        assert_eq!(rep.n_dropped, 0);
        assert_eq!(x_same.ranks, x.ranks);
        assert_eq!(
            x_same.to_dense(),
            x.to_dense(),
            "a trim that drops nothing must not touch the cores (bitwise)"
        );
        assert_eq!(x.clone().sigma_floor_trim_mut(f64::NAN).n_dropped, 0);
        assert_eq!(x.clone().sigma_floor_trim_mut(0.0).n_dropped, 0);

        let noise: TtVector<f64> = tt_randn(space, &x.ranks, &mut rng);
        let mut xb = &x + &(&noise * (1e-10 / noise.norm()));
        xb.scale_mut(1.0 / xb.norm());
        let launch = pooled(&xb);
        let extra: usize = (1..space.d).map(|b| launch[b] - base[b]).sum();
        assert!(
            extra > 0,
            "fixture gate: the admixture must raise the ranks"
        );
        let mut xt = xb.clone();
        let rep = xt.sigma_floor_trim_mut(1e-8);
        assert!(
            rep.n_dropped > 0 && rep.n_dropped <= extra,
            "dropped {} of at most {extra} admixture directions",
            rep.n_dropped
        );
        assert_eq!(pooled(&xt), base, "the trimmed grid is x's own");
        assert!(rep.sigma_dropped_max < 1e-8 && rep.sigma_dropped_max > 0.0);
        assert!(
            rep.dropped_mass_max < 1e-18,
            "dropped mass per bond {:.3e}",
            rep.dropped_mass_max
        );
        let dev = (&xt - &x).norm_lr(false);
        assert!(
            dev < 1e-9,
            "the trimmed state must be x up to the admixture: ‖x' − x‖ = {dev:.3e}"
        );

        let mut x1 = x.clone();
        let rep = x1.sigma_floor_trim_mut(10.0);
        assert!(rep.n_dropped > 0);
        for b in 1..space.d {
            assert_eq!(pooled(&x1)[b], 1, "bond {b} must keep exactly one value");
        }
        assert!(x1.norm() > 0.0);
    }
}

#[cfg(test)]
mod block_tests;
