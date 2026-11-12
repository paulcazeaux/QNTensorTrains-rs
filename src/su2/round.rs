// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Orthogonality-center bookkeeping, rounding, and norm-orthonormalization sweeps. The abelian
//! `src/round.rs` is the STRUCTURAL mirror for every function
//! here (the sweep logic is structurally identical to
//! the abelian implementation") — clone its control flow and substitute the SU(2) `coreops.rs`
//! kernels + the θ/centerweight hooks documented at each call site below.

use faer::prelude::ReborrowMut;

use crate::qn::{PerSector, Pooled};
use crate::round::RoundMode;
use crate::scalar::Scalar;
use crate::su2::chop::{chop_energy, chop_global};
use crate::su2::core::{Su2Core, assert_grid_eq};
use crate::su2::coreops::{mul_sketch_left, mul_sketch_right};
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{SymGrid, SymLabel, SymSpace, Symmetry};
use crate::su2::vector::Su2TtVector;

fn scale_sketch<T: Scalar, S: Symmetry>(sk: &mut Su2Sketch<T, S>, factor: f64) {
    sk.scale_mut(T::from_f64(factor));
}

/// Deterministic weighted sum with rounding (the abelian
/// [`crate::round::round_sum`] — generic over the vector trait, so free for SU(2) by dispatch;
/// [`crate::round::round_sum`] is concrete on the abelian
/// `TtVector<T>`, so the ~15-line binary-tree body is duplicated here
/// ([`crate::solvers::layer::TtState::round_sum`]). Structural mirror of
/// [`crate::round::round_sum`]
/// EXACTLY: `n==1` → `(&summands[0]*alpha[0]).round(tol,0,Abs)`; `n==2` → scale both, add, round at
/// `tol`; else split at `n/2`, recurse each half at `tol/3`, add, round at `tol/3`. The `tol/3`
/// thirds and `RoundMode::Abs` are load-bearing — copy them exactly. All SU(2)-specific weighting
/// (θ, centerweight) already lives inside [`Su2TtVector::round`]/the `+`/`*` impls, so this
/// function itself carries NO weights. Do NOT genericize `crate::round::round_sum` over `TtState`
/// instead (circular: `TtState::round_sum` is the very method being implemented).
pub fn round_sum<T: Scalar, S: Symmetry>(
    alpha: &[T],
    summands: &[Su2TtVector<T, S>],
    tol: f64,
) -> Su2TtVector<T, S> {
    assert!(!summands.is_empty(), "round_sum: no summands");
    assert_eq!(
        alpha.len(),
        summands.len(),
        "round_sum: alpha/summands length mismatch"
    );
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

impl<T: Scalar, S: Symmetry> Su2TtVector<T, S> {
    /// Install a neighbor pair of cores with an explicit rank-grid copy + consistency asserts
    /// (`set_cores`-equivalent; the rank-matrix aliasing becomes a `clone` here, matching the
    /// abelian `TtVector::set_cores`).
    pub fn set_cores(&mut self, first: Su2Core<T, S>, second: Su2Core<T, S>) {
        let k = first.k;
        debug_assert_eq!(second.k, k + 1, "set_cores: cores not adjacent");
        if cfg!(debug_assertions) {
            assert_grid_eq!(
                first.col_ranks,
                second.row_ranks,
                "set_cores: bond rank mismatch"
            );
        }
        self.ranks[k + 1] = first.col_ranks.clone();
        self.cores[k] = first;
        self.cores[k + 1] = second;
    }

    /// Replace a single core in place, updating both incident bond rank grids.
    pub fn set_core(&mut self, core: Su2Core<T, S>) {
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

    /// Move the orthogonality center to `pos`. Already-orthogonal
    /// trains sweep from the current center; non-orthogonal trains canonicalize fully (left sweep,
    /// then right-to-left sweep). Reverse loop order is load-bearing (mirror the abelian
    /// [`crate::vector::TtVector::move_core`] exactly).
    pub fn move_core(&mut self, pos: usize, keep_rank: bool) {
        let d = self.space.sites();
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
        self.move_core(self.space.sites() - 1, keep_rank);
    }

    /// Canonicalize right-to-left, center at the first core.
    pub fn right_orthogonalize(&mut self, keep_rank: bool) {
        self.move_core(0, keep_rank);
    }

    fn bond_rmax(&self, maxr: &[S::Grid<usize>], bond: usize, chi: Pooled) -> Pooled {
        if chi.is_none() {
            return Pooled::NONE;
        }
        let cap = self
            .space
            .bond_labels(bond)
            .iter()
            .fold(0u64, |a, &s| a.saturating_add(maxr[bond][s] as u64));
        Pooled(usize::try_from((chi.0 as u64).min(cap)).unwrap_or(usize::MAX))
    }

    fn round_horizontal_sweep<F>(&mut self, mut decide: F)
    where
        F: FnMut(usize, &Su2Sketch<T, S>) -> S::Grid<usize>,
    {
        let space = self.space;
        let d = space.sites();
        for k in (1..=(d - 1)).rev() {
            let (mut u, sval, vt) = self.cores[k].svd_horizontal();
            let ranks = decide(k, &sval);
            u.rmul_sketch_mut(&sval);

            let mut xk = Su2Core::<T, S>::zeros_sym(space, k, &ranks, &self.ranks[k + 1]);
            let urr = u.row_ranks();
            let mut c = Su2Sketch::<T, S>::zeros_sym(space, k, &urr, &ranks);
            for s in space.bond_labels(k) {
                let rk = ranks[s];
                if rk > 0
                    && let Some(vtb) = vt[s].as_ref()
                {
                    let mut trunc = vtb.as_ref().subrows(0, rk).to_owned();
                    xk.unscale_horizontal(&mut trunc, s);
                    xk.scatter_horizontal(s, trunc.as_ref());
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

    /// Rounded copy. Right-canonical result (center
    /// `0`). `chi` is a POOLED cap on each bond's TOTAL number of REDUCED MULTIPLETS
    /// ([`Pooled::NONE`] = uncapped); it is never comparable to an abelian `χ`.
    pub fn round(&self, eps: f64, chi: Pooled, mode: RoundMode) -> Self {
        let mut tt = self.clone();
        tt.round_mut(eps, chi, mode);
        tt
    }

    /// In-place ε-weighted round: left-orthogonalize, then [`Self::round_horizontal_sweep`] with
    /// per-bond budget `ep = base/√(d-1)·ϵ` (`base = mode==Rel ? norm(tt) : 1`, the fixed-Sz-member
    /// Frobenius norm). `d==1`: no sweep, `ortho=Some(0)`. `chi` is a POOLED cap on each bond's
    /// TOTAL number of REDUCED MULTIPLETS ([`Pooled::NONE`] = uncapped).
    pub fn round_mut(&mut self, eps: f64, chi: Pooled, mode: RoundMode) {
        let d = self.space.sites();
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
                let maxr = S::max_ranks_open(&self.space, self.boundary_width());
                (0..d).map(|b| self.bond_rmax(&maxr, b, chi)).collect()
            };
            self.round_horizontal_sweep(|k, sval| chop_energy(sval, ep, chis[k]));
        }
        self.ortho = Some(0);
        debug_assert!(self.check());
    }

    /// Rounded copy via one pooled global energy threshold. Result is NOT flagged orthogonal.
    /// `rmax` is a PER-SECTOR cap on the reduced rank of each sector, NOT a pooled one.
    pub fn round_global(&self, eps: f64, rmax: PerSector, mode: RoundMode) -> Self {
        let mut tt = self.clone();
        tt.round_global_mut(eps, rmax, mode);
        tt
    }

    /// In-place global round:
    /// [`Self::svdvals`] (left-canonicalizes, then a θ-weighted right-to-left sweep) +
    /// [`crate::su2::chop::chop_global`] (pooled WEIGHTED-σ threshold across ALL `d-1` bonds,
    /// `ranks_g[i] ↔ bond i+1`) + per-bond pathcount-rmax clamp + [`Su2Core::reduce_ranks`] on
    /// every core. `ortho=None` on exit. `rmax` is a PER-SECTOR cap on the reduced rank of each
    /// sector, applied after the global chop, NOT a pooled one.
    pub fn round_global_mut(&mut self, eps: f64, rmax: PerSector, mode: RoundMode) {
        let d = self.space.sites();
        let svals = self.svdvals();
        let base = match mode {
            RoundMode::Rel => self.norm(),
            RoundMode::Abs => 1.0,
        };
        if d > 1 {
            let ranks_g = chop_global(&svals, base * eps);
            let maxr = S::max_ranks_open(&self.space, self.boundary_width());
            for b in 1..=(d - 1) {
                let mut rk = ranks_g[b - 1].clone();
                for s in self.space.bond_labels(b) {
                    let mr = maxr[b][s];
                    let cap = if !rmax.is_none() { rmax.0.min(mr) } else { mr };
                    rk[s] = rk[s].min(cap);
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

    /// Left-to-right SVD rounding sweep. Asserts right-canonical input (`ortho == Some(0)`);
    /// produces a left-canonical result (`ortho == Some(d-1)`). SU(2) adds the `svd_vertical`
    /// `√centerweight(2s_col)` SCALAR scale + `unscale_vertical` on the pushed `C = S·Vt` factor
    /// (`U` stays RAW).
    pub fn round_lr_mut(&mut self, eps: f64, chi: Pooled, mode: RoundMode) {
        let d = self.space.sites();
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
            let maxr = S::max_ranks_open(&self.space, self.boundary_width());
            (0..=d).map(|b| self.bond_rmax(&maxr, b, chi)).collect()
        };
        for k in 0..=(d - 2) {
            let (u, sval, mut vt) = self.cores[k].svd_vertical();
            let ranks = chop_energy(&sval, ep, chis[k + 1]);
            for r in space.bond_labels(k + 1) {
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
            let mut xk = Su2Core::<T, S>::zeros_sym(space, k, &self.ranks[k], &ranks);
            let mut c = Su2Sketch::<T, S>::zeros_sym(space, k + 1, &ranks, &self.ranks[k + 1]);
            for r in space.bond_labels(k + 1) {
                let rk = ranks[r];
                if rk > 0
                    && let Some(ub) = u[r].as_ref()
                {
                    xk.scatter_vertical(r, ub.as_ref().subcols(0, rk));
                }
                if let (Some(vtb), Some(mut cb)) = (vt.block(r), c.block_mut(r)) {
                    let inv = T::from_f64(
                        1.0 / crate::su2::centerweight(r.two_j(), space.total_two_j()).sqrt(),
                    );
                    let cols = vtb.ncols();
                    for j in 0..cols {
                        for i in 0..rk {
                            *cb.rb_mut().get_mut(i, j) = *vtb.get(i, j) * inv;
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
    /// entry min-clamped against the available singular count. Mirror the abelian
    /// [`crate::vector::TtVector::round_ranks_mut`].
    pub fn round_ranks_mut(&mut self, ranks: &[S::Grid<usize>]) {
        let d = self.space.sites();
        assert_eq!(ranks.len(), d + 1, "round_ranks: need d+1 rank grids");
        debug_assert!(
            (0..=d).all(|b| ranks[b]
                .iter()
                .all(|(s, &v)| v == 0 || self.space.admissible(b, s))),
            "round_ranks: positive requested ranks must be on admissible sectors"
        );
        self.left_orthogonalize(false);
        if d == 1 {
            self.ortho = Some(0);
            return;
        }
        let targets: Vec<S::Grid<usize>> = ranks.to_vec();
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

    /// Per-value Schmidt-floor trim of the POINT — the SU(2) twin of
    /// [`crate::vector::TtVector::sigma_floor_trim_mut`]. The floor is compared against the
    /// CENTERWEIGHTED value `ŵ = √cw(2s, 2S)·σ` of each reduced singular value, the fixed-Sz
    /// Schmidt weight and the unit every other threshold on this layer uses
    /// ([`crate::su2::weingarten::min_schmidt`], the cone σ-floor), so one numeric floor is one
    /// physical threshold on both layers; `sigma_dropped_max` is reported in the same units.
    /// Strict no-op (bit-identical) when nothing is below the floor; keep-one fallback per bond;
    /// does NOT renormalize.
    pub fn sigma_floor_trim_mut(&mut self, floor: f64) -> crate::round::SigmaTrim {
        let d = self.space.sites();
        if d < 2 || !floor.is_finite() || floor <= 0.0 {
            return crate::round::SigmaTrim::default();
        }
        let two_j_total = self.space.total_two_j();
        let svals = self.clone().svdvals();
        let mut targets: Vec<S::Grid<usize>> = self.ranks.clone();
        let mut rep = crate::round::SigmaTrim::default();
        let mut changed = false;
        for b in 1..d {
            let sv = &svals[b - 1];
            let mut grid = <S::Grid<usize>>::filled(&self.space, 0);
            let mut dropped_sq = 0.0f64;
            let mut kept = 0usize;
            let mut best: Option<(S::Label, f64)> = None;
            for s in self.space.bond_labels(b) {
                let Some(blk) = sv.block(s) else { continue };
                let len = blk.nrows().min(blk.ncols());
                if len == 0 {
                    continue;
                }
                let w = crate::su2::centerweight(s.two_j(), two_j_total).sqrt();
                let lead = w * blk.get(0, 0).re_s();
                if best.is_none_or(|(_, v)| v < lead) {
                    best = Some((s, lead));
                }
                let mut keep = 0usize;
                for i in 0..len {
                    let v = w * blk.get(i, i).re_s();
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
            if !grid.eq_grid(&targets[b]) {
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

    /// Bond singular values: left-orthogonalize, then a right-to-left `svd_horizontal_to_core`
    /// sweep (θ-weighted; `Vt` unscaled INTO the right-canonical core on write-back). Returns the
    /// `d-1` interior-bond diagonal `Su2Sketch`es. Their σ are the θ-weighted (bond-metric)
    /// fixed-Sz Schmidt values — `scale_horizontal`'s θ is folded IN (as in
    /// [`Su2Core::svd_horizontal`], `coreops.rs`), but the `√centerweight(2s,2S)` column-degeneracy
    /// factor is NOT (the [`crate::su2::chop`] family applies it when pooling; R12 re-applies it to
    /// recover the squared fixed-Sz norm). `svals[i] ↔ bond i+1`. Leaves `self` right-canonical
    /// (`ortho=Some(0)`). `d==1` → empty `Vec`. Mirror [`crate::vector::TtVector::svdvals`].
    pub fn svdvals(&mut self) -> Vec<Su2Sketch<T, S>> {
        let d = self.space.sites();
        self.left_orthogonalize(false);
        if d == 1 {
            return Vec::new();
        }
        let mut svals: Vec<Option<Su2Sketch<T, S>>> = (0..d - 1).map(|_| None).collect();
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

    /// Left-orthonormalize in place, returning the per-step
    /// factor norms (`len == d`; their product is `‖self‖`). Terminal core normalized by
    /// [`Su2Core::norm_center`] with a zero-norm guard and an explicit `d==1` special case. `pub`,
    /// mirroring the abelian `TtVector::left_orthonormalize`, which the Krylov solvers need.
    pub fn left_orthonormalize(&mut self, keep_rank: bool) -> Vec<f64> {
        let d = self.space.sites();
        let mut nrm = vec![1.0f64; d];
        if d == 1 {
            let n = self.cores[0].norm_center();
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
        let n = self.cores[d - 1].norm_center();
        nrm[d - 1] = n;
        if n != 0.0 {
            self.cores[d - 1].scale_mut(T::from_f64(1.0 / n));
        }
        self.ortho = Some(d - 1);
        nrm
    }

    fn right_orthonormalize(&mut self, keep_rank: bool) -> Vec<f64> {
        let d = self.space.sites();
        let mut nrm = vec![1.0f64; d];
        if d == 1 {
            let n = self.cores[0].norm_center();
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
        let n = self.cores[0].norm_center();
        nrm[0] = n;
        if n != 0.0 {
            self.cores[0].scale_mut(T::from_f64(1.0 / n));
        }
        self.ortho = Some(0);
        nrm
    }

    /// Norm via a left-to-right orthonormalization sweep `= exp(Σ ln nrm)`. CLONES internally
    /// (value semantics — the in-place form mutates and rank-reduces its argument). `keep_rank` selects `qr_fixed` over `qc`.
    pub fn norm_lr(&self, keep_rank: bool) -> f64 {
        if let Some(c) = self.ortho {
            return self.cores[c].norm_center();
        }
        let mut tt = self.clone();
        let nrm = tt.left_orthonormalize(keep_rank);
        nrm.iter().map(|&x| x.ln()).sum::<f64>().exp()
    }

    /// Norm via a right-to-left orthonormalization sweep. CLONES
    /// internally.
    pub fn norm_rl(&self, keep_rank: bool) -> f64 {
        let mut tt = self.clone();
        let nrm = tt.right_orthonormalize(keep_rank);
        nrm.iter().map(|&x| x.ln()).sum::<f64>().exp()
    }
}

#[cfg(test)]
mod polish_cost_tests {
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::randomized_su2_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn pc2_su2_norm_lr_fast_path_matches_the_sweep() {
        let space = Su2Space::new(6, 6, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5702);
        for trial in 0..4 {
            let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
            let center = trial % space.d;
            x.move_core(center, true);
            assert_eq!(x.ortho, Some(center));
            let fast = x.norm_lr(false);
            let mut y = x.clone();
            y.ortho = None;
            let swept = y.norm_lr(false);
            assert!(
                (fast - swept).abs() <= 1e-12 * swept.max(1.0),
                "trial {trial}: fast {fast:.17e} vs swept {swept:.17e}"
            );
            let rl = x.norm_rl(false);
            assert!((fast - rl).abs() <= 1e-12 * rl.max(1.0));
        }
    }
}

#[cfg(test)]
mod s2_tests {
    use crate::round::RoundMode;
    use crate::su2::qn::{Su2RankGrid, Su2Space};
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randomized_su2_state, seeded};

    #[test]
    fn r8_round_to_prescribed_ranks() {
        let mut rng = seeded(20_108);
        let space = Su2Space::new(6, 4, 2);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        let mut y = x.clone();
        y.left_orthogonalize(false);

        let ranks: Vec<Su2RankGrid> = (0..=space.d)
            .map(|b| {
                let mut r = Su2RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    r[s] = y.ranks[b][s].min(2);
                }
                r
            })
            .collect();

        y.round_ranks_mut(&ranks);
        y.check();

        for (b, target) in ranks.iter().enumerate() {
            for s in space.bond_sectors(b) {
                assert!(
                    y.ranks[b][s] <= target[s],
                    "bond {b} sector {s:?}: rank {} exceeds prescribed {}",
                    y.ranks[b][s],
                    target[s]
                );
            }
        }
    }

    #[test]
    fn r10_rmax_caps_reduced_bond_ranks() {
        let mut rng = seeded(20_110);
        let space = Su2Space::new(7, 5, 1);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);

        let y = x.round(0.0, crate::qn::Pooled(3), RoundMode::Abs);
        y.check();
        for b in 1..space.d {
            let total: usize = space.bond_sectors(b).iter().map(|&s| y.ranks[b][s]).sum();
            assert!(total <= 3, "bond {b}: total rank {total} exceeds cap 3");
        }
    }
}

#[cfg(test)]
mod block_tests;
