// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Two-site contracted core for SU(2): the local MALS/TDVP iterate spanning sites `(k, k+1)`.
//! The CORE half of the two-site container and its factorizations (the algebra hooks:
//! `contract`, `factor_qc`, `factor_svd`, the θ/centerweighted `dot`/`axpy`/`scale`).

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;

use crate::core::algebra::{block_axpy, block_dot_cols, block_scale};
use crate::core::factor::my_qc;
use crate::core::twosite::noise_rng;
use crate::par;
use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};

mod noise;
pub use noise::{factor_svd_noise, factor_svd_perturbative};

mod average;
pub use average::factor_svd_average;
pub(crate) use average::weighted_noise_sum;

mod moderot;
pub(crate) use moderot::apply_mode_rotation_cs;
pub use moderot::{
    apply_mode_rotation, apply_mode_rotation_serial, apply_mode_swap, rotated_bond_svals,
    search_angle,
};

const PAR_MIN_MID_SECTORS: usize = 4;

/// Two-site contracted core: one dense block per MID sector (bond `k+1`) — unlike [`Su2Core`]'s
/// four families, `Su2ContractedCores` has NO family axis (the abelian analogue is
/// `crate::core::twosite::ContractedSparseCores`). Block shape at mid sector `m`: rows =
/// `Σ_{(f,l)∈mpred(space,k,m)} row_ranks[l]`, cols = `Σ_{(f,r)∈msucc(space,k,m)} col_ranks[r]` —
/// see [`mpred`]/[`msucc`] (the H2 window helpers).
#[derive(Clone)]
pub struct Su2ContractedCores<T: Scalar> {
    /// Symmetry descriptor shared with the parent train.
    pub space: Su2Space,
    /// 0-based site of the LEFT core of the pair; the pair spans sites `(k, k+1)`.
    pub k: usize,
    /// Admissible row sectors (bond `k`), canonical order.
    pub row_qn: Vec<Su2Sector>,
    /// Admissible mid sectors (bond `k+1`), canonical order — the only block axis.
    pub mid_qn: Vec<Su2Sector>,
    /// Admissible column sectors (bond `k+2`), canonical order.
    pub col_qn: Vec<Su2Sector>,
    /// Per-sector ranks at bond `k` (== a hypothetical `Xₖ.row_ranks`).
    pub row_ranks: Su2RankGrid,
    /// Per-sector ranks at bond `k+2` (== a hypothetical `Xₖ₊₁.col_ranks`).
    pub col_ranks: Su2RankGrid,
    blocks: Su2SectorGrid<Option<Mat<T>>>,
}

pub(crate) fn msucc(space: &Su2Space, k: usize, m: Su2Sector) -> Vec<(Family, Su2Sector)> {
    let g = space.pg.irrep_of(k + 1);
    Family::ALL
        .iter()
        .filter_map(|&f| {
            f.apply(m, g)
                .filter(|&r| space.contains(k + 2, r))
                .map(|r| (f, r))
        })
        .collect()
}

pub(crate) fn mpred(space: &Su2Space, k: usize, m: Su2Sector) -> Vec<(Family, Su2Sector)> {
    let g = space.pg.irrep_of(k);
    let mut out = Vec::new();
    for f in Family::ALL {
        let Some(l) = f.unapply(m, g) else { continue };
        if !space.contains(k, l) {
            continue;
        }
        out.push((f, l));
    }
    out
}

impl<T: Scalar> Su2ContractedCores<T> {
    /// Zero-initialized contracted core at site `k` with the given row/col bond ranks. `assert!(k+2
    /// <= space.d)` first (mirrors the abelian [`crate::core::twosite::ContractedSparseCores::zeros`]).
    /// Block shapes: rows via [`mpred`], cols via [`msucc`].
    pub fn zeros(
        space: Su2Space,
        k: usize,
        row_ranks: &Su2RankGrid,
        col_ranks: &Su2RankGrid,
    ) -> Self {
        assert!(
            k + 2 <= space.d,
            "Su2ContractedCores: site {k} needs bond k+2 <= d={}",
            space.d
        );
        let row_qn = space.bond_sectors(k);
        let mid_qn = space.bond_sectors(k + 1);
        let col_qn = space.bond_sectors(k + 2);

        let blocks = Su2SectorGrid::from_fn(&space, |m| {
            space.contains(k + 1, m).then(|| {
                let rows: usize = mpred(&space, k, m).iter().map(|&(_, l)| row_ranks[l]).sum();
                let cols: usize = msucc(&space, k, m).iter().map(|&(_, r)| col_ranks[r]).sum();
                Mat::<T>::zeros(rows, cols)
            })
        });

        Su2ContractedCores {
            space,
            k,
            row_qn,
            mid_qn,
            col_qn,
            row_ranks: row_ranks.clone(),
            col_ranks: col_ranks.clone(),
            blocks,
        }
    }

    /// Fresh zero-initialized clone of the same shape. NOT a data copy — see [`Clone`] for that;
    /// `*`/`scale` must clone-then-scale (the `t7a_contracted_scaling_regression`-mirror guard).
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.space, self.k, &self.row_ranks, &self.col_ranks)
    }

    /// `(row_qn.len(), col_qn.len())` (the cached `length(row_qn)`, `length(col_qn)`, do NOT store
    /// them).
    pub fn dims(&self) -> (usize, usize) {
        (self.row_qn.len(), self.col_qn.len())
    }

    /// The left core's 0-based site.
    pub fn site(&self) -> usize {
        self.k
    }

    /// The block at mid sector `mid`, if present.
    #[inline]
    pub fn block(&self, mid: Su2Sector) -> Option<MatRef<'_, T>> {
        let (n1, s1) = self.blocks.shape();
        let ni = self.blocks.nirrep();
        if mid.n >= n1 || mid.two_s >= s1 || usize::from(mid.irrep.index()) >= ni {
            return None;
        }
        self.blocks[mid].as_ref().map(|m| m.as_ref())
    }

    /// Mutable block access.
    #[inline]
    pub fn block_mut(&mut self, mid: Su2Sector) -> Option<MatMut<'_, T>> {
        let (n1, s1) = self.blocks.shape();
        let ni = self.blocks.nirrep();
        if mid.n >= n1 || mid.two_s >= s1 || usize::from(mid.irrep.index()) >= ni {
            return None;
        }
        self.blocks[mid].as_mut().map(|m| m.as_mut())
    }

    /// `self *= a` per block — REUSE-SHAPE, unweighted, elementwise; one function covers all three
    /// forms.
    pub fn scale_mut(&mut self, a: T) {
        for m in self.mid_qn.clone() {
            if let Some(b) = self.block_mut(m) {
                block_scale(b, a);
            }
        }
    }

    /// `self += a · other` per block. `debug_assert_eq!(self.k, other.k)`. REUSE-SHAPE, unweighted,
    /// elementwise.
    pub fn axpy(&mut self, a: T, other: &Self) {
        debug_assert_eq!(self.k, other.k, "axpy: site mismatch");
        for m in self.mid_qn.clone() {
            let ob = other.block(m);
            if let (Some(sb), Some(ob)) = (self.block_mut(m), ob) {
                block_axpy(sb, ob, a);
            }
        }
    }

    /// θ/centerweighted inner product `Σ_m Σ_{(f,r)∈msucc(m)} centerweight(r.two_s, space.two_s) ·
    /// dot(block(m)[:, slice_r], other.block(m)[:, slice_r])`, conjugating `self`
    /// (H2 site #2 — **the single most dangerous divergence in this file**).
    /// The weight is per column SLICE (the successor sector `r`, NOT the mid sector `m`), columns
    /// only — rows carry NO weight. Serial reduction in `mid_qn` canonical order (the
    /// determinism rule). `debug_assert_eq!` row/col ranks first.
    pub fn dot_cores(&self, other: &Self) -> T {
        debug_assert_eq!(
            self.row_ranks, other.row_ranks,
            "dot_cores: row rank mismatch"
        );
        debug_assert_eq!(
            self.col_ranks, other.col_ranks,
            "dot_cores: col rank mismatch"
        );
        let mut acc = T::zero();
        for &m in &self.mid_qn {
            let (Some(a), Some(b)) = (self.block(m), other.block(m)) else {
                continue;
            };
            let mut off = 0usize;
            for (_, r) in msucc(&self.space, self.k, m) {
                let w = self.col_ranks[r];
                let cw = T::from_f64(crate::su2::centerweight(r.two_s, self.space.two_s));
                acc += cw * block_dot_cols(a, b, off, off + w);
                off += w;
            }
        }
        acc
    }

    /// `sqrt(|Re dot_cores(self,self)|)`; the `.abs()` mirrors the shipped abelian
    /// [`crate::core::twosite::ContractedSparseCores::norm`] — keep it for parity with
    /// `Su2TtVector::norm`.
    pub fn norm(&self) -> f64 {
        self.dot_cores(self).re_s().abs().sqrt()
    }

    pub(crate) fn dot_cores_par(&self, other: &Self) -> T {
        if !par::par_vector_ops(self.num_coeffs()) {
            return self.dot_cores(other);
        }
        let (space, k) = (self.space, self.k);
        let (av, bv) = (self.blocks.values(), other.blocks.values());
        let parts: Vec<T> = (0..av.len())
            .into_par_iter()
            .map(|idx| {
                let mut acc = T::zero();
                let (Some(a), Some(b)) = (&av[idx], &bv[idx]) else {
                    return acc;
                };
                let m = self.blocks.sector_at(idx);
                let mut off = 0usize;
                for (_, r) in msucc(&space, k, m) {
                    let w = self.col_ranks[r];
                    let cw = T::from_f64(crate::su2::centerweight(r.two_s, space.two_s));
                    acc += cw * block_dot_cols(a.as_ref(), b.as_ref(), off, off + w);
                    off += w;
                }
                acc
            })
            .collect();
        parts.into_iter().fold(T::zero(), |acc, p| acc + p)
    }

    pub(crate) fn norm_par(&self) -> f64 {
        self.dot_cores_par(self).re_s().abs().sqrt()
    }

    pub(crate) fn axpy_par(&mut self, a: T, other: &Self) {
        debug_assert_eq!(self.k, other.k, "axpy: site mismatch");
        if !par::par_vector_ops(self.num_coeffs()) {
            self.axpy(a, other);
            return;
        }
        let ob = other.blocks.values();
        self.blocks
            .values_mut()
            .par_iter_mut()
            .zip(ob.par_iter())
            .for_each(|(sb, ob)| {
                if let (Some(sb), Some(ob)) = (sb, ob) {
                    block_axpy(sb.as_mut(), ob.as_ref(), a);
                }
            });
    }

    pub(crate) fn scale_mut_par(&mut self, a: T) {
        if !par::par_vector_ops(self.num_coeffs()) {
            self.scale_mut(a);
            return;
        }
        self.blocks.values_mut().par_iter_mut().for_each(|sb| {
            if let Some(sb) = sb {
                block_scale(sb.as_mut(), a);
            }
        });
    }

    /// Number of coefficients (sum of present-block sizes).
    pub fn num_coeffs(&self) -> usize {
        self.mid_qn
            .iter()
            .map(|&m| self.block(m).map_or(0, |b| b.nrows() * b.ncols()))
            .sum()
    }

    /// Visit every coefficient mutably in canonical (mid-sector-outer, column-major) order.
    pub fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, mut f: F) {
        for m in self.mid_qn.clone() {
            if let Some(mut b) = self.block_mut(m) {
                let (r, c) = (b.nrows(), b.ncols());
                for j in 0..c {
                    for i in 0..r {
                        f(b.rb_mut().get_mut(i, j));
                    }
                }
            }
        }
    }

    /// Zip two same-shaped contracted cores' coefficients in canonical order.
    pub fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, mut f: F) {
        for m in self.mid_qn.clone() {
            let ob = other.block(m);
            if let (Some(mut sb), Some(ob)) = (self.block_mut(m), ob) {
                let (r, c) = (sb.nrows(), sb.ncols());
                for j in 0..c {
                    for i in 0..r {
                        f(sb.rb_mut().get_mut(i, j), *ob.get(i, j));
                    }
                }
            }
        }
    }

    /// Read the coefficient at a flat canonical index (`0..num_coeffs()`).
    pub fn coeff(&self, idx: usize) -> T {
        let mut rem = idx;
        for &m in &self.mid_qn {
            if let Some(b) = self.block(m) {
                let n = b.nrows() * b.ncols();
                if rem < n {
                    let i = rem % b.nrows().max(1);
                    let j = rem / b.nrows().max(1);
                    return *b.get(i, j);
                }
                rem -= n;
            }
        }
        panic!("Su2ContractedCores::coeff: index {idx} out of range");
    }

    /// Write the coefficient at a flat canonical index.
    pub fn set_coeff(&mut self, idx: usize, val: T) {
        let mut rem = idx;
        for m in self.mid_qn.clone() {
            if let Some(mut b) = self.block_mut(m) {
                let (rows, n) = (b.nrows(), b.nrows() * b.ncols());
                if rem < n {
                    let i = rem % rows.max(1);
                    let j = rem / rows.max(1);
                    *b.rb_mut().get_mut(i, j) = val;
                    return;
                }
                rem -= n;
            }
        }
        panic!("Su2ContractedCores::set_coeff: index {idx} out of range");
    }
}

impl<T: Scalar> std::ops::Mul<T> for &Su2ContractedCores<T> {
    type Output = Su2ContractedCores<T>;
    fn mul(self, a: T) -> Su2ContractedCores<T> {
        let mut out = self.clone();
        out.scale_mut(a);
        out
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn contract_mid_block<'b, T: Scalar>(
    space: &Su2Space,
    k: usize,
    m: Su2Sector,
    row_ranks: &Su2RankGrid,
    col_ranks: &Su2RankGrid,
    lblock: &dyn Fn(Family, Su2Sector) -> Option<MatRef<'b, T>>,
    rblock: &dyn Fn(Family, Su2Sector) -> Option<MatRef<'b, T>>,
    mut blk: MatMut<'_, T>,
) {
    let mut roff = 0usize;
    for (fi, l) in mpred(space, k, m) {
        let r_l = row_ranks[l];
        let mut coff = 0usize;
        for (fj, r) in msucc(space, k, m) {
            let r_r = col_ranks[r];
            if r_l > 0
                && r_r > 0
                && let (Some(a), Some(b)) = (lblock(fi, l), rblock(fj, m))
            {
                let dst = blk.rb_mut().submatrix_mut(roff, coff, r_l, r_r);
                matmul(dst, Accum::Replace, a, b, T::one(), Par::Seq);
            }
            coff += r_r;
        }
        roff += r_l;
    }
}

/// Fuse two adjacent cores `(Xₖ, Xₖ₊₁)` into their two-site contraction. UNWEIGHTED (no θ, no
/// centerweight — those live in the inner product and the SVD split, same convention as
/// `hmatvec`). `debug_assert!` preconditions: `xk.k+1 == xk1.k`; `xk.col_ranks == xk1.row_ranks`.
/// `Par::Seq` on every faer call; the per-mid-sector loop forks to rayon (collect-then-write, gated
/// by `PAR_MIN_MID_SECTORS`). Sub-block tiling is `Family::ALL` order via `mpred`
/// (rows)/`msucc` (cols); offsets accumulate over the ADMISSIBLE window INCLUDING rank-0 entries
/// — do not filter the window by `rank > 0`, filter only the `matmul`.
pub fn contract<T: Scalar>(xk: &Su2Core<T>, xk1: &Su2Core<T>) -> Su2ContractedCores<T> {
    debug_assert_eq!(xk.k + 1, xk1.k, "contract: cores not adjacent");
    debug_assert_eq!(
        xk.col_ranks, xk1.row_ranks,
        "contract: shared mid-bond rank mismatch"
    );
    let space = xk.space;
    let k = xk.k;
    let mut out = Su2ContractedCores::zeros(space, k, &xk.row_ranks, &xk1.col_ranks);

    let mid_block = |m: Su2Sector| -> Option<(Su2Sector, Mat<T>)> {
        let (nr, nc) = {
            let b = out.block(m)?;
            (b.nrows(), b.ncols())
        };
        if nr == 0 || nc == 0 {
            return None;
        }
        let mut local = Mat::<T>::zeros(nr, nc);
        contract_mid_block(
            &space,
            k,
            m,
            &xk.row_ranks,
            &xk1.col_ranks,
            &|fi, l| xk.block(fi, l),
            &|fj, mm| xk1.block(fj, mm),
            local.as_mut(),
        );
        Some((m, local))
    };
    let results: Vec<(Su2Sector, Mat<T>)> = if xk.col_qn.len() >= PAR_MIN_MID_SECTORS {
        xk.col_qn.par_iter().filter_map(|&m| mid_block(m)).collect()
    } else {
        xk.col_qn.iter().filter_map(|&m| mid_block(m)).collect()
    };
    for (m, local) in results {
        if let Some(mut blk) = out.block_mut(m) {
            blk.copy_from(local.as_ref());
        }
    }
    out
}

/// Rank-revealing QC split of every mid block of `tc` (`factor_qc`).
/// UNWEIGHTED — NO θ anywhere: `Xₖ` comes out
/// left-orthonormal in the plain metric (the correct SU(2) left-canonical condition), and `Xₖ₊₁`
/// is deliberately NOT θ-right-canonical — do not "symmetrize" by θ-scaling `C`. Reuses the
/// shipped sector-agnostic `crate::core::factor::my_qc` verbatim; SU(2) never gets its own. No
/// rank cap here — `mid_rank` is whatever `my_qc` reveals.
pub fn factor_qc<T: Scalar>(tc: &Su2ContractedCores<T>) -> (Su2Core<T>, Su2Core<T>) {
    let space = tc.space;
    let k = tc.k;
    let mut mid_rank = Su2RankGrid::zeros(&space);
    let qc_of = |m: Su2Sector| -> (Su2Sector, Mat<T>, Mat<T>, usize) {
        let (q, c, rank) = match tc.block(m) {
            Some(b) => my_qc(b),
            None => (Mat::<T>::zeros(0, 0), Mat::<T>::zeros(0, 0), 0),
        };
        (m, q, c, rank)
    };
    let data: Vec<(Su2Sector, Mat<T>, Mat<T>, usize)> = if tc.mid_qn.len() >= PAR_MIN_MID_SECTORS {
        tc.mid_qn.par_iter().map(|&m| qc_of(m)).collect()
    } else {
        tc.mid_qn.iter().map(|&m| qc_of(m)).collect()
    };
    for &(m, _, _, rank) in &data {
        mid_rank[m] = rank;
    }
    let mut xk = Su2Core::zeros(space, k, &tc.row_ranks, &mid_rank);
    let mut xk1 = Su2Core::zeros(space, k + 1, &mid_rank, &tc.col_ranks);
    for (m, q, c, _) in &data {
        if mid_rank[*m] > 0 {
            xk.scatter_vertical(*m, q.as_ref());
            xk1.scatter_horizontal(*m, c.as_ref());
        }
    }
    (xk, xk1)
}

/// Per-mid-sector thin SVD split of `tc`, θ-scaled columns before the SVD, pooled-energy
/// [`crate::su2::chop::chop_energy`]-chopped (centerweighted by MID sector) across the bond to
/// `mid_rank[m]`, singular values optionally normalized by the CENTERWEIGHTED
/// [`Su2Sketch::norm`] (H2 site #3 — the most dangerous kernel in this file). `&tc` (NOT
/// destructive — clones the block before θ-scaling, the `norm_lr`/`dot_orth` cloning precedent).
/// `chi` is a POOLED cap on the split bond's TOTAL number of REDUCED MULTIPLETS;
/// [`Pooled::NONE`] means uncapped.
pub fn factor_svd<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    eps: f64,
    chi: Pooled,
    normalize: bool,
) -> (Su2Core<T>, Su2Sketch<T>, Su2Core<T>) {
    let (xk, s, xk1, _) = factor_svd_noise(tc, eps, chi, normalize, 0.0, &mut noise_rng(0));
    (xk, s, xk1)
}

impl<T: Scalar> crate::mem::Footprint for Su2ContractedCores<T> {
    fn bytes(&self) -> u64 {
        self.mid_qn
            .iter()
            .filter_map(|&m| self.block(m))
            .map(|b| crate::mem::block_bytes::<T>(b.nrows(), b.ncols()))
            .sum()
    }
}

#[cfg(test)]
mod s5a_tests {
    use crate::su2::{centerweight, theta};

    #[test]
    fn s5a_h2b_theta_argument_order() {
        for (row, col) in [(0usize, 1usize), (1, 2), (2, 1), (3, 2), (2, 4)] {
            let want = (((col + 1) as f64) / ((row + 1) as f64)).sqrt();
            let got = theta(row, col);
            assert!(
                (got - want).abs() <= 1e-14 * want,
                "h2b: theta({row},{col}) should be sqrt(({}+1)/({}+1)) = {want}, got {got}",
                col,
                row
            );
            let swapped = theta(col, row);
            if row != col {
                assert!(
                    (swapped - 1.0 / want).abs() <= 1e-14,
                    "h2b: theta is not reciprocal under argument swap"
                );
                assert!(
                    (swapped - want).abs() > 1e-6,
                    "h2b: vacuous — theta({row},{col}) and its swap coincide"
                );
            }
        }
        let two_s_tot = 2usize;
        for (a, b) in [(0usize, 1usize), (1, 2), (2, 3), (3, 2)] {
            let lhs = centerweight(a, two_s_tot) * theta(a, b).powi(2);
            let rhs = centerweight(b, two_s_tot);
            assert!(
                (lhs - rhs).abs() <= 1e-13 * rhs.abs().max(1.0),
                "h2b: centerweight({a})·theta({a},{b})² should equal centerweight({b}) ({lhs} \
                 vs {rhs})"
            );
        }
    }
}
