// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Two-site contracted core (MALS): the local iterate spanning sites `(k, k+1)`, with
//! `contract`, `factor_qc` and `factor_svd`.

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;

use crate::core::algebra::{block_axpy, block_dot_into, block_scale};
use crate::core::factor::{Sketch, my_qc};
use crate::core::{Family, SparseCore};
use crate::par;
use crate::qn::{Pooled, QnSpace, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

mod noise;
pub use noise::{TruncInfo, factor_svd_noise, factor_svd_perturbative, noise_rng};

mod average;
pub use average::{LoewdinInfo, factor_svd_average};
mod moderot;
pub(crate) use average::{loewdin_dense, weighted_noise_sum};
pub use moderot::{
    ModeRotOpts, apply_mode_rotation, apply_mode_rotation_serial, apply_mode_swap, gate16,
    rotated_bond_svals, search_angle,
};
pub(crate) use moderot::{apply_mode_rotation_cs, minimize_angle};

/// Two-site contracted core: one dense block per MID sector (bond `k+1`) — unlike `SparseCore`'s
/// four families, `ContractedSparseCores` has no family axis. Block shape at mid sector `m`: rows =
/// `Σ_{f∈Family::ALL} row_ranks[m⊖f]` (over admissible `m⊖f` at bond `k`, =
/// `SparseCore::vertical_height` restricted to `row_ranks`), cols = `Σ_{f∈Family::ALL}
/// col_ranks[m⊕f]` (admissible at bond `k+2`, = `horizontal_width` restricted to `col_ranks`).
#[derive(Clone)]
pub struct ContractedSparseCores<T: Scalar> {
    /// Symmetry descriptor shared with the parent train.
    pub space: QnSpace,
    /// 0-based site of the LEFT core of the pair; the pair spans sites `(k, k+1)`.
    pub k: usize,
    /// Admissible row sectors (bond `k`), canonical order.
    pub row_qn: Vec<Sector>,
    /// Admissible mid sectors (bond `k+1`), canonical order — the only block axis.
    pub mid_qn: Vec<Sector>,
    /// Admissible column sectors (bond `k+2`), canonical order.
    pub col_qn: Vec<Sector>,
    /// Per-sector ranks at bond `k` (the row bond; = a hypothetical `Xₖ.row_ranks`).
    pub row_ranks: RankGrid,
    /// Per-sector ranks at bond `k+2` (the col bond; = a hypothetical `Xₖ₊₁.col_ranks`).
    pub col_ranks: RankGrid,
    blocks: SectorGrid<Option<Mat<T>>>,
}

impl<T: Scalar> ContractedSparseCores<T> {
    /// Zero-initialized contracted core at site `k` with the given row/col bond ranks.
    pub fn zeros(space: QnSpace, k: usize, row_ranks: &RankGrid, col_ranks: &RankGrid) -> Self {
        assert!(
            k + 2 <= space.d,
            "ContractedSparseCores: site {k} needs bond k+2 ≤ d={}",
            space.d
        );
        let row_qn = space.bond_sectors(k);
        let mid_qn = space.bond_sectors(k + 1);
        let col_qn = space.bond_sectors(k + 2);

        let gl = space.pg.irrep_of(k);
        let gr = space.pg.irrep_of(k + 1);
        let blocks = SectorGrid::from_fn(&space, |m| {
            space.contains(k + 1, m).then(|| {
                let mut rows = 0usize;
                for f in Family::ALL {
                    if let Some(l) = f.unapply(m, gl)
                        && space.contains(k, l)
                    {
                        rows += row_ranks[l];
                    }
                }
                let mut cols = 0usize;
                for f in Family::ALL {
                    let r = f.apply(m, gr);
                    if space.contains(k + 2, r) {
                        cols += col_ranks[r];
                    }
                }
                Mat::<T>::zeros(rows, cols)
            })
        });

        ContractedSparseCores {
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

    /// Fresh zero-initialized clone of the same shape. NOT a data copy — see [`Clone`] for that.
    /// Only `zerovector`-style uses should call this; `*`/`scale` must clone-then-scale (the
    /// `t7a_contracted_scaling_regression` guard).
    pub fn zeros_like(&self) -> Self {
        Self::zeros(self.space, self.k, &self.row_ranks, &self.col_ranks)
    }

    /// `(row_qn.len(), col_qn.len())`.
    pub fn dims(&self) -> (usize, usize) {
        (self.row_qn.len(), self.col_qn.len())
    }

    /// The left core's 0-based site.
    pub fn site(&self) -> usize {
        self.k
    }

    /// The block at mid sector `mid`, if present.
    #[inline]
    pub fn block(&self, mid: Sector) -> Option<MatRef<'_, T>> {
        let (n1, d1) = self.blocks.shape();
        if mid.nup >= n1 || mid.ndn >= d1 || usize::from(mid.irrep.index()) >= self.blocks.nirrep()
        {
            return None;
        }
        self.blocks[mid].as_ref().map(|m| m.as_ref())
    }

    /// Mutable block access.
    #[inline]
    pub fn block_mut(&mut self, mid: Sector) -> Option<MatMut<'_, T>> {
        let (n1, d1) = self.blocks.shape();
        if mid.nup >= n1 || mid.ndn >= d1 || usize::from(mid.irrep.index()) >= self.blocks.nirrep()
        {
            return None;
        }
        self.blocks[mid].as_mut().map(|m| m.as_mut())
    }

    /// `self *= a` per block.
    pub fn scale_mut(&mut self, a: T) {
        for m in self.mid_qn.clone() {
            if let Some(b) = self.block_mut(m) {
                block_scale(b, a);
            }
        }
    }

    /// `self += a · other` per block. Precondition: same
    /// shape (same `k`/`mid_qn`/ranks).
    pub fn axpy(&mut self, a: T, other: &Self) {
        debug_assert_eq!(self.k, other.k, "axpy: site mismatch");
        for m in self.mid_qn.clone() {
            let ob = other.block(m);
            if let (Some(sb), Some(ob)) = (self.block_mut(m), ob) {
                block_axpy(sb, ob, a);
            }
        }
    }

    /// Package Hermitian inner product `Σ_m dot(blocks[m], other.blocks[m])`, conjugating `self`.
    /// Reduces over `mid_qn` in canonical order, serially (threading rule: never a parallel
    /// float reduction).
    pub fn dot_cores(&self, other: &Self) -> T {
        let mut acc = T::zero();
        for &m in &self.mid_qn {
            if let (Some(a), Some(b)) = (self.block(m), other.block(m)) {
                block_dot_into(&mut acc, a, b);
            }
        }
        acc
    }

    /// Package norm `sqrt(|Re⟨self,self⟩|)`.
    pub fn norm(&self) -> f64 {
        self.dot_cores(self).re_s().abs().sqrt()
    }

    pub(crate) fn dot_cores_par(&self, other: &Self) -> T {
        if !par::par_vector_ops(self.num_coeffs()) {
            return self.dot_cores(other);
        }
        let (a, b) = (self.blocks.values(), other.blocks.values());
        let parts: Vec<T> = a
            .par_iter()
            .zip(b.par_iter())
            .map(|(ab, bb)| {
                let mut acc = T::zero();
                if let (Some(ab), Some(bb)) = (ab, bb) {
                    block_dot_into(&mut acc, ab.as_ref(), bb.as_ref());
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

    /// Visit every coefficient mutably in canonical order.
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
        panic!("ContractedSparseCores::coeff: index {idx} out of range");
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
        panic!("ContractedSparseCores::set_coeff: index {idx} out of range");
    }
}

impl<T: Scalar> std::ops::Mul<T> for &ContractedSparseCores<T> {
    type Output = ContractedSparseCores<T>;
    fn mul(self, a: T) -> ContractedSparseCores<T> {
        let mut out = self.clone();
        out.scale_mut(a);
        out
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn contract_mid_block<'b, T: Scalar>(
    space: QnSpace,
    k: usize,
    m: Sector,
    row_ranks: &RankGrid,
    col_ranks: &RankGrid,
    lblock: &dyn Fn(Family, Sector) -> Option<MatRef<'b, T>>,
    rblock: &dyn Fn(Family, Sector) -> Option<MatRef<'b, T>>,
    mut blk: MatMut<'_, T>,
) {
    let gl = space.pg.irrep_of(k);
    let gr = space.pg.irrep_of(k + 1);
    let mut roff = 0usize;
    for fi in Family::ALL {
        let Some(l) = fi.unapply(m, gl) else {
            continue;
        };
        if !space.contains(k, l) {
            continue;
        }
        let r_l = row_ranks[l];
        let mut coff = 0usize;
        for fj in Family::ALL {
            let r = fj.apply(m, gr);
            if !space.contains(k + 2, r) {
                continue;
            }
            let r_r = col_ranks[r];
            if r_l > 0 && r_r > 0 {
                let (Some(a), Some(b)) = (lblock(fi, l), rblock(fj, m)) else {
                    coff += r_r;
                    continue;
                };
                let dst = blk.rb_mut().submatrix_mut(roff, coff, r_l, r_r);
                matmul(dst, Accum::Replace, a, b, T::one(), Par::Seq);
            }
            coff += r_r;
        }
        roff += r_l;
    }
}

/// Fuse two adjacent cores `(Xₖ, Xₖ₊₁)` into their two-site contraction. Sub-block tiling order is
/// `Family::ALL` (`[Empty,Up,Dn,Occ]`) on BOTH the row (decrement) and column (increment) sides —
/// NOT the constructor's `m₋` order (load-bearing: getting this wrong
/// silently corrupts the `factor_qc`/`factor_svd` splits).
pub fn contract<T: Scalar>(xk: &SparseCore<T>, xk1: &SparseCore<T>) -> ContractedSparseCores<T> {
    debug_assert_eq!(xk.k + 1, xk1.k, "contract: cores not adjacent");
    debug_assert_eq!(
        xk.col_ranks, xk1.row_ranks,
        "contract: shared mid-bond rank mismatch"
    );
    let space = xk.space;
    let k = xk.k;
    let mut out = ContractedSparseCores::zeros(space, k, &xk.row_ranks, &xk1.col_ranks);

    let results: Vec<(Sector, Mat<T>)> = xk
        .col_qn
        .par_iter()
        .filter_map(|&m| {
            let (nr, nc) = {
                let b = out.block(m)?;
                (b.nrows(), b.ncols())
            };
            if nr == 0 || nc == 0 {
                return None;
            }
            let mut local = Mat::<T>::zeros(nr, nc);
            contract_mid_block(
                space,
                k,
                m,
                &xk.row_ranks,
                &xk1.col_ranks,
                &|fi, l| xk.block(fi, l),
                &|fj, mm| xk1.block(fj, mm),
                local.as_mut(),
            );
            Some((m, local))
        })
        .collect();
    for (m, mres) in results {
        let mut blk = out
            .block_mut(m)
            .expect("contract: computed mid block admissible");
        blk.copy_from(mres.as_ref());
    }
    out
}

/// Rank-revealing QC split of every mid block of `tc`: per mid sector, `my_qc` the raw mid block
/// directly (it already IS both the vertical unfolding of `Xₖ` at col=mid and the horizontal
/// unfolding of `Xₖ₊₁` at row=mid), then scatter `Q` vertically into a fresh `Xₖ` and `C`
/// horizontally into a fresh `Xₖ₊₁`. `Xₖ` is left- orthonormal; `Xₖ · Xₖ₊₁ ≈ tc`.
pub fn factor_qc<T: Scalar>(tc: &ContractedSparseCores<T>) -> (SparseCore<T>, SparseCore<T>) {
    let space = tc.space;
    let k = tc.k;
    let mut mid_rank = RankGrid::zeros(&space);
    let data: Vec<(Sector, Mat<T>, Mat<T>, usize)> = tc
        .mid_qn
        .par_iter()
        .map(|&m| {
            let (q, c, rank) = match tc.block(m) {
                Some(b) => my_qc(b),
                None => (Mat::<T>::zeros(0, 0), Mat::<T>::zeros(0, 0), 0),
            };
            (m, q, c, rank)
        })
        .collect();
    for &(m, _, _, rank) in &data {
        mid_rank[m] = rank;
    }
    let mut xk = SparseCore::zeros(space, k, &tc.row_ranks, &mid_rank);
    let mut xk1 = SparseCore::zeros(space, k + 1, &mid_rank, &tc.col_ranks);
    for (m, q, c, _) in &data {
        if mid_rank[*m] > 0 {
            xk.scatter_vertical(*m, q.as_ref());
            xk1.scatter_horizontal(*m, c.as_ref());
        }
    }
    (xk, xk1)
}

/// Per-mid-sector thin SVD split of `tc`, pooled-energy `chop`ped across the bond to `mid_rank[m]`,
/// singular values optionally normalized by the bond Frobenius norm. Takes `&tc` and does not
/// mutate, matching the `norm_lr`/`dot_orth` cloning convention; `normalize=false` is the
/// TDVP hook, added early to avoid a later signature change). Returns `(Xₖ, S, Xₖ₊₁)` with
/// `Xₖ` left-orthonormal, `Xₖ₊₁` right- orthonormal, `S` the (optionally normalized) truncated
/// singular-value diagonal `Sketch` at bond `k+1`. `chi` is a POOLED cap on the split bond's
/// TOTAL dimension `Σ_q r[q]`; [`Pooled::NONE`] means uncapped (matches
/// [`crate::core::factor::chop_energy`]).
pub fn factor_svd<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    eps: f64,
    chi: Pooled,
    normalize: bool,
) -> (SparseCore<T>, Sketch<T>, SparseCore<T>) {
    let (xk, s, xk1, _) = factor_svd_noise(tc, eps, chi, normalize, 0.0, &mut noise_rng(0));
    (xk, s, xk1)
}

impl<T: Scalar> crate::mem::Footprint for ContractedSparseCores<T> {
    fn bytes(&self) -> u64 {
        self.mid_qn
            .iter()
            .filter_map(|&m| self.block(m))
            .map(|b| crate::mem::block_bytes::<T>(b.nrows(), b.ncols()))
            .sum()
    }
}

#[cfg(test)]
mod phase7_tests {
    use crate::TtVector;
    use crate::qn::QnSpace;
    use crate::solvers::mals::{ContractedSparseCores, contract, factor_svd};
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn assert_contracted_coeffs_close(
        a: &ContractedSparseCores<f64>,
        b: &ContractedSparseCores<f64>,
        tol: f64,
        msg: &str,
    ) {
        let n = a.num_coeffs();
        assert_eq!(n, b.num_coeffs(), "{msg}: coeff-count mismatch");
        let mut diff2 = 0.0f64;
        let mut bref2 = 0.0f64;
        for i in 0..n {
            let av = a.coeff(i);
            let bv = b.coeff(i);
            diff2 += (av - bv) * (av - bv);
            bref2 += bv * bv;
        }
        let bound = tol * bref2.sqrt().max(1.0);
        assert!(
            diff2.sqrt() <= bound,
            "{msg}: diff {} vs bound {}",
            diff2.sqrt(),
            bound
        );
    }

    #[test]
    fn t7a_contracted_scaling_regression() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A40);
        let x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
        x.check();

        let tc = contract(x.core(2), x.core(3));
        let tc_norm = tc.norm();
        assert!(tc_norm > 0.0, "contract(x2,x3) has zero norm");

        let tc2 = &tc * 2.5;
        assert!(
            (tc2.norm() - 2.5 * tc_norm).abs() <= 1e-12 * (2.5 * tc_norm),
            "‖2.5·tc‖ != 2.5·‖tc‖: {} vs {}",
            tc2.norm(),
            2.5 * tc_norm
        );
        for s in tc.mid_qn.clone() {
            let (Some(b), Some(b2)) = (tc.block(s), tc2.block(s)) else {
                continue;
            };
            let (r, c) = (b.nrows(), b.ncols());
            let mut diff2 = 0.0f64;
            let mut ref2 = 0.0f64;
            for j in 0..c {
                for i in 0..r {
                    let want = 2.5 * *b.get(i, j);
                    let got = *b2.get(i, j);
                    let d = got - want;
                    diff2 += d * d;
                    ref2 += want * want;
                }
            }
            assert!(
                diff2.sqrt() < 1e-12 * (1.0 + ref2.sqrt()),
                "mid sector {s:?}: scaled block mismatch"
            );
        }
    }

    #[test]
    fn t7a_factor_svd_reconstruction() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A80);
        let x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.check();
        let tc = contract(x.core(2), x.core(3));
        let tc_norm = tc.norm();
        assert!(tc_norm > 0.0, "contracted pair has zero norm");

        let (xk, s, mut xk1) = factor_svd(&tc, 1e-12, crate::qn::Pooled::NONE, false);
        xk1.fold_sketch_left_mut(&s);
        let recon = contract(&xk, &xk1);
        assert_contracted_coeffs_close(
            &recon,
            &tc,
            1e-10,
            "factor_svd(normalize=false) reconstruction",
        );

        let (xk_n, s_n, mut xk1_n) = factor_svd(&tc, 1e-12, crate::qn::Pooled::NONE, true);
        xk1_n.fold_sketch_left_mut(&s_n);
        let recon_n = contract(&xk_n, &xk1_n);
        assert!(
            (recon_n.norm() - 1.0).abs() < 1e-10,
            "normalize=true reconstruction norm {}",
            recon_n.norm()
        );
        let scaled = &recon_n * tc_norm;
        assert_contracted_coeffs_close(
            &scaled,
            &tc,
            1e-9,
            "factor_svd normalize arms differ beyond the ‖tc‖ scale",
        );

        let (_, s_cap, _) = factor_svd(&tc, 1e-12, crate::qn::Pooled(2), false);
        for m in tc.mid_qn.clone() {
            if let Some(blk) = s_cap.block(m) {
                assert!(
                    blk.nrows() <= 2,
                    "mid sector {m:?}: rank {} exceeds cap 2",
                    blk.nrows()
                );
            }
        }
    }
}
