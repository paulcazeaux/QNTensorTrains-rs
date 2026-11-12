// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, MatRef, Par};

use rayon::prelude::*;

use crate::core::{Family, SparseCore};
use crate::par;
use crate::qn::{RankGrid, Sector};
use crate::scalar::Scalar;

#[inline]
pub(crate) fn block_dot_cols_into<T: Scalar>(
    acc: &mut T,
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    j0: usize,
    j1: usize,
) {
    debug_assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "block_dot: shape"
    );
    match (a.try_as_col_major(), b.try_as_col_major()) {
        (Some(ac), Some(bc)) if par::block_slice() => {
            for j in j0..j1 {
                for (x, y) in ac.col(j).as_slice().iter().zip(bc.col(j).as_slice()) {
                    *acc += x.conj_s() * *y;
                }
            }
        }
        _ => {
            for j in j0..j1 {
                for i in 0..a.nrows() {
                    *acc += a.get(i, j).conj_s() * *b.get(i, j);
                }
            }
        }
    }
}

#[inline]
pub(crate) fn block_dot_into<T: Scalar>(acc: &mut T, a: MatRef<'_, T>, b: MatRef<'_, T>) {
    block_dot_cols_into(acc, a, b, 0, a.ncols());
}

#[inline]
pub(crate) fn block_dot_cols<T: Scalar>(
    a: MatRef<'_, T>,
    b: MatRef<'_, T>,
    j0: usize,
    j1: usize,
) -> T {
    let mut acc = T::zero();
    block_dot_cols_into(&mut acc, a, b, j0, j1);
    acc
}

#[inline]
pub(crate) fn block_dot<T: Scalar>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> T {
    block_dot_cols(a, b, 0, a.ncols())
}

#[inline]
pub(crate) fn block_abs2_into<T: Scalar>(acc: &mut f64, a: MatRef<'_, T>) {
    match a.try_as_col_major() {
        Some(ac) if par::block_slice() => {
            for j in 0..ac.ncols() {
                for x in ac.col(j).as_slice() {
                    *acc += x.abs_s() * x.abs_s();
                }
            }
        }
        _ => {
            for j in 0..a.ncols() {
                for i in 0..a.nrows() {
                    let v = *a.get(i, j);
                    *acc += v.abs_s() * v.abs_s();
                }
            }
        }
    }
}

#[inline]
pub(crate) fn block_abs2<T: Scalar>(a: MatRef<'_, T>) -> f64 {
    let mut acc = 0.0f64;
    block_abs2_into(&mut acc, a);
    acc
}

#[inline]
pub(crate) fn block_axpy<T: Scalar>(mut dst: MatMut<'_, T>, src: MatRef<'_, T>, alpha: T) {
    let (m, n) = (dst.nrows(), dst.ncols());
    debug_assert_eq!((m, n), (src.nrows(), src.ncols()), "block_axpy: shape");
    match (dst.rb_mut().try_as_col_major_mut(), src.try_as_col_major()) {
        (Some(mut dc), Some(sc)) if par::block_slice() => {
            for j in 0..n {
                for (d, s) in dc
                    .rb_mut()
                    .col_mut(j)
                    .as_slice_mut()
                    .iter_mut()
                    .zip(sc.col(j).as_slice())
                {
                    *d += alpha * *s;
                }
            }
        }
        _ => {
            for j in 0..n {
                for i in 0..m {
                    let v = *src.get(i, j);
                    *dst.rb_mut().get_mut(i, j) += alpha * v;
                }
            }
        }
    }
}

#[inline]
pub(crate) fn block_scale<T: Scalar>(mut dst: MatMut<'_, T>, alpha: T) {
    let (m, n) = (dst.nrows(), dst.ncols());
    match dst.rb_mut().try_as_col_major_mut() {
        Some(mut dc) if par::block_slice() => {
            for j in 0..n {
                for d in dc.rb_mut().col_mut(j).as_slice_mut() {
                    *d *= alpha;
                }
            }
        }
        _ => {
            for j in 0..n {
                for i in 0..m {
                    *dst.rb_mut().get_mut(i, j) *= alpha;
                }
            }
        }
    }
}

pub(crate) fn step_dot<T: Scalar>(a: &SparseCore<T>, b: &SparseCore<T>, family: Family) -> T {
    let (ab, bb) = (a.store().blocks(), b.store().blocks());
    let mut acc = T::zero();
    for i in a.step_range(family) {
        block_dot_into(&mut acc, ab[i].as_ref(), bb[i].as_ref());
    }
    acc
}

pub(crate) fn step_abs2<T: Scalar>(a: &SparseCore<T>, family: Family) -> f64 {
    let ab = a.store().blocks();
    let mut acc = 0.0f64;
    for i in a.step_range(family) {
        block_abs2_into(&mut acc, ab[i].as_ref());
    }
    acc
}

#[inline]
pub(crate) fn assert_same_blocks<T: Scalar>(a: &SparseCore<T>, b: &SparseCore<T>, what: &str) {
    assert!(
        a.store().same_keys(b.store()),
        "{what}: the two cores store different blocks at site {} (their rank grids differ)",
        a.k
    );
}

fn add_rank_grids(a: &RankGrid, b: &RankGrid) -> RankGrid {
    let mut r = a.clone();
    for (s, &v) in b.iter() {
        r[s] += v;
    }
    r
}

impl<T: Scalar> SparseCore<T> {
    /// Block-concatenation add `A ⊕ B`: boundary cores concatenate only along the non-boundary bond
    /// (columns at `k == 0`, rows at `k == d-1`), interior cores are block-diagonal in both bonds,
    /// and a `d == 1` train adds elementwise. Both operands share identical sector structure (a
    /// pure function of `(Nup,Ndn,d,k)`), so there is no presence-merge — zero-size blocks are
    /// natural no-ops.
    pub fn oplus(&self, other: &SparseCore<T>) -> SparseCore<T> {
        assert_eq!(self.space, other.space, "oplus: QnSpace mismatch");
        assert_eq!(self.k, other.k, "oplus: site mismatch");
        let space = self.space;
        let d = space.d;
        let k = self.k;

        let (row_ranks, col_ranks) = if d == 1 {
            assert_eq!(
                self.row_ranks, other.row_ranks,
                "oplus d==1: row ranks differ"
            );
            assert_eq!(
                self.col_ranks, other.col_ranks,
                "oplus d==1: col ranks differ"
            );
            (self.row_ranks.clone(), self.col_ranks.clone())
        } else if k == 0 {
            assert_eq!(
                self.row_ranks, other.row_ranks,
                "oplus k==0: row ranks differ"
            );
            (
                self.row_ranks.clone(),
                add_rank_grids(&self.col_ranks, &other.col_ranks),
            )
        } else if k == d - 1 {
            assert_eq!(
                self.col_ranks, other.col_ranks,
                "oplus k==d-1: col ranks differ"
            );
            (
                add_rank_grids(&self.row_ranks, &other.row_ranks),
                self.col_ranks.clone(),
            )
        } else {
            (
                add_rank_grids(&self.row_ranks, &other.row_ranks),
                add_rank_grids(&self.col_ranks, &other.col_ranks),
            )
        };

        let mut out = SparseCore::zeros(space, k, &row_ranks, &col_ranks);

        for f in Family::ALL {
            let secs: Vec<Sector> = self.transitions(f).collect();
            for s in secs {
                let a = self.block(f, s).expect("A transition present");
                let (ma, na) = (a.nrows(), a.ncols());
                let b = other.block(f, s).expect("B transition present");
                let (mb, nb) = (b.nrows(), b.ncols());
                let (row_off, col_off) = if d == 1 {
                    (0, 0)
                } else if k == 0 {
                    (0, na)
                } else if k == d - 1 {
                    (ma, 0)
                } else {
                    (ma, na)
                };
                let mut dst = out.block_mut(f, s).unwrap();
                for j in 0..na {
                    let acol = a
                        .col(j)
                        .try_as_col_major()
                        .expect("col-major block")
                        .as_slice();
                    let dcol = dst
                        .rb_mut()
                        .col_mut(j)
                        .try_as_col_major_mut()
                        .expect("col-major block")
                        .as_slice_mut();
                    dcol[..ma].copy_from_slice(&acol[..ma]);
                }
                for j in 0..nb {
                    let bcol = b
                        .col(j)
                        .try_as_col_major()
                        .expect("col-major block")
                        .as_slice();
                    let dcol = dst
                        .rb_mut()
                        .col_mut(col_off + j)
                        .try_as_col_major_mut()
                        .expect("col-major block")
                        .as_slice_mut();
                    if d == 1 {
                        for i in 0..mb {
                            dcol[i] += bcol[i];
                        }
                    } else {
                        dcol[row_off..row_off + mb].copy_from_slice(&bcol[..mb]);
                    }
                }
            }
        }
        out
    }

    /// Constant-chain augmentation `A ⊕ b`: adds one rank per present sector on each non-boundary
    /// bond; the new slot carries the scalar `b` at the first core (value `1` elsewhere), realizing
    /// `tt + b·𝟙`.
    pub fn oplus_scalar(&self, b: T) -> SparseCore<T> {
        let space = self.space;
        let d = space.d;
        let k = self.k;
        let caps = (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space));
        let reachable = |bond: usize, s: Sector| caps.as_ref().is_none_or(|c| c[bond][s] > 0);

        let mut row_ranks = self.row_ranks.clone();
        let mut col_ranks = self.col_ranks.clone();
        if k > 0 {
            for &s in &self.row_qn {
                if reachable(k, s) {
                    row_ranks[s] += 1;
                }
            }
        }
        if k < d - 1 {
            for &s in &self.col_qn {
                if reachable(k + 1, s) {
                    col_ranks[s] += 1;
                }
            }
        }

        let mut out = SparseCore::zeros(space, k, &row_ranks, &col_ranks);
        let v = if k == 0 { b } else { T::one() };

        let g = space.pg.irrep_of(k);
        for f in Family::ALL {
            let secs: Vec<Sector> = self.transitions(f).collect();
            for s in secs {
                let t = f.apply(s, g);
                let (ma, na) = {
                    let a = self.block(f, s).expect("transition present");
                    (a.nrows(), a.ncols())
                };
                {
                    let a = self.block(f, s).unwrap();
                    let mut dst = out.block_mut(f, s).unwrap();
                    for j in 0..na {
                        for i in 0..ma {
                            *dst.rb_mut().get_mut(i, j) = *a.get(i, j);
                        }
                    }
                }
                let row_slot = k > 0 && reachable(k, s);
                let col_slot = k < d - 1 && reachable(k + 1, t);
                if d == 1 {
                    let mut dst = out.block_mut(f, s).unwrap();
                    for j in 0..na {
                        for i in 0..ma {
                            *dst.rb_mut().get_mut(i, j) += b;
                        }
                    }
                } else if k == 0 {
                    if ma > 0 && col_slot {
                        *out.block_mut(f, s).unwrap().get_mut(0, na) = v;
                    }
                } else if k == d - 1 {
                    if na > 0 && row_slot {
                        *out.block_mut(f, s).unwrap().get_mut(ma, 0) = v;
                    }
                } else {
                    if row_slot && col_slot {
                        *out.block_mut(f, s).unwrap().get_mut(ma, na) = v;
                    }
                }
            }
        }
        out
    }

    /// Scale every block of every family in place (the stored ones; an empty block has nothing
    /// to scale).
    pub fn scale_mut(&mut self, a: T) {
        for m in self.store_mut().blocks_mut() {
            block_scale(m.as_mut(), a);
        }
    }

    /// Scale every block of a SINGLE family `f` in place, leaving the other three untouched.
    pub fn scale_family_mut(&mut self, f: Family, a: T) {
        let secs: Vec<Sector> = self.transitions(f).collect();
        for s in secs {
            let blk = self.block_mut(f, s).expect("transition present");
            block_scale(blk, a);
        }
    }

    /// Negation `-A` (a scaled deep copy).
    pub fn neg(&self) -> SparseCore<T> {
        let mut out = self.clone();
        out.scale_mut(T::from_f64(-1.0));
        out
    }

    /// Per-sector Kronecker product `A ⊗ B`: ranks multiply per sector. The `(m_a, m_b)` dimensions
    /// merge with the `A` index fastest, so `row = ia + ib·m_a`, `col = ja + jb·n_a`, `entry =
    /// a[ia,ja]·b[ib,jb]`.
    pub fn kron(&self, other: &SparseCore<T>) -> SparseCore<T> {
        assert_eq!(self.space, other.space, "kron: QnSpace mismatch");
        assert_eq!(self.k, other.k, "kron: site mismatch");
        let space = self.space;
        let k = self.k;

        let mut row_ranks = RankGrid::zeros(&space);
        let mut col_ranks = RankGrid::zeros(&space);
        for (s, &ra) in self.row_ranks.iter() {
            row_ranks[s] = ra * other.row_ranks[s];
        }
        for (s, &ca) in self.col_ranks.iter() {
            col_ranks[s] = ca * other.col_ranks[s];
        }

        let mut out = SparseCore::zeros(space, k, &row_ranks, &col_ranks);

        for f in Family::ALL {
            let secs: Vec<Sector> = self.transitions(f).collect();
            for s in secs {
                let (ma, na) = {
                    let a = self.block(f, s).expect("A transition present");
                    (a.nrows(), a.ncols())
                };
                let (mb, nb) = {
                    let b = other.block(f, s).expect("B transition present");
                    (b.nrows(), b.ncols())
                };
                let a = self.block(f, s).unwrap();
                let b = other.block(f, s).unwrap();
                let mut dst = out.block_mut(f, s).unwrap();
                for ja in 0..na {
                    for jb in 0..nb {
                        for ia in 0..ma {
                            for ib in 0..mb {
                                let a_ij = *a.get(ia, ja);
                                let b_ij = *b.get(ib, jb);
                                let row = ia + ib * ma;
                                let col = ja + jb * na;
                                *dst.rb_mut().get_mut(row, col) = a_ij * b_ij;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// Squared Frobenius norm `‖A‖²`: sum of
    /// `|·|²` over all four families in `row_qn` order (○○, up, dn, ●● within each sector).
    pub fn abs2(&self) -> f64 {
        let mut s = 0.0f64;
        for &sec in &self.row_qn {
            for f in Family::ALL {
                if let Some(b) = self.block(f, sec) {
                    block_abs2_into(&mut s, b);
                }
            }
        }
        s
    }

    /// Frobenius norm `‖A‖ = sqrt(abs2(A))`.
    pub fn norm(&self) -> f64 {
        self.abs2().sqrt()
    }

    /// Hermitian per-block inner product `⟨A, B⟩`, conjugating `self` (the first argument), summed
    /// in ○○, up, dn, ●● order.
    pub fn dot_cores(&self, other: &SparseCore<T>) -> T {
        let mut acc = T::zero();
        for &sec in &self.row_qn {
            for f in Family::ALL {
                if let (Some(a), Some(b)) = (self.block(f, sec), other.block(f, sec)) {
                    block_dot_into(&mut acc, a, b);
                }
            }
        }
        acc
    }

    /// `self += a · other`, elementwise over shared blocks — for the
    /// `InnerProductVector` trait (Davidson's Ritz-vector rebuild). Precondition: `self` and
    /// `other` share row/col ranks.
    pub fn axpy(&mut self, a: T, other: &SparseCore<T>) {
        for f in Family::ALL {
            let secs: Vec<Sector> = self.transitions(f).collect();
            for s in secs {
                let ob = other
                    .block(f, s)
                    .expect("axpy: other transition present (same shape)");
                let sb = self.block_mut(f, s).expect("axpy: self transition present");
                debug_assert_eq!(
                    (sb.nrows(), sb.ncols()),
                    (ob.nrows(), ob.ncols()),
                    "axpy: block shape mismatch"
                );
                block_axpy(sb, ob, a);
            }
        }
    }

    pub(crate) fn dot_cores_par(&self, other: &SparseCore<T>) -> T {
        assert_same_blocks(self, other, "dot_cores_par");
        if !par::par_vector_ops(self.coeff_hint()) {
            return self.dot_cores(other);
        }
        let parts: Vec<T> = (0..4usize)
            .into_par_iter()
            .map(|f| step_dot(self, other, Family::ALL[f]))
            .collect();
        parts.into_iter().fold(T::zero(), |acc, p| acc + p)
    }

    pub(crate) fn abs2_par(&self) -> f64 {
        if !par::par_vector_ops(self.coeff_hint()) {
            return self.abs2();
        }
        let parts: Vec<f64> = (0..4usize)
            .into_par_iter()
            .map(|f| step_abs2(self, Family::ALL[f]))
            .collect();
        parts.into_iter().sum()
    }

    pub(crate) fn norm_par(&self) -> f64 {
        self.abs2_par().sqrt()
    }

    pub(crate) fn axpy_par(&mut self, alpha: T, other: &SparseCore<T>) {
        assert_same_blocks(self, other, "axpy_par");
        if !par::par_vector_ops(self.coeff_hint()) {
            self.axpy(alpha, other);
            return;
        }
        self.store_mut()
            .blocks_mut()
            .par_iter_mut()
            .zip(other.store().blocks().par_iter())
            .for_each(|(sb, ob)| block_axpy(sb.as_mut(), ob.as_ref(), alpha));
    }

    pub(crate) fn scale_mut_par(&mut self, alpha: T) {
        if !par::par_vector_ops(self.coeff_hint()) {
            self.scale_mut(alpha);
            return;
        }
        self.store_mut()
            .blocks_mut()
            .par_iter_mut()
            .for_each(|sb| block_scale(sb.as_mut(), alpha));
    }

    /// Visit every coefficient by value in canonical order — the read-only twin of
    /// [`Self::for_each_coeff_mut`], walking exactly the blocks [`Self::num_coeffs`] counts and in
    /// the order [`Self::coeff`] indexes. `O(num_coeffs)`, unlike a `coeff(i)` loop, which re-walks
    /// every block per index; that is what makes it usable for serialization
    /// ([`crate::tt_io`]) at solver-scale ranks.
    pub fn for_each_coeff<F: FnMut(T)>(&self, mut f: F) {
        for fam in Family::ALL {
            for s in self.transitions(fam) {
                if let Some(b) = self.block(fam, s) {
                    let (m, n) = (b.nrows(), b.ncols());
                    for j in 0..n {
                        for i in 0..m {
                            f(*b.get(i, j));
                        }
                    }
                }
            }
        }
    }

    /// Visit every coefficient mutably in canonical order (metric-free raw entries).
    pub fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, mut f: F) {
        for fam in Family::ALL {
            let secs: Vec<Sector> = self.transitions(fam).collect();
            for s in secs {
                let mut b = self.block_mut(fam, s).expect("transition present");
                let (m, n) = (b.nrows(), b.ncols());
                for j in 0..n {
                    for i in 0..m {
                        f(b.rb_mut().get_mut(i, j));
                    }
                }
            }
        }
    }

    /// Zip two same-shaped cores' coefficients in canonical order: `f(&mut a_i, b_i)`.
    /// Precondition: `self` and `other` share row/col ranks.
    pub fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &SparseCore<T>, mut f: F) {
        for fam in Family::ALL {
            let secs: Vec<Sector> = self.transitions(fam).collect();
            for s in secs {
                let ob = other.block(fam, s).expect("zip_coeffs_mut: same shape");
                let mut b = self.block_mut(fam, s).expect("transition present");
                let (m, n) = (b.nrows(), b.ncols());
                debug_assert_eq!((m, n), (ob.nrows(), ob.ncols()), "zip_coeffs_mut: shape");
                for j in 0..n {
                    for i in 0..m {
                        let ov = *ob.get(i, j);
                        f(b.rb_mut().get_mut(i, j), ov);
                    }
                }
            }
        }
    }

    /// Read the coefficient at a flat canonical index (`0..num_coeffs()`).
    pub fn coeff(&self, idx: usize) -> T {
        let mut off = 0usize;
        for fam in Family::ALL {
            for s in self.transitions(fam) {
                if let Some(b) = self.block(fam, s) {
                    let (m, n) = (b.nrows(), b.ncols());
                    let sz = m * n;
                    if sz != 0 && idx < off + sz {
                        let local = idx - off;
                        return *b.get(local % m, local / m);
                    }
                    off += sz;
                }
            }
        }
        panic!("coeff: index {idx} out of range 0..{off}");
    }

    /// Write the coefficient at a flat canonical index.
    pub fn set_coeff(&mut self, idx: usize, val: T) {
        let mut off = 0usize;
        for fam in Family::ALL {
            let secs: Vec<Sector> = self.transitions(fam).collect();
            for s in secs {
                let (m, n) = match self.block(fam, s) {
                    Some(b) => (b.nrows(), b.ncols()),
                    None => continue,
                };
                let sz = m * n;
                if sz != 0 && idx < off + sz {
                    let local = idx - off;
                    *self
                        .block_mut(fam, s)
                        .expect("transition present")
                        .get_mut(local % m, local / m) = val;
                    return;
                }
                off += sz;
            }
        }
        panic!("set_coeff: index {idx} out of range 0..{off}");
    }

    /// Number of coefficients (sum of present-block sizes) — summed straight off the store, whose
    /// blocks are exactly the non-empty admissible slots (empty ones add 0); the transition walk
    /// is re-checked in debug builds.
    pub fn num_coeffs(&self) -> usize {
        let n: usize = self
            .store()
            .blocks()
            .iter()
            .map(|b| b.nrows() * b.ncols())
            .sum();
        #[cfg(debug_assertions)]
        {
            let mut walk = 0usize;
            for fam in Family::ALL {
                for s in self.transitions(fam) {
                    if let Some(b) = self.block(fam, s) {
                        walk += b.nrows() * b.ncols();
                    }
                }
            }
            debug_assert_eq!(n, walk, "SparseCore::num_coeffs: store and walk disagree");
        }
        n
    }

    /// Conjugate every block in place. Identity
    /// for real `T`.
    pub fn conj_mut(&mut self) {
        for f in Family::ALL {
            let secs: Vec<Sector> = self.transitions(f).collect();
            for s in secs {
                let mut blk = self.block_mut(f, s).expect("transition present");
                let (m, n) = (blk.nrows(), blk.ncols());
                for j in 0..n {
                    for i in 0..m {
                        let cell = blk.rb_mut().get_mut(i, j);
                        *cell = cell.conj_s();
                    }
                }
            }
        }
    }

    /// Fold a boundary matrix into the first core from the left: each block becomes `a · block`.
    pub fn lmul_boundary_mut(&mut self, a: MatRef<'_, T>) {
        assert_eq!(self.k, 0, "lmul_boundary_mut requires the first core");
        for f in Family::ALL {
            let secs: Vec<Sector> = self.transitions(f).collect();
            for s in secs {
                let (m, n) = {
                    let b = self.block(f, s).expect("transition present");
                    (b.nrows(), b.ncols())
                };
                if m == 0 || n == 0 {
                    continue;
                }
                let mut tmp = Mat::<T>::zeros(m, n);
                {
                    let b = self.block(f, s).unwrap();
                    matmul(tmp.as_mut(), Accum::Replace, a, b, T::one(), Par::Seq);
                }
                let mut dst = self.block_mut(f, s).unwrap();
                for j in 0..n {
                    for i in 0..m {
                        *dst.rb_mut().get_mut(i, j) = *tmp.get(i, j);
                    }
                }
            }
        }
    }

    /// Fold a boundary matrix into the last core from the right: each block becomes `block · b`.
    pub fn rmul_boundary_mut(&mut self, b: MatRef<'_, T>) {
        assert_eq!(
            self.k,
            self.space.d - 1,
            "rmul_boundary_mut requires the last core"
        );
        for f in Family::ALL {
            let secs: Vec<Sector> = self.transitions(f).collect();
            for s in secs {
                let (m, n) = {
                    let blk = self.block(f, s).expect("transition present");
                    (blk.nrows(), blk.ncols())
                };
                if m == 0 || n == 0 {
                    continue;
                }
                let mut tmp = Mat::<T>::zeros(m, n);
                {
                    let blk = self.block(f, s).unwrap();
                    matmul(tmp.as_mut(), Accum::Replace, blk, b, T::one(), Par::Seq);
                }
                let mut dst = self.block_mut(f, s).unwrap();
                for j in 0..n {
                    for i in 0..m {
                        *dst.rb_mut().get_mut(i, j) = *tmp.get(i, j);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod phase2_tests {
    use crate::TtVector;
    use crate::qn::QnSpace;
    use crate::test_util::randomized_state;
    use faer::Mat;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn boundary_matrix_folds_scale_image() {
        let d = 4;
        let space = QnSpace::from_n_two_sz(d, 2, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(800);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let xd = x.to_dense();
        let two = Mat::<f64>::from_fn(1, 1, |_, _| 2.0);

        let mut xl = x.clone();
        xl.cores[0].lmul_boundary_mut(two.as_ref());
        xl.check();
        for (got, &orig) in xl.to_dense().iter().zip(&xd) {
            assert!(
                (got - 2.0 * orig).abs() < 1e-13,
                "lmul: {got} vs {}",
                2.0 * orig
            );
        }

        let mut xr = x.clone();
        xr.cores[d - 1].rmul_boundary_mut(two.as_ref());
        xr.check();
        for (got, &orig) in xr.to_dense().iter().zip(&xd) {
            assert!(
                (got - 2.0 * orig).abs() < 1e-13,
                "rmul: {got} vs {}",
                2.0 * orig
            );
        }
    }

    #[test]
    fn d1_oplus_and_oplus_scalar_match_dense() {
        for (i, &(nup, ndn)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
            let space = QnSpace::new(1, nup, ndn);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(1500 + i as u64);
            let x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
            let y: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
            let (xd, yd) = (x.to_dense(), y.to_dense());

            let sum = &x + &y;
            sum.check();
            let want_sum: Vec<f64> = xd.iter().zip(&yd).map(|(&a, &b)| a + b).collect();
            for (g, w) in sum.to_dense().iter().zip(&want_sum) {
                assert!(
                    (g - w).abs() < 1e-13,
                    "d=1 x+y (nup={nup},ndn={ndn}): {g} vs {w}"
                );
            }

            let c = 0.9;
            let ones = crate::tt_ones::<f64>(space).to_dense();
            let shifted = x.add_scalar(c);
            shifted.check();
            let want_shift: Vec<f64> = xd.iter().zip(&ones).map(|(&a, &o)| a + c * o).collect();
            for (g, w) in shifted.to_dense().iter().zip(&want_shift) {
                assert!(
                    (g - w).abs() < 1e-13,
                    "d=1 x+c (nup={nup},ndn={ndn}): {g} vs {w}"
                );
            }
        }
    }
}
