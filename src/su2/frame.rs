// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) bond-frame `Sketch`. Block-diagonal per-sector bond matrix — the SU(2) analogue of
//! `crate::core::factor::Sketch`, keyed on `Su2Space`/ `Su2Sector`/`Su2RankGrid`/`Su2SectorGrid`
//! instead of the abelian `QnSpace`/`Sector`/ `RankGrid`/`SectorGrid`.

use faer::prelude::ReborrowMut;
use faer::{Mat, MatMut, MatRef};

use crate::core::factor::gemm;
use crate::scalar::Scalar;
use crate::su2::qn::{
    LabelTables, StateSym, Su2RankGrid, Su2SectorGrid, Su2Space, SymGrid, SymLabel, SymSpace,
    Symmetry,
};

/// Block-diagonal bond-frame factor. See the module docs for how this differs from a memory-backed
/// / type-tagged frame.
#[derive(Clone, Debug)]
pub struct Su2Sketch<T: Scalar, S: Symmetry = StateSym> {
    /// Symmetry descriptor (needed for canonical-order sector iteration).
    pub space: S::Space,
    /// Bond index this sketch lives on.
    pub k: usize,
    /// One optional block per sector, keyed by the label
    /// ([`Su2Sector`](crate::su2::qn::Su2Sector) for the state);
    /// `None` = structurally absent.
    pub blocks: S::Grid<Option<Mat<T>>>,
}

impl<T: Scalar> Su2Sketch<T> {
    /// Adopt a block grid at bond `k` — the state façade of [`Self::new_sym`].
    #[inline]
    pub fn new(space: Su2Space, k: usize, blocks: Su2SectorGrid<Option<Mat<T>>>) -> Self {
        Self::new_sym(space, k, blocks)
    }

    /// Block-diagonal zeros — the state façade of [`Self::zeros_sym`].
    #[inline]
    pub fn zeros(
        space: Su2Space,
        k: usize,
        row_ranks: &Su2RankGrid,
        col_ranks: &Su2RankGrid,
    ) -> Self {
        Self::zeros_sym(space, k, row_ranks, col_ranks)
    }

    /// Rank-`r` identity frame at bond `k` — the state façade of [`Self::identity_sym`].
    #[inline]
    pub fn identity(space: Su2Space, k: usize, r: usize) -> Self {
        Self::identity_sym(space, k, r)
    }
}

impl<T: Scalar, S: Symmetry> Su2Sketch<T, S> {
    /// Adopt a block grid at bond `k` (the blocks-adopting constructor;
    /// ranks inferred from block dimensions). The `issetequal(findall(assigned), qn)`
    /// presence check is a `debug_assert` (project convention; cf.
    /// [`crate::su2::core::Su2Core::from_blocks`]): present-block set == admissible-sector set at
    /// bond `k` — every admissible sector carries a block and no inadmissible sector does.
    pub fn new_sym(space: S::Space, k: usize, blocks: S::Grid<Option<Mat<T>>>) -> Self {
        debug_assert!(
            blocks
                .iter()
                .all(|(s, b)| b.is_some() == space.admissible(k, s)),
            "Su2Sketch::new: block presence at bond {k} disagrees with admissibility"
        );
        Su2Sketch { space, k, blocks }
    }

    /// Block-diagonal zeros with the given per-sector row/col ranks.
    pub fn zeros_sym(
        space: S::Space,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
    ) -> Self {
        let blocks = <S::Grid<Option<Mat<T>>>>::from_fn(&space, |s| {
            space
                .admissible(k, s)
                .then(|| Mat::<T>::zeros(row_ranks[s], col_ranks[s]))
        });
        Su2Sketch { space, k, blocks }
    }

    /// Rank-`r` identity frame at bond `k`: the `r×r` identity at every **reachable** sector, `0×0`
    /// at the admissible-but-unreachable ones. Callers pass `r` explicitly.
    pub fn identity_sym(space: S::Space, k: usize, r: usize) -> Self {
        let caps = (!space.all_admissible_reachable()).then(|| S::max_ranks(&space));
        let blocks = <S::Grid<Option<Mat<T>>>>::from_fn(&space, |s| {
            space.admissible(k, s).then(|| {
                let n = if caps.as_ref().is_none_or(|c| c[k][s] > 0) {
                    r
                } else {
                    0
                };
                let mut m = Mat::<T>::zeros(n, n);
                for i in 0..n {
                    *m.get_mut(i, i) = T::one();
                }
                m
            })
        });
        Su2Sketch { space, k, blocks }
    }

    /// The bond index.
    #[inline]
    pub fn site(&self) -> usize {
        self.k
    }

    /// Admissible sectors of this bond, canonical order (`n` fastest, `2s` outer).
    #[inline]
    pub fn sectors(&self) -> Vec<S::Label> {
        self.space.bond_labels(self.k)
    }

    /// The block at sector `s`, if present.
    pub fn block(&self, s: S::Label) -> Option<MatRef<'_, T>> {
        if !self.blocks.in_bounds(s) {
            return None;
        }
        self.blocks[s].as_ref().map(|m| m.as_ref())
    }

    /// Mutable block access.
    pub fn block_mut(&mut self, s: S::Label) -> Option<MatMut<'_, T>> {
        if !self.blocks.in_bounds(s) {
            return None;
        }
        self.blocks[s].as_mut().map(|m| m.as_mut())
    }

    /// Per-sector row ranks (block row counts).
    pub fn row_ranks(&self) -> S::Grid<usize> {
        self.block_dims(|b| b.nrows())
    }

    /// Per-sector column ranks (block column counts).
    pub fn col_ranks(&self) -> S::Grid<usize> {
        self.block_dims(|b| b.ncols())
    }

    #[allow(dead_code)]
    pub(crate) fn row_ranks_in(&self, tables: &LabelTables<S>) -> S::Grid<usize> {
        self.block_dims_in(tables, |b| b.nrows())
    }

    pub(crate) fn col_ranks_in(&self, tables: &LabelTables<S>) -> S::Grid<usize> {
        self.block_dims_in(tables, |b| b.ncols())
    }

    fn block_dims_in(
        &self,
        tables: &LabelTables<S>,
        dim: impl Fn(&Mat<T>) -> usize,
    ) -> S::Grid<usize> {
        if *tables.space() != self.space {
            return self.block_dims(dim);
        }
        let mut r = <S::Grid<usize>>::filled(&self.space, 0);
        for &l in tables.labels(self.k) {
            if let Some(b) = &self.blocks[l] {
                r[l] = dim(b);
            }
        }
        r
    }

    fn block_dims(&self, dim: impl Fn(&Mat<T>) -> usize) -> S::Grid<usize> {
        let mut r = <S::Grid<usize>>::filled(&self.space, 0);
        for (v, (s, b)) in r.values_mut().iter_mut().zip(self.blocks.iter()) {
            if let Some(b) = b
                && self.space.admissible(self.k, s)
            {
                *v = dim(b);
            }
        }
        r
    }

    /// THE degeneracy-weighted Frobenius norm: `sqrt(Σ_sector centerweight(2s,2S)·Σ|block|²)`
    /// via [`crate::su2::centerweight`]. **DIVERGES from the abelian `Sketch::norm`** (plain
    /// unweighted Frobenius) — do not collapse the two shapes. A `Su2Sketch` at bond `k` is a
    /// center matrix in the fixed-Sz-member normalization; a raw block at `(n,2s)` stands for a
    /// `(2s+1)`-fold spin multiplet, so its true contribution to the fixed-Sz vector norm² is
    /// `centerweight(2s,2S)·‖block‖²`.
    pub fn norm(&self) -> f64 {
        let mut acc = 0.0f64;
        for s in self.sectors() {
            if let Some(m) = self.block(s) {
                let cw = crate::su2::centerweight(s.two_j(), self.space.total_two_j());
                let mut block_sum = 0.0f64;
                for j in 0..m.ncols() {
                    for i in 0..m.nrows() {
                        let v = *m.get(i, j);
                        block_sum += v.abs_s() * v.abs_s();
                    }
                }
                acc += cw * block_sum;
            }
        }
        acc.sqrt()
    }

    /// Scale every block by a scalar.
    pub fn scale_mut(&mut self, a: T) {
        for s in self.sectors() {
            if let Some(mut b) = self.block_mut(s) {
                for j in 0..b.ncols() {
                    for i in 0..b.nrows() {
                        let c = b.rb_mut().get_mut(i, j);
                        *c *= a;
                    }
                }
            }
        }
    }

    /// `self ← self + other` per sector — twin of the abelian `Sketch::add_assign_sketch`, and the
    /// accumulator the SU(2) tangent-space Hessian environments need (`su2::hess`: each `L̃`/`R̃`
    /// step sums a transported partial with a freshly inserted component term). Sectors
    /// where either side is structurally absent are skipped: `None` IS a zero block. Shapes must
    /// agree wherever both are present (debug-asserted) — a mismatch means the two sketches came
    /// from different bra/ket rank grids.
    pub fn add_assign_sketch(&mut self, other: &Self) {
        debug_assert_eq!(self.k, other.k, "add_assign_sketch: bond mismatch");
        for s in self.sectors() {
            let Some(ob) = other.block(s) else {
                continue;
            };
            let Some(mut sb) = self.block_mut(s) else {
                continue;
            };
            debug_assert_eq!(
                (sb.nrows(), sb.ncols()),
                (ob.nrows(), ob.ncols()),
                "add_assign_sketch: block shape mismatch at sector {s:?}"
            );
            for j in 0..sb.ncols() {
                for i in 0..sb.nrows() {
                    *sb.rb_mut().get_mut(i, j) += *ob.get(i, j);
                }
            }
        }
    }

    /// `self ← self · b` per sector. Used by the rounding sweeps (`U := U·S`, folding the diagonal `S`
    /// sketch into `U`). Each `b` block must be square with dim == the contracted rank (same
    /// `debug_assert` discipline as the abelian `Sketch::rmul_sketch_mut`).
    pub fn rmul_sketch_mut(&mut self, b: &Self) {
        debug_assert_eq!(self.k, b.k, "rmul_sketch_mut: bond mismatch");
        for s in self.sectors() {
            let (Some(self_blk), Some(bb)) = (self.block(s), b.block(s)) else {
                continue;
            };
            debug_assert!(
                bb.nrows() == self_blk.ncols() && bb.ncols() == self_blk.ncols(),
                "rmul_sketch_mut: B block at {s:?} must be square, dims == col_rank(self)"
            );
            let prod = gemm(self_blk, bb);
            let mut dst = self.block_mut(s).unwrap();
            let (m, n) = (prod.nrows(), prod.ncols());
            for j in 0..n {
                for i in 0..m {
                    *dst.rb_mut().get_mut(i, j) = *prod.get(i, j);
                }
            }
        }
    }
}

impl<T: Scalar, S: Symmetry> crate::mem::Footprint for Su2Sketch<T, S> {
    fn bytes(&self) -> u64 {
        self.blocks
            .iter()
            .filter_map(|(_, b)| b.as_ref())
            .map(|m| crate::mem::block_bytes::<T>(m.nrows(), m.ncols()))
            .sum()
    }
}

#[cfg(test)]
mod spg2_tests {
    use super::*;

    #[test]
    fn spg2_sketch_identity_unchanged_at_trivial_group() {
        let d = 4;
        let space = Su2Space::new(d, 3, 1);
        let r = 2usize;
        let sketch = Su2Sketch::<f64>::identity(space, 1, r);
        for s in space.bond_sectors(1) {
            let block = sketch.block(s).unwrap();
            assert_eq!((block.nrows(), block.ncols()), (r, r));
            for i in 0..r {
                assert_eq!(*block.get(i, i), 1.0);
            }
        }
    }
}

#[cfg(test)]
mod s2_tests {
    use super::*;
    use crate::su2::qn::Su2Sector;

    #[test]
    fn tst2_su2sketch_norm_is_centerweighted() {
        let space = Su2Space::new(6, 5, 1);
        let k = 2usize;
        let sa = Su2Sector::new(2, 0);
        let sb = Su2Sector::new(2, 2);
        let mut row = Su2RankGrid::zeros(&space);
        let mut col = Su2RankGrid::zeros(&space);
        row[sa] = 2;
        col[sa] = 2;
        row[sb] = 2;
        col[sb] = 2;
        let mut sk = Su2Sketch::<f64>::zeros(space, k, &row, &col);
        {
            let mut ba = sk.block_mut(sa).unwrap();
            *ba.rb_mut().get_mut(0, 0) = 1.0;
            *ba.rb_mut().get_mut(1, 1) = 1.0;
        }
        {
            let mut bb = sk.block_mut(sb).unwrap();
            *bb.rb_mut().get_mut(0, 0) = 2.0;
        }

        let want = 7.0f64.sqrt();
        assert!(
            (sk.norm() - want).abs() < 1e-13,
            "Su2Sketch::norm={} != centerweighted sqrt(7)={want}",
            sk.norm()
        );
        assert!(
            (sk.norm() - 6.0f64.sqrt()).abs() > 0.1,
            "Su2Sketch::norm must NOT collapse to the unweighted Frobenius norm"
        );
    }
}

#[cfg(test)]
mod ranks_in_tests {}
