// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::prelude::ReborrowMut;
use faer::{Mat, MatRef};
use rand::Rng;

use crate::scalar::Scalar;
use crate::su2::qn::{SymLabel, SymSpace, SymStep, Symmetry};

use super::Su2Core;

impl<T: Scalar, S: Symmetry> Su2Core<T, S> {
    pub(crate) fn horizontal_width(&self, l: S::Label) -> usize {
        let mut w = 0usize;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            if let Some(t) = f.step_apply(l, g)
                && self.space.admissible(self.k + 1, t)
            {
                w += self.col_ranks[t];
            }
        }
        w
    }

    pub(crate) fn vertical_height(&self, r: S::Label) -> usize {
        let mut h = 0usize;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            let Some(s) = f.step_unapply(r, g) else {
                continue;
            };
            if !self.space.admissible(self.k, s) {
                continue;
            }
            h += self.row_ranks[s];
        }
        h
    }

    /// Horizontal (`:R`) unfolding of the row sector: `hcat[ ○○ | up | dn | ●● ]` over admissible
    /// column sectors in family order. Shape
    /// `row_rank × horizontal_width(l)`. UNWEIGHTED — θ is a separate step, see
    /// [`Self::scale_horizontal`].
    pub fn horizontal_unfolding(&self, l: S::Label) -> Mat<T> {
        let m = self.row_ranks[l];
        let width = self.horizontal_width(l);
        let mut out = Mat::<T>::zeros(m, width);
        let mut off = 0usize;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !self.space.admissible(self.k + 1, t) {
                continue;
            }
            let w = self.col_ranks[t];
            if let Some(b) = self.block(f, l) {
                for j in 0..w {
                    for i in 0..m {
                        *out.get_mut(i, off + j) = *b.get(i, j);
                    }
                }
            }
            off += w;
        }
        out
    }

    /// Vertical (`:L`) unfolding of the column sector: `vcat[ ○○ ; up ; dn ; ●● ]` over admissible
    /// incoming row sectors in family order (the §6.2 shifted-adjoint incoming lookups: `up` from
    /// `(n−1,2s−1)`, `dn` from `(n−1,2s+1)` — note the `+1`). Shape
    /// `vertical_height(r) × col_rank`. UNWEIGHTED.
    pub fn vertical_unfolding(&self, r: S::Label) -> Mat<T> {
        let n = self.col_ranks[r];
        let height = self.vertical_height(r);
        let mut out = Mat::<T>::zeros(height, n);
        let mut off = 0usize;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            let Some(s) = f.step_unapply(r, g) else {
                continue;
            };
            if !self.space.admissible(self.k, s) {
                continue;
            }
            let h = self.row_ranks[s];
            if let Some(b) = self.block(f, s) {
                for j in 0..n {
                    for i in 0..h {
                        *out.get_mut(off + i, j) = *b.get(i, j);
                    }
                }
            }
            off += h;
        }
        out
    }

    /// Inverse of [`Self::horizontal_unfolding`]: scatter `src` back into the four families of the
    /// row sector at cumulative column offsets.
    pub fn scatter_horizontal(&mut self, l: S::Label, src: MatRef<'_, T>) {
        let m = self.row_ranks[l];
        debug_assert_eq!(src.nrows(), m, "scatter_horizontal: row count mismatch");
        let mut off = 0usize;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !self.space.admissible(self.k + 1, t) {
                continue;
            }
            let w = self.col_ranks[t];
            if let Some(mut b) = self.block_mut(f, l) {
                for j in 0..w {
                    for i in 0..m {
                        *b.rb_mut().get_mut(i, j) = *src.get(i, off + j);
                    }
                }
            }
            off += w;
        }
    }

    /// Inverse of [`Self::vertical_unfolding`]: scatter `src` back into the four families of the
    /// column sector at cumulative row offsets.
    pub fn scatter_vertical(&mut self, r: S::Label, src: MatRef<'_, T>) {
        let n = self.col_ranks[r];
        debug_assert_eq!(src.ncols(), n, "scatter_vertical: col count mismatch");
        let mut off = 0usize;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            let Some(s) = f.step_unapply(r, g) else {
                continue;
            };
            if !self.space.admissible(self.k, s) {
                continue;
            }
            let h = self.row_ranks[s];
            if let Some(mut b) = self.block_mut(f, s) {
                for j in 0..n {
                    for i in 0..h {
                        *b.rb_mut().get_mut(i, j) = *src.get(off + i, j);
                    }
                }
            }
            off += h;
        }
    }

    /// θ-weight the column slices of a materialized horizontal unfolding IN PLACE: row spin is
    /// `l.two_j()` (the state's `2s`); each admissible target family's column slice (same family
    /// order/offsets as [`Self::horizontal_unfolding`]) is multiplied by
    /// `su2::theta(l.two_j(), t.two_j())` — θ reads the two labels' `two_j` and nothing else.
    /// `Empty`/`Occ` slices have `θ=1` and are left unchanged (the `w==1 &&
    /// continue`). `x` must have `x.ncols() == self.horizontal_width(l)`.
    pub fn scale_horizontal(&self, x: &mut Mat<T>, l: S::Label) {
        self.scale_horizontal_impl(x, l, false);
    }

    fn scale_horizontal_impl(&self, x: &mut Mat<T>, l: S::Label, inverse: bool) {
        let rows = x.nrows();
        let mut off = 0usize;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !self.space.admissible(self.k + 1, t) {
                continue;
            }
            let w = self.col_ranks[t];
            let theta = crate::su2::theta(l.two_j(), t.two_j());
            if theta != 1.0 {
                let factor = T::from_f64(if inverse { 1.0 / theta } else { theta });
                for j in off..off + w {
                    for i in 0..rows {
                        let c = x.get_mut(i, j);
                        *c *= factor;
                    }
                }
            }
            off += w;
        }
    }

    /// Inverse of [`Self::scale_horizontal`]: divides each column-sector slice by
    /// `su2::theta(l.two_j(), t.two_j())`. Applied to the
    /// ORTHOGONAL factor (`Q`/`Vt`) on write-back; the triangular/singular factor (`C`/`L`/`Σ`)
    /// stays RAW.
    pub fn unscale_horizontal(&self, x: &mut Mat<T>, l: S::Label) {
        self.scale_horizontal_impl(x, l, true);
    }

    /// Scale a materialized vertical unfolding IN PLACE by the single scalar
    /// `√centerweight(r.two_j(), space.total_two_j())` — the whole
    /// unfolding shares one column sector, so this is a uniform scalar, not a per-slice weight
    /// (unlike the horizontal direction).
    pub fn scale_vertical(&self, x: &mut Mat<T>, r: S::Label) {
        let factor =
            T::from_f64(crate::su2::centerweight(r.two_j(), self.space.total_two_j()).sqrt());
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let c = x.get_mut(i, j);
                *c *= factor;
            }
        }
    }

    /// Inverse of [`Self::scale_vertical`]: divides by
    /// `√centerweight(r.two_j(), space.total_two_j())`.
    /// Applied to the pushed `Σ·Vt` factor on write-back (`round_lr_mut`); `U` stays RAW.
    pub fn unscale_vertical(&self, x: &mut Mat<T>, r: S::Label) {
        let factor =
            T::from_f64(1.0 / crate::su2::centerweight(r.two_j(), self.space.total_two_j()).sqrt());
        for j in 0..x.ncols() {
            for i in 0..x.nrows() {
                let c = x.get_mut(i, j);
                *c *= factor;
            }
        }
    }

    /// `orthogonal == :left` Stewart fill: per COLUMN sector, draw an `m × min(m,c)` Haar frame
    /// (`crate::core::factor::stewart_frame`, reused verbatim) into the vertical unfolding, scaled
    /// by `√(m/c)` (`m = vertical_height(r)`, `c = col_ranks[r]`), zero-padded if `m<c`. UNWEIGHTED
    /// (no θ) — direct SU(2) analogue of `crate::core::SparseCore::fill_stewart_left`
    /// (`src/core/mod.rs`) with SU(2) 4-family connectivity substituted.
    pub fn fill_stewart_left<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let space = self.space;
        let k = self.k;
        for t in space.bond_labels(k + 1) {
            let c = self.col_ranks[t];
            if c == 0 {
                continue;
            }
            let m = self.vertical_height(t);
            if m == 0 {
                continue;
            }
            let scale = T::from_f64((m as f64 / c as f64).sqrt());
            let cols = c.min(m);
            let frame = crate::core::factor::stewart_frame::<T, R>(rng, m, cols);
            let mut f = Mat::<T>::zeros(m, c);
            for jj in 0..cols {
                for ii in 0..m {
                    *f.get_mut(ii, jj) = *frame.get(ii, jj) * scale;
                }
            }
            self.scatter_vertical(t, f.as_ref());
        }
    }

    /// `orthogonal == :right` Stewart fill: per ROW sector, draw an `r × min(n,r)`-transposed Haar
    /// frame into the horizontal unfolding, scaled by `√(n/r)` (`n = horizontal_width(l)`, `r =
    /// row_ranks[l]`), zero-padded if `n<r`, THEN `unscale_horizontal`-folded (the one θ hook)
    /// before scatter. Direct SU(2) analogue of `crate::core::SparseCore::fill_stewart_right`,
    /// θ≡1 there.
    pub fn fill_stewart_right<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let space = self.space;
        let k = self.k;
        for l in space.bond_labels(k) {
            let r = self.row_ranks[l];
            if r == 0 {
                continue;
            }
            let n = self.horizontal_width(l);
            if n == 0 {
                continue;
            }
            let scale = T::from_f64((n as f64 / r as f64).sqrt());
            let cols = r.min(n);
            let frame = crate::core::factor::stewart_frame::<T, R>(rng, n, cols);
            let mut f = Mat::<T>::zeros(r, n);
            for ii in 0..cols {
                for jj in 0..n {
                    *f.get_mut(ii, jj) = *frame.get(jj, ii) * scale;
                }
            }
            self.unscale_horizontal(&mut f, l);
            self.scatter_horizontal(l, f.as_ref());
        }
    }
}
