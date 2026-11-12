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

use super::sketch::Sketch;
use super::{ensure_seq, gemm};
use crate::core::{Family, SparseCore};
use crate::qn::{RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

#[inline]
fn unit_phase<T: Scalar>(z: T) -> T {
    if z == T::zero() {
        T::one()
    } else {
        z * T::from_f64(1.0 / z.abs_s())
    }
}

#[inline]
fn sign_scalar<T: Scalar>(z: T) -> T {
    if T::IS_COMPLEX {
        unit_phase(z)
    } else {
        T::from_f64(z.re_s().signum())
    }
}

#[inline]
fn frob_norm<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt()
}

#[inline]
fn sumabs2<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>()
}

/// First `c` Haar-random columns of an `m×m` unitary, by Stewart's algorithm (reflector sweep on
/// Gaussian vectors, sign-corrected identity slab). No faer factorization involved — a hand-rolled
/// reflector sweep, `O(m·c²)`. Requires `m ≥ c`. Test oracle: `‖Fᴴ·F − I_c‖ < 1e-12`.
pub fn stewart_frame<T: Scalar, R: Rng + ?Sized>(rng: &mut R, m: usize, c: usize) -> Mat<T> {
    assert!(m >= c, "stewart_frame requires m >= c");
    let mut q = Mat::<T>::zeros(m, c);
    if c == 0 {
        return q;
    }
    let nrefl = c.min(m - 1);
    let mut vs: Vec<Vec<T>> = Vec::with_capacity(nrefl);
    let mut betas: Vec<f64> = Vec::with_capacity(nrefl);

    for j in 0..nrefl {
        let len = m - j;
        let mut x: Vec<T> = (0..len).map(|_| T::sample_standard(rng)).collect();
        let s = unit_phase(x[0]);
        let sigma = frob_norm(&x);
        x[0] += s * T::from_f64(sigma);
        betas.push(2.0 / sumabs2(&x));
        *q.get_mut(j, j) = T::zero() - s;
        vs.push(x);
    }
    if c == m {
        *q.get_mut(m - 1, m - 1) = sign_scalar(T::sample_standard(rng));
    }

    for j in (0..nrefl).rev() {
        let v = &vs[j];
        let b = betas[j];
        for col in j..c {
            let mut r = T::zero();
            for i in j..m {
                r += v[i - j].conj_s() * *q.get(i, col);
            }
            let br = T::from_f64(b) * r;
            for i in j..m {
                let cur = *q.get(i, col);
                *q.get_mut(i, col) = cur - v[i - j] * br;
            }
        }
    }
    q
}

/// Rank-revealing pivoted QR of a single dense block:
/// `A ≈ Q·C` with `Q` (m×rank) column-orthonormal, `C = R·Pᵀ` unpermuted (rank×n), and
/// `rank = #{ i : |R[i,i]| > 16·|R[0,0]|·eps }` (a direct count, assuming no pivot monotonicity).
/// faer `col_piv_qr` + thin-Q slice stands in for LAPACK `geqp3`/`orgqr`; pivots may differ.
pub fn my_qc<T: Scalar>(a: MatRef<'_, T>) -> (Mat<T>, Mat<T>, usize) {
    my_qc_floor(a, 0.0)
}

/// [`my_qc`] with an ABSOLUTE pivot floor on top of the relative cut: `rank = #{ i : |R[i,i]| >
/// max(16·|R[0,0]|·eps, floor) }`. A residual window whose columns are all roundoff of a much
/// larger raw window has an `R[0,0]` that is itself roundoff, so a cut relative to it keeps a
/// noise pivot; the caller passes the raw scale.
pub fn my_qc_floor<T: Scalar>(a: MatRef<'_, T>, floor: f64) -> (Mat<T>, Mat<T>, usize) {
    let m = a.nrows();
    let n = a.ncols();
    if m == 0 || n == 0 {
        return (Mat::<T>::zeros(m, 0), Mat::<T>::zeros(0, n), 0);
    }

    ensure_seq();
    let qr = a.col_piv_qr();
    let r = qr.R();
    let size = m.min(n);

    let eps = (16.0 * r.get(0, 0).abs_s() * f64::EPSILON).max(floor);
    let mut rank = 0usize;
    for i in 0..size {
        if r.get(i, i).abs_s() > eps {
            rank += 1;
        }
    }

    let (fwd, _bwd) = qr.P().arrays();
    let mut c = Mat::<T>::zeros(rank, n);
    for (j, &jp) in fwd.iter().enumerate().take(n) {
        let imax = (j + 1).min(rank);
        for i in 0..imax {
            *c.get_mut(i, jp) = *r.get(i, j);
        }
    }

    let qfull = qr.compute_thin_Q();
    let mut q = Mat::<T>::zeros(m, rank);
    for j in 0..rank {
        for i in 0..m {
            *q.get_mut(i, j) = *qfull.get(i, j);
        }
    }
    (q, c, rank)
}

/// Row-orthonormal counterpart: `A ≈ C·Q` via `my_qc` of the
/// **plain transpose** (NOT adjoint — a construction choice picking which conjugate factor pair is
/// produced while keeping `A = C·Q` a plain product — see the module doc). Returns `(Q, C, rank)`
/// with `Q` (rank×n) Hermitian-row-orthonormal (`Q·Qᴴ = I`; for complex `T` `Q·Qᵀ ≠ I`) and
/// `C` (m×rank).
pub fn my_cq<T: Scalar>(a: MatRef<'_, T>) -> (Mat<T>, Mat<T>, usize) {
    let ta = a.transpose().to_owned();
    let (q, c, rank) = my_qc(ta.as_ref());
    (q.transpose().to_owned(), c.transpose().to_owned(), rank)
}

/// Thin (unpivoted) QR orthonormal factor `Q` of `a` (`m × min(m,n)`) — an orthonormal basis of
/// `col(a)`. faer-confined (`ensure_seq`).
pub fn thin_q<T: Scalar>(a: MatRef<'_, T>) -> Mat<T> {
    let (m, n) = (a.nrows(), a.ncols());
    if m == 0 || n == 0 {
        return Mat::<T>::zeros(m, 0);
    }
    ensure_seq();
    a.qr().compute_thin_Q()
}

/// A thin SVD `A = U·diag(s)·Vᴴ` of an `m×n` matrix: `u` is `m×size`, `v` is `n×size`
/// (columns are the right singular vectors, NOT transposed), `s` descending, `size = min(m,n)`.
pub struct ThinSvd<T: Scalar> {
    /// Left singular vectors, `m×size`.
    pub u: Mat<T>,
    /// Singular values, descending.
    pub s: Vec<f64>,
    /// Right singular vectors, `n×size` (so `A = u·diag(s)·vᴴ`).
    pub v: Mat<T>,
}

/// Thin SVD by bidiagonal **QR iteration only** — faer's divide-and-conquer branch is never
/// entered.
pub fn thin_svd_qr<T: Scalar>(a: MatRef<'_, T>) -> ThinSvd<T> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::svd::{ComputeSvdVectors, SvdParams, svd, svd_scratch};
    let (m, n) = (a.nrows(), a.ncols());
    let size = m.min(n);
    if size == 0 {
        return ThinSvd {
            u: Mat::<T>::zeros(m, 0),
            s: Vec::new(),
            v: Mat::<T>::zeros(n, 0),
        };
    }
    ensure_seq();
    let mut params = <SvdParams as faer::Auto<T>>::auto();
    params.recursion_threshold = usize::MAX;
    let spec: faer::Spec<SvdParams, T> = params.into();
    let mut s = faer::diag::Diag::<T>::zeros(size);
    let mut u = Mat::<T>::zeros(m, size);
    let mut v = Mat::<T>::zeros(n, size);
    let mut mem = MemBuffer::new(svd_scratch::<T>(
        m,
        n,
        ComputeSvdVectors::Thin,
        ComputeSvdVectors::Thin,
        faer::Par::Seq,
        spec,
    ));
    svd(
        a,
        s.as_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        faer::Par::Seq,
        MemStack::new(&mut mem),
        spec,
    )
    .unwrap_or_else(|e| panic!("thin SVD (QR iteration) of a {m}×{n} block: {e:?}"));
    let sc = s.column_vector();
    let svals: Vec<f64> = (0..size).map(|i| sc.get(i).re_s()).collect();
    ThinSvd { u, s: svals, v }
}

/// Singular VALUES only, by the same bidiagonal QR iteration as [`thin_svd_qr`]
/// (`recursion_threshold = usize::MAX`) with no vectors formed — the inner loop of the
/// mode-rotation angle search, which scores ~16 rotated two-site blocks per bond and keeps none of
/// their factors. Descending, `len == min(m, n)`.
pub fn svals_qr<T: Scalar>(a: MatRef<'_, T>) -> Vec<f64> {
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::svd::{ComputeSvdVectors, SvdParams, svd, svd_scratch};
    let (m, n) = (a.nrows(), a.ncols());
    let size = m.min(n);
    if size == 0 {
        return Vec::new();
    }
    ensure_seq();
    let mut params = <SvdParams as faer::Auto<T>>::auto();
    params.recursion_threshold = usize::MAX;
    let spec: faer::Spec<SvdParams, T> = params.into();
    let mut s = faer::diag::Diag::<T>::zeros(size);
    let mut mem = MemBuffer::new(svd_scratch::<T>(
        m,
        n,
        ComputeSvdVectors::No,
        ComputeSvdVectors::No,
        faer::Par::Seq,
        spec,
    ));
    svd(
        a,
        s.as_mut(),
        None,
        None,
        faer::Par::Seq,
        MemStack::new(&mut mem),
        spec,
    )
    .unwrap_or_else(|e| panic!("singular values (QR iteration) of a {m}×{n} block: {e:?}"));
    let sc = s.column_vector();
    (0..size).map(|i| sc.get(i).re_s()).collect()
}

/// Thin-SVD split `a = U·Σ·Vᴴ`: left singular vectors `U` (`m × size`), singular values (`len
/// size`, descending), and `Vt = Vᴴ` (`size × n`), where `size = min(m,n)`. faer-confined
/// (`ensure_seq`); the `(U, Σ, Vt)` bridge `factor_svd` (`core::twosite`, MALS) needs to
/// build the diagonal `Sketch` for `chop_energy` and scatter the truncated `U`/`Vt`. Companion of
/// `svd_u_svals` that additionally returns `Vt` (reuses the private `svd_blocks` extractor).
pub fn svd_uvt_svals<T: Scalar>(a: MatRef<'_, T>) -> (Mat<T>, Vec<f64>, Mat<T>) {
    let (m, n) = (a.nrows(), a.ncols());
    let size = m.min(n);
    if size == 0 {
        return (Mat::<T>::zeros(m, 0), Vec::new(), Mat::<T>::zeros(0, n));
    }
    let ThinSvd { u, s, v } = thin_svd_qr(a);
    debug_assert_eq!((u.nrows(), u.ncols()), (m, size));
    let vt = svd_vt(v.as_ref(), n, size);
    (u, s, vt)
}

impl<T: Scalar> SparseCore<T> {
    pub(crate) fn horizontal_width(&self, l: Sector) -> usize {
        let g = self.space.pg.irrep_of(self.k);
        let mut w = 0usize;
        for f in Family::ALL {
            let t = f.apply(l, g);
            if self.space.contains(self.k + 1, t) {
                w += self.col_ranks[t];
            }
        }
        w
    }

    pub(crate) fn vertical_height(&self, r: Sector) -> usize {
        let g = self.space.pg.irrep_of(self.k);
        let mut h = 0usize;
        for f in Family::ALL {
            let Some(s) = f.unapply(r, g) else {
                continue;
            };
            if self.space.contains(self.k, s) {
                h += self.row_ranks[s];
            }
        }
        h
    }

    /// Horizontal unfolding of the row sector: `[ ○○ | up | dn | ●● ]` over admissible
    /// column sectors in family order, shape `row_rank × Σ col_ranks` (order is load-bearing).
    pub fn horizontal_unfolding(&self, row_sector: Sector) -> Mat<T> {
        let l = row_sector;
        let m = self.row_ranks[l];
        let width = self.horizontal_width(l);
        let g = self.space.pg.irrep_of(self.k);
        let mut out = Mat::<T>::zeros(m, width);
        let mut off = 0usize;
        for f in Family::ALL {
            let t = f.apply(l, g);
            if !self.space.contains(self.k + 1, t) {
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

    /// Vertical unfolding of the column sector: `[ ○○ ; up ; dn ; ●● ]` over admissible
    /// row sectors in family order, shape `Σ row_ranks × col_rank` (order is load-bearing;
    /// electron decrements guard `usize` underflow).
    pub fn vertical_unfolding(&self, col_sector: Sector) -> Mat<T> {
        let r = col_sector;
        let n = self.col_ranks[r];
        let height = self.vertical_height(r);
        let g = self.space.pg.irrep_of(self.k);
        let mut out = Mat::<T>::zeros(height, n);
        let mut off = 0usize;
        for f in Family::ALL {
            let Some(s) = f.unapply(r, g) else {
                continue;
            };
            if !self.space.contains(self.k, s) {
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

    /// Inverse of [`Self::horizontal_unfolding`]: scatter `src` back into the four families of the row
    /// sector at cumulative column offsets.
    pub fn scatter_horizontal(&mut self, row_sector: Sector, src: MatRef<'_, T>) {
        let l = row_sector;
        let m = self.row_ranks[l];
        debug_assert_eq!(src.nrows(), m, "scatter_horizontal: row count mismatch");
        let g = self.space.pg.irrep_of(self.k);
        let mut off = 0usize;
        for f in Family::ALL {
            let t = f.apply(l, g);
            if !self.space.contains(self.k + 1, t) {
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

    /// Inverse of [`Self::vertical_unfolding`]: scatter `src` back into the four families of the column
    /// sector at cumulative row offsets.
    pub fn scatter_vertical(&mut self, col_sector: Sector, src: MatRef<'_, T>) {
        let r = col_sector;
        let n = self.col_ranks[r];
        debug_assert_eq!(src.ncols(), n, "scatter_vertical: col count mismatch");
        let g = self.space.pg.irrep_of(self.k);
        let mut off = 0usize;
        for f in Family::ALL {
            let Some(s) = f.unapply(r, g) else {
                continue;
            };
            if !self.space.contains(self.k, s) {
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

    /// Rank-revealing vertical QC: per column sector `my_qc` of the vertical unfolding. Returns `(Q
    /// core, C sketch @ site+1, reduced col ranks)`; `A ≈ Q·C`.
    pub fn qc(&self) -> (SparseCore<T>, Sketch<T>, RankGrid) {
        let space = self.space;
        let k = self.k;
        let mut col_ranks = RankGrid::zeros(&space);
        let mut qblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);
        let mut cblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);

        for r in space.bond_sectors(k + 1) {
            let av = self.vertical_unfolding(r);
            let (q, c, rank) = my_qc(av.as_ref());
            col_ranks[r] = rank;
            qblocks[r] = Some(q);
            cblocks[r] = Some(c);
        }

        let mut q_core = SparseCore::zeros(space, k, &self.row_ranks, &col_ranks);
        for r in space.bond_sectors(k + 1) {
            let qb = qblocks[r].take().unwrap();
            q_core.scatter_vertical(r, qb.as_ref());
        }
        let c_sketch = Sketch::new(space, k + 1, cblocks);
        (q_core, c_sketch, col_ranks)
    }

    /// Rank-revealing horizontal CQ: per row sector `my_cq` of the horizontal unfolding. Returns
    /// `(Q core, C sketch @ site, reduced row ranks)`; `A ≈ C·Q`.
    pub fn cq(&self) -> (SparseCore<T>, Sketch<T>, RankGrid) {
        let space = self.space;
        let k = self.k;
        let mut row_ranks = RankGrid::zeros(&space);
        let mut qblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);
        let mut cblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);

        for l in space.bond_sectors(k) {
            let ah = self.horizontal_unfolding(l);
            let (q, c, rank) = my_cq(ah.as_ref());
            row_ranks[l] = rank;
            qblocks[l] = Some(q);
            cblocks[l] = Some(c);
        }

        let mut q_core = SparseCore::zeros(space, k, &row_ranks, &self.col_ranks);
        for l in space.bond_sectors(k) {
            let qb = qblocks[l].take().unwrap();
            q_core.scatter_horizontal(l, qb.as_ref());
        }
        let c_sketch = Sketch::new(space, k, cblocks);
        (q_core, c_sketch, row_ranks)
    }

    /// Rank-preserving vertical QR: non-pivoted thin QR, zero- padding `Q` columns to keep block
    /// ranks fixed. Mutates `self` in place to hold `Q`; returns the upper-triangular `R` as a
    /// [`Sketch`] at bond `k+1`.
    pub fn qr_fixed(&mut self) -> Sketch<T> {
        ensure_seq();
        let space = self.space;
        let k = self.k;
        let mut r_sketch = Sketch::zeros(space, k + 1, &self.col_ranks, &self.col_ranks);

        for r in space.bond_sectors(k + 1) {
            let n = self.col_ranks[r];
            let av = self.vertical_unfolding(r);
            let m = av.nrows();
            if m == 0 || n == 0 {
                continue;
            }
            let qr = av.as_ref().qr();
            let rank = m.min(n);
            let thin_q = qr.compute_thin_Q();
            let thin_r = qr.thin_R();

            {
                let mut rb = r_sketch.block_mut(r).unwrap();
                for j in 0..n {
                    for i in 0..rank {
                        *rb.rb_mut().get_mut(i, j) = *thin_r.get(i, j);
                    }
                }
            }
            let mut padded = Mat::<T>::zeros(m, n);
            for j in 0..rank {
                for i in 0..m {
                    *padded.get_mut(i, j) = *thin_q.get(i, j);
                }
            }
            self.scatter_vertical(r, padded.as_ref());
        }
        r_sketch
    }

    /// Rank-preserving horizontal LQ: mirror of [`Self::qr_fixed`] via the plain-transpose construction
    /// (matches `my_cq`, NOT adjoint — this only selects the conjugate factor pair; the resulting
    /// `Q` rows are Hermitian-orthonormal, `Q·Qᴴ = I`). Returns `L` sketch at bond `k`.
    pub fn lq_fixed(&mut self) -> Sketch<T> {
        ensure_seq();
        let space = self.space;
        let k = self.k;
        let mut l_sketch = Sketch::zeros(space, k, &self.row_ranks, &self.row_ranks);

        for l in space.bond_sectors(k) {
            let m = self.row_ranks[l];
            let ah = self.horizontal_unfolding(l);
            let n = ah.ncols();
            if m == 0 || n == 0 {
                continue;
            }
            let t = ah.transpose().to_owned();
            let qr = t.as_ref().qr();
            let rank = m.min(n);
            let thin_q = qr.compute_thin_Q();
            let thin_r = qr.thin_R();
            {
                let mut lb = l_sketch.block_mut(l).unwrap();
                for j in 0..rank {
                    for i in 0..m {
                        *lb.rb_mut().get_mut(i, j) = *thin_r.get(j, i);
                    }
                }
            }
            let mut padded = Mat::<T>::zeros(m, n);
            for j in 0..n {
                for i in 0..rank {
                    *padded.get_mut(i, j) = *thin_q.get(j, i);
                }
            }
            self.scatter_horizontal(l, padded.as_ref());
        }
        l_sketch
    }

    /// Per-row-sector horizontal SVD. Returns `(U sketch, S sketch, raw Vt grid)`; the zero-rank
    /// edge sets `Vt` to `0 × Σ col_ranks`.
    pub fn svd_horizontal(&self) -> (Sketch<T>, Sketch<T>, SectorGrid<Option<Mat<T>>>) {
        ensure_seq();
        let space = self.space;
        let k = self.k;
        let mut ublocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);
        let mut sblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);
        let mut vtblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);

        for l in space.bond_sectors(k) {
            let m = self.row_ranks[l];
            let width = self.horizontal_width(l);
            let size = m.min(width);
            if size > 0 {
                let ah = self.horizontal_unfolding(l);
                let svd = thin_svd_qr(ah.as_ref());
                let (u, s, vt) = svd_blocks(svd, m, width, size);
                ublocks[l] = Some(u);
                sblocks[l] = Some(s);
                vtblocks[l] = Some(vt);
            } else {
                ublocks[l] = Some(Mat::<T>::zeros(m, 0));
                sblocks[l] = Some(Mat::<T>::zeros(0, 0));
                vtblocks[l] = Some(Mat::<T>::zeros(0, width));
            }
        }
        (
            Sketch::new(space, k, ublocks),
            Sketch::new(space, k, sblocks),
            vtblocks,
        )
    }

    /// Per-column-sector vertical SVD. Returns `(raw U grid, S sketch, Vt sketch)`; the zero-rank
    /// edge sets `U` to `Σ row_ranks × 0`.
    pub fn svd_vertical(&self) -> (SectorGrid<Option<Mat<T>>>, Sketch<T>, Sketch<T>) {
        ensure_seq();
        let space = self.space;
        let k = self.k;
        let mut ublocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);
        let mut sblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);
        let mut vtblocks: SectorGrid<Option<Mat<T>>> = SectorGrid::from_fn(&space, |_| None);

        for r in space.bond_sectors(k + 1) {
            let n = self.col_ranks[r];
            let height = self.vertical_height(r);
            let size = height.min(n);
            if size > 0 {
                let av = self.vertical_unfolding(r);
                let svd = thin_svd_qr(av.as_ref());
                let (u, s, vt) = svd_blocks(svd, height, n, size);
                ublocks[r] = Some(u);
                sblocks[r] = Some(s);
                vtblocks[r] = Some(vt);
            } else {
                ublocks[r] = Some(Mat::<T>::zeros(height, 0));
                sblocks[r] = Some(Mat::<T>::zeros(0, 0));
                vtblocks[r] = Some(Mat::<T>::zeros(0, n));
            }
        }
        (
            ublocks,
            Sketch::new(space, k + 1, sblocks),
            Sketch::new(space, k + 1, vtblocks),
        )
    }

    /// [`Self::svd_horizontal`] packaged back into a core for `svdvals`. Returns `(U sketch, S sketch, core carrying Vt)`.
    pub fn svd_horizontal_to_core(&self) -> (Sketch<T>, Sketch<T>, SparseCore<T>) {
        let space = self.space;
        let k = self.k;
        let (u, s, vt) = self.svd_horizontal();
        let mut c = SparseCore::zeros(space, k, &s.col_ranks(), &self.col_ranks);
        for l in space.bond_sectors(k) {
            if let Some(vtb) = vt[l].as_ref() {
                c.scatter_horizontal(l, vtb.as_ref());
            }
        }
        (u, s, c)
    }

    /// New core with the top-left sub-blocks of `self` truncated to `row_ranks ≤` / `col_ranks ≤`
    /// the current ranks — used by `round_global_mut`.
    pub fn reduce_ranks(&self, row_ranks: &RankGrid, col_ranks: &RankGrid) -> SparseCore<T> {
        let space = self.space;
        let k = self.k;
        for s in space.bond_sectors(k) {
            debug_assert!(row_ranks[s] <= self.row_ranks[s], "reduce_ranks: row grows");
        }
        for s in space.bond_sectors(k + 1) {
            debug_assert!(col_ranks[s] <= self.col_ranks[s], "reduce_ranks: col grows");
        }
        let mut b = SparseCore::zeros(space, k, row_ranks, col_ranks);
        let g = space.pg.irrep_of(k);
        for l in space.bond_sectors(k) {
            for f in Family::ALL {
                let t = f.apply(l, g);
                if !space.contains(k + 1, t) {
                    continue;
                }
                let (rr, cc) = (row_ranks[l], col_ranks[t]);
                let (Some(src), Some(mut dst)) = (self.block(f, l), b.block_mut(f, l)) else {
                    continue;
                };
                for j in 0..cc {
                    for i in 0..rr {
                        *dst.rb_mut().get_mut(i, j) = *src.get(i, j);
                    }
                }
            }
        }
        b
    }

    /// `self ← A · self`: fold the block-diagonal sketch into the row side of all four families
    /// (requires `A` square with `row_ranks(A) == col_ranks(A) == row_ranks(self)`).
    pub fn fold_sketch_left_mut(&mut self, a: &Sketch<T>) {
        let k = self.k;
        let g = self.space.pg.irrep_of(k);
        for l in self.row_qn.clone() {
            let Some(ablk) = a.block(l) else { continue };
            for f in Family::ALL {
                let t = f.apply(l, g);
                if !self.space.contains(k + 1, t) {
                    continue;
                }
                let Some(blk) = self.block(f, l) else {
                    continue;
                };
                debug_assert!(
                    ablk.nrows() == blk.nrows() && ablk.ncols() == blk.nrows(),
                    "fold_sketch_left_mut: A block at {l:?} must be square, dims == row_rank(self)"
                );
                let prod = gemm(ablk, blk);
                let mut dst = self.block_mut(f, l).unwrap();
                let (m, n) = (prod.nrows(), prod.ncols());
                for j in 0..n {
                    for i in 0..m {
                        *dst.rb_mut().get_mut(i, j) = *prod.get(i, j);
                    }
                }
            }
        }
    }

    /// `self ← self · B`: fold the sketch into the column side, selecting `B`'s block by the
    /// shifted target column sector.
    pub fn fold_sketch_right_mut(&mut self, b: &Sketch<T>) {
        let k = self.k;
        let g = self.space.pg.irrep_of(k);
        for l in self.row_qn.clone() {
            for f in Family::ALL {
                let t = f.apply(l, g);
                if !self.space.contains(k + 1, t) {
                    continue;
                }
                let (Some(blk), Some(bblk)) = (self.block(f, l), b.block(t)) else {
                    continue;
                };
                debug_assert!(
                    bblk.nrows() == blk.ncols() && bblk.ncols() == blk.ncols(),
                    "fold_sketch_right_mut: B block at {t:?} must be square, dims == col_rank(self)"
                );
                let prod = gemm(blk, bblk);
                let mut dst = self.block_mut(f, l).unwrap();
                let (m, n) = (prod.nrows(), prod.ncols());
                for j in 0..n {
                    for i in 0..m {
                        *dst.rb_mut().get_mut(i, j) = *prod.get(i, j);
                    }
                }
            }
        }
    }
}

fn svd_blocks<T: Scalar>(
    svd: ThinSvd<T>,
    m: usize,
    n: usize,
    size: usize,
) -> (Mat<T>, Mat<T>, Mat<T>) {
    debug_assert_eq!((svd.u.nrows(), svd.u.ncols()), (m, size));
    let mut s = Mat::<T>::zeros(size, size);
    for i in 0..size {
        *s.get_mut(i, i) = T::from_f64(svd.s[i]);
    }
    let vt = svd_vt(svd.v.as_ref(), n, size);
    (svd.u, s, vt)
}

fn svd_vt<T: Scalar>(vf: MatRef<'_, T>, n: usize, size: usize) -> Mat<T> {
    debug_assert_eq!((vf.nrows(), vf.ncols()), (n, size));
    let mut vt = Mat::<T>::zeros(size, n);
    for j in 0..n {
        for i in 0..size {
            *vt.get_mut(i, j) = vf.get(j, i).conj_s();
        }
    }
    vt
}

#[cfg(test)]
mod thin_svd_qr_tests {
    use super::{ThinSvd, thin_svd_qr};
    use faer::Mat;

    fn fixture() -> Mat<f64> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata/svd/faer_dc_nan_190x197.bin");
        let bytes =
            std::fs::read(&path).expect("tracked fixture testdata/svd/faer_dc_nan_190x197.bin");
        let m = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let n = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        assert_eq!((m, n), (190, 197));
        Mat::from_fn(m, n, |i, j| {
            let o = 16 + 8 * (j * m + i);
            f64::from_le_bytes(bytes[o..o + 8].try_into().unwrap())
        })
    }

    fn nonfinite(x: faer::MatRef<'_, f64>) -> usize {
        (0..x.ncols())
            .flat_map(|j| (0..x.nrows()).map(move |i| (i, j)))
            .filter(|&(i, j)| !x.get(i, j).is_finite())
            .count()
    }

    fn residual(a: &Mat<f64>, svd: &ThinSvd<f64>) -> (f64, f64) {
        let k = svd.s.len();
        let mut r = a.clone();
        let mut na = 0.0f64;
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                let acc: f64 = (0..k)
                    .map(|l| svd.u.get(i, l) * svd.s[l] * svd.v.get(j, l))
                    .sum();
                *r.get_mut(i, j) -= acc;
                na += a.get(i, j).powi(2);
            }
        }
        let nr: f64 = (0..r.ncols())
            .flat_map(|j| (0..r.nrows()).map(move |i| (i, j)))
            .map(|(i, j)| r.get(i, j).powi(2))
            .sum();
        let mut orth = 0.0f64;
        for a_ in 0..k {
            for b in 0..k {
                let dot: f64 = (0..a.nrows())
                    .map(|i| svd.u.get(i, a_) * svd.u.get(i, b))
                    .sum();
                orth += (dot - if a_ == b { 1.0 } else { 0.0 }).powi(2);
            }
        }
        ((nr / na).sqrt(), orth.sqrt())
    }

    #[test]
    fn qr_iteration_factors_the_dc_nan_block_exactly() {
        let a = fixture();
        let svd = thin_svd_qr(a.as_ref());
        assert_eq!((svd.u.ncols(), svd.v.ncols(), svd.s.len()), (190, 190, 190));
        assert_eq!(nonfinite(svd.u.as_ref()) + nonfinite(svd.v.as_ref()), 0);
        assert!(
            svd.s.windows(2).all(|w| w[0] >= w[1]),
            "singular values descending"
        );
        let (res, orth) = residual(&a, &svd);
        assert!(res < 1e-13, "‖A − UΣVᵀ‖/‖A‖ = {res:.3e}");
        assert!(orth < 1e-12, "‖UᵀU − I‖ = {orth:.3e}");
    }

    #[test]
    fn faer_default_thin_svd_still_returns_nan_vectors_on_the_block() {
        let a = fixture();
        let svd = a
            .as_ref()
            .thin_svd()
            .expect("faer default thin_svd returns Ok on this block");
        let bad = nonfinite(svd.U()) + nonfinite(svd.V());
        assert!(
            bad > 0,
            "faer's divide-and-conquer SVD returned finite singular vectors on the fixture \
             block — the upstream defect appears fixed; retire thin_svd_qr and this pin"
        );
    }

    #[test]
    fn qr_iteration_matches_faer_default_on_well_conditioned_blocks() {
        let mut state: u64 = 0x5EED;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        for &(m, n) in &[(7usize, 5usize), (5, 9), (40, 40), (150, 200), (200, 150)] {
            let a = Mat::<f64>::from_fn(m, n, |_, _| next());
            let ours = thin_svd_qr(a.as_ref());
            let theirs = a.as_ref().thin_svd().unwrap();
            let sf = theirs.S().column_vector();
            for i in 0..m.min(n) {
                assert!((ours.s[i] - sf.get(i)).abs() <= 1e-13 * ours.s[0], "σ[{i}]");
            }
            let (res, orth) = residual(&a, &ours);
            assert!(
                res < 1e-13 && orth < 1e-12,
                "{m}×{n}: res {res:.2e} orth {orth:.2e}"
            );
        }
    }
}

#[cfg(test)]
mod phase6_tests {}
