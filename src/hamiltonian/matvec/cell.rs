// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, MatRef, unzip, zip};

use crate::hamiltonian::csx::CsxBlock;
use crate::hamiltonian::instrument;
use crate::scalar::{Scalar, gemm_par};

/// A contiguous slice of a cell's OUTPUT COLUMN ORDINALS, used to split one oversized output
/// block across several tasks (`BLOCK_SPLIT`).
#[derive(Clone, Copy, Debug)]
pub struct ColShard {
    lo: u32,
    hi: u32,
    off: usize,
}

impl ColShard {
    /// The whole block — the unsharded path.
    pub const ALL: ColShard = ColShard {
        lo: 0,
        hi: u32::MAX,
        off: 0,
    };
    pub fn new(lo: u32, hi: u32, off: usize) -> Self {
        ColShard { lo, hi, off }
    }
}

pub(crate) type RowShard = ColShard;

pub(crate) trait BandKeep: Copy {
    fn keeps(&self, id: u32) -> bool;
    fn off(&self) -> usize;
    #[allow(dead_code)]
    fn is_all(&self) -> bool;
    #[inline]
    fn skips_cell(&self, _out_ids: &[i32]) -> bool {
        false
    }
}

impl BandKeep for ColShard {
    #[inline]
    fn keeps(&self, id: u32) -> bool {
        id >= self.lo && id < self.hi
    }
    #[inline]
    fn off(&self) -> usize {
        self.off
    }
    #[inline]
    fn is_all(&self) -> bool {
        self.lo == 0 && self.hi == u32::MAX && self.off == 0
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BandMask<'a> {
    words: &'a [u64],
}

impl<'a> BandMask<'a> {
    pub(crate) fn new(words: &'a [u64]) -> Self {
        BandMask { words }
    }

    pub(crate) fn words_of(ids: &[usize]) -> Vec<u64> {
        let n = ids.iter().max().map_or(0, |&m| m / 64 + 1);
        let mut words = vec![0u64; n];
        for &id in ids {
            words[id / 64] |= 1u64 << (id % 64);
        }
        words
    }
}

impl BandKeep for BandMask<'_> {
    #[inline]
    fn keeps(&self, id: u32) -> bool {
        self.words
            .get((id / 64) as usize)
            .is_some_and(|&w| (w >> (id % 64)) & 1 == 1)
    }
    #[inline]
    fn off(&self) -> usize {
        0
    }
    #[inline]
    fn is_all(&self) -> bool {
        false
    }
    #[inline]
    fn skips_cell(&self, out_ids: &[i32]) -> bool {
        !out_ids.iter().any(|&id| self.keeps(id as u32))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cell_apply_left<T: Scalar, K: BandKeep>(
    mut yb: MatMut<'_, T>,
    fblk: MatRef<'_, T>,
    xb: MatRef<'_, T>,
    i_ranges: &[Range<usize>],
    j_ranges: &[Range<usize>],
    csr: &CsxBlock<T>,
    csc: &CsxBlock<T>,
    shard: K,
) {
    if shard.skips_cell(&csc.ids) {
        return;
    }
    let f_rows = fblk.nrows();
    let nnz = csr.nnz();
    let nrow = csr.span_len();
    let ncol = csc.span_len();

    instrument::cell();
    let xcols = xb.ncols();
    instrument::footprint(f_rows * fblk.ncols(), f_rows * xb.nrows().max(xcols));
    let _tc = if instrument::cell_timers() {
        instrument::tic()
    } else {
        None
    };

    if ncol < nnz.min(nrow) {
        for pj in 0..csc.ids.len() {
            if !shard.keeps(csc.ids[pj] as u32) {
                continue;
            }
            let jr = &j_ranges[csc.ids[pj] as usize];
            let lo = csc.starts[pj] as usize;
            let hi = csc.starts[pj + 1] as usize;
            if hi > lo + 1 {
                let mut fj = Mat::<T>::zeros(f_rows, xb.nrows());
                for idx in lo..hi {
                    let ir = &i_ranges[csc.index[idx] as usize];
                    let alpha = csc.vals[idx];
                    let src = fblk.submatrix(0, ir.start, f_rows, ir.len());
                    zip!(fj.as_mut(), src).for_each(|unzip!(d, s)| *d += alpha * *s);
                    instrument::op_read(f_rows * ir.len());
                    instrument::tmp(2 * f_rows * ir.len());
                }
                let dst = yb
                    .rb_mut()
                    .submatrix_mut(0, jr.start - shard.off(), f_rows, jr.len());
                let par = gemm_par(f_rows, jr.len(), xb.nrows());
                matmul(dst, Accum::Add, fj.as_ref(), xb, T::one(), par);
                instrument::gemm(f_rows, jr.len(), xb.nrows());
                instrument::tmp(f_rows * xb.nrows());
                instrument::x_read(xb.nrows() * xcols);
                instrument::y_write(f_rows * jr.len());
            } else {
                let ir = &i_ranges[csc.index[lo] as usize];
                let alpha = csc.vals[lo];
                let fsub = fblk.submatrix(0, ir.start, f_rows, ir.len());
                let dst = yb
                    .rb_mut()
                    .submatrix_mut(0, jr.start - shard.off(), f_rows, jr.len());
                let par = gemm_par(f_rows, jr.len(), ir.len());
                matmul(dst, Accum::Add, fsub, xb, alpha, par);
                instrument::gemm(f_rows, jr.len(), ir.len());
                instrument::op_read(f_rows * ir.len());
                instrument::x_read(ir.len() * xcols);
                instrument::y_write(f_rows * jr.len());
            }
        }
        instrument::cell_done(instrument::slot::CELL_CSC_L, _tc);
    } else if nrow < nnz.min(ncol) {
        for pi in 0..csr.ids.len() {
            let ir = &i_ranges[csr.ids[pi] as usize];
            let lo = csr.starts[pi] as usize;
            let hi = csr.starts[pi + 1] as usize;
            let fsub = fblk.submatrix(0, ir.start, f_rows, ir.len());
            if hi > lo + 1 {
                if !(lo..hi).any(|idx| shard.keeps(csr.index[idx] as u32)) {
                    continue;
                }
                let mut fix = Mat::<T>::zeros(f_rows, xb.ncols());
                let par = gemm_par(f_rows, xcols, ir.len());
                matmul(fix.as_mut(), Accum::Replace, fsub, xb, T::one(), par);
                instrument::gemm(f_rows, xcols, ir.len());
                instrument::panel(f_rows, xcols, ir.len());
                instrument::op_read(f_rows * ir.len());
                instrument::x_read(ir.len() * xcols);
                instrument::tmp(f_rows * xcols);
                for idx in lo..hi {
                    if !shard.keeps(csr.index[idx] as u32) {
                        continue;
                    }
                    let jr = &j_ranges[csr.index[idx] as usize];
                    let alpha = csr.vals[idx];
                    let dst =
                        yb.rb_mut()
                            .submatrix_mut(0, jr.start - shard.off(), f_rows, jr.len());
                    zip!(dst, fix.as_ref()).for_each(|unzip!(d, s)| *d += alpha * *s);
                    instrument::tmp(f_rows * xcols);
                    instrument::y_write(f_rows * jr.len());
                }
            } else {
                if !shard.keeps(csr.index[lo] as u32) {
                    continue;
                }
                let jr = &j_ranges[csr.index[lo] as usize];
                let alpha = csr.vals[lo];
                let dst = yb
                    .rb_mut()
                    .submatrix_mut(0, jr.start - shard.off(), f_rows, jr.len());
                let par = gemm_par(f_rows, jr.len(), ir.len());
                matmul(dst, Accum::Add, fsub, xb, alpha, par);
                instrument::gemm(f_rows, jr.len(), ir.len());
                instrument::op_read(f_rows * ir.len());
                instrument::x_read(ir.len() * xcols);
                instrument::y_write(f_rows * jr.len());
            }
        }
        instrument::cell_done(instrument::slot::CELL_CSR_L, _tc);
    } else {
        for (row, col, alpha) in csr.entries() {
            if !shard.keeps(col as u32) {
                continue;
            }
            let ir = &i_ranges[row as usize];
            let jr = &j_ranges[col as usize];
            let fsub = fblk.submatrix(0, ir.start, f_rows, ir.len());
            let dst = yb
                .rb_mut()
                .submatrix_mut(0, jr.start - shard.off(), f_rows, jr.len());
            let par = gemm_par(f_rows, jr.len(), ir.len());
            matmul(dst, Accum::Add, fsub, xb, alpha, par);
            instrument::gemm(f_rows, jr.len(), ir.len());
            instrument::op_read(f_rows * ir.len());
            instrument::x_read(ir.len() * xcols);
            instrument::y_write(f_rows * jr.len());
        }
        instrument::cell_done(instrument::slot::CELL_COO_L, _tc);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cell_apply_right<T: Scalar>(
    yb: MatMut<'_, T>,
    xb: MatRef<'_, T>,
    fblk: MatRef<'_, T>,
    i_ranges: &[Range<usize>],
    j_ranges: &[Range<usize>],
    csr: &CsxBlock<T>,
    csc: &CsxBlock<T>,
) {
    cell_apply_right_shard(yb, xb, fblk, i_ranges, j_ranges, csr, csc, RowShard::ALL);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cell_apply_right_shard<T: Scalar, K: BandKeep>(
    mut yb: MatMut<'_, T>,
    xb: MatRef<'_, T>,
    fblk: MatRef<'_, T>,
    i_ranges: &[Range<usize>],
    j_ranges: &[Range<usize>],
    csr: &CsxBlock<T>,
    csc: &CsxBlock<T>,
    shard: K,
) {
    if shard.skips_cell(&csr.ids) {
        return;
    }
    let f_cols = fblk.ncols();
    let nnz = csr.nnz();
    let nrow = csr.span_len();
    let ncol = csc.span_len();

    instrument::cell();
    let (xr, xc) = (xb.nrows(), xb.ncols());
    instrument::footprint(fblk.nrows() * f_cols, xr.max(xc) * f_cols);
    let _tc = if instrument::cell_timers() {
        instrument::tic()
    } else {
        None
    };

    if ncol < nnz.min(nrow) {
        for pj in 0..csc.ids.len() {
            let jr = &j_ranges[csc.ids[pj] as usize];
            let lo = csc.starts[pj] as usize;
            let hi = csc.starts[pj + 1] as usize;
            let fsub = fblk.submatrix(jr.start, 0, jr.len(), f_cols);
            if hi > lo + 1 {
                if !(lo..hi).any(|idx| shard.keeps(csc.index[idx] as u32)) {
                    continue;
                }
                let mut xfj = Mat::<T>::zeros(xb.nrows(), f_cols);
                let par = gemm_par(xr, f_cols, jr.len());
                matmul(xfj.as_mut(), Accum::Replace, xb, fsub, T::one(), par);
                instrument::gemm(xr, f_cols, jr.len());
                instrument::panel(xr, f_cols, jr.len());
                instrument::op_read(jr.len() * f_cols);
                instrument::x_read(xr * xc);
                instrument::tmp(xr * f_cols);
                for idx in lo..hi {
                    if !shard.keeps(csc.index[idx] as u32) {
                        continue;
                    }
                    let ir = &i_ranges[csc.index[idx] as usize];
                    let alpha = csc.vals[idx];
                    let dst =
                        yb.rb_mut()
                            .submatrix_mut(ir.start - shard.off(), 0, ir.len(), f_cols);
                    zip!(dst, xfj.as_ref()).for_each(|unzip!(d, s)| *d += alpha * *s);
                    instrument::tmp(xr * f_cols);
                    instrument::y_write(ir.len() * f_cols);
                }
            } else {
                if !shard.keeps(csc.index[lo] as u32) {
                    continue;
                }
                let ir = &i_ranges[csc.index[lo] as usize];
                let alpha = csc.vals[lo];
                let dst = yb
                    .rb_mut()
                    .submatrix_mut(ir.start - shard.off(), 0, ir.len(), f_cols);
                let par = gemm_par(ir.len(), f_cols, jr.len());
                matmul(dst, Accum::Add, xb, fsub, alpha, par);
                instrument::gemm(xr, f_cols, jr.len());
                instrument::op_read(jr.len() * f_cols);
                instrument::x_read(xr * jr.len());
                instrument::y_write(ir.len() * f_cols);
            }
        }
        instrument::cell_done(instrument::slot::CELL_CSC_R, _tc);
    } else if nrow < nnz.min(ncol) {
        for pi in 0..csr.ids.len() {
            if !shard.keeps(csr.ids[pi] as u32) {
                continue;
            }
            let ir = &i_ranges[csr.ids[pi] as usize];
            let lo = csr.starts[pi] as usize;
            let hi = csr.starts[pi + 1] as usize;
            if hi > lo + 1 {
                let mut fi = Mat::<T>::zeros(xb.ncols(), f_cols);
                for idx in lo..hi {
                    let jr = &j_ranges[csr.index[idx] as usize];
                    let alpha = csr.vals[idx];
                    let src = fblk.submatrix(jr.start, 0, jr.len(), f_cols);
                    zip!(fi.as_mut(), src).for_each(|unzip!(d, s)| *d += alpha * *s);
                    instrument::op_read(jr.len() * f_cols);
                    instrument::tmp(2 * jr.len() * f_cols);
                }
                let dst = yb
                    .rb_mut()
                    .submatrix_mut(ir.start - shard.off(), 0, ir.len(), f_cols);
                let par = gemm_par(ir.len(), f_cols, xc);
                matmul(dst, Accum::Add, xb, fi.as_ref(), T::one(), par);
                instrument::gemm(xr, f_cols, xc);
                instrument::tmp(xc * f_cols);
                instrument::x_read(xr * xc);
                instrument::y_write(ir.len() * f_cols);
            } else {
                let jr = &j_ranges[csr.index[lo] as usize];
                let alpha = csr.vals[lo];
                let src = fblk.submatrix(jr.start, 0, jr.len(), f_cols);
                let dst = yb
                    .rb_mut()
                    .submatrix_mut(ir.start - shard.off(), 0, ir.len(), f_cols);
                let par = gemm_par(ir.len(), f_cols, jr.len());
                matmul(dst, Accum::Add, xb, src, alpha, par);
                instrument::gemm(xr, f_cols, jr.len());
                instrument::op_read(jr.len() * f_cols);
                instrument::x_read(xr * jr.len());
                instrument::y_write(ir.len() * f_cols);
            }
        }
        instrument::cell_done(instrument::slot::CELL_CSR_R, _tc);
    } else {
        for (row, col, alpha) in csr.entries() {
            if !shard.keeps(row as u32) {
                continue;
            }
            let ir = &i_ranges[row as usize];
            let jr = &j_ranges[col as usize];
            let fsub = fblk.submatrix(jr.start, 0, jr.len(), f_cols);
            let dst = yb
                .rb_mut()
                .submatrix_mut(ir.start - shard.off(), 0, ir.len(), f_cols);
            let par = gemm_par(ir.len(), f_cols, jr.len());
            matmul(dst, Accum::Add, xb, fsub, alpha, par);
            instrument::gemm(xr, f_cols, jr.len());
            instrument::op_read(jr.len() * f_cols);
            instrument::x_read(xr * jr.len());
            instrument::y_write(ir.len() * f_cols);
        }
        instrument::cell_done(instrument::slot::CELL_COO_R, _tc);
    }
}

#[cfg(test)]
mod branch_tests;
