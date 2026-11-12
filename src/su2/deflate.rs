// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use crate::core::deflate::BondDeflation;
use crate::core::factor::svd_uvt_svals;
use crate::scalar::Scalar;
use crate::su2::flux::channel_offset;
use crate::su2::qn::{Su2RankGrid, Su2Sector};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::weingarten::Su2BondGauge;

#[derive(Clone)]
struct Su2NullBlock<T: Scalar> {
    k: usize,
    t: Su2Sector,
    vn: Mat<T>,
    cut: Vec<f64>,
    g: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LiftMode {
    Fan,
    LabelOnly,
    OffsetZero,
}

#[derive(Clone)]
pub(crate) struct Su2ColumnDeflation<T: Scalar> {
    null: Vec<Su2NullBlock<T>>,
    bonds: Vec<BondDeflation>,
    grids: Vec<Su2RankGrid>,
}

impl<T: Scalar> Su2ColumnDeflation<T> {
    pub(crate) fn from_gauges(gauges: &[Su2BondGauge<T>], floor: f64) -> Self {
        Self::build(gauges, |s| (s < floor).then_some(1.0), true)
    }

    pub(crate) fn from_gauges_weighted(gauges: &[Su2BondGauge<T>], w: impl Fn(f64) -> f64) -> Self {
        Self::build(
            gauges,
            |s| {
                let c = 1.0 - w(s);
                (c > 1e-15).then_some(c)
            },
            false,
        )
    }

    fn build(
        gauges: &[Su2BondGauge<T>],
        select: impl Fn(f64) -> Option<f64>,
        cut_point: bool,
    ) -> Self {
        let mut null = Vec::new();
        let mut bonds = Vec::with_capacity(gauges.len());
        let mut grids = Vec::with_capacity(gauges.len());
        for g in gauges {
            let gi = grids.len();
            grids.push(g.r.row_ranks());
            let mut rep = BondDeflation {
                bond: g.k,
                total: 0,
                deflated: 0,
                smallest_kept: f64::INFINITY,
                largest_deflated: 0.0,
            };
            for t in g.r.sectors() {
                let Some(blk) = g.r.block(t) else { continue };
                if blk.nrows() == 0 || blk.ncols() == 0 {
                    continue;
                }
                let (_w, sv, vt) = svd_uvt_svals(blk);
                rep.total += sv.len();
                let (mut idx, mut cut) = (Vec::new(), Vec::new());
                for (i, &s) in sv.iter().enumerate() {
                    match select(s) {
                        Some(c) => {
                            idx.push(i);
                            cut.push(c);
                            if cut_point {
                                rep.largest_deflated = rep.largest_deflated.max(s);
                            }
                        }
                        None => {
                            if cut_point {
                                rep.smallest_kept = rep.smallest_kept.min(s);
                            }
                        }
                    }
                }
                if idx.is_empty() {
                    continue;
                }
                rep.deflated += idx.len();
                let vn = Mat::from_fn(vt.ncols(), idx.len(), |r, c| vt.get(idx[c], r).conj_s());
                null.push(Su2NullBlock {
                    k: g.k - 1,
                    t,
                    vn,
                    cut,
                    g: gi,
                });
            }
            bonds.push(rep);
        }
        Su2ColumnDeflation { null, bonds, grids }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.null.is_empty()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn bonds(&self) -> &[BondDeflation] {
        &self.bonds
    }

    pub(crate) fn apply(&self, xi: &mut Su2TtTangent<T>) {
        if xi.two_s_flux != 0 {
            self.apply_shifted(xi, false, false);
            return;
        }
        for nb in &self.null {
            let comp = &mut xi.components[nb.k];
            let mut c = comp.vertical_unfolding(nb.t);
            if c.nrows() == 0 || c.ncols() == 0 {
                continue;
            }
            let mut cv = Mat::<T>::zeros(c.nrows(), nb.vn.ncols());
            matmul(
                cv.as_mut(),
                Accum::Replace,
                c.as_ref(),
                nb.vn.as_ref(),
                T::one(),
                Par::Seq,
            );
            if nb.cut.iter().any(|&f| f != 1.0) {
                for (j, &f) in nb.cut.iter().enumerate() {
                    for i in 0..cv.nrows() {
                        let x = cv[(i, j)];
                        cv[(i, j)] = x * T::from_f64(f);
                    }
                }
            }
            matmul(
                c.as_mut(),
                Accum::Add,
                cv.as_ref(),
                nb.vn.as_ref().adjoint(),
                T::from_f64(-1.0),
                Par::Seq,
            );
            comp.scatter_vertical(nb.t, c.as_ref());
        }
    }

    fn apply_shifted(&self, xi: &mut Su2TtTangent<T>, label_only: bool, offset_zero: bool) {
        let two_k = xi.two_s_flux as i64;
        for nb in &self.null {
            let b = nb.k + 1;
            let r = &self.grids[nb.g];
            let ru = r[nb.t];
            debug_assert_eq!(nb.vn.nrows(), ru, "null block on the reference grid");
            let out = xi.components[nb.k].space;
            for t in out.bond_sectors(b) {
                let rd = xi.r_col[b][t];
                if rd == 0 || (label_only && t != nb.t) {
                    continue;
                }
                let Some(off) = channel_offset(t, nb.t, r, two_k, out.n) else {
                    continue;
                };
                let off = if offset_zero { 0 } else { off };
                let comp = &mut xi.components[nb.k];
                let mut c = comp.vertical_unfolding(t);
                debug_assert_eq!(c.ncols(), rd, "the dressed column grid");
                debug_assert!(off + ru <= c.ncols(), "the window fits the dressed label");
                if c.nrows() == 0 {
                    continue;
                }
                let mut cv = Mat::<T>::zeros(c.nrows(), nb.vn.ncols());
                matmul(
                    cv.as_mut(),
                    Accum::Replace,
                    c.as_ref().subcols(off, ru),
                    nb.vn.as_ref(),
                    T::one(),
                    Par::Seq,
                );
                if nb.cut.iter().any(|&f| f != 1.0) {
                    for (j, &f) in nb.cut.iter().enumerate() {
                        for i in 0..cv.nrows() {
                            let x = cv[(i, j)];
                            cv[(i, j)] = x * T::from_f64(f);
                        }
                    }
                }
                matmul(
                    c.as_mut().subcols_mut(off, ru),
                    Accum::Add,
                    cv.as_ref(),
                    nb.vn.as_ref().adjoint(),
                    T::from_f64(-1.0),
                    Par::Seq,
                );
                comp.scatter_vertical(t, c.as_ref());
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn apply_lift_mode(&self, xi: &mut Su2TtTangent<T>, mode: LiftMode) {
        if xi.two_s_flux == 0 || mode == LiftMode::Fan {
            self.apply(xi);
            return;
        }
        self.apply_shifted(
            xi,
            mode == LiftMode::LabelOnly,
            mode == LiftMode::OffsetZero,
        );
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn removed_dim(&self, template: &Su2TtTangent<T>) -> usize {
        if template.two_s_flux != 0 {
            let two_k = template.two_s_flux as i64;
            return self
                .null
                .iter()
                .map(|nb| {
                    let b = nb.k + 1;
                    let r = &self.grids[nb.g];
                    let comp = &template.components[nb.k];
                    comp.space
                        .bond_sectors(b)
                        .into_iter()
                        .filter(|&t| {
                            template.r_col[b][t] > 0
                                && channel_offset(t, nb.t, r, two_k, comp.space.n).is_some()
                        })
                        .map(|t| {
                            let rows = comp.vertical_unfolding(t).nrows();
                            rows.saturating_sub(template.r[b][t]) * nb.vn.ncols()
                        })
                        .sum::<usize>()
                })
                .sum();
        }
        self.null
            .iter()
            .map(|nb| {
                let c = template.components[nb.k].vertical_unfolding(nb.t);
                c.nrows().saturating_sub(c.ncols()) * nb.vn.ncols()
            })
            .sum()
    }
}
