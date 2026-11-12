// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) tangent-space machinery — the SU(2)-symmetric sibling of [`crate::core::tangent`], which
//! is duplicated-and-substituted here rather than dispatched over a layer trait.

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, Par};

use rayon::prelude::*;
use std::sync::Arc;

use crate::core::algebra::{block_axpy, block_scale};
use crate::par;
use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::coreops::{
    adjoint_core_times_core, assert_same_blocks, mul_sketch_left, mul_sketch_right, rproject,
    step_abs2_center, step_dot_center,
};
use crate::su2::flux::{dress_train, shifted_out_space};
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::Su2RankGrid;
use crate::su2::vector::Su2TtVector;

type BlockPair<'a, T> = (&'a mut Mat<T>, &'a Mat<T>);

/// Gauge-fixed SU(2) tangent vector at `base_l` (left-canonical)/`base_r` (θ-right-canonical)
/// (abelian sibling [`crate::core::tangent::TtTangent`]). `r` = the
/// `d+1` bond ROW rank grids of `base_l` (the FIXED manifold ranks; the gauge lives in
/// `base_l`/`base_r`, two orthogonalized copies of the same base point); `components` = the `d`
/// gauge-fixed component cores.
#[derive(Clone)]
pub struct Su2TtTangent<T: Scalar> {
    pub(crate) r: Vec<Su2RankGrid>,
    pub(crate) r_col: Vec<Su2RankGrid>,
    pub(crate) two_s_flux: usize,
    pub(crate) base_l: Arc<Su2TtVector<T>>,
    pub(crate) base_r: Arc<Su2TtVector<T>>,
    pub(crate) components: Vec<Su2Core<T>>,
    pub(crate) ncoeff: usize,
}

impl<T: Scalar> Su2TtTangent<T> {
    pub(crate) fn from_parts(
        r: Vec<Su2RankGrid>,
        base_l: impl Into<Arc<Su2TtVector<T>>>,
        base_r: impl Into<Arc<Su2TtVector<T>>>,
        components: Vec<Su2Core<T>>,
    ) -> Self {
        let r_col = r.clone();
        Self::from_parts_shifted(r, r_col, 0, base_l, base_r, components)
    }

    pub(crate) fn from_parts_shifted(
        r: Vec<Su2RankGrid>,
        r_col: Vec<Su2RankGrid>,
        two_s_flux: usize,
        base_l: impl Into<Arc<Su2TtVector<T>>>,
        base_r: impl Into<Arc<Su2TtVector<T>>>,
        components: Vec<Su2Core<T>>,
    ) -> Self {
        let (base_l, base_r) = (base_l.into(), base_r.into());
        debug_assert_eq!(
            r.len(),
            r_col.len(),
            "from_parts_shifted: grid count mismatch"
        );
        let ncoeff = components.iter().map(|c| c.num_coeffs()).sum();
        Su2TtTangent {
            r,
            r_col,
            two_s_flux,
            base_l,
            base_r,
            components,
            ncoeff,
        }
    }

    /// The spin flux `2κ` of this direction: `0` for an ordinary tangent vector, `2S'` for a
    /// direction of the shifted space `T^{S'}` (whose embedding lives in `space.with_two_s(2κ)`).
    #[inline]
    pub fn two_s_flux(&self) -> usize {
        self.two_s_flux
    }

    /// The components' COLUMN rank grids — the dressed fan `crate::su2::flux::flux_ranks` produces
    /// at `2κ`, equal to [`Self::rank`] at `two_s_flux == 0`.
    #[inline]
    pub fn rank_col(&self) -> &[Su2RankGrid] {
        &self.r_col
    }

    #[inline]
    pub(crate) fn refuse_if_shifted(&self, what: &str) {
        assert_eq!(
            self.two_s_flux, 0,
            "{what} is undefined on a shifted tangent (2S_flux = {}): x + alpha*xi is not a point \
             of any single typed space, and rounding a 2S > 0 train is not supported; \
             embed with to_vector() instead",
            self.two_s_flux
        );
    }

    pub(crate) fn gauge_fix_orthogonal_mut(&mut self) {
        use crate::su2::weingarten::gauge_complement_mut;
        let d = self.components.len();
        for k in 0..d.saturating_sub(1) {
            gauge_complement_mut(&mut self.components[k], self.base_l.core(k));
        }
    }

    #[inline]
    fn par_tasks_mut<'a>(&'a mut self, other: &'a Self) -> Vec<BlockPair<'a, T>> {
        self.components
            .iter_mut()
            .zip(other.components.iter())
            .flat_map(|(s, o)| {
                s.store_mut()
                    .blocks_mut()
                    .iter_mut()
                    .zip(o.store().blocks().iter())
            })
            .collect()
    }

    #[inline]
    fn assert_zippable(&self, other: &Self, what: &str) {
        for (s, o) in self.components.iter().zip(&other.components) {
            assert_same_blocks(s, o, what);
        }
    }
}

pub(crate) fn shifted_frames<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    two_s_flux: usize,
) -> (Su2TtVector<T>, Su2TtVector<T>) {
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "shifted_frames: the two frames must factorize the SAME state"
    );
    if two_s_flux == 0 {
        return (base_l.clone(), base_r.clone());
    }
    let space_out = shifted_out_space(base_l.space, two_s_flux);
    let mut spin = crate::su2::spinalgebra::SpinAlgebra::new();
    let u = dress_train(base_l, space_out, 0, &mut spin);
    let v = dress_train(base_r, space_out, two_s_flux as i64, &mut spin);
    (u, v)
}

impl<T: Scalar> Su2TtTangent<T> {
    /// The 3-arg gauge-fixed projection (abelian
    /// [`crate::core::tangent::TtTangent::project`]) of `dir`
    /// onto the tangent space `T_x(M_r)` at `x` (represented by its left- and right-canonical
    /// factorizations `base_l`/`base_r`).
    pub fn project(base_l: &Su2TtVector<T>, base_r: &Su2TtVector<T>, dir: &Su2TtVector<T>) -> Self {
        Self::project_impl(Arc::new(base_l.clone()), Arc::new(base_r.clone()), 0, dir)
    }

    pub(crate) fn project_shared(
        base_l: &Arc<Su2TtVector<T>>,
        base_r: &Arc<Su2TtVector<T>>,
        dir: &Su2TtVector<T>,
    ) -> Self {
        Self::project_impl(base_l.clone(), base_r.clone(), 0, dir)
    }

    /// `P^{S'}(dir)`: the gauge-fixed projection of `dir ∈ space_out` onto the SHIFTED tangent space
    /// `T^{S'}` at the frames `base_l`/`base_r`, `2κ = two_s_flux`.
    pub fn project_shifted(
        base_l: &Su2TtVector<T>,
        base_r: &Su2TtVector<T>,
        two_s_flux: usize,
        dir: &Su2TtVector<T>,
    ) -> Self {
        let (u, v) = shifted_frames(base_l, base_r, two_s_flux);
        Self::project_impl(Arc::new(u), Arc::new(v), two_s_flux, dir)
    }

    pub(crate) fn project_impl(
        base_l: Arc<Su2TtVector<T>>,
        base_r: Arc<Su2TtVector<T>>,
        two_s_flux: usize,
        dir: &Su2TtVector<T>,
    ) -> Self {
        let space = base_l.space;
        let d = space.d;
        debug_assert_eq!(
            base_l.space, base_r.space,
            "project: both frames must be stamped in the output space"
        );
        debug_assert!(
            two_s_flux != 0 || base_l.ranks == base_r.ranks,
            "project: at flux 0 the two frames must factorize the SAME point"
        );
        assert_eq!(
            dir.space, space,
            "project: the direction lives in {:?} but the shifted tangent space is in {:?}",
            dir.space, space
        );
        let r = base_l.ranks.clone();
        let r_col = base_r.ranks.clone();

        let mut p: Vec<Su2Sketch<T>> = Vec::with_capacity(d);
        p.push(Su2Sketch::identity(space, 0, 1));
        for kc in 0..d.saturating_sub(1) {
            let pd = mul_sketch_left(&p[kc], dir.core(kc));
            let next = adjoint_core_times_core(base_l.core(kc), &pd);
            p.push(next);
        }

        let mut q = Su2Sketch::identity(space, d, 1);
        let mut components: Vec<Option<Su2Core<T>>> = (0..d).map(|_| None).collect();
        for kc in (0..d).rev() {
            let dq = mul_sketch_right(dir.core(kc), &q);
            let mut comp = mul_sketch_left(&p[kc], &dq);

            if kc < d - 1 {
                for s in space.bond_sectors(kc + 1) {
                    let l = base_l.core(kc).vertical_unfolding(s);
                    let mut c = comp.vertical_unfolding(s);
                    if l.ncols() > 0 && c.ncols() > 0 && l.nrows() > 0 {
                        let mut lhc = Mat::<T>::zeros(l.ncols(), c.ncols());
                        matmul(
                            lhc.as_mut(),
                            Accum::Replace,
                            l.as_ref().adjoint(),
                            c.as_ref(),
                            T::one(),
                            Par::Seq,
                        );
                        matmul(
                            c.as_mut(),
                            Accum::Add,
                            l.as_ref(),
                            lhc.as_ref(),
                            T::from_f64(-1.0),
                            Par::Seq,
                        );
                    }
                    comp.scatter_vertical(s, c.as_ref());
                }
            }

            if kc > 0 {
                q = rproject(&dq, base_r.core(kc));
            }
            components[kc] = Some(comp);
        }

        let components: Vec<Su2Core<T>> = components.into_iter().map(|c| c.unwrap()).collect();
        Su2TtTangent::from_parts_shifted(r, r_col, two_s_flux, base_l, base_r, components)
    }

    /// Tangent of `base` along `dir` (abelian
    /// [`crate::core::tangent::TtTangent::from_base`]): copies of `base`,
    /// `left_orthogonalize(keep_rank=true)` / `right_orthogonalize(keep_rank=true)` of `base`
    /// (fixed-rank manifold — `keep_rank=true` is load-bearing, must not truncate), then
    /// [`Self::project`].
    pub fn from_base(base: &Su2TtVector<T>, dir: &Su2TtVector<T>) -> Self {
        let mut base_l = base.clone();
        base_l.left_orthogonalize(true);
        let mut base_r = base.clone();
        base_r.right_orthogonalize(true);
        Self::project(&base_l, &base_r, dir)
    }

    /// `TTvector(dx)` = the embedding `ι(ξ)` (abelian
    /// [`crate::core::tangent::TtTangent::to_vector`]): the tangent vector as a full TT in the
    /// output space, interior bonds doubled (`base_r` atop `component` atop `base_l`,
    /// block-triangular over the four SU(2) column-sector families `Family::ALL`). No
    /// θ/centerweight, pure block stacking.
    pub fn to_vector(&self) -> Su2TtVector<T> {
        let space = self.base_l.space;
        let d = space.d;
        if d == 1 {
            return Su2TtVector::new(space, vec![self.components[0].clone()]);
        }
        let r = &self.r;
        let rc = &self.r_col;
        let mut cores = Vec::with_capacity(d);
        for kc in 0..d {
            let has_v = kc > 0;
            let has_u = kc < d - 1;
            let mut rr = Su2RankGrid::zeros(&space);
            let mut cr = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(kc) {
                rr[s] = r[kc][s] + if has_v { rc[kc][s] } else { 0 };
            }
            for s in space.bond_sectors(kc + 1) {
                cr[s] = rc[kc + 1][s] + if has_u { r[kc + 1][s] } else { 0 };
            }
            let mut core = Su2Core::zeros(space, kc, &rr, &cr);
            let g = space.pg.irrep_of(kc);
            for l in space.bond_sectors(kc) {
                for f in Family::ALL {
                    let Some(t) = f.apply(l, g) else {
                        continue;
                    };
                    if !space.contains(kc + 1, t) {
                        continue;
                    }
                    let Some(mut dst) = core.block_mut(f, l) else {
                        continue;
                    };
                    let i0 = if has_v { rc[kc][l] } else { 0 };
                    let j0 = rc[kc + 1][t];
                    if let Some(b) = has_v.then(|| self.base_r.core(kc).block(f, l)).flatten() {
                        dst.rb_mut()
                            .submatrix_mut(0, 0, b.nrows(), b.ncols())
                            .copy_from(b);
                    }
                    if let Some(b) = self.components[kc].block(f, l) {
                        dst.rb_mut()
                            .submatrix_mut(i0, 0, b.nrows(), b.ncols())
                            .copy_from(b);
                    }
                    if let Some(b) = has_u.then(|| self.base_l.core(kc).block(f, l)).flatten() {
                        dst.rb_mut()
                            .submatrix_mut(i0, j0, b.nrows(), b.ncols())
                            .copy_from(b);
                    }
                }
            }
            cores.push(core);
        }
        Su2TtVector::new(space, cores)
    }

    /// Mirror of the abelian [`crate::core::tangent::TtTangent::retract_hosvd`]: the
    /// fixed-rank second-order retraction `round(x + α·dx, r)`, built directly at doubled rank
    /// (both base pieces present everywhere; the last core's bottom band carries `base_l +
    /// α·comp`) then HOSVD-truncated back to `r` via [`crate::su2::round`]'s
    /// `Su2TtVector::round_ranks_mut` ("no chopping"). **Reuse the shipped kernel — do NOT
    /// re-derive HOSVD**; `round_ranks_mut` is already θ/centerweight-aware, so the retraction
    /// inherits the correct fixed-Sz metric for free.
    pub fn retract_hosvd(&self, alpha: T) -> Su2TtVector<T> {
        self.refuse_if_shifted("retract_hosvd");
        let mut v = self.moved_point(alpha);
        v.round_ranks_mut(&self.r);
        v
    }

    /// The rank-ADAPTIVE retraction `round(x + α·dx, ϵ, chi)` — the SU(2) twin of
    /// [`crate::core::tangent::TtTangent::retract_round`]: the doubled-rank moved point truncated
    /// by the θ/centerweight-aware energy budget of `Su2TtVector::round_mut` instead of back to
    /// the base grid. Ranks are REDUCED multiplet ranks, as everywhere on this layer, and `chi`
    /// is a POOLED cap on each bond's TOTAL number of multiplets ([`crate::qn::Pooled::NONE`] = uncapped).
    /// Refuses a SHIFTED tangent (`two_s_flux != 0`) — see [`Self::moved_point`].
    pub fn retract_round(
        &self,
        alpha: T,
        eps: f64,
        chi: crate::qn::Pooled,
        mode: crate::round::RoundMode,
    ) -> Su2TtVector<T> {
        self.refuse_if_shifted("retract_round");
        let mut v = self.moved_point(alpha);
        v.round_mut(eps, chi, mode);
        v
    }

    /// `x + α·dx` at DOUBLED reduced rank, exact — the SU(2) twin of
    /// [`crate::core::tangent::TtTangent::moved_point`]; the retractions round it.
    pub fn moved_point(&self, alpha: T) -> Su2TtVector<T> {
        self.refuse_if_shifted("moved_point");
        let space = self.base_l.space;
        let d = space.d;
        if d == 1 {
            let mut core = self.base_l.core(0).clone();
            core.axpy(alpha, &self.components[0]);
            return Su2TtVector::new(space, vec![core]);
        }
        let r = &self.r;
        let mut cores = Vec::with_capacity(d);
        for kc in 0..d {
            let row_mul = if kc == 0 { 1 } else { 2 };
            let col_mul = if kc == d - 1 { 1 } else { 2 };
            let mut rr = Su2RankGrid::zeros(&space);
            let mut cr = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(kc) {
                rr[s] = row_mul * r[kc][s];
            }
            for s in space.bond_sectors(kc + 1) {
                cr[s] = col_mul * r[kc + 1][s];
            }
            let mut core = Su2Core::zeros(space, kc, &rr, &cr);
            let g = space.pg.irrep_of(kc);
            for l in space.bond_sectors(kc) {
                for f in Family::ALL {
                    let Some(t) = f.apply(l, g) else {
                        continue;
                    };
                    if !space.contains(kc + 1, t) {
                        continue;
                    }
                    let Some(mut dst) = core.block_mut(f, l) else {
                        continue;
                    };
                    let bl = self.base_l.core(kc).block(f, l);
                    let br = self.base_r.core(kc).block(f, l);
                    let bc = self.components[kc].block(f, l);
                    let m = r[kc][l];
                    let n = r[kc + 1][t];
                    for j in 0..n {
                        for i in 0..m {
                            let a = bl.map(|b| *b.get(i, j)).unwrap_or_else(T::zero);
                            let rr_v = br.map(|b| *b.get(i, j)).unwrap_or_else(T::zero);
                            let cc = bc.map(|b| *b.get(i, j)).unwrap_or_else(T::zero);
                            if kc == 0 {
                                *dst.rb_mut().get_mut(i, j) = alpha * cc;
                                *dst.rb_mut().get_mut(i, n + j) = a;
                            } else if kc == d - 1 {
                                *dst.rb_mut().get_mut(i, j) = rr_v;
                                *dst.rb_mut().get_mut(m + i, j) = a + alpha * cc;
                            } else {
                                *dst.rb_mut().get_mut(i, j) = rr_v;
                                *dst.rb_mut().get_mut(m + i, j) = alpha * cc;
                                *dst.rb_mut().get_mut(m + i, n + j) = a;
                            }
                        }
                    }
                }
            }
            cores.push(core);
        }
        Su2TtVector::new(space, cores)
    }

    /// `dot(dx, dy) = Σ_k dot(comp_x[k], comp_y[k])` (abelian
    /// [`crate::core::tangent::TtTangent::dot`] uses the PLAIN `SparseCore::dot_cores`).
    pub fn dot(&self, other: &Self) -> T {
        debug_assert_eq!(self.r, other.r, "dot: tangent ranks must agree");
        debug_assert_eq!(
            self.r_col, other.r_col,
            "dot: tangent column ranks must agree"
        );
        debug_assert_eq!(
            self.two_s_flux, other.two_s_flux,
            "dot: tangent spin-flux labels must agree"
        );
        self.assert_zippable(other, "Su2TtTangent::dot");
        if !par::par_vector_ops(self.ncoeff) {
            let mut acc = T::zero();
            for k in 0..self.components.len() {
                acc += self.components[k].dot_center(&other.components[k]);
            }
            return acc;
        }
        let (a, b) = (&self.components, &other.components);
        let parts: Vec<T> = (0..4 * a.len())
            .into_par_iter()
            .map(|i| {
                let (k, f) = (i / 4, i % 4);
                step_dot_center(&a[k], &b[k], Family::ALL[f])
            })
            .collect();
        parts.into_iter().fold(T::zero(), |acc, p| acc + p)
    }

    /// `norm(dx) = sqrt(Σ_k norm(comp[k])²)` (abelian
    /// [`crate::core::tangent::TtTangent::norm`] uses the PLAIN `SparseCore::abs2`).
    pub fn norm(&self) -> f64 {
        if !par::par_vector_ops(self.ncoeff) {
            let mut s = 0.0f64;
            for c in &self.components {
                s += c.abs2_center();
            }
            return s.sqrt();
        }
        let a = &self.components;
        let parts: Vec<f64> = (0..4 * a.len())
            .into_par_iter()
            .map(|i| step_abs2_center(&a[i / 4], Family::ALL[i % 4]))
            .collect();
        parts.into_iter().sum::<f64>().sqrt()
    }

    /// `dx + dy` (abelian
    /// [`crate::core::tangent::TtTangent::add`]): componentwise
    /// `axpy(+1, other)`; asserts equal ranks. STRUCTURAL DUPLICATE.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.axpy(T::one(), other);
        result
    }

    /// `dx - dy` (abelian [`crate::core::tangent::TtTangent::sub`]).
    /// STRUCTURAL DUPLICATE.
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.axpy(T::from_f64(-1.0), other);
        result
    }

    /// `a * dx` (abelian
    /// [`crate::core::tangent::TtTangent::scale`]). STRUCTURAL
    /// DUPLICATE.
    pub fn scale(&self, a: T) -> Self {
        let mut result = self.clone();
        result.scale_mut(a);
        result
    }

    /// `self *= a`: in-place [`Self::scale`] (abelian
    /// [`crate::core::tangent::TtTangent::scale_mut`]). STRUCTURAL
    /// DUPLICATE.
    pub fn scale_mut(&mut self, a: T) {
        if !par::par_vector_ops(self.ncoeff) {
            for c in &mut self.components {
                c.scale_mut(a);
            }
            return;
        }
        let mut tasks: Vec<&mut Mat<T>> = self
            .components
            .iter_mut()
            .flat_map(|c| c.store_mut().blocks_mut().iter_mut())
            .collect();
        tasks
            .par_iter_mut()
            .for_each(|m| block_scale(m.as_mut(), a));
    }

    /// Axpy (abelian
    /// [`crate::core::tangent::TtTangent::axpy`]): `self += a *
    /// other`. STRUCTURAL DUPLICATE.
    pub fn axpy(&mut self, a: T, other: &Self) {
        debug_assert_eq!(self.r, other.r, "axpy: tangent ranks must agree");
        debug_assert_eq!(
            self.r_col, other.r_col,
            "axpy: tangent column ranks must agree"
        );
        debug_assert_eq!(
            self.two_s_flux, other.two_s_flux,
            "axpy: tangent spin-flux labels must agree"
        );
        self.assert_zippable(other, "Su2TtTangent::axpy");
        if !par::par_vector_ops(self.ncoeff) {
            for k in 0..self.components.len() {
                self.components[k].axpy(a, &other.components[k]);
            }
            return;
        }
        let mut tasks = self.par_tasks_mut(other);
        tasks
            .par_iter_mut()
            .for_each(|(s, o)| block_axpy(s.as_mut(), o.as_ref(), a));
    }

    /// `self = a*other + b*self` (abelian
    /// [`crate::core::tangent::TtTangent::axpby`]). NOTE the abelian
    /// convention (`self = a·other + b·self`, the `axpby(a,dx,b,dy)` order computing
    /// `dy = a·dx + b·dy`) — duplicate that exact argument order, do not swap it. STRUCTURAL
    /// DUPLICATE.
    pub fn axpby(&mut self, a: T, other: &Self, b: T) {
        debug_assert_eq!(self.r, other.r, "axpby: tangent ranks must agree");
        debug_assert_eq!(
            self.r_col, other.r_col,
            "axpby: tangent column ranks must agree"
        );
        debug_assert_eq!(
            self.two_s_flux, other.two_s_flux,
            "axpby: tangent spin-flux labels must agree"
        );
        self.assert_zippable(other, "Su2TtTangent::axpby");
        if !par::par_vector_ops(self.ncoeff) {
            for k in 0..self.components.len() {
                self.components[k].scale_mut(b);
                self.components[k].axpy(a, &other.components[k]);
            }
            return;
        }
        let mut tasks = self.par_tasks_mut(other);
        tasks.par_iter_mut().for_each(|(s, o)| {
            block_scale(s.as_mut(), b);
            block_axpy(s.as_mut(), o.as_ref(), a);
        });
    }

    /// `rank(dx)` (abelian
    /// [`crate::core::tangent::TtTangent::rank`]). Trivial field access, kept for interface
    /// parity.
    #[inline]
    pub fn rank(&self) -> &[Su2RankGrid] {
        &self.r
    }

    /// The `k`-th gauge-fixed component core. `pub` (not `pub(crate)`): the `tangent_hessian` binary is a
    /// separate integration-test crate and its gauge-condition/anti-oracle clauses (§A2, §E1)
    /// read this directly, exactly as the abelian
    /// [`crate::core::tangent::TtTangent::component`] does for the `tangent_hessian` binary.
    #[inline]
    pub fn component(&self, k: usize) -> &Su2Core<T> {
        &self.components[k]
    }

    /// Total coefficient count: sum over components — the cached `ncoeff` (shape-invariant under
    /// every tangent op), the walk re-checked in debug builds.
    #[inline]
    pub fn num_coeffs(&self) -> usize {
        debug_assert_eq!(
            self.ncoeff,
            self.components
                .iter()
                .map(|c| c.num_coeffs())
                .sum::<usize>(),
            "Su2TtTangent: cached ncoeff out of date"
        );
        self.ncoeff
    }

    /// Visit every coefficient mutably in canonical (component-outer) order.
    pub fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, mut f: F) {
        for c in &mut self.components {
            c.for_each_coeff_mut(&mut f);
        }
    }

    /// Zip two same-shaped tangent vectors' coefficients in canonical order: `f(&mut a_i, b_i)`.
    pub fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, mut f: F) {
        debug_assert_eq!(self.components.len(), other.components.len());
        for (a, b) in self.components.iter_mut().zip(other.components.iter()) {
            a.zip_coeffs_mut(b, &mut f);
        }
    }

    /// Read the coefficient at a flat canonical index (`0..num_coeffs()`).
    pub fn coeff(&self, idx: usize) -> T {
        let mut rem = idx;
        for c in &self.components {
            let n = c.num_coeffs();
            if rem < n {
                return c.coeff(rem);
            }
            rem -= n;
        }
        panic!("Su2TtTangent::coeff: index {idx} out of range");
    }

    /// Write the coefficient at a flat canonical index.
    pub fn set_coeff(&mut self, idx: usize, val: T) {
        let mut rem = idx;
        for c in &mut self.components {
            let n = c.num_coeffs();
            if rem < n {
                c.set_coeff(rem, val);
                return;
            }
            rem -= n;
        }
        panic!("Su2TtTangent::set_coeff: index {idx} out of range");
    }
}

/// Projective transport (mirror of the abelian
/// [`crate::core::tangent::projective_transport`]): re-project the train `dx` represents onto the
/// tangent space at `y`. Free fn (not a method), verbatim structural duplicate — metric enters
/// only through [`Su2TtTangent::from_base`]
/// `→` [`Su2TtTangent::project`] (the H9-a `rproject` substitution); no extra metric use here.
/// Not called by the CG driver itself (it inlines the transport as
/// `Su2TtTangent::project(&base_l, &base_r, &eta.to_vector())`), but exercised directly by
/// the `tangent_hessian` binary §A8.
pub fn projective_transport<T: Scalar>(
    y: &Su2TtVector<T>,
    dx: &Su2TtTangent<T>,
) -> Su2TtTangent<T> {
    dx.refuse_if_shifted("projective_transport");
    let z = dx.to_vector();
    Su2TtTangent::from_base(y, &z)
}

/// `P_Γ ξ = (1/|G|) Σ_h χ_Γ(h) U_h ξ` — the component of a tangent direction transforming as `Γ`.
pub fn project_irrep<T: Scalar>(
    xi: &Su2TtTangent<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    pg: PointGroup,
    gamma: Irrep,
) -> (Su2TtTangent<T>, f64) {
    xi.refuse_if_shifted("project_irrep");
    let order = pg.order();
    let scale = 1.0 / order as f64;
    let v = xi.to_vector();
    let (bl, br) = (Arc::new(base_l.clone()), Arc::new(base_r.clone()));
    let mut acc: Option<Su2TtTangent<T>> = None;
    for hi in 0..order {
        let h = Irrep::new(hi);
        let ux = crate::su2::vector::apply_group_element(&v, pg, h);
        let mut term = Su2TtTangent::project_shared(&bl, &br, &ux);
        term.scale_mut(T::from_f64(gamma.character(h) * scale));
        acc = Some(match acc {
            None => term,
            Some(mut a) => {
                a.axpy(T::one(), &term);
                a
            }
        });
    }
    let out = acc.expect("point group has order >= 1");
    let n = xi.norm();
    let leak = if n > 0.0 {
        xi.sub(&out).norm() / n
    } else {
        0.0
    };
    (out, leak)
}

#[cfg(test)]
mod s6_tests {
    use super::*;
    use crate::round::RoundMode;
    use crate::su2::qn::Su2Space;
    use crate::test_util::{randomized_su2_state, seeded};

    #[test]
    fn s6_f1_d1_special_case() {
        let space = Su2Space::new(1, 1, 1);
        assert_eq!(space.d, 1, "F1 must exercise the d==1 branch");
        let mut rng = seeded(0x6f10);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        let nx = x.norm_lr(false);
        x.scale_mut(1.0 / nx);
        assert!(x.check());

        let mut base_l = x.clone();
        base_l.left_orthogonalize(true);
        let mut base_r = x.clone();
        base_r.right_orthogonalize(true);
        let mut dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
        dir.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        let nd = dir.norm_lr(false);
        dir.scale_mut(1.0 / nd);

        let dx = Su2TtTangent::project(&base_l, &base_r, &dir);
        let v = dx.to_vector();
        assert!(v.check(), "d==1 to_vector result fails check()");

        let r0 = dx.retract_hosvd(0.0);
        assert!((&r0 - &x).norm_lr(false) < 1e-12, "retract at alpha=0 == x");

        let alpha = 1e-2;
        let ra = dx.retract_hosvd(alpha);
        let linear = &x + &(&v * alpha);
        assert!(
            (&ra - &linear).norm_lr(false) < 1e-10,
            "d==1 retraction must be exact-linear (no truncation at the single-core rank target)"
        );
    }
}

#[cfg(test)]
mod tf_c4_su2_tests {
    use super::{Su2TtTangent, projective_transport};
    use crate::su2::expand::expand;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::su2::weingarten::tangent_basis;
    use crate::test_util::{randomized_su2_state, seeded};

    fn dense_axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi += alpha * xi;
        }
    }
    fn dense_nrm(a: &[f64]) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    fn members(two_s: usize) -> Vec<i64> {
        let s = two_s as i64;
        (0..=two_s).map(|i| -s + 2 * i as i64).collect()
    }

    fn dense_all_sz(x: &Su2TtVector<f64>) -> Vec<f64> {
        let mut out = Vec::new();
        for sz in members(x.space.two_s) {
            out.extend(expand(x, sz).to_dense());
        }
        out
    }

    fn frames(x: &Su2TtVector<f64>) -> (Su2TtVector<f64>, Su2TtVector<f64>) {
        let mut l = x.clone();
        l.left_orthogonalize(true);
        let mut r = x.clone();
        r.right_orthogonalize(true);
        (l, r)
    }

    #[test]
    fn tf_c4_su2_retract_dense_second_order() {
        let space = Su2Space::new(5, 3, 1);
        let mut rng = seeded(0xC4_51);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 2, &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let (bl, br) = frames(&x);
        let mut xi = Su2TtTangent::project(&bl, &br, &dir);
        let n = xi.norm();
        assert!(n > 1e-3, "anti-vacuity: projected direction vanished");
        xi.scale_mut(1.0 / n);

        let xd = dense_all_sz(&x);
        let xiv = dense_all_sz(&xi.to_vector());
        assert!(
            dense_nrm(&xiv) > 1e-6,
            "anti-vacuity: the embedded tangent must be nonzero in the dense image"
        );
        let resid = |alpha: f64| -> f64 {
            let got = dense_all_sz(&xi.retract_hosvd(alpha));
            let mut want = xd.clone();
            dense_axpy(&mut want, alpha, &xiv);
            let diff: Vec<f64> = got.iter().zip(&want).map(|(a, b)| a - b).collect();
            dense_nrm(&diff)
        };
        let (ra, rh) = (resid(1e-2), resid(5e-3));
        assert!(
            ra > 1e-9,
            "anti-vacuity: the truncated retraction must actually curve (residual {ra:e})"
        );
        let ratio = ra / rh.max(1e-300);
        assert!(
            (2.5..=6.0).contains(&ratio),
            "su2 retraction residual must be O(α²): r(α)/r(α/2) = {ratio} (r = {ra:e}, {rh:e})"
        );
    }

    #[test]
    fn tf_c4_su2_projective_transport_dense_value_oracle() {
        let space = Su2Space::new(4, 2, 0);
        let mut rng = seeded(0xC4_52);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 2, &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        let mut y: Su2TtVector<f64> = randomized_su2_state(space, 2, 2, &mut rng);
        y.scale_mut(1.0 / y.norm_lr(false));
        let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let (xl, xr) = frames(&x);
        let xi = Su2TtTangent::project(&xl, &xr, &dir);
        let z = xi.to_vector();

        let (yl, yr) = frames(&y);
        let basis = tangent_basis(&yl, &yr, false);
        assert!(!basis.is_empty(), "tangent basis at y must be nonempty");
        for (i, a) in basis.iter().enumerate() {
            for (j, b) in basis.iter().enumerate() {
                let td = a.dot(b);
                let vd = crate::su2::ops::dot(&a.to_vector(), &b.to_vector());
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (td - want).abs() <= 1e-10,
                    "premise: basis must be orthonormal in the tangent metric: ⟨b{i},b{j}⟩ = {td}"
                );
                assert!(
                    (vd - td).abs() <= 1e-10,
                    "premise: ι must be a g-isometry on the basis: tangent dot {td} vs embedded \
                     su2 dot {vd} at ({i},{j})"
                );
            }
        }

        let got = dense_all_sz(&projective_transport(&y, &xi).to_vector());
        let mut want = vec![0.0f64; got.len()];
        let mut mass = 0.0f64;
        for b in &basis {
            let c = crate::su2::ops::dot(&b.to_vector(), &z);
            mass += c * c;
            dense_axpy(&mut want, c, &dense_all_sz(&b.to_vector()));
        }
        assert!(
            mass.sqrt() > 1e-3,
            "anti-vacuity: the projection must retain O(1) mass, got {:e}",
            mass.sqrt()
        );
        let diff: Vec<f64> = got.iter().zip(&want).map(|(a, b)| a - b).collect();
        let err = dense_nrm(&diff) / dense_nrm(&want).max(1e-300);
        assert!(
            err <= 1e-9,
            "su2 projective_transport must equal the dense basis projection: rel err {err:e}"
        );
    }
}

#[cfg(test)]
mod shifted_tests;

#[cfg(test)]
mod gauge_fix_tests {
    use super::*;
    use crate::su2::qn::Su2Space;
    use crate::su2::riemannian::tangent_bases;
    use crate::test_util::randomized_su2_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn pin(
        qs: &[Su2TtTangent<f64>],
        reembed: impl Fn(&Su2TtVector<f64>) -> Su2TtTangent<f64>,
        what: &str,
    ) {
        for (i, q) in qs.iter().enumerate() {
            let mut g = q.clone();
            g.gauge_fix_orthogonal_mut();
            g.axpy(-1.0, q);
            assert!(
                g.norm() <= 1e-13 * q.norm(),
                "{what}: not the identity on q{i}"
            );
        }
        let mut w = qs[0].clone();
        w.for_each_coeff_mut(|c| *c = (*c * 7.3).sin());
        let mut g = w.clone();
        g.gauge_fix_orthogonal_mut();
        let mut gg = g.clone();
        gg.gauge_fix_orthogonal_mut();
        gg.axpy(-1.0, &g);
        assert!(
            gg.norm() <= 1e-13 * g.norm().max(1.0),
            "{what}: not idempotent"
        );
        let mut eg = w.clone();
        eg.axpy(-1.0, &g);
        let p = reembed(&w.to_vector());
        let mut ep = w.clone();
        ep.axpy(-1.0, &p);
        assert!(
            eg.norm() > 1e-2 && ep.norm() > 1e-2,
            "{what}: fixture precondition — w must sit off the gauge-fixed subspace"
        );
        let (mut og, mut op) = (0.0f64, 0.0f64);
        for q in &qs[1..] {
            og = og.max((q.dot(&eg) / (q.norm() * eg.norm())).abs());
            op = op.max((q.dot(&ep) / (q.norm() * ep.norm())).abs());
        }
        println!(
            "{what}: removed part's overlap with tangents: orthogonal {og:.2e}, project {op:.2e}"
        );
        assert!(
            og <= 1e-14,
            "{what}: the gauge fix is not centerweighted-orthogonal ({og:e})"
        );
        assert!(
            op >= 1e-4,
            "{what}: project∘to_vector came out orthogonal ({op:e}) — vacuous"
        );
    }

    #[test]
    fn su2_gauge_fix_orthogonal_at_flux_zero_and_on_the_triplet_space() {
        let space = Su2Space::new(4, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6F1B);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        let (bl, br) = tangent_bases(&x);
        let qs: Vec<Su2TtTangent<f64>> = (0..4)
            .map(|_| {
                let d: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
                Su2TtTangent::project(&bl, &br, &d)
            })
            .collect();
        pin(&qs, |v| Su2TtTangent::project(&bl, &br, v), "2S_flux 0");
        let space_out = space.with_two_s(2);
        let qs: Vec<Su2TtTangent<f64>> = (0..4)
            .map(|_| {
                let d: Su2TtVector<f64> = randomized_su2_state(space_out, 1, 2, &mut rng);
                Su2TtTangent::project_shifted(&bl, &br, 2, &d)
            })
            .collect();
        pin(
            &qs,
            |v| Su2TtTangent::project_shifted(&bl, &br, 2, v),
            "2S_flux 2",
        );
    }
}
