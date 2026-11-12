// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Tangent-space machinery for the abelian layer — required by `solvers::riemannian`, which uses
//! the tangent type, the retraction and the tangent `dot`/`norm`/`+`/`-`/scalar-`*` throughout.

use faer::Mat;
use faer::prelude::ReborrowMut;

use rayon::prelude::*;
use std::sync::Arc;

use crate::core::algebra::{assert_same_blocks, block_axpy, block_scale, step_abs2, step_dot};
use crate::core::factor::{
    Sketch, adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
};
use crate::core::{Family, SparseCore};
use crate::par;
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, RankGrid};
use crate::scalar::Scalar;
use crate::vector::TtVector;

type BlockPair<'a, T> = (&'a mut Mat<T>, &'a Mat<T>);

/// Gauge-fixed tangent vector at `base_l` (left-orthogonal)/`base_r` (right-orthogonal)
/// (`r` = the `d+1` bond rank grids of `base_l`, `components` = the `d` gauge-fixed component
/// cores).
#[derive(Clone)]
pub struct TtTangent<T: Scalar> {
    pub(crate) r: Vec<RankGrid>,
    pub(crate) r_col: Vec<RankGrid>,
    pub(crate) phi: Irrep,
    pub(crate) base_l: Arc<TtVector<T>>,
    pub(crate) base_r: Arc<TtVector<T>>,
    pub(crate) components: Vec<SparseCore<T>>,
    pub(crate) ncoeff: usize,
}

impl<T: Scalar> TtTangent<T> {
    pub(crate) fn from_parts(
        r: Vec<RankGrid>,
        base_l: impl Into<Arc<TtVector<T>>>,
        base_r: impl Into<Arc<TtVector<T>>>,
        components: Vec<SparseCore<T>>,
    ) -> Self {
        let r_col = r.clone();
        Self::from_parts_shifted(r, r_col, Irrep::IDENTITY, base_l, base_r, components)
    }

    pub(crate) fn from_parts_shifted(
        r: Vec<RankGrid>,
        r_col: Vec<RankGrid>,
        phi: Irrep,
        base_l: impl Into<Arc<TtVector<T>>>,
        base_r: impl Into<Arc<TtVector<T>>>,
        components: Vec<SparseCore<T>>,
    ) -> Self {
        let (base_l, base_r) = (base_l.into(), base_r.into());
        debug_assert_eq!(
            r.len(),
            r_col.len(),
            "from_parts_shifted: grid count mismatch"
        );
        let ncoeff = components.iter().map(|c| c.num_coeffs()).sum();
        TtTangent {
            r,
            r_col,
            phi,
            base_l,
            base_r,
            components,
            ncoeff,
        }
    }

    pub(crate) fn zero_shifted(
        phi: Irrep,
        r: Vec<RankGrid>,
        r_col: Vec<RankGrid>,
        base_l: impl Into<Arc<TtVector<T>>>,
        base_r: impl Into<Arc<TtVector<T>>>,
    ) -> Self {
        let base_l: Arc<TtVector<T>> = base_l.into();
        let space_out = base_l.space;
        let components = (0..space_out.d)
            .map(|k| SparseCore::zeros(space_out, k, &r[k], &r_col[k + 1]))
            .collect();
        Self::from_parts_shifted(r, r_col, phi, base_l, base_r, components)
    }

    /// The irrep flux of this tangent direction: [`Irrep::IDENTITY`] for an ordinary tangent
    /// vector, `φ` for a direction of the shifted space `T^φ` (whose embedding lives in
    /// `space.with_target(Γ0 ⊕ φ)`).
    #[inline]
    pub fn phi(&self) -> Irrep {
        self.phi
    }

    /// The components' COLUMN rank grids, `r_col[b] = rank()[b].relabel_irrep(phi)` — equal to
    /// [`Self::rank`] at `phi == IDENTITY`.
    #[inline]
    pub fn rank_col(&self) -> &[RankGrid] {
        &self.r_col
    }

    #[inline]
    pub(crate) fn refuse_if_shifted(&self, what: &str) {
        assert_eq!(
            self.phi,
            Irrep::IDENTITY,
            "{what} is undefined on a shifted tangent (phi = {:?}): x + alpha*xi is not a point \
             of any single typed space; embed with to_vector() instead",
            self.phi
        );
    }

    pub(crate) fn gauge_fix_orthogonal_mut(&mut self) {
        let d = self.components.len();
        let space = self.base_l.space;
        for k in 0..d.saturating_sub(1) {
            gauge_complement(&mut self.components[k], self.base_l.core(k), space, k + 1);
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

#[inline]
pub(crate) fn gauge_complement<T: Scalar>(
    comp: &mut SparseCore<T>,
    base_l_core: &SparseCore<T>,
    space: QnSpace,
    bond: usize,
) {
    debug_assert_eq!(
        comp.space, space,
        "gauge_complement: space is not the component's"
    );
    debug_assert_eq!(
        comp.k + 1,
        bond,
        "gauge_complement: bond is not the component's k + 1"
    );
    crate::core::weingarten::gauge_complement_mut(comp, base_l_core);
}

pub(crate) fn shifted_frames<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    phi: Irrep,
) -> (TtVector<T>, TtVector<T>) {
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "shifted_frames: the two frames must factorize the SAME state"
    );
    let space = base_l.space;
    let space_out = space.with_target(space.target.fuse(phi));
    let d = space.d;
    let mut u = TtVector::new(
        space_out,
        (0..d).map(|k| base_l.core(k).restamp(space_out)).collect(),
    );
    u.ortho = base_l.ortho;
    let mut v = TtVector::new(
        space_out,
        (0..d)
            .map(|k| base_r.core(k).clone().relabel_irrep_into(phi, space_out))
            .collect(),
    );
    v.ortho = base_r.ortho;
    (u, v)
}

pub(crate) fn shifted_channel_is_empty<T: Scalar>(seed: &Sketch<T>) -> bool {
    seed.blocks
        .iter()
        .all(|(_, b)| b.as_ref().is_none_or(|m| m.ncols() == 0))
}

impl<T: Scalar> TtTangent<T> {
    /// The gauge-fixed projection of `dir` onto the tangent space `T_x(M_r)` at `x`
    /// (represented by its left- and right-orthogonal factorizations `base_l`/`base_r`).
    pub fn project(base_l: &TtVector<T>, base_r: &TtVector<T>, dir: &TtVector<T>) -> Self {
        Self::project_impl(
            Arc::new(base_l.clone()),
            Arc::new(base_r.clone()),
            Irrep::IDENTITY,
            dir,
        )
    }

    pub(crate) fn project_shared(
        base_l: &Arc<TtVector<T>>,
        base_r: &Arc<TtVector<T>>,
        dir: &TtVector<T>,
    ) -> Self {
        Self::project_impl(base_l.clone(), base_r.clone(), Irrep::IDENTITY, dir)
    }

    /// `P^φ(dir)`: the gauge-fixed projection of `dir ∈ space_out` onto the SHIFTED tangent space
    /// `T^φ` at the frames `base_l`/`base_r`.
    pub fn project_shifted(
        base_l: &TtVector<T>,
        base_r: &TtVector<T>,
        phi: Irrep,
        dir: &TtVector<T>,
    ) -> Self {
        let (u, v) = shifted_frames(base_l, base_r, phi);
        Self::project_impl(Arc::new(u), Arc::new(v), phi, dir)
    }

    pub(crate) fn project_shifted_at_frames(
        u: &Arc<TtVector<T>>,
        v: &Arc<TtVector<T>>,
        phi: Irrep,
        dir: &TtVector<T>,
    ) -> Self {
        Self::project_impl(u.clone(), v.clone(), phi, dir)
    }

    fn project_impl(
        base_l: Arc<TtVector<T>>,
        base_r: Arc<TtVector<T>>,
        phi: Irrep,
        dir: &TtVector<T>,
    ) -> Self {
        let space_out = base_l.space;
        let space = space_out.with_target(space_out.target.fuse(phi));
        let d = space.d;
        let r = base_l.ranks.clone();
        let r_col: Vec<RankGrid> = r.iter().map(|g| g.relabel_irrep(phi)).collect();
        debug_assert_eq!(
            base_r.ranks, r_col,
            "project: the right frame must carry the phi-relabeled ranks of the left one"
        );
        assert_eq!(
            dir.space, space_out,
            "project: the direction lives in {:?} but the shifted tangent space is in {:?}",
            dir.space, space_out
        );

        let mut p: Vec<Sketch<T>> = Vec::with_capacity(d);
        p.push(Sketch::identity(space, 0, 1));
        for kc in 0..d.saturating_sub(1) {
            let pd = mul_sketch_left(&p[kc], dir.core(kc));
            let next = adjoint_core_times_core(base_l.core(kc), &pd);
            p.push(next);
        }

        let mut q = Sketch::identity(space_out, d, 1);
        if shifted_channel_is_empty(&q) {
            return Self::zero_shifted(phi, r, r_col, base_l, base_r);
        }
        let mut components: Vec<Option<SparseCore<T>>> = (0..d).map(|_| None).collect();
        for kc in (0..d).rev() {
            let dq = mul_sketch_right(dir.core(kc), &q);
            let mut comp = mul_sketch_left(&p[kc], &dq);

            if kc < d - 1 {
                gauge_complement(&mut comp, base_l.core(kc), space_out, kc + 1);
            }

            if kc > 0 {
                q = core_times_adjoint_core(&dq, base_r.core(kc));
            }
            components[kc] = Some(comp);
        }

        let components: Vec<SparseCore<T>> = components.into_iter().map(|c| c.unwrap()).collect();
        Self::from_parts_shifted(r, r_col, phi, base_l, base_r, components)
    }

    /// Left- AND right-orthogonalize copies of `base`, then [`Self::project`].
    pub fn from_base(base: &TtVector<T>, dir: &TtVector<T>) -> Self {
        let mut base_l = base.clone();
        base_l.left_orthogonalize(true);
        let mut base_r = base.clone();
        base_r.right_orthogonalize(true);
        Self::project(&base_l, &base_r, dir)
    }

    /// The embedding `ι(ξ)`: the tangent vector as a full TT in the output space,
    /// interior bonds doubled (`base_r` atop `component` atop `base_l`, block-triangular).
    pub fn to_vector(&self) -> TtVector<T> {
        let space = self.base_l.space;
        let d = space.d;
        if d == 1 {
            return TtVector::new(space, vec![self.components[0].clone()]);
        }
        let r = &self.r;
        let rc = &self.r_col;
        let mut cores = Vec::with_capacity(d);
        for kc in 0..d {
            let has_v = kc > 0;
            let has_u = kc < d - 1;
            let mut rr = RankGrid::zeros(&space);
            let mut cr = RankGrid::zeros(&space);
            for s in space.bond_sectors(kc) {
                rr[s] = r[kc][s] + if has_v { rc[kc][s] } else { 0 };
            }
            for s in space.bond_sectors(kc + 1) {
                cr[s] = rc[kc + 1][s] + if has_u { r[kc + 1][s] } else { 0 };
            }
            let mut core = SparseCore::zeros(space, kc, &rr, &cr);
            let g = space.pg.irrep_of(kc);
            for l in space.bond_sectors(kc) {
                for f in Family::ALL {
                    let t = f.apply(l, g);
                    if !space.contains(kc + 1, t) {
                        continue;
                    }
                    let i0 = if has_v { rc[kc][l] } else { 0 };
                    let j0 = rc[kc + 1][t];
                    let v_band = has_v.then(|| self.base_r.core(kc).block(f, l)).flatten();
                    if let Some(b) = v_band {
                        core.copy_block_into(f, l, 0, 0, b);
                    }
                    if let Some(b) = self.components[kc].block(f, l) {
                        core.copy_block_into(f, l, i0, 0, b);
                    }
                    let u_band = has_u.then(|| self.base_l.core(kc).block(f, l)).flatten();
                    if let Some(b) = u_band {
                        core.copy_block_into(f, l, i0, j0, b);
                    }
                }
            }
            cores.push(core);
        }
        TtVector::new(space, cores)
    }

    /// The fixed-rank retraction `round(x + α·dx, r)`,
    /// built directly at doubled rank, then HOSVD-truncated back to `r` (`round_ranks_mut`, "no
    /// chopping"). Same block layout as [`Self::to_vector`] except both base pieces are present
    /// everywhere (representing `x` redundantly) and the last core's bottom band carries
    /// `base_l + α·comp`.
    pub fn retract_hosvd(&self, alpha: T) -> TtVector<T> {
        self.refuse_if_shifted("retract_hosvd");
        let mut v = self.moved_point(alpha);
        v.round_ranks_mut(&self.r);
        v
    }

    /// The rank-ADAPTIVE retraction `round(x + α·dx, ϵ, chi)`: the same doubled-rank point as
    /// [`Self::retract_hosvd`], truncated by an energy budget (`round_mut`) instead of back to
    /// the base grid, so the result's ranks follow what the moved point needs — up to the POOLED
    /// cap `chi` on each bond's TOTAL dimension `Σ_q r[q]` ([`crate::qn::Pooled::NONE`] = uncapped) and never
    /// above the representing rank `2r` — and drop where a
    /// Schmidt value falls under the budget. What GAD's rank adaptation retracts through.
    pub fn retract_round(
        &self,
        alpha: T,
        eps: f64,
        chi: crate::qn::Pooled,
        mode: crate::round::RoundMode,
    ) -> TtVector<T> {
        self.refuse_if_shifted("retract_round");
        let mut v = self.moved_point(alpha);
        v.round_mut(eps, chi, mode);
        v
    }

    /// `x + α·dx` as a TT vector at DOUBLED rank (`2r` on every interior bond), exact: the block
    /// layout of [`Self::to_vector`] with both base pieces present everywhere (representing `x`
    /// redundantly) and the last core's bottom band carrying `base_l + α·comp`. Not
    /// orthogonalized; the retractions round it.
    pub fn moved_point(&self, alpha: T) -> TtVector<T> {
        self.refuse_if_shifted("moved_point");
        let space = self.base_l.space;
        let d = space.d;
        if d == 1 {
            let mut core = self.base_l.core(0).clone();
            core.axpy(alpha, &self.components[0]);
            return TtVector::new(space, vec![core]);
        }
        let r = &self.r;
        let mut cores = Vec::with_capacity(d);
        for kc in 0..d {
            let row_mul = if kc == 0 { 1 } else { 2 };
            let col_mul = if kc == d - 1 { 1 } else { 2 };
            let mut rr = RankGrid::zeros(&space);
            let mut cr = RankGrid::zeros(&space);
            for s in space.bond_sectors(kc) {
                rr[s] = row_mul * r[kc][s];
            }
            for s in space.bond_sectors(kc + 1) {
                cr[s] = col_mul * r[kc + 1][s];
            }
            let mut core = SparseCore::zeros(space, kc, &rr, &cr);
            let g = space.pg.irrep_of(kc);
            for l in space.bond_sectors(kc) {
                for f in Family::ALL {
                    let t = f.apply(l, g);
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
        TtVector::new(space, cores)
    }

    /// `dot(dx, dy) = Σ_k dot(comp_x[k], comp_y[k])`.
    pub fn dot(&self, other: &Self) -> T {
        debug_assert_eq!(self.r, other.r, "dot: tangent ranks must agree");
        debug_assert_eq!(
            self.r_col, other.r_col,
            "dot: tangent column grids must agree"
        );
        debug_assert_eq!(self.phi, other.phi, "dot: tangent flux labels must agree");
        self.assert_zippable(other, "TtTangent::dot");
        if !par::par_vector_ops(self.ncoeff) {
            let mut acc = T::zero();
            for k in 0..self.components.len() {
                acc += self.components[k].dot_cores(&other.components[k]);
            }
            return acc;
        }
        let (a, b) = (&self.components, &other.components);
        let parts: Vec<T> = (0..4 * a.len())
            .into_par_iter()
            .map(|i| {
                let (k, f) = (i / 4, i % 4);
                step_dot(&a[k], &b[k], Family::ALL[f])
            })
            .collect();
        parts.into_iter().fold(T::zero(), |acc, p| acc + p)
    }

    /// `norm(dx) = sqrt(Σ_k ‖comp[k]‖²)`. Block-parallel like [`Self::dot`].
    pub fn norm(&self) -> f64 {
        if !par::par_vector_ops(self.ncoeff) {
            let mut s = 0.0f64;
            for c in &self.components {
                s += c.abs2();
            }
            return s.sqrt();
        }
        let a = &self.components;
        let parts: Vec<f64> = (0..4 * a.len())
            .into_par_iter()
            .map(|i| step_abs2(&a[i / 4], Family::ALL[i % 4]))
            .collect();
        parts.into_iter().sum::<f64>().sqrt()
    }

    /// `dx + dy`: componentwise [`Self::axpy`] with `a = 1`; asserts equal ranks, column grids and
    /// flux labels (through [`Self::axpy`]).
    pub fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.axpy(T::one(), other);
        result
    }

    /// `dx - dy`. Same agreement asserts as [`Self::add`].
    pub fn sub(&self, other: &Self) -> Self {
        let mut result = self.clone();
        result.axpy(T::from_f64(-1.0), other);
        result
    }

    /// `a * dx`: scale every component core.
    pub fn scale(&self, a: T) -> Self {
        let mut result = self.clone();
        result.scale_mut(a);
        result
    }

    /// `dx ← a·dx`: in-place [`Self::scale`]. Block-parallel above
    /// [`par::PAR_MIN_COEFFS`] — writes are one disjoint stored block per task.
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

    /// `self += a * other`. Block-parallel above
    /// [`par::PAR_MIN_COEFFS`] — the MGS partner of [`Self::dot`]; writes are disjoint per task,
    /// and elementwise updates carry no reassociation at all.
    pub fn axpy(&mut self, a: T, other: &Self) {
        debug_assert_eq!(self.r, other.r, "axpy: tangent ranks must agree");
        debug_assert_eq!(
            self.r_col, other.r_col,
            "axpy: tangent column grids must agree"
        );
        debug_assert_eq!(self.phi, other.phi, "axpy: tangent flux labels must agree");
        self.assert_zippable(other, "TtTangent::axpy");
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

    /// `self = a*other + b*self`. Block-parallel above
    /// [`par::PAR_MIN_COEFFS`], fusing the scale and the axpy into ONE fork.
    pub fn axpby(&mut self, a: T, other: &Self, b: T) {
        debug_assert_eq!(self.r, other.r, "axpby: tangent ranks must agree");
        debug_assert_eq!(
            self.r_col, other.r_col,
            "axpby: tangent column grids must agree"
        );
        debug_assert_eq!(self.phi, other.phi, "axpby: tangent flux labels must agree");
        self.assert_zippable(other, "TtTangent::axpby");
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

    /// `rank(dx)`: the `d+1` bond rank grids. Genuinely trivial field access.
    #[inline]
    pub fn rank(&self) -> &[RankGrid] {
        &self.r
    }

    /// The `k`-th gauge-fixed component core. `pub` rather than `pub(crate)`: the gauge-condition
    /// and transport tests read it directly, and the `tangent_hessian` binary is a separate integration-test
    /// crate that cannot see `pub(crate)` items.
    #[inline]
    pub fn component(&self, k: usize) -> &SparseCore<T> {
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
            "TtTangent: cached ncoeff out of date"
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
        panic!("TtTangent::coeff: index {idx} out of range");
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
        panic!("TtTangent::set_coeff: index {idx} out of range");
    }
}

/// Projective transport: re-project the train `dx` represents onto the tangent space at `y`. Not
/// called by the CG driver itself (it inlines the transport as `TtTangent::project(&base_l,
/// &base_r, &eta.to_vector())`), but exercised by the transport tests.
/// Refuses a SHIFTED tangent: transport re-projects at a point of the SAME manifold, and a shifted
/// direction has no such manifold (see [`TtTangent::moved_point`]).
pub fn projective_transport<T: Scalar>(y: &TtVector<T>, dx: &TtTangent<T>) -> TtTangent<T> {
    dx.refuse_if_shifted("projective_transport");
    let z = dx.to_vector();
    TtTangent::from_base(y, &z)
}

const MGS_REL_TOL: f64 = 1e-8;

const MGS_EMPTY_SLOT: f64 = 1e-12;

const MGS_NOISE_FLOOR: f64 = 1e-13;

/// Dimension of the gauge-fixed tangent parametrization at `base_l`'s rank grid (radial mode
/// INCLUDED): raw component-slot dimensions minus the left-gauge constraint rows
/// `Σ_{k<d−1} Σ_s r_{k+1}[s]²`. Structural (shape-determined, independent of the point's
/// values): an UPPER bound on `tangent_basis(…, true).len() + 1`, with equality at a numerically
/// full-rank point. At a truncated or rank-deficient point the realized span sits below it (the
/// enumerator's MGS culls the embedding-null candidates). The Krylov drivers use it as
/// the exhaustion cap — a Krylov space can never legitimately exceed it, and β-threshold
/// breakdown tests alone are fooled once the true residual reaches the noise floor (ghost
/// directions) — so below it a run exhausts the realized space without reaching the cap.
pub fn tangent_dim<T: Scalar>(base_l: &TtVector<T>) -> usize {
    tangent_dim_grids(&base_l.space, &base_l.ranks, Irrep::IDENTITY)
}

/// Dimension of the gauge-fixed parametrization of the SHIFTED tangent space `T^φ` at the rank
/// grids `r`, from the grids alone:
pub fn tangent_dim_grids(space: &QnSpace, r: &[RankGrid], phi: Irrep) -> usize {
    let d = space.d;
    let shift =
        |s: crate::qn::Sector| crate::qn::Sector::with_irrep(s.nup, s.ndn, s.irrep.fuse(phi));
    let mut raw = 0usize;
    let mut gauge = 0usize;
    for k in 0..d {
        let g = space.pg.irrep_of(k);
        for l in space.bond_sectors(k) {
            for f in Family::ALL {
                let t = f.apply(l, g);
                if space.contains(k + 1, t) {
                    raw += r[k][l] * r[k + 1][shift(t)];
                }
            }
        }
        if k < d - 1 {
            for s in space.bond_sectors(k + 1) {
                gauge += r[k + 1][s] * r[k + 1][shift(s)];
            }
        }
    }
    raw - gauge
}

/// Deterministic `⟨·,·⟩`-orthonormal basis of the tangent space `T_xM` at the gauge frames
/// `base_l`/`base_r` — the canonical-basis input (GHC Table 1 step 1). Enumerates the
/// elementary component-core directions slot by slot (gauge-projected onto the `base_l` complement
/// for `k < d−1`), then modified Gram–Schmidt in the tangent metric, dropping numerically dependent
/// directions (a relative cull and two absolute floors). The enumeration is
/// complete: each slot's gauged subspace is exactly the span of its projected elementary cores,
/// and distinct slots are `dot`-orthogonal by the block-diagonal metric.
pub fn tangent_basis<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    exclude_radial: bool,
) -> Vec<TtTangent<T>> {
    tangent_basis_impl(base_l, base_r, Irrep::IDENTITY, exclude_radial)
}

/// Deterministic orthonormal basis of the SHIFTED tangent space `T^φ` at the frames
/// `base_l`/`base_r` (the ORIGINAL frames of `x`) — [`tangent_basis`]'s enumerator, one shared
/// body, with the component slots enumerated over the RELABELED column grid
/// (`j in 0..r_col[k+1][t]`) and every purification routed through
/// [`TtTangent::project_shifted`].
pub fn tangent_basis_shifted<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    phi: Irrep,
) -> Vec<TtTangent<T>> {
    tangent_basis_impl(base_l, base_r, phi, false)
}

fn tangent_basis_impl<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    phi: Irrep,
    exclude_radial: bool,
) -> Vec<TtTangent<T>> {
    use crate::core::weingarten::gauge_complement_mut;
    let space = base_l.space;
    let d = space.d;
    let r = base_l.ranks.clone();
    let r_col: Vec<RankGrid> = r.iter().map(|g| g.relabel_irrep(phi)).collect();
    let (frame_l, frame_r) = shifted_frames(base_l, base_r, phi);
    let (frame_l, frame_r) = (Arc::new(frame_l), Arc::new(frame_r));
    let space_out = frame_l.space;
    let zero_components = || -> Vec<SparseCore<T>> {
        (0..d)
            .map(|k| SparseCore::zeros(space_out, k, &r[k], &r_col[k + 1]))
            .collect()
    };

    let mut basis: Vec<TtTangent<T>> = Vec::new();
    if exclude_radial {
        debug_assert_eq!(
            phi,
            Irrep::IDENTITY,
            "there is no radial mode in a shifted tangent space"
        );
        let mut radial = TtTangent::project(base_l, base_r, base_l);
        let n = radial.norm();
        if n > 0.0 {
            radial.scale_mut(T::from_f64(1.0 / n));
            basis.push(radial);
        }
    }
    let n_seeds = basis.len();
    let dim = tangent_dim_grids(&space, &r, phi);
    let trace = crate::knobs::TB_TRACE.get();

    'enumerate: for k in 0..d {
        let g = space.pg.irrep_of(k);
        for l in space.bond_sectors(k) {
            for f in Family::ALL {
                let t = f.apply(l, g);
                if !space.contains(k + 1, t) {
                    continue;
                }
                for j in 0..r_col[k + 1][t] {
                    for i in 0..r[k][l] {
                        if basis.len() >= dim {
                            break 'enumerate;
                        }
                        let mut comps = zero_components();
                        {
                            let Some(mut blk) = comps[k].block_mut(f, l) else {
                                continue;
                            };
                            *blk.rb_mut().get_mut(i, j) = T::one();
                        }
                        if k < d - 1 {
                            gauge_complement_mut(&mut comps[k], base_l.core(k));
                        }
                        let u = TtTangent::from_parts_shifted(
                            r.clone(),
                            r_col.clone(),
                            phi,
                            frame_l.clone(),
                            frame_r.clone(),
                            comps,
                        );
                        let n_raw = u.norm();
                        let mut u = TtTangent::project_impl(
                            frame_l.clone(),
                            frame_r.clone(),
                            phi,
                            &u.to_vector(),
                        );
                        let n0 = u.norm();
                        if n0 <= MGS_EMPTY_SLOT {
                            if trace {
                                eprintln!(
                                    "  tb k={k} i={i} j={j} n_raw={n_raw:.3e} n0={n0:.3e} EMPTY"
                                );
                            }
                            continue;
                        }
                        for _ in 0..2 {
                            for b in &basis {
                                let c = b.dot(&u);
                                u.axpy(c * T::from_f64(-1.0), b);
                            }
                        }
                        let mut u = TtTangent::project_impl(
                            frame_l.clone(),
                            frame_r.clone(),
                            phi,
                            &u.to_vector(),
                        );
                        let nrm = u.norm();
                        let accept = nrm > MGS_REL_TOL * n0 && nrm > MGS_NOISE_FLOOR;
                        if trace {
                            eprintln!(
                                "  tb k={k} i={i} j={j} n_raw={n_raw:.3e} n0={n0:.3e} \
                                 nrm={nrm:.3e} rel={:.3e} {}",
                                nrm / n0,
                                if accept { "ACCEPT" } else { "cull" }
                            );
                        }
                        if accept {
                            u.scale_mut(T::from_f64(1.0 / nrm));
                            basis.push(u);
                        }
                    }
                }
            }
        }
    }
    basis.split_off(n_seeds)
}

/// The smallest bond Schmidt value at the frames `base_l`/`base_r` — the σ_min that every leakage
/// number reported by [`project_irrep`] must be quoted beside.
pub fn min_schmidt<T: Scalar>(base_l: &TtVector<T>, base_r: &TtVector<T>) -> f64 {
    use crate::core::weingarten::{FloorPolicy, bond_gauges};
    bond_gauges(base_l, base_r, FloorPolicy::default_hard())
        .iter()
        .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
        .fold(f64::INFINITY, f64::min)
}

/// `P_Γ ξ = (1/|G|) Σ_h χ_Γ(h) U_h ξ` — the component of a tangent direction transforming as `Γ`.
pub fn project_irrep<T: Scalar>(
    xi: &TtTangent<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    pg: crate::pointgroup::PointGroup,
    gamma: crate::pointgroup::Irrep,
) -> (TtTangent<T>, f64) {
    let order = pg.order();
    let scale = 1.0 / order as f64;
    let v = xi.to_vector();
    let (bl, br) = (Arc::new(base_l.clone()), Arc::new(base_r.clone()));
    let mut acc: Option<TtTangent<T>> = None;
    for hi in 0..order {
        let h = crate::pointgroup::Irrep::new(hi);
        let ux = crate::vector::apply_group_element(&v, pg, h);
        let mut term = TtTangent::project_shared(&bl, &br, &ux);
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
mod t8d_tests {
    use super::TtTangent;
    use crate::TtVector;
    use crate::qn::QnSpace;
    use crate::round::RoundMode;
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t8d_tangent_random_fd_ratio() {
        let d = 10usize;
        let space = QnSpace::from_n_two_sz(d, 6, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8D10);
        let mut x: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let mut dir: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
        dir.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        dir.scale_mut(1.0 / dir.norm());

        let mut base_l = x.clone();
        base_l.left_orthogonalize(true);
        let mut base_r = x.clone();
        base_r.right_orthogonalize(true);
        let dx = TtTangent::project(&base_l, &base_r, &dir);
        let dxv = dx.to_vector();

        let fd_at = |eps: f64| -> TtVector<f64> {
            let mut p = &x + &(&dir * eps);
            p.round_ranks_mut(&x.ranks);
            &(&p - &x) * (1.0 / eps)
        };
        let e1 = (&dxv - &fd_at(1e-6)).norm_lr(false);
        let e2 = (&dxv - &fd_at(2e-6)).norm_lr(false);
        assert!((e2 / e1 - 2.0).abs() < 0.1, "FD ratio e2/e1 = {}", e2 / e1);
    }

    #[test]
    fn t8d_moved_point_and_retract_round_oracle() {
        let d = 6usize;
        let space = QnSpace::new(d, 2, 2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8D30);
        let mut x: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let mut dir: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
        dir.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        dir.scale_mut(1.0 / dir.norm());
        let mut base_l = x.clone();
        base_l.left_orthogonalize(true);
        let mut base_r = x.clone();
        base_r.right_orthogonalize(true);
        let dx = TtTangent::project(&base_l, &base_r, &dir);
        let alpha = 0.37;

        let exact = &x + &(&dx.to_vector() * alpha);
        let moved = dx.moved_point(alpha);
        let err = (&moved - &exact).norm_lr(false) / exact.norm_lr(false);
        assert!(
            err < 1e-13,
            "moved_point ≠ x + α·dx: relative error {err:.3e}"
        );
        let pooled = |v: &TtVector<f64>, b: usize| -> usize {
            space
                .bond_sectors(b)
                .into_iter()
                .map(|s| v.ranks[b][s])
                .sum()
        };
        for b in 1..d {
            assert_eq!(
                pooled(&moved, b),
                2 * pooled(&x, b),
                "moved point is at doubled rank"
            );
        }

        let fine = dx.retract_round(alpha, 1e-14, crate::qn::Pooled::NONE, RoundMode::Rel);
        let err = (&fine - &exact).norm_lr(false) / exact.norm_lr(false);
        assert!(
            err < 1e-11,
            "retract_round at ϵ = 1e-14 must reproduce x + α·dx: {err:.3e}"
        );
        for b in 1..d {
            assert!(
                pooled(&fine, b) <= 2 * pooled(&x, b),
                "growth is bounded by 2r"
            );
        }

        let mut hosvd = dx.moved_point(alpha);
        hosvd.round_ranks_mut(&dx.r);
        let fixed = dx.retract_hosvd(alpha);
        let err = (&hosvd - &fixed).norm_lr(false);
        assert!(
            err < 1e-13,
            "retract_hosvd must be round_ranks(moved_point): {err:.3e}"
        );
        for b in 0..=d {
            assert_eq!(
                fixed.ranks[b], x.ranks[b],
                "retract_hosvd keeps the base grid"
            );
        }

        let coarse = dx.retract_round(alpha, 0.5, crate::qn::Pooled::NONE, RoundMode::Rel);
        let shrunk = (1..d).any(|b| pooled(&coarse, b) < pooled(&x, b));
        assert!(
            shrunk,
            "a 0.5 budget must shrink some bond below the base grid"
        );
    }

    #[test]
    fn t8d_tangent_gauge_condition() {
        let d = 6usize;
        let space = QnSpace::new(d, 2, 2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8D20);
        let x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
        let dir: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
        let mut base_l = x.clone();
        base_l.left_orthogonalize(true);
        let mut base_r = x.clone();
        base_r.right_orthogonalize(true);
        let dx = TtTangent::project(&base_l, &base_r, &dir);

        for k in 0..d - 1 {
            for s in space.bond_sectors(k + 1) {
                let l = base_l.core(k).vertical_unfolding(s);
                let c = dx.component(k).vertical_unfolding(s);
                let mut n2 = 0.0f64;
                for j in 0..c.ncols() {
                    for i in 0..l.ncols() {
                        let mut acc = 0.0f64;
                        for kk in 0..l.nrows() {
                            acc += *l.get(kk, i) * *c.get(kk, j);
                        }
                        n2 += acc * acc;
                    }
                }
                assert!(
                    n2.sqrt() < 1e-12,
                    "gauge condition at bond {}, sector {:?}: {}",
                    k + 1,
                    s,
                    n2.sqrt()
                );
            }
        }
    }
}

#[cfg(test)]
mod t8l_tests {
    use crate::TtVector;
    use crate::qn::QnSpace;
    use crate::solvers::riemannian::{normalize_mut, tangent_bases};
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t8l_tangent_basis_span_is_scale_free() {
        use crate::core::tangent::{tangent_basis, tangent_dim};
        for (d, nup, ndn, eps, exact) in [
            (4usize, 2usize, 1usize, 1e-4f64, true),
            (4, 2, 1, 1e-6, true),
            (4, 2, 1, 1e-10, false),
            (3, 1, 1, 1e-6, true),
        ] {
            let space = QnSpace::new(d, nup, ndn);
            let count_at = |scale: f64| -> (usize, usize) {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD3AD_BEEF);
                let x1: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
                let mut x2: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
                x2.scale_mut(scale);
                let mut x = &x1 + &x2;
                normalize_mut(&mut x);
                let (base_l, base_r) = tangent_bases(&x);
                (
                    tangent_basis(&base_l, &base_r, true).len(),
                    tangent_dim(&base_l),
                )
            };
            let (n_ref, _) = count_at(1.0);
            let (n_eps, dim) = count_at(eps);
            assert!(
                n_eps < dim,
                "d={d} eps={eps:.0e}: enumerator returned {} vectors (radial included), tangent_dim \
                 is {dim} (over-acceptance ⇒ rank-deficient Grams)",
                n_eps + 1
            );
            assert!(n_eps > 1, "d={d} eps={eps:.0e}: span collapsed");
            if exact {
                assert_eq!(
                    n_eps, n_ref,
                    "d={d} eps={eps:.0e}: accepted count {n_eps} vs O(1)-weight reference {n_ref} — \
                     a resolvable direction was culled (span not scale-free)"
                );
            }
        }
    }
}

#[cfg(test)]
mod tji_project_tests {
    use crate::core::tangent::{TtTangent, project_irrep};
    use crate::core::weingarten::{FloorPolicy, bond_gauges};
    use crate::pointgroup::{Irrep, PointGroup};
    use crate::qn::QnSpace;
    use crate::solvers::riemannian::{normalize_mut, tangent_bases};
    use crate::test_util::randomized_state;
    use crate::vector::{TtVector, apply_group_element, symmetry_defect};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::sync::OnceLock;

    struct PgFixture {
        #[allow(dead_code)]
        d: usize,
        pg: PointGroup,
        space_c1: QnSpace,
        gamma0: Irrep,
    }

    fn pg_fixture() -> &'static PgFixture {
        static CELL: OnceLock<PgFixture> = OnceLock::new();
        CELL.get_or_init(|| {
            let d = 4usize;
            let pg = PointGroup::new(
                2,
                &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
            );
            let space_c1 = QnSpace::new(d, 2, 1);
            PgFixture {
                d,
                pg,
                space_c1,
                gamma0: Irrep::IDENTITY,
            }
        })
    }

    fn pure_reference(x: &TtVector<f64>, pg: PointGroup, gamma0: Irrep) -> TtVector<f64> {
        let mut acc: Option<TtVector<f64>> = None;
        for h in 0..pg.order() {
            let h = Irrep::new(h);
            let term = &apply_group_element(x, pg, h) * gamma0.character(h);
            acc = Some(match acc {
                None => term,
                Some(a) => &a + &term,
            });
        }
        let mut out = acc.expect("pg.order() >= 1");
        normalize_mut(&mut out);
        out
    }

    fn sigma_min_at(base_l: &TtVector<f64>, base_r: &TtVector<f64>) -> f64 {
        bond_gauges(base_l, base_r, FloorPolicy::default_hard())
            .iter()
            .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn tji_pg_project_irrep_resolves_identity() {
        let fx = pg_fixture();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE103);
        let x_rand: TtVector<f64> = randomized_state(fx.space_c1, 1, 2, &mut rng);
        let x_pure = pure_reference(&x_rand, fx.pg, fx.gamma0);
        let (defect_gamma, defects) = symmetry_defect(&x_pure, fx.pg);
        let worst = defects.iter().cloned().fold(0.0f64, f64::max);
        assert_eq!(defect_gamma, fx.gamma0);
        let (base_l, base_r) = tangent_bases(&x_pure);
        let sigma_min = sigma_min_at(&base_l, &base_r);
        println!(
            "tji_pg_project_irrep_resolves_identity: base defect={worst:e} sigma_min={sigma_min:e}"
        );
        assert!(
            worst <= 1e-8,
            "the train-level projector must give an essentially exact symmetry eigenstate: {worst:e}"
        );

        let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0xE104);
        let dir: TtVector<f64> = randomized_state(fx.space_c1, 1, 2, &mut rng2);
        let xi = TtTangent::project(&base_l, &base_r, &dir);
        let n = xi.norm();
        assert!(n > 0.0, "degenerate test direction");

        let mut sum: Option<TtTangent<f64>> = None;
        let mut weights = Vec::new();
        let mut worst_leak = 0.0f64;
        for h in 0..fx.pg.order() {
            let (p, leak) = project_irrep(&xi, &base_l, &base_r, fx.pg, Irrep::new(h));
            weights.push((p.norm() / n).powi(2));
            worst_leak = worst_leak.max(leak);
            sum = Some(match sum {
                None => p,
                Some(a) => a.add(&p),
            });
        }
        let resid = xi.sub(&sum.unwrap()).norm() / n;
        println!(
            "tji_pg_project_irrep_resolves_identity: resid={resid:e} weights={weights:?} sigma_min={sigma_min:e}"
        );
        assert!(
            resid <= 1e-8,
            "Sum_Gamma P_Gamma != id: residual {resid:e} (sigma_min={sigma_min:e})"
        );
        let total: f64 = weights.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1e-8,
            "Sum_Gamma ||P_Gamma xi||^2/||xi||^2 = {total}, blocks are not orthogonal (sigma_min={sigma_min:e})"
        );
        let live = weights.iter().filter(|w| **w > 1e-6).count();
        assert!(
            live > 1,
            "anti-vacuity: a random direction landed in one irrep only: {weights:?}"
        );

        for h in 0..fx.pg.order() {
            let gamma = Irrep::new(h);
            let (p, _leak) = project_irrep(&xi, &base_l, &base_r, fx.pg, gamma);
            if p.norm() <= 1e-12 * xi.norm() {
                continue;
            }
            let (pp, leak2) = project_irrep(&p, &base_l, &base_r, fx.pg, gamma);
            assert!(
                p.sub(&pp).norm() <= 1e-8 * p.norm(),
                "P_Gamma not idempotent for {gamma:?}"
            );
            assert!(
                leak2 <= 1e-8,
                "leakage {leak2:e} on an already-pure vector (Gamma={gamma:?}, sigma_min={sigma_min:e})"
            );
        }
    }
}

#[cfg(test)]
mod tf_c4_tests {
    use super::{TtTangent, projective_transport, tangent_basis};
    use crate::qn::{QnSpace, max_ranks};
    use crate::test_util::randomized_state;
    use crate::vector::{TtVector, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn dense_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }
    fn dense_axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
        for (yi, xi) in y.iter_mut().zip(x) {
            *yi += alpha * xi;
        }
    }
    fn dense_nrm(a: &[f64]) -> f64 {
        a.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
    fn dense_dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn frames(x: &TtVector<f64>) -> (TtVector<f64>, TtVector<f64>) {
        let mut l = x.clone();
        l.left_orthogonalize(true);
        let mut r = x.clone();
        r.right_orthogonalize(true);
        (l, r)
    }

    #[test]
    fn tf_c4_retract_dense_identity_at_full_rank() {
        let space = QnSpace::new(4, 2, 1);
        let grids = max_ranks(&space);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC4_01);
        let mut x: TtVector<f64> = tt_randn(space, &grids, &mut rng);
        x.scale_mut(1.0 / x.norm());
        let dir: TtVector<f64> = tt_randn(space, &grids, &mut rng);
        let (bl, br) = frames(&x);
        let xi = TtTangent::project(&bl, &br, &dir);
        assert!(
            xi.norm() > 1e-3,
            "anti-vacuity: projected direction vanished"
        );

        let xd = x.to_dense();
        let xiv = xi.to_vector().to_dense();
        for alpha in [0.7f64, -1.3] {
            let got = xi.retract_hosvd(alpha).to_dense();
            let mut want = xd.clone();
            dense_axpy(&mut want, alpha, &xiv);
            let err = dense_nrm(&dense_sub(&got, &want)) / dense_nrm(&want);
            assert!(
                err <= 1e-12,
                "retract_hosvd at full rank must be exactly x + α·ι(ξ): rel err {err:e} at \
                 α = {alpha}"
            );
        }
    }

    #[test]
    fn tf_c4_retract_dense_second_order_at_truncated_rank() {
        let space = QnSpace::new(5, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC4_02);
        let mut x: TtVector<f64> = randomized_state(space, 2, 2, &mut rng);
        x.scale_mut(1.0 / x.norm());
        let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let (bl, br) = frames(&x);
        let mut xi = TtTangent::project(&bl, &br, &dir);
        let n = xi.norm();
        assert!(n > 1e-3, "anti-vacuity: projected direction vanished");
        xi.scale_mut(1.0 / n);

        let xd = x.to_dense();
        let xiv = xi.to_vector().to_dense();
        let resid = |alpha: f64| -> f64 {
            let got = xi.retract_hosvd(alpha).to_dense();
            let mut want = xd.clone();
            dense_axpy(&mut want, alpha, &xiv);
            dense_nrm(&dense_sub(&got, &want))
        };
        let (a, h) = (1e-2f64, 5e-3f64);
        let (ra, rh) = (resid(a), resid(h));
        assert!(
            ra > 1e-9,
            "anti-vacuity: the truncated retraction must actually curve (residual {ra:e}) — \
             otherwise the ratio below tests nothing"
        );
        let ratio = ra / rh.max(1e-300);
        assert!(
            (2.5..=6.0).contains(&ratio),
            "retraction residual must be O(α²): r(α)/r(α/2) = {ratio} (r = {ra:e}, {rh:e})"
        );
    }

    #[test]
    fn tf_c4_projective_transport_dense_value_oracle() {
        let space = QnSpace::new(4, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC4_03);
        let mut x: TtVector<f64> = randomized_state(space, 2, 2, &mut rng);
        x.scale_mut(1.0 / x.norm());
        let mut y: TtVector<f64> = randomized_state(space, 2, 2, &mut rng);
        y.scale_mut(1.0 / y.norm());
        let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let (xl, xr) = frames(&x);
        let xi = TtTangent::project(&xl, &xr, &dir);

        let got = projective_transport(&y, &xi).to_vector().to_dense();

        let (yl, yr) = frames(&y);
        let basis = tangent_basis(&yl, &yr, false);
        assert!(!basis.is_empty(), "tangent basis at y must be nonempty");
        let z = xi.to_vector().to_dense();
        let mut want = vec![0.0f64; z.len()];
        for b in &basis {
            let bd = b.to_vector().to_dense();
            dense_axpy(&mut want, dense_dot(&bd, &z), &bd);
        }
        let scale = dense_nrm(&want);
        assert!(
            scale > 1e-3 * dense_nrm(&z),
            "anti-vacuity: the projection must retain O(1) mass, got {scale:e}"
        );
        let err = dense_nrm(&dense_sub(&got, &want)) / scale;
        assert!(
            err <= 1e-9,
            "projective_transport must equal the dense basis projection value-for-value: \
             rel err {err:e}"
        );
    }
}

#[cfg(test)]
mod tangent_shifted_tests;

#[cfg(test)]
mod gauge_fix_tests {
    use super::*;
    use crate::solvers::riemannian::{normalize_mut, tangent_bases};
    use crate::test_util::{c2v_pg, randomized_state};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn pin(qs: &[TtTangent<f64>], reembed: impl Fn(&TtVector<f64>) -> TtTangent<f64>, what: &str) {
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
            "{what}: the gauge fix is not orthogonal ({og:e})"
        );
        assert!(
            op >= 1e-4,
            "{what}: project∘to_vector came out orthogonal ({op:e}) — vacuous"
        );
    }

    #[test]
    fn gauge_fix_orthogonal_at_identity_and_on_a_shifted_space() {
        let pg = c2v_pg(6);
        let space = QnSpace::with_point_group(6, 2, 2, pg, Irrep::IDENTITY);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6F1A);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, crate::round::RoundMode::Rel);
        normalize_mut(&mut x);
        let (bl, br) = tangent_bases(&x);
        let qs: Vec<TtTangent<f64>> = (0..4)
            .map(|_| {
                let d: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
                TtTangent::project(&bl, &br, &d)
            })
            .collect();
        pin(&qs, |v| TtTangent::project(&bl, &br, v), "flux IDENTITY");
        let phi = (1..pg.order())
            .map(Irrep::new)
            .find(|&phi| tangent_dim_grids(&space, &bl.ranks, phi) > 0)
            .expect("fixture premise: a live shifted channel");
        let space_out = space.with_target(space.target.fuse(phi));
        let qs: Vec<TtTangent<f64>> = (0..4)
            .map(|_| {
                let d: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
                TtTangent::project_shifted(&bl, &br, phi, &d)
            })
            .collect();
        pin(
            &qs,
            |v| TtTangent::project_shifted(&bl, &br, phi, v),
            &format!("flux {phi:?}"),
        );
    }
}
