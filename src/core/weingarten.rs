// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Weingarten curvature term `B·ξ = P_X (D_ξ P_X) Z` on the fixed-TT-rank manifold, the
//! second summand of the Riemannian Hessian `[PB Eq. 8]` (Psenka–Boumal, arXiv:2011.13395),
//! transcribed for the abelian `f64`/`C64` layer.

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, Par};

use crate::core::SparseCore;
use crate::core::factor::{
    Sketch, adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
    svd_uvt_svals,
};
use crate::core::tangent::{TtTangent, shifted_frames};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::matvec::OneSiteWorkspace;
use crate::pointgroup::Irrep;
use crate::qn::{RankGrid, Sector};
use crate::scalar::Scalar;
use crate::vector::TtVector;
use std::sync::Arc;

/// Σ⁻¹ floor policy for the bond-gauge inverse `R_k⁻¹` (the primary curvature hazard).
#[derive(Clone, Copy, Debug)]
pub enum FloorPolicy {
    /// Truncated pseudo-inverse: drop singular values `σ < floor` (invert-to-zero), report the
    /// dropped Schmidt mass. `floor` is absolute.
    Hard(f64),
    /// Tikhonov damping `σ ↦ σ/(σ² + ε²)`; never divides by ~0, keeps all directions.
    Tikhonov(f64),
}

impl FloorPolicy {
    /// The default, from the oracle sensitivity sweep: a hard floor at `1e-12`
    /// (well below any Schmidt value carrying curvature, above the round-off of a truncated bond).
    pub fn default_hard() -> Self {
        FloorPolicy::Hard(1e-12)
    }
}

/// Per-bond bond gauge `R_k` (`X_{≥k+1} = X̃_{≥k+1} R_k`, `[PB App. B.2]`) with its floored inverse
/// and Schmidt values. Exposed for the curvature apply (the `R⁻¹` term of `[PB Thm 2]`) and the
/// Williamson path (rotations available via [`Self::svals`]). Everything is per-sector
/// (block-diagonal over bond sectors).
#[derive(Clone)]
pub struct BondGauge<T: Scalar> {
    /// Interior bond index `1..d`.
    pub k: usize,
    /// `R_k` itself (the overlap sketch), `r_k×r_k` per sector.
    pub r: Sketch<T>,
    /// Floored `R_k⁻¹`, `r_k×r_k` per sector.
    pub r_inv: Sketch<T>,
    /// Per-sector Schmidt values (singular values of `R_k`), descending.
    pub sigma: Vec<(Sector, Vec<f64>)>,
    /// Dropped Schmidt mass `Σ σ²` over floored directions (never silent).
    pub dropped_mass: f64,
}

impl<T: Scalar> BondGauge<T> {
    /// Per-sector Schmidt values.
    pub fn svals(&self) -> &[(Sector, Vec<f64>)] {
        &self.sigma
    }
}

fn core_add_assign<T: Scalar>(out: &mut SparseCore<T>, a: T, other: &SparseCore<T>) {
    out.axpy(a, other);
}

pub(crate) type ZEnvs<'a, T> = (&'a [Sketch<T>], &'a [Sketch<T>]);

pub(crate) trait ZSource<T: Scalar> {
    fn left_mul(&self, s: &Sketch<T>, k: usize) -> SparseCore<T>;
    fn right_mul(&self, k: usize, s: &Sketch<T>) -> SparseCore<T>;
    fn bond_ranks(&self, b: usize) -> RankGrid;

    fn cached_envs(&self) -> Option<ZEnvs<'_, T>> {
        None
    }
}

pub(crate) struct ExplicitZ<'a, T: Scalar> {
    z: &'a TtVector<T>,
    envs: Option<ZEnvs<'a, T>>,
}

impl<'a, T: Scalar> ExplicitZ<'a, T> {
    pub(crate) fn new(z: &'a TtVector<T>) -> Self {
        ExplicitZ { z, envs: None }
    }

    pub(crate) fn with_envs(mut self, pz: &'a [Sketch<T>], qz: &'a [Sketch<T>]) -> Self {
        self.envs = Some((pz, qz));
        self
    }
}

impl<T: Scalar> ZSource<T> for ExplicitZ<'_, T> {
    fn left_mul(&self, s: &Sketch<T>, k: usize) -> SparseCore<T> {
        mul_sketch_left(s, self.z.core(k))
    }
    fn right_mul(&self, k: usize, s: &Sketch<T>) -> SparseCore<T> {
        mul_sketch_right(self.z.core(k), s)
    }
    fn bond_ranks(&self, b: usize) -> RankGrid {
        self.z.ranks[b].clone()
    }
    fn cached_envs(&self) -> Option<ZEnvs<'_, T>> {
        self.envs
    }
}

/// The tangent gauge projector `gauge_k(C) = (I − U^L_k U^L_kᴴ)·C` on the vertical unfoldings
/// (`[PB App. E.2]`): per column sector of bond `comp.k + 1`, `M = C − L·(Lᴴ·C)` — the single
/// gauge condition of the tangent parameterization. `l_core` is `base_l.core(k)` (left-orthonormal
/// for `k ≤ d−2`). Mutates `comp` in place.
pub fn gauge_complement_mut<T: Scalar>(comp: &mut SparseCore<T>, l_core: &SparseCore<T>) {
    let space = comp.space;
    let k = comp.k;
    for s in space.bond_sectors(k + 1) {
        let l = l_core.vertical_unfolding(s);
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

pub(crate) struct FusedTermA<'a, T: Scalar> {
    pub(crate) h: &'a SparseHamiltonian<T>,
    pub(crate) u: &'a TtVector<T>,
    pub(crate) v: &'a TtVector<T>,
    pub(crate) l: &'a [Sketch<T>],
    pub(crate) r: &'a [Sketch<T>],
    pub(crate) ws_u: &'a [OneSiteWorkspace],
    pub(crate) ws_v: &'a [OneSiteWorkspace],
    pub(crate) ws_du: &'a [OneSiteWorkspace],
}

impl<T: Scalar> FusedTermA<'_, T> {
    pub(crate) fn components(&self, xi: &[SparseCore<T>]) -> Vec<SparseCore<T>> {
        let (h, u, v) = (self.h, self.u, self.v);
        let d = u.space.d;

        let mut lt: Vec<Option<Sketch<T>>> = Vec::with_capacity(d);
        lt.push(None);
        for k in 0..d.saturating_sub(1) {
            let transported = lt[k].as_ref().map(|prev| {
                h.sketching_step_right_mixed(u.core(k), v.core(k), prev, &self.ws_v[k])
            });
            let inserted =
                h.sketching_step_right_mixed(u.core(k), &xi[k], &self.l[k], &self.ws_du[k]);
            lt.push(Some(match transported {
                Some(mut t) => {
                    t.add_assign_sketch(&inserted);
                    t
                }
                None => inserted,
            }));
        }

        let mut rt: Vec<Option<Sketch<T>>> = vec![None; d];
        for k in (0..d.saturating_sub(1)).rev() {
            let transported = rt[k + 1].as_ref().map(|prev| {
                h.sketching_step_left_mixed(v.core(k + 1), u.core(k + 1), prev, &self.ws_u[k + 1])
            });
            let inserted = h.sketching_step_left_mixed(
                v.core(k + 1),
                &xi[k + 1],
                &self.r[k + 1],
                &self.ws_du[k + 1],
            );
            rt[k] = Some(match transported {
                Some(mut t) => {
                    t.add_assign_sketch(&inserted);
                    t
                }
                None => inserted,
            });
        }

        let mut components = Vec::with_capacity(d);
        for k in 0..d {
            let mut comp = h.sketched_hamiltonian(&xi[k], &self.l[k], &self.r[k], &self.ws_du[k]);
            if let Some(ltk) = &lt[k] {
                let t = h.sketched_hamiltonian(v.core(k), ltk, &self.r[k], &self.ws_v[k]);
                comp.axpy(T::one(), &t);
            }
            if let Some(rtk) = &rt[k] {
                let t = h.sketched_hamiltonian(u.core(k), &self.l[k], rtk, &self.ws_u[k]);
                comp.axpy(T::one(), &t);
            }
            if k + 1 < d {
                gauge_complement_mut(&mut comp, u.core(k));
            }
            components.push(comp);
        }
        components
    }
}

/// Bond gauges `{R_k}` for the interior bonds `1..d`. Folds the overlap sketch
/// `G_k = base_l(0..k−1)ᴴ · base_r(0..k−1)` from the left (the identical fold `project` uses), SVDs
/// each per-sector block to get `(W_k, Σ_k, V_kᴴ)` and forms the floored `R_k⁻¹ = V_k Σ_k⁻¹ W_kᴴ`.
pub fn bond_gauges<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    floor: FloorPolicy,
) -> Vec<BondGauge<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut out = Vec::new();
    let mut pg = Sketch::<T>::identity(space, 0, 1);
    for b in 1..d {
        let br_folded = mul_sketch_left(&pg, base_r.core(b - 1));
        pg = adjoint_core_times_core(base_l.core(b - 1), &br_folded);
        let (r_inv, sigma, dropped) = invert_bond(&pg, floor);
        out.push(BondGauge {
            k: b,
            r: pg.clone(),
            r_inv,
            sigma,
            dropped_mass: dropped,
        });
    }
    out
}

type BondInverse<T> = (Sketch<T>, Vec<(Sector, Vec<f64>)>, f64);

fn invert_bond<T: Scalar>(r: &Sketch<T>, floor: FloorPolicy) -> BondInverse<T> {
    let space = r.space;
    let k = r.k;
    let ranks = r.row_ranks();
    let mut r_inv = Sketch::<T>::zeros(space, k, &ranks, &ranks);
    let mut sigma_all: Vec<(Sector, Vec<f64>)> = Vec::new();
    let mut dropped = 0.0f64;
    for s in r.sectors() {
        let Some(blk) = r.block(s) else { continue };
        let n = blk.nrows();
        if n == 0 {
            sigma_all.push((s, Vec::new()));
            continue;
        }
        let (w, svals, vt) = svd_uvt_svals(blk);
        sigma_all.push((s, svals.clone()));
        let mut sinv = vec![T::zero(); svals.len()];
        for (i, &sig) in svals.iter().enumerate() {
            let inv = match floor {
                FloorPolicy::Hard(f) => {
                    if sig < f {
                        dropped += sig * sig;
                        0.0
                    } else {
                        1.0 / sig
                    }
                }
                FloorPolicy::Tikhonov(eps) => sig / (sig * sig + eps * eps),
            };
            sinv[i] = T::from_f64(inv);
        }
        let size = svals.len();
        let mut tmp = Mat::<T>::zeros(size, n);
        for (i, &si) in sinv.iter().enumerate() {
            for j in 0..n {
                *tmp.get_mut(i, j) = si * w.get(j, i).conj_s();
            }
        }
        let mut rinv = Mat::<T>::zeros(n, n);
        if size > 0 {
            matmul(
                rinv.as_mut(),
                Accum::Replace,
                vt.as_ref().adjoint(),
                tmp.as_ref(),
                T::one(),
                Par::Seq,
            );
        }
        if let Some(mut dst) = r_inv.block_mut(s) {
            for j in 0..n {
                for i in 0..n {
                    *dst.rb_mut().get_mut(i, j) = *rinv.get(i, j);
                }
            }
        }
    }
    (r_inv, sigma_all, dropped)
}

pub(crate) fn left_env_z<T: Scalar>(base_l: &TtVector<T>, z: &dyn ZSource<T>) -> Vec<Sketch<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut p: Vec<Sketch<T>> = Vec::with_capacity(d + 1);
    p.push(Sketch::identity(space, 0, 1));
    for k in 0..d {
        let pd = z.left_mul(&p[k], k);
        p.push(adjoint_core_times_core(base_l.core(k), &pd));
    }
    p
}

pub(crate) fn right_env_z<T: Scalar>(base_r: &TtVector<T>, z: &dyn ZSource<T>) -> Vec<Sketch<T>> {
    right_env_z_visit(base_r, z, |_, _| {})
}

pub(crate) fn right_env_z_visit<T: Scalar, F>(
    base_r: &TtVector<T>,
    z: &dyn ZSource<T>,
    mut visit: F,
) -> Vec<Sketch<T>>
where
    F: FnMut(usize, &SparseCore<T>),
{
    let space = base_r.space;
    let d = space.d;
    let mut q: Vec<Option<Sketch<T>>> = (0..=d).map(|_| None).collect();
    q[d] = Some(Sketch::identity(space, d, 1));
    for k in (0..d).rev() {
        let dq = z.right_mul(k, q[k + 1].as_ref().unwrap());
        visit(k, &dq);
        q[k] = Some(core_times_adjoint_core(&dq, base_r.core(k)));
    }
    q.into_iter().map(|x| x.unwrap()).collect()
}

fn var_left_env<T: Scalar>(
    base_l: &TtVector<T>,
    z: &dyn ZSource<T>,
    comps_scaled: &[SparseCore<T>],
    pz: &[Sketch<T>],
) -> Vec<Sketch<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut evl: Vec<Sketch<T>> = Vec::with_capacity(d);
    evl.push(Sketch::zeros(space, 0, &base_l.ranks[0], &z.bond_ranks(0)));
    for k in 0..d - 1 {
        let t1a = z.left_mul(&evl[k], k);
        let mut next = adjoint_core_times_core(base_l.core(k), &t1a);
        let t2a = z.left_mul(&pz[k], k);
        let t2 = adjoint_core_times_core(&comps_scaled[k], &t2a);
        next.add_assign_sketch(&t2);
        evl.push(next);
    }
    evl
}

fn var_right_env<T: Scalar>(
    frame: &TtVector<T>,
    z: &dyn ZSource<T>,
    comps_r: &[SparseCore<T>],
    qz: &[Sketch<T>],
) -> Vec<Sketch<T>> {
    let space = frame.space;
    let d = space.d;
    let mut evr: Vec<Option<Sketch<T>>> = (0..=d).map(|_| None).collect();
    evr[0] = Some(Sketch::zeros(space, 0, &z.bond_ranks(0), &frame.ranks[0]));
    evr[d] = Some(Sketch::zeros(space, d, &z.bond_ranks(d), &frame.ranks[d]));
    for b in (1..d).rev() {
        let t1a = z.right_mul(b, evr[b + 1].as_ref().unwrap());
        let mut cur = core_times_adjoint_core(&t1a, frame.core(b));
        let t2a = z.right_mul(b, &qz[b + 1]);
        let t2 = core_times_adjoint_core(&t2a, &comps_r[b - 1]);
        cur.add_assign_sketch(&t2);
        evr[b] = Some(cur);
    }
    evr.into_iter().map(|x| x.unwrap()).collect()
}

fn adjoint_blocks<T: Scalar>(s: &Sketch<T>) -> Sketch<T> {
    let mut out = s.clone();
    for sec in s.sectors() {
        if let (Some(src), Some(mut dst)) = (s.block(sec), out.block_mut(sec)) {
            dst.copy_from(src.adjoint());
        }
    }
    out
}

type RightChainInputs<T> = (Vec<SparseCore<T>>, Vec<Sketch<T>>);

/// Selector for which summands to assemble — `all()` is the plain apply; a split is an ablation aid
/// and the shape of the `CurvatureReg::Tikhonov` apply (`[S1 + S2]` on `ξ`, `S3` on `D·ξ`). `s1` =
/// left frame derivative, `s2` = gauge-projector derivative (identically 0 for a normal `Z`, kept
/// as `[PB Thm 2]` insurance), `s3` = right frame derivative (telescoped: its `R⁻ᴴ` on the output's
/// columns, module docs). The assembly skips the chains a selection does not read — the right
/// chain and `right_project_components` without `s3`, the left chain without `s1`. The
/// `[PB Thm 3]` cross terms have no selector: they vanish identically for normal `Z` (module docs,
/// point 1).
#[derive(Clone, Copy, Debug)]
pub struct Terms {
    pub diag_s1: bool,
    pub diag_s2: bool,
    pub diag_s3: bool,
}

impl Terms {
    pub fn all() -> Self {
        Terms {
            diag_s1: true,
            diag_s2: true,
            diag_s3: true,
        }
    }

    pub(crate) const S1_S2: Terms = Terms {
        diag_s1: true,
        diag_s2: true,
        diag_s3: false,
    };

    pub(crate) const S3: Terms = Terms {
        diag_s1: false,
        diag_s2: false,
        diag_s3: true,
    };
}

/// The Weingarten apply for a **general** ambient `z`: normal-projects `z ↦ z − ι(P(z))` and
/// delegates to [`weingarten_assemble`]. Mathematically identical for every `z` — only the normal
/// part enters term B (`P(D_ξP)P = 0`, module docs) — and it keeps the normal-`Z` precondition of
/// the raw assembly self-enforcing. Prefer this entry point whenever `z` is not *already* the
/// output of [`crate::solvers::hess::normal_residual`] (whose result is normal by the same
/// construction).
pub fn weingarten_assemble_general<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    z: &TtVector<T>,
    xi: &TtTangent<T>,
    floor: FloorPolicy,
    terms: Terms,
) -> TtTangent<T> {
    let tang = TtTangent::project(base_l, base_r, z).to_vector();
    let zn = z - &tang;
    weingarten_assemble(base_l, base_r, &zn, xi, floor, terms)
}

/// Assemble `B·ξ` as a [`TtTangent`] at the gauge frames `base_l`/`base_r`, contracting the
/// **normal** residual `z = r⊥`. PRECONDITION: `P(z) = 0` — the diagonal-only assembly is wrong
/// (not merely suboptimal) for a `z` with a tangential component, because the terms that a
/// tangential part would activate (the `C_k`-carrying frame derivatives and the `[PB Thm 3]`
/// cross sum) vanish identically only under that hypothesis (module docs). Guarded by a
/// dev-build assert; for arbitrary ambient `z` use [`weingarten_assemble_general`]. `terms`
/// selects summands (ablation); ship with [`Terms::all`].
pub fn weingarten_assemble<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    z: &TtVector<T>,
    xi: &TtTangent<T>,
    floor: FloorPolicy,
    terms: Terms,
) -> TtTangent<T> {
    let gauges = bond_gauges(base_l, base_r, floor);
    weingarten_assemble_cached(base_l, base_r, z, xi, &gauges, terms)
}

/// [`weingarten_assemble`] with the bond gauges supplied by the caller — `gauges[b−1]` is bond `b`
/// from [`bond_gauges`] at the SAME `(base_l, base_r)` frames (the floor is baked into the cached
/// `R⁻¹`). The gauges depend only on the base point, not on `ξ` or `z`, so an iterative solver
/// making many `B·ξ` applies at a fixed base computes them once instead of one per-bond SVD sweep
/// per apply. The gauges are `φ`-INDEPENDENT (they are `x`'s own bond Grams), so ONE set serves
/// every channel.
pub fn weingarten_assemble_cached<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    z: &TtVector<T>,
    xi: &TtTangent<T>,
    gauges: &[BondGauge<T>],
    terms: Terms,
) -> TtTangent<T> {
    #[cfg(debug_assertions)]
    {
        let zn = z.norm_lr(false);
        if zn > 0.0 {
            let tang = TtTangent::project(base_l, base_r, z).norm();
            assert!(
                tang <= 1e-6 * zn,
                "weingarten_assemble: Z is not normal (‖P(Z)‖/‖Z‖ = {:.3e}) — \
                 use weingarten_assemble_general for a general ambient Z",
                tang / zn
            );
        }
    }
    let (bl, br) = (Arc::new(base_l.clone()), Arc::new(base_r.clone()));
    weingarten_assemble_sources(&bl, &br, &[&ExplicitZ::new(z)], xi, gauges, terms)
}

pub(crate) fn weingarten_assemble_sources<T: Scalar>(
    base_l: &Arc<TtVector<T>>,
    base_r: &Arc<TtVector<T>>,
    sources: &[&dyn ZSource<T>],
    xi: &TtTangent<T>,
    gauges: &[BondGauge<T>],
    terms: Terms,
) -> TtTangent<T> {
    let space = base_l.space;
    let d = space.d;
    let r = base_l.ranks.clone();
    let phi = xi.phi();
    let shifted = phi != Irrep::IDENTITY;
    let space_out = if shifted {
        space.with_target(space.target.fuse(phi))
    } else {
        space
    };
    debug_assert_eq!(
        base_l.space, base_r.space,
        "weingarten_assemble_sources: the two frames must be stamped the same space"
    );
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "weingarten_assemble_sources: the two frames must factorize the SAME state"
    );
    debug_assert_eq!(
        xi.rank(),
        r.as_slice(),
        "weingarten_assemble_sources: xi was built at different frames"
    );
    debug_assert_eq!(
        xi.base_l.space, space_out,
        "weingarten_assemble_sources: xi's frames are stamped {:?} but the flux {phi:?} on {space:?} \
         gives {space_out:?}",
        xi.base_l.space
    );

    let r_col: Option<Vec<RankGrid>> =
        shifted.then(|| r.iter().map(|g| g.relabel_irrep(phi)).collect());
    let rc = |b: usize| -> &RankGrid { r_col.as_ref().map_or(&r[b], |v| &v[b]) };

    if d == 1 {
        let comp = SparseCore::zeros(space_out, 0, &r[0], rc(1));
        return if shifted {
            let (u, v) = shifted_frames(base_l, base_r, phi);
            TtTangent::from_parts_shifted(r, r_col.expect("shifted"), phi, u, v, vec![comp])
        } else {
            TtTangent::from_parts(r, base_l.clone(), base_r.clone(), vec![comp])
        };
    }

    debug_assert_eq!(
        gauges.len(),
        d - 1,
        "weingarten_assemble_cached: gauge count"
    );

    let shifted_pair = shifted.then(|| shifted_frames(base_l, base_r, phi));
    let (fl, fr) = shifted_pair
        .as_ref()
        .map_or((&**base_l, &**base_r), |(u, v)| (u, v));

    if shifted {
        let q = Sketch::<T>::identity(space_out, d, 1);
        let empty_channel = q
            .blocks
            .iter()
            .all(|(_, b)| b.as_ref().is_none_or(|m| m.ncols() == 0));
        if empty_channel {
            let comps = (0..d)
                .map(|k| SparseCore::zeros(space_out, k, &r[k], rc(k + 1)))
                .collect();
            return TtTangent::from_parts_shifted(
                r,
                r_col.expect("shifted"),
                phi,
                fl.clone(),
                fr.clone(),
                comps,
            );
        }
    }

    let rinv_rel: Option<Vec<Sketch<T>>> =
        shifted.then(|| gauges.iter().map(|g| g.r_inv.relabel_irrep(phi)).collect());
    let rinv = |b: usize| -> &Sketch<T> {
        rinv_rel
            .as_ref()
            .map_or(&gauges[b - 1].r_inv, |v| &v[b - 1])
    };

    let left_terms = terms.diag_s1 || terms.diag_s2;
    let comps_scaled: Vec<SparseCore<T>> = if left_terms {
        (0..d - 1)
            .map(|k| mul_sketch_right(xi.component(k), rinv(k + 1)))
            .collect()
    } else {
        Vec::new()
    };
    let right_chain: Option<RightChainInputs<T>> = terms.diag_s3.then(|| {
        let comps_r: Vec<SparseCore<T>> = right_project_components(fl, fr, &xi.to_vector())
            .into_iter()
            .skip(1)
            .map(|c| {
                if shifted {
                    c.relabel_irrep_into(phi, space_out)
                } else {
                    c
                }
            })
            .collect();
        let rinv_h: Vec<Sketch<T>> = (1..d).map(|b| adjoint_blocks(rinv(b))).collect();
        (comps_r, rinv_h)
    });

    let u_phi: Option<TtVector<T>> = (shifted && (terms.diag_s1 || terms.diag_s3)).then(|| {
        let mut u = TtVector::new(
            space_out,
            (0..d)
                .map(|k| base_l.core(k).clone().relabel_irrep_into(phi, space_out))
                .collect(),
        );
        u.ortho = base_l.ortho;
        u
    });
    let frame_l = u_phi.as_ref().unwrap_or(base_l);

    let mut out_cores: Vec<SparseCore<T>> = (0..d)
        .map(|k| SparseCore::<T>::zeros(space_out, k, &r[k], rc(k + 1)))
        .collect();

    for &z in sources {
        let folded;
        let (pz, qz): (&[Sketch<T>], &[Sketch<T>]) = match z.cached_envs() {
            Some(pq) => pq,
            None => {
                folded = (left_env_z(base_l, z), right_env_z(base_r, z));
                (&folded.0, &folded.1)
            }
        };
        let evl = if terms.diag_s1 {
            var_left_env(frame_l, z, &comps_scaled, pz)
        } else {
            Vec::new()
        };
        let evr = match &right_chain {
            Some((comps_r, rinv_h)) => {
                let mut evr = var_right_env(frame_l, z, comps_r, qz);
                for (e, rh) in evr[1..d].iter_mut().zip(rinv_h) {
                    debug_assert!(
                        e.sectors().into_iter().all(|s| {
                            e.block(s).is_none_or(|b| b.nrows() * b.ncols() == 0)
                                || rh.block(s).is_some()
                        }),
                        "weingarten_assemble_sources: a live block of eVR'[{}] has no R⁻ᴴ block",
                        e.site()
                    );
                    e.rmul_sketch_mut(rh);
                }
                evr
            }
            None => Vec::new(),
        };

        for (k, middle) in out_cores.iter_mut().enumerate() {
            if k == d - 1 {
                if terms.diag_s1 {
                    let diag = z.left_mul(&evl[k], k);
                    let diag = if shifted {
                        diag.relabel_irrep_into(phi, space_out)
                    } else {
                        diag
                    };
                    core_add_assign(middle, T::one(), &diag);
                }
            } else {
                if left_terms {
                    let zq = z.right_mul(k, &qz[k + 1]);
                    if terms.diag_s1 {
                        let s1 = mul_sketch_left(&evl[k], &zq);
                        let s1 = if shifted {
                            s1.relabel_irrep_into(phi, space_out)
                        } else {
                            s1
                        };
                        core_add_assign(middle, T::one(), &s1);
                    }
                    if terms.diag_s2 {
                        let g20 = mul_sketch_left(&pz[k], &zq);
                        let w2 = adjoint_core_times_core(base_l.core(k), &g20);
                        let w2 = if shifted { w2.relabel_irrep(phi) } else { w2 };
                        let s2 = mul_sketch_right(&comps_scaled[k], &w2);
                        core_add_assign(middle, T::from_f64(-1.0), &s2);
                    }
                }
                if terms.diag_s3 {
                    let s3 = mul_sketch_left(&pz[k], &z.right_mul(k, &evr[k + 1]));
                    debug_assert!(
                        s3.space.same_lattice(&space_out),
                        "S3 is stamped {:?}, not a lattice-mate of {space_out:?}",
                        s3.space
                    );
                    core_add_assign(middle, T::one(), &s3);
                }
            }
        }
    }

    for (k, core) in out_cores.iter_mut().enumerate().take(d - 1) {
        gauge_complement_mut(core, base_l.core(k));
    }

    if shifted {
        TtTangent::from_parts_shifted(
            r,
            r_col.expect("shifted"),
            phi,
            fl.clone(),
            fr.clone(),
            out_cores,
        )
    } else {
        TtTangent::from_parts(r, base_l.clone(), base_r.clone(), out_cores)
    }
}

fn right_gauge_complement_mut<T: Scalar>(comp: &mut SparseCore<T>, m_core: &SparseCore<T>) {
    let space = comp.space;
    let k = comp.k;
    for l in space.bond_sectors(k) {
        let m = m_core.horizontal_unfolding(l);
        let mut c = comp.horizontal_unfolding(l);
        if m.nrows() > 0 && c.nrows() > 0 && m.ncols() > 0 {
            let mut cmh = Mat::<T>::zeros(c.nrows(), m.nrows());
            matmul(
                cmh.as_mut(),
                Accum::Replace,
                c.as_ref(),
                m.as_ref().adjoint(),
                T::one(),
                Par::Seq,
            );
            matmul(
                c.as_mut(),
                Accum::Add,
                cmh.as_ref(),
                m.as_ref(),
                T::from_f64(-1.0),
                Par::Seq,
            );
        }
        comp.scatter_horizontal(l, c.as_ref());
    }
}

fn right_project_components<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    dir: &TtVector<T>,
) -> Vec<SparseCore<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut p: Vec<Sketch<T>> = Vec::with_capacity(d);
    p.push(Sketch::identity(space, 0, 1));
    for k in 0..d.saturating_sub(1) {
        let pd = mul_sketch_left(&p[k], dir.core(k));
        p.push(adjoint_core_times_core(base_l.core(k), &pd));
    }
    let mut q = Sketch::identity(space, d, 1);
    let mut comps: Vec<Option<SparseCore<T>>> = (0..d).map(|_| None).collect();
    for k in (0..d).rev() {
        let dq = mul_sketch_right(dir.core(k), &q);
        let mut comp = mul_sketch_left(&p[k], &dq);
        if k >= 1 {
            right_gauge_complement_mut(&mut comp, base_r.core(k));
            q = core_times_adjoint_core(&dq, base_r.core(k));
        }
        comps[k] = Some(comp);
    }
    comps.into_iter().map(|c| c.unwrap()).collect()
}

mod symplectic;
pub use symplectic::{
    ReducedWilliamsonBasis, RpaMetricBlock, SignedWilliamson, SignedWilliamsonBasis,
    WilliamsonError, dense_svd_flat, extract_columns, rpa_generator_eigenvalues, skew_canonical,
    symplectic_generator_full, weingarten_fraction, weingarten_quadratic_forms, williamson_full,
    williamson_full_vectors, williamson_reduced, williamson_reduced_signed,
    williamson_reduced_signed_vectors, williamson_reduced_vectors,
};

#[cfg(test)]
mod t8r_tests {
    use super::{
        RpaMetricBlock, rpa_generator_eigenvalues, weingarten_fraction, williamson_reduced_signed,
        williamson_reduced_signed_vectors,
    };
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    const PSD_TOL: f64 = 1e-10;

    fn orthogonal_rows(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            for _ in 0..2 {
                for r in rows.iter() {
                    let d: f64 = r.iter().zip(v.iter()).map(|(p, q)| p * q).sum();
                    for (vi, ri) in v.iter_mut().zip(r.iter()) {
                        *vi -= d * ri;
                    }
                }
            }
            let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(nrm > 1e-6, "degenerate draw at seed {seed}");
            for vi in v.iter_mut() {
                *vi /= nrm;
            }
            rows.push(v);
        }
        rows
    }

    fn sym_from_spectrum(evals: &[f64], q: &[Vec<f64>], n: usize) -> Vec<f64> {
        let mut m = vec![0.0f64; n * n];
        for (c, qc) in q.iter().enumerate() {
            for i in 0..n {
                let li = evals[c] * qc[i];
                for j in 0..n {
                    m[i * n + j] += li * qc[j];
                }
            }
        }
        m
    }

    fn grams_from_blocks(hrr: &[f64], hii: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let a = hrr.iter().zip(hii).map(|(p, q)| 0.5 * (p + q)).collect();
        let b = hrr.iter().zip(hii).map(|(p, q)| 0.5 * (p - q)).collect();
        (a, b)
    }

    fn quad_col(m: &[f64], modes: &[f64], n: usize, k: usize, c: usize) -> f64 {
        let v: Vec<f64> = (0..n).map(|i| modes[i * k + c]).collect();
        (0..n)
            .map(|i| v[i] * (0..n).map(|j| m[i * n + j] * v[j]).sum::<f64>())
            .sum()
    }

    #[test]
    fn t8r_signed_omega2_matches_generator() {
        let n = 6;
        let q = orthogonal_rows(n, 0xB0_5EED);
        for (tag, rr, ii) in [
            (
                "Resonant{Rr}",
                vec![0.5, 1.0, 1.6, 2.3, 3.0, 4.1],
                vec![-0.28, 0.8, 1.2, 1.9, 2.5, 3.4],
            ),
            (
                "Resonant{Ii}",
                vec![-0.44, 1.0, 1.6, 2.3, 3.0, 4.1],
                vec![0.35, 0.8, 1.2, 1.9, 2.5, 3.4],
            ),
        ] {
            let hrr = sym_from_spectrum(&rr, &q, n);
            let hii = sym_from_spectrum(&ii, &q, n);
            let (a, b) = grams_from_blocks(&hrr, &hii);
            let signed = williamson_reduced_signed(&hrr, &hii, n, PSD_TOL)
                .unwrap_or_else(|e| panic!("{tag}: signed path refused: {e:?}"));

            let mut from_gen: Vec<f64> = rpa_generator_eigenvalues(&a, &b, n)
                .iter()
                .map(|z| {
                    let sq_re = z.re * z.re - z.im * z.im;
                    let sq_im = 2.0 * z.re * z.im;
                    assert!(
                        sq_im.abs() <= 1e-8 * (1.0 + sq_re.abs()),
                        "{tag}: eig(𝓜)² must be real when only one block is indefinite; got \
                         {sq_re} + {sq_im}i"
                    );
                    sq_re
                })
                .collect();
            from_gen.sort_by(|p, r| p.partial_cmp(r).unwrap());

            let mut want: Vec<f64> = signed.omega2.iter().flat_map(|&v| [v, v]).collect();
            want.sort_by(|p, r| p.partial_cmp(r).unwrap());

            let scale = from_gen.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let dev = from_gen
                .iter()
                .zip(&want)
                .map(|(p, r)| (p - r).abs())
                .fold(0.0f64, f64::max);
            assert!(
                dev <= 1e-9 * scale,
                "{tag}: signed ω² vs eig(𝓜)²: max|Δ| = {dev:.3e} (scale {scale:.3})"
            );
            assert!(
                signed.omega2[0] < -1e-6,
                "{tag}: fixture is not actually resonant (ω²[0] = {})",
                signed.omega2[0]
            );
        }
    }

    #[test]
    fn t8r_modes_carry_the_sign() {
        let n = 6;
        let q = orthogonal_rows(n, 0xC0_FFEE);
        let rr = vec![0.5, 1.0, 1.6, 2.3, 3.0, 4.1];
        let ii = vec![-0.28, 0.8, 1.2, 1.9, 2.5, 3.4];
        let hrr = sym_from_spectrum(&rr, &q, n);
        let hii = sym_from_spectrum(&ii, &q, n);
        let signed = williamson_reduced_signed(&hrr, &hii, n, PSD_TOL).expect("Resonant{Rr}");
        assert_eq!(signed.metric, RpaMetricBlock::Rr);

        for (j, &w2) in signed.omega2.iter().enumerate() {
            for (name, modes) in [("modes_r", &signed.modes_r), ("modes_i", &signed.modes_i)] {
                let nrm2: f64 = (0..n).map(|i| modes[i * n + j].powi(2)).sum();
                assert!(
                    (nrm2 - 1.0).abs() < 1e-12,
                    "{name}[{j}] must be unit norm, got ‖·‖² = {nrm2}"
                );
            }
            let r_ii = quad_col(&hii, &signed.modes_i, n, n, j);
            assert!(
                r_ii.signum() == w2.signum(),
                "modes_i[{j}]ᵀ H^ii modes_i[{j}] = {r_ii:.6e} must share the sign of ω² = {w2:.6e}"
            );
            let r_rr = quad_col(&hrr, &signed.modes_r, n, n, j);
            assert!(
                r_rr > 0.0,
                "modes_r[{j}]ᵀ H^rr modes_r[{j}] = {r_rr:.6e} must be positive (H^rr ≻ 0)"
            );
            assert!(
                (r_rr * r_ii).signum() == w2.signum(),
                "the paired quotients must reproduce sign(ω²) at mode {j}"
            );
        }
    }

    #[test]
    fn t8r_weingarten_fraction_limits() {
        let n = 5;
        let q = orthogonal_rows(n, 0xE2_7A11);
        let a = sym_from_spectrum(&[0.6, 1.1, 1.7, 2.4, 3.3], &q, n);

        let zero = vec![0.0f64; n * n];
        let modes: Vec<f64> = {
            let mut m = vec![0.0f64; n * n];
            for (c, qc) in q.iter().enumerate() {
                for (i, &v) in qc.iter().enumerate() {
                    m[i * n + c] = v;
                }
            }
            m
        };
        let f0 = weingarten_fraction(&a, &zero, &modes, n, n);
        assert!(
            f0.iter().all(|&f| f == 0.0),
            "B = 0 must give f_W = 0 exactly on every mode, got {f0:?}"
        );

        let f1 = weingarten_fraction(&a, &a, &modes, n, n);
        for (j, &f) in f1.iter().enumerate() {
            assert!(
                (f - 1.0).abs() < 1e-12,
                "B = A must give f_W = 1 on mode {j}, got {f}"
            );
        }

        let mut prev = 0.0;
        for s in [0.1, 0.3, 0.5, 0.8] {
            let bs: Vec<f64> = a.iter().map(|v| s * v).collect();
            let f = weingarten_fraction(&a, &bs, &modes, n, n);
            assert!(
                (f[0] - s).abs() < 1e-12,
                "f_W must be the B scale exactly when B = s·A: s = {s}, f_W = {}",
                f[0]
            );
            assert!(f[0] > prev, "f_W must increase with the curvature scale");
            prev = f[0];
        }
    }

    #[test]
    fn t8r_signed_basis_identities() {
        let n = 7;
        let q = orthogonal_rows(n, 0x11_ACED);
        for (tag, rr, ii, want_metric, want_neg) in [
            (
                "Resonant{Rr}, one imaginary mode",
                vec![0.45, 0.95, 1.4, 2.1, 2.7, 3.2, 4.3],
                vec![-0.30, 0.8, 1.15, 1.85, 2.35, 2.95, 3.6],
                RpaMetricBlock::Rr,
                1usize,
            ),
            (
                "Resonant{Ii}, two imaginary modes",
                vec![-0.5, -0.11, 1.4, 2.1, 2.7, 3.2, 4.3],
                vec![0.35, 0.8, 1.15, 1.85, 2.35, 2.95, 3.6],
                RpaMetricBlock::Ii,
                2usize,
            ),
        ] {
            let hrr = sym_from_spectrum(&rr, &q, n);
            let hii = sym_from_spectrum(&ii, &q, n);
            let b = williamson_reduced_signed_vectors(&hrr, &hii, n, PSD_TOL)
                .unwrap_or_else(|e| panic!("{tag}: refused {e:?}"));
            assert_eq!(b.metric, want_metric, "{tag}: metric");
            assert_eq!(b.n_neg, want_neg, "{tag}: n_neg");
            let n_pos = b.omega.len();
            assert_eq!(n_pos, n - want_neg, "{tag}: one basis column per REAL mode");

            for w in b.omega.windows(2) {
                assert!(w[0] <= w[1] + 1e-13, "{tag}: ω ascending");
            }
            for (j, &w) in b.omega.iter().enumerate() {
                let w2 = b.omega2[want_neg + j];
                assert!(
                    (w * w - w2).abs() <= 1e-10 * w2.abs().max(1.0),
                    "{tag}: ω[{j}]² = {} must be the signed spectrum's entry {w2}",
                    w * w
                );
            }

            let scale = b.omega.last().copied().unwrap_or(1.0).max(1.0);
            let check = |m: &[f64], s: &[f64], t: &[f64], want_diag: Option<&[f64]>, name: &str| {
                for a in 0..n_pos {
                    for c in 0..n_pos {
                        let mut acc = 0.0;
                        for i in 0..n {
                            let mut mi = 0.0;
                            for j in 0..n {
                                mi += m[i * n + j] * t[j * n_pos + c];
                            }
                            acc += s[i * n_pos + a] * mi;
                        }
                        let want = match want_diag {
                            Some(d) if a == c => d[a],
                            Some(_) => 0.0,
                            None if a == c => 1.0,
                            None => 0.0,
                        };
                        assert!(
                            (acc - want).abs() <= 1e-9 * scale,
                            "{tag}: {name}[{a},{c}] = {acc:.6e} vs {want:.6e}"
                        );
                    }
                }
            };
            let eye = sym_from_spectrum(&vec![1.0; n], &q, n);
            check(&hrr, &b.s_r, &b.s_r, Some(&b.omega), "S_rᵀ H^rr S_r");
            check(&hii, &b.s_i, &b.s_i, Some(&b.omega), "S_iᵀ H^ii S_i");
            check(&eye, &b.s_r, &b.s_i, None, "S_rᵀ S_i");
        }
    }

    #[test]
    fn t8r_fw_crossing_is_the_instability() {
        let n = 7;
        let q = orthogonal_rows(n, 0xF3_1CE5);
        let rr = vec![0.45, 0.95, 1.4, 2.1, 2.7, 3.2, 4.3];
        let ii = vec![-0.30, -0.05, 1.15, 1.85, 2.35, 2.95, 3.6];
        let hrr = sym_from_spectrum(&rr, &q, n);
        let hii = sym_from_spectrum(&ii, &q, n);
        let (a, b) = grams_from_blocks(&hrr, &hii);
        let signed = williamson_reduced_signed(&hrr, &hii, n, PSD_TOL).expect("Resonant{Rr}");
        assert_eq!(signed.n_neg, 2, "fixture must carry two unstable pairs");

        let cols: Vec<usize> = (0..n).collect();
        let modes = {
            let mut m = vec![0.0f64; n * n];
            for (c, &src) in cols.iter().enumerate() {
                for i in 0..n {
                    m[i * n + c] = signed.modes_i[i * n + src];
                }
            }
            m
        };
        let qf = crate::core::weingarten::weingarten_quadratic_forms(&a, &b, &modes, n, n);
        let f_w = weingarten_fraction(&a, &b, &modes, n, n);

        for (j, &w2) in signed.omega2.iter().enumerate() {
            let (qa, qb) = qf[j];
            assert!(
                qa > 0.0,
                "mode {j}: ⟨v,Av⟩ = {qa:.3e} must be positive — A is the (essentially PSD) imprint, \
                 and the identity below is stated for that sign"
            );
            assert!(
                qb > 0.0,
                "mode {j}: ⟨v,Bv⟩ = {qb:.3e} must be positive on this fixture"
            );
            let quot = quad_col(&hii, &modes, n, n, j);
            assert!(
                (quot - (qa - qb)).abs() <= 1e-10 * (qa.abs() + qb.abs()),
                "mode {j}: ⟨v,H^ii v⟩ = {quot:.6e} must equal ⟨v,Av⟩ − ⟨v,Bv⟩ = {:.6e}",
                qa - qb
            );
            assert_eq!(
                f_w[j] > 1.0,
                w2 < 0.0,
                "mode {j}: f_W = {} must exceed 1 exactly when ω² = {w2:.6e} is negative",
                f_w[j]
            );
        }
    }
}

#[cfg(test)]
mod shifted_tests;

#[cfg(test)]
pub(crate) mod right_chain_oracle;
