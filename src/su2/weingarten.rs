// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) Weingarten curvature term `B·ξ = P_X (D_ξ P_X) Z` — the duplicated layer of
//! `crate::core::weingarten` (the SU(2) layer duplicates wholesale, zero edits under `src/core/`
//! or `src/solvers/`; the dense Williamson kernels are REUSED from `crate::core::weingarten`
//! via imports).

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use crate::core::factor::svd_uvt_svals;
use crate::core::weingarten::{FloorPolicy, Terms};
use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::coreops::{adjoint_core_times_core, mul_sketch_left, mul_sketch_right, rproject};
use crate::su2::flux::{
    CrossDir, Su2ShiftedFrames, cross_wire, flux_grids, lift_bond_sketch, regrid_core,
    shifted_out_space,
};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::matvec::Su2OneSiteWorkspace;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::spinalgebra::SpinAlgebra;
use crate::su2::tangent::{Su2TtTangent, shifted_frames};
use crate::su2::vector::Su2TtVector;
use std::sync::Arc;

pub(crate) type Su2ZEnvs<'a, T> = (&'a [Su2Sketch<T>], &'a [Su2Sketch<T>]);

pub(crate) trait Su2ZSource<T: Scalar> {
    fn left_mul(&self, s: &Su2Sketch<T>, k: usize) -> Su2Core<T>;
    fn right_mul(&self, k: usize, s: &Su2Sketch<T>) -> Su2Core<T>;
    fn bond_ranks(&self, b: usize) -> Su2RankGrid;

    fn cached_envs(&self) -> Option<Su2ZEnvs<'_, T>> {
        None
    }
}

pub(crate) struct Su2ExplicitZ<'a, T: Scalar> {
    z: &'a Su2TtVector<T>,
    envs: Option<Su2ZEnvs<'a, T>>,
}

impl<'a, T: Scalar> Su2ExplicitZ<'a, T> {
    pub(crate) fn new(z: &'a Su2TtVector<T>) -> Self {
        Su2ExplicitZ { z, envs: None }
    }

    pub(crate) fn with_envs(mut self, pz: &'a [Su2Sketch<T>], qz: &'a [Su2Sketch<T>]) -> Self {
        self.envs = Some((pz, qz));
        self
    }
}

impl<T: Scalar> Su2ZSource<T> for Su2ExplicitZ<'_, T> {
    fn left_mul(&self, s: &Su2Sketch<T>, k: usize) -> Su2Core<T> {
        mul_sketch_left(s, self.z.core(k))
    }
    fn right_mul(&self, k: usize, s: &Su2Sketch<T>) -> Su2Core<T> {
        mul_sketch_right(self.z.core(k), s)
    }
    fn bond_ranks(&self, b: usize) -> Su2RankGrid {
        self.z.ranks[b].clone()
    }
    fn cached_envs(&self) -> Option<Su2ZEnvs<'_, T>> {
        self.envs
    }
}

/// The LEFT tangent gauge projector `gauge_k(C) = C − L·(LᴴC)` in sketch form (unweighted —
/// `base_l` is left-canonical, plain-orthonormal; same algebra as the abelian vertical-unfolding
/// version, `core/weingarten.rs::gauge_complement_mut`). Mutates `comp` in place.
pub fn gauge_complement_mut<T: Scalar>(comp: &mut Su2Core<T>, l_core: &Su2Core<T>) {
    let lhc = adjoint_core_times_core(l_core, comp);
    let corr = mul_sketch_right(l_core, &lhc);
    comp.axpy(T::from_f64(-1.0), &corr);
}

pub(crate) struct Su2FusedTermA<'a, T: Scalar> {
    pub(crate) h: &'a Su2SparseHamiltonian<T>,
    pub(crate) u: &'a Su2TtVector<T>,
    pub(crate) v: &'a Su2TtVector<T>,
    pub(crate) l: &'a [Su2Sketch<T>],
    pub(crate) r: &'a [Su2Sketch<T>],
    pub(crate) ws_u: &'a [Su2OneSiteWorkspace],
    pub(crate) ws_v: &'a [Su2OneSiteWorkspace],
    pub(crate) ws_du: &'a [Su2OneSiteWorkspace],
}

impl<T: Scalar> Su2FusedTermA<'_, T> {
    pub(crate) fn components(&self, xi: &[Su2Core<T>]) -> Vec<Su2Core<T>> {
        let (h, u, v) = (self.h, self.u, self.v);
        let d = u.space.d;

        let mut lt: Vec<Option<Su2Sketch<T>>> = Vec::with_capacity(d);
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

        let mut rt: Vec<Option<Su2Sketch<T>>> = vec![None; d];
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

fn right_gauge_complement_mut<T: Scalar>(comp: &mut Su2Core<T>, m_core: &Su2Core<T>) {
    let cm = rproject(comp, m_core);
    let corr = mul_sketch_left(&cm, m_core);
    comp.axpy(T::from_f64(-1.0), &corr);
}

/// Per-bond bond gauge `R_k` (`X_{≥k+1} = X̃_{≥k+1} R_k`) with its floored inverse and Schmidt
/// values — the SU(2) twin of `crate::core::weingarten::BondGauge` (per reduced sector; the
/// multiplet-degeneracy weights do NOT enter here: `R_k` is the plain interface change-of-basis
/// between the two factorizations, assembled by LEFT folds only).
#[derive(Clone)]
pub struct Su2BondGauge<T: Scalar> {
    /// Interior bond index `1..d`.
    pub k: usize,
    /// `R_k` itself (the overlap sketch), `r_k×r_k` per sector.
    pub r: Su2Sketch<T>,
    /// Floored `R_k⁻¹`, `r_k×r_k` per sector.
    pub r_inv: Su2Sketch<T>,
    /// Per-sector Schmidt values (singular values of `R_k`), descending.
    pub sigma: Vec<(Su2Sector, Vec<f64>)>,
    /// Dropped Schmidt mass `Σ σ²` over floored directions (never silent).
    pub dropped_mass: f64,
}

/// Bond gauges `{R_k}` for the interior bonds `1..d`: fold the overlap
/// `G_k = base_l(0..k−1)ᴴ · base_r(0..k−1)` from the left — plain folds only (`base_l` is
/// plain-left-canonical), the identical fold `Su2TtTangent::project`'s `p` chain uses.
pub fn bond_gauges<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    floor: FloorPolicy,
) -> Vec<Su2BondGauge<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut out = Vec::new();
    let mut pg = Su2Sketch::<T>::identity(space, 0, 1);
    for b in 1..d {
        let br_folded = mul_sketch_left(&pg, base_r.core(b - 1));
        pg = adjoint_core_times_core(base_l.core(b - 1), &br_folded);
        let (r_inv, sigma, dropped) = invert_bond(&pg, floor);
        out.push(Su2BondGauge {
            k: b,
            r: pg.clone(),
            r_inv,
            sigma,
            dropped_mass: dropped,
        });
    }
    out
}

/// The smallest CENTERWEIGHTED bond Schmidt value at the frames `base_l`/`base_r` — the SU(2)
/// twin of `crate::core::tangent::min_schmidt`, and the manifold-border tripwire the saddle
/// dynamics reads at every accepted point.
pub fn min_schmidt<T: Scalar>(base_l: &Su2TtVector<T>, base_r: &Su2TtVector<T>) -> f64 {
    let two_s_total = base_l.space.two_s;
    bond_gauges(base_l, base_r, FloorPolicy::default_hard())
        .iter()
        .flat_map(|g| {
            g.sigma.iter().flat_map(move |(s, v)| {
                let w = crate::su2::centerweight(s.two_s, two_s_total).sqrt();
                v.iter().map(move |&sig| w * sig)
            })
        })
        .fold(f64::INFINITY, f64::min)
}

type BondInverse<T> = (Su2Sketch<T>, Vec<(Su2Sector, Vec<f64>)>, f64);

fn invert_bond<T: Scalar>(r: &Su2Sketch<T>, floor: FloorPolicy) -> BondInverse<T> {
    use faer::prelude::ReborrowMut;
    let space = r.space;
    let k = r.site();
    let ranks = r.row_ranks();
    let mut r_inv = Su2Sketch::<T>::zeros(space, k, &ranks, &ranks);
    let mut sigma_all: Vec<(Su2Sector, Vec<f64>)> = Vec::new();
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

/// [`bond_gauges`] under `FloorPolicy::Tikhonov(s)` measured in CENTERWEIGHTED Schmidt units —
/// the SU(2) gauge set of the σ-damped trust-region model
/// ([`crate::solvers::joint::WDampRule`]). Nothing else may use it: this centerweighted damped
/// `Σ⁻¹` is a step model, never the Riemannian Hessian the spectral consumers need. The RAW-σ
/// damped set `bond_gauges(_, _, FloorPolicy::Tikhonov(ε))` is also the `CurvatureReg::Tikhonov`
/// curvature's (the opt-in congruence `Dc·B·Dc` of the spectral consumers) — this
/// centerweighted twin never is.
pub fn bond_gauges_tikhonov_cw<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    s: f64,
) -> Vec<Su2BondGauge<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut out = Vec::new();
    let mut pg = Su2Sketch::<T>::identity(space, 0, 1);
    for b in 1..d {
        let br_folded = mul_sketch_left(&pg, base_r.core(b - 1));
        pg = adjoint_core_times_core(base_l.core(b - 1), &br_folded);
        let (r_inv, sigma, dropped) = invert_bond_tikhonov_cw(&pg, s);
        out.push(Su2BondGauge {
            k: b,
            r: pg.clone(),
            r_inv,
            sigma,
            dropped_mass: dropped,
        });
    }
    out
}

fn invert_bond_tikhonov_cw<T: Scalar>(r: &Su2Sketch<T>, s: f64) -> BondInverse<T> {
    use faer::prelude::ReborrowMut;
    let space = r.space;
    let k = r.site();
    let two_s_total = space.two_s;
    let ranks = r.row_ranks();
    let mut r_inv = Su2Sketch::<T>::zeros(space, k, &ranks, &ranks);
    let mut sigma_all: Vec<(Su2Sector, Vec<f64>)> = Vec::new();
    for sec in r.sectors() {
        let Some(blk) = r.block(sec) else { continue };
        let n = blk.nrows();
        if n == 0 {
            sigma_all.push((sec, Vec::new()));
            continue;
        }
        let (w, svals, vt) = svd_uvt_svals(blk);
        sigma_all.push((sec, svals.clone()));
        let cw = crate::su2::centerweight(sec.two_s, two_s_total);
        let mut sinv = vec![T::zero(); svals.len()];
        for (i, &sig) in svals.iter().enumerate() {
            sinv[i] = T::from_f64(cw * sig / (cw * sig * sig + s * s));
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
        if let Some(mut dst) = r_inv.block_mut(sec) {
            for j in 0..n {
                for i in 0..n {
                    *dst.rb_mut().get_mut(i, j) = *rinv.get(i, j);
                }
            }
        }
    }
    (r_inv, sigma_all, 0.0)
}

pub(crate) fn left_env_z<T: Scalar>(
    base_l: &Su2TtVector<T>,
    z: &dyn Su2ZSource<T>,
) -> Vec<Su2Sketch<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut p: Vec<Su2Sketch<T>> = Vec::with_capacity(d + 1);
    p.push(Su2Sketch::identity(space, 0, 1));
    for k in 0..d {
        let pd = z.left_mul(&p[k], k);
        p.push(adjoint_core_times_core(base_l.core(k), &pd));
    }
    p
}

pub(crate) fn right_env_z<T: Scalar>(
    base_r: &Su2TtVector<T>,
    z: &dyn Su2ZSource<T>,
) -> Vec<Su2Sketch<T>> {
    right_env_z_visit(base_r, z, |_, _| {})
}

pub(crate) fn right_env_z_visit<T: Scalar, F>(
    base_r: &Su2TtVector<T>,
    z: &dyn Su2ZSource<T>,
    mut visit: F,
) -> Vec<Su2Sketch<T>>
where
    F: FnMut(usize, &Su2Core<T>),
{
    let space = base_r.space;
    let d = space.d;
    let mut q: Vec<Option<Su2Sketch<T>>> = (0..=d).map(|_| None).collect();
    q[d] = Some(Su2Sketch::identity(space, d, 1));
    for k in (0..d).rev() {
        let dq = z.right_mul(k, q[k + 1].as_ref().unwrap());
        visit(k, &dq);
        q[k] = Some(rproject(&dq, base_r.core(k)));
    }
    q.into_iter().map(|x| x.unwrap()).collect()
}

fn var_left_env<T: Scalar>(
    base_l: &Su2TtVector<T>,
    z: &dyn Su2ZSource<T>,
    comps_scaled: &[Su2Core<T>],
    pz: &[Su2Sketch<T>],
) -> Vec<Su2Sketch<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut evl: Vec<Su2Sketch<T>> = Vec::with_capacity(d);
    evl.push(Su2Sketch::zeros(
        space,
        0,
        &base_l.ranks[0],
        &z.bond_ranks(0),
    ));
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
    frame: &Su2TtVector<T>,
    z: &dyn Su2ZSource<T>,
    comps_r: &[Su2Core<T>],
    qz: &[Su2Sketch<T>],
) -> Vec<Su2Sketch<T>> {
    let space = frame.space;
    let d = space.d;
    let mut evr: Vec<Option<Su2Sketch<T>>> = (0..=d).map(|_| None).collect();
    evr[0] = Some(Su2Sketch::zeros(
        space,
        0,
        &z.bond_ranks(0),
        &frame.ranks[0],
    ));
    evr[d] = Some(Su2Sketch::zeros(
        space,
        d,
        &z.bond_ranks(d),
        &frame.ranks[d],
    ));
    for b in (1..d).rev() {
        let t1a = z.right_mul(b, evr[b + 1].as_ref().unwrap());
        let mut cur = rproject(&t1a, frame.core(b));
        let t2a = z.right_mul(b, &qz[b + 1]);
        let t2 = rproject(&t2a, &comps_r[b - 1]);
        cur.add_assign_sketch(&t2);
        evr[b] = Some(cur);
    }
    evr.into_iter().map(|x| x.unwrap()).collect()
}

fn adjoint_blocks<T: Scalar>(s: &Su2Sketch<T>) -> Su2Sketch<T> {
    let mut out = s.clone();
    for sec in s.sectors() {
        if let (Some(src), Some(mut dst)) = (s.block(sec), out.block_mut(sec)) {
            dst.copy_from(src.adjoint());
        }
    }
    out
}

type RightChainInputs<T> = (Vec<Su2Core<T>>, Vec<Su2Sketch<T>>);

fn right_project_components<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    dir: &Su2TtVector<T>,
) -> Vec<Su2Core<T>> {
    let space = base_l.space;
    let d = space.d;
    let mut p: Vec<Su2Sketch<T>> = Vec::with_capacity(d);
    p.push(Su2Sketch::identity(space, 0, 1));
    for k in 0..d.saturating_sub(1) {
        let pd = mul_sketch_left(&p[k], dir.core(k));
        p.push(adjoint_core_times_core(base_l.core(k), &pd));
    }
    let mut q = Su2Sketch::identity(space, d, 1);
    let mut comps: Vec<Option<Su2Core<T>>> = (0..d).map(|_| None).collect();
    for k in (0..d).rev() {
        let dq = mul_sketch_right(dir.core(k), &q);
        let mut comp = mul_sketch_left(&p[k], &dq);
        if k >= 1 {
            right_gauge_complement_mut(&mut comp, base_r.core(k));
            q = rproject(&dq, base_r.core(k));
        }
        comps[k] = Some(comp);
    }
    comps.into_iter().map(|c| c.unwrap()).collect()
}

/// Assemble `B·ξ` as a [`Su2TtTangent`] at the gauge frames, contracting the **normal** residual
/// `z = r⊥`. PRECONDITION `P(z) = 0` (dev-build guard) — same normal-`Z` derivation as the
/// abelian `crate::core::weingarten::weingarten_assemble` (whose docs hold verbatim), with the
/// θ² pairing substitutions of the module docs. `terms` selects summands (ablation); ship with
/// `Terms::all()`.
pub fn weingarten_assemble<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    z: &Su2TtVector<T>,
    xi: &Su2TtTangent<T>,
    floor: FloorPolicy,
    terms: Terms,
) -> Su2TtTangent<T> {
    let gauges = bond_gauges(base_l, base_r, floor);
    weingarten_assemble_cached(base_l, base_r, z, xi, &gauges, terms)
}

/// [`weingarten_assemble`] with the bond gauges supplied by the caller — `gauges[b−1]` is bond `b`
/// from [`bond_gauges`] at the SAME `(base_l, base_r)` frames (the floor is baked into the cached
/// `R⁻¹`). The gauges depend only on the base point, so an iterative solver making many `B·ξ`
/// applies at a fixed base computes them once (SU(2) twin of the abelian
/// `crate::core::weingarten::weingarten_assemble_cached`).
pub fn weingarten_assemble_cached<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    z: &Su2TtVector<T>,
    xi: &Su2TtTangent<T>,
    gauges: &[Su2BondGauge<T>],
    terms: Terms,
) -> Su2TtTangent<T> {
    #[cfg(debug_assertions)]
    {
        let zn = z.norm();
        if zn > 0.0 {
            let tang = Su2TtTangent::project(base_l, base_r, z).norm();
            assert!(
                tang <= 1e-6 * zn,
                "su2 weingarten_assemble: Z is not normal (‖P(Z)‖/‖Z‖ = {:.3e})",
                tang / zn
            );
        }
    }
    let (bl, br) = (Arc::new(base_l.clone()), Arc::new(base_r.clone()));
    weingarten_assemble_sources(&bl, &br, &[&Su2ExplicitZ::new(z)], xi, gauges, terms)
}

pub(crate) fn weingarten_assemble_sources<T: Scalar>(
    base_l: &Arc<Su2TtVector<T>>,
    base_r: &Arc<Su2TtVector<T>>,
    sources: &[&dyn Su2ZSource<T>],
    xi: &Su2TtTangent<T>,
    gauges: &[Su2BondGauge<T>],
    terms: Terms,
) -> Su2TtTangent<T> {
    let sf = (xi.two_s_flux() != 0).then(|| Su2ShiftedFrames::new(base_l, base_r, xi.two_s_flux()));
    weingarten_assemble_sources_dressed(base_l, base_r, sources, xi, gauges, terms, sf.as_ref())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn weingarten_assemble_sources_dressed<T: Scalar>(
    base_l: &Arc<Su2TtVector<T>>,
    base_r: &Arc<Su2TtVector<T>>,
    sources: &[&dyn Su2ZSource<T>],
    xi: &Su2TtTangent<T>,
    gauges: &[Su2BondGauge<T>],
    terms: Terms,
    sf: Option<&Su2ShiftedFrames<T>>,
) -> Su2TtTangent<T> {
    let space = base_l.space;
    let d = space.d;
    let r = base_l.ranks.clone();
    let two_s_flux = xi.two_s_flux();
    let two_kappa = two_s_flux as i64;
    let shifted = two_s_flux != 0;
    debug_assert_eq!(
        base_l.space, base_r.space,
        "weingarten_assemble_sources: the two frames must be stamped the same space"
    );
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "weingarten_assemble_sources: the two frames must factorize the SAME state"
    );
    debug_assert!(
        !shifted || sf.is_some_and(|s| s.two_s_flux == two_s_flux),
        "weingarten_assemble_sources_dressed: the bundle must be this channel's ({two_s_flux})"
    );
    debug_assert!(
        shifted || sf.is_none(),
        "weingarten_assemble_sources_dressed: xi carries no spin flux, so no shifted bundle may be \
         handed in (the flux-0 regime must run on x's OWN grids)"
    );
    let out_space = sf.map_or(space, |s| s.out_space);
    let rows: &[Su2RankGrid] = sf.map_or(r.as_slice(), |s| s.rm.as_slice());
    let cols: &[Su2RankGrid] = sf.map_or(r.as_slice(), |s| s.rd.as_slice());
    debug_assert_eq!(
        xi.rank(),
        rows,
        "weingarten_assemble_sources: xi was built at different frames or a different flux"
    );
    debug_assert_eq!(
        xi.rank_col(),
        cols,
        "weingarten_assemble_sources: xi's column grids are not this channel's dressed fan"
    );
    debug_assert!(
        !shifted || xi.base_l.space == out_space,
        "weingarten_assemble_sources: xi's frames are stamped {:?} but the flux {two_s_flux} on \
         {space:?} gives {out_space:?}",
        xi.base_l.space
    );

    if d == 1 {
        let comp = Su2Core::zeros(out_space, 0, &rows[0], &cols[1]);
        if let Some(s) = sf {
            return Su2TtTangent::from_parts_shifted(
                s.rm.clone(),
                s.rd.clone(),
                two_s_flux,
                s.frames.0.clone(),
                s.frames.1.clone(),
                vec![comp],
            );
        }
        return Su2TtTangent::from_parts(r, base_l.clone(), base_r.clone(), vec![comp]);
    }

    if let Some(s) = sf {
        let empty =
            s.rd[d].total() == 0 || (1..d).any(|b| s.rm[b].total() == 0 && s.rd[b].total() == 0);
        if empty {
            let comps = (0..d)
                .map(|k| Su2Core::zeros(out_space, k, &s.rm[k], &s.rd[k + 1]))
                .collect();
            return Su2TtTangent::from_parts_shifted(
                s.rm.clone(),
                s.rd.clone(),
                two_s_flux,
                s.frames.0.clone(),
                s.frames.1.clone(),
                comps,
            );
        }
    }

    debug_assert_eq!(
        gauges.len(),
        d - 1,
        "weingarten_assemble_cached: gauge count"
    );

    let (fl, fr) = sf.map_or((&**base_l, &**base_r), |s| (&*s.frames.0, &*s.frames.1));
    let zl = sf.map_or(&**base_l, |s| &s.zside);
    let mut spin = SpinAlgebra::new();

    let left_terms = terms.diag_s1 || terms.diag_s2;
    let comps_scaled: Vec<Su2Core<T>> = if left_terms {
        (0..d - 1)
            .map(|k| match sf {
                Some(s) => {
                    let lifted = lift_bond_sketch(
                        &gauges[k].r_inv,
                        &r[k + 1],
                        &s.rd[k + 1],
                        out_space,
                        two_kappa,
                    );
                    mul_sketch_right(xi.component(k), &lifted)
                }
                None => mul_sketch_right(xi.component(k), &gauges[k].r_inv),
            })
            .collect()
    } else {
        Vec::new()
    };
    let comps_scaled_z: Vec<Su2Core<T>> = match sf {
        Some(s) if terms.diag_s1 => (0..d - 1)
            .map(|k| regrid_core(&comps_scaled[k], space, &r[k], &s.rdz[k + 1]))
            .collect(),
        _ => Vec::new(),
    };
    let evl_comps = if shifted {
        &comps_scaled_z
    } else {
        &comps_scaled
    };
    let right_chain: Option<RightChainInputs<T>> = terms.diag_s3.then(|| {
        let comps_r = right_project_components(fl, fr, &xi.to_vector());
        let comps_r_z: Vec<Su2Core<T>> = match sf {
            Some(s) => (1..d)
                .map(|b| {
                    cross_wire(
                        &comps_r[b],
                        CrossDir::AtoB,
                        two_kappa,
                        (&s.rm[b], &s.rd[b + 1]),
                        (&s.rdz[b], &r[b + 1]),
                        out_space,
                        space,
                        (&r[b], &r[b + 1]),
                        &mut spin,
                    )
                })
                .collect(),
            None => comps_r.into_iter().skip(1).collect(),
        };
        let rinv_h: Vec<Su2Sketch<T>> = (1..d)
            .map(|b| match sf {
                Some(s) => adjoint_blocks(&lift_bond_sketch(
                    &gauges[b - 1].r_inv,
                    &r[b],
                    &s.rdz[b],
                    space,
                    two_kappa,
                )),
                None => adjoint_blocks(&gauges[b - 1].r_inv),
            })
            .collect();
        (comps_r_z, rinv_h)
    });

    let mut out_cores: Vec<Su2Core<T>> = (0..d)
        .map(|k| Su2Core::<T>::zeros(out_space, k, &rows[k], &cols[k + 1]))
        .collect();

    for &z in sources {
        let folded;
        let (pz, qz): (&[Su2Sketch<T>], &[Su2Sketch<T>]) = match z.cached_envs() {
            Some(pq) => pq,
            None => {
                folded = (left_env_z(base_l, z), right_env_z(base_r, z));
                (&folded.0, &folded.1)
            }
        };
        let evl = if terms.diag_s1 {
            var_left_env(zl, z, evl_comps, pz)
        } else {
            Vec::new()
        };
        let evr = match &right_chain {
            Some((comps_r, rinv_h)) => {
                let mut evr = var_right_env(zl, z, comps_r, qz);
                for (e, rh) in evr[1..d].iter_mut().zip(rinv_h) {
                    debug_assert!(
                        e.sectors().into_iter().all(|s| {
                            e.block(s).is_none_or(|b| b.nrows() * b.ncols() == 0)
                                || rh.block(s).is_some()
                        }),
                        "weingarten_assemble_sources_dressed: a live block of eVR'[{}] has no R⁻ᴴ \
                         block",
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
                    let diag = match sf {
                        Some(s) => {
                            let rzd = z.bond_ranks(d);
                            cross_wire(
                                &diag,
                                CrossDir::BtoA,
                                two_kappa,
                                (&s.rdz[k], &rzd),
                                (&s.rm[k], &s.rd[d]),
                                space,
                                out_space,
                                (&r[k], &r[d]),
                                &mut spin,
                            )
                        }
                        None => diag,
                    };
                    middle.axpy(T::one(), &diag);
                }
            } else {
                if left_terms {
                    let zq = z.right_mul(k, &qz[k + 1]);
                    if terms.diag_s1 {
                        let s1 = mul_sketch_left(&evl[k], &zq);
                        let s1 = match sf {
                            Some(s) => cross_wire(
                                &s1,
                                CrossDir::BtoA,
                                two_kappa,
                                (&s.rdz[k], &r[k + 1]),
                                (&s.rm[k], &s.rd[k + 1]),
                                space,
                                out_space,
                                (&r[k], &r[k + 1]),
                                &mut spin,
                            ),
                            None => s1,
                        };
                        middle.axpy(T::one(), &s1);
                    }
                    if terms.diag_s2 {
                        let g20 = mul_sketch_left(&pz[k], &zq);
                        let w2 = adjoint_core_times_core(base_l.core(k), &g20);
                        let w2 = match sf {
                            Some(s) => {
                                lift_bond_sketch(&w2, &r[k + 1], &s.rd[k + 1], out_space, two_kappa)
                            }
                            None => w2,
                        };
                        let s2 = mul_sketch_right(&comps_scaled[k], &w2);
                        middle.axpy(T::from_f64(-1.0), &s2);
                    }
                }
                if terms.diag_s3 {
                    let s3 = mul_sketch_left(&pz[k], &z.right_mul(k, &evr[k + 1]));
                    let s3 = match sf {
                        Some(s) => regrid_core(&s3, out_space, &s.rm[k], &s.rd[k + 1]),
                        None => s3,
                    };
                    middle.axpy(T::one(), &s3);
                }
            }
        }
    }

    for (k, core) in out_cores.iter_mut().enumerate().take(d - 1) {
        gauge_complement_mut(core, fl.core(k));
    }
    match sf {
        Some(s) => Su2TtTangent::from_parts_shifted(
            s.rm.clone(),
            s.rd.clone(),
            two_s_flux,
            s.frames.0.clone(),
            s.frames.1.clone(),
            out_cores,
        ),
        None => Su2TtTangent::from_parts(r, base_l.clone(), base_r.clone(), out_cores),
    }
}

/// General-`z` entry point: normal-projects `z ↦ z − ι(P(z))` then delegates to
/// [`weingarten_assemble`] (the SU(2) twin of the abelian `weingarten_assemble_general`).
pub fn weingarten_assemble_general<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    z: &Su2TtVector<T>,
    xi: &Su2TtTangent<T>,
    floor: FloorPolicy,
    terms: Terms,
) -> Su2TtTangent<T> {
    let tang = Su2TtTangent::project(base_l, base_r, z).to_vector();
    let zn = z - &tang;
    weingarten_assemble(base_l, base_r, &zn, xi, floor, terms)
}

const MGS_REL_TOL: f64 = 1e-8;

const MGS_EMPTY_SLOT: f64 = 1e-12;

const MGS_NOISE_FLOOR: f64 = 1e-13;

/// Dimension of the gauge-fixed SU(2) tangent parametrization at `base_l`'s rank grid (radial
/// INCLUDED) — the abelian `crate::core::tangent::tangent_dim` duplicated onto the SU(2) slot
/// structure. Structural: an UPPER bound on `tangent_basis(…, true).len() + 1`, with equality at
/// a numerically full-rank point; at a rank-padded point the realized span sits below it.
/// The Krylov drivers use it as the exhaustion cap.
pub fn tangent_dim<T: Scalar>(base_l: &Su2TtVector<T>) -> usize {
    tangent_dim_grids(&base_l.space, &base_l.ranks, 0)
}

/// `dim T^{S'}` — the STRUCTURAL dimension (shapes only) of the shifted tangent space at spin flux
/// `two_s_flux = 2κ` on the rank grids `r` of a reference in `space`:
pub fn tangent_dim_grids(space: &Su2Space, r: &[Su2RankGrid], two_s_flux: usize) -> usize {
    let space = *space;
    let out_space = shifted_out_space(space, two_s_flux);
    let d = space.d;
    let rows = flux_grids(r, 0, space, out_space);
    let cols = flux_grids(r, two_s_flux as i64, space, out_space);
    let mut raw = 0usize;
    let mut gauge = 0usize;
    for k in 0..d {
        let g = out_space.pg.irrep_of(k);
        for l in out_space.bond_sectors(k) {
            for f in Family::ALL {
                let Some(t) = f.apply(l, g) else { continue };
                if out_space.contains(k + 1, t) {
                    raw += rows[k][l] * cols[k + 1][t];
                }
            }
        }
        if k < d - 1 {
            for s in out_space.bond_sectors(k + 1) {
                gauge += rows[k + 1][s] * cols[k + 1][s];
            }
        }
    }
    raw - gauge
}

/// Deterministic `⟨·,·⟩`-orthonormal basis of the SU(2) tangent space (the abelian enumerator
/// duplicated): elementary component-core directions, LEFT-gauge-projected for `k < d−1`,
/// modified Gram–Schmidt in the CENTERWEIGHTED tangent metric (`Su2TtTangent::dot` — the H9
/// hazard: a plain-metric MGS here would silently produce a non-orthonormal basis).
/// `exclude_radial` seeds with the tangent representation of `x` (excluded from the output).
pub fn tangent_basis<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    exclude_radial: bool,
) -> Vec<Su2TtTangent<T>> {
    tangent_basis_impl(base_l, base_r, 0, exclude_radial)
}

/// The `⟨·,·⟩`-orthonormal basis of the SHIFTED tangent space `T^{S'}` at spin flux
/// `two_s_flux = 2κ` — the same enumerator, over the shifted frame pair and the RECTANGULAR slot
/// grid `r_row[k][l] × r'[k+1][f(l)]`.
pub fn tangent_basis_shifted<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    two_s_flux: usize,
) -> Vec<Su2TtTangent<T>> {
    tangent_basis_impl(base_l, base_r, two_s_flux, false)
}

fn tangent_basis_impl<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    two_s_flux: usize,
    exclude_radial: bool,
) -> Vec<Su2TtTangent<T>> {
    use faer::prelude::ReborrowMut;
    debug_assert!(
        two_s_flux == 0 || !exclude_radial,
        "tangent_basis_impl: a shifted space has no radial mode to exclude"
    );
    let space = base_l.space;
    let out_space = shifted_out_space(space, two_s_flux);
    let d = space.d;
    let (frame_l, frame_r) = shifted_frames(base_l, base_r, two_s_flux);
    let (frame_l, frame_r) = (Arc::new(frame_l), Arc::new(frame_r));
    let r = frame_l.ranks.clone();
    let r_col = frame_r.ranks.clone();
    let zero_components = || -> Vec<Su2Core<T>> {
        (0..d)
            .map(|k| Su2Core::zeros(out_space, k, &r[k], &r_col[k + 1]))
            .collect()
    };
    let purify = |dir: &Su2TtVector<T>| -> Su2TtTangent<T> {
        Su2TtTangent::project_impl(frame_l.clone(), frame_r.clone(), two_s_flux, dir)
    };

    let mut basis: Vec<Su2TtTangent<T>> = Vec::new();
    if exclude_radial {
        let mut radial = Su2TtTangent::project(base_l, base_r, base_l);
        let n = radial.norm();
        if n > 0.0 {
            radial.scale_mut(T::from_f64(1.0 / n));
            basis.push(radial);
        }
    }
    let n_seeds = basis.len();
    let dim = tangent_dim_grids(&space, &base_l.ranks, two_s_flux);

    'enumerate: for k in 0..d {
        let g = out_space.pg.irrep_of(k);
        for l in out_space.bond_sectors(k) {
            for f in Family::ALL {
                let Some(t) = f.apply(l, g) else { continue };
                if !out_space.contains(k + 1, t) {
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
                            gauge_complement_mut(&mut comps[k], frame_l.core(k));
                        }
                        let u = Su2TtTangent::from_parts_shifted(
                            r.clone(),
                            r_col.clone(),
                            two_s_flux,
                            frame_l.clone(),
                            frame_r.clone(),
                            comps,
                        );
                        let mut u = purify(&u.to_vector());
                        let n0 = u.norm();
                        if n0 <= MGS_EMPTY_SLOT {
                            continue;
                        }
                        for _ in 0..2 {
                            for b in &basis {
                                let c = b.dot(&u);
                                u.axpy(c * T::from_f64(-1.0), b);
                            }
                        }
                        let mut u = purify(&u.to_vector());
                        let nrm = u.norm();
                        if nrm > MGS_REL_TOL * n0 && nrm > MGS_NOISE_FLOOR {
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

#[cfg(test)]
mod shifted_tests;

#[cfg(test)]
mod cb_tests;

#[cfg(test)]
pub(crate) mod right_chain_oracle;
