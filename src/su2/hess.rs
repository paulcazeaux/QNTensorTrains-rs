// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) Riemannian Hessian of the energy — the duplicated layer of `crate::solvers::hess`.
//! Faer-free: the block algebra lives in
//! [`crate::su2::weingarten`]; the dense symplectic kernels (Eq.-20 reduced Williamson, full
//! 2n×2n Williamson, generator routes) are REUSED from `crate::core::weingarten` /
//! `crate::solvers::hess` behind their flat-slice interfaces.

use crate::core::weingarten::{FloorPolicy, Terms};
use crate::scalar::Scalar;
use crate::solvers::gad::{GadInfo, GadOpts, WModel, impl_gad};
use crate::solvers::hess::{CurvatureReg, deflate_subspace};
#[cfg(test)]
mod cb_tests;
#[cfg(test)]
mod curvature_tests;
#[cfg(test)]
mod zsplit_tests;

mod sector;
mod tda;

pub use sector::{Su2SectorCtx, Su2SectorResponse};
pub use tda::tda_lowest_davidson;

mod cone;
pub(crate) use cone::iso_defect as cone_iso_defect;
pub use cone::{
    Su2ConeMixedRows, cone_response, cone_response_matfree, cone_response_matfree_with_rows,
    cone_response_matfree_with_stats, cone_response_with_stats, try_cone_response,
};
mod davidson_paired;
pub use davidson_paired::paired_lowest_davidson;
mod symplectic_dense;
pub use symplectic_dense::{
    hessian_grams, hessian_grams_c64, hessian_grams_owned, symplectic_eigenvalues,
    symplectic_eigenvalues_c64,
};
mod newton_polish;
pub(crate) use newton_polish::{
    clean_tangent, deflate_radial, lowest_hessian_mode_at, radial_mode, reproject_tangent,
};
pub use newton_polish::{
    hessian_pd_probe, hessian_pd_probe_c64, hessian_pd_probe_c64_witness, hessian_pd_probe_witness,
    hessian_pd_probe_witness_with, newton_polish,
};
mod drivers;
pub(crate) use drivers::a_diagonal_squared;
#[cfg(test)]
pub(crate) use drivers::transport_seeds;
pub use drivers::{
    su2_select_rpa_metric, symplectic_lowest_signed_davidson,
    symplectic_lowest_signed_davidson_c64, symplectic_lowest_signed_davidson_targeted,
};
mod mode_check;
mod padding;
pub use mode_check::rpa_mode_checks;
pub use padding::{pad_singles, padded_context, singles_completeness};
mod channel;
pub use channel::{channel_ground_state, channel_response};

use crate::su2::core::Su2Core;
use crate::su2::deflate::Su2ColumnDeflation;
use crate::su2::flux::{Su2ShiftedFrames, shifted_out_space};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::matvec::Su2OneSiteWorkspace;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use crate::su2::qn::Su2RankGrid;
use crate::su2::riemannian::tangent_bases;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use crate::su2::weingarten::{
    Su2BondGauge, Su2ExplicitZ, Su2ZEnvs, Su2ZSource, bond_gauges, bond_gauges_tikhonov_cw,
    left_env_z, right_env_z, right_env_z_visit, weingarten_assemble_cached,
    weingarten_assemble_sources, weingarten_assemble_sources_dressed,
};
use std::sync::Arc;

/// `A·ξ = P_{T_xM}((H − λ)·ι(ξ))`, the effective-Hamiltonian term of the SU(2) Riemannian
/// Hessian — written as `P(H·ι(ξ)) − λ·ξ` via `P∘ι = id` (same identity as the abelian
/// `crate::solvers::hess::effham_apply`).
pub fn effham_apply<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    lambda: f64,
    xi: &Su2TtTangent<T>,
) -> Su2TtTangent<T> {
    let w = xi.to_vector();
    let hw = h.apply(&w);
    let mut out = Su2TtTangent::project(base_l, base_r, &hw);
    out.axpy(T::from_f64(-lambda), xi);
    out
}

pub(crate) fn assert_shifted_operator<T: Scalar>(
    h_out: &Su2SparseHamiltonian<T>,
    base_l: &Su2TtVector<T>,
    two_s_flux: usize,
) {
    let space_out = shifted_out_space(base_l.space, two_s_flux);
    assert_eq!(
        h_out.space, space_out,
        "the shifted Hamiltonian must be BUILT on space_out = {space_out:?} (the target 2S is baked \
         into the automaton's interned vertex set, `state_qns`, the per-site m_ord table and the \
         bond-d collapse, so a re-stamp is not enough); got {:?}",
        h_out.space
    );
}

/// `A^{S'}·ξ = P^{S'}((H_out − λ)·ι^{S'}(ξ))` — the spin-flux-`2κ` block of the effective-Hamiltonian
/// term of the SU(2) Riemannian Hessian, i.e. linear response from a SINGLET reference into the
/// `2S = 2κ` channel at the same particle number and the same point-group irrep.
pub fn effham_apply_shifted<T: Scalar>(
    h_out: &Su2SparseHamiltonian<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    two_s_flux: usize,
    lambda: f64,
    xi: &Su2TtTangent<T>,
) -> Su2TtTangent<T> {
    assert_shifted_operator(h_out, base_l, two_s_flux);
    let (u, v) = crate::su2::tangent::shifted_frames(base_l, base_r, two_s_flux);
    effham_apply_shifted_dressed(h_out, (&Arc::new(u), &Arc::new(v)), two_s_flux, lambda, xi)
}

pub(crate) fn effham_apply_shifted_dressed<T: Scalar>(
    h_out: &Su2SparseHamiltonian<T>,
    frames: (&Arc<Su2TtVector<T>>, &Arc<Su2TtVector<T>>),
    two_s_flux: usize,
    lambda: f64,
    xi: &Su2TtTangent<T>,
) -> Su2TtTangent<T> {
    let (u, v) = frames;
    assert_eq!(
        h_out.space, u.space,
        "effham_apply_shifted_dressed: the dressed frames are stamped {:?} but h_out is built on \
         {:?} — the dressed pair must live in h_out's own space_out",
        u.space, h_out.space
    );
    assert_eq!(
        xi.two_s_flux(),
        two_s_flux,
        "effham_apply_shifted: the direction carries spin flux {} but the channel is {two_s_flux}",
        xi.two_s_flux()
    );
    let hw = h_out.apply(&xi.to_vector());
    let mut out = Su2TtTangent::project_impl(u.clone(), v.clone(), two_s_flux, &hw);
    out.axpy(T::from_f64(-lambda), xi);
    out
}

/// The dense Gram `G_ij = ⟨b_i, A^{S'} b_j⟩` over a `⟨·,·⟩`-orthonormal basis of `T^{S'}`, row-major
/// `n×n` — the REAL layer only, exactly as the abelian
/// [`crate::solvers::hess::effham_gram_shifted`] is (on `C64` the Gram is HERMITIAN and a
/// `Vec<f64>` would silently drop `Im G`; the complex channel Gram is later work).
pub fn effham_gram_shifted(
    h_out: &Su2SparseHamiltonian<f64>,
    base_l: &Su2TtVector<f64>,
    base_r: &Su2TtVector<f64>,
    two_s_flux: usize,
    lambda: f64,
    basis: &[Su2TtTangent<f64>],
) -> Vec<f64> {
    assert_shifted_operator(h_out, base_l, two_s_flux);
    for (i, b) in basis.iter().enumerate() {
        assert_eq!(
            b.two_s_flux(),
            two_s_flux,
            "effham_gram_shifted: basis vector {i} carries spin flux {} but the channel is \
             {two_s_flux} — a Gram mixing fluxes is not the channel's operator",
            b.two_s_flux()
        );
    }
    let n = basis.len();
    let (u, v) = crate::su2::tangent::shifted_frames(base_l, base_r, two_s_flux);
    let (u, v) = (Arc::new(u), Arc::new(v));
    let pure: Vec<Su2TtTangent<f64>> = basis
        .iter()
        .map(|b| Su2TtTangent::project_impl(u.clone(), v.clone(), two_s_flux, &b.to_vector()))
        .collect();
    let mut a = vec![0.0f64; n * n];
    for (j, uj) in pure.iter().enumerate() {
        let aj = effham_apply_shifted_dressed(h_out, (&u, &v), two_s_flux, lambda, uj);
        for (i, ui) in pure.iter().enumerate() {
            a[i * n + j] = ui.dot(&aj);
        }
    }
    a
}

/// The **normal residual** `r⊥ = P⊥_{T_xM}(Hx − λx)` as a full SU(2) TT vector — the input the
/// Weingarten term contracts against.
pub fn normal_residual<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    x: &Su2TtVector<T>,
    lambda: f64,
) -> Su2TtVector<T> {
    let hx = h.apply(x);
    let full = &hx - &(x * T::from_f64(lambda));
    let tang = Su2TtTangent::project(base_l, base_r, &full).to_vector();
    &full - &tang
}

/// Precomputed context for the SU(2) Hessian applies at `x` (twin of
/// `crate::solvers::hess::WeingartenCtx`): gauge frames, `λ`, normal residual `Z = r⊥`, floor.
pub struct Su2WeingartenCtx<'h, T: Scalar> {
    h: &'h Su2SparseHamiltonian<T>,
    base_l: Arc<Su2TtVector<T>>,
    base_r: Arc<Su2TtVector<T>>,
    lambda: f64,
    floor: FloorPolicy,
    curvature: CurvatureReg,
    curv_op: std::sync::OnceLock<Option<Su2ColumnDeflation<T>>>,
    gauges: std::sync::OnceLock<Vec<Su2BondGauge<T>>>,
    gauges_curv: std::sync::OnceLock<Vec<Su2BondGauge<T>>>,
    gauges_damped: std::sync::OnceLock<(f64, Vec<Su2BondGauge<T>>)>,
    envs: std::sync::OnceLock<Su2HessEnvs<T>>,
    z_explicit: std::sync::OnceLock<Su2TtVector<T>>,
}

struct Su2HessEnvs<T: Scalar> {
    l: Vec<Su2Sketch<T>>,
    r: Vec<Su2Sketch<T>>,
    ws: Vec<Su2OneSiteWorkspace>,

    pz_hi: Vec<Su2Sketch<T>>,
    qz_hi: Vec<Su2Sketch<T>>,
    z_lo: Su2TtVector<T>,
    pz_lo: Vec<Su2Sketch<T>>,
    qz_lo: Vec<Su2Sketch<T>>,
    grad: Su2TtTangent<T>,
}

struct Su2ShiftedEnvs<T: Scalar> {
    l: Vec<Su2Sketch<T>>,
    r_kappa: Vec<Su2Sketch<T>>,
    ws_u: Vec<Su2OneSiteWorkspace>,
    ws_v: Vec<Su2OneSiteWorkspace>,
    ws_du: Vec<Su2OneSiteWorkspace>,
}

struct Su2ImplicitHZ<'a, T: Scalar> {
    h: &'a Su2SparseHamiltonian<T>,
    ket: &'a Su2TtVector<T>,
    ws: &'a [Su2OneSiteWorkspace],
    envs: Option<Su2ZEnvs<'a, T>>,
}

impl<T: Scalar> Su2ZSource<T> for Su2ImplicitHZ<'_, T> {
    fn left_mul(&self, s: &Su2Sketch<T>, k: usize) -> Su2Core<T> {
        let ket_k = self.ket.core(k);
        let mut y = Su2Core::zeros(ket_k.space, k, &s.row_ranks(), &self.h.col_ranks(ket_k));
        self.h.sketch_h_mul_into(
            &mut y,
            s,
            ket_k,
            |_| true,
            &self.ws[k].i_tab,
            &self.ws[k].j_tab,
        );
        y
    }

    fn right_mul(&self, k: usize, s: &Su2Sketch<T>) -> Su2Core<T> {
        let ket_k = self.ket.core(k);
        let mut y = Su2Core::zeros(ket_k.space, k, &self.h.row_ranks(ket_k), &s.col_ranks());
        self.h.h_sketch_mul_into(
            &mut y,
            ket_k,
            s,
            |_| true,
            &self.ws[k].i_tab,
            &self.ws[k].j_tab,
        );
        y
    }

    fn bond_ranks(&self, b: usize) -> Su2RankGrid {
        self.ket.ranks[b].clone()
    }

    fn cached_envs(&self) -> Option<Su2ZEnvs<'_, T>> {
        self.envs
    }
}

fn hz_stacks_and_projection<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    base_l: &Arc<Su2TtVector<T>>,
    base_r: &Arc<Su2TtVector<T>>,
    ket: &Su2TtVector<T>,
    ws: &[Su2OneSiteWorkspace],
) -> (Vec<Su2Sketch<T>>, Vec<Su2Sketch<T>>, Su2TtTangent<T>) {
    use crate::su2::coreops::mul_sketch_left;
    use crate::su2::weingarten::gauge_complement_mut;
    let d = base_l.space.d;
    let zhi = Su2ImplicitHZ {
        h,
        ket,
        ws,
        envs: None,
    };
    let pz_hi = left_env_z(base_l, &zhi);

    let mut slots: Vec<Option<Su2Core<T>>> = (0..d).map(|_| None).collect();
    let assemble = |k: usize, dq: &Su2Core<T>| -> Su2Core<T> {
        let mut c = mul_sketch_left(&pz_hi[k], dq);
        if k + 1 < d {
            gauge_complement_mut(&mut c, base_l.core(k));
        }
        c
    };
    let qz_hi = right_env_z_visit(base_r, &zhi, |k, dq| slots[k] = Some(assemble(k, dq)));
    let comps: Vec<Su2Core<T>> = slots.into_iter().map(|c| c.expect("visited")).collect();
    let proj =
        Su2TtTangent::from_parts(base_l.ranks.clone(), base_l.clone(), base_r.clone(), comps);
    (pz_hi, qz_hi, proj)
}

fn hz_stacks_and_gradient<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    base_l: &Arc<Su2TtVector<T>>,
    base_r: &Arc<Su2TtVector<T>>,
    lambda: f64,
    ws: &[Su2OneSiteWorkspace],
) -> (Vec<Su2Sketch<T>>, Vec<Su2Sketch<T>>, Su2TtTangent<T>) {
    let (pz_hi, qz_hi, mut grad) = hz_stacks_and_projection(h, base_l, base_r, base_l, ws);
    let px = Su2TtTangent::project_shared(base_l, base_r, base_l);
    grad.axpy(T::from_f64(-lambda), &px);
    (pz_hi, qz_hi, grad)
}

pub(crate) fn project_hmul_env<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    base_l: &Arc<Su2TtVector<T>>,
    base_r: &Arc<Su2TtVector<T>>,
    lambda: f64,
    w: &Su2TtVector<T>,
) -> Su2TtTangent<T> {
    use crate::su2::hamiltonian::matvec::su2_onesite_workspace;
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "project_hmul_env: frame mismatch"
    );
    let d = base_l.space.d;
    let ws: Vec<Su2OneSiteWorkspace> = (0..d)
        .map(|k| su2_onesite_workspace(h, w.core(k)))
        .collect();
    let mut out = hz_stacks_and_projection(h, base_l, base_r, w, &ws).2;
    if lambda != 0.0 {
        out.axpy(
            T::from_f64(-lambda),
            &Su2TtTangent::project_shared(base_l, base_r, w),
        );
    }
    out
}

pub(crate) fn project_kappa_hmul_env<T: Scalar>(
    op: &Su2SparseHamiltonian<T>,
    h: &Su2SparseHamiltonian<T>,
    base_l: &Arc<Su2TtVector<T>>,
    base_r: &Arc<Su2TtVector<T>>,
    ket: &Su2TtVector<T>,
) -> Su2TtTangent<T> {
    use crate::su2::hamiltonian::matvec::su2_onesite_workspace;
    use crate::su2::weingarten::gauge_complement_mut;
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "project_kappa_hmul_env: frame mismatch"
    );
    let space = base_l.space;
    let d = space.d;

    let mut er: Vec<Option<Su2Sketch<T>>> = (0..=d).map(|_| None).collect();
    er[d] = Some(Su2Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let yk = h.build_core(ket.core(k));
        let ws = su2_onesite_workspace(op, &yk);
        er[k] = Some(op.sketching_step_left_mixed(
            base_r.core(k),
            &yk,
            er[k + 1].as_ref().expect("right sweep runs rightward"),
            &ws,
        ));
    }

    let mut el = Su2Sketch::identity(space, 0, 1);
    let mut comps: Vec<Su2Core<T>> = Vec::with_capacity(d);
    for k in 0..d {
        let yk = h.build_core(ket.core(k));
        let ws = su2_onesite_workspace(op, &yk);
        debug_assert_eq!(
            el.site(),
            k,
            "project_kappa_hmul_env: left env bond mismatch"
        );
        let mut comp = op.sketched_hamiltonian(
            &yk,
            &el,
            er[k + 1].as_ref().expect("right env present at bond k+1"),
            &ws,
        );
        if k + 1 < d {
            gauge_complement_mut(&mut comp, base_l.core(k));
            el = op.sketching_step_right_mixed(base_l.core(k), &yk, &el, &ws);
        }
        comps.push(comp);
    }

    Su2TtTangent::from_parts(base_l.ranks.clone(), base_l.clone(), base_r.clone(), comps)
}

/// The SU(2) Riemannian gradient `P(Hx − λx)`, contracted through the operator — DETERMINISTIC
/// and `H·x`-free. Twin of `crate::solvers::hess::riemannian_gradient_env`.
pub fn riemannian_gradient_env<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    lambda: f64,
) -> Su2TtTangent<T> {
    use crate::su2::hamiltonian::matvec::su2_onesite_workspace;
    let d = base_l.space.d;
    let ws: Vec<Su2OneSiteWorkspace> = (0..d)
        .map(|k| su2_onesite_workspace(h, base_l.core(k)))
        .collect();
    let (bl, br) = (Arc::new(base_l.clone()), Arc::new(base_r.clone()));
    hz_stacks_and_gradient(h, &bl, &br, lambda, &ws).2
}

/// Bind the SU(2) Weingarten operator at `x`.
pub fn weingarten_operator<'h, T: Scalar>(
    h: &'h Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Su2WeingartenCtx<'h, T> {
    let (base_l, base_r) = tangent_bases(x);
    let lambda = rayleigh_quotient(h, x).re_s();
    Su2WeingartenCtx {
        h,
        base_l: Arc::new(base_l),
        base_r: Arc::new(base_r),
        lambda,
        floor: FloorPolicy::default_hard(),
        curvature: CurvatureReg::Off,
        curv_op: std::sync::OnceLock::new(),
        gauges: std::sync::OnceLock::new(),
        gauges_curv: std::sync::OnceLock::new(),
        gauges_damped: std::sync::OnceLock::new(),
        envs: std::sync::OnceLock::new(),
        z_explicit: std::sync::OnceLock::new(),
    }
}

impl<'h, T: Scalar> Su2WeingartenCtx<'h, T> {
    pub(crate) fn from_parts(
        h: &'h Su2SparseHamiltonian<T>,
        base_l: impl Into<Arc<Su2TtVector<T>>>,
        base_r: impl Into<Arc<Su2TtVector<T>>>,
        lambda: f64,
    ) -> Self {
        Su2WeingartenCtx {
            h,
            base_l: base_l.into(),
            base_r: base_r.into(),
            lambda,
            floor: FloorPolicy::default_hard(),
            curvature: CurvatureReg::Off,
            curv_op: std::sync::OnceLock::new(),
            gauges: std::sync::OnceLock::new(),
            gauges_curv: std::sync::OnceLock::new(),
            gauges_damped: std::sync::OnceLock::new(),
            envs: std::sync::OnceLock::new(),
            z_explicit: std::sync::OnceLock::new(),
        }
    }

    /// Override the Σ⁻¹ floor policy. Drops the cached bond gauges — the floor is baked into
    /// their `R⁻¹` — and the `Tikhonov` curvature apply's own gauge set with them (a rebuild only:
    /// that set never reads this floor, below). The term-A environments are NOT dropped (pure
    /// `⟨base|H|base⟩` folds, no gauge or floor dependence), and neither is the σ-damped gauge set
    /// of [`Self::bond_gauges_damped`] (built at its own scale, with no dependence on this
    /// `floor`), nor the [`CurvatureReg`] operator (it reads the SVD of `R_b` itself).
    pub fn with_floor(mut self, floor: FloorPolicy) -> Self {
        self.floor = floor;
        self.gauges = std::sync::OnceLock::new();
        self.gauges_curv = std::sync::OnceLock::new();
        self
    }

    pub(crate) fn warm_b_caches(&self) {
        let _ = self.bond_gauges();
        if self.curvature_op().is_some() && matches!(self.curvature, CurvatureReg::Tikhonov(_)) {
            let _ = self.curvature_gauges();
        }
    }

    pub(crate) fn warm_a_caches(&self) {
        let _ = self.envs();
    }

    fn curvature_op(&self) -> Option<&Su2ColumnDeflation<T>> {
        let reg = self.curvature;
        if reg == CurvatureReg::Off {
            return None;
        }
        self.curv_op
            .get_or_init(|| {
                let p = match reg {
                    CurvatureReg::Hard(f) => Su2ColumnDeflation::from_gauges(self.bond_gauges(), f),
                    CurvatureReg::Tikhonov(e) => {
                        Su2ColumnDeflation::from_gauges_weighted(self.bond_gauges(), |s| {
                            s * s / (s * s + e * e)
                        })
                    }
                    CurvatureReg::Off => unreachable!("returned above"),
                };
                (!p.is_identity()).then_some(p)
            })
            .as_ref()
    }

    fn envs(&self) -> &Su2HessEnvs<T> {
        self.envs.get_or_init(|| {
            use crate::su2::hamiltonian::matvec::su2_onesite_workspace;
            use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
            let d = self.base_l.space.d;
            let ws: Vec<Su2OneSiteWorkspace> = (0..d)
                .map(|k| su2_onesite_workspace(self.h, self.base_l.core(k)))
                .collect();

            let (pz_hi, qz_hi, grad) =
                hz_stacks_and_gradient(self.h, &self.base_l, &self.base_r, self.lambda, &ws);

            let z_lo = &(&*self.base_l * T::from_f64(-self.lambda)) - &grad.to_vector();
            let zlo_src = Su2ExplicitZ::new(&z_lo);
            let pz_lo = left_env_z(&self.base_l, &zlo_src);
            let qz_lo = right_env_z(&self.base_r, &zlo_src);

            Su2HessEnvs {
                l: left_to_right_sketching(self.h, &self.base_l),
                r: right_to_left_sketching(self.h, &self.base_r),
                ws,
                pz_hi,
                qz_hi,
                z_lo,
                pz_lo,
                qz_lo,
                grad,
            }
        })
    }

    /// The Riemannian gradient `P(Hx − λx)` at the base point. A by-product of the
    /// normal-residual split, so it is FREE once the context's environments exist — and, like
    /// everything else on this path, it never materializes `H·x`. Callers that want a gradient norm
    /// at a point (convergence checks, criticality diagnostics) should bind a context and read this
    /// rather than doing `project(base_l, base_r, &(H·x − λx))`, which pays the exact untruncated
    /// product for a scalar.
    pub fn gradient(&self) -> &Su2TtTangent<T> {
        &self.envs().grad
    }

    /// The per-bond gauges `{R_b, Σ_b, R_b⁻¹}` at the base point (interior bonds `1..d`), built
    /// lazily and shared across applies — Schmidt-spectrum diagnostics and the curvature-aware
    /// preconditioner read `sigma`/`dropped_mass` from here. Built at [`Self::with_floor`]'s policy,
    /// raised to `f` under `CurvatureReg::Hard(f)` (`dropped_mass` then includes the cut).
    pub fn bond_gauges(&self) -> &[Su2BondGauge<T>] {
        self.gauges
            .get_or_init(|| bond_gauges(&self.base_l, &self.base_r, self.inverse_floor()))
    }

    fn inverse_floor(&self) -> FloorPolicy {
        match (self.floor, self.curvature) {
            (FloorPolicy::Hard(g), CurvatureReg::Hard(f)) => FloorPolicy::Hard(g.max(f)),
            (floor, _) => floor,
        }
    }

    fn curvature_gauges(&self) -> &[Su2BondGauge<T>] {
        let CurvatureReg::Tikhonov(eps) = self.curvature else {
            panic!(
                "curvature_gauges: the damped curvature gauge set exists only under \
                 CurvatureReg::Tikhonov (this context is {:?})",
                self.curvature
            )
        };
        self.gauges_curv
            .get_or_init(|| bond_gauges(&self.base_l, &self.base_r, FloorPolicy::Tikhonov(eps)))
    }

    /// The SECOND gauge set: the same base-point bond gauges rebuilt under CENTERWEIGHTED
    /// Tikhonov damping at scale `s` ([`bond_gauges_tikhonov_cw`], whose rustdoc carries the
    /// raw-vs-centerweighted argument — the one place `s` does NOT transfer verbatim from the
    /// abelian layer). [`Self::bond_gauges`] is left at this context's own `floor`, so the
    /// spectral consumers sharing the context keep the undamped operator.
    pub fn bond_gauges_damped(&self, s: f64) -> &[Su2BondGauge<T>] {
        let (s0, g) = self
            .gauges_damped
            .get_or_init(|| (s, bond_gauges_tikhonov_cw(&self.base_l, &self.base_r, s)));
        assert!(
            *s0 == s,
            "bond_gauges_damped: this context cached the damped gauge set at s = {s0:e} and was \
             asked for s = {s:e} — the Tikhonov scale is base-point data (one outer iteration, \
             one s); rebuild the context to change it"
        );
        g
    }

    /// The gauge frames the context was bound at (build `crate::su2::weingarten::tangent_basis`
    /// against these).
    pub fn bases(&self) -> (&Su2TtVector<T>, &Su2TtVector<T>) {
        (&self.base_l, &self.base_r)
    }

    pub(crate) fn shared_bases(&self) -> (&Arc<Su2TtVector<T>>, &Arc<Su2TtVector<T>>) {
        (&self.base_l, &self.base_r)
    }

    /// The effective-Hamiltonian term `A·ξ`, contracted as fused environment sweeps — **`H·ι(ξ)`
    /// is never formed**. Twin of the abelian `WeingartenCtx::a_apply`; see its docs for the
    /// derivation, the cost argument (the exact-product order is quadratic in the operator band
    /// width `D_H`, this one is linear), and the `j<k / j=k / j>k` split.
    pub fn a_apply(&self, xi: &Su2TtTangent<T>) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::a_apply");
        if crate::knobs::A_APPLY_EXACT.get() {
            return self.a_apply_exact(xi);
        }
        self.a_apply_fused(xi)
    }

    /// The unfused reference term-A apply, [`effham_apply`] at this context's frames and `λ` —
    /// what `QNTT_A_APPLY_EXACT` routes [`Self::a_apply`] to. Refuses a SHIFTED tangent, as
    /// [`Self::a_apply`] does.
    pub fn a_apply_exact(&self, xi: &Su2TtTangent<T>) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::a_apply_exact");
        effham_apply(self.h, &self.base_l, &self.base_r, self.lambda, xi)
    }

    /// The fused environment-sweep term-A apply, bypassing the `QNTT_A_APPLY_EXACT` switch — the
    /// entry point the equivalence test pins against [`effham_apply`]. See [`Self::a_apply`];
    /// refuses a SHIFTED tangent, as that does.
    pub fn a_apply_fused(&self, xi: &Su2TtTangent<T>) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::a_apply_fused");
        let e = self.envs();
        let ws = &e.ws[..];
        let components = crate::su2::weingarten::Su2FusedTermA {
            h: self.h,
            u: &self.base_l,
            v: &self.base_r,
            l: &e.l,
            r: &e.r,
            ws_u: ws,
            ws_v: ws,
            ws_du: ws,
        }
        .components(&xi.components);

        let mut out = Su2TtTangent::from_parts(
            self.base_l.ranks.clone(),
            self.base_l.clone(),
            self.base_r.clone(),
            components,
        );
        out.axpy(T::from_f64(-self.lambda), xi);
        out
    }

    /// Approximate coefficient diagonal of the term-A apply (twin of the abelian
    /// `WeingartenCtx::a_diagonal`): per-component ALS-style effective diagonals in the tangent
    /// gauge frames, minus `λ`. **NO θ, NO centerweight** — the metric centerweight cancels in
    /// the Rayleigh-quotient diagonal (`su2/hamiltonian/matvec.rs` S5a verdict, oracle-gated),
    /// exactly as in the SU(2) ALS Davidson preconditioner. Approximate away from `d = 1`
    /// (ignores the gauge projector); preconditioner-grade only.
    pub fn a_diagonal(&self) -> Su2TtTangent<T> {
        let d = self.base_l.space.d;
        let e = self.envs();
        let mut components = Vec::with_capacity(d);
        for k in 0..d {
            components.push(self.h.effective_diagonal(
                &e.l[k],
                &e.r[k],
                self.base_l.core(k),
                &e.ws[k],
            ));
        }
        let mut diag = Su2TtTangent::from_parts(
            self.base_l.ranks.clone(),
            self.base_l.clone(),
            self.base_r.clone(),
            components,
        );
        let shift = T::from_f64(-self.lambda);
        diag.for_each_coeff_mut(|c| *c += shift);
        diag
    }

    /// The Weingarten curvature term `B·ξ`, assembled against the cached bond gauges and the
    /// SPLIT normal residual — `H·x` is never materialized. Twin of the abelian `b_apply`.
    pub fn b_apply(&self, xi: &Su2TtTangent<T>) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::b_apply");
        self.b_apply_any(xi, None)
    }

    pub(crate) fn b_apply_any(
        &self,
        xi: &Su2TtTangent<T>,
        sf: Option<&Su2ShiftedFrames<T>>,
    ) -> Su2TtTangent<T> {
        let Some(d) = self.curvature_op() else {
            return self.b_apply_raw(xi, sf);
        };
        let mut v = xi.clone();
        d.apply(&mut v);
        if matches!(self.curvature, CurvatureReg::Tikhonov(_)) {
            let g = self.curvature_gauges();
            let mut first = self.b_assemble(xi, sf, g, Terms::S1_S2);
            d.apply(&mut first);
            let second = self.b_assemble(&v, sf, g, Terms::S3);
            first.add(&second)
        } else {
            let mut b = self.b_apply_raw(&v, sf);
            d.apply(&mut b);
            b
        }
    }

    fn b_apply_raw(
        &self,
        xi: &Su2TtTangent<T>,
        sf: Option<&Su2ShiftedFrames<T>>,
    ) -> Su2TtTangent<T> {
        self.b_assemble(xi, sf, self.bond_gauges(), Terms::all())
    }

    fn b_assemble(
        &self,
        xi: &Su2TtTangent<T>,
        sf: Option<&Su2ShiftedFrames<T>>,
        gauges: &[Su2BondGauge<T>],
        terms: Terms,
    ) -> Su2TtTangent<T> {
        if crate::knobs::B_APPLY_EXPLICIT.read() {
            let z = self.z_explicit.get_or_init(|| {
                normal_residual(
                    self.h,
                    &self.base_l,
                    &self.base_r,
                    &self.base_l,
                    self.lambda,
                )
            });
            return weingarten_assemble_cached(&self.base_l, &self.base_r, z, xi, gauges, terms);
        }
        let e = self.envs();
        let zhi = Su2ImplicitHZ {
            h: self.h,
            ket: &self.base_l,
            ws: &e.ws,
            envs: Some((&e.pz_hi, &e.qz_hi)),
        };
        let zlo = Su2ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
        weingarten_assemble_sources_dressed(
            &self.base_l,
            &self.base_r,
            &[&zhi, &zlo],
            xi,
            gauges,
            terms,
            sf,
        )
    }

    /// [`Self::b_apply`] against the σ-DAMPED gauge set — the twin of the abelian
    /// `WeingartenCtx::b_apply_damped`, which carries the measured story; the SU(2) delta is that
    /// the weight is taken in CENTERWEIGHTED Schmidt units ([`bond_gauges_tikhonov_cw`]) so the
    /// same numeric `s` means the same physical threshold on both layers. Everything else — the
    /// split normal residual, the cached environments, the assembly kernel — is
    /// [`Self::b_apply`]'s, unchanged.
    pub fn b_apply_damped(&self, xi: &Su2TtTangent<T>, s: f64) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::b_apply_damped");
        if crate::knobs::B_APPLY_EXPLICIT.read() {
            let z = self.z_explicit.get_or_init(|| {
                normal_residual(
                    self.h,
                    &self.base_l,
                    &self.base_r,
                    &self.base_l,
                    self.lambda,
                )
            });
            return weingarten_assemble_cached(
                &self.base_l,
                &self.base_r,
                z,
                xi,
                self.bond_gauges_damped(s),
                Terms::all(),
            );
        }
        let e = self.envs();
        let zhi = Su2ImplicitHZ {
            h: self.h,
            ket: &self.base_l,
            ws: &e.ws,
            envs: Some((&e.pz_hi, &e.qz_hi)),
        };
        let zlo = Su2ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
        weingarten_assemble_sources(
            &self.base_l,
            &self.base_r,
            &[&zhi, &zlo],
            xi,
            self.bond_gauges_damped(s),
            Terms::all(),
        )
    }

    /// Whole Riemannian Hessian `H^rr = A·ξ + B·ξ` (real layer). Refuses a SHIFTED tangent — see
    /// [`Self::a_apply`]; the refusal is spelled here as well as inside the two halves so the
    /// message names the entry point the caller actually used.
    pub fn hess_apply(&self, xi: &Su2TtTangent<T>) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::hess_apply");
        self.a_apply(xi).add(&self.b_apply(xi))
    }

    /// `H^rr = A + B` (alias of [`Self::hess_apply`]). Refuses a SHIFTED tangent.
    pub fn hpp_apply(&self, xi: &Su2TtTangent<T>) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::hpp_apply");
        self.hess_apply(xi)
    }

    /// `H^ii = A − B`. Refuses a SHIFTED tangent.
    pub fn hmm_apply(&self, xi: &Su2TtTangent<T>) -> Su2TtTangent<T> {
        xi.refuse_if_shifted("Su2WeingartenCtx::hmm_apply");
        self.a_apply(xi).sub(&self.b_apply(xi))
    }
}

impl<'h> Su2WeingartenCtx<'h, f64> {
    /// Regularize this context's curvature term (`Off` restores the exact operator) — the twin of
    /// the abelian `WeingartenCtx::with_curvature`; [`CurvatureReg`] carries the contract.
    ///
    /// # Panics
    ///
    /// Unless the scale of a `Hard`/`Tikhonov` request is finite and `> 0` (the message names the
    /// value).
    pub fn with_curvature(mut self, reg: CurvatureReg) -> Self {
        reg.assert_valid();
        self.curvature = reg;
        self.curv_op = std::sync::OnceLock::new();
        self.gauges = std::sync::OnceLock::new();
        self.gauges_curv = std::sync::OnceLock::new();
        self
    }

    /// The curvature regularization this context applies (`Off` unless
    /// [`Self::with_curvature`] set one).
    pub fn curvature(&self) -> CurvatureReg {
        self.curvature
    }
}

impl_gad!(
    gad,
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
    Su2TtTangent<f64>,
    Su2WeingartenCtx,
    rayleigh_quotient,
    tangent_bases,
    crate::su2::riemannian::normalize_mut,
    crate::su2::hess::riemannian_gradient_env,
    crate::su2::weingarten::tangent_dim,
    crate::su2::weingarten::min_schmidt,
    1
);

#[cfg(test)]
mod h7_tests {
    use crate::scalar::Scalar;
    use crate::solvers::InnerProductVector;
    use crate::solvers::davidson::effective_diagonal_bruteforce;
    use crate::su2::core::Su2Core;
    use crate::su2::hamiltonian::matvec::su2_onesite_workspace;
    use crate::su2::hamiltonian::rayleigh_quotient;
    use crate::su2::hess::weingarten_operator;
    use crate::su2::qn::Su2Space;
    use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{generic_su2_h, randomized_su2_state};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn h7_tangent_diagonal_matches_bruteforce() {
        let d = 5usize;
        let space = Su2Space::new(d, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71B0);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        x.scale_mut(1.0 / x.norm());
        let h = generic_su2_h::<f64>(space);
        let lambda = rayleigh_quotient(&h, &x).re_s();
        let ctx = weingarten_operator(&h, &x);
        let diag = ctx.a_diagonal();
        let (base_l, base_r) = ctx.bases();
        let fl = left_to_right_sketching(&h, base_l);
        let fr = right_to_left_sketching(&h, base_r);
        let mut nonvacuous = 0usize;
        for k in 0..d {
            let ws = su2_onesite_workspace(&h, base_l.core(k));
            let mut op = |a: &Su2Core<f64>| h.sketched_hamiltonian(a, &fl[k], &fr[k], &ws);
            let o_diag = effective_diagonal_bruteforce(&mut op, base_l.core(k));
            if o_diag.norm() > 1e-6 {
                nonvacuous += 1;
            }
            let comp = diag.component(k);
            assert_eq!(comp.num_coeffs(), o_diag.num_coeffs(), "shape at k={k}");
            for i in 0..comp.num_coeffs() {
                let want = o_diag.coeff(i) - lambda;
                let got = comp.coeff(i);
                assert!(
                    (got - want).abs() < 1e-10 * (1.0 + want.abs()),
                    "diagonal coeff {i} at k={k}: got {got}, want {want}"
                );
            }
        }
        assert!(nonvacuous >= d - 1, "diagonal test mostly vacuous");
    }
}

#[cfg(test)]
mod signed_davidson_tests {}
