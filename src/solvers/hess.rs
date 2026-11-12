// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Riemannian Hessian of the energy on the fixed-TT-rank manifold: the
//! effective-Hamiltonian (Gauss–Newton) term `A = P_{T_xM}(H−λ)P_{T_xM}` and its tangent-space
//! spectrum. Conventions are paper-aligned to arXiv:2506.16420; the Weingarten curvature term
//! `B` and the symplectic/Williamson eigenvalues live in the submodules below.

use crate::core::SparseCore;
use crate::core::deflate::ColumnDeflation;
use crate::core::factor::Sketch;
use crate::core::factor::mul_sketch_left;
use crate::core::tangent::TtTangent;
use crate::core::weingarten::{
    BondGauge, ExplicitZ, FloorPolicy, Terms, ZEnvs, ZSource, bond_gauges, gauge_complement_mut,
    left_env_z, right_env_z, right_env_z_visit, weingarten_assemble_cached,
    weingarten_assemble_sources,
};
use crate::hamiltonian::matvec::{OneSiteWorkspace, onesite_workspace};
use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
use crate::pointgroup::Irrep;
use crate::qn::RankGrid;
use crate::scalar::Scalar;

use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::solvers::riemannian::tangent_bases;
use crate::vector::TtVector;
use std::sync::Arc;

mod channel;
mod cone;
#[cfg(test)]
pub(crate) mod curvature_tests;
mod davidson;
mod davidson_paired;
mod davidson_signed;
mod drivers;
mod lanczos;
mod lobpcg;
mod mode_check;
mod modes;
mod newton_polish;
mod padding;
mod sector;
mod sector_all;
mod sector_response;
#[cfg(test)]
mod sector_response_tests;
mod seed_search;
mod symplectic_dense;
mod tda;

pub use channel::{
    ChannelGsOpts, ChannelLabel, ChannelOpts, ChannelResult, ChannelSet, ChannelSpectrum,
    ChannelStatus, IrrepSel, Level, SpinSel, channel_ground_state, channel_response,
    channel_response_from_load,
};
pub(crate) use channel::{
    ChannelPlan, START_PERTURBATION, channel_plan, impl_channel, merge_levels, start_rank,
};
pub(crate) use cone::ISO_TOL;
pub(crate) use cone::drift_voids_record;
pub use cone::{
    ConeGuards, ConeOpts, ConeRefusal, ConeResponse, ConeStageStats, MatfreeOpts, MatfreeStats,
    cone_response, cone_response_matfree, cone_response_matfree_with_stats,
    cone_response_with_stats, try_cone_response,
};
pub(crate) use cone::{SolveMode, impl_cone, mixed_lanczos};
#[cfg(test)]
pub(crate) use davidson::omega_davidson_vectors;
pub(crate) use davidson::{REFUSE_TAU, omega_davidson, vector_budget};
pub use davidson::{RpaKernel, SympPrecond, SymplecticDavidsonOpts};
#[cfg(test)]
pub(crate) use davidson_paired::tests::assert_paired_wanted;
/// The paired-basis RPA Davidson — one Euclidean basis for both halves of `(X, Y)`, no block as
/// a metric; the default kernel of the signed entry points ([`RpaKernel`]).
pub use davidson_paired::{
    AnomalyRule, PairedDavidsonOpts, PairedDavidsonOut, PairedHook, PairedInfo, PairedMode,
    PairedOffset, PairedPrecond, WantedRank, paired_lowest_davidson,
};
pub(crate) use davidson_paired::{
    omega_davidson_paired, paired_image, paired_mu, paired_precond, realified,
};
pub(crate) use davidson_signed::omega_davidson_signed;
pub use drivers::{
    PassedOver, RpaAmplitudes, RpaMetric, RpaMode, RpaModeKind, RpaReadout, RpaReport,
    select_rpa_metric, symplectic_lowest_lobp4dcg, symplectic_lowest_signed_davidson,
    symplectic_lowest_signed_davidson_c64, symplectic_lowest_signed_davidson_targeted,
    unshifted_precond_inverse_diagnostic,
};
pub(crate) use drivers::{
    a_diagonal_squared, build_symp_precond, impl_symplectic_drivers, omega_from_signed,
    paired_route, readout_from_paired, readout_from_signed,
};
pub(crate) use lanczos::certify_negatives;
pub use lanczos::{
    ComplexRitz, IndefiniteReport, MetricEvent, NegativeWitness, SympDiagnostics,
    SymplecticLowestError, SymplecticLowestInfo,
};
pub(crate) use lobpcg::omega_lobp4dcg;
pub use lobpcg::{BasisSelection, SympLobpcgOpts};
pub use mode_check::{ModeCheck, ModeCheckOpts, rpa_mode_checks};
pub(crate) use mode_check::{ModeCheckOps, TanOp, mode_checks_core};
pub use modes::{ModeOpts, ModeResult, RitzPair, lowest_hessian_mode};
pub use newton_polish::{
    HessPart, NewtonPolishInfo, NewtonPolishOpts, PROBE_BLOCK_SEED, PdVerdict, ProbeDavidson,
    hessian_pd_probe, hessian_pd_probe_c64, hessian_pd_probe_c64_witness, hessian_pd_probe_witness,
    hessian_pd_probe_witness_with, newton_polish,
};
pub(crate) use newton_polish::{
    PcgFlag, VecScalar, add_complement_shift, clean_tangent, complement_shift, deflate_radial,
    deflate_subspace, diag_inf_scale, impl_hess_probes, impl_newton_polish, lowest_hessian_mode_at,
    pcg_solve, pd_probe_kernel, pd_probe_kernel_with, radial_mode, reproject_tangent,
};
pub use padding::{PadOpts, Padded, pad_singles, padded_context, singles_completeness};
pub(crate) use padding::{pad_weights, singles_pairs};
pub use sector::{SectorCtx, SectorOpts, SectorReadout, SectorResponse, SectorSpectrum};
pub(crate) use sector::{
    SectorPencil, check_sector_kernel, resolve_sector_precond, sector_d0_precond, sector_rpa_solve,
};
pub use sector_all::{
    BWeight, SectorAllOpts, SectorAllReadout, SectorChannel, SectorChannelStatus, SectorLevel,
    SectorObservables, SectorReference, SectorSpectrumAll, sector_response_all,
    sector_response_all_from_load,
};
pub use sector_response::{
    SectorDynamicOpts, SectorDynamicResponse, SectorDynamicSolveInfo, SectorMomentOpts, SectorPair,
    SectorResponseError, SectorSolveInfo, SectorStaticOpts, SectorStaticResponse,
    SectorTransitionMoments, property_flux,
};
pub use seed_search::{SeedSearch, SeedSearchOpts, geometric_seed_search};
pub(crate) use symplectic_dense::impl_hessian_grams;
pub use symplectic_dense::{
    SymplecticSpectrum, effham_lowest, hessian_grams, hessian_grams_c64, hessian_grams_owned,
    hll_from_grams_c64, symplectic_eigenvalues, symplectic_eigenvalues_c64, symplectic_from_grams,
    symplectic_from_grams_c64,
};
pub use tda::{TdaDavidsonOpts, TdaInfo, TdaModes, TdaPrecond, tda_lowest_davidson};
pub(crate) use tda::{TdaHooks, tda_lowest_core};

/// `A·ξ = P_{T_xM}((H − λ)·ι(ξ))`, the effective-Hamiltonian term (A) of the Riemannian Hessian
/// (H0 §1.3, §3). `ι = TtTangent::to_vector` is the embedding; `P = TtTangent::project` the tangent
/// projection at the gauge frames `base_l`/`base_r`; `λ` the Rayleigh quotient at `x`.
pub fn effham_apply<T: Scalar>(
    h: &SparseHamiltonian<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    lambda: f64,
    xi: &TtTangent<T>,
) -> TtTangent<T> {
    let w = xi.to_vector();
    let hw = h.apply(&w);
    let mut out = TtTangent::project(base_l, base_r, &hw);
    out.axpy(T::from_f64(-lambda), xi);
    out
}

/// `A^φ·ξ = P^φ((H_out − λ)·ι^φ(ξ))`, the φ-block of the effective-Hamiltonian term of the
/// Riemannian Hessian on the point-group-UNTYPED manifold through `x` — the directions whose
/// varied core carries irrep flux `φ`, so that `ι^φ(ξ)` lives in irrep `Γ0 ⊕ φ`. That Hessian is
/// block-diagonal in `φ` (the ambient term vanishes across irreps and the curvature term couples
/// only equal fluxes), so this apply is the whole `A` content of one response channel.
pub fn effham_apply_shifted<T: Scalar>(
    h_out: &SparseHamiltonian<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    phi: Irrep,
    lambda: f64,
    xi: &TtTangent<T>,
) -> TtTangent<T> {
    assert_shifted_operator(h_out, base_l, phi);
    assert_eq!(
        xi.phi(),
        phi,
        "effham_apply_shifted: the direction carries flux {:?} but the channel is {phi:?}",
        xi.phi()
    );
    let hw = h_out.apply(&xi.to_vector());
    let mut out = TtTangent::project_shifted(base_l, base_r, phi, &hw);
    out.axpy(T::from_f64(-lambda), xi);
    out
}

/// The dense Gram `G_ij = ⟨b_i, A^φ b_j⟩` over a `⟨·,·⟩`-orthonormal basis of `T^φ`, row-major
/// `n×n` — the REAL layer only (see below). Its eigenvalues are the TDA-like readout of the
/// channel, and a NEGATIVE one says the channel holds a state below `λ` — information about the
/// reference, not a solver failure. They are Ritz values of `H|_{Γ0 ⊕ φ} − λ` compressed to
/// `T^φ`, which is generally a PROPER subspace of that sector (and can be empty) — see
/// [`effham_apply_shifted`], "`T^φ` is generally a PROPER subspace", for what may and may not be
/// quoted from them.
pub fn effham_gram_shifted(
    h_out: &SparseHamiltonian<f64>,
    base_l: &TtVector<f64>,
    base_r: &TtVector<f64>,
    phi: Irrep,
    lambda: f64,
    basis: &[TtTangent<f64>],
) -> Vec<f64> {
    assert_shifted_operator(h_out, base_l, phi);
    for (i, b) in basis.iter().enumerate() {
        assert_eq!(
            b.phi(),
            phi,
            "effham_gram_shifted: basis vector {i} carries flux {:?} but the channel is {phi:?} —              a Gram mixing fluxes is not the channel's operator",
            b.phi()
        );
    }
    let n = basis.len();
    let pure: Vec<TtTangent<f64>> = basis
        .iter()
        .map(|u| TtTangent::project_shifted(base_l, base_r, phi, &u.to_vector()))
        .collect();
    let mut a = vec![0.0f64; n * n];
    for (j, uj) in pure.iter().enumerate() {
        let aj = effham_apply_shifted(h_out, base_l, base_r, phi, lambda, uj);
        for (i, ui) in pure.iter().enumerate() {
            a[i * n + j] = ui.dot(&aj);
        }
    }
    a
}

pub(crate) fn assert_shifted_operator<T: Scalar>(
    h_out: &SparseHamiltonian<T>,
    base_l: &TtVector<T>,
    phi: Irrep,
) {
    let space = base_l.space;
    let space_out = space.with_target(space.target.fuse(phi));
    assert_eq!(
        h_out.space, space_out,
        "the shifted Hamiltonian must be BUILT on space_out = {space_out:?} (the automaton reads \
         `target` for its caps and its bond-d collapse, so a re-stamp is not enough); got {:?}",
        h_out.space
    );
}

/// The **normal residual** `r⊥ = P⊥_{T_xM}(Hx − λx)` as a full TT vector — the object the H2
/// Weingarten term (B) contracts against (H0 §2.2/§3, Psenka–Boumal Eq. 8 term 2 depends only on
/// the normal part of the ambient gradient `∂f̄ = Hx`). Computed as `full − ι(P(full))` with
/// `full = Hx − λx`: subtract the tangent projection of the residual, leaving its normal part.
pub fn normal_residual<T: Scalar>(
    h: &SparseHamiltonian<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    x: &TtVector<T>,
    lambda: f64,
) -> TtVector<T> {
    let hx = h.apply(x);
    let full = &hx - &(x * T::from_f64(lambda));
    let tang = TtTangent::project(base_l, base_r, &full).to_vector();
    &full - &tang
}

/// Bind the effective-Hamiltonian operator at `x`: returns the gauge frames, `λ`, and a closure
/// `ξ ↦ A·ξ` suitable for [`crate::solvers::davidson`] or the H3 symplectic driver. Kept separate
/// from [`effham_lowest`] so the same operator can be reused for block/deflated solves and, later,
/// composed with the term-B apply into `A ± B`.
pub fn effham_operator<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
) -> (TtVector<T>, TtVector<T>, f64) {
    let (base_l, base_r) = tangent_bases(x);
    let lambda = rayleigh_quotient(h, x).re_s();
    (base_l, base_r, lambda)
}

/// Symmetric regularization of the Weingarten curvature term `B` near the manifold border,
/// applied as a congruence so `B` stays symmetric; `A`, the tangent space, `λ` and the point
/// are untouched.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CurvatureReg {
    /// The exact Riemannian Hessian wherever the context's Σ⁻¹ floor cuts nothing — the plain
    /// assembly with no extra float operation, equal to the earlier one-sided assembly to
    /// roundoff. Where the floor cuts a Schmidt value (the default `1e-12` on a padded
    /// grid, a raised [`WeingartenCtx::with_floor`]) it is the floored `B = S1·P_g + P_g·S1ᴴ`
    /// (`P_g` the floor's column projector), made symmetric — an `O(‖S3‖)` change on
    /// those directions.
    #[default]
    Off,
    /// `B ↦ P·B·P`, `P` the orthogonal column projector off the right Schmidt vectors of every
    /// bond gauge `R_b` with Schmidt value `< f`.
    Hard(f64),
    /// `B ↦ D·B·D`, `D = V diag(w) Vᴴ` per bond sector, `w = s²/(s² + ε²)`.
    Tikhonov(f64),
}

impl CurvatureReg {
    pub(crate) fn assert_valid(self) {
        let v = match self {
            CurvatureReg::Off => return,
            CurvatureReg::Hard(v) | CurvatureReg::Tikhonov(v) => v,
        };
        assert!(
            v.is_finite() && v > 0.0,
            "CurvatureReg: {self:?} has scale {v}, which must be finite and > 0 — \
             CurvatureReg::Off is the only spelling of \"no regularization\""
        );
    }
}

/// Precomputed context for the Weingarten term `B` at `x` (H2): the gauge frames, `λ`, the normal
/// residual `r⊥ = Z`, and the Σ⁻¹ floor policy. Bind once, then apply to many `ξ`. Kept separate
/// from [`effham_operator`] so `A·ξ` (linear/Hermitian) and `B·ξ` (antilinear/symmetric) stay
/// distinct applies — a real Hessian-vector product returns only `A+B`, so `A−B` would otherwise be
/// unrecoverable.
pub struct WeingartenCtx<'h, T: Scalar> {
    h: &'h SparseHamiltonian<T>,
    base_l: Arc<TtVector<T>>,
    base_r: Arc<TtVector<T>>,
    lambda: f64,
    floor: FloorPolicy,
    curvature: CurvatureReg,
    curv_op: std::sync::OnceLock<Option<ColumnDeflation<T>>>,
    gauges: std::sync::OnceLock<Vec<BondGauge<T>>>,
    gauges_curv: std::sync::OnceLock<Vec<BondGauge<T>>>,
    gauges_damped: std::sync::OnceLock<(f64, Vec<BondGauge<T>>)>,
    envs: std::sync::OnceLock<HessEnvs<T>>,
    z_explicit: std::sync::OnceLock<TtVector<T>>,
}

struct ImplicitHZ<'a, T: Scalar> {
    h: &'a SparseHamiltonian<T>,
    ket: &'a TtVector<T>,
    ws: &'a [OneSiteWorkspace],
    envs: Option<ZEnvs<'a, T>>,
}

impl<T: Scalar> ZSource<T> for ImplicitHZ<'_, T> {
    fn left_mul(&self, s: &Sketch<T>, k: usize) -> SparseCore<T> {
        let ket_k = self.ket.core(k);
        let mut y = SparseCore::zeros(ket_k.space, k, &s.row_ranks(), &self.h.col_ranks(ket_k));
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

    fn right_mul(&self, k: usize, s: &Sketch<T>) -> SparseCore<T> {
        let ket_k = self.ket.core(k);
        let mut y = SparseCore::zeros(ket_k.space, k, &self.h.row_ranks(ket_k), &s.col_ranks());
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

    fn bond_ranks(&self, b: usize) -> RankGrid {
        self.ket.ranks[b].clone()
    }

    fn cached_envs(&self) -> Option<ZEnvs<'_, T>> {
        self.envs
    }
}

fn hz_stacks_and_projection<T: Scalar>(
    h: &SparseHamiltonian<T>,
    base_l: &Arc<TtVector<T>>,
    base_r: &Arc<TtVector<T>>,
    ket: &TtVector<T>,
    ws: &[OneSiteWorkspace],
) -> (Vec<Sketch<T>>, Vec<Sketch<T>>, TtTangent<T>) {
    let d = base_l.space.d;
    let zhi = ImplicitHZ {
        h,
        ket,
        ws,
        envs: None,
    };
    let pz_hi = left_env_z(base_l, &zhi);

    let mut slots: Vec<Option<SparseCore<T>>> = (0..d).map(|_| None).collect();
    let assemble = |k: usize, dq: &SparseCore<T>| -> SparseCore<T> {
        let mut c = mul_sketch_left(&pz_hi[k], dq);
        if k + 1 < d {
            gauge_complement_mut(&mut c, base_l.core(k));
        }
        c
    };
    let qz_hi = right_env_z_visit(base_r, &zhi, |k, dq| slots[k] = Some(assemble(k, dq)));
    let comps: Vec<SparseCore<T>> = slots.into_iter().map(|c| c.expect("visited")).collect();
    let proj = TtTangent::from_parts(base_l.ranks.clone(), base_l.clone(), base_r.clone(), comps);
    (pz_hi, qz_hi, proj)
}

fn hz_stacks_and_gradient<T: Scalar>(
    h: &SparseHamiltonian<T>,
    base_l: &Arc<TtVector<T>>,
    base_r: &Arc<TtVector<T>>,
    lambda: f64,
    ws: &[OneSiteWorkspace],
) -> (Vec<Sketch<T>>, Vec<Sketch<T>>, TtTangent<T>) {
    let (pz_hi, qz_hi, mut grad) = hz_stacks_and_projection(h, base_l, base_r, base_l, ws);
    let px = TtTangent::project_shared(base_l, base_r, base_l);
    grad.axpy(T::from_f64(-lambda), &px);
    (pz_hi, qz_hi, grad)
}

pub(crate) fn project_hmul_env<T: Scalar>(
    h: &SparseHamiltonian<T>,
    base_l: &Arc<TtVector<T>>,
    base_r: &Arc<TtVector<T>>,
    lambda: f64,
    w: &TtVector<T>,
) -> TtTangent<T> {
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "project_hmul_env: frame mismatch"
    );
    let d = base_l.space.d;
    let ws: Vec<OneSiteWorkspace> = (0..d).map(|k| onesite_workspace(h, w.core(k))).collect();
    let mut out = hz_stacks_and_projection(h, base_l, base_r, w, &ws).2;
    if lambda != 0.0 {
        out.axpy(
            T::from_f64(-lambda),
            &TtTangent::project_shared(base_l, base_r, w),
        );
    }
    out
}

pub(crate) fn project_kappa_hmul_env<T: Scalar>(
    op: &SparseHamiltonian<T>,
    h: &SparseHamiltonian<T>,
    base_l: &Arc<TtVector<T>>,
    base_r: &Arc<TtVector<T>>,
    ket: &TtVector<T>,
) -> TtTangent<T> {
    debug_assert_eq!(
        base_l.ranks, base_r.ranks,
        "project_kappa_hmul_env: frame mismatch"
    );
    let space = base_l.space;
    let d = space.d;

    let mut er: Vec<Option<Sketch<T>>> = (0..=d).map(|_| None).collect();
    er[d] = Some(Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let yk = h.build_core(ket.core(k));
        let ws = onesite_workspace(op, &yk);
        er[k] = Some(op.sketching_step_left_mixed(
            base_r.core(k),
            &yk,
            er[k + 1].as_ref().expect("right sweep runs rightward"),
            &ws,
        ));
    }

    let mut el = Sketch::identity(space, 0, 1);
    let mut comps: Vec<SparseCore<T>> = Vec::with_capacity(d);
    for k in 0..d {
        let yk = h.build_core(ket.core(k));
        let ws = onesite_workspace(op, &yk);
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

    TtTangent::from_parts(base_l.ranks.clone(), base_l.clone(), base_r.clone(), comps)
}

/// The Riemannian gradient `P(Hx − λx)` at the frames `(base_l, base_r)`, contracted through the
/// operator — DETERMINISTIC and `H·x`-free.
pub fn riemannian_gradient_env<T: Scalar>(
    h: &SparseHamiltonian<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    lambda: f64,
) -> TtTangent<T> {
    let d = base_l.space.d;
    let ws: Vec<OneSiteWorkspace> = (0..d)
        .map(|k| onesite_workspace(h, base_l.core(k)))
        .collect();
    let (bl, br) = (Arc::new(base_l.clone()), Arc::new(base_r.clone()));
    hz_stacks_and_gradient(h, &bl, &br, lambda, &ws).2
}

struct HessEnvs<T: Scalar> {
    l: Vec<Sketch<T>>,
    r: Vec<Sketch<T>>,
    ws: Vec<OneSiteWorkspace>,

    pz_hi: Vec<Sketch<T>>,
    qz_hi: Vec<Sketch<T>>,
    z_lo: TtVector<T>,
    pz_lo: Vec<Sketch<T>>,
    qz_lo: Vec<Sketch<T>>,
    grad: TtTangent<T>,
}

/// Bind the Weingarten operator at `x` (§3.1–3.2): gauge frames, `λ`, normal residual `Z = r⊥`, and
/// the default Σ⁻¹ floor. `floor` is exposed via [`WeingartenCtx::with_floor`] for the sensitivity
/// sweep.
pub fn weingarten_operator<'h, T: Scalar>(
    h: &'h SparseHamiltonian<T>,
    x: &TtVector<T>,
) -> WeingartenCtx<'h, T> {
    let (base_l, base_r) = tangent_bases(x);
    let lambda = rayleigh_quotient(h, x).re_s();
    WeingartenCtx {
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

impl<'h, T: Scalar> WeingartenCtx<'h, T> {
    pub(crate) fn from_parts(
        h: &'h SparseHamiltonian<T>,
        base_l: impl Into<Arc<TtVector<T>>>,
        base_r: impl Into<Arc<TtVector<T>>>,
        lambda: f64,
    ) -> Self {
        WeingartenCtx {
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

    /// Override the Σ⁻¹ floor policy (§3.2 sensitivity sweep / H3 conditioning). Drops the cached
    /// bond gauges — the floor is baked into their `R⁻¹` — and the `Tikhonov` curvature apply's own
    /// gauge set with them (a rebuild only: that set never reads this floor, below). The term-A
    /// environments are NOT dropped: they are pure `⟨base|H|base⟩` folds with no gauge or floor
    /// dependence. Neither is the σ-damped gauge set of [`Self::bond_gauges_damped`]: it is built
    /// at its own `Tikhonov(s)` and carries no dependence on this `floor`. Nor is the
    /// [`CurvatureReg`] operator, which reads the SVD of `R_b` itself, never the floored `R⁻¹`; the
    /// curvature setting survives this call.
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

    fn curvature_op(&self) -> Option<&ColumnDeflation<T>> {
        let reg = self.curvature;
        if reg == CurvatureReg::Off {
            return None;
        }
        self.curv_op
            .get_or_init(|| {
                let p = match reg {
                    CurvatureReg::Hard(f) => ColumnDeflation::from_gauges(self.bond_gauges(), f),
                    CurvatureReg::Tikhonov(e) => {
                        ColumnDeflation::from_gauges_weighted(self.bond_gauges(), |s| {
                            s * s / (s * s + e * e)
                        })
                    }
                    CurvatureReg::Off => unreachable!("returned above"),
                };
                (!p.is_identity()).then_some(p)
            })
            .as_ref()
    }

    fn envs(&self) -> &HessEnvs<T> {
        self.envs.get_or_init(|| {
            let d = self.base_l.space.d;
            let ws: Vec<OneSiteWorkspace> = (0..d)
                .map(|k| onesite_workspace(self.h, self.base_l.core(k)))
                .collect();

            let (pz_hi, qz_hi, grad) =
                hz_stacks_and_gradient(self.h, &self.base_l, &self.base_r, self.lambda, &ws);

            let z_lo = &(&*self.base_l * T::from_f64(-self.lambda)) - &grad.to_vector();
            let zlo_src = ExplicitZ::new(&z_lo);
            let pz_lo = left_env_z(&self.base_l, &zlo_src);
            let qz_lo = right_env_z(&self.base_r, &zlo_src);

            HessEnvs {
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

    /// The Riemannian gradient `P(Hx − λx)` at the base point.
    pub fn gradient(&self) -> &TtTangent<T> {
        &self.envs().grad
    }

    /// The per-bond gauges `{R_b, Σ_b, R_b⁻¹}` at the base point (interior bonds `1..d`), built
    /// lazily and shared across applies. Schmidt-spectrum diagnostics and the curvature-aware
    /// preconditioner read `sigma`/`dropped_mass` from here. Built at [`Self::with_floor`]'s policy,
    /// raised to `f` under `CurvatureReg::Hard(f)` (`dropped_mass` then includes the cut).
    pub fn bond_gauges(&self) -> &[BondGauge<T>] {
        self.gauges
            .get_or_init(|| bond_gauges(&self.base_l, &self.base_r, self.inverse_floor()))
    }

    fn inverse_floor(&self) -> FloorPolicy {
        match (self.floor, self.curvature) {
            (FloorPolicy::Hard(g), CurvatureReg::Hard(f)) => FloorPolicy::Hard(g.max(f)),
            (floor, _) => floor,
        }
    }

    fn curvature_gauges(&self) -> &[BondGauge<T>] {
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

    /// The SECOND gauge set: the same base-point bond gauges rebuilt under
    /// [`FloorPolicy::Tikhonov`]`(s)`, for the σ-damped trust-region model
    /// ([`crate::solvers::joint::WDampRule`]). [`Self::bond_gauges`] is left at this context's own
    /// `floor`, so the spectral consumers sharing the context keep the undamped operator.
    pub fn bond_gauges_damped(&self, s: f64) -> &[BondGauge<T>] {
        let (s0, g) = self.gauges_damped.get_or_init(|| {
            (
                s,
                bond_gauges(&self.base_l, &self.base_r, FloorPolicy::Tikhonov(s)),
            )
        });
        assert!(
            *s0 == s,
            "bond_gauges_damped: this context cached the damped gauge set at s = {s0:e} and was \
             asked for s = {s:e} — the Tikhonov scale is base-point data (one outer iteration, \
             one s); rebuild the context to change it"
        );
        g
    }

    /// The gauge frames the context was bound at — build tangent bases against these
    /// (e.g. [`crate::core::tangent::tangent_basis`]) so applies and basis agree (H3).
    pub fn bases(&self) -> (&TtVector<T>, &TtVector<T>) {
        (&self.base_l, &self.base_r)
    }

    pub(crate) fn shared_bases(&self) -> (&Arc<TtVector<T>>, &Arc<TtVector<T>>) {
        (&self.base_l, &self.base_r)
    }

    /// The effective-Hamiltonian term `A·ξ` (composed for `hess`/`hpp`/`hmm`), contracted as
    /// fused environment sweeps — **`H·ι(ξ)` is never formed**.
    pub fn a_apply(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        xi.refuse_if_shifted("WeingartenCtx::a_apply");
        if crate::knobs::A_APPLY_EXACT.get() {
            return self.a_apply_exact(xi);
        }
        self.a_apply_fused(xi)
    }

    /// The unfused reference term-A apply, [`effham_apply`] at this context's frames and `λ` —
    /// what `QNTT_A_APPLY_EXACT` routes [`Self::a_apply`] to. Refuses a SHIFTED tangent, as
    /// [`Self::a_apply`] does.
    pub fn a_apply_exact(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        xi.refuse_if_shifted("WeingartenCtx::a_apply_exact");
        effham_apply(self.h, &self.base_l, &self.base_r, self.lambda, xi)
    }

    /// The fused environment-sweep term-A apply, bypassing the `QNTT_A_APPLY_EXACT` switch — the
    /// entry point the equivalence tests pin against [`effham_apply`]. See [`Self::a_apply`];
    /// refuses a SHIFTED tangent, as that does.
    pub fn a_apply_fused(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        xi.refuse_if_shifted("WeingartenCtx::a_apply_fused");
        let e = self.envs();
        let ws = &e.ws[..];
        let components = crate::core::weingarten::FusedTermA {
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

        let mut out = TtTangent::from_parts(
            self.base_l.ranks.clone(),
            self.base_l.clone(),
            self.base_r.clone(),
            components,
        );
        out.axpy(T::from_f64(-self.lambda), xi);
        out
    }

    /// Approximate coefficient diagonal of the term-A apply, as a tangent-shaped vector:
    /// component `k` is the ALS-style effective-Hamiltonian diagonal
    /// [`SparseHamiltonian::effective_diagonal`] evaluated in the tangent gauge frames (`Fᴸ`
    /// folded through `base_l`, `Fᴿ` through `base_r` — exactly the frames component `k` lives
    /// against), minus `λ` on every coefficient. This is the Davidson **preconditioner**
    /// diagonal (the deferred `effective_diagonal` lever), NOT an operator: it is
    /// exact at `d = 1` (where `P = id`) and elsewhere ignores the gauge-complement projector and
    /// the cross-component coupling through `P`. Cost: two environment sweeps (`O(d)`
    /// H-matvec-class folds) + `d` per-site diagonals, once per solve.
    pub fn a_diagonal(&self) -> TtTangent<T> {
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
        let mut diag = TtTangent::from_parts(
            self.base_l.ranks.clone(),
            self.base_l.clone(),
            self.base_r.clone(),
            components,
        );
        let shift = T::from_f64(-self.lambda);
        diag.for_each_coeff_mut(|c| *c += shift);
        diag
    }

    /// The Weingarten curvature term `B·ξ = P_X(D_ξ P_X) Z` (§3.3–3.5), assembled matrix-free
    /// against the cached bond gauges and the SPLIT normal residual — `H·x` is never
    /// materialized.
    pub fn b_apply(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        xi.refuse_if_shifted("WeingartenCtx::b_apply");
        self.b_apply_any(xi)
    }

    pub(crate) fn b_apply_any(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        let Some(d) = self.curvature_op() else {
            return self.b_apply_raw(xi);
        };
        let mut v = xi.clone();
        d.apply(&mut v);
        if matches!(self.curvature, CurvatureReg::Tikhonov(_)) {
            let g = self.curvature_gauges();
            let mut first = self.b_assemble(xi, g, Terms::S1_S2);
            d.apply(&mut first);
            let second = self.b_assemble(&v, g, Terms::S3);
            first.add(&second)
        } else {
            let mut b = self.b_apply_raw(&v);
            d.apply(&mut b);
            b
        }
    }

    fn b_apply_raw(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        self.b_assemble(xi, self.bond_gauges(), Terms::all())
    }

    fn b_assemble(&self, xi: &TtTangent<T>, gauges: &[BondGauge<T>], terms: Terms) -> TtTangent<T> {
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
        let zhi = ImplicitHZ {
            h: self.h,
            ket: &self.base_l,
            ws: &e.ws,
            envs: Some((&e.pz_hi, &e.qz_hi)),
        };
        let zlo = ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
        weingarten_assemble_sources(&self.base_l, &self.base_r, &[&zhi, &zlo], xi, gauges, terms)
    }

    /// [`Self::b_apply`] against the σ-DAMPED gauge set: the floored `Σ⁻¹` is replaced by the
    /// Tikhonov weighting `σ ↦ σ/(σ²+s²)`, which is algebraically the per-mode scaling
    /// `w(σ) = σ²/(σ²+s²)` of the Weingarten contribution — unity on healthy bonds, annihilating
    /// the near-null tail whose `1/σ` blow-up is this term's failure mechanism. Everything else —
    /// the split normal residual, the cached environments, the assembly kernel — is
    /// [`Self::b_apply`]'s, unchanged; only the gauge set is swapped.
    pub fn b_apply_damped(&self, xi: &TtTangent<T>, s: f64) -> TtTangent<T> {
        xi.refuse_if_shifted("WeingartenCtx::b_apply_damped");
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
        let zhi = ImplicitHZ {
            h: self.h,
            ket: &self.base_l,
            ws: &e.ws,
            envs: Some((&e.pz_hi, &e.qz_hi)),
        };
        let zlo = ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
        weingarten_assemble_sources(
            &self.base_l,
            &self.base_r,
            &[&zhi, &zlo],
            xi,
            self.bond_gauges_damped(s),
            Terms::all(),
        )
    }

    /// Whole Riemannian Hessian `H^rr = A·ξ + B·ξ` (the real-layer FD-of-gradient target, §4.3).
    /// Refuses a SHIFTED tangent — see [`Self::a_apply`].
    pub fn hess_apply(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        xi.refuse_if_shifted("WeingartenCtx::hess_apply");
        self.a_apply(xi).add(&self.b_apply(xi))
    }

    /// `H^rr = A + B` (alias of [`Self::hess_apply`]; the H3 symplectic driver's `(H^rr)^{1/2}`).
    pub fn hpp_apply(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        self.hess_apply(xi)
    }

    /// `H^ii = A − B` (the H3 symplectic driver's inner factor, §4.3).
    pub fn hmm_apply(&self, xi: &TtTangent<T>) -> TtTangent<T> {
        self.a_apply(xi).sub(&self.b_apply(xi))
    }
}

impl<'h> WeingartenCtx<'h, f64> {
    /// Regularize this context's curvature term (`Off` restores the exact operator).
    ///
    /// # Panics
    ///
    /// Unless the scale of a `Hard`/`Tikhonov` request is finite and `> 0` (the message names the
    /// value). `CurvatureReg::Off` is the only spelling of "no regularization".
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

#[cfg(test)]
mod t8m_tests {
    use crate::TtVector;
    use crate::coeff::CoeffTensor;
    use crate::core::tangent::TtTangent;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::rayleigh_quotient;
    use crate::round::RoundMode;
    use crate::scalar::Scalar;
    use crate::test_util::{generic_truncated_h, randomized_state};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t8m_tangent_diagonal_matches_bruteforce() {
        use crate::SparseCore;
        use crate::hamiltonian::matvec::onesite_workspace;
        use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
        use crate::solvers::davidson::effective_diagonal_bruteforce;
        use crate::solvers::hess::weingarten_operator;
        let d = 6usize;
        let space = QnSpace::new(d, 2, 2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71A4);
        let mut x: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let h = generic_truncated_h::<f64>(space);
        let lambda = rayleigh_quotient(&h, &x).re_s();
        let ctx = weingarten_operator(&h, &x);
        let diag = ctx.a_diagonal();
        let (base_l, base_r) = ctx.bases();
        let fl = left_to_right_sketching(&h, base_l);
        let fr = right_to_left_sketching(&h, base_r);
        for k in 0..d {
            let ws = onesite_workspace(&h, base_l.core(k));
            let mut op = |a: &SparseCore<f64>| h.sketched_hamiltonian(a, &fl[k], &fr[k], &ws);
            let o_diag = effective_diagonal_bruteforce(&mut op, base_l.core(k));
            assert!(
                o_diag.norm() > 1e-6,
                "vacuous effective diagonal at k={k} (‖O‖={})",
                o_diag.norm()
            );
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
    }

    #[test]
    fn t8m_tangent_diagonal_d1_exact() {
        use crate::solvers::davidson::effective_diagonal_bruteforce;
        use crate::solvers::hess::weingarten_operator;
        let space = QnSpace::new(1, 1, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71A5);
        let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
        x.scale_mut(1.0 / x.norm());
        let mut t = CoeffTensor::<f64>::zeros(1, 2);
        t.set(&[0, 0], 0.7);
        let v = CoeffTensor::<f64>::zeros(1, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let ctx = weingarten_operator(&h, &x);
        let diag = ctx.a_diagonal();
        let mut op = |xi: &TtTangent<f64>| ctx.a_apply(xi);
        let o_diag = effective_diagonal_bruteforce(&mut op, &diag);
        let mut delta = diag.clone();
        delta.axpy(-1.0, &o_diag);
        assert!(
            delta.norm() <= 1e-13 * (1.0 + o_diag.norm()),
            "d=1 diagonal must be exact: defect {:.3e}",
            delta.norm()
        );
    }

    #[test]
    fn t8m_pd_probe_witness_at_noncritical_point() {
        use crate::solvers::hess::{HessPart, PdVerdict, hessian_pd_probe, weingarten_operator};
        let d = 4usize;
        let space = QnSpace::new(d, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let ham = generic_truncated_h::<f64>(space);
        let ctx = weingarten_operator(&ham, &x);
        let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let (base_l, base_r) = ctx.bases();
        let v0 = TtTangent::project(base_l, base_r, &dir);
        let (verdict, _applies) = hessian_pd_probe(&ctx, HessPart::Rr, &v0, 60);
        match verdict {
            PdVerdict::WitnessNegative { lambda } => {
                assert!(
                    lambda < -1e-6,
                    "witness should be O(1) negative, got {lambda}"
                );
            }
            PdVerdict::LikelyPd {
                lambda_min,
                residual,
            }
            | PdVerdict::Inconclusive {
                lambda_min,
                residual,
            } => panic!(
                "expected a negative witness at a random non-critical point; got a non-negative \
                 verdict ({lambda_min}, res {residual})"
            ),
        }
    }
}

#[cfg(test)]
mod precond_default_tests {
    use crate::solvers::hess::{SympLobpcgOpts, SympPrecond, SymplecticDavidsonOpts};

    #[test]
    fn ta_auto_is_the_default_and_resolves_per_layer() {
        assert_eq!(SympPrecond::default(), SympPrecond::Auto);
        assert_eq!(
            SympPrecond::Auto.resolve_tt(),
            SympPrecond::DiagUnshifted { kappa: 0.0 }
        );
        assert_eq!(SympPrecond::Auto.resolve_joint(), SympPrecond::Off);
        #[allow(unused_mut)]
        let mut explicit = vec![SympPrecond::Off, SympPrecond::DiagUnshifted { kappa: 7.0 }];
        for p in explicit {
            assert_eq!(p.resolve_tt(), p);
            assert_eq!(p.resolve_joint(), p);
        }
    }

    #[test]
    fn sector_precond_auto_resolves_to_off() {
        use crate::solvers::hess::SectorOpts;
        assert_eq!(SympPrecond::Auto.resolve_sector(), SympPrecond::Off);
        assert_ne!(
            SympPrecond::Auto.resolve_sector(),
            SympPrecond::Auto.resolve_tt(),
            "the sector resolution is not the TT one (D0) until a measured table says so"
        );
        #[allow(unused_mut)]
        let mut explicit = vec![SympPrecond::Off, SympPrecond::DiagUnshifted { kappa: 7.0 }];
        for p in explicit {
            assert_eq!(p.resolve_sector(), p, "{p:?} passes through");
        }
        assert_eq!(
            SectorOpts::default().precond,
            SympPrecond::Auto,
            "Q3: the sector default is the layer-resolved Auto"
        );
        assert_eq!(
            SectorOpts::default().precond.resolve_sector(),
            SympPrecond::Off
        );
    }

    #[test]
    fn td0_default_arm_is_layer_resolved_auto() {
        assert_eq!(SympPrecond::default(), SympPrecond::Auto);
        assert_eq!(
            SymplecticDavidsonOpts::default().precond,
            SympPrecond::Auto,
            "the Davidson default preconditioner is the layer-resolved Auto (H11 promotion)"
        );
        assert_eq!(
            SympLobpcgOpts::default().precond,
            SympPrecond::Auto,
            "the ELOBP4dCG default preconditioner is the layer-resolved Auto (H11 promotion)"
        );
        assert_eq!(
            SympPrecond::default().resolve_tt(),
            SympPrecond::DiagUnshifted { kappa: 0.0 },
            "the TT resolution of the default is D0, never the refuted θ-shifted arm"
        );
        assert_eq!(
            SympPrecond::default().resolve_joint(),
            SympPrecond::Off,
            "the joint resolution of the default is Off, never the refuted θ-shifted arm"
        );
    }
}
