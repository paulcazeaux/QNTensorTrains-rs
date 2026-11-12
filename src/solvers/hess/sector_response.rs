// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::tangent::{TtTangent, min_schmidt};
use crate::core::weingarten::{RpaMetricBlock, WilliamsonError, williamson_reduced_signed_vectors};
use crate::elementaryops::{apply_cross_target, property_flux_classes};
use crate::integrals::PropertyOperator;
use crate::ops::dot;
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::QnSpace;
use crate::solvers::InnerProductVector;
use crate::solvers::joint::{
    DynamicSolver, NearResonanceHit, PropRoute, RefusalKind, oscillator_strengths,
    select_solver_from,
};
use crate::solvers::minres::{MinresFlag, minres};
use crate::vector::tt_zeros;

use super::{
    PcgFlag, RpaMetric, RpaModeKind, RpaReadout, SectorCtx, SectorOpts, SectorReadout,
    SectorSpectrum, pcg_solve,
};

/// Why a channel observable was refused.
#[derive(Clone, Debug)]
pub enum SectorResponseError {
    /// `‖P((H−λ)x)‖` exceeds `tt_grad_tol · max(1,|λ|)`. `B^φ` contracts against the normal
    /// residual and is invalid off criticality; the response expansion eliminates the displacement
    /// only at a stationary point. There is NO orbital leg on a channel — this is the whole gate,
    /// which is why the refusal is not spelled `NotJointCritical`.
    NotTtCritical {
        tt_gradnorm: f64,
        tt_bound: f64,
        lambda: f64,
    },
    /// `H^{rr,φ}` is indefinite on `T^φ`: `Γ0 ⊕ φ` holds a state below `λ`, so the reference is
    /// not the lowest state of this channel and `2pᵀ(H^rr)⁻¹p` is not a polarizability. A
    /// REFUSAL, never a Levenberg shift — criticality is not stability. Diagnose
    /// with [`SectorCtx::symplectic_eigenvalues`].
    HessianIndefinite {
        label: String,
        rho: f64,
        iters: usize,
    },
    /// The channel pencil admits no real symplectic basis at all — BOTH blocks indefinite,
    /// complex quartets — the kernel's own verdict.
    SymplecticUnstable { reason: WilliamsonError },
    /// `ω < 0` (or NaN). `α(ω)` is EVEN in `ω`; call with `|ω|`.
    NegativeOmega { omega: f64 },
    /// `ω` fell inside the caller-set relative exclusion window of a channel eigenvalue. Refused
    /// whenever a spectrum is supplied, INDEPENDENTLY of the solver: MINRES converges perfectly
    /// well next to a pole and returns a number whose error bar the caller cannot reconstruct.
    NearResonance {
        omega: f64,
        omega_n: f64,
        gap: f64,
        window: f64,
    },
    /// PCG met negative curvature at `ω > 0` — by the exact inertia count
    /// (`#{negative eig of K^φ(ω)} = #{n : ω_n < ω}`) a PROOF that `ω > ω₁`, not a claim about the
    /// reference. Remedy: MINRES.
    IndefiniteAbovePole {
        label: String,
        omega: f64,
        rho: f64,
        iters: usize,
    },
    /// The Krylov solve hit its cap. CG on the SPD system is a monotone LOWER bound, so a partial
    /// α is a bound of unknown tightness and is refused rather than published.
    SolveNotConverged {
        label: String,
        iters: usize,
        rel_residual: f64,
        tol: f64,
    },
    /// The property's flux class is not this channel's, or `m` carries more than one class.
    WrongFlux {
        label: String,
        classes: Vec<Irrep>,
        phi: Option<Irrep>,
    },
    /// The ITERATIVE moment route met a negative Ritz value `θ = ω² < 0` whose fresh-apply
    /// certificate did NOT come out negative (`NegativeWitness::certified() == false`).
    /// A negative Ritz value on its own proves nothing — a positive `fresh` beside it is a solver
    /// artifact (a corrupted projected problem), so it is neither published as a resonance nor
    /// dropped silently. Remedy: a tighter `solver.tol`, another seed, or the dense route.
    UncertifiedNegative {
        index: usize,
        theta: f64,
        fresh: f64,
    },
}

impl std::fmt::Display for SectorResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SectorResponseError::NotTtCritical {
                tt_gradnorm,
                tt_bound,
                lambda,
            } => write!(
                f,
                "the reference is not TT-critical: ‖P((H−λ)x)‖ = {tt_gradnorm:.3e} exceeds \
                 {tt_bound:.3e} at λ = {lambda:.9}. B^φ contracts against the normal residual and \
                 is invalid off criticality — polish the reference (there is no orbital leg on a \
                 channel to gate)"
            ),
            SectorResponseError::HessianIndefinite { label, rho, iters } => write!(
                f,
                "H^rr is indefinite on this channel (property {label:?}: curvature {rho:.3e} \
                 after {iters} iterations): Γ0 ⊕ φ holds a state below λ, so the reference is not \
                 the lowest state of the channel and 2pᵀ(H^rr)⁻¹p is not a polarizability. \
                 Diagnose with SectorCtx::symplectic_eigenvalues; a shift would not repair it"
            ),
            SectorResponseError::SymplecticUnstable { reason } => write!(
                f,
                "the channel pencil admits no real symplectic basis ({reason:?}): both H^rr and \
                 H^ii are indefinite, so there is no partial answer either"
            ),
            SectorResponseError::NegativeOmega { omega } => write!(
                f,
                "ω = {omega:e} is negative or NaN; α(ω) is EVEN in ω — call with |ω|"
            ),
            SectorResponseError::NearResonance {
                omega,
                omega_n,
                gap,
                window,
            } => write!(
                f,
                "ω = {omega:e} sits inside the exclusion window of the channel pole \
                 ω_n = {omega_n:e} (gap {gap:e}, window {window:e}·max(1,|ω_n|)); the residue is \
                 available without a solve as T^φ_n"
            ),
            SectorResponseError::IndefiniteAbovePole {
                label,
                omega,
                rho,
                iters,
            } => write!(
                f,
                "PCG met negative curvature ({rho:.3e} after {iters} iterations, property \
                 {label:?}) at ω = {omega:e}: by the exact inertia count that PROVES ω > ω₁, and \
                 says nothing about the reference. Use DynamicSolver::Minres above the first pole"
            ),
            SectorResponseError::SolveNotConverged {
                label,
                iters,
                rel_residual,
                tol,
            } => write!(
                f,
                "the channel solve for {label:?} stopped at {iters} iterations with a fresh \
                 relative residual {rel_residual:.3e} against tol {tol:e}; a partial CG iterate is \
                 a bound of unknown tightness, so nothing is published"
            ),
            SectorResponseError::WrongFlux {
                label,
                classes,
                phi,
            } => match phi {
                Some(phi) => write!(
                    f,
                    "property {label:?} carries the irrep flux classes {classes:?}, which is not \
                     this channel's φ = {phi:?}; split m by property_flux_classes and call once \
                     per class"
                ),
                None => write!(
                    f,
                    "property {label:?} carries {} irrep flux classes {classes:?}, so it belongs \
                     to no single channel; split m by property_flux_classes and call once per class",
                    classes.len()
                ),
            },
            SectorResponseError::UncertifiedNegative {
                index,
                theta,
                fresh,
            } => write!(
                f,
                "mode {index} of the iterative moment route has a negative Ritz value ω² = \
                 {theta:.3e} whose fresh-apply certificate is {fresh:.3e} (not negative): a solver \
                 artifact, not a resonance, so nothing is published. Tighten solver.tol, \
                 change the seed, or use transition_moments_dense"
            ),
        }
    }
}

impl std::error::Error for SectorResponseError {}

pub(super) fn sector_refusal_to_error(
    refusal: RefusalKind,
    label: String,
    omega: f64,
    iters: usize,
    rel_residual: f64,
    tol: f64,
) -> Option<SectorResponseError> {
    match refusal {
        RefusalKind::Curvature(rho) => Some(if omega == 0.0 {
            SectorResponseError::HessianIndefinite { label, rho, iters }
        } else {
            SectorResponseError::IndefiniteAbovePole {
                label,
                omega,
                rho,
                iters,
            }
        }),
        RefusalKind::NotConverged(Some(it)) => Some(SectorResponseError::SolveNotConverged {
            label,
            iters: it,
            rel_residual,
            tol,
        }),
        RefusalKind::NotConverged(None) => None,
    }
}

/// Which channel a property belongs to: `Γ_μ = property_flux_classes(&p.m, pg)[0]`, or `None` when
/// `m ≡ 0` (no support, no class) — **never a guess**.
///
/// # Errors
///
/// [`SectorResponseError::WrongFlux`] on a multi-class `m` — it has no single output space, so
/// picking one of its classes would be a guess. This free function is not bound to a channel, so
/// the refusal's `phi` field is `None`; `classes` carries the content.
pub fn property_flux(
    p: &PropertyOperator<f64>,
    pg: PointGroup,
) -> Result<Option<Irrep>, SectorResponseError> {
    let classes = property_flux_classes(&p.m, pg);
    match classes.len() {
        0 => Ok(None),
        1 => Ok(Some(classes[0])),
        _ => Err(SectorResponseError::WrongFlux {
            label: p.label.clone(),
            classes,
            phi: None,
        }),
    }
}

pub(super) struct ChannelCovector {
    pub(super) p: TtTangent<f64>,
    pub(super) norm: f64,
    pub(super) capture_defect: f64,
}

impl SectorCtx<'_, '_, f64> {
    /// `p_μ = P^φ(μ̂ x)` — the channel's property covector, in the plain tangent metric.
    ///
    /// # Errors
    ///
    /// [`SectorResponseError::WrongFlux`] — a single class that is not `φ`, or more than one class.
    /// **Both** are refused here, which is what separates this entry point from the batch ones: a
    /// caller who asked for THIS channel's covector of an out-of-flux component asked a question
    /// with no answer, while a caller looping components over channels
    /// ([`Self::polarizability_static`], [`Self::transition_moments_dense`],
    /// [`Self::linear_response`]) hits an out-of-flux component by construction and wants a
    /// structural zero ROW instead. A multi-class `m` is refused everywhere — each class maps the
    /// target to a different output space, so there is no single output train.
    ///
    /// `m ≡ 0` is NOT an error: it returns the shaped zero of `T^φ`, and it must be
    /// short-circuited rather than swept, because [`apply_cross_target`] returns `tt_zeros` in the
    /// INPUT space (target `Γ0`) on an empty support and `project_shifted` hard-asserts
    /// `space_out`.
    ///
    /// # Panics
    ///
    /// Through [`apply_cross_target`], on an `m` that is not a symmetric rank-2 `d×d` matrix.
    pub fn property_vector(
        &self,
        p: &PropertyOperator<f64>,
    ) -> Result<TtTangent<f64>, SectorResponseError> {
        let classes = property_flux_classes(&p.m, self.pg());
        if classes.first().is_some_and(|&g| g != self.phi()) || classes.len() > 1 {
            return Err(SectorResponseError::WrongFlux {
                label: p.label.clone(),
                classes,
                phi: Some(self.phi()),
            });
        }
        self.covector(p).map(|c| c.p)
    }

    fn zero_covector(&self) -> TtTangent<f64> {
        self.project(&tt_zeros::<f64>(self.space_out()))
    }

    fn pg(&self) -> PointGroup {
        self.base_space().pg
    }

    fn base_space(&self) -> QnSpace {
        let (bl, _) = self.bases();
        bl.space
    }

    fn covector(&self, p: &PropertyOperator<f64>) -> Result<ChannelCovector, SectorResponseError> {
        let pg = self.pg();
        let classes = property_flux_classes(&p.m, pg);
        if classes.is_empty() {
            return Ok(ChannelCovector {
                p: self.zero_covector(),
                norm: 0.0,
                capture_defect: 0.0,
            });
        }
        if classes.len() > 1 {
            return Err(SectorResponseError::WrongFlux {
                label: p.label.clone(),
                classes,
                phi: Some(self.phi()),
            });
        }
        if classes[0] != self.phi() {
            return Ok(ChannelCovector {
                p: self.zero_covector(),
                norm: 0.0,
                capture_defect: 1.0,
            });
        }
        let (bl, _) = self.bases();
        let mx = apply_cross_target(bl, &p.m, pg);
        debug_assert_eq!(
            mx.space,
            self.space_out(),
            "apply_cross_target's output space must be space_out at Γ_μ = φ"
        );
        let mx2 = dot(&mx, &mx);
        let pv = self.project(&mx);
        let norm = pv.norm();
        let capture_defect = if mx2 > 0.0 {
            (1.0 - (norm * norm) / mx2).max(0.0).sqrt()
        } else {
            0.0
        };
        Ok(ChannelCovector {
            p: pv,
            norm,
            capture_defect,
        })
    }

    pub(super) fn covectors(
        &self,
        props: &[PropertyOperator<f64>],
        entry: &str,
    ) -> Result<Vec<ChannelCovector>, SectorResponseError> {
        assert!(
            props.len() <= 3,
            "{entry}: at most three Cartesian components ({} given) — `oscillator_strengths` is a \
             three-row kernel whose 2/3 is the isotropic average over x, y, z, and a fourth \
             component is a caller error rather than a panic deep inside it",
            props.len()
        );
        props.iter().map(|p| self.covector(p)).collect()
    }

    fn tt_critical_gate(&self, tt_grad_tol: f64) -> Result<(), SectorResponseError> {
        let lambda = self.lambda();
        let tt_gradnorm = self.weingarten().gradient().norm();
        let tt_bound = tt_grad_tol * lambda.abs().max(1.0);
        if tt_gradnorm > tt_bound || tt_gradnorm.is_nan() {
            return Err(SectorResponseError::NotTtCritical {
                tt_gradnorm,
                tt_bound,
                lambda,
            });
        }
        Ok(())
    }

    fn sigma_min(&self) -> f64 {
        let (bl, br) = self.bases();
        min_schmidt(bl, br)
    }

    fn debug_assert_unit_reference(&self) {
        if cfg!(debug_assertions) {
            let (bl, _) = self.bases();
            let n = bl.norm();
            debug_assert!(
                (n - 1.0).abs() < 1e-8,
                "SectorCtx observables: the reference must be normalized (precondition), \
                 ‖x‖ = {n:.15e}"
            );
        }
    }
}

/// Options of [`SectorCtx::polarizability_static`].
#[derive(Clone, Debug)]
pub struct SectorStaticOpts {
    /// Relative residual tolerance of each PCG solve, in the plain tangent metric. Default `1e-8`.
    pub tol: f64,
    /// Iteration cap per solve. Default `500`; CG terminates in at most `dim T^φ` exact steps.
    pub max_iter: usize,
    /// TT-criticality gate on `‖P((H−λ)x)‖`, RELATIVE to `max(1,|λ|)`. Default `1e-6`.
    pub tt_grad_tol: f64,
    /// Hard-assert bound on the α symmetry defect, RELATIVE to `max(1, max|α|)`. `None` (the
    /// default) means `100·tol`, **recomputed from the ACTUAL [`Self::tol`] at solve time**, not
    /// frozen at `Default::default()`'s `tol`.
    pub sym_assert_rel: Option<f64>,
}

impl Default for SectorStaticOpts {
    fn default() -> Self {
        SectorStaticOpts {
            tol: 1e-8,
            max_iter: 500,
            tt_grad_tol: 1e-6,
            sym_assert_rel: None,
        }
    }
}

/// Per-property solve diagnostics of a channel.
#[derive(Clone, Debug)]
pub struct SectorSolveInfo {
    /// The property's [`PropertyOperator::label`], echoed so a caller reading `solves` alone can
    /// tell which row it is looking at.
    pub label: String,
    /// CG iterations taken (`= H^rr` applies inside the solve, each one `a_apply` + `b_apply`).
    /// `0` on a structural zero row and on an empty channel — the no-apply proxy.
    pub iters: usize,
    /// FRESH `‖p − H^rr ξ‖/‖p‖`: one extra apply after the solve, never CG's recursively updated
    /// residual. `0.0` (never NaN) when `p ≡ 0`.
    pub rel_residual: f64,
    /// `‖p_μ‖`. `0.0` ⇒ the property is invisible in this channel (its flux is not `φ`, or
    /// `m ≡ 0`, or the channel is empty) and its whole α row and column are exactly zero — no
    /// apply is paid.
    pub grad_norm: f64,
}

/// What [`SectorCtx::polarizability_static`] returns.
#[derive(Clone, Debug)]
pub struct SectorStaticResponse {
    /// SYMMETRIZED `α^φ`, row-major `n_props²`, in the caller's property order.
    pub alpha: Vec<f64>,
    /// `max_{μ<ν}|raw[μ,ν] − raw[ν,μ]|` BEFORE symmetrization, with `raw[μ,ν] = 2⟨p_μ, ξ_ν⟩`.
    /// `0.0` for a single property.
    pub alpha_defect: f64,
    /// Per-property diagnostics, same order.
    pub solves: Vec<SectorSolveInfo>,
    /// `dim T^φ` — **NOT** the target sector's dimension.
    pub dim: usize,
    /// The channel's irrep flux.
    pub phi: Irrep,
    /// `Γ0 ⊕ φ`, the irrep the response reaches.
    pub target: Irrep,
    /// `min_schmidt` at the base point.
    pub sigma_min: f64,
    /// `√(1 − ‖P^φ(μ̂x)‖²/‖μ̂x‖²)` per property — how much of the property image the channel's
    /// BORROWED-SUFFIX tangent space missed. `O(1)` on a proper-subspace channel (measured on the
    /// rank-2 typed LiH/STO-3G fixture of `tests/response/sector_util.rs`: `B1` has `dim 12`
    /// against a 56-determinant sector; step-18's pooled-rank-8 reference has 10/56 there — the
    /// count is a property of the grid), exactly `1` on an empty channel or an
    /// out-of-flux component, `0` for `m ≡ 0`. **Reported, never asserted zero.**
    pub capture_defect: Vec<f64>,
}

impl SectorCtx<'_, '_, f64> {
    /// `α^φ_μν = 2·⟨p_μ, (H^{rr,φ})⁻¹ p_ν⟩` on the channel.
    ///
    /// # Errors
    ///
    /// [`SectorResponseError::WrongFlux`] (before any apply),
    /// [`SectorResponseError::NotTtCritical`], [`SectorResponseError::HessianIndefinite`],
    /// [`SectorResponseError::SolveNotConverged`].
    ///
    /// # Panics
    ///
    /// On more than three properties. On an α symmetry defect beyond
    /// [`SectorStaticOpts::sym_assert_rel`] — that certifies OUR OWN assembly, hence an assert and
    /// not an `Err`.
    pub fn polarizability_static(
        &self,
        props: &[PropertyOperator<f64>],
        opts: &SectorStaticOpts,
    ) -> Result<SectorStaticResponse, SectorResponseError> {
        let covs = self.covectors(props, "SectorCtx::polarizability_static")?;
        self.polarizability_static_from_covectors(props, &covs, opts)
    }

    pub(super) fn polarizability_static_from_covectors(
        &self,
        props: &[PropertyOperator<f64>],
        covs: &[ChannelCovector],
        opts: &SectorStaticOpts,
    ) -> Result<SectorStaticResponse, SectorResponseError> {
        debug_assert_eq!(covs.len(), props.len());
        self.debug_assert_unit_reference();

        self.tt_critical_gate(opts.tt_grad_tol)?;

        let op = |q: &TtTangent<f64>| {
            let clean = self.project(&q.to_vector());
            self.hpp_apply(&clean)
        };
        let precond = |_: &mut TtTangent<f64>| {};
        let n = props.len();
        let mut sols: Vec<TtTangent<f64>> = Vec::with_capacity(n);
        let mut solves: Vec<SectorSolveInfo> = Vec::with_capacity(n);
        for (nu, c) in covs.iter().enumerate() {
            let label = props[nu].label.clone();
            let bn = c.norm;
            if bn == 0.0 {
                let mut zero = c.p.clone();
                zero.scale_mut(0.0);
                sols.push(zero);
                solves.push(SectorSolveInfo {
                    label,
                    iters: 0,
                    rel_residual: 0.0,
                    grad_norm: 0.0,
                });
                continue;
            }
            let (xi, flag, iters) = pcg_solve(op, precond, &c.p, opts.tol, opts.max_iter);
            let mut r = c.p.clone();
            r.axpy(-1.0, &op(&xi));
            let rel_residual = r.norm() / bn;
            let refusal = match flag {
                PcgFlag::Converged => RefusalKind::NotConverged(None),
                PcgFlag::MaxIter => RefusalKind::NotConverged(Some(iters)),
                PcgFlag::NegativeCurvature { rho } => RefusalKind::Curvature(rho),
            };
            if let Some(e) =
                sector_refusal_to_error(refusal, label.clone(), 0.0, iters, rel_residual, opts.tol)
            {
                return Err(e);
            }
            sols.push(xi);
            solves.push(SectorSolveInfo {
                label,
                iters,
                rel_residual,
                grad_norm: bn,
            });
        }

        let mut raw = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                raw[i * n + j] = 2.0 * covs[i].p.dot(&sols[j]);
            }
        }
        let (alpha, alpha_defect) = symmetrize_alpha(
            &raw,
            n,
            opts.sym_assert_rel.unwrap_or(100.0 * opts.tol),
            "polarizability_static",
            f64::NAN,
        );

        Ok(SectorStaticResponse {
            alpha,
            alpha_defect,
            solves,
            dim: self.dim(),
            phi: self.phi(),
            target: self.space_out().target,
            sigma_min: self.sigma_min(),
            capture_defect: covs.iter().map(|c| c.capture_defect).collect(),
        })
    }
}

fn symmetrize_alpha(
    raw: &[f64],
    n: usize,
    sym_assert_rel: f64,
    entry: &str,
    omega: f64,
) -> (Vec<f64>, f64) {
    let mut alpha_defect = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            alpha_defect = alpha_defect.max((raw[i * n + j] - raw[j * n + i]).abs());
        }
    }
    let raw_scale = raw.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
    assert!(
        alpha_defect <= sym_assert_rel * raw_scale,
        "SectorCtx::{entry}: α is not symmetric — defect {alpha_defect:.6e} exceeds {:.6e} \
         (sym_assert_rel · max(1,max|α|), max|α| = {raw_scale:.6e}){}. H^rr is self-adjoint in the \
         tangent metric and K^φ(ω) in the product metric, so a defect this large means the apply \
         is not self-adjoint — most likely the −ω coupling was fed an UNCLEANED leg, assembling \
         Π K Π' with Π ≠ Π' — OUR assembly, not the caller's input, hence an assert.",
        sym_assert_rel * raw_scale,
        if omega.is_nan() {
            String::new()
        } else {
            format!(" at ω = {omega:.6e}")
        }
    );
    let mut alpha = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            alpha[i * n + j] = 0.5 * (raw[i * n + j] + raw[j * n + i]);
        }
    }
    (alpha, alpha_defect)
}

/// Options of both moment routes.
#[derive(Clone, Debug)]
pub struct SectorMomentOpts {
    /// TT-criticality gate, RELATIVE to `max(1,|λ|)`. Default `1e-6`: the moments consume `B^φ`
    /// through the Grams.
    pub tt_grad_tol: f64,
    /// RELATIVE window grouping `ω_n` into degenerate clusters: `n` joins the previous cluster
    /// when `ω_n − ω_{n−1} ≤ degeneracy_tol · max(1, ω_n)`. Default `1e-8`.
    pub degeneracy_tol: f64,
    /// PSD tolerance of the Williamson reduction, matching
    /// [`SectorCtx::symplectic_eigenvalues`]' own literal. Default `1e-10`.
    pub psd_tol: f64,
    /// Iterative route only: the [`SectorOpts`] of the RPA solve (`tol`, `max_iter`, `seed`,
    /// `kernel`, `precond`, `max_subspace` — every one honoured). ★ [`SectorOpts::readout`] is
    /// IGNORED and set to [`super::SectorReadout::Rpa`] by the entry point, exactly as
    /// `sector_response_all` does for its own arms.
    pub solver: SectorOpts,
    /// Which route builds the covector. Default [`PropRoute::Materialized`], the only route; `Auto`
    /// resolves to it.
    pub route: PropRoute,
}

impl Default for SectorMomentOpts {
    fn default() -> Self {
        SectorMomentOpts {
            tt_grad_tol: 1e-6,
            degeneracy_tol: 1e-8,
            psd_tol: 1e-10,
            solver: SectorOpts::default(),
            route: PropRoute::Materialized,
        }
    }
}

/// What [`SectorCtx::transition_moments_dense`] and
/// [`SectorCtx::transition_moments_iterative`] return. Every field is in the caller's `props`
/// order.
#[derive(Clone, Debug)]
pub struct SectorTransitionMoments {
    /// The channel's symplectic frequencies, ASCENDING — the REAL modes of POSITIVE Krein type
    /// only (`n_pos`). On the iterative route: those among the `k` lowest real modes the solve
    /// returned (imaginary modes are extra, not counted in `k`; a de-excitation counts in `k` on
    /// the paired kernel but publishes no moment — it is in [`Self::readout`]).
    pub omega: Vec<f64>,
    /// `moments[p][n] = T^φ_n(p) = Σ_i γ_i·s_r[i*n_pos + n]`.
    pub moments: Vec<Vec<f64>>,
    /// [`PropertyOperator::label`], echoed so a caller reading `moments` alone can tell the rows
    /// apart.
    pub labels: Vec<String>,
    /// `2 Σ_n T_n(p)T_n(q)/ω_n` — the sum-over-states reconstruction of
    /// [`SectorStaticResponse::alpha`]. An ALGEBRAIC identity at ANY rank, hence the mutual pin
    /// with the static solve; its error budget is [`Self::completeness_defect`],
    /// [`Self::orth_error`] and that solve's own residual.
    pub alpha_sos: Option<Vec<f64>>,
    /// `f_n = (2/3)ω_nΣ_c T_n(c)²` from `crate::solvers::joint::oscillator_strengths`, reused
    /// VERBATIM — a component whose flux is not `φ` has `T^φ_n ≡ 0` exactly, so passing all three
    /// Cartesian rows to one channel gives the same numbers as passing only the in-channel ones
    /// and the `2/3` keeps its meaning. With fewer than three components this is a PARTIAL sum and
    /// the factor is NOT rescaled.
    pub oscillator: Vec<f64>,
    /// `‖p_μ − Σ_i γ_i b_i‖/‖p_μ‖` — how much of the covector the channel basis missed. `0.0`
    /// when `‖p_μ‖ = 0`. On the ITERATIVE route it is the residual of the OBLIQUE reconstruction
    /// `Σ_n T_n s_i_n` over the `k` computed modes (`S_i S_rᵀ` is idempotent by biorthogonality
    /// but not a contraction), so there it can EXCEED 1 and is not monotone in `k` (measured on
    /// `d6_minimizer` `φ = 3`: `0.98` at `k = 1`, `1.04` at `k = 2`); it is `≈ 0` only at `k = dim`.
    pub completeness_defect: Vec<f64>,
    /// `√(1 − ‖P^φ(μ̂x)‖²/‖μ̂x‖²)` — how much of `μ̂x` the CHANNEL missed. The channel-specific
    /// defect, `O(1)` on a proper-subspace channel. **Reported, never asserted zero.**
    pub capture_defect: Vec<f64>,
    /// `(Σ_n T_n(p)², ‖P^φ(μ̂x)‖²)` per property: EQUAL at `B^φ ≡ 0` (full rank, or a single-slot
    /// channel), and otherwise a DEFECT OF EITHER SIGN — not an inequality. With `γ_i = ⟨b_i, p⟩`,
    /// `Σ_n T_n² = γᵀ S_r S_rᵀ γ` and the Williamson identities (`S_rᵀ H^rr S_r = Ω`,
    /// `S_rᵀ S_i = I`) give `S_r S_rᵀ = (H^rr)⁻¹ # H^ii = (A+B)⁻¹ # (A−B)`: the matrix GEOMETRIC
    /// MEAN, the PD solution `X` of `X (A+B) X = A−B`. It is `≼ I` along `p` where `B ≽ 0` and
    /// `≽ I` where `B ≼ 0`, so the measured `−0.22` relative defect on `d6_minimizer` `φ = 3` is
    /// consistent with `B ≽ 0` along that covector — a measurement, not a law. On the iterative
    /// route the first entry is a PARTIAL sum over the `k` computed modes, monotone in `k`.
    /// The OPERATOR sum rule — the one that is exact and available; see [`Self::sum_rule_si`] for
    /// the reciprocal statement.
    pub sum_rule: Vec<(f64, f64)>,
    /// `Σ_n (S_iᵀγ)_n²` per property — the `S_i`-SIDE sum beside [`Self::sum_rule`]`.0`, i.e.
    /// `γᵀ S_i S_iᵀ γ` with `S_i S_iᵀ = (S_r S_rᵀ)⁻¹ = (A+B) # (A−B)⁻¹`, the reciprocal geometric
    /// mean. Equal to `Σ T²` and to `‖p_μ‖²` at `B^φ ≡ 0`; on a curved channel the two sums
    /// cannot BOTH sit below `‖p‖²` (Cauchy–Schwarz: `(γᵀXγ)(γᵀX⁻¹γ) ≥ ‖γ‖⁴`), which is what
    /// makes the sign of the sum-rule defect readable (the curved defect has NO fixed
    /// sign — measured `−0.22` and `+0.38` on the same channel). Free on both routes (`S_i` is in
    /// hand); partial over the `k` computed modes on the iterative one. `S_i` does NOT enter a
    /// moment — this is a diagnostic, never a `T_n`.
    pub sum_rule_si: Vec<f64>,
    /// `max_{ij}|⟨b_i,b_j⟩ − δ_ij|` over the channel basis — MEASURED here, not inherited: a
    /// shifted channel has no `DropReport`. On the iterative route it is the BIORTHOGONALITY
    /// defect `max_{ij}|⟨s_r_i, s_i_j⟩ − δ_ij|` of the computed modes instead, which is that
    /// route's own normalization statement.
    pub orth_error: f64,
    /// `min_schmidt` at the base point.
    pub sigma_min: f64,
    /// Half-open `[lo,hi)` ranges of degenerate `ω` clusters, partitioning `0..omega.len()`.
    /// Singletons are listed too. A cluster of size > 1 has NO per-mode moment.
    pub degenerate_clusters: Vec<(usize, usize)>,
    /// Imaginary modes NOT represented in `omega`/`moments` — resonances `ω = i√|ω²|`, which carry
    /// no real transition moment. `n_neg > 0` is what makes [`Self::alpha_sos`] `None`.
    pub n_neg: usize,
    /// The signed `ω² < 0` of those modes, ascending.
    pub unstable_omega2: Vec<f64>,
    /// `dim T^φ` — not the target sector's dimension.
    pub dim: usize,
    /// The channel's irrep flux.
    pub phi: Irrep,
    /// `Γ0 ⊕ φ`.
    pub target: Irrep,
    /// Which block served as the PD metric of the Williamson reduction (dense route). On the
    /// iterative route the solve's [`RpaReadout::side`]: the signed kernel's lane, or `Ii` under
    /// the paired kernel, which has no metric (its convention, the same echo as the readout's).
    pub metric: RpaMetricBlock,
    /// `true` on the DENSE route (a full basis, hence a genuine completeness statement); `false`
    /// on the iterative route, where `omega`/`moments` cover the requested `k` roots only and the
    /// sum rule and `alpha_sos` are partial sums by construction, monotone in `k`.
    pub complete_basis: bool,
    /// The iterative route's full RPA solve (`None` on the dense route): every mode it returned —
    /// the de-excitations and complex pairs that carry no moment here, the imaginary modes with
    /// their certificates — and the kernel's report (counting offset or metric events), so nothing
    /// the solve found is withheld. `omega`/`moments` are the positive-type real subset.
    pub readout: Option<RpaReadout>,
}

#[derive(Clone, Debug)]
pub(crate) struct SectorMomentPayload {
    pub omega: Vec<f64>,
    pub moments: Vec<Vec<f64>>,
    pub alpha_sos: Option<Vec<f64>>,
    pub completeness_defect: Vec<f64>,
    pub sum_rule: Vec<(f64, f64)>,
    pub sum_rule_si: Vec<f64>,
    pub degenerate_clusters: Vec<(usize, usize)>,
    pub n_neg: usize,
    pub unstable_omega2: Vec<f64>,
    pub metric: RpaMetricBlock,
}

pub(crate) fn moments_from_grams(
    a: &[f64],
    b: &[f64],
    gammas: &[Vec<f64>],
    opts: &SectorMomentOpts,
) -> Result<SectorMomentPayload, SectorResponseError> {
    assert_eq!(
        a.len(),
        b.len(),
        "moments_from_grams: the (A, B) pair must be the same shape"
    );
    let n = a.len().isqrt();
    assert_eq!(
        n * n,
        a.len(),
        "moments_from_grams: the Grams must be square row-major (got {} entries)",
        a.len()
    );
    for (p, g) in gammas.iter().enumerate() {
        assert_eq!(
            g.len(),
            n,
            "moments_from_grams: property {p}'s coordinate vector has {} entries against a \
             {n}-dimensional channel",
            g.len()
        );
    }
    let hrr: Vec<f64> = a.iter().zip(b).map(|(p, q)| p + q).collect();
    let hii: Vec<f64> = a.iter().zip(b).map(|(p, q)| p - q).collect();
    let sw = williamson_reduced_signed_vectors(&hrr, &hii, n, opts.psd_tol)
        .map_err(|reason| SectorResponseError::SymplecticUnstable { reason })?;
    let n_pos = sw.omega.len();
    let p_count = gammas.len();

    let mut moments: Vec<Vec<f64>> = Vec::with_capacity(p_count);
    let mut sum_rule: Vec<(f64, f64)> = Vec::with_capacity(p_count);
    let mut sum_rule_si: Vec<f64> = Vec::with_capacity(p_count);
    for g in gammas {
        let mut t = vec![0.0f64; n_pos];
        let mut t_i = vec![0.0f64; n_pos];
        for (i, &gi) in g.iter().enumerate() {
            if gi == 0.0 {
                continue;
            }
            for col in 0..n_pos {
                t[col] += gi * sw.s_r[i * n_pos + col];
                t_i[col] += gi * sw.s_i[i * n_pos + col];
            }
        }
        let t2: f64 = t.iter().map(|z| z * z).sum();
        let g2: f64 = g.iter().map(|z| z * z).sum();
        sum_rule.push((t2, g2));
        sum_rule_si.push(t_i.iter().map(|z| z * z).sum());
        moments.push(t);
    }

    let alpha_sos = (sw.n_neg == 0).then(|| {
        let mut out = vec![0.0f64; p_count * p_count];
        for (i, ti) in moments.iter().enumerate() {
            for (j, tj) in moments.iter().enumerate() {
                out[i * p_count + j] = 2.0
                    * ti.iter()
                        .zip(tj)
                        .zip(&sw.omega)
                        .map(|((&u, &w), &wn)| u * w / wn)
                        .sum::<f64>();
            }
        }
        out
    });

    Ok(SectorMomentPayload {
        degenerate_clusters: clusters_of(&sw.omega, opts.degeneracy_tol),
        omega: sw.omega,
        moments,
        alpha_sos,
        completeness_defect: vec![0.0; p_count],
        sum_rule,
        sum_rule_si,
        n_neg: sw.n_neg,
        unstable_omega2: sw.omega2.iter().take(sw.n_neg).copied().collect(),
        metric: sw.metric,
    })
}

fn clusters_of(omega: &[f64], tol: f64) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    if omega.is_empty() {
        return out;
    }
    let mut lo = 0usize;
    for k in 1..omega.len() {
        if omega[k] - omega[k - 1] > tol * omega[k].abs().max(1.0) {
            out.push((lo, k));
            lo = k;
        }
    }
    out.push((lo, omega.len()));
    out
}

fn pairing_defect(u: &[TtTangent<f64>], v: &[TtTangent<f64>]) -> f64 {
    let mut worst = 0.0f64;
    for (i, ui) in u.iter().enumerate() {
        for (j, vj) in v.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((ui.dot(vj) - want).abs());
        }
    }
    worst
}

impl SectorCtx<'_, '_, f64> {
    /// The channel's transition moments and oscillator strengths — DENSE route: [`Self::grams`]
    /// (which returns `(A, B)`, **never** `(H^rr, H^ii)` — [`moments_from_grams`] forms `A ± B`
    /// itself) then [`williamson_reduced_signed_vectors`] over the full [`Self::basis`].
    ///
    /// # Errors
    ///
    /// [`SectorResponseError::WrongFlux`] (before any apply),
    /// [`SectorResponseError::NotTtCritical`], [`SectorResponseError::SymplecticUnstable`] (both
    /// blocks indefinite: complex quartets, no real symplectic basis, partial or otherwise).
    ///
    /// # Panics
    ///
    /// On more than three properties, or on `opts.route == PropRoute::Shifted`.
    pub fn transition_moments_dense(
        &self,
        props: &[PropertyOperator<f64>],
        opts: &SectorMomentOpts,
    ) -> Result<SectorTransitionMoments, SectorResponseError> {
        assert_materialized(opts.route, "SectorCtx::transition_moments_dense");
        let covs = self.covectors(props, "SectorCtx::transition_moments_dense")?;
        self.tt_critical_gate(opts.tt_grad_tol)?;

        let basis = self.basis();
        let (a, b) = self.grams(&basis);
        self.transition_moments_from_grams(props, &covs, &basis, &a, &b, opts)
    }

    pub(super) fn transition_moments_from_grams(
        &self,
        props: &[PropertyOperator<f64>],
        covs: &[ChannelCovector],
        basis: &[TtTangent<f64>],
        a: &[f64],
        b: &[f64],
        opts: &SectorMomentOpts,
    ) -> Result<SectorTransitionMoments, SectorResponseError> {
        assert_materialized(opts.route, "SectorCtx::transition_moments_from_grams");
        debug_assert_eq!(covs.len(), props.len());
        self.debug_assert_unit_reference();
        self.tt_critical_gate(opts.tt_grad_tol)?;

        let gammas: Vec<Vec<f64>> = covs
            .iter()
            .map(|c| basis.iter().map(|bi| bi.dot(&c.p)).collect())
            .collect();
        let mut payload = moments_from_grams(a, b, &gammas, opts)?;

        for (p, c) in covs.iter().enumerate() {
            payload.completeness_defect[p] = if c.norm > 0.0 {
                let mut r = c.p.clone();
                for (i, bi) in basis.iter().enumerate() {
                    r.axpy(-gammas[p][i], bi);
                }
                r.norm() / c.norm
            } else {
                0.0
            };
            payload.sum_rule[p].1 = c.norm * c.norm;
        }

        Ok(self.finish_moments(
            props,
            covs,
            payload,
            pairing_defect(basis, basis),
            true,
            None,
        ))
    }

    /// The `k` lowest modes' moments — ITERATIVE route: the channel's RPA solve
    /// ([`Self::sector_lowest_vectors`] with `opts.solver`, its readout forced to `Rpa`), so the
    /// kernel is [`SectorOpts::kernel`] — the PAIRED kernel by default, the signed one
    /// through `RpaKernel::signed(lane)` — with the readout's seed block, the honoured
    /// [`SectorOpts::precond`] and [`SectorOpts::max_subspace`], `space_dim` with NO radial `−1`
    /// and both empty routes.
    ///
    /// # Errors
    ///
    /// As [`Self::transition_moments_dense`] except `SymplecticUnstable`, which this route never
    /// returns (neither kernel refuses an indefinite channel; its modes are reported in
    /// `readout`). ★ A solve that hit its budget (`converged == false`) is
    /// [`SectorResponseError::SolveNotConverged`] — `label` lists every property (the eigen-solve
    /// is shared), `iters` the basis vectors built, `rel_residual` the worst returned pencil
    /// residual over the largest `|ω|`, `tol` the `solver.tol` it lost against. Nothing from an
    /// unconverged run is published. An uncertified negative is
    /// [`SectorResponseError::UncertifiedNegative`] (`index` = the mode's position in the readout).
    ///
    /// # Panics
    ///
    /// On more than three properties, on `opts.route == PropRoute::Shifted`, and as
    /// [`Self::sector_lowest`] on its options.
    pub fn transition_moments_iterative(
        &self,
        props: &[PropertyOperator<f64>],
        k: usize,
        opts: &SectorMomentOpts,
    ) -> Result<SectorTransitionMoments, SectorResponseError> {
        assert_materialized(opts.route, "SectorCtx::transition_moments_iterative");
        let covs = self.covectors(props, "SectorCtx::transition_moments_iterative")?;
        self.debug_assert_unit_reference();
        self.tt_critical_gate(opts.tt_grad_tol)?;

        let p_count = props.len();
        let solver = SectorOpts {
            readout: SectorReadout::Rpa,
            ..opts.solver.clone()
        };
        let (resp, amps) = self.sector_lowest_vectors(k, &solver);
        let SectorSpectrum::Rpa(readout) = resp.spectrum else {
            unreachable!("sector_lowest_vectors answers the Rpa readout")
        };
        if readout.modes.is_empty() && readout.applies == 0 {
            let payload = SectorMomentPayload {
                omega: Vec::new(),
                moments: vec![Vec::new(); p_count],
                alpha_sos: Some(vec![0.0; p_count * p_count]),
                completeness_defect: vec![0.0; p_count],
                sum_rule: covs.iter().map(|c| (0.0, c.norm * c.norm)).collect(),
                sum_rule_si: vec![0.0; p_count],
                degenerate_clusters: Vec::new(),
                n_neg: 0,
                unstable_omega2: Vec::new(),
                metric: RpaMetricBlock::Rr,
            };
            return Ok(self.finish_moments(props, &covs, payload, 0.0, false, Some(readout)));
        }

        if !readout.converged {
            let worst = readout.residuals().iter().fold(0.0f64, |m, &r| m.max(r));
            let w_max = readout
                .modes
                .iter()
                .fold(0.0f64, |m, md| m.max(md.omega.norm()))
                .max(f64::MIN_POSITIVE);
            return Err(SectorResponseError::SolveNotConverged {
                label: props
                    .iter()
                    .map(|p| p.label.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
                iters: readout.iterations,
                rel_residual: worst / w_max,
                tol: opts.solver.tol,
            });
        }

        let mu_max = readout
            .modes
            .iter()
            .fold(0.0f64, |m, md| m.max(md.omega2.norm()))
            .max(f64::MIN_POSITIVE);
        let cut = opts.psd_tol * mu_max;
        let mut omega: Vec<f64> = Vec::new();
        let mut s_r: Vec<TtTangent<f64>> = Vec::new();
        let mut s_i: Vec<TtTangent<f64>> = Vec::new();
        let mut unstable_omega2: Vec<f64> = Vec::new();
        let mut withheld = false;
        for (n, (mode, z)) in readout.modes.iter().zip(&amps).enumerate() {
            match mode.kind {
                RpaModeKind::Imaginary => {
                    let w2 = mode.omega2.re;
                    if w2 < -cut {
                        match &mode.certificate {
                            Some(w) if w.certified() => unstable_omega2.push(w2),
                            Some(w) => {
                                return Err(SectorResponseError::UncertifiedNegative {
                                    index: n,
                                    theta: w2,
                                    fresh: w.fresh,
                                });
                            }
                            None => {}
                        }
                    }
                }
                RpaModeKind::Real if mode.krein > 0.0 => {
                    if mode.omega2.re <= cut {
                        continue;
                    }
                    let mut sr = z.x.clone();
                    sr.axpy(1.0, &z.y);
                    let mut si = z.x.clone();
                    si.axpy(-1.0, &z.y);
                    omega.push(mode.omega.re);
                    s_r.push(sr);
                    s_i.push(si);
                }
                RpaModeKind::Real | RpaModeKind::Complex => withheld = true,
            }
        }
        let n_neg = unstable_omega2.len();
        let mut moments: Vec<Vec<f64>> = Vec::with_capacity(p_count);
        let mut sum_rule: Vec<(f64, f64)> = Vec::with_capacity(p_count);
        let mut sum_rule_si: Vec<f64> = Vec::with_capacity(p_count);
        let mut completeness_defect: Vec<f64> = Vec::with_capacity(p_count);
        for c in &covs {
            let t: Vec<f64> = s_r.iter().map(|col| col.dot(&c.p)).collect();
            let t2: f64 = t.iter().map(|z| z * z).sum();
            sum_rule.push((t2, c.norm * c.norm));
            sum_rule_si.push(s_i.iter().map(|col| col.dot(&c.p).powi(2)).sum());
            completeness_defect.push(if c.norm > 0.0 {
                let mut r = c.p.clone();
                for (n, col) in s_i.iter().enumerate() {
                    r.axpy(-t[n], col);
                }
                r.norm() / c.norm
            } else {
                0.0
            });
            moments.push(t);
        }
        let alpha_sos = (n_neg == 0 && !withheld).then(|| {
            let mut out = vec![0.0f64; p_count * p_count];
            for (i, ti) in moments.iter().enumerate() {
                for (j, tj) in moments.iter().enumerate() {
                    out[i * p_count + j] = 2.0
                        * ti.iter()
                            .zip(tj)
                            .zip(&omega)
                            .map(|((&u, &w), &wn)| u * w / wn)
                            .sum::<f64>();
                }
            }
            out
        });
        let payload = SectorMomentPayload {
            degenerate_clusters: clusters_of(&omega, opts.degeneracy_tol),
            omega,
            moments,
            alpha_sos,
            completeness_defect,
            sum_rule,
            sum_rule_si,
            n_neg,
            unstable_omega2,
            metric: match readout.side() {
                RpaMetric::Rr => RpaMetricBlock::Rr,
                _ => RpaMetricBlock::Ii,
            },
        };
        let orth_error = pairing_defect(&s_r, &s_i);
        Ok(self.finish_moments(props, &covs, payload, orth_error, false, Some(readout)))
    }

    fn finish_moments(
        &self,
        props: &[PropertyOperator<f64>],
        covs: &[ChannelCovector],
        payload: SectorMomentPayload,
        orth_error: f64,
        complete_basis: bool,
        readout: Option<RpaReadout>,
    ) -> SectorTransitionMoments {
        let oscillator = oscillator_strengths(&payload.omega, &payload.moments);
        SectorTransitionMoments {
            omega: payload.omega,
            moments: payload.moments,
            labels: props.iter().map(|p| p.label.clone()).collect(),
            alpha_sos: payload.alpha_sos,
            oscillator,
            completeness_defect: payload.completeness_defect,
            capture_defect: covs.iter().map(|c| c.capture_defect).collect(),
            sum_rule: payload.sum_rule,
            sum_rule_si: payload.sum_rule_si,
            orth_error,
            sigma_min: self.sigma_min(),
            degenerate_clusters: payload.degenerate_clusters,
            n_neg: payload.n_neg,
            unstable_omega2: payload.unstable_omega2,
            dim: self.dim(),
            phi: self.phi(),
            target: self.space_out().target,
            metric: payload.metric,
            complete_basis,
            readout,
        }
    }
}

fn assert_materialized(route: PropRoute, entry: &str) {
    assert_ne!(
        route,
        PropRoute::Shifted,
        "{entry}: a channel has only the Materialized route — `project_hmul_env` needs the \
         property as a SparseHamiltonian on x's OWN space, which a cross-irrep property is not \
         (assert_property_representable refuses it by name). Pass PropRoute::Materialized"
    );
}

/// `[u; v]` — the unknown of `K^φ(ω)[u;v] = [p;0]`, an [`InnerProductVector`] in the PRODUCT
/// metric `⟨(u₁,v₁),(u₂,v₂)⟩ = ⟨u₁,u₂⟩ + ⟨v₁,v₂⟩`, in which
#[derive(Clone)]
pub struct SectorPair {
    /// The `u = X+Y` leg.
    pub u: TtTangent<f64>,
    /// The `v = X−Y` leg. Identically zero at `ω = 0`, at every iterate.
    pub v: TtTangent<f64>,
}

impl InnerProductVector for SectorPair {
    type Scalar = f64;

    fn dot(&self, other: &Self) -> f64 {
        self.u.dot(&other.u) + self.v.dot(&other.v)
    }

    fn norm(&self) -> f64 {
        self.dot(self).max(0.0).sqrt()
    }

    fn axpy(&mut self, a: f64, other: &Self) {
        self.u.axpy(a, &other.u);
        self.v.axpy(a, &other.v);
    }

    fn scale(&mut self, a: f64) {
        self.u.scale_mut(a);
        self.v.scale_mut(a);
    }

    fn for_each_coeff_mut<F: FnMut(&mut f64)>(&mut self, mut f: F) {
        self.u.for_each_coeff_mut(&mut f);
        self.v.for_each_coeff_mut(&mut f);
    }

    fn zip_coeffs_mut<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, mut f: F) {
        self.u.zip_coeffs_mut(&other.u, &mut f);
        self.v.zip_coeffs_mut(&other.v, &mut f);
    }

    fn coeff(&self, idx: usize) -> f64 {
        let n = self.u.num_coeffs();
        if idx < n {
            self.u.coeff(idx)
        } else {
            self.v.coeff(idx - n)
        }
    }

    fn set_coeff(&mut self, idx: usize, val: f64) {
        let n = self.u.num_coeffs();
        if idx < n {
            self.u.set_coeff(idx, val);
        } else {
            self.v.set_coeff(idx - n, val);
        }
    }

    fn num_coeffs(&self) -> usize {
        self.u.num_coeffs() + self.v.num_coeffs()
    }
}

/// Options of [`SectorCtx::linear_response`] and [`SectorCtx::linear_response_solve`]. Mirrors
/// `DynamicResponseOpts` minus the orbital gates.
#[derive(Clone, Debug)]
pub struct SectorDynamicOpts {
    /// Relative residual tolerance of each solve, in the PRODUCT metric. Default `1e-8`.
    pub tol: f64,
    /// Iteration cap per solve. Default `500`.
    pub max_iter: usize,
    /// Which Krylov solver. Default [`DynamicSolver::Auto`].
    pub solver: DynamicSolver,
    /// TT-criticality gate, RELATIVE to `max(1,|λ|)`. Default `1e-6`.
    pub tt_grad_tol: f64,
    /// Hard-assert bound on the α symmetry defect, RELATIVE to `max(1, max|α|)`. `None` ⇒
    /// `100·tol`, recomputed at solve time — see [`SectorStaticOpts::sym_assert_rel`] for why it
    /// must not be frozen inside `Default::default()`.
    pub sym_assert_rel: Option<f64>,
    /// Which route builds the covector. Default [`PropRoute::Materialized`], the only route; `Auto`
    /// resolves to it.
    pub route: PropRoute,
    /// RELATIVE pole-exclusion window: a solve is refused when
    /// `min_n |ω − ω_n| ≤ resonance_window · max(1, |ω_n|)`. Default `1e-3`. Consulted only when
    /// [`Self::spectrum`] is supplied, and never at `ω = 0`.
    pub resonance_window: f64,
    /// The channel's own `ω` (from [`SectorCtx::transition_moments_dense`] or
    /// [`SectorCtx::symplectic_eigenvalues`]).
    pub spectrum: Option<Vec<f64>>,
    /// How many IMAGINARY modes the channel carries
    /// ([`SectorTransitionMoments::n_neg`], or `SymplecticSpectrum::unstable_pairs`).
    pub unstable_pairs: usize,
}

impl Default for SectorDynamicOpts {
    fn default() -> Self {
        SectorDynamicOpts {
            tol: 1e-8,
            max_iter: 500,
            solver: DynamicSolver::Auto,
            tt_grad_tol: 1e-6,
            sym_assert_rel: None,
            route: PropRoute::Materialized,
            resonance_window: 1e-3,
            spectrum: None,
            unstable_pairs: 0,
        }
    }
}

/// Per-property diagnostics of an ω-dependent channel solve. Mirrors `DynamicSolveInfo` minus
/// the orbital weight.
#[derive(Clone, Debug)]
pub struct SectorDynamicSolveInfo {
    /// The property's [`PropertyOperator::label`].
    pub label: String,
    /// Iterations taken (`= 2·iters` Hessian-class applies: one `hpp`, one `hmm`, each of which is
    /// one `a_apply` + one `b_apply`).
    pub iters: usize,
    /// FRESH `‖b − K^φ(ω)[u;v]‖/‖b‖` in the product metric — one extra apply, on EVERY path,
    /// converged or not, and `0.0` (never NaN) when `p ≡ 0`.
    pub rel_residual: f64,
    /// `‖p_μ‖`. `0.0` ⇒ a structural zero row.
    pub grad_norm: f64,
    /// Which solver actually ran ([`DynamicSolver::Auto`] is resolved before the solve).
    pub solver_used: DynamicSolver,
    /// `‖v‖/‖u‖`. Exactly `0.0` at `ω = 0` by construction.
    pub v_weight: f64,
}

/// What [`SectorCtx::linear_response`] returns. No `values` field, for
/// [`SectorStaticResponse`]'s reason.
#[derive(Clone, Debug)]
pub struct SectorDynamicResponse {
    /// The frequency solved at.
    pub omega: f64,
    /// SYMMETRIZED `α^φ(ω)`, row-major `n_props²`, in the caller's order; out-of-channel rows and
    /// columns are EXACTLY zero.
    pub alpha: Vec<f64>,
    /// `max_{μ<ν}|raw[μ,ν] − raw[ν,μ]|` BEFORE symmetrization.
    pub alpha_defect: f64,
    /// Per-property diagnostics, same order.
    pub solves: Vec<SectorDynamicSolveInfo>,
    /// `dim T^φ`.
    pub dim: usize,
    /// The channel's irrep flux.
    pub phi: Irrep,
    /// `Γ0 ⊕ φ`.
    pub target: Irrep,
    /// `min_schmidt` at the base point.
    pub sigma_min: f64,
    /// `√(1 − ‖P^φ(μ̂x)‖²/‖μ̂x‖²)` per property — see [`SectorStaticResponse::capture_defect`].
    pub capture_defect: Vec<f64>,
}

impl SectorCtx<'_, '_, f64> {
    /// `α^φ_μν(ω) = 2⟨p_μ, u_ν(ω)⟩` on the channel: the coupled 2-block solve
    /// `K^φ(ω)[u;v] = [p;0]`, PCG strictly below the first channel pole and MINRES beyond it.
    ///
    /// # Errors
    ///
    /// [`SectorResponseError::WrongFlux`], [`SectorResponseError::NegativeOmega`],
    /// [`SectorResponseError::NotTtCritical`], [`SectorResponseError::NearResonance`],
    /// [`SectorResponseError::IndefiniteAbovePole`],
    /// [`SectorResponseError::SolveNotConverged`], and — at `ω = 0` only, where the system IS the
    /// static one — [`SectorResponseError::HessianIndefinite`]. A MINRES `γ = 0` breakdown
    /// (unreachable for a nonsingular system on the range of the operator) surfaces as
    /// `SolveNotConverged` with the freshly measured residual, because the caller's remedy is the
    /// same. An indefinite `K^φ(ω)` at `ω > 0` is NOT refused under MINRES: it is a legitimate
    /// solve there.
    ///
    /// # Panics
    ///
    /// On more than three properties, on `opts.route == PropRoute::Shifted`, or on an α
    /// symmetry defect beyond [`SectorDynamicOpts::sym_assert_rel`].
    pub fn linear_response(
        &self,
        props: &[PropertyOperator<f64>],
        omega: f64,
        opts: &SectorDynamicOpts,
    ) -> Result<SectorDynamicResponse, SectorResponseError> {
        assert_materialized(opts.route, "SectorCtx::linear_response");
        let covs = self.covectors(props, "SectorCtx::linear_response")?;
        self.debug_assert_unit_reference();
        if omega < 0.0 || omega.is_nan() {
            return Err(SectorResponseError::NegativeOmega { omega });
        }
        self.tt_critical_gate(opts.tt_grad_tol)?;
        let solver = self.select_solver(omega, opts)?;

        let n = props.len();
        let mut sols: Vec<SectorPair> = Vec::with_capacity(n);
        let mut solves: Vec<SectorDynamicSolveInfo> = Vec::with_capacity(n);
        for (nu, c) in covs.iter().enumerate() {
            let (pair, info) =
                self.solve_pair(&c.p, omega, opts, solver, props[nu].label.clone())?;
            sols.push(pair);
            solves.push(info);
        }

        let mut raw = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                raw[i * n + j] = 2.0 * covs[i].p.dot(&sols[j].u);
            }
        }
        let (alpha, alpha_defect) = symmetrize_alpha(
            &raw,
            n,
            opts.sym_assert_rel.unwrap_or(100.0 * opts.tol),
            "linear_response",
            omega,
        );

        Ok(SectorDynamicResponse {
            omega,
            alpha,
            alpha_defect,
            solves,
            dim: self.dim(),
            phi: self.phi(),
            target: self.space_out().target,
            sigma_min: self.sigma_min(),
            capture_defect: covs.iter().map(|c| c.capture_defect).collect(),
        })
    }

    /// [`Self::linear_response`]'s single-covector core, for a caller holding `p ∈ T^φ` directly
    /// (a hand-built probe, or a covector already projected).
    ///
    /// # Errors
    ///
    /// As [`Self::linear_response`], minus [`SectorResponseError::NotTtCritical`] and
    /// [`SectorResponseError::WrongFlux`].
    pub fn linear_response_solve(
        &self,
        p: &TtTangent<f64>,
        omega: f64,
        opts: &SectorDynamicOpts,
    ) -> Result<(SectorPair, SectorDynamicSolveInfo), SectorResponseError> {
        self.debug_assert_unit_reference();
        if omega < 0.0 || omega.is_nan() {
            return Err(SectorResponseError::NegativeOmega { omega });
        }
        let solver = self.select_solver(omega, opts)?;
        self.solve_pair(p, omega, opts, solver, "<covector>".to_string())
    }

    fn select_solver(
        &self,
        omega: f64,
        opts: &SectorDynamicOpts,
    ) -> Result<DynamicSolver, SectorResponseError> {
        select_solver_from(
            omega,
            opts.solver,
            opts.spectrum.as_deref(),
            opts.resonance_window,
            opts.unstable_pairs,
        )
        .map_err(|hit: NearResonanceHit| SectorResponseError::NearResonance {
            omega: hit.omega,
            omega_n: hit.omega_n,
            gap: hit.gap,
            window: hit.window,
        })
    }

    fn solve_pair(
        &self,
        p: &TtTangent<f64>,
        omega: f64,
        opts: &SectorDynamicOpts,
        solver: DynamicSolver,
        label: String,
    ) -> Result<(SectorPair, SectorDynamicSolveInfo), SectorResponseError> {
        let op = |pair: &SectorPair| {
            let cu = self.project(&pair.u.to_vector());
            let cv = self.project(&pair.v.to_vector());
            let mut hu = self.hpp_apply(&cu);
            hu.axpy(-omega, &cv);
            let mut hv = self.hmm_apply(&cv);
            hv.axpy(-omega, &cu);
            SectorPair { u: hu, v: hv }
        };
        let mut zero_v = p.clone();
        zero_v.scale_mut(0.0);
        let b = SectorPair {
            u: p.clone(),
            v: zero_v,
        };
        let bn = b.norm();
        if bn == 0.0 {
            let mut zero = b.clone();
            zero.scale(0.0);
            return Ok((
                zero,
                SectorDynamicSolveInfo {
                    label,
                    iters: 0,
                    rel_residual: 0.0,
                    grad_norm: 0.0,
                    solver_used: solver,
                    v_weight: 0.0,
                },
            ));
        }
        let (x, iters, refusal) = match solver {
            DynamicSolver::Minres | DynamicSolver::Auto => {
                let (x, flag, iters) = minres(op, &b, opts.tol, opts.max_iter);
                let refusal = match flag {
                    MinresFlag::Converged => None,
                    MinresFlag::MaxIter | MinresFlag::Breakdown { .. } => Some(iters),
                };
                (x, iters, RefusalKind::NotConverged(refusal))
            }
            DynamicSolver::Pcg => {
                let no_precond = |_: &mut SectorPair| {};
                let (x, flag, iters) = pcg_solve(op, no_precond, &b, opts.tol, opts.max_iter);
                let kind = match flag {
                    PcgFlag::Converged => RefusalKind::NotConverged(None),
                    PcgFlag::MaxIter => RefusalKind::NotConverged(Some(iters)),
                    PcgFlag::NegativeCurvature { rho } => RefusalKind::Curvature(rho),
                };
                (x, iters, kind)
            }
        };
        let mut r = b.clone();
        r.axpy(-1.0, &op(&x));
        let rel_residual = r.norm() / bn;
        if let Some(e) =
            sector_refusal_to_error(refusal, label.clone(), omega, iters, rel_residual, opts.tol)
        {
            return Err(e);
        }
        let un = x.u.norm();
        let v_weight = if un > 0.0 { x.v.norm() / un } else { 0.0 };
        Ok((
            x,
            SectorDynamicSolveInfo {
                label,
                iters,
                rel_residual,
                grad_norm: bn,
                solver_used: solver,
                v_weight,
            },
        ))
    }
}
