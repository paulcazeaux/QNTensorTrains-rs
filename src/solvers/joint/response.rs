// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::coeff::CoeffTensor;
use crate::core::tangent::TtTangent;
use crate::core::weingarten::WilliamsonError;
use crate::hamiltonian::{SparseHamiltonian, xt_h_y};
use crate::integrals::{Integrals, PropertyOperator, property_hamiltonian};
use crate::ops::dot;
use crate::rdm::transition_1rdm_h;
use crate::rotation::orbital_gradient;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;
use crate::solvers::hess::{
    PcgFlag, deflate_radial, pcg_solve, project_hmul_env, reproject_tangent,
};
use crate::solvers::minres::{MinresFlag, minres};

use super::{JointContext, JointVector};

/// Which representation the tangent covector leg is built through — the certification twin pair
/// of §4.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PropRoute {
    /// The layer's production route, resolved at the entry point: [`Self::Shifted`] on this
    /// (abelian joint) layer, [`Self::Materialized`] on the SU(2) joint layer and on the sector
    /// channels, which have no `Shifted` route. The default, so one options value serves every
    /// layer.
    #[default]
    Auto,
    /// `P((μ̂ − M⁰)·x)` through [`project_hmul_env`] — no `μ̂·x` train is ever formed. The abelian
    /// production path (§3, §4 spelling (i)).
    Shifted,
    /// `P(μ̂·x)` from a MATERIALIZED `μ̂·x`, then an explicit [`deflate_radial`] (§4 spelling (ii)).
    /// Materializing a one-body image costs an `O(d)` rank factor — the same whitelisted exception
    /// `κ̂x` takes in the parent module's covectors. This is the twin, not an alternative.
    Materialized,
}

/// Options for [`polarizability_static`].
#[derive(Clone, Debug)]
pub struct StaticResponseOpts {
    /// Relative residual tolerance of each PCG solve, measured in the JOINT metric (where a
    /// near-null orbital direction has norm `~√μ`). Default `1e-8`.
    pub tol: f64,
    /// Iteration cap per solve. Default `500`; CG terminates in at most `space_dim` exact steps.
    pub max_iter: usize,
    /// TT-criticality gate on `‖P((H−λ)x)‖`, RELATIVE to `max(1,|λ|)`. Default `1e-6`.
    pub tt_grad_tol: f64,
    /// Orbital-criticality gate on `‖Vᵀ g_L‖` (the KEPT rows). Default `1e-6`.
    pub orb_grad_tol: f64,
    /// Orbital-criticality gate on `‖g_L‖` over the FULL pair list. `None` ⇒ [`Self::orb_grad_tol`].
    /// Two-sided for the reason the optimizer's own gate is: the kept norm alone cannot see a
    /// gradient hiding in the dropped quotient directions.
    pub orb_grad_tol_full: Option<f64>,
    /// Hard-assert bound on the α symmetry defect, RELATIVE to `max(1, max|α|)`. `None` (the
    /// default) means `100.0 · tol` — recomputed from the ACTUAL [`Self::tol`] at solve time, not
    /// frozen at `Default::default()`'s `tol`. This must track `tol` because Rust's struct-update
    /// syntax (`StaticResponseOpts { tol: 1e-5, ..Default::default() }`) evaluates
    /// `Default::default()` first and then only overwrites the named fields: a plain `f64` default
    /// bound to `1e-8` would silently stay stale under a loosened `tol`, and by this same
    /// module's §8 bound (solve inexactness is first order in the residuals, ~`2·tol·‖u‖`) a
    /// caller-recoverable, perfectly healthy loosened-tol solve would then trip the hard assert
    /// below and get blamed for a broken assembly it does not have. `Some(_)` pins an explicit
    /// bound (looser OR tighter than `100·tol`) and is honored as given. This certifies OUR OWN
    /// assembly, hence an assert and not an `Err`.
    pub sym_assert_rel: Option<f64>,
    /// Which route builds the tangent covector leg. Default [`PropRoute::Auto`].
    pub route: PropRoute,
}

impl Default for StaticResponseOpts {
    fn default() -> Self {
        let tol = 1e-8;
        StaticResponseOpts {
            tol,
            max_iter: 500,
            tt_grad_tol: 1e-6,
            orb_grad_tol: 1e-6,
            orb_grad_tol_full: None,
            sym_assert_rel: None,
            route: PropRoute::Auto,
        }
    }
}

/// Per-property solve diagnostics.
#[derive(Clone, Debug)]
pub struct ResponseSolveInfo {
    /// The property's [`PropertyOperator::label`], echoed so a caller reading `solves` alone can
    /// tell which row it is looking at.
    pub label: String,
    /// CG iterations taken (`= hpp_apply` calls inside the solve).
    pub iters: usize,
    /// `‖b − (A+B)u‖_G / ‖b‖_G`, recomputed with ONE fresh apply after the solve rather than read
    /// off CG's recursively updated residual — `0.0` when `g ≡ 0`. The fresh value is the honest
    /// one; it may sit slightly above [`StaticResponseOpts::tol`] where the recursion drifted.
    pub rel_residual: f64,
    /// `‖g_μ‖_G`, the raised covector's joint norm. `0.0` ⇒ the property is invisible on this
    /// manifold and its whole α row and column are exactly zero.
    pub grad_norm: f64,
    /// `‖(0, g.c)‖²_G / ‖g‖²_G` — how much of this property's response is orbital relaxation.
    /// `0.0` at `n_eff == 0`.
    pub orbital_weight: f64,
}

/// What [`polarizability_static`] returns.
#[derive(Clone, Debug)]
pub struct StaticResponse {
    /// The SYMMETRIZED polarizability, row-major `n_props²`, in the caller's property order.
    pub alpha: Vec<f64>,
    /// `max_{μ<ν} |raw[μ,ν] − raw[ν,μ]|` BEFORE symmetrization, with
    /// `raw[μ,ν] = 2⟨g_μ, u_ν⟩` (§8). `0.0` for a single property.
    pub alpha_defect: f64,
    /// `⟨x|μ̂_ν|x⟩ / ⟨x,x⟩` at zero field, same order.
    pub values: Vec<f64>,
    /// Per-property diagnostics, same order.
    pub solves: Vec<ResponseSolveInfo>,
    /// `n_eff` at the context — `0` means the whole response is TT-only (full-rank degeneration).
    pub n_eff: usize,
}

/// Why [`polarizability_static`] refused.
#[derive(Clone, Debug)]
pub enum ResponseError {
    /// The reference is not a joint-critical point: either the TT gradient or the orbital gradient
    /// (kept and/or full) exceeds its bound. Both sides' measurements are reported, whichever one
    /// fired.
    NotJointCritical {
        tt_gradnorm: f64,
        tt_bound: f64,
        orb_grad_kept: f64,
        orb_grad_full: f64,
        orb_bound: f64,
        lambda: f64,
    },
    /// `A+B` is indefinite on the deflated space, so the reference is not a stable minimum and
    /// `2gᵀ(A+B)⁻¹g` is not a polarizability. **A refusal, never a fudge** — see §5's recorded
    /// negative on shifted solves.
    HessianIndefinite {
        label: String,
        rho: f64,
        iters: usize,
    },
    /// The joint Hessian is not symplectically stable over this basis (`H^rr ⊁ 0`, or `Ω² ⊀ 0`), so
    /// the reduced Williamson basis — and therefore every transition moment — does not exist.
    SymplecticUnstable { reason: WilliamsonError },
    /// `ω < 0` was requested. `α(ω)` is EVEN in `ω` (the closed form of §11 carries `ω²` only), so
    /// this is a caller-side sign slip rather than a regime: call with `|ω|`.
    NegativeOmega { omega: f64 },
    /// `ω` fell inside the caller-set exclusion window of a symplectic eigenvalue. §11: the
    /// solution amplitude carries `1/[2(ω_n−ω)]` there, so what comes back is a large number with
    /// no error bar — refused, exactly as §5 refuses a shifted static solve.
    NearResonance {
        omega: f64,
        omega_n: f64,
        gap: f64,
        window: f64,
    },
    /// PCG met negative curvature at `ω > 0`. **Not** [`ResponseError::HessianIndefinite`]: by
    /// §11's exact inertia count this is a PROOF that `ω > ω₁`, not evidence that the reference is
    /// unstable, and the remedy is a solver change rather than a re-minimization.
    IndefiniteAbovePole {
        label: String,
        omega: f64,
        rho: f64,
        iters: usize,
    },
    /// The CG hit its iteration cap. `α` would still be a variational LOWER bound (§2's C2), but
    /// of unknown tightness, so it is refused rather than published.
    SolveNotConverged {
        label: String,
        iters: usize,
        rel_residual: f64,
        tol: f64,
    },
}

impl std::fmt::Display for ResponseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResponseError::NotJointCritical {
                tt_gradnorm,
                tt_bound,
                orb_grad_kept,
                orb_grad_full,
                orb_bound,
                lambda,
            } => write!(
                f,
                "polarizability_static: the reference is not JOINT-critical — ‖P((H−λ)x)‖ = \
                 {tt_gradnorm:.6e} against {tt_bound:.6e} (tt_grad_tol · max(1,|λ|), λ = \
                 {lambda:.6e}), ‖Vᵀg_L‖ = {orb_grad_kept:.6e} and ‖g_L‖ = {orb_grad_full:.6e} \
                 against {orb_bound:.6e}. The second-order expansion α = 2·gᵀ(A+B)⁻¹g eliminates \
                 the displacement at a STATIONARY point; with a residual gradient the minimizer \
                 is not at v = 0 and α is wrong at first order in ‖grad‖. The TT leg is checked \
                 here and not inherited from joint_context, because a diagnosis-only context \
                 (tt_grad_tol = INFINITY) passes that gate vacuously while its B block — which \
                 hpp_apply uses — is invalid. Remedy: converge the reference (joint_ground_state \
                 reaches ‖Vᵀg_L‖ ~ 1e-7 on the shipped fixtures), not a looser tolerance."
            ),
            ResponseError::HessianIndefinite { label, rho, iters } => write!(
                f,
                "polarizability_static: A+B is indefinite on the deflated joint space at property \
                 {label:?} — the CG met curvature ⟨p,(A+B)p⟩/‖p‖² = {rho:.6e} ≤ 0 after {iters} \
                 iterations. The static polarizability is the second derivative of a MINIMUM; at \
                 an indefinite reference 2·gᵀ(A+B)⁻¹g is not a polarizability and no amount of \
                 iterating makes it one. A Levenberg-shifted (A+B+μ)⁻¹ is NOT an acceptable \
                 degradation path here (it is legal for a step direction, illegal for a response \
                 amplitude: it returns a systematically small α with no error bar). Remedy: the \
                 reference is not a minimum — re-minimize at higher rank, at a lower orb_floor, \
                 or on the complex layer."
            ),
            ResponseError::SymplecticUnstable { reason } => write!(
                f,
                "transition_moments: the joint Hessian is not symplectically stable over this \
                 basis — the reduced Williamson kernel refused with {reason:?}. There is no \
                 (ω, S_r, S_i) to contract a moment against, and 2·Σ T²/ω is not a polarizability \
                 at an unstable reference. Note the guard is STRICTER than the scalar path's (it \
                 forms Ω^{{−½}} and so cannot clamp a marginal negative to zero), so a reference \
                 that joint_symplectic_eigenvalues calls Stable with ω₁ ≈ 0 refuses here. Remedy: \
                 diagnose with joint_symplectic_eigenvalues (its Unstable arm carries the \
                 generator eigenvalues), then re-minimize — a clamped ω₁ = 0 would be divided by."
            ),
            ResponseError::NegativeOmega { omega } => write!(
                f,
                "linear_response: ω = {omega:.6e} < 0. α(ω) is EVEN in ω — the closed form \
                 2·Σ T²·ω_n/(ω_n²−ω²) carries ω² only — so a negative frequency is a caller-side \
                 sign slip, not a regime this function could serve differently. Remedy: call with \
                 |ω|."
            ),
            ResponseError::NearResonance {
                omega,
                omega_n,
                gap,
                window,
            } => write!(
                f,
                "linear_response: ω = {omega:.6e} is inside the exclusion window of the symplectic \
                 eigenvalue ω_n = {omega_n:.6e} — gap {gap:.6e} against {:.6e} \
                 (resonance_window · max(1, ω_n), resonance_window = {window:.6e}). The n-th \
                 component of the solution carries ω_n/[(ω_n−ω)(ω_n+ω)] ≈ 1/[2(ω_n−ω)], so what \
                 a solve returns there is a large number with no error bar; a regularized or \
                 shifted solve is NOT an acceptable degradation path for a response amplitude. \
                 Remedy: move ω off the pole, narrow resonance_window deliberately if the caller \
                 owns the error bar, or read the residue directly — the transition moment T_n IS \
                 the pole's residue (α ≈ 2T_n²·ω_n/[(ω_n−ω)(ω_n+ω)] there), and \
                 transition_moments computes it without a solve.",
                window * omega_n.abs().max(1.0)
            ),
            ResponseError::IndefiniteAbovePole {
                label,
                omega,
                rho,
                iters,
            } => write!(
                f,
                "linear_response: PCG met curvature ⟨p,K(ω)p⟩/‖p‖² = {rho:.6e} ≤ 0 at property \
                 {label:?}, ω = {omega:.6e}, after {iters} iterations. This is not a broken \
                 reference and not an unstable minimum: by the exact inertia count \
                 #{{negative eigenvalues of K(ω)}} = #{{n : ω_n < ω}}, it is a PROOF that ω lies \
                 above the first symplectic eigenvalue, where K(ω) is symmetric INDEFINITE and CG \
                 does not apply. Remedy: use DynamicSolver::Minres, or supply \
                 DynamicResponseOpts::spectrum so DynamicSolver::Auto picks it — not a shift, and \
                 not a re-minimization."
            ),
            ResponseError::SolveNotConverged {
                label,
                iters,
                rel_residual,
                tol,
            } => write!(
                f,
                "polarizability_static: the response solve for property {label:?} did not \
                 converge — ‖r‖_G/‖g‖_G = {rel_residual:.6e} against {tol:.6e} after {iters} \
                 iterations. CG on this SPD system is a MONOTONE lower bound (α − 2gᵀu_k = \
                 2‖u−u_k‖²_H ≥ 0), so the partial α is a lower bound of unknown tightness and is \
                 refused rather than published. Remedy: raise max_iter, enable the Jacobi \
                 preconditioner, or improve the conditioning (the kept orbital spectrum sets \
                 cond(A+B); a higher orb_floor drops the near-null tail)."
            ),
        }
    }
}

impl std::error::Error for ResponseError {}

/// `t' = t + f·m` entrywise, every other field cloned — the one-body field perturbation a
/// finite-difference response oracle ingests.
///
/// # Panics
///
/// If `m` or `ints.t` is not a rank-2 `norb × norb` tensor.
pub fn with_field<T: Scalar>(ints: &Integrals<T>, m: &CoeffTensor<T>, f: T) -> Integrals<T> {
    let d = ints.norb;
    assert!(
        ints.t.rank == 2 && ints.t.d == d,
        "with_field: ints.t must be rank-2 over d={d} (got rank {}, d {})",
        ints.t.rank,
        ints.t.d
    );
    assert!(
        m.rank == 2 && m.d == d,
        "with_field: the property matrix must be rank-2 over d={d} (got rank {}, d {})",
        m.rank,
        m.d
    );
    let mut t = ints.t.clone();
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], ints.t.get(&[i, j]) + f * m.get(&[i, j]));
        }
    }
    Integrals {
        t,
        v: ints.v.clone(),
        e0: ints.e0,
        norb: ints.norb,
        nelec: ints.nelec,
        ms2: ints.ms2,
        orbsym: ints.orbsym.clone(),
        isym: ints.isym,
    }
}

/// `M⁰ = ⟨x|μ̂|x⟩ / ⟨x,x⟩` at the context's base point.
pub fn property_value(ctx: &JointContext<'_>, mu: &SparseHamiltonian<f64>) -> f64 {
    let x = &ctx.x;
    let nrm2 = dot(x, x);
    debug_assert!(
        (nrm2 - 1.0).abs() < 1e-10,
        "property_value: the joint stack assumes ‖x‖ = 1 (got ⟨x,x⟩ = {nrm2:e})"
    );
    xt_h_y(x, mu, x) / nrm2
}

/// The RAISED, radial-deflated joint-manifold gradient covector `g_μ` of `⟨μ̂⟩`, in the
/// ½-convention: `⟨g_μ, w⟩_G = ½·dM/dw` for every joint direction `w` (§2's Fact 2, §3's two
/// components).
pub fn property_gradient(ctx: &JointContext<'_>, mu: &SparseHamiltonian<f64>) -> JointVector {
    property_gradient_with(ctx, mu, PropRoute::Shifted)
}

/// [`property_gradient`] with the tangent-leg route named — the certification twin entry point.
pub fn property_gradient_with(
    ctx: &JointContext<'_>,
    mu: &SparseHamiltonian<f64>,
    route: PropRoute,
) -> JointVector {
    let (base_l, base_r) = ctx.shared_bases();
    let m0 = property_value(ctx, mu);
    let radial = ctx.radial();

    let mut f_xi = match route {
        PropRoute::Auto | PropRoute::Shifted => project_hmul_env(mu, base_l, base_r, m0, &ctx.x),
        PropRoute::Materialized => {
            let mx = mu * &ctx.x;
            TtTangent::project_shared(base_l, base_r, &mx)
        }
    };
    deflate_radial(radial, &mut f_xi);

    let gam = transition_1rdm_h(&ctx.x, mu, &ctx.x);
    let fc_full: Vec<f64> = ctx
        .pairs()
        .iter()
        .map(|&(p, q)| gam.get(&[p, q]) - gam.get(&[q, p]))
        .collect();

    let mut g = ctx.joint_raise(f_xi, ctx.vt_times(&fc_full));
    deflate_radial(radial, &mut g.xi);
    g
}

/// The static polarizability matrix `α_μν = 2·g_μᵀ(A+B)⁻¹g_ν` on the joint manifold (§2).
///
/// # Errors
///
/// [`ResponseError::NotJointCritical`] (checked before any solve),
/// [`ResponseError::HessianIndefinite`], [`ResponseError::SolveNotConverged`].
///
/// # Panics
///
/// Through [`property_hamiltonian`], on a property that is not rank-2 `d×d`, is not symmetric, or
/// carries a cross-irrep component. On an α symmetry defect beyond
/// [`StaticResponseOpts::sym_assert_rel`] — that certifies this module's own assembly.
pub fn polarizability_static(
    ctx: &JointContext<'_>,
    props: &[PropertyOperator<f64>],
    opts: &StaticResponseOpts,
) -> Result<StaticResponse, ResponseError> {
    let mus: Vec<SparseHamiltonian<f64>> = props
        .iter()
        .map(|p| property_hamiltonian(ctx.x.space, p))
        .collect();

    joint_critical_gate(
        ctx,
        opts.tt_grad_tol,
        opts.orb_grad_tol,
        opts.orb_grad_tol_full,
    )?;

    let n = props.len();
    let values: Vec<f64> = mus.iter().map(|mu| property_value(ctx, mu)).collect();
    let grads: Vec<JointVector> = mus
        .iter()
        .map(|mu| property_gradient_with(ctx, mu, opts.route))
        .collect();

    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let op = |p: &JointVector| {
        let mut q = p.clone();
        reproject_tangent(base_l, base_r, radial, &mut q.xi);
        let mut hq = ctx.hpp_apply(&q);
        deflate_radial(radial, &mut hq.xi);
        hq
    };
    let jacobi: Option<JointVector> = None;
    let precond = |v: &mut JointVector| {
        if let Some(d) = jacobi.as_ref() {
            v.zip_coeffs_mut(d, |s, o| *s *= o);
        }
    };
    let mut sols: Vec<JointVector> = Vec::with_capacity(n);
    let mut solves: Vec<ResponseSolveInfo> = Vec::with_capacity(n);
    for (nu, g) in grads.iter().enumerate() {
        let label = props[nu].label.clone();
        let mut b = g.clone();
        reproject_tangent(base_l, base_r, radial, &mut b.xi);
        let bn = b.norm();
        let orbital_weight = orbital_weight_of(ctx, g, bn);

        if bn == 0.0 {
            let mut zero = g.clone();
            zero.scale(0.0);
            sols.push(zero);
            solves.push(ResponseSolveInfo {
                label,
                iters: 0,
                rel_residual: 0.0,
                grad_norm: 0.0,
                orbital_weight: 0.0,
            });
            continue;
        }

        let (u, flag, iters) = pcg_solve(op, precond, &b, opts.tol, opts.max_iter);
        let mut r = b.clone();
        r.axpy(-1.0, &op(&u));
        let rel_residual = r.norm() / bn;

        if let Some(err) = flag_to_error(&label, flag, iters, rel_residual, opts.tol) {
            return Err(err);
        }
        sols.push(u);
        solves.push(ResponseSolveInfo {
            label,
            iters,
            rel_residual,
            grad_norm: bn,
            orbital_weight,
        });
    }

    let mut raw = vec![0.0f64; n * n];
    for mu_i in 0..n {
        for nu_j in 0..n {
            raw[mu_i * n + nu_j] = 2.0 * grads[mu_i].dot(&sols[nu_j]);
        }
    }
    let mut alpha_defect = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            alpha_defect = alpha_defect.max((raw[i * n + j] - raw[j * n + i]).abs());
        }
    }
    let raw_scale = raw.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
    let sym_assert_rel = opts.sym_assert_rel.unwrap_or(100.0 * opts.tol);
    assert!(
        alpha_defect <= sym_assert_rel * raw_scale,
        "polarizability_static: α is not symmetric — defect {alpha_defect:.6e} exceeds \
         {:.6e} (sym_assert_rel · max(1,max|α|), max|α| = {raw_scale:.6e}). Solve inexactness is \
         first order in the residuals and bounded by ~2·tol·‖u‖, two decades below this bound; a \
         defect this large means the joint apply is not self-adjoint in the joint metric or the \
         raise is broken — OUR assembly, not the caller's input, hence an assert.",
        sym_assert_rel * raw_scale
    );
    let mut alpha = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            alpha[i * n + j] = 0.5 * (raw[i * n + j] + raw[j * n + i]);
        }
    }

    Ok(StaticResponse {
        alpha,
        alpha_defect,
        values,
        solves,
        n_eff: ctx.n_eff(),
    })
}

/// Options for [`transition_moments`].
#[derive(Clone, Debug)]
pub struct TransitionMomentOpts {
    /// TT-criticality gate, RELATIVE to `max(1,|λ|)` — the same gate and the same default
    /// (`1e-6`) [`StaticResponseOpts::tt_grad_tol`] carries, for the same reason: the moments
    /// consume `B` through the Grams.
    pub tt_grad_tol: f64,
    /// Orbital-criticality gate on `‖Vᵀ g_L‖` (the KEPT rows). Default `1e-6`.
    pub orb_grad_tol: f64,
    /// Orbital-criticality gate on `‖g_L‖` over the FULL pair list. `None` ⇒ [`Self::orb_grad_tol`].
    pub orb_grad_tol_full: Option<f64>,
    /// Which route builds the tangent covector leg. Default [`PropRoute::Auto`].
    pub route: PropRoute,
    /// RELATIVE window grouping `ω_n` into degenerate clusters: `n` joins the previous cluster
    /// when `ω_n − ω_{n−1} ≤ degeneracy_tol · max(1, ω_n)`. Default `1e-8`.
    pub degeneracy_tol: f64,
}

impl Default for TransitionMomentOpts {
    fn default() -> Self {
        TransitionMomentOpts {
            tt_grad_tol: 1e-6,
            orb_grad_tol: 1e-6,
            orb_grad_tol_full: None,
            route: PropRoute::Auto,
            degeneracy_tol: 1e-8,
        }
    }
}

/// What [`transition_moments`] returns. Every field is quoted in the caller's `props` order.
#[derive(Clone, Debug)]
pub struct TransitionMoments {
    /// The symplectic (excitation) frequencies, ASCENDING — `ω_n = E_n − E_0` at full rank.
    pub omega: Vec<f64>,
    /// `moments[p][n] = T_n^p = Σ_i γ_i·s_r[i*n + n]` (§10 ★★). **Only `|T_n|` and same-`n`
    /// products across two properties are observable**; a bare signed value is gauge.
    pub moments: Vec<Vec<f64>>,
    /// [`PropertyOperator::label`], echoed so a caller reading `moments` alone can tell the rows
    /// apart.
    pub labels: Vec<String>,
    /// `⟨x|μ̂_p|x⟩/⟨x,x⟩` at zero field (mirrors [`StaticResponse::values`]).
    pub values: Vec<f64>,
    /// `alpha_sos[p*P+q] = 2 Σ_n T_n^p T_n^q / ω_n` — the sum-over-states reconstruction of the
    /// STATIC polarizability. An ALGEBRAIC identity with [`polarizability_static`] at any rank
    /// (§10), hence the mutual pin; the three terms of its error budget are
    /// [`Self::completeness_defect`], [`Self::orth_error`] and the static solve's own residual.
    pub alpha_sos: Vec<f64>,
    /// `‖g_p − Σ_i γ_i e_i‖_G / ‖g_p‖_G`: how much of the covector the basis missed. Normally the
    /// DOMINANT term of the SOS-vs-solve budget. `0.0` when `‖g_p‖ = 0`.
    pub completeness_defect: Vec<f64>,
    /// [`super::JointContext::orth_error`] — the basis's departure-from-orthonormality budget.
    pub orth_error: f64,
    /// [`super::JointContext::sigma_min`] at the base point. Quoted beside everything:
    /// the bond-gauge inversion in any tangent round trip amplifies roundoff by `1/σ_min`.
    pub sigma_min: f64,
    /// Half-open index ranges `[lo,hi)` of degenerate `ω` clusters. A cluster of size > 1 has NO
    /// per-mode moment (§10); only `Σ_{n∈cluster} T_n²` is defined. Singletons are listed too, so
    /// the ranges partition `0..omega.len()`.
    pub degenerate_clusters: Vec<(usize, usize)>,
    /// `n_eff` at the context — `0` means full rank, where `B ≡ 0` and the moments are exactly the
    /// CI transition moments.
    pub n_eff: usize,
}

/// The transition moments `T_n^μ = ⟨Ψ_n|μ̂|Ψ_0⟩` (at full rank) of every property in `props`, from
/// the joint Hessian's reduced Williamson basis over `basis` (§10).
///
/// # Errors
///
/// [`ResponseError::NotJointCritical`] (checked before any apply) and
/// [`ResponseError::SymplecticUnstable`].
///
/// # Panics
///
/// Through [`property_hamiltonian`], on a property that is not rank-2 `d×d`, is not symmetric, or
/// carries a cross-irrep component.
pub fn transition_moments(
    ctx: &JointContext<'_>,
    basis: &[JointVector],
    props: &[PropertyOperator<f64>],
    opts: &TransitionMomentOpts,
) -> Result<TransitionMoments, ResponseError> {
    let mus: Vec<SparseHamiltonian<f64>> = props
        .iter()
        .map(|p| property_hamiltonian(ctx.x.space, p))
        .collect();

    joint_critical_gate(
        ctx,
        opts.tt_grad_tol,
        opts.orb_grad_tol,
        opts.orb_grad_tol_full,
    )?;

    let values: Vec<f64> = mus.iter().map(|mu| property_value(ctx, mu)).collect();
    let w = ctx
        .joint_williamson(basis)
        .map_err(|reason| ResponseError::SymplecticUnstable { reason })?;
    let n = w.omega.len();
    let n_rows = w.n;

    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let p_count = props.len();
    let mut moments: Vec<Vec<f64>> = Vec::with_capacity(p_count);
    let mut completeness_defect: Vec<f64> = Vec::with_capacity(p_count);
    for mu in mus.iter() {
        let mut g = property_gradient_with(ctx, mu, opts.route);
        reproject_tangent(base_l, base_r, radial, &mut g.xi);
        let gn = g.norm();

        let gamma: Vec<f64> = basis.iter().map(|e| e.dot(&g)).collect();
        let defect = if gn > 0.0 {
            let mut r = g.clone();
            for (i, e) in basis.iter().enumerate() {
                r.axpy(-gamma[i], e);
            }
            r.norm() / gn
        } else {
            0.0
        };
        completeness_defect.push(defect);

        let mut t = vec![0.0f64; n];
        debug_assert_eq!(
            w.s_r.len(),
            n_rows * n,
            "s_r must be n_rows × n_mode row-major"
        );
        for (i, &gi) in gamma.iter().enumerate() {
            if gi == 0.0 {
                continue;
            }
            for (col, tc) in t.iter_mut().enumerate() {
                *tc += gi * w.s_r[i * n + col];
            }
        }
        moments.push(t);
    }

    let mut alpha_sos = vec![0.0f64; p_count * p_count];
    for (i, ti) in moments.iter().enumerate() {
        for (j, tj) in moments.iter().enumerate() {
            let acc: f64 = ti
                .iter()
                .zip(tj.iter())
                .zip(w.omega.iter())
                .map(|((&a, &b), &wn)| a * b / wn)
                .sum();
            alpha_sos[i * p_count + j] = 2.0 * acc;
        }
    }

    let mut degenerate_clusters: Vec<(usize, usize)> = Vec::new();
    if n > 0 {
        let mut lo = 0usize;
        for k in 1..n {
            if w.omega[k] - w.omega[k - 1] > opts.degeneracy_tol * w.omega[k].abs().max(1.0) {
                degenerate_clusters.push((lo, k));
                lo = k;
            }
        }
        degenerate_clusters.push((lo, n));
    }

    Ok(TransitionMoments {
        omega: w.omega,
        moments,
        labels: props.iter().map(|p| p.label.clone()).collect(),
        values,
        alpha_sos,
        completeness_defect,
        orth_error: w.orth_error,
        sigma_min: ctx.sigma_min(),
        degenerate_clusters,
        n_eff: ctx.n_eff(),
    })
}

/// `f_n = (2/3)·ω_n·Σ_c (T_n^c)²` — dipole oscillator strengths in the length gauge, atomic units
/// (§10).
///
/// # Panics
///
/// If `moments_xyz` has more than three rows, or any row's length differs from `omega`'s.
pub fn oscillator_strengths(omega: &[f64], moments_xyz: &[Vec<f64>]) -> Vec<f64> {
    assert!(
        moments_xyz.len() <= 3,
        "oscillator_strengths: at most three Cartesian components, got {} — the 2/3 is the \
         isotropic average over x, y, z and has no meaning beyond them",
        moments_xyz.len()
    );
    for (c, row) in moments_xyz.iter().enumerate() {
        assert_eq!(
            row.len(),
            omega.len(),
            "oscillator_strengths: component {c} has {} moments against {} frequencies — all \
             components must come from ONE transition_moments call over ONE basis",
            row.len(),
            omega.len()
        );
    }
    omega
        .iter()
        .enumerate()
        .map(|(n, &w)| {
            let s: f64 = moments_xyz.iter().map(|row| row[n] * row[n]).sum();
            (2.0 / 3.0) * w * s
        })
        .collect()
}

/// `[u; v]` — the unknown of `K(ω)·[u;v] = [g;0]` (§11), an [`InnerProductVector`] in the PRODUCT
/// metric `⟨(u₁,v₁),(u₂,v₂)⟩ = ⟨u₁,u₂⟩_G + ⟨v₁,v₂⟩_G`, in which `K(ω)` is self-adjoint.
#[derive(Clone)]
pub struct JointPair {
    /// The `u = X+Y` leg.
    pub u: JointVector,
    /// The `v = X−Y` leg. Identically zero at `ω = 0` (§11).
    pub v: JointVector,
}

impl InnerProductVector for JointPair {
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
        self.u.scale(a);
        self.v.scale(a);
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

/// Which Krylov solver runs the `K(ω)` system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DynamicSolver {
    /// Pick from the spectrum: PCG strictly below `ω₁` (where `K(ω) ≻ 0` by §11's inertia count),
    /// MINRES otherwise. **Without a spectrum, `Auto` falls back to MINRES**, which is always
    /// valid — never to PCG, which would be a guess about definiteness.
    #[default]
    Auto,
    /// Force PCG. Legal only below the first pole; above it the negative-curvature verdict becomes
    /// [`ResponseError::IndefiniteAbovePole`], which is a proof, not a failure.
    Pcg,
    /// Force MINRES. Always valid; slower than PCG where PCG applies.
    Minres,
}

/// Options for [`linear_response`] and [`linear_response_solve`].
#[derive(Clone, Debug)]
pub struct DynamicResponseOpts {
    /// Relative residual tolerance of each solve, in the PRODUCT metric. Default `1e-8`.
    pub tol: f64,
    /// Iteration cap per solve. Default `500`.
    pub max_iter: usize,
    /// Which solver. Default [`DynamicSolver::Auto`].
    pub solver: DynamicSolver,
    /// TT-criticality gate, RELATIVE to `max(1,|λ|)`. Default `1e-6`.
    pub tt_grad_tol: f64,
    /// Orbital-criticality gate on the KEPT rows. Default `1e-6`.
    pub orb_grad_tol: f64,
    /// Orbital-criticality gate over the FULL pair list. `None` ⇒ [`Self::orb_grad_tol`].
    pub orb_grad_tol_full: Option<f64>,
    /// Hard-assert bound on the α symmetry defect, RELATIVE to `max(1, max|α|)`. `None` ⇒
    /// `100·tol`, recomputed at solve time — see [`StaticResponseOpts::sym_assert_rel`] for why it
    /// must not be frozen inside `Default::default()`.
    pub sym_assert_rel: Option<f64>,
    /// Which route builds the tangent covector leg. Default [`PropRoute::Auto`].
    pub route: PropRoute,
    /// RELATIVE pole-exclusion window: a solve is refused when
    /// `min_n |ω − ω_n| ≤ resonance_window · max(1, ω_n)`. Default `1e-3`. Consulted only when
    /// [`Self::spectrum`] is supplied.
    pub resonance_window: f64,
    /// The symplectic spectrum, if the caller has it (from
    /// [`super::JointContext::joint_symplectic_eigenvalues`] or
    /// [`super::JointContext::joint_williamson`]).
    pub spectrum: Option<Vec<f64>>,
    /// How many IMAGINARY modes the reference carries
    /// ([`crate::solvers::hess::SymplecticSpectrum::unstable_pairs`]). Default `0`.
    pub unstable_pairs: usize,
}

impl Default for DynamicResponseOpts {
    fn default() -> Self {
        DynamicResponseOpts {
            tol: 1e-8,
            max_iter: 500,
            solver: DynamicSolver::Auto,
            tt_grad_tol: 1e-6,
            orb_grad_tol: 1e-6,
            orb_grad_tol_full: None,
            sym_assert_rel: None,
            route: PropRoute::Auto,
            resonance_window: 1e-3,
            spectrum: None,
            unstable_pairs: 0,
        }
    }
}

/// Per-property diagnostics of an ω-dependent solve.
#[derive(Clone, Debug)]
pub struct DynamicSolveInfo {
    /// The property's [`PropertyOperator::label`].
    pub label: String,
    /// Iterations taken (`= 2·iters` raised applies, one `hpp` and one `hmm` each).
    pub iters: usize,
    /// FRESH `‖b − K(ω)·[u;v]‖ / ‖b‖` in the product metric — ONE extra apply after the solve,
    /// never the recursion's own value. Reported on EVERY path, converged or not, and `0.0`
    /// (never `NaN`) when `g ≡ 0`.
    pub rel_residual: f64,
    /// `‖g_μ‖_G`, the raised covector's joint norm. `0.0` ⇒ the property is invisible on this
    /// manifold and its whole α row and column are exactly zero.
    pub grad_norm: f64,
    /// Which solver actually ran ([`DynamicSolver::Auto`] is resolved before the solve).
    pub solver_used: DynamicSolver,
    /// `‖v‖ / ‖u‖`. Exactly `0.0` at `ω = 0` by construction (§11); elsewhere a diagnosis of how
    /// much of the response is the antisymmetric `X−Y` component.
    pub v_weight: f64,
}

/// What [`linear_response`] returns.
#[derive(Clone, Debug)]
pub struct DynamicResponse {
    /// The frequency solved at.
    pub omega: f64,
    /// The SYMMETRIZED `α(ω)`, row-major `n_props²`, in the caller's property order.
    pub alpha: Vec<f64>,
    /// `max_{μ<ν} |raw[μ,ν] − raw[ν,μ]|` BEFORE symmetrization (§8's discipline, unchanged).
    pub alpha_defect: f64,
    /// `⟨x|μ̂_ν|x⟩/⟨x,x⟩` at zero field, same order.
    pub values: Vec<f64>,
    /// Per-property diagnostics, same order.
    pub solves: Vec<DynamicSolveInfo>,
    /// `n_eff` at the context.
    pub n_eff: usize,
}

/// The frequency-dependent polarizability `α_μν(ω) = 2⟨g_μ, u_ν(ω)⟩_G` on the joint manifold
/// (§11).
///
/// # Errors
///
/// [`ResponseError::NegativeOmega`], [`ResponseError::NotJointCritical`],
/// [`ResponseError::NearResonance`], [`ResponseError::IndefiniteAbovePole`],
/// [`ResponseError::SolveNotConverged`], and — at `ω = 0` only, where the system really is the
/// static one — [`ResponseError::HessianIndefinite`].
///
/// A MINRES `γ = 0` breakdown (unreachable for a nonsingular system on the range of the operator)
/// surfaces as [`ResponseError::SolveNotConverged`] carrying the freshly measured residual: the
/// caller's remedy is the same, and a distinct variant for a state no fixture reaches would be
/// unexercised surface.
///
/// # Panics
///
/// Through [`property_hamiltonian`], as [`polarizability_static`] does. On an α symmetry defect
/// beyond [`DynamicResponseOpts::sym_assert_rel`] — that certifies this module's own assembly.
pub fn linear_response(
    ctx: &JointContext<'_>,
    props: &[PropertyOperator<f64>],
    omega: f64,
    opts: &DynamicResponseOpts,
) -> Result<DynamicResponse, ResponseError> {
    let mus: Vec<SparseHamiltonian<f64>> = props
        .iter()
        .map(|p| property_hamiltonian(ctx.x.space, p))
        .collect();

    if omega < 0.0 || omega.is_nan() {
        return Err(ResponseError::NegativeOmega { omega });
    }
    joint_critical_gate(
        ctx,
        opts.tt_grad_tol,
        opts.orb_grad_tol,
        opts.orb_grad_tol_full,
    )?;

    let solver = select_solver(omega, opts)?;

    let n = props.len();
    let values: Vec<f64> = mus.iter().map(|mu| property_value(ctx, mu)).collect();
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let grads: Vec<JointVector> = mus
        .iter()
        .map(|mu| {
            let mut g = property_gradient_with(ctx, mu, opts.route);
            reproject_tangent(base_l, base_r, radial, &mut g.xi);
            g
        })
        .collect();

    let mut sols: Vec<JointPair> = Vec::with_capacity(n);
    let mut solves: Vec<DynamicSolveInfo> = Vec::with_capacity(n);
    for (nu, g) in grads.iter().enumerate() {
        let (pair, info) = solve_pair(ctx, g, omega, opts, solver, props[nu].label.clone())?;
        sols.push(pair);
        solves.push(info);
    }

    let mut raw = vec![0.0f64; n * n];
    for mu_i in 0..n {
        for nu_j in 0..n {
            raw[mu_i * n + nu_j] = 2.0 * grads[mu_i].dot(&sols[nu_j].u);
        }
    }
    let mut alpha_defect = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            alpha_defect = alpha_defect.max((raw[i * n + j] - raw[j * n + i]).abs());
        }
    }
    let raw_scale = raw.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
    let sym_assert_rel = opts.sym_assert_rel.unwrap_or(100.0 * opts.tol);
    assert!(
        alpha_defect <= sym_assert_rel * raw_scale,
        "linear_response: α(ω) is not symmetric — defect {alpha_defect:.6e} exceeds {:.6e} \
         (sym_assert_rel · max(1,max|α|), max|α| = {raw_scale:.6e}) at ω = {omega:.6e}. K(ω) is \
         self-adjoint in the product metric ⟨u₁,u₂⟩_G + ⟨v₁,v₂⟩_G, so a defect this large means \
         either the −ω coupling was applied to an UNCLEANED leg (making the assembled operator \
         Π K Π' with Π ≠ Π', §11 ★) or the raise is broken — OUR assembly, hence an assert.",
        sym_assert_rel * raw_scale
    );
    let mut alpha = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            alpha[i * n + j] = 0.5 * (raw[i * n + j] + raw[j * n + i]);
        }
    }

    Ok(DynamicResponse {
        omega,
        alpha,
        alpha_defect,
        values,
        solves,
        n_eff: ctx.n_eff(),
    })
}

/// [`linear_response`]'s single-covector core: solve `K(ω)·[u;v] = [g;0]` for one already-raised,
/// already-cleaned `g`.
///
/// # Errors
///
/// As [`linear_response`], minus [`ResponseError::NotJointCritical`].
pub fn linear_response_solve(
    ctx: &JointContext<'_>,
    g: &JointVector,
    omega: f64,
    opts: &DynamicResponseOpts,
) -> Result<(JointPair, DynamicSolveInfo), ResponseError> {
    if omega < 0.0 || omega.is_nan() {
        return Err(ResponseError::NegativeOmega { omega });
    }
    let solver = select_solver(omega, opts)?;
    solve_pair(ctx, g, omega, opts, solver, "<covector>".to_string())
}

pub(crate) fn select_solver(
    omega: f64,
    opts: &DynamicResponseOpts,
) -> Result<DynamicSolver, ResponseError> {
    select_solver_from(
        omega,
        opts.solver,
        opts.spectrum.as_deref(),
        opts.resonance_window,
        opts.unstable_pairs,
    )
    .map_err(|hit| ResponseError::NearResonance {
        omega: hit.omega,
        omega_n: hit.omega_n,
        gap: hit.gap,
        window: hit.window,
    })
}

pub(crate) struct NearResonanceHit {
    pub omega: f64,
    pub omega_n: f64,
    pub gap: f64,
    pub window: f64,
}

pub(crate) fn select_solver_from(
    omega: f64,
    solver: DynamicSolver,
    spectrum: Option<&[f64]>,
    resonance_window: f64,
    unstable_pairs: usize,
) -> Result<DynamicSolver, NearResonanceHit> {
    let spec = spectrum;
    if omega != 0.0
        && let Some(s) = spec
    {
        let mut best: Option<(f64, f64)> = None;
        for &wn in s.iter() {
            let gap = (omega - wn).abs();
            if best.is_none_or(|(g, _)| gap < g) {
                best = Some((gap, wn));
            }
        }
        if let Some((gap, omega_n)) = best
            && gap <= resonance_window * omega_n.abs().max(1.0)
        {
            return Err(NearResonanceHit {
                omega,
                omega_n,
                gap,
                window: resonance_window,
            });
        }
    }
    Ok(match solver {
        DynamicSolver::Pcg => DynamicSolver::Pcg,
        DynamicSolver::Minres => DynamicSolver::Minres,
        DynamicSolver::Auto if unstable_pairs > 0 => DynamicSolver::Minres,
        DynamicSolver::Auto => {
            if omega == 0.0 {
                DynamicSolver::Pcg
            } else {
                let w1 = spec
                    .map(|s| s.iter().cloned().fold(f64::INFINITY, f64::min))
                    .unwrap_or(f64::NEG_INFINITY);
                if omega < w1 * (1.0 - resonance_window) {
                    DynamicSolver::Pcg
                } else {
                    DynamicSolver::Minres
                }
            }
        }
    })
}

fn solve_pair(
    ctx: &JointContext<'_>,
    g: &JointVector,
    omega: f64,
    opts: &DynamicResponseOpts,
    solver: DynamicSolver,
    label: String,
) -> Result<(JointPair, DynamicSolveInfo), ResponseError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();

    let op = |p: &JointPair| {
        let mut cu = p.u.clone();
        reproject_tangent(base_l, base_r, radial, &mut cu.xi);
        let mut cv = p.v.clone();
        reproject_tangent(base_l, base_r, radial, &mut cv.xi);

        let mut hu = ctx.hpp_apply(&cu);
        hu.axpy(-omega, &cv);
        let mut hv = ctx.hmm_apply(&cv);
        hv.axpy(-omega, &cu);
        deflate_radial(radial, &mut hu.xi);
        deflate_radial(radial, &mut hv.xi);
        JointPair { u: hu, v: hv }
    };

    let mut zero_v = g.clone();
    zero_v.scale(0.0);
    let b = JointPair {
        u: g.clone(),
        v: zero_v,
    };
    let bn = b.norm();

    if bn == 0.0 {
        let mut zero = b.clone();
        zero.scale(0.0);
        return Ok((
            zero,
            DynamicSolveInfo {
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
            let no_precond = |_: &mut JointPair| {};
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

    if let Some(e) = refusal_to_error(refusal, label.clone(), omega, iters, rel_residual, opts.tol)
    {
        return Err(e);
    }

    let un = x.u.norm();
    let v_weight = if un > 0.0 { x.v.norm() / un } else { 0.0 };
    Ok((
        x,
        DynamicSolveInfo {
            label,
            iters,
            rel_residual,
            grad_norm: bn,
            solver_used: solver,
            v_weight,
        },
    ))
}

pub(crate) fn refusal_to_error(
    refusal: RefusalKind,
    label: String,
    omega: f64,
    iters: usize,
    rel_residual: f64,
    tol: f64,
) -> Option<ResponseError> {
    match refusal {
        RefusalKind::Curvature(rho) => Some(if omega == 0.0 {
            ResponseError::HessianIndefinite { label, rho, iters }
        } else {
            ResponseError::IndefiniteAbovePole {
                label,
                omega,
                rho,
                iters,
            }
        }),
        RefusalKind::NotConverged(Some(it)) => Some(ResponseError::SolveNotConverged {
            label,
            iters: it,
            rel_residual,
            tol,
        }),
        RefusalKind::NotConverged(None) => None,
    }
}

pub(crate) enum RefusalKind {
    NotConverged(Option<usize>),
    Curvature(f64),
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&t| t * t).sum::<f64>().sqrt()
}

fn joint_critical_gate(
    ctx: &JointContext<'_>,
    tt_grad_tol: f64,
    orb_grad_tol: f64,
    orb_grad_tol_full: Option<f64>,
) -> Result<(), ResponseError> {
    let lambda = ctx.lambda();
    let tt_gradnorm = ctx.tt_gradnorm();
    let tt_bound = tt_grad_tol * lambda.abs().max(1.0);
    let orb_bound_kept = orb_grad_tol;
    let orb_bound_full = orb_grad_tol_full.unwrap_or(orb_grad_tol);

    let g_orb = orbital_gradient(ctx.h, &ctx.x);
    let gl: Vec<f64> = ctx
        .pairs()
        .iter()
        .map(|&(p, q)| g_orb.get(&[p, q]))
        .collect();
    let gk = ctx.vt_times(&gl);
    let orb_grad_kept = norm2(&gk);
    let orb_grad_full = norm2(&gl);

    let tt_bad = tt_gradnorm > tt_bound || tt_gradnorm.is_nan();
    let kept_bad = orb_grad_kept > orb_bound_kept || orb_grad_kept.is_nan();
    let full_bad = orb_grad_full > orb_bound_full || orb_grad_full.is_nan();
    if tt_bad || kept_bad || full_bad {
        return Err(ResponseError::NotJointCritical {
            tt_gradnorm,
            tt_bound,
            orb_grad_kept,
            orb_grad_full,
            orb_bound: if kept_bad {
                orb_bound_kept
            } else if full_bad {
                orb_bound_full
            } else {
                orb_bound_kept
            },
            lambda,
        });
    }
    Ok(())
}

fn orbital_weight_of(ctx: &JointContext<'_>, g: &JointVector, gnorm: f64) -> f64 {
    if ctx.n_eff() == 0 || gnorm == 0.0 {
        return 0.0;
    }
    let orb = JointVector::from_coeffs(ctx.spec(), ctx, g.c.clone());
    (orb.dot(&orb) / (gnorm * gnorm)).max(0.0)
}

fn flag_to_error(
    label: &str,
    flag: PcgFlag,
    iters: usize,
    rel_residual: f64,
    tol: f64,
) -> Option<ResponseError> {
    match flag {
        PcgFlag::Converged => None,
        PcgFlag::NegativeCurvature { rho } => Some(ResponseError::HessianIndefinite {
            label: label.to_string(),
            rho,
            iters,
        }),
        PcgFlag::MaxIter => Some(ResponseError::SolveNotConverged {
            label: label.to_string(),
            iters,
            rel_residual,
            tol,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_flag_mapping() {
        assert!(flag_to_error("p", PcgFlag::Converged, 7, 1e-12, 1e-8).is_none());

        match flag_to_error("p", PcgFlag::NegativeCurvature { rho: -3.5 }, 4, 1e-2, 1e-8) {
            Some(ResponseError::HessianIndefinite { label, rho, iters }) => {
                assert_eq!(label, "p");
                assert_eq!(rho, -3.5);
                assert_eq!(iters, 4);
            }
            other => panic!("expected HessianIndefinite, got {other:?}"),
        }

        match flag_to_error("p", PcgFlag::MaxIter, 500, 1e-3, 1e-8) {
            Some(ResponseError::SolveNotConverged {
                iters,
                rel_residual,
                tol,
                ..
            }) => {
                assert_eq!(iters, 500);
                assert_eq!(rel_residual, 1e-3);
                assert_eq!(tol, 1e-8);
            }
            other => panic!("expected SolveNotConverged, got {other:?}"),
        }
    }

    #[test]
    fn refusal_map_splits_at_omega_zero() {
        let e = refusal_to_error(RefusalKind::Curvature(-3.5), "x".into(), 0.0, 7, 1e-3, 1e-8)
            .expect("curvature must refuse");
        assert!(
            matches!(e, ResponseError::HessianIndefinite { rho, iters, .. } if rho == -3.5 && iters == 7),
            "ω = 0 must map to HessianIndefinite, got {e:?}"
        );
        let e = refusal_to_error(RefusalKind::Curvature(-3.5), "x".into(), 0.4, 7, 1e-3, 1e-8)
            .expect("curvature must refuse");
        assert!(
            matches!(e, ResponseError::IndefiniteAbovePole { omega, rho, .. } if omega == 0.4 && rho == -3.5),
            "ω > 0 must map to IndefiniteAbovePole, got {e:?}"
        );
        let e = refusal_to_error(
            RefusalKind::NotConverged(Some(500)),
            "x".into(),
            0.0,
            500,
            2.5e-4,
            1e-8,
        )
        .expect("non-convergence must refuse");
        assert!(
            matches!(e, ResponseError::SolveNotConverged { iters, rel_residual, tol, .. }
                     if iters == 500 && rel_residual == 2.5e-4 && tol == 1e-8),
            "got {e:?}"
        );
        assert!(
            refusal_to_error(
                RefusalKind::NotConverged(None),
                "x".into(),
                0.0,
                3,
                1e-12,
                1e-8
            )
            .is_none()
        );
    }

    #[test]
    fn response_error_messages_name_the_numbers() {
        let e = ResponseError::NotJointCritical {
            tt_gradnorm: 1.5e-3,
            tt_bound: 1e-6,
            orb_grad_kept: 4.6e-2,
            orb_grad_full: 4.6e-2,
            orb_bound: 1e-6,
            lambda: -7.25,
        };
        let s = e.to_string();
        assert!(s.contains("1.500000e-3"), "{s}");
        assert!(s.contains("4.600000e-2"), "{s}");
        assert!(s.contains("Remedy"), "{s}");
    }
}
