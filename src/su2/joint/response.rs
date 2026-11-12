// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Linear response on the SU(2) JOINT (TT ⊕ orbital-rotation) manifold: the property gradient
//! covector [`su2_property_gradient`], the static polarizability [`su2_polarizability_static`],
//! the transition moments [`su2_transition_moments`] and the ω-dependent coupled solve
//! [`su2_linear_response`]/[`su2_linear_response_solve`]. Real (`f64`) only, spin-adapted layer.

use crate::integrals::{PropertyOperator, su2_property_hamiltonian};
use crate::solvers::InnerProductVector;
use crate::solvers::hess::{PcgFlag, pcg_solve};
use crate::solvers::joint::{
    DynamicResponse, DynamicResponseOpts, DynamicSolveInfo, DynamicSolver, PropRoute, RefusalKind,
    ResponseError, ResponseSolveInfo, StaticResponse, StaticResponseOpts, TransitionMomentOpts,
    TransitionMoments, refusal_to_error, select_solver,
};
use crate::solvers::minres::{MinresFlag, minres};
use crate::su2::hamiltonian::{Su2SparseHamiltonian, xt_h_y};
use crate::su2::hess::{deflate_radial, reproject_tangent};
use crate::su2::ops::dot;
use crate::su2::orbital::orbital_gradient;
use crate::su2::tangent::Su2TtTangent;

use super::{Su2JointContext, Su2JointVector};

/// `M⁰ = ⟨x|μ̂|x⟩ / ⟨x,x⟩` at the context's base point — AMBIENT, unweighted.
pub fn su2_property_value(ctx: &Su2JointContext<'_>, mu: &Su2SparseHamiltonian<f64>) -> f64 {
    let x = ctx.state();
    let nrm2 = dot(x, x);
    debug_assert!(
        (nrm2 - 1.0).abs() < 1e-10,
        "su2_property_value: the joint stack assumes ‖x‖ = 1 (got ⟨x,x⟩ = {nrm2:e})"
    );
    xt_h_y(x, mu, x) / nrm2
}

/// The RAISED, radial-deflated joint-manifold gradient covector `g_μ` of `⟨μ̂⟩`, in the
/// ½-convention: `⟨g_μ, w⟩_G = ½·dM/dw` for every joint direction `w` (abelian §2/§3).
pub fn su2_property_gradient(
    ctx: &Su2JointContext<'_>,
    mu: &Su2SparseHamiltonian<f64>,
) -> Su2JointVector {
    su2_property_gradient_with(ctx, mu, PropRoute::Materialized)
}

/// [`su2_property_gradient`] with the tangent-leg route named.
///
/// # Panics
///
/// On [`PropRoute::Shifted`].
pub fn su2_property_gradient_with(
    ctx: &Su2JointContext<'_>,
    mu: &Su2SparseHamiltonian<f64>,
    route: PropRoute,
) -> Su2JointVector {
    refuse_shifted(route);
    let (base_l, base_r) = ctx.shared_bases();
    let x = ctx.state();
    let radial = ctx.radial();

    let mx = mu * x;
    let mut f_xi = Su2TtTangent::project_shared(base_l, base_r, &mx);
    deflate_radial(radial, &mut f_xi);

    let fc_full: Vec<f64> = ctx
        .orbital_directions()
        .iter()
        .map(|ua| xt_h_y(ua, mu, x))
        .collect();

    let mut g = ctx.joint_raise(f_xi, ctx.vt_times(&fc_full));
    deflate_radial(radial, &mut g.xi);
    g
}

/// The static polarizability matrix `α_μν = 2·g_μᵀ(A+B)⁻¹g_ν` on the SU(2) joint manifold
/// (abelian §2).
///
/// # Errors
///
/// [`ResponseError::NotJointCritical`] (checked before any solve),
/// [`ResponseError::HessianIndefinite`], [`ResponseError::SolveNotConverged`].
///
/// # Panics
///
/// Through [`su2_property_hamiltonian`], on a property that is not rank-2 `d×d`, is not symmetric,
/// or carries a cross-irrep component. On [`PropRoute::Shifted`] (§S2). On an α symmetry defect
/// beyond [`StaticResponseOpts::sym_assert_rel`] — that certifies this module's own assembly.
pub fn su2_polarizability_static(
    ctx: &Su2JointContext<'_>,
    props: &[PropertyOperator<f64>],
    opts: &StaticResponseOpts,
) -> Result<StaticResponse, ResponseError> {
    refuse_shifted(opts.route);
    let mus: Vec<Su2SparseHamiltonian<f64>> = props
        .iter()
        .map(|p| su2_property_hamiltonian(ctx.state().space, p))
        .collect();

    su2_joint_critical_gate(
        ctx,
        opts.tt_grad_tol,
        opts.orb_grad_tol,
        opts.orb_grad_tol_full,
    )?;

    let n = props.len();
    let values: Vec<f64> = mus.iter().map(|mu| su2_property_value(ctx, mu)).collect();
    let grads: Vec<Su2JointVector> = mus
        .iter()
        .map(|mu| su2_property_gradient_with(ctx, mu, opts.route))
        .collect();

    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let op = |p: &Su2JointVector| {
        let mut q = p.clone();
        reproject_tangent(base_l, base_r, radial, &mut q.xi);
        let mut hq = ctx.hpp_apply(&q);
        deflate_radial(radial, &mut hq.xi);
        hq
    };
    let jacobi: Option<Su2JointVector> = None;
    let precond = |v: &mut Su2JointVector| {
        if let Some(d) = jacobi.as_ref() {
            v.zip_coeffs_mut(d, |s, o| *s *= o);
        }
    };
    let mut sols: Vec<Su2JointVector> = Vec::with_capacity(n);
    let mut solves: Vec<ResponseSolveInfo> = Vec::with_capacity(n);
    for (nu, g) in grads.iter().enumerate() {
        let label = props[nu].label.clone();
        let mut b = g.clone();
        reproject_tangent(base_l, base_r, radial, &mut b.xi);
        let bn = b.norm();
        let orbital_weight = su2_orbital_weight_of(ctx, g, bn);

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

        if let Some(err) = su2_flag_to_error(&label, flag, iters, rel_residual, opts.tol) {
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
        "su2_polarizability_static: α is not symmetric — defect {alpha_defect:.6e} exceeds \
         {:.6e} (sym_assert_rel · max(1,max|α|), max|α| = {raw_scale:.6e}). Solve inexactness is \
         first order in the residuals and bounded by ~2·tol·‖u‖, two decades below this bound; a \
         defect this large means the SU(2) joint apply is not self-adjoint in the joint metric, or \
         the raise is broken, or the ι-isometry that licenses the metric split has degraded (quote \
         Su2JointContext::iso_defect, currently {:.6e}) — OUR assembly, not the caller's input, \
         hence an assert.",
        sym_assert_rel * raw_scale,
        ctx.iso_defect()
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

/// The transition moments `T_n^μ = ⟨Ψ_n|μ̂|Ψ_0⟩` (at full rank) of every property in `props`, from
/// the SU(2) joint Hessian's reduced Williamson basis over `basis` (abelian §10).
///
/// # Errors
///
/// [`ResponseError::NotJointCritical`] (checked before any apply) and
/// [`ResponseError::SymplecticUnstable`].
///
/// # Panics
///
/// Through [`su2_property_hamiltonian`], as [`su2_polarizability_static`]. On
/// [`PropRoute::Shifted`] (§S2).
pub fn su2_transition_moments(
    ctx: &Su2JointContext<'_>,
    basis: &[Su2JointVector],
    props: &[PropertyOperator<f64>],
    opts: &TransitionMomentOpts,
) -> Result<TransitionMoments, ResponseError> {
    refuse_shifted(opts.route);
    let mus: Vec<Su2SparseHamiltonian<f64>> = props
        .iter()
        .map(|p| su2_property_hamiltonian(ctx.state().space, p))
        .collect();

    su2_joint_critical_gate(
        ctx,
        opts.tt_grad_tol,
        opts.orb_grad_tol,
        opts.orb_grad_tol_full,
    )?;

    let values: Vec<f64> = mus.iter().map(|mu| su2_property_value(ctx, mu)).collect();
    let w = ctx
        .joint_williamson(basis)
        .map_err(|reason| ResponseError::SymplecticUnstable { reason })?;
    let n = w.omega.len();

    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let p_count = props.len();
    let mut moments: Vec<Vec<f64>> = Vec::with_capacity(p_count);
    let mut completeness_defect: Vec<f64> = Vec::with_capacity(p_count);
    for mu in mus.iter() {
        let mut g = su2_property_gradient_with(ctx, mu, opts.route);
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

/// `[u; v]` — the unknown of `K(ω)·[u;v] = [g;0]` (abelian §11) on the SU(2) joint manifold, an
/// [`InnerProductVector`] in the PRODUCT metric `⟨(u₁,v₁),(u₂,v₂)⟩ = ⟨u₁,u₂⟩_G + ⟨v₁,v₂⟩_G`, in
/// which `K(ω)` is self-adjoint (§S4 — the metric split enters only through the legs' own dots).
#[derive(Clone)]
pub struct Su2JointPair {
    /// The `u = X+Y` leg.
    pub u: Su2JointVector,
    /// The `v = X−Y` leg. Identically zero at `ω = 0` (abelian §11).
    pub v: Su2JointVector,
}

impl InnerProductVector for Su2JointPair {
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

/// The frequency-dependent polarizability `α_μν(ω) = 2⟨g_μ, u_ν(ω)⟩_G` on the SU(2) joint
/// manifold (abelian §11, inherited per §S4 — no formula is re-derived here).
///
/// # Errors
///
/// [`ResponseError::NegativeOmega`], [`ResponseError::NotJointCritical`],
/// [`ResponseError::NearResonance`], [`ResponseError::IndefiniteAbovePole`],
/// [`ResponseError::SolveNotConverged`], and — at `ω = 0` only, where the system really is the
/// static one — [`ResponseError::HessianIndefinite`].
///
/// # Panics
///
/// On [`PropRoute::Shifted`] (§S2), BEFORE any other work. Through
/// [`su2_property_hamiltonian`], as [`su2_polarizability_static`] does. On an α symmetry defect
/// beyond [`DynamicResponseOpts::sym_assert_rel`] — that certifies this module's own assembly.
pub fn su2_linear_response(
    ctx: &Su2JointContext<'_>,
    props: &[PropertyOperator<f64>],
    omega: f64,
    opts: &DynamicResponseOpts,
) -> Result<DynamicResponse, ResponseError> {
    refuse_shifted(opts.route);
    let mus: Vec<Su2SparseHamiltonian<f64>> = props
        .iter()
        .map(|p| su2_property_hamiltonian(ctx.state().space, p))
        .collect();

    if omega < 0.0 || omega.is_nan() {
        return Err(ResponseError::NegativeOmega { omega });
    }
    su2_joint_critical_gate(
        ctx,
        opts.tt_grad_tol,
        opts.orb_grad_tol,
        opts.orb_grad_tol_full,
    )?;

    let solver = select_solver(omega, opts)?;

    let n = props.len();
    let values: Vec<f64> = mus.iter().map(|mu| su2_property_value(ctx, mu)).collect();
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let grads: Vec<Su2JointVector> = mus
        .iter()
        .map(|mu| {
            let mut g = su2_property_gradient_with(ctx, mu, opts.route);
            reproject_tangent(base_l, base_r, radial, &mut g.xi);
            g
        })
        .collect();

    let mut sols: Vec<Su2JointPair> = Vec::with_capacity(n);
    let mut solves: Vec<DynamicSolveInfo> = Vec::with_capacity(n);
    for (nu, g) in grads.iter().enumerate() {
        let (pair, info) = su2_solve_pair(ctx, g, omega, opts, solver, props[nu].label.clone())?;
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
        "su2_linear_response: α(ω) is not symmetric — defect {alpha_defect:.6e} exceeds {:.6e} \
         (sym_assert_rel · max(1,max|α|), max|α| = {raw_scale:.6e}) at ω = {omega:.6e}. K(ω) is \
         self-adjoint in the product metric ⟨u₁,u₂⟩_G + ⟨v₁,v₂⟩_G, so a defect this large means \
         either the −ω coupling was applied to an UNCLEANED leg (making the assembled operator \
         Π K Π' with Π ≠ Π', abelian §11 ★), or the raise is broken, or the ι-isometry that \
         licenses this layer's metric split has degraded (quote Su2JointContext::iso_defect, \
         currently {:.6e}) — OUR assembly, not the caller's input, hence an assert.",
        sym_assert_rel * raw_scale,
        ctx.iso_defect()
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

/// [`su2_linear_response`]'s single-covector core: solve `K(ω)·[u;v] = [g;0]` for one
/// already-raised, already-cleaned `g`.
///
/// # Errors
///
/// As [`su2_linear_response`], minus [`ResponseError::NotJointCritical`].
pub fn su2_linear_response_solve(
    ctx: &Su2JointContext<'_>,
    g: &Su2JointVector,
    omega: f64,
    opts: &DynamicResponseOpts,
) -> Result<(Su2JointPair, DynamicSolveInfo), ResponseError> {
    if omega < 0.0 || omega.is_nan() {
        return Err(ResponseError::NegativeOmega { omega });
    }
    let solver = select_solver(omega, opts)?;
    su2_solve_pair(ctx, g, omega, opts, solver, "<covector>".to_string())
}

fn su2_solve_pair(
    ctx: &Su2JointContext<'_>,
    g: &Su2JointVector,
    omega: f64,
    opts: &DynamicResponseOpts,
    solver: DynamicSolver,
    label: String,
) -> Result<(Su2JointPair, DynamicSolveInfo), ResponseError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();

    let op = |p: &Su2JointPair| {
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
        Su2JointPair { u: hu, v: hv }
    };

    let mut zero_v = g.clone();
    zero_v.scale(0.0);
    let b = Su2JointPair {
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
            let no_precond = |_: &mut Su2JointPair| {};
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

#[inline]
fn refuse_shifted(route: PropRoute) {
    assert_ne!(
        route,
        PropRoute::Shifted,
        "su2 property gradient: PropRoute::Shifted is unavailable on the SU(2) layer. It names the \
         ENVIRONMENT contraction P((μ̂−M⁰)x) that never forms μ̂·x. The kernel now exists here \
         (crate::su2::hess::project_hmul_env), but nothing builds μ̂ as a Su2SparseHamiltonian and \
         nothing certifies the resulting covector, so the flag would not name what it names. Pass \
         PropRoute::Materialized: it projects a materialized μ̂·x and deflates the radial mode \
         explicitly, which is the SAME covector by §4 of the abelian header."
    );
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&t| t * t).sum::<f64>().sqrt()
}

fn su2_joint_critical_gate(
    ctx: &Su2JointContext<'_>,
    tt_grad_tol: f64,
    orb_grad_tol: f64,
    orb_grad_tol_full: Option<f64>,
) -> Result<(), ResponseError> {
    let lambda = ctx.lambda();
    let tt_gradnorm = ctx.tt_gradnorm();
    let tt_bound = tt_grad_tol * lambda.abs().max(1.0);
    let orb_bound_kept = orb_grad_tol;
    let orb_bound_full = orb_grad_tol_full.unwrap_or(orb_grad_tol);

    let g_orb = orbital_gradient(ctx.hamiltonian(), ctx.state());
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

fn su2_orbital_weight_of(ctx: &Su2JointContext<'_>, g: &Su2JointVector, gnorm: f64) -> f64 {
    if ctx.n_eff() == 0 || gnorm == 0.0 {
        return 0.0;
    }
    (ctx.spec().orbital_form(&g.c) / (gnorm * gnorm)).max(0.0)
}

fn su2_flag_to_error(
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
