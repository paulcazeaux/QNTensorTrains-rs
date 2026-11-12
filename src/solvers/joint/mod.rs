// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The **joint** (TT ⊕ orbital-rotation) Riemannian Hessian and its dense symplectic driver: the
//! response operator of the variational family `{e^{−κ̂}·M_r}` in which BOTH the tensor-train
//! coefficients and the one-particle basis are variational. Real (`f64`) layer only — see
//! §8 below for why that is structural and not an oversight.

pub mod cas;
mod davidson;
mod modeopt;
mod optimize;
mod ramp;
mod response;
mod tr;
mod white;

pub use davidson::{
    IrrepBasisReport, IrrepFilterReport, IrrepReference, JointSolverDiagnostics, joint_irrep_basis,
    joint_irrep_reference, joint_lowest_davidson, joint_lowest_davidson_irrep,
    joint_lowest_davidson_with_precond, joint_lowest_lobp4dcg,
    joint_raised_orbital_diagonal_diagnostic, joint_solver_diagnostics,
};
#[doc(hidden)]
pub use davidson::{WhiteRouteCost, joint_white_route_cost};
pub use modeopt::{ModeOptOpts, ModeOptOut, mode_optimize};
pub(crate) use optimize::stamp_dropped_diagnostic;
pub use optimize::{DevSchurArms, DevSchurRouteTiming, dev_schur_route_timing};
pub use optimize::{
    JointScfInfo, JointScfOpts, JointScfOut, JointSolver, SchurRoute, TrBall, TrModel, TrScore,
    W_DAMP_S_MAX, W_DAMP_S_MIN, WDampRule, joint_ground_state,
};
pub use ramp::{RampInfo, RampLevel, RampOpts, joint_ground_state_ramped};
pub use response::{
    DynamicResponse, DynamicResponseOpts, DynamicSolveInfo, DynamicSolver, JointPair, PropRoute,
    ResponseError, ResponseSolveInfo, StaticResponse, StaticResponseOpts, TransitionMomentOpts,
    TransitionMoments, linear_response, linear_response_solve, oscillator_strengths,
    polarizability_static, property_gradient, property_gradient_with, property_value,
    transition_moments, with_field,
};
pub(crate) use response::{
    NearResonanceHit, RefusalKind, refusal_to_error, select_solver, select_solver_from,
};
pub use tr::{BallDot, TrOpts, TrStepInfo, TrTrial, tr_krylov_step};

use std::sync::{Arc, OnceLock};

use crate::coeff::CoeffTensor;
use crate::core::SparseCore;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::tangent::{TtTangent, tangent_basis};
use crate::core::weingarten::{WilliamsonError, williamson_reduced_vectors};
use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient, xt_h_y};
use crate::ops::dot;
use crate::rdm::{transition_1rdm, transition_1rdm_h};
use crate::rotation::{PairMask, g_pq, kappa_operator, orbital_pairs};
use crate::solvers::InnerProductVector;
use crate::solvers::hess::{
    SymplecticSpectrum, WeingartenCtx, deflate_radial, normal_residual, project_hmul_env,
    project_kappa_hmul_env, radial_mode, symplectic_from_grams,
};
use crate::solvers::riemannian::tangent_bases;
use crate::vector::TtVector;

const DROP_REL: f64 = 0.5;

const JOINT_PSD_TOL: f64 = 1e-10;

/// Which of the two certified-equivalent routes assembles the `F_c` rows (§4, §5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FcRoute {
    /// Two [`transition_1rdm`]-class sweeps, `O(1)` in `n_orb`. The default, and the only route
    /// the raised applies use.
    Sweep,
    /// `O(n_orb)` environment contractions against the cached `u_a` / `uz_a` trains — the
    /// certification twin of [`FcRoute::Sweep`].
    Dots,
}

/// Which representation of the normal residual `z` the covectors consume (§6, §10).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ZRoute {
    /// `z = H·x + z_lo` kept SPLIT: the `H·x` half is contracted through the operator by the
    /// environment kernels, and only `z_lo = −λ·x − ι(P(Hx−λx))` (rank ≤ 3r) is a train. The
    /// default: nothing of rank `D_H·r` is ever materialized.
    #[default]
    Split,
    /// The materialized `z = P⊥(Hx − λx)`, built on first use — the original materialized route,
    /// kept as the certification twin the split route is pinned against. [`FcRoute::Dots`] and
    /// [`JointContext::normal_residual`] always take this path regardless of the route: their whole
    /// job is to be independent of the production assembly.
    Explicit,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum WTerm {
    Full,
    None,
}

/// Construction options for [`joint_context`].
#[derive(Clone, Debug)]
pub struct JointOpts {
    /// Criticality gate on `‖P((H−λ)x)‖`, RELATIVE to `max(1, |λ|)` (§6). Default `1e-6`.
    pub tt_grad_tol: f64,
    /// Keep-rule floor on the eigenvalues of `N = S − τᵀτ`, RELATIVE to `max(1, μ_max)` (§3).
    /// Default `1e-8`; the certified band is `1e-6 … 1e-10`. Negative is refused.
    pub orb_floor: f64,
    /// Orbital pair list. `None` ⇒ [`orbital_pairs`] over the state's point group — the single
    /// order-defining source. The order is determinism-load-bearing: it indexes `u`, `S`, `N`,
    /// `V`'s rows, `c` and every row of `F_c`.
    pub pairs: Option<PairMask>,
    /// Which residual representation the covectors consume. Default [`ZRoute::Split`].
    pub z_route: ZRoute,
    /// The GENERAL (off-critical) B form. Default `false` (today's z-form, exact at TT-critical
    /// points only — §6). When `true`, B's orb×orb and cross blocks pair the chart curvature
    /// with the FULL ambient residual `(H−λ)x = ι(g_t) + z` instead of the normal residual `z`
    /// alone: every contraction is linear in that bra, so the general form is the z-terms plus
    /// the SAME §5 formulas with the embedded tangential residual `ι(g_t)` as bra — terms that
    /// vanish identically at a TT-critical point. With this set, a context built at
    /// `tt_grad_tol = ∞` has a VALID B everywhere (the GAD / off-critical consumer's enabler),
    /// at the cost of two extra transition-1RDM sweeps (or `n_orb` extra trains on the Dots
    /// route) per B covector. The tt-Weingarten block needs no correction — the second
    /// fundamental form pairs the NORMAL residual by construction, at any point.
    pub b_general: bool,
}

impl Default for JointOpts {
    fn default() -> Self {
        JointOpts {
            tt_grad_tol: 1e-6,
            orb_floor: 1e-8,
            pairs: None,
            z_route: ZRoute::Split,
            b_general: false,
        }
    }
}

/// Which route assembles `u`/`τ`/`S`/`T` in the context construction. Named parameter for the
/// same reason [`FcRoute`] is: the routes are certified-equivalent (to roundoff, never bitwise)
/// and a test must call both by name; they are NOT equivalent in cost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GramRoute {
    /// The batched sweeps ([`crate::pairgram::pair_gram`] +
    /// [`crate::pairgram_tangent::orbital_tangent_matrix`]): `O(d³r³) + O(d⁴r²)` against the dot
    /// loop's `O(d⁵r³)` class. Loses a small constant factor below the crossover; refused under
    /// a nontrivial point group until the irrep gating lands.
    Sweep,
    /// The per-pair `g_pq` train builds + `O(n_orb²)` dot loops — today's construction,
    /// verbatim, and the certification twin.
    Dots,
    /// [`GramRoute::Sweep`] iff the point group is trivial AND `d ≥` [`PAIRGRAM_CROSSOVER_D`],
    /// else [`GramRoute::Dots`].
    #[default]
    Auto,
}

/// [`GramRoute::Auto`]'s crossover dimension. The derivation stands on the `d⁵r³`-vs-`d³r³`
/// structure alone; the constant itself is a machine-dependent measurement, re-stampable from
/// `examples/bench/pairgram.rs`.
pub const PAIRGRAM_CROSSOVER_D: usize = 13;

/// Construction options of [`joint_context_with`]. [`joint_context`] is the
/// `{ gram: Dots, lean: false }` special case, bitwise-unchanged for every existing caller.
#[derive(Clone, Copy, Debug, Default)]
pub struct JointCtxBuild {
    /// Which assembly route (default [`GramRoute::Auto`]).
    pub gram: GramRoute,
    /// Skip the `u_a` trains and `τ_a` tangents entirely — the memory lever at scale. Requires a
    /// resolved [`GramRoute::Sweep`]; [`JointContext::orbital_directions`],
    /// [`JointContext::orbital_projections`] and [`FcRoute::Dots`] panic on a lean context by
    /// name (the production apply path — `FcRoute::Sweep` + the raise + `hess_apply` — never
    /// reads them).
    pub lean: bool,
}

/// Why [`joint_context`] refused to build.
#[derive(Clone, Debug)]
pub enum JointGateError {
    /// The base point is not TT-critical, so the B block would be wrong at `O(1)` with a possible
    /// sign flip (§6).
    NotTtCritical {
        measured: f64,
        bound: f64,
        lambda: f64,
    },
    /// [`JointOpts::orb_floor`] was negative — refused up front, before any work.
    OrbFloorNegative { got: f64 },
    /// `VᵀNV` failed its Cholesky. A last-resort backstop, NOT the conditioning guard: `V` holds
    /// exact eigenvectors of a PSD `N` and every kept `μ` exceeds `orb_floor·anchor ≥ 0`, so this
    /// succeeds for every legal floor and fires only on a corrupted `V`/`N`. The real conditioning
    /// number is [`DropReport::orth_error`].
    NotPd {
        orb_kept: Vec<f64>,
        orb_spectrum: Vec<f64>,
    },
    /// [`JointCtxBuild::lean`] was requested with a resolved [`GramRoute::Dots`] — lean drops
    /// the `u_a`/`τ_a` caches, which are exactly what the dot-loop construction produces.
    LeanNeedsSweep,
    /// [`GramRoute::Sweep`] was requested explicitly under a nontrivial point group — the
    /// batched kernels' irrep gating has not landed yet ([`GramRoute::Auto`] falls back to the
    /// dot loops there instead of erroring).
    SweepNeedsTrivialPg,
}

impl std::fmt::Display for JointGateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JointGateError::NotTtCritical {
                measured,
                bound,
                lambda,
            } => write!(
                f,
                "joint_context: base point is not TT-critical — ‖P((H−λ)x)‖ = {measured:.6e} \
                 exceeds {bound:.6e} (tt_grad_tol · max(1, |λ|), λ = {lambda:.6e}). The curvature \
                 block B contracts the chart's second derivative against the NORMAL residual, \
                 which equals the ambient gradient only at criticality; off it the orb×orb and \
                 cross blocks are wrong at O(1) and can carry the wrong sign (measured: 101 % \
                 error with a sign flip), silently corrupting every ω downstream. Remedy: converge \
                 the point (riemannian_cg / newton_polish reach 1e-9…1e-13 on the shipped \
                 fixtures). To inspect a non-critical point anyway set tt_grad_tol = f64::INFINITY \
                 — the A block, the joint metric and orb_spectrum stay valid there, the B block \
                 does NOT."
            ),
            JointGateError::OrbFloorNegative { got } => write!(
                f,
                "joint_context: orb_floor must be non-negative (got {got}); it is a RELATIVE keep \
                 rule on the eigenvalues of N = S − τᵀτ, anchored at max(1, μ_max)."
            ),
            JointGateError::NotPd {
                orb_kept,
                orb_spectrum,
            } => write!(
                f,
                "joint_context: VᵀNV is not positive definite (kept μ = {orb_kept:?}, full \
                 spectrum = {orb_spectrum:?}). V holds exact eigenvectors of a PSD N and every \
                 kept μ is strictly positive, so this indicates a corrupted V/N, not a \
                 conditioning problem."
            ),
            JointGateError::LeanNeedsSweep => write!(
                f,
                "joint_context_with: lean = true requires the batched Sweep construction — the \
                 dot-loop route IS the u_a/τ_a caches lean exists to drop. Use GramRoute::Sweep \
                 (or Auto at d ≥ the crossover), or rebuild with lean = false."
            ),
            JointGateError::SweepNeedsTrivialPg => write!(
                f,
                "joint_context_with: GramRoute::Sweep under a nontrivial point group — the \
                 batched kernels' per-side identity-flux gating has not landed. Use \
                 GramRoute::Auto (falls back to the dot loops at |G| > 1) or GramRoute::Dots."
            ),
        }
    }
}

impl std::error::Error for JointGateError {}

/// The context-level shape and metric precomputes a [`JointVector`] carries, shared by `Arc`.
pub struct JointSpec {
    /// Number of kept (non-redundant) orbital coordinates — the length of [`JointVector::c`].
    pub n_eff: usize,
    tau_v: Vec<TtTangent<f64>>,
    w: Vec<f64>,
}

/// A tangent vector of the JOINT manifold: a TT tangent leg plus `n_eff` kept orbital coordinates.
#[derive(Clone)]
pub struct JointVector {
    /// The TT tangent leg, gauge-fixed at the context's base frames.
    pub xi: TtTangent<f64>,
    /// The kept orbital coordinates (length [`JointSpec::n_eff`]); the full pair-space direction
    /// is `a = V·c`.
    pub c: Vec<f64>,
    spec: Arc<JointSpec>,
}

impl JointVector {
    /// A pure-tangent joint vector (`c = 0`). Not a special case anywhere in the applies: `κ ≡ 0`
    /// is a valid rotation generator, so these take the same covector path as any other vector.
    pub fn from_tangent(spec: &Arc<JointSpec>, xi: TtTangent<f64>) -> Self {
        JointVector {
            xi,
            c: vec![0.0; spec.n_eff],
            spec: spec.clone(),
        }
    }

    /// A pure-orbital joint vector (`ξ = 0` at the context's base frames). The zero components
    /// satisfy the gauge condition trivially, the same construction `tangent_basis` seeds with.
    pub fn from_coeffs(spec: &Arc<JointSpec>, ctx: &JointContext<'_>, c: Vec<f64>) -> Self {
        debug_assert_eq!(c.len(), spec.n_eff, "from_coeffs: wrong coefficient count");
        debug_assert!(
            Arc::ptr_eq(spec, ctx.spec()),
            "from_coeffs: spec and ctx must come from the same JointContext"
        );
        JointVector {
            xi: ctx.zero_tangent(),
            c,
            spec: spec.clone(),
        }
    }

    /// The shared shape/metric precomputes this vector was built against.
    pub fn spec(&self) -> &Arc<JointSpec> {
        &self.spec
    }

    fn pair(&self, f_xi: &TtTangent<f64>, f_c: &[f64]) -> f64 {
        debug_assert_eq!(f_c.len(), self.c.len());
        self.xi.dot(f_xi) + self.c.iter().zip(f_c).map(|(c, f)| c * f).sum::<f64>()
    }

    #[inline]
    fn assert_same_shape(&self, other: &Self) {
        debug_assert!(
            Arc::ptr_eq(&self.spec, &other.spec),
            "JointVector operands come from different joint contexts"
        );
    }
}

impl InnerProductVector for JointVector {
    type Scalar = f64;

    fn dot(&self, other: &Self) -> f64 {
        self.assert_same_shape(other);
        let n = self.spec.n_eff;
        let mut s = self.xi.dot(&other.xi);
        for j in 0..n {
            s += other.c[j] * self.spec.tau_v[j].dot(&self.xi)
                + self.c[j] * self.spec.tau_v[j].dot(&other.xi);
        }
        for i in 0..n {
            let mut row = 0.0;
            for j in 0..n {
                row += self.spec.w[i * n + j] * other.c[j];
            }
            s += self.c[i] * row;
        }
        s
    }

    fn norm(&self) -> f64 {
        self.dot(self).max(0.0).sqrt()
    }

    fn axpy(&mut self, a: f64, other: &Self) {
        self.assert_same_shape(other);
        self.xi.axpy(a, &other.xi);
        for (s, &o) in self.c.iter_mut().zip(other.c.iter()) {
            *s += a * o;
        }
    }

    fn scale(&mut self, a: f64) {
        self.xi.scale_mut(a);
        for s in self.c.iter_mut() {
            *s *= a;
        }
    }

    fn for_each_coeff_mut<F: FnMut(&mut f64)>(&mut self, mut f: F) {
        self.xi.for_each_coeff_mut(&mut f);
        for s in self.c.iter_mut() {
            f(s);
        }
    }

    fn zip_coeffs_mut<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, mut f: F) {
        self.assert_same_shape(other);
        self.xi.zip_coeffs_mut(&other.xi, &mut f);
        for (s, &o) in self.c.iter_mut().zip(other.c.iter()) {
            f(s, o);
        }
    }

    fn coeff(&self, idx: usize) -> f64 {
        let nt = self.xi.num_coeffs();
        if idx < nt {
            self.xi.coeff(idx)
        } else {
            self.c[idx - nt]
        }
    }

    fn set_coeff(&mut self, idx: usize, val: f64) {
        let nt = self.xi.num_coeffs();
        if idx < nt {
            self.xi.set_coeff(idx, val);
        } else {
            self.c[idx - nt] = val;
        }
    }

    fn num_coeffs(&self) -> usize {
        self.xi.num_coeffs() + self.c.len()
    }
}

struct KappaImages {
    kop: SparseHamiltonian<f64>,
    kx: TtVector<f64>,
    k_mat: CoeffTensor<f64>,
}

/// What [`JointContext::joint_basis`] dropped, and how well-conditioned the basis it returned is.
#[derive(Clone, Debug)]
pub struct DropReport {
    /// Size of the wrapped tangent block (radial-excluded).
    pub n_tt: usize,
    /// Number of orbital seeds accepted.
    pub n_orb_kept: usize,
    /// Number of orbital seeds dropped by the relative MGS test.
    pub dropped: usize,
    /// Which kept-coordinate indices were dropped.
    pub dropped_idx: Vec<usize>,
    /// Post-MGS residual norm of each orbital seed, in kept-coordinate order.
    pub residual_norms: Vec<f64>,
    /// The kept eigenvalues `μ_j` of `N`, ascending — the same order as `residual_norms`.
    pub mu_kept: Vec<f64>,
    /// `residual_norms[j] / √μ_j`, which the theory of §3 puts at `1`.
    pub ratios: Vec<f64>,
    /// The FULL spectrum of `N`, ascending (diagnosis; includes the dropped tail).
    pub orb_spectrum: Vec<f64>,
    /// Sum of the dropped eigenvalues of `N`.
    pub dropped_mass: f64,
    /// Estimated departure of the returned basis's joint Gram from `I`,
    /// `ε·max(1, max|S|) / min(μ_kept)`.
    pub orth_error: f64,
}

/// The joint Hessian's precomputed context at a base point: the wrapped Weingarten (TT) context,
/// the orbital direction trains and their redundancy quotient, the normal residual, and the joint
/// metric's Schur factor.
pub struct JointContext<'h> {
    ctx: WeingartenCtx<'h, f64>,
    h: &'h SparseHamiltonian<f64>,
    x: TtVector<f64>,
    lambda: f64,
    pairs: Vec<(usize, usize)>,
    u: Vec<TtVector<f64>>,
    uz: OnceLock<Vec<TtVector<f64>>>,
    z_route: ZRoute,
    z: OnceLock<TtVector<f64>>,
    z_lo: OnceLock<TtVector<f64>>,
    gzx: OnceLock<CoeffTensor<f64>>,
    iota_gt: Option<TtVector<f64>>,
    g_gtx: OnceLock<CoeffTensor<f64>>,
    ugt: OnceLock<Vec<TtVector<f64>>>,
    tau: Vec<TtTangent<f64>>,
    s: Vec<f64>,
    tt_gradnorm: f64,
    orb_spectrum: Vec<f64>,
    orb_kept: Vec<f64>,
    dropped_mass: f64,
    orb_floor: f64,
    v: Vec<f64>,
    nv_chol: Vec<f64>,
    radial: OnceLock<Option<TtTangent<f64>>>,
    orth_error_cache: OnceLock<f64>,
    lean: bool,
    spec: Arc<JointSpec>,
}

/// Bind the joint Hessian at `x`: frames, `λ`, orbital directions, the redundancy quotient, the
/// normal residual and the joint metric — then GATE on TT criticality (§6).
///
/// # Errors
///
/// [`JointGateError::OrbFloorNegative`] (checked first, before any work),
/// [`JointGateError::NotTtCritical`], [`JointGateError::NotPd`]. Note there is no "complex scalar"
/// refusal: the module is `f64`-only by signature, so that is a compile error (§8).
pub fn joint_context<'h>(
    h: &'h SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    opts: &JointOpts,
) -> Result<JointContext<'h>, JointGateError> {
    joint_context_with(
        h,
        x,
        opts,
        &JointCtxBuild {
            gram: GramRoute::Dots,
            lean: false,
        },
    )
}

/// [`joint_context`] with explicit construction options ([`GramRoute`], lean mode). The
/// mathematical postconditions are identical on every route (certified by the route-equivalence
/// battery); only cost, memory, and the lean accessor surface differ.
pub fn joint_context_with<'h>(
    h: &'h SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    opts: &JointOpts,
    build: &JointCtxBuild,
) -> Result<JointContext<'h>, JointGateError> {
    if opts.orb_floor < 0.0 {
        return Err(JointGateError::OrbFloorNegative {
            got: opts.orb_floor,
        });
    }

    let (base_l, base_r) = tangent_bases(x);
    let lambda = rayleigh_quotient(h, x);
    let ctx = WeingartenCtx::from_parts(h, base_l, base_r, lambda);

    let tt_gradnorm = ctx.gradient().norm();
    let bound = opts.tt_grad_tol * lambda.abs().max(1.0);
    if tt_gradnorm > bound || tt_gradnorm.is_nan() {
        return Err(JointGateError::NotTtCritical {
            measured: tt_gradnorm,
            bound,
            lambda,
        });
    }

    let space = x.space;
    let pairs: Vec<(usize, usize)> = match &opts.pairs {
        Some(mask) => mask.keep.clone(),
        None => orbital_pairs(space.pg, space.d),
    };
    let n_orb = pairs.len();

    let sweep = match build.gram {
        GramRoute::Dots => false,
        GramRoute::Sweep => {
            if !space.pg.is_trivial() {
                return Err(JointGateError::SweepNeedsTrivialPg);
            }
            true
        }
        GramRoute::Auto => space.pg.is_trivial() && space.d >= PAIRGRAM_CROSSOVER_D,
    };
    if build.lean && !sweep {
        return Err(JointGateError::LeanNeedsSweep);
    }

    let (bl, br) = ctx.shared_bases();
    let mut u: Vec<TtVector<f64>> = Vec::new();
    let mut tau: Vec<TtTangent<f64>> = Vec::new();
    let s: Vec<f64>;
    let t: Vec<f64>;
    let mut theta_kept: Option<(Vec<f64>, usize)> = None;
    if sweep {
        s = crate::pairgram::pair_gram(x, &pairs);
        let theta = crate::pairgram_tangent::orbital_tangent_matrix(bl, br, x, &pairs);
        let ncoef = theta.len().checked_div(n_orb).unwrap_or(0);
        t = crate::pairgram_tangent::tangent_coeff_gram(&theta, ncoef, n_orb);
        if !build.lean {
            u = pairs
                .iter()
                .map(|&(p, q)| &g_pq(x, p, q) * (-1.0f64))
                .collect();
            let template = TtTangent::project_shared(bl, br, x).scale(0.0);
            tau = (0..n_orb)
                .map(|a| scatter_tangent(&template, &theta[a * ncoef..(a + 1) * ncoef]))
                .collect();
        }
        theta_kept = Some((theta, ncoef));
    } else {
        u = pairs
            .iter()
            .map(|&(p, q)| &g_pq(x, p, q) * (-1.0f64))
            .collect();
        tau = u
            .iter()
            .map(|ua| TtTangent::project_shared(bl, br, ua))
            .collect();

        let mut s_acc = vec![0.0f64; n_orb * n_orb];
        let mut t_acc = vec![0.0f64; n_orb * n_orb];
        for a in 0..n_orb {
            for b in a..n_orb {
                let sab = dot(&u[a], &u[b]);
                s_acc[a * n_orb + b] = sab;
                s_acc[b * n_orb + a] = sab;
                let tab = tau[a].dot(&tau[b]);
                t_acc[a * n_orb + b] = tab;
                t_acc[b * n_orb + a] = tab;
            }
        }
        s = s_acc;
        t = t_acc;
    }
    let n_flat: Vec<f64> = s.iter().zip(t.iter()).map(|(&sv, &tv)| sv - tv).collect();
    let n_flat = symmetrize_flat(&n_flat, n_orb);

    let (orb_spectrum, evecs) = if n_orb == 0 {
        (Vec::new(), Vec::new())
    } else {
        dense_self_adjoint_smallest::<f64>(&n_flat, n_orb, n_orb)
    };
    let anchor = orb_spectrum.last().copied().unwrap_or(0.0).max(1.0);
    let keep: Vec<usize> = (0..n_orb)
        .filter(|&k| orb_spectrum[k] > opts.orb_floor * anchor)
        .collect();
    let n_eff = keep.len();
    let orb_kept: Vec<f64> = keep.iter().map(|&k| orb_spectrum[k]).collect();
    let dropped_mass: f64 = (0..n_orb)
        .filter(|k| !keep.contains(k))
        .map(|k| orb_spectrum[k].max(0.0))
        .sum();

    let mut v = vec![0.0f64; n_orb * n_eff];
    for (j, &k) in keep.iter().enumerate() {
        for a in 0..n_orb {
            v[a * n_eff + j] = evecs[k][a];
        }
    }

    let tau_v: Vec<TtTangent<f64>> = if let Some((theta, ncoef)) = theta_kept.as_ref() {
        let template = TtTangent::project_shared(bl, br, x).scale(0.0);
        let tv_flat = crate::pairgram_tangent::theta_times(theta, *ncoef, n_orb, &v, n_eff);
        (0..n_eff)
            .map(|j| scatter_tangent(&template, &tv_flat[j * ncoef..(j + 1) * ncoef]))
            .collect()
    } else {
        (0..n_eff)
            .map(|j| {
                let mut acc = tau[0].scale(0.0);
                for (a, ta) in tau.iter().enumerate() {
                    acc.axpy(v[a * n_eff + j], ta);
                }
                acc
            })
            .collect()
    };

    let w = symmetrize_flat(&congruence(&v, &s, n_orb, n_eff), n_eff);
    let nv = symmetrize_flat(&congruence(&v, &n_flat, n_orb, n_eff), n_eff);
    debug_assert!(
        n_eff == 0
            || (0..n_eff)
                .flat_map(|i| (0..n_eff).map(move |j| (i, j)))
                .filter(|(i, j)| i != j)
                .all(|(i, j)| nv[i * n_eff + j].abs() <= 1e-8 * orb_kept[n_eff - 1].max(1.0)),
        "VᵀNV should be diagonal by construction (V are exact eigenvectors of N)"
    );
    let Some(nv_chol) = chol_flat(&nv, n_eff) else {
        return Err(JointGateError::NotPd {
            orb_kept,
            orb_spectrum,
        });
    };

    let spec = Arc::new(JointSpec { n_eff, tau_v, w });
    let iota_gt = opts.b_general.then(|| ctx.gradient().to_vector());
    Ok(JointContext {
        ctx,
        h,
        x: x.clone(),
        lambda,
        pairs,
        u,
        uz: OnceLock::new(),
        z_route: opts.z_route,
        z: OnceLock::new(),
        z_lo: OnceLock::new(),
        gzx: OnceLock::new(),
        iota_gt,
        g_gtx: OnceLock::new(),
        ugt: OnceLock::new(),
        tau,
        s,
        tt_gradnorm,
        orb_spectrum,
        orb_kept,
        dropped_mass,
        orb_floor: opts.orb_floor,
        v,
        nv_chol,
        radial: OnceLock::new(),
        orth_error_cache: OnceLock::new(),
        lean: build.lean,
        spec,
    })
}

fn scatter_tangent(template: &TtTangent<f64>, col: &[f64]) -> TtTangent<f64> {
    let mut out = template.clone();
    debug_assert_eq!(
        out.num_coeffs(),
        col.len(),
        "scatter_tangent: length mismatch"
    );
    let mut it = col.iter();
    out.for_each_coeff_mut(|c| *c = *it.next().expect("column length checked"));
    out
}

impl<'h> JointContext<'h> {
    /// The orbital pair list, `p > q`, in [`orbital_pairs`] order — the index of every `u_a`,
    /// every row of `S`/`N`/`V` and every row of an `F_c` covector.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Number of kept (non-redundant) orbital coordinates. `0` ⇒ the joint stack degenerates
    /// exactly to TT-RPA and every apply forwards literally to the TT applies.
    pub fn n_eff(&self) -> usize {
        self.spec.n_eff
    }

    /// The FULL ascending spectrum of `N = S − τᵀτ`, including the dropped tail.
    pub fn orb_spectrum(&self) -> &[f64] {
        &self.orb_spectrum
    }

    /// The kept eigenvalues of `N`, ascending.
    pub fn orb_kept(&self) -> &[f64] {
        &self.orb_kept
    }

    /// Sum of the dropped eigenvalues of `N` — how much orbital direction the quotient discarded.
    pub fn dropped_mass(&self) -> f64 {
        self.dropped_mass
    }

    /// The keep-rule floor this context was built with.
    pub fn orb_floor(&self) -> f64 {
        self.orb_floor
    }

    /// The measured `‖P((H−λ)x)‖` at the base point — the criticality gate quantity of §6.
    pub fn tt_gradnorm(&self) -> f64 {
        self.tt_gradnorm
    }

    /// The Rayleigh quotient at the base point.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Whether this context was built lean ([`JointCtxBuild::lean`]).
    pub fn is_lean(&self) -> bool {
        self.lean
    }

    /// The orbital direction trains `u_a = −Ĝ_a·x`, in [`Self::pairs`] order.
    ///
    /// # Panics
    ///
    /// On a lean context — rebuild with `lean: false`, or use [`Self::try_orbital_directions`].
    pub fn orbital_directions(&self) -> &[TtVector<f64>] {
        assert!(
            !self.lean,
            "JointContext::orbital_directions: this context is LEAN — the u_a trains were never \
             built. Rebuild with JointCtxBuild {{ lean: false, .. }} if a consumer needs them."
        );
        &self.u
    }

    /// [`Self::orbital_directions`] as an `Option` (`None` on a lean context).
    pub fn try_orbital_directions(&self) -> Option<&[TtVector<f64>]> {
        if self.lean { None } else { Some(&self.u) }
    }

    /// Their tangent projections `τ_a = P(u_a)` — the P-adjoint half of the metric's cross block.
    ///
    /// # Panics
    ///
    /// On a lean context — rebuild with `lean: false`, or use [`Self::try_orbital_projections`].
    pub fn orbital_projections(&self) -> &[TtTangent<f64>] {
        assert!(
            !self.lean,
            "JointContext::orbital_projections: this context is LEAN — the τ_a tangents were \
             never built. Rebuild with JointCtxBuild {{ lean: false, .. }} if a consumer needs \
             them."
        );
        &self.tau
    }

    /// [`Self::orbital_projections`] as an `Option` (`None` on a lean context).
    pub fn try_orbital_projections(&self) -> Option<&[TtTangent<f64>]> {
        if self.lean { None } else { Some(&self.tau) }
    }

    /// The ambient orbital Gram `S_ab = ⟨u_a,u_b⟩`, row-major `n_orb²`.
    pub fn orbital_gram(&self) -> &[f64] {
        &self.s
    }

    /// The gauge frames the context was bound at — build tangent bases against THESE.
    pub fn bases(&self) -> (&TtVector<f64>, &TtVector<f64>) {
        self.ctx.bases()
    }

    pub(crate) fn shared_bases(&self) -> (&Arc<TtVector<f64>>, &Arc<TtVector<f64>>) {
        self.ctx.shared_bases()
    }

    /// The wrapped TT-only Weingarten context (its `A`/`B` applies are the joint tt×tt blocks).
    pub fn weingarten(&self) -> &WeingartenCtx<'h, f64> {
        &self.ctx
    }

    /// The normal residual `z = P⊥(Hx − λx)`, MATERIALIZED on first use.
    pub fn normal_residual(&self) -> &TtVector<f64> {
        self.z.get_or_init(|| {
            let (bl, br) = self.ctx.shared_bases();
            normal_residual(self.h, bl, br, &self.x, self.lambda)
        })
    }

    fn z_lo(&self) -> &TtVector<f64> {
        self.z_lo.get_or_init(|| {
            let grad = self.ctx.gradient().to_vector();
            &(&self.x * (-self.lambda)) - &grad
        })
    }

    fn gamma_z(&self, ket: &TtVector<f64>) -> CoeffTensor<f64> {
        match self.z_route {
            ZRoute::Explicit => transition_1rdm(self.normal_residual(), ket),
            ZRoute::Split => {
                let d = self.x.space.d;
                let gh = transition_1rdm_h(ket, self.h, &self.x);
                let glo = transition_1rdm(self.z_lo(), ket);
                let mut out = CoeffTensor::<f64>::zeros(d, 2);
                for p in 0..d {
                    for q in 0..d {
                        out.set(&[p, q], gh.get(&[q, p]) + glo.get(&[p, q]));
                    }
                }
                out
            }
        }
    }

    fn gamma_zx(&self) -> &CoeffTensor<f64> {
        self.gzx.get_or_init(|| self.gamma_z(&self.x))
    }

    fn gamma_kappa_z_x(&self, im: &KappaImages, g1: &CoeffTensor<f64>) -> CoeffTensor<f64> {
        let d = self.x.space.d;
        let gzx = self.gamma_zx();
        let mut out = CoeffTensor::<f64>::zeros(d, 2);
        for r in 0..d {
            for s in 0..d {
                let mut comm = 0.0f64;
                for t in 0..d {
                    comm += im.k_mat.get(&[r, t]) * gzx.get(&[t, s])
                        - gzx.get(&[r, t]) * im.k_mat.get(&[t, s]);
                }
                out.set(&[r, s], comm - g1.get(&[r, s]));
            }
        }
        out
    }

    /// The shared shape/metric precomputes; pass it to [`JointVector::from_tangent`] /
    /// [`JointVector::from_coeffs`].
    pub fn spec(&self) -> &Arc<JointSpec> {
        &self.spec
    }

    /// `a = V·c`: kept orbital coordinates → the FULL `n_orb` pair-space direction.
    pub fn v_times(&self, c: &[f64]) -> Vec<f64> {
        let n_eff = self.n_eff();
        debug_assert_eq!(c.len(), n_eff);
        let n_orb = self.pairs.len();
        (0..n_orb)
            .map(|a| (0..n_eff).map(|j| self.v[a * n_eff + j] * c[j]).sum())
            .collect()
    }

    /// `Vᵀ·F_c`: a FULL `n_orb`-row covector → kept-coordinate space.
    pub fn vt_times(&self, fc_full: &[f64]) -> Vec<f64> {
        let n_eff = self.n_eff();
        let n_orb = self.pairs.len();
        debug_assert_eq!(fc_full.len(), n_orb);
        (0..n_eff)
            .map(|j| (0..n_orb).map(|a| self.v[a * n_eff + j] * fc_full[a]).sum())
            .collect()
    }

    fn uz(&self) -> &[TtVector<f64>] {
        self.uz.get_or_init(|| {
            let z = self.normal_residual();
            self.pairs
                .iter()
                .map(|&(p, q)| &g_pq(z, p, q) * (-1.0f64))
                .collect()
        })
    }

    fn zero_tangent(&self) -> TtTangent<f64> {
        let (bl, br) = self.ctx.shared_bases();
        let space = bl.space;
        let r = bl.ranks.clone();
        let comps: Vec<SparseCore<f64>> = (0..space.d)
            .map(|k| SparseCore::zeros(space, k, &r[k], &r[k + 1]))
            .collect();
        TtTangent::from_parts(r, bl.clone(), br.clone(), comps)
    }

    fn kappa_matrix(&self, a: &[f64]) -> CoeffTensor<f64> {
        debug_assert_eq!(a.len(), self.pairs.len());
        let mut k = CoeffTensor::<f64>::zeros(self.x.space.d, 2);
        for (i, &(p, q)) in self.pairs.iter().enumerate() {
            k.set(&[p, q], a[i]);
            k.set(&[q, p], -a[i]);
        }
        k
    }

    fn kappa_images(&self, a: &[f64]) -> Option<KappaImages> {
        if a.iter().all(|&t| t == 0.0) {
            return None;
        }
        let k_mat = self.kappa_matrix(a);
        let kop = kappa_operator(self.x.space, &k_mat);
        let kx = &kop * &self.x;
        Some(KappaImages { kop, kx, k_mat })
    }

    /// The A covector `(F_ξ, F_c)` at the joint direction `(ξ, a)`, with `a` the FULL `n_orb`-long
    /// pair-space direction (`a = V·c` for a [`JointVector`]).
    pub fn a_covector(
        &self,
        xi: &TtTangent<f64>,
        a: &[f64],
        route: FcRoute,
    ) -> (TtTangent<f64>, Vec<f64>) {
        let im = self.kappa_images(a);
        self.a_covector_im(xi, im.as_ref(), route)
    }

    /// The B covector `(F_ξ, F_c)` at `(ξ, a)`. **The z-form is valid only at a TT-critical
    /// base point; [`JointOpts::b_general`] is exact anywhere** (§6); [`joint_context`]'s gate
    /// is the default-protocol enforcement.
    pub fn b_covector(
        &self,
        xi: &TtTangent<f64>,
        a: &[f64],
        route: FcRoute,
    ) -> (TtTangent<f64>, Vec<f64>) {
        let im = self.kappa_images(a);
        self.b_covector_im(xi, im.as_ref(), route)
    }

    pub(crate) fn ab_covector(&self, xi: &TtTangent<f64>, a: &[f64]) -> (TtTangent<f64>, Vec<f64>) {
        let im = self.kappa_images(a);
        let (mut f_xi, mut fc) = self.a_covector_im(xi, im.as_ref(), FcRoute::Sweep);
        let (b_xi, b_c) = self.b_covector_im(xi, im.as_ref(), FcRoute::Sweep);
        f_xi.axpy(1.0, &b_xi);
        for (f, b) in fc.iter_mut().zip(b_c.iter()) {
            *f += b;
        }
        (f_xi, fc)
    }

    fn a_covector_im(
        &self,
        xi: &TtTangent<f64>,
        im: Option<&KappaImages>,
        route: FcRoute,
    ) -> (TtTangent<f64>, Vec<f64>) {
        let (bl, br) = self.ctx.shared_bases();
        let w1 = xi.to_vector();

        let mut f_xi = self.ctx.a_apply_fused(xi);
        let w_hat = match im {
            None => w1,
            Some(im) => {
                let leg = match self.z_route {
                    ZRoute::Split => project_hmul_env(self.h, bl, br, self.lambda, &im.kx),
                    ZRoute::Explicit => project_hmul_explicit(self.h, bl, br, self.lambda, &im.kx),
                };
                f_xi.axpy(-1.0, &leg);
                &w1 - &im.kx
            }
        };

        let fc = match route {
            FcRoute::Sweep => {
                let gh = transition_1rdm_h(&self.x, self.h, &w_hat);
                let g0 = transition_1rdm(&self.x, &w_hat);
                self.pairs
                    .iter()
                    .map(|&(p, q)| {
                        let gpq = gh.get(&[p, q]) - self.lambda * g0.get(&[p, q]);
                        let gqp = gh.get(&[q, p]) - self.lambda * g0.get(&[q, p]);
                        gpq - gqp
                    })
                    .collect()
            }
            FcRoute::Dots => {
                assert!(
                    !self.lean,
                    "FcRoute::Dots on a LEAN context — the dots route reads the u_a trains lean \
                     drops. The production route is FcRoute::Sweep; rebuild with lean: false for \
                     the certification twin."
                );
                self.u
                    .iter()
                    .map(|ua| xt_h_y(ua, self.h, &w_hat) - self.lambda * dot(ua, &w_hat))
                    .collect()
            }
        };
        (f_xi, fc)
    }

    fn b_covector_im(
        &self,
        xi: &TtTangent<f64>,
        im: Option<&KappaImages>,
        route: FcRoute,
    ) -> (TtTangent<f64>, Vec<f64>) {
        self.b_covector_im_w(xi, im, route, WTerm::Full)
    }

    fn b_covector_im_w(
        &self,
        xi: &TtTangent<f64>,
        im: Option<&KappaImages>,
        route: FcRoute,
        wterm: WTerm,
    ) -> (TtTangent<f64>, Vec<f64>) {
        let (bl, br) = self.ctx.shared_bases();
        let n_orb = self.pairs.len();
        let w1 = xi.to_vector();

        let mut f_xi = match wterm {
            WTerm::Full => self.ctx.b_apply(xi),
            WTerm::None => xi.scale(0.0),
        };

        let need_kz = self.z_route == ZRoute::Explicit || route == FcRoute::Dots;
        let kz = if need_kz {
            im.map(|im| &im.kop * self.normal_residual())
        } else {
            None
        };

        if let Some(im) = im {
            match self.z_route {
                ZRoute::Split => {
                    f_xi.axpy(
                        1.0,
                        &project_kappa_hmul_env(&im.kop, self.h, bl, br, &self.x),
                    );
                    f_xi.axpy(1.0, &project_hmul_env(&im.kop, bl, br, 0.0, self.z_lo()));
                }
                ZRoute::Explicit => {
                    let kz = kz.as_ref().expect("kz built on the explicit route");
                    f_xi.axpy(1.0, &TtTangent::project_shared(bl, br, kz));
                }
            }
        }

        let mut fc = vec![0.0f64; n_orb];
        match route {
            FcRoute::Sweep => {
                let gz = self.gamma_z(&w1);
                for (i, &(p, q)) in self.pairs.iter().enumerate() {
                    fc[i] = gz.get(&[q, p]) - gz.get(&[p, q]);
                }
                if let Some(im) = im {
                    let g1 = self.gamma_z(&im.kx);
                    let g2 = match self.z_route {
                        ZRoute::Split => self.gamma_kappa_z_x(im, &g1),
                        ZRoute::Explicit => transition_1rdm(
                            kz.as_ref().expect("kz built on the explicit route"),
                            &self.x,
                        ),
                    };
                    for (i, &(p, q)) in self.pairs.iter().enumerate() {
                        fc[i] += 0.5
                            * ((g1.get(&[p, q]) - g1.get(&[q, p]))
                                + (g2.get(&[q, p]) - g2.get(&[p, q])));
                    }
                }
            }
            FcRoute::Dots => {
                assert!(
                    !self.lean,
                    "FcRoute::Dots on a LEAN context — the dots route reads the u_a trains lean \
                     drops. The production route is FcRoute::Sweep; rebuild with lean: false for \
                     the certification twin."
                );
                let uz = self.uz();
                for (i, uzi) in uz.iter().enumerate() {
                    fc[i] = -dot(uzi, &w1);
                }
                if let Some(im) = im {
                    let kz = kz.as_ref().expect("kz built with im");
                    for i in 0..n_orb {
                        fc[i] += 0.5 * (dot(&uz[i], &im.kx) + dot(kz, &self.u[i]));
                    }
                }
            }
        }

        if let Some(gt) = self.iota_gt.as_ref() {
            let kgt = im.map(|im| &im.kop * gt);
            match route {
                FcRoute::Sweep => {
                    let ggt = transition_1rdm(gt, &w1);
                    for (i, &(p, q)) in self.pairs.iter().enumerate() {
                        fc[i] += ggt.get(&[q, p]) - ggt.get(&[p, q]);
                    }
                    if let Some(im) = im {
                        let g1 = transition_1rdm(gt, &im.kx);
                        let ggtx = self.g_gtx.get_or_init(|| transition_1rdm(gt, &self.x));
                        let d = self.x.space.d;
                        let mut g2 = CoeffTensor::<f64>::zeros(d, 2);
                        for r in 0..d {
                            for s_ in 0..d {
                                let mut comm = 0.0f64;
                                for t in 0..d {
                                    comm += im.k_mat.get(&[r, t]) * ggtx.get(&[t, s_])
                                        - ggtx.get(&[r, t]) * im.k_mat.get(&[t, s_]);
                                }
                                g2.set(&[r, s_], comm - g1.get(&[r, s_]));
                            }
                        }
                        for (i, &(p, q)) in self.pairs.iter().enumerate() {
                            fc[i] += 0.5
                                * ((g1.get(&[p, q]) - g1.get(&[q, p]))
                                    + (g2.get(&[q, p]) - g2.get(&[p, q])));
                        }
                    }
                }
                FcRoute::Dots => {
                    let ugt = self.ugt.get_or_init(|| {
                        self.pairs
                            .iter()
                            .map(|&(p, q)| &g_pq(gt, p, q) * (-1.0f64))
                            .collect()
                    });
                    for (i, ugti) in ugt.iter().enumerate() {
                        fc[i] += -dot(ugti, &w1);
                    }
                    if let Some(im) = im {
                        let kgt = kgt.as_ref().expect("kgt built with im");
                        for i in 0..n_orb {
                            fc[i] += 0.5 * (dot(&ugt[i], &im.kx) + dot(kgt, &self.u[i]));
                        }
                    }
                }
            }
            if let Some(kgt) = kgt.as_ref() {
                f_xi.axpy(1.0, &TtTangent::project_shared(bl, br, kgt));
            }
        }
        (f_xi, fc)
    }

    /// Raise a covector `(F_ξ, F_c)` — with `F_c` already in KEPT coordinates — to a
    /// [`JointVector`], through the Schur complement of the joint metric (§7). Takes ownership of
    /// `F_ξ`, which is updated in place into `ξ_r`.
    pub fn joint_raise(&self, mut f_xi: TtTangent<f64>, f_c: Vec<f64>) -> JointVector {
        let n = self.n_eff();
        if n == 0 {
            return JointVector {
                xi: f_xi,
                c: Vec::new(),
                spec: self.spec.clone(),
            };
        }
        assert_eq!(f_c.len(), n, "joint_raise: F_c must be in kept coordinates");
        let rhs: Vec<f64> = (0..n)
            .map(|j| f_c[j] - self.spec.tau_v[j].dot(&f_xi))
            .collect();
        let c_r = chol_solve(&self.nv_chol, &rhs, n);
        for (cj, tv) in c_r.iter().zip(self.spec.tau_v.iter()) {
            f_xi.axpy(-cj, tv);
        }
        JointVector {
            xi: f_xi,
            c: c_r,
            spec: self.spec.clone(),
        }
    }

    /// `A·v` in the joint metric.
    pub fn a_apply(&self, v: &JointVector) -> JointVector {
        if self.n_eff() == 0 {
            return JointVector::from_tangent(&self.spec, self.ctx.a_apply_fused(&v.xi));
        }
        let a = self.v_times(&v.c);
        let im = self.kappa_images(&a);
        let (f_xi, fc_full) = self.a_covector_im(&v.xi, im.as_ref(), FcRoute::Sweep);
        self.joint_raise(f_xi, self.vt_times(&fc_full))
    }

    /// `B·v` in the joint metric. **Valid only at a TT-critical base point** (§6) — unless the
    /// context was built with [`JointOpts::b_general`], which restores exactness at ANY point
    /// (certified off-critically at `‖P((H−λ)x)‖ = 1.6`: the
    /// z-form's 100% cross/orb-orb error vs the general form's FD-floor agreement).
    pub fn b_apply(&self, v: &JointVector) -> JointVector {
        if self.n_eff() == 0 {
            return JointVector::from_tangent(&self.spec, self.ctx.b_apply(&v.xi));
        }
        let a = self.v_times(&v.c);
        let im = self.kappa_images(&a);
        let (f_xi, fc_full) = self.b_covector_im(&v.xi, im.as_ref(), FcRoute::Sweep);
        self.joint_raise(f_xi, self.vt_times(&fc_full))
    }

    fn ab_raised(&self, v: &JointVector) -> (JointVector, JointVector) {
        let ((fa_xi, fa_c), (fb_xi, fb_c)) = self.ab_covectors(v);
        (self.joint_raise(fa_xi, fa_c), self.joint_raise(fb_xi, fb_c))
    }

    #[allow(clippy::type_complexity)]
    fn ab_covectors(
        &self,
        v: &JointVector,
    ) -> ((TtTangent<f64>, Vec<f64>), (TtTangent<f64>, Vec<f64>)) {
        if self.n_eff() == 0 {
            return (
                (self.ctx.a_apply_fused(&v.xi), Vec::new()),
                (self.ctx.b_apply(&v.xi), Vec::new()),
            );
        }
        let a = self.v_times(&v.c);
        let im = self.kappa_images(&a);
        let (fa_xi, fa_c) = self.a_covector_im(&v.xi, im.as_ref(), FcRoute::Sweep);
        let (fb_xi, fb_c) = self.b_covector_im(&v.xi, im.as_ref(), FcRoute::Sweep);
        ((fa_xi, self.vt_times(&fa_c)), (fb_xi, self.vt_times(&fb_c)))
    }

    /// The whole joint Riemannian Hessian `H^rr·v = (A + B)·v`.
    pub fn hess_apply(&self, v: &JointVector) -> JointVector {
        let (av, bv) = self.ab_raised(v);
        let mut out = av;
        out.axpy(1.0, &bv);
        out
    }

    /// `H^rr = A + B` (alias of [`Self::hess_apply`]).
    pub fn hpp_apply(&self, v: &JointVector) -> JointVector {
        self.hess_apply(v)
    }

    /// `H^ii = A − B`. The symplectic driver consumes the `hpp`/`hmm` PAIR, never a summed
    /// Hessian: `A+B` alone cannot recover `A−B`.
    pub fn hmm_apply(&self, v: &JointVector) -> JointVector {
        let (av, bv) = self.ab_raised(v);
        let mut out = av;
        out.axpy(-1.0, &bv);
        out
    }

    pub(crate) fn radial(&self) -> Option<&TtTangent<f64>> {
        self.radial.get_or_init(|| radial_mode(&self.ctx)).as_ref()
    }

    /// The G-raised joint gradient at this context's base point, radially deflated: the
    /// ½-convention covector pair `(F_ξ, F_c) = (P((H−λ)x), gk/2)` raised through
    /// [`Self::joint_raise`], then the radial mode removed from the tangent half.
    pub fn gradient_raised(&self, gk: &[f64]) -> JointVector {
        let f_xi = self.ctx.gradient().clone();
        let f_c: Vec<f64> = gk.iter().map(|g| 0.5 * g).collect();
        let mut out = self.joint_raise(f_xi, f_c);
        deflate_radial(self.radial(), &mut out.xi);
        out
    }

    /// [`Self::hess_apply`] restricted to the radial complement: the IMAGE's tangent half is
    /// radially deflated (two-pass, g-orthogonal), the operator itself untouched, nothing
    /// reprojected. This is the operator the coupled trust-region step drives: the physically
    /// defined joint Hessian lives on the radial complement (the basis of
    /// [`Self::joint_basis`] excludes the radial mode structurally), and whole-image deflation
    /// is the matrix-free statement of the same restriction.
    pub fn hess_apply_deflated(&self, v: &JointVector) -> JointVector {
        let mut out = self.hess_apply(v);
        deflate_radial(self.radial(), &mut out.xi);
        out
    }

    /// The W-LESS deflated apply: `A + B − W` where `W` is B's tt-Weingarten term — the coupled
    /// trust-region model under [`crate::solvers::joint::TrModel::WLess`]. See
    /// [`Self::b_covector_im_w`] for why the step subproblem drops the manifold-curvature bill
    /// the truncating trial never pays. NOT for spectral consumers.
    pub fn hess_apply_deflated_wless(&self, v: &JointVector) -> JointVector {
        self.hess_apply_deflated_w(v, WTerm::None)
    }

    fn hess_apply_deflated_w(&self, v: &JointVector, wterm: WTerm) -> JointVector {
        if self.n_eff() == 0 {
            let mut xi = self.ctx.a_apply_fused(&v.xi);
            match wterm {
                WTerm::Full => xi = xi.add(&self.ctx.b_apply(&v.xi)),
                WTerm::None => {}
            }
            let mut out = JointVector::from_tangent(&self.spec, xi);
            deflate_radial(self.radial(), &mut out.xi);
            return out;
        }
        let a = self.v_times(&v.c);
        let im = self.kappa_images(&a);
        let (fa_xi, fa_c) = self.a_covector_im(&v.xi, im.as_ref(), FcRoute::Sweep);
        let (fb_xi, fb_c) = self.b_covector_im_w(&v.xi, im.as_ref(), FcRoute::Sweep, wterm);
        let av = self.joint_raise(fa_xi, self.vt_times(&fa_c));
        let bv = self.joint_raise(fb_xi, self.vt_times(&fb_c));
        let mut out = av;
        out.axpy(1.0, &bv);
        deflate_radial(self.radial(), &mut out.xi);
        out
    }

    /// A joint-orthonormal basis: the wrapped tangent block first (radial-EXCLUDED, the RPA
    /// driver's convention), then the `n_eff` orbital unit directions, MGS'd in the JOINT dot.
    pub fn joint_basis(&self) -> (Vec<JointVector>, DropReport) {
        let (bl, br) = self.ctx.shared_bases();
        let tb = tangent_basis(bl, br, true);
        let n_tt = tb.len();
        let n_eff = self.n_eff();
        let mut basis: Vec<JointVector> = tb
            .into_iter()
            .map(|xi| JointVector::from_tangent(&self.spec, xi))
            .collect();

        let mut residual_norms = Vec::with_capacity(n_eff);
        let mut ratios = Vec::with_capacity(n_eff);
        let mut dropped_idx = Vec::new();
        let mut kept_min_mu = f64::INFINITY;
        for j in 0..n_eff {
            let mut c = vec![0.0f64; n_eff];
            c[j] = 1.0;
            let mut seed = JointVector::from_coeffs(&self.spec, self, c);
            for _pass in 0..2 {
                for b in basis.iter() {
                    let cf = b.dot(&seed);
                    seed.axpy(-cf, b);
                }
            }
            let nrm = seed.norm();
            let root_mu = self.orb_kept[j].max(0.0).sqrt();
            residual_norms.push(nrm);
            ratios.push(if root_mu > 0.0 { nrm / root_mu } else { 0.0 });
            if nrm >= DROP_REL * root_mu && nrm > 0.0 {
                seed.scale(1.0 / nrm);
                basis.push(seed);
                kept_min_mu = kept_min_mu.min(self.orb_kept[j]);
            } else {
                dropped_idx.push(j);
            }
        }

        let max_s = self.s.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
        let orth_error = if kept_min_mu.is_finite() && kept_min_mu > 0.0 {
            f64::EPSILON * max_s.max(1.0) / kept_min_mu
        } else {
            0.0
        };
        let _ = self.orth_error_cache.set(orth_error);
        let report = DropReport {
            n_tt,
            n_orb_kept: basis.len() - n_tt,
            dropped: dropped_idx.len(),
            dropped_idx,
            residual_norms,
            mu_kept: self.orb_kept.clone(),
            ratios,
            orb_spectrum: self.orb_spectrum.clone(),
            dropped_mass: self.dropped_mass,
            orth_error,
        };
        (basis, report)
    }

    /// Dense Gram matrices `A_ij = ⟨v_i, A·v_j⟩`, `B_ij = ⟨v_i, B·v_j⟩` over a joint-orthonormal
    /// basis, row-major `n×n` flats — the exact mirror of the TT-only
    /// [`crate::solvers::hess::hessian_grams`], including its loop order (`j` outer / `i` inner,
    /// `[i*n+j]`) so a reader diffing the two sees one shape.
    pub fn joint_hessian_grams(&self, basis: &[JointVector]) -> (Vec<f64>, Vec<f64>) {
        let (bl, br) = self.ctx.shared_bases();
        let pure: Vec<JointVector> = basis
            .iter()
            .map(|v| JointVector {
                xi: TtTangent::project_shared(bl, br, &v.xi.to_vector()),
                c: v.c.clone(),
                spec: v.spec.clone(),
            })
            .collect();
        let n = pure.len();
        let mut a = vec![0.0f64; n * n];
        let mut b = vec![0.0f64; n * n];
        for (j, vj) in pure.iter().enumerate() {
            let ((fa_xi, fa_c), (fb_xi, fb_c)) = self.ab_covectors(vj);
            for (i, vi) in pure.iter().enumerate() {
                a[i * n + j] = vi.pair(&fa_xi, &fa_c);
                b[i * n + j] = vi.pair(&fb_xi, &fb_c);
            }
        }
        (a, b)
    }

    /// The joint symplectic (RPA) spectrum over `basis`, through the UNCHANGED
    /// [`symplectic_from_grams`] / `williamson_reduced` kernels. Instability is a returned
    /// diagnosis, never a panic. At `n_eff == 0` this reproduces the TT-only spectrum exactly.
    pub fn joint_symplectic_eigenvalues(&self, basis: &[JointVector]) -> SymplecticSpectrum {
        let (a, b) = self.joint_hessian_grams(basis);
        symplectic_from_grams(&a, &b, basis.len(), JOINT_PSD_TOL)
    }

    /// The VECTOR-carrying twin of [`Self::joint_symplectic_eigenvalues`]: the reduced Williamson
    /// basis `(ω, S_r, S_i)` over the same Grams, the same [`JOINT_PSD_TOL`] and the same loop
    /// order. The two must not diverge — the only difference is that this one keeps the
    /// eigenvectors, which is what a transition moment contracts against.
    pub fn joint_williamson(
        &self,
        basis: &[JointVector],
    ) -> Result<JointWilliamson, WilliamsonError> {
        let n = basis.len();
        let (a, b) = self.joint_hessian_grams(basis);
        let mut hrr = a.clone();
        let mut hii = a;
        for i in 0..hrr.len() {
            hrr[i] += b[i];
            hii[i] -= b[i];
        }
        let (omega, s_r, s_i) = williamson_reduced_vectors(&hrr, &hii, n, JOINT_PSD_TOL)?;
        Ok(JointWilliamson {
            omega,
            s_r,
            s_i,
            n,
            n_neg: 0,
            orth_error: self.orth_error(),
        })
    }

    /// Smallest Schmidt coefficient over every interior bond of the base point.
    pub fn sigma_min(&self) -> f64 {
        self.ctx
            .bond_gauges()
            .iter()
            .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
            .fold(f64::INFINITY, f64::min)
    }

    /// [`DropReport::orth_error`] of the canonical [`Self::joint_basis`] — the error budget every
    /// Gram-derived number is quoted next to, without keeping the basis itself.
    pub fn orth_error(&self) -> f64 {
        match self.orth_error_cache.get() {
            Some(&e) => e,
            None => self.joint_basis().1.orth_error,
        }
    }
}

/// The reduced Williamson basis of the joint Hessian over a joint-orthonormal basis —
/// [`JointContext::joint_williamson`]'s payload.
#[derive(Clone, Debug)]
pub struct JointWilliamson {
    /// `ω`, ascending, length `n − n_neg` (`= n` on the definite path).
    pub omega: Vec<f64>,
    /// `S_r`, `n × omega.len()` row-major; column `j` is `u_j`.
    pub s_r: Vec<f64>,
    /// `S_i`, `n × omega.len()` row-major; column `j` is `v_j`.
    pub s_i: Vec<f64>,
    /// Basis size — the ROW dimension of `s_r`/`s_i`, `≥ omega.len()`.
    pub n: usize,
    /// Imaginary modes omitted from `omega`/`s_r`/`s_i`. Always `0` from
    /// [`JointContext::joint_williamson`] (which refuses instead).
    pub n_neg: usize,
    /// [`JointContext::orth_error`] — the budget every `ω` and every moment is quoted next to.
    pub orth_error: f64,
}

fn project_hmul_explicit(
    h: &SparseHamiltonian<f64>,
    base_l: &TtVector<f64>,
    base_r: &TtVector<f64>,
    lambda: f64,
    w: &TtVector<f64>,
) -> TtTangent<f64> {
    let hw = h.apply(w);
    let mut out = TtTangent::project(base_l, base_r, &hw);
    out.axpy(-lambda, &TtTangent::project(base_l, base_r, w));
    out
}

fn symmetrize_flat(m: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (m[i * n + j] + m[j * n + i]);
        }
    }
    out
}

fn congruence(v: &[f64], m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut mv = vec![0.0f64; rows * cols];
    for a in 0..rows {
        for j in 0..cols {
            let mut acc = 0.0;
            for b in 0..rows {
                acc += m[a * rows + b] * v[b * cols + j];
            }
            mv[a * cols + j] = acc;
        }
    }
    let mut out = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut acc = 0.0;
            for a in 0..rows {
                acc += v[a * cols + i] * mv[a * cols + j];
            }
            out[i * cols + j] = acc;
        }
    }
    out
}

fn chol_flat(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || s.is_nan() {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

fn chol_solve(l: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    chol_back(l, &chol_forward(l, rhs, n), n)
}

fn chol_forward(l: &[f64], rhs: &[f64], n: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let mut s = rhs[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

fn chol_back(l: &[f64], y: &[f64], n: usize) -> Vec<f64> {
    let mut xs = vec![0.0f64; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * xs[k];
        }
        xs[i] = s / l[i * n + i];
    }
    xs
}
