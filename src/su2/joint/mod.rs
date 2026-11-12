// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The SU(2) mirror of the **joint** (TT ⊕ orbital-rotation) Riemannian Hessian: the response
//! operator of the variational family `{e^{−κ̂}·M_r}` on the spin-adapted layer. Real (`f64`) only.

pub mod cas;
pub mod davidson;
mod modeopt;
pub mod optimize;
pub mod response;
mod white;
#[cfg(test)]
mod zsplit_ctx_tests;

#[doc(hidden)]
pub use davidson::su2_joint_white_route_cost;
pub use davidson::{
    su2_joint_lowest_davidson, su2_joint_lowest_lobp4dcg, su2_joint_raised_orbital_diagonal,
    su2_joint_solver_diagnostics,
};
pub use modeopt::{ModeOptOpts, Su2ModeOptOut, su2_mode_optimize};
pub use optimize::{Su2JointScfOut, su2_joint_ground_state};
pub use response::{
    Su2JointPair, su2_linear_response, su2_linear_response_solve, su2_polarizability_static,
    su2_property_gradient, su2_property_gradient_with, su2_property_value, su2_transition_moments,
};

use std::sync::{Arc, OnceLock};

use crate::coeff::CoeffTensor;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::weingarten::{WilliamsonError, williamson_reduced_vectors};
use crate::rotation::orbital_pairs;
use crate::solvers::InnerProductVector;
use crate::solvers::hess::{ISO_TOL, SymplecticSpectrum, symplectic_from_grams};
use crate::solvers::joint::{DropReport, FcRoute, JointOpts, JointWilliamson, WTerm, ZRoute};
use crate::su2::core::Su2Core;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient, xt_h_y};
use crate::su2::hess::{
    Su2WeingartenCtx, cone_iso_defect, deflate_radial, normal_residual, project_hmul_env,
    project_kappa_hmul_env, radial_mode,
};
use crate::su2::ops::dot;
use crate::su2::orbital::{g_pq, kappa_operator};
use crate::su2::qn::Su2RankGrid;
use crate::su2::riemannian::tangent_bases;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::transition1rdm::{su2_transition_1rdm, su2_transition_1rdm_h};
use crate::su2::vector::Su2TtVector;
use crate::su2::weingarten::tangent_basis;

const DROP_REL: f64 = 0.5;

const JOINT_PSD_TOL: f64 = 1e-10;

/// Why [`su2_joint_context`] refused to build.
#[derive(Clone, Debug)]
pub enum Su2JointGateError {
    /// The base point is not TT-critical, so the B block would be wrong at `O(1)` with a possible
    /// sign flip (`crate::solvers::joint` §6).
    NotTtCritical {
        measured: f64,
        bound: f64,
        lambda: f64,
    },
    /// [`JointOpts::orb_floor`] was negative — refused up front, before any work.
    OrbFloorNegative { got: f64 },
    /// ★ The ι-isometry defect exceeded [`ISO_TOL`]: the base point's reduced tangent frames are
    /// gauge-degenerate, so the centerweighted component metric and the ambient metric of the
    /// embeddings disagree — and `N = S − τᵀτ` mixes exactly those two. Off that slice `N`'s
    /// spectrum, hence `n_eff`, hence the whole quotient, is silently wrong (§S1).
    IsoViolated { iso: f64, tol: f64 },
    /// `VᵀNV` failed its Cholesky. A last-resort backstop, NOT the conditioning guard: `V` holds
    /// exact eigenvectors of a PSD `N` and every kept `μ` exceeds `orb_floor·anchor ≥ 0`.
    NotPd {
        orb_kept: Vec<f64>,
        orb_spectrum: Vec<f64>,
    },
}

impl std::fmt::Display for Su2JointGateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Su2JointGateError::NotTtCritical {
                measured,
                bound,
                lambda,
            } => write!(
                f,
                "su2_joint_context: base point is not TT-critical — ‖P((H−λ)x)‖ = {measured:.6e} \
                 (centerweighted) exceeds {bound:.6e} (tt_grad_tol · max(1, |λ|), λ = \
                 {lambda:.6e}). The curvature block B contracts the chart's second derivative \
                 against the NORMAL residual, which equals the ambient gradient only at \
                 criticality; off it the orb×orb and cross blocks are wrong at O(1) and can carry \
                 the wrong sign, silently corrupting every ω downstream. Remedy: converge the \
                 point (crate::su2::riemannian::riemannian_cg / crate::su2::hess::newton_polish). \
                 To inspect a non-critical point anyway set tt_grad_tol = f64::INFINITY — the A \
                 block, the joint metric and orb_spectrum stay valid there, the B block does NOT."
            ),
            Su2JointGateError::OrbFloorNegative { got } => write!(
                f,
                "su2_joint_context: orb_floor must be non-negative (got {got}); it is a RELATIVE \
                 keep rule on the eigenvalues of N = S − τᵀτ, anchored at max(1, μ_max)."
            ),
            Su2JointGateError::IsoViolated { iso, tol } => write!(
                f,
                "su2_joint_context: ι-isometry violated ({iso:.3e} ≥ {tol:.3e}) — non-gauge-fixed \
                 tangent content (the F7 mechanism). The joint quotient N = S − τᵀτ subtracts an \
                 AMBIENT Gram from a CENTERWEIGHTED one, which is only meaningful on the \
                 gauge-fixed slice where the two metrics agree; off it n_eff and every kept μ are \
                 meaningless. Remedy: trim the base point (round away the gauge-degenerate \
                 sectors) or lower its rank — the same remedy the H8 cone guard asks for."
            ),
            Su2JointGateError::NotPd {
                orb_kept,
                orb_spectrum,
            } => write!(
                f,
                "su2_joint_context: VᵀNV is not positive definite (kept μ = {orb_kept:?}, full \
                 spectrum = {orb_spectrum:?}). V holds exact eigenvectors of a PSD N and every \
                 kept μ is strictly positive, so this indicates a corrupted V/N, not a \
                 conditioning problem."
            ),
        }
    }
}

impl std::error::Error for Su2JointGateError {}

/// The context-level shape and metric precomputes a [`Su2JointVector`] carries, shared by `Arc`.
pub struct Su2JointSpec {
    /// Number of kept (non-redundant) orbital coordinates — the length of [`Su2JointVector::c`].
    pub n_eff: usize,
    tau_v: Vec<Su2TtTangent<f64>>,
    w: Vec<f64>,
}

impl Su2JointSpec {
    pub(crate) fn orbital_form(&self, c: &[f64]) -> f64 {
        let n = self.n_eff;
        debug_assert_eq!(c.len(), n, "orbital_form: wrong coefficient count");
        let mut s = 0.0f64;
        for i in 0..n {
            let mut row = 0.0;
            for (&wij, &cj) in self.w[i * n..(i + 1) * n].iter().zip(c) {
                row += wij * cj;
            }
            s += c[i] * row;
        }
        s
    }
}

/// A tangent vector of the JOINT SU(2) manifold: a spin-adapted TT tangent leg plus `n_eff` kept
/// orbital coordinates.
#[derive(Clone)]
pub struct Su2JointVector {
    /// The TT tangent leg, gauge-fixed at the context's base frames.
    pub xi: Su2TtTangent<f64>,
    /// The kept orbital coordinates (length [`Su2JointSpec::n_eff`]); the full pair-space direction
    /// is `a = V·c`.
    pub c: Vec<f64>,
    spec: Arc<Su2JointSpec>,
}

impl Su2JointVector {
    /// A pure-tangent joint vector (`c = 0`). Not a special case anywhere in the applies: `κ ≡ 0`
    /// is a valid rotation generator, so these take the same covector path as any other vector.
    pub fn from_tangent(spec: &Arc<Su2JointSpec>, xi: Su2TtTangent<f64>) -> Self {
        Su2JointVector {
            xi,
            c: vec![0.0; spec.n_eff],
            spec: spec.clone(),
        }
    }

    /// A pure-orbital joint vector (`ξ = 0` at the context's base frames). The zero components
    /// satisfy the gauge condition trivially, the same construction `tangent_basis` seeds with.
    pub fn from_coeffs(spec: &Arc<Su2JointSpec>, ctx: &Su2JointContext<'_>, c: Vec<f64>) -> Self {
        debug_assert_eq!(c.len(), spec.n_eff, "from_coeffs: wrong coefficient count");
        debug_assert!(
            Arc::ptr_eq(spec, ctx.spec()),
            "from_coeffs: spec and ctx must come from the same Su2JointContext"
        );
        Su2JointVector {
            xi: ctx.zero_tangent(),
            c,
            spec: spec.clone(),
        }
    }

    /// The shared shape/metric precomputes this vector was built against.
    pub fn spec(&self) -> &Arc<Su2JointSpec> {
        &self.spec
    }

    fn pair(&self, f_xi: &Su2TtTangent<f64>, f_c: &[f64]) -> f64 {
        debug_assert_eq!(f_c.len(), self.c.len());
        self.xi.dot(f_xi) + self.c.iter().zip(f_c).map(|(c, f)| c * f).sum::<f64>()
    }

    #[inline]
    fn assert_same_shape(&self, other: &Self) {
        debug_assert!(
            Arc::ptr_eq(&self.spec, &other.spec),
            "Su2JointVector operands come from different joint contexts"
        );
    }
}

impl InnerProductVector for Su2JointVector {
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

struct Su2KappaImages {
    kop: Su2SparseHamiltonian<f64>,
    kx: Su2TtVector<f64>,
    k_mat: CoeffTensor<f64>,
}

/// The SU(2) joint Hessian's precomputed context at a base point: the wrapped SU(2) Weingarten
/// (TT) context, the orbital direction trains and their redundancy quotient, and the joint
/// metric's Schur factor — plus the ι-isometry certificate (§S1) measured at construction.
pub struct Su2JointContext<'h> {
    ctx: Su2WeingartenCtx<'h, f64>,
    h: &'h Su2SparseHamiltonian<f64>,
    x: Su2TtVector<f64>,
    lambda: f64,
    pairs: Vec<(usize, usize)>,
    u: Vec<Su2TtVector<f64>>,
    uz: OnceLock<Vec<Su2TtVector<f64>>>,
    iota_gt: Option<Su2TtVector<f64>>,
    ugt: OnceLock<Vec<Su2TtVector<f64>>>,
    z: OnceLock<Su2TtVector<f64>>,
    z_lo: OnceLock<Su2TtVector<f64>>,
    gzx: OnceLock<CoeffTensor<f64>>,
    tau: Vec<Su2TtTangent<f64>>,
    s: Vec<f64>,
    tt_gradnorm: f64,
    iso: f64,
    orb_spectrum: Vec<f64>,
    orb_kept: Vec<f64>,
    dropped_mass: f64,
    orb_floor: f64,
    z_route: ZRoute,
    v: Vec<f64>,
    nv_chol: Vec<f64>,
    radial: OnceLock<Option<Su2TtTangent<f64>>>,
    orth_error_cache: OnceLock<f64>,
    spec: Arc<Su2JointSpec>,
}

fn su2_joint_iso_defect(
    base_l: &Su2TtVector<f64>,
    base_r: &Su2TtVector<f64>,
    tau: &[Su2TtTangent<f64>],
) -> f64 {
    let tb = tangent_basis(base_l, base_r, true);
    cone_iso_defect(&tb).max(cone_iso_defect(tau))
}

/// Bind the SU(2) joint Hessian at `x`: frames, `λ`, orbital directions, the ι-isometry
/// certificate, the redundancy quotient and the joint metric — GATING on TT criticality and on the
/// isometry (§S1).
///
/// # Errors
///
/// [`Su2JointGateError::OrbFloorNegative`] (checked first, before any work),
/// [`Su2JointGateError::NotTtCritical`], [`Su2JointGateError::IsoViolated`],
/// [`Su2JointGateError::NotPd`]. There is no "complex scalar" refusal: the module is `f64`-only by
/// signature, so a `C64` call is a compile error — which matters MORE here than on the abelian
/// layer, because `Su2TtTangent<C64>` exists and every SU(2) Hessian entry point is
/// `T: Scalar`-generic.
pub fn su2_joint_context<'h>(
    h: &'h Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    opts: &JointOpts,
) -> Result<Su2JointContext<'h>, Su2JointGateError> {
    if opts.orb_floor < 0.0 {
        return Err(Su2JointGateError::OrbFloorNegative {
            got: opts.orb_floor,
        });
    }
    let (base_l, base_r) = tangent_bases(x);
    let lambda = rayleigh_quotient(h, x);
    let ctx = Su2WeingartenCtx::from_parts(h, base_l, base_r, lambda);

    let tt_gradnorm = ctx.gradient().norm();
    let bound = opts.tt_grad_tol * lambda.abs().max(1.0);
    if tt_gradnorm > bound || tt_gradnorm.is_nan() {
        return Err(Su2JointGateError::NotTtCritical {
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

    let (bl, br) = ctx.shared_bases();
    let u: Vec<Su2TtVector<f64>> = pairs
        .iter()
        .map(|&(p, q)| &g_pq(x, p, q) * (-1.0f64))
        .collect();
    let tau: Vec<Su2TtTangent<f64>> = u
        .iter()
        .map(|ua| Su2TtTangent::project_shared(bl, br, ua))
        .collect();

    let iso = su2_joint_iso_defect(bl, br, &tau);
    if iso >= ISO_TOL || iso.is_nan() {
        return Err(Su2JointGateError::IsoViolated { iso, tol: ISO_TOL });
    }

    let mut s = vec![0.0f64; n_orb * n_orb];
    let mut t = vec![0.0f64; n_orb * n_orb];
    for a in 0..n_orb {
        for b in a..n_orb {
            let sab = dot(&u[a], &u[b]);
            s[a * n_orb + b] = sab;
            s[b * n_orb + a] = sab;
            let tab = tau[a].dot(&tau[b]);
            t[a * n_orb + b] = tab;
            t[b * n_orb + a] = tab;
        }
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

    let tau_v: Vec<Su2TtTangent<f64>> = (0..n_eff)
        .map(|j| {
            let mut acc = tau[0].scale(0.0);
            for (a, ta) in tau.iter().enumerate() {
                acc.axpy(v[a * n_eff + j], ta);
            }
            acc
        })
        .collect();

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
        return Err(Su2JointGateError::NotPd {
            orb_kept,
            orb_spectrum,
        });
    };

    let spec = Arc::new(Su2JointSpec { n_eff, tau_v, w });
    let iota_gt = opts.b_general.then(|| ctx.gradient().to_vector());
    Ok(Su2JointContext {
        ctx,
        h,
        x: x.clone(),
        lambda,
        pairs,
        u,
        uz: OnceLock::new(),
        iota_gt,
        ugt: OnceLock::new(),
        z: OnceLock::new(),
        z_lo: OnceLock::new(),
        gzx: OnceLock::new(),
        tau,
        s,
        tt_gradnorm,
        iso,
        orb_spectrum,
        orb_kept,
        dropped_mass,
        orb_floor: opts.orb_floor,
        z_route: opts.z_route,
        v,
        nv_chol,
        radial: OnceLock::new(),
        orth_error_cache: OnceLock::new(),
        spec,
    })
}

impl<'h> Su2JointContext<'h> {
    /// The orbital pair list, `p > q`, in [`orbital_pairs`] order — the index of every `u_a`,
    /// every row of `S`/`N`/`V` and every row of an `F_c` covector.
    pub fn pairs(&self) -> &[(usize, usize)] {
        &self.pairs
    }

    /// Number of kept (non-redundant) orbital coordinates. `0` ⇒ the joint stack degenerates
    /// exactly to SU(2) TT-RPA and every apply forwards literally to the TT applies.
    pub fn n_eff(&self) -> usize {
        self.spec.n_eff
    }

    /// ★ The ι-isometry defect measured at construction (§S1) — the certificate that licenses
    /// `N = S − τᵀτ`, [`Self::joint_raise`] and the raised orbital preconditioner diagonal.
    /// **Quote it beside every ω, α and moment read off this context**, exactly as the H8 cone
    /// module does. Always `< ISO_TOL`, or the context would not exist.
    pub fn iso_defect(&self) -> f64 {
        self.iso
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

    /// The measured `‖P((H−λ)x)‖` (CENTERWEIGHTED) at the base point — the criticality gate.
    pub fn tt_gradnorm(&self) -> f64 {
        self.tt_gradnorm
    }

    /// The Rayleigh quotient at the base point.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The base point as passed to [`su2_joint_context`].
    pub fn state(&self) -> &Su2TtVector<f64> {
        &self.x
    }

    /// The Hamiltonian this context was bound to.
    pub fn hamiltonian(&self) -> &'h Su2SparseHamiltonian<f64> {
        self.h
    }

    /// The orbital direction trains `u_a = −Ĝ_a·x` (AMBIENT), in [`Self::pairs`] order.
    pub fn orbital_directions(&self) -> &[Su2TtVector<f64>] {
        &self.u
    }

    /// Their tangent projections `τ_a = P(u_a)` — CENTERWEIGHTED, the P-adjoint half of the
    /// metric's cross block.
    pub fn orbital_projections(&self) -> &[Su2TtTangent<f64>] {
        &self.tau
    }

    /// The AMBIENT orbital Gram `S_ab = ⟨u_a,u_b⟩`, row-major `n_orb²`.
    pub fn orbital_gram(&self) -> &[f64] {
        &self.s
    }

    /// The gauge frames the context was bound at — build tangent bases against THESE.
    pub fn bases(&self) -> (&Su2TtVector<f64>, &Su2TtVector<f64>) {
        self.ctx.bases()
    }

    pub(crate) fn shared_bases(&self) -> (&Arc<Su2TtVector<f64>>, &Arc<Su2TtVector<f64>>) {
        self.ctx.shared_bases()
    }

    /// The wrapped SU(2) TT-only Weingarten context (its `A`/`B` applies are the joint tt×tt
    /// blocks).
    pub fn weingarten(&self) -> &Su2WeingartenCtx<'h, f64> {
        &self.ctx
    }

    /// The normal residual `z = P⊥(Hx − λx)`, MATERIALIZED on first use.
    pub fn normal_residual(&self) -> &Su2TtVector<f64> {
        self.z.get_or_init(|| {
            let (bl, br) = self.ctx.shared_bases();
            normal_residual(self.h, bl, br, &self.x, self.lambda)
        })
    }

    #[doc(hidden)]
    pub fn z_materialized(&self) -> bool {
        self.z.get().is_some()
    }

    fn z_lo(&self) -> &Su2TtVector<f64> {
        self.z_lo.get_or_init(|| {
            let grad = self.ctx.gradient().to_vector();
            &(&self.x * (-self.lambda)) - &grad
        })
    }

    fn gamma_z(&self, ket: &Su2TtVector<f64>) -> CoeffTensor<f64> {
        match self.z_route {
            ZRoute::Explicit => su2_transition_1rdm(self.normal_residual(), ket),
            ZRoute::Split => {
                let d = self.x.space.d;
                let gh = su2_transition_1rdm_h(ket, self.h, &self.x);
                let glo = su2_transition_1rdm(self.z_lo(), ket);
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

    fn gamma_kappa_z_x(&self, im: &Su2KappaImages, g1: &CoeffTensor<f64>) -> CoeffTensor<f64> {
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

    /// The shared shape/metric precomputes; pass it to [`Su2JointVector::from_tangent`] /
    /// [`Su2JointVector::from_coeffs`].
    pub fn spec(&self) -> &Arc<Su2JointSpec> {
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

    fn uz(&self) -> &[Su2TtVector<f64>] {
        self.uz.get_or_init(|| {
            let z = self.normal_residual();
            self.pairs
                .iter()
                .map(|&(p, q)| &g_pq(z, p, q) * (-1.0f64))
                .collect()
        })
    }

    fn zero_tangent(&self) -> Su2TtTangent<f64> {
        let (bl, br) = self.ctx.shared_bases();
        let space = bl.space;
        let r: Vec<Su2RankGrid> = bl.ranks.clone();
        let comps: Vec<Su2Core<f64>> = (0..space.d)
            .map(|k| Su2Core::zeros(space, k, &r[k], &r[k + 1]))
            .collect();
        Su2TtTangent::from_parts(r, bl.clone(), br.clone(), comps)
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

    fn kappa_images(&self, a: &[f64]) -> Option<Su2KappaImages> {
        if a.iter().all(|&t| t == 0.0) {
            return None;
        }
        let k_mat = self.kappa_matrix(a);
        let kop = kappa_operator(self.x.space, &k_mat);
        let kx = &kop * &self.x;
        Some(Su2KappaImages { kop, kx, k_mat })
    }

    /// The A covector `(F_ξ, F_c)` at the joint direction `(ξ, a)`, with `a` the FULL `n_orb`-long
    /// pair-space direction (`a = V·c` for a [`Su2JointVector`]).
    pub fn a_covector(
        &self,
        xi: &Su2TtTangent<f64>,
        a: &[f64],
        route: FcRoute,
    ) -> (Su2TtTangent<f64>, Vec<f64>) {
        let im = self.kappa_images(a);
        self.a_covector_im(xi, im.as_ref(), route)
    }

    /// The B covector `(F_ξ, F_c)` at `(ξ, a)`. **Valid only at a TT-critical base point**;
    /// [`su2_joint_context`]'s gate is the enforcement.
    pub fn b_covector(
        &self,
        xi: &Su2TtTangent<f64>,
        a: &[f64],
        route: FcRoute,
    ) -> (Su2TtTangent<f64>, Vec<f64>) {
        let im = self.kappa_images(a);
        self.b_covector_im(xi, im.as_ref(), route)
    }

    pub(crate) fn ab_covector(
        &self,
        xi: &Su2TtTangent<f64>,
        a: &[f64],
    ) -> (Su2TtTangent<f64>, Vec<f64>) {
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
        xi: &Su2TtTangent<f64>,
        im: Option<&Su2KappaImages>,
        route: FcRoute,
    ) -> (Su2TtTangent<f64>, Vec<f64>) {
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
                let gh = su2_transition_1rdm_h(&self.x, self.h, &w_hat);
                let g0 = su2_transition_1rdm(&self.x, &w_hat);
                self.pairs
                    .iter()
                    .map(|&(p, q)| {
                        let gpq = gh.get(&[p, q]) - self.lambda * g0.get(&[p, q]);
                        let gqp = gh.get(&[q, p]) - self.lambda * g0.get(&[q, p]);
                        gpq - gqp
                    })
                    .collect()
            }
            FcRoute::Dots => self
                .u
                .iter()
                .map(|ua| xt_h_y(ua, self.h, &w_hat) - self.lambda * dot(ua, &w_hat))
                .collect(),
        };
        (f_xi, fc)
    }

    fn b_covector_im(
        &self,
        xi: &Su2TtTangent<f64>,
        im: Option<&Su2KappaImages>,
        route: FcRoute,
    ) -> (Su2TtTangent<f64>, Vec<f64>) {
        self.b_covector_im_w(xi, im, route, WTerm::Full)
    }

    fn b_covector_im_w(
        &self,
        xi: &Su2TtTangent<f64>,
        im: Option<&Su2KappaImages>,
        route: FcRoute,
        wterm: WTerm,
    ) -> (Su2TtTangent<f64>, Vec<f64>) {
        let (bl, br) = self.ctx.shared_bases();
        let n_orb = self.pairs.len();
        let w1 = xi.to_vector();

        let mut f_xi = match wterm {
            WTerm::Full => self.ctx.b_apply(xi),
            WTerm::None => self.zero_tangent(),
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
                    f_xi.axpy(1.0, &Su2TtTangent::project_shared(bl, br, kz));
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
                        ZRoute::Explicit => su2_transition_1rdm(
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
                let uz = self.uz();
                for (i, uzi) in uz.iter().enumerate() {
                    fc[i] = -dot(uzi, &w1);
                }
                if let Some(im) = im {
                    let kz = kz
                        .as_ref()
                        .expect("kz is built on the Dots route whenever im is Some");
                    for i in 0..n_orb {
                        fc[i] += 0.5 * (dot(&uz[i], &im.kx) + dot(kz, &self.u[i]));
                    }
                }
            }
        }

        if let Some(gt) = self.iota_gt.as_ref() {
            let kgt = im.map(|im| &im.kop * gt);
            if let Some(kgt) = kgt.as_ref() {
                f_xi.axpy(1.0, &Su2TtTangent::project_shared(bl, br, kgt));
            }
            match route {
                FcRoute::Sweep => {
                    let ggt = su2_transition_1rdm(gt, &w1);
                    for (i, &(p, q)) in self.pairs.iter().enumerate() {
                        fc[i] += ggt.get(&[q, p]) - ggt.get(&[p, q]);
                    }
                    if let Some(im) = im {
                        let kgt = kgt.as_ref().expect("kgt is built exactly when im is Some");
                        let g1 = su2_transition_1rdm(gt, &im.kx);
                        let g2 = su2_transition_1rdm(kgt, &self.x);
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
                        let kgt = kgt.as_ref().expect("kgt is built exactly when im is Some");
                        for i in 0..n_orb {
                            fc[i] += 0.5 * (dot(&ugt[i], &im.kx) + dot(kgt, &self.u[i]));
                        }
                    }
                }
            }
        }
        (f_xi, fc)
    }

    /// Raise a covector `(F_ξ, F_c)` — with `F_c` already in KEPT coordinates — to a
    /// [`Su2JointVector`], through the Schur complement of the joint metric. Takes ownership of
    /// `F_ξ`, which is updated in place into `ξ_r`.
    pub fn joint_raise(&self, mut f_xi: Su2TtTangent<f64>, f_c: Vec<f64>) -> Su2JointVector {
        let n = self.n_eff();
        if n == 0 {
            return Su2JointVector {
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
        Su2JointVector {
            xi: f_xi,
            c: c_r,
            spec: self.spec.clone(),
        }
    }

    /// `A·v` in the joint metric.
    pub fn a_apply(&self, v: &Su2JointVector) -> Su2JointVector {
        if self.n_eff() == 0 {
            return Su2JointVector::from_tangent(&self.spec, self.ctx.a_apply_fused(&v.xi));
        }
        let a = self.v_times(&v.c);
        let im = self.kappa_images(&a);
        let (f_xi, fc_full) = self.a_covector_im(&v.xi, im.as_ref(), FcRoute::Sweep);
        self.joint_raise(f_xi, self.vt_times(&fc_full))
    }

    /// `B·v` in the joint metric. **Valid only at a TT-critical base point.**
    pub fn b_apply(&self, v: &Su2JointVector) -> Su2JointVector {
        if self.n_eff() == 0 {
            return Su2JointVector::from_tangent(&self.spec, self.ctx.b_apply(&v.xi));
        }
        let a = self.v_times(&v.c);
        let im = self.kappa_images(&a);
        let (f_xi, fc_full) = self.b_covector_im(&v.xi, im.as_ref(), FcRoute::Sweep);
        self.joint_raise(f_xi, self.vt_times(&fc_full))
    }

    fn ab_raised(&self, v: &Su2JointVector) -> (Su2JointVector, Su2JointVector) {
        let ((fa_xi, fa_c), (fb_xi, fb_c)) = self.ab_covectors(v);
        (self.joint_raise(fa_xi, fa_c), self.joint_raise(fb_xi, fb_c))
    }

    #[allow(clippy::type_complexity)]
    fn ab_covectors(
        &self,
        v: &Su2JointVector,
    ) -> ((Su2TtTangent<f64>, Vec<f64>), (Su2TtTangent<f64>, Vec<f64>)) {
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
    pub fn hess_apply(&self, v: &Su2JointVector) -> Su2JointVector {
        let (av, bv) = self.ab_raised(v);
        let mut out = av;
        out.axpy(1.0, &bv);
        out
    }

    /// `H^rr = A + B` (alias of [`Self::hess_apply`]).
    pub fn hpp_apply(&self, v: &Su2JointVector) -> Su2JointVector {
        self.hess_apply(v)
    }

    /// `H^ii = A − B`. The symplectic driver consumes the `hpp`/`hmm` PAIR, never a summed
    /// Hessian: `A+B` alone cannot recover `A−B`.
    pub fn hmm_apply(&self, v: &Su2JointVector) -> Su2JointVector {
        let (av, bv) = self.ab_raised(v);
        let mut out = av;
        out.axpy(-1.0, &bv);
        out
    }

    pub(crate) fn radial(&self) -> Option<&Su2TtTangent<f64>> {
        self.radial.get_or_init(|| radial_mode(&self.ctx)).as_ref()
    }

    /// The G-raised joint gradient at this context's base point, radially deflated — the twin of
    /// the abelian `JointContext::gradient_raised`, whose rustdoc carries the ½-convention and
    /// the radial-freeness derivation (both layer-independent; not restated here).
    pub fn gradient_raised(&self, gk: &[f64]) -> Su2JointVector {
        let f_xi = self.ctx.gradient().clone();
        let f_c: Vec<f64> = gk.iter().map(|g| 0.5 * g).collect();
        let mut out = self.joint_raise(f_xi, f_c);
        deflate_radial(self.radial(), &mut out.xi);
        out
    }

    /// [`Self::hess_apply`] restricted to the radial complement (whole-image radial-only
    /// deflation, operator untouched, nothing reprojected) — the twin of the abelian
    /// `JointContext::hess_apply_deflated`; see there for the restriction argument.
    pub fn hess_apply_deflated(&self, v: &Su2JointVector) -> Su2JointVector {
        let mut out = self.hess_apply(v);
        deflate_radial(self.radial(), &mut out.xi);
        out
    }

    /// The W-LESS deflated apply: `A + B − W` — the twin of the abelian
    /// `JointContext::hess_apply_deflated_wless` (see there for the measured mechanism: price
    /// the flat chart the truncating trial explores, not the manifold curvature it never
    /// follows). NOT for spectral consumers.
    pub fn hess_apply_deflated_wless(&self, v: &Su2JointVector) -> Su2JointVector {
        self.hess_apply_deflated_w(v, WTerm::None)
    }

    fn hess_apply_deflated_w(&self, v: &Su2JointVector, wterm: WTerm) -> Su2JointVector {
        if self.n_eff() == 0 {
            let mut xi = self.ctx.a_apply_fused(&v.xi);
            match wterm {
                WTerm::Full => xi = xi.add(&self.ctx.b_apply(&v.xi)),
                WTerm::None => {}
            }
            let mut out = Su2JointVector::from_tangent(&self.spec, xi);
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

    /// A joint-orthonormal basis: the wrapped SU(2) tangent block first (radial-EXCLUDED, the RPA
    /// driver's convention), then the `n_eff` orbital unit directions, MGS'd in the JOINT dot.
    pub fn joint_basis(&self) -> (Vec<Su2JointVector>, DropReport) {
        let (bl, br) = self.ctx.shared_bases();
        let tb = tangent_basis(bl, br, true);
        let n_tt = tb.len();
        let n_eff = self.n_eff();
        let mut basis: Vec<Su2JointVector> = tb
            .into_iter()
            .map(|xi| Su2JointVector::from_tangent(&self.spec, xi))
            .collect();

        let mut residual_norms = Vec::with_capacity(n_eff);
        let mut ratios = Vec::with_capacity(n_eff);
        let mut dropped_idx = Vec::new();
        let mut kept_min_mu = f64::INFINITY;
        for j in 0..n_eff {
            let mut c = vec![0.0f64; n_eff];
            c[j] = 1.0;
            let mut seed = Su2JointVector::from_coeffs(&self.spec, self, c);
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
    /// basis, row-major `n×n` flats — the exact mirror of the SU(2) TT-only
    /// [`crate::su2::hess::hessian_grams`], including its loop order (`j` outer / `i` inner,
    /// `[i*n+j]`) so a reader diffing the two sees one shape.
    pub fn joint_hessian_grams(&self, basis: &[Su2JointVector]) -> (Vec<f64>, Vec<f64>) {
        let (bl, br) = self.ctx.shared_bases();
        let pure: Vec<Su2JointVector> = basis
            .iter()
            .map(|v| Su2JointVector {
                xi: Su2TtTangent::project_shared(bl, br, &v.xi.to_vector()),
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

    /// The joint symplectic (RPA) spectrum over `basis`, through the UNCHANGED shared
    /// [`symplectic_from_grams`] / `williamson_reduced` kernels. Instability is a returned
    /// diagnosis, never a panic. At `n_eff == 0` this reproduces the SU(2) TT-only spectrum.
    pub fn joint_symplectic_eigenvalues(&self, basis: &[Su2JointVector]) -> SymplecticSpectrum {
        let (a, b) = self.joint_hessian_grams(basis);
        symplectic_from_grams(&a, &b, basis.len(), JOINT_PSD_TOL)
    }

    /// The VECTOR-carrying twin of [`Self::joint_symplectic_eigenvalues`]: the reduced Williamson
    /// basis `(ω, S_r, S_i)` over the same Grams, the same [`JOINT_PSD_TOL`] and the same loop
    /// order — what a transition moment contracts against.
    ///
    /// # Errors
    ///
    /// [`WilliamsonError`], propagated from the shared kernel.
    pub fn joint_williamson(
        &self,
        basis: &[Su2JointVector],
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

    /// Smallest CENTERWEIGHTED Schmidt coefficient over every interior bond of the base point.
    pub fn sigma_min(&self) -> f64 {
        let two_s_total = self.state().space.two_s;
        self.ctx
            .bond_gauges()
            .iter()
            .flat_map(|g| {
                g.sigma.iter().flat_map(move |(s, v)| {
                    let w = crate::su2::centerweight(s.two_s, two_s_total).sqrt();
                    v.iter().map(move |&sig| w * sig)
                })
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// [`DropReport::orth_error`] of the canonical [`Self::joint_basis`] — the error budget every
    /// Gram-derived number is quoted next to, without keeping the basis itself. Read from the
    /// value the first [`Self::joint_basis`] call recorded; the basis is rebuilt only when none
    /// has run yet.
    pub fn orth_error(&self) -> f64 {
        match self.orth_error_cache.get() {
            Some(&e) => e,
            None => self.joint_basis().1.orth_error,
        }
    }
}

#[cfg(test)]
static HMUL_EXPLICIT_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn project_hmul_explicit(
    h: &Su2SparseHamiltonian<f64>,
    base_l: &Su2TtVector<f64>,
    base_r: &Su2TtVector<f64>,
    lambda: f64,
    w: &Su2TtVector<f64>,
) -> Su2TtTangent<f64> {
    #[cfg(test)]
    HMUL_EXPLICIT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let hw = h.apply(w);
    let mut out = Su2TtTangent::project(base_l, base_r, &hw);
    out.axpy(-lambda, &Su2TtTangent::project(base_l, base_r, w));
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
