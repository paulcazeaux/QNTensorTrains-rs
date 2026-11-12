// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::coeff::CoeffTensor;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::tangent::{TtTangent, tangent_basis};
use crate::hamiltonian::{Construction, SparseHamiltonian, rayleigh_quotient};
use crate::hamiltonian::{PatternMask, TermScreen};
use crate::integrals::Integrals;
use crate::pointgroup::Irrep;
use crate::qn::QnSpace;
use crate::rotation::{
    PairMask, expm_antisym, expm_kappa_blocked, orbital_gradient, rotate_integrals,
};
use crate::solvers::InnerProductVector;
use crate::solvers::als::{AlsOptions, als};
use crate::solvers::hess::{
    NewtonPolishInfo, NewtonPolishOpts, PcgFlag, WeingartenCtx, deflate_radial, newton_polish,
    pcg_solve, reproject_tangent,
};
use crate::solvers::joint::{
    JointContext, JointOpts, JointVector, TrOpts, TrTrial, joint_context, tr_krylov_step,
};
use crate::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use crate::solvers::steihaug::SteihaugStatus;
use crate::vector::TtVector;

const ENERGY_NOISE_C: f64 = 32.0;

const GRAD_SANITY: f64 = 1e-5;

const TR_RESOLUTION_FLOOR: f64 = 5e-8;

pub(crate) fn energy_floor(e: f64) -> f64 {
    ENERGY_NOISE_C * f64::EPSILON * (1.0 + e.abs())
}

const MU_CAP: f64 = 1e6;

/// Which step rule [`joint_ground_state`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum JointSolver {
    /// The NESTED design (§2–§4): Schur-condensed modified-Newton orbital step, Armijo
    /// backtracking on the reduced energy, one warm TT leg per line-search TRIAL — the certified
    /// small-scale oracle, and this enum's `Default`. [`JointScfOpts::default`] nonetheless ships
    /// [`Self::TrKrylov`]; this arm stays fully supported and wins wall on dense-friendly small
    /// fixtures.
    #[default]
    DenseSchur,
    /// The SIMULTANEOUS design: one Steihaug–Toint solve on the coupled joint Hessian per outer
    /// iteration, joint retraction of BOTH halves from that single solve, trust-region ratio
    /// test on the RAW retracted energy (no TT leg per trial), one warm TT leg per ACCEPTED
    /// step. See [`super::tr_krylov_step`]'s module header for the model, the factor-2
    /// prediction, the acceptance constants and the ★ retry contract.
    TrKrylov,
}

/// Which norm the [`JointSolver::TrKrylov`] trust BALL is measured in. The Steihaug MODEL (CG,
/// curvature, predicted decrease) stays in the joint metric `G` under either choice — only the
/// ball geometry moves (the kernel's ★ metric split; `steihaug_cg_ball`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrBall {
    /// The CHART PRODUCT metric: `‖s‖²_ball = ‖ξ‖² + ‖δK(c)‖²_F = ‖ξ‖² + 2‖c‖₂²` (the keep
    /// basis `V` has orthonormal columns, so `‖δK‖_F = √2‖c‖₂` — the plain factor 2, NO free
    /// constant). Each block's chart variable equilibrates its own nonlinearity, so the two
    /// validity scales differ by a small factor the ρ-adaptive Δ absorbs and every probed
    /// direction's validity lands in ONE narrow window instead of G's spread over decades — which
    /// is what makes [`JointScfOpts::tr_init`] ≈ `0.05` a sensible starting radius. Under
    /// this choice `Δ` measures chart-ball LENGTH (`‖ξ‖` and radians of `‖δK‖_F` on equal footing).
    /// The dense branch's `trust0` cap on `‖δK‖_F` is this same ball restricted to the κ half — the
    /// consistency check.
    #[default]
    Chart,
}

/// Which operator the [`JointSolver::TrKrylov`] SUBPROBLEM drives. Spectral consumers always
/// use the full `A + B` regardless — this knob touches only the step model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrModel {
    /// The full joint Hessian `A + B`. Hazard: B's tt-Weingarten term prices manifold curvature
    /// `⟨z, II(ξ,ξ)⟩ ~ ‖z‖·‖ξ‖²/σ` that the truncating trial only follows within `t < σ/‖ξ‖` —
    /// decades below usable steps along compensated near-null tangents — so at near-converged
    /// points the subproblem carries SPURIOUS negative modes that Steihaug hunts scale-invariantly,
    /// the endgame rejection treadmill.
    #[default]
    Full,
    /// `A + B − W` (the tt-Weingarten term omitted, [`JointContext::hess_apply_deflated_wless`]):
    /// price the flat chart the trial actually explores. It removes the spurious negative modes
    /// exactly — the surviving spectrum edge is FD-certified by both trial charts — and leaves
    /// genuinely-carried curvature untouched; healthy directions lose a modest, honest
    /// correction the ρ-machinery absorbs.
    WLess,
}

/// Floor of the admissible Tikhonov band, and [`WDampRule::GradTracked`]'s lower clamp: the first
/// rung whose σ-damped model is clean at the regime that pins the lower edge.
pub const W_DAMP_S_MIN: f64 = 1e-7;

/// Ceiling of the admissible Tikhonov band, and [`WDampRule::GradTracked`]'s upper clamp. Pinned by
/// SYMMETRY, not by ρ: the `R·R⁻¹ = I` break `asym(W)` grew with `s`, and past this value
/// the symmetric-operator hypothesis of the certified `steihaug_cg` oracle left measured
/// ground. Measured before the damped `W` was made symmetric
/// (`WeingartenCtx::b_apply_damped`) and not re-measured since, so the ceiling's
/// rationale is open.
pub const W_DAMP_S_MAX: f64 = 1e-4;

/// How the σ-damped step model's Tikhonov scale `s` is set at each outer iteration. Both arms
/// feed `FloorPolicy::Tikhonov(s)` into a SECOND, trust-region-only bond-gauge set
/// (`WeingartenCtx::bond_gauges_damped`); the shared context the spectral consumers read is
/// untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WDampRule {
    /// A caller-supplied constant `s`. No derivation is claimed for any particular value; the
    /// recommended one when a number must be quoted is `1e-6` — a decade inside the band's lower
    /// edge and two decades inside its upper one
    /// (that edge's symmetry rationale predates the symmetric damped model).
    Fixed(f64),
    /// `s = clamp(‖gk‖, 1e-7, 1e-4)` ([`W_DAMP_S_MIN`] … [`W_DAMP_S_MAX`]), `‖gk‖` the KEPT
    /// orbital-gradient norm of the current outer iteration ([`JointScfInfo::g_orb_kept`]).
    /// **No free multiplier — `c = 1`**; the two clamp endpoints are the measured band edges
    /// themselves, not fit parameters.
    GradTracked,
}

impl WDampRule {
    /// The Tikhonov scale `s` this rule sets at an outer iteration whose kept orbital-gradient
    /// norm is `gk_norm`. [`Self::Fixed`] ignores its argument; [`Self::GradTracked`] clamps to
    /// `[1e-7, 1e-4]` ([`W_DAMP_S_MIN`] … [`W_DAMP_S_MAX`]).
    pub fn scale(self, gk_norm: f64) -> f64 {
        match self {
            WDampRule::Fixed(s) => s,
            WDampRule::GradTracked => {
                if gk_norm.is_finite() {
                    gk_norm.clamp(W_DAMP_S_MIN, W_DAMP_S_MAX)
                } else {
                    W_DAMP_S_MIN
                }
            }
        }
    }
}

/// How the [`JointSolver::TrKrylov`] TRIAL is scored (and which state an accepted step adopts).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TrScore {
    /// ONE one-site ALS sweep pair at the trial Hamiltonian from the retracted state, scored
    /// improve-only; an ACCEPTED step adopts the ALS-corrected state (never accept on a score you
    /// discard). One sweep pair tracks the fiber optimum to the converged oracle's precision at
    /// every probed length for a small fraction of its cost, so the validity window inherits the
    /// REDUCED function's wider one. Along near-redundant directions the fiber re-absorbs
    /// what the truncation drops, closing the score-vs-adopted seam of the W-less pairing.
    #[default]
    AlsRounded,
}

/// Options for [`joint_ground_state`].
#[derive(Clone, Debug)]
pub struct JointScfOpts {
    /// Which step rule the driver runs. Default [`JointSolver::TrKrylov`] with the STACK (chart
    /// ball, W-less model, ALS-rounded score, `tr_init = 0.05`); [`JointSolver::DenseSchur`] remains
    /// fully supported and wins wall on dense-friendly small fixtures.
    pub solver: JointSolver,
    /// [`JointSolver::TrKrylov`] only. Starting trust radius in the joint **G-norm**. Default
    /// `1.0` — a starting GUESS, never a floor: an exhausted rejection cascade returns its
    /// shrunken radius and the loop retries there, so the method reaches any smaller radius. At
    /// molecular scale pass `~1e-3` to skip the opening cascade — the honest radius TRACKS
    /// `σ_min` (the Weingarten curvature grows as `1/σ_min` near the manifold boundary), it is
    /// not `∝ 1/σ_min`.
    pub tr_init: f64,
    /// [`JointSolver::TrKrylov`] only. Growth cap: `Δ ← min(2Δ, tr_max)` on a very successful
    /// boundary-limited step. Default `4.0`.
    pub tr_max: f64,
    /// [`JointSolver::TrKrylov`] only. ×¼ shrinks per step CALL (≤ `tr_max_rej + 1` trials, each
    /// a rotation + Hamiltonian rebuild + Rayleigh quotient). Default `5`. Bounds one call,
    /// NEVER the method's reach down the radius axis (the retry contract). `usize`, so the
    /// `≥ 0` door check is a type-level fact.
    pub tr_max_rej: usize,
    /// [`JointSolver::TrKrylov`] only. Steihaug iteration cap per solve (operator applications).
    /// Default `200`. Distinct from the TT-LEG budgets [`Self::cg_max_iter`] /
    /// [`Self::cg_max_iter_initial`] — a different solver on a different space.
    pub tr_cg_max_iter: usize,
    /// [`JointSolver::TrKrylov`] only. Steihaug relative residual tolerance. `None` (default) ⇒
    /// the forcing sequence `min(0.5, √‖ĝ‖_G)`: loose far from convergence, where the quadratic
    /// model is untrusted anyway, tightening as the gradient falls.
    pub tr_cg_rtol: Option<f64>,
    /// [`JointSolver::TrKrylov`] only. Which norm the trust ball is measured in; the model stays
    /// in `G` either way. Default [`TrBall::Chart`], under which `Δ` and [`Self::tr_init`] are
    /// chart-ball lengths — `‖ξ‖` and radians of `‖δK‖_F` on equal footing — and the measured
    /// validity window is what puts `tr_init` at `0.05`.
    pub tr_ball: TrBall,
    /// [`JointSolver::TrKrylov`] only, honoured on BOTH layers. Which operator the Steihaug
    /// subproblem drives. Default [`TrModel::WLess`].
    pub tr_model: TrModel,
    /// [`JointSolver::TrKrylov`] only, honoured on BOTH layers. How trials are scored and which
    /// state an accept adopts. Default [`TrScore::AlsRounded`].
    pub tr_score: TrScore,
    /// [`JointSolver::TrKrylov`] only. `‖δK‖_F` threshold (FROBENIUS, never the G-norm — the two
    /// differ by orders of magnitude in the near-null-metric regime) above which the
    /// post-acceptance warm leg escalates ONCE to the cold budget
    /// [`Self::cg_max_iter_initial`], adopted only on STRICT energy improvement (the leg's
    /// noise-regime acceptance can hand back a higher energy). Default `0.1`.
    pub warm_threshold: f64,
    /// [`JointSolver::TrKrylov`] only — the MOVING-QUOTIENT protocol: run the post-acceptance TT
    /// leg on every `k`-th accepted step instead of every one, and build the interim joint
    /// contexts OFF-critically (`tt_grad_tol = ∞`, [`JointOpts::b_general`] `= true` — the
    /// certified general B is exact anywhere, so the interim step model is valid; §6). Default
    /// `1` — the leg-per-step protocol.
    pub tr_legs_every: usize,
    /// Cap on orbital iterations. The loop NEVER takes a step on its last permitted iteration —
    /// an orbital step no TT leg will re-certify would leave the returned `x` inconsistent with
    /// the returned `h`.
    pub orb_max_iter: usize,
    /// Convergence gate on the KEPT-space orbital gradient `‖Vᵀ·gL‖`. Default `1e-6`.
    pub orb_grad: f64,
    /// Threshold on the FULL-pair-list orbital gradient `‖gL‖` — the level above which a leftover
    /// along DROPPED directions is called [`JointScfInfo::rank_limited`]. `None` ⇒
    /// [`Self::orb_grad`].
    pub orb_grad_full: Option<f64>,
    /// Criticality tolerance of the TT leg, RELATIVE to `max(1, |λ|)`. Forwarded verbatim to
    /// [`JointOpts::tt_grad_tol`] **and** used as the absolute `gradtol` of every polish, so there
    /// is one number and no double standard: polishing to an ABSOLUTE `tt_grad_tol·max(1,|λ|)` is
    /// exactly the context's own bound, hence a leg that met its gate can never fail the context.
    pub tt_grad_tol: f64,
    /// Keep-rule floor on the eigenvalues of `N = S − τᵀτ`, forwarded to [`JointOpts::orb_floor`].
    /// Default `1e-8`, certified band `1e-6 … 1e-10`. **This is the knob a full-gradient refusal
    /// asks you to lower** (§5) — it is an option precisely so the documented remedy is reachable.
    pub orb_floor: f64,
    /// Resolution floor on `‖gk‖` (relative to `max(1, |λ|)`) below which a doubly-refused line
    /// search is reported as "at the Armijo resolution floor" rather than as a suspicious stall.
    /// Default `1e-7`; the measured crossing is `≈ 5e-8` (§4).
    pub armijo_floor: f64,
    /// Armijo sufficient-decrease constant `c₁`. Default `1e-4`, matching the TT line search.
    pub armijo_c: f64,
    /// Modified-Newton floor on `|eig(Heff)|`, relative to `max|eig(Heff)|`. Default `1e-6`.
    /// **A uniform PD shift is NOT an acceptable substitute** — it re-creates the stall (§3).
    pub eig_floor_rel: f64,
    /// Trust cap on `‖δK‖_F`, and the exact length the steepest-descent fallback is normalized to.
    /// Default `0.5`.
    pub trust0: f64,
    /// Line-search EVALUATIONS per direction (not halvings): trials run at
    /// `τ ∈ {t₀, t₀/2, …, t₀/2^{max_ls_trials−1}}`, and a full iteration costs at most
    /// `2·max_ls_trials` Hamiltonian rebuilds (Newton direction, then steepest descent). Default
    /// `4`. This is the dominant cost knob (§6).
    pub max_ls_trials: usize,
    /// Orbital pair list. `None` ⇒ [`crate::rotation::orbital_pairs`] over the space's point group. A mask
    /// containing a cross-irrep pair is refused UP FRONT rather than left to trip
    /// [`expm_kappa_blocked`]'s assert mid-line-search.
    pub pairs: Option<PairMask>,
    /// Which representation of the normal residual `z` every context this driver builds consumes.
    /// Default `ZRoute::Split` — nothing of rank `D_H·r` is materialized.
    pub z_route: super::ZRoute,
    /// Screening tolerance of every [`SparseHamiltonian`] rebuild (and, under [`Self::refill_h`],
    /// of the frozen class mask and its leak check). Default `f64::EPSILON`.
    pub ham_eps: f64,
    /// Automaton construction of every rebuild. Default [`Construction::Bipartite`] — the same
    /// choice `SparseHamiltonian::new` makes.
    pub construction: Construction,
    /// Refill instead of rebuild: the first trial mints a refillable skeleton — the
    /// `ham_eps` magnitude pattern at its (rotated) point, frozen as a class mask — and every later
    /// line-search / trust-region trial REFILLS a recycled instance from
    /// `rotate_integrals(ORIGINAL, U_try)` instead of re-assembling the automaton, re-minting only
    /// when the pattern has grown past the mask (`pattern_leak > 0`). The initial `U = I` operator
    /// and the cold leg are the plain build, unchanged. A refilled operator equals the plain
    /// rebuild up to classes whose `|W|` has since fallen to `≤ ham_eps` (kept as tiny values
    /// rather than dropped). Default `true` (measured bitwise the rebuild trajectory on
    /// h2o_sto3g, h2o_631g and lih; user decision 2026-09-17); `false` restores one assembly per
    /// trial. Ignored by the SU(2) driver (no refill path yet).
    pub refill_h: bool,
    /// Cap on warm [`riemannian_cg`] iterations before each line-search trial's polish. Default
    /// `5`: a trial starts from an `O(‖δK‖)`-close warm state and the polish does the work.
    pub cg_max_iter: usize,
    /// How far BELOW the orbital gate [`Self::orb_grad`] each TT leg is polished. Default `1e-2`.
    pub tt_leg_margin: f64,
    /// Cap on [`riemannian_cg`] iterations in the INITIAL leg, which starts from the caller's `x0`
    /// and is therefore COLD. Default `200`.
    pub cg_max_iter_initial: usize,
    /// Skip the CG stage when the warm gradient already satisfies
    /// `‖g‖ ≤ cg_skip_factor · tt_grad_tol · max(1,|λ|)`. Default `10.0`.
    pub cg_skip_factor: f64,
    /// Polish options; `gradtol` is OVERWRITTEN per leg with the absolute gate derived from
    /// [`Self::tt_grad_tol`].
    pub polish: NewtonPolishOpts,
    /// How `Hθθ⁻¹` is realized in the Schur condensation (§3). Default [`SchurRoute::Dense`] —
    /// see its own doc for the measurement. Both arms are usable; the choice is COST.
    pub schur_route: SchurRoute,
    /// [`SchurRoute::MatrixFree`] ONLY (ignored by the default route). Relative residual tolerance
    /// of each inner PCG solve. Default `1e-8`.
    pub schur_tol: f64,
    /// [`SchurRoute::MatrixFree`] ONLY. Iteration cap of each inner PCG solve. Default `200`.
    pub schur_max_iter: usize,
    /// Trace each orbital iteration to stderr (gate, step, per-stage wall time). Diagnosis only.
    pub verbose: bool,
    /// Grid preservation of the TT leg: forwarded verbatim to
    /// [`RiemannianOpts::preserve_grid`](crate::solvers::riemannian::RiemannianOpts::preserve_grid)
    /// at BOTH of the leg's `riemannian_cg` entries (the main stage and the mid-line-search
    /// re-entry). Default `false`.
    pub tt_preserve_grid: bool,
}

impl Default for JointScfOpts {
    fn default() -> Self {
        JointScfOpts {
            solver: JointSolver::TrKrylov,
            tr_init: 0.05,
            tr_max: 4.0,
            tr_max_rej: 5,
            tr_cg_max_iter: 200,
            tr_cg_rtol: None,
            tr_ball: TrBall::Chart,
            tr_model: TrModel::WLess,
            tr_score: TrScore::AlsRounded,
            warm_threshold: 0.1,
            tr_legs_every: 1,
            orb_max_iter: 30,
            orb_grad: 1e-6,
            orb_grad_full: None,
            tt_grad_tol: 1e-6,
            orb_floor: 1e-8,
            armijo_floor: 1e-7,
            armijo_c: 1e-4,
            eig_floor_rel: 1e-6,
            trust0: 0.5,
            max_ls_trials: 4,
            pairs: None,
            z_route: super::ZRoute::Split,
            ham_eps: f64::EPSILON,
            construction: Construction::Bipartite,
            refill_h: true,
            tt_leg_margin: 1e-2,
            cg_max_iter: 5,
            cg_max_iter_initial: 200,
            cg_skip_factor: 10.0,
            polish: NewtonPolishOpts {
                certify: false,
                ..NewtonPolishOpts::default()
            },
            schur_route: SchurRoute::Dense,
            schur_tol: 1e-8,
            schur_max_iter: 200,
            verbose: false,
            tt_preserve_grid: false,
        }
    }
}

/// Per-run diagnostics of [`joint_ground_state`].
#[derive(Clone, Debug)]
pub struct JointScfInfo {
    /// The KEPT orbital gradient and the TT criticality gate were both met at a TT-fresh point
    /// (§5). **Not on its own a publication licence — use `converged && !rank_limited`**: an
    /// uncertified endpoint of this optimizer can give a badly wrong lowest excitation from behind
    /// a perfect-looking self-consistency margin, because self-consistency is not validity.
    pub converged: bool,
    /// The orbital line search refused every trial while the shifted Newton model's OWN optimal
    /// decrease `½gᵀM⁻¹g` was already below the energy's roundoff floor `ENERGY_NOISE_C·ε·(1+|λ|)`.
    /// When set, [`Self::converged`] is `true` — the honest statement for a caller consuming the
    /// ENERGY, which is second-order insensitive (`ΔE ~ ‖gk‖²/λ`) — but the orbital gradient is at
    /// its attainable floor rather than at the gate, so a caller reading an **ω** (whose bias is
    /// FIRST order in `‖gk‖`) must gate on `converged && !stalled_at_floor`, or quote
    /// [`Self::g_orb_kept`] beside the spectrum. Mirrors `dense_casscf`'s flag of the same name.
    /// A genuine stall (line search refuses while the model still promises real decrease) leaves
    /// this `false` and `converged` `false`, so `converged && !stalled_at_floor` means exactly what
    /// `converged` meant before this field existed.
    pub stalled_at_floor: bool,
    /// ★ **The answer is limited by TT RANK, not by the optimizer.** `true` when the run converged
    /// (kept gradient and TT residual both met) while orbital-gradient mass above
    /// [`JointScfOpts::orb_grad_full`] remained along directions the redundancy quotient DROPPED.
    pub rank_limited: bool,
    /// The dropped-direction residual `‖(I − VVᵀ)·gL‖` at the point the run returned — the
    /// quantity [`Self::rank_limited`] compares against `orb_grad_full`. Zero-ish on a healthy run.
    pub g_dropped_final: f64,
    /// Largest energy slope a dropped direction can hide, `2·√(orb_floor · anchor)` per unit
    /// residual — the Cauchy–Schwarz bound on what the quotient discarded. Read
    /// [`Self::g_dropped_final`] against this: below it, the leftover is consistent with the cut
    /// itself; above it, the cut is not what is holding the run back.
    pub dropped_leak_bound: f64,
    /// The marginal dropped mode as an ANGLE, `θ = asin(√μ)` radians, where `μ` is the largest
    /// dropped eigenvalue of `N = S − τᵀτ`.
    pub dropped_angle: f64,
    /// Why the run stopped without certifying. `Some` whenever `converged == false`; also `None`
    /// on the [`Self::stalled_at_floor`] exit, where `converged == true` — that is not a refusal.
    pub refused_reason: Option<String>,
    /// Orbital iterations that ran (each one built a context and evaluated the gate).
    pub iters: usize,
    /// The reduced energy at the top of each orbital iteration.
    pub e_history: Vec<f64>,
    /// `‖gk‖` (KEPT space) per iteration.
    pub g_orb_kept: Vec<f64>,
    /// `‖gL‖` (FULL pair list) per iteration.
    pub g_orb_full: Vec<f64>,
    /// `‖gL − V·gk‖` per iteration — the structurally uncorrectable component, RECORDED for
    /// diagnosis and gated only through `g_orb_full`.
    pub g_orb_dropped: Vec<f64>,
    /// `‖P((H−λ)x)‖` per iteration, as the context measured it.
    pub tt_gradnorms: Vec<f64>,
    /// `‖δK‖_F` of each ACCEPTED orbital step (empty when no step was ever taken).
    pub delta_k_norms: Vec<f64>,
    /// Line-search trials EVALUATED in each iteration that attempted a step (Newton plus, if it
    /// refused, steepest descent). One entry per attempting iteration.
    pub trials_per_iter: Vec<usize>,
    /// [`SparseHamiltonian`] rebuilds performed by the LINE SEARCH across the run — the
    /// point-group cost axis (§6). Counts every trial's `build_h`, one per Armijo evaluation on
    /// both the Newton and (if it refused) the steepest-descent direction. Excludes the ONE
    /// initial `build_h` at `U = I` before the loop starts (§6's cost model likewise sums only
    /// over the orbital-iteration loop) — a caller budgeting total assemblies wants
    /// `ls_build_count + 1`. Under [`JointScfOpts::refill_h`] the same count is REFILLS, not
    /// assemblies.
    pub ls_build_count: usize,
    /// `n_eff` at the final point. See [`Self::iters`] for the caveat on the context-refusal
    /// exit, where "final point" means the last successfully gated context, not the returned
    /// `(u, h, x)`.
    pub n_eff: usize,
    /// The FULL ascending spectrum of `N = S − τᵀτ` at the final point (dropped tail included).
    /// Same context-refusal caveat as [`Self::iters`].
    pub orb_spectrum: Vec<f64>,
    /// Smallest Schmidt value of the final `x` — the `1/σ_min` conditioning hazard of the
    /// fixed-rank manifold, measured at exactly the frames the joint blocks were assembled at.
    /// Same context-refusal caveat as [`Self::iters`].
    pub sigma_min: f64,
    /// `max|UᵀU − I|` at the final point. `U` is NEVER re-orthonormalized (a product of matrix
    /// exponentials of antisymmetric generators is orthogonal to roundoff, and a QR re-fit would
    /// introduce column-sign/permutation hazards for no benefit) — this is REPORTED instead.
    /// Above `1e-10` it is worth investigating.
    pub u_orth_dev: f64,
    /// The final leg's PD-probe Rayleigh quotient, when [`JointScfOpts::polish`] had `certify` on.
    pub hrr_min: Option<f64>,
    /// Which step rule produced this run ([`JointScfOpts::solver`]), stamped unconditionally.
    pub solver: JointSolver,
    /// [`JointSolver::TrKrylov`] only. The trust radius in effect AFTER each attempting
    /// iteration's step call — the radius the NEXT attempt starts from, pushed unconditionally
    /// (accepted steps, rejected cascades and the growth cap all show here). Empty under
    /// [`JointSolver::DenseSchur`].
    pub tr_radii: Vec<f64>,
    /// [`JointSolver::TrKrylov`] only. One entry per TRIAL in evaluation order, FLAT across
    /// iterations: finite ρ, `NaN` for a ρ-floor event, `−∞` for a non-finite trial energy.
    pub rho_history: Vec<f64>,
    /// [`JointSolver::TrKrylov`] only. Steihaug operator applications per solve, rejected
    /// attempts included.
    pub cg_iters: Vec<usize>,
    /// [`JointSolver::TrKrylov`] only. Steihaug exit status per solve, aligned with
    /// [`Self::cg_iters`].
    pub cg_statuses: Vec<SteihaugStatus>,
    /// [`JointSolver::TrKrylov`] only. Total ×¼ shrinks across the run.
    pub rejections: usize,
    /// TT legs the DRIVER ran: the initial cold leg, the post-acceptance warm legs and their
    /// cold escalations, and the refreshing legs the moving-quotient discipline forces
    /// ([`JointScfOpts::tr_legs_every`]). The dense branch's per-trial line-search legs are NOT
    /// counted — this field is the TrKrylov protocol's cost accounting.
    pub tt_legs: usize,
}

/// The result of [`joint_ground_state`].
pub struct JointScfOut {
    /// The converged reduced energy `⟨x,Hx⟩` at the returned `(u, h, x)`. Add `ints.e0` for a total.
    pub lambda: f64,
    /// The TT state — ALWAYS the output of a TT leg run under [`Self::h`].
    pub x: TtVector<f64>,
    /// The accumulated one-particle rotation, row-major `d × d`. See the type's basis-identity
    /// contract above.
    pub u: Vec<f64>,
    /// `SparseHamiltonian` of `rotate_integrals(ints, u)` — the operator `x` and `lambda` refer to.
    pub h: SparseHamiltonian<f64>,
    /// Per-run diagnostics.
    pub info: JointScfInfo,
}

/// Alternating TT ⊕ orbital-rotation ground-state optimization of the reduced energy
/// `f(U) = min_{x ∈ M_r} ⟨x,H(U)x⟩` (see the module header for the whole design and for the
/// measurements that force it).
///
/// # Panics
///
/// On a malformed problem: `ints.norb != space.d`, a non-positive tolerance, `max_ls_trials == 0`,
/// or a [`PairMask`] containing a cross-irrep pair (refused up front rather than left to trip
/// [`expm_kappa_blocked`]'s assert in the middle of a line search).
///
/// # Example
///
/// ```no_run
/// # use qntt::{Integrals, QnSpace, TtVector};
/// # use qntt::solvers::joint::{JointScfOpts, joint_ground_state};
/// # fn demo(ints: &Integrals<f64>, space: QnSpace, x0: &TtVector<f64>) {
/// let out = joint_ground_state(ints, space, x0, &JointScfOpts::default());
/// assert!(out.info.converged, "{:?}", out.info.refused_reason);
/// // `out.x` / `out.h` live in the basis `out.u`; carry it with them.
/// let total = out.lambda + ints.e0;
/// # let _ = total;
/// # }
/// ```
pub fn joint_ground_state(
    ints: &Integrals<f64>,
    space: QnSpace,
    x0: &TtVector<f64>,
    opts: &JointScfOpts,
) -> JointScfOut {
    let d = space.d;
    assert_eq!(
        ints.norb, d,
        "joint_ground_state: integrals are over {} orbitals but the space has d = {d}",
        ints.norb
    );
    assert!(
        opts.orb_grad > 0.0 && opts.tt_grad_tol > 0.0 && opts.orb_floor >= 0.0,
        "joint_ground_state: orb_grad/tt_grad_tol must be positive and orb_floor non-negative"
    );
    assert!(
        opts.max_ls_trials >= 1,
        "joint_ground_state: max_ls_trials must be at least 1"
    );
    assert!(
        opts.tr_init > 0.0 && opts.tr_init.is_finite() && opts.tr_init * opts.tr_init > 0.0,
        "joint_ground_state: tr_init must be a positive, finite trust radius with a \
         representable square (got {:e}); below ~1.5e-154 the boundary quadratic evaluates 0/0",
        opts.tr_init
    );
    assert!(
        opts.tr_max.is_finite() && opts.tr_max >= opts.tr_init,
        "joint_ground_state: tr_max ({:e}) must be finite and >= tr_init ({:e})",
        opts.tr_max,
        opts.tr_init
    );
    assert!(
        opts.tr_cg_max_iter >= 1,
        "joint_ground_state: tr_cg_max_iter must be at least 1"
    );
    assert!(
        opts.tr_legs_every >= 1,
        "joint_ground_state: tr_legs_every is a period in accepted steps and must be at least 1 \
         (1 = the shipped leg-per-step protocol)"
    );
    assert!(
        opts.tr_legs_every == 1 || opts.solver == JointSolver::TrKrylov,
        "joint_ground_state: tr_legs_every > 1 is the TrKrylov moving-quotient protocol; the \
         DenseSchur branch runs a TT leg inside every line-search trial, so there is no \
         post-acceptance leg to defer there. Set solver = JointSolver::TrKrylov, or leave \
         tr_legs_every = 1."
    );
    if let Some(mask) = opts.pairs.as_ref() {
        for &(p, q) in mask.keep.iter() {
            assert!(
                p < d && q < d && space.pg.irrep_of_sites([p, q]) == Irrep::IDENTITY,
                "joint_ground_state: pair ({p},{q}) is not a legal same-irrep orbital rotation \
                 for this point group — a cross-irrep generator would break the block-diagonal \
                 rotation the whole no-re-screening argument rests on"
            );
        }
    }
    let orb_grad_full = opts.orb_grad_full.unwrap_or(opts.orb_grad);

    let mut u = identity(d);
    let mut h = build_h(ints, &u, space, opts);
    let pool = HamPool::new();
    let mut ls_build_count = 0usize;

    let mut x = x0.clone();
    normalize_mut(&mut x);
    let t_leg0 = std::time::Instant::now();
    let (mut lambda, x_leg, mut leg_info) = tt_leg(&h, &x, opts, opts.cg_max_iter_initial, None);
    if opts.verbose {
        eprintln!(
            "joint_scf: initial TT leg λ={lambda:.12e} ‖g‖={:.3e} in {:?}",
            leg_info.gnorm,
            t_leg0.elapsed()
        );
    }
    x = x_leg;

    let mut info = JointScfInfo {
        converged: false,
        stalled_at_floor: false,
        refused_reason: None,
        iters: 0,
        e_history: Vec::new(),
        g_orb_kept: Vec::new(),
        g_orb_full: Vec::new(),
        g_orb_dropped: Vec::new(),
        tt_gradnorms: Vec::new(),
        delta_k_norms: Vec::new(),
        trials_per_iter: Vec::new(),
        ls_build_count: 0,
        n_eff: 0,
        orb_spectrum: Vec::new(),
        rank_limited: false,
        g_dropped_final: 0.0,
        dropped_leak_bound: 0.0,
        dropped_angle: f64::NAN,
        sigma_min: f64::INFINITY,
        u_orth_dev: 0.0,
        hrr_min: leg_info.hrr_min,
        solver: opts.solver,
        tr_radii: Vec::new(),
        rho_history: Vec::new(),
        cg_iters: Vec::new(),
        cg_statuses: Vec::new(),
        rejections: 0,
        tt_legs: 1,
    };

    let mut tau_prev = 1.0f64;
    let mut delta = opts.tr_init;
    let mut since_leg = 0usize;
    let mut tt_fresh = true;

    for it in 0..opts.orb_max_iter {
        let jopts = JointOpts {
            tt_grad_tol: if tt_fresh {
                opts.tt_grad_tol
            } else {
                f64::INFINITY
            },
            orb_floor: opts.orb_floor,
            pairs: opts.pairs.clone(),
            z_route: opts.z_route,
            b_general: !tt_fresh,
        };
        let t_ctx = std::time::Instant::now();
        let jctx = match joint_context(&h, &x, &jopts) {
            Ok(c) => c,
            Err(e) => {
                info.iters = it;
                info.refused_reason = Some(format!(
                    "joint_ground_state: the joint context refused at orbital iteration {it} — {e}"
                ));
                break;
            }
        };
        info.iters = it + 1;
        let t_ctx = t_ctx.elapsed();

        lambda = jctx.lambda();
        let g = orbital_gradient(&h, &x);
        let gl: Vec<f64> = jctx.pairs().iter().map(|&(p, q)| g.get(&[p, q])).collect();
        let gk = jctx.vt_times(&gl);
        let gl_norm = norm2(&gl);
        let gk_norm = norm2(&gk);
        let g_dropped = {
            let back = jctx.v_times(&gk);
            norm2(&sub(&gl, &back))
        };

        info.e_history.push(lambda);
        info.g_orb_kept.push(gk_norm);
        info.g_orb_full.push(gl_norm);
        info.g_orb_dropped.push(g_dropped);
        info.tt_gradnorms.push(jctx.tt_gradnorm());
        info.n_eff = jctx.n_eff();
        info.orb_spectrum = jctx.orb_spectrum().to_vec();
        info.sigma_min = jctx.sigma_min();
        info.u_orth_dev = orth_dev(&u, d);

        if !tt_fresh && (gk_norm <= opts.orb_grad || jctx.n_eff() == 0) {
            refresh_leg(
                &h,
                opts,
                &mut x,
                &mut lambda,
                &mut leg_info,
                &mut info,
                &mut since_leg,
                &mut tt_fresh,
                it,
                "exit candidate",
            );
            continue;
        }
        let tt_ok = jctx.tt_gradnorm() <= opts.tt_grad_tol * lambda.abs().max(1.0);
        if gk_norm <= opts.orb_grad && tt_ok {
            info.converged = true;
            stamp_dropped_diagnostic(
                &mut info,
                jctx.orb_spectrum(),
                jctx.orb_floor(),
                g_dropped,
                orb_grad_full,
            );
            break;
        }
        if jctx.n_eff() == 0 {
            info.refused_reason = Some(format!(
                "joint_ground_state: the orbital sector is metric-null (n_eff == 0) but the TT \
                 gate is unmet at orbital iteration {it} (‖P((H−λ)x)‖ = {:.6e} against \
                 {:.6e}). No orbital step can improve this point; converge x0 further, or lower \
                 orb_floor if the sector should not have been empty.",
                jctx.tt_gradnorm(),
                opts.tt_grad_tol * lambda.abs().max(1.0)
            ));
            break;
        }
        if it + 1 == opts.orb_max_iter {
            info.refused_reason = Some(format!(
                "joint_ground_state: orbital iteration cap ({}) reached with ‖gk‖ = {gk_norm:.6e} \
                 (gate {:.3e}), ‖gL‖ = {gl_norm:.6e} (gate {orb_grad_full:.3e}).",
                opts.orb_max_iter, opts.orb_grad
            ));
            break;
        }

        if opts.solver == JointSolver::TrKrylov {
            let ghat = jctx.gradient_raised(&gk);
            let ghat_norm = ghat.norm();
            if info.e_history.len() >= 4 {
                let n = info.e_history.len();
                let best_recent = info.e_history[n - 3..]
                    .iter()
                    .fold(f64::INFINITY, |m, &v| m.min(v));
                let before = info.e_history[..n - 3]
                    .iter()
                    .fold(f64::INFINITY, |m, &v| m.min(v));
                if before - best_recent < energy_floor(lambda) && gk_norm <= GRAD_SANITY {
                    if !tt_fresh {
                        refresh_leg(
                            &h,
                            opts,
                            &mut x,
                            &mut lambda,
                            &mut leg_info,
                            &mut info,
                            &mut since_leg,
                            &mut tt_fresh,
                            it,
                            "TR exit candidate",
                        );
                        continue;
                    }
                    info.stalled_at_floor = true;
                    info.converged = true;
                    if opts.verbose {
                        eprintln!(
                            "joint_scf it={it}: TR stagnation exit — no resolvable energy \
                             improvement over 3 iterations at ‖gk‖={gk_norm:.3e} ⇒ converged \
                             (stalled_at_floor)"
                        );
                    }
                    break;
                }
            }
            if ghat_norm <= TR_RESOLUTION_FLOOR * lambda.abs().max(1.0) {
                if !tt_fresh {
                    refresh_leg(
                        &h,
                        opts,
                        &mut x,
                        &mut lambda,
                        &mut leg_info,
                        &mut info,
                        &mut since_leg,
                        &mut tt_fresh,
                        it,
                        "TR exit candidate",
                    );
                    continue;
                }
                if gk_norm <= GRAD_SANITY {
                    info.stalled_at_floor = true;
                    info.converged = true;
                    if opts.verbose {
                        eprintln!(
                            "joint_scf it={it}: TR floor exit — ‖ĝ‖_G={ghat_norm:.3e} below the \
                             resolution floor at ‖gk‖={gk_norm:.3e} ⇒ converged \
                             (stalled_at_floor)"
                        );
                    }
                    break;
                }
                info.refused_reason = Some(format!(
                    "joint_ground_state: raised joint gradient at the trust-region resolution \
                     floor (‖ĝ‖_G = {ghat_norm:.6e} ≤ {TR_RESOLUTION_FLOOR:.0e}·max(1,|λ|) = \
                     {:.3e}) at iteration {it} while the two-sided gate is unmet — below the \
                     floor every admissible radius predicts a decrease inside the roundoff of \
                     an O(1) energy, so the ratio test is a coin flip. The gates are tighter \
                     than this point can resolve: loosen orb_grad, or lower orb_floor if the \
                     unmet side is the full-pair gradient.",
                    TR_RESOLUTION_FLOOR * lambda.abs().max(1.0)
                ));
                break;
            }
            let tr_opts = TrOpts {
                tr_max: opts.tr_max,
                max_rej: opts.tr_max_rej,
                cg_max_iter: opts.tr_cg_max_iter,
                cg_rtol: opts.tr_cg_rtol,
                verbose: opts.verbose,
            };
            let delta_prev = delta;
            let t_tr = std::time::Instant::now();
            let msolve: Option<&dyn Fn(&JointVector) -> JointVector> = None;
            let ball_dot: super::tr::BallDot<'_, JointVector> = (opts.tr_ball == TrBall::Chart)
                .then_some(&chart_ball_dot as &dyn Fn(&JointVector, &JointVector) -> f64);
            let (acc, delta_new, st) = tr_krylov_step(
                |v| match opts.tr_model {
                    TrModel::Full => jctx.hess_apply_deflated(v),
                    TrModel::WLess => jctx.hess_apply_deflated_wless(v),
                },
                &ghat,
                lambda,
                delta,
                &tr_opts,
                msolve,
                ball_dot,
                |s: &JointVector| {
                    let a = jctx.v_times(&s.c);
                    let dkn = std::f64::consts::SQRT_2 * norm2(&a);
                    let kstep = jctx.kappa_matrix(&scaled(&a, -1.0));
                    let u_step = if space.pg.order() > 1 {
                        expm_kappa_blocked(&kstep, space.pg)
                    } else {
                        expm_antisym(&flatten(&kstep, space.d), space.d)
                    };
                    let u_try = matmul(&u, &u_step, space.d);
                    let h_try = pool.at(ints, &u_try, space, opts);
                    ls_build_count += 1;
                    let mut x_try = s.xi.retract_hosvd(1.0);
                    normalize_mut(&mut x_try);
                    let e_raw = rayleigh_quotient(&h_try, &x_try);
                    let (e_try, x_keep) = match opts.tr_score {
                        TrScore::AlsRounded => {
                            let aopts = AlsOptions {
                                max_iter: 1,
                                eps: 1e-9,
                                ..AlsOptions::default()
                            };
                            let (lam_a, x_a) = als(&h_try, &x_try, &aopts);
                            if lam_a < e_raw {
                                (lam_a, x_a)
                            } else {
                                (e_raw, x_try)
                            }
                        }
                    };
                    pool.park(u_try.clone(), h_try);
                    TrTrial {
                        payload: (u_try, x_keep),
                        e_try,
                        delta_k_norm: dkn,
                    }
                },
            );
            delta = delta_new;
            info.tr_radii.push(delta);
            info.rho_history.extend_from_slice(&st.rhos);
            info.cg_iters.extend_from_slice(&st.cg_iters);
            info.cg_statuses.extend_from_slice(&st.cg_statuses);
            info.rejections += st.rejections;
            info.trials_per_iter.push(st.rhos.len());
            if opts.verbose {
                eprintln!(
                    "joint_scf it={it}: λ={lambda:.12e} ‖gk‖={gk_norm:.3e} ‖ĝ‖_G={ghat_norm:.3e} \
                     n_eff={} | ctx {t_ctx:.2?} | tr {} trial(s) Δ→{delta:.3e} {:.2?} accepted={}",
                    jctx.n_eff(),
                    st.rhos.len(),
                    t_tr.elapsed(),
                    acc.is_some()
                );
            }
            match acc {
                None => {
                    if delta < delta_prev {
                        continue;
                    }
                    if !tt_fresh {
                        refresh_leg(
                            &h,
                            opts,
                            &mut x,
                            &mut lambda,
                            &mut leg_info,
                            &mut info,
                            &mut since_leg,
                            &mut tt_fresh,
                            it,
                            "TR non-shrinking refusal",
                        );
                        continue;
                    }
                    if gk_norm <= GRAD_SANITY && info.rho_history.last().is_some_and(|r| r.is_nan())
                    {
                        info.stalled_at_floor = true;
                        info.converged = true;
                        if opts.verbose {
                            eprintln!(
                                "joint_scf it={it}: TR floor exit — ρ-floor \
                                 refusal at an unshrinkable radius, ‖gk‖={gk_norm:.3e} ⇒ \
                                 converged (stalled_at_floor)"
                            );
                        }
                        break;
                    }
                    info.refused_reason = Some(format!(
                        "joint_ground_state: the trust-region step refused at an unshrinkable \
                         radius (Δ = {delta:.3e}) at iteration {it} with ‖gk‖ = {gk_norm:.6e}, \
                         ‖ĝ‖_G = {ghat_norm:.6e}: the model's predicted decrease is at the \
                         energy's resolution floor and the trial energy is resolvably worse, \
                         so no radius can certify a step here."
                    ));
                    break;
                }
                Some(tr) => {
                    let (u_try, x_try) = tr.payload;
                    let h_new = pool.take_parked(ints, &u_try, space, opts);
                    pool.recycle(std::mem::replace(&mut h, h_new));
                    u = u_try;
                    info.delta_k_norms.push(tr.delta_k_norm);
                    since_leg += 1;
                    if since_leg < opts.tr_legs_every {
                        x = x_try;
                        normalize_mut(&mut x);
                        lambda = tr.e_try;
                        tt_fresh = false;
                        continue;
                    }
                    since_leg = 0;
                    tt_fresh = true;
                    let leg_accuracy = None;
                    let (lam_w, x_w, leg_w) =
                        tt_leg(&h, &x_try, opts, opts.cg_max_iter, leg_accuracy);
                    x = x_w;
                    lambda = lam_w;
                    leg_info = leg_w;
                    info.hrr_min = leg_info.hrr_min;
                    info.tt_legs += 1;
                    if tr.delta_k_norm > opts.warm_threshold {
                        let (lam2, x2, leg2) =
                            tt_leg(&h, &x, opts, opts.cg_max_iter_initial, leg_accuracy);
                        info.tt_legs += 1;
                        if lam2 < lambda {
                            x = x2;
                            lambda = lam2;
                            leg_info = leg2;
                            info.hrr_min = leg_info.hrr_min;
                        }
                    }
                    continue;
                }
            }
        }

        let t_schur = std::time::Instant::now();
        let (dc, decrement) = schur_newton_step(&jctx, &gk, opts);
        let a_newton = jctx.v_times(&dc);
        let t_schur = t_schur.elapsed();

        let mut trials_here = 0usize;
        let t_ls = std::time::Instant::now();
        let mut accepted = linesearch(
            ints,
            space,
            &jctx,
            &u,
            &x,
            lambda,
            &gl,
            &a_newton,
            tau_prev.min(1.0),
            opts,
            &pool,
            &mut ls_build_count,
            &mut trials_here,
        );
        if accepted.is_none() {
            let a_sd = jctx.v_times(&scaled(&gk, -0.5));
            let n_sd = std::f64::consts::SQRT_2 * norm2(&a_sd);
            if n_sd > 0.0 {
                let a_sd = scaled(&a_sd, opts.trust0 / n_sd);
                accepted = linesearch(
                    ints,
                    space,
                    &jctx,
                    &u,
                    &x,
                    lambda,
                    &gl,
                    &a_sd,
                    1.0,
                    opts,
                    &pool,
                    &mut ls_build_count,
                    &mut trials_here,
                );
            }
        }
        if opts.verbose {
            eprintln!(
                "joint_scf it={it}: λ={lambda:.12e} ‖gk‖={gk_norm:.3e} ‖gL‖={gl_norm:.3e} \
                 n_eff={} | ctx {t_ctx:.2?} schur {t_schur:.2?} | ls {trials_here} trial(s) \
                 {:.2?} accepted={}",
                jctx.n_eff(),
                t_ls.elapsed(),
                accepted.is_some()
            );
        }
        info.trials_per_iter.push(trials_here);

        let Some(step) = accepted else {
            let efloor = energy_floor(lambda);
            if decrement <= efloor && gk_norm <= GRAD_SANITY {
                info.stalled_at_floor = true;
                info.converged = true;
                if opts.verbose {
                    eprintln!(
                        "joint_scf it={it}: floor exit — decrement={decrement:.3e} ≤ \
                         efloor={efloor:.3e}, ‖gk‖={gk_norm:.3e} ⇒ converged (stalled_at_floor)"
                    );
                }
                break;
            }
            info.refused_reason = Some(if gk_norm <= opts.armijo_floor * lambda.abs().max(1.0) {
                format!(
                    "joint_ground_state: orbital gradient at the Armijo resolution floor \
                     (‖gk‖ = {gk_norm:.6e} ≤ armijo_floor·max(1,|λ|) = {:.3e}) at iteration {it}, \
                     yet the model still promises decrease = {decrement:.3e} > efloor = \
                     {efloor:.3e}: the acceptance margin c₁·τ·|D| ∝ ‖gk‖²/κ is inside the roundoff \
                     of an O(1) energy re-evaluated by an independent TT leg per trial, so no step \
                     can be certified and iterating would be a coin flip. A deeper point needs a \
                     resolvable line search (exact-slope trust region), not a tighter gate.",
                    opts.armijo_floor * lambda.abs().max(1.0)
                )
            } else {
                format!(
                    "joint_ground_state: orbital line search stalled at iteration {it} with \
                     ‖gk‖ = {gk_norm:.6e} (well above the Armijo resolution floor {:.3e}) and \
                     E = {lambda:.12e} after {trials_here} trial(s); a direction whose analytic \
                     slope is non-negative is refused before any trial is spent, so a low count \
                     can mean the Newton direction never reached the ladder at all (measured at \
                     the Phase-3 pinned stall: an ascent Newton direction, zero Newton trials, \
                     all trials on steepest descent). This is NOT the expected roundoff-floor \
                     exit — suspect the curvature or the step scaling.",
                    opts.armijo_floor * lambda.abs().max(1.0)
                )
            });
            break;
        };

        info.delta_k_norms.push(step.step_norm);
        tau_prev = (2.0 * step.tau).min(1.0);
        u = step.u;
        pool.recycle(std::mem::replace(&mut h, step.h));
        x = step.x;
        lambda = step.lambda;
        leg_info = step.leg;
        info.hrr_min = leg_info.hrr_min;
    }

    if !info.converged && info.refused_reason.is_none() {
        info.refused_reason = Some(
            "joint_ground_state: stopped without certifying and without a recorded obstruction \
             (orb_max_iter == 0?)."
                .to_string(),
        );
    }
    info.ls_build_count = ls_build_count;
    info.u_orth_dev = orth_dev(&u, d);

    JointScfOut {
        lambda,
        x,
        u,
        h,
        info,
    }
}

#[allow(clippy::too_many_arguments)]
fn refresh_leg(
    h: &SparseHamiltonian<f64>,
    opts: &JointScfOpts,
    x: &mut TtVector<f64>,
    lambda: &mut f64,
    leg_info: &mut NewtonPolishInfo,
    info: &mut JointScfInfo,
    since_leg: &mut usize,
    tt_fresh: &mut bool,
    it: usize,
    why: &str,
) {
    let (lam_r, x_r, leg_r) = tt_leg(h, x, opts, opts.cg_max_iter, None);
    *x = x_r;
    *lambda = lam_r;
    *leg_info = leg_r;
    info.hrr_min = leg_info.hrr_min;
    info.tt_legs += 1;
    *since_leg = 0;
    *tt_fresh = true;
    if opts.verbose {
        eprintln!(
            "joint_scf it={it}: moving-quotient refresh — {why} at a stale point, TT leg \
             re-run before deciding (λ={lambda:.12e})"
        );
    }
}

fn tt_leg(
    h: &SparseHamiltonian<f64>,
    x_warm: &TtVector<f64>,
    opts: &JointScfOpts,
    cg_cap: usize,
    accuracy_override: Option<f64>,
) -> (f64, TtVector<f64>, NewtonPolishInfo) {
    let mut x = x_warm.clone();
    normalize_mut(&mut x);

    let lam0 = rayleigh_quotient(h, &x);
    let abs_gate = opts.tt_grad_tol * lam0.abs().max(1.0);
    let accuracy = accuracy_override.unwrap_or(opts.orb_grad * opts.tt_leg_margin);
    let polish_tol = abs_gate.min(accuracy).max(abs_gate * 1e-4);
    let g0 = {
        let (bl, br) = tangent_bases(&x);
        WeingartenCtx::from_parts(h, bl, br, lam0).gradient().norm()
    };

    if g0 > opts.cg_skip_factor * polish_tol {
        let mut copts = RiemannianOpts::new(&x.space);
        copts.eps = polish_tol / lam0.abs().max(f64::MIN_POSITIVE);
        copts.max_iter = cg_cap;
        copts.preserve_grid = opts.tt_preserve_grid;
        let (_l, xc) = riemannian_cg(h, &x, &copts);
        x = xc;
    }

    let popts = NewtonPolishOpts {
        gradtol: polish_tol,
        ..opts.polish
    };
    let (mut xp, mut pinfo) = newton_polish(h, &x, &popts);

    if !pinfo.converged && cg_cap < opts.cg_max_iter_initial {
        let mut ropts = RiemannianOpts::new(&xp.space);
        ropts.eps = polish_tol / lam0.abs().max(f64::MIN_POSITIVE);
        ropts.max_iter = cg_cap.saturating_mul(4).max(cg_cap + 1);
        ropts.preserve_grid = opts.tt_preserve_grid;
        let (_l, xc) = riemannian_cg(h, &xp, &ropts);
        let (xp2, pinfo2) = newton_polish(h, &xc, &popts);
        xp = xp2;
        pinfo = pinfo2;
    }

    normalize_mut(&mut xp);
    let lam = rayleigh_quotient(h, &xp);
    (lam, xp, pinfo)
}

/// How the `Hθθ⁻¹` of the Schur condensation (§3) is realized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchurRoute {
    /// Assemble the `n_tt × n_tt` tangent Gram once and invert it by symmetric eigendecomposition.
    /// **The default.**
    Dense,
    /// Realize `Hθθ` as an operator and invert it by `n_eff + 1` preconditioned-CG solves — no
    /// tangent Gram, no dense eigendecomposition, `O(n_tt)` memory.
    MatrixFree,
}

#[allow(clippy::type_complexity)]
fn schur_blocks(
    jctx: &JointContext<'_>,
    keep_raw: bool,
) -> (
    Vec<TtTangent<f64>>,
    Vec<TtTangent<f64>>,
    Vec<f64>,
    TtTangent<f64>,
    Option<Vec<TtTangent<f64>>>,
) {
    schur_blocks_staged(jctx, None, keep_raw)
}

#[allow(clippy::type_complexity)]
fn schur_blocks_staged(
    jctx: &JointContext<'_>,
    stages: Option<&mut [f64; 3]>,
    keep_raw: bool,
) -> (
    Vec<TtTangent<f64>>,
    Vec<TtTangent<f64>>,
    Vec<f64>,
    TtTangent<f64>,
    Option<Vec<TtTangent<f64>>>,
) {
    let n = jctx.n_eff();
    let wctx = jctx.weingarten();
    let t_basis = std::time::Instant::now();

    let (bl, br) = jctx.shared_bases();
    let tb: Vec<JointVector> = tangent_basis(bl, br, true)
        .into_iter()
        .map(|xi| JointVector::from_tangent(jctx.spec(), xi))
        .collect();
    let basis_secs = t_basis.elapsed().as_secs_f64();
    let t_mgs = std::time::Instant::now();
    let shear: Vec<TtTangent<f64>> = (0..n)
        .map(|j| {
            let mut c = vec![0.0f64; n];
            c[j] = 1.0;
            let mut seed = JointVector::from_coeffs(jctx.spec(), jctx, c);
            for _pass in 0..2 {
                for b in tb.iter() {
                    let cf = b.dot(&seed);
                    seed.axpy(-cf, b);
                }
            }
            seed.xi
        })
        .collect();
    let raw_tb: Option<Vec<TtTangent<f64>>> = if keep_raw {
        Some(tb.into_iter().map(|b| b.xi).collect())
    } else {
        drop(tb);
        None
    };
    let mgs_secs = t_mgs.elapsed().as_secs_f64();
    let t_cov = std::time::Instant::now();

    let mut hthc: Vec<TtTangent<f64>> = Vec::with_capacity(n);
    let mut hcc = vec![0.0f64; n * n];
    for j in 0..n {
        let mut e = vec![0.0f64; n];
        e[j] = 1.0;
        let a_j = jctx.v_times(&e);
        let (f_xi, fc_full) = jctx.ab_covector(&shear[j], &a_j);
        let fc_kept = jctx.vt_times(&fc_full);
        for (i, &val) in fc_kept.iter().enumerate() {
            hcc[i * n + j] = val + shear[i].dot(&f_xi);
        }
        hthc.push(f_xi);
    }
    let f_theta = wctx.gradient().clone();
    if let Some(s) = stages {
        *s = [basis_secs, mgs_secs, t_cov.elapsed().as_secs_f64()];
    }

    (shear, hthc, hcc, f_theta, raw_tb)
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevSchurArms {
    /// The census pass: dense arm alone, matrix-free fields `NaN`.
    DenseOnly,
    /// Both arms on the shared prefix, with the two `rel_diff` agreement columns.
    Both,
    /// The at-scale pass: matrix-free alone, dense and `rel_diff` fields `NaN` — a wall-clock
    /// measurement, never an accuracy claim.
    MfOnly,
    /// The prefix breakdown alone: neither route runs, and the three `prefix_*_secs` stage clocks
    /// are the payload — the pass that answers WHERE the shared-prefix wall goes. At production
    /// scale the prefix outweighs either route arm.
    PrefixOnly,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct DevSchurRouteTiming {
    /// Tangent-space dimension — the size the dense arm cubes.
    pub n_tt: usize,
    /// Kept orbital directions — the number of right-hand sides the matrix-free arm solves (+1).
    pub n_eff: usize,
    /// Seconds to build the shared prefix (shear + `n_eff` covectors). Charged to NEITHER arm.
    pub prefix_secs: f64,
    /// Prefix stage 1: the raw tangent-basis build (`n_tt` tangents wrapped as joint vectors).
    pub prefix_basis_secs: f64,
    /// Prefix stage 2: the shear's two-pass MGS — `2·n_eff·n_tt` joint dots + axpys.
    pub prefix_mgs_secs: f64,
    /// Prefix stage 3: the `n_eff` covector columns (`ab_covector` builds) + `Hcc` pairings.
    pub prefix_cov_secs: f64,
    /// Seconds in `schur_terms_dense`.
    pub dense_secs: f64,
    /// Seconds in `schur_terms_matrix_free`.
    pub mf_secs: f64,
    /// `‖Schur_mf − Schur_dense‖_F / ‖Schur_dense‖_F`.
    pub schur_rel_diff: f64,
    /// `‖coupling_mf − coupling_dense‖₂ / ‖coupling_dense‖₂`.
    pub coupling_rel_diff: f64,
}

#[doc(hidden)]
pub fn dev_schur_route_timing(
    jctx: &JointContext<'_>,
    opts: &JointScfOpts,
    arms: DevSchurArms,
) -> DevSchurRouteTiming {
    let n = jctx.n_eff();
    let mut stages = [f64::NAN; 3];
    let t0 = std::time::Instant::now();
    let (_shear, hthc, _hcc, f_theta, raw_tb) = schur_blocks_staged(jctx, Some(&mut stages), true);
    let prefix_secs = t0.elapsed().as_secs_f64();
    let raw_tb = raw_tb.expect("schur_blocks_staged: keep_raw returns the raw basis");
    let n_tt = raw_tb.len();

    let mut out = DevSchurRouteTiming {
        n_tt,
        n_eff: n,
        prefix_secs,
        prefix_basis_secs: stages[0],
        prefix_mgs_secs: stages[1],
        prefix_cov_secs: stages[2],
        dense_secs: f64::NAN,
        mf_secs: f64::NAN,
        schur_rel_diff: f64::NAN,
        coupling_rel_diff: f64::NAN,
    };
    if arms == DevSchurArms::PrefixOnly {
        return out;
    }

    let dense = (arms != DevSchurArms::MfOnly).then(|| {
        let t1 = std::time::Instant::now();
        let dc = schur_terms_dense(jctx, raw_tb, &hthc, &f_theta, opts);
        out.dense_secs = t1.elapsed().as_secs_f64();
        dc
    });
    if arms == DevSchurArms::DenseOnly {
        return out;
    }

    let t2 = std::time::Instant::now();
    let (sm, cm) = schur_terms_matrix_free(jctx, &hthc, &f_theta, opts);
    out.mf_secs = t2.elapsed().as_secs_f64();

    if let Some((sd, cd)) = dense {
        let nrm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let dif = |a: &[f64], b: &[f64]| -> f64 {
            let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
            nrm(&d) / nrm(b).max(1e-300)
        };
        out.schur_rel_diff = dif(&sm, &sd);
        out.coupling_rel_diff = dif(&cm, &cd);
    }
    out
}

fn schur_newton_step(jctx: &JointContext<'_>, gk: &[f64], opts: &JointScfOpts) -> (Vec<f64>, f64) {
    let n = jctx.n_eff();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let (shear, hthc, hcc, f_theta, raw_tb) =
        schur_blocks(jctx, opts.schur_route == SchurRoute::Dense);

    let (schur, coupling) = match opts.schur_route {
        SchurRoute::Dense => schur_terms_dense(
            jctx,
            raw_tb.expect("schur_blocks: keep_raw on the dense route"),
            &hthc,
            &f_theta,
            opts,
        ),
        SchurRoute::MatrixFree => schur_terms_matrix_free(jctx, &hthc, &f_theta, opts),
    };

    let mut heff = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            heff[i * n + j] = hcc[i * n + j] - schur[i * n + j];
        }
    }
    let heff = symmetrize(&heff, n);
    let rhs: Vec<f64> = (0..n)
        .map(|i| coupling[i] - 0.5 * gk[i] - shear[i].dot(&f_theta))
        .collect();

    let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&heff, n, n);
    let dpd = opts.eig_floor_rel * evals.iter().fold(1e-14f64, |m, &e| m.max(e.abs()));
    let mut dc = vec![0.0f64; n];
    let mut decrement = 0.0f64;
    for (k, &ev) in evals.iter().enumerate() {
        let proj: f64 = (0..n).map(|j| evecs[k][j] * rhs[j]).sum();
        let denom = ev.abs().max(dpd);
        let s = proj / denom;
        decrement += 0.5 * proj * proj / denom;
        for (i, out) in dc.iter_mut().enumerate() {
            *out += evecs[k][i] * s;
        }
    }
    (dc, decrement)
}

fn schur_terms_dense(
    jctx: &JointContext<'_>,
    raw_tb: Vec<TtTangent<f64>>,
    hthc: &[TtTangent<f64>],
    f_theta: &TtTangent<f64>,
    _opts: &JointScfOpts,
) -> (Vec<f64>, Vec<f64>) {
    let n = hthc.len();
    let wctx = jctx.weingarten();
    let (bl, br) = jctx.shared_bases();
    let tb: Vec<TtTangent<f64>> = raw_tb
        .into_iter()
        .map(|xi| TtTangent::project_shared(bl, br, &xi.to_vector()))
        .collect();
    let n_tt = tb.len();
    if n_tt == 0 {
        return (vec![0.0; n * n], vec![0.0; n]);
    }

    let mut htt = vec![0.0f64; n_tt * n_tt];
    for (j, xj) in tb.iter().enumerate() {
        let mut fj = wctx.a_apply_fused(xj);
        fj.axpy(1.0, &wctx.b_apply(xj));
        for (i, xi) in tb.iter().enumerate() {
            htt[i * n_tt + j] = xi.dot(&fj);
        }
    }
    let htt = symmetrize(&htt, n_tt);

    let mut hthc_c = vec![0.0f64; n_tt * n];
    for (j, cj) in hthc.iter().enumerate() {
        for (i, xi) in tb.iter().enumerate() {
            hthc_c[i * n + j] = xi.dot(cj);
        }
    }
    let f_theta_c: Vec<f64> = tb.iter().map(|xi| xi.dot(f_theta)).collect();

    let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&htt, n_tt, n_tt);
    let anchor = evals.iter().fold(1.0f64, |m, &e| m.max(e.abs()));
    let dtheta = 1e-10 * anchor;
    let solve_theta = |y: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0f64; n_tt];
        for (k, &ev) in evals.iter().enumerate() {
            let proj: f64 = (0..n_tt).map(|i| evecs[k][i] * y[i]).sum();
            let s = proj / ev.max(dtheta);
            for (i, o) in out.iter_mut().enumerate() {
                *o += evecs[k][i] * s;
            }
        }
        out
    };

    let mut schur = vec![0.0f64; n * n];
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for j in 0..n {
        let col: Vec<f64> = (0..n_tt).map(|i| hthc_c[i * n + j]).collect();
        cols.push(solve_theta(&col));
    }
    for i in 0..n {
        for j in 0..n {
            schur[i * n + j] = (0..n_tt).map(|k| hthc_c[k * n + i] * cols[j][k]).sum();
        }
    }
    let w = solve_theta(&f_theta_c);
    let coupling: Vec<f64> = (0..n)
        .map(|i| (0..n_tt).map(|k| hthc_c[k * n + i] * w[k]).sum())
        .collect();
    (schur, coupling)
}

fn schur_terms_matrix_free(
    jctx: &JointContext<'_>,
    hthc: &[TtTangent<f64>],
    f_theta: &TtTangent<f64>,
    opts: &JointScfOpts,
) -> (Vec<f64>, Vec<f64>) {
    let n = hthc.len();
    let wctx = jctx.weingarten();
    let (bl, br) = jctx.shared_bases();
    let radial = jctx.radial();

    let dprecond = opts.polish.precond.then(|| {
        let dg = wctx.a_diagonal();
        let mut maxd = 0.0f64;
        dg.for_each_coeff(|c| maxd = maxd.max(c.abs()));
        (dg, 1e-2 * maxd.max(1e-300))
    });
    let purify = |v: &TtTangent<f64>| -> TtTangent<f64> {
        let mut p = v.clone();
        reproject_tangent(bl, br, radial, &mut p);
        p
    };
    let mut mu = 0.0f64;
    let solve = |b: &TtTangent<f64>, mu: &mut f64| -> TtTangent<f64> {
        let bd = purify(b);
        loop {
            let (y, flag, _iters) = pcg_solve(
                |p: &TtTangent<f64>| {
                    let mut q = p.clone();
                    reproject_tangent(bl, br, radial, &mut q);
                    let mut hq = wctx.a_apply_fused(&q);
                    hq.axpy(1.0, &wctx.b_apply(&q));
                    if *mu > 0.0 {
                        hq.axpy(*mu, &q);
                    }
                    deflate_radial(radial, &mut hq);
                    hq
                },
                |t: &mut TtTangent<f64>| {
                    if let Some((dg, ridge)) = dprecond.as_ref() {
                        t.zip_coeffs_mut(dg, |ti, di| *ti /= di.abs() + ridge);
                    }
                },
                &bd,
                opts.schur_tol,
                opts.schur_max_iter,
            );
            if let PcgFlag::NegativeCurvature { rho } = flag {
                let next = (10.0 * *mu).max(1.5 * rho.abs()).max(1e-10);
                if next < MU_CAP {
                    *mu = next;
                    continue;
                }
            }
            *mu /= 3.0;
            if *mu < 1e-14 {
                *mu = 0.0;
            }
            break purify(&y);
        }
    };

    let ys: Vec<TtTangent<f64>> = hthc.iter().map(|c| solve(c, &mut mu)).collect();
    let w = solve(f_theta, &mut mu);

    let mut schur = vec![0.0f64; n * n];
    let hthc_p: Vec<TtTangent<f64>> = hthc.iter().map(&purify).collect();
    for i in 0..n {
        for j in 0..n {
            schur[i * n + j] = hthc_p[i].dot(&ys[j]);
        }
    }
    let coupling: Vec<f64> = (0..n).map(|i| hthc_p[i].dot(&w)).collect();
    (schur, coupling)
}

struct Accepted {
    u: Vec<f64>,
    h: SparseHamiltonian<f64>,
    x: TtVector<f64>,
    lambda: f64,
    tau: f64,
    step_norm: f64,
    leg: NewtonPolishInfo,
}

#[allow(clippy::too_many_arguments)]
fn linesearch(
    ints: &Integrals<f64>,
    space: QnSpace,
    jctx: &JointContext<'_>,
    u_acc: &[f64],
    x: &TtVector<f64>,
    e0: f64,
    gl: &[f64],
    a: &[f64],
    tau0: f64,
    opts: &JointScfOpts,
    pool: &HamPool,
    ls_build_count: &mut usize,
    n_trials: &mut usize,
) -> Option<Accepted> {
    let s = std::f64::consts::SQRT_2 * norm2(a);
    if s == 0.0 {
        return None;
    }
    let a: Vec<f64> = if s > opts.trust0 {
        scaled(a, opts.trust0 / s)
    } else {
        a.to_vec()
    };
    let dslope: f64 = gl.iter().zip(a.iter()).map(|(g, t)| g * t).sum();
    if dslope >= 0.0 || dslope.is_nan() {
        return None;
    }
    let step_len = std::f64::consts::SQRT_2 * norm2(&a);
    let efloor = energy_floor(e0);

    let mut tau = tau0;
    for _trial in 0..opts.max_ls_trials {
        let kstep = jctx.kappa_matrix(&scaled(&a, -tau));
        let u_step = if space.pg.order() > 1 {
            expm_kappa_blocked(&kstep, space.pg)
        } else {
            expm_antisym(&flatten(&kstep, space.d), space.d)
        };
        let u_try = matmul(u_acc, &u_step, space.d);
        let h_try = pool.at(ints, &u_try, space, opts);
        *ls_build_count += 1;
        *n_trials += 1;

        let (e_try, x_try, leg) = tt_leg(&h_try, x, opts, opts.cg_max_iter, None);

        let need = (opts.armijo_c * tau * dslope).min(-efloor);
        if e_try <= e0 + need {
            return Some(Accepted {
                u: u_try,
                h: h_try,
                x: x_try,
                lambda: e_try,
                tau,
                step_norm: tau * step_len,
                leg,
            });
        }
        pool.recycle(h_try);
        tau *= 0.5;
    }
    None
}

fn build_h(
    ints: &Integrals<f64>,
    u: &[f64],
    space: QnSpace,
    opts: &JointScfOpts,
) -> SparseHamiltonian<f64> {
    let rotated = rotate_integrals(ints, u);
    SparseHamiltonian::new_with(
        space,
        &rotated.t,
        &rotated.v,
        None,
        opts.ham_eps,
        opts.construction,
    )
}

fn mint_from_rotated(
    rotated: &Integrals<f64>,
    space: QnSpace,
    opts: &JointScfOpts,
) -> SparseHamiltonian<f64> {
    let mask = PatternMask::from_magnitude(&rotated.t, &rotated.v, opts.ham_eps, space.pg);
    SparseHamiltonian::new_refillable(
        space,
        &rotated.t,
        &rotated.v,
        None,
        TermScreen::Mask(std::sync::Arc::new(mask)),
        opts.construction,
    )
}

struct HamPool {
    spare: std::cell::RefCell<Option<SparseHamiltonian<f64>>>,
    parked: std::cell::RefCell<Option<(Vec<f64>, SparseHamiltonian<f64>)>>,
}

impl HamPool {
    fn new() -> Self {
        HamPool {
            spare: std::cell::RefCell::new(None),
            parked: std::cell::RefCell::new(None),
        }
    }

    fn at(
        &self,
        ints: &Integrals<f64>,
        u: &[f64],
        space: QnSpace,
        opts: &JointScfOpts,
    ) -> SparseHamiltonian<f64> {
        if opts.refill_h {
            let recycled = self
                .parked
                .borrow_mut()
                .take()
                .map(|(_, h)| h)
                .or_else(|| self.spare.borrow_mut().take());
            let rotated = rotate_integrals(ints, u);
            return match recycled {
                Some(mut h) if h.pattern_leak(&rotated.t, &rotated.v, opts.ham_eps) == 0 => {
                    h.refill(&rotated.t, &rotated.v, None);
                    h
                }
                _ => mint_from_rotated(&rotated, space, opts),
            };
        }
        build_h(ints, u, space, opts)
    }

    fn park(&self, u: Vec<f64>, h: SparseHamiltonian<f64>) {
        if let Some((_, old)) = self.parked.borrow_mut().replace((u, h)) {
            self.recycle(old);
        }
    }

    fn take_parked(
        &self,
        ints: &Integrals<f64>,
        u: &[f64],
        space: QnSpace,
        opts: &JointScfOpts,
    ) -> SparseHamiltonian<f64> {
        let parked = self.parked.borrow_mut().take();
        match parked {
            Some((pu, h)) => {
                assert!(
                    pu == u,
                    "HamPool: the parked trial operator is not the accepted trial's (the trust-region \
                     ladder accepted a trial other than the last evaluated)"
                );
                h
            }
            None => self.at(ints, u, space, opts),
        }
    }

    fn recycle(&self, h: SparseHamiltonian<f64>) {
        if h.is_refillable() {
            let mut spare = self.spare.borrow_mut();
            if spare.is_none() {
                *spare = Some(h);
            }
        }
    }
}

pub(crate) fn stamp_dropped_diagnostic(
    info: &mut JointScfInfo,
    orb_spectrum: &[f64],
    orb_floor: f64,
    g_dropped: f64,
    orb_grad_full: f64,
) {
    let anchor = orb_spectrum.last().copied().unwrap_or(0.0).max(1.0);
    let cut = orb_floor * anchor;
    let marginal = orb_spectrum
        .iter()
        .copied()
        .filter(|mu| *mu <= cut)
        .fold(f64::NAN, f64::max);
    info.g_dropped_final = g_dropped;
    info.dropped_leak_bound = 2.0 * cut.sqrt();
    info.dropped_angle = if marginal.is_finite() {
        marginal.max(0.0).sqrt().min(1.0).asin()
    } else {
        f64::NAN
    };
    info.rank_limited = g_dropped > orb_grad_full;
}

fn identity(d: usize) -> Vec<f64> {
    crate::rotation::identity_matrix(d)
}

fn matmul(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    crate::rotation::compose_rotations(a, b, d)
}

fn flatten(k: &CoeffTensor<f64>, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = k.get(&[i, j]);
        }
    }
    out
}

fn orth_dev(u: &[f64], d: usize) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            let s: f64 = (0..d).map(|k| u[k * d + i] * u[k * d + j]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((s - want).abs());
        }
    }
    worst
}

fn symmetrize(m: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (m[i * n + j] + m[j * n + i]);
        }
    }
    out
}

fn chart_ball_dot(a: &JointVector, b: &JointVector) -> f64 {
    a.xi.dot(&b.xi) + 2.0 * a.c.iter().zip(b.c.iter()).map(|(x, y)| x * y).sum::<f64>()
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|t| t * t).sum::<f64>().sqrt()
}

fn scaled(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|t| t * s).collect()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
