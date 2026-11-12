// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Gentlest-ascent / constrained-saddle dynamics (GAD) on the fixed-rank TT manifold: excited
//! states as **index-1 saddle points** of the Rayleigh quotient.

use crate::core::tangent::TtTangent;
use crate::core::weingarten::FloorPolicy;
use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
use crate::scalar::{C64, Scalar};
use crate::solvers::hess::{
    WeingartenCtx, deflate_radial, deflate_subspace, lowest_hessian_mode_at, radial_mode,
    reproject_tangent,
};
use crate::solvers::joint::WDampRule;
use crate::solvers::riemannian::{normalize_mut, tangent_bases};
use crate::vector::TtVector;

pub(crate) const GAD_INITIAL_ROUND_EPS: f64 = 1e-14;

pub(crate) const GAD_ALPHA_GROW: f64 = 1.5;

pub(crate) const GAD_ALPHA_SHRINK: f64 = 0.5;

pub(crate) const GAD_CLIMB_GROW: f64 = 2.0;

pub(crate) const GAD_GRAD_NOISE_C: f64 = 32.0;

#[inline]
pub(crate) fn gad_grad_floor(lambda: f64) -> f64 {
    GAD_GRAD_NOISE_C * f64::EPSILON * (1.0 + lambda.abs())
}

#[inline]
pub(crate) fn gad_near_accept(gt: f64, gnorm: f64, lambda: f64) -> bool {
    gt <= gnorm - gad_grad_floor(lambda)
}

#[inline]
pub(crate) fn gad_far_accept(gt: f64, gnorm: f64) -> bool {
    gt <= GAD_CLIMB_GROW * gnorm
}

#[cfg(test)]
mod gad_accept_tests {
    use super::{gad_far_accept, gad_grad_floor, gad_near_accept};

    #[test]
    fn gad_near_accept_floors_the_decrease_side_not_the_increase_side() {
        let lambda = -3.848431_f64;
        let gnorm = 3.187020e-3_f64;
        let gfloor = gad_grad_floor(lambda);
        assert!(
            gfloor > 0.0 && gfloor < 1e-12,
            "sanity: gfloor = {gfloor:.3e}"
        );

        let gt_up = gnorm + 0.5 * gfloor;
        assert!(
            !gad_near_accept(gt_up, gnorm, lambda),
            "an unresolvable INCREASE (gt = gnorm + gfloor/2) must be rejected by the near test \
             — accepting it is exactly the phase-1 noise-band crawl (592 consecutive floor-accepts \
             at frozen ‖grad‖ on the (4,2,2) 0xC3 fixture)"
        );
        let gt_down = gnorm - 2.0 * gfloor;
        assert!(
            gad_near_accept(gt_down, gnorm, lambda),
            "a resolvable decrease (gt = gnorm - 2*gfloor) must be accepted"
        );
        let gt_at_floor = gnorm - gfloor - 1e-30;
        assert!(
            gad_near_accept(gt_at_floor, gnorm, lambda),
            "a decrease landing just past the floor boundary must be accepted"
        );
        let gt_small_down = gnorm - 0.5 * gfloor;
        assert!(
            !gad_near_accept(gt_small_down, gnorm, lambda),
            "a decrease smaller than the floor is unresolved and must be rejected"
        );

        assert!(gad_far_accept(1.9 * gnorm, gnorm));
        assert!(!gad_far_accept(2.1 * gnorm, gnorm));
    }
}

pub(crate) const GAD_WARM_SEED_MIN: f64 = 1e-8;

pub(crate) const GAD_TRACK_RITZ: usize = 3;

pub(crate) const GAD_CERT_NEG_MARGIN: f64 = 1e-6;

pub(crate) const GAD_WITNESS_OVERLAP_TOL: f64 = 1e-3;

pub(crate) const GAD_BORDER_B_LIVE: f64 = 1e-6;

/// Which Hessian model the V-tracking eigensolver sees. The endpoint certificate is on the
/// undamped `A + B` under every arm, so this is never a correctness knob — but at TRUNCATED rank
/// it decides WHICH saddle the trajectory reaches, an effect larger than the off-critical
/// conditioning the arm list also addresses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WModel {
    /// `A + B_s`, the σ-damped Weingarten term ([`WeingartenCtx::b_apply_damped`]) with the
    /// Tikhonov scale set per accepted point by the rule. **Selectable, not the default** —
    /// indistinguishable from [`Self::Full`] (measured before the damped model was made
    /// symmetric, not re-measured), so it is the arm to reach for when the exact `B` is
    /// wanted in the tracking model but its off-critical conditioning is a worry; it inherits
    /// [`Self::Full`]'s truncated-rank saddle selection along with it.
    Damped(WDampRule),
    /// The hard switch: term-A only while `‖grad‖ > b_switch_tol`, the exact `A + B` below it.
    /// **This is the DEFAULT, at `b_switch_tol = 1e-2`** (see the enum's rustdoc).
    Switch { b_switch_tol: f64 },
    /// The exact `A + B` everywhere, undamped. Correct by construction and the worst-conditioned
    /// choice off-criticality. Measured identical to [`Self::Damped`], including on the truncated
    /// cells where both differ from the [`Self::Switch`] default (measured before the damped model
    /// was made symmetric, not re-measured) — so it is the arm that pins the `B`-carrying
    /// trajectory, and the reference the damping band is judged against.
    Full,
}

impl WModel {
    pub(crate) fn b_weight(self, gnorm: f64) -> f64 {
        match self {
            WModel::Switch { b_switch_tol } => f64::from(u8::from(gnorm <= b_switch_tol)),
            WModel::Full => 1.0,
            WModel::Damped(_) => f64::NAN,
        }
    }

    pub(crate) fn damp_scale(self, gnorm: f64) -> Option<f64> {
        match self {
            WModel::Damped(rule) => Some(rule.scale(gnorm)),
            WModel::Switch { .. } | WModel::Full => None,
        }
    }

    pub(crate) fn is_damped(self) -> bool {
        match self {
            WModel::Damped(_) => true,
            WModel::Switch { .. } | WModel::Full => false,
        }
    }

    pub(crate) fn is_blend(self) -> bool {
        false
    }

    #[allow(unused_variables)]
    pub(crate) fn is_full(self, gnorm: f64, s: f64) -> bool {
        match self {
            WModel::Damped(_) => false,
            WModel::Switch { b_switch_tol } => gnorm <= b_switch_tol,
            WModel::Full => true,
        }
    }
}

/// Which way [`RankAdaptOpts`] may move the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankAdaptMode {
    /// Grow AND shrink: the accepted step is re-retracted through `round(x + α·F, ϵ, chi)`
    /// ([`TtTangent::retract_round`]), so a bond gains rank where the moved point needs it and
    /// loses it where a Schmidt value falls under the budget — the manifold follows the
    /// trajectory both ways. Growth per step is bounded by the representing rank `2r`.
    Both,
    /// Shrink only: the fixed-rank step is kept and then `round(·, ϵ)`-ed, which can only drop
    /// rank. For a trajectory that reaches the manifold border (dead Schmidt values — the H2O
    /// crawl) without ever needing more rank than it was launched with.
    ShrinkOnly,
}

/// Rank-adaptation knobs, `GadOpts::rank_adapt = Some(..)`. **No measured defaults**:
/// `round_eps` is the truncation budget that DEFINES the adaptive manifold, and the number a
/// measurement has to set.
#[derive(Clone, Copy, Debug)]
pub struct RankAdaptOpts {
    /// Grow and shrink, or shrink only.
    pub mode: RankAdaptMode,
    /// POOLED cap on each bond's TOTAL dimension `Σ_q r[q]` in a growth round
    /// ([`crate::qn::Pooled::NONE`] = uncapped); consumed directly by `retract_round`, NOT as a
    /// per-sector cap. Ignored under [`RankAdaptMode::ShrinkOnly`]. A non-`NONE` value must be
    /// at least the launch point's own largest pooled total bond dimension (asserted at driver
    /// entry): below that the cap truncates the start on the first adapted step instead of
    /// adapting it.
    pub chi: crate::qn::Pooled,
    /// Energy budget of the adaptation round, `RoundMode::Rel` on the unit-norm state
    /// (`round_mut`'s per-bond budget `ϵ/√(d−1)`). Must be positive and finite.
    pub round_eps: f64,
    /// Adapt every `period` accepted steps (`1` = every step). Must be `≥ 1`.
    pub period: usize,
}

impl Default for RankAdaptOpts {
    fn default() -> Self {
        RankAdaptOpts {
            mode: RankAdaptMode::Both,
            chi: crate::qn::Pooled::NONE,
            round_eps: 1e-6,
            period: 1,
        }
    }
}

/// What the driver does at the manifold BORDER (bond Schmidt values `σ → 0`), where the
/// Weingarten term carries `Σ⁻¹` and the undamped certificate reads modes the state does not
/// have. Measured on water/6-31G m=32: every false "index 1" sat at `σ_min < 1e-10`
/// or at `‖B·V‖/‖A·V‖ ≥ 1.2` with `σ_min ≲ 1e-5`; every genuine molecular saddle had gain
/// `≤ 0.36`. The gain ceiling also refuses the `d = 4` SU(2) truncated cell's `B`-dominated
/// endpoint (gain 37.5 at LIVE `σ_min,cw = 8.55e-4`, 0.93 Ha below the dense `E₁`, certified
/// index 1 before this rule) — the intended reading: `B`'s dominance, not only dead directions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BorderPolicy {
    /// **The default, and the only policy.** Trajectory and merit untouched. At `gradtol`, BEFORE probe 1, the
    /// endpoint is a BORDER point when the undamped gain `‖B·V‖/`‖A·V‖ > gain_max`, or when
    /// `min_schmidt < schmidt_floor` (the tripwire's quantity: raw σ on the abelian layers,
    /// centerweighted on SU(2)) while `B` is live there (gain above `GAD_BORDER_B_LIVE`; at full
    /// rank `B ≡ 0` and a sub-floor Schmidt value is harmless); then no probe runs and the run reports
    /// [`GadInfo::border_verdict`] with the level, `index_estimate = None`, not certified, not
    /// converged. This is the "assert the level and the border verdict" rule, applied by
    /// the driver.
    Refuse { gain_max: f64 },
}

impl Default for BorderPolicy {
    fn default() -> Self {
        BorderPolicy::Refuse { gain_max: 1.0 }
    }
}

impl BorderPolicy {
    /// The endpoint gain ceiling.
    pub fn gain_max(self) -> f64 {
        match self {
            BorderPolicy::Refuse { gain_max } => gain_max,
        }
    }
}

/// Options for [`gad`] (and the `c64` / SU(2) twins).
#[derive(Clone, Copy, Debug)]
pub struct GadOpts {
    /// Stop when `‖P(Hx − λx)‖ ≤ gradtol`, at which point the certificate runs (if armed).
    pub gradtol: f64,
    /// Maximum ACCEPTED steps. Rejected trials are counted separately
    /// ([`GadInfo::rejected_steps`]) and do not consume this budget.
    pub max_steps: usize,
    /// Initial step size of the retracted Euler step.
    pub alpha0: f64,
    /// Lower clamp on the step size. A trial rejected at `alpha_min` ends the run in the FAR
    /// regime; in the near regime it first falls back to the far test at `alpha0`
    /// ([`GadInfo::relatches`]).
    pub alpha_min: f64,
    /// Upper clamp on the step size.
    pub alpha_max: f64,
    /// Rayleigh–Ritz basis expansions per V refresh (each = one Hessian-model apply). The design
    /// range is 5–10; the refresh is warm-started, so the unpreconditioned warm-up that guards a
    /// COLD probe against interior-eigenpair locking is switched off here and the whole budget
    /// buys preconditioned corrections.
    pub k_davidson: usize,
    /// Which Hessian model the V refresh sees (the certificate is undamped regardless).
    /// **Default: [`WModel::Switch`] at `b_switch_tol = 1e-2`** — see [`WModel`] for what the
    /// other two arms buy.
    pub w_model: WModel,
    /// Σ⁻¹ floor policy of the Weingarten term. `None` is NOT "no floor" — it is the context's
    /// own default, `FloorPolicy::Hard(1e-12)`; there is no unfloored arm.
    pub floor: Option<FloorPolicy>,
    /// Run the two-probe index-1 certificate on reaching `gradtol`. With it off, `converged`
    /// reports criticality only and [`GadInfo::index1_certified`] is always `false`.
    pub certify: bool,
    /// Rayleigh–Ritz budget of each certificate probe **and of the COLD V refresh** — including
    /// the very first one, when no `v0` is supplied and there is no transported mode to warm-start
    /// from. It is therefore not a certificate-only knob: too small a value under-converges `V₀`
    /// and the trajectory goes somewhere else. An `Inconclusive` verdict is never evidence of
    /// anything — raise this rather than reading the number it reports.
    pub probe_budget: usize,
    /// Manifold-border tripwire: an accepted point whose smallest bond Schmidt value falls below
    /// this flags [`GadInfo::schmidt_floor_hit`]. Only reported; rank adaptation is the eventual
    /// response.
    pub schmidt_floor: f64,
    /// Reserved. The default path draws NO random numbers — the `V₀` initializer seeds from the
    /// projected term-A diagonal, which is deterministic — so two runs with the same
    /// `(x0, v0, opts)` agree to machine precision with or without this.
    pub seed: u64,
    /// Grid-preservation opt-out of the entry cleanup round, verbatim the
    /// [`crate::solvers::riemannian::RiemannianOpts::preserve_grid`] contract: `false` opens with
    /// `round_mut(1e-14, 0, Abs)`, which collapses over-declared rank-deficient sectors and is
    /// PIN-BLIND (on a sector-pinned manifold the state then leaves the declared grid); `true`
    /// means the declared grid IS the manifold, the entry round is skipped, and the caller owns
    /// the conditioning (which shows up in [`GadInfo::min_schmidt_min`]). The retraction closes
    /// with the shrink-only `round_ranks_mut(&declared)` either way, so `ranks ≤ declared` is the
    /// guaranteed postcondition; exact preservation is MEASURED, not proven.
    pub preserve_grid: bool,
    /// `Some(..)` turns rank adaptation on (see [`RankAdaptOpts`], [`RankAdaptMode`]); `None`
    /// (the default) is the fixed declared grid with `ranks ≤ declared` as the postcondition.
    /// With adaptation on that postcondition is replaced by: ranks `≤ chi` (pooled) under
    /// `Both` ([`crate::qn::Pooled::NONE`] = no cap), never above the launch grid under
    /// `ShrinkOnly`, and a launch-zero
    /// sector is never opened (structural, see [`RankAdaptOpts`]). [`GadInfo::grid_changes`]
    /// counts the grid moves; [`GadStepRecord::pooled_rank`] logs the pooled rank per base point.
    pub rank_adapt: Option<RankAdaptOpts>,
    /// What happens at the manifold border — see [`BorderPolicy`]. Default
    /// [`BorderPolicy::Refuse`] at `gain_max = 1.0`.
    pub border: BorderPolicy,
}

impl Default for GadOpts {
    fn default() -> Self {
        GadOpts {
            gradtol: 1e-8,
            max_steps: 600,
            alpha0: 1.0,
            alpha_min: 1e-10,
            alpha_max: 10.0,
            k_davidson: 8,
            w_model: WModel::Switch { b_switch_tol: 1e-2 },
            floor: None,
            certify: true,
            probe_budget: 60,
            schmidt_floor: 1e-10,
            seed: 1234,
            preserve_grid: false,
            rank_adapt: None,
            border: BorderPolicy::default(),
        }
    }
}

/// One accepted point's record: everything the refresh and the step at that base point already
/// computed, kept per step instead of collapsed into a scalar.
#[derive(Clone, Copy, Debug)]
pub struct GadStepRecord {
    /// `‖grad‖` at this base point — `== gnorms[k]`.
    pub gnorm: f64,
    /// Rayleigh quotient at this base point. ELECTRONIC.
    pub lambda: f64,
    /// Smallest bond Schmidt value here (RAW `σ` on the abelian layers, CENTERWEIGHTED on
    /// SU(2) — the two are not the same quantity and a cross-layer column must say which).
    pub min_schmidt: f64,
    /// Largest POOLED bond rank (sum over sectors) of this base point — constant on a fixed
    /// grid, the rank trajectory under [`GadOpts::rank_adapt`] (reduced multiplet ranks on SU(2)).
    pub pooled_rank: usize,
    /// The Tikhonov scale the V-tracking model used, `Some` under the σ-damped arms only
    /// ([`WModel::Damped`]).
    pub damp_s: Option<f64>,
    /// Whether the model this refresh saw was the exact `A + B`: always under [`WModel::Full`],
    /// under [`WModel::Switch`] once `‖grad‖ ≤ b_switch_tol`, never under [`WModel::Damped`];
    /// [`WModel::is_full`] spells the rule for every arm.
    pub model_is_full: bool,
    /// The blend weight of a blended model at this refresh; `None` under every [`WModel`] arm of
    /// the default build.
    pub b_weight: Option<f64>,
    /// `‖(model − A)·V‖ / ‖A·V‖` — the gain of the `B`-part of the model ACTUALLY applied, on the
    /// unit tracked mode: equal to [`Self::b_gain`] where the model is the exact `A + B`, `0`
    /// on [`WModel::Switch`]'s term-A branch, and the damped `‖B_s·V‖/‖A·V‖` (one extra apply)
    /// under the σ-damped arms. `NaN` when `‖A·V‖` is exactly zero. This is the number a
    /// gain-capping rule is checked against.
    pub b_gain_model: f64,
    /// `true` at the very first refresh of a run started without a `v0`, where the initializer is
    /// term-A only whatever [`GadOpts::w_model`] says — the three model arms coincide there.
    pub a_only: bool,
    /// `‖B·V‖ / ‖A·V‖` at this point, on the UNDAMPED `B` and the unit tracked mode. `NaN` when
    /// `‖A·V‖` is exactly zero (the row is kept, not dropped).
    pub b_gain: f64,
    /// Lowest Ritz value of the MODEL at this refresh.
    pub theta_low: f64,
    /// Ritz value of the branch actually selected — the tracked mode's curvature. Equal to
    /// [`Self::theta_low`] away from an eigenvalue crossing.
    pub theta_chosen: f64,
    /// The track solve met its residual tolerance (or exhausted the deflated space exactly).
    pub track_converged: bool,
    /// Model applies this refresh consumed.
    pub track_applies: usize,
    /// `|⟨V_new, V_transported⟩|`, `None` at a refresh with no previous mode to compare against.
    pub v_overlap: Option<f64>,
    /// `θ_chosen − θ_lowest`, `None` under the same condition as [`Self::v_overlap`].
    pub branch_gap: Option<f64>,
    /// The merit slope `s = Re⟨g, Hess_model·F⟩` — the near/far key. See [`GadInfo::gnorms`].
    pub merit_slope: f64,
    /// The regime this base point ENTERED the trial loop in: `true` = near (strict decrease),
    /// `false` = far (permissive growth). A fallback inside the trial loop does not rewrite it;
    /// [`GadInfo::relatches`] counts those.
    pub near: bool,
    /// The step size actually accepted here, BEFORE the growth applied to the next base point.
    /// `NaN` at a base point that took no step (the endpoint, the budget break, a stationary
    /// exit).
    pub alpha: f64,
    /// Trials rejected at this base point before it was left.
    pub rejections: usize,
}

/// Diagnostics of a GAD run.
#[derive(Clone, Debug)]
pub struct GadInfo {
    /// `‖grad‖ ≤ gradtol` **and**, when [`GadOpts::certify`] is on, the index-1 certificate
    /// passed. An endpoint that is critical but fails the certificate reports `false` with an
    /// [`Self::index_estimate`] — criticality alone is not what this solver claims to find.
    pub converged: bool,
    /// Both certificate probes passed: the radially-deflated Hessian has a negative direction `w`
    /// (probe 1), and deflating `span{radial, w}` leaves a PD operator (probe 2). A statement
    /// about the POINT; whether the tracked `V` is `w` is [`Self::tracked_mode_is_lowest`].
    pub index1_certified: bool,
    /// Negative directions found by the certificate: `Some(0)` at a certified minimum, `Some(1)`
    /// at a certified index-1 saddle, `Some(2)` as a LOWER bound when probe 2 witnesses a second
    /// negative direction, `None` when a probe came back `Inconclusive` (which is never evidence)
    /// or when the certificate did not run.
    pub index_estimate: Option<usize>,
    /// `|⟨w, V⟩|` between probe 1's witness `w` (the lowest mode of the undamped Hessian at the
    /// endpoint) and the tracked mode `V`; `None` when the certificate did not run or probe 1
    /// found no negative direction. Over C64 the modulus of the complex inner product, i.e. the
    /// overlap with the whole complex line of `V`.
    pub witness_overlap: Option<f64>,
    /// `witness_overlap ≥ 1 − GAD_WITNESS_OVERLAP_TOL`: the tracker ended on the lowest branch.
    /// `false` with `index1_certified` means the point is an index-1 saddle reached with `V` on
    /// another negative-curvature branch (measured at truncated complex rank under the smooth
    /// tracking models: the partner mode inside the complex line). The dynamics
    /// converge to the same point either way; what differs is the step cost.
    pub tracked_mode_is_lowest: bool,
    /// The endpoint reached `gradtol` on a manifold BORDER — `min_schmidt < schmidt_floor` or
    /// `‖B·V‖/‖A·V‖ > gain_max` there ([`BorderPolicy`]) — so the certificate was not run: the
    /// level `λ` is reported, `index_estimate` is `None`, `converged` is `false`.
    pub border_verdict: bool,
    /// The undamped `‖B·V‖/‖A·V‖` at the LAST refresh (the endpoint's when the run reached
    /// `gradtol`); `NaN` before any refresh. The quantity the border gain test reads.
    pub b_gain_end: f64,
    /// Accepted steps whose base grid differs from the previous base point's: the adaptation
    /// rounds that moved the grid, plus the fixed-rank retraction's own rank-revealing collapse
    /// of a deficient sector (so not necessarily `0` without [`GadOpts::rank_adapt`]).
    pub grid_changes: usize,
    /// Accepted steps taken.
    pub steps: usize,
    /// Hessian-model applies consumed (V refreshes, the `b_gain` diagnostic, the near/far key's
    /// merit-slope apply and the certificate probes). Trial gradient evaluations are a different
    /// cost and are NOT counted here.
    pub applies: usize,
    /// Final `‖grad‖`.
    pub gnorm: f64,
    /// Final Rayleigh quotient. ELECTRONIC — add the integrals' core energy for a total.
    pub lambda: f64,
    /// Fresh-apply Rayleigh quotient of the certificate's probe-1 mode: the ascent eigenvalue at
    /// an index-1 endpoint (negative), or the spectral gap at a refused index-0 one (positive).
    /// `None` when the certificate did not run.
    pub eig1: Option<f64>,
    /// Overlap `|⟨V_new, V_transported⟩|` at the last refresh that had a previous mode to compare
    /// against — the tracking's own convergence diagnostic. `0.0` if no such refresh happened.
    pub v_overlap: f64,
    /// `max ‖B·V‖ / ‖A·V‖` over the refreshes, on the UNDAMPED `B` and the unit tracked mode.
    /// The conditioning diagnostic, and what the [`WModel`] arms are compared on: a run whose
    /// tracked mode has locked onto a Weingarten-dominated direction shows a gain two orders above
    /// the runs that reach the `E₁`-like saddle. Reported only — nothing in the driver
    /// reads this value.
    pub b_gain_max: f64,
    /// Smallest bond Schmidt value seen at any accepted point — the manifold-border tripwire.
    pub min_schmidt_min: f64,
    /// `true` if [`Self::min_schmidt_min`] ever fell below [`GadOpts::schmidt_floor`].
    pub schmidt_floor_hit: bool,
    /// Trial steps rejected by the merit test.
    pub rejected_steps: usize,
    /// Times the NEAR test was exhausted and the base point retried under the far test: the
    /// strict decrease test rejected every step size down to [`GadOpts::alpha_min`], so the
    /// driver reset `α ← alpha0` and re-ran the same base point permissively rather than ending
    /// the run. At most once per base point.
    pub relatches: usize,
    /// Far↔Near flips between consecutive base points — how often the criterion actually changed
    /// regime along the trajectory, as opposed to how often its fallback fired. Equal to the
    /// number of `near` flips in [`Self::trace`] by construction.
    pub regime_switches: usize,
    /// `‖grad‖` at every accepted point, `gnorms[0]` at the launch point: `gnorms.len() ==
    /// steps + 1`. The trajectory record the merit invariants are stated about.
    pub gnorms: Vec<f64>,
    /// `max |⟨V, P(x)/‖P(x)‖⟩|` over the refreshes — the radial-contamination tripwire. The
    /// radial direction is a spurious Hessian eigenvector on the projective manifold, and a `V`
    /// that leaks into it converges GAD to the GROUND state disguised as a saddle. Two-pass
    /// deflation holds this at roundoff.
    pub radial_overlap_max: f64,
    /// The Tikhonov scale used at each refresh under [`WModel::Damped`] (empty under the other
    /// arms) — the twin of the joint trust region's own per-iteration `s` log.
    pub damp_s: Vec<f64>,
    /// `θ_chosen − θ_lowest` at the last refresh that had a previous mode to compare against (the
    /// same refresh [`Self::v_overlap`] is read from) — the eigenvalue-crossing diagnostic design
    /// §3 asks branch selection to log beside the chosen overlap. `0.0` when the chosen branch
    /// coincides with the lowest Ritz pair, which is the common case away from a crossing; `0.0`
    /// also before any refresh has had a previous mode to compare against.
    pub branch_gap: f64,
    /// One [`GadStepRecord`] per base point that completed a V refresh — the per-step twin of
    /// the scalars above, in trajectory order. See [`GadStepRecord`] for the length contract.
    pub trace: Vec<GadStepRecord>,
}

macro_rules! impl_gad {
    (
        $(#[$doc:meta])*
        $name:ident, $ham:ty, $vec:ty, $tangent:ty,
        $ctx:ident, $rayleigh:path, $tbases:path, $normalize:path, $grad_env:path,
        $tangent_dim:path, $min_schmidt:path, $dim_mult:expr
    ) => {
        $(#[$doc])*
        pub fn $name(
            h: &$ham,
            x0: &$vec,
            v0: Option<&$tangent>,
            opts: &GadOpts,
        ) -> ($vec, GadInfo) {
            if let Some(ra) = opts.rank_adapt {
                assert!(
                    ra.round_eps.is_finite() && ra.round_eps > 0.0,
                    "gad: RankAdaptOpts::round_eps must be positive and finite (got {})",
                    ra.round_eps
                );
                assert!(ra.period >= 1, "gad: RankAdaptOpts::period must be ≥ 1");
                if ra.chi != crate::qn::Pooled::NONE {
                    let start_pooled =
                        <$vec as crate::solvers::layer::TtState>::max_total_bond_rank(x0);
                    assert!(
                        ra.chi.0 >= start_pooled,
                        "gad: RankAdaptOpts::chi = {} caps the POOLED total bond dimension \
                         (Σ_q r[q] at a bond), and the start point's largest pooled total bond \
                         dimension is {}; a pooled cap below the start's pooled rank truncates \
                         the start rather than adapting it — raise chi to at least {}",
                        ra.chi.0,
                        start_pooled,
                        start_pooled
                    );
                }
            }
            {
                let g = opts.border.gain_max();
                assert!(
                    g.is_finite() && g > 0.0,
                    "gad: BorderPolicy gain_max must be positive and finite (got {g})"
                );
            }
            match opts.w_model {
                WModel::Damped(_) | WModel::Switch { .. } | WModel::Full => {}
            }

            let mut x = x0.clone();
            if !opts.preserve_grid {
                x.round_mut(
                    crate::solvers::gad::GAD_INITIAL_ROUND_EPS,
                    crate::qn::Pooled::NONE,
                    crate::round::RoundMode::Abs,
                );
            }
            $normalize(&mut x);
            let launch_ranks = x.ranks.clone();
            let pooled_rank_max = |v: &$vec| -> usize {
                let sp = v.space;
                (0..=sp.d)
                    .map(|b| {
                        sp.bond_sectors(b)
                            .into_iter()
                            .map(|s| v.ranks[b][s])
                            .sum::<usize>()
                    })
                    .max()
                    .unwrap_or(0)
            };
            let mut grid_changes = 0usize;

            let mut v_cur: Option<$tangent> = v0.cloned();
            let mut v_seeded = v0.is_some();
            let mut alpha = opts.alpha0;
            let mut steps = 0usize;
            let mut rejected_steps = 0usize;
            let mut applies = 0usize;
            let mut gnorms: Vec<f64> = Vec::new();
            let mut damp_s: Vec<f64> = Vec::new();
            let mut b_gain_max = 0.0f64;
            let mut min_schmidt_min = f64::INFINITY;
            let mut schmidt_floor_hit = false;
            let mut radial_overlap_max = 0.0f64;
            let mut v_overlap = 0.0f64;
            let mut branch_gap = 0.0f64;
            let mut index1_certified = false;
            let mut index_estimate: Option<usize> = None;
            let mut eig1: Option<f64> = None;
            let mut witness_overlap: Option<f64> = None;
            let mut tracked_mode_is_lowest = false;
            let mut border_verdict = false;
            let mut b_gain_end = f64::NAN;
            let mut relatches = 0usize;
            let mut regime_switches = 0usize;
            let mut records: Vec<crate::solvers::gad::GadStepRecord> = Vec::new();
            let mut prev_near: Option<bool> = None;
            let trace = crate::knobs::GAD_TRACE.get();
            let (mut gnorm, mut lambda);

            'outer: loop {
                let (base_l, base_r) = $tbases(&x);
                let (base_l, base_r) = (std::sync::Arc::new(base_l), std::sync::Arc::new(base_r));
                lambda = $rayleigh(h, &x).re_s();
                let ctx = $ctx::from_parts(h, base_l.clone(), base_r.clone(), lambda);
                let ctx = match opts.floor {
                    Some(f) => ctx.with_floor(f),
                    None => ctx,
                };
                let grad = ctx.gradient().clone();
                gnorm = grad.norm();
                gnorms.push(gnorm);

                let ms = $min_schmidt(&base_l, &base_r);
                min_schmidt_min = min_schmidt_min.min(ms);
                if ms < opts.schmidt_floor {
                    schmidt_floor_hit = true;
                }

                let radial = radial_mode(&ctx);
                let diag = ctx.a_diagonal();
                let scale = crate::solvers::hess::diag_inf_scale(&diag);
                let space_dim = $dim_mult * $tangent_dim(&base_l)
                    - $dim_mult * radial.as_ref().map_or(0, |_| 1);

                let s_damp_opt = opts.w_model.damp_scale(gnorm);
                let s_damp = s_damp_opt.unwrap_or(0.0);
                let model_full = opts.w_model.is_full(gnorm, s_damp);
                let b_weight = opts.w_model.b_weight(gnorm);
                let model = |q: &$tangent| {
                    if model_full {
                        return ctx.hess_apply(q);
                    }
                    if opts.w_model.is_damped() {
                        ctx.a_apply(q).add(&ctx.b_apply_damped(q, s_damp))
                    } else if opts.w_model.is_blend() {
                        if b_weight <= 0.0 {
                            ctx.a_apply(q)
                        } else {
                            let mut b = ctx.b_apply(q);
                            b.scale_mut(
                                <$vec as crate::solvers::hess::VecScalar>::S::from_f64(b_weight),
                            );
                            ctx.a_apply(q).add(&b)
                        }
                    } else {
                        debug_assert!(matches!(opts.w_model, WModel::Switch { .. }));
                        ctx.a_apply(q)
                    }
                };

                let warm: Option<$tangent> = v_cur.as_ref().and_then(|vold| {
                    let mut w = vold.clone();
                    reproject_tangent(&base_l, &base_r, radial.as_ref(), &mut w);
                    let n = w.norm();
                    (n >= crate::solvers::gad::GAD_WARM_SEED_MIN).then(|| {
                        w.scale_mut(
                            <$vec as crate::solvers::hess::VecScalar>::S::from_f64(1.0 / n),
                        );
                        w
                    })
                });
                let det_seed = <$tangent>::project_shared(&base_l, &base_r, &diag.to_vector());
                let a_only = !v_seeded;
                let vapply = |q: &$tangent| {
                    if a_only {
                        ctx.a_apply(q)
                    } else {
                        model(q)
                    }
                };
                let track_opts = crate::solvers::hess::ModeOpts {
                    budget: opts.k_davidson.max(2),
                    n_ritz: crate::solvers::gad::GAD_TRACK_RITZ,
                    warmup: 0,
                    neg_early_exit: false,
                    space_dim,
                    scale,
                    ..Default::default()
                };
                let cold_opts = crate::solvers::hess::ModeOpts {
                    n_ritz: 1,
                    neg_early_exit: false,
                    budget: opts.probe_budget.max(2),
                    space_dim,
                    scale,
                    ..Default::default()
                };
                let res = match warm.as_ref() {
                    Some(w) => lowest_hessian_mode_at(
                        &base_l,
                        &base_r,
                        radial.as_ref(),
                        &diag,
                        &vapply,
                        &[],
                        w,
                        &track_opts,
                    ),
                    None => None,
                }
                .or_else(|| {
                    lowest_hessian_mode_at(
                        &base_l,
                        &base_r,
                        radial.as_ref(),
                        &diag,
                        &vapply,
                        &[],
                        &det_seed,
                        &cold_opts,
                    )
                });
                let Some(res) = res else {
                    break 'outer;
                };
                applies += res.applies;
                v_seeded = true;

                let select_best = |only_converged: bool| -> Option<($tangent, f64, usize)> {
                    let mut best: Option<($tangent, f64, usize)> = None;
                    for (idx, p) in res.ritz.iter().enumerate() {
                        if only_converged
                            && p.residual
                                > track_opts.tol_rel * track_opts.scale.max(p.theta.abs())
                        {
                            continue;
                        }
                        let mut u = p.vector.clone();
                        let n = u.norm();
                        if n == 0.0 {
                            continue;
                        }
                        u.scale_mut(
                            <$vec as crate::solvers::hess::VecScalar>::S::from_f64(1.0 / n),
                        );
                        let ov = match warm.as_ref() {
                            Some(w) => {
                                let c = <$tangent>::dot(&u, w);
                                let m = c.abs_s();
                                if m > 0.0 {
                                    u.scale_mut(
                                        c * <$vec as crate::solvers::hess::VecScalar>::S::from_f64(
                                            1.0 / m,
                                        ),
                                    );
                                }
                                m
                            }
                            None => 1.0,
                        };
                        let better = match best.as_ref() {
                            None => true,
                            Some((_, b, _)) => ov > *b,
                        };
                        if better {
                            best = Some((u, ov, idx));
                        }
                    }
                    best
                };
                let best = select_best(true).or_else(|| select_best(false));
                let Some((v, ov, chosen_idx)) = best else {
                    break 'outer;
                };
                if warm.is_some() {
                    v_overlap = ov;
                    branch_gap = res.ritz[chosen_idx].theta - res.ritz[0].theta;
                }
                if let Some(r) = radial.as_ref() {
                    radial_overlap_max = radial_overlap_max.max(<$tangent>::dot(r, &v).abs_s());
                }

                let an = ctx.a_apply(&v).norm();
                let bn = ctx.b_apply(&v).norm();
                applies += 2;
                let b_gain = if an > 0.0 { bn / an } else { f64::NAN };
                if an > 0.0 {
                    b_gain_max = b_gain_max.max(b_gain);
                }
                b_gain_end = b_gain;
                let b_gain_model = if model_full {
                    b_gain
                } else {
                    if opts.w_model.is_damped() {
                        let bsn = ctx.b_apply_damped(&v, s_damp).norm();
                        applies += 1;
                        if an > 0.0 { bsn / an } else { f64::NAN }
                    } else if opts.w_model.is_blend() {
                        b_weight * b_gain
                    } else {
                        0.0
                    }
                };
                if let Some(s) = s_damp_opt {
                    damp_s.push(s);
                }

                let c = <$tangent>::dot(&v, &grad);
                let mut f = grad.scale(
                    <$vec as crate::solvers::hess::VecScalar>::S::from_f64(-1.0),
                );
                f.axpy(
                    c * <$vec as crate::solvers::hess::VecScalar>::S::from_f64(2.0),
                    &v,
                );
                deflate_radial(radial.as_ref(), &mut f);

                let s_merit = <$tangent>::dot(&grad, &model(&f)).re_s();
                applies += 1;
                let near_state = s_merit < 0.0;
                if prev_near.is_some_and(|p| p != near_state) {
                    regime_switches += 1;
                }
                prev_near = Some(near_state);

                records.push(crate::solvers::gad::GadStepRecord {
                    gnorm,
                    lambda,
                    min_schmidt: ms,
                    pooled_rank: pooled_rank_max(&x),
                    damp_s: s_damp_opt,
                    model_is_full: model_full,
                    b_weight: opts.w_model.is_blend().then_some(b_weight),
                    b_gain_model,
                    a_only,
                    b_gain,
                    theta_low: res.ritz[0].theta,
                    theta_chosen: res.ritz[chosen_idx].theta,
                    track_converged: res.converged,
                    track_applies: res.applies,
                    v_overlap: warm.is_some().then_some(ov),
                    branch_gap: warm
                        .is_some()
                        .then(|| res.ritz[chosen_idx].theta - res.ritz[0].theta),
                    merit_slope: s_merit,
                    near: near_state,
                    alpha: f64::NAN,
                    rejections: 0,
                });

                if gnorm <= opts.gradtol {
                    border_verdict = b_gain > opts.border.gain_max()
                        || (ms < opts.schmidt_floor
                            && b_gain > crate::solvers::gad::GAD_BORDER_B_LIVE);
                    if opts.certify && !border_verdict {
                        let p1 = crate::solvers::hess::ModeOpts {
                            budget: opts.probe_budget.max(2),
                            n_ritz: 1,
                            warmup: 0,
                            neg_early_exit: false,
                            space_dim,
                            scale,
                            ..Default::default()
                        };
                        if let Some(r1) = lowest_hessian_mode_at(
                            &base_l,
                            &base_r,
                            radial.as_ref(),
                            &diag,
                            |q| ctx.hess_apply(q),
                            &[],
                            &v,
                            &p1,
                        ) {
                            applies += r1.applies;
                            eig1 = Some(r1.lambda_fresh);
                            let mut w = r1.witness.clone();
                            let wn = w.norm();
                            let wov = if wn > 0.0 {
                                w.scale_mut(
                                    <$vec as crate::solvers::hess::VecScalar>::S::from_f64(
                                        1.0 / wn,
                                    ),
                                );
                                <$tangent>::dot(&w, &v).abs_s()
                            } else {
                                0.0
                            };
                            if r1.lambda_fresh >= -crate::solvers::gad::GAD_CERT_NEG_MARGIN * scale {
                                index_estimate = match r1.verdict {
                                    crate::solvers::hess::PdVerdict::LikelyPd { .. } => Some(0),
                                    _ => None,
                                };
                            } else {
                                witness_overlap = Some(wov);
                                tracked_mode_is_lowest =
                                    wov >= 1.0 - crate::solvers::gad::GAD_WITNESS_OVERLAP_TOL;
                                let extra = [w.clone()];
                                let p2 = crate::solvers::hess::ModeOpts {
                                    budget: opts.probe_budget.max(2),
                                    n_ritz: 1,
                                    space_dim: space_dim.saturating_sub($dim_mult).max(1),
                                    scale,
                                    ..Default::default()
                                };
                                let mut seed2 = det_seed.clone();
                                deflate_radial(radial.as_ref(), &mut seed2);
                                deflate_subspace(&extra, &mut seed2);
                                match lowest_hessian_mode_at(
                                    &base_l,
                                    &base_r,
                                    radial.as_ref(),
                                    &diag,
                                    |q| ctx.hess_apply(q),
                                    &extra,
                                    &seed2,
                                    &p2,
                                ) {
                                    Some(r2) => {
                                        applies += r2.applies;
                                        match r2.verdict {
                                            crate::solvers::hess::PdVerdict::LikelyPd { .. } => {
                                                index1_certified = true;
                                                index_estimate = Some(1);
                                            }
                                            crate::solvers::hess::PdVerdict::WitnessNegative {
                                                ..
                                            } => {
                                                index_estimate = Some(2);
                                            }
                                            crate::solvers::hess::PdVerdict::Inconclusive {
                                                ..
                                            } => {
                                                index_estimate = None;
                                            }
                                        }
                                    }
                                    None => index_estimate = None,
                                }
                            }
                        }
                    }
                    break 'outer;
                }
                if steps >= opts.max_steps {
                    break 'outer;
                }

                v_cur = Some(v);

                let mut near = near_state;
                let mut fell_back = false;
                let mut accepted = false;
                let mut rejections = 0usize;
                loop {
                    let mut xt = f.retract_hosvd(
                        <$vec as crate::solvers::hess::VecScalar>::S::from_f64(alpha),
                    );
                    $normalize(&mut xt);
                    let (bl_t, br_t) = $tbases(&xt);
                    let lam_t = $rayleigh(h, &xt).re_s();
                    let gt = $grad_env(h, &bl_t, &br_t, lam_t).norm();
                    let ok = if near {
                        crate::solvers::gad::gad_near_accept(gt, gnorm, lambda)
                    } else {
                        crate::solvers::gad::gad_far_accept(gt, gnorm)
                    };
                    if trace {
                        println!(
                            "  gad[{steps}]: alpha={alpha:.3e} |g|={gnorm:.6e} -> {gt:.6e} \
                             ok={ok} near={near} s={s_merit:+.3e} vov={v_overlap:.9} \
                             lam={lambda:.9}"
                        );
                    }
                    if ok {
                        if let Some(r) = records.last_mut() {
                            r.alpha = alpha;
                        }
                        let mut xn = xt;
                        if let Some(ra) = opts.rank_adapt
                            && (steps + 1) % ra.period.max(1) == 0
                        {
                            match ra.mode {
                                crate::solvers::gad::RankAdaptMode::Both => {
                                    xn = f.retract_round(
                                        <$vec as crate::solvers::hess::VecScalar>::S::from_f64(
                                            alpha,
                                        ),
                                        ra.round_eps,
                                        ra.chi,
                                        crate::round::RoundMode::Rel,
                                    );
                                }
                                crate::solvers::gad::RankAdaptMode::ShrinkOnly => {
                                    xn.round_mut(ra.round_eps, crate::qn::Pooled::NONE, crate::round::RoundMode::Rel);
                                }
                            }
                            $normalize(&mut xn);
                            debug_assert!(
                                (0..=xn.space.d).all(|b| xn
                                    .space
                                    .bond_sectors(b)
                                    .into_iter()
                                    .all(|s| launch_ranks[b][s] > 0 || xn.ranks[b][s] == 0)),
                                "rank adaptation populated a launch-zero sector"
                            );
                        }
                        if xn.ranks != x.ranks {
                            grid_changes += 1;
                        }
                        x = xn;
                        alpha = (alpha * crate::solvers::gad::GAD_ALPHA_GROW).min(opts.alpha_max);
                        steps += 1;
                        accepted = true;
                        break;
                    }
                    rejected_steps += 1;
                    rejections += 1;
                    if alpha <= opts.alpha_min {
                        if near && !fell_back {
                            near = false;
                            fell_back = true;
                            relatches += 1;
                            alpha = opts.alpha0.min(opts.alpha_max).max(opts.alpha_min);
                            continue;
                        }
                        break;
                    }
                    alpha = (alpha * crate::solvers::gad::GAD_ALPHA_SHRINK).max(opts.alpha_min);
                }
                if let Some(r) = records.last_mut() {
                    r.rejections = rejections;
                }
                if !accepted {
                    break 'outer;
                }
            }

            let converged = gnorm <= opts.gradtol && (!opts.certify || index1_certified);
            if trace {
                println!(
                    "  gad: steps={steps} rejected={rejected_steps} relatches={relatches} \
                     regime_switches={regime_switches} applies={applies} |g|={gnorm:.6e} \
                     lam={lambda:.9} converged={converged} index={index_estimate:?}"
                );
            }
            (
                x,
                GadInfo {
                    converged,
                    index1_certified,
                    index_estimate,
                    witness_overlap,
                    tracked_mode_is_lowest,
                    border_verdict,
                    b_gain_end,
                    grid_changes,
                    steps,
                    applies,
                    gnorm,
                    lambda,
                    eig1,
                    v_overlap,
                    b_gain_max,
                    min_schmidt_min,
                    schmidt_floor_hit,
                    rejected_steps,
                    relatches,
                    gnorms,
                    radial_overlap_max,
                    damp_s,
                    branch_gap,
                    trace: records,
                    regime_switches,
                },
            )
        }
    };
}

pub(crate) use impl_gad;

impl_gad!(
    gad,
    SparseHamiltonian<f64>,
    TtVector<f64>,
    TtTangent<f64>,
    WeingartenCtx,
    rayleigh_quotient,
    tangent_bases,
    normalize_mut,
    crate::solvers::hess::riemannian_gradient_env,
    crate::core::tangent::tangent_dim,
    crate::core::tangent::min_schmidt,
    1
);

impl_gad!(
    gad_c64,
    SparseHamiltonian<C64>,
    TtVector<C64>,
    TtTangent<C64>,
    WeingartenCtx,
    rayleigh_quotient,
    tangent_bases,
    normalize_mut,
    crate::solvers::hess::riemannian_gradient_env,
    crate::core::tangent::tangent_dim,
    crate::core::tangent::min_schmidt,
    2
);
