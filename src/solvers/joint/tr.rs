// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::optimize::energy_floor;
use crate::solvers::InnerProductVector;
use crate::solvers::steihaug::{SteihaugStatus, steihaug_cg_ball, steihaug_cg_with};

/// The elliptical-ball slot of [`tr_krylov_step`] (§7): `Some` measures the trust BALL in the
/// supplied inner product while the MODEL stays in the type's own `dot`; `None` is the G-ball.
pub type BallDot<'a, V> = Option<&'a dyn Fn(&V, &V) -> f64>;

/// Knobs of one [`tr_krylov_step`] call. The initial radius is a separate argument (it is
/// per-iteration state owned by the driver, not a constant).
#[derive(Clone, Debug)]
pub struct TrOpts {
    /// Growth cap: `Δ ← min(2Δ, tr_max)` on a very successful boundary-limited step (§4).
    pub tr_max: f64,
    /// Maximum ×¼ re-solves per call; ≤ `max_rej + 1` trials (§4). Bounds one CALL, never the
    /// method's reach down the radius axis — the caller retries at the returned radius (§6).
    pub max_rej: usize,
    /// Steihaug iteration cap (operator applications per solve).
    pub cg_max_iter: usize,
    /// Steihaug relative tolerance; `None` ⇒ the forcing sequence `min(0.5, √‖ĝ‖_G)` (loose far
    /// from criticality where the model is untrusted anyway, tightening as the gradient falls).
    pub cg_rtol: Option<f64>,
    /// Print one line per trial.
    pub verbose: bool,
}

/// One trial evaluation, produced by the driver's closure: the retracted candidate state (opaque
/// to this module) plus the two numbers the acceptance logic reads.
pub struct TrTrial<P> {
    /// Whatever the driver needs back on acceptance (rotation, Hamiltonian, retracted TT, ...).
    pub payload: P,
    /// Raw energy at the jointly retracted trial point (§2). Non-finite ⇒ ordinary rejection.
    pub e_try: f64,
    /// The orbital chart displacement `‖δK‖_F` of this trial (§7).
    pub delta_k_norm: f64,
}

/// Diagnostics of one [`tr_krylov_step`] call. `rhos` carries one entry per trial in evaluation
/// order: finite ρ, `NaN` for a ρ-floor event, `−∞` for a non-finite trial energy (§4).
#[derive(Clone, Debug, Default)]
pub struct TrStepInfo {
    /// G-norm of the last solved step.
    pub snorm: f64,
    /// `2·model_red` of the last solve (§1).
    pub pred: f64,
    /// Per-trial ratios, evaluation order.
    pub rhos: Vec<f64>,
    /// Steihaug operator applications per solve.
    pub cg_iters: Vec<usize>,
    /// Steihaug exit status per solve.
    pub cg_statuses: Vec<SteihaugStatus>,
    /// Number of ×¼ shrinks taken in this call.
    pub rejections: usize,
    /// The kernel's returned-vector flag for the last solve (§4).
    pub boundary_hit: bool,
}

/// One coupled trust-region step: solve, trial, accept-or-shrink ladder (§4), retry contract on
/// exhaustion (§6).
#[allow(clippy::too_many_arguments)]
pub fn tr_krylov_step<V, P>(
    aop: impl Fn(&V) -> V,
    g_raised: &V,
    e0: f64,
    delta: f64,
    opts: &TrOpts,
    msolve: Option<&dyn Fn(&V) -> V>,
    ball_dot: BallDot<'_, V>,
    mut trial: impl FnMut(&V) -> TrTrial<P>,
) -> (Option<TrTrial<P>>, f64, TrStepInfo)
where
    V: InnerProductVector<Scalar = f64>,
{
    assert!(
        msolve.is_none() || ball_dot.is_none(),
        "tr_krylov_step: msolve and ball_dot cannot be combined — the ball kernel has no \
         preconditioner slot (steihaug §7: a combined variant needs a construction proven \
         consistent in BOTH pairings)"
    );
    let mut info = TrStepInfo::default();
    let gnorm = g_raised.norm();
    let rtol = opts.cg_rtol.unwrap_or_else(|| gnorm.sqrt().min(0.5));
    let efloor = energy_floor(e0);

    let mut delta_cur = delta;
    for _attempt in 0..=opts.max_rej {
        let (s, ci) = match ball_dot {
            Some(bd) => {
                steihaug_cg_ball(&aop, g_raised, delta_cur, rtol, 0.0, opts.cg_max_iter, bd)
            }
            None => steihaug_cg_with(
                &aop,
                g_raised,
                delta_cur,
                rtol,
                0.0,
                opts.cg_max_iter,
                msolve,
            ),
        };
        info.cg_iters.push(ci.iters);
        info.cg_statuses.push(ci.status);
        let snorm = s.norm();
        info.snorm = snorm;
        info.pred = 2.0 * ci.model_red;
        info.boundary_hit = ci.boundary_hit;
        if snorm == 0.0 {
            return (None, delta_cur, info);
        }

        let t = trial(&s);
        let pred = info.pred;

        if !t.e_try.is_finite() {
            info.rhos.push(f64::NEG_INFINITY);
            info.rejections += 1;
            if opts.verbose {
                println!("    tr: Δ={delta_cur:9.3e}  ‖s‖={snorm:9.3e}  E_try=non-finite  REJECT");
            }
            delta_cur /= 4.0;
            continue;
        }

        if pred < efloor {
            info.rhos.push(f64::NAN);
            if opts.verbose {
                println!(
                    "    tr: Δ={delta_cur:9.3e}  ‖s‖={snorm:9.3e}  pred={pred:9.3e} < floor \
                     {efloor:9.3e}  {}",
                    if t.e_try <= e0 {
                        "ACCEPT (not worse)"
                    } else {
                        "REFUSE"
                    }
                );
            }
            if t.e_try <= e0 {
                return (Some(t), delta_cur, info);
            }
            return (None, delta_cur, info);
        }

        let rho = (e0 - t.e_try) / pred;
        info.rhos.push(rho);
        let relaxed = (e0 - t.e_try) >= 0.1 * pred - 2.0 * efloor;
        if opts.verbose {
            println!(
                "    tr: Δ={delta_cur:9.3e}  ‖s‖={snorm:9.3e}  pred={pred:9.3e}  ρ={rho:8.4}  \
                 cg={} {:?}{}",
                ci.iters,
                ci.status,
                if rho >= 0.1 { "  ACCEPT" } else { "  REJECT" }
            );
        }
        if rho >= 0.1 || relaxed {
            if rho > 0.75 && ci.boundary_hit {
                delta_cur = (2.0 * delta_cur).min(opts.tr_max);
            }
            return (Some(t), delta_cur, info);
        }
        info.rejections += 1;
        delta_cur /= 4.0;
    }
    (None, delta_cur, info)
}

#[cfg(test)]
mod tests {
    use super::{TrOpts, TrTrial, tr_krylov_step};
    use crate::solvers::steihaug::testvec::{DVec, mat_apply, model_value};

    fn opts(max_rej: usize) -> TrOpts {
        TrOpts {
            tr_max: 1e6,
            max_rej,
            cg_max_iter: 100,
            cg_rtol: Some(1e-12),
            verbose: false,
        }
    }

    fn setup() -> (Vec<f64>, DVec, f64) {
        (vec![2.0, 0.3, 0.3, 1.0], DVec(vec![1.0, -0.5]), 10.0)
    }

    #[test]
    fn tr1_exact_model_rho_is_one_and_grow_gate() {
        let (a, g, e0) = setup();
        let delta = 1e-3;
        let (acc, dnew, info) = tr_krylov_step(
            |v| mat_apply(&a, 2, v),
            &g,
            e0,
            delta,
            &opts(3),
            None,
            None,
            |s| TrTrial {
                payload: (),
                e_try: e0 + 2.0 * model_value(&a, 2, &g, s),
                delta_k_norm: 0.0,
            },
        );
        assert!(acc.is_some());
        assert_eq!(info.rhos.len(), 1);
        assert!((info.rhos[0] - 1.0).abs() <= 1e-9, "ρ = {}", info.rhos[0]);
        assert!(info.boundary_hit);
        assert_eq!(
            dnew,
            2.0 * delta,
            "very successful boundary step must double the radius"
        );
        assert_eq!(info.rejections, 0);

        let delta = 1e3;
        let (acc, dnew, info) = tr_krylov_step(
            |v| mat_apply(&a, 2, v),
            &g,
            e0,
            delta,
            &opts(3),
            None,
            None,
            |s| TrTrial {
                payload: (),
                e_try: e0 + 2.0 * model_value(&a, 2, &g, s),
                delta_k_norm: 0.0,
            },
        );
        assert!(acc.is_some());
        assert!(!info.boundary_hit);
        assert_eq!(dnew, delta, "interior step must not grow the radius");
    }

    #[test]
    fn tr2_exhausted_cascade_returns_shrunken_radius() {
        let (a, g, e0) = setup();
        let delta = 1.0;
        let (acc, dnew, info) = tr_krylov_step(
            |v| mat_apply(&a, 2, v),
            &g,
            e0,
            delta,
            &opts(2),
            None,
            None,
            |_| TrTrial {
                payload: (),
                e_try: e0 + 1.0,
                delta_k_norm: 0.0,
            },
        );
        assert!(acc.is_none());
        assert_eq!(info.rhos.len(), 3, "max_rej = 2 ⇒ 3 trials");
        assert!(info.rhos.iter().all(|r| *r < 0.1));
        assert_eq!(info.rejections, 3);
        assert_eq!(dnew, delta / 64.0, "three ×¼ shrinks");
    }

    #[test]
    fn tr3_rho_floor_tie_accepts_delta_unchanged() {
        let (a, _, e0) = setup();
        let g = DVec(vec![1e-9, -0.5e-9]);
        let delta = 1.0;
        let (acc, dnew, info) = tr_krylov_step(
            |v| mat_apply(&a, 2, v),
            &g,
            e0,
            delta,
            &opts(3),
            None,
            None,
            |_| TrTrial {
                payload: (),
                e_try: e0,
                delta_k_norm: 0.0,
            },
        );
        assert!(acc.is_some());
        assert_eq!(dnew, delta, "ρ-floor accept must not change the radius");
        assert_eq!(info.rhos.len(), 1);
        assert!(info.rhos[0].is_nan(), "ρ-floor event is recorded as NaN");
        let (acc, dnew, info) = tr_krylov_step(
            |v| mat_apply(&a, 2, v),
            &g,
            e0,
            delta,
            &opts(3),
            None,
            None,
            |_| TrTrial {
                payload: (),
                e_try: e0 + 1e-6,
                delta_k_norm: 0.0,
            },
        );
        assert!(acc.is_none());
        assert_eq!(dnew, delta, "ρ-floor refusal must not change the radius");
        assert!(info.rhos[0].is_nan());
    }

    #[test]
    fn tr4_zero_gradient_zero_step_no_trial() {
        let (a, _, e0) = setup();
        let g = DVec(vec![0.0, 0.0]);
        let delta = 0.5;
        let mut trials = 0usize;
        let (acc, dnew, info) = tr_krylov_step(
            |v| mat_apply(&a, 2, v),
            &g,
            e0,
            delta,
            &opts(3),
            None,
            None,
            |_| {
                trials += 1;
                TrTrial {
                    payload: (),
                    e_try: e0,
                    delta_k_norm: 0.0,
                }
            },
        );
        assert!(acc.is_none());
        assert_eq!(trials, 0, "no trial may be paid for a zero step");
        assert_eq!(dnew, delta);
        assert!(info.rhos.is_empty());
    }

    #[test]
    fn tr5_nonfinite_trial_is_ordinary_rejection() {
        let (a, g, e0) = setup();
        let delta = 1e3;
        let mut call = 0usize;
        let (acc, dnew, info) = tr_krylov_step(
            |v| mat_apply(&a, 2, v),
            &g,
            e0,
            delta,
            &opts(3),
            None,
            None,
            |s| {
                call += 1;
                if call == 1 {
                    TrTrial {
                        payload: (),
                        e_try: f64::NAN,
                        delta_k_norm: 0.0,
                    }
                } else {
                    TrTrial {
                        payload: (),
                        e_try: e0 + 2.0 * model_value(&a, 2, &g, s),
                        delta_k_norm: 0.0,
                    }
                }
            },
        );
        assert!(acc.is_some());
        assert_eq!(info.rhos.len(), 2);
        assert_eq!(info.rhos[0], f64::NEG_INFINITY);
        assert!((info.rhos[1] - 1.0).abs() <= 1e-9);
        assert_eq!(info.rejections, 1);
        assert_eq!(
            dnew,
            delta / 4.0,
            "one shrink, interior second solve does not grow"
        );
    }
}

#[cfg(test)]
mod tjt_tests {
    use crate::solvers::joint::{W_DAMP_S_MAX, W_DAMP_S_MIN, WDampRule};

    #[test]
    fn tjt_t16b_w_damp_rule_scale() {
        assert_eq!(W_DAMP_S_MIN, 1e-7);
        assert_eq!(W_DAMP_S_MAX, 1e-4);
        assert_eq!(WDampRule::Fixed(3e-5).scale(1.0), 3e-5);
        assert_eq!(WDampRule::Fixed(3e-5).scale(f64::NAN), 3e-5);
        assert_eq!(WDampRule::GradTracked.scale(1e-5), 1e-5);
        assert_eq!(
            WDampRule::GradTracked.scale(1e-2),
            W_DAMP_S_MAX,
            "above the band the rule clamps to the ceiling"
        );
        assert_eq!(
            WDampRule::GradTracked.scale(1e-12),
            W_DAMP_S_MIN,
            "below the band the rule clamps to the floor"
        );
        assert_eq!(
            WDampRule::GradTracked.scale(f64::NAN),
            W_DAMP_S_MIN,
            "a non-finite ‖gk‖ must not propagate into the gauge inverse"
        );
        assert_eq!(WDampRule::GradTracked.scale(f64::INFINITY), W_DAMP_S_MIN);
    }
}
