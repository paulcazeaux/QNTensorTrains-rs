// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::gad::{GadOpts, WModel, gad};
use qntt::solvers::joint::WDampRule;
use qntt::{QnSpace, rayleigh_quotient};

use crate::util::{f64_full_rank_arm, f64_truncated_arm, generic_kick_from, longest_frozen_streak};

const FROZEN_REL: f64 = 1e-6;

fn kick_opts(max_steps: usize) -> GadOpts {
    GadOpts {
        gradtol: 1e-8,
        max_steps,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    }
}

#[test]
fn gad_c3_generic_kick_does_not_crawl_in_the_noise_band() {
    let arm = f64_full_rank_arm();
    let x0 = generic_kick_from(&arm.xgs, 5e-2, 0xC3);
    let opts = kick_opts(30);
    let (xe, info) = gad(&arm.ham, &x0, None, &opts);
    xe.check();

    assert_eq!(
        info.gnorms.len(),
        info.steps + 1,
        "gnorms must record every accepted point including x0"
    );
    let streak = longest_frozen_streak(&info.gnorms, FROZEN_REL);
    assert!(
        streak <= 6,
        "the merit was frozen (relative change below {FROZEN_REL:.0e}) over {streak} consecutive \
         accepted steps out of {} — the phase-1 noise-band crawl, MEASURED at 23 of the first 30 \
         accepted steps on this exact fixture and seed. ‖grad‖ = {:.6e}, rejections {}, \
         relatches {}",
        info.steps,
        info.gnorm,
        info.rejected_steps,
        info.relatches
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "radial leak {:.3e} on the 0xC3 trajectory",
        info.radial_overlap_max
    );
}

#[test]
fn gad_c3_generic_kick_converges_to_the_first_excited_state() {
    let arm = f64_full_rank_arm();
    let x0 = generic_kick_from(&arm.xgs, 5e-2, 0xC3);
    let opts = kick_opts(600);
    let (xe, info) = gad(&arm.ham, &x0, None, &opts);
    xe.check();

    let streak = longest_frozen_streak(&info.gnorms, FROZEN_REL);
    assert!(
        streak <= 20,
        "merit frozen over {streak} consecutive accepted steps (measured 592 of 600 under the \
         phase-1 controller): ‖grad‖ = {:.6e} after {} steps / {} rejections",
        info.gnorm,
        info.steps,
        info.rejected_steps
    );
    assert!(
        info.converged,
        "0xC3 did not converge: ‖grad‖ = {:.3e} after {} accepted steps ({} rejections, \
         {} relatches, longest frozen streak {streak})",
        info.gnorm, info.steps, info.rejected_steps, info.relatches
    );
    assert_eq!(info.index_estimate, Some(1));
    assert!(info.index1_certified);
    assert!(
        info.steps <= 400,
        "converged in {} accepted steps; the prototypes measured 164 and 189 on this seed, so a \
         count this high is a step-controller regression, not noise",
        info.steps
    );

    let lam = rayleigh_quotient(&arm.ham, &xe);
    assert!(
        (lam - arm.evals[1]).abs() < 1e-7,
        "0xC3 endpoint energy {lam} vs dense E₁ {}",
        arm.evals[1]
    );
    assert!(
        (lam - arm.evals[0]).abs() > 1e-3,
        "0xC3 endpoint stayed in the GROUND-state basin (λ = {lam}, E₀ = {}) — the phase-1 crawl \
         signature",
        arm.evals[0]
    );
    assert_eq!(
        info.gnorms.len(),
        info.steps + 1,
        "gnorms must record every accepted point including x0"
    );
    assert!(info.radial_overlap_max <= 1e-8);
}

#[test]
fn gad_c3_softest_kick_trajectory_is_not_perturbed() {
    let arm = f64_full_rank_arm();
    let opts = kick_opts(600);
    let (xe, info) = gad(&arm.ham, &arm.x0, None, &opts);
    xe.check();
    assert!(info.converged, "softest-mode control did not converge");
    assert_eq!(info.index_estimate, Some(1));
    assert!(
        info.steps <= 250,
        "the softest-mode control took {} accepted steps (phase 1: 156) — the criterion change \
         cost the canonical trajectory, which no measurement supports",
        info.steps
    );
    let streak = longest_frozen_streak(&info.gnorms, FROZEN_REL);
    assert!(
        streak <= 20,
        "the softest-mode control itself crawled ({streak} frozen accepted steps)"
    );
}

#[test]
fn gad_trace_is_self_consistent_with_the_phase1_scalars() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 40,
        k_davidson: 8,
        w_model: WModel::Damped(WDampRule::GradTracked),
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&ham, &x0, None, &opts);
    xe.check();
    assert!(info.steps > 0, "vacuous telemetry arm");

    let n = info.trace.len();
    assert!(
        n + 1 >= info.gnorms.len() && n <= info.gnorms.len(),
        "trace has {n} records against {} accepted points",
        info.gnorms.len()
    );
    assert!(n > 0, "no step record was produced at all");

    let mut b_gain_max = 0.0f64;
    for (k, r) in info.trace.iter().enumerate() {
        assert!(
            (r.gnorm - info.gnorms[k]).abs() <= 1e-13 * info.gnorms[k].abs().max(1.0),
            "trace[{k}].gnorm {} disagrees with gnorms[{k}] {}",
            r.gnorm,
            info.gnorms[k]
        );
        assert!(r.lambda.is_finite(), "trace[{k}].lambda is not finite");
        assert!(
            r.min_schmidt.is_finite() && r.min_schmidt > 0.0,
            "trace[{k}].min_schmidt = {}",
            r.min_schmidt
        );
        assert!(
            r.min_schmidt >= info.min_schmidt_min,
            "trace[{k}].min_schmidt {} is below the reported trajectory minimum {}",
            r.min_schmidt,
            info.min_schmidt_min
        );
        assert!(
            r.theta_low <= r.theta_chosen + 1e-10,
            "trace[{k}]: the chosen Ritz value {} sits BELOW the lowest one {}",
            r.theta_chosen,
            r.theta_low
        );
        assert!(
            r.merit_slope.is_finite(),
            "trace[{k}].merit_slope is not finite — the near/far key must be a number, not a NaN"
        );
        if r.b_gain.is_finite() {
            b_gain_max = b_gain_max.max(r.b_gain);
        }
        let s = r.damp_s.expect(
            "this arm asks for WModel::Damped explicitly, so every record must carry its \
             Tikhonov scale",
        );
        assert!(
            (s - info.damp_s[k]).abs() <= 1e-15,
            "trace[{k}].damp_s {s} disagrees with damp_s[{k}] {}",
            info.damp_s[k]
        );
    }
    assert!(
        (b_gain_max - info.b_gain_max).abs() <= 1e-12 * info.b_gain_max.max(1.0),
        "max over trace b_gain ({b_gain_max}) disagrees with the reported b_gain_max ({})",
        info.b_gain_max
    );

    let flips = info
        .trace
        .windows(2)
        .filter(|w| w[0].near != w[1].near)
        .count();
    assert_eq!(
        info.regime_switches, flips,
        "regime_switches {} disagrees with the {flips} Far<->Near flips in trace",
        info.regime_switches
    );
    assert_eq!(x0.space, QnSpace::new(4, 2, 1));
}

#[test]
fn gad_trace_alpha_and_rejections_match_the_run_level_counters() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 40,
        k_davidson: 8,
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&ham, &x0, None, &opts);
    xe.check();
    assert!(info.steps > 0, "vacuous telemetry arm");
    let n = info.trace.len();
    assert_eq!(
        n,
        info.gnorms.len(),
        "fixture gate: this arm assumes an ORDINARY exit (no seed-collapse short trace) so that \
         exactly the last record is the one that took no step"
    );

    for (k, r) in info.trace.iter().enumerate() {
        if k + 1 < n {
            assert!(
                r.alpha.is_finite(),
                "trace[{k}].alpha is NaN, but only the LAST record (index {}) should be — a base \
                 point that is not the last one must have taken an accepted step to produce the \
                 next record",
                n - 1
            );
            assert!(
                r.alpha >= opts.alpha_min - 1e-15 && r.alpha <= opts.alpha_max + 1e-15,
                "trace[{k}].alpha = {} is outside [alpha_min, alpha_max] = [{}, {}]",
                r.alpha,
                opts.alpha_min,
                opts.alpha_max
            );
        } else {
            assert!(
                r.alpha.is_nan(),
                "trace[{k}] is the LAST record on an ordinary exit and must carry the \
                 no-step-taken NaN, not {}",
                r.alpha
            );
        }
    }

    let sum_rejections: usize = info.trace.iter().map(|r| r.rejections).sum();
    assert_eq!(
        sum_rejections, info.rejected_steps,
        "sum of trace[k].rejections ({sum_rejections}) disagrees with the run-level \
         rejected_steps ({}) — a rejection must be attributed to exactly one record's trial loop",
        info.rejected_steps
    );
}
