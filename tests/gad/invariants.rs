// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::solvers::gad::{BorderPolicy, GadOpts, RankAdaptMode, RankAdaptOpts, WModel, gad};
use qntt::solvers::joint::{W_DAMP_S_MAX, W_DAMP_S_MIN, WDampRule};
use qntt::{QnSpace, rayleigh_quotient};

use crate::util::{
    coeff_vector, deflate_radial_span, dense_h_in_sector, dense_sector_eigen, det_states,
    f64_deficient_state, f64_full_rank_arm, f64_truncated_arm, generic_kick_from, radial_mode_at,
    su2_min_schmidt_raw, su2_truncated_arm,
};

#[test]
fn gad_f64_trajectory_invariants() {
    let arm = f64_full_rank_arm();
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 600,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &arm.x0, None, &opts);
    xe.check();
    assert!(
        info.converged,
        "invariant arm needs a converged trajectory (gnorm {:.3e}, {} steps)",
        info.gnorm, info.steps
    );

    assert!(
        info.radial_overlap_max.is_finite(),
        "radial_overlap_max is not finite"
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "max |<V, radial>| over the trajectory was {:.3e}; two-pass deflation should hold this at \
         roundoff",
        info.radial_overlap_max
    );

    let g = &info.gnorms;
    assert_eq!(
        g.len(),
        info.steps + 1,
        "gnorms must record every accepted point including x0"
    );
    assert!(
        g.len() >= 8,
        "trajectory of {} accepted points is too short for a quartile statement",
        g.len()
    );
    assert!(
        g.iter().all(|v| v.is_finite()),
        "non-finite gradient norm logged"
    );
    assert!(
        (g[g.len() - 1] - info.gnorm).abs() <= 1e-12 * info.gnorm.abs().max(1.0),
        "gnorms.last() {} disagrees with info.gnorm {}",
        g[g.len() - 1],
        info.gnorm
    );
    let q = (g.len() / 4).max(2);
    let start = g.len() - q;
    let m0 = g[start] * g[start];
    let m1 = g[g.len() - 1] * g[g.len() - 1];
    assert!(
        m1 <= m0 * (1.0 + 1e-8),
        "‖grad‖² did not decrease over the terminal quartile: {m0:.6e} -> {m1:.6e}"
    );
    for k in start..g.len() - 1 {
        let a = g[k] * g[k];
        let b = g[k + 1] * g[k + 1];
        assert!(
            b <= a * (1.0 + 1e-2),
            "‖grad‖² rose from {a:.6e} to {b:.6e} at accepted step {k} inside the terminal \
             quartile — outside the band"
        );
    }

    assert!(
        info.v_overlap >= 1.0 - 1e-6,
        "final v_overlap {} — the tracked branch is still moving at the endpoint",
        info.v_overlap
    );
}

#[test]
fn gad_certificate_refuses_index0_ground_state() {
    let arm = f64_full_rank_arm();
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 50,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &arm.xgs, None, &opts);
    xe.check();

    assert_eq!(
        info.steps, 0,
        "the certificate is evaluated at the TOP of the loop: a launch point already inside \
         gradtol must not take a step"
    );
    assert!(
        !info.index1_certified,
        "an index-0 minimum was certified as an index-1 saddle"
    );
    assert_eq!(
        info.index_estimate,
        Some(0),
        "the refusal must carry the index estimate, not just a bare false"
    );
    assert!(
        !info.converged,
        "an endpoint failing the index-1 certificate reports converged = false (design §3)"
    );
    let e = info
        .eig1
        .expect("the certificate ran, so its probe-1 eigenvalue must be reported");
    assert!(e > 0.0, "eig1 {e} at a minimum must be positive");
    let want = arm.evals[1] - arm.evals[0];
    assert!(
        (e - want).abs() <= 1e-3 * want,
        "eig1 {e} vs the dense gap E₁ − E₀ = {want}"
    );
    let lam = rayleigh_quotient(&arm.ham, &xe);
    assert!(
        (lam - arm.evals[0]).abs() < 1e-8,
        "a refused endpoint must still be the point it was handed ({lam} vs {})",
        arm.evals[0]
    );
}

#[test]
fn gad_truncated_rank_exercises_weingarten_and_logs_b_gain() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 400,
        k_davidson: 8,
        w_model: WModel::Damped(WDampRule::GradTracked),
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&ham, &x0, None, &opts);
    xe.check();

    assert!(
        info.b_gain_max.is_finite() && info.b_gain_max > 0.0,
        "b_gain_max = {} — either B is not being applied on this truncated fixture, or the \
         diagnostic is not being logged (Phase 2 reads this number)",
        info.b_gain_max
    );
    assert!(
        !info.damp_s.is_empty(),
        "WModel::Damped must log the Tikhonov scale it used at every refresh"
    );
    for &s in &info.damp_s {
        assert!(
            (W_DAMP_S_MIN..=W_DAMP_S_MAX).contains(&s),
            "damping scale {s:.3e} left the measured admissible band [{W_DAMP_S_MIN:.0e}, \
             {W_DAMP_S_MAX:.0e}]"
        );
    }
    assert!(
        info.min_schmidt_min.is_finite() && info.min_schmidt_min > 0.0,
        "min_schmidt_min = {} — the manifold-border tripwire is not being read",
        info.min_schmidt_min
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "radial leak {:.3e} on the truncated fixture",
        info.radial_overlap_max
    );

    if info.index1_certified {
        assert_eq!(info.index_estimate, Some(1));
        assert!(
            info.eig1.expect("certified endpoint reports eig1") < 0.0,
            "index-1 certified with a non-negative ascent eigenvalue"
        );
        assert!(
            info.converged,
            "certified index-1 endpoint must report converged"
        );
        assert!(info.gnorm <= opts.gradtol);
    } else {
        assert!(
            !info.converged,
            "converged must be false whenever the index-1 certificate did not pass"
        );
    }
}

#[test]
fn gad_wmodel_switch_and_full_arms_route_and_report() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let b_switch_tol = 1e-2;
    for (name, w_model) in [
        ("switch", WModel::Switch { b_switch_tol }),
        ("full", WModel::Full),
    ] {
        let opts = GadOpts {
            gradtol: 1e-9,
            max_steps: 120,
            k_davidson: 8,
            w_model,
            certify: false,
            preserve_grid: true,
            ..GadOpts::default()
        };
        let (xe, info) = gad(&ham, &x0, None, &opts);
        xe.check();
        assert!(info.steps > 0, "vacuous {name} arm: no step was taken");
        assert!(
            info.damp_s.is_empty(),
            "WModel::{name} logged {} Tikhonov scales — only WModel::Damped has one",
            info.damp_s.len()
        );
        assert!(
            info.b_gain_max.is_finite() && info.b_gain_max > 0.0,
            "{name}: b_gain_max = {} — the undamped ‖B·V‖/‖A·V‖ diagnostic is model-independent \
             and must be live on this truncated fixture under every arm",
            info.b_gain_max
        );
        let mut n_full = 0usize;
        for (k, r) in info.trace.iter().enumerate() {
            assert!(
                r.damp_s.is_none(),
                "WModel::{name} trace[{k}] carries a Tikhonov scale {:?}",
                r.damp_s
            );
            let want = match w_model {
                WModel::Full => true,
                WModel::Switch { b_switch_tol } => r.gnorm <= b_switch_tol,
                other => unreachable!("this arm constructs no {other:?} model"),
            };
            assert_eq!(
                r.model_is_full, want,
                "WModel::{name} trace[{k}]: model_is_full {} at ‖grad‖ {:.3e} (threshold \
                 {b_switch_tol:.0e})",
                r.model_is_full, r.gnorm
            );
            n_full += usize::from(r.model_is_full);
        }
        println!(
            "GAD WMODEL {name}: steps {} rejected {} applies {} ‖grad‖ {:.2e} b_gain_max {:.3e} \
             exact-model refreshes {n_full}/{}",
            info.steps,
            info.rejected_steps,
            info.applies,
            info.gnorm,
            info.b_gain_max,
            info.trace.len()
        );
        if matches!(w_model, WModel::Switch { .. }) {
            assert!(
                n_full > 0 && n_full < info.trace.len(),
                "the switch threshold {b_switch_tol:.0e} did not straddle this trajectory \
                 ({n_full} of {} refreshes took the exact model) — the arm then exercises only \
                 one branch",
                info.trace.len()
            );
        }
    }
}

#[test]
fn gad_preserve_grid_keeps_declared_grid() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let declared = x0.ranks.clone();
    let opts = GadOpts {
        gradtol: 1e-10,
        max_steps: 40,
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&ham, &x0, None, &opts);
    xe.check();
    assert!(info.steps > 0, "vacuous arm: no step was taken");
    assert_eq!(
        xe.ranks, declared,
        "preserve_grid = true must carry the caller's declared rank grid through every retraction"
    );
    assert!(
        !info.index1_certified,
        "certify = false must never produce a certificate"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn gad_su2_preserve_grid_keeps_declared_grid() {
    let (ham, _xstar, x0) = su2_truncated_arm();
    let space = x0.space;
    let declared = x0.ranks.clone();
    let opts = GadOpts {
        gradtol: 1e-10,
        max_steps: 40,
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = qntt::su2::hess::gad(&ham, &x0, None, &opts);
    xe.check();
    assert!(info.steps > 0, "vacuous SU(2) arm: no step was taken");
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert!(
                xe.ranks[b][s] <= declared[b][s],
                "SU(2) preserve_grid GREW sector {s:?} at bond {b}: {} > {}",
                xe.ranks[b][s],
                declared[b][s]
            );
            if declared[b][s] > 0 {
                assert!(
                    xe.ranks[b][s] > 0,
                    "SU(2) preserve_grid dropped declared sector {s:?} at bond {b} entirely — the \
                     state left the declared manifold"
                );
            }
        }
    }

    let (bl, br) = qntt::su2::riemannian::tangent_bases(&xe);
    let raw = su2_min_schmidt_raw(&bl, &br);
    assert!(
        raw > 0.0,
        "degenerate SU(2) endpoint: a bond Schmidt value collapsed to zero"
    );
    assert!(
        info.min_schmidt_min.is_finite() && info.min_schmidt_min > 0.0,
        "min_schmidt_min = {} — the SU(2) manifold-border tripwire is not being read",
        info.min_schmidt_min
    );
    assert!(
        info.min_schmidt_min <= 4.0 * raw,
        "min_schmidt_min {} is not a Schmidt value in either unit convention (endpoint raw σ_min \
         = {raw})",
        info.min_schmidt_min
    );
}

#[test]
fn gad_su2_min_schmidt_matches_the_centerweighted_sum_rule() {
    let (_ham, _xstar, x0) = su2_truncated_arm();
    let (bl, br) = qntt::su2::riemannian::tangent_bases(&x0);
    let two_s_total = bl.space.two_s;
    let gauges = qntt::su2::weingarten::bond_gauges(
        &bl,
        &br,
        qntt::core::weingarten::FloorPolicy::default_hard(),
    );
    assert!(!gauges.is_empty(), "no interior bonds — degenerate fixture");

    let mut global_min = f64::INFINITY;
    for g in &gauges {
        let mut sum = 0.0f64;
        for (s, sigmas) in &g.sigma {
            let cw = qntt::su2::centerweight(s.two_s, two_s_total);
            for &sig in sigmas {
                sum += cw * sig * sig;
                global_min = global_min.min(cw.sqrt() * sig);
            }
        }
        assert!(
            (sum - 1.0).abs() <= 1e-12,
            "centerweighted Schmidt sum rule failed at bond {}: Σ cw·σ² = {sum} (want 1)",
            g.k
        );
    }
    let reported = qntt::su2::weingarten::min_schmidt(&bl, &br);
    assert!(
        (reported - global_min).abs() <= 1e-12 * global_min.max(1.0),
        "su2::weingarten::min_schmidt {reported} disagrees with the independently-folded \
         centerweighted minimum {global_min}"
    );
}

#[test]
fn gad_preserve_grid_true_holds_a_genuinely_deficient_sector() {
    let (ham, x, bond, sector) = f64_deficient_state();
    let declared = x.ranks[bond][sector];
    assert_eq!(
        declared, 2,
        "fixture gate: expected declared rank 2 at the engineered sector"
    );
    let opts = GadOpts {
        max_steps: 0,
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, _info) = gad(&ham, &x, None, &opts);
    xe.check();
    assert_eq!(
        xe.ranks[bond][sector], declared,
        "preserve_grid = true dropped a declared sector whose content is genuinely near-zero \
         numerically — the entry round must be SKIPPED entirely, not merely tolerant of the \
         fixtures this file otherwise happens to use"
    );
}

#[test]
fn gad_preserve_grid_false_drops_a_genuinely_deficient_sector() {
    let (ham, x, bond, sector) = f64_deficient_state();
    let declared = x.ranks[bond][sector];
    assert_eq!(
        declared, 2,
        "fixture gate: expected declared rank 2 at the engineered sector"
    );
    let opts = GadOpts {
        max_steps: 0,
        certify: false,
        preserve_grid: false,
        ..GadOpts::default()
    };
    let (xe, _info) = gad(&ham, &x, None, &opts);
    xe.check();
    assert!(
        xe.ranks[bond][sector] < declared,
        "preserve_grid = false must drop the engineered near-zero direction via its entry \
         round_mut(1e-14, Abs) — sector stayed at the declared rank {declared}"
    );
}

#[test]
fn gad_converged_reports_true_without_certify_at_a_critical_point() {
    let arm = f64_full_rank_arm();
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 50,
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &arm.xgs, None, &opts);
    xe.check();
    assert_eq!(
        info.steps, 0,
        "already-critical launch point must take no step"
    );
    assert!(
        info.converged,
        "certify = false at a critical point (gnorm {:.3e} <= gradtol) must report converged = \
         true — the certificate half of the formula must short-circuit, not gate a \
         criticality-only run",
        info.gnorm
    );
    assert!(
        !info.index1_certified,
        "certify = false must never certify anything"
    );
    assert_eq!(
        info.index_estimate, None,
        "certify = false must never produce an index estimate"
    );
    assert_eq!(info.eig1, None, "certify = false must never report eig1");
}

#[test]
fn gad_budget_exhaustion_reports_not_converged_at_max_steps() {
    let arm = f64_full_rank_arm();
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 3,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &arm.x0, None, &opts);
    xe.check();
    assert_eq!(
        info.steps, 3,
        "max_steps must cap the accepted-step count exactly"
    );
    assert!(
        !info.converged,
        "a run that exhausts its step budget short of gradtol must report converged = false"
    );
}

#[test]
fn gad_inconclusive_probe1_never_reports_index0() {
    let arm = f64_full_rank_arm();
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 50,
        certify: true,
        probe_budget: 2,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &arm.xgs, None, &opts);
    xe.check();
    assert_eq!(
        info.steps, 0,
        "already-critical launch point must take no step"
    );
    let e = info
        .eig1
        .expect("the certificate ran, so its probe-1 eigenvalue must be reported");
    assert!(
        e > 0.0,
        "eig1 {e} must still be positive at a minimum even under a starved probe"
    );
    assert_eq!(
        info.index_estimate, None,
        "a starved (probe_budget = 2) probe 1 that did not converge must report NO index \
         estimate, not Some(0) — Inconclusive is never evidence"
    );
    assert!(!info.index1_certified);
    assert!(!info.converged);
}

fn expand_dense_eigenvector(arm: &crate::util::FullRankArm, n: usize) -> qntt::TtVector<f64> {
    let ndet = arm.dets.len();
    let mut psi = &arm.dets[0] * arm.evecs[n][0];
    for i in 1..ndet {
        psi = &psi + &(&arm.dets[i] * arm.evecs[n][i]);
    }
    psi.round_ranks_mut(&qntt::max_ranks(&arm.ham.space));
    let norm = psi.norm();
    psi.scale_mut(1.0 / norm);
    psi.check();
    psi
}

#[test]
fn gad_certificate_probe2_witness_negative_reports_lower_bound_two() {
    let arm = f64_full_rank_arm();
    let psi2 = expand_dense_eigenvector(&arm, 2);
    let lam2 = rayleigh_quotient(&arm.ham, &psi2);
    assert!(
        (lam2 - arm.evals[2]).abs() < 1e-8,
        "ψ2 construction failed: {lam2} vs dense E2 {}",
        arm.evals[2]
    );

    let psi0 = expand_dense_eigenvector(&arm, 0);
    let (bl, br) = qntt::solvers::riemannian::tangent_bases(&psi2);
    let radial = radial_mode_at(&bl, &br, &bl);
    let mut v0 = TtTangent::project(&bl, &br, &psi0);
    deflate_radial_span(&radial, &mut v0);
    let n = v0.norm();
    v0.scale_mut(1.0 / n);

    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 0,
        certify: true,
        probe_budget: 200,
        k_davidson: 8,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &psi2, Some(&v0), &opts);
    xe.check();
    assert_eq!(
        info.steps, 0,
        "ψ2 is already critical; the launch must take no step"
    );
    assert!(
        info.v_overlap >= 1.0 - 1e-6,
        "the driver failed to track the seeded ψ0-direction cleanly (v_overlap {}) — the fixture \
         setup, not the assertion below, needs revisiting",
        info.v_overlap
    );
    let e = info
        .eig1
        .expect("the certificate ran, so eig1 must be reported");
    assert!(
        e < 0.0,
        "eig1 {e} must be negative at ψ2 (a genuine ascent direction)"
    );
    assert_eq!(
        info.index_estimate,
        Some(2),
        "probe 2 must witness the SECOND negative direction (toward ψ1) after deflating only the \
         tracked ψ0-direction — got {:?}",
        info.index_estimate
    );
    assert!(!info.index1_certified, "ψ2 must never certify as index-1");
    assert!(!info.converged);
}

#[test]
fn gad_certificate_probe1_misaligned_witness_classifies_point_and_flags_branch() {
    let arm = f64_full_rank_arm();
    let psi2 = expand_dense_eigenvector(&arm, 2);
    let psi0 = expand_dense_eigenvector(&arm, 0);
    let psi1 = expand_dense_eigenvector(&arm, 1);

    let (bl, br) = qntt::solvers::riemannian::tangent_bases(&psi2);
    let radial = radial_mode_at(&bl, &br, &bl);
    let mut v_mix = TtTangent::project(&bl, &br, &psi0);
    let v1 = TtTangent::project(&bl, &br, &psi1);
    v_mix.axpy(1.0, &v1);
    deflate_radial_span(&radial, &mut v_mix);
    let n = v_mix.norm();
    v_mix.scale_mut(1.0 / n);

    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 0,
        certify: true,
        probe_budget: 200,
        k_davidson: 2,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &psi2, Some(&v_mix), &opts);
    xe.check();
    assert_eq!(
        info.steps, 0,
        "ψ2 is already critical; the launch must take no step"
    );
    let e = info
        .eig1
        .expect("the certificate ran, so eig1 must be reported");
    assert!(
        e < 0.0,
        "eig1 {e} must still be a genuine negative direction (probe 1 must find one)"
    );
    let wov = info
        .witness_overlap
        .expect("probe 1 found a negative direction, so the witness overlap must be reported");
    assert!(
        wov < 1.0 - 1e-3,
        "the tracked mixture must be misaligned with probe 1's witness (|<w,V>| = {wov})"
    );
    assert!(
        !info.tracked_mode_is_lowest,
        "a misaligned witness must be reported as tracked_mode_is_lowest = false"
    );
    assert_eq!(
        info.index_estimate,
        Some(2),
        "ψ2 is index 2 at full rank: deflating probe 1's own witness must leave the second \
         negative direction for probe 2 to witness (lower bound 2) — got {:?}",
        info.index_estimate
    );
    assert!(!info.index1_certified, "ψ2 must never certify as index-1");
    assert!(!info.converged);
}

#[test]
fn gad_certificate_wrong_branch_at_index1_saddle_certifies_point_flags_branch() {
    let arm = f64_full_rank_arm();
    let psi1 = expand_dense_eigenvector(&arm, 1);
    let psi0 = expand_dense_eigenvector(&arm, 0);
    let psi2 = expand_dense_eigenvector(&arm, 2);
    let (bl, br) = qntt::solvers::riemannian::tangent_bases(&psi1);
    let radial = radial_mode_at(&bl, &br, &bl);
    let direction = |psi: &qntt::TtVector<f64>| {
        let mut v = TtTangent::project(&bl, &br, psi);
        deflate_radial_span(&radial, &mut v);
        let n = v.norm();
        v.scale_mut(1.0 / n);
        v
    };
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 0,
        certify: true,
        probe_budget: 200,
        k_davidson: 2,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let mut wrong = direction(&psi2);
    wrong.axpy(0.1, &direction(&psi0));
    let n = wrong.norm();
    wrong.scale_mut(1.0 / n);
    for (name, v0, want_lowest) in [
        ("wrong branch (ψ2 + 0.1 ψ0 direction)", wrong, false),
        ("aligned (ψ0 direction)", direction(&psi0), true),
    ] {
        let (xe, info) = gad(&arm.ham, &psi1, Some(&v0), &opts);
        xe.check();
        assert_eq!(info.steps, 0, "{name}: ψ1 is critical, no step");
        let e = info.eig1.expect("certificate ran");
        assert!(
            e < 0.0,
            "{name}: eig1 {e} must be the negative direction toward ψ0"
        );
        let wov = info
            .witness_overlap
            .expect("probe 1 found a negative direction");
        assert_eq!(
            info.tracked_mode_is_lowest, want_lowest,
            "{name}: tracked_mode_is_lowest (|<w,V>| = {wov})"
        );
        if want_lowest {
            assert!(wov > 1.0 - 1e-3, "{name}: |<w,V>| = {wov}");
        } else {
            assert!(
                wov < 0.5,
                "{name}: |<w,V>| = {wov} — a positive-curvature v0 must not align"
            );
        }
        assert_eq!(
            info.index_estimate,
            Some(1),
            "{name}: ψ1 is an index-1 saddle whatever branch was tracked — got {:?}",
            info.index_estimate
        );
        assert!(info.index1_certified, "{name}: the point must certify");
        assert!(
            info.converged,
            "{name}: certified critical point must report converged"
        );
    }
}

#[test]
fn gad_schmidt_floor_hit_is_driven_both_ways() {
    let (ham, _xstar, x0) = f64_truncated_arm();

    let opts_hi = GadOpts {
        gradtol: 1e-9,
        max_steps: 5,
        certify: false,
        preserve_grid: true,
        schmidt_floor: 1.0,
        ..GadOpts::default()
    };
    let (xe_hi, info_hi) = gad(&ham, &x0, None, &opts_hi);
    xe_hi.check();
    assert!(
        info_hi.schmidt_floor_hit,
        "schmidt_floor = 1.0 must trip the tripwire (min_schmidt_min = {})",
        info_hi.min_schmidt_min
    );
    assert!(info_hi.min_schmidt_min < 1.0);

    let opts_lo = GadOpts {
        gradtol: 1e-9,
        max_steps: 5,
        certify: false,
        preserve_grid: true,
        schmidt_floor: 0.0,
        ..GadOpts::default()
    };
    let (xe_lo, info_lo) = gad(&ham, &x0, None, &opts_lo);
    xe_lo.check();
    assert!(
        !info_lo.schmidt_floor_hit,
        "schmidt_floor = 0.0 must never trip the tripwire (min_schmidt_min = {})",
        info_lo.min_schmidt_min
    );
}

#[test]
fn gad_stationary_exit_reports_incumbent_not_rejected_trial() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 40,
        certify: false,
        preserve_grid: true,
        alpha0: 100.0,
        alpha_min: 100.0,
        alpha_max: 100.0,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&ham, &x0, None, &opts);
    xe.check();
    assert_eq!(
        info.steps, 0,
        "the single oversized trial must be rejected, not accepted"
    );
    assert_eq!(
        info.rejected_steps, 1,
        "alpha_min == alpha0 must terminate after exactly one rejected trial"
    );
    assert!(!info.converged);
    let lam0 = rayleigh_quotient(&ham, &x0);
    let lame = rayleigh_quotient(&ham, &xe);
    assert!(
        (lam0 - lame).abs() <= 1e-12 * lam0.abs().max(1.0),
        "a stationary exit must return the INCUMBENT x0 (λ {lam0}), not the rejected trial (λ \
         {lame})"
    );
}

#[test]
fn gad_fixed_seed_is_deterministic() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 60,
        seed: 0x6AD5EED,
        certify: true,
        probe_budget: 120,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xa, ia) = gad(&ham, &x0, None, &opts);
    let (xb, ib) = gad(&ham, &x0, None, &opts);

    assert_eq!(ia.steps, ib.steps, "step count is not reproducible");
    assert_eq!(
        ia.rejected_steps, ib.rejected_steps,
        "rejection count is not reproducible"
    );
    assert_eq!(ia.applies, ib.applies, "apply count is not reproducible");
    assert_eq!(
        ia.index1_certified, ib.index1_certified,
        "certificate outcome is not reproducible"
    );
    assert!(
        (ia.lambda - ib.lambda).abs() <= 1e-13,
        "lambda {} vs {}",
        ia.lambda,
        ib.lambda
    );
    assert!(
        (ia.gnorm - ib.gnorm).abs() <= 1e-13,
        "gnorm {:.17e} vs {:.17e}",
        ia.gnorm,
        ib.gnorm
    );
    let dets = det_states::<f64>(4, 2, 1);
    let ca = coeff_vector(&dets, &xa);
    let cb = coeff_vector(&dets, &xb);
    let worst = ca
        .iter()
        .zip(cb.iter())
        .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
    assert!(
        worst <= 1e-13,
        "endpoint states differ by {worst:.3e} in the determinant basis"
    );
    assert_eq!(x0.space, QnSpace::new(4, 2, 1));
    assert!(ia.steps > 0, "vacuous determinism arm");
}

#[test]
#[should_panic(expected = "round_eps")]
fn gad_rank_adapt_refuses_a_non_positive_budget_by_name() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let opts = GadOpts {
        max_steps: 1,
        rank_adapt: Some(RankAdaptOpts {
            round_eps: 0.0,
            ..RankAdaptOpts::default()
        }),
        ..GadOpts::default()
    };
    let _ = gad(&ham, &x0, None, &opts);
}

#[test]
#[should_panic(expected = "period")]
fn gad_rank_adapt_refuses_a_zero_period_by_name() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let opts = GadOpts {
        max_steps: 1,
        rank_adapt: Some(RankAdaptOpts {
            period: 0,
            ..RankAdaptOpts::default()
        }),
        ..GadOpts::default()
    };
    let _ = gad(&ham, &x0, None, &opts);
}

#[test]
fn gad_generic_kick_does_not_stall_at_alpha_min() {
    let arm = f64_full_rank_arm();
    let x0 = generic_kick_from(&arm.xgs, 5e-2, 0xA1);
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 600,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &x0, None, &opts);
    xe.check();

    assert!(
        !(info.steps < 10 && info.gnorm > 1e-3),
        "step controller went stationary after {} accepted steps ({} rejections) at ‖grad‖ = \
         {:.3e} — the latch did not re-arm",
        info.steps,
        info.rejected_steps,
        info.gnorm
    );
    assert!(
        info.converged || info.steps >= opts.max_steps,
        "run ended neither converged nor at max_steps: steps {}, rejections {}, ‖grad‖ {:.3e}, \
         index {:?}",
        info.steps,
        info.rejected_steps,
        info.gnorm,
        info.index_estimate
    );
    let g = &info.gnorms;
    let mut re_climb = false;
    for k in 1..g.len() {
        if g[k] < g[k - 1] && g[k + 1..].iter().any(|v| *v > 2.0 * g[k]) {
            re_climb = true;
            break;
        }
    }
    assert!(
        re_climb,
        "the generic-kick launch was expected to descend and then re-climb (measured: a 7.0e-2 \
         trough followed by a peak above 0.6); no such pair exists in the {} recorded points, so \
         the arm no longer exercises the regime change and its premise must be re-measured. \
         relatches = {}",
        g.len(),
        info.relatches
    );
    assert_eq!(
        info.gnorms.len(),
        info.steps + 1,
        "gnorms must record every accepted point including x0"
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "radial leak {:.3e} on the generic-kick trajectory",
        info.radial_overlap_max
    );
}

#[test]
fn gad_su2_truncated_damped_arm() {
    let (ham, _xstar, x0) = su2_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 200,
        k_davidson: 8,
        w_model: WModel::Damped(WDampRule::GradTracked),
        certify: true,
        probe_budget: 150,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = qntt::su2::hess::gad(&ham, &x0, None, &opts);
    xe.check();

    assert!(
        !info.damp_s.is_empty(),
        "WModel::Damped must log the Tikhonov scale it used at every refresh"
    );
    for &s in &info.damp_s {
        assert!(
            (W_DAMP_S_MIN..=W_DAMP_S_MAX).contains(&s),
            "damping scale {s:.3e} left the measured admissible band [{W_DAMP_S_MIN:.0e}, \
             {W_DAMP_S_MAX:.0e}]"
        );
    }
    assert!(
        info.b_gain_max.is_finite() && info.b_gain_max > 0.0,
        "b_gain_max = {} — the centerweighted B is not being applied on this truncated SU(2) \
         fixture, or the diagnostic is not logged (Phase 2 reads this number)",
        info.b_gain_max
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "radial leak {:.3e} on the SU(2) truncated fixture — the {{x}} deflation is the guard \
         against converging to the ground state disguised as a saddle",
        info.radial_overlap_max
    );

    if info.index1_certified {
        assert_eq!(info.index_estimate, Some(1));
        assert!(
            info.eig1.expect("certified endpoint reports eig1") < 0.0,
            "index-1 certified with a non-negative ascent eigenvalue"
        );
        assert!(
            info.converged,
            "certified index-1 endpoint must report converged"
        );
        assert!(info.gnorm <= opts.gradtol);
    } else {
        assert!(
            !info.converged,
            "converged must be false whenever the index-1 certificate did not pass"
        );
    }
}

fn pooled_ranks(x: &qntt::TtVector<f64>) -> Vec<usize> {
    let sp = x.space;
    (0..=sp.d)
        .map(|b| sp.bond_sectors(b).into_iter().map(|s| x.ranks[b][s]).sum())
        .collect()
}

#[test]
fn gad_rank_adapt_both_grows_to_a_dense_eigenvalue() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let space = x0.space;
    let dets = det_states::<f64>(space.d, space.nup, space.ndn);
    let (evals, _) = dense_sector_eigen(&dense_h_in_sector(&dets, &ham), dets.len());
    let launch = pooled_ranks(&x0);
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 600,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        rank_adapt: Some(RankAdaptOpts {
            mode: RankAdaptMode::Both,
            chi: qntt::Pooled::NONE,
            round_eps: 1e-10,
            period: 1,
        }),
        ..GadOpts::default()
    };
    let (xe, info) = gad(&ham, &x0, None, &opts);
    xe.check();
    let end = pooled_ranks(&xe);
    println!(
        "GAD RANK-ADAPT both: launch {launch:?} -> end {end:?}, grid_changes {}, steps {} \
         applies {} converged {} index {:?} λ {:.9}",
        info.grid_changes,
        info.steps,
        info.applies,
        info.converged,
        info.index_estimate,
        rayleigh_quotient(&ham, &xe)
    );
    assert!(
        info.grid_changes >= 1,
        "Both never changed the grid from the truncated launch"
    );
    assert!(
        end.iter().zip(&launch).any(|(e, l)| e > l),
        "Both must GROW some bond above the truncated launch ({launch:?} -> {end:?})"
    );
    let n = (xe.norm() - 1.0).abs();
    assert!(n < 1e-12, "adapted endpoint must stay unit-norm ({n:.3e})");
    for (k, r) in info.trace.iter().enumerate() {
        assert!(r.pooled_rank >= 1, "trace[{k}]: pooled_rank not logged");
    }
    assert!(
        info.converged && info.index1_certified && info.index_estimate == Some(1),
        "the adaptive run must reach a certified index-1 saddle within 3× headroom: converged \
         {} index {:?} steps {}",
        info.converged,
        info.index_estimate,
        info.steps
    );
    let lam = rayleigh_quotient(&ham, &xe);
    let (k_best, gap) = evals
        .iter()
        .enumerate()
        .map(|(k, e)| (k, (e - lam).abs()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    println!(
        "GAD RANK-ADAPT both: nearest dense level E{k_best} = {:.9} (gap {gap:.3e})",
        evals[k_best]
    );
    assert!(
        gap < 1e-6,
        "at representing rank a certified saddle must sit on a dense level: nearest E{k_best} \
         off by {gap:.3e}"
    );
    assert!(
        k_best >= 1,
        "an index-1 saddle cannot be the ground state E0"
    );
}

#[test]
fn gad_rank_adapt_shrink_only_drops_border_directions_and_never_grows() {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    let space = QnSpace::new(8, 4, 4);
    let ham = crate::util::generic_truncated_h::<f64>(space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0B08_DE8B);
    let mut x: qntt::TtVector<f64> = crate::util::randomized_state(space, 1, 1, &mut rng);
    x.round_mut(1e-14, qntt::Pooled::NONE, qntt::round::RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let noise: qntt::TtVector<f64> = qntt::tt_randn(space, &x.ranks, &mut rng);
    let mut xb = &x + &(&noise * (1e-7 / noise.norm()));
    xb.scale_mut(1.0 / xb.norm());
    let launch = pooled_ranks(&xb);
    let base = pooled_ranks(&x);
    let full = qntt::max_ranks(&space);
    for b in 3..=space.d - 3 {
        let cap: usize = space.bond_sectors(b).into_iter().map(|s| full[b][s]).sum();
        assert!(
            launch[b] == 2 * base[b] && launch[b] < cap,
            "fixture gate: bond {b} must carry a doubled grid below the representing rank \
             ({} -> {}, cap {cap})",
            base[b],
            launch[b]
        );
    }
    let run = |rank_adapt: Option<RankAdaptOpts>| {
        let opts = GadOpts {
            gradtol: 1e-8,
            max_steps: 30,
            k_davidson: 8,
            certify: false,
            preserve_grid: true,
            rank_adapt,
            ..GadOpts::default()
        };
        let (xe, info) = gad(&ham, &xb, None, &opts);
        xe.check();
        assert!(info.steps >= 1, "vacuous: no accepted step");
        (xe, info)
    };
    let (xc, ic) = run(None);
    let (xe, info) = run(Some(RankAdaptOpts {
        mode: RankAdaptMode::ShrinkOnly,
        chi: qntt::Pooled::NONE,
        round_eps: 1e-5,
        period: 1,
    }));
    let end = pooled_ranks(&xe);
    let ctl = pooled_ranks(&xc);
    println!(
        "GAD RANK-ADAPT shrink: border launch {launch:?} -> control {ctl:?} ({} grid changes) | \
         shrink {end:?} ({} grid changes, {} steps)",
        ic.grid_changes, info.grid_changes, info.steps
    );
    assert!(
        info.grid_changes >= 1,
        "ShrinkOnly never dropped the border directions"
    );
    assert!(
        end.iter().zip(&launch).any(|(e, l)| e < l),
        "ShrinkOnly must shrink some bond below the border launch ({launch:?} -> {end:?})"
    );
    for b in 0..=space.d {
        assert!(
            end[b] <= ctl[b],
            "ShrinkOnly ended ABOVE the fixed-grid control at bond {b}: {} > {}",
            end[b],
            ctl[b]
        );
        for s in space.bond_sectors(b) {
            assert!(
                xe.ranks[b][s] <= xb.ranks[b][s],
                "ShrinkOnly GREW sector {s:?} at bond {b}: {} > {}",
                xe.ranks[b][s],
                xb.ranks[b][s]
            );
        }
    }
    let mut prev = usize::MAX;
    for (k, r) in info.trace.iter().enumerate() {
        assert!(
            r.pooled_rank <= prev,
            "trace[{k}]: pooled_rank rose {prev} -> {}",
            r.pooled_rank
        );
        prev = r.pooled_rank;
    }
}

#[test]
fn gad_rank_adapt_both_never_opens_a_launch_zero_sector() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let space = x0.space;
    let (bond, sector) = (1..space.d)
        .flat_map(|b| space.bond_sectors(b).into_iter().map(move |s| (b, s)))
        .find(|&(b, s)| x0.ranks[b][s] >= 1 && space.bond_sectors(b).len() >= 2)
        .expect("fixture gate: no interior multi-sector bond with a populated sector");
    let mut target = x0.ranks.clone();
    target[bond][sector] = 0;
    let mut xp = x0.clone();
    xp.round_ranks_mut(&target);
    xp.scale_mut(1.0 / xp.norm());
    assert_eq!(
        xp.ranks[bond][sector], 0,
        "fixture gate: the pin did not take"
    );
    for preserve_grid in [true, false] {
        let opts = GadOpts {
            gradtol: 1e-8,
            max_steps: 60,
            k_davidson: 8,
            certify: false,
            preserve_grid,
            rank_adapt: Some(RankAdaptOpts {
                mode: RankAdaptMode::Both,
                chi: qntt::Pooled::NONE,
                round_eps: 1e-10,
                period: 1,
            }),
            ..GadOpts::default()
        };
        let (xe, info) = gad(&ham, &xp, None, &opts);
        xe.check();
        println!(
            "GAD RANK-ADAPT zero-sector (preserve_grid = {preserve_grid}): bond {bond} sector \
             {sector:?} -> rank {}, grid_changes {}, steps {}",
            xe.ranks[bond][sector], info.grid_changes, info.steps
        );
        assert!(info.steps >= 1, "vacuous: no accepted step");
        assert!(
            info.grid_changes >= 1,
            "the growth rounds must adapt the populated sectors"
        );
        assert_eq!(
            xe.ranks[bond][sector], 0,
            "a launch-zero sector was OPENED by a growth round (preserve_grid = {preserve_grid})"
        );
    }
}

#[test]
fn gad_su2_rank_adapt_both_grows_within_cap() {
    let (ham, _xstar, x0) = su2_truncated_arm();
    let space = x0.space;
    let pooled = |x: &qntt::su2::vector::Su2TtVector<f64>| -> Vec<usize> {
        (0..=space.d)
            .map(|b| {
                space
                    .bond_sectors(b)
                    .into_iter()
                    .map(|s| x.ranks[b][s])
                    .sum()
            })
            .collect()
    };
    let start_pooled = *pooled(&x0).iter().max().expect("non-empty grid");
    let max_pooled = start_pooled + 1;
    let opts_for = |chi: qntt::Pooled| GadOpts {
        gradtol: 1e-8,
        max_steps: 300,
        k_davidson: 8,
        certify: true,
        probe_budget: 150,
        preserve_grid: true,
        rank_adapt: Some(RankAdaptOpts {
            mode: RankAdaptMode::Both,
            chi,
            round_eps: 1e-8,
            period: 1,
        }),
        ..GadOpts::default()
    };
    let (xe, info) = qntt::su2::hess::gad(&ham, &x0, None, &opts_for(qntt::Pooled(max_pooled)));
    xe.check();
    let (xu, info_u) = qntt::su2::hess::gad(&ham, &x0, None, &opts_for(qntt::Pooled::NONE));
    xu.check();
    println!(
        "GAD SU(2) RANK-ADAPT both: {:?} -> {:?} (chi = {max_pooled}, start pooled \
         {start_pooled}), grid_changes {}, steps {} converged {} index {:?}; UNCAPPED control -> \
         {:?}, grid_changes {}, steps {}",
        pooled(&x0),
        pooled(&xe),
        info.grid_changes,
        info.steps,
        info.converged,
        info.index_estimate,
        pooled(&xu),
        info_u.grid_changes,
        info_u.steps
    );
    assert!(info.grid_changes >= 1, "SU(2) Both never changed the grid");
    for b in 0..=space.d {
        let total: usize = space
            .bond_sectors(b)
            .into_iter()
            .map(|s| xe.ranks[b][s])
            .sum();
        assert!(
            total <= max_pooled,
            "bond {b} exceeds the POOLED cap: total bond dimension {total} > {max_pooled}"
        );
    }
    let end_capped = *pooled(&xe).iter().max().expect("non-empty grid");
    let end_uncapped = *pooled(&xu).iter().max().expect("non-empty grid");
    assert!(
        end_uncapped > max_pooled || end_capped == max_pooled,
        "the pooled cap did not bind: capped endpoint max pooled {end_capped}, uncapped \
         {end_uncapped}, chi = {max_pooled}"
    );
    assert!(
        (xe.norm() - 1.0).abs() < 1e-12,
        "SU(2) adapted endpoint must stay unit-norm"
    );
    assert!(
        info.trace.iter().all(|r| r.pooled_rank >= 1),
        "pooled_rank not logged on SU(2)"
    );
    if info.index1_certified {
        assert_eq!(info.index_estimate, Some(1));
        assert!(info.converged);
    } else {
        assert!(!info.converged);
    }
}

#[test]
fn gad_border_refuse_by_schmidt_floor_keeps_the_trajectory_and_drops_the_certificate() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let run = |schmidt_floor: f64| {
        let opts = GadOpts {
            gradtol: 1e-9,
            max_steps: 400,
            k_davidson: 8,
            certify: true,
            probe_budget: 200,
            preserve_grid: true,
            schmidt_floor,
            ..GadOpts::default()
        };
        let (xe, info) = gad(&ham, &x0, None, &opts);
        xe.check();
        (rayleigh_quotient(&ham, &xe), info)
    };
    let (lam_ref, r) = run(1e-10);
    let (lam, b) = run(1.0);
    assert!(
        r.gnorm <= 1e-9 && r.index1_certified,
        "fixture gate: the reference run must reach gradtol and certify (gnorm {:.2e}, index {:?})",
        r.gnorm,
        r.index_estimate
    );
    assert!(
        !r.border_verdict,
        "the default floor must not read this interior endpoint as a border"
    );
    assert!(
        b.border_verdict,
        "schmidt_floor = 1.0 must yield a border verdict at the endpoint"
    );
    assert!(
        !b.index1_certified && !b.converged,
        "a border endpoint is never certified nor converged"
    );
    assert_eq!(
        b.index_estimate, None,
        "no index is claimed at a border endpoint"
    );
    assert_eq!(b.eig1, None, "probe 1 must not have run");
    assert_eq!(
        b.steps, r.steps,
        "the refusal must not change the trajectory"
    );
    assert_eq!(
        b.gnorms, r.gnorms,
        "the refusal must not change the trajectory"
    );
    assert_eq!(
        lam.to_bits(),
        lam_ref.to_bits(),
        "same endpoint, bit for bit"
    );
    assert!(
        b.applies < r.applies,
        "the skipped probes must show in the apply count ({} vs {})",
        b.applies,
        r.applies
    );
    assert!(b.b_gain_end.is_finite() && b.b_gain_end == r.b_gain_end);
}

#[test]
fn gad_border_refuse_by_gain_ceiling_is_live_only_where_b_is() {
    let (ham, _xstar, x0) = f64_truncated_arm();
    let mk = |gain_max: f64| GadOpts {
        gradtol: 1e-9,
        max_steps: 400,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        border: BorderPolicy::Refuse { gain_max },
        ..GadOpts::default()
    };
    let (xr, r) = gad(&ham, &x0, None, &mk(1.0));
    let (xb, b) = gad(&ham, &x0, None, &mk(1e-3));
    xr.check();
    xb.check();
    assert!(
        r.index1_certified && !r.border_verdict,
        "gate: default ceiling certifies (gain {:.3e})",
        r.b_gain_end
    );
    assert!(
        r.b_gain_end > 1e-3,
        "gate: the truncated endpoint must carry a live B (gain {:.3e})",
        r.b_gain_end
    );
    assert!(b.border_verdict && !b.index1_certified && b.index_estimate.is_none());
    assert_eq!(
        rayleigh_quotient(&ham, &xb).to_bits(),
        rayleigh_quotient(&ham, &xr).to_bits(),
        "the gain refusal must not move the endpoint"
    );

    let arm = f64_full_rank_arm();
    let psi1 = expand_dense_eigenvector(&arm, 1);
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 0,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        border: BorderPolicy::Refuse { gain_max: 1e-3 },
        ..GadOpts::default()
    };
    let (xe, info) = gad(&arm.ham, &psi1, None, &opts);
    xe.check();
    assert!(
        info.b_gain_end.abs() < 1e-10,
        "B ≡ 0 at full rank, so the gain is roundoff ({:.3e})",
        info.b_gain_end
    );
    assert!(
        !info.border_verdict,
        "an absurd ceiling refuses nothing where B vanishes"
    );
    assert_eq!(info.index_estimate, Some(1));
    assert!(info.converged);
}
