// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::weingarten::{FloorPolicy, bond_gauges};
use qntt::solvers::gad::{GadOpts, WModel, gad};
use qntt::solvers::joint::{W_DAMP_S_MAX, W_DAMP_S_MIN, WDampRule};
use qntt::solvers::riemannian::{RiemannianOpts, riemannian_cg, riemannian_gd, tangent_bases};

use crate::util::{
    f64_deficient_declarable_state, f64_deficient_state, longest_frozen_streak, su2_full_rank_arm,
    su2_generic_kick_from, su2_truncated_arm,
};

#[test]
fn gad_par_preserve_grid_holds_a_deficient_sector_through_a_trajectory() {
    let (ham, x, bond, sector) = f64_deficient_state();
    let space = x.space;
    let declared = x.ranks.clone();
    let caps = qntt::max_ranks(&space);
    assert_eq!(
        declared[bond][sector], 2,
        "fixture gate: expected declared rank 2 at the engineered sector"
    );
    assert_eq!(
        caps[bond][sector], 1,
        "fixture gate: the engineered sector is expected to be declared ABOVE its path count \
         (cap {}); the finding this arm records is about that over-declaration",
        caps[bond][sector]
    );

    let sigma: Vec<f64> = {
        let (bl, br) = tangent_bases(&x);
        bond_gauges(&bl, &br, FloorPolicy::default_hard())
            .get(bond - 1)
            .and_then(|g| {
                g.sigma
                    .iter()
                    .find(|(s, _)| *s == sector)
                    .map(|(_, v)| v.clone())
            })
            .expect("the engineered bond/sector must carry a gauge block")
    };
    assert_eq!(sigma.len(), 2, "gauge block is not at the declared rank");
    assert_eq!(
        sigma[1], 0.0,
        "fixture gate: the over-declared direction is expected to be EXACTLY null (measured 0.0); \
         it now carries {:.3e}",
        sigma[1]
    );

    let opts = GadOpts {
        gradtol: 1e-10,
        max_steps: 40,
        k_davidson: 8,
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad(&ham, &x, None, &opts);
    xe.check();
    assert!(
        info.steps >= 5,
        "vacuous arm: only {} accepted steps were taken, so the retraction's fixed-rank round was \
         barely exercised",
        info.steps
    );

    let diff: Vec<(usize, usize, usize, usize)> = (0..=space.d)
        .flat_map(|b| space.bond_sectors(b).into_iter().map(move |s| (b, s)))
        .filter(|&(b, s)| xe.ranks[b][s] != declared[b][s])
        .map(|(b, s)| (b, declared[b][s], caps[b][s], xe.ranks[b][s]))
        .collect();
    println!(
        "GAD PRESERVE_GRID (over-declared fixture): {} accepted steps, sigma@launch {sigma:?}, \
         grid diff (bond, declared, cap, final) {diff:?}",
        info.steps
    );
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert_eq!(
                xe.ranks[b][s],
                declared[b][s].min(caps[b][s]),
                "preserve_grid must carry min(declared, max_ranks) at bond {b} sector {s:?} \
                 through every retraction: declared {}, cap {}, final {}",
                declared[b][s],
                caps[b][s],
                xe.ranks[b][s]
            );
        }
    }
    assert_eq!(
        diff.len(),
        12,
        "the measured over-declaration is 12 of the 14 interior (bond, sector) pairs; {} moved \
         instead: {diff:?}",
        diff.len()
    );
    assert_eq!(
        info.min_schmidt_min, 0.0,
        "the over-declared launch grid is expected to report an exactly-zero smallest Schmidt \
         value; it reports {}",
        info.min_schmidt_min
    );
    assert_eq!(
        info.trace[0].min_schmidt, 0.0,
        "the zero must be the LAUNCH point's, not a later one"
    );
    for (k, r) in info.trace.iter().enumerate().skip(1) {
        assert!(
            r.min_schmidt.is_finite() && r.min_schmidt > 0.0,
            "accepted point {k} still carries a zero Schmidt value ({}) — the retraction did not \
             restore a representable grid",
            r.min_schmidt
        );
    }

    let mut ropts = RiemannianOpts::new(&space);
    ropts.eps = 1e-13;
    ropts.max_iter = 40;
    ropts.gradtol = 1e-13;
    ropts.preserve_grid = true;
    let (_lg, xg) = riemannian_gd(&ham, &x, &ropts);
    let (_lc, xc) = riemannian_cg(&ham, &x, &ropts);
    assert_eq!(
        (xg.ranks.clone(), xc.ranks.clone()),
        (xe.ranks.clone(), xe.ranks.clone()),
        "riemannian_gd/cg at preserve_grid = true produced a DIFFERENT grid from GAD's on the \
         same launch point — the reveal would then be specific to one call site and this arm must \
         be escalated, not recorded"
    );
}

#[test]
fn gad_par_preserve_grid_holds_a_representable_deficient_sector() {
    let mut resolved = 0usize;
    for scale in [1e-6f64, 1e-12, 1e-13] {
        let (ham, x, bond, sector) = f64_deficient_declarable_state(scale);
        let declared = x.ranks.clone();
        assert_eq!(
            declared[bond][sector], 2,
            "fixture gate: the engineered sector must be declared at its cap 2"
        );
        let sigma: Vec<f64> = {
            let (bl, br) = tangent_bases(&x);
            bond_gauges(&bl, &br, FloorPolicy::default_hard())
                .get(bond - 1)
                .and_then(|g| {
                    g.sigma
                        .iter()
                        .find(|(s, _)| *s == sector)
                        .map(|(_, v)| v.clone())
                })
                .expect("the engineered bond/sector must carry a gauge block")
        };
        assert_eq!(sigma.len(), 2, "gauge block is not at the declared rank");

        let opts = GadOpts {
            gradtol: 1e-10,
            max_steps: 40,
            k_davidson: 8,
            certify: false,
            preserve_grid: true,
            ..GadOpts::default()
        };
        let (xe, info) = gad(&ham, &x, None, &opts);
        xe.check();
        assert!(
            info.steps >= 5,
            "vacuous arm at scale {scale:.0e}: only {} accepted steps",
            info.steps
        );
        let kept = xe.ranks[bond][sector];
        println!(
            "GAD PRESERVE_GRID (representable, scale {scale:.0e}): sigma@launch {sigma:?}, \
             rank {} -> {kept} after {} accepted steps, min_schmidt_min {:.3e}",
            declared[bond][sector], info.steps, info.min_schmidt_min
        );
        if sigma[1] > 1e-13 {
            resolved += 1;
            assert_eq!(
                xe.ranks, declared,
                "preserve_grid dropped a REPRESENTABLE declared direction whose Schmidt value is \
                 {:.3e} at launch (scale {scale:.0e}) — that is the hazard \
                 `RiemannianOpts::preserve_grid` documents, materializing on a grid that IS a \
                 manifold. Escalate it; do not weaken this assertion.",
                sigma[1]
            );
            assert!(
                info.min_schmidt_min > 0.0,
                "a representable declared grid must keep every Schmidt value positive"
            );
        }
    }
    assert!(
        resolved >= 2,
        "only {resolved} of the three rungs put the engineered direction above the reveal's \
         resolution — the ladder no longer brackets the threshold and says nothing"
    );
}

#[test]
fn gad_par_su2_trajectory_invariants() {
    let (ham, _xgs, x0) = su2_full_rank_arm();
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 600,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = qntt::su2::hess::gad(&ham, &x0, None, &opts);
    xe.check();
    assert!(
        info.converged,
        "SU(2) invariant arm needs a converged trajectory (‖grad‖ {:.3e}, {} steps)",
        info.gnorm, info.steps
    );

    assert!(
        info.radial_overlap_max.is_finite() && info.radial_overlap_max <= 1e-8,
        "max |⟨V, radial⟩| over the SU(2) trajectory was {:.3e}",
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
        "SU(2) trajectory of {} accepted points is too short for a quartile statement",
        g.len()
    );
    assert!(g.iter().all(|v| v.is_finite()));
    assert!(
        (g[g.len() - 1] - info.gnorm).abs() <= 1e-12 * info.gnorm.abs().max(1.0),
        "gnorms.last() disagrees with info.gnorm"
    );
    let q = (g.len() / 4).max(2);
    let start = g.len() - q;
    let (m0, m1) = (g[start] * g[start], g[g.len() - 1] * g[g.len() - 1]);
    assert!(
        m1 <= m0 * (1.0 + 1e-8),
        "SU(2) ‖grad‖² did not decrease over the terminal quartile: {m0:.6e} -> {m1:.6e}"
    );
    for k in start..g.len() - 1 {
        let (a, b) = (g[k] * g[k], g[k + 1] * g[k + 1]);
        assert!(
            b <= a * (1.0 + 1e-2),
            "SU(2) ‖grad‖² rose from {a:.6e} to {b:.6e} at accepted step {k} inside the terminal \
             quartile"
        );
    }
    assert!(
        info.v_overlap >= 1.0 - 1e-6,
        "SU(2) final v_overlap {}",
        info.v_overlap
    );
}

#[test]
fn gad_par_su2_certificate_refuses_index0_ground_state() {
    let (ham, xgs, _x0) = su2_full_rank_arm();
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 50,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = qntt::su2::hess::gad(&ham, &xgs, None, &opts);
    xe.check();

    assert_eq!(
        info.steps, 0,
        "an SU(2) launch point already inside gradtol must not take a step"
    );
    assert!(
        !info.index1_certified,
        "an SU(2) index-0 minimum was certified as an index-1 saddle"
    );
    assert_eq!(
        info.index_estimate,
        Some(0),
        "the SU(2) refusal must carry the index estimate, not just a bare false"
    );
    assert!(!info.converged);
    let e = info
        .eig1
        .expect("the certificate ran, so its probe-1 eigenvalue must be reported");
    assert!(
        e > 0.0,
        "SU(2) eig1 {e} at a minimum must be positive (it is the singlet gap)"
    );
    let lam0 = qntt::su2::hamiltonian::rayleigh_quotient(&ham, &xgs);
    let lame = qntt::su2::hamiltonian::rayleigh_quotient(&ham, &xe);
    assert!(
        (lam0 - lame).abs() <= 1e-12 * lam0.abs().max(1.0),
        "a refused SU(2) endpoint must still be the point it was handed ({lame} vs {lam0})"
    );
}

#[test]
fn gad_par_su2_fixed_seed_is_deterministic() {
    let (ham, _xgs, x0) = su2_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 60,
        k_davidson: 8,
        seed: 0x6AD5EED,
        certify: true,
        probe_budget: 150,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xa, ia) = qntt::su2::hess::gad(&ham, &x0, None, &opts);
    let (xb, ib) = qntt::su2::hess::gad(&ham, &x0, None, &opts);
    assert!(ia.steps > 0, "vacuous SU(2) determinism arm");
    assert_eq!(ia.steps, ib.steps, "SU(2) step count is not reproducible");
    assert_eq!(
        ia.rejected_steps, ib.rejected_steps,
        "SU(2) rejection count is not reproducible"
    );
    assert_eq!(
        ia.applies, ib.applies,
        "SU(2) apply count is not reproducible"
    );
    assert_eq!(ia.index1_certified, ib.index1_certified);
    assert!((ia.lambda - ib.lambda).abs() <= 1e-13);
    assert!(
        (ia.gnorm - ib.gnorm).abs() <= 1e-13,
        "SU(2) gnorm {:.17e} vs {:.17e}",
        ia.gnorm,
        ib.gnorm
    );
    let d = &(&xa - &xb);
    assert!(
        d.norm() <= 1e-12,
        "SU(2) endpoint states differ by {:.3e}",
        d.norm()
    );
}

#[test]
fn gad_par_su2_generic_kick_neither_stalls_nor_crawls() {
    let (ham, xgs, _x0) = su2_full_rank_arm();
    let x0 = su2_generic_kick_from(&xgs, 5e-2, 0xA1);
    let opts = GadOpts {
        gradtol: 1e-8,
        max_steps: 300,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = qntt::su2::hess::gad(&ham, &x0, None, &opts);
    xe.check();

    assert!(
        !(info.steps < 10 && info.gnorm > 1e-3),
        "SU(2) step controller went stationary after {} accepted steps ({} rejections) at ‖grad‖ \
         = {:.3e}",
        info.steps,
        info.rejected_steps,
        info.gnorm
    );
    let streak = longest_frozen_streak(&info.gnorms, 1e-6);
    assert!(
        streak <= 20,
        "SU(2) merit frozen over {streak} consecutive accepted steps — the noise-band crawl on \
         this layer (‖grad‖ = {:.3e} after {} steps)",
        info.gnorm,
        info.steps
    );
    assert!(
        info.converged || info.steps >= opts.max_steps,
        "SU(2) run ended neither converged nor at max_steps: steps {}, rejections {}, ‖grad‖ \
         {:.3e}",
        info.steps,
        info.rejected_steps,
        info.gnorm
    );
    assert_eq!(info.gnorms.len(), info.steps + 1);
    assert!(info.radial_overlap_max <= 1e-8);
}

#[test]
fn gad_par_su2_trace_is_self_consistent() {
    let (ham, _xgs, x0) = su2_truncated_arm();
    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 40,
        k_davidson: 8,
        w_model: WModel::Damped(WDampRule::GradTracked),
        certify: false,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = qntt::su2::hess::gad(&ham, &x0, None, &opts);
    xe.check();
    assert!(info.steps > 0, "vacuous SU(2) telemetry arm");

    let n = info.trace.len();
    assert!(
        n > 0 && n + 1 >= info.gnorms.len() && n <= info.gnorms.len(),
        "SU(2) trace has {n} records against {} accepted points",
        info.gnorms.len()
    );
    let mut b_gain_max = 0.0f64;
    for (k, r) in info.trace.iter().enumerate() {
        assert!(
            (r.gnorm - info.gnorms[k]).abs() <= 1e-13 * info.gnorms[k].abs().max(1.0),
            "SU(2) trace[{k}].gnorm {} vs gnorms[{k}] {}",
            r.gnorm,
            info.gnorms[k]
        );
        assert!(r.lambda.is_finite() && r.merit_slope.is_finite());
        assert!(
            r.min_schmidt.is_finite() && r.min_schmidt > 0.0,
            "SU(2) trace[{k}].min_schmidt = {} — the centerweighted tripwire is not being read",
            r.min_schmidt
        );
        assert!(
            r.theta_low <= r.theta_chosen + 1e-10,
            "SU(2) trace[{k}]: chosen Ritz value {} below the lowest {}",
            r.theta_chosen,
            r.theta_low
        );
        let s = r
            .damp_s
            .expect("WModel::Damped must record its Tikhonov scale on this layer too");
        assert!(
            (W_DAMP_S_MIN..=W_DAMP_S_MAX).contains(&s),
            "SU(2) damping scale {s:.3e} left the band [{W_DAMP_S_MIN:.0e}, {W_DAMP_S_MAX:.0e}]"
        );
        assert!(
            (s - info.damp_s[k]).abs() <= 1e-15,
            "SU(2) trace[{k}].damp_s {s} vs damp_s[{k}] {}",
            info.damp_s[k]
        );
        if r.b_gain.is_finite() {
            b_gain_max = b_gain_max.max(r.b_gain);
        }
    }
    assert!(
        b_gain_max > 0.0,
        "the centerweighted Weingarten term is dead on this truncated SU(2) fixture (max b_gain \
         over the trace = 0)"
    );
    assert!(
        (b_gain_max - info.b_gain_max).abs() <= 1e-12 * info.b_gain_max.max(1.0),
        "SU(2) max over trace b_gain ({b_gain_max}) vs b_gain_max ({})",
        info.b_gain_max
    );
    let flips = info
        .trace
        .windows(2)
        .filter(|w| w[0].near != w[1].near)
        .count();
    assert_eq!(info.regime_switches, flips);
}
