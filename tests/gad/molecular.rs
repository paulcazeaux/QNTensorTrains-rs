// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::gad::{GadOpts, WModel};

use crate::util::su2_truncated_arm;

#[test]
fn gad_default_wmodel_is_switch_and_reaches_the_e1_saddle_on_su2_trunc() {
    assert_eq!(
        GadOpts::default().w_model,
        WModel::Switch { b_switch_tol: 1e-2 },
        "the shipped GAD Hessian model changed. This arm pins the 2026-08-25 decision (36-run \
         sweep); if the default is being \
         moved on purpose, re-measure this cell and gad_su2_d4_truncated_converges together"
    );

    let (sham, _sx, sx0) = su2_truncated_arm();
    let opts = GadOpts {
        preserve_grid: true,
        ..GadOpts::default()
    };
    let t0 = std::time::Instant::now();
    let (xe, info) = qntt::su2::hess::gad(&sham, &sx0, None, &opts);
    let wall = t0.elapsed().as_secs_f64();
    xe.check();
    println!(
        "GAD DEFAULT WMODEL su2 d4 trunc: converged={} index={:?} steps={} rejected={} \
         applies={} gnorm={:.3e} lambda={:.9} eig1={:?} b_gain_max={:.3e} wall_s={wall:.1}",
        info.converged,
        info.index_estimate,
        info.steps,
        info.rejected_steps,
        info.applies,
        info.gnorm,
        info.lambda,
        info.eig1,
        info.b_gain_max
    );

    assert!(
        info.converged,
        "the default model must converge this cell (‖grad‖ = {:.3e} after {} steps)",
        info.gnorm, info.steps
    );
    assert_eq!(
        info.index_estimate,
        Some(1),
        "the endpoint must certify index 1 on the undamped A + B"
    );
    assert!(
        (info.lambda + 2.360583997).abs() < 1e-6,
        "the default model must reach the E₁-like saddle λ = −2.360583997, not the B-dominated \
         −3.566758128 that WModel::Full lands at: λ = {:.9}",
        info.lambda
    );
    let eig1 = info.eig1.expect("certificate ran");
    assert!(
        (eig1 + 1.635).abs() < 1e-2,
        "ascent eigenvalue drifted from the measured −1.635 (WModel::Full reads −155.9 here): \
         eig1 = {eig1:.6e}"
    );
    assert!(
        info.steps < 200,
        "the default model reaches this saddle in a measured 59 accepted steps at these opts; \
         {} is a trajectory change, not noise",
        info.steps
    );
}
