// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::joint::{JointScfOpts, JointSolver};

use crate::optimize::d2_fixture;

#[test]
fn tjtr_stagnation_exit() {
    let fx = d2_fixture();
    let ints_conv = qntt::rotate_integrals(&fx.ints, &fx.out.u);
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        orb_max_iter: 12,
        orb_grad: 1e-13,
        orb_grad_full: Some(1e-13),
        tt_grad_tol: 1e-6,
        construction: qntt::hamiltonian::assembly::Construction::Median,
        ..JointScfOpts::default()
    };
    let out = qntt::solvers::joint::joint_ground_state(&ints_conv, fx.space, &fx.out.x, &opts);
    assert!(
        out.info.converged && out.info.stalled_at_floor,
        "expected the graceful floor exit (converged + stalled_at_floor), got converged = {}, \
         stalled = {}, reason = {:?}",
        out.info.converged,
        out.info.stalled_at_floor,
        out.info.refused_reason
    );
    assert!(
        out.info.iters < opts.orb_max_iter,
        "the floor exit must fire before the iteration cap: {} vs {}",
        out.info.iters,
        opts.orb_max_iter
    );
    let e0 = out.info.e_history.first().copied().unwrap_or(f64::NAN);
    let efloor = 32.0 * f64::EPSILON * (1.0 + e0.abs());
    assert!(
        (out.lambda - fx.out.lambda).abs() <= 100.0 * efloor,
        "the stalled endpoint must still sit at the certified energy: {:.12e} vs {:.12e}",
        out.lambda,
        fx.out.lambda
    );
}
