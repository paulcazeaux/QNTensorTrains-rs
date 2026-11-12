// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::joint::{JointScfOpts, RampLevel, RampOpts, joint_ground_state_ramped};

use crate::optimize::{capped_randn, d2_fixture};

#[test]
#[should_panic(expected = "the ladder must have at least one level")]
fn tjrp_empty_ladder_refused() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 58);
    let opts = RampOpts {
        levels: Vec::new(),
        ..RampOpts::default()
    };
    let _ = joint_ground_state_ramped(&fx.ints, fx.space, &x0, &opts);
}

#[test]
#[should_panic(expected = "grow_eps must be positive")]
fn tjrp_zero_grow_eps_refused() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 58);
    let opts = RampOpts {
        levels: vec![RampLevel::rank(1)],
        grow_eps: 0.0,
        ..RampOpts::default()
    };
    let _ = joint_ground_state_ramped(&fx.ints, fx.space, &x0, &opts);
}

#[test]
#[should_panic(expected = "tt_preserve_grid is incompatible")]
fn tjrp_preserve_grid_refused() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 58);
    let opts = RampOpts {
        levels: vec![RampLevel::rank(1)],
        base: JointScfOpts {
            tt_preserve_grid: true,
            ..JointScfOpts::default()
        },
        ..RampOpts::default()
    };
    let _ = joint_ground_state_ramped(&fx.ints, fx.space, &x0, &opts);
}

#[test]
fn tjrp_single_level_rank1_certifies() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 59);
    let opts = RampOpts {
        levels: vec![RampLevel {
            rank: 1,
            orb_grad: Some(1e-4),
            tt_grad_tol: Some(1e-5),
            orb_max_iter: Some(20),
        }],
        base: JointScfOpts {
            construction: qntt::hamiltonian::assembly::Construction::Median,
            ..JointScfOpts::default()
        },
        ..RampOpts::default()
    };
    let (out, rinfo) = joint_ground_state_ramped(&fx.ints, fx.space, &x0, &opts);
    assert!(
        out.info.converged,
        "the single rung must certify at its own gates: {:?}",
        out.info.refused_reason
    );
    assert_eq!(rinfo.per_level.len(), 1);
    assert_eq!(rinfo.level_lambdas.len(), 1);
    assert!(
        (rinfo.level_lambdas[0] - out.lambda).abs() <= 1e-12 * out.lambda.abs().max(1.0),
        "the record's last λ must be the returned λ"
    );
    assert!(rinfo.basin_warnings.iter().all(|w| !w));
    assert!(
        out.lambda >= fx.e_fci - 1e-10,
        "variational floor: {:.12e} vs FCI {:.12e}",
        out.lambda,
        fx.e_fci
    );
}
