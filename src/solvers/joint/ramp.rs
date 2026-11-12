// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::integrals::Integrals;
use crate::qn::QnSpace;
use crate::rotation::rotate_integrals;
use crate::solvers::joint::{JointScfInfo, JointScfOpts, JointScfOut, joint_ground_state};
use crate::vector::TtVector;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// One rung of the ladder.
#[derive(Clone, Debug)]
pub struct RampLevel {
    /// Uniform interior bond-rank target (reachability-capped downstream, the
    /// `perturbation_uniform` convention).
    pub rank: usize,
    /// Level override of [`JointScfOpts::orb_grad`]; `None` ⇒ the base value.
    pub orb_grad: Option<f64>,
    /// Level override of [`JointScfOpts::tt_grad_tol`]; `None` ⇒ the base value.
    pub tt_grad_tol: Option<f64>,
    /// Level override of [`JointScfOpts::orb_max_iter`]; `None` ⇒ the base value.
    pub orb_max_iter: Option<usize>,
}

impl RampLevel {
    /// A rung with no overrides.
    pub fn rank(rank: usize) -> Self {
        RampLevel {
            rank,
            orb_grad: None,
            tt_grad_tol: None,
            orb_max_iter: None,
        }
    }
}

/// Options of [`joint_ground_state_ramped`].
#[derive(Clone, Debug)]
pub struct RampOpts {
    /// The ladder, low rank first. The LAST level's gates are the certification gates.
    pub levels: Vec<RampLevel>,
    /// The base options every level inherits (and the last level runs verbatim unless
    /// overridden). `tt_preserve_grid` must be `false` — the level transitions rely on the
    /// leg's entry round collapsing over-declared sectors.
    pub base: JointScfOpts,
    /// Rank-growth perturbation amplitude (relative). MUST be positive — see the module header.
    /// Default 1e-3.
    pub grow_eps: f64,
    /// RNG seed of the growth perturbations (recorded; the protocol is deterministic given it).
    pub seed: u64,
}

impl Default for RampOpts {
    fn default() -> Self {
        RampOpts {
            levels: Vec::new(),
            base: JointScfOpts::default(),
            grow_eps: 1e-3,
            seed: 0x52414D50,
        }
    }
}

/// Per-run record of [`joint_ground_state_ramped`].
#[derive(Clone, Debug)]
pub struct RampInfo {
    /// Each level's full solver diagnostics, in ladder order.
    pub per_level: Vec<JointScfInfo>,
    /// Each level's converged reduced energy (in ITS OWN rotated basis — energies are
    /// basis-invariant, so cross-level comparison is legal; states are not).
    pub level_lambdas: Vec<f64>,
    /// `true` where a level's energy rose above the previous level's by more than the energy
    /// floor — a basin-change WARNING (recorded, never a gate: a rank increase must not lose
    /// energy beyond noise, and a violation means the growth perturbation kicked the state
    /// somewhere new).
    pub basin_warnings: Vec<bool>,
}

/// Run the ladder. Returns the LAST level's output (its `u` is the TOTAL accumulated rotation
/// from the caller's original basis) and the per-level record.
///
/// # Panics
///
/// On an empty ladder, a non-positive `grow_eps`, or `tt_preserve_grid = true` (a pinned
/// manifold has a DECLARED grid; ramping it makes no sense and the entry round would fight the
/// pin).
pub fn joint_ground_state_ramped(
    ints: &Integrals<f64>,
    space: QnSpace,
    x0: &TtVector<f64>,
    opts: &RampOpts,
) -> (JointScfOut, RampInfo) {
    assert!(
        !opts.levels.is_empty(),
        "joint_ground_state_ramped: the ladder must have at least one level"
    );
    assert!(
        opts.grow_eps > 0.0,
        "joint_ground_state_ramped: grow_eps must be positive — exact zero-padding leaves \
         sigma_min = 0 sectors (the 1/sigma_min curvature hazard)"
    );
    assert!(
        !opts.base.tt_preserve_grid,
        "joint_ground_state_ramped: tt_preserve_grid is incompatible with rank continuation"
    );
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(opts.seed);
    let mut u_total: Vec<f64> = (0..d * d)
        .map(|i| if i % (d + 1) == 0 { 1.0 } else { 0.0 })
        .collect();
    let mut x_prev = x0.clone();
    let mut info = RampInfo {
        per_level: Vec::with_capacity(opts.levels.len()),
        level_lambdas: Vec::with_capacity(opts.levels.len()),
        basin_warnings: Vec::with_capacity(opts.levels.len()),
    };
    let mut out_last: Option<JointScfOut> = None;

    for level in opts.levels.iter() {
        let x_grown =
            crate::vector::perturbation_uniform(&x_prev, level.rank, opts.grow_eps, &mut rng);
        let ints_k = rotate_integrals(ints, &u_total);
        let mut level_opts = opts.base.clone();
        if let Some(g) = level.orb_grad {
            level_opts.orb_grad = g;
        }
        if let Some(t) = level.tt_grad_tol {
            level_opts.tt_grad_tol = t;
        }
        if let Some(m) = level.orb_max_iter {
            level_opts.orb_max_iter = m;
        }
        let out = joint_ground_state(&ints_k, space, &x_grown, &level_opts);

        let warn = match info.level_lambdas.last() {
            Some(&prev) => out.lambda > prev + 32.0 * f64::EPSILON * (1.0 + prev.abs()),
            None => false,
        };
        info.basin_warnings.push(warn);
        info.level_lambdas.push(out.lambda);
        info.per_level.push(out.info.clone());

        let composed = crate::rotation::compose_rotations(&u_total, &out.u, d);
        u_total = composed;
        x_prev = out.x.clone();
        out_last = Some(out);
    }

    let mut out = out_last.expect("ladder non-empty");
    out.u = u_total;
    (out, info)
}
