// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) state-specific MALS — the spin-adapted twin of [`crate::solvers::ssmals`], mirroring it
//! line for line with SU(2) type substitutions.

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::core::twosite::noise_rng;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;
use crate::solvers::davidson::precond;
use crate::solvers::mals::{MalsOptions, NoiseType, SweepTrunc};
use crate::solvers::multistate::{sa_maxsubspace, sa_restart_keep};
pub use crate::solvers::ssmals::{
    PenaltyRule, SsHistory, SsMode, SsSweep, StateSpecificOptions, penalty_weight,
};
use crate::solvers::ssmals::{SsLocalOut, delegate_sweeps, ss_local_core};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::matvec::{
    Su2TwoSiteWorkspace, su2_onesite_workspace, su2_twosite_workspace,
};
use crate::su2::hamiltonian::rayleigh_quotient;
use crate::su2::ops::dot;
use crate::su2::solvers::als::{INNER_TOL_CAP, boundary_idsketch, right_to_left_sketching};
use crate::su2::solvers::mals::{mals_run, split_two_site};
use crate::su2::solvers::overlap_env::{
    OverlapEnvs, local_overlap_vector, overlap_envs, overlap_step_left, overlap_step_right,
};
use crate::su2::twosite::{Su2ContractedCores, contract};
use crate::su2::vector::Su2TtVector;

use crate::solvers::ssmals::ROOTS_SALT;

/// Rank-adaptive two-site search for the lowest state ORTHOGONAL to (or penalized against) the
/// fixed `lower` states — the SU(2) twin of [`crate::solvers::ssmals::state_specific_mals`].
/// Returns `(λ, x)` with `λ` the PLAIN energy; does not mutate `x0`.
pub fn state_specific_mals<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    lower: &[Su2TtVector<T>],
    opts: &StateSpecificOptions,
) -> (f64, Su2TtVector<T>) {
    let (hist, x) = state_specific_mals_with_history(h, x0, lower, opts);
    (hist.lambda, x)
}

/// [`state_specific_mals`] keeping the per-iteration diagnostics.
pub fn state_specific_mals_with_history<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    lower: &[Su2TtVector<T>],
    opts: &StateSpecificOptions,
) -> (SsHistory, Su2TtVector<T>) {
    let mut x = x0.clone();
    let hist = state_specific_mals_mut(h, &mut x, lower, opts);
    (hist, x)
}

/// In-place [`state_specific_mals`]. `lower` is never mutated and need not be normalized: any
/// finite nonzero norm is accepted and `1/‖ψⱼ‖` is folded into the local images, the reference
/// energies and the recorded overlaps.
pub fn state_specific_mals_mut<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    lower: &[Su2TtVector<T>],
    opts: &StateSpecificOptions,
) -> SsHistory {
    assert!(
        !opts.mals.bond_dims.is_empty(),
        "MalsOptions::bond_dims must be non-empty (use [0] for uncapped)"
    );
    assert!(
        !opts.mals.noise.is_empty(),
        "MalsOptions::noise must be non-empty (use [0.0] for a noise-free run)"
    );
    assert!(
        opts.mals.noise.iter().all(|a| a.is_finite() && *a >= 0.0),
        "MalsOptions::noise entries must be finite and non-negative"
    );
    assert!(
        opts.roots >= 1,
        "state_specific_mals: roots = {} must be ≥ 1",
        opts.roots
    );
    if let PenaltyRule::Fixed(w) = opts.penalty {
        assert!(
            w.is_finite() && w >= 0.0,
            "state_specific_mals: PenaltyRule::Fixed({w}) must be finite and non-negative"
        );
    }
    let m = lower.len();
    let mut invn: Vec<f64> = Vec::with_capacity(m);
    for (j, psi) in lower.iter().enumerate() {
        assert_eq!(
            psi.space, x.space,
            "state_specific_mals: lower[{j}] lives in a different space than the iterate"
        );
        let n = psi.norm();
        assert!(
            n.is_finite() && n > 0.0,
            "state_specific_mals: lower[{j}] has norm {n} — a finite, NONZERO norm is required \
             (any other value is accepted and normalized internally, the state is never mutated)"
        );
        invn.push(if n == 1.0 { 1.0 } else { 1.0 / n });
    }
    if let Some(e) = &opts.lower_energies {
        assert_eq!(
            e.len(),
            m,
            "state_specific_mals: {} lower_energies for {m} lower state(s)",
            e.len()
        );
    }

    if m == 0 && opts.roots == 1 {
        return ss_delegate(h, x, &opts.mals);
    }

    let d = x.space.d;
    let ms = opts
        .maxsubspace
        .unwrap_or_else(|| sa_maxsubspace(opts.roots));
    let rk = opts
        .restart_keep
        .unwrap_or_else(|| sa_restart_keep(opts.roots, ms));
    let mut rng = noise_rng(opts.mals.seed);
    let mut rngr = noise_rng(opts.mals.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ROOTS_SALT);
    let mut sweeps: Vec<SsSweep> = Vec::with_capacity(opts.mals.max_iter);
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Su2Sketch<T>>> = right_to_left_sketching(h, x)
        .into_iter()
        .map(Some)
        .collect();
    let mut envs: Vec<OverlapEnvs<T>> = lower.iter().map(|psi| overlap_envs(x, psi)).collect();
    let e_lower: Vec<f64> = match &opts.lower_energies {
        Some(e) => e.clone(),
        None => (0..m)
            .map(|j| rayleigh_quotient(h, &lower[j]).re_s() * invn[j] * invn[j])
            .collect(),
    };

    let lambda_raw = rayleigh_quotient(h, x).re_s();
    let nx = x.norm();
    let mut lambda_x = lambda_raw / (nx * nx);
    let mut theta_pen = lambda_raw;
    let mut r = 1.0f64;
    let mut overlaps: Vec<f64> = (0..m)
        .map(|j| dot(&lower[j], x).abs_s() * invn[j] / nx)
        .collect();
    if opts.roots > 1 {
        let sigma: f64 = overlaps.iter().map(|o| o * o).sum();
        if sigma > 0.5 {
            eprintln!(
                "state_specific_mals: the start carries Σⱼ|⟨ψ̂ⱼ,x₀⟩|² = {sigma:.4} > 1/2 with \
                 roots = {} — the label the homing continues is a coin flip; a nonzero final \
                 `swaps` is the tell",
                opts.roots
            );
        }
    }
    let mut w_prev = 0.0f64;

    for it in 0..opts.mals.max_iter {
        let bond_dim = opts.mals.bond_dims[it.min(opts.mals.bond_dims.len() - 1)];
        let noise = opts.mals.noise[it.min(opts.mals.noise.len() - 1)];
        let raw = 0.01 * r;
        let inner_tol = if raw > INNER_TOL_CAP {
            INNER_TOL_CAP
        } else if raw < opts.mals.eps {
            opts.mals.eps
        } else {
            raw
        };
        let ep = if d > 1 {
            opts.mals.eps / ((d - 1) as f64).sqrt()
        } else {
            opts.mals.eps
        };
        let w = match (opts.mode, opts.penalty) {
            (SsMode::Project, _) => 0.0,
            (SsMode::Penalty, PenaltyRule::Auto) => {
                penalty_weight(lambda_x, &overlaps, &e_lower, w_prev)
            }
            (SsMode::Penalty, PenaltyRule::Fixed(v)) => v,
        };

        let mut fwd = ss_forward_sweep(
            h,
            x,
            &mut fr,
            &mut envs,
            lower,
            &invn,
            w,
            opts.mode,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.mals.noise_type,
            &mut rng,
            &mut rngr,
            opts.roots,
            ms,
            rk,
        );
        if opts.mals.verbose {
            println!(
                "SS-MALS iter {it} forward sweep: energy = {} (penalized {}), penalty = {w}",
                fwd.lambda, fwd.theta
            );
        }
        debug_assert!(
            fr.iter().all(Option::is_none),
            "the forward sweep must consume the whole right stack"
        );
        let back = ss_back_sweep(
            h,
            x,
            &mut fwd.env,
            &mut envs,
            lower,
            &invn,
            w,
            opts.mode,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.mals.noise_type,
            &mut rng,
            &mut rngr,
            opts.roots,
            ms,
            rk,
        );
        debug_assert!(
            fwd.env.iter().all(Option::is_none),
            "the back sweep must consume the whole left stack"
        );
        fr = back.env;

        let trunc = fwd.trunc.merge(back.trunc);
        for j in 0..m {
            overlaps[j] = dot(&lower[j], x).abs_s();
            if invn[j] != 1.0 {
                overlaps[j] *= invn[j];
            }
        }
        let sigma: f64 = overlaps.iter().map(|o| o * o).sum();
        let collapsed = opts.mode == SsMode::Penalty
            && matches!(opts.penalty, PenaltyRule::Auto)
            && sigma > 0.5;
        if collapsed {
            eprintln!(
                "state_specific_mals: iteration {it} collapsed onto the lower states \
                 (Σⱼ|⟨ψ̂ⱼ,x⟩|² = {sigma:.4} at penalty w = {w}, below the gap to the target); the \
                 rule raises the weight to ≥ {} next",
                10.0 * w
            );
        }
        r = (theta_pen - back.theta).abs() / back.theta.abs();
        let converged = r < opts.mals.convergence_tol()
            && inner_tol <= opts.mals.eps
            && noise == 0.0
            && !collapsed;
        theta_pen = back.theta;
        lambda_x = back.lambda;
        w_prev = w;
        sweeps.push(SsSweep {
            energy: back.lambda,
            penalized: back.theta,
            discarded_weight: trunc.max_discarded,
            max_bond_dim: trunc.max_kept,
            bond_dim,
            noise,
            overlaps: overlaps.clone(),
            penalty: w,
            swaps: fwd.swaps + back.swaps,
            numops: fwd.numops + back.numops,
            skipped: fwd.skipped + back.skipped,
            converged,
        });
        if opts.mals.verbose {
            println!(
                "SS-MALS iter {it} back sweep: energy = {} (penalized {}), relative improvement = \
                 {r}, discarded weight = {:.3e}, M = {}, noise = {noise:.1e}",
                back.lambda, back.theta, trunc.max_discarded, trunc.max_kept
            );
        }
        if converged {
            break;
        }
    }
    if sweeps.last().map(|s| !s.converged).unwrap_or(false) {
        eprintln!(
            "state_specific_mals: max_iter = {} reached without convergence (last relative change \
             {r}, penalty {}, overlaps {:?})",
            opts.mals.max_iter,
            w_prev,
            sweeps
                .last()
                .map(|s| s.overlaps.clone())
                .unwrap_or_default()
        );
    }
    SsHistory {
        lambda: lambda_x,
        sweeps,
    }
}

fn ss_delegate<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    opts: &MalsOptions,
) -> SsHistory {
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let lambda_raw = rayleigh_quotient(h, x).re_s();
    let hist = mals_run(h, x, opts);
    let sweeps = delegate_sweeps(&hist, lambda_raw, opts);
    if sweeps.last().map(|s| !s.converged).unwrap_or(false) {
        eprintln!(
            "state_specific_mals: max_iter = {} reached without convergence (mals delegate)",
            opts.max_iter
        );
    }
    SsHistory {
        lambda: hist.lambda,
        sweeps,
    }
}

struct SsSweepOut<T: Scalar> {
    theta: f64,
    lambda: f64,
    env: Vec<Option<Su2Sketch<T>>>,
    trunc: SweepTrunc,
    swaps: usize,
    numops: usize,
    skipped: usize,
}

fn ss_local_images<T: Scalar>(
    envs: &[OverlapEnvs<T>],
    lower: &[Su2TtVector<T>],
    invn: &[f64],
    k: usize,
) -> Vec<Su2ContractedCores<T>> {
    (0..lower.len())
        .map(|j| {
            let mut p = local_overlap_vector(&envs[j], &lower[j], k);
            if invn[j] != 1.0 {
                p.scale(T::from_f64(invn[j]));
            }
            p
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn ss_local_solve<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    fl: &Su2Sketch<T>,
    fr: &Su2Sketch<T>,
    cp0: &Su2ContractedCores<T>,
    ps: &[Su2ContractedCores<T>],
    w: f64,
    mode: SsMode,
    roots: usize,
    inner_tol: f64,
    maxsubspace: usize,
    restart_keep: usize,
    rngr: &mut Xoshiro256PlusPlus,
) -> SsLocalOut<Su2ContractedCores<T>> {
    let d_diag = h.two_site_effective_diagonal(fl, fr, cp0);
    let mut ws: Option<Su2TwoSiteWorkspace<T>> = su2_twosite_workspace();
    let mut op = |cp: &Su2ContractedCores<T>| {
        let mut y = h.two_site_sketched_hamiltonian_ws(&mut ws, cp, fl, fr);
        if w > 0.0 {
            let wt = T::from_f64(w);
            for p in ps.iter() {
                let c = p.dot(cp);
                y.axpy(wt * c, p);
            }
        }
        y
    };
    let mut applyprec = |t: &mut Su2ContractedCores<T>, theta: f64| precond(t, &d_diag, theta);
    ss_local_core(
        &mut op,
        &mut applyprec,
        cp0,
        ps,
        w,
        mode,
        roots,
        inner_tol,
        maxsubspace,
        restart_keep,
        rngr,
    )
}

#[allow(clippy::too_many_arguments)]
fn ss_forward_sweep<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fr: &mut [Option<Su2Sketch<T>>],
    envs: &mut [OverlapEnvs<T>],
    lower: &[Su2TtVector<T>],
    invn: &[f64],
    w: f64,
    mode: SsMode,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
    rngr: &mut Xoshiro256PlusPlus,
    roots: usize,
    maxsubspace: usize,
    restart_keep: usize,
) -> SsSweepOut<T> {
    debug_assert_eq!(x.ortho, Some(0), "SS-MALS forward sweep expects center 0");
    let d = x.space.d;
    let m = lower.len();
    let mut fl: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    fr[0] = None;
    let mut out = SsSweepOut {
        theta: 0.0,
        lambda: 0.0,
        env: Vec::new(),
        trunc: SweepTrunc::default(),
        swaps: 0,
        numops: 0,
        skipped: 0,
    };

    for k in 0..(d - 1) {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let ps = if m > 0 && (mode == SsMode::Project || w > 0.0) {
            ss_local_images(envs, lower, invn, k)
        } else {
            Vec::new()
        };
        let o = ss_local_solve(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &cp0,
            &ps,
            w,
            mode,
            roots,
            inner_tol,
            maxsubspace,
            restart_keep,
            rngr,
        );
        out.theta = o.theta;
        out.lambda = o.lambda;
        out.swaps += o.swap;
        out.numops += o.numops;
        out.skipped += o.skipped;
        let (xk, s, mut xk1, info) = split_two_site(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &o.u,
            ep,
            max_rank,
            noise,
            noise_type,
            true,
            rng,
        );
        fr[k + 1] = None;
        out.trunc.absorb(info);
        xk1.fold_sketch_left_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k + 1);
        if k < d - 2 {
            let ws = su2_onesite_workspace(h, x.core(k));
            let next = h.sketching_step_right(x, fl[k].as_ref().unwrap(), &ws);
            fl[k + 1] = Some(next);
            for j in 0..m {
                let prev = envs[j].l[k]
                    .as_ref()
                    .expect("l[k] built by the previous bond");
                let step = overlap_step_right(x, &lower[j], prev, k);
                envs[j].l[k + 1] = Some(step);
            }
        }
    }
    out.env = fl;
    out
}

#[allow(clippy::too_many_arguments)]
fn ss_back_sweep<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fl: &mut [Option<Su2Sketch<T>>],
    envs: &mut [OverlapEnvs<T>],
    lower: &[Su2TtVector<T>],
    invn: &[f64],
    w: f64,
    mode: SsMode,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
    rngr: &mut Xoshiro256PlusPlus,
    roots: usize,
    maxsubspace: usize,
    restart_keep: usize,
) -> SsSweepOut<T> {
    let d = x.space.d;
    let m = lower.len();
    if x.ortho != Some(d - 1) {
        x.move_core(d - 1, false);
    }
    let mut fr: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    fl[d - 1] = None;
    let mut out = SsSweepOut {
        theta: 0.0,
        lambda: 0.0,
        env: Vec::new(),
        trunc: SweepTrunc::default(),
        swaps: 0,
        numops: 0,
        skipped: 0,
    };

    for k in (0..(d - 1)).rev() {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let ps = if m > 0 && (mode == SsMode::Project || w > 0.0) {
            ss_local_images(envs, lower, invn, k)
        } else {
            Vec::new()
        };
        let o = ss_local_solve(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &cp0,
            &ps,
            w,
            mode,
            roots,
            inner_tol,
            maxsubspace,
            restart_keep,
            rngr,
        );
        out.theta = o.theta;
        out.lambda = o.lambda;
        out.swaps += o.swap;
        out.numops += o.numops;
        out.skipped += o.skipped;
        let (mut xk, s, xk1, info) = split_two_site(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &o.u,
            ep,
            max_rank,
            noise,
            noise_type,
            false,
            rng,
        );
        fl[k] = None;
        out.trunc.absorb(info);
        xk.fold_sketch_right_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k);
        if k > 0 {
            let ws = su2_onesite_workspace(h, x.core(k + 1));
            let next = h.sketching_step_left(x, fr[k + 1].as_ref().unwrap(), &ws);
            fr[k] = Some(next);
            for j in 0..m {
                let prev = envs[j].r[k + 2]
                    .as_ref()
                    .expect("r[k+2] built by the previous bond");
                let step = overlap_step_left(x, &lower[j], prev, k + 1);
                envs[j].r[k + 1] = Some(step);
            }
        }
    }
    out.env = fr;
    out
}
