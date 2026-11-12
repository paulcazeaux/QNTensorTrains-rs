// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::hamiltonian::SparseHamiltonian;
use crate::round::RoundMode;
use crate::solvers::layer::TtState;
use crate::vector::TtVector;

use super::inner::{PinvitInfo, PinvitOpts, pinvit_ground_state};
use super::precond::ExpSumPrecond;
use super::{PinvitTermination, bounds};

/// Outer-iteration options.
#[derive(Clone, Copy, Debug)]
pub struct OpinvitOpts {
    /// Base inner-iteration options; the inner `tau` is overwritten with `τ_m` each outer step.
    pub inner: PinvitOpts,
    /// Initial outer truncation budget `η_outer⁰` (halved each outer step).
    pub eta_outer: f64,
    /// Quasi-optimality slack `α > 0` (paper Algorithm 3); enters `κ`, `θ`, `τ_m`.
    pub alpha: f64,
    /// Outer termination target on the relative eigenvalue-error bound (paper eq 5.18).
    pub tau_outer: f64,
    /// Outer-iteration cap.
    pub max_outer: usize,
}

impl Default for OpinvitOpts {
    fn default() -> Self {
        OpinvitOpts {
            inner: PinvitOpts::default(),
            eta_outer: 1e-1,
            alpha: 1.0,
            tau_outer: 1e-6,
            max_outer: 30,
        }
    }
}

/// Outer-iteration diagnostics.
#[derive(Clone, Debug, Default)]
pub struct OpinvitInfo {
    /// Number of outer iterations performed.
    pub outer_iters: usize,
    /// Rayleigh quotient at the end of each outer step.
    pub lambdas: Vec<f64>,
    /// The inner diagnostics of each outer step.
    pub inner_infos: Vec<PinvitInfo>,
    /// Max total bond rank of the (post-inner) iterate per outer step.
    pub max_ranks: Vec<usize>,
    /// Whether the outer eigenvalue bound reached `tau_outer`.
    pub converged: bool,
}

/// Outer PINVIT (Algorithm 3). Returns the shifted electronic eigenvalue `λ(H_γ)`, the
/// preconditioned-coordinate iterate (the Hamiltonian eigenvector is
/// [`ExpSumPrecond::hamiltonian_vector`] of it), and per-outer-step diagnostics.
pub fn opinvit_ground_state(
    h: &SparseHamiltonian<f64>,
    precond: &ExpSumPrecond,
    y0: &TtVector<f64>,
    opts: &OpinvitOpts,
) -> (f64, TtVector<f64>, OpinvitInfo) {
    let c = opts.inner.consts.resolve_c();
    let delta = opts.inner.consts.delta;
    let alpha = opts.alpha;
    let k = y0.num_sites() as f64;
    let kappa = ((1.0 + c) / (1.0 - c)).sqrt() * (k - 1.0).max(0.0).sqrt();
    let denom = 2.0 * (1.0 + (1.0 + alpha) * kappa);
    let theta = (1.0 + alpha) * kappa / denom;

    let mut eta = opts.eta_outer;
    let mut y = y0.clone();
    let mut info = OpinvitInfo::default();
    let mut last_lambda = f64::NAN;

    for m in 0..opts.max_outer {
        let mut inner_opts = opts.inner;
        inner_opts.tau = eta / denom;
        inner_opts.termination = PinvitTermination::Eigenvector;
        let (lambda, x_pre, inner_info) = pinvit_ground_state(h, precond, &y, &inner_opts);
        last_lambda = lambda;
        info.outer_iters = m + 1;
        info.lambdas.push(lambda);
        info.max_ranks.push(x_pre.max_total_bond_rank());

        let rho = inner_info.rhos.last().copied().unwrap_or(f64::INFINITY);
        info.inner_infos.push(inner_info);

        if opts.inner.verbose {
            eprintln!(
                "opinvit m={m} λ={lambda:.10} η_outer={eta:.3e} rank={} ebound={:.3e}",
                x_pre.max_total_bond_rank(),
                bounds::eigenvalue(rho, delta)
            );
        }

        if bounds::eigenvalue(rho, delta) <= opts.tau_outer {
            info.converged = true;
            y = x_pre;
            break;
        }

        y = x_pre.round(theta * eta, opts.inner.chi, RoundMode::Abs);
        eta *= 0.5;
    }

    (last_lambda, y, info)
}
