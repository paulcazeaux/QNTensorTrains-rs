// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Lanczos ground-state solver. On numerical breakdown (`β·|γ_end| > 2·min(res)`) it
//! does NOT fall back to another solver — it only sets [`LanczosResult::breakdown`].

use crate::core::factor::dense_self_adjoint_smallest;
use crate::qn::Pooled;
use crate::round::RoundMode;
use crate::scalar::Scalar;
use crate::solvers::layer::{TtHamiltonian, TtState};

fn max_sector_rank_of<V: TtState>(tt: &V) -> usize {
    tt.max_sector_rank()
}

fn tridiag_smallest(dv: &[f64], ev: &[f64]) -> (f64, Vec<f64>) {
    let k = dv.len();
    let mut s = vec![0.0f64; k * k];
    for (i, &d) in dv.iter().enumerate() {
        s[i * k + i] = d;
    }
    for (i, &e) in ev.iter().enumerate() {
        s[i * k + (i + 1)] = e;
        s[(i + 1) * k + i] = e;
    }
    let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&s, k, 1);
    (evals[0], evecs[0].clone())
}

/// Lanczos driver options.
#[derive(Clone, Copy, Debug)]
pub struct LanczosOptions {
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for LanczosOptions {
    fn default() -> Self {
        LanczosOptions {
            tol: 1e-4,
            max_iter: 20,
        }
    }
}

/// Lanczos result. `lambdas`/`res`/`ranks`/`rounded_ranks` are per-iteration diagnostic vectors,
/// filled in by the driver and read by callers for reporting only.
#[derive(Clone, Debug)]
pub struct LanczosResult<V> {
    /// Smallest Ritz value at convergence/breakdown/`max_iter` (real; Hermitian `H`).
    pub lambda: f64,
    /// The (rounded, re-normalized) approximate ground state.
    pub x: V,
    /// `true` if the iteration hit numerical breakdown (`β·|γ_end| > 2·min(res)`) before
    /// converging. This module performs no fallback.
    pub breakdown: bool,
    pub lambdas: Vec<f64>,
    pub res: Vec<f64>,
    pub ranks: Vec<usize>,
    pub rounded_ranks: Vec<usize>,
}

/// Lanczos ground-state search on the un-preconditioned Krylov subspace `{x0, Hx0, H²x0, …}`, built
/// via 3-term recurrence with deterministic rounding at each step. Does not mutate `x0`.
pub fn lanczos<V: TtState, H: TtHamiltonian<V>>(
    h: &H,
    x0: &V,
    opts: &LanczosOptions,
) -> LanczosResult<V> {
    let tol = opts.tol;
    let max_iter = opts.max_iter;

    let mut q0 = x0.clone();
    q0.left_orthonormalize(false);
    let mut alpha = h.rayleigh_quotient(&q0).re_s();

    let mut qq: Vec<V> = vec![q0];
    let mut dv: Vec<f64> = vec![alpha];
    let mut ev: Vec<f64> = Vec::new();
    let mut res: Vec<f64> = Vec::new();
    let mut ranks: Vec<usize> = vec![max_sector_rank_of(&qq[0])];
    qq[0].round_mut(tol, Pooled::NONE, RoundMode::Abs);
    let mut rounded_ranks: Vec<usize> = vec![max_sector_rank_of(&qq[0])];
    let mut breakdown = false;

    let mut q = {
        let qlast = qq.last().unwrap();
        let hq = h.matvec(qlast);
        let sq = qlast.scaled(V::Scalar::from_f64(alpha));
        hq.sub(&sq)
    };
    ranks.push(max_sector_rank_of(&q));
    let mut nrm = q.left_orthonormalize(false);
    q.round_mut(tol, Pooled::NONE, RoundMode::Abs);
    rounded_ranks.push(max_sector_rank_of(&q));
    alpha = h.rayleigh_quotient(&q).re_s();
    let mut beta: f64 = nrm.iter().product();

    let (lam0, mut gamma) = tridiag_smallest(&dv, &ev);
    let mut lambdas: Vec<f64> = vec![lam0];

    if beta * gamma.last().unwrap().abs() > tol {
        for _ in 2..max_iter {
            qq.push(q.clone());
            dv.push(alpha);
            ev.push(beta);
            res.push(beta * gamma.last().unwrap().abs());

            let qtil = {
                let n = qq.len();
                let hq = h.matvec(&qq[n - 1]);
                let sq = qq[n - 1].scaled(V::Scalar::from_f64(alpha));
                let pq = qq[n - 2].scaled(V::Scalar::from_f64(beta));
                hq.sub(&sq).sub(&pq)
            };
            ranks.push(max_sector_rank_of(&qtil));
            q = qtil;
            nrm = q.left_orthonormalize(false);
            q.round_mut(tol, Pooled::NONE, RoundMode::Abs);
            rounded_ranks.push(max_sector_rank_of(&q));
            alpha = h.rayleigh_quotient(&q).re_s();
            beta = nrm.iter().product();

            let (lamk, g) = tridiag_smallest(&dv, &ev);
            lambdas.push(lamk);
            gamma = g;

            let crit = beta * gamma.last().unwrap().abs();
            if crit < tol {
                break;
            } else if crit > 2.0 * res.iter().cloned().fold(f64::INFINITY, f64::min) {
                breakdown = true;
                break;
            }
        }
    }

    let alpha_t: Vec<V::Scalar> = gamma.iter().map(|&g| V::Scalar::from_f64(g)).collect();
    let mut w = V::round_sum(&alpha_t, &qq, tol);
    w.left_orthonormalize(false);

    LanczosResult {
        lambda: *lambdas.last().unwrap(),
        x: w,
        breakdown,
        lambdas,
        res,
        ranks,
        rounded_ranks,
    }
}
