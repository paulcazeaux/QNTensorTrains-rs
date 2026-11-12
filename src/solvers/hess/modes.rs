// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::factor::dense_self_adjoint_smallest;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;

use super::PdVerdict;

pub(crate) const PROBE_TOL_REL: f64 = 1e-4;

pub(crate) const PROBE_NEG_MARGIN: f64 = 1e-6;

pub(crate) const PROBE_WARMUP: usize = 8;

/// Knobs of [`lowest_hessian_mode`].
#[derive(Clone, Copy, Debug)]
pub struct ModeOpts {
    /// Rayleigh–Ritz iterations. Each costs one `op` apply (the correction vector's image);
    /// total applies are `budget + 2` at most (the seed's image and the closing fresh apply).
    pub budget: usize,
    /// How many of the lowest Ritz pairs to return. Clamped to the basis size; only the LOWEST
    /// pair drives the convergence test and the witness certificate.
    pub n_ritz: usize,
    /// Relative residual tolerance: convergence is `‖r‖ ≤ tol_rel · max(scale, |θ|)`.
    pub tol_rel: f64,
    /// Unpreconditioned correction steps before `precond` engages ([`PROBE_WARMUP`] for a cold
    /// start; small for a warm one, where the seed already sits on the tracked branch).
    pub warmup: usize,
    /// Break out of the iteration on two consecutive Ritz values below `−1e-6·scale`. Right for a
    /// PD probe (the fresh-apply witness is already a proof); WRONG for ascent-mode tracking,
    /// where a negative lowest eigenvalue is the steady state.
    pub neg_early_exit: bool,
    /// Dimension of the space `op` acts on, AFTER deflation (`tangent_dim − 1` on the real
    /// layers, `2·tangent_dim − 2` over `C64` where the radial span is `{x, ix}`). Reaching it is
    /// exact exhaustion: the Ritz value is then λ_min, so the run counts as converged.
    pub space_dim: usize,
    /// Operator-scale proxy for the tolerances — [`crate::solvers::hess::diag_inf_scale`] of the
    /// preconditioner diagonal.
    pub scale: f64,
}

impl Default for ModeOpts {
    fn default() -> Self {
        ModeOpts {
            budget: 40,
            n_ritz: 1,
            tol_rel: PROBE_TOL_REL,
            warmup: PROBE_WARMUP,
            neg_early_exit: true,
            space_dim: usize::MAX,
            scale: 1.0,
        }
    }
}

/// One Rayleigh–Ritz pair of [`lowest_hessian_mode`]: the Ritz value, the (cleaned) Ritz vector,
/// and the true residual norm `‖op(v) − θ·v‖` measured on the Krylov images — free, no extra
/// apply. `residual` bounds the distance from `theta` to SOME eigenvalue of the deflated
/// operator, never that it is the smallest: only the lowest pair's residual, read together with
/// [`ModeResult::converged`], carries that claim.
#[derive(Clone, Debug)]
pub struct RitzPair<V> {
    /// Rayleigh–Ritz value.
    pub theta: f64,
    /// Ritz vector, cleaned through the caller's `clean` hook (tangent, radially deflated).
    pub vector: V,
    /// `‖op(v) − θ·v‖` at the recombination, before the closing clean.
    pub residual: f64,
}

/// Outcome of [`lowest_hessian_mode`].
#[derive(Clone, Debug)]
pub struct ModeResult<V> {
    /// The three-way PD verdict on the LOWEST pair, from the closing fresh apply.
    /// `WitnessNegative` is a proof (a variational upper bound on λ_min), `LikelyPd` is evidence,
    /// `Inconclusive` is neither — see [`PdVerdict`].
    pub verdict: PdVerdict,
    /// The vector the verdict was certified on: the cleaned lowest Ritz vector. Equal to
    /// `ritz[0].vector`.
    pub witness: V,
    /// Rayleigh quotient of the FRESH apply on [`Self::witness`] — the number inside
    /// [`Self::verdict`], surfaced so a caller need not match on the variant to read it.
    pub lambda_fresh: f64,
    /// The `n_ritz` lowest Ritz pairs, ascending in `theta`.
    pub ritz: Vec<RitzPair<V>>,
    /// Total `op` applies consumed (seed image + corrections + the closing fresh apply).
    pub applies: usize,
    /// The lowest pair's residual met `tol_rel`, or the basis exhausted `space_dim` exactly.
    /// **Not** implied by a small `ritz[0].residual` on its own.
    pub converged: bool,
}

/// Lowest eigenpair(s) of a deflated, ℝ-linear self-adjoint operator, with a FRESH-apply Rayleigh
/// quotient of the returned Ritz vector as the certificate — the shared kernel behind the
/// complex PD probes and the saddle-dynamics V-tracker.
pub fn lowest_hessian_mode<V, Op, Pr, Cl>(
    mut op: Op,
    mut precond: Pr,
    mut clean: Cl,
    v0: &V,
    opts: &ModeOpts,
) -> Option<ModeResult<V>>
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
    Pr: FnMut(&mut V, f64),
    Cl: FnMut(&mut V),
{
    let cap = opts.space_dim.max(1);
    let mut applies = 0usize;

    fn push_ortho<V: InnerProductVector, Op: FnMut(&V) -> V>(
        mut u: V,
        vs: &mut Vec<V>,
        hs: &mut Vec<V>,
        op: &mut Op,
        applies: &mut usize,
    ) -> bool {
        let n0 = u.norm();
        for _ in 0..2 {
            for v in vs.iter() {
                let c = v.dot(&u).re_s();
                u.axpy(V::Scalar::from_f64(-c), v);
            }
        }
        let n = u.norm();
        if n <= 1e-8 * n0.max(1e-300) || n == 0.0 {
            return false;
        }
        u.scale(V::Scalar::from_f64(1.0 / n));
        let hu = op(&u);
        *applies += 1;
        vs.push(u);
        hs.push(hu);
        true
    }

    let mut vs: Vec<V> = Vec::new();
    let mut hs: Vec<V> = Vec::new();
    let mut seed = v0.clone();
    clean(&mut seed);
    if !push_ortho(seed, &mut vs, &mut hs, &mut op, &mut applies) {
        return None;
    }

    let mut ritz: Vec<RitzPair<V>> = Vec::new();
    let mut converged = false;
    let mut neg_streak = 0usize;
    let trace = crate::knobs::PD_TRACE.get();
    for it in 0..opts.budget.max(1) {
        let m = vs.len();
        let mut t = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..m {
                t[i * m + j] = vs[i].dot(&hs[j]).re_s();
            }
        }
        for i in 0..m {
            for j in (i + 1)..m {
                let s = 0.5 * (t[i * m + j] + t[j * m + i]);
                t[i * m + j] = s;
                t[j * m + i] = s;
            }
        }
        let want = opts.n_ritz.max(1).min(m);
        let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&t, m, want);
        let mut r0: Option<V> = None;
        ritz = Vec::with_capacity(want);
        for (c, y) in evecs.iter().enumerate().take(want) {
            let theta = evals[c];
            let mut xr = vs[0].clone();
            xr.scale(V::Scalar::from_f64(y[0]));
            let mut hx = hs[0].clone();
            hx.scale(V::Scalar::from_f64(y[0]));
            for i in 1..m {
                xr.axpy(V::Scalar::from_f64(y[i]), &vs[i]);
                hx.axpy(V::Scalar::from_f64(y[i]), &hs[i]);
            }
            let mut r = hx;
            r.axpy(V::Scalar::from_f64(-theta), &xr);
            let residual = r.norm();
            if c == 0 {
                r0 = Some(r);
            }
            ritz.push(RitzPair {
                theta,
                vector: xr,
                residual,
            });
        }
        let theta = ritz[0].theta;
        let normres = ritz[0].residual;
        if trace {
            println!("      lowest_hessian_mode[{it}]: m={m} θ={theta:.9e} ‖r‖={normres:.3e}");
        }
        if normres <= opts.tol_rel * opts.scale.max(theta.abs()) {
            converged = true;
            break;
        }
        if opts.neg_early_exit {
            if theta < -PROBE_NEG_MARGIN * opts.scale {
                neg_streak += 1;
                if neg_streak >= 2 {
                    break;
                }
            } else {
                neg_streak = 0;
            }
        }
        if m >= cap {
            converged = true;
            break;
        }
        let mut r = r0.expect("the lowest Ritz pair is always built");
        if it >= opts.warmup {
            precond(&mut r, theta);
        }
        clean(&mut r);
        if !push_ortho(r, &mut vs, &mut hs, &mut op, &mut applies) {
            break;
        }
    }

    for p in ritz.iter_mut() {
        clean(&mut p.vector);
    }
    let normres = ritz[0].residual;
    let x = ritz[0].vector.clone();
    let xn2 = x.norm().powi(2).max(1e-300);
    let hx = op(&x);
    applies += 1;
    let lambda_fresh = x.dot(&hx).re_s() / xn2;
    let verdict = if lambda_fresh < 0.0 {
        PdVerdict::WitnessNegative {
            lambda: lambda_fresh,
        }
    } else if converged {
        PdVerdict::LikelyPd {
            lambda_min: lambda_fresh,
            residual: normres,
        }
    } else {
        PdVerdict::Inconclusive {
            lambda_min: lambda_fresh,
            residual: normres,
        }
    };
    Some(ModeResult {
        verdict,
        witness: x,
        lambda_fresh,
        ritz,
        applies,
        converged,
    })
}
