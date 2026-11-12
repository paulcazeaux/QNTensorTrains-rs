// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) ALS driver — one-site alternating least squares.

use crate::scalar::Scalar;
pub use crate::solvers::als::{AlsOptions, INNER_TOL_CAP};
use crate::solvers::davidson::{DavidsonOpts, davidson, precond};
use crate::su2::core::Su2Core;
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::matvec::{Su2OneSiteWorkspace, su2_onesite_workspace};
use crate::su2::vector::Su2TtVector;

/// Rank-`rank` identity [`Su2Sketch`] at bond `k` of `x`'s space (`boundary_idsketch`, abelian twin
/// `crate::solvers::als::boundary_idsketch`, `src/solvers/als.rs`). Thin wrapper over
/// [`crate::su2::frame::Su2Sketch::identity`] — the SAME 0-based bond mapping as the abelian
/// sibling (`x,1` → `k=0`; `x,d+1` → `k=d`).
pub fn boundary_idsketch<T: Scalar>(x: &Su2TtVector<T>, k: usize, rank: usize) -> Su2Sketch<T> {
    Su2Sketch::identity(x.space, k, rank)
}

/// The right-to-left environment stack `Fᴿ[0..d-1]`. `Fᴿ[d-1]` is the bond-`d` boundary identity;
/// `Fᴿ[k]` for descending `k` is built from `Fᴿ[k+1]` via
/// [`Su2SparseHamiltonian::sketching_step_left`] (the θ²-weighted `rproject` path — that weight
/// lives entirely inside the shipped kernel, not here).
pub fn right_to_left_sketching<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Vec<Su2Sketch<T>> {
    let d = x.space.d;
    let mut fr: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    for k in (0..d - 1).rev() {
        let ws = su2_onesite_workspace(h, x.core(k + 1));
        let next = h.sketching_step_left(x, fr[k + 1].as_ref().unwrap(), &ws);
        fr[k] = Some(next);
    }
    fr.into_iter().map(|o| o.unwrap()).collect()
}

/// The left-to-right environment stack `Fᴸ[0..d-1]`. Not on the ALS gate path: MALS-forward and
/// TDVP each build their `Fᴸ` stack incrementally (`boundary_idsketch` + per-step
/// `sketching_step_right`) rather than calling this. On the Hessian-context gate path instead —
/// [`crate::su2::hess::Su2WeingartenCtx::envs`] calls it directly — and exercised by
/// `tests/tangent_hessian/su2_h7.rs` and `tests/matvec/su2_noise.rs`.
pub fn left_to_right_sketching<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Vec<Su2Sketch<T>> {
    let d = x.space.d;
    let mut fl: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    for k in 1..d {
        let ws = su2_onesite_workspace(h, x.core(k - 1));
        let next = h.sketching_step_right(x, fl[k - 1].as_ref().unwrap(), &ws);
        fl[k] = Some(next);
    }
    fl.into_iter().map(|o| o.unwrap()).collect()
}

/// Alternating-least-squares ground-state search on the SU(2) layer ( abelian twin
/// `crate::solvers::als::als`, `src/solvers/als.rs`). Right-orthogonalizes a clone of `x0`,
/// alternately sweeps forward/backward solving each one-site local eigenproblem via the SHARED
/// generic `crate::solvers::davidson::davidson`, until the Rayleigh quotient stabilizes to
/// `opts.eps`. Returns `(λ, x)` — `λ: T` per the abelian ALS return type (mirrors, unlike MALS's
/// `f64`). Does not mutate `x0` (see [`als_mut`] for the in-place form).
pub fn als<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &AlsOptions,
) -> (T, Su2TtVector<T>) {
    let mut x = x0.clone();
    let lambda = als_mut(h, &mut x, opts);
    (lambda, x)
}

/// In-place [`als`]: mutates `x` to the converged ground-state train, returns `λ`. The
/// inner-tolerance schedule is the total clamp chain `clamp(0.01r, ε, INNER_TOL_CAP)` — NOT
/// `f64::clamp`, which panics when `opts.eps > INNER_TOL_CAP`.
pub fn als_mut<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    opts: &AlsOptions,
) -> T {
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Su2Sketch<T>>> = right_to_left_sketching(h, x)
        .into_iter()
        .map(Some)
        .collect();
    let mut lambda: T = crate::su2::hamiltonian::rayleigh_quotient(h, x);
    let mut r = 1.0f64;

    for it in 0..opts.max_iter {
        let raw = 0.01 * r;
        let inner_tol = if raw > INNER_TOL_CAP {
            INNER_TOL_CAP
        } else if raw < opts.eps {
            opts.eps
        } else {
            raw
        };

        let (lf, mut fl) = als_forward_sweep(h, x, &mut fr, inner_tol);
        debug_assert!(
            fr.iter().all(Option::is_none),
            "L1: the forward sweep must consume the whole right stack"
        );
        if opts.verbose {
            println!("ALS iter {it} forward sweep: energy = {}", lf.re_s());
        }
        let (lambda_n, fr_new) = als_back_sweep(h, x, &mut fl, inner_tol);
        debug_assert!(
            fl.iter().all(Option::is_none),
            "L1: the back sweep must consume the whole left stack"
        );
        fr = fr_new;

        r = (lambda - lambda_n).abs_s() / lambda_n.abs_s();
        lambda = lambda_n;

        if opts.verbose {
            println!(
                "ALS iter {it} back sweep: energy = {}, relative improvement = {r}",
                lambda.re_s()
            );
        }

        if r < opts.eps && inner_tol <= opts.eps {
            break;
        }
    }
    lambda
}

fn local_ground<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    fl: &Su2Sketch<T>,
    fr: &Su2Sketch<T>,
    ws: &Su2OneSiteWorkspace,
    v0: &Su2Core<T>,
    inner_tol: f64,
) -> (f64, Su2Core<T>) {
    let d_diag = h.effective_diagonal(fl, fr, v0, ws);
    let mut op = |a: &Su2Core<T>| h.sketched_hamiltonian(a, fl, fr, ws);
    let mut applyprec = |t: &mut Su2Core<T>, theta: f64| precond(t, &d_diag, theta);
    let dopts = DavidsonOpts {
        tol: inner_tol,
        ..Default::default()
    };
    let (theta, v, _info) = davidson(&mut op, v0, &mut applyprec, &dopts);
    (theta, v)
}

fn als_forward_sweep<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fr: &mut [Option<Su2Sketch<T>>],
    inner_tol: f64,
) -> (T, Vec<Option<Su2Sketch<T>>>) {
    debug_assert_eq!(
        x.ortho,
        Some(0),
        "forward sweep expects a right-orthogonal x"
    );
    let d = x.space.d;
    let mut fl: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    let mut lambda = T::zero();

    for k in 0..d {
        let ws = su2_onesite_workspace(h, x.core(k));
        let (theta, vec) = {
            let fl_k = fl[k].as_ref().unwrap();
            local_ground(h, fl_k, fr[k].as_ref().unwrap(), &ws, x.core(k), inner_tol)
        };
        lambda = T::from_f64(theta);
        x.cores[k] = vec;
        fr[k].take();
        if k < d - 1 {
            x.move_core(k + 1, true);
            let next = h.sketching_step_right(x, fl[k].as_ref().unwrap(), &ws);
            fl[k + 1] = Some(next);
        }
    }
    (lambda, fl)
}

fn als_back_sweep<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fl: &mut [Option<Su2Sketch<T>>],
    inner_tol: f64,
) -> (T, Vec<Option<Su2Sketch<T>>>) {
    let d = x.space.d;
    debug_assert_eq!(
        x.ortho,
        Some(d - 1),
        "back sweep expects a left-orthogonal x"
    );
    let mut fr: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    let mut lambda = T::zero();

    for k in (0..d).rev() {
        let ws = su2_onesite_workspace(h, x.core(k));
        let (theta, vec) = {
            let fl_k = fl[k].as_ref().unwrap();
            local_ground(h, fl_k, fr[k].as_ref().unwrap(), &ws, x.core(k), inner_tol)
        };
        lambda = T::from_f64(theta);
        x.cores[k] = vec;
        fl[k].take();
        if k > 0 {
            x.move_core(k - 1, true);
            let next = h.sketching_step_left(x, fr[k].as_ref().unwrap(), &ws);
            fr[k - 1] = Some(next);
        }
    }
    (lambda, fr)
}

#[cfg(test)]
mod s5b_tests {
    use super::*;

    #[test]
    fn s5b_i1_als_options_type_identity() {
        let _pin: crate::solvers::als::AlsOptions = AlsOptions::default();
        assert_eq!(
            INNER_TOL_CAP,
            crate::solvers::als::INNER_TOL_CAP,
            "su2 solvers must import (not duplicate) INNER_TOL_CAP"
        );
    }
}
