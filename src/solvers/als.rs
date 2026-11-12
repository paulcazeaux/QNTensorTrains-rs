// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! ALS driver + environment-stack construction. The streaming one-site sketched-apply kernels
//! themselves live on `SparseHamiltonian` (`hamiltonian::matvec`); this module owns the outer
//! sweep driver, the environment stacks
//! ([`right_to_left_sketching`]/[`left_to_right_sketching`]), and the boundary `IdSketch`.

use crate::core::SparseCore;
use crate::core::factor::Sketch;
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::matvec::{OneSiteWorkspace, onesite_workspace};
use crate::scalar::Scalar;
use crate::solvers::davidson::{DavidsonOpts, davidson, precond};
use crate::solvers::interface::clamp_hi_first;
use crate::vector::TtVector;

/// Absolute cap of the inner Davidson tolerance schedule `clamp(0.01·r, ε, INNER_TOL_CAP)`. A
/// plain `const`: nothing mutates it at runtime.
pub const INNER_TOL_CAP: f64 = 1e-4;

/// ALS driver options.
#[derive(Clone, Copy, Debug)]
pub struct AlsOptions {
    pub eps: f64,
    pub max_iter: usize,
    pub verbose: bool,
}

impl Default for AlsOptions {
    fn default() -> Self {
        AlsOptions {
            eps: 1e-4,
            max_iter: 20,
            verbose: false,
        }
    }
}

/// Rank-`rank` identity `Sketch` at bond `k` of `x`'s space — the two ALS-sweep boundary
/// environments (`k ∈ {0, d}`, `rank = 1`). Thin wrapper over
/// [`crate::core::factor::Sketch::identity`].
pub fn boundary_idsketch<T: Scalar>(x: &TtVector<T>, k: usize, rank: usize) -> Sketch<T> {
    Sketch::identity(x.space, k, rank)
}

/// The right-to-left environment stack `Fᴿ[0..d-1]`: `Fᴿ[d-1]` is the bond-`d` boundary identity,
/// `Fᴿ[k]` built from `Fᴿ[k+1]` via `SketchingStepLeft` for `k` descending.
pub fn right_to_left_sketching<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
) -> Vec<Sketch<T>> {
    let d = x.space.d;
    let mut fr: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    for k in (0..d - 1).rev() {
        let ws = onesite_workspace(h, x.core(k + 1));
        let next = h.sketching_step_left(x, fr[k + 1].as_ref().unwrap(), &ws);
        fr[k] = Some(next);
    }
    fr.into_iter().map(|o| o.unwrap()).collect()
}

/// The left-to-right environment stack `Fᴸ[0..d-1]`. Not on the ALS path (MALS-forward/TDVP
/// only); also consumed by the fused `RayleighQuotient`/effective-diagonal invariant tests.
pub fn left_to_right_sketching<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
) -> Vec<Sketch<T>> {
    let d = x.space.d;
    let mut fl: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    for k in 1..d {
        let ws = onesite_workspace(h, x.core(k - 1));
        let next = h.sketching_step_right(x, fl[k - 1].as_ref().unwrap(), &ws);
        fl[k] = Some(next);
    }
    fl.into_iter().map(|o| o.unwrap()).collect()
}

/// Alternating-least-squares ground-state search: right-orthogonalize `x0`, alternately sweep
/// forward/backward solving each one-site local eigenproblem via Davidson, until the Rayleigh
/// quotient stabilizes to `opts.eps`. Returns `(λ, x)`; does not mutate `x0` (see [`als_mut`] for
/// the in-place form).
pub fn als<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &AlsOptions,
) -> (T, TtVector<T>) {
    let mut x = x0.clone();
    let lambda = als_mut(h, &mut x, opts);
    (lambda, x)
}

/// In-place [`als`]: mutates `x` to the converged ground-state train, returns `λ`.
pub fn als_mut<T: Scalar>(h: &SparseHamiltonian<T>, x: &mut TtVector<T>, opts: &AlsOptions) -> T {
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Sketch<T>>> = right_to_left_sketching(h, x)
        .into_iter()
        .map(Some)
        .collect();
    let mut lambda: T = crate::hamiltonian::rayleigh_quotient(h, x);
    let mut r = 1.0f64;

    for it in 0..opts.max_iter {
        let inner_tol = clamp_hi_first(0.01 * r, opts.eps, INNER_TOL_CAP);

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

/// One forward+back one-site ALS sweep pair at a CALLER-SET inner Davidson tolerance
/// (`inner_tol` is the absolute local residual norm `‖H_eff u − θu‖` at unit `u`, see
/// `solvers::davidson`). The retraction-correction kernel of
/// [`crate::solvers::riemannian::Retraction::AlsRounded`]: after a truncated tangent
/// step, one pair re-minimizes every core's intra-core content while leaving the inter-core
/// (soft) displacement alone — block Gauss–Seidel is exactly blind to the latter.
pub fn als_sweep_pair<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    inner_tol: f64,
) -> T {
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Sketch<T>>> = right_to_left_sketching(h, x)
        .into_iter()
        .map(Some)
        .collect();
    let (_lf, mut fl) = als_forward_sweep(h, x, &mut fr, inner_tol);
    debug_assert!(
        fr.iter().all(Option::is_none),
        "L1: the forward sweep must consume the whole right stack"
    );
    let (lambda, _fr_new) = als_back_sweep(h, x, &mut fl, inner_tol);
    debug_assert!(
        fl.iter().all(Option::is_none),
        "L1: the back sweep must consume the whole left stack"
    );
    lambda
}

fn local_ground<T: Scalar>(
    h: &SparseHamiltonian<T>,
    fl: &Sketch<T>,
    fr: &Sketch<T>,
    ws: &OneSiteWorkspace,
    v0: &SparseCore<T>,
    inner_tol: f64,
) -> (f64, SparseCore<T>) {
    let d_diag = h.effective_diagonal(fl, fr, v0, ws);
    let mut op = |a: &SparseCore<T>| h.sketched_hamiltonian(a, fl, fr, ws);
    let mut applyprec = |t: &mut SparseCore<T>, theta: f64| precond(t, &d_diag, theta);
    let dopts = DavidsonOpts {
        tol: inner_tol,
        ..Default::default()
    };
    let (theta, v, _info) = davidson(&mut op, v0, &mut applyprec, &dopts);
    (theta, v)
}

fn als_forward_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fr: &mut [Option<Sketch<T>>],
    inner_tol: f64,
) -> (T, Vec<Option<Sketch<T>>>) {
    debug_assert_eq!(
        x.ortho,
        Some(0),
        "forward sweep expects a right-orthogonal x"
    );
    let d = x.space.d;
    let mut fl: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    let mut lambda = T::zero();

    for k in 0..d {
        let ws = onesite_workspace(h, x.core(k));
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
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fl: &mut [Option<Sketch<T>>],
    inner_tol: f64,
) -> (T, Vec<Option<Sketch<T>>>) {
    let d = x.space.d;
    debug_assert_eq!(
        x.ortho,
        Some(d - 1),
        "back sweep expects a left-orthogonal x"
    );
    let mut fr: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    let mut lambda = T::zero();

    for k in (0..d).rev() {
        let ws = onesite_workspace(h, x.core(k));
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
mod phase5_tests {
    use super::boundary_idsketch;
    use crate::TtVector;
    use crate::qn::{QnSpace, Sector};
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t1_boundary_idsketch_blocks_identity() {
        let d = 6usize;
        let space = QnSpace::new(d, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A01);
        let x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
        x.check();

        for &k in &[0usize, d] {
            let sk = boundary_idsketch(&x, k, 1);
            let admissible = space.bond_sectors(k);
            for nup in 0..=space.nup {
                for ndn in 0..=space.ndn {
                    let s = Sector::new(nup, ndn);
                    let present = sk.block(s).is_some();
                    let expected = admissible.contains(&s);
                    assert_eq!(
                        present, expected,
                        "boundary_idsketch presence mismatch at {s:?}"
                    );
                    if expected {
                        let b = sk.block(s).unwrap();
                        assert_eq!((b.nrows(), b.ncols()), (1, 1));
                        assert!((*b.get(0, 0) - 1.0).abs() < 1e-14);
                    }
                }
            }
        }
    }
}
