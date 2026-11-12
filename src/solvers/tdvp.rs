// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Two-site projector-splitting TDVP: structurally the MALS sweep with Davidson replaced by
//! [`super::expmv::expmv`] and an extra one-site *back half-step* per bond.
//! The environment step is built UNCONDITIONALLY every iteration — do not copy MALS's
//! conditional guard.

use crate::core::factor::Sketch;
use crate::core::twosite::{contract, factor_svd};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::matvec::onesite_workspace;
use crate::hamiltonian::rayleigh_quotient;
use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::solvers::als::{boundary_idsketch, right_to_left_sketching};
use crate::solvers::expmv::{ExpmvInfo, expmv};
use crate::vector::TtVector;

#[inline]
fn report_unconverged(realtime: bool, info: &ExpmvInfo, site: &str) {
    if realtime && !info.converged {
        eprintln!(
            "tdvp_realtime: {site} expmv did not reach exptol (krylovdim {})",
            info.krylovdim
        );
    }
}

const EXPMV_MAXDIM: usize = 30;

fn normalize_center<T: Scalar>(x: &mut TtVector<T>) {
    let n = x.norm();
    if n != 0.0 {
        x.scale_mut(T::from_f64(1.0 / n));
    }
}

/// `tdvp` keyword options.
#[derive(Clone, Copy, Debug)]
pub struct TdvpOptions {
    /// Imaginary-time half-step size; default `0.1`.
    pub dtau: f64,
    /// Rayleigh-quotient relative-change convergence tolerance; default `1e-4`.
    pub eps: f64,
    /// Per-bond SVD truncation budget (`ep = trunc/√(d-1)`); default `1e-8`.
    pub trunc: f64,
    pub max_sweeps: usize,
    /// Pooled total bond-dimension cap; [`Pooled::NONE`] = uncapped.
    pub chi: Pooled,
    /// [`super::expmv::expmv`] stopping tolerance; default `1e-10`.
    pub exptol: f64,
    pub verbose: bool,
}

impl Default for TdvpOptions {
    fn default() -> Self {
        TdvpOptions {
            dtau: 0.1,
            eps: 1e-4,
            trunc: 1e-8,
            max_sweeps: 100,
            chi: Pooled::NONE,
            exptol: 1e-10,
            verbose: false,
        }
    }
}

/// Imaginary-time two-site TDVP ground-state search: alternating forward/back second-order
/// symmetric sweeps of imaginary-time size `Δτ`, renormalized after each sweep, until the relative
/// Rayleigh-quotient change drops below `opts.eps`. Returns `(λ, x)` with `x` unit-norm.
pub fn tdvp<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &TdvpOptions,
) -> (f64, TtVector<T>) {
    let mut x = x0.clone();
    let d = x.space.d;
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Sketch<T>>> = right_to_left_sketching(h, &x)
        .into_iter()
        .map(Some)
        .collect();
    let ep = if d > 1 {
        opts.trunc / ((d - 1) as f64).sqrt()
    } else {
        opts.trunc
    };

    normalize_center(&mut x);
    let mut lambda = rayleigh_quotient(h, &x).re_s();
    let a = T::from_f64(-opts.dtau / 2.0);
    for it in 0..opts.max_sweeps {
        let mut fl = tdvp_forward_sweep(h, &mut x, &mut fr, a, ep, opts.chi, opts.exptol, true);
        fr = tdvp_back_sweep(h, &mut x, &mut fl, a, ep, opts.chi, opts.exptol, true);
        normalize_center(&mut x);
        let lambda_n = rayleigh_quotient(h, &x).re_s();
        let r = (lambda - lambda_n).abs() / lambda_n.abs();
        lambda = lambda_n;
        if opts.verbose {
            println!("TDVP sweep {it}: λ={lambda}, Δ={r}");
        }
        if r < opts.eps {
            break;
        }
    }
    (lambda, x)
}

#[allow(clippy::too_many_arguments)]
fn tdvp_forward_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fr: &mut [Option<Sketch<T>>],
    a: T,
    ep: f64,
    chi: Pooled,
    exptol: f64,
    normalize: bool,
) -> Vec<Option<Sketch<T>>> {
    debug_assert_eq!(x.ortho, Some(0), "TDVP forward sweep expects center 0");
    let d = x.space.d;
    let mut fl: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    fr[0].take();

    for k in 0..(d - 1) {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let cp = {
            let fl_k = fl[k].as_ref().unwrap();
            let fr_k1 = fr[k + 1].as_ref().unwrap();
            let (cp, info) = expmv(
                |c| h.two_site_sketched_hamiltonian(c, fl_k, fr_k1),
                a,
                &cp0,
                exptol,
                EXPMV_MAXDIM,
            );
            report_unconverged(!normalize, &info, "forward two-site");
            cp
        };
        let (xk, s, mut xk1) = factor_svd(&cp, ep, chi, normalize);
        xk1.fold_sketch_left_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k + 1);

        let ws_env = onesite_workspace(h, x.core(k));
        let next = h.sketching_step_right(x, fl[k].as_ref().unwrap(), &ws_env);
        fl[k + 1] = Some(next);

        if k < d - 2 {
            let y0 = x.core(k + 1).clone();
            let ws1 = onesite_workspace(h, &y0);
            let fl_c = fl[k + 1].as_ref().unwrap();
            let fr_c = fr[k + 1].as_ref().unwrap();
            let (c, info) = expmv(
                |y| h.sketched_hamiltonian(y, fl_c, fr_c, &ws1),
                -a,
                &y0,
                exptol,
                EXPMV_MAXDIM,
            );
            report_unconverged(!normalize, &info, "forward one-site");
            x.cores[k + 1] = c;
        }
        fr[k + 1].take();
    }
    fl
}

#[allow(clippy::too_many_arguments)]
fn tdvp_back_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fl: &mut [Option<Sketch<T>>],
    a: T,
    ep: f64,
    chi: Pooled,
    exptol: f64,
    normalize: bool,
) -> Vec<Option<Sketch<T>>> {
    let d = x.space.d;
    debug_assert_eq!(x.ortho, Some(d - 1), "TDVP back sweep expects center d-1");
    let mut fr: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    fl[d - 1].take();

    for k in (0..(d - 1)).rev() {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let cp = {
            let fl_k = fl[k].as_ref().unwrap();
            let fr_k1 = fr[k + 1].as_ref().unwrap();
            let (cp, info) = expmv(
                |c| h.two_site_sketched_hamiltonian(c, fl_k, fr_k1),
                a,
                &cp0,
                exptol,
                EXPMV_MAXDIM,
            );
            report_unconverged(!normalize, &info, "back two-site");
            cp
        };
        let (mut xk, s, xk1) = factor_svd(&cp, ep, chi, normalize);
        xk.fold_sketch_right_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k);

        let ws_env = onesite_workspace(h, x.core(k + 1));
        let next = h.sketching_step_left(x, fr[k + 1].as_ref().unwrap(), &ws_env);
        fr[k] = Some(next);

        if k > 0 {
            let y0 = x.core(k).clone();
            let ws1 = onesite_workspace(h, &y0);
            let fl_c = fl[k].as_ref().unwrap();
            let fr_c = fr[k].as_ref().unwrap();
            let (c, info) = expmv(
                |y| h.sketched_hamiltonian(y, fl_c, fr_c, &ws1),
                -a,
                &y0,
                exptol,
                EXPMV_MAXDIM,
            );
            report_unconverged(!normalize, &info, "back one-site");
            x.cores[k] = c;
        }
        fl[k].take();
    }
    fr
}

/// `tdvp_realtime` options. `dt`/`n_steps` have no default, so [`RealtimeOptions::new`] takes them
/// positionally.
#[derive(Clone, Copy, Debug)]
pub struct RealtimeOptions {
    pub dt: f64,
    pub n_steps: usize,
    pub trunc: f64,
    /// Pooled total bond-dimension cap; [`Pooled::NONE`] = uncapped.
    pub chi: Pooled,
    pub exptol: f64,
    pub verbose: bool,
}

impl RealtimeOptions {
    /// `dt`/`n_steps` required; `trunc`/`chi`/`exptol`/`verbose` at their defaults.
    pub fn new(dt: f64, n_steps: usize) -> Self {
        RealtimeOptions {
            dt,
            n_steps,
            trunc: 1e-8,
            chi: Pooled::NONE,
            exptol: 1e-10,
            verbose: false,
        }
    }
}

/// Real-time two-site TDVP: unitary evolution `x(t) = exp(-i·t·H)·x0`; `T` must be complex. Gated
/// by `t8e_tdvp_realtime_sanity`: per-step norm/energy
/// conservation via an observer plus the global-phase oracle `‖x(t) − e^{-iλt}·x0‖/‖x0‖ < 1e-6` on
/// an ALS eigenstate (detects sign/scale errors a bare conservation check would miss). Returns
/// `(t_final, x)`.
#[allow(clippy::type_complexity)]
pub fn tdvp_realtime<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RealtimeOptions,
    mut observer: Option<&mut dyn FnMut(usize, f64, &TtVector<T>)>,
) -> (f64, TtVector<T>) {
    assert!(
        T::IS_COMPLEX,
        "tdvp_realtime requires a complex scalar type (a = -i·dt/2)"
    );
    let mut x = x0.clone();
    let d = x.space.d;
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Sketch<T>>> = right_to_left_sketching(h, &x)
        .into_iter()
        .map(Some)
        .collect();
    let ep = if d > 1 {
        opts.trunc / ((d - 1) as f64).sqrt()
    } else {
        opts.trunc
    };
    let a = T::from_complex_f64(0.0, -opts.dt / 2.0);
    let mut t = 0.0f64;
    if let Some(obs) = observer.as_mut() {
        obs(0, t, &x);
    }
    for step in 1..=opts.n_steps {
        let mut fl = tdvp_forward_sweep(h, &mut x, &mut fr, a, ep, opts.chi, opts.exptol, false);
        fr = tdvp_back_sweep(h, &mut x, &mut fl, a, ep, opts.chi, opts.exptol, false);
        t += opts.dt;
        if let Some(obs) = observer.as_mut() {
            obs(step, t, &x);
        }
        if opts.verbose {
            let e = rayleigh_quotient(h, &x);
            println!("RT-TDVP step {step}: t={t}, ‖x‖={}, ⟨H⟩={:?}", x.norm(), e);
        }
    }
    (t, x)
}
