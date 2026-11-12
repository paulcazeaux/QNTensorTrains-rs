// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::lanczos::{LanczosOptions, lanczos};

use crate::err::{CResult, ptr_mut, ptr_slice, ptr_slice_mut};
use crate::handle::{Handle, borrow, emit};
use crate::layer::Layer;
use crate::opts::{
    QnttAlsOptions, QnttLanczosOptions, QnttMalsOptions, QnttMalsSweep, QnttRiemannianOpts,
    QnttTdvpOptions,
};

/// One-site ALS (DMRG) ground state.
///
/// # Safety
/// `opts`/`out_lambda`/`out_x` must be valid pointers.
pub unsafe fn als<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    opts: *const QnttAlsOptions,
    out_lambda: *mut L::T,
    out_x: *mut *mut Handle<L::Vector>,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let o = unsafe { crate::err::ptr_ref(opts, "opts")? }.to_core();
    let (lambda, x) = L::als(h, x0, &o);
    unsafe {
        *ptr_mut(out_lambda, "out_lambda")? = lambda;
        emit(x, out_x)
    }
}

/// Two-site rank-adaptive MALS (DMRG) ground state. `bond_dims` is the per-outer-iteration rank
/// cap schedule (`0` = uncapped) and must be non-empty; `noise` is the matching per-iteration
/// noise schedule and MAY be empty, meaning noise-free.
///
/// # Safety
/// `bond_dims`/`noise` must hold their stated lengths.
#[allow(clippy::too_many_arguments)]
pub unsafe fn mals<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    opts: *const QnttMalsOptions,
    bond_dims: *const usize,
    bond_dims_len: usize,
    noise: *const f64,
    noise_len: usize,
    out_lambda: *mut f64,
    out_x: *mut *mut Handle<L::Vector>,
    out_sweeps: *mut QnttMalsSweep,
    sweeps_cap: usize,
    out_nsweeps: *mut usize,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let bd = unsafe { ptr_slice(bond_dims, bond_dims_len, "bond_dims")? };
    let ns = unsafe { ptr_slice(noise, noise_len, "noise")? };
    let o = unsafe { crate::err::ptr_ref(opts, "opts")? }.to_core(bd, ns)?;
    let (hist, x) = L::mals(h, x0, &o);
    unsafe {
        *ptr_mut(out_lambda, "out_lambda")? = hist.lambda;
        if !out_nsweeps.is_null() {
            *out_nsweeps = hist.sweeps.len();
        }
        if sweeps_cap > 0 && !out_sweeps.is_null() {
            let n = sweeps_cap.min(hist.sweeps.len());
            let dst = ptr_slice_mut(out_sweeps, n, "out_sweeps")?;
            for (d, s) in dst.iter_mut().zip(hist.sweeps.iter()) {
                *d = QnttMalsSweep {
                    energy: s.energy,
                    discarded_weight: s.discarded_weight,
                    max_bond_dim: s.max_bond_dim,
                    bond_dim: s.bond_dim,
                    noise: s.noise,
                };
            }
        }
        emit(x, out_x)
    }
}

/// Imaginary-time two-site TDVP ground state.
///
/// # Safety
/// `opts` must be readable.
pub unsafe fn tdvp<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    opts: *const QnttTdvpOptions,
    out_lambda: *mut f64,
    out_x: *mut *mut Handle<L::Vector>,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let o = unsafe { crate::err::ptr_ref(opts, "opts")? }.to_core();
    let (lambda, x) = L::tdvp(h, x0, &o);
    unsafe {
        *ptr_mut(out_lambda, "out_lambda")? = lambda;
        emit(x, out_x)
    }
}

/// Riemannian optimization on the fixed-rank TT manifold. `use_cg != 0` selects conjugate
/// gradients, otherwise steepest descent.
///
/// # Safety
/// `opts` must be readable.
pub unsafe fn riemannian<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    opts: *const QnttRiemannianOpts,
    use_cg: i32,
    out_lambda: *mut f64,
    out_x: *mut *mut Handle<L::Vector>,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let o = unsafe { crate::err::ptr_ref(opts, "opts")? };
    let (lambda, x) = L::riemannian(h, x0, o, use_cg != 0)?;
    unsafe {
        *ptr_mut(out_lambda, "out_lambda")? = lambda;
        emit(x, out_x)
    }
}

/// Rounded-Lanczos ground state. The four optional history buffers each hold up to `hist_cap`
/// entries; `out_iters` reports how many were produced (never more than `opts.max_iter`).
///
/// # Safety
/// History buffers must hold `hist_cap` entries each, or be null.
#[allow(clippy::too_many_arguments)]
pub unsafe fn lanczos_solve<L: Layer>(
    h: *const Handle<L::Ham>,
    x0: *const Handle<L::Vector>,
    opts: *const QnttLanczosOptions,
    out_lambda: *mut f64,
    out_x: *mut *mut Handle<L::Vector>,
    out_breakdown: *mut i32,
    out_iters: *mut usize,
    hist_lambdas: *mut f64,
    hist_res: *mut f64,
    hist_ranks: *mut usize,
    hist_rounded_ranks: *mut usize,
    hist_cap: usize,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x0 = unsafe { borrow(x0)? };
    let o: LanczosOptions = unsafe { crate::err::ptr_ref(opts, "opts")? }.to_core();
    let r = lanczos(h, x0, &o);
    unsafe {
        *ptr_mut(out_lambda, "out_lambda")? = r.lambda;
        if !out_breakdown.is_null() {
            *out_breakdown = r.breakdown as i32;
        }
        if !out_iters.is_null() {
            *out_iters = r.lambdas.len();
        }
        let take = hist_cap.min(r.lambdas.len());
        if take > 0 {
            if !hist_lambdas.is_null() {
                ptr_slice_mut(hist_lambdas, take, "hist_lambdas")?
                    .copy_from_slice(&r.lambdas[..take]);
            }
            if !hist_res.is_null() {
                let n = take.min(r.res.len());
                ptr_slice_mut(hist_res, n, "hist_res")?.copy_from_slice(&r.res[..n]);
            }
            if !hist_ranks.is_null() {
                let n = take.min(r.ranks.len());
                ptr_slice_mut(hist_ranks, n, "hist_ranks")?.copy_from_slice(&r.ranks[..n]);
            }
            if !hist_rounded_ranks.is_null() {
                let n = take.min(r.rounded_ranks.len());
                ptr_slice_mut(hist_rounded_ranks, n, "hist_rounded_ranks")?
                    .copy_from_slice(&r.rounded_ranks[..n]);
            }
        }
        emit(r.x, out_x)
    }
}
