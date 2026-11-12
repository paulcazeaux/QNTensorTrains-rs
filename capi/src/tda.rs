// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The SAME-SECTOR Tamm–Dancoff readout (`qntt_{f64,su2_f64}_tda`): the `k` lowest eigenpairs of
//! the Hessian's effective-Hamiltonian term `A = P_T(H − λ)P_T` alone, on the radial-deflated
//! tangent space at a real reference `x` (`qntt::solvers::hess::tda_lowest_davidson` and its SU(2)
//! twin). The response stays inside `x`'s own sector — its particle numbers, its point-group irrep
//! and, on `su2_f64`, its spin (at a singlet reference: singlet excitations only).

use std::os::raw::c_int;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::solvers::TtState;

use crate::err::{Error, catch, copy_out, ptr_ref, ptr_slice_mut};
use crate::handle::{Handle, borrow, into_raw};
use crate::opts::{QnttTdaOpts, tda_precond_code};

fn check_cap(cap: usize, need: usize, what: &str) -> Result<(), Error> {
    if cap < need {
        return Err(Error::invalid(format!(
            "{what}: holds {cap} entries, need k = {need} (checked before the solve)"
        )));
    }
    Ok(())
}

macro_rules! export_tda {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, tangent = $tangent:ty,
        tda = $tda:path, randn = $randn:path
    ) => {
        paste::paste! {
            /// The `k` lowest same-sector Tamm–Dancoff roots at the real state `x`: `θ` ascending,
            /// in hartree RELATIVE to `λ = ⟨x, Hx⟩` (module docs: `λ + θ` is an excited-state
            /// energy estimate; a negative `θ` means `x` is not the lowest state of its sector).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tda>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                k: usize,
                opts: *const QnttTdaOpts,
                seed: u64,
                out_theta: *mut f64,
                cap: usize,
                out_len: *mut usize,
                out_residual: *mut f64,
                res_cap: usize,
                out_res_len: *mut usize,
                out_states: *mut *mut Handle<$vec>,
                states_cap: usize,
                out_states_len: *mut usize,
                out_converged: *mut i32,
                out_negatives_certified: *mut i32,
                out_iters: *mut usize,
                out_applies: *mut usize,
                out_space_dim: *mut usize,
                out_precond: *mut i32,
            ) -> c_int {
                catch(|| {
                    let o = unsafe { ptr_ref(opts, "opts")? }.to_core()?;
                    if out_theta.is_null() {
                        return Err(Error::null("out_theta"));
                    }
                    check_cap(cap, k, "out_theta")?;
                    if !out_residual.is_null() {
                        check_cap(res_cap, k, "out_residual")?;
                    }
                    if !out_states.is_null() {
                        check_cap(states_cap, k, "out_states")?;
                    }
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    let ctx = $ctx(h, x);
                    let (base_l, base_r) = ctx.bases();
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let v0s: Vec<$tangent> = (0..k + 2)
                        .map(|_| {
                            let dir = $randn(x.space, &x.ranks, &mut rng);
                            <$tangent>::project(base_l, base_r, &dir)
                        })
                        .collect();
                    let modes = $tda(&ctx, &v0s, k, &o);
                    let n = modes.theta.len();
                    let states: Vec<$vec> = if out_states.is_null() {
                        Vec::new()
                    } else {
                        modes
                            .vectors
                            .iter()
                            .map(|v| {
                                let mut s = v.to_vector();
                                let nrm = TtState::norm(&s);
                                if nrm > 0.0 {
                                    TtState::scale_mut(&mut s, 1.0 / nrm);
                                }
                                s
                            })
                            .collect()
                    };
                    unsafe {
                        copy_out(&modes.theta, out_theta, cap, out_len, "theta")?;
                        if !out_residual.is_null() {
                            copy_out(
                                &modes.info.residuals,
                                out_residual,
                                res_cap,
                                out_res_len,
                                "residual",
                            )?;
                        } else if !out_res_len.is_null() {
                            *out_res_len = modes.info.residuals.len();
                        }
                        if !out_converged.is_null() {
                            *out_converged = modes.info.converged as i32;
                        }
                        if !out_negatives_certified.is_null() {
                            *out_negatives_certified = modes.all_negatives_certified() as i32;
                        }
                        if !out_iters.is_null() {
                            *out_iters = modes.info.iterations;
                        }
                        if !out_applies.is_null() {
                            *out_applies = modes.info.applies;
                        }
                        if !out_space_dim.is_null() {
                            *out_space_dim = modes.info.space_dim;
                        }
                        if !out_precond.is_null() {
                            *out_precond = tda_precond_code(modes.info.precond);
                        }
                        if !out_states_len.is_null() {
                            *out_states_len = states.len();
                        }
                        if !out_states.is_null() {
                            let slots = ptr_slice_mut(out_states, k, "out_states")?;
                            debug_assert!(states.len() == n && n <= k);
                            let mut states = states.into_iter();
                            for slot in slots.iter_mut() {
                                *slot = states.next().map_or(std::ptr::null_mut(), into_raw);
                            }
                        }
                    }
                    Ok(())
                })
            }
        }
    };
}

export_tda!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    tangent = qntt::core::tangent::TtTangent<f64>,
    tda = qntt::solvers::hess::tda_lowest_davidson,
    randn = qntt::tt_randn
);

export_tda!(
    su2_f64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    vec = qntt::su2::vector::Su2TtVector<f64>,
    ctx = qntt::su2::hess::weingarten_operator,
    tangent = qntt::su2::tangent::Su2TtTangent<f64>,
    tda = qntt::su2::hess::tda_lowest_davidson,
    randn = qntt::su2::vector::tt_randn
);
