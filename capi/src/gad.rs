// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Gentlest-ascent dynamics: excited states as index-1 saddles of the Rayleigh quotient on the
//! fixed-rank manifold. One entry point per `f64` layer; the launch direction `v0` is not
//! exposed (no tangent handle exists on this ABI), so every run starts from the term-A
//! initializer at `x0` — prepare `x0` with `qntt_*_saddle_seed`.

use std::os::raw::c_int;

use crate::err::{catch, ptr_mut, ptr_ref};
use crate::handle::{Handle, borrow, emit};
use crate::opts::{QnttGadInfo, QnttGadOptions};

/// Defaults for [`QnttGadOptions`], from the core `GadOpts::default()`.
///
/// # Safety
/// `out` must be a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_gad_options_default(out: *mut QnttGadOptions) -> c_int {
    catch(|| {
        *unsafe { ptr_mut(out, "out")? } = QnttGadOptions::defaults();
        Ok(())
    })
}

macro_rules! export_gad {
    ($sfx:ident, ham = $ham:ty, vec = $vec:ty, gad = $gad:path) => {
        paste::paste! {
            /// Run GAD from `x0`. `out_x` is the endpoint (on `x0`'s grid, `ranks ≤ declared`),
            /// `out_info` the flat diagnostics. `info.converged` is criticality AND, with
            /// `certify` on, the index-1 certificate — read `index1_certified` /
            /// `index_estimate` rather than trusting a critical endpoint. On a sector-pinned
            /// grid pass `preserve_grid = 1`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _gad>](
                h: *const Handle<$ham>,
                x0: *const Handle<$vec>,
                opts: *const QnttGadOptions,
                out_x: *mut *mut Handle<$vec>,
                out_info: *mut QnttGadInfo,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let x0 = unsafe { borrow(x0)? };
                    let opts = unsafe { ptr_ref(opts, "opts")? }.to_core()?;
                    let (x, info) = $gad(h, x0, None, &opts);
                    unsafe {
                        if !out_info.is_null() {
                            *out_info = QnttGadInfo::from_core(&info);
                        }
                        emit(x, out_x)
                    }
                })
            }
        }
    };
}

export_gad!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    gad = qntt::solvers::gad::gad
);

export_gad!(
    su2_f64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    vec = qntt::su2::vector::Su2TtVector<f64>,
    gad = qntt::su2::hess::gad
);
