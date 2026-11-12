// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The joint TT ⊕ orbital-rotation stack: the sector-pinned CAS reference, the alternating
//! DMRG-SCF driver ("CASSCF" from the caller's side) and the joint symplectic (RPA) spectrum.

use std::os::raw::{c_char, c_int};

use qntt::core::weingarten::{weingarten_fraction, williamson_reduced_signed_vectors};
use qntt::solvers::hess::{SymplecticSpectrum, symplectic_from_grams};
use qntt::solvers::joint::JointScfInfo;
use qntt::solvers::joint::cas::cas_pairs;

use crate::err::{Error, catch, copy_out, ptr_mut, ptr_ref};
use crate::fcidump::CFcidump;
use crate::handle::{Handle, borrow, emit, release};
use crate::opts::{QnttCasPartition, QnttCasscfInfo, QnttCasscfOptions};
use crate::write_cstr;

/// Joint-RPA status codes (`QNTT_JOINT_RPA_*`). The spectrum's three arms plus the gate refusal;
/// none is an error return.
pub const QNTT_JOINT_RPA_STABLE: i32 = 0;
pub const QNTT_JOINT_RPA_RESONANT: i32 = 1;
pub const QNTT_JOINT_RPA_UNSTABLE: i32 = 2;
pub const QNTT_JOINT_RPA_REFUSED: i32 = 3;

const JOINT_PSD_TOL: f64 = 1e-10;

/// The layer-independent diagnostics handle: the per-iteration vectors and the refusal message
/// of a joint run, which do not fit a flat struct.
pub struct CasscfDiag {
    pub info: JointScfInfo,
}
crate::handle_type!(CasscfDiag, 0x09, "CasscfDiag");

/// Release a diagnostics handle (no-op on `NULL`).
///
/// # Safety
/// `h` must be null or a live handle not already released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_casscf_diag_free(h: *mut Handle<CasscfDiag>) {
    unsafe { release(h) }
}

/// Copy one per-iteration series out (`QNTT_CASSCF_SERIES_*`): 0 = `e_history`, 1 = `g_orb_kept`,
/// 2 = `g_orb_full`, 3 = `g_orb_dropped`, 4 = `tt_gradnorms`, 5 = `orb_spectrum`,
/// 6 = `tr_radii`, 7 = `rho_history`. Size-query with `cap = 0`.
///
/// # Safety
/// See the module contract in `include/qntt.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_casscf_diag_series(
    h: *const Handle<CasscfDiag>,
    which: i32,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let d = unsafe { borrow(h)? };
        let i = &d.info;
        let s: &[f64] = match which {
            0 => &i.e_history,
            1 => &i.g_orb_kept,
            2 => &i.g_orb_full,
            3 => &i.g_orb_dropped,
            4 => &i.tt_gradnorms,
            5 => &i.orb_spectrum,
            6 => &i.tr_radii,
            7 => &i.rho_history,
            v => {
                return Err(Error::invalid(format!(
                    "casscf diag series {v} (want 0..=7)"
                )));
            }
        };
        unsafe { copy_out(s, out, cap, out_len, "casscf series") }
    })
}

/// The refusal message (empty when the run was not refused). String-out: size-query with
/// `cap = 0`, then pass a buffer of `out_len + 1` bytes.
///
/// # Safety
/// See the module contract in `include/qntt.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_casscf_diag_reason(
    h: *const Handle<CasscfDiag>,
    buf: *mut c_char,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let d = unsafe { borrow(h)? };
        let s = d.info.refused_reason.as_deref().unwrap_or("");
        unsafe { write_cstr(s, buf, cap, out_len) }
    })
}

/// Defaults for [`QnttCasscfOptions`] — from the core `JointScfOpts::default()`, except
/// `tt_preserve_grid`, which is ON here (module docs).
///
/// # Safety
/// `out` must be a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_casscf_options_default(out: *mut QnttCasscfOptions) -> c_int {
    catch(|| {
        *unsafe { ptr_mut(out, "out")? } = QnttCasscfOptions::defaults();
        Ok(())
    })
}

unsafe fn write_msg_truncated(s: &str, buf: *mut c_char, cap: usize, out_len: *mut usize) {
    if !out_len.is_null() {
        unsafe { *out_len = s.len() };
    }
    if cap == 0 || buf.is_null() {
        return;
    }
    let mut n = s.len().min(cap - 1);
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, n + 1) };
    dst[..n].copy_from_slice(&s.as_bytes()[..n]);
    dst[n] = 0;
}

macro_rules! export_joint {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        reference = $reference:path, ground = $ground:path, context = $context:path
    ) => {
        paste::paste! {
            /// The sector-pinned CAS reference state on the pinned rank grid of `part`, with the
            /// active window at reduced rank `m_active` (`0` = full). Returns its Rayleigh
            /// quotient (ELECTRONIC) and the state. **Never MALS the result**: `ranks ≤ grid` is
            /// the postcondition, and a two-site sweep regrows the pin.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _cas_reference_state>](
                h: *const Handle<$ham>,
                part: *const QnttCasPartition,
                m_active: usize,
                seed: u64,
                out_lambda: *mut f64,
                out: *mut *mut Handle<$vec>,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let part = unsafe { ptr_ref(part, "part")? }.to_core();
                    let space = h.space;
                    let (lambda, x) = $reference(h, &space, &part, m_active, seed);
                    unsafe {
                        if !out_lambda.is_null() {
                            *out_lambda = lambda;
                        }
                        emit(x, out)
                    }
                })
            }

            /// Alternating TT ⊕ orbital-rotation ground-state optimization from `x0` on the
            /// pinned CAS manifold of `part`, over the integrals `ints` (whose orbitals are the
            /// INPUT basis). Outputs: the electronic energy `out_lambda`, the state `out_x`, the
            /// rotated operator `out_ham`, the accumulated rotation `u` (row-major `d × d`, size
            /// query with `u_cap = 0`), the flat diagnostics `out_info` and the diagnostics
            /// handle `out_diag` (per-iteration series and the refusal message; free it with
            /// `qntt_casscf_diag_free`). **`out_x` / `out_ham` live in the basis `u`.**
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _casscf>](
                ints: *const Handle<CFcidump<f64>>,
                x0: *const Handle<$vec>,
                part: *const QnttCasPartition,
                opts: *const QnttCasscfOptions,
                out_lambda: *mut f64,
                out_x: *mut *mut Handle<$vec>,
                out_ham: *mut *mut Handle<$ham>,
                out_u: *mut f64,
                u_cap: usize,
                out_u_len: *mut usize,
                out_info: *mut QnttCasscfInfo,
                out_diag: *mut *mut Handle<CasscfDiag>,
            ) -> c_int {
                catch(|| {
                    let ints = unsafe { borrow(ints)? };
                    let x0 = unsafe { borrow(x0)? };
                    let part = unsafe { ptr_ref(part, "part")? }.to_core();
                    let opts = unsafe { ptr_ref(opts, "opts")? };
                    let space = x0.space;
                    let d = space.d;
                    if ints.data.norb != d {
                        return Err(Error::invalid(format!(
                            "casscf: integrals carry norb = {} but the state has d = {d}",
                            ints.data.norb
                        )));
                    }
                    if part.d() != d {
                        return Err(Error::invalid(format!(
                            "casscf: partition {}+{}+{} does not cover d = {d}",
                            part.n_core, part.n_active, part.n_virt
                        )));
                    }
                    let core_opts = opts.to_core(cas_pairs(&part, d))?;
                    let out = $ground(&ints.data, space, x0, &core_opts);
                    unsafe {
                        copy_out(&out.u, out_u, u_cap, out_u_len, "u")?;
                        if !out_lambda.is_null() {
                            *out_lambda = out.lambda;
                        }
                        if !out_info.is_null() {
                            *out_info = QnttCasscfInfo::from_core(&out.info);
                        }
                        emit(out.x, out_x)?;
                        emit(out.h, out_ham)?;
                        emit(CasscfDiag { info: out.info }, out_diag)
                    }
                })
            }

            /// The joint (TT ⊕ orbital) symplectic spectrum at the joint-critical point `(h, x)`
            /// over the pinned pairs of `part`. `tt_grad_tol` is the criticality gate (relative
            /// to `max(1, |λ|)`), `orb_floor` the keep-rule floor on the orbital metric.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _joint_rpa>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                part: *const QnttCasPartition,
                tt_grad_tol: f64,
                orb_floor: f64,
                out_omega: *mut f64,
                cap: usize,
                out_len: *mut usize,
                out_fw: *mut f64,
                fw_cap: usize,
                out_fw_len: *mut usize,
                out_status: *mut i32,
                out_n_modes: *mut usize,
                out_n_eff: *mut usize,
                msg: *mut c_char,
                msg_cap: usize,
                out_msg_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    let part = unsafe { ptr_ref(part, "part")? }.to_core();
                    let d = x.space.d;
                    if part.d() != d {
                        return Err(Error::invalid(format!(
                            "joint_rpa: partition {}+{}+{} does not cover d = {d}",
                            part.n_core, part.n_active, part.n_virt
                        )));
                    }
                    let jopts = qntt::solvers::joint::JointOpts {
                        tt_grad_tol,
                        orb_floor,
                        pairs: Some(cas_pairs(&part, d)),
                        ..Default::default()
                    };
                    let (status, omega, fw, n_modes, n_eff, text) = match $context(h, x, &jopts) {
                        Err(e) => (QNTT_JOINT_RPA_REFUSED, Vec::new(), Vec::new(), 0, 0, e.to_string()),
                        Ok(ctx) => {
                            let (basis, _drop) = ctx.joint_basis();
                            let n = basis.len();
                            let n_eff = ctx.n_eff();
                            let (a, b) = ctx.joint_hessian_grams(&basis);
                            let spectrum = symplectic_from_grams(&a, &b, n, JOINT_PSD_TOL);
                            let fw_of = |n_neg: usize| -> Vec<f64> {
                                let mut hrr = a.clone();
                                let mut hii = a.clone();
                                for i in 0..hrr.len() {
                                    hrr[i] += b[i];
                                    hii[i] -= b[i];
                                }
                                match williamson_reduced_signed_vectors(&hrr, &hii, n, JOINT_PSD_TOL) {
                                    Ok(sb) => {
                                        let n_pos = n - sb.n_neg;
                                        let mut f = vec![f64::NAN; n_neg];
                                        f.extend(weingarten_fraction(&a, &b, &sb.s_i, n, n_pos));
                                        f
                                    }
                                    Err(_) => Vec::new(),
                                }
                            };
                            match spectrum {
                                SymplecticSpectrum::Stable { omega } => (
                                    QNTT_JOINT_RPA_STABLE,
                                    omega,
                                    fw_of(0),
                                    n,
                                    n_eff,
                                    String::from("stable: both Hessian blocks PD"),
                                ),
                                SymplecticSpectrum::Resonant { signed } => (
                                    QNTT_JOINT_RPA_RESONANT,
                                    signed
                                        .omega2
                                        .iter()
                                        .map(|w2| w2.signum() * w2.abs().sqrt())
                                        .collect(),
                                    fw_of(signed.n_neg),
                                    n,
                                    n_eff,
                                    format!(
                                        "resonant: {} imaginary pair(s); metric block {:?} (λ_min = {:.3e})",
                                        signed.n_neg, signed.metric, signed.metric_min_eig
                                    ),
                                ),
                                SymplecticSpectrum::Unstable { reason, .. } => (
                                    QNTT_JOINT_RPA_UNSTABLE,
                                    Vec::new(),
                                    Vec::new(),
                                    n,
                                    n_eff,
                                    format!("unstable: both blocks indefinite ({reason:?})"),
                                ),
                            }
                        }
                    };
                    unsafe {
                        copy_out(&omega, out_omega, cap, out_len, "omega")?;
                        copy_out(&fw, out_fw, fw_cap, out_fw_len, "f_w")?;
                        *ptr_mut(out_status, "out_status")? = status;
                        if !out_n_modes.is_null() {
                            *out_n_modes = n_modes;
                        }
                        if !out_n_eff.is_null() {
                            *out_n_eff = n_eff;
                        }
                        write_msg_truncated(&text, msg, msg_cap, out_msg_len);
                    }
                    Ok(())
                })
            }
        }
    };
}

export_joint!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    reference = qntt::solvers::joint::cas::cas_reference_state,
    ground = qntt::solvers::joint::joint_ground_state,
    context = qntt::solvers::joint::joint_context
);

export_joint!(
    su2_f64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    vec = qntt::su2::vector::Su2TtVector<f64>,
    reference = qntt::su2::joint::cas::su2_cas_reference_state,
    ground = qntt::su2::joint::su2_joint_ground_state,
    context = qntt::su2::joint::su2_joint_context
);
