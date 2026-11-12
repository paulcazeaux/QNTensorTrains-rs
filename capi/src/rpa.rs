// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Symplectic (RPA / linear-response) drivers.

use std::ffi::c_char;
use std::os::raw::c_int;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::C64;
use qntt::core::weingarten::RpaMetricBlock;
use qntt::solvers::hess::{
    RpaMetric, RpaModeKind, RpaReadout, RpaReport, SectorOpts, SectorReadout, SectorResponseError,
    SectorSpectrum, SymplecticSpectrum,
};
use qntt::{CoeffTensor, PropertyOperator};

use crate::err::{
    Error, QNTT_INVALID_ARG, QNTT_NOT_CONVERGED, catch, copy_out, ptr_mut, ptr_ref, ptr_slice,
};
use crate::handle::{Handle, borrow};
use crate::opts::{
    QNTT_RPA_METRIC_AUTO, QNTT_RPA_METRIC_II, QNTT_RPA_METRIC_RR, QnttSectorAlphaInfo,
    QnttSectorAlphaOpts, QnttSectorMomentOpts, QnttSectorMomentsInfo, QnttSectorOpts,
    QnttSymplecticLowestOpts, curvature_reg_from_code, rpa_kernel_from_code, rpa_metric_code,
    sector_precond_from_code,
};

/// The `kernel` request / `out_kernel` echo codes of the `rpa_signed` exports, defined beside
/// their decoder in [`crate::opts`] and re-exported here with the exports that read them.
pub use crate::opts::{QNTT_RPA_KERNEL_PAIRED, QNTT_RPA_KERNEL_SIGNED};

struct ReadoutWire {
    omega2: Vec<f64>,
    krein: Vec<f64>,
    residual: Vec<f64>,
    complex: Vec<f64>,
    kernel: i32,
    report: [i64; 5],
}

fn readout_wire(r: &RpaReadout) -> ReadoutWire {
    let real: Vec<_> = r
        .modes
        .iter()
        .filter(|m| m.kind != RpaModeKind::Complex)
        .collect();
    let (kernel, report) = match &r.report {
        RpaReport::Paired { offset, .. } => (
            QNTT_RPA_KERNEL_PAIRED,
            [offset.neg_apb as i64, offset.neg_amb as i64, -1, -1, -1],
        ),
        RpaReport::Signed {
            events,
            n_negative_basis,
            general_path,
            ..
        } => (
            QNTT_RPA_KERNEL_SIGNED,
            [
                -1,
                -1,
                events.len() as i64,
                *n_negative_basis as i64,
                i64::from(*general_path),
            ],
        ),
    };
    ReadoutWire {
        omega2: real.iter().map(|m| m.omega2.re).collect(),
        krein: real.iter().map(|m| m.krein).collect(),
        residual: real.iter().map(|m| m.residual).collect(),
        complex: r
            .complex()
            .flat_map(|m| [m.omega2.re, m.omega2.im.abs(), m.residual])
            .collect(),
        kernel,
        report,
    }
}

fn fw_side(r: &RpaReadout, requested: RpaMetric) -> RpaMetric {
    match r.report {
        RpaReport::Signed { metric, .. } => metric,
        RpaReport::Paired { .. } if requested == RpaMetric::Rr => RpaMetric::Rr,
        RpaReport::Paired { .. } => RpaMetric::Ii,
    }
}

macro_rules! export_rpa_dense {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, basis = $basis:path, eigs = $eigs:path
    ) => {
        paste::paste! {
            /// Dense (certified-oracle) symplectic spectrum over the full tangent basis at `x`.
            /// `out_stable = 0` means the critical point is not a stable minimizer; ω is then
            /// empty and the diagnosis is in the returned message.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _rpa_dense>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                out_omega: *mut f64,
                cap: usize,
                out_len: *mut usize,
                out_stable: *mut i32,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    let ctx = $ctx(h, x);
                    let (base_l, base_r) = ctx.bases();
                    let basis = $basis(base_l, base_r, true);
                    match $eigs(&ctx, &basis) {
                        SymplecticSpectrum::Stable { omega } => {
                            unsafe {
                                *ptr_mut(out_stable, "out_stable")? = 1;
                                copy_out(&omega, out_omega, cap, out_len, "omega")
                            }
                        }
                        SymplecticSpectrum::Resonant { signed } => {
                            unsafe {
                                *ptr_mut(out_stable, "out_stable")? = 0;
                                if !out_len.is_null() {
                                    *out_len = 0;
                                }
                            }
                            let w2 = signed.omega2.first().copied().unwrap_or(0.0);
                            let kappa = w2.abs().sqrt();
                            let (np, met) = (signed.n_neg, signed.metric);
                            let msg = format!("resonant reference: {np} unstable pair(s), metric {met:?}, lowest ω² = {w2:.6e} (ω = {kappa:.6e}i)");
                            Err(Error::new(QNTT_NOT_CONVERGED, msg))
                        }
                        SymplecticSpectrum::Unstable { reason, .. } => {
                            unsafe {
                                *ptr_mut(out_stable, "out_stable")? = 0;
                                if !out_len.is_null() {
                                    *out_len = 0;
                                }
                            }
                            Err(Error::new(
                                QNTT_NOT_CONVERGED,
                                format!("not a stable minimizer: {reason:?}"),
                            ))
                        }
                    }
                })
            }
        }
    };
}

export_rpa_dense!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    basis = qntt::core::tangent::tangent_basis,
    eigs = qntt::solvers::hess::symplectic_eigenvalues
);

export_rpa_dense!(
    c64,
    ham = qntt::SparseHamiltonian<C64>,
    vec = qntt::TtVector<C64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    basis = qntt::core::tangent::tangent_basis,
    eigs = qntt::solvers::hess::symplectic_eigenvalues_c64
);

export_rpa_dense!(
    su2_f64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    vec = qntt::su2::vector::Su2TtVector<f64>,
    ctx = qntt::su2::hess::weingarten_operator,
    basis = qntt::su2::weingarten::tangent_basis,
    eigs = qntt::su2::hess::symplectic_eigenvalues
);

export_rpa_dense!(
    su2_c64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<C64>,
    vec = qntt::su2::vector::Su2TtVector<C64>,
    ctx = qntt::su2::hess::weingarten_operator,
    basis = qntt::su2::weingarten::tangent_basis,
    eigs = qntt::su2::hess::symplectic_eigenvalues_c64
);

macro_rules! export_rpa_signed {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, tangent = $tangent:ty,
        signed = $signed:path, randn = $randn:path
    ) => {
        paste::paste! {
            /// Signed `ω²` at the real state `x`: every imaginary mode the solve found plus the
            /// `k` lowest real ones, ascending. Positive entries are excitations, negative ones
            /// resonances `ω = i√|ω²|`, each certified by fresh applies (`out_negatives_certified`;
            /// publish an `iκ` only under it). `kernel` is a `QNTT_RPA_KERNEL_*` request:
            /// `PAIRED` (`0`, the library default) or `SIGNED` (the W-metric Davidson); any other
            /// code is `QNTT_INVALID_ARG` before any work. `metric` is a `QNTT_RPA_METRIC_*` request
            /// for the side `q = X ∓ Y` the Weingarten fraction reads, and under `SIGNED` also the
            /// kernel's lane (`AUTO` = `II` under both, no probe); `out_metric` echoes the side
            /// used — `QNTT_RPA_METRIC_II = 0` or `_RR = 1`, **not** the `QNTT_HESS_*` codes.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _rpa_signed>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                k: usize,
                opts: *const QnttSymplecticLowestOpts,
                kernel: c_int,
                metric: c_int,
                curvature_kind: c_int,
                curvature_value: f64,
                seed: u64,
                out_omega2: *mut f64,
                cap: usize,
                out_len: *mut usize,
                out_fw: *mut f64,
                fw_cap: usize,
                out_fw_len: *mut usize,
                out_residual: *mut f64,
                res_cap: usize,
                out_res_len: *mut usize,
                out_krein: *mut f64,
                krein_cap: usize,
                out_krein_len: *mut usize,
                out_complex: *mut f64,
                cx_cap: usize,
                out_cx_len: *mut usize,
                out_metric: *mut i32,
                out_negatives_certified: *mut i32,
                out_converged: *mut i32,
                out_iters: *mut usize,
                out_applies: *mut usize,
                out_kernel: *mut i32,
                out_report: *mut i64,
                report_cap: usize,
                out_report_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let (kernel, metric) = rpa_kernel_from_code(kernel, metric)?;
                    let reg = curvature_reg_from_code(curvature_kind, curvature_value)?;
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    let mut o = unsafe { ptr_ref(opts, "opts")? }.to_davidson();
                    o.kernel = kernel;
                    let ctx = $ctx(h, x).with_curvature(reg);
                    let (base_l, base_r) = ctx.bases();
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let v0s: Vec<$tangent> = (0..k + 2)
                        .map(|_| {
                            let dir = $randn(x.space, &x.ranks, &mut rng);
                            <$tangent>::project(base_l, base_r, &dir)
                        })
                        .collect();
                    let (so, amps) = $signed(&ctx, &v0s, k, &o);
                    let side = fw_side(&so, metric);
                    let wire = readout_wire(&so);
                    let fw: Vec<f64> = amps
                        .iter()
                        .take(wire.omega2.len())
                        .map(|m| {
                            let v = &m.metric_side(side);
                            let qa = v.dot(&ctx.a_apply(v));
                            let qb = v.dot(&ctx.b_apply(v));
                            if qa == 0.0 { f64::INFINITY } else { qb.abs() / qa.abs() }
                        })
                        .collect();
                    unsafe {
                        copy_out(&wire.omega2, out_omega2, cap, out_len, "omega2")?;
                        copy_out(&fw, out_fw, fw_cap, out_fw_len, "f_w")?;
                        copy_out(&wire.residual, out_residual, res_cap, out_res_len, "residual")?;
                        copy_out(&wire.krein, out_krein, krein_cap, out_krein_len, "krein")?;
                        copy_out(&wire.complex, out_complex, cx_cap, out_cx_len, "complex")?;
                        copy_out(&wire.report, out_report, report_cap, out_report_len, "report")?;
                        if !out_metric.is_null() {
                            *out_metric = rpa_metric_code(side);
                        }
                        if !out_negatives_certified.is_null() {
                            *out_negatives_certified = so.all_negatives_certified() as i32;
                        }
                        if !out_converged.is_null() {
                            *out_converged = so.converged as i32;
                        }
                        if !out_iters.is_null() {
                            *out_iters = so.iterations;
                        }
                        if !out_applies.is_null() {
                            *out_applies = so.applies;
                        }
                        if !out_kernel.is_null() {
                            *out_kernel = wire.kernel;
                        }
                    }
                    Ok(())
                })
            }
        }
    };
}

export_rpa_signed!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    tangent = qntt::core::tangent::TtTangent<f64>,
    signed = qntt::solvers::hess::symplectic_lowest_signed_davidson,
    randn = qntt::tt_randn
);

export_rpa_signed!(
    su2_f64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    vec = qntt::su2::vector::Su2TtVector<f64>,
    ctx = qntt::su2::hess::weingarten_operator,
    tangent = qntt::su2::tangent::Su2TtTangent<f64>,
    signed = qntt::su2::hess::symplectic_lowest_signed_davidson,
    randn = qntt::su2::vector::tt_randn
);

macro_rules! export_rpa_signed_c64 {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, tangent = $tangent:ty,
        signed = $signed:path, randn = $randn:path
    ) => {
        paste::paste! {
            /// Signed `ω²` of the realified `C64` pencil at `x`, both copies of each: every negative
            /// entry plus the `k` lowest non-negative realified values, ascending, by the kernel
            /// `kernel` requests (`QNTT_RPA_KERNEL_*`; `SIGNED` runs on `RR`; module docs,
            /// [`export_rpa_signed_c64`]). Per-mode buffer-outs, the complex rows, the kernel code
            /// and the report read as on `qntt_*_rpa_signed` of the `f64` layers; there is no
            /// metric request, no curvature pair and no `out_metric`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _rpa_signed>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                k: usize,
                opts: *const QnttSymplecticLowestOpts,
                kernel: c_int,
                seed: u64,
                out_omega2: *mut f64,
                cap: usize,
                out_len: *mut usize,
                out_fw: *mut f64,
                fw_cap: usize,
                out_fw_len: *mut usize,
                out_residual: *mut f64,
                res_cap: usize,
                out_res_len: *mut usize,
                out_krein: *mut f64,
                krein_cap: usize,
                out_krein_len: *mut usize,
                out_complex: *mut f64,
                cx_cap: usize,
                out_cx_len: *mut usize,
                out_negatives_certified: *mut i32,
                out_converged: *mut i32,
                out_iters: *mut usize,
                out_applies: *mut usize,
                out_kernel: *mut i32,
                out_report: *mut i64,
                report_cap: usize,
                out_report_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let (kernel, _) = rpa_kernel_from_code(kernel, QNTT_RPA_METRIC_RR)?;
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    let mut o = unsafe { ptr_ref(opts, "opts")? }.to_davidson();
                    o.kernel = kernel;
                    let ctx = $ctx(h, x);
                    let (base_l, base_r) = ctx.bases();
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let v0s: Vec<$tangent> = (0..k + 2)
                        .map(|_| {
                            let dir = $randn(x.space, &x.ranks, &mut rng);
                            <$tangent>::project(base_l, base_r, &dir)
                        })
                        .collect();
                    let (so, amps) = $signed(&ctx, &v0s, k, &o);
                    let wire = readout_wire(&so);
                    let fw: Vec<f64> = amps
                        .iter()
                        .take(wire.omega2.len())
                        .map(|m| {
                            let v = &m.metric_side(RpaMetric::Rr);
                            let qa = v.dot(&ctx.a_apply(v)).re;
                            let qb = v.dot(&ctx.b_apply(v)).re;
                            if qa == 0.0 { f64::INFINITY } else { qb.abs() / qa.abs() }
                        })
                        .collect();
                    unsafe {
                        copy_out(&wire.omega2, out_omega2, cap, out_len, "omega2")?;
                        copy_out(&fw, out_fw, fw_cap, out_fw_len, "f_w")?;
                        copy_out(&wire.residual, out_residual, res_cap, out_res_len, "residual")?;
                        copy_out(&wire.krein, out_krein, krein_cap, out_krein_len, "krein")?;
                        copy_out(&wire.complex, out_complex, cx_cap, out_cx_len, "complex")?;
                        copy_out(&wire.report, out_report, report_cap, out_report_len, "report")?;
                        if !out_negatives_certified.is_null() {
                            *out_negatives_certified = so.all_negatives_certified() as i32;
                        }
                        if !out_converged.is_null() {
                            *out_converged = so.converged as i32;
                        }
                        if !out_iters.is_null() {
                            *out_iters = so.iterations;
                        }
                        if !out_applies.is_null() {
                            *out_applies = so.applies;
                        }
                        if !out_kernel.is_null() {
                            *out_kernel = wire.kernel;
                        }
                    }
                    Ok(())
                })
            }
        }
    };
}

export_rpa_signed_c64!(
    c64,
    ham = qntt::SparseHamiltonian<C64>,
    vec = qntt::TtVector<C64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    tangent = qntt::core::tangent::TtTangent<C64>,
    signed = qntt::solvers::hess::symplectic_lowest_signed_davidson_c64,
    randn = qntt::tt_randn
);

export_rpa_signed_c64!(
    su2_c64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<C64>,
    vec = qntt::su2::vector::Su2TtVector<C64>,
    ctx = qntt::su2::hess::weingarten_operator,
    tangent = qntt::su2::tangent::Su2TtTangent<C64>,
    signed = qntt::su2::hess::symplectic_lowest_signed_davidson_c64,
    randn = qntt::su2::vector::tt_randn
);

macro_rules! export_sector_rpa {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, sector = $sector:ty
    ) => {
        paste::paste! {
            fn [<sector_channel_ $sfx>](
                h_ref: &$ham,
                h_out: &$ham,
                x: &$vec,
                phi: u8,
            ) -> Result<qntt::pointgroup::Irrep, Error> {
                let space = x.space;
                if phi == 0 {
                    return Err(Error::invalid(concat!(
                        "sector_rpa: phi = 0 is the IN-SECTOR problem — T^IDENTITY is x's ",
                        "ordinary tangent space and A^IDENTITY its ordinary effective ",
                        "Hamiltonian; qntt_", stringify!($sfx), "_rpa_signed (or _rpa_dense) ",
                        "answers it. phi must be a NON-identity irrep of the state's ",
                        "point group."
                    )));
                }
                let order = space.pg.order();
                if phi >= order {
                    return Err(Error::invalid(format!(
                        "sector_rpa: flux phi = {phi} is outside the state's point group, whose \
                         order is {order} (a trivial group has order 1, so every phi != 0 is out \
                         of range there); the irreps are 0-based XOR indices"
                    )));
                }
                if h_ref.space != space {
                    return Err(Error::invalid(format!(
                        "sector_rpa: h_ref must be the operator on x's OWN space (it supplies \
                         lambda and the Weingarten environments); x is on {:?} and h_ref on {:?}",
                        space, h_ref.space
                    )));
                }
                let phi = qntt::pointgroup::Irrep::new(phi);
                let space_out = space.with_target(space.target.fuse(phi));
                if h_out.space != space_out {
                    return Err(Error::invalid(format!(
                        "sector_rpa: h_out must have been BUILT on space_out = {space_out:?} — \
                         the assembly automaton reads `target` for its caps and its bond-d \
                         collapse, so a re-stamp of x's own Hamiltonian is NOT enough, and the \
                         t, v, w and eps must be the ones h_ref was built from (a \
                         SparseHamiltonian records none of them, so only this space stamp is \
                         checked); got {:?}",
                        h_out.space
                    )));
                }
                Ok(phi)
            }

            /// Lowest `k` values of the flux-`phi` Hessian block at the typed reference `x` —
            /// `eig(A^phi)` (an ENERGY) under `QNTT_SECTOR_READOUT_TDA`, the signed `omega^2`
            /// under `_RPA`. `out_dim` is `dim T^phi`, which is NOT the target irrep's sector
            /// dimension. An empty `out_len` is a real answer.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _sector_rpa>](
                h_ref: *const Handle<$ham>,
                h_out: *const Handle<$ham>,
                x: *const Handle<$vec>,
                phi: u8,
                k: usize,
                opts: *const QnttSectorOpts,
                curvature_kind: c_int,
                curvature_value: f64,
                precond: c_int,
                precond_kappa: f64,
                seed: u64,
                out_theta_or_omega2: *mut f64,
                cap: usize,
                out_len: *mut usize,
                out_fw: *mut f64,
                fw_cap: usize,
                out_fw_len: *mut usize,
                out_residual: *mut f64,
                res_cap: usize,
                out_res_len: *mut usize,
                out_krein: *mut f64,
                krein_cap: usize,
                out_krein_len: *mut usize,
                out_complex: *mut f64,
                cx_cap: usize,
                out_cx_len: *mut usize,
                out_dim: *mut usize,
                out_metric: *mut i32,
                out_converged: *mut i32,
                out_iters: *mut usize,
                out_applies: *mut usize,
                out_negatives_certified: *mut i32,
                out_report: *mut i64,
                report_cap: usize,
                out_report_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let reg = curvature_reg_from_code(curvature_kind, curvature_value)?;
                    let precond = sector_precond_from_code(precond, precond_kappa)?;
                    let h_ref = unsafe { borrow(h_ref)? };
                    let h_out = unsafe { borrow(h_out)? };
                    let x = unsafe { borrow(x)? };
                    let (core, requested) = unsafe { ptr_ref(opts, "opts")? }.to_core(seed)?;
                    let o = SectorOpts { precond, ..core };
                    let phi = [<sector_channel_ $sfx>](h_ref, h_out, x, phi)?;
                    let ctx = $ctx(h_ref, x).with_curvature(reg);
                    let sc = <$sector>::new(&ctx, h_out, phi);
                    let (resp, fw) = match o.readout {
                        SectorReadout::Tda => (sc.sector_lowest_tda(k, &o), Vec::new()),
                        SectorReadout::Rpa => {
                            let (resp, vecs) = sc.sector_lowest_vectors(k, &o);
                            let readout = resp.rpa().expect("an Rpa readout");
                            let (lane, n2) =
                                (fw_side(readout, requested), readout.omega2().len());
                            let fw: Vec<f64> = vecs
                                .iter()
                                .take(n2)
                                .map(|m| {
                                    let v = &m.metric_side(lane);
                                    let qa = v.dot(&sc.a_apply(v));
                                    let qb = v.dot(&sc.b_apply(v));
                                    if qa == 0.0 { f64::INFINITY } else { qb.abs() / qa.abs() }
                                })
                                .collect();
                            (resp, fw)
                        }
                    };
                    let (vals, metric, wire) = match &resp.spectrum {
                        SectorSpectrum::Tda { theta, .. } => (
                            theta.clone(),
                            QNTT_RPA_METRIC_AUTO,
                            ReadoutWire {
                                omega2: Vec::new(),
                                krein: Vec::new(),
                                residual: Vec::new(),
                                complex: Vec::new(),
                                kernel: QNTT_RPA_KERNEL_SIGNED,
                                report: [-1; 5],
                            },
                        ),
                        SectorSpectrum::Rpa(r) => {
                            let wire = readout_wire(r);
                            (wire.omega2.clone(), rpa_metric_code(fw_side(r, requested)), wire)
                        }
                    };
                    unsafe {
                        copy_out(&vals, out_theta_or_omega2, cap, out_len, "theta_or_omega2")?;
                        copy_out(&fw, out_fw, fw_cap, out_fw_len, "f_w")?;
                        copy_out(&wire.residual, out_residual, res_cap, out_res_len, "residual")?;
                        copy_out(&wire.krein, out_krein, krein_cap, out_krein_len, "krein")?;
                        copy_out(&wire.complex, out_complex, cx_cap, out_cx_len, "complex")?;
                        copy_out(&wire.report, out_report, report_cap, out_report_len, "report")?;
                        if !out_dim.is_null() {
                            *out_dim = resp.dim;
                        }
                        if !out_metric.is_null() {
                            *out_metric = metric;
                        }
                        if !out_converged.is_null() {
                            *out_converged = resp.converged as i32;
                        }
                        if !out_iters.is_null() {
                            *out_iters = resp.iterations;
                        }
                        if !out_applies.is_null() {
                            *out_applies = resp.applies;
                        }
                        if !out_negatives_certified.is_null() {
                            *out_negatives_certified =
                                resp.all_negatives_certified() as i32;
                        }
                    }
                    Ok(())
                })
            }

            /// The DENSE certified oracle of the flux-`phi` block over the whole shifted tangent
            /// basis. `out_stable` is a TRI-state (1 stable / 0 resonant / -1 unstable); the
            /// signed `omega^2` IS written on the resonant arm, under `QNTT_OK`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _sector_rpa_dense>](
                h_ref: *const Handle<$ham>,
                h_out: *const Handle<$ham>,
                x: *const Handle<$vec>,
                phi: u8,
                out_omega2: *mut f64,
                cap: usize,
                out_len: *mut usize,
                out_dim: *mut usize,
                out_stable: *mut i32,
                out_metric: *mut i32,
            ) -> c_int {
                catch(|| {
                    let h_ref = unsafe { borrow(h_ref)? };
                    let h_out = unsafe { borrow(h_out)? };
                    let x = unsafe { borrow(x)? };
                    let phi = [<sector_channel_ $sfx>](h_ref, h_out, x, phi)?;
                    let ctx = $ctx(h_ref, x);
                    let sc = <$sector>::new(&ctx, h_out, phi);
                    let dim = sc.dim();
                    unsafe {
                        if !out_dim.is_null() {
                            *out_dim = dim;
                        }
                    }
                    let basis = sc.basis();
                    if basis.is_empty() {
                        unsafe {
                            *ptr_mut(out_stable, "out_stable")? = 1;
                            if !out_metric.is_null() {
                                *out_metric = QNTT_RPA_METRIC_AUTO;
                            }
                            return copy_out(&[], out_omega2, cap, out_len, "omega2");
                        }
                    }
                    let spec = sc.symplectic_eigenvalues(&basis);
                    let metric = match &spec {
                        SymplecticSpectrum::Resonant { signed } => match signed.metric {
                            RpaMetricBlock::Ii => QNTT_RPA_METRIC_II,
                            RpaMetricBlock::Rr => QNTT_RPA_METRIC_RR,
                        },
                        _ => QNTT_RPA_METRIC_AUTO,
                    };
                    unsafe {
                        if !out_metric.is_null() {
                            *out_metric = metric;
                        }
                    }
                    match &spec {
                        SymplecticSpectrum::Unstable { reason, .. } => {
                            unsafe {
                                *ptr_mut(out_stable, "out_stable")? = -1;
                                if !out_len.is_null() {
                                    *out_len = 0;
                                }
                            }
                            Err(Error::new(
                                QNTT_NOT_CONVERGED,
                                format!(
                                    "sector_rpa_dense: both Hessian blocks of channel phi = {} \
                                     are indefinite, so no real omega^2 exists on it: {reason:?}",
                                    phi.index()
                                ),
                            ))
                        }
                        _ => {
                            let stable =
                                if matches!(spec, SymplecticSpectrum::Stable { .. }) { 1 } else { 0 };
                            let w2 = spec.omega_squared().expect("not the Unstable arm");
                            unsafe {
                                *ptr_mut(out_stable, "out_stable")? = stable;
                                copy_out(&w2, out_omega2, cap, out_len, "omega2")
                            }
                        }
                    }
                })
            }
        }
    };
}

export_sector_rpa!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    sector = qntt::solvers::hess::SectorCtx<'_, '_, f64>
);

const QNTT_SECTOR_REFUSAL_NONE: i32 = 0;
const QNTT_SECTOR_REFUSAL_NOT_TT_CRITICAL: i32 = 1;
const QNTT_SECTOR_REFUSAL_HESSIAN_INDEFINITE: i32 = 2;
const QNTT_SECTOR_REFUSAL_SYMPLECTIC_UNSTABLE: i32 = 3;
const QNTT_SECTOR_REFUSAL_SOLVE_NOT_CONVERGED: i32 = 4;
const QNTT_SECTOR_REFUSAL_UNCERTIFIED_NEGATIVE: i32 = 5;
const QNTT_SECTOR_REFUSAL_NEGATIVE_OMEGA: i32 = 7;
const QNTT_SECTOR_REFUSAL_NEAR_RESONANCE: i32 = 8;
const QNTT_SECTOR_REFUSAL_INDEFINITE_ABOVE_POLE: i32 = 9;

fn sector_refusal_code(e: &SectorResponseError) -> (c_int, i32) {
    match e {
        SectorResponseError::WrongFlux { .. } => (QNTT_INVALID_ARG, QNTT_SECTOR_REFUSAL_NONE),
        SectorResponseError::NotTtCritical { .. } => {
            (QNTT_NOT_CONVERGED, QNTT_SECTOR_REFUSAL_NOT_TT_CRITICAL)
        }
        SectorResponseError::HessianIndefinite { .. } => {
            (QNTT_NOT_CONVERGED, QNTT_SECTOR_REFUSAL_HESSIAN_INDEFINITE)
        }
        SectorResponseError::SymplecticUnstable { .. } => {
            (QNTT_NOT_CONVERGED, QNTT_SECTOR_REFUSAL_SYMPLECTIC_UNSTABLE)
        }
        SectorResponseError::SolveNotConverged { .. } => {
            (QNTT_NOT_CONVERGED, QNTT_SECTOR_REFUSAL_SOLVE_NOT_CONVERGED)
        }
        SectorResponseError::UncertifiedNegative { .. } => {
            (QNTT_NOT_CONVERGED, QNTT_SECTOR_REFUSAL_UNCERTIFIED_NEGATIVE)
        }
        SectorResponseError::NegativeOmega { .. } => {
            (QNTT_NOT_CONVERGED, QNTT_SECTOR_REFUSAL_NEGATIVE_OMEGA)
        }
        SectorResponseError::NearResonance { .. } => {
            (QNTT_NOT_CONVERGED, QNTT_SECTOR_REFUSAL_NEAR_RESONANCE)
        }
        SectorResponseError::IndefiniteAbovePole { .. } => (
            QNTT_NOT_CONVERGED,
            QNTT_SECTOR_REFUSAL_INDEFINITE_ABOVE_POLE,
        ),
    }
}

fn property_block_len(m_len: usize, n_props: usize, norb: usize) -> Result<usize, Error> {
    if norb == 0 {
        return Err(Error::invalid(
            "norb = 0: a property matrix over zero orbitals has no entries and no flux class; \
             pass the state's own orbital count",
        ));
    }
    let block = norb
        .checked_mul(norb)
        .ok_or_else(|| Error::invalid(format!("norb = {norb} squared overflows a machine word")))?;
    let need = block.checked_mul(n_props).ok_or_else(|| {
        Error::invalid(format!(
            "n_props = {n_props} times norb^2 = {block} overflows a machine word"
        ))
    })?;
    if m_len != need {
        return Err(Error::invalid(format!(
            "m_len = {m_len} but n_props * norb * norb = {n_props} * {norb} * {norb} = {need}; \
             the property buffer holds n_props contiguous norb x norb row-major blocks and the \
             length is checked for EXACT equality, never truncated"
        )));
    }
    Ok(block)
}

fn property_coeffs_from_flat(
    m: &[f64],
    n_props: usize,
    norb: usize,
    labels: &[String],
) -> Result<Vec<PropertyOperator<f64>>, Error> {
    let block = property_block_len(m.len(), n_props, norb)?;
    let mut out = Vec::with_capacity(n_props);
    for (i, chunk) in m.chunks_exact(block).enumerate() {
        if let Some((k, bad)) = chunk
            .iter()
            .copied()
            .enumerate()
            .find(|(_, v)| !v.is_finite())
        {
            return Err(Error::invalid(format!(
                "property component {i} ({:?}) has a non-finite entry m[{}][{}] = {bad}: every \
                 entry must be finite, or property_flux_classes classifies it as support and \
                 every number downstream is poisoned with no diagnostic",
                labels[i],
                k / norb,
                k % norb
            )));
        }
        let mut t = CoeffTensor::<f64>::zeros(norb, 2);
        t.data.copy_from_slice(chunk);
        if let Some(msg) = qntt::integrals::property_hermiticity_error(&t, &labels[i]) {
            return Err(Error::invalid(format!(
                "property component {i}: {msg} — pass the FULL symmetric matrix, both triangles"
            )));
        }
        out.push(PropertyOperator {
            label: labels[i].clone(),
            m: t,
        });
    }
    Ok(out)
}

unsafe fn labels_or_defaults(
    labels: *const *const c_char,
    n_props: usize,
) -> Result<Vec<String>, Error> {
    const LABEL_MAX: usize = 256;
    let mut out = Vec::with_capacity(n_props);
    for i in 0..n_props {
        let p = if labels.is_null() {
            std::ptr::null()
        } else {
            unsafe { *labels.add(i) }
        };
        if p.is_null() {
            out.push(format!("p{i}"));
            continue;
        }
        let mut bytes: Vec<u8> = Vec::new();
        let mut k = 0usize;
        loop {
            if k > LABEL_MAX {
                return Err(Error::invalid(format!(
                    "label {i} is not NUL-terminated within {LABEL_MAX} bytes; labels are capped \
                     so a runaway pointer cannot make the boundary scan memory unbounded"
                )));
            }
            let b = unsafe { *p.add(k) } as u8;
            if b == 0 {
                break;
            }
            bytes.push(b);
            k += 1;
        }
        match std::str::from_utf8(&bytes) {
            Ok(s) => out.push(s.to_string()),
            Err(e) => {
                return Err(Error::invalid(format!(
                    "label {i} is not valid UTF-8 ({e}); labels cross as UTF-8 and appear in \
                     refusal messages"
                )));
            }
        }
    }
    Ok(out)
}

fn sector_unit_reference(n: f64) -> Result<(), Error> {
    if (n - 1.0).abs() < 1e-8 {
        Ok(())
    } else {
        Err(Error::invalid(format!(
            "the reference x must be normalized: ‖x‖ = {n:.15e}. alpha, T_n and lambda all scale \
             with the reference, so a non-unit x is refused in every build profile rather than \
             publishing numbers scaled by it"
        )))
    }
}

unsafe fn sector_obs_precheck(
    d: usize,
    m: *const f64,
    m_len: usize,
    n_props: usize,
    norb: usize,
    labels: *const *const c_char,
) -> Result<Vec<PropertyOperator<f64>>, Error> {
    if n_props == 0 {
        return Err(Error::invalid(
            "n_props = 0: a zero-component request has no alpha (it would be 0x0) and is a caller \
             bug, not a channel property; pass 1 to 3 components",
        ));
    }
    if n_props > 3 {
        return Err(Error::invalid(format!(
            "n_props = {n_props}: at most three Cartesian components — oscillator_strengths is a \
             three-row kernel whose 2/3 is the isotropic average over x, y, z, and a fourth \
             component is a caller error rather than a panic deep inside it"
        )));
    }
    if norb != d {
        return Err(Error::invalid(format!(
            "norb = {norb} but the state has d = {d} spatial orbitals; a property matrix is over \
             x's own orbitals"
        )));
    }
    let labels = unsafe { labels_or_defaults(labels, n_props)? };
    property_block_len(m_len, n_props, norb)?;
    let m = unsafe { ptr_slice(m, m_len, "m")? };
    property_coeffs_from_flat(m, n_props, norb, &labels)
}

macro_rules! export_sector_obs {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, sector = $sector:ty
    ) => {
        paste::paste! {
            /// `alpha^phi_uv = 2⟨p_u, (H^{rr,phi})^-1 p_v⟩` on one shifted irrep channel: one PCG
            /// solve per IN-CHANNEL component against `H^rr = A^phi + B^phi`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _sector_alpha>](
                h_ref: *const Handle<$ham>,
                h_out: *const Handle<$ham>,
                x: *const Handle<$vec>,
                phi: u8,
                m: *const f64,
                m_len: usize,
                n_props: usize,
                norb: usize,
                labels: *const *const c_char,
                opts: *const QnttSectorAlphaOpts,
                out_alpha: *mut f64,
                alpha_cap: usize,
                out_alpha_len: *mut usize,
                out_props: *mut f64,
                props_cap: usize,
                out_props_len: *mut usize,
                out_iters: *mut usize,
                iters_cap: usize,
                out_iters_len: *mut usize,
                out_info: *mut QnttSectorAlphaInfo,
                out_refusal: *mut i32,
            ) -> c_int {
                catch(|| {
                    if out_refusal.is_null() {
                        return Err(Error::null("out_refusal"));
                    }
                    unsafe { out_refusal.write(QNTT_SECTOR_REFUSAL_NONE) };
                    let h_ref = unsafe { borrow(h_ref)? };
                    let h_out = unsafe { borrow(h_out)? };
                    let x = unsafe { borrow(x)? };
                    let o = unsafe { ptr_ref(opts, "opts")? };
                    if out_info.is_null() {
                        return Err(Error::null("out_info"));
                    }
                    let props = unsafe {
                        sector_obs_precheck(x.space.d, m, m_len, n_props, norb, labels)?
                    };
                    let o = o.to_static()?;
                    sector_unit_reference(x.norm())?;
                    let phi = [<sector_channel_ $sfx>](h_ref, h_out, x, phi)?;
                    let ctx = $ctx(h_ref, x);
                    let sc = <$sector>::new(&ctx, h_out, phi);
                    let resp = match sc.polarizability_static(&props, &o) {
                        Ok(r) => r,
                        Err(e) => {
                            let (status, code) = sector_refusal_code(&e);
                            unsafe { out_refusal.write(code) };
                            return Err(Error::new(status, e.to_string()));
                        }
                    };
                    let mut table = Vec::with_capacity(3 * resp.solves.len());
                    for (i, s) in resp.solves.iter().enumerate() {
                        table.push(resp.capture_defect[i]);
                        table.push(s.grad_norm);
                        table.push(s.rel_residual);
                    }
                    let iters: Vec<usize> = resp.solves.iter().map(|s| s.iters).collect();
                    unsafe {
                        copy_out(&resp.alpha, out_alpha, alpha_cap, out_alpha_len, "alpha")?;
                        copy_out(&table, out_props, props_cap, out_props_len, "props")?;
                        copy_out(&iters, out_iters, iters_cap, out_iters_len, "iters")?;
                    }
                    unsafe {
                        out_info.write(QnttSectorAlphaInfo {
                            alpha_defect: resp.alpha_defect,
                            sigma_min: resp.sigma_min,
                            dim: resp.dim,
                            phi: c_int::from(resp.phi.index()),
                            target: c_int::from(resp.target.index()),
                        })
                    };
                    Ok(())
                })
            }

            /// The channel's transition moments and oscillator strengths, DENSE route: the
            /// `(A^phi, B^phi)` Grams over the WHOLE shifted tangent basis, then a signed
            /// reduced-Williamson reduction.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _sector_transition_moments>](
                h_ref: *const Handle<$ham>,
                h_out: *const Handle<$ham>,
                x: *const Handle<$vec>,
                phi: u8,
                m: *const f64,
                m_len: usize,
                n_props: usize,
                norb: usize,
                labels: *const *const c_char,
                opts: *const QnttSectorMomentOpts,
                out_omega: *mut f64,
                omega_cap: usize,
                out_omega_len: *mut usize,
                out_moments: *mut f64,
                moments_cap: usize,
                out_moments_len: *mut usize,
                out_oscillator: *mut f64,
                osc_cap: usize,
                out_osc_len: *mut usize,
                out_alpha_sos: *mut f64,
                sos_cap: usize,
                out_sos_len: *mut usize,
                out_props: *mut f64,
                props_cap: usize,
                out_props_len: *mut usize,
                out_unstable_omega2: *mut f64,
                unstable_cap: usize,
                out_unstable_len: *mut usize,
                out_clusters: *mut usize,
                clusters_cap: usize,
                out_clusters_len: *mut usize,
                out_info: *mut QnttSectorMomentsInfo,
                out_refusal: *mut i32,
            ) -> c_int {
                catch(|| {
                    if out_refusal.is_null() {
                        return Err(Error::null("out_refusal"));
                    }
                    unsafe { out_refusal.write(QNTT_SECTOR_REFUSAL_NONE) };
                    let h_ref = unsafe { borrow(h_ref)? };
                    let h_out = unsafe { borrow(h_out)? };
                    let x = unsafe { borrow(x)? };
                    let o = unsafe { ptr_ref(opts, "opts")? };
                    if out_info.is_null() {
                        return Err(Error::null("out_info"));
                    }
                    let props = unsafe {
                        sector_obs_precheck(x.space.d, m, m_len, n_props, norb, labels)?
                    };
                    let o = o.to_moment()?;
                    sector_unit_reference(x.norm())?;
                    let phi = [<sector_channel_ $sfx>](h_ref, h_out, x, phi)?;
                    let ctx = $ctx(h_ref, x);
                    let sc = <$sector>::new(&ctx, h_out, phi);
                    let tm = match sc.transition_moments_dense(&props, &o) {
                        Ok(t) => t,
                        Err(e) => {
                            let (status, code) = sector_refusal_code(&e);
                            unsafe { out_refusal.write(code) };
                            return Err(Error::new(status, e.to_string()));
                        }
                    };
                    let n_modes = tm.omega.len();
                    let mut moments = Vec::with_capacity(n_props * n_modes);
                    for row in &tm.moments {
                        moments.extend_from_slice(row);
                    }
                    let mut table = Vec::with_capacity(5 * n_props);
                    for i in 0..n_props {
                        table.push(tm.capture_defect[i]);
                        table.push(tm.completeness_defect[i]);
                        table.push(tm.sum_rule[i].0);
                        table.push(tm.sum_rule[i].1);
                        table.push(tm.sum_rule_si[i]);
                    }
                    let sos: &[f64] = tm.alpha_sos.as_deref().unwrap_or(&[]);
                    let mut clusters = Vec::with_capacity(2 * tm.degenerate_clusters.len());
                    for &(lo, hi) in &tm.degenerate_clusters {
                        clusters.push(lo);
                        clusters.push(hi);
                    }
                    unsafe {
                        copy_out(&tm.omega, out_omega, omega_cap, out_omega_len, "omega")?;
                        copy_out(&moments, out_moments, moments_cap, out_moments_len, "moments")?;
                        copy_out(
                            &tm.oscillator,
                            out_oscillator,
                            osc_cap,
                            out_osc_len,
                            "oscillator",
                        )?;
                        copy_out(sos, out_alpha_sos, sos_cap, out_sos_len, "alpha_sos")?;
                        copy_out(&table, out_props, props_cap, out_props_len, "props")?;
                        copy_out(
                            &tm.unstable_omega2,
                            out_unstable_omega2,
                            unstable_cap,
                            out_unstable_len,
                            "unstable_omega2",
                        )?;
                        copy_out(
                            &clusters,
                            out_clusters,
                            clusters_cap,
                            out_clusters_len,
                            "clusters",
                        )?;
                    }
                    let metric = if n_modes == 0 && tm.dim == 0 {
                        QNTT_RPA_METRIC_AUTO
                    } else {
                        match tm.metric {
                            RpaMetricBlock::Ii => QNTT_RPA_METRIC_II,
                            RpaMetricBlock::Rr => QNTT_RPA_METRIC_RR,
                        }
                    };
                    unsafe {
                        out_info.write(QnttSectorMomentsInfo {
                            orth_error: tm.orth_error,
                            sigma_min: tm.sigma_min,
                            dim: tm.dim,
                            n_modes,
                            n_neg: tm.n_neg,
                            n_clusters: tm.degenerate_clusters.len(),
                            metric,
                            has_alpha_sos: c_int::from(tm.alpha_sos.is_some()),
                            complete_basis: c_int::from(tm.complete_basis),
                            phi: c_int::from(tm.phi.index()),
                            target: c_int::from(tm.target.index()),
                            reserved: 0,
                        })
                    };
                    Ok(())
                })
            }
        }
    };
}

export_sector_obs!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    sector = qntt::solvers::hess::SectorCtx<'_, '_, f64>
);

#[cfg(test)]
mod tests {
    use super::property_block_len;

    #[test]
    fn property_block_len_exact_zero_and_overflow() {
        assert_eq!(property_block_len(2 * 3 * 3, 2, 3).unwrap(), 9);
        let short = property_block_len(17, 2, 3).unwrap_err().msg;
        assert!(short.contains("EXACT equality"), "{short}");
        let zero = property_block_len(0, 1, 0).unwrap_err().msg;
        assert!(zero.contains("norb = 0"), "{zero}");
        let over = property_block_len(0, 1, usize::MAX / 2).unwrap_err().msg;
        assert!(over.contains("overflows"), "{over}");
        let over2 = property_block_len(0, usize::MAX, 2).unwrap_err().msg;
        assert!(over2.contains("overflows"), "{over2}");
    }
}
