// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Positive-definiteness probes of the Riemannian Hessian, and the complex seed built from one.

use std::os::raw::c_int;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::solvers::hess::PdVerdict;
use qntt::{C64, Scalar};

use crate::enums;
use crate::err::{Error, QNTT_NOT_CONVERGED, catch, ptr_mut};
use crate::handle::{Handle, borrow, emit};

fn payload(v: &PdVerdict) -> (f64, f64) {
    match *v {
        PdVerdict::WitnessNegative { lambda } => (lambda, f64::NAN),
        PdVerdict::LikelyPd {
            lambda_min,
            residual,
        }
        | PdVerdict::Inconclusive {
            lambda_min,
            residual,
        } => (lambda_min, residual),
    }
}

fn err_ii_over_c64() -> Error {
    Error::invalid(
        "over C64 there is one Hessian, 𝔥, reached with part = 0 (rr); `−J𝔥J ≻ 0 ⟺ 𝔥 ≻ 0`, so \
         that single probe certifies the RPA precondition and part = 1 (ii) has no complex meaning",
    )
}

fn err_no_witness(v: &PdVerdict, budget: usize) -> Error {
    let msg = match *v {
        PdVerdict::LikelyPd {
            lambda_min,
            residual,
        } => format!(
            "complex seed: the H^ii probe found no negative direction (LikelyPd, \
             λ_min = {lambda_min:.6e}, residual = {residual:.3e}). There is nothing to complexify \
             along — this state is stable against complex perturbations at this rank."
        ),
        PdVerdict::Inconclusive {
            lambda_min,
            residual,
        } => format!(
            "complex seed: the H^ii probe exhausted its budget of {budget} iterations with \
             residual {residual:.3e} on a Ritz value of {lambda_min:.6e}. That is NOT evidence of \
             stability — rerun with a larger budget."
        ),
        PdVerdict::WitnessNegative { lambda } => format!(
            "complex seed: internal error — a witness verdict (λ = {lambda:.6e}) reached the \
             no-witness path"
        ),
    };
    Error::new(QNTT_NOT_CONVERGED, msg)
}

fn err_zero_witness(wn: f64) -> String {
    format!(
        "complex seed: the witness reprojected to norm {wn} on the complex tangent space, so no \
         step can be built from it"
    )
}

unsafe fn emit_verdict(
    v: &PdVerdict,
    out_verdict: *mut i32,
    out_lambda: *mut f64,
    out_residual: *mut f64,
    out_is_pd: *mut i32,
) -> Result<(), Error> {
    let (lambda, residual) = payload(v);
    unsafe {
        *ptr_mut(out_verdict, "out_verdict")? = enums::pd_verdict_code(v);
        if !out_lambda.is_null() {
            *out_lambda = lambda;
        }
        if !out_residual.is_null() {
            *out_residual = residual;
        }
        if !out_is_pd.is_null() {
            *out_is_pd = enums::pd_is_pd_code(v);
        }
    }
    Ok(())
}

macro_rules! export_probe {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, tangent = $tangent:ty,
        witness = $witness:path, randn = $randn:path, normalize = $normalize:path
    ) => {
        paste::paste! {
            /// The REAL saddle seed `normalize(retract(x, ε·ŵ))` along the `H^rr` witness at
            /// `x` — the launch point of a GAD run. Probes `H^rr = A + B` with `budget` Davidson
            /// iterations from a `seed`-determined random tangent; the witness is the proven
            /// negative-curvature direction (at a ground state, the softest mode of the
            /// deflated Hessian is used whether or not it is negative — see below). It is
            /// normalized, retracted with step `eps` onto `x`'s own rank grid and normalized
            /// again. `out_lambda` (optional) receives the witness's Rayleigh quotient.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _saddle_seed>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                eps: f64,
                budget: usize,
                seed: u64,
                out_lambda: *mut f64,
                out: *mut *mut Handle<$vec>,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    if !eps.is_finite() || eps <= 0.0 {
                        return Err(Error::invalid(format!(
                            "saddle seed: eps must be finite and positive, got {eps}"
                        )));
                    }
                    let ctx = $ctx(h, x);
                    let (base_l, base_r) = ctx.bases();
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let dir = $randn(x.space, &x.ranks, &mut rng);
                    let v0 = <$tangent>::project(base_l, base_r, &dir);
                    let (verdict, mut w, _applies) = $witness(
                        &ctx, qntt::solvers::hess::HessPart::Rr, &v0, budget,
                    );
                    let (lambda, _residual) = payload(&verdict);
                    let wn = w.norm();
                    if !wn.is_finite() || wn <= 0.0 {
                        return Err(Error::new(
                            QNTT_NOT_CONVERGED,
                            format!("saddle seed: the witness has norm {wn}; no step can be built"),
                        ));
                    }
                    w.scale_mut(1.0 / wn);
                    let mut y = w.retract_hosvd(eps);
                    $normalize(&mut y);
                    unsafe {
                        if !out_lambda.is_null() {
                            *out_lambda = lambda;
                        }
                        emit(y, out)
                    }
                })
            }

            /// Smallest-eigenvalue probe of the radially-deflated `H^rr` (`part = 0`) or `H^ii`
            /// (`part = 1`) at `x`. `seed` fixes the random starting tangent direction (drawn at
            /// `x`'s own ranks). The status stays `QNTT_OK` for every verdict — see the module
            /// docs in `capi/src/probe.rs`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _pd_probe>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                part: i32,
                budget: usize,
                seed: u64,
                out_verdict: *mut i32,
                out_lambda: *mut f64,
                out_residual: *mut f64,
                out_is_pd: *mut i32,
                out_applies: *mut usize,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    let part = enums::hess_part(part)?;
                    let ctx = $ctx(h, x);
                    let (base_l, base_r) = ctx.bases();
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let dir = $randn(x.space, &x.ranks, &mut rng);
                    let v0 = <$tangent>::project(base_l, base_r, &dir);
                    let (verdict, _w, applies) = $witness(&ctx, part, &v0, budget);
                    unsafe {
                        emit_verdict(&verdict, out_verdict, out_lambda, out_residual, out_is_pd)?;
                        if !out_applies.is_null() {
                            *out_applies = applies;
                        }
                    }
                    Ok(())
                })
            }
        }
    };
}

macro_rules! export_probe_c64 {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty,
        ctx = $ctx:path, tangent = $tangent:ty,
        witness = $witness:path, randn = $randn:path
    ) => {
        paste::paste! {
            /// Smallest-eigenvalue probe of the radially-deflated COMPLEX Riemannian Hessian `𝔥`
            /// at `x`. `part` must be `QNTT_HESS_RR` (`0`): over `C64`, `Rr` *is* `𝔥`, and one
            /// probe certifies the whole RPA precondition. `QNTT_HESS_II` is `QNTT_INVALID_ARG`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _pd_probe>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                part: i32,
                budget: usize,
                seed: u64,
                out_verdict: *mut i32,
                out_lambda: *mut f64,
                out_residual: *mut f64,
                out_is_pd: *mut i32,
                out_applies: *mut usize,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    match enums::hess_part(part)? {
                        qntt::solvers::hess::HessPart::Rr => {}
                        qntt::solvers::hess::HessPart::Ii => return Err(err_ii_over_c64()),
                    }
                    let ctx = $ctx(h, x);
                    let (base_l, base_r) = ctx.bases();
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                    let dir = $randn(x.space, &x.ranks, &mut rng);
                    let v0 = <$tangent>::project(base_l, base_r, &dir);
                    let (verdict, _w, applies) = $witness(&ctx, &v0, budget);
                    unsafe {
                        emit_verdict(&verdict, out_verdict, out_lambda, out_residual, out_is_pd)?;
                        if !out_applies.is_null() {
                            *out_applies = applies;
                        }
                    }
                    Ok(())
                })
            }
        }
    };
}

macro_rules! export_complexify {
    (
        $sfx:ident,
        ham = $ham:ty, vec = $vec:ty, vec_c = $vecc:ty,
        ctx = $ctx:path, tangent = $tangent:ty, tangent_c = $tangentc:ty,
        witness = $witness:path, randn = $randn:path,
        bases_c = $basesc:path, normalize_c = $normc:path
    ) => {
        paste::paste! {
            /// Real → complex embedding of a state: same space, same rank grid, every coefficient
            /// in the real part. Linear in the coefficient count. The orthogonality centre is
            /// carried over — it stays true, since a real column-orthonormal frame is orthonormal
            /// under the Hermitian inner product — so the cast train keeps the centre-core norm
            /// shortcut instead of paying a sweep to rediscover it.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _vec_to_c64>](
                x: *const Handle<$vec>,
                out: *mut *mut Handle<$vecc>,
            ) -> c_int {
                catch(|| {
                    let x = unsafe { borrow(x)? };
                    unsafe { emit(x.to_c64(), out) }
                })
            }

            /// Build the complex seed `normalize(retract(x, i·ε·ŵ))` from the `H^ii` witness at
            /// the REAL state `x` — the step that leaves the real manifold.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            #[allow(clippy::too_many_arguments)]
            pub unsafe extern "C" fn [<qntt_ $sfx _complex_seed>](
                h: *const Handle<$ham>,
                x: *const Handle<$vec>,
                eps: f64,
                budget: usize,
                seed: u64,
                out_lambda: *mut f64,
                out: *mut *mut Handle<$vecc>,
            ) -> c_int {
                catch(|| {
                    let h = unsafe { borrow(h)? };
                    let x = unsafe { borrow(x)? };
                    if !eps.is_finite() || eps <= 0.0 {
                        return Err(Error::invalid(format!(
                            "complex seed: eps must be finite and positive, got {eps}"
                        )));
                    }
                    let (verdict, w) = {
                        let ctx = $ctx(h, x);
                        let (base_l, base_r) = ctx.bases();
                        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                        let dir = $randn(x.space, &x.ranks, &mut rng);
                        let v0 = <$tangent>::project(base_l, base_r, &dir);
                        let (verdict, w, _applies) = $witness(
                            &ctx, qntt::solvers::hess::HessPart::Ii, &v0, budget,
                        );
                        (verdict, w)
                    };
                    let lambda = match verdict {
                        PdVerdict::WitnessNegative { lambda } => lambda,
                        ref v => return Err(err_no_witness(v, budget)),
                    };
                    let x_c = x.to_c64();
                    let (bl, br) = $basesc(&x_c);
                    let mut w_c = <$tangentc>::project(&bl, &br, &w.to_vector().to_c64());
                    let wn = w_c.norm();
                    if !wn.is_finite() || wn <= 0.0 {
                        return Err(Error::new(QNTT_NOT_CONVERGED, err_zero_witness(wn)));
                    }
                    w_c.scale_mut(C64::from_f64(1.0 / wn));
                    let mut y = w_c.retract_hosvd(C64::new(0.0, eps));
                    $normc(&mut y);
                    unsafe {
                        if !out_lambda.is_null() {
                            *out_lambda = lambda;
                        }
                        emit(y, out)
                    }
                })
            }
        }
    };
}

export_probe!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    tangent = qntt::core::tangent::TtTangent<f64>,
    witness = qntt::solvers::hess::hessian_pd_probe_witness,
    randn = qntt::tt_randn,
    normalize = qntt::solvers::riemannian::normalize_mut
);

export_probe_c64!(
    c64,
    ham = qntt::SparseHamiltonian<C64>,
    vec = qntt::TtVector<C64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    tangent = qntt::core::tangent::TtTangent<C64>,
    witness = qntt::solvers::hess::hessian_pd_probe_c64_witness,
    randn = qntt::tt_randn
);

export_probe!(
    su2_f64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    vec = qntt::su2::vector::Su2TtVector<f64>,
    ctx = qntt::su2::hess::weingarten_operator,
    tangent = qntt::su2::tangent::Su2TtTangent<f64>,
    witness = qntt::su2::hess::hessian_pd_probe_witness,
    randn = qntt::su2::vector::tt_randn,
    normalize = qntt::su2::riemannian::normalize_mut
);

export_probe_c64!(
    su2_c64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<C64>,
    vec = qntt::su2::vector::Su2TtVector<C64>,
    ctx = qntt::su2::hess::weingarten_operator,
    tangent = qntt::su2::tangent::Su2TtTangent<C64>,
    witness = qntt::su2::hess::hessian_pd_probe_c64_witness,
    randn = qntt::su2::vector::tt_randn
);

export_complexify!(
    f64,
    ham = qntt::SparseHamiltonian<f64>,
    vec = qntt::TtVector<f64>,
    vec_c = qntt::TtVector<C64>,
    ctx = qntt::solvers::hess::weingarten_operator,
    tangent = qntt::core::tangent::TtTangent<f64>,
    tangent_c = qntt::core::tangent::TtTangent<C64>,
    witness = qntt::solvers::hess::hessian_pd_probe_witness,
    randn = qntt::tt_randn,
    bases_c = qntt::solvers::riemannian::tangent_bases,
    normalize_c = qntt::solvers::riemannian::normalize_mut
);

export_complexify!(
    su2_f64,
    ham = qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    vec = qntt::su2::vector::Su2TtVector<f64>,
    vec_c = qntt::su2::vector::Su2TtVector<C64>,
    ctx = qntt::su2::hess::weingarten_operator,
    tangent = qntt::su2::tangent::Su2TtTangent<f64>,
    tangent_c = qntt::su2::tangent::Su2TtTangent<C64>,
    witness = qntt::su2::hess::hessian_pd_probe_witness,
    randn = qntt::su2::vector::tt_randn,
    bases_c = qntt::su2::riemannian::tangent_bases,
    normalize_c = qntt::su2::riemannian::normalize_mut
);
