// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) TDVP driver — two-site projector-splitting TDVP, imaginary + real time. Mirrors the
//! abelian `crate::solvers::tdvp` line-for-line with SU(2) type substitutions; every abelian
//! choice is carried: `EXPMV_MAXDIM = 30`, `report_unconverged`, the `T::IS_COMPLEX` runtime
//! assert standing in for a `T: Complex` bound, the `normalize_center` zero-norm guard, the
//! `d==1` `ep` guard, and the single-substep `expmv` (no adaptive time-substepping).

use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::solvers::expmv::{ExpmvInfo, expmv};
pub use crate::solvers::tdvp::{RealtimeOptions, TdvpOptions};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::matvec::{
    Su2TwoSiteWorkspace, su2_onesite_workspace, su2_twosite_workspace,
};
use crate::su2::hamiltonian::rayleigh_quotient;
use crate::su2::solvers::als::{boundary_idsketch, right_to_left_sketching};
use crate::su2::twosite::{contract, factor_svd};
use crate::su2::vector::Su2TtVector;

const EXPMV_MAXDIM: usize = 30;

#[inline]
fn report_unconverged(realtime: bool, info: &ExpmvInfo, site: &str) {
    if realtime && !info.converged {
        eprintln!(
            "su2 tdvp_realtime: {site} expmv did not reach exptol (krylovdim {})",
            info.krylovdim
        );
    }
}

fn normalize_center<T: Scalar>(x: &mut Su2TtVector<T>) {
    let n = x.norm();
    if n != 0.0 {
        x.scale_mut(T::from_f64(1.0 / n));
    }
}

/// Imaginary-time two-site TDVP ground-state search on the SU(2) layer. Alternating forward/back
/// second-order symmetric sweeps of imaginary-time size `Δτ`, renormalized after each sweep, until
/// the relative Rayleigh-quotient change drops below `opts.eps`. Returns `(λ, x)` with `x`
/// unit-norm and `x.ortho == Some(0)`.
pub fn tdvp<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &TdvpOptions,
) -> (f64, Su2TtVector<T>) {
    let mut x = x0.clone();
    let d = x.space.d;
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Su2Sketch<T>>> = right_to_left_sketching(h, &x)
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
            println!("SU(2) TDVP sweep {it}: λ={lambda}, Δ={r}");
        }
        if r < opts.eps {
            break;
        }
    }
    (lambda, x)
}

#[allow(clippy::too_many_arguments)]
fn tdvp_forward_sweep<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fr: &mut [Option<Su2Sketch<T>>],
    a: T,
    ep: f64,
    chi: Pooled,
    exptol: f64,
    normalize: bool,
) -> Vec<Option<Su2Sketch<T>>> {
    debug_assert_eq!(x.ortho, Some(0), "TDVP forward sweep expects center 0");
    let d = x.space.d;
    let mut fl: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    fr[0].take();

    for k in 0..(d - 1) {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let cp = {
            let fl_k = fl[k].as_ref().unwrap();
            let fr_k1 = fr[k + 1].as_ref().unwrap();
            let mut ws: Option<Su2TwoSiteWorkspace<T>> = su2_twosite_workspace();
            let (cp, info) = expmv(
                |c| h.two_site_sketched_hamiltonian_ws(&mut ws, c, fl_k, fr_k1),
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

        let ws_env = su2_onesite_workspace(h, x.core(k));
        let next = h.sketching_step_right(x, fl[k].as_ref().unwrap(), &ws_env);
        fl[k + 1] = Some(next);

        if k < d - 2 {
            let y0 = x.core(k + 1).clone();
            let ws1 = su2_onesite_workspace(h, &y0);
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
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fl: &mut [Option<Su2Sketch<T>>],
    a: T,
    ep: f64,
    chi: Pooled,
    exptol: f64,
    normalize: bool,
) -> Vec<Option<Su2Sketch<T>>> {
    let d = x.space.d;
    debug_assert_eq!(x.ortho, Some(d - 1), "TDVP back sweep expects center d-1");
    let mut fr: Vec<Option<Su2Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    fl[d - 1].take();

    for k in (0..(d - 1)).rev() {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let cp = {
            let fl_k = fl[k].as_ref().unwrap();
            let fr_k1 = fr[k + 1].as_ref().unwrap();
            let mut ws: Option<Su2TwoSiteWorkspace<T>> = su2_twosite_workspace();
            let (cp, info) = expmv(
                |c| h.two_site_sketched_hamiltonian_ws(&mut ws, c, fl_k, fr_k1),
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

        let ws_env = su2_onesite_workspace(h, x.core(k + 1));
        let next = h.sketching_step_left(x, fr[k + 1].as_ref().unwrap(), &ws_env);
        fr[k] = Some(next);

        if k > 0 {
            let y0 = x.core(k).clone();
            let ws1 = su2_onesite_workspace(h, &y0);
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

/// Real-time two-site TDVP on the SU(2) layer (abelian twin
/// `crate::solvers::tdvp::tdvp_realtime`): unitary evolution `x(t) = exp(-i·t·H)·x0`; `T` must
/// be complex. Asserts `T::IS_COMPLEX` at entry (a `T: Complex` bound is inexpressible on a
/// `Scalar`-generic fn —
/// **at `T = f64`, `T::from_complex_f64(0.0, -dt/2)` evaluates to exactly `0.0` and every `expmv`
/// silently no-ops**; the assert is the only thing standing between the user and that). Returns
/// `(t_final, x)`.
#[allow(clippy::type_complexity)]
pub fn tdvp_realtime<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &RealtimeOptions,
    mut observer: Option<&mut dyn FnMut(usize, f64, &Su2TtVector<T>)>,
) -> (f64, Su2TtVector<T>) {
    assert!(
        T::IS_COMPLEX,
        "tdvp_realtime requires a complex scalar type (a = -i·dt/2)"
    );
    let mut x = x0.clone();
    let d = x.space.d;
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Su2Sketch<T>>> = right_to_left_sketching(h, &x)
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
            println!(
                "SU(2) RT-TDVP step {step}: t={t}, ‖x‖={}, ⟨H⟩={:?}",
                x.norm(),
                e
            );
        }
    }
    (t, x)
}

#[cfg(test)]
mod ws_expmv_tests {
    use super::*;
    use crate::scalar::C64;
    use crate::su2::qn::Su2Space;
    use crate::su2::twosite::Su2ContractedCores;
    use crate::test_util::{generic_su2_h, randomized_su2_state, seeded};

    fn bits<T: Scalar>(c: &Su2ContractedCores<T>) -> Vec<(u64, u64)> {
        let mut v = Vec::new();
        c.clone()
            .for_each_coeff_mut(|z| v.push((z.re_s().to_bits(), z.im_s().to_bits())));
        v
    }

    fn check<T: Scalar>(a: T, seed: u64) {
        let space = Su2Space::new(6, 4, 0);
        let h = generic_su2_h::<T>(space);
        let mut rng = seeded(seed);
        let mut x = randomized_su2_state::<T, _>(space, 2, 4, &mut rng);
        x.right_orthogonalize(true);
        let d = space.d;
        let fr: Vec<Su2Sketch<T>> = right_to_left_sketching(&h, &x);
        let mut fl = vec![boundary_idsketch(&x, 0, 1)];
        for k in 0..d - 2 {
            let ws = su2_onesite_workspace(&h, x.core(k));
            let next = h.sketching_step_right(&x, &fl[k], &ws);
            fl.push(next);
        }
        for k in 0..d - 1 {
            let cp0 = contract(x.core(k), x.core(k + 1));
            let (want, iw) = expmv(
                |c| h.two_site_sketched_hamiltonian(c, &fl[k], &fr[k + 1]),
                a,
                &cp0,
                1e-12,
                EXPMV_MAXDIM,
            );
            let mut ws: Option<Su2TwoSiteWorkspace<T>> = su2_twosite_workspace();
            let (got, ig) = expmv(
                |c| h.two_site_sketched_hamiltonian_ws(&mut ws, c, &fl[k], &fr[k + 1]),
                a,
                &cp0,
                1e-12,
                EXPMV_MAXDIM,
            );
            assert!(iw.krylovdim >= 2, "bond {k}: vacuous Krylov run");
            assert_eq!(iw.krylovdim, ig.krylovdim, "bond {k}: Krylov dimension");
            assert!(want.norm() > 1e-8, "bond {k}: vacuous output");
            assert_eq!(
                bits(&want),
                bits(&got),
                "bond {k}: ws expmv not bit-identical"
            );
        }
    }

    #[test]
    fn ws_two_site_expmv_is_bit_identical() {
        check::<f64>(-0.25, 0x7d_0001);
        check::<C64>(C64::new(0.0, -0.25), 0x7d_0002);
    }
}
