// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::solvers::hess::{CurvatureReg, RpaKernel, RpaMetric, RpaReport};
use qntt::solvers::joint::cas::{cas_pairs, cas_reference_state};
use qntt::solvers::joint::{JointScfOpts, joint_ground_state};
use qntt_capi::err::QNTT_NULL_ARG;
use qntt_capi::gad::*;
use qntt_capi::handle::{Handle, borrow};
use qntt_capi::joint::*;
use qntt_capi::rpa::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const PART: QnttCasPartition = QnttCasPartition {
    n_core: 1,
    n_active: 2,
    n_virt: 1,
};

fn assert_orthogonal(u: &[f64], d: usize) {
    for i in 0..d {
        for j in 0..d {
            let dot: f64 = (0..d).map(|k| u[k * d + i] * u[k * d + j]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((dot - want).abs() < 1e-10, "uᵀu[{i},{j}] = {dot}");
        }
    }
}

struct RpaSignedOut {
    rc: i32,
    omega2: Vec<f64>,
    fw: Vec<f64>,
    residual: Vec<f64>,
    krein: Vec<f64>,
    complex: Vec<f64>,
    metric: i32,
    cert: i32,
    conv: i32,
    iters: usize,
    applies: usize,
    kernel: i32,
    report: Vec<i64>,
}

macro_rules! rpa_signed_call {
    ($f:ident, $h:expr, $x:expr, $k:expr, $opts:expr, $metric:expr) => {
        rpa_signed_call!($f, $h, $x, $k, $opts, $metric, QNTT_CURVATURE_REG_OFF, 0.0)
    };
    ($f:ident, $h:expr, $x:expr, $k:expr, $opts:expr, $metric:expr, $ckind:expr, $cval:expr) => {
        rpa_signed_call!(
            $f,
            $h,
            $x,
            $k,
            $opts,
            $metric,
            $ckind,
            $cval,
            QNTT_RPA_KERNEL_PAIRED
        )
    };
    ($f:ident, $h:expr, $x:expr, $k:expr, $opts:expr, $metric:expr, $ckind:expr, $cval:expr,
     $kernel:expr) => {{
        let n_max = ($opts).max_iter.max($k + 2);
        let (mut w2, mut fw, mut res) = (vec![0.0; n_max], vec![0.0; n_max], vec![0.0; n_max]);
        let (mut krein, mut cx) = (vec![0.0; n_max], vec![0.0; 3 * n_max]);
        let mut report = vec![i64::MIN; 5];
        let (mut w2len, mut fwlen, mut reslen, mut kreinlen, mut cxlen, mut replen) = (
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX,
        );
        let (mut metric, mut cert, mut conv, mut kernel) = (-1i32, -1i32, -1i32, -1i32);
        let (mut iters, mut applies) = (usize::MAX, usize::MAX);
        let rc = unsafe {
            $f(
                $h,
                $x,
                $k,
                $opts,
                $kernel,
                $metric,
                $ckind,
                $cval,
                1,
                w2.as_mut_ptr(),
                w2.len(),
                &mut w2len,
                fw.as_mut_ptr(),
                fw.len(),
                &mut fwlen,
                res.as_mut_ptr(),
                res.len(),
                &mut reslen,
                krein.as_mut_ptr(),
                krein.len(),
                &mut kreinlen,
                cx.as_mut_ptr(),
                cx.len(),
                &mut cxlen,
                &mut metric,
                &mut cert,
                &mut conv,
                &mut iters,
                &mut applies,
                &mut kernel,
                report.as_mut_ptr(),
                report.len(),
                &mut replen,
            )
        };
        let cut = |n: usize, cap: usize| if n == usize::MAX { 0 } else { n.min(cap) };
        RpaSignedOut {
            rc,
            omega2: w2[..cut(w2len, n_max)].to_vec(),
            fw: fw[..cut(fwlen, n_max)].to_vec(),
            residual: res[..cut(reslen, n_max)].to_vec(),
            krein: krein[..cut(kreinlen, n_max)].to_vec(),
            complex: cx[..cut(cxlen, 3 * n_max)].to_vec(),
            metric,
            cert,
            conv,
            iters,
            applies,
            kernel,
            report: report[..cut(replen, 5)].to_vec(),
        }
    }};
}

fn check_rpa_signed(out: &RpaSignedOut, k: usize, requested: i32, kernel: i32, layer: &str) {
    assert_eq!(
        out.rc,
        QNTT_OK,
        "{layer} metric {requested}: {}",
        last_error()
    );
    let n = out.omega2.len();
    assert!(n > 0, "{layer} metric {requested}: no mode returned");
    assert!(
        out.omega2.windows(2).all(|w| w[0] <= w[1]),
        "{layer}: not ascending"
    );
    assert!(out.omega2.iter().filter(|w| **w >= 0.0).count() <= k);
    assert_eq!(out.fw.len(), n, "{layer}: one f_W per mode");
    assert_eq!(out.residual.len(), n, "{layer}: one residual per mode");
    assert_eq!(out.krein.len(), n, "{layer}: one Krein margin per mode");
    assert!(out.fw.iter().all(|f| *f >= 0.0));
    assert!(out.residual.iter().all(|r| r.is_finite() && *r >= 0.0));
    assert!(
        out.krein
            .iter()
            .all(|m| m.is_finite() && m.abs() <= 1.0 + 1e-12)
    );
    assert!(
        out.omega2
            .iter()
            .zip(&out.krein)
            .all(|(w, m)| *w >= 0.0 || *m == 0.0),
        "{layer}: an imaginary mode is Krein-neutral"
    );
    assert_eq!(
        out.complex.len() % 3,
        0,
        "{layer}: complex rows are (re, im, residual)"
    );
    assert!(out.metric == QNTT_RPA_METRIC_II || out.metric == QNTT_RPA_METRIC_RR);
    if requested != QNTT_RPA_METRIC_AUTO {
        assert_eq!(out.metric, requested, "{layer}: a forced metric is echoed");
    }
    let lane = if requested == QNTT_RPA_METRIC_RR {
        QNTT_RPA_METRIC_RR
    } else {
        QNTT_RPA_METRIC_II
    };
    assert_eq!(out.metric, lane, "{layer} kernel {kernel}: AUTO is II");
    assert_eq!(
        out.kernel, kernel,
        "{layer}: the kernel echo is the request"
    );
    assert!(out.cert == 0 || out.cert == 1);
    assert!(out.conv == 0 || out.conv == 1);
    assert!(out.iters != usize::MAX && out.applies > 0 && out.applies != usize::MAX);
    assert_eq!(out.report.len(), 5, "{layer}: the report has five entries");
    let clean = match out.kernel {
        QNTT_RPA_KERNEL_PAIRED => {
            assert!(
                out.report[0] >= 0 && out.report[1] >= 0,
                "{layer}: {:?}",
                out.report
            );
            assert_eq!(
                &out.report[2..],
                &[-1, -1, -1],
                "{layer}: no signed entries"
            );
            out.report[0] + out.report[1] == 0
        }
        QNTT_RPA_KERNEL_SIGNED => {
            assert_eq!(&out.report[..2], &[-1, -1], "{layer}: no paired entries");
            assert!(out.report[2..].iter().all(|&r| r >= 0) && out.report[4] <= 1);
            out.report[2] == 0
        }
        other => panic!("{layer}: kernel code {other}"),
    };
    if clean {
        assert!(
            out.omega2
                .iter()
                .zip(&out.krein)
                .all(|(w, m)| *w < 0.0 || *m > 0.0),
            "{layer}: a clean report with a de-excitation"
        );
        assert!(out.complex.is_empty());
        if out.kernel == QNTT_RPA_KERNEL_SIGNED {
            assert_eq!(&out.report[3..], &[0, 0]);
        }
    }
}

fn check_rpa_signed_bad_metric(out: &RpaSignedOut, layer: &str) {
    assert_eq!(out.rc, QNTT_INVALID_ARG, "{layer}");
    assert!(
        last_error().contains("RPA metric"),
        "{layer}: {}",
        last_error()
    );
    assert!(
        out.omega2.is_empty() && out.applies == usize::MAX,
        "{layer}: no solve ran"
    );
}

#[test]
fn abelian_casscf_chain_matches_the_rust_api() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
    let d = space.d;

    let path = CString::new(h4_path().to_str().unwrap()).unwrap();
    let mut ints: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read(path.as_ptr(), 2, &mut ints) },
        QNTT_OK
    );
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new(
                d,
                space.nup,
                space.ndn,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK
    );

    let mut lambda0 = f64::NAN;
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_cas_reference_state(h, &PART, 0, 7, &mut lambda0, &mut x0) },
        QNTT_OK,
        "{}",
        last_error()
    );
    let part = PART.to_core();
    let direct = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let hr = SparseHamiltonian::new(space, &direct.t, &direct.v, None, f64::EPSILON);
    let (lambda0_r, x0_r) = cas_reference_state(&hr, &space, &part, 0, 7);
    assert!(
        (lambda0 - lambda0_r).abs() < 1e-10,
        "reference λ: abi {lambda0} vs rust {lambda0_r}"
    );

    let mut opts = QnttCasscfOptions::defaults();
    assert_eq!(opts.tt_preserve_grid, 1, "the pinned default must be ON");
    opts.orb_max_iter = 20;
    let mut lambda = f64::NAN;
    let mut x: *mut _ = ptr::null_mut();
    let mut h_rot: *mut _ = ptr::null_mut();
    let mut u = vec![0.0; d * d];
    let mut u_len = 0usize;
    let mut info = QnttCasscfInfo::default();
    let mut diag: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_casscf(
                ints,
                x0,
                &PART,
                &opts,
                &mut lambda,
                &mut x,
                &mut h_rot,
                u.as_mut_ptr(),
                u.len(),
                &mut u_len,
                &mut info,
                &mut diag,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert_eq!(u_len, d * d);
    assert_orthogonal(&u, d);
    assert!(
        lambda <= lambda0 + 1e-12,
        "the SCF must not raise the energy"
    );

    let rust_opts = JointScfOpts {
        orb_max_iter: 20,
        orb_grad_full: Some(1e-6),
        construction: qntt::hamiltonian::Construction::Median,
        pairs: Some(cas_pairs(&part, d)),
        tt_preserve_grid: true,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&direct, space, &x0_r, &rust_opts);
    assert_eq!(info.converged != 0, out.info.converged);
    assert_eq!(info.iters, out.info.iters);
    assert!(
        (lambda - out.lambda).abs() < 1e-9,
        "casscf λ: abi {lambda} vs rust {}",
        out.lambda
    );

    let mut n = 0usize;
    assert_eq!(
        unsafe { qntt_casscf_diag_series(diag, 1, ptr::null_mut(), 0, &mut n) },
        QNTT_OK
    );
    assert_eq!(n, out.info.g_orb_kept.len());
    let mut g = vec![0.0; n];
    assert_eq!(
        unsafe { qntt_casscf_diag_series(diag, 1, g.as_mut_ptr(), n, &mut n) },
        QNTT_OK
    );
    assert_eq!(
        unsafe { qntt_casscf_diag_series(diag, 99, ptr::null_mut(), 0, &mut n) },
        QNTT_INVALID_ARG
    );
    let mut rlen = 0usize;
    assert_eq!(
        unsafe { qntt_casscf_diag_reason(diag, ptr::null_mut(), 0, &mut rlen) },
        QNTT_OK
    );
    assert_eq!(
        rlen,
        out.info.refused_reason.as_ref().map_or(0, |s| s.len())
    );

    let mut omega = vec![0.0; 64];
    let mut fw = vec![0.0; 64];
    let (mut olen, mut fwlen) = (0usize, 0usize);
    let mut status = -1i32;
    let (mut n_modes, mut n_eff) = (0usize, 0usize);
    let mut msg = vec![0i8; 8];
    let mut mlen = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_joint_rpa(
                h_rot,
                x,
                &PART,
                1e-6,
                1e-8,
                omega.as_mut_ptr(),
                omega.len(),
                &mut olen,
                fw.as_mut_ptr(),
                fw.len(),
                &mut fwlen,
                &mut status,
                &mut n_modes,
                &mut n_eff,
                msg.as_mut_ptr(),
                msg.len(),
                &mut mlen,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert!((0..=3).contains(&status));
    assert_eq!(msg[7], 0);
    assert!(mlen > 7);
    if info.converged != 0 {
        assert_ne!(
            status, QNTT_JOINT_RPA_REFUSED,
            "a certified endpoint passes the gate"
        );
        assert_eq!(n_eff, info.n_eff);
        assert!(n_modes >= n_eff);
        if status == QNTT_JOINT_RPA_STABLE {
            assert!(olen > 0 && omega[..olen].windows(2).all(|w| w[0] <= w[1]));
            assert!(omega[0] > 0.0);
            assert_eq!(fwlen, olen);
            assert!(fw[..fwlen].iter().all(|f| *f >= 0.0));
        }
    }

    let sopts = QnttSymplecticLowestOpts {
        max_iter: 800,
        tol: 1e-6,
    };
    for metric in [QNTT_RPA_METRIC_AUTO, QNTT_RPA_METRIC_II, QNTT_RPA_METRIC_RR] {
        let out = rpa_signed_call!(qntt_f64_rpa_signed, h_rot, x, 3, &sopts, metric);
        check_rpa_signed(&out, 3, metric, QNTT_RPA_KERNEL_PAIRED, "f64");
    }
    let bad = rpa_signed_call!(qntt_f64_rpa_signed, h_rot, x, 3, &sopts, 7);
    check_rpa_signed_bad_metric(&bad, "f64");

    let mut lam_w = f64::NAN;
    let mut xk: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_saddle_seed(h_rot, x, 5e-2, 40, 3, &mut lam_w, &mut xk) },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert!(lam_w.is_finite());
    let mut g0 = f64::NAN;
    assert_eq!(unsafe { qntt_f64_grad_norm(h_rot, xk, &mut g0) }, QNTT_OK);
    assert!(
        g0 > 1e-6,
        "the kick left the state critical (‖grad‖ = {g0:e})"
    );
    assert_eq!(
        unsafe { qntt_f64_saddle_seed(h_rot, x, 0.0, 40, 3, &mut lam_w, &mut xk) },
        QNTT_INVALID_ARG
    );

    let mut gopts = QnttGadOptions::defaults();
    assert_eq!(gopts.w_model, 0);
    gopts.max_steps = 40;
    gopts.preserve_grid = 1;
    gopts.w_model = 1;
    let mut xe: *mut _ = ptr::null_mut();
    let mut ginfo = QnttGadInfo::default();
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &gopts, &mut xe, &mut ginfo) },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert!(ginfo.steps <= 40);
    assert!(ginfo.gnorm.is_finite() && ginfo.lambda.is_finite());
    assert!(ginfo.applies > 0);
    if ginfo.converged != 0 {
        assert_eq!(ginfo.index1_certified, 1);
        assert_eq!(ginfo.has_index_estimate, 1);
        assert_eq!(ginfo.index_estimate, 1);
    }
    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_f64_vec_ranks(xe, ptr::null_mut(), 0, &mut len) },
        QNTT_OK
    );
    let mut re = vec![0usize; len];
    let mut rk = vec![0usize; len];
    unsafe {
        qntt_f64_vec_ranks(xe, re.as_mut_ptr(), len, &mut len);
        qntt_f64_vec_ranks(xk, rk.as_mut_ptr(), len, &mut len);
    }
    assert!(
        re.iter().zip(&rk).all(|(a, b)| a <= b),
        "GAD grew the pinned grid"
    );
    assert_eq!(
        ginfo.grid_changes, 0,
        "a fixed-grid run reports no grid change"
    );
    if ginfo.has_witness_overlap != 0 {
        assert!((0.0..=1.0 + 1e-12).contains(&ginfo.witness_overlap));
    }
    gopts.w_model = 7;
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &gopts, &mut xe, &mut ginfo) },
        QNTT_INVALID_ARG
    );
    gopts.w_model = 0;

    let d = QnttGadOptions::defaults();
    assert_eq!(d.rank_adapt, 0);
    assert!(d.rank_eps > 0.0 && d.rank_period >= 1);
    let mut ga = gopts;
    ga.max_steps = 20;
    ga.certify = 0;
    ga.rank_adapt = 1;
    ga.rank_eps = 1e-8;
    ga.chi = 0;
    let mut xg: *mut _ = ptr::null_mut();
    let mut ginfo_g = QnttGadInfo::default();
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &ga, &mut xg, &mut ginfo_g) },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut rg = vec![0usize; len];
    unsafe {
        qntt_f64_vec_ranks(xg, rg.as_mut_ptr(), len, &mut len);
    }
    assert!(
        rg.iter().zip(&rk).all(|(a, b)| *b > 0 || *a == 0),
        "RANK_BOTH opened a launch-zero (pinned CAS) sector"
    );
    assert!(ginfo_g.steps >= 1 && ginfo_g.gnorm.is_finite());
    ga.rank_adapt = 2;
    ga.rank_eps = 1e-6;
    let mut xs: *mut _ = ptr::null_mut();
    let mut ginfo_s = QnttGadInfo::default();
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &ga, &mut xs, &mut ginfo_s) },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut rs = vec![0usize; len];
    unsafe {
        qntt_f64_vec_ranks(xs, rs.as_mut_ptr(), len, &mut len);
    }
    assert!(
        rs.iter().zip(&rk).all(|(a, b)| a <= b),
        "RANK_SHRINK grew the launch grid"
    );
    ga.rank_adapt = 3;
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &ga, &mut xs, &mut ginfo_s) },
        QNTT_INVALID_ARG
    );
    ga.rank_adapt = 1;
    ga.rank_eps = 0.0;
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &ga, &mut xs, &mut ginfo_s) },
        QNTT_INVALID_ARG
    );
    ga.rank_eps = 1e-8;
    ga.rank_period = 0;
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &ga, &mut xs, &mut ginfo_s) },
        QNTT_INVALID_ARG
    );
    unsafe {
        qntt_f64_vec_free(xg);
        qntt_f64_vec_free(xs);
    }

    assert_eq!(d.border_gain_max, 1.0);
    let mut gb = gopts;
    gb.max_steps = 20;
    gb.certify = 0;
    gb.border_gain_max = 0.5;
    let mut xt: *mut _ = ptr::null_mut();
    let mut ginfo_t = QnttGadInfo::default();
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &gb, &mut xt, &mut ginfo_t) },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert!(ginfo_t.steps >= 1);
    assert!(ginfo_t.border_verdict == 0 || ginfo_t.border_verdict == 1);
    gb.border_gain_max = 0.0;
    assert_eq!(
        unsafe { qntt_f64_gad(h_rot, xk, &gb, &mut xt, &mut ginfo_t) },
        QNTT_INVALID_ARG
    );
    unsafe {
        qntt_f64_vec_free(xt);
    }

    unsafe {
        qntt_f64_vec_free(xe);
        qntt_f64_vec_free(xk);
        qntt_f64_vec_free(x);
        qntt_f64_vec_free(x0);
        qntt_f64_ham_free(h_rot);
        qntt_f64_ham_free(h);
        qntt_casscf_diag_free(diag);
        qntt_f64_fcidump_free(ints);
    }
}

#[test]
fn su2_casscf_chain_and_gad_cross_the_abi() {
    let (norb, nelec, _ms2, t, v) = read_h4_through_abi();
    let d = norb;
    let path = CString::new(h4_path().to_str().unwrap()).unwrap();
    let mut ints: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read(path.as_ptr(), 2, &mut ints) },
        QNTT_OK
    );
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_ham_new(
                d,
                nelec,
                0,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK
    );

    let mut lambda0 = f64::NAN;
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_cas_reference_state(h, &PART, 0, 7, &mut lambda0, &mut x0) },
        QNTT_OK,
        "{}",
        last_error()
    );
    let bad = QnttCasPartition {
        n_core: 1,
        n_active: 1,
        n_virt: 1,
    };
    let mut junk: *mut _ = ptr::null_mut();
    let mut junk_h: *mut _ = ptr::null_mut();
    assert_ne!(
        unsafe { qntt_su2_f64_cas_reference_state(h, &bad, 0, 7, &mut lambda0, &mut junk) },
        QNTT_OK
    );

    let mut opts = QnttCasscfOptions::defaults();
    opts.orb_max_iter = 20;
    let mut lambda = f64::NAN;
    let mut x: *mut _ = ptr::null_mut();
    let mut h_rot: *mut _ = ptr::null_mut();
    let mut u_len = 0usize;
    let mut info = QnttCasscfInfo::default();
    let mut diag: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_casscf(
                ints,
                x0,
                &PART,
                &opts,
                &mut lambda,
                &mut x,
                &mut h_rot,
                ptr::null_mut(),
                0,
                &mut u_len,
                &mut info,
                &mut diag,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert_eq!(u_len, d * d);
    assert!(lambda <= lambda0 + 1e-12);
    assert!(lambda.is_finite());
    assert_eq!(
        unsafe {
            qntt_su2_f64_casscf(
                ints,
                x0,
                &bad,
                &opts,
                &mut lambda,
                &mut junk,
                &mut junk_h,
                ptr::null_mut(),
                0,
                &mut u_len,
                &mut info,
                &mut diag,
            )
        },
        QNTT_INVALID_ARG
    );

    let mut status = -1i32;
    let mut olen = 0usize;
    let (mut n_modes, mut n_eff) = (0usize, 0usize);
    assert_eq!(
        unsafe {
            qntt_su2_f64_joint_rpa(
                h_rot,
                x,
                &PART,
                1e-6,
                1e-8,
                ptr::null_mut(),
                0,
                &mut olen,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                &mut status,
                &mut n_modes,
                &mut n_eff,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert!((0..=3).contains(&status));

    let sopts = QnttSymplecticLowestOpts {
        max_iter: 800,
        tol: 1e-6,
    };
    let out = rpa_signed_call!(
        qntt_su2_f64_rpa_signed,
        h_rot,
        x,
        3,
        &sopts,
        QNTT_RPA_METRIC_AUTO
    );
    check_rpa_signed(
        &out,
        3,
        QNTT_RPA_METRIC_AUTO,
        QNTT_RPA_KERNEL_PAIRED,
        "su2_f64",
    );
    let bad = rpa_signed_call!(qntt_su2_f64_rpa_signed, h_rot, x, 3, &sopts, -1);
    check_rpa_signed_bad_metric(&bad, "su2_f64");

    let mut lam_w = f64::NAN;
    let mut xk: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_saddle_seed(h_rot, x, 5e-2, 40, 3, &mut lam_w, &mut xk) },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut gopts = QnttGadOptions::defaults();
    gopts.max_steps = 30;
    gopts.preserve_grid = 1;
    let mut xe: *mut _ = ptr::null_mut();
    let mut ginfo = QnttGadInfo::default();
    assert_eq!(
        unsafe { qntt_su2_f64_gad(h_rot, xk, &gopts, &mut xe, &mut ginfo) },
        QNTT_OK,
        "{}",
        last_error()
    );
    assert!(ginfo.steps <= 30 && ginfo.gnorm.is_finite());

    unsafe {
        qntt_su2_f64_vec_free(xe);
        qntt_su2_f64_vec_free(xk);
        qntt_su2_f64_vec_free(x);
        qntt_su2_f64_vec_free(x0);
        qntt_su2_f64_ham_free(h_rot);
        qntt_su2_f64_ham_free(h);
        qntt_casscf_diag_free(diag);
        qntt_f64_fcidump_free(ints);
    }
}

type F64Ham = Handle<SparseHamiltonian<f64>>;
type F64Vec = Handle<qntt::TtVector<f64>>;
type Su2Ham = Handle<qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>>;
type Su2Vec = Handle<qntt::su2::vector::Su2TtVector<f64>>;

const CURV_K: usize = 3;
const CURV_OPTS: QnttSymplecticLowestOpts = QnttSymplecticLowestOpts {
    max_iter: 400,
    tol: 1e-8,
};

fn curvature_fixture_f64() -> (*mut F64Ham, *mut F64Vec) {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
    let mut h: *mut F64Ham = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new(
                space.d,
                space.nup,
                space.ndn,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK
    );
    let flat = flat_ranks(&space, &uniform_ranks(space, 2));
    let mut raw: *mut F64Vec = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                0xC0FE,
                &mut raw,
            )
        },
        QNTT_OK
    );
    let mut nrm = 0.0f64;
    assert_eq!(unsafe { qntt_f64_vec_norm(raw, &mut nrm) }, QNTT_OK);
    let inv = 1.0 / nrm;
    let mut x: *mut F64Vec = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_scaled(raw, &inv, &mut x) }, QNTT_OK);
    unsafe { qntt_f64_vec_free(raw) };
    (h, x)
}

fn curvature_fixture_su2() -> (*mut Su2Ham, *mut Su2Vec) {
    let (norb, nelec, _ms2, t, v) = read_h4_through_abi();
    let mut h: *mut Su2Ham = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_ham_new(
                norb,
                nelec,
                0,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK
    );
    let flat = su2_capped_ranks(norb, nelec, 0, 2);
    let mut raw: *mut Su2Vec = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_randn(norb, nelec, 0, flat.as_ptr(), flat.len(), 0xC0FE, &mut raw)
        },
        QNTT_OK
    );
    let mut nrm = 0.0f64;
    assert_eq!(unsafe { qntt_su2_f64_vec_norm(raw, &mut nrm) }, QNTT_OK);
    let inv = 1.0 / nrm;
    let mut x: *mut Su2Vec = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_vec_scaled(raw, &inv, &mut x) },
        QNTT_OK
    );
    unsafe { qntt_su2_f64_vec_free(raw) };
    (h, x)
}

macro_rules! sigma_table {
    ($f:ident, $x:expr) => {{
        let (mut len, mut nb) = (0usize, 0usize);
        assert_eq!(
            unsafe {
                $f(
                    $x,
                    ptr::null_mut(),
                    0,
                    &mut len,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                    0,
                    &mut nb,
                )
            },
            QNTT_OK
        );
        let (mut vals, mut wts, mut counts) =
            (vec![0.0f64; len], vec![0.0f64; len], vec![0usize; nb]);
        assert_eq!(
            unsafe {
                $f(
                    $x,
                    vals.as_mut_ptr(),
                    vals.len(),
                    &mut len,
                    wts.as_mut_ptr(),
                    wts.len(),
                    counts.as_mut_ptr(),
                    counts.len(),
                    &mut nb,
                )
            },
            QNTT_OK
        );
        let mut sig = vals;
        sig.sort_by(f64::total_cmp);
        sig
    }};
}

macro_rules! rust_signed {
    ($ctx:path, $signed:path, $tangent:ty, $randn:path, $h:expr, $x:expr, $metric:expr, $reg:expr) => {
        rust_signed!(
            $ctx,
            $signed,
            $tangent,
            $randn,
            $h,
            $x,
            $metric,
            $reg,
            RpaKernel::Paired
        )
    };
    ($ctx:path, $signed:path, $tangent:ty, $randn:path, $h:expr, $x:expr, $metric:expr, $reg:expr,
     $kernel:expr) => {{
        let hr = unsafe { borrow($h as *const _) }.unwrap();
        let xr = unsafe { borrow($x as *const _) }.unwrap();
        let reg: Option<CurvatureReg> = $reg;
        let ctx = match reg {
            Some(r) => $ctx(hr, xr).with_curvature(r),
            None => $ctx(hr, xr),
        };
        let (bl, br) = ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let v0s: Vec<$tangent> = (0..CURV_K + 2)
            .map(|_| <$tangent>::project(bl, br, &$randn(xr.space, &xr.ranks, &mut rng)))
            .collect();
        let mut o = CURV_OPTS.to_davidson();
        o.kernel = $kernel;
        let (so, vecs) = $signed(&ctx, &v0s, CURV_K, &o);
        let requested: RpaMetric = $metric;
        let side = match so.report {
            RpaReport::Signed { metric, .. } => metric,
            RpaReport::Paired { .. } if requested == RpaMetric::Rr => RpaMetric::Rr,
            RpaReport::Paired { .. } => RpaMetric::Ii,
        };
        let fw: Vec<f64> = vecs
            .iter()
            .take(so.omega2().len())
            .map(|m| {
                let v = &m.metric_side(side);
                let qa = v.dot(&ctx.a_apply(v));
                let qb = v.dot(&ctx.b_apply(v));
                if qa == 0.0 {
                    f64::INFINITY
                } else {
                    qb.abs() / qa.abs()
                }
            })
            .collect();
        (so, fw, side)
    }};
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_abi_is_rust(
    out: &RpaSignedOut,
    so: &qntt::solvers::hess::RpaReadout,
    fw: &[f64],
    side: RpaMetric,
    what: &str,
) {
    use qntt::solvers::hess::RpaModeKind;
    assert_eq!(out.rc, QNTT_OK, "{what}: {}", last_error());
    assert_eq!(bits(&out.omega2), bits(&so.omega2()), "{what}: ω²");
    assert_eq!(bits(&out.fw), bits(fw), "{what}: f_W");
    let real: Vec<_> = so
        .modes
        .iter()
        .filter(|m| m.kind != RpaModeKind::Complex)
        .collect();
    let residuals: Vec<f64> = real.iter().map(|m| m.residual).collect();
    assert_eq!(bits(&out.residual), bits(&residuals), "{what}: residuals");
    let krein: Vec<f64> = real.iter().map(|m| m.krein).collect();
    assert_eq!(bits(&out.krein), bits(&krein), "{what}: Krein margins");
    let complex: Vec<f64> = so
        .complex()
        .flat_map(|m| [m.omega2.re, m.omega2.im.abs(), m.residual])
        .collect();
    assert_eq!(bits(&out.complex), bits(&complex), "{what}: complex rows");
    assert_eq!(out.applies, so.applies, "{what}: applies");
    assert_eq!(out.iters, so.iterations, "{what}: iterations");
    assert_eq!(out.conv, so.converged as i32, "{what}: converged");
    let (kernel, report) = match &so.report {
        RpaReport::Paired { offset, .. } => (
            QNTT_RPA_KERNEL_PAIRED,
            vec![offset.neg_apb as i64, offset.neg_amb as i64, -1, -1, -1],
        ),
        RpaReport::Signed {
            events,
            n_negative_basis,
            general_path,
            ..
        } => (
            QNTT_RPA_KERNEL_SIGNED,
            vec![
                -1,
                -1,
                events.len() as i64,
                *n_negative_basis as i64,
                i64::from(*general_path),
            ],
        ),
    };
    assert_eq!(out.kernel, kernel, "{what}: kernel");
    assert_eq!(out.report, report, "{what}: report");
    assert_eq!(out.metric, rpa_metric_code(side), "{what}: side echo");
}

macro_rules! curvature_arms {
    ($layer:expr, $call:ident, $schmidt:ident, $h:expr, $x:expr,
     ctx = $ctx:path, signed = $signed:path, tangent = $tangent:ty, randn = $randn:path) => {{
        let (h, x) = ($h, $x);
        let off = rpa_signed_call!($call, h, x, CURV_K, &CURV_OPTS, QNTT_RPA_METRIC_II, QNTT_CURVATURE_REG_OFF, f64::NAN);
        check_rpa_signed(&off, CURV_K, QNTT_RPA_METRIC_II, QNTT_RPA_KERNEL_PAIRED, $layer);
        let (so, fw, side) = rust_signed!($ctx, $signed, $tangent, $randn, h, x, RpaMetric::Ii, None);
        assert_abi_is_rust(&off, &so, &fw, side, &format!("{} C1 OFF(NaN) vs the plain context", $layer));
        let legacy = rpa_signed_call!($call, h, x, CURV_K, &CURV_OPTS, QNTT_RPA_METRIC_II);
        assert_eq!(bits(&legacy.omega2), bits(&off.omega2), "{}: OFF's value is ignored", $layer);
        assert_eq!(legacy.applies, off.applies);

        let sig = sigma_table!($schmidt, x);
        let top = sig[sig.len() - 1];
        let next = sig
            .iter()
            .rev()
            .copied()
            .find(|&w| w < top * (1.0 - 1e-9))
            .expect("fixture: at least two distinct Schmidt values");
        let f_mid = 0.5 * (top + next);
        let f_below = 0.5 * sig[0];
        println!("{} C2 fixture: σ ∈ [{:.4e}, {top:.4e}], floor {f_mid:.4e} (between {next:.4e} and {top:.4e})", $layer, sig[0]);
        for (kind, reg) in [
            (QNTT_CURVATURE_REG_HARD, CurvatureReg::Hard(f_mid)),
            (QNTT_CURVATURE_REG_TIKHONOV, CurvatureReg::Tikhonov(f_mid)),
        ] {
            let out = rpa_signed_call!($call, h, x, CURV_K, &CURV_OPTS, QNTT_RPA_METRIC_II, kind, f_mid);
            check_rpa_signed(&out, CURV_K, QNTT_RPA_METRIC_II, QNTT_RPA_KERNEL_PAIRED, $layer);
            let (so, fw, side) = rust_signed!($ctx, $signed, $tangent, $randn, h, x, RpaMetric::Ii, Some(reg));
            assert_abi_is_rust(&out, &so, &fw, side, &format!("{} C2 {reg:?}", $layer));
            println!("{} C2 {reg:?}: ω² {:?} (OFF {:?})", $layer, out.omega2, off.omega2);
            assert_ne!(
                bits(&out.omega2),
                bits(&off.omega2),
                "{} non-vacuity: {reg:?} must change ω²",
                $layer
            );
            let auto = rpa_signed_call!($call, h, x, CURV_K, &CURV_OPTS, QNTT_RPA_METRIC_AUTO, kind, f_mid);
            assert_eq!(auto.rc, QNTT_OK, "{}", last_error());
            assert_eq!(auto.metric, QNTT_RPA_METRIC_II, "{} {reg:?}: AUTO → II", $layer);
            assert_eq!(bits(&auto.omega2), bits(&out.omega2), "{} {reg:?}: AUTO is the II call", $layer);
            assert_eq!(auto.applies, out.applies);
        }
        let below = rpa_signed_call!($call, h, x, CURV_K, &CURV_OPTS, QNTT_RPA_METRIC_II, QNTT_CURVATURE_REG_HARD, f_below);
        assert_eq!(below.rc, QNTT_OK, "{}", last_error());
        assert_eq!(bits(&below.omega2), bits(&off.omega2), "{}: HARD below every raw σ is OFF", $layer);
        assert_eq!(bits(&below.fw), bits(&off.fw));
        assert_eq!(bits(&below.residual), bits(&off.residual));
        assert_eq!(below.applies, off.applies);
    }};
}

#[test]
fn rpa_signed_curvature_off_hard_tikhonov_are_the_rust_driver_f64() {
    let (h, x) = curvature_fixture_f64();
    curvature_arms!(
        "f64", qntt_f64_rpa_signed, qntt_f64_vec_schmidt, h, x,
        ctx = qntt::solvers::hess::weingarten_operator,
        signed = qntt::solvers::hess::symplectic_lowest_signed_davidson,
        tangent = qntt::core::tangent::TtTangent<f64>,
        randn = qntt::tt_randn
    );
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn rpa_signed_curvature_off_hard_tikhonov_are_the_rust_driver_su2() {
    let (h, x) = curvature_fixture_su2();
    curvature_arms!(
        "su2_f64", qntt_su2_f64_rpa_signed, qntt_su2_f64_vec_schmidt, h, x,
        ctx = qntt::su2::hess::weingarten_operator,
        signed = qntt::su2::hess::symplectic_lowest_signed_davidson,
        tangent = qntt::su2::tangent::Su2TtTangent<f64>,
        randn = qntt::su2::vector::tt_randn
    );
    unsafe {
        qntt_su2_f64_vec_free(x);
        qntt_su2_f64_ham_free(h);
    }
}

macro_rules! kernel_selector_arms {
    ($layer:expr, $call:ident, $h:expr, $x:expr,
     ctx = $ctx:path, signed = $signed:path, tangent = $tangent:ty, randn = $randn:path) => {{
        let (h, x) = ($h, $x);
        let mut outs = Vec::new();
        for kernel in [QNTT_RPA_KERNEL_PAIRED, QNTT_RPA_KERNEL_SIGNED] {
            for (metric, requested) in [
                (QNTT_RPA_METRIC_II, RpaMetric::Ii),
                (QNTT_RPA_METRIC_RR, RpaMetric::Rr),
                (QNTT_RPA_METRIC_AUTO, RpaMetric::Auto),
            ] {
                let rk = match (kernel, metric) {
                    (QNTT_RPA_KERNEL_PAIRED, _) => RpaKernel::Paired,
                    (_, QNTT_RPA_METRIC_RR) => RpaKernel::signed(RpaMetric::Rr),
                    _ => RpaKernel::signed(RpaMetric::Ii),
                };
                let out = rpa_signed_call!(
                    $call,
                    h,
                    x,
                    CURV_K,
                    &CURV_OPTS,
                    metric,
                    QNTT_CURVATURE_REG_OFF,
                    0.0,
                    kernel
                );
                let what = format!("{} kernel {kernel} metric {metric}", $layer);
                check_rpa_signed(&out, CURV_K, metric, kernel, &what);
                let (so, fw, side) =
                    rust_signed!($ctx, $signed, $tangent, $randn, h, x, requested, None, rk);
                assert_abi_is_rust(&out, &so, &fw, side, &what);
                println!("{what}: applies {} ω² {:?}", out.applies, out.omega2);
                outs.push(((kernel, metric), out));
            }
        }
        let get = |k: i32, m: i32| &outs.iter().find(|(km, _)| *km == (k, m)).unwrap().1;
        let (sig_ii, sig_rr, sig_auto) = (
            get(QNTT_RPA_KERNEL_SIGNED, QNTT_RPA_METRIC_II),
            get(QNTT_RPA_KERNEL_SIGNED, QNTT_RPA_METRIC_RR),
            get(QNTT_RPA_KERNEL_SIGNED, QNTT_RPA_METRIC_AUTO),
        );
        assert_eq!(
            bits(&sig_auto.omega2),
            bits(&sig_ii.omega2),
            "{}: SIGNED AUTO is the II call",
            $layer
        );
        assert_eq!(bits(&sig_auto.fw), bits(&sig_ii.fw), "{}", $layer);
        assert_eq!(
            sig_auto.applies, sig_ii.applies,
            "{}: no probe applies",
            $layer
        );
        let pair_ii = get(QNTT_RPA_KERNEL_PAIRED, QNTT_RPA_METRIC_II);
        for m in [QNTT_RPA_METRIC_RR, QNTT_RPA_METRIC_AUTO] {
            let o = get(QNTT_RPA_KERNEL_PAIRED, m);
            assert_eq!(
                bits(&o.omega2),
                bits(&pair_ii.omega2),
                "{}: PAIRED's metric is the f_W side only",
                $layer
            );
            assert_eq!(o.applies, pair_ii.applies, "{}", $layer);
        }
        assert!(
            sig_rr.applies != sig_ii.applies || bits(&sig_rr.omega2) != bits(&sig_ii.omega2),
            "{} non-vacuity: the SIGNED lane must reach the solve",
            $layer
        );
        assert!(
            sig_ii.applies != pair_ii.applies || bits(&sig_ii.omega2) != bits(&pair_ii.omega2),
            "{} non-vacuity: SIGNED must not be the paired solve",
            $layer
        );
    }};
}

#[test]
fn rpa_signed_kernel_selector_is_the_rust_driver_f64() {
    let (h, x) = curvature_fixture_f64();
    kernel_selector_arms!(
        "f64", qntt_f64_rpa_signed, h, x,
        ctx = qntt::solvers::hess::weingarten_operator,
        signed = qntt::solvers::hess::symplectic_lowest_signed_davidson,
        tangent = qntt::core::tangent::TtTangent<f64>,
        randn = qntt::tt_randn
    );
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn rpa_signed_kernel_selector_is_the_rust_driver_su2() {
    let (h, x) = curvature_fixture_su2();
    kernel_selector_arms!(
        "su2_f64", qntt_su2_f64_rpa_signed, h, x,
        ctx = qntt::su2::hess::weingarten_operator,
        signed = qntt::su2::hess::symplectic_lowest_signed_davidson,
        tangent = qntt::su2::tangent::Su2TtTangent<f64>,
        randn = qntt::su2::vector::tt_randn
    );
    unsafe {
        qntt_su2_f64_vec_free(x);
        qntt_su2_f64_ham_free(h);
    }
}

fn check_rpa_signed_bad_kernel(out: &RpaSignedOut, what: &str) {
    assert_eq!(out.rc, QNTT_INVALID_ARG, "{what}: {}", last_error());
    let msg = last_error();
    assert!(
        msg.contains("RPA kernel") && msg.contains("0 = paired") && msg.contains("1 = signed"),
        "{what}: the message names the kernel codes, got {msg:?}"
    );
    assert!(
        out.omega2.is_empty()
            && out.applies == usize::MAX
            && out.iters == usize::MAX
            && out.metric == -1
            && out.kernel == -1
            && out.conv == -1,
        "{what}: no solve ran"
    );
}

macro_rules! c64_bad_kernel_call {
    ($f:ident, $kernel:expr) => {{
        let opts = CURV_OPTS;
        let n_max = opts.max_iter.max(CURV_K + 2);
        let (mut w2, mut fw, mut res, mut krein) = (
            vec![0.0; n_max],
            vec![0.0; n_max],
            vec![0.0; n_max],
            vec![0.0; n_max],
        );
        let (mut cx, mut report) = (vec![0.0; 3 * n_max], vec![i64::MIN; 5]);
        let mut lens = [usize::MAX; 6];
        let (mut cert, mut conv, mut kernel) = (-1i32, -1i32, -1i32);
        let (mut iters, mut applies) = (usize::MAX, usize::MAX);
        let [l0, l1, l2, l3, l4, l5] = &mut lens;
        let rc = unsafe {
            $f(
                ptr::null(),
                ptr::null(),
                CURV_K,
                &opts,
                $kernel,
                1,
                w2.as_mut_ptr(),
                n_max,
                l0,
                fw.as_mut_ptr(),
                n_max,
                l1,
                res.as_mut_ptr(),
                n_max,
                l2,
                krein.as_mut_ptr(),
                n_max,
                l3,
                cx.as_mut_ptr(),
                3 * n_max,
                l4,
                &mut cert,
                &mut conv,
                &mut iters,
                &mut applies,
                &mut kernel,
                report.as_mut_ptr(),
                5,
                l5,
            )
        };
        (rc, lens, applies, iters, kernel, conv)
    }};
}

#[test]
fn rpa_signed_refuses_a_bad_kernel_code_before_any_work() {
    let (h, x) = curvature_fixture_f64();
    let (hs, xs) = curvature_fixture_su2();
    for kernel in [2, -1] {
        let out = rpa_signed_call!(
            qntt_f64_rpa_signed,
            h,
            x,
            CURV_K,
            &CURV_OPTS,
            QNTT_RPA_METRIC_II,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            kernel
        );
        check_rpa_signed_bad_kernel(&out, &format!("f64 kernel {kernel}"));
        let out = rpa_signed_call!(
            qntt_su2_f64_rpa_signed,
            hs,
            xs,
            CURV_K,
            &CURV_OPTS,
            QNTT_RPA_METRIC_II,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            kernel
        );
        check_rpa_signed_bad_kernel(&out, &format!("su2_f64 kernel {kernel}"));
        for (what, got) in [
            ("c64", c64_bad_kernel_call!(qntt_c64_rpa_signed, kernel)),
            (
                "su2_c64",
                c64_bad_kernel_call!(qntt_su2_c64_rpa_signed, kernel),
            ),
        ] {
            let (rc, lens, applies, iters, kern, conv) = got;
            assert_eq!(
                rc,
                QNTT_INVALID_ARG,
                "{what} kernel {kernel}: {}",
                last_error()
            );
            let msg = last_error();
            assert!(
                msg.contains("RPA kernel")
                    && msg.contains("0 = paired")
                    && msg.contains("1 = signed"),
                "{what} kernel {kernel}: {msg:?}"
            );
            assert!(
                lens.iter().all(|&l| l == usize::MAX)
                    && applies == usize::MAX
                    && iters == usize::MAX
                    && kern == -1
                    && conv == -1,
                "{what} kernel {kernel}: no solve ran"
            );
        }
    }
    let both = rpa_signed_call!(
        qntt_f64_rpa_signed,
        h,
        x,
        CURV_K,
        &CURV_OPTS,
        7,
        QNTT_CURVATURE_REG_OFF,
        0.0,
        2
    );
    check_rpa_signed_bad_kernel(&both, "bad kernel and bad metric");
    let null_h = rpa_signed_call!(
        qntt_f64_rpa_signed,
        ptr::null(),
        x,
        CURV_K,
        &CURV_OPTS,
        QNTT_RPA_METRIC_II,
        QNTT_CURVATURE_REG_OFF,
        0.0,
        2
    );
    assert_ne!(
        null_h.rc, QNTT_NULL_ARG,
        "the kernel is decoded before the borrow"
    );
    check_rpa_signed_bad_kernel(&null_h, "null h, bad kernel");
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
        qntt_su2_f64_vec_free(xs);
        qntt_su2_f64_ham_free(hs);
    }
}

fn check_rpa_signed_bad_curvature(out: &RpaSignedOut, what: &str) {
    assert_eq!(out.rc, QNTT_INVALID_ARG, "{what}: {}", last_error());
    assert!(
        out.omega2.is_empty() && out.applies == usize::MAX && out.metric == -1,
        "{what}: no solve ran"
    );
}

#[test]
fn rpa_signed_refuses_a_bad_curvature_request_before_any_work() {
    let (h, x) = curvature_fixture_f64();
    let (hs, xs) = curvature_fixture_su2();
    for kind in [3, -1] {
        let out = rpa_signed_call!(
            qntt_f64_rpa_signed,
            h,
            x,
            CURV_K,
            &CURV_OPTS,
            QNTT_RPA_METRIC_II,
            kind,
            1e-3
        );
        check_rpa_signed_bad_curvature(&out, &format!("f64 kind {kind}"));
        let msg = last_error();
        assert!(
            msg.contains("0 = off") && msg.contains("1 = hard") && msg.contains("2 = tikhonov"),
            "kind {kind}: the message names the codes, got {msg:?}"
        );
        let out = rpa_signed_call!(
            qntt_su2_f64_rpa_signed,
            hs,
            xs,
            CURV_K,
            &CURV_OPTS,
            QNTT_RPA_METRIC_II,
            kind,
            1e-3
        );
        check_rpa_signed_bad_curvature(&out, &format!("su2_f64 kind {kind}"));
    }
    let bad_values = [
        (QNTT_CURVATURE_REG_HARD, 0.0),
        (QNTT_CURVATURE_REG_HARD, -1.0),
        (QNTT_CURVATURE_REG_HARD, f64::NAN),
        (QNTT_CURVATURE_REG_HARD, f64::INFINITY),
        (QNTT_CURVATURE_REG_TIKHONOV, 0.0),
    ];
    for (kind, value) in bad_values {
        let out = rpa_signed_call!(
            qntt_f64_rpa_signed,
            h,
            x,
            CURV_K,
            &CURV_OPTS,
            QNTT_RPA_METRIC_II,
            kind,
            value
        );
        check_rpa_signed_bad_curvature(&out, &format!("f64 kind {kind} value {value}"));
        let out = rpa_signed_call!(
            qntt_su2_f64_rpa_signed,
            hs,
            xs,
            CURV_K,
            &CURV_OPTS,
            QNTT_RPA_METRIC_II,
            kind,
            value
        );
        check_rpa_signed_bad_curvature(&out, &format!("su2_f64 kind {kind} value {value}"));
    }
    let both = rpa_signed_call!(qntt_f64_rpa_signed, h, x, CURV_K, &CURV_OPTS, 7, 3, 1e-3);
    check_rpa_signed_bad_curvature(&both, "bad metric and bad kind");
    assert!(
        last_error().contains("RPA metric"),
        "the metric is decoded first: {}",
        last_error()
    );
    let null_h = rpa_signed_call!(
        qntt_f64_rpa_signed,
        ptr::null(),
        x,
        CURV_K,
        &CURV_OPTS,
        QNTT_RPA_METRIC_II,
        3,
        1e-3
    );
    assert_ne!(
        null_h.rc, QNTT_NULL_ARG,
        "the curvature pair is decoded before the borrow"
    );
    check_rpa_signed_bad_curvature(&null_h, "null h, bad kind");
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
        qntt_su2_f64_vec_free(xs);
        qntt_su2_f64_ham_free(hs);
    }
}

#[test]
fn curvature_reg_from_code_decodes_and_validates() {
    assert_eq!(
        (
            QNTT_CURVATURE_REG_OFF,
            QNTT_CURVATURE_REG_HARD,
            QNTT_CURVATURE_REG_TIKHONOV
        ),
        (0, 1, 2)
    );
    assert_eq!(
        curvature_reg_from_code(0, f64::NAN).ok(),
        Some(CurvatureReg::Off)
    );
    assert_eq!(
        curvature_reg_from_code(0, -5.0).ok(),
        Some(CurvatureReg::Off)
    );
    assert_eq!(
        curvature_reg_from_code(1, 1e-3).ok(),
        Some(CurvatureReg::Hard(1e-3))
    );
    assert_eq!(
        curvature_reg_from_code(2, 3e-4).ok(),
        Some(CurvatureReg::Tikhonov(3e-4))
    );
    for (kind, value) in [
        (3, 1e-3),
        (-1, 1e-3),
        (1, 0.0),
        (1, -1.0),
        (1, f64::NAN),
        (1, f64::INFINITY),
        (2, 0.0),
        (2, f64::NAN),
    ] {
        let err = curvature_reg_from_code(kind, value).expect_err("must be refused");
        assert_eq!(
            err.code, QNTT_INVALID_ARG,
            "kind {kind} value {value}: {}",
            err.msg
        );
    }
    let unknown = curvature_reg_from_code(7, 1.0).expect_err("unknown kind");
    assert!(
        unknown.msg.contains("0 = off")
            && unknown.msg.contains("1 = hard")
            && unknown.msg.contains("2 = tikhonov"),
        "{}",
        unknown.msg
    );
}

#[test]
fn the_rpa_signed_declaration_carries_the_curvature_pair() {
    let hdr = include_str!("../../include/qntt.h");
    let flat = hdr
        .replace('\\', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let start = flat
        .find("int32_t qntt_##SFX##_rpa_signed(")
        .expect("qntt.h declares qntt_##SFX##_rpa_signed");
    let end = start + flat[start..].find(");").expect("the declaration ends");
    let decl = &flat[start..end];
    assert!(
        decl.contains(
            "const QnttSymplecticLowestOpts *opts, int32_t kernel, int32_t metric, int32_t \
             curvature_kind, double curvature_value, uint64_t seed"
        ),
        "the declaration does not carry the kernel after opts and the curvature pair after the \
         metric: {decl}"
    );
    let start_c64 = end
        + flat[end..]
            .find("int32_t qntt_##SFX##_rpa_signed(")
            .expect("qntt.h declares the C64 qntt_##SFX##_rpa_signed");
    let end_c64 = start_c64 + flat[start_c64..].find(");").expect("the declaration ends");
    let decl_c64 = &flat[start_c64..end_c64];
    assert!(
        decl_c64.contains("const QnttSymplecticLowestOpts *opts, int32_t kernel, uint64_t seed"),
        "the C64 declaration does not carry the kernel after opts: {decl_c64}"
    );
}
