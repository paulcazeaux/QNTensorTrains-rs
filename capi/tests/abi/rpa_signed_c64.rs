// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;
use qntt::core::tangent::TtTangent;
use qntt::solvers::hess::{
    RpaModeKind, RpaReadout, RpaReport, symplectic_lowest_signed_davidson_c64, weingarten_operator,
};
use qntt_capi::handle::borrow;
use qntt_capi::rpa::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const K: usize = 3;
const SEED: u64 = 0x5EED;

macro_rules! rust_default {
    ($T:ty, $h:expr, $x:expr, $kreq:expr, |$ctx:ident, $v0s:ident, $o:ident| $call:expr) => {{
        let h = unsafe { borrow($h as *const _) }.unwrap();
        let x: &qntt::TtVector<$T> = unsafe { borrow($x as *const _) }.unwrap();
        let $ctx = weingarten_operator(h, x);
        let (bl, br) = $ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
        let $v0s: Vec<TtTangent<$T>> = (0..$kreq + 2)
            .map(|_| TtTangent::project(bl, br, &qntt::tt_randn(x.space, &x.ranks, &mut rng)))
            .collect();
        let $o = QnttSymplecticLowestOpts::defaults().to_davidson();
        let (so, _amps): (RpaReadout, _) = $call;
        so
    }};
}

struct Signed {
    rc: i32,
    omega2: Vec<f64>,
    fw_len: usize,
    residual: Vec<f64>,
    krein: Vec<f64>,
    complex: Vec<f64>,
    cert: i32,
    conv: i32,
    iters: usize,
    applies: usize,
    kernel: i32,
    report: Vec<i64>,
}

macro_rules! signed_call {
    (@run $opts:ident, $kreq:expr, |$w2:ident, $fw:ident, $res:ident, $krein:ident, $cx:ident,
     $report:ident, $len:ident, $fwl:ident, $resl:ident, $kreinl:ident, $cxl:ident, $repl:ident,
     $cert:ident, $conv:ident, $iters:ident, $applies:ident, $kernel:ident, $nmax:ident| $call:expr) => {{
        let $opts = QnttSymplecticLowestOpts::defaults();
        let $nmax = $opts.max_iter.max($kreq + 2);
        let (mut $w2, mut $fw, mut $res) = (vec![0.0; $nmax], vec![0.0; $nmax], vec![0.0; $nmax]);
        let (mut $krein, mut $cx, mut $report) =
            (vec![0.0; $nmax], vec![0.0; 3 * $nmax], vec![i64::MIN; 5]);
        let (mut $len, mut $fwl, mut $resl, mut $kreinl, mut $cxl, mut $repl) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut $cert, mut $conv, mut $kernel) = (-1i32, -1i32, -1i32);
        let (mut $iters, mut $applies) = (0usize, 0usize);
        let rc = unsafe { $call };
        let ok = rc == QNTT_OK;
        let cut = |v: &[f64], n: usize| if ok { v[..n].to_vec() } else { Vec::new() };
        Signed {
            rc,
            omega2: cut(&$w2, $len),
            fw_len: $fwl,
            residual: cut(&$res, $resl),
            krein: cut(&$krein, $kreinl),
            complex: cut(&$cx, $cxl),
            cert: $cert,
            conv: $conv,
            iters: $iters,
            applies: $applies,
            kernel: $kernel,
            report: if ok { $report[..$repl].to_vec() } else { Vec::new() },
        }
    }};
    (c64 $f:ident, $h:expr, $x:expr, $kreq:expr, $kernel_req:expr) => {
        signed_call!(@run opts, $kreq, |w2, fw, res, krein, cx, report, len, fwl, resl, kreinl,
            cxl, repl, cert, conv, iters, applies, kernel, nmax| $f(
            $h, $x, $kreq, &opts, $kernel_req, SEED, w2.as_mut_ptr(), nmax, &mut len,
            fw.as_mut_ptr(), nmax, &mut fwl, res.as_mut_ptr(), nmax, &mut resl,
            krein.as_mut_ptr(), nmax, &mut kreinl, cx.as_mut_ptr(), 3 * nmax, &mut cxl,
            &mut cert, &mut conv, &mut iters, &mut applies, &mut kernel, report.as_mut_ptr(), 5,
            &mut repl,
        ))
    };
    (f64 $f:ident, $h:expr, $x:expr, $kreq:expr, $kernel_req:expr, $metric:expr) => {
        signed_call!(@run opts, $kreq, |w2, fw, res, krein, cx, report, len, fwl, resl, kreinl,
            cxl, repl, cert, conv, iters, applies, kernel, nmax| $f(
            $h, $x, $kreq, &opts, $kernel_req, $metric, QNTT_CURVATURE_REG_OFF, 0.0, SEED,
            w2.as_mut_ptr(), nmax, &mut len, fw.as_mut_ptr(), nmax, &mut fwl, res.as_mut_ptr(),
            nmax, &mut resl, krein.as_mut_ptr(), nmax, &mut kreinl, cx.as_mut_ptr(), 3 * nmax,
            &mut cxl, ptr::null_mut(), &mut cert, &mut conv, &mut iters, &mut applies,
            &mut kernel, report.as_mut_ptr(), 5, &mut repl,
        ))
    };
}

macro_rules! h4_states {
    ($T:ty, $ham_new:ident, $randn:ident, $norm:ident, $scaled:ident, $als:ident, $conv:expr) => {{
        let (norb, nelec, ms2, t, v) = read_h4_through_abi();
        let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
        let (t, v): (Vec<$T>, Vec<$T>) = (
            t.iter().map(|&a| $conv(a)).collect(),
            v.iter().map(|&a| $conv(a)).collect(),
        );
        let mut h: *mut _ = ptr::null_mut();
        let rc = unsafe {
            $ham_new(
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
        };
        assert_eq!(rc, QNTT_OK, "{}", last_error());
        let flat = flat_ranks(&space, &uniform_ranks(space, 2));
        let mut raw: *mut _ = ptr::null_mut();
        let rc = unsafe {
            $randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                7,
                &mut raw,
            )
        };
        assert_eq!(rc, QNTT_OK, "{}", last_error());
        let mut nrm = 0.0f64;
        assert_eq!(unsafe { $norm(raw, &mut nrm) }, QNTT_OK);
        let inv: $T = $conv(1.0 / nrm);
        let mut x_rand: *mut _ = ptr::null_mut();
        assert_eq!(unsafe { $scaled(raw, &inv, &mut x_rand) }, QNTT_OK);
        let mut opts = QnttAlsOptions::defaults();
        opts.eps = 1e-10;
        opts.max_iter = 50;
        let mut lambda: $T = $conv(0.0);
        let mut x_gs: *mut _ = ptr::null_mut();
        let rc = unsafe { $als(h, raw, &opts, &mut lambda, &mut x_gs) };
        assert_eq!(rc, QNTT_OK, "{}", last_error());
        (h, raw, x_rand, x_gs)
    }};
}

#[test]
fn rpa_signed_c64_is_the_default_signed_entry() {
    let (h, raw, x_rand, x_gs) = h4_states!(
        C64,
        qntt_c64_ham_new,
        qntt_c64_vec_randn,
        qntt_c64_vec_norm,
        qntt_c64_vec_scaled,
        qntt_c64_als,
        |a: f64| C64::new(a, 0.0)
    );
    let kreq = 2 * K;
    for (what, x, indefinite) in [
        ("ground state", x_gs, false),
        ("random state", x_rand, true),
    ] {
        let so = rust_default!(C64, h, x, kreq, |ctx, v0s, o| {
            symplectic_lowest_signed_davidson_c64(&ctx, &v0s, kreq, &o)
        });
        assert_eq!(
            so.anomalous(),
            indefinite,
            "{what}: fixture premise {:?}",
            so.omega2()
        );
        let got = signed_call!(c64 qntt_c64_rpa_signed, h, x, kreq, QNTT_RPA_KERNEL_PAIRED);
        assert_eq!(got.rc, QNTT_OK, "{what}: {}", last_error());
        let real: Vec<_> = so
            .modes
            .iter()
            .filter(|m| m.kind != RpaModeKind::Complex)
            .collect();
        assert_eq!(&got.omega2[..], &so.omega2()[..], "{what}: ω²");
        let want_res: Vec<f64> = real.iter().map(|m| m.residual).collect();
        assert_eq!(&got.residual[..], &want_res[..], "{what}: residuals");
        let want_krein: Vec<f64> = real.iter().map(|m| m.krein).collect();
        assert_eq!(&got.krein[..], &want_krein[..], "{what}: Krein margins");
        assert_eq!(got.fw_len, got.omega2.len(), "{what}: one f_W per mode");
        let want_cx: Vec<f64> = so
            .complex()
            .flat_map(|m| [m.omega2.re, m.omega2.im.abs(), m.residual])
            .collect();
        assert_eq!(&got.complex[..], &want_cx[..], "{what}: complex rows");
        let RpaReport::Paired { offset, .. } = &so.report else {
            panic!("{what}: the default kernel is the paired one")
        };
        assert_eq!(got.kernel, QNTT_RPA_KERNEL_PAIRED, "{what}");
        assert_eq!(
            &got.report[..],
            &[offset.neg_apb as i64, offset.neg_amb as i64, -1, -1, -1],
            "{what}: report"
        );
        assert_eq!(got.cert, so.all_negatives_certified() as i32, "{what}");
        assert_eq!(
            (got.conv, got.iters, got.applies),
            (so.converged as i32, so.iterations, so.applies),
            "{what}"
        );
    }
    unsafe {
        qntt_c64_vec_free(x_gs);
        qntt_c64_vec_free(x_rand);
        qntt_c64_vec_free(raw);
        qntt_c64_ham_free(h);
    }
}

fn assert_publishable(got: &Signed, kernel: i32, what: &str) {
    assert_eq!(got.rc, QNTT_OK, "{what}: {}", last_error());
    assert_eq!(got.kernel, kernel, "{what}: the kernel echo is the request");
    assert_eq!(got.conv, 1, "{what}: converged");
    assert!(
        got.omega2.iter().all(|&w| w >= 0.0),
        "{what}: {:?}",
        got.omega2
    );
    assert!(
        got.krein.iter().all(|&m| m > 0.0),
        "{what}: {:?}",
        got.krein
    );
    assert!(
        got.complex.is_empty(),
        "{what}: complex rows {:?}",
        got.complex
    );
    let clean = match kernel {
        QNTT_RPA_KERNEL_PAIRED => got.report[..2] == [0, 0],
        _ => got.report[2] == 0,
    };
    assert!(clean, "{what}: report {:?}", got.report);
}

#[test]
fn rpa_signed_both_kernels_match_dense_at_the_h4_ground_state() {
    let (h, raw, x_rand, x_gs) = h4_states!(
        f64,
        qntt_f64_ham_new,
        qntt_f64_vec_randn,
        qntt_f64_vec_norm,
        qntt_f64_vec_scaled,
        qntt_f64_als,
        |a: f64| a
    );
    let mut dense = vec![0.0f64; 256];
    let (mut dlen, mut stable) = (0usize, -1i32);
    let rc = unsafe {
        qntt_f64_rpa_dense(
            h,
            x_gs,
            dense.as_mut_ptr(),
            dense.len(),
            &mut dlen,
            &mut stable,
        )
    };
    assert_eq!((rc, stable), (QNTT_OK, 1), "f64 dense: {}", last_error());
    dense.truncate(dlen);
    assert!(dlen >= K, "f64 dense: {dlen} values");
    for kernel in [QNTT_RPA_KERNEL_PAIRED, QNTT_RPA_KERNEL_SIGNED] {
        let what = format!("f64 kernel {kernel}");
        let got = signed_call!(f64 qntt_f64_rpa_signed, h, x_gs, K, kernel, QNTT_RPA_METRIC_II);
        assert_publishable(&got, kernel, &what);
        assert_eq!(got.omega2.len(), K, "{what}: k excitations");
        for (j, (&w2, &wd)) in got.omega2.iter().zip(&dense).enumerate() {
            let rel = (w2.sqrt() - wd).abs() / wd;
            println!(
                "{what} [{j}]: ω {:.12} dense {wd:.12} rel {rel:.2e}",
                w2.sqrt()
            );
            assert!(rel <= 1e-6, "{what} [{j}]: ω {} vs dense {wd}", w2.sqrt());
        }
    }
    unsafe {
        qntt_f64_vec_free(x_gs);
        qntt_f64_vec_free(x_rand);
        qntt_f64_vec_free(raw);
        qntt_f64_ham_free(h);
    }

    let (h, raw, x_rand, x_gs) = h4_states!(
        C64,
        qntt_c64_ham_new,
        qntt_c64_vec_randn,
        qntt_c64_vec_norm,
        qntt_c64_vec_scaled,
        qntt_c64_als,
        |a: f64| C64::new(a, 0.0)
    );
    let mut dense = vec![0.0f64; 256];
    let (mut dlen, mut stable) = (0usize, -1i32);
    let rc = unsafe {
        qntt_c64_rpa_dense(
            h,
            x_gs,
            dense.as_mut_ptr(),
            dense.len(),
            &mut dlen,
            &mut stable,
        )
    };
    assert_eq!((rc, stable), (QNTT_OK, 1), "c64 dense: {}", last_error());
    dense.truncate(dlen);
    assert!(dlen >= K, "c64 dense: {dlen} values");
    for kernel in [QNTT_RPA_KERNEL_PAIRED, QNTT_RPA_KERNEL_SIGNED] {
        let what = format!("c64 kernel {kernel}");
        let got = signed_call!(c64 qntt_c64_rpa_signed, h, x_gs, 2 * K, kernel);
        assert_publishable(&got, kernel, &what);
        assert_eq!(
            got.omega2.len(),
            2 * K,
            "{what}: both realified copies of k excitations"
        );
        for (j, &wd) in dense.iter().take(K).enumerate() {
            for copy in [2 * j, 2 * j + 1] {
                let w = got.omega2[copy].sqrt();
                let rel = (w - wd).abs() / wd;
                println!("{what} [{j}] copy {copy}: ω {w:.12} dense {wd:.12} rel {rel:.2e}");
                assert!(rel <= 1e-6, "{what} [{j}] copy {copy}: ω {w} vs dense {wd}");
            }
        }
    }
    unsafe {
        qntt_c64_vec_free(x_gs);
        qntt_c64_vec_free(x_rand);
        qntt_c64_vec_free(raw);
        qntt_c64_ham_free(h);
    }
}
