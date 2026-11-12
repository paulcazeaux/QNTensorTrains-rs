// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::core::tangent::TtTangent;
use qntt::solvers::hess::{TdaDavidsonOpts, TdaModes, TdaPrecond};
use qntt::su2::tangent::Su2TtTangent;
use qntt_capi::err::QNTT_NULL_ARG;
use qntt_capi::handle::{Handle, HandleType, borrow};
use qntt_capi::tda::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const K: usize = 3;
const SEED: u64 = 0x7DA;

struct TdaOut<V: HandleType> {
    rc: i32,
    theta: Vec<f64>,
    residual: Vec<f64>,
    states: Vec<*mut Handle<V>>,
    states_len: usize,
    conv: i32,
    cert: i32,
    iters: usize,
    applies: usize,
    space_dim: usize,
    precond: i32,
}

macro_rules! tda_call {
    ($f:ident, $V:ty, $h:expr, $x:expr, $k:expr, $opts:expr, $cap:expr, $states:expr) => {{
        let cap: usize = $cap;
        let (mut theta, mut res) = (vec![f64::NAN; cap], vec![f64::NAN; cap]);
        let mut slots: Vec<*mut Handle<$V>> = vec![ptr::null_mut(); cap];
        let (mut len, mut reslen, mut slen) = (usize::MAX, usize::MAX, usize::MAX);
        let (mut conv, mut cert, mut pre) = (-7i32, -7i32, -7i32);
        let (mut iters, mut applies, mut dim) = (usize::MAX, usize::MAX, usize::MAX);
        let rc = unsafe {
            $f(
                $h,
                $x,
                $k,
                $opts,
                SEED,
                theta.as_mut_ptr(),
                cap,
                &mut len,
                res.as_mut_ptr(),
                cap,
                &mut reslen,
                if $states {
                    slots.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                cap,
                &mut slen,
                &mut conv,
                &mut cert,
                &mut iters,
                &mut applies,
                &mut dim,
                &mut pre,
            )
        };
        let cut = |n: usize| if n == usize::MAX { 0 } else { n.min(cap) };
        TdaOut::<$V> {
            rc,
            theta: theta[..cut(len)].to_vec(),
            residual: res[..cut(reslen)].to_vec(),
            states: slots[..cut(slen)].to_vec(),
            states_len: slen,
            conv,
            cert,
            iters,
            applies,
            space_dim: dim,
            precond: pre,
        }
    }};
}

macro_rules! rust_tda {
    ($ctxf:path, $tdaf:path, $randn:path, $Tan:ty, $h:expr, $x:expr, $k:expr, $o:expr) => {{
        let h = unsafe { borrow($h as *const _) }.unwrap();
        let x = unsafe { borrow($x as *const _) }.unwrap();
        let ctx = $ctxf(h, x);
        let (bl, br) = ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
        let v0s: Vec<$Tan> = (0..$k + 2)
            .map(|_| <$Tan>::project(bl, br, &$randn(x.space, &x.ranks, &mut rng)))
            .collect();
        let m: TdaModes<$Tan> = $tdaf(&ctx, &v0s, $k, $o);
        m
    }};
}

fn check_against_rust<V: HandleType, T>(
    out: &TdaOut<V>,
    rust: &TdaModes<T>,
    tol: f64,
    layer: &str,
) {
    assert_eq!(out.rc, QNTT_OK, "{layer}: {}", last_error());
    assert_eq!(out.theta.len(), K, "{layer}: k roots");
    assert!(
        out.theta.windows(2).all(|w| w[0] <= w[1]),
        "{layer}: θ not ascending {:?}",
        out.theta
    );
    assert_eq!(out.conv, 1, "{layer}: not converged");
    assert_eq!(out.cert, 1, "{layer}: an uncertified negative θ");
    assert_eq!(out.residual.len(), K, "{layer}: one residual per root");
    assert!(
        out.residual
            .iter()
            .all(|r| r.is_finite() && *r <= 10.0 * tol),
        "{layer}: fresh residuals {:?} above 10·tol",
        out.residual
    );
    assert_eq!(out.theta, rust.theta, "{layer}: θ vs the Rust entry");
    assert_eq!(out.residual, rust.info.residuals, "{layer}: residuals");
    assert_eq!(out.iters, rust.info.iterations, "{layer}: iterations");
    assert_eq!(out.applies, rust.info.applies, "{layer}: applies");
    assert_eq!(out.space_dim, rust.info.space_dim, "{layer}: space dim");
    assert_eq!(out.conv, rust.info.converged as i32, "{layer}: converged");
    assert_eq!(
        out.precond, QNTT_TDA_PRECOND_DIAG,
        "{layer}: AUTO runs as DIAG (kappa 10)"
    );
    assert_eq!(rust.info.precond, TdaPrecond::Diag { kappa: 10.0 });
}

fn su2_h4_ham_and_ground_state(
    r: usize,
) -> (
    *mut Handle<qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>>,
    *mut Handle<qntt::su2::vector::Su2TtVector<f64>>,
) {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    assert_eq!(ms2, 0, "the H4 fixture is a singlet");
    let mut h: *mut _ = ptr::null_mut();
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
        QNTT_OK,
        "{}",
        last_error()
    );
    let flat = su2_capped_ranks(norb, nelec, 0, r);
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_vec_randn(norb, nelec, 0, flat.as_ptr(), flat.len(), 7, &mut x0) },
        QNTT_OK
    );
    let mut opts = QnttAlsOptions::defaults();
    opts.eps = 1e-10;
    opts.max_iter = 50;
    let mut lambda = 0.0f64;
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_als(h, x0, &opts, &mut lambda, &mut x) },
        QNTT_OK,
        "{}",
        last_error()
    );
    unsafe { qntt_su2_f64_vec_free(x0) };
    (h, x)
}

macro_rules! check_states {
    ($out:expr, $h:expr, $x:expr, $rayleigh:ident, $norm:ident, $dot:ident, $free:ident,
     $layer:expr) => {{
        let out = &$out;
        assert_eq!(out.states_len, K, "{}: one handle per root", $layer);
        let mut lambda = f64::NAN;
        assert_eq!(unsafe { $rayleigh($h, $x, &mut lambda) }, QNTT_OK);
        for (j, &s) in out.states.iter().enumerate() {
            assert!(!s.is_null(), "{}: state {j} is null", $layer);
            let (mut nrm, mut e, mut ov) = (f64::NAN, f64::NAN, f64::NAN);
            assert_eq!(unsafe { $norm(s, &mut nrm) }, QNTT_OK);
            assert_eq!(unsafe { $rayleigh($h, s, &mut e) }, QNTT_OK);
            assert_eq!(unsafe { $dot($x, s, &mut ov) }, QNTT_OK);
            assert!((nrm - 1.0).abs() < 1e-12, "{}: ‖s{j}‖ = {nrm}", $layer);
            assert!(ov.abs() < 1e-8, "{}: ⟨x, s{j}⟩ = {ov:e}", $layer);
            assert!(
                (e - (lambda + out.theta[j])).abs() < 1e-8,
                "{}: E(s{j}) = {e} vs λ + θ = {}",
                $layer,
                lambda + out.theta[j]
            );
            unsafe { $free(s) };
        }
    }};
}

#[test]
fn tda_f64_is_the_rust_entry_and_its_states_carry_lambda_plus_theta() {
    let (h, x) = h4_ham_and_ground_state(2);
    let opts = QnttTdaOpts::defaults();
    let out = tda_call!(qntt_f64_tda, qntt::TtVector<f64>, h, x, K, &opts, K, true);
    let rust = rust_tda!(
        qntt::solvers::hess::weingarten_operator,
        qntt::solvers::hess::tda_lowest_davidson,
        qntt::tt_randn,
        TtTangent<f64>,
        h,
        x,
        K,
        &TdaDavidsonOpts::default()
    );
    check_against_rust(&out, &rust, opts.tol, "f64");
    check_states!(
        out,
        h,
        x,
        qntt_f64_ham_rayleigh,
        qntt_f64_vec_norm,
        qntt_f64_vec_dot,
        qntt_f64_vec_free,
        "f64"
    );

    let bare = tda_call!(qntt_f64_tda, qntt::TtVector<f64>, h, x, K, &opts, K, false);
    assert_eq!(bare.rc, QNTT_OK);
    assert_eq!(bare.states_len, 0);
    assert_eq!(bare.theta, out.theta);

    for (code, want) in [
        (QNTT_TDA_PRECOND_OFF, QNTT_TDA_PRECOND_OFF),
        (QNTT_TDA_PRECOND_DIAG, QNTT_TDA_PRECOND_DIAG),
        (QNTT_TDA_PRECOND_SHIFTED, QNTT_TDA_PRECOND_SHIFTED),
    ] {
        let o = QnttTdaOpts {
            precond: code,
            ..opts
        };
        let arm = tda_call!(qntt_f64_tda, qntt::TtVector<f64>, h, x, K, &o, K, false);
        assert_eq!(arm.rc, QNTT_OK, "precond {code}: {}", last_error());
        assert_eq!(arm.precond, want, "precond {code}: echo");
        assert_eq!(arm.conv, 1, "precond {code}: not converged");
        for (a, b) in arm.theta.iter().zip(&out.theta) {
            assert!((a - b).abs() < 1e-7, "precond {code}: θ {a} vs {b}");
        }
    }
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn tda_su2_f64_is_the_rust_entry_and_its_states_carry_lambda_plus_theta() {
    let (h, x) = su2_h4_ham_and_ground_state(2);
    let opts = QnttTdaOpts::defaults();
    let out = tda_call!(
        qntt_su2_f64_tda,
        qntt::su2::vector::Su2TtVector<f64>,
        h,
        x,
        K,
        &opts,
        K,
        true
    );
    let rust = rust_tda!(
        qntt::su2::hess::weingarten_operator,
        qntt::su2::hess::tda_lowest_davidson,
        qntt::su2::vector::tt_randn,
        Su2TtTangent<f64>,
        h,
        x,
        K,
        &TdaDavidsonOpts::default()
    );
    check_against_rust(&out, &rust, opts.tol, "su2_f64");
    check_states!(
        out,
        h,
        x,
        qntt_su2_f64_ham_rayleigh,
        qntt_su2_f64_vec_norm,
        qntt_su2_f64_vec_dot,
        qntt_su2_f64_vec_free,
        "su2_f64"
    );
    unsafe {
        qntt_su2_f64_vec_free(x);
        qntt_su2_f64_ham_free(h);
    }
}

#[test]
fn tda_refuses_every_caller_error_before_the_solve() {
    let (h, x) = h4_ham_and_ground_state(2);
    let good = QnttTdaOpts::defaults();
    let refused = |out: &TdaOut<qntt::TtVector<f64>>, code: i32, what: &str| {
        assert_eq!(out.rc, code, "{what}: {}", last_error());
        assert!(
            out.iters == usize::MAX && out.applies == usize::MAX && out.theta.is_empty(),
            "{what}: a solve ran"
        );
    };
    for (o, what, needle) in [
        (
            QnttTdaOpts { precond: 7, ..good },
            "unknown precond",
            "TDA precond 7",
        ),
        (
            QnttTdaOpts {
                precond: -1,
                ..good
            },
            "negative precond",
            "TDA precond -1",
        ),
        (
            QnttTdaOpts { tol: -1e-8, ..good },
            "negative tol",
            "TDA tol",
        ),
        (
            QnttTdaOpts {
                tol: f64::NAN,
                ..good
            },
            "NaN tol",
            "TDA tol",
        ),
        (
            QnttTdaOpts {
                verbosity: -1,
                ..good
            },
            "negative verbosity",
            "TDA verbosity",
        ),
    ] {
        let out = tda_call!(qntt_f64_tda, qntt::TtVector<f64>, h, x, K, &o, K, true);
        refused(&out, QNTT_INVALID_ARG, what);
        assert!(last_error().contains(needle), "{what}: {}", last_error());
    }
    let short = tda_call!(
        qntt_f64_tda,
        qntt::TtVector<f64>,
        h,
        x,
        K,
        &good,
        K - 1,
        true
    );
    refused(&short, QNTT_INVALID_ARG, "short buffer");
    assert!(last_error().contains("need k = 3"), "{}", last_error());
    {
        let mut theta = vec![0.0; K];
        let mut slots: Vec<*mut Handle<qntt::TtVector<f64>>> = vec![ptr::null_mut(); K - 1];
        let mut iters = usize::MAX;
        let rc = unsafe {
            qntt_f64_tda(
                h,
                x,
                K,
                &good,
                SEED,
                theta.as_mut_ptr(),
                K,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                slots.as_mut_ptr(),
                K - 1,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                &mut iters,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        assert_eq!(rc, QNTT_INVALID_ARG);
        assert!(last_error().contains("out_states"), "{}", last_error());
        assert_eq!(iters, usize::MAX, "a solve ran");
        assert!(slots.iter().all(|s| s.is_null()), "a handle leaked out");
    }
    let mut theta = vec![0.0; K];
    let null_opts = unsafe {
        qntt_f64_tda(
            h,
            x,
            K,
            ptr::null(),
            SEED,
            theta.as_mut_ptr(),
            K,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(null_opts, QNTT_NULL_ARG);
    let null_theta = unsafe {
        qntt_f64_tda(
            h,
            x,
            K,
            &good,
            SEED,
            ptr::null_mut(),
            K,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(null_theta, QNTT_NULL_ARG);
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn tda_defaults_and_wire_codes_match_the_library_and_the_header() {
    let mut o = QnttTdaOpts {
        tol: f64::NAN,
        max_iter: 0,
        max_subspace: 9,
        restart_keep: 9,
        nwarm: 9,
        max_concurrent: 9,
        precond_kappa: f64::NAN,
        precond: -1,
        verbosity: -1,
    };
    assert_eq!(unsafe { qntt_tda_options_default(&mut o) }, QNTT_OK);
    let lib = TdaDavidsonOpts::default();
    let core = o.to_core().unwrap();
    assert_eq!(core.tol, lib.tol);
    assert_eq!(core.max_iter, lib.max_iter);
    assert_eq!(core.max_subspace, lib.max_subspace);
    assert_eq!(core.restart_keep, lib.restart_keep);
    assert_eq!(core.nwarm, lib.nwarm);
    assert_eq!(core.max_concurrent, lib.max_concurrent);
    assert_eq!(core.precond, lib.precond);
    assert_eq!(core.verbosity, lib.verbosity);
    assert_eq!(o.precond, QNTT_TDA_PRECOND_AUTO);
    assert_eq!(
        TdaPrecond::Diag {
            kappa: o.precond_kappa
        },
        TdaPrecond::Auto.resolve(),
        "precond_kappa is the cap AUTO resolves to"
    );
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let c = QnttTdaOpts {
            precond: QNTT_TDA_PRECOND_DIAG,
            precond_kappa: bad,
            ..o
        }
        .to_core()
        .unwrap();
        assert_eq!(c.precond, TdaPrecond::Diag { kappa: 0.0 }, "kappa {bad}");
    }
    let loud = QnttTdaOpts {
        verbosity: 1000,
        ..o
    };
    assert_eq!(loud.to_core().unwrap().verbosity, u8::MAX);
    assert_eq!(
        QnttTdaOpts { tol: 0.0, ..o }.to_core().unwrap().tol,
        0.0,
        "tol = 0 is legal"
    );

    let hdr = include_str!("../../include/qntt.h");
    let define = |name: &str| -> i32 {
        let pat = format!("#define {name} ");
        let i = hdr
            .find(&pat)
            .unwrap_or_else(|| panic!("qntt.h defines no {name}"));
        hdr[i + pat.len()..]
            .split_whitespace()
            .next()
            .expect("a value")
            .parse()
            .expect("an integer literal")
    };
    assert_eq!(define("QNTT_TDA_PRECOND_AUTO"), QNTT_TDA_PRECOND_AUTO);
    assert_eq!(define("QNTT_TDA_PRECOND_OFF"), QNTT_TDA_PRECOND_OFF);
    assert_eq!(define("QNTT_TDA_PRECOND_DIAG"), QNTT_TDA_PRECOND_DIAG);
    assert_eq!(define("QNTT_TDA_PRECOND_SHIFTED"), QNTT_TDA_PRECOND_SHIFTED);
}
