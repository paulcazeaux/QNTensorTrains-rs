// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn pd_probe_reports_a_verdict_and_never_fails_on_one() {
    let (h, x) = h4_ham_and_ground_state(2);

    for part in [0i32, 1] {
        let (mut verdict, mut is_pd) = (-99i32, -99i32);
        let (mut lambda, mut residual) = (f64::NAN, f64::NAN);
        let mut applies = 0usize;
        assert_eq!(
            unsafe {
                qntt_f64_pd_probe(
                    h,
                    x,
                    part,
                    40,
                    1234,
                    &mut verdict,
                    &mut lambda,
                    &mut residual,
                    &mut is_pd,
                    &mut applies,
                )
            },
            QNTT_OK,
            "a verdict is a diagnosis, not a failure (part {part}): {}",
            last_error()
        );
        assert!((0..=2).contains(&verdict), "verdict code {verdict}");
        assert!(applies > 0, "the probe must spend at least one apply");
        let expect_is_pd = match verdict {
            0 => 0,
            1 => 1,
            _ => -1,
        };
        assert_eq!(is_pd, expect_is_pd, "is_pd must track the verdict exactly");
        if verdict == 0 {
            assert!(lambda < 0.0, "a witness verdict means λ < 0, got {lambda}");
            assert!(residual.is_nan(), "residual is unspecified on a witness");
        } else {
            assert!(lambda >= 0.0, "λ = {lambda} with verdict {verdict}");
            assert!(
                residual.is_finite() && residual >= 0.0,
                "residual {residual}"
            );
        }
    }

    let mut verdict = -99i32;
    assert_eq!(
        unsafe {
            qntt_f64_pd_probe(
                h,
                x,
                0,
                20,
                1,
                &mut verdict,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_OK
    );
    assert!((0..=2).contains(&verdict));
    assert_eq!(
        unsafe {
            qntt_f64_pd_probe(
                h,
                x,
                0,
                20,
                1,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        qntt_capi::err::QNTT_NULL_ARG
    );
    assert_eq!(
        unsafe {
            qntt_f64_pd_probe(
                h,
                x,
                7,
                20,
                1,
                &mut verdict,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_INVALID_ARG
    );

    unsafe { qntt_f64_vec_free(x) };
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
fn c64_pd_probe_takes_rr_as_the_complex_hessian_and_refuses_ii() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
    let tc: Vec<C64> = t.iter().map(|&z| C64::new(z, 0.0)).collect();
    let vc: Vec<C64> = v.iter().map(|&z| C64::new(z, 0.0)).collect();
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_c64_ham_new(
                space.d,
                space.nup,
                space.ndn,
                tc.as_ptr(),
                tc.len(),
                vc.as_ptr(),
                vc.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK
    );
    let flat = flat_ranks(&space, &uniform_ranks(space, 2));
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_c64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                5,
                &mut x0,
            )
        },
        QNTT_OK
    );
    let mut opts = QnttAlsOptions::defaults();
    opts.eps = 1e-10;
    opts.max_iter = 60;
    let mut lambda = C64::new(0.0, 0.0);
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_c64_als(h, x0, &opts, &mut lambda, &mut x) },
        QNTT_OK
    );

    let mut verdict = -99i32;
    let mut lam = f64::NAN;
    assert_eq!(
        unsafe {
            qntt_c64_pd_probe(
                h,
                x,
                0,
                40,
                3,
                &mut verdict,
                &mut lam,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "complex 𝔥 probe failed: {}",
        last_error()
    );
    assert!((0..=2).contains(&verdict));
    assert!(lam.is_finite());

    assert_eq!(
        unsafe {
            qntt_c64_pd_probe(
                h,
                x,
                1,
                10,
                3,
                &mut verdict,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        },
        QNTT_INVALID_ARG
    );
    assert!(
        last_error().contains("part = 0"),
        "message: {}",
        last_error()
    );

    for hv in [x0, x] {
        unsafe { qntt_c64_vec_free(hv) };
    }
    unsafe { qntt_c64_ham_free(h) };
}

#[test]
fn vec_to_c64_is_the_identity_on_amplitudes() {
    let space = QnSpace::new(6, 3, 3);
    let flat = flat_ranks(&space, &uniform_ranks(space, 3));
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                11,
                &mut x,
            )
        },
        QNTT_OK
    );
    let mut y: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_to_c64(x, &mut y) },
        QNTT_OK,
        "vec_to_c64 failed: {}",
        last_error()
    );

    let (mut nx, mut ny) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_f64_vec_norm(x, &mut nx) }, QNTT_OK);
    assert_eq!(unsafe { qntt_c64_vec_norm(y, &mut ny) }, QNTT_OK);
    assert!(nx > 0.0);
    assert_eq!(nx, ny, "the embedding does not touch a single coefficient");

    let mut got = vec![0usize; flat.len()];
    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_c64_vec_ranks(y, got.as_mut_ptr(), got.len(), &mut len) },
        QNTT_OK
    );
    assert_eq!(got, flat);

    let n = 1usize << (2 * space.d);
    let mut xd = vec![0.0f64; n];
    let mut yd = vec![C64::new(0.0, 0.0); n];
    assert_eq!(
        unsafe { qntt_f64_vec_to_dense(x, xd.as_mut_ptr(), n, &mut len) },
        QNTT_OK
    );
    assert_eq!(
        unsafe { qntt_c64_vec_to_dense(y, yd.as_mut_ptr(), n, &mut len) },
        QNTT_OK
    );
    for (i, (&a, &b)) in xd.iter().zip(yd.iter()).enumerate() {
        assert_eq!(b, C64::new(a, 0.0), "amplitude {i}");
    }

    let mut before = 0isize;
    assert_eq!(
        unsafe { qntt_f64_vec_left_orthonormalize(x, 1, ptr::null_mut()) },
        QNTT_OK
    );
    assert_eq!(unsafe { qntt_f64_vec_ortho(x, &mut before) }, QNTT_OK);
    assert!(
        before >= 0,
        "left_orthonormalize must set an orthogonality centre"
    );
    let mut yc: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_to_c64(x, &mut yc) }, QNTT_OK);
    let mut after = -99isize;
    assert_eq!(unsafe { qntt_c64_vec_ortho(yc, &mut after) }, QNTT_OK);
    assert_eq!(
        after, before,
        "the orthogonality centre must survive the cast"
    );
    let (mut nxo, mut nyo) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_f64_vec_norm(x, &mut nxo) }, QNTT_OK);
    assert_eq!(unsafe { qntt_c64_vec_norm(yc, &mut nyo) }, QNTT_OK);
    assert!((nxo - nyo).abs() <= 1e-13 * nxo.max(1.0), "{nxo} vs {nyo}");

    unsafe { qntt_c64_vec_free(yc) };
    unsafe { qntt_f64_vec_free(x) };
    unsafe { qntt_c64_vec_free(y) };
}

#[test]
fn complex_seed_either_leaves_the_real_manifold_or_says_why_not() {
    let (h, x) = h4_ham_and_ground_state(2);

    let mut lambda = f64::NAN;
    let mut y: *mut _ = ptr::null_mut();
    let code = unsafe { qntt_f64_complex_seed(h, x, 1e-3, 60, 99, &mut lambda, &mut y) };
    if code == QNTT_OK {
        assert!(
            lambda < 0.0,
            "a seed implies a proven negative λ, got {lambda}"
        );
        let mut ny = 0.0f64;
        assert_eq!(unsafe { qntt_c64_vec_norm(y, &mut ny) }, QNTT_OK);
        assert!((ny - 1.0).abs() < 1e-10, "the seed is normalized, got {ny}");
        let mut ok = 0i32;
        assert_eq!(unsafe { qntt_c64_vec_check(y, &mut ok) }, QNTT_OK);
        assert_eq!(ok, 1);
        unsafe { qntt_c64_vec_free(y) };
    } else {
        assert_eq!(code, qntt_capi::err::QNTT_NOT_CONVERGED, "{}", last_error());
        assert!(
            last_error().contains("H^ii"),
            "the message must name what was probed: {}",
            last_error()
        );
        assert!(y.is_null(), "no handle may be emitted on the error path");
    }

    assert_eq!(
        unsafe { qntt_f64_complex_seed(h, x, 0.0, 10, 1, ptr::null_mut(), &mut y) },
        QNTT_INVALID_ARG
    );

    unsafe { qntt_f64_vec_free(x) };
    unsafe { qntt_f64_ham_free(h) };
}
