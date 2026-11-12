// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn fcidump_read_matches_the_rust_reader() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let direct = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).expect("read h4");
    assert_eq!((norb, nelec, ms2), (direct.norb, direct.nelec, direct.ms2));
    assert_eq!(t, direct.t.data);
    assert_eq!(v, direct.v.data);
}

#[test]
fn als_through_the_abi_matches_the_rust_api() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);

    let mut h: *mut _ = ptr::null_mut();
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
        QNTT_OK,
        "ham_new failed: {}",
        last_error()
    );

    let ranks = uniform_ranks(space, 4);
    let flat = flat_ranks(&space, &ranks);
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                2024,
                &mut x0,
            )
        },
        QNTT_OK
    );

    let mut opts = QnttAlsOptions::defaults();
    opts.eps = 1e-8;
    opts.max_iter = 30;
    let mut lambda_abi = 0.0f64;
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_als(h, x0, &opts, &mut lambda_abi, &mut x) },
        QNTT_OK,
        "als failed: {}",
        last_error()
    );

    let direct_data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let hd = SparseHamiltonian::new(space, &direct_data.t, &direct_data.v, None, f64::EPSILON);
    use rand::SeedableRng;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(2024);
    let x0d = qntt::tt_randn::<f64, _>(space, &ranks, &mut rng);
    let (lambda_direct, _) = qntt::solvers::als::als(
        &hd,
        &x0d,
        &qntt::solvers::als::AlsOptions {
            eps: 1e-8,
            max_iter: 30,
            verbose: false,
        },
    );

    assert!(
        (lambda_abi - lambda_direct).abs() <= 1e-12 * lambda_direct.abs().max(1.0),
        "ABI λ = {lambda_abi:.15}, direct λ = {lambda_direct:.15}"
    );

    let mut rq = 0.0f64;
    assert_eq!(unsafe { qntt_f64_ham_rayleigh(h, x, &mut rq) }, QNTT_OK);
    assert!(
        (rq - lambda_abi).abs() <= 1e-8 * lambda_abi.abs().max(1.0),
        "rayleigh {rq} vs λ {lambda_abi}"
    );

    let mut hx: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_ham_apply(h, x, &mut hx) }, QNTT_OK);
    let mut xhx = 0.0f64;
    assert_eq!(unsafe { qntt_f64_vec_dot(x, hx, &mut xhx) }, QNTT_OK);
    let mut nx = 0.0f64;
    assert_eq!(unsafe { qntt_f64_vec_norm(x, &mut nx) }, QNTT_OK);
    assert!(
        (xhx / (nx * nx) - lambda_abi).abs() <= 1e-8 * lambda_abi.abs().max(1.0),
        "⟨x,Hx⟩/⟨x,x⟩ = {} vs λ = {lambda_abi}",
        xhx / (nx * nx)
    );

    for hv in [x0, x, hx] {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
fn mals_through_the_abi_lowers_the_energy() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);

    let mut h: *mut _ = ptr::null_mut();
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

    let ranks = uniform_ranks(space, 2);
    let flat = flat_ranks(&space, &ranks);
    let mut x0: *mut _ = ptr::null_mut();
    unsafe {
        qntt_f64_vec_randn(
            space.d,
            space.nup,
            space.ndn,
            flat.as_ptr(),
            flat.len(),
            11,
            &mut x0,
        )
    };

    let mut rq0 = 0.0f64;
    assert_eq!(unsafe { qntt_f64_ham_rayleigh(h, x0, &mut rq0) }, QNTT_OK);
    let mut expect0 = 0.0f64;
    assert_eq!(
        unsafe { qntt_f64_ham_expectation(h, x0, &mut expect0) },
        QNTT_OK
    );
    let mut n0 = 0.0f64;
    unsafe { qntt_f64_vec_norm(x0, &mut n0) };
    assert!(
        (rq0 - expect0 / (n0 * n0)).abs() <= 1e-10 * rq0.abs().max(1.0),
        "rayleigh must be the expectation divided by ⟨x,x⟩"
    );

    let mut opts = QnttMalsOptions::defaults();
    opts.eps = 1e-8;
    opts.max_iter = 20;
    let bond_dims = [0usize];
    let mut lambda = 0.0f64;
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_mals(
                h,
                x0,
                &opts,
                bond_dims.as_ptr(),
                bond_dims.len(),
                ptr::null(),
                0,
                &mut lambda,
                &mut x,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "mals failed: {}",
        last_error()
    );
    assert!(
        lambda < rq0,
        "MALS should lower the Rayleigh quotient ({lambda} vs start {rq0})"
    );

    let status = unsafe {
        qntt_f64_mals(
            h,
            x0,
            &opts,
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut lambda,
            &mut ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, QNTT_INVALID_ARG);
    assert!(last_error().contains("bond_dims"), "got {:?}", last_error());

    unsafe { qntt_f64_vec_free(x0) };
    unsafe { qntt_f64_vec_free(x) };
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
fn mals_noise_schedule_and_history_cross_the_abi() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);

    let mut h: *mut _ = ptr::null_mut();
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

    let occ: Vec<i32> = (0..space.d).map(|k| (k < space.nup) as i32).collect();
    let mut hf: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_tt_state(occ.as_ptr(), occ.as_ptr(), space.d, &mut hf) },
        QNTT_OK
    );

    let mut opts = QnttMalsOptions::defaults();
    opts.eps = 1e-12;
    opts.max_iter = 30;
    opts.seed = 3;
    let bond_dims = [0usize];
    let noise = [1e-1f64, 1e-2, 1e-3, 1e-4, 0.0, 0.0, 0.0];

    let (mut lam_quiet, mut n_quiet) = (0.0f64, 0usize);
    let mut xq: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_mals(
                h,
                hf,
                &opts,
                bond_dims.as_ptr(),
                bond_dims.len(),
                ptr::null(),
                0,
                &mut lam_quiet,
                &mut xq,
                ptr::null_mut(),
                0,
                &mut n_quiet,
            )
        },
        QNTT_OK
    );

    let mut sweeps = vec![QnttMalsSweep::default(); opts.max_iter];
    let (mut lam_noisy, mut n_noisy) = (0.0f64, 0usize);
    let mut xn: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_mals(
                h,
                hf,
                &opts,
                bond_dims.as_ptr(),
                bond_dims.len(),
                noise.as_ptr(),
                noise.len(),
                &mut lam_noisy,
                &mut xn,
                sweeps.as_mut_ptr(),
                sweeps.len(),
                &mut n_noisy,
            )
        },
        QNTT_OK,
        "noisy mals failed: {}",
        last_error()
    );

    assert!(n_quiet > 0 && n_noisy > 0);
    assert!(
        lam_noisy < lam_quiet - 1e-6,
        "noise did not escape the determinant trap through the ABI: {lam_noisy} vs {lam_quiet}"
    );

    for (i, s) in sweeps.iter().take(n_noisy).enumerate() {
        assert_eq!(s.noise, noise[i.min(noise.len() - 1)]);
        assert!(s.discarded_weight >= 0.0 && s.discarded_weight <= 1.0);
        assert!(s.max_bond_dim > 0);
    }
    assert_eq!(
        sweeps[n_noisy - 1].noise,
        0.0,
        "the schedule must end noise-free"
    );
    assert_eq!(sweeps[n_noisy - 1].energy, lam_noisy);

    let bad = [-1.0f64];
    let status = unsafe {
        qntt_f64_mals(
            h,
            hf,
            &opts,
            bond_dims.as_ptr(),
            bond_dims.len(),
            bad.as_ptr(),
            bad.len(),
            &mut lam_noisy,
            &mut ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, QNTT_INVALID_ARG);
    assert!(last_error().contains("noise"), "got {:?}", last_error());

    for hv in [hf, xq, xn] {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}
