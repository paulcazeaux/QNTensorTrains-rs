// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn complex_hamiltonian_gives_a_real_energy() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
    let tc: Vec<C64> = t.iter().map(|&x| C64::new(x, 0.0)).collect();
    let vc: Vec<C64> = v.iter().map(|&x| C64::new(x, 0.0)).collect();

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
        QNTT_OK,
        "c64 ham_new failed: {}",
        last_error()
    );

    let ranks = uniform_ranks(space, 3);
    let flat = flat_ranks(&space, &ranks);
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
    opts.eps = 1e-8;
    opts.max_iter = 20;
    let mut lambda = C64::new(0.0, 0.0);
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_c64_als(h, x0, &opts, &mut lambda, &mut x) },
        QNTT_OK,
        "c64 als failed: {}",
        last_error()
    );
    assert!(
        lambda.im.abs() <= 1e-10 * lambda.re.abs().max(1.0),
        "a Hermitian operator must give a real energy, got {lambda}"
    );

    let direct_data = read_fcidump::<f64>(&h4_path(), TwoBodySymmetry::Eightfold).unwrap();
    let hd = SparseHamiltonian::new(space, &direct_data.t, &direct_data.v, None, f64::EPSILON);
    use rand::SeedableRng;
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(5);
    let x0d = qntt::tt_randn::<f64, _>(space, &ranks, &mut rng);
    let (lambda_real, _) = qntt::solvers::als::als(
        &hd,
        &x0d,
        &qntt::solvers::als::AlsOptions {
            eps: 1e-8,
            max_iter: 20,
            verbose: false,
        },
    );
    assert!(
        (lambda.re - lambda_real).abs() <= 1e-6 * lambda_real.abs().max(1.0),
        "complex λ = {}, real λ = {lambda_real}",
        lambda.re
    );

    unsafe { qntt_c64_vec_free(x0) };
    unsafe { qntt_c64_vec_free(x) };
    unsafe { qntt_c64_ham_free(h) };
}

#[test]
fn su2_space_and_csf_are_reachable() {
    assert_eq!(qntt_su2_space_check(4, 4, 0), QNTT_OK);
    assert_ne!(qntt_su2_space_check(4, 4, 6), QNTT_OK);

    let (mut rows, mut cols) = (0usize, 0usize);
    assert_eq!(
        unsafe { qntt_su2_space_grid_extent(4, 4, 0, &mut rows, &mut cols) },
        QNTT_OK
    );
    assert_eq!((rows, cols), (5, 5), "SU(2) grid is (N+1)×(N+1)");

    let codes = [3i32, 3, 0, 0];
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_tt_csf(codes.as_ptr(), codes.len(), &mut x) },
        QNTT_OK,
        "tt_csf failed: {}",
        last_error()
    );
    let mut norm = 0.0f64;
    assert_eq!(unsafe { qntt_su2_f64_vec_norm(x, &mut norm) }, QNTT_OK);
    assert!((norm - 1.0).abs() < 1e-12, "a CSF is unit-norm, got {norm}");

    let (mut d, mut n, mut two_s) = (0usize, 0usize, 0usize);
    assert_eq!(
        unsafe { qntt_su2_f64_vec_space(x, &mut d, &mut n, &mut two_s) },
        QNTT_OK
    );
    assert_eq!((d, n, two_s), (4, 4, 0));

    unsafe { qntt_su2_f64_vec_free(x) };
}

#[test]
fn su2_als_matches_the_abelian_ground_state() {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    assert_eq!(ms2, 0, "the H4 fixture is a singlet");
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);

    let mut hs: *mut _ = ptr::null_mut();
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
                &mut hs,
            )
        },
        QNTT_OK,
        "su2 ham_new failed: {}",
        last_error()
    );

    let su2_flat = su2_capped_ranks(norb, nelec, 0, 64);
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_randn(
                norb,
                nelec,
                0,
                su2_flat.as_ptr(),
                su2_flat.len(),
                17,
                &mut x0,
            )
        },
        QNTT_OK,
        "su2 randn failed: {}",
        last_error()
    );

    let mut opts = QnttMalsOptions::defaults();
    opts.eps = 1e-10;
    opts.max_iter = 40;
    let bond_dims = [0usize];
    let mut lambda_su2 = 0.0f64;
    let mut xs: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_mals(
                hs,
                x0,
                &opts,
                bond_dims.as_ptr(),
                bond_dims.len(),
                ptr::null(),
                0,
                &mut lambda_su2,
                &mut xs,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            )
        },
        QNTT_OK,
        "su2 mals failed: {}",
        last_error()
    );

    let mut ha: *mut _ = ptr::null_mut();
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
                &mut ha,
            )
        },
        QNTT_OK
    );
    let ranks = uniform_ranks(space, 64);
    let flat = flat_ranks(&space, &ranks);
    let mut xa0: *mut _ = ptr::null_mut();
    unsafe {
        qntt_f64_vec_randn(
            space.d,
            space.nup,
            space.ndn,
            flat.as_ptr(),
            flat.len(),
            3,
            &mut xa0,
        )
    };
    let mut lambda_ab = 0.0f64;
    let mut xa: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_mals(
                ha,
                xa0,
                &opts,
                bond_dims.as_ptr(),
                bond_dims.len(),
                ptr::null(),
                0,
                &mut lambda_ab,
                &mut xa,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            )
        },
        QNTT_OK
    );

    assert!(
        (lambda_su2 - lambda_ab).abs() <= 1e-6 * lambda_ab.abs().max(1.0),
        "SU(2) λ = {lambda_su2:.12}, abelian λ = {lambda_ab:.12}"
    );

    for hv in [x0, xs] {
        unsafe { qntt_su2_f64_vec_free(hv) };
    }
    for hv in [xa0, xa] {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_su2_f64_ham_free(hs) };
    unsafe { qntt_f64_ham_free(ha) };
}
