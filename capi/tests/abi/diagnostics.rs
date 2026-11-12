// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn bilinear_dot_is_the_unconjugated_contraction_and_detects_a_real_state() {
    let space = QnSpace::new(4, 2, 2);
    let flat = flat_ranks(&space, &uniform_ranks(space, 2));

    let mut xr: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                3,
                &mut xr,
            )
        },
        QNTT_OK
    );
    let (mut herm, mut bil) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_f64_vec_dot(xr, xr, &mut herm) }, QNTT_OK);
    assert_eq!(
        unsafe { qntt_f64_vec_dot_bilinear(xr, xr, &mut bil) },
        QNTT_OK,
        "dot_bilinear failed: {}",
        last_error()
    );
    assert_eq!(
        herm, bil,
        "over f64 the bilinear and Hermitian forms are one map"
    );

    let mut xc: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_to_c64(xr, &mut xc) }, QNTT_OK);
    let (mut hc, mut bc) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0));
    assert_eq!(unsafe { qntt_c64_vec_dot(xc, xc, &mut hc) }, QNTT_OK);
    assert_eq!(
        unsafe { qntt_c64_vec_dot_bilinear(xc, xc, &mut bc) },
        QNTT_OK
    );
    let rho_real = bc.norm() / hc.re;
    assert!(
        (rho_real - 1.0).abs() < 1e-12,
        "a real-embedded state must read rho = 1, got {rho_real}"
    );

    let phase = C64::new(0.6, 0.8);
    let mut xp: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_c64_vec_scaled(xc, &phase, &mut xp) }, QNTT_OK);
    let (mut hp, mut bp) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0));
    assert_eq!(unsafe { qntt_c64_vec_dot(xp, xp, &mut hp) }, QNTT_OK);
    assert_eq!(
        unsafe { qntt_c64_vec_dot_bilinear(xp, xp, &mut bp) },
        QNTT_OK
    );
    let rho_phased = bp.norm() / hp.re;
    assert!(
        (rho_phased - 1.0).abs() < 1e-12,
        "rho must be invariant under a global phase, got {rho_phased}"
    );

    let mut yr: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                99,
                &mut yr,
            )
        },
        QNTT_OK
    );
    let mut yc: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_to_c64(yr, &mut yc) }, QNTT_OK);
    let i = C64::new(0.0, 1.0);
    let mut iy: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_c64_vec_scaled(yc, &i, &mut iy) }, QNTT_OK);
    let mut z: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_c64_vec_add(xc, iy, 1, &mut z) }, QNTT_OK);
    let (mut hz, mut bz) = (C64::new(0.0, 0.0), C64::new(0.0, 0.0));
    assert_eq!(unsafe { qntt_c64_vec_dot(z, z, &mut hz) }, QNTT_OK);
    assert_eq!(unsafe { qntt_c64_vec_dot_bilinear(z, z, &mut bz) }, QNTT_OK);
    let rho_complex = bz.norm() / hz.re;
    assert!(
        rho_complex < 1.0 - 1e-6,
        "x + i·y is not real up to a phase, but rho read {rho_complex}"
    );

    for hv in [xr, yr] {
        unsafe { qntt_f64_vec_free(hv) };
    }
    for hv in [xc, xp, yc, iy, z] {
        unsafe { qntt_c64_vec_free(hv) };
    }
}

#[test]
fn schmidt_spectrum_crosses_with_per_bond_counts() {
    let space = QnSpace::new(6, 3, 3);
    let flat = flat_ranks(&space, &uniform_ranks(space, 3));
    let mut raw: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                21,
                &mut raw,
            )
        },
        QNTT_OK
    );
    let mut nraw = 0.0f64;
    assert_eq!(unsafe { qntt_f64_vec_norm(raw, &mut nraw) }, QNTT_OK);
    assert!(nraw > 0.0);
    let inv = 1.0 / nraw;
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_scaled(raw, &inv, &mut x) }, QNTT_OK);
    unsafe { qntt_f64_vec_free(raw) };

    let (mut len, mut nbonds) = (0usize, 0usize);
    assert_eq!(
        unsafe {
            qntt_f64_vec_schmidt(
                x,
                ptr::null_mut(),
                0,
                &mut len,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
                &mut nbonds,
            )
        },
        QNTT_OK,
        "schmidt size query failed: {}",
        last_error()
    );
    assert_eq!(nbonds, space.d - 1, "one entry per INTERIOR bond");
    assert!(len > 0);

    let mut vals = vec![0.0f64; len];
    let mut wts = vec![0.0f64; len];
    let mut counts = vec![0usize; nbonds];
    assert_eq!(
        unsafe {
            qntt_f64_vec_schmidt(
                x,
                vals.as_mut_ptr(),
                vals.len(),
                &mut len,
                wts.as_mut_ptr(),
                wts.len(),
                counts.as_mut_ptr(),
                counts.len(),
                &mut nbonds,
            )
        },
        QNTT_OK
    );
    assert!(
        wts.iter().all(|&w| w == 1.0),
        "the abelian layer has no multiplet degeneracy: every weight must be exactly 1"
    );
    assert_eq!(
        counts.iter().sum::<usize>(),
        len,
        "counts must partition the values"
    );

    let mut off = 0usize;
    for (b, &c) in counts.iter().enumerate() {
        let block = &vals[off..off + c];
        assert!(
            !block.is_empty(),
            "bond {} carries no Schmidt values",
            b + 1
        );
        for w in block.windows(2) {
            assert!(w[0] >= w[1], "bond {} is not sorted descending", b + 1);
        }
        assert!(
            block.iter().all(|&s| s > 0.0 && s <= 1.0 + 1e-10),
            "bond {}: a Schmidt value of a unit-norm state is in (0, 1]",
            b + 1
        );
        let mass: f64 = block.iter().map(|s| s * s).sum();
        assert!(
            (mass - 1.0).abs() < 1e-10,
            "bond {}: Σσ² = {mass}, must be 1 for a unit-norm state",
            b + 1
        );
        off += c;
    }

    assert_eq!(
        unsafe {
            qntt_f64_vec_schmidt(
                x,
                vals.as_mut_ptr(),
                1,
                &mut len,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
                &mut nbonds,
            )
        },
        QNTT_INVALID_ARG
    );

    unsafe { qntt_f64_vec_free(x) };
}

#[test]
fn grad_norm_vanishes_at_the_ground_state_and_not_before() {
    let (h, x) = h4_ham_and_ground_state(4);

    let (norb, nelec, ms2, _t, _v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
    let flat = flat_ranks(&space, &uniform_ranks(space, 4));
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
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

    let (mut g0, mut g1) = (0.0f64, 0.0f64);
    assert_eq!(
        unsafe { qntt_f64_grad_norm(h, x0, &mut g0) },
        QNTT_OK,
        "grad_norm failed: {}",
        last_error()
    );
    assert_eq!(unsafe { qntt_f64_grad_norm(h, x, &mut g1) }, QNTT_OK);
    assert!(g0.is_finite() && g0 > 0.0, "random start gradient {g0}");
    assert!(g1.is_finite() && g1 >= 0.0, "converged gradient {g1}");
    assert!(
        g1 < g0,
        "the converged state must be more critical than the random start: {g1} vs {g0}"
    );
    assert!(g1 < 1e-6, "converged gradient {g1} is not near-critical");

    let mut zero: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_zeros(space.d, space.nup, space.ndn, &mut zero) },
        QNTT_OK
    );
    let mut gz = 0.0f64;
    assert_eq!(
        unsafe { qntt_f64_grad_norm(h, zero, &mut gz) },
        QNTT_INVALID_ARG
    );

    for hv in [x0, x, zero] {
        unsafe { qntt_f64_vec_free(hv) };
    }
    unsafe { qntt_f64_ham_free(h) };
}

#[test]
fn su2_schmidt_sum_rule_holds_on_singlet_and_non_singlet() {
    for (n, two_s) in [(6usize, 0usize), (6, 2), (6, 4)] {
        let (d, r) = (6usize, 4usize);
        let flat = su2_capped_ranks(d, n, two_s, r);
        let mut raw: *mut _ = ptr::null_mut();
        assert_eq!(
            unsafe { qntt_su2_f64_vec_randn(d, n, two_s, flat.as_ptr(), flat.len(), 7, &mut raw) },
            QNTT_OK,
            "randn failed at 2S = {two_s}: {}",
            last_error()
        );
        let mut nrm = 0.0f64;
        assert_eq!(unsafe { qntt_su2_f64_vec_norm(raw, &mut nrm) }, QNTT_OK);
        assert!(nrm > 0.0);
        let inv = 1.0 / nrm;
        let mut x: *mut _ = ptr::null_mut();
        assert_eq!(
            unsafe { qntt_su2_f64_vec_scaled(raw, &inv, &mut x) },
            QNTT_OK
        );
        unsafe { qntt_su2_f64_vec_free(raw) };

        let (mut len, mut nbonds) = (0usize, 0usize);
        assert_eq!(
            unsafe {
                qntt_su2_f64_vec_schmidt(
                    x,
                    ptr::null_mut(),
                    0,
                    &mut len,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                    0,
                    &mut nbonds,
                )
            },
            QNTT_OK
        );
        assert_eq!(nbonds, d - 1);
        let mut vals = vec![0.0f64; len];
        let mut wts = vec![0.0f64; len];
        let mut counts = vec![0usize; nbonds];
        assert_eq!(
            unsafe {
                qntt_su2_f64_vec_schmidt(
                    x,
                    vals.as_mut_ptr(),
                    vals.len(),
                    &mut len,
                    wts.as_mut_ptr(),
                    wts.len(),
                    counts.as_mut_ptr(),
                    counts.len(),
                    &mut nbonds,
                )
            },
            QNTT_OK
        );

        let mut off = 0usize;
        let mut saw_non_unit_weight = false;
        for (b, &c) in counts.iter().enumerate() {
            let sig = &vals[off..off + c];
            let w = &wts[off..off + c];
            assert!(!sig.is_empty(), "bond {} carries no Schmidt values", b + 1);

            let weighted: f64 = sig.iter().zip(w).map(|(&s, &w)| w * s * s).sum();
            assert!(
                (weighted - 1.0).abs() < 1e-12,
                "2S = {two_s}, bond {}: Σ w σ² = {weighted}, must be 1",
                b + 1
            );

            let degeneracy_only: f64 = sig
                .iter()
                .zip(w)
                .map(|(&s, &w)| w * (two_s + 1) as f64 * s * s)
                .sum();
            assert!(
                (degeneracy_only - (two_s + 1) as f64).abs() < 1e-12,
                "2S = {two_s}, bond {}: the un-normalized degeneracy sum is {degeneracy_only}, \
                 must be 2S+1 = {}",
                b + 1,
                two_s + 1
            );

            if w.iter().any(|&w| (w - 1.0).abs() > 1e-12) {
                saw_non_unit_weight = true;
            }
            off += c;
        }
        assert!(
            saw_non_unit_weight,
            "2S = {two_s}: every weight was 1, so this fixture does not exercise the weighting \
             at all"
        );

        let mut off = 0usize;
        for &c in &counts {
            let plain: f64 = vals[off..off + c].iter().map(|s| s * s).sum();
            if two_s == 0 {
                assert!(
                    plain <= 1.0 + 1e-12,
                    "singlet: unweighted Σσ² = {plain} > 1"
                );
            } else {
                assert!(
                    plain > 1.0,
                    "2S = {two_s}: unweighted Σσ² = {plain}, expected > 1"
                );
            }
            off += c;
        }

        unsafe { qntt_su2_f64_vec_free(x) };
    }
}
