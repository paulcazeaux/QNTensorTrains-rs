// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::*;

fn symmetric_flat_coeffs(pg: &PointGroup, d: usize) -> (Vec<f64>, Vec<f64>) {
    let mut t = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of(i) == pg.irrep_of(j) {
                t[i * d + j] = 1.0 + (i * d + j) as f64 * 0.013;
            }
        }
    }
    let mut v = vec![0.0f64; d * d * d * d];
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let ir = pg
                        .irrep_of(i)
                        .fuse(pg.irrep_of(j))
                        .fuse(pg.irrep_of(k))
                        .fuse(pg.irrep_of(l));
                    if ir == Irrep::IDENTITY {
                        let idx = ((i * d + j) * d + k) * d + l;
                        v[idx] = 0.011 * (1.0 + idx as f64 * 0.001);
                    }
                }
            }
        }
    }
    (t, v)
}

#[test]
fn a5_tt_csf_pg_derives_and_validates_the_target() {
    let d = 2usize;
    let orbsym: Vec<u8> = vec![2, 3];
    let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table");
    assert_ne!(pg.irrep_of(0), pg.irrep_of(1), "fixture bug");

    let codes = [1i32, 1i32];
    let derived = pg.irrep_of(0).fuse(pg.irrep_of(1));
    assert_ne!(
        derived,
        Irrep::IDENTITY,
        "fixture bug: derived target must be nonidentity"
    );

    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_tt_csf_pg(
                codes.as_ptr(),
                d,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(derived.index()),
                &mut x,
            )
        },
        QNTT_OK,
        "the derived target must be ACCEPTED when fed back in"
    );

    let (mut nbits, mut tgt, mut irr_len) = (0i32, 0i32, 0usize);
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_point_group(x, &mut nbits, &mut tgt, ptr::null_mut(), 0, &mut irr_len)
        },
        QNTT_OK
    );
    assert_eq!(irr_len, d);
    assert_eq!(tgt, i32::from(derived.index()));
    let mut irreps_out = vec![0u8; d];
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_point_group(
                x,
                &mut nbits,
                &mut tgt,
                irreps_out.as_mut_ptr(),
                irreps_out.len(),
                &mut irr_len,
            )
        },
        QNTT_OK
    );
    let mut want_irreps = vec![0u8; d];
    let mut want_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_pg_orbsym_to_irreps(
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                want_irreps.as_mut_ptr(),
                want_irreps.len(),
                &mut want_len,
            )
        },
        QNTT_OK
    );
    assert_eq!(irreps_out, want_irreps);
    unsafe { qntt_su2_f64_vec_free(x) };

    let wrong_target = i32::from(derived.index()) ^ 1;
    let mut x_bad: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_tt_csf_pg(
                codes.as_ptr(),
                d,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                wrong_target,
                &mut x_bad,
            )
        },
        QNTT_INVALID_ARG,
        "a target that disagrees with the derived one must be rejected"
    );

    let dn_codes = [2i32];
    let orbsym1 = [1u8];
    let mut x_dn: *mut _ = ptr::null_mut();
    let status = unsafe {
        qntt_su2_f64_tt_csf_pg(
            dn_codes.as_ptr(),
            1,
            PG_C1,
            orbsym1.as_ptr(),
            orbsym1.len(),
            0,
            &mut x_dn,
        )
    };
    assert_eq!(
        status, QNTT_INVALID_ARG,
        "a Dn step from two_s=0 must be a clean error, not a QNTT_PANIC crossing"
    );
}

#[test]
fn a6_ham_new_pg_matches_rust_and_rejects_nondefault_construction() {
    let d = 4usize;
    let (n, two_s) = (2usize, 0usize);
    let target = Irrep::IDENTITY;
    let pg = c2v_pg(d);
    let orbsym = c2v_orbsym(d);
    let (t_flat, v_flat) = symmetric_flat_coeffs(&pg, d);

    let t_tensor = CoeffTensor::<f64> {
        d,
        rank: 2,
        data: t_flat.clone(),
    };
    let v_tensor = CoeffTensor::<f64> {
        d,
        rank: 4,
        data: v_flat.clone(),
    };
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h_rust = Su2SparseHamiltonian::new(space, &t_tensor, &v_tensor, None, f64::EPSILON);
    let codes = [Family::Occ, Family::Empty, Family::Empty, Family::Empty];
    let x_rust: Su2TtVector<f64> = tt_csf_with_point_group(&codes, pg);
    assert_eq!(x_rust.space.target, target);
    let rq_rust = rayleigh_quotient(&h_rust, &x_rust);
    assert!(
        rq_rust.is_finite() && rq_rust != 0.0,
        "fixture must be nonvacuous"
    );

    let mut h_ptr: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_ham_new_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                t_flat.as_ptr(),
                t_flat.len(),
                v_flat.as_ptr(),
                v_flat.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h_ptr,
            )
        },
        QNTT_OK
    );

    let codes_wire: Vec<i32> = vec![3, 0, 0, 0];
    let mut x_ptr: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_tt_csf_pg(
                codes_wire.as_ptr(),
                d,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                &mut x_ptr,
            )
        },
        QNTT_OK
    );

    let mut rq_abi = 0.0f64;
    assert_eq!(
        unsafe { qntt_su2_f64_ham_expectation(h_ptr, x_ptr, &mut rq_abi) },
        QNTT_OK
    );
    assert!(
        (rq_abi - rq_rust).abs() <= 1e-12 * rq_rust.abs().max(1.0),
        "ham_new_pg through the ABI must match the Rust-built Hamiltonian: abi={rq_abi}, \
         rust={rq_rust}"
    );

    let (mut nbits, mut tgt, mut irr_len) = (0i32, 0i32, 0usize);
    assert_eq!(
        unsafe {
            qntt_su2_f64_ham_point_group(
                h_ptr,
                &mut nbits,
                &mut tgt,
                ptr::null_mut(),
                0,
                &mut irr_len,
            )
        },
        QNTT_OK
    );
    assert_eq!((nbits, tgt, irr_len), (2, i32::from(target.index()), d));

    let mut ranks_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_space_max_ranks_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                ptr::null_mut(),
                0,
                &mut ranks_len,
            )
        },
        QNTT_OK
    );
    let mut ranks = vec![0usize; ranks_len];
    assert_eq!(
        unsafe {
            qntt_su2_space_max_ranks_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                ranks.as_mut_ptr(),
                ranks.len(),
                &mut ranks_len,
            )
        },
        QNTT_OK
    );
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_randn_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                ranks.as_ptr(),
                ranks.len(),
                70_601,
                &mut x0,
            )
        },
        QNTT_OK
    );
    let mut y: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_f64_ham_apply(h_ptr, x0, &mut y) },
        QNTT_OK
    );
    let mut ok = 0i32;
    assert_eq!(unsafe { qntt_su2_f64_vec_check(y, &mut ok) }, QNTT_OK);
    assert_eq!(ok, 1);

    let mut h_bad: *mut _ = ptr::null_mut();
    let status = unsafe {
        qntt_su2_f64_ham_new_pg(
            d,
            n,
            two_s,
            PG_C2V,
            orbsym.as_ptr(),
            orbsym.len(),
            i32::from(target.index()),
            t_flat.as_ptr(),
            t_flat.len(),
            v_flat.as_ptr(),
            v_flat.len(),
            ptr::null(),
            f64::EPSILON,
            2,
            &mut h_bad,
        )
    };
    assert_eq!(status, QNTT_INVALID_ARG);

    unsafe {
        qntt_su2_f64_vec_free(x_ptr);
        qntt_su2_f64_vec_free(x0);
        qntt_su2_f64_vec_free(y);
        qntt_su2_f64_ham_free(h_ptr);
    }
}
