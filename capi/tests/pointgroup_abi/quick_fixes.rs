// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ptr;

use qntt::fcidump::TwoBodySymmetry;
use qntt::{CoeffTensor, Irrep, PointGroup, QnSpace, rayleigh_quotient};
use qntt_capi::err::{QNTT_INVALID_ARG, QNTT_OK};
use qntt_capi::exports::*;
use qntt_capi::fcidump::*;
use qntt_capi::pointgroup::*;

use crate::util::{c2v_orbsym, c2v_pg};
use crate::{PG_AUTO, PG_C2V};

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
fn bond_sectors_pg_allows_partial_null_arrays() {
    let d = 4usize;
    let orbsym = c2v_orbsym(d);

    let mut len = 0usize;
    assert_eq!(
        unsafe {
            qntt_space_bond_sectors_pg(
                d,
                2,
                2,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                0,
                2,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                &mut len,
            )
        },
        QNTT_OK
    );
    assert!(
        len > 0,
        "the C2v H4-shaped space must have sectors at bond 2"
    );

    let mut a = vec![0usize; len];
    let mut b = vec![0usize; len];
    let mut len2 = 0usize;
    let status = unsafe {
        qntt_space_bond_sectors_pg(
            d,
            2,
            2,
            PG_C2V,
            orbsym.as_ptr(),
            orbsym.len(),
            0,
            2,
            a.as_mut_ptr(),
            b.as_mut_ptr(),
            ptr::null_mut(),
            len,
            &mut len2,
        )
    };
    assert_eq!(
        status, QNTT_OK,
        "a null irrep array with the others sized must succeed"
    );
    assert_eq!(len2, len);
    assert!(
        a.iter().any(|&x| x > 0),
        "sector data must actually have been copied into `a`"
    );

    let mut irreps = vec![0u8; len];
    let mut len3 = 0usize;
    assert_eq!(
        unsafe {
            qntt_space_bond_sectors_pg(
                d,
                2,
                2,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                0,
                2,
                ptr::null_mut(),
                ptr::null_mut(),
                irreps.as_mut_ptr(),
                len,
                &mut len3,
            )
        },
        QNTT_OK
    );
    assert_eq!(len3, len);
}

#[test]
fn ham_new_pg_matches_rust_hamiltonian_energy() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
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
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
    let h_rust = qntt::SparseHamiltonian::new(space, &t_tensor, &v_tensor, None, f64::EPSILON);
    let x_rust = qntt::tt_ones::<f64>(space);
    let rq_rust = rayleigh_quotient(&h_rust, &x_rust);
    assert!(
        rq_rust.is_finite() && rq_rust != 0.0,
        "fixture must be nonvacuous"
    );

    let mut h_ptr: *mut _ = ptr::null_mut();
    let status = unsafe {
        qntt_f64_ham_new_pg(
            d,
            nup,
            ndn,
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
    };
    assert_eq!(status, QNTT_OK);

    let mut x_ptr: *mut _ = ptr::null_mut();
    let status = unsafe {
        qntt_f64_tt_ones_pg(
            d,
            nup,
            ndn,
            PG_C2V,
            orbsym.as_ptr(),
            orbsym.len(),
            i32::from(target.index()),
            &mut x_ptr,
        )
    };
    assert_eq!(status, QNTT_OK);

    let mut rq_abi = 0.0f64;
    assert_eq!(
        unsafe { qntt_f64_ham_expectation(h_ptr, x_ptr, &mut rq_abi) },
        QNTT_OK
    );

    assert!(
        (rq_abi - rq_rust).abs() <= 1e-12 * rq_rust.abs().max(1.0),
        "ham_new_pg through the ABI must match the Rust-built Hamiltonian: abi={rq_abi}, \
         rust={rq_rust}"
    );

    unsafe {
        qntt_f64_ham_free(h_ptr);
        qntt_f64_vec_free(x_ptr);
    }
}

#[test]
fn vec_block_get_set_pg_round_trip_at_c2v() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
    let orbsym = c2v_orbsym(d);

    let mut src: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_tt_ones_pg(
                d,
                nup,
                ndn,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                &mut src,
            )
        },
        QNTT_OK
    );
    let mut ranks_len = 0usize;
    assert_eq!(
        unsafe { qntt_f64_vec_ranks_pg(src, ptr::null_mut(), 0, &mut ranks_len) },
        QNTT_OK
    );
    let mut ranks = vec![0usize; ranks_len];
    assert_eq!(
        unsafe { qntt_f64_vec_ranks_pg(src, ranks.as_mut_ptr(), ranks.len(), &mut ranks_len) },
        QNTT_OK
    );
    let mut dst: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_from_ranks_pg(
                d,
                nup,
                ndn,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                ranks.as_ptr(),
                ranks.len(),
                &mut dst,
            )
        },
        QNTT_OK
    );

    let pg = c2v_pg(d);
    let nirrep = pg.order();
    let mut copied = 0usize;
    for site in 0..d {
        for family in 0..4i32 {
            for sa in 0..=nup {
                for sb in 0..=ndn {
                    for irrep in 0..nirrep {
                        let (mut rows, mut cols) = (0usize, 0usize);
                        assert_eq!(
                            unsafe {
                                qntt_f64_vec_block_get_pg(
                                    src,
                                    site,
                                    family,
                                    sa,
                                    sb,
                                    irrep,
                                    ptr::null_mut(),
                                    0,
                                    &mut rows,
                                    &mut cols,
                                )
                            },
                            QNTT_OK,
                            "site={site} family={family} sector=({sa},{sb},{irrep})"
                        );
                        if rows * cols == 0 {
                            continue;
                        }
                        let mut buf = vec![0.0f64; rows * cols];
                        assert_eq!(
                            unsafe {
                                qntt_f64_vec_block_get_pg(
                                    src,
                                    site,
                                    family,
                                    sa,
                                    sb,
                                    irrep,
                                    buf.as_mut_ptr(),
                                    buf.len(),
                                    &mut rows,
                                    &mut cols,
                                )
                            },
                            QNTT_OK
                        );
                        assert_eq!(
                            unsafe {
                                qntt_f64_vec_block_set_pg(
                                    dst,
                                    site,
                                    family,
                                    sa,
                                    sb,
                                    irrep,
                                    buf.as_ptr(),
                                    rows,
                                    cols,
                                )
                            },
                            QNTT_OK
                        );
                        copied += 1;
                    }
                }
            }
        }
    }
    assert!(
        copied > 0,
        "the C2v tt_ones fixture must have at least one nonempty block"
    );

    let mut diff: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_add(src, dst, -1, &mut diff) },
        QNTT_OK
    );
    let (mut nd, mut ns) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_f64_vec_norm(diff, &mut nd) }, QNTT_OK);
    assert_eq!(unsafe { qntt_f64_vec_norm(src, &mut ns) }, QNTT_OK);
    assert!(
        nd <= 1e-14 * ns.max(1.0),
        "block round trip through the _pg accessors changed the train: ‖Δ‖={nd:e}, ‖x‖={ns:e}"
    );

    for h in [src, dst, diff] {
        unsafe { qntt_f64_vec_free(h) };
    }
}

#[test]
fn orbsym_to_irreps_matches_the_scalar_conversion() {
    let d = 6usize;
    let orbsym = c2v_orbsym(d);

    let mut len = 0usize;
    assert_eq!(
        unsafe {
            qntt_pg_orbsym_to_irreps(
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                ptr::null_mut(),
                0,
                &mut len,
            )
        },
        QNTT_OK
    );
    assert_eq!(len, d);

    let mut irreps = vec![0u8; d];
    assert_eq!(
        unsafe {
            qntt_pg_orbsym_to_irreps(
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                irreps.as_mut_ptr(),
                irreps.len(),
                &mut len,
            )
        },
        QNTT_OK
    );

    for (k, &id) in orbsym.iter().enumerate() {
        let mut ir = 0i32;
        assert_eq!(
            unsafe { qntt_pg_irrep_from_molpro(PG_C2V, i32::from(id), &mut ir) },
            QNTT_OK
        );
        assert_eq!(
            irreps[k], ir as u8,
            "bulk conversion must agree with the scalar conversion at site {k}"
        );
    }
}

#[test]
fn fcidump_read_pg_explicit_kind_is_remembered_and_reported_back() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/h4.FCIDUMP");
    let path_c = std::ffi::CString::new(path.to_str().unwrap()).unwrap();

    let mut h_auto: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read_pg(path_c.as_ptr(), 2, PG_AUTO, &mut h_auto) },
        QNTT_OK
    );
    let (mut kind_auto, mut isym_auto, mut target_auto) = (0i32, 0i64, 0i32);
    assert_eq!(
        unsafe {
            qntt_f64_fcidump_point_group(h_auto, &mut kind_auto, &mut isym_auto, &mut target_auto)
        },
        QNTT_OK
    );
    assert_eq!(kind_auto, 0, "C1 wire code");

    const PG_D2: i32 = 4;
    let mut h_explicit: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read_pg(path_c.as_ptr(), 2, PG_D2, &mut h_explicit) },
        QNTT_OK,
        "an all-ones ORBSYM is valid under ANY explicit kind"
    );
    let (mut kind_ex, mut isym_ex, mut target_ex) = (0i32, 0i64, 0i32);
    assert_eq!(
        unsafe {
            qntt_f64_fcidump_point_group(h_explicit, &mut kind_ex, &mut isym_ex, &mut target_ex)
        },
        QNTT_OK
    );
    assert_eq!(
        kind_ex, PG_D2,
        "the explicit kind must be remembered, not re-guessed"
    );
    assert_eq!(
        isym_ex, isym_auto,
        "ISYM itself is read the same regardless of kind"
    );

    let mut t_auto = vec![0.0f64; 0];
    let mut len_auto = 0usize;
    unsafe { qntt_f64_fcidump_t(h_auto, ptr::null_mut(), 0, &mut len_auto) };
    t_auto.resize(len_auto, 0.0);
    assert_eq!(
        unsafe { qntt_f64_fcidump_t(h_auto, t_auto.as_mut_ptr(), t_auto.len(), &mut len_auto) },
        QNTT_OK
    );
    let mut t_explicit = vec![0.0f64; len_auto];
    let mut len_ex = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_fcidump_t(
                h_explicit,
                t_explicit.as_mut_ptr(),
                t_explicit.len(),
                &mut len_ex,
            )
        },
        QNTT_OK
    );
    assert_eq!(t_auto, t_explicit);

    let _ = QNTT_INVALID_ARG;

    unsafe {
        qntt_f64_fcidump_free(h_auto);
        qntt_f64_fcidump_free(h_explicit);
    }
}

#[test]
fn fcidump_read_pg_auto_matches_plain_read_on_c1_fixture() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/h4.FCIDUMP");
    let (data, _pg) = qntt::fcidump::read_fcidump_with_point_group::<f64>(
        &path,
        TwoBodySymmetry::Eightfold,
        None,
    )
    .expect("read via Rust API");

    let path_c = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read_pg(path_c.as_ptr(), 2, PG_AUTO, &mut h) },
        QNTT_OK
    );
    let mut norb = 0usize;
    let (mut nelec, mut ms2, mut e0) = (0usize, 0i64, 0.0f64);
    assert_eq!(
        unsafe { qntt_f64_fcidump_info(h, &mut norb, &mut nelec, &mut ms2, &mut e0) },
        QNTT_OK
    );
    assert_eq!(norb, data.norb);
    assert_eq!(nelec, data.nelec);
    assert_eq!(ms2, data.ms2);
    assert_eq!(e0, data.e0);

    unsafe { qntt_f64_fcidump_free(h) };
}
