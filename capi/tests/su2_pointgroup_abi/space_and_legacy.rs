// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::*;

fn last_error() -> String {
    let mut len = 0usize;
    unsafe { qntt_last_error(ptr::null_mut(), 0, &mut len) };
    let mut buf = vec![0u8; len + 1];
    unsafe {
        qntt_last_error(
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            buf.len(),
            &mut len,
        )
    };
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

#[test]
fn a1_space_check_pg_rejects_every_bad_descriptor_never_panics() {
    let d = 6usize;
    let orbsym = c2v_orbsym(d);

    assert_eq!(
        unsafe { qntt_su2_space_check_pg(d, 6, 0, PG_C2V, orbsym.as_ptr(), orbsym.len(), 0) },
        QNTT_OK,
        "the baseline descriptor must be accepted"
    );

    assert_eq!(
        unsafe { qntt_su2_space_check_pg(d, 6, 0, 99, orbsym.as_ptr(), orbsym.len(), 0) },
        QNTT_INVALID_ARG
    );

    assert_eq!(
        unsafe { qntt_su2_space_check_pg(d, 6, 0, PG_C2V, orbsym.as_ptr(), orbsym.len() - 1, 0) },
        QNTT_INVALID_ARG
    );

    let mut bad_orbsym = orbsym.clone();
    bad_orbsym[0] = 9;
    assert_eq!(
        unsafe {
            qntt_su2_space_check_pg(d, 6, 0, PG_C2V, bad_orbsym.as_ptr(), bad_orbsym.len(), 0)
        },
        QNTT_INVALID_ARG
    );

    assert_eq!(
        unsafe { qntt_su2_space_check_pg(d, 6, 0, PG_C2V, orbsym.as_ptr(), orbsym.len(), 4) },
        QNTT_INVALID_ARG
    );

    let orbsym128: Vec<u8> = (0..128).map(|k| [2u8, 3, 4, 1][k % 4]).collect();
    assert_eq!(
        unsafe {
            qntt_su2_space_check_pg(128, 6, 0, PG_C2V, orbsym128.as_ptr(), orbsym128.len(), 0)
        },
        QNTT_INVALID_ARG
    );

    let orbsym128_c1 = [1u8; 128];
    assert_eq!(
        unsafe {
            qntt_su2_space_check_pg(
                128,
                6,
                0,
                PG_C1,
                orbsym128_c1.as_ptr(),
                orbsym128_c1.len(),
                0,
            )
        },
        QNTT_OK
    );

    let _ = QNTT_PANIC;
}

#[test]
fn a2_max_ranks_pg_round_trips_and_the_flat_stride_locates_the_right_scalar() {
    let d = 4usize;
    let (n, two_s) = (4usize, 0usize);
    let target = Irrep::new(0);
    let orbsym = c2v_orbsym(d);
    let pg = c2v_pg(d);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let caps_rust = su2_max_ranks(&space);

    let mut len = 0usize;
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
                &mut len,
            )
        },
        QNTT_OK
    );
    let rows = n + 1;
    let cols = n + 1;
    let nir = pg.order() as usize;
    assert_eq!(len, (d + 1) * nir * rows * cols);

    let mut flat = vec![0usize; len];
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
                flat.as_mut_ptr(),
                flat.len(),
                &mut len,
            )
        },
        QNTT_OK
    );

    let mid = d / 2;
    let mut probe = None;
    'outer: for ir in 1..nir {
        for nn in 1..=n {
            for ss in 0..=n {
                let sector = Su2Sector::with_irrep(nn, ss, Irrep::new(ir as u8));
                let cap = caps_rust[mid][sector];
                if cap > 0 {
                    probe = Some((ir, nn, ss, cap));
                    break 'outer;
                }
            }
        }
    }
    let (ir, nn, ss, expect_val) = probe.expect(
        "fixture bug: the middle bond must have a nonzero-cap sector with ir > 0 && n > 0 \
         (needed to pin the irrep axis's position in the flat layout)",
    );
    let idx = mid * (nir * rows * cols) + ir * (rows * cols) + nn * cols + ss;
    assert_eq!(
        flat[idx], expect_val,
        "flat-stride mismatch at bond={mid} irrep={ir} n={nn} 2s={ss}"
    );

    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_from_ranks_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                i32::from(target.index()),
                flat.as_ptr(),
                flat.len(),
                &mut x,
            )
        },
        QNTT_OK
    );
    let mut out = vec![0usize; len];
    let mut out_len = 0usize;
    assert_eq!(
        unsafe { qntt_su2_f64_vec_ranks_pg(x, out.as_mut_ptr(), out.len(), &mut out_len) },
        QNTT_OK
    );
    assert_eq!(out_len, len);
    assert_eq!(
        out, flat,
        "max_ranks_pg -> vec_from_ranks_pg -> vec_ranks_pg must be the identity"
    );

    unsafe { qntt_su2_f64_vec_free(x) };
}

#[test]
fn a3_rule_l_is_live_on_su2_nontrivial_point_group() {
    let d = 4usize;
    let (n, two_s) = (2usize, 0usize);
    let orbsym = c2v_orbsym(d);

    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_zeros_pg(
                d,
                n,
                two_s,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                0,
                &mut x,
            )
        },
        QNTT_OK
    );

    let mut ranks_len = 0usize;
    assert_eq!(
        unsafe { qntt_su2_f64_vec_ranks(x, ptr::null_mut(), 0, &mut ranks_len) },
        QNTT_INVALID_ARG
    );
    assert!(
        last_error().contains("vec_ranks_pg"),
        "error must name the _pg twin: {}",
        last_error()
    );

    let (mut rows, mut cols) = (0usize, 0usize);
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_block_get(x, 0, 0, 0, 0, ptr::null_mut(), 0, &mut rows, &mut cols)
        },
        QNTT_INVALID_ARG
    );
    assert!(
        last_error().contains("vec_block_get_pg"),
        "error must name the _pg twin: {}",
        last_error()
    );

    let buf = [0.0f64; 1];
    assert_eq!(
        unsafe { qntt_su2_f64_vec_block_set(x, 0, 0, 0, 0, buf.as_ptr(), 1, 1) },
        QNTT_INVALID_ARG
    );
    assert!(
        last_error().contains("vec_block_set_pg"),
        "error must name the _pg twin: {}",
        last_error()
    );

    unsafe { qntt_su2_f64_vec_free(x) };
}

#[test]
fn a4_c1_reduction_matches_legacy_element_wise() {
    let d = 4usize;
    let (n, two_s) = (3usize, 1usize);
    let orbsym = vec![1u8; d];

    let mut legacy_len = 0usize;
    assert_eq!(
        unsafe { qntt_su2_space_max_ranks(d, n, two_s, ptr::null_mut(), 0, &mut legacy_len) },
        QNTT_OK
    );
    let mut legacy_ranks = vec![0usize; legacy_len];
    assert_eq!(
        unsafe {
            qntt_su2_space_max_ranks(
                d,
                n,
                two_s,
                legacy_ranks.as_mut_ptr(),
                legacy_ranks.len(),
                &mut legacy_len,
            )
        },
        QNTT_OK
    );

    let mut pg_len = 0usize;
    assert_eq!(
        unsafe {
            qntt_su2_space_max_ranks_pg(
                d,
                n,
                two_s,
                PG_C1,
                orbsym.as_ptr(),
                orbsym.len(),
                0,
                ptr::null_mut(),
                0,
                &mut pg_len,
            )
        },
        QNTT_OK
    );
    assert_eq!(
        pg_len, legacy_len,
        "nirrep=1 must collapse to the legacy element count"
    );
    let mut pg_ranks = vec![0usize; pg_len];
    assert_eq!(
        unsafe {
            qntt_su2_space_max_ranks_pg(
                d,
                n,
                two_s,
                PG_C1,
                orbsym.as_ptr(),
                orbsym.len(),
                0,
                pg_ranks.as_mut_ptr(),
                pg_ranks.len(),
                &mut pg_len,
            )
        },
        QNTT_OK
    );
    assert_eq!(
        pg_ranks, legacy_ranks,
        "at C1 the _pg max-ranks flat layout must equal the legacy one term-for-term"
    );

    let (mut rows, mut cols, mut nirrep) = (0usize, 0usize, 0usize);
    assert_eq!(
        unsafe {
            qntt_su2_space_grid_extent_pg(
                d,
                n,
                two_s,
                PG_C1,
                orbsym.as_ptr(),
                orbsym.len(),
                0,
                &mut rows,
                &mut cols,
                &mut nirrep,
            )
        },
        QNTT_OK
    );
    assert_eq!(nirrep, 1);

    let mut x_pg: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_from_ranks_pg(
                d,
                n,
                two_s,
                PG_C1,
                orbsym.as_ptr(),
                orbsym.len(),
                0,
                pg_ranks.as_ptr(),
                pg_ranks.len(),
                &mut x_pg,
            )
        },
        QNTT_OK
    );
    let mut x_legacy: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_from_ranks(
                d,
                n,
                two_s,
                legacy_ranks.as_ptr(),
                legacy_ranks.len(),
                &mut x_legacy,
            )
        },
        QNTT_OK
    );

    let mut out_pg = vec![0usize; pg_len];
    let mut len2 = 0usize;
    assert_eq!(
        unsafe { qntt_su2_f64_vec_ranks_pg(x_pg, out_pg.as_mut_ptr(), out_pg.len(), &mut len2) },
        QNTT_OK
    );
    let mut out_legacy = vec![0usize; legacy_len];
    assert_eq!(
        unsafe {
            qntt_su2_f64_vec_ranks(
                x_legacy,
                out_legacy.as_mut_ptr(),
                out_legacy.len(),
                &mut len2,
            )
        },
        QNTT_OK
    );
    assert_eq!(out_pg, out_legacy);

    unsafe {
        qntt_su2_f64_vec_free(x_pg);
        qntt_su2_f64_vec_free(x_legacy);
    }
}
