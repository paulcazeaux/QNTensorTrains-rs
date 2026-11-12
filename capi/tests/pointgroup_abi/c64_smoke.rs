// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ptr;

use qntt::{C64, Irrep};
use qntt_capi::err::QNTT_OK;
use qntt_capi::exports::*;

use crate::PG_C2V;
use crate::util::{c2v_coeffs, c2v_orbsym, c2v_pg, last_error};

#[test]
fn c64_pg_smoke_matches_the_f64_path() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
    let pg = c2v_pg(d);
    let orbsym = c2v_orbsym(d);
    let (t, v) = c2v_coeffs(&pg, d);
    let tc: Vec<C64> = t.data.iter().map(|&x| C64::new(x, 0.0)).collect();
    let vc: Vec<C64> = v.data.iter().map(|&x| C64::new(x, 0.0)).collect();
    let tgt = i32::from(target.index());

    let mut hc: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_c64_ham_new_pg(
                d,
                nup,
                ndn,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                tgt,
                tc.as_ptr(),
                tc.len(),
                vc.as_ptr(),
                vc.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut hc,
            )
        },
        QNTT_OK,
        "c64 ham_new_pg failed: {}",
        last_error()
    );
    let mut xc: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_c64_tt_ones_pg(
                d,
                nup,
                ndn,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                tgt,
                &mut xc,
            )
        },
        QNTT_OK
    );

    let (mut nbits, mut tgt_back, mut irr_len) = (0i32, 0i32, 0usize);
    let mut irreps = vec![0u8; d];
    assert_eq!(
        unsafe {
            qntt_c64_ham_point_group(
                hc,
                &mut nbits,
                &mut tgt_back,
                irreps.as_mut_ptr(),
                d,
                &mut irr_len,
            )
        },
        QNTT_OK
    );
    assert_eq!((nbits, tgt_back, irr_len), (2, tgt, d));
    let expect: Vec<u8> = (0..d).map(|k| pg.irrep_of(k).index()).collect();
    assert_eq!(irreps, expect);

    let mut ec = C64::new(0.0, 0.0);
    assert_eq!(
        unsafe { qntt_c64_ham_expectation(hc, xc, &mut ec) },
        QNTT_OK
    );

    let mut hr: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new_pg(
                d,
                nup,
                ndn,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                tgt,
                t.data.as_ptr(),
                t.data.len(),
                v.data.as_ptr(),
                v.data.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut hr,
            )
        },
        QNTT_OK
    );
    let mut xr: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_tt_ones_pg(
                d,
                nup,
                ndn,
                PG_C2V,
                orbsym.as_ptr(),
                orbsym.len(),
                tgt,
                &mut xr,
            )
        },
        QNTT_OK
    );
    let mut er = 0.0f64;
    assert_eq!(
        unsafe { qntt_f64_ham_expectation(hr, xr, &mut er) },
        QNTT_OK
    );
    assert!(er != 0.0, "fixture must be nonvacuous");
    assert!(
        (ec.re - er).abs() <= 1e-12 * er.abs().max(1.0) && ec.im.abs() <= 1e-12,
        "c64 _pg path must reproduce the f64 one: {ec} vs {er}"
    );

    let mut nr = 0usize;
    assert_eq!(
        unsafe { qntt_c64_vec_ranks_pg(xc, ptr::null_mut(), 0, &mut nr) },
        QNTT_OK
    );
    assert_eq!(
        nr,
        (d + 1) * usize::from(pg.order()) * (nup + 1) * (ndn + 1)
    );

    unsafe {
        qntt_c64_vec_free(xc);
        qntt_c64_ham_free(hc);
        qntt_f64_vec_free(xr);
        qntt_f64_ham_free(hr);
    }
}
