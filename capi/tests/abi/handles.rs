// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn version_and_last_error_size_query() {
    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_version(ptr::null_mut(), 0, &mut len) },
        QNTT_OK
    );
    assert!(len > 0, "version string should be non-empty");

    let mut buf = vec![0u8; len + 1];
    assert_eq!(
        unsafe { qntt_version(buf.as_mut_ptr() as *mut i8, buf.len(), &mut len) },
        QNTT_OK
    );
    assert_eq!(&buf[..len], env!("CARGO_PKG_VERSION").as_bytes());

    assert_eq!(
        unsafe { qntt_version(buf.as_mut_ptr() as *mut i8, 1, &mut len) },
        QNTT_INVALID_ARG
    );
}

#[test]
fn null_argument_is_reported_not_dereferenced() {
    let mut out: *mut _ = ptr::null_mut();
    let status = unsafe { qntt_f64_vec_zeros(4, 2, 2, &mut out) };
    assert_eq!(status, QNTT_OK);

    let status = unsafe { qntt_f64_vec_zeros(4, 2, 2, ptr::null_mut()) };
    assert_ne!(status, QNTT_OK);
    assert!(
        last_error().contains("null pointer"),
        "got {:?}",
        last_error()
    );

    unsafe { qntt_f64_vec_free(out) };
}

#[test]
fn free_is_null_safe_and_typed() {
    unsafe { qntt_f64_vec_free(ptr::null_mut()) };
    unsafe { qntt_f64_ham_free(ptr::null_mut()) };
    unsafe { qntt_su2_c64_vec_free(ptr::null_mut()) };
}

#[test]
fn wrong_handle_type_is_caught_by_the_tag() {
    let mut v: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_zeros(4, 2, 2, &mut v) }, QNTT_OK);

    let mut norm = 0.0f64;
    let status = unsafe { qntt_c64_vec_norm(v as *const _, &mut norm) };
    assert_eq!(status, QNTT_INVALID_ARG);
    assert!(
        last_error().contains("type mismatch"),
        "got {:?}",
        last_error()
    );

    unsafe { qntt_f64_vec_free(v) };
}

#[test]
fn invalid_arguments_are_rejected_before_any_work() {
    let d = 3usize;
    let t = vec![0.0f64; d * d];
    let v = vec![0.0f64; d * d * d * d];
    let mut h: *mut _ = ptr::null_mut();
    let status = unsafe {
        qntt_f64_ham_new(
            d,
            4,
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
    };
    assert_eq!(status, QNTT_INVALID_ARG);
    assert!(last_error().contains("Nup"), "got {:?}", last_error());

    let status = unsafe {
        qntt_f64_ham_new(
            d,
            1,
            1,
            t.as_ptr(),
            t.len() - 1,
            v.as_ptr(),
            v.len(),
            ptr::null(),
            f64::EPSILON,
            0,
            &mut h,
        )
    };
    assert_eq!(status, QNTT_INVALID_ARG);

    let mut lambda = 0.0f64;
    let mut x: *mut _ = ptr::null_mut();
    let opts = QnttMalsOptions::defaults();
    let status = unsafe {
        qntt_f64_mals(
            ptr::null(),
            ptr::null(),
            &opts,
            ptr::null(),
            0,
            ptr::null(),
            0,
            &mut lambda,
            &mut x,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(status, qntt_capi::err::QNTT_NULL_ARG);
    assert!(
        last_error().contains("null pointer"),
        "got {:?}",
        last_error()
    );
}

#[test]
fn panic_boundary_returns_a_message() {
    let mut x: *mut _ = ptr::null_mut();
    let up: Vec<i32> = vec![];
    let dn: Vec<i32> = vec![];
    let status = unsafe { qntt_f64_tt_state(up.as_ptr(), dn.as_ptr(), 0, &mut x) };
    assert_eq!(status, QNTT_PANIC, "expected a caught panic, got {status}");
    assert!(last_error().starts_with("panic:"), "got {:?}", last_error());
}
