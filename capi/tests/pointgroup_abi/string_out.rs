// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ptr;

use qntt_capi::err::{QNTT_INVALID_ARG, QNTT_NULL_ARG, QNTT_OK};
use qntt_capi::pointgroup::*;

use crate::PG_C2V;
use crate::util::last_error;

fn assert_string_out_protocol(
    expect: &str,
    call: impl Fn(*mut std::os::raw::c_char, usize, *mut usize) -> i32,
) {
    let mut len = 0usize;
    assert_eq!(call(ptr::null_mut(), 0, &mut len), QNTT_OK);
    assert_eq!(len, expect.len(), "size query reported the wrong length");

    let mut buf = vec![0xAAu8; len + 2];
    assert_eq!(
        call(
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            len,
            ptr::null_mut()
        ),
        QNTT_OK,
        "cap == len must succeed (documented): {}",
        last_error()
    );
    assert_eq!(&buf[..len], expect.as_bytes());
    assert_eq!(buf[len], 0xAA, "cap == len must NOT write a NUL");

    let mut buf = vec![0xAAu8; len + 2];
    let mut len2 = 0usize;
    assert_eq!(
        call(
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            len + 1,
            &mut len2
        ),
        QNTT_OK
    );
    assert_eq!(len2, len);
    assert_eq!(&buf[..len], expect.as_bytes());
    assert_eq!(buf[len], 0, "cap == len + 1 must NUL-terminate");
    assert_eq!(buf[len + 1], 0xAA, "nothing past the NUL may be written");

    let mut buf = vec![0xAAu8; len + 2];
    let mut len3 = 0usize;
    assert_eq!(
        call(
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            len - 1,
            &mut len3
        ),
        QNTT_INVALID_ARG
    );
    assert_eq!(len3, len, "the size query must work even on the error path");
    assert!(
        last_error().contains(&format!("need {len}")),
        "the too-small-cap message must state the requirement: {}",
        last_error()
    );

    assert_eq!(call(ptr::null_mut(), len, ptr::null_mut()), QNTT_NULL_ARG);
}

#[test]
fn pg_kind_name_string_out_protocol() {
    assert_string_out_protocol("C2v", |buf, cap, out_len| unsafe {
        qntt_pg_kind_name(PG_C2V, buf, cap, out_len)
    });
    assert_string_out_protocol("C1", |buf, cap, out_len| unsafe {
        qntt_pg_kind_name(0, buf, cap, out_len)
    });
}

#[test]
fn pg_irrep_name_string_out_protocol() {
    assert_string_out_protocol("B1", |buf, cap, out_len| unsafe {
        qntt_pg_irrep_name(PG_C2V, 2, buf, cap, out_len)
    });
    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_pg_irrep_name(PG_C2V, 4, ptr::null_mut(), 0, &mut len) },
        QNTT_INVALID_ARG
    );
    assert!(last_error().contains("out of range"), "{}", last_error());
}
