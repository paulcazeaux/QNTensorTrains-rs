// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ptr;

use qntt_capi::err::{QNTT_INVALID_ARG, QNTT_OK, QNTT_PANIC};
use qntt_capi::exports::*;
use qntt_capi::pointgroup::*;

use crate::util::{check_pg, last_error};

#[test]
fn pg_space_caps_d_at_128_even_for_c1() {
    let ok = vec![1u8; 128];
    assert_eq!(
        check_pg(128, 0, 0, 0, &ok, 0),
        QNTT_OK,
        "d = 128 at C1 must still resolve: {}",
        last_error()
    );

    let over = vec![1u8; 129];
    let status = check_pg(129, 0, 0, 0, &over, 0);
    assert_ne!(
        status, QNTT_PANIC,
        "a C1 descriptor past the cap must not surface as a caught panic"
    );
    assert_eq!(status, QNTT_INVALID_ARG);
    let msg = last_error();
    assert!(
        msg.contains("129") && msg.contains("128-site") && msg.contains("C1"),
        "the message must name the offending d, the cap and the kind: {msg}"
    );
    assert!(
        msg.contains("non-_pg"),
        "the message must point at the any-d escape hatch: {msg}"
    );

    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_zeros_pg(129, 0, 0, 0, over.as_ptr(), over.len(), 0, &mut x) },
        QNTT_INVALID_ARG
    );
    assert!(x.is_null(), "no handle may be emitted on the error path");
    let mut nrows = (0usize, 0usize, 0usize);
    assert_eq!(
        unsafe {
            qntt_space_grid_extent_pg(
                129,
                0,
                0,
                0,
                over.as_ptr(),
                over.len(),
                0,
                &mut nrows.0,
                &mut nrows.1,
                &mut nrows.2,
            )
        },
        QNTT_INVALID_ARG
    );
}
