// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! C ABI for `qntt` — the surface the package in `../julia` calls through.

#![allow(rustdoc::private_intra_doc_links)]

pub mod api;
pub mod enums;
pub mod err;
pub mod exports;
pub mod fcidump;
pub mod gad;
pub mod handle;
pub mod joint;
pub mod layer;
pub mod opts;
pub mod pointgroup;
pub mod probe;
pub mod rpa;
pub mod su2_pointgroup;
pub mod tda;
pub mod types;

use std::os::raw::{c_char, c_int};

use err::{QNTT_OK, catch, with_last_error};

/// Crate version as a NUL-terminated string (`"0.1.0"`).
///
/// # Safety
/// `buf` must be writable for `cap` bytes, or null when `cap == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_version(buf: *mut c_char, cap: usize, out_len: *mut usize) -> c_int {
    catch(|| {
        let v = env!("CARGO_PKG_VERSION");
        unsafe { write_cstr(v, buf, cap, out_len) }
    })
}

/// Message describing the calling thread's most recent failed call (empty if none).
///
/// # Safety
/// `buf` must be writable for `cap` bytes, or null when `cap == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_last_error(
    buf: *mut c_char,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    let msg = with_last_error(|s| s.to_string());
    match unsafe { write_cstr(&msg, buf, cap, out_len) } {
        Ok(()) => QNTT_OK,
        Err(e) => e.code,
    }
}

/// Size the global rayon pool (`n == 0` lets rayon choose). Only the first call in a process has
/// an effect — a second call is reported as `QNTT_OK` because the pool is already sized, which is
/// the behaviour a driver wants when it re-runs a script in one session.
#[unsafe(no_mangle)]
pub extern "C" fn qntt_init(nthreads: usize) -> c_int {
    catch(|| {
        let mut b = rayon::ThreadPoolBuilder::new();
        if nthreads > 0 {
            b = b.num_threads(nthreads);
        }
        let _ = b.build_global();
        Ok(())
    })
}

/// Number of threads in the global rayon pool.
///
/// # Safety
/// `out` must be a valid writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_num_threads(out: *mut usize) -> c_int {
    catch(|| {
        let out = unsafe { err::ptr_mut(out, "out")? };
        *out = rayon::current_num_threads();
        Ok(())
    })
}

pub(crate) unsafe fn write_cstr(
    s: &str,
    buf: *mut c_char,
    cap: usize,
    out_len: *mut usize,
) -> Result<(), err::Error> {
    if !out_len.is_null() {
        unsafe { *out_len = s.len() };
    }
    if cap == 0 {
        return Ok(());
    }
    if buf.is_null() {
        return Err(err::Error::null("buf"));
    }
    if cap < s.len() + 1 {
        return Err(err::Error::invalid(format!(
            "string buffer holds {cap} bytes, need {}",
            s.len() + 1
        )));
    }
    let dst = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, s.len() + 1) };
    dst[..s.len()].copy_from_slice(s.as_bytes());
    dst[s.len()] = 0;
    Ok(())
}
