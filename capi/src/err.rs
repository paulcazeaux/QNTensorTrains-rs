// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Status codes, the thread-local last-error slot, and the `catch_unwind` guard every exported
//! function runs its body inside.

use std::cell::RefCell;
use std::panic::AssertUnwindSafe;

/// Success.
pub const QNTT_OK: i32 = 0;
/// A required pointer argument was null.
pub const QNTT_NULL_ARG: i32 = 1;
/// An argument was non-null but invalid (bad enum value, wrong handle type, short buffer, …).
pub const QNTT_INVALID_ARG: i32 = 2;
/// A Rust panic (typically a core-crate `assert!`) was caught at the boundary.
pub const QNTT_PANIC: i32 = 3;
/// Filesystem / parse failure (FCIDUMP readers).
pub const QNTT_IO: i32 = 4;
/// An iterative driver failed to converge or broke down.
pub const QNTT_NOT_CONVERGED: i32 = 5;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// A failed call: the status code the entry point returns plus the message stashed for
/// `qntt_last_error`.
#[derive(Debug)]
pub struct Error {
    pub code: i32,
    pub msg: String,
}

impl Error {
    pub fn new(code: i32, msg: impl Into<String>) -> Self {
        Error {
            code,
            msg: msg.into(),
        }
    }
    pub fn null(what: &str) -> Self {
        Error::new(QNTT_NULL_ARG, format!("null pointer argument: {what}"))
    }
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::new(QNTT_INVALID_ARG, msg)
    }
}

/// Every exported function's body returns this.
pub type CResult = Result<(), Error>;

pub fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg.into());
}

pub fn clear_last_error() {
    LAST_ERROR.with(|e| e.borrow_mut().clear());
}

pub fn with_last_error<R>(f: impl FnOnce(&str) -> R) -> R {
    LAST_ERROR.with(|e| f(&e.borrow()))
}

/// Run `f`, catching panics. Clears the last-error slot on entry so a stale message can never be
/// mistaken for the current call's.
pub fn catch(f: impl FnOnce() -> CResult) -> i32 {
    clear_last_error();
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => QNTT_OK,
        Ok(Err(e)) => {
            set_last_error(e.msg);
            e.code
        }
        Err(payload) => {
            let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                (*s).to_string()
            } else if let Some(s) = payload.downcast_ref::<String>() {
                s.clone()
            } else {
                "panic with a non-string payload".to_string()
            };
            set_last_error(format!("panic: {msg}"));
            QNTT_PANIC
        }
    }
}

/// `Ok(&T)` from a `*const T` that must be non-null.
///
/// # Safety
/// `p` must be null or point to a valid `T` that outlives `'a`.
pub unsafe fn ptr_ref<'a, T>(p: *const T, what: &str) -> Result<&'a T, Error> {
    if p.is_null() {
        Err(Error::null(what))
    } else {
        Ok(unsafe { &*p })
    }
}

/// `Ok(&mut T)` from a `*mut T` that must be non-null.
///
/// # Safety
/// `p` must be null or point to a valid, uniquely-owned `T` that outlives `'a`.
pub unsafe fn ptr_mut<'a, T>(p: *mut T, what: &str) -> Result<&'a mut T, Error> {
    if p.is_null() {
        Err(Error::null(what))
    } else {
        Ok(unsafe { &mut *p })
    }
}

/// Borrow a caller-provided input buffer as a slice (`len == 0` is allowed with a null pointer).
///
/// # Safety
/// `p` must point to `len` initialized `T`s valid for `'a`, or be null with `len == 0`.
pub unsafe fn ptr_slice<'a, T>(p: *const T, len: usize, what: &str) -> Result<&'a [T], Error> {
    if len == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(Error::null(what));
    }
    Ok(unsafe { std::slice::from_raw_parts(p, len) })
}

/// Borrow a caller-provided output buffer as a mutable slice.
///
/// # Safety
/// `p` must point to `len` writable `T`s valid for `'a`, or be null with `len == 0`.
pub unsafe fn ptr_slice_mut<'a, T>(
    p: *mut T,
    len: usize,
    what: &str,
) -> Result<&'a mut [T], Error> {
    if len == 0 {
        return Ok(&mut []);
    }
    if p.is_null() {
        return Err(Error::null(what));
    }
    Ok(unsafe { std::slice::from_raw_parts_mut(p, len) })
}

/// Copy `src` into the caller's buffer, always reporting the required length through `out_len`
/// (so a caller can size-query with `cap == 0`). Fails with [`QNTT_INVALID_ARG`] when the buffer
/// is non-empty but too short.
///
/// # Safety
/// `dst`/`out_len` must be null or valid writable pointers as described.
pub unsafe fn copy_out<T: Copy>(
    src: &[T],
    dst: *mut T,
    cap: usize,
    out_len: *mut usize,
    what: &str,
) -> CResult {
    if !out_len.is_null() {
        unsafe { *out_len = src.len() };
    }
    if cap == 0 && dst.is_null() {
        return Ok(());
    }
    if cap < src.len() {
        return Err(Error::invalid(format!(
            "{what}: output buffer holds {cap} elements, need {}",
            src.len()
        )));
    }
    let dst = unsafe { ptr_slice_mut(dst, src.len(), what)? };
    dst.copy_from_slice(src);
    Ok(())
}
