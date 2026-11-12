// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Opaque, refcounted, type-tagged handles.

use std::sync::Arc;

use crate::err::{Error, QNTT_INVALID_ARG};

/// The two scalars the ABI instantiates, carrying the tag bits that keep an `f64` handle from
/// being accepted where a `C64` one belongs. Implemented only for `f64` and `C64`, so a `Layer`
/// impl generic in `T: ScalarTag` automatically has handle types for every object it names.
pub trait ScalarTag: qntt::Scalar {
    /// OR-ed into every handle tag of this scalar.
    const TAG_BITS: u64;
}

impl ScalarTag for f64 {
    const TAG_BITS: u64 = 0x0000;
}

impl ScalarTag for qntt::C64 {
    const TAG_BITS: u64 = 0x0100;
}

/// Implemented by every type exposed as an opaque handle.
pub trait HandleType: Sized + Send + Sync + 'static {
    /// Distinct per exposed type; checked on borrow.
    const TAG: u64;
    /// Used in error messages.
    const NAME: &'static str;
}

/// Heap payload behind an opaque pointer.
#[repr(C)]
pub struct Handle<T: HandleType> {
    tag: u64,
    value: T,
}

impl<T: HandleType> Handle<T> {
    pub fn value(&self) -> &T {
        &self.value
    }
}

/// Leak a new handle to the caller.
pub fn into_raw<T: HandleType>(value: T) -> *mut Handle<T> {
    Arc::into_raw(Arc::new(Handle { tag: T::TAG, value })) as *mut Handle<T>
}

/// Write a freshly created handle through an out-pointer, or drop it if the caller passed null.
///
/// # Safety
/// `out` must be null or a valid writable `*mut *mut Handle<T>`.
pub unsafe fn emit<T: HandleType>(value: T, out: *mut *mut Handle<T>) -> Result<(), Error> {
    if out.is_null() {
        return Err(Error::null("output handle"));
    }
    unsafe { *out = into_raw(value) };
    Ok(())
}

/// Borrow the payload of a handle, checking null and the type tag.
///
/// # Safety
/// `p` must be null or a pointer previously returned by [`into_raw`] for the same `T` and not yet
/// released.
pub unsafe fn borrow<'a, T: HandleType>(p: *const Handle<T>) -> Result<&'a T, Error> {
    if p.is_null() {
        return Err(Error::null(T::NAME));
    }
    let h = unsafe { &*p };
    if h.tag != T::TAG {
        return Err(Error::new(
            QNTT_INVALID_ARG,
            format!("handle type mismatch: expected {}", T::NAME),
        ));
    }
    Ok(&h.value)
}

/// Borrow the payload mutably. Succeeds only when the caller's pointer is the *sole* reference,
/// which is the normal case: the C API hands out one reference per object and only takes a second
/// for parent-keepalive. An aliased handle is reported rather
/// than silently copied, so a driver can never mutate an object another handle observes.
///
/// # Safety
/// Same as [`borrow`], plus: no other thread may be using `p` concurrently (the C-side contract —
/// mutation is single-threaded per object).
pub unsafe fn borrow_mut<'a, T: HandleType>(p: *mut Handle<T>) -> Result<&'a mut T, Error> {
    unsafe { borrow(p as *const Handle<T>)? };
    let arc = std::mem::ManuallyDrop::new(unsafe { Arc::from_raw(p as *const Handle<T>) });
    if Arc::strong_count(&arc) != 1 {
        return Err(Error::new(
            QNTT_INVALID_ARG,
            format!(
                "{} handle is shared; in-place mutation needs sole ownership",
                T::NAME
            ),
        ));
    }
    // SAFETY: strong_count == 1 means this pointer is the only live reference, so a `&mut` is
    Ok(unsafe { &mut (*p).value })
}

/// Take an owning reference to a handle (for parent-keepalive), leaving the caller's pointer
/// valid.
///
/// # Safety
/// Same as [`borrow`].
pub unsafe fn retain<T: HandleType>(p: *const Handle<T>) -> Result<Arc<Handle<T>>, Error> {
    unsafe { borrow(p)? };
    unsafe {
        Arc::increment_strong_count(p);
        Ok(Arc::from_raw(p))
    }
}

/// Drop the caller's reference. Null is a no-op; a wrong tag is ignored (freeing is infallible by
/// contract, and a mistyped free is the one case where returning an error would leak).
///
/// # Safety
/// `p` must be null or a pointer from [`into_raw`] that has not already been released.
pub unsafe fn release<T: HandleType>(p: *mut Handle<T>) {
    if p.is_null() {
        return;
    }
    unsafe {
        if (*p).tag != T::TAG {
            return;
        }
        drop(Arc::from_raw(p as *const Handle<T>));
    }
}

/// Declare a handle type with its tag. Tags are `b"qntt"` in the high word plus a per-type id, so
/// an uninitialized-memory tag collision is vanishingly unlikely.
#[macro_export]
macro_rules! handle_type {
    ($ty:ty, $id:expr, $name:expr) => {
        impl $crate::handle::HandleType for $ty {
            const TAG: u64 = 0x716e_7474_0000_0000u64 | ($id as u64);
            const NAME: &'static str = $name;
        }
    };
}
