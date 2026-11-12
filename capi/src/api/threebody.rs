// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::{Scalar, Sparse3Body};

use crate::err::{CResult, ptr_mut, ptr_slice};
use crate::handle::{Handle, borrow, borrow_mut, emit};

/// # Safety
/// `out` must be a valid handle slot.
pub unsafe fn w_new<T: Scalar>(norb: usize, out: *mut *mut Handle<Sparse3Body<T>>) -> CResult
where
    Sparse3Body<T>: crate::handle::HandleType,
{
    unsafe { emit(Sparse3Body::zeros(norb), out) }
}

/// Bulk COO fill: `idx` holds `n` six-index tuples (creation/annihilation slot order, 0-based),
/// `vals` the matching entries.
///
/// # Safety
/// `idx` must hold `6·n` indices and `vals` `n` scalars; `w` must be unshared.
pub unsafe fn w_set_coo<T: Scalar>(
    w: *mut Handle<Sparse3Body<T>>,
    idx: *const usize,
    vals: *const T,
    n: usize,
) -> CResult
where
    Sparse3Body<T>: crate::handle::HandleType,
{
    let idx = unsafe { ptr_slice(idx, 6 * n, "idx")? };
    let vals = unsafe { ptr_slice(vals, n, "vals")? };
    let w = unsafe { borrow_mut(w)? };
    for k in 0..n {
        let i: [usize; 6] = idx[6 * k..6 * k + 6].try_into().expect("chunk of 6");
        w.set(&i, vals[k]);
    }
    Ok(())
}

/// Number of stored (nonzero) three-body entries.
///
/// # Safety
/// `out` must be writable.
pub unsafe fn w_nnz<T: Scalar>(w: *const Handle<Sparse3Body<T>>, out: *mut usize) -> CResult
where
    Sparse3Body<T>: crate::handle::HandleType,
{
    let w = unsafe { borrow(w)? };
    unsafe { *ptr_mut(out, "out")? = w.nnz() };
    Ok(())
}
