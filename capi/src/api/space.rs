// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::err::{CResult, Error, copy_out, ptr_mut};
use crate::layer::Layer;

use super::ranks::ranks_to_flat;

/// Validate a space descriptor without building anything.
pub fn space_check<L: Layer>(d: usize, a: usize, b: usize) -> CResult {
    L::space_new(d, a, b).map(|_| ())
}

/// `(rows, cols)` of the layer's sector grid.
///
/// # Safety
/// The out-pointers must be null or writable.
pub unsafe fn space_grid_extent<L: Layer>(
    d: usize,
    a: usize,
    b: usize,
    out_rows: *mut usize,
    out_cols: *mut usize,
) -> CResult {
    let space = L::space_new(d, a, b)?;
    let (rows, cols) = L::grid_extent(&space);
    unsafe {
        *ptr_mut(out_rows, "out_rows")? = rows;
        *ptr_mut(out_cols, "out_cols")? = cols;
    }
    Ok(())
}

/// Admissible sectors at `bond`, in canonical order, as parallel `(a, b)` arrays.
///
/// # Safety
/// Output buffers must hold `cap` elements or be null when `cap == 0` (size query).
#[allow(clippy::too_many_arguments)]
pub unsafe fn space_bond_sectors<L: Layer>(
    d: usize,
    a: usize,
    b: usize,
    bond: usize,
    out_a: *mut usize,
    out_b: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let space = L::space_new(d, a, b)?;
    if bond > d {
        return Err(Error::invalid(format!("bond {bond} out of range 0..={d}")));
    }
    let sectors = L::bond_sectors(&space, bond);
    let (av, bv): (Vec<usize>, Vec<usize>) = sectors
        .into_iter()
        .map(|s| {
            let (x, y) = L::sector_parts(s);
            (x, y)
        })
        .unzip();
    unsafe {
        copy_out(&av, out_a, cap, out_len, "bond sectors (a)")?;
        copy_out(&bv, out_b, cap, std::ptr::null_mut(), "bond sectors (b)")?;
    }
    Ok(())
}

/// The maximal admissible rank grid stack (`(d+1)·rows·cols` entries).
///
/// # Safety
/// `out` must hold `cap` elements or be null when `cap == 0`.
pub unsafe fn space_max_ranks<L: Layer>(
    d: usize,
    a: usize,
    b: usize,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let space = L::space_new(d, a, b)?;
    let flat = ranks_to_flat::<L>(&space, &L::max_ranks(&space));
    unsafe { copy_out(&flat, out, cap, out_len, "max ranks") }
}
