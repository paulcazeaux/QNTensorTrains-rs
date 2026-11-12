// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::core::Family;
use qntt::{
    CoeffTensor, Irrep, QnSpace, Sector, Sparse3Body, SparseHamiltonian, ThreeBody, TtVector,
};

use crate::enums;
use crate::err::{CResult, Error, copy_out, ptr_mut, ptr_slice, ptr_slice_mut};
use crate::handle::{Handle, ScalarTag, borrow, borrow_mut, emit};

use super::space::{pg_space, ranks_from_flat_pg, ranks_to_flat_pg, report_point_group};

/// # Safety
/// As [`pg_space`]; `out` must be a valid handle slot.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_zeros_pg<T: ScalarTag>(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out: *mut *mut Handle<TtVector<T>>,
) -> CResult {
    let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    unsafe { emit(qntt::tt_zeros::<T>(space), out) }
}

/// # Safety
/// As [`pg_space`]; `ranks` must hold `ranks_len` entries.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_from_ranks_pg<T: ScalarTag>(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    ranks: *const usize,
    ranks_len: usize,
    out: *mut *mut Handle<TtVector<T>>,
) -> CResult {
    let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    let flat = unsafe { ptr_slice(ranks, ranks_len, "ranks")? };
    let grids = ranks_from_flat_pg(&space, flat)?;
    let cores = (0..space.d)
        .map(|k| qntt::core::SparseCore::zeros(space, k, &grids[k], &grids[k + 1]))
        .collect();
    unsafe { emit(TtVector::new(space, cores), out) }
}

/// Gaussian random train. Ranks at sectors the point group makes unreachable are **silently
/// masked to zero** by the core (`tt_randn` clamps against `max_ranks`), so a caller may feed a
/// uniform grid and still get an irrep-pure train.
///
/// # Safety
/// As [`vec_from_ranks_pg`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_randn_pg<T: ScalarTag>(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    ranks: *const usize,
    ranks_len: usize,
    seed: u64,
    out: *mut *mut Handle<TtVector<T>>,
) -> CResult {
    let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    let flat = unsafe { ptr_slice(ranks, ranks_len, "ranks")? };
    let grids = ranks_from_flat_pg(&space, flat)?;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    unsafe { emit(qntt::tt_randn::<T, _>(space, &grids, &mut rng), out) }
}

/// A single determinant placed in a point-group space. `(Nup, Ndn)` come from the occupation
/// flags; the determinant's total irrep must equal `target`, which is checked here (the core
/// asserts, and an assert would cross as `QNTT_PANIC`).
///
/// # Safety
/// `occ_up`/`occ_dn` must hold `d` entries each; the rest as [`pg_space`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn tt_state_pg<T: ScalarTag>(
    occ_up: *const i32,
    occ_dn: *const i32,
    d: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out: *mut *mut Handle<TtVector<T>>,
) -> CResult {
    let up = unsafe { ptr_slice(occ_up, d, "occ_up")? };
    let dn = unsafe { ptr_slice(occ_dn, d, "occ_dn")? };
    let up: Vec<bool> = up.iter().map(|&v| v != 0).collect();
    let dn: Vec<bool> = dn.iter().map(|&v| v != 0).collect();
    let nup = up.iter().filter(|&&o| o).count();
    let ndn = dn.iter().filter(|&&o| o).count();
    let space = unsafe { pg_space(d, nup, ndn, pg_kind, orbsym, orbsym_len, target)? };
    let mut s = Sector::new(0, 0);
    for k in 0..d {
        s = Family::from_occupation(up[k], dn[k]).apply(s, space.pg.irrep_of(k));
    }
    if s.irrep != space.target {
        return Err(Error::invalid(format!(
            "determinant's total irrep {} differs from the requested target {}",
            s.irrep.index(),
            space.target.index()
        )));
    }
    unsafe { emit(qntt::tt_state_in::<T>(space, &up, &dn), out) }
}

/// The 0/1 indicator of the `(Nup, Ndn, target)` block. At C1 this is the plain `(Nup, Ndn)`
/// indicator; it has no legacy counterpart on the C surface.
///
/// # Safety
/// As [`pg_space`]; `out` must be a valid handle slot.
#[allow(clippy::too_many_arguments)]
pub unsafe fn tt_ones_pg<T: ScalarTag>(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out: *mut *mut Handle<TtVector<T>>,
) -> CResult {
    let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    let caps = qntt::max_ranks(&space);
    if caps[d][Sector::with_irrep(space.nup, space.ndn, space.target)] == 0 {
        return Err(Error::invalid(format!(
            "no determinant has ({}, {}) electrons and irrep {} in this space — the target-irrep \
             indicator is the zero vector",
            space.nup,
            space.ndn,
            space.target.index()
        )));
    }
    unsafe { emit(qntt::tt_ones::<T>(space), out) }
}

/// The train's per-bond rank grids, flattened with the irrep axis.
///
/// # Safety
/// `out` must hold `cap` entries or be null when `cap == 0`.
pub unsafe fn vec_ranks_pg<T: ScalarTag>(
    x: *const Handle<TtVector<T>>,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let flat = ranks_to_flat_pg(&x.space, &x.ranks);
    unsafe { copy_out(&flat, out, cap, out_len, "ranks") }
}

/// Copy one block out, column-major, addressing the sector by `(nup, ndn, irrep)`. An absent or
/// zero-rank block reports `rows = cols = 0` and copies nothing — the common case at `|G| > 1`,
/// where admissible strictly exceeds reachable.
///
/// # Safety
/// `out` must hold `cap` scalars or be null when `cap == 0` (size query).
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_block_get_pg<T: ScalarTag>(
    x: *const Handle<TtVector<T>>,
    site: usize,
    family: i32,
    sa: usize,
    sb: usize,
    irrep: u8,
    out: *mut T,
    cap: usize,
    out_rows: *mut usize,
    out_cols: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let f = enums::family(family)?;
    let d = x.space.d;
    if site >= d {
        return Err(Error::invalid(format!("site {site} out of range 0..{d}")));
    }
    let s = sector_arg(&x.space, sa, sb, irrep)?;
    let (rows, cols) = x.cores[site].block_dims(f, s).unwrap_or((0, 0));
    unsafe {
        *ptr_mut(out_rows, "out_rows")? = rows;
        *ptr_mut(out_cols, "out_cols")? = cols;
    }
    let n = rows * cols;
    if cap == 0 || n == 0 {
        return Ok(());
    }
    if cap < n {
        return Err(Error::invalid(format!(
            "block buffer holds {cap} scalars, need {n}"
        )));
    }
    let dst = unsafe { ptr_slice_mut(out, n, "block out")? };
    x.cores[site].block_copy_out(f, s, dst);
    Ok(())
}

/// Overwrite one block from a column-major buffer. Requires a uniquely-owned handle.
///
/// # Safety
/// `src` must hold `rows·cols` scalars.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_block_set_pg<T: ScalarTag>(
    x: *mut Handle<TtVector<T>>,
    site: usize,
    family: i32,
    sa: usize,
    sb: usize,
    irrep: u8,
    src: *const T,
    rows: usize,
    cols: usize,
) -> CResult {
    let f = enums::family(family)?;
    let xv = unsafe { borrow_mut(x)? };
    let d = xv.space.d;
    if site >= d {
        return Err(Error::invalid(format!("site {site} out of range 0..{d}")));
    }
    let src = unsafe { ptr_slice(src, rows * cols, "block src")? };
    let s = sector_arg(&xv.space, sa, sb, irrep)?;
    let Some((br, bc)) = xv.cores[site].block_dims(f, s) else {
        return Err(Error::invalid(format!(
            "no block at site {site}, family {family}, sector ({sa}, {sb}, {irrep})"
        )));
    };
    if br != rows || bc != cols {
        return Err(Error::invalid(format!(
            "block is {br}×{bc}, source is {rows}×{cols}"
        )));
    }
    xv.cores[site].block_copy_in(f, s, src);
    Ok(())
}

fn sector_arg(space: &QnSpace, sa: usize, sb: usize, irrep: u8) -> Result<Sector, Error> {
    if irrep >= space.pg.order() {
        return Err(Error::invalid(format!(
            "irrep {irrep} is out of range 0..{} for this train's point group",
            space.pg.order()
        )));
    }
    Ok(Sector::with_irrep(sa, sb, Irrep::new(irrep)))
}

/// # Safety
/// Out-pointers as described in [`report_point_group`].
pub unsafe fn vec_point_group<T: ScalarTag>(
    x: *const Handle<TtVector<T>>,
    out_nbits: *mut i32,
    out_target: *mut i32,
    out_irreps: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    unsafe { report_point_group(&x.space, out_nbits, out_target, out_irreps, cap, out_len) }
}

/// # Safety
/// Out-pointers as described in [`report_point_group`].
pub unsafe fn ham_point_group<T: ScalarTag>(
    h: *const Handle<SparseHamiltonian<T>>,
    out_nbits: *mut i32,
    out_target: *mut i32,
    out_irreps: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let h = unsafe { borrow(h)? };
    unsafe { report_point_group(&h.space, out_nbits, out_target, out_irreps, cap, out_len) }
}

/// [`crate::api::ham_new`]'s point-group twin: identical marshalling, a point-group space.
///
/// # Safety
/// `t`/`v` must hold `t_len`/`v_len` scalars; `w` null or a live three-body handle.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ham_new_pg<T: ScalarTag>(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    t: *const T,
    t_len: usize,
    v: *const T,
    v_len: usize,
    w: *const Handle<Sparse3Body<T>>,
    eps: f64,
    construction: i32,
    out: *mut *mut Handle<SparseHamiltonian<T>>,
) -> CResult {
    let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    let t = coeff_from_flat(d, 2, unsafe { ptr_slice(t, t_len, "t")? })?;
    let v = coeff_from_flat(d, 4, unsafe { ptr_slice(v, v_len, "v")? })?;
    let c = enums::construction(construction)?;
    let w_ref = if w.is_null() {
        None
    } else {
        Some(unsafe { borrow(w)? })
    };
    let w_dyn: Option<&dyn ThreeBody<T>> = w_ref.map(|w| w as &dyn ThreeBody<T>);
    let h = match c {
        None => SparseHamiltonian::new(space, &t, &v, w_dyn, eps),
        Some(c) => SparseHamiltonian::new_with(space, &t, &v, w_dyn, eps, c),
    };
    unsafe { emit(h, out) }
}

fn coeff_from_flat<T: qntt::Scalar>(
    d: usize,
    rank: usize,
    data: &[T],
) -> Result<CoeffTensor<T>, Error> {
    let need = d
        .checked_pow(rank as u32)
        .ok_or_else(|| Error::invalid(format!("d^{rank} overflows a machine word for d = {d}")))?;
    if data.len() != need {
        return Err(Error::invalid(format!(
            "rank-{rank} coefficient tensor over d={d} needs {need} entries, got {}",
            data.len()
        )));
    }
    Ok(CoeffTensor {
        d,
        rank,
        data: data.to_vec(),
    })
}
