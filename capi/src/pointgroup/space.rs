// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::os::raw::{c_char, c_int};

use qntt::{Irrep, PointGroup, PointGroupKind, QnSpace, RankGrid, Sector};

use crate::enums;
use crate::err::{CResult, Error, catch, copy_out, ptr_mut, ptr_slice};
use crate::su2_pointgroup::{ranks_to_flat_pg as su2_ranks_to_flat_pg, su2_pg_space};

const MAX_POINT_GROUP_SITES: usize = 127;

pub(super) unsafe fn write_name(
    s: &str,
    buf: *mut c_char,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    if !out_len.is_null() {
        unsafe { *out_len = s.len() };
    }
    if cap == 0 {
        return Ok(());
    }
    if buf.is_null() {
        return Err(Error::null("buf"));
    }
    if cap < s.len() {
        return Err(Error::invalid(format!(
            "string buffer holds {cap} bytes, need {}",
            s.len()
        )));
    }
    let n = if cap > s.len() { s.len() + 1 } else { s.len() };
    let dst = unsafe { std::slice::from_raw_parts_mut(buf as *mut u8, n) };
    dst[..s.len()].copy_from_slice(s.as_bytes());
    if n > s.len() {
        dst[s.len()] = 0;
    }
    Ok(())
}

pub(crate) unsafe fn pg_space(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
) -> Result<QnSpace, Error> {
    let kind = enums::pg_kind(pg_kind)?;
    if a > d || b > d {
        return Err(Error::invalid(format!(
            "abelian space: need Nup ({a}) <= d ({d}) and Ndn ({b}) <= d"
        )));
    }
    let (pg, target) = unsafe { pg_table(d, kind, orbsym, orbsym_len, target)? };
    Ok(QnSpace::with_point_group(d, a, b, pg, target))
}

pub(crate) unsafe fn pg_table(
    d: usize,
    kind: PointGroupKind,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
) -> Result<(PointGroup, Irrep), Error> {
    let (site_cap, why) = if kind.order() > 1 {
        (
            MAX_POINT_GROUP_SITES,
            "the automaton's byte-packed orbital key",
        )
    } else {
        (
            PointGroup::MAX_SITES,
            "the packed point-group table's u128 plane width",
        )
    };
    if d > site_cap {
        return Err(Error::invalid(format!(
            "d={d} exceeds the {site_cap}-site point-group limit for {} ({why}); the legacy \
             (non-_pg) symbols are the any-d path for trivial-group work",
            kind.name()
        )));
    }
    if orbsym_len != d {
        return Err(Error::invalid(format!(
            "orbsym has {orbsym_len} entries, need d = {d}"
        )));
    }
    let ids = unsafe { ptr_slice(orbsym, orbsym_len, "orbsym")? };
    let pg = if kind.order() == 1 {
        for (k, &id) in ids.iter().enumerate() {
            if id != 1 {
                return Err(Error::invalid(format!(
                    "point group {}: Molpro id {id} at site {k} is out of range 1..=1",
                    kind.name()
                )));
            }
        }
        PointGroup::TRIVIAL
    } else {
        PointGroup::from_orbsym(kind, ids).map_err(|e| Error::invalid(e.to_string()))?
    };
    if target < 0 || target as u32 >= u32::from(kind.order()) {
        return Err(Error::invalid(format!(
            "target irrep {target} is out of range 0..{} for point group {}",
            kind.order(),
            kind.name()
        )));
    }
    Ok((pg, Irrep::new(target as u8)))
}

pub(crate) fn ranks_to_flat_pg(space: &QnSpace, grids: &[RankGrid]) -> Vec<usize> {
    let (rows, cols, nir) = (space.nup + 1, space.ndn + 1, space.pg.order() as usize);
    let mut out = Vec::with_capacity(grids.len() * nir * rows * cols);
    for g in grids {
        for ir in 0..nir {
            for i in 0..rows {
                for j in 0..cols {
                    out.push(g[Sector::with_irrep(i, j, Irrep::new(ir as u8))]);
                }
            }
        }
    }
    out
}

pub(crate) fn ranks_from_flat_pg(space: &QnSpace, flat: &[usize]) -> Result<Vec<RankGrid>, Error> {
    let (rows, cols, nir) = (space.nup + 1, space.ndn + 1, space.pg.order() as usize);
    let d = space.d;
    let stride = nir * rows * cols;
    if flat.len() != (d + 1) * stride {
        return Err(Error::invalid(format!(
            "rank grid buffer has {} entries, need (d+1)·|G|·rows·cols = {}",
            flat.len(),
            (d + 1) * stride
        )));
    }
    Ok((0..=d)
        .map(|b| {
            let mut g = RankGrid::zeros(space);
            let mut p = b * stride;
            for ir in 0..nir {
                for i in 0..rows {
                    for j in 0..cols {
                        let r = flat[p];
                        p += 1;
                        if r != 0 {
                            g[Sector::with_irrep(i, j, Irrep::new(ir as u8))] = r;
                        }
                    }
                }
            }
            g
        })
        .collect())
}

pub(super) unsafe fn report_point_group(
    space: &QnSpace,
    out_nbits: *mut i32,
    out_target: *mut i32,
    out_irreps: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    unsafe {
        *ptr_mut(out_nbits, "out_nbits")? = i32::from(space.pg.nbits());
        *ptr_mut(out_target, "out_target")? = i32::from(space.target.index());
    }
    let irreps: Vec<u8> = (0..space.d).map(|k| space.pg.irrep_of(k).index()).collect();
    unsafe { copy_out(&irreps, out_irreps, cap, out_len, "point group irreps") }
}

/// Validate an SU(2) point-group space descriptor without building anything.
///
/// # Safety
/// `orbsym` must hold `orbsym_len` bytes, or be null when `orbsym_len == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_su2_space_check_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
) -> c_int {
    catch(|| unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target).map(|_| ()) })
}

/// `(rows, cols, nirrep)` of the SU(2) sector grid — `(N+1, N+1, |G|)`.
///
/// # Safety
/// Out-pointers must be writable.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_su2_space_grid_extent_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out_rows: *mut usize,
    out_cols: *mut usize,
    out_nirrep: *mut usize,
) -> c_int {
    catch(|| {
        let space = unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
        unsafe {
            *ptr_mut(out_rows, "out_rows")? = space.n + 1;
            *ptr_mut(out_cols, "out_cols")? = space.n + 1;
            *ptr_mut(out_nirrep, "out_nirrep")? = space.pg.order() as usize;
        }
        Ok(())
    })
}

/// Admissible SU(2) sectors at `bond`, in the core's canonical order, as three parallel arrays
/// (`n`, `2s`, 0-based XOR irrep). Any output array may be null; `out_len` always reports the
/// count.
///
/// # Safety
/// Output buffers must hold `cap` elements or be null when `cap == 0` (size query).
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_su2_space_bond_sectors_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    bond: usize,
    out_a: *mut usize,
    out_b: *mut usize,
    out_irrep: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let space = unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
        if bond > d {
            return Err(Error::invalid(format!("bond {bond} out of range 0..={d}")));
        }
        let sectors = space.bond_sectors(bond);
        let av: Vec<usize> = sectors.iter().map(|s| s.n).collect();
        let bv: Vec<usize> = sectors.iter().map(|s| s.two_s).collect();
        let iv: Vec<u8> = sectors.iter().map(|s| s.irrep.index()).collect();
        unsafe {
            let cap_a = if out_a.is_null() { 0 } else { cap };
            let cap_b = if out_b.is_null() { 0 } else { cap };
            let cap_irrep = if out_irrep.is_null() { 0 } else { cap };
            copy_out(&av, out_a, cap_a, out_len, "bond sectors (n)")?;
            copy_out(&bv, out_b, cap_b, std::ptr::null_mut(), "bond sectors (2s)")?;
            copy_out(
                &iv,
                out_irrep,
                cap_irrep,
                std::ptr::null_mut(),
                "bond sectors (irrep)",
            )
        }
    })
}

/// The maximal admissible SU(2) rank-grid stack, `(d+1)·|G|·rows·cols` entries in the layout
/// described in the module docs. Reduced (multiplet) ranks, not abelian state counts.
///
/// # Safety
/// `out` must hold `cap` elements or be null when `cap == 0`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_su2_space_max_ranks_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let space = unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
        let flat = su2_ranks_to_flat_pg(&space, &qntt::su2::qn::max_ranks(&space));
        unsafe { copy_out(&flat, out, cap, out_len, "max ranks") }
    })
}

/// Validate a point-group space descriptor without building anything.
///
/// # Safety
/// `orbsym` must hold `orbsym_len` bytes, or be null when `orbsym_len == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_space_check_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
) -> c_int {
    catch(|| unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target).map(|_| ()) })
}

/// `(rows, cols, nirrep)` of the sector grid — `(Nup+1, Ndn+1, |G|)`.
///
/// # Safety
/// Out-pointers must be writable.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_space_grid_extent_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out_rows: *mut usize,
    out_cols: *mut usize,
    out_nirrep: *mut usize,
) -> c_int {
    catch(|| {
        let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
        unsafe {
            *ptr_mut(out_rows, "out_rows")? = space.nup + 1;
            *ptr_mut(out_cols, "out_cols")? = space.ndn + 1;
            *ptr_mut(out_nirrep, "out_nirrep")? = space.pg.order() as usize;
        }
        Ok(())
    })
}

/// Admissible sectors at `bond`, in the core's canonical order, as three parallel arrays
/// (`nup`, `ndn`, 0-based XOR irrep). Any output array may be null; `out_len` always reports the
/// count.
///
/// # Safety
/// Output buffers must hold `cap` elements or be null when `cap == 0` (size query).
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_space_bond_sectors_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    bond: usize,
    out_a: *mut usize,
    out_b: *mut usize,
    out_irrep: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
        if bond > d {
            return Err(Error::invalid(format!("bond {bond} out of range 0..={d}")));
        }
        let sectors = space.bond_sectors(bond);
        let av: Vec<usize> = sectors.iter().map(|s| s.nup).collect();
        let bv: Vec<usize> = sectors.iter().map(|s| s.ndn).collect();
        let iv: Vec<u8> = sectors.iter().map(|s| s.irrep.index()).collect();
        unsafe {
            let cap_a = if out_a.is_null() { 0 } else { cap };
            let cap_b = if out_b.is_null() { 0 } else { cap };
            let cap_irrep = if out_irrep.is_null() { 0 } else { cap };
            copy_out(&av, out_a, cap_a, out_len, "bond sectors (a)")?;
            copy_out(&bv, out_b, cap_b, std::ptr::null_mut(), "bond sectors (b)")?;
            copy_out(
                &iv,
                out_irrep,
                cap_irrep,
                std::ptr::null_mut(),
                "bond sectors (irrep)",
            )
        }
    })
}

/// The maximal admissible rank-grid stack, `(d+1)·|G|·rows·cols` entries in the layout described
/// in the module docs.
///
/// # Safety
/// `out` must hold `cap` elements or be null when `cap == 0`.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_space_max_ranks_pg(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let space = unsafe { pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
        let flat = ranks_to_flat_pg(&space, &qntt::max_ranks(&space));
        unsafe { copy_out(&flat, out, cap, out_len, "max ranks") }
    })
}
