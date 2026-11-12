// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use qntt::{Irrep, PointGroupKind};

use crate::enums;
use crate::err::{Error, catch, copy_out, ptr_mut, ptr_slice};

use super::space::write_name;

/// Number of irreps of a point-group kind (`1`, `2`, `4` or `8`).
///
/// # Safety
/// `out_order` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_order(kind: i32, out_order: *mut usize) -> c_int {
    catch(|| {
        let k = enums::pg_kind(kind)?;
        unsafe { *ptr_mut(out_order, "out_order")? = k.order() as usize };
        Ok(())
    })
}

/// The conventional kind of a given order — the same guess the FCIDUMP reader makes when no kind
/// is supplied. Numerically free: all same-order abelian groups share one product table, so the
/// guess affects irrep NAMES only. `order` outside `{1, 2, 4, 8}` is `QNTT_INVALID_ARG`.
///
/// # Safety
/// `out_kind` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_default_kind(order: usize, out_kind: *mut i32) -> c_int {
    catch(|| {
        if !matches!(order, 1 | 2 | 4 | 8) {
            return Err(Error::invalid(format!(
                "point group order {order} (want 1, 2, 4 or 8)"
            )));
        }
        let k = PointGroupKind::default_for_order(order as u8);
        unsafe { *ptr_mut(out_kind, "out_kind")? = enums::pg_kind_code(k) };
        Ok(())
    })
}

/// Wire code of a kind named as a string (ASCII case-insensitive: `"C2v"`, `"c2v"`, `"D2H"`, …).
///
/// # Safety
/// `name` must be a NUL-terminated string; `out_kind` writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_kind_from_name(name: *const c_char, out_kind: *mut i32) -> c_int {
    catch(|| {
        if name.is_null() {
            return Err(Error::null("name"));
        }
        let s = unsafe { CStr::from_ptr(name) }
            .to_str()
            .map_err(|_| Error::invalid("name is not valid UTF-8"))?;
        let k = PointGroupKind::from_name(s)
            .ok_or_else(|| Error::invalid(format!("unknown point group name {s:?}")))?;
        unsafe { *ptr_mut(out_kind, "out_kind")? = enums::pg_kind_code(k) };
        Ok(())
    })
}

/// Canonical name of a kind (`"C1"`, `"Ci"`, `"C2"`, `"Cs"`, `"D2"`, `"C2v"`, `"C2h"`, `"D2h"`).
///
/// # Safety
/// See the module contract in `include/qntt.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_kind_name(
    kind: i32,
    buf: *mut c_char,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let k = enums::pg_kind(kind)?;
        unsafe { write_name(k.name(), buf, cap, out_len) }
    })
}

/// Name of one irrep, indexed by its **0-based XOR index** (PySCF's canonical ordering), e.g.
/// C2v XOR 2 is `"B1"`.
///
/// # Safety
/// See the module contract in `include/qntt.h`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_irrep_name(
    kind: i32,
    irrep: i32,
    buf: *mut c_char,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let k = enums::pg_kind(kind)?;
        let ir = irrep_arg(k, irrep)?;
        let name = k
            .irrep_name(ir)
            .map_err(|e| Error::invalid(e.to_string()))?;
        unsafe { write_name(name, buf, cap, out_len) }
    })
}

/// 1-based Molpro id → 0-based XOR irrep index. The FCIDUMP boundary's only shift.
///
/// # Safety
/// `out_irrep` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_irrep_from_molpro(
    kind: i32,
    molpro_id: i32,
    out_irrep: *mut i32,
) -> c_int {
    catch(|| {
        let k = enums::pg_kind(kind)?;
        let id = u8::try_from(molpro_id)
            .map_err(|_| Error::invalid(format!("Molpro id {molpro_id} does not fit in a byte")))?;
        let ir = k
            .irrep_from_molpro(id)
            .map_err(|e| Error::invalid(e.to_string()))?;
        unsafe { *ptr_mut(out_irrep, "out_irrep")? = i32::from(ir.index()) };
        Ok(())
    })
}

/// 0-based XOR irrep index → 1-based Molpro id (the inverse of [`qntt_pg_irrep_from_molpro`]).
///
/// # Safety
/// `out_molpro_id` must be writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_molpro_from_irrep(
    kind: i32,
    irrep: i32,
    out_molpro_id: *mut i32,
) -> c_int {
    catch(|| {
        let k = enums::pg_kind(kind)?;
        let ir = irrep_arg(k, irrep)?;
        let id = k
            .molpro_from_irrep(ir)
            .map_err(|e| Error::invalid(e.to_string()))?;
        unsafe { *ptr_mut(out_molpro_id, "out_molpro_id")? = i32::from(id) };
        Ok(())
    })
}

/// Map a whole `ORBSYM` run of 1-based Molpro ids to 0-based XOR irreps, one per site.
///
/// # Safety
/// `orbsym` must hold `n` bytes; `out_irreps` `cap` bytes or null when `cap == 0`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qntt_pg_orbsym_to_irreps(
    kind: i32,
    orbsym: *const u8,
    n: usize,
    out_irreps: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> c_int {
    catch(|| {
        let k = enums::pg_kind(kind)?;
        let ids = unsafe { ptr_slice(orbsym, n, "orbsym")? };
        let irreps: Result<Vec<u8>, Error> = ids
            .iter()
            .map(|&id| {
                k.irrep_from_molpro(id)
                    .map(|ir| ir.index())
                    .map_err(|e| Error::invalid(e.to_string()))
            })
            .collect();
        unsafe { copy_out(&irreps?, out_irreps, cap, out_len, "irreps") }
    })
}

fn irrep_arg(kind: PointGroupKind, irrep: i32) -> Result<Irrep, Error> {
    if irrep < 0 || irrep as u32 >= u32::from(kind.order()) {
        return Err(Error::invalid(format!(
            "irrep {irrep} is out of range 0..{} for point group {}",
            kind.order(),
            kind.name()
        )));
    }
    Ok(Irrep::new(irrep as u8))
}
