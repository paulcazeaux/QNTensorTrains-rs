// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::{TtHamiltonian, TtState};
use qntt::{CoeffTensor, Scalar, Sparse3Body, ThreeBody};

use crate::enums;
use crate::err::{CResult, Error, QNTT_INVALID_ARG, copy_out, ptr_mut, ptr_slice};
use crate::handle::{Handle, borrow, emit};
use crate::layer::Layer;

fn coeff_from_flat<T: Scalar>(d: usize, rank: usize, data: &[T]) -> Result<CoeffTensor<T>, Error> {
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

/// Compile the second-quantized operator. `t`/`v` are flat **row-major** (first axis most
/// significant) `d²`/`d⁴` buffers; `w` is an optional sparse three-body handle.
///
/// # Safety
/// `t`/`v` must hold `t_len`/`v_len` scalars; `w` null or a live three-body handle.
#[allow(clippy::too_many_arguments)]
pub unsafe fn ham_new<L: Layer>(
    d: usize,
    a: usize,
    b: usize,
    t: *const L::T,
    t_len: usize,
    v: *const L::T,
    v_len: usize,
    w: *const Handle<Sparse3Body<L::T>>,
    eps: f64,
    construction: i32,
    out: *mut *mut Handle<L::Ham>,
) -> CResult
where
    Sparse3Body<L::T>: crate::handle::HandleType,
{
    let space = L::space_new(d, a, b)?;
    let t = coeff_from_flat(d, 2, unsafe { ptr_slice(t, t_len, "t")? })?;
    let v = coeff_from_flat(d, 4, unsafe { ptr_slice(v, v_len, "v")? })?;
    let c = enums::construction(construction)?;
    let w_ref = if w.is_null() {
        None
    } else {
        Some(unsafe { borrow(w)? })
    };
    let w_dyn: Option<&dyn ThreeBody<L::T>> = w_ref.map(|w| w as &dyn ThreeBody<L::T>);
    let h = L::ham_new(space, &t, &v, w_dyn, eps, c)?;
    unsafe { emit(h, out) }
}

/// # Safety
/// Out-pointers must be null or writable.
pub unsafe fn ham_space<L: Layer>(
    h: *const Handle<L::Ham>,
    out_d: *mut usize,
    out_a: *mut usize,
    out_b: *mut usize,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let (d, a, b) = L::space_parts(&L::ham_space(h));
    unsafe {
        *ptr_mut(out_d, "out_d")? = d;
        *ptr_mut(out_a, "out_a")? = a;
        *ptr_mut(out_b, "out_b")? = b;
    }
    Ok(())
}

/// Summed-over-sector automaton bond dimensions (one per bond). Reports
/// [`QNTT_INVALID_ARG`] on layers that do not expose them.
///
/// # Safety
/// `out` must hold `cap` entries or be null when `cap == 0`.
pub unsafe fn ham_bond_dims<L: Layer>(
    h: *const Handle<L::Ham>,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let Some(dims) = L::ham_bond_dims(h) else {
        return Err(Error::new(
            QNTT_INVALID_ARG,
            "this layer does not report automaton bond dimensions",
        ));
    };
    unsafe { copy_out(&dims, out, cap, out_len, "bond dims") }
}

/// `H·x`, unrounded.
///
/// # Safety
/// `out` must be a valid handle slot.
pub unsafe fn ham_apply<L: Layer>(
    h: *const Handle<L::Ham>,
    x: *const Handle<L::Vector>,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x = unsafe { borrow(x)? };
    unsafe { emit(h.matvec(x), out) }
}

/// `⟨x, H·x⟩` — the raw expectation, NOT divided by `⟨x, x⟩`.
///
/// # Safety
/// `out` must be writable.
pub unsafe fn ham_expectation<L: Layer>(
    h: *const Handle<L::Ham>,
    x: *const Handle<L::Vector>,
    out: *mut L::T,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x = unsafe { borrow(x)? };
    unsafe { *ptr_mut(out, "out")? = h.rayleigh_quotient(x) };
    Ok(())
}

/// `⟨x, H·x⟩ / ⟨x, x⟩` — the actual Rayleigh quotient, safe on an unnormalized train.
///
/// # Safety
/// `out` must be writable.
pub unsafe fn ham_rayleigh<L: Layer>(
    h: *const Handle<L::Ham>,
    x: *const Handle<L::Vector>,
    out: *mut L::T,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x = unsafe { borrow(x)? };
    let n = x.norm();
    if n == 0.0 {
        return Err(Error::invalid("Rayleigh quotient of the zero train"));
    }
    let scale = <L::T as Scalar>::from_f64(1.0 / (n * n));
    unsafe { *ptr_mut(out, "out")? = h.rayleigh_quotient(x) * scale };
    Ok(())
}

/// `⟨x, H·y⟩` (no normalization).
///
/// # Safety
/// `out` must be writable.
pub unsafe fn ham_xthy<L: Layer>(
    x: *const Handle<L::Vector>,
    h: *const Handle<L::Ham>,
    y: *const Handle<L::Vector>,
    out: *mut L::T,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x = unsafe { borrow(x)? };
    let y = unsafe { borrow(y)? };
    unsafe { *ptr_mut(out, "out")? = L::xt_h_y(x, h, y) };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coeff_from_flat_accepts_matching_length() {
        let t = match coeff_from_flat::<f64>(3, 2, &[0.0; 9]) {
            Ok(t) => t,
            Err(e) => panic!("expected Ok, got error code {}: {}", e.code, e.msg),
        };
        assert_eq!(t.data.len(), 9);
    }

    #[test]
    fn coeff_from_flat_reports_length_mismatch() {
        match coeff_from_flat::<f64>(3, 2, &[0.0; 8]) {
            Ok(_) => panic!("expected a length-mismatch error"),
            Err(e) => assert_eq!(e.code, QNTT_INVALID_ARG),
        }
    }

    #[test]
    fn coeff_from_flat_reports_overflow_cleanly_instead_of_wrapping() {
        match coeff_from_flat::<f64>(1 << 20, 4, &[]) {
            Ok(_) => panic!("expected an overflow error"),
            Err(e) => assert_eq!(e.code, QNTT_INVALID_ARG),
        }
    }
}
