// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::solvers::TtState;

use crate::enums;
use crate::err::{CResult, Error, copy_out, ptr_mut, ptr_slice, ptr_slice_mut};
use crate::handle::{Handle, borrow, borrow_mut, emit};
use crate::layer::Layer;

use super::ranks::{ranks_from_flat, ranks_to_flat, refuse_point_group};

/// # Safety
/// `out` must be a valid writable handle slot.
pub unsafe fn vec_zeros<L: Layer>(
    d: usize,
    a: usize,
    b: usize,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let space = L::space_new(d, a, b)?;
    unsafe { emit(L::vec_zeros(space), out) }
}

/// All-zero blocks at the prescribed per-bond ranks.
///
/// # Safety
/// `ranks` must hold `ranks_len` entries; `out` a valid handle slot.
pub unsafe fn vec_from_ranks<L: Layer>(
    d: usize,
    a: usize,
    b: usize,
    ranks: *const usize,
    ranks_len: usize,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let space = L::space_new(d, a, b)?;
    let flat = unsafe { ptr_slice(ranks, ranks_len, "ranks")? };
    let grids = ranks_from_flat::<L>(&space, flat)?;
    unsafe { emit(L::vec_from_ranks(space, &grids), out) }
}

/// Gaussian random train at the prescribed ranks, from a `seed`-determined Xoshiro256++ stream.
///
/// # Safety
/// As [`vec_from_ranks`].
pub unsafe fn vec_randn<L: Layer>(
    d: usize,
    a: usize,
    b: usize,
    ranks: *const usize,
    ranks_len: usize,
    seed: u64,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let space = L::space_new(d, a, b)?;
    let flat = unsafe { ptr_slice(ranks, ranks_len, "ranks")? };
    let grids = ranks_from_flat::<L>(&space, flat)?;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    unsafe { emit(L::vec_randn(space, &grids, &mut rng), out) }
}

/// # Safety
/// `x` must be a live vector handle; `out` a valid handle slot.
pub unsafe fn vec_clone<L: Layer>(
    x: *const Handle<L::Vector>,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let x = unsafe { borrow(x)? };
    unsafe { emit(x.clone(), out) }
}

/// `(d, a, b)` of the train's space.
///
/// # Safety
/// Out-pointers must be null or writable.
pub unsafe fn vec_space<L: Layer>(
    x: *const Handle<L::Vector>,
    out_d: *mut usize,
    out_a: *mut usize,
    out_b: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let (d, a, b) = L::space_parts(&L::vec_space(x));
    unsafe {
        *ptr_mut(out_d, "out_d")? = d;
        *ptr_mut(out_a, "out_a")? = a;
        *ptr_mut(out_b, "out_b")? = b;
    }
    Ok(())
}

/// The train's per-bond rank grids, flattened.
///
/// # Safety
/// `out` must hold `cap` entries or be null when `cap == 0`.
pub unsafe fn vec_ranks<L: Layer>(
    x: *const Handle<L::Vector>,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let space = L::vec_space(x);
    refuse_point_group::<L>(&space, "qntt_<scalar>_vec_ranks_pg")?;
    let flat = ranks_to_flat::<L>(&space, L::vec_ranks(x));
    unsafe { copy_out(&flat, out, cap, out_len, "ranks") }
}

/// Orthogonality centre, or `-1` when the train is not in canonical form.
///
/// # Safety
/// `out` must be writable.
pub unsafe fn vec_ortho<L: Layer>(x: *const Handle<L::Vector>, out: *mut isize) -> CResult {
    let x = unsafe { borrow(x)? };
    unsafe { *ptr_mut(out, "out")? = L::vec_ortho(x).map_or(-1, |k| k as isize) };
    Ok(())
}

/// Structural self-check (rank/sector consistency across cores).
///
/// # Safety
/// `out` must be writable.
pub unsafe fn vec_check<L: Layer>(x: *const Handle<L::Vector>, out: *mut i32) -> CResult {
    let x = unsafe { borrow(x)? };
    unsafe { *ptr_mut(out, "out")? = L::vec_check(x) as i32 };
    Ok(())
}

/// Structural sizes: number of sites, largest single-sector rank, largest summed bond dimension.
///
/// # Safety
/// Out-pointers must be null or writable.
pub unsafe fn vec_sizes<L: Layer>(
    x: *const Handle<L::Vector>,
    out_sites: *mut usize,
    out_max_sector_rank: *mut usize,
    out_max_bond_rank: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    unsafe {
        *ptr_mut(out_sites, "out_sites")? = x.num_sites();
        *ptr_mut(out_max_sector_rank, "out_max_sector_rank")? = x.max_sector_rank();
        *ptr_mut(out_max_bond_rank, "out_max_bond_rank")? = x.max_total_bond_rank();
    }
    Ok(())
}

/// `(x, y)` — the BILINEAR, unconjugated contraction, as opposed to [`vec_dot`]'s Hermitian one.
///
/// # Safety
/// `out` must be writable.
pub unsafe fn vec_dot_bilinear<L: Layer>(
    x: *const Handle<L::Vector>,
    y: *const Handle<L::Vector>,
    out: *mut L::T,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let y = unsafe { borrow(y)? };
    let mut xb = x.clone();
    L::vec_conj_mut(&mut xb);
    unsafe { *ptr_mut(out, "out")? = xb.dot(y) };
    Ok(())
}

/// Schmidt (singular) values at every INTERIOR bond `1..d`, pooled over each bond's sectors and
/// sorted descending, together with each value's multiplet WEIGHT: `out` receives the values
/// concatenated bond by bond, `out_weights` the parallel weights (exactly `*out_len` of them),
/// `out_counts` how many belong to each bond — one entry per interior bond, `d - 1` of them.
///
/// # Safety
/// Each output must hold its stated capacity in elements, or be null when that capacity is zero.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_schmidt<L: Layer>(
    x: *const Handle<L::Vector>,
    out: *mut f64,
    cap: usize,
    out_len: *mut usize,
    out_weights: *mut f64,
    weights_cap: usize,
    out_counts: *mut usize,
    counts_cap: usize,
    out_nbonds: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let per_bond = L::bond_schmidt(x);
    let counts: Vec<usize> = per_bond.iter().map(|v| v.len()).collect();
    let (flat, weights): (Vec<f64>, Vec<f64>) = per_bond.into_iter().flatten().unzip();
    unsafe {
        copy_out(
            &counts,
            out_counts,
            counts_cap,
            out_nbonds,
            "schmidt counts",
        )?;
        copy_out(
            &weights,
            out_weights,
            weights_cap,
            std::ptr::null_mut(),
            "schmidt weights",
        )?;
        copy_out(&flat, out, cap, out_len, "schmidt values")
    }
}

/// `||P_T(H x - lambda x)||` with `lambda` the Rayleigh quotient at `x` — the Riemannian gradient
/// norm, i.e. how critical the state is. The RPA reference gate is `< 1e-9`.
///
/// # Safety
/// `out` must be writable.
pub unsafe fn grad_norm<L: Layer>(
    h: *const Handle<L::Ham>,
    x: *const Handle<L::Vector>,
    out: *mut f64,
) -> CResult {
    let h = unsafe { borrow(h)? };
    let x = unsafe { borrow(x)? };
    unsafe { *ptr_mut(out, "out")? = L::grad_norm(h, x)? };
    Ok(())
}

fn tt_io_err(e: qntt::tt_io::TtIoError) -> Error {
    Error::new(crate::err::QNTT_IO, e.to_string())
}

/// Write the train to `path`, truncating.
///
/// # Safety
/// `x` must be a live vector handle; `path` a NUL-terminated string.
pub unsafe fn vec_save<L: Layer>(
    x: *const Handle<L::Vector>,
    path: *const std::os::raw::c_char,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let p = unsafe { crate::fcidump::path_arg(path, "path")? };
    L::vec_save(x, p).map_err(tt_io_err)
}

/// Read a train written by [`vec_save`] into a fresh handle.
///
/// # Safety
/// `path` must be a NUL-terminated string; `out` a valid handle slot.
pub unsafe fn vec_load<L: Layer>(
    path: *const std::os::raw::c_char,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let p = unsafe { crate::fcidump::path_arg(path, "path")? };
    let x = L::vec_load(p).map_err(tt_io_err)?;
    unsafe { emit(x, out) }
}

/// Copy one block out, column-major (faer's own layout, which is also 's).
///
/// # Safety
/// `out` must hold `cap` scalars or be null when `cap == 0` (size query).
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_block_get<L: Layer>(
    x: *const Handle<L::Vector>,
    site: usize,
    family: i32,
    sa: usize,
    sb: usize,
    out: *mut L::T,
    cap: usize,
    out_rows: *mut usize,
    out_cols: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    refuse_point_group::<L>(&L::vec_space(x), "qntt_<scalar>_vec_block_get_pg")?;
    let f = L::family(family)?;
    let d = x.num_sites();
    if site >= d {
        return Err(Error::invalid(format!("site {site} out of range 0..{d}")));
    }
    let s = L::sector_new(sa, sb);
    let (rows, cols) = L::vec_block_dims(x, site, f, s).unwrap_or((0, 0));
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
    L::vec_block_copy_out(x, site, f, s, dst);
    Ok(())
}

/// Overwrite one block from a column-major buffer. Requires a uniquely-owned handle.
///
/// # Safety
/// `src` must hold `rows·cols` scalars.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_block_set<L: Layer>(
    x: *mut Handle<L::Vector>,
    site: usize,
    family: i32,
    sa: usize,
    sb: usize,
    src: *const L::T,
    rows: usize,
    cols: usize,
) -> CResult {
    let f = L::family(family)?;
    let xv = unsafe { borrow_mut(x)? };
    refuse_point_group::<L>(&L::vec_space(xv), "qntt_<scalar>_vec_block_set_pg")?;
    let d = xv.num_sites();
    if site >= d {
        return Err(Error::invalid(format!("site {site} out of range 0..{d}")));
    }
    let src = unsafe { ptr_slice(src, rows * cols, "block src")? };
    let s = L::sector_new(sa, sb);
    let Some((br, bc)) = L::vec_block_dims(xv, site, f, s) else {
        return Err(Error::invalid(format!(
            "no block at site {site}, family {family}, sector ({sa}, {sb})"
        )));
    };
    if br != rows || bc != cols {
        return Err(Error::invalid(format!(
            "block is {br}×{bc}, source is {rows}×{cols}"
        )));
    }
    L::vec_block_copy_in(xv, site, f, s, src);
    Ok(())
}

/// # Safety
/// `out` must be writable.
pub unsafe fn vec_norm<L: Layer>(x: *const Handle<L::Vector>, out: *mut f64) -> CResult {
    let x = unsafe { borrow(x)? };
    unsafe { *ptr_mut(out, "out")? = x.norm() };
    Ok(())
}

/// `⟨x, y⟩` in the layer metric, conjugating `x`.
///
/// # Safety
/// `out` must be writable.
pub unsafe fn vec_dot<L: Layer>(
    x: *const Handle<L::Vector>,
    y: *const Handle<L::Vector>,
    out: *mut L::T,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let y = unsafe { borrow(y)? };
    unsafe { *ptr_mut(out, "out")? = x.dot(y) };
    Ok(())
}

/// # Safety
/// `a` must be readable; `out` a valid handle slot.
pub unsafe fn vec_scaled<L: Layer>(
    x: *const Handle<L::Vector>,
    a: *const L::T,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let a = *unsafe { crate::err::ptr_ref(a, "a")? };
    unsafe { emit(x.scaled(a), out) }
}

/// `x + y` (`sign > 0`) or `x - y` (`sign < 0`), unrounded.
///
/// # Safety
/// `out` must be a valid handle slot.
pub unsafe fn vec_add<L: Layer>(
    x: *const Handle<L::Vector>,
    y: *const Handle<L::Vector>,
    sign: i32,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let y = unsafe { borrow(y)? };
    let z = if sign >= 0 { x.add(y) } else { x.sub(y) };
    unsafe { emit(z, out) }
}

/// Deterministic SVD rounding (`chi = 0` uncapped), returning a new train. `chi` is a pooled
/// total bond-dimension cap (SU(2): reduced multiplets).
///
/// # Safety
/// `out` must be a valid handle slot.
pub unsafe fn vec_round<L: Layer>(
    x: *const Handle<L::Vector>,
    eps: f64,
    chi: usize,
    mode: i32,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let m = enums::round_mode(mode)?;
    unsafe { emit(x.rounded(eps, qntt::Pooled(chi), m), out) }
}

/// In-place deterministic SVD rounding. Requires a uniquely-owned handle.
///
/// # Safety
/// `x` must be a live, unshared vector handle.
pub unsafe fn vec_round_mut<L: Layer>(
    x: *mut Handle<L::Vector>,
    eps: f64,
    chi: usize,
    mode: i32,
) -> CResult {
    let m = enums::round_mode(mode)?;
    let x = unsafe { borrow_mut(x)? };
    x.round_mut(eps, qntt::Pooled(chi), m);
    Ok(())
}

/// Left-orthonormalization sweep; writes the product of the per-bond norm factors.
///
/// # Safety
/// `x` must be unshared; `out_norm` null or writable.
pub unsafe fn vec_left_orthonormalize<L: Layer>(
    x: *mut Handle<L::Vector>,
    keep_rank: i32,
    out_norm: *mut f64,
) -> CResult {
    let x = unsafe { borrow_mut(x)? };
    let factors = x.left_orthonormalize(keep_rank != 0);
    if !out_norm.is_null() {
        unsafe { *out_norm = factors.iter().product::<f64>() };
    }
    Ok(())
}

pub(crate) unsafe fn borrow_many<'a, L: Layer>(
    vecs: *const *const Handle<L::Vector>,
    n: usize,
) -> Result<Vec<&'a L::Vector>, Error> {
    let hs = unsafe { ptr_slice(vecs, n, "vectors")? };
    hs.iter().map(|&p| unsafe { borrow(p) }).collect()
}

/// `round(Σ αᵢ xᵢ, tol)` — the deterministic weighted sum.
///
/// # Safety
/// `alpha` must hold `n` scalars and `vecs` `n` live handles.
pub unsafe fn vec_round_sum<L: Layer>(
    alpha: *const L::T,
    vecs: *const *const Handle<L::Vector>,
    n: usize,
    tol: f64,
    out: *mut *mut Handle<L::Vector>,
) -> CResult {
    if n == 0 {
        return Err(Error::invalid("round_sum needs at least one summand"));
    }
    let alpha = unsafe { ptr_slice(alpha, n, "alpha")? };
    let refs = unsafe { borrow_many::<L>(vecs, n)? };
    let owned: Vec<L::Vector> = refs.into_iter().cloned().collect();
    unsafe { emit(L::Vector::round_sum(alpha, &owned, tol), out) }
}
