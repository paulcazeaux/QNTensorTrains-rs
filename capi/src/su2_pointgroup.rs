// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The SU(2) point-group surface: the `_pg` space queries and the generic bodies behind
//! every `qntt_su2_<scalar>_*_pg` shim in [`crate::exports`].

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::su2::core::{Family, Su2Core};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use qntt::su2::vector::Su2TtVector;
use qntt::{CoeffTensor, Irrep, PointGroup, Sparse3Body, ThreeBody};

use crate::enums;
use crate::err::{CResult, Error, copy_out, ptr_mut, ptr_slice, ptr_slice_mut};
use crate::handle::{Handle, ScalarTag, borrow, borrow_mut, emit};
use crate::pointgroup::pg_table;

/// The group tables are layer-independent and shared verbatim with the abelian family — never
/// duplicated. Re-exported here so a caller working only on the SU(2) layer needs a single import.
pub use crate::pointgroup::{
    qntt_pg_default_kind, qntt_pg_irrep_from_molpro, qntt_pg_irrep_name, qntt_pg_kind_from_name,
    qntt_pg_kind_name, qntt_pg_molpro_from_irrep, qntt_pg_orbsym_to_irreps, qntt_pg_order,
};

/// The four scalar-free SU(2) space queries are DEFINED in [`crate::pointgroup`] — not because
/// they belong to the abelian layer, but because `capi/tests/pointgroup_abi.rs`'s
/// `header_declares_every_point_group_symbol` (the only check keeping the hand-maintained
/// `qntt.h` honest) scans exactly `exports.rs`, `pointgroup.rs` and `fcidump.rs` for
/// `#[no_mangle] extern "C" fn`. They are re-exported here so this module is still the single
/// import an SU(2)-only consumer needs.
pub use crate::pointgroup::{
    qntt_su2_space_bond_sectors_pg, qntt_su2_space_check_pg, qntt_su2_space_grid_extent_pg,
    qntt_su2_space_max_ranks_pg,
};

pub(crate) unsafe fn su2_pg_space(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
) -> Result<Su2Space, Error> {
    let kind = enums::pg_kind(pg_kind)?;
    if a > 2 * d {
        return Err(Error::invalid(format!(
            "SU(2) space: need N ({a}) <= 2d ({})",
            2 * d
        )));
    }
    if b > a || !(a - b).is_multiple_of(2) {
        return Err(Error::invalid(format!(
            "SU(2) space: need 2S ({b}) <= N ({a}) with N - 2S even"
        )));
    }
    let (pg, target) = unsafe { pg_table(d, kind, orbsym, orbsym_len, target)? };
    Ok(Su2Space::with_point_group(d, a, b, pg, target))
}

pub(crate) fn ranks_to_flat_pg(space: &Su2Space, grids: &[Su2RankGrid]) -> Vec<usize> {
    let (rows, cols) = (space.n + 1, space.n + 1);
    let nir = space.pg.order() as usize;
    let mut out = Vec::with_capacity(grids.len() * nir * rows * cols);
    for g in grids {
        for ir in 0..nir {
            for i in 0..rows {
                for j in 0..cols {
                    out.push(g[Su2Sector::with_irrep(i, j, Irrep::new(ir as u8))]);
                }
            }
        }
    }
    out
}

pub(crate) fn ranks_from_flat_pg(
    space: &Su2Space,
    flat: &[usize],
) -> Result<Vec<Su2RankGrid>, Error> {
    let (rows, cols) = (space.n + 1, space.n + 1);
    let nir = space.pg.order() as usize;
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
            let mut g = Su2RankGrid::zeros(space);
            let mut p = b * stride;
            for ir in 0..nir {
                for i in 0..rows {
                    for j in 0..cols {
                        let r = flat[p];
                        p += 1;
                        if r != 0 {
                            g[Su2Sector::with_irrep(i, j, Irrep::new(ir as u8))] = r;
                        }
                    }
                }
            }
            g
        })
        .collect())
}

unsafe fn report_point_group(
    space: &Su2Space,
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

fn sector_arg(space: &Su2Space, sa: usize, sb: usize, irrep: u8) -> Result<Su2Sector, Error> {
    if irrep >= space.pg.order() {
        return Err(Error::invalid(format!(
            "irrep {irrep} is out of range 0..{} for this train's point group",
            space.pg.order()
        )));
    }
    Ok(Su2Sector::with_irrep(sa, sb, Irrep::new(irrep)))
}

pub(crate) fn su2_family(v: i32) -> Result<Family, Error> {
    match v {
        0..=3 => Ok(Family::ALL[v as usize]),
        _ => Err(Error::invalid(format!("family {v} (want 0..=3)"))),
    }
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

/// # Safety
/// As [`su2_pg_space`]; `out` must be a valid handle slot.
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_zeros_pg<T: ScalarTag>(
    d: usize,
    a: usize,
    b: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out: *mut *mut Handle<Su2TtVector<T>>,
) -> CResult {
    let space = unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    unsafe { emit(qntt::su2::vector::tt_zeros::<T>(space), out) }
}

/// # Safety
/// As [`su2_pg_space`]; `ranks` must hold `ranks_len` entries.
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
    out: *mut *mut Handle<Su2TtVector<T>>,
) -> CResult {
    let space = unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    let flat = unsafe { ptr_slice(ranks, ranks_len, "ranks")? };
    let grids = ranks_from_flat_pg(&space, flat)?;
    let cores = (0..space.d)
        .map(|k| Su2Core::zeros(space, k, &grids[k], &grids[k + 1]))
        .collect();
    unsafe { emit(Su2TtVector::new(space, cores), out) }
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
    out: *mut *mut Handle<Su2TtVector<T>>,
) -> CResult {
    let space = unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    let flat = unsafe { ptr_slice(ranks, ranks_len, "ranks")? };
    let grids = ranks_from_flat_pg(&space, flat)?;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    unsafe {
        emit(
            qntt::su2::vector::tt_randn::<T, _>(space, &grids, &mut rng),
            out,
        )
    }
}

/// A rank-1 CSF in a point-group space — the SU(2) counterpart of the abelian `tt_state_pg`.
///
/// # Safety
/// `codes` must hold `d` entries; the rest as [`su2_pg_space`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn tt_csf_pg<T: ScalarTag>(
    codes: *const i32,
    d: usize,
    pg_kind: i32,
    orbsym: *const u8,
    orbsym_len: usize,
    target: i32,
    out: *mut *mut Handle<Su2TtVector<T>>,
) -> CResult {
    let raw = unsafe { ptr_slice(codes, d, "codes")? };
    let fams: Vec<Family> = raw
        .iter()
        .map(|&c| su2_family(c))
        .collect::<Result<_, Error>>()?;
    let mut two_s = 0i64;
    for (k, f) in fams.iter().enumerate() {
        two_s += f.delta().1;
        if two_s < 0 {
            return Err(Error::invalid(format!(
                "invalid genealogical path: spin-lowering (dn) step at site {k} from a spin-0 \
                 sector"
            )));
        }
    }
    let kind = enums::pg_kind(pg_kind)?;
    let (pg, want) = unsafe { pg_table(d, kind, orbsym, orbsym_len, target)? };
    let derived = derived_irrep(&fams, pg);
    if derived != want {
        return Err(Error::invalid(format!(
            "the CSF's total irrep {} differs from the requested target {}",
            derived.index(),
            want.index()
        )));
    }
    unsafe {
        emit(
            qntt::su2::vector::tt_csf_with_point_group::<T>(&fams, pg),
            out,
        )
    }
}

fn derived_irrep(fams: &[Family], pg: PointGroup) -> Irrep {
    fams.iter()
        .enumerate()
        .fold(Irrep::IDENTITY, |acc, (k, &f)| {
            acc.fuse(f.flux(pg.irrep_of(k)))
        })
}

/// The train's per-bond rank grids, flattened with the irrep axis.
///
/// # Safety
/// `out` must hold `cap` entries or be null when `cap == 0`.
pub unsafe fn vec_ranks_pg<T: ScalarTag>(
    x: *const Handle<Su2TtVector<T>>,
    out: *mut usize,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let x = unsafe { borrow(x)? };
    let flat = ranks_to_flat_pg(&x.space, &x.ranks);
    unsafe { copy_out(&flat, out, cap, out_len, "ranks") }
}

/// Copy one block out, column-major, addressing the sector by `(n, 2s, irrep)`. An absent or
/// zero-rank block reports `rows = cols = 0` and copies nothing — the common case at `|G| > 1`,
/// where admissible strictly exceeds reachable.
///
/// # Safety
/// `out` must hold `cap` scalars or be null when `cap == 0` (size query).
#[allow(clippy::too_many_arguments)]
pub unsafe fn vec_block_get_pg<T: ScalarTag>(
    x: *const Handle<Su2TtVector<T>>,
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
    let f = su2_family(family)?;
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
    x: *mut Handle<Su2TtVector<T>>,
    site: usize,
    family: i32,
    sa: usize,
    sb: usize,
    irrep: u8,
    src: *const T,
    rows: usize,
    cols: usize,
) -> CResult {
    let f = su2_family(family)?;
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

/// # Safety
/// Out-pointers as described in [`report_point_group`].
pub unsafe fn vec_point_group<T: ScalarTag>(
    x: *const Handle<Su2TtVector<T>>,
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
    h: *const Handle<Su2SparseHamiltonian<T>>,
    out_nbits: *mut i32,
    out_target: *mut i32,
    out_irreps: *mut u8,
    cap: usize,
    out_len: *mut usize,
) -> CResult {
    let h = unsafe { borrow(h)? };
    unsafe { report_point_group(&h.space, out_nbits, out_target, out_irreps, cap, out_len) }
}

/// The SU(2) `ham_new`'s point-group twin: identical marshalling, a point-group space. The SU(2)
/// assembly has a single operator construction and rejects a non-default request (as
/// `crate::layer::Layer::ham_new` for `Su2<T>` does).
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
    out: *mut *mut Handle<Su2SparseHamiltonian<T>>,
) -> CResult {
    let space = unsafe { su2_pg_space(d, a, b, pg_kind, orbsym, orbsym_len, target)? };
    let t = coeff_from_flat(d, 2, unsafe { ptr_slice(t, t_len, "t")? })?;
    let v = coeff_from_flat(d, 4, unsafe { ptr_slice(v, v_len, "v")? })?;
    if enums::construction(construction)?.is_some() {
        return Err(Error::invalid(
            "the SU(2) layer has a single operator construction; pass the default",
        ));
    }
    let w_ref = if w.is_null() {
        None
    } else {
        Some(unsafe { borrow(w)? })
    };
    let w_dyn: Option<&dyn ThreeBody<T>> = w_ref.map(|w| w as &dyn ThreeBody<T>);
    let h = Su2SparseHamiltonian::new(space, &t, &v, w_dyn, eps);
    unsafe { emit(h, out) }
}
