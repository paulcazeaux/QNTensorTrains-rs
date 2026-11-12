// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! FCIDUMP / TCDUMP readers.

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::path::Path;

use qntt::C64;
use qntt::PointGroupKind;
use qntt::Scalar;
use qntt::Sparse3Body;
use qntt::fcidump::{FcidumpData, read_fcidump, read_tcdump, screen_point_group_forbidden};

use crate::enums;
use crate::err::{CResult, Error, QNTT_IO, catch, copy_out, ptr_mut};
use crate::handle::{Handle, HandleType, borrow, borrow_mut, emit, release};

/// Parsed integrals plus the point-group kind the reader was called with (`None` = guess per
/// order, the legacy reader's behaviour).
pub struct CFcidump<T: Scalar> {
    pub data: FcidumpData<T>,
    pub kind: Option<PointGroupKind>,
}

/// Borrow a NUL-terminated path argument. The one path convention in this crate — the TT state
/// cache (`api::vec_save`/`vec_load`) borrows it from here rather than growing a second one.
///
/// # Safety
/// `p` must be null or point to a NUL-terminated byte string.
pub unsafe fn path_arg<'a>(p: *const c_char, what: &str) -> Result<&'a Path, Error> {
    if p.is_null() {
        return Err(Error::null(what));
    }
    let s = unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| Error::invalid(format!("{what} is not valid UTF-8")))?;
    Ok(Path::new(s))
}

fn io_err(e: qntt::fcidump::FcidumpError) -> Error {
    Error::new(QNTT_IO, e.to_string())
}

unsafe fn read_impl<T: Scalar>(
    path: *const c_char,
    symmetry: i32,
    pg_kind: i32,
    out: *mut *mut Handle<CFcidump<T>>,
) -> CResult
where
    CFcidump<T>: HandleType,
{
    let p = unsafe { path_arg(path, "path")? };
    let sym = enums::two_body_symmetry(symmetry)?;
    let kind = enums::pg_kind_opt(pg_kind)?;
    let data = read_fcidump::<T>(p, sym).map_err(io_err)?;
    if kind.is_some() {
        data.point_group(kind).map_err(io_err)?;
    }
    unsafe { emit(CFcidump { data, kind }, out) }
}

macro_rules! export_fcidump {
    ($sfx:ident, $T:ty) => {
        paste::paste! {
            /// Release a parsed-integrals handle (no-op on `NULL`).
            ///
            /// # Safety
            /// `h` must be null or a live handle not already released.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_free>](h: *mut Handle<CFcidump<$T>>) {
                unsafe { release(h) }
            }

            /// Parse an FCIDUMP file. `symmetry`: 0 = none, 1 = pair-swap, 2 = eightfold.
            /// `ORBSYM`/`ISYM` are parsed and kept, but no group is forced — this is exactly
            /// `qntt_*_fcidump_read_pg(path, symmetry, QNTT_PG_AUTO)`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_read>](
                path: *const c_char, symmetry: i32,
                out: *mut *mut Handle<CFcidump<$T>>,
            ) -> c_int {
                catch(|| unsafe { read_impl::<$T>(path, symmetry, -1, out) })
            }

            /// Parse an FCIDUMP file, interpreting `ORBSYM`/`ISYM` in `pg_kind`
            /// (`QNTT_PG_C1`..`QNTT_PG_D2H`, or `QNTT_PG_AUTO` to guess from `max(ORBSYM)`).
            /// The integrals are byte-identical to `qntt_*_fcidump_read`'s on the same file;
            /// only the kind remembered for `qntt_*_fcidump_point_group` differs.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_read_pg>](
                path: *const c_char, symmetry: i32, pg_kind: i32,
                out: *mut *mut Handle<CFcidump<$T>>,
            ) -> c_int {
                catch(|| unsafe { read_impl::<$T>(path, symmetry, pg_kind, out) })
            }

            /// Orbital count, electron count and `2·Sz` of the parsed dump.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_info>](
                h: *const Handle<CFcidump<$T>>,
                out_norb: *mut usize, out_nelec: *mut usize, out_ms2: *mut i64,
                out_e0: *mut $T,
            ) -> c_int {
                catch(|| {
                    let d = unsafe { borrow(h)? };
                    unsafe {
                        *ptr_mut(out_norb, "out_norb")? = d.data.norb;
                        *ptr_mut(out_nelec, "out_nelec")? = d.data.nelec;
                        *ptr_mut(out_ms2, "out_ms2")? = d.data.ms2;
                        *ptr_mut(out_e0, "out_e0")? = d.data.e0;
                    }
                    Ok(())
                })
            }

            /// Copy the one-body integrals out (flat row-major `norb²`).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_t>](
                h: *const Handle<CFcidump<$T>>,
                out: *mut $T, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let d = unsafe { borrow(h)? };
                    unsafe { copy_out(&d.data.t.data, out, cap, out_len, "t") }
                })
            }

            /// Copy the two-body integrals out (flat row-major `norb⁴`, chemists' `(ij|kl)`).
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_v>](
                h: *const Handle<CFcidump<$T>>,
                out: *mut $T, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let d = unsafe { borrow(h)? };
                    unsafe { copy_out(&d.data.v.data, out, cap, out_len, "v") }
                })
            }

            /// Copy the header's raw 1-based Molpro `ORBSYM` ids out, one per orbital, in file
            /// (site) order. `out_len == 0` when the header carried no `ORBSYM` at all — that is
            /// "no point-group information", equivalent to all-ones for every purpose here.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_orbsym>](
                h: *const Handle<CFcidump<$T>>,
                out: *mut u8, cap: usize, out_len: *mut usize,
            ) -> c_int {
                catch(|| {
                    let d = unsafe { borrow(h)? };
                    let ids: Result<Vec<u8>, Error> = d.data.orbsym.iter().enumerate()
                        .map(|(k, &id)| u8::try_from(id).map_err(|_| Error::invalid(format!(
                            "ORBSYM entry {id} at orbital {k} does not fit in a byte"
                        ))))
                        .collect();
                    unsafe { copy_out(&ids?, out, cap, out_len, "orbsym") }
                })
            }

            /// The dump's **resolved** point group: `out_kind` is the wire code of the kind the
            /// ids were interpreted in (explicit, or the per-order guess — never `QNTT_PG_AUTO`),
            /// `out_isym` the RAW `ISYM` as parsed (`0` when absent or literally `ISYM=0`), and
            /// `out_target` the 0-based XOR target irrep. Feed `out_kind`, the `ORBSYM` ids and
            /// `out_target` straight into any `_pg` entry point's descriptor.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_point_group>](
                h: *const Handle<CFcidump<$T>>,
                out_kind: *mut i32, out_isym: *mut i64, out_target: *mut i32,
            ) -> c_int {
                catch(|| {
                    let d = unsafe { borrow(h)? };
                    let pg = d.data.point_group(d.kind).map_err(io_err)?;
                    unsafe {
                        *ptr_mut(out_kind, "out_kind")? = enums::pg_kind_code(pg.kind);
                        *ptr_mut(out_isym, "out_isym")? = d.data.isym.unwrap_or(0);
                        *ptr_mut(out_target, "out_target")? = i32::from(pg.target.index());
                    }
                    Ok(())
                })
            }

            /// Project out (or reject) the integrals the dump's own `ORBSYM` forbids, IN PLACE
            /// `pg_kind` is `QNTT_PG_AUTO` or an explicit kind, as in
            /// `qntt_*_fcidump_read_pg`.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _fcidump_screen_pg>](
                h: *mut Handle<CFcidump<$T>>, pg_kind: i32,
            ) -> c_int {
                catch(|| {
                    let kind = enums::pg_kind_opt(pg_kind)?;
                    let d = unsafe { borrow_mut(h)? };
                    let pg = d.data.point_group(kind).map_err(io_err)?;
                    screen_point_group_forbidden(&mut d.data.t, &mut d.data.v, pg.pg)
                        .map_err(io_err)?;
                    Ok(())
                })
            }

            /// Parse a TCDUMP three-body file into a sparse coefficient handle.
            /// `symmetry`: 0 = none, 1 = pair-permutation, 2 = full 48-fold.
            ///
            /// # Safety
            /// See the module contract in `include/qntt.h`.
            #[unsafe(no_mangle)]
            pub unsafe extern "C" fn [<qntt_ $sfx _tcdump_read>](
                path: *const c_char, norb: usize, symmetry: i32,
                out: *mut *mut Handle<Sparse3Body<$T>>,
            ) -> c_int {
                catch(|| {
                    let p = unsafe { path_arg(path, "path")? };
                    let sym = enums::three_body_symmetry(symmetry)?;
                    let w = read_tcdump::<$T>(p, norb, sym).map_err(io_err)?;
                    unsafe { emit(w, out) }
                })
            }
        }
    };
}

export_fcidump!(f64, f64);
export_fcidump!(c64, C64);
