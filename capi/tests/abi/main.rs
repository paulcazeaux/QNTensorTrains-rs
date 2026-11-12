// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ffi::CString;
use std::ptr;

use qntt::C64;
use qntt::fcidump::{TwoBodySymmetry, read_fcidump};
use qntt::solvers::mals::NoiseType;
use qntt::{QnSpace, RankGrid, Sector, SparseHamiltonian};
use qntt_capi::err::{QNTT_INVALID_ARG, QNTT_OK, QNTT_PANIC};
use qntt_capi::exports::*;
use qntt_capi::fcidump::*;
use qntt_capi::opts::*;
use qntt_capi::pointgroup::{qntt_pg_molpro_from_irrep, qntt_space_max_ranks_pg};
use qntt_capi::probe::*;
use qntt_capi::{qntt_last_error, qntt_version};

fn h4_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/h4.FCIDUMP")
}

fn last_error() -> String {
    let mut len = 0usize;
    unsafe { qntt_last_error(ptr::null_mut(), 0, &mut len) };
    let mut buf = vec![0u8; len + 1];
    unsafe { qntt_last_error(buf.as_mut_ptr() as *mut i8, buf.len(), &mut len) };
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

fn flat_ranks(space: &QnSpace, grids: &[RankGrid]) -> Vec<usize> {
    let mut out = Vec::new();
    for g in grids {
        for nup in 0..=space.nup {
            for ndn in 0..=space.ndn {
                out.push(g[Sector::new(nup, ndn)]);
            }
        }
    }
    out
}

fn uniform_ranks(space: QnSpace, r: usize) -> Vec<RankGrid> {
    (0..=space.d)
        .map(|b| {
            RankGrid::from_fn(&space, |s| {
                if space.contains(b, s) {
                    (space.max_rank(b, s) as usize).min(r)
                } else {
                    0
                }
            })
        })
        .collect()
}

fn su2_capped_ranks(d: usize, n: usize, two_s: usize, r: usize) -> Vec<usize> {
    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_su2_space_max_ranks(d, n, two_s, ptr::null_mut(), 0, &mut len) },
        QNTT_OK
    );
    let mut flat = vec![0usize; len];
    assert_eq!(
        unsafe { qntt_su2_space_max_ranks(d, n, two_s, flat.as_mut_ptr(), len, &mut len) },
        QNTT_OK
    );
    for v in &mut flat {
        *v = (*v).min(r);
    }
    flat
}

fn read_h4_through_abi() -> (usize, usize, i64, Vec<f64>, Vec<f64>) {
    let path = CString::new(h4_path().to_str().unwrap()).unwrap();
    let mut data: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_fcidump_read(path.as_ptr(), 2, &mut data) },
        QNTT_OK,
        "fcidump read failed: {}",
        last_error()
    );

    let (mut norb, mut nelec, mut ms2, mut e0) = (0usize, 0usize, 0i64, 0.0f64);
    assert_eq!(
        unsafe { qntt_f64_fcidump_info(data, &mut norb, &mut nelec, &mut ms2, &mut e0) },
        QNTT_OK
    );

    let mut len = 0usize;
    unsafe { qntt_f64_fcidump_t(data, ptr::null_mut(), 0, &mut len) };
    let mut t = vec![0.0f64; len];
    assert_eq!(
        unsafe { qntt_f64_fcidump_t(data, t.as_mut_ptr(), t.len(), &mut len) },
        QNTT_OK
    );

    unsafe { qntt_f64_fcidump_v(data, ptr::null_mut(), 0, &mut len) };
    let mut v = vec![0.0f64; len];
    assert_eq!(
        unsafe { qntt_f64_fcidump_v(data, v.as_mut_ptr(), v.len(), &mut len) },
        QNTT_OK
    );

    unsafe { qntt_f64_fcidump_free(data) };
    (norb, nelec, ms2, t, v)
}

fn h4_ham_and_ground_state(
    r: usize,
) -> (
    *mut qntt_capi::handle::Handle<SparseHamiltonian<f64>>,
    *mut qntt_capi::handle::Handle<qntt::TtVector<f64>>,
) {
    let (norb, nelec, ms2, t, v) = read_h4_through_abi();
    let space = QnSpace::from_n_two_sz(norb, nelec, ms2);
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new(
                space.d,
                space.nup,
                space.ndn,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "ham_new failed: {}",
        last_error()
    );
    let flat = flat_ranks(&space, &uniform_ranks(space, r));
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                7,
                &mut x0,
            )
        },
        QNTT_OK
    );
    let mut opts = QnttAlsOptions::defaults();
    opts.eps = 1e-10;
    opts.max_iter = 50;
    let mut lambda = 0.0f64;
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_als(h, x0, &opts, &mut lambda, &mut x) },
        QNTT_OK,
        "als failed: {}",
        last_error()
    );
    unsafe { qntt_f64_vec_free(x0) };
    (h, x)
}

const PG_C2V: i32 = 5;
const READOUT_TDA: i32 = 0;
const READOUT_RPA: i32 = 1;
const METRIC_II: i32 = 0;
const METRIC_RR: i32 = 1;
const METRIC_AUTO: i32 = 2;

const D: usize = 4;
const NUP: usize = 2;
const NDN: usize = 1;
const SITE_IRREPS: [i32; D] = [0, 1, 2, 1];

fn orbsym() -> Vec<u8> {
    SITE_IRREPS
        .iter()
        .map(|&ir| {
            let mut id = 0i32;
            assert_eq!(
                unsafe { qntt_pg_molpro_from_irrep(PG_C2V, ir, &mut id) },
                QNTT_OK,
                "{}",
                last_error()
            );
            u8::try_from(id).expect("a Molpro id fits a byte")
        })
        .collect()
}

fn fx4_coeffs() -> (Vec<f64>, Vec<f64>) {
    let mut t = vec![0.0f64; D * D];
    let mut v = vec![0.0f64; D * D * D * D];
    let ti = |i: usize, j: usize| i * D + j;
    let vi = |i: usize, j: usize, k: usize, l: usize| ((i * D + j) * D + k) * D + l;
    for p in 0..D {
        t[ti(p, p)] = 0.17 * (p as f64 - 0.5 * D as f64);
    }
    t[ti(1, 3)] = -0.85;
    t[ti(3, 1)] = -0.85;
    for p in 0..D {
        for q in 0..D {
            if p != q {
                let w = 1.0 + (p as f64 - q as f64).abs();
                v[vi(p, p, q, q)] = 0.5 / w;
                v[vi(p, q, q, p)] = 0.23 / w;
            }
        }
    }
    for idx in [[1, 3, 0, 0], [3, 1, 0, 0], [0, 0, 1, 3], [0, 0, 3, 1]] {
        v[vi(idx[0], idx[1], idx[2], idx[3])] = 0.19;
    }
    (t, v)
}

fn capped_ranks_pg(target: i32, r: usize) -> Vec<usize> {
    let ob = orbsym();
    let mut len = 0usize;
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                D,
                NUP,
                NDN,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                target,
                ptr::null_mut(),
                0,
                &mut len,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut flat = vec![0usize; len];
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                D,
                NUP,
                NDN,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                target,
                flat.as_mut_ptr(),
                len,
                &mut len,
            )
        },
        QNTT_OK
    );
    for x in &mut flat {
        *x = (*x).min(r);
    }
    flat
}

fn ham_at_target(target: i32) -> *mut qntt_capi::handle::Handle<SparseHamiltonian<f64>> {
    let (t, v) = fx4_coeffs();
    let ob = orbsym();
    let mut h: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new_pg(
                D,
                NUP,
                NDN,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                target,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "ham_new_pg at target {target} failed: {}",
        last_error()
    );
    h
}

fn reference() -> (
    *mut qntt_capi::handle::Handle<SparseHamiltonian<f64>>,
    *mut qntt_capi::handle::Handle<qntt::TtVector<f64>>,
) {
    let h = ham_at_target(0);
    let ob = orbsym();
    let flat = capped_ranks_pg(0, 2);
    let mut x0: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn_pg(
                D,
                NUP,
                NDN,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                flat.as_ptr(),
                flat.len(),
                0x7505,
                &mut x0,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut opts = QnttAlsOptions::defaults();
    opts.eps = 1e-12;
    opts.max_iter = 60;
    let mut lambda = 0.0f64;
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_als(h, x0, &opts, &mut lambda, &mut x) },
        QNTT_OK,
        "als failed: {}",
        last_error()
    );
    unsafe { qntt_f64_vec_free(x0) };
    (h, x)
}

mod diagnostics;
mod end_to_end;
mod handles;
mod joint_gad;
mod layers;
mod marshalling;
mod multistate;
mod options_cache;
mod pd_probe;
mod rpa_signed_c64;
mod sector_observables;
mod sector_rpa;
mod tda;
