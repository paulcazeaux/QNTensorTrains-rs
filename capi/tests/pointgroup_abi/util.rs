// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ptr;

use qntt::{CoeffTensor, Irrep, PointGroup, PointGroupKind};
use qntt_capi::pointgroup::qntt_space_check_pg;
use qntt_capi::qntt_last_error;

pub(crate) fn c2v_orbsym(d: usize) -> Vec<u8> {
    (0..d).map(|k| [2u8, 3, 4, 1][k % 4]).collect()
}

pub(crate) fn c2v_pg(d: usize) -> PointGroup {
    PointGroup::from_orbsym(PointGroupKind::C2v, &c2v_orbsym(d)).expect("valid orbsym table")
}

pub(crate) fn last_error() -> String {
    let mut len = 0usize;
    unsafe { qntt_last_error(ptr::null_mut(), 0, &mut len) };
    let mut buf = vec![0u8; len + 1];
    unsafe {
        qntt_last_error(
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            buf.len(),
            &mut len,
        )
    };
    String::from_utf8_lossy(&buf[..len]).into_owned()
}

fn pseudo(key: u64) -> f64 {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

pub(crate) fn c2v_coeffs(pg: &PointGroup, d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let mut t_set = vec![false; d * d];
    for i in 0..d {
        for j in 0..d {
            if t_set[i * d + j] {
                continue;
            }
            let val = if pg.irrep_of(i) == pg.irrep_of(j) {
                pseudo((i * d + j) as u64)
            } else {
                0.0
            };
            for &(a, b) in &[(i, j), (j, i)] {
                t.set(&[a, b], val);
                t_set[a * d + b] = true;
            }
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    let mut v_set = vec![false; d * d * d * d];
    let lin = |a: usize, b: usize, c: usize, e: usize| ((a * d + b) * d + c) * d + e;
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if v_set[lin(i, j, k, l)] {
                        continue;
                    }
                    let allowed = pg
                        .irrep_of(i)
                        .fuse(pg.irrep_of(j))
                        .fuse(pg.irrep_of(k))
                        .fuse(pg.irrep_of(l))
                        == Irrep::IDENTITY;
                    let val = if allowed {
                        0.25 * pseudo(0x5EED_0000 + lin(i, j, k, l) as u64)
                    } else {
                        0.0
                    };
                    for &(a, b, c, e) in &[
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    ] {
                        v.set(&[a, b, c, e], val);
                        v_set[lin(a, b, c, e)] = true;
                    }
                }
            }
        }
    }
    (t, v)
}

pub(crate) fn check_pg(d: usize, a: usize, b: usize, kind: i32, orbsym: &[u8], target: i32) -> i32 {
    unsafe { qntt_space_check_pg(d, a, b, kind, orbsym.as_ptr(), orbsym.len(), target) }
}
