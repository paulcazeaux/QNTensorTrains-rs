// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::Mat4;

pub(super) fn descending(two_s: i32) -> impl Iterator<Item = (usize, i32)> {
    (-two_s..=two_s).rev().step_by(2).enumerate()
}

pub(super) fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut o = [[0.0f64; 4]; 4];
    for (i, orow) in o.iter_mut().enumerate() {
        for k in 0..4 {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for (j, ov) in orow.iter_mut().enumerate() {
                *ov += aik * b[k][j];
            }
        }
    }
    o
}

pub(super) fn mat4_neg(a: &Mat4) -> Mat4 {
    let mut o = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            o[i][j] = -a[i][j];
        }
    }
    o
}

pub(super) fn frob(a: &Mat4) -> f64 {
    a.iter()
        .flat_map(|r| r.iter())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

#[inline]
pub(super) fn pow4(p: usize) -> usize {
    1usize << (2 * p)
}
