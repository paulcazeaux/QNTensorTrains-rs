// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s1::*;
pub(crate) use qntt_testkit::randomized_su2_state;

pub(crate) fn assert_dense_close(got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "dense length mismatch");
    let err: f64 = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - w) * (g - w))
        .sum::<f64>()
        .sqrt();
    assert!(err <= tol, "dense mismatch: err={err} > tol={tol}");
}

pub(crate) fn random_csf<T: Scalar, R: Rng + ?Sized>(
    d: usize,
    n: usize,
    two_s: usize,
    rng: &mut R,
) -> Su2TtVector<T> {
    let space = Su2Space::new(d, n, two_s);
    let deltas: [(Su2Family, i64, i64); 4] = [
        (Su2Family::Empty, 0, 0),
        (Su2Family::Up, 1, 1),
        (Su2Family::Dn, 1, -1),
        (Su2Family::Occ, 2, 0),
    ];
    let mut codes = Vec::with_capacity(d);
    let (mut cn, mut ct): (i64, i64) = (0, 0);
    for k in 0..d {
        let mut moves = Vec::new();
        for &(f, dn, dt) in &deltas {
            let nn = cn + dn;
            let nt = ct + dt;
            if nt < 0 {
                continue;
            }
            if space.contains(k + 1, Su2Sector::new(nn as usize, nt as usize)) {
                moves.push((f, nn, nt));
            }
        }
        assert!(
            !moves.is_empty(),
            "random_csf: no admissible move at site {k}"
        );
        let idx = rng.random_range(0..moves.len());
        let (f, nn, nt) = moves[idx];
        codes.push(f);
        cn = nn;
        ct = nt;
    }
    tt_csf(&codes)
}
