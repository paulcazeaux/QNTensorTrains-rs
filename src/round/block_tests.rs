// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::TtVector;
use crate::ops::block_tests::open_block_uniform;
use crate::qn::QnSpace;
use crate::round::RoundMode;

#[test]
fn round_error_bound_holds_in_the_block_frobenius_norm() {
    let space = QnSpace::new(6, 3, 2);
    let x: TtVector<f64> = open_block_uniform(space, 6, 3, 17);
    let eps = 0.3;
    let y = x.round(eps, crate::Pooled::NONE, RoundMode::Rel);
    assert!(
        (1..space.d).any(|b| y.ranks[b] != x.ranks[b]),
        "eps = {eps} truncated nothing — the bound is vacuous"
    );
    let err2: f64 = (0..3)
        .map(|j| {
            let (a, b) = (x.column(j).to_dense(), y.column(j).to_dense());
            a.iter()
                .zip(&b)
                .map(|(u, v)| (u - v) * (u - v))
                .sum::<f64>()
        })
        .sum();
    assert!(
        err2.sqrt() <= eps * x.norm(),
        "{} > {}",
        err2.sqrt(),
        eps * x.norm()
    );
}
