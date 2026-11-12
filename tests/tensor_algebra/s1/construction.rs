// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s1::util::*;
use crate::s1::*;

#[test]
fn s1_8_tt_csf_endpoint_typing_reformulated() {
    let x: Su2TtVector<f64> = tt_csf(&[
        Su2Family::Up,
        Su2Family::Dn,
        Su2Family::Up,
        Su2Family::Empty,
    ]);
    assert_eq!(x.n_electrons(), 3);
    assert_eq!(x.two_s(), 1);
    assert_eq!(x.d(), 4);
}

#[test]
fn s1_10_tt_csf_exactly_normalized() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5F0_C5F0);
    for _ in 0..5 {
        let x: Su2TtVector<f64> = random_csf(7, 5, 1, &mut rng);
        assert!(x.check());
        assert_eq!(x.norm(), 1.0);
    }
}
