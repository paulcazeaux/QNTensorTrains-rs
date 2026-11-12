// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg2::*;

#[test]
fn trivial_group_structural_invariants_pin() {
    let d = 6;
    let occ_up = [true, false, true, false, false, false];
    let occ_dn = [false, true, false, false, true, false];
    let x: TtVector<f64> = tt_state(&occ_up, &occ_dn);
    assert!(x.check());

    let dense = x.to_dense();
    assert_eq!(
        dense.len(),
        1usize << (2 * d),
        "to_dense length must be 4^d"
    );

    let expected_idx: usize = (0..d)
        .map(|k| {
            let digit = match (occ_up[k], occ_dn[k]) {
                (false, false) => 0usize,
                (true, false) => 1,
                (false, true) => 2,
                (true, true) => 3,
            };
            digit << (2 * k)
        })
        .sum();
    let nonzero: Vec<usize> = dense
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v != 0.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        nonzero,
        vec![expected_idx],
        "tt_state at |G|=1 must remain a single-determinant indicator"
    );
}

#[test]
fn trivial_group_tt_ones_and_tt_randn_check_pass() {
    let d = 6;
    let space = QnSpace::from_n_two_sz(d, 5, 1);
    let ones: TtVector<f64> = tt_ones(space);
    assert!(ones.check());

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(9001);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(1..=5)
                };
            }
            r
        })
        .collect();
    let rnd: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(rnd.check());
}
