// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use crate::round::RoundMode;
use crate::su2::ops::block_tests::su2_open_block_with;
use crate::su2::qn::{Su2Space, max_ranks, max_ranks_open};
use crate::su2::vector::Su2TtVector;

fn generic_block(space: Su2Space, rho: usize) -> Su2TtVector<f64> {
    su2_open_block_with(space, rho, 5, || 60)
}

fn open_caps_are_the_revealed_ranks(space: Su2Space, rho: usize) {
    let x = generic_block(space, rho);
    let y = x.round(1e-10, crate::Pooled::NONE, RoundMode::Rel);
    let caps = max_ranks_open(&space, rho);
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert_eq!(
                caps[b][s], y.ranks[b][s],
                "bond {b} sector {s:?} at ρ = {rho}"
            );
        }
    }
}

#[test]
fn su2_open_caps_are_the_svd_revealed_ranks() {
    for rho in [1, 2, 4] {
        open_caps_are_the_revealed_ranks(Su2Space::new(5, 3, 1), rho);
    }
    open_caps_are_the_revealed_ranks(Su2Space::new(5, 4, 2), 3);
    open_caps_are_the_revealed_ranks(Su2Space::new(4, 4, 0), 3);
}

#[test]
fn su2_open_caps_reduce_to_max_ranks() {
    for space in [
        Su2Space::new(6, 5, 1),
        Su2Space::new(6, 4, 2),
        Su2Space::new(4, 4, 0),
    ] {
        let closed = max_ranks(&space);
        assert_eq!(max_ranks_open(&space, 1), closed);
        let open = max_ranks_open(&space, 7);
        for b in 0..=space.d {
            for s in space.bond_sectors(b) {
                assert!(open[b][s] >= closed[b][s]);
                assert_eq!(open[b][s] == 0, closed[b][s] == 0);
            }
        }
    }
}
