// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::QnSpace;
use qntt::qn::max_ranks;

fn assert_max_ranks_matches_binomial(space: &QnSpace) {
    let caps = max_ranks(space);
    assert_eq!(caps.len(), space.d + 1, "max_ranks must return d+1 grids");
    for (b, caps_b) in caps.iter().enumerate() {
        for s in space.bond_sectors(b) {
            assert_eq!(
                caps_b[s] as u64,
                space.max_rank(b, s),
                "d={} Nup={} Ndn={} bond={b} sector={s:?}: DP cap != binomial max_rank",
                space.d,
                space.nup,
                space.ndn
            );
        }
    }
}

#[test]
fn max_ranks_matches_binomial_max_rank_boundary_cases() {
    for d in [1usize, 2, 6, 8, 12] {
        for (nup, ndn) in [(0, 0), (d, d), (d, 0), (0, d)] {
            assert_max_ranks_matches_binomial(&QnSpace::new(d, nup, ndn));
        }
    }
    assert_max_ranks_matches_binomial(&QnSpace::new(12, 7, 3));
    assert_max_ranks_matches_binomial(&QnSpace::new(12, 2, 9));
    assert_max_ranks_matches_binomial(&QnSpace::new(10, 10, 0));
    assert_max_ranks_matches_binomial(&QnSpace::new(10, 0, 10));
}
