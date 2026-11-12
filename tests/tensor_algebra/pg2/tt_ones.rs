// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg2::*;

#[test]
#[allow(clippy::needless_range_loop)]
fn tt_ones_is_target_irrep_indicator_at_nontrivial_group() {
    let d = 4;
    let pg = c2v_table(d);
    let (nup, ndn, target) = (2, 1, Irrep::new(1));
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

    let x: TtVector<f64> = tt_ones(space);
    assert!(x.check(), "tt_ones output must pass check()");
    let dense = x.to_dense();

    let mut n_ones = 0usize;
    let mut n_block = 0usize;
    for (idx, &val) in dense.iter().enumerate() {
        let dig = digits(idx, d);
        let in_particle_block = config_counts(&dig) == (nup, ndn);
        let want = if in_particle_block && config_irrep_from_digits(&pg, &dig) == target {
            1.0
        } else {
            0.0
        };
        assert_eq!(
            val, want,
            "config {idx}: tt_ones must be the EXACT indicator"
        );
        if in_particle_block {
            n_block += 1;
        }
        if want == 1.0 {
            n_ones += 1;
        }
    }
    assert!(
        0 < n_ones && n_ones < n_block,
        "vacuous fixture: indicator is empty or the whole particle block ({n_ones} of {n_block})"
    );

    let caps = max_ranks(&space);
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            assert_eq!(
                x.rank(b, s),
                usize::from(caps[b][s] > 0),
                "bond {b} sector {s:?}: tt_ones rank must be 1 iff reachable, 0 otherwise"
            );
        }
    }
}
