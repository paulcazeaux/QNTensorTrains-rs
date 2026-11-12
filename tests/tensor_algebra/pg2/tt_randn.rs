// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg2::*;

#[test]
fn tt_randn_allocates_only_max_ranks_reachable_sectors() {
    let d = 6;
    let pg = c2v_table(d);
    let census = irrep_census(&pg, d);
    let (nup, ndn, target) = pick_nontrivial_target(&census, d);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(4242);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(1..=4)
                };
            }
            r
        })
        .collect();
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());

    let caps = max_ranks(&space);
    let mut any_masked = false;
    let mut any_over_capped = false;
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            let allocated = x.rank(b, s);
            if allocated > 0 {
                assert!(
                    caps[b][s] > 0,
                    "bond {b} sector {s:?}: tt_randn allocated rank {allocated} at a sector \
                     max_ranks says is unreachable (cap 0) — a clamp instead of a mask, or a \
                     missing mask entirely"
                );
            }
            if caps[b][s] == 0 && ranks[b][s] > 0 {
                any_masked = true;
                assert_eq!(
                    allocated, 0,
                    "bond {b} sector {s:?}: unreachable sector must end up at rank 0"
                );
            }
            if caps[b][s] > 0 {
                assert_eq!(
                    allocated, ranks[b][s],
                    "bond {b} sector {s:?}: cap {} > 0 but allocated rank {allocated} != \
                     requested rank {} — tt_randn is clamping a positive-cap sector instead of \
                     masking only cap-0 sectors",
                    caps[b][s], ranks[b][s]
                );
                if ranks[b][s] > caps[b][s] {
                    any_over_capped = true;
                }
            }
        }
    }
    assert!(
        any_masked,
        "vacuous: the naive grid never touched an irrep-unreachable sector at any bond"
    );
    assert!(
        any_over_capped,
        "vacuous: no positive-cap sector ever received a requested rank exceeding its cap, so \
         this fixture cannot distinguish a mask from a clamp"
    );

    assert_purity(&pg, d, nup, ndn, target, &x.to_dense());
}
