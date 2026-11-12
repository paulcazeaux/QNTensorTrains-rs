// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg2::*;

#[test]
fn two_site_contract_factor_qc_round_trip_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let census = irrep_census(&pg, d);
    let (nup, ndn, target) = pick_nontrivial_target(&census, d);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(77);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(1..=3)
                };
            }
            r
        })
        .collect();
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());

    let k = 2;
    let tc = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&tc);
    let tc2 = contract(&xk, &xk1);

    let mut any_nonempty = false;
    for m in space.bond_sectors(k + 1) {
        match (tc.block(m), tc2.block(m)) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(
                    (a.nrows(), a.ncols()),
                    (b.nrows(), b.ncols()),
                    "mid sector {m:?}: shape mismatch after factor_qc round trip"
                );
                if a.nrows() > 0 && a.ncols() > 0 {
                    any_nonempty = true;
                }
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        let diff = (*a.get(i, j) - *b.get(i, j)).abs_s();
                        assert!(
                            diff < 1e-9,
                            "mid sector {m:?} entry ({i},{j}): factor_qc round trip mismatch \
                             {diff:e} — likely an irrep_of(k)/irrep_of(k+1) row/column swap"
                        );
                    }
                }
            }
            other => panic!("mid sector {m:?}: presence mismatch after round trip: {other:?}"),
        }
    }
    assert!(
        any_nonempty,
        "vacuous: no nonempty mid block found to compare"
    );
}
