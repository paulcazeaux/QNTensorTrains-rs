// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg2::*;

#[test]
fn tt_state_in_matching_target_succeeds_with_exactly_one_nonzero() {
    let d = 6;
    let pg = c2v_table(d);
    let census = irrep_census(&pg, d);
    let (nup, ndn, target) = pick_nontrivial_target(&census, d);
    let (occ_up, occ_dn) = find_determinant(&pg, d, nup, ndn, target);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

    let x: TtVector<f64> = tt_state_in(space, &occ_up, &occ_dn);
    assert!(x.check());

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

    let dense = x.to_dense();
    let nonzero: Vec<usize> = dense
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v != 0.0)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        nonzero,
        vec![expected_idx],
        "tt_state_in must place exactly one nonzero amplitude, at the requested determinant"
    );
}

#[test]
#[should_panic(expected = "differs from the space target")]
fn tt_state_in_rejects_mismatched_target_irrep() {
    let d = 6;
    let pg = c2v_table(d);
    let census = irrep_census(&pg, d);
    let (nup, ndn, target) = pick_nontrivial_target(&census, d);
    let (occ_up, occ_dn) = find_determinant(&pg, d, nup, ndn, target);

    let wrong_target = Irrep::new((target.index() + 1) % pg.order());
    assert_ne!(wrong_target, target);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, wrong_target);

    let _x: TtVector<f64> = tt_state_in(space, &occ_up, &occ_dn);
}
