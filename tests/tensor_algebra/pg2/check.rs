// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg2::*;

#[test]
#[should_panic(expected = "right boundary sector")]
fn check_rejects_wrong_right_boundary_irrep() {
    let d = 6;
    let pg = c2v_table(d);
    let (nup, ndn) = (3, 2);
    let census = irrep_census(&pg, d);
    assert!(
        census
            .get(&(nup, ndn, Irrep::IDENTITY))
            .copied()
            .unwrap_or(0)
            > 0,
        "fixture must have an IDENTITY-irrep determinant at (nup={nup}, ndn={ndn})"
    );
    let target = census
        .iter()
        .filter(|&(&(u, v, ir), &c)| u == nup && v == ndn && ir != Irrep::IDENTITY && c > 0)
        .map(|(&(_, _, ir), _)| ir)
        .next()
        .expect("fixture must have a reachable nonidentity target at (nup, ndn)");

    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
    let space0 = QnSpace::with_point_group(d, nup, ndn, pg, Irrep::IDENTITY);
    let ranks = max_ranks(&space0);

    let cores: Vec<SparseCore<f64>> = (0..d)
        .map(|k| SparseCore::zeros(space, k, &ranks[k], &ranks[k + 1]))
        .collect();
    let x: TtVector<f64> = TtVector::new(space, cores);
    x.check();
}

#[test]
#[should_panic(expected = "left boundary sector")]
fn check_rejects_wrong_left_boundary_irrep() {
    let d = 4;
    let pg = c2v_table(d);
    let space = QnSpace::with_point_group(d, 0, 0, pg, Irrep::IDENTITY);
    let wrong = Irrep::new(1);
    assert_ne!(wrong, Irrep::IDENTITY);

    let wrong_sector = Sector::with_irrep(0, 0, wrong);
    let mut ranks = Vec::with_capacity(d + 1);
    for _ in 0..=d {
        let mut r = RankGrid::zeros(&space);
        r[wrong_sector] = 1;
        ranks.push(r);
    }
    let cores: Vec<SparseCore<f64>> = (0..d)
        .map(|k| SparseCore::zeros(space, k, &ranks[k], &ranks[k + 1]))
        .collect();
    let x: TtVector<f64> = TtVector::new(space, cores);
    x.check();
}

#[test]
fn tt_ones_at_identity_target_nontrivial_group_is_positive() {
    let d = 4;
    let pg = c2v_table(d);
    assert!(pg.order() > 1);
    let (nup, ndn, target) = (2, 1, Irrep::IDENTITY);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

    let x: TtVector<f64> = tt_ones(space);
    assert!(x.check());
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
        assert_eq!(val, want, "config {idx}");
        if in_particle_block {
            n_block += 1;
        }
        if want == 1.0 {
            n_ones += 1;
        }
    }
    assert!(
        0 < n_ones && n_ones < n_block,
        "the identity-target indicator at |G|=4 must be a PROPER, non-empty subset ({n_ones} of \
         {n_block}) — a vacuous split here means the group was not actually irrep-live"
    );
}

#[test]
fn check_passes_on_every_constructor_output_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let census = irrep_census(&pg, d);
    let (nup, ndn, target) = pick_nontrivial_target(&census, d);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

    let z: TtVector<f64> = tt_zeros(space);
    assert!(z.check());

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(555);
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
    let rnd: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(rnd.check());

    let (occ_up, occ_dn) = find_determinant(&pg, d, nup, ndn, target);
    let st: TtVector<f64> = tt_state_in(space, &occ_up, &occ_dn);
    assert!(st.check());

    let ones: TtVector<f64> = tt_ones(space);
    assert!(ones.check());
}
