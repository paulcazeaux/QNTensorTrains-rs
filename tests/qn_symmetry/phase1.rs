// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::{QnSpace, RankGrid, TtVector, tt_ones, tt_randn};
use qntt_testkit::pg::config_counts;
use qntt_testkit::pg::digits;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn tt_ones_dense_is_sector_indicator() {
    let d = 8;
    let space = QnSpace::from_n_two_sz(d, 5, 1);
    let x: TtVector<f64> = tt_ones(space);
    x.check();
    let dense = x.to_dense();
    for (idx, &v) in dense.iter().enumerate() {
        let (nup, ndn) = config_counts(&digits(idx, d));
        let expected = if nup == space.nup && ndn == space.ndn {
            1.0
        } else {
            0.0
        };
        assert_eq!(v, expected, "config {idx} (nup={nup}, ndn={ndn})");
    }
}

#[test]
fn tt_randn_dense_supported_exactly_on_sector() {
    let d = 8;
    let space = QnSpace::from_n_two_sz(d, 6, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(102);
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
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    x.check();

    let dense = x.to_dense();
    let mut nonzero_admissible = 0usize;
    for (idx, &v) in dense.iter().enumerate() {
        let (nup, ndn) = config_counts(&digits(idx, d));
        if nup == space.nup && ndn == space.ndn {
            assert!(v != 0.0, "admissible config {idx} unexpectedly zero");
            nonzero_admissible += 1;
        } else {
            assert_eq!(v, 0.0, "inadmissible config {idx} must be exactly zero");
        }
    }
    assert_eq!(nonzero_admissible, 70 * 28);
}

#[test]
fn tt_randn_complex_support() {
    let d = 6;
    let space = QnSpace::from_n_two_sz(d, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
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
    let x: TtVector<qntt::C64> = tt_randn(space, &ranks, &mut rng);
    x.check();
    let dense = x.to_dense();
    for (idx, &v) in dense.iter().enumerate() {
        let (nup, ndn) = config_counts(&digits(idx, d));
        if nup == space.nup && ndn == space.ndn {
            assert!(v.norm() > 0.0, "admissible config {idx} unexpectedly zero");
        } else {
            assert_eq!(v, qntt::C64::new(0.0, 0.0));
        }
    }
}
