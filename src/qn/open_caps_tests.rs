// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use crate::ops::block_tests::open_block_uniform;
use crate::qn::{QnSpace, Sector, max_ranks, max_ranks_open};
use crate::test_util::pg3_space_d4;
use crate::{Irrep, TtVector};

fn dense_rank(mut a: Vec<Vec<f64>>) -> usize {
    let (m, n) = (a.len(), a.first().map_or(0, Vec::len));
    let scale = a
        .iter()
        .flatten()
        .fold(0.0f64, |acc, v| acc.max(v.abs()))
        .max(1e-300);
    let (mut rank, mut row) = (0, 0);
    for col in 0..n {
        if row == m {
            break;
        }
        let (p, big) = (row..m)
            .map(|i| (i, a[i][col].abs()))
            .fold((row, 0.0), |b, c| if c.1 > b.1 { c } else { b });
        if big <= 1e-9 * scale {
            continue;
        }
        a.swap(row, p);
        for i in row + 1..m {
            let f = a[i][col] / a[row][col];
            for j in col..n {
                a[i][j] -= f * a[row][j];
            }
        }
        row += 1;
        rank += 1;
    }
    rank
}

fn left_sector(space: &QnSpace, idx: usize, b: usize) -> Sector {
    let (mut nup, mut ndn, mut ir) = (0, 0, Irrep::IDENTITY);
    for k in 0..b {
        let dig = (idx >> (2 * k)) & 3;
        nup += usize::from(dig == 1 || dig == 3);
        ndn += usize::from(dig == 2 || dig == 3);
        if dig == 1 || dig == 2 {
            ir = ir.fuse(space.pg.irrep_of(k));
        }
    }
    Sector::with_irrep(nup, ndn, ir)
}

fn unfolding_rank(space: &QnSpace, cols: &[Vec<f64>], b: usize, s: Sector) -> usize {
    let d = space.d;
    let (nl, nr) = (1usize << (2 * b), 1usize << (2 * (d - b)));
    let rows: Vec<usize> = (0..nl).filter(|&l| left_sector(space, l, b) == s).collect();
    let a: Vec<Vec<f64>> = rows
        .iter()
        .map(|&l| {
            cols.iter()
                .flat_map(|c| (0..nr).map(move |r| c[l | (r << (2 * b))]))
                .collect()
        })
        .collect();
    dense_rank(a)
}

fn caps_match_unfolding_ranks(space: QnSpace, rho: usize) {
    let d = space.d;
    let x: TtVector<f64> = open_block_uniform(space, 40, rho, 78);
    let cols: Vec<Vec<f64>> = (0..rho).map(|j| x.column(j).to_dense()).collect();
    let caps = max_ranks_open(&space, rho);
    assert_eq!(caps.len(), d + 1);
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            let want = unfolding_rank(&space, &cols, b, s);
            assert_eq!(caps[b][s], want, "bond {b} sector {s:?} at ρ = {rho}");
        }
    }
}

#[test]
fn open_caps_match_brute_force_unfolding_ranks() {
    for rho in [1, 2, 3, 5] {
        caps_match_unfolding_ranks(QnSpace::new(4, 2, 1), rho);
    }
    caps_match_unfolding_ranks(QnSpace::new(5, 2, 2), 4);
}

#[test]
fn open_caps_match_brute_force_at_nontrivial_point_group() {
    let (_pg, space) = pg3_space_d4();
    for rho in [1, 3] {
        caps_match_unfolding_ranks(space, rho);
    }
}

#[test]
fn open_caps_reduce_to_max_ranks_and_are_monotone() {
    let (_pg, pgspace) = pg3_space_d4();
    for space in [QnSpace::new(6, 3, 2), QnSpace::new(3, 1, 1), pgspace] {
        let closed = max_ranks(&space);
        assert_eq!(max_ranks_open(&space, 1), closed);
        let (l, _r) = crate::qn::path_counts(&space);
        let open = max_ranks_open(&space, 7);
        for b in 0..=space.d {
            for s in space.bond_sectors(b) {
                assert!(open[b][s] >= closed[b][s]);
                assert!(open[b][s] as u64 <= l[b][s]);
                assert_eq!(open[b][s] == 0, closed[b][s] == 0);
            }
        }
    }
}
