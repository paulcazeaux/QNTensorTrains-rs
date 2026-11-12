// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg2::*;

fn nontrivial_randn_fixture(d: usize, seed: u64) -> TtVector<f64> {
    let pg = c2v_table(d);
    let census = irrep_census(&pg, d);
    let (nup, ndn, target) = pick_nontrivial_target(&census, d);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
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
    x
}

#[test]
fn dot_matches_dense_oracle_at_nontrivial_group() {
    let x = nontrivial_randn_fixture(6, 31415);

    let via_dot = dot(&x, &x);
    let via_dense: f64 = x.to_dense().iter().map(|v| v * v).sum();
    assert!(
        (via_dot - via_dense).abs() < 1e-9 * (1.0 + via_dense.abs()),
        "dot(x,x) = {via_dot} but sum of |amplitude|^2 from to_dense = {via_dense} — a \
         wrong-irrep unapply in ops.rs::dot (or a wrong right-boundary sector) silently drops or \
         misroutes contributions at |G|>1"
    );
    assert!(via_dense > 0.0, "vacuous: x is the zero vector");
}

#[test]
fn qr_lr_round_trip_at_nontrivial_group() {
    let x = nontrivial_randn_fixture(6, 2718);

    let (y, rblock) = x.qr_lr();
    assert!(y.check());
    let r_scalar = *rblock.get(0, 0);

    let x_dense = x.to_dense();
    let y_dense = y.to_dense();
    assert_eq!(x_dense.len(), y_dense.len());
    let mut any_nonzero = false;
    for (&xv, &yv) in x_dense.iter().zip(y_dense.iter()) {
        let recon = yv * r_scalar;
        if xv.abs() > 1e-12 {
            any_nonzero = true;
        }
        assert!(
            (recon - xv).abs() < 1e-8 * (1.0 + xv.abs()),
            "qr_lr round trip mismatch: y*R = {recon} but x = {xv}"
        );
    }
    assert!(any_nonzero, "vacuous: x is the zero vector");
}

#[test]
fn qc_orthonormality_via_adjoint_core_times_core_at_nontrivial_group() {
    let x = nontrivial_randn_fixture(6, 909);
    let k = 3;
    let (q_core, _c_sketch, new_col_ranks) = x.core(k).qc();

    let qtq = adjoint_core_times_core(&q_core, &q_core);
    let mut any_checked = false;
    for s in x.space.bond_sectors(k + 1) {
        let n = new_col_ranks[s];
        if n == 0 {
            continue;
        }
        let block = qtq
            .block(s)
            .unwrap_or_else(|| panic!("adjoint_core_times_core: missing block at {s:?}"));
        assert_eq!((block.nrows(), block.ncols()), (n, n));
        any_checked = true;
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = *block.get(i, j);
                assert!(
                    (got - expected).abs() < 1e-8,
                    "Q^H Q at sector {s:?} ({i},{j}): expected {expected}, got {got} — a \
                     wrong-irrep unapply in qc/adjoint_core_times_core breaks orthonormality at \
                     |G|>1"
                );
            }
        }
    }
    assert!(any_checked, "vacuous: no positive-rank column sector found");
}
