// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::{InnerProductVector, TtState};
use qntt::su2::qn::{Su2Space, max_ranks};
use qntt::su2::twosite::{contract, factor_qc, factor_svd};
use qntt::su2::vector::Su2TtVector;

use crate::spg4b::util::*;

#[test]
fn spg4b_x1_contract_block_dims_match_the_independent_window_oracle() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB1_0001);

    let mut some_positive = false;
    let mut some_swap_differs = false;
    for &k in &[0usize, 2, 4] {
        let gl = pg.irrep_of(k);
        let gr = pg.irrep_of(k + 1);
        assert_ne!(
            gl, gr,
            "fixture bug: gl == gr at k={k}, not a swap-detector"
        );
        let cp = contract(x.core(k), x.core(k + 1));
        for &m in &cp.mid_qn {
            let rows_correct: usize = window_rows(&space, k, m, gl)
                .iter()
                .map(|&(_, l)| x.core(k).row_ranks[l])
                .sum();
            let cols_correct: usize = window_cols(&space, k, m, gr)
                .iter()
                .map(|&(_, r)| x.core(k + 1).col_ranks[r])
                .sum();
            let got = cp
                .block(m)
                .unwrap_or_else(|| panic!("k={k}: admissible mid sector {m:?} has no block"));
            assert_eq!(
                got.nrows(),
                rows_correct,
                "k={k} m={m:?}: row dim mismatches the mpred oracle"
            );
            assert_eq!(
                got.ncols(),
                cols_correct,
                "k={k} m={m:?}: col dim mismatches the msucc oracle"
            );
            if got.nrows() > 0 && got.ncols() > 0 {
                some_positive = true;
            }

            let rows_swapped: usize = window_rows(&space, k, m, gr)
                .iter()
                .map(|&(_, l)| x.core(k).row_ranks[l])
                .sum();
            let cols_swapped: usize = window_cols(&space, k, m, gl)
                .iter()
                .map(|&(_, r)| x.core(k + 1).col_ranks[r])
                .sum();
            if rows_swapped != rows_correct || cols_swapped != cols_correct {
                some_swap_differs = true;
            }
        }
    }
    assert!(some_positive, "vacuous: no positive-dim mid block found");
    assert!(
        some_swap_differs,
        "vacuous: swapping gl/gr never changed any block dims — the fixture cannot distinguish \
         the two conventions"
    );
}

#[test]
fn spg4b_x2_contract_factor_qc_roundtrip_is_exact_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB2_0001);

    for &k in &[0usize, 2, 4] {
        let tc = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&tc);
        let mut cores = x.cores.clone();
        cores[k] = xk;
        cores[k + 1] = xk1;
        let y = Su2TtVector::new(space, cores);
        assert_expand_matches(&y, &x, 1e-12);
        assert!(y.check(), "k={k}: factor_qc roundtrip failed check()");
        assert_expand_purity_chain(&y, &pg);
    }
}

#[test]
fn spg4b_x3_factor_svd_roundtrip_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB3_0001);
    let caps = max_ranks(&space);

    for &k in &[0usize, 2, 4] {
        let tc = contract(x.core(k), x.core(k + 1));
        let (xk, s, xk1) = factor_svd(&tc, 0.0, qntt::Pooled::NONE, false);
        for m in space.bond_sectors(k + 1) {
            if caps[k + 1][m] == 0 {
                assert_eq!(
                    s.row_ranks()[m],
                    0,
                    "k={k} m={m:?}: cap-0 mid sector kept nonzero rank"
                );
            }
        }
        let y = rebuild_two_site(&x, k, xk, &s, xk1);
        assert_expand_matches(&y, &x, 1e-12);
        assert!(y.check(), "k={k}: factor_svd roundtrip failed check()");
        assert_expand_purity_chain(&y, &pg);
    }
}

#[test]
fn spg4b_x4_dot_cores_norm_equals_the_state_norm() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let x0 = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB4_0001);

    for &k in &[0usize, 2, 4] {
        let mut x = x0.clone();
        x.move_core(k, true);
        let cp = contract(x.core(k), x.core(k + 1));
        let n1 = InnerProductVector::norm(&cp);
        let n2 = TtState::norm(&x);
        assert!(n2 > 1e-8, "k={k}: vacuous — state norm ~0");
        assert!(
            (n1 - n2).abs() <= 1e-10 * n2.max(1.0),
            "k={k}: dot_cores norm {n1} vs state norm {n2}"
        );
    }
}
