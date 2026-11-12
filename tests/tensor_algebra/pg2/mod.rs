// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;

use qntt::core::factor::adjoint_core_times_core;
use qntt::core::twosite::{contract, factor_qc};
use qntt::{
    Irrep, PointGroup, QnSpace, RankGrid, Scalar, Sector, SparseCore, TtVector, dot, max_ranks,
    tt_ones, tt_randn, tt_state, tt_state_in, tt_zeros,
};
use qntt_testkit::pg::c2v_table;
use qntt_testkit::pg::config_counts;
use qntt_testkit::pg::config_irrep_from_digits;
use qntt_testkit::pg::digits;
use qntt_testkit::pg::irrep_census;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn config_occ(dig: &[usize]) -> (Vec<bool>, Vec<bool>) {
    let occ_up = dig.iter().map(|&x| x == 1 || x == 3).collect();
    let occ_dn = dig.iter().map(|&x| x == 2 || x == 3).collect();
    (occ_up, occ_dn)
}

fn find_determinant(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    irrep: Irrep,
) -> (Vec<bool>, Vec<bool>) {
    for idx in 0..(1usize << (2 * d)) {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            continue;
        }
        if config_irrep_from_digits(pg, &dig) == irrep {
            return config_occ(&dig);
        }
    }
    panic!("d={d}: no determinant with (nup={nup}, ndn={ndn}, irrep={irrep:?})");
}

fn pick_nontrivial_target(
    census: &HashMap<(usize, usize, Irrep), u64>,
    d: usize,
) -> (usize, usize, Irrep) {
    census
        .iter()
        .filter(|&(&(_, _, ir), &c)| ir != Irrep::IDENTITY && c > 0)
        .map(|(&(u, v, ir), _)| (u, v, ir))
        .min_by_key(|&(u, v, ir)| (u, v, ir.index()))
        .unwrap_or_else(|| panic!("d={d}: fixture has no reachable nonidentity-irrep determinant"))
}

fn assert_purity<T: Scalar>(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    target: Irrep,
    dense: &[T],
) {
    assert_eq!(dense.len(), 1usize << (2 * d));
    let mut any_nonzero_in_target = false;
    for (idx, v) in dense.iter().enumerate() {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            assert!(
                *v == T::zero(),
                "d={d}: config {idx} outside (Nup={nup},Ndn={ndn}) must be exactly zero, got {v:?}"
            );
            continue;
        }
        let irrep = config_irrep_from_digits(pg, &dig);
        if irrep == target {
            if *v != T::zero() {
                any_nonzero_in_target = true;
            }
        } else {
            assert!(
                *v == T::zero(),
                "THE campaign oracle: d={d} config {idx} has irrep {irrep:?} != target \
                 {target:?} but amplitude is {v:?} (not exactly zero) — a hand-rolled inversion \
                 or a Family::flux/qn::MOVES divergence would show up here"
            );
        }
    }
    assert!(
        any_nonzero_in_target,
        "d={d}: oracle vacuous — no nonzero amplitude found inside the target irrep {target:?}"
    );
}

mod bonus;
mod check;
mod coverage_gaps;
mod trivial_group;
mod tt_ones;
mod tt_randn;
mod tt_state_in;
