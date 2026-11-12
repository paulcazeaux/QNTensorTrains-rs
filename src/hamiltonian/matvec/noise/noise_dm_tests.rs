// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashSet;

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::noise_density_matrix;
use crate::core::twosite::{contract, factor_qc};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::assembly::ChannelDesc;
use crate::hamiltonian::matvec::sketch_h_mul;
use crate::qn::{QnSpace, RankGrid};
use crate::solvers::als::left_to_right_sketching;
use crate::vector::TtVector;
use crate::{CoeffTensor, Scalar};

fn randomized_state<T: Scalar>(
    space: QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> TtVector<T> {
    let d = space.d;
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    crate::tt_randn(space, &ranks, rng)
}

#[test]
fn each_channel_contributes_unit_trace() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d - 1 {
        t.set(&[i, i + 1], -1.0);
        t.set(&[i + 1, i], -1.0);
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -0.7);
    v.set(&[1, 3, 0, 2], -0.7);
    v.set(&[4, 0, 5, 1], 0.4);
    v.set(&[0, 0, 2, 2], 0.9);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_FF_EE_11);
    let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    x.right_orthogonalize(true);
    let k = 2usize;
    x.move_core(k, true);
    x.check();

    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = sketch_h_mul(&fl, &h, &xk);
    let noise_dm = noise_density_matrix(&h, &half, &xk, &xk1, true);

    let mut total_trace = 0.0f64;
    for m in space.bond_sectors(k + 1) {
        let nd = &noise_dm[m];
        for i in 0..nd.nrows().min(nd.ncols()) {
            total_trace += nd[(i, i)];
        }
    }

    let mut groups: HashSet<ChannelDesc> = HashSet::default();
    for &s in &half.col_qn {
        for desc in h.channel_descs(k + 1, s) {
            groups.insert(*desc);
        }
    }
    let n_groups = groups.len();

    assert!(
        n_groups >= 2,
        "test is only meaningful with multiple channels; got {n_groups}"
    );
    let nearest = total_trace.round();
    assert!(
        (total_trace - nearest).abs() <= 1e-9 * total_trace.max(1.0),
        "Σ tr(ρ_noise) = {total_trace} is not an integer count of unit-trace channels"
    );
    assert!(
        (1.0..=n_groups as f64 + 1e-9).contains(&total_trace),
        "Σ tr(ρ_noise) = {total_trace} outside [1, {n_groups}] active-channel range"
    );
}
