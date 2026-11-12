// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::round::RoundMode;
use qntt::{CoeffTensor, QnSpace, Scalar, SparseHamiltonian, TtVector, tt_state};
pub(crate) use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn state_from_sites<T: Scalar>(
    d: usize,
    up_sites: &[usize],
    dn_sites: &[usize],
) -> TtVector<T> {
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in up_sites {
        occ_up[s] = true;
    }
    for &s in dn_sites {
        occ_dn[s] = true;
    }
    tt_state(&occ_up, &occ_dn)
}

pub(crate) fn two_body_swap_h(
    space: QnSpace,
    a: usize,
    b: usize,
    c: usize,
    e: usize,
) -> SparseHamiltonian<f64> {
    let d = space.d;
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[c, a, e, b], -1.0);
    v.set(&[b, e, a, c], -1.0);
    SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub(crate) fn span_states(d: usize, a: usize, b: usize, c: usize, e: usize) -> Vec<TtVector<f64>> {
    let mut span: Vec<(usize, usize)> = vec![(a, b), (c, e), (b, a), (e, c)];
    span.sort_unstable();
    span.dedup();
    span.iter()
        .map(|&(u, w)| state_from_sites(d, &[u], &[w]))
        .collect()
}

#[allow(clippy::type_complexity)]
pub(crate) fn tangent_test_setup(
    seed: u64,
) -> (
    QnSpace,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
) {
    let d = 6usize;
    let space = QnSpace::new(d, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    let mut dir1: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    dir1.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    dir1.scale_mut(1.0 / dir1.norm());
    let mut dir2: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    dir2.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    dir2.scale_mut(1.0 / dir2.norm());
    (space, x, base_l, base_r, dir1, dir2)
}
