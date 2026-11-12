// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::pointgroup::Irrep;
use qntt::round::RoundMode;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks};
use qntt::su2::transition1rdm::{
    su2_transition_1rdm, su2_transition_1rdm_bruteforce, su2_transition_1rdm_h,
};
use qntt::su2::vector::Su2TtVector;

use crate::su2_util::c2v_table;
use crate::su2_util::spin_free_tv_symmetric;
use qntt_testkit::randomized_su2_state;
use qntt_testkit::seeded;

fn st_state(space: Su2Space, lo: usize, hi: usize, seed: u64) -> Su2TtVector<f64> {
    let mut rng = seeded(seed);
    let mut v: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
    v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    v
}

fn h_materialized(
    x: &Su2TtVector<f64>,
    h: &Su2SparseHamiltonian<f64>,
    w: &Su2TtVector<f64>,
) -> CoeffTensor<f64> {
    su2_transition_1rdm(x, &h.apply(w))
}

fn maxg_maxdiff(g: &CoeffTensor<f64>, r: &CoeffTensor<f64>, d: usize) -> (f64, f64) {
    let mut maxg = 0.0f64;
    let mut maxd = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            maxg = maxg.max(r.get(&[p, q]).abs());
            maxd = maxd.max((g.get(&[p, q]) - r.get(&[p, q])).abs());
        }
    }
    (maxg, maxd)
}

fn scale_of(x: &Su2TtVector<f64>, y: &Su2TtVector<f64>) -> f64 {
    (x.norm() * y.norm()).max(1.0)
}

#[test]
fn tst6_point_group_cross_irrep_exact_zero() {
    let d = 6usize;
    let pg = c2v_table(d);
    let space = Su2Space::with_point_group(d, 3, 1, pg, Irrep::IDENTITY);
    let caps = max_ranks(&space);
    assert!(
        caps[d][Su2Sector::with_irrep(3, 1, Irrep::IDENTITY)] > 0,
        "vacuous C2v target at (N=3, 2S=1) — pick another fixture rather than assert a false floor"
    );
    let (t, v) = spin_free_tv_symmetric(d, pg);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = st_state(space, 1, 2, 0x5F01);
    let y = st_state(space, 1, 3, 0x5F02);
    let sc = scale_of(&x, &y);

    for (name, g) in [
        ("sweep", su2_transition_1rdm(&x, &y)),
        ("h-materialized", h_materialized(&x, &h, &y)),
        ("h-streamed", su2_transition_1rdm_h(&x, &h, &y)),
    ] {
        let mut same = 0.0f64;
        for p in 0..d {
            for q in 0..d {
                if pg.irrep_of(p) == pg.irrep_of(q) {
                    same = same.max(g.get(&[p, q]).abs());
                } else {
                    assert_eq!(
                        g.get(&[p, q]),
                        0.0,
                        "{name}: cross-irrep cell [{p},{q}] must be EXACTLY 0.0"
                    );
                }
            }
        }
        assert!(same > 1e-6, "{name}: vacuous same-irrep block under C2v");
    }

    let gb = su2_transition_1rdm_bruteforce(&x, &y);
    let (_, diff) = maxg_maxdiff(&su2_transition_1rdm(&x, &y), &gb, d);
    assert!(
        diff <= 1e-12 * sc,
        "C2v sweep vs bruteforce: maxdiff {diff} > {}",
        1e-12 * sc
    );

    let hy = h.apply(&y);
    let gm = su2_transition_1rdm(&x, &hy);
    let (_, dh) = maxg_maxdiff(&su2_transition_1rdm_h(&x, &h, &y), &gm, d);
    assert!(
        dh <= 1e-11 * (x.norm() * hy.norm()).max(1.0),
        "C2v H-streamed sweep vs γ(x, H·y): maxdiff {dh}"
    );
}
