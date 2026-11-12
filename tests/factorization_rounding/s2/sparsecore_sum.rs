// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::expand::expand;
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;

use crate::s2::util::{
    assert_dense_close, close_rel_abs, frob, frob_diff, randomized_su2_state, seeded,
};

#[test]
fn s2_6a_scale_unscale_roundtrip_horizontal_and_vertical() {
    let mut rng = seeded(20_206);
    let space = Su2Space::new(5, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let a = x.core(2).clone();

    for l in a.row_qn.iter().copied() {
        if a.row_ranks[l] == 0 {
            continue;
        }
        let h = a.horizontal_unfolding(l);
        let mut h2 = h.clone();
        a.scale_horizontal(&mut h2, l);
        a.unscale_horizontal(&mut h2, l);
        let dev = frob_diff(h.as_ref(), h2.as_ref());
        assert!(
            dev <= 1e-15 * frob(h.as_ref()).max(1.0),
            "horizontal roundtrip dev {dev}"
        );
    }

    for r in a.col_qn.iter().copied() {
        if a.col_ranks[r] == 0 {
            continue;
        }
        let v = a.vertical_unfolding(r);
        let mut v2 = v.clone();
        a.scale_vertical(&mut v2, r);
        a.unscale_vertical(&mut v2, r);
        let dev = frob_diff(v.as_ref(), v2.as_ref());
        assert!(
            dev <= 1e-15 * frob(v.as_ref()).max(1.0),
            "vertical roundtrip dev {dev}"
        );
    }
}

#[test]
fn s2_6b_unfolding_scatter_write_back_preserves_dense_image() {
    let mut rng = seeded(20_207);
    let space = Su2Space::new(5, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let want = expand(&x, space.two_s as i64).to_dense();

    let row_qn = x.core(2).row_qn.clone();
    let col_qn = x.core(2).col_qn.clone();
    let row_ranks = x.core(2).row_ranks.clone();
    let col_ranks = x.core(2).col_ranks.clone();
    for l in row_qn {
        if row_ranks[l] == 0 {
            continue;
        }
        let h = x.core(2).horizontal_unfolding(l);
        x.cores[2].scatter_horizontal(l, h.as_ref());
    }
    for r in col_qn {
        if col_ranks[r] == 0 {
            continue;
        }
        let v = x.core(2).vertical_unfolding(r);
        x.cores[2].scatter_vertical(r, v.as_ref());
    }

    let got = expand(&x, space.two_s as i64).to_dense();
    assert_dense_close(&got, &want, 1e-14);
}

#[test]
fn sum_move_core_then_lmul_scales_norm_and_keeps_orthogonal() {
    let mut rng = seeded(20_301);
    let space = Su2Space::new(6, 4, 0);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let nrm = x.norm();

    let mut w = x.clone();
    w.move_core(2, false);
    let w2 = &w * 2.0;

    assert!(
        close_rel_abs(w2.norm(), 2.0 * nrm, 1e-12, 0.0),
        "norm(2w)={} != 2*norm(x)={}",
        w2.norm(),
        2.0 * nrm
    );
    assert!(
        w2.ortho.is_some(),
        "scaling the center must preserve the orthogonal flag"
    );
}
