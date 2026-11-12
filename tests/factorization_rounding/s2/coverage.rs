// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::MatRef;

use qntt::su2::centerweight;
use qntt::su2::core::Family as Su2Family;
use qntt::su2::expand::expand;
use qntt::su2::frame::Su2Sketch;
use qntt::su2::qn::{Su2RankGrid, Su2Space};
use qntt::su2::vector::{Orthogonal, Su2TtVector, tt_randn_orth};

use qntt::{C64, RoundMode, Scalar};

use crate::s2::util::{
    assert_dense_close, close_rel_abs, dense_diff_norm, dense_norm, max_dev_identity,
    randomized_su2_state, seeded,
};

#[test]
fn tst2a_norm_lr_rl_equal_fixed_sz_norm() {
    let mut rng = seeded(20_720);
    let space = Su2Space::new(7, 5, 1);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 4, &mut rng);
    let n_expand = dense_norm(&expand(&x, space.two_s as i64).to_dense());
    assert!(n_expand > 0.0);

    for &keep in &[false, true] {
        let nlr = x.norm_lr(keep);
        let nrl = x.norm_rl(keep);
        assert!(
            close_rel_abs(nlr, n_expand, 1e-9, 0.0),
            "norm_lr(keep={keep})={nlr} != ||expand||={n_expand}"
        );
        assert!(
            close_rel_abs(nrl, n_expand, 1e-9, 0.0),
            "norm_rl(keep={keep})={nrl} != ||expand||={n_expand}"
        );
    }
}

#[test]
fn tst2b_round_lr_mut_invariance_and_left_canonicity() {
    let mut rng = seeded(20_721);
    let space = Su2Space::new(7, 5, 1);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 5, &mut rng);
    let want = expand(&x, space.two_s as i64).to_dense();
    let nrm = dense_norm(&want);

    x.right_orthogonalize(false);
    assert_eq!(x.ortho, Some(0), "round_lr requires center 0 input");

    x.round_lr_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    assert_eq!(x.ortho, Some(space.d - 1), "round_lr lands left-canonical");

    let got = expand(&x, space.two_s as i64).to_dense();
    assert!(
        dense_diff_norm(&got, &want) <= 1e-10 * nrm.max(1.0),
        "round_lr_mut not lossless: err={}",
        dense_diff_norm(&got, &want)
    );

    let dev = crate::s2::util::su2_left_canonicity_dev(&x, 0..(space.d - 1));
    assert!(dev < 1e-13, "round_lr_mut left-canonicity dev {dev}");
}

#[test]
fn r12_complex_svdvals_weighted_sigma2_sum() {
    let mut rng = seeded(20_113);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<C64> = randomized_su2_state(space, 1, 4, &mut rng);
    let nrm2 = dense_norm(&expand(&x, space.two_s as i64).to_dense()).powi(2);

    let sv = x.svdvals();
    for k in 1..space.d {
        let sk = &sv[k - 1];
        let mut s2 = 0.0f64;
        for sec in sk.sectors() {
            let Some(b) = sk.block(sec) else { continue };
            let cw = centerweight(sec.two_s, space.two_s);
            let n = b.nrows().min(b.ncols());
            for i in 0..n {
                let v = b.get(i, i).abs_s();
                s2 += cw * v * v;
            }
        }
        assert!(
            close_rel_abs(s2, nrm2, 1e-10, 0.0),
            "bond {k}: weighted sigma^2 sum {s2} != ||x||^2 {nrm2}"
        );
    }
}

#[test]
fn tst5_adjoint_core_times_core_left_canonical_identity() {
    let mut rng = seeded(20_805);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    x.left_orthogonalize(false);

    for k in 0..(space.d - 1) {
        let core = x.core(k);
        let g = qntt::su2::coreops::adjoint_core_times_core(core, core);
        for t in space.bond_sectors(k + 1) {
            if core.col_ranks[t] == 0 {
                continue;
            }
            let b = g.block(t).expect("gram block present");
            assert!(
                max_dev_identity(b) < 1e-12,
                "core {k} sector {t:?}: Gram not identity"
            );
        }
    }
}

#[test]
fn tst5_core_algebra_axpy_axpby_dot_center_identity() {
    let mut rng = seeded(20_800);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let a = x.core(2).clone();

    let (f0, l0) = {
        let mut found = None;
        'outer: for f in Su2Family::ALL {
            for l in a.row_qn.iter().copied() {
                if let Some(b) = a.block(f, l)
                    && b.nrows() > 0
                    && b.ncols() > 0
                {
                    found = Some((f, l));
                    break 'outer;
                }
            }
        }
        found.expect("core 2 has a nonempty block")
    };
    let a0 = a.block(f0, l0).unwrap().to_owned();
    let block_max_diff = |b: MatRef<'_, f64>, factor: f64| -> f64 {
        let mut m = 0.0f64;
        for j in 0..a0.ncols() {
            for i in 0..a0.nrows() {
                m = m.max((*b.get(i, j) - factor * *a0.get(i, j)).abs());
            }
        }
        m
    };

    let mut w = a.clone();
    w.axpy(2.0, &a);
    assert!(
        block_max_diff(w.block(f0, l0).unwrap(), 3.0) < 1e-13,
        "axpy"
    );

    let mut w2 = a.clone();
    w2.axpby(2.0, &a, 0.5);
    assert!(
        block_max_diff(w2.block(f0, l0).unwrap(), 2.5) < 1e-13,
        "axpby"
    );

    let dcaa = a.dot_center(&a);
    assert!(close_rel_abs(dcaa, a.abs2_center(), 1e-12, 0.0));
    assert!(dcaa > 0.0);
    let mut b2 = a.clone();
    b2.scale_mut(2.0);
    let dcab = a.dot_center(&b2);
    assert!(
        close_rel_abs(dcab, 2.0 * dcaa, 1e-12, 0.0),
        "dot_center linearity"
    );

    let idsk = Su2Sketch::<f64>::identity(space, 3, 2);
    for s in space.bond_sectors(3) {
        let b = idsk.block(s).expect("identity block present");
        assert_eq!((b.nrows(), b.ncols()), (2, 2));
        assert!(max_dev_identity(b) < 1e-15, "identity sector {s:?}");
    }
}

#[test]
fn tst5_tt_randn_orth_stewart_left_right_construct() {
    let mut rng = seeded(20_810);
    let space = Su2Space::new(6, 4, 2);
    let d = space.d;
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d { 1 } else { 2 };
            }
            r
        })
        .collect();

    for mode in [Orthogonal::Left, Orthogonal::Right] {
        let x = tt_randn_orth::<f64, _>(space, &ranks, mode, &mut rng);
        assert!(x.check(), "stewart {mode:?}: invalid train");
        assert!(
            x.ortho.is_none(),
            "stewart trains are never flagged orthogonal"
        );
        let n = dense_norm(&expand(&x, space.two_s as i64).to_dense());
        assert!(
            n.is_finite() && n > 0.0,
            "stewart {mode:?}: degenerate norm {n}"
        );
    }
}

#[test]
fn tst5_d1_special_cases() {
    let mut rng = seeded(20_820);
    let space = Su2Space::new(1, 2, 0);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
    let want = expand(&x, 0).to_dense();

    let mut y = x.clone();
    y.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Abs);
    assert_eq!(y.ortho, Some(0));
    assert_dense_close(&expand(&y, 0).to_dense(), &want, 1e-12);

    let mut z = x.clone();
    assert!(z.svdvals().is_empty(), "d==1 svdvals must be empty");

    assert!(close_rel_abs(x.norm_lr(false), x.norm(), 1e-10, 0.0));
    assert!(close_rel_abs(x.norm_rl(false), x.norm(), 1e-10, 0.0));

    let ranks: Vec<Su2RankGrid> = (0..=1)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = 1;
            }
            r
        })
        .collect();
    let mut w = x.clone();
    w.round_ranks_mut(&ranks);
    assert_dense_close(&expand(&w, 0).to_dense(), &want, 1e-12);
}
