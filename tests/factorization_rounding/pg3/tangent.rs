// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg3::*;

#[allow(clippy::type_complexity)]
fn pg3_tangent_setup(
    seed: u64,
) -> (
    PointGroup,
    QnSpace,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
) {
    let (pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks = capped_ranks(&space, 2, 3, &mut rng);
    let mut x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);

    let ranks1 = capped_ranks(&space, 2, 3, &mut rng);
    let mut dir1: TtVector<f64> = tt_randn(space, &ranks1, &mut rng);
    dir1.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    dir1.scale_mut(1.0 / dir1.norm());

    let ranks2 = capped_ranks(&space, 2, 3, &mut rng);
    let mut dir2: TtVector<f64> = tt_randn(space, &ranks2, &mut rng);
    dir2.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    dir2.scale_mut(1.0 / dir2.norm());

    (pg, space, x, base_l, base_r, dir1, dir2)
}

#[test]
fn pg3_tangent_project_to_vector_checks_and_is_pure() {
    let (pg, space, _x, base_l, base_r, dir1, _dir2) = pg3_tangent_setup(0x9503_0001);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let v = dx.to_vector();
    assert!(
        v.check(),
        "the fix oracle: tangent projection must satisfy the boundary law"
    );
    assert_pure(&pg, &space, &v);

    assert_eq!(v.ranks[0][Sector::new(0, 0)], 1);
    for s in space.bond_sectors(0) {
        assert!(
            s == Sector::new(0, 0) || v.ranks[0][s] == 0,
            "bond 0 phantom rank at {s:?}"
        );
    }
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    for s in space.bond_sectors(space.d) {
        assert!(
            s == last || v.ranks[space.d][s] == 0,
            "bond d phantom rank at {s:?}"
        );
    }
}

#[test]
fn pg3_tangent_gauge_condition() {
    let (_pg, space, _x, base_l, base_r, dir1, _dir2) = pg3_tangent_setup(0x9503_0002);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let d = space.d;
    let mut checked = 0usize;
    for k in 0..d - 1 {
        for s in space.bond_sectors(k + 1) {
            let l = base_l.core(k).vertical_unfolding(s);
            let c = dx.component(k).vertical_unfolding(s);
            if l.ncols() == 0 || c.ncols() == 0 {
                continue;
            }
            let res = gauge_residual(l.as_ref(), c.as_ref());
            assert!(
                res < 1e-12,
                "gauge condition at bond {}, sector {s:?}: {res}",
                k + 1
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "vacuous: no positive-rank sector checked");
}

#[test]
fn pg3_tangent_linearity_idempotence_metric() {
    let (_pg, _space, _x, base_l, base_r, dir1, dir2) = pg3_tangent_setup(0x9503_0003);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let dy = TtTangent::project(&base_l, &base_r, &dir2);

    let combo = &(&dir1 * 0.7) - &(&dir2 * 1.3);
    let dxy = TtTangent::project(&base_l, &base_r, &combo);
    let want = dx.scale(0.7).sub(&dy.scale(1.3));
    let diff = &dxy.to_vector() - &want.to_vector();
    assert!(diff.norm_lr(false) < 1e-12, "projection linearity");

    let dx2 = TtTangent::project(&base_l, &base_r, &dx.to_vector());
    let diff2 = &dx.to_vector() - &dx2.to_vector();
    assert!(diff2.norm_lr(false) < 1e-12, "projection idempotence");

    assert!(
        (dx.norm() - dx.to_vector().norm_lr(false)).abs() < 1e-12 * dx.norm().max(1.0),
        "norm consistency"
    );
    assert!(
        (dx.dot(&dy) - dot(&dx.to_vector(), &dy.to_vector())).abs() < 1e-12,
        "dot consistency"
    );
    assert!(
        (dx.dot(&dx) - dx.norm() * dx.norm()).abs() < 1e-12,
        "dot(dx,dx) == norm(dx)^2"
    );
}

#[test]
fn pg3_tangent_retraction_and_transport() {
    let (_pg, space, x, base_l, base_r, dir1, _dir2) = pg3_tangent_setup(0x9503_0004);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);

    let r0 = dx.retract_hosvd(0.0);
    assert!((&r0 - &x).norm_lr(false) < 1e-12, "retract at α=0 == x");

    let alpha = 1e-3;
    let lin_at = |a: f64| -> f64 {
        let ra = dx.retract_hosvd(a);
        assert!(ra.check(), "retraction must satisfy the boundary law");
        let linear = &x + &(&dx.to_vector() * a);
        (&ra - &linear).norm_lr(false)
    };
    let e1 = lin_at(alpha);
    let e2 = lin_at(2.0 * alpha);
    assert!(
        e1 < 10.0 * alpha * alpha,
        "first-order retraction error e1={e1}"
    );
    assert!(
        (e2 / e1 - 4.0).abs() < 1.0,
        "retraction ratio e2/e1 = {}",
        e2 / e1
    );

    let self_transport = projective_transport(&x, &dx);
    let diff = &self_transport.to_vector() - &dx.to_vector();
    assert!(diff.norm_lr(false) < 1e-12, "self-transport is identity");

    let mut rng_y = Xoshiro256PlusPlus::seed_from_u64(0x9503_0005);
    let ranks_y = capped_ranks(&space, 2, 3, &mut rng_y);
    let mut y: TtVector<f64> = tt_randn(space, &ranks_y, &mut rng_y);
    y.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    y.scale_mut(1.0 / y.norm());
    let mut y_base_l = y.clone();
    y_base_l.left_orthogonalize(true);
    let transported = projective_transport(&y, &dx);
    let mut checked = 0usize;
    for k in 0..space.d - 1 {
        for s in space.bond_sectors(k + 1) {
            let l = y_base_l.core(k).vertical_unfolding(s);
            let c = transported.component(k).vertical_unfolding(s);
            if l.ncols() == 0 || c.ncols() == 0 {
                continue;
            }
            let res = gauge_residual(l.as_ref(), c.as_ref());
            assert!(
                res < 1e-12,
                "transported gauge condition at bond {}, sector {s:?}",
                k + 1
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "vacuous: no positive-rank sector checked in the transported gauge check"
    );
}
