// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg3::*;

#[test]
fn pg3_left_orthogonalize_keeprank_conserves_ranks() {
    let (_pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_002);
    let ranks = capped_ranks(&space, 1, 2, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let xd = x.to_dense();
    let n0 = x.norm();
    let r0: Vec<RankGrid> = x.ranks.clone();

    let mut y = x.clone();
    y.left_orthogonalize(true);
    assert!(y.check());

    assert!(
        rel_err(&y.to_dense(), &xd) < 1e-13,
        "keeprank gauge equality"
    );
    assert!(
        (y.norm() - n0).abs() <= 1e-10 * n0,
        "ortho-fast-path norm exact after keeprank left-orthogonalize"
    );
    for (b, (yr, r0r)) in y.ranks.iter().zip(&r0).enumerate() {
        assert_eq!(yr, r0r, "bond {b} ranks conserved");
    }
}

#[test]
fn pg3_right_orthogonalize_keeprank_conserves_ranks() {
    let (_pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_004);
    let ranks = capped_ranks(&space, 1, 2, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let xd = x.to_dense();
    let n0 = x.norm();
    let r0: Vec<RankGrid> = x.ranks.clone();

    let mut y = x.clone();
    y.right_orthogonalize(true);
    assert!(y.check());

    assert!(
        rel_err(&y.to_dense(), &xd) < 1e-13,
        "keeprank gauge equality"
    );
    assert!(
        (y.norm() - n0).abs() <= 1e-10 * n0,
        "ortho-fast-path norm exact after keeprank right-orthogonalize"
    );
    for (b, (yr, r0r)) in y.ranks.iter().zip(&r0).enumerate() {
        assert_eq!(yr, r0r, "bond {b} ranks conserved");
    }
}

#[test]
fn pg3_right_orthogonalize_revealing_c64() {
    let (pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_303_006);
    let ranks = naive_ranks(&space, 0, 3, &mut rng);
    let x: TtVector<C64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let xd = x.to_dense();
    let n0 = dense_norm(&xd);

    let mut y = x.clone();
    y.right_orthogonalize(false);
    assert!(y.check());

    assert!(
        rel_err(&y.to_dense(), &xd) < 1e-13,
        "complex gauge equality"
    );
    assert_eq!(y.ortho, Some(0));
    let (dev, checked) = core_right_orthonormality_dev(&y);
    assert!(dev < 1e-12, "complex right-orthonormal (Q·Qᴴ=I), dev={dev}");
    assert!(checked > 0);
    assert!(
        (y.norm() - n0).abs() <= 1e-10 * n0,
        "ortho-fast-path norm ≈ dense norm on a right-canonical complex train"
    );
    assert_pure(&pg, &space, &y);
}

#[test]
fn pg3_move_core_roundtrip_positions() {
    let (pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_007);
    let ranks = capped_ranks(&space, 1, 2, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let xd = x.to_dense();
    let d = space.d;

    for pos in 0..d {
        let mut y = x.clone();
        y.move_core(pos, false);
        assert!(y.check());
        assert_eq!(y.ortho, Some(pos), "center at {pos}");
        assert!(
            rel_err(&y.to_dense(), &xd) < 1e-12,
            "gauge invariance at center {pos}"
        );
        assert_pure(&pg, &space, &y);
    }

    let mut y = x.clone();
    for pos in 0..d {
        y.move_core(pos, false);
        assert!(rel_err(&y.to_dense(), &xd) < 1e-12, "up-sweep to {pos}");
    }
    for pos in (0..d).rev() {
        y.move_core(pos, false);
        assert!(rel_err(&y.to_dense(), &xd) < 1e-12, "down-sweep to {pos}");
    }
}
