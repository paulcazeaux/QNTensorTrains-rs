// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase3::*;

#[test]
fn right_orthogonalize_is_exact_gauge_c64() {
    let space = QnSpace::from_n_two_sz(8, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_022);
    let x: TtVector<C64> = randomized_state(space, 0, 6, &mut rng);
    let xd = x.to_dense();
    let n0 = dense_norm(&xd);

    let mut y = x.clone();
    y.right_orthogonalize(false);
    y.check();

    assert!(
        rel_err(&y.to_dense(), &xd) < 1e-13,
        "complex gauge equality"
    );
    assert_eq!(y.ortho, Some(0));
    assert!(
        core_right_orthonormality_dev(&y) < 1e-12,
        "complex right-orthonormal (Q·Qᴴ = I)"
    );
    assert!(
        (y.norm() - n0).abs() <= 1e-10 * n0,
        "ortho-fast-path norm ≈ dense norm on a right-canonical complex train"
    );
}

#[test]
fn round_keep_one_fallback_nonzero() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(103_033);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let nx = x.norm();

    let y = x.round(10.0 * nx, qntt::Pooled::NONE, RoundMode::Abs);
    y.check();
    for b in 1..space.d {
        assert!(
            y.ranks[b].total() <= 1,
            "bond {b} keeps at most one direction (keep-one fallback)"
        );
    }
    assert!(
        dense_diff_norm(&y.to_dense(), &x.to_dense()) <= 10.0 * nx,
        "keep-one result stays within ε"
    );
}

#[test]
fn d1_special_cases() {
    let space = QnSpace::new(1, 1, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_718_023);
    let x: TtVector<C64> = randomized_state(space, 1, 1, &mut rng);
    let xd = x.to_dense();
    let n0 = dense_norm(&xd);
    assert!(n0 > 0.0, "single-site train is nonzero");

    let y = x.round(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
    y.check();
    assert!(
        rel_err(&y.to_dense(), &xd) < 1e-12,
        "d=1 round preserves the tensor"
    );

    assert!((x.norm_lr(false) - n0).abs() <= 1e-10 * n0, "d=1 norm_lr");
    assert!((x.norm_rl(false) - n0).abs() <= 1e-10 * n0, "d=1 norm_rl");

    let re = x.real_part().to_dense();
    let im = x.imag_part().to_dense();
    let want_re: Vec<f64> = xd.iter().map(|z| z.re_s()).collect();
    let want_im: Vec<f64> = xd.iter().map(|z| z.im_s()).collect();
    let got_re: Vec<f64> = re.iter().map(|z| z.re_s()).collect();
    let got_im: Vec<f64> = im.iter().map(|z| z.re_s()).collect();
    assert!(dense_diff_norm(&got_re, &want_re) < 1e-13, "d=1 real_part");
    assert!(dense_diff_norm(&got_im, &want_im) < 1e-13, "d=1 imag_part");
    assert!(
        re.iter().chain(&im).all(|z| z.im_s().abs() < 1e-13),
        "d=1 real/imag outputs are real-valued"
    );
}
