// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::{C64, QnSpace, Scalar, TtVector, dot, hadamard};
use qntt_testkit::pg::config_counts;
use qntt_testkit::pg::digits;
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn dense_dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    let mut s = T::zero();
    for (a, b) in x.iter().zip(y) {
        s += a.conj_s() * *b;
    }
    s
}

fn dense_norm<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt()
}

fn rel_err<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    let mut num = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let diff = *x - *y;
        num += diff.abs_s() * diff.abs_s();
    }
    num.sqrt() / dense_norm(b)
}

#[test]
fn dot_matches_dense_contraction_real() {
    let space = QnSpace::from_n_two_sz(8, 5, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(104);
    let x: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
    let y: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
    x.check();
    y.check();

    let (xd, yd) = (x.to_dense(), y.to_dense());
    for (idx, (&vx, &vy)) in xd.iter().zip(&yd).enumerate() {
        let (nup, ndn) = config_counts(&digits(idx, 8));
        if nup != space.nup || ndn != space.ndn {
            assert_eq!(vx, 0.0, "off-sector config {idx} of x must be zero");
            assert_eq!(vy, 0.0, "off-sector config {idx} of y must be zero");
        }
    }

    let d_tt = dot(&x, &y);
    let d_dense = dense_dot(&xd, &yd);
    assert!(
        (d_tt - d_dense).abs() <= 1e-8 * d_dense.abs(),
        "dot {d_tt} vs dense {d_dense}"
    );
}

#[test]
fn norm_matches_dense_oracle() {
    let space = QnSpace::from_n_two_sz(8, 5, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(55);
    let x: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);
    let n_tt = x.norm();
    let n_dense = dense_norm(&x.to_dense());
    assert!(
        (n_tt - n_dense).abs() <= 1e-14 * n_dense,
        "norm {n_tt} vs dense {n_dense}"
    );
}

#[test]
fn linear_combination_matches_dense() {
    let space = QnSpace::from_n_two_sz(8, 6, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303);
    let x1: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
    let x2: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
    let (alpha, beta) = (1.3, -0.6);

    let ax = alpha * &x1;
    let bx = beta * &x2;
    let z = &ax + &bx;
    z.check();

    let (xd1, xd2) = (x1.to_dense(), x2.to_dense());
    let want: Vec<f64> = xd1
        .iter()
        .zip(&xd2)
        .map(|(&a, &b)| alpha * a + beta * b)
        .collect();
    let err = rel_err(&z.to_dense(), &want);
    assert!(err < 50.0 * 8.0 * f64::EPSILON, "add rel err {err}");
}

#[test]
fn subtraction_matches_dense() {
    let space = QnSpace::from_n_two_sz(8, 6, -2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(404);
    let x1: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
    let x2: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
    let (alpha, beta) = (0.9, 1.4);

    let ax = alpha * &x1;
    let bx = beta * &x2;
    let z = &ax - &bx;
    z.check();

    let (xd1, xd2) = (x1.to_dense(), x2.to_dense());
    let want: Vec<f64> = xd1
        .iter()
        .zip(&xd2)
        .map(|(&a, &b)| alpha * a - beta * b)
        .collect();
    let err = rel_err(&z.to_dense(), &want);
    assert!(err < 50.0 * 8.0 * f64::EPSILON, "sub rel err {err}");
}

#[test]
fn scalar_multiply_matches_dense() {
    let space = QnSpace::from_n_two_sz(8, 6, 4);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(505);
    let x: TtVector<f64> = randomized_state(space, 1, 5, &mut rng);
    let c = std::f64::consts::PI;

    let left = c * &x;
    let right = &x * c;
    let xd = x.to_dense();
    let want: Vec<f64> = xd.iter().map(|&a| c * a).collect();
    let bound = 50.0 * 8.0 * f64::EPSILON;
    assert!(rel_err(&left.to_dense(), &want) < bound, "c*x");
    assert!(rel_err(&right.to_dense(), &want) < bound, "x*c");

    let back = &left / c;
    assert!(rel_err(&back.to_dense(), &xd) < bound, "(c*x)/c");
}

#[test]
fn hadamard_matches_elementwise_dense() {
    let space = QnSpace::from_n_two_sz(8, 7, -1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(606);
    let x1: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);
    let x2: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);

    let z = hadamard(&x1, &x2);
    z.check();
    let (xd1, xd2) = (x1.to_dense(), x2.to_dense());
    let want: Vec<f64> = xd1.iter().zip(&xd2).map(|(&a, &b)| a * b).collect();
    let err = rel_err(&z.to_dense(), &want);
    assert!(err < 50.0 * 8.0 * f64::EPSILON, "hadamard rel err {err}");
}

#[test]
fn complex_linear_combination_matches_dense() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1107);
    let x1: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
    let x2: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
    let alpha = C64::new(0.7, 1.1);
    let beta = C64::new(-1.2, 0.5);

    let ax = alpha * &x1;
    let bx = beta * &x2;
    let z = &ax + &bx;
    z.check();

    let (xd1, xd2) = (x1.to_dense(), x2.to_dense());
    let want: Vec<C64> = xd1
        .iter()
        .zip(&xd2)
        .map(|(&a, &b)| alpha * a + beta * b)
        .collect();
    let err = rel_err(&z.to_dense(), &want);
    assert!(err < 50.0 * 6.0 * f64::EPSILON, "complex add rel err {err}");
}

#[test]
fn division_by_extreme_scalar_is_robust() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1300);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let xd = x.to_dense();
    for &b in &[1e200_f64, 1e-200_f64, -1e160_f64, 2.5_f64] {
        let back = &(&x / b) * b;
        let err = rel_err(&back.to_dense(), &xd);
        assert!(err < 1e-13, "(x/{b})*{b}: rel err {err}");
    }

    let mut rngc = Xoshiro256PlusPlus::seed_from_u64(1301);
    let xc: TtVector<C64> = randomized_state(space, 1, 3, &mut rngc);
    let xcd = xc.to_dense();
    let b = C64::new(1e180, -2e180);
    let back = &(&xc / b) * b;
    let err = rel_err(&back.to_dense(), &xcd);
    assert!(err < 1e-13, "complex (x/b)*b: rel err {err}");
}

#[test]
fn core_dot_abs2_norm_match_dense_across_families() {
    for (i, &(nup, ndn)) in [(0, 0), (1, 0), (0, 1), (1, 1)].iter().enumerate() {
        let space = QnSpace::new(1, nup, ndn);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1400 + i as u64);
        let x: TtVector<C64> = randomized_state(space, 1, 1, &mut rng);
        let y: TtVector<C64> = randomized_state(space, 1, 1, &mut rng);
        let (xd, yd) = (x.to_dense(), y.to_dense());

        let a2 = x.cores[0].abs2();
        let nd = dense_norm(&xd);
        assert!(
            (a2 - nd * nd).abs() <= 1e-13 * (nd * nd).max(1.0),
            "abs2 (nup={nup},ndn={ndn}): {a2} vs {}",
            nd * nd
        );
        assert!(
            (x.cores[0].norm() - nd).abs() <= 1e-14 * nd.max(1.0),
            "norm (nup={nup},ndn={ndn})"
        );

        let got = x.cores[0].dot_cores(&y.cores[0]);
        let want = dense_dot(&xd, &yd);
        assert!(
            (got - want).abs_s() <= 1e-13 * want.abs_s().max(1.0),
            "dot_cores (nup={nup},ndn={ndn}): {got} vs {want}"
        );
    }
}
