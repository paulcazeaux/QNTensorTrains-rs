// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use super::gemm_par_hook::{Hook, pool};
use super::{C64, GEMM_PAR_MIN_WORK, Scalar, gemm_par};
use crate::test_util::seeded;

fn rand_mat<T: Scalar>(m: usize, n: usize, rng: &mut impl rand::Rng) -> Mat<T> {
    Mat::from_fn(m, n, |_, _| T::sample_standard(rng))
}

fn max_diff<T: Scalar>(a: &Mat<T>, b: &Mat<T>) -> (f64, f64) {
    let (mut d, mut y) = (0.0f64, 0.0f64);
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            d = d.max((*a.get(i, j) - *b.get(i, j)).abs_s());
            y = y.max(a.get(i, j).abs_s());
        }
    }
    (d, y)
}

fn check<T: Scalar>(label: &str) {
    let mut rng = seeded(0xD32_6E44);
    let shapes = [
        (64, 64, 64, false),
        (100, 100, 100, false),
        (512, 48, 64, false),
        (128, 128, 128, true),
        (192, 192, 192, true),
        (256, 48, 256, true),
        (384, 384, 48, true),
        (96, 96, 384, true),
        (2048, 1, 2048, false),
        (1, 2048, 2048, false),
    ];
    let hook = Hook::observed();
    let p8 = pool(8, &hook);
    for (m, n, k, fires) in shapes {
        assert_eq!(
            m * n * k >= GEMM_PAR_MIN_WORK && m.min(n).min(k) > 1,
            fires,
            "{label} {m}×{n}·{k}"
        );
        let a = rand_mat::<T>(m, k, &mut rng);
        let b = rand_mat::<T>(k, n, &mut rng);
        let c0 = rand_mat::<T>(m, n, &mut rng);
        for accum in [Accum::Replace, Accum::Add] {
            let mut seq = c0.clone();
            matmul(
                seq.as_mut(),
                accum,
                a.as_ref(),
                b.as_ref(),
                T::one(),
                Par::Seq,
            );
            let before = hook.taken();
            let gated = p8.install(|| {
                let mut c = c0.clone();
                matmul(
                    c.as_mut(),
                    accum,
                    a.as_ref(),
                    b.as_ref(),
                    T::one(),
                    gemm_par(m, n, k),
                );
                c
            });
            assert_eq!(
                hook.taken() - before,
                usize::from(fires),
                "{label} {m}×{n}·{k}: gate decision"
            );
            let (d, y) = max_diff(&seq, &gated);
            let ulps = d / (y * f64::EPSILON);
            println!("{label} {m}×{n}·{k} {accum:?}: max|Δ| = {d:.3e} = {ulps:.2} ULP of max|y|");
            if fires {
                assert!(
                    ulps <= 4.0 && d <= 1e-14 * y,
                    "{label} {m}×{n}·{k} {accum:?}: {ulps:.2} ULP > 4"
                );
            } else {
                assert_eq!(
                    d, 0.0,
                    "{label} {m}×{n}·{k} {accum:?}: below the floor, not bitwise"
                );
            }
        }
    }
    let forced = Hook::forced();
    pool(1, &forced).install(|| assert_eq!(gemm_par(384, 384, 384), Par::Seq));
    assert_eq!((forced.calls(), forced.taken()), (1, 0));
}

#[test]
fn gemm_par_matches_seq_f64() {
    check::<f64>("f64");
}

#[test]
fn gemm_par_matches_seq_c64() {
    check::<C64>("c64");
}
