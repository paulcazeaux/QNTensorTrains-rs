// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul_with_conj;
use faer::{Accum, Conj, Mat, MatRef, Par};

use super::equal_cuts;
use crate::C64;
use crate::scalar::Scalar;
use crate::test_util::seeded;
use rand::Rng;

fn randm<T: Scalar>(r: usize, c: usize, rng: &mut impl Rng) -> Mat<T> {
    Mat::from_fn(r, c, |_, _| T::sample_standard(rng))
}

fn compare<T: Scalar>(whole: MatRef<'_, T>, got: MatRef<'_, T>) -> (bool, f64) {
    let scale = (0..whole.ncols())
        .flat_map(|j| (0..whole.nrows()).map(move |i| (i, j)))
        .map(|(i, j)| whole.get(i, j).abs_s())
        .fold(0.0f64, f64::max);
    let mut bitwise = true;
    let mut worst = 0.0f64;
    for j in 0..whole.ncols() {
        for i in 0..whole.nrows() {
            let (x, y) = (*whole.get(i, j), *got.get(i, j));
            for (a, b) in [(x.re_s(), y.re_s()), (x.im_s(), y.im_s())] {
                if a.to_bits() != b.to_bits() {
                    bitwise = false;
                    let ulp = f64::EPSILON * a.abs().max(f64::MIN_POSITIVE);
                    let unit = ulp.max(1e-14 * scale / 4.0);
                    worst = worst.max((a - b).abs() / unit);
                }
            }
        }
    }
    (bitwise, worst)
}

#[allow(clippy::too_many_arguments)]
fn sliced_matches<T: Scalar>(
    m: usize,
    n: usize,
    k: usize,
    ns_col: usize,
    ns_row: usize,
    tl: bool,
    tr: bool,
    accum: Accum,
    rng: &mut impl Rng,
) -> ((bool, f64), (bool, f64)) {
    let a_st: Mat<T> = if tl {
        randm(k, m, rng)
    } else {
        randm(m, k, rng)
    };
    let b_st: Mat<T> = if tr {
        randm(n, k, rng)
    } else {
        randm(k, n, rng)
    };
    let (a, ca) = if tl {
        (a_st.as_ref().transpose(), Conj::Yes)
    } else {
        (a_st.as_ref(), Conj::No)
    };
    let b = if tr {
        b_st.as_ref().transpose()
    } else {
        b_st.as_ref()
    };
    let d0: Mat<T> = randm(m, n, rng);
    let alpha = T::from_complex_f64(0.75, -0.3);
    let mut whole = d0.clone();
    matmul_with_conj(whole.as_mut(), accum, a, ca, b, Conj::No, alpha, Par::Seq);

    let mut cols = d0.clone();
    for c in equal_cuts(n, ns_col) {
        matmul_with_conj(
            cols.as_mut().submatrix_mut(0, c.start, m, c.len()),
            accum,
            a,
            ca,
            b.submatrix(0, c.start, k, c.len()),
            Conj::No,
            alpha,
            Par::Seq,
        );
    }
    let mut rows = d0.clone();
    for c in equal_cuts(m, ns_row) {
        matmul_with_conj(
            rows.as_mut().submatrix_mut(c.start, 0, c.len(), n),
            accum,
            a.submatrix(c.start, 0, c.len(), k),
            ca,
            b,
            Conj::No,
            alpha,
            Par::Seq,
        );
    }
    (
        compare(whole.as_ref(), cols.as_ref()),
        compare(whole.as_ref(), rows.as_ref()),
    )
}

fn nano(fixed: usize, k: usize, w: usize) -> bool {
    fixed * k * w <= 16 * 16 * 16
}

fn sweep<T: Scalar>(label: &str) {
    let mut rng = seeded(0x005A_1CE5);
    let dims = [2usize, 3, 5, 8, 17, 33, 64, 257];
    let ks = [2usize, 3, 8, 17, 64, 130, 600];
    let mut tally = [[0usize; 2]; 2];
    let mut worst = 0.0f64;
    let mut case = 0usize;
    for &m in &dims {
        for &n in &dims {
            for &k in &ks {
                for s in 2..=4usize {
                    if n / s < 2 && m / s < 2 {
                        continue;
                    }
                    case += 1;
                    let (tl, tr) = (case % 2 == 1, case % 3 == 1);
                    let accum = if case % 5 < 3 {
                        Accum::Add
                    } else {
                        Accum::Replace
                    };
                    let (sc, sr) = sliced_matches::<T>(m, n, k, s, s, tl, tr, accum, &mut rng);
                    for ((bitwise, units), axis, full, fixed) in
                        [(sc, "column", n, m), (sr, "row", m, n)]
                    {
                        if full / s < 2 {
                            continue;
                        }
                        assert!(
                            units <= 4.0,
                            "{label}: {axis} slices ×{s} of m={m} n={n} k={k} (tl={tl}, \
                             tr={tr}, {accum:?}) off by {units:.2} units (> 4 ULP / 1e-14 rel)"
                        );
                        worst = worst.max(units);
                        let side = usize::from(nano(fixed, k, full) != nano(fixed, k, full / s));
                        tally[side][0] += 1;
                        tally[side][1] += usize::from(!bitwise);
                    }
                }
            }
        }
    }
    println!(
        "{label}: all {} slicings within 4 ULP / 1e-14 rel (worst {worst:.2} units); bitwise \
         different on this ISA: same nano-gemm side {}/{}, crossing {}/{}",
        tally[0][0] + tally[1][0],
        tally[0][1],
        tally[0][0],
        tally[1][1],
        tally[1][0]
    );
    assert!(
        tally[0][0] + tally[1][0] > 1000,
        "{label}: the grid exercised too few slicings"
    );
}

#[test]
fn gemm_slices_match_whole_to_machine_precision_f64() {
    sweep::<f64>("f64");
}

#[test]
fn gemm_slices_match_whole_to_machine_precision_c64() {
    sweep::<C64>("c64");
}
