// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s1::util::*;
use crate::s1::*;

#[test]
fn s1_5_core_construction_plus_oplus_rank_sum_and_bilinearity() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5CF5_CF5C);
    let d = 6;
    let x: Su2TtVector<f64> = random_csf(d, 4, 2, &mut rng);
    let y: Su2TtVector<f64> = random_csf(d, 4, 2, &mut rng);
    let z = &x + &y;
    assert!(z.check());

    let sum_x: usize = (1..d).map(|k| x.ranks[k].total()).sum();
    let sum_y: usize = (1..d).map(|k| y.ranks[k].total()).sum();
    let sum_z: usize = (1..d).map(|k| z.ranks[k].total()).sum();
    assert_eq!(sum_z, sum_x + sum_y);

    let dxx = su2_dot(&x, &x);
    let dxy = su2_dot(&x, &y);
    let dyy = su2_dot(&y, &y);
    let dzz = su2_dot(&z, &z);
    assert!((dzz - (dxx + 2.0 * dxy + dyy)).abs() < 1e-14);
}

#[test]
fn sum_linear_combination_and_subtraction_commute_with_expand() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7777_8888);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let y: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let two_sz = space.two_s as i64;
    let ex = expand(&x, two_sz);
    let ey = expand(&y, two_sz);

    let alpha = 2.5f64;
    let combo = &(alpha * &x) + &y;
    let e_combo = expand(&combo, two_sz);
    let want_combo: Vec<f64> = ex
        .to_dense()
        .iter()
        .zip(ey.to_dense())
        .map(|(a, b)| alpha * a + b)
        .collect();
    let scale = want_combo.iter().cloned().fold(1.0, f64::max);
    assert_dense_close(&e_combo.to_dense(), &want_combo, 1e-12 * scale);

    let diff = &x - &y;
    let e_diff = expand(&diff, two_sz);
    let want_diff: Vec<f64> = ex
        .to_dense()
        .iter()
        .zip(ey.to_dense())
        .map(|(a, b)| a - b)
        .collect();
    assert_dense_close(&e_diff.to_dense(), &want_diff, 1e-12 * scale);
}
