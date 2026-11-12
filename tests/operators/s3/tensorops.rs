// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s3::*;

#[test]
fn s5_c1_spin_summed_1rdm_vs_abelian() {
    let d = 6;
    let n = 2;
    let two_s = 0;
    let space = Su2Space::new(d, n, two_s);
    let mut rng = seeded(733);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    for &(i, j) in &[(2usize, 2usize), (1, 4), (4, 0), (3, 3)] {
        let y = su2_adag_i_a_j(&x, i, j);
        assert!(y.check());
        let two_sz = 0i64;
        let xe: TtVector<f64> = expand(&x, two_sz);
        let yref_up = ab_adag_i_a_j(
            &xe,
            Orbital::new(i, Spin::Up),
            Orbital::new(j, Spin::Up),
            1.0,
        );
        let yref_dn = ab_adag_i_a_j(
            &xe,
            Orbital::new(i, Spin::Dn),
            Orbital::new(j, Spin::Dn),
            1.0,
        );
        let yref = &yref_up + &yref_dn;
        let ye: TtVector<f64> = expand(&y, two_sz);
        let got = ye.to_dense();
        let want = yref.to_dense();
        let err: f64 = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (g - w) * (g - w))
            .sum::<f64>()
            .sqrt();
        let norm_ref: f64 = want.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!(err / norm_ref.max(1.0) <= 1e-10, "C1 ({i},{j}) err={err}");
    }
}

#[test]
fn s5_e2_hermitian_h_real_rayleigh() {
    let d = 5;
    let n = 4;
    let two_s = 2;
    let space = Su2Space::new(d, n, two_s);
    let mut rng = seeded(20908);
    let x: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
    let y: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
    let t0 = randn_matrix::<C64, _>(d, &mut rng);
    let mut t = zeros_matrix::<C64>(d);
    for i in 0..d {
        for j in 0..d {
            let tij = t0.get(&[i, j]);
            let tji = t0.get(&[j, i]);
            t.set(&[i, j], (tij + tji.conj_s()) * C64::from(0.5));
        }
    }
    let v = zeros_4tensor::<C64>(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let rq = su2_rayleigh_quotient(&h, &x);
    assert!(rq.im_s().abs() <= 1e-10, "E2 Im(rayleigh)={}", rq.im_s());
    let xy = su2_xt_h_y(&x, &h, &y);
    let yx = su2_xt_h_y(&y, &h, &x);
    assert!((xy - yx.conj_s()).abs_s() <= 1e-10 * xy.abs_s().max(1.0));
}

#[test]
fn s5_e3_complex_tensorops_adag_i_a_j() {
    let d = 6;
    let n = 2;
    let two_s = 0;
    let space = Su2Space::new(d, n, two_s);
    let mut rng = seeded(739);
    let x: Su2TtVector<C64> = randomized_su2_state(space, 2, 4, &mut rng);
    let y = su2_adag_i_a_j(&x, 1, 4);
    assert!(y.check());
    let two_sz = 0i64;
    let xe: TtVector<C64> = expand(&x, two_sz);
    let yref_up = ab_adag_i_a_j(
        &xe,
        Orbital::new(1, Spin::Up),
        Orbital::new(4, Spin::Up),
        C64::from(1.0),
    );
    let yref_dn = ab_adag_i_a_j(
        &xe,
        Orbital::new(1, Spin::Dn),
        Orbital::new(4, Spin::Dn),
        C64::from(1.0),
    );
    let yref = &yref_up + &yref_dn;
    let ye: TtVector<C64> = expand(&y, two_sz);
    let got = ye.to_dense();
    let want = yref.to_dense();
    let err: f64 = got
        .iter()
        .zip(&want)
        .map(|(g, w)| (*g - *w).abs_s() * (*g - *w).abs_s())
        .sum::<f64>()
        .sqrt();
    assert!(err <= 1e-10, "E3 err={err}");
}
