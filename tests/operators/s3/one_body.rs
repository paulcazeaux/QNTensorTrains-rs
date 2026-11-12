// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s3::*;

#[test]
fn s2_b4_rayleigh_quotient_and_xt_h_y_match_abelian() {
    let d = 5;
    let n = 4;
    let two_s = 2;
    let space = Su2Space::new(d, n, two_s);
    let mut rng = seeded(777);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let y: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let t = randn_matrix::<f64, _>(d, &mut rng);
    let v = randn_4tensor::<f64, _>(d, &mut rng);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let ab_space = QnSpace::from_n_two_sz(d, n, two_s as i64);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, f64::EPSILON);
    let x_exp: TtVector<f64> = expand(&x, two_s as i64);
    let y_exp: TtVector<f64> = expand(&y, two_s as i64);

    let rq_su2 = su2_rayleigh_quotient(&h, &x);
    let rq_ab = qntt::rayleigh_quotient(&h_ab, &x_exp);
    assert!((rq_su2 - rq_ab).abs() <= 1e-10 * rq_ab.abs().max(1.0));

    let xthy_su2 = su2_xt_h_y(&x, &h, &y);
    let hy_ab = &h_ab * &y_exp;
    let xthy_ab = ab_dot(&x_exp, &hy_ab);
    assert!((xthy_su2 - xthy_ab).abs() <= 1e-10 * xthy_ab.abs().max(1.0));
}
