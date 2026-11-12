// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5a_e2_contracted_coeff_order_pin() {
    let mut rng = seeded(0x5a41);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let mut tc = contract(x.core(1), x.core(2));
    let mut idx = 0usize;
    for &m in tc.mid_qn.clone().iter() {
        if let Some(mut b) = tc.block_mut(m) {
            let (nr, nc) = (b.nrows(), b.ncols());
            for j in 0..nc {
                for i in 0..nr {
                    *b.as_mut().get_mut(i, j) = 1000.0 + idx as f64;
                    idx += 1;
                }
            }
        }
    }
    let n = tc.num_coeffs();
    assert_eq!(n, idx, "e2: num_coeffs disagrees with the fill walk");
    for i in 0..n {
        assert_eq!(
            tc.coeff(i),
            1000.0 + i as f64,
            "e2: coeff({i}) order mismatch"
        );
    }

    let canonical: Vec<f64> = (0..n).map(|i| tc.coeff(i)).collect();
    let mut reversed: Vec<f64> = Vec::new();
    for &m in tc.mid_qn.clone().iter().rev() {
        if let Some(b) = tc.block(m) {
            for j in 0..b.ncols() {
                for i in 0..b.nrows() {
                    reversed.push(*b.get(i, j));
                }
            }
        }
    }
    assert_eq!(
        reversed.len(),
        canonical.len(),
        "e2: the two traversals must visit the same coefficients"
    );
    assert_ne!(
        canonical, reversed,
        "e2: VACUOUS — the canonical mid-sector order coincides with the reversed one, so the pin \
         discriminates nothing"
    );

    let mut seen: Vec<f64> = Vec::new();
    tc.for_each_coeff_mut(|v| seen.push(*v));
    assert_eq!(
        seen, canonical,
        "e2: for_each_coeff_mut order/count disagrees with coeff()"
    );

    let mut tc2 = tc.clone();
    tc2.set_coeff(0, -1.0);
    assert_eq!(tc2.coeff(0), -1.0, "e2: set_coeff round-trip failed");
    #[allow(clippy::needless_range_loop)]
    for i in 1..n {
        assert_eq!(
            tc2.coeff(i),
            canonical[i],
            "e2: set_coeff(0) disturbed coefficient {i}"
        );
    }

    let raw_sum: f64 = (0..n).map(|i| tc.coeff(i) * tc.coeff(i)).sum();
    let metric_sum = tc.dot_cores(&tc).re_s();
    assert!(
        (raw_sum - metric_sum).abs() > 1e-3 * raw_sum.max(1.0),
        "e2: dot_cores should diverge from the raw coefficient sum-of-squares at twoS=2 (else the metric split is broken)"
    );
}

#[test]
fn s5a_e3_algebra_consistency_both_implementors() {
    let mut rng = seeded(0x5a42);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let core = x.core(2).clone();
    let n0 = InnerProductVector::norm(&core);
    let n_c = core.num_coeffs();
    let raw2: f64 = (0..n_c).map(|i| core.coeff(i).powi(2)).sum();
    assert!(
        (n0 * n0 - raw2).abs() > 1e-3 * raw2.max(1.0),
        "e3: Su2Core::norm should diverge from the raw coefficient 2-norm at twoS=2"
    );
    let d_self = InnerProductVector::dot(&core, &core).re_s();
    assert!(
        (n0 * n0 - d_self).abs() <= 1e-13 * d_self.abs().max(1.0),
        "e3: Su2Core norm()² must equal Re dot(self,self) ({} vs {d_self})",
        n0 * n0
    );
    let before: Vec<f64> = (0..n_c).map(|i| core.coeff(i)).collect();
    let mut scaled = core.clone();
    InnerProductVector::scale(&mut scaled, 2.0);
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_c {
        assert!(
            (scaled.coeff(i) - 2.0 * before[i]).abs() <= 1e-14 * before[i].abs().max(1.0),
            "e3: Su2Core::scale(2.0) wrong at coefficient {i}"
        );
    }
    let mut core_y = Su2Core::<f64>::zeros(space, core.k, &core.row_ranks, &core.col_ranks);
    core_y.fill_randn(&mut rng);
    let ycoef: Vec<f64> = (0..n_c).map(|i| core_y.coeff(i)).collect();
    let mut axpied = core.clone();
    InnerProductVector::axpy(&mut axpied, -0.75, &core_y);
    for i in 0..n_c {
        let want = before[i] - 0.75 * ycoef[i];
        assert!(
            (axpied.coeff(i) - want).abs() <= 1e-14 * want.abs().max(1.0),
            "e3: Su2Core::axpy wrong at coefficient {i}"
        );
    }

    let tcx = contract(x.core(1), x.core(2));
    let mut tcy = tcx.zeros_like();
    tcy.for_each_coeff_mut(|v| *v = f64::sample_standard(&mut rng));
    let n_t = tcx.num_coeffs();
    let t_before: Vec<f64> = (0..n_t).map(|i| tcx.coeff(i)).collect();
    let t_y: Vec<f64> = (0..n_t).map(|i| tcy.coeff(i)).collect();
    let tn = InnerProductVector::norm(&tcx);
    let t_raw2: f64 = t_before.iter().map(|v| v * v).sum();
    assert!(
        (tn * tn - t_raw2).abs() > 1e-3 * t_raw2.max(1.0),
        "e3: Su2ContractedCores::norm should diverge from the raw coefficient 2-norm at twoS=2"
    );
    let t_dself = InnerProductVector::dot(&tcx, &tcx).re_s();
    assert!(
        (tn * tn - t_dself).abs() <= 1e-13 * t_dself.abs().max(1.0),
        "e3: Su2ContractedCores norm()² must equal Re dot(self,self)"
    );
    let mut t_scaled = tcx.clone();
    InnerProductVector::scale(&mut t_scaled, 2.0);
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_t {
        assert!(
            (t_scaled.coeff(i) - 2.0 * t_before[i]).abs() <= 1e-14 * t_before[i].abs().max(1.0),
            "e3: Su2ContractedCores::scale(2.0) wrong at coefficient {i}"
        );
    }
    let mut t_axpied = tcx.clone();
    InnerProductVector::axpy(&mut t_axpied, -0.75, &tcy);
    for i in 0..n_t {
        let want = t_before[i] - 0.75 * t_y[i];
        assert!(
            (t_axpied.coeff(i) - want).abs() <= 1e-14 * want.abs().max(1.0),
            "e3: Su2ContractedCores::axpy wrong at coefficient {i}"
        );
    }
}

#[test]
fn s5a_e3c_dot_conjugate_symmetry_complex() {
    let mut rng = seeded(0x5a43);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<C64> = randomized_su2_state(space, 2, 4, &mut rng);

    let cx = x.core(2).clone();
    let mut cy = Su2Core::<C64>::zeros(space, cx.k, &cx.row_ranks, &cx.col_ranks);
    cy.fill_randn(&mut rng);
    let d_xy = InnerProductVector::dot(&cx, &cy);
    let d_yx = InnerProductVector::dot(&cy, &cx);
    assert!(
        d_xy.im.abs() > 1e-6,
        "e3c: vacuous — the complex one-site dot has no imaginary part to conjugate"
    );
    assert!(
        (d_xy - d_yx.conj_s()).abs_s() <= 1e-13 * d_xy.abs_s().max(1.0),
        "e3c: Su2Core dot must satisfy dot(x,y) == conj(dot(y,x)) ({d_xy} vs {d_yx})"
    );
    assert!(
        InnerProductVector::dot(&cx, &cx).im.abs() <= 1e-13,
        "e3c: dot(x,x) must be real"
    );

    let tcx = contract(x.core(1), x.core(2));
    let mut tcy = tcx.zeros_like();
    tcy.for_each_coeff_mut(|v| *v = C64::sample_standard(&mut rng));
    let t_xy = InnerProductVector::dot(&tcx, &tcy);
    let t_yx = InnerProductVector::dot(&tcy, &tcx);
    assert!(
        t_xy.im.abs() > 1e-6,
        "e3c: vacuous — the complex two-site dot has no imaginary part to conjugate"
    );
    assert!(
        (t_xy - t_yx.conj_s()).abs_s() <= 1e-13 * t_xy.abs_s().max(1.0),
        "e3c: Su2ContractedCores dot must satisfy dot(x,y) == conj(dot(y,x))"
    );
    assert!(
        InnerProductVector::dot(&tcx, &tcx).im.abs() <= 1e-13,
        "e3c: two-site dot(x,x) must be real"
    );
}
