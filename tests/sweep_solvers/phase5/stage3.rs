// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase5::*;

fn state_from_sites<T: Scalar>(d: usize, up_sites: &[usize], dn_sites: &[usize]) -> TtVector<T> {
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in up_sites {
        occ_up[s] = true;
    }
    for &s in dn_sites {
        occ_dn[s] = true;
    }
    tt_state(&occ_up, &occ_dn)
}

#[test]
fn t3_als_onebody_polarized_ground() {
    let d = 6usize;
    let pairs: [(usize, usize); 8] = [
        (0, 0),
        (0, 1),
        (0, 5),
        (1, 4),
        (2, 3),
        (3, 5),
        (1, 3),
        (2, 5),
    ];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let space = QnSpace::from_n_two_sz(d, 2, 2);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[i, j], -1.0);
        t.set(&[j, i], -1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A10 + idx as u64);
        let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        x0.check();
        let opts = AlsOptions {
            eps: 1e-6,
            max_iter: 20,
            ..Default::default()
        };
        let (lambda, x) = als(&h, &x0, &opts);
        assert!(
            (lambda + 1.0).abs() < 1e-4,
            "pair ({i},{j}): energy {lambda}"
        );
        let hx = &h * &x;
        let xl = &x * lambda;
        let resid = &hx - &xl;
        assert!(resid.norm_lr(false) < 1e-4, "pair ({i},{j}): residual");
    }
}

#[test]
fn t3_als_twobody_span_projection() {
    let d = 6usize;
    let cases: [((usize, usize, usize, usize), f64); 3] = [
        ((0, 1, 2, 3), -0.5),
        ((0, 0, 2, 2), -1.0),
        ((0, 0, 2, 3), -std::f64::consts::FRAC_1_SQRT_2),
    ];
    for (ci, &((a, b, c, e), analytic)) in cases.iter().enumerate() {
        let space = QnSpace::new(d, 1, 1);
        let t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[c, a, e, b], -1.0);
        v.set(&[b, e, a, c], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut span: Vec<(usize, usize)> = vec![(a, b), (c, e), (b, a), (e, c)];
        span.sort_unstable();
        span.dedup();
        let zs: Vec<TtVector<f64>> = span
            .iter()
            .map(|&(u, w)| state_from_sites(d, &[u], &[w]))
            .collect();

        let mut lambda = f64::INFINITY;
        let mut x = tt_zeros::<f64>(space);
        for trial in 0..4u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A20 + ci as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
            x0.check();
            let opts = AlsOptions {
                eps: 1e-6,
                max_iter: 20,
                ..Default::default()
            };
            let (l, xt) = als(&h, &x0, &opts);
            lambda = l;
            x = xt;
            if (lambda - analytic).abs() < 1e-4 {
                break;
            }
        }

        assert!(lambda <= -0.5 + 1e-4, "case {ci}: energy bound {lambda}");
        assert!(
            (lambda - analytic).abs() < 1e-4,
            "case {ci}: analytic energy {lambda} vs {analytic}"
        );
        let proj: f64 = zs
            .iter()
            .map(|z| {
                let ov = dot(&x, z);
                ov.abs_s() * ov.abs_s()
            })
            .sum();
        assert!(
            (proj - 1.0).abs() < 1e-4,
            "case {ci}: span projection {proj}"
        );
        let hx = &h * &x;
        let xl = &x * lambda;
        let resid = &hx - &xl;
        assert!(resid.norm_lr(false) < 1e-4, "case {ci}: residual");
    }
}

#[test]
fn t3_als_tight_schedule_env_reuse() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 4, 1], -1.0);
    v.set(&[1, 4, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut lambda = f64::INFINITY;
    let mut x = tt_zeros::<f64>(space);
    for trial in 0..3u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A30 + trial);
        let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        x0.check();
        let opts = AlsOptions {
            eps: 1e-6,
            max_iter: 20,
            ..Default::default()
        };
        let (l, xt) = als(&h, &x0, &opts);
        lambda = l;
        x = xt;
        if lambda <= -0.5 + 1e-4 {
            break;
        }
    }
    assert!(
        lambda <= -0.5 + 1e-4,
        "tight-schedule energy bound {lambda}"
    );
    let hx = &h * &x;
    let xl = &x * lambda;
    let resid = &hx - &xl;
    assert!(resid.norm_lr(false) < 1e-4, "tight-schedule residual");
}

#[test]
fn t3_als_sweep_pair_matches_one_als_iteration() {
    use qntt::solvers::als::als_sweep_pair;
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A70);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (la, xa) = als(
        &h,
        &x0,
        &AlsOptions {
            eps: 1e-4,
            max_iter: 1,
            ..Default::default()
        },
    );
    let mut xb = x0.clone();
    let lb = als_sweep_pair(&h, &mut xb, 1e-4);
    assert!(
        (la - lb).abs() <= 1e-13 * la.abs(),
        "λ: als {la} vs pair {lb}"
    );
    let ov = dot(&xa, &xb).abs_s() / (xa.norm() * xb.norm());
    assert!(
        1.0 - ov <= 1e-12,
        "states differ: 1 − |⟨a,b⟩| = {}",
        1.0 - ov
    );
}

#[test]
fn t3_als_sweep_pair_invariants() {
    use qntt::solvers::als::als_sweep_pair;
    use qntt::solvers::riemannian::{normalize_mut, riemannian_gradient, tangent_bases};
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A71);
    let mut x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    normalize_mut(&mut x0);
    let rq0 = rayleigh_quotient(&h, &x0);
    let ranks0 = x0.ranks.clone();
    let mut x = x0.clone();
    let lambda = als_sweep_pair(&h, &mut x, 1e-4);
    let rq = rayleigh_quotient(&h, &x) / x.norm().powi(2);
    assert!(
        (lambda - rq).abs() <= 1e-12 * lambda.abs(),
        "λ {lambda} is not the Rayleigh quotient {rq}"
    );
    assert!(lambda <= rq0 + 1e-12, "not variational: {lambda} > {rq0}");
    assert_eq!(x.ranks, ranks0, "ranks changed");
    assert_eq!(x.ortho, Some(0), "not center-0 on exit");

    let space = QnSpace::from_n_two_sz(d, 2, 2);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[0, 1], -1.0);
    t.set(&[1, 0], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A72);
    let mut y: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    normalize_mut(&mut y);
    let mut lambda = 0.0;
    for _ in 0..30 {
        lambda = als_sweep_pair(&h, &mut y, 1e-10);
    }
    normalize_mut(&mut y);
    let (bl, br) = tangent_bases(&y);
    let (g, _) = riemannian_gradient(&h, &y, &bl, &br, 1e-12, 1234);
    let rel = g.norm() / lambda.abs();
    println!("t3_als_sweep_pair_invariants: ‖g‖/|λ| after 30 pairs at 1e-10 = {rel:.3e}");
    assert!(
        rel < 1e-8,
        "caller tolerance not honoured: ‖g‖/|λ| = {rel:.3e}"
    );
}
