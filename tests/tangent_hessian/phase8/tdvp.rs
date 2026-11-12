// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::tdvp::{RealtimeOptions, TdvpOptions, tdvp, tdvp_realtime};
use qntt::{
    C64, CoeffTensor, QnSpace, Scalar, SparseHamiltonian, TtVector, dot, rayleigh_quotient,
    tt_zeros,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t8e_tdvp_onebody() {
    let d = 6usize;
    let pairs: [(usize, usize); 3] = [(0, 1), (1, 4), (2, 5)];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[i, j], -1.0);
        t.set(&[j, i], -1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E00 + idx as u64);
        let x0: TtVector<f64> = randomized_state(space, 4, 6, &mut rng);
        x0.check();
        let opts = TdvpOptions {
            dtau: 5.0,
            eps: 1e-6,
            max_sweeps: 50,
            ..TdvpOptions::default()
        };
        let (lambda, x) = tdvp(&h, &x0, &opts);
        x.check();
        assert!(
            (lambda + 2.0).abs() < 1e-4,
            "pair ({i},{j}): tdvp energy {lambda}"
        );
        let resid = &(&h * &x) - &(&x * lambda);
        assert!(resid.norm_lr(false) < 1e-4, "pair ({i},{j}): tdvp residual");
    }
}

#[test]
fn t8e_tdvp_twobody() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut count = 0u64;
    let mut cases: Vec<(usize, usize, usize, usize)> = Vec::new();
    for a in 0..d {
        for b in 0..d {
            for c in 0..d {
                for e in 0..d {
                    if (a, b) == (c, e) {
                        continue;
                    }
                    count += 1;
                    if count.is_multiple_of(200) {
                        cases.push((a, b, c, e));
                    }
                }
            }
        }
    }
    for (ci, &(a, b, c, e)) in cases.iter().enumerate() {
        let h = two_body_swap_h(space, a, b, c, e);
        let zs = span_states(d, a, b, c, e);

        let mut lambda = f64::INFINITY;
        let mut x = tt_zeros::<f64>(space);
        for trial in 0..3u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E10 + ci as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
            x0.check();
            let opts = TdvpOptions {
                dtau: 5.0,
                eps: 1e-6,
                max_sweeps: 50,
                ..TdvpOptions::default()
            };
            let (lt, xt) = tdvp(&h, &x0, &opts);
            lambda = lt;
            x = xt;
            if lambda <= -0.5 + 1e-4 {
                break;
            }
        }
        x.check();
        assert!(
            lambda <= -0.5 + 1e-4,
            "case {ci}: tdvp energy bound {lambda}"
        );
        let proj: f64 = zs.iter().map(|z| dot(&x, z).abs_s().powi(2)).sum();
        assert!(
            (proj - 1.0).abs() < 1e-4,
            "case {ci}: tdvp span projection {proj}"
        );
        let resid = &(&h * &x) - &(&x * lambda);
        assert!(resid.norm_lr(false) < 1e-4, "case {ci}: tdvp residual");
    }
}

#[test]
fn t8e_tdvp_monotonic() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let h = two_body_swap_h(space, 0, 1, 2, 4);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E20);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    x0.check();

    let opts2 = TdvpOptions {
        dtau: 0.2,
        eps: 0.0,
        max_sweeps: 2,
        ..TdvpOptions::default()
    };
    let (lambda2, _) = tdvp(&h, &x0, &opts2);

    let opts8 = TdvpOptions {
        dtau: 0.2,
        eps: 0.0,
        max_sweeps: 8,
        ..TdvpOptions::default()
    };
    let (lambda8, _) = tdvp(&h, &x0, &opts8);

    let opts_inf = TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 50,
        ..TdvpOptions::default()
    };
    let (lambda_inf, _) = tdvp(&h, &x0, &opts_inf);

    assert!(
        lambda8 <= lambda2 + 1e-10,
        "λ8={lambda8} should be ≤ λ2={lambda2}"
    );
    assert!(
        lambda_inf <= lambda8 + 1e-10,
        "λ∞={lambda_inf} should be ≤ λ8={lambda8}"
    );
    assert!((lambda_inf + 0.5).abs() < 1e-4, "λ∞ = {lambda_inf} != -0.5");
}

#[test]
fn t8e_tdvp_realtime_sanity() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    t.set(&[1, 3], -C64::new(0.9f64.cos(), 0.9f64.sin()));
    t.set(&[3, 1], -C64::new(0.9f64.cos(), -0.9f64.sin()));
    let v = CoeffTensor::<C64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut lambda = f64::INFINITY;
    let mut x0 = tt_zeros::<C64>(space);
    for trial in 0..5u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E30 + trial);
        let start: TtVector<C64> = randomized_state(space, 2, 4, &mut rng);
        start.check();
        let (l, xg) = als(
            &h,
            &start,
            &AlsOptions {
                eps: 1e-8,
                ..AlsOptions::default()
            },
        );
        lambda = l.re_s();
        x0 = xg;
        if (lambda + 2.0).abs() < 1e-4 {
            break;
        }
    }
    assert!((lambda + 2.0).abs() < 1e-4, "ALS ground energy {lambda}");
    x0.check();
    let n0 = x0.norm();

    let opts = RealtimeOptions {
        exptol: 1e-12,
        trunc: 1e-10,
        ..RealtimeOptions::new(0.05, 10)
    };

    let mut norms: Vec<f64> = Vec::new();
    let mut energies: Vec<C64> = Vec::new();
    let mut ncalls = 0usize;
    let (t_final, x) = {
        let mut obs = |_step: usize, _t: f64, xc: &TtVector<C64>| {
            ncalls += 1;
            norms.push(xc.norm());
            energies.push(rayleigh_quotient(&h, xc));
        };
        tdvp_realtime(&h, &x0, &opts, Some(&mut obs))
    };
    x.check();

    assert_eq!(ncalls, opts.n_steps + 1, "observer fired at step 0..nSteps");
    assert!((t_final - 0.5).abs() < 1e-9, "accumulated time {t_final}");
    for &nrm in &norms {
        assert!(
            (nrm - n0).abs() < 1e-6 * n0,
            "norm conservation: {nrm} vs {n0}"
        );
    }
    for e in &energies {
        assert!(
            (e.re_s() - lambda).abs() < 1e-6,
            "energy conservation re {}",
            e.re_s()
        );
        assert!(e.im_s().abs() < 1e-8, "⟨H⟩ real: im {}", e.im_s());
    }
    let phase = C64::new((-lambda * t_final).cos(), (-lambda * t_final).sin());
    let diff = &x - &(&x0 * phase);
    assert!(diff.norm_lr(false) / n0 < 1e-6, "global phase only");
}

#[test]
fn t8e_tdvp_complex_onebody() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    t.set(&[1, 3], -C64::new(0.9f64.cos(), 0.9f64.sin()));
    t.set(&[3, 1], -C64::new(0.9f64.cos(), -0.9f64.sin()));
    let v = CoeffTensor::<C64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut lambda = f64::INFINITY;
    let mut x = tt_zeros::<C64>(space);
    for trial in 0..3u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E40 + trial);
        let x0: TtVector<C64> = randomized_state(space, 5, 10, &mut rng);
        x0.check();
        let opts = TdvpOptions {
            dtau: 5.0,
            eps: 1e-6,
            max_sweeps: 40,
            ..TdvpOptions::default()
        };
        let (l, xt) = tdvp(&h, &x0, &opts);
        lambda = l;
        x = xt;
        if (lambda + 2.0).abs() < 1e-4 {
            break;
        }
    }
    x.check();
    assert!((lambda + 2.0).abs() < 1e-4, "complex tdvp energy {lambda}");
    let resid = &(&h * &x) - &(&x * C64::from_f64(lambda));
    assert!(resid.norm_lr(false) < 1e-4, "complex tdvp residual");
}
