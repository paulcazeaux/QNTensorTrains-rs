// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5b_t1_tdvp_singlet_ground() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b20);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 50,
        ..Default::default()
    };
    let (lambda, x) = tdvp(&h, &x0, &opts);
    assert!(x.check());
    assert!(
        (lambda - (-2.0)).abs() <= 1e-4,
        "tdvp singlet energy {lambda}"
    );
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "tdvp singlet residual {resid}");
    assert_eq!(
        x.ortho,
        Some(0),
        "tdvp must return center-0 (right-orthogonal)"
    );
    assert!(
        (TtState::norm(&x) - 1.0).abs() < 1e-8,
        "tdvp must return a unit-norm state, got {}",
        TtState::norm(&x)
    );
    let (lambda2, _) = tdvp(&h, &x0, &opts);
    assert_eq!(lambda, lambda2, "tdvp is not deterministic on a rerun");
}

#[test]
fn s5b_t2_tdvp_hb_ground() {
    let space = Su2Space::new(6, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0x5b21);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 50,
        ..Default::default()
    };
    let (lambda, x) = tdvp(&h, &x0, &opts);
    assert!(x.check());
    assert!((lambda - (-0.5)).abs() <= 1e-4, "tdvp HB energy {lambda}");
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "tdvp HB residual {resid}");
}

#[test]
fn s5b_t3_tdvp_weighted_path_polarized() {
    let space = Su2Space::new(6, 2, 2);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b22);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 50,
        ..Default::default()
    };
    let (lambda, x) = tdvp(&h, &x0, &opts);
    assert!(x.check());
    assert!(
        (lambda - (-1.0)).abs() <= 1e-4,
        "tdvp polarized energy {lambda}"
    );
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "tdvp polarized residual {resid}");
    let (n_theta, n_cw) = count_theta_cw(&x);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight trivial (n_theta={n_theta}, n_cw={n_cw})"
    );
}

#[test]
fn s5b_t4_tdvp_monotone_energy_decrease() {
    let space = Su2Space::new(6, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0x5b23);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 5, 10, &mut rng);

    let opts2 = TdvpOptions {
        dtau: 0.2,
        eps: 0.0,
        max_sweeps: 2,
        ..Default::default()
    };
    let (lambda2, _) = tdvp(&h, &x0, &opts2);

    let opts8 = TdvpOptions {
        dtau: 0.2,
        eps: 0.0,
        max_sweeps: 8,
        ..Default::default()
    };
    let (lambda8, _) = tdvp(&h, &x0, &opts8);

    let opts_inf = TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 50,
        ..Default::default()
    };
    let (lambda_inf, _) = tdvp(&h, &x0, &opts_inf);

    assert!(
        lambda8 <= lambda2 + 1e-10,
        "lambda8={lambda8} > lambda2={lambda2}"
    );
    assert!(
        lambda_inf <= lambda8 + 1e-10,
        "lambda_inf={lambda_inf} > lambda8={lambda8}"
    );
    assert!(
        (lambda_inf - (-0.5)).abs() < 1e-4,
        "lambda_inf={lambda_inf} != -0.5"
    );
    assert!(
        lambda2 > lambda_inf + 1e-3,
        "non-vacuity: lambda2={lambda2} should not already be converged to lambda_inf={lambda_inf}"
    );
}

#[test]
fn s5b_t5_tdvp_complex_onebody() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2_c64(space, 1, 3, 0.9);
    let opts = TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 40,
        ..Default::default()
    };
    let mut ok = false;
    let mut last_lambda = f64::INFINITY;
    for trial in 0..3u64 {
        let mut rng = seeded(0x5b24 + trial);
        let x0: Su2TtVector<C64> = randomized_su2_state(space, 5, 10, &mut rng);
        let (lambda, x) = tdvp(&h, &x0, &opts);
        last_lambda = lambda;
        if (lambda - (-2.0)).abs() <= 1e-4 {
            let hx = TtHamiltonian::matvec(&h, &x);
            let r = TtState::sub(&hx, &TtState::scaled(&x, C64::from_f64(lambda)));
            let resid = TtState::norm(&r) / TtState::norm(&x).max(1e-30);
            assert!(resid <= 1e-4, "tdvp complex residual {resid}");
            ok = true;
            break;
        }
    }
    assert!(ok, "tdvp complex onebody: last energy {last_lambda}");
}

#[test]
fn s5b_r1_tdvp_realtime_global_phase() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2_c64(space, 1, 3, 0.9);

    let mut lambda = f64::INFINITY;
    let mut x0 = qntt::su2::vector::tt_zeros::<C64>(space);
    for trial in 0..5u64 {
        let mut rng = seeded(0x5b30 + trial);
        let start: Su2TtVector<C64> = randomized_su2_state(space, 2, 4, &mut rng);
        let (l, xg) = als(
            &h,
            &start,
            &AlsOptions {
                eps: 1e-8,
                ..Default::default()
            },
        );
        lambda = l.re_s();
        x0 = xg;
        if (lambda - (-2.0)).abs() < 1e-4 {
            break;
        }
    }
    assert!((lambda - (-2.0)).abs() < 1e-4, "als ground energy {lambda}");
    assert!(x0.check());
    let n0 = TtState::norm(&x0);

    let opts = RealtimeOptions {
        exptol: 1e-12,
        trunc: 1e-10,
        ..RealtimeOptions::new(0.05, 10)
    };

    let mut norms: Vec<f64> = Vec::new();
    let mut energies: Vec<C64> = Vec::new();
    let mut ncalls = 0usize;
    let (t_final, x) = {
        let mut obs = |_step: usize, _t: f64, xc: &Su2TtVector<C64>| {
            ncalls += 1;
            norms.push(TtState::norm(xc));
            energies.push(qntt::su2::hamiltonian::rayleigh_quotient(&h, xc));
        };
        tdvp_realtime(&h, &x0, &opts, Some(&mut obs))
    };
    assert!(x.check());

    assert_eq!(
        ncalls,
        opts.n_steps + 1,
        "observer fired at step 0..n_steps"
    );
    assert!((t_final - 0.5).abs() < 1e-9, "accumulated time {t_final}");
    for &nrm in &norms {
        assert!(
            (nrm - n0).abs() < 1e-6 * n0.max(1.0),
            "norm not conserved: {nrm} vs {n0}"
        );
    }
    for &e in &energies {
        assert!(
            (e.re_s() - lambda).abs() < 1e-6,
            "Re<H> drift: {} vs {lambda}",
            e.re_s()
        );
        assert!(e.im_s().abs() < 1e-8, "Im<H> should vanish: {}", e.im_s());
    }

    let phase = C64::new((-lambda * t_final).cos(), (-lambda * t_final).sin());
    let expected = TtState::scaled(&x0, phase);
    let diff = TtState::sub(&x, &expected);
    let rel = TtState::norm(&diff) / n0.max(1e-30);
    assert!(rel < 1e-6, "global phase oracle: relative deviation {rel}");
}

#[test]
#[should_panic(expected = "complex")]
fn s5b_r2_tdvp_realtime_real_scalar_panics() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b31);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = RealtimeOptions::new(0.05, 4);
    let _ = tdvp_realtime(&h, &x0, &opts, None);
}
