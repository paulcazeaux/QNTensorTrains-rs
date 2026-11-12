// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5b_a1_als_singlet_ground() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b01);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = AlsOptions {
        eps: 1e-6,
        ..Default::default()
    };
    let (lambda, x) = als(&h, &x0, &opts);
    assert!(x.check(), "als result fails check()");
    let rq = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x);
    assert!((rq - (-2.0)).abs() <= 1e-4, "als singlet energy {rq}");
    assert!(
        (lambda - rq).abs() <= 1e-4,
        "driver-reported λ={lambda} vs independent RQ={rq}"
    );
    let resid = global_eigen_residual(&h, &x, rq);
    assert!(resid <= 1e-4, "als singlet residual {resid}");

    let (lambda2, _) = als(&h, &x0, &opts);
    assert_eq!(lambda, lambda2, "als is not deterministic on a rerun");
}

#[test]
fn s5b_a2_als_hb_ground() {
    let space = Su2Space::new(6, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0x5b02);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 3, 6, &mut rng);
    let opts = AlsOptions {
        eps: 1e-6,
        ..Default::default()
    };
    let (_lambda, x) = als(&h, &x0, &opts);
    assert!(x.check());
    let rq = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x);
    assert!((rq - (-0.5)).abs() <= 1e-4, "als HB energy {rq}");
    let resid = global_eigen_residual(&h, &x, rq);
    assert!(resid <= 1e-4, "als HB residual {resid}");
}

#[test]
fn s5b_a3_als_weighted_path_polarized() {
    let space = Su2Space::new(6, 2, 2);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b03);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = AlsOptions {
        eps: 1e-6,
        ..Default::default()
    };
    let (_lambda, x) = als(&h, &x0, &opts);
    assert!(x.check());
    let rq = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x);
    assert!((rq - (-1.0)).abs() <= 1e-4, "als polarized energy {rq}");
    let resid = global_eigen_residual(&h, &x, rq);
    assert!(resid <= 1e-4, "als polarized residual {resid}");
    let (n_theta, n_cw) = count_theta_cw(&x);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight are trivial at this size (n_theta={n_theta}, n_cw={n_cw})"
    );
}

#[test]
fn s5b_a4_als_inner_tol_schedule_no_panic() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b04);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = AlsOptions {
        eps: 1e-3,
        ..Default::default()
    };
    let (_lambda, x) = als(&h, &x0, &opts);
    let rq = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x);
    assert!((rq - (-2.0)).abs() <= 1e-3, "als loose-eps energy {rq}");
}

#[test]
fn s5b_a5_als_tight_schedule_env_reuse() {
    let space = Su2Space::new(6, 2, 0);
    let h = hb_su2(space);
    let opts = AlsOptions {
        eps: 1e-6,
        ..Default::default()
    };
    let mut ok = false;
    let mut last_rq = f64::INFINITY;
    let mut winner: Option<(Su2TtVector<f64>, f64)> = None;
    for trial in 0..3u64 {
        let mut rng = seeded(0x5b05 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 5, 10, &mut rng);
        let (_lambda, x) = als(&h, &x0, &opts);
        let rq = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x);
        last_rq = rq;
        if (rq - (-0.5)).abs() <= 1e-4 {
            ok = true;
            winner = Some((x, rq));
            break;
        }
    }
    assert!(ok, "als tight-schedule env reuse: last energy {last_rq}");
    let (x, rq) = winner.expect("ok implies a winner");
    assert!(
        x.check(),
        "als tight-schedule env reuse result fails check()"
    );
    let resid = global_eigen_residual(&h, &x, rq);
    assert!(
        resid <= 1e-4,
        "als tight-schedule env reuse residual {resid}"
    );
}
