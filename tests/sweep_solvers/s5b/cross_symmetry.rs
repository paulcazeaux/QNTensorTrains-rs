// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5b_x1_cross_symmetry_als() {
    let su2_space = Su2Space::new(6, 2, 0);
    let h_su2 = ha_su2(su2_space, 0, 1);
    let mut rng_su2 = seeded(0x5b40);
    let x0_su2: Su2TtVector<f64> = randomized_su2_state(su2_space, 2, 4, &mut rng_su2);
    let su2_opts = AlsOptions {
        eps: 1e-6,
        ..Default::default()
    };
    let (l_su2, x_su2) = als(&h_su2, &x0_su2, &su2_opts);
    let lambda_su2 = qntt::su2::hamiltonian::rayleigh_quotient(&h_su2, &x_su2);
    let (l_su2b, _) = als(&h_su2, &x0_su2, &su2_opts);
    assert_eq!(
        l_su2, l_su2b,
        "su2 als is not deterministic on a rerun (X1)"
    );

    let ab_space = QnSpace::new(6, 1, 1);
    let h_ab = ha_abelian(ab_space, 0, 1);
    let mut rng_ab = seeded(0x5b40 + 1000);
    let x0_ab: TtVector<f64> = randomized_state(ab_space, 2, 4, &mut rng_ab);
    let (lambda_ab, x_ab) = qntt::solvers::als::als(
        &h_ab,
        &x0_ab,
        &qntt::solvers::als::AlsOptions {
            eps: 1e-6,
            ..Default::default()
        },
    );
    assert!(
        (lambda_ab - (-2.0)).abs() <= 1e-4,
        "abelian als energy {lambda_ab}"
    );
    let resid_ab = global_eigen_residual_abelian(&h_ab, &x_ab, lambda_ab);
    assert!(resid_ab <= 1e-4, "abelian als residual {resid_ab}");
    assert!(
        (lambda_su2 - lambda_ab).abs() <= 2e-4,
        "cross-symmetry mismatch: su2={lambda_su2} vs abelian={lambda_ab}"
    );
}

#[test]
fn s5b_x2_cross_symmetry_mals() {
    let su2_space = Su2Space::new(6, 2, 0);
    let h_su2 = hb_su2(su2_space);
    let mut rng_su2 = seeded(0x5b41);
    let x0_su2: Su2TtVector<f64> = randomized_su2_state(su2_space, 1, 2, &mut rng_su2);
    let su2_opts = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![0],
        ..Default::default()
    };
    let (lambda_su2, x_su2) = mals(&h_su2, &x0_su2, &su2_opts);
    assert!(x_su2.check());
    let (lambda_su2b, _) = mals(&h_su2, &x0_su2, &su2_opts);
    assert_eq!(
        lambda_su2, lambda_su2b,
        "su2 mals is not deterministic on a rerun (X2)"
    );

    let ab_space = QnSpace::new(6, 1, 1);
    let h_ab = hb_abelian(ab_space);
    let mut rng_ab = seeded(0x5b41 + 1000);
    let x0_ab: TtVector<f64> = randomized_state(ab_space, 1, 2, &mut rng_ab);
    let (lambda_ab, x_ab) = qntt::solvers::mals::mals(
        &h_ab,
        &x0_ab,
        &qntt::solvers::mals::MalsOptions {
            eps: 1e-6,
            bond_dims: vec![0],
            ..Default::default()
        },
    );
    assert!(
        (lambda_ab - (-0.5)).abs() <= 1e-4,
        "abelian mals energy {lambda_ab}"
    );
    let resid_ab = global_eigen_residual_abelian(&h_ab, &x_ab, lambda_ab);
    assert!(resid_ab <= 1e-4, "abelian mals residual {resid_ab}");
    assert!(
        (lambda_su2 - lambda_ab).abs() <= 2e-4,
        "cross-symmetry mismatch: su2={lambda_su2} vs abelian={lambda_ab}"
    );
}

#[test]
fn s5b_x3_cross_symmetry_tdvp() {
    let su2_space = Su2Space::new(6, 2, 0);
    let h_su2 = ha_su2(su2_space, 0, 1);
    let mut rng_su2 = seeded(0x5b42);
    let x0_su2: Su2TtVector<f64> = randomized_su2_state(su2_space, 2, 4, &mut rng_su2);
    let opts_su2 = TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 50,
        ..Default::default()
    };
    let (lambda_su2, _) = tdvp(&h_su2, &x0_su2, &opts_su2);
    let (lambda_su2b, _) = tdvp(&h_su2, &x0_su2, &opts_su2);
    assert_eq!(
        lambda_su2, lambda_su2b,
        "su2 tdvp is not deterministic on a rerun (X3)"
    );

    let ab_space = QnSpace::new(6, 1, 1);
    let h_ab = ha_abelian(ab_space, 0, 1);
    let mut rng_ab = seeded(0x5b42 + 1000);
    let x0_ab: TtVector<f64> = randomized_state(ab_space, 2, 4, &mut rng_ab);
    let opts_ab = qntt::solvers::tdvp::TdvpOptions {
        dtau: 5.0,
        eps: 1e-6,
        max_sweeps: 50,
        ..Default::default()
    };
    let (lambda_ab, _) = qntt::solvers::tdvp::tdvp(&h_ab, &x0_ab, &opts_ab);
    assert!(
        (lambda_ab - (-2.0)).abs() <= 1e-4,
        "abelian tdvp energy {lambda_ab}"
    );
    assert!(
        (lambda_su2 - lambda_ab).abs() <= 2e-4,
        "cross-symmetry mismatch: su2={lambda_su2} vs abelian={lambda_ab}"
    );
}

#[test]
fn s5b_x4_cross_symmetry_polarized_multiplet_selection() {
    let d = 6usize;
    let t_generic = |d: usize| -> CoeffTensor<f64> {
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for i in 0..d {
            for j in (i + 1)..d {
                let val = -1.0 / (1.0 + (i as f64 - j as f64).abs());
                t.set(&[i, j], val);
                t.set(&[j, i], val);
            }
            t.set(&[i, i], 0.1 * i as f64);
        }
        t
    };
    let t = t_generic(d);
    let v = zero_v::<f64>(d);

    let space2 = Su2Space::new(d, 4, 2);
    let h2 = Su2SparseHamiltonian::new(space2, &t, &v, None, 1e-12);
    let space4 = Su2Space::new(d, 4, 4);
    let h4 = Su2SparseHamiltonian::new(space4, &t, &v, None, 1e-12);

    let mut lambda2 = f64::INFINITY;
    let mut x2 = qntt::su2::vector::tt_zeros::<f64>(space2);
    let mut lambda4 = f64::INFINITY;
    for trial in 0..4u64 {
        let mut rng = seeded(0x5b43 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space2, 1, 2, &mut rng);
        let (l, xt) = mals(
            &h2,
            &x0,
            &MalsOptions {
                eps: 1e-6,
                bond_dims: vec![0],
                ..Default::default()
            },
        );
        if l < lambda2 {
            lambda2 = l;
            x2 = xt;
        }
    }
    for trial in 0..4u64 {
        let mut rng = seeded(0x5b44 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space4, 1, 2, &mut rng);
        let (l, _xt) = mals(
            &h4,
            &x0,
            &MalsOptions {
                eps: 1e-6,
                bond_dims: vec![0],
                ..Default::default()
            },
        );
        if l < lambda4 {
            lambda4 = l;
        }
    }

    let ab_space = QnSpace::new(d, 3, 1);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-12);
    let mut lambda_ab = f64::INFINITY;
    let mut x_ab = qntt::tt_zeros::<f64>(ab_space);
    for trial in 0..4u64 {
        let mut rng = seeded(0x5b45 + trial);
        let x0: TtVector<f64> = randomized_state(ab_space, 1, 2, &mut rng);
        let (l, xt) = qntt::solvers::mals::mals(
            &h_ab,
            &x0,
            &qntt::solvers::mals::MalsOptions {
                eps: 1e-6,
                bond_dims: vec![0],
                ..Default::default()
            },
        );
        if l < lambda_ab {
            lambda_ab = l;
            x_ab = xt;
        }
    }

    let lambda_min = lambda2.min(lambda4);
    assert!(
        (lambda_ab - lambda_min).abs() <= 2e-4,
        "X4 mismatch: abelian={lambda_ab} vs min(su2_2S=2={lambda2}, su2_2S=4={lambda4})={lambda_min}"
    );
    assert!(
        (lambda2 - lambda4).abs() > 1e-3,
        "non-vacuity: 2S=2 and 2S=4 energies must not tie ({lambda2} vs {lambda4}); re-pin t_generic if they do"
    );
    assert!(x2.check());
    let (n_theta, n_cw) = count_theta_cw(&x2);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight trivial at 2S=2 (n_theta={n_theta}, n_cw={n_cw})"
    );
    let resid_ab = global_eigen_residual_abelian(&h_ab, &x_ab, lambda_ab);
    assert!(resid_ab <= 1e-4, "abelian(3,1) residual {resid_ab}");
}

#[test]
fn s5b_i2_als_fixed_point_is_shared_davidson() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b50);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (lambda, x) = als(
        &h,
        &x0,
        &AlsOptions {
            eps: 1e-8,
            ..Default::default()
        },
    );
    assert!(x.check());

    let fr = right_to_left_sketching(&h, &x);
    let fl0 = boundary_idsketch(&x, 0, 1);
    let ws = su2_onesite_workspace(&h, x.core(0));
    let mut op = |a: &qntt::su2::core::Su2Core<f64>| h.sketched_hamiltonian(a, &fl0, &fr[0], &ws);
    let d_diag = h.effective_diagonal(&fl0, &fr[0], x.core(0), &ws);

    let v0 = x.core(0);
    let n0 = op(v0);
    assert!(
        InnerProductVector::norm(&n0) > 1e-6,
        "non-vacuity: local op(x0) must not be ~zero, got {}",
        InnerProductVector::norm(&n0)
    );

    let mut applyprec = |t: &mut qntt::su2::core::Su2Core<f64>, theta: f64| {
        qntt::solvers::davidson::precond(t, &d_diag, theta)
    };
    let dopts = qntt::solvers::davidson::DavidsonOpts {
        tol: 1e-8,
        ..Default::default()
    };
    let (theta, v, _info) = qntt::solvers::davidson::davidson(&mut op, v0, &mut applyprec, &dopts);

    assert!(
        (theta - lambda).abs() <= 1e-6 * lambda.abs().max(1.0),
        "theta={theta} vs driver lambda={lambda}"
    );
    let lv = op(&v);
    let mut resid = lv.clone();
    InnerProductVector::axpy(&mut resid, -theta, &v);
    let resid_norm = InnerProductVector::norm(&resid);
    assert!(
        resid_norm <= 1e-6 * theta.abs().max(1.0),
        "local eigenpair residual {resid_norm}"
    );
}
