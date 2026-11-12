// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::Scalar;
use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::qn::QnSpace;
use qntt::round::RoundMode;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::ops::dot as su2_dot;
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{
    CgVariant, RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, riemannian_gradient,
    tangent_bases,
};
use qntt::su2::solvers::als::{AlsOptions, als};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_zeros as su2_tt_zeros};
use qntt::vector::{TtVector, tt_zeros};

use crate::s6::util::{
    count_theta_cw, global_eigen_residual, global_eigen_residual_abelian, ha_abelian, ha_su2,
    randomized_state, randomized_su2_state, seeded, two_body_su2, zero_v,
};

#[test]
fn s6_c1_cg_onebody_singlet() {
    let d = 6usize;
    let pairs: [(usize, usize); 3] = [(0, 1), (1, 4), (2, 5)];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let space = Su2Space::new(d, 2, 0);
        let h = ha_su2(space, i, j);
        let mut lambda = f64::INFINITY;
        let mut x = su2_tt_zeros::<f64>(space);
        for trial in 0..4u64 {
            let mut rng = seeded(0x6c10 + idx as u64 * 10 + trial);
            let x0: Su2TtVector<f64> = randomized_su2_state(space, 4, 6, &mut rng);
            assert!(x0.check());
            let opts = RiemannianOpts {
                eps: 1e-5,
                max_iter: 100,
                variant: CgVariant::Pr,
                ..RiemannianOpts::new(&space)
            };
            let (lc, xc) = riemannian_cg(&h, &x0, &opts);
            lambda = lc;
            x = xc;
            if (lambda + 2.0).abs() < 1e-4 {
                break;
            }
        }
        assert!(x.check());
        assert!(
            (lambda + 2.0).abs() < 1e-4,
            "pair ({i},{j}): CG energy {lambda}"
        );
        let resid = global_eigen_residual(&h, &x, lambda);
        assert!(resid <= 1e-4, "pair ({i},{j}): CG residual {resid}");
    }
}

#[test]
fn s6_c1b_cg_onebody_fr() {
    let d = 6usize;
    let (i, j) = (0usize, 1usize);
    let space = Su2Space::new(d, 2, 0);
    let h = ha_su2(space, i, j);
    let mut lambda = f64::INFINITY;
    let mut x = su2_tt_zeros::<f64>(space);
    for trial in 0..4u64 {
        let mut rng = seeded(0x6cf0 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 4, 6, &mut rng);
        assert!(x0.check());
        let opts = RiemannianOpts {
            eps: 1e-5,
            max_iter: 100,
            variant: CgVariant::Fr,
            ..RiemannianOpts::new(&space)
        };
        let (lc, xc) = riemannian_cg(&h, &x0, &opts);
        lambda = lc;
        x = xc;
        if (lambda + 2.0).abs() < 1e-4 {
            break;
        }
    }
    assert!(x.check());
    assert!(
        (lambda + 2.0).abs() < 1e-4,
        "Fr pair ({i},{j}): CG energy {lambda}"
    );
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "Fr pair ({i},{j}): CG residual {resid}");
}

#[test]
fn s6_c2_gd_onebody_singlet() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 2);
    let mut lambda = f64::INFINITY;
    let mut x = su2_tt_zeros::<f64>(space);
    for trial in 0..3u64 {
        let mut rng = seeded(0x6c20 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 4, 6, &mut rng);
        assert!(x0.check());
        let opts = RiemannianOpts {
            eps: 1e-5,
            max_iter: 200,
            ..RiemannianOpts::new(&space)
        };
        let (lg, xg) = riemannian_gd(&h, &x0, &opts);
        lambda = lg;
        x = xg;
        if (lambda + 2.0).abs() < 1e-4 {
            break;
        }
    }
    assert!(x.check());
    assert!((lambda + 2.0).abs() < 1e-4, "GD energy {lambda}");
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "GD residual {resid}");
}

#[test]
fn s6_c3_cg_twobody() {
    let d = 6usize;
    let tuples: [(usize, usize, usize, usize); 3] = [(0, 1, 2, 3), (0, 3, 1, 4), (1, 2, 0, 5)];
    for (idx, &(a, b, c, e)) in tuples.iter().enumerate() {
        let space = Su2Space::new(d, 2, 0);
        let h = two_body_su2(space, a, b, c, e);
        let mut lambda = f64::INFINITY;
        let mut x = su2_tt_zeros::<f64>(space);
        for trial in 0..4u64 {
            let mut rng = seeded(0x6c30 + idx as u64 * 10 + trial);
            let x0: Su2TtVector<f64> = randomized_su2_state(space, 5, 10, &mut rng);
            assert!(x0.check());
            let opts = RiemannianOpts {
                eps: 1e-5,
                max_iter: 100,
                ..RiemannianOpts::new(&space)
            };
            let (lc, xc) = riemannian_cg(&h, &x0, &opts);
            lambda = lc;
            x = xc;
            if lambda <= -0.5 + 1e-4 {
                break;
            }
        }
        assert!(x.check());
        assert!(
            lambda <= -0.5 + 1e-4,
            "case {idx}: CG energy bound {lambda}"
        );
        let resid = global_eigen_residual(&h, &x, lambda);
        assert!(resid <= 1e-4, "case {idx}: CG residual {resid}");
    }
}

#[test]
fn s6_c4_cg_weighted_path_polarized() {
    let space = Su2Space::new(6, 2, 2);
    let h = ha_su2(space, 0, 1);
    let mut lambda = f64::INFINITY;
    let mut x = su2_tt_zeros::<f64>(space);
    for trial in 0..4u64 {
        let mut rng = seeded(0x6c40 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 4, 6, &mut rng);
        assert!(x0.check());
        let opts = RiemannianOpts {
            eps: 1e-5,
            max_iter: 100,
            ..RiemannianOpts::new(&space)
        };
        let (lc, xc) = riemannian_cg(&h, &x0, &opts);
        lambda = lc;
        x = xc;
        if (lambda + 1.0).abs() < 1e-4 {
            break;
        }
    }
    assert!(x.check());
    assert!((lambda + 1.0).abs() <= 1e-4, "polarized CG energy {lambda}");
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "polarized CG residual {resid}");
    let (n_theta, n_cw) = count_theta_cw(&x);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight trivial (n_theta={n_theta}, n_cw={n_cw})"
    );
}

#[test]
fn s6_c5_gradient_matches_residual() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x6c50);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    let (base_l, base_r) = tangent_bases(&x);
    let (g, lambda) = riemannian_gradient(&h, &x, &base_l, &base_r, 1e-12, 1234);

    let hx = h.apply(&x);
    let exact_resid = &hx - &(&x * lambda);
    let gref = Su2TtTangent::project(&base_l, &base_r, &exact_resid);
    let diff = (&g.to_vector() - &gref.to_vector()).norm_lr(false);
    assert!(
        diff < 1e-6,
        "gradient == projected exact residual, diff={diff}"
    );

    let radial = su2_dot(&x, &g.to_vector()).abs_s();
    assert!(radial < 1e-8, "radial orthogonality: {radial}");
}

#[test]
fn s6_c6_gradient_vanishes_at_als_critical() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 1, 3);
    let mut rng = seeded(0x6c60);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 4, 6, &mut rng);
    let (lambda_a, xa) = als(
        &h,
        &x0,
        &AlsOptions {
            eps: 1e-8,
            ..Default::default()
        },
    );
    let mut x = xa;
    normalize_mut(&mut x);
    let (base_l, base_r) = tangent_bases(&x);
    let (g, _lambda) = riemannian_gradient(&h, &x, &base_l, &base_r, 1e-12, 1234);
    assert!(
        g.norm() / lambda_a.abs() < 1e-4,
        "gradient norm / |lambda| = {}",
        g.norm() / lambda_a.abs()
    );
}

#[test]
fn s6_d1_cross_symmetry_cg_onebody() {
    let su2_space = Su2Space::new(6, 2, 0);
    let h_su2 = ha_su2(su2_space, 0, 1);
    let mut lambda_su2 = f64::INFINITY;
    let mut x0_su2 = su2_tt_zeros::<f64>(su2_space);
    for trial in 0..4u64 {
        let mut rng = seeded(0x6d10 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(su2_space, 4, 6, &mut rng);
        let opts = RiemannianOpts {
            eps: 1e-5,
            max_iter: 100,
            variant: CgVariant::Pr,
            ..RiemannianOpts::new(&su2_space)
        };
        let (l, _x) = riemannian_cg(&h_su2, &x0, &opts);
        lambda_su2 = l;
        x0_su2 = x0;
        if (lambda_su2 + 2.0).abs() < 1e-4 {
            break;
        }
    }
    assert!(
        (lambda_su2 + 2.0).abs() <= 1e-4,
        "su2 CG energy {lambda_su2}"
    );

    let opts_rerun = RiemannianOpts {
        eps: 1e-5,
        max_iter: 100,
        variant: CgVariant::Pr,
        ..RiemannianOpts::new(&su2_space)
    };
    let (lambda_su2b, _) = riemannian_cg(&h_su2, &x0_su2, &opts_rerun);
    assert_eq!(
        lambda_su2, lambda_su2b,
        "su2 riemannian_cg is not deterministic on a rerun"
    );

    let ab_space = QnSpace::new(6, 1, 1);
    let h_ab = ha_abelian(ab_space, 0, 1);
    let mut lambda_ab = f64::INFINITY;
    let mut x_ab: TtVector<f64> = tt_zeros(ab_space);
    for trial in 0..4u64 {
        let mut rng = seeded(0x6d10 + 1000 + trial);
        let x0: TtVector<f64> = randomized_state(ab_space, 4, 6, &mut rng);
        let opts = qntt::solvers::riemannian::RiemannianOpts {
            eps: 1e-5,
            max_iter: 100,
            variant: qntt::solvers::riemannian::CgVariant::Pr,
            ..qntt::solvers::riemannian::RiemannianOpts::new(&ab_space)
        };
        let (l, x) = qntt::solvers::riemannian::riemannian_cg(&h_ab, &x0, &opts);
        lambda_ab = l;
        x_ab = x;
        if (lambda_ab + 2.0).abs() < 1e-4 {
            break;
        }
    }
    assert!(
        (lambda_ab + 2.0).abs() <= 1e-4,
        "abelian CG energy {lambda_ab}"
    );
    let resid_ab = global_eigen_residual_abelian(&h_ab, &x_ab, lambda_ab);
    assert!(resid_ab <= 1e-4, "abelian CG residual {resid_ab}");

    assert!(
        (lambda_su2 - lambda_ab).abs() <= 2e-4,
        "cross-symmetry mismatch: su2={lambda_su2} vs abelian={lambda_ab}"
    );
}

#[test]
fn s6_d2_cross_symmetry_cg_weighted() {
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
    let mut x2 = su2_tt_zeros::<f64>(space2);
    for trial in 0..8u64 {
        let mut rng = seeded(0x6d20 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space2, 1, 8, &mut rng);
        let opts = RiemannianOpts {
            eps: 1e-7,
            max_iter: 300,
            ..RiemannianOpts::new(&space2)
        };
        let (l, x) = riemannian_cg(&h2, &x0, &opts);
        if l < lambda2 {
            lambda2 = l;
            x2 = x;
        }
    }

    let mut lambda4 = f64::INFINITY;
    for trial in 0..8u64 {
        let mut rng = seeded(0x6d30 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space4, 1, 8, &mut rng);
        let opts = RiemannianOpts {
            eps: 1e-7,
            max_iter: 300,
            ..RiemannianOpts::new(&space4)
        };
        let (l, _x) = riemannian_cg(&h4, &x0, &opts);
        if l < lambda4 {
            lambda4 = l;
        }
    }

    let ab_space = QnSpace::new(d, 3, 1);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-12);
    let mut lambda_ab = f64::INFINITY;
    let mut x_ab: TtVector<f64> = tt_zeros(ab_space);
    for trial in 0..8u64 {
        let mut rng = seeded(0x6d40 + trial);
        let x0: TtVector<f64> = randomized_state(ab_space, 1, 8, &mut rng);
        let opts = qntt::solvers::riemannian::RiemannianOpts {
            eps: 1e-7,
            max_iter: 300,
            ..qntt::solvers::riemannian::RiemannianOpts::new(&ab_space)
        };
        let (l, x) = qntt::solvers::riemannian::riemannian_cg(&h_ab, &x0, &opts);
        if l < lambda_ab {
            lambda_ab = l;
            x_ab = x;
        }
    }

    let lambda_min = lambda2.min(lambda4);
    assert!(
        (lambda_ab - lambda_min).abs() <= 2e-4,
        "D2 mismatch: abelian={lambda_ab} vs min(su2_2S=2={lambda2}, su2_2S=4={lambda4})={lambda_min}"
    );
    assert!(
        (lambda2 - lambda4).abs() > 1e-3,
        "non-vacuity: 2S=2 and 2S=4 energies must not tie ({lambda2} vs {lambda4})"
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
