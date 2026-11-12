// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::core::tangent::{TtTangent, projective_transport};
use qntt::round::RoundMode;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::penalty::Penalty;
use qntt::solvers::riemannian::{
    CertifyOpts, CertifyVerdict, CgExit, CgVariant, Retraction, RiemannianOpts, normalize_mut,
    riemannian_cg, riemannian_cg_certified, riemannian_cg_observed, riemannian_cg_penalized,
    riemannian_cg_report, riemannian_gd, riemannian_gradient, tangent_bases,
};
use qntt::{CoeffTensor, QnSpace, Scalar, SparseHamiltonian, TtVector, dot, tt_zeros};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t8d_tangent_product_state_fd() {
    let d = 8usize;
    let space = QnSpace::from_n_two_sz(d, 6, 2);
    let x: TtVector<f64> = state_from_sites(d, &[1, 3, 5, 7], &[2, 5]);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8D00);
    let dir: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);

    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    let dx = TtTangent::project(&base_l, &base_r, &dir);

    let eps = 1e-6;
    let mut perturbed = &x + &(&dir * eps);
    perturbed.round_ranks_mut(&x.ranks);
    let fd = &(&perturbed - &x) * (1.0 / eps);

    let diff = &dx.to_vector() - &fd;
    assert!(diff.norm_lr(false) < 1e-5, "tangent-to-product-state FD");
}

#[test]
fn t8d_tangent_projection_linearity() {
    let (_space, _x, base_l, base_r, dir1, dir2) = tangent_test_setup(0x8D30);
    let combo = &(&dir1 * 0.7) - &(&dir2 * 1.3);
    let dxy = TtTangent::project(&base_l, &base_r, &combo);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let dy = TtTangent::project(&base_l, &base_r, &dir2);
    let want = dx.scale(0.7).sub(&dy.scale(1.3));
    let diff = &dxy.to_vector() - &want.to_vector();
    assert!(diff.norm_lr(false) < 1e-12, "projection linearity");
}

#[test]
fn t8d_tangent_idempotence() {
    let (_space, _x, base_l, base_r, dir1, _dir2) = tangent_test_setup(0x8D40);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let dx2 = TtTangent::project(&base_l, &base_r, &dx.to_vector());
    let diff = &dx.to_vector() - &dx2.to_vector();
    assert!(diff.norm_lr(false) < 1e-12, "projection idempotence");
}

#[test]
fn t8d_tangent_metric_consistency() {
    let (_space, _x, base_l, base_r, dir1, dir2) = tangent_test_setup(0x8D50);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let dy = TtTangent::project(&base_l, &base_r, &dir2);
    assert!(
        (dx.norm() - dx.to_vector().norm_lr(false)).abs() < 1e-12 * dx.norm().max(1.0),
        "norm consistency"
    );
    assert!(
        (dx.dot(&dy) - dot(&dx.to_vector(), &dy.to_vector())).abs() < 1e-12,
        "dot consistency"
    );
    assert!(
        (dx.dot(&dx) - dx.norm() * dx.norm()).abs() < 1e-12,
        "dot(dx,dx) == norm(dx)^2"
    );
}

#[test]
fn t8d_tangent_vector_space_embedding() {
    let (_space, _x, base_l, base_r, dir1, dir2) = tangent_test_setup(0x8D60);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let dy = TtTangent::project(&base_l, &base_r, &dir2);

    let sum_v = &dx.to_vector() + &dy.to_vector();
    let v_sum = dx.add(&dy).to_vector();
    assert!((&sum_v - &v_sum).norm_lr(false) < 1e-12, "to_vector(dx+dy)");

    let diff_v = &dx.to_vector() - &dy.to_vector();
    let v_diff = dx.sub(&dy).to_vector();
    assert!(
        (&diff_v - &v_diff).norm_lr(false) < 1e-12,
        "to_vector(dx-dy)"
    );

    let scaled = &dx.to_vector() * 2.5;
    let v_scaled = dx.scale(2.5).to_vector();
    assert!(
        (&scaled - &v_scaled).norm_lr(false) < 1e-12,
        "to_vector(2.5*dx)"
    );

    let mut dz = dx.clone();
    dz.axpy(0.3, &dy);
    let want_axpy = &dx.to_vector() + &(&dy.to_vector() * 0.3);
    assert!(
        (&dz.to_vector() - &want_axpy).norm_lr(false) < 1e-12,
        "axpy!(0.3, dy, dz)"
    );

    let mut dw = dx.clone();
    dw.axpby(0.3, &dy, -1.7);
    let want_axpby = &(&dy.to_vector() * 0.3) + &(&dx.to_vector() * (-1.7));
    assert!(
        (&dw.to_vector() - &want_axpby).norm_lr(false) < 1e-12,
        "axpby!(0.3, dy, -1.7, dw)"
    );

    let mut dv = dx.clone();
    dv.scale_mut(-0.6);
    let want_scale_mut = &dx.to_vector() * (-0.6);
    assert!(
        (&dv.to_vector() - &want_scale_mut).norm_lr(false) < 1e-12,
        "lmul!/rmul!(-0.6, dx)"
    );
}

#[test]
fn t8d_tangent_retraction() {
    let (_space, x, base_l, base_r, dir1, _dir2) = tangent_test_setup(0x8D70);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);

    let r0 = dx.retract_hosvd(0.0);
    assert!((&r0 - &x).norm_lr(false) < 1e-12, "retract at α=0 == x");

    let alpha = 1e-3;
    let lin_at = |a: f64| -> f64 {
        let ra = dx.retract_hosvd(a);
        let linear = &x + &(&dx.to_vector() * a);
        (&ra - &linear).norm_lr(false)
    };
    let e1 = lin_at(alpha);
    let e2 = lin_at(2.0 * alpha);
    assert!(
        e1 < 10.0 * alpha * alpha,
        "first-order retraction error e1={e1}"
    );
    assert!(
        (e2 / e1 - 4.0).abs() < 1.0,
        "retraction ratio e2/e1 = {}",
        e2 / e1
    );
}

#[test]
fn t8d_tangent_transport() {
    let (space, x, base_l, base_r, dir1, _dir2) = tangent_test_setup(0x8D80);
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let self_transport = projective_transport(&x, &dx);
    let diff = &self_transport.to_vector() - &dx.to_vector();
    assert!(diff.norm_lr(false) < 1e-12, "self-transport is identity");

    let mut rng_y = Xoshiro256PlusPlus::seed_from_u64(0x8D81);
    let mut y: TtVector<f64> = randomized_state(space, 2, 4, &mut rng_y);
    y.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    y.scale_mut(1.0 / y.norm());
    let mut y_base_l = y.clone();
    y_base_l.left_orthogonalize(true);
    let transported = projective_transport(&y, &dx);
    for k in 0..space.d - 1 {
        for s in space.bond_sectors(k + 1) {
            let l = y_base_l.core(k).vertical_unfolding(s);
            let c = transported.component(k).vertical_unfolding(s);
            let mut n2 = 0.0f64;
            for j in 0..c.ncols() {
                for i in 0..l.ncols() {
                    let mut acc = 0.0f64;
                    for kk in 0..l.nrows() {
                        acc += *l.get(kk, i) * *c.get(kk, j);
                    }
                    n2 += acc * acc;
                }
            }
            assert!(
                n2.sqrt() < 1e-12,
                "transported gauge condition at bond {}",
                k + 1
            );
        }
    }
}

#[test]
fn t8d_cg_onebody() {
    let d = 6usize;
    let pairs: [(usize, usize); 3] = [(0, 1), (1, 4), (2, 5)];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[i, j], -1.0);
        t.set(&[j, i], -1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut lambda = f64::INFINITY;
        let mut x = tt_zeros::<f64>(space);
        for trial in 0..4u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8D90 + idx as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 4, 6, &mut rng);
            x0.check();
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
        x.check();
        assert!(
            (lambda + 2.0).abs() < 1e-4,
            "pair ({i},{j}): CG energy {lambda}"
        );
        let resid = &(&h * &x) - &(&x * lambda);
        assert!(resid.norm_lr(false) < 1e-4, "pair ({i},{j}): CG residual");
    }
}

#[test]
fn t8d_cg_onebody_fr() {
    let d = 6usize;
    let (i, j) = (0usize, 1usize);
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[i, j], -1.0);
    t.set(&[j, i], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut lambda = f64::INFINITY;
    let mut x = tt_zeros::<f64>(space);
    for trial in 0..4u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DF0 + trial);
        let x0: TtVector<f64> = randomized_state(space, 4, 6, &mut rng);
        x0.check();
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
    x.check();
    assert!(
        (lambda + 2.0).abs() < 1e-4,
        "Fr pair ({i},{j}): CG energy {lambda}"
    );
    let resid = &(&h * &x) - &(&x * lambda);
    assert!(
        resid.norm_lr(false) < 1e-4,
        "Fr pair ({i},{j}): CG residual"
    );
}

#[test]
fn t8d_gd_onebody() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[0, 2], -1.0);
    t.set(&[2, 0], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut lambda = f64::INFINITY;
    let mut x = tt_zeros::<f64>(space);
    for trial in 0..3u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DA0 + trial);
        let x0: TtVector<f64> = randomized_state(space, 4, 6, &mut rng);
        x0.check();
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
    x.check();
    assert!((lambda + 2.0).abs() < 1e-4, "GD energy {lambda}");
    let resid = &(&h * &x) - &(&x * lambda);
    assert!(resid.norm_lr(false) < 1e-4, "GD residual");
}

#[test]
fn t8d_cg_twobody() {
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
                    if count.is_multiple_of(140) {
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
        for trial in 0..4u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DB0 + ci as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
            x0.check();
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
        x.check();
        assert!(lambda <= -0.5 + 1e-4, "case {ci}: CG energy bound {lambda}");
        let proj: f64 = zs.iter().map(|z| dot(&x, z).abs_s().powi(2)).sum();
        assert!(
            (proj - 1.0).abs() < 1e-4,
            "case {ci}: CG span projection {proj}"
        );
        let resid = &(&h * &x) - &(&x * lambda);
        assert!(resid.norm_lr(false) < 1e-4, "case {ci}: CG residual");
    }
}

#[test]
fn t8d_gradient_vanishes_at_als_critical() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[1, 3], -1.0);
    t.set(&[3, 1], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DD0);
    let x0: TtVector<f64> = randomized_state(space, 4, 6, &mut rng);
    let (lambda_a, xa) = als(
        &h,
        &x0,
        &AlsOptions {
            eps: 1e-8,
            ..AlsOptions::default()
        },
    );
    let mut x = xa;
    normalize_mut(&mut x);
    let (base_l, base_r) = tangent_bases(&x);
    let (g, _lambda) = riemannian_gradient(&h, &x, &base_l, &base_r, 1e-12, 1234);
    assert!(
        g.norm() / lambda_a.abs() < 1e-4,
        "gradient norm / |λ| = {}",
        g.norm() / lambda_a.abs()
    );
}

#[test]
fn t8d_cg_als_default_is_hosvd() {
    let space = QnSpace::new(6, 1, 1);
    assert_eq!(RiemannianOpts::new(&space).retraction, Retraction::Hosvd);
}

#[test]
#[should_panic(expected = "undefined for the penalized functional")]
fn t8d_cg_als_penalized_refused() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let h = two_body_swap_h(space, 0, 1, 2, 3);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DE0);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let psi: TtVector<f64> = randomized_state(space, 3, 6, &mut rng);
    let penalty: Penalty<TtVector<f64>> = Penalty::new(3.0, std::slice::from_ref(&psi));
    let opts = RiemannianOpts {
        eps: 1e-5,
        max_iter: 5,
        retraction: Retraction::AlsRounded {
            pairs: 1,
            delta0: None,
        },
        ..RiemannianOpts::new(&space)
    };
    let _ = riemannian_cg_penalized(&h, &x0, &penalty, &opts);
}

#[test]
fn t8d_cg_als_twobody() {
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
                    if count.is_multiple_of(140) {
                        cases.push((a, b, c, e));
                    }
                }
            }
        }
    }
    for (ci, &(a, b, c, e)) in cases.iter().enumerate().step_by(3) {
        let h = two_body_swap_h(space, a, b, c, e);
        let zs = span_states(d, a, b, c, e);

        let mut lambda = f64::INFINITY;
        let mut x = tt_zeros::<f64>(space);
        for trial in 0..4u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DB0 + ci as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
            x0.check();
            let opts = RiemannianOpts {
                eps: 1e-5,
                max_iter: 100,
                retraction: Retraction::AlsRounded {
                    pairs: 1,
                    delta0: None,
                },
                ..RiemannianOpts::new(&space)
            };
            let (lc, xc) = riemannian_cg(&h, &x0, &opts);
            lambda = lc;
            x = xc;
            if lambda <= -0.5 + 1e-4 {
                break;
            }
        }
        x.check();
        assert!(
            x.ortho.is_some(),
            "case {ci}: accepted point is not canonical"
        );
        assert!(
            lambda <= -0.5 + 1e-4,
            "case {ci}: CG-ALS energy bound {lambda}"
        );
        let proj: f64 = zs.iter().map(|z| dot(&x, z).abs_s().powi(2)).sum();
        assert!(
            (proj - 1.0).abs() < 1e-4,
            "case {ci}: CG-ALS span projection {proj}"
        );
        let resid = &(&h * &x) - &(&x * lambda);
        assert!(resid.norm_lr(false) < 1e-4, "case {ci}: CG-ALS residual");
    }
}

#[test]
fn t8d_cg_als_monotone_grid_preserved() {
    let d = 6usize;
    let space = QnSpace::new(d, 2, 2);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        t.set(&[i, i], -0.3 * (i as f64 + 1.0));
        if i + 1 < d {
            t.set(&[i, i + 1], -1.0);
            t.set(&[i + 1, i], -1.0);
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -0.7);
    v.set(&[1, 3, 0, 2], -0.7);
    v.set(&[4, 1, 5, 2], -0.4);
    v.set(&[2, 5, 1, 4], -0.4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DE1);
    let mut x0: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    normalize_mut(&mut x0);
    let (bl, br) = tangent_bases(&x0);
    let (g0, lambda0) = riemannian_gradient(&h, &x0, &bl, &br, 1e-12, 1234);
    let g0 = g0.norm();
    let efloor = 32.0 * f64::EPSILON * (1.0 + lambda0.abs());

    for retraction in [
        Retraction::Hosvd,
        Retraction::AlsRounded {
            pairs: 1,
            delta0: None,
        },
    ] {
        let opts = RiemannianOpts {
            eps: 1e-9,
            max_iter: 90,
            retraction,
            ..RiemannianOpts::new(&space)
        };
        let mut energies: Vec<f64> = Vec::new();
        let mut ranks0 = None;
        let mut calls = 0usize;
        let mut obs = |_it: usize, s: &TtVector<f64>, _g: f64| -> bool {
            calls += 1;
            let e = qntt::rayleigh_quotient(&h, s) / s.norm().powi(2);
            assert!(e.is_finite(), "{retraction:?}: non-finite energy");
            energies.push(e);
            let r = ranks0.get_or_insert_with(|| s.ranks.clone());
            assert_eq!(&s.ranks, r, "{retraction:?}: rank grid changed");
            true
        };
        let (_, x) = riemannian_cg_observed(&h, &x0, &opts, &mut obs);
        for w in energies.windows(2) {
            assert!(
                w[1] <= w[0] + 4.0 * efloor,
                "{retraction:?}: energy rose {:.3e} → {:.3e}",
                w[0],
                w[1]
            );
        }
        let (bl, br) = tangent_bases(&x);
        let (g, _) = riemannian_gradient(&h, &x, &bl, &br, 1e-12, 1234);
        println!(
            "t8d_cg_als_monotone_grid_preserved {retraction:?}: {calls} accepted iterates, ‖g‖ {:.3e} → {:.3e}",
            g0,
            g.norm()
        );
        assert!(
            g.norm() <= 0.1 * g0,
            "{retraction:?}: gradient not contracted 10×: {:.3e} vs {:.3e}",
            g.norm(),
            g0
        );
    }
}

#[test]
fn t8d_cg_certified_reports_a_pd_minimizer() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let h = two_body_swap_h(space, 0, 1, 2, 3);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DE2);
    let mut x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    normalize_mut(&mut x0);
    let opts = RiemannianOpts {
        eps: 1e-8,
        max_iter: 200,
        retraction: Retraction::AlsRounded {
            pairs: 1,
            delta0: None,
        },
        ..RiemannianOpts::new(&space)
    };
    let plain = riemannian_cg_report(&h, &x0, &opts);
    assert_eq!(plain.exit, CgExit::Gradient, "{:?}", plain.exit);
    let one = riemannian_cg_report(
        &h,
        &x0,
        &RiemannianOpts {
            max_iter: 1,
            ..opts.clone()
        },
    );
    assert_eq!((one.exit, one.iterations), (CgExit::Budget, 1));
    let rep = riemannian_cg_certified(
        &h,
        &x0,
        &opts,
        &CertifyOpts {
            probe_budget: 300,
            ..CertifyOpts::default()
        },
        None,
    );
    println!(
        "t8d_cg_certified: {} it, exit {:?}, verdict {:?}, rr {:?}, ii {:?}",
        rep.iterations, rep.exit, rep.verdict, rep.rr, rep.ii
    );
    assert!(
        matches!(rep.verdict, CertifyVerdict::Minimizer { .. }),
        "{:?}",
        rep.verdict
    );
    assert!(rep.kicks.is_empty());
    assert!(rep.ii.is_some());
    assert_eq!(rep.iterations, plain.iterations);
    let ov = dot(&rep.x, &plain.x).abs_s();
    assert!(
        (1.0 - ov).abs() < 1e-12,
        "certified endpoint moved: 1−|⟨a,b⟩| = {:.2e}",
        1.0 - ov
    );
}
