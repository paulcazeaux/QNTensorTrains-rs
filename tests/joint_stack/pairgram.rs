// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::pairgram::{generalized_gamma2, pair_gram};
use qntt::pairgram_tangent::orbital_tangent_matrix;
use qntt::rdm::transition_1rdm;
use qntt::solvers::joint::{
    GramRoute, JointCtxBuild, JointGateError, JointOpts, joint_context, joint_context_with,
};
use qntt::solvers::riemannian::tangent_bases;
use qntt::{QnSpace, TtVector, e_pq, g_pq};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::optimize::capped_randn;
use crate::util::lih_cation_rank1_fixture;

fn assert_g1(x: &TtVector<f64>, label: &str) {
    let d = x.space.d;
    let g2 = generalized_gamma2(x);
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;

    let mut reference = vec![0.0f64; d * d * d * d];
    for k in 0..d {
        for l in 0..d {
            let y = e_pq(x, k, l, 1.0);
            let gam = transition_1rdm(x, &y);
            for i in 0..d {
                for j in 0..d {
                    reference[idx(i, j, k, l)] = gam.get(&[i, j]);
                }
            }
        }
    }
    let scale = reference
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-30);
    let mut worst = 0.0f64;
    let mut worst_at = (0, 0, 0, 0);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let dev = (g2[idx(i, j, k, l)] - reference[idx(i, j, k, l)]).abs();
                    if dev > worst {
                        worst = dev;
                        worst_at = (i, j, k, l);
                    }
                }
            }
        }
    }
    assert!(
        worst <= 1e-12 * scale,
        "{label}: worst |sweep − oracle| = {worst:e} at {worst_at:?} (scale {scale:e}); \
         sweep = {:e}, oracle = {:e}",
        g2[idx(worst_at.0, worst_at.1, worst_at.2, worst_at.3)],
        reference[idx(worst_at.0, worst_at.1, worst_at.2, worst_at.3)]
    );
}

#[test]
fn tpg_g1_composition_oracle() {
    let space = QnSpace::new(4, 2, 1);
    let x = capped_randn(space, 1, 2, 71);
    assert_g1(&x, "d=4 (2,1) truncated");

    let x = capped_randn(space, 8, 8, 72);
    assert_g1(&x, "d=4 (2,1) full-rank");

    let space = QnSpace::new(4, 2, 0);
    let x = capped_randn(space, 1, 2, 73);
    assert_g1(&x, "d=4 (2,0) single-channel");

    let space = QnSpace::new(5, 3, 2);
    let x = capped_randn(space, 1, 2, 74);
    assert_g1(&x, "d=5 (3,2) truncated");
}

#[test]
fn tpg_g2_trace_pin() {
    let space = QnSpace::new(5, 2, 2);
    let x = capped_randn(space, 1, 2, 81);
    let d = space.d;
    let n_elec = (space.nup + space.ndn) as f64;
    let g2 = generalized_gamma2(&x);
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    let gam = transition_1rdm(&x, &x);
    let scale = (0..d)
        .flat_map(|k| (0..d).map(move |l| (k, l)))
        .fold(0.0f64, |m, (k, l)| m.max(gam.get(&[k, l]).abs()))
        .max(1e-30);
    for k in 0..d {
        for l in 0..d {
            let tr: f64 = (0..d).map(|i| g2[idx(i, i, k, l)]).sum();
            let expect = n_elec * gam.get(&[k, l]);
            assert!(
                (tr - expect).abs() <= 1e-12 * n_elec * scale,
                "trace pin failed at (k,l) = ({k},{l}): Σ_i Γ₂[i,i,k,l] = {tr:e} vs N·γ = \
                 {expect:e}"
            );
        }
    }
}

#[test]
fn tpg_g3_symmetry() {
    let space = QnSpace::new(5, 3, 1);
    let x = capped_randn(space, 1, 2, 91);
    let d = space.d;
    let g2 = generalized_gamma2(&x);
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    let scale = g2.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let a = g2[idx(i, j, k, l)];
                    let b = g2[idx(l, k, j, i)];
                    assert!(
                        (a - b).abs() <= 1e-12 * scale,
                        "adjoint symmetry failed: Γ₂[{i},{j},{k},{l}] = {a:e} vs \
                         Γ₂[{l},{k},{j},{i}] = {b:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn tpg_s_equiv_dots() {
    for (space, seed, label) in [
        (QnSpace::new(4, 2, 1), 101u64, "d=4 truncated"),
        (QnSpace::new(5, 2, 2), 102u64, "d=5 truncated"),
        (QnSpace::new(4, 2, 1), 103u64, "d=4 full-rank"),
    ] {
        let (lo, hi) = if label.contains("full") {
            (8, 8)
        } else {
            (1, 2)
        };
        let x = capped_randn(space, lo, hi, seed);
        let d = space.d;
        let pairs: Vec<(usize, usize)> = (0..d).flat_map(|p| (0..p).map(move |q| (p, q))).collect();
        let s_sweep = pair_gram(&x, &pairs);

        let u: Vec<TtVector<f64>> = pairs
            .iter()
            .map(|&(p, q)| &g_pq(&x, p, q) * (-1.0f64))
            .collect();
        let n = pairs.len();
        let mut s_dots = vec![0.0f64; n * n];
        for a in 0..n {
            for b in a..n {
                let v = qntt::dot(&u[a], &u[b]);
                s_dots[a * n + b] = v;
                s_dots[b * n + a] = v;
            }
        }
        let scale = s_dots.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
        let mut worst = 0.0f64;
        for i in 0..n * n {
            worst = worst.max((s_sweep[i] - s_dots[i]).abs());
        }
        assert!(
            worst <= 1e-12 * scale,
            "{label}: worst |pair_gram − dots| = {worst:e} (scale {scale:e})"
        );
    }
}

#[test]
fn tpg_theta_signed_pin() {
    for (space, seed) in [
        (QnSpace::new(4, 2, 1), 111u64),
        (QnSpace::new(5, 3, 2), 112u64),
    ] {
        let x = capped_randn(space, 1, 2, seed);
        let (bl, br) = tangent_bases(&x);
        let d = space.d;
        let pairs: Vec<(usize, usize)> = (0..d).flat_map(|p| (0..p).map(move |q| (p, q))).collect();
        let theta = orbital_tangent_matrix(&bl, &br, &x, &pairs);
        let ncol = pairs.len();
        assert_eq!(theta.len() % ncol, 0);
        let ncoef = theta.len() / ncol;

        for (a, &(p, q)) in pairs.iter().enumerate() {
            let u = &g_pq(&x, p, q) * (-1.0f64);
            let tau = TtTangent::project(&bl, &br, &u);
            assert_eq!(
                tau.num_coeffs(),
                ncoef,
                "column length must match the tangent coefficient count"
            );
            let col = &theta[a * ncoef..(a + 1) * ncoef];
            let scale = (0..ncoef)
                .fold(0.0f64, |m, i| m.max(tau.coeff(i).abs()))
                .max(1e-30);
            let mut worst = 0.0f64;
            for (i, &c) in col.iter().enumerate() {
                worst = worst.max((c - tau.coeff(i)).abs());
            }
            assert!(
                worst <= 1e-12 * scale,
                "pair ({p},{q}): worst |Θ − P(−g_pq)| = {worst:e} (scale {scale:e})"
            );

            let tau_plus = TtTangent::project(&bl, &br, &g_pq(&x, p, q));
            let col_norm = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            let dist_plus = (0..ncoef)
                .map(|i| (col[i] - tau_plus.coeff(i)).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                col_norm > 1e-8,
                "anti-vacuity: pair ({p},{q}) column must be nonzero, got {col_norm:e}"
            );
            assert!(
                dist_plus >= col_norm,
                "anti-oracle: pair ({p},{q}) must sit O(1) away from the +g_pq projection                  (dist {dist_plus:e} vs norm {col_norm:e})"
            );
        }
    }
}

#[test]
fn tpg_ctx_route_equiv() {
    let (_space, ham, x, _t, _v) = lih_cation_rank1_fixture();
    let opts = JointOpts::default();
    let dots = joint_context(&ham, &x, &opts).expect("dots context must gate");
    let sweep = joint_context_with(
        &ham,
        &x,
        &opts,
        &JointCtxBuild {
            gram: GramRoute::Sweep,
            lean: false,
        },
    )
    .expect("sweep context must gate");

    assert_eq!(sweep.n_eff(), dots.n_eff(), "n_eff parity");
    let anchor = dots.orb_spectrum().last().copied().unwrap_or(0.0).max(1.0);
    for (a, b) in sweep.orb_spectrum().iter().zip(dots.orb_spectrum()) {
        assert!(
            (a - b).abs() <= 1e-11 * anchor,
            "N spectrum mismatch: {a:e} vs {b:e} (anchor {anchor:e})"
        );
    }

    let n_orb = dots.pairs().len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2024);
    for _ in 0..4 {
        let g: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
        let p1 = dots.v_times(&dots.vt_times(&g));
        let p2 = sweep.v_times(&sweep.vt_times(&g));
        let scale = p1.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        let dev = p1
            .iter()
            .zip(&p2)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(
            dev <= 1e-8 * scale,
            "keep-projector mismatch: {dev:e} vs scale {scale:e}"
        );
    }

    let y = capped_randn(_space, 1, 2, 3131);
    let fc_full: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    let (bl_d, br_d) = dots.bases();
    let fxi_d = TtTangent::project(bl_d, br_d, &y);
    let (bl_s, br_s) = sweep.bases();
    let fxi_s = TtTangent::project(bl_s, br_s, &y);
    let r1 = dots.joint_raise(fxi_d, dots.vt_times(&fc_full));
    let r2 = sweep.joint_raise(fxi_s, sweep.vt_times(&fc_full));
    let xi_scale = r1.xi.norm().max(1e-30);
    let mut dxi = r1.xi.clone();
    dxi.axpy(-1.0, &r2.xi);
    assert!(
        dxi.norm() <= 1e-8 * xi_scale,
        "raise tangent-half mismatch: {:e} vs scale {xi_scale:e}",
        dxi.norm()
    );
    let c1 = dots.v_times(&r1.c);
    let c2 = sweep.v_times(&r2.c);
    let c_scale = c1.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let c_dev = c1
        .iter()
        .zip(&c2)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        c_dev <= 1e-8 * c_scale,
        "raise kept-half mismatch: {c_dev:e}"
    );

    let w1 = dots.hess_apply(&r1);
    let w2 = sweep.hess_apply(&r2);
    let wxi_scale = w1.xi.norm().max(1e-30);
    let mut dwxi = w1.xi.clone();
    dwxi.axpy(-1.0, &w2.xi);
    assert!(
        dwxi.norm() <= 1e-7 * wxi_scale,
        "hess_apply tangent-half mismatch: {:e} vs scale {wxi_scale:e}",
        dwxi.norm()
    );
    let wc1 = dots.v_times(&w1.c);
    let wc2 = sweep.v_times(&w2.c);
    let wc_scale = wc1.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let wc_dev = wc1
        .iter()
        .zip(&wc2)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        wc_dev <= 1e-7 * wc_scale,
        "hess_apply kept-half mismatch: {wc_dev:e}"
    );
}

#[test]
fn tpg_lean_matches_full() {
    let (_space, ham, x, _t, _v) = lih_cation_rank1_fixture();
    let opts = JointOpts::default();
    let full = joint_context_with(
        &ham,
        &x,
        &opts,
        &JointCtxBuild {
            gram: GramRoute::Sweep,
            lean: false,
        },
    )
    .expect("full sweep context must gate");
    let lean = joint_context_with(
        &ham,
        &x,
        &opts,
        &JointCtxBuild {
            gram: GramRoute::Sweep,
            lean: true,
        },
    )
    .expect("lean sweep context must gate");
    assert!(lean.is_lean() && !full.is_lean());
    assert!(lean.try_orbital_directions().is_none());
    assert_eq!(lean.n_eff(), full.n_eff());

    let n_orb = full.pairs().len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2025);
    let y = capped_randn(_space, 1, 2, 3232);
    let fc_full: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    let (bl, br) = full.bases();
    let r1 = full.joint_raise(TtTangent::project(bl, br, &y), full.vt_times(&fc_full));
    let (bl2, br2) = lean.bases();
    let r2 = lean.joint_raise(TtTangent::project(bl2, br2, &y), lean.vt_times(&fc_full));
    let w1 = full.hess_apply(&r1);
    let w2 = lean.hess_apply(&r2);
    let scale = w1.xi.norm().max(1e-30);
    let mut dxi = w1.xi.clone();
    dxi.axpy(-1.0, &w2.xi);
    assert!(
        dxi.norm() <= 1e-13 * scale,
        "lean and full sweep contexts must run the identical apply path: {:e}",
        dxi.norm()
    );
    let wc1 = full.v_times(&w1.c);
    let wc2 = lean.v_times(&w2.c);
    let wc_scale = wc1.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
    let wc_dev = wc1
        .iter()
        .zip(&wc2)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        wc_dev <= 1e-13 * wc_scale,
        "lean kept-half mismatch: {wc_dev:e}"
    );
}

#[test]
fn tpg_lean_refusals_err() {
    let (_space, ham, x, _t, _v) = lih_cation_rank1_fixture();
    let out = joint_context_with(
        &ham,
        &x,
        &JointOpts::default(),
        &JointCtxBuild {
            gram: GramRoute::Dots,
            lean: true,
        },
    );
    assert!(
        matches!(out, Err(JointGateError::LeanNeedsSweep)),
        "lean + Dots must refuse at the door"
    );
}

#[test]
#[should_panic(expected = "LEAN")]
fn tpg_lean_refusals_directions_panic() {
    let (_space, ham, x, _t, _v) = lih_cation_rank1_fixture();
    let lean = joint_context_with(
        &ham,
        &x,
        &JointOpts::default(),
        &JointCtxBuild {
            gram: GramRoute::Sweep,
            lean: true,
        },
    )
    .expect("lean sweep context must gate");
    let _ = lean.orbital_directions();
}

#[test]
fn tpg_sweep_d10_equiv() {
    let space = QnSpace::new(10, 3, 2);
    let x = capped_randn(space, 1, 2, 4242);
    let opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..JointOpts::default()
    };
    let ham = qntt::SparseHamiltonian::new(
        space,
        &crate::util::generic_truncated_tv(10).0,
        &crate::util::generic_truncated_tv(10).1,
        None,
        f64::EPSILON,
    );
    let dots = joint_context(&ham, &x, &opts).expect("diagnosis context");
    let sweep = joint_context_with(
        &ham,
        &x,
        &opts,
        &JointCtxBuild {
            gram: GramRoute::Sweep,
            lean: false,
        },
    )
    .expect("diagnosis context");
    assert_eq!(sweep.n_eff(), dots.n_eff());
    let anchor = dots.orb_spectrum().last().copied().unwrap_or(0.0).max(1.0);
    for (a, b) in sweep.orb_spectrum().iter().zip(dots.orb_spectrum()) {
        assert!(
            (a - b).abs() <= 1e-10 * anchor,
            "d=10 N spectrum mismatch: {a:e} vs {b:e}"
        );
    }
    let n_orb = dots.pairs().len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(2026);
    for _ in 0..3 {
        let g: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
        let p1 = dots.v_times(&dots.vt_times(&g));
        let p2 = sweep.v_times(&sweep.vt_times(&g));
        let scale = p1.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-30);
        let dev = p1
            .iter()
            .zip(&p2)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(dev <= 1e-7 * scale, "d=10 keep-projector mismatch: {dev:e}");
    }
}

#[test]
fn tpg_auto_crossover_branch() {
    let opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..JointOpts::default()
    };
    let build = |g: GramRoute| JointCtxBuild {
        gram: g,
        lean: false,
    };

    {
        let d = 13usize;
        assert!(d >= qntt::solvers::joint::PAIRGRAM_CROSSOVER_D);
        let space = QnSpace::new(d, 2, 2);
        let (t, v) = crate::util::generic_truncated_tv(d);
        let ham = qntt::SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let x = capped_randn(space, 1, 2, 0xA070);
        let auto = joint_context_with(&ham, &x, &opts, &build(GramRoute::Auto))
            .expect("auto context at d=13");
        let sweep = joint_context_with(&ham, &x, &opts, &build(GramRoute::Sweep))
            .expect("sweep context at d=13");
        let dots = joint_context(&ham, &x, &opts).expect("dots context at d=13");
        assert_eq!(auto.n_eff(), sweep.n_eff());
        assert_eq!(auto.n_eff(), dots.n_eff());
        let av: Vec<u64> = auto.orb_spectrum().iter().map(|x| x.to_bits()).collect();
        let sv: Vec<u64> = sweep.orb_spectrum().iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            av, sv,
            "at d=13 Auto must take the Sweep path (bitwise orb_spectrum)"
        );
        let anchor = dots.orb_spectrum().last().copied().unwrap_or(0.0).max(1.0);
        let mut max_dev = 0.0f64;
        for (a, b) in auto.orb_spectrum().iter().zip(dots.orb_spectrum()) {
            max_dev = max_dev.max((a - b).abs());
            assert!(
                (a - b).abs() <= 1e-10 * anchor,
                "d=13 Auto-vs-Dots spectrum mismatch: {a:e} vs {b:e}"
            );
        }
        assert!(
            max_dev > 0.0,
            "Dots and Sweep agree BITWISE at d=13 — the bitwise Auto discriminator is vacuous; \
             strengthen the fixture"
        );
    }

    {
        let d = 12usize;
        assert!(d < qntt::solvers::joint::PAIRGRAM_CROSSOVER_D);
        let space = QnSpace::new(d, 2, 2);
        let (t, v) = crate::util::generic_truncated_tv(d);
        let ham = qntt::SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let x = capped_randn(space, 1, 2, 0xA071);
        let auto = joint_context_with(&ham, &x, &opts, &build(GramRoute::Auto))
            .expect("auto context at d=12");
        let dots = joint_context(&ham, &x, &opts).expect("dots context at d=12");
        let av: Vec<u64> = auto.orb_spectrum().iter().map(|x| x.to_bits()).collect();
        let dv: Vec<u64> = dots.orb_spectrum().iter().map(|x| x.to_bits()).collect();
        assert_eq!(
            av, dv,
            "below the crossover Auto must take the Dots path (bitwise orb_spectrum)"
        );
    }
}
