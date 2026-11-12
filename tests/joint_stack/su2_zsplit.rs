// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::sync::OnceLock;

use qntt::round::RoundMode;
use qntt::solvers::joint::{FcRoute, JointOpts, ZRoute};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hess::riemannian_gradient_env;
use qntt::su2::joint::{Su2JointContext, Su2JointVector, su2_joint_context};
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::normalize_mut;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::theta;
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::tangent_basis;

use crate::su2_util::su2_minimizer;
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::randomized_su2_state;
use qntt_testkit::seeded;
use rand::RngExt;

fn count_theta(x: &Su2TtVector<f64>) -> usize {
    let space = x.space;
    let mut n = 0usize;
    for b in 0..space.d {
        for m in space.bond_sectors(b) {
            if x.rank(b, m) == 0 {
                continue;
            }
            for succ in space.bond_sectors(b + 1) {
                if x.rank(b + 1, succ) == 0 {
                    continue;
                }
                if (theta(m.two_s, succ.two_s) - 1.0).abs() > 1e-12 {
                    n += 1;
                }
            }
        }
    }
    n
}

struct Fixture {
    h: Su2SparseHamiltonian<f64>,
    x: Su2TtVector<f64>,
}

fn zsplit_fixture(seed: u64) -> &'static Fixture {
    static FX: OnceLock<Fixture> = OnceLock::new();
    FX.get_or_init(|| build_zsplit_fixture(seed))
}

fn build_zsplit_fixture(seed: u64) -> Fixture {
    let d = 5usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 3, 1);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&h, space, seed, 1, 2);
    assert!(
        count_theta(&x) > 0,
        "zsplit_fixture: θ ≡ 1 at this base point — every metric arm built on it is vacuous"
    );
    Fixture { h, x }
}

fn perturbed_state(x: &Su2TtVector<f64>, eps: f64, seed: u64) -> Su2TtVector<f64> {
    let mut rng = seeded(seed);
    let mut noise: Su2TtVector<f64> = randomized_su2_state(x.space, 1, 2, &mut rng);
    normalize_mut(&mut noise);
    let mut out = x + &(&noise * eps);
    out.round_mut(1e-13, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut out);
    assert!(out.check(), "perturbed_state: failed check()");
    out
}

fn zsplit_opts(z_route: ZRoute, b_general: bool) -> JointOpts {
    JointOpts {
        tt_grad_tol: 1e-4,
        orb_floor: 1e-8,
        pairs: None,
        z_route,
        b_general,
    }
}

fn route_pair<'h>(
    h: &'h Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    b_general: bool,
) -> (Su2JointContext<'h>, Su2JointContext<'h>) {
    let split = su2_joint_context(h, x, &zsplit_opts(ZRoute::Split, b_general))
        .unwrap_or_else(|e| panic!("Split context refused: {e}"));
    let explicit = su2_joint_context(h, x, &zsplit_opts(ZRoute::Explicit, b_general))
        .unwrap_or_else(|e| panic!("Explicit context refused: {e}"));
    assert_eq!(
        split.n_eff(),
        explicit.n_eff(),
        "z_route must not change n_eff"
    );
    assert_eq!(
        split.pairs().len(),
        explicit.pairs().len(),
        "z_route must not change the pair list"
    );
    assert!(
        split.n_eff() >= 1,
        "fixture carries no orbital redundancy — every κ-dependent arm would be vacuous"
    );
    (split, explicit)
}

fn unit_pair_direction(n_orb: usize, i: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n_orb];
    a[i] = 1.0;
    a
}

#[test]
fn tsz_t1_a_covector_routes_agree() {
    let fx = zsplit_fixture(0x2C_0100);
    let (split, explicit) = route_pair(&fx.h, &fx.x, false);
    let n_orb = split.pairs().len();
    assert!(n_orb > 0, "vacuous pair list");

    let (bl, br) = split.bases();
    let tb = tangent_basis(bl, br, true);
    assert!(tb.len() >= 3, "need a nontrivial tangent basis");

    let mut rng = seeded(0xA_C0_1E);
    let xi = {
        let mut v: Su2TtVector<f64> = randomized_su2_state(fx.x.space, 1, 2, &mut rng);
        v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        Su2TtTangent::project(bl, br, &v)
    };
    let zeros = vec![0.0f64; n_orb];
    let a_dir: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();

    for (label, a) in [("kappa=0", &zeros), ("kappa!=0", &a_dir)] {
        for route in [FcRoute::Sweep, FcRoute::Dots] {
            let (fxi_s, fc_s) = split.a_covector(&xi, a, route);
            let (fxi_e, fc_e) = explicit.a_covector(&xi, a, route);

            let mut scale_fxi = 0.0f64;
            let mut worst_fxi = 0.0f64;
            for slot in tb.iter().take(8) {
                let got = slot.dot(&fxi_s);
                let want = slot.dot(&fxi_e);
                scale_fxi = scale_fxi.max(got.abs()).max(want.abs());
                worst_fxi = worst_fxi.max((got - want).abs());
            }
            let scale_fc = fc_e.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let worst_fc = fc_s
                .iter()
                .zip(&fc_e)
                .fold(0.0f64, |m, (s, e)| m.max((s - e).abs()));

            assert!(
                scale_fxi > 1e-6,
                "A covector F_ξ is numerically zero ({label}, {route:?}) — the arm is vacuous"
            );
            assert!(
                scale_fc > 1e-6,
                "A covector F_c is numerically zero ({label}, {route:?}) — the arm is vacuous"
            );
            assert!(
                worst_fxi <= 1e-11 * scale_fxi,
                "A covector F_ξ Split vs Explicit ({label}, {route:?}): {worst_fxi:e} > {:e}",
                1e-11 * scale_fxi
            );
            if !a.iter().all(|v| *v == 0.0) {
                assert!(
                    worst_fxi > 0.0,
                    "★ A covector F_ξ is BIT-IDENTICAL across routes at κ ≠ 0 ({route:?}). The \
                     two routes should reach H·κ̂x by different associations; identical output \
                     means the Split arm is calling the explicit kernel — design-doc site #2 has \
                     been reverted and the standing rule is silently broken."
                );
            }
            assert!(
                worst_fc <= 1e-11 * scale_fc,
                "A covector F_c Split vs Explicit ({label}, {route:?}): {worst_fc:e} > {:e}",
                1e-11 * scale_fc
            );
        }
    }

    let (fxi_k0, _) = split.a_covector(&xi, &zeros, FcRoute::Sweep);
    let (fxi_k1, _) = split.a_covector(&xi, &a_dir, FcRoute::Sweep);
    let leg = fxi_k1.add(&fxi_k0.scale(-1.0)).norm();
    assert!(
        leg > 1e-6 * fxi_k0.norm().max(1.0),
        "the κ-dependent ambient leg of A is numerically absent (‖Δ‖ = {leg:e}) — T1 would pass \
         with project_hmul_env returning zero"
    );
}

#[test]
fn tsz_t6_kappa_chain_and_gamma_kappa_z_x_pin() {
    for b_general in [false, true] {
        let fx = zsplit_fixture(0x2C_0100);
        let (split, explicit) = route_pair(&fx.h, &fx.x, b_general);
        let (bl, br) = split.bases();
        let tb = tangent_basis(bl, br, true);
        assert!(tb.len() >= 3, "need a nontrivial tangent basis");
        let n_orb = split.pairs().len();
        assert!(n_orb >= 1, "need orbital pairs for this arm");

        assert!(
            count_theta(explicit.normal_residual()) > 0,
            "θ ≡ 1 on the normal residual — the κ̂z metric arm is vacuous (b_general={b_general})"
        );

        let mut scale_fxi = 0.0f64;
        let mut worst_fxi = 0.0f64;
        let mut scale_fc = 0.0f64;
        let mut worst_fc = 0.0f64;
        let mut nonvacuous_pairs = 0usize;

        for pair_idx in 0..n_orb {
            let a = unit_pair_direction(n_orb, pair_idx);
            let (fxi_s, fc_s) = split.b_covector(&tb[0], &a, FcRoute::Sweep);
            let (fxi_e, fc_e) = explicit.b_covector(&tb[0], &a, FcRoute::Sweep);

            let mut pair_scale = 0.0f64;
            for slot in tb.iter().take(5) {
                let got = slot.dot(&fxi_s);
                let want = slot.dot(&fxi_e);
                pair_scale = pair_scale.max(got.abs()).max(want.abs());
                scale_fxi = scale_fxi.max(got.abs()).max(want.abs());
                worst_fxi = worst_fxi.max((got - want).abs());
            }
            for (&s, &e) in fc_s.iter().zip(&fc_e) {
                pair_scale = pair_scale.max(s.abs()).max(e.abs());
                scale_fc = scale_fc.max(s.abs()).max(e.abs());
                worst_fc = worst_fc.max((s - e).abs());
            }
            if pair_scale > 1e-8 {
                nonvacuous_pairs += 1;
            }
        }

        assert!(
            nonvacuous_pairs >= n_orb / 2,
            "kappa chain vacuous over most pairs: only {nonvacuous_pairs}/{n_orb} carry content \
             (b_general={b_general})"
        );
        assert!(
            scale_fxi > 1e-6,
            "kappa chain ξ-leg vacuous overall (b_general={b_general})"
        );
        assert!(
            scale_fc > 1e-6,
            "kappa chain F_c vacuous overall (b_general={b_general})"
        );
        assert!(
            worst_fxi <= 1e-11 * scale_fxi,
            "kappa chain ξ-leg Split vs Explicit: {worst_fxi:e} > {:e} (b_general={b_general})",
            1e-11 * scale_fxi
        );
        assert!(
            worst_fc <= 1e-11 * scale_fc,
            "kappa chain F_c Split vs Explicit: {worst_fc:e} > {:e} (b_general={b_general})",
            1e-11 * scale_fc
        );
    }
}

#[test]
fn tsz_t6b_kappa_chain_is_odd_in_kappa() {
    let fx = zsplit_fixture(0x2C_0100);
    let (split, explicit) = route_pair(&fx.h, &fx.x, false);
    let (bl, br) = split.bases();
    let tb = tangent_basis(bl, br, true);
    let n_orb = split.pairs().len();

    let mut rng = seeded(0x6B_0FF);
    let a: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    let neg: Vec<f64> = a.iter().map(|v| -v).collect();

    let (_, fc_pos_s) = split.b_covector(&tb[0], &a, FcRoute::Sweep);
    let (_, fc_neg_s) = split.b_covector(&tb[0], &neg, FcRoute::Sweep);
    let (_, fc_pos_e) = explicit.b_covector(&tb[0], &a, FcRoute::Sweep);
    let (_, fc_neg_e) = explicit.b_covector(&tb[0], &neg, FcRoute::Sweep);

    let odd_s: Vec<f64> = fc_pos_s
        .iter()
        .zip(&fc_neg_s)
        .map(|(p, n)| 0.5 * (p - n))
        .collect();
    let odd_e: Vec<f64> = fc_pos_e
        .iter()
        .zip(&fc_neg_e)
        .map(|(p, n)| 0.5 * (p - n))
        .collect();
    let scale = odd_e.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = odd_s
        .iter()
        .zip(&odd_e)
        .fold(0.0f64, |m, (s, e)| m.max((s - e).abs()));
    assert!(
        scale > 1e-6,
        "the κ-ODD part of F_c is numerically zero — gamma_kappa_z_x contributes nothing here and \
         the arm is vacuous"
    );
    assert!(
        worst <= 1e-11 * scale,
        "gamma_kappa_z_x (κ-odd F_c part) Split vs Explicit: {worst:e} > {:e}",
        1e-11 * scale
    );
}

#[test]
fn tsz_t5_split_hess_apply_never_materializes_z() {
    let fx = zsplit_fixture(0x2C_0100);
    let (split, explicit) = route_pair(&fx.h, &fx.x, false);
    let n_eff = split.n_eff();
    assert!(n_eff >= 1, "need kept orbital coordinates for a full apply");

    let (bl, br) = split.bases();
    let mut rng = seeded(0x5_11FE);
    let xi = {
        let mut v: Su2TtVector<f64> = randomized_su2_state(fx.x.space, 1, 2, &mut rng);
        v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        Su2TtTangent::project(bl, br, &v)
    };
    assert!(xi.norm() > 1e-8, "(b) the tangent leg of the input is zero");
    let c: Vec<f64> = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
    assert!(
        c.iter().any(|v| v.abs() > 1e-3),
        "(b) the orbital leg of the input is zero"
    );

    let mut v = Su2JointVector::from_tangent(split.spec(), xi.clone());
    v.c = c.clone();

    assert!(
        !split.z_materialized(),
        "z was materialized before any apply — construction itself is on the explicit train"
    );
    let out = split.hess_apply(&v);
    assert!(
        out.xi.norm() > 1e-8 || out.c.iter().any(|t| t.abs() > 1e-8),
        "(c) hess_apply returned ~0 — it may be short-circuiting before it would touch z"
    );
    assert!(
        !split.z_materialized(),
        "★ STANDING RULE VIOLATED: a full hess_apply on a ZRoute::Split context materialized the \
         rank-D_H·r normal residual. No production path may form H·x."
    );

    let mut v_e = Su2JointVector::from_tangent(explicit.spec(), xi);
    v_e.c = c;
    let _ = explicit.hess_apply(&v_e);
    assert!(
        explicit.z_materialized(),
        "z_materialized() never reads true even on the Explicit route — the probe is broken and \
         the Split assertion above proves nothing"
    );
}

#[test]
fn tsz_t5b_sweep_covectors_stay_off_the_explicit_train() {
    let fx = zsplit_fixture(0x2C_0100);
    let (split, _explicit) = route_pair(&fx.h, &fx.x, true);
    let n_orb = split.pairs().len();
    let (bl, br) = split.bases();
    let tb = tangent_basis(bl, br, true);

    let mut rng = seeded(0x5B_A11);
    let a: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();

    let (fxi_a, fc_a) = split.a_covector(&tb[0], &a, FcRoute::Sweep);
    let (fxi_b, fc_b) = split.b_covector(&tb[0], &a, FcRoute::Sweep);
    assert!(
        fxi_a.norm() > 1e-8 && fxi_b.norm() > 1e-8,
        "covector legs are vacuous — nothing was exercised"
    );
    assert!(
        fc_a.iter().chain(fc_b.iter()).any(|v| v.abs() > 1e-8),
        "F_c rows are vacuous — nothing was exercised"
    );
    assert!(
        !split.z_materialized(),
        "★ STANDING RULE VIOLATED: a Sweep-route covector materialized z"
    );

    let _ = split.b_covector(&tb[0], &a, FcRoute::Dots);
    assert!(
        split.z_materialized(),
        "FcRoute::Dots must stay on the materialized twin — its independence from the production \
         assembly is the whole reason it certifies anything"
    );
}

#[test]
fn tsz_t7_gradient_paths_agree_bitwise_and_are_deterministic() {
    let fx = zsplit_fixture(0x2C_0100);
    let x = perturbed_state(&fx.x, 5e-2, 0x2C_0700);
    let opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..zsplit_opts(ZRoute::Split, false)
    };
    let ctx = su2_joint_context(&fx.h, &x, &opts)
        .unwrap_or_else(|e| panic!("Split context refused at the perturbed point: {e}"));
    let (bl, br) = ctx.bases();

    let g_ctx = ctx.weingarten().gradient();
    let g_env = riemannian_gradient_env(ctx.hamiltonian(), bl, br, ctx.lambda());

    assert!(
        g_ctx.num_coeffs() > 0,
        "gradient has no coefficients — vacuous arm"
    );
    let scale = (0..g_ctx.num_coeffs()).fold(0.0f64, |m, i| m.max(g_ctx.coeff(i).abs()));
    assert!(scale > 1e-12, "gradient is numerically zero — vacuous arm");
    assert_eq!(
        g_env.num_coeffs(),
        g_ctx.num_coeffs(),
        "riemannian_gradient_env and ctx.gradient() disagree on tangent shape"
    );
    for i in 0..g_ctx.num_coeffs() {
        assert_eq!(
            g_env.coeff(i),
            g_ctx.coeff(i),
            "riemannian_gradient_env vs ctx.gradient(): coeff {i} differs — the two callers of \
             hz_stacks_and_gradient must run the identical body on identical inputs"
        );
    }

    let g_again = riemannian_gradient_env(ctx.hamiltonian(), bl, br, ctx.lambda());
    for i in 0..g_ctx.num_coeffs() {
        assert_eq!(
            g_again.coeff(i),
            g_env.coeff(i),
            "riemannian_gradient_env is NOT deterministic at coeff {i}"
        );
    }

    let digest = (0..g_ctx.num_coeffs()).fold(0xcbf29ce484222325u64, |h, i| {
        (h ^ g_ctx.coeff(i).to_bits()).wrapping_mul(0x100000001b3)
    });
    println!(
        "tsz_t7 digest = {digest:#018x} over {} coeffs",
        g_ctx.num_coeffs()
    );
}
