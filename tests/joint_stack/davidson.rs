// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::{TtTangent, tangent_basis};
use qntt::solvers::hess::{SymplecticDavidsonOpts, SymplecticLowestError, SymplecticSpectrum};
use qntt::solvers::joint::{
    FcRoute, JointContext, JointOpts, JointVector, ZRoute, joint_context, joint_lowest_davidson,
};
use qntt::{QnSpace, SparseHamiltonian, TtVector, dot, xt_h_y};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::optimize::{d2_fixture, d2_point, d2r22_point};
use crate::util::*;

fn joint_seed_block(
    jctx: &JointContext<'_>,
    b: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<JointVector> {
    let (base_l, base_r) = jctx.bases();
    let spec = jctx.spec();
    let n_eff = jctx.n_eff();
    (0..b)
        .map(|_| {
            let dir: TtVector<f64> = randomized_state(base_l.space, 1, 2, rng);
            let xi = TtTangent::project(base_l, base_r, &dir);
            let mut v = JointVector::from_tangent(spec, xi);
            v.c = (0..n_eff)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect();
            v
        })
        .collect()
}

fn route_pair<'h>(
    ham: &'h SparseHamiltonian<f64>,
    x: &TtVector<f64>,
) -> (JointContext<'h>, JointContext<'h>) {
    let split = joint_context(
        ham,
        x,
        &JointOpts {
            z_route: ZRoute::Split,
            ..JointOpts::default()
        },
    )
    .expect("Split route must be TT-critical");
    let explicit = joint_context(
        ham,
        x,
        &JointOpts {
            z_route: ZRoute::Explicit,
            ..JointOpts::default()
        },
    )
    .expect("Explicit route must be TT-critical");
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
    (split, explicit)
}

fn assert_routes_agree(
    split: &JointContext<'_>,
    explicit: &JointContext<'_>,
    tol: f64,
    label: &str,
) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE07E);
    let vecs = joint_seed_block(split, 4, &mut rng);

    let mut scale_a = 0.0f64;
    let mut worst_a_xi = 0.0f64;
    let mut worst_a_c = 0.0f64;
    let mut scale_b = 0.0f64;
    let mut worst_b_xi = 0.0f64;
    let mut worst_b_c = 0.0f64;
    for v in &vecs {
        let a_s = split.a_apply(v);
        let a_e = explicit.a_apply(v);
        scale_a = scale_a.max(a_s.xi.norm()).max(a_e.xi.norm());
        let mut dxi = a_s.xi.clone();
        dxi.axpy(-1.0, &a_e.xi);
        worst_a_xi = worst_a_xi.max(dxi.norm());
        worst_a_c = worst_a_c.max(
            a_s.c
                .iter()
                .zip(&a_e.c)
                .fold(0.0f64, |acc, (&s, &e)| acc.max((s - e).abs())),
        );

        let b_s = split.b_apply(v);
        let b_e = explicit.b_apply(v);
        scale_b = scale_b.max(b_s.xi.norm()).max(b_e.xi.norm());
        let mut dbxi = b_s.xi.clone();
        dbxi.axpy(-1.0, &b_e.xi);
        worst_b_xi = worst_b_xi.max(dbxi.norm());
        worst_b_c = worst_b_c.max(
            b_s.c
                .iter()
                .zip(&b_e.c)
                .fold(0.0f64, |acc, (&s, &e)| acc.max((s - e).abs())),
        );
    }
    assert!(
        scale_a > 1e-6,
        "[{label}] A apply vacuous in route comparison"
    );
    assert!(
        scale_b > 1e-6,
        "[{label}] B apply vacuous in route comparison"
    );
    assert!(
        worst_a_xi <= tol * scale_a.max(1.0),
        "[{label}] a_apply ξ-leg Split vs Explicit: {worst_a_xi:e} > {:e}",
        tol * scale_a.max(1.0)
    );
    assert!(
        worst_a_c <= tol * scale_a.max(1.0),
        "[{label}] a_apply c-leg Split vs Explicit: {worst_a_c:e} > {:e}",
        tol * scale_a.max(1.0)
    );
    assert!(
        worst_b_xi <= tol * scale_b.max(1.0),
        "[{label}] b_apply ξ-leg Split vs Explicit: {worst_b_xi:e} > {:e}",
        tol * scale_b.max(1.0)
    );
    assert!(
        worst_b_c <= tol * scale_b.max(1.0),
        "[{label}] b_apply c-leg Split vs Explicit: {worst_b_c:e} > {:e}",
        tol * scale_b.max(1.0)
    );
}

#[test]
fn tjd_env_eq_explicit() {
    {
        let d = 5usize;
        let space = QnSpace::new(d, 2, 1);
        let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
        let (split, explicit) = route_pair(&ham, &x);
        assert!(split.n_eff() >= 1, "need orbital redundancy for this arm");
        assert_routes_agree(&split, &explicit, 1e-9, "d5-synthetic");
    }

    {
        let fx = d2_fixture();
        let out = &fx.out;
        assert!(
            out.info.converged,
            "D2 fixture must converge: {:?}",
            out.info.refused_reason
        );
        let (split, explicit) = route_pair(&out.h, &out.x);
        assert!(split.n_eff() >= 1, "D2 fixture must be genuinely truncated");
        assert_routes_agree(&split, &explicit, 1e-11, "D2");
    }
}

#[test]
fn tjd_kappa_chain_pin() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let (split, explicit) = route_pair(&ham, &x);
    assert!(split.n_eff() >= 1, "need orbital redundancy for this arm");

    let (base_l, base_r) = split.bases();
    let tb = tangent_basis(base_l, base_r, true);
    assert!(tb.len() >= 3, "need a nontrivial tangent basis");
    let n_orb = split.pairs().len();
    assert!(n_orb >= 1, "need orbital pairs for this arm");

    let mut scale_fxi = 0.0f64;
    let mut worst_fxi = 0.0f64;
    let mut scale_fc = 0.0f64;
    let mut worst_fc = 0.0f64;
    let mut nonvacuous_pairs = 0usize;

    for pair_idx in 0..n_orb {
        let mut a = vec![0.0f64; n_orb];
        a[pair_idx] = 1.0;

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
        "kappa chain vacuous over most pairs: only {nonvacuous_pairs}/{n_orb} pairs carry content"
    );
    assert!(scale_fxi > 1e-6, "kappa chain ξ-leg vacuous overall");
    assert!(scale_fc > 1e-6, "kappa chain F_c vacuous overall");
    assert!(
        worst_fxi <= 1e-12 * scale_fxi.max(1.0),
        "kappa chain ξ-leg Split vs Explicit: {worst_fxi:e} > {:e}",
        1e-12 * scale_fxi.max(1.0)
    );
    assert!(
        worst_fc <= 1e-12 * scale_fc.max(1.0),
        "kappa chain F_c Split vs Explicit: {worst_fc:e} > {:e}",
        1e-12 * scale_fc.max(1.0)
    );
}

#[test]
fn tjd_raised_diag_pin() {
    let fx = d2_fixture();
    let out = &fx.out;
    assert!(
        out.info.converged,
        "D2 fixture must converge: {:?}",
        out.info.refused_reason
    );
    let jctx = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("the D2 solution must be joint-critical");
    let n_eff = jctx.n_eff();
    assert!(n_eff >= 1, "need orbital redundancy for this arm");
    let lambda = jctx.lambda();
    let mu = jctx.orb_kept();
    assert_eq!(mu.len(), n_eff);
    let mu_min = mu.iter().cloned().fold(f64::INFINITY, f64::min);
    let mu_max = mu.iter().cloned().fold(0.0f64, f64::max).max(1e-300);
    assert!(
        mu_min < 1e-3 * mu_max,
        "μ spread not real (arm would be vacuous): μ_min={mu_min:e}, μ_max={mu_max:e}"
    );

    let dirs = jctx.orbital_directions();
    let projs = jctx.orbital_projections();
    let n_orb = dirs.len();
    assert!(n_orb >= 1);
    let spec = jctx.spec().clone();

    let d_prod = qntt::solvers::joint::joint_raised_orbital_diagonal_diagnostic(&jctx);
    assert_eq!(d_prod.len(), n_eff);

    let mut max_true = 0.0f64;
    let mut max_raised = 0.0f64;
    let mut worst_raised_rel = 0.0f64;
    let mut worst_naive_overshoot_rel = 0.0f64;
    let mut worst_prod_vs_oracle_rel = 0.0f64;

    for j in 0..n_eff {
        let mut ej = vec![0.0f64; n_eff];
        ej[j] = 1.0;
        let a = jctx.v_times(&ej);

        let mut u_hat = &dirs[0] * a[0];
        for idx in 1..n_orb {
            u_hat = &u_hat + &(&dirs[idx] * a[idx]);
        }
        let mut tau_v = projs[0].clone();
        tau_v.scale_mut(a[0]);
        for idx in 1..n_orb {
            tau_v.axpy(a[idx], &projs[idx]);
        }
        let tau_v_vec = tau_v.to_vector();

        let f_c_true = xt_h_y(&u_hat, &out.h, &u_hat) - lambda * dot(&u_hat, &u_hat);
        let cross = xt_h_y(&tau_v_vec, &out.h, &u_hat) - lambda * dot(&tau_v_vec, &u_hat);
        let raised = (f_c_true - cross) / mu[j];
        let naive = f_c_true / mu[j];

        let e_jv = JointVector::from_coeffs(&spec, &jctx, ej.clone());
        let d_true = jctx.a_apply(&e_jv).c[j];

        max_true = max_true.max(d_true.abs());
        max_raised = max_raised.max(raised.abs());
        worst_raised_rel = worst_raised_rel.max((raised - d_true).abs() / (1.0 + d_true.abs()));
        worst_naive_overshoot_rel =
            worst_naive_overshoot_rel.max((naive - d_true).abs() / (1.0 + d_true.abs()));
        worst_prod_vs_oracle_rel =
            worst_prod_vs_oracle_rel.max((d_prod[j] - d_true).abs() / (1.0 + d_true.abs()));
    }

    assert!(max_true <= 1e3, "dense A-diagonal not O(1): {max_true:e}");
    assert!(
        max_raised <= 1e3,
        "raised diagonal not O(1) (the Schur cross term should cancel the 1/μ blowup): {max_raised:e}"
    );
    assert!(
        worst_raised_rel <= 1e-6 * (1.0 + max_true),
        "raised diagonal vs dense a_apply oracle: {worst_raised_rel:e}"
    );
    assert!(
        worst_prod_vs_oracle_rel <= 1e-6 * (1.0 + max_true),
        "production raised_orbital_diagonal vs dense a_apply oracle: {worst_prod_vs_oracle_rel:e}"
    );

    eprintln!(
        "tjd_raised_diag_pin: naive overshoot rel = {worst_naive_overshoot_rel:e} at μ_min={mu_min:e}, \
         μ_max={mu_max:e}"
    );
    assert!(
        worst_naive_overshoot_rel >= 1e-2,
        "naive arm expected to overshoot by O(1/μ) at this μ spread; measured only \
         {worst_naive_overshoot_rel:e} — arm may be vacuous"
    );
}

#[test]
fn tjd_white_davidson_matches_dense_to_roundoff() {
    let p = d2_point();
    let jctx = joint_context(&p.h, &p.x, &JointOpts::default())
        .expect("the tracked D2 endpoint must be joint-critical");
    assert!(jctx.n_eff() >= 1, "the white route needs an orbital block");
    let (basis, _report) = jctx.joint_basis();
    let SymplecticSpectrum::Stable { omega: omega_dense } =
        jctx.joint_symplectic_eigenvalues(&basis)
    else {
        panic!("expected a stable D2 joint spectrum");
    };
    let k = 4usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDD03);
    let v0s = joint_seed_block(&jctx, k + 2, &mut rng);
    let (omega, info) = joint_lowest_davidson(&jctx, &v0s, k, &SymplecticDavidsonOpts::default())
        .expect("D2 joint Davidson must not error");
    assert!(
        info.converged,
        "D2 joint Davidson must converge: {:?}",
        info.residuals
    );
    let errs: Vec<f64> = omega
        .iter()
        .zip(&omega_dense)
        .map(|(w, d)| (w - d).abs() / d)
        .collect();
    eprintln!(
        "tjd_white_davidson: built {} | rel ω err {:?}",
        info.iterations,
        errs.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>()
    );
    for (i, &e) in errs.iter().enumerate() {
        assert!(
            e <= 2e-13,
            "D2 ω[{i}]: Davidson {} vs dense {} (rel {e:.2e})",
            omega[i],
            omega_dense[i]
        );
    }
}

#[test]
fn tjd_condensed_davidson_matches_dense_at_stiff_point() {
    let p = d2r22_point();
    let jctx = joint_context(&p.h, &p.x, &JointOpts::default())
        .expect("the tracked stiff endpoint must be joint-critical");
    let (basis, _report) = jctx.joint_basis();
    let SymplecticSpectrum::Stable { omega: omega_dense } =
        jctx.joint_symplectic_eigenvalues(&basis)
    else {
        panic!("expected a stable joint spectrum at the stiff tracked optimum");
    };
    let k = 4usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDD07);
    let v0s = joint_seed_block(&jctx, k + 2, &mut rng);
    let (omega, info) = joint_lowest_davidson(&jctx, &v0s, k, &SymplecticDavidsonOpts::default())
        .expect("stiff-point joint Davidson must not error");
    assert!(info.converged, "must converge: {:?}", info.residuals);
    let errs: Vec<f64> = omega
        .iter()
        .zip(&omega_dense)
        .map(|(w, d)| (w - d).abs() / d)
        .collect();
    eprintln!(
        "tjd_condensed_stiff: n {} built {} applies {} | rel ω err {:?}",
        basis.len(),
        info.iterations,
        info.applies,
        errs.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>()
    );
    for (i, &e) in errs.iter().enumerate() {
        assert!(
            e <= BOUND_STIFF,
            "stiff ω[{i}]: Davidson {} vs dense {} (rel {e:.2e})",
            omega[i],
            omega_dense[i]
        );
    }
}

const BOUND_STIFF: f64 = 3e-13;

#[test]
fn tjd_davidson_eq_dense() {
    {
        let d = 3usize;
        let space = QnSpace::new(d, 1, 1);
        let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B2, 1, 1);
        let jctx = joint_context(&ham, &x, &JointOpts::default())
            .expect("full-rank fixture must be TT-critical");
        assert_eq!(jctx.n_eff(), 0, "fixture must be full rank for this arm");

        let k = 3usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDD01);
        let v0s = joint_seed_block(&jctx, k + 2, &mut rng);
        let opts = SymplecticDavidsonOpts::default();
        let (omega, info) = joint_lowest_davidson(&jctx, &v0s, k, &opts)
            .expect("full-rank joint Davidson must not error");
        assert!(
            info.converged,
            "full-rank joint Davidson must converge: {:?}",
            info.residuals
        );

        let (basis, _report) = jctx.joint_basis();
        let SymplecticSpectrum::Stable { omega: omega_dense } =
            jctx.joint_symplectic_eigenvalues(&basis)
        else {
            panic!("expected a stable full-rank joint spectrum");
        };
        assert_eq!(omega.len(), k);
        for (i, &w) in omega.iter().enumerate() {
            let want = omega_dense[i];
            assert!(
                (w - want).abs() <= 1e-8 * want.max(1.0),
                "full-rank ω[{i}]: Davidson {w} vs dense {want}"
            );
        }
    }

    {
        let fx = d2_fixture();
        let out = &fx.out;
        assert!(
            out.info.converged,
            "D2 fixture must converge: {:?}",
            out.info.refused_reason
        );
        let jctx = joint_context(&out.h, &out.x, &JointOpts::default())
            .expect("the D2 solution must be joint-critical");
        assert!(jctx.n_eff() >= 1, "D2 fixture must be genuinely truncated");

        let k = 4usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDD03);
        let v0s = joint_seed_block(&jctx, k + 2, &mut rng);
        let opts = SymplecticDavidsonOpts::default();
        let (omega, info) =
            joint_lowest_davidson(&jctx, &v0s, k, &opts).expect("D2 joint Davidson must not error");
        assert!(
            info.converged,
            "D2 joint Davidson must converge: {:?}",
            info.residuals
        );
        let (basis, _report) = jctx.joint_basis();
        let SymplecticSpectrum::Stable { omega: omega_dense } =
            jctx.joint_symplectic_eigenvalues(&basis)
        else {
            panic!("expected a stable D2 joint spectrum");
        };
        assert_eq!(omega.len(), k);
        for (i, &w) in omega.iter().enumerate() {
            let want = omega_dense[i];
            assert!(
                (w - want).abs() <= 1e-8 * want.max(1.0),
                "D2 ω[{i}]: Davidson {w} vs dense {want}"
            );
        }
    }
}

#[test]
fn tjd_publish_gate() {
    let fx = d2_fixture();
    let out = &fx.out;
    assert!(
        out.info.converged,
        "D2 fixture must converge: {:?}",
        out.info.refused_reason
    );
    let jctx = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("the D2 solution must be joint-critical");
    assert!(jctx.n_eff() >= 1, "D2 fixture must be genuinely truncated");

    let k = 4usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDD40);
    let v0s = joint_seed_block(&jctx, k + 2, &mut rng);
    let opts = SymplecticDavidsonOpts {
        max_basis: 4,
        ..SymplecticDavidsonOpts::default()
    };
    match joint_lowest_davidson(&jctx, &v0s, k, &opts) {
        Ok((_omega, info)) => {
            assert!(
                !info.converged,
                "a starved max_basis must NOT be reported as converged -- the publish-gate \
                 discipline (joint_lowest_davidson's own module doc) requires callers to check \
                 `converged` before trusting ω: an uncertified endpoint has been measured to look \
                 self-consistent while being 13% wrong"
            );
        }
        Err(SymplecticLowestError::MetricNotPd { .. })
        | Err(SymplecticLowestError::NegativeRitz { .. }) => {}
    }
}
