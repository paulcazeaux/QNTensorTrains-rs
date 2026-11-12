// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::core::tangent::TtTangent;
use qntt::{QnSpace, Scalar, TtVector, rayleigh_quotient};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t8m_newton_polish_recovers_minimizer() {
    use qntt::solvers::hess::{NewtonPolishOpts, newton_polish};
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let lam0 = rayleigh_quotient(&ham, &x).re_s();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81D0);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let mut xi = TtTangent::project(&base_l, &base_r, &dir);
    let n = xi.norm();
    xi = xi.scale(3e-2 / n);
    let xp = xi.retract_hosvd(1.0);
    let gp = grad_exact_at(&ham, &xp).norm();
    assert!(gp > 1e-4, "perturbation vacuous (‖grad‖={gp:.1e})");

    let opts_n = NewtonPolishOpts {
        gradtol: 1e-11,
        ..Default::default()
    };
    let (xn, info) = newton_polish(&ham, &xp, &opts_n);
    println!(
        "t8m_newton: outer={} applies={} gnorm={:.2e} λ={:.12} (λ₀={lam0:.12})",
        info.outer, info.applies, info.gnorm, info.lambda
    );
    assert!(
        info.gnorm <= 1e-9,
        "polish must reach the 1e-9 class (got {:.2e}; converged flag: {})",
        info.gnorm,
        info.converged
    );
    let gn_check = grad_exact_at(&ham, &xn).norm();
    assert!(
        gn_check <= 1e-9,
        "independent gradient check: {gn_check:.2e}"
    );
    assert!(
        (info.lambda - lam0).abs() < 1e-9,
        "polished energy {} vs minimizer {}",
        info.lambda,
        lam0
    );
}

#[test]
fn t8m_pd_probe_matches_dense() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        HessPart, PdVerdict, hessian_grams, hessian_pd_probe, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    let (a, b) = hessian_grams(&ctx, &basis);
    let dense_min = |sign: f64| -> f64 {
        let mut m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = a[i * n + j] + sign * b[i * n + j];
            }
        }
        sym_eigvals_jacobi(m)[0]
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E0);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let v0 = TtTangent::project(&base_l, &base_r, &dir);
    for (part, sign) in [(HessPart::Rr, 1.0), (HessPart::Ii, -1.0)] {
        let dmin = dense_min(sign);
        assert!(
            dmin > 0.0,
            "fixture must be stable ({part:?}: dense λ_min = {dmin})"
        );
        let (verdict, applies) = hessian_pd_probe(&ctx, part, &v0, 60);
        match verdict {
            PdVerdict::LikelyPd {
                lambda_min,
                residual,
            } => {
                println!(
                    "t8m_pd_probe {part:?}: λ_min {lambda_min:.6} (dense {dmin:.6}, res {residual:.1e}, {applies} applies)"
                );
                assert!(
                    lambda_min >= dmin - 1e-10,
                    "probe below the true minimum?! {lambda_min} vs {dmin}"
                );
                assert!(
                    lambda_min - dmin <= residual + 0.5 * dmin.abs() + 1e-6,
                    "probe estimate too far above dense: {lambda_min} vs {dmin} (res {residual})"
                );
            }
            PdVerdict::Inconclusive {
                lambda_min,
                residual,
            } => panic!(
                "probe ran out of budget at a small fixture ({part:?}: λ_min {lambda_min}, res {residual}) \
                 — the calibration premise is that 60 applies converge here"
            ),
            PdVerdict::WitnessNegative { lambda } => {
                panic!("false negative witness at a certified-stable point ({part:?}: {lambda})")
            }
        }
    }
}

#[test]
fn t8m_certify_probe_block_matches_dense() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::davidson::{DAVIDSON_MAXSUBSPACE, DAVIDSON_NWARM, DAVIDSON_RESTART_KEEP};
    use qntt::solvers::hess::{
        HessPart, PdVerdict, ProbeDavidson, hessian_grams, weingarten_operator,
    };
    use qntt::solvers::riemannian::certify_probe;
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let lambda = rayleigh_quotient(&ham, &x);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    let (a, b) = hessian_grams(&ctx, &basis);
    let dense_eigs = |sign: f64| -> Vec<f64> {
        let mut m = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                m[i][j] = a[i * n + j] + sign * b[i * n + j];
            }
        }
        sym_eigvals_jacobi(m)
    };
    let bauer_fike = |eigs: &[f64], lambda_min: f64, residual: f64, what: &str| {
        let gap = eigs
            .iter()
            .map(|e| (lambda_min - e).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            gap <= residual + 1e-8,
            "{what}: LikelyPd {lambda_min} is {gap:e} from every dense eigenvalue (res {residual}) — a ghost"
        );
    };
    let dflt = ProbeDavidson::default();
    assert_eq!(
        (dflt.maxsubspace, dflt.restart_keep, dflt.nwarm, dflt.block),
        (48, 12, 2, 4),
        "the shipped PD-probe Davidson changed. This pins the 2026-09-09 measured decision \
         (blocks 4/8 are the only settings that found the bottom pair on the hcho cap-256 \
         cell, and the concurrent batch makes block-4 1.09× the single-vector wall). If the \
         default is being moved on purpose, re-measure that cell and update the record"
    );
    let retired = ProbeDavidson {
        maxsubspace: DAVIDSON_MAXSUBSPACE,
        restart_keep: DAVIDSON_RESTART_KEEP,
        nwarm: DAVIDSON_NWARM,
        block: 1,
    };
    let single16 = ProbeDavidson {
        maxsubspace: 16,
        ..retired
    };
    let block4 = ProbeDavidson {
        maxsubspace: 32,
        restart_keep: 8,
        nwarm: 4,
        block: 4,
    };
    for (part, sign) in [(HessPart::Rr, 1.0), (HessPart::Ii, -1.0)] {
        let eigs = dense_eigs(sign);
        let dmin = eigs[0];
        assert!(dmin > 0.0, "fixture must be stable ({part:?}: {dmin})");
        let (v, _, applies) = certify_probe(&ham, &x, lambda, part, 90, 0x5EED, &retired);
        println!(
            "t8m_certify_probe RETIRED 8/1/2/1 {part:?}: {v:?} after {applies} applies (dense {dmin:.6})"
        );
        if let PdVerdict::LikelyPd {
            lambda_min,
            residual,
        } = v
        {
            bauer_fike(&eigs, lambda_min, residual, "RETIRED 8/1/2/1");
        }
        for (name, dav, exact) in [
            ("default 48/12/2/4", &dflt, true),
            ("single16", &single16, false),
            ("block4", &block4, true),
        ] {
            let (verdict, _w, applies) = certify_probe(&ham, &x, lambda, part, 90, 0x5EED, dav);
            match verdict {
                PdVerdict::LikelyPd {
                    lambda_min,
                    residual,
                } => {
                    println!(
                        "t8m_certify_probe {name} {part:?}: λ_min {lambda_min:.9} (dense {dmin:.9}, res {residual:.1e}, {applies} applies)"
                    );
                    assert!(
                        lambda_min >= dmin - 1e-10,
                        "{name}: {lambda_min} < dense {dmin}"
                    );
                    bauer_fike(&eigs, lambda_min, residual, name);
                    let slack = if exact { 1e-8 } else { residual + 1e-8 };
                    assert!(
                        lambda_min - dmin <= slack,
                        "{name}: {lambda_min} vs dense {dmin} (res {residual})"
                    );
                }
                other => panic!("{name} {part:?}: expected LikelyPd, got {other:?}"),
            }
        }
    }
}
