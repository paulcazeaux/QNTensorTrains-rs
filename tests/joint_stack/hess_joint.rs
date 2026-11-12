// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::{TtTangent, tangent_basis};
use qntt::rdm::{transition_1rdm, transition_1rdm_h};
use qntt::rotation::PairMask;
use qntt::solvers::InnerProductVector;
use qntt::solvers::hess::{
    SymplecticSpectrum, hessian_grams, normal_residual, symplectic_from_grams, weingarten_operator,
};
use qntt::solvers::joint::{FcRoute, JointGateError, JointOpts, JointVector, joint_context};
use qntt::{QnSpace, rayleigh_quotient};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::*;

#[test]
fn tjh_b1_fullrank_degeneration() {
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B2, 1, 1);

    let dets = dets_for(d, 1, 1);
    let (t, v) = generic_truncated_tv(d);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let (evals, _evecs) = dense_eigen(&hd);
    let lambda = rayleigh_quotient(&ham, &x);
    assert!(
        (lambda - evals[0]).abs() < 1e-8,
        "manifold minimizer vs dense ground energy: {lambda} vs {}",
        evals[0]
    );

    let jctx = joint_context(&ham, &x, &JointOpts::default())
        .expect("full-rank fixture must be TT-critical");
    assert_eq!(
        jctx.n_eff(),
        0,
        "full rank ⇒ orbital directions entirely tangential"
    );
    let mu_max = jctx
        .orb_spectrum()
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(1.0);
    assert!(
        jctx.orb_spectrum()
            .iter()
            .all(|&m| m.abs() <= 1e-8 * mu_max),
        "orb_spectrum not a pure-roundoff cluster at full rank: {:?}",
        jctx.orb_spectrum()
    );

    let wctx = weingarten_operator(&ham, &x);
    let tb = tangent_basis(&base_l, &base_r, true);
    assert_eq!(tb.len(), dets.len() - 1, "projective tangent dimension");
    let (a_tt, b_tt) = hessian_grams(&wctx, &tb);
    let bmax = b_tt.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax < 1e-10,
        "B must vanish on the full manifold: {bmax:.3e}"
    );
    let SymplecticSpectrum::Stable { omega: omega_tt } =
        symplectic_from_grams(&a_tt, &b_tt, tb.len(), 1e-10)
    else {
        panic!("expected a stable TT spectrum at the ground state");
    };

    let (basis, report) = jctx.joint_basis();
    assert_eq!(basis.len(), tb.len(), "joint basis length at full rank");
    assert_eq!(report.n_tt, tb.len());
    assert_eq!(report.n_orb_kept, 0);
    assert_eq!(report.dropped, 0);

    let SymplecticSpectrum::Stable { omega: omega_joint } =
        jctx.joint_symplectic_eigenvalues(&basis)
    else {
        panic!("expected a stable joint spectrum at the ground state");
    };
    assert_eq!(omega_joint.len(), omega_tt.len());
    for (k, (&wj, &wt)) in omega_joint.iter().zip(&omega_tt).enumerate() {
        assert!(
            (wj - wt).abs() < 1e-9 * wt.max(1.0),
            "ω[{k}]: joint {wj} vs TT-only {wt}"
        );
    }
    for (k, &w) in omega_tt.iter().enumerate() {
        let want = evals[k + 1] - evals[0];
        assert!(
            (w - want).abs() < 1e-6 * want.max(1.0),
            "ω[{k}] = {w} vs dense FCI gap E_{} − E_0 = {want}",
            k + 1
        );
    }
}

#[test]
fn tjh_b3_scalar_fd_fullrank() {
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, x, _base_l, _base_r) = weingarten_minimizer(space, 0xB0B3, 1, 1);
    let jctx = joint_context(&ham, &x, &JointOpts::default())
        .expect("full-rank fixture must be TT-critical");
    assert_eq!(jctx.n_eff(), 0, "fixture must be full rank for this arm");

    let (basis, _report) = jctx.joint_basis();
    assert!(basis.len() >= 2, "need a nontrivial tangent space");
    let n = basis.len();
    let (a, b) = jctx.joint_hessian_grams(&basis);
    let bmax = b.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax < 1e-10,
        "B unexpectedly nonzero at full rank (z should be 0): {bmax:.3e}"
    );

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (j, vj) in basis.iter().enumerate() {
        let fd = hess_fd_central(&ham, &x, &vj.xi, 1e-4);
        for (i, vi) in basis.iter().enumerate() {
            let fdij = vi.xi.dot(&fd);
            let hij = a[i * n + j] + b[i * n + j];
            num += (fdij - hij) * (fdij - hij);
            den += fdij * fdij;
        }
    }
    let rel = (num / den).sqrt();
    assert!(
        rel < 1e-6,
        "dense FD matrix vs joint A+B (full rank): Frobenius-relative error {rel:.3e}"
    );
}

#[test]
fn tjh_gate_refusal() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, xstar, bl_star, br_star) = weingarten_minimizer(space, 0xB0B0, 1, 2);

    let jctx = joint_context(&ham, &xstar, &JointOpts::default())
        .expect("expected success at a TT-critical point");
    assert!(
        jctx.tt_gradnorm() < 1e-9,
        "gate accessor disagrees with the fixture's own criticality: {:.3e}",
        jctx.tt_gradnorm()
    );

    let x = noncritical_point(&xstar, &bl_star, &br_star, 3e-2, 0x81D1);
    let gnorm_want = grad_exact_at(&ham, &x).norm();
    assert!(gnorm_want > 1e-3, "perturbation vacuous: {gnorm_want:.3e}");
    let err = match joint_context(&ham, &x, &JointOpts::default()) {
        Err(e) => e,
        Ok(_) => panic!("expected a gate refusal at a non-critical point"),
    };
    match err {
        JointGateError::NotTtCritical {
            measured, bound, ..
        } => {
            assert!(
                measured > 1e-3,
                "gate-reported measured gradnorm too small: {measured:.3e}"
            );
            assert!(bound > 0.0, "gate bound must be positive: {bound:.3e}");
            assert!(
                (measured - gnorm_want).abs() < 1e-9 * gnorm_want,
                "gate-reported measured {measured:.3e} vs independent grad_exact_at {gnorm_want:.3e}"
            );
        }
        other => panic!("expected JointGateError::NotTtCritical, got {other:?}"),
    }

    let opts_inf = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..JointOpts::default()
    };
    let jctx_inf =
        joint_context(&ham, &x, &opts_inf).expect("INFINITY tt_grad_tol must always succeed");
    assert!(
        jctx_inf.tt_gradnorm() > 1e-3,
        "diagnosis-only context should report the real gradnorm"
    );
    assert!(
        (jctx_inf.tt_gradnorm() - gnorm_want).abs() <= 1e-9 * jctx_inf.tt_gradnorm(),
        "diagnosis-only tt_gradnorm vs independent grad_exact_at"
    );

    let opts_neg = JointOpts {
        orb_floor: -1.0,
        ..JointOpts::default()
    };
    let err2 = match joint_context(&ham, &xstar, &opts_neg) {
        Err(e) => e,
        Ok(_) => panic!("negative orb_floor must be refused"),
    };
    match err2 {
        JointGateError::OrbFloorNegative { got } => assert_eq!(got, -1.0),
        other => panic!("expected JointGateError::OrbFloorNegative, got {other:?}"),
    }
}

#[test]
fn tjh_basis() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let jctx = joint_context(&ham, &x, &JointOpts::default()).expect("fixture must be TT-critical");
    assert!(
        jctx.n_eff() >= 1,
        "fixture must carry genuine orbital redundancy for this arm (n_eff={})",
        jctx.n_eff()
    );

    let (bl, br) = jctx.bases();
    let tb = tangent_basis(bl, br, true);
    let (basis, report) = jctx.joint_basis();

    assert_eq!(
        basis.len(),
        tb.len() + jctx.n_eff(),
        "tangent block + n_eff orbital directions"
    );
    assert_eq!(report.n_tt, tb.len());
    assert_eq!(report.n_orb_kept, jctx.n_eff());
    assert_eq!(
        report.orb_spectrum.len(),
        jctx.pairs().len(),
        "full orb_spectrum carried"
    );
    assert!(
        report.orb_spectrum.windows(2).all(|w| w[0] <= w[1] + 1e-12),
        "orb_spectrum must be ascending: {:?}",
        report.orb_spectrum
    );
    assert!(report.dropped_mass >= 0.0);
    assert_eq!(report.dropped, report.dropped_idx.len());

    let mut worst = 0.0f64;
    for (i, vi) in basis.iter().enumerate() {
        for (j, vj) in basis.iter().enumerate() {
            let g = vi.dot(vj);
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((g - want).abs());
        }
    }
    assert!(
        worst < 1e-6,
        "joint basis not joint-orthonormal: worst deviation {worst:.3e}"
    );
}

#[test]
fn tjh_fc_routes() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, _bl0, _br0) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let jctx = joint_context(&ham, &x, &JointOpts::default()).expect("fixture must be TT-critical");
    let (bl, br) = jctx.bases();
    let tb = tangent_basis(bl, br, true);
    assert!(!tb.is_empty());
    let n_orb = jctx.pairs().len();
    assert!(n_orb >= 1, "need orbital pairs for this arm");
    let lambda = jctx.lambda();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFC01);
    let a: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    let xi = tb[0].clone();

    let (fxi_s, fc_s) = jctx.a_covector(&xi, &a, FcRoute::Sweep);
    let (fxi_d, fc_d) = jctx.a_covector(&xi, &a, FcRoute::Dots);
    let mut diff = fxi_s.clone();
    diff.axpy(-1.0, &fxi_d);
    assert!(
        diff.norm() <= 1e-9 * fxi_s.norm().max(1.0),
        "A: sweep vs dots ξ-leg mismatch {:.3e}",
        diff.norm()
    );
    let cscale_a = fc_s
        .iter()
        .cloned()
        .fold(0.0f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let cmax_a = fc_s
        .iter()
        .zip(&fc_d)
        .fold(0.0f64, |acc, (&s, &dd)| acc.max((s - dd).abs()));
    assert!(
        cmax_a <= 1e-9 * cscale_a,
        "A: sweep vs dots F_c mismatch {cmax_a:.3e}"
    );

    let (bxi_s, bfc_s) = jctx.b_covector(&xi, &a, FcRoute::Sweep);
    let (bxi_d, bfc_d) = jctx.b_covector(&xi, &a, FcRoute::Dots);
    let mut bdiff = bxi_s.clone();
    bdiff.axpy(-1.0, &bxi_d);
    assert!(
        bdiff.norm() <= 1e-8 * bxi_s.norm().max(1.0),
        "B: sweep vs dots ξ-leg mismatch {:.3e}",
        bdiff.norm()
    );
    let cscale_b = bfc_s
        .iter()
        .cloned()
        .fold(0.0f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let cmax_b = bfc_s
        .iter()
        .zip(&bfc_d)
        .fold(0.0f64, |acc, (&s, &dd)| acc.max((s - dd).abs()));
    assert!(
        cmax_b <= 1e-8 * cscale_b,
        "B: sweep vs dots F_c mismatch {cmax_b:.3e}"
    );

    let zeros = vec![0.0f64; n_orb];
    let (fxi0, fc0) = jctx.a_covector(&xi, &zeros, FcRoute::Sweep);
    let w1 = xi.to_vector();
    let hw = ham.apply(&w1);
    let mut want_fxi = TtTangent::project(bl, br, &hw);
    want_fxi.axpy(-lambda, &TtTangent::project(bl, br, &w1));
    let mut fdiff = fxi0.clone();
    fdiff.axpy(-1.0, &want_fxi);
    assert!(
        fdiff.norm() <= 1e-9 * want_fxi.norm().max(1.0),
        "zero-κ F_ξ vs independent reference: {:.3e}",
        fdiff.norm()
    );

    let gh = transition_1rdm_h(&x, &ham, &w1);
    let g0 = transition_1rdm(&x, &w1);
    let want_fc: Vec<f64> = jctx
        .pairs()
        .iter()
        .map(|&(p, q)| {
            let gpq = gh.get(&[p, q]) - lambda * g0.get(&[p, q]);
            let gqp = gh.get(&[q, p]) - lambda * g0.get(&[q, p]);
            gpq - gqp
        })
        .collect();
    let cscale0 = want_fc
        .iter()
        .cloned()
        .fold(0.0f64, |acc, v| acc.max(v.abs()))
        .max(1.0);
    let cmax0 = fc0
        .iter()
        .zip(&want_fc)
        .fold(0.0f64, |acc, (&got, &want)| acc.max((got - want).abs()));
    assert!(
        cmax0 <= 1e-9 * cscale0,
        "zero-κ F_c vs independent reference: {cmax0:.3e}"
    );
}

#[test]
fn tjh_neff_zero_equals_tt() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let wctx = weingarten_operator(&ham, &x);
    let tb = tangent_basis(&base_l, &base_r, true);
    assert!(tb.len() >= 2, "need a nontrivial tangent space");

    let empty_pairs = PairMask::with_predicate(d, |_, _| false);
    let opts = JointOpts {
        pairs: Some(empty_pairs),
        ..JointOpts::default()
    };
    let jctx = joint_context(&ham, &x, &opts).expect("fixture must be TT-critical");
    assert_eq!(
        jctx.pairs().len(),
        0,
        "empty PairMask must produce zero pairs"
    );
    assert_eq!(jctx.n_eff(), 0);

    let spec = jctx.spec();
    for xi in tb.iter().take(4) {
        let v = JointVector::from_tangent(spec, xi.clone());
        let av = jctx.a_apply(&v);
        assert!(av.c.is_empty(), "c must be empty at n_eff=0");
        let want_a = wctx.a_apply(xi);
        let mut diff = av.xi.clone();
        diff.axpy(-1.0, &want_a);
        assert!(
            diff.norm() <= 1e-13 * want_a.norm().max(1.0),
            "a_apply forwarding mismatch: {:.3e}",
            diff.norm()
        );

        let bv = jctx.b_apply(&v);
        let want_b = wctx.b_apply(xi);
        let mut bdiff = bv.xi.clone();
        bdiff.axpy(-1.0, &want_b);
        assert!(
            bdiff.norm() <= 1e-13 * want_b.norm().max(1.0),
            "b_apply forwarding mismatch: {:.3e}",
            bdiff.norm()
        );
    }

    let (basis, report) = jctx.joint_basis();
    assert_eq!(basis.len(), tb.len());
    assert_eq!(report.n_orb_kept, 0);
    assert_eq!(report.dropped, 0);

    let (a_tt, b_tt) = hessian_grams(&wctx, &tb);
    let bmax = b_tt.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax > 1e-8,
        "fixture must be genuinely truncated (B_tt vacuous otherwise)"
    );
    let SymplecticSpectrum::Stable { omega: omega_tt } =
        symplectic_from_grams(&a_tt, &b_tt, tb.len(), 1e-10)
    else {
        panic!("expected a stable TT spectrum");
    };
    let SymplecticSpectrum::Stable { omega: omega_joint } =
        jctx.joint_symplectic_eigenvalues(&basis)
    else {
        panic!("expected a stable joint spectrum");
    };
    assert_eq!(omega_joint.len(), omega_tt.len());
    for (k, (&wj, &wt)) in omega_joint.iter().zip(&omega_tt).enumerate() {
        assert!(
            (wj - wt).abs() <= 1e-9 * wt.max(1.0),
            "ω[{k}]: joint {wj} vs TT-only {wt}"
        );
    }
}

#[test]
fn tjh_c6_hess_hpp_hmm_and_driver() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let jctx = joint_context(&ham, &x, &JointOpts::default()).expect("fixture must be TT-critical");
    assert!(
        jctx.n_eff() >= 1,
        "fixture must carry genuine orbital redundancy for this arm (n_eff={})",
        jctx.n_eff()
    );
    let tb = tangent_basis(&base_l, &base_r, true);
    assert!(tb.len() >= 2, "need a nontrivial tangent space");
    let spec = jctx.spec();
    let n_eff = jctx.n_eff();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC6C7);
    let mut vecs: Vec<JointVector> = Vec::new();
    for i in 0..2 {
        let mut vv = JointVector::from_tangent(spec, tb[i % tb.len()].clone());
        vv.c = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
        vecs.push(vv);
    }
    vecs.push(JointVector::from_tangent(spec, tb[0].clone()));
    let c_only: Vec<f64> = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
    vecs.push(JointVector::from_coeffs(spec, &jctx, c_only));

    let mut scale_b = 0.0f64;
    for v in &vecs {
        let av = jctx.a_apply(v);
        let bv = jctx.b_apply(v);
        scale_b = scale_b.max(bv.norm());

        let mut want_pp = av.clone();
        want_pp.axpy(1.0, &bv);
        let mut want_mm = av.clone();
        want_mm.axpy(-1.0, &bv);

        for (name, got, want) in [
            ("hess_apply", jctx.hess_apply(v), want_pp.clone()),
            ("hpp_apply", jctx.hpp_apply(v), want_pp.clone()),
            ("hmm_apply", jctx.hmm_apply(v), want_mm.clone()),
        ] {
            let mut xi_diff = got.xi.clone();
            xi_diff.axpy(-1.0, &want.xi);
            let scale = want.xi.norm().max(1.0);
            assert!(
                xi_diff.norm() <= 1e-10 * scale,
                "{name}: ξ leg vs A±B mismatch {:.3e}",
                xi_diff.norm()
            );
            let worst_c = got
                .c
                .iter()
                .zip(&want.c)
                .fold(0.0f64, |acc, (&g, &w)| acc.max((g - w).abs()));
            let cscale = want.c.iter().fold(1.0f64, |acc, &c| acc.max(c.abs()));
            assert!(
                worst_c <= 1e-10 * cscale,
                "{name}: c leg vs A±B mismatch {worst_c:.3e}"
            );
        }
    }
    assert!(scale_b > 1e-8, "B vacuous in this arm: scale {scale_b:.3e}");

    let (basis, report) = jctx.joint_basis();
    assert_eq!(basis.len(), tb.len() + n_eff);
    assert_eq!(report.dropped, 0, "fixture should keep every orbital seed");
    let n = basis.len();
    let (a, b) = jctx.joint_hessian_grams(&basis);
    let scale_a = a.iter().fold(1.0f64, |acc, &v| acc.max(v.abs()));
    let scale_b = b.iter().fold(1.0f64, |acc, &v| acc.max(v.abs()));

    let mut worst_sym_a = 0.0f64;
    let mut worst_sym_b = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            worst_sym_a = worst_sym_a.max((a[i * n + j] - a[j * n + i]).abs());
            worst_sym_b = worst_sym_b.max((b[i * n + j] - b[j * n + i]).abs());
        }
    }
    assert!(
        worst_sym_a <= 1e-8 * scale_a,
        "joint A-Gram not symmetric: {worst_sym_a:.3e} (scale {scale_a:.3e})"
    );
    assert!(
        worst_sym_b <= 1e-10 * scale_b,
        "joint B-Gram not symmetric: {worst_sym_b:.3e} (scale {scale_b:.3e})"
    );

    let mut worst_a = 0.0f64;
    let mut worst_b = 0.0f64;
    for &(i, j) in &[(0usize, 0usize), (0, n - 1), (n - 1, 0), (n / 2, n - 1)] {
        let want_a = basis[i].dot(&jctx.a_apply(&basis[j]));
        let want_b = basis[i].dot(&jctx.b_apply(&basis[j]));
        worst_a = worst_a.max((a[i * n + j] - want_a).abs());
        worst_b = worst_b.max((b[i * n + j] - want_b).abs());
    }
    assert!(
        worst_a <= 1e-8 * scale_a,
        "ab_raised A leg vs independent a_apply: {worst_a:.3e} (scale {scale_a:.3e})"
    );
    assert!(
        worst_b <= 1e-8 * scale_b,
        "ab_raised B leg vs independent b_apply: {worst_b:.3e} (scale {scale_b:.3e})"
    );

    match jctx.joint_symplectic_eigenvalues(&basis) {
        SymplecticSpectrum::Stable { omega } => {
            assert_eq!(omega.len(), n, "one ω per basis vector at n_eff>0");
            assert!(
                omega.iter().all(|&w| w > 0.0),
                "stable spectrum must be strictly positive: {omega:?}"
            );
        }
        SymplecticSpectrum::Resonant { signed } => {
            assert_eq!(
                signed.omega2.len(),
                n,
                "the signed reduced form returns one ω² per basis vector: {n} basis vectors"
            );
            assert!(
                signed.n_neg > 0,
                "Resonant must carry at least one negative ω²: {:?}",
                signed.omega2.first()
            );
        }
        SymplecticSpectrum::Unstable { generator_eigs, .. } => {
            assert_eq!(
                generator_eigs.len(),
                2 * n,
                "generator is 2n×2n: {} basis vectors",
                n
            );
        }
    }
}

#[test]
fn tjh_coeff_contract() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xC0EF, 1, 2);
    let jctx = joint_context(&ham, &x, &JointOpts::default()).expect("fixture must be TT-critical");
    assert!(jctx.n_eff() >= 1, "need orbital coordinates for this arm");
    let tb = tangent_basis(&base_l, &base_r, true);
    assert!(!tb.is_empty());
    let spec = jctx.spec();
    let n_eff = jctx.n_eff();
    let nt = tb[0].num_coeffs();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0EF1);
    let mut v = JointVector::from_tangent(spec, tb[0].clone());
    v.c = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();

    assert_eq!(v.num_coeffs(), nt + n_eff, "num_coeffs = ξ-coeffs + n_eff");
    for i in 0..nt {
        assert_eq!(v.coeff(i), v.xi.coeff(i), "coeff({i}) must read the ξ leg");
    }
    for j in 0..n_eff {
        assert_eq!(
            v.coeff(nt + j),
            v.c[j],
            "coeff({}) must read c[{j}]",
            nt + j
        );
    }

    let before: Vec<f64> = (0..v.num_coeffs()).map(|i| v.coeff(i)).collect();
    let touch = nt + n_eff / 2;
    v.set_coeff(touch, 12.5);
    assert_eq!(v.coeff(touch), 12.5);
    for (i, &b) in before.iter().enumerate() {
        if i != touch {
            assert_eq!(v.coeff(i), b, "set_coeff({touch}) disturbed slot {i}");
        }
    }
    v.set_coeff(0, before[0]);
    v.set_coeff(touch, before[touch]);

    let mut clone = v.clone();
    clone.set_coeff(0, 999.0);
    assert_eq!(
        v.coeff(0),
        before[0],
        "mutating a clone touched the original"
    );
    assert_eq!(clone.coeff(0), 999.0);

    let mut via_foreach = v.clone();
    via_foreach.for_each_coeff_mut(|x| *x += 1.0);
    for i in 0..v.num_coeffs() {
        assert_eq!(
            via_foreach.coeff(i),
            v.coeff(i) + 1.0,
            "for_each_coeff_mut missed or double-hit slot {i}"
        );
    }

    let other = {
        let mut o = JointVector::from_tangent(spec, tb[0].clone());
        o.c = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
        o
    };
    let mut via_zip = v.clone();
    via_zip.zip_coeffs_mut(&other, |s, o| *s += o);
    for i in 0..v.num_coeffs() {
        assert_eq!(
            via_zip.coeff(i),
            v.coeff(i) + other.coeff(i),
            "zip_coeffs_mut mismatch at slot {i}"
        );
    }
}

#[test]
fn tjh_c1_selfadjoint_and_dense() {
    let (space, ham, x, t, v) = lih_cation_rank1_fixture();
    let jctx =
        joint_context(&ham, &x, &JointOpts::default()).expect("LiH+ fixture must be TT-critical");
    assert!(
        jctx.n_eff() >= 2,
        "need orbital redundancy for the pure-orbital arm"
    );
    let (bl, br) = jctx.bases();
    let tb = tangent_basis(bl, br, true);
    assert!(tb.len() >= 2);
    let spec = jctx.spec();
    let n_eff = jctx.n_eff();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC100);
    let mut vecs: Vec<JointVector> = Vec::new();
    for i in 0..2 {
        let mut vv = JointVector::from_tangent(spec, tb[i % tb.len()].clone());
        vv.c = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
        vecs.push(vv);
    }
    vecs.push(JointVector::from_tangent(spec, tb[0].clone()));
    let c_only: Vec<f64> = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
    vecs.push(JointVector::from_coeffs(spec, &jctx, c_only));

    assert_c1_selfadjoint(&jctx, &vecs);

    let dets = dets_for(space.d, space.nup, space.ndn);
    let hd = dense_hamiltonian_matrix(space.d, &t, &v, &dets);
    let lambda = jctx.lambda();
    let c = dense_state_vector(&dets, &x);
    let wctx_ref = jctx.weingarten();
    let z = normal_residual(&ham, bl, br, &x, lambda);
    let zd = dense_state_vector(&dets, &z);
    let pairs = jctx.pairs().to_vec();
    let gd: Vec<Vec<Vec<f64>>> = pairs
        .iter()
        .map(|&(p, q)| dense_gpq(space.d, p, q, &dets))
        .collect();

    let amb = |jv: &JointVector| -> Vec<f64> {
        let mut out = dense_state_vector(&dets, &jv.xi.to_vector());
        let a = jctx.v_times(&jv.c);
        for (idx, &ai) in a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            let gc = mat_vec(&gd[idx], &c);
            for (k, val) in gc.iter().enumerate() {
                out[k] -= ai * val;
            }
        }
        out
    };
    let kappa_dense = |jv: &JointVector| -> Vec<Vec<f64>> {
        let a = jctx.v_times(&jv.c);
        let n = dets.len();
        let mut k = vec![vec![0.0f64; n]; n];
        for (idx, &ai) in a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            for r in 0..n {
                for cc in 0..n {
                    k[r][cc] += ai * gd[idx][r][cc];
                }
            }
        }
        k
    };
    let zeta_dense = |jv: &JointVector| dense_state_vector(&dets, &jv.xi.to_vector());

    let bden = |vv: &JointVector, ww: &JointVector| -> f64 {
        let term1 = ww.xi.dot(&wctx_ref.b_apply(&vv.xi));
        let kv = kappa_dense(vv);
        let kw = kappa_dense(ww);
        let zetav = zeta_dense(vv);
        let zetaw = zeta_dense(ww);
        let term2 = dot_vec(&zd, &mat_vec(&kv, &zetaw));
        let term3 = dot_vec(&zd, &mat_vec(&kw, &zetav));
        let kvkw_c = mat_vec(&kv, &mat_vec(&kw, &c));
        let kwkv_c = mat_vec(&kw, &mat_vec(&kv, &c));
        let sum: Vec<f64> = kvkw_c
            .iter()
            .zip(&kwkv_c)
            .map(|(&x1, &x2)| x1 + x2)
            .collect();
        let term4 = 0.5 * dot_vec(&zd, &sum);
        term1 - term2 - term3 + term4
    };

    let mut worst_metric = 0.0f64;
    let mut worst_a_dense = 0.0f64;
    let mut worst_b_dense = 0.0f64;
    for i in 0..vecs.len() {
        for j in 0..vecs.len() {
            let amb_i = amb(&vecs[i]);
            let amb_j = amb(&vecs[j]);
            let metric_want = dot_vec(&amb_i, &amb_j);
            let metric_got = vecs[i].dot(&vecs[j]);
            worst_metric =
                worst_metric.max((metric_got - metric_want).abs() / metric_want.abs().max(1.0));

            let a_want = dense_a_entry(&amb_i, &amb_j, &hd, lambda);
            let a_got = vecs[i].dot(&jctx.a_apply(&vecs[j]));
            worst_a_dense = worst_a_dense.max((a_got - a_want).abs() / a_want.abs().max(1.0));

            let b_want = bden(&vecs[j], &vecs[i]);
            let b_got = vecs[i].dot(&jctx.b_apply(&vecs[j]));
            worst_b_dense = worst_b_dense.max((b_got - b_want).abs() / b_want.abs().max(1.0));
        }
    }
    assert!(
        worst_metric <= 1e-9,
        "metric vs dense ⟨amb(v),amb(w)⟩: rel {worst_metric:.3e}"
    );
    assert!(
        worst_a_dense <= 1e-8,
        "A apply vs dense ⟨amb(w),(Hd−λ)amb(v)⟩: rel {worst_a_dense:.3e}"
    );
    assert!(
        worst_b_dense <= 1e-8,
        "B apply vs dense reference: rel {worst_b_dense:.3e}"
    );
}

fn joint_energy_dense_swapped(
    hd: &[Vec<f64>],
    c: &[f64],
    kd: &[Vec<f64>],
    zeta: &[f64],
    s: f64,
) -> f64 {
    let expo = mat_expm(&mat_scale(kd, -s));
    let ekc = mat_vec(&expo, c);
    let psi: Vec<f64> = ekc.iter().zip(zeta).map(|(&a, &b)| a + s * b).collect();
    let hpsi = mat_vec(hd, &psi);
    dot_vec(&psi, &hpsi) / dot_vec(&psi, &psi)
}

fn q2_dense_swapped(
    hd: &[Vec<f64>],
    c: &[f64],
    kd: &[Vec<f64>],
    zeta: &[f64],
    e0: f64,
    h: f64,
) -> f64 {
    let ep2 = joint_energy_dense_swapped(hd, c, kd, zeta, 2.0 * h);
    let ep1 = joint_energy_dense_swapped(hd, c, kd, zeta, h);
    let em1 = joint_energy_dense_swapped(hd, c, kd, zeta, -h);
    let em2 = joint_energy_dense_swapped(hd, c, kd, zeta, -2.0 * h);
    (-ep2 + 16.0 * ep1 - 30.0 * e0 + 16.0 * em1 - em2) / (24.0 * h * h)
}

#[allow(clippy::too_many_arguments)]
fn q2_polarized_swapped(
    hd: &[Vec<f64>],
    c: &[f64],
    kd_i: &[Vec<f64>],
    zeta_i: &[f64],
    kd_j: &[Vec<f64>],
    zeta_j: &[f64],
    e0: f64,
    h: f64,
) -> f64 {
    let kd_sum = mat_add(kd_i, kd_j);
    let zeta_sum: Vec<f64> = zeta_i.iter().zip(zeta_j).map(|(&a, &b)| a + b).collect();
    let q_sum = q2_dense_swapped(hd, c, &kd_sum, &zeta_sum, e0, h);
    let q_i = q2_dense_swapped(hd, c, kd_i, zeta_i, e0, h);
    let q_j = q2_dense_swapped(hd, c, kd_j, zeta_j, e0, h);
    0.5 * (q_sum - q_i - q_j)
}

#[test]
fn tjh_b2_piece_pins() {
    let (space, ham, x, t, v) = lih_cation_rank1_fixture();
    let d = space.d;
    let jctx = joint_context(&ham, &x, &JointOpts::default())
        .expect("LiH+ rank-1 fixture must be TT-critical");
    let (bl, br) = jctx.bases();
    let lambda = jctx.lambda();

    let z = normal_residual(&ham, bl, br, &x, lambda);
    assert!(
        z.norm_lr(false) >= 1e-3,
        "‖z‖ vacuous: {:.3e}",
        z.norm_lr(false)
    );
    let og = qntt::rotation::orbital_gradient(&ham, &x);
    let mut og_norm2 = 0.0f64;
    for &(p, q) in jctx.pairs() {
        let gv = og.get(&[p, q]);
        og_norm2 += gv * gv;
    }
    assert!(
        og_norm2.sqrt() > 1e-6,
        "orbital gradient vacuous (point is orbital-critical)"
    );
    assert!(
        jctx.n_eff() >= 1,
        "need orbital redundancy: n_eff={}",
        jctx.n_eff()
    );

    let dets = dets_for(d, space.nup, space.ndn);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let c = dense_state_vector(&dets, &x);
    let zd = dense_state_vector(&dets, &z);
    let pairs = jctx.pairs().to_vec();
    let n_orb = pairs.len();
    let gd: Vec<Vec<Vec<f64>>> = pairs
        .iter()
        .map(|&(p, q)| dense_gpq(d, p, q, &dets))
        .collect();
    let dense_u = |idx: usize| -> Vec<f64> { mat_vec(&gd[idx], &c).iter().map(|v| -v).collect() };

    let (bl_owned, br_owned) = (bl.clone(), br.clone());
    let tb = tangent_basis(&bl_owned, &br_owned, true);
    assert!(tb.len() >= 3, "need at least 3 tangent directions");
    let zetad: Vec<Vec<f64>> = tb
        .iter()
        .map(|xi| dense_state_vector(&dets, &xi.to_vector()))
        .collect();

    let unit = |n: usize, k: usize| -> Vec<f64> {
        let mut v = vec![0.0f64; n];
        v[k] = 1.0;
        v
    };
    let zero_tan = &tb[0].scale(0.0);

    let mut worst_p3 = 0.0f64;
    for i in 0..3 {
        for j in 0..3 {
            let want = dense_a_entry(&zetad[i], &zetad[j], &hd, lambda);
            let (fxi, _fc) = jctx.a_covector(&tb[j], &vec![0.0; n_orb], FcRoute::Sweep);
            let got = tb[i].dot(&fxi);
            worst_p3 = worst_p3.max((got - want).abs() / want.abs().max(1.0));
        }
    }
    assert!(
        worst_p3 <= 1e-8,
        "P3 (A tt) dense vs a_covector: rel {worst_p3:.3e}"
    );

    let sub: Vec<usize> = vec![1, 2, 3, 10, 11];
    let mut worst_p4 = 0.0f64;
    for &pa in &sub {
        let (_fxi, fc) = jctx.a_covector(zero_tan, &unit(n_orb, pa), FcRoute::Sweep);
        for &pb in &sub {
            let want = dense_a_entry(&dense_u(pb), &dense_u(pa), &hd, lambda);
            let got = fc[pb];
            worst_p4 = worst_p4.max((got - want).abs() / want.abs().max(1.0));
        }
    }
    assert!(
        worst_p4 <= 1e-8,
        "P4 (A orb) dense vs a_covector: rel {worst_p4:.3e}"
    );

    let mut worst_p5 = 0.0f64;
    for i in 0..3 {
        let (_fxi, fc) = jctx.a_covector(&tb[i], &vec![0.0; n_orb], FcRoute::Sweep);
        for &pb in &sub {
            let want = dense_a_entry(&dense_u(pb), &zetad[i], &hd, lambda);
            let got = fc[pb];
            worst_p5 = worst_p5.max((got - want).abs() / want.abs().max(1.0));
        }
    }
    assert!(
        worst_p5 <= 1e-8,
        "P5 (A cross) dense vs a_covector: rel {worst_p5:.3e}"
    );

    for &pa in sub.iter().take(2) {
        let (p, q) = pairs[pa];
        let ua_tt = &qntt::rotation::g_pq(&x, p, q) * (-1.0f64);
        let ua_dense_want = dense_u(pa);
        let ua_dense_got = dense_state_vector(&dets, &ua_tt);
        let diff: f64 = ua_dense_got
            .iter()
            .zip(&ua_dense_want)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(diff <= 1e-8, "dense u_a vs TT u_a: {diff:.3e}");
        let tau_a = TtTangent::project(&bl_owned, &br_owned, &ua_tt);
        for i in 0..3 {
            let want = dot_vec(&zetad[i], &ua_dense_want);
            let got = tb[i].dot(&tau_a);
            assert!(
                (got - want).abs() <= 1e-8 * want.abs().max(1.0),
                "P6 P-adjoint identity ⟨ξ,τ_a⟩ vs ⟨Z,U_a⟩: got {got} want {want}"
            );
        }
    }

    let mut worst_p7 = 0.0f64;
    let mut max_b_orb = 0.0f64;
    let mut b_orb_sub = vec![vec![0.0f64; sub.len()]; sub.len()];
    for (ia, &pa) in sub.iter().enumerate() {
        let (_fxi, fc) = jctx.b_covector(zero_tan, &unit(n_orb, pa), FcRoute::Sweep);
        for (ib, &pb) in sub.iter().enumerate() {
            let want = dense_b_orb(&zd, &gd[pa], &gd[pb], &c);
            let got = fc[pb];
            worst_p7 = worst_p7.max((got - want).abs() / want.abs().max(1.0));
            max_b_orb = max_b_orb.max(want.abs());
            b_orb_sub[ia][ib] = got;
        }
    }
    assert!(
        max_b_orb > 1e-3,
        "B_orb vacuous on the sampled subset: {max_b_orb:.3e}"
    );
    assert!(
        worst_p7 <= 1e-6,
        "P7 (B orb×orb) dense vs b_covector: rel {worst_p7:.3e}"
    );

    let mut worst_p8 = 0.0f64;
    let mut max_b_cross = 0.0f64;
    let mut b_cross_sub = vec![vec![0.0f64; sub.len()]; 3];
    for i in 0..3 {
        let (_fxi, fc) = jctx.b_covector(&tb[i], &vec![0.0; n_orb], FcRoute::Sweep);
        for (ib, &pb) in sub.iter().enumerate() {
            let want = dense_b_cross(&zd, &gd[pb], &zetad[i]);
            let got_c_side = fc[pb];
            worst_p8 = worst_p8.max((got_c_side - want).abs() / want.abs().max(1.0));
            max_b_cross = max_b_cross.max(want.abs());
            b_cross_sub[i][ib] = got_c_side;

            let (fxi_b, _fc_b) = jctx.b_covector(zero_tan, &unit(n_orb, pb), FcRoute::Sweep);
            let got_xi_side = tb[i].dot(&fxi_b);
            worst_p8 = worst_p8.max((got_xi_side - want).abs() / want.abs().max(1.0));
        }
    }
    assert!(
        max_b_cross > 1e-3,
        "B_cross vacuous on the sampled subset: {max_b_cross:.3e}"
    );
    assert!(
        worst_p8 <= 1e-6,
        "P8 (B cross, both TT forms) dense vs b_covector: rel {worst_p8:.3e}"
    );

    let n_t = 3usize;
    let n_o = sub.len();
    let m = n_t + n_o;
    let zero90 = vec![0.0f64; dets.len()];
    let zero_mat = vec![vec![0.0f64; dets.len()]; dets.len()];

    let v_cols: Vec<Vec<f64>> = (0..n_t)
        .map(|i| zetad[i].clone())
        .chain(sub.iter().map(|&pa| dense_u(pa)))
        .collect();
    let k_mats: Vec<Vec<Vec<f64>>> = (0..n_t)
        .map(|_| zero_mat.clone())
        .chain(sub.iter().map(|&pa| gd[pa].clone()))
        .collect();
    let zeta_cols: Vec<Vec<f64>> = (0..n_t)
        .map(|i| zetad[i].clone())
        .chain(sub.iter().map(|_| zero90.clone()))
        .collect();

    let mut aj = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        for j in 0..m {
            aj[i][j] = dense_a_entry(&v_cols[i], &v_cols[j], &hd, lambda);
        }
    }

    let mut bj = vec![vec![0.0f64; m]; m];
    for i in 0..n_t {
        for (ib, _) in sub.iter().enumerate() {
            bj[i][n_t + ib] = b_cross_sub[i][ib];
            bj[n_t + ib][i] = b_cross_sub[i][ib];
        }
    }
    for ia in 0..n_o {
        for ib in 0..n_o {
            bj[n_t + ia][n_t + ib] = b_orb_sub[ia][ib];
        }
    }

    let h = 2e-3;
    let mut fdj = vec![vec![0.0f64; m]; m];
    for i in 0..m {
        fdj[i][i] = q2_dense(&hd, &c, &k_mats[i], &zeta_cols[i], lambda, h);
    }
    for i in 0..m {
        for j in (i + 1)..m {
            let off = q2_polarized(
                &hd,
                &c,
                &k_mats[i],
                &zeta_cols[i],
                &k_mats[j],
                &zeta_cols[j],
                lambda,
                h,
            );
            fdj[i][j] = off;
            fdj[j][i] = off;
        }
    }

    let ob = n_t..m;
    let dev_orb = |mat: &[Vec<f64>]| -> f64 {
        let mut w = 0.0f64;
        for i in ob.clone() {
            for j in ob.clone() {
                w = w.max((fdj[i][j] - mat[i][j]).abs());
            }
        }
        w
    };
    let dev_cross = |mat: &[Vec<f64>]| -> f64 {
        let mut w = 0.0f64;
        for i in 0..n_t {
            for j in ob.clone() {
                w = w.max((fdj[i][j] - mat[i][j]).abs());
            }
        }
        w
    };
    let combine = |scale_b: f64| -> Vec<Vec<f64>> {
        (0..m)
            .map(|i| (0..m).map(|j| aj[i][j] + scale_b * bj[i][j]).collect())
            .collect()
    };

    let max_aj = aj.iter().flatten().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    let tol_fd = 1e-7 * max_aj.max(1.0);
    let mut max_bj_orb = 0.0f64;
    for i in ob.clone() {
        for j in ob.clone() {
            max_bj_orb = max_bj_orb.max(bj[i][j].abs());
        }
    }
    let mut max_bj_cross = 0.0f64;
    for row in bj.iter().take(n_t) {
        for j in ob.clone() {
            max_bj_cross = max_bj_cross.max(row[j].abs());
        }
    }
    let teeth_orb = 0.5 * max_bj_orb;
    let teeth_cross = 0.5 * max_bj_cross;
    assert!(
        teeth_orb > 1e3 * tol_fd,
        "B not large enough to give this gate teeth on the orb block: teeth={teeth_orb:.3e} tol_fd={tol_fd:.3e}"
    );
    assert!(
        teeth_cross > 1e3 * tol_fd,
        "B not large enough to give this gate teeth on the cross block: teeth={teeth_cross:.3e} tol_fd={tol_fd:.3e}"
    );

    let ab = combine(1.0);
    assert!(
        dev_orb(&ab) <= tol_fd,
        "P10 dev_orb(A+B) = {:.3e} > tol {tol_fd:.3e}",
        dev_orb(&ab)
    );
    assert!(
        dev_cross(&ab) <= tol_fd,
        "P10 dev_cross(A+B) = {:.3e} > tol {tol_fd:.3e}",
        dev_cross(&ab)
    );

    let a_only = combine(0.0);
    assert!(
        dev_orb(&a_only) > teeth_orb,
        "negative arm (B dropped) did not fail on orb block"
    );
    assert!(
        dev_cross(&a_only) > teeth_cross,
        "negative arm (B dropped) did not fail on cross block"
    );

    let a_minus_b = combine(-1.0);
    assert!(
        dev_orb(&a_minus_b) > teeth_orb,
        "negative arm (B sign-flipped) did not fail on orb block"
    );
    assert!(
        dev_cross(&a_minus_b) > teeth_cross,
        "negative arm (B sign-flipped) did not fail on cross block"
    );

    let a_plus_2b = combine(2.0);
    assert!(
        dev_orb(&a_plus_2b) > teeth_orb,
        "negative arm (B doubled) did not fail on orb block"
    );
    assert!(
        dev_cross(&a_plus_2b) > teeth_cross,
        "negative arm (B doubled) did not fail on cross block"
    );

    let mut worst_swapped_vs_a = 0.0f64;
    let mut worst_swapped_vs_true = 0.0f64;
    for i in 0..n_t {
        for (jo, j) in ob.clone().enumerate() {
            let fd_swapped = q2_polarized_swapped(
                &hd,
                &c,
                &k_mats[i],
                &zeta_cols[i],
                &k_mats[j],
                &zeta_cols[j],
                lambda,
                h,
            );
            worst_swapped_vs_a = worst_swapped_vs_a.max((fd_swapped - aj[i][j]).abs());
            worst_swapped_vs_true = worst_swapped_vs_true.max((fd_swapped - fdj[i][j]).abs());
            let _ = jo;
        }
    }
    assert!(
        worst_swapped_vs_a <= tol_fd,
        "swapped-ordering curve should reproduce A alone on the cross block: {worst_swapped_vs_a:.3e}"
    );
    assert!(
        worst_swapped_vs_true > teeth_cross,
        "swapped-ordering curve should MISS the true FD by ~‖B‖: {worst_swapped_vs_true:.3e}"
    );
}
