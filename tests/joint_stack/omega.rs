// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::sync::OnceLock;

use qntt::integrals::PropertyOperator;
use qntt::solvers::joint::{
    DynamicResponseOpts, DynamicSolver, JointOpts, StaticResponseOpts, TransitionMomentOpts,
    joint_context, linear_response, oscillator_strengths, polarizability_static,
    transition_moments,
};
#[allow(unused_imports)]
use qntt::solvers::minres::{MinresFlag, minres};
use qntt::{CoeffTensor, QnSpace, TtVector};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::optimize::d2_fixture;
use crate::util::*;

struct OmegaSosFixture {
    space: QnSpace,
    ham: qntt::SparseHamiltonian<f64>,
    x: TtVector<f64>,
    evals: Vec<f64>,
    evecs: Vec<Vec<f64>>,
    m1: CoeffTensor<f64>,
    m2: CoeffTensor<f64>,
}

fn build_sos_fixture() -> OmegaSosFixture {
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xABCD, 1, 1);
    let dets = dets_for(d, 1, 1);
    let (t, v) = generic_truncated_tv(d);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let (evals, evecs) = dense_eigen(&hd);

    let mut m1 = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        m1.set(&[i, i], (i as f64) - 1.0);
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9002);
    let mut m2 = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in i..d {
            use rand::RngExt;
            let val = rng.random_range(-1.0f64..1.0);
            m2.set(&[i, j], val);
            m2.set(&[j, i], val);
        }
    }
    OmegaSosFixture {
        space,
        ham,
        x,
        evals,
        evecs,
        m1,
        m2,
    }
}

fn sos_fixture() -> &'static OmegaSosFixture {
    static CELL: OnceLock<OmegaSosFixture> = OnceLock::new();
    CELL.get_or_init(build_sos_fixture)
}

fn dense_alpha_omega(evals: &[f64], evecs: &[Vec<f64>], mu: &[Vec<f64>], omega: f64) -> f64 {
    let v0 = &evecs[0];
    let mu_v0 = mat_vec(mu, v0);
    let mut acc = 0.0;
    for n in 1..evals.len() {
        let tn = dot_vec(&evecs[n], &mu_v0);
        let wn = evals[n] - evals[0];
        acc += tn * tn * wn / (wn * wn - omega * omega);
    }
    2.0 * acc
}

#[test]
fn tjw_moments_dense_pin() {
    let fx = sos_fixture();
    let jctx = joint_context(&fx.ham, &fx.x, &JointOpts::default())
        .expect("SOS fixture must be TT-critical");
    assert_eq!(
        jctx.n_eff(),
        0,
        "SOS fixture must be full rank (n_eff == 0)"
    );

    let props = [
        PropertyOperator::new("diag", fx.m1.clone()),
        PropertyOperator::new("offdiag", fx.m2.clone()),
    ];
    let (basis, report) = jctx.joint_basis();
    let opts = TransitionMomentOpts::default();
    let moments =
        transition_moments(&jctx, &basis, &props, &opts).expect("full-rank moments must build");

    let n_modes = moments.omega.len();
    assert_eq!(
        n_modes,
        fx.evals.len() - 1,
        "one Williamson mode per excited CI state at full rank"
    );
    for n in 1..n_modes {
        assert!(
            moments.omega[n] >= moments.omega[n - 1] - 1e-12,
            "omega must be ascending"
        );
    }

    let mut clusters: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0usize;
    for n in 1..n_modes {
        if (moments.omega[n] - moments.omega[n - 1]).abs() > 1e-6 * moments.omega[n].max(1.0) {
            clusters.push((lo, n));
            lo = n;
        }
    }
    clusters.push((lo, n_modes));

    let mu_d: Vec<Vec<Vec<f64>>> = props
        .iter()
        .map(|p| dense_kappa_matrix(fx.space.d, &p.m, &dets_for(fx.space.d, 1, 1)))
        .collect();
    let mu_v0: Vec<Vec<f64>> = mu_d.iter().map(|m| mat_vec(m, &fx.evecs[0])).collect();

    for &(clo, chi) in &clusters {
        for (p, mv0) in mu_v0.iter().enumerate() {
            let mut want_sq = 0.0f64;
            let mut got_sq = 0.0f64;
            for n in clo..chi {
                let dense_t = dot_vec(&fx.evecs[n + 1], mv0);
                want_sq += dense_t * dense_t;
                got_sq += moments.moments[p][n] * moments.moments[p][n];
            }
            assert_close(
                got_sq,
                want_sq,
                1e-8 * want_sq.max(1.0),
                &format!("cluster [{clo},{chi}) prop {p}: sum of |T_n|^2"),
            );
        }

        let mut want_cross = 0.0f64;
        let mut got_cross = 0.0f64;
        for n in clo..chi {
            let dt0 = dot_vec(&fx.evecs[n + 1], &mu_v0[0]);
            let dt1 = dot_vec(&fx.evecs[n + 1], &mu_v0[1]);
            want_cross += dt0 * dt1;
            got_cross += moments.moments[0][n] * moments.moments[1][n];
        }
        assert_close(
            got_cross,
            want_cross,
            1e-8 * want_cross.abs().max(1e-6),
            &format!("cluster [{clo},{chi}): cross-property sign product"),
        );
    }

    let max_t = moments
        .moments
        .iter()
        .flatten()
        .fold(0.0f64, |a, &v| a.max(v.abs()));
    assert!(max_t > 1e-3, "anti-vacuity: max|T_n| = {max_t:e}");
    println!(
        "tjw_moments_dense_pin: n_modes={n_modes} clusters={} orth_error={:e} sigma_min={:e}",
        clusters.len(),
        moments.orth_error,
        moments.sigma_min
    );
    assert!(moments.orth_error.is_finite() && moments.orth_error >= 0.0);
    assert!(
        moments.orth_error <= 1e3 * report.orth_error.max(1e-300),
        "moments.orth_error={:e} wildly exceeds the basis's own orth_error={:e}",
        moments.orth_error,
        report.orth_error
    );

    let osc = oscillator_strengths(&moments.omega, &moments.moments);
    assert_eq!(osc.len(), n_modes);
    assert!(
        osc.iter().all(|f| f.is_finite() && *f >= 0.0),
        "oscillator strengths must be finite and non-negative: {osc:?}"
    );
}

#[test]
fn tjw_sos_eq_solve() {
    let fx = sos_fixture();
    let jctx = joint_context(&fx.ham, &fx.x, &JointOpts::default())
        .expect("SOS fixture must be TT-critical");
    let props = [
        PropertyOperator::new("diag", fx.m1.clone()),
        PropertyOperator::new("offdiag", fx.m2.clone()),
    ];
    let (basis, _report) = jctx.joint_basis();
    let moments = transition_moments(&jctx, &basis, &props, &TransitionMomentOpts::default())
        .expect("full-rank moments must build");
    let sopts = StaticResponseOpts {
        tol: 1e-10,
        ..Default::default()
    };
    let resp = polarizability_static(&jctx, &props, &sopts).expect("full-rank solve must succeed");
    let n = props.len();
    for i in 0..n {
        for j in 0..n {
            assert_close(
                moments.alpha_sos[i * n + j],
                resp.alpha[i * n + j],
                1e-8 * resp.alpha[i * n + j].abs().max(1.0),
                &format!("alpha_sos[{i},{j}] vs polarizability_static (full rank)"),
            );
        }
    }

    let d2 = d2_fixture();
    let out = &d2.out;
    let jctx2 = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("D2 fixture must be TT-critical");
    assert!(jctx2.n_eff() > 0, "D2 fixture must be genuinely truncated");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0771);
    let m = random_sym_t(d2.space.d, &mut rng);
    let prop2 = [PropertyOperator::new("d2_sos", m)];
    let (basis2, report2) = jctx2.joint_basis();
    let moments2 = transition_moments(&jctx2, &basis2, &prop2, &TransitionMomentOpts::default())
        .expect("truncated moments must build");
    let resp2 =
        polarizability_static(&jctx2, &prop2, &sopts).expect("truncated solve must succeed");
    println!(
        "tjw_sos_eq_solve truncated: orth_error={:e} completeness_defect={:e} n_eff={}",
        report2.orth_error,
        moments2.completeness_defect[0],
        jctx2.n_eff()
    );
    assert_close(
        moments2.alpha_sos[0],
        resp2.alpha[0],
        1e-6 * resp2.alpha[0].abs().max(1.0),
        "alpha_sos vs polarizability_static (truncated, D2)",
    );
}

#[test]
fn tjw_omega_zero_degenerates() {
    let d2 = d2_fixture();
    let out = &d2.out;
    let jctx = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("D2 fixture must be TT-critical");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0772);
    let m = random_sym_t(d2.space.d, &mut rng);
    let props = [PropertyOperator::new("omega0", m)];

    let sopts = StaticResponseOpts {
        tol: 1e-9,
        ..Default::default()
    };
    let static_resp =
        polarizability_static(&jctx, &props, &sopts).expect("static solve must succeed");

    let dopts = DynamicResponseOpts {
        tol: 1e-9,
        ..Default::default()
    };
    let dyn_resp = linear_response(&jctx, &props, 0.0, &dopts).expect("omega=0 solve must succeed");

    assert_close(
        dyn_resp.alpha[0],
        static_resp.alpha[0],
        1e-10 * static_resp.alpha[0].abs().max(1.0),
        "omega=0 must degenerate EXACTLY to the static path",
    );
    assert!(
        dyn_resp.solves[0].v_weight <= 1e-8,
        "the v-block must stay ~0 at omega=0 (spec 1 §1.6): v_weight={:e}",
        dyn_resp.solves[0].v_weight
    );
    assert!(
        dyn_resp.solves[0].rel_residual <= dopts.tol * 10.0,
        "achieved residual must be reported and small: {:e}",
        dyn_resp.solves[0].rel_residual
    );
}

#[test]
fn tjw_alpha_omega_dense() {
    let fx = sos_fixture();
    let jctx = joint_context(&fx.ham, &fx.x, &JointOpts::default())
        .expect("SOS fixture must be TT-critical");
    let props = [PropertyOperator::new("offdiag", fx.m2.clone())];
    let mu_d = dense_kappa_matrix(fx.space.d, &fx.m2, &dets_for(fx.space.d, 1, 1));

    let omega1 = fx.evals[1] - fx.evals[0];
    let omega2 = fx.evals[2] - fx.evals[0];
    assert!(
        omega2 > omega1 + 1e-6,
        "fixture must have non-degenerate low excitations: omega1={omega1:e} omega2={omega2:e}"
    );
    let spectrum: Vec<f64> = fx.evals[1..].iter().map(|&e| e - fx.evals[0]).collect();

    let w1 = 0.5 * omega1;
    let opts1 = DynamicResponseOpts {
        tol: 1e-9,
        solver: DynamicSolver::Pcg,
        ..Default::default()
    };
    let r1 = linear_response(&jctx, &props, w1, &opts1).expect("below-pole solve must converge");
    let want1 = dense_alpha_omega(&fx.evals, &fx.evecs, &mu_d, w1);
    assert_close(
        r1.alpha[0],
        want1,
        1e-6 * want1.abs().max(1.0),
        "alpha(0.5*omega1) vs dense closed form (PCG arm)",
    );
    assert_eq!(
        r1.solves[0].solver_used,
        DynamicSolver::Pcg,
        "arm 1 requested Pcg explicitly and must report it"
    );

    let w2 = 0.5 * (omega1 + omega2);
    let opts2 = DynamicResponseOpts {
        tol: 1e-9,
        solver: DynamicSolver::Minres,
        ..Default::default()
    };
    let r2 = linear_response(&jctx, &props, w2, &opts2)
        .expect("between-poles solve must converge via MINRES");
    let want2 = dense_alpha_omega(&fx.evals, &fx.evecs, &mu_d, w2);
    assert_close(
        r2.alpha[0],
        want2,
        1e-4 * want2.abs().max(1.0),
        "alpha(midway(omega1,omega2)) vs dense closed form (MINRES arm)",
    );
    assert_eq!(
        r2.solves[0].solver_used,
        DynamicSolver::Minres,
        "arm 2 requested Minres explicitly and must report it"
    );

    let w3 = 0.98 * omega1;
    let opts3 = DynamicResponseOpts {
        tol: 1e-9,
        solver: DynamicSolver::Auto,
        spectrum: Some(spectrum.clone()),
        ..Default::default()
    };
    let r3 = linear_response(&jctx, &props, w3, &opts3).expect("0.98*omega1 (Auto) must converge");
    let want3 = dense_alpha_omega(&fx.evals, &fx.evecs, &mu_d, w3);
    assert_close(
        r3.alpha[0],
        want3,
        5e-3 * want3.abs().max(1.0),
        "alpha(0.98*omega1) vs dense closed form (Auto arm, close to a pole)",
    );
    assert_eq!(
        r3.solves[0].solver_used,
        DynamicSolver::Pcg,
        "Auto must resolve to Pcg strictly below omega1: got {:?}",
        r3.solves[0].solver_used
    );

    let default_window = DynamicResponseOpts::default().resonance_window;
    let w4 = omega2 * 0.9995;
    let opts4 = DynamicResponseOpts {
        tol: 1e-9,
        solver: DynamicSolver::Auto,
        spectrum: Some(spectrum.clone()),
        ..Default::default()
    };
    match linear_response(&jctx, &props, w4, &opts4) {
        Err(qntt::solvers::joint::ResponseError::NearResonance {
            omega,
            omega_n,
            gap,
            window,
        }) => {
            assert_close(omega, w4, 1e-12, "reported omega");
            assert_close(
                omega_n,
                omega2,
                1e-6 * omega2,
                "reported omega_n must be omega2",
            );
            assert_close(
                gap,
                omega2 - w4,
                1e-9 * omega2,
                "reported gap must be the true |omega - omega2|",
            );
            assert_close(
                window,
                default_window,
                1e-12,
                "reported window must be opts.resonance_window",
            );
        }
        other => panic!("expected NearResonance near omega2, got {other:?}"),
    }

    let w5 = omega2 * (1.0 - 1.5 * default_window);
    let opts5 = DynamicResponseOpts {
        tol: 1e-9,
        solver: DynamicSolver::Auto,
        spectrum: Some(spectrum),
        ..Default::default()
    };
    let r5 = linear_response(&jctx, &props, w5, &opts5)
        .expect("just outside omega2's window must be accepted, not refused");
    let want5 = dense_alpha_omega(&fx.evals, &fx.evecs, &mu_d, w5);
    assert_close(
        r5.alpha[0],
        want5,
        1e-3 * want5.abs().max(1.0),
        "alpha(omega2*(1-1.5*window)) vs dense closed form (just outside the window)",
    );
}

#[test]
fn tjw_pcg_refuses_above_the_first_pole() {
    let fx = sos_fixture();
    let jctx = joint_context(&fx.ham, &fx.x, &JointOpts::default())
        .expect("SOS fixture must be TT-critical");
    let props = [PropertyOperator::new("offdiag", fx.m2.clone())];

    let omega1 = fx.evals[1] - fx.evals[0];
    let omega2 = fx.evals[2] - fx.evals[0];
    assert!(
        omega2 > omega1 + 1e-6,
        "fixture must have non-degenerate low excitations: omega1={omega1:e} omega2={omega2:e}"
    );

    let w = 0.5 * (omega1 + omega2);
    let opts = DynamicResponseOpts {
        tol: 1e-9,
        solver: DynamicSolver::Pcg,
        ..Default::default()
    };
    match linear_response(&jctx, &props, w, &opts) {
        Err(qntt::solvers::joint::ResponseError::IndefiniteAbovePole {
            label, omega, rho, ..
        }) => {
            assert_eq!(label, "offdiag");
            assert_close(omega, w, 1e-12, "reported omega");
            assert!(
                rho < 0.0,
                "the negative-curvature quotient rho must be negative: {rho:e}"
            );
            println!("tjw_pcg_refuses_above_the_first_pole: IndefiniteAbovePole rho={rho:e}");
        }
        other => {
            panic!("PCG forced above omega1 must refuse with IndefiniteAbovePole, got {other:?}")
        }
    }
}
