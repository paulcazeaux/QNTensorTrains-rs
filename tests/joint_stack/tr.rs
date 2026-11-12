// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::Integrals;
use qntt::core::tangent::TtTangent;
use qntt::solvers::InnerProductVector;
use qntt::solvers::joint::{
    JointOpts, JointScfOpts, JointSolver, JointVector, joint_context, joint_ground_state,
};
use qntt::solvers::steihaug::{SteihaugStatus, steihaug_cg};
use qntt::{QnSpace, orbital_gradient, rotate_integrals};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::optimize::{D2_ORB_GRAD, capped_randn, d2_fixture};
use crate::util::*;

fn d2_tr_opts() -> JointScfOpts {
    JointScfOpts {
        solver: JointSolver::TrKrylov,
        orb_max_iter: 40,
        orb_grad: D2_ORB_GRAD,
        tt_grad_tol: 1e-6,
        construction: qntt::hamiltonian::assembly::Construction::Median,
        ..JointScfOpts::default()
    }
}

#[test]
fn tjt_t1_wiring_matches_certified_grams() {
    let fx = d2_fixture();
    let jctx = joint_context(&fx.out.h, &fx.out.x, &JointOpts::default())
        .expect("the D2 endpoint must re-gate");

    let g = orbital_gradient(&fx.out.h, &fx.out.x);
    let gl: Vec<f64> = jctx.pairs().iter().map(|&(p, q)| g.get(&[p, q])).collect();
    let gk = jctx.vt_times(&gl);

    let (basis_raw, _report) = jctx.joint_basis();
    let n = basis_raw.len();
    assert!(
        n > 10,
        "anti-vacuity: the D2 endpoint must have a rich joint basis, got {n}"
    );
    let (bl, br) = jctx.bases();
    let basis: Vec<_> = basis_raw
        .into_iter()
        .map(|mut b| {
            b.xi = TtTangent::project(bl, br, &b.xi.to_vector());
            b
        })
        .collect();
    let (ga, gb) = jctx.joint_hessian_grams(&basis);
    let hrr: Vec<f64> = ga.iter().zip(gb.iter()).map(|(x, y)| x + y).collect();

    let ghat = jctx.gradient_raised(&gk);
    let f_xi = jctx.weingarten().gradient();
    let g_scale = ghat.norm().max(1e-30);
    for (i, b) in basis.iter().enumerate() {
        let lhs = b.dot(&ghat);
        let mut rhs = b.xi.dot(f_xi);
        for (cj, gj) in b.c.iter().zip(gk.iter()) {
            rhs += cj * 0.5 * gj;
        }
        assert!(
            (lhs - rhs).abs() <= 1e-10 * g_scale.max(rhs.abs()),
            "raised-gradient covector identity failed at basis vector {i}: ⟨b,ĝ⟩_G = {lhs:e} \
             vs covector {rhs:e}"
        );
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(4242);
    let r: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
    let mut v = basis[0].clone();
    v.scale(r[0]);
    for i in 1..n {
        v.axpy(r[i], &basis[i]);
    }
    let w = jctx.hess_apply(&v);
    let h_scale = hrr.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let r_norm = r.iter().map(|x| x * x).sum::<f64>().sqrt();
    for (i, b) in basis.iter().enumerate() {
        let lhs = b.dot(&w);
        let rhs: f64 = (0..n).map(|j| hrr[i * n + j] * r[j]).sum();
        assert!(
            (lhs - rhs).abs() <= 1e-8 * h_scale * r_norm,
            "operator-vs-certified-Gram identity failed at row {i}: ⟨b,H·v⟩_G = {lhs:e} vs \
             Gram·r = {rhs:e} (scale {h_scale:e})"
        );
    }

    let r_mode = TtTangent::project(bl, br, bl);
    let rn = r_mode.norm();
    assert!(
        rn > 0.0,
        "the radial projection must not vanish at this endpoint"
    );
    let wd = jctx.hess_apply_deflated(&v);
    let leak_w = r_mode.dot(&wd.xi).abs() / rn;
    assert!(
        leak_w <= 1e-10 * wd.norm().max(1e-30),
        "hess_apply_deflated must return a radial-free tangent half: ⟨radial, w.ξ⟩ = {leak_w:e}"
    );
    let leak_g = r_mode.dot(&ghat.xi).abs() / rn;
    assert!(
        leak_g <= 1e-10 * ghat.norm().max(1e-30),
        "gradient_raised must return a radial-free tangent half: ⟨radial, ĝ.ξ⟩ = {leak_g:e}"
    );
}

#[test]
fn tjt_t3_factor2_pin() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);
    let opts = JointScfOpts {
        tr_init: 5e-5,
        orb_max_iter: 2,
        ..d2_tr_opts()
    };
    let out = joint_ground_state(&fx.ints, fx.space, &x0, &opts);

    let rho0 = *out
        .info
        .rho_history
        .first()
        .expect("the run must have attempted at least one trial");
    assert!(
        rho0.is_finite(),
        "the first trial must not be a ρ-floor event: {rho0}"
    );
    assert!(
        (rho0 - 1.0).abs() <= 0.15,
        "★ factor-2 pin: first-trial ρ at a Cauchy-regime radius must be ≈ 1, got {rho0}"
    );
    assert_eq!(
        out.info.cg_statuses.first().copied(),
        Some(SteihaugStatus::Boundary),
        "the premise of the pin is a boundary-truncated first solve"
    );
    assert!(
        out.info.delta_k_norms.first().copied().unwrap_or(0.0) > 0.0,
        "a ρ ≈ 1 trial must have been accepted"
    );
}

#[test]
fn tjt_t4_branch_coverage() {
    let fx = d2_fixture();

    let x0 = capped_randn(fx.space, 1, 1, 57);
    let probe = joint_ground_state(
        &fx.ints,
        fx.space,
        &x0,
        &JointScfOpts {
            orb_max_iter: 1,
            ..d2_tr_opts()
        },
    );
    assert!(
        !probe.info.converged,
        "one iteration cannot certify the hot D2 start"
    );
    let jctx = joint_context(&probe.h, &probe.x, &JointOpts::default())
        .expect("the initial leg's endpoint must gate");
    let g = orbital_gradient(&probe.h, &probe.x);
    let gl: Vec<f64> = jctx.pairs().iter().map(|&(p, q)| g.get(&[p, q])).collect();
    let gk = jctx.vt_times(&gl);
    let gk_norm = gk.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        gk_norm > 1e-3,
        "anti-vacuity: the probe point must be orbital-hot, ‖gk‖ = {gk_norm:e}"
    );
    let ghat = jctx.gradient_raised(&gk);
    let delta = 1e6;
    let (s, ci) = steihaug_cg(
        |v| jctx.hess_apply_deflated(v),
        &ghat,
        delta,
        1e-10,
        0.0,
        100,
    );
    assert!(ci.iters >= 1, "the solve must be non-vacuous");
    assert!(
        ci.model_red >= 0.0,
        "model_red must be non-negative by construction"
    );
    assert!(
        s.norm() <= delta * (1.0 + 1e-12),
        "the step must respect the ball"
    );
    if ci.boundary_hit {
        assert!(
            matches!(
                ci.status,
                SteihaugStatus::Boundary | SteihaugStatus::NegCurve
            ),
            "boundary_hit without a truncation status: {:?}",
            ci.status
        );
    }

    let ints_conv = rotate_integrals(&fx.ints, &fx.out.u);
    let out = joint_ground_state(&ints_conv, fx.space, &fx.out.x, &d2_tr_opts());
    assert!(
        out.info.converged,
        "a converged start must re-certify: {:?}",
        out.info.refused_reason
    );
    assert_eq!(out.info.iters, 1, "the gate must fire at iteration 0");
    assert!(
        out.info.tr_radii.is_empty(),
        "no TR call may have run: {:?}",
        out.info.tr_radii
    );
    assert!(out.info.delta_k_norms.is_empty());
    assert_eq!(out.info.rejections, 0);
}

#[test]
fn tjt_t5_refusal_parity() {
    let (space, _ham, x, t, v) = lih_cation_rank1_fixture();
    let ints = Integrals::<f64> {
        t,
        v,
        e0: 0.0,
        norb: space.d,
        nelec: space.n(),
        ms2: 1,
        orbsym: Vec::new(),
        isym: None,
    };
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        orb_max_iter: 30,
        orb_grad: 1e-6,
        orb_grad_full: Some(1e-9),
        orb_floor: 1e-2,
        construction: qntt::hamiltonian::assembly::Construction::Median,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&ints, space, &x, &opts);
    assert!(
        out.info.converged,
        "the kept gradient and TT gate are both met, so this must certify: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.info.refused_reason.is_none(),
        "a rank-limited point is not a refusal: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.info.rank_limited,
        "the TR branch must flag the same rank limitation the dense branch does"
    );
    let gk_last = out.info.g_orb_kept.last().copied().unwrap_or(f64::INFINITY);
    let gl_last = out.info.g_orb_full.last().copied().unwrap_or(0.0);
    assert!(
        gk_last <= opts.orb_grad,
        "the KEPT gradient must have met its own gate — the refusal is about the DROPPED \
         part: {gk_last:e}"
    );
    assert!(
        gl_last > opts.orb_grad_full.unwrap(),
        "the FULL gradient must NOT have met the tight full threshold: {gl_last:e}"
    );
    assert!(
        out.info.g_dropped_final > opts.orb_grad_full.unwrap(),
        "rank_limited must be driven by the DROPPED component: {:e}",
        out.info.g_dropped_final
    );
    assert!(
        out.info.dropped_angle.is_finite() && out.info.dropped_angle > 0.0,
        "a dropped mode exists, so its angle must be reported: {}",
        out.info.dropped_angle
    );
    assert!(
        out.info.iters < opts.orb_max_iter,
        "the rank-limited exit must fire before the iteration cap: {} vs {}",
        out.info.iters,
        opts.orb_max_iter
    );
}

#[test]
fn tjt_t9_cascade_retry() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);
    let opts = JointScfOpts {
        tr_init: 1e3,
        tr_max: 1e3,
        tr_max_rej: 2,
        orb_max_iter: 8,
        ..d2_tr_opts()
    };
    let out = joint_ground_state(&fx.ints, fx.space, &x0, &opts);

    assert!(
        !out.info.delta_k_norms.is_empty(),
        "★ retry contract: a step must eventually be accepted after the cascades — a \
         terminal-cascade break never gets here. radii: {:?}, ρ: {:?}",
        out.info.tr_radii,
        out.info.rho_history
    );
    assert!(
        out.info.tr_radii.len() > out.info.delta_k_norms.len(),
        "anti-vacuity: at least one whole cascade must have been exhausted and retried \
         (radii {:?} vs accepted {:?})",
        out.info.tr_radii,
        out.info.delta_k_norms
    );
    assert!(
        out.info.tr_radii.first().copied().unwrap_or(f64::INFINITY) < 1e3,
        "the first call must have returned a SHRUNKEN radius: {:?}",
        out.info.tr_radii
    );
    assert!(
        out.info.rejections > opts.tr_max_rej,
        "at least one full cascade's worth of shrinks must be recorded: {}",
        out.info.rejections
    );
    let e0 = *out
        .info
        .e_history
        .first()
        .expect("at least one iteration ran");
    assert!(
        out.lambda < e0 && out.lambda >= fx.e_fci - 1e-10,
        "the retried run must make genuine progress: λ = {:.12e}, start {:.12e}, FCI {:.12e}",
        out.lambda,
        e0,
        fx.e_fci
    );
}

fn tiny_problem() -> (Integrals<f64>, QnSpace) {
    let d = 3usize;
    let (t, v) = generic_truncated_tv(d);
    let ints = Integrals::<f64> {
        t,
        v,
        e0: 0.0,
        norb: d,
        nelec: 2,
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    };
    (ints, QnSpace::new(d, 1, 1))
}

#[test]
#[should_panic(expected = "tr_init")]
fn tjt_t10a_door_tr_init_zero() {
    let (ints, space) = tiny_problem();
    let x0 = capped_randn(space, 1, 1, 7);
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        tr_init: 0.0,
        ..JointScfOpts::default()
    };
    let _ = joint_ground_state(&ints, space, &x0, &opts);
}

#[test]
#[should_panic(expected = "tr_init")]
fn tjt_t10b_door_tr_init_square_underflows() {
    let (ints, space) = tiny_problem();
    let x0 = capped_randn(space, 1, 1, 7);
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        tr_init: 1e-170,
        ..JointScfOpts::default()
    };
    let _ = joint_ground_state(&ints, space, &x0, &opts);
}

#[test]
#[should_panic(expected = "tr_max")]
fn tjt_t10c_door_tr_max_below_init() {
    let (ints, space) = tiny_problem();
    let x0 = capped_randn(space, 1, 1, 7);
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        tr_init: 1.0,
        tr_max: 0.5,
        ..JointScfOpts::default()
    };
    let _ = joint_ground_state(&ints, space, &x0, &opts);
}

#[test]
#[should_panic(expected = "tr_cg_max_iter")]
fn tjt_t10d_door_zero_cg_budget_branch_independent() {
    let (ints, space) = tiny_problem();
    let x0 = capped_randn(space, 1, 1, 7);
    let opts = JointScfOpts {
        solver: JointSolver::DenseSchur,
        tr_cg_max_iter: 0,
        ..JointScfOpts::default()
    };
    let _ = joint_ground_state(&ints, space, &x0, &opts);
}

fn tr_probe_coeffs(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn tr_combine(basis: &[JointVector], c: &[f64]) -> JointVector {
    let mut v = basis[0].clone();
    v.scale(0.0);
    for (b, &ci) in basis.iter().zip(c.iter()) {
        v.axpy(ci, b);
    }
    v
}

fn tr_image_dev(a: &JointVector, b: &JointVector) -> f64 {
    let mut dxi = a.xi.clone();
    dxi.axpy(-1.0, &b.xi);
    let dc: f64 =
        a.c.iter()
            .zip(b.c.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
    let num = (dxi.dot(&dxi) + dc).sqrt();
    let den = (a.xi.dot(&a.xi) + a.c.iter().map(|x| x * x).sum::<f64>()).sqrt();
    num / den.max(1e-30)
}

#[test]
fn tjt_t16_wless_gate_removes_exactly_the_weingarten_term() {
    let fx = d2_fixture();
    let cb = joint_context(&fx.out.h, &fx.out.x, &JointOpts::default())
        .expect("the D2 endpoint must re-gate");
    let (bb, _) = cb.joint_basis();
    let n = bb.len();
    assert!(
        n > 10,
        "anti-vacuity: the D2 endpoint must have a rich joint basis, got {n}"
    );
    let v = tr_combine(&bb, &tr_probe_coeffs(n, 0x5EED_5EED));
    let full = cb.hess_apply_deflated(&v);
    let wless = cb.hess_apply_deflated_wless(&v);
    let live = tr_image_dev(&full, &wless);
    assert!(
        live > 1e-3,
        "anti-vacuity: the tt-Weingarten term must be live at the D2 endpoint — \
         dev(Full, WLess) = {live:e}"
    );
    let wxi = cb.weingarten().b_apply(&v.xi);
    let mut delta_wless = full.clone();
    delta_wless.axpy(-1.0, &wless);
    let scale = full.norm().max(1e-30);
    let mut pure = 0usize;
    let mut biggest = 0.0f64;
    for (i, b) in bb.iter().enumerate() {
        if !b.c.iter().all(|&x| x == 0.0) {
            continue;
        }
        pure += 1;
        let target_w = b.xi.dot(&wxi);
        biggest = biggest.max(target_w.abs());
        let dev_w = (b.dot(&delta_wless) - target_w).abs();
        assert!(
            dev_w <= 1e-10 * scale,
            "the W-LESS gate must remove EXACTLY the tt-Weingarten term at basis vector {i}: \
             ⟨b, Full − WLess⟩ = {:e} vs ⟨ξ_b, W ξ_v⟩ = {target_w:e} (scale {scale:e})",
            b.dot(&delta_wless)
        );
    }
    assert!(
        pure > 0 && biggest > 1e-8 * scale,
        "anti-vacuity: the identity must be checked on pure-tangent basis vectors against a \
         resolvable W pairing — {pure} such vectors, largest |⟨ξ_b, W ξ_v⟩| = {biggest:e}"
    );
}

const TJT_T17_LAMBDA: f64 = -4.738_699_220_252_271;
const TJT_T17_RADII: &[f64] = &[0.1, 0.2, 0.4, 0.8, 0.4];
const TJT_T17_RHO: &[f64] = &[
    1.0072138860339466,
    1.0086056008005637,
    1.0049021726563638,
    0.8725782133754202,
    -0.9070664876894372,
    0.8916486878971903,
];
const TJT_T17_CG: &[usize] = &[1, 1, 1, 2, 1, 1];

#[test]
fn tjt_t17_knobs_off_short_run_matches_the_recorded_trace() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);
    let opts = JointScfOpts {
        orb_max_iter: 6,
        ..d2_tr_opts()
    };
    let out = joint_ground_state(&fx.ints, fx.space, &x0, &opts);
    println!(
        "tjt_t17: λ = {:.17e}, radii = {:?}, ρ = {:?}, cg = {:?}",
        out.lambda, out.info.tr_radii, out.info.rho_history, out.info.cg_iters
    );

    const LAMBDA: f64 = TJT_T17_LAMBDA;
    assert!(
        (out.lambda - LAMBDA).abs() <= 1e-12 * LAMBDA.abs(),
        "the knobs-off trajectory moved: λ = {:.17e} against the recorded {LAMBDA:.17e}",
        out.lambda
    );
    assert_eq!(
        out.info.tr_radii.len(),
        TJT_T17_RADII.len(),
        "iteration count moved: {:?}",
        out.info.tr_radii
    );
    for (i, (&got, &want)) in out
        .info
        .tr_radii
        .iter()
        .zip(TJT_T17_RADII.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "the radius trace moved at iteration {i}: {got:e} against the recorded {want:e}"
        );
    }
    assert_eq!(
        out.info.rho_history.len(),
        TJT_T17_RHO.len(),
        "trial count moved: {:?}",
        out.info.rho_history
    );
    for (i, (&got, &want)) in out
        .info
        .rho_history
        .iter()
        .zip(TJT_T17_RHO.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() <= 1e-7 * want.abs().max(1.0),
            "the ρ history moved at trial {i}: {got:e} against the recorded {want:e}"
        );
    }
    assert_eq!(
        out.info.cg_iters.as_slice(),
        TJT_T17_CG,
        "the Steihaug iteration counts moved"
    );
}

#[test]
#[should_panic(expected = "cached the damped gauge set")]
fn tjt_t18c_damped_gauge_cache_is_one_s_per_context() {
    let fx = d2_fixture();
    let jctx = joint_context(&fx.out.h, &fx.out.x, &JointOpts::default())
        .expect("the D2 endpoint must re-gate");
    let (basis, _) = jctx.joint_basis();
    let v = tr_combine(&basis, &tr_probe_coeffs(basis.len(), 11));
    let wctx = jctx.weingarten();
    let _ = wctx.b_apply_damped(&v.xi, 1e-6);
    let _ = wctx.b_apply_damped(&v.xi, 1e-5);
}

#[test]
#[should_panic(expected = "tr_legs_every is a period in accepted steps")]
fn tjt_t22a_door_legs_every_zero() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);
    let opts = JointScfOpts {
        tr_legs_every: 0,
        ..d2_tr_opts()
    };
    let _ = joint_ground_state(&fx.ints, fx.space, &x0, &opts);
}

#[test]
#[should_panic(expected = "moving-quotient protocol")]
fn tjt_t22b_door_legs_every_needs_trkrylov() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);
    let opts = JointScfOpts {
        solver: JointSolver::DenseSchur,
        tr_legs_every: 2,
        ..d2_tr_opts()
    };
    let _ = joint_ground_state(&fx.ints, fx.space, &x0, &opts);
}

#[test]
fn tjt_t22_moving_quotient_certifies_with_fewer_legs() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);
    let base = d2_tr_opts();
    let out1 = joint_ground_state(&fx.ints, fx.space, &x0, &base);
    let out3 = joint_ground_state(
        &fx.ints,
        fx.space,
        &x0,
        &JointScfOpts {
            tr_legs_every: 3,
            ..d2_tr_opts()
        },
    );
    println!(
        "tjt_t22: k=1 λ = {:.15e} legs = {} iters = {} | k=3 λ = {:.15e} legs = {} iters = {} \
         | Δλ = {:.3e}",
        out1.lambda,
        out1.info.tt_legs,
        out1.info.iters,
        out3.lambda,
        out3.info.tt_legs,
        out3.info.iters,
        out3.lambda - out1.lambda
    );
    assert!(
        out1.info.converged && out3.info.converged,
        "both protocols must certify: k=1 {:?}, k=3 {:?}",
        out1.info.refused_reason,
        out3.info.refused_reason
    );
    assert!(
        out3.info.tt_legs < out1.info.tt_legs,
        "the deferred protocol must run fewer driver legs: {} vs {}",
        out3.info.tt_legs,
        out1.info.tt_legs
    );
    assert!(
        (out3.lambda - out1.lambda).abs() <= 1e-9 * out1.lambda.abs(),
        "the two protocols must land on the same certified endpoint: Δλ = {:.3e}",
        out3.lambda - out1.lambda
    );
}
