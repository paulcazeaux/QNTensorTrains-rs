// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! [`su2_joint_ground_state`] — the alternating TT ⊕ orbital-rotation ground-state optimizer
//! (spin-adapted DMRG-SCF) built on the SU(2) joint Hessian of the parent module.

use crate::coeff::CoeffTensor;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::integrals::Integrals;
use crate::pointgroup::Irrep;
use crate::rotation::{expm_antisym, expm_kappa_blocked, rotate_integrals};
use crate::solvers::InnerProductVector;
use crate::solvers::hess::{NewtonPolishInfo, NewtonPolishOpts};
use crate::solvers::joint::{
    BallDot, JointOpts, JointScfInfo, JointScfOpts, JointSolver, SchurRoute, TrBall, TrModel,
    TrOpts, TrScore, TrTrial, stamp_dropped_diagnostic, tr_krylov_step,
};
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use crate::su2::hess::{Su2WeingartenCtx, newton_polish};
use crate::su2::joint::{Su2JointContext, Su2JointVector, su2_joint_context};
use crate::su2::orbital::orbital_gradient;
use crate::su2::qn::Su2Space;
use crate::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use crate::su2::solvers::als::{AlsOptions, als};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use crate::su2::weingarten::tangent_basis;

const ENERGY_NOISE_C: f64 = 32.0;

const GRAD_SANITY: f64 = 1e-5;

const TR_RESOLUTION_FLOOR: f64 = 5e-8;

/// The result of [`su2_joint_ground_state`].
pub struct Su2JointScfOut {
    /// The converged reduced energy `⟨x,Hx⟩` at the returned `(u, h, x)`. Add `ints.e0` for a
    /// total.
    pub lambda: f64,
    /// The SU(2) TT state — ALWAYS the output of a TT leg run under [`Self::h`].
    pub x: Su2TtVector<f64>,
    /// The accumulated one-particle rotation, row-major `d × d`.
    pub u: Vec<f64>,
    /// `Su2SparseHamiltonian` of `rotate_integrals(ints, u)` — the operator `x` and `lambda` refer
    /// to.
    pub h: Su2SparseHamiltonian<f64>,
    /// Per-run diagnostics. The shared abelian [`JointScfInfo`] is reused unchanged: it names no
    /// abelian TT type in its fields.
    pub info: JointScfInfo,
}

/// Alternating TT ⊕ orbital-rotation ground-state optimization of the reduced energy
/// `f(U) = min_{x ∈ M_r} ⟨x,H(U)x⟩` on the spin-adapted layer.
///
/// # Panics
///
/// On a malformed problem: `ints.norb != space.d`, a non-positive tolerance, `max_ls_trials == 0`,
/// a `PairMask` containing a cross-irrep pair (refused up front rather than left to trip
/// [`expm_kappa_blocked`]'s assert in the middle of a line search), or
/// [`SchurRoute::MatrixFree`], which this layer does not have.
pub fn su2_joint_ground_state(
    ints: &Integrals<f64>,
    space: Su2Space,
    x0: &Su2TtVector<f64>,
    opts: &JointScfOpts,
) -> Su2JointScfOut {
    let d = space.d;
    assert_eq!(
        ints.norb, d,
        "su2_joint_ground_state: integrals are over {} orbitals but the space has d = {d}",
        ints.norb
    );
    assert!(
        opts.orb_grad > 0.0 && opts.tt_grad_tol > 0.0 && opts.orb_floor >= 0.0,
        "su2_joint_ground_state: orb_grad/tt_grad_tol must be positive and orb_floor non-negative"
    );
    assert!(
        opts.max_ls_trials >= 1,
        "su2_joint_ground_state: max_ls_trials must be at least 1"
    );
    assert_eq!(
        opts.schur_route,
        SchurRoute::Dense,
        "su2_joint_ground_state: SchurRoute::MatrixFree is unavailable on the SU(2) layer — it \
         needs a pcg_solve binding over Su2TtTangent plus the SU(2) a_diagonal preconditioner \
         plumbing. SchurRoute::Dense is the default on both layers and the measured winner at \
         small n_eff; this refusal is a plumbing gap, not a soundness verdict."
    );
    assert!(
        opts.tr_init > 0.0 && opts.tr_init.is_finite() && opts.tr_init * opts.tr_init > 0.0,
        "su2_joint_ground_state: tr_init must be a positive, finite trust radius with a \
         representable square (got {:e})",
        opts.tr_init
    );
    assert!(
        opts.tr_max.is_finite() && opts.tr_max >= opts.tr_init,
        "su2_joint_ground_state: tr_max ({:e}) must be finite and >= tr_init ({:e})",
        opts.tr_max,
        opts.tr_init
    );
    assert!(
        opts.tr_cg_max_iter >= 1,
        "su2_joint_ground_state: tr_cg_max_iter must be at least 1"
    );

    assert!(
        opts.tr_legs_every >= 1,
        "su2_joint_ground_state: tr_legs_every is a period in accepted steps and must be at \
         least 1 (1 = the shipped leg-per-step protocol)"
    );
    assert!(
        opts.tr_legs_every == 1 || opts.solver == JointSolver::TrKrylov,
        "su2_joint_ground_state: tr_legs_every > 1 is the TrKrylov moving-quotient protocol; \
         the DenseSchur branch runs a TT leg inside every line-search trial, so there is no \
         post-acceptance leg to defer there. Set solver = JointSolver::TrKrylov, or leave \
         tr_legs_every = 1."
    );

    if let Some(mask) = opts.pairs.as_ref() {
        for &(p, q) in mask.keep.iter() {
            assert!(
                p < d && q < d && space.pg.irrep_of_sites([p, q]) == Irrep::IDENTITY,
                "su2_joint_ground_state: pair ({p},{q}) is not a legal same-irrep orbital rotation \
                 for this point group — a cross-irrep generator would break the block-diagonal \
                 rotation the whole no-re-screening argument rests on"
            );
        }
    }
    let orb_grad_full = opts.orb_grad_full.unwrap_or(opts.orb_grad);

    let mut u = identity(d);
    let mut h = build_h(ints, &u, space, opts);
    let mut ls_build_count = 0usize;

    let mut x = x0.clone();
    normalize_mut(&mut x);
    let t_leg0 = std::time::Instant::now();
    let (mut lambda, x_leg, mut leg_info) = tt_leg(&h, &x, opts, opts.cg_max_iter_initial, None);
    if opts.verbose {
        eprintln!(
            "su2_joint_scf: initial TT leg λ={lambda:.12e} ‖g‖={:.3e} in {:?}",
            leg_info.gnorm,
            t_leg0.elapsed()
        );
    }
    x = x_leg;

    let mut info = JointScfInfo {
        converged: false,
        stalled_at_floor: false,
        refused_reason: None,
        iters: 0,
        e_history: Vec::new(),
        g_orb_kept: Vec::new(),
        g_orb_full: Vec::new(),
        g_orb_dropped: Vec::new(),
        tt_gradnorms: Vec::new(),
        delta_k_norms: Vec::new(),
        trials_per_iter: Vec::new(),
        ls_build_count: 0,
        n_eff: 0,
        orb_spectrum: Vec::new(),
        rank_limited: false,
        g_dropped_final: 0.0,
        dropped_leak_bound: 0.0,
        dropped_angle: f64::NAN,
        sigma_min: f64::INFINITY,
        u_orth_dev: 0.0,
        hrr_min: leg_info.hrr_min,
        solver: opts.solver,
        tr_radii: Vec::new(),
        rho_history: Vec::new(),
        cg_iters: Vec::new(),
        cg_statuses: Vec::new(),
        rejections: 0,
        tt_legs: 1,
    };

    let mut tau_prev = 1.0f64;
    let mut delta = opts.tr_init;
    let mut since_leg = 0usize;
    let mut tt_fresh = true;

    for it in 0..opts.orb_max_iter {
        let jopts = JointOpts {
            tt_grad_tol: if tt_fresh {
                opts.tt_grad_tol
            } else {
                f64::INFINITY
            },
            orb_floor: opts.orb_floor,
            pairs: opts.pairs.clone(),
            z_route: opts.z_route,
            b_general: !tt_fresh,
        };
        let t_ctx = std::time::Instant::now();
        let jctx = match su2_joint_context(&h, &x, &jopts) {
            Ok(c) => c,
            Err(e) => {
                info.iters = it;
                info.refused_reason = Some(format!(
                    "su2_joint_ground_state: the joint context refused at orbital iteration {it} \
                     — {e}"
                ));
                break;
            }
        };
        info.iters = it + 1;
        let t_ctx = t_ctx.elapsed();

        lambda = jctx.lambda();
        let g = orbital_gradient(&h, &x);
        let gl: Vec<f64> = jctx.pairs().iter().map(|&(p, q)| g.get(&[p, q])).collect();
        let gk = jctx.vt_times(&gl);
        let gl_norm = norm2(&gl);
        let gk_norm = norm2(&gk);
        let g_dropped = {
            let back = jctx.v_times(&gk);
            norm2(&sub(&gl, &back))
        };

        info.e_history.push(lambda);
        info.g_orb_kept.push(gk_norm);
        info.g_orb_full.push(gl_norm);
        info.g_orb_dropped.push(g_dropped);
        info.tt_gradnorms.push(jctx.tt_gradnorm());
        info.n_eff = jctx.n_eff();
        info.orb_spectrum = jctx.orb_spectrum().to_vec();
        info.sigma_min = jctx.sigma_min();
        info.u_orth_dev = orth_dev(&u, d);

        if !tt_fresh && (gk_norm <= opts.orb_grad || jctx.n_eff() == 0) {
            refresh_leg(
                &h,
                opts,
                &mut x,
                &mut lambda,
                &mut leg_info,
                &mut info,
                &mut since_leg,
                &mut tt_fresh,
                it,
                "exit candidate",
            );
            continue;
        }
        let tt_ok = jctx.tt_gradnorm() <= opts.tt_grad_tol * lambda.abs().max(1.0);
        if gk_norm <= opts.orb_grad && tt_ok {
            info.converged = true;
            stamp_dropped_diagnostic(
                &mut info,
                jctx.orb_spectrum(),
                jctx.orb_floor(),
                g_dropped,
                orb_grad_full,
            );
            break;
        }
        if jctx.n_eff() == 0 {
            info.refused_reason = Some(format!(
                "su2_joint_ground_state: the orbital sector is metric-null (n_eff == 0) but the TT \
                 gate is unmet at orbital iteration {it} (‖P((H−λ)x)‖ = {:.6e} against {:.6e}). No \
                 orbital step can improve this point; converge x0 further, or lower orb_floor if \
                 the sector should not have been empty.",
                jctx.tt_gradnorm(),
                opts.tt_grad_tol * lambda.abs().max(1.0)
            ));
            break;
        }
        if it + 1 == opts.orb_max_iter {
            info.refused_reason = Some(format!(
                "su2_joint_ground_state: orbital iteration cap ({}) reached with \
                 ‖gk‖ = {gk_norm:.6e} (gate {:.3e}), ‖gL‖ = {gl_norm:.6e} (gate \
                 {orb_grad_full:.3e}).",
                opts.orb_max_iter, opts.orb_grad
            ));
            break;
        }

        if opts.solver == JointSolver::TrKrylov {
            let ghat = jctx.gradient_raised(&gk);
            let ghat_norm = ghat.norm();
            if info.e_history.len() >= 4 {
                let n = info.e_history.len();
                let best_recent = info.e_history[n - 3..]
                    .iter()
                    .fold(f64::INFINITY, |m, &v| m.min(v));
                let before = info.e_history[..n - 3]
                    .iter()
                    .fold(f64::INFINITY, |m, &v| m.min(v));
                let efloor = ENERGY_NOISE_C * f64::EPSILON * (1.0 + lambda.abs());
                if before - best_recent < efloor && gk_norm <= GRAD_SANITY {
                    if !tt_fresh {
                        refresh_leg(
                            &h,
                            opts,
                            &mut x,
                            &mut lambda,
                            &mut leg_info,
                            &mut info,
                            &mut since_leg,
                            &mut tt_fresh,
                            it,
                            "TR exit candidate",
                        );
                        continue;
                    }
                    info.stalled_at_floor = true;
                    info.converged = true;
                    if opts.verbose {
                        eprintln!(
                            "su2_joint_scf it={it}: TR stagnation exit — no resolvable energy \
                             improvement over 3 iterations at ‖gk‖={gk_norm:.3e} ⇒ converged \
                             (stalled_at_floor)"
                        );
                    }
                    break;
                }
            }
            if ghat_norm <= TR_RESOLUTION_FLOOR * lambda.abs().max(1.0) {
                if !tt_fresh {
                    refresh_leg(
                        &h,
                        opts,
                        &mut x,
                        &mut lambda,
                        &mut leg_info,
                        &mut info,
                        &mut since_leg,
                        &mut tt_fresh,
                        it,
                        "TR exit candidate",
                    );
                    continue;
                }
                if gk_norm <= GRAD_SANITY {
                    info.stalled_at_floor = true;
                    info.converged = true;
                    if opts.verbose {
                        eprintln!(
                            "su2_joint_scf it={it}: TR floor exit — ‖ĝ‖_G={ghat_norm:.3e} below the \
                             resolution floor at ‖gk‖={gk_norm:.3e} ⇒ converged \
                             (stalled_at_floor)"
                        );
                    }
                    break;
                }
                info.refused_reason = Some(format!(
                    "su2_joint_ground_state: raised joint gradient at the trust-region \
                     resolution floor (‖ĝ‖_G = {ghat_norm:.6e} ≤ \
                     {TR_RESOLUTION_FLOOR:.0e}·max(1,|λ|) = {:.3e}) at iteration {it} while the \
                     two-sided gate is unmet — below the floor the ratio test is a coin flip on \
                     the roundoff of an O(1) energy. Loosen orb_grad, or lower orb_floor if the \
                     unmet side is the full-pair gradient.",
                    TR_RESOLUTION_FLOOR * lambda.abs().max(1.0)
                ));
                break;
            }
            let tr_opts = TrOpts {
                tr_max: opts.tr_max,
                max_rej: opts.tr_max_rej,
                cg_max_iter: opts.tr_cg_max_iter,
                cg_rtol: opts.tr_cg_rtol,
                verbose: opts.verbose,
            };
            let delta_prev = delta;
            let t_tr = std::time::Instant::now();
            let ball_dot: BallDot<'_, Su2JointVector> = (opts.tr_ball == TrBall::Chart)
                .then_some(&chart_ball_dot as &dyn Fn(&Su2JointVector, &Su2JointVector) -> f64);
            let (acc, delta_new, st) = tr_krylov_step(
                |v| match opts.tr_model {
                    TrModel::Full => jctx.hess_apply_deflated(v),
                    TrModel::WLess => jctx.hess_apply_deflated_wless(v),
                },
                &ghat,
                lambda,
                delta,
                &tr_opts,
                None,
                ball_dot,
                |s: &Su2JointVector| {
                    let a = jctx.v_times(&s.c);
                    let dkn = std::f64::consts::SQRT_2 * norm2(&a);
                    let kstep = jctx.kappa_matrix(&scaled(&a, -1.0));
                    let u_step = if space.pg.order() > 1 {
                        expm_kappa_blocked(&kstep, space.pg)
                    } else {
                        expm_antisym(&flatten(&kstep, space.d), space.d)
                    };
                    let u_try = matmul(&u, &u_step, space.d);
                    let h_try = build_h(ints, &u_try, space, opts);
                    ls_build_count += 1;
                    let mut x_try = s.xi.retract_hosvd(1.0);
                    normalize_mut(&mut x_try);
                    let e_raw = rayleigh_quotient(&h_try, &x_try);
                    let (e_try, x_keep) = match opts.tr_score {
                        TrScore::AlsRounded => {
                            let aopts = AlsOptions {
                                max_iter: 1,
                                eps: 1e-9,
                                ..AlsOptions::default()
                            };
                            let (lam_a, x_a) = als(&h_try, &x_try, &aopts);
                            if lam_a < e_raw {
                                (lam_a, x_a)
                            } else {
                                (e_raw, x_try)
                            }
                        }
                    };
                    TrTrial {
                        payload: (u_try, h_try, x_keep),
                        e_try,
                        delta_k_norm: dkn,
                    }
                },
            );
            delta = delta_new;
            info.tr_radii.push(delta);
            info.rho_history.extend_from_slice(&st.rhos);
            info.cg_iters.extend_from_slice(&st.cg_iters);
            info.cg_statuses.extend_from_slice(&st.cg_statuses);
            info.rejections += st.rejections;
            info.trials_per_iter.push(st.rhos.len());
            if opts.verbose {
                eprintln!(
                    "su2_joint_scf it={it}: λ={lambda:.12e} ‖gk‖={gk_norm:.3e} \
                     ‖ĝ‖_G={ghat_norm:.3e} n_eff={} | ctx {t_ctx:.2?} | tr {} trial(s) \
                     Δ→{delta:.3e} {:.2?} accepted={}",
                    jctx.n_eff(),
                    st.rhos.len(),
                    t_tr.elapsed(),
                    acc.is_some()
                );
            }
            match acc {
                None => {
                    if delta < delta_prev {
                        continue;
                    }
                    if !tt_fresh {
                        refresh_leg(
                            &h,
                            opts,
                            &mut x,
                            &mut lambda,
                            &mut leg_info,
                            &mut info,
                            &mut since_leg,
                            &mut tt_fresh,
                            it,
                            "TR non-shrinking refusal",
                        );
                        continue;
                    }
                    if gk_norm <= GRAD_SANITY && info.rho_history.last().is_some_and(|r| r.is_nan())
                    {
                        info.stalled_at_floor = true;
                        info.converged = true;
                        if opts.verbose {
                            eprintln!(
                                "su2_joint_scf it={it}: TR floor exit — ρ-floor \
                                 refusal at an unshrinkable radius, ‖gk‖={gk_norm:.3e} ⇒ \
                                 converged (stalled_at_floor)"
                            );
                        }
                        break;
                    }
                    info.refused_reason = Some(format!(
                        "su2_joint_ground_state: the trust-region step refused at an \
                         unshrinkable radius (Δ = {delta:.3e}) at iteration {it} with \
                         ‖gk‖ = {gk_norm:.6e}, ‖ĝ‖_G = {ghat_norm:.6e}: the model's predicted \
                         decrease is at the energy's resolution floor and the trial energy is \
                         resolvably worse, so no radius can certify a step here."
                    ));
                    break;
                }
                Some(tr) => {
                    let (u_try, h_try, x_try) = tr.payload;
                    u = u_try;
                    h = h_try;
                    info.delta_k_norms.push(tr.delta_k_norm);
                    since_leg += 1;
                    if since_leg < opts.tr_legs_every {
                        x = x_try;
                        normalize_mut(&mut x);
                        lambda = tr.e_try;
                        tt_fresh = false;
                        continue;
                    }
                    since_leg = 0;
                    tt_fresh = true;
                    let leg_accuracy = None;
                    let (lam_w, x_w, leg_w) =
                        tt_leg(&h, &x_try, opts, opts.cg_max_iter, leg_accuracy);
                    x = x_w;
                    lambda = lam_w;
                    leg_info = leg_w;
                    info.hrr_min = leg_info.hrr_min;
                    info.tt_legs += 1;
                    if tr.delta_k_norm > opts.warm_threshold {
                        let (lam2, x2, leg2) =
                            tt_leg(&h, &x, opts, opts.cg_max_iter_initial, leg_accuracy);
                        info.tt_legs += 1;
                        if lam2 < lambda {
                            x = x2;
                            lambda = lam2;
                            leg_info = leg2;
                            info.hrr_min = leg_info.hrr_min;
                        }
                    }
                    continue;
                }
            }
        }

        let t_schur = std::time::Instant::now();
        let (dc, decrement) = schur_newton_step(&jctx, &gk, opts);
        let a_newton = jctx.v_times(&dc);
        let t_schur = t_schur.elapsed();

        let mut trials_here = 0usize;
        let t_ls = std::time::Instant::now();
        let mut accepted = linesearch(
            ints,
            space,
            &jctx,
            &u,
            &x,
            lambda,
            &gl,
            &a_newton,
            tau_prev.min(1.0),
            opts,
            &mut ls_build_count,
            &mut trials_here,
        );
        if accepted.is_none() {
            let a_sd = jctx.v_times(&scaled(&gk, -0.5));
            let n_sd = std::f64::consts::SQRT_2 * norm2(&a_sd);
            if n_sd > 0.0 {
                let a_sd = scaled(&a_sd, opts.trust0 / n_sd);
                accepted = linesearch(
                    ints,
                    space,
                    &jctx,
                    &u,
                    &x,
                    lambda,
                    &gl,
                    &a_sd,
                    1.0,
                    opts,
                    &mut ls_build_count,
                    &mut trials_here,
                );
            }
        }
        if opts.verbose {
            eprintln!(
                "su2_joint_scf it={it}: λ={lambda:.12e} ‖gk‖={gk_norm:.3e} ‖gL‖={gl_norm:.3e} \
                 n_eff={} iso={:.3e} | ctx {t_ctx:.2?} schur {t_schur:.2?} | ls {trials_here} \
                 trial(s) {:.2?} accepted={}",
                jctx.n_eff(),
                jctx.iso_defect(),
                t_ls.elapsed(),
                accepted.is_some()
            );
        }
        info.trials_per_iter.push(trials_here);

        let Some(step) = accepted else {
            let efloor = ENERGY_NOISE_C * f64::EPSILON * (1.0 + lambda.abs());
            if decrement <= efloor && gk_norm <= GRAD_SANITY {
                info.stalled_at_floor = true;
                info.converged = true;
                if opts.verbose {
                    eprintln!(
                        "su2_joint_scf it={it}: floor exit — decrement={decrement:.3e} ≤ \
                         efloor={efloor:.3e}, ‖gk‖={gk_norm:.3e} ⇒ converged (stalled_at_floor)"
                    );
                }
                break;
            }
            info.refused_reason = Some(if gk_norm <= opts.armijo_floor * lambda.abs().max(1.0) {
                format!(
                    "su2_joint_ground_state: orbital gradient at the Armijo resolution floor \
                     (‖gk‖ = {gk_norm:.6e} ≤ armijo_floor·max(1,|λ|) = {:.3e}) at iteration {it}, \
                     yet the model still promises decrease = {decrement:.3e} > efloor = \
                     {efloor:.3e}: the acceptance margin c₁·τ·|D| ∝ ‖gk‖²/κ is inside the roundoff \
                     of an O(1) energy re-evaluated by an independent TT leg per trial, so no step \
                     can be certified here. A deeper point needs a resolvable line search \
                     (exact-slope trust region), not a tighter gate.",
                    opts.armijo_floor * lambda.abs().max(1.0)
                )
            } else {
                format!(
                    "su2_joint_ground_state: orbital line search stalled at iteration {it} with \
                     ‖gk‖ = {gk_norm:.6e} (well above the Armijo resolution floor {:.3e}) and \
                     E = {lambda:.12e} after {trials_here} trials on both the Newton and the \
                     steepest-descent direction. This is NOT the expected roundoff-floor exit — \
                     suspect the curvature or the step scaling.",
                    opts.armijo_floor * lambda.abs().max(1.0)
                )
            });
            break;
        };

        info.delta_k_norms.push(step.step_norm);
        tau_prev = (2.0 * step.tau).min(1.0);
        u = step.u;
        h = step.h;
        x = step.x;
        lambda = step.lambda;
        leg_info = step.leg;
        info.hrr_min = leg_info.hrr_min;
    }

    if !info.converged && info.refused_reason.is_none() {
        info.refused_reason = Some(
            "su2_joint_ground_state: stopped without certifying and without a recorded obstruction \
             (orb_max_iter == 0?)."
                .to_string(),
        );
    }
    info.ls_build_count = ls_build_count;
    info.u_orth_dev = orth_dev(&u, d);

    Su2JointScfOut {
        lambda,
        x,
        u,
        h,
        info,
    }
}

#[allow(clippy::too_many_arguments)]
fn refresh_leg(
    h: &Su2SparseHamiltonian<f64>,
    opts: &JointScfOpts,
    x: &mut Su2TtVector<f64>,
    lambda: &mut f64,
    leg_info: &mut NewtonPolishInfo,
    info: &mut JointScfInfo,
    since_leg: &mut usize,
    tt_fresh: &mut bool,
    it: usize,
    why: &str,
) {
    let (lam_r, x_r, leg_r) = tt_leg(h, x, opts, opts.cg_max_iter, None);
    *x = x_r;
    *lambda = lam_r;
    *leg_info = leg_r;
    info.hrr_min = leg_info.hrr_min;
    info.tt_legs += 1;
    *since_leg = 0;
    *tt_fresh = true;
    if opts.verbose {
        eprintln!(
            "su2_joint_scf it={it}: moving-quotient refresh — {why} at a stale point, TT leg \
             re-run before deciding (λ={lambda:.12e})"
        );
    }
}

fn tt_leg(
    h: &Su2SparseHamiltonian<f64>,
    x_warm: &Su2TtVector<f64>,
    opts: &JointScfOpts,
    cg_cap: usize,
    accuracy_override: Option<f64>,
) -> (f64, Su2TtVector<f64>, NewtonPolishInfo) {
    let mut x = x_warm.clone();
    normalize_mut(&mut x);

    let lam0 = rayleigh_quotient(h, &x);
    let abs_gate = opts.tt_grad_tol * lam0.abs().max(1.0);
    let accuracy = accuracy_override.unwrap_or(opts.orb_grad * opts.tt_leg_margin);
    let polish_tol = abs_gate.min(accuracy).max(abs_gate * 1e-4);
    let g0 = {
        let (bl, br) = tangent_bases(&x);
        Su2WeingartenCtx::from_parts(h, bl, br, lam0)
            .gradient()
            .norm()
    };

    if g0 > opts.cg_skip_factor * polish_tol {
        let mut copts = RiemannianOpts::new(&x.space);
        copts.eps = polish_tol / lam0.abs().max(f64::MIN_POSITIVE);
        copts.max_iter = cg_cap;
        copts.preserve_grid = opts.tt_preserve_grid;
        let (_l, xc) = riemannian_cg(h, &x, &copts);
        x = xc;
    }

    let popts = NewtonPolishOpts {
        gradtol: polish_tol,
        ..opts.polish
    };
    let (mut xp, mut pinfo) = newton_polish(h, &x, &popts);

    if !pinfo.converged && cg_cap < opts.cg_max_iter_initial {
        let mut ropts = RiemannianOpts::new(&xp.space);
        ropts.eps = polish_tol / lam0.abs().max(f64::MIN_POSITIVE);
        ropts.max_iter = cg_cap.saturating_mul(4).max(cg_cap + 1);
        ropts.preserve_grid = opts.tt_preserve_grid;
        let (_l, xc) = riemannian_cg(h, &xp, &ropts);
        let (xp2, pinfo2) = newton_polish(h, &xc, &popts);
        xp = xp2;
        pinfo = pinfo2;
    }

    normalize_mut(&mut xp);
    let lam = rayleigh_quotient(h, &xp);
    (lam, xp, pinfo)
}

fn schur_newton_step(
    jctx: &Su2JointContext<'_>,
    gk: &[f64],
    opts: &JointScfOpts,
) -> (Vec<f64>, f64) {
    let n = jctx.n_eff();
    if n == 0 {
        return (Vec::new(), 0.0);
    }
    let wctx = jctx.weingarten();

    let (shear, raw_tb): (Vec<Su2TtTangent<f64>>, Vec<Su2TtTangent<f64>>) = {
        let (bl, br) = jctx.shared_bases();
        let tb: Vec<Su2JointVector> = tangent_basis(bl, br, true)
            .into_iter()
            .map(|xi| Su2JointVector::from_tangent(jctx.spec(), xi))
            .collect();
        let shear = (0..n)
            .map(|j| {
                let mut c = vec![0.0f64; n];
                c[j] = 1.0;
                let mut seed = Su2JointVector::from_coeffs(jctx.spec(), jctx, c);
                for _pass in 0..2 {
                    for b in tb.iter() {
                        let cf = b.dot(&seed);
                        seed.axpy(-cf, b);
                    }
                }
                seed.xi
            })
            .collect();
        (shear, tb.into_iter().map(|b| b.xi).collect())
    };

    let mut hthc: Vec<Su2TtTangent<f64>> = Vec::with_capacity(n);
    let mut hcc = vec![0.0f64; n * n];
    for j in 0..n {
        let mut e = vec![0.0f64; n];
        e[j] = 1.0;
        let a_j = jctx.v_times(&e);
        let (f_xi, fc_full) = jctx.ab_covector(&shear[j], &a_j);
        let fc_kept = jctx.vt_times(&fc_full);
        for (i, &val) in fc_kept.iter().enumerate() {
            hcc[i * n + j] = val + shear[i].dot(&f_xi);
        }
        hthc.push(f_xi);
    }
    let f_theta = wctx.gradient().clone();

    let (schur, coupling) = schur_terms_dense(jctx, raw_tb, &hthc, &f_theta);

    let mut heff = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            heff[i * n + j] = hcc[i * n + j] - schur[i * n + j];
        }
    }
    let heff = symmetrize(&heff, n);
    let rhs: Vec<f64> = (0..n)
        .map(|i| coupling[i] - 0.5 * gk[i] - shear[i].dot(&f_theta))
        .collect();

    let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&heff, n, n);
    let dpd = opts.eig_floor_rel * evals.iter().fold(1e-14f64, |m, &e| m.max(e.abs()));
    let mut dc = vec![0.0f64; n];
    let mut decrement = 0.0f64;
    for (k, &ev) in evals.iter().enumerate() {
        let proj: f64 = (0..n).map(|j| evecs[k][j] * rhs[j]).sum();
        let denom = ev.abs().max(dpd);
        let s = proj / denom;
        decrement += 0.5 * proj * proj / denom;
        for (i, out) in dc.iter_mut().enumerate() {
            *out += evecs[k][i] * s;
        }
    }
    (dc, decrement)
}

fn schur_terms_dense(
    jctx: &Su2JointContext<'_>,
    raw_tb: Vec<Su2TtTangent<f64>>,
    hthc: &[Su2TtTangent<f64>],
    f_theta: &Su2TtTangent<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let n = hthc.len();
    let wctx = jctx.weingarten();
    let (bl, br) = jctx.shared_bases();
    let tb: Vec<Su2TtTangent<f64>> = raw_tb
        .into_iter()
        .map(|xi| Su2TtTangent::project_shared(bl, br, &xi.to_vector()))
        .collect();
    let n_tt = tb.len();
    if n_tt == 0 {
        return (vec![0.0; n * n], vec![0.0; n]);
    }

    let mut htt = vec![0.0f64; n_tt * n_tt];
    for (j, xj) in tb.iter().enumerate() {
        let mut fj = wctx.a_apply_fused(xj);
        fj.axpy(1.0, &wctx.b_apply(xj));
        for (i, xi) in tb.iter().enumerate() {
            htt[i * n_tt + j] = xi.dot(&fj);
        }
    }
    let htt = symmetrize(&htt, n_tt);

    let mut hthc_c = vec![0.0f64; n_tt * n];
    for (j, cj) in hthc.iter().enumerate() {
        for (i, xi) in tb.iter().enumerate() {
            hthc_c[i * n + j] = xi.dot(cj);
        }
    }
    let f_theta_c: Vec<f64> = tb.iter().map(|xi| xi.dot(f_theta)).collect();

    let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&htt, n_tt, n_tt);
    let anchor = evals.iter().fold(1.0f64, |m, &e| m.max(e.abs()));
    let dtheta = 1e-10 * anchor;
    let solve_theta = |y: &[f64]| -> Vec<f64> {
        let mut out = vec![0.0f64; n_tt];
        for (k, &ev) in evals.iter().enumerate() {
            let proj: f64 = (0..n_tt).map(|i| evecs[k][i] * y[i]).sum();
            let s = proj / ev.max(dtheta);
            for (i, o) in out.iter_mut().enumerate() {
                *o += evecs[k][i] * s;
            }
        }
        out
    };

    let mut schur = vec![0.0f64; n * n];
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for j in 0..n {
        let col: Vec<f64> = (0..n_tt).map(|i| hthc_c[i * n + j]).collect();
        cols.push(solve_theta(&col));
    }
    for i in 0..n {
        for j in 0..n {
            schur[i * n + j] = (0..n_tt).map(|k| hthc_c[k * n + i] * cols[j][k]).sum();
        }
    }
    let w = solve_theta(&f_theta_c);
    let coupling: Vec<f64> = (0..n)
        .map(|i| (0..n_tt).map(|k| hthc_c[k * n + i] * w[k]).sum())
        .collect();
    (schur, coupling)
}

struct Accepted {
    u: Vec<f64>,
    h: Su2SparseHamiltonian<f64>,
    x: Su2TtVector<f64>,
    lambda: f64,
    tau: f64,
    step_norm: f64,
    leg: NewtonPolishInfo,
}

#[allow(clippy::too_many_arguments)]
fn linesearch(
    ints: &Integrals<f64>,
    space: Su2Space,
    jctx: &Su2JointContext<'_>,
    u_acc: &[f64],
    x: &Su2TtVector<f64>,
    e0: f64,
    gl: &[f64],
    a: &[f64],
    tau0: f64,
    opts: &JointScfOpts,
    ls_build_count: &mut usize,
    n_trials: &mut usize,
) -> Option<Accepted> {
    let s = std::f64::consts::SQRT_2 * norm2(a);
    if s == 0.0 {
        return None;
    }
    let a: Vec<f64> = if s > opts.trust0 {
        scaled(a, opts.trust0 / s)
    } else {
        a.to_vec()
    };
    let dslope: f64 = gl.iter().zip(a.iter()).map(|(g, t)| g * t).sum();
    if dslope >= 0.0 || dslope.is_nan() {
        return None;
    }
    let step_len = std::f64::consts::SQRT_2 * norm2(&a);
    let efloor = ENERGY_NOISE_C * f64::EPSILON * (1.0 + e0.abs());

    let mut tau = tau0;
    for _trial in 0..opts.max_ls_trials {
        let kstep = jctx.kappa_matrix(&scaled(&a, -tau));
        let u_step = if space.pg.order() > 1 {
            expm_kappa_blocked(&kstep, space.pg)
        } else {
            expm_antisym(&flatten(&kstep, space.d), space.d)
        };
        let u_try = matmul(u_acc, &u_step, space.d);
        let h_try = build_h(ints, &u_try, space, opts);
        *ls_build_count += 1;
        *n_trials += 1;

        let (e_try, x_try, leg) = tt_leg(&h_try, x, opts, opts.cg_max_iter, None);

        let need = (opts.armijo_c * tau * dslope).min(-efloor);
        if e_try <= e0 + need {
            return Some(Accepted {
                u: u_try,
                h: h_try,
                x: x_try,
                lambda: e_try,
                tau,
                step_norm: tau * step_len,
                leg,
            });
        }
        tau *= 0.5;
    }
    None
}

fn build_h(
    ints: &Integrals<f64>,
    u: &[f64],
    space: Su2Space,
    opts: &JointScfOpts,
) -> Su2SparseHamiltonian<f64> {
    let rotated = rotate_integrals(ints, u);
    Su2SparseHamiltonian::new(space, &rotated.t, &rotated.v, None, opts.ham_eps)
}

fn identity(d: usize) -> Vec<f64> {
    crate::rotation::identity_matrix(d)
}

fn matmul(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    crate::rotation::compose_rotations(a, b, d)
}

fn flatten(k: &CoeffTensor<f64>, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = k.get(&[i, j]);
        }
    }
    out
}

fn orth_dev(u: &[f64], d: usize) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            let s: f64 = (0..d).map(|k| u[k * d + i] * u[k * d + j]).sum();
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((s - want).abs());
        }
    }
    worst
}

fn symmetrize(m: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = 0.5 * (m[i * n + j] + m[j * n + i]);
        }
    }
    out
}

fn chart_ball_dot(a: &Su2JointVector, b: &Su2JointVector) -> f64 {
    a.xi.dot(&b.xi) + 2.0 * a.c.iter().zip(b.c.iter()).map(|(x, y)| x * y).sum::<f64>()
}

fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|t| t * t).sum::<f64>().sqrt()
}

fn scaled(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|t| t * s).collect()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
