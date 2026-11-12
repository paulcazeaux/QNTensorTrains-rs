// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! State-specific MALS: one train swept against a list of FIXED lower states.

use rand_xoshiro::Xoshiro256PlusPlus;

use crate::core::factor::{Sketch, dense_self_adjoint_smallest};
use crate::core::twosite::{ContractedSparseCores, contract, factor_qc, noise_rng};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::matvec::onesite_workspace;
use crate::hamiltonian::rayleigh_quotient;
use crate::ops::dot;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;
use crate::solvers::als::{INNER_TOL_CAP, boundary_idsketch, right_to_left_sketching};
use crate::solvers::davidson::precond;
use crate::solvers::davidson_block::{
    BlockDavidsonOpts, davidson_block, davidson_block_projected, home_roots,
};
use crate::solvers::interface::clamp_hi_first;
use crate::solvers::mals::{
    MalsHistory, MalsOptions, NoiseType, SweepTrunc, THETA_TOL_FACTOR, mals_run, split_two_site,
};
use crate::solvers::multistate::{sa_maxsubspace, sa_restart_keep};
use crate::solvers::overlap_env::{
    OverlapEnvs, local_overlap_vector, overlap_envs, overlap_step_left, overlap_step_right,
};
use crate::vector::TtVector;

/// Which excited-state route the sweep takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SsMode {
    /// Add `w·Σⱼ|ψ̂ⱼ⟩⟨ψ̂ⱼ|` to the local operator and take its lowest root. The default.
    #[default]
    Penalty,
    /// Leave the operator alone and search the orthogonal complement of the lower states'
    /// local images. `penalty` is recorded `0.0` throughout.
    Project,
}

/// How [`SsMode::Penalty`] chooses its weight.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum PenaltyRule {
    /// [`penalty_weight`] at the top of every iteration: `w = max(10·(λ⊥ − minⱼEⱼ), 1)` keyed on
    /// the ORTHOGONAL-COMPLEMENT energy estimate, with a collapse escalation. The default.
    #[default]
    Auto,
    /// A constant weight, UNGUARDED: no floor, no escalation, no collapse gate on the stop test.
    /// Below the gap to the target this has the lower state as its own global minimizer (a
    /// recorded negative — see the module header).
    Fixed(f64),
}

/// State-specific MALS options: the single-state [`MalsOptions`] plus what only a penalized
/// sweep has to say.
#[derive(Clone, Debug)]
pub struct StateSpecificOptions {
    /// The sweep schedule: `eps`, `bond_dims`, `noise`, `noise_type`, `max_iter`, `verbose`,
    /// `conv_tol` — read exactly as [`crate::solvers::mals::mals`] reads them, and
    /// `seed` is the same single noise stream. `NoiseType::Random` IS allowed here (unlike the
    /// state-average sweep): the truncation is single-state.
    pub mals: MalsOptions,
    /// Penalty or projection (see [`SsMode`]).
    pub mode: SsMode,
    /// The weight rule; ignored in [`SsMode::Project`], where `w` is 0 by construction.
    pub penalty: PenaltyRule,
    /// How many local roots to solve for. `1` labels by energy; `> 1` solves a block and keeps
    /// the root that CONTINUES the incoming block (greedy homing), which is what lets a run
    /// follow a start that is not the lowest penalized state.
    pub roots: usize,
    /// Reference energies `Eⱼ` for the weight rule, one per lower state. `None` computes
    /// `RQ(ψⱼ)/‖ψⱼ‖²` once at entry.
    pub lower_energies: Option<Vec<f64>>,
    /// Block-Davidson subspace ceiling; `None` = [`sa_maxsubspace`]`(roots)`.
    pub maxsubspace: Option<usize>,
    /// Thick-restart keep count; `None` = [`sa_restart_keep`]`(roots, maxsubspace)`. At
    /// `roots = 1` the resolved pair is the single-root solver's own defaults, which is what
    /// makes the `w = 0` parity with `mals` bitwise.
    pub restart_keep: Option<usize>,
}

impl Default for StateSpecificOptions {
    fn default() -> Self {
        StateSpecificOptions {
            mals: MalsOptions::default(),
            mode: SsMode::default(),
            penalty: PenaltyRule::default(),
            roots: 1,
            lower_energies: None,
            maxsubspace: None,
            restart_keep: None,
        }
    }
}

/// One outer iteration's worth of diagnostics (a forward + a back sweep).
#[derive(Clone, Debug)]
pub struct SsSweep {
    /// The PLAIN energy `λ = θ − w·Σⱼ|⟨pⱼ,u⟩|²` at the back sweep's last bond — the Rayleigh
    /// quotient of `H` alone, and what the caller reports as the state's energy.
    pub energy: f64,
    /// The PENALIZED Ritz value `θ` at the same bond. The sweep minimizes THIS, so it is what the
    /// stop test runs on; it equals `energy` at `w = 0` and in [`SsMode::Project`].
    pub penalized: f64,
    /// Largest relative discarded weight over this iteration's bonds.
    pub discarded_weight: f64,
    /// Largest total (summed-over-sectors) bond dimension reached.
    pub max_bond_dim: usize,
    /// The rank cap in force this iteration (`0` = uncapped).
    pub bond_dim: usize,
    /// The noise amplitude in force this iteration.
    pub noise: f64,
    /// `|⟨ψ̂ⱼ, x⟩|` at the END of this iteration, one per lower state in the caller's order.
    /// Empty on the delegate path.
    pub overlaps: Vec<f64>,
    /// The weight `w` in force this iteration; `0.0` in [`SsMode::Project`] and on the delegate.
    pub penalty: f64,
    /// Bonds where the homed continuation was not the lowest Ritz vector (`roots > 1` only).
    pub swaps: usize,
    /// Local operator applies over both sweeps. `0` on the delegate path (documented).
    pub numops: usize,
    /// Bonds whose local space could not host the requested vectors and were taken as
    /// `u = cp0/‖cp0‖` instead of solved.
    pub skipped: usize,
    /// Whether the stop test fired on this iteration.
    pub converged: bool,
}

/// [`state_specific_mals_with_history`]'s result: the final PLAIN energy plus every iteration.
#[derive(Clone, Debug)]
pub struct SsHistory {
    pub lambda: f64,
    pub sweeps: Vec<SsSweep>,
}

pub(crate) const ROOTS_SALT: u64 = 0x5353_4D41_4C53_0001;

/// `w = max(10·(λ⊥ − minⱼ Eⱼ), 1)` with the orthogonal-complement estimate
/// `λ⊥ = (λₓ − Σⱼ Eⱼ oⱼ²)/(1 − Σⱼ oⱼ²)`, falling back to `λ⊥ = λₓ` once the overlaps saturate
/// (`Σoⱼ² > 1 − 1e-4`, where the quotient is numerical noise); with the COLLAPSE ESCALATION
/// `Σoⱼ² > 1/2 ⇒ w ≥ 10·w_prev`.
///
/// # Panics
/// When `overlaps` and `e_lower` disagree on length.
pub fn penalty_weight(lambda_x: f64, overlaps: &[f64], e_lower: &[f64], w_prev: f64) -> f64 {
    assert_eq!(
        overlaps.len(),
        e_lower.len(),
        "penalty_weight: {} overlaps for {} reference energies — one of each per lower state",
        overlaps.len(),
        e_lower.len()
    );
    let m = e_lower.len();
    if m == 0 {
        return 0.0;
    }
    let mut sigma = 0.0f64;
    let mut s = 0.0f64;
    for j in 0..m {
        let o2 = overlaps[j] * overlaps[j];
        sigma += o2;
        s += e_lower[j] * o2;
    }
    let e_min = e_lower.iter().copied().fold(f64::INFINITY, f64::min);
    let lambda_perp = if sigma <= 1.0 - 1e-4 {
        (lambda_x - s) / (1.0 - sigma)
    } else {
        lambda_x
    };
    let mut w = (10.0 * (lambda_perp - e_min)).max(1.0);
    if sigma > 0.5 {
        w = w.max(10.0 * w_prev);
    }
    w
}

pub(crate) struct SsLocalOut<V> {
    pub(crate) theta: f64,
    pub(crate) lambda: f64,
    pub(crate) u: V,
    pub(crate) swap: usize,
    pub(crate) numops: usize,
    pub(crate) skipped: usize,
}

pub(crate) fn local_basis<V: InnerProductVector>(ps: &[V]) -> Vec<V> {
    let m = ps.len();
    if m == 0 {
        return Vec::new();
    }
    let mut g: Vec<V::Scalar> = vec![V::Scalar::zero(); m * m];
    for i in 0..m {
        for j in 0..m {
            g[i * m + j] = ps[i].dot(&ps[j]);
        }
    }
    let (mu, vecs) = dense_self_adjoint_smallest(&g, m, m);
    let mu_max = mu[m - 1];
    let floor = (1e-10 * mu_max).max(1e-20);
    let mut qs = Vec::with_capacity(m);
    for c in 0..m {
        if mu[c] <= floor {
            continue;
        }
        let s = 1.0 / mu[c].sqrt();
        let mut q = ps[0].clone();
        q.scale(vecs[c][0] * V::Scalar::from_f64(s));
        for (j, p) in ps.iter().enumerate().skip(1) {
            q.axpy(vecs[c][j] * V::Scalar::from_f64(s), p);
        }
        qs.push(q);
    }
    qs
}

pub(crate) fn project_off<V: InnerProductVector>(t: &mut V, qs: &[V]) {
    for q in qs {
        let c = q.dot(t);
        t.axpy(-c, q);
    }
}

pub(crate) fn randn_like<V: InnerProductVector>(v: &V, rng: &mut Xoshiro256PlusPlus) -> V {
    let mut r = v.clone();
    r.for_each_coeff_mut(|z| *z = V::Scalar::sample_standard(rng));
    r
}

pub(crate) fn overlap_sq_sum<V: InnerProductVector>(ps: &[V], u: &V) -> f64 {
    let mut acc = 0.0f64;
    for p in ps {
        let c = p.dot(u).abs_s();
        acc += c * c;
    }
    acc
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn ss_local_core<V, Op, Prec>(
    op: &mut Op,
    applyprec: &mut Prec,
    cp0: &V,
    ps: &[V],
    w: f64,
    mode: SsMode,
    roots: usize,
    inner_tol: f64,
    maxsubspace: usize,
    restart_keep: usize,
    rngr: &mut Xoshiro256PlusPlus,
) -> SsLocalOut<V>
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
    Prec: FnMut(&mut V, f64),
{
    let need_basis = !ps.is_empty() && (mode == SsMode::Project || w > 0.0);
    let qs = if need_basis {
        local_basis(ps)
    } else {
        Vec::new()
    };
    let blocked = if mode == SsMode::Project { qs.len() } else { 0 };
    let mut skip = cp0.num_coeffs() < roots + blocked;

    let mut v0 = cp0.clone();
    if !skip && need_basis {
        let n0 = cp0.norm();
        let mut g = cp0.clone();
        project_off(&mut g, &qs);
        let ng = g.norm();
        if ng <= f64::EPSILON.sqrt() * n0 {
            let rr = randn_like(cp0, rngr);
            let rn = rr.norm();
            let mut gr = cp0.clone();
            if rn > 0.0 {
                gr.axpy(V::Scalar::from_f64(n0 / rn), &rr);
            }
            project_off(&mut gr, &qs);
            if gr.norm() <= f64::EPSILON.sqrt() * n0 {
                skip = true;
            } else {
                v0 = gr;
            }
        } else if ng < 0.5 * n0 {
            g.scale(V::Scalar::from_f64(n0 / ng));
            v0 = g;
        }
    }

    if skip {
        let mut u = cp0.clone();
        let n = u.norm();
        if n > 0.0 {
            u.scale(V::Scalar::from_f64(1.0 / n));
        }
        let hu = op(&u);
        let theta = u.dot(&hu).re_s();
        let lambda = if mode == SsMode::Penalty && w > 0.0 {
            theta - w * overlap_sq_sum(ps, &u)
        } else {
            theta
        };
        return SsLocalOut {
            theta,
            lambda,
            u,
            swap: 0,
            numops: 1,
            skipped: 1,
        };
    }

    let mut v0s = Vec::with_capacity(roots);
    v0s.push(v0);
    for _ in 1..roots {
        let rr = randn_like(cp0, rngr);
        let rn = rr.norm();
        let n0 = v0s[0].norm();
        let mut gi = v0s[0].clone();
        if rn > 0.0 {
            gi.axpy(V::Scalar::from_f64(0.1 * n0 / rn), &rr);
        }
        v0s.push(gi);
    }

    let bopts = BlockDavidsonOpts {
        tol: inner_tol,
        theta_tol: Some(inner_tol * THETA_TOL_FACTOR),
        maxsubspace,
        restart_keep,
        ..BlockDavidsonOpts::for_roots(roots)
    };
    let (thetas, us, info) = if mode == SsMode::Project {
        let mut project = |t: &mut V| project_off(t, &qs);
        davidson_block_projected(op, &v0s, applyprec, &mut project, roots, &bopts)
    } else {
        davidson_block(op, &v0s, applyprec, roots, &bopts)
    };
    assert!(
        us.len() == roots,
        "state_specific_mals: the block Davidson returned {} of {roots} roots — the subspace \
         stagnated before it could host them; supply roots distinct guesses",
        us.len()
    );

    let idx = if roots > 1 {
        let (perm, _) = home_roots(std::slice::from_ref(cp0), &us);
        perm[0]
    } else {
        0
    };
    let theta = thetas[idx];
    let u = us.into_iter().nth(idx).expect("idx < us.len()");
    let lambda = if mode == SsMode::Penalty && w > 0.0 {
        theta - w * overlap_sq_sum(ps, &u)
    } else {
        theta
    };
    SsLocalOut {
        theta,
        lambda,
        u,
        swap: usize::from(idx != 0),
        numops: info.numops,
        skipped: 0,
    }
}

pub(crate) fn delegate_sweeps(
    hist: &MalsHistory,
    lambda_raw: f64,
    opts: &MalsOptions,
) -> Vec<SsSweep> {
    let mut sweeps = Vec::with_capacity(hist.sweeps.len());
    let mut r_prev = 1.0f64;
    for (it, e) in hist.sweeps.iter().enumerate() {
        let inner_tol = clamp_hi_first(0.01 * r_prev, opts.eps, INNER_TOL_CAP);
        let prev = if it == 0 {
            lambda_raw
        } else {
            hist.sweeps[it - 1].energy
        };
        let r = (prev - e.energy).abs() / e.energy.abs();
        let converged = r < opts.convergence_tol() && inner_tol <= opts.eps && e.noise == 0.0;
        r_prev = r;
        sweeps.push(SsSweep {
            energy: e.energy,
            penalized: e.energy,
            discarded_weight: e.discarded_weight,
            max_bond_dim: e.max_bond_dim,
            bond_dim: e.bond_dim,
            noise: e.noise,
            overlaps: Vec::new(),
            penalty: 0.0,
            swaps: 0,
            numops: 0,
            skipped: 0,
            converged,
        });
    }
    sweeps
}

/// Rank-adaptive two-site search for the lowest state ORTHOGONAL to (or penalized against) the
/// fixed `lower` states. Returns `(λ, x)` with `λ` the PLAIN energy; does not mutate `x0`.
pub fn state_specific_mals<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    lower: &[TtVector<T>],
    opts: &StateSpecificOptions,
) -> (f64, TtVector<T>) {
    let (hist, x) = state_specific_mals_with_history(h, x0, lower, opts);
    (hist.lambda, x)
}

/// [`state_specific_mals`] keeping the per-iteration diagnostics: the plain and penalized
/// energies, the weight in force, the measured overlaps, and the swap/skip/apply counts.
pub fn state_specific_mals_with_history<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    lower: &[TtVector<T>],
    opts: &StateSpecificOptions,
) -> (SsHistory, TtVector<T>) {
    let mut x = x0.clone();
    let hist = state_specific_mals_mut(h, &mut x, lower, opts);
    (hist, x)
}

/// In-place [`state_specific_mals`]. `lower` is never mutated and need not be normalized: any
/// finite nonzero norm is accepted and `1/‖ψⱼ‖` is folded into the local images, the reference
/// energies and the recorded overlaps.
pub fn state_specific_mals_mut<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    lower: &[TtVector<T>],
    opts: &StateSpecificOptions,
) -> SsHistory {
    assert!(
        !opts.mals.bond_dims.is_empty(),
        "MalsOptions::bond_dims must be non-empty (use [0] for uncapped)"
    );
    assert!(
        !opts.mals.noise.is_empty(),
        "MalsOptions::noise must be non-empty (use [0.0] for a noise-free run)"
    );
    assert!(
        opts.mals.noise.iter().all(|a| a.is_finite() && *a >= 0.0),
        "MalsOptions::noise entries must be finite and non-negative"
    );
    assert!(
        opts.roots >= 1,
        "state_specific_mals: roots = {} must be ≥ 1",
        opts.roots
    );
    if let PenaltyRule::Fixed(w) = opts.penalty {
        assert!(
            w.is_finite() && w >= 0.0,
            "state_specific_mals: PenaltyRule::Fixed({w}) must be finite and non-negative"
        );
    }
    let m = lower.len();
    let mut invn: Vec<f64> = Vec::with_capacity(m);
    for (j, psi) in lower.iter().enumerate() {
        assert_eq!(
            psi.space, x.space,
            "state_specific_mals: lower[{j}] lives in a different space than the iterate"
        );
        let n = psi.norm();
        assert!(
            n.is_finite() && n > 0.0,
            "state_specific_mals: lower[{j}] has norm {n} — a finite, NONZERO norm is required \
             (any other value is accepted and normalized internally, the state is never mutated)"
        );
        invn.push(if n == 1.0 { 1.0 } else { 1.0 / n });
    }
    if let Some(e) = &opts.lower_energies {
        assert_eq!(
            e.len(),
            m,
            "state_specific_mals: {} lower_energies for {m} lower state(s)",
            e.len()
        );
    }

    if m == 0 && opts.roots == 1 {
        return ss_delegate(h, x, &opts.mals);
    }

    let d = x.space.d;
    let ms = opts
        .maxsubspace
        .unwrap_or_else(|| sa_maxsubspace(opts.roots));
    let rk = opts
        .restart_keep
        .unwrap_or_else(|| sa_restart_keep(opts.roots, ms));
    let mut rng = noise_rng(opts.mals.seed);
    let mut rngr = noise_rng(opts.mals.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ROOTS_SALT);
    let mut sweeps: Vec<SsSweep> = Vec::with_capacity(opts.mals.max_iter);
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr: Vec<Option<Sketch<T>>> = right_to_left_sketching(h, x)
        .into_iter()
        .map(Some)
        .collect();
    let mut envs: Vec<OverlapEnvs<T>> = lower.iter().map(|psi| overlap_envs(x, psi)).collect();
    let e_lower: Vec<f64> = match &opts.lower_energies {
        Some(e) => e.clone(),
        None => (0..m)
            .map(|j| rayleigh_quotient(h, &lower[j]).re_s() * invn[j] * invn[j])
            .collect(),
    };

    let lambda_raw = rayleigh_quotient(h, x).re_s();
    let nx = x.norm();
    let mut lambda_x = lambda_raw / (nx * nx);
    let mut theta_pen = lambda_raw;
    let mut r = 1.0f64;
    let mut overlaps: Vec<f64> = (0..m)
        .map(|j| dot(&lower[j], x).abs_s() * invn[j] / nx)
        .collect();
    if opts.roots > 1 {
        let sigma: f64 = overlaps.iter().map(|o| o * o).sum();
        if sigma > 0.5 {
            eprintln!(
                "state_specific_mals: the start carries Σⱼ|⟨ψ̂ⱼ,x₀⟩|² = {sigma:.4} > 1/2 with \
                 roots = {} — the label the homing continues is a coin flip; a nonzero final \
                 `swaps` is the tell",
                opts.roots
            );
        }
    }
    let mut w_prev = 0.0f64;

    for it in 0..opts.mals.max_iter {
        let bond_dim = opts.mals.bond_dims[it.min(opts.mals.bond_dims.len() - 1)];
        let noise = opts.mals.noise[it.min(opts.mals.noise.len() - 1)];
        let inner_tol = clamp_hi_first(0.01 * r, opts.mals.eps, INNER_TOL_CAP);
        let ep = if d > 1 {
            opts.mals.eps / ((d - 1) as f64).sqrt()
        } else {
            opts.mals.eps
        };
        let w = match (opts.mode, opts.penalty) {
            (SsMode::Project, _) => 0.0,
            (SsMode::Penalty, PenaltyRule::Auto) => {
                penalty_weight(lambda_x, &overlaps, &e_lower, w_prev)
            }
            (SsMode::Penalty, PenaltyRule::Fixed(v)) => v,
        };

        let mut fwd = ss_forward_sweep(
            h,
            x,
            &mut fr,
            &mut envs,
            lower,
            &invn,
            w,
            opts.mode,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.mals.noise_type,
            &mut rng,
            &mut rngr,
            opts.roots,
            ms,
            rk,
        );
        if opts.mals.verbose {
            println!(
                "SS-MALS iter {it} forward sweep: energy = {} (penalized {}), penalty = {w}",
                fwd.lambda, fwd.theta
            );
        }
        debug_assert!(
            fr.iter().all(Option::is_none),
            "the forward sweep must consume the whole right stack"
        );
        let back = ss_back_sweep(
            h,
            x,
            &mut fwd.env,
            &mut envs,
            lower,
            &invn,
            w,
            opts.mode,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.mals.noise_type,
            &mut rng,
            &mut rngr,
            opts.roots,
            ms,
            rk,
        );
        debug_assert!(
            fwd.env.iter().all(Option::is_none),
            "the back sweep must consume the whole left stack"
        );
        fr = back.env;

        let trunc = fwd.trunc.merge(back.trunc);
        for j in 0..m {
            overlaps[j] = dot(&lower[j], x).abs_s();
            if invn[j] != 1.0 {
                overlaps[j] *= invn[j];
            }
        }
        let sigma: f64 = overlaps.iter().map(|o| o * o).sum();
        let collapsed = opts.mode == SsMode::Penalty
            && matches!(opts.penalty, PenaltyRule::Auto)
            && sigma > 0.5;
        if collapsed {
            eprintln!(
                "state_specific_mals: iteration {it} collapsed onto the lower states \
                 (Σⱼ|⟨ψ̂ⱼ,x⟩|² = {sigma:.4} at penalty w = {w}, below the gap to the target); the \
                 rule raises the weight to ≥ {} next",
                10.0 * w
            );
        }
        r = (theta_pen - back.theta).abs() / back.theta.abs();
        let converged = r < opts.mals.convergence_tol()
            && inner_tol <= opts.mals.eps
            && noise == 0.0
            && !collapsed;
        theta_pen = back.theta;
        lambda_x = back.lambda;
        w_prev = w;
        sweeps.push(SsSweep {
            energy: back.lambda,
            penalized: back.theta,
            discarded_weight: trunc.max_discarded,
            max_bond_dim: trunc.max_kept,
            bond_dim,
            noise,
            overlaps: overlaps.clone(),
            penalty: w,
            swaps: fwd.swaps + back.swaps,
            numops: fwd.numops + back.numops,
            skipped: fwd.skipped + back.skipped,
            converged,
        });
        if opts.mals.verbose {
            println!(
                "SS-MALS iter {it} back sweep: energy = {} (penalized {}), relative improvement = \
                 {r}, discarded weight = {:.3e}, M = {}, noise = {noise:.1e}",
                back.lambda, back.theta, trunc.max_discarded, trunc.max_kept
            );
        }
        if converged {
            break;
        }
    }
    if sweeps.last().map(|s| !s.converged).unwrap_or(false) {
        eprintln!(
            "state_specific_mals: max_iter = {} reached without convergence (last relative change \
             {r}, penalty {}, overlaps {:?})",
            opts.mals.max_iter,
            w_prev,
            sweeps
                .last()
                .map(|s| s.overlaps.clone())
                .unwrap_or_default()
        );
    }
    SsHistory {
        lambda: lambda_x,
        sweeps,
    }
}

fn ss_delegate<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    opts: &MalsOptions,
) -> SsHistory {
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let lambda_raw = rayleigh_quotient(h, x).re_s();
    let hist = mals_run(h, x, opts);
    let sweeps = delegate_sweeps(&hist, lambda_raw, opts);
    if sweeps.last().map(|s| !s.converged).unwrap_or(false) {
        eprintln!(
            "state_specific_mals: max_iter = {} reached without convergence (mals delegate)",
            opts.max_iter
        );
    }
    SsHistory {
        lambda: hist.lambda,
        sweeps,
    }
}

struct SsSweepOut<T: Scalar> {
    theta: f64,
    lambda: f64,
    env: Vec<Option<Sketch<T>>>,
    trunc: SweepTrunc,
    swaps: usize,
    numops: usize,
    skipped: usize,
}

fn ss_local_images<T: Scalar>(
    envs: &[OverlapEnvs<T>],
    lower: &[TtVector<T>],
    invn: &[f64],
    k: usize,
) -> Vec<ContractedSparseCores<T>> {
    (0..lower.len())
        .map(|j| {
            let mut p = local_overlap_vector(&envs[j], &lower[j], k);
            if invn[j] != 1.0 {
                p.scale(T::from_f64(invn[j]));
            }
            p
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn ss_local_solve<T: Scalar>(
    h: &SparseHamiltonian<T>,
    fl: &Sketch<T>,
    fr: &Sketch<T>,
    cp0: &ContractedSparseCores<T>,
    ps: &[ContractedSparseCores<T>],
    w: f64,
    mode: SsMode,
    roots: usize,
    inner_tol: f64,
    maxsubspace: usize,
    restart_keep: usize,
    rngr: &mut Xoshiro256PlusPlus,
) -> SsLocalOut<ContractedSparseCores<T>> {
    let d_diag = h.two_site_effective_diagonal(fl, fr, cp0);
    let (xk0, xk10) = factor_qc(cp0);
    let il = h.row_ranges(&xk0);
    let jr = h.col_ranges(&xk10);
    let mut op = |cp: &ContractedSparseCores<T>| {
        let mut y = h.two_site_sketched_hamiltonian_cached(cp, fl, fr, &il, &jr);
        if w > 0.0 {
            let wt = T::from_f64(w);
            for p in ps.iter() {
                let c = p.dot(cp);
                y.axpy(wt * c, p);
            }
        }
        y
    };
    let mut applyprec = |t: &mut ContractedSparseCores<T>, theta: f64| precond(t, &d_diag, theta);
    ss_local_core(
        &mut op,
        &mut applyprec,
        cp0,
        ps,
        w,
        mode,
        roots,
        inner_tol,
        maxsubspace,
        restart_keep,
        rngr,
    )
}

#[allow(clippy::too_many_arguments)]
fn ss_forward_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fr: &mut [Option<Sketch<T>>],
    envs: &mut [OverlapEnvs<T>],
    lower: &[TtVector<T>],
    invn: &[f64],
    w: f64,
    mode: SsMode,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
    rngr: &mut Xoshiro256PlusPlus,
    roots: usize,
    maxsubspace: usize,
    restart_keep: usize,
) -> SsSweepOut<T> {
    debug_assert_eq!(x.ortho, Some(0), "SS-MALS forward sweep expects center 0");
    let d = x.space.d;
    let m = lower.len();
    let mut fl: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(x, 0, 1));
    fr[0] = None;
    let mut out = SsSweepOut {
        theta: 0.0,
        lambda: 0.0,
        env: Vec::new(),
        trunc: SweepTrunc::default(),
        swaps: 0,
        numops: 0,
        skipped: 0,
    };

    for k in 0..(d - 1) {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let ps = if m > 0 && (mode == SsMode::Project || w > 0.0) {
            ss_local_images(envs, lower, invn, k)
        } else {
            Vec::new()
        };
        let o = ss_local_solve(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &cp0,
            &ps,
            w,
            mode,
            roots,
            inner_tol,
            maxsubspace,
            restart_keep,
            rngr,
        );
        out.theta = o.theta;
        out.lambda = o.lambda;
        out.swaps += o.swap;
        out.numops += o.numops;
        out.skipped += o.skipped;
        let (xk, s, mut xk1, info) = split_two_site(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &o.u,
            ep,
            max_rank,
            noise,
            noise_type,
            true,
            rng,
        );
        fr[k + 1] = None;
        out.trunc.absorb(info);
        xk1.fold_sketch_left_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k + 1);
        if k < d - 2 {
            let ws = onesite_workspace(h, x.core(k));
            let next = h.sketching_step_right(x, fl[k].as_ref().unwrap(), &ws);
            fl[k + 1] = Some(next);
            for j in 0..m {
                let prev = envs[j].l[k]
                    .as_ref()
                    .expect("l[k] built by the previous bond");
                let step = overlap_step_right(x, &lower[j], prev, k);
                envs[j].l[k + 1] = Some(step);
            }
        }
    }
    out.env = fl;
    out
}

#[allow(clippy::too_many_arguments)]
fn ss_back_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fl: &mut [Option<Sketch<T>>],
    envs: &mut [OverlapEnvs<T>],
    lower: &[TtVector<T>],
    invn: &[f64],
    w: f64,
    mode: SsMode,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
    rngr: &mut Xoshiro256PlusPlus,
    roots: usize,
    maxsubspace: usize,
    restart_keep: usize,
) -> SsSweepOut<T> {
    let d = x.space.d;
    let m = lower.len();
    if x.ortho != Some(d - 1) {
        x.move_core(d - 1, false);
    }
    let mut fr: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(x, d, 1));
    fl[d - 1] = None;
    let mut out = SsSweepOut {
        theta: 0.0,
        lambda: 0.0,
        env: Vec::new(),
        trunc: SweepTrunc::default(),
        swaps: 0,
        numops: 0,
        skipped: 0,
    };

    for k in (0..(d - 1)).rev() {
        let cp0 = contract(x.core(k), x.core(k + 1));
        let ps = if m > 0 && (mode == SsMode::Project || w > 0.0) {
            ss_local_images(envs, lower, invn, k)
        } else {
            Vec::new()
        };
        let o = ss_local_solve(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &cp0,
            &ps,
            w,
            mode,
            roots,
            inner_tol,
            maxsubspace,
            restart_keep,
            rngr,
        );
        out.theta = o.theta;
        out.lambda = o.lambda;
        out.swaps += o.swap;
        out.numops += o.numops;
        out.skipped += o.skipped;
        let (mut xk, s, xk1, info) = split_two_site(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &o.u,
            ep,
            max_rank,
            noise,
            noise_type,
            false,
            rng,
        );
        fl[k] = None;
        out.trunc.absorb(info);
        xk.fold_sketch_right_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k);
        if k > 0 {
            let ws = onesite_workspace(h, x.core(k + 1));
            let next = h.sketching_step_left(x, fr[k + 1].as_ref().unwrap(), &ws);
            fr[k] = Some(next);
            for j in 0..m {
                let prev = envs[j].r[k + 2]
                    .as_ref()
                    .expect("r[k+2] built by the previous bond");
                let step = overlap_step_left(x, &lower[j], prev, k + 1);
                envs[j].r[k + 1] = Some(step);
            }
        }
    }
    out.env = fr;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::mals::MalsSweep;
    use crate::solvers::steihaug::testvec::DVec;
    use rand::SeedableRng;

    fn unit(n: usize, i: usize) -> DVec {
        let mut v = vec![0.0; n];
        v[i] = 1.0;
        DVec(v)
    }

    fn cosine(a: &DVec, b: &DVec) -> f64 {
        a.dot(b).abs() / (a.norm() * b.norm())
    }

    fn run_core(
        diag: &[f64],
        cp0: &DVec,
        ps: &[DVec],
        w: f64,
        mode: SsMode,
        roots: usize,
        rngr: &mut Xoshiro256PlusPlus,
    ) -> (SsLocalOut<DVec>, Vec<DVec>) {
        let ms = sa_maxsubspace(roots);
        let rk = sa_restart_keep(roots, ms);
        let mut seen: Vec<DVec> = Vec::new();
        let out = {
            let mut op = |v: &DVec| {
                seen.push(v.clone());
                let mut y = DVec(v.0.iter().zip(diag).map(|(a, d)| a * d).collect());
                if w > 0.0 && mode == SsMode::Penalty {
                    for p in ps {
                        let c = p.dot(v);
                        y.axpy(w * c, p);
                    }
                }
                y
            };
            let mut prec = |_t: &mut DVec, _th: f64| {};
            ss_local_core(
                &mut op, &mut prec, cp0, ps, w, mode, roots, 1e-12, ms, rk, rngr,
            )
        };
        (out, seen)
    }

    #[test]
    fn ss_local_core_leaves_a_healthy_guess_alone() {
        let n = 4;
        let ps = vec![unit(n, 0)];
        let cp0 = DVec(vec![0.1, 0.9, 0.0, 0.0]);
        let mut rngr = Xoshiro256PlusPlus::seed_from_u64(7);
        let before = format!("{rngr:?}");
        let (out, seen) = run_core(
            &[1.0, 2.0, 3.0, 4.0],
            &cp0,
            &ps,
            1.0,
            SsMode::Penalty,
            1,
            &mut rngr,
        );
        assert_eq!(out.skipped, 0);
        assert!(
            cosine(&seen[0], &cp0) > 1.0 - 1e-14,
            "branch (iii) must hand the solver cp0 itself (cos = {})",
            cosine(&seen[0], &cp0)
        );
        assert_eq!(
            before,
            format!("{rngr:?}"),
            "branch (iii) must not advance the roots stream"
        );
    }

    #[test]
    fn ss_local_core_rescales_a_dominated_guess_to_its_complement() {
        let n = 4;
        let ps = vec![unit(n, 0)];
        let cp0 = DVec(vec![1.0, 0.2, 0.0, 0.0]);
        let mut rngr = Xoshiro256PlusPlus::seed_from_u64(7);
        let (out, seen) = run_core(
            &[1.0, 2.0, 3.0, 4.0],
            &cp0,
            &ps,
            1.0,
            SsMode::Penalty,
            1,
            &mut rngr,
        );
        assert_eq!(out.skipped, 0);
        let e1 = unit(n, 1);
        assert!(
            cosine(&seen[0], &e1) > 1.0 - 1e-14,
            "branch (ii) must hand the solver the COMPLEMENT (cos vs e₁ = {}, vs cp0 = {})",
            cosine(&seen[0], &e1),
            cosine(&seen[0], &cp0)
        );
    }

    #[test]
    fn ss_local_core_redraws_an_annihilated_guess_from_the_roots_stream() {
        let n = 4;
        let ps = vec![unit(n, 0)];
        let cp0 = unit(n, 0);
        let mut rngr = Xoshiro256PlusPlus::seed_from_u64(7);
        let before = format!("{rngr:?}");
        let (out, seen) = run_core(
            &[1.0, 2.0, 3.0, 4.0],
            &cp0,
            &ps,
            1.0,
            SsMode::Penalty,
            1,
            &mut rngr,
        );
        assert_eq!(out.skipped, 0, "a live 3-dim complement is not a skip");
        assert_ne!(
            before,
            format!("{rngr:?}"),
            "branch (i) must draw from the roots stream"
        );
        assert!(
            seen[0].0[0].abs() <= 1e-14,
            "the redrawn guess must be projected off span{{qⱼ}} (component {})",
            seen[0].0[0]
        );
    }

    #[test]
    fn ss_local_core_skips_when_the_projected_complement_cannot_host_the_roots() {
        let n = 3;
        let ps = vec![unit(n, 0), unit(n, 1)];
        let cp0 = DVec(vec![0.3, 0.4, 0.5]);
        let mut rngr = Xoshiro256PlusPlus::seed_from_u64(11);
        let (out, seen) = run_core(
            &[1.0, 2.0, 3.0],
            &cp0,
            &ps,
            0.0,
            SsMode::Project,
            2,
            &mut rngr,
        );
        assert_eq!(
            out.skipped, 1,
            "Project roots = 2 on a 3-dim space must skip"
        );
        assert_eq!(out.numops, 1, "a skipped bond pays exactly one apply");
        assert_eq!(out.swap, 0);
        assert!(
            (out.u.norm() - 1.0).abs() <= 1e-14,
            "the skipped bond returns the normalized incoming block"
        );
        assert_eq!(seen.len(), 1, "no solve on a skipped bond");
    }

    #[test]
    fn ss_local_core_penalty_does_not_count_the_blocked_directions() {
        let n = 3;
        let ps = vec![unit(n, 0), unit(n, 1)];
        let cp0 = DVec(vec![0.3, 0.4, 0.5]);
        let mut rngr = Xoshiro256PlusPlus::seed_from_u64(11);
        let (out, seen) = run_core(
            &[1.0, 2.0, 3.0],
            &cp0,
            &ps,
            1.0,
            SsMode::Penalty,
            2,
            &mut rngr,
        );
        assert_eq!(out.skipped, 0, "Penalty roots = 2 on a 3-dim space solves");
        assert!(out.numops >= 1);
        assert!(seen.len() > 1, "the block solver ran");
    }

    #[test]
    fn ss_local_core_penalized_root_matches_the_dense_oracle() {
        let n = 4;
        let ps = vec![unit(n, 0)];
        let cp0 = DVec(vec![0.2, 0.5, 0.7, 0.3]);
        let mut rngr = Xoshiro256PlusPlus::seed_from_u64(3);
        let (out, _) = run_core(
            &[1.0, 2.0, 3.0, 4.0],
            &cp0,
            &ps,
            10.0,
            SsMode::Penalty,
            1,
            &mut rngr,
        );
        assert!(
            (out.theta - 2.0).abs() <= 1e-9,
            "penalized Ritz value {} vs 2",
            out.theta
        );
        assert!(
            (out.lambda - 2.0).abs() <= 1e-9,
            "the plain energy at a root orthogonal to ψ equals θ ({})",
            out.lambda
        );
    }

    #[test]
    fn ss_local_core_zero_weight_keeps_lambda_equal_to_theta() {
        let n = 4;
        let ps = vec![unit(n, 0)];
        let cp0 = DVec(vec![0.2, 0.5, 0.7, 0.3]);
        let mut rngr = Xoshiro256PlusPlus::seed_from_u64(5);
        let (out, _) = run_core(
            &[1.0, 2.0, 3.0, 4.0],
            &cp0,
            &ps,
            0.0,
            SsMode::Penalty,
            1,
            &mut rngr,
        );
        assert_eq!(out.lambda.to_bits(), out.theta.to_bits());
    }

    #[test]
    fn local_basis_drops_a_vanishing_image_on_the_absolute_floor() {
        let n = 3;
        let live = unit(n, 0);
        let mut tiny = unit(n, 1);
        tiny.scale(1e-16);
        let qs = local_basis(&[live.clone(), tiny]);
        assert_eq!(qs.len(), 1, "only the live image spans");
        assert!(cosine(&qs[0], &live) > 1.0 - 1e-12);
    }

    #[test]
    fn delegate_sweeps_keys_convergence_on_conv_tol_not_eps() {
        let opts = MalsOptions {
            eps: 1e-8,
            conv_tol: Some(1e-12),
            bond_dims: vec![0],
            noise: vec![0.0],
            max_iter: 4,
            ..Default::default()
        };
        let mk = |energy: f64| MalsSweep {
            energy,
            discarded_weight: 0.0,
            max_bond_dim: 4,
            bond_dim: 0,
            noise: 0.0,
        };
        let e0 = -1.0f64;
        let e1 = e0 * (1.0 + 1e-7);
        let e2 = e1 * (1.0 + 1e-10);
        let e3 = e2 * (1.0 + 1e-14);
        let hist = MalsHistory {
            lambda: e3,
            sweeps: vec![mk(e1), mk(e2), mk(e3)],
        };
        let r = |a: f64, b: f64| (a - b).abs() / b.abs();
        assert!(r(e1, e2) < opts.eps && r(e1, e2) >= opts.conv_tol.unwrap());
        let sweeps = delegate_sweeps(&hist, e0, &opts);
        assert!(
            !sweeps[0].converged,
            "iteration 0 is far from any tolerance"
        );
        assert!(
            !sweeps[1].converged,
            "an iteration with r = {:.1e} in [conv_tol, eps) is NOT converged",
            r(e1, e2)
        );
        assert!(
            sweeps[2].converged,
            "r = {:.1e} < conv_tol converges (inner_tol has tightened by then)",
            r(e2, e3)
        );
        let loose = MalsOptions {
            conv_tol: None,
            ..opts.clone()
        };
        let sl = delegate_sweeps(&hist, e0, &loose);
        assert!(
            sl[1].converged,
            "at conv_tol = eps the middle iteration DOES converge — the discriminator"
        );
        let noisy = MalsOptions {
            noise: vec![1e-4],
            ..opts.clone()
        };
        let mut hn = hist.clone();
        for s in hn.sweeps.iter_mut() {
            s.noise = 1e-4;
        }
        let sn = delegate_sweeps(&hn, e0, &noisy);
        assert!(
            sn.iter().all(|s| !s.converged),
            "a noisy iteration never records the stop test firing"
        );
    }
}
