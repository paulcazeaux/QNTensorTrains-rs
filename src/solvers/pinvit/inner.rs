// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::hamiltonian::{SparseHamiltonian, xt_h_y};
use crate::ops::dot;
use crate::qn::Pooled;
use crate::vector::TtVector;

use super::precond::ExpSumPrecond;
use super::{PinvitConsts, PinvitTermination, TruncBackend, bounds};

/// Which algorithm runs, and the tolerances that algorithm — and only that algorithm — has.
#[derive(Clone, Copy, Debug)]
pub enum PinvitMethod {
    /// Paper Algorithm 2, at [`crate::round::RoundMode::Abs`] throughout. The certified reference
    /// path, and the one §5.6's quasi-optimality argument applies to.
    Det(PinvitDetOpts),
}

impl Default for PinvitMethod {
    fn default() -> Self {
        PinvitMethod::Det(PinvitDetOpts::default())
    }
}

impl PinvitMethod {
    /// The rounding kernel this method drives.
    pub fn backend(&self) -> TruncBackend {
        match self {
            PinvitMethod::Det(_) => TruncBackend::DeterministicSvd,
        }
    }

    /// `t_eig`, which both paths share (§5.3.2 is independent of how the rounding is done).
    pub fn t_eig(&self) -> f64 {
        match self {
            PinvitMethod::Det(o) => o.t_eig,
        }
    }
}

/// Tolerances for [`PinvitMethod::Det`] — paper Algorithm 2, absolute throughout.
#[derive(Clone, Copy, Debug)]
pub struct PinvitDetOpts {
    /// `ε_num ∈ (0, 1−ε_iter)` of Theorem 5.1; the residual tolerance is
    /// `ζ_res = (1+c)^{-1}ε_num` when `c` is known, else `ε_num` directly. Paper's numerics use
    /// `ε_num = ½(1−ε_iter)` with `ε_iter + ε_num ∈ (½, 1)`.
    pub eps_num: f64,
    /// §5.3.2 aggressiveness; see `PinvitRandOpts::t_eig`. `0` disables it, leaving
    /// [`Self::eta_inner`] in force as a fixed absolute tolerance.
    pub t_eig: f64,
    /// Absolute rounding tolerance for each `S`-apply.
    pub eta_precond: f64,
    /// Absolute FLOOR on the residual-assembly tolerance, and on the iterate tolerance under
    /// `t_eig > 0`.
    pub eta_resid: f64,
    /// Absolute iterate tolerance when `t_eig == 0`; a floor otherwise.
    pub eta_inner: f64,
    /// Cap on the retries of the §5.3.1 accept-or-shrink loop per iteration.
    pub max_retry: usize,
    /// Apply the `J₀` drop of §5.3.1: sort the summands by norm and omit the smallest whose norms
    /// sum to `≤ η_res/3`, spending the freed budget on the terms that carry the residual.
    pub drop_small_terms: bool,
}

impl Default for PinvitDetOpts {
    fn default() -> Self {
        PinvitDetOpts {
            eps_num: 0.1,
            t_eig: 0.5,
            eta_precond: 1e-10,
            eta_resid: 1e-12,
            eta_inner: 1e-8,
            max_retry: 48,
            drop_small_terms: true,
        }
    }
}

/// Inner-iteration options: what both paths share, plus which path runs.
#[derive(Clone, Copy, Debug)]
pub struct PinvitOpts {
    /// Problem constants (`γ`, gap `δ`, quality `c`).
    pub consts: PinvitConsts,
    /// Which algorithm, and its tolerances.
    pub method: PinvitMethod,
    /// Pooled cap on the TOTAL bond dimension of every rounded intermediate (the conventional DMRG
    /// `M`); [`Pooled::NONE`] = uncapped, the default.
    pub chi: Pooled,
    /// Termination target. Under [`PinvitTermination::Residual`] this is an ABSOLUTE residual in
    /// Hartree, so `1e-3` is milli-Hartree — roughly chemical accuracy. Under the two paper bounds
    /// it is instead a RELATIVE error on `λ(H_γ) = γ + E`, which rescales with `γ`; the same number
    /// then means different accuracies on different systems.
    pub tau: f64,
    /// Which certificate `tau` is tested against. See the enum's docs before changing it: this is a
    /// correctness choice, not a speed knob.
    pub termination: PinvitTermination,
    /// Iteration cap.
    pub max_iter: usize,
    /// Record every rounding call into [`PinvitInfo::round_calls`]. Off by default: it costs two
    /// `norm()` contractions per call.
    pub record_rounding: bool,
    /// Emit per-iteration diagnostics to stderr.
    pub verbose: bool,
}

impl Default for PinvitOpts {
    fn default() -> Self {
        PinvitOpts {
            consts: PinvitConsts::default(),
            method: PinvitMethod::default(),
            chi: Pooled::NONE,
            tau: 1e-6,
            termination: PinvitTermination::default(),
            max_iter: 100,
            record_rounding: false,
            verbose: false,
        }
    }
}

/// Wall-clock seconds per stage, accumulated over the whole solve.
#[derive(Clone, Copy, Debug, Default)]
pub struct StageTimes {
    /// `y = S·x`.
    pub s_apply_x: f64,
    /// `w = (H_γ−λ)·y` — the Hamiltonian-carrying stage.
    pub h_shift: f64,
    /// `r = S·w`.
    pub s_apply_w: f64,
    /// `z = S·r`, consumed only by the 2×2 pencil.
    pub s_apply_r: f64,
    /// `x ← z₁x + z₂r`.
    pub iterate: f64,
    /// `⟨H_γ y, y⟩` for the Rayleigh quotient, at `rank(y) = rank(Sx)`.
    pub bilinear_rayleigh: f64,
    /// The 2×2 pencil's `⟨H_γ y, z⟩` and `⟨H_γ z, z⟩`, at `rank(z) = rank(S·r)` — whose rank is set
    /// by [`PinvitRandOpts::theta_z`].
    pub bilinear_pencil: f64,
    /// Everything else in the loop body (norms, bounds, bookkeeping).
    pub other: f64,
}

/// One rounding call inside the inner loop, as it was actually issued.
#[derive(Clone, Copy, Debug)]
pub struct RoundCall {
    /// Which PINVIT iteration issued it.
    pub iter: usize,
    /// Which of the five call sites, in loop order.
    pub site: &'static str,
    /// The tolerance passed to the backend.
    pub tol: f64,
    /// Whether this CALL asked for a relative budget or an absolute one — a property of the call
    /// site, not of the backend.
    pub tol_relative: bool,
    /// `tol·‖out‖` when the call was relative, else `tol`. The comparable quantity across paths.
    pub effective_abs: f64,
    /// Euclidean norm of the input, and of the rounded output.
    pub norm_in: f64,
    /// Euclidean norm of the rounded output.
    pub norm_out: f64,
    /// Max total (pooled over sectors) bond rank in, and out.
    pub rank_in: usize,
    /// Max total bond rank of the rounded output.
    pub rank_out: usize,
}

/// Per-iteration diagnostics (the A/B levers: `lambdas`, `rhos`, `max_ranks`).
#[derive(Clone, Debug, Default)]
pub struct PinvitInfo {
    /// Number of iterations performed.
    pub iters: usize,
    /// Rayleigh quotient `λⁿ = λ(xⁿ)` per iteration (the `H_γ` eigenvalue estimate).
    pub lambdas: Vec<f64>,
    /// Residual bound `ρⁿ` per iteration (paper eq 5.8).
    pub rhos: Vec<f64>,
    /// The eigenVECTOR-error bound (Thm 5.7) per iteration; `∞` while the gap condition fails.
    /// Recorded whether or not it is the active criterion — the two are the main
    /// diagnostic pair, and recomputing them after the fact needs the whole `rho` trace.
    pub bound_eigenvector: Vec<f64>,
    /// The relative eigenVALUE-error bound (eq 5.18) per iteration; `∞` while the gap condition
    /// fails. Always `≤ bound_eigenvector`.
    pub bound_eigenvalue: Vec<f64>,
    /// Extrapolated estimate of the remaining eigenvalue error `λ_n − λ₁` in Hartree, per
    /// iteration; `∞` until three iterates exist and the differences are monotone with a ratio in
    /// `(0,1)`. An ESTIMATE, not a bound — see [`super::PinvitTermination::Extrapolated`].
    pub err_extrapolated: Vec<f64>,
    /// The residual-assembly tolerance actually used at each iteration. Under `eta_rel > 0` this
    /// should start loose and fall with the residual — if it sits at the floor from the first
    /// iteration, the adaptation is not doing anything.
    pub eta_resid_used: Vec<f64>,
    /// The ITERATE-truncation tolerance actually used at each iteration (paper §5.3.2). Constant at
    /// `PinvitOpts::eta_inner` when `t_eig = 0`. One shorter than `lambdas`: the final iteration
    /// returns without an update.
    pub eta_inner_used: Vec<f64>,
    /// `λ*`, the Rayleigh–Ritz minimum over `span{x, r}`, per iteration. One shorter than
    /// `lambdas`: the final iteration returns without taking a step.
    pub lam_star: Vec<f64>,
    /// Wall-clock per stage over the whole solve; see [`StageTimes`].
    pub stage_times: StageTimes,
    /// Every rounding call, in issue order — empty unless [`PinvitOpts::record_rounding`] is set.
    /// Five per iteration; see [`RoundCall`]. Costs two norms per call, so it is opt-in.
    pub round_calls: Vec<RoundCall>,
    /// `‖H_γ y − λ y‖ / ‖y‖` with `y = S x`, per iteration, in HARTREE.
    pub resid_hartree: Vec<f64>,
    /// Max total bond rank of the iterate `xⁿ` per iteration.
    pub max_ranks: Vec<usize>,
    /// The smallest Rayleigh quotient of `A = S H_γ S` seen on any iterate,
    /// `min_n ⟨A xⁿ, xⁿ⟩ / ⟨xⁿ, xⁿ⟩`. Free — both inner products are already computed.
    pub min_rayleigh_a: f64,
    /// The largest Rayleigh quotient of `A` seen, the upper companion of [`Self::min_rayleigh_a`].
    /// `‖I − A‖₂ ≤ c` requires `spec(A) ⊆ [1−c, 1+c]`, so this witnesses the other side of the band.
    pub max_rayleigh_a: f64,
    /// `true` when a Rayleigh quotient of `A` left the band `[1−c, 1+c]` on either side, i.e. the
    /// run produced a proof that the declared `c` is unsound. The reported `λ` is unaffected — the PINVIT fixed point does
    /// not depend on `c` — but `rhos`, both bounds, and the termination point are all optimistic.
    /// Fix by deriving `c` with [`super::precond::ExpSumPrecond::calibrate`].
    pub c_understated: bool,
    /// Whether the a-posteriori bound reached `tau`.
    pub converged: bool,
}

fn h_gamma_bilinear(
    a: &TtVector<f64>,
    h: &SparseHamiltonian<f64>,
    b: &TtVector<f64>,
    gamma: f64,
) -> f64 {
    xt_h_y(a, h, b) + gamma * dot(a, b)
}

fn solve_2x2_gsym_smallest(a: [[f64; 2]; 2], e: [[f64; 2]; 2]) -> (f64, [f64; 2]) {
    let l11 = e[0][0].max(0.0).sqrt();
    let l21 = e[0][1] / l11;
    let l22 = (e[1][1] - l21 * l21).max(0.0).sqrt();
    let li = [[1.0 / l11, 0.0], [-l21 / (l11 * l22), 1.0 / l22]];
    let m = [
        [
            li[0][0] * a[0][0] + li[0][1] * a[1][0],
            li[0][0] * a[0][1] + li[0][1] * a[1][1],
        ],
        [
            li[1][0] * a[0][0] + li[1][1] * a[1][0],
            li[1][0] * a[0][1] + li[1][1] * a[1][1],
        ],
    ];
    let lit = [[li[0][0], li[1][0]], [li[0][1], li[1][1]]];
    let c00 = m[0][0] * lit[0][0] + m[0][1] * lit[1][0];
    let c01 = m[0][0] * lit[0][1] + m[0][1] * lit[1][1];
    let c11 = m[1][0] * lit[0][1] + m[1][1] * lit[1][1];
    let tr = c00 + c11;
    let det = c00 * c11 - c01 * c01;
    let disc = (tr * tr - 4.0 * det).max(0.0).sqrt();
    let lam = 0.5 * (tr - disc);
    let mut w = if c01.abs() > 1e-300 {
        [c01, lam - c00]
    } else if (c00 - lam).abs() <= (c11 - lam).abs() {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let wn = (w[0] * w[0] + w[1] * w[1]).sqrt();
    if wn > 0.0 {
        w[0] /= wn;
        w[1] /= wn;
    }
    let v = [
        lit[0][0] * w[0] + lit[0][1] * w[1],
        lit[1][0] * w[0] + lit[1][1] * w[1],
    ];
    (lam, v)
}

fn extrapolated_error(lams: &[f64]) -> f64 {
    const WINDOW: usize = 3;
    const SPREAD: f64 = 1.3;
    let n = lams.len();
    if n < WINDOW + 2 {
        return f64::INFINITY;
    }
    let mut rs = Vec::with_capacity(WINDOW);
    for k in 0..WINDOW {
        let i = n - 1 - k;
        let d_last = lams[i - 1] - lams[i];
        let d_prev = lams[i - 2] - lams[i - 1];
        if d_prev <= 0.0 || d_last <= 0.0 {
            return f64::INFINITY;
        }
        let r = d_last / d_prev;
        if !(r > 0.0 && r < 1.0) {
            return f64::INFINITY;
        }
        rs.push(r);
    }
    let (lo, hi) = (
        rs.iter().cloned().fold(f64::MAX, f64::min),
        rs.iter().cloned().fold(0.0f64, f64::max),
    );
    if hi / lo > SPREAD {
        return f64::INFINITY;
    }
    let d_last = lams[n - 2] - lams[n - 1];
    d_last * hi / (1.0 - hi)
}

struct Shared<'a> {
    h: &'a SparseHamiltonian<f64>,
    precond: &'a ExpSumPrecond,
    back: TruncBackend,
    gamma: f64,
    delta: f64,
    c: f64,
    c_known: bool,
    inv_sqrt_1mc: f64,
    chi: Pooled,
    tau: f64,
    termination: PinvitTermination,
    max_iter: usize,
    t_eig: f64,
    record: bool,
    verbose: bool,
}

#[allow(clippy::too_many_arguments)]
fn rec_call(
    sh: &Shared,
    calls: &mut Vec<RoundCall>,
    iter: usize,
    site: &'static str,
    tol: f64,
    tol_relative: bool,
    inp: &TtVector<f64>,
    out: &TtVector<f64>,
) {
    if !sh.record {
        return;
    }
    let norm_out = out.norm();
    calls.push(RoundCall {
        iter,
        site,
        tol,
        tol_relative,
        effective_abs: if tol_relative { tol * norm_out } else { tol },
        norm_in: inp.norm(),
        norm_out,
        rank_in: inp.max_total_bond_rank(),
        rank_out: out.max_total_bond_rank(),
    });
}

fn delta_a(t_eig: f64, lambda: f64, lam_star: f64, rho: f64, floor: f64) -> f64 {
    let gap = (lambda - lam_star).max(0.0);
    let q2 = if lam_star > 0.0 {
        (lam_star / (lam_star + t_eig * gap)).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let disc = (1.0 + rho * rho - q2).max(0.0);
    let d = ((-q2.sqrt() * rho + disc.sqrt()) / (1.0 + rho * rho)).clamp(0.0, 1.0);
    d.max(floor)
}

fn finish_iteration(
    sh: &Shared,
    info: &mut PinvitInfo,
    it: usize,
    lambda: f64,
    rho: f64,
    resid_h: f64,
    rank: usize,
) -> bool {
    let err_ext = extrapolated_error(&info.lambdas);
    info.err_extrapolated.push(err_ext);
    info.rhos.push(rho);
    info.resid_hartree.push(resid_h);

    let (b_vec, b_val) = if sh.c_known {
        (
            bounds::eigenvector(rho, sh.delta),
            bounds::eigenvalue(rho, sh.delta),
        )
    } else {
        (f64::NAN, f64::NAN)
    };
    info.bound_eigenvector.push(b_vec);
    info.bound_eigenvalue.push(b_val);
    let term = match sh.termination {
        PinvitTermination::Residual => resid_h,
        PinvitTermination::Extrapolated => err_ext,
        PinvitTermination::Eigenvalue => b_val,
        PinvitTermination::Eigenvector => b_vec,
    };
    if sh.verbose {
        eprintln!(
            "pinvit it={it} λ={lambda:.10} ρ={rho:.3e} term={term:.3e} \
             (vec {b_vec:.3e} / val {b_val:.3e}) rank={rank} resid={resid_h:.3e}"
        );
    }
    info.iters = it + 1;
    term <= sh.tau || rho < 1e-13
}

fn residual_summands(
    precond: &ExpSumPrecond,
    h: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    shift: f64,
) -> (Vec<f64>, Vec<TtVector<f64>>, f64) {
    let m = precond.num_terms();
    let balance = |t: &mut TtVector<f64>| -> f64 {
        let mut scale = 1.0f64;
        for core in t.cores.iter_mut() {
            let n = core.norm();
            if n > 0.0 && n.is_finite() {
                core.scale_mut(n.recip());
                scale *= n;
            }
        }
        scale
    };

    let mut v: Vec<(TtVector<f64>, f64)> = Vec::with_capacity(m);
    for kp in 0..m {
        let mut u = precond.apply_summand(kp, x);
        let scale_kp = balance(&mut u);
        let hu = h.apply(&u);
        v.push((&hu + &(&u * shift), scale_kp));
    }
    let mut w2 = 0.0f64;
    for (k, (vk, sk)) in v.iter().enumerate() {
        for (kp, (vkp, skp)) in v.iter().enumerate() {
            w2 += precond.alphas[k] * precond.alphas[kp] * sk * skp * dot(vk, vkp);
        }
    }
    let w_norm = w2.max(0.0).sqrt();

    let mut weights = Vec::with_capacity(m * m);
    let mut summands = Vec::with_capacity(m * m);
    for (kp, (vk, scale_kp)) in v.iter().enumerate() {
        for k in 0..m {
            let mut s = precond.apply_summand(k, vk);
            let scale_k = balance(&mut s);
            weights.push(precond.alphas[k] * precond.alphas[kp] * scale_k * scale_kp);
            summands.push(s);
        }
    }
    (weights, summands, w_norm)
}

/// Inner PINVIT. Returns the shifted electronic eigenvalue `λ(H_γ)`, the preconditioned-coordinate
/// iterate `x` (A-normalized; the Hamiltonian eigenvector is `y = S·x`, obtained via
/// [`ExpSumPrecond::hamiltonian_vector`]), and per-iteration diagnostics.
pub fn pinvit_ground_state(
    h: &SparseHamiltonian<f64>,
    precond: &ExpSumPrecond,
    x0: &TtVector<f64>,
    opts: &PinvitOpts,
) -> (f64, TtVector<f64>, PinvitInfo) {
    let c_known = opts.consts.c.is_some();
    let c = if c_known {
        opts.consts.resolve_c()
    } else {
        assert!(
            matches!(
                opts.termination,
                PinvitTermination::Residual | PinvitTermination::Extrapolated
            ),
            "PinvitConsts::c is unset, which is only allowed under \
             PinvitTermination::Residual or ::Extrapolated (the paper's bounds are functions of \
             c). Either measure it with `ExpSumPrecond::calibrate` or use one of those criteria."
        );
        assert!(
            opts.consts.delta > 1.0,
            "PinvitConsts::delta must exceed 1, got {}",
            opts.consts.delta
        );
        0.0
    };
    let sh = Shared {
        h,
        precond,
        back: opts.method.backend(),
        gamma: opts.consts.gamma,
        delta: opts.consts.delta,
        c,
        c_known,
        inv_sqrt_1mc: (1.0 - c).max(1e-30).powf(-0.5),
        chi: opts.chi,
        tau: opts.tau,
        termination: opts.termination,
        max_iter: opts.max_iter,
        t_eig: opts.method.t_eig(),
        record: opts.record_rounding,
        verbose: opts.verbose,
    };
    match opts.method {
        PinvitMethod::Det(do_) => inner_det(&sh, x0, &do_),
    }
}

fn inner_det(
    sh: &Shared,
    x0: &TtVector<f64>,
    dopts: &PinvitDetOpts,
) -> (f64, TtVector<f64>, PinvitInfo) {
    let mut x = x0.clone();
    let mut info = PinvitInfo {
        min_rayleigh_a: f64::INFINITY,
        max_rayleigh_a: f64::NEG_INFINITY,
        ..PinvitInfo::default()
    };
    let mut last_lambda = f64::NAN;
    let zeta_res = if sh.c_known {
        dopts.eps_num / (1.0 + sh.c)
    } else {
        dopts.eps_num
    };
    let mut eta_carry = f64::INFINITY;

    for it in 0..sh.max_iter {
        let mut y = sh
            .precond
            .apply_abs(&x, dopts.eta_precond, sh.chi, &sh.back);
        rec_call(
            sh,
            &mut info.round_calls,
            it,
            "1:S·x",
            dopts.eta_precond,
            false,
            &x,
            &y,
        );

        let x_a2 = h_gamma_bilinear(&y, sh.h, &y, sh.gamma);
        let yy = dot(&y, &y);
        assert!(
            x_a2 > 0.0,
            "H_γ is not positive definite at the current iterate (⟨A x, x⟩ = {x_a2:.3e}); \
             increase the elliptic shift γ (see choose_gamma)"
        );
        let lambda = x_a2 / yy;
        info.lambdas.push(lambda);
        info.max_ranks.push(x.max_total_bond_rank());
        witness_c(sh, &mut info, &x, x_a2);

        let scale = x_a2.sqrt().recip();
        x.scale_mut(scale);
        y.scale_mut(scale);

        let (wts, sums, w_norm) = residual_summands(sh.precond, sh.h, &x, sh.gamma - lambda);
        let mut idx: Vec<usize> = (0..sums.len()).collect();
        let norms: Vec<f64> = sums
            .iter()
            .zip(&wts)
            .map(|(s, w)| w.abs() * s.norm())
            .collect();
        idx.sort_by(|&a, &b| norms[a].partial_cmp(&norms[b]).expect("finite norms"));
        let total: f64 = norms.iter().sum();
        let mut eta_r = (2.0 * eta_carry).min(0.8 * zeta_res * total);

        let mut r;
        let mut tries = 0usize;
        loop {
            let mut dropped = 0.0f64;
            let mut keep: Vec<usize> = Vec::with_capacity(idx.len());
            for &j in &idx {
                if dopts.drop_small_terms && dropped + norms[j] <= eta_r / 3.0 {
                    dropped += norms[j];
                    continue;
                }
                keep.push(j);
            }
            if keep.is_empty() {
                keep.push(*idx.last().expect("at least one summand"));
            }
            let kept_total: f64 = keep.iter().map(|&j| norms[j]).sum();
            let kw: Vec<f64> = keep.iter().map(|&j| wts[j]).collect();
            let ks: Vec<TtVector<f64>> = keep.iter().map(|&j| sums[j].clone()).collect();
            let tol_sum = if kept_total > 0.0 { eta_r / 3.0 } else { eta_r };
            r = sh
                .back
                .round_weighted_sum_abs(&kw, &ks, tol_sum, sh.chi, &x);
            if eta_r <= dopts.eta_resid || tries >= dopts.max_retry {
                break;
            }
            if r.norm() - eta_r >= eta_r / zeta_res {
                break;
            }
            eta_r = (0.8 * eta_r).max(dopts.eta_resid);
            tries += 1;
        }
        eta_carry = eta_r;
        info.eta_resid_used.push(eta_r);
        rec_call(
            sh,
            &mut info.round_calls,
            it,
            "2+3:Σ s_j",
            eta_r,
            false,
            &y,
            &r,
        );

        let y_norm = y.norm();
        let resid_h = if y_norm > 0.0 {
            w_norm / y_norm
        } else {
            f64::INFINITY
        };
        let rho = sh.inv_sqrt_1mc * r.norm();

        let rank = x.max_total_bond_rank();
        let done = finish_iteration(sh, &mut info, it, lambda, rho, resid_h, rank);
        last_lambda = lambda;
        if done {
            info.converged = true;
            break;
        }
        if it + 1 == sh.max_iter {
            break;
        }

        let z = sh
            .precond
            .apply_abs(&r, dopts.eta_precond, sh.chi, &sh.back);
        rec_call(
            sh,
            &mut info.round_calls,
            it,
            "4:S·r",
            dopts.eta_precond,
            false,
            &r,
            &z,
        );
        let (lam_star, v) = ritz_step(sh, &y, &z, None);
        info.lam_star.push(lam_star);

        let eta_in = if sh.t_eig > 0.0 {
            let a_xr = h_gamma_bilinear(&y, sh.h, &z, sh.gamma);
            let a_rr = h_gamma_bilinear(&z, sh.h, &z, sh.gamma);
            let x_star_a = (v[0] * v[0] + 2.0 * v[0] * v[1] * a_xr + v[1] * v[1] * a_rr)
                .max(0.0)
                .sqrt();
            let scale = if sh.c_known { (1.0 + sh.c).sqrt() } else { 1.0 };
            (delta_a(sh.t_eig, lambda, lam_star, rho, 0.0) * x_star_a / scale).max(dopts.eta_inner)
        } else {
            dopts.eta_inner
        };
        info.eta_inner_used.push(eta_in);

        let x_prev = x.clone();
        x = sh
            .back
            .round_weighted_sum_abs(&[v[0], v[1]], &[x.clone(), r], eta_in, sh.chi, x0);
        rec_call(
            sh,
            &mut info.round_calls,
            it,
            "5:z₁x+z₂r",
            eta_in,
            false,
            &x_prev,
            &x,
        );
    }
    (last_lambda, x, info)
}

fn ritz_step(
    sh: &Shared,
    y: &TtVector<f64>,
    z: &TtVector<f64>,
    xr_identity: Option<(f64, f64)>,
) -> (f64, [f64; 2]) {
    let yz = dot(y, z);
    let a_xr = match xr_identity {
        Some((lambda, rr)) => rr + lambda * yz,
        None => h_gamma_bilinear(y, sh.h, z, sh.gamma),
    };
    let a_rr = h_gamma_bilinear(z, sh.h, z, sh.gamma);
    let a2 = [[1.0, a_xr], [a_xr, a_rr]];
    let e2 = [[dot(y, y), yz], [yz, dot(z, z)]];
    solve_2x2_gsym_smallest(a2, e2)
}

fn witness_c(sh: &Shared, info: &mut PinvitInfo, x: &TtVector<f64>, x_a2: f64) {
    let rq_a = x_a2 / dot(x, x);
    info.min_rayleigh_a = info.min_rayleigh_a.min(rq_a);
    info.max_rayleigh_a = info.max_rayleigh_a.max(rq_a);
    if sh.c_known && (rq_a < 1.0 - sh.c || rq_a > 1.0 + sh.c) {
        info.c_understated = true;
    }
}

#[cfg(test)]
mod tests {
    use super::solve_2x2_gsym_smallest;

    #[test]
    fn tpvi_2x2_diagonal_pairs_lambda_with_its_own_axis() {
        for (a00, a11, want_lam, want_axis) in [
            (3.0, 1.0, 1.0, 1usize),
            (1.0, 3.0, 1.0, 0usize),
            (2.0, 2.0, 2.0, 0usize),
        ] {
            let a = [[a00, 0.0], [0.0, a11]];
            let e = [[1.0, 0.0], [0.0, 1.0]];
            let (lam, v) = solve_2x2_gsym_smallest(a, e);
            assert!((lam - want_lam).abs() < 1e-12, "lam {lam} != {want_lam}");
            let num = v[0] * v[0] * a00 + v[1] * v[1] * a11;
            let den = v[0] * v[0] + v[1] * v[1];
            assert!(
                (num / den - lam).abs() < 1e-12,
                "A = diag({a00}, {a11}): returned v = {v:?} has Rayleigh quotient {} but lam = {lam}",
                num / den
            );
            assert!(
                v[want_axis].abs() > 0.5,
                "A = diag({a00}, {a11}): expected the eigenvector on axis {want_axis}, got {v:?}"
            );
        }
    }

    #[test]
    fn tpvi_2x2_generic_and_scaled() {
        let a = [[2.0, 0.5], [0.5, 3.0]];
        let e = [[1.0, 0.0], [0.0, 1.0]];
        let (lam, _) = solve_2x2_gsym_smallest(a, e);
        let want = 0.5 * (5.0 - (1.0f64 + 4.0 * 0.25).sqrt());
        assert!((lam - want).abs() < 1e-12, "{lam} != {want}");

        for eps in [1e-3, 1e-6, 1e-9] {
            let a_s = [[2.0, 0.5 * eps], [0.5 * eps, 3.0 * eps * eps]];
            let e_s = [[1.0, 0.0], [0.0, eps * eps]];
            let (lam_s, _) = solve_2x2_gsym_smallest(a_s, e_s);
            assert!(
                (lam_s - want).abs() < 1e-9,
                "eps = {eps:e}: {lam_s} != {want}"
            );
        }
    }
}
