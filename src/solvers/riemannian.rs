// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Riemannian gradient descent / nonlinear CG on the fixed-rank TT manifold, over the
//! tangent-space machinery in [`crate::core::tangent`] (see that module's doc).

use crate::core::tangent::TtTangent;
use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient, xt_h_y};
use crate::ops::dot;
use crate::qn::{Pooled, QnSpace};
use crate::round::RoundMode;
use crate::scalar::Scalar;
use crate::solvers::als::{INNER_TOL_CAP, als_sweep_pair};
use crate::solvers::penalty::Penalty;
use crate::vector::TtVector;

const INITIAL_ROUND_EPS: f64 = 1e-14;

const ARMIJO_C: f64 = 1e-4;
const BACKTRACK_RHO: f64 = 0.5;
const MAX_BACKTRACK: usize = 30;

const ENERGY_NOISE_C: f64 = 32.0;
const GRAD_ACCEPT: f64 = 0.9;
const MAX_GRAD_TRIALS: usize = 3;

const SECANT_MAX_STEPS: usize = 4;
const SECANT_GROWTH: f64 = 20.0;
const SECANT_DONE: f64 = 0.1;

pub(crate) fn floor_secant() -> bool {
    crate::knobs::LS_SECANT.get()
}

pub(crate) const SLOPE_FLOOR_K: f64 = 10.0;
const SLOPE_MAX_BISECT: usize = 8;
pub(crate) const SLOPE_GUARD_K: f64 = 4.0;

pub(crate) fn slope_floor() -> bool {
    crate::knobs::LS_SLOPE_FLOOR.get()
}

pub(crate) fn secant_all() -> bool {
    crate::knobs::LS_SECANT_ALL.get()
}

pub(crate) struct SlopeTrial<K> {
    pub(crate) alpha: f64,
    pub(crate) lambda: f64,
    pub(crate) phi: f64,
    pub(crate) keep: K,
}

pub(crate) fn slope_step<K>(
    slope: f64,
    slopes_only: bool,
    first: SlopeTrial<K>,
    mut trial: impl FnMut(f64) -> Option<SlopeTrial<K>>,
) -> Option<SlopeTrial<K>> {
    let accept = (1.0 - 2.0 * ARMIJO_C) * slope.abs();
    let done = SECANT_DONE * slope.abs();
    let admissible = |t: &SlopeTrial<K>| !slopes_only || t.phi <= accept;
    let mut lo = (0.0f64, slope);
    let mut hi: Option<(f64, f64)> = None;
    let mut prev = (0.0f64, slope);
    let mut cur = (first.alpha, first.phi);
    let mut best: Option<SlopeTrial<K>> = None;
    let fold = |t: SlopeTrial<K>,
                lo: &mut (f64, f64),
                hi: &mut Option<(f64, f64)>,
                best: &mut Option<SlopeTrial<K>>| {
        if t.phi < 0.0 {
            if t.alpha > lo.0 {
                *lo = (t.alpha, t.phi);
            }
        } else if hi.is_none_or(|h| t.alpha < h.0) {
            *hi = Some((t.alpha, t.phi));
        }
        if admissible(&t) && best.as_ref().is_none_or(|b| t.phi.abs() < b.phi.abs()) {
            *best = Some(t);
        }
    };
    fold(first, &mut lo, &mut hi, &mut best);
    for step in 0..SECANT_MAX_STEPS + SLOPE_MAX_BISECT {
        if let Some(b) = &best
            && (b.phi.abs() <= done || step >= SECANT_MAX_STEPS)
        {
            break;
        }
        let a_next = match hi {
            Some((ah, ph)) => {
                let s = lo.0 - lo.1 * (ah - lo.0) / (ph - lo.1);
                if best.is_some() && s.is_finite() && s > lo.0 && s < ah {
                    s
                } else {
                    ls_stats::bump(&ls_stats::SLOPE_BISECTS);
                    0.5 * (lo.0 + ah)
                }
            }
            None => {
                let ((a0, p0), (a1, p1)) = (prev, cur);
                if p1 == p0 {
                    break;
                }
                let s = a1 - p1 * (a1 - a0) / (p1 - p0);
                if !s.is_finite() || s <= a1 {
                    break;
                }
                s.min(SECANT_GROWTH * a1)
            }
        };
        let Some(t) = trial(a_next) else {
            break;
        };
        prev = cur;
        cur = (t.alpha, t.phi);
        fold(t, &mut lo, &mut hi, &mut best);
    }
    best
}

/// Line-search counters (the `ls_floor_ab` bench's instrument), shared by both layers.
/// Process-global and relaxed: read them around a single-threaded driver call.
#[doc(hidden)]
pub mod ls_stats {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    /// Line-search calls.
    pub static CALLS: AtomicUsize = AtomicUsize::new(0);
    /// Retract + energy evaluations (trials).
    pub static TRIALS: AtomicUsize = AtomicUsize::new(0);
    /// Calls that entered the noise/floor regime.
    pub static FLOOR_ENTRIES: AtomicUsize = AtomicUsize::new(0);
    /// Environment gradients computed INSIDE the line search.
    pub static GRAD_EVALS: AtomicUsize = AtomicUsize::new(0);
    /// first trials accepted outright in the noise regime.
    pub static FIRST_ACCEPTS: AtomicUsize = AtomicUsize::new(0);
    /// Calls returning `None`.
    pub static NONE: AtomicUsize = AtomicUsize::new(0);
    /// Secant floor step: extrapolated trials, and calls that returned one.
    pub static SECANT_TRIALS: AtomicUsize = AtomicUsize::new(0);
    pub static SECANT_ACCEPTS: AtomicUsize = AtomicUsize::new(0);
    /// Driver gradients taken from a line search instead of recomputed.
    pub static CACHE_HITS: AtomicUsize = AtomicUsize::new(0);
    /// calls decided by slopes alone; bisection steps; slope-regime trials whose energy
    /// sat more than `SLOPE_GUARD_K` floors above the trapezoid estimate (counted, not refused);
    /// CG's steepest-descent retries of a failed conjugate direction.
    pub static SLOPE_ENTRIES: AtomicUsize = AtomicUsize::new(0);
    pub static SLOPE_BISECTS: AtomicUsize = AtomicUsize::new(0);
    pub static SLOPE_GUARD: AtomicUsize = AtomicUsize::new(0);
    pub static SD_RETRIES: AtomicUsize = AtomicUsize::new(0);

    pub(crate) fn bump(c: &AtomicUsize) {
        c.fetch_add(1, Relaxed);
    }

    /// `[calls, trials, floor_entries, grad_evals, first_accepts, none]`.
    pub fn snapshot() -> [usize; 6] {
        [
            &CALLS,
            &TRIALS,
            &FLOOR_ENTRIES,
            &GRAD_EVALS,
            &FIRST_ACCEPTS,
            &NONE,
        ]
        .map(|c| c.load(Relaxed))
    }

    pub fn reset() {
        for c in [
            &CALLS,
            &TRIALS,
            &FLOOR_ENTRIES,
            &GRAD_EVALS,
            &FIRST_ACCEPTS,
            &NONE,
            &SECANT_TRIALS,
            &SECANT_ACCEPTS,
            &CACHE_HITS,
            &SLOPE_ENTRIES,
            &SLOPE_BISECTS,
            &SLOPE_GUARD,
            &SD_RETRIES,
        ] {
            c.store(0, Relaxed);
        }
    }
}

const ALS_INNER_TOL_C: f64 = 0.02;
const ALS_DELTA_MAX: f64 = 1.0;
const ALS_MAX_REJ: usize = 8;
const ALS_RESTART_RATIO: f64 = 1.0;
const ALS_NOISE_K: f64 = 8.0;

/// Nonlinear-CG update formula.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CgVariant {
    /// Fletcher-Reeves.
    Fr,
    /// Polak-Ribière (with the `max(0, ·)` restart floor) — the default.
    Pr,
}

/// Options for [`riemannian_gd`]/[`riemannian_cg`].
#[derive(Clone, Debug)]
pub struct RiemannianOpts {
    /// Gradient-norm stopping ratio `‖g‖/|λ| < eps`.
    pub eps: f64,
    pub max_iter: usize,
    /// Inert: the gradient is the exact environment contraction, so no residual is
    /// rounded. Kept for API stability; setting it changes nothing.
    pub gradtol: f64,
    /// Base seed of the certification probes (`riemannian_cg_certified`, one derived seed per
    /// probe). The gradient itself is deterministic and does not read it.
    pub seed: u64,
    /// CG only; ignored by [`riemannian_gd`]. Defaults to Polak-Ribière.
    pub variant: CgVariant,
    /// Grid-preservation opt-out of the entry cleanup round.
    pub preserve_grid: bool,
    /// How a CG trial point is produced from the tangent step. `Hosvd` (the default)
    /// is today's behaviour byte for byte. CG only: [`riemannian_gd`] ignores it (the `variant`
    /// precedent). Not on the C ABI (the `preserve_grid` precedent).
    pub retraction: Retraction,
}

/// The retraction of [`riemannian_cg`]'s trial points.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Retraction {
    /// `normalize(round_ranks(x + α·η, r))`, scored by the ambient quartic line model plus Armijo
    /// backtracking ([`linesearch_impl`]).
    #[default]
    Hosvd,
    /// [`Retraction::Hosvd`] followed by `pairs` one-site ALS forward+back sweeps
    /// ([`als_sweep_pair`]) at a gradient-tied inner tolerance, the step chosen by a 1-D trust
    /// radius on the TANGENT norm (`Δ = ‖α·η‖`) with the joint layer's ρ acceptance
    /// (`solvers::joint::tr` §4). Why: on a rank-capped state the truncation of a tangent step
    /// leaves stiff intra-core content that the ambient line model prices at ~10 Ha of curvature;
    /// one block-exact ALS pass removes it and cannot undo the inter-core displacement the step
    /// was for. Measured on the cap-256 typed formaldehyde state: usable step 3e-5 → 1e-3 (unit
    /// tangent norm), gain per step ×32, pair cost ≈ 15 % of a CG iteration. `delta0 = None` seeds
    /// the radius from the ambient model's minimizer at the first step (a certified lower bound of
    /// the usable step); `Some(Δ)` seeds it directly. Undefined for the penalized functional
    /// ([`riemannian_cg_penalized`] refuses it).
    AlsRounded { pairs: usize, delta0: Option<f64> },
}

/// Why a CG run stopped ([`riemannian_cg_report`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CgExit {
    /// `‖g‖ ≤ eps·|λ|`.
    Gradient,
    /// No trial decreased the energy resolvably (HOSVD arm) / no radius or noise-regime trial was
    /// accepted (ALS arm): stationary to the driver's resolution, or a saddle (certify to tell).
    LineSearch,
    /// `max_iter` accepted iterates.
    Budget,
    /// The observer returned `false`.
    Observer,
}

/// [`riemannian_cg`]'s result with its exit reason and iteration count.
#[derive(Clone, Debug)]
pub struct CgReport<T: Scalar> {
    pub lambda: f64,
    pub x: TtVector<T>,
    /// Accepted iterates.
    pub iterations: usize,
    pub gnorm: f64,
    pub exit: CgExit,
}

/// Options of [`riemannian_cg_certified`].
#[derive(Clone, Copy, Debug)]
pub struct CertifyOpts {
    /// Davidson budget of each block probe at a stopping point, counted in APPLIES. At rank 256
    /// (hcho cap-256) budgets ≲ 100 were measured inconclusive and 400 converged under the RETIRED
    /// `8/1/2/1` Davidson; under the shipped block-4 default the same cell converges in 623
    /// applies, so budget ≳ 700 there. Small fixtures need far less: the d = 4 19-dim
    /// block converges exactly inside 90.
    pub probe_budget: usize,
    /// Also probe `A − B` at the final point (reported, never kicked along: an `A − B` witness is
    /// a complex instability, not a real descent direction).
    pub probe_ii: bool,
    /// Seeds tried per block before an `Inconclusive` verdict is accepted (a single-seed
    /// probe can stall on an interior pair; a fresh seed usually resolves it). A negative witness
    /// or a PD verdict from any seed ends the sequence.
    pub probe_seeds: usize,
    /// Krylov tunables of the block probes ([`crate::solvers::hess::ProbeDavidson`]). The default
    /// is the 4-root block on a 48-vector subspace, whose applies run CONCURRENTLY — the setting
    /// measured to find the bottom pair rather than an interior one, at 1.09× the single-vector
    /// wall. `probe_seeds` retries are still worth having: they change WHICH directions
    /// seed the block.
    pub davidson: crate::solvers::hess::ProbeDavidson,
    /// Kicks along an `A + B` witness before the run reports the saddle.
    pub max_kicks: usize,
    /// First kick length in unit tangent norm (measured usable range of the witness ray on the
    /// rank-capped states ≈ 1e-3) and the number of halvings tried below it.
    pub kick_alpha0: f64,
    pub kick_ladder: u32,
}

impl Default for CertifyOpts {
    fn default() -> Self {
        CertifyOpts {
            probe_budget: 400,
            probe_ii: true,
            probe_seeds: 2,
            davidson: crate::solvers::hess::ProbeDavidson::default(),
            max_kicks: 3,
            kick_alpha0: 1e-3,
            kick_ladder: 8,
        }
    }
}

/// One kick of [`riemannian_cg_certified`].
#[derive(Clone, Copy, Debug)]
pub struct KickRecord {
    /// Accepted iterates before the kick.
    pub at_iteration: usize,
    /// The witness's fresh-apply Rayleigh quotient (`< 0`).
    pub witness_lambda: f64,
    /// Applies the probe spent finding it.
    pub probe_applies: usize,
    /// Accepted kick length (unit tangent norm) and its energy change.
    pub alpha: f64,
    pub delta_e: f64,
}

/// The certified polish's verdict about its final point.
#[derive(Clone, Copy, Debug)]
pub enum CertifyVerdict {
    /// `A + B` probed PD at a point where the descent stopped on its own criterion.
    Minimizer { lambda_rr: f64 },
    /// `A + B` has a certified negative direction and no kick along it lowered the energy (or
    /// the kick budget is spent).
    Saddle { witness_lambda: f64 },
    /// The `A + B` probe ran out of budget without a verdict — raise `probe_budget`.
    Inconclusive { lambda_rr: f64, residual: f64 },
    /// The descent ended on `Budget` or `Observer`: nothing was probed.
    NotProbed,
}

/// Result of [`riemannian_cg_certified`].
#[derive(Clone, Debug)]
pub struct CertifyReport<T: Scalar> {
    pub lambda: f64,
    pub x: TtVector<T>,
    /// Accepted iterates over all descent segments.
    pub iterations: usize,
    pub gnorm: f64,
    /// Exit of the last descent segment.
    pub exit: CgExit,
    pub kicks: Vec<KickRecord>,
    /// The final `A + B` / `A − B` probe verdicts (`None` when not probed).
    pub rr: Option<crate::solvers::hess::PdVerdict>,
    pub ii: Option<crate::solvers::hess::PdVerdict>,
    pub verdict: CertifyVerdict,
}

impl RiemannianOpts {
    /// Defaults. `space` is not read; it stays in the signature so the SU(2) twin and every call
    /// site keep one spelling.
    pub fn new(_space: &QnSpace) -> Self {
        RiemannianOpts {
            eps: 1e-4,
            max_iter: 100,
            gradtol: 1e-6,
            seed: 1234,
            variant: CgVariant::Pr,
            preserve_grid: false,
            retraction: Retraction::Hosvd,
        }
    }
}

fn mix_seed(base: u64, k: u64) -> u64 {
    let mut z = base ^ k;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Riemannian steepest descent. Returns `(λ, x)`
/// with `x` unit-norm.
pub fn riemannian_gd<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RiemannianOpts,
) -> (f64, TtVector<T>) {
    gd_impl(h, x0, opts, None)
}

fn gd_impl<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RiemannianOpts,
    penalty: Option<&Penalty<TtVector<T>>>,
) -> (f64, TtVector<T>) {
    let mut x = x0.clone();
    if !opts.preserve_grid {
        x.round_mut(INITIAL_ROUND_EPS, Pooled::NONE, RoundMode::Abs);
    }
    normalize_mut(&mut x);
    let mut lambda = rayleigh_quotient(h, &x).re_s();
    if let Some(p) = penalty {
        lambda += p.energy(&p.overlaps(&x));
    }
    let mut alpha = 1.0f64;
    let mut lambda_known: Option<f64> = None;
    let mut pre: Option<GradientAt<T>> = None;
    for it in 0..opts.max_iter {
        let (grad, l) = match pre.take() {
            Some((_bl, _br, g, l)) => {
                ls_stats::bump(&ls_stats::CACHE_HITS);
                (g, l)
            }
            None => {
                let (base_l, base_r) = tangent_bases(&x);
                let seed = mix_seed(opts.seed, it as u64);
                gradient_at(
                    h,
                    &x,
                    &base_l,
                    &base_r,
                    opts.gradtol,
                    seed,
                    lambda_known,
                    penalty,
                )
            }
        };
        lambda = l;
        let gnorm = grad.norm();
        if gnorm <= opts.eps * lambda.abs() {
            break;
        }
        let eta = grad.scale(T::from_f64(-1.0));
        let slope = -gnorm * gnorm;
        match linesearch_cached(
            h,
            &x,
            &eta,
            lambda,
            slope,
            1.0f64.min(2.0 * alpha),
            gnorm,
            Some(slope),
            penalty,
            slope_floor(),
            secant_all(),
        ) {
            None => break,
            Some((a_n, x_n, l_n, p_n)) => {
                alpha = a_n;
                x = x_n;
                lambda = l_n;
                lambda_known = Some(l_n);
                pre = p_n;
            }
        }
    }
    (lambda, x)
}

/// Riemannian nonlinear CG: FR/PR⁺ update with projective transport of the prior search direction
/// and a restart-on-non-descent guard. Returns `(λ, x)` with `x` unit-norm.
pub fn riemannian_cg<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RiemannianOpts,
) -> (f64, TtVector<T>) {
    let r = cg_impl(h, x0, opts, None, None, 0);
    (r.lambda, r.x)
}

/// [`riemannian_cg`] returning the exit reason and iteration count as well ([`CgReport`]).
pub fn riemannian_cg_report<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RiemannianOpts,
) -> CgReport<T> {
    cg_impl(h, x0, opts, None, None, 0)
}

/// An observer for [`riemannian_cg_observed`]: called once per ACCEPTED iterate with the
/// 1-based iteration count, the new state (normalized, in the gauge the driver holds) and the
/// Riemannian gradient norm there. Returning `false` stops the run at that iterate.
pub type CgObserver<'a, T> = &'a mut dyn FnMut(usize, &TtVector<T>, f64) -> bool;

type ObserverRef<'b, 'a, T> = &'b mut (dyn FnMut(usize, &TtVector<T>, f64) -> bool + 'a);

/// [`riemannian_cg`] with an observer on the genuine trajectory — the same iterates a plain call
/// produces, exposed for instruments that need to look at intermediate states (PD signature
/// along a polish, rank/σ drift) without perturbing the CG memory by re-entering it. The observer
/// runs AFTER the step is accepted and the next gradient is formed; anything it computes on the
/// state (probes, readouts) leaves the driver's own bases and direction untouched.
pub fn riemannian_cg_observed<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RiemannianOpts,
    observer: CgObserver<'_, T>,
) -> (f64, TtVector<T>) {
    let r = cg_impl(h, x0, opts, None, Some(observer), 0);
    (r.lambda, r.x)
}

/// The certified polish (the `newton_polish` `certify` loop on top of
/// [`riemannian_cg`]): descend; where the descent stops on its own criterion (`Gradient` or
/// `LineSearch`) probe `A + B` exactly as `newton_polish` does (radial deflation, diagonal
/// preconditioner, tangent-cleaned Davidson, deterministic diagonal seed); on a negative witness
/// kick along ±witness on a halving ladder with the driver's own trial (HOSVD retraction, plus
/// the ALS pair under [`Retraction::AlsRounded`]), take the longest length that lowers the energy
/// resolvably, and resume within the shared `opts.max_iter` budget, up to `max_kicks` times. The
/// report carries every verdict, the kicks, and the final `A − B` probe when asked for.
pub fn riemannian_cg_certified<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RiemannianOpts,
    copts: &CertifyOpts,
    mut observer: Option<CgObserver<'_, T>>,
) -> CertifyReport<T> {
    use crate::solvers::hess::{HessPart, PdVerdict};
    let trace = crate::knobs::CG_TRACE.get();
    let pairs = match opts.retraction {
        Retraction::Hosvd => 0,
        Retraction::AlsRounded { pairs, .. } => pairs,
    };
    let mut x = x0.clone();
    let mut used = 0usize;
    let mut kicks: Vec<KickRecord> = Vec::new();
    loop {
        let mut o = opts.clone();
        o.max_iter = opts.max_iter.saturating_sub(used);
        let r = cg_impl(h, &x, &o, None, observer.as_deref_mut(), used);
        used += r.iterations;
        x = r.x;
        let (lambda, gnorm, exit) = (r.lambda, r.gnorm, r.exit);
        let finish = |x: TtVector<T>,
                      rr: Option<PdVerdict>,
                      ii: Option<PdVerdict>,
                      verdict: CertifyVerdict,
                      kicks: Vec<KickRecord>| CertifyReport {
            lambda,
            x,
            iterations: used,
            gnorm,
            exit,
            kicks,
            rr,
            ii,
            verdict,
        };
        if !matches!(exit, CgExit::Gradient | CgExit::LineSearch) {
            return finish(x, None, None, CertifyVerdict::NotProbed, kicks);
        }
        let probe = |part: HessPart, name: &str| {
            let mut last = None;
            let mut total = 0usize;
            for k in 0..copts.probe_seeds.max(1) {
                let (v, w, a) = certify_probe(
                    h,
                    &x,
                    lambda,
                    part,
                    copts.probe_budget,
                    mix_seed(opts.seed, k as u64),
                    &copts.davidson,
                );
                total += a;
                if trace {
                    eprintln!("CG certify {name} seed {k}: {v:?} after {a} applies (it {used})");
                }
                let inconclusive = matches!(v, PdVerdict::Inconclusive { .. });
                last = Some((v, w, total));
                if !inconclusive {
                    break;
                }
            }
            last.expect("probe_seeds >= 1")
        };
        let (vrr, witness, applies) = probe(HessPart::Rr, "A+B");
        let ii = || copts.probe_ii.then(|| probe(HessPart::Ii, "A−B").0);
        match vrr {
            PdVerdict::LikelyPd { lambda_min, .. } => {
                let ii = ii();
                return finish(
                    x,
                    Some(vrr),
                    ii,
                    CertifyVerdict::Minimizer {
                        lambda_rr: lambda_min,
                    },
                    kicks,
                );
            }
            PdVerdict::Inconclusive {
                lambda_min,
                residual,
            } => {
                let ii = ii();
                return finish(
                    x,
                    Some(vrr),
                    ii,
                    CertifyVerdict::Inconclusive {
                        lambda_rr: lambda_min,
                        residual,
                    },
                    kicks,
                );
            }
            PdVerdict::WitnessNegative { lambda: lneg } => {
                if kicks.len() < copts.max_kicks && used < opts.max_iter {
                    let floor = (0.1 * opts.eps * lambda.abs()).max(1e-12);
                    let tol = (ALS_INNER_TOL_C * gnorm).max(floor).min(INNER_TOL_CAP);
                    if let Some((x_new, l_new, alpha)) = kick_along(
                        h,
                        &x,
                        lambda,
                        &witness,
                        pairs,
                        tol,
                        copts.kick_alpha0,
                        copts.kick_ladder,
                        trace,
                    ) {
                        kicks.push(KickRecord {
                            at_iteration: used,
                            witness_lambda: lneg,
                            probe_applies: applies,
                            alpha,
                            delta_e: l_new - lambda,
                        });
                        x = x_new;
                        continue;
                    }
                }
                let ii = ii();
                return finish(
                    x,
                    Some(vrr),
                    ii,
                    CertifyVerdict::Saddle {
                        witness_lambda: lneg,
                    },
                    kicks,
                );
            }
        }
    }
}

/// Riemannian nonlinear CG on the PENALIZED functional `J` of [`crate::solvers::penalty`]:
/// `riemannian_cg` against `H` plus the overlap penalty `w Σⱼ|⟨ψ̂ⱼ,x⟩|²`, with the radial shift
/// `J` in the gradient and `J` scoring every line-search trial.
pub fn riemannian_cg_penalized<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    penalty: &Penalty<TtVector<T>>,
    opts: &RiemannianOpts,
) -> (f64, TtVector<T>) {
    let r = cg_impl(h, x0, opts, penalty.active(), None, 0);
    (r.lambda, r.x)
}

fn cg_impl<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &RiemannianOpts,
    penalty: Option<&Penalty<TtVector<T>>>,
    mut observer: Option<ObserverRef<'_, '_, T>>,
    it0: usize,
) -> CgReport<T> {
    let als_arm = match opts.retraction {
        Retraction::Hosvd => None,
        Retraction::AlsRounded { pairs, delta0 } => {
            assert!(
                penalty.is_none(),
                "riemannian_cg: Retraction::AlsRounded is undefined for the penalized functional"
            );
            assert!(
                pairs >= 1,
                "riemannian_cg: Retraction::AlsRounded needs pairs >= 1"
            );
            Some((pairs, delta0))
        }
    };
    let mut delta: Option<f64> = None;

    let mut x = x0.clone();
    if !opts.preserve_grid {
        x.round_mut(INITIAL_ROUND_EPS, Pooled::NONE, RoundMode::Abs);
    }
    normalize_mut(&mut x);

    let (mut base_l, mut base_r) = tangent_bases(&x);
    let (mut grad, mut lambda) = gradient_at(
        h,
        &x,
        &base_l,
        &base_r,
        opts.gradtol,
        mix_seed(opts.seed, 0),
        None,
        penalty,
    );
    let mut gnorm = grad.norm();
    let mut eta = grad.scale(T::from_f64(-1.0));
    let mut alpha = 1.0f64;
    let trace = crate::knobs::CG_TRACE.get();
    let mut iters_run = 0usize;
    let mut exit = CgExit::Budget;
    let mut it = 0usize;

    {
        while it < opts.max_iter {
            if trace {
                eprintln!(
                    "CG it={it} λ={lambda:.16e} ‖g‖={gnorm:.6e} stop={:.6e} ls_grads={} hits={}",
                    opts.eps * lambda.abs(),
                    crate::solvers::riemannian::ls_stats::GRAD_EVALS
                        .load(std::sync::atomic::Ordering::Relaxed),
                    crate::solvers::riemannian::ls_stats::CACHE_HITS
                        .load(std::sync::atomic::Ordering::Relaxed),
                );
            }
            if gnorm <= opts.eps * lambda.abs() {
                exit = CgExit::Gradient;
                break;
            }
            iters_run = it + 1;

            let mut slope = grad.dot(&eta).re_s();
            let mut is_sd = it == 0;
            if slope >= 0.0 {
                eta = grad.scale(T::from_f64(-1.0));
                slope = -gnorm * gnorm;
                is_sd = true;
            }

            let lambda_known;
            let mut restart_beta = false;
            let mut pre = None;
            let step = match als_arm {
                None => {
                    let search = |eta: &TtTangent<T>, slope: f64| {
                        linesearch_cached(
                            h,
                            &x,
                            eta,
                            lambda,
                            slope,
                            1.0f64.min(2.0 * alpha),
                            gnorm,
                            Some(slope),
                            penalty,
                            slope_floor(),
                            secant_all(),
                        )
                    };
                    let mut found = search(&eta, slope);
                    if found.is_none() && !is_sd && slope_floor() {
                        ls_stats::bump(&ls_stats::SD_RETRIES);
                        eta = grad.scale(T::from_f64(-1.0));
                        slope = -gnorm * gnorm;
                        found = search(&eta, slope);
                    }
                    found.map(|(a, xa, la, p)| {
                        pre = p;
                        (a, xa, la)
                    })
                }
                Some((pairs, delta0)) => {
                    let ctl = AlsStepCtl {
                        pairs,
                        delta0,
                        delta,
                        opts,
                        is_sd,
                        it: it0 + it,
                        trace,
                    };
                    als_step(h, &x, &mut eta, &mut slope, &grad, gnorm, lambda, &ctl).map(|s| {
                        delta = Some(s.delta);
                        restart_beta = s.restart_beta;
                        pre = s.pre;
                        (s.alpha, s.x, s.lambda)
                    })
                }
            };
            match step {
                None => {
                    exit = CgExit::LineSearch;
                    break;
                }
                Some((a_n, x_n, l_n)) => {
                    alpha = a_n;
                    x = x_n;
                    lambda_known = Some(l_n);
                }
            }

            let (bl_n, br_n, gradn, l) = match pre.take() {
                Some(p) => {
                    ls_stats::bump(&ls_stats::CACHE_HITS);
                    p
                }
                None => {
                    let (bl_n, br_n) = tangent_bases(&x);
                    let (gradn, l) = gradient_at(
                        h,
                        &x,
                        &bl_n,
                        &br_n,
                        opts.gradtol,
                        mix_seed(opts.seed, (it0 + it) as u64 + 1),
                        lambda_known,
                        penalty,
                    );
                    (bl_n, br_n, gradn, l)
                }
            };
            base_l = bl_n;
            base_r = br_n;
            lambda = l;
            let gnormn = gradn.norm();

            let eta_t = TtTangent::project(&base_l, &base_r, &eta.to_vector());
            let beta = match opts.variant {
                CgVariant::Fr => gnormn * gnormn / (gnorm * gnorm),
                CgVariant::Pr => {
                    let g_t = TtTangent::project(&base_l, &base_r, &grad.to_vector());
                    let num = gradn.dot(&gradn.sub(&g_t)).re_s();
                    (num / (gnorm * gnorm)).max(0.0)
                }
            };
            let beta = if restart_beta { 0.0 } else { beta };
            eta = gradn
                .scale(T::from_f64(-1.0))
                .add(&eta_t.scale(T::from_f64(beta)));
            grad = gradn;
            gnorm = gnormn;
            it += 1;
            if let Some(obs) = observer.as_mut()
                && !obs(it0 + it, &x, gnorm)
            {
                exit = CgExit::Observer;
                break;
            }
        }
    }
    if trace {
        eprintln!(
            "CG exit={exit:?} after {iters_run}/{} iterations, ‖g‖={gnorm:.6e}, λ={lambda:.16e}",
            opts.max_iter
        );
    }
    CgReport {
        lambda,
        x,
        iterations: iters_run,
        gnorm,
        exit,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StepVerdict {
    Accept,
    FloorAccept,
    Reject,
    FloorRefuse,
}

fn radius_verdict(e0: f64, e_try: f64, pred: f64, efloor: f64, delta: f64) -> (StepVerdict, f64) {
    if !e_try.is_finite() {
        return (StepVerdict::Reject, delta / 4.0);
    }
    if pred < efloor {
        return if e_try <= e0 {
            (StepVerdict::FloorAccept, delta)
        } else {
            (StepVerdict::FloorRefuse, delta)
        };
    }
    let dec = e0 - e_try;
    let rho = dec / pred;
    let relaxed = dec >= 0.1 * pred - 2.0 * efloor;
    if rho >= 0.1 || relaxed {
        let next = if rho > 0.75 {
            (2.0 * delta).min(ALS_DELTA_MAX)
        } else {
            delta
        };
        (StepVerdict::Accept, next)
    } else {
        (StepVerdict::Reject, delta / 4.0)
    }
}

fn ambient_model_alpha<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    eta: &TtTangent<T>,
    lambda0: f64,
    gdot: f64,
) -> Option<f64> {
    let w = eta.to_vector();
    let s1 = dot(x, &w).re_s();
    let s2 = dot(&w, &w).re_s();
    let c1 = gdot + lambda0 * s1;
    let c2 = xt_h_y(&w, h, &w).re_s();
    let a = c2 * s1 - c1 * s2;
    let b = c2 - lambda0 * s2;
    let c = c1 - lambda0 * s1;
    let pred =
        |al: f64| (lambda0 + 2.0 * c1 * al + c2 * al * al) / (1.0 + 2.0 * s1 * al + s2 * al * al);
    let mut cand: Vec<f64> = Vec::new();
    if a.abs() > f64::EPSILON * b.abs().max(c.abs()).max(1.0) {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            cand.push((-b + disc.sqrt()) / (2.0 * a));
            cand.push((-b - disc.sqrt()) / (2.0 * a));
        }
    } else if b.abs() > 0.0 {
        cand.push(-c / b);
    }
    cand.retain(|&al| al > 0.0 && (1.0 + 2.0 * s1 * al + s2 * al * al) > 0.0);
    cand.into_iter().min_by(|p, q| {
        pred(*p)
            .partial_cmp(&pred(*q))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

fn als_trial<T: Scalar>(
    h: &SparseHamiltonian<T>,
    eta: &TtTangent<T>,
    alpha: f64,
    pairs: usize,
    tol: f64,
) -> (TtVector<T>, f64, f64) {
    let mut xa = eta.retract_hosvd(T::from_f64(alpha));
    normalize_mut(&mut xa);
    let xh = xa.clone();
    let mut lam = 0.0f64;
    for _ in 0..pairs {
        lam = als_sweep_pair(h, &mut xa, tol).re_s();
    }
    normalize_mut(&mut xa);
    if pairs == 0 {
        lam = rayleigh_quotient(h, &xa).re_s();
    }
    let ov = dot(&xh, &xa).re_s().abs().min(1.0);
    let d_als = (2.0 * (1.0 - ov)).max(0.0).sqrt();
    (xa, lam, d_als)
}

struct AlsStepCtl<'a> {
    pairs: usize,
    delta0: Option<f64>,
    delta: Option<f64>,
    opts: &'a RiemannianOpts,
    is_sd: bool,
    it: usize,
    trace: bool,
}

type GradientAt<T> = (TtVector<T>, TtVector<T>, TtTangent<T>, f64);

struct AlsStep<T: Scalar> {
    alpha: f64,
    x: TtVector<T>,
    lambda: f64,
    delta: f64,
    restart_beta: bool,
    pre: Option<GradientAt<T>>,
}

#[allow(clippy::too_many_arguments)]
fn als_step<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    eta: &mut TtTangent<T>,
    slope: &mut f64,
    grad: &TtTangent<T>,
    gnorm: f64,
    lambda0: f64,
    ctl: &AlsStepCtl,
) -> Option<AlsStep<T>> {
    let efloor = ENERGY_NOISE_C * f64::EPSILON * (1.0 + lambda0.abs());
    let floor = (0.1 * ctl.opts.eps * lambda0.abs()).max(1e-12);
    let tol = (ALS_INNER_TOL_C * gnorm).max(floor).min(INNER_TOL_CAP);
    let mut enorm = eta.norm();
    let mut delta = match (ctl.delta, ctl.delta0) {
        (Some(d), _) => d,
        (None, Some(d0)) => d0,
        (None, None) => {
            ambient_model_alpha(h, x, eta, lambda0, *slope).map_or(enorm, |a| a * enorm)
        }
    }
    .min(ALS_DELTA_MAX);
    let mut is_sd = ctl.is_sd;
    let mut rej = 0usize;
    loop {
        let alpha = delta / enorm;
        let pred = -*slope * alpha;
        let a_m = if pred < 100.0 * ALS_NOISE_K * efloor {
            ambient_model_alpha(h, x, eta, lambda0, *slope)
        } else {
            None
        };
        let noise = match a_m {
            Some(a) => -*slope * a < ALS_NOISE_K * efloor,
            None => pred < ALS_NOISE_K * efloor,
        };
        if noise {
            let mut a = a_m.unwrap_or(alpha);
            for bt in 0..=MAX_GRAD_TRIALS {
                let (x_try, l_try, d_als) = als_trial(h, eta, a, ctl.pairs, tol);
                let d_step = a * enorm;
                let restart_beta = d_als > ALS_RESTART_RATIO * d_step;
                let not_worse = l_try - lambda0 <= efloor;
                let next = d_step.min(ALS_DELTA_MAX);
                let (bl, br) = tangent_bases(&x_try);
                let (g_try, l_g) = gradient_at(
                    h,
                    &x_try,
                    &bl,
                    &br,
                    ctl.opts.gradtol,
                    mix_seed(ctl.opts.seed, ctl.it as u64 + 1),
                    Some(l_try),
                    None,
                );
                let gn = g_try.norm();
                let ok = not_worse && gn <= gnorm;
                if ctl.trace {
                    eprintln!(
                        "CG-ALS it={} NOISE bt={bt} α={a:.3e} Δ={d_step:.3e} dλ={:+.3e} \
                         gnorm0={gnorm:.3e} gnorm_a={gn:.3e} tol={tol:.1e} d_als/Δ={:.2} {}",
                        ctl.it,
                        l_try - lambda0,
                        d_als / d_step,
                        if ok { "Accept" } else { "Reject" }
                    );
                }
                if ok {
                    return Some(AlsStep {
                        alpha: a,
                        x: x_try,
                        lambda: l_try,
                        delta: next,
                        restart_beta,
                        pre: Some((bl, br, g_try, l_g)),
                    });
                }
                a *= BACKTRACK_RHO;
            }
            return None;
        }
        let (x_try, l_try, d_als) = als_trial(h, eta, alpha, ctl.pairs, tol);
        let restart_beta = d_als > ALS_RESTART_RATIO * delta;
        {
            let (verdict, next) = radius_verdict(lambda0, l_try, pred, efloor, delta);
            if ctl.trace {
                eprintln!(
                    "CG-ALS it={} try={rej} Δ={delta:.3e} α={alpha:.3e} pred={pred:.3e} dλ={:+.3e} \
                     ρ={:.3} tol={tol:.1e} d_als/Δ={:.2} {verdict:?}",
                    ctl.it,
                    l_try - lambda0,
                    (lambda0 - l_try) / pred,
                    d_als / delta
                );
            }
            match verdict {
                StepVerdict::Accept | StepVerdict::FloorAccept => {
                    return Some(AlsStep {
                        alpha,
                        x: x_try,
                        lambda: l_try,
                        delta: next,
                        restart_beta,
                        pre: None,
                    });
                }
                StepVerdict::FloorRefuse => return None,
                StepVerdict::Reject => {
                    delta = next;
                    rej += 1;
                }
            }
        }
        if rej > ALS_MAX_REJ {
            if is_sd {
                return None;
            }
            *eta = grad.scale(T::from_f64(-1.0));
            *slope = -gnorm * gnorm;
            enorm = gnorm;
            is_sd = true;
            rej = 0;
        }
    }
}

/// One block probe at `x` (`lambda` = its Rayleigh quotient), as `newton_polish` runs it (radial
/// deflation, diagonal preconditioner, tangent-cleaned Davidson) but seeded with the diagonal
/// PLUS a seeded random tangent of the same norm: the diagonal alone inherits every symmetry of
/// `x`, so at an exactly symmetric saddle whose negative direction breaks that symmetry a
/// diagonal-seeded Davidson has ZERO overlap with the witness and would certify "PD" (the
/// eps-1e-3 untyped water state was caught only because the polish had already drifted 3e-9 off
/// A1). With `dav.block > 1` the further seeds are independent random tangents. Returns the
/// verdict, the certified Ritz vector (meaningful on `WitnessNegative`) and the applies spent.
/// A probe reports a Ritz value, not `λ_min`. This is the probe
/// [`riemannian_cg_certified`] runs; it is public so the probe can be measured on a saved state
/// without the descent.
pub fn certify_probe<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    lambda: f64,
    part: crate::solvers::hess::HessPart,
    budget: usize,
    seed: u64,
    dav: &crate::solvers::hess::ProbeDavidson,
) -> (crate::solvers::hess::PdVerdict, TtTangent<T>, usize) {
    use crate::solvers::hess::{
        HessPart, WeingartenCtx, add_complement_shift, clean_tangent, complement_shift,
        deflate_radial, diag_inf_scale, pd_probe_kernel_with, radial_mode,
    };
    use rand::SeedableRng;
    let (base_l, base_r) = tangent_bases(x);
    let ctx = WeingartenCtx::from_parts(h, base_l.clone(), base_r.clone(), lambda);
    let radial = radial_mode(&ctx);
    let dprobe = ctx.a_diagonal();
    let scale = diag_inf_scale(&dprobe);
    let mut v0 = TtTangent::project(&base_l, &base_r, &dprobe.to_vector());
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(mix_seed(seed, 0xC3));
    let mut randn = || {
        TtTangent::project(
            &base_l,
            &base_r,
            &crate::vector::tt_randn(x.space, &x.ranks, &mut rng),
        )
    };
    let mut r = randn();
    let (nv, nr) = (v0.norm(), r.norm());
    if nr > 0.0 {
        r.scale_mut(T::from_f64(if nv > 0.0 { nv / nr } else { 1.0 }));
        v0.axpy(T::one(), &r);
    }
    let mut seeds = vec![v0];
    for _ in 1..dav.block.max(1) {
        seeds.push(randn());
    }
    let mu = complement_shift(scale);
    pd_probe_kernel_with(
        |p: &TtTangent<T>| {
            let mut q = p.clone();
            clean_tangent(radial.as_ref(), &mut q);
            let mut hq = match part {
                HessPart::Rr => ctx.hpp_apply(&q),
                HessPart::Ii => ctx.hmm_apply(&q),
            };
            deflate_radial(radial.as_ref(), &mut hq);
            add_complement_shift(&mut hq, p, q, mu);
            hq
        },
        |t: &mut TtTangent<T>, theta: f64| crate::solvers::davidson::precond(t, &dprobe, theta),
        |t: &mut TtTangent<T>| clean_tangent(radial.as_ref(), t),
        &seeds,
        budget,
        scale,
        dav,
    )
}

#[allow(clippy::too_many_arguments)]
fn kick_along<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    lambda: f64,
    witness: &TtTangent<T>,
    pairs: usize,
    tol: f64,
    alpha0: f64,
    ladder: u32,
    trace: bool,
) -> Option<(TtVector<T>, f64, f64)> {
    let _ = x;
    let efloor = ENERGY_NOISE_C * f64::EPSILON * (1.0 + lambda.abs());
    let wn = witness.norm();
    if wn <= 0.0 {
        return None;
    }
    for sign in [1.0f64, -1.0] {
        let dir = witness.scale(T::from_f64(sign / wn));
        let mut a = alpha0;
        for _ in 0..=ladder {
            let (x_try, l_try, _) = als_trial(h, &dir, a, pairs, tol);
            let ok = l_try < lambda - efloor;
            if trace {
                eprintln!(
                    "CG kick sign={sign:+} α={a:.3e} dλ={:+.3e} {}",
                    l_try - lambda,
                    if ok { "Accept" } else { "reject" }
                );
            }
            if ok {
                return Some((x_try, l_try, a));
            }
            a *= 0.5;
        }
    }
    None
}

/// Left- and right-orthogonalized copies of `x` at its own ranks — the pinned left/right-orthogonal factorizations that gauge-fix every tangent
/// projection at `x`.
pub fn tangent_bases<T: Scalar>(x: &TtVector<T>) -> (TtVector<T>, TtVector<T>) {
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    (base_l, base_r)
}

/// Scale by `inv(norm_lr(x))`, zero-norm-guarded. Reaches unit Frobenius norm; `x`'s canonical
/// form is not redistributed (no downstream consumer relies on it: `tangent_bases`
/// re-orthogonalizes, `rayleigh_quotient` is gauge-free).
pub fn normalize_mut<T: Scalar>(x: &mut TtVector<T>) {
    let n = x.norm_lr(false);
    if n != 0.0 {
        x.scale_mut(T::from_f64(1.0 / n));
    }
}

/// The Riemannian gradient at `x` against the tangent bases `base_l`/`base_r`:
/// `λ = Re RayleighQuotient(H, x)` and `P(H·x − λ·x)`, contracted through `H`'s environments
/// without forming `H·x` (exact to roundoff). Returns `(grad, λ)`. `tol` and `seed` are inert.
pub fn riemannian_gradient<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    tol: f64,
    seed: u64,
) -> (TtTangent<T>, f64) {
    gradient_impl(h, x, base_l, base_r, tol, seed)
}

/// [`riemannian_gradient`] for the PENALIZED functional: `P((H − J)x) + w Σⱼ oⱼ P(ψ̂ⱼ)` at the
/// frames `(base_l, base_r)`, and `J` itself. Returns `(grad, J)`.
#[allow(clippy::too_many_arguments)]
pub fn riemannian_gradient_penalized<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    penalty: &Penalty<TtVector<T>>,
    tol: f64,
    seed: u64,
) -> (TtTangent<T>, f64) {
    gradient_at(h, x, base_l, base_r, tol, seed, None, penalty.active())
}

/// The EXACT gate `‖P((H − λ)x)‖` at unit-norm `x`, `λ = ⟨x,Hx⟩` — one environment-contracted
/// gradient at fresh [`tangent_bases`], no `H·x`.
pub fn exact_gradnorm<T: Scalar>(h: &SparseHamiltonian<T>, x: &TtVector<T>) -> f64 {
    let (base_l, base_r) = tangent_bases(x);
    let lambda = rayleigh_quotient(h, x).re_s();
    crate::solvers::hess::riemannian_gradient_env(h, &base_l, &base_r, lambda).norm()
}

/// The exact gate of the PENALIZED functional at unit-norm `x`:
/// `‖P((H − J)x) + w Σⱼ oⱼ P(ψ̂ⱼ)‖`, by environment contraction — no `H·x`.
pub fn penalized_exact_gradnorm<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    penalty: &Penalty<TtVector<T>>,
) -> f64 {
    let Some(p) = penalty.active() else {
        return exact_gradnorm(h, x);
    };
    let (base_l, base_r) = tangent_bases(x);
    let os = p.overlaps(x);
    let j = rayleigh_quotient(h, x).re_s() + p.energy(&os);
    let mut grad = crate::solvers::hess::riemannian_gradient_env(h, &base_l, &base_r, j);
    p.tangent_axpy(&mut grad, &os, |psi| {
        TtTangent::project(&base_l, &base_r, psi)
    });
    grad.norm()
}

fn gradient_impl<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    tol: f64,
    seed: u64,
) -> (TtTangent<T>, f64) {
    gradient_at(h, x, base_l, base_r, tol, seed, None, None)
}

#[allow(clippy::too_many_arguments)]
fn gradient_at<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    tol: f64,
    seed: u64,
    lambda_known: Option<f64>,
    penalty: Option<&Penalty<TtVector<T>>>,
) -> (TtTangent<T>, f64) {
    let _ = (tol, seed);
    let os = penalty.map(|p| p.overlaps(x));
    let lambda = match lambda_known {
        Some(l) => l,
        _ => match (penalty, &os) {
            (Some(p), Some(os)) => rayleigh_quotient(h, x).re_s() + p.energy(os),
            _ => rayleigh_quotient(h, x).re_s(),
        },
    };
    let mut grad = crate::solvers::hess::riemannian_gradient_env(h, base_l, base_r, lambda);
    if let (Some(p), Some(os)) = (penalty, &os) {
        p.tangent_axpy(&mut grad, os, |psi| TtTangent::project(base_l, base_r, psi));
    }
    (grad, lambda)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linesearch<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    eta: &TtTangent<T>,
    lambda0: f64,
    slope: f64,
    alpha0: f64,
    gnorm0: f64,
    gdot: Option<f64>,
) -> Option<(f64, TtVector<T>, f64)> {
    linesearch_impl(h, x, eta, lambda0, slope, alpha0, gnorm0, gdot, None)
}

fn floor_slope_at<T: Scalar>(
    h: &SparseHamiltonian<T>,
    xt: &TtVector<T>,
    lt: f64,
    w: &TtVector<T>,
    penalty: Option<&Penalty<TtVector<T>>>,
) -> (f64, GradientAt<T>) {
    let (bl, br) = tangent_bases(xt);
    let mut g = crate::solvers::hess::riemannian_gradient_env(h, &bl, &br, lt);
    if let Some(p) = penalty {
        p.tangent_axpy(&mut g, &p.overlaps(xt), |psi| {
            TtTangent::project(&bl, &br, psi)
        });
    }
    ls_stats::bump(&ls_stats::GRAD_EVALS);
    let phi = g.dot(&TtTangent::project(&bl, &br, w)).re_s();
    (phi, (bl, br, g, lt))
}

type LsCached<T> = (f64, TtVector<T>, f64, Option<GradientAt<T>>);

#[allow(clippy::too_many_arguments)]
fn linesearch_impl<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    eta: &TtTangent<T>,
    lambda0: f64,
    slope: f64,
    alpha0: f64,
    gnorm0: f64,
    gdot: Option<f64>,
    penalty: Option<&Penalty<TtVector<T>>>,
) -> Option<(f64, TtVector<T>, f64)> {
    linesearch_cached(
        h, x, eta, lambda0, slope, alpha0, gnorm0, gdot, penalty, false, false,
    )
    .map(|(a, xa, la, _)| (a, xa, la))
}

#[allow(clippy::too_many_arguments)]
fn linesearch_cached<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    eta: &TtTangent<T>,
    lambda0: f64,
    slope: f64,
    alpha0: f64,
    gnorm0: f64,
    gdot: Option<f64>,
    penalty: Option<&Penalty<TtVector<T>>>,
    slopes: bool,
    verify: bool,
) -> Option<LsCached<T>> {
    let w = eta.to_vector();
    let s1 = dot(x, &w).re_s();
    let s2 = dot(&w, &w).re_s();
    let (c1h, via_gdot) = match gdot {
        Some(g) => (g + lambda0 * s1, true),
        _ => (xt_h_y(x, h, &w).re_s(), false),
    };
    let c2h = xt_h_y(&w, h, &w).re_s();
    let (c1, c2) = match penalty {
        None => (c1h, c2h),
        Some(p) => {
            let (dc1, dc2) = p.linesearch_coeffs(x, &w);
            (if via_gdot { c1h } else { c1h + dc1 }, c2h + dc2)
        }
    };
    let a = c2 * s1 - c1 * s2;
    let b = c2 - lambda0 * s2;
    let c = c1 - lambda0 * s1;
    let pred =
        |al: f64| (lambda0 + 2.0 * c1 * al + c2 * al * al) / (1.0 + 2.0 * s1 * al + s2 * al * al);

    let mut cand: Vec<f64> = Vec::new();
    if a.abs() > f64::EPSILON * b.abs().max(c.abs()).max(1.0) {
        let disc = b * b - 4.0 * a * c;
        if disc >= 0.0 {
            cand.push((-b + disc.sqrt()) / (2.0 * a));
            cand.push((-b - disc.sqrt()) / (2.0 * a));
        }
    } else if b.abs() > 0.0 {
        cand.push(-c / b);
    }
    cand.retain(|&al| al > 0.0 && (1.0 + 2.0 * s1 * al + s2 * al * al) > 0.0);
    let mut alpha = if cand.is_empty() {
        alpha0
    } else {
        *cand
            .iter()
            .min_by(|p, q| {
                pred(**p)
                    .partial_cmp(&pred(**q))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    };

    let trace = crate::knobs::LS_TRACE.get();
    let efloor = ENERGY_NOISE_C * f64::EPSILON * (1.0 + lambda0.abs());
    ls_stats::bump(&ls_stats::CALLS);
    let mut noise_entered = false;
    let mut grad_trials = 0usize;
    let mut best_alpha = 0.0f64;
    let mut best_x: Option<TtVector<T>> = None;
    let mut best_lambda = lambda0;
    let slopes = slopes && gnorm0 > 0.0;
    let slope_trial = |a: f64| {
        let mut xa = eta.retract_hosvd(T::from_f64(a));
        normalize_mut(&mut xa);
        let la = match penalty {
            None => rayleigh_quotient(h, &xa).re_s(),
            Some(p) => rayleigh_quotient(h, &xa).re_s() + p.energy(&p.overlaps(&xa)),
        };
        ls_stats::bump(&ls_stats::TRIALS);
        let (phi, pre) = floor_slope_at(h, &xa, la, &w, penalty);
        SlopeTrial {
            alpha: a,
            lambda: la,
            phi,
            keep: (xa, pre),
        }
    };
    for bt in 0..MAX_BACKTRACK {
        if slopes && alpha * slope.abs() < SLOPE_FLOOR_K * efloor {
            ls_stats::bump(&ls_stats::FLOOR_ENTRIES);
            ls_stats::bump(&ls_stats::SLOPE_ENTRIES);
            let logged = |t: SlopeTrial<(TtVector<T>, GradientAt<T>)>| {
                let r = (t.lambda - lambda0 - t.alpha * (slope + t.phi)) / efloor;
                if r > SLOPE_GUARD_K {
                    ls_stats::bump(&ls_stats::SLOPE_GUARD);
                }
                if trace {
                    eprintln!(
                        "LS slope bt={bt} alpha={:.3e} phi/slope={:+.4} dλ/efloor={:+.3} r={r:+.3}",
                        t.alpha,
                        t.phi / slope,
                        (t.lambda - lambda0) / efloor
                    );
                }
                t
            };
            let first = logged(slope_trial(alpha));
            let best = slope_step(slope, true, first, |a| {
                ls_stats::bump(&ls_stats::SECANT_TRIALS);
                Some(logged(slope_trial(a)))
            });
            return match best {
                Some(t) => {
                    if t.alpha != alpha {
                        ls_stats::bump(&ls_stats::SECANT_ACCEPTS);
                    }
                    let (xb, pre) = t.keep;
                    Some((t.alpha, xb, t.lambda, Some(pre)))
                }
                None => {
                    if trace {
                        eprintln!("LS -> None (slope regime: no admissible trial)");
                    }
                    ls_stats::bump(&ls_stats::NONE);
                    None
                }
            };
        }
        let mut xa = eta.retract_hosvd(T::from_f64(alpha));
        normalize_mut(&mut xa);
        let la = match penalty {
            None => rayleigh_quotient(h, &xa).re_s(),
            Some(p) => rayleigh_quotient(h, &xa).re_s() + p.energy(&p.overlaps(&xa)),
        };
        crate::solvers::riemannian::ls_stats::bump(&crate::solvers::riemannian::ls_stats::TRIALS);
        if trace {
            eprintln!(
                "LS bt={bt} alpha={alpha:.3e} λ0={lambda0:.16e} λa={la:.16e} \
                 dλ={:.3e} armijo_rhs_delta={:.3e} slope={slope:.3e}",
                la - lambda0,
                ARMIJO_C * alpha * slope,
            );
        }
        if la <= lambda0 + (ARMIJO_C * alpha * slope).min(-efloor) {
            if verify && bt == 0 {
                let cap = la + efloor;
                let (phi, pre) = floor_slope_at(h, &xa, la, &w, penalty);
                let first = SlopeTrial {
                    alpha,
                    lambda: la,
                    phi,
                    keep: (xa, pre),
                };
                let t = slope_step(slope, false, first, |a| {
                    let mut xn = eta.retract_hosvd(T::from_f64(a));
                    normalize_mut(&mut xn);
                    let ln = match penalty {
                        None => rayleigh_quotient(h, &xn).re_s(),
                        Some(p) => rayleigh_quotient(h, &xn).re_s() + p.energy(&p.overlaps(&xn)),
                    };
                    ls_stats::bump(&ls_stats::TRIALS);
                    ls_stats::bump(&ls_stats::SECANT_TRIALS);
                    if ln > cap || ln > lambda0 + (ARMIJO_C * a * slope).min(-efloor) {
                        return None;
                    }
                    let (pn, pre_n) = floor_slope_at(h, &xn, ln, &w, penalty);
                    if trace {
                        eprintln!(
                            "LS secant alpha={a:.3e} phi/slope={:.4} dλ={:.3e}",
                            pn / slope,
                            ln - lambda0
                        );
                    }
                    Some(SlopeTrial {
                        alpha: a,
                        lambda: ln,
                        phi: pn,
                        keep: (xn, pre_n),
                    })
                })
                .expect("slope_step: an energy-certified first trial is admissible");
                if t.alpha != alpha {
                    ls_stats::bump(&ls_stats::SECANT_ACCEPTS);
                }
                let (xb, pre) = t.keep;
                return Some((t.alpha, xb, t.lambda, Some(pre)));
            }
            return Some((alpha, xa, la, None));
        }
        if lambda0 - la >= efloor && la < best_lambda {
            best_alpha = alpha;
            best_lambda = la;
            best_x = Some(xa.clone());
        }
        if gnorm0 > 0.0 && alpha * slope.abs() < efloor && la - lambda0 <= efloor {
            if !noise_entered {
                noise_entered = true;
                ls_stats::bump(&ls_stats::FLOOR_ENTRIES);
            }
            if bt == 0 {
                ls_stats::bump(&ls_stats::FIRST_ACCEPTS);
                if !floor_secant() {
                    return Some((alpha, xa, la, None));
                }
                let (mut a_prev, mut p_prev) = (0.0f64, slope);
                let mut a_cur = alpha;
                let (mut p_cur, pre0) = floor_slope_at(h, &xa, la, &w, penalty);
                let mut best_pre = pre0;
                let (mut best_a, mut best_l, mut best_p) = (alpha, la, p_cur);
                let mut best_x = xa;
                for _ in 0..SECANT_MAX_STEPS {
                    if best_p.abs() <= SECANT_DONE * slope.abs() || p_cur == p_prev {
                        break;
                    }
                    let a_next = a_cur - p_cur * (a_cur - a_prev) / (p_cur - p_prev);
                    if !a_next.is_finite() || a_next <= 0.0 {
                        break;
                    }
                    let a_next = a_next.min(SECANT_GROWTH * a_cur);
                    let mut xn = eta.retract_hosvd(T::from_f64(a_next));
                    normalize_mut(&mut xn);
                    let ln = match penalty {
                        None => rayleigh_quotient(h, &xn).re_s(),
                        Some(p) => rayleigh_quotient(h, &xn).re_s() + p.energy(&p.overlaps(&xn)),
                    };
                    ls_stats::bump(&ls_stats::TRIALS);
                    ls_stats::bump(&ls_stats::SECANT_TRIALS);
                    if ln - lambda0 > efloor {
                        break;
                    }
                    let (pn, pre_n) = floor_slope_at(h, &xn, ln, &w, penalty);
                    if trace {
                        eprintln!(
                            "LS secant alpha={a_next:.3e} phi/slope={:.4} dλ={:.3e}",
                            pn / slope,
                            ln - lambda0
                        );
                    }
                    if pn.abs() < best_p.abs() {
                        (best_a, best_l, best_p) = (a_next, ln, pn);
                        best_x = xn;
                        best_pre = pre_n;
                    }
                    (a_prev, p_prev, a_cur, p_cur) = (a_cur, p_cur, a_next, pn);
                }
                if best_a != alpha {
                    ls_stats::bump(&ls_stats::SECANT_ACCEPTS);
                }
                return Some((best_a, best_x, best_l, Some(best_pre)));
            }
            if grad_trials < MAX_GRAD_TRIALS {
                grad_trials += 1;
                ls_stats::bump(&ls_stats::GRAD_EVALS);
                let (bl, br) = tangent_bases(&xa);
                let ga = match penalty {
                    None => crate::solvers::hess::riemannian_gradient_env(h, &bl, &br, la).norm(),
                    Some(p) => {
                        let mut gt = crate::solvers::hess::riemannian_gradient_env(h, &bl, &br, la);
                        p.tangent_axpy(&mut gt, &p.overlaps(&xa), |psi| {
                            TtTangent::project(&bl, &br, psi)
                        });
                        gt.norm()
                    }
                };
                if trace {
                    eprintln!("LS grad-trial bt={bt} gnorm0={gnorm0:.3e} gnorm_a={ga:.3e}");
                }
                if ga <= GRAD_ACCEPT * gnorm0 {
                    return Some((alpha, xa, la, None));
                }
            }
        }
        alpha *= BACKTRACK_RHO;
    }
    if trace && best_x.is_none() {
        eprintln!("LS -> None (no trial decreased below λ0={lambda0:.16e})");
    }
    if best_x.is_none() {
        ls_stats::bump(&ls_stats::NONE);
    }
    best_x.map(|xb| (best_alpha, xb, best_lambda, None))
}

#[cfg(test)]
mod t8d_tests {
    use crate::TtVector;
    use crate::coeff::CoeffTensor;
    use crate::core::tangent::TtTangent;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::hamiltonian::{rayleigh_quotient, xt_h_y};
    use crate::ops::dot;
    use crate::qn::QnSpace;
    use crate::round::RoundMode;
    use crate::scalar::Scalar;
    use crate::solvers::penalty::Penalty;
    use crate::solvers::riemannian::{
        RiemannianOpts, cg_impl, exact_gradnorm, gd_impl, gradient_at, linesearch_impl,
        normalize_mut, penalized_exact_gradnorm, riemannian_cg_report, riemannian_gd,
        riemannian_gradient, tangent_bases,
    };
    use crate::test_util::{randomized_state, seeded};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t8d_als_radius_verdict_spelling() {
        use super::{ALS_DELTA_MAX, StepVerdict, radius_verdict};
        let efloor = 1e-12;
        let (v, d) = radius_verdict(0.0, -1e-6, 1e-6, efloor, 1e-3);
        assert_eq!((v, d), (StepVerdict::Accept, 2e-3));
        let (v, d) = radius_verdict(0.0, -1e-6, 1e-6, efloor, 0.8);
        assert_eq!((v, d), (StepVerdict::Accept, ALS_DELTA_MAX));
        let (v, d) = radius_verdict(0.0, -5e-7, 1e-6, efloor, 1e-3);
        assert_eq!((v, d), (StepVerdict::Accept, 1e-3));
        let (v, d) = radius_verdict(0.0, -5e-8, 1e-6, efloor, 1e-3);
        assert_eq!((v, d), (StepVerdict::Reject, 2.5e-4));
        let pred = 1e-10;
        let dec = 0.1 * pred - 1.5 * efloor;
        let (v, d) = radius_verdict(0.0, -dec, pred, efloor, 1e-3);
        assert_eq!((v, d), (StepVerdict::Accept, 1e-3));
        let (v, d) = radius_verdict(0.0, -1e-14, 1e-13, efloor, 1e-3);
        assert_eq!((v, d), (StepVerdict::FloorAccept, 1e-3));
        let (v, d) = radius_verdict(0.0, 1e-14, 1e-13, efloor, 1e-3);
        assert_eq!((v, d), (StepVerdict::FloorRefuse, 1e-3));
        let (v, d) = radius_verdict(0.0, f64::NAN, 1e-6, efloor, 1e-3);
        assert_eq!((v, d), (StepVerdict::Reject, 2.5e-4));
    }

    fn generic_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for i in 0..d {
            t.set(&[i, i], 0.15 * (i as f64 - 3.0));
            for j in 0..d {
                if i != j {
                    let g = 1.0 + (i as f64 - j as f64).abs();
                    t.set(&[i, j], -1.0 / g);
                    v.set(&[i, i, j, j], 0.5 / g);
                    v.set(&[i, j, j, i], 0.2 / g);
                }
            }
            v.set(&[i, i, i, i], 0.8);
        }
        (t, v)
    }

    fn unit_state(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
        let mut rng = seeded(seed);
        let mut x: TtVector<f64> = randomized_state(space, lo, hi, &mut rng);
        x.round_mut(1e-14, crate::qn::Pooled::NONE, RoundMode::Abs);
        normalize_mut(&mut x);
        x
    }

    #[test]
    fn t8d_penalty_gdot_shortcut_matches_exact_c1() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let (t, v) = generic_tv(d);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let x = unit_state(space, 3, 5, 0xD2_0101);
        let psi = unit_state(space, 2, 4, 0xD2_0102);
        let p = Penalty::new(2.5, std::slice::from_ref(&psi));
        assert!(p.is_active(), "the fixture's penalty must be active");

        let (base_l, base_r) = tangent_bases(&x);
        let (g, j) = gradient_at(&h, &x, &base_l, &base_r, 1e-12, 1234, None, Some(&p));
        let eta = g.scale(-1.0);
        let gnorm = g.norm();
        let slope = g.dot(&eta).re_s();
        assert!(
            (slope + gnorm * gnorm).abs() <= 1e-10 * (gnorm * gnorm),
            "slope must be −‖g‖² for the steepest-descent direction"
        );

        let wv = eta.to_vector();
        let s1 = dot(&x, &wv).re_s();
        let (dc1, _dc2) = p.linesearch_coeffs(&x, &wv);
        let c1_shortcut = slope + j * s1;
        let c1_exact = xt_h_y(&x, &h, &wv).re_s() + dc1;
        println!(
            "t8d gdot: c1_shortcut = {c1_shortcut:.16e}, ⟨x,Hι(η)⟩ + Δc₁ = {c1_exact:.16e}, \
             Δc₁ = {dc1:.6e}"
        );
        assert!(
            dc1.abs() > 1e-6 * c1_exact.abs().max(1.0),
            "Δc₁ = {dc1:.3e} is negligible against c₁ = {c1_exact:.3e}; this arm would not \
             discriminate a double count"
        );
        assert!(
            (c1_shortcut - c1_exact).abs() <= 1e-9 * c1_exact.abs().max(1.0),
            "the gdot shortcut is not ⟨x,Hι(η)⟩ + Δc₁ ({:.3e}); adding Δc₁ to that branch would \
             double-count it on every GD/CG step",
            (c1_shortcut - c1_exact).abs()
        );

        let (a_s, _, l_s) =
            linesearch_impl(&h, &x, &eta, j, slope, 1.0, gnorm, Some(slope), Some(&p))
                .expect("the shortcut branch found no decrease");
        let (a_e, _, l_e) = linesearch_impl(&h, &x, &eta, j, slope, 1.0, gnorm, None, Some(&p))
            .expect("the exact-c1 branch found no decrease");
        println!("t8d gdot: α {a_s:.12e} vs {a_e:.12e}, J⁺ {l_s:.16e} vs {l_e:.16e}");
        assert!(
            (a_s - a_e).abs() <= 1e-10 * a_e.abs().max(1.0),
            "the two c₁ branches chose different steps ({a_s:.6e} vs {a_e:.6e})"
        );
        assert!(
            (l_s - l_e).abs() <= 1e-10 * l_e.abs().max(1.0),
            "the two c₁ branches landed at different energies ({l_s:.16e} vs {l_e:.16e})"
        );
    }

    #[test]
    fn t8d_penalty_gd_tracks_the_penalized_functional() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let (t, v) = generic_tv(d);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let x = unit_state(space, 3, 5, 0xD2_0111);
        let psi = unit_state(space, 2, 4, 0xD2_0112);
        let p = Penalty::new(3.0, std::slice::from_ref(&psi));
        let opts = RiemannianOpts {
            eps: 1e-10,
            max_iter: 40,
            gradtol: 1e-12,
            ..RiemannianOpts::new(&space)
        };

        let j0 = rayleigh_quotient(&h, &x).re_s() + p.energy(&p.overlaps(&x));
        let gate0 = penalized_exact_gradnorm(&h, &x, &p);
        let (j, xf) = gd_impl(&h, &x, &opts, p.active());
        let lam = rayleigh_quotient(&h, &xf).re_s();
        let pen = p.energy(&p.overlaps(&xf));
        let gate = penalized_exact_gradnorm(&h, &xf, &p);
        println!(
            "t8d gd-pen: J {j0:.12e} → {j:.12e} (λ {lam:.12e} + pen {pen:.6e}), gate \
             {gate0:.3e} → {gate:.3e}"
        );
        assert!(
            (j - lam - pen).abs() <= 1e-10 * j.abs().max(1.0),
            "the penalized GD's reported energy is not λ + pen at the returned point ({:.3e})",
            (j - lam - pen).abs()
        );
        assert!(
            j < j0,
            "the penalized GD did not decrease J ({j0:.12e} → {j:.12e})"
        );
        assert!(
            gate < gate0,
            "the penalized GD did not reduce the exact penalized gate ({gate0:.3e} → {gate:.3e})"
        );

        let inactive = Penalty::new(0.0, std::slice::from_ref(&psi));
        let (jb, xb) = gd_impl(&h, &x, &opts, inactive.active());
        let (lb, xr) = riemannian_gd(&h, &x, &opts);
        assert_eq!(
            jb.to_bits(),
            lb.to_bits(),
            "an inactive penalty changed the GD energy ({jb:.17e} vs {lb:.17e})"
        );
        for k in 0..d {
            let (ca, cb) = (xb.core(k), xr.core(k));
            assert_eq!(
                ca.num_coeffs(),
                cb.num_coeffs(),
                "core {k} coefficient count"
            );
            for i in 0..ca.num_coeffs() {
                assert_eq!(
                    ca.coeff(i).to_bits(),
                    cb.coeff(i).to_bits(),
                    "an inactive penalty changed GD core {k} coefficient {i}"
                );
            }
        }
    }

    #[test]
    fn t8d_polish_passes_the_energy_floor() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let (t, v) = generic_tv(d);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let efloor = |l: f64| super::ENERGY_NOISE_C * f64::EPSILON * (1.0 + l.abs());
        let base = RiemannianOpts {
            max_iter: 500,
            gradtol: 1e-12,
            ..RiemannianOpts::new(&space)
        };
        let with_eps = |eps: f64| RiemannianOpts {
            eps,
            ..base.clone()
        };

        let x0 = unit_state(space, 1, 2, 0xD2_0121);
        let ra = riemannian_cg_report(&h, &x0, &with_eps(1e-6));
        let r1 = riemannian_cg_report(&h, &ra.x, &with_eps(1e-9 / ra.lambda.abs()));
        let g1 = exact_gradnorm(&h, &r1.x);
        println!(
            "t8d floor: λa = {:.16e} ({} it, {:?}), λ1 = {:.16e} ({} it, {:?}), ‖g1‖ = {g1:.3e}",
            ra.lambda, ra.iterations, ra.exit, r1.lambda, r1.iterations, r1.exit
        );
        assert!(g1 <= 2e-9, "the staging polish stopped at ‖g‖ = {g1:.3e}");

        let (bl, br) = tangent_bases(&r1.x);
        let (g, l) = gradient_at(&h, &r1.x, &bl, &br, 1e-12, base.seed, None, None);
        let gn = g.norm();
        let slope = -gn * gn;
        let (a, _, la) = linesearch_impl(
            &h,
            &r1.x,
            &g.scale(-1.0),
            l,
            slope,
            1.0,
            gn,
            Some(slope),
            None,
        )
        .expect("the line search refused every trial at ‖g‖ ≈ 1e-9");
        println!(
            "t8d floor: α·|slope| = {:.3e} vs floor {:.3e}, measured dλ = {:.3e}",
            a * slope.abs(),
            efloor(l),
            la - l
        );
        assert!(
            a * slope.abs() < efloor(l),
            "the fixture does not reach the noise regime (α·|slope| = {:.3e} ≥ {:.3e})",
            a * slope.abs(),
            efloor(l)
        );

        let deep = with_eps(1e-12 / r1.lambda.abs());
        let r2 = riemannian_cg_report(&h, &r1.x, &deep);
        let g2 = exact_gradnorm(&h, &r2.x);
        let (lg, xg) = riemannian_gd(&h, &r1.x, &deep);
        let gg = exact_gradnorm(&h, &xg);
        println!(
            "t8d floor: CG λ2 = {:.16e} ({} it, {:?}), ‖g2‖ = {g2:.3e}; GD λ = {lg:.16e}, \
             ‖g‖ = {gg:.3e}",
            r2.lambda, r2.iterations, r2.exit
        );
        assert!(
            g2 <= 1e-11,
            "CG stopped at ‖g‖ = {g2:.3e} (old floor ≈ 1e-9)"
        );
        assert!(
            r2.lambda <= r1.lambda + efloor(r1.lambda),
            "CG gave up energy below the floor ({:.3e})",
            r2.lambda - r1.lambda
        );
        assert!(
            gg <= 1e-10,
            "GD stopped at ‖g‖ = {gg:.3e} (old floor ≈ 1e-9)"
        );
        assert!(
            lg <= r1.lambda + efloor(r1.lambda),
            "GD gave up energy below the floor ({:.3e})",
            lg - r1.lambda
        );

        let p = Penalty::new(1.0, std::slice::from_ref(&r2.x));
        let xp0 = unit_state(space, 1, 2, 0xD2_0122);
        let rp1 = cg_impl(&h, &xp0, &with_eps(1e-6), p.active(), None, 0);
        let rp2 = cg_impl(
            &h,
            &rp1.x,
            &with_eps(1e-12 / rp1.lambda.abs()),
            p.active(),
            None,
            0,
        );
        let gp = penalized_exact_gradnorm(&h, &rp2.x, &p);
        println!(
            "t8d floor: penalized J = {:.16e} ({} + {} it, {:?}), ‖g‖ = {gp:.3e}",
            rp2.lambda, rp1.iterations, rp2.iterations, rp2.exit
        );
        assert!(
            gp <= 1e-11,
            "penalized CG stopped at ‖g‖ = {gp:.3e} (old floor ≈ 1e-9)"
        );
    }

    fn gd_by_hand(
        h: &SparseHamiltonian<f64>,
        x0: &TtVector<f64>,
        slopes: bool,
        verify: bool,
        gtol: f64,
        budget: usize,
    ) -> (f64, TtVector<f64>, usize, f64, usize) {
        let (mut x, mut known, mut pre, mut alpha, mut handed) = (x0.clone(), None, None, 1.0, 0);
        for it in 0..budget {
            let (g, l) = match pre.take() {
                Some((_, _, g, l)) => (g, l),
                None => {
                    let (bl, br) = tangent_bases(&x);
                    gradient_at(h, &x, &bl, &br, 1e-12, 0, known, None)
                }
            };
            let gn = g.norm();
            if gn <= gtol {
                return (l, x, it, gn, handed);
            }
            let slope = -gn * gn;
            let step = super::linesearch_cached(
                h,
                &x,
                &g.scale(-1.0),
                l,
                slope,
                1.0f64.min(2.0 * alpha),
                gn,
                Some(slope),
                None,
                slopes,
                verify,
            );
            let (a, xn, ln, p) = step
                .unwrap_or_else(|| panic!("line search returned None at it {it}, ‖g‖ {gn:.3e}"));
            handed += usize::from(p.is_some());
            (alpha, x, known, pre) = (a, xn, Some(ln), p);
        }
        let l = rayleigh_quotient(h, &x).re_s();
        let g = exact_gradnorm(h, &x);
        (l, x, budget, g, handed)
    }

    #[test]
    fn t8d_slope_rule_polishes_through_the_floor() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let (t, v) = generic_tv(d);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let efloor = |l: f64| super::ENERGY_NOISE_C * f64::EPSILON * (1.0 + l.abs());
        let opts = |eps: f64| RiemannianOpts {
            eps,
            max_iter: 500,
            gradtol: 1e-12,
            ..RiemannianOpts::new(&space)
        };
        let x0 = unit_state(space, 1, 2, 0xD2_0121);
        let ra = riemannian_cg_report(&h, &x0, &opts(1e-6));
        let r2 = riemannian_cg_report(&h, &ra.x, &opts(1e-12 / ra.lambda.abs()));
        let budget = 3000;
        let mut handed = [0usize; 2];
        for (k, verify) in [false, true].into_iter().enumerate() {
            let (l, _, it, g, n) = gd_by_hand(&h, &ra.x, true, verify, 1e-12, budget);
            println!(
                "t8d slope rule verify={verify}: λ = {l:.16e} (CG {:.16e}), {it} it, ‖g‖ = {g:.3e}, \
                 {n} gradients handed over",
                r2.lambda
            );
            assert!(
                g <= 1e-12,
                "slope-rule GD stopped at ‖g‖ = {g:.3e} after {it} it"
            );
            assert!(
                (l - r2.lambda).abs() <= efloor(l),
                "slope-rule GD ended {:.3e} from the shipped CG's energy",
                l - r2.lambda
            );
            handed[k] = n;
        }
        assert!(
            handed[1] > handed[0],
            "verify handed over no more gradients ({handed:?}): the extension did not run"
        );
    }

    #[test]
    fn t8d_gradient_matches_residual() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[0, 1], -1.0);
        t.set(&[1, 0], -1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8DC0);
        let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        let (base_l, base_r) = tangent_bases(&x);
        let (g, lambda) = riemannian_gradient(&h, &x, &base_l, &base_r, 1e-12, 1234);

        let exact_resid = &(&h * &x) - &(&x * lambda);
        let gref = TtTangent::project(&base_l, &base_r, &exact_resid);
        let diff = g.to_vector();
        let diff2 = &diff - &gref.to_vector();
        assert!(
            diff2.norm_lr(false) < 1e-6,
            "gradient == projected exact residual"
        );
        assert!(dot(&x, &g.to_vector()).abs() < 1e-8, "radial orthogonality");
    }
}

#[cfg(test)]
mod slope_step_tests {
    use super::{
        ARMIJO_C, SECANT_DONE, SECANT_MAX_STEPS, SLOPE_MAX_BISECT, SlopeTrial, slope_step,
    };

    const SLOPE: f64 = -2.0;

    fn at(phi: impl Fn(f64) -> f64, a: f64) -> SlopeTrial<()> {
        SlopeTrial {
            alpha: a,
            lambda: 1.1,
            phi: phi(a),
            keep: (),
        }
    }

    fn run(phi: impl Fn(f64) -> f64 + Copy, a0: f64, slopes_only: bool) -> (Option<f64>, usize) {
        let mut n = 0usize;
        let best = slope_step(SLOPE, slopes_only, at(phi, a0), |a| {
            n += 1;
            Some(at(phi, a))
        });
        (best.map(|t| t.alpha), n)
    }

    #[test]
    fn slope_step_finds_the_root_of_a_linear_slope() {
        let phi = |a: f64| SLOPE * (1.0 - a);
        let (a, n) = run(phi, 0.25, true);
        let a = a.expect("a descending model step is admissible");
        assert!(
            (a - 1.0).abs() < 1e-12,
            "one secant on a linear φ′ lands at α* = 1, got {a}"
        );
        assert_eq!(n, 1, "one extra trial");
        let (a, n) = run(phi, 3.0, true);
        let a = a.expect("bisection reaches an admissible trial");
        assert!(
            (a - 1.0).abs() < 1e-12,
            "bracketed secant on a linear φ′ is exact, got {a}"
        );
        assert_eq!(n, 2, "one bisection, one secant");
    }

    #[test]
    fn slope_step_admits_exactly_the_trapezoid_armijo_trials() {
        let edge = (1.0 - 2.0 * ARMIJO_C) * SLOPE.abs();
        let flat = |c: f64| move |_: f64| c;
        let (a, n) = run(flat(-0.5), 1.0, true);
        assert_eq!(
            (a, n),
            (Some(1.0), 1),
            "a descending φ′ is admissible; equal slopes stop"
        );
        let (a, _) = run(flat(edge), 1.0, true);
        assert_eq!(a, Some(1.0), "φ′ exactly at (1 − 2c)|slope| is admissible");
        let (a, n) = run(flat(edge * (1.0 + 1e-12)), 1.0, true);
        assert_eq!(a, None, "no trial passes the trapezoid Armijo test");
        assert_eq!(
            n,
            SECANT_MAX_STEPS + SLOPE_MAX_BISECT,
            "the budget, all bisections"
        );
        let (a, n) = run(flat(edge * 2.0), 1.0, false);
        assert!(a.is_some() && n <= SECANT_MAX_STEPS);
    }

    #[test]
    fn slope_step_stops_where_it_must() {
        let (a, n) = run(|a| SLOPE * 0.5 * SECANT_DONE * (1.0 - a), 0.5, true);
        assert_eq!((a, n), (Some(0.5), 0));
        let (a, n) = run(|a| SLOPE * (1.0 + a), 0.5, true);
        assert_eq!((a, n), (Some(0.5), 0));
        let phi = |a: f64| SLOPE * (1.0 - a);
        let best = slope_step(SLOPE, false, at(phi, 0.25), |_| None);
        assert_eq!(best.map(|t| t.alpha), Some(0.25));
        let mut seen = Vec::new();
        let phi = |a: f64| SLOPE * (1.0 - a / 1000.0);
        let _ = slope_step(SLOPE, true, at(phi, 1.0), |a| {
            seen.push(a);
            Some(at(phi, a))
        });
        assert_eq!(
            seen[0], 20.0,
            "the first extrapolation is clamped to 20× the model step"
        );
    }
}
