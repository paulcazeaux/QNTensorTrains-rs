// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) Riemannian gradient descent / nonlinear CG on the fixed-rank TT manifold — the
//! SU(2)-symmetric sibling of [`crate::solvers::riemannian`].
//! `riemannian` is DUPLICATED per layer rather than dispatched (same shape as the SU(2)
//! ALS/MALS/TDVP). The abelian [`crate::solvers::riemannian`] is mirrored **line-for-line** — the
//! Armijo / PR-FR-CG / exact-Rayleigh-minimizer line search is metric-agnostic and is copied
//! VERBATIM; the only substantive difference is the tangent/gradient machinery being SU(2)
//! ([`crate::su2::tangent`], [`crate::su2::hamiltonian::Su2SparseHamiltonian`]) and the metric
//! being the fixed-Sz-member (centerweighted) inner product. One divergence (2026-09-08): the
//! abelian driver's `Retraction::AlsRounded` is not mirrored here yet — port it once the
//! formaldehyde measurement graduates the arm (this layer's `RiemannianOpts` has no `retraction`).

use crate::qn::Pooled;
use crate::round::RoundMode;
use crate::scalar::Scalar;
use crate::solvers::penalty::Penalty;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient, xt_h_y};
use crate::su2::ops::dot;
use crate::su2::qn::Su2Space;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;

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
use crate::solvers::riemannian::{
    SLOPE_FLOOR_K, SLOPE_GUARD_K, SlopeTrial, secant_all, slope_floor, slope_step,
};

/// Nonlinear-CG update formula (abelian [`crate::solvers::riemannian::CgVariant`]). Copied
/// verbatim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CgVariant {
    /// Fletcher-Reeves.
    Fr,
    /// Polak-Ribière (with the `max(0, ·)` restart floor) — the default.
    Pr,
}

/// `riemannian_gd`/`riemannian_cg` options (abelian
/// [`crate::solvers::riemannian::RiemannianOpts`], `solvers/riemannian.rs`). Field names
/// kept identical to the abelian struct so the `tangent_hessian` binary §D cross-symmetry clauses read
/// cleanly against `crate::solvers::riemannian::RiemannianOpts`.
#[derive(Clone, Debug)]
pub struct RiemannianOpts {
    /// Gradient-norm stopping ratio `‖g‖/|λ| < eps`.
    pub eps: f64,
    pub max_iter: usize,
    /// Inert: the gradient is the exact environment contraction, so no residual is
    /// rounded. Kept for API stability; setting it changes nothing.
    pub gradtol: f64,
    /// Inert: the gradient is deterministic. Kept for API stability.
    pub seed: u64,
    /// CG only; ignored by [`riemannian_gd`]. Defaults to Polak-Ribière.
    pub variant: CgVariant,
    /// Grid-preservation opt-out of the entry cleanup round; the abelian
    /// [`crate::solvers::riemannian::RiemannianOpts::preserve_grid`] mirrored field for field (the
    /// two structs deliberately share spellings so the cross-symmetry clauses of `tests/su2_s6`
    /// read across). `false` (the default) is today's behaviour byte for byte; `true` skips
    /// `round_mut(1e-14, 0, Abs)` so a declared-but-rank-deficient REDUCED-MULTIPLET sector
    /// survives the loop instead of being collapsed.
    pub preserve_grid: bool,
}

impl RiemannianOpts {
    /// Defaults. `space` is not read; it stays in the signature so the abelian twin and every call
    /// site keep one spelling.
    pub fn new(_space: &Su2Space) -> Self {
        RiemannianOpts {
            eps: 1e-4,
            max_iter: 100,
            gradtol: 1e-6,
            seed: 1234,
            variant: CgVariant::Pr,
            preserve_grid: false,
        }
    }
}

fn mix_seed(base: u64, k: u64) -> u64 {
    let mut z = base ^ k;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Riemannian steepest descent (abelian twin
/// [`crate::solvers::riemannian::riemannian_gd`]). Returns `(λ,
/// x)` with `x` unit-norm. Mirrors the abelian driver line-for-line: round-to-achievable +
/// normalize start, per-iteration `tangent_bases`/gradient/line-search. **HAZARD H11**: `gnorm =
/// grad.norm()` and the convergence/slope computations MUST route through
/// [`Su2TtTangent::norm`] (Route A, centerweighted) — see module docs.
pub fn riemannian_gd<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &RiemannianOpts,
) -> (f64, Su2TtVector<T>) {
    gd_impl(h, x0, opts, None)
}

fn gd_impl<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &RiemannianOpts,
    penalty: Option<&Penalty<Su2TtVector<T>>>,
) -> (f64, Su2TtVector<T>) {
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
                crate::solvers::riemannian::ls_stats::bump(
                    &crate::solvers::riemannian::ls_stats::CACHE_HITS,
                );
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

/// Riemannian nonlinear CG (abelian twin
/// [`crate::solvers::riemannian::riemannian_cg`]): FR/PR⁺ update
/// with projective transport of the prior search direction (inlined via
/// `Su2TtTangent::project(&base_l, &base_r, &eta.to_vector())` — `projective_transport` itself is
/// NOT called by the driver, same as abelian) and a restart-on-non-descent guard. Returns `(λ,
/// x)` with `x` unit-norm. **HAZARD H11**: `gnorm`/`gnormn` (Route A norm), `slope =
/// grad.dot(&eta)` (Route A dot), the FR `β = gnormn²/gnorm²` and PR `β =
/// max(0, gradn.dot(&gradn.sub(&g_t))/gnorm²)` (Route A dot/norm throughout) — cross-check every
/// one against `solvers/riemannian.rs`; a metric used in one
/// place but not another is the classic silent bug.
pub fn riemannian_cg<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &RiemannianOpts,
) -> (f64, Su2TtVector<T>) {
    cg_impl(h, x0, opts, None)
}

/// SU(2) Riemannian nonlinear CG on the PENALIZED functional `J` of [`crate::solvers::penalty`]
/// (abelian twin [`crate::solvers::riemannian::riemannian_cg_penalized`]): `riemannian_cg` against
/// `H` plus the overlap penalty `w Σⱼ|⟨ψ̂ⱼ,x⟩|²`, radial shift `J`, every trial scored by `J`.
pub fn riemannian_cg_penalized<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    penalty: &Penalty<Su2TtVector<T>>,
    opts: &RiemannianOpts,
) -> (f64, Su2TtVector<T>) {
    cg_impl(h, x0, opts, penalty.active())
}

fn cg_impl<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &RiemannianOpts,
    penalty: Option<&Penalty<Su2TtVector<T>>>,
) -> (f64, Su2TtVector<T>) {
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
    let mut exit = "max_iter";

    for it in 0..opts.max_iter {
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
            exit = "gnorm";
            break;
        }
        iters_run = it + 1;

        let mut slope = grad.dot(&eta).re_s();
        let reset = slope >= 0.0;
        if reset {
            eta = grad.scale(T::from_f64(-1.0));
            slope = -gnorm * gnorm;
        }

        let lambda_known;
        let mut pre: Option<GradientAt<T>>;
        let search = |eta: &Su2TtTangent<T>, slope: f64| {
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
        if found.is_none() && !reset && it > 0 && slope_floor() {
            crate::solvers::riemannian::ls_stats::bump(
                &crate::solvers::riemannian::ls_stats::SD_RETRIES,
            );
            eta = grad.scale(T::from_f64(-1.0));
            slope = -gnorm * gnorm;
            found = search(&eta, slope);
        }
        match found {
            None => {
                exit = "linesearch";
                break;
            }
            Some((a_n, x_n, l_n, p_n)) => {
                pre = p_n;
                alpha = a_n;
                x = x_n;
                lambda_known = Some(l_n);
            }
        }

        let (bl_n, br_n, gradn, l) = match pre.take() {
            Some(p) => {
                crate::solvers::riemannian::ls_stats::bump(
                    &crate::solvers::riemannian::ls_stats::CACHE_HITS,
                );
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
                    mix_seed(opts.seed, it as u64 + 1),
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

        let eta_t = Su2TtTangent::project(&base_l, &base_r, &eta.to_vector());
        let beta = match opts.variant {
            CgVariant::Fr => gnormn * gnormn / (gnorm * gnorm),
            CgVariant::Pr => {
                let g_t = Su2TtTangent::project(&base_l, &base_r, &grad.to_vector());
                let num = gradn.dot(&gradn.sub(&g_t)).re_s();
                (num / (gnorm * gnorm)).max(0.0)
            }
        };
        if trace {
            eprintln!(
                "CG step it={it} alpha={alpha:.3e} r={:.4} beta={beta:.4e} reset={reset}",
                gradn.dot(&eta_t).re_s() / slope
            );
        }
        eta = gradn
            .scale(T::from_f64(-1.0))
            .add(&eta_t.scale(T::from_f64(beta)));
        grad = gradn;
        gnorm = gnormn;
    }
    if trace {
        eprintln!(
            "CG exit={exit} after {iters_run}/{} iterations, ‖g‖={gnorm:.6e}, λ={lambda:.16e}",
            opts.max_iter
        );
    }
    (lambda, x)
}

/// Abelian twin [`crate::solvers::riemannian::tangent_bases`]: left- and right-orthogonalized
/// copies of `x` at its own ranks — the pinned left/right-canonical factorizations that gauge-fix every tangent projection at
/// `x`. `pub fn` (the integration binaries call it directly).
pub fn tangent_bases<T: Scalar>(x: &Su2TtVector<T>) -> (Su2TtVector<T>, Su2TtVector<T>) {
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    (base_l, base_r)
}

/// `x ← x / norm_lr(x)` (abelian [`crate::solvers::riemannian::normalize_mut`]),
/// zero-norm-guarded. `pub fn`.
pub fn normalize_mut<T: Scalar>(x: &mut Su2TtVector<T>) {
    let n = x.norm_lr(false);
    if n != 0.0 {
        x.scale_mut(T::from_f64(1.0 / n));
    }
}

/// The Riemannian gradient (abelian [`crate::solvers::riemannian::riemannian_gradient`]):
/// `λ = Re rayleigh_quotient(H,x)`, the residual projected through `Su2TtTangent::project(baseL,
/// baseR, res)` (the `rproject` right-accumulation — the metric enters before this driver).
/// Returns `(grad, λ)`. Exact environment contraction; `tol` and `seed` are inert.
pub fn riemannian_gradient<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    tol: f64,
    seed: u64,
) -> (Su2TtTangent<T>, f64) {
    gradient_impl(h, x, base_l, base_r, tol, seed)
}

/// [`riemannian_gradient`] for the PENALIZED functional (abelian twin
/// [`crate::solvers::riemannian::riemannian_gradient_penalized`]): `P((H − J)x) + w Σⱼ oⱼ P(ψ̂ⱼ)`
/// at the frames `(base_l, base_r)`, and `J` itself. Returns `(grad, J)`.
#[allow(clippy::too_many_arguments)]
pub fn riemannian_gradient_penalized<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    penalty: &Penalty<Su2TtVector<T>>,
    tol: f64,
    seed: u64,
) -> (Su2TtTangent<T>, f64) {
    gradient_at(h, x, base_l, base_r, tol, seed, None, penalty.active())
}

/// The EXACT gate `‖P((H − λ)x)‖` at unit-norm `x`, `λ = ⟨x,Hx⟩` — one environment-contracted
/// gradient at fresh [`tangent_bases`], no `H·x`. Abelian twin
/// [`crate::solvers::riemannian::exact_gradnorm`]; coefficient for coefficient the gate
/// `crate::su2::hess::weingarten_operator(h, x).gradient().norm()` reports.
pub fn exact_gradnorm<T: Scalar>(h: &Su2SparseHamiltonian<T>, x: &Su2TtVector<T>) -> f64 {
    let (base_l, base_r) = tangent_bases(x);
    let lambda = rayleigh_quotient(h, x).re_s();
    crate::su2::hess::riemannian_gradient_env(h, &base_l, &base_r, lambda).norm()
}

/// The exact gate of the PENALIZED functional at unit-norm `x`:
/// `‖P((H − J)x) + w Σⱼ oⱼ P(ψ̂ⱼ)‖`, by environment contraction — no `H·x`. Abelian twin
/// [`crate::solvers::riemannian::penalized_exact_gradnorm`]. An INACTIVE penalty takes
/// [`exact_gradnorm`]'s branch exactly.
pub fn penalized_exact_gradnorm<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    penalty: &Penalty<Su2TtVector<T>>,
) -> f64 {
    let Some(p) = penalty.active() else {
        return exact_gradnorm(h, x);
    };
    let (base_l, base_r) = tangent_bases(x);
    let os = p.overlaps(x);
    let j = rayleigh_quotient(h, x).re_s() + p.energy(&os);
    let mut grad = crate::su2::hess::riemannian_gradient_env(h, &base_l, &base_r, j);
    p.tangent_axpy(&mut grad, &os, |psi| {
        Su2TtTangent::project(&base_l, &base_r, psi)
    });
    grad.norm()
}

fn gradient_impl<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    tol: f64,
    seed: u64,
) -> (Su2TtTangent<T>, f64) {
    gradient_at(h, x, base_l, base_r, tol, seed, None, None)
}

#[allow(clippy::too_many_arguments)]
fn gradient_at<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    tol: f64,
    seed: u64,
    lambda_known: Option<f64>,
    penalty: Option<&Penalty<Su2TtVector<T>>>,
) -> (Su2TtTangent<T>, f64) {
    let _ = (tol, seed);
    let os = penalty.map(|p| p.overlaps(x));
    let lambda = match lambda_known {
        Some(l) => l,
        _ => match (penalty, &os) {
            (Some(p), Some(os)) => rayleigh_quotient(h, x).re_s() + p.energy(os),
            _ => rayleigh_quotient(h, x).re_s(),
        },
    };
    let mut grad = crate::su2::hess::riemannian_gradient_env(h, base_l, base_r, lambda);
    if let (Some(p), Some(os)) = (penalty, &os) {
        p.tangent_axpy(&mut grad, os, |psi| {
            Su2TtTangent::project(base_l, base_r, psi)
        });
    }
    (grad, lambda)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn linesearch<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    eta: &Su2TtTangent<T>,
    lambda0: f64,
    slope: f64,
    alpha0: f64,
    gnorm0: f64,
    gdot: Option<f64>,
) -> Option<(f64, Su2TtVector<T>, f64)> {
    linesearch_impl(h, x, eta, lambda0, slope, alpha0, gnorm0, gdot, None)
}

type GradientAt<T> = (Su2TtVector<T>, Su2TtVector<T>, Su2TtTangent<T>, f64);

fn floor_slope_at<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    xt: &Su2TtVector<T>,
    lt: f64,
    w: &Su2TtVector<T>,
    penalty: Option<&Penalty<Su2TtVector<T>>>,
) -> (f64, GradientAt<T>) {
    let (bl, br) = tangent_bases(xt);
    let mut g = crate::su2::hess::riemannian_gradient_env(h, &bl, &br, lt);
    if let Some(p) = penalty {
        p.tangent_axpy(&mut g, &p.overlaps(xt), |psi| {
            Su2TtTangent::project(&bl, &br, psi)
        });
    }
    crate::solvers::riemannian::ls_stats::bump(&crate::solvers::riemannian::ls_stats::GRAD_EVALS);
    let phi = g.dot(&Su2TtTangent::project(&bl, &br, w)).re_s();
    (phi, (bl, br, g, lt))
}

type LsCached<T> = (f64, Su2TtVector<T>, f64, Option<GradientAt<T>>);

#[allow(clippy::too_many_arguments)]
fn linesearch_impl<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    eta: &Su2TtTangent<T>,
    lambda0: f64,
    slope: f64,
    alpha0: f64,
    gnorm0: f64,
    gdot: Option<f64>,
    penalty: Option<&Penalty<Su2TtVector<T>>>,
) -> Option<(f64, Su2TtVector<T>, f64)> {
    linesearch_cached(
        h, x, eta, lambda0, slope, alpha0, gnorm0, gdot, penalty, false, false,
    )
    .map(|(a, xa, la, _)| (a, xa, la))
}

#[allow(clippy::too_many_arguments)]
fn linesearch_cached<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    eta: &Su2TtTangent<T>,
    lambda0: f64,
    slope: f64,
    alpha0: f64,
    gnorm0: f64,
    gdot: Option<f64>,
    penalty: Option<&Penalty<Su2TtVector<T>>>,
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
    crate::solvers::riemannian::ls_stats::bump(&crate::solvers::riemannian::ls_stats::CALLS);
    let mut noise_entered = false;
    let mut grad_trials = 0usize;
    let mut best_alpha = 0.0f64;
    let mut best_x: Option<Su2TtVector<T>> = None;
    let mut best_lambda = lambda0;
    use crate::solvers::riemannian::ls_stats;
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
            let logged = |t: SlopeTrial<(Su2TtVector<T>, GradientAt<T>)>| {
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
                crate::solvers::riemannian::ls_stats::bump(
                    &crate::solvers::riemannian::ls_stats::FLOOR_ENTRIES,
                );
            }
            if bt == 0 {
                crate::solvers::riemannian::ls_stats::bump(
                    &crate::solvers::riemannian::ls_stats::FIRST_ACCEPTS,
                );
                if !crate::solvers::riemannian::floor_secant() {
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
                    crate::solvers::riemannian::ls_stats::bump(
                        &crate::solvers::riemannian::ls_stats::TRIALS,
                    );
                    crate::solvers::riemannian::ls_stats::bump(
                        &crate::solvers::riemannian::ls_stats::SECANT_TRIALS,
                    );
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
                    crate::solvers::riemannian::ls_stats::bump(
                        &crate::solvers::riemannian::ls_stats::SECANT_ACCEPTS,
                    );
                }
                return Some((best_a, best_x, best_l, Some(best_pre)));
            }
            if grad_trials < MAX_GRAD_TRIALS {
                grad_trials += 1;
                crate::solvers::riemannian::ls_stats::bump(
                    &crate::solvers::riemannian::ls_stats::GRAD_EVALS,
                );
                let (bl, br) = tangent_bases(&xa);
                let ga = match penalty {
                    None => crate::su2::hess::riemannian_gradient_env(h, &bl, &br, la).norm(),
                    Some(p) => {
                        let mut gt = crate::su2::hess::riemannian_gradient_env(h, &bl, &br, la);
                        p.tangent_axpy(&mut gt, &p.overlaps(&xa), |psi| {
                            Su2TtTangent::project(&bl, &br, psi)
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
        crate::solvers::riemannian::ls_stats::bump(&crate::solvers::riemannian::ls_stats::NONE);
    }
    best_x.map(|xb| (best_alpha, xb, best_lambda, None))
}

#[cfg(test)]
mod s6d_tests {
    use crate::coeff::CoeffTensor;
    use crate::scalar::Scalar;
    use crate::solvers::penalty::Penalty;
    use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient, xt_h_y};
    use crate::su2::ops::dot;
    use crate::su2::qn::Su2Space;
    use crate::su2::riemannian::{
        RiemannianOpts, cg_impl, exact_gradnorm, gd_impl, gradient_at, linesearch_impl,
        normalize_mut, penalized_exact_gradnorm, riemannian_cg, riemannian_gd, tangent_bases,
    };
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randomized_su2_state, seeded};

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

    fn unit_state(space: Su2Space, lo: usize, hi: usize, seed: u64) -> Su2TtVector<f64> {
        let mut rng = seeded(seed);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
        x.round_mut(1e-14, crate::qn::Pooled::NONE, crate::round::RoundMode::Abs);
        normalize_mut(&mut x);
        x
    }

    #[test]
    fn s6d_polish_passes_the_energy_floor() {
        let d = 6usize;
        let space = Su2Space::new(d, 3, 1);
        let (t, v) = generic_tv(d);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
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

        let x0 = unit_state(space, 1, 2, 0xD2_1121);
        let (la, xa) = riemannian_cg(&h, &x0, &with_eps(1e-6));
        let (l1, x1) = riemannian_cg(&h, &xa, &with_eps(1e-9 / la.abs()));
        let g1 = exact_gradnorm(&h, &x1);
        println!("s6d floor: λa = {la:.16e}, λ1 = {l1:.16e}, ‖g1‖ = {g1:.3e}");
        assert!(g1 <= 2e-9, "the staging polish stopped at ‖g‖ = {g1:.3e}");

        let (bl, br) = tangent_bases(&x1);
        let (g, l) = gradient_at(&h, &x1, &bl, &br, 1e-12, base.seed, None, None);
        let gn = g.norm();
        let slope = -gn * gn;
        let (a, _, lt) = linesearch_impl(
            &h,
            &x1,
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
            "s6d floor: α·|slope| = {:.3e} vs floor {:.3e}, measured dλ = {:.3e}",
            a * slope.abs(),
            efloor(l),
            lt - l
        );
        assert!(
            a * slope.abs() < efloor(l),
            "the fixture does not reach the noise regime (α·|slope| = {:.3e} ≥ {:.3e})",
            a * slope.abs(),
            efloor(l)
        );

        let deep = with_eps(1e-12 / l1.abs());
        let (l2, x2) = riemannian_cg(&h, &x1, &deep);
        let g2 = exact_gradnorm(&h, &x2);
        let (lg, xg) = riemannian_gd(&h, &x1, &deep);
        let gg = exact_gradnorm(&h, &xg);
        println!("s6d floor: CG λ2 = {l2:.16e}, ‖g2‖ = {g2:.3e}; GD λ = {lg:.16e}, ‖g‖ = {gg:.3e}");
        assert!(
            g2 <= 1e-11,
            "CG stopped at ‖g‖ = {g2:.3e} (old floor ≈ 1e-9)"
        );
        assert!(
            l2 <= l1 + efloor(l1),
            "CG gave up energy below the floor ({:.3e})",
            l2 - l1
        );
        assert!(
            gg <= 1e-10,
            "GD stopped at ‖g‖ = {gg:.3e} (old floor ≈ 1e-9)"
        );
        assert!(
            lg <= l1 + efloor(l1),
            "GD gave up energy below the floor ({:.3e})",
            lg - l1
        );

        let p = Penalty::new(1.0, std::slice::from_ref(&x2));
        let xp0 = unit_state(space, 1, 2, 0xD2_1122);
        let (jp1, xp1) = cg_impl(&h, &xp0, &with_eps(1e-6), p.active());
        let (jp2, xp2) = cg_impl(&h, &xp1, &with_eps(1e-12 / jp1.abs()), p.active());
        let gp = penalized_exact_gradnorm(&h, &xp2, &p);
        println!("s6d floor: penalized J = {jp2:.16e}, ‖g‖ = {gp:.3e}");
        assert!(
            gp <= 1e-11,
            "penalized CG stopped at ‖g‖ = {gp:.3e} (old floor ≈ 1e-9)"
        );
    }

    #[test]
    fn s6d_slope_rule_polishes_through_the_floor() {
        let d = 6usize;
        let space = Su2Space::new(d, 3, 1);
        let (t, v) = generic_tv(d);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
        let efloor = |l: f64| super::ENERGY_NOISE_C * f64::EPSILON * (1.0 + l.abs());
        let opts = |eps: f64| RiemannianOpts {
            eps,
            max_iter: 500,
            gradtol: 1e-12,
            ..RiemannianOpts::new(&space)
        };
        let x0 = unit_state(space, 1, 2, 0xD2_1121);
        let (la, xa) = riemannian_cg(&h, &x0, &opts(1e-6));
        let (l2, _) = riemannian_cg(&h, &xa, &opts(1e-12 / la.abs()));
        let budget = 3000;
        let mut handed = [0usize; 2];
        for (k, verify) in [false, true].into_iter().enumerate() {
            let (mut x, mut known, mut pre, mut alpha, mut n) = (xa.clone(), None, None, 1.0, 0);
            let mut it = 0;
            let (l, g) = loop {
                let (g, l) = match pre.take() {
                    Some((_, _, g, l)) => (g, l),
                    None => {
                        let (bl, br) = tangent_bases(&x);
                        gradient_at(&h, &x, &bl, &br, 1e-12, 0, known, None)
                    }
                };
                let gn = g.norm();
                if gn <= 1e-12 || it == budget {
                    break (l, gn);
                }
                let slope = -gn * gn;
                let step = super::linesearch_cached(
                    &h,
                    &x,
                    &g.scale(-1.0),
                    l,
                    slope,
                    1.0f64.min(2.0 * alpha),
                    gn,
                    Some(slope),
                    None,
                    true,
                    verify,
                );
                let (a, xn, ln, p) = step.unwrap_or_else(|| {
                    panic!("line search returned None at it {it}, ‖g‖ {gn:.3e}")
                });
                n += usize::from(p.is_some());
                (alpha, x, known, pre) = (a, xn, Some(ln), p);
                it += 1;
            };
            println!(
                "s6d slope rule verify={verify}: λ = {l:.16e} (CG {l2:.16e}), {it} it, \
                 ‖g‖ = {g:.3e}, {n} gradients handed over"
            );
            assert!(
                g <= 1e-12,
                "slope-rule GD stopped at ‖g‖ = {g:.3e} after {it} it"
            );
            assert!(
                (l - l2).abs() <= efloor(l),
                "slope-rule GD ended {:.3e} from the shipped CG's energy",
                l - l2
            );
            handed[k] = n;
        }
        assert!(
            handed[1] > handed[0],
            "verify handed over no more gradients ({handed:?}): the extension did not run"
        );
    }

    #[test]
    fn s6d_penalty_gdot_shortcut_matches_exact_c1() {
        let d = 6usize;
        let space = Su2Space::new(d, 2, 0);
        let (t, v) = generic_tv(d);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
        let x = unit_state(space, 3, 5, 0xD2_1101);
        let psi = unit_state(space, 2, 4, 0xD2_1102);
        let p = Penalty::new(2.5, std::slice::from_ref(&psi));
        assert!(p.is_active(), "the fixture's penalty must be active");

        let (base_l, base_r) = tangent_bases(&x);
        let (g, j) = gradient_at(&h, &x, &base_l, &base_r, 1e-12, 1234, None, Some(&p));
        let eta = g.scale(-1.0);
        let gnorm = g.norm();
        let slope = g.dot(&eta).re_s();

        let wv = eta.to_vector();
        let s1 = dot(&x, &wv).re_s();
        let (dc1, _dc2) = p.linesearch_coeffs(&x, &wv);
        let c1_shortcut = slope + j * s1;
        let c1_exact = xt_h_y(&x, &h, &wv).re_s() + dc1;
        println!(
            "s6d gdot: c1_shortcut = {c1_shortcut:.16e}, ⟨x,Hι(η)⟩ + Δc₁ = {c1_exact:.16e}, \
             Δc₁ = {dc1:.6e}"
        );
        assert!(
            dc1.abs() > 1e-6 * c1_exact.abs().max(1.0),
            "Δc₁ = {dc1:.3e} is negligible against c₁ = {c1_exact:.3e}; this arm would not \
             discriminate a double count"
        );
        assert!(
            (c1_shortcut - c1_exact).abs() <= 1e-9 * c1_exact.abs().max(1.0),
            "the gdot shortcut is not ⟨x,Hι(η)⟩ + Δc₁ ({:.3e})",
            (c1_shortcut - c1_exact).abs()
        );

        let (a_s, _, l_s) =
            linesearch_impl(&h, &x, &eta, j, slope, 1.0, gnorm, Some(slope), Some(&p))
                .expect("the shortcut branch found no decrease");
        let (a_e, _, l_e) = linesearch_impl(&h, &x, &eta, j, slope, 1.0, gnorm, None, Some(&p))
            .expect("the exact-c1 branch found no decrease");
        println!("s6d gdot: α {a_s:.12e} vs {a_e:.12e}, J⁺ {l_s:.16e} vs {l_e:.16e}");
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
    fn s6d_penalty_gd_tracks_the_penalized_functional() {
        let _lock = crate::par::lock_par_switches();
        let d = 6usize;
        let space = Su2Space::new(d, 2, 0);
        let (t, v) = generic_tv(d);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
        let x = unit_state(space, 3, 5, 0xD2_1111);
        let psi = unit_state(space, 2, 4, 0xD2_1112);
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
            "s6d gd-pen: J {j0:.12e} → {j:.12e} (λ {lam:.12e} + pen {pen:.6e}), gate \
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
            "an inactive penalty changed the SU(2) GD energy ({jb:.17e} vs {lb:.17e})"
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
                    "an inactive penalty changed SU(2) GD core {k} coefficient {i}"
                );
            }
        }
    }
}
