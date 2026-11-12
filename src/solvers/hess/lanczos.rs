// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;

/// A converged **negative** Ritz value `θ = ω² < 0` of the product operator, carrying the
/// fresh-apply variational certificate that decides whether it is physics or a solver artifact.
#[derive(Clone, Copy, Debug)]
pub struct NegativeWitness {
    /// Index into the returned signed-`ω²` list.
    pub index: usize,
    /// The Ritz value as read off the projected matrix: `θ = ω²` on the RPA readouts, an energy
    /// relative to `λ` on the TDA readouts (`tda_lowest_davidson`, `SectorCtx::sector_lowest_tda`).
    pub theta: f64,
    /// `⟨u, R u⟩ / ⟨v, u⟩` with `u = W v` and both applies recomputed FRESH on the reprojected
    /// Ritz vector `v`. The denominator `⟨v, W v⟩ = ‖v‖²_W > 0`, so this quantity is on `θ`'s
    /// scale and shares its sign whenever the projected problem is sound.
    pub fresh: f64,
    /// The fresh W-Rayleigh quotient `⟨v, W v⟩ / ‖v‖²` of the same Ritz vector. Negative only on
    /// the signed Davidson's signature −1 entries, where the vector is itself a variational witness
    /// that the METRIC block is indefinite.
    pub fresh_w: f64,
}

impl NegativeWitness {
    /// **The gate.** A negative Ritz value may be reported as a resonance only when the
    /// independent fresh-apply quadratic form also comes out negative. A positive `fresh` beside
    /// a negative `theta` is a solver artifact — a corrupted projected problem — and must be
    /// refused, not published.
    pub fn certified(&self) -> bool {
        self.fresh < 0.0
    }
}

/// Zero-apply health counters of the BLOCK W-metric eigensolvers (Davidson / ELOBP4dCG), recorded
/// to diagnose *why* a run stalled before choosing a preconditioner repair rather than after.
#[derive(Clone, Debug, Default)]
pub struct SympDiagnostics {
    /// Outer (Rayleigh–Ritz) iterations performed.
    pub outer_iterations: usize,
    /// Thick restarts fired (always `0` for the restart-free ELOBP4dCG kernel, which has none).
    pub restarts: usize,
    /// Candidates dropped by the rank-revealing test (relative g-norm `≤ 1e-10` after the two-pass
    /// MGS): the direction was already in the basis.
    pub drops_rank: usize,
    /// Candidates dropped at the W-seminorm floor (healthy g-norm, W-norm down at the
    /// `1e3·ε·max|θ|` floor): nothing usable along the direction in the W metric.
    pub drops_wnull: usize,
    /// Candidates arriving with exactly zero g-norm after deflation + reprojection.
    pub drops_zero: usize,
    /// Retained/previous-direction triples dropped by the zero-apply triple
    /// re-orthonormalization — any of its four exits, the sanctioned cancellation-roundoff PD drop
    /// included by design (see that helper's doc). A high count on the `P′` path is
    /// basis-selection information, not an error.
    pub drops_triple: usize,
    /// MGS survival ratio `‖candidate‖_g after / before` per ACCEPTED basis vector, warm-start
    /// seeds included; `len() == SymplecticLowestInfo::iterations`.
    pub survival: Vec<f64>,
    /// `|⟨t, x⟩| / (‖t‖·‖x‖)` per ATTEMPTED correction `t` against its own Ritz vector `x`,
    /// recorded AFTER the preconditioner (identity during the `nwarm` warm-up) and BEFORE the
    /// acceptance push; `NaN` when a factor's norm is zero, so a zero-norm attempt stays visible
    /// without being counted as collapse evidence. Values ≈ 1 are the collapse signature.
    pub collapse_angle: Vec<f64>,
    /// Smallest eigenvalue of the symmetrized BASIS W-Gram `G_ij = Re⟨w_i, v_j⟩` — the matrix the
    /// kernels ASSUME equals `I` — one entry per outer iteration.
    pub gram_min_eig: Vec<f64>,
    /// Where the SURVIVAL GUARD fired, if it did: `Some(n)` says the preconditioner closure was
    /// permanently discarded at the moment `n` basis vectors had been accepted (`n =
    /// survival.len()` at the switch — a COUNT, not a 0-based index, so `Some(0)` is unreachable:
    /// the guard needs a full trailing window before it can decide).
    pub guard_tripped: Option<usize>,
}

/// One non-positive W-norm² met by the signed Davidson's push: the candidate was accepted with
/// Krein signature −1, or dropped in the negative null band `−τ‖u‖² ≤ β² < 0`. These are the numbers
/// a `MetricNotPd` refusal used to drop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricEvent {
    /// Basis dimension the candidate was orthogonalized against.
    pub basis_len: usize,
    /// g-norm of the candidate before the two-pass MGS (after its first reprojection).
    pub gn_before: f64,
    /// g-norm after the MGS and the second reprojection.
    pub gn_after: f64,
    /// `⟨u, W u⟩` of the cleaned candidate.
    pub wnorm2: f64,
    /// `true` = kept with signature −1; `false` = dropped in the null band.
    pub accepted: bool,
}

/// A complex Ritz pair of the signed pencil (the `Im θ > 0` member of a conjugate pair) — only
/// when BOTH blocks are indefinite on the Krylov span (`θ = ω²` complex: `ω` a complex quartet).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComplexRitz {
    pub re: f64,
    pub im: f64,
    /// True residual g-norm `‖M x − θ x‖` of the pair.
    pub residual: f64,
}

/// What a signed solve met that a positive-definite metric would not have: every
/// non-positive-W candidate, how many signature −1 vectors the basis held at most, whether the
/// projected problem left the definite path (the OUTER factor witnessed indefinite), and the
/// complex pairs. `SymplecticLowestInfo::indefinite` is `Some` exactly when `events` is non-empty.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct IndefiniteReport {
    pub events: Vec<MetricEvent>,
    pub n_negative_basis: usize,
    pub general_path: bool,
    pub complex: Vec<ComplexRitz>,
}

/// Diagnostics returned with the iterative ω — the honesty contract of the identity-
/// preconditioned path: iteration/apply counts and residuals are reported, never hidden.
#[derive(Clone, Debug)]
pub struct SymplecticLowestInfo {
    /// All wanted residual estimates met `tol` (false = `max_iter` hit; the returned ω are the
    /// best Rayleigh–Ritz values so far and `residuals` says how converged they are).
    pub converged: bool,
    /// Krylov dimension built.
    pub iterations: usize,
    /// Number of Hessian-class applies consumed (each = one `a_apply` plus one `b_apply`); 2 per
    /// iteration + 1 for the warm start.
    pub applies: usize,
    /// Final TRUE residual g-norms `‖M x_j − θ_j x_j‖_g` for the returned pairs, in ω² units
    /// (both kernels report the same quantity).
    pub residuals: Vec<f64>,
    /// One entry per returned `ω² < 0`, with its fresh-apply certificate. Empty on a stable
    /// point, which is the overwhelmingly common case — so the two extra applies per negative
    /// value cost nothing where nothing is negative.
    pub negative: Vec<NegativeWitness>,
    /// Per-run solver-health counters: `Some` from every kernel (Davidson, signed Davidson,
    /// ELOBP4dCG). See [`SympDiagnostics`].
    pub diag: Option<SympDiagnostics>,
    /// Krein signature `sign(x_jᵀ W x_j)` of each returned pair, in the same order as the values:
    /// `+1` throughout from the refusing kernels (their metric is positive definite on the basis
    /// or they refuse); `−1` marks a signed-Davidson instability met on an indefinite metric, `0`
    /// a Krein-neutral pair (general path only).
    pub signature: Vec<i8>,
    /// `Some` exactly when a signed solve met an indefinite metric (see [`IndefiniteReport`]);
    /// `None` from every refusing kernel and from a signed solve whose metric stayed definite.
    pub indefinite: Option<IndefiniteReport>,
}

pub(crate) fn certify_negatives<V, R, W, D, Rp, F>(
    theta: &[f64],
    k: usize,
    mut ritz: F,
    r_apply: &mut R,
    w_apply: &mut W,
    deflate: &mut D,
    reproject: &mut Rp,
) -> (Vec<NegativeWitness>, usize)
where
    V: InnerProductVector,
    R: FnMut(&V) -> V,
    W: FnMut(&V) -> V,
    D: FnMut(&mut V),
    Rp: FnMut(&mut V),
    F: FnMut(usize) -> Option<V>,
{
    let scale = theta
        .iter()
        .take(k)
        .fold(0.0f64, |a, &e| a.max(e.abs()))
        .max(f64::MIN_POSITIVE);
    let mut out = Vec::new();
    let mut applies = 0usize;
    for (index, &th) in theta.iter().enumerate().take(k) {
        if th >= -1e-8 * scale {
            continue;
        }
        let Some(mut v) = ritz(index) else { continue };
        reproject(&mut v);
        let mut u = w_apply(&v);
        applies += 1;
        deflate(&mut u);
        let wn2 = v.dot(&u).re_s();
        let fresh_w = wn2 / v.norm().powi(2).max(f64::MIN_POSITIVE);
        if wn2 <= 0.0 {
            out.push(NegativeWitness {
                index,
                theta: th,
                fresh: f64::NAN,
                fresh_w,
            });
            continue;
        }
        let ru = r_apply(&u);
        applies += 1;
        out.push(NegativeWitness {
            index,
            theta: th,
            fresh: u.dot(&ru).re_s() / wn2,
            fresh_w,
        });
    }
    (out, applies)
}

/// Instability diagnoses of the iterative path (the Krylov analogue of
/// [`crate::solvers::hess::SymplecticSpectrum::Unstable`] — diagnosis, not a crash;
/// same fallback semantics).
#[derive(Clone, Copy, Debug)]
pub enum SymplecticLowestError {
    /// A Krylov vector's W-norm² came out negative beyond roundoff: the metric operator (`H^ii`
    /// on the real layer, `𝔥` on the complex layer) is not PD on the deflated tangent space —
    /// the critical point is not a stable minimizer.
    MetricNotPd { wnorm2: f64 },
    /// A converged Ritz value of `R∘W` is negative beyond tolerance: the outer factor (`H^rr`,
    /// resp. `−J𝔥J`) is indefinite.
    NegativeRitz { theta: f64 },
}

#[cfg(test)]
mod t8s_certify_negatives_tests {
    use super::{NegativeWitness, certify_negatives};
    use crate::solvers::InnerProductVector;

    #[derive(Clone, Debug)]
    struct V2([f64; 2]);

    impl InnerProductVector for V2 {
        type Scalar = f64;
        fn dot(&self, o: &Self) -> f64 {
            self.0[0] * o.0[0] + self.0[1] * o.0[1]
        }
        fn norm(&self) -> f64 {
            self.dot(self).sqrt()
        }
        fn axpy(&mut self, a: f64, o: &Self) {
            for (x, &y) in self.0.iter_mut().zip(&o.0) {
                *x += a * y;
            }
        }
        fn scale(&mut self, a: f64) {
            for x in &mut self.0 {
                *x *= a;
            }
        }
        fn for_each_coeff_mut<F: FnMut(&mut f64)>(&mut self, mut f: F) {
            for x in &mut self.0 {
                f(x);
            }
        }
        fn zip_coeffs_mut<F: FnMut(&mut f64, f64)>(&mut self, o: &Self, mut f: F) {
            for (x, &y) in self.0.iter_mut().zip(&o.0) {
                f(x, y);
            }
        }
        fn coeff(&self, i: usize) -> f64 {
            self.0[i]
        }
        fn set_coeff(&mut self, i: usize, v: f64) {
            self.0[i] = v;
        }
        fn num_coeffs(&self) -> usize {
            2
        }
    }

    fn apply(m: [f64; 4], v: &V2) -> V2 {
        V2([m[0] * v.0[0] + m[1] * v.0[1], m[2] * v.0[0] + m[3] * v.0[1]])
    }

    const HRR: [f64; 4] = [2.0, 1.0, 1.0, 2.0];
    const HII: [f64; 4] = [0.0, 1.0, 1.0, 0.0];
    const THETA: [f64; 2] = [-1.0, 3.0];

    fn e_minus() -> V2 {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        V2([s, -s])
    }

    #[test]
    fn t8s_certify_negatives_certifies_the_analytic_resonance_at_fresh_equals_theta() {
        let (out, applies) = certify_negatives(
            &THETA,
            2,
            |j| (j == 0).then(e_minus),
            &mut |v: &V2| apply(HII, v),
            &mut |v: &V2| apply(HRR, v),
            &mut |_: &mut V2| {},
            &mut |_: &mut V2| {},
        );
        assert_eq!(applies, 2, "one W-apply and one R-apply per negative value");
        assert_eq!(out.len(), 1, "only the negative entry is certified");
        let NegativeWitness {
            index,
            theta,
            fresh,
            fresh_w,
        } = out[0];
        assert_eq!(index, 0, "the witness indexes the signed ω² list");
        assert_eq!(theta, -1.0);
        assert!(
            (fresh + 1.0).abs() <= 1e-14,
            "fresh must be θ itself here: {fresh} vs −1"
        );
        assert!((fresh_w - 1.0).abs() <= 1e-14, "fresh_w {fresh_w} vs 1");
        assert!(out[0].certified(), "a genuine resonance is CERTIFIED");
    }

    #[test]
    fn t8s_certify_negatives_refuses_a_positive_fresh_form() {
        let (out, _) = certify_negatives(
            &THETA,
            2,
            |j| (j == 0).then(e_minus),
            &mut |v: &V2| apply([-HII[0], -HII[1], -HII[2], -HII[3]], v),
            &mut |v: &V2| apply(HRR, v),
            &mut |_: &mut V2| {},
            &mut |_: &mut V2| {},
        );
        assert_eq!(out.len(), 1);
        assert!((out[0].fresh - 1.0).abs() <= 1e-14, "{:?}", out[0]);
        assert!(
            !out[0].certified(),
            "fresh ≥ 0 beside θ < 0 is a solver artifact and must NOT certify"
        );
    }

    #[test]
    fn t8s_certify_negatives_reports_nan_on_a_non_positive_w_norm() {
        let (out, applies) = certify_negatives(
            &THETA,
            2,
            |j| (j == 0).then(e_minus),
            &mut |v: &V2| apply(HII, v),
            &mut |v: &V2| apply([-HRR[0], -HRR[1], -HRR[2], -HRR[3]], v),
            &mut |_: &mut V2| {},
            &mut |_: &mut V2| {},
        );
        assert_eq!(applies, 1, "the R-apply is never spent on this exit");
        assert!(out[0].fresh.is_nan(), "{:?}", out[0]);
        assert!(!out[0].certified(), "NaN must not certify");
    }

    #[test]
    fn t8s_certify_negatives_is_free_on_a_stable_spectrum() {
        let (out, applies) = certify_negatives(
            &[2.0, 12.0],
            2,
            |_| Some(e_minus()),
            &mut |v: &V2| apply(HII, v),
            &mut |v: &V2| apply(HRR, v),
            &mut |_: &mut V2| {},
            &mut |_: &mut V2| {},
        );
        assert!(out.is_empty());
        assert_eq!(applies, 0);
    }
}
