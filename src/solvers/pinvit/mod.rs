// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Preconditioned inverse iteration (PINVIT) ground-state eigensolver for the block-sparse
//! abelian TT layer.

mod bounds;
mod inner;
mod outer;
pub mod precond;

pub use inner::{
    PinvitDetOpts, PinvitInfo, PinvitMethod, PinvitOpts, RoundCall, pinvit_ground_state,
};
pub use outer::{OpinvitInfo, OpinvitOpts, opinvit_ground_state};
pub use precond::{
    CalibrateOpts, Calibration, ExpSumPrecond, choose_gamma, diagonal_one_particle, gamma_elliptic,
    gamma_for_positive_diagonal, gamma_one_body,
};

use crate::hamiltonian::SparseHamiltonian;
use crate::ops::dot;
use crate::qn::Pooled;
use crate::vector::TtVector;

/// Fixed algorithmic constants of the preconditioned problem, shared by the inner and outer
/// iterations. All are dimensionless except `gamma` (an energy).
#[derive(Clone, Copy, Debug)]
pub struct PinvitConsts {
    /// Elliptic shift `γ` making `H_γ = H + γI` positive definite on the sector.
    pub gamma: f64,
    /// Spectral-gap bound `δ ≥ λ₂ / (λ₂ − λ₁) > 1` on the SHIFTED spectrum (paper §5.4).
    pub delta: f64,
    /// `‖I − A‖₂ ≤ c ∈ (0,1)` with `A = S H_γ S`, the preconditioner-quality constant (eq 4.14).
    /// Sets the `A`-norm/Euclidean equivalence `ρ = (1−c)^{-1/2}‖r‖`, hence every certificate.
    pub c: Option<f64>,
}

impl Default for PinvitConsts {
    fn default() -> Self {
        PinvitConsts {
            gamma: 0.0,
            delta: 10.0,
            c: None,
        }
    }
}

impl PinvitConsts {
    /// The constants for a calibrated preconditioner: `c` comes from the measurement, not from a
    /// declaration. This is the intended way to build them.
    pub fn calibrated(gamma: f64, delta: f64, cal: &Calibration) -> Self {
        PinvitConsts {
            gamma,
            delta,
            c: Some(cal.c),
        }
    }

    pub(crate) fn resolve_c(&self) -> f64 {
        let c = self.c.expect(
            "PinvitConsts::c is unset. It is a property of `A = S H_γ S`, not a tunable: a value \
             that is too small makes every a-posteriori bound WRONG, not conservative. Measure it \
             with `ExpSumPrecond::calibrate` (which also applies the eq-4.11 rescale that makes it \
             small) and build the constants with `PinvitConsts::calibrated`, or pass `Some(_)` \
             deliberately if you are asserting a bound you have established some other way.",
        );
        assert!(
            c > 0.0 && c < 1.0,
            "PinvitConsts::c must lie in (0,1), got {c}: `ρ = (1−c)^{{-1/2}}‖r‖` is meaningless \
             otherwise, and `c ≥ 1` means the preconditioned operator is not close to the identity \
             at all — recalibrate rather than clamping."
        );
        assert!(
            self.delta > 1.0,
            "PinvitConsts::delta must exceed 1 (it bounds λ₂/(λ₂−λ₁) > 1), got {}",
            self.delta
        );
        c
    }
}

/// Which a-posteriori certificate `PinvitOpts::tau` is tested against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PinvitTermination {
    /// Stop on the relative eigenVALUE bound (paper eq 5.18). The certificate for the quantity
    /// callers actually read, and `O(ρ²)` where the eigenvector bound is `O(ρ)` — but it still
    /// needs `c` and a guessed `δ`. Use when the paper's a-priori theory is what is wanted.
    Eigenvalue,
    /// Stop on `‖H_γ y − λ y‖ / ‖y‖` with `y = S x`, in HARTREE. **The default.**
    #[default]
    Residual,
    /// Stop on an EXTRAPOLATED estimate of the remaining eigenvalue error, in Hartree.
    Extrapolated,
    /// Stop on the eigenVECTOR bound (paper Theorem 5.7). Required when the caller consumes the
    /// iterate rather than the energy — in particular by [`opinvit_ground_state`], whose `τ_m` is an
    /// `A`-norm budget on `‖u₁ − yᵐ‖` that the outer rank truncation's quasi-optimality argument is
    /// stated in. Also the right choice for a caller feeding the eigenvector onward.
    Eigenvector,
}

/// Selects how a weighted sum of trains is truncated.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub enum TruncBackend {
    /// Deterministic binary-tree TT-SVD sum-rounding ([`crate::round::round_sum`]).
    #[default]
    DeterministicSvd,
}

fn weighted_sum_norm(alpha: &[f64], summands: &[TtVector<f64>]) -> f64 {
    let m = summands.len();
    let mut acc = 0.0;
    for i in 0..m {
        acc += alpha[i] * alpha[i] * dot(&summands[i], &summands[i]);
        for j in (i + 1)..m {
            acc += 2.0 * alpha[i] * alpha[j] * dot(&summands[i], &summands[j]);
        }
    }
    acc.max(0.0).sqrt()
}

impl TruncBackend {
    /// Round `Σ_i alpha[i]·summands[i]` to a RELATIVE tolerance:
    /// `‖out − Σ α_i s_i‖ ≤ tol_rel·‖Σ α_i s_i‖`.
    pub fn round_weighted_sum_rel(
        &self,
        alpha: &[f64],
        summands: &[TtVector<f64>],
        tol_rel: f64,
        chi: Pooled,
        template: &TtVector<f64>,
    ) -> TtVector<f64> {
        match self {
            TruncBackend::DeterministicSvd => {
                let n = weighted_sum_norm(alpha, summands);
                self.round_weighted_sum_abs(alpha, summands, tol_rel * n, chi, template)
            }
        }
    }

    /// Round `Σ_i alpha[i]·summands[i]` to an ABSOLUTE tolerance:
    /// `‖out − Σ α_i s_i‖ ≤ tol_abs`.
    pub fn round_weighted_sum_abs(
        &self,
        alpha: &[f64],
        summands: &[TtVector<f64>],
        tol_abs: f64,
        chi: Pooled,
        template: &TtVector<f64>,
    ) -> TtVector<f64> {
        let _ = template;
        let mut out = match self {
            TruncBackend::DeterministicSvd => crate::round::round_sum(alpha, summands, tol_abs),
        };
        if !chi.is_none() {
            out.round_mut(tol_abs, chi, crate::round::RoundMode::Abs);
        }
        out
    }

    /// Assemble and round `(H + shift·I)·sx = H·sx + shift·sx` to a RELATIVE tolerance. The
    /// matvec-carrying stage. Same `chi` convention as [`Self::round_weighted_sum_rel`].
    pub fn apply_h_shift_rel(
        &self,
        h: &SparseHamiltonian<f64>,
        sx: &TtVector<f64>,
        shift: f64,
        tol_rel: f64,
        chi: Pooled,
    ) -> TtVector<f64> {
        match self {
            TruncBackend::DeterministicSvd => {
                let hsx = h.apply(sx);
                let s = [hsx, sx.clone()];
                let n = weighted_sum_norm(&[1.0, shift], &s);
                let mut out = crate::round::round_sum(&[1.0, shift], &s, tol_rel * n);
                if !chi.is_none() {
                    out.round_mut(tol_rel * n, chi, crate::round::RoundMode::Abs);
                }
                out
            }
        }
    }

    /// Assemble and round `(H + shift·I)·sx` to an ABSOLUTE tolerance — the Algorithm 2 contract.
    pub fn apply_h_shift_abs(
        &self,
        h: &SparseHamiltonian<f64>,
        sx: &TtVector<f64>,
        shift: f64,
        tol_abs: f64,
        chi: Pooled,
    ) -> TtVector<f64> {
        let mut out = match self {
            TruncBackend::DeterministicSvd => {
                let hsx = h.apply(sx);
                crate::round::round_sum(&[1.0, shift], &[hsx, sx.clone()], tol_abs)
            }
        };
        if !chi.is_none() {
            out.round_mut(tol_abs, chi, crate::round::RoundMode::Abs);
        }
        out
    }
}
