// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Layer traits for the shared Krylov solvers: the Rust two
//! layer-hook traits. `lanczos`, `inexact_arnoldi`, `sketched_rayleigh_ritz`, and `sketched_gmres`
//! are written against them so a second symmetry layer (SU(2)) reuses them unchanged; the sweep
//! drivers (`als`, `mals`, `tdvp`, `riemannian`) stay concrete — their kernel surface (sketching
//! steps, workspaces, range-table caching) is layer-local and is duplicated per layer.

use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
use crate::qn::Pooled;
use crate::round::RoundMode;
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// The layer-agnostic TT-state surface the shared Krylov solvers are written against.
/// Implementors: the abelian [`TtVector`] and the SU(2) `Su2TtVector`.
pub trait TtState: Clone {
    /// Scalar field (`f64` or `Complex<f64>`); the crate [`Scalar`].
    type Scalar: Scalar;

    /// Number of sites `d`.
    fn num_sites(&self) -> usize;

    /// Largest single per-sector rank over all bonds (a `lanczos` diagnostic).
    fn max_sector_rank(&self) -> usize;

    /// Max over bonds of the summed-over-sector bond dimension.
    fn max_total_bond_rank(&self) -> usize;

    /// Hermitian inner product `⟨self, other⟩` in the layer metric, conjugating `self`.
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// Layer norm `sqrt(Re⟨self,self⟩)`.
    fn norm(&self) -> f64;

    /// `self *= a`.
    fn scale_mut(&mut self, a: Self::Scalar);

    /// `self · a` (owned; `&x * a`).
    fn scaled(&self, a: Self::Scalar) -> Self;

    /// `self + other` (owned; `&x + &y`).
    fn add(&self, other: &Self) -> Self;

    /// `self - other` (owned; `&x - &y`).
    fn sub(&self, other: &Self) -> Self;

    /// Left-orthonormalization sweep returning the per-bond norm factors.
    fn left_orthonormalize(&mut self, keep_rank: bool) -> Vec<f64>;

    /// In-place SVD rounding; `chi` is a POOLED cap on each bond's TOTAL dimension
    /// `Σ_q r[q]` ([`Pooled::NONE`] = uncapped).
    fn round_mut(&mut self, eps: f64, chi: Pooled, mode: RoundMode);

    /// Owned SVD rounding (`round`); `chi` is POOLED, as in [`Self::round_mut`].
    fn rounded(&self, eps: f64, chi: Pooled, mode: RoundMode) -> Self;

    /// Deterministic weighted sum with rounding.
    fn round_sum(alpha: &[Self::Scalar], summands: &[Self], tol: f64) -> Self;
}

/// The layer-agnostic Hamiltonian surface for the shared Krylov solvers. The generic call sites
/// never pass an operator cache, so the adaptive verbs build theirs internally.
pub trait TtHamiltonian<V: TtState> {
    /// `H·x` (materialized matvec).
    fn matvec(&self, x: &V) -> V;

    /// `⟨x, H·x⟩` — the UNNORMALIZED quadratic form, despite the name. Callers holding a
    /// non-unit `x` must divide by `⟨x, x⟩` themselves.
    fn rayleigh_quotient(&self, x: &V) -> V::Scalar;
}

impl<T: Scalar> TtState for TtVector<T> {
    type Scalar = T;

    fn num_sites(&self) -> usize {
        self.space.d
    }

    fn max_sector_rank(&self) -> usize {
        TtVector::max_sector_rank(self)
    }

    fn max_total_bond_rank(&self) -> usize {
        TtVector::max_total_bond_rank(self)
    }

    fn dot(&self, other: &Self) -> T {
        crate::ops::dot(self, other)
    }

    fn norm(&self) -> f64 {
        TtVector::norm(self)
    }

    fn scale_mut(&mut self, a: T) {
        TtVector::scale_mut(self, a)
    }

    fn scaled(&self, a: T) -> Self {
        self * a
    }

    fn add(&self, other: &Self) -> Self {
        self + other
    }

    fn sub(&self, other: &Self) -> Self {
        self - other
    }

    fn left_orthonormalize(&mut self, keep_rank: bool) -> Vec<f64> {
        TtVector::left_orthonormalize(self, keep_rank)
    }

    fn round_mut(&mut self, eps: f64, chi: Pooled, mode: RoundMode) {
        TtVector::round_mut(self, eps, chi, mode)
    }

    fn rounded(&self, eps: f64, chi: Pooled, mode: RoundMode) -> Self {
        TtVector::round(self, eps, chi, mode)
    }

    fn round_sum(alpha: &[T], summands: &[Self], tol: f64) -> Self {
        crate::round::round_sum(alpha, summands, tol)
    }
}

impl<T: Scalar> TtHamiltonian<TtVector<T>> for SparseHamiltonian<T> {
    fn matvec(&self, x: &TtVector<T>) -> TtVector<T> {
        self * x
    }

    fn rayleigh_quotient(&self, x: &TtVector<T>) -> T {
        rayleigh_quotient(self, x)
    }
}
