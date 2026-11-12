// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Solver-layer glue: the `InnerProductVector` abstraction the in-house Davidson eigensolver is
//! written against, plus the ALS driver and Davidson submodules.

pub mod als;
pub mod davidson;
pub mod davidson_block;
pub mod envstore;
pub mod excited;
pub mod expmv;
pub mod gad;
pub mod hess;
pub(crate) mod interface;
pub mod joint;
pub mod lanczos;
pub mod layer;
pub mod mals;
pub mod minres;
pub mod moderot;
pub mod multistate;
pub mod overlap_env;
pub mod penalty;
pub mod pinvit;
pub mod riemannian;
pub mod ssmals;
pub mod steihaug;
pub mod tdvp;

pub use layer::{TtHamiltonian, TtState};
pub use pinvit::{
    CalibrateOpts, Calibration, ExpSumPrecond, OpinvitInfo, OpinvitOpts, PinvitConsts,
    PinvitDetOpts, PinvitInfo, PinvitMethod, PinvitOpts, PinvitTermination, RoundCall,
    TruncBackend, choose_gamma, diagonal_one_particle, gamma_elliptic, gamma_for_positive_diagonal,
    gamma_one_body, opinvit_ground_state, pinvit_ground_state,
};
pub use steihaug::{SteihaugInfo, SteihaugStatus, steihaug_cg};

use crate::core::SparseCore;
use crate::core::twosite::ContractedSparseCores;
use crate::scalar::Scalar;

/// Local eigenproblem vector: the algebra Davidson needs, in the package (metric-carrying) inner
/// product, plus coefficient-basis (metric-free) elementwise access for the diagonal
/// preconditioner. Deliberately plain-generic (NOT `dyn`-object-safe):
/// `dot`/`norm` return the scalar/real field and `for_each_coeff_mut`/`zip_coeffs_mut` take a
/// generic closure, neither of which is object-safe. `davidson` is written as
/// `davidson<V: InnerProductVector>(…)`.
pub trait InnerProductVector: Clone {
    /// Scalar field of the inner product (`f64` or `Complex<f64>`); the crate [`Scalar`].
    type Scalar: Scalar;

    /// Package (metric-carrying) Hermitian inner product `⟨self, other⟩`, conjugating `self`.
    /// (Abelian: `SparseCore::dot_cores`.) Precondition: same shape.
    fn dot(&self, other: &Self) -> Self::Scalar;

    /// Package norm `sqrt(Re⟨self,self⟩)`. (Abelian: `SparseCore::norm`.)
    fn norm(&self) -> f64;

    /// `self += a · other`, elementwise over shared coefficients. Precondition: same shape.
    fn axpy(&mut self, a: Self::Scalar, other: &Self);

    /// `self *= a`, every coefficient. (Abelian: `SparseCore::scale_mut`.)
    fn scale(&mut self, a: Self::Scalar);

    /// Visit every coefficient mutably in CANONICAL order. Metric-free (raw entries). Powers
    /// `precond` and `effective_diagonal_bruteforce`.
    fn for_each_coeff_mut<F: FnMut(&mut Self::Scalar)>(&mut self, f: F);

    /// Visit every coefficient by value in CANONICAL order — the read-only twin of
    /// [`Self::for_each_coeff_mut`]. The default clones `self` to reach the mutable visitor; the
    /// tangent implementors (whose clone copies both base frames) override it.
    fn for_each_coeff<F: FnMut(Self::Scalar)>(&self, mut f: F) {
        let mut c = self.clone();
        c.for_each_coeff_mut(|x| f(*x));
    }

    /// Zip two same-shaped vectors' coefficients in CANONICAL order: `f(&mut a_i, b_i)`. Powers
    /// the residual pass `r ← Hu − θ·u` and the diagonal-quotient divide. Precondition: same
    /// shape.
    fn zip_coeffs_mut<F: FnMut(&mut Self::Scalar, Self::Scalar)>(&mut self, other: &Self, f: F);

    /// Read the coefficient at a flat canonical index (`0..num_coeffs()`).
    fn coeff(&self, idx: usize) -> Self::Scalar;

    /// Write the coefficient at a flat canonical index.
    fn set_coeff(&mut self, idx: usize, val: Self::Scalar);

    /// Number of coefficients (sum of present-block sizes).
    fn num_coeffs(&self) -> usize;
}

impl<T: Scalar> InnerProductVector for SparseCore<T> {
    type Scalar = T;

    fn dot(&self, other: &Self) -> T {
        self.dot_cores_par(other)
    }

    fn norm(&self) -> f64 {
        SparseCore::norm_par(self)
    }

    fn axpy(&mut self, a: T, other: &Self) {
        SparseCore::axpy_par(self, a, other)
    }

    fn scale(&mut self, a: T) {
        self.scale_mut_par(a)
    }

    fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, f: F) {
        SparseCore::for_each_coeff_mut(self, f)
    }

    fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, f: F) {
        SparseCore::zip_coeffs_mut(self, other, f)
    }

    fn coeff(&self, idx: usize) -> T {
        SparseCore::coeff(self, idx)
    }

    fn set_coeff(&mut self, idx: usize, val: T) {
        SparseCore::set_coeff(self, idx, val)
    }

    fn num_coeffs(&self) -> usize {
        SparseCore::num_coeffs(self)
    }
}

impl<T: Scalar> InnerProductVector for crate::core::tangent::TtTangent<T> {
    type Scalar = T;

    fn dot(&self, other: &Self) -> T {
        crate::core::tangent::TtTangent::dot(self, other)
    }

    fn norm(&self) -> f64 {
        crate::core::tangent::TtTangent::norm(self)
    }

    fn axpy(&mut self, a: T, other: &Self) {
        crate::core::tangent::TtTangent::axpy(self, a, other)
    }

    fn scale(&mut self, a: T) {
        self.scale_mut(a)
    }

    fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, f: F) {
        crate::core::tangent::TtTangent::for_each_coeff_mut(self, f)
    }

    fn for_each_coeff<F: FnMut(T)>(&self, mut f: F) {
        for c in &self.components {
            c.for_each_coeff(&mut f);
        }
    }

    fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, f: F) {
        crate::core::tangent::TtTangent::zip_coeffs_mut(self, other, f)
    }

    fn coeff(&self, idx: usize) -> T {
        crate::core::tangent::TtTangent::coeff(self, idx)
    }

    fn set_coeff(&mut self, idx: usize, val: T) {
        crate::core::tangent::TtTangent::set_coeff(self, idx, val)
    }

    fn num_coeffs(&self) -> usize {
        crate::core::tangent::TtTangent::num_coeffs(self)
    }
}

impl<T: Scalar> InnerProductVector for ContractedSparseCores<T> {
    type Scalar = T;

    fn dot(&self, other: &Self) -> T {
        self.dot_cores_par(other)
    }

    fn norm(&self) -> f64 {
        ContractedSparseCores::norm_par(self)
    }

    fn axpy(&mut self, a: T, other: &Self) {
        ContractedSparseCores::axpy_par(self, a, other)
    }

    fn scale(&mut self, a: T) {
        self.scale_mut_par(a)
    }

    fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, f: F) {
        ContractedSparseCores::for_each_coeff_mut(self, f)
    }

    fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, f: F) {
        ContractedSparseCores::zip_coeffs_mut(self, other, f)
    }

    fn coeff(&self, idx: usize) -> T {
        ContractedSparseCores::coeff(self, idx)
    }

    fn set_coeff(&mut self, idx: usize, val: T) {
        ContractedSparseCores::set_coeff(self, idx, val)
    }

    fn num_coeffs(&self) -> usize {
        ContractedSparseCores::num_coeffs(self)
    }
}

#[cfg(test)]
mod read_visitor_tests {
    use super::InnerProductVector;
    use crate::core::tangent::TtTangent;
    use crate::qn::QnSpace;
    use crate::test_util::{randomized_state, seeded};

    fn bits_mut<V: InnerProductVector<Scalar = f64>>(v: &V) -> Vec<u64> {
        let mut out = Vec::new();
        v.clone().for_each_coeff_mut(|c| out.push(c.to_bits()));
        out
    }

    #[test]
    fn tangent_read_visitor_matches_mut_walk() {
        let mut rng = seeded(0x7ead);
        let space = QnSpace::new(6, 2, 2);
        let x = randomized_state::<f64, _>(space, 1, 3, &mut rng);
        let y = randomized_state::<f64, _>(space, 1, 3, &mut rng);
        let (bl, br) = crate::solvers::riemannian::tangent_bases(&x);
        let t = TtTangent::project(&bl, &br, &y);
        let mut got = Vec::new();
        InnerProductVector::for_each_coeff(&t, |c: f64| got.push(c.to_bits()));
        assert!(got.len() > 10, "vacuous: {} coefficients", got.len());
        assert_eq!(got, bits_mut(&t), "read visitor vs mutable walk");
        let by_index: Vec<u64> = (0..t.num_coeffs()).map(|i| t.coeff(i).to_bits()).collect();
        assert_eq!(got, by_index, "read visitor vs coeff(i)");
    }
}
