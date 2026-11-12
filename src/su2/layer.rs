// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! `TtState`/`TtHamiltonian`/`InnerProductVector` impls for the SU(2) layer: with these
//! impls, `lanczos`, `inexact_arnoldi`, `sketched_rayleigh_ritz(_schedule)`, `sketched_gmres`,
//! `davidson`, and `expmv` run on SU(2) objects with ZERO edits under `src/solvers/`. Mirrors the
//! abelian impls in `src/solvers/layer.rs`.

use crate::qn::Pooled;
use crate::round::RoundMode;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;
use crate::solvers::layer::{TtHamiltonian, TtState};
use crate::su2::core::Su2Core;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::{SymSpace, Symmetry};
use crate::su2::twosite::Su2ContractedCores;
use crate::su2::vector::Su2TtVector;

impl<T: Scalar, S: Symmetry> TtState for Su2TtVector<T, S> {
    type Scalar = T;

    fn num_sites(&self) -> usize {
        self.space.sites()
    }

    fn max_sector_rank(&self) -> usize {
        Su2TtVector::max_sector_rank(self)
    }

    fn max_total_bond_rank(&self) -> usize {
        Su2TtVector::max_total_bond_rank(self)
    }

    fn dot(&self, other: &Self) -> T {
        crate::su2::ops::dot(self, other)
    }

    fn norm(&self) -> f64 {
        Su2TtVector::norm(self)
    }

    fn scale_mut(&mut self, a: T) {
        Su2TtVector::scale_mut(self, a)
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
        Su2TtVector::left_orthonormalize(self, keep_rank)
    }

    fn round_mut(&mut self, eps: f64, chi: Pooled, mode: RoundMode) {
        Su2TtVector::round_mut(self, eps, chi, mode)
    }

    fn rounded(&self, eps: f64, chi: Pooled, mode: RoundMode) -> Self {
        Su2TtVector::round(self, eps, chi, mode)
    }

    fn round_sum(alpha: &[T], summands: &[Self], tol: f64) -> Self {
        crate::su2::round::round_sum(alpha, summands, tol)
    }
}

impl<T: Scalar> TtHamiltonian<Su2TtVector<T>> for Su2SparseHamiltonian<T> {
    fn matvec(&self, x: &Su2TtVector<T>) -> Su2TtVector<T> {
        self * x
    }

    fn rayleigh_quotient(&self, x: &Su2TtVector<T>) -> T {
        crate::su2::hamiltonian::rayleigh_quotient(self, x)
    }
}

impl<T: Scalar> InnerProductVector for Su2Core<T> {
    type Scalar = T;

    fn dot(&self, other: &Self) -> T {
        self.dot_center_par(other)
    }

    fn norm(&self) -> f64 {
        self.norm_center_par()
    }

    fn axpy(&mut self, a: T, other: &Self) {
        Su2Core::axpy_par(self, a, other)
    }

    fn scale(&mut self, a: T) {
        self.scale_mut_par(a)
    }

    fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, f: F) {
        Su2Core::for_each_coeff_mut(self, f)
    }

    fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, f: F) {
        Su2Core::zip_coeffs_mut(self, other, f)
    }

    fn coeff(&self, idx: usize) -> T {
        Su2Core::coeff(self, idx)
    }

    fn set_coeff(&mut self, idx: usize, val: T) {
        Su2Core::set_coeff(self, idx, val)
    }

    fn num_coeffs(&self) -> usize {
        Su2Core::num_coeffs(self)
    }
}

impl<T: Scalar> InnerProductVector for Su2ContractedCores<T> {
    type Scalar = T;

    fn dot(&self, other: &Self) -> T {
        self.dot_cores_par(other)
    }

    fn norm(&self) -> f64 {
        Su2ContractedCores::norm_par(self)
    }

    fn axpy(&mut self, a: T, other: &Self) {
        Su2ContractedCores::axpy_par(self, a, other)
    }

    fn scale(&mut self, a: T) {
        self.scale_mut_par(a)
    }

    fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, f: F) {
        Su2ContractedCores::for_each_coeff_mut(self, f)
    }

    fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, f: F) {
        Su2ContractedCores::zip_coeffs_mut(self, other, f)
    }

    fn coeff(&self, idx: usize) -> T {
        Su2ContractedCores::coeff(self, idx)
    }

    fn set_coeff(&mut self, idx: usize, val: T) {
        Su2ContractedCores::set_coeff(self, idx, val)
    }

    fn num_coeffs(&self) -> usize {
        Su2ContractedCores::num_coeffs(self)
    }
}

impl<T: Scalar> InnerProductVector for crate::su2::tangent::Su2TtTangent<T> {
    type Scalar = T;

    fn dot(&self, other: &Self) -> T {
        crate::su2::tangent::Su2TtTangent::dot(self, other)
    }

    fn norm(&self) -> f64 {
        crate::su2::tangent::Su2TtTangent::norm(self)
    }

    fn axpy(&mut self, a: T, other: &Self) {
        crate::su2::tangent::Su2TtTangent::axpy(self, a, other)
    }

    fn scale(&mut self, a: T) {
        self.scale_mut(a)
    }

    fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, f: F) {
        crate::su2::tangent::Su2TtTangent::for_each_coeff_mut(self, f)
    }

    fn for_each_coeff<F: FnMut(T)>(&self, mut f: F) {
        for c in &self.components {
            c.for_each_coeff(&mut f);
        }
    }

    fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, f: F) {
        crate::su2::tangent::Su2TtTangent::zip_coeffs_mut(self, other, f)
    }

    fn coeff(&self, idx: usize) -> T {
        crate::su2::tangent::Su2TtTangent::coeff(self, idx)
    }

    fn set_coeff(&mut self, idx: usize, val: T) {
        crate::su2::tangent::Su2TtTangent::set_coeff(self, idx, val)
    }

    fn num_coeffs(&self) -> usize {
        crate::su2::tangent::Su2TtTangent::num_coeffs(self)
    }
}

#[cfg(test)]
mod s5a_tests {
    use crate::su2::core::Family as Su2Family;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randomized_su2_state, seeded};

    #[test]
    fn s5a_e1_su2core_coeff_order_pin() {
        let mut rng = seeded(0x5a40);
        let space = Su2Space::new(6, 4, 2);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        let mut core = x.core(2).clone();
        let mut idx = 0usize;
        for f in Su2Family::ALL {
            for s in core.transitions(f).collect::<Vec<_>>() {
                if let Some(mut b) = core.block_mut(f, s) {
                    let (nr, nc) = (b.nrows(), b.ncols());
                    for j in 0..nc {
                        for i in 0..nr {
                            *b.as_mut().get_mut(i, j) = 1000.0 + idx as f64;
                            idx += 1;
                        }
                    }
                }
            }
        }
        let n = core.num_coeffs();
        assert_eq!(n, idx, "e1: num_coeffs disagrees with the fill walk");
        for i in 0..n {
            assert_eq!(
                core.coeff(i),
                1000.0 + i as f64,
                "e1: coeff({i}) order mismatch"
            );
        }

        let canonical: Vec<f64> = (0..n).map(|i| core.coeff(i)).collect();
        let mut sector_outer: Vec<f64> = Vec::new();
        for s in core.row_qn.clone() {
            for f in Su2Family::ALL {
                if !core.transitions(f).any(|t| t == s) {
                    continue;
                }
                if let Some(b) = core.block(f, s) {
                    for j in 0..b.ncols() {
                        for i in 0..b.nrows() {
                            sector_outer.push(*b.get(i, j));
                        }
                    }
                }
            }
        }
        assert_eq!(
            sector_outer.len(),
            canonical.len(),
            "e1: the two traversals must visit the same coefficients"
        );
        assert_ne!(
            canonical, sector_outer,
            "e1: VACUOUS at (6,4,2) — the canonical family-outer order coincides with the \
             sector-outer/family-inner order, so the pin discriminates nothing; raise the size/\
             rank spread until they differ"
        );

        let mut seen: Vec<f64> = Vec::new();
        core.for_each_coeff_mut(|v| seen.push(*v));
        assert_eq!(
            seen, canonical,
            "e1: for_each_coeff_mut order/count disagrees with coeff()"
        );

        let mut other = core.clone();
        other.for_each_coeff_mut(|v| *v = -*v);
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        let mut probe = core.clone();
        probe.zip_coeffs_mut(&other, |a, b| pairs.push((*a, b)));
        assert_eq!(pairs.len(), n, "e1: zip_coeffs_mut visited the wrong count");
        for (i, (a, b)) in pairs.iter().enumerate() {
            assert_eq!(*a, canonical[i], "e1: zip_coeffs_mut self order at {i}");
            assert_eq!(*b, -canonical[i], "e1: zip_coeffs_mut other order at {i}");
        }

        core.set_coeff(0, -1.0);
        assert_eq!(core.coeff(0), -1.0, "e1: set_coeff round-trip failed");
        #[allow(clippy::needless_range_loop)]
        for i in 1..n {
            assert_eq!(
                core.coeff(i),
                canonical[i],
                "e1: set_coeff(0) disturbed coefficient {i}"
            );
        }
    }

    #[test]
    fn su2_tangent_read_visitor_matches_mut_walk() {
        use crate::solvers::InnerProductVector;
        use crate::su2::tangent::Su2TtTangent;
        let mut rng = seeded(0x7eae);
        let space = Su2Space::new(6, 4, 2);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let y: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let (bl, br) = crate::su2::riemannian::tangent_bases(&x);
        let t = Su2TtTangent::project(&bl, &br, &y);
        let mut got = Vec::new();
        InnerProductVector::for_each_coeff(&t, |c: f64| got.push(c.to_bits()));
        assert!(got.len() > 10, "vacuous: {} coefficients", got.len());
        let mut walk = Vec::new();
        t.clone().for_each_coeff_mut(|c| walk.push(c.to_bits()));
        assert_eq!(got, walk, "read visitor vs mutable walk");
        let by_index: Vec<u64> = (0..t.num_coeffs()).map(|i| t.coeff(i).to_bits()).collect();
        assert_eq!(got, by_index, "read visitor vs coeff(i)");
    }
}
