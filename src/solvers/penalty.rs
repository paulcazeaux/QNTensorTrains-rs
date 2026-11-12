// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The overlap penalty of the excited-state ladder: the scalar bookkeeping of

use crate::scalar::Scalar;
use crate::solvers::{InnerProductVector, TtState};

/// A set of fixed lower states `{ψⱼ}` and the weight `w` of the overlap penalty
/// `w Σⱼ |⟨ψ̂ⱼ,x⟩|²`.
#[derive(Clone)]
pub struct Penalty<V> {
    /// Penalty weight; finite and `≥ 0`.
    pub w: f64,
    /// The lower states, in the caller's order (every helper sums over them in that order).
    pub states: Vec<V>,
    /// `invn[j] = 1/‖ψⱼ‖`, exactly `1.0` when `‖ψⱼ‖ == 1.0`.
    pub invn: Vec<f64>,
}

impl<V: TtState> Penalty<V> {
    /// Clone `states` and precompute `invn`.
    pub fn new(w: f64, states: &[V]) -> Self {
        assert!(
            w.is_finite() && w >= 0.0,
            "Penalty::new: the weight must be finite and non-negative (got {w})"
        );
        let mut invn = Vec::with_capacity(states.len());
        for (j, psi) in states.iter().enumerate() {
            assert_eq!(
                psi.num_sites(),
                states[0].num_sites(),
                "Penalty::new: states[{j}] has {} sites but states[0] has {}",
                psi.num_sites(),
                states[0].num_sites()
            );
            let n = psi.norm();
            assert!(
                n.is_finite() && n > 0.0,
                "Penalty::new: states[{j}] has norm {n} — a finite, NONZERO norm is required (the \
                 penalty divides by it); a zero state carries no direction to penalize"
            );
            invn.push(if n == 1.0 { 1.0 } else { 1.0 / n });
        }
        Penalty {
            w,
            states: states.to_vec(),
            invn,
        }
    }

    /// `w > 0 ∧ !states.is_empty()` — the guard that lets the drivers route an inactive penalty to
    /// the unpenalized path, so `Penalty::new(0.0, &[])` (and either half of it) reproduces the
    /// bare optimizer BITWISE rather than merely to roundoff.
    pub fn is_active(&self) -> bool {
        self.w > 0.0 && !self.states.is_empty()
    }

    /// `Some(self)` iff [`Self::is_active`] — the spelling the drivers use to fold an inactive
    /// penalty into their `None` path.
    pub fn active(&self) -> Option<&Self> {
        if self.is_active() { Some(self) } else { None }
    }

    /// `oⱼ = ⟨ψ̂ⱼ, x⟩ = ⟨ψⱼ, x⟩ · invnⱼ`, in state order. One embedded dot per state; nothing is
    /// normalized in place.
    pub fn overlaps(&self, x: &V) -> Vec<V::Scalar> {
        self.states
            .iter()
            .zip(self.invn.iter())
            .map(|(psi, &invn)| psi.dot(x) * V::Scalar::from_f64(invn))
            .collect()
    }

    /// `pen = w · Σⱼ |oⱼ|²` from precomputed overlaps — the amount by which the penalty raises the
    /// functional at a UNIT-norm point (`J = λ + pen` there).
    pub fn energy(&self, os: &[V::Scalar]) -> f64 {
        debug_assert_eq!(
            os.len(),
            self.states.len(),
            "Penalty::energy: overlap count"
        );
        let mut acc = 0.0f64;
        for o in os {
            acc += o.re_s() * o.re_s() + o.im_s() * o.im_s();
        }
        self.w * acc
    }

    /// `g += w Σⱼ oⱼ · invnⱼ · project(ψⱼ)` — the penalty's contribution to the Riemannian
    /// gradient, in state order.
    pub fn tangent_axpy<G, P>(&self, g: &mut G, os: &[V::Scalar], project: P)
    where
        G: InnerProductVector<Scalar = V::Scalar>,
        P: Fn(&V) -> G,
    {
        debug_assert_eq!(
            os.len(),
            self.states.len(),
            "Penalty::tangent_axpy: overlap count"
        );
        for ((psi, &invn), &o) in self.states.iter().zip(self.invn.iter()).zip(os.iter()) {
            let coef = V::Scalar::from_f64(self.w * invn) * o;
            g.axpy(coef, &project(psi));
        }
    }

    /// The penalty's contribution to the line-search model along `x + α·v`:
    /// `(Δc₁, Δc₂) = (w Σⱼ Re(conj(oˣⱼ)·oᵛⱼ), w Σⱼ |oᵛⱼ|²)` with `oˣ = overlaps(x)`,
    /// `oᵛ = overlaps(v)`.
    pub fn linesearch_coeffs(&self, x: &V, v: &V) -> (f64, f64) {
        let ox = self.overlaps(x);
        let ov = self.overlaps(v);
        let mut d1 = 0.0f64;
        let mut d2 = 0.0f64;
        for (a, b) in ox.iter().zip(ov.iter()) {
            d1 += a.re_s() * b.re_s() + a.im_s() * b.im_s();
            d2 += b.re_s() * b.re_s() + b.im_s() * b.im_s();
        }
        (self.w * d1, self.w * d2)
    }
}
