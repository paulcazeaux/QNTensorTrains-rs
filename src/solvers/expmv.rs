// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Lanczos matrix-exponential-times-vector. Generic over
//! [`super::InnerProductVector`], so it serves both the two-site
//! ([`crate::core::twosite::ContractedSparseCores`]) and one-site ([`crate::core::SparseCore`])
//! TDVP half-steps.

use crate::core::factor::tridiag_exp_e1;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;

/// Convergence/breakdown diagnostics for one [`expmv`] call.
#[derive(Clone, Copy, Debug)]
pub struct ExpmvInfo {
    /// Krylov dimension actually built.
    pub krylovdim: usize,
    /// Whether the a-posteriori error estimate dropped below `tol` (or a happy breakdown fired)
    /// before `maxdim` was reached.
    pub converged: bool,
    /// Whether the Krylov recurrence broke down (exp-invariant subspace reached) — the current
    /// partial sum is then the EXACT result, not an approximation.
    pub breakdown: bool,
}

/// `y ≈ exp(a·A)·x0` for Hermitian `A` (given as the closure `op: &V -> V`), coefficient
/// `a: V::Scalar` (real for imaginary-time TDVP, complex for real-time), stopping tolerance `tol`
/// and max Krylov dimension `maxdim`. Full-
/// reorthogonalized Lanczos; a happy breakdown returns the exact result. Does not mutate `x0`.
/// The algorithm body: 3-term recurrence + full MGS reorthogonalization + a
/// real-tridiagonal-exponential a-posteriori error estimate.
pub fn expmv<V, F>(mut op: F, a: V::Scalar, x0: &V, tol: f64, maxdim: usize) -> (V, ExpmvInfo)
where
    V: InnerProductVector,
    F: FnMut(&V) -> V,
{
    let beta0 = x0.norm();
    if beta0 == 0.0 {
        return (
            x0.clone(),
            ExpmvInfo {
                krylovdim: 0,
                converged: true,
                breakdown: true,
            },
        );
    }

    let mut v_basis: Vec<V> = Vec::with_capacity(maxdim.max(1));
    {
        let mut v1 = x0.clone();
        v1.scale(<V::Scalar as Scalar>::from_f64(1.0 / beta0));
        v_basis.push(v1);
    }
    let mut dv: Vec<f64> = Vec::new();
    let mut ev: Vec<f64> = Vec::new();
    let converged;
    let breakdown;
    let mut u: Vec<V::Scalar>;

    loop {
        let j = v_basis.len();
        let mut w = op(v_basis.last().unwrap());
        let alpha = v_basis.last().unwrap().dot(&w).re_s();
        dv.push(alpha);
        w.axpy(
            <V::Scalar as Scalar>::from_f64(-alpha),
            v_basis.last().unwrap(),
        );
        if let Some(&beta_prev) = ev.last() {
            let idx = v_basis.len() - 2;
            w.axpy(<V::Scalar as Scalar>::from_f64(-beta_prev), &v_basis[idx]);
        }
        for vi in &v_basis {
            let c = vi.dot(&w);
            w.axpy(-c, vi);
        }
        let betaj = w.norm();

        u = tridiag_exp_e1(&dv, &ev, a);
        let err = beta0 * betaj * u[j - 1].abs_s();

        if betaj <= beta0 * 1e-12 || betaj <= 1e-13 {
            converged = true;
            breakdown = true;
            break;
        }
        if err < tol {
            converged = true;
            breakdown = false;
            break;
        }
        if j >= maxdim {
            converged = false;
            breakdown = false;
            break;
        }
        ev.push(betaj);
        let mut vnext = w;
        vnext.scale(<V::Scalar as Scalar>::from_f64(1.0 / betaj));
        v_basis.push(vnext);
    }

    let k = v_basis.len();
    let mut y = v_basis[0].clone();
    y.scale(<V::Scalar as Scalar>::from_f64(beta0) * u[0]);
    for i in 1..k {
        y.axpy(<V::Scalar as Scalar>::from_f64(beta0) * u[i], &v_basis[i]);
    }
    (
        y,
        ExpmvInfo {
            krylovdim: k,
            converged,
            breakdown,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factor::tridiag_exp_e1;
    use crate::scalar::C64;

    #[derive(Clone)]
    struct DiagVec {
        data: Vec<C64>,
    }

    impl InnerProductVector for DiagVec {
        type Scalar = C64;
        fn dot(&self, o: &Self) -> C64 {
            self.data
                .iter()
                .zip(&o.data)
                .map(|(&a, &b)| a.conj_s() * b)
                .fold(C64::zero(), |s, x| s + x)
        }
        fn norm(&self) -> f64 {
            self.dot(self).re_s().sqrt()
        }
        fn axpy(&mut self, a: C64, o: &Self) {
            for (x, &y) in self.data.iter_mut().zip(&o.data) {
                *x += a * y;
            }
        }
        fn scale(&mut self, a: C64) {
            for x in &mut self.data {
                *x *= a;
            }
        }
        fn for_each_coeff_mut<F: FnMut(&mut C64)>(&mut self, mut f: F) {
            for x in &mut self.data {
                f(x);
            }
        }
        fn zip_coeffs_mut<F: FnMut(&mut C64, C64)>(&mut self, o: &Self, mut f: F) {
            for (x, &y) in self.data.iter_mut().zip(&o.data) {
                f(x, y);
            }
        }
        fn coeff(&self, i: usize) -> C64 {
            self.data[i]
        }
        fn set_coeff(&mut self, i: usize, v: C64) {
            self.data[i] = v;
        }
        fn num_coeffs(&self) -> usize {
            self.data.len()
        }
    }

    fn diag_op(d: &[f64], v: &DiagVec) -> DiagVec {
        DiagVec {
            data: v
                .data
                .iter()
                .enumerate()
                .map(|(i, &x)| C64::from_f64(d[i]) * x)
                .collect(),
        }
    }

    #[test]
    fn expmv_diagonal_matches_analytic() {
        let d = [0.3, -1.1, 2.0, 0.7, -0.5];
        let x0 = DiagVec {
            data: vec![
                C64::new(1.0, 0.2),
                C64::new(-0.5, 0.1),
                C64::new(0.3, -0.4),
                C64::new(0.8, 0.0),
                C64::new(-0.2, 0.6),
            ],
        };
        for &a in &[
            C64::new(-0.5, 0.0),
            C64::new(0.0, -0.05),
            C64::new(0.1, -0.2),
        ] {
            let (y, info) = expmv(|v: &DiagVec| diag_op(&d, v), a, &x0, 1e-12, 30);
            assert!(info.converged, "a={a:?}: expmv should converge");
            for (i, &di) in d.iter().enumerate() {
                let want = (a * C64::from_f64(di)).exp_s() * x0.data[i];
                assert!(
                    (y.coeff(i) - want).abs_s() < 1e-10,
                    "a={a:?} coeff {i}: {} vs {}",
                    y.coeff(i),
                    want
                );
            }
        }
    }

    #[test]
    fn expmv_eigenvector_breaks_down_exactly() {
        let d = [0.3, -1.1, 2.0, 0.7, -0.5];
        let mut e2 = DiagVec {
            data: vec![C64::zero(); 5],
        };
        e2.data[2] = C64::one();
        let a = C64::new(0.0, -0.1);
        let (y, info) = expmv(|v: &DiagVec| diag_op(&d, v), a, &e2, 1e-12, 30);
        assert_eq!(info.krylovdim, 1);
        assert!(info.breakdown && info.converged);
        let want = (a * C64::from_f64(d[2])).exp_s();
        assert!((y.coeff(2) - want).abs_s() < 1e-13);
    }

    #[test]
    fn expmv_zero_input_returns_zero() {
        let d = [0.3, -1.1, 2.0, 0.7, -0.5];
        let z = DiagVec {
            data: vec![C64::zero(); 5],
        };
        let (y, info) = expmv(
            |v: &DiagVec| diag_op(&d, v),
            C64::new(0.0, -0.1),
            &z,
            1e-12,
            30,
        );
        assert_eq!(info.krylovdim, 0);
        assert!(info.converged && info.breakdown);
        assert_eq!(y.norm(), 0.0);
    }

    #[test]
    fn expmv_hits_maxdim_unconverged() {
        let d = [0.3, -1.1, 2.0, 0.7, -0.5];
        let x0 = DiagVec {
            data: vec![
                C64::new(1.0, 0.2),
                C64::new(-0.5, 0.1),
                C64::new(0.3, -0.4),
                C64::new(0.8, 0.0),
                C64::new(-0.2, 0.6),
            ],
        };
        let (_, info) = expmv(
            |v: &DiagVec| diag_op(&d, v),
            C64::new(0.0, -0.1),
            &x0,
            1e-14,
            1,
        );
        assert!(!info.converged);
        assert!(!info.breakdown);
        assert_eq!(info.krylovdim, 1);
    }

    fn t_apply(dv: &[f64], ev: &[f64], a: C64, x: &[C64]) -> Vec<C64> {
        let k = dv.len();
        let mut y = vec![C64::zero(); k];
        for i in 0..k {
            let mut acc = C64::from_f64(dv[i]) * x[i];
            if i > 0 {
                acc += C64::from_f64(ev[i - 1]) * x[i - 1];
            }
            if i + 1 < k {
                acc += C64::from_f64(ev[i]) * x[i + 1];
            }
            y[i] = a * acc;
        }
        y
    }

    fn dense_expm_e1(dv: &[f64], ev: &[f64], a: C64) -> Vec<C64> {
        let k = dv.len();
        let mut term = vec![C64::zero(); k];
        term[0] = C64::one();
        let mut result = term.clone();
        for n in 1..80 {
            let tt = t_apply(dv, ev, a, &term);
            for (t, &v) in term.iter_mut().zip(tt.iter()) {
                *t = v * C64::from_f64(1.0 / n as f64);
            }
            for (r, &t) in result.iter_mut().zip(term.iter()) {
                *r += t;
            }
        }
        result
    }

    #[test]
    fn tridiag_exp_e1_matches_dense_taylor() {
        let a1 = C64::new(0.0, -0.3);
        let u1 = tridiag_exp_e1(&[0.7], &[], a1);
        assert!((u1[0] - (a1 * C64::from_f64(0.7)).exp_s()).abs_s() < 1e-14);

        let dv = [0.5, -0.8, 1.2];
        let ev = [0.3, -0.4];
        for &a in &[
            C64::new(-0.6, 0.0),
            C64::new(0.0, -0.2),
            C64::new(0.1, -0.15),
        ] {
            let u = tridiag_exp_e1(&dv, &ev, a);
            let want = dense_expm_e1(&dv, &ev, a);
            for (i, (&ui, &wi)) in u.iter().zip(want.iter()).enumerate() {
                assert!((ui - wi).abs_s() < 1e-10, "a={a:?} i={i}: {ui} vs {wi}");
            }
        }
    }
}
