// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! MINRES: minimum-residual Krylov solve of `op·x = b` for a **symmetric, possibly INDEFINITE**
//! `op` on any [`InnerProductVector`] space, started at `x₀ = 0`.

use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;

/// How [`minres`] stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MinresFlag {
    /// `|φ̄| ≤ tol_rel·‖b‖`, or the Lanczos recurrence closed on an exact invariant subspace
    /// (`β_{k+1} = 0`), where the solve is exact.
    Converged,
    /// The iteration cap was reached. The returned `x` is the minimum-residual iterate over the
    /// Krylov space built so far, not a failure value — quote its FRESH residual (§5).
    MaxIter,
    /// `γ = hypot(γ̄, β_{k+1}) = 0`: the rotation radius collapsed. For a nonsingular system on the
    /// range of `op` this cannot happen; it is reported rather than papered over, because the
    /// alternative (dividing by it) manufactures an `inf` iterate that looks like divergence.
    Breakdown { gamma: f64 },
}

/// MINRES for a symmetric (possibly indefinite) `op`, from `x₀ = 0`. Returns
/// `(x, flag, iterations)`; `op` calls `== iterations`.
pub fn minres<V, Op>(mut op: Op, b: &V, tol_rel: f64, max_iter: usize) -> (V, MinresFlag, usize)
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
{
    let beta1 = b.norm();
    let mut x = b.clone();
    x.scale(V::Scalar::from_f64(0.0));
    if beta1 == 0.0 {
        return (x, MinresFlag::Converged, 0);
    }

    let zero = x.clone();
    let mut v = b.clone();
    v.scale(V::Scalar::from_f64(1.0 / beta1));
    let mut v_prev = zero.clone();
    let mut w = zero.clone();
    let mut w_prev = zero;

    let mut beta_prev = 0.0f64;
    let (mut c, mut s) = (-1.0f64, 0.0f64);
    let mut dbar = 0.0f64;
    let mut eps = 0.0f64;
    let mut phibar = beta1;

    for it in 0..max_iter {
        let mut p = op(&v);
        let alpha = v.dot(&p).re_s();
        p.axpy(V::Scalar::from_f64(-alpha), &v);
        if beta_prev != 0.0 {
            p.axpy(V::Scalar::from_f64(-beta_prev), &v_prev);
        }
        let beta_next = p.norm();

        let eps_old = eps;
        let delta = c * dbar + s * alpha;
        let gbar = s * dbar - c * alpha;
        eps = s * beta_next;
        dbar = -c * beta_next;

        let gamma = gbar.hypot(beta_next);
        if gamma == 0.0 {
            return (x, MinresFlag::Breakdown { gamma }, it + 1);
        }
        c = gbar / gamma;
        s = beta_next / gamma;
        let phi = c * phibar;
        phibar *= s;

        let mut w_new = v.clone();
        w_new.axpy(V::Scalar::from_f64(-delta), &w);
        w_new.axpy(V::Scalar::from_f64(-eps_old), &w_prev);
        w_new.scale(V::Scalar::from_f64(1.0 / gamma));
        x.axpy(V::Scalar::from_f64(phi), &w_new);
        w_prev = w;
        w = w_new;

        if phibar.abs() <= tol_rel * beta1 {
            return (x, MinresFlag::Converged, it + 1);
        }
        if beta_next == 0.0 {
            return (x, MinresFlag::Converged, it + 1);
        }

        v_prev = v;
        p.scale(V::Scalar::from_f64(1.0 / beta_next));
        v = p;
        beta_prev = beta_next;
    }
    (x, MinresFlag::MaxIter, max_iter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct DVec(Vec<f64>);

    impl InnerProductVector for DVec {
        type Scalar = f64;
        fn dot(&self, other: &Self) -> f64 {
            self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
        }
        fn norm(&self) -> f64 {
            self.dot(self).max(0.0).sqrt()
        }
        fn axpy(&mut self, a: f64, other: &Self) {
            for (t, &o) in self.0.iter_mut().zip(&other.0) {
                *t += a * o;
            }
        }
        fn scale(&mut self, a: f64) {
            for t in self.0.iter_mut() {
                *t *= a;
            }
        }
        fn for_each_coeff_mut<F: FnMut(&mut f64)>(&mut self, mut f: F) {
            for t in self.0.iter_mut() {
                f(t);
            }
        }
        fn zip_coeffs_mut<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, mut f: F) {
            for (t, &o) in self.0.iter_mut().zip(&other.0) {
                f(t, o);
            }
        }
        fn coeff(&self, idx: usize) -> f64 {
            self.0[idx]
        }
        fn set_coeff(&mut self, idx: usize, val: f64) {
            self.0[idx] = val;
        }
        fn num_coeffs(&self) -> usize {
            self.0.len()
        }
    }

    fn sym_indefinite(n: usize) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            a[i][i] = if i % 2 == 0 {
                2.0 + i as f64
            } else {
                -1.0 - i as f64
            };
            if i + 1 < n {
                a[i][i + 1] = 0.6;
                a[i + 1][i] = 0.6;
            }
        }
        a
    }

    fn apply(a: &[Vec<f64>], v: &DVec) -> DVec {
        DVec(
            a.iter()
                .map(|row| row.iter().zip(&v.0).map(|(x, y)| x * y).sum())
                .collect(),
        )
    }

    fn dense_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = b.len();
        let mut m: Vec<Vec<f64>> = a
            .iter()
            .zip(b)
            .map(|(row, &rhs)| {
                let mut r = row.clone();
                r.push(rhs);
                r
            })
            .collect();
        for k in 0..n {
            let piv =
                (k..n).max_by(|&i, &j| m[i][k].abs().partial_cmp(&m[j][k].abs()).expect("finite"));
            m.swap(k, piv.expect("nonempty"));
            let d = m[k][k];
            m[k].iter_mut().skip(k).for_each(|e| *e /= d);
            let pivot_row = m[k].clone();
            for (i, row) in m.iter_mut().enumerate() {
                if i != k && row[k] != 0.0 {
                    let f = row[k];
                    for (e, &pe) in row.iter_mut().zip(pivot_row.iter()).skip(k) {
                        *e -= f * pe;
                    }
                }
            }
        }
        (0..n).map(|i| m[i][n]).collect()
    }

    #[test]
    fn minres_solves_an_indefinite_system() {
        let n = 8usize;
        let a = sym_indefinite(n);
        let b = DVec((0..n).map(|i| 0.3 + 0.7 * (i as f64).sin()).collect());
        let (x, flag, iters) = minres(|v: &DVec| apply(&a, v), &b, 1e-12, 4 * n);
        assert_eq!(flag, MinresFlag::Converged, "iters = {iters}");
        assert!(
            iters <= n,
            "MINRES must terminate in at most n steps: {iters}"
        );

        let want = dense_solve(&a, &b.0);
        for (i, (&got, &wnt)) in x.0.iter().zip(want.iter()).enumerate() {
            assert!(
                (got - wnt).abs() <= 1e-9 * wnt.abs().max(1.0),
                "component {i}: got {got:e} want {wnt:e}"
            );
        }
        let mut r = b.clone();
        r.axpy(-1.0, &apply(&a, &x));
        assert!(
            r.norm() <= 1e-10 * b.norm(),
            "fresh residual {:e}",
            r.norm()
        );
    }

    #[test]
    fn minres_residual_is_monotone() {
        let n = 8usize;
        let a = sym_indefinite(n);
        let b = DVec((0..n).map(|i| 1.0 - 0.2 * i as f64).collect());
        let mut prev = f64::INFINITY;
        for k in 1..=n {
            let (x, _flag, _it) = minres(|v: &DVec| apply(&a, v), &b, 0.0, k);
            let mut r = b.clone();
            r.axpy(-1.0, &apply(&a, &x));
            let rn = r.norm();
            assert!(
                rn <= prev * (1.0 + 1e-10) + 1e-14,
                "residual grew at k={k}: {rn:e} > {prev:e}"
            );
            prev = rn;
        }
    }

    #[test]
    fn minres_c0_sign_is_load_bearing() {
        let n = 6usize;
        let a = sym_indefinite(n);
        let b = DVec((0..n).map(|i| 0.5 + 0.3 * i as f64).collect());
        let want = dense_solve(&a, &b.0);

        let (x_ok, _f, _i) = minres(|v: &DVec| apply(&a, v), &b, 1e-12, 4 * n);
        let err_ok: f64 = x_ok
            .0
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        assert!(err_ok <= 1e-9, "shipped seed must be right: {err_ok:e}");

        let x_bad = minres_flipped_seed(|v: &DVec| apply(&a, v), &b, 4 * n);
        let err_bad: f64 = x_bad
            .0
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        assert!(
            err_bad > 1e3 * err_ok.max(1e-16),
            "the c₀ = +1 seed must be measurably WRONG (got {err_bad:e} vs {err_ok:e}); if this \
             arm ever passes vacuously the sign trap is no longer guarded"
        );
    }

    fn minres_flipped_seed<V, Op>(mut op: Op, b: &V, max_iter: usize) -> V
    where
        V: InnerProductVector,
        Op: FnMut(&V) -> V,
    {
        let beta1 = b.norm();
        let mut x = b.clone();
        x.scale(V::Scalar::from_f64(0.0));
        let zero = x.clone();
        let mut v = b.clone();
        v.scale(V::Scalar::from_f64(1.0 / beta1));
        let mut v_prev = zero.clone();
        let mut w = zero.clone();
        let mut w_prev = zero;
        let mut beta_prev = 0.0f64;
        let (mut c, mut s) = (1.0f64, 0.0f64);
        let (mut dbar, mut eps, mut phibar) = (0.0f64, 0.0f64, beta1);
        for _ in 0..max_iter {
            let mut p = op(&v);
            let alpha = v.dot(&p).re_s();
            p.axpy(V::Scalar::from_f64(-alpha), &v);
            if beta_prev != 0.0 {
                p.axpy(V::Scalar::from_f64(-beta_prev), &v_prev);
            }
            let beta_next = p.norm();
            let eps_old = eps;
            let delta = c * dbar + s * alpha;
            let gbar = s * dbar - c * alpha;
            eps = s * beta_next;
            dbar = -c * beta_next;
            let gamma = gbar.hypot(beta_next);
            if gamma == 0.0 {
                return x;
            }
            c = gbar / gamma;
            s = beta_next / gamma;
            let phi = c * phibar;
            phibar *= s;
            let mut w_new = v.clone();
            w_new.axpy(V::Scalar::from_f64(-delta), &w);
            w_new.axpy(V::Scalar::from_f64(-eps_old), &w_prev);
            w_new.scale(V::Scalar::from_f64(1.0 / gamma));
            x.axpy(V::Scalar::from_f64(phi), &w_new);
            w_prev = w;
            w = w_new;
            if beta_next == 0.0 {
                return x;
            }
            v_prev = v;
            p.scale(V::Scalar::from_f64(1.0 / beta_next));
            v = p;
            beta_prev = beta_next;
        }
        x
    }

    #[test]
    fn minres_edge_cases() {
        let n = 5usize;
        let mut a = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            a[i][i] = 1.0 + i as f64;
            if i + 1 < n {
                a[i][i + 1] = 0.25;
                a[i + 1][i] = 0.25;
            }
        }
        let zero = DVec(vec![0.0; n]);
        let mut applies = 0usize;
        let (x0, flag0, it0) = minres(
            |v: &DVec| {
                applies += 1;
                apply(&a, v)
            },
            &zero,
            1e-10,
            50,
        );
        assert_eq!(flag0, MinresFlag::Converged);
        assert_eq!(it0, 0);
        assert_eq!(applies, 0, "b = 0 must pay no apply");
        assert_eq!(x0, zero);

        let b = DVec((0..n).map(|i| 1.0 + 0.1 * i as f64).collect());
        let (x, flag, _it) = minres(|v: &DVec| apply(&a, v), &b, 1e-13, 4 * n);
        assert_eq!(flag, MinresFlag::Converged);
        let want = dense_solve(&a, &b.0);
        for (i, (&got, &wnt)) in x.0.iter().zip(want.iter()).enumerate() {
            assert!(
                (got - wnt).abs() <= 1e-10 * wnt.abs().max(1.0),
                "SPD arm component {i}"
            );
        }
    }

    #[test]
    fn minres_maxiter_returns_the_partial_minimizer() {
        let n = 10usize;
        let a = sym_indefinite(n);
        let b = DVec((0..n).map(|i| (i as f64 * 0.7).cos()).collect());
        let (x, flag, iters) = minres(|v: &DVec| apply(&a, v), &b, 1e-14, 3);
        assert_eq!(flag, MinresFlag::MaxIter);
        assert_eq!(iters, 3);
        let mut r = b.clone();
        r.axpy(-1.0, &apply(&a, &x));
        assert!(
            r.norm() < b.norm(),
            "the partial iterate must reduce the residual"
        );
    }
}
