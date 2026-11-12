// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Steihaug–Toint truncated CG: approximate minimization of the quadratic model
//! `m(s) = ⟨g,s⟩ + ½⟨s, A·s⟩` over the trust ball `‖s‖ ≤ Δ`, in whatever metric the
//! [`InnerProductVector`] type carries in its `dot`.

use super::InnerProductVector;

/// How [`steihaug_cg`] exited. See the module header for the semantics of each arm; note that
/// the returned vector is the truncated point only when [`SteihaugInfo::boundary_hit`] says so
/// (§3/§4), regardless of this status.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SteihaugStatus {
    /// Residual tolerance met strictly inside the ball.
    Interior,
    /// A CG update crossed the ball; truncated at the crossing point.
    Boundary,
    /// `⟨p, A·p⟩ ≤ 0`; rode the direction to the boundary.
    NegCurve,
    /// Iteration cap reached.
    MaxIter,
    /// Model value rose (asymmetric operator, §4) or the metric nulled the residual (§5).
    Stagnated,
}

/// Diagnostics of one [`steihaug_cg`] solve.
#[derive(Clone, Debug)]
pub struct SteihaugInfo {
    /// Exit arm. The returned vector is the best iterate whenever the final one lost (§4).
    pub status: SteihaugStatus,
    /// Number of operator applications.
    pub iters: usize,
    /// `m(0) − m(s_returned) ≥ 0`, from the maintained residual (§2).
    pub model_red: f64,
    /// `‖g‖` in the type's own metric.
    pub gnorm: f64,
    /// `‖r‖` at the last completed **interior** iterate (a truncated boundary point does not
    /// update it — its residual is not driven to zero and would misreport solve quality).
    pub resnorm: f64,
    /// ★ True iff a truncation fired AND the truncated point is the returned vector (§3). The
    /// trust-region grow gate reads this flag and nothing else.
    pub boundary_hit: bool,
}

/// Approximately minimize `m(s) = ⟨g,s⟩ + ½⟨s, A·s⟩` over `‖s‖ ≤ delta` by truncated CG.
///
/// # Panics
///
/// If `delta` is not positive and finite with a representable square (§6).
pub fn steihaug_cg<V, F>(
    mut aop: F,
    g: &V,
    delta: f64,
    rtol: f64,
    atol: f64,
    max_iter: usize,
) -> (V, SteihaugInfo)
where
    V: InnerProductVector<Scalar = f64>,
    F: FnMut(&V) -> V,
{
    assert!(
        delta.is_finite() && delta > 0.0 && delta * delta > 0.0,
        "steihaug_cg: trust radius must be positive and finite with a representable square \
         (got {delta:e}); below ~1.5e-154 the boundary quadratic evaluates 0/0 and NaN-poisons \
         the step"
    );

    let gnorm = g.norm();
    if gnorm == 0.0 || gnorm <= atol {
        let mut zero = g.clone();
        zero.scale(0.0);
        return (
            zero,
            SteihaugInfo {
                status: SteihaugStatus::Interior,
                iters: 0,
                model_red: 0.0,
                gnorm,
                resnorm: gnorm,
                boundary_hit: false,
            },
        );
    }
    let tol = atol.max(rtol * gnorm);

    let mut s = g.clone();
    s.scale(0.0);
    let mut r = g.clone();
    let mut p = g.clone();
    p.scale(-1.0);
    let mut rho = r.dot(&r);
    let mut resnorm = gnorm;

    let mut m_cur = 0.0_f64;
    let mut m_best = 0.0_f64;
    let mut s_best: Option<V> = None;

    let mut iters = 0_usize;
    let mut status = SteihaugStatus::MaxIter;
    let mut truncated_m: Option<f64> = None;

    for _ in 0..max_iter {
        let q = aop(&p);
        iters += 1;
        let pq = p.dot(&q);

        let ss = s.dot(&s);
        let sp = s.dot(&p);
        let pp = p.dot(&p);

        if pq <= 0.0 {
            let sigma = boundary_root(ss, sp, pp, delta);
            s.axpy(sigma, &p);
            r.axpy(sigma, &q);
            let m_t = 0.5 * (g.dot(&s) + s.dot(&r));
            status = SteihaugStatus::NegCurve;
            truncated_m = Some(m_t);
            break;
        }
        let alpha = rho / pq;
        if ss + alpha * (2.0 * sp + alpha * pp) > delta * delta {
            let sigma = boundary_root(ss, sp, pp, delta);
            s.axpy(sigma, &p);
            r.axpy(sigma, &q);
            let m_t = 0.5 * (g.dot(&s) + s.dot(&r));
            status = SteihaugStatus::Boundary;
            truncated_m = Some(m_t);
            break;
        }

        s.axpy(alpha, &p);
        r.axpy(alpha, &q);
        let m_new = 0.5 * (g.dot(&s) + s.dot(&r));
        debug_assert!(m_new.is_finite(), "steihaug_cg: non-finite model value");
        if m_new > m_best + 16.0 * f64::EPSILON * m_best.abs().max(1.0) {
            status = SteihaugStatus::Stagnated;
            m_cur = m_new;
            break;
        }
        m_cur = m_new;
        if m_new < m_best {
            m_best = m_new;
            s_best = Some(s.clone());
        }

        let rho_new = r.dot(&r);
        resnorm = rho_new.max(0.0).sqrt();
        if resnorm <= tol {
            status = SteihaugStatus::Interior;
            break;
        }
        if rho_new <= 0.0 {
            status = SteihaugStatus::Stagnated;
            break;
        }
        let beta = rho_new / rho;
        p.scale(beta);
        p.axpy(-1.0, &r);
        rho = rho_new;
    }

    let (s_ret, m_ret, boundary_hit) = match truncated_m {
        Some(m_t) if m_t <= m_best => (s, m_t, true),
        Some(_) => (take_best(s_best, g), m_best, false),
        None if m_cur <= m_best => (s, m_cur, false),
        None => (take_best(s_best, g), m_best, false),
    };

    (
        s_ret,
        SteihaugInfo {
            status,
            iters,
            model_red: -m_ret,
            gnorm,
            resnorm,
            boundary_hit,
        },
    )
}

fn boundary_root(ss: f64, sp: f64, pp: f64, delta: f64) -> f64 {
    debug_assert!(pp > 0.0, "boundary_root: metric-null direction");
    let c = ss - delta * delta;
    let disc = (sp * sp - pp * c).max(0.0);
    let root = disc.sqrt();
    if sp >= 0.0 {
        -c / (sp + root)
    } else {
        (root - sp) / pp
    }
}

fn take_best<V: InnerProductVector<Scalar = f64>>(best: Option<V>, g: &V) -> V {
    best.unwrap_or_else(|| {
        let mut zero = g.clone();
        zero.scale(0.0);
        zero
    })
}

/// [`steihaug_cg`] with a preconditioner `M⁻¹` — the SEPARATE preconditioned twin (the
/// unpreconditioned kernel above is a certified oracle and stays byte-identical; keep the ball
/// and exit logic of the two in lockstep when editing either).
pub fn steihaug_cg_with<V, F>(
    mut aop: F,
    g: &V,
    delta: f64,
    rtol: f64,
    atol: f64,
    max_iter: usize,
    msolve: Option<&dyn Fn(&V) -> V>,
) -> (V, SteihaugInfo)
where
    V: InnerProductVector<Scalar = f64>,
    F: FnMut(&V) -> V,
{
    let Some(msolve) = msolve else {
        return steihaug_cg(aop, g, delta, rtol, atol, max_iter);
    };
    assert!(
        delta.is_finite() && delta > 0.0 && delta * delta > 0.0,
        "steihaug_cg_with: trust radius must be positive and finite with a representable square \
         (got {delta:e})"
    );
    let gnorm = g.norm();
    if gnorm == 0.0 || gnorm <= atol {
        let mut zero = g.clone();
        zero.scale(0.0);
        return (
            zero,
            SteihaugInfo {
                status: SteihaugStatus::Interior,
                iters: 0,
                model_red: 0.0,
                gnorm,
                resnorm: gnorm,
                boundary_hit: false,
            },
        );
    }
    let tol = atol.max(rtol * gnorm);

    let mut s = g.clone();
    s.scale(0.0);
    let mut r = g.clone();
    let z = msolve(&r);
    let mut rho = r.dot(&z);
    let mut p = z;
    p.scale(-1.0);
    let mut resnorm = gnorm;

    let mut m_cur = 0.0_f64;
    let mut m_best = 0.0_f64;
    let mut s_best: Option<V> = None;
    let mut iters = 0_usize;
    let mut status = SteihaugStatus::MaxIter;
    let mut truncated_m: Option<f64> = None;

    if rho <= 0.0 {
        status = SteihaugStatus::Stagnated;
        return (
            s,
            SteihaugInfo {
                status,
                iters,
                model_red: 0.0,
                gnorm,
                resnorm,
                boundary_hit: false,
            },
        );
    }

    for _ in 0..max_iter {
        let q = aop(&p);
        iters += 1;
        let pq = p.dot(&q);
        let ss = s.dot(&s);
        let sp = s.dot(&p);
        let pp = p.dot(&p);

        if pq <= 0.0 {
            let sigma = boundary_root(ss, sp, pp, delta);
            s.axpy(sigma, &p);
            r.axpy(sigma, &q);
            let m_t = 0.5 * (g.dot(&s) + s.dot(&r));
            status = SteihaugStatus::NegCurve;
            truncated_m = Some(m_t);
            break;
        }
        let alpha = rho / pq;
        if ss + alpha * (2.0 * sp + alpha * pp) > delta * delta {
            let sigma = boundary_root(ss, sp, pp, delta);
            s.axpy(sigma, &p);
            r.axpy(sigma, &q);
            let m_t = 0.5 * (g.dot(&s) + s.dot(&r));
            status = SteihaugStatus::Boundary;
            truncated_m = Some(m_t);
            break;
        }
        s.axpy(alpha, &p);
        r.axpy(alpha, &q);
        let m_new = 0.5 * (g.dot(&s) + s.dot(&r));
        debug_assert!(
            m_new.is_finite(),
            "steihaug_cg_with: non-finite model value"
        );
        if m_new > m_best + 16.0 * f64::EPSILON * m_best.abs().max(1.0) {
            status = SteihaugStatus::Stagnated;
            m_cur = m_new;
            break;
        }
        m_cur = m_new;
        if m_new < m_best {
            m_best = m_new;
            s_best = Some(s.clone());
        }
        resnorm = r.norm();
        if resnorm <= tol {
            status = SteihaugStatus::Interior;
            break;
        }
        let znew = msolve(&r);
        let rho_new = r.dot(&znew);
        if rho_new <= 0.0 {
            status = SteihaugStatus::Stagnated;
            break;
        }
        let beta = rho_new / rho;
        p.scale(beta);
        p.axpy(-1.0, &znew);
        rho = rho_new;
    }

    let (s_ret, m_ret, boundary_hit) = match truncated_m {
        Some(m_t) if m_t <= m_best => (s, m_t, true),
        Some(_) => (take_best(s_best, g), m_best, false),
        None if m_cur <= m_best => (s, m_cur, false),
        None => (take_best(s_best, g), m_best, false),
    };
    (
        s_ret,
        SteihaugInfo {
            status,
            iters,
            model_red: -m_ret,
            gnorm,
            resnorm,
            boundary_hit,
        },
    )
}

/// [`steihaug_cg`] with the trust BALL measured in a caller-supplied inner product — the
/// ELLIPTICAL-ball twin (the unpreconditioned kernel above is a certified oracle and stays
/// byte-identical, exactly as [`steihaug_cg_with`] does; keep the exit logic of all three in
/// lockstep when editing any).
pub fn steihaug_cg_ball<V, F>(
    mut aop: F,
    g: &V,
    delta: f64,
    rtol: f64,
    atol: f64,
    max_iter: usize,
    ball_dot: &dyn Fn(&V, &V) -> f64,
) -> (V, SteihaugInfo)
where
    V: InnerProductVector<Scalar = f64>,
    F: FnMut(&V) -> V,
{
    assert!(
        delta.is_finite() && delta > 0.0 && delta * delta > 0.0,
        "steihaug_cg_ball: trust radius must be positive and finite with a representable square \
         (got {delta:e}); below ~1.5e-154 the boundary quadratic evaluates 0/0 and NaN-poisons \
         the step"
    );

    let gnorm = g.norm();
    if gnorm == 0.0 || gnorm <= atol {
        let mut zero = g.clone();
        zero.scale(0.0);
        return (
            zero,
            SteihaugInfo {
                status: SteihaugStatus::Interior,
                iters: 0,
                model_red: 0.0,
                gnorm,
                resnorm: gnorm,
                boundary_hit: false,
            },
        );
    }
    let tol = atol.max(rtol * gnorm);

    let mut s = g.clone();
    s.scale(0.0);
    let mut r = g.clone();
    let mut p = g.clone();
    p.scale(-1.0);
    let mut rho = r.dot(&r);
    let mut resnorm = gnorm;

    let mut m_cur = 0.0_f64;
    let mut m_best = 0.0_f64;
    let mut s_best: Option<V> = None;

    let mut iters = 0_usize;
    let mut status = SteihaugStatus::MaxIter;
    let mut truncated_m: Option<f64> = None;

    for _ in 0..max_iter {
        let q = aop(&p);
        iters += 1;
        let pq = p.dot(&q);

        let ss = ball_dot(&s, &s);
        let sp = ball_dot(&s, &p);
        let pp = ball_dot(&p, &p);

        if pq <= 0.0 {
            let sigma = boundary_root(ss, sp, pp, delta);
            s.axpy(sigma, &p);
            r.axpy(sigma, &q);
            let m_t = 0.5 * (g.dot(&s) + s.dot(&r));
            status = SteihaugStatus::NegCurve;
            truncated_m = Some(m_t);
            break;
        }
        let alpha = rho / pq;
        if ss + alpha * (2.0 * sp + alpha * pp) > delta * delta {
            let sigma = boundary_root(ss, sp, pp, delta);
            s.axpy(sigma, &p);
            r.axpy(sigma, &q);
            let m_t = 0.5 * (g.dot(&s) + s.dot(&r));
            status = SteihaugStatus::Boundary;
            truncated_m = Some(m_t);
            break;
        }

        s.axpy(alpha, &p);
        r.axpy(alpha, &q);
        let m_new = 0.5 * (g.dot(&s) + s.dot(&r));
        debug_assert!(
            m_new.is_finite(),
            "steihaug_cg_ball: non-finite model value"
        );
        if m_new > m_best + 16.0 * f64::EPSILON * m_best.abs().max(1.0) {
            status = SteihaugStatus::Stagnated;
            m_cur = m_new;
            break;
        }
        m_cur = m_new;
        if m_new < m_best {
            m_best = m_new;
            s_best = Some(s.clone());
        }

        let rho_new = r.dot(&r);
        resnorm = rho_new.max(0.0).sqrt();
        if resnorm <= tol {
            status = SteihaugStatus::Interior;
            break;
        }
        if rho_new <= 0.0 {
            status = SteihaugStatus::Stagnated;
            break;
        }
        let beta = rho_new / rho;
        p.scale(beta);
        p.axpy(-1.0, &r);
        rho = rho_new;
    }

    let (s_ret, m_ret, boundary_hit) = match truncated_m {
        Some(m_t) if m_t <= m_best => (s, m_t, true),
        Some(_) => (take_best(s_best, g), m_best, false),
        None if m_cur <= m_best => (s, m_cur, false),
        None => (take_best(s_best, g), m_best, false),
    };

    (
        s_ret,
        SteihaugInfo {
            status,
            iters,
            model_red: -m_ret,
            gnorm,
            resnorm,
            boundary_hit,
        },
    )
}

#[cfg(test)]
pub(crate) mod testvec {
    use crate::solvers::InnerProductVector;

    #[derive(Clone, Debug)]
    pub(crate) struct DVec(pub Vec<f64>);

    impl InnerProductVector for DVec {
        type Scalar = f64;
        fn dot(&self, other: &Self) -> f64 {
            self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum()
        }
        fn norm(&self) -> f64 {
            self.dot(self).max(0.0).sqrt()
        }
        fn axpy(&mut self, a: f64, other: &Self) {
            for (x, y) in self.0.iter_mut().zip(&other.0) {
                *x += a * y;
            }
        }
        fn scale(&mut self, a: f64) {
            for x in self.0.iter_mut() {
                *x *= a;
            }
        }
        fn for_each_coeff_mut<F: FnMut(&mut f64)>(&mut self, mut f: F) {
            for x in self.0.iter_mut() {
                f(x);
            }
        }
        fn zip_coeffs_mut<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, mut f: F) {
            for (x, &y) in self.0.iter_mut().zip(&other.0) {
                f(x, y);
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

    pub(crate) fn mat_apply(a: &[f64], n: usize, v: &DVec) -> DVec {
        let mut out = vec![0.0; n];
        for (i, o) in out.iter_mut().enumerate() {
            for j in 0..n {
                *o += a[i * n + j] * v.0[j];
            }
        }
        DVec(out)
    }

    pub(crate) fn model_value(a: &[f64], n: usize, g: &DVec, s: &DVec) -> f64 {
        let av = mat_apply(a, n, s);
        g.dot(s) + 0.5 * s.dot(&av)
    }

    pub(crate) fn dense_solve(a: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
        let mut m = a.to_vec();
        let mut x = b.to_vec();
        for k in 0..n {
            let piv = (k..n)
                .max_by(|&i, &j| m[i * n + k].abs().total_cmp(&m[j * n + k].abs()))
                .unwrap();
            if piv != k {
                for j in 0..n {
                    m.swap(k * n + j, piv * n + j);
                }
                x.swap(k, piv);
            }
            let d = m[k * n + k];
            assert!(d.abs() > 1e-300, "singular test matrix");
            for i in k + 1..n {
                let f = m[i * n + k] / d;
                for j in k..n {
                    m[i * n + j] -= f * m[k * n + j];
                }
                x[i] -= f * x[k];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= m[i * n + j] * x[j];
            }
            x[i] /= m[i * n + i];
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::testvec::{DVec, dense_solve, mat_apply, model_value};
    use super::{SteihaugStatus, steihaug_cg, steihaug_cg_ball};
    use crate::solvers::InnerProductVector;

    fn spd5() -> (Vec<f64>, DVec) {
        #[rustfmt::skip]
        let a = vec![
            4.0, 1.0, 0.0, 0.5, 0.0,
            1.0, 5.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 6.0, 1.0, 0.5,
            0.5, 0.0, 1.0, 3.0, 1.0,
            0.0, 0.0, 0.5, 1.0, 2.0,
        ];
        let g = DVec(vec![1.0, -2.0, 0.5, 0.3, -1.0]);
        (a, g)
    }

    #[test]
    fn st1_interior_matches_dense_inverse() {
        let (a, g) = spd5();
        let (s, info) = steihaug_cg(|v| mat_apply(&a, 5, v), &g, 1e6, 1e-12, 0.0, 100);
        assert_eq!(info.status, SteihaugStatus::Interior);
        assert!(!info.boundary_hit);
        let neg_g: Vec<f64> = g.0.iter().map(|x| -x).collect();
        let s_exact = dense_solve(&a, 5, &neg_g);
        let err: f64 =
            s.0.iter()
                .zip(&s_exact)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt();
        let scale: f64 = s_exact.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err <= 1e-8 * scale, "err {err:e} vs scale {scale:e}");
        let m = model_value(&a, 5, &g, &s);
        assert!(
            (info.model_red - (-m)).abs() <= 1e-10 * m.abs().max(1.0),
            "model_red {} vs −m(s) {}",
            info.model_red,
            -m
        );
        assert!(info.model_red > 0.0);
    }

    #[test]
    fn st2_boundary_truncation_at_least_cauchy() {
        let (a, g) = spd5();
        let delta = 1e-2;
        let (s, info) = steihaug_cg(|v| mat_apply(&a, 5, v), &g, delta, 1e-12, 0.0, 100);
        assert_eq!(info.status, SteihaugStatus::Boundary);
        assert!(info.boundary_hit);
        assert!(
            (s.norm() - delta).abs() <= 1e-12 * delta,
            "‖s‖ = {}",
            s.norm()
        );
        let gnorm = g.norm();
        let ag = mat_apply(&a, 5, &g);
        let curv = g.dot(&ag) / (gnorm * gnorm);
        let t = (gnorm / curv).min(delta);
        let m_cauchy = -t * gnorm + 0.5 * t * t * curv;
        assert!(
            info.model_red >= -m_cauchy * (1.0 - 1e-12),
            "model_red {} < Cauchy {}",
            info.model_red,
            -m_cauchy
        );
    }

    #[test]
    fn st3a_negcurv_first_iteration() {
        #[rustfmt::skip]
        let a = vec![
            -1.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
        ];
        let g = DVec(vec![1.0, 0.0, 0.0]);
        let delta = 0.7;
        let (s, info) = steihaug_cg(|v| mat_apply(&a, 3, v), &g, delta, 1e-12, 0.0, 100);
        assert_eq!(info.status, SteihaugStatus::NegCurve);
        assert_eq!(info.iters, 1);
        assert!(info.boundary_hit);
        assert!((s.norm() - delta).abs() <= 1e-12 * delta);
        assert!(info.model_red > 0.0);
    }

    #[test]
    fn st3b_negcurv_mid_cg_model_reeval() {
        #[rustfmt::skip]
        let a = vec![
            2.0, 0.3, 0.1,
            0.3, 2.0, 0.4,
            0.1, 0.4, -1.0,
        ];
        let g = DVec(vec![1.0, 1.0, 0.1]);
        let gag = {
            let ag = mat_apply(&a, 3, &g);
            g.dot(&ag)
        };
        assert!(
            gag > 0.0,
            "test premise: first direction must be positive-curvature"
        );
        let (s, info) = steihaug_cg(|v| mat_apply(&a, 3, v), &g, 1e6, 1e-14, 0.0, 100);
        assert_eq!(info.status, SteihaugStatus::NegCurve);
        assert!(
            info.iters >= 2,
            "negative curvature must be found mid-CG, iters = {}",
            info.iters
        );
        let m = model_value(&a, 3, &g, &s);
        assert!(
            (info.model_red - (-m)).abs() <= 1e-10 * m.abs().max(1.0),
            "model_red {} vs −m(s) {}",
            info.model_red,
            -m
        );
    }

    #[test]
    fn st4_forcing_and_maxiter() {
        let (a, g) = spd5();
        let (_, info) = steihaug_cg(|v| mat_apply(&a, 5, v), &g, 1e6, 1e-14, 0.0, 1);
        assert_eq!(info.status, SteihaugStatus::MaxIter);
        assert_eq!(info.iters, 1);
        assert!(
            info.model_red > 0.0,
            "one CG step is the Cauchy point — it must decrease"
        );
        let (_, info) = steihaug_cg(|v| mat_apply(&a, 5, v), &g, 1e6, 0.99, 0.0, 100);
        assert_eq!(info.status, SteihaugStatus::Interior);
        assert!(
            info.iters <= 2,
            "rtol 0.99 must stop within 2 applies, took {}",
            info.iters
        );
    }

    #[test]
    fn st5_asymmetric_stagnation_best_iterate() {
        #[rustfmt::skip]
        let a = vec![
            1.0,  30.0, 0.0,
            -30.0, 1.0, 0.5,
            0.0,  -0.5, 2.0,
        ];
        let g = DVec(vec![1.0, 0.7, -0.3]);
        let (s, info) = steihaug_cg(|v| mat_apply(&a, 3, v), &g, 1e6, 1e-14, 0.0, 50);
        assert_eq!(info.status, SteihaugStatus::Stagnated);
        assert!(!info.boundary_hit);
        assert!(info.model_red >= 0.0);
        let m = model_value(&a, 3, &g, &s);
        assert!(
            (info.model_red - (-m)).abs() <= 1e-10 * m.abs().max(1.0),
            "returned vector's model value must match model_red"
        );
    }

    #[test]
    fn st5b_truncated_point_can_lose_to_best() {
        let w = 3.0;
        #[rustfmt::skip]
        let a = vec![
            1.0,  w,
             -w, 1.0,
        ];
        let g = DVec(vec![1.0, 0.0]);
        let (s, info) = steihaug_cg(|v| mat_apply(&a, 2, v), &g, 1.5, 1e-14, 0.0, 50);
        assert_eq!(info.status, SteihaugStatus::Boundary);
        assert!(
            !info.boundary_hit,
            "truncated point lost to the interior best"
        );
        assert!(
            (info.model_red - 0.5).abs() <= 1e-12,
            "model_red = {}",
            info.model_red
        );
        assert!(
            (s.0[0] + 1.0).abs() <= 1e-12 && s.0[1].abs() <= 1e-12,
            "s = {:?}",
            s.0
        );
        let m = model_value(&a, 2, &g, &s);
        assert!((info.model_red - (-m)).abs() <= 1e-12);
    }

    #[test]
    fn st6_zero_gradient_zero_step() {
        let (a, _) = spd5();
        let g = DVec(vec![0.0; 5]);
        let (s, info) = steihaug_cg(|v| mat_apply(&a, 5, v), &g, 1.0, 1e-12, 0.0, 100);
        assert_eq!(info.status, SteihaugStatus::Interior);
        assert_eq!(info.iters, 0, "no operator application on a zero gradient");
        assert_eq!(s.norm(), 0.0);
        assert_eq!(info.model_red, 0.0);
    }

    #[test]
    #[should_panic(expected = "trust radius")]
    fn st7a_radius_zero() {
        let (a, g) = spd5();
        let _ = steihaug_cg(|v| mat_apply(&a, 5, v), &g, 0.0, 1e-12, 0.0, 10);
    }

    #[test]
    #[should_panic(expected = "trust radius")]
    fn st7b_radius_negative() {
        let (a, g) = spd5();
        let _ = steihaug_cg(|v| mat_apply(&a, 5, v), &g, -1.0, 1e-12, 0.0, 10);
    }

    #[test]
    #[should_panic(expected = "trust radius")]
    fn st7c_radius_infinite() {
        let (a, g) = spd5();
        let _ = steihaug_cg(|v| mat_apply(&a, 5, v), &g, f64::INFINITY, 1e-12, 0.0, 10);
    }

    #[test]
    #[should_panic(expected = "trust radius")]
    fn st7d_radius_square_underflows() {
        let (a, g) = spd5();
        let _ = steihaug_cg(|v| mat_apply(&a, 5, v), &g, 1e-170, 1e-12, 0.0, 10);
    }

    fn wdot(u: &DVec, v: &DVec) -> f64 {
        const W: [f64; 5] = [1.0, 4.0, 9.0, 4.0, 1.0];
        u.0.iter()
            .zip(&v.0)
            .zip(W.iter())
            .map(|((a, b), w)| w * a * b)
            .sum()
    }

    #[test]
    fn stb1_truncation_at_weighted_norm() {
        let (a, g) = spd5();
        let delta = 1e-2;
        let (s, info) =
            steihaug_cg_ball(|v| mat_apply(&a, 5, v), &g, delta, 1e-12, 0.0, 100, &wdot);
        assert_eq!(info.status, SteihaugStatus::Boundary);
        assert!(info.boundary_hit);
        let ball_norm = wdot(&s, &s).sqrt();
        assert!(
            (ball_norm - delta).abs() <= 1e-12 * delta,
            "‖s‖_ball = {ball_norm:e} vs Δ = {delta:e}"
        );
        assert!(
            (s.norm() - delta).abs() > 1e-3 * delta,
            "anti-vacuity: the Euclidean norm must differ resolvably from Δ, got ‖s‖₂ = {:e}",
            s.norm()
        );
        let m = model_value(&a, 5, &g, &s);
        assert!(
            (info.model_red - (-m)).abs() <= 1e-10 * m.abs().max(1.0),
            "model_red {} vs −m(s) {}",
            info.model_red,
            -m
        );

        #[rustfmt::skip]
        let an = vec![
             1.0, 0.0,  0.0, 0.0, 0.0,
             0.0, 1.0,  0.0, 0.0, 0.0,
             0.0, 0.0, -1.0, 0.0, 0.0,
             0.0, 0.0,  0.0, 1.0, 0.0,
             0.0, 0.0,  0.0, 0.0, 1.0,
        ];
        let gn = DVec(vec![0.0, 0.0, 1.0, 0.0, 0.0]);
        let (sn, infon) =
            steihaug_cg_ball(|v| mat_apply(&an, 5, v), &gn, 0.7, 1e-12, 0.0, 100, &wdot);
        assert_eq!(infon.status, SteihaugStatus::NegCurve);
        assert!(infon.boundary_hit);
        let bn = wdot(&sn, &sn).sqrt();
        assert!((bn - 0.7).abs() <= 1e-12 * 0.7, "‖s‖_ball = {bn:e}");
        assert!(
            (sn.norm() - 0.7 / 3.0).abs() <= 1e-12,
            "the Euclidean length of the ride must be Δ/3, got {:e}",
            sn.norm()
        );
    }

    #[test]
    fn stb2_plain_ball_matches_oracle() {
        let plain = |u: &DVec, v: &DVec| u.dot(v);
        let (a, g) = spd5();
        #[rustfmt::skip]
        let an = vec![
            -1.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let gn = DVec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let cases: [(&[f64], &DVec, f64); 3] = [(&a, &g, 1e6), (&a, &g, 1e-2), (&an, &gn, 0.7)];
        for (m, gv, delta) in cases {
            let (s0, i0) = steihaug_cg(|v| mat_apply(m, 5, v), gv, delta, 1e-12, 0.0, 100);
            let (s1, i1) =
                steihaug_cg_ball(|v| mat_apply(m, 5, v), gv, delta, 1e-12, 0.0, 100, &plain);
            assert_eq!(i0.status, i1.status);
            assert_eq!(i0.iters, i1.iters);
            assert_eq!(i0.boundary_hit, i1.boundary_hit);
            for (x, y) in s0.0.iter().zip(&s1.0) {
                assert!(
                    (x - y).abs() <= 1e-15 * x.abs().max(1.0),
                    "step mismatch at Δ = {delta:e}: {x:e} vs {y:e}"
                );
            }
            assert!(
                (i0.model_red - i1.model_red).abs() <= 1e-15 * i0.model_red.max(1.0),
                "model_red mismatch: {} vs {}",
                i0.model_red,
                i1.model_red
            );
        }
    }

    #[test]
    fn stb3_interior_unchanged_by_ball() {
        let (a, g) = spd5();
        let (s0, i0) = steihaug_cg(|v| mat_apply(&a, 5, v), &g, 1e6, 1e-12, 0.0, 100);
        let (s1, i1) = steihaug_cg_ball(|v| mat_apply(&a, 5, v), &g, 1e6, 1e-12, 0.0, 100, &wdot);
        assert_eq!(i0.status, SteihaugStatus::Interior);
        assert_eq!(i1.status, SteihaugStatus::Interior);
        assert_eq!(i0.iters, i1.iters);
        let scale = s0.norm();
        for (x, y) in s0.0.iter().zip(&s1.0) {
            assert!(
                (x - y).abs() <= 1e-14 * scale,
                "interior step must not depend on the ball: {x:e} vs {y:e}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "trust radius")]
    fn stb4_radius_door() {
        let (a, g) = spd5();
        let _ = steihaug_cg_ball(|v| mat_apply(&a, 5, v), &g, 0.0, 1e-12, 0.0, 10, &wdot);
    }
}
