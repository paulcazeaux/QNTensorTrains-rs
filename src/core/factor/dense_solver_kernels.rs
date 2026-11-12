// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::prelude::SolveLstsq;
use faer::{Mat, MatRef, Par};

use super::coreops::thin_svd_qr;
use super::ensure_seq;
use crate::scalar::Scalar;

/// Self-adjoint eigen of a small dense row-major `m×m` Hermitian matrix `s` (`m ≲ 8`, Davidson's
/// projected matrix): eigenvalues ascending, the first `want_cols` eigenvectors as columns (faer
/// `self_adjoint_eigen`, `Side::Lower`) — the faer-confined home for Davidson's small dense
/// eigensolve (`solvers::davidson` must not call faer directly). faer guarantees
/// ascending eigenvalues, so index 0 is provably the smallest — no manual argmin needed by the
/// caller.
pub fn dense_self_adjoint_smallest<T: Scalar>(
    s: &[T],
    m: usize,
    want_cols: usize,
) -> (Vec<f64>, Vec<Vec<T>>) {
    ensure_seq();
    let mut a = Mat::<T>::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            *a.get_mut(i, j) = s[i * m + j];
        }
    }
    let e = a
        .as_ref()
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("projected self-adjoint eigen");
    let evals: Vec<f64> = (0..m)
        .map(|c| e.S().column_vector().get(c).re_s())
        .collect();
    let evecs: Vec<Vec<T>> = (0..want_cols.min(m))
        .map(|c| (0..m).map(|r| *e.U().get(r, c)).collect())
        .collect();
    (evals, evecs)
}

/// Full self-adjoint eigendecomposition of a small dense Hermitian `a` (`n×n`), faer-confined
/// (`ensure_seq`, so it stays serial inside the crate's per-sector rayon loops). Returns
/// `(eigenvalues ASCENDING, U)` with `U`'s columns the matching eigenvectors. The perturbative
/// density-matrix truncation ([`crate::core::twosite::factor_svd_perturbative`]) diagonalizes
/// `ρ' = ρ + α·ρ_noise` per sector this way (White's density-matrix noise).
pub fn dense_self_adjoint_eigen<T: Scalar>(a: MatRef<'_, T>) -> (Vec<f64>, Mat<T>) {
    let n = a.nrows();
    if n == 0 {
        return (Vec::new(), Mat::<T>::zeros(0, 0));
    }
    ensure_seq();
    let e = a
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("self-adjoint eigen of ρ'");
    let evals: Vec<f64> = (0..n)
        .map(|i| e.S().column_vector().get(i).re_s())
        .collect();
    let mut u = Mat::<T>::zeros(n, n);
    let eu = e.U();
    for c in 0..n {
        for r in 0..n {
            *u.get_mut(r, c) = *eu.get(r, c);
        }
    }
    (evals, u)
}

/// `exp(a·T)·e₁` for the real symmetric tridiagonal `T` (`dv`: `k` diagonals,
/// `ev`: `k-1` off-diagonals), coefficient `a: T` (real for imaginary-time TDVP, complex for
/// real-time). The tridiagonal is real symmetric, so `T = QΛQᵀ` with a REAL spectrum `{λ_m}` and
/// a real orthogonal `Q`; then `u_i = Σ_m Q[i,m]·exp(a·λ_m)·Q[0,m]`. Because `u` is reconstructed
/// as a full spectral sum it is invariant under faer-vs-LAPACK eigenvector sign/ordering choices
/// Faer-confined home for the Lanczos [`crate::solvers::expmv`];
/// `solvers/` stays faer-free. `exp(a·λ)` uses [`Scalar::exp_s`] (complex-aware).
pub fn tridiag_exp_e1<T: Scalar>(dv: &[f64], ev: &[f64], a: T) -> Vec<T> {
    let k = dv.len();
    if k == 0 {
        return Vec::new();
    }
    debug_assert!(
        ev.len() + 1 == k || (k == 1 && ev.is_empty()),
        "tridiag_exp_e1: ev must have length dv.len()-1"
    );
    if k == 1 {
        return vec![(a * T::from_f64(dv[0])).exp_s()];
    }
    ensure_seq();
    let mut m = Mat::<f64>::zeros(k, k);
    for (i, &d) in dv.iter().enumerate() {
        *m.get_mut(i, i) = d;
    }
    for (i, &e) in ev.iter().enumerate() {
        *m.get_mut(i + 1, i) = e;
        *m.get_mut(i, i + 1) = e;
    }
    let e = m
        .as_ref()
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("tridiagonal self-adjoint eigen");
    let s = e.S();
    let q = e.U();
    let mut u = vec![T::zero(); k];
    for col in 0..k {
        let lam = s.column_vector().get(col).re_s();
        let expfac = (a * T::from_f64(lam)).exp_s();
        let coeff = expfac * T::from_f64(*q.get(0, col));
        for (i, ui) in u.iter_mut().enumerate() {
            *ui += T::from_f64(*q.get(i, col)) * coeff;
        }
    }
    u
}

/// General (deliberately *non*-symmetrized) eigendecomposition of a small row-major `n×n` matrix
/// `a` (`n ≲ m+1 = 11` in `solvers::inexact`): returns `(eigenvalues, eigenvectors-as-columns)`,
/// both complex ([`crate::scalar::C64`]) regardless of the scalar `T`. For real `T` faer runs the
/// real Schur back-substitution and packs complex-conjugate pairs into complex form; the solver
/// takes `.re_s()` of whichever component it needs. Faer-confined (`ensure_seq`); deterministic (no
/// RNG). Under inexact matvecs the projected Hessenberg is not Hermitian, so
/// [`dense_self_adjoint_smallest`] is the wrong tool — symmetrizing it would manufacture
/// sub-spectral ghost eigenvalues.
pub fn dense_eigen_general<T: Scalar>(
    a: &[T],
    n: usize,
) -> (Vec<crate::scalar::C64>, Vec<Vec<crate::scalar::C64>>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    ensure_seq();
    let mut m = Mat::<T>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *m.get_mut(i, j) = a[i * n + j];
        }
    }
    let e = m
        .as_ref()
        .eigen()
        .expect("general dense eigendecomposition");
    let s = e.S();
    let svec = s.column_vector();
    let evals: Vec<crate::scalar::C64> = (0..n).map(|c| *svec.get(c)).collect();
    let u = e.U();
    let mut evecs: Vec<Vec<crate::scalar::C64>> = (0..n)
        .map(|c| (0..n).map(|r| *u.get(r, c)).collect())
        .collect();
    let mut evals = evals;
    if !T::IS_COMPLEX {
        let re: Vec<f64> = a.iter().map(|x| x.re_s()).collect();
        repair_repeated_eigenpairs(&re, n, &mut evals, &mut evecs);
    }
    (evals, evecs)
}

fn eig_residual(m: &[f64], n: usize, lam: crate::scalar::C64, v: &[crate::scalar::C64]) -> f64 {
    let mut r2 = 0.0f64;
    for i in 0..n {
        let mut acc = -lam * v[i];
        for (j, vj) in v.iter().enumerate() {
            acc += *vj * m[i * n + j];
        }
        r2 += acc.norm_sqr();
    }
    let nv = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    if nv == 0.0 || !nv.is_finite() {
        return f64::INFINITY;
    }
    r2.sqrt() / nv
}

fn repair_repeated_eigenpairs(
    m: &[f64],
    n: usize,
    evals: &mut [crate::scalar::C64],
    evecs: &mut [Vec<crate::scalar::C64>],
) {
    use crate::scalar::C64;
    let mnorm = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    let tol = 1e3 * n as f64 * f64::EPSILON * mnorm;
    let ctol = 1e-6 * mnorm;
    let fails: Vec<bool> = (0..n)
        .map(|c| {
            let r = eig_residual(m, n, evals[c], &evecs[c]);
            r.is_nan() || r > tol
        })
        .collect();
    let mut done = vec![false; n];
    for f in 0..n {
        if !fails[f] || done[f] || !(evals[f].re.is_finite() && evals[f].im.is_finite()) {
            continue;
        }
        let cluster: Vec<usize> = (0..n)
            .filter(|&c| !done[c] && (evals[c] - evals[f]).norm() <= ctol)
            .collect();
        cluster.iter().for_each(|&c| done[c] = true);
        let mean = cluster
            .iter()
            .fold(C64::new(0.0, 0.0), |s, &c| s + evals[c])
            / cluster.len() as f64;
        let pairs = cluster_null_space(m, n, mean, cluster.len(), mean.im.abs() <= ctol);
        if pairs
            .iter()
            .all(|(lam, v)| eig_residual(m, n, *lam, v) <= tol)
        {
            for (&c, (lam, v)) in cluster.iter().zip(pairs) {
                evals[c] = lam;
                evecs[c] = v;
            }
        }
    }
}

pub(crate) fn cluster_null_space(
    m: &[f64],
    n: usize,
    lam: crate::scalar::C64,
    r: usize,
    real: bool,
) -> Vec<(crate::scalar::C64, Vec<crate::scalar::C64>)> {
    use crate::scalar::C64;
    let vecs: Vec<Vec<C64>> = if real {
        let a = Mat::<f64>::from_fn(n, n, |i, j| {
            m[i * n + j] - if i == j { lam.re } else { 0.0 }
        });
        let v = thin_svd_qr(a.as_ref()).v;
        (n - r..n)
            .map(|c| (0..n).map(|i| C64::new(v[(i, c)], 0.0)).collect())
            .collect()
    } else {
        let a = Mat::<C64>::from_fn(n, n, |i, j| {
            C64::new(m[i * n + j], 0.0) - if i == j { lam } else { C64::new(0.0, 0.0) }
        });
        let v = thin_svd_qr(a.as_ref()).v;
        (n - r..n)
            .map(|c| (0..n).map(|i| v[(i, c)]).collect())
            .collect()
    };
    vecs.into_iter()
        .map(|v| {
            let mut q = C64::new(0.0, 0.0);
            for i in 0..n {
                let mv: C64 = (0..n).map(|j| v[j] * m[i * n + j]).sum();
                q += v[i].conj() * mv;
            }
            (if real { C64::new(q.re, 0.0) } else { q }, v)
        })
        .collect()
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SignedRitzPath {
    Definite,
    General,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct SignedRitz {
    pub(crate) theta: Vec<f64>,
    pub(crate) y: Vec<Vec<f64>>,
    pub(crate) sign: Vec<i8>,
    pub(crate) complex: Vec<(crate::scalar::C64, Vec<f64>, Vec<f64>)>,
    pub(crate) path: SignedRitzPath,
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn signed_ritz(t: &[f64], g: &[f64], m: usize) -> SignedRitz {
    use faer::linalg::triangular_solve::{
        solve_lower_triangular_in_place, solve_upper_triangular_in_place,
    };
    assert_eq!(t.len(), m * m, "signed_ritz: T is not m×m");
    assert_eq!(g.len(), m * m, "signed_ritz: G is not m×m");
    ensure_seq();
    let sym = |a: &[f64], i: usize, j: usize| 0.5 * (a[i * m + j] + a[j * m + i]);
    let diag_ok = (0..m).all(|i| t[i * m + i] > 0.0 && t[i * m + i].is_finite());
    if diag_ok && m > 0 {
        let d: Vec<f64> = (0..m).map(|i| 1.0 / t[i * m + i].sqrt()).collect();
        let te = Mat::<f64>::from_fn(m, m, |i, j| d[i] * sym(t, i, j) * d[j]);
        let ge = Mat::<f64>::from_fn(m, m, |i, j| d[i] * sym(g, i, j) * d[j]);
        if let Ok(llt) = te.as_ref().llt(faer::Side::Lower) {
            let l = llt.L();
            let mut x = ge.clone();
            solve_lower_triangular_in_place(l, x.as_mut(), Par::Seq);
            let mut sm = x.transpose().to_owned();
            solve_lower_triangular_in_place(l, sm.as_mut(), Par::Seq);
            let ssym = Mat::<f64>::from_fn(m, m, |i, j| 0.5 * (sm[(i, j)] + sm[(j, i)]));
            let e = ssym
                .as_ref()
                .self_adjoint_eigen(faer::Side::Lower)
                .expect("signed_ritz: definite-path symmetric eigen");
            let mut pairs: Vec<(f64, Vec<f64>, i8)> = Vec::with_capacity(m);
            for k in 0..m {
                let mu = *e.S().column_vector().get(k);
                if mu == 0.0 || !mu.is_finite() {
                    continue;
                }
                let mut yk = Mat::<f64>::from_fn(m, 1, |r, _| *e.U().get(r, k));
                solve_upper_triangular_in_place(l.transpose(), yk.as_mut(), Par::Seq);
                let s = 1.0 / mu.abs().sqrt();
                let y: Vec<f64> = (0..m).map(|r| d[r] * yk[(r, 0)] * s).collect();
                pairs.push((1.0 / mu, y, if mu > 0.0 { 1 } else { -1 }));
            }
            pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut out = SignedRitz {
                theta: Vec::with_capacity(pairs.len()),
                y: Vec::with_capacity(pairs.len()),
                sign: Vec::with_capacity(pairs.len()),
                complex: Vec::new(),
                path: SignedRitzPath::Definite,
            };
            for (th, y, sg) in pairs {
                out.theta.push(th);
                out.y.push(y);
                out.sign.push(sg);
            }
            return out;
        }
    }
    use faer::prelude::Solve;
    let gm = Mat::<f64>::from_fn(m, m, |i, j| sym(g, i, j));
    let mut x = Mat::<f64>::from_fn(m, m, |i, j| sym(t, i, j));
    gm.as_ref().partial_piv_lu().solve_in_place(x.as_mut());
    let flat: Vec<f64> = (0..m * m).map(|ij| x[(ij / m, ij % m)]).collect();
    let (evals, evecs) = dense_eigen_general::<f64>(&flat, m);
    let gnorm = (0..m)
        .flat_map(|i| (0..m).map(move |j| (i, j)))
        .fold(0.0f64, |a, (i, j)| a.max(sym(g, i, j).abs()));
    let scale = evals
        .iter()
        .fold(0.0f64, |a, z| a.max(z.norm()))
        .max(f64::MIN_POSITIVE);
    let mut real: Vec<(f64, Vec<f64>, i8)> = Vec::new();
    let mut complex: Vec<(crate::scalar::C64, Vec<f64>, Vec<f64>)> = Vec::new();
    for (lam, v) in evals.iter().zip(&evecs) {
        if lam.im.abs() <= 1e-12 * scale {
            let piv = v
                .iter()
                .copied()
                .max_by(|a, b| a.norm().total_cmp(&b.norm()))
                .unwrap_or(crate::scalar::C64::new(1.0, 0.0));
            let ph = if piv.norm() > 0.0 {
                piv.conj() / piv.norm()
            } else {
                crate::scalar::C64::new(1.0, 0.0)
            };
            let mut y: Vec<f64> = v.iter().map(|c| (c * ph).re).collect();
            let n2: f64 = y.iter().map(|a| a * a).sum::<f64>().max(f64::MIN_POSITIVE);
            let yn = n2.sqrt();
            y.iter_mut().for_each(|a| *a /= yn);
            let ygy: f64 = (0..m)
                .map(|i| y[i] * (0..m).map(|j| sym(g, i, j) * y[j]).sum::<f64>())
                .sum();
            let sign = if ygy.abs() <= 1e-10 * gnorm.max(f64::MIN_POSITIVE) {
                0
            } else {
                let s = 1.0 / ygy.abs().sqrt();
                y.iter_mut().for_each(|a| *a *= s);
                if ygy > 0.0 { 1 } else { -1 }
            };
            real.push((lam.re, y, sign));
        } else if lam.im > 0.0 {
            let n2: f64 = v
                .iter()
                .map(|c| c.norm_sqr())
                .sum::<f64>()
                .max(f64::MIN_POSITIVE);
            let yn = n2.sqrt();
            let mut yr: Vec<f64> = v.iter().map(|c| c.re / yn).collect();
            let mut yi: Vec<f64> = v.iter().map(|c| c.im / yn).collect();
            g_orthogonal_split(&mut yr, &mut yi, g, m);
            complex.push((*lam, yr, yi));
        }
    }
    real.sort_by(|a, b| a.0.total_cmp(&b.0));
    complex.sort_by(|a, b| a.0.re.total_cmp(&b.0.re));
    let mut out = SignedRitz {
        theta: Vec::with_capacity(real.len()),
        y: Vec::with_capacity(real.len()),
        sign: Vec::with_capacity(real.len()),
        complex,
        path: SignedRitzPath::General,
    };
    for (th, y, sg) in real {
        out.theta.push(th);
        out.y.push(y);
        out.sign.push(sg);
    }
    out
}

pub(crate) fn g_orthogonal_split(yr: &mut [f64], yi: &mut [f64], g: &[f64], m: usize) {
    let sym = |i: usize, j: usize| 0.5 * (g[i * m + j] + g[j * m + i]);
    let gf = |a: &[f64], b: &[f64]| -> f64 {
        (0..m)
            .map(|i| a[i] * (0..m).map(|j| sym(i, j) * b[j]).sum::<f64>())
            .sum()
    };
    let (alpha, gamma) = (0.5 * (gf(yr, yr) - gf(yi, yi)), gf(yr, yi));
    let phi = 0.5 * (-gamma).atan2(alpha);
    let (c, s) = (phi.cos(), phi.sin());
    for (r, i) in yr.iter_mut().zip(yi.iter_mut()) {
        let (a, b) = (*r, *i);
        *r = a * c - b * s;
        *i = a * s + b * c;
    }
}

/// Column-pivoted-QR least squares `argmin_x ‖[cols]·x − rhs‖₂`. `cols[j]` is the `j`-th column
/// (each length `nrows`); returns `x ∈ T^{cols.len()}`. faer `col_piv_qr().solve_lstsq` —
/// column-pivoted (rank-revealing) QR, which handles the near-rank-deficiency the sketched-GMRES
/// `D`/window Gram matrices develop as the iteration converges. `solvers/` stays faer-free by
/// routing both the window `h` fit and the `y` fit through this one helper, and BOTH must stay
/// pivoted: an unpivoted solve breaks down exactly where these fits get interesting.
pub fn dense_lstsq_colpiv<T: Scalar>(nrows: usize, cols: &[&[T]], rhs: &[T]) -> Vec<T> {
    let ncols = cols.len();
    if ncols == 0 {
        return Vec::new();
    }
    assert_eq!(rhs.len(), nrows, "rhs length must equal nrows");
    ensure_seq();
    let mut a = Mat::<T>::zeros(nrows, ncols);
    for (j, col) in cols.iter().enumerate() {
        assert_eq!(col.len(), nrows, "each column length must equal nrows");
        for (i, &c) in col.iter().enumerate() {
            *a.get_mut(i, j) = c;
        }
    }
    let mut b = Mat::<T>::zeros(nrows, 1);
    for (i, &r) in rhs.iter().enumerate() {
        *b.get_mut(i, 0) = r;
    }
    let x = a.col_piv_qr().solve_lstsq(&b);
    (0..ncols).map(|i| *x.get(i, 0)).collect()
}

#[cfg(test)]
mod factor_inx2_tests {}

#[cfg(test)]
mod eigen_general_tests {}

#[cfg(test)]
mod signed_ritz_tests {
    use super::{
        SignedRitzPath, dense_eigen_general, dense_self_adjoint_smallest, g_orthogonal_split,
        signed_ritz,
    };
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn matvec(a: &[f64], m: usize, y: &[f64]) -> Vec<f64> {
        (0..m)
            .map(|i| (0..m).map(|j| a[i * m + j] * y[j]).sum())
            .collect()
    }

    fn quad(a: &[f64], m: usize, y: &[f64]) -> f64 {
        y.iter().zip(matvec(a, m, y)).map(|(p, q)| p * q).sum()
    }

    fn pencil(m: usize, n_neg: usize, eps: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let a: Vec<f64> = (0..m * m).map(|_| rng.random_range(-1.0..1.0)).collect();
        let mut t = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                t[i * m + j] = (0..m).map(|k| a[k * m + i] * a[k * m + j]).sum::<f64>()
                    + if i == j { 1.0 } else { 0.0 };
            }
        }
        let mut g = vec![0.0; m * m];
        for i in 0..m {
            g[i * m + i] = if i < n_neg { -1.0 } else { 1.0 };
        }
        for i in 0..m {
            for j in i..m {
                let s = eps * rng.random_range(-1.0..1.0);
                g[i * m + j] += s;
                if i != j {
                    g[j * m + i] += s;
                }
            }
        }
        (t, g)
    }

    fn reference_eigs(t: &[f64], g: &[f64], m: usize) -> Vec<crate::scalar::C64> {
        let mut aug: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let mut r = g[i * m..(i + 1) * m].to_vec();
                r.extend_from_slice(&t[i * m..(i + 1) * m]);
                r
            })
            .collect();
        for c in 0..m {
            let p = (c..m)
                .max_by(|&a, &b| aug[a][c].abs().total_cmp(&aug[b][c].abs()))
                .unwrap();
            aug.swap(c, p);
            let piv = aug[c][c];
            for v in aug[c].iter_mut() {
                *v /= piv;
            }
            for r in 0..m {
                if r != c {
                    let f = aug[r][c];
                    let row_c = aug[c].clone();
                    for (x, y) in aug[r].iter_mut().zip(row_c) {
                        *x -= f * y;
                    }
                }
            }
        }
        let x: Vec<f64> = (0..m * m).map(|ij| aug[ij / m][m + ij % m]).collect();
        dense_eigen_general::<f64>(&x, m).0
    }

    #[test]
    fn definite_path_matches_the_general_eigenvalues_and_counts_the_negatives() {
        let m = 12;
        let n_neg = 3;
        let (t, g) = pencil(m, n_neg, 1e-3, 0x51_6E);
        let out = signed_ritz(&t, &g, m);
        assert_eq!(out.path, SignedRitzPath::Definite);
        assert!(out.complex.is_empty());
        assert_eq!(out.theta.len(), m);
        let mut refs: Vec<f64> = reference_eigs(&t, &g, m)
            .iter()
            .map(|z| {
                assert!(
                    z.im.abs() <= 1e-9 * z.norm(),
                    "definite pencil: complex {z}"
                );
                z.re
            })
            .collect();
        refs.sort_by(f64::total_cmp);
        let scale = refs.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        for (k, (&a, &b)) in out.theta.iter().zip(&refs).enumerate() {
            assert!(
                (a - b).abs() <= 1e-10 * scale,
                "θ[{k}] {a} vs reference {b}"
            );
        }
        assert_eq!(out.theta.iter().filter(|&&x| x < 0.0).count(), n_neg);
        let tn = t.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        for k in 0..m {
            let y = &out.y[k];
            let ty = matvec(&t, m, y);
            let gy = matvec(&g, m, y);
            let res: f64 = ty
                .iter()
                .zip(&gy)
                .map(|(p, q)| (p - out.theta[k] * q).powi(2))
                .sum::<f64>()
                .sqrt();
            let yn = y.iter().map(|a| a * a).sum::<f64>().sqrt();
            assert!(
                res <= 1e-10 * tn * yn,
                "pencil residual {res:.2e} at θ[{k}]"
            );
            assert!(
                (quad(&g, m, y).abs() - 1.0).abs() <= 1e-10,
                "|yᵀGy| ≠ 1 at {k}"
            );
            assert_eq!(out.sign[k], if out.theta[k] > 0.0 { 1 } else { -1 });
        }
    }

    #[test]
    fn definite_path_is_invariant_under_a_badly_scaled_basis_vector() {
        let m = 10;
        let (t, g) = pencil(m, 2, 0.0, 0xE9_01);
        let base = signed_ritz(&t, &g, m);
        let c = 1e4;
        let d: Vec<f64> = (0..m).map(|i| if i == 1 { c } else { 1.0 }).collect();
        let ts: Vec<f64> = (0..m * m).map(|ij| d[ij / m] * t[ij] * d[ij % m]).collect();
        let gs: Vec<f64> = (0..m * m).map(|ij| d[ij / m] * g[ij] * d[ij % m]).collect();
        let scaled = signed_ritz(&ts, &gs, m);
        assert_eq!(scaled.path, SignedRitzPath::Definite);
        let scale = base.theta.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        for (k, (&a, &b)) in base.theta.iter().zip(&scaled.theta).enumerate() {
            assert!((a - b).abs() <= 1e-9 * scale, "θ[{k}] moved: {a} vs {b}");
        }
    }

    #[test]
    fn general_path_reports_the_analytic_complex_pair() {
        let t = [1.0, 2.0, 2.0, 1.0];
        let g = [1.0, 0.0, 0.0, -1.0];
        let out = signed_ritz(&t, &g, 2);
        assert_eq!(out.path, SignedRitzPath::General);
        assert!(out.theta.is_empty(), "no real Ritz value: {:?}", out.theta);
        assert_eq!(out.complex.len(), 1);
        let (lam, yr, yi) = &out.complex[0];
        assert!(
            lam.re.abs() <= 1e-14 && (lam.im - 3f64.sqrt()).abs() <= 1e-14,
            "{lam}"
        );
        for i in 0..2 {
            let (mut re, mut im) = (0.0, 0.0);
            for j in 0..2 {
                let tg_re = t[i * 2 + j] - lam.re * g[i * 2 + j];
                let tg_im = -lam.im * g[i * 2 + j];
                re += tg_re * yr[j] - tg_im * yi[j];
                im += tg_re * yi[j] + tg_im * yr[j];
            }
            assert!(
                re.abs() <= 1e-14 && im.abs() <= 1e-14,
                "row {i}: ({re}, {im})"
            );
        }
    }

    #[test]
    fn g_orthogonal_split_makes_the_complex_parts_g_orthogonal_with_opposite_g_norms() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9A5E);
        let m = 6;
        let mut pairs = 0usize;
        for _ in 0..40 {
            let mut t = vec![0.0; m * m];
            for i in 0..m {
                for j in i..m {
                    let v: f64 = rng.random_range(-1.0..1.0);
                    t[i * m + j] = v;
                    t[j * m + i] = v;
                }
            }
            let mut g = vec![0.0; m * m];
            for i in 0..m {
                g[i * m + i] = if i < 3 { 1.0 } else { -1.0 };
            }
            let mut out = signed_ritz(&t, &g, m);
            for (_, yr, yi) in out.complex.iter_mut() {
                let gfr = |a: &[f64], b: &[f64]| -> f64 {
                    (0..m)
                        .map(|i| a[i] * (0..m).map(|j| g[i * m + j] * b[j]).sum::<f64>())
                        .sum()
                };
                let n2: f64 = yr.iter().chain(yi.iter()).map(|v| v * v).sum();
                assert!(
                    gfr(yr, yi).abs() <= 1e-12 * n2,
                    "signed_ritz's pair is not split"
                );
                let phi: f64 = rng.random_range(0.0..std::f64::consts::TAU);
                let (c, s) = (phi.cos(), phi.sin());
                for (r, i) in yr.iter_mut().zip(yi.iter_mut()) {
                    let (a, b) = (*r, *i);
                    *r = a * c - b * s;
                    *i = a * s + b * c;
                }
                g_orthogonal_split(yr, yi, &g, m);
            }
            let gf = |a: &[f64], b: &[f64]| -> f64 {
                (0..m)
                    .map(|i| a[i] * (0..m).map(|j| g[i * m + j] * b[j]).sum::<f64>())
                    .sum()
            };
            for (lam, yr, yi) in &out.complex {
                pairs += 1;
                let (a, b, c) = (gf(yr, yr), gf(yi, yi), gf(yr, yi));
                let n2: f64 = yr.iter().chain(yi.iter()).map(|v| v * v).sum();
                assert!(
                    c.abs() <= 1e-12 * n2,
                    "parts not G-orthogonal: {c:e} (|y|² {n2})"
                );
                assert!(
                    a > 0.0 && b < 0.0,
                    "G-norms {a} / {b}: the real part carries +ρ"
                );
                assert!((a + b).abs() <= 1e-10 * n2, "Krein-neutral pair: {a} + {b}");
                for i in 0..m {
                    let (mut re, mut im) = (0.0, 0.0);
                    for j in 0..m {
                        let tg_re = t[i * m + j] - lam.re * g[i * m + j];
                        let tg_im = -lam.im * g[i * m + j];
                        re += tg_re * yr[j] - tg_im * yi[j];
                        im += tg_re * yi[j] + tg_im * yr[j];
                    }
                    assert!(
                        re.abs() <= 1e-10 && im.abs() <= 1e-10,
                        "row {i}: ({re}, {im})"
                    );
                }
            }
        }
        assert!(
            pairs >= 10,
            "nonvacuity: only {pairs} complex pairs in 40 pencils"
        );
    }

    #[test]
    fn general_path_keeps_real_ritz_values_with_their_signatures() {
        let t = [2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 3.0];
        let g = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
        let out = signed_ritz(&t, &g, 3);
        assert_eq!(out.path, SignedRitzPath::General);
        assert!(out.complex.is_empty());
        assert_eq!(out.theta, vec![-3.0, -1.0, 2.0]);
        assert_eq!(out.sign, vec![-1, 1, 1]);
    }

    #[test]
    fn identity_gram_reproduces_the_symmetric_eigenvalues() {
        let m = 9;
        let (t, _) = pencil(m, 0, 0.0, 0x1D_E7);
        let mut g = vec![0.0; m * m];
        for i in 0..m {
            g[i * m + i] = 1.0;
        }
        let out = signed_ritz(&t, &g, m);
        let (ev, _) = dense_self_adjoint_smallest::<f64>(&t, m, 0);
        assert_eq!(out.path, SignedRitzPath::Definite);
        for (k, (&a, &b)) in out.theta.iter().zip(&ev).enumerate() {
            assert!((a - b).abs() <= 1e-11 * ev[m - 1], "θ[{k}] {a} vs {b}");
        }
    }
}
