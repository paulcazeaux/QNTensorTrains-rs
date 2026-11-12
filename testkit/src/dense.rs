// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::dense_apply_string;
use qntt::coeff::CoeffTensor;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::integrals::Integrals;
use qntt::qn::QnSpace;
use qntt::round::RoundMode;
use qntt::{Orbital, Scalar, SparseHamiltonian, Spin, TtVector, dot, tt_state};

use crate::randomized_state;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn det_tt(d: usize, up_sites: &[usize], dn_sites: &[usize]) -> TtVector<f64> {
    let occ_up = occ_from_sites(d, up_sites);
    let occ_dn = occ_from_sites(d, dn_sites);
    tt_state::<f64>(&occ_up, &occ_dn)
}

fn occ_from_sites(d: usize, sites: &[usize]) -> Vec<bool> {
    let mut occ = vec![false; d];
    for &s in sites {
        occ[s] = true;
    }
    occ
}

pub fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur: Vec<usize> = (0..n).collect();
    if n == 0 {
        return vec![vec![]];
    }
    if n > d {
        return out;
    }
    loop {
        out.push(cur.clone());
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if cur[i] != i + d - n {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        cur[i] += 1;
        for j in i + 1..n {
            cur[j] = cur[j - 1] + 1;
        }
    }
}

pub fn dets_for(d: usize, nup: usize, ndn: usize) -> Vec<(Vec<bool>, Vec<bool>)> {
    combinations(d, nup)
        .into_iter()
        .flat_map(|up| {
            combinations(d, ndn)
                .into_iter()
                .map(move |dn| (up.clone(), dn))
        })
        .map(|(up, dn)| (occ_from_sites(d, &up), occ_from_sites(d, &dn)))
        .collect()
}

pub fn dense_apply_hamiltonian_12<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    vec: &[T],
) -> Vec<T> {
    let len = vec.len();
    let mut out = vec![T::zero(); len];
    let spins = [Spin::Up, Spin::Dn];
    let accumulate = |ops: &[(Orbital, bool)], coeff: T, out: &mut [T]| {
        let term = dense_apply_string(d, ops, vec);
        for (o, tm) in out.iter_mut().zip(term) {
            *o += tm * coeff;
        }
    };
    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c == T::zero() {
                continue;
            }
            for &s in &spins {
                accumulate(
                    &[(Orbital::new(i, s), true), (Orbital::new(j, s), false)],
                    c,
                    &mut out,
                );
            }
        }
    }
    let half = T::from_f64(0.5);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c == T::zero() {
                        continue;
                    }
                    for &s1 in &spins {
                        for &s2 in &spins {
                            let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                            let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                            if oi != ok && oj != ol {
                                accumulate(
                                    &[(oi, true), (ok, true), (ol, false), (oj, false)],
                                    c * half,
                                    &mut out,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn basis_index(occ_up: &[bool], occ_dn: &[bool]) -> usize {
    let d = occ_up.len();
    let mut idx = 0usize;
    for k in (0..d).rev() {
        let fam = (occ_up[k] as usize) | ((occ_dn[k] as usize) << 1);
        idx = (idx << 2) | fam;
    }
    idx
}

pub fn dense_epq_matrix(
    d: usize,
    p: usize,
    q: usize,
    dets: &[(Vec<bool>, Vec<bool>)],
) -> Vec<Vec<f64>> {
    let len = 1usize << (2 * d);
    let n = dets.len();
    let mut mat = vec![vec![0.0f64; n]; n];
    for (col, (ou, od)) in dets.iter().enumerate() {
        let mut e = vec![0.0f64; len];
        e[basis_index(ou, od)] = 1.0;
        let up = dense_apply_string(
            d,
            &[
                (Orbital::new(p, Spin::Up), true),
                (Orbital::new(q, Spin::Up), false),
            ],
            &e,
        );
        let dn = dense_apply_string(
            d,
            &[
                (Orbital::new(p, Spin::Dn), true),
                (Orbital::new(q, Spin::Dn), false),
            ],
            &e,
        );
        for (row, (ru, rd)) in dets.iter().enumerate() {
            let idx = basis_index(ru, rd);
            mat[row][col] = up[idx] + dn[idx];
        }
    }
    mat
}

pub fn dense_kappa_matrix(
    d: usize,
    kappa: &CoeffTensor<f64>,
    dets: &[(Vec<bool>, Vec<bool>)],
) -> Vec<Vec<f64>> {
    let n = dets.len();
    let mut out = vec![vec![0.0f64; n]; n];
    for p in 0..d {
        for q in 0..d {
            let kpq = kappa.get(&[p, q]);
            if kpq == 0.0 {
                continue;
            }
            let epq = dense_epq_matrix(d, p, q, dets);
            for i in 0..n {
                for j in 0..n {
                    out[i][j] += kpq * epq[i][j];
                }
            }
        }
    }
    out
}

pub fn dense_hamiltonian_matrix(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    dets: &[(Vec<bool>, Vec<bool>)],
) -> Vec<Vec<f64>> {
    let len = 1usize << (2 * d);
    let n = dets.len();
    let mut mat = vec![vec![0.0f64; n]; n];
    for (col, (ou, od)) in dets.iter().enumerate() {
        let mut e = vec![0.0f64; len];
        e[basis_index(ou, od)] = 1.0;
        let he = dense_apply_hamiltonian_12(d, t, v, &e);
        for (row, (ru, rd)) in dets.iter().enumerate() {
            mat[row][col] = he[basis_index(ru, rd)];
        }
    }
    mat
}

#[allow(clippy::needless_range_loop)]
pub fn dense_eigen(mat: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    let mut m = faer::Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *m.get_mut(i, j) = mat[i][j];
        }
    }
    let (evals, u) = dense_self_adjoint_eigen(m.as_ref());
    let evecs: Vec<Vec<f64>> = (0..n)
        .map(|c| (0..n).map(|r| *u.get(r, c)).collect())
        .collect();
    (evals, evecs)
}

pub fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let k = b.len();
    let m = b[0].len();
    let mut out = vec![vec![0.0f64; m]; n];
    for i in 0..n {
        for l in 0..k {
            let ail = a[i][l];
            if ail == 0.0 {
                continue;
            }
            for j in 0..m {
                out[i][j] += ail * b[l][j];
            }
        }
    }
    out
}

pub fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot_vec(row, v)).collect()
}

pub fn mat_transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = a[0].len();
    let mut out = vec![vec![0.0f64; n]; m];
    for i in 0..n {
        for j in 0..m {
            out[j][i] = a[i][j];
        }
    }
    out
}

pub fn flat_to_nested(v: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n).map(|i| v[i * n..(i + 1) * n].to_vec()).collect()
}

pub fn neg_flat(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| -x).collect()
}

pub fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn mat_identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect()
}

pub fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|row| row.iter().map(|x| x * s).collect())
        .collect()
}

pub fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

pub fn max_abs(a: &[Vec<f64>]) -> f64 {
    a.iter().flatten().fold(0.0f64, |acc, &x| acc.max(x.abs()))
}

pub fn max_abs_diff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let mut m = 0.0f64;
    for (ra, rb) in a.iter().zip(b) {
        for (&x, &y) in ra.iter().zip(rb) {
            m = m.max((x - y).abs());
        }
    }
    m
}

#[allow(clippy::needless_range_loop)]
pub fn mat_det(a: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut det = 1.0f64;
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if m[r][col].abs() > m[piv][col].abs() {
                piv = r;
            }
        }
        if m[piv][col].abs() < 1e-300 {
            return 0.0;
        }
        if piv != col {
            m.swap(piv, col);
            det = -det;
        }
        det *= m[col][col];
        for r in col + 1..n {
            let f = m[r][col] / m[col][col];
            for c in col..n {
                m[r][c] -= f * m[col][c];
            }
        }
    }
    det
}

pub fn mat_expm(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let bound = max_abs(a) * n as f64;
    let mut s = 0i32;
    let mut scale = 1.0f64;
    while bound * scale > 0.5 {
        scale *= 0.5;
        s += 1;
    }
    let scaled = mat_scale(a, scale);
    let mut term = mat_identity(n);
    let mut result = mat_identity(n);
    for k in 1..=40 {
        term = mat_scale(&mat_mul(&term, &scaled), 1.0 / k as f64);
        result = mat_add(&result, &term);
    }
    for _ in 0..s {
        result = mat_mul(&result, &result);
    }
    result
}

pub fn random_antisym_flat(n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for x in a.iter_mut() {
        *x = rng.sample::<f64, _>(StandardNormal);
    }
    let mut k = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            k[i * n + j] = 0.5 * (a[i * n + j] - a[j * n + i]);
        }
    }
    k
}

pub fn flat_to_coeff(d: usize, flat: &[f64]) -> CoeffTensor<f64> {
    let mut c = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            c.set(&[i, j], flat[i * d + j]);
        }
    }
    c
}

pub fn random_antisym_coeff(d: usize, rng: &mut Xoshiro256PlusPlus) -> CoeffTensor<f64> {
    flat_to_coeff(d, &random_antisym_flat(d, rng))
}

pub fn scale_flat(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

pub fn transpose_flat(v: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[j * n + i] = v[i * n + j];
        }
    }
    out
}

pub fn scale_coeff(c: &CoeffTensor<f64>, s: f64) -> CoeffTensor<f64> {
    let d = c.d;
    let mut out = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            out.set(&[i, j], c.get(&[i, j]) * s);
        }
    }
    out
}

pub fn strict_lower_only(c: &CoeffTensor<f64>) -> CoeffTensor<f64> {
    let d = c.d;
    let mut out = CoeffTensor::<f64>::zeros(d, 2);
    for p in 1..d {
        for q in 0..p {
            out.set(&[p, q], c.get(&[p, q]));
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
pub fn random_sym_t(d: usize, rng: &mut Xoshiro256PlusPlus) -> CoeffTensor<f64> {
    let mut a = vec![vec![0.0f64; d]; d];
    for row in a.iter_mut() {
        for x in row.iter_mut() {
            *x = rng.sample::<f64, _>(StandardNormal);
        }
    }
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], 0.5 * (a[i][j] + a[j][i]));
        }
    }
    t
}

pub fn random_sym_v(d: usize, rng: &mut Xoshiro256PlusPlus) -> CoeffTensor<f64> {
    let n = d * d * d * d;
    let mut b = vec![0.0f64; n];
    for x in b.iter_mut() {
        *x = rng.sample::<f64, _>(StandardNormal);
    }
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let val = 0.25
                        * (b[idx(i, j, k, l)]
                            + b[idx(j, i, l, k)]
                            + b[idx(k, l, i, j)]
                            + b[idx(l, k, j, i)]);
                    v.set(&[i, j, k, l], val);
                }
            }
        }
    }
    v
}

pub fn random_asym_t(d: usize, rng: &mut Xoshiro256PlusPlus) -> CoeffTensor<f64> {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], rng.sample::<f64, _>(StandardNormal));
        }
    }
    t
}

pub struct A5Fixture {
    pub space: QnSpace,
    pub d: usize,
    pub t: CoeffTensor<f64>,
    pub v: CoeffTensor<f64>,
    pub k: CoeffTensor<f64>,
    pub x: TtVector<f64>,
    pub ham: SparseHamiltonian<f64>,
    pub ints: Integrals<f64>,
    pub dets: Vec<(Vec<bool>, Vec<bool>)>,
    pub c: Vec<f64>,
}

pub fn build_a5_fixture(seed: u64) -> A5Fixture {
    let d = 4usize;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let t = random_sym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    let kfull = random_antisym_coeff(d, &mut rng);
    let k = scale_coeff(&kfull, 0.1);
    let ham = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-13, qntt::Pooled::NONE, RoundMode::Rel);
    let n = x.norm();
    x.scale_mut(1.0 / n);
    let ints = Integrals {
        t: t.clone(),
        v: v.clone(),
        e0: 0.0,
        norb: d,
        nelec: space.n(),
        ms2: 1,
        orbsym: Vec::new(),
        isym: None,
    };
    let dets = dets_for(d, space.nup, space.ndn);
    let c: Vec<f64> = dets
        .iter()
        .map(|(ou, od)| {
            let s: TtVector<f64> = tt_state(ou, od);
            dot(&x, &s)
        })
        .collect();
    A5Fixture {
        space,
        d,
        t,
        v,
        k,
        x,
        ham,
        ints,
        dets,
        c,
    }
}

pub fn assert_close(got: f64, want: f64, tol: f64, msg: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{msg}: got {got:e}, want {want:e}, |diff|={:e} > tol {tol:e}",
        (got - want).abs()
    );
}

pub fn assert_dense_close(got: &[f64], want: &[f64], tol: f64, msg: &str) {
    let num: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let den: f64 = want.iter().map(|b| b * b).sum::<f64>().sqrt().max(1e-8);
    assert!(
        num <= tol * den,
        "{msg}: rel-diff {:e} > tol {tol:e} (num={num:e}, den={den:e})",
        num / den
    );
}

#[cfg(test)]
mod self_tests {
    use super::{
        assert_close, dense_epq_matrix, dets_for, mat_det, mat_expm, mat_identity, max_abs_diff,
        random_sym_v,
    };
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t_util_dense_epq_diagonal_matches_occupation() {
        let d = 4;
        let dets = dets_for(d, 2, 1);
        for p in 0..d {
            let mat = dense_epq_matrix(d, p, p, &dets);
            for (i, (ou, od)) in dets.iter().enumerate() {
                let want = (ou[p] as i32 + od[p] as i32) as f64;
                assert_close(mat[i][i], want, 1e-13, "diag(E_pp) vs occupation");
                for (j, row) in mat.iter().enumerate() {
                    if j != i {
                        assert_close(row[i], 0.0, 1e-13, "E_pp off-diagonal must vanish");
                    }
                }
            }
        }
    }

    #[test]
    fn t_util_dense_linalg_sanity() {
        let id = mat_identity(5);
        assert_close(mat_det(&id), 1.0, 1e-12, "det(I)");
        let zero = vec![vec![0.0f64; 5]; 5];
        let e = mat_expm(&zero);
        assert!(max_abs_diff(&e, &id) < 1e-13, "exp(0) != I");
    }

    #[test]
    fn t_util_random_sym_v_pair_symmetries_exact() {
        let d = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xF00D);
        let v = random_sym_v(d, &mut rng);
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        let a = v.get(&[i, j, k, l]);
                        assert_close(
                            v.get(&[j, i, l, k]),
                            a,
                            1e-13 * a.abs().max(1.0),
                            "v_ijkl != v_jilk",
                        );
                        assert_close(
                            v.get(&[k, l, i, j]),
                            a,
                            1e-13 * a.abs().max(1.0),
                            "v_ijkl != v_klij",
                        );
                    }
                }
            }
        }
    }
}
