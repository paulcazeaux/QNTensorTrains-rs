// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! CIS at a closed-shell determinant, in the integrals' own MO basis: the dense singlet and
//! triplet single-excitation problems whose amplitudes `X_ai` weight the singles padding of an RPA
//! reference ([`crate::solvers::hess::pad_singles`]) and seed targeted RPA solves.

use crate::core::factor::dense_self_adjoint_smallest;
use crate::integrals::Integrals;

/// One CIS root: `x[i_k·nvirt + a_k] = X_ai`, unit norm.
#[derive(Clone, Debug)]
pub struct CisRoot {
    pub triplet: bool,
    /// Excitation energy `E − E_det` (Ha).
    pub energy: f64,
    pub x: Vec<f64>,
}

/// The virtual orbitals of `occ`: every orbital of `0..d` not in it, ascending.
pub fn virtuals(d: usize, occ: &[usize]) -> Vec<usize> {
    (0..d).filter(|p| !occ.contains(p)).collect()
}

fn check_occ(d: usize, occ: &[usize]) {
    assert!(
        occ.windows(2).all(|w| w[0] < w[1]) && occ.last().is_none_or(|&i| i < d),
        "cis: the occupied orbitals must be ascending, distinct and below d = {d}: {occ:?}"
    );
}

/// The Fock matrix `f_pq` (row-major `d × d`) of the determinant doubly occupying `occ`.
pub fn fock_occ(ints: &Integrals<f64>, occ: &[usize]) -> Vec<f64> {
    let d = ints.norb;
    check_occ(d, occ);
    let (t, v) = (&ints.t.data, &ints.v.data);
    let eri = |p: usize, q: usize, r: usize, s: usize| v[((p * d + q) * d + r) * d + s];
    let mut f = vec![0.0; d * d];
    for p in 0..d {
        for q in 0..d {
            f[p * d + q] = t[p * d + q]
                + occ
                    .iter()
                    .map(|&i| 2.0 * eri(p, q, i, i) - eri(p, i, i, q))
                    .sum::<f64>();
        }
    }
    f
}

/// [`fock_occ`] of the determinant occupying `0..nocc`.
pub fn fock(ints: &Integrals<f64>, nocc: usize) -> Vec<f64> {
    fock_occ(ints, &(0..nocc).collect::<Vec<_>>())
}

/// The CIS matrix `A` (row-major `n × n`, `n = |occ|·nvirt`) of the singlet or triplet singles.
pub fn cis_matrix_occ(ints: &Integrals<f64>, occ: &[usize], triplet: bool) -> Vec<f64> {
    let d = ints.norb;
    let virt = virtuals(d, occ);
    let nv = virt.len();
    let n = occ.len() * nv;
    let f = fock_occ(ints, occ);
    let v = &ints.v.data;
    let eri = |p: usize, q: usize, r: usize, s: usize| v[((p * d + q) * d + r) * d + s];
    let kappa = if triplet { 0.0 } else { 2.0 };
    let mut a = vec![0.0; n * n];
    for (ik, &i) in occ.iter().enumerate() {
        for (ai, &aa) in virt.iter().enumerate() {
            for (jk, &j) in occ.iter().enumerate() {
                for (bi, &b) in virt.iter().enumerate() {
                    let mut x = kappa * eri(i, aa, j, b) - eri(i, j, aa, b);
                    if i == j {
                        x += f[aa * d + b];
                    }
                    if ai == bi {
                        x -= f[i * d + j];
                    }
                    a[(ik * nv + ai) * n + jk * nv + bi] = x;
                }
            }
        }
    }
    a
}

/// [`cis_matrix_occ`] at the determinant occupying `0..nocc`.
pub fn cis_matrix(ints: &Integrals<f64>, nocc: usize, triplet: bool) -> Vec<f64> {
    cis_matrix_occ(ints, &(0..nocc).collect::<Vec<_>>(), triplet)
}

/// Every singlet and triplet CIS root at the determinant occupying `occ`, merged and ascending in
/// energy.
pub fn cis_roots_occ(ints: &Integrals<f64>, occ: &[usize]) -> Vec<CisRoot> {
    let n = occ.len() * virtuals(ints.norb, occ).len();
    let mut roots = Vec::with_capacity(2 * n);
    for triplet in [false, true] {
        let (e, u) = dense_self_adjoint_smallest(&cis_matrix_occ(ints, occ, triplet), n, n);
        roots.extend(
            e.into_iter()
                .zip(u)
                .map(|(energy, x)| CisRoot { triplet, energy, x }),
        );
    }
    roots.sort_by(|p, q| p.energy.total_cmp(&q.energy));
    roots
}

/// [`cis_roots_occ`] at the determinant occupying `0..nocc`.
pub fn cis_roots(ints: &Integrals<f64>, nocc: usize) -> Vec<CisRoot> {
    cis_roots_occ(ints, &(0..nocc).collect::<Vec<_>>())
}
