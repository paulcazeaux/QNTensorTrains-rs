// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Entanglement functionals of a tensor-train state: bond (Schmidt-spectrum) functionals, and the
//! orbital entropies and mutual information built on the orbital reduced density matrices.

use crate::core::factor::dense_self_adjoint_smallest;
use crate::rdm::OrbitalRdms;
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// A scalar figure of merit of one bond's pooled Schmidt spectrum; smaller is less entangled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BondFunctional {
    /// Rényi entropy `S_α = ln(Σ p_i^α)/(1 − α)`, `α > 0`, `α ≠ 1`.
    Renyi(f64),
    /// von Neumann entropy `−Σ p_i ln p_i` (`0·ln 0 := 0`).
    VonNeumann,
    /// Weight outside the `max_rank` largest values, `Σ_{i > max_rank} p_i` — exactly what a
    /// pooled rank cap discards.
    DiscardedWeight { max_rank: usize },
}

impl Default for BondFunctional {
    fn default() -> Self {
        BondFunctional::Renyi(0.5)
    }
}

/// Evaluate `f` on a pooled list of singular values (any order, any normalization). The values are
/// sorted and reduced SERIALLY from the smallest up, so the result is independent of how the
/// caller gathered them. An all-zero spectrum returns `0.0`.
pub fn bond_functional(svals: &[f64], f: BondFunctional) -> f64 {
    let mut s2: Vec<f64> = svals.iter().map(|&s| s * s).filter(|&w| w > 0.0).collect();
    if s2.is_empty() {
        return 0.0;
    }
    s2.sort_by(|a, b| a.total_cmp(b));
    let total: f64 = s2.iter().sum();
    match f {
        BondFunctional::Renyi(alpha) => {
            assert!(
                alpha > 0.0 && (alpha - 1.0).abs() > 1e-12,
                "BondFunctional::Renyi: α must be positive and ≠ 1 (got {alpha}); use VonNeumann"
            );
            let sum: f64 = s2.iter().map(|&w| (w / total).powf(alpha)).sum();
            sum.ln() / (1.0 - alpha)
        }
        BondFunctional::VonNeumann => -s2
            .iter()
            .map(|&w| {
                let p = w / total;
                p * p.ln()
            })
            .sum::<f64>(),
        BondFunctional::DiscardedWeight { max_rank } => {
            let keep = max_rank.min(s2.len());
            let dropped: f64 = s2[..s2.len() - keep].iter().sum();
            dropped / total
        }
    }
}

/// `f` on every interior bond of `x` (bonds `1..d`), via [`TtVector::svdvals`] — which leaves the
/// train right-canonical. Returns the per-bond values; their sum is the run-level diagnostic.
pub fn bond_functional_profile<T: Scalar>(x: &mut TtVector<T>, f: BondFunctional) -> Vec<f64> {
    x.svdvals()
        .iter()
        .map(|sk| {
            let mut pooled: Vec<f64> = Vec::new();
            for s in sk.sectors() {
                if let Some(b) = sk.block(s) {
                    for i in 0..b.nrows().min(b.ncols()) {
                        pooled.push(b.get(i, i).re_s());
                    }
                }
            }
            bond_functional(&pooled, f)
        })
        .collect()
}

/// Orbital entropies and the two-orbital mutual information of one state, from
/// [`orbital_entanglement`]. All `d×d` matrices are row-major.
#[derive(Clone, Debug)]
pub struct OrbitalEntanglement {
    /// The number of orbitals.
    pub d: usize,
    /// `s1[i] = s_i = −Tr ρ_i ln ρ_i`.
    pub s1: Vec<f64>,
    /// `s2[i·d + j] = s_ij` for `i ≠ j` (symmetric), and `s2[i·d + i] = s_i`.
    pub s2: Vec<f64>,
    /// `mutual[i·d + j] = I_ij = s_i + s_j − s_ij` for `i ≠ j` — natural log, NO ½ — and
    /// `mutual[i·d + i] = 0`. Evaluated once per pair and written to both halves, so it is exactly
    /// symmetric. Not clamped: roundoff can leave a tiny negative value on a pair with `I_ij ≈ 0`.
    pub mutual: Vec<f64>,
    /// The most negative eigenvalue over every `ρ_i` and `ρ_ij`, or `0.0` if none is negative.
    pub min_eigenvalue: f64,
}

impl OrbitalEntanglement {
    /// `I_tot = Σ_i s_i` (`i` ascending).
    pub fn i_tot(&self) -> f64 {
        let mut acc = 0.0f64;
        for &s in &self.s1 {
            acc += s;
        }
        acc
    }

    /// `I_dist(p) = Σ_{i≠j} I_ij |i − j|^p` over ORDERED pairs, so each unordered pair counts twice
    /// (the definition of Krumnow et al., arXiv:1906.00205, at `p = 2`). Summed `i` then `j`
    /// ascending.
    pub fn i_dist(&self, power: u32) -> f64 {
        let p = i32::try_from(power).expect("OrbitalEntanglement::i_dist: power exceeds i32::MAX");
        let d = self.d;
        let mut acc = 0.0f64;
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    acc += self.mutual[i * d + j] * (i.abs_diff(j) as f64).powi(p);
                }
            }
        }
        acc
    }
}

fn von_neumann_eigen<T: Scalar>(rho: &[T], m: usize, min_eig: &mut f64) -> f64 {
    debug_assert_eq!(rho.len(), m * m, "von_neumann_eigen: rho must be m×m");
    let (evals, _) = dense_self_adjoint_smallest::<T>(rho, m, 0);
    let mut s = 0.0f64;
    for &lam in &evals {
        if lam < *min_eig {
            *min_eig = lam;
        }
        if lam > 0.0 {
            s -= lam * lam.ln();
        }
    }
    s
}

/// Orbital entropies `s_i`, pair entropies `s_ij` and mutual information `I_ij = s_i + s_j − s_ij`
/// (natural log, no ½) of the state whose orbital RDMs are `rdms`.
pub fn orbital_entanglement<T: Scalar>(rdms: &OrbitalRdms<T>) -> OrbitalEntanglement {
    let d = rdms.d();
    let mut min_eig = 0.0f64;
    let mut s1 = vec![0.0f64; d];
    for (i, si) in s1.iter_mut().enumerate() {
        *si = von_neumann_eigen(&rdms.rho1(i), 4, &mut min_eig);
    }
    let mut s2 = vec![0.0f64; d * d];
    let mut mutual = vec![0.0f64; d * d];
    for i in 0..d {
        s2[i * d + i] = s1[i];
        for j in (i + 1)..d {
            let sij = von_neumann_eigen(&rdms.rho2(i, j), 16, &mut min_eig);
            s2[i * d + j] = sij;
            s2[j * d + i] = sij;
            let m = s1[i] + s1[j] - sij;
            mutual[i * d + j] = m;
            mutual[j * d + i] = m;
        }
    }
    OrbitalEntanglement {
        d,
        s1,
        s2,
        mutual,
        min_eigenvalue: min_eig,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_functional_reference_values() {
        let s = [0.5, 2.0, 0.25, 1.0];
        let tot: f64 = s.iter().map(|x| x * x).sum();
        let p: Vec<f64> = s.iter().map(|x| x * x / tot).collect();
        let vn: f64 = -p.iter().map(|q| q * q.ln()).sum::<f64>();
        assert!((bond_functional(&s, BondFunctional::VonNeumann) - vn).abs() < 1e-14);
        let l1: f64 = s.iter().sum();
        let half = 2.0 * (l1 / tot.sqrt()).ln();
        assert!((bond_functional(&s, BondFunctional::Renyi(0.5)) - half).abs() < 1e-14);
        let r2 = -(p.iter().map(|q| q * q).sum::<f64>()).ln();
        assert!((bond_functional(&s, BondFunctional::Renyi(2.0)) - r2).abs() < 1e-14);
        let dw = (0.25f64 * 0.25 + 0.5 * 0.5) / tot;
        let got = bond_functional(&s, BondFunctional::DiscardedWeight { max_rank: 2 });
        assert!((got - dw).abs() < 1e-15, "{got} vs {dw}");
        assert_eq!(
            bond_functional(&s, BondFunctional::DiscardedWeight { max_rank: 9 }),
            0.0
        );
        for f in [BondFunctional::Renyi(0.5), BondFunctional::VonNeumann] {
            assert_eq!(bond_functional(&[3.0, 0.0], f), 0.0);
        }
        assert_eq!(bond_functional(&[], BondFunctional::VonNeumann), 0.0);
    }

    #[test]
    fn von_neumann_eigen_skips_nonpositive_and_reports_min() {
        let mut rho = [0.0f64; 16];
        rho[0] = 0.5;
        rho[5] = 0.5;
        rho[15] = -1e-13;
        let mut min_eig = 0.0f64;
        let s = von_neumann_eigen(&rho, 4, &mut min_eig);
        assert!((s - std::f64::consts::LN_2).abs() <= 1e-15, "s = {s}");
        assert!((min_eig - (-1e-13)).abs() <= 1e-28, "min_eig = {min_eig:e}");
        let mut psd = [0.0f64; 16];
        psd[0] = 1.0;
        let s = von_neumann_eigen(&psd, 4, &mut min_eig);
        assert!(s.abs() <= 1e-15 && min_eig < 0.0);
        let mut fresh = 0.0f64;
        let s = von_neumann_eigen(&psd, 4, &mut fresh);
        assert!(s.abs() <= 1e-15, "pure state s = {s:e}");
        assert_eq!(fresh, 0.0, "PSD input must leave min_eig at exactly 0.0");
        use crate::C64;
        let mut z = [C64::new(0.0, 0.0); 16];
        z[0] = C64::new(0.5, 0.0);
        z[1] = C64::new(0.0, 0.3);
        z[4] = C64::new(0.0, -0.3);
        z[5] = C64::new(0.5, 0.0);
        let mut m = 0.0f64;
        let s = von_neumann_eigen(&z, 4, &mut m);
        let want = -(0.8f64 * 0.8f64.ln() + 0.2 * 0.2f64.ln());
        assert!((s - want).abs() <= 1e-14, "s = {s}, want {want}");
    }
}
