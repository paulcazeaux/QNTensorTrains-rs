// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::tangent::TtTangent;
use crate::core::weingarten::{
    SignedWilliamson, WilliamsonError, rpa_generator_eigenvalues, symplectic_generator_full,
    williamson_full, williamson_reduced_signed,
};
use crate::hamiltonian::SparseHamiltonian;
use crate::scalar::{C64, Scalar};
use crate::solvers::davidson::{DavidsonOpts, davidson};
use crate::vector::TtVector;

use super::{WeingartenCtx, effham_apply, effham_operator};

fn refuse_shifted_basis<T: Scalar>(basis: &[TtTangent<T>], what: &str) {
    for (i, u) in basis.iter().enumerate() {
        u.refuse_if_shifted(&format!("{what} (basis vector {i})"));
    }
}

macro_rules! impl_hessian_grams {
    (
        ctx: $ctx:ident,
        tangent: $tangent:ident,
        tag: $tag:literal,
        $(#[$doc_hessian_grams:meta])*
        hessian_grams,
        $(#[$doc_hessian_grams_owned:meta])*
        hessian_grams_owned,
        $(#[$doc_symplectic_eigenvalues:meta])*
        symplectic_eigenvalues,
        $(#[$doc_hessian_grams_c64:meta])*
        hessian_grams_c64,
        $(#[$doc_symplectic_eigenvalues_c64:meta])*
        symplectic_eigenvalues_c64 $(,)?
    ) => {
        $(#[$doc_hessian_grams])*
        pub fn hessian_grams(
            ctx: &$ctx<'_, f64>,
            basis: &[$tangent<f64>],
        ) -> (Vec<f64>, Vec<f64>) {
            refuse_shifted_basis(basis, concat!($tag, "hessian_grams"));
            let (bl, br) = ctx.shared_bases();
            let pure: Vec<$tangent<f64>> = basis
                .iter()
                .map(|u| $tangent::project_shared(bl, br, &u.to_vector()))
                .collect();
            grams_over_purified(ctx, &pure)
        }

        $(#[$doc_hessian_grams_owned])*
        pub fn hessian_grams_owned(
            ctx: &$ctx<'_, f64>,
            mut basis: Vec<$tangent<f64>>,
        ) -> (Vec<f64>, Vec<f64>) {
            refuse_shifted_basis(&basis, concat!($tag, "hessian_grams_owned"));
            let (bl, br) = ctx.shared_bases();
            for u in basis.iter_mut() {
                *u = $tangent::project_shared(bl, br, &u.to_vector());
            }
            grams_over_purified(ctx, &basis)
        }

        fn grams_over_purified(
            ctx: &$ctx<'_, f64>,
            basis: &[$tangent<f64>],
        ) -> (Vec<f64>, Vec<f64>) {
            let n = basis.len();
            let mut a = vec![0.0f64; n * n];
            let mut b = vec![0.0f64; n * n];
            for (j, uj) in basis.iter().enumerate() {
                let aj = ctx.a_apply(uj);
                let bj = ctx.b_apply(uj);
                for (i, ui) in basis.iter().enumerate() {
                    a[i * n + j] = ui.dot(&aj);
                    b[i * n + j] = ui.dot(&bj);
                }
            }
            (a, b)
        }

        $(#[$doc_symplectic_eigenvalues])*
        pub fn symplectic_eigenvalues(
            ctx: &$ctx<'_, f64>,
            basis: &[$tangent<f64>],
        ) -> SymplecticSpectrum {
            refuse_shifted_basis(basis, concat!($tag, "symplectic_eigenvalues"));
            let (a, b) = hessian_grams(ctx, basis);
            symplectic_from_grams(&a, &b, basis.len(), 1e-10)
        }

        $(#[$doc_hessian_grams_c64])*
        pub fn hessian_grams_c64(
            ctx: &$ctx<'_, C64>,
            basis: &[$tangent<C64>],
        ) -> (Vec<C64>, Vec<C64>) {
            refuse_shifted_basis(basis, concat!($tag, "hessian_grams_c64"));
            let (bl, br) = ctx.shared_bases();
            let basis: Vec<$tangent<C64>> = basis
                .iter()
                .map(|u| $tangent::project_shared(bl, br, &u.to_vector()))
                .collect();
            let n = basis.len();
            let zero = C64::new(0.0, 0.0);
            let mut a = vec![zero; n * n];
            let mut b = vec![zero; n * n];
            for (j, uj) in basis.iter().enumerate() {
                let aj = ctx.a_apply(uj);
                let bj = ctx.b_apply(uj);
                for (i, ui) in basis.iter().enumerate() {
                    a[i * n + j] = ui.dot(&aj);
                    b[i * n + j] = ui.dot(&bj);
                }
            }
            (a, b)
        }

        $(#[$doc_symplectic_eigenvalues_c64])*
        pub fn symplectic_eigenvalues_c64(
            ctx: &$ctx<'_, C64>,
            basis: &[$tangent<C64>],
        ) -> SymplecticSpectrum {
            refuse_shifted_basis(basis, concat!($tag, "symplectic_eigenvalues_c64"));
            let (a, b) = hessian_grams_c64(ctx, basis);
            symplectic_from_grams_c64(&a, &b, basis.len(), 1e-10)
        }
    };
}

pub(crate) use impl_hessian_grams;

impl_hessian_grams!(
    ctx: WeingartenCtx,
    tangent: TtTangent,
    tag: "",
    hessian_grams,
    hessian_grams_owned,
    symplectic_eigenvalues,
    hessian_grams_c64,
    symplectic_eigenvalues_c64,
);

/// Outcome of the symplectic-eigenvalue driver.
#[derive(Clone, Debug)]
pub enum SymplecticSpectrum {
    /// Reduced Williamson path `[GHC Eq. 20]` succeeded: `ω` ascending — the linear-response
    /// excitation estimates at fixed rank.
    Stable { omega: Vec<f64> },
    /// **Exactly one** of `H^rr`, `H^ii` carries negative curvature — the other is still PD,
    /// so the signed reduced form [`crate::core::weingarten::williamson_reduced_signed`] applies
    /// and the whole spectrum is real in `ω²`. Negative `ω²` are **resonances** (`ω = i√|ω²|`,
    /// an imaginary `±ω` pair of `𝓜`), not a solver failure.
    Resonant { signed: SignedWilliamson },
    /// **Both** blocks indefinite: the symplectic eigenvalues are genuinely complex quartets and
    /// no symmetric reduction of any kind exists. Diagnosis via the non-Hermitian generator
    /// `𝓜 = [[A,B],[−B,−A]]`; this path diagnoses instead of crashing.
    Unstable {
        reason: WilliamsonError,
        generator_eigs: Vec<C64>,
    },
}

impl SymplecticSpectrum {
    /// `ω²` ascending and signed on the `Stable` and `Resonant` arms, `None` on `Unstable` (where
    /// `ω²` is not real). The one spelling every consumer of a possibly-resonant spectrum should
    /// use — `Stable` squares its `ω`, `Resonant` returns its signed values verbatim.
    pub fn omega_squared(&self) -> Option<Vec<f64>> {
        match self {
            SymplecticSpectrum::Stable { omega } => Some(omega.iter().map(|w| w * w).collect()),
            SymplecticSpectrum::Resonant { signed } => Some(signed.omega2.clone()),
            SymplecticSpectrum::Unstable { .. } => None,
        }
    }

    /// Number of unstable (negative-`ω²`) pairs: `0` on `Stable`, `signed.n_neg` on `Resonant`,
    /// `None` on `Unstable` (quartets are not counted this way).
    pub fn unstable_pairs(&self) -> Option<usize> {
        match self {
            SymplecticSpectrum::Stable { .. } => Some(0),
            SymplecticSpectrum::Resonant { signed } => Some(signed.n_neg),
            SymplecticSpectrum::Unstable { .. } => None,
        }
    }
}

/// Symplectic eigenvalues from precomputed Gram matrices (see [`hessian_grams`]); `psd_tol` is the
/// relative PD guard of `williamson_reduced`.
pub fn symplectic_from_grams(a: &[f64], b: &[f64], n: usize, psd_tol: f64) -> SymplecticSpectrum {
    let hrr: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let hii: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
    match williamson_reduced_signed(&hrr, &hii, n, psd_tol) {
        Ok(signed) if signed.n_neg == 0 => {
            let mut omega: Vec<f64> = signed.omega2.iter().map(|&v| v.max(0.0).sqrt()).collect();
            omega.sort_by(|p, q| p.partial_cmp(q).expect("finite ω"));
            SymplecticSpectrum::Stable { omega }
        }
        Ok(signed) => SymplecticSpectrum::Resonant { signed },
        Err(reason) => SymplecticSpectrum::Unstable {
            reason,
            generator_eigs: rpa_generator_eigenvalues(a, b, n),
        },
    }
}

/// Assemble the real-symmetric `2n×2n` `𝔥^LR = [[H^rr, H^ri],[H^ir, H^ii]]` `[GHC Eqs. 14–17]`
/// from the complex Grams, over the canonical real basis `(u_1,…,u_n, −J u_1,…,−J u_n)`
/// (`J = i·`, H0 §1.2): with `A` ℂ-linear Hermitian and `B` antilinear complex-symmetric,
/// ```text
/// H^rr = Re A + Re B      H^ri = Im A − Im B
/// H^ir = −Im A − Im B     H^ii = Re A − Re B
/// ```
/// (`H^ir = (H^ri)ᵀ` by the Hermitian/symmetric Gram structure; the whole `𝔥^LR` is symmetric.)
/// Real layer embedded: `Im A = Im B = 0` recovers the block-diagonal `H^rr = A+B`, `H^ii = A−B`
/// of the Eq.-20 path. Sign assignment oracle-pinned (t8k_ FD along `ξ` and `iξ`).
pub fn hll_from_grams_c64(a: &[C64], b: &[C64], n: usize) -> Vec<f64> {
    let two_n = 2 * n;
    let mut h = vec![0.0f64; two_n * two_n];
    for j in 0..n {
        for k in 0..n {
            let (ajk, bjk) = (a[j * n + k], b[j * n + k]);
            h[j * two_n + k] = ajk.re + bjk.re;
            h[j * two_n + (n + k)] = ajk.im - bjk.im;
            h[(n + j) * two_n + k] = -ajk.im - bjk.im;
            h[(n + j) * two_n + (n + k)] = ajk.re - bjk.re;
        }
    }
    h
}

/// Symplectic eigenvalues from precomputed complex Grams (see [`hessian_grams_c64`]): full
/// `2n×2n` Williamson `[GHC Eq. 18]`, with the `eig(J𝔥)` generator as the unstable-diagnosis
/// fallback. Returns `n` values on the stable path (one per complex tangent dimension).
pub fn symplectic_from_grams_c64(
    a: &[C64],
    b: &[C64],
    n: usize,
    psd_tol: f64,
) -> SymplecticSpectrum {
    let hll = hll_from_grams_c64(a, b, n);
    match williamson_full(&hll, 2 * n, psd_tol) {
        Ok(omega) => SymplecticSpectrum::Stable { omega },
        Err(reason) => SymplecticSpectrum::Unstable {
            reason,
            generator_eigs: symplectic_generator_full(&hll, 2 * n),
        },
    }
}

/// Lowest eigenvalue of the effective Hamiltonian `A` on the tangent space at `x` — the H1
/// single-mode / effective-Hamiltonian excitation estimate (term A only; the full Riemannian
/// Hessian adds the curvature term B in H2, and the symplectic/RPA spectrum comes in H3). `v0` is a
/// tangent-space warm start (e.g. `TtTangent::from_base(x, &random_dir)`).
pub fn effham_lowest<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    v0: &TtTangent<T>,
    opts: &DavidsonOpts,
) -> (f64, TtTangent<T>) {
    let (base_l, base_r, lambda) = effham_operator(h, x);
    let mut op = |xi: &TtTangent<T>| effham_apply(h, &base_l, &base_r, lambda, xi);
    let mut prec = |_v: &mut TtTangent<T>, _theta: f64| {};
    let (val, vec, _info) = davidson(&mut op, v0, &mut prec, opts);
    (val, vec)
}

#[cfg(test)]
mod t8r_tests {
    use crate::core::weingarten::{RpaMetricBlock, WilliamsonError, williamson_reduced};
    use crate::solvers::hess::{SymplecticSpectrum, symplectic_from_grams};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    const PSD_TOL: f64 = 1e-10;

    fn orthogonal_rows(n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut v: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            for _ in 0..2 {
                for r in rows.iter() {
                    let d: f64 = r.iter().zip(v.iter()).map(|(p, q)| p * q).sum();
                    for (vi, ri) in v.iter_mut().zip(r.iter()) {
                        *vi -= d * ri;
                    }
                }
            }
            let nrm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(nrm > 1e-6, "degenerate draw at seed {seed}");
            for vi in v.iter_mut() {
                *vi /= nrm;
            }
            rows.push(v);
        }
        rows
    }

    fn sym_from_spectrum(evals: &[f64], q: &[Vec<f64>], n: usize) -> Vec<f64> {
        let mut m = vec![0.0f64; n * n];
        for (c, qc) in q.iter().enumerate() {
            for i in 0..n {
                let li = evals[c] * qc[i];
                for j in 0..n {
                    m[i * n + j] += li * qc[j];
                }
            }
        }
        m
    }

    fn grams_from_blocks(hrr: &[f64], hii: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let a = hrr.iter().zip(hii).map(|(p, q)| 0.5 * (p + q)).collect();
        let b = hrr.iter().zip(hii).map(|(p, q)| 0.5 * (p - q)).collect();
        (a, b)
    }

    #[test]
    fn t8r_routing_and_inertia() {
        let n = 7;
        let q = orthogonal_rows(n, 0xA9_1234);

        let cases: Vec<(&str, Vec<f64>, Vec<f64>)> = vec![
            (
                "both PD ⇒ Stable",
                vec![0.4, 0.9, 1.3, 2.0, 2.6, 3.1, 4.4],
                vec![0.3, 0.7, 1.1, 1.8, 2.2, 2.9, 3.7],
            ),
            (
                "H^rr PD, H^ii one negative ⇒ Resonant{Rr}, n_neg = 1",
                vec![0.4, 0.9, 1.3, 2.0, 2.6, 3.1, 4.4],
                vec![-0.35, 0.7, 1.1, 1.8, 2.2, 2.9, 3.7],
            ),
            (
                "H^rr PD, H^ii two negative ⇒ Resonant{Rr}, n_neg = 2",
                vec![0.4, 0.9, 1.3, 2.0, 2.6, 3.1, 4.4],
                vec![-0.35, -0.12, 1.1, 1.8, 2.2, 2.9, 3.7],
            ),
            (
                "H^ii PD, H^rr one negative ⇒ Resonant{Ii}, n_neg = 1",
                vec![-0.5, 0.9, 1.3, 2.0, 2.6, 3.1, 4.4],
                vec![0.3, 0.7, 1.1, 1.8, 2.2, 2.9, 3.7],
            ),
            (
                "both indefinite ⇒ Unstable (complex quartets)",
                vec![-0.5, 0.9, 1.3, 2.0, 2.6, 3.1, 4.4],
                vec![-0.35, 0.7, 1.1, 1.8, 2.2, 2.9, 3.7],
            ),
        ];

        for (tag, rr, ii) in cases {
            let hrr = sym_from_spectrum(&rr, &q, n);
            let hii = sym_from_spectrum(&ii, &q, n);
            let (a, b) = grams_from_blocks(&hrr, &hii);
            let rr_neg = rr.iter().filter(|&&v| v < 0.0).count();
            let ii_neg = ii.iter().filter(|&&v| v < 0.0).count();
            let spec = symplectic_from_grams(&a, &b, n, PSD_TOL);
            match (&spec, rr_neg, ii_neg) {
                (SymplecticSpectrum::Stable { omega }, 0, 0) => {
                    assert_eq!(omega.len(), n, "{tag}: one ω per dimension");
                    assert!(omega.iter().all(|&w| w > 0.0), "{tag}: ω must be positive");
                    assert_eq!(spec.unstable_pairs(), Some(0), "{tag}");
                }
                (SymplecticSpectrum::Resonant { signed }, 0, k) if k > 0 => {
                    assert_eq!(
                        signed.metric,
                        RpaMetricBlock::Rr,
                        "{tag}: metric must be H^rr"
                    );
                    assert_eq!(signed.n_neg, k, "{tag}: n_neg = ind(H^ii) by Sylvester");
                    assert_eq!(signed.omega2.len(), n, "{tag}");
                }
                (SymplecticSpectrum::Resonant { signed }, k, 0) if k > 0 => {
                    assert_eq!(
                        signed.metric,
                        RpaMetricBlock::Ii,
                        "{tag}: metric must be H^ii"
                    );
                    assert_eq!(signed.n_neg, k, "{tag}: n_neg = ind(H^rr) by Sylvester");
                }
                (SymplecticSpectrum::Unstable { generator_eigs, .. }, p, r) if p > 0 && r > 0 => {
                    assert_eq!(generator_eigs.len(), 2 * n, "{tag}: 𝓜 is 2n×2n");
                    assert_eq!(
                        spec.unstable_pairs(),
                        None,
                        "{tag}: quartets are not counted"
                    );
                }
                _ => panic!("{tag}: wrong routing — got {spec:?} for inertia ({rr_neg}, {ii_neg})"),
            }
            if let Some(w2) = spec.omega_squared() {
                for w in w2.windows(2) {
                    assert!(w[0] <= w[1] + 1e-13, "{tag}: ω² must come back ascending");
                }
            }
        }
    }

    #[test]
    fn t8r_stable_arm_matches_williamson_reduced_exactly() {
        let n = 8;
        let q = orthogonal_rows(n, 0xD1_5EA5);
        let rr = vec![0.4, 0.9, 1.3, 2.0, 2.6, 3.1, 3.8, 4.4];
        let ii = vec![0.3, 0.7, 1.1, 1.8, 2.2, 2.9, 3.3, 3.7];
        let hrr = sym_from_spectrum(&rr, &q, n);
        let hii = sym_from_spectrum(&ii, &q, n);
        let (a, b) = grams_from_blocks(&hrr, &hii);

        let hrr_rt: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        let hii_rt: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p - q).collect();

        let want = williamson_reduced(&hrr_rt, &hii_rt, n, PSD_TOL).expect("PD/PD must be Stable");
        let SymplecticSpectrum::Stable { omega } = symplectic_from_grams(&a, &b, n, PSD_TOL) else {
            panic!("PD/PD must route to Stable");
        };
        assert_eq!(
            omega, want,
            "the Stable arm must be BIT-IDENTICAL to williamson_reduced on the same blocks, not \
             merely close — the H9 routing may not perturb a stable point at all"
        );
    }

    fn analytic_pair(u: f64, v: f64) -> (Vec<f64>, Vec<f64>) {
        let hrr = [u, 1.0, 1.0, u];
        let hii = [v, 1.0, 1.0, v];
        let a: Vec<f64> = hrr.iter().zip(&hii).map(|(p, q)| 0.5 * (p + q)).collect();
        let b: Vec<f64> = hrr.iter().zip(&hii).map(|(p, q)| 0.5 * (p - q)).collect();
        for (k, (&p, &q)) in hrr.iter().zip(&hii).enumerate() {
            assert_eq!(a[k] + b[k], p, "A+B must reconstruct H^rr bit-exactly");
            assert_eq!(a[k] - b[k], q, "A−B must reconstruct H^ii bit-exactly");
        }
        (a, b)
    }

    fn analytic_omega2(u: f64, v: f64) -> [f64; 2] {
        let mut w = [(u - 1.0) * (v - 1.0), (u + 1.0) * (v + 1.0)];
        w.sort_by(|p, q| p.partial_cmp(q).unwrap());
        w
    }

    #[test]
    fn t8r_analytic_family_pins_all_three_verdicts_with_closed_form_omega2() {
        const N: usize = 2;

        let (a, b) = analytic_pair(2.0, 3.0);
        let want = analytic_omega2(2.0, 3.0);
        assert_eq!(want, [2.0, 12.0], "the closed form must be {{2, 12}}");
        let spec = symplectic_from_grams(&a, &b, N, PSD_TOL);
        let SymplecticSpectrum::Stable { omega } = &spec else {
            panic!("both blocks PD must route to Stable, got {spec:?}");
        };
        for (w, &w2) in omega.iter().zip(&want) {
            assert!((w - w2.sqrt()).abs() <= 1e-14 * w2.sqrt(), "ω {w} vs √{w2}");
        }
        assert_eq!(spec.unstable_pairs(), Some(0));
        for (got, &w2) in spec
            .omega_squared()
            .expect("Stable is readable")
            .iter()
            .zip(&want)
        {
            assert!((got - w2).abs() <= 1e-13 * w2, "ω² {got} vs {w2}");
        }

        let (a, b) = analytic_pair(2.0, 0.0);
        assert_eq!(a, vec![1.0, 1.0, 1.0, 1.0], "A = ½(H^rr + H^ii)");
        assert_eq!(b, vec![1.0, 0.0, 0.0, 1.0], "B = ½(H^rr − H^ii) = I");
        let want = analytic_omega2(2.0, 0.0);
        assert_eq!(want, [-1.0, 3.0], "the closed form must be {{−1, 3}}");
        let spec = symplectic_from_grams(&a, &b, N, PSD_TOL);
        let SymplecticSpectrum::Resonant { signed } = &spec else {
            panic!("H^rr ≻ 0 with H^ii indefinite must route to Resonant, got {spec:?}");
        };
        assert_eq!(
            signed.metric,
            RpaMetricBlock::Rr,
            "H^rr is tried first and is the PD block here"
        );
        assert_eq!(
            signed.n_neg, 1,
            "n_neg = ind(H^ii) = 1 by Sylvester — exact, not a tolerance artifact"
        );
        assert!(
            (signed.metric_min_eig - 1.0).abs() <= 1e-14,
            "λ_min(H^rr) = u − 1 = 1, got {}",
            signed.metric_min_eig
        );
        assert_eq!(spec.unstable_pairs(), Some(1));
        let got = spec.omega_squared().expect("Resonant is readable");
        assert_eq!(got.len(), N, "one signed ω² per dimension");
        for (g, &w2) in got.iter().zip(&want) {
            assert!((g - w2).abs() <= 1e-13, "signed ω² {g} vs {w2}");
        }
        assert!(
            got[0] < 0.0,
            "the resonance is PUBLISHED as a negative ω², never as ω"
        );
        let hii = [0.0, 1.0, 1.0, 0.0];
        let col: Vec<f64> = (0..N).map(|i| signed.modes_i[i * N]).collect();
        let quot: f64 = (0..N)
            .map(|i| col[i] * (0..N).map(|j| hii[i * N + j] * col[j]).sum::<f64>())
            .sum();
        assert!(
            (quot + 1.0).abs() <= 1e-13,
            "the unstable mode's H^ii Rayleigh quotient must be v − 1 = −1, got {quot}"
        );

        let (a, b) = analytic_pair(0.0, 0.5);
        let want = analytic_omega2(0.0, 0.5);
        assert_eq!(want, [0.5, 1.5], "the closed form must be {{½, 3/2}}");
        let spec = symplectic_from_grams(&a, &b, N, PSD_TOL);
        let SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } = &spec
        else {
            panic!("both blocks indefinite must route to Unstable, got {spec:?}");
        };
        let WilliamsonError::HrrNotPd { min_eig } = reason else {
            panic!("the both-indefinite refusal is reported as HrrNotPd, got {reason:?}");
        };
        assert!(
            (min_eig + 1.0).abs() <= 1e-14,
            "the reported λ_min is H^rr's own, u − 1 = −1, got {min_eig}"
        );
        assert_eq!(spec.omega_squared(), None, "ω² is not real on this arm");
        assert_eq!(spec.unstable_pairs(), None, "quartets are not counted");
        assert_eq!(generator_eigs.len(), 2 * N, "𝓜 is 2n×2n");
        let mut sq: Vec<f64> = generator_eigs
            .iter()
            .map(|e| {
                assert!(e.im.abs() <= 1e-13, "λ must come out REAL here, got {e:?}");
                e.re * e.re
            })
            .collect();
        sq.sort_by(|p, q| p.partial_cmp(q).unwrap());
        for (got, w2) in sq.iter().zip([0.5, 0.5, 1.5, 1.5]) {
            assert!((got - w2).abs() <= 1e-13, "λ² {got} vs {w2}");
        }
    }
}
