// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The exponential-sum preconditioner `S ≈ D^{-1/2}` (paper §4) and the one-particle diagonal
//! `D = Σ_i d_i a_i* a_i` it approximates.

use crate::core::Family;
use crate::hamiltonian::SparseHamiltonian;
use crate::integrals::Integrals;
use crate::ops::dot;
use crate::qn::Pooled;
use crate::round::round_sum;
use crate::solvers::lanczos::{LanczosOptions, lanczos};
use crate::vector::TtVector;

use super::TruncBackend;

/// The exponential-sum preconditioner. Holds only the sinc-quadrature scalars `(α_m, β_m)` and the
/// one-particle diagonal `d`; the operator is never materialized as an MPO.
#[derive(Clone, Debug)]
pub struct ExpSumPrecond {
    /// Quadrature weights `α_m`, length `M = M⁺−M⁻+1` (the MPO rank of `S`).
    pub alphas: Vec<f64>,
    /// Quadrature exponents `β_m`, same length.
    pub betas: Vec<f64>,
    /// One-particle diagonal `d_i` (per spatial orbital), length `norb`. Strictly positive.
    pub d: Vec<f64>,
}

impl ExpSumPrecond {
    /// Number of exponential-sum terms `M = M⁺−M⁻+1` (equivalently the MPO rank of `S`).
    pub fn num_terms(&self) -> usize {
        self.alphas.len()
    }

    /// The scalar function `S_scalar(λ) = Σ_m α_m e^{-β_m λ}` — the many-body eigenvalue of `S` on
    /// a configuration whose `D`-eigenvalue is `λ`. Approximates `λ^{-1/2}`.
    pub fn scalar(&self, lambda: f64) -> f64 {
        self.alphas
            .iter()
            .zip(&self.betas)
            .map(|(&a, &b)| a * (-b * lambda).exp())
            .sum()
    }

    /// Build `S ≈ D^{-1/2}` from the one-particle diagonal `d` and target relative accuracy
    /// `c₀ ∈ (0,1)`, over the `(nup, ndn)` sector.
    pub fn from_diagonal(d: &[f64], nup: usize, ndn: usize, c0: f64) -> ExpSumPrecond {
        assert!(c0 > 0.0 && c0 < 1.0, "c0 must lie in (0,1), got {c0}");
        assert!(!d.is_empty(), "empty diagonal");
        assert!(
            d.iter().all(|di| di.is_finite()),
            "one-particle diagonal must be finite"
        );
        assert!(
            nup <= d.len() && ndn <= d.len() && nup + ndn >= 1,
            "sector (nup {nup}, ndn {ndn}) is not representable on {} orbitals",
            d.len()
        );
        let mut asc: Vec<f64> = d.to_vec();
        asc.sort_by(|a, b| a.partial_cmp(b).expect("finite diagonal"));
        let lo_sum = |m: usize| -> f64 { asc[..m].iter().sum() };
        let hi_sum = |m: usize| -> f64 { asc[asc.len() - m..].iter().sum() };
        let t_min: f64 = lo_sum(nup) + lo_sum(ndn);
        let t_max: f64 = hi_sum(nup) + hi_sum(ndn);
        assert!(
            t_min > 0.0,
            "D is not positive on the (nup {nup}, ndn {ndn}) sector: the smallest reachable eigenvalue \
             Σ_i d_i n_i is {t_min:.6e}. Individual d_i MAY be negative — only the sector minimum \
             must be positive — but this diagonal fails even that, so D^{{-1/2}} does not exist."
        );
        let big_t = t_max / t_min;

        let a = (std::f64::consts::PI.sqrt() * c0 / 4.0).ln();
        let abs_a = a.abs();
        let h = std::f64::consts::PI.powi(2) / (8.0 * std::f64::consts::SQRT_2 / c0 + 1.0).ln();
        let m_plus = ((abs_a.ln()) / h).ceil() as i64;
        let m_minus = -(((2.0 * abs_a + big_t.ln()) / h).ceil() as i64);
        assert!(
            m_minus <= m_plus,
            "empty exponential sum (M⁻={m_minus} > M⁺={m_plus}); check c0/spectrum"
        );

        let pref = h / std::f64::consts::PI.sqrt();
        let inv_sqrt_tmin = t_min.powf(-0.5);
        let mut alphas = Vec::with_capacity((m_plus - m_minus + 1) as usize);
        let mut betas = Vec::with_capacity((m_plus - m_minus + 1) as usize);
        for m in m_minus..=m_plus {
            let mh = m as f64 * h;
            betas.push(mh.exp() / t_min);
            alphas.push(inv_sqrt_tmin * pref * (mh / 2.0).exp());
        }
        let d_neg = d
            .iter()
            .cloned()
            .fold(0.0f64, |acc, di| acc.max((-di).max(0.0)));
        let beta_max = betas.iter().cloned().fold(f64::MIN, f64::max);
        let worst_exponent = 2.0 * beta_max * d_neg;
        assert!(
            worst_exponent < 600.0,
            "the exponential sum is not representable in f64: the doubly-occupied factor \
             e^{{-2 β d}} needs exponent {worst_exponent:.1} (overflow near 709). The driver is a \
             negative diagonal entry (min d = {:.3}) against a small sector minimum \
             (t_min = {t_min:.3}), since β_max ~ 1/t_min. RAISE γ: it lifts every d_i by γ/N and \
             t_min by γ, shrinking both factors. `gamma_for_positive_diagonal` returns the SMALLEST \
             admissible γ, which is exactly the worst case for this — pass it a larger margin.",
            -d_neg
        );

        ExpSumPrecond {
            alphas,
            betas,
            d: d.to_vec(),
        }
    }

    /// The `m`-th rank-1 summand `S_m x = (⊗_i diag(1, e^{-β_m d_i}))·x`, WITHOUT the `α_m` weight
    /// (the weights are applied by the truncation backend in `Self::apply`). A per-site rescale
    /// of the occupied families; the orthogonality flag is cleared.
    pub fn apply_summand(&self, m: usize, x: &TtVector<f64>) -> TtVector<f64> {
        let beta = self.betas[m];
        let mut y = x.clone();
        for (k, core) in y.cores.iter_mut().enumerate() {
            let bd = beta * self.d[k];
            let e1 = (-bd).exp();
            let e2 = (-2.0 * bd).exp();
            core.scale_family_mut(Family::Up, e1);
            core.scale_family_mut(Family::Dn, e1);
            core.scale_family_mut(Family::Occ, e2);
        }
        y.ortho = None;
        y
    }

    fn balanced_summands(&self, x: &TtVector<f64>) -> (Vec<f64>, Vec<TtVector<f64>>) {
        let mut summands = Vec::with_capacity(self.num_terms());
        let mut weights = Vec::with_capacity(self.num_terms());
        for m in 0..self.num_terms() {
            let mut y = self.apply_summand(m, x);
            let mut scale = 1.0f64;
            for core in y.cores.iter_mut() {
                let n = core.norm();
                if n > 0.0 && n.is_finite() {
                    core.scale_mut(n.recip());
                    scale *= n;
                }
            }
            summands.push(y);
            weights.push(self.alphas[m] * scale);
        }
        (weights, summands)
    }

    /// `S·x` to a RELATIVE tolerance: `‖out − Sx‖ ≤ tol_rel·‖Sx‖`.
    pub fn apply_rel(
        &self,
        x: &TtVector<f64>,
        tol_rel: f64,
        chi: Pooled,
        back: &TruncBackend,
    ) -> TtVector<f64> {
        let (weights, summands) = self.balanced_summands(x);
        back.round_weighted_sum_rel(&weights, &summands, tol_rel, chi, x)
    }

    /// `S·x` to an ABSOLUTE tolerance: `‖out − Sx‖ ≤ tol_abs`. The contract paper Algorithm 2 is
    /// written against, and what the deterministic solver path uses.
    pub fn apply_abs(
        &self,
        x: &TtVector<f64>,
        tol_abs: f64,
        chi: Pooled,
        back: &TruncBackend,
    ) -> TtVector<f64> {
        let (weights, summands) = self.balanced_summands(x);
        back.round_weighted_sum_abs(&weights, &summands, tol_abs, chi, x)
    }

    /// Recover the Hamiltonian eigenvector `y = S·x` (unit TT norm) from a preconditioned-coordinate
    /// iterate `x` returned by [`crate::solvers::pinvit::pinvit_ground_state`]. The PINVIT fixed
    /// point gives `H_γ (S x) = λ (S x)`, so `S·x` is the `H_γ` (and hence `H`) eigenvector.
    pub fn hamiltonian_vector(
        &self,
        x: &TtVector<f64>,
        tol: f64,
        chi: Pooled,
        back: &TruncBackend,
    ) -> TtVector<f64> {
        let mut y = self.apply_rel(x, tol, chi, back);
        let yn = y.norm();
        if yn > 0.0 {
            y.scale_mut(yn.recip());
        }
        y
    }
}

/// Options for [`ExpSumPrecond::calibrate`].
#[derive(Clone, Copy, Debug)]
pub struct CalibrateOpts {
    /// Power-iteration cap per end of the spectrum. Default `400`.
    pub iters: usize,
    /// Relative change in the Rayleigh quotient below which an end is declared settled.
    /// Default `1e-9`.
    pub tol: f64,
    /// Safety inflation on the derived `c`, as a fraction of the remaining headroom `1 − c`.
    /// Default `0.05`, TRIPLED automatically when either end of the spectrum failed to settle.
    pub margin: f64,
    /// Rounding tolerance for the two `S`-applies and the matvec inside each power step.
    /// Default `1e-10`.
    pub round_tol: f64,
    /// Truncation backend for those applies. Default [`TruncBackend::DeterministicSvd`].
    pub trunc: TruncBackend,
    /// Pooled total bond-dimension cap for those applies; [`Pooled::NONE`] = uncapped. Should
    /// match the solver's, or the measured spectrum belongs to a different operator than the one
    /// the solve will use.
    pub chi: Pooled,
}

impl Default for CalibrateOpts {
    fn default() -> Self {
        CalibrateOpts {
            iters: 400,
            tol: 1e-9,
            margin: 0.05,
            round_tol: 1e-10,
            trunc: TruncBackend::DeterministicSvd,
            chi: Pooled::NONE,
        }
    }
}

/// What [`ExpSumPrecond::calibrate`] measured and applied.
#[derive(Clone, Copy, Debug)]
pub struct Calibration {
    /// Estimated smallest eigenvalue of `A = S H_γ S`, BEFORE the rescale.
    pub lambda_min: f64,
    /// Estimated largest eigenvalue of `A = S H_γ S`, BEFORE the rescale.
    pub lambda_max: f64,
    /// The rescale applied to every `α_m`: `σ = √(2/(λ_min + λ_max))`.
    pub sigma: f64,
    /// The derived preconditioner-quality constant, margin included — feed this to
    /// [`super::PinvitConsts::c`].
    pub c: f64,
    /// `c` before the safety margin, i.e. `(λ_max − λ_min)/(λ_max + λ_min)` exactly. Diagnostic.
    pub c_raw: f64,
    /// Whether both ends settled to `tol` inside `iters`. A `false` here does not invalidate the
    /// result — the margin and the runtime witness still apply — but it means the bracket is loose.
    pub converged: bool,
}

impl ExpSumPrecond {
    /// Centre `spec(S H_γ S)` on `1` and derive `c` from the operator instead of accepting one.
    pub fn calibrate(
        &mut self,
        h: &SparseHamiltonian<f64>,
        gamma: f64,
        x0: &TtVector<f64>,
        opts: &CalibrateOpts,
    ) -> Calibration {
        let apply_a = |p: &Self, x: &TtVector<f64>| -> TtVector<f64> {
            let sx = p.apply_rel(x, opts.round_tol, opts.chi, &opts.trunc);
            let hsx = opts
                .trunc
                .apply_h_shift_rel(h, &sx, gamma, opts.round_tol, opts.chi);
            p.apply_rel(&hsx, opts.round_tol, opts.chi, &opts.trunc)
        };

        let (lambda_max, conv_hi) = {
            let mut x = x0.clone();
            let n = x.norm();
            if n > 0.0 {
                x.scale_mut(n.recip());
            }
            let (mut theta, mut converged) = (0.0, false);
            for _ in 0..opts.iters {
                let ax = apply_a(self, &x);
                let next = dot(&x, &ax) / dot(&x, &x);
                converged = (next - theta).abs() <= opts.tol * next.abs().max(1e-30);
                theta = next;
                let nn = ax.norm();
                if nn <= 0.0 || converged {
                    break;
                }
                x = ax;
                x.scale_mut(nn.recip());
            }
            (theta, converged)
        };

        let mu = 1.02 * lambda_max;
        let (lambda_min, conv_lo) = {
            let mut x = x0.clone();
            let n = x.norm();
            if n > 0.0 {
                x.scale_mut(n.recip());
            }
            let (mut theta, mut converged) = (lambda_max, false);
            for _ in 0..opts.iters {
                let ax = apply_a(self, &x);
                let bx = round_sum(&[mu, -1.0], &[x.clone(), ax], opts.round_tol);
                let next = mu - dot(&x, &bx) / dot(&x, &x);
                converged = (next - theta).abs() <= opts.tol * next.abs().max(1e-30);
                theta = next;
                let nn = bx.norm();
                if nn <= 0.0 || converged {
                    break;
                }
                x = bx;
                x.scale_mut(nn.recip());
            }
            (theta, converged)
        };

        assert!(
            lambda_min > 0.0 && lambda_max >= lambda_min,
            "calibrate: spec(S H_γ S) must be positive — got [{lambda_min:.3e}, {lambda_max:.3e}]. \
             `A` is SPD only when `H_γ` is positive definite on the sector; raise the elliptic \
             shift γ (see `choose_gamma`)."
        );

        let sigma = (2.0 / (lambda_min + lambda_max)).sqrt();
        let c_raw = (lambda_max - lambda_min) / (lambda_max + lambda_min);
        let converged = conv_hi && conv_lo;
        let margin = if converged {
            opts.margin
        } else {
            3.0 * opts.margin
        };
        let c = (c_raw + margin * (1.0 - c_raw)).min(1.0 - 1e-12);
        for a in self.alphas.iter_mut() {
            *a *= sigma;
        }
        Calibration {
            lambda_min,
            lambda_max,
            sigma,
            c,
            c_raw,
            converged,
        }
    }
}

/// The one-particle diagonal weights `d_i = θ_i` of paper §7.1.1, a Fock-like diagonal of
/// `H + γI` restricted to the sector:
pub fn diagonal_one_particle(ints: &Integrals<f64>, gamma: f64, n_particles: usize) -> Vec<f64> {
    let norb = ints.norb;
    let n = n_particles.max(1);
    let d: Vec<f64> = (0..norb)
        .map(|i| {
            let t_ii = ints.t.get(&[i, i]);
            let ki = (i + 1).saturating_sub(n);
            let mut acc = 0.0;
            for jj in 0..n {
                let j1 = ki + 1 + jj;
                if j1 < 1 || j1 > norb {
                    continue;
                }
                let j = j1 - 1;
                let jval = ints.v.get(&[i, i, j, j]);
                let kval = ints.v.get(&[i, j, j, i]);
                acc += jval - kval;
            }
            gamma / n as f64 + t_ii + 0.5 * acc
        })
        .collect();
    d
}

/// The smallest elliptic shift `γ` for which the auxiliary `D` of [`diagonal_one_particle`] is
/// positive on the `(nup, ndn)` sector, plus `margin`.
pub fn gamma_for_positive_diagonal(
    ints: &Integrals<f64>,
    nup: usize,
    ndn: usize,
    n_particles: usize,
    margin: f64,
) -> f64 {
    let nelec = nup + ndn;
    assert!(nelec >= 1, "empty sector");
    let d0 = diagonal_one_particle(ints, 0.0, n_particles);
    let mut asc = d0;
    asc.sort_by(|a, b| a.partial_cmp(b).expect("finite diagonal"));
    let lo_sum = |m: usize| -> f64 { asc[..m].iter().sum::<f64>() };
    let t_min0 = lo_sum(nup) + lo_sum(ndn);
    let slope = nelec as f64 / n_particles.max(1) as f64;
    ((margin - t_min0) / slope).max(0.0)
}

/// The one-body elliptic shift: `γ = margin − (Σ_{k<nup} ε_k + Σ_{k<ndn} ε_k)`, where `ε` are the
/// ascending eigenvalues of the one-body matrix `t`.
pub fn gamma_one_body(ints: &Integrals<f64>, nup: usize, ndn: usize, margin: f64) -> f64 {
    let k = ints.norb;
    assert!(nup <= k && ndn <= k, "sector exceeds the orbital count");
    let mut a: Vec<f64> = (0..k * k)
        .map(|idx| ints.t.get(&[idx / k, idx % k]))
        .collect();
    for _ in 0..100 {
        let off: f64 = (0..k)
            .flat_map(|i| ((i + 1)..k).map(move |j| (i, j)))
            .map(|(i, j)| a[i * k + j] * a[i * k + j])
            .sum();
        if off.sqrt() < 1e-13 {
            break;
        }
        for p in 0..k {
            for q in (p + 1)..k {
                if a[p * k + q].abs() < 1e-15 {
                    continue;
                }
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * a[p * k + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for m in 0..k {
                    let (amp, amq) = (a[m * k + p], a[m * k + q]);
                    a[m * k + p] = c * amp - sn * amq;
                    a[m * k + q] = sn * amp + c * amq;
                }
                for m in 0..k {
                    let (apm, aqm) = (a[p * k + m], a[q * k + m]);
                    a[p * k + m] = c * apm - sn * aqm;
                    a[q * k + m] = sn * apm + c * aqm;
                }
            }
        }
    }
    let mut eps: Vec<f64> = (0..k).map(|i| a[i * k + i]).collect();
    eps.sort_by(|x, y| x.partial_cmp(y).expect("finite one-body spectrum"));
    let e_lower: f64 = eps[..nup].iter().sum::<f64>() + eps[..ndn].iter().sum::<f64>();
    margin - e_lower
}

/// The analytic elliptic shift: `γ = 9θ² + ¼` with `θ(N, Z) = √N · max(N, Z)`, where `N` is the
/// electron count and `Z = Σ_ν Z_ν` the total nuclear charge.
pub fn gamma_elliptic(n_electrons: usize, z_total: f64) -> f64 {
    assert!(n_electrons >= 1, "need at least one electron");
    assert!(z_total > 0.0, "total nuclear charge must be positive");
    let n = n_electrons as f64;
    let theta = n.sqrt() * n.max(z_total);
    9.0 * theta * theta + 0.25
}

/// Pick an elliptic shift `γ` from a short Lanczos estimate of `λ_min`.
pub fn choose_gamma(h: &SparseHamiltonian<f64>, x0: &TtVector<f64>, margin: f64) -> f64 {
    let opts = LanczosOptions {
        tol: 1e-3,
        max_iter: 15,
    };
    let est = lanczos(h, x0, &opts).lambda;
    margin - est
}

#[cfg(test)]
mod precond_tests {
    use crate::hamiltonian::SparseHamiltonian;
    use crate::integrals::{Integrals, IntegralsOptions, load_fixture_dir};
    use crate::ops::dot;
    use crate::qn::{Pooled, QnSpace};
    use crate::round::round_sum;
    use crate::solvers::pinvit::{ExpSumPrecond, TruncBackend, diagonal_one_particle};
    use crate::vector::{TtVector, tt_state_in};
    use std::path::{Path, PathBuf};

    fn fixture_dir(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(name)
    }

    fn load(name: &str) -> (SparseHamiltonian<f64>, QnSpace, Integrals<f64>) {
        let opts = IntegralsOptions::default();
        let ld = load_fixture_dir::<f64>(&fixture_dir(name), &opts)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let h = ld.hamiltonian(opts.eps);
        (h, ld.space, ld.ints)
    }

    fn hf_determinant(space: QnSpace) -> TtVector<f64> {
        let d = space.d;
        let occ_up: Vec<bool> = (0..d).map(|i| i < space.nup).collect();
        let occ_dn: Vec<bool> = (0..d).map(|i| i < space.ndn).collect();
        tt_state_in(space, &occ_up, &occ_dn)
    }

    #[test]
    fn tpv_precond_scalar_bounds() {
        let d = vec![1.0, 1.7, 2.5, 4.0, 6.5, 9.0];
        let (nup, ndn) = (3usize, 1usize);
        for &c0 in &[0.2, 0.1, 0.05] {
            let p = ExpSumPrecond::from_diagonal(&d, nup, ndn, c0);
            assert!(p.num_terms() >= 1, "empty exponential sum");

            let mut asc = d.clone();
            asc.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = |m: usize| -> f64 { asc[..m].iter().sum() };
            let hi = |m: usize| -> f64 { asc[asc.len() - m..].iter().sum() };
            let t_min: f64 = lo(nup) + lo(ndn);
            let t_max: f64 = hi(nup) + hi(ndn);

            let steps = 50;
            for k in 0..=steps {
                let lam = t_min + (t_max - t_min) * (k as f64 / steps as f64);
                let approx = p.scalar(lam);
                let exact = lam.powf(-0.5);
                let rel = (approx - exact).abs() / exact;
                assert!(
                    rel <= c0 * 1.05,
                    "c0={c0}: |S(λ)−λ^-1/2|/λ^-1/2 = {rel:.3e} > {c0} at λ={lam:.3}"
                );
            }
        }
    }

    #[test]
    fn tpv_precond_symmetric_pd() {
        let (_, space, ints) = load("lih_sto3g");
        let d = diagonal_one_particle(&ints, 12.0, space.nup + space.ndn);
        let p = ExpSumPrecond::from_diagonal(&d, space.nup, space.ndn, 0.1);
        let back = TruncBackend::DeterministicSvd;

        let det = |u: [usize; 2], dn: [usize; 2]| -> TtVector<f64> {
            let mut ou = vec![false; space.d];
            let mut od = vec![false; space.d];
            for i in u {
                ou[i] = true;
            }
            for i in dn {
                od[i] = true;
            }
            tt_state_in(space, &ou, &od)
        };
        let da = det([0, 1], [0, 1]);
        let db = det([0, 2], [0, 1]);
        let dc = det([1, 2], [0, 2]);
        let x = round_sum(&[0.8, 0.6], &[da.clone(), db], 1e-14);
        let y = round_sum(&[0.5, 0.5], &[da, dc], 1e-14);

        let sx = p.apply_abs(&x, 1e-14, Pooled::NONE, &back);
        let sy = p.apply_abs(&y, 1e-14, Pooled::NONE, &back);
        let lhs = dot(&y, &sx);
        let rhs = dot(&sy, &x);
        assert!(
            (lhs - rhs).abs() < 1e-10,
            "S not symmetric: ⟨y,Sx⟩={lhs:.12} ⟨Sy,x⟩={rhs:.12}"
        );
        assert!(dot(&x, &sx) > 0.0, "⟨x,Sx⟩ must be positive");
        let s2x = p.apply_abs(&sx, 1e-14, Pooled::NONE, &back);
        assert!(dot(&x, &s2x) > 0.0, "⟨x,S²x⟩ must be positive");
    }

    #[test]
    fn tpv_precond_summand_preserves_determinant() {
        let (_, space, ints) = load("lih_sto3g");
        let d = diagonal_one_particle(&ints, 12.0, space.nup + space.ndn);
        let p = ExpSumPrecond::from_diagonal(&d, space.nup, space.ndn, 0.1);
        let det = hf_determinant(space);
        for m in 0..p.num_terms() {
            let sm = p.apply_summand(m, &det);
            let ratio = dot(&sm, &det) / dot(&det, &det);
            let residual = {
                let scaled = &det * ratio;
                let diff = &sm - &scaled;
                diff.norm()
            };
            assert!(
                residual < 1e-6,
                "summand {m}: S_m det is not a scalar multiple of det (residual {residual:.3e})"
            );
        }
    }
}
