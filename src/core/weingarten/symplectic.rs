// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::Mat;
use rayon::prelude::*;

use crate::core::factor::{dense_eigen_general, dense_self_adjoint_smallest, svd_uvt_svals};
use crate::scalar::C64;

/// Failure modes of the reduced Williamson path — the triggers for the non-Hermitian
/// generator fallback.
#[derive(Clone, Copy, Debug)]
pub enum WilliamsonError {
    /// `H^rr = A+B` is not positive-definite (Morse index > 0: `x` is not a stable minimizer, the
    /// reduced Eq. 20 form does not apply).
    HrrNotPd { min_eig: f64 },
    /// `Ω̃² = (H^rr)^½ H^ii (H^rr)^½` has a negative eigenvalue (RPA instability in `H^ii = A−B`).
    OmegaSqNegative { min_eig: f64 },
    /// The full `2n×2n` `𝔥^LR` is not positive-definite (complex-layer Williamson: Morse
    /// index > 0, `x` is not a stable minimizer — the Williamson theorem does not apply).
    HLrNotPd { min_eig: f64 },
}

const PAR_MIN_DIM: usize = 128;

fn matmul_flat(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    let row = |i: usize, ci: &mut [f64]| {
        for (k, &aik) in a[i * n..(i + 1) * n].iter().enumerate() {
            if aik != 0.0 {
                for j in 0..n {
                    ci[j] += aik * b[k * n + j];
                }
            }
        }
    };
    if n >= PAR_MIN_DIM {
        c.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, ci)| row(i, ci));
    } else {
        c.chunks_mut(n).enumerate().for_each(|(i, ci)| row(i, ci));
    }
    c
}

fn symmetrize_flat(a: &[f64], n: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            s[i * n + j] = 0.5 * (a[i * n + j] + a[j * n + i]);
        }
    }
    s
}

/// Reduced real-layer Williamson form `[GHC Eq. 20]`: the symplectic eigenvalues of
/// `𝔥 = diag(H^rr, H^ii)` are `ω = eig(Ω̃)`, `Ω̃ = ((H^rr)^½ H^ii (H^rr)^½)^½`, returned ascending.
/// `hrr`/`hii` are dense symmetric `n×n` row-major matrices over a `⟨·,·⟩`-orthonormal tangent
/// basis (radial mode excluded). `psd_tol` is the relative positive-definiteness guard: `H^rr`
/// must have `min_eig > psd_tol·max|eig|`, and `Ω̃²` eigenvalues below `−psd_tol·max|eig|` are an
/// error (small negatives above that are clamped to 0). Matrix roots via eigendecomposition of
/// the small Gram, through [`dense_self_adjoint_smallest`].
pub fn williamson_reduced(
    hrr: &[f64],
    hii: &[f64],
    n: usize,
    psd_tol: f64,
) -> Result<Vec<f64>, WilliamsonError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let hrr_s = symmetrize_flat(hrr, n);
    let hii_s = symmetrize_flat(hii, n);
    let (lam, vecs) = dense_self_adjoint_smallest::<f64>(&hrr_s, n, n);
    let max_abs = lam.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let min_eig = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_eig <= psd_tol * max_abs {
        return Err(WilliamsonError::HrrNotPd { min_eig });
    }
    let sq: Vec<f64> = lam.iter().map(|l| l.sqrt()).collect();
    let half = sym_apply_weights(&vecs, &sq, n);
    let m = matmul_flat(&matmul_flat(&half, &hii_s, n), &half, n);
    let m = symmetrize_flat(&m, n);
    let (mu, _) = dense_self_adjoint_smallest::<f64>(&m, n, 0);
    let mu_max = mu.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let mu_min = mu.iter().cloned().fold(f64::INFINITY, f64::min);
    if mu_min < -psd_tol * mu_max {
        return Err(WilliamsonError::OmegaSqNegative { min_eig: mu_min });
    }
    let mut omega: Vec<f64> = mu.iter().map(|&v| v.max(0.0).sqrt()).collect();
    omega.sort_by(|a, b| a.partial_cmp(b).expect("finite ω"));
    Ok(omega)
}

/// Which block of the block-diagonal real-layer `𝔥 = diag(H^rr, H^ii)` served as the PD metric in
/// [`williamson_reduced_signed`]; the OTHER block is the indefinite one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpaMetricBlock {
    /// `H^rr = A+B` was PD — the real-minimum / complex-saddle case, `H^ii` indefinite.
    Rr,
    /// `H^ii = A−B` was PD — `H^rr` indefinite, i.e. `x` is not a minimizer of the real energy.
    Ii,
}

/// The signed reduced-Williamson readout of [`williamson_reduced_signed`].
#[derive(Clone, Debug)]
pub struct SignedWilliamson {
    /// Signed `ω²`, ascending. A negative entry is a resonance: `ω = i·√|ω²|`, and `±ω` is then an
    /// imaginary eigenvalue pair of the generator `𝓜 = [[A,B],[−B,−A]]`.
    pub omega2: Vec<f64>,
    /// Count of entries strictly below `−psd_tol·max|ω²|` — the number of unstable pairs.
    pub n_neg: usize,
    /// Which block was PD and served as the metric.
    pub metric: RpaMetricBlock,
    /// `λ_min` of the metric block — the certificate that it really was PD.
    pub metric_min_eig: f64,
    /// Modes on the `H^rr` side, row-major `n×n`, column per mode, unit norm.
    pub modes_r: Vec<f64>,
    /// Modes on the `H^ii` side, row-major `n×n`, column per mode, unit norm.
    pub modes_i: Vec<f64>,
}

/// **Signed** reduced Williamson: the real-layer symplectic spectrum at a point where ONE of
/// `H^rr`, `H^ii` carries negative curvature — where [`williamson_reduced`] refuses with
/// `OmegaSqNegative` (resp. `HrrNotPd`) and the iterative kernels refuse with `MetricNotPd`.
pub fn williamson_reduced_signed(
    hrr: &[f64],
    hii: &[f64],
    n: usize,
    psd_tol: f64,
) -> Result<SignedWilliamson, WilliamsonError> {
    if n == 0 {
        return Ok(SignedWilliamson {
            omega2: Vec::new(),
            n_neg: 0,
            metric: RpaMetricBlock::Rr,
            metric_min_eig: f64::INFINITY,
            modes_r: Vec::new(),
            modes_i: Vec::new(),
        });
    }
    let hrr_s = symmetrize_flat(hrr, n);
    let hii_s = symmetrize_flat(hii, n);
    let (metric, lam, vecs, q) = {
        let (lam, vecs) = dense_self_adjoint_smallest::<f64>(&hrr_s, n, n);
        if pd_ok(&lam, psd_tol) {
            (RpaMetricBlock::Rr, lam, vecs, hii_s)
        } else {
            let rr_min = lam.iter().cloned().fold(f64::INFINITY, f64::min);
            let (lam_i, vecs_i) = dense_self_adjoint_smallest::<f64>(&hii_s, n, n);
            if pd_ok(&lam_i, psd_tol) {
                (RpaMetricBlock::Ii, lam_i, vecs_i, hrr_s)
            } else {
                return Err(WilliamsonError::HrrNotPd { min_eig: rr_min });
            }
        }
    };
    let metric_min_eig = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    let sq: Vec<f64> = lam.iter().map(|l| l.sqrt()).collect();
    let inv_sq: Vec<f64> = sq.iter().map(|s| 1.0 / s).collect();
    let p_half = sym_apply_weights(&vecs, &sq, n);
    let p_inv_half = sym_apply_weights(&vecs, &inv_sq, n);

    let m = symmetrize_flat(&matmul_flat(&matmul_flat(&p_half, &q, n), &p_half, n), n);
    let (mu, uvecs) = dense_self_adjoint_smallest::<f64>(&m, n, n);
    let mu_max = mu.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let n_neg = mu.iter().filter(|&&v| v < -psd_tol * mu_max).count();

    let mut u = vec![0.0f64; n * n];
    for (c, uc) in uvecs.iter().enumerate() {
        for (i, &ui) in uc.iter().enumerate() {
            u[i * n + c] = ui;
        }
    }
    let mut y = matmul_flat(&p_half, &u, n);
    let mut x = matmul_flat(&p_inv_half, &u, n);
    normalize_columns(&mut y, n);
    normalize_columns(&mut x, n);
    let (modes_r, modes_i) = match metric {
        RpaMetricBlock::Rr => (x, y),
        RpaMetricBlock::Ii => (y, x),
    };
    Ok(SignedWilliamson {
        omega2: mu,
        n_neg,
        metric,
        metric_min_eig,
        modes_r,
        modes_i,
    })
}

/// The symplectic BASIS at a resonant point ([`williamson_reduced_signed_vectors`]): the full
/// signed `ω²` spectrum, plus `S_r`/`S_i` restricted to the modes with `ω² > 0`.
#[derive(Clone, Debug)]
pub struct SignedWilliamsonBasis {
    /// Signed `ω²` ascending, all `n` modes.
    pub omega2: Vec<f64>,
    /// Count of negative entries — the modes for which no real symplectic basis exists.
    pub n_neg: usize,
    /// Which block served as the PD metric.
    pub metric: RpaMetricBlock,
    /// The POSITIVE `ω` ascending: `n_pos = n − n_neg` entries.
    pub omega: Vec<f64>,
    /// `S_r`, row-major `n × n_pos` (basis index × mode index) — the `H^rr` side.
    pub s_r: Vec<f64>,
    /// `S_i`, row-major `n × n_pos` — the `H^ii` side.
    pub s_i: Vec<f64>,
}

/// [`williamson_reduced_signed`] **with the symplectic basis of the stable modes** — the readout
/// a resonant reference needs in order to still report transition moments and oscillator
/// strengths for its real excitations.
pub fn williamson_reduced_signed_vectors(
    hrr: &[f64],
    hii: &[f64],
    n: usize,
    psd_tol: f64,
) -> Result<SignedWilliamsonBasis, WilliamsonError> {
    if n == 0 {
        return Ok(SignedWilliamsonBasis {
            omega2: Vec::new(),
            n_neg: 0,
            metric: RpaMetricBlock::Rr,
            omega: Vec::new(),
            s_r: Vec::new(),
            s_i: Vec::new(),
        });
    }
    let hrr_s = symmetrize_flat(hrr, n);
    let hii_s = symmetrize_flat(hii, n);
    let (metric, lam, vecs, q) = {
        let (lam, vecs) = dense_self_adjoint_smallest::<f64>(&hrr_s, n, n);
        if pd_ok(&lam, psd_tol) {
            (RpaMetricBlock::Rr, lam, vecs, hii_s)
        } else {
            let rr_min = lam.iter().cloned().fold(f64::INFINITY, f64::min);
            let (lam_i, vecs_i) = dense_self_adjoint_smallest::<f64>(&hii_s, n, n);
            if pd_ok(&lam_i, psd_tol) {
                (RpaMetricBlock::Ii, lam_i, vecs_i, hrr_s)
            } else {
                return Err(WilliamsonError::HrrNotPd { min_eig: rr_min });
            }
        }
    };
    let sq: Vec<f64> = lam.iter().map(|l| l.sqrt()).collect();
    let inv_sq: Vec<f64> = sq.iter().map(|s| 1.0 / s).collect();
    let p_half = sym_apply_weights(&vecs, &sq, n);
    let p_inv_half = sym_apply_weights(&vecs, &inv_sq, n);

    let m = symmetrize_flat(&matmul_flat(&matmul_flat(&p_half, &q, n), &p_half, n), n);
    let (mu, uvecs) = dense_self_adjoint_smallest::<f64>(&m, n, n);
    let mu_max = mu.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let cut = psd_tol * mu_max;
    let keep: Vec<usize> = (0..n).filter(|&j| mu[j] > cut).collect();
    let n_neg = mu.iter().filter(|&&v| v < -cut).count();
    let n_pos = keep.len();

    let mut u_wp = vec![0.0f64; n * n_pos];
    let mut u_wm = vec![0.0f64; n * n_pos];
    let mut omega = Vec::with_capacity(n_pos);
    for (c, &src) in keep.iter().enumerate() {
        let w = mu[src].sqrt();
        omega.push(w);
        let wsq = w.sqrt();
        for (i, &ui) in uvecs[src].iter().enumerate() {
            u_wp[i * n_pos + c] = ui * wsq;
            u_wm[i * n_pos + c] = ui / wsq;
        }
    }
    let s_p = matmul_rect(&p_inv_half, &u_wp, n, n_pos);
    let s_q = matmul_rect(&p_half, &u_wm, n, n_pos);
    let (s_r, s_i) = match metric {
        RpaMetricBlock::Rr => (s_p, s_q),
        RpaMetricBlock::Ii => (s_q, s_p),
    };
    Ok(SignedWilliamsonBasis {
        omega2: mu,
        n_neg,
        metric,
        omega,
        s_r,
        s_i,
    })
}

fn matmul_rect(a: &[f64], b: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * p];
    let row = |i: usize, ci: &mut [f64]| {
        for (k, &aik) in a[i * n..(i + 1) * n].iter().enumerate() {
            if aik != 0.0 {
                for j in 0..p {
                    ci[j] += aik * b[k * p + j];
                }
            }
        }
    };
    if n >= PAR_MIN_DIM {
        c.par_chunks_mut(p)
            .enumerate()
            .for_each(|(i, ci)| row(i, ci));
    } else {
        c.chunks_mut(p).enumerate().for_each(|(i, ci)| row(i, ci));
    }
    c
}

fn pd_ok(lam: &[f64], psd_tol: f64) -> bool {
    let max_abs = lam.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let min_eig = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    min_eig > psd_tol * max_abs
}

fn normalize_columns(a: &mut [f64], n: usize) {
    for c in 0..n {
        let nrm2: f64 = (0..n).map(|i| a[i * n + c] * a[i * n + c]).sum();
        if nrm2 > 0.0 {
            let s = nrm2.sqrt().recip();
            for i in 0..n {
                a[i * n + c] *= s;
            }
        }
    }
}

/// Per-mode **Weingarten fraction** `f_W(v) = |⟨v, B v⟩| / |⟨v, A v⟩|` over the dense term-A and
/// term-B Grams (row-major `n×n`), for the modes stored as COLUMNS of `modes` (row-major `n×k`,
/// the layout [`SignedWilliamson::modes_r`]/`modes_i` use with `k = n`).
pub fn weingarten_fraction(a: &[f64], b: &[f64], modes: &[f64], n: usize, k: usize) -> Vec<f64> {
    weingarten_quadratic_forms(a, b, modes, n, k)
        .into_iter()
        .map(|(qa, qb)| {
            if qa == 0.0 {
                f64::INFINITY
            } else {
                qb.abs() / qa.abs()
            }
        })
        .collect()
}

/// The **signed** pair `(⟨v, A v⟩, ⟨v, B v⟩)` per mode — the raw material of
/// [`weingarten_fraction`], kept separately because the `f_W = 1` identity documented there is a
/// statement about signs as well as magnitudes, and `|·|/|·|` discards them.
pub fn weingarten_quadratic_forms(
    a: &[f64],
    b: &[f64],
    modes: &[f64],
    n: usize,
    k: usize,
) -> Vec<(f64, f64)> {
    (0..k)
        .into_par_iter()
        .map(|c| {
            let col: Vec<f64> = (0..n).map(|i| modes[i * k + c]).collect();
            (quad_form_flat(a, &col, n), quad_form_flat(b, &col, n))
        })
        .collect()
}

/// Gather the listed columns of a row-major `n×n` flat into a row-major `n×cols.len()` flat — the
/// selector in front of [`weingarten_fraction`], which is `O(k·n²)` and so must never be handed
/// all `n` modes at molecular scale.
pub fn extract_columns(a: &[f64], n: usize, cols: &[usize]) -> Vec<f64> {
    let k = cols.len();
    let mut out = vec![0.0f64; n * k];
    for (c, &src) in cols.iter().enumerate() {
        for i in 0..n {
            out[i * k + c] = a[i * n + src];
        }
    }
    out
}

fn quad_form_flat(m: &[f64], v: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|i| v[i] * (0..n).map(|j| m[i * n + j] * v[j]).sum::<f64>())
        .sum()
}

/// Full `2n×2n` Williamson symplectic eigenvalues `[GHC Eq. 18]` (complex layer — `H^ri ≠ 0`
/// so the reduced [`williamson_reduced`] Eq.-20 shortcut is invalid). `hll` is the real-symmetric
/// `2n×2n` `𝔥^LR` (row-major flat) over the canonical real basis `(u_j, −J u_j)`, in which
/// `J = [[0, I],[−I, 0]]`. `two_n = 2n` must be even. `psd_tol` is the relative PD guard
/// (Williamson requires `𝔥^LR ≻ 0`; on failure use [`symplectic_generator_full`] to diagnose).
pub fn williamson_full(
    hll: &[f64],
    two_n: usize,
    psd_tol: f64,
) -> Result<Vec<f64>, WilliamsonError> {
    if two_n == 0 {
        return Ok(Vec::new());
    }
    debug_assert!(
        two_n.is_multiple_of(2),
        "williamson_full: dimension must be even"
    );
    let n = two_n / 2;
    let hs = symmetrize_flat(hll, two_n);
    let (lam, vecs) = dense_self_adjoint_smallest::<f64>(&hs, two_n, two_n);
    let max_abs = lam.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let min_eig = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_eig <= psd_tol * max_abs {
        return Err(WilliamsonError::HLrNotPd { min_eig });
    }
    let mut m = vec![0.0f64; two_n * two_n];
    for (c, vc) in vecs.iter().enumerate() {
        let sq = lam[c].sqrt();
        for (i, &vi) in vc.iter().enumerate() {
            m[i * two_n + c] = vi * sq;
        }
    }
    let mut jm = vec![0.0f64; two_n * two_n];
    for i in 0..n {
        for j in 0..two_n {
            jm[i * two_n + j] = m[(n + i) * two_n + j];
            jm[(n + i) * two_n + j] = -m[i * two_n + j];
        }
    }
    let mut mt = vec![0.0f64; two_n * two_n];
    for i in 0..two_n {
        for j in 0..two_n {
            mt[j * two_n + i] = m[i * two_n + j];
        }
    }
    let k = matmul_flat(&mt, &jm, two_n);
    let mut kt = vec![0.0f64; two_n * two_n];
    for i in 0..two_n {
        for j in 0..two_n {
            kt[j * two_n + i] = k[i * two_n + j];
        }
    }
    let s = symmetrize_flat(&matmul_flat(&k, &kt, two_n), two_n);
    let (mu, _) = dense_self_adjoint_smallest::<f64>(&s, two_n, 0);
    let mu_max = mu.last().copied().unwrap_or(0.0).abs();
    let mut omega = Vec::with_capacity(n);
    for i in 0..n {
        let (lo, hi) = (mu[2 * i], mu[2 * i + 1]);
        debug_assert!(
            (hi - lo).abs() <= 1e-6 * mu_max.max(1e-300),
            "williamson_full: −K² eigenvalue pairing broken at {i}: {lo} vs {hi}"
        );
        omega.push((0.5 * (lo.max(0.0) + hi.max(0.0))).sqrt());
    }
    Ok(omega)
}

/// Thin SVD of a real row-major `m×n` flat: `(U m×r, σ descending, Vᵀ r×n)` as flats,
/// `r = min(m, n)` — the flat-slice face of [`crate::core::factor::svd_uvt_svals`] so
/// `solvers/hess.rs` (faer-free) can decompose small dense Grams.
pub fn dense_svd_flat(a: &[f64], m: usize, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut mat = Mat::<f64>::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            *mat.get_mut(i, j) = a[i * n + j];
        }
    }
    let (u, s, vt) = svd_uvt_svals(mat.as_ref());
    let r = s.len();
    let mut uf = vec![0.0f64; m * r];
    for i in 0..m {
        for j in 0..r {
            uf[i * r + j] = *u.get(i, j);
        }
    }
    let mut vtf = vec![0.0f64; r * n];
    for i in 0..r {
        for j in 0..n {
            vtf[i * n + j] = *vt.get(i, j);
        }
    }
    (uf, s, vtf)
}

fn sym_pow_flat(a_sym: &[f64], n: usize, pow: f64, psd_tol: f64) -> Result<Vec<f64>, f64> {
    let (lam, vecs) = dense_self_adjoint_smallest::<f64>(a_sym, n, n);
    let max_abs = lam.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let min_eig = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_eig <= psd_tol * max_abs {
        return Err(min_eig);
    }
    let w: Vec<f64> = lam.iter().map(|&l| l.powf(pow)).collect();
    Ok(sym_apply_weights(&vecs, &w, n))
}

fn sym_apply_weights(vecs: &[Vec<f64>], w: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    let row = |i: usize, oi: &mut [f64]| {
        for (c, vc) in vecs.iter().enumerate() {
            let viw = vc[i] * w[c];
            for (j, &vj) in vc.iter().enumerate() {
                oi[j] += viw * vj;
            }
        }
    };
    if n >= PAR_MIN_DIM {
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, oi)| row(i, oi));
    } else {
        out.chunks_mut(n).enumerate().for_each(|(i, oi)| row(i, oi));
    }
    out
}

fn jmul_flat(a: &[f64], two_n: usize) -> Vec<f64> {
    let n = two_n / 2;
    let mut ja = vec![0.0f64; two_n * two_n];
    for i in 0..n {
        for j in 0..two_n {
            ja[i * two_n + j] = a[(n + i) * two_n + j];
            ja[(n + i) * two_n + j] = -a[i * two_n + j];
        }
    }
    ja
}

/// Canonical form of a real **nonsingular antisymmetric** `K` (row-major `2n×2n`): returns
/// `(c, q)` with `c` ascending positive and `q` orthogonal (row-major, columns `[U | V]`) such
/// that `QᵀKQ = [[0, C], [−C, 0]]`, `C = diag(c)`.
pub fn skew_canonical(k: &[f64], two_n: usize) -> (Vec<f64>, Vec<f64>) {
    let n = two_n / 2;
    debug_assert!(two_n.is_multiple_of(2), "skew_canonical: odd dimension");
    if two_n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut kt = vec![0.0f64; two_n * two_n];
    for i in 0..two_n {
        for j in 0..two_n {
            kt[j * two_n + i] = k[i * two_n + j];
        }
    }
    let s = symmetrize_flat(&matmul_flat(k, &kt, two_n), two_n);
    let (_, vecs) = dense_self_adjoint_smallest::<f64>(&s, two_n, two_n);
    let scale = s
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs()))
        .sqrt()
        .max(1e-300);

    let mut us: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut cs: Vec<f64> = Vec::with_capacity(n);
    for w in &vecs {
        if us.len() == n {
            break;
        }
        let mut u = w.clone();
        for _pass in 0..2 {
            for basis in us.iter().chain(vs.iter()) {
                let c: f64 = basis.iter().zip(&u).map(|(&b, &x)| b * x).sum();
                for (x, &b) in u.iter_mut().zip(basis) {
                    *x -= c * b;
                }
            }
        }
        let un: f64 = u.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if un <= 1e-8 {
            continue;
        }
        for x in &mut u {
            *x /= un;
        }
        let mut ku = vec![0.0f64; two_n];
        for (i, kui) in ku.iter_mut().enumerate() {
            *kui = k[i * two_n..(i + 1) * two_n]
                .iter()
                .zip(&u)
                .map(|(&kij, &uj)| kij * uj)
                .sum();
        }
        let c: f64 = ku.iter().map(|&x| x * x).sum::<f64>().sqrt();
        debug_assert!(
            c > 1e-12 * scale,
            "skew_canonical: (near-)singular K (block value {c:.3e})"
        );
        let v: Vec<f64> = ku.iter().map(|&x| -x / c).collect();
        us.push(u);
        vs.push(v);
        cs.push(c);
    }
    debug_assert_eq!(us.len(), n, "skew_canonical: block extraction incomplete");

    let mut order: Vec<usize> = (0..cs.len()).collect();
    order.sort_by(|&a, &b| cs[a].partial_cmp(&cs[b]).expect("finite block values"));
    let mut c_sorted = Vec::with_capacity(n);
    let mut q = vec![0.0f64; two_n * two_n];
    for (slot, &src) in order.iter().enumerate() {
        c_sorted.push(cs[src]);
        for i in 0..two_n {
            q[i * two_n + slot] = us[src][i];
            q[i * two_n + (n + slot)] = vs[src][i];
        }
    }
    (c_sorted, q)
}

/// `(ω ascending, S_r, S_i)` from [`williamson_reduced_vectors`] — `n×n` row-major flats.
pub type ReducedWilliamsonBasis = (Vec<f64>, Vec<f64>, Vec<f64>);

/// [`williamson_reduced`] with the symplectic basis: for the block-diagonal real-layer
/// `𝔥 = diag(H^rr, H^ii)` returns `(ω ascending, S_r, S_i)` (`n×n` row-major each) with
/// `S_rᵀ H^rr S_r = Ω`, `S_iᵀ H^ii S_i = Ω`, `S_rᵀ S_i = I` — i.e. `S = diag(S_r, S_i)` is
/// symplectic (`SᵀJS = J`) and `Sᵀ𝔥S = diag(Ω, Ω)` in the canonical real basis.
pub fn williamson_reduced_vectors(
    hrr: &[f64],
    hii: &[f64],
    n: usize,
    psd_tol: f64,
) -> Result<ReducedWilliamsonBasis, WilliamsonError> {
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let hrr_s = symmetrize_flat(hrr, n);
    let hii_s = symmetrize_flat(hii, n);
    let (lam, vecs) = dense_self_adjoint_smallest::<f64>(&hrr_s, n, n);
    let max_abs = lam.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let min_eig = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_eig <= psd_tol * max_abs {
        return Err(WilliamsonError::HrrNotPd { min_eig });
    }
    let sq: Vec<f64> = lam.iter().map(|l| l.sqrt()).collect();
    let inv_sq: Vec<f64> = sq.iter().map(|s| 1.0 / s).collect();
    let half = sym_apply_weights(&vecs, &sq, n);
    let inv_half = sym_apply_weights(&vecs, &inv_sq, n);
    let m = symmetrize_flat(&matmul_flat(&matmul_flat(&half, &hii_s, n), &half, n), n);
    let (mu, uvecs) = dense_self_adjoint_smallest::<f64>(&m, n, n);
    let mu_max = mu.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    let mu_min = mu.iter().cloned().fold(f64::INFINITY, f64::min);
    if mu_min <= psd_tol * mu_max {
        return Err(WilliamsonError::OmegaSqNegative { min_eig: mu_min });
    }
    let omega: Vec<f64> = mu.iter().map(|&v| v.sqrt()).collect();
    let mut u_wp = vec![0.0f64; n * n];
    let mut u_wm = vec![0.0f64; n * n];
    for (c, uc) in uvecs.iter().enumerate() {
        let wsq = omega[c].sqrt();
        for (i, &ui) in uc.iter().enumerate() {
            u_wp[i * n + c] = ui * wsq;
            u_wm[i * n + c] = ui / wsq;
        }
    }
    let s_r = matmul_flat(&inv_half, &u_wp, n);
    let s_i = matmul_flat(&half, &u_wm, n);
    Ok((omega, s_r, s_i))
}

/// [`williamson_full`] with the symplectic basis: returns `(ω ascending, S)` with `S` row-major
/// `2n×2n`, `SᵀJS = J` and `Sᵀ𝔥S = diag(Ω, Ω)` over the canonical real basis
/// (`J = [[0, I],[−I, 0]]`).
pub fn williamson_full_vectors(
    hll: &[f64],
    two_n: usize,
    psd_tol: f64,
) -> Result<(Vec<f64>, Vec<f64>), WilliamsonError> {
    if two_n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    debug_assert!(
        two_n.is_multiple_of(2),
        "williamson_full_vectors: dimension must be even"
    );
    let n = two_n / 2;
    let hs = symmetrize_flat(hll, two_n);
    let inv_half = sym_pow_flat(&hs, two_n, -0.5, psd_tol)
        .map_err(|min_eig| WilliamsonError::HLrNotPd { min_eig })?;
    let k = matmul_flat(&inv_half, &jmul_flat(&inv_half, two_n), two_n);
    let (c, q) = skew_canonical(&k, two_n);
    let mut omega = Vec::with_capacity(n);
    let mut q_scaled = vec![0.0f64; two_n * two_n];
    for j in 0..n {
        let src = n - 1 - j;
        let w = 1.0 / c[src];
        omega.push(w);
        let wsq = w.sqrt();
        for i in 0..two_n {
            q_scaled[i * two_n + j] = q[i * two_n + src] * wsq;
            q_scaled[i * two_n + (n + j)] = q[i * two_n + (n + src)] * wsq;
        }
    }
    let s = matmul_flat(&inv_half, &q_scaled, two_n);
    Ok((omega, s))
}

/// Eigenvalues of the full-form symplectic generator: `−i·eig(J·𝔥^LR)` over the canonical real
/// basis (`J = [[0, I],[−I, 0]]`). At a stable minimizer (`𝔥^LR ≻ 0`) `eig(J𝔥) = ±iω`, so the
/// returned values are real `±ω` pairs — the same semantics as [`rpa_generator_eigenvalues`] on
/// the real layer; complex entries diagnose symplectic instability. Serves as the cross-check
/// of [`williamson_full`] and the complex-layer fallback path.
pub fn symplectic_generator_full(hll: &[f64], two_n: usize) -> Vec<C64> {
    if two_n == 0 {
        return Vec::new();
    }
    debug_assert!(
        two_n.is_multiple_of(2),
        "symplectic_generator_full: dimension must be even"
    );
    let n = two_n / 2;
    let hs = symmetrize_flat(hll, two_n);
    let mut jh = vec![0.0f64; two_n * two_n];
    for i in 0..n {
        for j in 0..two_n {
            jh[i * two_n + j] = hs[(n + i) * two_n + j];
            jh[(n + i) * two_n + j] = -hs[i * two_n + j];
        }
    }
    let (eigs, _) = dense_eigen_general::<f64>(&jh, two_n);
    eigs.into_iter().map(|e| C64::new(e.im, -e.re)).collect()
}

/// Eigenvalues of the non-Hermitian RPA generator `𝓜 = [[A, B], [−B, −A]]` (fallback and
/// cross-check path): `±ω` pairs, complex iff the critical point is symplectically unstable.
/// `a`/`b` are the dense `n×n` row-major Gram matrices of the term-A and term-B applies.
pub fn rpa_generator_eigenvalues(a: &[f64], b: &[f64], n: usize) -> Vec<C64> {
    let nn = 2 * n;
    let mut g = vec![0.0f64; nn * nn];
    for i in 0..n {
        for j in 0..n {
            g[i * nn + j] = a[i * n + j];
            g[i * nn + (n + j)] = b[i * n + j];
            g[(n + i) * nn + j] = -b[i * n + j];
            g[(n + i) * nn + (n + j)] = -a[i * n + j];
        }
    }
    dense_eigen_general::<f64>(&g, nn).0
}
