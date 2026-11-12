// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The frozen-RDM global orbital step: minimize `E(U) = ⟨x|H(U)|x⟩` over `U ∈ SO(d)` for a FIXED
//! state `x`, through its one- and two-body reduced density matrices.

use std::collections::VecDeque;

use crate::coeff::CoeffTensor;
use crate::core::factor::{dense_self_adjoint_smallest, gemm_nt_rowblocks};
use crate::integrals::Integrals;
use crate::pairgram::generalized_gamma2;
use crate::rdm::transition_1rdm;
use crate::rotation::{
    PairMask, compose_rotations, expm_antisym, identity_matrix, rotate_integrals,
};
use crate::vector::TtVector;

/// The spin-summed one- and two-body RDMs of a unit-norm state, as the frozen orbital step
/// consumes them.
#[derive(Clone, Debug)]
pub struct Rdm12 {
    /// Orbital count.
    pub d: usize,
    /// Electron count `N` of the state's sector.
    pub n: usize,
    /// `γ[p,q] = ⟨x|E_pq|x⟩`, rank-2 `d×d`. Symmetric only to roundoff.
    pub gamma1: CoeffTensor<f64>,
    /// `Γ₂[i,j,k,l] = ⟨x|E_ij E_kl|x⟩` (generalized, NOT normal-ordered, spin-summed), dense
    /// `d⁴` row-major at `((i·d + j)·d + k)·d + l`. It satisfies
    /// `Σ_k Γ₂[i,j,k,k] = N·γ[i,j]`.
    pub gamma2: Vec<f64>,
}

impl Rdm12 {
    /// Both RDMs of `x`, by [`transition_1rdm`]`(x, x)` and [`generalized_gamma2`]`(x)`.
    ///
    /// # Panics
    ///
    /// On a point-group-typed space (`|G| > 1`), and when `|‖x‖ − 1| > 1e-10` (or `‖x‖` is not
    /// finite), both checked before any sweep. Non-unit boundary ranks panic inside the
    /// consumed kernels.
    pub fn of(x: &TtVector<f64>) -> Self {
        let space = x.space;
        assert!(
            space.pg.is_trivial(),
            "Rdm12::of: a point-group-typed space (|G| = {}) is not supported; the two-body \
             moment sweep has no point-group gating",
            space.pg.order()
        );
        let dev = (x.norm() - 1.0).abs();
        assert!(
            dev <= 1e-10,
            "Rdm12::of: x must be unit-norm (|‖x‖ − 1| = {dev:e} > 1e-10); the RDMs are raw \
             ⟨x|·|x⟩ moments"
        );
        Rdm12 {
            d: space.d,
            n: space.n(),
            gamma1: transition_1rdm(x, x),
            gamma2: generalized_gamma2(x),
        }
    }
}

fn check_ints_rdm(ints: &Integrals<f64>, rdm: &Rdm12, who: &str) {
    let d = rdm.d;
    assert!(
        ints.norb == d,
        "{who}: ints.norb = {} does not match rdm.d = {d}",
        ints.norb
    );
    assert!(
        ints.t.rank == 2 && ints.t.d == d,
        "{who}: t must be rank-2 over d = {d} (ints.norb)"
    );
    assert!(
        ints.v.rank == 4 && ints.v.d == d,
        "{who}: v must be rank-4 over d = {d} (ints.norb)"
    );
    assert!(
        rdm.gamma1.rank == 2 && rdm.gamma1.d == d,
        "{who}: rdm.gamma1 must be rank-2 over rdm.d = {d}"
    );
    assert!(
        rdm.gamma2.len() == d * d * d * d,
        "{who}: rdm.gamma2 must hold d⁴ = {} entries (got {})",
        d * d * d * d,
        rdm.gamma2.len()
    );
}

fn form_p(rdm: &Rdm12) -> Vec<f64> {
    let d = rdm.d;
    let g = &rdm.gamma1.data;
    let mut p = rdm.gamma2.clone();
    for a in 0..d {
        for b in 0..d {
            for e in 0..d {
                p[((a * d + b) * d + b) * d + e] -= g[a * d + e];
            }
        }
    }
    p
}

fn energy_tvp(t: &[f64], v: &[f64], gamma: &[f64], p: &[f64]) -> f64 {
    let mut e1 = 0.0f64;
    for (a, b) in t.iter().zip(gamma) {
        e1 += a * b;
    }
    let mut e2 = 0.0f64;
    for (a, b) in v.iter().zip(p) {
        e2 += a * b;
    }
    e1 + 0.5 * e2
}

const FOCK_A_BLOCK: usize = 9;

fn fock_work(d: usize) -> usize {
    d.saturating_pow(5).saturating_mul(4)
}

fn fock_tvp(t: &[f64], v: &[f64], gamma: &[f64], p: &[f64], d: usize) -> Vec<f64> {
    if d >= 2 && crate::par::par_driver_fork(fock_work(d)) {
        return fock_tvp_par(t, v, gamma, p, d, FOCK_A_BLOCK);
    }
    let d2 = d * d;
    let d3 = d2 * d;
    let mut y = fock_one_body(t, gamma, d);
    let mut s1 = vec![0.0f64; d2];
    for i in 0..d {
        let vi = &v[i * d3..(i + 1) * d3];
        for a in 0..d {
            s1[i * d + a] = dot(vi, &p[a * d3..(a + 1) * d3]);
        }
    }
    let mut s2 = vec![0.0f64; d2];
    for b in 0..d {
        for i in 0..d {
            let off_v = (b * d + i) * d2;
            let vbi = &v[off_v..off_v + d2];
            for a in 0..d {
                let off_p = (b * d + a) * d2;
                s2[i * d + a] += dot(vbi, &p[off_p..off_p + d2]);
            }
        }
    }
    let mut s3 = vec![0.0f64; d2];
    for w in 0..d2 {
        for i in 0..d {
            let off_v = (w * d + i) * d;
            let vwi = &v[off_v..off_v + d];
            for a in 0..d {
                let off_p = (w * d + a) * d;
                s3[i * d + a] += dot(vwi, &p[off_p..off_p + d]);
            }
        }
    }
    let s4 = fock_slot4(v, p, d);
    for (f, yf) in y.iter_mut().enumerate() {
        *yf += 0.5 * (((s1[f] + s2[f]) + s3[f]) + s4[f]);
    }
    y
}

fn fock_one_body(t: &[f64], gamma: &[f64], d: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; d * d];
    for i in 0..d {
        for a in 0..d {
            let mut s = 0.0f64;
            for b in 0..d {
                s += t[i * d + b] * gamma[a * d + b];
            }
            let mut s2 = 0.0f64;
            for b in 0..d {
                s2 += t[b * d + i] * gamma[b * d + a];
            }
            y[i * d + a] = s + s2;
        }
    }
    y
}

fn fock_slot4(v: &[f64], p: &[f64], d: usize) -> Vec<f64> {
    let d3 = d * d * d;
    let mut s4 = vec![0.0f64; d * d];
    for w in 0..d3 {
        let vw = &v[w * d..(w + 1) * d];
        let pw = &p[w * d..(w + 1) * d];
        for (i, &vi) in vw.iter().enumerate() {
            let row = &mut s4[i * d..(i + 1) * d];
            for (acc, &pa) in row.iter_mut().zip(pw) {
                *acc += vi * pa;
            }
        }
    }
    s4
}

fn fock_slots123_block(
    v: &[f64],
    p: &[f64],
    d: usize,
    i: usize,
    a_rng: std::ops::Range<usize>,
) -> Vec<f64> {
    let d2 = d * d;
    let d3 = d2 * d;
    let n = a_rng.len();
    let vi = &v[i * d3..(i + 1) * d3];
    let s1: Vec<f64> = a_rng
        .clone()
        .map(|a| dot(vi, &p[a * d3..(a + 1) * d3]))
        .collect();
    let mut s2 = vec![0.0f64; n];
    for b in 0..d {
        let off_v = (b * d + i) * d2;
        let vbi = &v[off_v..off_v + d2];
        for (acc, a) in s2.iter_mut().zip(a_rng.clone()) {
            let off_p = (b * d + a) * d2;
            *acc += dot(vbi, &p[off_p..off_p + d2]);
        }
    }
    let mut s3 = vec![0.0f64; n];
    for w in 0..d2 {
        let off_v = (w * d + i) * d;
        let vwi = &v[off_v..off_v + d];
        for (acc, a) in s3.iter_mut().zip(a_rng.clone()) {
            let off_p = (w * d + a) * d;
            *acc += dot(vwi, &p[off_p..off_p + d]);
        }
    }
    (0..n).map(|k| (s1[k] + s2[k]) + s3[k]).collect()
}

fn fock_tvp_par(t: &[f64], v: &[f64], gamma: &[f64], p: &[f64], d: usize, ab: usize) -> Vec<f64> {
    use rayon::prelude::*;
    assert!(ab >= 1, "fock_tvp_par: a-block size must be positive");
    let nb = d.div_ceil(ab);
    let ((mut y, s4), q) = rayon::join(
        || crate::par::nested_scope(|| (fock_one_body(t, gamma, d), fock_slot4(v, p, d))),
        || {
            (0..d * nb)
                .into_par_iter()
                .map(|task| {
                    let (i, a0) = (task / nb, (task % nb) * ab);
                    crate::par::nested_scope(|| {
                        fock_slots123_block(v, p, d, i, a0..(a0 + ab).min(d))
                    })
                })
                .collect::<Vec<Vec<f64>>>()
                .concat()
        },
    );
    for (f, yf) in y.iter_mut().enumerate() {
        *yf += 0.5 * (q[f] + s4[f]);
    }
    y
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// The electronic energy `⟨t, γ⟩ + ½⟨v, P⟩` of the state the RDMs came from, under the
/// Hamiltonian of `ints` (`ints.e0` is NOT added). For a unit-norm `x` it equals
/// `⟨x|H(ints.t, ints.v)|x⟩` to roundoff, for any real `(t, v)`.
///
/// # Panics
///
/// When `ints.norb ≠ rdm.d`, or on inconsistent tensor shapes.
pub fn energy_from_rdms(ints: &Integrals<f64>, rdm: &Rdm12) -> f64 {
    check_ints_rdm(ints, rdm, "energy_from_rdms");
    let p = form_p(rdm);
    energy_tvp(&ints.t.data, &ints.v.data, &rdm.gamma1.data, &p)
}

/// The four-slot generalized Fock matrix `Y[i,a] = ∂E(U)/∂U[i,a]` at `U = I` (formula in the
/// module header), `d×d` row-major `Y[i·d + a]`. Uses no symmetry of `t` or `v`.
///
/// # Panics
///
/// When `ints.norb ≠ rdm.d`, or on inconsistent tensor shapes.
pub fn generalized_fock(ints: &Integrals<f64>, rdm: &Rdm12) -> Vec<f64> {
    check_ints_rdm(ints, rdm, "generalized_fock");
    let p = form_p(rdm);
    fock_tvp(&ints.t.data, &ints.v.data, &rdm.gamma1.data, &p, rdm.d)
}

/// `g = Yᵀ − Y` from [`generalized_fock`]: the frozen-state orbital gradient, under the crate's
/// sign pin `d/dh ⟨x|H(rotate_integrals(ints, exp(−hκ)))|x⟩|₀ = ½ Σ_{p,q} g[p,q]·κ[p,q]`.
///
/// # Panics
///
/// When `ints.norb ≠ rdm.d`, or on inconsistent tensor shapes.
pub fn frozen_orbital_gradient(ints: &Integrals<f64>, rdm: &Rdm12) -> CoeffTensor<f64> {
    check_ints_rdm(ints, rdm, "frozen_orbital_gradient");
    let d = rdm.d;
    let y = generalized_fock(ints, rdm);
    let mut g = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                g.data[p * d + q] = y[q * d + p] - y[p * d + q];
            }
        }
    }
    g
}

/// The exact frozen-RDM orbital Hessian at `U = I` of the chart `k ↦ E(exp(−κ(k)))` on `ints`
/// (formula in the module header), `n×n` row-major `h[a·n + b]` with `n = d(d−1)/2` and rows and
/// columns in [`PairMask::all`] order, in pair-norm coordinates: `E(exp(−κ(k))) = E + g·k +
/// ½kᵀHk + O(k³)` with `g` the half-vector of [`frozen_orbital_gradient`]. At a point `U` the
/// BODY-frame Hessian (the chart `U·exp(−κ(k))`) is the same function of
/// `rotate_integrals(ints, U)` with the same RDMs. Uses no symmetry of `t`, `v`, `γ` or `P`, and
/// is EXACTLY symmetric (`h[a·n + b]` and `h[b·n + a]` bit-identical). Empty for `d ≤ 1`.
///
/// # Panics
///
/// When `ints.norb ≠ rdm.d`, or on inconsistent tensor shapes.
pub fn frozen_orbital_hessian(ints: &Integrals<f64>, rdm: &Rdm12) -> Vec<f64> {
    check_ints_rdm(ints, rdm, "frozen_orbital_hessian");
    let d = rdm.d;
    let p = form_p(rdm);
    let (t, v, gamma) = (&ints.t.data, &ints.v.data, &rdm.gamma1.data);
    let y = fock_tvp(t, v, gamma, &p, d);
    hessian_tvp(t, v, gamma, &p, &y, d, &PairMask::all(d).keep)
}

const SLOT_PAIRS: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [0, 2, 1, 3],
    [0, 3, 1, 2],
    [1, 2, 0, 3],
    [1, 3, 0, 2],
    [2, 3, 0, 1],
];

fn permute4(src: &[f64], d: usize, perm: [usize; 4]) -> Vec<f64> {
    let mut out = vec![0.0f64; src.len()];
    let mut x = [0usize; 4];
    let mut f = 0usize;
    for x0 in 0..d {
        x[0] = x0;
        for x1 in 0..d {
            x[1] = x1;
            for x2 in 0..d {
                x[2] = x2;
                for x3 in 0..d {
                    x[3] = x3;
                    let o = ((x[perm[0]] * d + x[perm[1]]) * d + x[perm[2]]) * d + x[perm[3]];
                    out[o] = src[f];
                    f += 1;
                }
            }
        }
    }
    out
}

fn hessian_tvp(
    t: &[f64],
    v: &[f64],
    gamma: &[f64],
    p: &[f64],
    y: &[f64],
    d: usize,
    pairs: &[(usize, usize)],
) -> Vec<f64> {
    let n = pairs.len();
    if n == 0 {
        return Vec::new();
    }
    let d2 = d * d;
    let mut msum = gemm_nt_rowblocks(v, p, d2, d2, d2);
    for &perm in &SLOT_PAIRS[1..] {
        let vp = permute4(v, d, perm);
        let pp = permute4(p, d, perm);
        let m = gemm_nt_rowblocks(&vp, &pp, d2, d2, d2);
        for (acc, x) in msum.iter_mut().zip(&m) {
            *acc += x;
        }
    }
    for (f, w) in msum.iter_mut().enumerate() {
        *w = t[f / d2] * gamma[f % d2] + 0.5 * *w;
    }
    let tw = msum;
    let t4 = |i: usize, j: usize, a: usize, b: usize| tw[((i * d + j) * d + a) * d + b];
    let sym = |x: usize, z: usize| 0.5 * (y[x * d + z] + y[z * d + x]);
    let mut h = vec![0.0f64; n * n];
    for (ia, &(pp, qq)) in pairs.iter().enumerate() {
        for (ib, &(rr, ss)) in pairs.iter().enumerate().skip(ia) {
            let c_pr = ((t4(pp, rr, qq, ss) - t4(pp, ss, qq, rr)) - t4(qq, rr, pp, ss))
                + t4(qq, ss, pp, rr);
            let c_rp = ((t4(rr, pp, ss, qq) - t4(rr, qq, ss, pp)) - t4(ss, pp, rr, qq))
                + t4(ss, qq, rr, pp);
            let mut dd = 0.0f64;
            if qq == rr {
                dd += sym(pp, ss);
            }
            if qq == ss {
                dd -= sym(pp, rr);
            }
            if pp == rr {
                dd -= sym(qq, ss);
            }
            if pp == ss {
                dd += sym(qq, rr);
            }
            let val = (c_pr + c_rp) + dd;
            h[ia * n + ib] = val;
            h[ib * n + ia] = val;
        }
    }
    h
}

/// Options of [`frozen_orbital_step`]. Every norm is the pair norm `√(Σ_{p>q} a[p,q]²)` (full
/// Frobenius over `√2`).
#[derive(Clone, Copy, Debug)]
pub struct FrozenOrbOpts {
    /// Maximum number of accepted steps.
    pub max_iter: usize,
    /// Stop when the body-frame gradient norm is at most this (checked at entry and after every
    /// accepted step). Finite, `≥ 0`.
    pub grad_tol: f64,
    /// Under [`FrozenOrbMethod::Lbfgs`], the per-iteration cap on the trial step:
    /// `α‖κ‖ ≤ max_step`. Under [`FrozenOrbMethod::Newton`], the initial trust radius
    /// `Δ₀ = min(max_step, radius_max)`. Finite, `> 0`.
    pub max_step: f64,
    /// Number of stored L-BFGS curvature pairs; `0` is steepest descent. L-BFGS only (ignored
    /// under Newton).
    pub memory: usize,
    /// Armijo sufficient-decrease constant, in `(0, 1)`. L-BFGS only (ignored under Newton, but
    /// validated under both).
    pub armijo_c1: f64,
    /// Step halvings after the first trial of an iteration before giving up
    /// ([`FrozenOrbExit::LineSearch`]); at most `max_backtracks + 1` trials per iteration.
    /// L-BFGS only (ignored under Newton).
    pub max_backtracks: usize,
    /// The step rule; default [`FrozenOrbMethod::Lbfgs`].
    pub method: FrozenOrbMethod,
}

/// The step rule of [`frozen_orbital_step`] (algorithms in the module header).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum FrozenOrbMethod {
    /// The monotone, step-capped L-BFGS (`memory`, `armijo_c1`, `max_backtracks`, `max_step` as
    /// the per-iteration cap).
    #[default]
    Lbfgs,
    /// Exact-Hessian trust-region Newton ([`frozen_orbital_hessian`] at every iterate, the
    /// trust-region subproblem solved exactly in its eigenbasis); `max_step` is the initial radius.
    Newton(FrozenTrOpts),
}

/// Trust-region settings of [`FrozenOrbMethod::Newton`]. Every radius is a pair norm.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrozenTrOpts {
    /// Upper bound on the trust radius: growth is `Δ ← min(2Δ, radius_max)`, and the initial radius
    /// is `min(max_step, radius_max)`. Finite, `> 0`.
    pub radius_max: f64,
    /// A trial is accepted iff `ρ = (E_k − E_trial)/pred ≥ eta` AND its energy is strictly lower.
    /// In `[0, 0.25)`.
    pub eta: f64,
    /// The iteration gives up ([`FrozenOrbExit::LineSearch`]) once the radius falls below this.
    /// Finite, `> 0`.
    pub min_radius: f64,
}

impl Default for FrozenTrOpts {
    fn default() -> Self {
        FrozenTrOpts {
            radius_max: 1.0,
            eta: 1e-4,
            min_radius: 1e-10,
        }
    }
}

impl Default for FrozenOrbOpts {
    fn default() -> Self {
        FrozenOrbOpts {
            max_iter: 30,
            grad_tol: 1e-7,
            max_step: 0.2,
            memory: 8,
            armijo_c1: 1e-4,
            max_backtracks: 20,
            method: FrozenOrbMethod::Lbfgs,
        }
    }
}

/// Why [`frozen_orbital_step`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrozenOrbExit {
    /// `‖g‖ ≤ grad_tol` after at least one accepted step.
    GradTol,
    /// `max_iter` accepted steps.
    MaxIter,
    /// No trial of an iteration was accepted: under L-BFGS, `max_backtracks` halvings were
    /// exhausted; under Newton, the trust radius fell below `min_radius`, or the model's predicted
    /// decrease reached the energy's resolution floor and the one trial taken there did not lower
    /// `E`, or the Hessian was not finite. The last accepted iterate is returned.
    LineSearch,
    /// `‖g‖ ≤ grad_tol` at entry: no step, `U` is the identity bitwise.
    AlreadyCritical,
}

/// What [`frozen_orbital_step`] did.
#[derive(Clone, Copy, Debug)]
pub struct FrozenOrbInfo {
    /// `E(I) = energy_from_rdms(ints, rdm)`, bitwise.
    pub e_before: f64,
    /// `E(U) = energy_from_rdms(&rotate_integrals(ints, &u), rdm)`, bitwise; `≤ e_before`,
    /// strictly when `iters ≥ 1`.
    pub e_after: f64,
    /// Pair norm of the gradient at `I`.
    pub g_before: f64,
    /// Pair norm of the body-frame gradient at the returned `U`.
    pub g_after: f64,
    /// Accepted steps.
    pub iters: usize,
    /// Energy evaluations, including `e_before` (so `1 +` the number of trials).
    pub n_evals: usize,
    /// Path length of the accepted steps (pair norm): `Σ_k α_k‖κ_k‖` under L-BFGS, `Σ_k ‖k_k‖`
    /// under Newton; an upper bound on the geodesic distance of `U` from `I`.
    pub kappa_norm: f64,
    /// Why the optimizer stopped.
    pub exit: FrozenOrbExit,
    /// Hessians built and eigendecomposed (Newton only; `0` under L-BFGS and at
    /// `AlreadyCritical`). Equal to `iters` at `GradTol`/`MaxIter`, `iters` or `iters + 1` at
    /// `LineSearch`.
    pub n_hessians: usize,
    /// The smallest Hessian eigenvalue over EVERY Hessian built in the call (so `≤ λ_min` at the
    /// start); NaN exactly when `n_hessians == 0`. Not a stability certificate of the returned
    /// `U`: at a `GradTol` exit the last Hessian was built at the previous iterate.
    pub min_curvature: f64,
}

fn grad_half(y: &[f64], pairs: &[(usize, usize)], d: usize) -> Vec<f64> {
    pairs
        .iter()
        .map(|&(p, q)| y[q * d + p] - y[p * d + q])
        .collect()
}

fn full_antisym(h: &[f64], pairs: &[(usize, usize)], d: usize) -> Vec<f64> {
    let mut k = vec![0.0f64; d * d];
    for (&(p, q), &val) in pairs.iter().zip(h) {
        k[p * d + q] = val;
        k[q * d + p] = -val;
    }
    k
}

fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn polar_cleanup(u: &[f64], d: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0f64;
            for k in 0..d {
                s += u[k * d + i] * u[k * d + j];
            }
            c[i * d + j] = if i == j { 3.0 - s } else { -s };
        }
    }
    let mut out = vec![0.0f64; d * d];
    for i in 0..d {
        for k in 0..d {
            let uik = 0.5 * u[i * d + k];
            for j in 0..d {
                out[i * d + j] += uik * c[k * d + j];
            }
        }
    }
    out
}

/// One frozen-RDM global orbital step: a monotone, right-trivialized minimization of
/// `E(U) = energy_from_rdms(&rotate_integrals(ints, U), rdm)` on SO(d), by step-capped L-BFGS
/// (default) or trust-region Newton (`opts.method`; both algorithms, and the L-BFGS registered
/// no-transport approximation, are in the module header).
///
/// # Panics
///
/// When `ints.norb ≠ rdm.d` or on inconsistent tensor shapes; when `max_step` is not finite and
/// positive, `armijo_c1` not in `(0, 1)`, or `grad_tol` not finite and non-negative (under either
/// method); under `Newton`, when `radius_max` or `min_radius` is not finite and positive, or `eta`
/// not in `[0, 0.25)`. Every refusal fires before any work, including at a critical point.
pub fn frozen_orbital_step(
    ints: &Integrals<f64>,
    rdm: &Rdm12,
    opts: &FrozenOrbOpts,
) -> (Vec<f64>, FrozenOrbInfo) {
    check_ints_rdm(ints, rdm, "frozen_orbital_step");
    assert!(
        opts.max_step.is_finite() && opts.max_step > 0.0,
        "frozen_orbital_step: max_step must be finite and positive (got {})",
        opts.max_step
    );
    assert!(
        opts.armijo_c1 > 0.0 && opts.armijo_c1 < 1.0,
        "frozen_orbital_step: armijo_c1 must lie in (0, 1) (got {})",
        opts.armijo_c1
    );
    assert!(
        opts.grad_tol.is_finite() && opts.grad_tol >= 0.0,
        "frozen_orbital_step: grad_tol must be finite and non-negative (got {})",
        opts.grad_tol
    );
    let tr = match opts.method {
        FrozenOrbMethod::Lbfgs => None,
        FrozenOrbMethod::Newton(tr) => {
            assert!(
                tr.radius_max.is_finite() && tr.radius_max > 0.0,
                "frozen_orbital_step: radius_max must be finite and positive (got {})",
                tr.radius_max
            );
            assert!(
                tr.eta >= 0.0 && tr.eta < 0.25,
                "frozen_orbital_step: eta must lie in [0, 0.25) (got {})",
                tr.eta
            );
            assert!(
                tr.min_radius.is_finite() && tr.min_radius > 0.0,
                "frozen_orbital_step: min_radius must be finite and positive (got {})",
                tr.min_radius
            );
            Some(tr)
        }
    };

    let d = rdm.d;
    let pairs = PairMask::all(d).keep;
    let p = form_p(rdm);
    let gamma = &rdm.gamma1.data;

    let e_before = energy_tvp(&ints.t.data, &ints.v.data, gamma, &p);
    let mut n_evals = 1usize;
    let y0 = fock_tvp(&ints.t.data, &ints.v.data, gamma, &p, d);
    let mut g_k = grad_half(&y0, &pairs, d);
    let g_before = norm2(&g_k);
    if g_before <= opts.grad_tol {
        return (
            identity_matrix(d),
            FrozenOrbInfo {
                e_before,
                e_after: e_before,
                g_before,
                g_after: g_before,
                iters: 0,
                n_evals,
                kappa_norm: 0.0,
                exit: FrozenOrbExit::AlreadyCritical,
                n_hessians: 0,
                min_curvature: f64::NAN,
            },
        );
    }
    if let Some(tr) = tr {
        let start = NewtonStart {
            ints,
            gamma,
            p: &p,
            pairs: &pairs,
            d,
            e_before,
            g_before,
            g0: g_k,
            y0,
        };
        return frozen_newton(start, opts, &tr);
    }
    drop(y0);

    let mut mem: VecDeque<(Vec<f64>, Vec<f64>, f64)> = VecDeque::with_capacity(opts.memory);
    let mut u_k = identity_matrix(d);
    let mut e_k = e_before;
    let mut g_norm = g_before;
    let mut iters = 0usize;
    let mut path = 0.0f64;
    let exit = loop {
        if iters == opts.max_iter {
            break FrozenOrbExit::MaxIter;
        }

        let mut q = g_k.clone();
        let mut alphas = vec![0.0f64; mem.len()];
        for (idx, (s, y, rho)) in mem.iter().enumerate().rev() {
            let a = rho * dot(s, &q);
            alphas[idx] = a;
            for (qi, yi) in q.iter_mut().zip(y) {
                *qi -= a * yi;
            }
        }
        let h0 = mem.back().map_or(1.0, |(s, y, _)| dot(s, y) / dot(y, y));
        let mut r: Vec<f64> = q.iter().map(|qi| h0 * qi).collect();
        for (idx, (s, y, rho)) in mem.iter().enumerate() {
            let b = rho * dot(y, &r);
            let c = alphas[idx] - b;
            for (ri, si) in r.iter_mut().zip(s) {
                *ri += c * si;
            }
        }
        let mut kappa: Vec<f64> = r.iter().map(|ri| -ri).collect();

        let mut slope = dot(&g_k, &kappa);
        if slope >= 0.0 || slope.is_nan() {
            mem.clear();
            kappa = g_k.iter().map(|gi| -gi).collect();
            slope = -dot(&g_k, &g_k);
        }
        let kappa_nrm = norm2(&kappa);
        if !(kappa_nrm.is_finite() && kappa_nrm > 0.0) {
            break FrozenOrbExit::LineSearch;
        }

        let mut alpha = (opts.max_step / kappa_nrm).min(1.0);
        let mut accepted: Option<(Vec<f64>, Integrals<f64>, f64)> = None;
        for bt in 0..=opts.max_backtracks {
            if bt > 0 {
                alpha *= 0.5;
            }
            let step: Vec<f64> = kappa.iter().map(|ki| -(alpha * ki)).collect();
            let w = expm_antisym(&full_antisym(&step, &pairs, d), d);
            let u_t = polar_cleanup(&compose_rotations(&u_k, &w, d), d);
            let ints_t = rotate_integrals(ints, &u_t);
            let e_t = energy_tvp(&ints_t.t.data, &ints_t.v.data, gamma, &p);
            n_evals += 1;
            if e_t < e_k && e_t <= e_k + opts.armijo_c1 * alpha * slope {
                accepted = Some((u_t, ints_t, e_t));
                break;
            }
        }
        let Some((u_t, ints_t, e_t)) = accepted else {
            break FrozenOrbExit::LineSearch;
        };

        let g_new = grad_half(
            &fock_tvp(&ints_t.t.data, &ints_t.v.data, gamma, &p, d),
            &pairs,
            d,
        );
        let s: Vec<f64> = kappa.iter().map(|ki| alpha * ki).collect();
        let y: Vec<f64> = g_new.iter().zip(&g_k).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if opts.memory > 0 && sy > 1e-12 * norm2(&s) * norm2(&y) {
            mem.push_back((s, y, 1.0 / sy));
            while mem.len() > opts.memory {
                mem.pop_front();
            }
        }
        path += alpha * kappa_nrm;
        u_k = u_t;
        e_k = e_t;
        g_k = g_new;
        g_norm = norm2(&g_k);
        iters += 1;

        if g_norm <= opts.grad_tol {
            break FrozenOrbExit::GradTol;
        }
    };

    (
        u_k,
        FrozenOrbInfo {
            e_before,
            e_after: e_k,
            g_before,
            g_after: g_norm,
            iters,
            n_evals,
            kappa_norm: path,
            exit,
            n_hessians: 0,
            min_curvature: f64::NAN,
        },
    )
}

struct NewtonStart<'a> {
    ints: &'a Integrals<f64>,
    gamma: &'a [f64],
    p: &'a [f64],
    pairs: &'a [(usize, usize)],
    d: usize,
    e_before: f64,
    g_before: f64,
    g0: Vec<f64>,
    y0: Vec<f64>,
}

const ORB_ENERGY_NOISE_C: f64 = 32.0;

fn frozen_newton(
    s: NewtonStart<'_>,
    opts: &FrozenOrbOpts,
    tr: &FrozenTrOpts,
) -> (Vec<f64>, FrozenOrbInfo) {
    let NewtonStart {
        ints,
        gamma,
        p,
        pairs,
        d,
        e_before,
        g_before,
        g0,
        y0,
    } = s;
    let n = pairs.len();
    let mut u_k = identity_matrix(d);
    let mut e_k = e_before;
    let mut g_k = g0;
    let mut y_k = y0;
    let mut ints_k: Option<Integrals<f64>> = None;
    let mut delta = opts.max_step.min(tr.radius_max);
    let mut g_norm = g_before;
    let (mut iters, mut n_evals, mut n_hessians) = (0usize, 1usize, 0usize);
    let mut path = 0.0f64;
    let mut min_curv = f64::NAN;
    let exit = 'outer: loop {
        if iters == opts.max_iter {
            break FrozenOrbExit::MaxIter;
        }
        if delta < tr.min_radius {
            break FrozenOrbExit::LineSearch;
        }

        let (lam, vecs) = {
            let cur = ints_k.as_ref().unwrap_or(ints);
            let h = hessian_tvp(&cur.t.data, &cur.v.data, gamma, p, &y_k, d, pairs);
            if !h.iter().all(|x| x.is_finite()) {
                break FrozenOrbExit::LineSearch;
            }
            dense_self_adjoint_smallest::<f64>(&h, n, n)
        };
        n_hessians += 1;
        min_curv = min_curv.min(lam[0]);
        let ghat: Vec<f64> = vecs.iter().map(|q| dot(q, &g_k)).collect();
        let floor = ORB_ENERGY_NOISE_C * f64::EPSILON * (1.0 + e_k.abs());

        let (u_t, ints_t, e_t, knorm) = loop {
            let st = tr_solve_eig(&lam, &ghat, delta);
            let at_floor = st.pred.is_nan() || st.pred <= floor;
            if at_floor && (st.pred.is_nan() || st.pred <= 0.0) {
                break 'outer FrozenOrbExit::LineSearch;
            }
            let mut k = vec![0.0f64; n];
            for (c, q) in vecs.iter().enumerate() {
                let a = st.khat[c];
                for (kr, qr) in k.iter_mut().zip(q) {
                    *kr += a * qr;
                }
            }
            let knorm = norm2(&k);
            let step: Vec<f64> = k.iter().map(|x| -x).collect();
            let w = expm_antisym(&full_antisym(&step, pairs, d), d);
            let u_t = polar_cleanup(&compose_rotations(&u_k, &w, d), d);
            let ints_t = rotate_integrals(ints, &u_t);
            let e_t = energy_tvp(&ints_t.t.data, &ints_t.v.data, gamma, p);
            n_evals += 1;
            if at_floor {
                if e_t < e_k {
                    break (u_t, ints_t, e_t, knorm);
                }
                break 'outer FrozenOrbExit::LineSearch;
            }
            let rho = (e_k - e_t) / st.pred;
            let accept = rho >= tr.eta && e_t < e_k;
            if !accept || rho < 0.25 {
                delta = 0.25 * knorm;
            } else if rho > 0.75 && st.boundary {
                delta = (2.0 * delta).min(tr.radius_max);
            }
            if accept {
                break (u_t, ints_t, e_t, knorm);
            }
            if delta < tr.min_radius {
                break 'outer FrozenOrbExit::LineSearch;
            }
        };

        y_k = fock_tvp(&ints_t.t.data, &ints_t.v.data, gamma, p, d);
        g_k = grad_half(&y_k, pairs, d);
        path += knorm;
        u_k = u_t;
        e_k = e_t;
        ints_k = Some(ints_t);
        g_norm = norm2(&g_k);
        iters += 1;
        if g_norm <= opts.grad_tol {
            break FrozenOrbExit::GradTol;
        }
    };

    (
        u_k,
        FrozenOrbInfo {
            e_before,
            e_after: e_k,
            g_before,
            g_after: g_norm,
            iters,
            n_evals,
            kappa_norm: path,
            exit,
            n_hessians,
            min_curvature: min_curv,
        },
    )
}

#[derive(Clone, Debug)]
#[cfg_attr(not(test), allow(dead_code))]
struct TrStep {
    khat: Vec<f64>,
    mu: f64,
    pred: f64,
    boundary: bool,
    hard: bool,
}

const TR_CLUSTER_REL: f64 = 1e-12;
const TR_HARD_REL: f64 = 1e-12;
const TR_SECULAR_TOL: f64 = 1e-12;
const TR_SECULAR_MAX_IT: usize = 200;

fn tr_solve_eig(lam: &[f64], ghat: &[f64], delta: f64) -> TrStep {
    let n = lam.len();
    let finish = |khat: Vec<f64>, mu: f64, boundary: bool, hard: bool| {
        let mut m = 0.0f64;
        for i in 0..n {
            m += ghat[i] * khat[i] + 0.5 * lam[i] * khat[i] * khat[i];
        }
        TrStep {
            khat,
            mu,
            pred: -m,
            boundary,
            hard,
        }
    };
    let gn = norm2(ghat);
    let l0 = lam[0];
    if l0 > 0.0 {
        let k: Vec<f64> = (0..n).map(|i| -ghat[i] / lam[i]).collect();
        if norm2(&k) <= delta {
            return finish(k, 0.0, false, false);
        }
    } else {
        let width = TR_CLUSTER_REL * l0.abs().max(lam[n - 1].abs());
        let nj = lam.iter().take_while(|&&l| l - l0 <= width).count();
        if norm2(&ghat[..nj]) <= TR_HARD_REL * gn {
            let mu = if l0 < 0.0 { -l0 } else { 0.0 };
            let mut k = vec![0.0f64; n];
            for j in nj..n {
                k[j] = -ghat[j] / (lam[j] + mu);
            }
            let kr = norm2(&k);
            if kr <= delta {
                if l0 < 0.0 {
                    let tau = (delta * delta - kr * kr).max(0.0).sqrt();
                    k[0] += if ghat[0] > 0.0 { -tau } else { tau };
                    return finish(k, mu, true, true);
                }
                return finish(k, 0.0, false, false);
            }
        }
    }

    let lo = if l0 < 0.0 { -l0 } else { 0.0 };
    let (mut a, mut b) = (lo, lo + gn / delta);
    let mut mu = if l0 > 0.0 { 0.0 } else { b };
    for _ in 0..TR_SECULAR_MAX_IT {
        let (mut nk2, mut w, mut inside) = (0.0f64, 0.0f64, true);
        for i in 0..n {
            let si = lam[i] + mu;
            let g2 = ghat[i] * ghat[i];
            if si.is_nan() || si <= 0.0 {
                if g2 != 0.0 {
                    inside = false;
                    break;
                }
                continue;
            }
            nk2 += g2 / (si * si);
            w += g2 / (si * si * si);
        }
        let mut cand = f64::NAN;
        if inside {
            let nk = nk2.sqrt();
            if (nk - delta).abs() <= TR_SECULAR_TOL * delta {
                break;
            }
            if nk > delta {
                a = mu;
            } else {
                b = mu;
            }
            cand = mu + (nk2 / w) * (nk - delta) / delta;
        } else {
            a = mu;
        }
        let next = if cand.is_finite() && cand > a && cand < b {
            cand
        } else {
            0.5 * (a + b)
        };
        if next == mu || b - a <= 4.0 * f64::EPSILON * b.abs() {
            break;
        }
        mu = next;
    }
    let mut khat: Vec<f64> = (0..n)
        .map(|i| {
            if ghat[i] == 0.0 {
                0.0
            } else {
                -ghat[i] / (lam[i] + mu)
            }
        })
        .collect();
    let kn = norm2(&khat);
    if kn > delta {
        let sc = delta / kn;
        for x in khat.iter_mut() {
            *x *= sc;
        }
    } else if kn < delta * (1.0 - TR_SECULAR_TOL) {
        let width = TR_CLUSTER_REL * l0.abs().max(lam[n - 1].abs());
        let nj = lam.iter().take_while(|&&l| l - l0 <= width).count();
        let kj = norm2(&khat[..nj]);
        let rest2: f64 = khat[nj..].iter().map(|x| x * x).sum();
        let room2 = delta * delta - rest2;
        if kj > 0.0 && room2 > 0.0 {
            let sc = room2.sqrt() / kj;
            for x in khat[..nj].iter_mut() {
                *x *= sc;
            }
        }
    }
    finish(khat, mu, true, false)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;
    use crate::integrals::Integrals;
    use crate::{CoeffTensor, QnSpace, TtVector, tt_state};

    const D: usize = 3;

    fn ints(nup: usize, ndn: usize, t: CoeffTensor<f64>, v: CoeffTensor<f64>) -> Integrals<f64> {
        let space = QnSpace::new(D, nup, ndn);
        Integrals::<f64> {
            t,
            v,
            e0: 0.0,
            norb: D,
            nelec: space.n(),
            ms2: nup as i64 - ndn as i64,
            orbsym: Vec::new(),
            isym: None,
        }
    }

    fn sym_t() -> CoeffTensor<f64> {
        let mut t = CoeffTensor::<f64>::zeros(D, 2);
        let vals = [[-1.1, 0.23, -0.31], [0.23, -0.4, 0.17], [-0.31, 0.17, 0.9]];
        for i in 0..D {
            for j in 0..D {
                t.set(&[i, j], vals[i][j]);
            }
        }
        t
    }

    fn sym8_v() -> CoeffTensor<f64> {
        let s = |p: usize, q: usize| 0.3 + 0.1 * (p + q) as f64 + 0.07 * (p * q) as f64;
        let mut v = CoeffTensor::<f64>::zeros(D, 4);
        for i in 0..D {
            for j in 0..D {
                for k in 0..D {
                    for l in 0..D {
                        v.set(
                            &[i, j, k, l],
                            s(i, j) * s(k, l) + 0.25 * (s(i, j) + s(k, l)),
                        );
                    }
                }
            }
        }
        v
    }

    fn assert_y(y: &[f64], want: &[[f64; D]; D], what: &str) {
        assert_eq!(y.len(), D * D, "{what}: Y is d×d");
        for i in 0..D {
            for a in 0..D {
                assert!(
                    (y[i * D + a] - want[i][a]).abs() <= 1e-14,
                    "{what}: Y[{i},{a}] = {} ≠ {}",
                    y[i * D + a],
                    want[i][a]
                );
            }
        }
    }

    fn assert_g_is_yt_minus_y(ints: &Integrals<f64>, rdm: &Rdm12, y: &[f64], what: &str) {
        let g = frozen_orbital_gradient(ints, rdm);
        for p in 0..D {
            assert!(g.get(&[p, p]) == 0.0, "{what}: g[{p},{p}] ≠ 0");
            for q in 0..D {
                assert!(
                    g.get(&[p, q]) == -g.get(&[q, p]),
                    "{what}: g not exactly antisymmetric at ({p},{q})"
                );
                if p != q {
                    let want = y[q * D + p] - y[p * D + q];
                    assert!(
                        (g.get(&[p, q]) - want).abs() <= 1e-14,
                        "{what}: g[{p},{q}] = {} ≠ Y[{q},{p}] − Y[{p},{q}] = {want}",
                        g.get(&[p, q])
                    );
                }
            }
        }
    }

    #[test]
    fn orbopt_y_sign_one_electron() {
        let (e0, e1, e2) = (-0.7, 0.4, 1.3);
        let mut t = CoeffTensor::<f64>::zeros(D, 2);
        t.set(&[0, 0], e0);
        t.set(&[1, 1], e1);
        t.set(&[2, 2], e2);
        let ints = ints(1, 0, t, CoeffTensor::<f64>::zeros(D, 4));
        let none = [false; D];
        let a: TtVector<f64> = tt_state(&[true, false, false], &none);
        let b: TtVector<f64> = tt_state(&[false, true, false], &none);
        let mut x = &a + &b;
        x.scale_mut(std::f64::consts::FRAC_1_SQRT_2);
        let rdm = Rdm12::of(&x);
        let half = [[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 0.0]];
        for p in 0..D {
            for q in 0..D {
                assert!(
                    (rdm.gamma1.get(&[p, q]) - half[p][q]).abs() <= 1e-14,
                    "(A) γ[{p},{q}]"
                );
            }
        }
        let e = energy_from_rdms(&ints, &rdm);
        assert!((e - 0.5 * (e0 + e1)).abs() <= 1e-14, "(A) E = {e}");
        let y = generalized_fock(&ints, &rdm);
        assert_y(&y, &[[e0, e0, 0.0], [e1, e1, 0.0], [0.0, 0.0, 0.0]], "(A)");
        assert_g_is_yt_minus_y(&ints, &rdm, &y, "(A)");
        let g = frozen_orbital_gradient(&ints, &rdm);
        assert!(
            (g.get(&[1, 0]) - (e0 - e1)).abs() <= 1e-14,
            "(A) g[1,0] = ε₀ − ε₁"
        );
    }

    #[test]
    fn orbopt_y_closed_shell() {
        let (t, v) = (sym_t(), sym8_v());
        let want_e = 2.0 * t.get(&[0, 0]) + v.get(&[0, 0, 0, 0]);
        let mut want = [[0.0; D]; D];
        for i in 0..D {
            want[i][0] = 4.0 * (t.get(&[i, 0]) + v.get(&[i, 0, 0, 0]));
        }
        let ints = ints(1, 1, t, v);
        let x: TtVector<f64> = tt_state(&[true, false, false], &[true, false, false]);
        let rdm = Rdm12::of(&x);
        let e = energy_from_rdms(&ints, &rdm);
        assert!((e - want_e).abs() <= 1e-14, "(B) E = {e} vs {want_e}");
        let y = generalized_fock(&ints, &rdm);
        assert_y(&y, &want, "(B)");
        assert_g_is_yt_minus_y(&ints, &rdm, &y, "(B)");
    }

    #[test]
    fn orbopt_y_spinless_exchange() {
        let (t, v) = (sym_t(), sym8_v());
        let occ = [0usize, 1];
        let want_e = t.get(&[0, 0]) + t.get(&[1, 1]) + v.get(&[0, 0, 1, 1]) - v.get(&[0, 1, 1, 0]);
        let mut want = [[0.0; D]; D];
        for i in 0..D {
            for &a in &occ {
                let mut f = t.get(&[i, a]);
                for &j in &occ {
                    f += v.get(&[i, a, j, j]) - v.get(&[i, j, j, a]);
                }
                want[i][a] = 2.0 * f;
            }
        }
        assert!(
            (v.get(&[0, 1, 1, 0]) - v.get(&[0, 0, 1, 1])).abs() > 1e-2,
            "(C) exchange must differ from Coulomb, or the δ term is invisible"
        );
        let ints = ints(2, 0, t, v);
        let x: TtVector<f64> = tt_state(&[true, true, false], &[false, false, false]);
        let rdm = Rdm12::of(&x);
        let e = energy_from_rdms(&ints, &rdm);
        assert!((e - want_e).abs() <= 1e-14, "(C) E = {e} vs {want_e}");
        let y = generalized_fock(&ints, &rdm);
        assert_y(&y, &want, "(C)");
        assert_g_is_yt_minus_y(&ints, &rdm, &y, "(C)");
    }

    #[test]
    fn orbopt_y_slot_isolation() {
        let mut t = CoeffTensor::<f64>::zeros(D, 2);
        for (i, j, val) in [
            (0, 0, -1.0),
            (1, 1, -0.5),
            (0, 2, 0.3),
            (2, 0, 0.3),
            (1, 2, 0.7),
            (2, 1, -0.2),
        ] {
            t.set(&[i, j], val);
        }
        let mut v = CoeffTensor::<f64>::zeros(D, 4);
        for (idx, val) in [
            ([2, 1, 1, 0], 0.5),
            ([1, 2, 0, 0], 0.7),
            ([1, 0, 2, 1], 1.1),
            ([0, 0, 1, 2], 1.3),
            ([0, 0, 0, 0], 0.9),
        ] {
            v.set(&idx, val);
        }
        let ints = ints(2, 1, t, v);
        let x: TtVector<f64> = tt_state(&[true, true, false], &[true, false, false]);
        let rdm = Rdm12::of(&x);
        assert_eq!((rdm.d, rdm.n), (3, 3));
        let want_g1 = [2.0, 1.0, 0.0];
        for p in 0..D {
            for q in 0..D {
                let w = if p == q { want_g1[p] } else { 0.0 };
                assert!(
                    (rdm.gamma1.get(&[p, q]) - w).abs() <= 1e-14,
                    "(D) γ[{p},{q}]"
                );
            }
        }
        let idx = |i: usize, j: usize, k: usize, l: usize| ((i * D + j) * D + k) * D + l;
        let mut want_g2 = vec![0.0f64; D * D * D * D];
        for (i, j, k, l, val) in [
            (0, 0, 0, 0, 4.0),
            (0, 0, 1, 1, 2.0),
            (1, 1, 0, 0, 2.0),
            (1, 1, 1, 1, 1.0),
            (0, 1, 1, 0, 1.0),
            (0, 2, 2, 0, 2.0),
            (1, 2, 2, 1, 1.0),
        ] {
            want_g2[idx(i, j, k, l)] = val;
        }
        assert_eq!(rdm.gamma2.len(), want_g2.len());
        for (f, (got, want)) in rdm.gamma2.iter().zip(&want_g2).enumerate() {
            assert!(
                (got - want).abs() <= 1e-14,
                "(D) Γ₂ flat {f}: {got} vs {want}"
            );
        }
        let e = energy_from_rdms(&ints, &rdm);
        assert!((e - (-1.6)).abs() <= 1e-14, "(D) E = {e}");
        let y = generalized_fock(&ints, &rdm);
        assert_y(
            &y,
            &[[-0.4, 0.0, 0.0], [0.0, -1.0, 0.0], [0.4, 2.5, 0.0]],
            "(D)",
        );
        assert_g_is_yt_minus_y(&ints, &rdm, &y, "(D)");
    }

    #[test]
    fn orbopt_fock_threaded_is_bitwise_serial() {
        fn dot_ref(a: &[f64], b: &[f64]) -> f64 {
            let mut s = 0.0f64;
            for (x, y) in a.iter().zip(b) {
                s += x * y;
            }
            s
        }
        fn serial(t: &[f64], v: &[f64], gamma: &[f64], p: &[f64], d: usize) -> Vec<f64> {
            let d2 = d * d;
            let d3 = d2 * d;
            let mut y = vec![0.0f64; d2];
            for i in 0..d {
                for a in 0..d {
                    let mut s = 0.0f64;
                    for b in 0..d {
                        s += t[i * d + b] * gamma[a * d + b];
                    }
                    let mut s2 = 0.0f64;
                    for b in 0..d {
                        s2 += t[b * d + i] * gamma[b * d + a];
                    }
                    y[i * d + a] = s + s2;
                }
            }
            let mut s1 = vec![0.0f64; d2];
            for i in 0..d {
                let vi = &v[i * d3..(i + 1) * d3];
                for a in 0..d {
                    s1[i * d + a] = dot_ref(vi, &p[a * d3..(a + 1) * d3]);
                }
            }
            let mut s2 = vec![0.0f64; d2];
            for b in 0..d {
                for i in 0..d {
                    let off_v = (b * d + i) * d2;
                    let vbi = &v[off_v..off_v + d2];
                    for a in 0..d {
                        let off_p = (b * d + a) * d2;
                        s2[i * d + a] += dot_ref(vbi, &p[off_p..off_p + d2]);
                    }
                }
            }
            let mut s3 = vec![0.0f64; d2];
            for w in 0..d2 {
                for i in 0..d {
                    let off_v = (w * d + i) * d;
                    let vwi = &v[off_v..off_v + d];
                    for a in 0..d {
                        let off_p = (w * d + a) * d;
                        s3[i * d + a] += dot_ref(vwi, &p[off_p..off_p + d]);
                    }
                }
            }
            let mut s4 = vec![0.0f64; d2];
            for w in 0..d3 {
                let vw = &v[w * d..(w + 1) * d];
                let pw = &p[w * d..(w + 1) * d];
                for (i, &vi) in vw.iter().enumerate() {
                    let row = &mut s4[i * d..(i + 1) * d];
                    for (acc, &pa) in row.iter_mut().zip(pw) {
                        *acc += vi * pa;
                    }
                }
            }
            for (f, yf) in y.iter_mut().enumerate() {
                *yf += 0.5 * (((s1[f] + s2[f]) + s3[f]) + s4[f]);
            }
            y
        }
        struct Fx {
            t: Vec<f64>,
            v: Vec<f64>,
            g: Vec<f64>,
            p: Vec<f64>,
            trap: usize,
        }
        fn fixture(d: usize, seed: u64) -> Fx {
            let (i0, a0) = (0usize, d - 1);
            let mut state = seed;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            };
            let digits = |mut k: usize, n: usize| {
                let mut q = [0usize; 4];
                for slot in q[..n].iter_mut().rev() {
                    *slot = k % d;
                    k /= d;
                }
                q
            };
            let neg_or = |has: bool, next: &mut dyn FnMut() -> f64| {
                let r = next();
                if has { -(r.abs() + 0.01) } else { r }
            };
            let t: Vec<f64> = (0..d * d)
                .map(|k| neg_or(digits(k, 2)[..2].contains(&i0), &mut next))
                .collect();
            let v: Vec<f64> = (0..d.pow(4))
                .map(|k| neg_or(digits(k, 4).contains(&i0), &mut next))
                .collect();
            let g: Vec<f64> = (0..d * d)
                .map(|k| {
                    if digits(k, 2)[..2].contains(&a0) {
                        0.0
                    } else {
                        next()
                    }
                })
                .collect();
            let p: Vec<f64> = (0..d.pow(4))
                .map(|k| {
                    if digits(k, 4).contains(&a0) {
                        0.0
                    } else {
                        next()
                    }
                })
                .collect();
            Fx {
                t,
                v,
                g,
                p,
                trap: i0 * d + a0,
            }
        }
        fn bits_eq(got: &[f64], want: &[f64], what: &str) {
            assert_eq!(got.len(), want.len(), "{what}: Y length");
            for (f, (a, b)) in got.iter().zip(want).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{what}: Y[{f}] = {a:e} is not bitwise the serial {b:e}"
                );
            }
        }
        let pool = |n: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("rayon pool")
        };
        struct Restore {
            fork: usize,
            on: bool,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                crate::par::set_par_min_fork_work(self.fork);
                crate::par::set_vector_par(self.on);
            }
        }

        let _lock = crate::par::lock_par_switches();
        let _restore = Restore {
            fork: crate::par::par_min_fork_work(),
            on: crate::par::set_vector_par(true),
        };
        let cases: Vec<(usize, Fx, Vec<f64>)> = [
            (3usize, 0x9E37_79B9_7F4A_7C15u64),
            (8, 0xF0C4_F1F1_D00D_5EEDu64),
        ]
        .into_iter()
        .map(|(d, seed)| {
            let fx = fixture(d, seed);
            let want = serial(&fx.t, &fx.v, &fx.g, &fx.p, d);
            assert_eq!(
                want[fx.trap].to_bits(),
                0.0f64.to_bits(),
                "d = {d}: the signed-zero trap is not armed (serial Y[trap] must be +0.0)"
            );
            (d, fx, want)
        })
        .collect();

        for (d, fx, want) in &cases {
            let d = *d;
            let run = || fock_tvp(&fx.t, &fx.v, &fx.g, &fx.p, d);
            bits_eq(
                &pool(1).install(run),
                want,
                &format!("d = {d}, pool 1, shipped floor"),
            );
            let forked = pool(4).install(|| crate::par::par_driver_fork(fock_work(d)));
            eprintln!("orbopt_fock: d = {d}, pool 4, shipped floor: parallel branch = {forked}");
            bits_eq(
                &pool(4).install(run),
                want,
                &format!("d = {d}, pool 4, shipped floor"),
            );
            bits_eq(
                &pool(4).install(|| crate::par::nested_scope(run)),
                want,
                &format!("d = {d}, pool 4, nested scope"),
            );
        }

        crate::par::set_par_min_fork_work(0);
        for (d, fx, want) in &cases {
            let d = *d;
            let got = pool(4).install(|| {
                assert!(
                    crate::par::par_driver_fork(fock_work(d)),
                    "d = {d}: the forced-parallel arm would run the serial branch (vacuous)"
                );
                fock_tvp(&fx.t, &fx.v, &fx.g, &fx.p, d)
            });
            bits_eq(&got, want, &format!("d = {d}, pool 4, floor 0"));
            let blocks: &[usize] = if d == 8 {
                &[FOCK_A_BLOCK, 1, 3, 8]
            } else {
                &[FOCK_A_BLOCK, 1, 2, 3]
            };
            for &ab in blocks {
                let got = pool(4).install(|| fock_tvp_par(&fx.t, &fx.v, &fx.g, &fx.p, d, ab));
                bits_eq(
                    &got,
                    want,
                    &format!("d = {d}, fock_tvp_par ab = {ab}, pool 4"),
                );
            }
            let got = pool(1).install(|| fock_tvp_par(&fx.t, &fx.v, &fx.g, &fx.p, d, FOCK_A_BLOCK));
            bits_eq(
                &got,
                want,
                &format!("d = {d}, fock_tvp_par production ab, pool 1"),
            );
        }
    }

    fn ints_d(
        d: usize,
        nup: usize,
        ndn: usize,
        t: CoeffTensor<f64>,
        v: CoeffTensor<f64>,
    ) -> Integrals<f64> {
        let space = QnSpace::new(d, nup, ndn);
        Integrals::<f64> {
            t,
            v,
            e0: 0.0,
            norb: d,
            nelec: space.n(),
            ms2: nup as i64 - ndn as i64,
            orbsym: Vec::new(),
            isym: None,
        }
    }

    fn generic_asym(d: usize, rank: usize, phase: f64) -> CoeffTensor<f64> {
        let mut c = CoeffTensor::<f64>::zeros(d, rank);
        for (f, e) in c.data.iter_mut().enumerate() {
            *e = (1.7 * f as f64 + phase).sin() * (1.0 + 0.1 * (f % 7) as f64);
        }
        c
    }

    fn assert_h3(h: &[f64], want: &[[f64; 3]; 3], what: &str) {
        assert_eq!(h.len(), 9, "{what}: n_pairs = 3 at d = 3");
        for a in 0..3 {
            for b in 0..3 {
                assert!(
                    (h[a * 3 + b] - want[a][b]).abs() <= 1e-14,
                    "{what}: H[{a},{b}] = {} ≠ {}",
                    h[a * 3 + b],
                    want[a][b]
                );
                assert!(
                    h[a * 3 + b].to_bits() == h[b * 3 + a].to_bits(),
                    "{what}: H not exactly symmetric at ({a},{b})"
                );
            }
        }
    }

    #[test]
    fn orbopt_hessian_one_electron_hand() {
        let x: TtVector<f64> = tt_state(&[true, false, false], &[false; D]);
        let rdm = Rdm12::of(&x);

        let (t00, t11, t12, t22) = (-0.9, 0.35, 0.21, 1.4);
        let mut t = CoeffTensor::<f64>::zeros(D, 2);
        t.set(&[0, 0], t00);
        t.set(&[1, 1], t11);
        t.set(&[1, 2], t12);
        t.set(&[2, 1], t12);
        t.set(&[2, 2], t22);
        let ints_a = ints(1, 0, t, generic_asym(D, 4, 0.3));
        let h = frozen_orbital_hessian(&ints_a, &rdm);
        assert_h3(
            &h,
            &[
                [2.0 * (t11 - t00), 2.0 * t12, 0.0],
                [2.0 * t12, 2.0 * (t22 - t00), 0.0],
                [0.0, 0.0, 0.0],
            ],
            "(H1a)",
        );

        let t = generic_asym(D, 2, 0.8);
        let g = |i: usize, j: usize| t.get(&[i, j]);
        for (i, j) in [(0, 1), (0, 2), (1, 0), (2, 0)] {
            assert!(g(i, j).abs() > 0.05, "(H1b) t[{i},{j}] must be non-zero");
        }
        assert!(
            (g(0, 1) - g(1, 0)).abs() > 0.05,
            "(H1b) t must be non-symmetric"
        );
        let want = [
            [
                2.0 * (g(1, 1) - g(0, 0)),
                g(1, 2) + g(2, 1),
                0.5 * (g(2, 0) + g(0, 2)),
            ],
            [
                g(1, 2) + g(2, 1),
                2.0 * (g(2, 2) - g(0, 0)),
                -0.5 * (g(1, 0) + g(0, 1)),
            ],
            [0.5 * (g(2, 0) + g(0, 2)), -0.5 * (g(1, 0) + g(0, 1)), 0.0],
        ];
        let ints_b = ints(1, 0, t.clone(), generic_asym(D, 4, 1.1));
        let h = frozen_orbital_hessian(&ints_b, &rdm);
        assert_h3(&h, &want, "(H1b)");
    }

    #[test]
    fn orbopt_hessian_closed_shell_hand() {
        let t = generic_asym(D, 2, 0.45);
        let v = generic_asym(D, 4, 2.3);
        let tv = |i: usize, j: usize| t.get(&[i, j]);
        let vv = |idx: [usize; 4]| v.get(&idx);
        let at = |s: usize, i: usize| {
            let mut idx = [0usize; 4];
            idx[s] = i;
            vv(idx)
        };
        let at2 = |s: usize, i: usize, s2: usize, j: usize| {
            let mut idx = [0usize; 4];
            idx[s] = i;
            idx[s2] = j;
            vv(idx)
        };
        let y0 = |j: usize| 2.0 * (tv(j, 0) + tv(0, j)) + (0..4).map(|s| at(s, j)).sum::<f64>();
        let a = |p: usize, r: usize| {
            let mut two = 0.0;
            for s in 0..4 {
                for s2 in 0..4 {
                    if s != s2 {
                        two += at2(s, p, s2, r);
                    }
                }
            }
            2.0 * (tv(p, r) + tv(r, p)) + two - if p == r { y0(0) } else { 0.0 }
        };
        assert!(
            (y0(0) - (4.0 * tv(0, 0) + 4.0 * vv([0, 0, 0, 0]))).abs() <= 1e-14,
            "(H2) Y_00 = 4t00 + 4v0000"
        );
        let want = [
            [a(1, 1), a(1, 2), 0.5 * y0(2)],
            [a(2, 1), a(2, 2), -0.5 * y0(1)],
            [0.5 * y0(2), -0.5 * y0(1), 0.0],
        ];
        assert!(
            (a(1, 2) - a(2, 1)).abs() <= 1e-14,
            "(H2) the hand A must be symmetric (it is a Hessian)"
        );
        let ints = ints(1, 1, t.clone(), v.clone());
        let x: TtVector<f64> = tt_state(&[true, false, false], &[true, false, false]);
        let rdm = Rdm12::of(&x);
        let h = frozen_orbital_hessian(&ints, &rdm);
        assert_h3(&h, &want, "(H2)");
        let mut half_two = 0.0;
        for s in 0..4 {
            for s2 in (s + 1)..4 {
                half_two += at2(s, 1, s2, 1);
            }
        }
        assert!(
            half_two.abs() > 1e-2,
            "(H2) the two-body part of A_11 is too small to discriminate"
        );
    }

    #[test]
    fn orbopt_hessian_exactly_symmetric() {
        let d = 4;
        let det = |u: usize, dn: usize| {
            let mut ou = [false; 4];
            let mut od = [false; 4];
            ou[u] = true;
            od[dn] = true;
            tt_state::<f64>(&ou, &od)
        };
        let mut x = &(&det(0, 1) * 0.6) + &(&det(1, 2) * 0.48);
        x = &x + &(&det(2, 3) * 0.64);
        let rdm = Rdm12::of(&x);
        let ints = ints_d(d, 1, 1, generic_asym(d, 2, 0.2), generic_asym(d, 4, 0.9));
        let h = frozen_orbital_hessian(&ints, &rdm);
        let n = d * (d - 1) / 2;
        assert_eq!(h.len(), n * n);
        let mut off = 0.0f64;
        for a in 0..n {
            for b in 0..n {
                assert!(h[a * n + b].is_finite(), "H[{a},{b}] not finite");
                assert!(
                    h[a * n + b].to_bits() == h[b * n + a].to_bits(),
                    "H not exactly symmetric at ({a},{b}): {:e} vs {:e}",
                    h[a * n + b],
                    h[b * n + a]
                );
                if a != b {
                    off = off.max(h[a * n + b].abs());
                }
            }
        }
        assert!(
            off > 1e-2,
            "the fixture's Hessian is (near) diagonal ({off:e}): vacuous"
        );
    }

    #[test]
    fn orbopt_tr_solve_eig_vs_brute_force() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        enum Want {
            Interior,
            Boundary,
            Hard,
            Singular,
            NearHard,
        }
        let model = |lam: &[f64], g: &[f64], k: &[f64]| -> f64 {
            (0..lam.len())
                .map(|i| g[i] * k[i] + 0.5 * lam[i] * k[i] * k[i])
                .sum()
        };
        let nrm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut unif = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        type Case = (Vec<f64>, Vec<f64>, f64, Want, &'static str);
        let cases: Vec<Case> = vec![
            (
                vec![0.5, 1.0, 3.0],
                vec![0.1, -0.2, 0.3],
                10.0,
                Want::Interior,
                "interior",
            ),
            (
                vec![0.5, 1.0, 3.0],
                vec![1.0, -2.0, 3.0],
                0.5,
                Want::Boundary,
                "boundary 3×3",
            ),
            (
                vec![0.2, 0.7, 1.5, 4.0],
                vec![0.3, 0.1, -0.8, 2.0],
                0.3,
                Want::Boundary,
                "boundary 4×4",
            ),
            (
                vec![-1.0, 0.5, 2.0, 4.0],
                vec![0.3, -0.7, 0.2, 1.1],
                0.8,
                Want::Boundary,
                "indefinite 4×4",
            ),
            (
                vec![-2.0, -0.5, 1.0],
                vec![0.05, 0.4, -0.3],
                5.0,
                Want::Boundary,
                "indefinite 3×3",
            ),
            (
                vec![-2.0, 1.0, 3.0, 5.0],
                vec![0.0, 1.0, 0.5, -0.2],
                10.0,
                Want::Hard,
                "hard",
            ),
            (
                vec![-2.0, 1.0, 3.0, 5.0],
                vec![0.0, 1.0, 0.5, -0.2],
                0.1,
                Want::Boundary,
                "hard fall-through",
            ),
            (
                vec![-1.0, -1.0, 2.0, 3.0],
                vec![0.0, 0.0, 0.4, -0.3],
                5.0,
                Want::Hard,
                "hard, degenerate cluster",
            ),
            (
                vec![-1.0, -1.0, 2.0],
                vec![0.0, 0.5, 0.2],
                0.3,
                Want::Boundary,
                "degenerate cluster, easy",
            ),
            (
                vec![0.0, 1.0, 2.0],
                vec![0.0, 0.5, -0.4],
                10.0,
                Want::Singular,
                "PSD-singular",
            ),
            (
                vec![-2.0, 1.0, 3.0, 5.0],
                vec![1e-11, 1.0, 0.5, -0.2],
                10.0,
                Want::NearHard,
                "near-hard 1e-11",
            ),
            (
                vec![-2.0, 1.0, 3.0, 5.0],
                vec![1e-10, 1.0, 0.5, -0.2],
                10.0,
                Want::NearHard,
                "near-hard 1e-10",
            ),
            (
                vec![-2.0, 1.0, 3.0, 5.0],
                vec![-1e-6, 1.0, 0.5, -0.2],
                10.0,
                Want::NearHard,
                "near-hard 1e-6",
            ),
            (vec![2.0], vec![3.0], 0.5, Want::Boundary, "n = 1"),
            (
                vec![-3.0],
                vec![1e-3],
                1.0,
                Want::Boundary,
                "n = 1, negative",
            ),
        ];
        for (lam, g, delta, want, what) in cases {
            let n = lam.len();
            let st = tr_solve_eig(&lam, &g, delta);
            let k = &st.khat;
            assert_eq!(k.len(), n, "{what}: khat length");
            let kn = nrm(k);
            let gn = nrm(&g);
            let scale = gn.max(1.0) * (1.0 + delta);
            assert!(st.mu >= 0.0, "{what}: μ = {} < 0", st.mu);
            assert!(
                lam[0] + st.mu >= -1e-12 * scale,
                "{what}: λ₀ + μ = {} < 0",
                lam[0] + st.mu
            );
            assert!(
                kn <= delta * (1.0 + 1e-12),
                "{what}: ‖k‖ = {kn:e} > Δ = {delta:e}"
            );
            for i in 0..n {
                let r = (lam[i] + st.mu) * k[i] + g[i];
                assert!(
                    r.abs() <= 1e-10 * scale,
                    "{what}: (λ_{i} + μ)k_{i} + ĝ_{i} = {r:e}"
                );
            }
            assert!(
                (st.mu * (delta - kn)).abs() <= 1e-10 * (1.0 + st.mu) * delta,
                "{what}: complementarity μ(Δ − ‖k‖) = {:e}",
                st.mu * (delta - kn)
            );
            let m = model(&lam, &g, k);
            assert!(
                (st.pred + m).abs() <= 1e-14 * m.abs().max(1.0),
                "{what}: pred = {} is not −m(k) = {}",
                st.pred,
                -m
            );
            assert!(st.pred > 0.0, "{what}: pred = {} ≤ 0", st.pred);
            let mut best = f64::INFINITY;
            for s in 0..20_000 {
                let mut dir: Vec<f64> = (0..n)
                    .map(|_| {
                        let (u1, u2) = (unif().max(1e-300), unif());
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    })
                    .collect();
                let dn = nrm(&dir);
                let r = if s % 2 == 0 {
                    delta
                } else {
                    delta * unif().powf(1.0 / n as f64)
                };
                for x in dir.iter_mut() {
                    *x *= r / dn;
                }
                best = best.min(model(&lam, &g, &dir));
            }
            assert!(
                m <= best + 1e-12 * scale,
                "{what}: m(k) = {m:e} above a sampled feasible point {best:e}"
            );
            match want {
                Want::Interior => {
                    assert!(!st.boundary && !st.hard, "{what}: flags");
                    assert!(st.mu == 0.0, "{what}: μ = {}", st.mu);
                    for i in 0..n {
                        assert!(
                            (k[i] + g[i] / lam[i]).abs() <= 1e-15,
                            "{what}: k ≠ −ĝ/λ at {i}"
                        );
                    }
                }
                Want::Boundary => {
                    assert!(
                        st.boundary && !st.hard,
                        "{what}: flags boundary {} hard {}",
                        st.boundary,
                        st.hard
                    );
                    assert!(
                        st.mu > (-lam[0]).max(0.0),
                        "{what}: μ = {} not above max(0, −λ₀)",
                        st.mu
                    );
                    assert!(
                        (kn - delta).abs() <= 1e-12 * delta,
                        "{what}: ‖k‖ = {kn:e} not on the sphere"
                    );
                    let norm_at = |mu: f64| {
                        (0..n)
                            .map(|i| (g[i] / (lam[i] + mu)).powi(2))
                            .sum::<f64>()
                            .sqrt()
                    };
                    let (mut lo, mut hi) =
                        ((-lam[0]).max(0.0), (-lam[0]).max(0.0) + gn / delta + 1.0);
                    for _ in 0..400 {
                        let mid = 0.5 * (lo + hi);
                        if mid <= lo || mid >= hi {
                            break;
                        }
                        if norm_at(mid) > delta {
                            lo = mid
                        } else {
                            hi = mid
                        }
                    }
                    for i in 0..n {
                        let kb = -g[i] / (lam[i] + hi);
                        assert!(
                            (k[i] - kb).abs() <= 1e-9 * delta,
                            "{what}: k_{i} = {} vs bisection {kb}",
                            k[i]
                        );
                    }
                }
                Want::Hard => {
                    assert!(
                        st.boundary && st.hard,
                        "{what}: flags boundary {} hard {}",
                        st.boundary,
                        st.hard
                    );
                    assert!(
                        (st.mu + lam[0]).abs() <= 1e-14 * lam[0].abs().max(1.0),
                        "{what}: μ = {} ≠ −λ₀",
                        st.mu
                    );
                    assert!(
                        (kn - delta).abs() <= 1e-12 * delta,
                        "{what}: ‖k‖ = {kn:e} not on the sphere"
                    );
                }
                Want::Singular => {
                    assert!(st.mu == 0.0, "{what}: μ = {} (λ₀ = 0, ĝ₀ = 0)", st.mu);
                }
                Want::NearHard => {
                    assert!(
                        st.boundary && !st.hard,
                        "{what}: flags boundary {} hard {}",
                        st.boundary,
                        st.hard
                    );
                    assert!(
                        (kn - delta).abs() <= 1e-12 * delta,
                        "{what}: ‖k‖/Δ − 1 = {:e}, not on the sphere",
                        kn / delta - 1.0
                    );
                    let mut g_hard = g.clone();
                    g_hard[0] = 0.0;
                    let hard = tr_solve_eig(&lam, &g_hard, delta);
                    assert!(
                        hard.hard,
                        "{what}: the ĝ₀ = 0 reference is not the hard case"
                    );
                    let gap = (st.pred - hard.pred).abs();
                    assert!(
                        gap <= g[0].abs() * delta + 1e-12 * scale,
                        "{what}: pred {} vs hard-case {} (gap {gap:e} > |ĝ₀|Δ = {:e})",
                        st.pred,
                        hard.pred,
                        g[0].abs() * delta
                    );
                }
            }
        }
    }

    #[test]
    fn orbopt_newton_step_is_pool_invariant() {
        let d = 9;
        let d4 = d * d * d * d;
        let gamma1 = generic_asym(d, 2, 0.6);
        let gamma2: Vec<f64> = (0..d4)
            .map(|f| 0.1 * (0.37 * f as f64 + 1.3).sin())
            .collect();
        let rdm = Rdm12 {
            d,
            n: 2,
            gamma1,
            gamma2,
        };
        let ints = ints_d(d, 1, 1, generic_asym(d, 2, 0.1), generic_asym(d, 4, 0.5));
        let pool = |n: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("rayon pool")
        };
        struct Restore(usize);
        impl Drop for Restore {
            fn drop(&mut self) {
                crate::par::set_par_min_fork_work(self.0);
            }
        }
        let _lock = crate::par::lock_par_switches();
        let _restore = Restore(crate::par::set_par_min_fork_work(0));
        let opts = FrozenOrbOpts {
            max_iter: 3,
            method: FrozenOrbMethod::Newton(FrozenTrOpts::default()),
            ..FrozenOrbOpts::default()
        };
        let run = || {
            (
                frozen_orbital_hessian(&ints, &rdm),
                frozen_orbital_step(&ints, &rdm, &opts),
            )
        };
        let (h1, (u1, i1)) = pool(1).install(run);
        let (h4, (u4, i4)) = pool(4).install(run);
        assert!(
            i1.n_hessians >= 1,
            "no Hessian built: the Newton path is not exercised"
        );
        assert_eq!(h1.len(), h4.len());
        for (f, (a, b)) in h1.iter().zip(&h4).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "H[{f}]: pool 1 {a:e} vs pool 4 {b:e}"
            );
        }
        for (f, (a, b)) in u1.iter().zip(&u4).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "U[{f}]: pool 1 {a:e} vs pool 4 {b:e}"
            );
        }
        assert_eq!(i1.e_after.to_bits(), i4.e_after.to_bits(), "e_after");
        assert_eq!(i1.g_after.to_bits(), i4.g_after.to_bits(), "g_after");
        assert_eq!(
            i1.kappa_norm.to_bits(),
            i4.kappa_norm.to_bits(),
            "kappa_norm"
        );
        assert_eq!(
            i1.min_curvature.to_bits(),
            i4.min_curvature.to_bits(),
            "min_curvature"
        );
        assert_eq!(
            (i1.iters, i1.n_evals, i1.n_hessians),
            (i4.iters, i4.n_evals, i4.n_hessians),
            "counts"
        );
        assert_eq!(i1.exit, i4.exit, "exit");
    }
}
