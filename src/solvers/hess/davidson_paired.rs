// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::factor::{cluster_null_space, dense_eigen_general, dense_self_adjoint_smallest};
use crate::par;
use crate::scalar::{C64, Scalar};
use crate::solvers::InnerProductVector;
use rayon::prelude::*;
use std::time::Instant;

use super::NegativeWitness;
use super::davidson::recombine;
use super::{
    WeingartenCtx, add_complement_shift, clean_tangent, complement_shift, diag_inf_scale,
    radial_mode, reproject_tangent,
};
use crate::core::tangent::TtTangent;

const SPLIT_TOL: f64 = 1e-10;
const DROP_TOL: f64 = 1e-10;
const COLLAPSE_FLOOR: f64 = 1.5e-8;

/// Options of [`omega_davidson_paired`].
#[derive(Clone, Copy, Debug)]
pub struct PairedDavidsonOpts {
    /// Relative residual tolerance in `ω` units (module doc).
    pub tol: f64,
    /// Total budget of accepted basis vectors; each costs one `A` and one `B` apply.
    pub max_basis: usize,
    /// Thick-restart bound on the live basis; `0` = auto, `max(8·(k+b), 32)` as the W-metric
    /// kernels, so equal settings hold equal memory: three stored vectors per basis
    /// vector on both (`v, Av, Bv` here; `v, Wv, Mv` there). The live basis exceeds it only when a
    /// restart must keep more (module doc; [`PairedInfo::peak_live`]).
    pub max_subspace: usize,
    /// Leading outer iterations whose corrections skip the preconditioner.
    pub nwarm: usize,
    /// `false` = the `k` lowest real pairs count both Krein types — the signed kernel's rule.
    /// `true` = they count POSITIVE-type pairs only (the ones the local variational principle
    /// bounds), and the negative-type real pairs inside that window join the anomalies. A
    /// selective wanted set (a target, or a [`WantedRank`] other than `Omega`) always counts
    /// positive-type pairs only: no gate decides a negative-type pair's place.
    pub positive_window: bool,
    /// What the solve does with an anomalous Ritz pair (imaginary, complex in the window, and under
    /// `positive_window` a negative-type real pair in it).
    pub anomalies: AnomalyRule,
    /// How the `k` wanted real pairs are chosen when no target is given.
    pub rank: WantedRank,
}

impl Default for PairedDavidsonOpts {
    fn default() -> Self {
        PairedDavidsonOpts {
            tol: 1e-8,
            max_basis: 400,
            max_subspace: 0,
            nwarm: 2,
            positive_window: false,
            anomalies: AnomalyRule::Chase,
            rank: WantedRank::Omega,
        }
    }
}

/// The wanted-set ranking of [`PairedDavidsonOpts::rank`] (no target vectors).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum WantedRank {
    /// The `k` lowest eligible real `ω`.
    #[default]
    Omega,
    /// The `k` lowest eligible real `ω` among the pairs with `ρ_A ≤ c·ω`
    /// (`ρ_A = XᵀAX/XᵀX`, [`PairedMode::rho_a`], read off `Â` at no apply); every eligible pair
    /// below the largest wanted `ω` that was not wanted is reported in
    /// [`PairedDavidsonOut::untargeted_below`], as with a target. The excess `ρ_A/ω − 1 = −XᵀBY/(ω XᵀX)` (from
    /// `AX + BY = ωX`) is the mode's `B`-coupling share: ladder 2026-10-23 `ρ_A/ω` ≥ 4.6 on
    /// artifacts, ≤ 2.34 on the levels TDA targeting kept.
    AExcess(f64),
    /// With a target: the `k` lowest eligible real `ω` among the pairs whose `X` overlaps the
    /// target span by at least `τ` (`‖P_S X‖²/‖X‖² ≥ τ`) — the span as an eligibility gate, not a
    /// ranking. Overlap RANKING trades low levels for high ones once the span holds more than `k`
    /// states, and misses a window-edge level the span under-represents once it holds `k` (water
    /// cap 128 `Hard(3e-4)`, k 8, HF/CIS seeds: 16 seeds 4/8 of the TDA arm's levels, 8 seeds 7/8).
    TargetGate(f64),
    /// With a target: [`Self::TargetGate`]`(τ)` OR [`Self::AExcess`]`(c)` — the `k` lowest
    /// eligible real `ω` among the pairs with overlap `≥ τ` or `ρ_A ≤ c·ω`. A level the target
    /// span cannot reach (a doubly excited state against single-excitation seeds: overlap of order
    /// the reference's singles weight) passes on its `B`-coupling share; a curvature artifact
    /// fails both.
    TargetGateOrExcess(f64, f64),
}

/// The anomalous-pair rule of [`PairedDavidsonOpts::anomalies`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum AnomalyRule {
    /// Wanted: corrected, kept at a restart and in the convergence test (the signed kernel's rule).
    #[default]
    Chase,
    /// Wanted only once its residual is `≤ f·|ω|`; the rest are counted in
    /// [`PairedInfo::gated_out`], get no correction and are not kept at a restart — so a genuine
    /// anomaly is found only if the expansion converges it incidentally (measured on C5: it is NOT).
    Gate(f64),
    /// Reported and kept at every restart, never corrected and not in the convergence test:
    /// `converged` = the positive window (or the target). The restart also keeps the negative
    /// eigenvectors of `Â ± B̂`, so the counting offset survives it (module doc). Needs
    /// `positive_window`.
    Freeze,
}

/// The counting-offset monitor of the projected pencil. With `Â ≻ 0` (`neg_a == 0`),
/// `n₋(T̂(0)) = n₋(Â+B̂) + n₋(Â−B̂)` (Haynsworth): the offset the positive window's Ritz bounds
/// need (module doc). Without a restart it never decreases as the basis grows (Cauchy interlacing),
/// and it never exceeds the full pencil's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PairedOffset {
    pub neg_apb: usize,
    pub neg_amb: usize,
    pub neg_a: usize,
}

impl PairedOffset {
    /// `n₋(Â+B̂) + n₋(Â−B̂)`.
    pub fn total(&self) -> usize {
        self.neg_apb + self.neg_amb
    }
}

/// One wanted Ritz pair of the paired pencil, its member with `Re ω > 0` (or `Im ω > 0` for an
/// imaginary pair), with `z = (X, Y)` Euclidean-normalized. The imaginary parts are `None` for a
/// real `ω`.
pub struct PairedMode<V> {
    pub omega: C64,
    /// `‖r‖ = ‖H z − ω Σ z‖` from the cached images, `‖z‖ = 1`.
    pub residual: f64,
    /// Krein margin `(‖X‖² − ‖Y‖²)/(‖X‖² + ‖Y‖²)`: `+1` with `Y = 0`, `0` for an imaginary or
    /// complex pair; `1/|margin|` is the eigenvalue's condition number relative to a normal one.
    pub krein: f64,
    /// `‖P_S X‖² / ‖X‖²` against the target space `S` (`None` without a target).
    pub overlap: Option<f64>,
    /// `ρ_A = XᵀAX / XᵀX` from `Â` (real and imaginary parts summed); `ρ_A − Re ω` is the mode's
    /// distance from a pure Tamm–Dancoff one ([`WantedRank::AExcess`]).
    pub rho_a: f64,
    /// Rank among the positive-type (`positive_window`; else all) real Ritz pairs, ascending in
    /// `ω` (`None` for an anomaly).
    pub window_index: Option<usize>,
    pub x_re: V,
    pub y_re: V,
    pub x_im: Option<V>,
    pub y_im: Option<V>,
}

impl<V> PairedMode<V> {
    /// `ω²` (complex in general; real for a real or imaginary `ω`).
    pub fn omega2(&self) -> C64 {
        self.omega * self.omega
    }
}

/// Counts and verdict of one paired solve.
#[derive(Clone, Debug, Default)]
pub struct PairedInfo {
    pub converged: bool,
    /// The Rayleigh–Ritz problem was the full pencil: the basis reached `space_dim`, or every raw
    /// residual collapsed into its span at the roundoff floor — the realized space, below the
    /// structural `space_dim` at a numerically rank-deficient point. Implies `converged`; the
    /// per-mode residuals then sit at the operator's roundoff and asymmetry floor, whatever `tol`
    /// asked.
    pub exhausted: bool,
    /// Accepted basis vectors over the run.
    pub built: usize,
    /// `(A v, B v)` apply pairs (one per accepted vector; with `B ≡ 0` the B half is free).
    pub apply_pairs: usize,
    pub outer_iterations: usize,
    pub restarts: usize,
    /// Anomalous Ritz pairs the last Rayleigh–Ritz left out under [`AnomalyRule::Gate`].
    pub gated_out: usize,
    /// Anomalous Ritz pairs the last Rayleigh–Ritz reported uncorrected under
    /// [`AnomalyRule::Freeze`].
    pub frozen: usize,
    /// Correction vectors queued from anomalous slots over the run (0 under `Freeze`).
    pub anomaly_corrections: usize,
    /// The offset monitor on the final basis.
    pub offset: PairedOffset,
    /// Restarts after which `n₋(Â+B̂) + n₋(Â−B̂)` was lower than before.
    pub offset_drops: usize,
    /// Under a target gate: the wanted pairs of the last Rayleigh–Ritz that fail the gate (the
    /// top-up by overlap). Nonzero at exit = fewer than `k` gate-passing pairs were found.
    pub gate_short: usize,
    /// The largest live basis the run held. Above the `max_subspace` bound when a thick restart
    /// had to keep more than the bound leaves room for (module doc, `# Wanted set and units`).
    pub peak_live: usize,
    /// The run stopped with the convergence test unmet and nothing it could add: either nothing to
    /// correct (fewer than `k` eligible real pairs in the basis, every one of them converged — an
    /// invariant subspace the seeds did not leave), or every correction collapsed into span(V)
    /// while a residual stayed above the roundoff floor (an inconsistent projection, never read
    /// as an exhaustion). With `converged` false and `built < max_basis` this is why it stopped.
    pub stalled: bool,
    /// Wall-clock seconds per phase of the run.
    pub phases: PairedPhases,
    /// One fresh-apply certificate per returned imaginary pair, `index` into
    /// [`PairedDavidsonOut::real`]: the pencil re-solved on the mode's own span
    /// `{Re X, Im X, Re Y, Im Y}` with FRESH `A`/`B` applies — `fresh` its `ω²` nearest the Ritz
    /// value, `fresh_w` the lowest eigenvalue of `A + B` and `A − B` on that span (negative = a
    /// variational witness that one block is indefinite). A SIMPLE imaginary pair's `Y` lies in
    /// `{Re X, Im X}` and the span collapses to two directions (two apply pairs); a doubled one's
    /// (every pair on the realified C64 layer) does not, and needs all four (up to four apply
    /// pairs), counted in `apply_pairs`.
    pub negative: Vec<NegativeWitness>,
}

/// Wall-clock seconds per phase of one paired solve, summed over the run. `total` is the
/// kernel's own wall; whatever the phases leave of it is bookkeeping (the trace included).
#[derive(Clone, Copy, Debug, Default)]
pub struct PairedPhases {
    /// The `A` applies (on the TT layer the caller's cleaning and complement term included).
    pub apply_a: f64,
    /// The `B` applies.
    pub apply_b: f64,
    /// Every candidate's projection and two-pass MGS, accepted or not.
    pub orth: f64,
    /// The incremental `Â`, `B̂` columns and target overlaps of each accepted vector.
    pub gram: f64,
    /// The dense Rayleigh–Ritz problem and the wanted-set selection.
    pub ritz: f64,
    /// The batched recombinations `V·c`, `(AV)·c`, `(BV)·c` and the residuals.
    pub residual: f64,
    /// The thick restarts: offset monitor, recombination, re-orthonormalization, rebuilt Grams.
    pub restart: f64,
    /// The preconditioner.
    pub precond: f64,
    pub total: f64,
}

impl std::fmt::Display for PairedPhases {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "apply A {:.2}s B {:.2}s | orth {:.2}s gram {:.2}s ritz {:.2}s residual {:.2}s \
             restart {:.2}s precond {:.2}s | total {:.2}s",
            self.apply_a,
            self.apply_b,
            self.orth,
            self.gram,
            self.ritz,
            self.residual,
            self.restart,
            self.precond,
            self.total
        )
    }
}

/// Everything a paired solve returns: `real` = every imaginary pair (ascending `ω²`) then the `k`
/// lowest real `ω ≥ 0` (or, with a target, the `k` of largest overlap, ascending) — the signed
/// kernel's `omega2` list — and `complex` = the wanted quadruplets, `Im ω² > 0` member.
pub struct PairedDavidsonOut<V> {
    pub real: Vec<PairedMode<V>>,
    pub complex: Vec<PairedMode<V>>,
    /// With a target: every eligible real Ritz pair below the largest targeted `ω` that was NOT
    /// targeted (artifacts and missed modes), ascending; empty without a target.
    pub untargeted_below: Vec<PairedMode<V>>,
    pub info: PairedInfo,
}

impl<V> PairedDavidsonOut<V> {
    /// The signed kernel's `omega2` list: `ω²` of every entry of [`Self::real`].
    pub fn omega2(&self) -> Vec<f64> {
        self.real.iter().map(|m| m.omega2().re).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RitzKind {
    Real,
    Imaginary,
    Complex,
}

#[derive(Clone, Debug)]
pub(crate) struct PairedRitz {
    pub(crate) omega: C64,
    pub(crate) kind: RitzKind,
    pub(crate) a_re: Vec<f64>,
    pub(crate) a_im: Vec<f64>,
    pub(crate) b_re: Vec<f64>,
    pub(crate) b_im: Vec<f64>,
    pub(crate) krein: f64,
}

pub(crate) fn paired_ritz(ah: &[f64], bh: &[f64], m: usize) -> Vec<PairedRitz> {
    let n2 = 2 * m;
    let mut h = vec![0.0f64; n2 * n2];
    for i in 0..m {
        for j in 0..m {
            let (a, b) = (ah[i * m + j], bh[i * m + j]);
            h[i * n2 + j] = a;
            h[i * n2 + m + j] = b;
            h[(m + i) * n2 + j] = -b;
            h[(m + i) * n2 + m + j] = -a;
        }
    }
    let (mut ev, mut evec) = dense_eigen_general::<f64>(&h, n2);
    let scale = ev
        .iter()
        .fold(0.0f64, |a, z| a.max(z.norm()))
        .max(f64::MIN_POSITIVE);
    let tz = SPLIT_TOL * scale;
    realify_near_real_pairs(&h, n2, tz, &mut ev, &mut evec);
    let mut order: Vec<usize> = (0..n2).collect();
    order.sort_by(|&p, &q| {
        ev[p]
            .re
            .total_cmp(&ev[q].re)
            .then(ev[p].im.total_cmp(&ev[q].im))
    });
    let mut used = vec![false; n2];
    let mut reps: Vec<usize> = Vec::with_capacity(m);
    for &i in &order {
        if used[i] {
            continue;
        }
        used[i] = true;
        let mut best: Option<(usize, f64)> = None;
        for j in 0..n2 {
            if used[j] {
                continue;
            }
            let d = (ev[j] + ev[i]).norm();
            if best.is_none_or(|(_, bd)| d < bd) {
                best = Some((j, d));
            }
        }
        let rep = match best {
            Some((j, _)) => {
                used[j] = true;
                let (zi, zj) = (ev[i], ev[j]);
                let pick_i = if zi.re.abs() > tz || zj.re.abs() > tz {
                    zi.re >= zj.re
                } else {
                    zi.im >= zj.im
                };
                if pick_i { i } else { j }
            }
            None => i,
        };
        reps.push(rep);
    }
    let mut out: Vec<PairedRitz> = Vec::with_capacity(reps.len());
    for c in reps {
        let lam = ev[c];
        let kind = if lam.im.abs() <= tz {
            RitzKind::Real
        } else if lam.re.abs() <= tz {
            RitzKind::Imaginary
        } else {
            RitzKind::Complex
        };
        if kind == RitzKind::Complex && lam.im < 0.0 {
            continue;
        }
        let mut z = evec[c].clone();
        let p = (0..n2)
            .max_by(|&p, &q| z[p].norm().total_cmp(&z[q].norm()))
            .unwrap_or(0);
        let zp = z[p];
        if zp.norm() > 0.0 {
            let ph = zp.conj() * (1.0 / zp.norm());
            z.iter_mut().for_each(|x| *x *= ph);
        }
        let (omega, real) = match kind {
            RitzKind::Real => (C64::new(lam.re, 0.0), true),
            _ => (lam, false),
        };
        let mut a_re: Vec<f64> = z[..m].iter().map(|x| x.re).collect();
        let mut b_re: Vec<f64> = z[m..].iter().map(|x| x.re).collect();
        let (mut a_im, mut b_im): (Vec<f64>, Vec<f64>) = if real {
            (vec![0.0; m], vec![0.0; m])
        } else {
            (
                z[..m].iter().map(|x| x.im).collect(),
                z[m..].iter().map(|x| x.im).collect(),
            )
        };
        let sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
        let (na, nb) = (sq(&a_re) + sq(&a_im), sq(&b_re) + sq(&b_im));
        let nz = (na + nb).sqrt().max(f64::MIN_POSITIVE);
        for v in [&mut a_re, &mut a_im, &mut b_re, &mut b_im] {
            v.iter_mut().for_each(|x| *x /= nz);
        }
        let krein = if kind == RitzKind::Real {
            (na - nb) / (na + nb).max(f64::MIN_POSITIVE)
        } else {
            0.0
        };
        out.push(PairedRitz {
            omega,
            kind,
            a_re,
            a_im,
            b_re,
            b_im,
            krein,
        });
    }
    let key = |r: &PairedRitz| match r.kind {
        RitzKind::Real => r.omega.re,
        _ => (r.omega * r.omega).re,
    };
    out.sort_by(|p, q| key(p).total_cmp(&key(q)));
    out
}

fn realify_near_real_pairs(h: &[f64], n: usize, tz: f64, ev: &mut [C64], evec: &mut [Vec<C64>]) {
    let dot = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(x, y)| x * y).sum::<f64>();
    let mut done = vec![false; n];
    for i in 0..n {
        if done[i] || !(ev[i].im > 0.0 && ev[i].im <= tz) {
            continue;
        }
        let target = ev[i].conj();
        let Some(j) = (0..n)
            .filter(|&j| !done[j] && j != i && ev[j].im < 0.0)
            .min_by(|&p, &q| (ev[p] - target).norm().total_cmp(&(ev[q] - target).norm()))
        else {
            continue;
        };
        if (ev[j] - target).norm() > tz {
            continue;
        }
        done[i] = true;
        done[j] = true;
        let lam = ev[i].re;
        let re: Vec<f64> = evec[i].iter().map(|z| z.re).collect();
        let im: Vec<f64> = evec[i].iter().map(|z| z.im).collect();
        let (mut u, mut w) = if dot(&re, &re) >= dot(&im, &im) {
            (re, im)
        } else {
            (im, re)
        };
        let nu = dot(&u, &u).sqrt();
        u.iter_mut().for_each(|x| *x /= nu);
        for _ in 0..2 {
            let c = dot(&u, &w);
            w.iter_mut().zip(&u).for_each(|(x, y)| *x -= c * y);
        }
        let nw = dot(&w, &w).sqrt();
        let plane: Vec<Vec<C64>> = if nw > 1e-3 * nu {
            w.iter_mut().for_each(|x| *x /= nw);
            [u, w]
                .into_iter()
                .map(|v| v.into_iter().map(|x| C64::new(x, 0.0)).collect())
                .collect()
        } else {
            cluster_null_space(h, n, C64::new(lam, 0.0), 2, true)
                .into_iter()
                .map(|(_, v)| v)
                .collect()
        };
        for (c, v) in [i, j].into_iter().zip(plane) {
            ev[c] = C64::new(lam, 0.0);
            evec[c] = v;
        }
    }
}

fn orthonormalize_against<V: InnerProductVector>(mut v: V, basis: &[V]) -> Option<V> {
    let n0 = v.norm();
    if n0 <= 0.0 || !n0.is_finite() {
        return None;
    }
    for _ in 0..2 {
        for u in basis {
            let c = u.dot(&v).re_s();
            v.axpy(V::Scalar::from_f64(-c), u);
        }
    }
    let nr = v.norm();
    if nr > DROP_TOL * n0 {
        v.scale(V::Scalar::from_f64(1.0 / nr));
        Some(v)
    } else {
        None
    }
}

fn orthonormal_coefficients(cs: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::with_capacity(cs.len());
    for mut c in cs {
        let n0 = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        for _ in 0..2 {
            for q in &out {
                let d: f64 = q.iter().zip(&c).map(|(x, y)| x * y).sum();
                c.iter_mut().zip(q).for_each(|(x, y)| *x -= d * y);
            }
        }
        let nr = c.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nr > DROP_TOL * n0 && nr > 0.0 {
            c.iter_mut().for_each(|x| *x /= nr);
            out.push(c);
        }
    }
    out
}

fn push_candidate<V, Pj>(mut v: V, basis: &[V], project: &mut Pj) -> Option<V>
where
    V: InnerProductVector,
    Pj: FnMut(&mut V),
{
    project(&mut v);
    let mut v = orthonormalize_against(v, basis)?;
    project(&mut v);
    let n0 = v.norm();
    for u in basis {
        let c = u.dot(&v).re_s();
        v.axpy(V::Scalar::from_f64(-c), u);
    }
    let nr = v.norm();
    if nr > DROP_TOL * n0 && nr > 0.0 {
        v.scale(V::Scalar::from_f64(1.0 / nr));
        Some(v)
    } else {
        None
    }
}

fn gram_column<V: InnerProductVector + Send + Sync>(vs: &[V], images: &[V], j: usize) -> Vec<f64> {
    let n_coeff = vs.first().map_or(0, |v| v.num_coeffs());
    let entry =
        |i: usize| -> f64 { 0.5 * (vs[i].dot(&images[j]).re_s() + vs[j].dot(&images[i]).re_s()) };
    if j >= 1 && par::par_driver_fork((j + 1).saturating_mul(n_coeff)) {
        (0..=j)
            .into_par_iter()
            .map(|i| {
                let _nest = par::NestGuard::new();
                entry(i)
            })
            .collect()
    } else {
        (0..=j).map(entry).collect()
    }
}

fn grow_symmetric(g: &mut Vec<f64>, m: usize, col: &[f64]) {
    let n = m + 1;
    let mut out = vec![0.0f64; n * n];
    for i in 0..m {
        out[i * n..i * n + m].copy_from_slice(&g[i * m..i * m + m]);
    }
    for (i, &c) in col.iter().enumerate() {
        out[i * n + m] = c;
        out[m * n + i] = c;
    }
    *g = out;
}

struct PairedBasis<V> {
    vs: Vec<V>,
    av: Vec<V>,
    bv: Vec<V>,
    ah: Vec<f64>,
    bh: Vec<f64>,
    g: Vec<Vec<f64>>,
}

impl<V: InnerProductVector + Send + Sync> PairedBasis<V> {
    fn new() -> Self {
        PairedBasis {
            vs: Vec::new(),
            av: Vec::new(),
            bv: Vec::new(),
            ah: Vec::new(),
            bh: Vec::new(),
            g: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.vs.len()
    }

    fn append(&mut self, v: V, av: V, bv: V, qs: &[V]) {
        let m = self.vs.len();
        self.g.push(qs.iter().map(|q| q.dot(&v).re_s()).collect());
        self.vs.push(v);
        self.av.push(av);
        self.bv.push(bv);
        let ca = gram_column(&self.vs, &self.av, m);
        let cb = gram_column(&self.vs, &self.bv, m);
        grow_symmetric(&mut self.ah, m, &ca);
        grow_symmetric(&mut self.bh, m, &cb);
    }
}

fn target_overlap(g: &[Vec<f64>], s: usize, a_re: &[f64], a_im: &[f64]) -> f64 {
    let mut num = 0.0;
    for i in 0..s {
        let (mut re, mut im) = (0.0, 0.0);
        for (j, gj) in g.iter().enumerate() {
            re += gj[i] * a_re[j];
            im += gj[i] * a_im[j];
        }
        num += re * re + im * im;
    }
    let den: f64 = a_re.iter().chain(a_im).map(|x| x * x).sum();
    if den > 0.0 { num / den } else { 0.0 }
}

fn a_energy(ah: &[f64], m: usize, a_re: &[f64], a_im: &[f64]) -> f64 {
    let quad = |a: &[f64]| -> f64 {
        (0..m)
            .map(|i| a[i] * (0..m).map(|j| ah[i * m + j] * a[j]).sum::<f64>())
            .sum()
    };
    let den: f64 = a_re.iter().chain(a_im).map(|x| x * x).sum();
    if den > 0.0 {
        (quad(a_re) + quad(a_im)) / den
    } else {
        0.0
    }
}

const NEG_TOL: f64 = 1e-12;

fn negative_part(s: &[f64], m: usize, vectors: bool) -> (usize, Vec<Vec<f64>>) {
    if m == 0 {
        return (0, Vec::new());
    }
    let (ev, q) = dense_self_adjoint_smallest::<f64>(s, m, if vectors { m } else { 0 });
    let scale = ev.iter().fold(0.0f64, |a, x| a.max(x.abs()));
    let n = ev.iter().filter(|&&x| x < -NEG_TOL * scale).count();
    (n, q.into_iter().take(n).collect())
}

fn offset_of(ah: &[f64], bh: &[f64], m: usize, vectors: bool) -> (PairedOffset, Vec<Vec<f64>>) {
    let apb: Vec<f64> = ah.iter().zip(bh).map(|(a, b)| a + b).collect();
    let amb: Vec<f64> = ah.iter().zip(bh).map(|(a, b)| a - b).collect();
    let (neg_apb, mut q) = negative_part(&apb, m, vectors);
    let (neg_amb, q2) = negative_part(&amb, m, vectors);
    let (neg_a, _) = negative_part(ah, m, false);
    q.extend(q2);
    (
        PairedOffset {
            neg_apb,
            neg_amb,
            neg_a,
        },
        q,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    Wanted,
    Anomaly,
    Report,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn omega_davidson_paired<V, A, B, Pj, P>(
    mut a_apply: A,
    mut b_apply: B,
    mut project: Pj,
    mut precond: Option<P>,
    b_zero: bool,
    v0s: &[V],
    target: &[V],
    k: usize,
    space_dim: usize,
    opts: &PairedDavidsonOpts,
) -> PairedDavidsonOut<V>
where
    V: InnerProductVector + Send + Sync,
    A: FnMut(&V) -> V,
    B: FnMut(&V) -> V,
    Pj: FnMut(&mut V),
    P: FnMut(&mut V, f64),
{
    assert!(k >= 1, "omega_davidson_paired: k must be ≥ 1");
    assert!(
        space_dim >= 1,
        "omega_davidson_paired: empty iteration space"
    );
    assert!(!v0s.is_empty(), "omega_davidson_paired: empty start block");
    assert!(
        opts.positive_window || opts.anomalies != AnomalyRule::Freeze,
        "omega_davidson_paired: AnomalyRule::Freeze needs positive_window"
    );
    let t_solve = Instant::now();
    let b = v0s.len();
    let max_subspace = if opts.max_subspace == 0 {
        (8 * (k + b)).max(32)
    } else {
        opts.max_subspace
    }
    .clamp((4 * (k + b)).min(space_dim), space_dim);
    let max_basis = opts.max_basis.max(k);
    let trace = std::env::var("PAIRED_TRACE").is_ok();
    let mut info = PairedInfo::default();
    let zero = {
        let mut z = v0s[0].clone();
        z.scale(V::Scalar::from_f64(0.0));
        z
    };
    let mut qs: Vec<V> = Vec::with_capacity(target.len());
    for t in target {
        let mut t = t.clone();
        project(&mut t);
        if let Some(q) = orthonormalize_against(t, &qs) {
            qs.push(q);
        }
    }
    assert!(
        target.is_empty() || !qs.is_empty(),
        "omega_davidson_paired: every target vector collapsed under projection — caller bug"
    );
    let s = qs.len();
    let excess = match opts.rank {
        WantedRank::AExcess(c) | WantedRank::TargetGateOrExcess(_, c) => {
            assert!(c > 1.0, "omega_davidson_paired: AExcess(c) needs c > 1");
            Some(c)
        }
        _ => None,
    };
    let gate = match opts.rank {
        WantedRank::TargetGate(tau) | WantedRank::TargetGateOrExcess(tau, _) => {
            assert!(
                s > 0 && tau > 0.0 && tau <= 1.0,
                "omega_davidson_paired: TargetGate(τ) needs a target and 0 < τ ≤ 1"
            );
            Some(tau)
        }
        _ => None,
    };
    assert!(
        !(matches!(opts.rank, WantedRank::AExcess(_)) && s > 0),
        "omega_davidson_paired: AExcess and a target are exclusive"
    );
    let selective = s > 0 || excess.is_some();
    let mut basis: PairedBasis<V> = PairedBasis::new();
    let mut push = |v: V, basis: &mut PairedBasis<V>, info: &mut PairedInfo| -> bool {
        let t = Instant::now();
        let cand = push_candidate(v, &basis.vs, &mut project);
        info.phases.orth += t.elapsed().as_secs_f64();
        let Some(v) = cand else {
            return false;
        };
        let t = Instant::now();
        let av = a_apply(&v);
        info.phases.apply_a += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let bv = if b_zero { zero.clone() } else { b_apply(&v) };
        info.phases.apply_b += t.elapsed().as_secs_f64();
        info.apply_pairs += 1;
        info.built += 1;
        let t = Instant::now();
        basis.append(v, av, bv, &qs);
        info.phases.gram += t.elapsed().as_secs_f64();
        info.peak_live = info.peak_live.max(basis.len());
        true
    };
    for v0 in v0s {
        if basis.len() == space_dim {
            break;
        }
        push(v0.clone(), &mut basis, &mut info);
    }
    assert!(
        basis.len() > 0,
        "omega_davidson_paired: every seed collapsed under projection — caller bug"
    );

    let mut last_real: Vec<PairedMode<V>>;
    let mut last_cplx: Vec<PairedMode<V>>;
    let mut last_untargeted: Vec<PairedMode<V>>;
    let freeze = opts.anomalies == AnomalyRule::Freeze;
    let mut live_cap = max_subspace;
    let mut outer = 0usize;
    loop {
        let m = basis.len();
        info.outer_iterations += 1;
        let t_ritz = Instant::now();
        let ritz = paired_ritz(&basis.ah, &basis.bh, m);
        let pw = opts.positive_window || selective;
        let overlap_of = |r: &PairedRitz| target_overlap(&basis.g, s, &r.a_re, &r.a_im);
        let rho_a_of = |r: &PairedRitz| a_energy(&basis.ah, m, &r.a_re, &r.a_im);
        let pos_all: Vec<&PairedRitz> = ritz
            .iter()
            .filter(|r| r.kind == RitzKind::Real && (!pw || r.krein >= 0.0))
            .collect();
        let (pos, buffer): (Vec<&PairedRitz>, Vec<&PairedRitz>) = if !selective {
            (
                pos_all.iter().copied().take(k).collect(),
                pos_all.iter().copied().skip(k).take(b).collect(),
            )
        } else if excess.is_some() || gate.is_some() {
            let ok: Vec<&PairedRitz> = pos_all
                .iter()
                .copied()
                .filter(|&r| {
                    gate.is_some_and(|tau| overlap_of(r) >= tau)
                        || excess.is_some_and(|c| rho_a_of(r) <= c * r.omega.re)
                })
                .collect();
            let mut pos: Vec<&PairedRitz> = ok.iter().copied().take(k).collect();
            if gate.is_some() {
                info.gate_short = k.saturating_sub(pos.len());
            }
            if gate.is_some() && pos.len() < k {
                let mut rest: Vec<(&PairedRitz, f64)> = pos_all
                    .iter()
                    .copied()
                    .filter(|&r| !pos.iter().any(|&p| std::ptr::eq(p, r)))
                    .map(|r| (r, overlap_of(r)))
                    .collect();
                rest.sort_by(|p, q| q.1.total_cmp(&p.1));
                pos.extend(rest.iter().take(k - pos.len()).map(|p| p.0));
                pos.sort_by(|p, q| p.omega.re.total_cmp(&q.omega.re));
            }
            (pos, ok.iter().copied().skip(k).take(b).collect())
        } else {
            let mut by: Vec<(&PairedRitz, f64)> =
                pos_all.iter().map(|&r| (r, overlap_of(r))).collect();
            by.sort_by(|p, q| {
                q.1.total_cmp(&p.1)
                    .then(p.0.omega.re.total_cmp(&q.0.omega.re))
            });
            let mut w: Vec<&PairedRitz> = by.iter().take(k).map(|p| p.0).collect();
            w.sort_by(|p, q| p.omega.re.total_cmp(&q.omega.re));
            (w, by.iter().skip(k).take(b).map(|p| p.0).collect())
        };
        let window = if pos.len() == k {
            pos.iter().fold(0.0f64, |a, r| a.max(r.omega.re)).powi(2)
        } else {
            f64::INFINITY
        };
        let in_window = |r: &PairedRitz| (r.omega * r.omega).re <= window;
        let wanted = |r: &PairedRitz| pos.iter().any(|&p| std::ptr::eq(p, r));
        let mut real_list: Vec<(&PairedRitz, Role)> =
            pos.iter().map(|&r| (r, Role::Wanted)).collect();
        if pw {
            real_list.extend(
                ritz.iter()
                    .filter(|r| r.kind == RitzKind::Real && r.krein < 0.0 && in_window(r))
                    .map(|r| (r, Role::Anomaly)),
            );
            real_list.sort_by(|p, q| p.0.omega.re.total_cmp(&q.0.omega.re));
        }
        let cands: Vec<(&PairedRitz, Role)> = ritz
            .iter()
            .filter(|r| r.kind == RitzKind::Imaginary)
            .map(|r| (r, Role::Anomaly))
            .chain(real_list)
            .chain(
                ritz.iter()
                    .filter(|r| r.kind == RitzKind::Complex && in_window(r))
                    .map(|r| (r, Role::Anomaly)),
            )
            .chain(
                pos_all
                    .iter()
                    .copied()
                    .filter(|&r| selective && !wanted(r) && r.omega.re.powi(2) < window)
                    .map(|r| (r, Role::Report)),
            )
            .collect();
        let scale_pos = pos
            .iter()
            .fold(0.0f64, |a, r| a.max(r.omega.re))
            .max(f64::MIN_POSITIVE);
        let cands_scale = cands.iter().fold(0.0f64, |a, (r, _)| a.max(r.omega.norm()));
        info.phases.ritz += t_ritz.elapsed().as_secs_f64();

        let t_resid = Instant::now();
        let mut cs: Vec<Vec<f64>> = Vec::with_capacity(4 * cands.len());
        for (r, _) in &cands {
            cs.push(r.a_re.clone());
            cs.push(r.b_re.clone());
            if r.kind != RitzKind::Real {
                cs.push(r.a_im.clone());
                cs.push(r.b_im.clone());
            }
        }
        let nc = cs.len();
        let vc = recombine(&basis.vs, &cs, nc);
        let ac = recombine(&basis.av, &cs, nc);
        let bc = recombine(&basis.bv, &cs, nc);
        let sc = |x: f64| V::Scalar::from_f64(x);
        let mut modes: Vec<PairedMode<V>> = Vec::with_capacity(cands.len());
        let mut cplx_modes: Vec<PairedMode<V>> = Vec::new();
        let mut untargeted: Vec<PairedMode<V>> = Vec::new();
        let mut kept_cs: Vec<Vec<f64>> = Vec::with_capacity(nc);
        let mut corrections: Vec<(V, f64)> = Vec::new();
        let mut at = 0usize;
        let mut conv_all = true;
        info.gated_out = 0;
        info.frozen = 0;
        for &(r, role) in &cands {
            let (alpha, beta) = (r.omega.re, r.omega.im);
            let (ar, br) = (at, at + 1);
            let first = at;
            let mut rx_re = ac[ar].clone();
            rx_re.axpy(sc(1.0), &bc[br]);
            rx_re.axpy(sc(-alpha), &vc[ar]);
            let mut ry_re = bc[ar].clone();
            ry_re.axpy(sc(1.0), &ac[br]);
            ry_re.axpy(sc(alpha), &vc[br]);
            let mut parts: Vec<(V, f64)> = Vec::with_capacity(4);
            let (x_im, y_im) = if r.kind == RitzKind::Real {
                at += 2;
                parts.push((rx_re, alpha));
                parts.push((ry_re, -alpha));
                (None, None)
            } else {
                let (ai, bi) = (at + 2, at + 3);
                at += 4;
                rx_re.axpy(sc(beta), &vc[ai]);
                ry_re.axpy(sc(-beta), &vc[bi]);
                let mut rx_im = ac[ai].clone();
                rx_im.axpy(sc(1.0), &bc[bi]);
                rx_im.axpy(sc(-alpha), &vc[ai]);
                rx_im.axpy(sc(-beta), &vc[ar]);
                let mut ry_im = bc[ai].clone();
                ry_im.axpy(sc(1.0), &ac[bi]);
                ry_im.axpy(sc(alpha), &vc[bi]);
                ry_im.axpy(sc(beta), &vc[br]);
                parts.push((rx_re, alpha));
                parts.push((ry_re, -alpha));
                parts.push((rx_im, alpha));
                parts.push((ry_im, -alpha));
                (Some(vc[ai].clone()), Some(vc[bi].clone()))
            };
            let residual = parts
                .iter()
                .map(|(p, _)| p.norm().powi(2))
                .sum::<f64>()
                .sqrt();
            let anomalous = role == Role::Anomaly;
            if anomalous
                && let AnomalyRule::Gate(f) = opts.anomalies
                && residual > f * r.omega.norm()
            {
                info.gated_out += 1;
                continue;
            }
            let mode = PairedMode {
                omega: r.omega,
                residual,
                krein: r.krein,
                overlap: (s > 0).then(|| overlap_of(r)),
                rho_a: rho_a_of(r),
                window_index: pos_all.iter().position(|&p| std::ptr::eq(p, r)),
                x_re: vc[ar].clone(),
                y_re: vc[br].clone(),
                x_im,
                y_im,
            };
            if role == Role::Report {
                untargeted.push(mode);
                continue;
            }
            kept_cs.extend(cs[first..at].iter().cloned());
            if anomalous && freeze {
                info.frozen += 1;
            } else {
                let bound = if role == Role::Wanted {
                    opts.tol * scale_pos
                } else {
                    opts.tol * r.omega.norm()
                };
                let conv = residual <= bound;
                conv_all &= conv;
                if !conv {
                    if anomalous {
                        info.anomaly_corrections += parts.len();
                    }
                    corrections.extend(parts);
                }
            }
            if r.kind == RitzKind::Complex {
                cplx_modes.push(mode);
            } else {
                modes.push(mode);
            }
        }
        last_real = modes;
        last_cplx = cplx_modes;
        last_untargeted = untargeted;
        info.phases.residual += t_resid.elapsed().as_secs_f64();
        if trace {
            let list = |f: &dyn Fn(&PairedMode<V>) -> String| {
                last_real.iter().map(f).collect::<Vec<_>>().join(",")
            };
            let (off, _) = offset_of(&basis.ah, &basis.bh, m, false);
            eprintln!(
                "  paired davidson outer={outer} m={m} built={} ω²=[{}] res=[{}] krein=[{}] \
                 overlap=[{}] rho_a=[{}] complex={} gated_out={} frozen={} untargeted_below={} \
                 offset=(apb {}, amb {}, a {}) drops={}",
                info.built,
                list(&|p| format!("{:.12e}", p.omega2().re)),
                list(&|p| format!("{:.2e}", p.residual)),
                list(&|p| format!("{:+.3}", p.krein)),
                list(&|p| p.overlap.map_or("-".into(), |o| format!("{o:.3}"))),
                list(&|p| format!("{:.4e}", p.rho_a)),
                last_cplx.len(),
                info.gated_out,
                info.frozen,
                last_untargeted.len(),
                off.neg_apb,
                off.neg_amb,
                off.neg_a,
                info.offset_drops
            );
        }
        if conv_all && pos.len() >= k.min(m) && m >= k {
            info.converged = true;
            break;
        }
        if m == space_dim {
            info.exhausted = true;
            info.converged = true;
            break;
        }
        if info.built >= max_basis {
            break;
        }

        if m >= live_cap {
            let t_restart = Instant::now();
            info.restarts += 1;
            let (before, negs) = offset_of(&basis.ah, &basis.bh, m, freeze);
            let mut kept: Vec<Vec<f64>> = kept_cs.clone();
            let reserve = kept_cs.len() + negs.len();
            kept.extend(negs);
            for r in &buffer {
                kept.push(r.a_re.clone());
                kept.push(r.b_re.clone());
            }
            kept.truncate(max_subspace.saturating_sub(1).max(reserve));
            let kept = orthonormal_coefficients(kept);
            let nk = kept.len();
            let new_v = recombine(&basis.vs, &kept, nk);
            let new_a = recombine(&basis.av, &kept, nk);
            let new_b = recombine(&basis.bv, &kept, nk);
            let mut rebuilt: PairedBasis<V> = PairedBasis::new();
            for ((v, av), bv) in new_v.into_iter().zip(new_a).zip(new_b) {
                rebuilt.append(v, av, bv, &qs);
            }
            basis = rebuilt;
            let (after, _) = offset_of(&basis.ah, &basis.bh, basis.len(), false);
            if after.total() < before.total() {
                info.offset_drops += 1;
            }
            live_cap = max_subspace.max(basis.len() + 2 * (k + b));
            info.phases.restart += t_restart.elapsed().as_secs_f64();
        }
        let use_precond = outer >= opts.nwarm && precond.is_some();
        let mut accepted = 0usize;
        for (r, shift) in &corrections {
            if basis.len() >= live_cap.min(space_dim) || info.built >= max_basis {
                break;
            }
            let mut z = r.clone();
            if use_precond && let Some(p) = precond.as_mut() {
                let t = Instant::now();
                p(&mut z, *shift);
                info.phases.precond += t.elapsed().as_secs_f64();
            }
            if push(z, &mut basis, &mut info) {
                accepted += 1;
            }
        }
        if accepted == 0 && use_precond {
            for (r, _) in &corrections {
                if basis.len() >= live_cap.min(space_dim) || info.built >= max_basis {
                    break;
                }
                if push(r.clone(), &mut basis, &mut info) {
                    accepted += 1;
                }
            }
        }
        if accepted == 0 {
            let floor = COLLAPSE_FLOOR * scale_pos.max(cands_scale);
            if corrections.is_empty() || corrections.iter().any(|(r, _)| r.norm() > floor) {
                info.stalled = true;
            } else {
                info.exhausted = true;
                info.converged = true;
            }
            break;
        }
        outer += 1;
    }
    for (index, mode) in last_real.iter().enumerate() {
        let theta = mode.omega2().re;
        if theta >= 0.0 {
            continue;
        }
        let mut span: Vec<V> = Vec::with_capacity(4);
        for v in [
            Some(&mode.x_re),
            mode.x_im.as_ref(),
            Some(&mode.y_re),
            mode.y_im.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            let mut v = v.clone();
            project(&mut v);
            if let Some(u) = orthonormalize_against(v, &span) {
                span.push(u);
            }
        }
        let n = span.len();
        let (fresh, fresh_w) = if n == 0 {
            (f64::NAN, f64::NAN)
        } else {
            let av: Vec<V> = span.iter().map(&mut a_apply).collect();
            let bv: Vec<V> = span
                .iter()
                .map(|u| if b_zero { zero.clone() } else { b_apply(u) })
                .collect();
            info.apply_pairs += n;
            let gram = |img: &[V]| -> Vec<f64> {
                let mut g = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        g[i * n + j] =
                            0.5 * (span[i].dot(&img[j]).re_s() + span[j].dot(&img[i]).re_s());
                    }
                }
                g
            };
            let (ah, bh) = (gram(&av), gram(&bv));
            let fresh = paired_ritz(&ah, &bh, n)
                .iter()
                .map(|r| (r.omega * r.omega).re)
                .min_by(|p, q| (p - theta).abs().total_cmp(&(q - theta).abs()))
                .unwrap_or(f64::NAN);
            let lowest = |g: Vec<f64>| -> f64 {
                dense_self_adjoint_smallest::<f64>(&g, n, 0)
                    .0
                    .first()
                    .copied()
                    .unwrap_or(f64::NAN)
            };
            let apb = ah.iter().zip(&bh).map(|(a, b)| a + b).collect();
            let amb = ah.iter().zip(&bh).map(|(a, b)| a - b).collect();
            (fresh, lowest(apb).min(lowest(amb)))
        };
        info.negative.push(NegativeWitness {
            index,
            theta,
            fresh,
            fresh_w,
        });
    }
    info.offset = offset_of(&basis.ah, &basis.bh, basis.len(), false).0;
    info.phases.total = t_solve.elapsed().as_secs_f64();
    PairedDavidsonOut {
        real: last_real,
        complex: last_cplx,
        untargeted_below: last_untargeted,
        info,
    }
}

/// Preconditioner of [`paired_lowest_davidson`], built once per solve from `D_A`
/// ([`WeingartenCtx::a_diagonal`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PairedPrecond {
    Off,
    /// The fixed unshifted inverse `((D_A)_+ + c)⁻¹`, `c = max(D_A)/κ` — D0's construction
    /// ([`crate::solvers::davidson::precond_fixed_inverse`]) on `D_A` itself rather than on `D_A²`.
    Unshifted {
        kappa: f64,
    },
    /// `(D_A − s)⁻¹` with the Davidson denominator clamp, `s = ±Re ω` per half.
    Shifted,
}

/// How [`paired_lowest_davidson`] keeps the basis and the images on the tangent space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairedHook {
    /// `project∘to_vector` + radial deflation on every candidate and every image — the signed
    /// kernel's hook, OBLIQUE in the coefficient metric off the gauge-fixed subspace (owl 1220382
    /// ran this).
    Oblique,
    /// The coefficient-ORTHOGONAL gauge fix + radial deflation (`Q`, the shared `clean_tangent`)
    /// on candidates (before AND after their MGS) and on each
    /// apply's input and image, and the complement term `μ·(v − Q v)` on the `A` image, `μ =
    /// 10·(1 + max|D_A|)`: the operator is then `[QAQ + μ(I−Q), QBQ; QBQ, QAQ + μ(I−Q)]`, one
    /// symmetric pencil on the whole coefficient space whose complement sits at `ω = μ`, so MGS
    /// roundoff amplified off the subspace cannot enter the window (the tda-solver campaign's
    /// finding for the block-Davidson callers, 2026-10-21).
    Orthogonal,
}

pub(crate) fn realified(complex: bool) -> usize {
    if complex { 2 } else { 1 }
}

fn tt_clean<T: Scalar>(
    hook: PairedHook,
    base_l: &std::sync::Arc<crate::vector::TtVector<T>>,
    base_r: &std::sync::Arc<crate::vector::TtVector<T>>,
    radial: Option<&TtTangent<T>>,
    v: &mut TtTangent<T>,
) {
    match hook {
        PairedHook::Oblique => reproject_tangent(base_l, base_r, radial, v),
        PairedHook::Orthogonal => clean_tangent(radial, v),
    }
}

pub(crate) fn paired_image<V: InnerProductVector>(
    clean: impl Fn(&mut V),
    apply: impl FnOnce(&V) -> V,
    mu: f64,
    v: &V,
    a_side: bool,
) -> V {
    let mut q = v.clone();
    clean(&mut q);
    let mut img = apply(&q);
    clean(&mut img);
    if a_side {
        add_complement_shift(&mut img, v, q, mu);
    }
    img
}

fn tt_image<T: Scalar>(
    ctx: &WeingartenCtx<'_, T>,
    hook: PairedHook,
    radial: Option<&TtTangent<T>>,
    mu: f64,
    v: &TtTangent<T>,
    a_side: bool,
) -> TtTangent<T> {
    let (base_l, base_r) = ctx.shared_bases();
    paired_image(
        |w: &mut TtTangent<T>| tt_clean(hook, base_l, base_r, radial, w),
        |q| {
            if a_side {
                ctx.a_apply(q)
            } else {
                ctx.b_apply(q)
            }
        },
        mu,
        v,
        a_side,
    )
}

pub(crate) fn paired_mu<V: InnerProductVector>(hook: PairedHook, d_a: &V) -> f64 {
    if hook == PairedHook::Orthogonal {
        complement_shift(diag_inf_scale(d_a))
    } else {
        0.0
    }
}

pub(crate) fn paired_precond<V: InnerProductVector>(
    precond: PairedPrecond,
    d_a: V,
) -> Option<impl FnMut(&mut V, f64)> {
    let model = match precond {
        PairedPrecond::Off => None,
        PairedPrecond::Unshifted { kappa } => {
            let mut d = d_a;
            crate::solvers::davidson::precond_fixed_inverse(&mut d, kappa);
            Some((false, d))
        }
        PairedPrecond::Shifted => Some((true, d_a)),
    };
    model.map(|(shifted, d)| {
        move |t: &mut V, s: f64| {
            if shifted {
                crate::solvers::davidson::precond(t, &d, s);
            } else {
                t.zip_coeffs_mut(&d, |ti, di| *ti *= di);
            }
        }
    })
}

/// The lowest RPA modes at `ctx`'s point by the paired-basis Davidson (module doc), TT layer:
/// `A` = [`WeingartenCtx::a_apply`], `B` = [`WeingartenCtx::b_apply`] (so a
/// [`CurvatureReg`](super::CurvatureReg) on the context is honoured), `space_dim` = the tangent
/// dimension minus the radial mode, the tangent-space discipline chosen by `hook`
/// ([`PairedHook`]). `tda` takes `B ≡ 0` (the Tamm–Dancoff problem `eig(A)`, no `B` apply);
/// `target` empty = the `k` lowest, else the `k` pairs of largest overlap with it (module doc).
#[allow(clippy::too_many_arguments)]
pub fn paired_lowest_davidson<T: Scalar>(
    ctx: &WeingartenCtx<'_, T>,
    v0s: &[TtTangent<T>],
    target: &[TtTangent<T>],
    k: usize,
    precond: PairedPrecond,
    tda: bool,
    hook: PairedHook,
    opts: &PairedDavidsonOpts,
) -> PairedDavidsonOut<TtTangent<T>> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = radial_mode(ctx);
    let radial = radial.as_ref();
    let space_dim = realified(T::IS_COMPLEX)
        * (crate::core::tangent::tangent_dim(base_l) - radial.map_or(0, |_| 1));
    let d_a = ctx.a_diagonal();
    let mu = paired_mu(hook, &d_a);
    let precond_fn = paired_precond(precond, d_a);
    let clean = |v: &mut TtTangent<T>| tt_clean(hook, base_l, base_r, radial, v);
    omega_davidson_paired(
        |v| tt_image(ctx, hook, radial, mu, v, true),
        |v| tt_image(ctx, hook, radial, mu, v, false),
        clean,
        precond_fn,
        tda,
        v0s,
        target,
        k,
        space_dim,
        opts,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::super::deflate_radial;
    use super::*;
    use crate::solvers::steihaug::testvec::{DVec, mat_apply};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn orthogonal(n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<Vec<f64>> {
        let mut q: Vec<Vec<f64>> = Vec::with_capacity(n);
        while q.len() < n {
            let mut v: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            for _ in 0..2 {
                for u in &q {
                    let c: f64 = u.iter().zip(&v).map(|(a, b)| a * b).sum();
                    v.iter_mut().zip(u).for_each(|(x, y)| *x -= c * y);
                }
            }
            let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            if nv > 1e-8 {
                q.push(v.iter().map(|a| a / nv).collect());
            }
        }
        q
    }

    fn sym(evals: &[f64], rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
        let n = evals.len();
        let q = orthogonal(n, rng);
        let mut a = vec![0.0; n * n];
        for (c, &l) in evals.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] += l * q[c][i] * q[c][j];
                }
            }
        }
        a
    }

    fn spread(n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64)
            .collect()
    }

    fn ab_from_rw(r: &[f64], w: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let a = r.iter().zip(w).map(|(p, q)| 0.5 * (p + q)).collect();
        let b = r.iter().zip(w).map(|(p, q)| 0.5 * (p - q)).collect();
        (a, b)
    }

    fn seeds(n: usize, b: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<DVec> {
        (0..b)
            .map(|_| DVec((0..n).map(|_| rng.random_range(-1.0..1.0)).collect()))
            .collect()
    }

    fn oracle(a: &[f64], b: &[f64], n: usize) -> (Vec<f64>, Vec<C64>) {
        let n2 = 2 * n;
        let mut h = vec![0.0; n2 * n2];
        for i in 0..n {
            for j in 0..n {
                h[i * n2 + j] = a[i * n + j];
                h[i * n2 + n + j] = b[i * n + j];
                h[(n + i) * n2 + j] = -b[i * n + j];
                h[(n + i) * n2 + n + j] = -a[i * n + j];
            }
        }
        let (ev, _) = dense_eigen_general::<f64>(&h, n2);
        let scale = ev.iter().fold(0.0f64, |m, z| m.max(z.norm()));
        let t = 1e-10 * scale;
        let mut real: Vec<f64> = Vec::new();
        let mut cplx: Vec<C64> = Vec::new();
        for z in &ev {
            if z.im.abs() <= t {
                if z.re > 0.0 {
                    real.push(z.re * z.re);
                }
            } else if z.re.abs() <= t {
                if z.im > 0.0 {
                    real.push(-z.im * z.im);
                }
            } else if z.re > 0.0 && z.im > 0.0 {
                cplx.push(*z * *z);
            }
        }
        real.sort_by(f64::total_cmp);
        cplx.sort_by(|p, q| p.re.total_cmp(&q.re));
        (real, cplx)
    }

    fn run(
        a: &[f64],
        b: &[f64],
        n: usize,
        v0s: &[DVec],
        k: usize,
        opts: &PairedDavidsonOpts,
    ) -> PairedDavidsonOut<DVec> {
        omega_davidson_paired(
            |v: &DVec| mat_apply(a, n, v),
            |v: &DVec| mat_apply(b, n, v),
            |_: &mut DVec| {},
            None::<fn(&mut DVec, f64)>,
            false,
            v0s,
            &[],
            k,
            n,
            opts,
        )
    }

    fn assert_wanted(
        label: &str,
        out: &PairedDavidsonOut<DVec>,
        real: &[f64],
        cplx: &[C64],
        k: usize,
        tol_abs: f64,
    ) {
        let negs: Vec<f64> = real.iter().copied().filter(|&x| x < 0.0).collect();
        let pos: Vec<f64> = real.iter().copied().filter(|&x| x >= 0.0).take(k).collect();
        let want: Vec<f64> = negs.iter().chain(&pos).copied().collect();
        let got = out.omega2();
        println!(
            "{label}: converged {} exhausted {} built {} restarts {} ω² {got:?} complex {:?}",
            out.info.converged,
            out.info.exhausted,
            out.info.built,
            out.info.restarts,
            out.complex.iter().map(|m| m.omega2()).collect::<Vec<_>>()
        );
        assert!(out.info.converged, "{label}: not converged");
        assert_eq!(
            got.len(),
            want.len(),
            "{label}: wanted set {got:?} vs {want:?}"
        );
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= tol_abs,
                "{label}: ω²[{j}] {g} vs dense {w}"
            );
        }
        let window = pos.last().copied().unwrap_or(f64::INFINITY);
        let dense_in: Vec<&C64> = cplx.iter().filter(|z| z.re <= window).collect();
        assert_eq!(
            out.complex.len(),
            dense_in.len(),
            "{label}: complex pairs in the window"
        );
        for (m, z) in out.complex.iter().zip(&dense_in) {
            assert!(
                (m.omega2() - **z).norm() <= tol_abs,
                "{label}: complex {} vs dense {z}",
                m.omega2()
            );
        }
    }

    #[test]
    fn paired_dense_spectrum_on_every_inertia_class() {
        let n = 48;
        let k = 3;
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 2000,
            max_subspace: 24,
            ..PairedDavidsonOpts::default()
        };
        let cases: [(&str, u64, Vec<f64>, Vec<f64>); 4] = [
            ("C1 PD", 0xC1_C1, spread(n, 0.5, 5.0), spread(n, 0.3, 4.0)),
            ("C3 W indefinite", 0xC3_C3, spread(n, 0.5, 5.0), {
                let mut w = spread(n, 0.25, 4.0);
                w[0] = -0.6;
                w[1] = -0.35;
                w[2] = -0.2;
                w
            }),
            (
                "C4 R indefinite",
                0xC4_C4,
                {
                    let mut r = spread(n, 0.5, 5.0);
                    r[0] = -0.8;
                    r[1] = -0.3;
                    r
                },
                spread(n, 0.3, 4.0),
            ),
            (
                "C5 both indefinite",
                0xC5_C5,
                {
                    let mut r = spread(n, 0.5, 5.0);
                    r[0] = -0.7;
                    r[1] = -0.25;
                    r
                },
                {
                    let mut w = spread(n, 0.3, 4.0);
                    w[0] = -0.5;
                    w[1] = -0.2;
                    w
                },
            ),
        ];
        for (label, seed, rev, wev) in cases {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let r = sym(&rev, &mut rng);
            let w = sym(&wev, &mut rng);
            let (a, b) = ab_from_rw(&r, &w);
            let v0s = seeds(n, k + 2, &mut rng);
            let (real, cplx) = oracle(&a, &b, n);
            let out = run(&a, &b, n, &v0s, k, &opts);
            let scale = real
                .iter()
                .map(|x| x.abs())
                .chain(cplx.iter().map(|z| z.norm()))
                .fold(0.0f64, f64::max);
            assert!(out.info.restarts >= 1, "{label}: the fixture must restart");
            assert_wanted(label, &out, &real, &cplx, k, 1e-8 * scale);
        }
    }

    #[test]
    fn paired_exhaustion_is_the_full_pencil() {
        let n = 32;
        let k = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE7_E7);
        let mut rev = spread(n, 0.5, 5.0);
        rev[0] = -0.9;
        rev[1] = -0.4;
        let mut wev = spread(n, 0.3, 4.0);
        wev[0] = -0.6;
        let r = sym(&rev, &mut rng);
        let w = sym(&wev, &mut rng);
        let (a, b) = ab_from_rw(&r, &w);
        let v0s = seeds(n, 2, &mut rng);
        let opts = PairedDavidsonOpts {
            tol: 1e-300,
            max_basis: 10 * n,
            max_subspace: n,
            nwarm: 0,
            positive_window: false,
            anomalies: AnomalyRule::Chase,
            rank: WantedRank::Omega,
        };
        let out = run(&a, &b, n, &v0s, k, &opts);
        assert!(out.info.exhausted, "the tolerance forces exhaustion");
        let (real, cplx) = oracle(&a, &b, n);
        let scale = real
            .iter()
            .map(|x| x.abs())
            .chain(cplx.iter().map(|z| z.norm()))
            .fold(0.0f64, f64::max);
        assert_wanted("exhaustion", &out, &real, &cplx, k, 1e-10 * scale);
        let rmax = out.real.iter().fold(0.0f64, |m, p| m.max(p.residual));
        assert!(rmax <= 1e-10 * scale.sqrt(), "exhausted residual {rmax:e}");
    }

    #[test]
    fn paired_ritz_values_bound_a_positive_type_window() {
        use crate::core::factor::dense_self_adjoint_smallest;
        let n = 24;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0_0D);
        let t_of = |a: &[f64], b: &[f64], m: usize, om: f64| -> Vec<f64> {
            let (ev, q) = dense_self_adjoint_smallest::<f64>(a, m, m);
            let mut s = vec![0.0; m * m];
            for (c, qc) in q.iter().enumerate() {
                let bq: Vec<f64> = (0..m)
                    .map(|i| (0..m).map(|j| b[i * m + j] * qc[j]).sum())
                    .collect();
                let w = 1.0 / (ev[c] + om);
                for i in 0..m {
                    for j in 0..m {
                        s[i * m + j] += w * bq[i] * bq[j];
                    }
                }
            }
            (0..m * m)
                .map(|ij| a[ij] - if ij / m == ij % m { om } else { 0.0 } - s[ij])
                .collect()
        };
        let nneg = |s: &[f64], m: usize| -> usize {
            let (ev, _) = dense_self_adjoint_smallest::<f64>(s, m, 0);
            ev.iter().filter(|&&e| e < 0.0).count()
        };
        let mut checked = 0usize;
        let mut deficit_violations = 0usize;
        let mut anomalous_draws = 0usize;
        for _ in 0..40 {
            let a = sym(
                &(0..n)
                    .map(|_| rng.random_range(0.3..3.0))
                    .collect::<Vec<_>>(),
                &mut rng,
            );
            let mut b: Vec<f64> = (0..n * n)
                .map(|_| 0.05 * rng.random_range(-1.0..1.0))
                .collect();
            for i in 0..n {
                for j in 0..i {
                    let s = 0.5 * (b[i * n + j] + b[j * n + i]);
                    b[i * n + j] = s;
                    b[j * n + i] = s;
                }
            }
            for _ in 0..3 {
                let mut u: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
                let nu = u.iter().map(|x| x * x).sum::<f64>().sqrt();
                u.iter_mut().for_each(|x| *x /= nu);
                let auu: f64 = (0..n)
                    .map(|i| (0..n).map(|j| u[i] * a[i * n + j] * u[j]).sum::<f64>())
                    .sum();
                let c = rng.random_range(0.9..1.3)
                    * auu
                    * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
                for i in 0..n {
                    for j in 0..n {
                        b[i * n + j] += c * u[i] * u[j];
                    }
                }
            }
            let full = paired_ritz(&a, &b, n);
            let anom: Vec<f64> = full
                .iter()
                .filter(|r| r.kind != RitzKind::Real || r.krein < 0.0)
                .map(|r| r.omega.norm())
                .collect();
            if !anom.is_empty() {
                anomalous_draws += 1;
            }
            let om_lo = anom.iter().fold(0.0f64, |m, &x| m.max(x)) + 1e-3;
            let exact: Vec<f64> = full
                .iter()
                .filter(|r| r.kind == RitzKind::Real && r.krein > 0.0 && r.omega.re > om_lo)
                .map(|r| r.omega.re)
                .collect();
            let tfull = t_of(&a, &b, n, om_lo);
            let n_lo = nneg(&tfull, n);
            let (tev, tq) = dense_self_adjoint_smallest::<f64>(&tfull, n, n);
            let mut cols: Vec<Vec<f64>> = (0..n)
                .filter(|&c| tev[c] < 0.0)
                .map(|c| tq[c].clone())
                .collect();
            let seeded = rng.random_bool(0.5);
            if !seeded {
                cols.clear();
            }
            cols.extend(orthogonal(n, &mut rng));
            let q = {
                let mut q: Vec<Vec<f64>> = Vec::new();
                for v in cols {
                    let mut v = v;
                    for _ in 0..2 {
                        for u in &q {
                            let c: f64 = u.iter().zip(&v).map(|(x, y)| x * y).sum();
                            v.iter_mut().zip(u).for_each(|(x, y)| *x -= c * y);
                        }
                    }
                    let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if nv > 1e-8 {
                        q.push(v.iter().map(|x| x / nv).collect());
                    }
                }
                q
            };
            let mut prev: Option<Vec<f64>> = None;
            for m in (n_lo + 4..=n).step_by(4) {
                let proj = |g: &[f64]| -> Vec<f64> {
                    let mut p = vec![0.0; m * m];
                    for i in 0..m {
                        for j in 0..m {
                            p[i * m + j] = (0..n)
                                .map(|s| {
                                    q[i][s] * (0..n).map(|t| g[s * n + t] * q[j][t]).sum::<f64>()
                                })
                                .sum();
                        }
                    }
                    p
                };
                let (ah, bh) = (proj(&a), proj(&b));
                let matched = nneg(&t_of(&ah, &bh, m, om_lo), m) == n_lo;
                let ritz: Vec<f64> = paired_ritz(&ah, &bh, m)
                    .iter()
                    .filter(|r| r.kind == RitzKind::Real && r.krein > 0.0 && r.omega.re > om_lo)
                    .map(|r| r.omega.re)
                    .collect();
                for (j, (&t, &e)) in ritz.iter().zip(&exact).enumerate() {
                    if matched {
                        checked += 1;
                        assert!(
                            t >= e - 1e-9,
                            "Ritz {j} = {t} below exact {e} (m = {m}, offset captured)"
                        );
                    } else if t < e - 1e-9 {
                        deficit_violations += 1;
                    }
                }
                if matched {
                    if let Some(p) = &prev {
                        for (j, (&t, &s)) in ritz.iter().zip(p).enumerate() {
                            assert!(
                                t <= s + 1e-9,
                                "Ritz {j} rose {s} → {t} as the basis grew to {m}"
                            );
                        }
                    }
                    prev = Some(ritz);
                }
            }
        }
        println!(
            "bound pin: {checked} matched Ritz values checked, {deficit_violations} violations \
             with a deficit, {anomalous_draws}/40 draws with an anomalous mode"
        );
        assert!(checked >= 50, "too few matched comparisons: {checked}");
        assert!(
            anomalous_draws >= 1,
            "no draw had an anomalous mode: the window is vacuous"
        );
        assert!(
            deficit_violations >= 1,
            "no deficit basis violated the bound: the offset hypothesis is untested"
        );
    }

    #[test]
    fn paired_b_zero_is_the_tamm_dancoff_problem() {
        use crate::core::factor::dense_self_adjoint_smallest;
        let n = 30;
        let k = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7DA);
        let a = sym(&spread(n, 0.4, 3.0), &mut rng);
        let v0s = seeds(n, k + 1, &mut rng);
        let mut b_calls = 0usize;
        let out = omega_davidson_paired(
            |v: &DVec| mat_apply(&a, n, v),
            |v: &DVec| {
                b_calls += 1;
                v.clone()
            },
            |_: &mut DVec| {},
            None::<fn(&mut DVec, f64)>,
            true,
            &v0s,
            &[],
            k,
            n,
            &PairedDavidsonOpts {
                tol: 1e-10,
                ..PairedDavidsonOpts::default()
            },
        );
        let (ev, _) = dense_self_adjoint_smallest::<f64>(&a, n, 0);
        assert!(out.info.converged);
        assert_eq!(b_calls, 0, "b_zero must not call b_apply");
        for (j, m) in out.real.iter().enumerate() {
            assert!(
                (m.omega.re - ev[j]).abs() <= 1e-9,
                "ω[{j}] {} vs eig(A) {}",
                m.omega.re,
                ev[j]
            );
            assert!((m.krein - 1.0).abs() <= 1e-12, "Krein margin {}", m.krein);
        }
    }

    fn tt_grams(ctx: &WeingartenCtx<'_, f64>) -> (Vec<f64>, Vec<f64>, usize) {
        use crate::core::tangent::tangent_basis;
        use crate::solvers::hess::hessian_grams;
        let (bl, br) = ctx.bases();
        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let (a, b) = hessian_grams(ctx, &basis);
        (a, b, n)
    }

    fn tt_seeds(ctx: &WeingartenCtx<'_, f64>, count: usize, seed: u64) -> Vec<TtTangent<f64>> {
        let (bl, br) = ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                TtTangent::project(
                    bl,
                    br,
                    &crate::vector::tt_randn(bl.space, &bl.ranks, &mut rng),
                )
            })
            .collect()
    }

    pub(crate) fn assert_paired_wanted<V>(
        label: &str,
        out: &PairedDavidsonOut<V>,
        a: &[f64],
        b: &[f64],
        n: usize,
        k: usize,
        tol: f64,
    ) {
        let (real, cplx) = oracle(a, b, n);
        let negs: Vec<f64> = real.iter().copied().filter(|&x| x < 0.0).collect();
        let pos: Vec<f64> = real.iter().copied().filter(|&x| x >= 0.0).take(k).collect();
        let want: Vec<f64> = negs.iter().chain(&pos).copied().collect();
        let got = out.omega2();
        let rmax = out.real.iter().fold(0.0f64, |m, p| m.max(p.residual));
        println!(
            "{label}: n {n} converged {} exhausted {} built {} restarts {} max residual {rmax:.2e}\n  ω² {got:?}\n  dense {want:?}\n  complex {:?} (dense in window {})",
            out.info.converged,
            out.info.exhausted,
            out.info.built,
            out.info.restarts,
            out.complex.iter().map(|m| m.omega2()).collect::<Vec<_>>(),
            cplx.iter()
                .filter(|z| z.re <= pos.last().copied().unwrap_or(f64::INFINITY))
                .count()
        );
        assert!(out.info.converged, "{label}: not converged");
        assert_eq!(got.len(), want.len(), "{label}: wanted set");
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= tol * w.abs().max(1.0),
                "{label}: ω²[{j}] {g} vs dense {w}"
            );
        }
        let window = pos.last().copied().unwrap_or(f64::INFINITY);
        let dense_in: Vec<&C64> = cplx.iter().filter(|z| z.re <= window).collect();
        assert_eq!(
            out.complex.len(),
            dense_in.len(),
            "{label}: complex pairs in the window"
        );
        for (m, z) in out.complex.iter().zip(&dense_in) {
            assert!(
                (m.omega2() - **z).norm() <= tol * z.norm().max(1.0),
                "{label}: complex {} vs dense {z}",
                m.omega2()
            );
        }
    }

    #[test]
    fn paired_live_tangent_space_matches_the_dense_grams() {
        use crate::qn::QnSpace;
        use crate::round::RoundMode;
        use crate::solvers::hess::weingarten_operator;
        use crate::test_util::{generic_truncated_h, randomized_state};
        use crate::vector::TtVector;
        let space = QnSpace::new(4, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let h = generic_truncated_h::<f64>(space);
        let ctx = weingarten_operator(&h, &x);
        let (a, b, n) = tt_grams(&ctx);
        let k = 3;
        let v0s = tt_seeds(&ctx, k + 2, 0x81E2);
        let opts = PairedDavidsonOpts {
            tol: 1e-11,
            max_basis: 400,
            max_subspace: n,
            ..PairedDavidsonOpts::default()
        };
        for hook in [PairedHook::Orthogonal, PairedHook::Oblique] {
            for pre in [
                PairedPrecond::Off,
                PairedPrecond::Unshifted { kappa: 1e3 },
                PairedPrecond::Shifted,
            ] {
                let out = paired_lowest_davidson(&ctx, &v0s, &[], k, pre, false, hook, &opts);
                assert_paired_wanted(&format!("C7 {hook:?} {pre:?}"), &out, &a, &b, n, k, 1e-8);
            }
        }
    }

    #[test]
    fn paired_kernel_behind_the_signed_entry_is_the_kernel() {
        use crate::qn::QnSpace;
        use crate::round::RoundMode;
        use crate::solvers::hess::{
            RpaKernel, RpaMetric, RpaModeKind, RpaReport, SympPrecond, SymplecticDavidsonOpts,
            symplectic_lowest_signed_davidson, weingarten_operator,
        };
        use crate::test_util::{generic_truncated_h, randomized_state};
        use crate::vector::TtVector;
        let space = QnSpace::new(4, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let h = generic_truncated_h::<f64>(space);
        let ctx = weingarten_operator(&h, &x);
        let (_, _, n) = tt_grams(&ctx);
        let radial = super::super::radial_mode(&ctx);
        let k = 3;
        let v0s = tt_seeds(&ctx, k + 2, 0x81E2);
        for precond in [SympPrecond::Off, SympPrecond::Auto] {
            let opts = SymplecticDavidsonOpts {
                tol: 1e-11,
                max_basis: 400,
                max_subspace: n,
                precond,
                ..SymplecticDavidsonOpts::default()
            };
            let (route_pre, popts) = super::super::paired_route(&opts).expect("a paired route");
            assert_eq!(popts.max_basis, opts.max_basis);
            let direct = paired_lowest_davidson(
                &ctx,
                &v0s,
                &[],
                k,
                route_pre,
                false,
                PairedHook::Orthogonal,
                &popts,
            );
            let (so, vecs) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
            let tag = format!("{precond:?}");
            let RpaReport::Paired { offset, .. } = &so.report else {
                panic!("{tag}: the paired kernel did not run")
            };
            assert_eq!(*offset, direct.info.offset, "{tag}: offset");
            assert_eq!(so.side(), RpaMetric::Ii, "{tag}");
            assert_eq!(so.omega2(), direct.omega2(), "{tag}: entry ≠ kernel");
            assert_eq!(so.iterations, direct.info.built, "{tag}");
            assert_eq!(so.applies, direct.info.apply_pairs, "{tag}");
            assert_eq!(
                vecs.len(),
                direct.real.len() + direct.complex.len(),
                "{tag}: one amplitude pair per mode"
            );
            let scale = so.omega2().iter().fold(0.0f64, |m, w| m.max(w.abs()));
            for (j, ((m, rm), mode)) in vecs.iter().zip(&so.modes).zip(&direct.real).enumerate() {
                let w2 = rm.omega2.re;
                if rm.kind == RpaModeKind::Real {
                    assert_eq!(rm.krein, mode.krein, "{tag}: Krein margin {j}");
                }
                assert_eq!(rm.residual, mode.residual, "{tag}: residual {j}");
                let c = m.x.norm() / mode.x_re.norm();
                let mut dx = m.x.clone();
                dx.axpy(-c, &mode.x_re);
                let mut dy = m.y.clone();
                dy.axpy(-c, &mode.y_re);
                assert!(
                    dx.norm() <= 1e-12 * m.x.norm()
                        && dy.norm() <= 1e-12 * m.x.norm()
                        && m.x_im.is_some() == mode.x_im.is_some(),
                    "{tag}: mode {j}'s amplitudes are not the kernel's pair"
                );
                for metric in [RpaMetric::Ii, RpaMetric::Rr] {
                    let q = &m.metric_side(metric);
                    let rr = metric == RpaMetric::Rr;
                    let mut w = if rr {
                        ctx.hpp_apply(q)
                    } else {
                        ctx.hmm_apply(q)
                    };
                    deflate_radial(radial.as_ref(), &mut w);
                    let mut r = if rr {
                        ctx.hmm_apply(&w)
                    } else {
                        ctx.hpp_apply(&w)
                    };
                    deflate_radial(radial.as_ref(), &mut r);
                    r.axpy(-w2, q);
                    let rel = r.norm() / (scale * q.norm());
                    assert!(
                        rel <= 1e-6,
                        "{tag} {metric:?}: mode {j} ω² {w2}: product residual {rel:.2e}"
                    );
                }
            }
        }
        for (label, opts) in [
            (
                "kernel Signed",
                SymplecticDavidsonOpts {
                    kernel: RpaKernel::signed(RpaMetric::Ii),
                    ..SymplecticDavidsonOpts::default()
                },
            ),
            (
                "instrument",
                SymplecticDavidsonOpts {
                    instrument: true,
                    ..SymplecticDavidsonOpts::default()
                },
            ),
            (
                "survival guard",
                SymplecticDavidsonOpts {
                    guard_survival: 1e-5,
                    ..SymplecticDavidsonOpts::default()
                },
            ),
        ] {
            assert!(super::super::paired_route(&opts).is_none(), "{label}");
            let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
            assert!(
                matches!(so.report, RpaReport::Signed { .. }),
                "{label}: fell back"
            );
        }
    }

    #[test]
    fn selective_ranks_behind_the_signed_entries_are_the_kernel() {
        use crate::qn::QnSpace;
        use crate::round::RoundMode;
        use crate::solvers::hess::{
            SympPrecond, SymplecticDavidsonOpts, symplectic_lowest_signed_davidson,
            symplectic_lowest_signed_davidson_targeted, weingarten_operator,
        };
        use crate::test_util::{generic_truncated_h, randomized_state};
        use crate::vector::TtVector;
        let space = QnSpace::new(5, 2, 2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let h = generic_truncated_h::<f64>(space);
        let ctx = weingarten_operator(&h, &x);
        let n = crate::core::tangent::tangent_dim(ctx.shared_bases().0);
        let k = 3;
        let v0s = tt_seeds(&ctx, k + 2, 0x81E2);
        let target = tt_seeds(&ctx, 2, 0x81E3);
        let base = SymplecticDavidsonOpts {
            tol: 1e-8,
            max_basis: 400,
            max_subspace: n,
            precond: SympPrecond::Off,
            ..SymplecticDavidsonOpts::default()
        };
        let mut nonempty = 0;
        for (label, rank, tgt) in [
            ("AExcess(1.05)", WantedRank::AExcess(1.05), &[][..]),
            ("target, Omega", WantedRank::Omega, &target[..]),
            (
                "target, TargetGateOrExcess(0.9, 1.05)",
                WantedRank::TargetGateOrExcess(0.9, 1.05),
                &target[..],
            ),
        ] {
            let opts = SymplecticDavidsonOpts { rank, ..base };
            let (pre, popts) = super::super::paired_route(&opts).expect("a paired route");
            assert_eq!(popts.rank, rank, "{label}: the route passes the rank");
            let direct = paired_lowest_davidson(
                &ctx,
                &v0s,
                tgt,
                k,
                pre,
                false,
                PairedHook::Orthogonal,
                &popts,
            );
            let (so, _) = if tgt.is_empty() {
                symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts)
            } else {
                symplectic_lowest_signed_davidson_targeted(&ctx, &v0s, tgt, k, &opts)
            };
            assert_eq!(so.omega2(), direct.omega2(), "{label}: entry ≠ kernel");
            assert_eq!(so.iterations, direct.info.built, "{label}");
            assert_eq!(so.applies, direct.info.apply_pairs, "{label}");
            assert_eq!(
                so.passed_over.len(),
                direct.untargeted_below.len(),
                "{label}: passed over"
            );
            for (p, m) in so.passed_over.iter().zip(&direct.untargeted_below) {
                assert_eq!(p.omega, m.omega.re, "{label}");
                assert_eq!(p.krein, m.krein, "{label}");
                assert_eq!(p.rho_a_over_omega, m.rho_a / m.omega.re, "{label}");
                assert_eq!(p.residual, m.residual, "{label}");
            }
            println!(
                "{label}: tangent dim {n}, ω² {:?}, passed over {:?}",
                so.omega2(),
                so.passed_over
            );
            nonempty += usize::from(!so.passed_over.is_empty());
        }
        assert!(nonempty > 0, "vacuous: no arm passed a pair over");
    }

    #[test]
    #[should_panic(expected = "needs the paired route")]
    fn selective_rank_needs_the_paired_route() {
        use crate::solvers::hess::{RpaKernel, RpaMetric, SymplecticDavidsonOpts};
        let opts = SymplecticDavidsonOpts {
            kernel: RpaKernel::signed(RpaMetric::Ii),
            rank: WantedRank::AExcess(3.0),
            ..SymplecticDavidsonOpts::default()
        };
        let _ = super::super::paired_route(&opts);
    }

    #[test]
    fn paired_report_splits_the_counting_offset() {
        use crate::core::factor::dense_self_adjoint_smallest;
        use crate::qn::QnSpace;
        use crate::round::RoundMode;
        use crate::solvers::hess::{
            RpaReport, SymplecticDavidsonOpts, symplectic_lowest_signed_davidson,
            weingarten_operator,
        };
        use crate::test_util::{generic_truncated_h, randomized_state};
        use crate::vector::TtVector;
        let space = QnSpace::new(4, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let h = generic_truncated_h::<f64>(space);
        let ctx = weingarten_operator(&h, &x);
        let (a, b, n) = tt_grams(&ctx);
        let inertia = |sign: f64| {
            let m: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + sign * q).collect();
            let (ev, _) = dense_self_adjoint_smallest::<f64>(&m, n, 0);
            let scale = ev.iter().fold(1.0f64, |acc, e| acc.max(e.abs()));
            ev.iter().filter(|&&e| e < -1e-10 * scale).count()
        };
        let (apb, amb) = (inertia(1.0), inertia(-1.0));
        assert!(
            apb + amb > 0,
            "fixture premise: an indefinite point ({apb}, {amb})"
        );
        let k = 3;
        let v0s = tt_seeds(&ctx, k + 2, 0x81E2);
        let opts = SymplecticDavidsonOpts {
            tol: 1e-11,
            max_basis: 8 * n,
            max_subspace: n,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        let RpaReport::Paired { offset, .. } = &so.report else {
            panic!("the paired kernel did not run")
        };
        println!(
            "C7 n {n}: dense (n-(A+B), n-(A-B)) = ({apb}, {amb}); report {offset:?}, {} built",
            so.iterations
        );
        assert!(
            so.iterations >= n,
            "premise: the basis spans the space ({} built, n {n})",
            so.iterations
        );
        assert_eq!(
            (offset.neg_apb, offset.neg_amb),
            (apb, amb),
            "the offset split at exhaustion is the dense inertia"
        );
        assert!(
            so.anomalous() && so.omega().is_none(),
            "an indefinite point publishes no ω"
        );
    }

    #[test]
    fn readout_returns_the_pair_each_kernel_solved_for() {
        use crate::core::border_fixture::{BOND, base_state, with_small_schmidt};
        use crate::solvers::hess::{
            ModeCheckOpts, RpaKernel, RpaMetric, RpaReport, SymplecticDavidsonOpts,
            rpa_mode_checks, symplectic_lowest_signed_davidson, weingarten_operator,
        };
        use crate::test_util::generic_truncated_h;
        let x = with_small_schmidt(&base_state(), BOND, 1e-6);
        let h = generic_truncated_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let (_, _, n) = tt_grams(&ctx);
        let radial = super::super::radial_mode(&ctx);
        let k = 3;
        let v0s = tt_seeds(&ctx, k + 2, 0xDEF5);
        for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
            let opts = SymplecticDavidsonOpts {
                tol: if kernel == RpaKernel::Paired {
                    1e-10
                } else {
                    2.5e-11
                },
                max_basis: 8 * n,
                max_subspace: n,
                kernel,
                ..SymplecticDavidsonOpts::default()
            };
            let (so, modes) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
            let paired = matches!(so.report, RpaReport::Paired { .. });
            assert_eq!(
                paired,
                kernel == RpaKernel::Paired,
                "{kernel:?}: the kernel that ran"
            );
            let checks = rpa_mode_checks(&ctx, &so, &modes, &ModeCheckOpts::default());
            let mut gain = 0.0f64;
            let mut compared = 0usize;
            for c in checks.iter().filter(|c| c.omega2 > 0.0) {
                let reported = so.modes[c.index].residual;
                let omega = c.omega2.sqrt();
                let q = modes[c.index].metric_side(RpaMetric::Ii);
                let mut wq = ctx.hmm_apply(&q);
                deflate_radial(radial.as_ref(), &mut wq);
                let mut p = wq.clone();
                p.scale_mut(1.0 / omega);
                let mut r = ctx.hpp_apply(&p);
                deflate_radial(radial.as_ref(), &mut r);
                r.axpy(-omega, &q);
                let eta_q = r.norm() / (omega * (p.norm().powi(2) + q.norm().powi(2)).sqrt());
                println!(
                    "{kernel:?} mode {}: ω² {:.6e} reported {reported:.3e} fresh {:.3e} η {:.2e} | \
                     q-only η {eta_q:.2e}",
                    c.index, c.omega2, c.residual_fresh, c.eta
                );
                let ratio = reported / c.residual_fresh;
                assert!(
                    (0.5..=2.0).contains(&ratio),
                    "{kernel:?} mode {}: reported residual {reported:e} vs fresh {:e}",
                    c.index,
                    c.residual_fresh
                );
                gain = gain.max(eta_q / c.eta);
                compared += 1;
            }
            assert!(compared >= 1, "{kernel:?}: no positive mode to compare");
            if paired {
                assert!(
                    gain >= 10.0,
                    "premise: the q-only rebuild is within {gain:.1}× of the pair — this point \
                     cannot tell the two readouts apart"
                );
            }
        }
    }

    #[test]
    fn paired_certifies_every_imaginary_pair_with_fresh_applies() {
        let (n, t, k) = (32, 4, 2);
        let (a, b) = many_imaginary(n, t, 0xCE_47);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCE_01);
        let v0s = seeds(n, k + 2, &mut rng);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 2000,
            ..PairedDavidsonOpts::default()
        };
        let out = run(&a, &b, n, &v0s, k, &opts);
        assert!(out.info.converged, "fixture premise: converged");
        let negs: Vec<usize> = out
            .real
            .iter()
            .enumerate()
            .filter(|(_, m)| m.omega2().re < 0.0)
            .map(|(j, _)| j)
            .collect();
        assert_eq!(
            negs.len(),
            t,
            "fixture premise: {t} imaginary pairs returned"
        );
        assert_eq!(
            out.info
                .negative
                .iter()
                .map(|w| w.index)
                .collect::<Vec<_>>(),
            negs
        );
        for w in &out.info.negative {
            println!("{w:?}");
            assert!(w.certified(), "uncertified {w:?}");
            assert!(
                (w.fresh - w.theta).abs() <= 1e-8 * w.theta.abs(),
                "fresh {} vs θ {}",
                w.fresh,
                w.theta
            );
            assert!(w.fresh_w < 0.0, "no block witnessed indefinite: {w:?}");
        }
    }

    #[test]
    fn paired_target_on_the_live_tangent_space() {
        use crate::core::tangent::tangent_basis;
        use crate::qn::QnSpace;
        use crate::round::RoundMode;
        use crate::solvers::hess::weingarten_operator;
        use crate::test_util::{generic_truncated_h, randomized_state};
        use crate::vector::TtVector;
        let space = QnSpace::new(4, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let h = generic_truncated_h::<f64>(space);
        let ctx = weingarten_operator(&h, &x);
        let (a, b, n) = tt_grams(&ctx);
        let (bl, br) = ctx.bases();
        let basis = tangent_basis(bl, br, true);
        let k = 3;
        let pos: Vec<PairedRitz> = paired_ritz(&a, &b, n)
            .into_iter()
            .filter(|r| r.kind == RitzKind::Real && r.krein >= 0.0)
            .collect();
        assert!(
            pos.len() >= k + 2,
            "fixture premise: room for interior targets"
        );
        let noise = tt_seeds(&ctx, k, 0x81E3);
        let target: Vec<TtTangent<f64>> = pos[2..2 + k]
            .iter()
            .zip(&noise)
            .map(|(r, e)| {
                let mut t = e.clone();
                t.scale_mut(1e-2 / e.norm());
                for (c, bv) in r.a_re.iter().zip(&basis) {
                    t.axpy(*c, bv);
                }
                t
            })
            .collect();
        let want: Vec<f64> = pos[2..2 + k].iter().map(|r| r.omega.re).collect();
        let v0s = tt_seeds(&ctx, k + 2, 0x81E2);
        let opts = PairedDavidsonOpts {
            tol: 1e-11,
            max_basis: 400,
            max_subspace: n,
            positive_window: true,
            anomalies: AnomalyRule::Freeze,
            ..PairedDavidsonOpts::default()
        };
        let pick = |out: &PairedDavidsonOut<TtTangent<f64>>| -> Vec<f64> {
            out.real
                .iter()
                .filter(|m| m.omega.im == 0.0 && m.krein >= 0.0)
                .map(|m| m.omega.re)
                .collect()
        };
        for hook in [PairedHook::Orthogonal, PairedHook::Oblique] {
            let run_tt = |target: &[TtTangent<f64>]| {
                paired_lowest_davidson(
                    &ctx,
                    &v0s,
                    target,
                    k,
                    PairedPrecond::Off,
                    false,
                    hook,
                    &opts,
                )
            };
            let lowest = run_tt(&[]);
            let out = run_tt(&target);
            let got = pick(&out);
            println!(
                "C7 target {hook:?}: n {n} converged {} built {} targeted {got:?} want {want:?} \
                 overlaps {:?} untargeted below {}; lowest twin {:?}",
                out.info.converged,
                out.info.built,
                out.real.iter().map(|m| m.overlap).collect::<Vec<_>>(),
                out.untargeted_below.len(),
                pick(&lowest)
            );
            assert!(out.info.converged, "{hook:?}: not converged");
            assert_eq!(got.len(), k, "{hook:?}: k targeted pairs");
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 1e-8, "{hook:?}: targeted ω {g} vs {w}");
            }
            assert!(
                (pick(&lowest)[0] - pos[0].omega.re).abs() <= 1e-8,
                "{hook:?}: the lowest-k twin must return the lowest mode"
            );
            assert_eq!(
                out.untargeted_below.len(),
                2,
                "{hook:?}: modes 0–1 untargeted below"
            );
        }
    }

    #[test]
    fn paired_border_point_exhaustion_is_the_dense_pencil() {
        use crate::core::border_fixture::{BOND, base_state, with_small_schmidt};
        use crate::solvers::hess::weingarten_operator;
        use crate::test_util::generic_truncated_h;
        let x = with_small_schmidt(&base_state(), BOND, 1e-6);
        let h = generic_truncated_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let (a, b, n) = tt_grams(&ctx);
        let k = 3;
        let v0s = tt_seeds(&ctx, k + 2, 0xDEF5);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 4 * n,
            max_subspace: n,
            ..PairedDavidsonOpts::default()
        };
        for hook in [PairedHook::Orthogonal, PairedHook::Oblique] {
            let out =
                paired_lowest_davidson(&ctx, &v0s, &[], k, PairedPrecond::Off, false, hook, &opts);
            assert_paired_wanted(&format!("border 1e-6 {hook:?}"), &out, &a, &b, n, k, 1e-8);
        }
    }

    #[test]
    fn paired_orthogonal_hook_is_one_symmetric_operator() {
        use crate::core::border_fixture::base_state;
        use crate::solvers::hess::weingarten_operator;
        use crate::test_util::generic_truncated_h;
        let x = base_state();
        let h = generic_truncated_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.shared_bases();
        let radial = radial_mode(&ctx);
        let radial = radial.as_ref();
        let mu = paired_mu(PairedHook::Orthogonal, &ctx.a_diagonal());
        let qs = tt_seeds(&ctx, 3, 0x0B11);
        let off = |q: &TtTangent<f64>, f: f64| {
            let mut w = q.clone();
            w.for_each_coeff_mut(|c| *c = (*c * f).sin());
            w
        };
        let (u, w) = (off(&qs[0], 7.3), off(&qs[1], 3.1));
        for q in &qs {
            let mut q0 = q.clone();
            deflate_radial(radial, &mut q0);
            let mut g = q0.clone();
            clean_tangent(radial, &mut g);
            g.axpy(-1.0, &q0);
            assert!(
                g.norm() <= 1e-13 * q0.norm(),
                "Q is not the identity on a tangent"
            );
        }
        let mut g = u.clone();
        clean_tangent(radial, &mut g);
        let mut gg = g.clone();
        clean_tangent(radial, &mut gg);
        gg.axpy(-1.0, &g);
        assert!(gg.norm() <= 1e-13 * g.norm(), "Q is not idempotent");
        let removed = |hook: PairedHook| {
            let mut c = u.clone();
            tt_clean(hook, bl, br, radial, &mut c);
            let mut e = u.clone();
            e.axpy(-1.0, &c);
            qs[1..].iter().fold(0.0f64, |m, q| {
                let mut q = q.clone();
                deflate_radial(radial, &mut q);
                m.max((q.dot(&e) / (q.norm() * e.norm())).abs())
            })
        };
        let (og, ob) = (
            removed(PairedHook::Orthogonal),
            removed(PairedHook::Oblique),
        );
        println!("removed part's overlap with tangents: orthogonal {og:.2e}, oblique {ob:.2e}");
        assert!(og <= 1e-14, "Q is not orthogonal: {og:e}");
        assert!(
            ob >= 1e-4,
            "project∘to_vector came out orthogonal ({ob:e}) — vacuous"
        );
        let asym = |hook: PairedHook, a_side: bool| {
            let m = paired_mu(hook, &ctx.a_diagonal());
            let iu = tt_image(&ctx, hook, radial, m, &u, a_side);
            let iw = tt_image(&ctx, hook, radial, m, &w, a_side);
            let (p, q) = (u.dot(&iw), iu.dot(&w));
            (p - q).abs() / (iu.norm() * w.norm()).max(f64::MIN_POSITIVE)
        };
        for a_side in [true, false] {
            let (so, sb) = (
                asym(PairedHook::Orthogonal, a_side),
                asym(PairedHook::Oblique, a_side),
            );
            println!(
                "{} asymmetry: orthogonal {so:.2e}, oblique {sb:.2e}",
                if a_side { "A" } else { "B" }
            );
            assert!(
                so <= 1e-12,
                "orthogonal-hook operator not symmetric: {so:e}"
            );
            assert!(
                sb >= 1e-6,
                "oblique-hook operator came out symmetric ({sb:e}) — vacuous"
            );
        }
        let mut c = u.clone();
        c.axpy(-1.0, &g);
        let mut ac = tt_image(&ctx, PairedHook::Orthogonal, radial, mu, &c, true);
        ac.axpy(-mu, &c);
        let bc = tt_image(&ctx, PairedHook::Orthogonal, radial, mu, &c, false);
        assert!(
            ac.norm() <= 1e-12 * mu * c.norm(),
            "A′c ≠ μc: {:e}",
            ac.norm()
        );
        assert!(
            bc.norm() <= 1e-12 * mu * c.norm(),
            "B′c ≠ 0: {:e}",
            bc.norm()
        );
    }

    #[test]
    fn paired_positive_window_keeps_the_genuine_anomalies() {
        let n = 48;
        let k = 3;
        let base = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 2000,
            max_subspace: 24,
            ..PairedDavidsonOpts::default()
        };
        let window_only = PairedDavidsonOpts {
            positive_window: true,
            ..base
        };
        let gated = PairedDavidsonOpts {
            anomalies: AnomalyRule::Gate(0.1),
            ..window_only
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC1_C1);
        let r = sym(&spread(n, 0.5, 5.0), &mut rng);
        let w = sym(&spread(n, 0.3, 4.0), &mut rng);
        let (a, b) = ab_from_rw(&r, &w);
        let v0s = seeds(n, k + 2, &mut rng);
        let o0 = run(&a, &b, n, &v0s, k, &base);
        for o1 in [
            run(&a, &b, n, &v0s, k, &window_only),
            run(&a, &b, n, &v0s, k, &gated),
        ] {
            assert_eq!(o0.info.built, o1.info.built);
            for (p, q) in o0.omega2().iter().zip(&o1.omega2()) {
                assert_eq!(
                    p.to_bits(),
                    q.to_bits(),
                    "PD: the rule changed ω² {p} → {q}"
                );
            }
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_C5);
        let mut rev = spread(n, 0.5, 5.0);
        rev[0] = -0.7;
        rev[1] = -0.25;
        let mut wev = spread(n, 0.3, 4.0);
        wev[0] = -0.5;
        wev[1] = -0.2;
        let r = sym(&rev, &mut rng);
        let w = sym(&wev, &mut rng);
        let (a, b) = ab_from_rw(&r, &w);
        let v0s = seeds(n, k + 2, &mut rng);
        let full = paired_ritz(&a, &b, n);
        let pos: Vec<f64> = full
            .iter()
            .filter(|r| r.kind == RitzKind::Real && r.krein >= 0.0)
            .take(k)
            .map(|r| r.omega.re)
            .collect();
        let window = pos[k - 1].powi(2);
        let imag: Vec<f64> = full
            .iter()
            .filter(|r| r.kind == RitzKind::Imaginary)
            .map(|r| (r.omega * r.omega).re)
            .collect();
        let cplx: Vec<C64> = full
            .iter()
            .filter(|r| r.kind == RitzKind::Complex && (r.omega * r.omega).re <= window)
            .map(|r| r.omega * r.omega)
            .collect();
        let g = run(&a, &b, n, &v0s, k, &gated);
        let g_pos: Vec<f64> = g
            .real
            .iter()
            .filter(|m| m.omega.im == 0.0 && m.krein >= 0.0)
            .map(|m| m.omega.re)
            .collect();
        println!(
            "C5 gated 0.1: converged {} built {} gated_out {} positive-type {g_pos:?}",
            g.info.converged, g.info.built, g.info.gated_out
        );
        assert!(g.info.converged, "gated C5 not converged");
        for (x, e) in g_pos.iter().zip(&pos) {
            assert!((x - e).abs() <= 1e-8, "gated: positive-type ω {x} vs {e}");
        }
        let out = run(&a, &b, n, &v0s, k, &window_only);
        println!(
            "C5 positive window: converged {} built {} gated_out {} ω² {:?} complex {:?}; dense imag {imag:?} \
             pos {pos:?} complex {cplx:?}",
            out.info.converged,
            out.info.built,
            out.info.gated_out,
            out.omega2(),
            out.complex.iter().map(|m| m.omega2()).collect::<Vec<_>>()
        );
        assert!(out.info.converged, "positive-window C5 not converged");
        assert!(
            !imag.is_empty() && !cplx.is_empty(),
            "fixture premise: anomalies in the window"
        );
        let got_imag: Vec<f64> = out
            .real
            .iter()
            .filter(|m| m.omega.im.abs() > 0.0)
            .map(|m| m.omega2().re)
            .collect();
        let got_pos: Vec<f64> = out
            .real
            .iter()
            .filter(|m| m.omega.im == 0.0 && m.krein >= 0.0)
            .map(|m| m.omega.re)
            .collect();
        assert_eq!(got_imag.len(), imag.len(), "imaginary pairs");
        for (g, e) in got_imag.iter().zip(&imag) {
            assert!(
                (g - e).abs() <= 1e-8 * e.abs().max(1.0),
                "imaginary ω² {g} vs {e}"
            );
        }
        assert_eq!(got_pos.len(), k, "k positive-type pairs");
        for (g, e) in got_pos.iter().zip(&pos) {
            assert!((g - e).abs() <= 1e-8, "positive-type ω {g} vs {e}");
        }
        assert_eq!(out.complex.len(), cplx.len(), "complex pairs in the window");
        for (m, z) in out.complex.iter().zip(&cplx) {
            assert!(
                (m.omega2() - *z).norm() <= 1e-8 * z.norm().max(1.0),
                "complex {} vs {z}",
                m.omega2()
            );
        }
    }

    fn c8() -> (Vec<f64>, Vec<f64>, usize, usize) {
        let n = 48;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC8_C8);
        let q = orthogonal(n, &mut rng);
        let mut a0 = vec![0.0; n * n];
        let mut b0 = vec![0.0; n * n];
        for (i, &x) in spread(n, 1.0, 4.0).iter().enumerate() {
            a0[i * n + i] = x;
        }
        for i in 0..n {
            for j in 0..=i {
                let x = 0.05 * rng.random_range(-1.0..1.0);
                b0[i * n + j] = x;
                b0[j * n + i] = x;
            }
        }
        for (t, &bv) in [-4.5, -4.6, 4.6, 4.7].iter().enumerate() {
            let i = n - 4 + t;
            b0[i * n + i] = bv;
        }
        let rot = |m0: &[f64]| -> Vec<f64> {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] = (0..n)
                        .map(|s| q[s][i] * (0..n).map(|t| m0[s * n + t] * q[t][j]).sum::<f64>())
                        .sum();
                }
            }
            m
        };
        let (a, b) = (rot(&a0), rot(&b0));
        let apb: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let amb: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        let (na, _) = negative_part(&a, n, false);
        let (np, _) = negative_part(&apb, n, false);
        let (nm, _) = negative_part(&amb, n, false);
        assert!(
            na == 0 && np >= 1 && nm >= 1,
            "C8 premise: A ≻ 0, A ± B indefinite ({na}, {np}, {nm})"
        );
        (a, b, n, np + nm)
    }

    fn dense_positive(a: &[f64], b: &[f64], n: usize, k: usize) -> Vec<f64> {
        paired_ritz(a, b, n)
            .iter()
            .filter(|r| r.kind == RitzKind::Real && r.krein >= 0.0)
            .take(k)
            .map(|r| r.omega.re)
            .collect()
    }

    fn positive_type(out: &PairedDavidsonOut<DVec>) -> Vec<f64> {
        out.real
            .iter()
            .filter(|m| m.omega.im == 0.0 && m.krein >= 0.0)
            .map(|m| m.omega.re)
            .collect()
    }

    #[test]
    fn paired_freeze_keeps_the_offset_and_converges_the_window() {
        let (a, b, n, dense_offset) = c8();
        let k = 4;
        let base = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 4000,
            max_subspace: 24,
            positive_window: true,
            ..PairedDavidsonOpts::default()
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC8_01);
        let v0s = seeds(n, k + 2, &mut rng);
        let pos = dense_positive(&a, &b, n, k);
        let mut drops = Vec::new();
        for rule in [
            AnomalyRule::Freeze,
            AnomalyRule::Chase,
            AnomalyRule::Gate(1e-12),
        ] {
            let opts = PairedDavidsonOpts {
                anomalies: rule,
                ..base
            };
            let out = run(&a, &b, n, &v0s, k, &opts);
            let got = positive_type(&out);
            println!(
                "C8 {rule:?}: converged {} built {} restarts {} frozen {} anomaly_corrections {} \
                 gated_out {} offset {:?} (dense {dense_offset}) drops {} ω {got:?} dense {pos:?}",
                out.info.converged,
                out.info.built,
                out.info.restarts,
                out.info.frozen,
                out.info.anomaly_corrections,
                out.info.gated_out,
                out.info.offset,
                out.info.offset_drops
            );
            assert!(
                out.info.offset.total() <= dense_offset,
                "{rule:?}: the projected offset exceeds the pencil's (interlacing)"
            );
            drops.push(out.info.offset_drops);
            if rule == AnomalyRule::Freeze {
                assert!(out.info.converged, "Freeze: not converged");
                assert!(out.info.restarts >= 1, "fixture premise: forced restarts");
                assert!(
                    out.info.frozen >= 1,
                    "fixture premise: anomalies in the window"
                );
                assert_eq!(
                    out.info.anomaly_corrections, 0,
                    "Freeze corrected an anomaly"
                );
                assert_eq!(out.info.offset_drops, 0, "Freeze lost offset at a restart");
                assert_eq!(got.len(), k, "Freeze: k positive-type pairs");
                for (g, e) in got.iter().zip(&pos) {
                    assert!((g - e).abs() <= 1e-8, "Freeze: positive-type ω {g} vs {e}");
                }
            }
        }
        assert!(
            drops[2] >= 1,
            "the window-only restart never lost offset: the restart rule is untested"
        );
    }

    pub(crate) fn many_imaginary(n: usize, t: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let q = orthogonal(n, &mut rng);
        let mut a0 = vec![0.0; n * n];
        let mut b0 = vec![0.0; n * n];
        for (i, &x) in spread(n, 1.0, 4.0).iter().enumerate() {
            a0[i * n + i] = x;
        }
        for i in 0..n {
            for j in 0..=i {
                let x = 0.05 * rng.random_range(-1.0..1.0);
                b0[i * n + j] = x;
                b0[j * n + i] = x;
            }
        }
        for (s, &bv) in spread(t, 4.5, 5.5).iter().enumerate() {
            let i = n - t + s;
            b0[i * n + i] = if s % 2 == 0 { bv } else { -bv };
        }
        let rot = |m0: &[f64]| -> Vec<f64> {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] = (0..n)
                        .map(|s| q[s][i] * (0..n).map(|u| m0[s * n + u] * q[u][j]).sum::<f64>())
                        .sum();
                }
            }
            m
        };
        (rot(&a0), rot(&b0))
    }

    #[test]
    fn paired_restarts_keep_the_images_consistent() {
        let (n, t, k) = (64, 12, 2);
        let (a, b) = many_imaginary(n, t, 0x1A_6E);
        let (real, cplx) = oracle(&a, &b, n);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 4000,
            max_subspace: 24,
            anomalies: AnomalyRule::Chase,
            ..PairedDavidsonOpts::default()
        };
        let ap = |v: &DVec| mat_apply(&a, n, v);
        let bp = |v: &DVec| mat_apply(&b, n, v);
        for block in [153u64, 154, 161, 170, 196, 197] {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1A_01 + block * 7919);
            let v0s = seeds(n, k + 2, &mut rng);
            let out = run(&a, &b, n, &v0s, k, &opts);
            let label = format!("seed block {block}");
            assert!(
                out.info.restarts >= 1,
                "{label}: fixture premise: forced restarts"
            );
            assert_wanted(&label, &out, &real, &cplx, k, 1e-8);
            for m in &out.real {
                let (al, be) = (m.omega.re, m.omega.im);
                let zero = DVec(vec![0.0; n]);
                let (xi, yi) = (
                    m.x_im.as_ref().unwrap_or(&zero),
                    m.y_im.as_ref().unwrap_or(&zero),
                );
                let (xr, yr) = (&m.x_re, &m.y_re);
                let comb = |terms: &[(f64, &DVec)]| -> f64 {
                    let mut acc = vec![0.0; n];
                    for (c, v) in terms {
                        acc.iter_mut().zip(&v.0).for_each(|(o, x)| *o += c * x);
                    }
                    acc.iter().map(|x| x * x).sum()
                };
                let (axr, bxr, ayr, byr) = (ap(xr), bp(xr), ap(yr), bp(yr));
                let (axi, bxi, ayi, byi) = (ap(xi), bp(xi), ap(yi), bp(yi));
                let fresh = (comb(&[(1.0, &axr), (1.0, &byr), (-al, xr), (be, xi)])
                    + comb(&[(1.0, &bxr), (1.0, &ayr), (al, yr), (-be, yi)])
                    + comb(&[(1.0, &axi), (1.0, &byi), (-al, xi), (-be, xr)])
                    + comb(&[(1.0, &bxi), (1.0, &ayi), (al, yi), (be, yr)]))
                .sqrt();
                assert!(
                    (fresh - m.residual).abs() <= 1e-9 * m.omega.norm().max(1.0),
                    "{label}: ω² {}: fresh residual {fresh:.3e} vs the kernel's {:.3e}",
                    m.omega2(),
                    m.residual
                );
            }
        }
    }

    #[test]
    fn paired_restart_leaves_room_past_many_anomalies() {
        let (n, t, k) = (64, 12, 2);
        let (a, b) = many_imaginary(n, t, 0x1A_6E);
        let (real, cplx) = oracle(&a, &b, n);
        assert_eq!(
            real.iter().filter(|&&x| x < 0.0).count(),
            t,
            "fixture premise: {t} imaginary pairs"
        );
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1A_01);
        let v0s = seeds(n, k + 2, &mut rng);
        for (rule, pw) in [(AnomalyRule::Chase, false), (AnomalyRule::Freeze, true)] {
            let opts = PairedDavidsonOpts {
                tol: 1e-10,
                max_basis: 4000,
                max_subspace: 24,
                positive_window: pw,
                anomalies: rule,
                ..PairedDavidsonOpts::default()
            };
            let out = run(&a, &b, n, &v0s, k, &opts);
            let i = &out.info;
            println!(
                "{rule:?}: converged {} stalled {} built {} restarts {} peak live {} frozen {} \
                 phases {}",
                i.converged, i.stalled, i.built, i.restarts, i.peak_live, i.frozen, i.phases
            );
            assert!(
                i.restarts >= 1,
                "{rule:?}: fixture premise: forced restarts"
            );
            assert!(i.peak_live > 24, "{rule:?}: the live bound never grew");
            assert!(!i.stalled, "{rule:?}: stalled");
            let p = i.phases;
            let sum = p.apply_a + p.apply_b + p.orth + p.gram + p.ritz + p.residual + p.restart;
            assert!(
                p.apply_a > 0.0 && p.ritz > 0.0 && sum + p.precond <= p.total,
                "{rule:?}: the phases are disjoint parts of the wall: {p}"
            );
            if rule == AnomalyRule::Chase {
                assert_wanted("many imaginary, Chase", &out, &real, &cplx, k, 1e-8);
            } else {
                assert!(i.converged, "Freeze: not converged");
                let got = positive_type(&out);
                let pos = dense_positive(&a, &b, n, k);
                assert_eq!(got.len(), k, "Freeze: k positive-type pairs");
                for (g, e) in got.iter().zip(&pos) {
                    assert!((g - e).abs() <= 1e-8, "Freeze: positive-type ω {g} vs {e}");
                }
            }
        }
    }

    #[test]
    fn paired_freeze_window_ritz_values_are_upper_bounds() {
        let (a, b, n, dense_offset) = c8();
        let k = 4;
        let pos = dense_positive(&a, &b, n, k);
        let full = paired_ritz(&a, &b, n);
        assert!(
            !full
                .iter()
                .any(|r| r.kind == RitzKind::Real && r.krein < 0.0 && r.omega.re <= pos[k - 1]),
            "C8 premise: no negative-type real eigenvalue in the window"
        );
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC8_01);
        let v0s = seeds(n, k + 2, &mut rng);
        let (mut checked, mut restarted) = (0usize, 0usize);
        for budget in (k + 2..=200).step_by(4) {
            let opts = PairedDavidsonOpts {
                tol: 1e-10,
                max_basis: budget,
                max_subspace: 24,
                positive_window: true,
                anomalies: AnomalyRule::Freeze,
                ..PairedDavidsonOpts::default()
            };
            let out = run(&a, &b, n, &v0s, k, &opts);
            let off = out.info.offset;
            if off.neg_a == 0 && off.total() == dense_offset {
                for (j, (t, e)) in positive_type(&out).iter().zip(&pos).enumerate() {
                    assert!(
                        *t >= e - 1e-9,
                        "budget {budget}: Ritz {j} = {t} below the pencil's {e} with the offset held"
                    );
                    checked += 1;
                }
                if out.info.restarts > 0 {
                    restarted += 1;
                }
            }
            if out.info.converged {
                break;
            }
        }
        println!(
            "Freeze bound pin: {checked} Ritz values checked, {restarted} budgets past a restart"
        );
        assert!(checked >= 20, "too few offset-holding budgets: {checked}");
        assert!(
            restarted >= 1,
            "no checked budget had restarted: the restart rule is untested"
        );
    }

    #[allow(clippy::type_complexity)]
    fn planted_artifacts(
        n: usize,
        k: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<(f64, f64)>) {
        let q = orthogonal(n, rng);
        let np = n - 3;
        let mut a0 = vec![0.0; n * n];
        let mut b0 = vec![0.0; n * n];
        for (i, &x) in spread(np, 1.0, 4.0).iter().enumerate() {
            a0[i * n + i] = x;
        }
        for i in 0..np {
            for j in 0..=i {
                let x = 0.05 * rng.random_range(-1.0..1.0);
                b0[i * n + j] = x;
                b0[j * n + i] = x;
            }
        }
        for (t, (&av, &dv)) in [3.0, 3.2, 3.4].iter().zip(&[0.05, 0.07, 0.09]).enumerate() {
            let i = np + t;
            a0[i * n + i] = av;
            b0[i * n + i] = av - dv;
        }
        let rot = |m0: &[f64]| -> Vec<f64> {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] = (0..n)
                        .map(|s| q[s][i] * (0..n).map(|t| m0[s * n + t] * q[t][j]).sum::<f64>())
                        .sum();
                }
            }
            m
        };
        let (a, b) = (rot(&a0), rot(&b0));
        let full = paired_ritz(&a, &b, n);
        let physical: Vec<f64> = full
            .iter()
            .filter(|r| r.kind == RitzKind::Real && r.krein > 0.5)
            .take(k)
            .map(|r| r.omega.re)
            .collect();
        let artifacts: Vec<(f64, f64)> = full
            .iter()
            .filter(|r| r.kind == RitzKind::Real && r.krein > 0.0 && r.krein <= 0.5)
            .map(|r| (r.omega.re, r.krein))
            .collect();
        assert!(
            artifacts.len() == 3 && artifacts.iter().all(|&(w, _)| w < physical[0]),
            "fixture premise: three artifacts below the physical spectrum {artifacts:?} {physical:?}"
        );
        (a, b, q, physical, artifacts)
    }

    #[test]
    fn paired_targets_the_seed_space_past_planted_artifacts() {
        let n = 80;
        let k = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A_7A);
        let (a, b, q, physical, artifacts) = planted_artifacts(n, k, &mut rng);
        let target: Vec<DVec> = (0..k)
            .map(|c| {
                DVec(
                    q[c].iter()
                        .map(|x| x + 1e-2 * rng.random_range(-1.0..1.0))
                        .collect(),
                )
            })
            .collect();
        let v0s = seeds(n, k + 2, &mut rng);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 2000,
            max_subspace: 44,
            positive_window: true,
            anomalies: AnomalyRule::Freeze,
            ..PairedDavidsonOpts::default()
        };
        let solve = |target: &[DVec]| {
            omega_davidson_paired(
                |v: &DVec| mat_apply(&a, n, v),
                |v: &DVec| mat_apply(&b, n, v),
                |_: &mut DVec| {},
                None::<fn(&mut DVec, f64)>,
                false,
                &v0s,
                target,
                k,
                n,
                &opts,
            )
        };
        let lowest = solve(&[]);
        let out = solve(&target);
        let got = positive_type(&out);
        let untargeted: Vec<(f64, f64, Option<f64>)> = out
            .untargeted_below
            .iter()
            .map(|m| (m.omega.re, m.krein, m.overlap))
            .collect();
        println!(
            "targeted: converged {} built {} restarts {} ω {got:?} overlaps {:?} window_index {:?}\n  \
             untargeted below {untargeted:?}\n  dense physical {physical:?} artifacts {artifacts:?}\n  \
             lowest-k twin: converged {} built {} ω {:?}",
            out.info.converged,
            out.info.built,
            out.info.restarts,
            out.real.iter().map(|m| m.overlap).collect::<Vec<_>>(),
            out.real.iter().map(|m| m.window_index).collect::<Vec<_>>(),
            lowest.info.converged,
            lowest.info.built,
            positive_type(&lowest)
        );
        assert!(out.info.converged, "targeted solve not converged");
        assert_eq!(got.len(), k, "k targeted pairs");
        for (g, e) in got.iter().zip(&physical) {
            assert!((g - e).abs() <= 1e-8, "targeted ω {g} vs physical {e}");
        }
        assert!(
            out.real.iter().all(|m| m.krein > 0.5),
            "an artifact was targeted"
        );
        assert!(
            !untargeted.is_empty() && untargeted.iter().all(|&(_, kr, _)| kr <= 0.5),
            "untargeted pairs below the window must be the artifacts: {untargeted:?}"
        );
        assert!(
            (positive_type(&lowest)[0] - artifacts[0].0).abs() <= 1e-6,
            "fixture premise: the lowest-k twin converges onto the lowest artifact"
        );
    }

    #[test]
    fn paired_target_gate_takes_the_k_lowest_in_an_oversized_span() {
        let n = 80;
        let k = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A_7A);
        let (a, b, q, physical, artifacts) = planted_artifacts(n, k, &mut rng);
        let target: Vec<DVec> = (0..2 * k)
            .map(|c| {
                DVec(
                    q[c].iter()
                        .map(|x| x + 1e-2 * rng.random_range(-1.0..1.0))
                        .collect(),
                )
            })
            .collect();
        let v0s = seeds(n, k + 2, &mut rng);
        let solve = |rank: WantedRank| {
            let opts = PairedDavidsonOpts {
                tol: 1e-10,
                max_basis: 2000,
                max_subspace: 44,
                positive_window: true,
                anomalies: AnomalyRule::Freeze,
                rank,
                ..PairedDavidsonOpts::default()
            };
            omega_davidson_paired(
                |v: &DVec| mat_apply(&a, n, v),
                |v: &DVec| mat_apply(&b, n, v),
                |_: &mut DVec| {},
                None::<fn(&mut DVec, f64)>,
                false,
                &v0s,
                &target,
                k,
                n,
                &opts,
            )
        };
        let ranked = solve(WantedRank::Omega);
        let out = solve(WantedRank::TargetGate(0.5));
        let got = positive_type(&out);
        let below: Vec<(f64, f64, Option<f64>)> = out
            .untargeted_below
            .iter()
            .map(|m| (m.omega.re, m.krein, m.overlap))
            .collect();
        println!(
            "TargetGate(0.5): converged {} built {} ω {got:?} overlaps {:?}\n  below {below:?}\n  \
             overlap-rank twin: converged {} built {} ω {:?}\n  dense physical {physical:?} \
             artifacts {artifacts:?}",
            out.info.converged,
            out.info.built,
            out.real.iter().map(|m| m.overlap).collect::<Vec<_>>(),
            ranked.info.converged,
            ranked.info.built,
            positive_type(&ranked)
        );
        assert!(out.info.converged, "gated solve not converged");
        assert_eq!(got.len(), k, "k wanted pairs");
        for (g, e) in got.iter().zip(&physical) {
            assert!((g - e).abs() <= 1e-8, "gated ω {g} vs physical {e}");
        }
        assert!(
            out.real.iter().all(|m| m.overlap.is_some_and(|o| o >= 0.5)),
            "a wanted pair below the gate"
        );
        assert!(
            !below.is_empty()
                && below
                    .iter()
                    .all(|&(_, kr, ov)| kr <= 0.5 && ov.unwrap() < 0.5),
            "pairs reported below the window must be the artifacts: {below:?}"
        );
    }

    #[test]
    fn paired_gate_or_excess_recovers_a_mode_the_target_misses() {
        let n = 80;
        let k = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A_7A);
        let (a, b, q, physical, artifacts) = planted_artifacts(n, k, &mut rng);
        let target: Vec<DVec> = (0..2 * k)
            .filter(|&c| c != 1)
            .map(|c| {
                DVec(
                    q[c].iter()
                        .map(|x| x + 1e-2 * rng.random_range(-1.0..1.0))
                        .collect(),
                )
            })
            .collect();
        let v0s = seeds(n, k + 2, &mut rng);
        let solve = |rank: WantedRank| {
            let opts = PairedDavidsonOpts {
                tol: 1e-10,
                max_basis: 2000,
                max_subspace: 44,
                positive_window: true,
                anomalies: AnomalyRule::Freeze,
                rank,
                ..PairedDavidsonOpts::default()
            };
            omega_davidson_paired(
                |v: &DVec| mat_apply(&a, n, v),
                |v: &DVec| mat_apply(&b, n, v),
                |_: &mut DVec| {},
                None::<fn(&mut DVec, f64)>,
                false,
                &v0s,
                &target,
                k,
                n,
                &opts,
            )
        };
        let below_of = |out: &PairedDavidsonOut<DVec>| -> Vec<(f64, f64, f64)> {
            out.untargeted_below
                .iter()
                .map(|m| (m.omega.re, m.krein, m.rho_a / m.omega.re))
                .collect()
        };
        let gated = solve(WantedRank::TargetGate(0.5));
        let union = solve(WantedRank::TargetGateOrExcess(0.5, 3.0));
        let (got_g, got_u) = (positive_type(&gated), positive_type(&union));
        println!(
            "TargetGate(0.5): converged {} built {} gate_short {} ω {got_g:?}\n  below {:?}\n  \
             TargetGateOrExcess(0.5, 3): converged {} built {} gate_short {} ω {got_u:?}\n  below \
             {:?}\n  dense physical {physical:?} artifacts {artifacts:?}",
            gated.info.converged,
            gated.info.built,
            gated.info.gate_short,
            below_of(&gated),
            union.info.converged,
            union.info.built,
            union.info.gate_short,
            below_of(&union)
        );
        assert!(
            gated.info.gate_short == 0 && !got_g.iter().any(|g| (g - physical[1]).abs() <= 1e-6),
            "fixture premise: the gate alone keeps the mode the target misses out of its wanted set"
        );
        assert!(union.info.converged, "union solve not converged");
        assert_eq!(got_u.len(), k, "k wanted pairs");
        for (g, e) in got_u.iter().zip(&physical) {
            assert!((g - e).abs() <= 1e-8, "union ω {g} vs physical {e}");
        }
        let below = below_of(&union);
        assert!(
            !below.is_empty()
                && below
                    .iter()
                    .all(|&(_, kr, excess)| kr <= 0.5 && excess > 3.0),
            "pairs reported below the window must be the artifacts: {below:?}"
        );
    }

    #[test]
    fn paired_a_excess_rank_skips_planted_artifacts() {
        let n = 80;
        let k = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A_7A);
        let (a, b, _, physical, artifacts) = planted_artifacts(n, k, &mut rng);
        let v0s = seeds(n, k + 2, &mut rng);
        for rank in [WantedRank::AExcess(3.0)] {
            let opts = PairedDavidsonOpts {
                tol: 1e-10,
                max_basis: 2000,
                max_subspace: 44,
                positive_window: true,
                anomalies: AnomalyRule::Freeze,
                rank,
                ..PairedDavidsonOpts::default()
            };
            let out = omega_davidson_paired(
                |v: &DVec| mat_apply(&a, n, v),
                |v: &DVec| mat_apply(&b, n, v),
                |_: &mut DVec| {},
                None::<fn(&mut DVec, f64)>,
                false,
                &v0s,
                &[],
                k,
                n,
                &opts,
            );
            let got = positive_type(&out);
            let below: Vec<(f64, f64, f64)> = out
                .untargeted_below
                .iter()
                .map(|m| (m.omega.re, m.krein, m.rho_a))
                .collect();
            println!(
                "{rank:?}: converged {} built {} restarts {} ω {got:?} ρ_A {:?}\n  reported below \
             {below:?}\n  dense physical {physical:?} artifacts {artifacts:?}",
                out.info.converged,
                out.info.built,
                out.info.restarts,
                out.real.iter().map(|m| m.rho_a).collect::<Vec<_>>()
            );
            assert!(out.info.converged, "A-excess solve not converged");
            assert_eq!(got.len(), k, "k wanted pairs");
            for (g, e) in got.iter().zip(&physical) {
                assert!((g - e).abs() <= 1e-8, "wanted ω {g} vs physical {e}");
            }
            assert!(
                out.real.iter().all(|m| m.krein > 0.5),
                "an artifact was wanted"
            );
            assert!(
                !below.is_empty() && below.iter().all(|&(w, kr, ra)| kr <= 0.5 && ra > 2.0 * w),
                "pairs reported below the window must be the artifacts, ρ_A ≫ ω: {below:?}"
            );
        }
    }

    #[test]
    fn paired_selective_rank_counts_positive_type_pairs_only() {
        let n = 60;
        let k = 4;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E_1E);
        let q = orthogonal(n, &mut rng);
        let np = n - 4;
        let mut a0 = vec![0.0; n * n];
        let mut b0 = vec![0.0; n * n];
        for (i, &x) in spread(np, 1.0, 4.0).iter().enumerate() {
            a0[i * n + i] = x;
        }
        for i in 0..np {
            for j in 0..=i {
                let x = 0.05 * rng.random_range(-1.0..1.0);
                b0[i * n + j] = x;
                b0[j * n + i] = x;
            }
        }
        for (t, (&av, &bv)) in [3.0, 3.2, 3.4, -1.1]
            .iter()
            .zip(&[2.95, 3.13, 3.31, 0.3])
            .enumerate()
        {
            let i = np + t;
            a0[i * n + i] = av;
            b0[i * n + i] = bv;
        }
        let rot = |m0: &[f64]| -> Vec<f64> {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    m[i * n + j] = (0..n)
                        .map(|s| q[s][i] * (0..n).map(|t| m0[s * n + t] * q[t][j]).sum::<f64>())
                        .sum();
                }
            }
            m
        };
        let (a, b) = (rot(&a0), rot(&b0));
        let full = paired_ritz(&a, &b, n);
        let physical: Vec<f64> = full
            .iter()
            .filter(|r| r.kind == RitzKind::Real && r.krein > 0.5)
            .take(k)
            .map(|r| r.omega.re)
            .collect();
        let negative: Vec<f64> = full
            .iter()
            .filter(|r| r.kind == RitzKind::Real && r.krein < 0.0)
            .map(|r| r.omega.re)
            .collect();
        assert!(
            negative.len() == 1 && negative[0] > physical[0] && negative[0] < physical[k - 1],
            "fixture premise: one negative-type pair inside the physical window {negative:?} \
             {physical:?}"
        );
        let v0s = seeds(n, k + 2, &mut rng);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 2000,
            max_subspace: 44,
            rank: WantedRank::AExcess(3.0),
            ..PairedDavidsonOpts::default()
        };
        assert!(!opts.positive_window && opts.anomalies == AnomalyRule::Chase);
        let out = run(&a, &b, n, &v0s, k, &opts);
        let got = positive_type(&out);
        let neg: Vec<(f64, f64)> = out
            .real
            .iter()
            .filter(|m| m.omega.im == 0.0 && m.krein < 0.0)
            .map(|m| (m.omega.re, m.residual))
            .collect();
        println!(
            "AExcess(3), Chase, all-Krein window: converged {} built {} positive-type {got:?} \
             negative-type {neg:?}; dense physical {physical:?} negative {negative:?}",
            out.info.converged, out.info.built
        );
        assert!(out.info.converged, "not converged");
        assert_eq!(got.len(), k, "k positive-type pairs wanted");
        for (g, e) in got.iter().zip(&physical) {
            assert!((g - e).abs() <= 1e-8, "wanted ω {g} vs physical {e}");
        }
        assert!(
            out.real.iter().all(|m| m.krein < 0.0 || m.krein > 0.5),
            "an artifact was wanted"
        );
        assert_eq!(neg.len(), 1, "the negative-type pair is returned");
        assert!(
            (neg[0].0 - negative[0]).abs() <= 1e-8,
            "negative-type ω {} vs {}",
            neg[0].0,
            negative[0]
        );
    }

    #[test]
    fn paired_push_cleans_after_the_mgs() {
        let (n, r) = (40, 30);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9C1E);
        let q = orthogonal(n, &mut rng);
        let in_space = |v: &[f64]| -> Vec<f64> {
            let mut out = vec![0.0; n];
            for u in &q[..r] {
                let c: f64 = u.iter().zip(v).map(|(a, b)| a * b).sum();
                out.iter_mut().zip(u).for_each(|(o, x)| *o += c * x);
            }
            out
        };
        let mut project = |v: &mut DVec| v.0 = in_space(&v.0);
        let basis: Vec<DVec> = (0..5).map(|i| DVec(q[i].clone())).collect();
        let mut cand = vec![0.0; n];
        for (i, u) in q[..5].iter().enumerate() {
            cand.iter_mut()
                .zip(u)
                .for_each(|(c, x)| *c += (1.0 + i as f64) * x);
        }
        cand.iter_mut().zip(&q[7]).for_each(|(c, x)| *c += 1e-8 * x);
        let off_before = |v: &[f64]| -> f64 {
            let p = in_space(v);
            v.iter()
                .zip(&p)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let mut noisy = cand.clone();
        noisy
            .iter_mut()
            .zip(&q[r + 2])
            .for_each(|(c, x)| *c += 1e-14 * x);
        let mut pre = DVec(noisy.clone());
        project(&mut pre);
        pre.0
            .iter_mut()
            .zip(&q[r + 2])
            .for_each(|(c, x)| *c += 1e-14 * x);
        let old = orthonormalize_against(pre, &basis).expect("the new direction survives");
        let old_off = off_before(&old.0);
        let new = push_candidate(DVec(noisy), &basis, &mut project).expect("survives");
        let new_off = off_before(&new.0);
        let ortho = basis.iter().fold(0.0f64, |m, u| m.max(u.dot(&new).abs()));
        println!(
            "off-space content: MGS only {old_off:.2e}, push_candidate {new_off:.2e}; orthogonality {ortho:.1e}"
        );
        assert!(
            old_off >= 1e-7,
            "fixture: the MGS did not amplify ({old_off:e})"
        );
        assert!(
            new_off <= 1e-14,
            "push_candidate left {new_off:e} off the space"
        );
        assert!(ortho <= 1e-12 && (new.norm() - 1.0).abs() <= 1e-12);
    }

    fn rand_sym(n: usize, s: f64, rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let x = s * rng.random_range(-1.0..1.0);
                m[i * n + j] = x;
                m[j * n + i] = x;
            }
        }
        m
    }

    fn realified_pencil(nc: usize, bs: f64, t: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let ar = sym(&spread(nc, 1.0, 4.0), &mut rng);
        let mut ai = vec![0.0; nc * nc];
        for i in 0..nc {
            for j in 0..i {
                let x = 0.1 * rng.random_range(-1.0..1.0);
                ai[i * nc + j] = x;
                ai[j * nc + i] = -x;
            }
        }
        let mut br = rand_sym(nc, bs, &mut rng);
        if t > 0 {
            let q = orthogonal(nc, &mut rng);
            for (s, &bv) in spread(t, 4.5, 5.5).iter().enumerate() {
                let sg = if s % 2 == 0 { bv } else { -bv };
                for i in 0..nc {
                    for j in 0..nc {
                        br[i * nc + j] += sg * q[s][i] * q[s][j];
                    }
                }
            }
        }
        let bi = rand_sym(nc, bs, &mut rng);
        let n = 2 * nc;
        let (mut a, mut b) = (vec![0.0; n * n], vec![0.0; n * n]);
        for i in 0..nc {
            for j in 0..nc {
                let (p, q) = (i * nc + j, nc);
                a[i * n + j] = ar[p];
                a[i * n + q + j] = -ai[p];
                a[(q + i) * n + j] = ai[p];
                a[(q + i) * n + q + j] = ar[p];
                b[i * n + j] = br[p];
                b[i * n + q + j] = bi[p];
                b[(q + i) * n + j] = bi[p];
                b[(q + i) * n + q + j] = -br[p];
            }
        }
        (a, b)
    }

    fn sigma2(u: &[f64], v: &[f64]) -> f64 {
        let nu = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let c = u.iter().zip(v).map(|(x, y)| x * y).sum::<f64>() / (nu * nv);
        (1.0 - c.abs()).max(0.0).sqrt() * (1.0 + c.abs()).sqrt() / std::f64::consts::SQRT_2
    }

    #[test]
    fn paired_ritz_keeps_both_copies_of_a_doubled_real_omega() {
        let m = 10;
        let mut worst = f64::INFINITY;
        for seed in 0..500u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let a1 = spread(5, 1.0, 4.0);
            let b1: Vec<f64> = (0..5).map(|_| 0.3 * rng.random_range(-1.0..1.0)).collect();
            let q = orthogonal(m, &mut rng);
            let rot = |d: &[f64]| {
                let mut x = vec![0.0; m * m];
                for (c, &l) in d.iter().chain(d).enumerate() {
                    for i in 0..m {
                        for j in 0..m {
                            x[i * m + j] += l * q[c][i] * q[c][j];
                        }
                    }
                }
                x
            };
            for (form, (a, b)) in [(rot(&a1), rot(&b1)), realified_pencil(5, 0.3, 0, seed)]
                .into_iter()
                .enumerate()
            {
                let r = paired_ritz(&a, &b, m);
                let reals: Vec<&PairedRitz> =
                    r.iter().filter(|p| p.kind == RitzKind::Real).collect();
                assert_eq!(reals.len(), m, "form {form} seed {seed}: every ω real");
                let z = |p: &PairedRitz| -> Vec<f64> {
                    p.a_re.iter().chain(&p.b_re).copied().collect()
                };
                for p in &reals {
                    let z = z(p);
                    let w = p.omega.re;
                    let mut r2 = 0.0f64;
                    for i in 0..m {
                        let (mut top, mut bot) = (-w * z[i], -w * z[m + i]);
                        for j in 0..m {
                            top += a[i * m + j] * z[j] + b[i * m + j] * z[m + j];
                            bot += -b[i * m + j] * z[j] - a[i * m + j] * z[m + j];
                        }
                        r2 += top * top + bot * bot;
                    }
                    assert!(
                        r2.sqrt() <= 1e-10,
                        "form {form} seed {seed}: ω {w} residual {:e}",
                        r2.sqrt()
                    );
                }
                for c in reals.chunks(2) {
                    let (w0, w1) = (c[0].omega.re, c[1].omega.re);
                    assert!(
                        (w0 - w1).abs() <= 1e-9,
                        "form {form} seed {seed}: {w0} vs {w1}"
                    );
                    let s = sigma2(&z(c[0]), &z(c[1]));
                    worst = worst.min(s);
                    assert!(s >= 1e-3, "form {form} seed {seed}: ω {w0} copies σ₂ {s:e}");
                }
            }
        }
        println!("worst σ₂ {worst:e}");
    }

    #[test]
    fn paired_certifies_a_doubled_imaginary_pair() {
        let (nc, k) = (8, 2);
        let n = 2 * nc;
        let (a, b) = realified_pencil(nc, 0.05, 2, 0xD0_0B);
        let (real, _) = oracle(&a, &b, n);
        assert_eq!(
            real.iter().filter(|&&w| w < 0.0).count(),
            4,
            "fixture premise: two doubled imaginary pairs"
        );
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0_01);
        let v0s = seeds(n, k + 2, &mut rng);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 2000,
            ..PairedDavidsonOpts::default()
        };
        let out = run(&a, &b, n, &v0s, k, &opts);
        assert!(out.info.converged, "fixture premise: converged");
        let negs: Vec<usize> = out
            .real
            .iter()
            .enumerate()
            .filter(|(_, m)| m.omega2().re < 0.0)
            .map(|(j, _)| j)
            .collect();
        assert_eq!(
            negs.len(),
            4,
            "fixture premise: every imaginary copy returned"
        );
        let worst = negs
            .iter()
            .map(|&j| {
                let m = &out.real[j];
                let th = m.omega2().re;
                let fx = fresh_on(
                    &a,
                    &b,
                    n,
                    &[&m.x_re, m.x_im.as_ref().expect("imaginary")],
                    th,
                );
                (fx - th).abs() / th.abs()
            })
            .fold(0.0, f64::max);
        assert!(
            worst >= 1e-2,
            "fixture premise: span{{Re X, Im X}} alone misses some mode's ω² (max rel {worst:e})"
        );
        assert_eq!(
            out.info
                .negative
                .iter()
                .map(|w| w.index)
                .collect::<Vec<_>>(),
            negs
        );
        for w in &out.info.negative {
            println!("{w:?}");
            assert!(w.certified(), "uncertified {w:?}");
            assert!(
                (w.fresh - w.theta).abs() <= 1e-8 * w.theta.abs(),
                "fresh {} vs θ {} (span{{Re X, Im X}} alone: up to {worst:.2e} rel)",
                w.fresh,
                w.theta
            );
        }
    }
    fn fresh_on(a: &[f64], b: &[f64], n: usize, vs: &[&DVec], theta: f64) -> f64 {
        let mut basis: Vec<DVec> = Vec::new();
        for v in vs {
            if let Some(u) = orthonormalize_against((*v).clone(), &basis) {
                basis.push(u);
            }
        }
        let m = basis.len();
        let av: Vec<DVec> = basis.iter().map(|u| mat_apply(a, n, u)).collect();
        let bv: Vec<DVec> = basis.iter().map(|u| mat_apply(b, n, u)).collect();
        let g = |img: &[DVec]| {
            let mut x = vec![0.0; m * m];
            for i in 0..m {
                for j in 0..m {
                    x[i * m + j] = 0.5 * (basis[i].dot(&img[j]) + basis[j].dot(&img[i]));
                }
            }
            x
        };
        paired_ritz(&g(&av), &g(&bv), m)
            .iter()
            .map(|r| (r.omega * r.omega).re)
            .min_by(|p, q| (p - theta).abs().total_cmp(&(q - theta).abs()))
            .unwrap_or(f64::NAN)
    }
}
