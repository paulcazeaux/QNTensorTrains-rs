// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::factor::dense_self_adjoint_smallest;
use crate::par;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;
use rayon::prelude::*;

use super::davidson_paired::WantedRank;
use super::lanczos::{SympDiagnostics, SymplecticLowestError, SymplecticLowestInfo};
use super::lobpcg::{gram_matrix_symmetrized, rr_matrix_symmetrized};

/// Preconditioner selection for the W-metric ω-eigensolvers ([`omega_davidson`] and the
/// restart-free ELOBP4dCG sibling [`super::lobpcg::omega_lobp4dcg`]) — one ladder, shared by
/// every layer wrapper (f64/C64, TT-only/joint, abelian/SU(2)).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SympPrecond {
    /// **The default: layer-resolved.** Each wrapper resolves `Auto` before anything runs —
    /// TT-layer wrappers to [`Self::DiagUnshifted`]`{ kappa: 0.0 }` (the D0 fixed unshifted
    /// inverse), joint wrappers to [`Self::Off`] — so the default is the best measured arm on
    /// each layer without a per-layer opts type. Among SHIPPABLE arms D0 wins or ties every TT
    /// cell and is the only one that formally converges the hard d8 cell; a DENSE exact clamped
    /// fold-inverse is not beaten on rate there but stagnates short of `tol` (basis saturation)
    /// and is unshippable as spelled. On the JOINT layer D0 is refuted, which is why `Auto`
    /// resolves to `Off` there.
    #[default]
    Auto,
    /// No preconditioner — the joint layer's resolved default, and the explicit
    /// no-preconditioner arm everywhere; **a measured choice, not a stylistic one.**
    /// Unpreconditioned + the folded product spectrum is a first-class method (structure-preserving
    /// thick-restart Lanczos without preconditioning: arXiv:2503.20920), and on the joint layer it
    /// is the only arm certified sound in every regime measured so far:
    Off,
    /// **Candidate D0 — the fixed unshifted SPD diagonal**, the literature default (Vecharynski,
    /// Brabec, Shao, Govind & Yang, Comput. Phys. Commun. 221 (2017) 42, "choice of the
    /// preconditioner": fixed unshifted `T = D²`; the shifted form is the JD-flavored variant that
    /// belongs inside a projected correction). Built ONCE per solve from the layer's
    /// squared-diagonal model `d` by [`crate::solvers::davidson::precond_fixed_inverse`]:
    /// `T_i = ((Re d_i)_+ + c)⁻¹` with `c = (max_i (Re d_i)_+).max(f64::MIN_POSITIVE) / κ`, so `T`
    /// is SPD by construction and its dynamic range is capped, `κ(T) ≤ kappa + 1`. `kappa <= 0`
    /// (or non-finite) selects the auto value `1e3`. The closure ignores θ entirely; `nwarm` still
    /// applies (kernel-side warmup discipline, unchanged).
    DiagUnshifted { kappa: f64 },
}

impl SympPrecond {
    /// The TT-layer resolution of [`Self::Auto`]: the D0 fixed unshifted inverse (auto `κ`),
    /// the best measured arm on every TT cell. Explicit
    /// variants pass through unchanged. Every TT-layer wrapper calls this exactly once at
    /// entry, so nothing downstream ever observes `Auto`.
    pub fn resolve_tt(self) -> Self {
        match self {
            SympPrecond::Auto => SympPrecond::DiagUnshifted { kappa: 0.0 },
            other => other,
        }
    }

    /// The joint-layer resolution of [`Self::Auto`]: no preconditioner — every SHIPPABLE fixed
    /// SPD model measured on the joint manifold ([`Self::DiagUnshifted`] among them) loses to the
    /// unpreconditioned lane there.
    pub fn resolve_joint(self) -> Self {
        match self {
            SympPrecond::Auto => SympPrecond::Off,
            other => other,
        }
    }

    /// The SECTOR-layer resolution of [`Self::Auto`] (the cross-sector readouts of
    /// [`crate::solvers::hess::SectorCtx`] and its SU(2) twin, through
    /// [`crate::solvers::hess::SectorOpts::precond`]): no preconditioner, until a measured table
    /// promotes the shifted fixed diagonal `D0^φ` (`SectorCtx::a_diagonal`; `Off` stays the
    /// default — one flip point for Rust, the C `QNTT_SECTOR_PRECOND_AUTO` and Julia's
    /// `:auto`). Explicit variants pass through unchanged; the sector entry points then accept
    /// `Off` and [`Self::DiagUnshifted`] on both layers (`D0^φ`; SU(2) `D0^{S'}`, the fixed inverse
    /// of `Su2SectorCtx::a_diagonal`, since rpa-floor-selection P4), and refuse every other arm by
    /// name.
    pub fn resolve_sector(self) -> Self {
        match self {
            SympPrecond::Auto => SympPrecond::Off,
            other => other,
        }
    }
}

/// The kernel behind the signed entry points ([`crate::solvers::hess::symplectic_lowest_signed_davidson`]
/// and its C64 twin, both layers), chosen by [`SymplecticDavidsonOpts::kernel`]; no other entry
/// point reads it. Both fill the same [`crate::solvers::hess::RpaReadout`]; its `report` says which
/// one ran.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RpaKernel {
    /// The paired-basis Davidson ([`crate::solvers::hess::paired_lowest_davidson`]): one
    /// Euclidean basis for both halves of `(X, Y)`, no Hessian block as a metric, so an indefinite
    /// `A ± B` changes nothing in the iteration. The default since 2026-11-02 (campaign
    /// `rpa-paired`). The wanted set is the signed kernel's (`Chase`, no positive window:
    /// every imaginary pair, the `k` lowest real `ω`, the complex pairs in their window, all
    /// corrected), with `max_basis` apply pairs (the budget is in pair-equivalents; the signed
    /// kernel spends two applies per basis vector, so it accepts `⌈max_basis/2⌉`) and the
    /// orthogonal hook. A knob it has no arm for runs the signed kernel on `Ii`
    /// with no probe instead: a preconditioner other than `Off` or the D0 diagonal,
    /// `guard_survival > 0`, `instrument`, or a column-deflation floor.
    #[default]
    Paired,
    /// The signed W-metric Davidson (`davidson_signed`): the metric block's signature tracked per
    /// basis vector. `metric` is the block that serves as `W` (`RpaMetric::Auto`
    /// probes both with `probe_budget` applies per block, `Ii` first; a regularized context takes
    /// `Ii` with no probe); the C64 layer's metric is always `Rr` and refuses `Ii`.
    Signed {
        metric: crate::solvers::hess::RpaMetric,
        probe_budget: usize,
    },
}

impl RpaKernel {
    /// The signed kernel on `metric` with no probe budget.
    pub fn signed(metric: crate::solvers::hess::RpaMetric) -> Self {
        RpaKernel::Signed {
            metric,
            probe_budget: 0,
        }
    }
}

/// Options for [`crate::solvers::hess::symplectic_lowest_signed_davidson`] /
/// [`crate::solvers::hess::symplectic_lowest_signed_davidson_c64`] (and the SU(2) twins), and of
/// the W-metric block Davidson behind the joint wrappers.
#[derive(Clone, Copy, Debug)]
pub struct SymplecticDavidsonOpts {
    /// Relative residual tolerance; its units depend on the entry.
    pub tol: f64,
    /// Thick-restart bound on the live basis dimension; `0` = auto (`max(8·(k+b), 32)` for
    /// block size `b`, clamped to `space_dim`). Too tight a bound limit-cycles, and which arm
    /// survives depends on the bound. Restart
    /// discards Krylov history, so prefer memory over restarts when the spectrum is hard.
    pub max_subspace: usize,
    /// Number of leading outer iterations whose expansions skip the preconditioner (the
    /// `davidson.rs` `nwarm` discipline — the first Ritz values are too poor to shift with).
    pub nwarm: usize,
    /// Total budget of the whole run in PAIR-EQUIVALENTS (one `(A, B)` apply pair each): the
    /// paired kernel builds at most `max_basis` pairs; the W-metric kernels (the signed one and
    /// the joint layer's unsigned one) accept at most `⌈max_basis/2⌉` basis vectors, two applies
    /// each (floored at `k`). Default 800.
    pub max_basis: usize,
    /// Which preconditioner the expansion applies to each correction — see [`SympPrecond`], where
    /// every arm carries its own evidence. Default [`SympPrecond::Auto`] (layer-resolved);
    /// the θ-shifted diagonal is REFUTED and was deleted.
    pub precond: SympPrecond,
    /// Collect the per-outer-iteration basis W-Gram diagnostic
    /// ([`crate::solvers::hess::SympDiagnostics::gram_min_eig`]): `m²` extra dots per outer
    /// iteration, ZERO operator applies, and no effect on the iteration itself — the Gram is
    /// read-only on the basis, so ω, residuals and every count are bit-identical either way.
    /// Default `false`. The cheap diagnostics (drop counters, survival ratios, collapse angles)
    /// are collected unconditionally. Unrelated to the crate's `instrument` cargo feature, which
    /// is Hamiltonian-construction telemetry.
    pub instrument: bool,
    /// SURVIVAL GUARD: collapse threshold on the trailing-window median of the MGS survival ratio
    /// ([`crate::solvers::hess::SympDiagnostics::survival`]). `0.0` = OFF, the default; a positive
    /// value arms the guard, and when the median of the last `b` (block size) accepted survival
    /// ratios falls strictly below it the preconditioner closure is discarded PERMANENTLY for the
    /// remainder of the solve, with the switch recorded in
    /// [`crate::solvers::hess::SympDiagnostics::guard_tripped`].
    pub guard_survival: f64,
    /// The kernel of the signed entry points ([`RpaKernel`], default `Paired`); no other entry
    /// point reads it.
    pub kernel: RpaKernel,
    /// The paired kernel's wanted-set rule on the signed entry points ([`WantedRank`], default
    /// `Omega`: the `k` lowest real ω). Opt-in, and no rule is guaranteed: under a curvature cut,
    /// B-dominated curvature modes (low Krein margin, large `ρ_A/ω`, no HF or TDA overlap) can take
    /// slots among the `k` lowest, and a selective rule skips them at the risk of skipping a genuine
    /// level that looks the same. On 18 calibration states whose tangent space holds the single
    /// excitations (hcho STO-3G, water 6-31G at equilibrium and stretched, `Hard(1e-3)` and
    /// `Hard(1e-4)`, FCI-labelled) the 8 lowest genuine levels were kept by `Omega` in 6,
    /// `AExcess(3)` in 12 and `TargetGateOrExcess(0.5, 3)` over a CIS target span in 16; every miss
    /// was a strongly correlated genuine level (`ρ_A/ω` up to 59 at stretched water). The pairs a
    /// selective rule passes over are listed in [`crate::solvers::hess::RpaReadout::passed_over`].
    /// A rank other than `Omega` needs the paired route (kernel `Paired` and every other knob
    /// paired-capable) and panics otherwise; `TargetGate` and `TargetGateOrExcess` need a target,
    /// [`crate::solvers::hess::symplectic_lowest_signed_davidson_targeted`].
    pub rank: WantedRank,
}

impl Default for SymplecticDavidsonOpts {
    fn default() -> Self {
        SymplecticDavidsonOpts {
            tol: 1e-8,
            max_subspace: 0,
            nwarm: 2,
            max_basis: 800,
            precond: SympPrecond::Auto,
            instrument: false,
            guard_survival: 0.0,
            kernel: RpaKernel::Paired,
            rank: WantedRank::Omega,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum CandidateSource {
    Unknown,
    Seed(usize),
    Correction {
        j: usize,
        theta: f64,
        preconditioned: bool,
    },
}

pub(crate) struct MetricRefusal {
    pub(crate) wnorm2: f64,
    pub(crate) gn_before: f64,
    pub(crate) gn_after: f64,
    pub(crate) source: CandidateSource,
    pub(crate) outer: usize,
    pub(crate) basis_len: usize,
    pub(crate) built: usize,
    pub(crate) ritz: Vec<f64>,
    pub(crate) residuals: Vec<f64>,
}

pub(crate) enum KernelErr {
    MetricNotPd(Box<MetricRefusal>),
}

impl KernelErr {
    fn with_context(
        self,
        source: CandidateSource,
        outer: usize,
        built: usize,
        ritz: &[f64],
        residuals: &[f64],
    ) -> Self {
        let KernelErr::MetricNotPd(mut r) = self;
        r.source = source;
        r.outer = outer;
        r.built = built;
        r.ritz = ritz.to_vec();
        r.residuals = residuals.to_vec();
        KernelErr::MetricNotPd(r)
    }
}

impl From<KernelErr> for SymplecticLowestError {
    fn from(e: KernelErr) -> Self {
        match e {
            KernelErr::MetricNotPd(r) => SymplecticLowestError::MetricNotPd { wnorm2: r.wnorm2 },
        }
    }
}

impl std::fmt::Debug for KernelErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let KernelErr::MetricNotPd(r) = self;
        f.debug_struct("MetricNotPd")
            .field("wnorm2", &r.wnorm2)
            .field("gn_before", &r.gn_before)
            .field("gn_after", &r.gn_after)
            .field("source", &r.source)
            .field("outer", &r.outer)
            .field("basis_len", &r.basis_len)
            .field("built", &r.built)
            .finish_non_exhaustive()
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_orthonormalize_push<V, W, D, Rp>(
    u: V,
    vs: &mut Vec<V>,
    ws: &mut Vec<V>,
    w_apply: &mut W,
    deflate: &mut D,
    reproject: &mut Rp,
    scale: f64,
    applies: &mut usize,
    diag: &mut SympDiagnostics,
    trace: bool,
) -> Result<bool, KernelErr>
where
    V: InnerProductVector,
    W: FnMut(&V) -> V,
    D: FnMut(&mut V),
    Rp: FnMut(&mut V),
{
    push_core(
        u, vs, ws, None, false, w_apply, deflate, reproject, scale, REFUSE_TAU, applies, diag,
        trace,
    )
    .map(|r| r.outcome.accepted())
}

pub(crate) const REFUSE_TAU: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PushOutcome {
    Accepted { sign: i8 },
    Dropped,
}

impl PushOutcome {
    pub(crate) fn accepted(self) -> bool {
        matches!(self, PushOutcome::Accepted { .. })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PushReport {
    pub(crate) outcome: PushOutcome,
    pub(crate) gn_before: f64,
    pub(crate) gn_after: f64,
    pub(crate) wnorm2: f64,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn push_core<V, W, D, Rp>(
    mut u: V,
    vs: &mut Vec<V>,
    ws: &mut Vec<V>,
    mut signs: Option<&mut Vec<i8>>,
    g_ortho: bool,
    w_apply: &mut W,
    deflate: &mut D,
    reproject: &mut Rp,
    scale: f64,
    tau: f64,
    applies: &mut usize,
    diag: &mut SympDiagnostics,
    trace: bool,
) -> Result<PushReport, KernelErr>
where
    V: InnerProductVector,
    W: FnMut(&V) -> V,
    D: FnMut(&mut V),
    Rp: FnMut(&mut V),
{
    reproject(&mut u);
    let gn_before = u.norm();
    let report = |outcome, gn_after, wnorm2| PushReport {
        outcome,
        gn_before,
        gn_after,
        wnorm2,
    };
    if gn_before == 0.0 {
        diag.drops_zero += 1;
        if trace {
            eprintln!("    push m={} gb=0 drop_zero", vs.len());
        }
        return Ok(report(PushOutcome::Dropped, 0.0, f64::NAN));
    }
    for _ in 0..2 {
        for (i, (vi, wi)) in vs.iter().zip(ws.iter()).enumerate() {
            let c = if g_ortho {
                vi.dot(&u).re_s()
            } else {
                let d = wi.dot(&u).re_s();
                match signs.as_deref() {
                    Some(sg) if sg[i] < 0 => -d,
                    _ => d,
                }
            };
            u.axpy(V::Scalar::from_f64(-c), vi);
        }
    }
    reproject(&mut u);
    let gn_after = u.norm();
    if gn_after <= 1e-10 * gn_before {
        diag.drops_rank += 1;
        if trace {
            eprintln!(
                "    push m={} gb={gn_before:.3e} ga={gn_after:.3e} surv={:.3e} drop_rank",
                vs.len(),
                gn_after / gn_before
            );
        }
        return Ok(report(PushOutcome::Dropped, gn_after, f64::NAN));
    }
    let mut wu = w_apply(&u);
    *applies += 1;
    deflate(&mut wu);
    let beta2 = u.dot(&wu).re_s();
    let trace_line = |verdict: &str| {
        eprintln!(
            "    push m={} gb={gn_before:.3e} ga={gn_after:.3e} surv={:.3e} b2={beta2:.3e} \
             ratio={:+.3e} {verdict}",
            vs.len(),
            gn_after / gn_before,
            beta2 / (gn_after * gn_after)
        );
    };
    if beta2 < -tau * gn_after * gn_after {
        if let Some(sg) = signs.as_deref_mut() {
            if trace {
                trace_line("accept_negative");
            }
            let s = V::Scalar::from_f64(if g_ortho {
                1.0 / gn_after
            } else {
                1.0 / (-beta2).sqrt()
            });
            u.scale(s);
            wu.scale(s);
            diag.survival.push(gn_after / gn_before);
            vs.push(u);
            ws.push(wu);
            sg.push(-1);
            return Ok(report(PushOutcome::Accepted { sign: -1 }, gn_after, beta2));
        }
        if trace {
            trace_line("REFUSE");
        }
        let basis_len = vs.len();
        return Err(KernelErr::MetricNotPd(Box::new(MetricRefusal {
            wnorm2: beta2,
            gn_before,
            gn_after,
            source: CandidateSource::Unknown,
            outer: 0,
            basis_len,
            built: 0,
            ritz: Vec::new(),
            residuals: Vec::new(),
        })));
    }
    let beta = beta2.max(0.0).sqrt();
    if beta <= 1e3 * f64::EPSILON * scale.max(f64::MIN_POSITIVE) {
        diag.drops_wnull += 1;
        if trace {
            trace_line("drop_wnull");
        }
        return Ok(report(PushOutcome::Dropped, gn_after, beta2));
    }
    if trace {
        trace_line("accept");
    }
    let s = V::Scalar::from_f64(if g_ortho { 1.0 / gn_after } else { 1.0 / beta });
    u.scale(s);
    wu.scale(s);
    diag.survival.push(gn_after / gn_before);
    vs.push(u);
    ws.push(wu);
    if let Some(sg) = signs {
        sg.push(1);
    }
    Ok(report(PushOutcome::Accepted { sign: 1 }, gn_after, beta2))
}

pub(crate) fn survival_guard_trips(survival: &[f64], window: usize, threshold: f64) -> bool {
    if threshold <= 0.0 || window == 0 || survival.len() < window {
        return false;
    }
    let mut tail = survival[survival.len() - window..].to_vec();
    tail.sort_by(f64::total_cmp);
    let median = if window % 2 == 1 {
        tail[window / 2]
    } else {
        0.5 * (tail[window / 2 - 1] + tail[window / 2])
    };
    median < threshold
}

pub(crate) fn recombine_cols<V: InnerProductVector + Send + Sync>(
    basis: &[V],
    n_out: usize,
    coeff: impl Fn(usize, usize) -> f64 + Sync,
) -> Vec<V> {
    let build = |j: usize| -> V {
        let mut acc = basis[0].clone();
        acc.scale(V::Scalar::from_f64(coeff(0, j)));
        for (i, b) in basis.iter().enumerate().skip(1) {
            acc.axpy(V::Scalar::from_f64(coeff(i, j)), b);
        }
        acc
    };
    if n_out < 2 || !par::par_driver_fork(basis[0].num_coeffs().saturating_mul(basis.len())) {
        return (0..n_out).map(build).collect();
    }
    (0..n_out).into_par_iter().map(build).collect()
}

pub(crate) fn recombine<V: InnerProductVector + Send + Sync>(
    basis: &[V],
    y: &[Vec<f64>],
    keep: usize,
) -> Vec<V> {
    recombine_cols(basis, keep, |i, j| y[j][i])
}

pub(crate) fn vector_budget(opts: &SymplecticDavidsonOpts) -> usize {
    opts.max_basis.div_ceil(2)
}

pub(crate) fn resolve_max_subspace(
    opts: &SymplecticDavidsonOpts,
    k: usize,
    b: usize,
    space_dim: usize,
) -> usize {
    if opts.max_subspace == 0 {
        (8 * (k + b)).max(32)
    } else {
        opts.max_subspace
    }
    .clamp((2 * (k + b)).min(space_dim), space_dim)
}

pub(crate) fn w_restart_compact<V: InnerProductVector + Send + Sync>(
    vs: &[V],
    ws: &[V],
    ms: &[V],
    y: &[Vec<f64>],
    keep: usize,
) -> (Vec<V>, Vec<V>, Vec<V>) {
    let mut new_vs = recombine(vs, y, keep);
    let mut new_ws = recombine(ws, y, keep);
    let mut new_ms = recombine(ms, y, keep);
    for j in 0..keep {
        let (v_head, v_tail) = new_vs.split_at_mut(j);
        let (w_head, w_tail) = new_ws.split_at_mut(j);
        let (m_head, m_tail) = new_ms.split_at_mut(j);
        let (vj, wj, mj) = (&mut v_tail[0], &mut w_tail[0], &mut m_tail[0]);
        for _ in 0..2 {
            for i in 0..j {
                let c = w_head[i].dot(vj).re_s();
                let sc = V::Scalar::from_f64(-c);
                vj.axpy(sc, &v_head[i]);
                wj.axpy(sc, &w_head[i]);
                mj.axpy(sc, &m_head[i]);
            }
        }
        let beta2 = vj.dot(wj).re_s();
        if beta2 > 0.0 {
            let s = V::Scalar::from_f64(1.0 / beta2.sqrt());
            vj.scale(s);
            wj.scale(s);
            mj.scale(s);
        }
    }
    (new_vs, new_ws, new_ms)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn omega_davidson<V, R, W, D, Rp, P>(
    r_apply: R,
    w_apply: W,
    deflate: D,
    reproject: Rp,
    precond: Option<P>,
    v0s: &[V],
    k: usize,
    space_dim: usize,
    opts: &SymplecticDavidsonOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError>
where
    V: InnerProductVector + Send + Sync,
    R: FnMut(&V) -> V,
    W: FnMut(&V) -> V,
    D: FnMut(&mut V),
    Rp: FnMut(&mut V),
    P: FnMut(&mut V, f64),
{
    omega_davidson_vectors(
        r_apply, w_apply, deflate, reproject, precond, v0s, k, space_dim, opts,
    )
    .map(|(omega2, _vecs, info)| (omega2, info))
    .map_err(SymplecticLowestError::from)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn omega_davidson_vectors<V, R, W, D, Rp, P>(
    mut r_apply: R,
    mut w_apply: W,
    mut deflate: D,
    mut reproject: Rp,
    mut precond: Option<P>,
    v0s: &[V],
    k: usize,
    space_dim: usize,
    opts: &SymplecticDavidsonOpts,
) -> Result<(Vec<f64>, Vec<V>, SymplecticLowestInfo), KernelErr>
where
    V: InnerProductVector + Send + Sync,
    R: FnMut(&V) -> V,
    W: FnMut(&V) -> V,
    D: FnMut(&mut V),
    Rp: FnMut(&mut V),
    P: FnMut(&mut V, f64),
{
    assert!(k >= 1, "omega_davidson: k must be ≥ 1");
    assert!(space_dim >= 1, "omega_davidson: empty iteration space");
    assert!(!v0s.is_empty(), "omega_davidson: empty start block");
    let b = v0s.len();
    let max_subspace = resolve_max_subspace(opts, k, b, space_dim);
    let max_basis = vector_budget(opts).max(k);
    let mut applies = 0usize;
    let mut total_built = 0usize;
    let mut diag = SympDiagnostics::default();

    let mut vs: Vec<V> = Vec::new();
    let mut ws: Vec<V> = Vec::new();
    let mut ms: Vec<V> = Vec::new();
    let omega_trace = crate::knobs::OMEGA_TRACE.get();

    for (i, v0) in v0s.iter().enumerate() {
        if vs.len() == space_dim {
            break;
        }
        if w_orthonormalize_push(
            v0.clone(),
            &mut vs,
            &mut ws,
            &mut w_apply,
            &mut deflate,
            &mut reproject,
            0.0,
            &mut applies,
            &mut diag,
            omega_trace,
        )
        .map_err(|e| e.with_context(CandidateSource::Seed(i), 0, total_built, &[], &[]))?
        {
            total_built += 1;
        }
    }
    assert!(
        !vs.is_empty(),
        "omega_davidson: every seed collapsed under deflation — caller bug"
    );
    while ms.len() < vs.len() {
        let mut m_img = r_apply(&ws[ms.len()]);
        applies += 1;
        reproject(&mut m_img);
        ms.push(m_img);
    }

    let mut evals: Vec<f64>;
    let mut evecs: Vec<Vec<f64>>;
    let mut residuals: Vec<f64>;
    let converged;
    let mut outer = 0usize;
    loop {
        let t_iter0 = std::time::Instant::now();
        let m = vs.len();
        diag.outer_iterations += 1;
        if opts.instrument {
            let g = gram_matrix_symmetrized(&ws, &vs);
            let (gev, _) = dense_self_adjoint_smallest::<f64>(&g, m, 0);
            diag.gram_min_eig.push(gev[0]);
        }
        let t = rr_matrix_symmetrized(&ws, &ms);
        let t_assembly = t_iter0.elapsed();
        let keep_target = (k + b).min(m);
        let want = k.min(m).max(keep_target);
        let t_eig0 = std::time::Instant::now();
        let (ev, yv) = dense_self_adjoint_smallest::<f64>(&t, m, want);
        evals = ev;
        evecs = yv;
        let t_eig = t_eig0.elapsed();
        let scale = evals
            .iter()
            .take(k.min(m))
            .fold(0.0f64, |acc, &e| acc.max(e.abs()))
            .max(f64::MIN_POSITIVE);

        let t_resid0 = std::time::Instant::now();
        let nwant = k.min(m);
        let rs = recombine(&ms, &evecs, nwant);
        let xs = recombine(&vs, &evecs, nwant);
        let mut res_vecs: Vec<V> = Vec::with_capacity(nwant);
        residuals = Vec::with_capacity(nwant);
        for (j, mut r) in rs.into_iter().enumerate() {
            r.axpy(V::Scalar::from_f64(-evals[j]), &xs[j]);
            residuals.push(r.norm());
            res_vecs.push(r);
        }
        let t_resid = t_resid0.elapsed();
        if omega_trace {
            let rmax = residuals.iter().fold(0.0f64, |a, &r| a.max(r));
            eprintln!(
                "  davidson outer={outer} m={m} built={total_built} θ0={:.6e} maxres={rmax:.3e}",
                evals[0]
            );
            eprintln!(
                "    phase timers: assembly={:.3}s eig={:.3}s resid={:.3}s",
                t_assembly.as_secs_f64(),
                t_eig.as_secs_f64(),
                t_resid.as_secs_f64(),
            );
        }
        if m >= k && residuals.iter().all(|&r| r <= opts.tol * scale) {
            converged = true;
            break;
        }
        if vs.len() == space_dim {
            converged = true;
            break;
        }
        if total_built >= max_basis {
            converged = false;
            break;
        }

        let t_restart0 = std::time::Instant::now();
        let did_restart = vs.len() >= max_subspace && vs.len() > keep_target;
        if did_restart {
            diag.restarts += 1;
            let keep = keep_target.min(evecs.len());
            (vs, ws, ms) = w_restart_compact(&vs, &ws, &ms, &evecs, keep);
            let mlen = vs.len();
            evecs = (0..keep)
                .map(|j| {
                    let mut e = vec![0.0; mlen];
                    e[j] = 1.0;
                    e
                })
                .collect();
        }
        let t_restart = t_restart0.elapsed();

        let t_expand0 = std::time::Instant::now();
        let use_precond = outer >= opts.nwarm;
        let mut accepted_this_round = 0usize;
        for (j, mut r) in res_vecs.into_iter().enumerate() {
            if vs.len() >= max_subspace.min(space_dim) || total_built >= max_basis {
                break;
            }
            if residuals[j] <= opts.tol * scale {
                continue;
            }
            if use_precond
                && precond.is_some()
                && survival_guard_trips(&diag.survival, b, opts.guard_survival)
            {
                precond = None;
                diag.guard_tripped = Some(diag.survival.len());
            }
            let preconditioned = use_precond && precond.is_some();
            if use_precond && let Some(p) = precond.as_mut() {
                p(&mut r, evals[j]);
            }
            let den = r.norm() * xs[j].norm();
            diag.collapse_angle.push(if den > 0.0 {
                r.dot(&xs[j]).abs_s() / den
            } else {
                f64::NAN
            });
            if w_orthonormalize_push(
                r,
                &mut vs,
                &mut ws,
                &mut w_apply,
                &mut deflate,
                &mut reproject,
                scale,
                &mut applies,
                &mut diag,
                omega_trace,
            )
            .map_err(|e| {
                let source = CandidateSource::Correction {
                    j,
                    theta: evals[j],
                    preconditioned,
                };
                e.with_context(source, outer, total_built, &evals, &residuals)
            })? {
                total_built += 1;
                accepted_this_round += 1;
                let mut m_img = r_apply(&ws[ws.len() - 1]);
                applies += 1;
                reproject(&mut m_img);
                ms.push(m_img);
            }
        }
        let t_expand = t_expand0.elapsed();
        if omega_trace {
            eprintln!(
                "    phase timers: restart={:.3}s ({}) expand={:.3}s ({accepted_this_round} \
                 accepted)  iter_total={:.3}s",
                t_restart.as_secs_f64(),
                if did_restart { "fired" } else { "skipped" },
                t_expand.as_secs_f64(),
                t_iter0.elapsed().as_secs_f64(),
            );
        }
        if accepted_this_round == 0 {
            converged = residuals.iter().all(|&r| r <= opts.tol * scale) && vs.len() >= k;
            break;
        }
        outer += 1;
    }

    let scale = evals
        .iter()
        .take(k)
        .fold(0.0f64, |acc, &e| acc.max(e.abs()))
        .max(f64::MIN_POSITIVE);
    let _ = scale;
    let omega2: Vec<f64> = evals.iter().take(k).copied().collect();
    let (negative, cert_applies) = super::lanczos::certify_negatives(
        &evals,
        k,
        |j| {
            (j < evecs.len() && !vs.is_empty())
                .then(|| recombine(&vs, &evecs[j..j + 1], 1).pop())
                .flatten()
        },
        &mut r_apply,
        &mut w_apply,
        &mut deflate,
        &mut reproject,
    );
    applies += cert_applies;
    residuals.truncate(omega2.len());
    let nv = omega2.len().min(evecs.len());
    let ritz = if vs.is_empty() || nv == 0 {
        Vec::new()
    } else {
        recombine(&vs, &evecs[..nv], nv)
    };
    let nret = omega2.len();
    Ok((
        omega2,
        ritz,
        SymplecticLowestInfo {
            converged,
            iterations: total_built,
            applies,
            residuals,
            negative,
            diag: Some(diag),
            signature: vec![1; nret],
            indefinite: None,
        },
    ))
}

#[cfg(test)]
mod w_orthonormalize_push_tests {
    use super::{SympDiagnostics, w_orthonormalize_push};
    use crate::core::SparseCore;
    use crate::qn::{QnSpace, RankGrid};
    use crate::solvers::InnerProductVector;

    fn vector(vals: &[f64]) -> SparseCore<f64> {
        let space = QnSpace::new(2, 1, 0);
        let mut r0 = RankGrid::zeros(&space);
        for s in space.bond_sectors(0) {
            r0[s] = 1;
        }
        let mut r1 = RankGrid::zeros(&space);
        for s in space.bond_sectors(1) {
            r1[s] = 1;
        }
        let mut c = SparseCore::<f64>::zeros(space, 0, &r0, &r1);
        for i in 0..c.num_coeffs() {
            c.set_coeff(i, vals[i % vals.len()]);
        }
        c
    }

    #[test]
    fn zero_and_wnull_drops_are_counted() {
        let mut diag = SympDiagnostics::default();
        let mut vs: Vec<SparseCore<f64>> = Vec::new();
        let mut ws: Vec<SparseCore<f64>> = Vec::new();
        let mut applies = 0usize;

        let u = vector(&[1.0, -0.5, 2.0]);
        let accepted = w_orthonormalize_push(
            u,
            &mut vs,
            &mut ws,
            &mut |v: &SparseCore<f64>| v.clone(),
            &mut |_: &mut SparseCore<f64>| {},
            &mut |v: &mut SparseCore<f64>| v.scale(0.0),
            1.0,
            &mut applies,
            &mut diag,
            false,
        )
        .expect("an annihilated candidate is a drop, not an error");
        assert!(!accepted);
        assert_eq!(
            diag.drops_zero, 1,
            "the annihilated candidate must be counted"
        );
        assert_eq!(applies, 0, "dropped before paying the W-apply");

        let u = vector(&[1.0, -0.5, 2.0]);
        let accepted = w_orthonormalize_push(
            u,
            &mut vs,
            &mut ws,
            &mut |v: &SparseCore<f64>| {
                let mut z = v.clone();
                z.scale(0.0);
                z
            },
            &mut |_: &mut SparseCore<f64>| {},
            &mut |_: &mut SparseCore<f64>| {},
            1.0,
            &mut applies,
            &mut diag,
            false,
        )
        .expect("a W-null candidate is a drop, not a MetricNotPd");
        assert!(!accepted);
        assert_eq!(diag.drops_wnull, 1, "the W-null candidate must be counted");
        assert_eq!(
            applies, 1,
            "the W-apply was paid before the W-seminorm test"
        );
        assert!(
            vs.is_empty() && ws.is_empty(),
            "rejected candidates never enter the basis"
        );
        assert!(
            diag.survival.is_empty(),
            "no acceptance, no survival record"
        );
        assert_eq!(
            diag.drops_rank + diag.drops_triple,
            0,
            "no other counter may fire"
        );
    }
}

#[cfg(test)]
mod metric_refusal_record_tests {
    use super::{
        CandidateSource, KernelErr, SympDiagnostics, SymplecticDavidsonOpts, omega_davidson,
        omega_davidson_vectors, w_orthonormalize_push,
    };
    use crate::core::SparseCore;
    use crate::qn::{QnSpace, RankGrid};
    use crate::solvers::InnerProductVector;
    use crate::solvers::hess::lanczos::SymplecticLowestError;

    fn vector<F: FnMut(usize) -> f64>(space: QnSpace, rank: usize, mut f: F) -> SparseCore<f64> {
        let mut r0 = RankGrid::zeros(&space);
        for s in space.bond_sectors(0) {
            r0[s] = rank;
        }
        let mut r1 = RankGrid::zeros(&space);
        for s in space.bond_sectors(1) {
            r1[s] = rank;
        }
        let mut c = SparseCore::<f64>::zeros(space, 0, &r0, &r1);
        for i in 0..c.num_coeffs() {
            c.set_coeff(i, f(i));
        }
        c
    }

    fn apply(a: &[f64], v: &SparseCore<f64>) -> SparseCore<f64> {
        let n = v.num_coeffs();
        let mut y = v.clone();
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..n {
                acc += a[i * n + j] * v.coeff(j);
            }
            y.set_coeff(i, acc);
        }
        y
    }

    #[test]
    fn push_refusal_carries_the_candidate_it_read() {
        let space = QnSpace::new(2, 1, 0);
        let u0 = vector(space, 1, |i| [1.0, -0.5, 2.0][i % 3]);
        let mut vs: Vec<SparseCore<f64>> = Vec::new();
        let mut ws: Vec<SparseCore<f64>> = Vec::new();
        let mut applies = 0usize;
        let mut diag = SympDiagnostics::default();
        let neg = |v: &SparseCore<f64>| {
            let mut y = v.clone();
            y.scale(-1.0);
            y
        };
        let err = w_orthonormalize_push(
            u0.clone(),
            &mut vs,
            &mut ws,
            &mut |v: &SparseCore<f64>| neg(v),
            &mut |_: &mut SparseCore<f64>| {},
            &mut |_: &mut SparseCore<f64>| {},
            1.0,
            &mut applies,
            &mut diag,
            false,
        )
        .expect_err("W = −I must refuse");
        let KernelErr::MetricNotPd(r) = err;
        assert_eq!(applies, 1, "the refusal is read after exactly one W-apply");
        assert_eq!(r.gn_before.to_bits(), u0.norm().to_bits());
        assert_eq!(r.gn_after.to_bits(), u0.norm().to_bits());
        assert_eq!(r.wnorm2.to_bits(), u0.dot(&neg(&u0)).to_bits());
        assert_eq!(r.basis_len, 0);
        assert_eq!(r.source, CandidateSource::Unknown, "the push cannot know");
        assert_eq!(
            diag.drops_zero + diag.drops_rank + diag.drops_wnull,
            0,
            "a refusal is a diagnosis, not a drop"
        );
        assert!(vs.is_empty() && ws.is_empty() && diag.survival.is_empty());
    }

    #[test]
    fn kernel_refusal_names_the_correction_and_keeps_the_ritz_data() {
        let space = QnSpace::new(4, 2, 1);
        let probe = vector(space, 2, |_| 0.0);
        let n = probe.num_coeffs();
        assert!(n >= 12, "fixture space too small: {n}");
        let mut r_mat = vec![0.0; n * n];
        let mut w_mat = vec![0.0; n * n];
        for i in 0..n {
            r_mat[i * n + i] = 1.0;
            w_mat[i * n + i] = if i == 0 {
                -1.0
            } else {
                1.0 + 0.5 * (i as f64 - 1.0)
            };
        }
        r_mat[1] = 0.5;
        r_mat[n] = 0.5;
        let seeds: Vec<SparseCore<f64>> = (0..3)
            .map(|j| {
                vector(space, 2, |i| {
                    if i == 0 {
                        0.0
                    } else {
                        1.0 + ((5 * i + 3 * j) % 7) as f64
                    }
                })
            })
            .collect();
        let opts = SymplecticDavidsonOpts {
            tol: 1e-12,
            max_basis: 8 * n,
            ..SymplecticDavidsonOpts::default()
        };
        let run = || {
            omega_davidson_vectors(
                |w: &SparseCore<f64>| apply(&r_mat, w),
                |v: &SparseCore<f64>| apply(&w_mat, v),
                |_: &mut SparseCore<f64>| {},
                |_: &mut SparseCore<f64>| {},
                None::<fn(&mut SparseCore<f64>, f64)>,
                &seeds,
                2,
                n,
                &opts,
            )
        };
        let Err(KernelErr::MetricNotPd(r)) = run() else {
            panic!("an indefinite W with a coupled negative direction must refuse")
        };
        println!(
            "kernel refusal: n={n} source={:?} outer={} basis_len={} built={} ratio={:.3e} \
             surv={:.3e}",
            r.source,
            r.outer,
            r.basis_len,
            r.built,
            r.wnorm2 / (r.gn_after * r.gn_after),
            r.gn_after / r.gn_before
        );
        assert!(
            matches!(r.source, CandidateSource::Correction { .. }),
            "the seeds are W-positive, so the refusal must be a correction: {:?}",
            r.source
        );
        assert!(r.outer >= 1 || r.basis_len >= 3, "past the warm start");
        assert!(
            !r.ritz.is_empty() && !r.residuals.is_empty(),
            "Ritz data kept"
        );
        assert!(r.wnorm2 < -1e-6 * r.gn_after * r.gn_after);
        let public = omega_davidson(
            |w: &SparseCore<f64>| apply(&r_mat, w),
            |v: &SparseCore<f64>| apply(&w_mat, v),
            |_: &mut SparseCore<f64>| {},
            |_: &mut SparseCore<f64>| {},
            None::<fn(&mut SparseCore<f64>, f64)>,
            &seeds,
            2,
            n,
            &opts,
        );
        match public {
            Err(SymplecticLowestError::MetricNotPd { wnorm2 }) => {
                assert_eq!(wnorm2.to_bits(), r.wnorm2.to_bits())
            }
            other => panic!("the public entry must refuse identically, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod survival_guard_tests {
    use super::{SymplecticDavidsonOpts, omega_davidson, survival_guard_trips};
    use crate::core::SparseCore;
    use crate::qn::{QnSpace, RankGrid};
    use crate::solvers::InnerProductVector;
    use std::cell::Cell;

    #[test]
    fn predicate_edges() {
        let collapsed = [1e-9, 1e-9, 1e-9];
        assert!(!survival_guard_trips(&collapsed, 3, 0.0));
        assert!(!survival_guard_trips(&collapsed, 3, -1.0));
        assert!(!survival_guard_trips(&collapsed, 0, 1e-5));
        assert!(!survival_guard_trips(&collapsed, 4, 1e-5));
        assert!(!survival_guard_trips(&[], 1, 1e-5));

        assert!(survival_guard_trips(&[1e-9, 1e-9, 1.0], 3, 1e-5));
        assert!(!survival_guard_trips(&[1e-9, 1.0, 1.0], 3, 1e-5));
        assert!(!survival_guard_trips(&[1e-9, 1e-9, 1.0, 1.0], 4, 1e-5));
        assert!(survival_guard_trips(&[1e-9, 1e-9, 1e-9, 1.0], 4, 1e-5));

        assert!(!survival_guard_trips(&[1e-12, 1.0, 1.0, 1.0], 3, 1e-5));
        assert!(survival_guard_trips(&[1.0, 1e-12, 1e-12, 1e-12], 3, 1e-5));

        assert!(!survival_guard_trips(&[1e-5, 1e-5, 1e-5], 3, 1e-5));
        assert!(survival_guard_trips(&[1e-5, 1e-5, 1e-5], 3, 1.0000001e-5));
    }

    fn vector<F: FnMut(usize) -> f64>(space: QnSpace, rank: usize, mut f: F) -> SparseCore<f64> {
        let mut r0 = RankGrid::zeros(&space);
        for s in space.bond_sectors(0) {
            r0[s] = rank;
        }
        let mut r1 = RankGrid::zeros(&space);
        for s in space.bond_sectors(1) {
            r1[s] = rank;
        }
        let mut c = SparseCore::<f64>::zeros(space, 0, &r0, &r1);
        for i in 0..c.num_coeffs() {
            c.set_coeff(i, f(i));
        }
        c
    }

    #[test]
    fn guard_trips_mid_solve_and_stays_off() {
        let space = QnSpace::new(4, 2, 1);
        const RANK: usize = 2;
        let probe = vector(space, RANK, |_| 0.0);
        let n = probe.num_coeffs();
        assert!(
            n >= 12,
            "the fixture space must be wide enough for a block iteration: got {n} coefficients"
        );

        let dia: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * (i as f64)).collect();
        let m_apply = |v: &SparseCore<f64>| {
            let mut y = v.clone();
            for (i, &d) in dia.iter().enumerate() {
                y.set_coeff(i, v.coeff(i) * d);
            }
            y
        };
        let seeds: Vec<SparseCore<f64>> = (0..3)
            .map(|j| vector(space, RANK, |i| 1.0 + ((7 * i + 3 * j) % 5) as f64))
            .collect();
        let mut swallow = seeds[0].clone();
        let s = 1.0 / swallow.norm();
        swallow.scale(s);

        let k = 2usize;
        let opts_at = |guard_survival: f64| SymplecticDavidsonOpts {
            nwarm: 0,
            max_basis: 120,
            guard_survival,
            ..SymplecticDavidsonOpts::default()
        };

        let run = |guard_survival: f64, budget: usize| {
            let calls = Cell::new(0usize);
            let precond = |t: &mut SparseCore<f64>, _theta: f64| {
                let used = calls.get();
                calls.set(used + 1);
                if used < budget {
                    let a = 1e6 * t.norm();
                    t.axpy(a, &swallow);
                }
            };
            let out = omega_davidson(
                m_apply,
                |v: &SparseCore<f64>| v.clone(),
                |_: &mut SparseCore<f64>| {},
                |_: &mut SparseCore<f64>| {},
                Some(precond),
                &seeds,
                k,
                n,
                &opts_at(guard_survival),
            )
            .expect("W = I is positive definite, so no metric diagnosis can fire");
            (out, calls.get())
        };

        let ((om_g, info_g), calls_g) = run(1e-5, usize::MAX);
        let diag_g = info_g
            .diag
            .as_ref()
            .expect("the Davidson reports diagnostics");
        let trip = diag_g.guard_tripped.expect(
            "the guard must fire: the closure buries every correction in a direction the basis \
             already holds, so the survival window collapses to ~1e-6 within one block",
        );
        let min_surv = diag_g
            .survival
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        println!(
            "guard_trips_mid_solve_and_stays_off: n={n} built={} trip={trip} precond calls={calls_g} \
             min surv={min_surv:.3e} drops r/w/z={}/{}/{}",
            info_g.iterations, diag_g.drops_rank, diag_g.drops_wnull, diag_g.drops_zero
        );
        assert!(
            trip >= 3,
            "a trip needs a full window (the 3-seed block) of accepted survival ratios: got {trip}"
        );
        assert!(
            calls_g >= 1,
            "this cell is worthless unless the trip is MID-solve — the closure must have acted \
             before the guard shut it off"
        );
        assert!(
            min_surv < 1e-5,
            "the synthetic arm must reproduce the measured signature — candidates ACCEPTED with a \
             collapsed survival ratio — or the trip is measuring something else: min survival was \
             {min_surv:.3e}"
        );

        let ((om_n, info_n), _) = run(0.0, calls_g);
        assert_eq!(om_g.len(), om_n.len());
        for (j, (a, b)) in om_g.iter().zip(om_n.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ω[{j}] differs, so the guarded run preconditioned a correction after the trip: \
                 {a:.17e} vs {b:.17e}"
            );
        }
        assert_eq!(info_g.iterations, info_n.iterations);
        assert_eq!(info_g.applies, info_n.applies);
        assert_eq!(info_g.converged, info_n.converged);
        assert_eq!(
            info_n.diag.as_ref().and_then(|d| d.guard_tripped),
            None,
            "guard_survival = 0.0 must never record a trip"
        );
    }
}

#[cfg(test)]
mod restart_stagnation_tests {
    use super::{SymplecticDavidsonOpts, omega_davidson_vectors};
    use crate::solvers::steihaug::testvec::{DVec, mat_apply};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::cell::Cell;

    fn spd(n: usize, lo: f64, hi: f64, rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
        let mut q: Vec<Vec<f64>> = Vec::new();
        while q.len() < n {
            let mut v: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            for _ in 0..2 {
                for u in &q {
                    let c: f64 = u.iter().zip(&v).map(|(a, b)| a * b).sum();
                    v.iter_mut().zip(u).for_each(|(x, y)| *x -= c * y);
                }
            }
            let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            q.push(v.iter().map(|a| a / nv).collect());
        }
        let mut a = vec![0.0; n * n];
        for (c, qc) in q.iter().enumerate() {
            let l = lo + (hi - lo) * c as f64 / (n - 1) as f64;
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] += l * qc[i] * qc[j];
                }
            }
        }
        a
    }

    #[test]
    fn stagnation_right_after_a_restart_returns_the_kept_ritz_vectors() {
        let n = 40;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x57A6);
        let r = spd(n, 0.5, 5.0, &mut rng);
        let w = spd(n, 0.3, 4.0, &mut rng);
        let v0s: Vec<DVec> = (0..3)
            .map(|_| DVec((0..n).map(|_| rng.random_range(-1.0..1.0)).collect()))
            .collect();
        let k = 2;
        let opts = SymplecticDavidsonOpts {
            tol: 1e-13,
            max_basis: 800,
            max_subspace: 12,
            nwarm: 0,
            ..SymplecticDavidsonOpts::default()
        };
        let mut restarted_runs = 0usize;
        for n_switch in 0..60usize {
            let calls = Cell::new(0usize);
            let pre = |t: &mut DVec, _th: f64| {
                if calls.get() >= n_switch {
                    t.0.iter_mut().for_each(|x| *x = 0.0);
                }
                calls.set(calls.get() + 1);
            };
            let (om, vecs, info) = omega_davidson_vectors(
                |v: &DVec| mat_apply(&r, n, v),
                |v: &DVec| mat_apply(&w, n, v),
                |_: &mut DVec| {},
                |_: &mut DVec| {},
                Some(pre),
                &v0s,
                k,
                n,
                &opts,
            )
            .expect("PD metric");
            let d = info.diag.as_ref().unwrap();
            for (j, (x, &th)) in vecs.iter().zip(&om).enumerate() {
                let wx = mat_apply(&w, n, x);
                let mx = mat_apply(&r, n, &wx);
                let res =
                    mx.0.iter()
                        .zip(&x.0)
                        .map(|(a, b)| (a - th * b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                let wn: f64 = x.0.iter().zip(&wx.0).map(|(a, b)| a * b).sum();
                assert!(
                    (res - info.residuals[j]).abs() <= 1e-8 * (1.0 + info.residuals[j]),
                    "n_switch {n_switch}, pair {j}: dense residual {res:.3e} vs reported {:.3e} \
                     (restarts {})",
                    info.residuals[j],
                    d.restarts
                );
                assert!(
                    (wn - 1.0).abs() <= 1e-8,
                    "n_switch {n_switch}, pair {j}: W-norm² {wn} (restarts {})",
                    d.restarts
                );
            }
            if d.restarts > 0 {
                restarted_runs += 1;
            }
        }
        assert!(
            restarted_runs >= 10,
            "only {restarted_runs} runs restarted at all"
        );
    }
}
