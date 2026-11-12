// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::tangent::TtTangent;
use crate::scalar::{C64, Scalar};
use crate::solvers::InnerProductVector;
use crate::vector::TtVector;
use std::sync::Arc;

use super::davidson::{REFUSE_TAU, RpaKernel, SympPrecond, SymplecticDavidsonOpts};
use super::davidson_paired::{
    AnomalyRule, PairedDavidsonOpts, PairedDavidsonOut, PairedMode, PairedOffset, PairedPrecond,
    WantedRank,
};
use super::davidson_signed::{SignedDavidsonOut, omega_davidson_signed};
use super::lanczos::{
    MetricEvent, NegativeWitness, SympDiagnostics, SymplecticLowestError, SymplecticLowestInfo,
};
use super::lobpcg::{SympLobpcgOpts, omega_lobp4dcg};
use super::newton_polish::{
    HessPart, PdVerdict, clean_tangent, deflate_radial, hessian_pd_probe, radial_mode,
    reproject_tangent,
};
use super::{CurvatureReg, WeingartenCtx};

pub(crate) fn omega_from_signed(
    omega2: &[f64],
    info: &SymplecticLowestInfo,
    kernel: &str,
) -> Result<Vec<f64>, SymplecticLowestError> {
    let scale = omega2
        .iter()
        .fold(0.0f64, |acc, &e| acc.max(e.abs()))
        .max(f64::MIN_POSITIVE);
    let mut omega = Vec::with_capacity(omega2.len());
    for &theta in omega2 {
        if theta < -1e-8 * scale {
            if let Some(w) = info.negative.iter().find(|w| w.theta == theta) {
                eprintln!(
                    "{kernel} NegativeRitz certificate: θ={theta:.6e}, fresh ⟨u,Ru⟩/⟨v,u⟩ = \
                     {:.6e} — {}",
                    w.fresh,
                    if w.certified() {
                        "GENUINE indefiniteness (variational witness); use the *_signed entry \
                         point to read it as a resonance"
                    } else {
                        "fresh quadratic form is NOT negative: projected problem corrupted \
                         (solver artifact, not physics)"
                    }
                );
            }
            return Err(SymplecticLowestError::NegativeRitz { theta });
        }
        omega.push(theta.max(0.0).sqrt());
    }
    Ok(omega)
}

/// Which block of `𝔥 = diag(H^rr, H^ii)` the W-metric Krylov kernels use as their METRIC.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RpaMetric {
    /// Probe both blocks and take the PD one, `H^ii` first. See [`select_rpa_metric`].
    #[default]
    Auto,
    /// Force `W = H^ii`, `R = H^rr` — the fixed assignment, valid wherever `H^ii ≻ 0`.
    Ii,
    /// Force `W = H^rr`, `R = H^ii` — **the swap**. Valid at a genuine minimizer of the real
    /// energy (`H^rr ≻ 0`) whose `H^ii` is indefinite: the physical RPA instability, which the
    /// fixed [`Self::Ii`] assignment refuses with `MetricNotPd`.
    Rr,
}

impl RpaMetric {
    pub(crate) fn y_sign(self) -> f64 {
        match self {
            RpaMetric::Ii => 1.0,
            RpaMetric::Rr => -1.0,
            RpaMetric::Auto => panic!("RpaMetric::Auto: the solve's metric was never resolved"),
        }
    }
}

macro_rules! impl_symplectic_drivers {
    (
        ctx: $ctx:ident,
        tangent: $tangent:ident,
        vec: $vec:ident,
        tangent_dim: $tangent_dim:path,
        paired: $paired:path,
        tag: $tag:literal,
        $(#[$doc_select_rpa_metric:meta])*
        $select:ident,
        $(#[$doc_symplectic_lowest_signed_davidson:meta])*
        symplectic_lowest_signed_davidson,
        $(#[$doc_symplectic_lowest_signed_davidson_targeted:meta])*
        symplectic_lowest_signed_davidson_targeted,
        $(#[$doc_symplectic_lowest_signed_davidson_c64:meta])*
        symplectic_lowest_signed_davidson_c64 $(,)?
    ) => {
        $(#[$doc_select_rpa_metric])*
        pub fn $select(
            ctx: &$ctx<'_, f64>,
            v0: &$tangent<f64>,
            budget: usize,
        ) -> (RpaMetric, PdVerdict, Option<PdVerdict>) {
            let (ii, _) = hessian_pd_probe(ctx, HessPart::Ii, v0, budget);
            match ii.is_pd() {
                Some(true) | None => (RpaMetric::Ii, ii, None),
                Some(false) => {
                    let (rr, _) = hessian_pd_probe(ctx, HessPart::Rr, v0, budget);
                    (RpaMetric::Rr, ii, Some(rr))
                }
            }
        }

        pub(crate) fn transport_seeds<T: Scalar>(
            base_l: &Arc<$vec<T>>,
            base_r: &Arc<$vec<T>>,
            radial: Option<&$tangent<T>>,
            seeds: &[$tangent<T>],
        ) -> Vec<$tangent<T>> {
            seeds
                .iter()
                .map(|v| {
                    let mut t = v.clone();
                    reproject_tangent(base_l, base_r, radial, &mut t);
                    t
                })
                .collect()
        }

        pub(crate) fn a_diagonal_squared<T: Scalar>(ctx: &$ctx<'_, T>) -> $tangent<T> {
            let mut d = ctx.a_diagonal();
            d.for_each_coeff_mut(|c| {
                let sq = *c * *c;
                *c = sq;
            });
            d
        }

        $(#[$doc_symplectic_lowest_signed_davidson])*
        pub fn symplectic_lowest_signed_davidson(
            ctx: &$ctx<'_, f64>,
            v0s: &[$tangent<f64>],
            k: usize,
            opts: &SymplecticDavidsonOpts,
        ) -> (
            crate::solvers::hess::RpaReadout,
            Vec<crate::solvers::hess::RpaAmplitudes<$tangent<f64>>>,
        ) {
            let (metric, probe_budget) = match opts.kernel {
                crate::solvers::hess::RpaKernel::Signed { metric, probe_budget } => {
                    (metric, probe_budget)
                }
                crate::solvers::hess::RpaKernel::Paired => (RpaMetric::Ii, 0),
            };
            if let Some((precond, popts)) = crate::solvers::hess::paired_route(opts) {
                let out = $paired(
                    ctx,
                    v0s,
                    &[],
                    k,
                    precond,
                    false,
                    crate::solvers::hess::PairedHook::Orthogonal,
                    &popts,
                );
                return crate::solvers::hess::readout_from_paired(out);
            }
            let (base_l, base_r) = ctx.shared_bases();
            let radial = radial_mode(ctx);
            let v0s_t = transport_seeds(base_l, base_r, radial.as_ref(), v0s);
            let v0s = &v0s_t[..];
            let space_dim = $tangent_dim(base_l) - radial.as_ref().map_or(0, |_| 1);
            let metric = match metric {
                RpaMetric::Auto if ctx.curvature() != CurvatureReg::Off => RpaMetric::Ii,
                RpaMetric::Auto => {
                    let seed = v0s
                        .first()
                        .expect(concat!($tag, "symplectic_lowest_signed_davidson: no seeds"));
                    $select(ctx, seed, probe_budget).0
                }
                m => m,
            };
            let precond_fn = build_symp_precond(opts.precond.resolve_tt(), || a_diagonal_squared(ctx));
            let out = match metric {
                RpaMetric::Rr => omega_davidson_signed(
                    |w| ctx.hmm_apply(w),
                    |v| ctx.hpp_apply(v),
                    |v| deflate_radial(radial.as_ref(), v),
                    |v| clean_tangent(radial.as_ref(), v),
                    precond_fn,
                    v0s,
                    k,
                    space_dim,
                    opts,
                    REFUSE_TAU,
                ),
                _ => omega_davidson_signed(
                    |w| ctx.hpp_apply(w),
                    |v| ctx.hmm_apply(v),
                    |v| deflate_radial(radial.as_ref(), v),
                    |v| clean_tangent(radial.as_ref(), v),
                    precond_fn,
                    v0s,
                    k,
                    space_dim,
                    opts,
                    REFUSE_TAU,
                ),
            };
            let applies = out.info.applies;
            crate::solvers::hess::readout_from_signed(out, metric, applies)
        }

        $(#[$doc_symplectic_lowest_signed_davidson_targeted])*
        pub fn symplectic_lowest_signed_davidson_targeted(
            ctx: &$ctx<'_, f64>,
            v0s: &[$tangent<f64>],
            target: &[$tangent<f64>],
            k: usize,
            opts: &SymplecticDavidsonOpts,
        ) -> (
            crate::solvers::hess::RpaReadout,
            Vec<crate::solvers::hess::RpaAmplitudes<$tangent<f64>>>,
        ) {
            let (precond, popts) = crate::solvers::hess::paired_route(opts).expect(concat!(
                $tag,
                "symplectic_lowest_signed_davidson_targeted: a target needs the paired route \
                 (kernel Paired, the preconditioner Off or D0, no survival guard, instrument or \
                 column-deflation floor)"
            ));
            let out = $paired(
                ctx,
                v0s,
                target,
                k,
                precond,
                false,
                crate::solvers::hess::PairedHook::Orthogonal,
                &popts,
            );
            crate::solvers::hess::readout_from_paired(out)
        }

        $(#[$doc_symplectic_lowest_signed_davidson_c64])*
        pub fn symplectic_lowest_signed_davidson_c64(
            ctx: &$ctx<'_, C64>,
            v0s: &[$tangent<C64>],
            k: usize,
            opts: &SymplecticDavidsonOpts,
        ) -> (
            crate::solvers::hess::RpaReadout,
            Vec<crate::solvers::hess::RpaAmplitudes<$tangent<C64>>>,
        ) {
            if let crate::solvers::hess::RpaKernel::Signed { metric: RpaMetric::Ii, .. } = opts.kernel {
                panic!(concat!(
                    $tag,
                    "symplectic_lowest_signed_davidson_c64: the realified pencil's metric is \
                     h = A + B (`Rr`); `Ii` names no block of this layer"
                ));
            }
            if let Some((precond, popts)) = crate::solvers::hess::paired_route(opts) {
                let out = $paired(
                    ctx,
                    v0s,
                    &[],
                    k,
                    precond,
                    false,
                    crate::solvers::hess::PairedHook::Orthogonal,
                    &popts,
                );
                return crate::solvers::hess::readout_from_paired(out);
            }
            let (base_l, base_r) = ctx.shared_bases();
            let radial = radial_mode(ctx);
            let v0s_t = transport_seeds(base_l, base_r, radial.as_ref(), v0s);
            let v0s = &v0s_t[..];
            let space_dim =
                2 * $tangent_dim(base_l) - radial.as_ref().map_or(0, |_| 2);
            let i_unit = C64::new(0.0, 1.0);
            let mi_unit = C64::new(0.0, -1.0);
            let precond_fn = build_symp_precond(opts.precond.resolve_tt(), || a_diagonal_squared(ctx));
            let out = omega_davidson_signed(
                |w: &$tangent<C64>| {
                    let jw = w.scale(i_unit);
                    let hjw = ctx.hess_apply(&jw);
                    hjw.scale(mi_unit)
                },
                |v| ctx.hess_apply(v),
                |v| deflate_radial(radial.as_ref(), v),
                |v| clean_tangent(radial.as_ref(), v),
                precond_fn,
                v0s,
                k,
                space_dim,
                opts,
                REFUSE_TAU,
            );
            let applies = out.info.applies;
            crate::solvers::hess::readout_from_signed(out, RpaMetric::Rr, applies)
        }
    };
}

pub(crate) use impl_symplectic_drivers;

impl_symplectic_drivers!(
    ctx: WeingartenCtx,
    tangent: TtTangent,
    vec: TtVector,
    tangent_dim: crate::core::tangent::tangent_dim,
    paired: super::davidson_paired::paired_lowest_davidson,
    tag: "",
    select_rpa_metric,
    symplectic_lowest_signed_davidson,
    symplectic_lowest_signed_davidson_targeted,
    symplectic_lowest_signed_davidson_c64,
);

pub(crate) fn paired_route(
    opts: &SymplecticDavidsonOpts,
) -> Option<(PairedPrecond, PairedDavidsonOpts)> {
    let route = paired_route_any(opts);
    assert!(
        route.is_some() || opts.rank == WantedRank::Omega,
        "SymplecticDavidsonOpts::rank = {:?} needs the paired route (kernel Paired, the \
         preconditioner Off or D0, no survival guard, instrument or column-deflation floor)",
        opts.rank
    );
    route
}

fn paired_route_any(opts: &SymplecticDavidsonOpts) -> Option<(PairedPrecond, PairedDavidsonOpts)> {
    if !matches!(opts.kernel, RpaKernel::Paired) || opts.guard_survival > 0.0 || opts.instrument {
        return None;
    }
    let precond = match opts.precond.resolve_tt() {
        SympPrecond::Off => PairedPrecond::Off,
        SympPrecond::DiagUnshifted { kappa } => PairedPrecond::Unshifted { kappa },
        #[allow(unreachable_patterns)]
        _ => return None,
    };
    Some((
        precond,
        PairedDavidsonOpts {
            tol: opts.tol,
            max_basis: opts.max_basis,
            max_subspace: opts.max_subspace,
            nwarm: opts.nwarm,
            positive_window: false,
            anomalies: AnomalyRule::Chase,
            rank: opts.rank,
        },
    ))
}

pub(crate) fn readout_from_paired<V>(
    out: PairedDavidsonOut<V>,
) -> (RpaReadout, Vec<RpaAmplitudes<V>>)
where
    V: InnerProductVector,
{
    let i = out.info;
    let mut modes = Vec::with_capacity(out.real.len() + out.complex.len());
    let mut amps = Vec::with_capacity(modes.capacity());
    for m in &out.real {
        let imaginary = m.omega2().re < 0.0;
        modes.push(RpaMode {
            omega: m.omega,
            omega2: m.omega2(),
            kind: if imaginary {
                RpaModeKind::Imaginary
            } else {
                RpaModeKind::Real
            },
            krein: if imaginary { 0.0 } else { m.krein },
            residual: m.residual,
            certificate: None,
        });
        amps.push(amplitudes_of(m));
    }
    for m in &out.complex {
        modes.push(RpaMode {
            omega: m.omega,
            omega2: m.omega2(),
            kind: RpaModeKind::Complex,
            krein: 0.0,
            residual: m.residual,
            certificate: None,
        });
        amps.push(amplitudes_of(m));
    }
    for w in i.negative {
        modes[w.index].certificate = Some(w);
    }
    let passed_over = out
        .untargeted_below
        .iter()
        .map(|m| PassedOver {
            omega: m.omega.re,
            krein: m.krein,
            rho_a_over_omega: m.rho_a / m.omega.re,
            residual: m.residual,
        })
        .collect();
    (
        RpaReadout {
            modes,
            passed_over,
            converged: i.converged,
            iterations: i.built,
            applies: i.apply_pairs,
            report: RpaReport::Paired {
                offset: i.offset,
                exhausted: i.exhausted,
                stalled: i.stalled,
                restarts: i.restarts,
            },
        },
        amps,
    )
}

fn amplitudes_of<V: InnerProductVector>(m: &PairedMode<V>) -> RpaAmplitudes<V> {
    let mut a = RpaAmplitudes {
        x: m.x_re.clone(),
        y: m.y_re.clone(),
        x_im: m.x_im.clone(),
        y_im: m.y_im.clone(),
    };
    a.normalize();
    a
}

pub(crate) fn readout_from_signed<V>(
    out: SignedDavidsonOut<V>,
    metric: RpaMetric,
    applies: usize,
) -> (RpaReadout, Vec<RpaAmplitudes<V>>)
where
    V: InnerProductVector,
{
    let y_sign = metric.y_sign();
    let info = out.info;
    let n = out.omega2.len() + out.complex.len();
    let mut modes = Vec::with_capacity(n);
    let mut amps = Vec::with_capacity(n);
    let pencil = |r: f64, q2: f64, w2: f64, w_abs2: f64| r / (w2 + w_abs2 * q2).sqrt();
    for (j, ((&theta, q), wq)) in out
        .omega2
        .iter()
        .zip(out.vectors)
        .zip(out.w_vectors)
        .enumerate()
    {
        let residual = pencil(
            info.residuals[j],
            q.norm().powi(2),
            wq.norm().powi(2),
            theta.abs(),
        );
        let a = amplitudes_from_metric_side(C64::new(theta, 0.0), q, None, wq, None, y_sign);
        let imaginary = theta < 0.0;
        modes.push(RpaMode {
            omega: C64::new(theta, 0.0).sqrt(),
            omega2: C64::new(theta, 0.0),
            kind: if imaginary {
                RpaModeKind::Imaginary
            } else {
                RpaModeKind::Real
            },
            krein: if imaginary { 0.0 } else { a.krein_margin() },
            residual,
            certificate: None,
        });
        amps.push(a);
    }
    for c in out.complex {
        let q2 = c.x_re.norm().powi(2) + c.x_im.norm().powi(2);
        let w2 = c.w_re.norm().powi(2) + c.w_im.norm().powi(2);
        let residual = pencil(c.residual, q2, w2, c.theta.norm());
        modes.push(RpaMode {
            omega: c.theta.sqrt(),
            omega2: c.theta,
            kind: RpaModeKind::Complex,
            krein: 0.0,
            residual,
            certificate: None,
        });
        amps.push(amplitudes_from_metric_side(
            c.theta,
            c.x_re,
            Some(c.x_im),
            c.w_re,
            Some(c.w_im),
            y_sign,
        ));
    }
    for w in info.negative {
        modes[w.index].certificate = Some(w);
    }
    let report = info.indefinite.unwrap_or_default();
    (
        RpaReadout {
            passed_over: Vec::new(),
            modes,
            converged: info.converged,
            iterations: info.iterations,
            applies,
            report: RpaReport::Signed {
                metric,
                events: report.events,
                n_negative_basis: report.n_negative_basis,
                general_path: report.general_path,
                diag: info.diag,
            },
        },
        amps,
    )
}

pub(crate) fn amplitudes_from_metric_side<V: InnerProductVector>(
    theta: C64,
    q_re: V,
    q_im: Option<V>,
    w_re: V,
    w_im: Option<V>,
    y_sign: f64,
) -> RpaAmplitudes<V> {
    let f = |a: f64| V::Scalar::from_f64(a);
    let combo = |terms: &[(f64, &V)]| -> V {
        let mut v = terms[0].1.clone();
        v.scale(f(terms[0].0));
        for (c, t) in &terms[1..] {
            v.axpy(f(*c), t);
        }
        v
    };
    let mut a = if theta.im == 0.0 && theta.re > 0.0 {
        let w = theta.re.sqrt();
        RpaAmplitudes {
            x: combo(&[(0.5, &q_re), (0.5 / w, &w_re)]),
            y: combo(&[(-0.5 * y_sign, &q_re), (0.5 * y_sign / w, &w_re)]),
            x_im: None,
            y_im: None,
        }
    } else if theta == C64::new(0.0, 0.0) {
        RpaAmplitudes {
            x: combo(&[(0.5, &q_re)]),
            y: combo(&[(-0.5 * y_sign, &q_re)]),
            x_im: None,
            y_im: None,
        }
    } else {
        let inv = C64::new(1.0, 0.0) / theta.sqrt();
        let (cr, ci) = (inv.re, inv.im);
        let p_re = match &w_im {
            Some(wi) => combo(&[(cr, &w_re), (-ci, wi)]),
            None => combo(&[(cr, &w_re)]),
        };
        let p_im = match &w_im {
            Some(wi) => combo(&[(ci, &w_re), (cr, wi)]),
            None => combo(&[(ci, &w_re)]),
        };
        let half = |p: &V, q: Option<&V>, sp: f64, sq: f64| match q {
            Some(q) => combo(&[(sp, p), (sq, q)]),
            None => combo(&[(sp, p)]),
        };
        RpaAmplitudes {
            x: half(&p_re, Some(&q_re), 0.5, 0.5),
            y: half(&p_re, Some(&q_re), 0.5 * y_sign, -0.5 * y_sign),
            x_im: Some(half(&p_im, q_im.as_ref(), 0.5, 0.5)),
            y_im: Some(half(&p_im, q_im.as_ref(), 0.5 * y_sign, -0.5 * y_sign)),
        }
    };
    a.normalize();
    a
}

/// What kind of eigenvalue an RPA mode is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RpaModeKind {
    /// A real `ω ≥ 0`; its Krein margin says excitation (`> 0`) or de-excitation (`< 0`).
    Real,
    /// `ω = iν`, `ω² = −ν² < 0`: a resonance, the reference is unstable along it.
    Imaginary,
    /// A complex `ω` (first quadrant) of a quartet `±ω, ±ω̄`: both `A + B` and `A − B` are
    /// indefinite on the span.
    Complex,
}

/// One mode of an RPA solve ([`RpaReadout::modes`]); its amplitudes are the entry of the
/// [`RpaAmplitudes`] list at the same index.
#[derive(Clone, Copy, Debug)]
pub struct RpaMode {
    /// `ω`: real (`im = 0`) for [`RpaModeKind::Real`], `iν` for `Imaginary`, first quadrant for
    /// `Complex`.
    pub omega: C64,
    /// `ω²`, each kernel's own number (the signed kernel solves for `ω²`, the paired one for `ω`;
    /// the other is its exact relation).
    pub omega2: C64,
    pub kind: RpaModeKind,
    /// Krein margin `(‖X‖² − ‖Y‖²)/(‖X‖² + ‖Y‖²)` of a real mode: `+1` with `Y = 0`, negative
    /// for a de-excitation, `1/|margin|` the eigenvalue's condition relative to a normal one.
    /// `0` for an imaginary or complex mode (Krein-neutral).
    pub krein: f64,
    /// The pencil residual `‖Hz − ωΣz‖/‖z‖` of the returned amplitudes (Ha), the same quantity
    /// from both kernels.
    pub residual: f64,
    /// The fresh-apply certificate of an imaginary mode: `Some` for every `ω² < 0`
    /// beyond the kernel's floor, `None` otherwise. Gate a published `iν` on
    /// [`NegativeWitness::certified`].
    pub certificate: Option<NegativeWitness>,
}

/// What the kernel that ran met, in its own terms.
#[derive(Clone, Debug)]
pub enum RpaReport {
    /// The paired Davidson ([`RpaKernel::Paired`]).
    Paired {
        offset: PairedOffset,
        exhausted: bool,
        stalled: bool,
        restarts: usize,
    },
    /// The signed W-metric Davidson ([`RpaKernel::Signed`]).
    Signed {
        metric: RpaMetric,
        events: Vec<MetricEvent>,
        n_negative_basis: usize,
        general_path: bool,
        diag: Option<SympDiagnostics>,
    },
}

impl RpaReport {
    /// The kernel that ran: `"paired"` or `"signed"`.
    pub fn kernel_name(&self) -> &'static str {
        match self {
            RpaReport::Paired { .. } => "paired",
            RpaReport::Signed { .. } => "signed",
        }
    }
}

/// The result of an RPA solve, from either kernel: the modes (every imaginary one, the `k` lowest
/// real ones ascending, then the complex ones by `Re ω²`) and the kernel's report. The entries
/// return it beside the modes' amplitudes.
#[derive(Clone, Debug)]
pub struct RpaReadout {
    pub modes: Vec<RpaMode>,
    /// The real pairs a selective wanted set passed over ([`SymplecticDavidsonOpts::rank`] other
    /// than `Omega`, or a target): eligible, below the largest wanted `ω`, not wanted, ascending in
    /// `ω`. Reported, never corrected — their numbers are the last Ritz values, not converged
    /// ones. Empty under `Omega` without a target and on the signed kernel.
    pub passed_over: Vec<PassedOver>,
    /// Every wanted mode met the kernel's `tol`.
    pub converged: bool,
    /// Basis vectors built.
    pub iterations: usize,
    /// Operator applies, in `(A, B)` pairs (one signed `H^±` apply is one pair's work).
    pub applies: usize,
    pub report: RpaReport,
}

/// One real pair a selective wanted set passed over ([`RpaReadout::passed_over`]), at its last
/// Ritz values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PassedOver {
    /// `ω` (real, `> 0`).
    pub omega: f64,
    /// Krein margin `(‖X‖² − ‖Y‖²)/(‖X‖² + ‖Y‖²)`.
    pub krein: f64,
    /// `ρ_A/ω` with `ρ_A = XᵀAX/XᵀX`: the [`WantedRank::AExcess`] test reads `ρ_A ≤ c·ω`.
    pub rho_a_over_omega: f64,
    /// The last Ritz residual `‖Hz − ωΣz‖` at `‖z‖ = 1`.
    pub residual: f64,
}

impl RpaReadout {
    pub(crate) fn empty(converged: bool, kernel: RpaKernel) -> Self {
        RpaReadout {
            passed_over: Vec::new(),
            modes: Vec::new(),
            converged,
            iterations: 0,
            applies: 0,
            report: match kernel {
                RpaKernel::Paired => RpaReport::Paired {
                    offset: PairedOffset::default(),
                    exhausted: false,
                    stalled: false,
                    restarts: 0,
                },
                RpaKernel::Signed { metric, .. } => RpaReport::Signed {
                    metric,
                    events: Vec::new(),
                    n_negative_basis: 0,
                    general_path: false,
                    diag: None,
                },
            },
        }
    }

    /// The signed `ω²` of the real and imaginary modes (complex ones excluded), ascending.
    pub fn omega2(&self) -> Vec<f64> {
        self.modes
            .iter()
            .filter(|m| m.kind != RpaModeKind::Complex)
            .map(|m| m.omega2.re)
            .collect()
    }

    /// Every mode's pencil residual ([`RpaMode::residual`]), in mode order.
    pub fn residuals(&self) -> Vec<f64> {
        self.modes.iter().map(|m| m.residual).collect()
    }

    /// The complex modes.
    pub fn complex(&self) -> impl Iterator<Item = &RpaMode> {
        self.modes.iter().filter(|m| m.kind == RpaModeKind::Complex)
    }

    /// Anything a stable point would not show: an imaginary or complex mode, a real mode with a
    /// non-positive Krein margin (a de-excitation), an outstanding negativity certificate, a metric
    /// event (signed kernel) or a nonzero counting offset (paired kernel).
    pub fn anomalous(&self) -> bool {
        self.modes
            .iter()
            .any(|m| m.kind != RpaModeKind::Real || m.krein <= 0.0 || m.certificate.is_some())
            || match &self.report {
                RpaReport::Paired { offset, .. } => offset.total() > 0,
                RpaReport::Signed { events, .. } => !events.is_empty(),
            }
    }

    /// **The publication gate**: the excitation energies `ω`, ascending — `Some` only for a
    /// converged, non-empty, non-[anomalous](Self::anomalous) solve. Criticality is not
    /// stability: an anomalous solve is published mode by mode (its `omega2`, certificates and
    /// [`crate::solvers::hess::rpa_mode_checks`]), never as a real spectrum.
    pub fn omega(&self) -> Option<Vec<f64>> {
        (self.converged && !self.modes.is_empty() && !self.anomalous())
            .then(|| self.modes.iter().map(|m| m.omega.re).collect())
    }

    /// Every imaginary mode's certificate is certified (vacuously true without one). **Gate a
    /// published `iν` on this**.
    pub fn all_negatives_certified(&self) -> bool {
        self.modes
            .iter()
            .filter_map(|m| m.certificate)
            .all(|w| w.certified())
    }

    /// The number of imaginary modes.
    pub fn n_negative(&self) -> usize {
        self.modes
            .iter()
            .filter(|m| m.kind == RpaModeKind::Imaginary)
            .count()
    }

    /// The side `q` of the amplitudes a metric-side quantity reads ([`RpaAmplitudes::metric_side`]):
    /// the signed kernel's metric, `Ii` for the paired kernel.
    pub fn side(&self) -> RpaMetric {
        match &self.report {
            RpaReport::Signed { metric, .. } => *metric,
            RpaReport::Paired { .. } => RpaMetric::Ii,
        }
    }

    /// Metric events of a signed solve (`0` for the paired kernel, which has no metric).
    pub fn metric_events(&self) -> usize {
        match &self.report {
            RpaReport::Signed { events, .. } => events.len(),
            RpaReport::Paired { .. } => 0,
        }
    }
}

/// One mode's RPA amplitudes `z = (X, Y)`, an eigenvector of the pencil `H z = ω Σ z`:
/// `A X + B Y = ω X` and `B X + A Y = −ω Y`, aligned with an entry of [`RpaReadout::modes`].
/// Both kernels return the pair they solved for, so a consumer never rebuilds one half from the
/// other: the paired kernel holds `(X, Y)` itself, the signed kernel completes its metric-side
/// vector `q` with its own W-image (`p = Wq/ω`). The mode's `residual` is this pair's.
#[derive(Clone, Debug)]
pub struct RpaAmplitudes<V> {
    /// `X` (its real part for an imaginary or complex `ω`).
    pub x: V,
    /// `Y` (its real part for an imaginary or complex `ω`).
    pub y: V,
    /// The imaginary part of `X`; `Some` only for an imaginary or complex `ω`.
    pub x_im: Option<V>,
    /// The imaginary part of `Y`; `Some` only for an imaginary or complex `ω`.
    pub y_im: Option<V>,
}

impl<V: InnerProductVector> RpaAmplitudes<V> {
    /// The metric-side vector `q` of the product `M = R∘W`: `X − Y` under [`RpaMetric::Ii`]
    /// (`W = A − B`), `X + Y` under [`RpaMetric::Rr`] (`W = A + B`). For an imaginary or complex
    /// `ω` the larger of the real and imaginary parts'. Not
    /// normalized. Panics on [`RpaMetric::Auto`].
    pub fn metric_side(&self, metric: RpaMetric) -> V {
        let c = V::Scalar::from_f64(-metric.y_sign());
        let mut q = self.x.clone();
        q.axpy(c, &self.y);
        if let (Some(xi), Some(yi)) = (&self.x_im, &self.y_im) {
            let mut qi = xi.clone();
            qi.axpy(c, yi);
            if qi.norm() > q.norm() {
                q = qi;
            }
        }
        q
    }

    /// `‖X‖² − ‖Y‖²` (real parts plus imaginary parts): the Krein norm, `0` for an imaginary or
    /// complex `ω`.
    pub fn krein_norm(&self) -> f64 {
        let n2 = |v: &Option<V>| v.as_ref().map_or(0.0, |v| v.norm().powi(2));
        self.x.norm().powi(2) + n2(&self.x_im) - self.y.norm().powi(2) - n2(&self.y_im)
    }

    /// `(‖X‖² − ‖Y‖²)/(‖X‖² + ‖Y‖²)`, the Krein margin ([`RpaMode::krein`]).
    pub fn krein_margin(&self) -> f64 {
        self.krein_norm() / self.norm().powi(2)
    }

    /// `‖z‖ = (‖X‖² + ‖Y‖²)^½`.
    pub fn norm(&self) -> f64 {
        let n2 = |v: &Option<V>| v.as_ref().map_or(0.0, |v| v.norm().powi(2));
        (self.x.norm().powi(2) + n2(&self.x_im) + self.y.norm().powi(2) + n2(&self.y_im)).sqrt()
    }

    fn scale_all(&mut self, c: f64) {
        let c = V::Scalar::from_f64(c);
        self.x.scale(c);
        self.y.scale(c);
        for v in [&mut self.x_im, &mut self.y_im].into_iter().flatten() {
            v.scale(c);
        }
    }

    fn normalize(&mut self) {
        let k = self.krein_norm();
        if self.x_im.is_none() && k.is_finite() && k != 0.0 {
            self.scale_all(1.0 / k.abs().sqrt());
        } else {
            let n = self.norm();
            if n > 0.0 && n.is_finite() {
                self.scale_all(1.0 / n);
            }
        }
    }
}

pub(crate) fn build_symp_precond<V: InnerProductVector>(
    choice: SympPrecond,
    model: impl FnOnce() -> V,
) -> Option<impl FnMut(&mut V, f64)> {
    let d = match choice {
        SympPrecond::Auto => {
            panic!("unresolved SympPrecond::Auto reached the preconditioner factory")
        }
        SympPrecond::Off => return None,
        SympPrecond::DiagUnshifted { kappa } => {
            let mut d = model();
            crate::solvers::davidson::precond_fixed_inverse(&mut d, kappa);
            d
        }
    };
    Some(move |t: &mut V, _theta: f64| t.zip_coeffs_mut(&d, |ti, di| *ti *= di))
}

/// Diagnostic accessor for the PRODUCTION D0 fixed inverse `T` at `ctx`
/// ([`SympPrecond::DiagUnshifted`]): exactly the vector [`build_symp_precond`] hands the kernels,
/// i.e. [`crate::solvers::davidson::precond_fixed_inverse`] applied to [`a_diagonal_squared`].
/// Exposed so tests outside this crate's module tree can pin the clamp, the floor `c` and the
/// `κ(T) ≤ κ+1` bound on the operator the solver actually uses rather than on a reimplementation
/// of it. Not part of any solving path.
pub fn unshifted_precond_inverse_diagnostic<T: Scalar>(
    ctx: &WeingartenCtx<'_, T>,
    kappa: f64,
) -> TtTangent<T> {
    let mut d = a_diagonal_squared(ctx);
    crate::solvers::davidson::precond_fixed_inverse(&mut d, kappa);
    d
}

/// The lowest `k` symplectic (RPA) eigenvalues ω at `ctx`'s ground state — the **non-default**
/// paired locally-optimal block CG (ELOBP4dCG), real `f64` layer: the same `W`-metric product
/// `M = H^rr∘H^ii` and the same [`SympPrecond`] preconditioner ladder as
/// the W-metric block Davidson ([`super::davidson::omega_davidson`], the signed entry's kernel on
/// a definite metric), but a fixed-dimension search space rebuilt every iteration
/// instead of a growing basis with thick restarts — constant memory, and no restart to interact
/// with the preconditioner. Block size `= v0s.len()`, which must be `≥ k` (shifting deflation is
/// not implemented). Certified at fixture scale against the dense
/// [`crate::solvers::hess::symplectic_eigenvalues`] oracle and against the Davidson kernel; it is
/// NOT the default solver. Derivation and deferrals: [`super::lobpcg`].
pub fn symplectic_lowest_lobp4dcg(
    ctx: &WeingartenCtx<'_, f64>,
    v0s: &[TtTangent<f64>],
    k: usize,
    opts: &SympLobpcgOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = radial_mode(ctx);
    let v0s_t = transport_seeds(base_l, base_r, radial.as_ref(), v0s);
    let v0s = &v0s_t[..];
    let space_dim = crate::core::tangent::tangent_dim(base_l) - radial.as_ref().map_or(0, |_| 1);
    let precond_fn = build_symp_precond(opts.precond.resolve_tt(), || a_diagonal_squared(ctx));
    let (omega2, info) = omega_lobp4dcg(
        |w| ctx.hpp_apply(w),
        |v| ctx.hmm_apply(v),
        |v| deflate_radial(radial.as_ref(), v),
        |v| clean_tangent(radial.as_ref(), v),
        precond_fn,
        v0s,
        k,
        space_dim,
        opts,
    )?;
    let omega = omega_from_signed(&omega2, &info, "omega_lobp4dcg")?;
    Ok((omega, info))
}

#[cfg(test)]
mod build_symp_precond_tests {
    use super::{SympPrecond, build_symp_precond};
    use crate::core::SparseCore;
    use crate::qn::{QnSpace, RankGrid};
    use std::cell::Cell;

    fn model_vector(vals: &[f64]) -> SparseCore<f64> {
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
        let n = c.num_coeffs();
        assert!(n > 1, "degenerate fixture: {n} coefficient(s)");
        for i in 0..n {
            c.set_coeff(i, vals[i % vals.len()]);
        }
        c
    }

    #[test]
    fn d0_closure_is_theta_independent() {
        let d = model_vector(&[4.0, 0.25, -1e-3, 9.0]);
        let mut p = build_symp_precond(SympPrecond::DiagUnshifted { kappa: 0.0 }, || d.clone())
            .expect("D0 builds a closure");
        let t0 = model_vector(&[1.0, -2.0, 0.5, 3.0]);
        let (mut a, mut b) = (t0.clone(), t0.clone());
        p(&mut a, 0.0);
        p(&mut b, 1.234e6);
        for i in 0..t0.num_coeffs() {
            assert_eq!(
                a.coeff(i).to_bits(),
                b.coeff(i).to_bits(),
                "D0 output depends on θ at coefficient {i}: {} vs {}",
                a.coeff(i),
                b.coeff(i)
            );
        }
        for i in 0..t0.num_coeffs() {
            let (x, y) = (t0.coeff(i), a.coeff(i));
            assert!(x * y > 0.0, "coefficient {i} changed sign: {x} → {y}");
        }
    }

    #[test]
    #[should_panic(expected = "unresolved SympPrecond::Auto reached the preconditioner factory")]
    fn unresolved_auto_reaching_the_factory_panics() {
        let _ = build_symp_precond(SympPrecond::Auto, || model_vector(&[1.0, 2.0]));
    }

    #[test]
    fn off_lane_never_builds_the_model() {
        let built = Cell::new(0usize);
        let p = build_symp_precond(SympPrecond::Off, || {
            built.set(built.get() + 1);
            model_vector(&[1.0])
        });
        assert!(p.is_none(), "Off must hand the kernel no preconditioner");
        assert_eq!(built.get(), 0, "Off must not build the diagonal model");
    }

    #[test]
    fn non_off_arms_build_the_model_once() {
        for choice in [SympPrecond::DiagUnshifted { kappa: 1e3 }] {
            let built = Cell::new(0usize);
            let mut p = build_symp_precond(choice, || {
                built.set(built.get() + 1);
                model_vector(&[4.0, 0.25, -1e-3, 9.0])
            })
            .expect("every non-Off arm builds a closure");
            let mut t = model_vector(&[1.0, -2.0, 0.5, 3.0]);
            for it in 0..4 {
                p(&mut t, 0.1 * it as f64);
            }
            drop(p);
            assert_eq!(
                built.get(),
                1,
                "{choice:?} built the model {} times",
                built.get()
            );
        }
    }

    #[test]
    fn d0_stays_finite_at_degenerate_edges() {
        let mut p = build_symp_precond(SympPrecond::DiagUnshifted { kappa: 0.0 }, || {
            model_vector(&[-1.0, -2.0, -3.0])
        })
        .expect("D0 builds a closure");
        let mut t = model_vector(&[1.0, 1.0, 0.0]);
        p(&mut t, 0.0);
        for i in 0..t.num_coeffs() {
            assert!(
                t.coeff(i).is_finite(),
                "all-nonpositive model produced a non-finite entry at {i}: {}",
                t.coeff(i)
            );
        }

        let mut p = build_symp_precond(SympPrecond::DiagUnshifted { kappa: 1e300 }, || {
            model_vector(&[1e-30, 0.0, 1e-30])
        })
        .expect("D0 builds a closure");
        let mut t = model_vector(&[1.0, 0.0, 1.0]);
        p(&mut t, 0.0);
        for i in 0..t.num_coeffs() {
            assert!(
                t.coeff(i).is_finite(),
                "huge-kappa/small-dmax model produced a non-finite entry at {i}: {}",
                t.coeff(i)
            );
        }
    }
}

#[cfg(test)]
mod t8s_negatives_published_tests {
    use super::{RpaMode, RpaModeKind, RpaReadout, RpaReport, omega_from_signed};
    use crate::scalar::C64;
    use crate::solvers::hess::lanczos::{SymplecticLowestError, SymplecticLowestInfo};
    use crate::solvers::hess::{NegativeWitness, RpaMetric};

    const OMEGA2: [f64; 2] = [-1.0, 3.0];

    fn info(negative: Vec<NegativeWitness>) -> SymplecticLowestInfo {
        SymplecticLowestInfo {
            converged: true,
            iterations: 4,
            applies: 9,
            residuals: vec![1e-12, 1e-12],
            negative,
            diag: None,
            signature: vec![1, 1],
            indefinite: None,
        }
    }

    #[test]
    fn t8s_omega_from_signed_refuses_the_analytic_resonance_either_way() {
        let certified = info(vec![NegativeWitness {
            index: 0,
            theta: -1.0,
            fresh: -1.0,
            fresh_w: 1.0,
        }]);
        match omega_from_signed(&OMEGA2, &certified, "t8s") {
            Err(SymplecticLowestError::NegativeRitz { theta }) => {
                assert_eq!(theta, -1.0, "the refusal names the offending ω², not ω")
            }
            other => panic!("a genuine negative ω² must be NegativeRitz, got {other:?}"),
        }
        let artifact = info(vec![NegativeWitness {
            index: 0,
            theta: -1.0,
            fresh: 0.25,
            fresh_w: 1.0,
        }]);
        assert!(
            matches!(
                omega_from_signed(&OMEGA2, &artifact, "t8s"),
                Err(SymplecticLowestError::NegativeRitz { theta }) if theta == -1.0
            ),
            "an UNcertified negative refuses the same way — the certificate is a message, not a \
             branch"
        );
        let signed = |i: SymplecticLowestInfo| {
            let mut modes: Vec<RpaMode> = OMEGA2
                .iter()
                .map(|&w2| RpaMode {
                    omega: C64::new(w2, 0.0).sqrt(),
                    omega2: C64::new(w2, 0.0),
                    kind: if w2 < 0.0 {
                        RpaModeKind::Imaginary
                    } else {
                        RpaModeKind::Real
                    },
                    krein: if w2 < 0.0 { 0.0 } else { 1.0 },
                    residual: 1e-12,
                    certificate: None,
                })
                .collect();
            for w in i.negative {
                modes[w.index].certificate = Some(w);
            }
            RpaReadout {
                passed_over: Vec::new(),
                modes,
                converged: i.converged,
                iterations: i.iterations,
                applies: i.applies,
                report: RpaReport::Signed {
                    metric: RpaMetric::Rr,
                    events: Vec::new(),
                    n_negative_basis: 0,
                    general_path: false,
                    diag: None,
                },
            }
        };
        assert!(signed(certified).all_negatives_certified());
        assert!(!signed(artifact).all_negatives_certified());
        assert!(
            signed(info(Vec::new())).all_negatives_certified(),
            "no negatives ⇒ vacuously certified"
        );
    }

    #[test]
    fn t8s_readout_omega_withholds_every_anomaly() {
        use super::super::davidson_paired::PairedOffset;
        use crate::solvers::hess::MetricEvent;
        let real = |w2: f64, krein: f64| RpaMode {
            omega: C64::new(w2, 0.0).sqrt(),
            omega2: C64::new(w2, 0.0),
            kind: RpaModeKind::Real,
            krein,
            residual: 1e-12,
            certificate: None,
        };
        let paired = |modes: Vec<RpaMode>, neg: (usize, usize)| RpaReadout {
            passed_over: Vec::new(),
            modes,
            converged: true,
            iterations: 4,
            applies: 4,
            report: RpaReport::Paired {
                offset: PairedOffset {
                    neg_apb: neg.0,
                    neg_amb: neg.1,
                    neg_a: 0,
                },
                exhausted: false,
                stalled: false,
                restarts: 0,
            },
        };
        let stable = paired(vec![real(2.0, 1.0), real(12.0, 0.8)], (0, 0));
        let w = stable.omega().expect("the stable control publishes");
        assert!((w[0] - 2.0f64.sqrt()).abs() <= 1e-15 && (w[1] - 12.0f64.sqrt()).abs() <= 1e-15);
        assert!(!stable.anomalous());
        let imaginary = RpaMode {
            omega: C64::new(0.0, 1.0),
            omega2: C64::new(-1.0, 0.0),
            kind: RpaModeKind::Imaginary,
            krein: 0.0,
            residual: 1e-12,
            certificate: None,
        };
        let complex = RpaMode {
            omega: C64::new(1.5, 0.3).sqrt(),
            omega2: C64::new(1.5, 0.3),
            kind: RpaModeKind::Complex,
            krein: 0.0,
            residual: 1e-12,
            certificate: None,
        };
        let witnessed = RpaMode {
            certificate: Some(NegativeWitness {
                index: 0,
                theta: -1.0,
                fresh: -1.0,
                fresh_w: 1.0,
            }),
            ..real(2.0, 1.0)
        };
        let rows: Vec<(&str, RpaReadout)> = vec![
            (
                "imaginary",
                paired(vec![imaginary, real(12.0, 1.0)], (0, 0)),
            ),
            ("complex", paired(vec![real(2.0, 1.0), complex], (0, 0))),
            (
                "de-excitation",
                paired(vec![real(2.0, 1.0), real(12.0, -0.5)], (0, 0)),
            ),
            (
                "offset",
                paired(vec![real(2.0, 1.0), real(12.0, 1.0)], (1, 0)),
            ),
            (
                "certificate",
                paired(vec![witnessed, real(12.0, 1.0)], (0, 0)),
            ),
            (
                "metric event",
                RpaReadout {
                    passed_over: Vec::new(),
                    report: RpaReport::Signed {
                        metric: RpaMetric::Ii,
                        events: vec![MetricEvent {
                            basis_len: 3,
                            gn_before: 1.0,
                            gn_after: 0.9,
                            wnorm2: -4e-7,
                            accepted: false,
                        }],
                        n_negative_basis: 0,
                        general_path: false,
                        diag: None,
                    },
                    ..stable.clone()
                },
            ),
            (
                "unconverged",
                RpaReadout {
                    passed_over: Vec::new(),
                    converged: false,
                    ..stable.clone()
                },
            ),
            ("empty", paired(Vec::new(), (0, 0))),
        ];
        for (what, r) in rows {
            assert!(r.omega().is_none(), "{what}: ω published");
        }
    }

    #[test]
    fn t8s_omega_from_signed_clamps_inside_the_relative_roundoff_band() {
        let inside = [-1e-12, 3.0];
        let omega = omega_from_signed(&inside, &info(Vec::new()), "t8s").expect("inside the band");
        assert_eq!(omega[0], 0.0, "a roundoff negative clamps to ω = 0");
        assert!((omega[1] - 3.0f64.sqrt()).abs() <= 1e-15);
        let outside = [-3.1e-8, 3.0];
        assert!(
            matches!(
                omega_from_signed(&outside, &info(Vec::new()), "t8s"),
                Err(SymplecticLowestError::NegativeRitz { .. })
            ),
            "the band is relative to max|ω²| = 3, so −3.1e-8 is outside it"
        );
    }
}

#[cfg(test)]
mod orthogonal_hook_tests {
    use crate::core::tangent::TtTangent;
    use crate::qn::QnSpace;
    use crate::solvers::InnerProductVector;
    use crate::solvers::hess::SympDiagnostics;
    use crate::solvers::hess::davidson::{PushOutcome, push_core};
    use crate::solvers::hess::{
        clean_tangent, radial_mode, reproject_tangent, weingarten_operator,
    };
    use crate::test_util::{build_h_matvec, randomized_state, seeded};
    use crate::vector::TtVector;

    pub(crate) fn near_dependent_push<V, H>(
        basis: &[V],
        extra: &V,
        delta: f64,
        hook: &mut H,
    ) -> (f64, f64)
    where
        V: InnerProductVector<Scalar = f64> + Clone,
        H: FnMut(&mut V),
    {
        let (mut vs, mut ws, mut signs) = (Vec::new(), Vec::new(), Vec::<i8>::new());
        let (mut applies, mut diag) = (0usize, SympDiagnostics::default());
        let mut w = |v: &V| v.clone();
        let mut no_deflate = |_: &mut V| {};
        for b in basis {
            let rep = push_core(
                b.clone(),
                &mut vs,
                &mut ws,
                Some(&mut signs),
                false,
                &mut w,
                &mut no_deflate,
                hook,
                1.0,
                1e-6,
                &mut applies,
                &mut diag,
                false,
            )
            .expect("W = id never refuses");
            assert_eq!(
                rep.outcome,
                PushOutcome::Accepted { sign: 1 },
                "premise: independent basis vectors"
            );
        }
        let m = vs.len();
        let mut u = vs[0].clone();
        u.axpy(delta, extra);
        let rep = push_core(
            u,
            &mut vs,
            &mut ws,
            Some(&mut signs),
            false,
            &mut w,
            &mut no_deflate,
            hook,
            1.0,
            1e-6,
            &mut applies,
            &mut diag,
            false,
        )
        .expect("W = id never refuses");
        assert_eq!(
            rep.outcome,
            PushOutcome::Accepted { sign: 1 },
            "premise: the near-dependent candidate is accepted (survival above the 1e-10 rank drop)"
        );
        let v = &vs[m];
        let overlap = vs[..m]
            .iter()
            .map(|vi| vi.dot(v).abs() / (vi.norm() * v.norm()))
            .fold(0.0f64, f64::max);
        (rep.gn_after / rep.gn_before, overlap)
    }

    #[test]
    fn near_dependent_push_stays_orthogonal_only_with_the_orthogonal_hook() {
        let space = QnSpace::new(4, 2, 1);
        let h = build_h_matvec(space);
        let mut rng = seeded(0x0B11);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let n = x.norm();
        x.scale_mut(1.0 / n);
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.shared_bases();
        let radial = radial_mode(&ctx);
        let tangent = |rng: &mut _| -> TtTangent<f64> {
            let d: TtVector<f64> = randomized_state(space, 1, 2, rng);
            TtTangent::project_shared(bl, br, &d)
        };
        let basis: Vec<TtTangent<f64>> = (0..3).map(|_| tangent(&mut rng)).collect();
        let extra = tangent(&mut rng);
        let delta = 1e-9;
        let (s_orth, ov_orth) = near_dependent_push(&basis, &extra, delta, &mut |v| {
            clean_tangent(radial.as_ref(), v)
        });
        let (s_obl, ov_obl) = near_dependent_push(&basis, &extra, delta, &mut |v| {
            reproject_tangent(bl, br, radial.as_ref(), v)
        });
        println!(
            "abelian: survival {s_orth:.2e} / {s_obl:.2e}; overlap with the basis: orthogonal hook \
             {ov_orth:.2e}, oblique hook {ov_obl:.2e}"
        );
        assert!(
            (1e-10..1e-7).contains(&s_orth),
            "premise: a near-dependent candidate (survival {s_orth:e})"
        );
        assert!(
            ov_orth <= 1e-12,
            "orthogonal hook: the accepted vector must stay orthogonal ({ov_orth:e})"
        );
        assert!(
            ov_obl >= 1e-10,
            "anti-oracle: the oblique hook must visibly break it ({ov_obl:e})"
        );
    }

    #[test]
    fn su2_near_dependent_push_stays_orthogonal_only_with_the_orthogonal_hook() {
        use crate::su2::tangent::Su2TtTangent;
        use crate::su2::vector::Su2TtVector;
        let x = crate::su2::border_fixture::base_state();
        let h = crate::test_util::generic_su2_h::<f64>(x.space);
        let ctx = crate::su2::hess::weingarten_operator(&h, &x);
        let (bl, br) = ctx.shared_bases();
        let radial = crate::su2::hess::radial_mode(&ctx);
        let mut rng = seeded(0x0B12);
        let space = x.space;
        let mut tangent = || -> Su2TtTangent<f64> {
            let d: Su2TtVector<f64> = crate::test_util::randomized_su2_state(space, 1, 2, &mut rng);
            Su2TtTangent::project(bl, br, &d)
        };
        let basis: Vec<Su2TtTangent<f64>> = (0..3).map(|_| tangent()).collect();
        let extra = tangent();
        let delta = 1e-10;
        let (s_orth, ov_orth) = near_dependent_push(&basis, &extra, delta, &mut |v| {
            crate::su2::hess::clean_tangent(radial.as_ref(), v)
        });
        let (s_obl, ov_obl) = near_dependent_push(&basis, &extra, delta, &mut |v| {
            crate::su2::hess::reproject_tangent(bl, br, radial.as_ref(), v)
        });
        println!(
            "su2: survival {s_orth:.2e} / {s_obl:.2e}; overlap with the basis: orthogonal hook \
             {ov_orth:.2e}, oblique hook {ov_obl:.2e}"
        );
        assert!(
            (1e-10..1e-7).contains(&s_orth),
            "premise: a near-dependent candidate (survival {s_orth:e})"
        );
        assert!(
            ov_orth <= 1e-12,
            "orthogonal hook: the accepted vector must stay orthogonal ({ov_orth:e})"
        );
        assert!(
            ov_obl >= 1e-10,
            "anti-oracle: the oblique hook must visibly break it ({ov_obl:e})"
        );
    }

    fn flat_bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn omega2_gap(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q).abs() / p.abs().max(q.abs()).max(f64::MIN_POSITIVE))
            .fold(0.0f64, f64::max)
    }

    #[test]
    fn signed_entry_is_the_kernel_with_the_orthogonal_hook() {
        use super::{a_diagonal_squared, symplectic_lowest_signed_davidson, transport_seeds};
        use crate::core::border_fixture::base_state;
        use crate::solvers::hess::davidson::{REFUSE_TAU, SympPrecond, SymplecticDavidsonOpts};
        use crate::solvers::hess::davidson_signed::omega_davidson_signed;
        use crate::solvers::hess::{RpaMetric, build_symp_precond, deflate_radial};
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        let _lock = crate::par::lock_par_switches();
        let x = base_state();
        let h = crate::test_util::generic_truncated_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.shared_bases();
        let radial = radial_mode(&ctx);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4C9);
        let k = 2;
        let v0s: Vec<TtTangent<f64>> = (0..k + 2)
            .map(|_| {
                TtTangent::project_shared(
                    bl,
                    br,
                    &crate::vector::tt_randn(x.space, &x.ranks, &mut rng),
                )
            })
            .collect();
        let space_dim = crate::core::tangent::tangent_dim(bl) - radial.as_ref().map_or(0, |_| 1);
        let hook = |orthogonal: bool| {
            let radial = radial.as_ref();
            move |v: &mut TtTangent<f64>| {
                if orthogonal {
                    clean_tangent(radial, v)
                } else {
                    reproject_tangent(bl, br, radial, v)
                }
            }
        };
        let opts_for = |precond: SympPrecond| SymplecticDavidsonOpts {
            tol: 1e-12,
            max_basis: 32,
            precond,
            kernel: crate::solvers::hess::RpaKernel::signed(RpaMetric::Ii),
            ..SymplecticDavidsonOpts::default()
        };
        let replay = |orthogonal: bool, opts: &SymplecticDavidsonOpts| {
            let seeds = transport_seeds(bl, br, radial.as_ref(), &v0s);
            let precond =
                build_symp_precond(opts.precond.resolve_tt(), || a_diagonal_squared(&ctx));
            omega_davidson_signed(
                |w: &TtTangent<f64>| ctx.hpp_apply(w),
                |v: &TtTangent<f64>| ctx.hmm_apply(v),
                |v: &mut TtTangent<f64>| deflate_radial(radial.as_ref(), v),
                hook(orthogonal),
                precond,
                &seeds,
                k,
                space_dim,
                opts,
                REFUSE_TAU,
            )
        };
        let mut gaps = Vec::new();
        for precond in [SympPrecond::Auto, SympPrecond::Off] {
            let opts = opts_for(precond);
            let (so, vecs) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
            let orth = replay(true, &opts);
            let obl = replay(false, &opts);
            let gap = omega2_gap(&obl.omega2, &orth.omega2);
            println!(
                "{precond:?}: entry {:?} ({} iters, {} applies) | orthogonal replay {:?} | oblique \
                 replay {:?} ({} iters) — oblique vs orthogonal omega2 gap {gap:.2e}",
                so.omega2(),
                so.iterations,
                so.applies,
                orth.omega2,
                obl.omega2,
                obl.info.iterations
            );
            let (orth_applies, obl_applies) = (orth.info.applies, obl.info.applies);
            let (orth, orth_amps) =
                crate::solvers::hess::readout_from_signed(orth, RpaMetric::Ii, orth_applies);
            let (obl, _) =
                crate::solvers::hess::readout_from_signed(obl, RpaMetric::Ii, obl_applies);
            assert!(
                flat_bits(&obl.omega2()) != flat_bits(&so.omega2())
                    || flat_bits(&obl.residuals()) != flat_bits(&so.residuals())
                    || obl.iterations != so.iterations,
                "{precond:?} premise: the oblique replay is bitwise the entry — this fixture \
                 cannot tell the two hooks apart"
            );
            assert!(
                flat_bits(&orth.omega2()) == flat_bits(&so.omega2()),
                "{precond:?}: the entry's omega2 is not the orthogonal-hook kernel's"
            );
            assert_eq!(orth.iterations, so.iterations, "{precond:?}: iterations");
            assert_eq!(orth.applies, so.applies, "{precond:?}: applies");
            assert!(
                flat_bits(&orth.residuals()) == flat_bits(&so.residuals()),
                "{precond:?}: residuals"
            );
            assert_eq!(orth_amps.len(), vecs.len(), "{precond:?}: mode count");
            for (j, (p, q)) in orth_amps.iter().zip(&vecs).enumerate() {
                let same = |a: &TtTangent<f64>, b: &TtTangent<f64>| {
                    (0..a.num_coeffs()).all(|i| a.coeff(i).to_bits() == b.coeff(i).to_bits())
                };
                assert!(
                    same(&p.x, &q.x) && same(&p.y, &q.y) && q.x_im.is_none() == p.x_im.is_none(),
                    "{precond:?}: mode {j}'s amplitudes differ from the orthogonal replay's"
                );
            }
            gaps.push(gap);
        }
        println!(
            "oblique-vs-orthogonal omega2 gap: D0 {:.2e}, Off {:.2e}",
            gaps[0], gaps[1]
        );
        assert!(
            gaps[0] >= 1e-6,
            "D0: the hook acts at first order on a preconditioned correction (gap {:e})",
            gaps[0]
        );
        assert!(
            gaps[1] <= 1e-9,
            "Off: the hook acts at roundoff on an unpreconditioned solve (gap {:e})",
            gaps[1]
        );
    }

    #[test]
    fn su2_signed_entry_is_the_kernel_with_the_orthogonal_hook() {
        use crate::solvers::hess::davidson::{REFUSE_TAU, SympPrecond, SymplecticDavidsonOpts};
        use crate::solvers::hess::davidson_signed::omega_davidson_signed;
        use crate::solvers::hess::{RpaMetric, build_symp_precond};
        use crate::su2::hess::{
            a_diagonal_squared, deflate_radial, symplectic_lowest_signed_davidson, transport_seeds,
        };
        use crate::su2::tangent::Su2TtTangent;
        use crate::su2::vector::Su2TtVector;
        let _lock = crate::par::lock_par_switches();
        let x = crate::su2::border_fixture::base_state();
        let h = crate::test_util::generic_su2_h::<f64>(x.space);
        let ctx = crate::su2::hess::weingarten_operator(&h, &x);
        let (bl, br) = ctx.shared_bases();
        let radial = crate::su2::hess::radial_mode(&ctx);
        let mut rng = seeded(0x4CA);
        let space = x.space;
        let k = 2;
        let v0s: Vec<Su2TtTangent<f64>> = (0..k + 2)
            .map(|_| {
                let d: Su2TtVector<f64> =
                    crate::test_util::randomized_su2_state(space, 1, 2, &mut rng);
                Su2TtTangent::project(bl, br, &d)
            })
            .collect();
        let space_dim = crate::su2::weingarten::tangent_dim(bl) - radial.as_ref().map_or(0, |_| 1);
        let hook = |orthogonal: bool| {
            let radial = radial.as_ref();
            move |v: &mut Su2TtTangent<f64>| {
                if orthogonal {
                    crate::su2::hess::clean_tangent(radial, v)
                } else {
                    crate::su2::hess::reproject_tangent(bl, br, radial, v)
                }
            }
        };
        let opts_for = |precond: SympPrecond| SymplecticDavidsonOpts {
            tol: 1e-12,
            max_basis: 32,
            precond,
            kernel: crate::solvers::hess::RpaKernel::signed(RpaMetric::Ii),
            ..SymplecticDavidsonOpts::default()
        };
        let replay = |orthogonal: bool, opts: &SymplecticDavidsonOpts| {
            let seeds = transport_seeds(bl, br, radial.as_ref(), &v0s);
            let precond =
                build_symp_precond(opts.precond.resolve_tt(), || a_diagonal_squared(&ctx));
            omega_davidson_signed(
                |w: &Su2TtTangent<f64>| ctx.hpp_apply(w),
                |v: &Su2TtTangent<f64>| ctx.hmm_apply(v),
                |v: &mut Su2TtTangent<f64>| deflate_radial(radial.as_ref(), v),
                hook(orthogonal),
                precond,
                &seeds,
                k,
                space_dim,
                opts,
                REFUSE_TAU,
            )
        };
        let mut gaps = Vec::new();
        for precond in [SympPrecond::Auto, SympPrecond::Off] {
            let opts = opts_for(precond);
            let (so, vecs) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
            let orth = replay(true, &opts);
            let obl = replay(false, &opts);
            let gap = omega2_gap(&obl.omega2, &orth.omega2);
            println!(
                "su2 {precond:?}: entry {:?} ({} iters, {} applies) | orthogonal replay {:?} | \
                 oblique replay {:?} ({} iters) — oblique vs orthogonal omega2 gap {gap:.2e}",
                so.omega2(),
                so.iterations,
                so.applies,
                orth.omega2,
                obl.omega2,
                obl.info.iterations
            );
            let (orth_applies, obl_applies) = (orth.info.applies, obl.info.applies);
            let (orth, orth_amps) =
                crate::solvers::hess::readout_from_signed(orth, RpaMetric::Ii, orth_applies);
            let (obl, _) =
                crate::solvers::hess::readout_from_signed(obl, RpaMetric::Ii, obl_applies);
            assert!(
                flat_bits(&obl.omega2()) != flat_bits(&so.omega2())
                    || flat_bits(&obl.residuals()) != flat_bits(&so.residuals())
                    || obl.iterations != so.iterations,
                "su2 {precond:?} premise: the oblique replay is bitwise the entry — this fixture \
                 cannot tell the two hooks apart"
            );
            assert!(
                flat_bits(&orth.omega2()) == flat_bits(&so.omega2()),
                "su2 {precond:?}: the entry's omega2 is not the orthogonal-hook kernel's"
            );
            assert_eq!(
                orth.iterations, so.iterations,
                "su2 {precond:?}: iterations"
            );
            assert_eq!(orth.applies, so.applies, "su2 {precond:?}: applies");
            assert!(
                flat_bits(&orth.residuals()) == flat_bits(&so.residuals()),
                "su2 {precond:?}: residuals"
            );
            assert_eq!(orth_amps.len(), vecs.len(), "su2 {precond:?}: mode count");
            for (j, (p, q)) in orth_amps.iter().zip(&vecs).enumerate() {
                let same = |a: &Su2TtTangent<f64>, b: &Su2TtTangent<f64>| {
                    (0..a.num_coeffs()).all(|i| a.coeff(i).to_bits() == b.coeff(i).to_bits())
                };
                assert!(
                    same(&p.x, &q.x) && same(&p.y, &q.y) && q.x_im.is_none() == p.x_im.is_none(),
                    "su2 {precond:?}: mode {j}'s amplitudes differ from the orthogonal replay's"
                );
            }
            gaps.push(gap);
        }
        println!(
            "su2 oblique-vs-orthogonal omega2 gap: D0 {:.2e}, Off {:.2e}",
            gaps[0], gaps[1]
        );
        assert!(
            gaps[0] >= 1e-6,
            "su2 D0: the hook acts at first order on a preconditioned correction (gap {:e})",
            gaps[0]
        );
        assert!(
            gaps[1] <= 1e-9,
            "su2 Off: the hook acts at roundoff on an unpreconditioned solve (gap {:e})",
            gaps[1]
        );
    }
}
