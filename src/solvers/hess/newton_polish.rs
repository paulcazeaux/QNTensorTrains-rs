// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::tangent::TtTangent;
use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
use crate::scalar::{C64, Scalar};
use crate::solvers::InnerProductVector;
use crate::solvers::davidson::{DavidsonOpts, davidson};
use crate::solvers::riemannian::tangent_bases;
use crate::vector::TtVector;
use std::sync::Arc;

use super::WeingartenCtx;
use super::modes::{ModeOpts, ModeResult, PROBE_NEG_MARGIN, PROBE_TOL_REL, lowest_hessian_mode};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PcgFlag {
    Converged,
    MaxIter,
    NegativeCurvature { rho: f64 },
}

pub(crate) fn pcg_solve<V, Op, Pr>(
    mut op: Op,
    mut precond: Pr,
    b: &V,
    tol_rel: f64,
    max_iter: usize,
) -> (V, PcgFlag, usize)
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
    Pr: FnMut(&mut V),
{
    let bn = b.norm();
    let mut x = b.clone();
    x.scale(V::Scalar::from_f64(0.0));
    if bn == 0.0 {
        return (x, PcgFlag::Converged, 0);
    }
    let mut r = b.clone();
    let mut z = r.clone();
    precond(&mut z);
    let mut p = z.clone();
    let mut rz = r.dot(&z).re_s();
    for it in 0..max_iter {
        let ap = op(&p);
        let pap = p.dot(&ap).re_s();
        if pap <= 0.0 {
            let pn2 = p.norm().powi(2).max(1e-300);
            return (x, PcgFlag::NegativeCurvature { rho: pap / pn2 }, it + 1);
        }
        let alpha = rz / pap;
        x.axpy(V::Scalar::from_f64(alpha), &p);
        r.axpy(V::Scalar::from_f64(-alpha), &ap);
        if r.norm() <= tol_rel * bn {
            return (x, PcgFlag::Converged, it + 1);
        }
        let mut zn = r.clone();
        precond(&mut zn);
        let rzn = r.dot(&zn).re_s();
        let beta = rzn / rz;
        rz = rzn;
        let mut pn = zn;
        pn.axpy(V::Scalar::from_f64(beta), &p);
        p = pn;
    }
    (x, PcgFlag::MaxIter, max_iter)
}

/// Verdict of a positive-definiteness probe on a deflated Hessian part (H^rr or H^ii).
#[derive(Clone, Copy, Debug)]
pub enum PdVerdict {
    /// Proven indefinite: `lambda` = fresh-apply Rayleigh quotient of the witness (< 0). Valid
    /// whether or not the Davidson converged — a negative Rayleigh quotient is a variational
    /// upper bound on λ_min and needs no convergence.
    WitnessNegative { lambda: f64 },
    /// The Davidson CONVERGED and found no negative direction; `lambda_min` = fresh-apply
    /// Rayleigh quotient of the best Ritz vector, `residual` its Davidson residual norm.
    LikelyPd { lambda_min: f64, residual: f64 },
    /// The Davidson exhausted its budget with the residual still above tolerance and the best
    /// Ritz vector's fresh Rayleigh quotient non-negative. **Not** a positive-definiteness
    /// claim — only "no negative direction found in this many applies". Rerun with a larger
    /// budget, or certify densely.
    Inconclusive { lambda_min: f64, residual: f64 },
}

impl PdVerdict {
    /// `Some(true)` PD-certified, `Some(false)` proven indefinite, `None` inconclusive — the
    /// three-way read every caller should be doing.
    pub fn is_pd(&self) -> Option<bool> {
        match self {
            PdVerdict::WitnessNegative { .. } => Some(false),
            PdVerdict::LikelyPd { .. } => Some(true),
            PdVerdict::Inconclusive { .. } => None,
        }
    }
}

/// Which Hessian part a probe targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HessPart {
    /// `H^rr = A + B` — the real-layer energy Hessian (PD ⟺ the point is a local minimizer,
    /// after radial deflation).
    Rr,
    /// `H^ii = A − B` — curvature along the imaginary directions (PD is the RPA/Williamson
    /// requirement the REAL optimizer never probes; indefiniteness here is a physical
    /// instability diagnosis, not an optimization failure).
    Ii,
}

pub(crate) fn diag_inf_scale<V: InnerProductVector>(d: &V) -> f64 {
    let mut m = 0.0f64;
    d.for_each_coeff(|x| m = m.max(x.abs_s()));
    m.max(1.0)
}

pub(crate) fn pd_probe_kernel<V, Op, Pr, Cl>(
    op: Op,
    precond: Pr,
    clean: Cl,
    v0: &V,
    budget: usize,
    scale: f64,
) -> (PdVerdict, V, usize)
where
    V: InnerProductVector + Send + Sync,
    Op: Fn(&V) -> V + Sync,
    Pr: FnMut(&mut V, f64),
    Cl: FnMut(&mut V),
{
    pd_probe_kernel_with(
        op,
        precond,
        clean,
        std::slice::from_ref(v0),
        budget,
        scale,
        &ProbeDavidson::default(),
    )
}

/// Krylov tunables of the PD probe's Davidson. The default is a **4-root block on a 48-vector
/// subspace** ([`Self::default`]), NOT the [`crate::solvers::davidson::DavidsonOpts`] default: the
/// latter was tuned for the warm MALS local eigenproblem, and a COLD probe of a stiff Hessian
/// block (hcho untyped cap 256: `λ_min/scale ≈ 3e-4`) is a different regime — a thick restart to
/// one vector discards the Krylov history a slowly-separating bottom needs, and a small block can
/// CONVERGE on an interior pair and report `LikelyPd` off the wrong eigenvalue (measured
/// 4 of 8 seeds at blocks 1 and 2). The measured default and its arm table are on the `Default`
/// impl.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeDavidson {
    /// Subspace size at which a thick restart collapses the basis.
    pub maxsubspace: usize,
    /// Ritz vectors kept across a thick restart (`≥ block`).
    pub restart_keep: usize,
    /// Unpreconditioned (Lanczos-like) correction steps before `(D_A − θ)⁻¹` engages.
    pub nwarm: usize,
    /// Ritz pairs iterated together (`1` = the single-vector `solvers::davidson`; `> 1` =
    /// [`crate::solvers::davidson_block`], one apply per unconverged root per iteration, the
    /// verdict read off the LOWEST root).
    pub block: usize,
}

impl Default for ProbeDavidson {
    fn default() -> Self {
        ProbeDavidson {
            maxsubspace: 48,
            restart_keep: 12,
            nwarm: 2,
            block: 4,
        }
    }
}

pub(crate) fn pd_probe_kernel_with<V, Op, Pr, Cl>(
    op: Op,
    mut precond: Pr,
    mut clean: Cl,
    seeds: &[V],
    budget: usize,
    scale: f64,
    dav: &ProbeDavidson,
) -> (PdVerdict, V, usize)
where
    V: InnerProductVector + Send + Sync,
    Op: Fn(&V) -> V + Sync,
    Pr: FnMut(&mut V, f64),
    Cl: FnMut(&mut V),
{
    assert!(!seeds.is_empty(), "pd_probe_kernel_with: no seed");
    let tol = PROBE_TOL_REL * scale;
    let verbosity = u8::from(crate::knobs::PD_TRACE.get());
    let (mut u, converged, normres, numops) = if dav.block <= 1 {
        let opts = DavidsonOpts {
            tol,
            maxsubspace: dav.maxsubspace,
            maxiter: budget.max(2),
            nwarm: dav.nwarm,
            restart_keep: dav.restart_keep,
            verbosity,
            stop_below: Some(-PROBE_NEG_MARGIN * scale),
            ..Default::default()
        };
        let mut prec_clean = |t: &mut V, theta: f64| {
            precond(t, theta);
            clean(t);
        };
        let mut op_mut = &op;
        let (_theta, u, info) = davidson(&mut op_mut, &seeds[0], &mut prec_clean, &opts);
        (u, info.converged, info.normres, info.numops)
    } else {
        use crate::solvers::davidson_block::{BlockDavidsonOpts, davidson_block_projected_batched};
        use rayon::prelude::*;
        let b = dav.block;
        let opts = BlockDavidsonOpts {
            tol,
            maxsubspace: dav.maxsubspace.max(2 * b),
            maxiter: (budget / b).max(2),
            nwarm: dav.nwarm,
            theta_tol: None,
            restart_keep: dav.restart_keep.clamp(b, dav.maxsubspace.max(2 * b) - 1),
            verbosity,
        };
        let n = seeds.len().min(b);
        let nest_serial = std::env::var("QNTT_PROBE_NEST").is_ok_and(|v| v == "serial");
        let mut op_batch = |vs: &[V]| -> Vec<V> {
            let work = vs
                .len()
                .saturating_mul(vs.first().map_or(0, |v| v.num_coeffs()));
            if vs.len() < 2 || !crate::par::par_driver_fork(work) {
                return vs.iter().map(&op).collect();
            }
            vs.par_iter()
                .map(|v| {
                    if nest_serial {
                        crate::par::nested_scope(|| op(v))
                    } else {
                        op(v)
                    }
                })
                .collect()
        };
        let (_thetas, mut us, info) = davidson_block_projected_batched(
            &mut op_batch,
            &seeds[..n],
            &mut precond,
            &mut clean,
            b,
            &opts,
        );
        let u = us.swap_remove(0);
        let r0 = info.normres.first().copied().unwrap_or(f64::INFINITY);
        (u, r0 <= tol, r0, info.numops)
    };
    clean(&mut u);
    let un2 = u.norm().powi(2).max(1e-300);
    let hu = op(&u);
    let lambda = u.dot(&hu).re_s() / un2;
    let applies = numops + 1;
    if lambda < 0.0 {
        (PdVerdict::WitnessNegative { lambda }, u, applies)
    } else if converged {
        (
            PdVerdict::LikelyPd {
                lambda_min: lambda,
                residual: normres,
            },
            u,
            applies,
        )
    } else {
        (
            PdVerdict::Inconclusive {
                lambda_min: lambda,
                residual: normres,
            },
            u,
            applies,
        )
    }
}

macro_rules! impl_hess_probes {
    (
        ctx: $ctx:ident,
        tangent: $tangent:ident,
        vec: $vec:ident,
        tangent_dim: $tangent_dim:path,
        tt_randn: $tt_randn:path,
        tag: $tag:literal,
        $(#[$doc_hessian_pd_probe:meta])*
        hessian_pd_probe,
        $(#[$doc_hessian_pd_probe_witness:meta])*
        hessian_pd_probe_witness,
        $(#[$doc_hessian_pd_probe_witness_with:meta])*
        hessian_pd_probe_witness_with,
        $(#[$doc_hessian_pd_probe_c64:meta])*
        hessian_pd_probe_c64,
        $(#[$doc_hessian_pd_probe_c64_witness:meta])*
        hessian_pd_probe_c64_witness $(,)?
    ) => {
        pub(crate) fn radial_mode<T: Scalar>(ctx: &$ctx<'_, T>) -> Option<$tangent<T>> {
            let (base_l, base_r) = ctx.shared_bases();
            let mut r = $tangent::project_shared(base_l, base_r, base_l);
            let n = r.norm();
            if n > 0.0 {
                r.scale_mut(T::from_f64(1.0 / n));
                Some(r)
            } else {
                None
            }
        }

        pub(crate) fn deflate_radial<T: Scalar>(radial: Option<&$tangent<T>>, v: &mut $tangent<T>) {
            if let Some(r) = radial {
                for _ in 0..2 {
                    let c = $tangent::dot(r, v);
                    v.axpy(c * T::from_f64(-1.0), r);
                }
            }
        }

        pub(crate) fn reproject_tangent<T: Scalar>(
            base_l: &Arc<$vec<T>>,
            base_r: &Arc<$vec<T>>,
            radial: Option<&$tangent<T>>,
            v: &mut $tangent<T>,
        ) {
            *v = $tangent::project_shared(base_l, base_r, &v.to_vector());
            deflate_radial(radial, v);
        }

        pub(crate) fn clean_tangent<T: Scalar>(radial: Option<&$tangent<T>>, v: &mut $tangent<T>) {
            v.gauge_fix_orthogonal_mut();
            deflate_radial(radial, v);
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn lowest_hessian_mode_at<T, Ap>(
            base_l: &Arc<$vec<T>>,
            base_r: &Arc<$vec<T>>,
            radial: Option<&$tangent<T>>,
            diag: &$tangent<T>,
            apply: Ap,
            extra: &[$tangent<T>],
            v0: &$tangent<T>,
            opts: &ModeOpts,
        ) -> Option<ModeResult<$tangent<T>>>
        where
            T: Scalar,
            Ap: Fn(&$tangent<T>) -> $tangent<T>,
        {
            let op = |p: &$tangent<T>| {
                let mut q = p.clone();
                reproject_tangent(base_l, base_r, radial, &mut q);
                deflate_subspace(extra, &mut q);
                let mut hq = apply(&q);
                deflate_radial(radial, &mut hq);
                deflate_subspace(extra, &mut hq);
                hq
            };
            let precond =
                |t: &mut $tangent<T>, theta: f64| crate::solvers::davidson::precond(t, diag, theta);
            let clean = |t: &mut $tangent<T>| {
                reproject_tangent(base_l, base_r, radial, t);
                deflate_subspace(extra, t);
            };
            lowest_hessian_mode(op, precond, clean, v0, opts)
        }

        $(#[$doc_hessian_pd_probe])*
        pub fn hessian_pd_probe(
            ctx: &$ctx<'_, f64>,
            part: HessPart,
            v0: &$tangent<f64>,
            budget: usize,
        ) -> (PdVerdict, usize) {
            let (verdict, _witness, applies) = hessian_pd_probe_witness(ctx, part, v0, budget);
            (verdict, applies)
        }

        $(#[$doc_hessian_pd_probe_witness])*
        pub fn hessian_pd_probe_witness(
            ctx: &$ctx<'_, f64>,
            part: HessPart,
            v0: &$tangent<f64>,
            budget: usize,
        ) -> (PdVerdict, $tangent<f64>, usize) {
            hessian_pd_probe_witness_with(
                ctx,
                part,
                std::slice::from_ref(v0),
                budget,
                PROBE_BLOCK_SEED,
                &ProbeDavidson::default(),
            )
        }

        $(#[$doc_hessian_pd_probe_witness_with])*
        pub fn hessian_pd_probe_witness_with(
            ctx: &$ctx<'_, f64>,
            part: HessPart,
            seeds: &[$tangent<f64>],
            budget: usize,
            seed: u64,
            dav: &ProbeDavidson,
        ) -> (PdVerdict, $tangent<f64>, usize) {
            use rand::SeedableRng;
            assert!(
                !seeds.is_empty(),
                concat!($tag, "hessian_pd_probe_witness_with: at least one seed is required")
            );
            let (base_l, base_r) = ctx.shared_bases();
            let radial = radial_mode(ctx);
            let d = ctx.a_diagonal();
            ctx.warm_b_caches();
            let scale = diag_inf_scale(&d);
            let mu = complement_shift(scale);
            let op = |p: &$tangent<f64>| {
                let mut q = p.clone();
                clean_tangent(radial.as_ref(), &mut q);
                let mut hq = match part {
                    HessPart::Rr => ctx.hpp_apply(&q),
                    HessPart::Ii => ctx.hmm_apply(&q),
                };
                deflate_radial(radial.as_ref(), &mut hq);
                add_complement_shift(&mut hq, p, q, mu);
                hq
            };
            let precond =
                |t: &mut $tangent<f64>, theta: f64| crate::solvers::davidson::precond(t, &d, theta);
            let clean = |t: &mut $tangent<f64>| clean_tangent(radial.as_ref(), t);
            let mut block: Vec<$tangent<f64>> = seeds
                .iter()
                .map(|v| {
                    let mut t = v.clone();
                    reproject_tangent(base_l, base_r, radial.as_ref(), &mut t);
                    t
                })
                .collect();
            if block.len() < dav.block {
                let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
                while block.len() < dav.block {
                    let r = $tt_randn(base_l.space, &base_l.ranks, &mut rng);
                    block.push($tangent::project_shared(base_l, base_r, &r));
                }
            }
            pd_probe_kernel_with(op, precond, clean, &block, budget, scale, dav)
        }

        $(#[$doc_hessian_pd_probe_c64])*
        pub fn hessian_pd_probe_c64(
            ctx: &$ctx<'_, C64>,
            v0: &$tangent<C64>,
            budget: usize,
        ) -> (PdVerdict, usize) {
            let (verdict, _witness, applies) = hessian_pd_probe_c64_witness(ctx, v0, budget);
            (verdict, applies)
        }

        $(#[$doc_hessian_pd_probe_c64_witness])*
        pub fn hessian_pd_probe_c64_witness(
            ctx: &$ctx<'_, C64>,
            v0: &$tangent<C64>,
            budget: usize,
        ) -> (PdVerdict, $tangent<C64>, usize) {
            let (base_l, base_r) = ctx.shared_bases();
            let radial = radial_mode(ctx);
            let d = ctx.a_diagonal();
            let space_dim =
                2 * $tangent_dim(base_l) - radial.as_ref().map_or(0, |_| 2);
            let opts = ModeOpts {
                budget,
                space_dim,
                scale: diag_inf_scale(&d),
                ..Default::default()
            };
            let res = lowest_hessian_mode_at(
                base_l,
                base_r,
                radial.as_ref(),
                &d,
                |q| ctx.hess_apply(q),
                &[],
                v0,
                &opts,
            )
            .expect(concat!(
                $tag,
                "hessian_pd_probe_c64_witness: the seed collapsed under deflation — caller bug"
            ));
            (res.verdict, res.witness, res.applies)
        }
    };
}

pub(crate) use impl_hess_probes;

impl_hess_probes!(
    ctx: WeingartenCtx,
    tangent: TtTangent,
    vec: TtVector,
    tangent_dim: crate::core::tangent::tangent_dim,
    tt_randn: crate::vector::tt_randn,
    tag: "",
    hessian_pd_probe,
    hessian_pd_probe_witness,
    hessian_pd_probe_witness_with,
    hessian_pd_probe_c64,
    hessian_pd_probe_c64_witness,
);

/// Default seed of the extra block directions [`hessian_pd_probe_witness`] needs to fill
/// [`ProbeDavidson::default`]'s block. A FIXED constant, so the legacy single-`v0` entry stays a
/// pure function of `(ctx, part, v0, budget)` — reproducible run to run and across machines.
/// Callers who want independent draws per probe (the tracker, `certify_probe`) go through
/// [`hessian_pd_probe_witness_with`] and pass their own.
pub const PROBE_BLOCK_SEED: u64 = 0xC3;

/// Options for [`newton_polish`] (and the SU(2) twin).
#[derive(Clone, Copy, Debug)]
pub struct NewtonPolishOpts {
    /// Stop when `‖P(Hx − λx)‖ ≤ gradtol` (the RPA criticality gate is 1e-9; default one order
    /// below it).
    pub gradtol: f64,
    /// Outer Newton iterations.
    pub max_outer: usize,
    /// Inner PCG iterations per outer step (each = one `hess_apply` = 1 H-matvec + 1 Weingarten
    /// assembly).
    pub max_inner: usize,
    /// Inexact-Newton forcing: inner relative tolerance `min(eta, √(‖grad‖/‖grad₀‖))` — loose
    /// far out, tightening toward superlinear near the solution.
    pub eta: f64,
    /// Use the `|D_A| + ridge` tangent-diagonal preconditioner in the inner CG.
    pub precond: bool,
    /// Hard cap on TOTAL `hess_apply` calls across the whole polish: without it an indefinite
    /// point can ladder through many full inner solves.
    /// μ-escalation retries count against it.
    pub max_applies: usize,
    /// On reaching `gradtol`, run the H^rr PD probe; a negative witness marks the point a
    /// SADDLE and the polish descends along the witness direction and continues, exiting only
    /// on a positive probe (or budget). `false` = old behavior (criticality only).
    pub certify: bool,
    /// Davidson budget of each PD probe.
    pub probe_budget: usize,
    /// Maximum saddle escapes before giving up (uncertified exit with the witness λ reported).
    pub max_escapes: usize,
}

impl Default for NewtonPolishOpts {
    fn default() -> Self {
        NewtonPolishOpts {
            gradtol: 1e-10,
            max_outer: 15,
            max_inner: 200,
            eta: 0.1,
            precond: true,
            max_applies: 2000,
            certify: true,
            probe_budget: 40,
            max_escapes: 3,
        }
    }
}

/// Diagnostics of the Newton polish.
#[derive(Clone, Debug)]
pub struct NewtonPolishInfo {
    /// `‖grad‖ ≤ gradtol` reached.
    pub converged: bool,
    /// Outer Newton steps taken.
    pub outer: usize,
    /// Total `hess_apply` calls (inner CG iterations).
    pub applies: usize,
    /// Final gradient norm.
    pub gnorm: f64,
    /// Final Rayleigh quotient.
    pub lambda: f64,
    /// `gradtol` was reached AND the H^rr PD probe found no negative direction — the exit
    /// point is a certified (probe-grade) local minimizer, the RPA precondition. Always
    /// `false` when `opts.certify` is off or budgets ran out first.
    pub minimizer_certified: bool,
    /// Fresh-apply Rayleigh quotient from the final H^rr probe (negative = the saddle witness
    /// the polish could not escape within `max_escapes`).
    pub hrr_min: Option<f64>,
    /// Saddle escapes taken.
    pub escapes: usize,
}

macro_rules! impl_newton_polish {
    (
        $(#[$doc:meta])*
        $name:ident, $ham:ty, $vec:ty, $tangent:ty,
        $ctx:ident, $rayleigh:path, $tbases:path, $normalize:path, $linesearch:path
    ) => {
        $(#[$doc])*
        pub fn $name(
            h: &$ham,
            x0: &$vec,
            opts: &NewtonPolishOpts,
        ) -> ($vec, NewtonPolishInfo) {
            let mut x = x0.clone();
            $normalize(&mut x);
            let mut mu = 0.0f64;
            let mut applies = 0usize;
            let mut outer = 0usize;
            let mut gnorm0: Option<f64> = None;
            let mut minimizer_certified = false;
            let mut hrr_min: Option<f64> = None;
            let mut escapes = 0usize;
            let (mut gnorm, mut lambda);
            loop {
                let (base_l, base_r) = $tbases(&x);
                let (base_l, base_r) = (std::sync::Arc::new(base_l), std::sync::Arc::new(base_r));
                lambda = $rayleigh(h, &x).re_s();
                let ctx = $ctx::from_parts(h, base_l.clone(), base_r.clone(), lambda);
                gnorm = ctx.gradient().norm();
                let g0 = *gnorm0.get_or_insert(gnorm);
                let grad = ctx.gradient().clone();
                let radial = {
                    let mut r = <$tangent>::project_shared(&base_l, &base_r, &base_l);
                    let n = r.norm();
                    (n > 0.0).then(|| {
                        r.scale_mut(<$vec as crate::solvers::hess::VecScalar>::S::from_f64(1.0 / n));
                        r
                    })
                };
                if gnorm <= opts.gradtol {
                    if !opts.certify || applies >= opts.max_applies {
                        break;
                    }
                    let dprobe = ctx.a_diagonal();
                    let probe_scale = crate::solvers::hess::diag_inf_scale(&dprobe);
                    let (verdict, witness, pa) = crate::solvers::hess::pd_probe_kernel(
                        |p: &$tangent| {
                            let mut q = p.clone();
                            clean_tangent(radial.as_ref(), &mut q);
                            let mut hq = ctx.hpp_apply(&q);
                            deflate_radial(radial.as_ref(), &mut hq);
                            crate::solvers::hess::add_complement_shift(
                                &mut hq,
                                p,
                                q,
                                crate::solvers::hess::complement_shift(probe_scale),
                            );
                            hq
                        },
                        |t: &mut $tangent, theta: f64| {
                            crate::solvers::davidson::precond(t, &dprobe, theta)
                        },
                        |t: &mut $tangent| {
                            clean_tangent(radial.as_ref(), t);
                        },
                        &{
                            let mut seed = dprobe.clone();
                            seed = <$tangent>::project_shared(&base_l, &base_r, &seed.to_vector());
                            seed
                        },
                        opts.probe_budget,
                        probe_scale,
                    );
                    applies += pa;
                    match verdict {
                        crate::solvers::hess::PdVerdict::LikelyPd { lambda_min, .. } => {
                            hrr_min = Some(lambda_min);
                            minimizer_certified = true;
                            break;
                        }
                        crate::solvers::hess::PdVerdict::Inconclusive { lambda_min, .. } => {
                            hrr_min = Some(lambda_min);
                            break;
                        }
                        crate::solvers::hess::PdVerdict::WitnessNegative { lambda: lneg } => {
                            hrr_min = Some(lneg);
                            escapes += 1;
                            if escapes > opts.max_escapes || applies >= opts.max_applies {
                                break;
                            }
                            let slope = grad.dot(&witness).re_s().min(0.0);
                            let stepped = match $linesearch(
                                h, &x, &witness, lambda, slope, 1.0, 0.0, None,
                            )
                            {
                                Some((_a, xn, _l)) => {
                                    x = xn;
                                    true
                                }
                                None => {
                                    let wneg = witness.scale(
                                        <$vec as crate::solvers::hess::VecScalar>::S::from_f64(
                                            -1.0,
                                        ),
                                    );
                                    match $linesearch(h, &x, &wneg, lambda, slope, 1.0, 0.0, None) {
                                        Some((_a, xn, _l)) => {
                                            x = xn;
                                            true
                                        }
                                        None => false,
                                    }
                                }
                            };
                            if !stepped {
                                break;
                            }
                            $normalize(&mut x);
                            mu = 0.0;
                            continue;
                        }
                    }
                }
                if outer >= opts.max_outer || applies >= opts.max_applies {
                    break;
                }
                let dinv = opts.precond.then(|| {
                    let d = ctx.a_diagonal();
                    let mut maxd = 0.0f64;
                    crate::solvers::InnerProductVector::for_each_coeff(&d, |c| {
                        maxd = maxd.max(c.abs())
                    });
                    (d, 1e-2 * maxd.max(1e-300))
                });
                let mut b = grad.scale(<$vec as crate::solvers::hess::VecScalar>::S::from_f64(-1.0));
                deflate_radial(radial.as_ref(), &mut b);
                let inner_tol = opts.eta.min((gnorm / g0).sqrt()).max(1e-4);
                let delta = loop {
                    let (delta, flag, iters) = crate::solvers::hess::pcg_solve(
                        |p: &$tangent| {
                            let mut q = p.clone();
                            reproject_tangent(&base_l, &base_r, radial.as_ref(), &mut q);
                            let mut hq = ctx.hess_apply(&q);
                            if mu > 0.0 {
                                hq.axpy(<$vec as crate::solvers::hess::VecScalar>::S::from_f64(mu), &q);
                            }
                            deflate_radial(radial.as_ref(), &mut hq);
                            hq
                        },
                        |t: &mut $tangent| {
                            if let Some((d, ridge)) = dinv.as_ref() {
                                t.zip_coeffs_mut(d, |ti, di| *ti /= di.abs() + ridge);
                            }
                        },
                        &b,
                        inner_tol,
                        opts.max_inner,
                    );
                    applies += iters;
                    if let crate::solvers::hess::PcgFlag::NegativeCurvature { rho } = flag {
                        mu = (10.0 * mu).max(1.5 * rho.abs()).max(1e-10);
                        if mu < 1e6 && applies < opts.max_applies {
                            continue;
                        }
                    }
                    break delta;
                };
                let slope = grad.dot(&delta).re_s();
                let (dir, slope) = if slope < 0.0 {
                    (delta, slope)
                } else {
                    (
                        grad.scale(<$vec as crate::solvers::hess::VecScalar>::S::from_f64(-1.0)),
                        -gnorm * gnorm,
                    )
                };
                match $linesearch(h, &x, &dir, lambda, slope, 1.0, gnorm, Some(slope)) {
                    Some((_a, xn, _l)) => {
                        x = xn;
                        $normalize(&mut x);
                    }
                    None => break,
                }
                mu = (mu / 3.0).max(0.0);
                if mu < 1e-14 {
                    mu = 0.0;
                }
                outer += 1;
            }
            let converged = gnorm <= opts.gradtol;
            (
                x,
                NewtonPolishInfo {
                    converged,
                    outer,
                    applies,
                    gnorm,
                    lambda,
                    minimizer_certified,
                    hrr_min,
                    escapes,
                },
            )
        }
    };
}

pub(crate) trait VecScalar {
    type S: Scalar;
}
impl<T: Scalar> VecScalar for TtVector<T> {
    type S = T;
}
impl<T: Scalar> VecScalar for crate::su2::vector::Su2TtVector<T> {
    type S = T;
}

pub(crate) use impl_newton_polish;

impl_newton_polish!(
    newton_polish,
    SparseHamiltonian<f64>,
    TtVector<f64>,
    TtTangent<f64>,
    WeingartenCtx,
    rayleigh_quotient,
    tangent_bases,
    crate::solvers::riemannian::normalize_mut,
    crate::solvers::riemannian::linesearch
);

pub(crate) fn deflate_subspace<V: InnerProductVector>(modes: &[V], v: &mut V) {
    for _ in 0..2 {
        for q in modes {
            let c = q.dot(v);
            v.axpy(c * <V::Scalar as Scalar>::from_f64(-1.0), q);
        }
    }
}

pub(crate) fn complement_shift(scale: f64) -> f64 {
    if scale.is_finite() {
        10.0 * (1.0 + scale.min(1e100))
    } else {
        0.0
    }
}

pub(crate) fn add_complement_shift<V: InnerProductVector>(image: &mut V, v: &V, mut q: V, mu: f64) {
    if mu == 0.0 {
        return;
    }
    q.axpy(<V::Scalar as Scalar>::from_f64(-1.0), v);
    image.axpy(<V::Scalar as Scalar>::from_f64(-mu), &q);
}
