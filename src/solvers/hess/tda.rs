// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::tangent::{TtTangent, tangent_dim};
use crate::solvers::InnerProductVector;
use crate::solvers::davidson::DAVIDSON_NWARM;
use crate::solvers::davidson_block::{BlockDavidsonOpts, davidson_block_projected_batched};

use super::{
    NegativeWitness, SympPrecond, WeingartenCtx, add_complement_shift, certify_negatives,
    clean_tangent, complement_shift, deflate_radial, diag_inf_scale, radial_mode,
    sector_d0_precond,
};

/// The TDA readout's preconditioner.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum TdaPrecond {
    /// Resolves to [`Self::Diag`] at `kappa = 10` ([`Self::resolve`]): the measured arm
    /// that converges wherever `Off` does, to the same roots, with about half its `A` applies and
    /// half its wall at the molecular references. The auto cap `1e3` is NOT the default: it
    /// stalls at an indefinite base point under thick restarts where `Off` and `kappa = 10`
    /// converge.
    #[default]
    Auto,
    /// Plain residual corrections.
    Off,
    /// D0: the fixed, θ-INDEPENDENT inverse of the clamped `D_A`
    /// ([`crate::solvers::davidson::precond_fixed_inverse`]; `kappa` caps its condition number, a
    /// non-positive or non-finite value meaning the auto `1e3`), behind the sector layer's overflow
    /// guard: a model `≤ 0` everywhere has no informative inverse, and the arm then runs as `Off`
    /// ([`TdaInfo::precond`] says so).
    Diag { kappa: f64 },
    /// The classical θ-shifted Davidson correction `(D_A − θᵢ)⁻¹` at each root's own Ritz value
    /// ([`crate::solvers::davidson::precond`], with its denominator safeguard).
    Shifted,
}

impl TdaPrecond {
    /// `Auto` → `Diag { kappa: 10.0 }`; every explicit arm passes through unchanged.
    pub fn resolve(self) -> TdaPrecond {
        match self {
            TdaPrecond::Auto => TdaPrecond::Diag { kappa: 10.0 },
            p => p,
        }
    }
}

/// Tunables of [`tda_lowest_davidson`]. `k` is a parameter of the entry point, not a field.
#[derive(Clone, Copy, Debug)]
pub struct TdaDavidsonOpts {
    /// ABSOLUTE residual tolerance per root, in hartree: a root is converged when its projected
    /// residual `‖Π(Aξ − θξ)‖ ≤ tol` — the only residual that can reach zero on the deflated
    /// space. A converged root's eigenvalue error is at most `tol` (the symmetric Bauer–Fike bound)
    /// and is typically `tol²/gap`.
    pub tol: f64,
    /// Outer iterations: one Rayleigh–Ritz and at most one apply per unconverged root each.
    pub max_iter: usize,
    /// Thick-restart bound on the live basis. `0` = the auto `max(8·(nroots + 2), 32)`; any value
    /// is raised to `2·nroots` (a restart keeps the roots and must still admit one correction
    /// each). It is NOT capped at the space dimension: a basis that fills the space is exact
    /// exhaustion, and capping there would turn exhaustion into restarts.
    pub max_subspace: usize,
    /// Ritz columns kept across a restart: `0` = `2·nroots`; any value is clamped into
    /// `[nroots, max_subspace − 1]`.
    pub restart_keep: usize,
    /// Unpreconditioned warm-up iterations before [`TdaPrecond::Diag`] / [`TdaPrecond::Shifted`]
    /// engage.
    pub nwarm: usize,
    pub precond: TdaPrecond,
    /// Most images computed at once: `0` = a whole batch (at most one per unconverged root), `1` =
    /// one at a time. Each live image is a whole tangent, so this bounds the solve's transient RSS.
    /// The values do not depend on it (the fork is bit-identical to the serial map).
    pub max_concurrent: usize,
    /// `≥ 1` prints the block Davidson's per-iteration line (θ and ‖r‖ of every root, stdout).
    pub verbosity: u8,
}

impl Default for TdaDavidsonOpts {
    fn default() -> Self {
        TdaDavidsonOpts {
            tol: 1e-8,
            max_iter: 300,
            max_subspace: 0,
            restart_keep: 0,
            nwarm: DAVIDSON_NWARM,
            precond: TdaPrecond::Auto,
            max_concurrent: 0,
            verbosity: 0,
        }
    }
}

/// Diagnostics of a TDA solve — the counts, the fresh residuals and the certificates a reader
/// needs before quoting a `θ`.
#[derive(Clone, Debug)]
pub struct TdaInfo {
    /// Every requested root came back with its projected residual `≤ tol` at the kernel's last
    /// Rayleigh–Ritz. `false` on an exhausted budget, a stagnated (short) solve, or an empty start.
    pub converged: bool,
    /// Roots converged at the last Rayleigh–Ritz.
    pub nconv: usize,
    /// Outer iterations performed.
    pub iterations: usize,
    /// `A` applies paid, ALL of them: the solve's, one fresh apply per returned vector, and one per
    /// certified negative. There is no `B` apply anywhere on this path.
    pub applies: usize,
    /// FRESH residuals `‖Π A ξⱼ − θⱼ ξⱼ‖` of the returned pairs, from a new apply on each
    /// reprojected unit vector (not the cached images the kernel tested).
    pub residuals: Vec<f64>,
    /// Fresh Rayleigh quotients `⟨ξⱼ, A ξⱼ⟩` of the returned unit vectors: `E_ξ − λ` exactly
    /// (module header), and within `O(‖r‖²/gap)` of `θⱼ`.
    pub rayleigh: Vec<f64>,
    /// The kernel's own projected residuals at its last Rayleigh–Ritz, in the same order.
    pub kernel_residuals: Vec<f64>,
    /// One entry per returned `θ < 0` (beyond `−1e-8·max|θ|`), with the fresh-apply variational
    /// certificate at metric `W = id` (`fresh = ⟨ξ, Aξ⟩/‖ξ‖²`, `fresh_w = 1` up to
    /// roundoff): a certified entry is an explicit state BELOW `λ` in the tangent span.
    pub negative: Vec<NegativeWitness>,
    /// Seeds dropped before or at seeding: annihilated by the projector (e.g. a radial seed),
    /// zero, or dependent on an earlier seed.
    pub dropped_seeds: usize,
    /// Dimension of the space solved on: `tangent_dim − 1` (the radial mode deflated).
    pub space_dim: usize,
    /// The preconditioner that RAN: `opts.precond` resolved, and [`TdaPrecond::Off`] where the D0
    /// guard skipped a collapsed model.
    pub precond: TdaPrecond,
}

/// The `k` lowest TDA pairs: `theta` ascending (hartree, relative to `λ` — an energy, never
/// `ω²`), `vectors` the matching reprojected, radial-free, unit Ritz vectors (orthonormal in the
/// layer's tangent metric to the solve's accuracy), and the solve's [`TdaInfo`].
#[derive(Clone, Debug)]
pub struct TdaModes<V> {
    pub theta: Vec<f64>,
    pub vectors: Vec<V>,
    pub info: TdaInfo,
}

impl<V> TdaModes<V> {
    /// Every negative `θ` carries a certified fresh-apply witness — the publication gate
    /// applied before a negative entry is read as a lower state rather than a solver artifact.
    pub fn all_negatives_certified(&self) -> bool {
        self.info.negative.iter().all(NegativeWitness::certified)
    }
}

pub(crate) struct TdaHooks<A, Rp, Df, Dg> {
    pub(crate) a: A,
    pub(crate) reproject: Rp,
    pub(crate) deflate: Df,
    pub(crate) diag: Dg,
    pub(crate) complement_shift: f64,
}

fn resolve_sizes(nroots: usize, opts: &TdaDavidsonOpts) -> (usize, usize) {
    let requested = if opts.max_subspace == 0 {
        (8 * (nroots + 2)).max(32)
    } else {
        opts.max_subspace
    };
    let max_subspace = requested.max(2 * nroots);
    let keep = if opts.restart_keep == 0 {
        2 * nroots
    } else {
        opts.restart_keep
    };
    (max_subspace, keep.clamp(nroots, max_subspace - 1))
}

fn batch_apply<V, Op>(op: &Op, vs: &[V], chunk: usize) -> Vec<V>
where
    V: InnerProductVector + Send + Sync,
    Op: Fn(&V) -> V + Sync,
{
    use rayon::prelude::*;
    let work = vs
        .len()
        .saturating_mul(vs.first().map_or(0, |v| v.num_coeffs()));
    if vs.len() < 2 || chunk < 2 || !crate::par::par_driver_fork(work) {
        return vs.iter().map(op).collect();
    }
    let mut out = Vec::with_capacity(vs.len());
    for c in vs.chunks(chunk) {
        out.extend(c.par_iter().map(op).collect::<Vec<V>>());
    }
    out
}

pub(crate) fn tda_lowest_core<V, A, Rp, Df, Dg>(
    hooks: TdaHooks<A, Rp, Df, Dg>,
    v0s: &[V],
    k: usize,
    space_dim: usize,
    opts: &TdaDavidsonOpts,
) -> TdaModes<V>
where
    V: InnerProductVector<Scalar = f64> + Send + Sync,
    A: Fn(&V) -> V + Sync,
    Rp: Fn(&mut V) + Sync,
    Df: Fn(&mut V) + Sync,
    Dg: FnOnce() -> V,
{
    assert!(
        opts.tol.is_finite() && opts.tol >= 0.0,
        "tda_lowest_davidson: tol = {} must be finite and ≥ 0 (an absolute residual in hartree)",
        opts.tol
    );
    let TdaHooks {
        a,
        reproject,
        deflate,
        diag,
        complement_shift: mu,
    } = hooks;
    let requested = opts.precond.resolve();
    let nroots = k.min(space_dim);
    let empty = |converged: bool, dropped_seeds: usize| TdaModes {
        theta: Vec::new(),
        vectors: Vec::new(),
        info: TdaInfo {
            converged,
            nconv: 0,
            iterations: 0,
            applies: 0,
            residuals: Vec::new(),
            rayleigh: Vec::new(),
            kernel_residuals: Vec::new(),
            negative: Vec::new(),
            dropped_seeds,
            space_dim,
            precond: requested,
        },
    };
    if nroots == 0 {
        return empty(true, 0);
    }
    assert!(
        !v0s.is_empty(),
        "tda_lowest_davidson: at least one seed is required (k = {k}, space_dim = {space_dim}) — \
         pass e.g. k + 2 projected random tangents"
    );

    let mut dropped = 0usize;
    let seeds: Vec<V> = v0s
        .iter()
        .filter_map(|g| {
            let n0 = g.norm();
            assert!(
                n0.is_finite(),
                "tda_lowest_davidson: non-finite seed norm (caller bug)"
            );
            let mut t = g.clone();
            reproject(&mut t);
            if t.norm() > f64::EPSILON.sqrt() * n0 {
                Some(t)
            } else {
                dropped += 1;
                None
            }
        })
        .collect();
    if seeds.is_empty() {
        return empty(false, dropped);
    }

    let (max_subspace, restart_keep) = resolve_sizes(nroots, opts);
    let dopts = BlockDavidsonOpts {
        tol: opts.tol,
        maxsubspace: max_subspace,
        maxiter: opts.max_iter.max(1),
        nwarm: opts.nwarm,
        theta_tol: None,
        restart_keep,
        verbosity: opts.verbosity,
    };

    let mut diag = Some(diag);
    let mut fixed = match requested {
        TdaPrecond::Diag { kappa } => sector_d0_precond(
            SympPrecond::DiagUnshifted { kappa },
            diag.take().expect("the diagonal model is read once"),
        ),
        _ => None,
    };
    let shifted: Option<V> = match requested {
        TdaPrecond::Shifted => Some(diag.take().expect("the diagonal model is read once")()),
        _ => None,
    };
    let ran = match requested {
        TdaPrecond::Diag { .. } if fixed.is_none() => TdaPrecond::Off,
        p => p,
    };
    let mut applyprec = |t: &mut V, theta: f64| {
        if let Some(p) = fixed.as_mut() {
            p(t, theta);
        }
        if let Some(d) = shifted.as_ref() {
            crate::solvers::davidson::precond(t, d, theta);
        }
    };

    let op = |v: &V| -> V {
        let mut q = v.clone();
        reproject(&mut q);
        let mut aq = a(&q);
        deflate(&mut aq);
        add_complement_shift(&mut aq, v, q, mu);
        aq
    };
    let chunk = if opts.max_concurrent == 0 {
        usize::MAX
    } else {
        opts.max_concurrent
    };
    let mut applies = 0usize;
    let mut op_batch = |vs: &[V]| -> Vec<V> {
        applies += vs.len();
        batch_apply(&op, vs, chunk)
    };
    let mut project = |t: &mut V| reproject(t);
    let (theta, mut vectors, kinfo) = davidson_block_projected_batched(
        &mut op_batch,
        &seeds,
        &mut applyprec,
        &mut project,
        nroots,
        &dopts,
    );

    for v in vectors.iter_mut() {
        reproject(v);
        let n = v.norm();
        if n > 0.0 {
            v.scale(1.0 / n);
        }
    }
    let images = batch_apply(&op, &vectors, chunk);
    applies += vectors.len();
    let mut residuals = Vec::with_capacity(vectors.len());
    let mut rayleigh = Vec::with_capacity(vectors.len());
    for ((v, av), &th) in vectors.iter().zip(&images).zip(&theta) {
        rayleigh.push(v.dot(av));
        let mut r = av.clone();
        r.axpy(-th, v);
        residuals.push(r.norm());
    }
    drop(images);

    let (negative, _pair_applies) = certify_negatives(
        &theta,
        theta.len(),
        |j| vectors.get(j).cloned(),
        &mut |v: &V| {
            applies += 1;
            op(v)
        },
        &mut |v: &V| v.clone(),
        &mut |_v: &mut V| {},
        &mut |v: &mut V| reproject(v),
    );

    TdaModes {
        theta,
        vectors,
        info: TdaInfo {
            converged: kinfo.converged,
            nconv: kinfo.nconv,
            iterations: kinfo.niter,
            applies,
            residuals,
            rayleigh,
            kernel_residuals: kinfo.normres,
            negative,
            dropped_seeds: dropped + kinfo.dropped_guesses,
            space_dim,
            precond: ran,
        },
    }
}

/// The `k` lowest eigenpairs of the Hessian's effective-Hamiltonian term `A = P_T(H − λ)P_T` at
/// `ctx`'s base point, on the radial-deflated tangent space — the in-sector Tamm–Dancoff spectrum.
///
/// # Panics
///
/// On an empty `v0s` (for `k ≥ 1` on a non-empty space), on a SHIFTED seed (`φ ≠ IDENTITY` — the
/// cross-sector TDA is `SectorCtx::sector_lowest_tda`), and on a negative or non-finite `tol`.
pub fn tda_lowest_davidson(
    ctx: &WeingartenCtx<'_, f64>,
    v0s: &[TtTangent<f64>],
    k: usize,
    opts: &TdaDavidsonOpts,
) -> TdaModes<TtTangent<f64>> {
    for (i, v) in v0s.iter().enumerate() {
        v.refuse_if_shifted(&format!("tda_lowest_davidson (seed {i})"));
    }
    let (base_l, _) = ctx.bases();
    let radial = radial_mode(ctx);
    let space_dim = tangent_dim(base_l) - radial.as_ref().map_or(0, |_| 1);
    ctx.warm_a_caches();
    let radial = radial.as_ref();
    let d_a = ctx.a_diagonal();
    let mu = complement_shift(diag_inf_scale(&d_a));
    tda_lowest_core(
        TdaHooks {
            a: |v: &TtTangent<f64>| ctx.a_apply(v),
            reproject: |v: &mut TtTangent<f64>| clean_tangent(radial, v),
            deflate: |v: &mut TtTangent<f64>| deflate_radial(radial, v),
            diag: move || d_a,
            complement_shift: mu,
        },
        v0s,
        k,
        space_dim,
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TtVector;
    use crate::core::factor::dense_self_adjoint_smallest;
    use crate::core::tangent::tangent_basis;
    use crate::core::weingarten::FloorPolicy;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::round::RoundMode;
    use crate::solvers::hess::{CurvatureReg, hessian_grams, weingarten_operator};
    use crate::solvers::hess::{add_complement_shift, clean_tangent, complement_shift};
    use crate::test_util::{generic_truncated_h, randomized_state};
    use crate::vector::tt_randn;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn c7_point() -> (SparseHamiltonian<f64>, TtVector<f64>) {
        let space = QnSpace::new(4, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        (generic_truncated_h::<f64>(space), x)
    }

    fn seeds(ctx: &WeingartenCtx<'_, f64>, n: usize, seed: u64) -> Vec<TtTangent<f64>> {
        let (bl, br) = ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .map(|_| TtTangent::project(bl, br, &tt_randn(bl.space, &bl.ranks, &mut rng)))
            .collect()
    }

    fn dense_a(ctx: &WeingartenCtx<'_, f64>) -> Vec<f64> {
        let (bl, br) = ctx.bases();
        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let (a, _b) = hessian_grams(ctx, &basis);
        dense_self_adjoint_smallest(&a, n, n).0
    }

    fn leak(ctx: &WeingartenCtx<'_, f64>, vs: &[TtTangent<f64>]) -> (f64, f64) {
        let (bl, br) = ctx.bases();
        let radial = radial_mode(ctx).expect("non-degenerate point");
        let (mut rad, mut tan) = (0.0f64, 0.0f64);
        for v in vs {
            rad = rad.max(radial.dot(v).abs());
            let mut p = TtTangent::project(bl, br, &v.to_vector());
            p.axpy(-1.0, v);
            tan = tan.max(p.norm());
        }
        (rad, tan)
    }

    #[test]
    fn tda_u1_matches_the_dense_a_gram_at_an_indefinite_point() {
        let (h, x) = c7_point();
        let ctx = weingarten_operator(&h, &x);
        let dense = dense_a(&ctx);
        let n = dense.len();
        assert_eq!(
            n,
            tangent_dim(ctx.bases().0) - 1,
            "the dense basis spans the deflated space"
        );
        println!("U1: n = {n}, dense eig(A) = {dense:?}");
        assert!(
            dense[0] < -1e-3,
            "fixture precondition: A is indefinite at this non-critical point (λ_min {})",
            dense[0]
        );
        let k = 4;
        let tol = 1e-11;
        let n_neg = dense[..k].iter().filter(|&&t| t < 0.0).count();
        for precond in [
            TdaPrecond::Off,
            TdaPrecond::Diag { kappa: 10.0 },
            TdaPrecond::Shifted,
        ] {
            let opts = TdaDavidsonOpts {
                tol,
                max_iter: 500,
                max_subspace: 2 * k,
                restart_keep: k,
                precond,
                ..TdaDavidsonOpts::default()
            };
            let out = tda_lowest_davidson(&ctx, &seeds(&ctx, k + 2, 0x7DA1), k, &opts);
            println!(
                "U1 {precond:?}: converged {} it {} applies {} θ {:?} res {:?}",
                out.info.converged,
                out.info.iterations,
                out.info.applies,
                out.theta,
                out.info.residuals
            );
            assert!(out.info.converged, "{precond:?}: not converged");
            assert_eq!(out.info.precond, precond, "the arm that ran");
            assert_eq!(out.info.space_dim, n);
            assert_eq!(out.theta.len(), k);
            let kernel_applies = out.info.applies - k - out.info.negative.len();
            assert!(
                kernel_applies > 2 * k,
                "{precond:?}: {kernel_applies} kernel applies never filled the 2k = {} subspace",
                2 * k
            );
            for (j, want) in dense.iter().enumerate().take(k) {
                let th = out.theta[j];
                assert!(
                    (th - want).abs() <= 1e-9,
                    "{precond:?} root {j}: θ {th} vs dense {want}"
                );
                assert!(
                    (out.info.rayleigh[j] - th).abs() <= 1e-9,
                    "{precond:?} root {j}: fresh Rayleigh {} vs θ {th}",
                    out.info.rayleigh[j]
                );
                assert!(
                    out.info.residuals[j] <= 10.0 * tol,
                    "{precond:?} root {j}: fresh residual {}",
                    out.info.residuals[j]
                );
            }
            assert_eq!(out.info.negative.len(), n_neg, "{precond:?}: witnesses");
            assert!(
                out.all_negatives_certified(),
                "{precond:?}: an uncertified negative"
            );
            for w in &out.info.negative {
                assert!(
                    (w.fresh - w.theta).abs() <= 1e-9 && (w.fresh_w - 1.0).abs() <= 1e-12,
                    "{precond:?}: witness {w:?}"
                );
            }
            let (rad, tan) = leak(&ctx, &out.vectors);
            assert!(
                rad <= 1e-10 && tan <= 1e-10,
                "{precond:?}: radial weight {rad:e}, tangency defect {tan:e}"
            );
            for i in 0..k {
                for j in 0..k {
                    let g = out.vectors[i].dot(&out.vectors[j]);
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((g - want).abs() <= 1e-9, "{precond:?}: ⟨ξ{i}, ξ{j}⟩ = {g}");
                }
            }
        }
    }

    #[test]
    fn tda_u2_anti_oracle_without_radial_deflation_trips_the_leak_floor() {
        let (h, x) = c7_point();
        let ctx = weingarten_operator(&h, &x);
        let dense = dense_a(&ctx);
        let (bl, _) = ctx.bases();
        let k = 4;
        let opts = TdaDavidsonOpts {
            tol: 1e-11,
            max_iter: 500,
            ..TdaDavidsonOpts::default()
        };
        let out = tda_lowest_core(
            TdaHooks {
                a: |v: &TtTangent<f64>| ctx.a_apply(v),
                reproject: |v: &mut TtTangent<f64>| v.gauge_fix_orthogonal_mut(),
                deflate: |_v: &mut TtTangent<f64>| {},
                diag: || ctx.a_diagonal(),
                complement_shift: complement_shift(diag_inf_scale(&ctx.a_diagonal())),
            },
            &seeds(&ctx, k + 2, 0x7DA1),
            k,
            tangent_dim(bl),
            &opts,
        );
        let (rad, _) = leak(&ctx, &out.vectors);
        let shift = out
            .theta
            .iter()
            .zip(&dense)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        println!(
            "U2: undeflated θ {:?} vs deflated dense {:?}; radial weight {rad:e}",
            out.theta,
            &dense[..k]
        );
        assert!(out.info.converged);
        assert!(
            rad > 1e-4,
            "the undeflated solve must carry radial weight (got {rad:e}) — else U1's floor is vacuous"
        );
        assert!(
            shift > 1e-6,
            "the undeflated spectrum must differ from the deflated one (max shift {shift:e})"
        );
    }

    #[test]
    fn tda_u3_theta_is_bitwise_independent_of_curvature_and_floor() {
        let _lock = crate::par::PAR_SWITCH_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (h, x) = c7_point();
        let plain = weingarten_operator(&h, &x);
        let cut = weingarten_operator(&h, &x)
            .with_curvature(CurvatureReg::Hard(0.3))
            .with_floor(FloorPolicy::Hard(1e-2));
        let damped = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(0.3));
        let v0 = seeds(&plain, 6, 0x7DA3);
        let b0 = plain.b_apply(&v0[0]);
        for (what, c) in [("Hard", &cut), ("Tikhonov", &damped)] {
            let mut db = c.b_apply(&v0[0]);
            db.axpy(-1.0, &b0);
            println!(
                "U3 {what}: ‖ΔB ξ‖ = {:.3e} (‖B ξ‖ = {:.3e})",
                db.norm(),
                b0.norm()
            );
            assert!(
                db.norm() > 1e-6 * b0.norm().max(1e-300),
                "{what}: B unchanged"
            );
        }
        let opts = TdaDavidsonOpts {
            tol: 1e-10,
            precond: TdaPrecond::Diag { kappa: 0.0 },
            ..TdaDavidsonOpts::default()
        };
        let base = tda_lowest_davidson(&plain, &v0, 4, &opts);
        for (what, c) in [("Hard", &cut), ("Tikhonov", &damped)] {
            let out = tda_lowest_davidson(c, &v0, 4, &opts);
            assert_eq!(out.info.applies, base.info.applies, "{what}: applies");
            assert_eq!(
                out.info.iterations, base.info.iterations,
                "{what}: iterations"
            );
            for (j, (a, b)) in out.theta.iter().zip(&base.theta).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{what} root {j}: θ {a} vs {b}");
            }
            for (j, (a, b)) in out
                .info
                .residuals
                .iter()
                .zip(&base.info.residuals)
                .enumerate()
            {
                assert_eq!(a.to_bits(), b.to_bits(), "{what} root {j}: residual");
            }
        }
    }

    #[test]
    fn tda_u4_option_resolution_clamps_and_empty_returns() {
        assert_eq!(TdaPrecond::default(), TdaPrecond::Auto);
        assert_eq!(TdaDavidsonOpts::default().precond, TdaPrecond::Auto);
        assert_eq!(TdaPrecond::Auto.resolve(), TdaPrecond::Diag { kappa: 10.0 });
        for p in [
            TdaPrecond::Off,
            TdaPrecond::Diag { kappa: 7.0 },
            TdaPrecond::Shifted,
        ] {
            assert_eq!(p.resolve(), p, "{p:?} passes through");
        }
        let o = |max_subspace, restart_keep| TdaDavidsonOpts {
            max_subspace,
            restart_keep,
            ..TdaDavidsonOpts::default()
        };
        assert_eq!(
            resolve_sizes(4, &o(0, 0)),
            (48, 8),
            "auto: max(8(k+2), 32), 2k"
        );
        assert_eq!(resolve_sizes(1, &o(0, 0)), (32, 2), "auto floor 32");
        assert_eq!(
            resolve_sizes(4, &o(3, 0)),
            (8, 7),
            "raised to 2k, keep below it"
        );
        assert_eq!(
            resolve_sizes(4, &o(20, 100)),
            (20, 19),
            "keep < max_subspace"
        );
        assert_eq!(resolve_sizes(4, &o(20, 1)), (20, 4), "keep ≥ nroots");

        let (h, x) = c7_point();
        let ctx = weingarten_operator(&h, &x);
        let (bl, _) = ctx.bases();
        let radial = radial_mode(&ctx).expect("non-degenerate point");
        let space_dim = tangent_dim(bl) - 1;
        let (c, r) = (&ctx, &radial);
        let hooks = |diag_read_panics: bool| TdaHooks {
            a: move |v: &TtTangent<f64>| c.a_apply(v),
            reproject: move |v: &mut TtTangent<f64>| {
                v.gauge_fix_orthogonal_mut();
                deflate_radial(Some(r), v);
            },
            deflate: move |v: &mut TtTangent<f64>| deflate_radial(Some(r), v),
            diag: move || {
                assert!(!diag_read_panics, "the diagonal model was read");
                let mut d = c.a_diagonal();
                d.for_each_coeff_mut(|e| *e = -1.0);
                d
            },
            complement_shift: 100.0,
        };
        let v0 = seeds(&ctx, 6, 0x7DA4);
        let opts = TdaDavidsonOpts {
            tol: 1e-9,
            precond: TdaPrecond::Off,
            ..TdaDavidsonOpts::default()
        };
        let out = tda_lowest_core(hooks(true), &v0, 3, space_dim, &opts);
        assert!(out.info.converged && out.info.precond == TdaPrecond::Off);
        for precond in [TdaPrecond::Diag { kappa: 0.0 }, TdaPrecond::Auto] {
            let opts = TdaDavidsonOpts {
                tol: 1e-9,
                precond,
                ..TdaDavidsonOpts::default()
            };
            let out = tda_lowest_core(hooks(false), &v0, 3, space_dim, &opts);
            assert!(out.info.converged, "{precond:?}");
            assert_eq!(
                out.info.precond,
                TdaPrecond::Off,
                "{precond:?}: collapsed model ⇒ ran as Off"
            );
        }
        let out = tda_lowest_davidson(&ctx, &v0, 0, &TdaDavidsonOpts::default());
        assert!(out.theta.is_empty() && out.info.converged && out.info.applies == 0);
        let out = tda_lowest_davidson(
            &ctx,
            &[radial.clone(), radial.clone()],
            2,
            &TdaDavidsonOpts::default(),
        );
        assert!(out.theta.is_empty() && !out.info.converged);
        assert_eq!((out.info.dropped_seeds, out.info.applies), (2, 0));
        let out = tda_lowest_davidson(&ctx, &v0, space_dim + 5, &TdaDavidsonOpts::default());
        assert_eq!(out.theta.len(), space_dim);
        assert!(out.info.converged, "exhaustion of the whole deflated space");
    }

    #[test]
    #[should_panic(expected = "at least one seed is required")]
    fn tda_u4_an_empty_seed_list_is_refused() {
        let (h, x) = c7_point();
        let ctx = weingarten_operator(&h, &x);
        let _ = tda_lowest_davidson(&ctx, &[], 2, &TdaDavidsonOpts::default());
    }

    #[test]
    fn tda_u5_the_projector_hook_is_orthogonal_and_the_near_exhaustion_solve_converges() {
        let (h, x) = c7_point();
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.bases();
        let qs = seeds(&ctx, 4, 0x7DA5);
        let mut w = qs[0].clone();
        w.for_each_coeff_mut(|c| *c = (*c * 7.3).sin());
        let mut g = w.clone();
        g.gauge_fix_orthogonal_mut();
        let mut e_g = w.clone();
        e_g.axpy(-1.0, &g);
        let p = TtTangent::project(bl, br, &w.to_vector());
        let mut e_p = w.clone();
        e_p.axpy(-1.0, &p);
        assert!(
            e_g.norm() > 1e-2 && e_p.norm() > 1e-2,
            "w is off the subspace"
        );
        let mut gg = g.clone();
        gg.gauge_fix_orthogonal_mut();
        gg.axpy(-1.0, &g);
        let mut q0 = qs[1].clone();
        q0.gauge_fix_orthogonal_mut();
        q0.axpy(-1.0, &qs[1]);
        assert!(gg.norm() <= 1e-14 && q0.norm() <= 1e-14);
        let (mut worst_g, mut worst_p) = (0.0f64, 0.0f64);
        for q in &qs[1..] {
            worst_g = worst_g.max((q.dot(&e_g) / (q.norm() * e_g.norm())).abs());
            worst_p = worst_p.max((q.dot(&e_p) / (q.norm() * e_p.norm())).abs());
        }
        println!(
            "U5: off-subspace overlap with tangents: orthogonal {worst_g:.2e}, project {worst_p:.2e}"
        );
        assert!(
            worst_g <= 1e-14,
            "the gauge fix is orthogonal ({worst_g:e})"
        );
        assert!(
            worst_p >= 1e-4,
            "project∘to_vector is oblique off the subspace ({worst_p:e}) — else this arm says nothing"
        );
        let dense = dense_a(&ctx);
        let opts = TdaDavidsonOpts {
            tol: 1e-11,
            max_iter: 500,
            max_subspace: dense.len() - 1,
            restart_keep: 4,
            precond: TdaPrecond::Off,
            ..TdaDavidsonOpts::default()
        };
        let out = tda_lowest_davidson(&ctx, &seeds(&ctx, 6, 0x7DA1), 4, &opts);
        println!(
            "U5: converged {} it {} θ {:?} (dense {:?})",
            out.info.converged,
            out.info.iterations,
            out.theta,
            &dense[..4]
        );
        assert!(out.info.converged);
        for (j, t) in out.theta.iter().enumerate() {
            assert!(
                (t - dense[j]).abs() <= 1e-9,
                "root {j}: θ {t} vs dense {} (a Ritz value below the minimum is a ghost)",
                dense[j]
            );
        }
    }

    #[test]
    fn tda_u6_complement_term_is_exact_on_clean_vectors_and_guarded() {
        assert_eq!(
            complement_shift(f64::NAN),
            0.0,
            "a NaN scale disables the shift"
        );
        assert_eq!(
            complement_shift(f64::INFINITY),
            0.0,
            "an infinite scale disables the shift"
        );
        assert_eq!(
            complement_shift(1e300),
            10.0 * (1.0 + 1e100),
            "capped at the 1e100 band"
        );
        assert_eq!(complement_shift(2.0), 30.0);
        let (h, x) = c7_point();
        let ctx = weingarten_operator(&h, &x);
        let radial = radial_mode(&ctx);
        let vs = seeds(&ctx, 2, 0x7DA6);
        let mu = 500.0;
        let mut v = vs[0].clone();
        clean_tangent(radial.as_ref(), &mut v);
        let mut q = v.clone();
        clean_tangent(radial.as_ref(), &mut q);
        let image = ctx.a_apply(&q);
        let mut shifted = image.clone();
        add_complement_shift(&mut shifted, &v, q.clone(), mu);
        let same = (0..image.num_coeffs()).all(|i| {
            if (q.coeff(i) - v.coeff(i)) != 0.0 {
                return true;
            }
            shifted.coeff(i).to_bits() == image.coeff(i).to_bits()
        });
        assert!(same, "the term moved an image entry where q == v");
        let mut w = vs[1].clone();
        w.for_each_coeff_mut(|c| *c = (*c * 3.1).sin());
        let mut qw = w.clone();
        clean_tangent(radial.as_ref(), &mut qw);
        let mut want = w.clone();
        want.axpy(-1.0, &qw);
        want.scale_mut(mu);
        let mut got = ctx.a_apply(&qw);
        let base = got.clone();
        add_complement_shift(&mut got, &w, qw, mu);
        got.axpy(-1.0, &base);
        got.axpy(-1.0, &want);
        assert!(
            got.norm() <= 1e-12 * want.norm(),
            "μ·(v − q): defect {:e}",
            got.norm()
        );
        let mut z = base.clone();
        add_complement_shift(&mut z, &w, w.clone(), 0.0);
        assert!((0..z.num_coeffs()).all(|i| z.coeff(i).to_bits() == base.coeff(i).to_bits()));
    }
}
