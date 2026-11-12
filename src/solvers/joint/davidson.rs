// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::cell::Cell;

use crate::core::tangent::{TtTangent, min_schmidt, project_irrep, tangent_basis, tangent_dim};
use crate::hamiltonian::xt_h_y;
use crate::pointgroup::{Irrep, PointGroup};
use crate::solvers::InnerProductVector;
use crate::solvers::hess::{
    SympLobpcgOpts, SympPrecond, SymplecticDavidsonOpts, SymplecticLowestError,
    SymplecticLowestInfo, a_diagonal_squared, deflate_radial, omega_davidson, omega_from_signed,
    omega_lobp4dcg, reproject_tangent,
};

use crate::vector::symmetry_defect;

use super::white::{CondensedWhite, JointWhite};
use super::{JointContext, JointVector};

fn white_route(ctx: &JointContext<'_>, choice: SympPrecond) -> bool {
    ctx.n_eff() > 0 && choice == SympPrecond::Off && crate::knobs::JOINT_WHITE.get()
}

fn condensed_white(ctx: &JointContext<'_>, max_basis: usize) -> Option<CondensedWhite> {
    let trace = crate::knobs::OMEGA_TRACE.get();
    if !(crate::knobs::JOINT_CONDENSE.get() && 2 * ctx.n_eff() <= max_basis) {
        if trace {
            eprintln!(
                "  joint white route: direct (n_eff {}, max_basis {max_basis})",
                ctx.n_eff()
            );
        }
        return None;
    }
    let t0 = std::time::Instant::now();
    let accurate = crate::knobs::JOINT_ACCURATE_METRIC
        .get()
        .then(|| ctx.accurate_metric_chol());
    let t_metric = t0.elapsed().as_secs_f64();
    let l = match &accurate {
        Some(Some((l, _))) => l.as_slice(),
        _ => ctx.nv_chol.as_slice(),
    };
    let cw = CondensedWhite::build(ctx, l);
    if trace {
        let metric = match &accurate {
            None => "production".to_string(),
            Some(None) => "production (accurate M′ not PD)".to_string(),
            Some(Some((_, dev))) => {
                format!("accurate ({t_metric:.2e}s; production deviation {dev:.1e})")
            }
        };
        eprintln!(
            "  joint white route: condensed (n_eff {}, {} precompute applies {:.2e}s; metric {metric}; \
             orbital-block asymmetry removed: A {:.1e}, B {:.1e})",
            ctx.n_eff(),
            cw.n_pre,
            t0.elapsed().as_secs_f64() - t_metric,
            cw.asym_a,
            cw.asym_b
        );
    }
    Some(cw)
}

/// Conditioning diagnostics of a joint solve, gathered from the context and its joint basis — the
/// numbers an ω must be quoted next to (§5).
#[derive(Clone, Debug)]
pub struct JointSolverDiagnostics {
    /// Kept orbital coordinates (`0` ⇒ the solve degenerates exactly to TT-RPA).
    pub n_eff: usize,
    /// Tangent-basis dimension (radial excluded) at the context's frames.
    pub n_tt: usize,
    /// The iteration-space dimension the kernels were capped at.
    pub space_dim: usize,
    /// Smallest kept eigenvalue of `N = S − τᵀτ` (`f64::INFINITY` when `n_eff == 0`).
    pub mu_min: f64,
    /// Largest kept eigenvalue of `N` (`0.0` when `n_eff == 0`).
    pub mu_max: f64,
    /// Orbital seeds the joint-basis MGS had to drop.
    pub dropped: usize,
    /// [`super::DropReport::orth_error`] — the G-orthonormality error budget of the joint basis.
    pub orth_error: f64,
}

/// Diagnostic accessor for the production RAISED orbital preconditioner diagonal (§2), exposed so
/// integration tests outside this crate's module tree (which cannot see the `pub(crate)`
/// [`raised_orbital_diagonal`]) can pin the ACTUAL production formula against the dense
/// `a_apply(e_j).c[j]` oracle, rather than only an independently reimplemented copy of it.
/// Not part of the solving path — [`joint_lowest_davidson`]/[`joint_lowest_lobp4dcg`] call
/// `raised_orbital_diagonal` directly.
pub fn joint_raised_orbital_diagonal_diagnostic(ctx: &JointContext<'_>) -> Vec<f64> {
    raised_orbital_diagonal(ctx)
}

/// Gather [`JointSolverDiagnostics`] at `ctx`. Costs one [`super::JointContext::joint_basis`]
/// pass (`n_eff` MGS rounds over `n_tt + n_eff` vectors, no applies), so call it once beside a
/// solve rather than inside a loop.
pub fn joint_solver_diagnostics(ctx: &JointContext<'_>) -> JointSolverDiagnostics {
    let (_basis, report) = ctx.joint_basis();
    let mu = ctx.orb_kept();
    JointSolverDiagnostics {
        n_eff: ctx.n_eff(),
        n_tt: report.n_tt,
        space_dim: joint_space_dim(ctx),
        mu_min: mu.iter().copied().fold(f64::INFINITY, f64::min),
        mu_max: mu.iter().copied().fold(0.0f64, f64::max),
        dropped: report.dropped,
        orth_error: report.orth_error,
    }
}

fn joint_space_dim(ctx: &JointContext<'_>) -> usize {
    let (base_l, _base_r) = ctx.bases();
    let radial = ctx.radial();
    tangent_dim(base_l) - radial.map_or(0, |_| 1) + ctx.n_eff()
}

pub(crate) fn raised_orbital_diagonal(ctx: &JointContext<'_>) -> Vec<f64> {
    let n_eff = ctx.n_eff();
    if n_eff == 0 {
        return Vec::new();
    }
    let n_orb = ctx.pairs().len();
    let h = ctx.h;
    let lambda = ctx.lambda();
    let u = ctx.orbital_directions();
    let tau = ctx.orbital_projections();
    let s = ctx.orbital_gram();

    let mut d_pairs = vec![0.0f64; n_orb * n_orb];
    for a in 0..n_orb {
        for b in a..n_orb {
            let val = xt_h_y(&u[a], h, &u[b]) - lambda * s[a * n_orb + b];
            d_pairs[a * n_orb + b] = val;
            d_pairs[b * n_orb + a] = val;
        }
    }

    let mut out = Vec::with_capacity(n_eff);
    for j in 0..n_eff {
        let mut ej = vec![0.0f64; n_eff];
        ej[j] = 1.0;
        let vj = ctx.v_times(&ej);

        let mut f_c = 0.0f64;
        for a in 0..n_orb {
            let mut row = 0.0f64;
            for b in 0..n_orb {
                row += d_pairs[a * n_orb + b] * vj[b];
            }
            f_c += vj[a] * row;
        }

        let tv = &ctx.spec().tau_v[j];
        let tv_vec = tv.to_vector();
        let mut cross = 0.0f64;
        for b in 0..n_orb {
            if vj[b] == 0.0 {
                continue;
            }
            cross += vj[b] * (xt_h_y(&tv_vec, h, &u[b]) - lambda * tv.dot(&tau[b]));
        }

        out.push((f_c - cross) / ctx.orb_kept()[j]);
    }
    out
}

fn joint_diagonal_squared(ctx: &JointContext<'_>) -> JointVector {
    let mut d = JointVector::from_tangent(ctx.spec(), a_diagonal_squared(ctx.weingarten()));
    d.c = raised_orbital_diagonal(ctx)
        .into_iter()
        .map(|x| x * x)
        .collect();
    d
}

type JointPrecondFn<'a> = Box<dyn FnMut(&mut JointVector, f64) + 'a>;

fn build_joint_symp_precond<'a>(
    ctx: &'a JointContext<'_>,
    choice: SympPrecond,
) -> Option<JointPrecondFn<'a>> {
    crate::solvers::hess::build_symp_precond(choice, || joint_diagonal_squared(ctx))
        .map(|f| Box::new(f) as JointPrecondFn<'a>)
}

/// The lowest `k` joint symplectic (RPA) eigenvalues ω at `ctx`'s base point — preconditioned
/// block Davidson on the W-metric product `M = H^rr∘H^ii`, matrix-free, real `f64` layer.
///
/// # Errors
///
/// [`SymplecticLowestError::MetricNotPd`] when `W = A−B` is indefinite on the deflated space
/// (§4); [`SymplecticLowestError::NegativeRitz`] when the product spectrum carries a genuinely
/// negative Ritz value (a fresh-apply variational certificate is printed first).
pub fn joint_lowest_davidson(
    ctx: &JointContext<'_>,
    v0s: &[JointVector],
    k: usize,
    opts: &SymplecticDavidsonOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let space_dim = tangent_dim(base_l) - radial.map_or(0, |_| 1) + ctx.n_eff();
    let choice = opts.precond.resolve_joint();
    let (omega2, info) = if white_route(ctx, choice) {
        let cw = condensed_white(ctx, crate::solvers::hess::vector_budget(opts));
        let y0s: Vec<JointWhite> = v0s
            .iter()
            .map(|v| match &cw {
                Some(cw) => cw.to_white(ctx, v),
                None => ctx.to_white(v),
            })
            .collect();
        let (omega2, mut info) = omega_davidson(
            |y| match &cw {
                Some(cw) => cw.apply(ctx, y, 1.0),
                None => ctx.hpp_white_direct(y),
            },
            |y| match &cw {
                Some(cw) => cw.apply(ctx, y, -1.0),
                None => ctx.hmm_white_direct(y),
            },
            |y: &mut JointWhite| deflate_radial(radial, &mut y.xi),
            |y: &mut JointWhite| reproject_tangent(base_l, base_r, radial, &mut y.xi),
            None::<fn(&mut JointWhite, f64)>,
            &y0s,
            k,
            space_dim,
            opts,
        )?;
        info.applies += cw.as_ref().map_or(0, |cw| cw.n_pre);
        (omega2, info)
    } else {
        omega_davidson(
            |w| ctx.hpp_apply(w),
            |v| ctx.hmm_apply(v),
            |v: &mut JointVector| deflate_radial(radial, &mut v.xi),
            |v: &mut JointVector| reproject_tangent(base_l, base_r, radial, &mut v.xi),
            build_joint_symp_precond(ctx, choice),
            v0s,
            k,
            space_dim,
            opts,
        )?
    };
    let omega = omega_from_signed(&omega2, &info, "joint_lowest_davidson")?;
    Ok((omega, info))
}

#[doc(hidden)]
pub fn joint_lowest_davidson_with_precond(
    ctx: &JointContext<'_>,
    v0s: &[JointVector],
    k: usize,
    opts: &SymplecticDavidsonOpts,
    precond: impl FnMut(&mut JointVector, f64),
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let space_dim = tangent_dim(base_l) - radial.map_or(0, |_| 1) + ctx.n_eff();
    let (omega2, info) = omega_davidson(
        |w| ctx.hpp_apply(w),
        |v| ctx.hmm_apply(v),
        |v: &mut JointVector| deflate_radial(radial, &mut v.xi),
        |v: &mut JointVector| reproject_tangent(base_l, base_r, radial, &mut v.xi),
        Some(precond),
        v0s,
        k,
        space_dim,
        opts,
    )?;
    let omega = omega_from_signed(&omega2, &info, "joint_lowest_davidson_with_precond")?;
    Ok((omega, info))
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct WhiteRouteCost {
    /// Kept orbital coordinates.
    pub n_eff: usize,
    /// Coefficients in one tangent: the unit of the stored images.
    pub tangent_coeffs: usize,
    /// One raised `A + B` apply (`hpp_apply`, the route before the white coordinates),
    /// median seconds.
    pub raised_apply: f64,
    /// One direct white `A + B` apply, median seconds.
    pub direct_white_apply: f64,
    /// One condensed `H⁺` apply, median seconds.
    pub condensed_apply: f64,
    /// The condensed precompute against `nv_chol` (`n_eff` direct applies), seconds, one call.
    pub precompute: f64,
    /// The accurate metric factor (`white` §7), seconds, one call.
    pub accurate_metric: f64,
    /// Bytes the condensed operator holds: `2·n_eff` tangents and the two `n_eff²` blocks.
    pub condensed_bytes: usize,
}

fn median_wall(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t: Vec<f64> = (0..reps.max(1))
        .map(|_| {
            let t0 = std::time::Instant::now();
            f();
            t0.elapsed().as_secs_f64()
        })
        .collect();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}

#[doc(hidden)]
pub fn joint_white_route_cost(
    ctx: &JointContext<'_>,
    v: &JointVector,
    reps: usize,
) -> WhiteRouteCost {
    assert!(
        ctx.n_eff() > 0,
        "joint_white_route_cost: no orbital block (n_eff = 0)"
    );
    let y = ctx.to_white(v);
    let raised_apply = median_wall(reps, || {
        std::hint::black_box(ctx.hpp_apply(v));
    });
    let direct_white_apply = median_wall(reps, || {
        std::hint::black_box(ctx.hpp_white_direct(&y));
    });
    let t0 = std::time::Instant::now();
    let cw = CondensedWhite::build(ctx, &ctx.nv_chol);
    let precompute = t0.elapsed().as_secs_f64();
    let condensed_apply = median_wall(reps, || {
        std::hint::black_box(cw.apply(ctx, &y, 1.0));
    });
    let t0 = std::time::Instant::now();
    std::hint::black_box(ctx.accurate_metric_chol());
    let accurate_metric = t0.elapsed().as_secs_f64();
    let n = ctx.n_eff();
    let tangent_coeffs = v.xi.num_coeffs();
    WhiteRouteCost {
        n_eff: n,
        tangent_coeffs,
        raised_apply,
        direct_white_apply,
        condensed_apply,
        precompute,
        accurate_metric,
        condensed_bytes: 8 * (2 * n * tangent_coeffs + 2 * n * n),
    }
}

/// The lowest `k` joint symplectic (RPA) eigenvalues ω — an opt-in paired locally-optimal
/// block CG (ELOBP4dCG), the constant-memory / no-restart alternative to
/// [`joint_lowest_davidson`]. Same operator, same metric, same four closures and the same
/// preconditioner model (§2); only the subspace strategy differs, which is what makes an
/// arm-for-arm comparison of the two meaningful. Block size `= v0s.len()`, which must be `≥ k`
/// (shifting deflation is not implemented). Derivation and deferrals:
/// [`crate::solvers::hess::SympLobpcgOpts`] and the `lobpcg` module header.
///
/// # Errors
///
/// As [`joint_lowest_davidson`].
pub fn joint_lowest_lobp4dcg(
    ctx: &JointContext<'_>,
    v0s: &[JointVector],
    k: usize,
    opts: &SympLobpcgOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let space_dim = tangent_dim(base_l) - radial.map_or(0, |_| 1) + ctx.n_eff();
    let choice = opts.precond.resolve_joint();
    let (omega2, info) = if white_route(ctx, choice) {
        let cw = condensed_white(ctx, opts.max_basis);
        let y0s: Vec<JointWhite> = v0s
            .iter()
            .map(|v| match &cw {
                Some(cw) => cw.to_white(ctx, v),
                None => ctx.to_white(v),
            })
            .collect();
        let (omega2, mut info) = omega_lobp4dcg(
            |y| match &cw {
                Some(cw) => cw.apply(ctx, y, 1.0),
                None => ctx.hpp_white_direct(y),
            },
            |y| match &cw {
                Some(cw) => cw.apply(ctx, y, -1.0),
                None => ctx.hmm_white_direct(y),
            },
            |y: &mut JointWhite| deflate_radial(radial, &mut y.xi),
            |y: &mut JointWhite| reproject_tangent(base_l, base_r, radial, &mut y.xi),
            None::<fn(&mut JointWhite, f64)>,
            &y0s,
            k,
            space_dim,
            opts,
        )?;
        info.applies += cw.as_ref().map_or(0, |cw| cw.n_pre);
        (omega2, info)
    } else {
        omega_lobp4dcg(
            |w| ctx.hpp_apply(w),
            |v| ctx.hmm_apply(v),
            |v: &mut JointVector| deflate_radial(radial, &mut v.xi),
            |v: &mut JointVector| reproject_tangent(base_l, base_r, radial, &mut v.xi),
            build_joint_symp_precond(ctx, choice),
            v0s,
            k,
            space_dim,
            opts,
        )?
    };
    let omega = omega_from_signed(&omega2, &info, "joint_lowest_lobp4dcg")?;
    Ok((omega, info))
}

/// The reference's own irrep and how far it is from being a symmetry eigenstate — the
/// PRECONDITION of every per-irrep number, in one struct so it can be quoted beside one.
#[derive(Clone, Debug)]
pub struct IrrepReference {
    /// The measured Γ₀ of the base point.
    pub gamma0: Irrep,
    /// `‖U_{h_j}x − χ_{Γ₀}(h_j)x‖/‖x‖` over the generators `h_j = 2^j`, `j` ascending. Empty at a
    /// trivial group — FOLD it, never `.unwrap()` a `max()`.
    pub defects: Vec<f64>,
    /// `max(defects)`, `0.0` at a trivial group.
    pub worst_defect: f64,
    /// σ_min at the base point. A defect is not interpretable without it.
    pub sigma_min: f64,
}

/// Measure the base point's irrep and its symmetry breaking. Cheap: `log2|G|` sign sweeps plus
/// one bond-gauge fold, no applies.
pub fn joint_irrep_reference(ctx: &JointContext<'_>, pg: PointGroup) -> IrrepReference {
    let (base_l, base_r) = ctx.shared_bases();
    let (gamma0, defects) = symmetry_defect(&ctx.x, pg);
    let worst_defect = defects.iter().copied().fold(0.0f64, f64::max);
    IrrepReference {
        gamma0,
        defects,
        worst_defect,
        sigma_min: min_schmidt(base_l, base_r),
    }
}

/// What [`joint_irrep_basis`] built, and how trustworthy it is.
#[derive(Clone, Debug)]
pub struct IrrepBasisReport {
    /// The irrep this block was filtered to.
    pub gamma: Irrep,
    /// Vectors kept (`= basis.len()`), the MEASURED dimension of the Γ block.
    pub kept: usize,
    /// Projected candidates culled by the MGS floors.
    pub dropped: usize,
    /// Largest `P_Γ` leakage seen while projecting the tangent candidates.
    pub max_leakage: f64,
    /// σ_min at the base point — the mandatory companion to `max_leakage`.
    pub sigma_min: f64,
    /// `min_i Σ_Γ ‖P_Γ e_i‖²/‖e_i‖²` over the tangent candidates: how well the projectors resolve
    /// the identity at this base point. `1 − O(ε)` is healthy; a visible shortfall means the
    /// decomposition is being run at too small a σ_min (trim the state and retry).
    pub min_closure: f64,
    /// Whether the orbital sector joined this block (`gamma == gamma0`, §7).
    pub orbital_kept: bool,
}

const IRREP_MGS_REL_TOL: f64 = 1e-8;
const IRREP_MGS_NOISE_FLOOR: f64 = 1e-13;

/// The Γ block of the joint basis: `P_Γ` every radial-excluded tangent basis vector, cull what
/// falls below the MGS floors, re-orthonormalize, and append the `n_eff` orbital seeds **only when
/// `gamma == gamma0`** (§7 — for any other Γ the orbital sector is structurally absent, not small).
pub fn joint_irrep_basis(
    ctx: &JointContext<'_>,
    pg: PointGroup,
    gamma: Irrep,
    gamma0: Irrep,
) -> (Vec<JointVector>, IrrepBasisReport) {
    let (base_l, base_r) = ctx.shared_bases();
    let candidates = tangent_basis(base_l, base_r, true);
    let mut block: Vec<TtTangent<f64>> = Vec::new();
    let mut max_leakage = 0.0f64;
    let mut min_closure = f64::INFINITY;
    let mut dropped = 0usize;

    for xi in &candidates {
        let n0 = xi.norm();
        let mut closure = 0.0f64;
        let mut proj_gamma: Option<TtTangent<f64>> = None;
        for hi in 0..pg.order() {
            let g = Irrep::new(hi);
            let (p, leak) = project_irrep(xi, base_l, base_r, pg, g);
            if n0 > 0.0 {
                closure += (p.norm() / n0).powi(2);
            }
            if g == gamma {
                max_leakage = max_leakage.max(leak);
                proj_gamma = Some(p);
            }
        }
        if n0 > 0.0 {
            min_closure = min_closure.min(closure);
        }
        let mut v = proj_gamma.expect("gamma must be an irrep of pg");
        for _pass in 0..2 {
            for b in block.iter() {
                let c = TtTangent::dot(b, &v);
                v.axpy(-c, b);
            }
        }
        let nrm = v.norm();
        if nrm > IRREP_MGS_REL_TOL * n0.max(f64::MIN_POSITIVE) && nrm > IRREP_MGS_NOISE_FLOOR {
            v.scale_mut(1.0 / nrm);
            block.push(v);
        } else {
            dropped += 1;
        }
    }

    let mut basis: Vec<JointVector> = block
        .into_iter()
        .map(|xi| JointVector::from_tangent(ctx.spec(), xi))
        .collect();

    let orbital_kept = gamma == gamma0;
    if orbital_kept {
        let n_eff = ctx.n_eff();
        for j in 0..n_eff {
            let mut c = vec![0.0f64; n_eff];
            c[j] = 1.0;
            let mut seed = JointVector::from_coeffs(ctx.spec(), ctx, c);
            for _pass in 0..2 {
                for b in basis.iter() {
                    let cf = b.dot(&seed);
                    seed.axpy(-cf, b);
                }
            }
            let nrm = seed.norm();
            let root_mu = ctx.orb_kept()[j].max(0.0).sqrt();
            if nrm >= super::DROP_REL * root_mu && nrm > 0.0 {
                seed.scale(1.0 / nrm);
                basis.push(seed);
            } else {
                dropped += 1;
            }
        }
    }

    let report = IrrepBasisReport {
        gamma,
        kept: basis.len(),
        dropped,
        max_leakage,
        sigma_min: min_schmidt(base_l, base_r),
        min_closure: if min_closure.is_finite() {
            min_closure
        } else {
            0.0
        },
        orbital_kept,
    };
    (basis, report)
}

/// Drift diagnostics of a Γ-filtered Krylov run — what the reproject hook had to remove.
#[derive(Clone, Debug)]
pub struct IrrepFilterReport {
    /// The irrep the iteration was confined to.
    pub gamma: Irrep,
    /// The reference's irrep, as supplied by the caller.
    pub gamma0: Irrep,
    /// Largest `P_Γ` leakage seen at any reprojection — the drift readout. At an exactly
    /// symmetry-pure reference this stays at the projector's own `ε/σ_min` floor; a growing value
    /// is the base point's symmetry breaking showing up in the iteration.
    pub max_leakage: f64,
    /// Largest leakage seen while projecting the SEEDS (before the first apply): how far the
    /// caller's start block sat outside the block. A large value is a seeding problem, not drift.
    pub seed_leakage: f64,
    /// σ_min at the base point — the mandatory companion to both leakage numbers.
    pub sigma_min: f64,
    /// Whether the orbital leg was kept (`gamma == gamma0`, §7).
    pub orbital_kept: bool,
}

/// [`joint_lowest_davidson`] with `P_Γ` composed into the reprojection hook — the lowest `k` joint
/// symplectic (RPA) eigenvalues **of the Γ block alone**.
///
/// # Panics
///
/// If every seed collapses under `P_Γ` (no start block left): the caller asked for a block that
/// their seeds do not reach, or for a Γ that is empty at this base point.
///
/// # Errors
///
/// As [`joint_lowest_davidson`].
pub fn joint_lowest_davidson_irrep(
    ctx: &JointContext<'_>,
    v0s: &[JointVector],
    k: usize,
    pg: PointGroup,
    gamma: Irrep,
    gamma0: Irrep,
    opts: &SymplecticDavidsonOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo, IrrepFilterReport), SymplecticLowestError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let space_dim = joint_space_dim(ctx);
    let orbital_kept = gamma == gamma0;
    let leak = Cell::new(0.0f64);

    let filter = |v: &mut JointVector| {
        reproject_tangent(base_l, base_r, radial, &mut v.xi);
        let (p, l) = project_irrep(&v.xi, base_l, base_r, pg, gamma);
        v.xi = p;
        leak.set(leak.get().max(l));
        deflate_radial(radial, &mut v.xi);
        if !orbital_kept {
            v.c.iter_mut().for_each(|c| *c = 0.0);
        }
    };

    let seeds: Vec<JointVector> = v0s
        .iter()
        .map(|v| {
            let mut w = v.clone();
            filter(&mut w);
            w
        })
        .filter(|w| w.norm() > 0.0)
        .collect();
    let seed_leakage = leak.get();
    assert!(
        !seeds.is_empty(),
        "joint_lowest_davidson_irrep: every seed collapsed under P_Γ (Γ={gamma:?}) — \
         seed the block, or check that Γ is non-empty at this base point"
    );

    let precond_fn = build_joint_symp_precond(ctx, opts.precond.resolve_joint());
    let (omega2, info) = omega_davidson(
        |w| ctx.hpp_apply(w),
        |v| ctx.hmm_apply(v),
        |v: &mut JointVector| deflate_radial(radial, &mut v.xi),
        filter,
        precond_fn,
        &seeds,
        k,
        space_dim,
        opts,
    )?;
    let omega = omega_from_signed(&omega2, &info, "joint_lowest_davidson_irrep")?;
    let report = IrrepFilterReport {
        gamma,
        gamma0,
        max_leakage: leak.get(),
        seed_leakage,
        sigma_min: min_schmidt(base_l, base_r),
        orbital_kept,
    };
    Ok((omega, info, report))
}
