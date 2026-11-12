// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Matrix-free symplectic (RPA) eigensolvers on the SU(2) JOINT (TT ⊕ orbital-rotation) manifold:
//! the preconditioned block Davidson [`su2_joint_lowest_davidson`] and the opt-in paired
//! locally-optimal block CG [`su2_joint_lowest_lobp4dcg`]. Both bind the UNCHANGED generic kernels
//! ([`omega_davidson`] / [`omega_lobp4dcg`]) to [`Su2JointVector`] applies through its
//! `InnerProductVector` impl — nothing about the iteration is layer-specific.

use crate::solvers::hess::{
    SympLobpcgOpts, SympPrecond, SymplecticDavidsonOpts, SymplecticLowestError,
    SymplecticLowestInfo, omega_davidson, omega_from_signed, omega_lobp4dcg,
};
use crate::solvers::joint::{JointSolverDiagnostics, WhiteRouteCost};
use crate::su2::hamiltonian::xt_h_y;
use crate::su2::hess::{a_diagonal_squared, deflate_radial, reproject_tangent};
use crate::su2::weingarten::tangent_dim;

use super::white::{Su2CondensedWhite, Su2JointWhite};
use super::{Su2JointContext, Su2JointVector};

fn white_route(ctx: &Su2JointContext<'_>, choice: SympPrecond) -> bool {
    ctx.n_eff() > 0 && choice == SympPrecond::Off && crate::knobs::JOINT_WHITE.get()
}

fn condensed_white(ctx: &Su2JointContext<'_>, max_basis: usize) -> Option<Su2CondensedWhite> {
    let trace = crate::knobs::OMEGA_TRACE.get();
    if !(crate::knobs::JOINT_CONDENSE.get() && 2 * ctx.n_eff() <= max_basis) {
        if trace {
            eprintln!(
                "  su2 joint white route: direct (n_eff {}, max_basis {max_basis})",
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
    let cw = Su2CondensedWhite::build(ctx, l);
    if trace {
        let metric = match &accurate {
            None => "production".to_string(),
            Some(None) => "production (accurate M′ not PD)".to_string(),
            Some(Some((_, dev))) => {
                format!("accurate ({t_metric:.2e}s; production deviation {dev:.1e})")
            }
        };
        eprintln!(
            "  su2 joint white route: condensed (n_eff {}, {} precompute applies {:.2e}s; metric \
             {metric}; orbital-block asymmetry removed: A {:.1e}, B {:.1e})",
            ctx.n_eff(),
            cw.n_pre,
            t0.elapsed().as_secs_f64() - t_metric,
            cw.asym_a,
            cw.asym_b
        );
    }
    Some(cw)
}

/// Diagnostic accessor for the production RAISED orbital preconditioner diagonal, exposed so
/// integration tests outside this crate's module tree can pin the ACTUAL production formula
/// against the dense `a_apply(e_j).c[j]` oracle rather than only a reimplemented copy of it. Not
/// part of the solving path.
pub fn su2_joint_raised_orbital_diagonal(ctx: &Su2JointContext<'_>) -> Vec<f64> {
    raised_orbital_diagonal(ctx)
}

/// Gather [`JointSolverDiagnostics`] at `ctx` — the numbers an ω must be quoted next to. Costs one
/// [`Su2JointContext::joint_basis`] pass (no applies), so call it once beside a solve rather than
/// inside a loop.
pub fn su2_joint_solver_diagnostics(ctx: &Su2JointContext<'_>) -> JointSolverDiagnostics {
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

fn joint_space_dim(ctx: &Su2JointContext<'_>) -> usize {
    let (base_l, _base_r) = ctx.bases();
    let radial = ctx.radial();
    tangent_dim(base_l) - radial.map_or(0, |_| 1) + ctx.n_eff()
}

pub(crate) fn raised_orbital_diagonal(ctx: &Su2JointContext<'_>) -> Vec<f64> {
    let n_eff = ctx.n_eff();
    if n_eff == 0 {
        return Vec::new();
    }
    let n_orb = ctx.pairs().len();
    let h = ctx.hamiltonian();
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

fn joint_diagonal_squared(ctx: &Su2JointContext<'_>) -> Su2JointVector {
    let mut d = Su2JointVector::from_tangent(ctx.spec(), a_diagonal_squared(ctx.weingarten()));
    d.c = raised_orbital_diagonal(ctx)
        .into_iter()
        .map(|x| x * x)
        .collect();
    d
}

type Su2JointPrecondFn<'a> = Box<dyn FnMut(&mut Su2JointVector, f64) + 'a>;

fn build_joint_symp_precond<'a>(
    ctx: &'a Su2JointContext<'_>,
    choice: SympPrecond,
) -> Option<Su2JointPrecondFn<'a>> {
    crate::solvers::hess::build_symp_precond(choice, || joint_diagonal_squared(ctx))
        .map(|f| Box::new(f) as Su2JointPrecondFn<'a>)
}

/// The lowest `k` SU(2) joint symplectic (RPA) eigenvalues ω at `ctx`'s base point —
/// preconditioned block Davidson on the W-metric product `M = H^rr∘H^ii`, matrix-free, real `f64`.
///
/// # Errors
///
/// [`SymplecticLowestError::MetricNotPd`] when `W = A−B` is indefinite on the deflated space;
/// [`SymplecticLowestError::NegativeRitz`] when the product spectrum carries a genuinely negative
/// Ritz value.
pub fn su2_joint_lowest_davidson(
    ctx: &Su2JointContext<'_>,
    v0s: &[Su2JointVector],
    k: usize,
    opts: &SymplecticDavidsonOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let space_dim = joint_space_dim(ctx);
    let choice = opts.precond.resolve_joint();
    let (omega2, info) = if white_route(ctx, choice) {
        let cw = condensed_white(ctx, crate::solvers::hess::vector_budget(opts));
        let y0s: Vec<Su2JointWhite> = v0s
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
            |y: &mut Su2JointWhite| deflate_radial(radial, &mut y.xi),
            |y: &mut Su2JointWhite| reproject_tangent(base_l, base_r, radial, &mut y.xi),
            None::<fn(&mut Su2JointWhite, f64)>,
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
            |v: &mut Su2JointVector| deflate_radial(radial, &mut v.xi),
            |v: &mut Su2JointVector| reproject_tangent(base_l, base_r, radial, &mut v.xi),
            build_joint_symp_precond(ctx, choice),
            v0s,
            k,
            space_dim,
            opts,
        )?
    };
    let omega = omega_from_signed(&omega2, &info, "su2_joint_lowest_davidson")?;
    Ok((omega, info))
}

/// The lowest `k` SU(2) joint symplectic (RPA) eigenvalues ω — an opt-in paired
/// locally-optimal block CG (ELOBP4dCG), the constant-memory / no-restart alternative to
/// [`su2_joint_lowest_davidson`]. Same operator, same metric, same four closures and the same
/// preconditioner model; only the subspace strategy differs. Block size `= v0s.len()`, which must
/// be `≥ k`.
///
/// # Errors
///
/// As [`su2_joint_lowest_davidson`].
pub fn su2_joint_lowest_lobp4dcg(
    ctx: &Su2JointContext<'_>,
    v0s: &[Su2JointVector],
    k: usize,
    opts: &SympLobpcgOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError> {
    let (base_l, base_r) = ctx.shared_bases();
    let radial = ctx.radial();
    let space_dim = joint_space_dim(ctx);
    let choice = opts.precond.resolve_joint();
    let (omega2, info) = if white_route(ctx, choice) {
        let cw = condensed_white(ctx, opts.max_basis);
        let y0s: Vec<Su2JointWhite> = v0s
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
            |y: &mut Su2JointWhite| deflate_radial(radial, &mut y.xi),
            |y: &mut Su2JointWhite| reproject_tangent(base_l, base_r, radial, &mut y.xi),
            None::<fn(&mut Su2JointWhite, f64)>,
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
            |v: &mut Su2JointVector| deflate_radial(radial, &mut v.xi),
            |v: &mut Su2JointVector| reproject_tangent(base_l, base_r, radial, &mut v.xi),
            build_joint_symp_precond(ctx, choice),
            v0s,
            k,
            space_dim,
            opts,
        )?
    };
    let omega = omega_from_signed(&omega2, &info, "su2_joint_lowest_lobp4dcg")?;
    Ok((omega, info))
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
pub fn su2_joint_white_route_cost(
    ctx: &Su2JointContext<'_>,
    v: &Su2JointVector,
    reps: usize,
) -> WhiteRouteCost {
    assert!(
        ctx.n_eff() > 0,
        "su2_joint_white_route_cost: no orbital block (n_eff = 0)"
    );
    let y = ctx.to_white(v);
    let raised_apply = median_wall(reps, || {
        std::hint::black_box(ctx.hpp_apply(v));
    });
    let direct_white_apply = median_wall(reps, || {
        std::hint::black_box(ctx.hpp_white_direct(&y));
    });
    let t0 = std::time::Instant::now();
    let cw = Su2CondensedWhite::build(ctx, &ctx.nv_chol);
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
