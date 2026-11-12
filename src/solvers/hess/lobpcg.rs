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

use super::davidson::{SympPrecond, recombine, survival_guard_trips, w_orthonormalize_push};
use super::lanczos::{SympDiagnostics, SymplecticLowestError, SymplecticLowestInfo};

/// How [`omega_lobp4dcg`] forms and conditions the locally-optimal previous-direction block `P′`
/// — the ONE place the kernel's search space can lose rank, and (measured) the place it does.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum BasisSelection {
    /// The certified path: `P′` is the raw Ritz-coefficient tail recombination, appended one
    /// column at a time through the sequential greedy [`w_reorthonormalize_triple`], which DROPS a
    /// whole candidate triple whenever the direction collapses.
    #[default]
    Legacy,
}

/// Options for [`omega_lobp4dcg`] and its layer wrapper
/// [`crate::solvers::hess::symplectic_lowest_lobp4dcg`]. Deliberately a subset of
/// [`super::davidson::SymplecticDavidsonOpts`] with the SAME field semantics where the names
/// coincide — the two solvers are compared arm-for-arm, and a knob that meant something slightly
/// different in one of them would silently make the comparison meaningless. There is no
/// `max_subspace`: this kernel's search space has a fixed dimension by construction.
#[derive(Clone, Copy, Debug)]
pub struct SympLobpcgOpts {
    /// Residual tolerance relative to the largest wanted Ritz value: converge when every wanted
    /// pair's TRUE residual g-norm `‖M x_j − θ_j x_j‖_g` is `≤ tol·max|θ|` (θ = ω² units).
    /// Identical semantics AND test expression to the joint layer's unsigned W-metric Davidson
    /// ([`super::davidson::SymplecticDavidsonOpts::tol`] there); the signed entries read that
    /// field in ω units instead.
    pub tol: f64,
    /// Maximum outer iterations. Each costs at most one `W`-apply and one `R`-apply per accepted
    /// correction column (`≤ k` of them), plus zero-apply block algebra.
    pub max_iter: usize,
    /// Total budget of accepted correction VECTORS across the run (2 applies each), kept so
    /// built-vector counts are directly comparable with the W-metric Davidson. That one's
    /// [`super::davidson::SymplecticDavidsonOpts::max_basis`] counts pair-equivalents: equal work
    /// is `SymplecticDavidsonOpts::max_basis = 2 ×` this.
    pub max_basis: usize,
    /// Which preconditioner to apply to each correction — the same [`super::davidson::SympPrecond`]
    /// ladder and the same layer-built closures as the Davidson, so the two kernels stay
    /// comparable arm for arm. Default [`super::davidson::SympPrecond::Auto`], resolved per layer
    /// by the wrapper exactly as the Davidson's is (TT → the D0 fixed unshifted inverse, joint →
    /// [`super::davidson::SympPrecond::Off`]). The θ-shifted arm HARMS this kernel, and
    /// because it has NO restart to blame, its refutation here is the decisive one; D0
    /// has never been measured on this kernel.
    pub precond: SympPrecond,
    /// Leading outer iterations whose corrections skip the preconditioner — the first Ritz values
    /// are too poor to shift with ([`super::davidson::SymplecticDavidsonOpts::nwarm`]).
    pub nwarm: usize,
    /// Collect the per-outer-iteration basis W-Gram diagnostic
    /// ([`crate::solvers::hess::SympDiagnostics::gram_min_eig`]): `m²` extra dots per outer
    /// iteration, ZERO operator applies, and no effect on the iteration itself — the Gram is
    /// read-only on the basis, so ω, residuals and every count are bit-identical either way.
    /// Default `false`. The cheap diagnostics (drop counters, survival ratios, collapse angles)
    /// are collected unconditionally. Unrelated to the crate's `instrument` cargo feature, which
    /// is Hamiltonian-construction telemetry.
    pub instrument: bool,
    /// How the locally-optimal previous-direction block `P′` is formed and conditioned — see
    /// [`BasisSelection`]. Default [`BasisSelection::Legacy`], the certified path.
    pub basis: BasisSelection,
    /// SURVIVAL GUARD threshold, with the identical semantics — and the identical statistic,
    /// window rule and default `0.0` = off — as
    /// [`super::davidson::SymplecticDavidsonOpts::guard_survival`], which carries the measured
    /// discussion. The only kernel-specific detail: this kernel's window is its block size
    /// `vs.len()` AFTER the warm start, i.e. the number of seeds that survived it, so a collapsed
    /// seed narrows the window here where it would not in the Davidson.
    pub guard_survival: f64,
}

impl Default for SympLobpcgOpts {
    fn default() -> Self {
        SympLobpcgOpts {
            tol: 1e-8,
            max_iter: 200,
            max_basis: 400,
            precond: SympPrecond::Auto,
            nwarm: 2,
            instrument: false,
            basis: BasisSelection::Legacy,
            guard_survival: 0.0,
        }
    }
}

pub(crate) fn rr_matrix_symmetrized<V: InnerProductVector + Send + Sync>(
    ws: &[V],
    ms: &[V],
) -> Vec<f64> {
    let m = ws.len();
    let n_coeff = ws.first().map_or(0, |w| w.num_coeffs());
    let mut t = if m >= 2 && par::par_driver_fork(m.saturating_mul(n_coeff)) {
        (0..m * m)
            .into_par_iter()
            .map(|ij| {
                let _nest = par::NestGuard::new();
                ws[ij / m].dot(&ms[ij % m]).re_s()
            })
            .collect::<Vec<f64>>()
    } else {
        let mut t = vec![0.0f64; m * m];
        for i in 0..m {
            for j in 0..m {
                t[i * m + j] = ws[i].dot(&ms[j]).re_s();
            }
        }
        t
    };
    for i in 0..m {
        for j in (i + 1)..m {
            let s = 0.5 * (t[i * m + j] + t[j * m + i]);
            t[i * m + j] = s;
            t[j * m + i] = s;
        }
    }
    t
}

pub(crate) fn gram_matrix_symmetrized<V: InnerProductVector + Send + Sync>(
    ws: &[V],
    vs: &[V],
) -> Vec<f64> {
    rr_matrix_symmetrized(ws, vs)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w_reorthonormalize_triple<V: InnerProductVector>(
    vj: &mut V,
    wj: &mut V,
    mj: &mut V,
    vs: &[V],
    ws: &[V],
    ms: &[V],
    scale: f64,
    diag: &mut SympDiagnostics,
) -> bool {
    let gn_before = vj.norm();
    if gn_before == 0.0 {
        diag.drops_triple += 1;
        return false;
    }
    for _ in 0..2 {
        for ((vi, wi), mi) in vs.iter().zip(ws.iter()).zip(ms.iter()) {
            let c = wi.dot(vj).re_s();
            let sc = V::Scalar::from_f64(-c);
            vj.axpy(sc, vi);
            wj.axpy(sc, wi);
            mj.axpy(sc, mi);
        }
    }
    let gn_after = vj.norm();
    if gn_after <= 1e-10 * gn_before {
        diag.drops_triple += 1;
        return false;
    }
    let beta2 = vj.dot(wj).re_s();
    if beta2 <= 0.0 {
        diag.drops_triple += 1;
        return false;
    }
    let beta = beta2.sqrt();
    if beta <= 1e3 * f64::EPSILON * scale.max(f64::MIN_POSITIVE) {
        diag.drops_triple += 1;
        return false;
    }
    let s = V::Scalar::from_f64(1.0 / beta);
    vj.scale(s);
    wj.scale(s);
    mj.scale(s);
    true
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn omega_lobp4dcg<V, R, W, D, Rp, P>(
    mut r_apply: R,
    mut w_apply: W,
    mut deflate: D,
    mut reproject: Rp,
    mut precond: Option<P>,
    v0s: &[V],
    k: usize,
    space_dim: usize,
    opts: &SympLobpcgOpts,
) -> Result<(Vec<f64>, SymplecticLowestInfo), SymplecticLowestError>
where
    V: InnerProductVector + Send + Sync,
    R: FnMut(&V) -> V,
    W: FnMut(&V) -> V,
    D: FnMut(&mut V),
    Rp: FnMut(&mut V),
    P: FnMut(&mut V, f64),
{
    assert!(k >= 1, "omega_lobp4dcg: k must be ≥ 1");
    assert!(space_dim >= 1, "omega_lobp4dcg: empty iteration space");
    assert!(!v0s.is_empty(), "omega_lobp4dcg: empty start block");
    assert!(
        v0s.len() >= k,
        "omega_lobp4dcg: block size must be ≥ k (shifting deflation is not implemented)"
    );
    let max_basis = opts.max_basis.max(k);
    let mut applies = 0usize;
    let mut total_built = 0usize;
    let mut diag = SympDiagnostics::default();

    let mut vs: Vec<V> = Vec::new();
    let mut ws: Vec<V> = Vec::new();
    let mut ms: Vec<V> = Vec::new();
    let omega_trace = crate::knobs::OMEGA_TRACE.get();

    for v0 in v0s {
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
        )? {
            total_built += 1;
        }
    }
    assert!(
        !vs.is_empty(),
        "omega_lobp4dcg: every seed collapsed under deflation — caller bug"
    );
    while ms.len() < vs.len() {
        let mut m_img = r_apply(&ws[ms.len()]);
        applies += 1;
        reproject(&mut m_img);
        ms.push(m_img);
    }
    let block = vs.len();
    let mut head = vs.len();

    let mut evals: Vec<f64>;
    let mut residuals: Vec<f64>;
    let mut ritz: Vec<Option<V>>;
    let converged;
    let mut outer = 0usize;

    loop {
        let m = vs.len();
        diag.outer_iterations += 1;
        if opts.instrument {
            let g = gram_matrix_symmetrized(&ws, &vs);
            let (gev, _) = dense_self_adjoint_smallest::<f64>(&g, m, 0);
            diag.gram_min_eig.push(gev[0]);
        }
        let t = rr_matrix_symmetrized(&ws, &ms);
        let want = block.min(m).max(k.min(m));
        let (ev, evecs) = dense_self_adjoint_smallest::<f64>(&t, m, want);
        evals = ev;
        let scale = evals
            .iter()
            .take(k.min(m))
            .fold(0.0f64, |acc, &e| acc.max(e.abs()))
            .max(f64::MIN_POSITIVE);

        let nv = block.min(m).min(evecs.len());
        let mut new_vs = recombine(&vs, &evecs, nv);
        let mut new_ws = recombine(&ws, &evecs, nv);
        let mut new_ms = recombine(&ms, &evecs, nv);

        let nwant = k.min(nv);
        residuals = Vec::with_capacity(nwant);
        ritz = Vec::with_capacity(nwant);
        let mut res_vecs: Vec<V> = Vec::with_capacity(nwant);
        for j in 0..nwant {
            let mut r = new_ms[j].clone();
            r.axpy(V::Scalar::from_f64(-evals[j]), &new_vs[j]);
            residuals.push(r.norm());
            res_vecs.push(r);
            ritz.push((evals[j] < 0.0).then(|| new_vs[j].clone()));
        }
        if omega_trace {
            let rmax = residuals.iter().fold(0.0f64, |a, &r| a.max(r));
            eprintln!(
                "  lobp4dcg outer={outer} m={m} built={total_built} θ0={:.6e} maxres={rmax:.3e}",
                evals[0]
            );
        }
        if nv >= k && residuals.iter().all(|&r| r <= opts.tol * scale) {
            converged = true;
            break;
        }
        if m >= space_dim {
            converged = true;
            break;
        }
        if total_built >= max_basis || outer >= opts.max_iter {
            converged = false;
            break;
        }

        let p_block = match opts.basis {
            BasisSelection::Legacy => (head < m).then(|| {
                let y_tail: Vec<Vec<f64>> = evecs
                    .iter()
                    .take(nv)
                    .map(|yj| yj[head..m].to_vec())
                    .collect();
                (
                    recombine(&vs[head..m], &y_tail, nv),
                    recombine(&ws[head..m], &y_tail, nv),
                    recombine(&ms[head..m], &y_tail, nv),
                )
            }),
        };

        for j in 0..nv {
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
        if let Some((pv, pw, pm)) = p_block {
            match opts.basis {
                BasisSelection::Legacy => {
                    for ((mut vj, mut wj), mut mj) in pv.into_iter().zip(pw).zip(pm) {
                        if new_vs.len() >= space_dim {
                            break;
                        }
                        if w_reorthonormalize_triple(
                            &mut vj, &mut wj, &mut mj, &new_vs, &new_ws, &new_ms, scale, &mut diag,
                        ) {
                            new_vs.push(vj);
                            new_ws.push(wj);
                            new_ms.push(mj);
                        }
                    }
                }
            }
        }
        vs = new_vs;
        ws = new_ws;
        ms = new_ms;
        head = nv;

        let use_precond = outer >= opts.nwarm;
        let mut accepted_this_round = 0usize;
        for (j, mut r) in res_vecs.into_iter().enumerate() {
            if vs.len() >= space_dim || total_built >= max_basis {
                break;
            }
            if residuals[j] <= opts.tol * scale {
                continue;
            }
            if use_precond
                && precond.is_some()
                && survival_guard_trips(&diag.survival, block, opts.guard_survival)
            {
                precond = None;
                diag.guard_tripped = Some(diag.survival.len());
            }
            if use_precond && let Some(p) = precond.as_mut() {
                p(&mut r, evals[j]);
            }
            let den = r.norm() * vs[j].norm();
            diag.collapse_angle.push(if den > 0.0 {
                r.dot(&vs[j]).abs_s() / den
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
            )? {
                total_built += 1;
                accepted_this_round += 1;
                let mut m_img = r_apply(&ws[ws.len() - 1]);
                applies += 1;
                reproject(&mut m_img);
                ms.push(m_img);
            }
        }
        if accepted_this_round == 0 {
            converged = residuals.iter().all(|&r| r <= opts.tol * scale) && nv >= k;
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
        |j| ritz.get(j).and_then(|o| o.clone()),
        &mut r_apply,
        &mut w_apply,
        &mut deflate,
        &mut reproject,
    );
    applies += cert_applies;
    residuals.truncate(omega2.len());
    let nret = omega2.len();
    Ok((
        omega2,
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
