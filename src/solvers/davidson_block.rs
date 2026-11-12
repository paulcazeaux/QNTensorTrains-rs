// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Block Davidson — the `nroots` lowest Ritz pairs of one local effective Hamiltonian, for the
//! multi-state (state-average / state-specific) sweeps. Layer-free: written against
//! [`InnerProductVector`] + `Clone` alone, so the SU(2) two-site block reaches it through the same
//! trait impls as the abelian one, with the centerweighted metric carried by `dot`/`norm`.

use super::InnerProductVector;
use super::davidson::{DAVIDSON_MAXSUBSPACE, DAVIDSON_NWARM, DAVIDSON_RESTART_KEEP};
use crate::core::factor::dense_self_adjoint_smallest;
use crate::scalar::Scalar;

/// Block-Davidson tunables. `tol`, `maxiter`, `nwarm`, `theta_tol` and `verbosity` carry exactly
/// the meaning they have in [`crate::solvers::davidson::DavidsonOpts`]; `maxsubspace` and
/// `restart_keep` are their block generalizations (see [`Self::for_roots`]).
#[derive(Clone, Copy, Debug)]
pub struct BlockDavidsonOpts {
    /// Absolute residual tolerance, per root (no scale factor) — the single-root rule.
    pub tol: f64,
    /// Subspace size at which a thick restart collapses the basis. Must be `≥ 2·nroots`: the
    /// collapse keeps the `nroots` Ritz vectors and must still admit one correction each.
    pub maxsubspace: usize,
    /// Iteration budget (one Rayleigh–Ritz and up to `nroots` applies each).
    pub maxiter: usize,
    /// Unpreconditioned warm-up iterations before the diagonal preconditioner engages.
    pub nwarm: usize,
    /// Optional per-root Ritz-value early exit, `|θᵢ − θᵢ_prev| ≤ θtol`. `None` is the plain
    /// residual rule; the two-site MALS dispatch opts in with `inner_tol · THETA_TOL_FACTOR`.
    pub theta_tol: Option<f64>,
    /// Ritz columns kept across a thick restart. Must satisfy `nroots ≤ restart_keep < maxsubspace`
    /// (a collapse below `nroots` loses a root; one at `maxsubspace` frees no slot).
    pub restart_keep: usize,
    pub verbosity: u8,
}

impl BlockDavidsonOpts {
    /// Defaults for an `nroots`-root solve: `maxsubspace = max(DAVIDSON_MAXSUBSPACE, 4·nroots)`,
    /// `restart_keep = DAVIDSON_RESTART_KEEP` clamped into `[nroots, maxsubspace − 1]`, everything
    /// else the single-root defaults. At `nroots = 1` this reproduces
    /// [`crate::solvers::davidson::DavidsonOpts::default`] field for field — the precondition that
    /// makes the bitwise single-root parity meaningful at all (asserted by `tms_a1`).
    pub fn for_roots(nroots: usize) -> Self {
        assert!(
            nroots >= 1,
            "BlockDavidsonOpts::for_roots: nroots = {nroots} must be ≥ 1"
        );
        let maxsubspace = DAVIDSON_MAXSUBSPACE.max(4 * nroots);
        BlockDavidsonOpts {
            tol: 1e-8,
            maxsubspace,
            maxiter: 100,
            nwarm: DAVIDSON_NWARM,
            theta_tol: None,
            restart_keep: DAVIDSON_RESTART_KEEP.clamp(nroots, maxsubspace - 1),
            verbosity: 0,
        }
    }
}

/// Convergence/iteration diagnostics returned alongside the block of Ritz pairs. `normres` is
/// per-root, in the same (ascending-θ) order as the returned pairs, and is the PROJECTED residual
/// in the [`davidson_block_projected`] variant.
#[derive(Clone, Debug)]
pub struct BlockDavidsonInfo {
    /// Every requested root came back and converged.
    pub converged: bool,
    /// Roots converged at the last Rayleigh–Ritz.
    pub nconv: usize,
    pub normres: Vec<f64>,
    pub numops: usize,
    pub niter: usize,
    /// Guesses dropped at seeding: dependent on an earlier seed, zero, or annihilated by
    /// `project`.
    pub dropped_guesses: usize,
}

/// The `nroots` smallest-algebraic Ritz pairs of the local effective Hamiltonian `op`
/// (`v ↦ H_eff·v`) in the package inner product, seeded from `v0s` in order. `applyprec(t, θ)`
/// overwrites `t` in place with the preconditioned correction at that root's own Ritz value.
/// Returns `(θs ascending, us orthonormal in `V::dot`, info)` — exactly `nroots` pairs unless the
/// subspace stagnated before it could host them (the documented short return, with
/// `info.converged == false`).
pub fn davidson_block<V, Op, Prec>(
    op: &mut Op,
    v0s: &[V],
    applyprec: &mut Prec,
    nroots: usize,
    opts: &BlockDavidsonOpts,
) -> (Vec<f64>, Vec<V>, BlockDavidsonInfo)
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
    Prec: FnMut(&mut V, f64),
{
    let mut op_batch = |vs: &[V]| -> Vec<V> { vs.iter().map(&mut *op).collect() };
    davidson_block_impl::<V, _, Prec, fn(&mut V)>(&mut op_batch, v0s, applyprec, None, nroots, opts)
}

/// [`davidson_block`] restricted to `range(project)`: `project` is applied to every guess, every
/// correction and every residual, so the reported `normres` and the convergence test are the
/// PROJECTED residuals (module header — the full residual is floored by whatever was projected
/// out, and testing it would never converge). `project` must be an orthogonal projector in the
/// package metric.
pub fn davidson_block_projected<V, Op, Prec, Proj>(
    op: &mut Op,
    v0s: &[V],
    applyprec: &mut Prec,
    project: &mut Proj,
    nroots: usize,
    opts: &BlockDavidsonOpts,
) -> (Vec<f64>, Vec<V>, BlockDavidsonInfo)
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
    Prec: FnMut(&mut V, f64),
    Proj: FnMut(&mut V),
{
    let mut op_batch = |vs: &[V]| -> Vec<V> { vs.iter().map(&mut *op).collect() };
    davidson_block_impl(&mut op_batch, v0s, applyprec, Some(project), nroots, opts)
}

/// [`davidson_block_projected`] whose images come from an apply-BATCH closure: `op_batch(vs)`
/// returns `H·vs[k]` for every `k`, in INPUT ORDER. The loop calls it once per batch (all
/// surviving seeds at seeding, all accepted corrections each iteration) after the batch is
/// orthogonalized, so a caller with an expensive `op` can run the batch's applies CONCURRENTLY —
/// which is the point of this entry, and why the concurrency lives in the caller and never in this
/// module (module header, "Batched apply").
pub fn davidson_block_projected_batched<V, AB, Prec, Proj>(
    op_batch: &mut AB,
    v0s: &[V],
    applyprec: &mut Prec,
    project: &mut Proj,
    nroots: usize,
    opts: &BlockDavidsonOpts,
) -> (Vec<f64>, Vec<V>, BlockDavidsonInfo)
where
    V: InnerProductVector,
    AB: FnMut(&[V]) -> Vec<V>,
    Prec: FnMut(&mut V, f64),
    Proj: FnMut(&mut V),
{
    davidson_block_impl(op_batch, v0s, applyprec, Some(project), nroots, opts)
}

fn extend_hermitian<V: InnerProductVector>(
    s: &[V::Scalar],
    m: usize,
    vv: &[V],
    t: &V,
    ht: &V,
) -> Vec<V::Scalar> {
    let mp = m + 1;
    let mut snew = vec![V::Scalar::zero(); mp * mp];
    for i in 0..m {
        for j in 0..m {
            snew[i * mp + j] = s[i * m + j];
        }
    }
    for i in 0..m {
        let sij = vv[i].dot(ht);
        snew[i * mp + m] = sij;
        snew[m * mp + i] = sij.conj_s();
    }
    snew[m * mp + m] = t.dot(ht);
    snew
}

#[allow(clippy::too_many_arguments)]
fn rayleigh_ritz<V, Proj>(
    s: &[V::Scalar],
    m: usize,
    vv: &[V],
    ww: &[V],
    nroots: usize,
    want: usize,
    project: &mut Option<&mut Proj>,
    us: &mut [V],
    hus: &mut [V],
    rs: &mut [V],
    thetas: &mut [f64],
    normres: &mut [f64],
) -> (Vec<f64>, Vec<Vec<V::Scalar>>, usize)
where
    V: InnerProductVector,
    Proj: FnMut(&mut V),
{
    let (evals, evecs) = dense_self_adjoint_smallest(s, m, want);
    let nr = nroots.min(m);
    for i in 0..nr {
        thetas[i] = evals[i];
        let y = &evecs[i];
        us[i].clone_from(&vv[0]);
        us[i].scale(y[0]);
        for j in 1..m {
            us[i].axpy(y[j], &vv[j]);
        }
        hus[i].clone_from(&ww[0]);
        hus[i].scale(y[0]);
        for j in 1..m {
            hus[i].axpy(y[j], &ww[j]);
        }
        rs[i].clone_from(&hus[i]);
        rs[i].axpy(V::Scalar::from_f64(-thetas[i]), &us[i]);
        if let Some(p) = project.as_mut() {
            p(&mut rs[i]);
        }
        normres[i] = rs[i].norm();
    }
    (evals, evecs, nr)
}

fn davidson_block_impl<V, AB, Prec, Proj>(
    op_batch: &mut AB,
    v0s: &[V],
    applyprec: &mut Prec,
    mut project: Option<&mut Proj>,
    nroots: usize,
    opts: &BlockDavidsonOpts,
) -> (Vec<f64>, Vec<V>, BlockDavidsonInfo)
where
    V: InnerProductVector,
    AB: FnMut(&[V]) -> Vec<V>,
    Prec: FnMut(&mut V, f64),
    Proj: FnMut(&mut V),
{
    assert!(nroots >= 1, "davidson_block: nroots = {nroots} must be ≥ 1");
    assert!(
        !v0s.is_empty(),
        "davidson_block: at least one guess vector is required (the guess list is empty; \
         nroots = {nroots})"
    );
    assert!(
        opts.maxsubspace >= 2 * nroots,
        "davidson_block: maxsubspace = {} must be ≥ 2·nroots = {} — the thick restart keeps the \
         nroots Ritz vectors and must still admit one correction per root",
        opts.maxsubspace,
        2 * nroots
    );
    assert!(
        nroots <= opts.restart_keep && opts.restart_keep < opts.maxsubspace,
        "davidson_block: restart_keep = {} must satisfy nroots = {nroots} ≤ restart_keep < \
         maxsubspace = {} (a collapse below nroots loses a root; one at maxsubspace frees no slot)",
        opts.restart_keep,
        opts.maxsubspace
    );

    let mut vv: Vec<V> = Vec::with_capacity(opts.maxsubspace);
    let mut dropped_guesses = 0usize;
    for g in v0s {
        let mut t = g.clone();
        let nt0 = t.norm();
        assert!(
            nt0.is_finite(),
            "davidson_block: non-finite guess norm (caller bug)"
        );
        if let Some(p) = project.as_mut() {
            p(&mut t);
        }
        for _ in 0..2 {
            for vj in vv.iter() {
                let c = vj.dot(&t);
                t.axpy(-c, vj);
            }
        }
        let nt = if vv.is_empty() && project.is_none() {
            nt0
        } else {
            t.norm()
        };
        if nt <= f64::EPSILON.sqrt() * nt0 {
            dropped_guesses += 1;
            continue;
        }
        t.scale(V::Scalar::from_f64(1.0 / nt));
        vv.push(t);
    }
    assert!(
        !vv.is_empty(),
        "davidson_block: every guess was dropped at seeding — all {} guesses were zero, dependent \
         or annihilated by the projector",
        v0s.len()
    );
    assert!(
        vv[0].num_coeffs() >= nroots,
        "davidson_block: local dimension {} < nroots = {nroots} — the local space cannot host that \
         many orthonormal states (enrich the ranks)",
        vv[0].num_coeffs()
    );

    let hvs = op_batch(&vv);
    assert_eq!(
        hvs.len(),
        vv.len(),
        "davidson_block: the apply batch returned {} images for {} vectors (caller bug)",
        hvs.len(),
        vv.len()
    );
    let mut numops = 1usize;
    let mut ww: Vec<V> = Vec::with_capacity(opts.maxsubspace);
    let mut hv_iter = hvs.into_iter();
    let hv1 = hv_iter.next().expect("davidson_block: non-empty basis");
    let s00 = vv[0].dot(&hv1);
    let mut s: Vec<V::Scalar> = vec![s00];
    let mut m = 1usize;
    ww.push(hv1);
    for (i, ht) in hv_iter.enumerate() {
        let i = i + 1;
        numops += 1;
        s = extend_hermitian(&s, m, &vv[..i], &vv[i], &ht);
        ww.push(ht);
        m += 1;
    }

    let mut us: Vec<V> = (0..nroots).map(|_| vv[0].clone()).collect();
    let mut hus: Vec<V> = (0..nroots).map(|_| ww[0].clone()).collect();
    let mut rs: Vec<V> = (0..nroots).map(|_| ww[0].clone()).collect();

    let mut thetas = vec![f64::INFINITY; nroots];
    let mut theta_prev = vec![f64::INFINITY; nroots];
    let mut normres = vec![f64::INFINITY; nroots];
    let mut conv = vec![false; nroots];
    let mut nr = nroots.min(m);
    let mut nconv = 0usize;
    let mut converged = false;
    let mut it = 0usize;

    while it < opts.maxiter {
        it += 1;

        let want = opts.restart_keep.min(m);
        let (evals, evecs, nr_now) = rayleigh_ritz(
            &s,
            m,
            &vv,
            &ww,
            nroots,
            want,
            &mut project,
            &mut us,
            &mut hus,
            &mut rs,
            &mut thetas,
            &mut normres,
        );
        nr = nr_now;

        nconv = 0;
        for i in 0..nr {
            let dtheta = (thetas[i] - theta_prev[i]).abs();
            theta_prev[i] = thetas[i];
            conv[i] = normres[i] <= opts.tol || opts.theta_tol.is_some_and(|ttol| dtheta <= ttol);
            if conv[i] {
                nconv += 1;
            }
        }
        for c in conv.iter_mut().skip(nr) {
            *c = false;
        }
        if opts.verbosity > 0 {
            println!(
                "  davidson_block it={it}  dim={m}  nconv={nconv}/{nroots}  θ={:?}  ‖r‖={:?}",
                &thetas[..nr],
                &normres[..nr]
            );
        }
        if nr == nroots && nconv == nroots {
            converged = true;
            break;
        }

        let mut batch: Vec<(V, f64)> = Vec::with_capacity(nr);
        for i in 0..nr {
            if conv[i] {
                continue;
            }
            let mut t = rs[i].clone();
            if it > opts.nwarm {
                applyprec(&mut t, thetas[i]);
            }
            let nt0 = t.norm();
            if let Some(p) = project.as_mut() {
                p(&mut t);
            }
            batch.push((t, nt0));
        }

        if vv.len() + batch.len() > opts.maxsubspace {
            debug_assert!(
                nr <= opts.maxsubspace - batch.len(),
                "davidson_block: restart clamp bounds crossed (nr = {nr}, maxsubspace = {}, \
                 corrections = {})",
                opts.maxsubspace,
                batch.len()
            );
            let keep = opts.restart_keep.clamp(nr, opts.maxsubspace - batch.len());
            let mut vk: Vec<V> = Vec::with_capacity(opts.maxsubspace);
            let mut wk: Vec<V> = Vec::with_capacity(opts.maxsubspace);
            for i in 0..nr {
                vk.push(us[i].clone());
                wk.push(hus[i].clone());
            }
            for yj in evecs.iter().take(keep).skip(nr) {
                let mut uj = vv[0].clone();
                uj.scale(yj[0]);
                for i in 1..vv.len() {
                    uj.axpy(yj[i], &vv[i]);
                }
                let mut huj = ww[0].clone();
                huj.scale(yj[0]);
                for i in 1..ww.len() {
                    huj.axpy(yj[i], &ww[i]);
                }
                vk.push(uj);
                wk.push(huj);
            }
            if keep == 1 {
                s = vec![vk[0].dot(&wk[0])];
            } else {
                let mut snew = vec![V::Scalar::zero(); keep * keep];
                for (d, sd) in snew.iter_mut().step_by(keep + 1).enumerate() {
                    *sd = V::Scalar::from_f64(evals[d]);
                }
                s = snew;
            }
            vv = vk;
            ww = wk;
            m = keep;
        }

        let m0 = m;
        for (mut t, nt0) in batch {
            for _ in 0..2 {
                for vj in vv.iter() {
                    let c = vj.dot(&t);
                    t.axpy(-c, vj);
                }
            }
            let nt = t.norm();
            if nt <= f64::EPSILON.sqrt() * nt0 {
                continue;
            }
            t.scale(V::Scalar::from_f64(1.0 / nt));
            vv.push(t);
        }
        let accepted = vv.len() - m0;
        if accepted > 0 {
            let hts = op_batch(&vv[m0..]);
            assert_eq!(
                hts.len(),
                accepted,
                "davidson_block: the apply batch returned {} images for {accepted} vectors \
                 (caller bug)",
                hts.len()
            );
            for ht in hts {
                s = extend_hermitian(&s, m, &vv[..m], &vv[m], &ht);
                ww.push(ht);
                m += 1;
                numops += 1;
            }
        }
        if accepted == 0 {
            break;
        }
    }

    if nr < nroots.min(m) {
        let want = opts.restart_keep.min(m);
        let (_evals, _evecs, nr_now) = rayleigh_ritz(
            &s,
            m,
            &vv,
            &ww,
            nroots,
            want,
            &mut project,
            &mut us,
            &mut hus,
            &mut rs,
            &mut thetas,
            &mut normres,
        );
        nr = nr_now;
        nconv = 0;
        for i in 0..nr {
            let dtheta = (thetas[i] - theta_prev[i]).abs();
            theta_prev[i] = thetas[i];
            conv[i] = normres[i] <= opts.tol || opts.theta_tol.is_some_and(|ttol| dtheta <= ttol);
            if conv[i] {
                nconv += 1;
            }
        }
        converged = nr == nroots && nconv == nroots;
    }

    us.truncate(nr);
    (
        thetas[..nr].to_vec(),
        us,
        BlockDavidsonInfo {
            converged,
            nconv,
            normres: normres[..nr].to_vec(),
            numops,
            niter: it,
            dropped_guesses,
        },
    )
}

/// Greedy root homing: assign each previous root `prev[i]` the distinct new root `us[j]` it
/// overlaps most in the package metric, largest `|⟨prev, us⟩|` first. Returns `(perm, gram)` with
/// `us[perm[i]]` the continuation of `prev[i]` and `gram[i][j] = ⟨prev[i], us[j]⟩` (rows over
/// `prev`, columns over `us`) — the full cross Gram, so a caller can judge how clean the assignment
/// was without recomputing a single `dot`. Columns left unassigned (`us.len() > prev.len()`) are
/// new roots.
pub fn home_roots<V: InnerProductVector>(
    prev: &[V],
    us: &[V],
) -> (Vec<usize>, Vec<Vec<V::Scalar>>) {
    assert!(
        !prev.is_empty(),
        "home_roots: prev must hold at least one previous root (there is nothing to home)"
    );
    assert!(
        us.len() >= prev.len(),
        "home_roots: need us.len() = {} ≥ prev.len() = {} — fewer new roots than previous ones \
         admits no injective assignment",
        us.len(),
        prev.len()
    );
    let np = prev.len();
    let nu = us.len();
    let gram: Vec<Vec<V::Scalar>> = prev
        .iter()
        .map(|p| us.iter().map(|u| p.dot(u)).collect())
        .collect();

    let mut perm = vec![usize::MAX; np];
    let mut row_free = vec![true; np];
    let mut col_free = vec![true; nu];
    for _ in 0..np {
        let mut best = -1.0f64;
        let mut bi = usize::MAX;
        let mut bj = usize::MAX;
        for j in 0..nu {
            if !col_free[j] {
                continue;
            }
            for i in 0..np {
                if !row_free[i] {
                    continue;
                }
                let a = gram[i][j].abs_s();
                if a > best {
                    best = a;
                    bi = i;
                    bj = j;
                }
            }
        }
        debug_assert!(
            bi < np && bj < nu,
            "home_roots: no free (row, column) pair left"
        );
        perm[bi] = bj;
        row_free[bi] = false;
        col_free[bj] = false;
    }
    (perm, gram)
}
