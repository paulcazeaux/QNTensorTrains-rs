// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! In-house Davidson eigensolver — the diagonally-preconditioned block Krylov solver ALS and MALS
//! use for the local one-site and two-site ground-state searches.

use super::InnerProductVector;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::scalar::Scalar;

/// Unpreconditioned warm-up iteration count before the diagonal preconditioner engages: the first
/// `nwarm` correction steps take the plain residual, seating the subspace in the ground-state basin
/// before the diagonal preconditioner (which can amplify an interior eigenvector from a cold start)
/// is engaged.
pub const DAVIDSON_NWARM: usize = 2;

/// Subspace size at which a thick restart collapses the basis.
pub const DAVIDSON_MAXSUBSPACE: usize = 8;

/// Ritz columns kept across a thick-restart collapse. `keep == 1` (single-vector restart) is the
/// default; `keep > 1` is exercised by `t2_davidson_thick_restart_keep2` (the `sweep_solvers` binary).
pub const DAVIDSON_RESTART_KEEP: usize = 1;

/// Convergence/iteration diagnostics returned alongside the Ritz pair.
#[derive(Clone, Copy, Debug)]
pub struct DavidsonInfo {
    pub converged: bool,
    pub normres: f64,
    pub numops: usize,
    pub niter: usize,
}

/// Davidson tunables.
#[derive(Clone, Copy, Debug)]
pub struct DavidsonOpts {
    pub tol: f64,
    pub maxsubspace: usize,
    pub maxiter: usize,
    pub nwarm: usize,
    pub restart_keep: usize,
    pub verbosity: u8,
    /// Optional Ritz-value (energy) early-exit: when `Some(θtol)`, also converge once the Ritz
    /// value has stabilized — `|θ − θ_prev| ≤ θtol` between consecutive iterations — even if the
    /// residual has not yet drained below [`Self::tol`]. `None` (the default) is the plain
    /// residual-only stopping rule. The energy exit is enabled ONLY by the two-site MALS driver,
    /// to trim the
    /// FP-order-sensitive tail iteration where θ has already converged quadratically but the faer-
    /// factorized residual lingers just above `tol`. Gated so it never fires on the first iteration (`θ_prev = ∞`), preserving the
    /// direct-Davidson iteration-count tests, which leave this `None`.
    pub theta_tol: Option<f64>,
    /// Optional early exit on CLEAR indefiniteness: when `Some(s)`, break out (deliberately
    /// UNCONVERGED) as soon as the Ritz value drops below `s`. Enabled only by the PD probes
    /// (`s = −margin·scale`): a Ritz value that far below zero will be certified by the caller's
    /// fresh-apply variational witness — a negative fresh Rayleigh quotient needs no convergence
    /// — so burning the remaining budget adds nothing. `None` (the default) everywhere else.
    pub stop_below: Option<f64>,
}

impl Default for DavidsonOpts {
    fn default() -> Self {
        DavidsonOpts {
            tol: 1e-8,
            maxsubspace: DAVIDSON_MAXSUBSPACE,
            maxiter: 100,
            nwarm: DAVIDSON_NWARM,
            restart_keep: DAVIDSON_RESTART_KEEP,
            verbosity: 0,
            theta_tol: None,
            stop_below: None,
        }
    }
}

/// Smallest-algebraic eigenpair of the local effective Hamiltonian `op` (`v ↦ H_eff·v`),
/// warm-started from `v0`, in the package (metric-carrying) inner product `V::dot`/`V::norm`.
/// `applyprec(t, θ)` overwrites the residual `t` in place with the preconditioned correction `(D −
/// θ)⁻¹t`; the Olsen correction is not implemented. One `op` application per
/// iteration. Returns `(θ, u, info)`.
pub fn davidson<V, Op, Prec>(
    op: &mut Op,
    v0: &V,
    applyprec: &mut Prec,
    opts: &DavidsonOpts,
) -> (f64, V, DavidsonInfo)
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
    Prec: FnMut(&mut V, f64),
{
    assert!(opts.maxsubspace >= 2, "davidson: maxsubspace must be ≥ 2");

    let mut v1 = v0.clone();
    let n0 = v1.norm();
    assert!(n0 > 0.0, "davidson: zero-norm warm start v0 (caller bug)");
    v1.scale(V::Scalar::from_f64(1.0 / n0));

    let hv1 = op(&v1);
    let mut numops = 1usize;

    let s00 = v1.dot(&hv1);
    let mut s: Vec<V::Scalar> = vec![s00];
    let mut m = 1usize;
    let mut theta = s00.re_s();

    let mut u = v1.clone();
    let mut hu = hv1.clone();
    let mut r = hv1.clone();

    let mut vv: Vec<V> = vec![v1];
    let mut ww: Vec<V> = vec![hv1];

    let mut normres = f64::INFINITY;
    let mut converged = false;
    let mut it = 0usize;
    let mut theta_prev = f64::INFINITY;

    while it < opts.maxiter {
        it += 1;

        let want = opts.restart_keep.max(1).min(m);
        let (evals, evecs) = dense_self_adjoint_smallest(&s, m, want);
        theta = evals[0];
        let y = &evecs[0];

        u.clone_from(&vv[0]);
        u.scale(y[0]);
        for i in 1..m {
            u.axpy(y[i], &vv[i]);
        }
        hu.clone_from(&ww[0]);
        hu.scale(y[0]);
        for i in 1..m {
            hu.axpy(y[i], &ww[i]);
        }

        r.clone_from(&hu);
        r.axpy(V::Scalar::from_f64(-theta), &u);

        normres = r.norm();
        let dtheta = (theta - theta_prev).abs();
        theta_prev = theta;
        if opts.verbosity > 0 {
            println!("  davidson it={it}  θ={theta}  ‖r‖={normres}  Δθ={dtheta}  dim={m}");
        }
        if normres <= opts.tol || opts.theta_tol.is_some_and(|ttol| dtheta <= ttol) {
            converged = true;
            break;
        }
        if opts.stop_below.is_some_and(|s| theta < s) {
            break;
        }

        let mut t = r.clone();
        if it > opts.nwarm {
            applyprec(&mut t, theta);
        }

        if vv.len() >= opts.maxsubspace {
            let keep = opts.restart_keep.clamp(1, vv.len());
            if keep == 1 {
                let su = u.dot(&hu);
                vv = vec![u.clone()];
                ww = vec![hu.clone()];
                s = vec![su];
                m = 1;
            } else {
                let mut vk = vec![u.clone()];
                let mut wk = vec![hu.clone()];
                for yj in evecs.iter().take(keep).skip(1) {
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
                let mut snew = vec![V::Scalar::zero(); keep * keep];
                for (d, sd) in snew.iter_mut().step_by(keep + 1).enumerate() {
                    *sd = V::Scalar::from_f64(evals[d]);
                }
                vv = vk;
                ww = wk;
                s = snew;
                m = keep;
            }
        }

        let nt0 = t.norm();
        for _ in 0..2 {
            for vj in vv.iter() {
                let c = vj.dot(&t);
                t.axpy(-c, vj);
            }
        }

        let nt = t.norm();
        if nt <= f64::EPSILON.sqrt() * nt0 {
            break;
        }
        t.scale(V::Scalar::from_f64(1.0 / nt));
        let ht = op(&t);
        numops += 1;

        let mp = m + 1;
        let mut snew = vec![V::Scalar::zero(); mp * mp];
        for i in 0..m {
            for j in 0..m {
                snew[i * mp + j] = s[i * m + j];
            }
        }
        for i in 0..m {
            let sij = vv[i].dot(&ht);
            snew[i * mp + m] = sij;
            snew[m * mp + i] = sij.conj_s();
        }
        snew[m * mp + m] = t.dot(&ht);
        vv.push(t);
        ww.push(ht);
        s = snew;
        m = mp;
    }

    (
        theta,
        u,
        DavidsonInfo {
            converged,
            normres,
            numops,
            niter: it,
        },
    )
}

/// In-place diagonal preconditioner `t_i ← t_i / (D_i − θ)`, with the standard Davidson
/// denominator safeguard `δ = 1e-8·max(1,|θ|)`. `d` is
/// the `H_eff` diagonal (same shape as `t`), read in the metric-free coefficient order.
pub fn precond<V: InnerProductVector>(t: &mut V, d: &V, theta: f64) {
    let delta = 1e-8_f64 * theta.abs().max(1.0);
    let th = V::Scalar::from_f64(theta);
    t.zip_coeffs_mut(d, |ti, di| {
        let mut den = di - th;
        let a = den.abs_s();
        if a < delta {
            den = if den == V::Scalar::zero() {
                V::Scalar::from_f64(delta)
            } else {
                den * V::Scalar::from_f64(delta / a)
            };
        }
        *ti *= den.recip();
    });
}

/// In-place FIXED, UNSHIFTED, SPD diagonal inverse: overwrite the squared-diagonal model `d` with
/// `T_i = ((Re d_i)_+ + c)⁻¹`, where `c = dmax/κ`, `dmax = (max_i (Re d_i)_+).max(f64::MIN_POSITIVE)`
/// and `κ = kappa` (auto `1e3` when `kappa` is not a positive finite number). Two passes over the
/// canonical coefficient order — deterministic, no reduction reordering.
pub fn precond_fixed_inverse<V: InnerProductVector>(d: &mut V, kappa: f64) {
    let kappa = if kappa.is_finite() && kappa > 0.0 {
        kappa
    } else {
        1e3
    };
    let mut dmax = 0.0f64;
    d.for_each_coeff_mut(|c| dmax = dmax.max(c.re_s().max(0.0)));
    let c = (dmax.max(f64::MIN_POSITIVE) / kappa).max(f64::MIN_POSITIVE);
    d.for_each_coeff_mut(|di| {
        *di = V::Scalar::from_f64(1.0 / (di.re_s().max(0.0) + c));
    });
}

/// Brute-force exact diagonal of `H_eff`: apply `op` to every canonical unit basis vector and read
/// back the matching coefficient. `O(#coeffs)` matvecs — a slow, obviously-correct oracle for
/// `effective_diagonal`.
pub fn effective_diagonal_bruteforce<V, Op>(op: &mut Op, template: &V) -> V
where
    V: InnerProductVector,
    Op: FnMut(&V) -> V,
{
    let mut d = template.clone();
    let mut e = template.clone();
    let n = e.num_coeffs();
    for idx in 0..n {
        e.for_each_coeff_mut(|c| *c = V::Scalar::zero());
        e.set_coeff(idx, V::Scalar::one());
        let r = op(&e);
        d.set_coeff(idx, r.coeff(idx));
    }
    d
}

#[cfg(test)]
mod phase5_tests {
    use super::{DavidsonOpts, davidson, precond};
    use crate::Family;
    use crate::core::SparseCore;
    use crate::qn::{QnSpace, RankGrid, Sector};
    use faer::prelude::ReborrowMut;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t2_davidson_converges_smallest_eig() {
        let space = QnSpace::new(1, 0, 0);
        let r = 8usize;
        let s0 = Sector::new(0, 0);
        let mut row_ranks = RankGrid::zeros(&space);
        row_ranks[s0] = r;
        let mut col_ranks = RankGrid::zeros(&space);
        col_ranks[s0] = 1;

        let mut hd = faer::Mat::<f64>::zeros(r, r);
        for i in 0..r {
            *hd.get_mut(i, i) = 1.0 + i as f64;
        }
        for i in 0..r {
            for j in (i + 1)..r {
                let v = 0.05 / (1.0 + (i as f64 - j as f64).abs());
                *hd.get_mut(i, j) = v;
                *hd.get_mut(j, i) = v;
            }
        }
        let eigs = hd
            .as_ref()
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .expect("dense self-adjoint eigendecomposition");
        let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);

        let mut d_core = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        {
            let mut b = d_core.block_mut(Family::Empty, s0).unwrap();
            for i in 0..r {
                *b.rb_mut().get_mut(i, 0) = *hd.get(i, i);
            }
        }

        let op = |a: &SparseCore<f64>| -> SparseCore<f64> {
            let ab = a.block(Family::Empty, s0).unwrap();
            let mut out = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
            {
                let mut ob = out.block_mut(Family::Empty, s0).unwrap();
                for i in 0..r {
                    let mut acc = 0.0f64;
                    for j in 0..r {
                        acc += *hd.get(i, j) * *ab.get(j, 0);
                    }
                    *ob.rb_mut().get_mut(i, 0) = acc;
                }
            }
            out
        };
        let mut op_mut = op;

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A09);
        let mut v0 = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        v0.fill_randn(&mut rng);

        let mut applyprec = |t: &mut SparseCore<f64>, theta: f64| {
            precond(t, &d_core, theta);
        };

        let opts = DavidsonOpts {
            tol: 1e-8,
            maxsubspace: 8,
            maxiter: 100,
            nwarm: 2,
            restart_keep: 1,
            verbosity: 0,
            theta_tol: None,
            stop_below: None,
        };
        let (lambda, v, info) = davidson(&mut op_mut, &v0, &mut applyprec, &opts);

        assert!(
            (lambda - min_eig).abs() < 1e-6,
            "davidson eigenvalue mismatch: {lambda} vs {min_eig}"
        );
        assert!(info.converged, "davidson did not converge");
        let hv = op_mut(&v);
        let mut resid = hv.clone();
        resid.axpy(-lambda, &v);
        assert!(
            resid.norm() < 1e-6,
            "davidson residual too large: {}",
            resid.norm()
        );
    }

    #[test]
    fn t2_davidson_thick_restart_keep2() {
        let space = QnSpace::new(1, 0, 0);
        let r = 8usize;
        let s0 = Sector::new(0, 0);
        let mut row_ranks = RankGrid::zeros(&space);
        row_ranks[s0] = r;
        let mut col_ranks = RankGrid::zeros(&space);
        col_ranks[s0] = 1;

        let mut hd = faer::Mat::<f64>::zeros(r, r);
        for i in 0..r {
            *hd.get_mut(i, i) = 1.0 + i as f64;
        }
        for i in 0..r {
            for j in (i + 1)..r {
                let v = 0.05 / (1.0 + (i as f64 - j as f64).abs());
                *hd.get_mut(i, j) = v;
                *hd.get_mut(j, i) = v;
            }
        }
        let min_eig = hd
            .as_ref()
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .expect("dense self-adjoint eigendecomposition")
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        let mut d_core = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        {
            let mut b = d_core.block_mut(Family::Empty, s0).unwrap();
            for i in 0..r {
                *b.rb_mut().get_mut(i, 0) = *hd.get(i, i);
            }
        }

        let mut op_mut = |a: &SparseCore<f64>| -> SparseCore<f64> {
            let ab = a.block(Family::Empty, s0).unwrap();
            let mut out = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
            {
                let mut ob = out.block_mut(Family::Empty, s0).unwrap();
                for i in 0..r {
                    let mut acc = 0.0f64;
                    for j in 0..r {
                        acc += *hd.get(i, j) * *ab.get(j, 0);
                    }
                    *ob.rb_mut().get_mut(i, 0) = acc;
                }
            }
            out
        };

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A0A);
        let mut v0 = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        v0.fill_randn(&mut rng);

        let mut applyprec = |t: &mut SparseCore<f64>, theta: f64| {
            precond(t, &d_core, theta);
        };

        let opts = DavidsonOpts {
            tol: 1e-8,
            maxsubspace: 4,
            maxiter: 300,
            nwarm: 2,
            restart_keep: 2,
            verbosity: 0,
            theta_tol: None,
            stop_below: None,
        };
        let (lambda, v, info) = davidson(&mut op_mut, &v0, &mut applyprec, &opts);

        assert!(
            (lambda - min_eig).abs() < 1e-6,
            "davidson (keep=2) eigenvalue mismatch: {lambda} vs {min_eig}"
        );
        assert!(info.converged, "davidson (keep=2) did not converge");
        let hv = op_mut(&v);
        let mut resid = hv.clone();
        resid.axpy(-lambda, &v);
        assert!(
            resid.norm() < 1e-6,
            "davidson (keep=2) residual too large: {}",
            resid.norm()
        );
    }

    #[test]
    fn t2_davidson_stagnation_returns_not_converged() {
        let space = QnSpace::new(1, 0, 0);
        let r = 2usize;
        let s0 = Sector::new(0, 0);
        let mut row_ranks = RankGrid::zeros(&space);
        row_ranks[s0] = r;
        let mut col_ranks = RankGrid::zeros(&space);
        col_ranks[s0] = 1;
        let diag = [1.0f64, 3.0];

        let mut d_core = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        {
            let mut b = d_core.block_mut(Family::Empty, s0).unwrap();
            for (i, &dv) in diag.iter().enumerate() {
                *b.rb_mut().get_mut(i, 0) = dv;
            }
        }
        let mut op_mut = |a: &SparseCore<f64>| -> SparseCore<f64> {
            let ab = a.block(Family::Empty, s0).unwrap();
            let mut out = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
            {
                let mut ob = out.block_mut(Family::Empty, s0).unwrap();
                for (i, &dv) in diag.iter().enumerate() {
                    *ob.rb_mut().get_mut(i, 0) = dv * *ab.get(i, 0);
                }
            }
            out
        };

        let mut v0 = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        {
            let mut b = v0.block_mut(Family::Empty, s0).unwrap();
            let inv_sqrt2 = 1.0 / 2.0f64.sqrt();
            for i in 0..r {
                *b.rb_mut().get_mut(i, 0) = inv_sqrt2;
            }
        }

        let mut applyprec = |t: &mut SparseCore<f64>, theta: f64| {
            precond(t, &d_core, theta);
        };
        let opts = DavidsonOpts {
            tol: 1e-8,
            maxsubspace: 8,
            maxiter: 100,
            nwarm: 0,
            restart_keep: 1,
            verbosity: 0,
            theta_tol: None,
            stop_below: None,
        };
        let (lambda, _v, info) = davidson(&mut op_mut, &v0, &mut applyprec, &opts);

        assert!(!info.converged, "stagnation must report converged=false");
        assert_eq!(info.niter, 1, "stagnation should break on iteration 1");
        assert_eq!(
            info.numops, 1,
            "no extra op application after the stagnation break"
        );
        assert!(
            (lambda - 2.0).abs() < 1e-12,
            "stagnation Ritz value should be the best-so-far (1+3)/2=2, got {lambda}"
        );
        assert!(
            info.normres > 0.5,
            "residual must be genuinely nonzero: {}",
            info.normres
        );
    }

    #[test]
    fn t2_davidson_maxiter_exhaustion_not_converged() {
        let space = QnSpace::new(1, 0, 0);
        let r = 8usize;
        let s0 = Sector::new(0, 0);
        let mut row_ranks = RankGrid::zeros(&space);
        row_ranks[s0] = r;
        let mut col_ranks = RankGrid::zeros(&space);
        col_ranks[s0] = 1;

        let mut hd = faer::Mat::<f64>::zeros(r, r);
        for i in 0..r {
            *hd.get_mut(i, i) = 1.0 + i as f64;
        }
        for i in 0..r {
            for j in (i + 1)..r {
                let v = 0.05 / (1.0 + (i as f64 - j as f64).abs());
                *hd.get_mut(i, j) = v;
                *hd.get_mut(j, i) = v;
            }
        }
        let mut d_core = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        {
            let mut b = d_core.block_mut(Family::Empty, s0).unwrap();
            for i in 0..r {
                *b.rb_mut().get_mut(i, 0) = *hd.get(i, i);
            }
        }
        let mut op_mut = |a: &SparseCore<f64>| -> SparseCore<f64> {
            let ab = a.block(Family::Empty, s0).unwrap();
            let mut out = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
            {
                let mut ob = out.block_mut(Family::Empty, s0).unwrap();
                for i in 0..r {
                    let mut acc = 0.0f64;
                    for j in 0..r {
                        acc += *hd.get(i, j) * *ab.get(j, 0);
                    }
                    *ob.rb_mut().get_mut(i, 0) = acc;
                }
            }
            out
        };

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A0D);
        let mut v0 = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        v0.fill_randn(&mut rng);
        let mut applyprec = |t: &mut SparseCore<f64>, theta: f64| {
            precond(t, &d_core, theta);
        };
        let opts = DavidsonOpts {
            tol: 1e-8,
            maxsubspace: 8,
            maxiter: 1,
            nwarm: 2,
            restart_keep: 1,
            verbosity: 0,
            theta_tol: None,
            stop_below: None,
        };
        let (_lambda, _v, info) = davidson(&mut op_mut, &v0, &mut applyprec, &opts);

        assert!(!info.converged, "maxiter=1 must not converge on this H_eff");
        assert_eq!(
            info.niter, 1,
            "maxiter=1 should record exactly one iteration"
        );
        assert_eq!(info.numops, 2, "warm-start op + one subspace-extension op");
        assert!(
            info.normres > 1e-8,
            "residual must exceed tol on a non-converged exit"
        );
    }

    #[test]
    fn t2_davidson_theta_tol_energy_exit() {
        let space = QnSpace::new(1, 0, 0);
        let r = 8usize;
        let s0 = Sector::new(0, 0);
        let mut row_ranks = RankGrid::zeros(&space);
        row_ranks[s0] = r;
        let mut col_ranks = RankGrid::zeros(&space);
        col_ranks[s0] = 1;

        let mut hd = faer::Mat::<f64>::zeros(r, r);
        for i in 0..r {
            *hd.get_mut(i, i) = 1.0 + i as f64;
        }
        for i in 0..r {
            for j in (i + 1)..r {
                let v = 0.05 / (1.0 + (i as f64 - j as f64).abs());
                *hd.get_mut(i, j) = v;
                *hd.get_mut(j, i) = v;
            }
        }
        let eigs = hd
            .as_ref()
            .self_adjoint_eigenvalues(faer::Side::Lower)
            .expect("dense self-adjoint eigendecomposition");
        let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);

        let mut d_core = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
        {
            let mut b = d_core.block_mut(Family::Empty, s0).unwrap();
            for i in 0..r {
                *b.rb_mut().get_mut(i, 0) = *hd.get(i, i);
            }
        }
        let op = |a: &SparseCore<f64>| -> SparseCore<f64> {
            let ab = a.block(Family::Empty, s0).unwrap();
            let mut out = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
            {
                let mut ob = out.block_mut(Family::Empty, s0).unwrap();
                for i in 0..r {
                    let mut acc = 0.0f64;
                    for j in 0..r {
                        acc += *hd.get(i, j) * *ab.get(j, 0);
                    }
                    *ob.rb_mut().get_mut(i, 0) = acc;
                }
            }
            out
        };
        let mut op_mut = op;

        let mut run = |theta_tol: Option<f64>| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A09);
            let mut v0 = SparseCore::<f64>::zeros(space, 0, &row_ranks, &col_ranks);
            v0.fill_randn(&mut rng);
            let mut applyprec = |t: &mut SparseCore<f64>, theta: f64| {
                precond(t, &d_core, theta);
            };
            let opts = DavidsonOpts {
                tol: 1e-8,
                maxsubspace: 8,
                maxiter: 100,
                nwarm: 2,
                restart_keep: 1,
                verbosity: 0,
                theta_tol,
                stop_below: None,
            };
            davidson(&mut op_mut, &v0, &mut applyprec, &opts)
        };

        let (lambda_ref, _v_ref, info_ref) = run(None);
        assert!(info_ref.converged, "residual-only baseline must converge");
        assert!(
            (lambda_ref - min_eig).abs() < 1e-9,
            "residual-only baseline eigenvalue mismatch: {lambda_ref} vs {min_eig}"
        );

        let (lambda_exit, _v_exit, info_exit) = run(Some(1e-5));
        assert!(info_exit.converged, "energy-exit run must converge");
        assert!(
            info_exit.numops < info_ref.numops,
            "theta_tol=1e-5 did not trim any applies vs the residual-only baseline \
             ({} vs {})",
            info_exit.numops,
            info_ref.numops
        );
        assert!(
            (lambda_exit - min_eig).abs() < 1e-8,
            "energy-exit eigenvalue drifted too far from the true ground state: {lambda_exit} vs \
             {min_eig}"
        );

        let (lambda_loose, _v_loose, _info_loose) = run(Some(1e-3));
        assert!(
            (lambda_loose - min_eig).abs() > 1e-4,
            "expected a 1e-3 theta_tol to visibly degrade accuracy on this problem (it no longer \
             does -- re-examine whether this guard still documents a real failure mode)"
        );
    }
}
