// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) MALS driver — rank-adaptive two-site alternating least squares. Mirrors the abelian
//! `crate::solvers::mals` line-for-line with SU(2) type substitutions.

use crate::core::twosite::TruncInfo;
use crate::core::twosite::noise_rng;
use crate::hamiltonian::instrument;
use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::solvers::davidson::{DavidsonOpts, davidson, precond};
use crate::solvers::davidson_block::{BlockDavidsonInfo, BlockDavidsonOpts, davidson_block};
pub use crate::solvers::mals::MalsOptions;
use crate::solvers::mals::THETA_TOL_FACTOR;
use crate::solvers::mals::{MalsHistory, MalsSweep, NoiseType, SweepTrunc};
use crate::su2::core::Su2Core;
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::matvec::{
    Su2TwoSiteWorkspace, noise_density_matrix_streamed, su2_onesite_workspace,
    su2_twosite_workspace,
};
use crate::su2::hamiltonian::rayleigh_quotient;
use crate::su2::solvers::als::{INNER_TOL_CAP, boundary_idsketch};
use crate::su2::solvers::envstore::Su2EnvStore;
pub use crate::su2::twosite::{
    Su2ContractedCores, contract, factor_qc, factor_svd, factor_svd_noise, factor_svd_perturbative,
};
use crate::su2::vector::Su2TtVector;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Rank-adaptive two-site ground-state search on the SU(2) layer ( abelian twin
/// `crate::solvers::mals::mals`, `src/solvers/mals.rs`). Right-orthogonalizes a clone of
/// `x0`, alternately sweeps forward/backward solving each two-site local eigenproblem via the
/// shared generic [`crate::solvers::davidson::davidson`] and truncating with [`factor_svd`], until
/// the Rayleigh quotient stabilizes to `opts.eps`. Returns `(λ, x)` — `λ: f64`, mirroring the
/// abelian MALS's `f64`-vs-ALS's-`T` return-type asymmetry verbatim (so the pair stays
/// diffable). Does not mutate `x0` (see
/// [`mals_mut`] for the in-place form).
pub fn mals<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &MalsOptions,
) -> (f64, Su2TtVector<T>) {
    let mut x = x0.clone();
    let lambda = mals_mut(h, &mut x, opts);
    (lambda, x)
}

/// In-place [`mals`]: mutates `x` to the converged (possibly rank-grown) ground-state train,
/// returns `λ` (abelian twin `crate::solvers::mals::mals_mut`).
/// Carries the abelian's `bond_dims`-non-empty guard (panics with a descriptive message on
/// an empty schedule), the hoisted `ep = eps/√(d-1)` with its `d == 1` guard, and the
/// inner-tolerance clamp chain (NOT `f64::clamp`).
pub fn mals_mut<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    opts: &MalsOptions,
) -> f64 {
    mals_run(h, x, opts).lambda
}

/// [`mals`] keeping the per-iteration diagnostics — the SU(2) twin of
/// [`crate::solvers::mals::mals_with_history`].
pub fn mals_with_history<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
    opts: &MalsOptions,
) -> (MalsHistory, Su2TtVector<T>) {
    let mut x = x0.clone();
    let history = mals_run(h, &mut x, opts);
    (history, x)
}

pub(crate) fn mals_run<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    opts: &MalsOptions,
) -> MalsHistory {
    assert!(
        !opts.bond_dims.is_empty(),
        "MalsOptions::bond_dims must be non-empty (use [0] for uncapped)"
    );
    assert!(
        !opts.noise.is_empty(),
        "MalsOptions::noise must be non-empty (use [0.0] for a noise-free run)"
    );
    assert!(
        opts.noise.iter().all(|a| a.is_finite() && *a >= 0.0),
        "MalsOptions::noise entries must be finite and non-negative"
    );
    let mut rng = noise_rng(opts.seed);
    let mut sweeps: Vec<MalsSweep> = Vec::with_capacity(opts.max_iter);
    let d = x.space.d;
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr = build_right_store(h, x);
    let mut lambda = rayleigh_quotient(h, x).re_s();
    let mut r = 1.0f64;

    for it in 0..opts.max_iter {
        let bond_dim = opts.bond_dims[it.min(opts.bond_dims.len() - 1)];
        let noise = opts.noise[it.min(opts.noise.len() - 1)];
        let raw = 0.01 * r;
        let inner_tol = if raw > INNER_TOL_CAP {
            INNER_TOL_CAP
        } else if raw < opts.eps {
            opts.eps
        } else {
            raw
        };
        let ep = if d > 1 {
            opts.eps / ((d - 1) as f64).sqrt()
        } else {
            opts.eps
        };

        let (lf, mut fl, tf) = mals_forward_sweep(
            h,
            x,
            &mut fr,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.noise_type,
            &mut rng,
        );
        if opts.verbose {
            println!("MALS iter {it} forward sweep: energy = {lf}");
        }
        debug_assert!(
            fr.all_empty(),
            "L1: the forward sweep must consume the whole right stack"
        );
        let (lambda_n, fr_new, tb) = mals_back_sweep(
            h,
            x,
            &mut fl,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.noise_type,
            &mut rng,
        );
        debug_assert!(
            fl.all_empty(),
            "L1: the back sweep must consume the whole left stack"
        );
        fr = fr_new;

        let trunc = tf.merge(tb);
        r = (lambda - lambda_n).abs() / lambda_n.abs();
        lambda = lambda_n;
        sweeps.push(MalsSweep {
            energy: lambda,
            discarded_weight: trunc.max_discarded,
            max_bond_dim: trunc.max_kept,
            bond_dim,
            noise,
        });
        if opts.verbose {
            println!(
                "MALS iter {it} back sweep: energy = {lambda}, relative improvement = {r}, \
                 discarded weight = {:.3e}, M = {}, noise = {noise:.1e}",
                trunc.max_discarded, trunc.max_kept
            );
        }
        if noise == 0.0 && r < opts.convergence_tol() && inner_tol <= opts.eps {
            break;
        }
    }
    MalsHistory { lambda, sweeps }
}

pub(crate) fn local_ground_2s<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    fl: &Su2Sketch<T>,
    fr: &Su2Sketch<T>,
    cp0: &Su2ContractedCores<T>,
    inner_tol: f64,
) -> (f64, Su2ContractedCores<T>) {
    let d_diag = h.two_site_effective_diagonal(fl, fr, cp0);
    let mut ws: Option<Su2TwoSiteWorkspace<T>> = su2_twosite_workspace();
    let mut op =
        |cp: &Su2ContractedCores<T>| h.two_site_sketched_hamiltonian_ws(&mut ws, cp, fl, fr);
    let mut applyprec = |t: &mut Su2ContractedCores<T>, theta: f64| precond(t, &d_diag, theta);
    let dopts = DavidsonOpts {
        tol: inner_tol,
        theta_tol: Some(inner_tol * THETA_TOL_FACTOR),
        ..Default::default()
    };
    let (theta, v, _info) = davidson(&mut op, cp0, &mut applyprec, &dopts);
    (theta, v)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn local_lowest_2s<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    fl: &Su2Sketch<T>,
    fr: &Su2Sketch<T>,
    cp0s: &[Su2ContractedCores<T>],
    nroots: usize,
    inner_tol: f64,
    maxsubspace: usize,
    restart_keep: usize,
) -> (Vec<f64>, Vec<Su2ContractedCores<T>>, BlockDavidsonInfo) {
    assert!(
        !cp0s.is_empty(),
        "local_lowest_2s: at least one two-site guess block is required"
    );
    let cp0 = &cp0s[0];
    for (i, g) in cp0s.iter().enumerate().skip(1) {
        assert!(
            g.k == cp0.k && g.row_ranks == cp0.row_ranks && g.col_ranks == cp0.col_ranks,
            "local_lowest_2s: guess {i} has a different two-site shape from guess 0 at pair ({}, {})",
            cp0.k,
            cp0.k + 1
        );
    }
    let d_diag = h.two_site_effective_diagonal(fl, fr, cp0);
    let mut ws: Option<Su2TwoSiteWorkspace<T>> = su2_twosite_workspace();
    let mut op =
        |cp: &Su2ContractedCores<T>| h.two_site_sketched_hamiltonian_ws(&mut ws, cp, fl, fr);
    let mut applyprec = |t: &mut Su2ContractedCores<T>, theta: f64| precond(t, &d_diag, theta);
    let bopts = BlockDavidsonOpts {
        tol: inner_tol,
        theta_tol: Some(inner_tol * THETA_TOL_FACTOR),
        maxsubspace,
        restart_keep,
        ..BlockDavidsonOpts::for_roots(nroots)
    };
    let (thetas, us, info) = davidson_block(&mut op, cp0s, &mut applyprec, nroots, &bopts);
    assert!(
        us.len() == nroots,
        "local_lowest_2s: the block Davidson returned {} of {nroots} roots at pair ({}, {}) — the \
         subspace stagnated before it could host them; supply nroots distinct guesses",
        us.len(),
        cp0.k,
        cp0.k + 1
    );
    (thetas, us, info)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn split_two_site<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    fl: &Su2Sketch<T>,
    fr: &Su2Sketch<T>,
    cp: &Su2ContractedCores<T>,
    ep: f64,
    max_rank: usize,
    noise: f64,
    noise_type: NoiseType,
    forward: bool,
    rng: &mut Xoshiro256PlusPlus,
) -> (Su2Core<T>, Su2Sketch<T>, Su2Core<T>, TruncInfo) {
    if noise <= 0.0 {
        return factor_svd_noise(cp, ep, Pooled(max_rank), true, 0.0, rng);
    }
    match noise_type {
        NoiseType::Random => factor_svd_noise(cp, ep, Pooled(max_rank), true, noise, rng),
        NoiseType::Perturbative => {
            let (xk, xk1) = factor_qc(cp);
            let noise_dm =
                noise_density_matrix_streamed(h, if forward { fl } else { fr }, &xk, &xk1, forward);
            factor_svd_perturbative(cp, &noise_dm, ep, Pooled(max_rank), true, noise, forward)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mals_forward_sweep<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fr: &mut Su2EnvStore<T>,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, Su2EnvStore<T>, SweepTrunc) {
    debug_assert_eq!(x.ortho, Some(0), "MALS forward sweep expects center 0");
    let d = x.space.d;
    let mut fl = Su2EnvStore::new(x.space, d, "L");
    fl.put(0, boundary_idsketch(x, 0, 1));
    fr.drop_slot(0);
    fr.prefetch(1);
    let mut lambda = 0.0f64;
    let mut trunc = SweepTrunc::default();

    for k in 0..(d - 1) {
        fr.prefetch(k + 2);
        let _t = instrument::tic();
        let cp0 = contract(x.core(k), x.core(k + 1));
        instrument::stage_end(instrument::stage::CONTRACT, _t);
        let _t = instrument::tic();
        let (theta, cp) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            local_ground_2s(h, fl_k, fr_k1, &cp0, inner_tol)
        };
        instrument::stage_end(instrument::stage::DAVIDSON, _t);
        lambda = theta;
        let _t = instrument::tic();
        let (xk, s, mut xk1, info) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            split_two_site(
                h, fl_k, fr_k1, &cp, ep, max_rank, noise, noise_type, true, rng,
            )
        };
        instrument::stage_end(instrument::stage::SPLIT, _t);
        trunc.absorb(info);
        fr.drop_slot(k + 1);
        xk1.fold_sketch_left_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k + 1);
        if k < d - 2 {
            let _t = instrument::tic();
            let ws = su2_onesite_workspace(h, x.core(k));
            let next = h.sketching_step_right(x, fl.get(k), &ws);
            fl.put(k + 1, next);
            fl.release(k);
            instrument::stage_end(instrument::stage::ENV, _t);
        }
    }
    (lambda, fl, trunc)
}

#[allow(clippy::too_many_arguments)]
fn mals_back_sweep<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &mut Su2TtVector<T>,
    fl: &mut Su2EnvStore<T>,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, Su2EnvStore<T>, SweepTrunc) {
    let d = x.space.d;
    if x.ortho != Some(d - 1) {
        x.move_core(d - 1, false);
    }
    let mut fr = Su2EnvStore::new(x.space, d, "R");
    fr.put(d - 1, boundary_idsketch(x, d, 1));
    fl.drop_slot(d - 1);
    if d >= 2 {
        fl.prefetch(d - 2);
    }
    let mut lambda = 0.0f64;
    let mut trunc = SweepTrunc::default();

    for k in (0..(d - 1)).rev() {
        if k >= 1 {
            fl.prefetch(k - 1);
        }
        let _t = instrument::tic();
        let cp0 = contract(x.core(k), x.core(k + 1));
        instrument::stage_end(instrument::stage::CONTRACT, _t);
        let _t = instrument::tic();
        let (theta, cp) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            local_ground_2s(h, fl_k, fr_k1, &cp0, inner_tol)
        };
        instrument::stage_end(instrument::stage::DAVIDSON, _t);
        lambda = theta;
        let _t = instrument::tic();
        let (mut xk, s, xk1, info) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            split_two_site(
                h, fl_k, fr_k1, &cp, ep, max_rank, noise, noise_type, false, rng,
            )
        };
        instrument::stage_end(instrument::stage::SPLIT, _t);
        trunc.absorb(info);
        fl.drop_slot(k);
        xk.fold_sketch_right_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k);
        if k > 0 {
            let _t = instrument::tic();
            let ws = su2_onesite_workspace(h, x.core(k + 1));
            let next = h.sketching_step_left(x, fr.get(k + 1), &ws);
            fr.put(k, next);
            fr.release(k + 1);
            instrument::stage_end(instrument::stage::ENV, _t);
        }
    }
    (lambda, fr, trunc)
}

pub(crate) fn build_right_store<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
) -> Su2EnvStore<T> {
    let d = x.space.d;
    let mut fr = Su2EnvStore::new(x.space, d, "R");
    fr.put(d - 1, boundary_idsketch(x, d, 1));
    for k in (0..d - 1).rev() {
        let ws = su2_onesite_workspace(h, x.core(k + 1));
        let next = h.sketching_step_left(x, fr.get(k + 1), &ws);
        fr.put(k, next);
        fr.release(k + 1);
    }
    fr
}

#[cfg(test)]
mod tms_a10su2_local_lowest_tests {
    use super::{Su2ContractedCores, contract, factor_qc, local_ground_2s, local_lowest_2s};
    use crate::coeff::CoeffTensor;
    use crate::core::factor::dense_self_adjoint_eigen;
    use crate::solvers::InnerProductVector;
    use crate::solvers::davidson::{DAVIDSON_MAXSUBSPACE, DAVIDSON_RESTART_KEEP};
    use crate::su2::frame::Su2Sketch;
    use crate::su2::hamiltonian::Su2SparseHamiltonian;
    use crate::su2::qn::Su2Space;
    use crate::su2::solvers::als::{boundary_idsketch, right_to_left_sketching};
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randomized_su2_state, seeded};
    use faer::Mat;

    fn fixture() -> (
        Su2SparseHamiltonian<f64>,
        Su2Sketch<f64>,
        Su2Sketch<f64>,
        Su2ContractedCores<f64>,
    ) {
        let d = 6usize;
        let space = Su2Space::new(d, 2, 0);
        let t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], -1.0);
        v.set(&[1, 3, 0, 2], -1.0);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = seeded(0x5A_1000);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        x.right_orthogonalize(true);
        let fr = right_to_left_sketching(&h, &x);
        let fl0 = boundary_idsketch(&x, 0, 1);
        let cp0 = contract(x.core(0), x.core(1));
        (h, fl0, fr[1].clone(), cp0)
    }

    #[test]
    fn tms_a10su2_effective_diagonal_depends_only_on_the_two_site_shape() {
        let (h, fl0, fr1, cp0) = fixture();
        let d_full = h.two_site_effective_diagonal(&fl0, &fr1, &cp0);
        let mut e = cp0.clone();
        e.for_each_coeff_mut(|c| *c = 0.0);
        e.set_coeff(0, 1.0);
        assert_ne!(
            factor_qc(&e).0.col_ranks,
            factor_qc(&cp0).0.col_ranks,
            "tms_a10su2: the rank-1 probe did not perturb the QC mid-rank grid — the pin is vacuous"
        );
        let d_e = h.two_site_effective_diagonal(&fl0, &fr1, &e);
        for i in 0..d_full.num_coeffs() {
            assert_eq!(
                d_full.coeff(i).to_bits(),
                d_e.coeff(i).to_bits(),
                "tms_a10su2: effective diagonal entry {i} moved with the guess ({} vs {})",
                d_full.coeff(i),
                d_e.coeff(i)
            );
        }
    }
    #[test]
    fn tms_a10su2_local_lowest_matches_local_ground_at_one_root() {
        let (h, fl0, fr1, cp0) = fixture();
        let inner_tol = 1e-8;
        let (theta_g, u_g) = local_ground_2s(&h, &fl0, &fr1, &cp0, inner_tol);
        let (thetas, us, _info) = local_lowest_2s(
            &h,
            &fl0,
            &fr1,
            std::slice::from_ref(&cp0),
            1,
            inner_tol,
            DAVIDSON_MAXSUBSPACE,
            DAVIDSON_RESTART_KEEP,
        );

        assert_eq!(thetas.len(), 1, "tms_a10su2: root count");
        assert_eq!(us.len(), 1, "tms_a10su2: vector count");
        assert!(
            (thetas[0] - theta_g).abs() <= 1e-12 * theta_g.abs().max(1.0),
            "tms_a10su2: θ {} vs local_ground_2s {theta_g} (Δ = {:.3e})",
            thetas[0],
            (thetas[0] - theta_g).abs()
        );
        assert_eq!(
            thetas[0].to_bits(),
            theta_g.to_bits(),
            "tms_a10su2: θ {} and local_ground_2s {theta_g} are not BITWISE equal",
            thetas[0]
        );
        assert_eq!(
            us[0].num_coeffs(),
            u_g.num_coeffs(),
            "tms_a10su2: coefficient count"
        );
        for i in 0..u_g.num_coeffs() {
            assert_eq!(
                us[0].coeff(i).to_bits(),
                u_g.coeff(i).to_bits(),
                "tms_a10su2: coefficient {i} ({} vs {}) not bitwise equal",
                us[0].coeff(i),
                u_g.coeff(i)
            );
        }
    }

    #[test]
    #[should_panic(expected = "supply nroots distinct guesses")]
    #[allow(clippy::needless_range_loop)]
    fn tms_a10su2_local_lowest_panics_when_the_subspace_stagnates() {
        let (h, fl0, fr1, cp0) = fixture();
        let n = cp0.num_coeffs();
        let basis: Vec<Su2ContractedCores<f64>> = (0..n)
            .map(|j| {
                let mut e = cp0.clone();
                e.for_each_coeff_mut(|c| *c = 0.0);
                e.set_coeff(j, 1.0);
                e
            })
            .collect();
        let images: Vec<Su2ContractedCores<f64>> = basis
            .iter()
            .map(|e| h.two_site_sketched_hamiltonian(e, &fl0, &fr1))
            .collect();
        let g: Vec<f64> = basis.iter().map(|e| e.dot(e)).collect();
        let mut b = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                *b.get_mut(i, j) = basis[i].dot(&images[j]) / (g[i] * g[j]).sqrt();
            }
        }
        let (_ev, y) = dense_self_adjoint_eigen(b.as_ref());
        let mut u_exact = cp0.clone();
        u_exact.for_each_coeff_mut(|c| *c = 0.0);
        for i in 0..n {
            u_exact.set_coeff(i, *y.get(i, 0) / g[i].sqrt());
        }
        let _ = local_lowest_2s(
            &h,
            &fl0,
            &fr1,
            std::slice::from_ref(&u_exact),
            2,
            1e-6,
            DAVIDSON_MAXSUBSPACE,
            2,
        );
    }

    fn block_at(k: usize) -> Su2ContractedCores<f64> {
        let d = 6usize;
        let space = Su2Space::new(d, 2, 0);
        let mut rng = seeded(0x5A_1000);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        x.right_orthogonalize(true);
        contract(x.core(k), x.core(k + 1))
    }

    #[allow(clippy::needless_range_loop)]
    fn dense_local(
        h: &Su2SparseHamiltonian<f64>,
        fl0: &Su2Sketch<f64>,
        fr1: &Su2Sketch<f64>,
        cp0: &Su2ContractedCores<f64>,
    ) -> (Vec<f64>, Su2ContractedCores<f64>) {
        let n = cp0.num_coeffs();
        let basis: Vec<Su2ContractedCores<f64>> = (0..n)
            .map(|j| {
                let mut e = cp0.clone();
                e.for_each_coeff_mut(|c| *c = 0.0);
                e.set_coeff(j, 1.0);
                e
            })
            .collect();
        let images: Vec<Su2ContractedCores<f64>> = basis
            .iter()
            .map(|e| h.two_site_sketched_hamiltonian(e, fl0, fr1))
            .collect();
        let g: Vec<f64> = basis.iter().map(|e| e.dot(e)).collect();
        let mut b = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                *b.get_mut(i, j) = basis[i].dot(&images[j]) / (g[i] * g[j]).sqrt();
            }
        }
        let (ev, y) = dense_self_adjoint_eigen(b.as_ref());
        let mut u_exact = cp0.clone();
        u_exact.for_each_coeff_mut(|c| *c = 0.0);
        for i in 0..n {
            u_exact.set_coeff(i, *y.get(i, 0) / g[i].sqrt());
        }
        (ev, u_exact)
    }

    #[test]
    fn tms_a10su2_local_lowest_dispatches_every_guess() {
        let (h, fl0, fr1, cp0) = fixture();
        let (ev, u_exact) = dense_local(&h, &fl0, &fr1, &cp0);
        let nroots = 2usize;
        let (thetas, us, _info) = local_lowest_2s(
            &h,
            &fl0,
            &fr1,
            &[u_exact.clone(), cp0.clone()],
            nroots,
            1e-10,
            16,
            4,
        );
        assert_eq!(thetas.len(), nroots, "tms_a10su2: root count");
        assert_eq!(us.len(), nroots, "tms_a10su2: vector count");
        for i in 0..nroots {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "tms_a10su2: θ[{i}] = {} vs the generalized-pencil level {} (Δ = {:.3e})",
                thetas[i],
                ev[i],
                (thetas[i] - ev[i]).abs()
            );
        }
        for i in 0..nroots {
            for j in 0..nroots {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (us[i].dot(&us[j]) - want).abs() <= 1e-10,
                    "tms_a10su2: ⟨u_{i}, u_{j}⟩ = {} (want {want})",
                    us[i].dot(&us[j])
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "different two-site shape")]
    fn tms_a10su2_local_lowest_rejects_a_guess_of_another_shape() {
        let (h, fl0, fr1, cp0) = fixture();
        let other = block_at(1);
        let _ = local_lowest_2s(&h, &fl0, &fr1, &[cp0, other], 2, 1e-8, 16, 4);
    }
}

#[cfg(test)]
mod l1_stack_lifetime_tests {
    use super::*;
    use crate::su2::qn::Su2Space;
    use crate::test_util::{generic_su2_h, randomized_su2_state, seeded};

    #[test]
    fn sweeps_consume_the_stack_they_read() {
        let space = Su2Space::new(6, 4, 0);
        let h = generic_su2_h::<f64>(space);
        let mut rng = seeded(0x11_0002);
        let mut x = randomized_su2_state::<f64, _>(space, 1, 3, &mut rng);
        x.right_orthogonalize(true);
        let mut fr = build_right_store(&h, &x);
        let mut noise = noise_rng(0);
        let (_, mut fl, _) = mals_forward_sweep(
            &h,
            &mut x,
            &mut fr,
            1e-8,
            0,
            1e-6,
            0.0,
            NoiseType::Perturbative,
            &mut noise,
        );
        assert!(fr.all_empty(), "forward sweep left Fᴿ slots alive");
        assert_eq!(fl.occupied(), x.space.d - 1);
        let (_, fr2, _) = mals_back_sweep(
            &h,
            &mut x,
            &mut fl,
            1e-8,
            0,
            1e-6,
            0.0,
            NoiseType::Perturbative,
            &mut noise,
        );
        assert!(fl.all_empty(), "back sweep left Fᴸ slots alive");
        assert_eq!(fr2.occupied(), x.space.d - 1);
    }
}
