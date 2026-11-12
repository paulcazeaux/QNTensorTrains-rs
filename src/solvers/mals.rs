// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! MALS driver: rank-adaptive two-site alternating least squares.

pub use crate::core::twosite::{
    ContractedSparseCores, TruncInfo, contract, factor_qc, factor_svd, factor_svd_noise,
    factor_svd_perturbative, noise_rng,
};
use crate::qn::Pooled;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::core::SparseCore;
use crate::core::factor::Sketch;
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::instrument;
use crate::hamiltonian::matvec::onesite_workspace;
use crate::hamiltonian::rayleigh_quotient;
use crate::scalar::Scalar;
use crate::solvers::als::{INNER_TOL_CAP, boundary_idsketch};
use crate::solvers::davidson::{DavidsonOpts, davidson, precond};
use crate::solvers::davidson_block::{BlockDavidsonInfo, BlockDavidsonOpts, davidson_block};
use crate::solvers::envstore::EnvStore;
use crate::solvers::interface::clamp_hi_first;
use crate::vector::TtVector;

pub(crate) const THETA_TOL_FACTOR: f64 = 2e-4;

/// Which flavour of DMRG noise the schedule applies.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum NoiseType {
    /// Isotropic random perturbation of the two-site block before its SVD
    /// ([`crate::core::twosite::factor_svd_noise`]). Measured to escape the determinant fixed point
    /// and converge well at amplitude `~1e-4`. Amplitude-sensitive, and not variational — the
    /// STATE is perturbed, not merely the choice of kept basis, so an energy measured while
    /// `noise > 0` can sit below the manifold minimum.
    Random,
    /// Perturbative / White-style density-matrix noise ([`crate::core::twosite::factor_svd_perturbative`]).
    /// The **default**: enriches the kept basis with the true White channel `E = L_w·B` (the operator
    /// applied to the STATE-weighted two-site block), forming the density matrix
    /// `ρ' = ρ + α_eff·ρ_noise` with `ρ_noise = Σ_w (L_w B)(L_w B)ᴴ/‖L_w B‖²` and diagonalising it
    /// (Gram/eigen route, no wide `E`). Each operator channel `w` is **unit-trace-normalized** (every
    /// channel scaled to unit weight) so identity/number channels don't dominate, and a global
    /// `α_eff = α/‖ρ_noise‖_F` makes `ρ_noise` bond-portable — so a single nominal `α` (`~1e-4`) works
    /// across bonds and molecules. Operator-targeted, deterministic, and variational
    /// throughout (the retained state is `B` projected onto an orthonormal kept basis, so even a
    /// noisy iteration carries a strict energy upper bound), exact at full rank, and measured to
    /// track/beat [`Self::Random`] on N₂/cc-pVDZ Fiedler. Amplitude `α` is **load-bearing**: it
    /// sets how far the basis tilts toward the operator-coupled directions a determinant has no
    /// weight in. Both layers dispatch it
    /// ([`crate::su2::twosite::factor_svd_perturbative`] /
    /// [`crate::su2::hamiltonian::matvec::noise_density_matrix`]), where the enrichment is built in the
    /// θ-scaled metric so it means the same thing across reduced multiplet sectors.
    #[default]
    Perturbative,
}

/// MALS driver options. `bond_dims = [0]` (the default) means "no explicit rank cap" — the
/// two-site QC keeps the ε-truncated rank at every bond (`chop_energy(…, max_rank=0)` treats
/// `0` as uncapped).
#[derive(Clone, Debug)]
pub struct MalsOptions {
    pub eps: f64,
    /// Per-outer-iteration rank cap schedule (clamped to the last entry once `it` exceeds the
    /// schedule length). `0` = uncapped. Pooled total bond dimension (`Pooled`) summed over sectors
    /// — the conventional DMRG `M`; on SU(2) reduced multiplets.
    pub bond_dims: Vec<usize>,
    /// Per-outer-iteration noise amplitude, read with the same clamp as `bond_dims`. `0.0` disables
    /// noise entirely and takes a path bit-identical to the noise-free driver.
    pub noise: Vec<f64>,
    /// Seed for the noise draws (only the [`NoiseType::Random`] path draws; perturbative noise is
    /// deterministic). Unused when every `noise` entry is `0.0`.
    pub seed: u64,
    /// Which noise flavour the schedule applies. Defaults to [`NoiseType::Perturbative`];
    /// [`NoiseType::Random`] is the explicit opt-in backup. Ignored when `noise` is all-zero.
    pub noise_type: NoiseType,
    pub max_iter: usize,
    pub verbose: bool,
    /// Convergence test on the relative energy improvement between outer iterations.
    pub conv_tol: Option<f64>,
}

impl MalsOptions {
    /// The relative-improvement threshold the sweep stops on: [`conv_tol`](Self::conv_tol) when
    /// set, else [`eps`](Self::eps).
    #[inline]
    pub fn convergence_tol(&self) -> f64 {
        self.conv_tol.unwrap_or(self.eps)
    }
}

impl Default for MalsOptions {
    fn default() -> Self {
        MalsOptions {
            eps: 1e-6,
            conv_tol: None,
            bond_dims: vec![0],
            noise: vec![0.0],
            seed: 1234,
            noise_type: NoiseType::default(),
            max_iter: 20,
            verbose: false,
        }
    }
}

/// One outer iteration's worth of diagnostics (a forward + a back sweep).
#[derive(Clone, Copy, Debug)]
pub struct MalsSweep {
    /// Rayleigh quotient after the back sweep. NOT variational while `noise > 0` under
    /// [`NoiseType::Random`] (which perturbs the state); [`NoiseType::Perturbative`] keeps a strict
    /// upper bound throughout.
    pub energy: f64,
    /// Largest relative discarded weight over the bonds of this iteration — the binding
    /// truncation, and the standard rank-convergence diagnostic.
    pub discarded_weight: f64,
    /// Largest total (summed-over-sectors) bond dimension reached.
    pub max_bond_dim: usize,
    /// The rank cap in force this iteration (`0` = uncapped).
    pub bond_dim: usize,
    /// The noise amplitude in force this iteration.
    pub noise: f64,
}

/// [`mals_with_history`]'s result: the final energy plus a record of every outer iteration.
#[derive(Clone, Debug)]
pub struct MalsHistory {
    pub lambda: f64,
    pub sweeps: Vec<MalsSweep>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SweepTrunc {
    pub(crate) max_discarded: f64,
    pub(crate) max_kept: usize,
}

impl SweepTrunc {
    pub(crate) fn absorb(&mut self, info: TruncInfo) {
        self.max_discarded = self.max_discarded.max(info.discarded_weight);
        self.max_kept = self.max_kept.max(info.kept);
    }
    pub(crate) fn merge(self, other: SweepTrunc) -> SweepTrunc {
        SweepTrunc {
            max_discarded: self.max_discarded.max(other.max_discarded),
            max_kept: self.max_kept.max(other.max_kept),
        }
    }
}

/// Rank-adaptive two-site ground-state search. Right-orthogonalizes a clone of `x0`, alternately
/// sweeps forward/backward solving each two-site local eigenproblem via Davidson and truncating
/// with [`factor_svd`], until the Rayleigh quotient stabilizes to `opts.eps`. Returns `(λ, x)`;
/// does not mutate `x0` (see [`mals_mut`] for the in-place form).
pub fn mals<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &MalsOptions,
) -> (f64, TtVector<T>) {
    let mut x = x0.clone();
    let lambda = mals_mut(h, &mut x, opts);
    (lambda, x)
}

/// [`mals`] keeping the per-iteration diagnostics: energy, discarded weight, bond dimension and
/// the noise in force. Use this when you need to judge whether a run converged rather than just
/// take its number — with a rank ramp, the discarded weight is what says so.
pub fn mals_with_history<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    opts: &MalsOptions,
) -> (MalsHistory, TtVector<T>) {
    let mut x = x0.clone();
    let history = mals_run(h, &mut x, opts);
    (history, x)
}

/// In-place [`mals`]: mutates `x` to the converged (possibly rank-grown) ground-state train,
/// returns `λ`.
pub fn mals_mut<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    opts: &MalsOptions,
) -> f64 {
    mals_run(h, x, opts).lambda
}

pub(crate) fn mals_run<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
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
    let d = x.space.d;
    let mut rng = noise_rng(opts.seed);
    let mut sweeps: Vec<MalsSweep> = Vec::with_capacity(opts.max_iter);
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr = build_right_store(h, x);
    let mut lambda = rayleigh_quotient(h, x).re_s();
    let mut r = 1.0f64;

    for it in 0..opts.max_iter {
        let bond_dim = opts.bond_dims[it.min(opts.bond_dims.len() - 1)];
        let noise = opts.noise[it.min(opts.noise.len() - 1)];
        let inner_tol = clamp_hi_first(0.01 * r, opts.eps, INNER_TOL_CAP);
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
        if r < opts.convergence_tol() && inner_tol <= opts.eps && noise == 0.0 {
            break;
        }
    }
    MalsHistory { lambda, sweeps }
}

pub(crate) fn local_ground_2s<T: Scalar>(
    h: &SparseHamiltonian<T>,
    fl: &Sketch<T>,
    fr: &Sketch<T>,
    cp0: &ContractedSparseCores<T>,
    inner_tol: f64,
) -> (f64, ContractedSparseCores<T>) {
    let d_diag = h.two_site_effective_diagonal(fl, fr, cp0);
    let (xk0, xk10) = factor_qc(cp0);
    let il = h.row_ranges(&xk0);
    let jr = h.col_ranges(&xk10);
    let mut op = |cp: &ContractedSparseCores<T>| {
        h.two_site_sketched_hamiltonian_cached(cp, fl, fr, &il, &jr)
    };
    let mut applyprec = |t: &mut ContractedSparseCores<T>, theta: f64| precond(t, &d_diag, theta);
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
    h: &SparseHamiltonian<T>,
    fl: &Sketch<T>,
    fr: &Sketch<T>,
    cp0s: &[ContractedSparseCores<T>],
    nroots: usize,
    inner_tol: f64,
    maxsubspace: usize,
    restart_keep: usize,
) -> (Vec<f64>, Vec<ContractedSparseCores<T>>, BlockDavidsonInfo) {
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
    let (xk0, xk10) = factor_qc(cp0);
    let il = h.row_ranges(&xk0);
    let jr = h.col_ranges(&xk10);
    let mut op = |cp: &ContractedSparseCores<T>| {
        h.two_site_sketched_hamiltonian_cached(cp, fl, fr, &il, &jr)
    };
    let mut applyprec = |t: &mut ContractedSparseCores<T>, theta: f64| precond(t, &d_diag, theta);
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
    h: &SparseHamiltonian<T>,
    fl: &Sketch<T>,
    fr: &Sketch<T>,
    cp: &ContractedSparseCores<T>,
    ep: f64,
    max_rank: usize,
    noise: f64,
    noise_type: NoiseType,
    forward: bool,
    rng: &mut Xoshiro256PlusPlus,
) -> (SparseCore<T>, Sketch<T>, SparseCore<T>, TruncInfo) {
    if noise <= 0.0 {
        return factor_svd_noise(cp, ep, Pooled(max_rank), true, 0.0, rng);
    }
    match noise_type {
        NoiseType::Random => factor_svd_noise(cp, ep, Pooled(max_rank), true, noise, rng),
        NoiseType::Perturbative => {
            let (xk, xk1) = factor_qc(cp);
            let noise_dm = crate::hamiltonian::matvec::noise_density_matrix_streamed(
                h,
                if forward { fl } else { fr },
                &xk,
                &xk1,
                forward,
            );
            factor_svd_perturbative(cp, &noise_dm, ep, Pooled(max_rank), true, noise, forward)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mals_forward_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fr: &mut EnvStore<T>,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, EnvStore<T>, SweepTrunc) {
    debug_assert_eq!(x.ortho, Some(0), "MALS forward sweep expects center 0");
    let d = x.space.d;
    let mut fl = EnvStore::new(x.space, d, "L");
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
            let ws = onesite_workspace(h, x.core(k));
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
    h: &SparseHamiltonian<T>,
    x: &mut TtVector<T>,
    fl: &mut EnvStore<T>,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, EnvStore<T>, SweepTrunc) {
    let d = x.space.d;
    if x.ortho != Some(d - 1) {
        x.move_core(d - 1, false);
    }
    let mut fr = EnvStore::new(x.space, d, "R");
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
            let ws = onesite_workspace(h, x.core(k + 1));
            let next = h.sketching_step_left(x, fr.get(k + 1), &ws);
            fr.put(k, next);
            fr.release(k + 1);
            instrument::stage_end(instrument::stage::ENV, _t);
        }
    }
    (lambda, fr, trunc)
}

pub(crate) fn build_right_store<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
) -> EnvStore<T> {
    let d = x.space.d;
    let mut fr = EnvStore::new(x.space, d, "R");
    fr.put(d - 1, boundary_idsketch(x, d, 1));
    for k in (0..d - 1).rev() {
        let ws = onesite_workspace(h, x.core(k + 1));
        let next = h.sketching_step_left(x, fr.get(k + 1), &ws);
        fr.put(k, next);
        fr.release(k + 1);
    }
    fr
}

const _: () = assert!(
    THETA_TOL_FACTOR > 0.0 && THETA_TOL_FACTOR <= 5e-4,
    "THETA_TOL_FACTOR left its calibrated safe range (0, 5e-4] -- re-run the N2 energy-sensitivity \
     calibration scan before changing this constant"
);

#[cfg(test)]
mod tms_a10_local_lowest_tests {
    use super::{contract, factor_qc, local_ground_2s, local_lowest_2s};
    use crate::coeff::CoeffTensor;
    use crate::core::factor::{Sketch, dense_self_adjoint_eigen};
    use crate::core::twosite::ContractedSparseCores;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::solvers::InnerProductVector;
    use crate::solvers::als::{boundary_idsketch, right_to_left_sketching};
    use crate::solvers::davidson::{DAVIDSON_MAXSUBSPACE, DAVIDSON_RESTART_KEEP};
    use crate::test_util::{randomized_state, seeded};
    use crate::vector::TtVector;
    use faer::Mat;

    fn fixture() -> (
        SparseHamiltonian<f64>,
        Sketch<f64>,
        Sketch<f64>,
        ContractedSparseCores<f64>,
    ) {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 4, 1], -1.0);
        v.set(&[1, 4, 0, 2], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = seeded(0x7A30);
        let mut x: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        x.right_orthogonalize(true);
        let fr = right_to_left_sketching(&h, &x);
        let fl0 = boundary_idsketch(&x, 0, 1);
        let cp0 = contract(x.core(0), x.core(1));
        (h, fl0, fr[1].clone(), cp0)
    }

    #[test]
    fn tms_a10_effective_diagonal_depends_only_on_the_two_site_shape() {
        let (h, fl0, fr1, cp0) = fixture();
        let d_full = h.two_site_effective_diagonal(&fl0, &fr1, &cp0);
        let mut e = cp0.clone();
        e.for_each_coeff_mut(|c| *c = 0.0);
        e.set_coeff(0, 1.0);
        assert_ne!(
            factor_qc(&e).0.col_ranks,
            factor_qc(&cp0).0.col_ranks,
            "tms_a10: the rank-1 probe did not perturb the QC mid-rank grid — the pin is vacuous"
        );
        let d_e = h.two_site_effective_diagonal(&fl0, &fr1, &e);
        for i in 0..d_full.num_coeffs() {
            assert_eq!(
                d_full.coeff(i).to_bits(),
                d_e.coeff(i).to_bits(),
                "tms_a10: effective diagonal entry {i} moved with the guess ({} vs {})",
                d_full.coeff(i),
                d_e.coeff(i)
            );
        }
    }
    #[test]
    fn tms_a10_local_lowest_matches_local_ground_at_one_root() {
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

        assert_eq!(thetas.len(), 1, "tms_a10: root count");
        assert_eq!(us.len(), 1, "tms_a10: vector count");
        assert!(
            (thetas[0] - theta_g).abs() <= 1e-12 * theta_g.abs().max(1.0),
            "tms_a10: θ {} vs local_ground_2s {theta_g} (Δ = {:.3e})",
            thetas[0],
            (thetas[0] - theta_g).abs()
        );
        assert_eq!(
            thetas[0].to_bits(),
            theta_g.to_bits(),
            "tms_a10: θ {} and local_ground_2s {theta_g} are not BITWISE equal",
            thetas[0]
        );
        assert_eq!(
            us[0].num_coeffs(),
            u_g.num_coeffs(),
            "tms_a10: coefficient count"
        );
        for i in 0..u_g.num_coeffs() {
            assert_eq!(
                us[0].coeff(i).to_bits(),
                u_g.coeff(i).to_bits(),
                "tms_a10: coefficient {i} ({} vs {}) not bitwise equal",
                us[0].coeff(i),
                u_g.coeff(i)
            );
        }
    }

    #[test]
    #[should_panic(expected = "supply nroots distinct guesses")]
    #[allow(clippy::needless_range_loop)]
    fn tms_a10_local_lowest_panics_when_the_subspace_stagnates() {
        let (h, fl0, fr1, cp0) = fixture();
        let n = cp0.num_coeffs();
        let basis: Vec<ContractedSparseCores<f64>> = (0..n)
            .map(|j| {
                let mut e = cp0.clone();
                e.for_each_coeff_mut(|c| *c = 0.0);
                e.set_coeff(j, 1.0);
                e
            })
            .collect();
        let images: Vec<ContractedSparseCores<f64>> = basis
            .iter()
            .map(|e| h.two_site_sketched_hamiltonian(e, &fl0, &fr1))
            .collect();
        let mut m = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                *m.get_mut(i, j) = basis[i].dot(&images[j]);
            }
        }
        let (_ev, y) = dense_self_adjoint_eigen(m.as_ref());
        let mut u_exact = cp0.clone();
        u_exact.for_each_coeff_mut(|c| *c = 0.0);
        for i in 0..n {
            u_exact.set_coeff(i, *y.get(i, 0));
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

    fn block_at(k: usize) -> ContractedSparseCores<f64> {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut rng = seeded(0x7A30);
        let mut x: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        x.right_orthogonalize(true);
        contract(x.core(k), x.core(k + 1))
    }

    #[allow(clippy::needless_range_loop)]
    fn dense_local(
        h: &SparseHamiltonian<f64>,
        fl0: &Sketch<f64>,
        fr1: &Sketch<f64>,
        cp0: &ContractedSparseCores<f64>,
    ) -> (Vec<f64>, ContractedSparseCores<f64>) {
        let n = cp0.num_coeffs();
        let basis: Vec<ContractedSparseCores<f64>> = (0..n)
            .map(|j| {
                let mut e = cp0.clone();
                e.for_each_coeff_mut(|c| *c = 0.0);
                e.set_coeff(j, 1.0);
                e
            })
            .collect();
        let images: Vec<ContractedSparseCores<f64>> = basis
            .iter()
            .map(|e| h.two_site_sketched_hamiltonian(e, fl0, fr1))
            .collect();
        let mut m = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                *m.get_mut(i, j) = basis[i].dot(&images[j]);
            }
        }
        let (ev, y) = dense_self_adjoint_eigen(m.as_ref());
        let mut u_exact = cp0.clone();
        u_exact.for_each_coeff_mut(|c| *c = 0.0);
        for i in 0..n {
            u_exact.set_coeff(i, *y.get(i, 0));
        }
        (ev, u_exact)
    }

    #[test]
    fn tms_a10_local_lowest_dispatches_every_guess() {
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
        assert_eq!(thetas.len(), nroots, "tms_a10: root count");
        assert_eq!(us.len(), nroots, "tms_a10: vector count");
        for i in 0..nroots {
            assert!(
                (thetas[i] - ev[i]).abs() <= 1e-10,
                "tms_a10: θ[{i}] = {} vs the dense local level {} (Δ = {:.3e})",
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
                    "tms_a10: ⟨u_{i}, u_{j}⟩ = {} (want {want})",
                    us[i].dot(&us[j])
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "different two-site shape")]
    fn tms_a10_local_lowest_rejects_a_guess_of_another_shape() {
        let (h, fl0, fr1, cp0) = fixture();
        let other = block_at(1);
        let _ = local_lowest_2s(&h, &fl0, &fr1, &[cp0, other], 2, 1e-8, 16, 4);
    }
}

#[cfg(test)]
mod l1_stack_lifetime_tests {
    use super::*;
    use crate::qn::QnSpace;
    use crate::test_util::{build_h_matvec, randomized_state, seeded};

    #[test]
    fn sweeps_consume_the_stack_they_read() {
        let space = QnSpace::new(6, 2, 2);
        let h = build_h_matvec(space);
        let mut rng = seeded(0x11_0001);
        let mut x = randomized_state::<f64, _>(space, 1, 3, &mut rng);
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
