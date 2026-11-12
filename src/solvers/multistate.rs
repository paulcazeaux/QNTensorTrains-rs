// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! State-average MALS: `n` eigenstates carried by ONE train that shares every core but the
//! orthogonality centre.

use crate::core::SparseCore;
use crate::core::factor::{Sketch, mul_sketch_right};
use crate::core::twosite::{
    ContractedSparseCores, LoewdinInfo, TruncInfo, contract, factor_qc, factor_svd_average,
    noise_rng, weighted_noise_sum,
};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::matvec::{noise_density_matrix_streamed, onesite_workspace};
use crate::hamiltonian::rayleigh_quotient;
use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::solvers::InnerProductVector;
use crate::solvers::als::{INNER_TOL_CAP, boundary_idsketch, right_to_left_sketching};
use crate::solvers::davidson::{DAVIDSON_MAXSUBSPACE, DAVIDSON_RESTART_KEEP};
use crate::solvers::davidson_block::home_roots;
use crate::solvers::interface::clamp_hi_first;
use crate::solvers::mals::{MalsOptions, NoiseType, SweepTrunc, local_lowest_2s};
use crate::solvers::overlap_env::overlap_envs;
use crate::vector::TtVector;

/// Block-Davidson subspace ceiling for an `n`-root state-average solve:
/// `max(DAVIDSON_MAXSUBSPACE, 8n)`.
pub fn sa_maxsubspace(n: usize) -> usize {
    DAVIDSON_MAXSUBSPACE.max(8 * n)
}

/// Thick-restart keep count for an `n`-root state-average solve: `2n`, clamped below
/// `maxsubspace`; `1` at `n = 1` (the bit-identity carve-out — `DAVIDSON_RESTART_KEEP`).
pub fn sa_restart_keep(n: usize, maxsubspace: usize) -> usize {
    if n == 1 {
        DAVIDSON_RESTART_KEEP
    } else {
        assert!(
            maxsubspace >= 2 * n,
            "sa_restart_keep: maxsubspace = {maxsubspace} < 2·nroots = {} (the block solver needs \
             room for the {n} Ritz vectors and their corrections)",
            2 * n
        );
        (2 * n).min(maxsubspace - 1)
    }
}

/// State-average MALS options: the single-state [`MalsOptions`] plus what only a multi-root sweep
/// has to say.
#[derive(Clone, Debug)]
pub struct StateAverageOptions {
    /// The sweep schedule: `eps`, `bond_dims`, `noise`, `noise_type`, `max_iter`, `verbose`,
    /// `conv_tol` — read exactly as [`crate::solvers::mals::mals`] reads them. `seed` is
    /// also the seed [`MultiTt::new`] draws its distinct centre perturbations from.
    pub mals: MalsOptions,
    /// Per-root weights in label order, `None` = `1/n` each. Finite, strictly positive,
    /// NON-INCREASING and summing to 1 (1e-12) — all four load-bearing for the pooled chop's
    /// ABSOLUTE tolerance (`Σᵢ wᵢ‖Bᵢ‖²` is the trace of ρ).
    pub weights: Option<Vec<f64>>,
    /// Follow each root by maximum overlap (`home_roots`) instead of relabelling by energy at
    /// every bond. ANCHORED: the first bond of iteration 1 always labels by ascending Ritz value,
    /// so the labels are reproducible; from the next bond on they follow the states. Default
    /// `false` (label by energy everywhere).
    pub homing: bool,
    /// Löwdin-re-orthonormalize the `n` truncated centres at every bond (default `true`). `false`
    /// returns the raw sub-unit projections and only REPORTS the centre Gram.
    pub reorthonormalize: bool,
    /// Relative size of the random perturbation [`MultiTt::new`] adds to centres `1..n` so the
    /// block solver gets `n` distinct guesses (default `0.1`). Centre 0 is never perturbed — that
    /// is what makes `nroots = 1` the single-state start bit for bit.
    pub scale: f64,
    /// Block-Davidson subspace ceiling; `None` = [`sa_maxsubspace`].
    pub maxsubspace: Option<usize>,
    /// Block-Davidson thick-restart keep count; `None` = [`sa_restart_keep`].
    pub restart_keep: Option<usize>,
}

impl Default for StateAverageOptions {
    fn default() -> Self {
        StateAverageOptions {
            mals: MalsOptions::default(),
            weights: None,
            homing: false,
            reorthonormalize: true,
            scale: 0.1,
            maxsubspace: None,
            restart_keep: None,
        }
    }
}

/// One outer iteration's worth of diagnostics (a forward + a back sweep) — the multi-root twin of
/// [`crate::solvers::mals::MalsSweep`]. Not `Copy`: it owns the per-root vectors.
#[derive(Clone, Debug)]
pub struct SaSweep {
    /// The last bond's Ritz values, in LABEL order (ascending under the default
    /// `homing = false`).
    pub energies: Vec<f64>,
    /// The two-site Ritz values, BEFORE that bond's truncation, at the bond of this iteration
    /// (forward and back sweep) whose weighted sum `Σᵢ wᵢ λᵢ` is lowest — the "sweep energy"
    /// Block reports (its running minimum over block steps; equal weights ⇒ its plain sum). Lower
    /// than the truncated state's energies [`SaSweep::energies`] by roughly the truncation cost at
    /// that bond; reported for comparison with such numbers, never used by the iteration.
    pub lowest_energies: Vec<f64>,
    /// `Σᵢ wᵢ λᵢ` — the quantity the sweep actually minimizes.
    pub energy: f64,
    /// `maxᵢ |λᵢ − λᵢ_prev| / |λᵢ|` — the per-root relative change, the tell a single averaged
    /// energy hides when one root is still moving.
    pub max_dev: f64,
    /// Largest `LoewdinInfo::overlap_dev` over this iteration's bonds: how far the truncated
    /// roots drifted from orthogonality before the Löwdin step put them back.
    pub overlap_dev: f64,
    /// Largest POOLED discarded weight over this iteration's bonds.
    pub discarded_weight: f64,
    /// Per root, the largest discarded weight over this iteration's bonds. A lightly weighted
    /// root can lose far more than the pooled number ever shows — that is what this vector is
    /// for. NOTE the maxima are taken over possibly DIFFERENT bonds, so the per-bond identity
    /// `Σᵢ wᵢ dwᵢ = pooled` does not survive into these history entries; only
    /// `maxᵢ dwᵢ ≥ pooled` does. The SU(2) metric caveat on [`SaSweep::discarded_weight`] applies
    /// entry by entry here too.
    pub discarded_weights: Vec<f64>,
    /// Largest total (summed-over-sectors) bond dimension reached.
    pub max_bond_dim: usize,
    /// The rank cap in force this iteration (`0` = uncapped).
    pub bond_dim: usize,
    /// The noise amplitude in force this iteration.
    pub noise: f64,
    /// Local operator applies over both sweeps (the block solver's `numops`).
    pub numops: usize,
}

/// [`state_average_mals_with_history`]'s record: the final per-root energies plus every outer
/// iteration.
#[derive(Clone, Debug)]
pub struct SaHistory {
    /// The converged Ritz values in label order (the initial Rayleigh quotients when
    /// `max_iter == 0` — the `mals` convention).
    pub lambdas: Vec<f64>,
    pub sweeps: Vec<SaSweep>,
}

/// `n` TT states sharing every core but the orthogonality centre.
#[derive(Clone, Debug)]
pub struct MultiTt<T: Scalar> {
    /// The shared train, carrying `centers[0]` at `core_position`.
    pub base: TtVector<T>,
    /// The `n` centre cores at `core_position`, in label order.
    pub centers: Vec<SparseCore<T>>,
    /// Per-root weights, in label order.
    pub weights: Vec<f64>,
    /// Site the centres sit at.
    pub core_position: usize,
}

impl<T: Scalar> MultiTt<T> {
    /// Seed `n` states from one start: right-orthogonalize a clone of `x0`, keep its core 0 as
    /// centre 0 UNCHANGED, and perturb centres `1..n` by `scale·‖core(0)‖·Rᵢ/‖Rᵢ‖` with `Rᵢ` an
    /// i.i.d. standard-normal fill of the core's coefficient basis, drawn in canonical order from
    /// ONE `Xoshiro256PlusPlus::seed_from_u64(seed)` stream.
    ///
    /// # Panics
    /// When any two-site local space of the start holds fewer than `n` coefficients: a rank-1
    /// determinant/CSF cannot seed a state average (campaign decision 1), and the pointed error
    /// here is far more useful than the block solver's later short return.
    pub fn new(x0: &TtVector<T>, n: usize, seed: u64, scale: f64) -> Self {
        assert!(n >= 1, "MultiTt::new: nroots must be ≥ 1 (got {n})");
        let mut base = x0.clone();
        if base.ortho != Some(0) {
            base.right_orthogonalize(true);
        }
        let c0 = base.core(0).clone();
        let mut rng = noise_rng(seed);
        let s = scale * c0.norm();
        let mut centers = Vec::with_capacity(n);
        centers.push(c0.clone());
        for _ in 1..n {
            let mut r = c0.clone();
            r.for_each_coeff_mut(|z| *z = T::sample_standard(&mut rng));
            let rn = r.norm();
            let mut c = c0.clone();
            if rn > 0.0 {
                c.axpy(T::from_f64(s / rn), &r);
            }
            centers.push(c);
        }
        let m = MultiTt {
            base,
            centers,
            weights: vec![1.0 / n as f64; n],
            core_position: 0,
        };
        let d = m.base.space.d;
        for k in 0..d.saturating_sub(1) {
            let dim = contract(m.base.core(k), m.base.core(k + 1)).num_coeffs();
            assert!(
                dim >= n,
                "MultiTt::new: the two-site local space at pair ({k}, {}) holds {dim} \
                 coefficient(s), fewer than the {n} states requested — a rank-1 determinant/CSF \
                 start cannot seed a state average; enrich x0 first (a noisy MALS iteration, or \
                 `crate::vector::perturbation`)",
                k + 1,
            );
        }
        m.check();
        m
    }

    /// Number of states carried.
    pub fn nstates(&self) -> usize {
        self.centers.len()
    }

    /// Assert the container invariant (see the type doc). Runs at the end of
    /// [`new`](Self::new) and at every driver entry.
    ///
    /// # Panics
    /// With a `MultiTt: …` message naming the clause that broke.
    pub fn check(&self) {
        let n = self.centers.len();
        assert!(n >= 1, "MultiTt: no states (centers is empty)");
        let d = self.base.space.d;
        assert!(
            self.core_position < d,
            "MultiTt: core_position {} outside 0..{d}",
            self.core_position
        );
        assert_eq!(
            self.base.ortho,
            Some(self.core_position),
            "MultiTt: the shared train must be orthogonal with ortho == Some({}) — it was \
             re-gauged behind the container's back (found {:?})",
            self.core_position,
            self.base.ortho,
        );
        for (i, c) in self.centers.iter().enumerate() {
            assert!(
                c.k == self.core_position
                    && c.row_ranks == self.centers[0].row_ranks
                    && c.col_ranks == self.centers[0].col_ranks,
                "MultiTt: center {i} (site {}, ranks {:?}/{:?}) does not have center 0's shape \
                 (site {}, ranks {:?}/{:?})",
                c.k,
                c.row_ranks,
                c.col_ranks,
                self.core_position,
                self.centers[0].row_ranks,
                self.centers[0].col_ranks,
            );
        }
        assert!(
            core_value_eq(self.base.core(self.core_position), &self.centers[0]),
            "MultiTt: base.core({}) is not centers[0] — the base was mutated behind the \
             container's back",
            self.core_position,
        );
        validate_weights(&self.weights, n, "MultiTt");
    }

    /// Seed a container from `n` states that already exist — an excited-state ladder, say — so a
    /// state average can start from them instead of from one perturbed train.
    ///
    /// # Panics
    /// On an empty list, on mixed spaces, on weights that break clause B1, and (hazard 5) when a
    /// two-site local space of the sum holds fewer than `n` coefficients.
    pub fn from_states(xs: &[TtVector<T>], weights: Option<Vec<f64>>) -> Self {
        assert!(
            !xs.is_empty(),
            "MultiTt::from_states: needs at least one state (the list is empty)"
        );
        let n = xs.len();
        let space = xs[0].space;
        for (i, x) in xs.iter().enumerate() {
            assert_eq!(
                x.space, space,
                "MultiTt::from_states: xs[{i}] lives in a different space than xs[0]"
            );
        }
        let w = weights.unwrap_or_else(|| vec![1.0 / n as f64; n]);
        validate_weights(&w, n, "MultiTt::from_states");

        let mut base = xs[0].clone();
        for x in &xs[1..] {
            base = &base + x;
        }
        base.right_orthogonalize(false);

        let centers: Vec<SparseCore<T>> = xs
            .iter()
            .map(|x| {
                let env = overlap_envs(&base, x);
                let r1 = env.r[1].as_ref().expect("overlap_envs builds r[1..=d]");
                mul_sketch_right(x.core(0), r1)
            })
            .collect();
        base.set_core(centers[0].clone());

        let m = MultiTt {
            base,
            centers,
            weights: w,
            core_position: 0,
        };
        let d = m.base.space.d;
        for k in 0..d.saturating_sub(1) {
            let dim = contract(m.base.core(k), m.base.core(k + 1)).num_coeffs();
            assert!(
                dim >= n,
                "MultiTt::from_states: the two-site local space at pair ({k}, {}) holds {dim} \
                 coefficient(s), fewer than the {n} states supplied — the states span fewer \
                 directions than they claim (duplicates, or determinant/CSF trains)",
                k + 1,
            );
        }
        m.check();
        m
    }

    /// Gram matrix against another container: `G[i, j] = ⟨self.state(i), other.state(j)⟩`, of
    /// shape `self.nstates() × other.nstates()`, without materializing a state. The two
    /// containers need not share a base or a centre position — this is the block inner product
    /// with the open leg at any site.
    ///
    /// # Panics
    /// On a space mismatch, or when either base lacks unit boundary ranks.
    pub fn gram(&self, other: &Self) -> faer::Mat<T> {
        let grams = crate::ops::gram_fold(
            &self.base,
            Some((self.core_position, &self.centers)),
            &other.base,
            Some((other.core_position, &other.centers)),
        );
        let (nx, ny) = (self.nstates(), other.nstates());
        faer::Mat::from_fn(nx, ny, |i, j| {
            let g = &grams[i * ny + j];
            assert_eq!(
                (g.nrows(), g.ncols()),
                (1, 1),
                "MultiTt::gram: unit boundary ranks"
            );
            *g.get(0, 0)
        })
    }

    /// The `n` states as INDEPENDENT trains: `base` cloned with its centre replaced. Never an
    /// alias of the container's cores (hazard 1) — a caller may round, orthogonalize or mutate
    /// what it gets back.
    pub fn states(&self) -> Vec<TtVector<T>> {
        (0..self.nstates()).map(|i| self.state(i)).collect()
    }

    /// State `i` alone; see [`states`](Self::states).
    pub fn state(&self, i: usize) -> TtVector<T> {
        let mut x = self.base.clone();
        x.set_core(self.centers[i].clone());
        x.ortho = Some(self.core_position);
        x
    }
}

fn core_value_eq<T: Scalar>(a: &SparseCore<T>, b: &SparseCore<T>) -> bool {
    if a.k != b.k || a.row_ranks != b.row_ranks || a.col_ranks != b.col_ranks {
        return false;
    }
    let n = a.num_coeffs();
    if n != b.num_coeffs() {
        return false;
    }
    (0..n).all(|i| a.coeff(i) == b.coeff(i))
}

pub(crate) fn validate_weights(w: &[f64], n: usize, who: &str) {
    assert!(
        w.len() == n,
        "{who}: {} weights for {n} states — one per root, in label order",
        w.len(),
    );
    assert!(
        w.iter().all(|a| a.is_finite() && *a > 0.0),
        "{who}: weights must be finite and strictly positive (got {w:?}) — a zero-weight state is \
         truncated away with no trace in the ledger",
    );
    assert!(
        w.windows(2).all(|p| p[0] >= p[1]),
        "{who}: weights must be non-increasing in label order (got {w:?})",
    );
    let sum: f64 = w.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-12,
        "{who}: weights must sum to 1 (got {sum}) — the pooled chop's tolerance is ABSOLUTE and \
         `Σwᵢ‖Bᵢ‖²` is the trace of ρ",
    );
}

/// Rank-adaptive state-average two-site search: the `nroots` lowest eigenpairs sharing one train.
/// Seeds a [`MultiTt`] from `x0` ([`MultiTt::new`], with `opts.mals.seed` and `opts.scale`),
/// sweeps it, and returns `(λs, states)`.
pub fn state_average_mals<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    nroots: usize,
    opts: &StateAverageOptions,
) -> (Vec<f64>, Vec<TtVector<T>>) {
    let (hist, xs) = state_average_mals_with_history(h, x0, nroots, opts);
    (hist.lambdas, xs)
}

/// [`state_average_mals`] keeping the per-iteration diagnostics. With a rank ramp or several
/// roots, the per-root `max_dev` and the per-state `discarded_weights` are what say whether the
/// run converged — a single averaged energy cannot.
pub fn state_average_mals_with_history<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    nroots: usize,
    opts: &StateAverageOptions,
) -> (SaHistory, Vec<TtVector<T>>) {
    assert!(
        nroots >= 1,
        "state_average_mals: nroots must be ≥ 1 (got {nroots})"
    );
    let mut m = MultiTt::new(x0, nroots, opts.mals.seed, opts.scale);
    let hist = state_average_mals_mut(h, &mut m, opts);
    let xs = m.states();
    (hist, xs)
}

/// In-place state-average MALS on a caller-owned container. The container must sit at core
/// position 0 (fresh from [`MultiTt::new`], or after a completed run) — shared cores are never
/// re-orthogonalized here.
pub fn state_average_mals_mut<T: Scalar>(
    h: &SparseHamiltonian<T>,
    m: &mut MultiTt<T>,
    opts: &StateAverageOptions,
) -> SaHistory {
    let n = m.nstates();
    if let Some(w) = &opts.weights {
        validate_weights(w, n, "state_average_mals");
        m.weights = w.clone();
    }
    m.check();
    assert!(
        !opts.mals.bond_dims.is_empty(),
        "MalsOptions::bond_dims must be non-empty (use [0] for uncapped)"
    );
    assert!(
        !opts.mals.noise.is_empty(),
        "MalsOptions::noise must be non-empty (use [0.0] for a noise-free run)"
    );
    assert!(
        opts.mals.noise.iter().all(|a| a.is_finite() && *a >= 0.0),
        "MalsOptions::noise entries must be finite and non-negative"
    );
    assert!(
        opts.mals.noise_type != NoiseType::Random,
        "state_average_mals: state-average sweeps define only Perturbative noise — \
         NoiseType::Random perturbs the two-site STATE before its SVD, which has no per-state \
         density matrix to weight and is not variational"
    );
    assert_eq!(
        m.core_position, 0,
        "state_average_mals_mut: the container must sit at core position 0 (fresh from \
         MultiTt::new or after a completed run); shared cores are never re-orthogonalized here"
    );
    let ms = opts.maxsubspace.unwrap_or_else(|| sa_maxsubspace(n));
    let rk = opts.restart_keep.unwrap_or_else(|| sa_restart_keep(n, ms));

    let d = m.base.space.d;
    let mut sweeps: Vec<SaSweep> = Vec::with_capacity(opts.mals.max_iter);
    let mut fr: Vec<Option<Sketch<T>>> = right_to_left_sketching(h, &m.base)
        .into_iter()
        .map(Some)
        .collect();
    let mut lambdas: Vec<f64> = (0..n)
        .map(|i| rayleigh_quotient(h, &m.state(i)).re_s())
        .collect();
    let mut lambda_avg = weighted_sum(&m.weights, &lambdas);
    let mut r = 1.0f64;

    for it in 0..opts.mals.max_iter {
        let bond_dim = opts.mals.bond_dims[it.min(opts.mals.bond_dims.len() - 1)];
        let noise = opts.mals.noise[it.min(opts.mals.noise.len() - 1)];
        let inner_tol = clamp_hi_first(0.01 * r, opts.mals.eps, INNER_TOL_CAP);
        let ep = if d > 1 {
            opts.mals.eps / ((d - 1) as f64).sqrt()
        } else {
            opts.mals.eps
        };
        let home_from = if it == 0 { 1 } else { 0 };

        let mut fwd = sa_forward_sweep(
            h,
            m,
            &mut fr,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.homing,
            home_from,
            opts.reorthonormalize,
            ms,
            rk,
        );
        if opts.mals.verbose {
            println!(
                "SA-MALS iter {it} forward sweep: energies = {:?}",
                fwd.lambdas
            );
        }
        debug_assert!(
            fr.iter().all(Option::is_none),
            "the forward sweep must consume the whole right stack"
        );
        let back = sa_back_sweep(
            h,
            m,
            &mut fwd.env,
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.homing,
            0,
            opts.reorthonormalize,
            ms,
            rk,
        );
        debug_assert!(
            fwd.env.iter().all(Option::is_none),
            "the back sweep must consume the whole left stack"
        );
        fr = back.env;

        let trunc = fwd.trunc.merge(back.trunc);
        let lambdas_n = back.lambdas;
        let lambda_avg_n = weighted_sum(&m.weights, &lambdas_n);
        let mut max_dev = 0.0f64;
        for i in 0..n {
            max_dev = max_dev.max((lambdas[i] - lambdas_n[i]).abs() / lambdas_n[i].abs());
        }
        let r_avg = (lambda_avg - lambda_avg_n).abs() / lambda_avg_n.abs();
        r = r_avg.max(max_dev);
        lambdas = lambdas_n;
        lambda_avg = lambda_avg_n;
        let dws: Vec<f64> = (0..n)
            .map(|i| fwd.dws[i].max(back.dws[i]))
            .collect::<Vec<_>>();
        let lowest_energies = lower_snapshot(&m.weights, &fwd.lowest, &back.lowest);
        sweeps.push(SaSweep {
            energies: lambdas.clone(),
            lowest_energies,
            energy: lambda_avg,
            max_dev,
            overlap_dev: fwd.overlap_dev.max(back.overlap_dev),
            discarded_weight: trunc.max_discarded,
            discarded_weights: dws,
            max_bond_dim: trunc.max_kept,
            bond_dim,
            noise,
            numops: fwd.numops + back.numops,
        });
        if opts.mals.verbose {
            println!(
                "SA-MALS iter {it} back sweep: energies = {lambdas:?}, average = {lambda_avg}, \
                 relative improvement = {r}, discarded weight = {:.3e}, M = {}, noise = \
                 {noise:.1e}",
                trunc.max_discarded, trunc.max_kept
            );
        }
        if r < opts.mals.convergence_tol() && inner_tol <= opts.mals.eps && noise == 0.0 {
            break;
        }
    }
    if opts.homing && lambdas.windows(2).any(|p| p[1] < p[0]) {
        eprintln!(
            "state_average_mals: homed root labels are not energy-sorted at the final bond \
             (weights follow labels): {lambdas:?}"
        );
    }
    SaHistory { lambdas, sweeps }
}
fn lower_snapshot(w: &[f64], a: &[f64], b: &[f64]) -> Vec<f64> {
    if b.is_empty() || (!a.is_empty() && weighted_sum(w, a) <= weighted_sum(w, b)) {
        a.to_vec()
    } else {
        b.to_vec()
    }
}

fn weighted_sum(w: &[f64], l: &[f64]) -> f64 {
    w.iter().zip(l.iter()).map(|(a, b)| a * b).sum()
}

struct SaSweepOut<T: Scalar> {
    lambdas: Vec<f64>,
    lowest: Vec<f64>,
    env: Vec<Option<Sketch<T>>>,
    trunc: SweepTrunc,
    dws: Vec<f64>,
    overlap_dev: f64,
    numops: usize,
}

#[allow(clippy::too_many_arguments)]
fn sa_forward_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    m: &mut MultiTt<T>,
    fr: &mut [Option<Sketch<T>>],
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    homing: bool,
    home_from: usize,
    reorthonormalize: bool,
    maxsubspace: usize,
    restart_keep: usize,
) -> SaSweepOut<T> {
    assert_eq!(
        m.core_position, 0,
        "state_average_mals_mut: the forward sweep expects the container at core position 0, \
         found {} (shared cores are never re-orthogonalized here)",
        m.core_position
    );
    debug_assert_eq!(m.base.ortho, Some(0), "SA forward sweep expects center 0");
    let d = m.base.space.d;
    let n = m.nstates();
    let mut fl: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fl[0] = Some(boundary_idsketch(&m.base, 0, 1));
    fr[0] = None;
    let mut out = SaSweepOut {
        lambdas: vec![0.0; n],
        lowest: Vec::new(),
        env: Vec::new(),
        trunc: SweepTrunc::default(),
        dws: vec![0.0; n],
        overlap_dev: 0.0,
        numops: 0,
    };

    for k in 0..(d - 1) {
        let cp0s: Vec<ContractedSparseCores<T>> = (0..n)
            .map(|i| contract(&m.centers[i], m.base.core(k + 1)))
            .collect();
        let (thetas, us, info) = local_lowest_2s(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &cp0s,
            n,
            inner_tol,
            maxsubspace,
            restart_keep,
        );
        out.numops += info.numops;
        let (thetas, us) = if homing && k >= home_from {
            home_relabel(&cp0s, thetas, us)
        } else {
            (thetas, us)
        };
        let (shared, s, mut centers, infos, pooled, loewdin) = split_two_site_average(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &us,
            &m.weights,
            ep,
            max_rank,
            noise,
            true,
            reorthonormalize,
        );
        fr[k + 1] = None;
        out.trunc.absorb(pooled);
        for (i, ti) in infos.iter().enumerate() {
            out.dws[i] = out.dws[i].max(ti.discarded_weight);
        }
        out.overlap_dev = out.overlap_dev.max(loewdin.overlap_dev);
        for c in centers.iter_mut() {
            c.fold_sketch_left_mut(&s);
        }
        m.base.set_cores(shared, centers[0].clone());
        m.base.ortho = Some(k + 1);
        m.centers = centers;
        m.core_position = k + 1;
        if out.lowest.is_empty()
            || weighted_sum(&m.weights, &thetas) < weighted_sum(&m.weights, &out.lowest)
        {
            out.lowest = thetas.clone();
        }
        out.lambdas = thetas;
        if k < d - 2 {
            let ws = onesite_workspace(h, m.base.core(k));
            let next = h.sketching_step_right(&m.base, fl[k].as_ref().unwrap(), &ws);
            fl[k + 1] = Some(next);
        }
    }
    out.env = fl;
    out
}

#[allow(clippy::too_many_arguments)]
fn sa_back_sweep<T: Scalar>(
    h: &SparseHamiltonian<T>,
    m: &mut MultiTt<T>,
    fl: &mut [Option<Sketch<T>>],
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    homing: bool,
    home_from: usize,
    reorthonormalize: bool,
    maxsubspace: usize,
    restart_keep: usize,
) -> SaSweepOut<T> {
    let d = m.base.space.d;
    let n = m.nstates();
    assert_eq!(
        m.core_position,
        d - 1,
        "state_average_mals_mut: the back sweep expects the container at core position {}, found \
         {} (the single-state driver's move_core fallback is forbidden on shared cores)",
        d - 1,
        m.core_position
    );
    let mut fr: Vec<Option<Sketch<T>>> = (0..d).map(|_| None).collect();
    fr[d - 1] = Some(boundary_idsketch(&m.base, d, 1));
    fl[d - 1] = None;
    let mut out = SaSweepOut {
        lambdas: vec![0.0; n],
        lowest: Vec::new(),
        env: Vec::new(),
        trunc: SweepTrunc::default(),
        dws: vec![0.0; n],
        overlap_dev: 0.0,
        numops: 0,
    };

    for k in (0..(d - 1)).rev() {
        let cp0s: Vec<ContractedSparseCores<T>> = (0..n)
            .map(|i| contract(m.base.core(k), &m.centers[i]))
            .collect();
        let (thetas, us, info) = local_lowest_2s(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &cp0s,
            n,
            inner_tol,
            maxsubspace,
            restart_keep,
        );
        out.numops += info.numops;
        let (thetas, us) = if homing && k >= home_from {
            home_relabel(&cp0s, thetas, us)
        } else {
            (thetas, us)
        };
        let (shared, s, mut centers, infos, pooled, loewdin) = split_two_site_average(
            h,
            fl[k].as_ref().unwrap(),
            fr[k + 1].as_ref().unwrap(),
            &us,
            &m.weights,
            ep,
            max_rank,
            noise,
            false,
            reorthonormalize,
        );
        fl[k] = None;
        out.trunc.absorb(pooled);
        for (i, ti) in infos.iter().enumerate() {
            out.dws[i] = out.dws[i].max(ti.discarded_weight);
        }
        out.overlap_dev = out.overlap_dev.max(loewdin.overlap_dev);
        for c in centers.iter_mut() {
            c.fold_sketch_right_mut(&s);
        }
        m.base.set_cores(centers[0].clone(), shared);
        m.base.ortho = Some(k);
        m.centers = centers;
        m.core_position = k;
        if out.lowest.is_empty()
            || weighted_sum(&m.weights, &thetas) < weighted_sum(&m.weights, &out.lowest)
        {
            out.lowest = thetas.clone();
        }
        out.lambdas = thetas;
        if k > 0 {
            let ws = onesite_workspace(h, m.base.core(k + 1));
            let next = h.sketching_step_left(&m.base, fr[k + 1].as_ref().unwrap(), &ws);
            fr[k] = Some(next);
        }
    }
    out.env = fr;
    out
}

fn home_relabel<V: InnerProductVector>(
    prev: &[V],
    thetas: Vec<f64>,
    us: Vec<V>,
) -> (Vec<f64>, Vec<V>) {
    let (perm, _gram) = home_roots(prev, &us);
    let mut slots: Vec<Option<V>> = us.into_iter().map(Some).collect();
    let mut new_us = Vec::with_capacity(perm.len());
    let mut new_thetas = Vec::with_capacity(perm.len());
    for &j in &perm {
        new_us.push(slots[j].take().expect("home_roots returns a permutation"));
        new_thetas.push(thetas[j]);
    }
    (new_thetas, new_us)
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn split_two_site_average<T: Scalar>(
    h: &SparseHamiltonian<T>,
    fl: &Sketch<T>,
    fr: &Sketch<T>,
    us: &[ContractedSparseCores<T>],
    w: &[f64],
    ep: f64,
    max_rank: usize,
    noise: f64,
    forward: bool,
    reorthonormalize: bool,
) -> (
    SparseCore<T>,
    Sketch<T>,
    Vec<SparseCore<T>>,
    Vec<TruncInfo>,
    TruncInfo,
    LoewdinInfo,
) {
    let nd = (noise > 0.0).then(|| {
        let nds: Vec<_> = us
            .iter()
            .map(|cp| {
                let (xk, xk1) = factor_qc(cp);
                noise_density_matrix_streamed(h, if forward { fl } else { fr }, &xk, &xk1, forward)
            })
            .collect();
        if nds.len() == 1 {
            nds.into_iter().next().unwrap()
        } else {
            weighted_noise_sum(&nds, w, &us[0], forward)
        }
    });
    factor_svd_average(
        us,
        w,
        nd.as_ref(),
        noise,
        ep,
        Pooled(max_rank),
        forward,
        reorthonormalize,
    )
}

mod bridge;

#[cfg(test)]
mod bridge_tests;
#[cfg(test)]
mod gram_tests;
