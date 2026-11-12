// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The excited-state ladder: a state average followed by one state-specific sweep and one
//! penalized Riemannian polish per root, bottom-up, with orthogonality / ordering / convergence
//! gates.

use crate::scalar::Scalar;

/// Where each state-specific sweep starts.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SsStart {
    /// Root `i`'s own state-average state `xsa[i]` (the default): the state average has already
    /// separated the roots, so each sweep starts in the right basin.
    #[default]
    Sa,
}

/// Everything the ladder recorded, in root order.
#[derive(Clone, Debug)]
pub struct ExcitedInfo<V> {
    /// The returned energies — the Rayleigh quotients of the returned roots. Repeats the first
    /// element of the driver's return triple.
    pub e: Vec<f64>,
    /// The state-specific sweep's own final energy for each root, BEFORE the polish
    /// (`SsHistory::lambda`).
    pub ss_energies: Vec<f64>,
    /// The penalized functional `J` at the state-specific state.
    pub j_ss: Vec<f64>,
    /// The penalized functional `J` at the returned (polished, trimmed) root, at the SAME weight.
    /// `j_pol[i] ≤ j_ss[i]` holds only while `passes[i] == 0`: a trim discards Schmidt mass and
    /// the re-polish restarts from a different point.
    pub j_pol: Vec<f64>,
    /// The state-average stage's history (its `lambdas` are the state-average energies in label
    /// order).
    pub sa_history: crate::solvers::multistate::SaHistory,
    /// One state-specific history per root.
    pub ss_history: Vec<crate::solvers::ssmals::SsHistory>,
    /// The normalized `(m+n)×(m+n)` Gram of `[lower; roots]`, ABSOLUTE values
    /// (`|⟨ψ̂ₐ, ψ̂_b⟩|`), row-major in that order.
    pub overlaps: Vec<Vec<f64>>,
    /// The largest off-diagonal entry of [`Self::overlaps`] (`0.0` for a one-state ladder).
    pub overlap_dev: f64,
    /// How many rungs of `Λ = [sort(E_lower); e]` sit below the rung under them by more than
    /// `10·eps`.
    pub swaps: usize,
    /// Those rungs' 0-based indices INTO `Λ`; index `i ≥ m` is root `i − m`. Labels are never
    /// reordered.
    pub swapped: Vec<usize>,
    /// The exact penalized gate `‖P((H − J)x + w Σⱼ oⱼ ψ̂ⱼ)‖` at each returned root.
    /// `f64::NAN` when `polish = false` (there is no critical point to certify).
    pub gradnorms: Vec<f64>,
    /// The exact UNPENALIZED gate `‖P((H − λⱼ)ψ̂ⱼ)‖` of each supplied lower state, in the
    /// caller's order. Diagnostic, not a gate: it is warned about, never refused.
    pub lower_gradnorms: Vec<f64>,
    /// The penalty weight each root's polish ran at (`0.0` when that root had no lower states).
    pub penalties: Vec<f64>,
    /// Trim passes performed per root, `0..=trim_passes`.
    pub passes: Vec<usize>,
    /// `gradnorms[i] > grad_gate` (false when `polish = false`).
    pub stalled: Vec<bool>,
    /// `ss_converged[i] ∧ !stalled[i]`.
    pub converged: Vec<bool>,
    /// Whether stage 0's enrichment iteration ran.
    pub enriched: bool,
    /// `all(converged) ∧ overlap_dev ≤ overlap_tol ∧ swaps == 0`. Read the module doc before
    /// treating it as a completeness statement — it is not one.
    pub licence: bool,
    /// The state-average ladder, when `keep_sa` asked for it. Independent clones; nothing here
    /// aliases a returned root.
    pub sa_states: Option<Vec<V>>,
}

pub(crate) fn default_ramp(rank: usize) -> Vec<usize> {
    if rank == 0 {
        return vec![0];
    }
    vec![
        (rank / 4).max(6),
        (rank / 2).max(10),
        (3 * rank / 4).max(14),
        rank,
        rank,
        rank,
    ]
}

pub(crate) fn ladder_swaps(ladder: &[f64], eps: f64) -> (usize, Vec<usize>) {
    let thr = 10.0 * eps;
    let mut swapped = Vec::new();
    for i in 1..ladder.len() {
        if ladder[i] < ladder[i - 1] - thr {
            swapped.push(i);
        }
    }
    (swapped.len(), swapped)
}

macro_rules! impl_excited_states {
    (
        $(#[$doc:meta])*
        $name:ident, $ham:ident, $vec:ident, $multi:ident,
        $space:ty, $ropts:ty,
        $mals_mut:path, $sa_mut:path, $ss_hist:path,
        $cg_pen:path, $gate_pen:path,
        $tbases:path, $normalize:path,
        $rayleigh:path, $dot:path, $min_schmidt:path, $contract:path
    ) => {
        /// Options of the excited-state ladder (see [`crate::solvers::excited`] for the stages).
        #[derive(Clone, Debug)]
        pub struct ExcitedOpts {
            /// How many roots the ladder returns, counting the ground state. `≥ 1`.
            pub nroots: usize,
            /// State-average weights in label order; `None` = `1/n` each. Validated by the
            /// state-average driver (finite, positive, non-increasing, summing to 1).
            pub weights: Option<Vec<f64>>,
            /// The bond-rank cap of the state-specific sweeps and the polish manifold; `0` = full
            /// rank. Pooled total bond dimension (`Pooled`); on SU(2) reduced multiplets.
            pub rank: usize,
            /// The state-average ramp; `None` = the default ramp of `rank`. Pooled total bond
            /// dimension (`Pooled`); on SU(2) reduced multiplets.
            pub bond_dims: Option<Vec<usize>>,
            /// Truncation tolerance of every sweep, and the ABSOLUTE scale of the swap scan
            /// (`10·eps`).
            pub eps: f64,
            /// Outer-iteration budget of every sweep (state-average and state-specific alike).
            pub max_iter: usize,
            /// The state-average noise schedule; `noise[0]` is also the enrichment amplitude.
            pub noise: Vec<f64>,
            /// The state-specific noise schedule (usually `[0.0]`: the rungs start converged).
            pub ss_noise: Vec<f64>,
            /// Noise kind for both sweeps. `Random` is refused — the state-average truncation
            /// defines only the perturbative density-matrix noise.
            pub noise_type: crate::solvers::mals::NoiseType,
            /// Seed of every stage's noise stream and of the state-average start perturbation.
            pub seed: u64,
            /// Penalty or projection in the state-specific sweeps.
            pub mode: crate::solvers::ssmals::SsMode,
            /// The weight rule. `Fixed(w)` also fixes the POLISH weight; `Auto` recomputes it per
            /// root from the sweep's final overlaps.
            pub penalty: crate::solvers::ssmals::PenaltyRule,
            /// Where each sweep starts.
            pub ss_start: crate::solvers::excited::SsStart,
            /// Run the penalized Riemannian polish after each sweep. `false` returns the sweep's
            /// own state and records `gradnorms = NaN`, `stalled = false`, `passes = 0`.
            pub polish: bool,
            /// Warn when a supplied lower state's exact gate exceeds this.
            pub lower_grad_tol: f64,
            /// Rank-trim tolerance: a polished root whose smallest Schmidt value is below
            /// `0.3·trim_tol` is rounded at `trim_tol` and re-polished. On SU(2) the Schmidt
            /// value read is the CENTERWEIGHTED one, so one numeric threshold is one physical
            /// threshold on both layers.
            pub trim_tol: f64,
            /// Trim budget per root.
            pub trim_passes: usize,
            /// The polish CG. `new` sets `eps = 1e-13`, `gradtol = 1e-12`, `max_iter = 600` — the
            /// response ground-state pipeline's stop; `RiemannianOpts::new`'s own `eps = 1e-4`
            /// would stop the polish far above `polish_tol` and stall every root. (`gradtol` is
            /// inert on this code path: the gradient is the exact environment contraction.)
            pub cg: $ropts,
            /// Extra CG calls per polish while the exact gate is above `polish_tol` and each
            /// restart still contracts it.
            pub cg_restarts: usize,
            /// The exact-gate target the restart loop drives at.
            pub polish_tol: f64,
            /// The exact gate above which a root is `stalled` (and the licence refused).
            pub grad_gate: f64,
            /// The largest off-diagonal Gram entry the licence tolerates. At a BINDING rank cap
            /// the fixed-rank penalized minimizer keeps an `O(1/w)` residual overlap, so this gate
            /// is unreachable there — the remedy is a higher `rank` or a larger explicit penalty,
            /// never a looser tolerance.
            pub overlap_tol: f64,
            /// Keep the state-average ladder in `ExcitedInfo::sa_states`.
            pub keep_sa: bool,
            /// Per-stage progress lines on stdout.
            pub verbose: bool,
        }

        impl ExcitedOpts {
            /// The response ground-state schedule, at this layer's types.
            pub fn new(space: &$space) -> Self {
                let mut cg = <$ropts>::new(space);
                cg.eps = 1e-13;
                cg.gradtol = 1e-12;
                cg.max_iter = 600;
                ExcitedOpts {
                    nroots: 2,
                    weights: None,
                    rank: 0,
                    bond_dims: None,
                    eps: 1e-10,
                    max_iter: 30,
                    noise: vec![3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0],
                    ss_noise: vec![0.0],
                    noise_type: crate::solvers::mals::NoiseType::Perturbative,
                    seed: 1234,
                    mode: crate::solvers::ssmals::SsMode::Penalty,
                    penalty: crate::solvers::ssmals::PenaltyRule::Auto,
                    ss_start: crate::solvers::excited::SsStart::Sa,
                    polish: true,
                    lower_grad_tol: 1e-6,
                    trim_tol: 1e-6,
                    trim_passes: 3,
                    cg,
                    cg_restarts: 10,
                    polish_tol: 1e-10,
                    grad_gate: 1e-9,
                    overlap_tol: 1e-6,
                    keep_sa: false,
                    verbose: false,
                }
            }
        }

        $(#[$doc])*
        pub fn $name<T: Scalar>(
            h: &$ham<T>,
            x0: &$vec<T>,
            lower: &[$vec<T>],
            opts: &ExcitedOpts,
        ) -> (Vec<f64>, Vec<$vec<T>>, crate::solvers::excited::ExcitedInfo<$vec<T>>) {
            use crate::solvers::excited::{SsStart, default_ramp, ladder_swaps};
            use crate::solvers::mals::{MalsOptions, NoiseType};
            use crate::solvers::multistate::StateAverageOptions;
            use crate::solvers::penalty::Penalty;
            use crate::solvers::ssmals::{PenaltyRule, StateSpecificOptions, penalty_weight};

            let nroots = opts.nroots;
            assert!(
                nroots >= 1,
                "excited_states: `nroots` must be at least 1 (got {nroots}) — a ladder with no \
                 rungs has nothing to return"
            );
            let ramp = opts
                .bond_dims
                .clone()
                .unwrap_or_else(|| default_ramp(opts.rank));
            assert!(
                !ramp.is_empty(),
                "excited_states: `bond_dims` must be non-empty (use [0] for uncapped)"
            );
            assert!(
                !opts.noise.is_empty(),
                "excited_states: `noise` must be non-empty (use [0.0] for a noise-free run)"
            );
            assert!(
                opts.noise.iter().all(|a| a.is_finite() && *a >= 0.0),
                "excited_states: `noise` entries must be finite and non-negative"
            );
            assert!(
                !opts.ss_noise.is_empty(),
                "excited_states: `ss_noise` must be non-empty (use [0.0] for a noise-free run)"
            );
            assert!(
                opts.ss_noise.iter().all(|a| a.is_finite() && *a >= 0.0),
                "excited_states: `ss_noise` entries must be finite and non-negative"
            );
            assert!(
                opts.noise_type != NoiseType::Random,
                "excited_states: state-average sweeps define only Perturbative noise (the pooled \
                 density matrix is what the weighted chop truncates); pass \
                 NoiseType::Perturbative"
            );
            if let PenaltyRule::Fixed(w) = opts.penalty {
                assert!(
                    w.is_finite() && w >= 0.0,
                    "excited_states: PenaltyRule::Fixed({w}) must be finite and non-negative"
                );
            }

            let m = lower.len();
            let mut invn_all: Vec<f64> = Vec::with_capacity(m + nroots);
            let mut e_all: Vec<f64> = Vec::with_capacity(m + nroots);
            let mut lower_gradnorms: Vec<f64> = Vec::with_capacity(m);
            let bare = Penalty::new(0.0, &[]);
            for (j, psi) in lower.iter().enumerate() {
                assert_eq!(
                    psi.space, x0.space,
                    "excited_states: lower[{j}] lives in a different symmetry space than `x0`"
                );
                let n = psi.norm();
                assert!(
                    n.is_finite() && n > 0.0,
                    "excited_states: lower[{j}] has norm {n} — a finite, NONZERO norm is required \
                     (the penalty divides by it); a zero state carries no direction to exclude"
                );
                let invn = if n == 1.0 { 1.0 } else { 1.0 / n };
                invn_all.push(invn);
                let mut psi_hat = psi.clone();
                if invn != 1.0 {
                    psi_hat.scale_mut(T::from_f64(invn));
                }
                let el = $rayleigh(h, &psi_hat).re_s();
                e_all.push(el);
                let g = $gate_pen(h, &psi_hat, &bare);
                lower_gradnorms.push(g);
                if g > opts.lower_grad_tol {
                    eprintln!(
                        "excited_states: lower[{j}] is not a converged eigenstate (exact \
                         ‖P((H−λ)ψ̂)‖ = {g:.3e} > lower_grad_tol = {:.1e}; λ = {el:.12e}) — the \
                         new ladder is orthogonalized against a direction that is not an \
                         eigenvector, so its roots can sit BELOW the true rungs; a larger penalty \
                         does not repair this. Converge lower[{j}] first.",
                        opts.lower_grad_tol
                    );
                }
            }
            let e_lower_sorted = {
                let mut v = e_all.clone();
                v.sort_by(f64::total_cmp);
                v
            };

            let d = x0.space.d;
            let mut x = x0.clone();
            if x.ortho != Some(0) {
                x.right_orthogonalize(true);
            }
            let short_pair = |y: &$vec<T>| -> Option<(usize, usize)> {
                for k in 0..d.saturating_sub(1) {
                    let dim = $contract(y.core(k), y.core(k + 1)).num_coeffs();
                    if dim < nroots {
                        return Some((k, dim));
                    }
                }
                None
            };
            let mut enriched = false;
            if short_pair(&x).is_some() {
                let enrich = MalsOptions {
                    eps: opts.eps,
                    bond_dims: vec![ramp[0]],
                    noise: vec![opts.noise[0]],
                    noise_type: opts.noise_type,
                    seed: opts.seed,
                    max_iter: 1,
                    verbose: opts.verbose,
                    ..MalsOptions::default()
                };
                let _ = $mals_mut(h, &mut x, &enrich);
                enriched = true;
                if opts.verbose {
                    println!(
                        "excited_states: rank-deficient start enriched by one noisy MALS \
                         iteration at bond_dims {} / noise {:.1e}",
                        ramp[0], opts.noise[0]
                    );
                }
            }
            if let Some((k, dim)) = short_pair(&x) {
                panic!(
                    "excited_states: after the enrichment iteration the two-site local space at \
                     pair ({k}, {}) still holds {dim} coefficient(s), fewer than the {nroots} \
                     roots requested — supply a noisier first `noise` entry, a larger \
                     `bond_dims[0]`, or a start that already carries rank",
                    k + 1
                );
            }

            let mut mt = $multi::new(&x, nroots, opts.seed, 0.1);
            let sa_opts = StateAverageOptions {
                mals: MalsOptions {
                    eps: opts.eps,
                    bond_dims: ramp.clone(),
                    noise: opts.noise.clone(),
                    noise_type: opts.noise_type,
                    seed: opts.seed,
                    max_iter: opts.max_iter,
                    verbose: opts.verbose,
                    ..MalsOptions::default()
                },
                weights: opts.weights.clone(),
                ..StateAverageOptions::default()
            };
            let sa_history = $sa_mut(h, &mut mt, &sa_opts);
            let xsa = mt.states();
            let sa_states = if opts.keep_sa { Some(xsa.clone()) } else { None };

            let mut lower_all: Vec<$vec<T>> = lower.to_vec();
            let mut roots: Vec<$vec<T>> = Vec::with_capacity(nroots);
            let mut e: Vec<f64> = Vec::with_capacity(nroots);
            let mut ss_energies: Vec<f64> = Vec::with_capacity(nroots);
            let mut ss_history: Vec<crate::solvers::ssmals::SsHistory> =
                Vec::with_capacity(nroots);
            let mut j_ss: Vec<f64> = Vec::with_capacity(nroots);
            let mut j_pol: Vec<f64> = Vec::with_capacity(nroots);
            let mut penalties: Vec<f64> = Vec::with_capacity(nroots);
            let mut gradnorms: Vec<f64> = Vec::with_capacity(nroots);
            let mut passes: Vec<usize> = Vec::with_capacity(nroots);
            let mut stalled: Vec<bool> = Vec::with_capacity(nroots);
            let mut converged: Vec<bool> = Vec::with_capacity(nroots);

            let lower_overlaps = |states: &[$vec<T>], invn: &[f64], y: &$vec<T>| -> Vec<f64> {
                states
                    .iter()
                    .zip(invn.iter())
                    .map(|(psi, &iv)| $dot(psi, y).abs_s() * iv)
                    .collect()
            };

            for i in 0..nroots {
                let start: &$vec<T> = match opts.ss_start {
                    SsStart::Sa => &xsa[i],
                };
                let ss_opts = StateSpecificOptions {
                    mals: MalsOptions {
                        eps: opts.eps,
                        bond_dims: vec![opts.rank],
                        noise: opts.ss_noise.clone(),
                        noise_type: opts.noise_type,
                        seed: opts.seed,
                        max_iter: opts.max_iter,
                        verbose: opts.verbose,
                        ..MalsOptions::default()
                    },
                    mode: opts.mode,
                    penalty: opts.penalty,
                    roots: 1,
                    lower_energies: Some(e_all.clone()),
                    maxsubspace: None,
                    restart_keep: None,
                };
                let (hist, mut xi) = $ss_hist(h, start, &lower_all, &ss_opts);
                let ss_conv = hist.sweeps.last().map(|s| s.converged).unwrap_or(false);
                ss_energies.push(hist.lambda);
                let nx = xi.norm();
                assert!(
                    nx.is_finite() && nx > 0.0,
                    "excited_states: the state-specific sweep returned a state of norm {nx} for \
                     root {i} — nothing downstream is defined there"
                );

                let ov: Vec<f64> = lower_overlaps(&lower_all, &invn_all, &xi)
                    .into_iter()
                    .map(|o| o / nx)
                    .collect();
                let w = if lower_all.is_empty() {
                    0.0
                } else {
                    match opts.penalty {
                        PenaltyRule::Fixed(v) => v,
                        PenaltyRule::Auto => penalty_weight(hist.lambda, &ov, &e_all, 0.0),
                    }
                };
                let pen = Penalty::new(w, &lower_all);
                let lam_ss = $rayleigh(h, &xi).re_s() / (nx * nx);
                j_ss.push(lam_ss + w * ov.iter().map(|o| o * o).sum::<f64>());

                let mut npass = 0usize;
                if opts.polish {
                    let polish = |y0: &$vec<T>| -> $vec<T> {
                        let (_j, mut y) = $cg_pen(h, y0, &pen, &opts.cg);
                        let mut g = $gate_pen(h, &y, &pen);
                        for _ in 0..opts.cg_restarts {
                            if g <= opts.polish_tol {
                                break;
                            }
                            let (_j2, y2) = $cg_pen(h, &y, &pen, &opts.cg);
                            let g2 = $gate_pen(h, &y2, &pen);
                            let stall = g2 >= g;
                            y = y2;
                            g = g2;
                            if stall {
                                break;
                            }
                        }
                        y
                    };
                    xi = polish(&xi);
                    let (mut bl, mut br) = $tbases(&xi);
                    let mut smin = $min_schmidt(&bl, &br);
                    for p in 0..opts.trim_passes {
                        if smin > 0.3 * opts.trim_tol {
                            break;
                        }
                        npass = p + 1;
                        xi.round_mut(opts.trim_tol, crate::qn::Pooled::NONE, crate::round::RoundMode::Abs);
                        $normalize(&mut xi);
                        xi = polish(&xi);
                        let bases = $tbases(&xi);
                        bl = bases.0;
                        br = bases.1;
                        smin = $min_schmidt(&bl, &br);
                        if opts.verbose {
                            println!(
                                "excited_states root {i}: trim pass {} → σ_min = {smin:.3e}",
                                p + 1
                            );
                        }
                    }
                    gradnorms.push($gate_pen(h, &xi, &pen));
                } else {
                    $normalize(&mut xi);
                    gradnorms.push(f64::NAN);
                }
                let stall = opts.polish && gradnorms[i] > opts.grad_gate;
                if stall {
                    eprintln!(
                        "excited_states: root {i} is not a certified critical point of the \
                         penalized functional (exact ‖grad J‖ = {:.3e} > grad_gate = {:.1e} at \
                         penalty weight {w:.3e}) — raise `cg_restarts`, tighten `cg.eps`, or read \
                         it as a rank-limited root",
                        gradnorms[i], opts.grad_gate
                    );
                }

                let lam = $rayleigh(h, &xi).re_s();
                let ov2 = lower_overlaps(&lower_all, &invn_all, &xi);
                j_pol.push(lam + w * ov2.iter().map(|o| o * o).sum::<f64>());
                penalties.push(w);
                passes.push(npass);
                stalled.push(stall);
                converged.push(ss_conv && !stall);
                if opts.verbose {
                    println!(
                        "excited_states root {i}: SA λ = {:.12e}, SS λ = {:.12e}, polished λ = \
                         {lam:.12e}, w = {w:.3e}, gate = {:.3e}, passes = {npass}",
                        sa_history.lambdas[i], ss_energies[i], gradnorms[i]
                    );
                }
                ss_history.push(hist);
                e.push(lam);
                lower_all.push(xi.clone());
                e_all.push(lam);
                invn_all.push(1.0);
                roots.push(xi);
            }

            let nall = m + nroots;
            let mut overlaps = vec![vec![0.0f64; nall]; nall];
            for a in 0..nall {
                for b in 0..nall {
                    overlaps[a][b] =
                        $dot(&lower_all[a], &lower_all[b]).abs_s() * invn_all[a] * invn_all[b];
                }
            }
            let mut overlap_dev = 0.0f64;
            for a in 0..nall {
                for b in 0..nall {
                    if a != b {
                        overlap_dev = overlap_dev.max(overlaps[a][b]);
                    }
                }
            }

            let mut ladder = e_lower_sorted;
            ladder.extend_from_slice(&e);
            let (swaps, swapped) = ladder_swaps(&ladder, opts.eps);
            for &idx in &swapped {
                eprintln!(
                    "excited_states: rung {idx} of Λ = [sort(E_lower); e] sits {:.3e} below the \
                     rung under it (more than 10·eps = {:.1e}) — the ladder is out of energy \
                     order; labels are KEPT, not reordered (an index ≥ {m} is root {})",
                    ladder[idx - 1] - ladder[idx],
                    10.0 * opts.eps,
                    idx.saturating_sub(m)
                );
            }

            let licence =
                converged.iter().all(|&c| c) && overlap_dev <= opts.overlap_tol && swaps == 0;
            if !licence {
                if !converged.iter().all(|&c| c) {
                    let bad: Vec<usize> = (0..nroots).filter(|&i| !converged[i]).collect();
                    eprintln!(
                        "excited_states: no licence — root(s) {bad:?} did not converge (stalled \
                         {stalled:?}); read `ss_history[i].sweeps.last()` for the sweep's own \
                         stop test and `gradnorms` for the exact gate"
                    );
                } else if overlap_dev > opts.overlap_tol {
                    eprintln!(
                        "excited_states: no licence — the ladder's largest off-diagonal overlap \
                         is {overlap_dev:.3e} > overlap_tol = {:.1e}. At a BINDING rank cap the \
                         fixed-rank penalized minimizer keeps an O(1/w) residual overlap, so the \
                         remedy is a higher `rank` or a larger explicit penalty, NEVER a looser \
                         `overlap_tol`. An unconverged supplied lower state is a different cause \
                         — see `lower_gradnorms`.",
                        opts.overlap_tol
                    );
                } else {
                    eprintln!(
                        "excited_states: no licence — {swaps} rung(s) {swapped:?} of \
                         Λ = [sort(E_lower); e] are out of energy order"
                    );
                }
            }

            let info = crate::solvers::excited::ExcitedInfo {
                e: e.clone(),
                ss_energies,
                j_ss,
                j_pol,
                sa_history,
                ss_history,
                overlaps,
                overlap_dev,
                swaps,
                swapped,
                gradnorms,
                lower_gradnorms,
                penalties,
                passes,
                stalled,
                converged,
                enriched,
                licence,
                sa_states,
            };
            (e, roots, info)
        }
    };
}

pub(crate) use impl_excited_states;

use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
use crate::solvers::multistate::MultiTt;
use crate::vector::TtVector;

impl_excited_states!(
    excited_states,
    SparseHamiltonian,
    TtVector,
    MultiTt,
    crate::qn::QnSpace,
    crate::solvers::riemannian::RiemannianOpts,
    crate::solvers::mals::mals_mut,
    crate::solvers::multistate::state_average_mals_mut,
    crate::solvers::ssmals::state_specific_mals_with_history,
    crate::solvers::riemannian::riemannian_cg_penalized,
    crate::solvers::riemannian::penalized_exact_gradnorm,
    crate::solvers::riemannian::tangent_bases,
    crate::solvers::riemannian::normalize_mut,
    rayleigh_quotient,
    crate::ops::dot,
    crate::core::tangent::min_schmidt,
    crate::core::twosite::contract
);

/// [`excited_states`] over `C64` — the same driver at the complex scalar type.
pub fn excited_states_c64(
    h: &SparseHamiltonian<crate::scalar::C64>,
    x0: &TtVector<crate::scalar::C64>,
    lower: &[TtVector<crate::scalar::C64>],
    opts: &ExcitedOpts,
) -> (
    Vec<f64>,
    Vec<TtVector<crate::scalar::C64>>,
    ExcitedInfo<TtVector<crate::scalar::C64>>,
) {
    excited_states(h, x0, lower, opts)
}

#[cfg(test)]
mod tests {
    use super::{default_ramp, ladder_swaps};

    #[test]
    fn tms_d_default_ramp_overshoots_a_small_cap_and_ends_at_it() {
        assert_eq!(default_ramp(0), vec![0], "rank 0 is uncapped");
        assert_eq!(default_ramp(8), vec![6, 10, 14, 8, 8, 8]);
        assert_eq!(default_ramp(64), vec![16, 32, 48, 64, 64, 64]);
        for r in [1usize, 8, 16, 64, 300] {
            let ramp = default_ramp(r);
            assert_eq!(
                *ramp.last().expect("non-empty"),
                r,
                "the ramp must END at the requested cap (rank {r})"
            );
        }
    }

    #[test]
    fn tms_d_ladder_swaps_uses_an_absolute_ten_eps_threshold() {
        let e0 = -3.25f64;
        assert_eq!(ladder_swaps(&[e0, e0 + 1.0], 1e-10), (0, vec![]));
        assert_eq!(ladder_swaps(&[e0, e0 - 1e-8], 1e-10), (1, vec![1]));
        assert_eq!(ladder_swaps(&[e0, e0 - 5e-10], 1e-10), (0, vec![]));
        assert_eq!(ladder_swaps(&[e0, e0 - 2e-9], 1e-10), (1, vec![1]));
        assert_eq!(
            ladder_swaps(&[0.0, -1.0, -0.5, -2.0], 1e-12),
            (2, vec![1, 3])
        );
        assert_eq!(ladder_swaps(&[], 1e-10), (0, vec![]));
        assert_eq!(ladder_swaps(&[1.0], 1e-10), (0, vec![]));
    }
}
