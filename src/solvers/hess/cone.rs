// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::tangent::{TtTangent, tangent_basis};
use crate::hamiltonian::{SparseHamiltonian, xt_h_y};
use crate::ops::dot;
use crate::qn::{Pooled, RankGrid};
use crate::round::RoundMode;
use crate::vector::TtVector;

use rayon::prelude::*;

use super::symplectic_dense::{SymplecticSpectrum, hessian_grams, symplectic_from_grams};
use super::{WeingartenCtx, normal_residual, weingarten_operator};

/// Options for [`cone_response`].
#[derive(Clone, Copy, Debug)]
pub struct ConeOpts {
    /// Number of targeted excitations `J` (also the number of reported gaps).
    pub targets: usize,
    /// Budget of added normal directions `k`. The standing reference-channel candidate
    /// `N((H − λ)x)` counts against it (see [`ConeResponse::seeded`]), so `max_added == 1`
    /// degenerates to "reference repair only" — the seed fills the budget and the targeted loop
    /// body never runs. At `max_added == 0` no candidate is offered at all and the solve is the
    /// pure tangent-only arm.
    pub max_added: usize,
    /// Safeguard on targeted rounds (each round proposes up to `targets` candidates).
    pub max_rounds: usize,
    /// Ambient-MGS acceptance threshold on the orthogonalized candidate's norm.
    pub accept_tol: f64,
    /// TT rank-hygiene rounding applied inside the MGS passes and the Ritz-vector assembly
    /// (TT subtraction direct-sums ranks, so unrounded accumulation blows up).
    pub round_eps: f64,
    /// After the targeted loop closes (Krylov exhaustion of the targeted channels), top up
    /// with random normal-projected candidates to reach `max_added`. Needed for
    /// full-augmentation oracles; `None` (default) keeps the solve fully deterministic.
    pub topup_seed: Option<u64>,
    /// ALSO assemble the extended `(A_ext, B_ext)` pair — certified Weingarten tangent block,
    /// zero-padded `B` on the new blocks — and return the fixed-reference augmented Williamson
    /// spectrum. **Study switch**: this readout is measured to degrade monotonically with
    /// augmentation and is not the production readout.
    pub williamson: bool,
    /// Opt-in dual-path correctness certificate: assemble the FINAL CI Gram a second time via
    /// the pairwise `xt_h_y` bilinear forms and report the maximum relative deviation from the
    /// assembled one in [`ConeGuards::gram`] (hard-asserted). Because Gram rows are only ever
    /// appended, the final Gram contains every intermediate round's Gram as a leading principal
    /// block, so one check certifies all rounds. It restores the full `O(dim²)` bilinear-form
    /// cost, hence `false` by default — a debug device, but a usable one at molecular scale:
    /// the tolerance is DERIVED from the run's own measured error budget (see
    /// [`ConeGuards::gram`]), not a fixed constant, so it does not spuriously fire when
    /// `round_eps` is loosened or the tangent basis is poorly conditioned.
    pub gram_check: bool,
    /// Per-target convergence threshold on the pre-MGS normal-residual norm
    /// `‖N((H − ε_j)v_j)‖`. Targets whose signal falls below it stop consuming budget and
    /// H-applies (they remain reported); the freed slots go to the still-unconverged targets,
    /// fewest-served first. `0.0` disables the tracking entirely (a residual norm is never `≤ 0`
    /// in practice — the documented sentinel, used by the Krylov-closure pin).
    pub conv_tol: f64,
}

impl Default for ConeOpts {
    fn default() -> Self {
        ConeOpts {
            targets: 4,
            max_added: 8,
            max_rounds: 32,
            accept_tol: 1e-6,
            round_eps: 1e-12,
            topup_seed: None,
            williamson: false,
            gram_check: false,
            conv_tol: 1e-5,
        }
    }
}

/// Measured basis-integrity guard values (the non-gauge-content failure class — see
/// `crate::core::tangent::tangent_basis`). Sampled where exhaustive checks would be
/// quadratic in the ansatz size; the pinned tests run the exhaustive versions.
#[derive(Clone, Copy, Debug)]
pub struct ConeGuards {
    /// ι-isometry deviation over the tangent block: `max_ij |⟨ξ_i,ξ_j⟩ − ⟨ι(ξ_i),ι(ξ_j)⟩|`.
    pub iso: f64,
    /// Fresh-transpose asymmetry of the ambient `A` entries (sampled).
    pub asym: f64,
    /// Ambient-orthonormality deviation of the mixed set `|G − I|` (sampled).
    pub ortho: f64,
    /// Dual-path CI-Gram deviation `max_ij |G_ij − ⟨r_i, H r_j⟩| / max(1, |⟨r_i, H r_j⟩|)`
    /// between the assembled Gram and a fresh pairwise `xt_h_y` assembly, measured when
    /// [`ConeOpts::gram_check`] is on and `0.0` otherwise (the check is opt-in, so `0.0` is the
    /// "did not run" sentinel — a check that DID run reports its roundoff-level, strictly
    /// positive measured deviation).
    pub gram: f64,
}

/// Result of [`cone_response`].
#[derive(Clone, Debug)]
pub struct ConeResponse {
    /// Excitation gaps `E_j^CI − E_0^CI`, `j = 1..=targets` — the relaxed-reference readout.
    pub gaps: Vec<f64>,
    /// The relaxed reference energy `E_0^CI` (≤ λ; the difference is the reference-relaxation
    /// the mixed space affords).
    pub e0: f64,
    /// Absolute Ritz energies `E_0^CI ≤ E_1^CI ≤ …` of the FINAL relaxed-reference solve,
    /// length `min(targets + 1, dim)`. Invariants: `energies[0] == e0` and
    /// `gaps[j] == energies[j + 1] − energies[0]` to roundoff (both come from the same dense
    /// solve). Reported so a caller can judge an excited level on its own convergence rather
    /// than through the reference's error: the GAP undershoots once an excited Ritz value
    /// converges past `E_0^CI`'s own error (the measured H2O finding).
    pub energies: Vec<f64>,
    /// Whether the standing reference-channel candidate `N((H − λ)x)` was accepted. `false`
    /// when `max_added == 0`, and also when the reference is critical to the acceptance floor
    /// (the no-op degradation, which consumes no budget).
    pub seeded: bool,
    /// Normal directions actually added (may be below `max_added` on Krylov closure of the
    /// targeted channels — that is a finding, not an error).
    pub added: usize,
    /// Targeted rounds run.
    pub rounds: usize,
    /// Measured basis-integrity guards.
    pub guards: ConeGuards,
    /// The fixed-reference augmented Williamson spectrum, when [`ConeOpts::williamson`] —
    /// study readout only (measured to degrade with augmentation).
    pub williamson: Option<SymplecticSpectrum>,
    /// Per-state composition of the FINAL relaxed-reference solve's Ritz vectors over the mixed
    /// basis blocks: `[reference c₀², tangent Σᵢc_i², cone Σₘc_m²]`. Aligned with
    /// [`ConeResponse::energies`] (same length): `character[0]` is the relaxed GROUND state and
    /// `character[j + 1]` pairs with `gaps[j]`. Reading them as AMBIENT weights is exact up to the
    /// mixed-set orthonormality guard ([`ConeGuards::ortho`]). Zero extra `H`-work — the
    /// coefficients are the ones the final dense solve already returned.
    pub character: Vec<[f64; 3]>,
}

pub(crate) const DRIFT_REARM_FACTOR: f64 = 10.0;

pub(crate) fn drift_voids_record(recorded: f64, now: f64, conv_tol: f64) -> bool {
    (recorded - now).abs() > DRIFT_REARM_FACTOR * conv_tol
}

pub(crate) const ISO_TOL: f64 = 1e-8;

/// A cone solve's ENTRY refusal, reported instead of panicking: the caller's point has
/// gauge-degenerate tangent frames (ι-isometry defect ≥ [`ISO_TOL`]), so no mixed Gram was
/// assembled and no spectrum exists to report.
#[derive(Clone, Debug)]
pub struct ConeRefusal {
    /// The measured ι-isometry defect that lost against [`ISO_TOL`].
    pub iso: f64,
    /// The message [`cone_response`] would have panicked with.
    pub message: String,
}

impl std::fmt::Display for ConeRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ConeRefusal {}

/// Options for the MATRIX-FREE mixed-space eigensolver ([`cone_response_matfree`]).
#[derive(Clone, Copy, Debug)]
pub struct MatfreeOpts {
    /// Lanczos iteration cap per deflated restart (the Krylov depth, not the total work cap).
    pub max_iter: usize,
    /// Ritz-pair acceptance: the residual bound `β_m·|s_{m,i}|` must fall below
    /// `tol · max(1, |θ_i|)`. The floor worth requesting is set by the rank-hygiene rounds
    /// inside the operator application (`≈ round_eps · ‖H·v‖`), not by machine ε.
    pub tol: f64,
}

impl Default for MatfreeOpts {
    fn default() -> Self {
        MatfreeOpts {
            max_iter: 300,
            tol: 1e-11,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SolveMode<'m> {
    Dense,
    Matfree(&'m MatfreeOpts),
}

/// Instrumentation of the stages OUTSIDE the Krylov machinery — the ones [`MatfreeStats`] does
/// not cover and that the two solve modes SHARE.
#[derive(Clone, Debug, Default)]
pub struct ConeStageStats {
    /// The whole of `cone_solve_with`, entry to exit.
    pub t_total: std::time::Duration,
    /// Building the gauge-fixed tangent basis (`tangent_basis`, the `O(n_t²)` MGS).
    pub t_tangent_basis: std::time::Duration,
    /// `basis.len()` — the mixed space's tangent block size. NOTE this is `tangent_dim(bl) − 1`:
    /// `tangent_dim` counts the radial direction that `tangent_basis(…, true)` excludes.
    pub n_tangent: usize,
    /// Materializing `iota[i] = ι(ξ_i)` (the `n_t` embeddings).
    pub t_iota: std::time::Duration,
    /// Assembling the mixed reference set `refs` (`x` cloned, `iota` MOVED in — no second copy).
    pub t_refs: std::time::Duration,
    /// Live coefficient bytes held by `basis`: every tangent's component stack plus the ONE pair
    /// of base frames the basis shares (tangents hold their frames by reference count).
    pub bytes_basis: usize,
    /// Live coefficient bytes held by `iota`.
    pub bytes_iota: usize,
    /// Live coefficient bytes held by `refs` (`‖x‖ + bytes_iota`: `iota` now lives inside it).
    pub bytes_refs: usize,
    /// `reference_block` — the `n_t` term-A applies of the adjointness identity. Dense only.
    pub t_reference_block: std::time::Duration,
    /// The cached x-row (`n_t` tangent dots + one projection). Matrix-free path only.
    pub t_xrow: std::time::Duration,
    /// The standing `r⊥` seed's `normal_residual` — `Hx − λx` materialized UNROUNDED, then
    /// projected, then rounded — plus that round. The suspect stage.
    pub t_seed_residual: std::time::Duration,
    /// Candidates generated by the targeted loop, INCLUDING the ground-channel probes.
    pub n_cand: usize,
    /// Assembling each candidate's Ritz vector (`ritz_vector` / `mixed_ritz_vector`).
    pub t_cand_assemble: std::time::Duration,
    /// The candidates' `H·v` — an UNROUNDED, band-expanded product (`SparseHamiltonian::apply`
    /// rounds nothing), so read it beside [`Self::max_rank_hv`].
    pub t_cand_hv: std::time::Duration,
    /// Forming `w = H·v − ε·v` and rounding it back down. The round is `O(d·r³)` in the
    /// band-expanded rank, which is what makes the pair `(t_cand_hv, t_cand_residual)` the
    /// apply-vs-round discriminator.
    pub t_cand_residual: std::time::Duration,
    /// The candidates' `normal_project` (projection + embed + subtraction + round).
    pub t_cand_project: std::time::Duration,
    /// `accept_normal` invocations (seed + targeted + ground channel + top-up).
    pub n_accept_calls: usize,
    /// Those that survived the MGS/norm ladder — equals `ConeResponse::added` (pinned).
    pub n_accept_ok: usize,
    /// The two-pass MGS ladder inside `accept_normal` (`mgs_accept` + the re-projection).
    pub t_accept_mgs: std::time::Duration,
    /// The accepted normal's `w_m = round(H·n_m)` — one band-expanded apply plus its round.
    pub t_accept_hn: std::time::Duration,
    /// The materialize-and-dot row: `dot(r, w_m)` against every existing reference. `O(n_t)` LARGE-TT dots per
    /// accepted normal, and in matrix-free mode the only reason `refs` is materialized at all.
    pub t_accept_row: std::time::Duration,
    /// Guard 2 (sampled transpose asymmetry).
    pub t_guard_asym: std::time::Duration,
    /// Guard 3 (sampled mixed-set orthonormality — up to `32²` large-TT dots).
    pub t_guard_ortho: std::time::Duration,
    /// Guard 4 (opt-in exhaustive dual-path Gram certificate; dense path only).
    pub t_guard_gram: std::time::Duration,
    /// The optional augmented-Williamson study block.
    pub t_williamson: std::time::Duration,
    /// Largest total bond rank seen on an UNROUNDED `H·v` (candidate residuals and `H·n_m`).
    pub max_rank_hv: usize,
    /// Largest total bond rank of a candidate AFTER its rank-hygiene round.
    pub max_rank_cand: usize,
}

impl ConeStageStats {
    /// `t_total` minus [`Self::accounted`] — the work no timer covers, which on a matrix-free
    /// run is dominated by the Krylov solves ([`MatfreeStats`] measures those separately).
    /// Saturating, so timer overlap reads as "fully accounted" rather than as a negative.
    pub fn unaccounted(&self) -> std::time::Duration {
        self.t_total.saturating_sub(self.accounted())
    }

    /// Every named stage summed. The timers are disjoint by construction, so this is a genuine
    /// partial sum of `t_total` and never double-counts.
    pub fn accounted(&self) -> std::time::Duration {
        self.t_tangent_basis
            + self.t_iota
            + self.t_refs
            + self.t_reference_block
            + self.t_xrow
            + self.t_seed_residual
            + self.t_cand_assemble
            + self.t_cand_hv
            + self.t_cand_residual
            + self.t_cand_project
            + self.t_accept_mgs
            + self.t_accept_hn
            + self.t_accept_row
            + self.t_guard_asym
            + self.t_guard_ortho
            + self.t_guard_gram
            + self.t_williamson
    }
}

/// Instrumentation of the matrix-free solve: stage timers and counters, populated
/// by every matfree run (the overhead is a handful of `Instant::now` calls per iteration, well
/// under 1% of one) and surfaced by [`cone_response_matfree_with_stats`].
#[derive(Clone, Debug, Default)]
pub struct MatfreeStats {
    /// `mixed_apply` invocations (== [`Self::lanczos_iters`]; pinned).
    pub applies: usize,
    /// Total Lanczos inner iterations across all round solves, restarts and probes.
    pub lanczos_iters: usize,
    /// Fill restarts launched (a Krylov sequence seeded to ADD eigenpairs).
    pub restarts: usize,
    /// Fill restarts that started from a WARM seed (a previous solve's Ritz vector) rather
    /// than a canonical basis vector.
    pub warm_seeds: usize,
    /// Verification probes launched (post-fill degeneracy sweeps).
    pub probes: usize,
    /// Iterations consumed by each `solve` call (round solves in order, final solve last).
    pub round_iters: Vec<usize>,
    /// The dense path's operator-apply count on the same problem (`n_t` term-A applies +
    /// one `H·n` per accepted normal) — the head-to-head denominator.
    pub dense_equivalent_applies: usize,
    /// Assembling the tangent part of `v` (componentwise combo + ONE embed + round).
    pub t_assemble_tangent: std::time::Duration,
    /// The ONE fused operator application per iteration (`a_apply` + λ-shift) — no product is
    /// materialized.
    pub t_h_apply: std::time::Duration,
    /// The tangent dots against the fused image (the block form runs no projection sweep).
    pub t_project: std::time::Duration,
    /// All inner products of the back-projection (x-row, tangent dots, normal dots).
    pub t_dots: std::time::Duration,
    /// Tridiagonal eigensolves inside the Lanczos (scalar work; sanity column).
    pub t_tridiag: std::time::Duration,
    /// The stages outside the Krylov machinery, which the two solve modes
    /// SHARE. Unlike every field above, this one is populated on the DENSE path too (reachable
    /// via `cone_response_with_stats`), because it is those shared stages and not the Krylov ones
    /// that hold the saddle-scale wall.
    pub stages: ConeStageStats,
}

pub(crate) fn mixed_lanczos(
    apply: &mut dyn FnMut(&[f64]) -> Vec<f64>,
    dim: usize,
    want: usize,
    mf: &MatfreeOpts,
    stats: &std::cell::RefCell<MatfreeStats>,
    seeds: &[Vec<f64>],
    verify: bool,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let dotc = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let normc = |a: &[f64]| -> f64 { dotc(a, a).sqrt() };
    let orth_against = |v: &mut [f64], set: &[Vec<f64>]| {
        for _ in 0..2 {
            for b in set {
                let c = dotc(b, v);
                for (vi, bi) in v.iter_mut().zip(b) {
                    *vi -= c * bi;
                }
            }
        }
    };

    let want = want.min(dim);
    let mut conv_val: Vec<f64> = Vec::new();
    let mut conv_vec: Vec<Vec<f64>> = Vec::new();

    const CLEAN_PASSES: usize = 2;
    let mut next_seed = 0usize;
    let mut warm = 0usize;
    let mut clean = 0usize;
    let mut krylov_complete = false;
    while warm < seeds.len() || next_seed < dim {
        let filling = conv_val.len() < want;
        if !filling && (!verify || krylov_complete || clean >= CLEAN_PASSES) {
            break;
        }
        let take_warm = filling && warm < seeds.len();
        if !take_warm && next_seed >= dim {
            break;
        }
        let mut q = if take_warm {
            let mut q = seeds[warm].clone();
            q.resize(dim, 0.0);
            warm += 1;
            stats.borrow_mut().warm_seeds += 1;
            q
        } else {
            let mut q = vec![0.0f64; dim];
            q[next_seed] = 1.0;
            next_seed += 1;
            q
        };
        orth_against(&mut q, &conv_vec);
        let n0 = normc(&q);
        if n0 < 1e-8 {
            continue;
        }
        for qi in q.iter_mut() {
            *qi /= n0;
        }

        {
            let mut st = stats.borrow_mut();
            if filling {
                st.restarts += 1;
            } else {
                st.probes += 1;
            }
        }
        let need = if filling { want - conv_val.len() } else { 1 };
        let deflated_cap = dim - conv_val.len();
        let max_m = mf.max_iter.min(deflated_cap);
        let mut vs: Vec<Vec<f64>> = vec![q];
        let mut alphas: Vec<f64> = Vec::new();
        let mut betas: Vec<f64> = Vec::new();

        let ((tv, ts), last_beta) = loop {
            stats.borrow_mut().lanczos_iters += 1;
            let qj = vs.last().expect("nonempty Krylov block");
            let mut w = apply(qj);
            let a = dotc(&w, qj);
            alphas.push(a);
            if vs.len() >= 2 {
                let qprev = &vs[vs.len() - 2];
                let bprev = betas[betas.len() - 1];
                for (wi, (qi, pi)) in w.iter_mut().zip(qj.iter().zip(qprev)) {
                    *wi -= a * qi + bprev * pi;
                }
            } else {
                for (wi, qi) in w.iter_mut().zip(qj) {
                    *wi -= a * qi;
                }
            }
            orth_against(&mut w, &vs);
            orth_against(&mut w, &conv_vec);
            let b = normc(&w);

            let m = alphas.len();
            let mut flat = vec![0.0f64; m * m];
            for i in 0..m {
                flat[i * m + i] = alphas[i];
                if i + 1 < m {
                    flat[i * m + (i + 1)] = betas[i];
                    flat[(i + 1) * m + i] = betas[i];
                }
            }
            let t_tri = std::time::Instant::now();
            let (tv, ts) = dense_self_adjoint_smallest::<f64>(&flat, m, m.min(need + 1));
            stats.borrow_mut().t_tridiag += t_tri.elapsed();
            let converged_here = (0..need.min(tv.len()))
                .all(|i| b * ts[i][m - 1].abs() <= mf.tol * tv[i].abs().max(1.0));
            if converged_here || b < 1e-13 || m >= max_m {
                if filling && b < 1e-13 && m >= deflated_cap {
                    krylov_complete = true;
                }
                break ((tv, ts), b);
            }
            let mut qn = w;
            for qi in qn.iter_mut() {
                *qi /= b;
            }
            betas.push(b);
            vs.push(qn);
        };

        let m = alphas.len();
        let mut found: Vec<(f64, Vec<f64>)> = Vec::new();
        for (i, (&theta, sv)) in tv.iter().zip(ts.iter()).enumerate().take(need) {
            let res = last_beta * sv[m - 1].abs();
            let forced = filling && found.is_empty() && i == 0 && m >= max_m;
            if res > mf.tol * theta.abs().max(1.0) && !forced {
                break;
            }
            let mut y = vec![0.0f64; dim];
            for (sj, qv) in sv.iter().zip(&vs) {
                for (yi, qi) in y.iter_mut().zip(qv) {
                    *yi += sj * qi;
                }
            }
            orth_against(&mut y, &conv_vec);
            let ny = normc(&y);
            if ny < 1e-8 {
                continue;
            }
            for yi in y.iter_mut() {
                *yi /= ny;
            }
            found.push((theta, y));
        }

        if filling {
            for (theta, y) in found {
                conv_val.push(theta);
                conv_vec.push(y);
            }
        } else if let Some((theta, y)) = found.into_iter().next() {
            let (worst_idx, &worst) = conv_val
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite Ritz values"))
                .expect("nonempty converged set");
            if theta < worst - mf.tol * worst.abs().max(1.0) {
                conv_val.swap_remove(worst_idx);
                conv_vec.swap_remove(worst_idx);
                conv_val.push(theta);
                conv_vec.push(y);
                clean = 0;
            } else {
                clean += 1;
            }
        } else {
            let saw_intruder = !conv_val.is_empty()
                && tv.first().is_some_and(|&t0| {
                    let worst = conv_val.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    t0 < worst - mf.tol * worst.abs().max(1.0)
                });
            if !saw_intruder {
                clean += 1;
            }
        }
    }

    let mut idx: Vec<usize> = (0..conv_val.len()).collect();
    idx.sort_by(|&a, &b| {
        conv_val[a]
            .partial_cmp(&conv_val[b])
            .expect("finite Ritz values")
    });
    let vals: Vec<f64> = idx.iter().map(|&i| conv_val[i]).collect();
    let vecs: Vec<Vec<f64>> = idx.iter().map(|&i| conv_vec[i].clone()).collect();
    (vals, vecs)
}

macro_rules! impl_cone {
    (
        ham: $ham:ty,
        vec: $vec:ty,
        tangent: $tangent:ty,
        ctx: $ctx:ident,
        grid: $grid:ident,
        weingarten_operator: $weingarten_operator:path,
        normal_residual: $normal_residual:path,
        tangent_basis: $tangent_basis:path,
        hessian_grams: $hessian_grams:path,
        xt_h_y: $xt_h_y:path,
        dot: $dot:path,
        tt_randn: $tt_randn:path,
        tag: $tag:literal,
        gram_note: $gram_note:literal,
        $(#[$doc_response:meta])*
        cone_response,
        $(#[$doc_try:meta])*
        try_cone_response,
        $(#[$doc_matfree:meta])*
        cone_response_matfree,
        $(#[$doc_matfree_stats:meta])*
        cone_response_matfree_with_stats,
        $(#[$doc_with_stats:meta])*
        cone_response_with_stats $(,)?
    ) => {
        pub(crate) fn iso_defect(basis: &[$tangent]) -> f64 {
            let n = basis.len();
            let sample = n.min(24);
            let head: Vec<$vec> = basis[..sample].iter().map(|u| u.to_vector()).collect();
            let mut iso = 0.0f64;
            for i in 0..sample {
                for j in 0..sample {
                    iso = iso.max((basis[i].dot(&basis[j]) - $dot(&head[i], &head[j])).abs());
                }
            }
            for (i, bi) in basis.iter().enumerate() {
                let d2 = match head.get(i) {
                    Some(ii) => $dot(ii, ii),
                    None => {
                        let ii = bi.to_vector();
                        $dot(&ii, &ii)
                    }
                };
                iso = iso.max((bi.dot(bi) - d2).abs());
            }
            iso
        }

        #[derive(Clone, Copy, Debug)]
        pub(crate) struct IsoRefusal {
            pub(crate) iso: f64,
        }

        impl IsoRefusal {
            pub(crate) fn message(&self) -> String {
                format!(
                    concat!(
                        $tag,
                        "cone_response: ι-isometry violated ({:.3e}) — non-gauge-fixed tangent \
                         content (the F7 mechanism); the mixed Gram would be meaningless"
                    ),
                    self.iso
                )
            }
        }

        fn normal_project(bl: &$vec, br: &$vec, v: &$vec) -> $vec {
            let tang = <$tangent>::project(bl, br, v).to_vector();
            let mut n = v - &tang;
            n.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
            n
        }

        fn mgs_accept(
            bl: &$vec,
            br: &$vec,
            normals: &[$vec],
            v: $vec,
            accept_tol: f64,
            round_eps: f64,
        ) -> Option<$vec> {
            let mut v = v;
            let n0 = v.norm();
            if n0 < 1e-10 {
                return None;
            }
            v.scale_mut(1.0 / n0);
            for _pass in 0..2 {
                v = normal_project(bl, br, &v);
                for b in normals {
                    let c = $dot(b, &v);
                    v = &v - &(b * c);
                }
                v.round_mut(round_eps, Pooled::NONE, RoundMode::Rel);
            }
            let nrm = v.norm();
            if nrm > accept_tol {
                v.scale_mut(1.0 / nrm);
                Some(v)
            } else {
                None
            }
        }

        pub(crate) struct MixedGram {
            pub(crate) refs: Vec<$vec>,
            n_ref: usize,
            h_normals: Vec<$vec>,
            hrows: Vec<Vec<f64>>,
        }

        impl MixedGram {
            pub(crate) fn normals(&self) -> &[$vec] {
                &self.refs[self.n_ref..]
            }

            fn flat(&self, dim: usize) -> Vec<f64> {
                let mut flat = vec![0.0f64; dim * dim];
                for (i, row) in self.hrows.iter().enumerate().take(dim) {
                    flat[i * dim..(i + 1) * dim].copy_from_slice(&row[..dim]);
                }
                flat
            }
        }

        pub(crate) fn ritz_vector(refs: &[$vec], coeffs: &[f64], round_eps: f64) -> $vec {
            let mut v = &refs[0] * coeffs[0];
            for (c, r) in coeffs.iter().zip(refs.iter()).skip(1) {
                v = &v + &(r * *c);
                v.round_mut(round_eps, Pooled::NONE, RoundMode::Rel);
            }
            v
        }

        #[allow(clippy::too_many_arguments)]
        fn accept_normal(
            ham: &$ham,
            bl: &$vec,
            br: &$vec,
            gram: &mut MixedGram,
            cand: $vec,
            opts: &ConeOpts,
            mirror_rows: bool,
            st: &mut ConeStageStats,
        ) -> bool {
            use std::time::Instant;
            st.n_accept_calls += 1;
            let t_mgs = Instant::now();
            let Some(mut n) = mgs_accept(
                bl,
                br,
                gram.normals(),
                cand,
                opts.accept_tol,
                opts.round_eps,
            ) else {
                st.t_accept_mgs += t_mgs.elapsed();
                return false;
            };
            n = normal_project(bl, br, &n);
            let nn = n.norm();
            if nn < 1e-8 {
                st.t_accept_mgs += t_mgs.elapsed();
                return false;
            }
            n.scale_mut(1.0 / nn);
            st.t_accept_mgs += t_mgs.elapsed();
            {
                let t_hn = Instant::now();
                let mut w = ham.apply(&n);
                st.max_rank_hv = st.max_rank_hv.max(max_total_rank(&w));
                w.round_mut(opts.round_eps, Pooled::NONE, RoundMode::Rel);
                st.t_accept_hn += t_hn.elapsed();
                let t_row = Instant::now();
                let mut row: Vec<f64> = gram.refs.par_iter().map(|r| $dot(r, &w)).collect();
                row.push($dot(&n, &w));
                st.t_accept_row += t_row.elapsed();
                if mirror_rows {
                    for (i, hr) in gram.hrows.iter_mut().enumerate() {
                        hr.push(row[i]);
                    }
                }
                gram.hrows.push(row);
                gram.h_normals.push(w);
            }
            gram.refs.push(n);
            st.n_accept_ok += 1;
            true
        }

        fn reference_block(
            ctx: &$ctx<'_, f64>,
            basis: &[$tangent],
            x: &$vec,
            lam: f64,
        ) -> Vec<Vec<f64>> {
            let (bl, br) = ctx.shared_bases();
            let n_t = basis.len();
            let grad = ctx.gradient();
            let px = <$tangent>::project_shared(bl, br, x);

            let mut hrows = vec![vec![0.0f64; 1 + n_t]; 1 + n_t];
            hrows[0][0] = lam;
            for (j, xj) in basis.iter().enumerate() {
                let v = grad.dot(xj) + lam * px.dot(xj);
                hrows[0][1 + j] = v;
                hrows[1 + j][0] = v;
            }

            let cols: Vec<Vec<f64>> = (0..n_t)
                .into_par_iter()
                .map(|j| {
                    let mut y = ctx.a_apply(&basis[j]);
                    y.axpy(lam, &basis[j]);
                    basis.iter().map(|bi| bi.dot(&y)).collect()
                })
                .collect();
            for i in 0..n_t {
                for j in 0..n_t {
                    hrows[1 + i][1 + j] = 0.5 * (cols[j][i] + cols[i][j]);
                }
            }
            hrows
        }

        $(#[$doc_response])*
        pub fn cone_response(
            h: &$ham,
            x: &$vec,
            opts: &ConeOpts,
        ) -> ConeResponse {
            match cone_solve(h, x, opts) {
                Ok(out) => out.response,
                Err(refusal) => panic!("{}", refusal.message()),
            }
        }

        $(#[$doc_try])*
        pub fn try_cone_response(
            h: &$ham,
            x: &$vec,
            opts: &ConeOpts,
        ) -> Result<ConeResponse, ConeRefusal> {
            match cone_solve(h, x, opts) {
                Ok(out) => Ok(out.response),
                Err(refusal) => Err(ConeRefusal {
                    iso: refusal.iso,
                    message: refusal.message(),
                }),
            }
        }

        fn tt_bytes(v: &$vec) -> usize {
            v.cores.iter().map(|c| c.num_coeffs()).sum::<usize>() * std::mem::size_of::<f64>()
        }

        fn basis_bytes(basis: &[$tangent]) -> usize {
            let comps: usize = basis.iter().map(|u| u.num_coeffs()).sum();
            let frames = basis
                .first()
                .map_or(0, |u| tt_bytes(&u.base_l) + tt_bytes(&u.base_r));
            comps * std::mem::size_of::<f64>() + frames
        }

        fn max_total_rank(v: &$vec) -> usize {
            v.ranks.iter().map($grid::total).max().unwrap_or(0)
        }

        #[allow(clippy::too_many_arguments)]
        fn mixed_apply(
            ctx: &$ctx<'_, f64>,
            basis: &[$tangent],
            xrow: &[f64],
            lam: f64,
            nrows: &[Vec<f64>],
            c: &[f64],
            stats: &mut MatfreeStats,
        ) -> Vec<f64> {
            use std::time::Instant;
            let n_t = basis.len();
            let k = nrows.len();
            let dim = 1 + n_t + k;
            debug_assert_eq!(c.len(), dim);
            stats.applies += 1;

            let t0 = Instant::now();
            let combo = if n_t > 0 {
                let mut combo = basis[0].clone();
                combo.scale_mut(c[1]);
                for (i, xi) in basis.iter().enumerate().skip(1) {
                    combo.axpy(c[1 + i], xi);
                }
                Some(combo)
            } else {
                None
            };
            let t1 = Instant::now();
            stats.t_assemble_tangent += t1 - t0;

            let y = combo.as_ref().map(|xi| {
                let mut y = ctx.a_apply(xi);
                y.axpy(lam, xi);
                y
            });
            let t2 = Instant::now();
            stats.t_h_apply += t2 - t1;

            let mut out = vec![0.0f64; dim];
            out[0] = lam * c[0]
                + xrow
                    .iter()
                    .zip(&c[1..1 + n_t])
                    .map(|(g, ci)| g * ci)
                    .sum::<f64>();
            for (m, row) in nrows.iter().enumerate() {
                out[0] += row[0] * c[1 + n_t + m];
            }
            if let Some(y) = &y {
                for (i, xi) in basis.iter().enumerate() {
                    out[1 + i] = xi.dot(y);
                }
            }
            let t3 = Instant::now();
            stats.t_project += t3 - t2;
            for (i, o) in out[1..1 + n_t].iter_mut().enumerate() {
                *o += c[0] * xrow[i];
                for (m, row) in nrows.iter().enumerate() {
                    *o += row[1 + i] * c[1 + n_t + m];
                }
            }
            for m in 0..k {
                let mut acc = 0.0f64;
                for (j, cj) in c.iter().enumerate().take(nrows[m].len()) {
                    acc += nrows[m][j] * cj;
                }
                for mm in (m + 1)..k {
                    acc += nrows[mm][1 + n_t + m] * c[1 + n_t + mm];
                }
                out[1 + n_t + m] = acc;
            }
            stats.t_dots += t3.elapsed();
            out
        }

        fn mixed_ritz_vector(
            x: &$vec,
            basis: &[$tangent],
            normals: &[$vec],
            coeffs: &[f64],
            round_eps: f64,
        ) -> $vec {
            let n_t = basis.len();
            let mut v = x * coeffs[0];
            if n_t > 0 {
                let mut combo = basis[0].clone();
                combo.scale_mut(coeffs[1]);
                for (i, xi) in basis.iter().enumerate().skip(1) {
                    combo.axpy(coeffs[1 + i], xi);
                }
                v = &v + &combo.to_vector();
                v.round_mut(round_eps, Pooled::NONE, RoundMode::Rel);
            }
            let k_in = coeffs.len().saturating_sub(1 + n_t);
            for (m, nm) in normals.iter().enumerate().take(k_in) {
                v = &v + &(nm * coeffs[1 + n_t + m]);
                v.round_mut(round_eps, Pooled::NONE, RoundMode::Rel);
            }
            v
        }

        $(#[$doc_matfree])*
        pub fn cone_response_matfree(
            h: &$ham,
            x: &$vec,
            opts: &ConeOpts,
            mf: &MatfreeOpts,
        ) -> ConeResponse {
            cone_response_matfree_with_stats(h, x, opts, mf).0
        }

        $(#[$doc_matfree_stats])*
        pub fn cone_response_matfree_with_stats(
            h: &$ham,
            x: &$vec,
            opts: &ConeOpts,
            mf: &MatfreeOpts,
        ) -> (ConeResponse, MatfreeStats) {
            match cone_solve_with(h, x, opts, SolveMode::Matfree(mf)) {
                Ok(out) => (out.response, out.matfree_stats),
                Err(refusal) => panic!("{}", refusal.message()),
            }
        }

        pub(crate) struct ConeSolveOut {
            pub(crate) response: ConeResponse,
            #[allow(dead_code)]
            pub(crate) gram: MixedGram,
            #[allow(dead_code)]
            pub(crate) evecs: Vec<Vec<f64>>,
            pub(crate) matfree_stats: MatfreeStats,
            #[allow(dead_code)]
            pub(crate) lam: f64,
        }

        pub(crate) fn cone_solve(
            h: &$ham,
            x: &$vec,
            opts: &ConeOpts,
        ) -> Result<ConeSolveOut, IsoRefusal> {
            cone_solve_with(h, x, opts, SolveMode::Dense)
        }

        $(#[$doc_with_stats])*
        pub fn cone_response_with_stats(
            h: &$ham,
            x: &$vec,
            opts: &ConeOpts,
        ) -> (ConeResponse, ConeStageStats) {
            match cone_solve_with(h, x, opts, SolveMode::Dense) {
                Ok(out) => (out.response, out.matfree_stats.stages),
                Err(refusal) => panic!("{}", refusal.message()),
            }
        }

        pub(crate) fn cone_solve_with(
            h: &$ham,
            x: &$vec,
            opts: &ConeOpts,
            mode: SolveMode<'_>,
        ) -> Result<ConeSolveOut, IsoRefusal> {
            let dense = matches!(mode, SolveMode::Dense);
            assert!(
                dense || !opts.gram_check,
                "cone_response_matfree: gram_check requires the dense path — no Gram is assembled here \
                 (the dual-path certificate for the matrix-free solver is the dense-agreement pin)"
            );
            use std::time::Instant;
            let t_entry = Instant::now();
            let mut st = ConeStageStats::default();

            let ctx = $weingarten_operator(h, x);
            let (bl, br) = ctx.shared_bases();
            let lam = ctx.lambda;
            let t_basis = Instant::now();
            let basis = $tangent_basis(bl, br, true);
            st.t_tangent_basis = t_basis.elapsed();
            let n_t = basis.len();
            st.n_tangent = n_t;
            st.bytes_basis = basis_bytes(&basis);
            let t_iota = Instant::now();
            let iota: Vec<$vec> = basis.iter().map(|u| u.to_vector()).collect();
            st.t_iota = t_iota.elapsed();
            st.bytes_iota = iota.iter().map(tt_bytes).sum();

            let mut iso = iso_defect(&basis);
            if iso >= ISO_TOL {
                return Err(IsoRefusal { iso });
            }

            let t_refs = Instant::now();
            let mut refs: Vec<$vec> = Vec::with_capacity(1 + n_t + opts.max_added);
            refs.push(x.clone());
            refs.extend(iota);
            st.t_refs = t_refs.elapsed();
            st.bytes_refs = refs.iter().map(tt_bytes).sum();
            let t_refblock = Instant::now();
            let mut gram = MixedGram {
                refs,
                n_ref: 1 + n_t,
                h_normals: Vec::new(),
                hrows: if dense {
                    reference_block(&ctx, &basis, x, lam)
                } else {
                    Vec::new()
                },
            };
            if dense {
                st.t_reference_block = t_refblock.elapsed();
            }

            let mf_stats = std::cell::RefCell::new(MatfreeStats::default());
            let t_xrow = Instant::now();
            let xrow: Vec<f64> = if dense {
                Vec::new()
            } else {
                let grad = ctx.gradient();
                let px = <$tangent>::project_shared(bl, br, x);
                basis
                    .iter()
                    .map(|xj| grad.dot(xj) + lam * px.dot(xj))
                    .collect()
            };
            if !dense {
                st.t_xrow = t_xrow.elapsed();
            }

            let prev_evecs: std::cell::RefCell<Vec<Vec<f64>>> = std::cell::RefCell::new(Vec::new());
            let solve = |gram: &MixedGram, want: usize, verify: bool| -> (Vec<f64>, Vec<Vec<f64>>) {
                match mode {
                    SolveMode::Dense => {
                        let dim = gram.refs.len();
                        dense_self_adjoint_smallest::<f64>(&gram.flat(dim), dim, want)
                    }
                    SolveMode::Matfree(mf) => {
                        let dim = gram.refs.len();
                        let before = mf_stats.borrow().lanczos_iters;
                        let mut apply = |c: &[f64]| {
                            let mut st = mf_stats.borrow_mut();
                            mixed_apply(&ctx, &basis, &xrow, lam, &gram.hrows, c, &mut st)
                        };
                        let seeds = prev_evecs.borrow().clone();
                        let out = mixed_lanczos(&mut apply, dim, want, mf, &mf_stats, &seeds, verify);
                        *prev_evecs.borrow_mut() = out.1.clone();
                        let mut st = mf_stats.borrow_mut();
                        let after = st.lanczos_iters;
                        st.round_iters.push(after - before);
                        out
                    }
                }
            };

            let mut seeded = false;
            if opts.max_added >= 1 {
                let t_seed = Instant::now();
                let mut rp = $normal_residual(h, bl, br, x, lam);
                st.max_rank_hv = st.max_rank_hv.max(max_total_rank(&rp));
                rp.round_mut(opts.round_eps, Pooled::NONE, RoundMode::Rel);
                let cand = normal_project(bl, br, &rp);
                st.max_rank_cand = st.max_rank_cand.max(max_total_rank(&cand));
                st.t_seed_residual += t_seed.elapsed();
                seeded = accept_normal(h, bl, br, &mut gram, cand, opts, dense, &mut st);
            }

            let tracking = opts.conv_tol > 0.0;
            let mut conv_at: Vec<Option<f64>> = vec![None; opts.targets + 1];
            let mut served: Vec<usize> = vec![0usize; opts.targets + 1];
            let mut rounds = 0usize;
            while gram.normals().len() < opts.max_added && rounds < opts.max_rounds {
                rounds += 1;
                let want = (opts.targets + 1).min(gram.refs.len());
                let (ev, evec) = solve(&gram, want, false);

                for j in 0..want {
                    if let Some(e) = conv_at[j]
                        && drift_voids_record(e, ev[j], opts.conv_tol)
                    {
                        conv_at[j] = None;
                    }
                }
                let mut active: Vec<usize> = (1..want).filter(|&j| conv_at[j].is_none()).collect();
                if tracking {
                    active.sort_by_key(|&j| (served[j], j));
                }

                let mut accepted = 0usize;
                for j in active {
                    if gram.normals().len() >= opts.max_added {
                        break;
                    }
                    st.n_cand += 1;
                    let t_asm = Instant::now();
                    let v = if dense {
                        ritz_vector(&gram.refs, &evec[j], opts.round_eps)
                    } else {
                        mixed_ritz_vector(x, &basis, gram.normals(), &evec[j], opts.round_eps)
                    };
                    st.t_cand_assemble += t_asm.elapsed();
                    let t_hv = Instant::now();
                    let hv = h.apply(&v);
                    st.max_rank_hv = st.max_rank_hv.max(max_total_rank(&hv));
                    st.t_cand_hv += t_hv.elapsed();
                    let t_res = Instant::now();
                    let mut w = &hv - &(&v * ev[j]);
                    w.round_mut(opts.round_eps, Pooled::NONE, RoundMode::Rel);
                    st.t_cand_residual += t_res.elapsed();
                    let t_proj = Instant::now();
                    let cand = normal_project(bl, br, &w);
                    st.t_cand_project += t_proj.elapsed();
                    st.max_rank_cand = st.max_rank_cand.max(max_total_rank(&cand));
                    if tracking && cand.norm() <= opts.conv_tol {
                        conv_at[j] = Some(ev[j]);
                        continue;
                    }
                    if accept_normal(h, bl, br, &mut gram, cand, opts, dense, &mut st) {
                        accepted += 1;
                        served[j] += 1;
                    }
                }
                if accepted > 0 {
                    continue;
                }

                if !tracking || conv_at[0].is_some() || gram.normals().len() >= opts.max_added {
                    break;
                }
                st.n_cand += 1;
                let t_asm = Instant::now();
                let v0 = if dense {
                    ritz_vector(&gram.refs, &evec[0], opts.round_eps)
                } else {
                    mixed_ritz_vector(x, &basis, gram.normals(), &evec[0], opts.round_eps)
                };
                st.t_cand_assemble += t_asm.elapsed();
                let t_hv = Instant::now();
                let hv0 = h.apply(&v0);
                st.max_rank_hv = st.max_rank_hv.max(max_total_rank(&hv0));
                st.t_cand_hv += t_hv.elapsed();
                let t_res = Instant::now();
                let mut w0 = &hv0 - &(&v0 * ev[0]);
                w0.round_mut(opts.round_eps, Pooled::NONE, RoundMode::Rel);
                st.t_cand_residual += t_res.elapsed();
                let t_proj = Instant::now();
                let cand = normal_project(bl, br, &w0);
                st.t_cand_project += t_proj.elapsed();
                st.max_rank_cand = st.max_rank_cand.max(max_total_rank(&cand));
                if cand.norm() <= opts.conv_tol {
                    conv_at[0] = Some(ev[0]);
                    break;
                }
                if !accept_normal(h, bl, br, &mut gram, cand, opts, dense, &mut st) {
                    break;
                }
                served[0] += 1;
            }

            if let Some(seed) = opts.topup_seed {
                use rand::{RngExt, SeedableRng};
                let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
                let space = x.space;
                let mut attempts = 0usize;
                while gram.normals().len() < opts.max_added && attempts < 40 * opts.max_added.max(1) {
                    attempts += 1;
                    let d = space.d;
                    let ranks: Vec<$grid> = (0..=d)
                        .map(|b| {
                            let mut r = $grid::zeros(&space);
                            for s in space.bond_sectors(b) {
                                r[s] = if b == 0 || b == d {
                                    1
                                } else {
                                    rng.random_range(1..=2)
                                };
                            }
                            r
                        })
                        .collect();
                    let v: $vec = $tt_randn(space, &ranks, &mut rng);
                    let cand = normal_project(bl, br, &v);
                    let _ = accept_normal(h, bl, br, &mut gram, cand, opts, dense, &mut st);
                }
            }

            let iota = &gram.refs[1..1 + n_t];
            let amb =
                |vi: &$vec, vj: &$vec| -> f64 { $xt_h_y(vi, h, vj) - lam * $dot(vi, vj) };
            let t_guard = Instant::now();
            let mut asym = 0.0f64;
            for nm in gram.normals().iter().take(3) {
                for ii in iota.iter().take(3) {
                    asym = asym.max((amb(nm, ii) - amb(ii, nm)).abs());
                }
            }
            st.t_guard_asym = t_guard.elapsed();
            assert!(
                asym < 1e-6,
                concat!($tag, "cone_response: A_ext asymmetry {asym:.3e} — assembly bug"),
                asym = asym
            );

            let t_guard = Instant::now();
            let mut ortho = 0.0f64;
            let osample = gram.refs.len().min(32);
            for i in 0..osample {
                for j in 0..osample {
                    let g = $dot(&gram.refs[i], &gram.refs[j]);
                    ortho = ortho.max((g - if i == j { 1.0 } else { 0.0 }).abs());
                }
            }
            for n in gram.normals().iter() {
                let g = $dot(n, n);
                ortho = ortho.max((g - 1.0).abs());
            }
            st.t_guard_ortho = t_guard.elapsed();
            assert!(
                ortho < 1e-6,
                concat!($tag, "cone_response: mixed-set orthonormality violated ({ortho:.3e})"),
                ortho = ortho
            );

            let dim = gram.refs.len();
            let mut gram_dev = 0.0f64;
            let t_guard = Instant::now();
            if opts.gram_check {
                let mut edefect = 0.0f64;
                for i in 0..n_t {
                    let mut rowsum = 0.0f64;
                    for j in 0..n_t {
                        let e = (basis[i].dot(&basis[j]) - $dot(&iota[i], &iota[j])).abs();
                        rowsum += e;
                        iso = iso.max(e);
                    }
                    edefect = edefect.max(rowsum);
                }

                let mut colsq = vec![0.0f64; dim];
                for i in 0..dim {
                    for j in 0..=i {
                        let p = $xt_h_y(&gram.refs[i], h, &gram.refs[j]);
                        colsq[i] += p * p;
                        if i != j {
                            colsq[j] += p * p;
                        }
                        let scale = p.abs().max(1.0);
                        gram_dev = gram_dev.max((gram.hrows[i][j] - p).abs() / scale);
                        gram_dev = gram_dev.max((gram.hrows[j][i] - p).abs() / scale);
                    }
                }
                let hscale = colsq
                    .iter()
                    .fold(lam.abs().max(1.0), |m, &c| m.max(c.sqrt()));
                let wmax = gram.h_normals.iter().fold(0.0f64, |m, w| m.max(w.norm()));
                let t_iso = 4.0 * edefect * hscale;
                let t_round = 4.0 * opts.round_eps * wmax;
                let t_fp = 64.0 * f64::EPSILON * (dim as f64).sqrt() * hscale;
                let tol = t_iso + t_round + t_fp;
                assert!(
                    gram_dev <= tol,
                    concat!(
                        $tag,
                        "cone_response: restructured vs pairwise CI Gram deviate by {gram_dev:.3e}, \
                         above the measured error budget {tol:.3e} (ι-isometry {t_iso:.3e} from ‖E‖ \
                         {edefect:.3e} × hscale {hscale:.3e}; w-round {t_round:.3e} from round_eps \
                         {:.1e} × ‖H·n‖ {wmax:.3e}; fp floor {t_fp:.3e}) — an L1/L2 ASSEMBLY defect",
                        $gram_note,
                        ", not a conditioning or rounding one"
                    ),
                    opts.round_eps,
                    gram_dev = gram_dev,
                    tol = tol,
                    t_iso = t_iso,
                    edefect = edefect,
                    hscale = hscale,
                    t_round = t_round,
                    wmax = wmax,
                    t_fp = t_fp
                );
            }
            st.t_guard_gram = t_guard.elapsed();

            let want = (opts.targets + 1).min(dim);
            let (ev, evecs) = solve(&gram, want, true);
            let energies: Vec<f64> = ev[..want].to_vec();
            let e0 = energies[0];
            let gaps: Vec<f64> = energies[1..].iter().map(|e| e - e0).collect();

            let character: Vec<[f64; 3]> = evecs[..want]
                .iter()
                .map(|c| {
                    let sq = |acc: f64, z: &f64| acc + z * z;
                    [
                        c[0] * c[0],
                        c[1..1 + n_t].iter().fold(0.0f64, sq),
                        c[1 + n_t..dim].iter().fold(0.0f64, sq),
                    ]
                })
                .collect();

            let t_will = Instant::now();
            let williamson = opts.williamson.then(|| {
                let (a_tt, b_tt) = $hessian_grams(&ctx, &basis);
                let normals = gram.normals();
                let n_n = normals.len();
                let nmax = n_t + n_n;
                let mut a_ext = vec![0.0f64; nmax * nmax];
                let mut b_ext = vec![0.0f64; nmax * nmax];
                for i in 0..n_t {
                    for j in 0..n_t {
                        a_ext[i * nmax + j] = a_tt[i * n_t + j];
                        b_ext[i * nmax + j] = b_tt[i * n_t + j];
                    }
                }
                for (m, nm) in normals.iter().enumerate() {
                    for (i, ii) in iota.iter().enumerate() {
                        let v = amb(ii, nm);
                        a_ext[i * nmax + (n_t + m)] = v;
                        a_ext[(n_t + m) * nmax + i] = v;
                    }
                    for (m2, nm2) in normals.iter().enumerate().take(m + 1) {
                        let v = amb(nm, nm2);
                        a_ext[(n_t + m) * nmax + (n_t + m2)] = v;
                        a_ext[(n_t + m2) * nmax + (n_t + m)] = v;
                    }
                }
                symplectic_from_grams(&a_ext, &b_ext, nmax, 1e-10)
            });
            st.t_williamson = t_will.elapsed();

            let response = ConeResponse {
                gaps,
                e0,
                energies,
                seeded,
                added: gram.normals().len(),
                rounds,
                guards: ConeGuards {
                    iso,
                    asym,
                    ortho,
                    gram: gram_dev,
                },
                williamson,
                character,
            };
            let mut matfree_stats = mf_stats.into_inner();
            matfree_stats.dense_equivalent_applies = n_t + gram.normals().len();
            st.t_total = t_entry.elapsed();
            matfree_stats.stages = st;
            Ok(ConeSolveOut {
                response,
                gram,
                evecs,
                matfree_stats,
                lam,
            })
        }
    };
}

pub(crate) use impl_cone;

impl_cone!(
    ham: SparseHamiltonian<f64>,
    vec: TtVector<f64>,
    tangent: TtTangent<f64>,
    ctx: WeingartenCtx,
    grid: RankGrid,
    weingarten_operator: weingarten_operator,
    normal_residual: normal_residual,
    tangent_basis: tangent_basis,
    hessian_grams: hessian_grams,
    xt_h_y: xt_h_y,
    dot: dot,
    tt_randn: crate::vector::tt_randn,
    tag: "",
    gram_note: "",
    cone_response,
    try_cone_response,
    cone_response_matfree,
    cone_response_matfree_with_stats,
    cone_response_with_stats,
);

#[cfg(test)]
mod tests {
    use super::{DRIFT_REARM_FACTOR, MatfreeOpts, drift_voids_record, mixed_lanczos};

    #[test]
    fn mixed_lanczos_recovers_exact_multiplicity() {
        let n = 8usize;
        let d = [-2.0f64, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let v: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let vtv: f64 = v.iter().map(|x| x * x).sum();
        let q = |i: usize, j: usize| -> f64 {
            (if i == j { 1.0 } else { 0.0 }) - 2.0 * v[i] * v[j] / vtv
        };
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for (k, dk) in d.iter().enumerate() {
                    s += q(i, k) * dk * q(j, k);
                }
                a[i * n + j] = s;
            }
        }
        let mut apply = |c: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| (0..n).map(|j| a[i * n + j] * c[j]).sum())
                .collect()
        };
        let stats = std::cell::RefCell::new(super::MatfreeStats::default());
        let (vals, vecs) =
            mixed_lanczos(&mut apply, n, 4, &MatfreeOpts::default(), &stats, &[], true);
        assert_eq!(vals.len(), 4);
        assert!((vals[0] + 2.0).abs() < 1e-9, "vals[0] = {}", vals[0]);
        assert!(
            (vals[1] + 2.0).abs() < 1e-9,
            "vals[1] = {} — second copy lost",
            vals[1]
        );
        assert!((vals[2] + 1.0).abs() < 1e-9);
        assert!((vals[3] - 0.0).abs() < 1e-9);
        for i in 0..4 {
            for j in 0..4 {
                let g: f64 = vecs[i].iter().zip(&vecs[j]).map(|(x, y)| x * y).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((g - want).abs() < 1e-9, "gram[{i}][{j}] = {g}");
            }
            let ay = apply(&vecs[i]);
            let res: f64 = ay
                .iter()
                .zip(&vecs[i])
                .map(|(a, y)| (a - vals[i] * y).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(res < 1e-8, "residual[{i}] = {res:.3e}");
        }
    }

    #[test]
    fn drift_rearm_ignores_convergence_creep() {
        let conv_tol = 1e-5;
        for drop in [0.0, 1e-8, 1.96e-6, 9.99e-5] {
            assert!(
                !drift_voids_record(-1.0, -1.0 - drop, conv_tol),
                "a {drop:.3e} settle at conv_tol {conv_tol:.0e} is creep, not a crossing"
            );
        }
    }

    #[test]
    fn drift_rearm_fires_on_a_crossing() {
        let conv_tol = 1e-5;
        assert!(
            drift_voids_record(-1.0, -1.01, conv_tol),
            "a 1e-2 drop at conv_tol 1e-5 is a level crossing and must re-arm the target"
        );
        assert!(
            drift_voids_record(-1.0, -1.0 - 1.001 * DRIFT_REARM_FACTOR * conv_tol, conv_tol),
            "the gate must be `> DRIFT_REARM_FACTOR · conv_tol`"
        );
        assert!(
            !drift_voids_record(-1.0, -1.0 - 0.999 * DRIFT_REARM_FACTOR * conv_tol, conv_tol),
            "the gate must not fire below `DRIFT_REARM_FACTOR · conv_tol`"
        );
    }

    #[test]
    fn drift_rearm_is_symmetric_in_the_move() {
        let conv_tol = 1e-5;
        let big = 1e-2;
        assert_eq!(
            drift_voids_record(-1.0, -1.0 - big, conv_tol),
            drift_voids_record(-1.0 - big, -1.0, conv_tol),
            "the re-arm must not depend on the SIGN of the move"
        );
        assert!(drift_voids_record(-1.0 - big, -1.0, conv_tol));
    }

    #[test]
    fn drift_rearm_at_the_disable_sentinel() {
        assert!(!drift_voids_record(-1.0, -1.0, 0.0));
        assert!(
            drift_voids_record(-1.0, -1.0 - 1e-12, 0.0),
            "at conv_tol 0 the gate is `> 0`, so any move that is representable at this scale \
             voids the record"
        );
    }
}
