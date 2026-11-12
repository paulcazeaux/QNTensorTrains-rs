// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Symplectic (RPA / linear-response) excitation spectrum of a small molecule on the fixed-rank
//! TT manifold, end to end.
//!
//! # Introduction
//!
//! Pipeline: load a pyscf-exported fixture from `testdata/` (`meta.txt` + the FCIDUMP it names;
//! `lih_sto3g`, `h2o_sto3g`, `n2_ccpvdz`, … all work) → converge a fixed-rank Riemannian ground
//! state (HF determinant + noise warm start → ALS → the CERTIFIED polish:
//! [`riemannian_cg_certified`] under [`Retraction::AlsRounded`]) → bind the Weingarten context →
//! matrix-free [`symplectic_lowest_signed_davidson`] (the paired RPA Davidson).
//!
//! The polish is the step's second lesson. A plain Riemannian CG on a rank-capped manifold
//! converges slowly, taking many small steps; the ALS-rounded retraction re-minimizes each core
//! once after the truncation and allows much longer steps. And a descent that stops on its own
//! criterion has proved criticality, not minimality: the certified polish probes `A + B`
//! there, and a certified negative direction is kicked along before the descent resumes, so what
//! the response stage receives is a reported minimizer, a reported saddle, or an honest
//! "inconclusive" — never a silent one.
//!
//! Cross-checks (each on by default, individually skippable):
//! - `nodense` skips the dense driver (`tangent_basis` → Grams → Williamson) run on the same
//!   point, whose ω are printed side by side with the matrix-free ones.
//! - `nofci` skips the exact determinant-basis diagonalization (auto-skipped above 4096 dets):
//!   exact `E_k − E_0` printed next to ω for physical context. NOTE ω are fixed-rank
//!   LINEAR-RESPONSE estimates — they approximate, not equal, the FCI gaps; they coincide exactly
//!   only when the manifold is the full space (e.g. `rank` ≥ every sector's maximal rank).
//!
//! # Builds on
//!
//! step-02, step-05
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-08                        # LiH/STO-3G, rank cap 2, six ω
//! cargo run --release --example step-08 -- h2o_sto3g 3 8       # water, rank cap 3, eight ω
//! cargo run --release --example step-08 -- lih_sto3g 2 6 nofci # no determinant-basis check
//! cargo run --release --example step-08 -- --help
//! ```
//!
//! Arguments — every one optional. The three positionals are read in the order below; `nodense`
//! and `nofci` are bare-token switches matched by name, so they may sit anywhere in the argument
//! list (the third invocation above names `nofci` without naming `nodense` first):
//!
//! - `fixture` — subdir of `testdata/` (default `lih_sto3g`)
//! - `rank` — per-(bond,sector) interior rank cap of the manifold (default `2`)
//! - `k` — number of lowest ω to compute (default `6`)
//! - `nodense` — skip the certified dense driver (a switch; default off)
//! - `nofci` — skip the exact determinant-basis diagonalization (a switch; default off)
//!
//! `--format text|tsv` selects the output format (default `text`; the ambient default comes from
//! `QNTT_FORMAT`), and `-h`/`--help` prints the argument list above.
//!
//! # Results
//!
//! A run prints the ground-state stage first, then the two response routes, then the table of ω.
//! On the default LiH/STO-3G at rank cap 2 the HF+noise start sits at −7.742631071 Ha, ALS brings
//! it down to −7.882361410 Ha, and the certified polish leaves it there: 9 CG iterations under the
//! ALS-rounded retraction to `‖grad‖ = 5.4e-13` (exit on the gradient stop), then the `A + B` and
//! `A − B` probes both report PD, so the point is a certified minimizer, not merely critical —
//! 99.8% of the correlation energy (HF −7.862023860, FCI −7.882401932). Read the two lowest Ritz
//! values the probes converged, 0.11595 (`A + B`) and 0.11605 (`A − B`): their geometric mean
//! 0.11600 is the `ω₀` of the table below, the certificate and the response agreeing with each
//! other. The re-polish after the trim takes 0 iterations (nothing was dropped at this cap) and
//! re-certifies the same point. `‖r⊥‖ = 1.3e-2` is the part of `Hx` that leaves the tangent
//! space; it is nonzero exactly because the manifold is truncated, and it is what makes the
//! Weingarten term B live. The per-bond Schmidt block is the curvature scale: `σ_min` down at
//! 1.4e-4 on bond 2 means a strongly curved manifold there.
//!
//! The two response routes then agree to 7.6e-14 relative over the reported k — the matrix-free
//! paired Davidson (71 basis vectors, one `A`- and one `B`-apply each) against the dense
//! Williamson oracle on the same tangent space (n = 71). At this size the basis grows to the whole
//! tangent space, so the solve ends exhausted; on a larger manifold it stops at `tol` long before.
//! That agreement is the point of the step: the cheap route is the certified one.
//!
//! The FCI column is the physics check, and it is honest about the truncation. The two lowest ω,
//! 0.11600146 and 0.13320608 Ha (3.16 and 3.62 eV), reproduce the exact gaps 0.11598346 and
//! 0.13318575 to ~2e-5 Ha; from `j = 2` up they do not — 0.39986901 against 0.16594792 — because
//! a rank-2 manifold simply does not contain those excitations. Note also that the FCI column
//! repeats degenerate gaps (0.16594792 twice, 0.18544994 twice) that the rank-2 manifold does not
//! contain, so the two columns are not row-by-row partners.
//!
//! # Possibilities for extensions
//!
//! - Raise `rank` to 3 or 4 and watch the higher ω fall towards the FCI gaps; at a cap that no
//!   sector's maximal rank exceeds the manifold is the whole sector and the two columns coincide.
//! - Run `h2o_sto3g` (seven orbitals, ten electrons) for a harder operator, or `n2_ccpvdz`, where
//!   the determinant count passes 4096 and the FCI cross-check skips itself.
//! - As `rank` — and with it the tangent dimension n — grows, the dense oracle (n `A`- and n
//!   `B`-applies plus an n² Gram build) becomes expensive. Pass `nodense` once it stops being
//!   affordable and keep the matrix-free column alone.
//! - Print the trimmed rank grid against the requested one (`per_bond_max` before and after the
//!   `round_mut` trim): when a sector's σ is at machine zero the requested grid was never
//!   reached, and the well-posed manifold is the trimmed one.
//! - Switch `opts.retraction` back to [`Retraction::Hosvd`] and count the CG iterations the
//!   plain line search needs for the same `‖grad‖`.

use qntt::core::factor::dense_self_adjoint_smallest;
use qntt::core::tangent::{TtTangent, tangent_basis};
use qntt::core::weingarten::{FloorPolicy, bond_gauges};
use qntt::round::RoundMode;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::hess::{
    PdVerdict, SymplecticDavidsonOpts, SymplecticSpectrum, normal_residual, symplectic_eigenvalues,
    symplectic_lowest_signed_davidson, weingarten_operator,
};
use qntt::solvers::riemannian::{
    CertifyOpts, CertifyReport, CertifyVerdict, Retraction, RiemannianOpts, normalize_mut,
    riemannian_cg_certified,
};
use qntt::{QnSpace, Scalar, SparseHamiltonian, TtVector, dot, rayleigh_quotient, tt_state};
use qntt_exkit::grad::abelian::norm_weingarten as grad_norm_at;
use qntt_exkit::prelude::*;
use qntt_exkit::ranks;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

const HARTREE_EV: f64 = 27.211386245988;

/// One line for the certified polish's report: the verdict, the lowest Ritz value each block
/// probe converged, and the kicks it took to get there. The Ritz values are what one seeded
/// Davidson converged, not a proven `λ_min`; the dense oracle below is the exact statement at
/// this scale — and the reader can check that
/// `√(λ_min(A+B)·λ_min(A−B))` reproduces its lowest ω.
fn certificate_line(rep: &CertifyReport<f64>) -> String {
    let kicks = match rep.kicks.len() {
        0 => String::new(),
        n => format!(" after {n} kick(s) along a certified negative direction"),
    };
    let ii = match rep.ii {
        Some(PdVerdict::LikelyPd { lambda_min, .. }) => format!(", A−B PD (λ_min {lambda_min:.5})"),
        Some(PdVerdict::WitnessNegative { lambda }) => {
            format!(", A−B NEGATIVE (λ {lambda:+.2e}, complex instability)")
        }
        Some(PdVerdict::Inconclusive { .. }) => ", A−B inconclusive".to_string(),
        None => String::new(),
    };
    match rep.verdict {
        CertifyVerdict::Minimizer { lambda_rr } => {
            format!("certified minimizer (A+B PD, λ_min {lambda_rr:.5}{ii}){kicks}")
        }
        CertifyVerdict::Saddle { witness_lambda } => {
            format!("SADDLE (A+B witness λ = {witness_lambda:+.2e}{ii}){kicks}")
        }
        CertifyVerdict::Inconclusive { .. } => format!("A+B probe inconclusive{ii}{kicks}"),
        CertifyVerdict::NotProbed => format!("not probed (descent ended on {:?})", rep.exit),
    }
}

const TRIM_EPS: f64 = 1e-8;

/// All n-of-d index combinations, lexicographic (determinant enumeration for the FCI check).
fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut c: Vec<usize> = (0..n).collect();
    if n > d {
        return out;
    }
    loop {
        out.push(c.clone());
        // next lexicographic combination
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if c[i] != i + d - n {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        c[i] += 1;
        for j in i + 1..n {
            c[j] = c[j - 1] + 1;
        }
    }
}

fn occ_mask(d: usize, sites: &[usize]) -> Vec<bool> {
    let mut m = vec![false; d];
    for &s in sites {
        m[s] = true;
    }
    m
}

fn main() {
    let m = Spec::new("step-08")
        .about(
            "Symplectic (RPA / linear-response) excitation spectrum of a small molecule on the \
             fixed-rank TT manifold",
        )
        .pos(
            "fixture",
            "subdir of `testdata/`",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
        .pos(
            "rank",
            "per-(bond,sector) interior rank cap of the manifold",
            Val::Usize,
            Def::Usize(2),
        )
        .pos(
            "k",
            "number of lowest ω to compute",
            Val::Usize,
            Def::Usize(6),
        )
        .flag("nodense", "skip the certified dense driver")
        .flag("nofci", "skip the exact determinant-basis diagonalization")
        .parse();
    let fixture = m.str("fixture").to_string();
    let rank: usize = m.usize("rank");
    let k: usize = m.usize("k");
    let want_dense = !m.flag("nodense");
    let want_fci = !m.flag("nofci");

    // ---- 1. the fixture and its operator ----------------------------------------------------
    //
    // Same three objects as step-01: a fixture directory under `testdata/`, the space it declares,
    // and the Hamiltonian assembled at `f64::EPSILON` (every integral kept — this step compares
    // against an exact diagonalization, so the operator must not be approximated).
    let repo = qntt_exkit::repo!();
    let fx = repo.load(&fixture);
    let (d, nup, ndn) = (fx.d(), fx.nup(), fx.ndn());
    let (e_nuc, e_hf, e_fci) = (fx.e_core(), fx.e_hf(), fx.e_ref());
    banner(format_args!(
        "RPA/linear-response ω on the fixed-rank TT manifold — {fixture}"
    ));
    println!("d={d} (spatial orbitals), N↑={nup} N↓={ndn}, manifold rank cap {rank}, k={k}");
    let space = QnSpace::new(d, nup, ndn);
    let ham = SparseHamiltonian::new(space, fx.t(), fx.v(), None, f64::EPSILON);

    // ---- 2. the point the spectrum is taken at ----------------------------------------------
    //
    // Linear response is a property of a CRITICAL point of the energy on the manifold, so this
    // stage is not optional scaffolding: the ω below are only meaningful once ‖grad‖ is at
    // machine zero. Fixed-rank Riemannian ground state: HF + noise warm start, GD + CG restarts.
    let t0 = Instant::now();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x11A2);
    let hf: TtVector<f64> = tt_state(
        &occ_mask(d, &(0..nup).collect::<Vec<_>>()),
        &occ_mask(d, &(0..ndn).collect::<Vec<_>>()),
    );
    // Enrich HF to the target per-sector rank grid (`perturbation` direct-sums a random train
    // sized to reach the targets), then land EXACTLY on the grid with the fixed-rank round.
    // NB `round_mut`'s `chi` is a POOLED (total-per-bond) cap, not per-sector — using it here collapses
    // every sector to rank 1 (the HF direction dominates the shared budget) and silently swaps
    // the manifold for the rank-1 one; `round_ranks_mut` is the per-sector tool.
    let grid = ranks::abelian::uniform(space, rank);
    let mut x0 = qntt::vector::perturbation(&hf, &grid, 0.2, &mut rng);
    x0.round_ranks_mut(&grid);
    normalize_mut(&mut x0);
    x0.check();
    let per_bond_max = |y: &TtVector<f64>| -> Vec<usize> { ranks::abelian::per_bond_max(space, y) };
    println!(
        "warm start:    E(x0) = {:.9} Ha, per-bond max sector ranks {:?}",
        rayleigh_quotient(&ham, &x0).re_s() + e_nuc,
        per_bond_max(&x0)
    );

    // ALS warm start at the fixed rank grid: without it, plain CG from HF+noise converges to a
    // high-lying critical point (on lih_sto3g: 5% of the correlation energy captured, and the
    // response stage correctly reports a saddle).
    let (lals, xa) = als(
        &ham,
        &x0,
        &AlsOptions {
            eps: 1e-10,
            max_iter: 60,
            ..AlsOptions::default()
        },
    );
    println!(
        "ALS:           E = {:.9} Ha, ranks {:?}",
        lals.re_s() + e_nuc,
        per_bond_max(&xa)
    );
    // The certified polish: CG under the ALS-rounded retraction (`pairs` ALS sweep pairs after
    // each truncated step; `delta0: None` lets the first step set the initial trust radius), and
    // at the point where the descent stops on its own criterion
    // an `A + B` probe that certifies the minimizer or kicks along a proven negative direction and
    // resumes. The report says which happened; nothing has to be re-entered by hand.
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    opts.retraction = Retraction::AlsRounded {
        pairs: 1,
        delta0: None,
    };
    let copts = CertifyOpts {
        probe_budget: 200,
        ..CertifyOpts::default()
    };
    let polish = |x0: &TtVector<f64>, label: &str| -> TtVector<f64> {
        let rep = riemannian_cg_certified(&ham, x0, &opts, &copts, None);
        println!(
            "{label:<14} {} CG iterations, exit {:?}, ‖grad‖ = {:.3e}; {}",
            rep.iterations,
            rep.exit,
            rep.gnorm,
            certificate_line(&rep)
        );
        rep.x
    };
    let mut x = polish(&xa, "polish:");

    // Trim numerically rank-deficient directions and re-polish: if the converged state does not
    // need part of the requested grid (Schmidt σ at machine-zero — e.g. lih_sto3g at cap 2
    // carries σ ≈ 1e-14 on two bonds), it sits at a BOUNDARY point of the fixed-rank manifold
    // where the geometry degenerates (curvature ~ 1/σ) and fixed-rank linear response is
    // ill-posed — the iterative path correctly diagnoses NegativeRitz there. The well-posed
    // object is the manifold of the state's own trimmed rank grid.
    normalize_mut(&mut x);
    x.round_mut(TRIM_EPS, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    x = polish(&x, "re-polish:");
    normalize_mut(&mut x);
    x.check();
    println!(
        "trimmed grid:  per-bond max sector ranks {:?} (dropped σ < {TRIM_EPS:.0e} rel.)",
        per_bond_max(&x)
    );
    let gnorm = grad_norm_at(&ham, &x);
    let lam = rayleigh_quotient(&ham, &x).re_s();
    println!(
        "ground state:  E = {:.9} Ha  (HF {:.9}, FCI {:.9}; corr. captured {:.1}%)",
        lam + e_nuc,
        e_hf,
        e_fci,
        100.0 * (lam + e_nuc - e_hf) / (e_fci - e_hf)
    );
    println!(
        "               ‖grad‖ = {gnorm:.3e}, converged in {:.2}s",
        t0.elapsed().as_secs_f64()
    );
    if gnorm > 1e-9 {
        println!(
            "WARNING: not fully critical (‖grad‖ > 1e-9) — the symplectic spectrum below may be \
             biased or diagnosed unstable"
        );
    }

    // ---- 3. the Weingarten context and the matrix-free lowest-k ω ---------------------------
    //
    // `weingarten_operator` binds the orthonormal left/right frames of `x` once; every later apply
    // reuses them. The Hessian on a fixed-rank manifold has two parts — the Euclidean term and the
    // curvature (Weingarten) term B, which is live precisely when `‖r⊥‖ ≠ 0`.
    let ctx = weingarten_operator(&ham, &x);
    let (base_l, base_r) = {
        let (l, r) = ctx.bases();
        (l.clone(), r.clone())
    };
    let rperp = normal_residual(&ham, &base_l, &base_r, &x, lam).norm();
    println!("‖r⊥‖ = {rperp:.3e}  (truncated manifold ⇒ Weingarten term B live)");
    // Schmidt spectrum at each interior bond — Σ⁻¹ is the intrinsic curvature scale;
    // a tiny σ_min means a strongly curved manifold and a large B.
    let gauges = bond_gauges(&base_l, &base_r, FloorPolicy::default_hard());
    for g in &gauges {
        let (smin, smax) = g
            .svals()
            .iter()
            .flat_map(|(_, v)| v.iter())
            .fold((f64::INFINITY, 0.0f64), |(lo, hi), &s| {
                (lo.min(s), hi.max(s))
            });
        println!(
            "  bond {}: Schmidt σ ∈ [{smin:.3e}, {smax:.3e}], dropped mass {:.1e}",
            g.k, g.dropped_mass
        );
    }

    // The Davidson solve needs start vectors IN the tangent space: draw k + 2 trains on the same
    // rank grid and project each onto the tangent space at `x`.
    let v0s: Vec<TtTangent<f64>> = (0..k + 2)
        .map(|_| {
            let dirv: TtVector<f64> =
                qntt::tt_randn(space, &ranks::abelian::uniform(space, rank), &mut rng);
            TtTangent::project(&base_l, &base_r, &dirv)
        })
        .collect();
    let t1 = Instant::now();
    let (so, _) =
        symplectic_lowest_signed_davidson(&ctx, &v0s, k, &SymplecticDavidsonOpts::default());
    let iter_time = t1.elapsed().as_secs_f64();
    // ω is published only for a converged solve that met no anomaly. An
    // imaginary or complex mode, or a negative Krein margin, means the point is critical but not
    // RPA-stable; its modes are still reported.
    let iterative = match so.omega() {
        Some(omega) => {
            println!(
                "iterative ω:   {:.2}s — {} basis vectors, {} applies ({} kernel), converged=true",
                iter_time,
                so.iterations,
                so.applies,
                so.report.kernel_name()
            );
            Some(omega)
        }
        None => {
            println!(
                "iterative ω:   {:.2}s — no stable spectrum: converged={}, ω² {:?}, report {:?} \
                 — the fixed-rank critical point is not RPA-stable (the dense generator below \
                 gives the independent diagnosis)",
                iter_time,
                so.converged,
                so.modes.iter().map(|m| m.omega2).collect::<Vec<_>>(),
                so.report
            );
            println!(
                "               pencil residuals ‖Hz − ωΣz‖/‖z‖: {:?}",
                so.residuals()
            );
            None
        }
    };

    // ---- 4. dense oracle on the same point. Runs even when the
    // iterative path diagnosed instability: the generator route gives the independent second
    // opinion (±ω structure / complex pairs). -------------------------------------------------
    let dense = if want_dense {
        let t2 = Instant::now();
        let basis = tangent_basis(&base_l, &base_r, true);
        let n = basis.len();
        match symplectic_eigenvalues(&ctx, &basis) {
            SymplecticSpectrum::Stable { omega } => {
                println!(
                    "dense oracle:  {:.2}s — tangent dim n={n}, {} A- + {} B-applies + n² Grams \
                     + Williamson",
                    t2.elapsed().as_secs_f64(),
                    n,
                    n
                );
                Some(omega)
            }
            SymplecticSpectrum::Resonant { signed } => {
                println!(
                    "dense oracle:  RESONANT — {} unstable pair(s) (metric {:?}, λ_min = {:.3e}); \
                     signed ω² lowest = {:.4e} ⇒ ω = {:.4}i, n={n}",
                    signed.n_neg,
                    signed.metric,
                    signed.metric_min_eig,
                    signed.omega2.first().copied().unwrap_or(0.0),
                    signed.omega2.first().copied().unwrap_or(0.0).abs().sqrt()
                );
                None
            }
            SymplecticSpectrum::Unstable {
                reason,
                generator_eigs,
            } => {
                let max_im = generator_eigs.iter().fold(0.0f64, |a, e| a.max(e.im.abs()));
                let min_re = generator_eigs
                    .iter()
                    .filter(|e| e.re > 0.0)
                    .fold(f64::INFINITY, |a, e| a.min(e.re));
                println!(
                    "dense oracle:  Unstable ({reason:?}) — generator eig(𝓜): max|Im| = \
                     {max_im:.3e}, smallest positive Re = {min_re:.4}, n={n}"
                );
                None
            }
        }
    } else {
        None
    };
    let omega = match iterative {
        Some(o) => o,
        None => {
            println!(
                "\nNo stable spectrum at this fixed-rank point — both routes agree it is not an \
                 RPA-stable minimizer. Try a different [rank] (manifold) or inspect the Schmidt \
                 spectrum above."
            );
            return;
        }
    };

    // ---- 5. exact FCI gaps for physical context (small determinant spaces only) -------------
    //
    // Enumerate every determinant of the sector as a rank-1 train, build the dense matrix by
    // applying `ham` once per column, and diagonalize. Only affordable because the fixtures here
    // are tiny — the `ndet <= 4096` guard just below skips the whole stage on anything larger.
    let ups = combinations(d, nup);
    let dns = combinations(d, ndn);
    let ndet = ups.len() * dns.len();
    let fci = if want_fci && ndet <= 4096 {
        let t3 = Instant::now();
        let mut states: Vec<TtVector<f64>> = Vec::with_capacity(ndet);
        for u in &ups {
            for dn in &dns {
                states.push(tt_state(&occ_mask(d, u), &occ_mask(d, dn)));
            }
        }
        let mut hmat = vec![0.0f64; ndet * ndet];
        for (j, sj) in states.iter().enumerate() {
            let hs = &ham * sj;
            for (i, si) in states.iter().enumerate() {
                hmat[i * ndet + j] = dot(si, &hs);
            }
        }
        let (evals, _) = dense_self_adjoint_smallest::<f64>(&hmat, ndet, 1);
        println!(
            "FCI check:     {:.2}s — {ndet} determinants, E_0 = {:.9} Ha (meta: {:.9})",
            t3.elapsed().as_secs_f64(),
            evals[0] + e_nuc,
            e_fci
        );
        Some(evals)
    } else {
        if want_fci {
            println!("FCI check:     skipped ({ndet} determinants > 4096)");
        }
        None
    };

    // ---- 6. report --------------------------------------------------------------------------
    println!(
        "\nlowest {} excitation estimates (Hartree / eV):",
        omega.len()
    );
    let mut t = Table::new(vec![
        Col::int("j", 5),
        Col::float("omega_iter", 15, 8).header("ω (iterative)"),
        Col::float("omega_ev", 11, 4).header("ω (eV)"),
        Col::float("omega_dense", 15, 8).header("ω (dense)"),
        Col::float("fci_gap", 15, 8).header("FCI E_j − E_0"),
    ])
    .no_rule()
    .start();
    for (j, w) in omega.iter().enumerate() {
        let dense_s = dense
            .as_ref()
            .and_then(|dv| dv.get(j))
            .map(|wd| format!("{wd:>14.8}"))
            .unwrap_or_else(|| format!("{:>14}", "—"));
        let fci_s = fci
            .as_ref()
            .and_then(|ev| ev.get(j + 1).map(|e| e - ev[0]))
            .map(|de| format!("{de:>14.8}"))
            .unwrap_or_else(|| format!("{:>14}", "—"));
        t.row(row![j, *w, w * HARTREE_EV, dense_s, fci_s]);
    }
    t.finish();
    if let Some(dv) = &dense {
        let max_rel = omega
            .iter()
            .zip(dv.iter())
            .map(|(wi, wd)| (wi - wd).abs() / wd.max(1e-12))
            .fold(0.0f64, f64::max);
        println!("max |ω_iter − ω_dense|/ω over the reported k: {max_rel:.3e}");
    }
    println!(
        "\nNOTE: ω are fixed-rank linear-response estimates; they differ from the FCI gaps by \
         the manifold truncation (the FCI column repeats degenerate gaps the manifold may not \
         contain). Increase [rank] to watch ω approach E_j − E_0."
    );
}
