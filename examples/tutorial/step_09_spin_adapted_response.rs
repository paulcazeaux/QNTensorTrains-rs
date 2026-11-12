// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Spin-adapted response: singlet excitation energies on the fixed-rank SU(2) TT manifold.
//!
//! # Introduction
//!
//! This is the singlet-sector twin of step-08. Because the manifold is built at fixed total spin
//! S (here twoS = 0, singlets), the linear response spectrum is the SINGLET excitation spectrum
//! directly — the triplet states the abelian Sz=0 engine also carries (and linear response then
//! has to skip as dark) never enter. This is the natural engine for comparing against a paper that
//! reports singlet excitation energies.
//!
//! Pipeline: load a fixture directory (`meta.txt` + FCIDUMP) → build the closed-shell
//! singlet HF CSF (lowest N/2 orbitals doubly occupied) → enrich to the target per-sector rank grid
//! → SU(2) ALS + Riemannian-CG ground state → Weingarten context → matrix-free
//! `symplectic_lowest_signed_davidson` (the paired RPA Davidson), with the certified dense
//! Williamson oracle on the same point.
//!
//! # Builds on
//!
//! step-07, step-08
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-09                          # LiH/STO-3G, rank 4, six ω
//! cargo run --release --example step-09 -- h2o_sto3g              # water/STO-3G, rank 4, six ω
//! cargo run --release --example step-09 -- h2o_631g 32 8 nodense  # bigger, matrix-free path only
//! cargo run --release --example step-09 -- lih_sto3g 8 6 tol=1e-6 norpa   # ground state only
//! cargo run --release --example step-09 -- --help
//! ```
//!
//! - `fixture` (str, default `lih_sto3g`): a fixture directory (`meta.txt` + FCIDUMP), or a name
//!   under `testdata/`. It must be closed-shell (`ms2 = 0`); the driver asserts that up front.
//! - `rank` (usize, default 4): the manifold rank cap — the top of the MALS bond-dimension ramp
//!   and the per-sector rank of the random tangent start.
//! - `k` (usize, default 6): how many lowest ω to report.
//! - `tol=` (float, optional; absent = the fixed-M ramp): switches MALS to rank-adaptive mode,
//!   uncapped ranks with this SVD truncation tolerance.
//! - `nodense` (flag, default off): skip the dense Williamson oracle.
//! - `norpa` (flag, default off): stop after the ground state.
//! - `--format <text|tsv>` (default `text`): output format.
//!
//! # Results
//!
//! The run prints, in order: the fixture banner and its `d`/`N`/rank/`k` line; which MALS mode was
//! chosen; the verbose MALS sweep log; the MALS energy with its per-bond maximal sector ranks; the
//! CG-polished ground-state energy next to the fixture's HF and reference energies, with the
//! percentage of the correlation energy captured; and `‖grad‖` with the wall time to that point.
//! `‖grad‖` is the criticality check: above 1e-9 the program says so, and the ω below it are to be
//! read as biased, because linear response is only meaningful at a stationary point.
//!
//! The response stage then prints `‖r⊥‖`, the part of `Hx − λx` normal to the tangent space, which
//! is how far the fixed-rank point sits from a true eigenvector; the matrix-free
//! `symplectic_lowest_signed_davidson` timing with its basis, apply and convergence counts; and,
//! unless `nodense` was passed, the dense oracle timing and the tangent dimension `n`. A resonant
//! or unstable point prints a diagnosis instead of a spectrum and the run stops there — that is a
//! real answer about the point, not a failure of the solver.
//!
//! The table is one row per excitation: ω from the matrix-free driver in hartree, the same number
//! in eV, and ω from the dense oracle. The two columns are two independent computations of the same
//! symplectic spectrum at the same point, so the final `max |ω_iter − ω_dense|/ω` line is the
//! cross-check: it is the accuracy of the matrix-free path, not a physical quantity.
//!
//! # Possibilities for extensions
//!
//! - Run the same fixture through step-08 and line the two spectra up: step-08's abelian Sz=0 list
//!   interleaves triplets that linear response cannot reach, this one does not.
//! - Move to `h2o_631g` (d = 13) at a larger `rank`, with `nodense` once the tangent dimension
//!   makes the oracle the expensive half — the `max |ω_iter − ω_dense|` line is what buys the
//!   right to drop it.
//! - Use `tol=` instead of a rank cap and watch the ranks MALS chooses, and the captured
//!   correlation percentage, move together as the tolerance tightens.
//! - Pass `norpa` and sweep `rank`: a cheap ground-state-only convergence curve, without paying for
//!   the tangent-space stage at each point.

use qntt::Scalar;
use qntt::solvers::hess::{SymplecticDavidsonOpts, SymplecticSpectrum};
use qntt::solvers::mals::NoiseType;
use qntt::su2::core::Family;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use qntt::su2::hess::{
    normal_residual, symplectic_eigenvalues, symplectic_lowest_signed_davidson, weingarten_operator,
};
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg};
use qntt::su2::solvers::mals::{MalsOptions, mals};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_csf, tt_randn};
// `‖P(Hy − λy)‖` at `y`, the projected-gradient norm, read from a Weingarten context bound at `y`.
// Called inside the CG-polish loop.
use qntt_exkit::grad::su2::norm_weingarten as grad_norm_at;
use qntt_exkit::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

const HARTREE_EV: f64 = 27.211386245988;

fn main() {
    let m = Spec::new("step-09")
        .about(
            "spin-adapted (SU(2)) RPA / linear-response excitation spectrum on the fixed-rank TT \
             manifold",
        )
        .pos(
            "fixture",
            "fixture directory (meta.txt + FCIDUMP), or a name under testdata/",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
        .pos(
            "rank",
            "manifold rank cap: top of the MALS bond-dimension ramp (pooled) and the per-sector \
             rank of the random tangent start",
            Val::Usize,
            Def::Usize(4),
        )
        .pos(
            "k",
            "how many lowest ω to report",
            Val::Usize,
            Def::Usize(6),
        )
        .key(
            "tol",
            "rank-adaptive MALS: SVD truncation tolerance (uncapped ranks); absent = fixed-M ramp",
            Val::F64,
            Def::Optional,
        )
        .flag("nodense", "skip the dense Williamson oracle")
        .flag("norpa", "stop after the ground state")
        .parse();
    let fixture = m.str("fixture").to_string();
    let rank = m.usize("rank");
    let k = m.usize("k");
    let want_dense = !m.flag("nodense");

    let fx = qntt_exkit::repo!().load(&fixture);
    // Driver precondition, not a format rule.
    assert_eq!(
        fx.ms2(),
        0,
        "SU(2) singlet driver expects twoSz=0 (closed-shell)"
    );
    let (d, nelec) = (fx.d(), fx.nelec());
    let (e_nuc, e_hf, e_fci) = (fx.e_core(), fx.e_hf(), fx.e_ref());
    let (t, v) = fx.into_tv();
    banner(format!(
        "SU(2) singlet RPA/linear-response ω on the fixed-rank TT manifold — {fixture}"
    ));
    println!(
        "d={d} (spatial orbitals), N={nelec} (singlet twoS=0), manifold rank cap {rank}, k={k}"
    );
    let space = Su2Space::new(d, nelec, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    // ---- Ground state: singlet HF CSF → MALS with a decaying noise ramp → CG polish ---------
    // MALS (two-site, rank-adaptive) is the robust GS solver at d≳13 where one-site ALS stalls; the
    // noise schedule enriches the kept basis at each truncation to escape the determinant/local
    // trap that snags a NOISE-FREE HF-start two-site sweep. This step uses NoiseType::Random. With
    // noise live from the first sweep, the HF CSF is the best start — it already carries the mean
    // field, so MALS begins near the minimum rather than climbing out of a random state. The
    // schedule MUST end at 0.0 (the driver will not stop on a noisy iteration, and under Random
    // only the noise-free tail carries a variational energy).
    let t0 = Instant::now();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x52_11A2);
    let nocc = nelec / 2;
    let codes: Vec<Family> = (0..d)
        .map(|i| if i < nocc { Family::Occ } else { Family::Empty })
        .collect();
    let mut x0: Su2TtVector<f64> = tt_csf(&codes);
    x0.right_orthogonalize(true);
    // Two convergence modes (CLI):
    //   tol=<x>  → RANK-ADAPTIVE: uncapped bond_dims, SVD truncation tolerance eps=x. MALS grows
    //              ranks until every bond's discarded weight is under `x` and stops sweeping when
    //              the relative energy improvement falls below `x`. Tighter x → closer to FCI.
    //   (else)   → FIXED-M: bond-dim cap ramped to `rank` (the M-scan mode).
    let tol = m.f64_opt("tol");
    let (bond_dims, mals_eps) = match tol {
        Some(x) => (vec![0usize], x),
        None => (
            vec![
                (rank / 4).max(6),
                (rank / 2).max(10),
                (3 * rank / 4).max(14),
                rank,
                rank,
                rank,
            ],
            1e-10,
        ),
    };
    if let Some(x) = tol {
        println!("MALS mode: rank-adaptive to truncation tolerance eps={x:.1e} (uncapped ranks)");
    } else {
        println!("MALS mode: fixed bond-dim cap ramp to M={rank}");
    }
    let mopts = MalsOptions {
        eps: mals_eps,
        bond_dims,
        // Decaying ramp with a long noise-free tail so the last sweeps are variational.
        noise: vec![
            3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        seed: 0x9E37_79B9,
        noise_type: NoiseType::Random,
        max_iter: 30,
        verbose: true,
        ..Default::default()
    };
    let (lmals, xa) = mals(&ham, &x0, &mopts);
    println!(
        "MALS:          E = {:.9} Ha, per-bond max sector ranks {:?}",
        lmals + e_nuc,
        qntt_exkit::ranks::su2::per_bond_max(&space, &xa)
    );

    // MALS lands close, but linear response needs a CRITICAL point, not just a low energy: the
    // Hessian of a non-stationary point has no physical spectrum. So restart Riemannian CG until
    // the projected gradient is below 1e-10, then report ‖grad‖ so a run that failed to get there
    // is visible in the output rather than hidden inside the ω.
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let cg_polish = |x0: &Su2TtVector<f64>| -> Su2TtVector<f64> {
        let (_l, mut y) = riemannian_cg(&ham, x0, &opts);
        for _ in 0..30 {
            if grad_norm_at(&ham, &y) < 1e-10 {
                break;
            }
            let (_l2, y2) = riemannian_cg(&ham, &y, &opts);
            y = y2;
        }
        y
    };
    let mut x = cg_polish(&xa);
    normalize_mut(&mut x);
    x.check();
    let gnorm = grad_norm_at(&ham, &x);
    let lam = rayleigh_quotient(&ham, &x).re_s();
    println!(
        "ground state:  E = {:.9} Ha  (HF {:.9}, ref {:.9}; corr. captured {:.1}%)",
        lam + e_nuc,
        e_hf,
        e_fci,
        100.0 * (lam + e_nuc - e_hf) / (e_fci - e_hf)
    );
    println!(
        "               ‖grad‖ = {gnorm:.3e}, ranks {:?}, converged in {:.2}s",
        qntt_exkit::ranks::su2::per_bond_max(&space, &x),
        t0.elapsed().as_secs_f64()
    );
    if gnorm > 1e-9 {
        println!("WARNING: not fully critical (‖grad‖ > 1e-9) — the spectrum below may be biased");
    }

    // `norpa`: stop after the ground state (cheap GS-energy convergence curve, skip the costly
    // d≳13 tangent-space RPA).
    if m.flag("norpa") {
        return;
    }

    // ---- Weingarten context + matrix-free lowest-k ω --------------------------------------------
    //
    // The Weingarten context binds the ground state once and caches the environments every tangent
    // apply needs; `ctx.bases()` are the left/right orthonormal frames that define the tangent
    // space at `x`. Everything below — the normal residual, the matrix-free Davidson, the dense
    // oracle — is evaluated at this single point and through this one context.
    let ctx = weingarten_operator(&ham, &x);
    let (base_l, base_r) = {
        let (l, r) = ctx.bases();
        (l.clone(), r.clone())
    };
    // `‖grad‖` above measured the TANGENT part of `Hx − λx`, which CG drove to zero. This is the
    // NORMAL part, which it cannot touch: it is the error the rank cap itself imposes, and it stays
    // wherever the manifold leaves it however well converged the point is.
    let rperp = normal_residual(&ham, &base_l, &base_r, &x, lam).norm();
    println!("‖r⊥‖ = {rperp:.3e}");

    // The Davidson solve needs starting directions IN the tangent space: draw k + 2 random trains at
    // the same rank grid and project each onto the frames. The projection, not the draw, is what
    // makes them tangent.
    let v0s: Vec<Su2TtTangent<f64>> = (0..k + 2)
        .map(|_| {
            let dirv: Su2TtVector<f64> = tt_randn(
                space,
                &qntt_exkit::ranks::su2::uniform(&space, rank),
                &mut rng,
            );
            Su2TtTangent::project(&base_l, &base_r, &dirv)
        })
        .collect();
    let t1 = Instant::now();
    let (so, _) =
        symplectic_lowest_signed_davidson(&ctx, &v0s, k, &SymplecticDavidsonOpts::default());
    // ω is published only for a converged solve that met no anomaly.
    let iterative = match so.omega() {
        Some(omega) => {
            println!(
                "iterative ω:   {:.2}s — {} basis vectors, {} applies ({} kernel), converged=true",
                t1.elapsed().as_secs_f64(),
                so.iterations,
                so.applies,
                so.report.kernel_name()
            );
            Some(omega)
        }
        None => {
            println!(
                "iterative ω:   no stable spectrum: converged={}, ω² {:?}, report {:?}",
                so.converged,
                so.modes.iter().map(|m| m.omega2).collect::<Vec<_>>(),
                so.report
            );
            None
        }
    };

    // The oracle: materialize an explicit basis of the whole tangent space, form the Grams, and
    // solve the Williamson problem densely. Same point, same operator, no Krylov approximation —
    // and a cost that grows with the tangent dimension `n`, which is why it is optional.
    let dense = if want_dense {
        use qntt::su2::weingarten::tangent_basis;
        let t2 = Instant::now();
        let basis = tangent_basis(&base_l, &base_r, true);
        let n = basis.len();
        match symplectic_eigenvalues(&ctx, &basis) {
            SymplecticSpectrum::Stable { omega } => {
                println!(
                    "dense oracle:  {:.2}s — tangent dim n={n}",
                    t2.elapsed().as_secs_f64()
                );
                Some(omega)
            }
            SymplecticSpectrum::Resonant { signed } => {
                println!(
                    "dense oracle:  RESONANT ({} unstable pair(s), metric {:?}, lowest ω² = \
                     {:.3e}), n={n} — no real ω spectrum here",
                    signed.n_neg,
                    signed.metric,
                    signed.omega2.first().copied().unwrap_or(0.0)
                );
                None
            }
            SymplecticSpectrum::Unstable { reason, .. } => {
                println!("dense oracle:  Unstable ({reason:?}), n={n}");
                None
            }
        }
    } else {
        None
    };

    let omega = match iterative {
        Some(o) => o,
        None => {
            println!("\nNo stable spectrum at this fixed-rank point.");
            return;
        }
    };

    // Every ω here is a singlet: the manifold carries no other spin. They are fixed-rank
    // linear-response ESTIMATES of the excitation energies, exact only where the manifold is the
    // full space.
    println!("\nlowest {} SINGLET excitation estimates:", omega.len());
    println!(
        "  {:>3}  {:>14}  {:>10}  {:>14}",
        "j", "ω (iterative)", "ω (eV)", "ω (dense)"
    );
    for (j, w) in omega.iter().enumerate() {
        let dense_s = dense
            .as_ref()
            .and_then(|dv| dv.get(j))
            .map(|wd| format!("{wd:>14.8}"))
            .unwrap_or_else(|| format!("{:>14}", "—"));
        println!("  {j:>3}  {w:>14.8}  {:>10.4}  {dense_s}", w * HARTREE_EV);
    }
    if let Some(dv) = &dense {
        let max_rel = omega
            .iter()
            .zip(dv.iter())
            .map(|(wi, wd)| (wi - wd).abs() / wd.max(1e-12))
            .fold(0.0f64, f64::max);
        println!("max |ω_iter − ω_dense|/ω over the reported k: {max_rel:.3e}");
    }
}
