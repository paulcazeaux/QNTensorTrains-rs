// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Two-site MALS from a determinant start: what each noise flavour recovers of E_corr.
//!
//! # Introduction
//!
//! Two-site MALS in the canonical orbital order from the Hartree–Fock determinant, run with three
//! truncation modes — noise-free, random and perturbative (White density-matrix) noise — at a few
//! bond dimensions `M`, printing the percentage of the correlation energy each recovers.
//!
//! One-site ALS keeps the ranks it is given (step-01, step-02): started from a determinant — a
//! rank-1 train — it can only ever return the determinant's own energy. The two-site sweep (MALS)
//! is the answer to that. It merges two neighbouring cores, solves the local eigenproblem on the
//! merged block, then splits the block again and keeps whatever the truncation decides to keep, so
//! the rank at that bond is chosen rather than fixed. `bond_dims` here is a ramp: the cap on the
//! TOTAL bond dimension doubles 8, 16, 32, … per outer iteration until it clamps at `M`.
//!
//! What the split keeps is where the three arms differ, and where the determinant start bites:
//!
//! - **noise-free** — the plain SVD of the merged block, an all-zero schedule. The control arm.
//! - **random** — an isotropic random perturbation of the block before its SVD
//!   ([`NoiseType::Random`]). Amplitude-sensitive and not variational (the STATE is perturbed, not
//!   merely the choice of kept basis), but it is the one route out of a BARE determinant start,
//!   where an enrichment built from the current state has no direction to grow into.
//! - **perturbative** — White-style density-matrix noise ([`NoiseType::Perturbative`]), the crate
//!   default: the kept basis is enriched with the operator applied to the state-weighted two-site
//!   block — deterministic, operator-targeted, and variational throughout. Its one
//!   counter-indication is exactly the start used here: the enrichment cannot leave the current
//!   state's subspace, so from a bare determinant MALS can stall at a genuine manifold critical
//!   point at every rank cap, uncapped included.
//!
//! So the table is not a ranking of the flavours in general: a bare determinant is the one start
//! where the recommended default is the wrong choice.
//!
//! The schedule is amplitude 1e-4 for the first four outer iterations, 1e-5 for the next three and
//! zero from the eighth on, so every run ends on sweeps that take the noise-free path exactly and
//! no reported energy is computed while noise is in force. `QNTT_NOISE_SCALE` multiplies the
//! whole schedule.
//!
//! # Builds on
//!
//! step-02, step-03
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-05
//! QNTT_N2_MS=16,32,64 cargo run --release --example step-05
//! QNTT_N2_DIR=/tmp/n2_sto3g QNTT_N2_MS=16,32,64 cargo run --release --example step-05
//! cargo run --release --example step-05 -- --help
//! ```
//!
//! ```text
//! --format <text|tsv>   output format                                     [default text]
//! -h, --help            this text
//!
//! environment:
//! QNTT_N2_DIR=<path>         integral directory (meta.txt + FCIDUMP); defaults to the
//!                            cc-pVDZ N2 fixture
//! QNTT_N2_MS=<usize>[,…]     comma-separated bond-dimension list           [default 64,128]
//! QNTT_NOISE_SCALE=<float>   scales the noise schedule                     [default 1.0]
//! ```
//!
//! There are no positional arguments: the fixture and the M ladder are set through the
//! environment, and a bare name in `QNTT_N2_DIR` resolves under `testdata/`. `--format tsv`
//! switches the report to a machine-readable form.
//!
//! # Results
//!
//! A run prints one banner line, a table caption, then one row per entry of the M ladder. The
//! banner reports the fixture's `d`, `nelec` and `ms2`, the Rayleigh quotient of the HF
//! determinant as a train against the `e_hf` its meta.txt records, and the correlated reference
//! the percentages are taken against. On the default N₂/cc-pVDZ fixture it reads
//!
//! ```text
//! d=28 nelec=14 ms2=0  E_hf(tt)=-108.94937788 (ref -108.94937788)  E_ref=-109.28030507
//! ```
//!
//! — the determinant reproduces the fixture's own HF energy to the printed digits, which fixes the
//! two ends of the scale: 0% of E_corr is that determinant, 100% is `E_ref`. Each ladder row then
//! prints the three arms at that `M`: the percentage of E_corr each recovered after at most 20
//! MALS sweeps, with the largest total bond rank any sweep of that arm reached in brackets (the
//! noise-free arm reports no rank). Read a row across, not down — the three runs share the
//! fixture, the start, the ramp, the schedule and the seed, and differ only in how the two-site
//! block was truncated.
//!
//! At the default ladder (`M = 64, 128`) that is six 20-sweep MALS runs on a `d = 28` operator,
//! which is minutes of work. A short ladder, or a smaller fixture, is what makes this step
//! interactive.
//!
//! # Possibilities for extensions
//!
//! - Set a cheap ladder (`QNTT_N2_MS=8,16,32`): the question the table answers — at which `M` each
//!   arm stops gaining — is visible at the small end, and the default 64,128 is not.
//! - Point `QNTT_N2_DIR` at a smaller tracked fixture (`lih_sto3g`, `h2o_sto3g`): any directory
//!   with a `meta.txt` carrying `e_hf` and a correlated reference energy works, and the ladder
//!   finishes in seconds rather than minutes.
//! - Sweep `QNTT_NOISE_SCALE` (0.1, 1, 10) at one fixed `M`: `Random` is documented as
//!   amplitude-sensitive and `Perturbative` as portable across bonds and molecules at a single
//!   nominal amplitude, and this is the knob that puts the two claims side by side.
//! - Print the whole `hist.sweeps` trace instead of its maximum: the sweep at which an arm's rank
//!   stops growing is what separates a converged run from one parked at a critical point.

use qntt::solvers::mals::{MalsOptions, NoiseType, mals_with_history};
use qntt::{rayleigh_quotient, tt_state};
use qntt_exkit::args::Spec;

fn main() {
    Spec::new("step-05")
        .about(
            "DMRG noise sweep: % of E_corr recovered across a few M for noise-free / random / perturbative truncation",
        )
        .env(
            "QNTT_N2_DIR",
            "<path>",
            "integral directory (meta.txt + FCIDUMP); defaults to the cc-pVDZ N2 fixture",
        )
        .env(
            "QNTT_N2_MS",
            "<usize>[,…]",
            "comma-separated bond-dimension list [default 64,128]",
        )
        .env(
            "QNTT_NOISE_SCALE",
            "<float>",
            "scales the noise schedule [default 1.0]",
        )
        .parse();

    // ---- 1. the fixture and its determinant ---------------------------------------------------
    //
    // `QNTT_N2_DIR` names an integral directory (a bare name resolves under `testdata/`); unset,
    // the tracked N₂/cc-pVDZ fixture is used. `occ` fills the lowest `nup` spatial orbitals for
    // each spin: the Hartree–Fock determinant in the CANONICAL orbital order the FCIDUMP came in,
    // with no reordering. As a train it has rank 1 at every bond, and `e_hf_tt` is step-01's check
    // that the integrals, the space and the operator agree with what the fixture's meta.txt says.
    let repo = qntt_exkit::repo!();
    let fx = match std::env::var("QNTT_N2_DIR") {
        Ok(dir) => repo.load(&dir),
        Err(_) => repo.n2(),
    };
    let h = fx.ham();
    let (d, nelec, ms2) = (fx.d(), fx.nelec(), fx.ms2());
    let e_nuc = fx.e_core();
    let e_hf = fx.e_hf();
    let e_ref = fx.e_ref();
    let occ: Vec<bool> = (0..d).map(|k| k < fx.nup()).collect();
    let hf = tt_state::<f64>(&occ, &occ);
    let e_hf_tt = rayleigh_quotient(&h, &hf) / qntt::dot(&hf, &hf) + e_nuc;
    println!(
        "d={d} nelec={nelec} ms2={ms2}  E_hf(tt)={e_hf_tt:.8} (ref {e_hf:.8})  E_ref={e_ref:.8}"
    );

    // ---- 2. one arm: a bond-dimension ramp, a noise schedule, 20 MALS sweeps -------------------
    //
    // `recovered` turns an electronic eigenvalue into the percentage of the fixture's correlation
    // energy it captures: 0% is the determinant, 100% is the correlated reference in meta.txt.
    // `ramp` is the per-outer-iteration cap on the TOTAL bond dimension, doubling 8, 16, 32, …
    // until it clamps at `m`; `noise` is the amplitude in force at those same iterations, strongest
    // while the ranks are still growing and zero from the eighth on, so every run ends on sweeps
    // that take the noise-free path exactly.
    let recovered = |lambda: f64| 100.0 * (e_hf - (lambda + e_nuc)) / (e_hf - e_ref);
    // Returns (% E_corr, max total bond rank reached) — the kept-state count for each run.
    let run = |m: usize, nt: Option<NoiseType>| -> (f64, usize) {
        let ramp: Vec<usize> = (3..14).map(|k| m.min(1 << k)).collect();
        let scale = qntt_exkit::env_f64("QNTT_NOISE_SCALE", 1.0);
        let noise = match nt {
            None => vec![0.0],
            Some(_) => [
                1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]
            .iter()
            .map(|a| a * scale)
            .collect(),
        };
        let opts = MalsOptions {
            eps: 1e-10,
            bond_dims: ramp,
            noise,
            // `None` is the noise-free arm (all-zero schedule), so the flavour is moot; take the
            // crate default.
            noise_type: nt.unwrap_or_default(),
            seed: 3,
            max_iter: 20,
            ..MalsOptions::default()
        };
        let hist = mals_with_history(&h, &hf, &opts).0;
        let maxrank = hist
            .sweeps
            .iter()
            .map(|s| s.max_bond_dim)
            .max()
            .unwrap_or(0);
        (recovered(hist.lambda), maxrank)
    };

    // ---- 3. the ladder: three arms at each M --------------------------------------------------
    //
    // The three runs share the fixture, the start, the ramp, the schedule and the seed, and differ
    // only in how the two-site block is truncated, so the columns are directly comparable. The
    // bracketed number is the largest total bond rank any sweep of that arm reached.
    let ms: Vec<usize> = std::env::var("QNTT_N2_MS")
        .ok()
        .map(|s| s.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![64, 128]);
    println!("\ncanonical, HF determinant, 20 sweeps, % of E_corr (max total bond rank in []):");
    for m in ms {
        let (quiet, _) = run(m, None);
        let (rnd, rr) = run(m, Some(NoiseType::Random));
        let (pert, pr) = run(m, Some(NoiseType::Perturbative));
        println!(
            "  M={m:<4}  noise-free {quiet:6.2}%   random {rnd:6.2}% [{rr}]   perturbative {pert:6.2}% [{pr}]"
        );
    }
}
