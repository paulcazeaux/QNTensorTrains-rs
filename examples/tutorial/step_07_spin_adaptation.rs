// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Spin adaptation: the same N2/cc-pVDZ molecule as step-06, solved in the SU(2) singlet sector.
//!
//! # Introduction
//!
//! The abelian layer of the first six steps conserves `Nup` and `Ndn` separately. The SU(2) layer
//! conserves total spin instead: a bond block carries a particle count and a spin label, and its
//! rank counts MULTIPLETS rather than states. One consequence governs every number below — an
//! SU(2) rank is a REDUCED multiplet rank and is never comparable to an abelian pooled bond
//! dimension `M`. What this program prints is a histogram of per-(bond, sector) multiplet ranks;
//! the pooled `M` of step-02 and step-06 has no counterpart here, and putting the two side by side
//! is a category error, not a comparison.
//!
//! This is the SU(2) twin of step-06 (`examples/tutorial/step_06_scaling_up.rs`): same molecule,
//! same real integrals (`testdata/n2_ccpvdz`).
//!
//! Energy check: the closed-shell RHF determinant is the singlet CSF (7 doubly-occupied orbitals),
//! so its SU(2) Rayleigh quotient must reproduce the abelian `RQ(H,HF) = -132.0844109291` — a
//! cross-symmetry check that the SU(2) Hamiltonian is built correctly. The run asserts it to 1e-6
//! and stops if it fails.
//!
//! # Builds on
//!
//! step-06
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-07                   # matvec at per-sector rank 24
//! cargo run --release --example step-07 -- mals 50 2      # rank-capped MALS, 2 outer sweeps
//! cargo run --release --example step-07 -- --help
//! ```
//!
//! The first positional argument selects the mode and defaults to `matvec`; the fixture is fixed
//! (`testdata/n2_ccpvdz`) and is not an argument.
//!
//! - `matvec` (default) — build a random state at a controlled per-sector rank, print its rank
//!   histogram, then time an environment sweep, `round`, and one ALS sweep on it.
//!   - `R` `<usize>` — per-sector rank [default 24]
//! - `mals` — run a rank-capped MALS from a random start.
//!   - `R` `<usize>` — bond-dim cap [default 50]
//!   - `sweeps` `<usize>` — outer iterations [default 2]
//! - `--format <text|tsv>` — output format [default `text`]
//! - `-h`, `--help` — print the usage text and exit
//!
//! The program declares no `QNTT_*` environment variable of its own. Two shared variables still
//! apply: `QNTT_FORMAT` (`text` | `tsv`) sets the ambient output format that `--format` then
//! overrides, and `QNTT_REPO_ROOT` overrides the checkout directory the fixture is read from. The
//! `KNOBS:` line after the banner lists the library's tuning settings in effect.
//!
//! # Results
//!
//! Every mode starts with the same prologue: load the integrals, build the SU(2) Hamiltonian, and
//! run the energy check, which prints `RayleighQuotient(H,HF) = -132.0844109291` with
//! `+e_nuc = -108.94937788` against the fixture's `e_hf = -108.94937788`, and passes.
//!
//! In `matvec` at the default `R = 24` the state has 708 nonzero (bond, sector) blocks with mean
//! multiplet rank 19.3 and maximum 24. The distribution is bimodal: 9.9% of the blocks are rank 1
//! (70 of them, the sectors the symmetry barely populates) and 75.7% fall in the 17-63 bucket,
//! with none at 64 or above. The timed lines follow; the ALS sweep line (its wall time and λ) is
//! the slowest and prints last — the recorded output below stops before it.
//!
//! # Possibilities for extensions
//!
//! - Raise the rank (`matvec 96`) and watch the histogram move up a bucket.
//! - Run `mals 50 2` and read the histogram of the converged state instead of the random one. Its
//!   ranks are multiplet ranks: resist reading them against the bond dimensions step-06 prints for
//!   the same molecule, which is exactly the confusion this step exists to prevent.
//! - Run step-06's own `matvec` mode and compare its printed `RayleighQuotient(H,HF)` with the
//!   number this step asserts — that one value is where the two layers are allowed to meet.

use std::time::Instant;

use qntt::RoundMode;
use qntt::su2::core::Family;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::als::{AlsOptions, als, right_to_left_sketching};
use qntt::su2::solvers::mals::{MalsOptions, mals};
use qntt::su2::vector::{Su2TtVector, tt_csf, tt_randn};
use qntt_exkit::out::{banner, flush};
use qntt_exkit::prelude::{Def, Spec, Val};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn randomized_state(space: Su2Space, r: usize, rng: &mut Xoshiro256PlusPlus) -> Su2TtVector<f64> {
    tt_randn(space, &qntt_exkit::ranks::su2::uniform(&space, r), rng)
}

/// Histogram of per-(bond,sector) multiplet bond ranks. Mirrors the abelian `block_histogram` of
/// step-06 (`examples/tutorial/step_06_scaling_up.rs`).
fn block_histogram(x: &Su2TtVector<f64>) {
    let buckets = [1usize, 2, 5, 17, 64, 128, usize::MAX];
    let labels = ["=1", "2-4", "5-16", "17-63", "64-127", "128+"];
    let mut counts = [0usize; 6];
    let (mut total, mut sum_rank, mut maxr) = (0usize, 0usize, 0usize);
    for b in 0..=x.space.d {
        for s in x.space.bond_sectors(b) {
            let r = x.rank(b, s);
            if r == 0 {
                continue;
            }
            total += 1;
            sum_rank += r;
            maxr = maxr.max(r);
            for i in 0..6 {
                if r >= buckets[i] && r < buckets[i + 1] {
                    counts[i] += 1;
                    break;
                }
            }
        }
    }
    println!(
        "  per-sector block ranks: {total} nonzero sectors, mean {:.1}, max {maxr}",
        sum_rank as f64 / total.max(1) as f64
    );
    for i in 0..6 {
        let pct = 100.0 * counts[i] as f64 / total.max(1) as f64;
        println!("    {:<7} {:>6}  ({:>4.1}%)", labels[i], counts[i], pct);
    }
    let (mut w_ge64, mut w_all) = (0.0f64, 0.0f64);
    for b in 0..=x.space.d {
        for s in x.space.bond_sectors(b) {
            let r = x.rank(b, s) as f64;
            w_all += r * r * r;
            if r >= 64.0 {
                w_ge64 += r * r * r;
            }
        }
    }
    println!(
        "    Σrank³ fraction in ≥64 blocks: {:.1}%",
        100.0 * w_ge64 / w_all.max(1.0)
    );
}

fn main() {
    let spec = Spec::new("step-07")
        .about("SU(2) spin-adapted N2/cc-pVDZ (d=28, N=14, singlet twoS=0) diagnostics harness")
        .mode(
            "matvec",
            "profile the env sweep, round, and one ALS sweep at a controlled per-sector rank",
            |m| m.pos("R", "per-sector rank", Val::Usize, Def::Usize(24)),
        )
        .mode("mals", "run a rank-capped MALS from a random start", |m| {
            m.pos("R", "bond-dim cap", Val::Usize, Def::Usize(50)).pos(
                "sweeps",
                "outer iterations",
                Val::Usize,
                Def::Usize(2),
            )
        });
    let m = spec.default_mode("matvec").parse();
    let mode = m.mode();

    banner(format!(
        "N2/cc-pVDZ (REAL integrals) SU(2) diagnostics — mode={mode}"
    ));
    flush(format!("KNOBS: {}", qntt::knobs::line()));
    let t0 = Instant::now();
    let fx = qntt_exkit::repo!().n2();
    let (d, nelec, e_nuc, e_hf) = (fx.d(), fx.nelec(), fx.e_core(), fx.e_hf());
    let (t, v) = fx.into_tv();
    flush(format!(
        "load integrals             {:>12.3} ms  (d={d}, nelec={nelec}, singlet twoS=0)",
        t0.elapsed().as_secs_f64() * 1e3
    ));

    // SU(2) space at fixed total spin S: twoS = 0 (singlet ground state of N2).
    let space = Su2Space::new(d, nelec, 0);
    let t0 = Instant::now();
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    flush(format!(
        "Su2SparseHamiltonian::new  {:>12.3} ms",
        t0.elapsed().as_secs_f64() * 1e3
    ));

    // Energy check: the closed-shell RHF determinant = singlet CSF with the lowest N/2 orbitals
    // doubly occupied. Its SU(2) Rayleigh quotient must match the abelian RQ(H,HF).
    let nocc = nelec / 2;
    let codes: Vec<Family> = (0..d)
        .map(|i| if i < nocc { Family::Occ } else { Family::Empty })
        .collect();
    let hf: Su2TtVector<f64> = tt_csf(&codes);
    let t0 = Instant::now();
    let rq = rayleigh_quotient(&h, &hf);
    let hf_norm2 = qntt::su2::ops::dot(&hf, &hf);
    let e_rq = rq / hf_norm2;
    flush(format!(
        "RayleighQuotient(H,HF)     {:>12.3} ms  = {:.10}  (+e_nuc={:.8}, e_hf={:.8})",
        t0.elapsed().as_secs_f64() * 1e3,
        e_rq,
        e_rq + e_nuc,
        e_hf
    ));
    assert!(
        (e_rq - (-132.0844109291)).abs() < 1e-6,
        "SU(2) RQ check failed: {e_rq} (expected abelian -132.0844109291)"
    );
    println!("  ✓ energy check passed (SU(2) HF matches abelian to 1e-6)\n");

    if mode == "mals" {
        let r0: usize = m.usize("R");
        let sweeps: usize = m.usize("sweeps");
        let bond_dims: Vec<usize> = vec![r0 / 4, r0 / 2, r0, r0];
        println!("MALS from random start: bond_dims={bond_dims:?}, max_iter={sweeps}");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4E32);
        let mut x0 = randomized_state(space, 5, &mut rng);
        x0.right_orthogonalize(true);
        let opts = MalsOptions {
            eps: 1e-6,
            bond_dims,
            max_iter: sweeps,
            verbose: true,
            ..Default::default()
        };
        let t0 = Instant::now();
        let (lambda, x) = mals(&h, &x0, &opts);
        flush(format!(
            "\nMALS total {:>10.3} ms  λ={:.8}  E={:.8}",
            t0.elapsed().as_secs_f64() * 1e3,
            lambda,
            lambda + e_nuc
        ));
        block_histogram(&x);
        return;
    }

    // --- matvec mode: controlled per-sector rank R ---
    let r: usize = m.usize("R");
    println!("controlled state: per-sector rank R={r}");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4E32);
    let t0 = Instant::now();
    let mut x = randomized_state(space, r, &mut rng);
    x.right_orthogonalize(true);
    flush(format!(
        "build+orthogonalize state  {:>12.3} ms",
        t0.elapsed().as_secs_f64() * 1e3
    ));
    block_histogram(&x);
    println!();

    // (a) Environment sweep, right to left.
    let t0 = Instant::now();
    let fr = right_to_left_sketching(&h, &x);
    std::hint::black_box(&fr);
    flush(format!(
        "env sweep right_to_left    {:>12.3} ms  (d-1 environment steps, threaded)",
        t0.elapsed().as_secs_f64() * 1e3
    ));

    // (b) round(x): one truncating SVD sweep.
    let t0 = Instant::now();
    let xr = x.round(1e-6, qntt::Pooled::NONE, RoundMode::Rel);
    std::hint::black_box(&xr);
    let nsec: usize = (0..=d).map(|b| space.bond_sectors(b).len()).sum();
    flush(format!(
        "round(x)                   {:>12.3} ms   [{} sectors]",
        t0.elapsed().as_secs_f64() * 1e3,
        nsec
    ));

    // (c) One full ALS sweep.
    let opts = AlsOptions {
        eps: 1e-8,
        max_iter: 1,
        verbose: false,
    };
    let t0 = Instant::now();
    let (lam, _) = als(&h, &x, &opts);
    flush(format!(
        "als 1 sweep                {:>12.3} ms  λ={:.6}",
        t0.elapsed().as_secs_f64() * 1e3,
        lam
    ));
}
