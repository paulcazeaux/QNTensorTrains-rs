// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Scaling up: N₂/cc-pVDZ on the real integrals, where every step has a cost.
//!
//! # Introduction
//!
//! The same objects as step-01, at `d = 28` orbitals and 14 electrons (`Sz = 0`) instead of 6
//! orbitals: the fixture is `testdata/n2_ccpvdz/` (`meta.txt` + `N2.FCIDUMP`, PySCF canonical
//! orbitals, chemists' `(ij|kl)`). Every stage is timed, because at this size none of them is free.
//! Before anything else the program checks `⟨HF|H|HF⟩ = -132.0844109290647` (electronic; adding
//! `e_nuc` gives `e_hf = -108.94937787901756`), which catches an index-order, chemists'/physicists'
//! or missing-factor mistake in one number.
//!
//! Two modes (first positional):
//!
//! - `matvec` (default) — a random train at uniform per-sector rank `R`, its histogram of block
//!   ranks, and the cost of the pieces of one sweep: building the right environments of `H`, one
//!   rounding sweep `round(x)`, and one full ALS sweep;
//! - `mals` — MALS from a slightly perturbed HF state with a bond-dimension ramp up to `R`, and the
//!   block-rank histogram of the converged state.
//!
//! # Builds on
//!
//! step-05
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-06                   # matvec at per-sector rank R = 24
//! cargo run --release --example step-06 -- matvec 30      # a larger profiled state
//! cargo run --release --example step-06 -- mals 50 2      # the MALS workflow
//! cargo run --release --example step-06 -- --help
//! ```
//!
//! The first positional selects the mode; `matvec` is the default, and each mode takes its own
//! positionals:
//!
//! - `matvec R` — `R`: per-sector rank [default 24].
//! - `mals R sweeps` — `R`: total bond-dim cap per outer iter [default 50]; `sweeps`: outer
//!   iterations [default 2].
//!
//! Options: `--format <text|tsv>` [default text] and `-h, --help`.
//!
//! Environment: `QNTT_CONSTRUCTION=bipartite|persector|median` selects the Hamiltonian
//! construction (step-04); unset, the library default is used. The choice is echoed as
//! `CONSTRUCTION=<arm>` under the header.
//!
//! # Results
//!
//! Both modes open with the same preamble: the integrals load, the Hamiltonian is built with its
//! construction printed, and `⟨HF|H|HF⟩` is checked against the reference value to 1e-6 — if that
//! check fails, the program aborts and nothing below it is worth reading.
//!
//! `matvec` then prints the per-(bond, sector) rank histogram of the profiled state — at the
//! default `R = 24`, 1240 nonzero sectors, mean rank 19.9, max 24, with 78.2 % of the sectors in
//! the 17–63 bucket, none at 64 or above, and so 0.0 % of the Σrank³ cost proxy in ≥64 blocks.
//! Then come the three timings: the environment sweep, one rounding sweep over those 1240 sectors,
//! and one full ALS sweep (λ = -132.385430), which contains an environment sweep and dominates the
//! other two.
//!
//! `mals` prints its sweep-by-sweep MALS log, the total wall time, λ and `E = λ + e_nuc`, and the
//! histogram of the CONVERGED state — where MALS actually put the rank, against the uniform grid
//! `matvec` profiles.
//!
//! # Possibilities for extensions
//!
//! - Sweep `R` in `matvec` (16, 24, 32, 48) and watch the histogram move: the rank at which
//!   sectors first land in the 64–127 bucket is the rank at which the Σrank³ fraction stops
//!   reading 0.0 %.
//! - Run `mals 100 4` and compare the converged histogram to the uniform-`R` one: MALS puts rank
//!   where the Schmidt spectrum needs it, which a uniform grid cannot.
//! - Set `QNTT_CONSTRUCTION=median` (or `persector`) and compare the `SparseHamiltonian::new`
//!   line across constructions.

use std::time::Instant;

use qntt::solvers::als::{AlsOptions, als, right_to_left_sketching};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::vector::perturbation;
use qntt::{QnSpace, RoundMode, SparseHamiltonian, TtVector, rayleigh_quotient, tt_state};
use qntt_exkit::args::{Def, Spec, Val};
use qntt_exkit::out::{banner, flush};
use qntt_exkit::ranks::abelian::uniform;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn randomized_state(space: QnSpace, r: usize, rng: &mut Xoshiro256PlusPlus) -> TtVector<f64> {
    qntt::tt_randn(space, &uniform(space, r), rng)
}

/// Histogram of per-(bond,sector) bond ranks — the block dimensions of the train.
fn block_histogram(x: &TtVector<f64>) {
    let buckets = [1usize, 2, 5, 17, 64, 128, usize::MAX];
    let labels = ["=1", "2-4", "5-16", "17-63", "64-127", "128+"];
    let mut counts = [0usize; 6];
    let mut total = 0usize;
    let mut sum_rank = 0usize;
    let mut maxr = 0usize;
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
    // Work-weighted: fraction of Σrank³ (a matvec-cost proxy) living in ≥64 blocks.
    let mut w_ge64 = 0.0f64;
    let mut w_all = 0.0f64;
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
    let spec = Spec::new("step-06")
        .about("Large-scale N2/cc-pVDZ (d=28, N=14, Sz=0) diagnostics on the REAL integrals")
        .mode(
            "matvec",
            "profile H·x, the round sweep and one ALS sweep",
            |md| md.pos("R", "per-sector rank", Val::Usize, Def::Usize(24)),
        )
        .mode("mals", "run the real MALS workflow", |md| {
            md.pos(
                "R",
                "total bond-dim cap per outer iter",
                Val::Usize,
                Def::Usize(50),
            )
            .pos("sweeps", "outer iterations", Val::Usize, Def::Usize(2))
        });
    let m = spec
        .default_mode("matvec")
        .env(
            "QNTT_CONSTRUCTION",
            "bipartite|persector|median",
            "Hamiltonian construction arm; the fallback is the crate DEFAULT_CONSTRUCTION",
        )
        .parse();
    let mode = m.mode();

    // ---- the preamble every mode shares: load, build, check --------------------------------
    //
    // Same four objects as step-01, but at d = 28 instead of 6: every line below is timed, because
    // at this scale each one is a cost the earlier steps could ignore.
    banner(format!(
        "N2/cc-pVDZ (REAL integrals) diagnostics — mode={mode}"
    ));
    let t0 = Instant::now();
    let fx = qntt_exkit::repo!().n2();
    let (d, nelec) = (fx.d(), fx.nelec());
    let (e_nuc, e_hf) = (fx.e_core(), fx.e_hf());
    let (t, v) = fx.into_tv();
    println!(
        "load integrals             {:>12.3} ms  (d={d}, nelec={nelec})",
        t0.elapsed().as_secs_f64() * 1e3
    );

    // The closed-shell HF determinant: the lowest `nelec/2` orbitals doubly occupied, as a rank-1
    // train. `check()` verifies the block structure of the cores against the space.
    let space = QnSpace::from_n_two_sz(d, nelec, 0);
    let nocc = nelec / 2;
    let occ: Vec<bool> = (0..d).map(|i| i < nocc).collect();
    let hf: TtVector<f64> = tt_state(&occ, &occ);
    hf.check();

    // QNTT_CONSTRUCTION selects the construction; unset, the library default
    // `DEFAULT_CONSTRUCTION` is used.
    let construction = match std::env::var("QNTT_CONSTRUCTION").as_deref() {
        Ok("bipartite") | Ok("Bipartite") => qntt::hamiltonian::Construction::Bipartite,
        Ok("persector") | Ok("PerSector") => qntt::hamiltonian::Construction::BipartitePerSector,
        Ok("median") | Ok("Median") => qntt::hamiltonian::Construction::Median,
        _ => qntt::hamiltonian::assembly::DEFAULT_CONSTRUCTION,
    };
    let t0 = Instant::now();
    let h = SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, construction);
    println!("CONSTRUCTION={construction:?}");
    println!(
        "SparseHamiltonian::new     {:>12.3} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );

    let t0 = Instant::now();
    let rq = rayleigh_quotient(&h, &hf);
    println!(
        "RayleighQuotient(H,HF)     {:>12.3} ms  = {rq:.10}  (+e_nuc={:.8}, e_hf={:.8})",
        t0.elapsed().as_secs_f64() * 1e3,
        rq + e_nuc,
        e_hf
    );
    // The convention check: on the real integrals, ⟨HF|H|HF⟩ is a single number that catches an
    // index order, a chemist/physicist mix-up or a missing factor at once. Nothing below is
    // meaningful if it fails, so it aborts rather than warns.
    assert!(
        (rq - (-132.0844109290647)).abs() < 1e-6,
        "RQ check failed: {rq}"
    );
    println!("  ✓ energy check passed (matches the reference energy to 1e-6)\n");

    if mode == "mals" {
        let r0: usize = m.usize("R");
        let sweeps: usize = m.usize("sweeps");
        // schedule ramps toward r0 (total bond dim cap per outer iter).
        let bond_dims: Vec<usize> = vec![r0 / 4, r0 / 2, r0, r0];
        println!("MALS from perturbed HF: bond_dims={bond_dims:?}, max_iter={sweeps}");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4E32);
        let x0 = perturbation(&hf, &uniform(space, 5), 0.01, &mut rng);
        let opts = MalsOptions {
            eps: 1e-6,
            bond_dims,
            max_iter: sweeps,
            verbose: true,
            ..Default::default()
        };
        let t0 = Instant::now();
        let (lambda, x) = mals(&h, &x0, &opts);
        println!(
            "\nMALS total {:>10.3} ms  λ={:.8}  E={:.8}",
            t0.elapsed().as_secs_f64() * 1e3,
            lambda,
            lambda + e_nuc
        );
        block_histogram(&x);
        return;
    }

    // --- matvec mode: controlled per-sector rank R ---
    //
    // A random state at a UNIFORM per-sector rank, so the block dimensions are known rather than
    // whatever a solver happened to produce. That is what makes the histogram below readable.
    let r: usize = m.usize("R");
    println!("controlled state: per-sector rank R={r}");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4E32);
    let t0 = Instant::now();
    let mut x = randomized_state(space, r, &mut rng);
    x.right_orthogonalize(true);
    println!(
        "build+orthogonalize state  {:>12.3} ms",
        t0.elapsed().as_secs_f64() * 1e3
    );
    block_histogram(&x);
    println!();

    // The solvers never form the full product H·x, which is very large at this scale; they work
    // with the environments of H around each site. The three timings below are the pieces of a
    // sweep.

    // (a) Environment sweep: the right environments of H at every site.
    let t0 = Instant::now();
    let fr = right_to_left_sketching(&h, &x);
    std::hint::black_box(&fr);
    flush(format!(
        "env sweep right_to_left    {:>12.3} ms  (d-1 environment steps, threaded)",
        t0.elapsed().as_secs_f64() * 1e3
    ));

    // (b) One rounding sweep: `round(x)`, an SVD per sector at each bond.
    let t0 = Instant::now();
    let xr = x.round(1e-6, qntt::Pooled::NONE, RoundMode::Rel);
    std::hint::black_box(&xr);
    let nsec: usize = (0..=d).map(|b| space.bond_sectors(b).len()).sum();
    flush(format!(
        "round(x) SERIAL            {:>12.3} ms   [{} sectors]",
        t0.elapsed().as_secs_f64() * 1e3,
        nsec
    ));

    // (c) One full ALS sweep — the whole solver iteration (Davidson local solves + env updates).
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
