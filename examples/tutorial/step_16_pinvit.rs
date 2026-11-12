// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! A second ground-state solver: preconditioned inverse iteration
//!
//! # Introduction
//!
//! Steps 01 and 02 reached the ground state by sweeping: ALS visits one site at a time and keeps
//! the ranks it was handed. PINVIT is the other kind of solver — a global iteration that applies
//! the shifted Hamiltonian `H_γ = H + γI` and a preconditioner `S` to the whole train and rounds
//! the result back down at every step. Nothing here fixes the rank: it is whatever the rounding
//! tolerance leaves. And the stopping rule is not a sweep-to-sweep energy difference but an
//! a-posteriori error bound, which the `conv` column below reports as reached or not. The
//! preconditioner sets the rate, not the answer: the fixed point `S H_γ S x = λ S² x` implies
//! `H_γ (S x) = λ (S x)`, so the converged eigenvalue is an eigenvalue of `H_γ` for any invertible
//! SPD `S`, and the Hamiltonian eigenvector is `y = S·x`.
//!
//! The program runs PINVIT with deterministic TT-SVD rounding and reports, per fixture, the
//! energy, whether the a-posteriori bound was reached, the iteration count, the peak total bond
//! rank over the iteration, and the wall time.
//!
//! **Fixture scope.** Nothing in the solver caps rank: every rounding call is tolerance-only. On
//! larger fixtures (`norb >= 13`, e.g. `c2h4_sto3g`) the iterates grow until memory runs out, so
//! those are not in the default list; pass them explicitly to try anyway.
//!
//! # Builds on
//!
//! step-02
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-16                     # lih_sto3g, then h2o_sto3g
//! cargo run --release --example step-16 -- lih_sto3g        # one fixture only
//! cargo run --release --example step-16 -- --help
//! ```
//!
//! - `fixture …` — variadic: the fixture directories under `testdata/` to run, in order
//!   [default `lih_sto3g,h2o_sto3g`]. A name that does not load is reported as a skipped row
//!   instead of aborting the table.
//! - `--format <text|tsv>` — output format [default `text`].
//! - `QNTT_PINVIT_ETA=<float>` — absolute rounding tolerances [default `1e-10`].
//!
//! # Results
//!
//! Two `#` lines per fixture carry the calibration, then the table row. On `lih_sto3g` the shift
//! is `gamma = 15.520`, the calibrated spectrum `spec(A) = [7.30e-1, 1.61e0]` gives
//! `c_raw 0.3770 -> c 0.4704`, and PINVIT converges to `-7.88240193` Ha in 72 iterations at a
//! max total bond rank of 16. On `h2o_sto3g` the shift is much larger (`gamma = 106.498`,
//! `min d_i = -19.318`), `c` rises to `0.7219`, and the run converges to `-75.01985480` Ha at
//! rank 29 in 46 iterations.
//!
//! Three columns are easy to misread. `conv` is the a-posteriori bound, not the energy: a `NO`
//! row merely ran out of the 400 allowed iterations and its energy carries no certificate.
//! `max_bond_dim` is the max over the iteration's per-iterate total bond ranks, so it records the
//! peak the run had to store, not the size of the answer. And `(settled: false)`, printed on both
//! fixtures here, says the calibration's power iteration did not tighten to its tolerance — the
//! bracket on `c` is loose, which the safety margin and the runtime `c_understated` check (an
//! assert in the loop) are there to absorb.
//!
//! The energies are total energies: the solver returns the shifted electronic eigenvalue, and the
//! program undoes both conventions with `lam - gamma + e0`.
//!
//! # Possibilities for extensions
//!
//! - Pass other small fixtures explicitly — `h4_chain_stretch`, `h2o_sto3g_stretch` — and watch
//!   what stronger correlation does to the rank the tolerance-only rounding settles on.
//! - Sweep `QNTT_PINVIT_ETA` (say `1e-8`, `1e-12`): it moves `max_bond_dim`, the wall time, and
//!   eventually whether `conv` still says `yes`.
//! - Print the gap to the reference energy in `meta.txt` (`fx.e_ref()`, as step-01 does): the
//!   table certifies the iteration, but says nothing about where the answer sits against FCI.
//! - Print `info.max_ranks` in full rather than its maximum, to see the rank grow and then settle
//!   over the iteration.

use std::time::Instant;

use qntt::solvers::pinvit::{
    CalibrateOpts, ExpSumPrecond, PinvitConsts, PinvitDetOpts, PinvitMethod, PinvitOpts,
    choose_gamma, diagonal_one_particle, gamma_for_positive_diagonal, pinvit_ground_state,
};
use qntt::vector::tt_state_in;
use qntt_exkit::args::{Def, Spec, Val};
use qntt_exkit::out::{Col, Table};

fn main() {
    let m = Spec::new("step-16")
        .about("PINVIT ground states with deterministic TT-SVD rounding")
        .rest(
            "fixture",
            "fixtures to run",
            Val::Str,
            // Small fixtures only — see the module header on `norb >= 13`.
            Def::List(&["lih_sto3g", "h2o_sto3g"]),
        )
        .env(
            "QNTT_PINVIT_ETA",
            "<float>",
            "absolute rounding tolerances [default 1e-10]",
        )
        .parse();
    let repo = qntt_exkit::repo!();
    let fixtures = m.strs("fixture");

    let mut t = Table::new(vec![
        Col::text("fixture", 14),
        Col::text("backend", 14),
        Col::float("energy", 16, 8).header("energy (Ha)"),
        Col::tag("conv", 5),
        Col::int("iters", 7),
        Col::int("max_bond_dim", 9),
        Col::float("wall", 10, 1).header("wall (ms)"),
    ])
    .no_rule()
    .start();
    println!("{}", "-".repeat(80));

    for name in fixtures.iter().map(String::as_str) {
        let fx = match repo.try_load(name) {
            Ok(f) => f,
            Err(e) => {
                t.skip(name, e);
                continue;
            }
        };
        let h = fx.ham();
        let space = fx.space();
        let nelec = space.nup + space.ndn;

        // The start is the aufbau determinant, a rank-1 train. Unlike ALS in step-02, PINVIT is
        // not confined to the ranks of its start: every iterate is rounded to a TOLERANCE, not to
        // a rank grid, so the rank is free to grow (and, at the end, to settle back down).
        let occ_up: Vec<bool> = (0..space.d).map(|i| i < space.nup).collect();
        let occ_dn: Vec<bool> = (0..space.d).map(|i| i < space.ndn).collect();
        let x0 = tt_state_in(space, &occ_up, &occ_dn);

        // Both constraints: H_gamma positive definite AND D positive on the sector.
        let gamma = choose_gamma(&h, &x0, 2.0).max(gamma_for_positive_diagonal(
            fx.ints(),
            space.nup,
            space.ndn,
            nelec,
            5.0,
        ));
        let d = diagonal_one_particle(fx.ints(), gamma, nelec);
        // `c0 = 0.6`, the exponential-sum accuracy: a smaller `c0` lengthens the sum without
        // improving the calibrated `c`.
        let mut precond = ExpSumPrecond::from_diagonal(&d, space.nup, space.ndn, 0.6);
        // Centre spec(S H_γ S) on 1 and read `c` off the operator. Without this the declared `c`
        // is a guess and the a-posteriori bound can be optimistic rather than conservative.
        let cal = precond.calibrate(&h, gamma, &x0, &CalibrateOpts::default());
        let precond = precond;
        let consts = PinvitConsts::calibrated(gamma, 50.0, &cal);
        // The exponential sum applies e^{-beta_m d_i} per site. A large NEGATIVE d_i makes that a
        // large positive exponent, and beta_max ~ 1/t_min, so the dynamic range is governed by
        // |min d| / t_min. Print it: it is the quantity that decides whether the per-site factors
        // stay representable.
        let dmin = d.iter().cloned().fold(f64::MAX, f64::min);
        let mut asc = d.clone();
        asc.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let t_min: f64 =
            asc[..space.nup].iter().sum::<f64>() + asc[..space.ndn].iter().sum::<f64>();
        let t_max: f64 = asc[asc.len() - space.nup..].iter().sum::<f64>()
            + asc[asc.len() - space.ndn..].iter().sum::<f64>();
        let beta_max = precond.betas.iter().cloned().fold(f64::MIN, f64::max);
        println!(
            "# {name}: gamma = {gamma:.3}  D = [{t_min:.3}, {t_max:.3}]  T = {:.2}  min d_i = {dmin:.3}",
            t_max / t_min
        );
        println!(
            "#   spec(A) = [{:.4e}, {:.4e}]  c_raw {:.4} -> c {:.4}  (settled: {})  \
             max exponent beta*|min d| = {:.1} (e^x overflows f64 near 709)",
            cal.lambda_min,
            cal.lambda_max,
            cal.c_raw,
            cal.c,
            cal.converged,
            beta_max * dmin.abs()
        );

        // QNTT_PINVIT_ETA overrides the absolute rounding tolerances of `PinvitDetOpts`.
        let eta: f64 = qntt_exkit::env_f64("QNTT_PINVIT_ETA", 1e-10);
        #[allow(unused_mut)]
        let mut methods = vec![(
            "deterministic",
            PinvitMethod::Det(PinvitDetOpts {
                eta_precond: eta,
                eta_resid: eta,
                eta_inner: eta * 100.0,
                ..PinvitDetOpts::default()
            }),
        )];
        for (label, method) in methods {
            let opts = PinvitOpts {
                consts,
                method,
                tau: 1e-6,
                max_iter: 400,
                ..PinvitOpts::default()
            };
            let t0 = Instant::now();
            let (lam, _y, info) = pinvit_ground_state(&h, &precond, &x0, &opts);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            // Undo both conventions at once: the solver returns the SHIFTED ELECTRONIC eigenvalue
            // `λ(H_γ)`, so the physical total energy is `λ − γ + e0`.
            let energy = lam - gamma + fx.ints().e0;
            let max_rank = info.max_ranks.iter().copied().max().unwrap_or(0);
            // `converged` is the only thing separating a certified energy from one that merely ran
            // out of iterations; printing the energy without it invites quoting the latter.
            let conv = if info.converged { "yes" } else { "NO" };
            assert!(
                !info.c_understated,
                "{name}/{label}: the Rayleigh witness proved c understated \
                 (min ⟨Ax,x⟩/⟨x,x⟩ = {:.4e} < 1 − c = {:.4e}) — the calibration is wrong",
                info.min_rayleigh_a,
                1.0 - cal.c
            );
            t.row(qntt_exkit::row![
                name, label, energy, conv, info.iters, max_rank, ms
            ]);
        }
    }
    t.finish();
}
