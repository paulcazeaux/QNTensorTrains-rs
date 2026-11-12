// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! A first ground state: load a molecule, build its Hamiltonian, run one-site ALS, compare to FCI.
//!
//! # Introduction
//!
//! Everything in this library rests on four objects, and this step introduces them in the order
//! they depend on each other:
//!
//! 1. a **space** ([`QnSpace`]): `d` spatial orbitals holding `Nup` spin-up and `Ndn` spin-down
//!    electrons. Every state lives in the sector of fixed `(Nup, Ndn)`, and every tensor-train
//!    bond is split into blocks labelled by how many electrons of each spin sit to its left;
//! 2. a **Hamiltonian** ([`SparseHamiltonian`]): the second-quantized operator built from the
//!    one- and two-electron integrals of a fixture directory under `testdata/`;
//! 3. a **state** ([`TtVector`]): a tensor train (matrix product state) whose per-bond,
//!    per-sector ranks are what "rank" means throughout the course;
//! 4. a **solver**: here the simplest one, one-site alternating least squares ([`als`]), which
//!    sweeps back and forth solving a small eigenproblem at each site and keeps the ranks it was
//!    given.
//!
//! The molecule is LiH in a minimal basis, six orbitals and four electrons, small enough that a
//! train at full rank *is* the exact wavefunction. ALS at that rank is then an exact
//! diagonalization performed in the tensor-train format, and its energy must agree with the
//! full-CI energy PySCF recorded in the fixture's `meta.txt` to solver tolerance. The program
//! asserts that agreement at the end.
//!
//! Two bookkeeping conventions appear here and hold everywhere:
//!
//! - **Rayleigh quotients are electronic.** `⟨x|H|x⟩` excludes the nuclear-repulsion (or, on a
//!   CAS fixture, the frozen-core) energy `e_core`; add it once, at the end, to compare to a total
//!   energy such as `e_hf` or `e_fci`.
//! - **Everything is 0-based**: sites `0..d`, bonds `0..=d`, and a sector counts the electrons to
//!   the left of the bond.
//!
//! # Builds on
//!
//! Nothing — this is the entry point.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-01                      # LiH/STO-3G at full rank
//! cargo run --release --example step-01 -- h2o_sto3g 8       # another fixture, per-sector rank cap 8
//! cargo run --release --example step-01 -- --help
//! ```
//!
//! Any directory under `testdata/` with a `meta.txt` carrying a reference energy works; the
//! assertion on the FCI gap is applied only when the rank cap does not bind (full rank).
//!
//! # Results
//!
//! On LiH the Hartree–Fock check reproduces PySCF's `e_hf` to ~1e-14, the normalized random start
//! sits a few hartree above it, and ALS at full rank lands on `e_fci` to ~1e-14. The residual
//! `‖Hx − λx‖` stays near 1e-7: its floor is the inner Davidson
//! tolerance, not `eps`, which is why the energy (second-order accurate) converges so much further
//! than the state. At a binding per-sector rank cap (`step-01 lih_sto3g 2`) the
//! energy stays variationally above FCI; how far above, and how fast the gap closes with rank, is
//! the subject of step-02.
//!
//! # Possibilities for extensions
//!
//! - Run with `h2o_sto3g` or `h4_chain_stretch`: the second is a stretched hydrogen chain whose
//!   correlation is strong, so watch how far the HF energy sits from FCI.
//! - Print the per-bond, per-sector ranks of the converged train (`x.rank(b, s)` over
//!   `space.bond_sectors(b)`) and compare them to the cap `space.max_rank(b, s)` — this is the
//!   grid step-02 sweeps.
//! - Replace the random start by the Hartree–Fock determinant (`fx.hf()`): ALS cannot grow a
//!   rank-1 train, so it will return `e_hf` exactly. That is the determinant trap step-05 escapes.

use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::riemannian::normalize_mut;
use qntt::{QnSpace, SparseHamiltonian, TtVector, rayleigh_quotient, tt_randn};
use qntt_exkit::prelude::*;
use qntt_exkit::ranks::abelian::uniform;
use qntt_exkit::stats::time;

/// Does the uniform-`r` grid already reach every sector's achievable rank? If so, the train can
/// represent any vector of the sector and ALS is an exact diagonalization.
fn cap_binds(space: QnSpace, r: usize) -> bool {
    (0..=space.d).any(|b| {
        space
            .bond_sectors(b)
            .into_iter()
            .any(|s| (space.max_rank(b, s) as usize) > r && b != 0 && b != space.d)
    })
}

fn main() {
    let m = Spec::new("step-01")
        .about("a first ground state: fixture → space → Hamiltonian → ALS → compare to FCI")
        .pos(
            "fixture",
            "a directory under testdata/ (or a path) with integrals and a meta.txt",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
        .pos(
            "R",
            "per-sector rank cap of the random start; the default exceeds every achievable rank on LiH",
            Val::Usize,
            Def::Usize(64),
        )
        .key(
            "eps",
            "ALS stops when the Rayleigh quotient moves by less than this between sweeps",
            Val::F64,
            Def::F64(1e-10),
        )
        .key("sweeps", "maximum number of ALS sweeps", Val::Usize, Def::Usize(40))
        .key("seed", "seed of the random start", Val::Usize, Def::Usize(7))
        .key(
            "tol",
            "at full rank, |E − E_fci| must be below this or the step fails",
            Val::F64,
            Def::F64(1e-6),
        )
        .parse();
    let fixture = m.str("fixture").to_string();
    let r = m.usize("R");
    let eps = m.f64("eps");
    let sweeps = m.usize("sweeps");
    let seed = m.usize("seed") as u64;
    let tol = m.f64("tol");

    // ---- 1. the fixture and its space ------------------------------------------------------------
    //
    // `repo!()` locates the checkout this binary was built from, so `testdata/` resolves from any
    // working directory. A fixture directory holds the integrals plus `meta.txt`, which records the
    // orbital count, the electron count, and PySCF's reference energies.
    let fx = qntt_exkit::repo!().load(&fixture);
    let space: QnSpace = fx.space();
    banner_kv(
        "step-01",
        &[
            ("fixture", fixture.clone()),
            ("d", fx.d().to_string()),
            ("nelec", fx.nelec().to_string()),
            ("Nup", space.nup.to_string()),
            ("Ndn", space.ndn.to_string()),
        ],
    );

    // ---- 2. the Hamiltonian -----------------------------------------------------------------------
    //
    // `ham()` assembles the operator at the assembly tolerance `f64::EPSILON`: every integral is
    // kept. Later steps trade that for speed; here the point is exactness.
    let (h, t_build): (SparseHamiltonian<f64>, _) = time(|| fx.ham());
    flush(format!(
        "Hamiltonian built in {:>9.3} ms",
        t_build.as_secs_f64() * 1e3
    ));

    // ---- 3. a first state: the Hartree–Fock determinant --------------------------------------------
    //
    // A determinant is a rank-1 train. Its Rayleigh quotient plus the core energy must reproduce the
    // Hartree–Fock energy PySCF wrote to meta.txt — the first thing to check on any new fixture,
    // because it validates the integrals, the space and the operator at once.
    let e_core = fx.e_core();
    let hf: TtVector<f64> = fx.hf();
    let e_hf_here = rayleigh_quotient(&h, &hf) + e_core;
    flush(format!(
        "HF determinant     E = {e_hf_here:.10}   (meta e_hf = {:.10}, gap {:+.1e})",
        fx.e_hf(),
        e_hf_here - fx.e_hf()
    ));

    // ---- 4. a random start at rank R ---------------------------------------------------------------
    //
    // ALS keeps the ranks it is given, so the start decides the manifold. `uniform(space, r)` puts
    // rank `min(r, achievable)` in every interior sector and 1 at the boundaries; at r = 64 on LiH
    // no cap binds and the manifold is the whole sector.
    let full = !cap_binds(space, r);
    let grid = uniform(space, r);
    // `tt_randn` draws unit-variance entries, not a unit-norm state, and `rayleigh_quotient` is
    // `⟨x|H|x⟩` with no division by `⟨x|x⟩` — every caller in this library passes a normalized
    // state. Normalize once, here.
    let mut x0: TtVector<f64> = tt_randn(space, &grid, &mut qntt_exkit::seeded(seed));
    normalize_mut(&mut x0);
    let e0 = rayleigh_quotient(&h, &x0) + e_core;
    flush(format!(
        "random start       E = {e0:.10}   per-sector rank cap {r} ({})",
        if full {
            "full rank: the train is exact"
        } else {
            "cap binds: variational above FCI"
        }
    ));

    // ---- 5. one-site ALS ------------------------------------------------------------------------------
    //
    // Each sweep visits every site, solves the local eigenproblem with Davidson at an inner tolerance
    // that tightens as the energy settles, and stops when the Rayleigh quotient moves by less than
    // `eps` between sweeps.
    let opts = AlsOptions {
        eps,
        max_iter: sweeps,
        ..AlsOptions::default()
    };
    let ((lambda, x), t_als) = time(|| als(&h, &x0, &opts));
    let e = lambda + e_core;

    // A check that owes nothing to PySCF: the residual of the eigen-equation in the TT format. Its
    // floor is set by the inner Davidson tolerance, not by `eps`, so expect ~1e-7 even when the
    // energy is converged to 1e-15.
    let residual = (&h.apply(&x) - &(&x * lambda)).norm();

    // ---- 6. compare to FCI ---------------------------------------------------------------------------
    let e_fci = fx.e_ref();
    let de = e - e_fci;
    flush(format!(
        "ALS ({sweeps} sweeps max) E = {e:.10}   in {:>9.3} ms",
        t_als.as_secs_f64() * 1e3
    ));
    flush(format!(
        "FCI (meta)         E = {e_fci:.10}   ΔE = {de:+.3e}   ‖Hx − λx‖ = {residual:.2e}"
    ));

    Summary::new("step-01")
        .s("fixture", &fixture)
        .i("R", r as i64)
        .b("full_rank", full)
        .f("E", e, 10)
        .f("E_fci", e_fci, 10)
        .e("dE", de, 3)
        .e("residual", residual, 2)
        .f("als_ms", t_als.as_secs_f64() * 1e3, 3)
        .emit();

    if full {
        assert!(
            de.abs() < tol,
            "at full rank ALS is an exact diagonalization, but |E − E_fci| = {:.3e} ≥ tol = {tol:.1e}",
            de.abs()
        );
    }
}
