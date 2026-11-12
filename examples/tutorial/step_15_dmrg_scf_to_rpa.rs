// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The whole stack end to end: a DMRG-SCF ground state, then RPA response on the joint manifold.
//!
//! # Introduction
//!
//! A full DMRG-SCF → RPA run on the joint (TT ⊕ orbital) manifold, end to end, with every
//! available cross-check printed beside it.
//!
//! ## Pipeline
//!
//! ```text
//! fixture (meta.txt + FCIDUMP)
//!   → prepared x0 at pooled rank M (HF ⊕ noise → ALS → Riemannian polish)
//!   → joint_ground_state           the alternating TT ⊕ orbital-rotation optimizer
//!   → joint_context                the criticality-gated response context at the optimum
//!   ├─ joint_symplectic_eigenvalues   dense ω (the reference)
//!   ├─ joint_lowest_davidson          matrix-free lowest-k ω
//!   ├─ transition_moments             T_n = ⟨Ψ_n|μ̂|Ψ_0⟩ + oscillator_strengths
//!   └─ polarizability_static          α(0), cross-checked against the moments' own SOS sum
//! ```
//!
//! Reported alongside: the orbital redundancy spectrum `N = S − τᵀτ` (whose nonzero eigenvalues
//! ARE the extra response directions the orbital leg contributes), an energy-vs-rank scan against
//! the TT-only pipeline at the same ranks and the same seed, the exact FCI gaps by configuration
//! enumeration, and the full convergence/refusal diagnostics of every stage.
//!
//! ## Three rules this example demonstrates
//!
//! 1. **Check `converged` before publishing any ω.** [`qntt::solvers::joint::JointScfInfo::converged`] is checked
//!    before a single response quantity is computed, and [`qntt::solvers::hess::SymplecticLowestInfo::converged`] before
//!    a single Davidson ω is printed. Self-consistency is not validity: an unconverged endpoint of
//!    this optimizer can give a badly wrong ω₁ behind a perfect-looking self-consistency
//!    margin.
//! 2. **The basis-identity contract.** [`qntt::solvers::joint::JointScfOut`] returns `u`, and `x`/`h`/`lambda` and every
//!    downstream quantity live in the ROTATED one-particle basis. The excitation ENERGIES are
//!    immune (an orbital rotation conserves `Nup`/`Ndn` separately, so `H(U)` is unitarily
//!    equivalent within the sector and the FCI gaps computed once in the native basis stay
//!    correct), but a PROPERTY operator is not: this example builds its dipole-like operators in
//!    the input basis and rotates them by the same congruence before handing them to
//!    [`transition_moments`]. Skipping that rotation is an `O(1)` error.
//! 3. **`precond = false` under Davidson.** The recommended setting at the joint layer; the
//!    example prints the iteration count, applies and `converged`.
//!
//! # Builds on
//!
//! step-13
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-15                        # lih_sto3g, defaults
//! cargo run --release --example step-15 -- h2o_sto3g m=3 k=4
//! cargo run --release --example step-15 -- lih_sto3g ranks=2,3,4 prop=/path/dipole_z.txt
//! cargo run --release --example step-15 -- h2o_sto3g m=4 modeopt   # + the mode-optimized start
//! ```
//!
//! - `fixture` a name under `testdata/`, or a path (default `lih_sto3g`)
//! - `m=` per-sector rank cap of the response reference (uniform across sectors, default 3)
//! - `k=` Davidson ω wanted (default 4)
//! - `targets=` exact levels tabulated (default 6)
//! - `jiter=` [`JointScfOpts::orb_max_iter`] (default 30)
//! - `ranks=` the energy-vs-rank scan, `m` appended if missing (default `2,3`)
//! - `prop=` comma-separated property files (`qntt::integrals::load_property` format, at most
//!   three components — omitted ⇒ the synthetic dipole-like triple below; default none)
//! - `nofci` skip the exact reference
//! - `modeopt` also run the mode-optimizing pre-stage (`mode_optimize`: local orbital rotations
//!   inside the two-site sweep) at `m` and hand its `(x₀, u₀)` to the joint solver; prints the
//!   second start beside the first and the total rotation `u₀·u₁`
//! - `--format text|tsv` output format (default `text`)
//!
//! # Results
//!
//! The exact reference enumerates the 225-configuration sector to `E0 = -8.877719570`
//! (electronic; total `-7.882401932`). Each rank-scan reference is prepared with the certified
//! polish of step 08 ([`qntt::solvers::riemannian::riemannian_cg_certified`] under the
//! ALS-rounded retraction) and reported as a certified minimizer of the TT-only problem before the
//! joint optimizer starts from it: `joint_ground_state` leaves basin selection to its caller. The
//! joint optimizer converges at both `M = 2` and `M = 3`; the orbital leg closes 4.05e-5 Ha over
//! the TT-only column at `M = 2` and 1.32e-6 at `M = 3`. The redundancy spectrum keeps all 7 of
//! its nonzero directions out of 15 pairs (nothing dropped), so the response space is
//! `n_tt 75 + n_orb_kept 7 = 82` modes. The dense spectrum is `Stable`, and Davidson at `k = 4`
//! agrees with it to about 4e-14. Mode `j = 1` matches its FCI gap to 3.3e-11 while `j = 2, 3`
//! sit 2.0e-2 away: that is the index misalignment the ⚠ line describes, read against the printed
//! exact gaps (0.11598 0.13319 0.16595 0.16595 0.18545 0.18545). `Σ f = 12.3971` over all 82
//! modes against `N_e = 4` reflects the synthetic property, not a TRK violation. α(0) from the
//! independent solve and α(0) reconstructed from the moments agree to 5.4e-9.
//!
//! ★ **Check `λ − E0_exact` before reading a single ω.** Where the joint optimum lands on the exact
//! sector ground state, the normal residual — and with it the Weingarten term `B` — vanishes, and
//! the RPA readout reduces to CI, which is why an ω can then match its FCI gap to roundoff. That
//! is a property of the fixture at that rank, not of the method: `‖z‖` at a saturated reference
//! sits orders of magnitude below a genuinely truncated one. Use a truncated fixture to exercise
//! `B`. The default run is in this regime: `λ − E0_exact ≈ 4e-13` Ha at `M = 3`, with
//! `‖z‖ ≈ 9e-7`.
//!
//! ★ A per-`j` FCI column on an untyped exact list can be misaligned by index without any solver
//! error: the exact list's levels are not labelled by irrep, so a joint mode can sit far from
//! slot `j` and close to another level entirely. step-11 resolves the channel assignment.
//!
//! ★ `Σ f` against `N_e` is a fixed-rank TRK diagnostic, not a reference value — and with
//! synthetic property components it is not even that. The moments' own SOS reconstruction against
//! [`polarizability_static`] is the cross-check worth reading.
//!
//! # Possibilities for extensions
//!
//! - Run `h2o_sto3g` (`d = 7`, ten electrons): every stage grows, and `nofci` turns off the
//!   configuration enumeration when the exact reference stops being the cheap part.
//! - Widen the scan (`ranks=2,3,4`) and watch the `closed [Ha]` column shrink as the TT-only
//!   reference approaches the same optimum the orbital leg reaches.
//! - Compare the printed `‖z‖` at `m=2` and `m=3`: that is the saturation the first ★ is about,
//!   and it decides whether the run exercises `B` at all.
//! - Supply real one-body components with `prop=` (`load_property` format): the oscillator
//!   strengths and α(0) then mean something, and `Σ f` becomes a diagnostic worth quoting.

use qntt::coeff::CoeffTensor;
use qntt::core::tangent::TtTangent;
use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::{PropertyOperator, load_property};
use qntt::rotation::{compose_rotations, orthogonality_defect};
use qntt::round::RoundMode;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::hess::{SympPrecond, SymplecticDavidsonOpts, SymplecticSpectrum};
use qntt::solvers::joint::{
    JointContext, JointOpts, JointScfOpts, JointVector, ModeOptOpts, StaticResponseOpts,
    TransitionMomentOpts, joint_context, joint_ground_state, joint_lowest_davidson, mode_optimize,
    oscillator_strengths, polarizability_static, transition_moments,
};
use qntt::solvers::mals::{MalsOptions, NoiseType};
use qntt::solvers::moderot::ModeRotSweepOpts;
use qntt::solvers::riemannian::{
    CertifyOpts, CertifyVerdict, Retraction, RiemannianOpts, normalize_mut, riemannian_cg_certified,
};
use qntt::{
    Integrals, QnSpace, RankGrid, SparseHamiltonian, TtVector, dot, rotate_integrals, tt_randn,
    tt_state,
};
use qntt_exkit::args::{Def, Spec, Val};
use qntt_exkit::out::banner;
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

const HARTREE_EV: f64 = 27.211386245988;

// --- exact reference by configuration enumeration ------------------------------------------------
// Carried locally: the library exposes no dense-FCI helper.

fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur: Vec<usize> = (0..n).collect();
    if n == 0 {
        return vec![vec![]];
    }
    if n > d {
        return out;
    }
    loop {
        out.push(cur.clone());
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if cur[i] != i + d - n {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        cur[i] += 1;
        for j in i + 1..n {
            cur[j] = cur[j - 1] + 1;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn sym_eigvals_jacobi(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _sweep in 0..200 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-15 {
                    continue;
                }
                let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut v: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v
}

fn exact_config_spectrum(space: QnSpace, ham: &SparseHamiltonian<f64>) -> Vec<f64> {
    let d = space.d;
    let states: Vec<TtVector<f64>> = combinations(d, space.nup)
        .into_iter()
        .flat_map(|up| {
            combinations(d, space.ndn)
                .into_iter()
                .map(move |dn| (up.clone(), dn))
        })
        .map(|(up, dn)| {
            let mut occ_up = vec![false; d];
            let mut occ_dn = vec![false; d];
            for &s in &up {
                occ_up[s] = true;
            }
            for &s in &dn {
                occ_dn[s] = true;
            }
            tt_state(&occ_up, &occ_dn)
        })
        .collect();
    let n = states.len();
    let mut hmat = vec![vec![0.0f64; n]; n];
    for (j, sj) in states.iter().enumerate() {
        let hs = ham * sj;
        for (i, si) in states.iter().enumerate() {
            hmat[i][j] = dot(si, &hs);
        }
    }
    sym_eigvals_jacobi(hmat)
}

// --- the prepared reference ----------------------------------------------------------------------
// `joint_ground_state`'s own doc: "`x0` selects the fixed-rank manifold and the basin — basin
// escape is the CALLER's job". This is that preparation, HF ⊕ noise → ALS → Riemannian polish,
// so the joint optimizer starts from a converged TT point rather than from noise.

fn ground_state(
    space: QnSpace,
    ham: &SparseHamiltonian<f64>,
    m: usize,
    seed: u64,
) -> TtVector<f64> {
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let occ: Vec<bool> = (0..d).map(|i| i < space.nup).collect();
    let hf: TtVector<f64> = tt_state(&occ, &occ);
    let grid = qntt_exkit::ranks::abelian::uniform(space, m);
    let mut x0 = qntt::vector::perturbation(&hf, &grid, 0.2, &mut rng);
    x0.round_ranks_mut(&grid);
    normalize_mut(&mut x0);
    let (_lals, xa) = als(
        ham,
        &x0,
        &AlsOptions {
            eps: 1e-10,
            max_iter: 60,
            ..AlsOptions::default()
        },
    );
    // The certified polish (step 08 explains it): CG under the ALS-rounded retraction, and an
    // `A + B` probe where the descent stops that certifies the point or kicks along a proven
    // negative direction and resumes. The joint optimizer below must start from a MINIMIZER of
    // the TT-only problem, not merely a critical point — this is where that is established.
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
    let polish = |y0: &TtVector<f64>| -> (TtVector<f64>, String) {
        let rep = riemannian_cg_certified(ham, y0, &opts, &copts, None);
        let verdict = match rep.verdict {
            CertifyVerdict::Minimizer { .. } => "certified minimizer",
            CertifyVerdict::Saddle { .. } => "SADDLE",
            CertifyVerdict::Inconclusive { .. } => "probe inconclusive",
            CertifyVerdict::NotProbed => "not probed",
        };
        let line = format!(
            "{} it → ‖grad‖ {:.1e}, {verdict}{}",
            rep.iterations,
            rep.gnorm,
            if rep.kicks.is_empty() {
                String::new()
            } else {
                format!(" after {} kick(s)", rep.kicks.len())
            }
        );
        (rep.x, line)
    };
    let (mut x, first) = polish(&xa);
    normalize_mut(&mut x);
    x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    let (mut x, second) = polish(&x);
    normalize_mut(&mut x);
    x.check();
    println!("  x0 at M={m}: certified polish {first}; after the trim {second}");
    x
}

// --- properties ---------------------------------------------------------------------------------

/// A synthetic "dipole-like" triple in the INPUT orbital basis: one diagonal component (a crude
/// charge-separation model) plus two random symmetric ones with distinct seeds.
///
/// Synthetic on purpose — the shipped fixtures carry integrals, not dipole matrices, and a
/// component invented here must not be mistaken for the molecule's. Consequences the printout
/// repeats: the oscillator strengths below are dimensionally `f`-shaped but physically meaningless,
/// and the TRK sum `Σf = N_e` is neither expected nor an oracle. Supply `prop=` for real
/// components.
fn synthetic_dipole(d: usize) -> Vec<PropertyOperator<f64>> {
    let mut out = Vec::new();
    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        m.set(&[i, i], i as f64 - 0.5 * (d as f64 - 1.0));
    }
    out.push(PropertyOperator::new("synth_diag", m));
    for (c, seed) in [(1usize, 0x9002u64), (2, 0x9003)] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut m = CoeffTensor::<f64>::zeros(d, 2);
        for i in 0..d {
            for j in i..d {
                let val = rng.random_range(-1.0f64..1.0);
                m.set(&[i, j], val);
                m.set(&[j, i], val);
            }
        }
        out.push(PropertyOperator::new(format!("synth_rand{c}"), m));
    }
    out
}

/// The property in the ROTATED basis: the same congruence `m′[a,b] = Σ_ij U[i,a]·U[j,b]·m[i,j]`
/// that [`rotate_integrals`] applies to `t`.
///
/// Realized by wrapping `m` as an `Integrals`' one-body block with a zero two-body block and
/// calling [`rotate_integrals`], rather than hand-rolling the double contraction, so the index
/// convention is the library's own.
fn rotate_property(p: &PropertyOperator<f64>, u: &[f64], d: usize) -> PropertyOperator<f64> {
    let wrapped = Integrals::<f64> {
        t: p.m.clone(),
        v: CoeffTensor::<f64>::zeros(d, 4),
        e0: 0.0,
        norb: d,
        nelec: 0,
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    };
    PropertyOperator::new(p.label.clone(), rotate_integrals(&wrapped, u).t)
}

// --- Davidson seeds -------------------------------------------------------------------------------

/// `b` independent joint seeds: a random tangent (per-sector rank window `1..=2`) projected at the
/// context's own frames, plus `n_eff` i.i.d. standard-normal orbital coordinates. The kernel
/// W-orthonormalizes every accepted vector, so the relative scale of the two legs is irrelevant.
fn joint_seed_block(
    jctx: &JointContext<'_>,
    b: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<JointVector> {
    let (base_l, base_r) = jctx.bases();
    let space = base_l.space;
    let spec = jctx.spec();
    let n_eff = jctx.n_eff();
    (0..b)
        .map(|_| {
            let ranks: Vec<RankGrid> = (0..=space.d)
                .map(|bo| {
                    let mut r = RankGrid::zeros(&space);
                    for s in space.bond_sectors(bo) {
                        r[s] = if bo == 0 || bo == space.d {
                            1
                        } else {
                            rng.random_range(1..=2)
                        };
                    }
                    r
                })
                .collect();
            let dir: TtVector<f64> = tt_randn(space, &ranks, rng);
            let mut v = JointVector::from_tangent(spec, TtTangent::project(base_l, base_r, &dir));
            v.c = (0..n_eff)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect();
            v
        })
        .collect()
}

// --- driver ----------------------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn main() {
    let cli = Spec::new("step-15")
        .about("a full DMRG-SCF → RPA run on the joint (TT ⊕ orbital) manifold, end to end")
        .pos(
            "fixture",
            "a name under testdata/, or a path",
            Val::Path,
            Def::Str("lih_sto3g"),
        )
        .key(
            "m",
            "per-sector rank cap of the response reference (uniform across sectors)",
            Val::Usize,
            Def::Usize(3),
        )
        .key("k", "Davidson ω wanted", Val::Usize, Def::Usize(4))
        .key(
            "targets",
            "exact levels tabulated",
            Val::Usize,
            Def::Usize(6),
        )
        .key(
            "jiter",
            "JointScfOpts::orb_max_iter",
            Val::Usize,
            Def::Usize(30),
        )
        .key(
            "ranks",
            "the energy-vs-rank scan (m is appended if missing)",
            Val::List(&Val::Usize),
            Def::List(&["2", "3"]),
        )
        .key(
            "prop",
            "property files (load_property format, at most three components — omitted ⇒ the \
             synthetic dipole-like triple)",
            Val::List(&Val::Path),
            Def::List(&[]),
        )
        .flag("nofci", "skip the exact reference")
        .flag(
            "modeopt",
            "also run the mode-optimizing pre-stage (local orbital rotations in the two-site sweep) \
             at M and hand it to the joint solver — prints the second start beside the first",
        )
        .parse();
    let t_all = Instant::now();
    let name = cli.str("fixture").to_string();
    let m = cli.usize("m");
    let k = cli.usize("k");
    let targets = cli.usize("targets");
    let jiter = cli.usize("jiter");
    let want_fci = !cli.flag("nofci");
    let want_modeopt = cli.flag("modeopt");
    let mut ranks: Vec<usize> = cli
        .strs("ranks")
        .iter()
        .filter_map(|x| x.parse().ok())
        .collect();
    if !ranks.contains(&m) {
        ranks.push(m);
    }
    ranks.sort_unstable();
    let prop_paths: Vec<String> = cli.strs("prop").to_vec();

    // ---- fixture --------------------------------------------------------------------------------
    let fx = qntt_exkit::repo!().load(&name);
    let meta = fx.load.meta.clone();
    let ints = fx.ints();
    let (d, nelec) = (fx.d(), fx.nelec());
    let (nup, ndn) = (nelec / 2, nelec / 2);
    let space = QnSpace::new(d, nup, ndn);
    // `Median` on every construction in this file, for a deterministic summation order
    // (`JointScfOpts` defaults to `Bipartite`).
    let ham = SparseHamiltonian::new_with(
        space,
        &ints.t,
        &ints.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );

    banner("rpa_casscf: DMRG-SCF ground state + RPA response on the joint manifold");
    println!(
        "{name}: d={d} spatial orbitals, N={nelec} — sector ({nup},{ndn}); reference M={m}, k={k}"
    );
    if let Some(mt) = &meta {
        println!(
            "meta: e_core = {:.9}, e_hf = {:?}, e_fci (TOTAL) = {:?}",
            mt.e_core, mt.e_hf, mt.e_ref
        );
    }

    // ---- exact reference -------------------------------------------------------------------------
    let exact: Option<Vec<f64>> = if want_fci {
        let t0 = Instant::now();
        let ev = exact_config_spectrum(space, &ham);
        println!(
            "exact reference: dim {}, {:.1}s; E0 = {:.9} (electronic; +e_core = {:.9})",
            ev.len(),
            t0.elapsed().as_secs_f64(),
            ev[0],
            ev[0] + ints.e0
        );
        Some(ev)
    } else {
        None
    };

    // ---- energy vs rank: TT-only pipeline against the joint optimizer -----------------------------
    println!(
        "\nenergy vs rank (electronic; the TT-only column is the SAME x0 with no orbital step)"
    );
    println!(
        "   {:>2}  {:>22}  {:>14}  {:>14}  {:>11}  {:>7}  {:>5}  {:>9}",
        "M", "bond totals", "E_tt-only", "E_joint", "closed [Ha]", "%FCI", "iters", "converged"
    );
    let mut chosen: Option<(usize, TtVector<f64>, f64, qntt::solvers::joint::JointScfOut)> = None;
    for &r in &ranks {
        let x_r = ground_state(space, &ham, r, 0x11A2);
        let lam_tt = qntt::rayleigh_quotient(&ham, &x_r);
        let out = joint_ground_state(
            ints,
            space,
            &x_r,
            &JointScfOpts {
                orb_max_iter: jiter,
                construction: Construction::Median,
                ..JointScfOpts::default()
            },
        );
        let totals: Vec<usize> = x_r.ranks.iter().map(RankGrid::total).collect();
        let closed = lam_tt - out.lambda;
        let pct = exact
            .as_ref()
            .map(|ev| 100.0 * closed / (lam_tt - ev[0]).max(f64::MIN_POSITIVE))
            .unwrap_or(f64::NAN);
        println!(
            "   {r:>2}  {:>22}  {lam_tt:>14.9}  {:>14.9}  {closed:>11.2e}  {pct:>6.1}%  {:>5}  {:>9}",
            format!("{totals:?}"),
            out.lambda,
            out.info.iters,
            out.info.converged
        );
        if r == m {
            chosen = Some((r, x_r, lam_tt, out));
        }
    }
    let (_, _x_ref, lam_tt, out) = chosen.expect("the scan always includes M");

    // ---- no response quantity is computed from an unconverged endpoint -----------------------------
    if !out.info.converged {
        println!(
            "\nREFUSED: joint_ground_state did not certify at M={m} after {} iterations.\n  \
             reason: {:?}\n  ‖gk‖ history: {:?}\n  ‖g_L‖ history: {:?}\n  TT ‖grad‖ history: {:?}\n  \
             trials/iter: {:?}, line-search H rebuilds: {}\n\
             No ω is published from this point: self-consistency is not validity. Retry with a larger `jiter=`, a different `m=`, or a lower \
             `orb_floor`.",
            out.info.iters,
            out.info.refused_reason,
            out.info.g_orb_kept,
            out.info.g_orb_full,
            out.info.tt_gradnorms,
            out.info.trials_per_iter,
            out.info.ls_build_count,
        );
        return;
    }
    println!(
        "\njoint optimum at M={m}: λ = {:.9} (total {:.9}); {} orbital iters, {} line-search H \
         rebuilds, terminal ‖gk‖ {:.1e} / ‖g_L‖ {:.1e} / ‖g_dropped‖ {:.1e}, TT ‖grad‖ {:.1e}, \
         u_orth_dev {:.1e}, σ_min {:.2e}",
        out.lambda,
        out.lambda + ints.e0,
        out.info.iters,
        out.info.ls_build_count,
        out.info.g_orb_kept.last().copied().unwrap_or(f64::NAN),
        out.info.g_orb_full.last().copied().unwrap_or(f64::NAN),
        out.info.g_orb_dropped.last().copied().unwrap_or(f64::NAN),
        out.info.tt_gradnorms.last().copied().unwrap_or(f64::NAN),
        out.info.u_orth_dev,
        out.info.sigma_min,
    );
    if let (Some(ev), Some(mt)) = (exact.as_ref(), meta.as_ref()) {
        println!(
            "   variational check: λ − E0_exact = {:+.2e} Ha; total − e_fci(meta) = {:+.2e} Ha",
            out.lambda - ev[0],
            out.lambda + ints.e0 - mt.e_ref.unwrap_or(f64::NAN)
        );
    }
    println!(
        "   TT-only at the same rank/seed: {lam_tt:.9} — the orbital leg closed {:.3e} Ha",
        lam_tt - out.lambda
    );

    // ---- optional: the mode-optimizing pre-stage as a second start ----------------------------------
    // The two-site sweep with LOCAL orbital rotations (`mode_optimize`) adapts the one-particle
    // basis bond by bond while it converges the state at a POOLED bond cap, and hands the joint
    // solver `(x₀, u₀)`. The bookkeeping it teaches: the joint stage runs in the rotated basis and
    // returns ITS rotation, so the total is `compose_rotations(&u₀, &out.u, d)` — first rotation
    // leftmost, never the other way — and every property would be rotated by that TOTAL. The
    // response section below stays on the first start; this block only prints the comparison.
    if want_modeopt {
        let cap = _x_ref.ranks.iter().map(RankGrid::total).max().unwrap_or(1);
        let t_pre = Instant::now();
        let pre = mode_optimize(
            ints,
            space,
            &_x_ref,
            &ModeOptOpts {
                mals: MalsOptions {
                    eps: 1e-8,
                    bond_dims: vec![cap],
                    noise: vec![0.0],
                    noise_type: NoiseType::Random,
                    max_iter: 4,
                    ..MalsOptions::default()
                },
                sweep: ModeRotSweepOpts::default(),
            },
        );
        let t_pre = t_pre.elapsed();
        let gates: usize = pre.history.sweeps.iter().map(|s| s.n_gates).sum();
        let ints_0 = rotate_integrals(ints, &pre.u);
        let out2 = joint_ground_state(
            &ints_0,
            space,
            &pre.x,
            &JointScfOpts {
                orb_max_iter: jiter,
                construction: Construction::Median,
                ..JointScfOpts::default()
            },
        );
        let u_total = compose_rotations(&pre.u, &out2.u, d);
        println!(
            "
mode-optimized start at M={m} (pooled cap {cap}): pre-stage {:.2?}, {gates} gates, \
             λ_pre = {:.9}; joint from it: λ = {:.9} in {} iters / {} refills, converged = {}; \
             u_total = u₀·u₁ with max|uᵀu − I| = {:.1e}. Against the first start: Δλ = {:+.3e} Ha, \
             iters {} → {}.",
            t_pre,
            pre.lambda,
            out2.lambda,
            out2.info.iters,
            out2.info.ls_build_count,
            out2.info.converged,
            orthogonality_defect(&u_total, d),
            out2.lambda - out.lambda,
            out.info.iters,
            out2.info.iters
        );
    }

    // ---- the response context + the orbital spectrum ------------------------------------------------
    let jctx = match joint_context(&out.h, &out.x, &JointOpts::default()) {
        Ok(c) => c,
        Err(e) => {
            println!(
                "\nREFUSED: the certified optimum failed to re-gate a response context: {e:?}"
            );
            return;
        }
    };
    let (basis, report) = jctx.joint_basis();
    println!(
        "\norbital redundancy spectrum of N = S − τᵀτ ({} pairs, floor {:.0e}): n_eff = {}, kept \
         {}, dropped {} (mass {:.1e})",
        jctx.pairs().len(),
        jctx.orb_floor(),
        jctx.n_eff(),
        report.n_orb_kept,
        report.dropped,
        report.dropped_mass
    );
    print!("   μ =");
    for mu in jctx.orb_spectrum() {
        print!(" {mu:.2e}");
    }
    println!();
    println!(
        "   response space: n_tt {} + n_orb_kept {} = {} modes; orth_error {:.1e}, σ_min {:.2e}, \
         ‖z‖ {:.2e}",
        report.n_tt,
        report.n_orb_kept,
        basis.len(),
        report.orth_error,
        jctx.sigma_min(),
        jctx.normal_residual().norm(),
    );

    // ---- dense ω (the reference) ---------------------------------------------------------------------
    let t_dense = Instant::now();
    let omega_dense = match jctx.joint_symplectic_eigenvalues(&basis) {
        SymplecticSpectrum::Stable { omega } => {
            println!(
                "dense oracle: Stable, {} ω over a {}-dim joint basis, {:.1}s",
                omega.len(),
                basis.len(),
                t_dense.elapsed().as_secs_f64()
            );
            omega
        }
        SymplecticSpectrum::Resonant { signed } => {
            println!(
                "dense oracle: RESONANT ({} unstable pair(s), metric {:?}, λ_min = {:.2e}; lowest \
                 ω² = {:.4e} ⇒ ω = {:.4}i) — the joint critical point is a saddle on the \
                 complexified manifold, not an RPA-stable minimizer. No real ω published.",
                signed.n_neg,
                signed.metric,
                signed.metric_min_eig,
                signed.omega2.first().copied().unwrap_or(0.0),
                signed.omega2.first().copied().unwrap_or(0.0).abs().sqrt()
            );
            return;
        }
        SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } => {
            let max_im = generator_eigs.iter().fold(0.0f64, |a, e| a.max(e.im.abs()));
            println!(
                "dense oracle: UNSTABLE ({reason:?}, max|Im eig(𝓜)| = {max_im:.2e}) — the joint \
                 critical point is not an RPA-stable minimizer. No ω published."
            );
            return;
        }
    };

    // ---- matrix-free lowest-k ω -------------------------------------------------------------------
    // `precond = false` (`SympPrecond::Off`): the recommended setting at the joint layer.
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCA55);
    let block = k + 2;
    let v0s = joint_seed_block(&jctx, block, &mut rng);
    let t_dav = Instant::now();
    let dav = joint_lowest_davidson(
        &jctx,
        &v0s,
        k,
        &SymplecticDavidsonOpts {
            precond: SympPrecond::Off,
            ..SymplecticDavidsonOpts::default()
        },
    );
    let omega_iter = match dav {
        Ok((omega, info)) => {
            println!(
                "Davidson (precond=false, block {block}, k={k}): converged={} in {} iterations / \
                 {} applies, {:.1}s",
                info.converged,
                info.iterations,
                info.applies,
                t_dav.elapsed().as_secs_f64()
            );
            if info.converged {
                Some(omega)
            } else {
                // An unconverged Davidson's ω is not published.
                println!(
                    "   NOT converged — residuals {:?}. No Davidson ω published; the dense oracle \
                     below stands alone.",
                    info.residuals
                );
                None
            }
        }
        Err(e) => {
            println!(
                "Davidson: {e:?} — an instability diagnosis at a certified minimizer is physics, \
                 not a solver bug (see joint_lowest_davidson's own contract). No Davidson ω."
            );
            None
        }
    };

    // ---- the ω table ------------------------------------------------------------------------------
    let n_show = k.min(omega_dense.len());
    println!("\nlowest ω (Hartree / eV) — Davidson vs dense oracle vs exact FCI gaps");
    println!(
        "   {:>3}  {:>15}  {:>12}  {:>8}  {:>10}  {:>13}  {:>9}",
        "j", "ω (Davidson)", "ω (dense)", "|diff|", "ω (eV)", "FCI E_j−E_0", "(ω−FCI)"
    );
    let mut worst = 0.0f64;
    for (j, &wd) in omega_dense.iter().enumerate().take(n_show) {
        let wi = omega_iter.as_ref().and_then(|o| o.get(j).copied());
        let diff = wi.map(|w| (w - wd).abs());
        if let Some(dv) = diff {
            worst = worst.max(dv);
        }
        let fci = exact
            .as_ref()
            .and_then(|ev| ev.get(j + 1).map(|e| e - ev[0]));
        println!(
            "   {j:>3}  {:>15}  {wd:>12.8}  {:>8}  {:>10.4}  {:>13}  {:>9}",
            wi.map(|w| format!("{w:>15.8}"))
                .unwrap_or_else(|| format!("{:>15}", "—")),
            diff.map(|v| format!("{v:>8.1e}"))
                .unwrap_or_else(|| format!("{:>8}", "—")),
            wd * HARTREE_EV,
            fci.map(|v| format!("{v:>13.8}"))
                .unwrap_or_else(|| format!("{:>13}", "—")),
            fci.map(|v| format!("{:>9.1e}", (wd - v).abs()))
                .unwrap_or_else(|| format!("{:>9}", "—")),
        );
    }
    if omega_iter.is_some() {
        println!("   max |ω_Davidson − ω_dense| over the reported k: {worst:.2e}");
    }
    if exact.is_some() {
        println!(
            "   ⚠ the (ω−FCI) column is INDEX-ALIGNED against an untyped exact list: every arm \
             here reads out ONE symmetry channel (the reference's), so a level of a different \
             total irrep occupying slot j makes that entry a misalignment, not a modelling error. \
             step-11 resolves the channel assignment."
        );
        if let Some(ev) = &exact {
            let extra = targets.min(ev.len() - 1);
            print!("   exact gaps (j = 1..{extra}):");
            for e in &ev[1..=extra] {
                print!(" {:.5}", e - ev[0]);
            }
            println!();
        }
    }

    // ---- transition moments + oscillator strengths ---------------------------------------------------
    // ★ The basis trap: `jctx` lives in the basis `out.u`, so a property expressed in the INPUT
    // orbitals must be rotated by the same congruence before it is handed over. Skipping this is an
    // O(1) error, not a small one.
    let props_in: Vec<PropertyOperator<f64>> = if prop_paths.is_empty() {
        synthetic_dipole(d)
    } else {
        prop_paths
            .iter()
            .map(|p| {
                load_property::<f64>(std::path::Path::new(p), d)
                    .unwrap_or_else(|e| panic!("property {p}: {e}"))
            })
            .collect()
    };
    let props: Vec<PropertyOperator<f64>> = props_in
        .iter()
        .map(|p| rotate_property(p, &out.u, d))
        .collect();
    println!(
        "\ntransition moments over {} propert{} [{}] — {} (rotated into the joint basis U before \
         use: the JointScfOut basis-identity contract)",
        props.len(),
        if props.len() == 1 { "y" } else { "ies" },
        props
            .iter()
            .map(|p| p.label.as_str())
            .collect::<Vec<_>>()
            .join(", "),
        if prop_paths.is_empty() {
            "SYNTHETIC, not this molecule's dipole"
        } else {
            "caller-supplied"
        }
    );
    let tm = match transition_moments(&jctx, &basis, &props, &TransitionMomentOpts::default()) {
        Ok(tm) => tm,
        Err(e) => {
            println!("   REFUSED: {e:?} — no moments published.");
            return;
        }
    };
    let f = if tm.moments.len() <= 3 {
        Some(oscillator_strengths(&tm.omega, &tm.moments))
    } else {
        println!(
            "   ({} components > 3 — oscillator strengths skipped: the 2/3 factor is the isotropic \
             average over x, y, z and has no meaning beyond them)",
            tm.moments.len()
        );
        None
    };
    println!(
        "   ⟨x|μ̂|x⟩ = {:?}; completeness_defect = {:?}; orth_error {:.1e}, σ_min {:.2e}, n_eff {}",
        tm.values
            .iter()
            .map(|v| format!("{v:.6}"))
            .collect::<Vec<_>>(),
        tm.completeness_defect
            .iter()
            .map(|v| format!("{v:.1e}"))
            .collect::<Vec<_>>(),
        tm.orth_error,
        tm.sigma_min,
        tm.n_eff
    );
    println!(
        "   {:>3}  {:>12}  {:>10}  {:>34}  {:>11}  {:>7}",
        "n", "ω", "ω (eV)", "|T_n| per component", "f_n", "cluster"
    );
    for n in 0..n_show.min(tm.omega.len()) {
        // Inside a degenerate cluster the individual T_n are an arbitrary orthogonal mixing —
        // only Σ_{n∈cluster} T_n² is defined. The cluster column says which numbers exist.
        let cl = tm
            .degenerate_clusters
            .iter()
            .find(|(lo, hi)| n >= *lo && n < *hi)
            .copied()
            .unwrap_or((n, n + 1));
        let mags: Vec<String> = tm
            .moments
            .iter()
            .map(|row| format!("{:>10.3e}", row[n].abs()))
            .collect();
        println!(
            "   {n:>3}  {:>12.8}  {:>10.4}  {:>34}  {:>11}  {:>7}",
            tm.omega[n],
            tm.omega[n] * HARTREE_EV,
            mags.join(" "),
            f.as_ref()
                .map(|v| format!("{:>11.4e}", v[n]))
                .unwrap_or_else(|| format!("{:>11}", "—")),
            if cl.1 - cl.0 > 1 {
                format!("[{},{})", cl.0, cl.1)
            } else {
                "-".to_string()
            }
        );
    }
    if let Some(fv) = &f {
        println!(
            "   Σ f over all {} modes = {:.4} (TRK would give N_e = {nelec} at full rank in a \
             complete basis — a DIAGNOSTIC here, never an oracle, and meaningless for a synthetic \
             property)",
            fv.len(),
            fv.iter().sum::<f64>()
        );
    }

    // ---- α: the moments' SOS reconstruction against the independent solve -----------------------------
    // An algebraic identity at any rank, so a disagreement is a bug in one of the two paths, not a
    // truncation effect — the example's own self-check.
    match polarizability_static(&jctx, &props, &StaticResponseOpts::default()) {
        Ok(sr) => {
            let p = props.len();
            let mut worst_rel = 0.0f64;
            for i in 0..p {
                for j in 0..p {
                    let (a, b) = (sr.alpha[i * p + j], tm.alpha_sos[i * p + j]);
                    worst_rel = worst_rel.max((a - b).abs() / a.abs().max(1.0));
                }
            }
            println!(
                "\nα(0) diagonal (solve): {:?}\n   α(0) diagonal (SOS from the moments): {:?}\n   \
                 worst |α_solve − α_SOS| / max(1,|α|) = {worst_rel:.2e}  (an algebraic identity — \
                 the error budget is completeness_defect, orth_error and the solve residual; α \
                 symmetry defect {:.1e}, solves {:?})",
                (0..p)
                    .map(|i| format!("{:.8}", sr.alpha[i * p + i]))
                    .collect::<Vec<_>>(),
                (0..p)
                    .map(|i| format!("{:.8}", tm.alpha_sos[i * p + i]))
                    .collect::<Vec<_>>(),
                sr.alpha_defect,
                sr.solves
                    .iter()
                    .map(|s| format!("{}it/{:.0e}", s.iters, s.rel_residual))
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => println!("\nα(0): REFUSED — {e:?} (reported, not hidden)"),
    }

    println!("\n[total {:.1}s]", t_all.elapsed().as_secs_f64());
}
