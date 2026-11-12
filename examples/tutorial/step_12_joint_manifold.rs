// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Joint (TT ⊕ orbital) response against cone-augmented response, at one shared reference.
//!
//! # Introduction
//!
//! ## The thesis: complementary, not competing
//!
//! Both stacks answer "the fixed-rank tangent space is too small to carry this excitation", and
//! they answer it along different axes:
//!
//! - **Cone** ([`qntt::solvers::hess::cone_response`]) widens the RESPONSE SPACE at a fixed
//!   reference, by admitting rank-INCREASING normal directions. Because those directions are not
//!   tangent to the manifold the reference sits on, the fixed-reference symplectic (Williamson)
//!   readout is no longer valid there — the point is not stationary in the added directions — so
//!   cone must relax the reference INSIDE the readout and reports a CI spectrum instead
//!   (`ConeResponse::gaps` measured from the relaxed `e0`). The price is that the object read out
//!   is not an RPA spectrum at the caller's point.
//! - **Joint** ([`qntt::solvers::joint`]) widens the MANIFOLD, from `M_r` to `{e^{−κ̂}·M_r}`, and
//!   then RESTORES STATIONARITY on the widened manifold ([`joint_ground_state`]). At the joint
//!   critical point the fixed-reference symplectic readout is valid again, so the answer is a
//!   genuine RPA spectrum with transition moments, oscillator strengths and analytic
//!   polarizabilities hanging off it. The price is that `κ̂` is a `d×d` ONE-PARTICLE unitary: it
//!   redresses which orbitals the rank is spent on, it cannot manufacture TT rank.
//!
//! So the axis is **reachability vs. principled response**. Cone reaches states the tangent space
//! structurally cannot represent, at the cost of the readout's status; joint keeps a certified
//! symplectic readout and buys accuracy through a better reference, at the cost of never leaving
//! the one-particle-rotation orbit of the fixed-rank manifold.
//!
//! At LiH the complementarity holds in the strong sense: each stack recovers a degenerate pair
//! the other misses at these budgets. ★ What that shows is WHICH CONSTRUCTION reaches a level,
//! never the MECHANISM — an untyped cone can cross an irrep wall as readily as a rank one, and a
//! joint direction `u_a = −Ĝ_pq·x` with `Γ_p ≠ Γ_q` carries character the reference's own channel
//! does not, which is a symmetry wall rather than a rank one. Read a class label as "this arm
//! reaches it, that one does not" and nothing stronger; confirm any mechanism with the
//! per-channel driver.
//!
//! ## Arms (per fixture, at one per-sector rank `m`)
//!
//! | arm | what it is | reference |
//! |---|---|---|
//! | (a) `tangent-CI` / `tangent-W` | `cone_response{max_added: 0, williamson: true}` — the two plain RPA readouts, zero augmentation | the cone GS at `m` |
//! | (b) `joint@fixed` | `joint_context` at the SAME `x`, orbital directions ADDED, reference NOT re-optimized | the same `x` |
//! | (c) `joint@opt` | `joint_ground_state` seeded from that same `x`, then the symplectic readout at the joint optimum — the recommended use | the joint optimum |
//! | (d) `cone k=…` | `cone_response` defaults at growing budgets | the same `x` |
//!
//! ★ **The two readouts are not the same object, and the table says so.** Arms (b)/(c) produce a
//! fixed-reference Williamson spectrum; the like-for-like cone comparator for them is `tangent-W`,
//! NOT `tangent-CI`, and the cone arms' `gaps` are CI values measured from a RELAXED `e0`.
//! Comparing (c) against `cone k=16` is a comparison of two different constructions that happen to
//! estimate the same physical gaps — which is the point of the example, but it is not a horse race
//! between two implementations of one algorithm.
//!
//! ★★ **Arm (b) is deliberately non-stationary and its numbers carry that.** [`joint_context`]
//! checks TT-criticality only. It does NOT check the ORBITAL gradient, and
//! `joint_symplectic_eigenvalues` does not either (only the response entry points do). So arm (b)
//! reads `B` at a point that is not stationary in exactly the directions it just added —
//! structurally the same pathology as the cone's own fixed-reference augmented Williamson.
//! `‖g_L‖ = ‖orbital_gradient‖` is printed beside every (b) number as the amplitude of that bias.
//! Arm (c) exists because arm (b) is not a trustworthy readout.
//!
//! ## Reachability classification
//!
//! Per exact level, the first arm class whose gap error clears [`REACH_TOL`]: `tangent` (the
//! tangent space already carries it) → `orbital` (a joint arm's added one-particle directions
//! carry it) → `cone` (only rank-increasing directions carry it) → `unreached`. The class is a
//! statement about which CONSTRUCTION reaches the level, not a quality score.
//!
//! ⚠ **A missing level is not automatically a RANK wall.** Every arm here reads out ONE symmetry
//! channel — the one the reference lives in — and the untyped (`PointGroup::TRIVIAL`) spaces this
//! example builds do not label the exact list by irrep, so an index-aligned per-`j` error can be
//! comparing against a level of a different total irrep (LiH has exactly this trap; the
//! per-channel driver of step-11 resolves it). Read `unreached` as "no arm here reaches slot
//! `j`", not as "rank was the obstruction".
//!
//! For that reason the run prints the classification TWICE. The first table is INDEX-ALIGNED,
//! which makes it comparable row by row and is what carries the misalignment. The second is a
//! NEAREST-MODE match, `min_n |ω_n − gap_j|` over each class's whole spectrum, which asks the
//! misalignment-proof question — does this construction carry a mode near this level ANYWHERE —
//! and is the one the reachability verdict is read off. A class that published NO spectrum
//! prints `—`, never a number, and is excluded from the asymmetry claims in both directions: "the
//! other stack refused" is not evidence that this one reached anything.
//!
//! ## What each fixture is for
//!
//! **LiH/STO-3G, `M = 3`** is the REACHABILITY regime: the shared reference is TT-critical and
//! orbital-NON-critical, arm (c) certifies in a handful of orbital iterations and lands on the
//! exact sector ground state — and STILL does not see the pair cone `k=16` resolves, while a joint
//! arm carries a pair no cone budget here reaches (the untyped cone ladder SATURATES, so larger
//! `k` give identical results). Both joint arms show the reverse asymmetry, so it is not an
//! artifact of the orbital relaxation.
//!
//! **H2O/STO-3G, `M = 3`** is the ACCURACY regime — its tangent space misses no state entirely —
//! and NEITHER joint arm publishes a spectrum there:
//!
//! - arm (b) goes `Unstable` with `HrrNotPd`: the indefinite orbital block at an
//!   orbital-NON-stationary point is exactly the pathology the arm exists to expose, and here it
//!   destroys `H^rr ≻ 0` outright rather than biasing it;
//! - arm (c) does NOT certify: the orbital line search stalls ABOVE the Armijo resolution floor,
//!   so `joint_ground_state` reports its "NOT the expected roundoff-floor exit" refusal rather
//!   than a quiet success. Energetically it closes almost the whole reference FCI gap, but the
//!   convergence test is on the GRADIENT and this example publishes no ω from an uncertified
//!   endpoint; it does not loosen `orb_grad` to work around that.
//!
//! ★ Where a class published NOTHING the comparison is VACUOUS on that side, and the example says
//! so instead of awarding the other stack an uncontested win. Cone's broad-spectrum improvement
//! over tangent-W is also not across the board: at the level tangent-W already resolves best,
//! cone can be slightly worse.
//!
//! # Builds on
//!
//! step-10
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-12                       # both fixtures, defaults
//! cargo run --release --example step-12 -- lih_sto3g m=3 k=4,16
//! cargo run --release --example step-12 -- h2o_sto3g jiter=8
//! cargo run --release --example step-12 -- --help
//!
//! fixtures    <str>        fixtures under testdata/       [default lih_sto3g,h2o_sto3g]
//! m=          <usize>      per-sector rank cap of the shared reference (uniform across sectors) [default 3]
//! k=          <usize>[,…]  cone budgets                                          [default 4,16]
//! targets=    <usize>      levels reported                                       [default 6]
//! jiter=      <usize>      cap on JointScfOpts::orb_max_iter for arm (c)         [default 30]
//! jverbose    <bool>       trace arm (c)'s orbital iterations to stderr       [default false]
//! --format <text|tsv>      output format                                       [default text]
//! ```
//!
//! ⏱ **Budget arm (c) before running the default set.** Arm (c) is a full DMRG-SCF optimization
//! and dominates the cost of the run, most of all on H2O, where it also does NOT certify. Each
//! arm prints its diagnosis as it finishes, so a long H2O run is visibly progressing; `jiter=`
//! bounds it, and an arm (c) that exhausts its budget or stalls reports `NOT converged` and
//! publishes no ω rather than quoting an uncertified spectrum.
//!
//! # Results
//!
//! Per fixture the run prints the dense reference (`E0` by configuration enumeration, plus the
//! `(nup+1, ndn−1)` partner sector that supplies the S labels), the shared reference's diagnosis,
//! one line per arm as it finishes, a recap of those lines, the INDEX-ALIGNED gap table, the
//! NEAREST-MODE table, and the two-directional asymmetry verdict.
//!
//! On LiH/STO-3G the dense reference is 225 configurations (+ 120 in the partner sector),
//! `E0 = −8.877719570`; the shared `M = 3` reference sits 1.32e-6 Ha above it with
//! `‖grad‖ = 6.9e-13` (TT-critical) and `‖g_L‖ = 1.98e-4` (orbital-NON-critical), which is exactly
//! the setup arm (b) is built to expose. Arm (c) certifies in 3 orbital iterations (2 Hamiltonian
//! rebuilds), closing 100.00 % of the reference's gap to `E0` and leaving `‖g_L‖ = 6.9e-8`. In the
//! nearest-mode table levels j = 1, 2 are `tangent`; j = 3, 4 are cone-only (cone 1.2e-7 against
//! joint 2.0e-2) and j = 5, 6 are joint-only (joint 2.6e-4 against cone 2.0e-2) — the
//! complementarity in both directions, printed as the two ★ lines.
//!
//! H2O/STO-3G is the long fixture, and the recorded output below stops inside it, after the dense
//! reference (441 + 245 configurations, `E0 = −83.821320365`), the `M = 3` reference (4.18e-4 Ha
//! above `E0`, `‖grad‖ = 7.2e-12`, `‖g_L‖ = 1.21e-3`) and arm (b), which reads out
//! `Unstable { HrrNotPd { min_eig: −8197.8 } }` — no ω published, as described above. The rest of
//! that fixture's behaviour is described in the section above; run it yourself with a budget.
//!
//! # Possibilities for extensions
//!
//! - Run one fixture at a time (`step-12 lih_sto3g`) and raise `targets=` past 6: the levels the
//!   dense list adds are where the index-aligned and nearest-mode tables disagree most.
//! - Push `k=4,16,32` at LiH: the untyped cone ladder saturates, so the extra budget should change
//!   nothing — a cheap check that the j = 5, 6 asymmetry is not simply an under-budgeted cone.
//! - Raise `m=`: a fatter shared reference moves both stacks at once, and the classification is
//!   only interesting while the tangent space still misses levels.
//! - Run `jverbose jiter=8` on H2O to watch arm (c)'s orbital line search stall above the Armijo
//!   floor, which is what the `NOT converged` refusal is reporting.
//! - Take a level this run leaves `unreached` to step-11's per-channel driver, which labels the
//!   exact list by channel and so separates a symmetry wall from a rank wall.

use qntt::hamiltonian::assembly::Construction;
use qntt::round::RoundMode;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::hess::{ConeOpts, ConeResponse, SymplecticSpectrum, cone_response};
use qntt::solvers::joint::{JointOpts, JointScfOpts, joint_context, joint_ground_state};
use qntt::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg};
use qntt::{QnSpace, RankGrid, SparseHamiltonian, TtVector, dot, orbital_gradient, tt_state};
use qntt_exkit::args::{Def, Spec, Val};
use qntt_exkit::grad::abelian::norm_projected as grad_norm_at;
use qntt_exkit::out::banner;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

/// Absolute gap-error threshold at which a level counts as REACHED by an arm class. Gaps here are
/// `O(0.1–0.5)` Ha, so this is ~1 % relative — deliberately loose, because the classification is a
/// statement about STRUCTURE (does this construction see the state at all), and the separation
/// between a reached and an unreached level measures four to six decades, not marginal. Printed
/// with the table so a reader can re-classify at their own bar from the raw errors.
const REACH_TOL: f64 = 1e-3;

// --- exact reference: configuration enumeration + sector-degeneracy spin labels ---------------
// Copied WHOLESALE from step-10 (`examples/tutorial/step_10_cone_response.rs`); an example
// cannot import another example's items, and the library exposes no dense-FCI helper. Keeping the
// enumeration order identical is what makes the printed `E0` reproduce that step's exactly.

/// All `n`-element subsets of `0..d`, lexicographic.
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

/// Dense symmetric eigenvalues via cyclic Jacobi (the example-local oracle, faer-free).
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

// --- the shared reference: step-10's ground-state recipe, unchanged -----------------------------
// Reused UNCHANGED so the reference is not a tuned variable across the two stacks. Its
// TT-gradient must clear `JointOpts::tt_grad_tol`'s bound (`1e-6·max(1,|λ|)`) for
// `joint_context` to accept it, which is what makes arm (b) legal to build at all.

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
    let grid: Vec<RankGrid> = qntt_exkit::ranks::abelian::uniform(space, m);
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
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let polish = |y0: &TtVector<f64>| -> TtVector<f64> {
        let (_l, mut y) = riemannian_cg(ham, y0, &opts);
        for _ in 0..30 {
            if grad_norm_at(ham, &y) < 1e-10 {
                break;
            }
            let (_l2, y2) = riemannian_cg(ham, &y, &opts);
            y = y2;
        }
        y
    };
    let mut x = polish(&xa);
    normalize_mut(&mut x);
    x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    let x = polish(&x);
    let mut x = x;
    normalize_mut(&mut x);
    x.check();
    x
}

/// Frobenius norm of the orbital gradient `g[p,q] = 2(γ_pq − γ_qp)` — the amplitude of arm (b)'s
/// non-stationarity, and the quantity arm (c) drives to `orb_grad`.
fn orb_grad_norm(ham: &SparseHamiltonian<f64>, x: &TtVector<f64>) -> f64 {
    orbital_gradient(ham, x)
        .data
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

// --- arms ---------------------------------------------------------------------------------------

/// Which construction an arm belongs to — the classification ladder reads these in order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ArmClass {
    Tangent,
    Joint,
    Cone,
}

struct Arm {
    label: String,
    class: ArmClass,
    /// Ascending gaps, or `None` when the arm refused / did not converge. **Every ω published here
    /// has already passed its arm's own acceptance check**; a `None` is printed as `—`, never as
    /// a number.
    gaps: Option<Vec<f64>>,
    /// One-line diagnosis, printed as soon as the arm finishes AND again above the table.
    note: String,
}

/// Record an arm and print its diagnosis immediately. Arm (c) at H2O scale runs for a long time;
/// collecting every note for a single print at the end would leave the run silent for that whole
/// stretch, which is indistinguishable from a hang.
fn push_arm(arms: &mut Vec<Arm>, arm: Arm) {
    println!("   {}", arm.note);
    arms.push(arm);
}

fn spectrum_gaps(s: &SymplecticSpectrum, what: &str) -> Option<Vec<f64>> {
    match s {
        SymplecticSpectrum::Stable { omega } => Some(omega.clone()),
        SymplecticSpectrum::Resonant { signed } => {
            println!(
                "   {what}: Resonant ({} unstable pair(s), metric {:?}, lowest ω² = {:.2e}) — \
                 reported, not hidden; no real ω published from this arm",
                signed.n_neg,
                signed.metric,
                signed.omega2.first().copied().unwrap_or(0.0)
            );
            None
        }
        SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } => {
            let max_im = generator_eigs.iter().fold(0.0f64, |a, e| a.max(e.im.abs()));
            println!(
                "   {what}: Unstable ({reason:?}, max|Im eig(𝓜)| = {max_im:.2e}) — reported, not \
                 hidden; no ω published from this arm"
            );
            None
        }
    }
}

/// `cone_response` under a suppressed panic hook: the basis-integrity entry guard PANICS at a
/// gauge-degenerate reference. The shared reference here is a certified cone GS so this should
/// never fire, but an arm that dies with a stack trace tells the reader nothing about the
/// comparison.
fn cone_arm(
    ham: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    opts: &ConeOpts,
) -> Result<ConeResponse, String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cone_response(ham, x, opts)));
    std::panic::set_hook(prev);
    r.map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| "non-string panic payload".to_string())
    })
}

fn fmt_cell(g: Option<f64>, exact: f64) -> String {
    match g {
        Some(v) => format!("{v:>8.5} ({:.0e})", (v - exact).abs()),
        None => format!("{:>8} {:>7}", "—", ""),
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::fn_params_excessive_bools)]
fn run_fixture(name: &str, m: usize, ks: &[usize], targets: usize, jiter: usize, jverbose: bool) {
    let t_all = Instant::now();
    let ints = qntt_exkit::repo!().load(name).into_ints();
    let (d, nelec) = (ints.norb, ints.nelec);
    let (nup, ndn) = (nelec / 2, nelec / 2);

    println!("\n================================================================================");
    banner(format_args!(
        "{name}: d={d} spatial orbitals, N={nelec} — sector ({nup},{ndn}), M={m}"
    ));

    // `Construction::Median` on BOTH stacks (a deterministic summation order), while
    // `JointScfOpts` defaults to `Bipartite`. A mismatch would make arm (c)'s λ differ from the
    // others' at 1e-14 for no reason.
    let space = QnSpace::new(d, nup, ndn);
    let ham = SparseHamiltonian::new_with(
        space,
        &ints.t,
        &ints.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );

    // ---- exact reference + sector-degeneracy spin labels ------------------------------------
    let t0 = Instant::now();
    let ev = exact_config_spectrum(space, &ham);
    let sp_p = QnSpace::new(d, nup + 1, ndn - 1);
    let ham_p = SparseHamiltonian::new_with(
        sp_p,
        &ints.t,
        &ints.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );
    let ev_p = exact_config_spectrum(sp_p, &ham_p);
    let is_singlet =
        |e: f64| -> bool { ev_p.iter().all(|p| (e - p).abs() > 1e-8 * e.abs().max(1.0)) };
    println!(
        "exact reference: dim {} (+ partner {}), {:.1}s; E0 = {:.9}",
        ev.len(),
        ev_p.len(),
        t0.elapsed().as_secs_f64(),
        ev[0]
    );
    let exact_gaps: Vec<(f64, bool)> = ev[1..=targets.min(ev.len() - 1)]
        .iter()
        .map(|e| (e - ev[0], is_singlet(*e)))
        .collect();

    // ---- the shared reference every fixed-reference arm reads at -----------------------------
    let tm = Instant::now();
    let x = ground_state(space, &ham, m, 0x11A2);
    let lam = qntt::rayleigh_quotient(&ham, &x);
    let gnorm = grad_norm_at(&ham, &x);
    let gl = orb_grad_norm(&ham, &x);
    let xranks: Vec<usize> = x.ranks.iter().map(RankGrid::total).collect();
    println!(
        "reference M={m}: E = {:.9} (E−E0 = {:.2e}), ‖grad‖ = {gnorm:.1e}, ‖g_L‖ = {gl:.2e}, \
         bond totals {xranks:?}, {:.1}s",
        lam,
        lam - ev[0],
        tm.elapsed().as_secs_f64()
    );
    if gnorm > 1e-9 {
        println!(
            "   WARNING: reference not fully TT-critical — every arm below inherits that bias"
        );
    }

    let mut arms: Vec<Arm> = Vec::new();

    // ---- (a) tangent-only: relaxed-reference CI + fixed-reference Williamson ------------------
    let base = cone_arm(
        &ham,
        &x,
        &ConeOpts {
            targets,
            max_added: 0,
            williamson: true,
            ..Default::default()
        },
    )
    .unwrap_or_else(|msg| panic!("the shared reference must be cone-admissible: {msg}"));
    let tw = base
        .williamson
        .as_ref()
        .and_then(|s| spectrum_gaps(s, "tangent-W"));
    push_arm(
        &mut arms,
        Arm {
            label: "tangent-CI".into(),
            class: ArmClass::Tangent,
            gaps: Some(base.gaps.clone()),
            note: format!(
                "(a) tangent-CI: relaxed-reference CI over the tangent space, E0^CI−E0 = {:+.2e}",
                base.e0 - ev[0]
            ),
        },
    );
    push_arm(
        &mut arms,
        Arm {
            label: "tangent-W".into(),
            class: ArmClass::Tangent,
            gaps: tw,
            note:
                "(a) tangent-W: fixed-reference Williamson — the LIKE-FOR-LIKE comparator for the \
               joint arms"
                    .into(),
        },
    );

    // ---- (b) joint at the SAME fixed reference ------------------------------------------------
    let tb = Instant::now();
    match joint_context(&ham, &x, &JointOpts::default()) {
        Ok(jctx) => {
            let (basis, report) = jctx.joint_basis();
            let g = jctx.joint_symplectic_eigenvalues(&basis);
            let note = format!(
                "(b) joint@fixed: n_tt {} + n_orb_kept {} = {} modes (n_eff {}, dropped {}), \
                 orth_error {:.1e}, σ_min {:.2e}, ‖g_L‖ {gl:.2e} — NOT orbital-stationary, the ω \
                 carry a first-order-in-‖g_L‖ bias; {:.1}s",
                report.n_tt,
                report.n_orb_kept,
                basis.len(),
                jctx.n_eff(),
                report.dropped,
                report.orth_error,
                jctx.sigma_min(),
                tb.elapsed().as_secs_f64()
            );
            push_arm(
                &mut arms,
                Arm {
                    label: "joint@fixed".into(),
                    class: ArmClass::Joint,
                    gaps: spectrum_gaps(&g, "joint@fixed"),
                    note,
                },
            );
        }
        Err(e) => push_arm(
            &mut arms,
            Arm {
                label: "joint@fixed".into(),
                class: ArmClass::Joint,
                gaps: None,
                note: format!("(b) joint@fixed: context REFUSED — {e:?}"),
            },
        ),
    }

    // ---- (c) joint at the JOINT-OPTIMIZED reference (the recommended use) ----------------------
    // Seeded from the SAME `x`, so the only variable against arms (a)/(b)/(d) is the orbital
    // relaxation. `joint_ground_state`'s own doc asks the caller to prepare the basin; a converged
    // cone GS is exactly that prepared `x0`.
    let tc = Instant::now();
    let jopts = JointScfOpts {
        orb_max_iter: jiter,
        // `jverbose` traces each orbital iteration to stderr. Arm (c) dominates the run's cost,
        // and without the trace a long run is indistinguishable from a hang.
        verbose: jverbose,
        construction: Construction::Median,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&ints, space, &x, &jopts);
    let dtc = tc.elapsed().as_secs_f64();
    if out.info.converged {
        // The rotated basis is a UNITARY equivalence within the sector (an orbital rotation
        // conserves Nup/Ndn separately), so `ev` computed once in the native basis stays the
        // correct reference for this arm's gaps. It is NOT a legal reference for a property
        // operator — see step-15 (`tutorial/step_15_dmrg_scf_to_rpa.rs`), which rotates one.
        let closed = lam - out.lambda;
        let frac = closed / (lam - ev[0]).max(f64::MIN_POSITIVE);
        match joint_context(&out.h, &out.x, &JointOpts::default()) {
            Ok(jctx) => {
                let (basis, report) = jctx.joint_basis();
                let g = jctx.joint_symplectic_eigenvalues(&basis);
                let note = format!(
                    "(c) joint@opt: converged in {} orbital iters ({} H rebuilds); \
                     λ_joint−E0 = {:+.2e}, E_tt−E_joint = {closed:.3e} Ha ({:.2} % of the \
                     reference's FCI gap closed); n_tt {} + {} orb, n_eff {}, ‖g_L‖ {:.1e}, \
                     orth_error {:.1e}, σ_min {:.2e}; {dtc:.1}s",
                    out.info.iters,
                    out.info.ls_build_count,
                    out.lambda - ev[0],
                    100.0 * frac,
                    report.n_tt,
                    report.n_orb_kept,
                    jctx.n_eff(),
                    out.info.g_orb_full.last().copied().unwrap_or(f64::NAN),
                    report.orth_error,
                    jctx.sigma_min(),
                );
                push_arm(
                    &mut arms,
                    Arm {
                        label: "joint@opt".into(),
                        class: ArmClass::Joint,
                        gaps: spectrum_gaps(&g, "joint@opt"),
                        note,
                    },
                );
            }
            Err(e) => push_arm(
                &mut arms,
                Arm {
                    label: "joint@opt".into(),
                    class: ArmClass::Joint,
                    gaps: None,
                    note: format!(
                        "(c) joint@opt: optimizer converged but the readout context REFUSED — {e:?}"
                    ),
                },
            ),
        }
    } else {
        // An uncertified endpoint's ω is not published — a badly wrong ω₁ can sit behind a
        // perfect-looking self-consistency margin.
        push_arm(
            &mut arms,
            Arm {
                label: "joint@opt".into(),
                class: ArmClass::Joint,
                gaps: None,
                note: format!(
                    "(c) joint@opt: NOT converged after {} iters — no ω published. reason: {:?} \
                 (λ = {:+.2e} vs E0, ‖g_L‖ {:.1e}); {dtc:.1}s",
                    out.info.iters,
                    out.info.refused_reason,
                    out.lambda - ev[0],
                    out.info.g_orb_full.last().copied().unwrap_or(f64::NAN),
                ),
            },
        );
    }

    // ---- (d) cone-CI at the same fixed reference ----------------------------------------------
    for &k in ks {
        let tk = Instant::now();
        match cone_arm(
            &ham,
            &x,
            &ConeOpts {
                targets,
                max_added: k,
                ..Default::default()
            },
        ) {
            Ok(res) => {
                let gs = res.character.first().copied().unwrap_or([f64::NAN; 3]);
                let note = format!(
                    "(d) cone k={k}: added {} in {} rounds, E0^CI−E0 = {:+.2e}, gs r/t/c = \
                     {:.6}/{:.1e}/{:.1e}, guards iso {:.0e} / ortho {:.0e}; {:.1}s",
                    res.added,
                    res.rounds,
                    res.e0 - ev[0],
                    gs[0],
                    gs[1],
                    gs[2],
                    res.guards.iso,
                    res.guards.ortho,
                    tk.elapsed().as_secs_f64()
                );
                push_arm(
                    &mut arms,
                    Arm {
                        label: format!("cone k={k}"),
                        class: ArmClass::Cone,
                        gaps: Some(res.gaps.clone()),
                        note,
                    },
                );
            }
            Err(msg) => push_arm(
                &mut arms,
                Arm {
                    label: format!("cone k={k}"),
                    class: ArmClass::Cone,
                    gaps: None,
                    note: format!("(d) cone k={k}: REFUSED by the basis-integrity guard — {msg}"),
                },
            ),
        }
    }

    // ---- the summary table ---------------------------------------------------------------------
    // The per-arm notes were printed as each arm finished (a fixture can take a while);
    // recapped here so the table and its diagnoses can be read as one block.
    println!("\n   -- recap of the arm diagnoses --");
    for a in &arms {
        println!("   {}", a.note);
    }
    println!();
    print!("   {:>3} {:>2} {:>9}", "j", "S", "exact");
    for a in &arms {
        print!("  {:>17}", a.label);
    }
    println!("  {:>10}", "class");
    for (j, (ge, singlet)) in exact_gaps.iter().enumerate() {
        let lbl = if *singlet { "S" } else { "T+" };
        print!("   {:>3} {:>2} {:>9.5}", j + 1, lbl, ge);
        for a in &arms {
            print!(
                "  {:>17}",
                fmt_cell(a.gaps.as_ref().and_then(|g| g.get(j).copied()), *ge)
            );
        }
        // First class in the ladder that reaches this level.
        let reaches = |c: ArmClass| -> bool {
            arms.iter().filter(|a| a.class == c).any(|a| {
                a.gaps
                    .as_ref()
                    .and_then(|g| g.get(j).copied())
                    .is_some_and(|v| (v - ge).abs() <= REACH_TOL)
            })
        };
        let class = if reaches(ArmClass::Tangent) {
            "tangent"
        } else if reaches(ArmClass::Joint) {
            "orbital"
        } else if reaches(ArmClass::Cone) {
            "cone"
        } else {
            "unreached"
        };
        println!("  {class:>10}");
    }
    println!(
        "   class ladder: tangent → orbital (a joint arm's added one-particle directions) → cone \
         (rank-increasing directions) → unreached, at |Δgap| ≤ {REACH_TOL:.0e}, INDEX-ALIGNED."
    );

    // ---- nearest-mode match: the misalignment-proof reading -------------------------------------
    // The table above is index-aligned, which is what makes it comparable row by row — and
    // which is exactly where the trap lives: every arm reads out ONE symmetry
    // channel, so when a level of a different total irrep occupies slot `j` in the untyped exact
    // list, every arm's slot `j` reports its own NEXT available mode and the error column measures
    // a misalignment, not a modelling failure. The classification below asks the question that
    // survives that: does the arm's spectrum contain a mode near this exact level ANYWHERE?
    // `best[j]` is `min_n |ω_n − gap_j|` over the arm's whole ω list.
    let best = |a: &Arm, ge: f64| -> Option<f64> {
        a.gaps.as_ref().map(|g| {
            g.iter()
                .fold(f64::INFINITY, |acc, &v| acc.min((v - ge).abs()))
        })
    };
    // `None` when NO arm of that class produced a spectrum at all — structurally different from
    // "produced one and missed": a class with nothing to compare cannot support a reachability
    // claim in either direction, and printing `inf` in that cell would read as a measured miss.
    let best_class = |c: ArmClass, ge: f64| -> Option<f64> {
        arms.iter()
            .filter(|a| a.class == c)
            .filter_map(|a| best(a, ge))
            .fold(None, |acc: Option<f64>, v| {
                Some(acc.map_or(v, |a| a.min(v)))
            })
    };
    let reached = |b: Option<f64>| b.is_some_and(|v| v <= REACH_TOL);
    let missed = |b: Option<f64>| b.is_some_and(|v| v > REACH_TOL);
    let cell = |b: Option<f64>| b.map_or_else(|| format!("{:>10}", "—"), |v| format!("{v:>10.1e}"));
    println!("\n   nearest-mode match (min |ω − exact| over each class's WHOLE spectrum):");
    println!(
        "   {:>3} {:>2} {:>9}  {:>10}  {:>10}  {:>10}  {:>10}",
        "j", "S", "exact", "tangent", "joint", "cone", "class"
    );
    // Levels only one of the two stacks carries — the complementarity check.
    let mut cone_only: Vec<String> = Vec::new();
    let mut joint_only: Vec<String> = Vec::new();
    for (j, (ge, singlet)) in exact_gaps.iter().enumerate() {
        let (bt, bj, bc) = (
            best_class(ArmClass::Tangent, *ge),
            best_class(ArmClass::Joint, *ge),
            best_class(ArmClass::Cone, *ge),
        );
        let class = if reached(bt) {
            "tangent"
        } else if reached(bj) {
            "orbital"
        } else if reached(bc) {
            "cone"
        } else {
            "unreached"
        };
        // An asymmetry claim needs BOTH sides to have produced a spectrum: `missed` is false for a
        // class that published nothing, so a level where one stack refused outright is excluded
        // from both lists rather than counted as a win for the other.
        if !reached(bt) {
            if reached(bc) && missed(bj) {
                cone_only.push(format!(
                    "j={} (cone {:.1e} vs joint {:.1e})",
                    j + 1,
                    bc.unwrap(),
                    bj.unwrap()
                ));
            }
            if reached(bj) && missed(bc) {
                joint_only.push(format!(
                    "j={} (joint {:.1e} vs cone {:.1e})",
                    j + 1,
                    bj.unwrap(),
                    bc.unwrap()
                ));
            }
        }
        println!(
            "   {:>3} {:>2} {:>9.5}  {}  {}  {}  {class:>10}",
            j + 1,
            if *singlet { "S" } else { "T+" },
            ge,
            cell(bt),
            cell(bj),
            cell(bc)
        );
    }

    // ---- complementarity, checked in BOTH directions -------------------------------------------
    // The cone-only direction follows from the rank argument (a d×d one-particle unitary
    // redresses which orbitals the rank is spent on but cannot manufacture TT rank). The reverse
    // direction is reported just as loudly if it fires — the orbital leg's own reach is checked,
    // not assumed.
    println!("   — reachability asymmetry, both directions (tangent-invisible levels only):");
    // A class with no spectrum at all makes the comparison VACUOUS on that side. Say so, rather
    // than let the other stack collect an uncontested "only" — the arm diagnoses above already
    // recorded WHY it published nothing (an Unstable readout, or the convergence check refusing).
    for (c, what) in [
        (ArmClass::Tangent, "tangent"),
        (ArmClass::Joint, "joint"),
        (ArmClass::Cone, "cone"),
    ] {
        if arms
            .iter()
            .filter(|a| a.class == c)
            .all(|a| a.gaps.is_none())
        {
            println!(
                "     ({what}: NO arm of this class published a spectrum — see its diagnosis \
                 above. Every asymmetry claim involving {what} is vacuous at this fixture and is \
                 suppressed, not scored against it.)"
            );
        }
    }
    if cone_only.is_empty() {
        println!("     cone-only: NONE at this fixture/budget.");
    } else {
        println!(
            "     ★ cone-only (no joint arm carries it): {} — the predicted asymmetry. \
             Rank-increasing directions reach it; a one-particle rotation of the SAME fixed-rank \
             manifold does not, even from a reference driven to the exact ground state.",
            cone_only.join(", ")
        );
    }
    if joint_only.is_empty() {
        println!("     joint-only: NONE at this fixture/budget.");
    } else {
        println!(
            "     ★★ joint-only (no cone budget here carries it): {} — the NOT-predicted \
             direction, reported as a finding. Candidate mechanism, and explicitly NOT rank \
             creation: at PointGroup::TRIVIAL the orbital pair list is the full lower triangle, so \
             a direction u_a with Γ_p ≠ Γ_q transforms as Γ_p ⊗ Γ_q ≠ A1 and carries character the \
             reference's own channel does not — i.e. the wall crossed is a SYMMETRY wall, which is \
             a different obstruction from the one the cone lifts. Confirm the channel assignment \
             with the per-channel driver before reading it any other way.",
            joint_only.join(", ")
        );
    }
    println!("[{name} total {:.1}s]", t_all.elapsed().as_secs_f64());
}

fn main() {
    let cli = Spec::new("step-12")
        .about("joint (TT ⊕ orbital) response against cone-augmented response, same references")
        .rest(
            "fixtures",
            "fixtures under testdata/",
            Val::Str,
            Def::List(&["lih_sto3g", "h2o_sto3g"]),
        )
        .key(
            "m",
            "per-sector rank cap of the shared reference (uniform across sectors)",
            Val::Usize,
            Def::Usize(3),
        )
        .key(
            "k",
            "cone budgets",
            Val::List(&Val::Usize),
            Def::List(&["4", "16"]),
        )
        .key("targets", "levels reported", Val::Usize, Def::Usize(6))
        .key(
            "jiter",
            "cap on JointScfOpts::orb_max_iter for arm (c)",
            Val::Usize,
            Def::Usize(30),
        )
        .flag("jverbose", "trace arm (c)'s orbital iterations to stderr")
        .parse();
    let fixtures: Vec<String> = cli.strs("fixtures").to_vec();
    let m = cli.usize("m");
    let ks: Vec<usize> = cli
        .strs("k")
        .iter()
        .filter_map(|x| x.parse().ok())
        .collect();
    let targets = cli.usize("targets");
    let jiter = cli.usize("jiter");
    let jverbose = cli.flag("jverbose");

    println!(
        "joint (TT ⊕ orbital) vs cone-augmented response — complementary, not competing: cone \
         widens the RESPONSE SPACE at a fixed reference and must relax that reference inside the \
         readout; joint widens the MANIFOLD and restores stationarity, so the fixed-reference \
         symplectic readout stays valid. Reachability vs principled response."
    );
    println!("M={m}, cone k∈{ks:?}, targets={targets}, joint orb_max_iter={jiter}");
    for f in &fixtures {
        run_fixture(f, m, &ks, targets, jiter, jverbose);
    }
}
