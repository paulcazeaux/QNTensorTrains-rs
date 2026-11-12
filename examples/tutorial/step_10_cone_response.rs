// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Cone response: rank-increasing directions in the response ansatz at a fixed low-rank reference.
//!
//! # Introduction
//!
//! Molecular demonstration: cone-augmented response spectra of small molecules (library API
//! `solvers::hess::cone_response` and its SU(2) twin).
//!
//! The point being demonstrated: at a FIXED low-rank ground state, adding a handful of targeted
//! normal (rank-increasing) directions to the response ansatz recovers excitation gaps far beyond
//! what the tangent space alone supports — decoupling the ground state's rank from the response
//! space's expressiveness. The fixtures are small enough (sector dims 225/441) that the exact
//! spectrum is computable by dense configuration enumeration, and spin labels come free from the
//! sector-degeneracy trick: an `S ≥ 1` level in `(nup, ndn)` has an exact degenerate partner in
//! `(nup+1, ndn−1)`, a singlet does not.
//!
//! Arms per fixture and rank `M` (per-sector on the abelian arms, pooled reduced-multiplet on
//! the SU(2) arm):
//!   - `tCI` / `tW` — tangent-only relaxed-reference CI and fixed-reference Williamson
//!     (the plain RPA readouts, zero cone directions);
//!   - `k=…` — `cone_response` at growing normal-direction budgets (CI readout);
//!   - exact — configuration enumeration, with S labels;
//!   - SU(2) — the spin-adapted twin on the singlet manifold, at its own reduced-multiplet cap
//!     (not comparable to the abelian `M`, see below), whose gaps must line up with the abelian
//!     SINGLET-labeled levels — a cross-layer check.
//!
//! Every arm here reads out ONE symmetry channel — the one the reference lives in. The levels
//! that no budget `k` reaches because they carry a different total irrep or a different spin are
//! the subject of step-11 (the per-channel composition driver).
//!
//! # Builds on
//!
//! step-08
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-10                       # both fixtures, defaults
//! cargo run --release --example step-10 -- testdata/lih_sto3g m=2,3 k=4,8 nosu2
//! cargo run --release --example step-10 -- testdata/h2o_sto3g m= mref=4 msu2=8
//! ```
//!
//! - `fixture …` — one or more fixture directories under `testdata/`
//!   (default `testdata/lih_sto3g testdata/h2o_sto3g`)
//! - `m=<usize>[,…]` — abelian per-sector ranks to scan, empty skips the scan (default `2,3`)
//! - `k=<usize>[,…]` — normal-direction budgets for the cone arms (default `4,8,16`)
//! - `mref=<usize>` — rank of the reference-comparison arm (default: last M + 1)
//! - `msu2=<usize>` — SU(2) reduced multiplet rank cap (default: last M)
//! - `conv=<float>` — per-target retirement threshold, `0` disables
//!   (default: [`ConeOpts::conv_tol`])
//! - `nosu2` — skip the SU(2) arm (default off)
//! - `matfree` — rerun the last cone arm matrix-free (default off)
//! - `--format text|tsv` (default `text`), `--help`
//!
//! `m=` (empty) skips the abelian rank scan; `mref=` sets the reference-comparison arm's rank
//! (default: last M + 1); `msu2=` sets the SU(2) arm's REDUCED multiplet rank cap (default:
//! last M). Note the SU(2) reduced cap is not comparable to the abelian per-sector M — at H2O the
//! singlet manifold needs `msu2=8` for a converged reference; a smaller cap stays in the
//! determinant trap even with the overshooting MALS ramp the arm uses. `conv=` overrides
//! [`ConeOpts::conv_tol`], the per-target retirement threshold, on every cone arm (`conv=0`
//! disables it — the greedy-to-closure schedule). The threshold is fixture-scale, not universal:
//! the residual that corresponds to a given gap accuracy moves with the size of the gaps.
//!
//! # Results
//!
//! Each fixture prints its exact configuration-enumeration reference (its dimension, the partner
//! sector's dimension, and `E0`), then one block per per-sector rank `M`: the ground-state energy and
//! gradient norm, one summary line per cone budget `k`, and a table of the first six excitation
//! gaps — exact value, spin label, then the tangent-CI, tangent-Williamson and cone-`k`
//! readouts, each with its absolute error against the exact gap.
//!
//! Cone arms also report the CHARACTER fractions: `gs r/t/c` on the per-`k` summary line is
//! the relaxed ground state's split over the mixed basis's reference / tangent / cone blocks, and
//! `c=` in each cone cell is that state's cone fraction. A cone-dominated level is one no tangent
//! construction at this reference could have produced.
//!
//! On LiH (dim 225, `E0 = -8.877719570`) the demonstration lands hardest. At `M = 2` the two
//! lowest gaps are already accurate to 1e-4 or better in every arm, but levels 3 and 4 — the
//! degenerate `T+` pair at 0.16595 — are missed by 2e-1 and 4e-1 alike by tangent-CI, by
//! tangent-Williamson, and by the cone at `k = 4` and `k = 8`; then `k = 16` finds them at
//! 0.16610 (2e-4) with cone fraction `c = 0.95`, i.e. almost entirely outside the tangent space.
//! At `M = 3` the same pair reaches 1e-7. Spending the rank on the REFERENCE instead does not buy
//! them: the `M = 4` tangent-CI arm still errs by 2e-1 and 4e-1 on `j = 3…6`.
//!
//! On H2O (dim 441, `E0 = -83.821320365`) nothing in the first six levels is tangent-invisible —
//! every cone fraction stays at 0.00–0.01 — and the budget instead tightens errors already in
//! reach: at `M = 2` the worst tangent-CI error is 1e-2 and the worst cone `k = 16` error 1e-3.
//!
//! The SU(2) arm runs at the default `msu2` = last `M` = 3. On LiH that reference sits 1.92e-4
//! above `E0` and its `k = 16` singlet gaps match the abelian singlet-labeled levels to a worst
//! 1.6e-2. On H2O the same cap sits 1.66e-2 above `E0` and its gaps are off by 7e-1 to 8e-1:
//! that is the determinant trap the `msu2=8` note above describes, visible in the default run.
//!
//! # Possibilities for extensions
//!
//! - Add `msu2=8` on H2O and re-read the cross-layer line: the SU(2) reference leaves the
//!   determinant trap, and its singlet gaps become comparable to the abelian singlet levels.
//! - Extend the budget (`k=4,8,16,32`) on LiH at `M = 3` and watch whether levels 5 and 6 — the
//!   singlet pair at 0.18545, still 2e-1/4e-1 out at `k = 16` — come into reach.
//! - Pass `matfree` to rerun the largest cone arm through the matrix-free eigensolver; it prints
//!   the agreement with the dense readout and its iteration counts.

use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::{IntegralsOptions, load_fixture_dir};
use qntt::round::RoundMode;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::hess::{ConeOpts, SymplecticSpectrum, cone_response};
use qntt::solvers::mals::NoiseType;
use qntt::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg};
use qntt::{QnSpace, RankGrid, SparseHamiltonian, TtVector, dot, tt_state};
use qntt_exkit::args::{Def, Val};
use qntt_exkit::grad::abelian::norm_projected as grad_norm_at;
use qntt_exkit::grad::su2::norm_weingarten as su2_grad_norm_at;
use qntt_exkit::out::banner;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::time::Instant;

// --- exact reference: configuration enumeration + sector-degeneracy spin labels ---------------

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

/// Dense symmetric eigenvalues via cyclic Jacobi (the test-local oracle, faer-free).
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

// --- abelian ground state at per-sector rank M ------------------------------------------------

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

// --- SU(2) arm ---------------------------------------------------------------------------------

fn su2_ground_state(
    space: qntt::su2::qn::Su2Space,
    ham: &qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    m: usize,
) -> qntt::su2::vector::Su2TtVector<f64> {
    use qntt::su2::core::Family;
    use qntt::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg};
    use qntt::su2::solvers::mals::{MalsOptions, mals};
    use qntt::su2::vector::tt_csf;

    let nocc = space.n / 2;
    let codes: Vec<Family> = (0..space.d)
        .map(|i| if i < nocc { Family::Occ } else { Family::Empty })
        .collect();
    let mut x0: qntt::su2::vector::Su2TtVector<f64> = tt_csf(&codes);
    x0.right_orthogonalize(true);
    // The ramp transiently OVERSHOOTS the target rank before capping at m: a never-exceeding
    // ramp leaves MALS in the determinant trap at H2O scale, at a vanishing gradient.
    let mopts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![
            (m / 4).max(6),
            (m / 2).max(10),
            (3 * m / 4).max(14),
            m,
            m,
            m,
        ],
        noise: vec![
            3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        seed: 0x9E37_79B9,
        // Random rather than the crate-wide Perturbative default.
        noise_type: NoiseType::Random,
        max_iter: 30,
        ..Default::default()
    };
    let (_lmals, xa) = mals(ham, &x0, &mopts);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let grad_norm = |y: &qntt::su2::vector::Su2TtVector<f64>| -> f64 { su2_grad_norm_at(ham, y) };
    let (_l, mut x) = riemannian_cg(ham, &xa, &opts);
    for _ in 0..30 {
        if grad_norm(&x) < 1e-10 {
            break;
        }
        let (_l2, x2) = riemannian_cg(ham, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    x.check();
    x
}

// --- driver ------------------------------------------------------------------------------------

fn fmt_gap(g: Option<f64>, exact: f64) -> String {
    match g {
        Some(v) => {
            let err = (v - exact).abs();
            format!("{v:>9.5} ({err:.0e})")
        }
        None => format!("{:>9} {:>7}", "—", ""),
    }
}

/// The SPUR marker threshold: a state whose CONE fraction exceeds this is one the tangent space
/// at this reference structurally cannot represent. A printed marker only.
const SPUR_CONE_FRACTION: f64 = 0.5;

/// A cone-arm cell: gap, absolute error, and the state's CONE fraction from
/// [`qntt::solvers::hess::ConeResponse::character`] — the block of the mixed basis the Ritz
/// vector actually lives in. `character[j + 1]` is the row paired with `gaps[j]`.
fn fmt_gap_cone(res: &qntt::solvers::hess::ConeResponse, j: usize, exact: f64) -> String {
    let Some(v) = res.gaps.get(j).copied() else {
        return format!("{:>9} {:>7} {:>7}", "—", "", "");
    };
    let err = (v - exact).abs();
    match res.character.get(j + 1) {
        Some(f) => {
            let mark = if f[2] > SPUR_CONE_FRACTION { "*" } else { " " };
            format!("{v:>9.5} ({err:.0e}) c={:.2}{mark}", f[2])
        }
        None => format!("{v:>9.5} ({err:.0e}) {:>7}", ""),
    }
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::too_many_arguments)]
fn run_fixture(
    dir: &str,
    ms: &[usize],
    ks: &[usize],
    do_su2: bool,
    targets: usize,
    mref: Option<usize>,
    msu2: Option<usize>,
    conv_tol: f64,
    matfree: bool,
) {
    let t_all = Instant::now();
    let load = load_fixture_dir::<f64>(std::path::Path::new(dir), &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{dir}: {e}"));
    let ints = load.ints;
    let (d, nelec) = (ints.norb, ints.nelec);
    let (nup, ndn) = (nelec / 2, nelec / 2);
    let (t, v) = (ints.t, ints.v);
    println!("\n================================================================================");
    banner(format!(
        "{dir}: d={d} spatial orbitals, N={nelec} — sector ({nup},{ndn})"
    ));

    // Exact reference with sector-degeneracy spin labels. `Median` construction (a deterministic
    // summation order).
    let space = QnSpace::new(d, nup, ndn);
    let ham = SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Median);
    let t0 = Instant::now();
    let ev = exact_config_spectrum(space, &ham);
    let sp_p = QnSpace::new(d, nup + 1, ndn - 1);
    let ham_p = SparseHamiltonian::new_with(sp_p, &t, &v, None, f64::EPSILON, Construction::Median);
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

    // SU(2) singlet reference list (for the cross-layer lines below).
    let singlet_gaps: Vec<f64> = ev[1..]
        .iter()
        .filter(|e| is_singlet(**e))
        .map(|e| e - ev[0])
        .collect();

    for &m in ms {
        let tm = Instant::now();
        let x = ground_state(space, &ham, m, 0x11A2);
        let lam = qntt::rayleigh_quotient(&ham, &x);
        let gnorm = grad_norm_at(&ham, &x);
        println!(
            "\n-- abelian M={m}: E(M) = {:.9} (E(M)−E0 = {:.2e}), ‖grad‖ = {gnorm:.1e}, {:.1}s",
            lam,
            lam - ev[0],
            tm.elapsed().as_secs_f64()
        );
        if gnorm > 1e-9 {
            println!("   WARNING: reference not fully critical — gaps below carry that bias");
        }

        // Tangent-only arms (k = 0): relaxed-reference CI + fixed-reference Williamson.
        let base = cone_response(
            &ham,
            &x,
            &ConeOpts {
                targets,
                max_added: 0,
                williamson: true,
                ..Default::default()
            },
        );
        let tw: Option<Vec<f64>> = base.williamson.as_ref().and_then(|s| match s {
            SymplecticSpectrum::Stable { omega } => Some(omega.clone()),
            SymplecticSpectrum::Resonant { .. } | SymplecticSpectrum::Unstable { .. } => None,
        });
        if tw.is_none() {
            println!("   tangent Williamson: not stable at this point (reported, not hidden)");
        }

        // Cone arms.
        let mut cones: Vec<(usize, qntt::solvers::hess::ConeResponse)> = Vec::new();
        for &k in ks {
            let tk = Instant::now();
            let res = cone_response(
                &ham,
                &x,
                &ConeOpts {
                    targets,
                    max_added: k,
                    conv_tol,
                    ..Default::default()
                },
            );
            // `gs r/t/c` is the ground row of the character fractions: the CONE entry is the
            // weight the reference-channel seed `N((H−λ)x)` carries, i.e. how much of `E_0^CI`'s
            // improvement over `λ` the mixed space actually bought. A vanishing `c` with a
            // stubborn `E0^CI−E0` is the E0-pinning failure mode.
            let gs = res.character.first().copied().unwrap_or([f64::NAN; 3]);
            println!(
                "   cone k={k:<2}: added {} in {} rounds, E0^CI−E0 = {:+.2e}, gs r/t/c={:.6}/{:.1e}/{:.1e}, guards iso {:.0e} / ortho {:.0e}, {:.1}s",
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
            cones.push((k, res));
        }

        // Matfree arm (opt-in): rerun the LAST cone arm through the matrix-free eigensolver and
        // report its agreement with the dense readout. Not a table column — the two routes give
        // the same readouts.
        if matfree && let Some(&(k_last, ref dense_last)) = cones.last() {
            let tk = Instant::now();
            let (res, st) = qntt::solvers::hess::cone_response_matfree_with_stats(
                &ham,
                &x,
                &ConeOpts {
                    targets,
                    max_added: k_last,
                    conv_tol,
                    ..Default::default()
                },
                &qntt::solvers::hess::MatfreeOpts::default(),
            );
            let tm = tk.elapsed().as_secs_f64();
            let dmax = dense_last
                .energies
                .iter()
                .zip(res.energies.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            println!(
                "   matfree k={k_last:<2}: added {} in {} rounds, {tm:.1}s, max|dE| vs dense {dmax:.1e} (trajectory {})",
                res.added,
                res.rounds,
                if res.added == dense_last.added && res.rounds == dense_last.rounds {
                    "identical"
                } else {
                    "DIVERGED"
                },
            );
            let secs = |d: std::time::Duration| d.as_secs_f64();
            println!(
                "     stats: {} iters ({} restarts, {} probes) vs dense-equiv {} applies; H*v {:.1}s project {:.1}s",
                st.lanczos_iters,
                st.restarts,
                st.probes,
                st.dense_equivalent_applies,
                secs(st.t_h_apply),
                secs(st.t_project),
            );
        }

        // Table: absolute gaps, (abs error vs exact) per arm. Cone arms also carry the state's
        // cone fraction `c=`; tangent-only arms have none by construction (their cone block is
        // empty).
        print!(
            "   {:>3} {:>2} {:>10}  {:>17}  {:>17}",
            "j", "S", "exact", "tangent-CI", "tangent-W"
        );
        for (k, _) in &cones {
            print!("  {:>25}", format!("cone k={k}"));
        }
        println!();
        for (j, (ge, singlet)) in exact_gaps.iter().enumerate() {
            let lbl = if *singlet { "S" } else { "T+" };
            print!(
                "   {:>3} {:>2} {:>10.5}  {:>17}  {:>17}",
                j + 1,
                lbl,
                ge,
                fmt_gap(base.gaps.get(j).copied(), *ge),
                fmt_gap(tw.as_ref().and_then(|w| w.get(j).copied()), *ge),
            );
            for (_, res) in &cones {
                print!("  {:>25}", fmt_gap_cone(res, j, *ge));
            }
            println!();
        }
        println!(
            "   legend: c = the state's CONE fraction of the mixed basis; \
             * = c > {SPUR_CONE_FRACTION} (descriptive, never gated)"
        );
    }

    // Spend-the-rank-on-the-reference arm: tangent-only at M+1 (or `mref=`), against the
    // largest cone arm. A near-rank-deficient M+1 reference can be REFUSED by the library's
    // basis-integrity guard (weakly pinned added rank → gauge-degenerate tangent frame at small
    // σ_min) — that refusal is itself a finding this arm reports, not an error.
    if let Some(m_ref) = mref.or_else(|| ms.last().map(|m| m + 1)) {
        let x1 = ground_state(space, &ham, m_ref, 0x11A2);
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let arm = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cone_response(
                &ham,
                &x1,
                &ConeOpts {
                    targets,
                    max_added: 0,
                    ..Default::default()
                },
            )
        }));
        std::panic::set_hook(prev_hook);
        match arm {
            Ok(base1) => {
                println!(
                    "\n-- abelian M={m_ref} (rank spent on the REFERENCE instead), tangent-CI gap errors:"
                );
                print!("   ");
                for (j, (ge, _)) in exact_gaps.iter().enumerate() {
                    print!(
                        " j={}: {:.0e}",
                        j + 1,
                        base1
                            .gaps
                            .get(j)
                            .map(|g| (g - ge).abs())
                            .unwrap_or(f64::NAN)
                    );
                }
                println!();
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .cloned()
                    .unwrap_or_else(|| "non-string panic payload".to_string());
                println!(
                    "\n-- abelian M={m_ref} reference arm REFUSED by the basis-integrity guard:"
                );
                println!("   {msg}");
                println!(
                    "   (the added rank is weakly pinned at this fixture — its gauge frame is \
                     degenerate, so the tangent-space readout is unreliable there; the cone \
                     augmentation at fixed M sidesteps exactly this)"
                );
            }
        }
    }

    // SU(2) arm at the largest rank cap: singlet gaps against the singlet-labeled reference.
    if do_su2 {
        let m_su2 = msu2.or_else(|| ms.last().copied()).unwrap_or(3);
        let t2 = Instant::now();
        let sspace = qntt::su2::qn::Su2Space::new(d, nelec, 0);
        let sham =
            qntt::su2::hamiltonian::Su2SparseHamiltonian::new(sspace, &t, &v, None, f64::EPSILON);
        let sx = su2_ground_state(sspace, &sham, m_su2);
        let slam = qntt::su2::hamiltonian::rayleigh_quotient(&sham, &sx);
        let sgrad = su2_grad_norm_at(&sham, &sx);
        println!(
            "\n-- SU(2) M={m_su2} (reduced multiplet ranks): E = {:.9} (E−E0 = {:.2e}), ‖grad‖ = {sgrad:.1e}, {:.1}s",
            slam,
            slam - ev[0],
            t2.elapsed().as_secs_f64()
        );
        let st = targets.min(singlet_gaps.len());
        for &k in ks {
            let tk = Instant::now();
            let res = qntt::su2::hess::cone_response(
                &sham,
                &sx,
                &ConeOpts {
                    targets: st,
                    max_added: k,
                    conv_tol,
                    ..Default::default()
                },
            );
            let worst = res
                .gaps
                .iter()
                .zip(singlet_gaps.iter())
                .map(|(g, e)| (g - e).abs())
                .fold(0.0f64, f64::max);
            let gs = res.character.first().copied().unwrap_or([f64::NAN; 3]);
            print!(
                "   cone k={k:<2} ({} added, gs r/t/c={:.6}/{:.1e}/{:.1e}, {:.1}s): singlet gaps ",
                res.added,
                gs[0],
                gs[1],
                gs[2],
                tk.elapsed().as_secs_f64()
            );
            for (j, (g, e)) in res.gaps.iter().zip(singlet_gaps.iter()).enumerate() {
                let c = res.character.get(j + 1).map_or(f64::NAN, |f| f[2]);
                print!(" {g:.5}({:.0e}) c={c:.2}", (g - e).abs());
            }
            println!("  worst {worst:.1e}");
        }
        println!("   (cross-layer: errors above are vs the ABELIAN exact singlet-labeled levels)");
    }
    println!("[{dir} total {:.1}s]", t_all.elapsed().as_secs_f64());
}

fn main() {
    let m = qntt_exkit::args::Spec::new("step-10")
        .about("cone-augmented response spectra of small molecules")
        .rest(
            "fixture",
            "fixture directories under testdata/",
            Val::Path,
            Def::List(&["testdata/lih_sto3g", "testdata/h2o_sto3g"]),
        )
        .key(
            "m",
            "abelian per-sector ranks to scan (empty skips the scan)",
            Val::List(&Val::Usize),
            Def::List(&["2", "3"]),
        )
        .key(
            "k",
            "normal-direction budgets for the cone arms",
            Val::List(&Val::Usize),
            Def::List(&["4", "8", "16"]),
        )
        .key(
            "mref",
            "rank of the reference-comparison arm",
            Val::Usize,
            Def::Derived("last M + 1"),
        )
        .key(
            "msu2",
            "SU(2) reduced multiplet rank cap",
            Val::Usize,
            Def::Derived("last M"),
        )
        .key(
            "conv",
            "per-target retirement threshold (ConeOpts::conv_tol; 0 disables)",
            Val::F64,
            Def::Derived("ConeOpts::conv_tol"),
        )
        .flag("nosu2", "skip the SU(2) arm")
        .flag("matfree", "rerun the last cone arm matrix-free")
        .parse();

    let fixtures: Vec<String> = m.strs("fixture").to_vec();
    let parse_list = |key: &str| -> Vec<usize> {
        m.strs(key)
            .iter()
            .map(|s| s.parse().expect("validated at parse"))
            .collect()
    };
    let ms = parse_list("m");
    let ks = parse_list("k");
    // `m=` (empty) skips the abelian rank scan; `mref=`/`msu2=` set the reference-comparison
    // and SU(2) arm ranks explicitly (defaults: last M + 1, and last M).
    let mref = m.usize_opt("mref");
    let msu2 = m.usize_opt("msu2");
    let do_su2 = !m.flag("nosu2");
    let matfree = m.flag("matfree");
    // `conv=` overrides `ConeOpts::conv_tol` on every cone arm (`conv=0` disables it, i.e. the
    // greedy-to-closure schedule).
    let conv_tol = m.f64_or("conv", ConeOpts::default().conv_tol);
    let targets = 6usize;

    println!("Cone-augmented molecular response — arms: tangent-only CI / Williamson, cone");
    println!("k∈{ks:?} at M∈{ms:?}, exact configuration-enumeration reference with spin labels");
    // One self-contained block per fixture: integrals, exact reference, then every arm at every
    // requested `M`. Nothing is shared between fixtures — each is an independent run.
    for f in &fixtures {
        run_fixture(f, &ms, &ks, do_su2, targets, mref, msu2, conv_tol, matfree);
    }
}
