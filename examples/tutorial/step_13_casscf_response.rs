// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Linear response on a pinned CAS manifold: the CASSCF-RPA anchor, then the `ω(m_active)` ladder.
//!
//! # Introduction
//!
//! **TT-CASSCF linear response**: the CASSCF-RPA anchor at a CAS-MO fixture, then the
//! truncated-active-rank ladder `ω(m_active)` interpolating CASSCF-RPA → TT-truncated response,
//! with transition moments, oscillator strengths and the static polarizability at every rung.
//!
//! ```text
//!   load testdata/<fixture>/            CASSCF-converged MOs + fingerprinted dipole .props
//!     ├─ cas_reference_state            pinned CAS(n,m) TT at each ladder rung m_active
//!     ├─ joint_ground_state             re-converge on the rung's manifold
//!     ├─ joint_symplectic_eigenvalues   the rung's ω
//!     ├─ transition_moments             T_n from the dipoles, rotated into the joint basis
//!     └─ polarizability_static          α(0) diagonal
//! ```
//!
//! Paper comparison (arXiv:2604.13753 Table II, Hartree; matching is by value, since the
//! spectrum also carries Sz = 0 triplet components): the paper's LR column corresponds to the
//! full-rank rung; its SS/SA columns are different methods, printed for context only. The paper
//! reports no moments and no α.
//!
//! An ω is printed only when the joint SCF reports `info.converged`; a rung that does not
//! converge prints the solver's refusal and no spectrum.
//!
//! # Builds on
//!
//! step-12
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-13                               # h2o_631g_cas, rungs 1,2,4
//! cargo run --release --example step-13 -- h2o_sto3g_cas rungs=2 k=4  # the small CAS fixture
//! cargo run --release --example step-13 -- hcho_631g_cas
//! cargo run --release --example step-13 -- --help
//! ```
//!
//! * `fixture` — positional CAS fixture dir under `testdata/` (default `h2o_631g_cas`;
//!   `hcho_631g_cas`, `c2h4_631g_cas`, `h2o_sto3g_cas` also ship),
//! * `rungs=` — comma-separated truncated `m_active` values prepended to the full-rank rung
//!   (default `1,2,4`; `0` means full and is always run last),
//! * `k=` — how many ω to tabulate (default 8),
//! * `omegas=` — comma-separated frequencies (Ha): at the anchor rung, `α(ω)` by the ω-dependent
//!   `linear_response` solve AND by the sum over states `Σ_n 2ω_n T_nᵖ T_nᵠ / (ω_n² − ω²)` from
//!   the same Williamson moments, with their relative difference and the nearest pole,
//! * `omega_probe` — bare flag: the same comparison at three derived frequencies — `½ω_min`, the
//!   widest pole gap between the paper's ω₁ mode and the next bright pole, and the widest gap
//!   between two dark poles below the first bright one,
//! * `disp=` — `α(ω)` on this many evenly spaced points of `[0, 0.98·ω_b]`, `ω_b` the first pole
//!   with `f > 1e-6` (default `0`, off): a dispersion curve up to the first bright pole,
//! * `lr_tol=` — relative residual tolerance of each `α(ω)` solve (default `1e-11`).
//!
//! `--format text|tsv` and `-h`/`--help` are the shared exkit options.
//!
//! # Results
//!
//! Two header lines first: the fixture's shape (`h2o_631g_cas`: `d = 13`, `N = (5,5)`, CAS(8,6)
//! with the 1/6/6 core/active/virtual partition, `casscf_gs = -76.039557117` read from its
//! `meta.txt`) and, when the fixture matches a paper key, that system's Table II row — for water
//! `ω1 SS/SA/LR = 0.299801/0.279149/0.313274` and `ω2 = 0.377632/0.357262/0.388773`.
//!
//! Then one line per ladder rung, `m = 1, 2, 4` and finally `full`: the rung's total energy
//! `λ + e0` and its offset `Δ` from `casscf_gs`, the orbital iteration count, `n_eff` (how many
//! orbital directions the joint context kept), the normal residual `‖z‖`, the last kept orbital
//! gradient `‖g_L‖`, the first `k` symplectic eigenvalues ω, the polarizability diagonal
//! `α(xx, yy, zz)` and the rung's wall time. A rung that does not converge prints its refusal
//! instead — `DID NOT CERTIFY`, a `Resonant` line carrying the unstable-pair count and the lowest
//! ω², or an `Unstable` line carrying the reason — and no ω at all. The anchor rung then prints
//! the lowest `k` modes as `ω | f | T_x, T_y, T_z | w_orb`, and, on a paper fixture, the
//! nearest-mode deviation of each paper LR ω. `w_orb` is the orbital share of each mode: the
//! squared coefficient mass of its Williamson column on the orbital rows of the joint-orthonormal
//! basis, once for `u = X+Y` and once for `v = X−Y` — `0` is a pure CI excitation, `1` a pure
//! orbital rotation. The closing NOTE restates the scope: the full-rank rung is CASSCF
//! Riemannian RPA on this fixture, the truncated rungs interpolate toward TT-truncated response,
//! and the moment/α columns have no paper counterpart.
//!
//! The default fixture is not a quick run: allow minutes for the four rungs of `h2o_631g_cas`.
//!
//! # Possibilities for extensions
//!
//! - Run `h2o_sto3g_cas`: the same CAS machinery at STO-3G scale, small enough to watch the whole
//!   ladder finish quickly. It carries no Table II row, so the paper lines drop out.
//! - Run `hcho_631g_cas` or `c2h4_631g_cas`: both have paper rows, so the anchor rung's
//!   nearest-mode deviations are printed for them too.
//! - Change the ladder with `rungs=`: `rungs=0` runs the anchor alone, `rungs=1,2,3,4,6` resolves
//!   how ω(m_active) approaches its full-rank value.
//! - Raise `k=` to tabulate more modes, or print the off-diagonal `α` components — the row shows
//!   only `alpha[0]`, `alpha[4]`, `alpha[8]` of the 3×3 returned by `polarizability_static`.

use std::time::Instant;

use qntt_exkit::prelude::{Def, Spec, Val};

use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::{Integrals, PropertyOperator, load_property_matching};
use qntt::qn::QnSpace;
use qntt::rotation::rotate_integrals;
use qntt::solvers::hess::SymplecticSpectrum;
use qntt::solvers::joint::cas::{CasPartition, cas_pairs, cas_reference_state};
use qntt::solvers::joint::{
    DynamicResponseOpts, JointOpts, JointScfOpts, StaticResponseOpts, TransitionMomentOpts,
    joint_context, joint_ground_state, linear_response, oscillator_strengths,
    polarizability_static, transition_moments,
};
use qntt::{CoeffTensor, SparseHamiltonian};

/// Paper Table II (Hartree): (system key, [ω1, ω2] × [SS, SA, LR]).
const PAPER: &[(&str, [[f64; 3]; 2])] = &[
    (
        "h2o",
        [
            [0.299801, 0.279149, 0.313274],
            [0.377632, 0.357262, 0.388773],
        ],
    ),
    (
        "hcho",
        [
            [0.144960, 0.147317, 0.138848],
            [0.335188, 0.464643, 0.325269],
        ],
    ),
    (
        "c2h4",
        [
            [0.384667, 0.388940, 0.354318],
            [0.567533, 0.568822, 0.562656],
        ],
    ),
];

/// Oscillator strength above which a pole counts as BRIGHT for the α(ω) probe geometry. Dark
/// poles of these symmetric molecules carry roundoff-size moments (f ≲ 1e-12) in a C1 calculation.
const F_BRIGHT: f64 = 1e-6;

/// Index of the mode nearest `w`.
fn nearest(omega: &[f64], w: f64) -> usize {
    (0..omega.len())
        .min_by(|&a, &b| {
            (omega[a] - w)
                .abs()
                .partial_cmp(&(omega[b] - w).abs())
                .unwrap()
        })
        .expect("non-empty spectrum")
}

/// The consecutive pair `(n, n+1)`, `lo ≤ n < hi`, with the widest `ω[n+1] − ω[n]`.
fn widest_gap(omega: &[f64], lo: usize, hi: usize) -> (usize, usize) {
    let n = (lo..hi)
        .max_by(|&a, &b| {
            (omega[a + 1] - omega[a])
                .partial_cmp(&(omega[b + 1] - omega[b]))
                .unwrap()
        })
        .expect("lo < hi");
    (n, n + 1)
}

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

fn main() {
    let args = Spec::new("step-13")
        .about("TT-CASSCF linear response: the CASSCF-RPA anchor and the ω(m_active) ladder")
        .pos(
            "fixture",
            "CAS fixture dir under testdata/ (hcho_631g_cas, c2h4_631g_cas, h2o_sto3g_cas also ship)",
            Val::Path,
            Def::Str("h2o_631g_cas"),
        )
        .key(
            "rungs",
            "truncated m_active values prepended to the full-rank rung (0 means full, always last)",
            Val::List(&Val::Usize),
            Def::List(&["1", "2", "4"]),
        )
        .key("k", "how many ω to tabulate", Val::Usize, Def::Usize(8))
        .key(
            "omegas",
            "anchor rung: frequencies (Ha) at which to evaluate α(ω) by solve AND by SOS",
            Val::List(&Val::F64),
            Def::Optional,
        )
        .flag(
            "omega_probe",
            "anchor rung: α(ω) at ½ω_min, inside a dark gap, and between the paper ω₁ and the \
             next bright pole (pole geometry asserted first)",
        )
        .key(
            "disp",
            "anchor rung: α(ω) on this many points of [0, 0.98·ω_bright] (0 = off)",
            Val::Usize,
            Def::Usize(0),
        )
        .key(
            "lr_tol",
            "relative residual tolerance of each α(ω) solve",
            Val::F64,
            Def::F64(1e-11),
        )
        .parse();

    let fixture = args.str("fixture").to_string();
    let mut rungs: Vec<usize> = args
        .strs("rungs")
        .iter()
        .map(|s| s.parse().expect("validated at parse"))
        .collect();
    let k = args.usize("k");
    let omegas_user: Vec<f64> = args
        .strs_opt("omegas")
        .map(|v| {
            v.iter()
                .map(|s| s.parse().expect("validated at parse"))
                .collect()
        })
        .unwrap_or_default();
    let omega_probe = args.flag("omega_probe");
    let disp = args.usize("disp");
    let lr_tol = args.f64("lr_tol");
    rungs.retain(|&m| m != 0);
    rungs.push(0); // full rank, always last — the anchor rung

    // ---- the fixture: CASSCF-converged orbitals, the CAS partition, the reference energy --------
    //
    // `meta.txt` carries the partition (n_core/n_active/n_virt) and `casscf_gs`, the energy every
    // rung's total is measured against.
    let repo = qntt_exkit::repo!();
    let fx = repo.load(&fixture);
    let meta_full = fx.meta();
    let getk = |k: &str| -> usize { meta_full.keys.get(k).unwrap().parse().unwrap() };
    let part = CasPartition {
        n_core: getk("n_core"),
        n_active: getk("n_active"),
        n_virt: getk("n_virt"),
    };
    let casscf_gs: f64 = meta_full.keys.get("casscf_gs").unwrap().parse().unwrap();
    let ints = fx.ints();
    let nup = (ints.nelec + ints.ms2.unsigned_abs() as usize) / 2;
    let ndn = ints.nelec - nup;
    let space = QnSpace::new(ints.norb, nup, ndn);
    let d = space.d;

    println!(
        "casscf_response — {fixture}: d = {d}, N = ({nup},{ndn}), CAS({},{}) partition \
         {}/{}/{}, casscf_gs = {casscf_gs:.9}",
        getk("cas_nelec"),
        getk("cas_norb"),
        part.n_core,
        part.n_active,
        part.n_virt
    );
    let paper = PAPER
        .iter()
        .find(|(m, _)| fixture.starts_with(&format!("{m}_631g")))
        .map(|(_, t)| *t);
    if let Some(t) = paper {
        println!(
            "paper Table II (Ha)   :  ω1 SS/SA/LR = {:.6}/{:.6}/{:.6}   ω2 = {:.6}/{:.6}/{:.6}",
            t[0][0], t[0][1], t[0][2], t[1][0], t[1][1], t[1][2]
        );
    }

    // Disk dipoles, fingerprint-checked against THIS meta.
    let props_raw: Vec<PropertyOperator<f64>> = ["x", "y", "z"]
        .iter()
        .map(|c| {
            load_property_matching::<f64>(&fx.dir.join(format!("dipole_{c}.prop")), d, meta_full)
                .expect("fingerprinted dipole")
        })
        .collect();

    // ---- the joint SCF settings shared by every rung --------------------------------------------
    //
    // `pairs` restricts the orbital rotations to the non-redundant CAS pairs, and
    // `tt_preserve_grid` keeps the pinned sector grid through the TT leg — a pinned CAS state must
    // come back on the manifold it was pinned to.
    let scf_opts = JointScfOpts {
        pairs: Some(cas_pairs(&part, d)),
        orb_max_iter: 40,
        orb_grad: 1e-6,
        orb_grad_full: Some(1e-6),
        tt_grad_tol: 1e-6,
        orb_floor: 1e-8,
        construction: Construction::Median,
        tt_preserve_grid: true,
        ..JointScfOpts::default()
    };

    println!(
        "\n  rung      λ+e0 (Ha)     Δ vs CASSCF   it  n_eff   ‖z‖       ‖g_L‖     ω[0..{k}]…  \
         α(xx,yy,zz)"
    );
    // ---- the ladder: one rung per m_active, the full-rank anchor last ---------------------------
    for &m in &rungs {
        let t0 = Instant::now();
        let ham = SparseHamiltonian::new_with(
            space,
            &ints.t,
            &ints.v,
            None,
            f64::EPSILON,
            Construction::Median,
        );
        let (_l0, x0) = cas_reference_state(&ham, &space, &part, m, 0xCA5F0 + m as u64);
        let out = joint_ground_state(ints, space, &x0, &scf_opts);
        let rung_label = if m == 0 {
            "full".to_string()
        } else {
            format!("m={m}")
        };
        // An ω is printed only under `info.converged`. A rung that does not converge prints its
        // refusal and no spectrum at all.
        if !out.info.converged {
            println!(
                "  {rung_label:>5}  DID NOT CERTIFY — no ω published. Refusal: {:?}",
                out.info.refused_reason
            );
            continue;
        }
        let total = out.lambda + ints.e0;
        let jctx = match joint_context(
            &out.h,
            &out.x,
            &JointOpts {
                pairs: Some(cas_pairs(&part, d)),
                ..JointOpts::default()
            },
        ) {
            Ok(c) => c,
            Err(e) => {
                println!("  {rung_label:>5}  context refused: {e}");
                continue;
            }
        };
        let zn = jctx.normal_residual().norm();
        let gl = out.info.g_orb_kept.last().copied().unwrap_or(f64::NAN);
        let (basis, rep) = jctx.joint_basis();
        // A resonant or an unstable spectrum carries no real ω, so those arms report what they saw
        // and the ladder moves on.
        let omega = match jctx.joint_symplectic_eigenvalues(&basis) {
            SymplecticSpectrum::Stable { omega } => omega,
            SymplecticSpectrum::Resonant { signed } => {
                println!(
                    "  {rung_label:>5}  Resonant spectrum: {} unstable pair(s), metric {:?}, \
                     lowest ω² = {:.3e} — no real ω published",
                    signed.n_neg,
                    signed.metric,
                    signed.omega2.first().copied().unwrap_or(0.0)
                );
                continue;
            }
            SymplecticSpectrum::Unstable { reason, .. } => {
                println!("  {rung_label:>5}  Unstable spectrum: {reason:?} — no ω published");
                continue;
            }
        };
        // Everything downstream of `joint_ground_state` lives in the ROTATED basis, so the disk
        // dipoles are rotated by `out.u` before they reach any response quantity.
        let props: Vec<PropertyOperator<f64>> = props_raw
            .iter()
            .map(|p| rotate_property(p, &out.u, d))
            .collect();
        // Orbital-criticality tolerance. The SCF above stops at `orb_grad = 1e-6`, but the
        // response entry points re-check the orbital gradient at the endpoint, and the value they
        // read can differ slightly between CPU architectures. A looser 3e-5 keeps a roundoff-level
        // excess from refusing an otherwise converged point.
        let gate = |o: f64| (o, Some(o));
        let (orb_grad_tol, orb_grad_tol_full) = gate(3e-5);
        let alpha = polarizability_static(
            &jctx,
            &props,
            &StaticResponseOpts {
                orb_grad_tol,
                orb_grad_tol_full,
                ..StaticResponseOpts::default()
            },
        );
        let adiag = alpha
            .as_ref()
            .map(|a| (a.alpha[0], a.alpha[4], a.alpha[8]))
            .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
        let wk: Vec<String> = omega.iter().take(k).map(|w| format!("{w:.5}")).collect();
        println!(
            "  {rung_label:>5}  {total:.9}  {:+.3e}  {:>2}  {:>4}  {zn:.3e}  {gl:.3e}  [{}]  \
             ({:.4}, {:.4}, {:.4})   [{:.3} ms]",
            total - casscf_gs,
            out.info.iters,
            jctx.n_eff(),
            wk.join(", "),
            adiag.0,
            adiag.1,
            adiag.2,
            t0.elapsed().as_secs_f64() * 1e3
        );
        if m == 0 {
            // The anchor rung: moments + oscillator strengths from the disk dipoles.
            let tm = transition_moments(
                &jctx,
                &basis,
                &props,
                &TransitionMomentOpts {
                    orb_grad_tol,
                    orb_grad_tol_full,
                    ..TransitionMomentOpts::default()
                },
            )
            .expect("moments at the anchor rung");
            let f = oscillator_strengths(&tm.omega, &tm.moments);
            // ---- orbital-vs-CI character of each mode ----------------------------------------
            //
            // `joint_basis()` puts the `n_tt` pure-tangent (CI) vectors FIRST and the orbital
            // seeds after, and it is joint-orthonormal, so the squared coefficient mass of a
            // Williamson column on rows `n_tt..` is a true fraction in [0, 1]. `s_r` columns are
            // `u = X+Y`, `s_i` columns `v = X−Y`; the two fractions differ when de-excitation
            // amplitudes are not small, so both are printed. The ω agreement with `tm.omega` is
            // printed as a consistency check.
            let t_w = Instant::now();
            let wl = jctx
                .joint_williamson(&basis)
                .expect("the anchor rung's Williamson basis (transition_moments just built it)");
            let (nm, nr, n_tt) = (wl.omega.len(), wl.n, rep.n_tt);
            assert_eq!(
                nm,
                tm.omega.len(),
                "Williamson mode count ≠ moments' mode count"
            );
            let orb_frac = |s: &[f64], j: usize| -> f64 {
                let (mut tot, mut orb) = (0.0f64, 0.0f64);
                for i in 0..nr {
                    let v = s[i * nm + j];
                    tot += v * v;
                    if i >= n_tt {
                        orb += v * v;
                    }
                }
                orb / tot
            };
            let w_r: Vec<f64> = (0..nm).map(|j| orb_frac(&wl.s_r, j)).collect();
            let w_i: Vec<f64> = (0..nm).map(|j| orb_frac(&wl.s_i, j)).collect();
            let dw = wl
                .omega
                .iter()
                .zip(tm.omega.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            println!(
                "\n  anchor rung: {nm} modes = {n_tt} CI-tangent + {} orbital (n_eff = {}); \
                 max|ω(williamson) − ω(moments)| = {dw:.2e}; orth_error = {:.2e}; \
                 completeness_defect = ({:.2e}, {:.2e}, {:.2e}); w_orb ∈ [{:.3e}, {:.6}] \
                 [{:.1} s]",
                nr - n_tt,
                jctx.n_eff(),
                wl.orth_error,
                tm.completeness_defect[0],
                tm.completeness_defect[1],
                tm.completeness_defect[2],
                w_r.iter()
                    .chain(w_i.iter())
                    .copied()
                    .fold(f64::INFINITY, f64::min),
                w_r.iter()
                    .chain(w_i.iter())
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max),
                t_w.elapsed().as_secs_f64()
            );
            println!(
                "  anchor rung, lowest {k} modes (n | ω | f | T_x, T_y, T_z | w_orb(u=X+Y), \
                 w_orb(v=X−Y)):"
            );
            for (n, (w, fv)) in tm.omega.iter().zip(f.iter()).take(k).enumerate() {
                println!(
                    "    [{n:>3}] ω = {w:.6}  f = {fv:.4e}  T = ({:+.4e}, {:+.4e}, {:+.4e})  \
                     w_orb = ({:.4}, {:.4})",
                    tm.moments[0][n], tm.moments[1][n], tm.moments[2][n], w_r[n], w_i[n]
                );
            }
            if let Some(t) = paper {
                for (j, row) in t.iter().enumerate() {
                    let lr = row[2];
                    let dev = omega
                        .iter()
                        .map(|w| (w - lr).abs())
                        .fold(f64::INFINITY, f64::min);
                    println!(
                        "  paper ω{} LR = {lr:.6}: nearest-mode dev = {dev:.2e} (SS {:.6}, SA \
                         {:.6} printed for context)",
                        j + 1,
                        row[0],
                        row[1]
                    );
                }
            }

            // ---- α(ω): the ω-dependent solve against the sum over states ---------------------
            //
            // SOS(ω)_pq = Σ_n 2 ω_n T_n^p T_n^q / (ω_n² − ω²) over the FULL Williamson spectrum
            // (n_neg = 0 on this path), and `linear_response` solves K(ω)·[u;v] = [g;0] at the
            // same context. Both are exact over the same basis, so they agree to the solve
            // tolerance; at ω = 0 the SOS is `tm.alpha_sos`.
            let bright = |n: usize| f[n] > F_BRIGHT;
            let mut probes: Vec<(String, f64)> = omegas_user
                .iter()
                .map(|&w| ("user".to_string(), w))
                .collect();
            let nb = (0..nm).find(|&n| bright(n));
            if omega_probe {
                probes.push(("½ω_min".to_string(), 0.5 * tm.omega[0]));
                // Between the paper's ω₁ mode and the next bright pole above it: the midpoint of
                // the widest pole gap inside that interval.
                if let Some(t) = paper {
                    let i1 = nearest(&tm.omega, t[0][2]);
                    match (i1 + 1..nm).find(|&n| bright(n)) {
                        Some(j) => {
                            let (lo, hi) = widest_gap(&tm.omega, i1, j);
                            let mid = 0.5 * (tm.omega[lo] + tm.omega[hi]);
                            // Check the pole geometry before the frequency is used.
                            assert!(
                                tm.omega[i1] < mid && mid < tm.omega[j] && hi == lo + 1,
                                "pole geometry: ω[{i1}] = {} < {mid} < ω[{j}] = {} violated",
                                tm.omega[i1],
                                tm.omega[j]
                            );
                            println!(
                                "  probe geometry: paper ω₁ ↔ ω[{i1}] = {:.6} (f = {:.2e}); next \
                                 bright pole ω[{j}] = {:.6} (f = {:.2e}); widest gap inside is \
                                 (ω[{lo}], ω[{hi}]) = ({:.6}, {:.6}) → ω = {mid:.6}",
                                tm.omega[i1], f[i1], tm.omega[j], f[j], tm.omega[lo], tm.omega[hi]
                            );
                            probes.push((format!("(ω[{i1}],ω[{j}])"), mid));
                        }
                        None => println!("  probe geometry: no bright pole above paper ω₁"),
                    }
                }
                // A dark gap: the widest gap between two consecutive DARK poles at or below the
                // first bright pole (or below ω[16] when nothing is bright).
                let top = nb.unwrap_or(16.min(nm.saturating_sub(1)));
                let dark_gap =
                    (0..top)
                        .filter(|&n| !bright(n) && !bright(n + 1))
                        .max_by(|&a, &b| {
                            (tm.omega[a + 1] - tm.omega[a])
                                .partial_cmp(&(tm.omega[b + 1] - tm.omega[b]))
                                .unwrap()
                        });
                match dark_gap {
                    Some(n) => {
                        let mid = 0.5 * (tm.omega[n] + tm.omega[n + 1]);
                        println!(
                            "  probe geometry: dark gap (ω[{n}], ω[{}]) = ({:.6}, {:.6}), f = \
                             ({:.2e}, {:.2e}) → ω = {mid:.6}",
                            n + 1,
                            tm.omega[n],
                            tm.omega[n + 1],
                            f[n],
                            f[n + 1]
                        );
                        probes.push((format!("dark(ω[{n}],ω[{}])", n + 1), mid));
                    }
                    None => println!(
                        "  probe geometry: no gap between two dark poles below the first bright \
                         pole"
                    ),
                }
            }
            if disp > 0 {
                match nb {
                    Some(b) => {
                        let top = 0.98 * tm.omega[b];
                        println!(
                            "  dispersion grid: {disp} points on [0, 0.98·ω[{b}]] = [0, {top:.6}] \
                             (ω[{b}] = {:.6} is the first pole with f > {F_BRIGHT:.0e}, f = {:.3e})",
                            tm.omega[b], f[b]
                        );
                        let den = (disp.max(2) - 1) as f64;
                        for i in 0..disp {
                            probes.push((format!("disp{i}"), top * i as f64 / den));
                        }
                    }
                    None => println!("  dispersion grid: no bright pole, skipped"),
                }
            }
            if !probes.is_empty() {
                let dyn_opts = DynamicResponseOpts {
                    tol: lr_tol,
                    max_iter: 4 * nm.max(500),
                    orb_grad_tol,
                    orb_grad_tol_full,
                    spectrum: Some(tm.omega.clone()),
                    ..DynamicResponseOpts::default()
                };
                println!(
                    "\n  α(ω) — solve vs SOS (lr_tol = {lr_tol:.0e}; nearest pole printed; rel = \
                     max_pq|solve−SOS| / max_pq|SOS|):"
                );
                println!(
                    "    {:<22} {:>10}  {:>14} {:>14} {:>14}   {:>9}  {:>18}  {:>6} {:>9}  wall",
                    "tag", "ω", "α_xx", "α_yy", "α_zz", "rel", "nearest pole", "iters", "max res"
                );
                for (tag, w) in &probes {
                    let t_p = Instant::now();
                    let sos: Vec<f64> = (0..9)
                        .map(|pq| {
                            let (p, q) = (pq / 3, pq % 3);
                            (0..nm)
                                .map(|n| {
                                    let wn = tm.omega[n];
                                    2.0 * wn * tm.moments[p][n] * tm.moments[q][n]
                                        / (wn * wn - w * w)
                                })
                                .sum()
                        })
                        .collect();
                    let np = nearest(&tm.omega, *w);
                    let pole = format!("ω[{np}]={:.5}", tm.omega[np]);
                    match linear_response(&jctx, &props, *w, &dyn_opts) {
                        Ok(r) => {
                            let scale = sos.iter().fold(0.0f64, |a, v| a.max(v.abs()));
                            let rel = r
                                .alpha
                                .iter()
                                .zip(sos.iter())
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0f64, f64::max)
                                / scale;
                            let iters = r.solves.iter().map(|s| s.iters).max().unwrap_or(0);
                            let res = r
                                .solves
                                .iter()
                                .map(|s| s.rel_residual)
                                .fold(0.0f64, f64::max);
                            println!(
                                "    {tag:<22} {w:>10.6}  {:>14.8} {:>14.8} {:>14.8}   \
                                 {rel:>9.2e}  {pole:>18}  {iters:>6} {res:>9.2e}  {:.1} s",
                                r.alpha[0],
                                r.alpha[4],
                                r.alpha[8],
                                t_p.elapsed().as_secs_f64()
                            );
                            println!(
                                "    {:<22} {:>10}  {:>14.8} {:>14.8} {:>14.8}   (SOS; solver \
                                 {:?})",
                                "", "", sos[0], sos[4], sos[8], r.solves[0].solver_used
                            );
                        }
                        Err(e) => println!(
                            "    {tag:<22} {w:>10.6}  solve REFUSED: {e}   SOS = ({:.8}, {:.8}, \
                             {:.8})  {pole}",
                            sos[0], sos[4], sos[8]
                        ),
                    }
                }
                if let Ok(a) = alpha.as_ref() {
                    println!(
                        "  static α (polarizability_static) = ({:.10}, {:.10}, {:.10}) — the ω = 0 \
                         row must reproduce it",
                        a.alpha[0], a.alpha[4], a.alpha[8]
                    );
                }
            }
        }
    }
    println!(
        "\nNOTE: the full-rank rung IS CASSCF Riemannian RPA on this fixture; the \
         truncated rungs are the interpolation toward TT-truncated response that neither method \
         has alone. Moments/α columns have no paper counterpart."
    );
}
