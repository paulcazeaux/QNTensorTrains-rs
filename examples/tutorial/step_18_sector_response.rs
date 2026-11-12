// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Cross-sector response: linear response INTO another point-group irrep from one typed reference
//!
//! # Introduction
//!
//! step-11 taught the walls: a typed fixed-rank reference is irrep-CLOSED, so a `B1` excitation of
//! an `A1` state is not "hard to reach at a larger budget" — it is structurally absent from the
//! response space, and the remedy there was to re-converge a separate ground state per channel.
//! This step goes through the wall instead, without re-converging anything.
//!
//! Embed the typed state in the point-group-UNTYPED fixed-rank manifold (the same cores, bond ranks
//! pooled over the irreps). The embedded point is interior, its tangent space splits into one block
//! per irrep flux `φ`, and the Riemannian Hessian is block-diagonal across those blocks — so each
//! other irrep is a separate, smaller response problem, evaluated at typed cost. The reference is a
//! CRITICAL point of that untyped problem even though nothing was re-converged: the residual of a
//! typed state lives in the reference's own irrep, hence is exactly orthogonal to every other
//! channel. That orthogonality is the whole licence for a Williamson reduction at a fixed
//! reference.
//!
//! Two things the tables below are here to teach, and both are properties of the GROUP and the rank
//! grid rather than of the rank alone:
//!
//! - `dim T^φ` is NOT the size of the `Γ0 ⊕ φ` sector. The shifted tangent space borrows the
//!   reference's suffix bases at the relabelled sectors, so it is generally a PROPER subspace of
//!   that sector and the reported values are then variational upper bounds on the channel's levels.
//!   The step prints `dim T^φ` beside the target irrep's own determinant count, which the library
//!   does not compute (it is exponential in `d`).
//! - a channel can be EMPTY, and that is an answer. On LiH at the default reference the `A2` flux
//!   has `dim T^φ = 0` while its sector holds 44 determinants and step-11's driver converges a
//!   state there, so the two drivers need not reach the same set of irreps.
//!
//! Arms:
//!   - the reference: one typed fixed-rank ground state (MALS ramp → Riemannian-CG polish, the
//!     library's `channel_ground_state` recipe), with `λ`, `‖P_T grad‖`, `σ_min` and its pooled bond
//!     dimensions. The polish is what makes this a response point at all;
//!   - the channel table: one row per irrep flux — `dim T^φ`, the target sector's determinant
//!     count, their ratio, `‖B^φ‖/‖A^φ‖` with the tag saying how it was scored, the channel status
//!     and the per-channel wall time;
//!   - both readouts side by side per live channel: TDA `θ = eig(A^φ)`, an ENERGY relative to `λ`,
//!     and RPA signed `ω²` from the pair `(A^φ, B^φ)`. These are two independent solves in two
//!     different units, not one number read twice. For a `Z₂ⁿ` group `Γ ⊗ Γ` is the identity, so a
//!     flux block is ALLOWED a nonzero Weingarten term at every truncated rank — and it still
//!     vanishes identically where the channel varies one single core slot, because the manifold
//!     then contains a straight line through the reference. The step prints `θ² − ω²` beside
//!     `‖B^φ‖/‖A^φ‖` and says which case each channel is in. A real `ω` is published only where
//!     every `ω²` is positive with no negative witness outstanding; a negative value is a lower
//!     state in that channel, not a solver failure;
//!   - the in-sector leg, the in-sector dense driver at the same reference, so the reader sees the
//!     complete labelled spectrum assembled from the reference channel plus the shifted ones;
//!   - `channel_response` beside it — the relaxed-reference gaps of step-11. DIFFERENT OBJECTS: one
//!     re-converges a fixed-rank ground state per irrep and differences two independently
//!     truncated variational energies, the other linearizes at one reference. Printed, bounded,
//!     never asserted equal;
//!   - SYNTHETIC observables per channel: one symmetry-typed one-body operator per irrep flux, built
//!     from the fixture's ORBSYM — **NOT this molecule's dipole** (no symmetric fixture ships dipole
//!     matrices, and which irrep `x`, `y`, `z` carry depends on the generator's orientation) — routed
//!     by the driver to the ONE channel its derived flux names. Per live channel: the static `α^φ`,
//!     the lowest `|T_n|` and `f_n`, the operator sum rule `Σ_n T_n²` against `‖P^φ(μ̂x)‖²` (exact
//!     where `B^φ ≡ 0`, a defect of EITHER sign on a curved channel, never asserted zero) and the two
//!     defects — `capture` (what the channel's borrowed-suffix tangent space misses of `μ̂x`, `O(1)`
//!     on a proper-subspace channel) and `completeness` (what the channel basis misses of the
//!     covector);
//!   - the dense cross-check, last: the dense `symplectic_eigenvalues` over the whole shifted
//!     basis against the iterative `ω²`, the untyped-embedding check (the UNTYPED `hessian_grams`
//!     cut into irrep blocks by `project_irrep`) for one channel, and the
//!     structural identity `Σ_φ dim T^φ + tangent_dim(x) = tangent_dim(forget_point_group(x))`,
//!     which is this step's one exact assertion.
//!
//! **Screening.** The fixture is loaded with `PointGroupPolicy::Resolve(None)`, whose `screen: true`
//! default runs `fcidump::screen_point_group_forbidden` before the typed space is built: a symmetric
//! dump carries ~1e-14 symmetry-forbidden residue that the Hamiltonian construction rejects.
//! `sector_response_all_from_load` re-screens on its own copy, which is what makes it the
//! recommended entry — it builds the reference operator and every channel operator from the same
//! screened integrals.
//!
//! # Builds on
//!
//! step-08, step-11
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-18                              # lih_sto3g, rank 8, both readouts
//! cargo run --release --example step-18 -- lih_sto3g --rank 16        # dim T^φ climbs toward the sector size
//! cargo run --release --example step-18 -- lih_sto3g --readout tda    # the A-only readout alone
//! cargo run --release --example step-18 -- h2o_sto3g --rank 8 --k 1   # a curved channel: the readouts separate
//! cargo run --release --example step-18 -- --help
//! ```
//!
//! - `fixture` — subdir of `testdata/`; it must carry a nontrivial `ORBSYM` [default `lih_sto3g`].
//! - `--rank <usize>` — POOLED bond-rank cap of the typed reference, `0` = uncapped [default `8`].
//! - `--k <usize>` — roots per channel per readout [default `2`].
//! - `--readout <tda|rpa|both>` — which readout(s) to run per channel [default `both`].
//! - `--dense_cap <usize>` — run the dense cross-check on channels with `dim ≤ this`; `0`
//!   turns it off, and the whole last arm with it [default `64`].
//! - `--format <text|tsv>` — output format [default `text`].
//!
//! # Results
//!
//! The default run on `testdata/lih_sto3g` (`d = 6`, `N = 4`, sector `(2,2)`, `C2v` of order 4,
//! target `A1`): the reference lands at `λ = −8.877710643` Ha electronic (`−7.882393` Ha total)
//! with `‖P_T grad‖ ≈ 1e-12`, `‖r⊥‖ = 6.0e-3` and `σ_min = 8.1e-4`, on pooled bond dimensions
//! `[1, 4, 8, 7, 7, 4, 1]`.
//!
//! The channel table is the lesson. The `A2` flux is EMPTY — `dim T^φ = 0` over a sector of 44
//! determinants — and no operator is built for it, so `built = 2` rather than 3. `B1` and `B2` carry
//! 10 and 8 directions against sectors of 56, i.e. 18% and 14% of the target irrep, so their values
//! are variational upper bounds on those irreps' levels and not the levels themselves.
//!
//! Both live channels come out with `‖B‖_F/‖A‖_F ≈ 1e-15` and `θ² − ω² ≈ 3e-15`: the RPA readout
//! collapses onto the TDA one. Since `‖r⊥‖ = 6.0e-3` is nowhere near zero, that is not "the
//! reference is an exact eigenvector" — it is the structural reason, a channel every one of whose
//! directions varies the same single core slot, where the second fundamental form is identically
//! zero. `h2o_sto3g --rank 8` is the contrasting run: there `A2` and `B2` stay single-slot
//! (`‖B‖/‖A‖` of `0` and `3e-14`) while `B1` is curved, `‖B‖_F/‖A‖_F = 1.8e-2`, and its two
//! readouts separate — `θ = 0.557342` against `ω = 0.533030`.
//!
//! `channel_response` on the same load re-converges a state in all four irreps and reports gaps of
//! `0.166` Ha (`B1`, `B2`) and `0.633` Ha (`A2`), against linearized `θ` of `0.5765` and `0.5766` on
//! the two live channels. That is not a discrepancy to reconcile: at this cap the reference's
//! borrowed suffix bases are a poor space for the other irreps, and the two columns answer different
//! questions — one differences two independently truncated variational energies, the other
//! linearizes at one point. The in-sector leg reports its own `ω` at the same reference
//! (`0.116, 0.133, 0.399, 0.564`), and the merged list carries each level's channel label.
//!
//! The observables arm is labelled SYNTHETIC throughout — the three components are symmetry-typed
//! stand-ins built from ORBSYM, not LiH's dipole. The `A2` component is skipped with its empty
//! channel; `B1` and `B2` report `α^φ = 3.396e-1` and `3.393e-1` on their own components (every
//! other entry of the `3×3` is a structural zero, so `alpha_total`'s diagonal is
//! `[0, 3.396e-1, 3.393e-1]`), a `capture` defect of `0.961` on both — the 10- and 8-direction
//! channels miss almost all of `μ̂x` over their 56-determinant sectors, the same fact `dim/dets`
//! states — a `completeness` defect at roundoff, and the operator sum rule `Σ T²/‖p‖² = 1.000000`
//! exactly, because both channels are single-slot (`B ≡ 0`). The LOWEST mode of each channel is
//! dark to the operator (`|T_0| ~ 1e-16`), so the table quotes the brightest mode instead;
//! `Σ_channels Σ_n f_n = 0.149` is printed beside the sentence that it is not Thomas–Reiche–Kuhn.
//!
//! The last arm checks the computation rather than the physics: the dense Williamson `ω²` over the
//! whole shifted basis reproduces the iterative values to about `2e-15`, the untyped-embedding
//! check agrees with the shifted assembly to `3e-15` absolute on `A`, `A+B` and `A−B` at a scale of
//! `3.2`, and `Σ_φ dim T^φ + tangent_dim(x) = 57 = tangent_dim(forget_point_group(x))` holds exactly,
//! as an integer.
//!
//! # Possibilities for extensions
//!
//! - Raise `--rank` and watch `dim T^φ` climb toward the sector's determinant count while the
//!   linearized values come down: at `--rank 16` on this fixture `B1` reaches `dim 12`. Lower it and
//!   the channels die out one by one — at `--rank 2` every flux is EMPTY and the driver reports
//!   three empty rows and no operator built, which is the same lesson from the other side.
//! - `h2o_sto3g --rank 8` for a curved channel (`‖B‖_F/‖A‖_F = 1.8e-2` on `B1`, and the two readouts
//!   separate); `h2o_c2v` for the same molecule at a larger sector; `ch2_triplet_c2v` for an
//!   open-shell reference where the reference channel is not a singlet.
//! - `--readout rpa` on a reference that is NOT the lowest state of its own irrep (converge in a
//!   different `target`, or cap hard enough that the polish lands elsewhere): the status column then
//!   reports `Indefinite` with its witnesses, which says the reference sits above a state of
//!   that channel — criticality is not stability.
//! - Set `--dense_cap 0` to see what the iterative path alone reports, and how the `‖B‖/‖A‖`
//!   column changes from a Gram-norm ratio (`gram`) to a seed-block estimate (`seed`). The two are
//!   different numbers, which is why the tag is printed.

use qntt::coeff::ThreeBody;
use qntt::core::tangent::{
    TtTangent, min_schmidt, project_irrep, tangent_basis, tangent_basis_shifted, tangent_dim,
};
use qntt::integrals::{IntegralsOptions, PointGroupPolicy};
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::solvers::hess::{
    BWeight, ChannelGsOpts, ChannelOpts, ChannelSet, ConeOpts, IrrepSel, SectorAllOpts,
    SectorAllReadout, SectorChannel, SectorChannelStatus, SectorCtx, SectorObservables, SectorOpts,
    SectorReadout, SymplecticSpectrum, channel_ground_state, channel_response_from_load,
    hessian_grams, normal_residual, property_flux, sector_response_all_from_load,
    weingarten_operator,
};
use qntt::{CoeffTensor, PropertyOperator, QnSpace, SparseHamiltonian, TtVector};
use qntt_exkit::prelude::*;

/// Relative tolerance of the last arm's algebraic identities — the untyped-embedding check and the
/// dense-vs-iterative comparison. Both are algebra at whatever base point the polish produced, so
/// they do not depend on the trajectory; the dimension identity below is exact and integer.
const ORACLE_TOL: f64 = 1e-8;

// -------------------------------------------------------------------------------------------------
// The determinant count per irrep — the number `dim T^φ` has to be read against.
// -------------------------------------------------------------------------------------------------

/// All `n`-element subsets of `0..d`, lexicographic (step-11's helper).
fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if n > d {
        return out;
    }
    if n == 0 {
        return vec![vec![]];
    }
    let mut cur: Vec<usize> = (0..n).collect();
    loop {
        out.push(cur.clone());
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if cur[i] != i + d - n {
                cur[i] += 1;
                for j in i + 1..n {
                    cur[j] = cur[j - 1] + 1;
                }
                break;
            }
        }
    }
}

/// Determinants of `(nup, ndn)` per total irrep, by the determinant FUSE — a channel assignment
/// that owes nothing to the tensor-train representation. This is the count the library does not
/// compute for you: it is exponential in `d`, and `dim T^φ` is meaningless without it.
fn irrep_det_counts(space: &QnSpace, pg: PointGroup) -> Vec<usize> {
    let mut counts = vec![0usize; pg.order() as usize];
    let ups = combinations(space.d, space.nup);
    let dns = combinations(space.d, space.ndn);
    for up in &ups {
        for dn in &dns {
            let ir = up
                .iter()
                .chain(dn.iter())
                .fold(Irrep::IDENTITY, |acc, &s| acc.fuse(pg.irrep_of(s)));
            counts[ir.index() as usize] += 1;
        }
    }
    counts
}

// -------------------------------------------------------------------------------------------------
// Small dense helpers for the last arm.
// -------------------------------------------------------------------------------------------------

/// Ascending eigenvalues of a row-major symmetric `n × n` Gram, by cyclic Jacobi. The shifted
/// blocks at tutorial scale are tens of directions, so this is ample and keeps the arm dependency-free.
#[allow(clippy::needless_range_loop)]
fn gram_eigs(g: &[f64], n: usize) -> Vec<f64> {
    let mut a: Vec<Vec<f64>> = (0..n).map(|i| g[i * n..(i + 1) * n].to_vec()).collect();
    for _ in 0..100 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off += a[i][j] * a[i][j];
                }
            }
        }
        if off.sqrt() < 1e-13 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-300 {
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
    let mut ev: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    ev
}

/// `a ± b` of two row-major Grams.
fn gram_combine(a: &[f64], b: &[f64], sign: f64) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + sign * y).collect()
}

/// Twice-repeated modified Gram–Schmidt on a set of tangents, dropping what collapses. The
/// untyped basis is orthonormal but its irrep projections are not, so the block needs
/// re-orthonormalizing before its Gram can be compared with the shifted one.
fn mgs(cands: Vec<TtTangent<f64>>) -> Vec<TtTangent<f64>> {
    let mut out: Vec<TtTangent<f64>> = Vec::new();
    for mut u in cands {
        let n0 = u.norm();
        if n0 <= 1e-8 {
            continue;
        }
        for _ in 0..2 {
            for b in &out {
                let c = b.dot(&u);
                u.axpy(-c, b);
            }
        }
        let nrm = u.norm();
        if nrm > 1e-8 * n0 && nrm > 1e-10 {
            u.scale_mut(1.0 / nrm);
            out.push(u);
        }
    }
    out
}

/// The worst absolute difference between two lists, over the overlap.
fn worst_dev(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

// -------------------------------------------------------------------------------------------------
// SYNTHETIC symmetry-typed observables — this step's stand-in for dipole matrices, which no
// symmetric fixture ships and whose axis irreps depend on the generator's orientation.
// -------------------------------------------------------------------------------------------------

/// One SYNTHETIC one-body operator per irrep flux `φ ≠ A` that the ORBSYM table can carry:
/// `m[p,q] = m[q,p] = c_pq` on every pair `p < q` with `Γ_p ⊗ Γ_q = φ`, zero elsewhere (no
/// diagonal — `Γ_p ⊗ Γ_p = A` would make it multi-class), so the flux class DERIVED from the matrix
/// is exactly `[φ]` and the driver routes it to that one channel. The weights are a fixed
/// deterministic pattern in `±[0.5, 0.9]`, bounded away from zero so no cell of the support is
/// screened out. NOT this molecule's dipole: it has the symmetry TYPE a dipole component has, which
/// is everything the routing, the moments and the sum rule depend on. At most three (the
/// `oscillator_strengths` cap), in flux order.
fn synthetic_observables(pg: PointGroup, d: usize) -> Vec<PropertyOperator<f64>> {
    let mut out = Vec::new();
    for g in 1..pg.order() {
        let phi = Irrep::new(g);
        let mut m = CoeffTensor::<f64>::zeros(d, 2);
        let mut k = 0usize;
        for p in 0..d {
            for q in (p + 1)..d {
                if pg.irrep_of(p).fuse(pg.irrep_of(q)) == phi {
                    k += 1;
                    let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
                    let c = sign * (0.5 + 0.1 * (k % 5) as f64);
                    m.set(&[p, q], c);
                    m.set(&[q, p], c);
                }
            }
        }
        if k == 0 {
            continue;
        }
        out.push(PropertyOperator::new(format!("mu[flux {g}]"), m));
        if out.len() == 3 {
            break;
        }
    }
    out
}

// -------------------------------------------------------------------------------------------------
// Rendering
// -------------------------------------------------------------------------------------------------

/// The channel's printable irrep name, or its raw index when no group KIND was named.
fn irrep_tag(ch: &SectorChannel) -> String {
    ch.label
        .name
        .clone()
        .unwrap_or_else(|| format!("irrep {}", ch.target.index()))
}

/// `‖B^φ‖/‖A^φ‖` and how it was scored. The two spellings are DIFFERENT NUMBERS — a Frobenius norm
/// ratio off the full Grams, and a per-direction estimate on the readout's seed block — so the tag
/// travels with the value and never gets folded into one column.
fn b_weight_cells(w: Option<&BWeight>) -> (String, &'static str) {
    match w {
        None => ("—".to_string(), "—"),
        Some(BWeight::GramRatio(r)) => (format!("{r:.3e}"), "gram"),
        Some(BWeight::SeedFraction { mean, max, n }) => {
            (format!("{mean:.3e}/{max:.3e} (n={n})"), "seed")
        }
    }
}

/// One-word status of a dense Williamson verdict, for the cross-check table.
fn dense_tag(s: &SymplecticSpectrum) -> &'static str {
    match s {
        SymplecticSpectrum::Stable { .. } => "stable",
        SymplecticSpectrum::Resonant { .. } => "resonant",
        SymplecticSpectrum::Unstable { .. } => "unstable",
    }
}

fn main() {
    let m = Spec::new("step-18")
        .about(
            "cross-sector linear response: one response block per point-group irrep at a single \
             typed reference",
        )
        .pos(
            "fixture",
            "subdir of `testdata/` (must carry a nontrivial ORBSYM)",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
        .opt(
            "rank",
            "bond-rank cap of the typed reference (0 = uncapped)",
            Val::Usize,
            Def::Usize(8),
        )
        .opt(
            "k",
            "roots per channel per readout",
            Val::Usize,
            Def::Usize(2),
        )
        .opt(
            "readout",
            "which readout(s) to run per channel",
            Val::Enum(&["tda", "rpa", "both"]),
            Def::Str("both"),
        )
        .opt(
            "dense_cap",
            "certified dense oracle on channels with dim <= this (0 = off, and the last arm with it)",
            Val::Usize,
            Def::Usize(64),
        )
        .parse();
    let fixture = m.str("fixture").to_string();
    let rank = m.usize("rank");
    let k = m.usize("k").max(1);
    let readout = match m.str("readout") {
        "tda" => SectorAllReadout::Tda,
        "rpa" => SectorAllReadout::Rpa,
        _ => SectorAllReadout::Both,
    };
    let dense_cap = m.usize("dense_cap");

    // SCREENED BY CONSTRUCTION: `Resolve` plus the default `screen: true` runs
    // `screen_point_group_forbidden` before the typed space exists.
    let opts_load = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let fx = qntt_exkit::repo!().load_with(&fixture, &opts_load);
    let space = fx.space();
    let pg = fx.pg().pg;
    let kind = fx.pg().kind;
    let (d, nelec) = (fx.d(), fx.nelec());
    banner(format_args!(
        "cross-sector response — {fixture} (d={d}, N={nelec}, sector ({},{}), {kind:?} of order \
         {}, target irrep {})",
        space.nup,
        space.ndn,
        pg.order(),
        space.target.index()
    ));
    if pg.is_trivial() {
        println!(
            "this fixture resolved to the TRIVIAL point group, so there is no other symmetry \
             channel to reach and the driver would report an empty row list. Pick a fixture with a \
             nontrivial ORBSYM (`lih_sto3g`, `h2o_sto3g`, `h2o_c2v`, `ch2_triplet_c2v`)."
        );
        return;
    }
    let w: Option<&dyn ThreeBody<f64>> = fx.load.w.as_ref().map(|w| w as &dyn ThreeBody<f64>);

    // ---- the reference ------------------------------------------------------------------------
    //
    // The library's per-channel recipe, `channel_ground_state`: typed start → rank-adaptive
    // MALS ramp → Riemannian-CG polish → trim → re-polish. The polish is not decoration here — the
    // whole construction reads a Hessian at a point it assumes is critical, and `‖P_T grad‖` below
    // is the number that says whether it is.
    let ham = fx.ham();
    let gs = ChannelGsOpts {
        chi: qntt::Pooled(rank),
        ..ChannelGsOpts::default()
    };
    let ((x, lambda, gradnorm), t_ref) = time(|| channel_ground_state(&ham, space, &gs));
    let ctx = weingarten_operator(&ham, &x);
    let (base_l, base_r) = ctx.bases();
    let sigma_min = min_schmidt(base_l, base_r);
    let pooled: Vec<usize> = x.ranks.iter().map(|g| g.total()).collect();
    // `||r_perp||` earns its place on this line: the Weingarten term is LINEAR in the normal
    // residual, so it is the number that separates "B vanished because the reference is an exact
    // eigenvector" from "B vanished for the structural reason" the readouts arm names below.
    let r_perp = normal_residual(&ham, base_l, base_r, &x, lambda).norm_lr(false);
    println!(
        "reference: lambda = {lambda:.9} Ha (electronic), ||P_T grad|| = {gradnorm:.3e}, \
         ||r_perp|| = {r_perp:.3e}, sigma_min = {sigma_min:.3e}, pooled bond dims {pooled:?}, \
         {:.2} s",
        t_ref.as_secs_f64()
    );
    println!(
        "  e_core = {:.9} Ha; every energy below is ELECTRONIC unless the column says otherwise.",
        fx.e_core()
    );

    // ---- the channel table -------------------------------------------------------------------
    //
    // One driver call does every channel: it builds `H(space_out)` per flux and reads it out.
    // `include_reference` additionally runs the in-sector dense driver at the same point, which is
    // the `φ = 0` row of the labelled spectrum further down.
    //
    // The SYNTHETIC observables ride along in the same call: the driver resolves each component's
    // flux class from its matrix, routes it to that one channel and evaluates `alpha^phi` and the
    // dense moments there (the moments under the same `dense_cap` gate as the dense spectrum).
    let props = synthetic_observables(pg, d);
    let sopts = SectorAllOpts {
        irreps: IrrepSel::All,
        kind: Some(kind),
        readout,
        k,
        solver: SectorOpts::default(),
        dense_cap: (dense_cap > 0).then_some(dense_cap),
        eps_h: f64::EPSILON,
        include_reference: true,
        properties: Some(props.clone()),
        observables: SectorObservables::Both,
    };
    let (spec, t_all) = time(|| sector_response_all_from_load(&fx.load, &x, &sopts));
    let dets = irrep_det_counts(&space, pg);

    println!(
        "\n-- channels ({:.2} s total, {} operator build(s)) --",
        t_all.as_secs_f64(),
        spec.built
    );
    let mut tbl = Table::new(vec![
        Col::int("phi", 4),
        Col::text("irrep", 7).header("target"),
        Col::int("dim", 6).header("dim T^phi"),
        Col::int("dets", 7).header("sector dets"),
        Col::float("cover", 7, 3).header("dim/dets"),
        Col::text("bnorm", 24).header("||B||/||A||"),
        Col::text("how", 5),
        Col::text("status", 34),
        Col::float("secs", 7, 2).header("s"),
    ])
    .start();
    for ch in &spec.channels {
        let n_dets = dets[ch.target.index() as usize];
        let (bw, how) = b_weight_cells(ch.b_weight.as_ref());
        tbl.row(row![
            ch.phi.index(),
            irrep_tag(ch),
            ch.dim,
            n_dets,
            if n_dets == 0 {
                0.0
            } else {
                ch.dim as f64 / n_dets as f64
            },
            bw,
            how,
            format!("{:?}", ch.status),
            ch.secs
        ]);
    }
    tbl.finish();
    println!(
        "  `dim T^phi` is a STRUCTURAL count over the rank grids, not the size of the target \
         sector: the shifted space borrows the reference's suffix bases at the relabelled sectors, \
         so where dim/dets < 1 the values below are Ritz values of H|_target − lambda compressed to \
         T^phi — upper bounds by Cauchy interlacing, not that sector's levels."
    );
    for ch in spec.empty_channels() {
        println!(
            "  {} is EMPTY (dim T^phi = 0) over a sector of {} determinants — the flux is \
             unreachable from THIS reference's grid, a property of the group and the grid rather \
             than a defect, and no operator was built for it.",
            irrep_tag(ch),
            dets[ch.target.index() as usize]
        );
    }

    // ---- the two readouts, side by side -------------------------------------------------------
    //
    // TDA reports `eig(A^phi)`, an ENERGY relative to `lambda`; RPA reports the signed `omega^2` of
    // the pair `(A^phi, B^phi)`. Different quantities in different units, from two independent
    // solves off the same seed block — the entry points refuse each other's readout rather than
    // answer it under the caller's label. They separate exactly as far as `B^phi` is nonzero, and
    // they coincide on a channel that varies a single core, where the manifold contains a straight
    // line through the reference and the second fundamental form vanishes identically.
    println!("\n-- readouts per live channel (k = {k}) --");
    let mut rt = Table::new(vec![
        Col::text("irrep", 7).header("target"),
        Col::int("dim", 5),
        Col::int("j", 3),
        Col::float("theta", 14, 8).header("theta = eig(A)"),
        Col::float("etda", 16, 9).header("lambda + theta"),
        Col::float("w2", 15, 8).header("signed omega^2"),
        Col::text("omega", 14).header("omega (published)"),
        Col::text("applies", 11).header("applies t/r"),
    ])
    .start();
    for ch in &spec.channels {
        if ch.status == SectorChannelStatus::Empty {
            rt.skip(&irrep_tag(ch), "dim T^phi = 0");
            continue;
        }
        let theta = ch.tda.as_ref().and_then(|r| r.theta()).unwrap_or(&[]);
        let w2 = ch.rpa.as_ref().and_then(|r| r.omega2()).unwrap_or_default();
        // `omega()` is fallible ON PURPOSE: it hands back a real spectrum only when the RPA
        // readout is non-empty, every `omega^2` is strictly positive, and no negative witness is
        // outstanding. A resonance is `omega = i*sqrt|omega^2|`, which a `Vec<f64>` cannot carry,
        // so it is never published as a real number.
        let omega = ch.rpa.as_ref().and_then(|r| r.omega());
        let applies = format!(
            "{}/{}",
            ch.tda.as_ref().map_or(0, |r| r.applies),
            ch.rpa.as_ref().map_or(0, |r| r.applies)
        );
        for j in 0..theta.len().max(w2.len()) {
            rt.row(row![
                if j == 0 { irrep_tag(ch) } else { String::new() },
                ch.dim,
                j,
                theta
                    .get(j)
                    .map(|t| format!("{t:.8}"))
                    .unwrap_or_else(|| "—".to_string()),
                theta
                    .get(j)
                    .map(|t| format!("{:.9}", spec.reference.lambda + t))
                    .unwrap_or_else(|| "—".to_string()),
                w2.get(j)
                    .map(|z| format!("{z:.8}"))
                    .unwrap_or_else(|| "—".to_string()),
                omega
                    .as_ref()
                    .and_then(|o| o.get(j))
                    .map(|z| format!("{z:.8}"))
                    .unwrap_or_else(|| "—".to_string()),
                if j == 0 {
                    applies.clone()
                } else {
                    String::new()
                }
            ]);
        }
    }
    rt.finish();
    // Where the two readouts COINCIDE, `theta^2 == omega^2` to roundoff. That is the whole content
    // of a vanishing `B^phi`, and it has two possible causes which `||r_perp||` above tells apart:
    // `B` is linear in the normal residual, so it is zero at an exact eigenvector; and it is
    // IDENTICALLY zero, at any base point, on a channel whose directions all vary the same single
    // core slot, because the manifold then contains a straight line through the reference and the
    // second fundamental form vanishes on it. Geometry, not a defect.
    for ch in &spec.channels {
        let (Some(theta), Some(w2)) = (
            ch.tda.as_ref().and_then(|r| r.theta()),
            ch.rpa.as_ref().and_then(|r| r.omega2()),
        ) else {
            continue;
        };
        let n = theta.len().min(w2.len());
        if n == 0 {
            continue;
        }
        let dev = worst_dev(
            &theta[..n].iter().map(|t| t * t).collect::<Vec<f64>>(),
            &w2[..n],
        );
        let (bw, how) = b_weight_cells(ch.b_weight.as_ref());
        let verdict = match (dev <= 1e-10, r_perp > 1e-8) {
            (true, true) => {
                "the RPA readout has COLLAPSED onto the TDA one, and ||r_perp|| above \
                             is NOT small, so this is the structural reason: the channel varies a \
                             single core slot"
            }
            (true, false) => {
                "the RPA readout has COLLAPSED onto the TDA one; ||r_perp|| above is \
                              already at the noise floor, so an exact eigenvector explains it and \
                              the channel's curvature is untested here"
            }
            (false, _) => "the two readouts SEPARATE: B^phi is carrying curvature on this channel",
        };
        println!(
            "  {}: max |theta^2 - omega^2| = {dev:.3e}, ||B||/||A|| = {bw} ({how}) — {verdict}",
            irrep_tag(ch)
        );
    }
    for ch in &spec.channels {
        let neg = ch
            .tda
            .as_ref()
            .into_iter()
            .chain(ch.rpa.as_ref())
            .map(|r| r.negatives().len())
            .sum::<usize>();
        if neg > 0 {
            println!(
                "  {}: {neg} negative value(s) — the reference is NOT the lowest state of that \
                 irrep. Criticality is not stability; the status join carries the certification \
                 flag, and an uncertified negative is not yet a verdict.",
                irrep_tag(ch)
            );
        }
    }

    // ---- the in-sector leg --------------------------------------------------------------------
    //
    // The `phi = 0` block is a different problem (the reference Hamiltonian's own sector, with the
    // radial mode removed — a shifted channel has none) and a different result type, so the driver
    // never puts it in a channel row. `include_reference`
    // stores it beside them, which is how the complete labelled spectrum at this reference is
    // assembled: the reference channel plus the shifted ones.
    match &spec.reference_spectrum {
        Some(SymplecticSpectrum::Stable { omega }) => println!(
            "\nin-sector (phi = 0, tangent_dim {}): omega = [{}]",
            spec.reference.tangent_dim,
            omega
                .iter()
                .take(4)
                .map(|w| format!("{w:.9}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Some(other) => println!(
            "\nin-sector (phi = 0, tangent_dim {}): dense verdict {} — the reference is critical \
             but not a stable minimum of its own channel, which the shifted channels' status \
             column reports independently.",
            spec.reference.tangent_dim,
            dense_tag(other)
        ),
        None => println!("\nin-sector (phi = 0): not requested"),
    }
    // The merge is a METHOD taking the readout, never a stored field: `eig(A^phi)` is an energy and
    // `omega^2` is not, so one merged list can only ever hold one of them. Every level keeps the
    // channel that produced it, which is what makes the assembled spectrum LABELLED.
    let merged = spec.levels(SectorReadout::Tda);
    if !merged.is_empty() {
        println!("cross-sector TDA levels, ascending (channel-labelled):");
        for l in merged.iter().take(6) {
            println!(
                "  {:>16.9} Ha total   gap {:>12.9}   {} (root {})",
                l.energy + fx.e_core(),
                l.value,
                irrep_tag(&spec.channels[l.channel]),
                l.k
            );
        }
    }

    // ---- `channel_response` beside it ---------------------------------------------------------
    //
    // step-11's driver on the same load. It RE-CONVERGES a fixed-rank ground state per irrep and
    // differences two independently truncated variational energies; this step LINEARIZES at one
    // reference and reports `eig(A^phi)` there. Different objects, so the two columns are printed
    // and bounded, never asserted equal — and at a truncated reference they are far apart, because
    // the borrowed suffix bases are a poor space for the other irreps.
    let copts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::All,
            ..ChannelSet::default()
        },
        kind: Some(kind),
        gs: ChannelGsOpts {
            chi: qntt::Pooled(rank.max(4)),
            max_iter: 20,
            ..ChannelGsOpts::default()
        },
        cone: ConeOpts {
            targets: 1,
            max_added: 0,
            ..ConeOpts::default()
        },
        eps_h: f64::EPSILON,
    };
    let (cspec, t_ch) = time(|| channel_response_from_load(&fx.load, &copts));
    let e_ref = cspec.channels[cspec.reference].e_gs;
    println!(
        "\n-- relaxed-reference gaps from `channel_response` ({:.2} s) — a DIFFERENT object --",
        t_ch.as_secs_f64()
    );
    let mut ct = Table::new(vec![
        Col::text("irrep", 7).header("target"),
        Col::float("egs", 16, 9).header("E_gs (relaxed)"),
        Col::float("gap", 13, 9).header("gap vs ref"),
        Col::float("tda", 13, 9).header("lowest theta"),
        Col::text("status", 20),
    ])
    .start();
    for c in &cspec.channels {
        if c.label.spin_delta != 0 {
            continue;
        }
        let gap = match (c.e_gs, e_ref) {
            (Some(e), Some(e0)) => format!("{:.9}", e - e0),
            _ => "—".to_string(),
        };
        let tda = spec
            .channels
            .iter()
            .find(|ch| ch.target == c.label.irrep)
            .and_then(|ch| ch.tda.as_ref())
            .and_then(|r| r.theta())
            .and_then(|t| t.first().copied());
        ct.row(row![
            c.label
                .name
                .clone()
                .unwrap_or_else(|| format!("irrep {}", c.label.irrep.index())),
            c.e_gs
                .map(|e| format!("{e:.9}"))
                .unwrap_or_else(|| "—".to_string()),
            gap,
            tda.map(|t| format!("{t:.9}"))
                .unwrap_or_else(|| "—".to_string()),
            format!("{:?}", c.status)
        ]);
    }
    ct.finish();
    println!(
        "  the `lowest theta` column is THIS step's linearization at the one typed reference; the \
         `gap` column re-converged a ground state in each irrep. They answer different questions, \
         and an irrep that `channel_response` inhabits can still be an EMPTY shifted channel here \
         — the shifted tangent space is a subspace of the target sector, never the whole of it."
    );

    // ---- SYNTHETIC observables per channel -----------------------------------------------------
    //
    // A cross-irrep one-body operator lives in exactly ONE shifted channel: `mu x` carries the flux
    // class read off its matrix, so `P^phi(mu x) = 0` on every other channel by space mismatch, not
    // by cancellation. At a totally symmetric reference every dipole-ALLOWED transition is therefore
    // in a shifted channel and the in-sector RPA above saw only the forbidden ones. Per channel the
    // driver reports the static `alpha^phi = 2 <p, (H^rr)^-1 p>` (one PCG against `A + B`), the
    // dense moments `T_n = (S_r^T gamma)_n` with `f_n = (2/3) omega_n Σ_c T_n(c)^2`, and the operator
    // sum rule `Σ_n T_n^2` against `||P^phi(mu x)||^2` — EXACT where `B^phi ≡ 0` (a single-slot
    // channel, as both live LiH channels are), a defect of EITHER sign on a curved one, never
    // asserted zero here.
    //
    // ★ SYNTHETIC. No symmetric fixture ships dipole matrices, and which irrep `x`, `y`, `z` carry
    // depends on the orientation the integral generator chose, so the components below are
    // symmetry-TYPED stand-ins built from ORBSYM. Every number in this arm is a property of the
    // machinery, not of this molecule.
    println!(
        "\n-- SYNTHETIC observables per live channel ({} symmetry-typed component(s), one per flux; \
         NOT this molecule's dipole) --",
        props.len()
    );
    let n_props = props.len();
    let mut ot = Table::new(vec![
        Col::text("irrep", 7).header("target"),
        Col::int("dim", 5),
        Col::text("comp", 11).header("component"),
        Col::sci("alpha", 11, 3).header("alpha^phi"),
        Col::sci("tmax", 10, 3).header("max|T_n|"),
        Col::int("nmax", 4).header("at n"),
        Col::sci("fn", 10, 3).header("f_n"),
        Col::float("rule", 10, 6).header("ΣT²/||p||²"),
        Col::sci("cap", 9, 2).header("capture"),
        Col::sci("cmp", 9, 2).header("complete."),
        Col::text("status", 26),
    ])
    .start();
    for ch in &spec.channels {
        if ch.status == SectorChannelStatus::Empty {
            ot.skip(&irrep_tag(ch), "dim T^phi = 0");
            continue;
        }
        let Some(own) = props
            .iter()
            .position(|p| property_flux(p, pg).ok().flatten() == Some(ch.phi))
        else {
            ot.skip(&irrep_tag(ch), "no representable component of this flux");
            continue;
        };
        let comp = props[own].label.clone();
        match (&ch.alpha, &ch.moments) {
            (Some(a), Some(m)) => {
                let (sum_t2, p2) = m.sum_rule[own];
                // The BRIGHTEST mode, not the lowest: on LiH the lowest mode of both live channels
                // is dark to this operator (|T_1| ~ 1e-16), so mode 1 alone would show nothing.
                let n_max = (0..m.omega.len())
                    .max_by(|&i, &j| m.moments[own][i].abs().total_cmp(&m.moments[own][j].abs()))
                    .unwrap_or(0);
                ot.row(row![
                    irrep_tag(ch),
                    ch.dim,
                    comp,
                    a.alpha[own * n_props + own],
                    m.moments[own].get(n_max).map_or(0.0, |z| z.abs()),
                    n_max,
                    m.oscillator.get(n_max).copied().unwrap_or(0.0),
                    if p2 > 0.0 { sum_t2 / p2 } else { f64::NAN },
                    m.capture_defect[own],
                    m.completeness_defect[own],
                    format!("{:?}", ch.status)
                ]);
            }
            (Some(a), None) => ot.row(row![
                irrep_tag(ch),
                ch.dim,
                comp,
                a.alpha[own * n_props + own],
                f64::NAN,
                0,
                f64::NAN,
                f64::NAN,
                a.capture_defect[own],
                f64::NAN,
                "moments not attempted (dense_cap)".to_string()
            ]),
            _ => ot.skip(&irrep_tag(ch), format!("{:?}", ch.status)),
        }
    }
    ot.finish();
    println!(
        "  alpha^phi is the channel's static polarizability of its OWN component — every other \
         component's row and column are structural zeros, so the channel blocks add up into one \
         tensor (alpha_total below). max|T_n| is the BRIGHTEST mode's moment, at mode n (0-based, \
         ascending omega), f_n its oscillator strength; on LiH the LOWEST mode of both channels is \
         dark to this operator (|T_0| ~ 1e-16, an exact zero by symmetry rather than a small number \
         — this step does not resolve which symmetry). ΣT²/||p||² is the operator sum \
         rule: exactly 1 where B^phi ≡ 0 (both live LiH channels are single-slot), a ratio of either \
         side of 1 on a curved channel (h2o_sto3g's B1), reported and never asserted. `capture` is \
         the fraction of mu x the channel's borrowed-suffix tangent space MISSED — O(1) on a \
         proper-subspace channel, read it against dim/dets above — and `complete.` what the channel \
         basis missed of the covector."
    );
    if let Some(total) = &spec.alpha_total {
        println!(
            "  SYNTHETIC alpha_total diagonal (Σ over channels; the in-sector A-flux block is NOT \
             included — that is the in-sector problem's): {:?}",
            (0..n_props)
                .map(|r| format!("{:.6e}", total[r * n_props + r]))
                .collect::<Vec<_>>()
        );
    }
    if let Some(f) = spec.oscillator_total {
        println!(
            "  Σ_channels Σ_n f_n = {f:.6e} — NOT Thomas–Reiche–Kuhn (synthetic operator, incomplete \
             basis, in-sector channel excluded): a diagnostic, never an oracle."
        );
    }
    if !spec.skipped.is_empty() {
        println!(
            "  components computed nowhere (IDENTITY flux or multi-class): {:?}",
            spec.skipped
        );
    }

    // ---- dense cross-check, last --------------------------------------------------------------
    //
    // Three checks, in increasing strength. (i) the dense Williamson reduction over the WHOLE
    // shifted basis against the iterative `omega^2`; (ii) the untyped-embedding check —
    // the same Hessian block computed by machinery that knows nothing about shifted tangent spaces;
    // (iii) the structural dimension identity, exact and integer, which is this step's one
    // assertion.
    println!("\n-- dense cross-check --");
    let mut dense_dev = 0.0f64;
    let mut oracle_dev = 0.0f64;
    let mut oracle_done = false;
    if dense_cap == 0 {
        println!("  skipped: --dense_cap 0");
    } else {
        let mut dt = Table::new(vec![
            Col::text("irrep", 7).header("target"),
            Col::int("n", 5).header("dim"),
            Col::text("verdict", 9),
            Col::sci("dev", 11, 3).header("dense vs iter"),
            Col::text("note", 40),
        ])
        .start();
        for ch in &spec.channels {
            if ch.status == SectorChannelStatus::Empty {
                continue;
            }
            if ch.dim > dense_cap {
                dt.skip(&irrep_tag(ch), format!("dim {} > cap {dense_cap}", ch.dim));
                continue;
            }
            let space_out = space.with_target(ch.target);
            let h_out = SparseHamiltonian::<f64>::new(space_out, fx.t(), fx.v(), w, f64::EPSILON);
            let sc = SectorCtx::new(&ctx, &h_out, ch.phi);
            // The frames are the CONTEXT's own — a second factorization of `x` would silently
            // decouple the basis from the applies.
            let basis = tangent_basis_shifted(base_l, base_r, ch.phi);
            let spectrum = sc.symplectic_eigenvalues(&basis);
            let want = ch.rpa.as_ref().and_then(|r| r.omega2()).unwrap_or_default();
            let (dev, note) = match spectrum.omega_squared() {
                Some(w2) if !want.is_empty() => {
                    let dv = worst_dev(&w2[..w2.len().min(want.len())], &want);
                    dense_dev = dense_dev.max(dv);
                    (
                        dv,
                        format!("{} dense values, {} iterative", w2.len(), want.len()),
                    )
                }
                Some(w2) => (
                    f64::NAN,
                    format!("{} dense values; no iterative omega^2 to compare", w2.len()),
                ),
                None => (
                    f64::NAN,
                    "both blocks indefinite: no real omega^2 exists".to_string(),
                ),
            };
            dt.row(row![
                irrep_tag(ch),
                basis.len(),
                dense_tag(&spectrum),
                dev,
                note
            ]);

            // The untyped-embedding check, run on the FIRST live channel. `x_c1` is the
            // direct-sum embedding of `x` into the trivial-point-group space; the UNTYPED
            // `weingarten_operator`/`tangent_basis`/`hessian_grams` compute the Riemannian Hessian
            // there, and `project_irrep` cuts the untyped tangent basis into irrep blocks. The
            // target irrep's block must be the shifted channel — same dimension, same spectra.
            //
            // ★ `hessian_grams` and `SectorCtx::grams` both return **(A, B)**, never
            // (H^rr, H^ii): `H^rr = A + B` and `H^ii = A − B` are formed by hand below.
            if oracle_done {
                continue;
            }
            oracle_done = true;
            let x_c1: TtVector<f64> = x.forget_point_group();
            let h_c1 = SparseHamiltonian::<f64>::new(
                QnSpace::new(d, space.nup, space.ndn),
                fx.t(),
                fx.v(),
                w,
                f64::EPSILON,
            );
            let ctx_c1 = weingarten_operator(&h_c1, &x_c1);
            let (bl1, br1) = ctx_c1.bases();
            let block = mgs(tangent_basis(bl1, br1, false)
                .iter()
                .map(|b| project_irrep(b, bl1, br1, pg, ch.target).0)
                .collect());
            let n = block.len();
            if n != basis.len() {
                println!(
                    "  ORACLE SKIPPED for {}: the untyped irrep block has {n} directions and \
                     T^phi has {} — comparing spectra of different dimensions is meaningless",
                    irrep_tag(ch),
                    basis.len()
                );
                continue;
            }
            let (a_c1, b_c1) = hessian_grams(&ctx_c1, &block);
            let (a_phi, b_phi) = sc.grams(&basis);
            let mut worst = 0.0f64;
            for (label, sign) in [("A", None), ("A+B", Some(1.0)), ("A-B", Some(-1.0))] {
                let (g1, g2) = match sign {
                    None => (a_c1.clone(), a_phi.clone()),
                    Some(s) => (
                        gram_combine(&a_c1, &b_c1, s),
                        gram_combine(&a_phi, &b_phi, s),
                    ),
                };
                let (e1, e2) = (gram_eigs(&g1, n), gram_eigs(&g2, n));
                let sc_l = e1.iter().fold(0.0f64, |mx, z| mx.max(z.abs())).max(1.0);
                let dv = worst_dev(&e1, &e2);
                worst = worst.max(dv / sc_l);
                println!(
                    "  untyped oracle {} [{label}]: n = {n}, scale {sc_l:.3e}, worst \
                     |eig difference| {dv:.3e}",
                    irrep_tag(ch)
                );
            }
            oracle_dev = worst;
        }
        dt.finish();
        println!(
            "  the dense driver is one Hessian-class apply pair per tangent direction — an oracle, \
             not a production path. Its verdict is the three-way Williamson routing; the iterative \
             readout carries signed omega^2 plus witnesses only."
        );
    }

    // The structural identity: the shifted spaces are the irrep BLOCKS of ONE untyped tangent
    // space, so they partition it. Exact and integer — no tolerance, and no dependence on the
    // trajectory the polish took.
    let untyped_dim = tangent_dim(&x.forget_point_group());
    println!(
        "\ndimension identity: sum_phi dim T^phi + tangent_dim(x) = {} vs \
         tangent_dim(forget_point_group(x)) = {untyped_dim}",
        spec.dim_total()
    );

    let mut s = Summary::new("step-18")
        .s("fixture", &fixture)
        .i("rank", rank as i64)
        .i("k", k as i64)
        .s("readout", m.str("readout"))
        .f("lambda", lambda, 9)
        .e("gradnorm", gradnorm, 3)
        .e("sigma_min", sigma_min, 3)
        .i("built", spec.built as i64)
        .i("dim_total", spec.dim_total() as i64)
        .i("untyped_dim", untyped_dim as i64)
        .e("f_total_synthetic", spec.oscillator_total.unwrap_or(0.0), 3);
    if dense_cap > 0 {
        s = s
            .e("dense_dev", dense_dev, 3)
            .e("oracle_dev", oracle_dev, 3);
    }
    s.emit();

    let dims_ok = spec.dim_total() == untyped_dim;
    let oracle_ok = !oracle_done || oracle_dev <= ORACLE_TOL;
    println!(
        "\nACCEPTANCE {fixture} rank {rank}: {} (criterion: the dimension identity holds exactly, \
         and every algebraic cross-check that ran agrees to {ORACLE_TOL:.0e} relative)",
        if dims_ok && oracle_ok { "PASS" } else { "FAIL" }
    );
    assert!(
        dims_ok,
        "sum_phi dim T^phi + tangent_dim(x) = {} but tangent_dim(forget_point_group(x)) = \
         {untyped_dim} — the shifted spaces must PARTITION the untyped tangent space",
        spec.dim_total()
    );
}
