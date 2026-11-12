// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::{min_schmidt, tangent_basis, tangent_dim, tangent_dim_grids};
use qntt::pointgroup::Irrep;
use qntt::solvers::hess::{
    BWeight, ChannelGsOpts, ChannelOpts, ChannelSet, ChannelStatus, ConeOpts, IrrepSel, RpaKernel,
    RpaMetric, SectorAllOpts, SectorAllReadout, SectorChannel, SectorChannelStatus, SectorCtx,
    SectorOpts, SectorReadout, SectorResponse, SectorSpectrumAll, SymplecticSpectrum,
    channel_response, sector_response_all, sector_response_all_from_load, symplectic_eigenvalues,
    weingarten_operator,
};
use qntt::solvers::riemannian::{normalize_mut, tangent_bases};
use qntt::{QnSpace, Scalar, SparseHamiltonian, TtVector, max_ranks, rayleigh_quotient};
use qntt_testkit::dense::dets_for;
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::sector_util::*;

fn arm_values(r: Option<&SectorResponse>, which: SectorReadout) -> Vec<f64> {
    let Some(r) = r else { return Vec::new() };
    match which {
        SectorReadout::Tda => r
            .theta()
            .unwrap_or_else(|| {
                panic!("a row's `tda` field must carry a Tda readout, not an Rpa one")
            })
            .to_vec(),
        SectorReadout::Rpa => r
            .omega2()
            .unwrap_or_else(|| {
                panic!("a row's `rpa` field must carry an Rpa readout, not a Tda one")
            })
            .to_vec(),
    }
}

fn frobenius(g: &[f64]) -> f64 {
    g.iter().map(|z| z * z).sum::<f64>().sqrt()
}

fn report_bits(got: &[f64], want: &[f64], what: &str) -> bool {
    let same = got.len() == want.len()
        && got
            .iter()
            .zip(want.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
    println!("  {what}: bit-identical = {same} ({} values)", got.len());
    same
}

fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: value COUNT {} != {}\n got  {got:?}\n want {want:?}",
        got.len(),
        want.len()
    );
    let scale = want.iter().fold(1.0f64, |m, z| m.max(z.abs()));
    for (j, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol * scale,
            "{what}: value {j} is {a}, want {b} (tol {tol:e} × scale {scale:e})"
        );
    }
}

fn assert_specs_agree(a: &SectorSpectrumAll, b: &SectorSpectrumAll, tol: f64, what: &str) {
    assert_eq!(
        a.channels.len(),
        b.channels.len(),
        "{what}: channel COUNT differs"
    );
    assert_eq!(
        a.reference.target, b.reference.target,
        "{what}: reference target differs"
    );
    assert_eq!(
        a.reference.tangent_dim, b.reference.tangent_dim,
        "{what}: reference tangent_dim differs"
    );
    assert_close(
        &[a.reference.lambda, a.reference.sigma_min],
        &[b.reference.lambda, b.reference.sigma_min],
        tol,
        &format!("{what}: reference (lambda, sigma_min)"),
    );
    assert_eq!(
        a.reference_spectrum.is_some(),
        b.reference_spectrum.is_some(),
        "{what}: one run stored a reference spectrum and the other did not"
    );
    for (i, (p, q)) in a.channels.iter().zip(b.channels.iter()).enumerate() {
        assert_eq!(p.phi, q.phi, "{what}: row {i} phi");
        assert_eq!(p.target, q.target, "{what}: row {i} target");
        assert_eq!(p.label.irrep, q.label.irrep, "{what}: row {i} label irrep");
        assert_eq!(p.dim, q.dim, "{what}: row {i} dim");
        assert_eq!(p.status, q.status, "{what}: row {i} status");
        for which in [SectorReadout::Tda, SectorReadout::Rpa] {
            let pick = |c: &SectorChannel| match which {
                SectorReadout::Tda => c.tda.clone(),
                SectorReadout::Rpa => c.rpa.clone(),
            };
            assert_eq!(
                pick(p).is_some(),
                pick(q).is_some(),
                "{what}: row {i}: the {which:?} arm ran in only one of the two runs"
            );
            let u = arm_values(pick(p).as_ref(), which);
            let w = arm_values(pick(q).as_ref(), which);
            let tag = format!("{what}: row {i} ({:?}) {which:?}", p.phi);
            report_bits(&u, &w, &tag);
            assert_close(&u, &w, tol, &tag);
        }
        assert_eq!(
            p.dense.is_some(),
            q.dense.is_some(),
            "{what}: row {i}: the dense route fired in only one of the two runs"
        );
        if let (Some(u), Some(w)) = (p.dense.as_ref(), q.dense.as_ref()) {
            let tag = format!("{what}: row {i} dense signed omega^2");
            let (lu, lw) = (spectrum_label(u), spectrum_label(w));
            assert_eq!(
                u.omega_squared().is_some(),
                w.omega_squared().is_some(),
                "{tag}: one run has a real signed spectrum and the other does not ({lu} vs {lw})"
            );
            let uu = u.omega_squared().unwrap_or_default();
            let ww = w.omega_squared().unwrap_or_default();
            if uu.is_empty() && ww.is_empty() {
                println!("  {tag}: both runs report {lu} / {lw}");
            } else {
                report_bits(&uu, &ww, &tag);
                assert_close(&uu, &ww, tol, &tag);
            }
        }
        assert_eq!(
            b_weight_tag(p.b_weight.as_ref()),
            b_weight_tag(q.b_weight.as_ref()),
            "{what}: row {i}: the ||B||/||A|| SPELLING differs between runs — a Frobenius norm \
             ratio and a seed-block fraction are different numbers (D-spec §1.8)"
        );
    }
}

fn b_weight_tag(w: Option<&BWeight>) -> &'static str {
    match w {
        None => "none",
        Some(BWeight::GramRatio(_)) => "gram",
        Some(BWeight::SeedFraction { .. }) => "seed",
    }
}

fn print_row(ch: &SectorChannel) {
    println!(
        "  phi {:?} -> target {:?} (label {:?}/{:?}) dim {} status {:?} b_weight {:?} \
         secs {:.4} tda {:?} rpa {:?} dense {}",
        ch.phi,
        ch.target,
        ch.label.irrep,
        ch.label.name,
        ch.dim,
        ch.status,
        ch.b_weight,
        ch.secs,
        ch.tda.as_ref().and_then(|r| r.theta()),
        ch.rpa.as_ref().and_then(|r| r.omega2()),
        ch.dense.as_ref().map_or("-".to_string(), spectrum_label),
    );
}

fn tda_opts(k: usize) -> SectorAllOpts {
    SectorAllOpts {
        readout: SectorAllReadout::Tda,
        k,
        ..SectorAllOpts::default()
    }
}

#[test]
fn tsd_all_enumerates_every_flux_once_and_skips_empty_channels_without_an_operator() {
    let pg = fx4_pg();
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, br) = tangent_bases(&x);
    let order = space.pg.order();
    let live = live_channels(&space, &bl.ranks);
    println!(
        "tsd enumeration: sigma_min = {:e}, ||grad|| = {:e}, live = {live:?}",
        min_schmidt(&bl, &br),
        grad_norm_at(&ham, &x)
    );

    let empty_phis: Vec<Irrep> = (0..order)
        .map(Irrep::new)
        .filter(|&phi| phi != Irrep::IDENTITY && tangent_dim_grids(&space, &bl.ranks, phi) == 0)
        .collect();
    assert!(
        !empty_phis.is_empty(),
        "fixture premise: this arm needs at least one structurally empty channel (live = {live:?})"
    );
    for &phi in &empty_phis {
        let gamma = space.target.fuse(phi);
        let n = dets_in_irrep(D4, T_NUP, T_NDN, pg, gamma).len();
        println!("tsd enumeration: empty phi {phi:?} -> target {gamma:?} holds {n} determinants");
        assert!(
            n > 0,
            "fixture premise: channel {phi:?} must be empty because of the BORROWED SUFFIX BASES, \
             not because its target sector {gamma:?} has no states in it"
        );
    }

    let spec = sector_response_all(&ham, &x, &t, &v, None, &tda_opts(1));
    println!("tsd enumeration [All]: reference {:?}", spec.reference);
    for ch in &spec.channels {
        print_row(ch);
    }
    assert_eq!(
        spec.channels.len(),
        usize::from(order) - 1,
        "IrrepSel::All must enumerate 1..pg.order(), i.e. every flux but IDENTITY"
    );
    let phis: Vec<Irrep> = spec.channels.iter().map(|c| c.phi).collect();
    let mut sorted = phis.clone();
    sorted.sort_by_key(|p| p.index());
    sorted.dedup();
    assert_eq!(
        phis, sorted,
        "IrrepSel::All must enumerate ascending by irrep index, each flux exactly once: {phis:?}"
    );
    assert!(
        !phis.contains(&Irrep::IDENTITY),
        "IDENTITY is not a cross-sector channel and must never be a row (D-spec §1.5)"
    );
    for ch in &spec.channels {
        assert_eq!(
            ch.target,
            space.target.fuse(ch.phi),
            "row {:?}: target must be Gamma0 (+) phi",
            ch.phi
        );
        assert_eq!(
            ch.label.irrep, ch.target,
            "row {:?}: the shipped ChannelLabel's irrep IS the target (D-spec §1.4)",
            ch.phi
        );
        assert_eq!(
            ch.label.spin_delta, 0,
            "row {:?}: an irrep channel carries no spin offset",
            ch.phi
        );
        assert!(
            ch.label.name.is_none(),
            "row {:?}: `kind: None` must leave the rendered name unset",
            ch.phi
        );
        assert_eq!(
            ch.dim,
            tangent_dim_grids(&space, &bl.ranks, ch.phi),
            "row {:?}: `dim` must be the structural tangent_dim_grids, the same quantity the skip \
             test reads (D-spec §1.6) — otherwise the two can drift",
            ch.phi
        );
    }

    let live_secs = spec
        .channels
        .iter()
        .filter(|c| c.status != SectorChannelStatus::Empty)
        .map(|c| c.secs)
        .fold(f64::INFINITY, f64::min);
    assert!(
        live_secs.is_finite(),
        "anti-vacuity: no LIVE channel in this run, so there is no operator-build wall to compare \
         the empty rows against"
    );
    let mut n_empty = 0usize;
    for ch in &spec.channels {
        if !empty_phis.contains(&ch.phi) {
            assert_ne!(
                ch.status,
                SectorChannelStatus::Empty,
                "row {:?} has dim {} but was reported Empty",
                ch.phi,
                ch.dim
            );
            continue;
        }
        n_empty += 1;
        assert_eq!(ch.status, SectorChannelStatus::Empty, "row {:?}", ch.phi);
        assert_eq!(ch.dim, 0, "an Empty row's dim must be 0");
        assert!(
            ch.tda.is_none(),
            "an Empty row must not carry a Tda readout"
        );
        assert!(
            ch.rpa.is_none(),
            "an Empty row must not carry an Rpa readout"
        );
        assert!(
            ch.dense.is_none(),
            "an Empty row must not carry a dense spectrum"
        );
        assert!(
            ch.b_weight.is_none(),
            "an Empty row has no A^phi to score B^phi against (D-spec §1.8)"
        );
        assert!(
            ch.secs <= 0.25 * live_secs,
            "row {:?}: an Empty channel took {:.6} s against the cheapest live channel's {:.6} s \
             — the skip is supposed to happen BEFORE the operator build, and an automaton build is \
             three orders of magnitude more than a tangent_dim_grids sum at this scale",
            ch.phi,
            ch.secs,
            live_secs
        );
    }
    assert_eq!(n_empty, empty_phis.len());
    assert_eq!(
        spec.empty_channels().count(),
        empty_phis.len(),
        "empty_channels() must report exactly the Empty rows"
    );
    assert_eq!(
        spec.built,
        spec.channels.len() - n_empty,
        "`built` counts the channel Hamiltonians actually CONSTRUCTED, so an Empty row must \
         contribute none (D-spec §9.1 ruling 1): {} rows, {n_empty} of them Empty, built {}",
        spec.channels.len(),
        spec.built
    );
    for ch in spec.empty_channels() {
        assert_eq!(ch.status, SectorChannelStatus::Empty);
    }

    let want_order = vec![Irrep::new(3), Irrep::new(1)];
    let listed = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            irreps: IrrepSel::List(vec![
                Irrep::IDENTITY,
                Irrep::new(3),
                Irrep::new(3),
                Irrep::new(1),
            ]),
            ..tda_opts(1)
        },
    );
    for ch in &listed.channels {
        print_row(ch);
    }
    assert_eq!(
        listed.channels.iter().map(|c| c.phi).collect::<Vec<_>>(),
        want_order,
        "IrrepSel::List must preserve the caller's order, drop duplicates, and filter IDENTITY \
         silently (D-spec §1.3) — asking for IDENTITY is a request for the in-sector spectrum, \
         which this driver does not answer"
    );

    let ref_only = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            irreps: IrrepSel::RefOnly,
            ..tda_opts(1)
        },
    );
    assert!(
        ref_only.channels.is_empty(),
        "IrrepSel::RefOnly names the reference channel, which is phi = IDENTITY, which this driver \
         never enumerates — so the row list is EMPTY (D-spec §1.3), not a reference row"
    );
    assert_eq!(
        ref_only.dim_total(),
        ref_only.reference.tangent_dim,
        "dim_total() sums the ENUMERATED channels plus the reference tangent dimension, so with no \
         channels it is the reference's own dimension (D-spec §1.7)"
    );
    assert_eq!(ref_only.reference.target, space.target);
    assert_eq!(
        ref_only.built, 0,
        "IrrepSel::RefOnly enumerates no channel, so no operator may be built at all"
    );
}

#[test]
fn tsd_all_reproduces_a_hand_rolled_sector_ctx_loop() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    let live = live_channels(&space, &bl.ranks);
    let k = 2usize;
    println!(
        "tsd hand-rolled: sigma_min = {:e}, ||grad|| = {:e}, live = {live:?}",
        min_schmidt(&bl, &br),
        grad_norm_at(&ham, &x)
    );
    assert!(
        live.iter().filter(|&&p| p != Irrep::IDENTITY).count() >= 2,
        "anti-vacuity: the loop must run over at least two live channels (live = {live:?})"
    );

    let opts = SectorAllOpts {
        readout: SectorAllReadout::Both,
        k,
        dense_cap: Some(usize::MAX),
        ..SectorAllOpts::default()
    };
    let spec = sector_response_all(&ham, &x, &t, &v, None, &opts);
    for ch in &spec.channels {
        print_row(ch);
    }

    let ctx = weingarten_operator(&ham, &x);
    let mut compared = 0usize;
    for ch in &spec.channels {
        if ch.status == SectorChannelStatus::Empty {
            assert_eq!(ch.dim, 0);
            continue;
        }
        let space_out = space.with_target(space.target.fuse(ch.phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, opts.eps_h);
        let sctx = SectorCtx::new(&ctx, &h_out, ch.phi);
        assert_eq!(
            ch.dim,
            sctx.dim(),
            "row {:?}: the driver's `dim` must be SectorCtx::dim",
            ch.phi
        );

        let want_tda = sctx.sector_lowest_tda(
            k,
            &SectorOpts {
                readout: SectorReadout::Tda,
                ..opts.solver.clone()
            },
        );
        let want_rpa = sctx.sector_lowest(
            k,
            &SectorOpts {
                readout: SectorReadout::Rpa,
                ..opts.solver.clone()
            },
        );
        println!(
            "tsd hand-rolled {:?}: dim {} tda applies {}/{} rpa applies {}/{}",
            ch.phi,
            ch.dim,
            ch.tda.as_ref().map_or(0, |r| r.applies),
            want_tda.applies,
            ch.rpa.as_ref().map_or(0, |r| r.applies),
            want_rpa.applies,
        );

        let got_t = arm_values(ch.tda.as_ref(), SectorReadout::Tda);
        let got_r = arm_values(ch.rpa.as_ref(), SectorReadout::Rpa);
        let want_t = arm_values(Some(&want_tda), SectorReadout::Tda);
        let want_r = arm_values(Some(&want_rpa), SectorReadout::Rpa);
        let tag_t = format!("row {:?} theta", ch.phi);
        let tag_r = format!("row {:?} signed omega^2", ch.phi);
        report_bits(&got_t, &want_t, &tag_t);
        report_bits(&got_r, &want_r, &tag_r);
        assert_close(&got_t, &want_t, 1e-12, &tag_t);
        assert_close(&got_r, &want_r, 1e-12, &tag_r);
        let row_tda = ch.tda.as_ref().expect("a Both run fills the Tda arm");
        let row_rpa = ch.rpa.as_ref().expect("a Both run fills the Rpa arm");
        assert_eq!(row_tda.converged, want_tda.converged, "row {:?}", ch.phi);
        assert_eq!(row_tda.applies, want_tda.applies, "row {:?}", ch.phi);
        assert_eq!(row_rpa.converged, want_rpa.converged, "row {:?}", ch.phi);
        assert_eq!(row_rpa.applies, want_rpa.applies, "row {:?}", ch.phi);
        assert_eq!(row_tda.phi, ch.phi);
        assert_eq!(row_tda.target, ch.target);

        let basis = sctx.basis();
        let want_dense = sctx.symplectic_eigenvalues(&basis);
        let got_dense = ch
            .dense
            .as_ref()
            .expect("dense_cap = usize::MAX admits every channel");
        println!(
            "tsd hand-rolled {:?}: dense {} vs kernel {} (basis {})",
            ch.phi,
            spectrum_label(got_dense),
            spectrum_label(&want_dense),
            basis.len()
        );
        assert_eq!(
            got_dense.omega_squared().is_some(),
            want_dense.omega_squared().is_some(),
            "row {:?}: the driver's dense route and the kernel disagree on whether a real signed \
             spectrum exists ({} vs {})",
            ch.phi,
            spectrum_label(got_dense),
            spectrum_label(&want_dense)
        );
        let a = got_dense.omega_squared().unwrap_or_default();
        let b = want_dense.omega_squared().unwrap_or_default();
        let tag = format!("row {:?} dense signed omega^2", ch.phi);
        report_bits(&a, &b, &tag);
        assert_close(&a, &b, 1e-12, &tag);
        compared += 1;
    }
    assert!(
        compared >= 2,
        "anti-vacuity: fewer than two live channels were compared"
    );
}

#[test]
fn tsd_all_include_reference_stores_the_shipped_in_sector_spectrum() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    let sigma_min = min_schmidt(&bl, &br);
    let lambda = rayleigh_quotient(&ham, &x).re_s();
    println!(
        "tsd include_reference: lambda = {lambda}, sigma_min = {sigma_min:e}, ||grad|| = {:e}",
        grad_norm_at(&ham, &x)
    );

    let off = sector_response_all(&ham, &x, &t, &v, None, &tda_opts(1));
    let on = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            include_reference: true,
            ..tda_opts(1)
        },
    );

    for (what, spec) in [
        ("include_reference off", &off),
        ("include_reference on", &on),
    ] {
        println!(
            "tsd include_reference [{what}]: reference {:?}",
            spec.reference
        );
        assert!(
            (spec.reference.lambda - lambda).abs() <= 1e-12 * lambda.abs().max(1.0),
            "[{what}] reference lambda {} != rayleigh_quotient {lambda}",
            spec.reference.lambda
        );
        assert_eq!(
            spec.reference.target, space.target,
            "[{what}] the reference target is Gamma0"
        );
        assert_eq!(
            spec.reference.tangent_dim,
            tangent_dim_grids(&space, &x.ranks, Irrep::IDENTITY),
            "[{what}] reference tangent_dim must be the IDENTITY-flux tangent dimension (D-spec \
             §1.6); tangent_dim(base_l) = {} is the same quantity",
            tangent_dim(&bl)
        );
        assert_eq!(
            spec.reference.tangent_dim,
            tangent_dim(&bl),
            "[{what}] the two spellings of the reference tangent dimension must agree"
        );
        assert!(
            (spec.reference.sigma_min - sigma_min).abs() <= 1e-14 * sigma_min.abs().max(1.0),
            "[{what}] reference sigma_min {} != min_schmidt {sigma_min:e}",
            spec.reference.sigma_min
        );
    }

    assert!(
        off.reference_spectrum.is_none(),
        "include_reference = false must leave reference_spectrum None (D-spec §9 Q1): the caller \
         composes the in-sector leg itself"
    );
    let got = on
        .reference_spectrum
        .as_ref()
        .expect("include_reference = true must store the in-sector spectrum");

    let ctx = weingarten_operator(&ham, &x);
    let (cbl, cbr) = ctx.bases();
    let basis = tangent_basis(cbl, cbr, true);
    let want = symplectic_eigenvalues(&ctx, &basis);
    println!(
        "tsd include_reference: stored {} vs shipped {} (basis {}, tangent_dim {})",
        spectrum_label(got),
        spectrum_label(&want),
        basis.len(),
        on.reference.tangent_dim
    );
    assert!(
        matches!(want, SymplecticSpectrum::Stable { .. }),
        "FIXTURE PRECONDITION: the in-sector Williamson reduction must exist at this endpoint for \
         the comparison to have content — got {} (sigma_min {sigma_min:e}). A different ISA \
         landing elsewhere fails HERE, loudly, rather than silently weakening the arm",
        spectrum_label(&want)
    );
    let a = got
        .omega_squared()
        .expect("the stored reference spectrum is Stable, so omega^2 is real");
    let b = want
        .omega_squared()
        .expect("the shipped in-sector spectrum is Stable");
    report_bits(&a, &b, "reference signed omega^2");
    assert_close(&a, &b, 1e-12, "reference signed omega^2");
    assert_eq!(
        std::mem::discriminant(got),
        std::mem::discriminant(&want),
        "the stored spectrum must be the same Williamson ARM as the shipped call, not just the \
         same numbers"
    );

    let mut off_rows = off.clone();
    off_rows.reference_spectrum = None;
    let mut on_rows = on.clone();
    on_rows.reference_spectrum = None;
    assert_specs_agree(
        &on_rows,
        &off_rows,
        1e-12,
        "include_reference on vs off (rows only)",
    );
}

#[test]
fn tsd_all_levels_merge_is_sorted_labelled_and_unit_pure() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    println!(
        "tsd levels: sigma_min = {:e}, live = {:?}",
        min_schmidt(&bl, &br),
        live_channels(&space, &bl.ranks)
    );
    let k = 2usize;
    let spec = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            readout: SectorAllReadout::Both,
            k,
            ..SectorAllOpts::default()
        },
    );
    for ch in &spec.channels {
        print_row(ch);
    }
    let lambda = spec.reference.lambda;

    for which in [SectorReadout::Tda, SectorReadout::Rpa] {
        let levels = spec.levels(which);
        let want_total: usize = spec
            .channels
            .iter()
            .map(|c| {
                arm_values(
                    match which {
                        SectorReadout::Tda => c.tda.as_ref(),
                        SectorReadout::Rpa => c.rpa.as_ref(),
                    },
                    which,
                )
                .len()
            })
            .sum();
        println!(
            "tsd levels [{which:?}]: {} levels, rows contribute {want_total}",
            levels.len()
        );
        assert_eq!(
            levels.len(),
            want_total,
            "[{which:?}] the merge must carry exactly the values the rows produced — not a \
             channel's structural dim, and not a k per empty channel"
        );
        assert!(
            levels.len() >= 4,
            "anti-vacuity [{which:?}]: a merge of fewer than four levels cannot exercise ordering \
             ACROSS channels"
        );
        let mut seen_channels = std::collections::BTreeSet::new();
        for (j, lv) in levels.iter().enumerate() {
            assert_eq!(
                lv.readout, which,
                "[{which:?}] level {j} carries the wrong readout tag — that tag is what tells a \
                 consumer whether `value` is an energy or an omega^2"
            );
            let ch = &spec.channels[lv.channel];
            seen_channels.insert(lv.channel);
            let vals = arm_values(
                match which {
                    SectorReadout::Tda => ch.tda.as_ref(),
                    SectorReadout::Rpa => ch.rpa.as_ref(),
                },
                which,
            );
            assert!(
                lv.k < vals.len(),
                "[{which:?}] level {j} points at root {} of channel {} which has {} values",
                lv.k,
                lv.channel,
                vals.len()
            );
            assert_eq!(
                lv.value.to_bits(),
                vals[lv.k].to_bits(),
                "[{which:?}] level {j} must carry the row's own value verbatim, not a re-derived \
                 one: {} vs {}",
                lv.value,
                vals[lv.k]
            );
            match which {
                SectorReadout::Tda => assert!(
                    (lv.energy - (lambda + lv.value)).abs() <= 1e-12 * lv.energy.abs().max(1.0),
                    "[Tda] level {j}: energy must be lambda + eig(A^phi) = {} + {}, got {}",
                    lambda,
                    lv.value,
                    lv.energy
                ),
                SectorReadout::Rpa => {
                    if lv.value >= 0.0 {
                        let want = lambda + lv.value.sqrt();
                        assert!(
                            (lv.energy - want).abs() <= 1e-12 * want.abs().max(1.0),
                            "[Rpa] level {j}: energy must be lambda + sqrt(omega^2) = {want}, got \
                             {}",
                            lv.energy
                        );
                    } else {
                        assert!(
                            lv.energy.is_nan(),
                            "[Rpa] level {j}: a NEGATIVE omega^2 is a resonance (omega = \
                             i·sqrt|omega^2|), which no real energy can carry — `energy` must be \
                             NaN and `value` is the number to read (D-spec §1.7), got {}",
                            lv.energy
                        );
                    }
                }
            }
            if j > 0 {
                let p = &levels[j - 1];
                assert!(
                    p.value <= lv.value,
                    "[{which:?}] the merge must be ascending in `value`: level {} = {} then {} = \
                     {}",
                    j - 1,
                    p.value,
                    j,
                    lv.value
                );
                if p.value == lv.value {
                    assert!(
                        (p.channel, p.k) < (lv.channel, lv.k),
                        "[{which:?}] equal values must be ordered by (channel, k): ({}, {}) then \
                         ({}, {})",
                        p.channel,
                        p.k,
                        lv.channel,
                        lv.k
                    );
                }
            }
        }
        assert!(
            seen_channels.len() >= 2,
            "anti-vacuity [{which:?}]: the merge drew from {} channel(s); with one it is a sort, \
             not a merge",
            seen_channels.len()
        );
    }

    let rpa_only = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            readout: SectorAllReadout::Rpa,
            k: 1,
            ..SectorAllOpts::default()
        },
    );
    assert!(
        rpa_only.channels.iter().all(|c| c.tda.is_none()),
        "an Rpa-only run must not fill the Tda arm"
    );
    assert!(
        !rpa_only.levels(SectorReadout::Rpa).is_empty(),
        "anti-vacuity: the Rpa-only run produced no omega^2 at all, so the next assertion would \
         hold for the wrong reason"
    );
    assert!(
        rpa_only.levels(SectorReadout::Tda).is_empty(),
        "levels(Tda) on an Rpa-only result must be EMPTY: a channel that produced no values of \
         that readout contributes nothing (D-spec §1.7), and an omega^2 must never appear under a \
         Tda tag"
    );
}

#[test]
fn tsd_all_full_rank_union_is_the_dense_irrep_resolved_spectrum() {
    let pg = synth_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = synth_tv(pg);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);

    let caps = max_ranks(&space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7601);
    let mut x: TtVector<f64> = randomized_state(space, 3, 3, &mut rng);
    x.round_ranks_mut(&caps);
    normalize_mut(&mut x);
    x.check();
    let (bl, br) = tangent_bases(&x);
    for (b, cap) in caps.iter().enumerate() {
        assert_eq!(
            bl.ranks[b], *cap,
            "this arm needs the max_ranks grid at bond {b}: that is where T^phi can be the whole \
             sector"
        );
    }
    let lambda = rayleigh_quotient(&ham, &x).re_s();
    let g = grad_norm_at(&ham, &x);
    println!(
        "tsd union: lambda = {lambda}, ||grad|| = {g:e}, sigma_min = {:e}, live = {:?}",
        min_schmidt(&bl, &br),
        live_channels(&space, &bl.ranks)
    );
    assert!(
        g > 1e-2,
        "anti-vacuity: the identity is BASE-POINT-FREE, so it must be pinned at a non-critical \
         point or it is an eigenvector statement in disguise (||grad|| = {g:e})"
    );

    let k = (1..pg.order())
        .map(|p| tangent_dim_grids(&space, &bl.ranks, Irrep::new(p)))
        .max()
        .expect("the group has more than one irrep");
    assert!(
        k >= 4,
        "anti-vacuity: the live channel carries {k} directions"
    );
    let spec = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            readout: SectorAllReadout::Tda,
            k,
            dense_cap: Some(usize::MAX),
            ..SectorAllOpts::default()
        },
    );
    for ch in &spec.channels {
        print_row(ch);
    }

    let mut union: Vec<f64> = Vec::new();
    let mut want_union: Vec<f64> = Vec::new();
    let mut checked = 0usize;
    for ch in &spec.channels {
        let want = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, ch.target);
        want_union.extend(want.iter().copied());
        if ch.status == SectorChannelStatus::Empty {
            assert_eq!(ch.dim, 0);
            assert!(
                want.is_empty(),
                "row {:?}: this arm's fixture was chosen so that an empty CHANNEL coincides with \
                 an empty target SECTOR ({:?} holds {} determinants); on a fixture where it does \
                 not, the union identity would be false and the arm would have to state the \
                 interlacing bound instead",
                ch.phi,
                ch.target,
                want.len()
            );
            continue;
        }
        assert_eq!(
            ch.dim,
            want.len(),
            "row {:?}: `A^phi = H|_{{{:?}}} - lambda` holds only where T^phi IS the target \
             sector; dim T^phi = {} against {} determinants, so only Cauchy interlacing would \
             survive here and the equality below must not be asserted",
            ch.phi,
            ch.target,
            ch.dim,
            want.len()
        );
        let theta = arm_values(ch.tda.as_ref(), SectorReadout::Tda);
        let r = ch.tda.as_ref().expect("a Tda run fills the Tda arm");
        println!(
            "tsd union {:?}: dim {} converged {} iters {} applies {} status {:?} dense {}",
            ch.phi,
            ch.dim,
            r.converged,
            r.iterations,
            r.applies,
            ch.status,
            ch.dense.as_ref().map_or("-".to_string(), spectrum_label),
        );
        assert!(
            r.converged,
            "row {:?}: k = dim was requested and the start block spans the channel, so the block \
             Davidson must exhaust it",
            ch.phi
        );
        let got: Vec<f64> = theta.iter().map(|z| z + lambda).collect();
        assert_multiset_eq(
            &got,
            &want,
            1e-8,
            &format!(
                "channel {:?} levels vs the dense irrep-{:?} spectrum",
                ch.phi, ch.target
            ),
        );
        union.extend(got);
        checked += 1;
    }
    assert!(
        checked >= 1,
        "anti-vacuity: no live channel met the equality precondition, so nothing was compared"
    );
    union.sort_by(|a, b| a.partial_cmp(b).expect("finite level"));
    want_union.sort_by(|a, b| a.partial_cmp(b).expect("finite level"));
    assert_multiset_eq(
        &union,
        &want_union,
        1e-8,
        "the union over the ENUMERATED channels vs the dense levels of every non-reference irrep",
    );
    let own = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, space.target);
    assert!(
        !own.is_empty() && own.len() + want_union.len() == dets_for(D4, F_NUP, F_NDN).len(),
        "bookkeeping: the reference irrep holds {} of the sector's {} determinants and the \
         enumerated channels the remaining {}",
        own.len(),
        dets_for(D4, F_NUP, F_NDN).len(),
        want_union.len()
    );
}

#[test]
fn tsd_all_from_load_screens_and_agrees_with_the_explicit_entry() {
    let (space, pg, t, v, _median_ham, x) = lih_typed_fixture();
    let space = *space;
    let x = x.clone();
    let opts = tda_opts(1);

    let screened = lih_load(true);
    assert!(
        x.space.same_lattice(&screened.space),
        "the reference must live on the load's lattice (d, nup, ndn, pg) — `target` deliberately \
         is NOT compared: a caller who converged in a non-default irrep keeps its own target"
    );
    assert_eq!(x.space.pg, screened.space.pg);
    assert_eq!(pg, &screened.pg.pg);

    let ham = SparseHamiltonian::<f64>::new(space, t, v, None, opts.eps_h);
    println!(
        "tsd from_load: target {:?}, ||grad|| = {:e}, kind {:?}",
        space.target,
        grad_norm_at(&ham, &x),
        screened.pg.kind
    );
    let explicit = sector_response_all(&ham, &x, t, v, None, &opts);
    for ch in &explicit.channels {
        print_row(ch);
    }
    assert!(
        explicit.channels.iter().all(|c| c.label.name.is_none()),
        "the explicit entry with `kind: None` leaves the rendered names unset"
    );

    for screen in [true, false] {
        let load = lih_load(screen);
        let got = sector_response_all_from_load(&load, &x, &opts);
        println!("tsd from_load [screen = {screen}]:");
        for ch in &got.channels {
            print_row(ch);
        }
        assert!(
            got.channels.iter().all(|c| c.label.name.is_some()),
            "[screen = {screen}] the loader entry fills `kind` from load.pg.kind, so every channel \
             comes out NAMED (D-spec §1.2)"
        );
        assert_specs_agree(
            &got,
            &explicit,
            1e-12,
            &format!("from_load(screen = {screen}) vs the explicit entry"),
        );
    }

    let n_live = explicit
        .channels
        .iter()
        .filter(|c| c.status != SectorChannelStatus::Empty)
        .count();
    println!(
        "tsd from_load: {n_live} live channel(s) of {}",
        explicit.channels.len()
    );
    assert!(
        n_live >= 1,
        "anti-vacuity: every channel was Empty, so the three entries agreed about nothing"
    );
}

#[test]
fn tsd_all_channel_set_relates_to_the_channel_response_driver() {
    let (space, _pg, t, v, ham, x) = lih_typed_fixture();
    let space = *space;
    let g = grad_norm_at(ham, x);
    println!(
        "tsd side-by-side: target {:?}, ||grad|| = {g:e}",
        space.target
    );
    assert!(g < 1e-8, "LiH fixture not critical (||grad|| = {g:e})");

    let copts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::All,
            ..ChannelSet::default()
        },
        gs: ChannelGsOpts {
            chi: qntt::Pooled(4),
            max_iter: 20,
            ..ChannelGsOpts::default()
        },
        cone: ConeOpts {
            targets: 1,
            max_added: 0,
            ..ConeOpts::default()
        },
        ..ChannelOpts::default()
    };
    let cspec = channel_response(t, v, space, &copts);
    let e_ref = cspec.channels[cspec.reference]
        .e_gs
        .expect("the reference channel must converge");
    let mut ch_live: Vec<Irrep> = Vec::new();
    for ch in &cspec.channels {
        let gamma = ch.label.irrep;
        println!(
            "  channel_response {gamma:?} ({:?}): status {:?} gap {:?}",
            ch.label.name,
            ch.status,
            ch.e_gs.map(|e| e - e_ref)
        );
        if gamma != space.target && ch.status == ChannelStatus::Ok && ch.e_gs.is_some() {
            ch_live.push(gamma);
        }
    }

    let spec = sector_response_all(ham, x, t, v, None, &tda_opts(1));
    let mut sec_live: Vec<Irrep> = Vec::new();
    for ch in &spec.channels {
        print_row(ch);
        if ch.status != SectorChannelStatus::Empty {
            sec_live.push(ch.target);
        }
    }
    println!(
        "tsd side-by-side: channel_response live {ch_live:?} vs sector_response_all live \
         {sec_live:?}"
    );
    assert!(
        !ch_live.is_empty() && !sec_live.is_empty(),
        "anti-vacuity: one of the two drivers produced no comparable channel at all"
    );
    for gamma in &sec_live {
        assert!(
            ch_live.contains(gamma),
            "the sector driver reached irrep {gamma:?} but channel_response found no state there \
             — the shifted tangent space is a SUBSPACE of the target sector, so a live shifted \
             channel over an uninhabited sector is a defect, not a modelling difference"
        );
    }
    for gamma in &ch_live {
        if sec_live.contains(gamma) {
            continue;
        }
        let row = spec
            .channels
            .iter()
            .find(|c| c.target == *gamma)
            .unwrap_or_else(|| {
                panic!(
                    "irrep {gamma:?} is inhabited but the sector driver produced no row for it at \
                     all — every flux must be enumerated, live or not"
                )
            });
        assert_eq!(
            row.status,
            SectorChannelStatus::Empty,
            "irrep {gamma:?} is inhabited and the sector driver did not reach it, so its row must \
             say Empty (dim T^phi = 0 — the borrowed suffix bases, §1.6), not {:?}",
            row.status
        );
        println!(
            "tsd side-by-side: irrep {gamma:?} is max_ranks-inhabited but dim T^phi = 0 at this \
             reference — the measured LiH case, and the reason set EQUALITY is not the contract"
        );
    }
}

#[test]
fn tsd_all_dense_cap_gates_the_dense_route_and_the_b_weight_spelling() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    println!(
        "tsd dense_cap: sigma_min = {:e}, live = {:?}",
        min_schmidt(&bl, &br),
        live_channels(&space, &bl.ranks)
    );

    let off = sector_response_all(&ham, &x, &t, &v, None, &tda_opts(1));
    for ch in &off.channels {
        print_row(ch);
        assert!(
            ch.dense.is_none(),
            "dense_cap: None must NEVER run the dense route (row {:?}, dim {})",
            ch.phi,
            ch.dim
        );
        match ch.status {
            SectorChannelStatus::Empty => assert!(
                ch.b_weight.is_none(),
                "row {:?}: an Empty row has no b_weight",
                ch.phi
            ),
            _ => assert_eq!(
                b_weight_tag(ch.b_weight.as_ref()),
                "seed",
                "row {:?}: with no dense route the only available spelling is the seed-block \
                 FRACTION (D-spec §1.8), and it must be tagged as one",
                ch.phi
            ),
        }
    }
    let dims: Vec<usize> = off
        .channels
        .iter()
        .filter(|c| c.status != SectorChannelStatus::Empty)
        .map(|c| c.dim)
        .collect();
    let (lo, hi) = (
        *dims.iter().min().expect("at least one live channel"),
        *dims.iter().max().expect("at least one live channel"),
    );
    println!("tsd dense_cap: live dims {dims:?}");
    assert!(
        lo < hi,
        "anti-vacuity: every live channel has dim {lo}, so no threshold can separate them and the \
         gate would be untested"
    );

    let cap = (lo + hi) / 2;
    assert!(
        lo <= cap && cap < hi,
        "the threshold must split the run: {lo} <= {cap} < {hi}"
    );
    let ctx = weingarten_operator(&ham, &x);
    let capped = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            dense_cap: Some(cap),
            ..tda_opts(1)
        },
    );
    let mut fired = 0usize;
    let mut skipped = 0usize;
    for ch in &capped.channels {
        print_row(ch);
        let want = ch.status != SectorChannelStatus::Empty && ch.dim <= cap;
        assert_eq!(
            ch.dense.is_some(),
            want,
            "row {:?} (dim {}, cap {cap}): the dense route must run exactly for a LIVE channel \
             with dim <= cap",
            ch.phi,
            ch.dim
        );
        if ch.status == SectorChannelStatus::Empty {
            assert!(ch.b_weight.is_none());
            continue;
        }
        if !want {
            skipped += 1;
            assert_eq!(
                b_weight_tag(ch.b_weight.as_ref()),
                "seed",
                "row {:?}",
                ch.phi
            );
            match ch.b_weight.as_ref() {
                Some(BWeight::SeedFraction { mean, max, n }) => {
                    println!(
                        "  row {:?}: SeedFraction mean {mean:e} max {max:e} over n = {n} seeds",
                        ch.phi
                    );
                    assert!(
                        *n >= 1,
                        "row {:?}: a SeedFraction over no seeds is not an estimate",
                        ch.phi
                    );
                    assert!(
                        mean <= max || (mean.is_infinite() && max.is_infinite()),
                        "row {:?}: mean {mean} must not exceed max {max}",
                        ch.phi
                    );
                    assert!(
                        !mean.is_nan() && !max.is_nan(),
                        "row {:?}: NaN is not a reported estimate — an A-form that vanishes is \
                         INFINITY by contract (D-spec §1.8)",
                        ch.phi
                    );
                }
                other => panic!("row {:?}: expected a SeedFraction, got {other:?}", ch.phi),
            }
            continue;
        }
        fired += 1;
        assert_eq!(
            b_weight_tag(ch.b_weight.as_ref()),
            "gram",
            "row {:?}: where the dense route ran, the HONEST norm ratio is available and must be \
             reported as GramRatio (D-spec §1.8)",
            ch.phi
        );
        let space_out = space.with_target(space.target.fuse(ch.phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, capped_eps());
        let sctx = SectorCtx::new(&ctx, &h_out, ch.phi);
        let basis = sctx.basis();
        let (a, b) = sctx.grams(&basis);
        let (na, nb) = (frobenius(&a), frobenius(&b));
        let want_ratio = nb / na;
        let Some(BWeight::GramRatio(got_ratio)) = ch.b_weight else {
            unreachable!("tagged `gram` above")
        };
        println!(
            "  row {:?}: ||B||_F/||A||_F = {got_ratio:e} vs hand-rolled {want_ratio:e} \
             (||A||_F = {na:e}, ||B||_F = {nb:e}, basis {})",
            ch.phi,
            basis.len()
        );
        assert!(
            na > 1e-8,
            "row {:?}: anti-vacuity — ||A||_F = {na:e}, so the ratio has no scale",
            ch.phi
        );
        assert!(
            (got_ratio - want_ratio).abs() <= 1e-10 * want_ratio.max(1.0),
            "row {:?}: GramRatio {got_ratio} != ||B||_F/||A||_F {want_ratio} — §1.8 specifies the \
             FROBENIUS ratio of the full Grams; a spectral-norm ratio would differ by up to sqrt(n)",
            ch.phi
        );
    }
    assert!(
        fired >= 1 && skipped >= 1,
        "anti-vacuity: the cap must have fired on {fired} channel(s) and skipped {skipped} — with \
         either at zero the gate is untested"
    );

    let all = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            dense_cap: Some(usize::MAX),
            ..tda_opts(1)
        },
    );
    for ch in &all.channels {
        assert_eq!(
            ch.dense.is_some(),
            ch.status != SectorChannelStatus::Empty,
            "row {:?}: dense_cap = Some(usize::MAX) admits every LIVE channel and no Empty one",
            ch.phi
        );
    }
}

fn capped_eps() -> f64 {
    SectorAllOpts::default().eps_h
}

#[test]
fn tsd_all_threaded_matches_serial() {
    let (space, t, v, ham, x) = d6_minimizer();
    let (bl, br) = tangent_bases(&x);
    let live = live_channels(&space, &bl.ranks);
    println!(
        "tsd threading: sigma_min = {:e}, live = {live:?}",
        min_schmidt(&bl, &br)
    );
    assert!(
        live.iter().filter(|&&p| p != Irrep::IDENTITY).count() >= 2,
        "anti-vacuity: at least two live channels, or the loop forks over nothing"
    );
    let opts = SectorAllOpts {
        readout: SectorAllReadout::Tda,
        k: 2,
        dense_cap: Some(usize::MAX),
        include_reference: true,
        ..SectorAllOpts::default()
    };
    let run = || sector_response_all(&ham, &x, &t, &v, None, &opts);

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build 1-thread rayon pool");
    let serial = serial_pool.install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            1,
            "serial arm did not run in the 1-thread pool"
        );
        run()
    });
    let par_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("build 8-thread rayon pool");
    let par = par_pool.install(|| {
        assert!(
            rayon::current_num_threads() > 1,
            "parallel arm did not run multi-threaded ({} threads)",
            rayon::current_num_threads()
        );
        run()
    });

    for ch in &serial.channels {
        print_row(ch);
    }
    let n_values: usize = serial
        .channels
        .iter()
        .map(|c| arm_values(c.tda.as_ref(), SectorReadout::Tda).len())
        .sum();
    println!(
        "tsd threading: {n_values} theta values across {} rows",
        serial.channels.len()
    );
    assert!(
        n_values >= 4,
        "anti-vacuity: only {n_values} readout values were compared across pools"
    );
    assert!(
        serial.channels.iter().any(|c| matches!(
            c.b_weight,
            Some(BWeight::GramRatio(r)) if r > 1e-3
        )),
        "anti-vacuity: no channel has a B^phi worth forking over (b_weights {:?}) — a \
         matches-serial pin over a B that is identically zero proves nothing",
        serial
            .channels
            .iter()
            .map(|c| c.b_weight)
            .collect::<Vec<_>>()
    );
    assert_specs_agree(&par, &serial, 1e-12, "8-thread pool vs 1-thread pool");
    let (ls, lp) = (
        serial.levels(SectorReadout::Tda),
        par.levels(SectorReadout::Tda),
    );
    assert_eq!(
        ls.len(),
        lp.len(),
        "the merge produced a different number of levels across pools"
    );
    for (j, (a, b)) in ls.iter().zip(lp.iter()).enumerate() {
        assert_eq!((a.channel, a.k), (b.channel, b.k), "level {j} back-index");
        assert!(
            (a.value - b.value).abs() <= 1e-12 * a.value.abs().max(1.0),
            "level {j}: {} vs {}",
            a.value,
            b.value
        );
    }
    assert_eq!(
        serial.dim_total(),
        par.dim_total(),
        "dim_total is structural (shapes only) and cannot depend on the pool"
    );
}

#[test]
fn tsd_all_indefinite_channel_is_a_status_not_a_panic() {
    let pg = synth_pg();
    let gamma0 = Irrep::new(1);
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, gamma0);
    let (t, v) = synth_tv(pg);
    let (x, e0, _levels) = exact_typed_ground(space, &t, &v);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);

    let lower = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, Irrep::IDENTITY);
    assert!(
        lower[0] < e0 - 1e-3,
        "fixture premise: irrep 0's ground ({}) must lie BELOW the reference ({e0})",
        lower[0]
    );
    let want = lower[0] - e0;
    let (bl, br) = tangent_bases(&x);
    println!(
        "tsd indefinite: E0 = {e0}, expected negative = {want}, sigma_min = {:e}, ||grad|| = {:e}",
        min_schmidt(&bl, &br),
        grad_norm_at(&ham, &x)
    );

    let tda = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            dense_cap: Some(usize::MAX),
            ..tda_opts(3)
        },
    );
    let mut hit = 0usize;
    for ch in &tda.channels {
        print_row(ch);
        if ch.target != Irrep::IDENTITY {
            continue;
        }
        hit += 1;
        assert_ne!(
            ch.status,
            SectorChannelStatus::Empty,
            "fixture premise: the channel reaching the GLOBAL ground's irrep must be live here"
        );
        let r = ch.tda.as_ref().expect("a Tda run fills the Tda arm");
        let theta = arm_values(Some(r), SectorReadout::Tda);
        assert!(
            theta.iter().all(|z| z.is_finite()),
            "the readout must not produce NaN/inf: {theta:?}"
        );
        assert!(
            theta[0] < 0.0,
            "the lowest A^phi eigenvalue must be negative in a channel holding a lower state: \
             {theta:?}"
        );
        assert!(
            (theta[0] - want).abs() <= 1e-6 * want.abs().max(1.0),
            "at full rank the negative must be exactly E_0(lower) - E_0(ref) = {want}, got {}",
            theta[0]
        );
        match &ch.status {
            SectorChannelStatus::Indefinite {
                n_negative,
                certified,
                n_negative_signature,
                metric_indefinite,
            } => {
                println!(
                    "tsd indefinite [Tda]: Indefinite {{ n_negative: {n_negative}, certified: \
                     {certified}, n_negative_signature: {n_negative_signature}, \
                     metric_indefinite: {metric_indefinite} }}, witnesses {:?}",
                    r.negatives()
                );
                assert_eq!(
                    (*n_negative_signature, *metric_indefinite),
                    (0, false),
                    "a Tda-only run carries no Krein signature and no metric report"
                );
                assert!(
                    *n_negative >= 1,
                    "an Indefinite status must count the negatives it saw"
                );
                assert_eq!(
                    *n_negative,
                    theta.iter().filter(|z| **z < 0.0).count(),
                    "n_negative must count the REPORTED negative values"
                );
                assert!(
                    *certified,
                    "a negative direction must come with a CERTIFIED fresh-apply witness, \
                     and at an exact eigenvector the fresh apply reproduces the Ritz value — \
                     witnesses {:?}",
                    r.negatives()
                );
                assert_eq!(
                    *certified,
                    r.all_negatives_certified(),
                    "the status's `certified` IS SectorResponse::all_negatives_certified (D-spec \
                     §1.4)"
                );
            }
            other => panic!(
                "a channel whose lowest A^phi eigenvalue is {} must be reported Indefinite, not \
                 {other:?} — negative directions are information (README.md)",
                theta[0]
            ),
        }
        let low = tda
            .levels(SectorReadout::Tda)
            .into_iter()
            .find(|l| l.channel == tda.channels.iter().position(|c| c.phi == ch.phi).unwrap())
            .expect("the indefinite channel contributes levels");
        assert!(
            low.energy < tda.reference.lambda,
            "the lowest level of a channel holding a lower state must sit BELOW lambda: {} vs {}",
            low.energy,
            tda.reference.lambda
        );
        let dense = ch.dense.as_ref().expect("dense_cap admits this channel");
        println!(
            "tsd indefinite [Tda]: dense route says {}",
            spectrum_label(dense)
        );
        assert!(
            !matches!(dense, SymplecticSpectrum::Stable { .. }),
            "the dense Williamson reduction cannot be Stable in a channel holding a state below \
             the reference: {}",
            spectrum_label(dense)
        );
    }
    assert_eq!(
        hit, 1,
        "fixture premise: exactly one enumerated channel targets irrep 0 (the global ground's)"
    );

    let dim1 = tangent_dim_grids(&space, &bl.ranks, Irrep::new(1));
    assert!(
        dim1 > 0,
        "fixture premise: the phi = 1 channel is live (dim {dim1})"
    );
    for kernel in [RpaKernel::signed(RpaMetric::Auto), RpaKernel::Paired] {
        let signed = !matches!(kernel, RpaKernel::Paired);
        let rpa = sector_response_all(
            &ham,
            &x,
            &t,
            &v,
            None,
            &SectorAllOpts {
                readout: SectorAllReadout::Rpa,
                k: dim1,
                dense_cap: Some(usize::MAX),
                solver: SectorOpts {
                    kernel,
                    ..SectorOpts::default()
                },
                ..SectorAllOpts::default()
            },
        );
        let mut hit2 = 0usize;
        for (ci, ch) in rpa.channels.iter().enumerate() {
            print_row(ch);
            if ch.target != Irrep::IDENTITY {
                continue;
            }
            hit2 += 1;
            let r = ch.rpa.as_ref().expect("an Rpa run fills the Rpa arm");
            println!(
                "tsd indefinite [Rpa]: status {:?}, omega2 {:?}, signature {:?}, converged {}, \
                 applies {}, flagged {}, metric in spectrum {:?}",
                ch.status,
                r.omega2(),
                krein_signs(r.rpa()),
                r.converged,
                r.applies,
                r.metric_indefinite(),
                match &r.spectrum {
                    qntt::solvers::hess::SectorSpectrum::Rpa(rr) => format!("{:?}", rr.side()),
                    _ => "-".to_string(),
                }
            );
            assert!(
                r.omega().is_none(),
                "omega() must be None wherever a resonance or an indefinite block is in play — a \
                 `Vec<f64>` of omega cannot carry `i·sqrt|omega^2|`"
            );
            let w = r.omega2().expect("an Rpa readout publishes omega2()");
            assert!(
                !w.is_empty() && w.iter().all(|z| z.is_finite()),
                "the signed readout answers here: a non-empty, finite omega^2 ({w:?})"
            );
            assert_eq!(
                r.metric_indefinite(),
                signed,
                "{kernel:?}: the Ii metric W = A - B = A is indefinite at this exact reference: \
                 FLAGGED on the signed kernel only"
            );
            assert!(r.anomalous(), "{kernel:?}: the de-excitation is an anomaly");
            let n_sig = krein_signs(r.rpa()).iter().filter(|&&s| s == -1).count();
            match &ch.status {
                SectorChannelStatus::Indefinite {
                    n_negative_signature,
                    metric_indefinite,
                    ..
                } => {
                    assert_eq!(
                        *metric_indefinite, signed,
                        "the status carries the solve's flag"
                    );
                    assert!(n_sig >= 1, "{kernel:?}: the lower state is an s = -1 entry");
                    assert_eq!(
                        *n_negative_signature, n_sig,
                        "the status counts the Rpa entries with s = -1"
                    );
                }
                other => panic!(
                    "{kernel:?}: a solve holding a de-excitation must be reported Indefinite, not \
                     {other:?}"
                ),
            }
            match &r.spectrum {
                qntt::solvers::hess::SectorSpectrum::Rpa(rr) => {
                    assert_ne!(
                        rr.side(),
                        RpaMetric::Auto,
                        "wherever applies > 0 the metric is RESOLVED, never the request Auto"
                    );
                    assert_eq!(
                        rr.side(),
                        RpaMetric::Ii,
                        "the sector readout runs Ii for Auto (no probe, no retry); Ii is also the \
                         paired readout's convention"
                    );
                }
                _ => panic!("an Rpa readout must carry an Rpa spectrum"),
            }
            let from_this: Vec<_> = rpa
                .levels(SectorReadout::Rpa)
                .into_iter()
                .filter(|l| l.channel == ci)
                .collect();
            assert_eq!(
                from_this.len(),
                w.len(),
                "every omega^2 entry of the channel stays a level"
            );
            if signed {
                assert!(
                    from_this.iter().all(|l| l.energy.is_nan()),
                    "a FLAGGED solve publishes no energy on any entry (Q6): {from_this:?}"
                );
            } else {
                let signs = krein_signs(r.rpa());
                for l in &from_this {
                    assert_eq!(
                        l.energy.is_nan(),
                        signs[l.k] == -1,
                        "paired: entry {} (s = {}) energy {}",
                        l.k,
                        signs[l.k],
                        l.energy
                    );
                }
            }
        }
        assert_eq!(hit2, 1, "fixture premise: one channel targets irrep 0");
    }
}

#[test]
fn tsd_all_dim_total_matches_the_untyped_tangent_dimension() {
    for (label, (space, t, v, ham, x)) in [
        (
            "fx4 (2,2) truncated, two empty channels",
            truncated_reference(),
        ),
        ("d6 minimizer, all four channels live", d6_minimizer()),
    ] {
        let (bl, br) = tangent_bases(&x);
        let x_c1 = x.forget_point_group();
        let (bl_c1, _br_c1) = tangent_bases(&x_c1);
        let want = tangent_dim(&bl_c1);
        let spec = sector_response_all(&ham, &x, &t, &v, None, &tda_opts(1));
        println!(
            "tsd dim_total [{label}]: sigma_min = {:e}, live = {:?}, reference tangent_dim = {}, \
             untyped tangent_dim = {want}",
            min_schmidt(&bl, &br),
            live_channels(&space, &bl.ranks),
            spec.reference.tangent_dim
        );
        for ch in &spec.channels {
            print_row(ch);
        }
        let sum: usize = spec.channels.iter().map(|c| c.dim).sum();
        assert_eq!(
            spec.dim_total(),
            sum + spec.reference.tangent_dim,
            "[{label}] dim_total() is the enumerated channels' dims plus the reference's own \
             (D-spec §1.7)"
        );
        assert_eq!(
            spec.dim_total(),
            want,
            "[{label}] the shifted spaces are the irrep BLOCKS of one untyped tangent space, so \
             sum_phi dim T^phi + tangent_dim(x) must be tangent_dim(x.forget_point_group()) — got \
             {} against {want}",
            spec.dim_total()
        );
        let n_empty = spec.empty_channels().count();
        println!("tsd dim_total [{label}]: {n_empty} empty channel(s)");
        if label.starts_with("fx4") {
            assert!(
                n_empty >= 2,
                "[{label}] anti-vacuity: this leg exists to check that EMPTY rows contribute 0 to \
                 the sum, and it found {n_empty} of them"
            );
        } else {
            assert_eq!(
                n_empty, 0,
                "[{label}] anti-vacuity: this leg is the all-live one"
            );
        }
    }
}

#[test]
fn tsd_all_solver_readout_field_is_ignored() {
    let (space, t, v, ham, x) = truncated_reference();
    let (bl, br) = tangent_bases(&x);
    println!(
        "tsd solver.readout: sigma_min = {:e}, live = {:?}",
        min_schmidt(&bl, &br),
        live_channels(&space, &bl.ranks)
    );
    let base = SectorAllOpts {
        readout: SectorAllReadout::Both,
        k: 1,
        ..SectorAllOpts::default()
    };
    let mut runs = Vec::new();
    for stale in [
        SectorReadout::Tda,
        SectorReadout::Rpa,
        SectorOpts::default().readout,
    ] {
        let opts = SectorAllOpts {
            solver: SectorOpts {
                readout: stale,
                ..base.solver.clone()
            },
            ..base.clone()
        };
        let spec = sector_response_all(&ham, &x, &t, &v, None, &opts);
        println!("tsd solver.readout [stale = {stale:?}]:");
        for ch in &spec.channels {
            print_row(ch);
        }
        runs.push((stale, spec));
    }
    let (_, first) = &runs[0];
    assert!(
        first
            .channels
            .iter()
            .any(|c| c.tda.is_some() && c.rpa.is_some()),
        "anti-vacuity: a Both run must fill both arms on at least one live channel — with the \
         field passed through, this is where the panic would have happened instead"
    );
    for (stale, spec) in &runs[1..] {
        assert_specs_agree(
            spec,
            first,
            1e-12,
            &format!("solver.readout = {stale:?} vs {:?}", runs[0].0),
        );
    }

    let rpa_only = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            readout: SectorAllReadout::Rpa,
            ..base.clone()
        },
    );
    for (ch_both, ch_rpa) in first.channels.iter().zip(rpa_only.channels.iter()) {
        assert_eq!(ch_both.phi, ch_rpa.phi);
        let a = arm_values(ch_both.rpa.as_ref(), SectorReadout::Rpa);
        let b = arm_values(ch_rpa.rpa.as_ref(), SectorReadout::Rpa);
        let tag = format!("row {:?}: Both's Rpa arm vs an Rpa-only run", ch_both.phi);
        report_bits(&a, &b, &tag);
        assert_close(&a, &b, 1e-12, &tag);
    }
}

#[test]
#[should_panic(expected = "sector_response_all: k == 0")]
fn tsd_all_k_zero_is_refused() {
    let (_space, t, v, ham, x) = truncated_reference();
    let _ = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            k: 0,
            ..SectorAllOpts::default()
        },
    );
}
