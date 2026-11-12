// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::solvers::hess::{
    RpaKernel, RpaMetric, RpaReadout, RpaReport, SympDiagnostics, SympLobpcgOpts, SympPrecond,
    SymplecticDavidsonOpts, symplectic_lowest_lobp4dcg, symplectic_lowest_signed_davidson,
    weingarten_operator,
};
use qntt::{QnSpace, SparseHamiltonian, TtVector};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[allow(clippy::type_complexity)]
fn d4_fixture() -> (
    QnSpace,
    SparseHamiltonian<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
) {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    (space, ham, x, base_l, base_r)
}

fn signed_diag(so: &RpaReadout) -> &SympDiagnostics {
    let RpaReport::Signed { diag, .. } = &so.report else {
        panic!("the signed kernel must have run: {:?}", so.report);
    };
    diag.as_ref()
        .expect("the block Davidson kernel reports SympDiagnostics")
}

fn assert_diag_sane(tag: &str, diag: &SympDiagnostics, iterations: usize, instrumented: bool) {
    assert_eq!(
        diag.survival.len(),
        iterations,
        "[{tag}] one survival ratio per ACCEPTED basis vector: survival.len()={} vs \
         iterations={iterations}",
        diag.survival.len()
    );
    for (i, &s) in diag.survival.iter().enumerate() {
        assert!(
            s.is_finite() && s > 0.0,
            "[{tag}] survival[{i}] = {s:.6e} is not a positive finite ratio — an accepted push \
             cannot have had zero g-norm before or after its MGS"
        );
    }
    for (i, &a) in diag.collapse_angle.iter().enumerate() {
        assert!(
            a.is_finite() && (0.0..=1.0 + 1e-12).contains(&a),
            "[{tag}] collapse_angle[{i}] = {a:.6e} is not a normalized |cos| in [0, 1]"
        );
    }
    assert!(
        diag.outer_iterations >= 1,
        "[{tag}] a converged run performed at least one outer (Rayleigh-Ritz) iteration"
    );
    if instrumented {
        assert_eq!(
            diag.gram_min_eig.len(),
            diag.outer_iterations,
            "[{tag}] one basis-Gram λ_min per outer iteration: gram_min_eig.len()={} vs \
             outer_iterations={}",
            diag.gram_min_eig.len(),
            diag.outer_iterations
        );
        for (i, &g) in diag.gram_min_eig.iter().enumerate() {
            assert!(
                g > 0.99,
                "[{tag}] gram_min_eig[{i}] = {g:.6e}: the kernels ASSUME the basis W-Gram is I, \
                 so λ_min ≈ 1 on a healthy fixture (signed — a small or negative value is the \
                 block-Gram degradation signature, not a scale artifact)"
            );
        }
    } else {
        assert!(
            diag.gram_min_eig.is_empty(),
            "[{tag}] the m²-dot basis Gram is gated behind `instrument`; got {} entries with the \
             flag off",
            diag.gram_min_eig.len()
        );
    }
}

#[test]
fn ti_diag_present_and_sane() {
    let (space, ham, x, base_l, base_r) = d4_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81C0);
    let k = 4usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    let opts = SymplecticDavidsonOpts {
        precond: SympPrecond::Off,
        instrument: true,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
    assert!(
        so.converged,
        "healthy fixture must converge on the Off lane"
    );
    let omega = so
        .omega()
        .expect("a converged stable solve publishes its ω");
    assert_eq!(omega.len(), k);
    let diag = signed_diag(&so);
    println!(
        "ti_diag_present_and_sane: built={} applies={} outer={} restarts={} drops r/w/z/t={}/{}/{}/{} \
         angles={} gram_min={:?}",
        so.iterations,
        so.applies,
        diag.outer_iterations,
        diag.restarts,
        diag.drops_rank,
        diag.drops_wnull,
        diag.drops_zero,
        diag.drops_triple,
        diag.collapse_angle.len(),
        diag.gram_min_eig
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
    );
    assert_diag_sane("davidson", diag, so.iterations, true);
    assert_eq!(
        diag.restarts, 0,
        "at the auto subspace bound this fixture exits by exhaustion — no thick restart can fire"
    );
}

#[test]
fn ti_diag_observation_only() {
    let (space, ham, x, base_l, base_r) = d4_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let k = 4usize;
    let run = |instrument: bool| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81C1);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
        let opts = SymplecticDavidsonOpts {
            precond: SympPrecond::Off,
            instrument,
            kernel: RpaKernel::signed(RpaMetric::Ii),
            ..Default::default()
        };
        symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts).0
    };
    let on = run(true);
    let off = run(false);

    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    assert_eq!(
        bits(on.omega2()),
        bits(off.omega2()),
        "ω² must be bit-identical with instrument on/off"
    );
    assert_eq!(
        bits(on.residuals()),
        bits(off.residuals()),
        "residuals must be bit-identical with instrument on/off"
    );
    assert_eq!(on.iterations, off.iterations, "built count");
    assert_eq!(on.applies, off.applies, "apply count");
    assert_eq!(on.converged, off.converged, "converged flag");

    let diag_on = signed_diag(&on);
    let diag_off = signed_diag(&off);
    assert_diag_sane("instr-on", diag_on, on.iterations, true);
    assert_diag_sane("instr-off", diag_off, off.iterations, false);
    assert_eq!(diag_on.outer_iterations, diag_off.outer_iterations);
    assert_eq!(diag_on.restarts, diag_off.restarts);
    assert_eq!(diag_on.drops_rank, diag_off.drops_rank);
    assert_eq!(diag_on.drops_wnull, diag_off.drops_wnull);
    assert_eq!(diag_on.drops_zero, diag_off.drops_zero);
    assert_eq!(diag_on.drops_triple, diag_off.drops_triple);
    assert_eq!(diag_on.survival, diag_off.survival);
    assert_eq!(diag_on.collapse_angle, diag_off.collapse_angle);
}

#[test]
fn ti_restart_counter_fires() {
    let (space, ham, x, base_l, base_r) = d4_fixture();
    let ctx = weingarten_operator(&ham, &x);

    let (k, b) = (2usize, 2usize);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81C2);
    let v0s = seed_block::<f64>(space, &base_l, &base_r, b, &mut rng);
    let opts = SymplecticDavidsonOpts {
        max_subspace: k + b + 1,
        precond: SympPrecond::Off,
        instrument: true,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
    let diag = signed_diag(&so);
    println!(
        "ti_restart_counter_fires[forced]: built={} converged={} outer={} restarts={}",
        so.iterations, so.converged, diag.outer_iterations, diag.restarts
    );
    assert!(
        diag.restarts >= 1,
        "a `max_subspace` at the floor must fire at least one thick restart (got {})",
        diag.restarts
    );
    assert!(
        diag.restarts <= diag.outer_iterations,
        "a restart fires inside an outer iteration: restarts={} > outer={}",
        diag.restarts,
        diag.outer_iterations
    );
    assert_diag_sane("forced-restart", diag, so.iterations, true);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81C3);
    let k2 = 4usize;
    let v0s2 = seed_block::<f64>(space, &base_l, &base_r, k2 + 2, &mut rng);
    let opts2 = SymplecticDavidsonOpts {
        precond: SympPrecond::Off,
        instrument: true,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so2, _) = symplectic_lowest_signed_davidson(&ctx, &v0s2, k2, &opts2);
    let diag2 = signed_diag(&so2);
    assert_eq!(
        diag2.restarts, 0,
        "the auto subspace bound exceeds space_dim here — no restart may be reported"
    );
}

#[test]
fn ti_dropcount_nonsilent() {
    let (space, ham, x, base_l, base_r) = d4_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81C4);
    let k = 4usize;
    let mut v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    v0s[1] = v0s[0].clone();
    let opts = SymplecticDavidsonOpts {
        precond: SympPrecond::Off,
        instrument: true,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
    let diag = signed_diag(&so);
    println!(
        "ti_dropcount_nonsilent: built={} drops r/w/z/t={}/{}/{}/{}",
        so.iterations, diag.drops_rank, diag.drops_wnull, diag.drops_zero, diag.drops_triple
    );
    assert!(
        diag.drops_rank >= 1,
        "the duplicated seed must be COUNTED as a rank-revealing drop (got drops_rank={}, \
         drops_wnull={}, drops_zero={})",
        diag.drops_rank,
        diag.drops_wnull,
        diag.drops_zero
    );
    assert_eq!(
        so.iterations,
        diag.survival.len(),
        "a dropped candidate is not an accepted basis vector"
    );
    assert_diag_sane("duplicate-seed", diag, so.iterations, true);
}

#[test]
fn ti_lobp_diag_present() {
    let (space, ham, x, base_l, base_r) = d4_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81C5);
    let k = 4usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    let opts = SympLobpcgOpts {
        precond: SympPrecond::Off,
        instrument: true,
        ..SympLobpcgOpts::default()
    };
    let (omega, info) =
        symplectic_lowest_lobp4dcg(&ctx, &v0s, k, &opts).expect("stable d=4 fixture");
    assert!(
        info.converged,
        "healthy fixture must converge on the Off lane"
    );
    assert_eq!(omega.len(), k);
    let diag = info
        .diag
        .as_ref()
        .expect("the ELOBP4dCG kernel reports SympDiagnostics");
    println!(
        "ti_lobp_diag_present: built={} applies={} outer={} drops r/w/z/t={}/{}/{}/{}",
        info.iterations,
        info.applies,
        diag.outer_iterations,
        diag.drops_rank,
        diag.drops_wnull,
        diag.drops_zero,
        diag.drops_triple
    );
    assert_diag_sane("lobp4dcg", diag, info.iterations, true);
    assert_eq!(
        diag.restarts, 0,
        "the ELOBP4dCG kernel has no restart — the counter must stay 0"
    );
}
