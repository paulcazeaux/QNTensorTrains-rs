// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::QnSpace;
use qntt::solvers::hess::{
    RpaKernel, RpaMetric, RpaReadout, RpaReport, SympLobpcgOpts, SympPrecond,
    SymplecticDavidsonOpts, SymplecticLowestInfo, symplectic_lowest_lobp4dcg,
    symplectic_lowest_signed_davidson, weingarten_operator,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const GUARD_RECOMMENDED: f64 = 1e-5;

const GUARD_ALWAYS: f64 = 1e300;

fn assert_run_identical(
    tag: &str,
    (om_a, info_a): (&[f64], &SymplecticLowestInfo),
    (om_b, info_b): (&[f64], &SymplecticLowestInfo),
) {
    assert_eq!(om_a.len(), om_b.len(), "[{tag}] ω length");
    for (j, (x, y)) in om_a.iter().zip(om_b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "[{tag}] ω[{j}] not bit-identical: {x:.17e} vs {y:.17e}"
        );
    }
    assert_eq!(
        info_a.residuals.len(),
        info_b.residuals.len(),
        "[{tag}] residual history length"
    );
    for (j, (x, y)) in info_a
        .residuals
        .iter()
        .zip(info_b.residuals.iter())
        .enumerate()
    {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "[{tag}] residual[{j}] not bit-identical: {x:.17e} vs {y:.17e}"
        );
    }
    assert_eq!(info_a.iterations, info_b.iterations, "[{tag}] built count");
    assert_eq!(info_a.applies, info_b.applies, "[{tag}] apply count");
    assert_eq!(info_a.converged, info_b.converged, "[{tag}] converged flag");
}

fn runs_differ(
    (om_a, info_a): (&[f64], &SymplecticLowestInfo),
    (om_b, info_b): (&[f64], &SymplecticLowestInfo),
) -> bool {
    info_a.iterations != info_b.iterations
        || info_a.applies != info_b.applies
        || info_a.converged != info_b.converged
        || om_a.len() != om_b.len()
        || om_a
            .iter()
            .zip(om_b.iter())
            .any(|(x, y)| x.to_bits() != y.to_bits())
        || info_a.residuals.len() != info_b.residuals.len()
        || info_a
            .residuals
            .iter()
            .zip(info_b.residuals.iter())
            .any(|(x, y)| x.to_bits() != y.to_bits())
}

fn tripped_at(tag: &str, info: &SymplecticLowestInfo) -> Option<usize> {
    info.diag
        .as_ref()
        .unwrap_or_else(|| panic!("[{tag}] the block kernels report SympDiagnostics"))
        .guard_tripped
}

fn assert_readout_identical(tag: &str, a: &RpaReadout, b: &RpaReadout) {
    let (w2a, w2b) = (a.omega2(), b.omega2());
    assert_eq!(w2a.len(), w2b.len(), "[{tag}] ω² length");
    for (j, (x, y)) in w2a.iter().zip(w2b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "[{tag}] ω²[{j}] not bit-identical: {x:.17e} vs {y:.17e}"
        );
    }
    let (ra, rb) = (a.residuals(), b.residuals());
    assert_eq!(ra.len(), rb.len(), "[{tag}] residual length");
    for (j, (x, y)) in ra.iter().zip(rb.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "[{tag}] residual[{j}] not bit-identical: {x:.17e} vs {y:.17e}"
        );
    }
    assert_eq!(a.iterations, b.iterations, "[{tag}] built count");
    assert_eq!(a.applies, b.applies, "[{tag}] apply count");
    assert_eq!(a.converged, b.converged, "[{tag}] converged flag");
}

fn readouts_differ(a: &RpaReadout, b: &RpaReadout) -> bool {
    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    a.iterations != b.iterations
        || a.applies != b.applies
        || a.converged != b.converged
        || bits(a.omega2()) != bits(b.omega2())
        || bits(a.residuals()) != bits(b.residuals())
}

fn readout_tripped_at(tag: &str, so: &RpaReadout) -> Option<usize> {
    let RpaReport::Signed { diag, .. } = &so.report else {
        panic!("[{tag}] the signed kernel must have run: {:?}", so.report);
    };
    diag.as_ref()
        .unwrap_or_else(|| panic!("[{tag}] the block kernels report SympDiagnostics"))
        .guard_tripped
}

#[test]
fn tg_guard_fires() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let k = 4usize;
    let nseed = k + 2;

    let arm = SympPrecond::DiagUnshifted { kappa: 0.0 };
    let run = |precond: SympPrecond, guard_survival: f64| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81F0);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, nseed, &mut rng);
        let opts = SymplecticDavidsonOpts {
            precond,
            nwarm: 0,
            guard_survival,
            instrument: true,
            kernel: RpaKernel::signed(RpaMetric::Ii),
            ..SymplecticDavidsonOpts::default()
        };
        symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts).0
    };

    let guarded = run(arm, GUARD_ALWAYS);
    let off = run(SympPrecond::Off, GUARD_ALWAYS);
    let unguarded = run(arm, 0.0);

    let trip = readout_tripped_at("guarded", &guarded);
    println!(
        "tg_guard_fires: guarded built={} applies={} trip={:?} | off built={} applies={} | \
         unguarded built={} applies={}",
        guarded.iterations,
        guarded.applies,
        trip,
        off.iterations,
        off.applies,
        unguarded.iterations,
        unguarded.applies,
    );

    let n = trip.expect(
        "the guard must fire: at `nwarm = 0` the trailing window (= the block size) is already \
         full from the warm-start block when the first correction is preconditioned, and no finite \
         median clears a 1e300 threshold. `None` here means either the check never runs on the \
         preconditioned path or the window/threshold predicate is wired the wrong way round",
    );
    assert!(
        n >= 1 && n <= nseed,
        "the earliest possible trip is at the end of the warm-start block: guard_tripped = \
         Some({n}), seeds = {nseed}"
    );

    assert_readout_identical("guard-trip-vs-off", &guarded, &off);
    assert_eq!(
        readout_tripped_at("off", &off),
        None,
        "the `Off` lane is handed no closure at all, so no threshold may ever record a trip"
    );

    assert!(
        readouts_differ(&unguarded, &off),
        "the unguarded D0 arm is indistinguishable from Off at this fixture, so the guard-trip \
         identity proves nothing — pick a fixture/arm where the preconditioner has an effect"
    );
    assert_eq!(
        readout_tripped_at("unguarded", &unguarded),
        None,
        "guard_survival = 0.0 must never trip"
    );
}

#[test]
fn tg_guard_fires_lobp() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let k = 4usize;
    let nseed = k + 2;

    let arm = SympPrecond::DiagUnshifted { kappa: 0.0 };
    let run = |precond: SympPrecond, guard_survival: f64| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81F1);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, nseed, &mut rng);
        let opts = SympLobpcgOpts {
            precond,
            nwarm: 0,
            guard_survival,
            instrument: true,
            ..SympLobpcgOpts::default()
        };
        symplectic_lowest_lobp4dcg(&ctx, &v0s, k, &opts).expect("stable d=4 fixture")
    };

    let (om_guarded, info_guarded) = run(arm, GUARD_ALWAYS);
    let (om_off, info_off) = run(SympPrecond::Off, GUARD_ALWAYS);
    let (om_unguarded, info_unguarded) = run(arm, 0.0);

    let trip = tripped_at("guarded", &info_guarded);
    println!(
        "tg_guard_fires_lobp: guarded built={} applies={} trip={:?} | off built={} applies={} | \
         unguarded built={} applies={}",
        info_guarded.iterations,
        info_guarded.applies,
        trip,
        info_off.iterations,
        info_off.applies,
        info_unguarded.iterations,
        info_unguarded.applies,
    );
    let n = trip.expect("the ELOBP4dCG guard must fire under a 1e300 threshold at nwarm = 0");
    assert!(
        n >= 1 && n <= nseed,
        "the earliest possible trip is at the end of the warm-start block: guard_tripped = \
         Some({n}), seeds = {nseed} (the lobp window is the ACCEPTED seed count, so `n` may be \
         below `nseed` if a seed was dropped)"
    );
    assert_run_identical(
        "lobp-guard-trip-vs-off",
        (&om_guarded, &info_guarded),
        (&om_off, &info_off),
    );
    assert_eq!(tripped_at("off", &info_off), None);
    assert!(
        runs_differ((&om_unguarded, &info_unguarded), (&om_off, &info_off)),
        "the unguarded D0 arm is indistinguishable from Off in the ELOBP4dCG kernel at this \
         fixture, so the guard-trip identity proves nothing"
    );
}

#[test]
fn tg_guard_silent_when_off() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let k = 4usize;

    let run_dav = |guard: Option<f64>| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81F2);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
        let opts = match guard {
            None => SymplecticDavidsonOpts {
                precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
                kernel: RpaKernel::signed(RpaMetric::Ii),
                ..SymplecticDavidsonOpts::default()
            },
            Some(g) => SymplecticDavidsonOpts {
                precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
                guard_survival: g,
                kernel: RpaKernel::signed(RpaMetric::Ii),
                ..SymplecticDavidsonOpts::default()
            },
        };
        symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts).0
    };
    let zero = run_dav(Some(0.0));
    let omitted = run_dav(None);
    assert_eq!(
        readout_tripped_at("guard-zero", &zero),
        None,
        "guard_survival = 0.0 is OFF: no trip may be recorded"
    );
    assert_eq!(readout_tripped_at("guard-omitted", &omitted), None);
    assert_readout_identical("guard-zero-vs-omitted", &zero, &omitted);

    let run_off = |guard: f64| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81F3);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
        let opts = SymplecticDavidsonOpts {
            precond: SympPrecond::Off,
            guard_survival: guard,
            kernel: RpaKernel::signed(RpaMetric::Ii),
            ..SymplecticDavidsonOpts::default()
        };
        symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts).0
    };
    let off0 = run_off(0.0);
    let offg = run_off(GUARD_ALWAYS);
    assert_eq!(readout_tripped_at("off-armed", &offg), None);
    assert_readout_identical("off-lane-armed", &off0, &offg);

    let run_lobp = |guard: Option<f64>| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81F4);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
        let opts = match guard {
            None => SympLobpcgOpts {
                precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
                ..SympLobpcgOpts::default()
            },
            Some(g) => SympLobpcgOpts {
                precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
                guard_survival: g,
                ..SympLobpcgOpts::default()
            },
        };
        symplectic_lowest_lobp4dcg(&ctx, &v0s, k, &opts).expect("stable d=4 fixture")
    };
    let (om_lz, info_lz) = run_lobp(Some(0.0));
    let (om_lo, info_lo) = run_lobp(None);
    assert_eq!(tripped_at("lobp-guard-zero", &info_lz), None);
    assert_run_identical(
        "lobp-guard-zero-vs-omitted",
        (&om_lz, &info_lz),
        (&om_lo, &info_lo),
    );
}

#[test]
fn tg_guard_no_false_fire() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let k = 4usize;

    let (arm, precond) = ("d0", SympPrecond::DiagUnshifted { kappa: 0.0 });
    {
        let run = |guard: f64| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81F5);
            let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
            let opts = SymplecticDavidsonOpts {
                precond,
                guard_survival: guard,
                instrument: true,
                kernel: RpaKernel::signed(RpaMetric::Ii),
                ..SymplecticDavidsonOpts::default()
            };
            symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts).0
        };
        let armed = run(GUARD_RECOMMENDED);
        let free = run(0.0);
        let min_surv = match &armed.report {
            RpaReport::Signed { diag: Some(d), .. } => {
                d.survival.iter().cloned().fold(f64::INFINITY, f64::min)
            }
            _ => f64::NAN,
        };
        println!(
            "tg_guard_no_false_fire[{arm}]: built={} converged={} min survival={min_surv:.6e} \
             threshold={GUARD_RECOMMENDED:.1e} trip={:?}",
            armed.iterations,
            armed.converged,
            readout_tripped_at(arm, &armed)
        );
        assert_eq!(
            readout_tripped_at(arm, &armed),
            None,
            "[{arm}] the guard must not fire on a healthy fixture at the recommended threshold \
             (min survival over the whole run was {min_surv:.6e}, and the guarded statistic is a \
             trailing MEDIAN, which is larger still)"
        );
        assert_readout_identical(&format!("{arm}-armed-vs-free"), &armed, &free);
    }
}
