// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::QnSpace;
use qntt::core::tangent::tangent_basis;
use qntt::solvers::hess::{ConeOpts, ConeResponse, cone_response};

fn level_errors(res: &ConeResponse, exact: &[f64]) -> Vec<f64> {
    res.energies[1..]
        .iter()
        .zip(exact[1..].iter())
        .map(|(e, x)| e - x)
        .collect()
}

fn worst_best(errs: &[f64]) -> (f64, f64) {
    (
        errs.iter().copied().fold(0.0f64, f64::max),
        errs.iter().copied().fold(f64::INFINITY, f64::min),
    )
}

fn fmt(errs: &[f64]) -> String {
    errs.iter()
        .map(|e| format!("{e:.3e}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn t8p_per_target_budget_recovers_starved_targets() {
    const CONV_TOL: f64 = 5e-3;
    let space = QnSpace::new(5, 2, 2);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let exact = exact_config_spectrum(space, &ham);

    for (k, greedy_floor) in [(24usize, 5e-5f64), (28, 5e-6)] {
        let run = |conv_tol: f64| {
            cone_response(
                &ham,
                &x,
                &ConeOpts {
                    targets: 6,
                    max_added: k,
                    conv_tol,
                    ..Default::default()
                },
            )
        };
        let greedy = run(0.0);
        let fair = run(CONV_TOL);
        let (ge, fe) = (level_errors(&greedy, &exact), level_errors(&fair, &exact));
        let (gw, gb) = worst_best(&ge);
        let (fw, fb) = worst_best(&fe);
        println!(
            "t8p budget k={k}: greedy added {} rounds {} worst {gw:.3e} best {gb:.3e} \
             spread {:.1e}\n           errs {}",
            greedy.added,
            greedy.rounds,
            gw / gb,
            fmt(&ge)
        );
        println!(
            "t8p budget k={k}: fair   added {} rounds {} worst {fw:.3e} best {fb:.3e} \
             spread {:.1e}\n           errs {}",
            fair.added,
            fair.rounds,
            fw / fb,
            fmt(&fe)
        );

        assert!(
            ge.iter().all(|&e| e > -1e-12),
            "k={k}: the CI levels are variational upper bounds; a negative error means the \
             comparison lost its exact reference ({})",
            fmt(&ge)
        );
        assert!(
            gw > greedy_floor,
            "k={k}: the greedy arm's worst level error {gw:.3e} is already below the {greedy_floor:.0e} \
             starvation floor — this fixture no longer exhibits the mechanism, so the improvement \
             gate would be vacuous. Re-calibrate the fixture before touching the gate."
        );
        assert!(
            fw < gw / 4.0,
            "k={k}: per-target accounting did not recover the starved targets — worst level error \
             {fw:.3e} vs greedy {gw:.3e} (gate 4×, measured 13.4×/22.1×)"
        );
        assert!(
            fw / fb < 200.0,
            "k={k}: the fair arm's per-target spread is {:.1e} (worst {fw:.3e}, best {fb:.3e}) — \
             the budget is still concentrated on a few targets (gate 200, measured 28/12; greedy \
             {:.1e})",
            fw / fb,
            gw / gb
        );
        assert!(
            fair.added <= greedy.added,
            "k={k}: the fair arm spent MORE budget ({} vs {}) — converged targets are still \
             consuming slots",
            fair.added,
            greedy.added
        );
    }
}

#[test]
fn t8p_conv_tol_zero_is_the_greedy_baseline() {
    let space = QnSpace::new(5, 2, 2);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let exact = exact_config_spectrum(space, &ham);
    let res = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 6,
            max_added: 24,
            conv_tol: 0.0,
            ..Default::default()
        },
    );
    let errs = level_errors(&res, &exact);
    let (w, b) = worst_best(&errs);
    println!(
        "t8p sentinel: added {} rounds {} E0err {:.3e} worst {w:.3e} spread {:.1e}",
        res.added,
        res.rounds,
        res.e0 - exact[0],
        w / b
    );
    assert_eq!(
        (res.added, res.rounds),
        (24, 4),
        "conv_tol = 0.0 must reproduce the greedy acceptance schedule exactly (measured \
         added 24 in 4 rounds)"
    );
    assert!(
        (1e-4..5e-4).contains(&w),
        "conv_tol = 0.0 changed the greedy arm's worst level error: {w:.3e} outside the measured \
         band around 2.057e-4"
    );
    assert!(
        w / b > 500.0,
        "the greedy arm's per-target spread collapsed to {:.1e} (measured 2.2e3) without the \
         tracking being enabled",
        w / b
    );
}

#[test]
fn t8p_targeted_loop_is_deterministic() {
    let space = QnSpace::new(5, 2, 2);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let opts = ConeOpts {
        targets: 6,
        max_added: 24,
        conv_tol: 5e-3,
        ..Default::default()
    };
    let a = cone_response(&ham, &x, &opts);
    let b = cone_response(&ham, &x, &opts);
    assert_eq!((a.added, a.rounds, a.seeded), (b.added, b.rounds, b.seeded));
    assert_eq!(a.energies, b.energies, "energies are not bit-reproducible");
    assert_eq!(a.gaps, b.gaps, "gaps are not bit-reproducible");
    assert_eq!(
        a.character, b.character,
        "character fractions are not bit-reproducible"
    );
}

#[test]
fn t8p_character_is_a_partition_of_unity() {
    let space = QnSpace::new(5, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    for k in [0usize, 2, 12] {
        let res = cone_response(
            &ham,
            &x,
            &ConeOpts {
                targets: 4,
                max_added: k,
                conv_tol: 0.0,
                ..Default::default()
            },
        );
        println!(
            "t8p character k={k}: added {} {:?}",
            res.added, res.character
        );
        assert_eq!(
            res.character.len(),
            res.energies.len(),
            "k={k}: `character` must be aligned with `energies` (ground row included), got {} vs {}",
            res.character.len(),
            res.energies.len()
        );
        for (s, f) in res.character.iter().enumerate() {
            for (blk, &c) in f.iter().enumerate() {
                assert!(
                    (0.0..=1.0 + 1e-12).contains(&c) && c.is_finite(),
                    "k={k} state {s} block {blk}: fraction {c} is not a weight in [0,1]"
                );
            }
            let sum = f[0] + f[1] + f[2];
            assert!(
                (sum - 1.0).abs() <= 1e-12,
                "k={k} state {s}: fractions {f:?} sum to {sum} — the coefficients are a unit \
                 eigenvector in an AMBIENT-orthonormal basis, so this is a real measurement of \
                 that (measured deviation ≤ 6.0e-15) and not an artifact of any rescaling"
            );
        }
        if k == 0 {
            for (s, f) in res.character.iter().enumerate() {
                assert_eq!(
                    f[2], 0.0,
                    "k=0 state {s}: the tangent-only arm has no cone block at all, so its cone \
                     fraction must be exactly 0.0 (got {})",
                    f[2]
                );
            }
            assert!(
                (res.character[0][0] - 1.0).abs() <= 1e-12,
                "k=0: the tangent-only relaxed ground state is `x` itself, so its reference \
                 fraction must be 1 (got {})",
                res.character[0][0]
            );
        } else {
            assert!(
                res.character[0][2] > 0.0,
                "k={k}: the reference-channel seed N((H−λ)x) is in the space and carries the \
                 ground state's own error direction, so the ground row's cone fraction must be \
                 strictly positive (measured 3.3e-4 at k=2, 3.4e-4 at k=12)"
            );
        }
    }
}

#[test]
fn t8p_character_ground_row_carries_the_seed_repair() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, bl, br) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let exact = exact_config_spectrum(space, &ham);
    let n_t = tangent_basis(&bl, &br, true).len();
    let n_full = exact.len() - 1 - n_t;
    let res = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 4,
            max_added: n_full,
            topup_seed: Some(0x8E82),
            conv_tol: 0.0,
            ..Default::default()
        },
    );
    println!(
        "t8p character (full aug, n_t={n_t} n_full={n_full}): added {} {:?}",
        res.added, res.character
    );
    assert_eq!(res.added, n_full, "full augmentation not reached");
    assert_eq!(res.character.len(), res.energies.len());
    for (s, f) in res.character.iter().enumerate() {
        let sum = f[0] + f[1] + f[2];
        assert!(
            (sum - 1.0).abs() <= 1e-12,
            "state {s}: fractions {f:?} sum to {sum}"
        );
    }
    assert!(
        res.character[0][0] > 0.9,
        "at full augmentation the relaxed ground state is still dominated by the reference \
         (measured 0.9981), got {}",
        res.character[0][0]
    );
    assert!(
        res.character[0][2] > 0.0,
        "the ground row's cone fraction is the seed's repair weight and must be strictly positive \
         (measured 1.8e-3)"
    );
    for (j, f) in res.character.iter().enumerate().skip(1) {
        assert!(
            f[0] < 1e-2,
            "excited state {j} is orthogonal to the relaxed ground, which owns the x direction, \
             so its reference fraction must be ~0 (measured ≤ 1e-4), got {}",
            f[0]
        );
    }
}

#[test]
fn t8p_tracking_does_not_freeze_the_ground_channel() {
    let space = QnSpace::new(5, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let exact = exact_config_spectrum(space, &ham);
    let run = |conv_tol: f64| {
        cone_response(
            &ham,
            &x,
            &ConeOpts {
                targets: 2,
                max_added: 24,
                conv_tol,
                ..Default::default()
            },
        )
    };
    let greedy = run(0.0);
    let tracked = run(ConeOpts::default().conv_tol);
    let (eg, et) = (greedy.e0 - exact[0], tracked.e0 - exact[0]);
    println!(
        "t8p ground: greedy added {} rounds {} E0err {eg:.3e} gaps {:?}\n            tracked \
         added {} rounds {} E0err {et:.3e} gaps {:?}",
        greedy.added, greedy.rounds, greedy.gaps, tracked.added, tracked.rounds, tracked.gaps
    );
    assert!(
        et.abs() <= (10.0 * eg.abs()).max(1e-12),
        "the default tracking arm's ground error {et:.3e} is far above the untracked arm's \
         {eg:.3e} — the loop stopped while E_0^CI was still improving and the budget still had \
         {} of {} slots free",
        24 - tracked.added,
        24
    );
    for (j, (g, w)) in tracked.gaps.iter().zip(greedy.gaps.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-9 * w.abs().max(1.0),
            "gap {j}: tracked {g} vs untracked {w} — a frozen e0 shifts every gap by its own error"
        );
    }
}
