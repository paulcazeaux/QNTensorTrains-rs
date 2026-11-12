// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::core::factor::dense_self_adjoint_smallest;
use qntt::core::tangent::tangent_basis;
use qntt::round::RoundMode;
use qntt::solvers::hess::{hessian_grams, weingarten_operator};
use qntt::{QnSpace, TtVector, dot, rayleigh_quotient, xt_h_y};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t8n_cone_augmented_ci_reproduces_exact_at_full_augmentation() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, bl, br) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let lam = rayleigh_quotient(&ham, &x);
    let ctx = weingarten_operator(&ham, &x);

    let evals = exact_config_spectrum(space, &ham);
    let gaps: Vec<f64> = evals[1..].iter().map(|e| e - evals[0]).collect();

    let basis = tangent_basis(&bl, &br, true);
    let n_t = basis.len();
    let (_a_tt, b_tt) = hessian_grams(&ctx, &basis);
    let bmax = b_tt.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    assert!(
        bmax > 1e-8,
        "fixture not genuinely truncated (max|B| = {bmax:.3e})"
    );
    let iota: Vec<TtVector<f64>> = basis.iter().map(|u| u.to_vector()).collect();
    let mut iso = 0.0f64;
    for i in 0..n_t {
        for j in 0..n_t {
            iso = iso.max((basis[i].dot(&basis[j]) - dot(&iota[i], &iota[j])).abs());
        }
    }
    assert!(iso < 1e-10, "ι-isometry violated: {iso:.3e}");

    let mut refs: Vec<TtVector<f64>> = Vec::with_capacity(evals.len());
    refs.push(x.clone());
    refs.extend(iota.iter().cloned());
    let n_full = evals.len() - 1 - n_t;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E80);
    let mut normals: Vec<TtVector<f64>> = Vec::new();
    let mut attempts = 0;
    while normals.len() < n_full && attempts < 200 {
        attempts += 1;
        let v: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let cand = normal_project_ambient(&bl, &br, &v);
        if let Some(mut n) = mgs_accept_ambient(&refs, cand) {
            n = normal_project_ambient(&bl, &br, &n);
            let nn = n.norm();
            if nn < 1e-8 {
                continue;
            }
            n.scale_mut(1.0 / nn);
            refs.push(n.clone());
            normals.push(n);
        }
    }
    assert_eq!(normals.len(), n_full, "failed to complete the normal block");

    let mut worst = 0.0f64;
    for i in 0..refs.len() {
        for j in 0..refs.len() {
            let g = dot(&refs[i], &refs[j]);
            worst = worst.max((g - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    assert!(worst < 1e-8, "mixed-set orthonormality: {worst:.3e}");

    let amb = |vi: &TtVector<f64>, vj: &TtVector<f64>| -> f64 {
        xt_h_y(vi, &ham, vj) - lam * dot(vi, vj)
    };
    let mut asym = 0.0f64;
    for nm in normals.iter().take(3) {
        for ii in iota.iter().take(3) {
            asym = asym.max((amb(nm, ii) - amb(ii, nm)).abs());
        }
    }
    assert!(asym < 1e-9, "A_ext asymmetry {asym:.3e}");

    let dim = refs.len();
    let mut hci = vec![vec![0.0f64; dim]; dim];
    for i in 0..dim {
        for j in 0..=i {
            let v = xt_h_y(&refs[i], &ham, &refs[j]);
            hci[i][j] = v;
            hci[j][i] = v;
        }
    }
    let eci = sym_eigvals_jacobi(hci);
    for (k, (g, e)) in eci[1..]
        .iter()
        .map(|e| e - eci[0])
        .zip(gaps.iter())
        .enumerate()
        .take(gaps.len())
    {
        assert!(
            (g - e).abs() < 1e-8 * e.max(1.0),
            "gap {k}: CI {g} vs exact {e} — full augmentation must be exact (measured floor 1e-14)"
        );
    }
}

#[test]
fn t8n_targeted_selection_beats_random() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, bl, br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let evals = exact_config_spectrum(space, &ham);
    let gaps: Vec<f64> = evals[1..].iter().map(|e| e - evals[0]).collect();

    let basis = tangent_basis(&bl, &br, true);
    let n_t = basis.len();
    let (_a_tt, _b_tt) = hessian_grams(&ctx, &basis);
    let iota: Vec<TtVector<f64>> = basis.iter().map(|u| u.to_vector()).collect();

    let ci_gap_err = |normals: &[TtVector<f64>], jgap: usize| -> f64 {
        let mut all: Vec<&TtVector<f64>> = Vec::with_capacity(1 + n_t + normals.len());
        all.push(&x);
        all.extend(iota.iter());
        all.extend(normals.iter());
        let dim = all.len();
        let mut h = vec![vec![0.0f64; dim]; dim];
        for i in 0..dim {
            for j in 0..=i {
                let v = xt_h_y(all[i], &ham, all[j]);
                h[i][j] = v;
                h[j][i] = v;
            }
        }
        let e = sym_eigvals_jacobi(h);
        ((e[1 + jgap] - e[0]) - gaps[jgap]).abs()
    };
    let e0 = ci_gap_err(&[], 0);

    const K: usize = 2;
    let targeted = {
        let mut refs: Vec<TtVector<f64>> = Vec::with_capacity(1 + n_t + K);
        refs.push(x.clone());
        refs.extend(iota.iter().cloned());
        let mut normals: Vec<TtVector<f64>> = Vec::new();
        'outer: for _round in 0..2 * K {
            let dim = refs.len();
            let mut flat = vec![0.0f64; dim * dim];
            for i in 0..dim {
                for j in 0..=i {
                    let v = xt_h_y(&refs[i], &ham, &refs[j]);
                    flat[i * dim + j] = v;
                    flat[j * dim + i] = v;
                }
            }
            let want = 3.min(dim);
            let (ev, evec) = dense_self_adjoint_smallest::<f64>(&flat, dim, want);
            for j in 1..want {
                if normals.len() >= K {
                    break 'outer;
                }
                let mut v = &refs[0] * evec[j][0];
                for (c, r) in evec[j].iter().zip(refs.iter()).skip(1) {
                    v = &v + &(r * *c);
                    v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
                }
                let hv = ham.apply(&v);
                let mut w = &hv - &(&v * ev[j]);
                w.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
                let cand = normal_project_ambient(&bl, &br, &w);
                if let Some(mut n) = mgs_accept_ambient(&refs, cand) {
                    n = normal_project_ambient(&bl, &br, &n);
                    let nn = n.norm();
                    if nn < 1e-8 {
                        continue;
                    }
                    n.scale_mut(1.0 / nn);
                    refs.push(n.clone());
                    normals.push(n);
                }
            }
        }
        normals
    };
    let random = {
        let mut refs: Vec<TtVector<f64>> = Vec::with_capacity(1 + n_t + K);
        refs.push(x.clone());
        refs.extend(iota.iter().cloned());
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E81);
        let mut normals: Vec<TtVector<f64>> = Vec::new();
        let mut attempts = 0;
        while normals.len() < K && attempts < 100 {
            attempts += 1;
            let v: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
            let cand = normal_project_ambient(&bl, &br, &v);
            if let Some(mut n) = mgs_accept_ambient(&refs, cand) {
                n = normal_project_ambient(&bl, &br, &n);
                let nn = n.norm();
                if nn < 1e-8 {
                    continue;
                }
                n.scale_mut(1.0 / nn);
                refs.push(n.clone());
                normals.push(n);
            }
        }
        normals
    };
    assert!(
        targeted.len() >= K && random.len() >= K,
        "arm construction starved"
    );

    let et = ci_gap_err(&targeted[..K], 0);
    let er = ci_gap_err(&random[..K], 0);
    println!(
        "t8n selection: gap-1 CI error tangent-only {e0:.3e}, targeted(k={K}) {et:.3e}, random(k={K}) {er:.3e}"
    );
    assert!(
        et < e0 / 5.0,
        "targeted selection did not improve the targeted gap: {et:.3e} vs tangent-only {e0:.3e}"
    );
    assert!(
        er > e0 / 2.0,
        "random selection unexpectedly improved gap 1 ({er:.3e} vs {e0:.3e}) — the separation \
         nonvacuity is void; re-derive the gates"
    );
}

#[test]
fn t8n_library_cone_response_matches_exact_at_full_augmentation() {
    use qntt::solvers::hess::{ConeOpts, cone_response};
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, bl, br) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let evals = exact_config_spectrum(space, &ham);
    let gaps: Vec<f64> = evals[1..].iter().map(|e| e - evals[0]).collect();
    let n_t = tangent_basis(&bl, &br, true).len();
    let n_full = evals.len() - 1 - n_t;

    let opts = ConeOpts {
        targets: 4,
        max_added: n_full,
        topup_seed: Some(0x8E82),
        williamson: true,
        ..Default::default()
    };
    let res = cone_response(&ham, &x, &opts);
    assert_eq!(res.added, n_full, "full augmentation not reached");
    assert!(
        res.guards.iso < 1e-10 && res.guards.ortho < 1e-8 && res.guards.asym < 1e-9,
        "guards: {:?}",
        res.guards
    );
    for (k, (g, e)) in res.gaps.iter().zip(gaps.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-8 * e.max(1.0),
            "gap {k}: library {g} vs exact {e} (measured floor ~1e-14)"
        );
    }
    assert!(
        (res.e0 - evals[0]).abs() < 1e-8,
        "relaxed reference must reach E0 at full aug"
    );
    assert!(
        res.williamson.is_some(),
        "williamson study switch must return a spectrum"
    );
}

#[test]
fn t8n_library_targeted_improvement() {
    use qntt::solvers::hess::{ConeOpts, cone_response};
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let evals = exact_config_spectrum(space, &ham);
    let gap1 = evals[1] - evals[0];

    let base = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 0,
            ..Default::default()
        },
    );
    let e0 = (base.gaps[0] - gap1).abs();
    let aug = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 2,
            ..Default::default()
        },
    );
    let et = (aug.gaps[0] - gap1).abs();
    println!(
        "t8n library: gap-1 error tangent-only {e0:.3e} → k=2 (seed + targeted) {et:.3e} \
         (seeded {}, rounds {})",
        aug.seeded, aug.rounds
    );
    assert!(
        aug.added == 2 && aug.rounds >= 1,
        "budget not filled by seed + one targeted direction: added {} in {} rounds",
        aug.added,
        aug.rounds
    );
    assert!(
        et < e0 / 5.0,
        "targeted library solve did not improve the first gap: {et:.3e} vs {e0:.3e}"
    );
}

#[test]
fn t8n_gram_check_certificate() {
    use qntt::solvers::hess::{ConeOpts, cone_response};
    for (d, seed, kmax) in [(4usize, 0xB0B1u64, 4usize), (5, 0xB0B0, 2)] {
        let space = QnSpace::new(d, 2, 1);
        let (ham, x, _bl, _br) = weingarten_minimizer(space, seed, 1, 2);
        let run = |round_eps: f64| {
            cone_response(
                &ham,
                &x,
                &ConeOpts {
                    targets: 2,
                    max_added: kmax,
                    gram_check: true,
                    round_eps,
                    ..Default::default()
                },
            )
        };

        let res = run(1e-12);
        println!(
            "t8n gram_check d={d}: added {} rounds {} dev {:.3e}",
            res.added, res.rounds, res.guards.gram
        );
        assert!(
            res.guards.gram.is_finite() && res.guards.gram > 0.0,
            "d={d}: gram_check produced no certificate (guards.gram = {:.3e}) — the restructured \
             and pairwise Grams were not both assembled",
            res.guards.gram
        );
        assert!(
            res.guards.gram <= 1e-10,
            "d={d}: restructured vs pairwise CI Gram deviate by {:.3e} (> 1e-10) — the L1/L2 \
             identities do not hold as assembled",
            res.guards.gram
        );

        const LOOSE: f64 = 1e-2;
        let loose = run(LOOSE);
        println!(
            "t8n gram_check d={d} (round_eps {LOOSE:.0e}): added {} rounds {} dev {:.3e}",
            loose.added, loose.rounds, loose.guards.gram
        );
        assert!(
            loose.guards.gram.is_finite() && loose.guards.gram > 0.0,
            "d={d}: loosened-round certificate did not run ({:.3e})",
            loose.guards.gram
        );
        assert!(
            loose.guards.gram > 1e-6,
            "d={d}: loosening round_eps to {LOOSE:.0e} left the certificate at {:.3e} (default \
             {:.3e}) — the L2 w-rounding term is still not exercised, so nothing pins it",
            loose.guards.gram,
            res.guards.gram
        );
        assert!(
            loose.guards.gram < 1e-1,
            "d={d}: loosened-round deviation {:.3e} is far above round_eps·‖H·n‖ — not the \
             rank-hygiene round",
            loose.guards.gram
        );
    }
}

#[test]
fn t8n_reference_channel_repair() {
    use qntt::solvers::hess::{ConeOpts, cone_response};
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let evals = exact_config_spectrum(space, &ham);

    let base = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 0,
            ..Default::default()
        },
    );
    assert!(
        !base.seeded,
        "max_added = 0 must stay byte-identical to the tangent-only arm (no seeding, no H·x)"
    );
    let base_err = base.e0 - evals[0];

    let aug = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 2,
            ..Default::default()
        },
    );
    let aug_err = aug.e0 - evals[0];
    println!(
        "t8n repair: E0^CI−E0 tangent-only {base_err:.4e} → seeded k=2 {aug_err:.4e} \
         (added {}, rounds {}, seeded {})",
        aug.added, aug.rounds, aug.seeded
    );
    assert!(
        aug.seeded,
        "the standing reference-channel candidate N((H−λ)x) was not accepted on a genuinely \
         truncated fixture (‖r⊥‖ ≈ 8.2e-2 there — this is not the critical-reference no-op)"
    );
    assert_eq!(aug.added, 2, "budget not filled (seed + one targeted)");
    assert!(
        aug_err > 0.0 && aug.e0 <= base.e0 + 1e-12,
        "variational: enlarging the mixed space cannot raise E0^CI ({} vs {})",
        aug.e0,
        base.e0
    );
    assert!(
        aug_err < base_err / 50.0,
        "reference channel not repaired: E0^CI−E0 {aug_err:.3e} vs tangent-only {base_err:.3e} \
         (gate 50×, measured 198× with the seed / 14.6× without)"
    );

    for (tag, r) in [("base", &base), ("aug", &aug)] {
        assert_eq!(
            r.energies.len(),
            r.gaps.len() + 1,
            "{tag}: energies must be the absolute Ritz levels of the final solve"
        );
        assert!(
            (r.energies[0] - r.e0).abs() <= 1e-12,
            "{tag}: energies[0] {} != e0 {}",
            r.energies[0],
            r.e0
        );
        for (j, g) in r.gaps.iter().enumerate() {
            assert!(
                (g - (r.energies[j + 1] - r.energies[0])).abs() <= 1e-12,
                "{tag}: gaps[{j}] {g} != energies[{}] − energies[0]",
                j + 1
            );
        }
    }

    let closed = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 24,
            conv_tol: 0.0,
            ..Default::default()
        },
    );
    println!(
        "t8n repair (closure): added {} rounds {} E0^CI−E0 {:.3e}",
        closed.added,
        closed.rounds,
        closed.e0 - evals[0]
    );
    assert!(
        closed.added < 24,
        "the targeted loop did not close — this arm is meant to run to Krylov exhaustion"
    );
    assert!(
        (closed.e0 - evals[0]).abs() <= 1e-8,
        "at targeted-Krylov closure E0^CI must reach the exact E0: {} vs {} (measured 2e-14)",
        closed.e0,
        evals[0]
    );
}

#[test]
fn t8n_threaded_matches_serial() {
    use qntt::solvers::hess::{ConeOpts, cone_response};
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let opts = ConeOpts {
        targets: 3,
        max_added: 4,
        gram_check: true,
        ..Default::default()
    };

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build 1-thread rayon pool");
    let s = serial_pool.install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            1,
            "serial arm did not run in the 1-thread pool"
        );
        cone_response(&ham, &x, &opts)
    });
    let par_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("build 8-thread rayon pool");
    let p = par_pool.install(|| {
        assert!(
            rayon::current_num_threads() > 1,
            "parallel arm did not run multi-threaded ({} threads)",
            rayon::current_num_threads()
        );
        cone_response(&ham, &x, &opts)
    });

    assert_eq!(s.added, p.added, "threaded vs serial: added differs");
    assert_eq!(s.rounds, p.rounds, "threaded vs serial: rounds differ");
    assert_eq!(s.seeded, p.seeded, "threaded vs serial: seeded differs");
    assert!(
        (s.e0 - p.e0).abs() <= 1e-13 * s.e0.abs().max(1.0),
        "threaded vs serial e0: {} vs {}",
        s.e0,
        p.e0
    );
    assert_eq!(s.gaps.len(), p.gaps.len(), "threaded vs serial: gap count");
    for (j, (a, b)) in s.gaps.iter().zip(p.gaps.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-13 * a.abs().max(1.0),
            "threaded vs serial gap {j}: {a} vs {b}"
        );
    }
    assert!(
        !s.energies.is_empty() && s.energies.len() == p.energies.len(),
        "threaded vs serial: energies must be populated ({} vs {})",
        s.energies.len(),
        p.energies.len()
    );
    for (j, (a, b)) in s.energies.iter().zip(p.energies.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-13 * a.abs().max(1.0),
            "threaded vs serial energy {j}: {a} vs {b}"
        );
    }
    assert!(
        !s.character.is_empty() && s.character.len() == p.character.len(),
        "threaded vs serial: character must be populated ({} vs {})",
        s.character.len(),
        p.character.len()
    );
    for (j, (a, b)) in s.character.iter().zip(p.character.iter()).enumerate() {
        for blk in 0..3 {
            assert!(
                (a[blk] - b[blk]).abs() <= 1e-13,
                "threaded vs serial character[{j}][{blk}]: {} vs {}",
                a[blk],
                b[blk]
            );
        }
    }
    assert!(
        s.guards.gram.is_finite() && p.guards.gram.is_finite(),
        "threaded vs serial: the gram certificate must have run in both arms ({:.3e} / {:.3e})",
        s.guards.gram,
        p.guards.gram
    );
    assert!(
        (s.guards.gram - p.guards.gram).abs() <= 1e-13 * s.guards.gram.abs().max(1.0),
        "threaded vs serial gram deviation: {:.3e} vs {:.3e}",
        s.guards.gram,
        p.guards.gram
    );
}
