// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::hess::ConeOpts;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh};
use qntt::su2::hess::cone_response as su2_cone_response;
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::tangent_basis as su2_tangent_basis;
use qntt::{QnSpace, Scalar, SparseHamiltonian, dot, tt_state};
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

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
        if off < 1e-28 {
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
    let states: Vec<qntt::TtVector<f64>> = combinations(d, space.nup)
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

fn su2_grad_norm(ham: &Su2SparseHamiltonian<f64>, y: &Su2TtVector<f64>) -> f64 {
    use qntt::su2::riemannian::tangent_bases;
    use qntt::su2::tangent::Su2TtTangent;
    let lam = su2_rayleigh(ham, y).re_s();
    let hy = ham.apply(y);
    let res = &hy - &(y * lam);
    let (bl, br) = tangent_bases(y);
    Su2TtTangent::project(&bl, &br, &res).norm()
}

fn su2_minimizer(
    ham: &Su2SparseHamiltonian<f64>,
    space: Su2Space,
    seed: u64,
    lo: usize,
    hi: usize,
) -> Su2TtVector<f64> {
    use qntt::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd};
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x0 = randomized_su2_state(space, lo, hi, &mut rng);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let (_l, xg) = riemannian_gd(ham, &x0, &opts);
    let (_l, mut x) = riemannian_cg(ham, &xg, &opts);
    for _ in 0..8 {
        if su2_grad_norm(ham, &x) < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(ham, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    x.check();
    let gnorm = su2_grad_norm(ham, &x);
    assert!(
        gnorm < 1e-9,
        "SU(2) fixture not critical (‖grad‖={gnorm:.3e})"
    );
    x
}

#[test]
fn h8_su2_cone_matches_abelian_singlets() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);

    let sp22 = QnSpace::new(d, 2, 2);
    let h22 = SparseHamiltonian::new(sp22, &t, &v, None, f64::EPSILON);
    let ev22 = exact_config_spectrum(sp22, &h22);
    let sp31 = QnSpace::new(d, 3, 1);
    let h31 = SparseHamiltonian::new(sp31, &t, &v, None, f64::EPSILON);
    let ev31 = exact_config_spectrum(sp31, &h31);
    let singlets: Vec<f64> = ev22
        .iter()
        .copied()
        .filter(|e| ev31.iter().all(|p| (e - p).abs() > 1e-8 * e.abs().max(1.0)))
        .collect();
    assert_eq!(ev22.len(), 36);
    assert_eq!(ev31.len(), 16);
    assert_eq!(
        singlets.len(),
        20,
        "sector-degeneracy labeling failed (accidental degeneracy at this fixture?)"
    );

    let space = Su2Space::new(d, 4, 0);
    let hsu = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&hsu, space, 0xC0A1, 1, 2);
    let lam = su2_rayleigh(&hsu, &x);
    assert!(
        lam > singlets[0],
        "fixed-rank SU(2) energy must sit above the exact singlet ground state"
    );

    let (bl, br) = qntt::su2::riemannian::tangent_bases(&x);
    let n_t = su2_tangent_basis(&bl, &br, true).len();
    let n_full = singlets.len() - 1 - n_t;
    let targets = 4usize;
    let opts = ConeOpts {
        targets,
        max_added: n_full,
        topup_seed: Some(0xC0A2),
        ..Default::default()
    };
    let res = su2_cone_response(&hsu, &x, &opts);
    assert_eq!(
        res.added, n_full,
        "full augmentation not reached (n_t={n_t}, guards {:?})",
        res.guards
    );

    assert!(
        (res.e0 - singlets[0]).abs() < 1e-8,
        "E0: su2 {} vs abelian singlet {}",
        res.e0,
        singlets[0]
    );
    for k in 0..targets {
        let exact = singlets[k + 1] - singlets[0];
        assert!(
            (res.gaps[k] - exact).abs() < 1e-8 * exact.max(1.0),
            "gap {k}: su2 {} vs abelian singlet {}",
            res.gaps[k],
            exact
        );
    }
}

fn su2_cone_fixture() -> (Su2SparseHamiltonian<f64>, Su2TtVector<f64>, Vec<f64>) {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let sp22 = QnSpace::new(d, 2, 2);
    let h22 = SparseHamiltonian::new(sp22, &t, &v, None, f64::EPSILON);
    let ev22 = exact_config_spectrum(sp22, &h22);
    let sp31 = QnSpace::new(d, 3, 1);
    let h31 = SparseHamiltonian::new(sp31, &t, &v, None, f64::EPSILON);
    let ev31 = exact_config_spectrum(sp31, &h31);
    let singlets: Vec<f64> = ev22
        .iter()
        .copied()
        .filter(|e| ev31.iter().all(|p| (e - p).abs() > 1e-8 * e.abs().max(1.0)))
        .collect();
    let space = Su2Space::new(d, 4, 0);
    let hsu = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&hsu, space, 0xC0A1, 1, 2);
    (hsu, x, singlets)
}

#[test]
fn h8_su2_gram_check_certificate() {
    let (hsu, x, _singlets) = su2_cone_fixture();
    let run = |round_eps: f64| {
        su2_cone_response(
            &hsu,
            &x,
            &ConeOpts {
                targets: 2,
                max_added: 3,
                gram_check: true,
                round_eps,
                ..Default::default()
            },
        )
    };

    let res = run(1e-12);
    println!(
        "h8 su2 gram_check: added {} rounds {} dev {:.3e}",
        res.added, res.rounds, res.guards.gram
    );
    assert!(
        res.guards.gram.is_finite() && res.guards.gram > 0.0,
        "su2 gram_check produced no certificate (guards.gram = {:.3e})",
        res.guards.gram
    );
    assert!(
        res.guards.gram <= 1e-10,
        "su2 restructured vs pairwise CI Gram deviate by {:.3e} (> 1e-10) — the centerweighted \
         metric-adjointness chain does not hold as assembled",
        res.guards.gram
    );

    const LOOSE: f64 = 1e-2;
    let loose = run(LOOSE);
    println!(
        "h8 su2 gram_check (round_eps {LOOSE:.0e}): added {} rounds {} dev {:.3e}",
        loose.added, loose.rounds, loose.guards.gram
    );
    assert!(
        loose.guards.gram.is_finite() && loose.guards.gram > 0.0,
        "su2 loosened-round certificate did not run ({:.3e})",
        loose.guards.gram
    );
    assert!(
        loose.guards.gram > 1e-6,
        "su2: loosening round_eps to {LOOSE:.0e} left the certificate at {:.3e} (default {:.3e}) \
         — the L2 w-rounding term is still not exercised, so nothing pins it",
        loose.guards.gram,
        res.guards.gram
    );
    assert!(
        loose.guards.gram < 1e-1,
        "su2 loosened-round deviation {:.3e} is far above round_eps·‖H·n‖ — not the rank-hygiene \
         round",
        loose.guards.gram
    );
}

#[test]
fn h8_su2_reference_channel_repair() {
    let (hsu, x, singlets) = su2_cone_fixture();

    let base = su2_cone_response(
        &hsu,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 0,
            ..Default::default()
        },
    );
    assert!(!base.seeded, "max_added = 0 must not seed");
    let base_err = base.e0 - singlets[0];

    let aug = su2_cone_response(
        &hsu,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 1,
            ..Default::default()
        },
    );
    let aug_err = aug.e0 - singlets[0];
    println!(
        "h8 su2 repair: E0^CI−E0 tangent-only {base_err:.4e} → seeded k=1 {aug_err:.4e} \
         (added {}, seeded {})",
        aug.added, aug.seeded
    );
    assert!(
        aug.seeded,
        "the standing reference-channel candidate was not accepted (‖r⊥‖ ≈ 2.4e-1 on this \
         fixture — not the critical-reference no-op)"
    );
    assert_eq!(aug.added, 1, "the seed must occupy the single budget slot");
    assert!(
        aug_err > 0.0 && aug.e0 <= base.e0 + 1e-12,
        "variational: {} vs {}",
        aug.e0,
        base.e0
    );
    assert!(
        aug_err < base_err / 20.0,
        "su2 reference channel not repaired: {aug_err:.3e} vs tangent-only {base_err:.3e} \
         (gate 20×, measured 82× with the seed / 1.01× without)"
    );

    for (tag, r) in [("base", &base), ("aug", &aug)] {
        assert_eq!(r.energies.len(), r.gaps.len() + 1, "{tag}: energies length");
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
}

#[test]
fn h8_su2_threaded_matches_serial() {
    let (hsu, x, _singlets) = su2_cone_fixture();
    let opts = ConeOpts {
        targets: 2,
        max_added: 3,
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
        su2_cone_response(&hsu, &x, &opts)
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
        su2_cone_response(&hsu, &x, &opts)
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
    for (j, (a, b)) in s.gaps.iter().zip(p.gaps.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-13 * a.abs().max(1.0),
            "threaded vs serial gap {j}: {a} vs {b}"
        );
    }
    assert!(
        !s.energies.is_empty() && s.energies.len() == p.energies.len(),
        "threaded vs serial: energies must be populated"
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
        "threaded vs serial: the gram certificate must have run in both arms"
    );
    assert!(
        (s.guards.gram - p.guards.gram).abs() <= 1e-13 * s.guards.gram.abs().max(1.0),
        "threaded vs serial gram deviation: {:.3e} vs {:.3e}",
        s.guards.gram,
        p.guards.gram
    );
}

#[test]
fn h8_su2_character_is_a_partition_of_unity() {
    let (hsu, x, _singlets) = su2_cone_fixture();
    for k in [0usize, 3] {
        let res = su2_cone_response(
            &hsu,
            &x,
            &ConeOpts {
                targets: 2,
                max_added: k,
                conv_tol: 0.0,
                ..Default::default()
            },
        );
        println!(
            "h8 su2 character k={k}: added {} {:?}",
            res.added, res.character
        );
        assert_eq!(
            res.character.len(),
            res.energies.len(),
            "k={k}: `character` must be aligned with `energies` (ground row included)"
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
                "k={k} state {s}: fractions {f:?} sum to {sum} — the reduced-layer coefficients \
                 are a unit eigenvector in the ambient metric, so no centerweight enters here"
            );
        }
        if k == 0 {
            for (s, f) in res.character.iter().enumerate() {
                assert_eq!(
                    f[2], 0.0,
                    "k=0 state {s}: no cone block exists, so its fraction must be exactly 0.0"
                );
            }
            assert!(
                (res.character[0][0] - 1.0).abs() <= 1e-12,
                "k=0: the tangent-only relaxed ground state is `x` itself"
            );
        } else {
            assert!(
                res.character[0][2] > 0.0,
                "k={k}: the seed N((H−λ)x) carries the ground state's own error direction, so the \
                 ground row's cone fraction must be strictly positive — on this fixture the seed \
                 alone moves E0^CI−E0 by 82× (`h8_su2_reference_channel_repair`), which it could \
                 not do with zero weight"
            );
        }
    }
}

#[test]
fn h8_su2_channel_two_s_matches_abelian_triplets() {
    use qntt::solvers::hess::{ChannelOpts, ChannelSet, ChannelStatus, IrrepSel, SpinSel};
    use qntt::su2::hess::channel_response as su2_channel_response;

    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);

    let sp22 = QnSpace::new(d, 2, 2);
    let ev22 = exact_config_spectrum(
        sp22,
        &SparseHamiltonian::new(sp22, &t, &v, None, f64::EPSILON),
    );
    let sp31 = QnSpace::new(d, 3, 1);
    let ev31 = exact_config_spectrum(
        sp31,
        &SparseHamiltonian::new(sp31, &t, &v, None, f64::EPSILON),
    );
    let sp40 = QnSpace::new(d, 4, 0);
    let ev40 = exact_config_spectrum(
        sp40,
        &SparseHamiltonian::new(sp40, &t, &v, None, f64::EPSILON),
    );
    let singlets: Vec<f64> = ev22
        .iter()
        .copied()
        .filter(|e| ev31.iter().all(|p| (e - p).abs() > 1e-8 * e.abs().max(1.0)))
        .collect();
    let triplets: Vec<f64> = ev31
        .iter()
        .copied()
        .filter(|e| ev40.iter().all(|p| (e - p).abs() > 1e-8 * e.abs().max(1.0)))
        .collect();
    assert_eq!(
        (singlets.len(), triplets.len(), ev40.len()),
        (20, 15, 1),
        "sector-degeneracy labeling failed (accidental degeneracy at this fixture?)"
    );

    let base = Su2Space::new(d, 4, 0);
    let opts = ChannelOpts {
        channels: ChannelSet {
            irreps: IrrepSel::RefOnly,
            spin: SpinSel::Raised(1),
        },
        cone: ConeOpts {
            targets: 4,
            max_added: 40,
            topup_seed: Some(0xC4C2),
            conv_tol: 0.0,
            ..Default::default()
        },
        eps_h: f64::EPSILON,
        ..Default::default()
    };
    let spec = su2_channel_response(&t, &v, base, &opts);
    for (i, c) in spec.channels.iter().enumerate() {
        println!(
            "  su2 ch {i}: N {} 2S {} spin_delta {} status {:?} e_gs {:?} e0 {:?}",
            c.space.unwrap().n,
            c.space.unwrap().two_s,
            c.label.spin_delta,
            c.status,
            c.e_gs,
            c.cone.as_ref().map(|r| r.e0)
        );
    }

    assert_eq!(
        spec.channels.len(),
        2,
        "RefOnly × Raised(1) is the singlet reference plus the 2S = 2 channel"
    );
    assert_eq!(spec.reference, 0);
    let sector = |i: usize| {
        let sp = spec.channels[i]
            .space
            .expect("a feasible channel carries its typed space");
        (sp.n, sp.two_s)
    };
    assert_eq!(sector(0), (4, 0), "channel 0 is the base singlet space");
    assert_eq!(
        sector(1),
        (4, 2),
        "SU(2) Raised(1) is `two_s + 2·delta`, not `two_s + delta`"
    );
    assert!(
        spec.levels.iter().all(|l| !l.redundant),
        "SU(2) 2S channels hold disjoint multiplet content — nothing merged here is a copy"
    );

    for (i, exact) in [(0usize, &singlets), (1, &triplets)] {
        let c = &spec.channels[i];
        assert_eq!(c.status, ChannelStatus::Ok, "channel {i}: {:?}", c.status);
        let cone = c
            .cone
            .as_ref()
            .expect("an Ok channel carries a cone readout");
        assert!(
            (cone.e0 - exact[0]).abs() < 1e-8,
            "channel {i}: relaxed-reference CI ground {} vs the abelian-labeled exact {}",
            cone.e0,
            exact[0]
        );
        for (j, g) in cone.gaps.iter().enumerate() {
            let want = exact[j + 1] - exact[0];
            assert!(
                (g - want).abs() < 1e-8 * want.max(1.0),
                "channel {i} gap {j}: su2 {g} vs the abelian-labeled exact {want}"
            );
        }
    }

    let gap =
        spec.channels[1].cone.as_ref().unwrap().e0 - spec.channels[0].cone.as_ref().unwrap().e0;
    println!(
        "h8 su2 channel: singlet→triplet gap {gap:.9} (exact {:.9})",
        triplets[0] - singlets[0]
    );
    assert!(
        (gap - (triplets[0] - singlets[0])).abs() < 1e-8,
        "the 2S channel gap {gap} does not match the exact triplet-singlet difference {}",
        triplets[0] - singlets[0]
    );
}

#[test]
fn h8_su2_matfree_matches_dense() {
    use qntt::solvers::hess::MatfreeOpts;
    use qntt::su2::hess::cone_response_matfree as su2_cone_response_matfree;

    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 4, 0);
    let hsu = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&hsu, space, 0xC0A1, 1, 2);

    let (bl, br) = qntt::su2::riemannian::tangent_bases(&x);
    let n_t = su2_tangent_basis(&bl, &br, true).len();
    let n_full = 20 - 1 - n_t;

    let opts = ConeOpts {
        targets: 4,
        max_added: n_full,
        topup_seed: Some(0xC0A2),
        ..Default::default()
    };
    let dense = su2_cone_response(&hsu, &x, &opts);
    let mf = su2_cone_response_matfree(&hsu, &x, &opts, &MatfreeOpts::default());

    assert_eq!(dense.added, mf.added, "selection trajectories diverged");
    assert_eq!(dense.rounds, mf.rounds, "selection trajectories diverged");
    for (i, (a, b)) in dense.energies.iter().zip(mf.energies.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-8 * a.abs().max(1.0),
            "energy {i}: dense {a} vs matfree {b}"
        );
    }
    for (i, (ca, cb)) in dense.character.iter().zip(mf.character.iter()).enumerate() {
        for k in 0..3 {
            assert!(
                (ca[k] - cb[k]).abs() < 1e-6,
                "character[{i}][{k}]: dense {} vs matfree {}",
                ca[k],
                cb[k]
            );
        }
    }
    assert_eq!(
        mf.guards.gram, 0.0,
        "matfree must report the no-Gram sentinel"
    );
}

#[test]
fn h8_su2_stage_stats_and_rows() {
    use qntt::solvers::hess::MatfreeOpts;
    use qntt::su2::hess::{cone_response_matfree_with_rows, cone_response_with_stats};

    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 4, 0);
    let hsu = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&hsu, space, 0xC0A1, 1, 2);

    let opts = ConeOpts {
        targets: 4,
        max_added: 4,
        ..Default::default()
    };
    let (dres, dst) = cone_response_with_stats(&hsu, &x, &opts);
    let (mres, mfs, mixed) =
        cone_response_matfree_with_rows(&hsu, &x, &opts, &MatfreeOpts::default());
    let mst = &mfs.stages;

    for (tag, res, st) in [("dense", &dres, &dst), ("matfree", &mres, mst)] {
        assert!(st.n_tangent > 0, "{tag}: tangent block must be nonempty");
        assert_eq!(
            st.n_accept_ok, res.added,
            "{tag}: accepted normals must equal ConeResponse::added"
        );
        assert!(
            st.t_tangent_basis.as_nanos() > 0 && st.t_iota.as_nanos() > 0,
            "{tag}: basis and iota stages must be timed"
        );
        assert!(
            st.accounted() <= st.t_total,
            "{tag}: the stage timers must be disjoint"
        );
        assert!(
            st.max_rank_hv >= st.max_rank_cand && st.max_rank_cand > 0,
            "{tag}: unrounded H*v rank {} must dominate the rounded candidate rank {}",
            st.max_rank_hv,
            st.max_rank_cand
        );
    }
    assert_eq!(
        dst.n_tangent, mst.n_tangent,
        "same point, same tangent block"
    );
    assert!(
        dst.t_reference_block.as_nanos() > 0 && mst.t_xrow.as_nanos() > 0,
        "each mode must pay its own assembly stage"
    );

    assert_eq!(
        mixed.rows.len(),
        mres.added,
        "one cached row per accepted normal"
    );
    assert_eq!(
        mixed.normals.len(),
        mres.added,
        "one normal per accepted row — the caller needs them for the exact λ shift"
    );
    for (i, ni) in mixed.normals.iter().enumerate() {
        let nn = qntt::su2::ops::dot(ni, ni);
        assert!(
            (nn - 1.0).abs() < 1e-8,
            "normal {i} is not unit ({nn}) — the −λ·δ diagonal shift would be wrong"
        );
    }
    assert!(
        mixed.lam.is_finite() && mixed.lam < 0.0,
        "lam must be the electronic Rayleigh quotient, got {}",
        mixed.lam
    );
    for (m, row) in mixed.rows.iter().enumerate() {
        assert_eq!(
            row.len(),
            1 + mst.n_tangent + m + 1,
            "row {m} must run to its own diagonal entry"
        );
        assert!(
            row.iter().all(|e| e.is_finite()),
            "row {m} carries a non-finite entry"
        );
    }
}

#[test]
#[should_panic(expected = "gram_check requires the dense path")]
fn h8_su2_matfree_rejects_gram_check() {
    use qntt::solvers::hess::MatfreeOpts;

    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 4, 0);
    let hsu = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&hsu, space, 0xC0A1, 1, 2);
    let opts = ConeOpts {
        targets: 2,
        max_added: 2,
        gram_check: true,
        ..Default::default()
    };
    let _ = qntt::su2::hess::cone_response_matfree(&hsu, &x, &opts, &MatfreeOpts::default());
}
