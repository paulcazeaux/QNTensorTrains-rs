// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::solvers::mals::{MalsOptions, NoiseType};
use qntt::solvers::ssmals::{PenaltyRule, SsMode, StateSpecificOptions, penalty_weight};
use qntt::su2::expand::expand;
use qntt::su2::hamiltonian::rayleigh_quotient as su2_rayleigh_quotient;
use qntt::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use qntt::su2::solvers::excited::{ExcitedOpts, SsStart, excited_states};
use qntt::su2::solvers::mals::mals_with_history;
use qntt::su2::solvers::ssmals::state_specific_mals_with_history as su2_state_specific_mals_with_history;
use qntt::su2::tt_csf;
use qntt::su2::vector::Su2TtVector;
use qntt_testkit::randomized_su2_state;

fn base_opts(space: Su2Space, nroots: usize) -> ExcitedOpts {
    let mut cg = RiemannianOpts::new(&space);
    cg.eps = 1e-13;
    cg.max_iter = 600;
    cg.gradtol = 1e-12;
    ExcitedOpts {
        nroots,
        weights: None,
        rank: 0,
        bond_dims: Some(vec![0]),
        eps: 1e-10,
        max_iter: 30,
        noise: vec![0.0],
        ss_noise: vec![0.0],
        noise_type: NoiseType::Perturbative,
        seed: 1234,
        mode: SsMode::Penalty,
        penalty: PenaltyRule::Auto,
        ss_start: SsStart::Sa,
        polish: true,
        lower_grad_tol: 1e-6,
        trim_tol: 1e-12,
        trim_passes: 3,
        cg,
        cg_restarts: 10,
        polish_tol: 1e-10,
        grad_gate: 1e-9,
        overlap_tol: 1e-6,
        keep_sa: true,
        verbose: false,
    }
}

fn projection_su2(
    x: &Su2TtVector<f64>,
    two_sz: i64,
    dets: &[(Vec<bool>, Vec<bool>)],
    evec: &[f64],
) -> f64 {
    projection_on_dense(&expand(x, two_sz), dets, evec)
}

fn nearest_level(levels: &[f64], e: f64) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (j, &l) in levels.iter().enumerate() {
        let dv = (e - l).abs();
        if dv < best.1 {
            best = (j, dv);
        }
    }
    best
}

#[test]
fn tms_d1su2_full_rank_ladder_matches_dense_singlets() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xD1_1000, 3, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    assert!(
        levels.len() >= 4,
        "the fixture must resolve at least four singlet levels"
    );
    let mut rng = seeded(0xD1_1001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let opts = base_opts(space, 3);
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    assert_eq!(e.len(), 3, "three energies requested");
    assert_eq!(info.e, e, "`info.e` must repeat the returned energies");
    for i in 0..3 {
        println!(
            "tms_d1su2 root {i}: e {:.12e} vs singlet level {:.12e} (Δ {:.3e}), SA {:.12e}, \
             SS {:.12e}, w {:.3e}, gate {:.3e}",
            e[i],
            levels[i],
            (e[i] - levels[i]).abs(),
            info.sa_history.lambdas[i],
            info.ss_energies[i],
            info.penalties[i],
            info.gradnorms[i]
        );
        assert!(
            (e[i] - levels[i]).abs() <= 1e-8,
            "root {i}: e {:.12e} vs singlet level {:.12e}",
            e[i],
            levels[i]
        );
        assert!(
            (su2_rayleigh_quotient(&h, &roots[i]).re_s() - e[i]).abs() <= 1e-8,
            "root {i}: `e[i]` is not the Rayleigh quotient of the RETURNED root"
        );
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-12,
            "root {i} is not unit ({:.3e})",
            roots[i].norm()
        );
        assert!(
            projection_su2(&roots[i], 0, &fx.dets11, &fx.evec11[fx.singlets[i]]) >= 1.0 - 1e-8,
            "root {i} is not the dense singlet eigenvector"
        );
        assert!(
            global_eigen_residual_su2(&h, &roots[i], e[i]) <= 1e-6,
            "root {i} is not an eigenvector of H (residual {:.3e})",
            global_eigen_residual_su2(&h, &roots[i], e[i])
        );
        assert!(
            info.gradnorms[i] <= opts.grad_gate,
            "root {i} was returned above the exact gate ({:.3e})",
            info.gradnorms[i]
        );
        assert!(
            (info.ss_energies[i] - levels[i]).abs() <= 1e-8,
            "the SS stage's energy for root {i} misses the singlet level"
        );
        assert!(
            (info.sa_history.lambdas[i] - levels[i]).abs() <= 1e-8,
            "the SA stage's energy for root {i} misses the singlet level at full rank"
        );
    }

    assert!(
        gram_dev_su2(&roots) <= 1e-6,
        "the returned SU(2) ladder is not orthonormal ({:.3e})",
        gram_dev_su2(&roots)
    );
    assert_eq!(info.overlaps.len(), 3, "the Gram is (m+n)×(m+n)");
    for a in 0..3 {
        assert!(
            (info.overlaps[a][a] - 1.0).abs() <= 1e-10,
            "the recorded Gram diagonal is not 1 at {a}"
        );
    }
    assert!(
        info.overlap_dev <= opts.overlap_tol,
        "overlap_dev too large"
    );
    assert_eq!(info.converged, vec![true; 3], "every root must converge");
    assert_eq!(info.stalled, vec![false; 3], "no root may stall");
    assert_eq!(info.swaps, 0, "the ladder is energy-ordered");
    assert!(info.swapped.is_empty(), "no index may be flagged");
    assert!(info.licence, "the licence must hold on this fixture");
    assert!(
        info.lower_gradnorms.is_empty(),
        "there are no lower states to diagnose"
    );
    assert_eq!(
        info.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "the bottom rung has no lower states, so its polish weight is exactly 0"
    );
    for i in 1..3 {
        assert!(
            info.penalties[i] >= 1.0,
            "root {i}'s polish weight {:.3e} is below `penalty_weight`'s floor of 1",
            info.penalties[i]
        );
    }
}

#[test]
fn tms_d1su2_polarized_ladder_matches_dense() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xD1_1100, 3, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xD1_1101);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let opts = base_opts(space, 3);
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    for (i, r) in roots.iter().enumerate() {
        let (n_theta, n_cw) = count_theta_cw(r);
        println!(
            "tms_d1su2 polarized root {i}: e {:.12e} vs ev20 {:.12e}, n_theta {n_theta}, \
             n_cw {n_cw}",
            e[i], fx.ev20[i]
        );
        assert!(
            (e[i] - fx.ev20[i]).abs() <= 1e-8,
            "root {i}: {:.12e} vs dense (2,0) level {:.12e}",
            e[i],
            fx.ev20[i]
        );
        assert!(
            n_theta >= 2 && n_cw >= 2,
            "root {i} does not exercise the weighted paths (n_theta = {n_theta}, n_cw = {n_cw}); \
             the 2S = 2 arm would be vacuous"
        );
        assert!(
            projection_su2(r, 2, &fx.dets20, &fx.evec20[i]) >= 1.0 - 1e-8,
            "root {i} is not the dense (2,0) eigenvector"
        );
    }
    assert!(
        gram_dev_su2(&roots) <= 1e-6,
        "the polarized ladder is not orthonormal ({:.3e})",
        gram_dev_su2(&roots)
    );
    assert!(info.licence, "the polarized ladder must be certified");
}

#[test]
fn tms_d6su2_polish_lowers_j_until_a_trim_fires() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xD6_1000, 3, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xD6_1001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let opts_a = base_opts(space, 3);
    let (e_a, roots_a, info_a) = excited_states(&h, &x0, &[], &opts_a);
    println!(
        "tms_d6su2 run A: overlap_dev {:.3e}, licence {}",
        info_a.overlap_dev, info_a.licence
    );
    assert_eq!(
        info_a.passes,
        vec![0; 3],
        "at trim_tol = 1e-12 no trim pass may fire"
    );
    for i in 0..3 {
        println!(
            "tms_d6su2 run A root {i}: J_ss {:.16e}, J_pol {:.16e}, λ {:.16e}",
            info_a.j_ss[i], info_a.j_pol[i], e_a[i]
        );
        assert!(
            info_a.j_pol[i] <= info_a.j_ss[i] + 1e-10,
            "root {i}: the polish RAISED the penalized functional ({:.16e} → {:.16e})",
            info_a.j_ss[i],
            info_a.j_pol[i]
        );
        assert!(
            info_a.j_ss[i] >= e_a[i] - 1e-12,
            "root {i}: J ≥ λ must hold (the penalty is PSD)"
        );
    }

    let smins: Vec<f64> = roots_a
        .iter()
        .map(|r| {
            let (bl, br) = tangent_bases(r);
            qntt::su2::weingarten::min_schmidt(&bl, &br)
        })
        .collect();
    println!("tms_d6su2: centerweighted σ_min at the polished roots = {smins:?}");
    let trim_tol_b = 1.2e-3f64;
    let expect_trim: Vec<bool> = smins.iter().map(|&s| s < 0.3 * trim_tol_b).collect();
    assert!(
        expect_trim.iter().any(|&b| b),
        "no polished SU(2) root carries a centerweighted Schmidt value below 0.3·{trim_tol_b} \
         ({smins:?}), so run B would not exercise the trim loop — a FIXTURE finding, never a \
         reason to drop the arm"
    );

    let mut opts_b = base_opts(space, 3);
    opts_b.trim_tol = trim_tol_b;
    let (_e_b, _roots_b, info_b) = excited_states(&h, &x0, &[], &opts_b);
    println!(
        "tms_d6su2 run B: passes {:?}, gates {:?}, overlap_dev {:.3e}, licence {}",
        info_b.passes, info_b.gradnorms, info_b.overlap_dev, info_b.licence
    );
    for i in 0..3 {
        if expect_trim[i] {
            assert!(
                info_b.passes[i] >= 1,
                "root {i} carried σ_min = {:.3e} < 0.3·{trim_tol_b} yet no trim pass ran",
                smins[i]
            );
        }
        assert!(
            info_b.passes[i] <= opts_b.trim_passes,
            "root {i} ran {} trim passes, above the budget",
            info_b.passes[i]
        );
        assert!(
            info_b.gradnorms[i] <= opts_b.grad_gate,
            "root {i}: the re-polish after the trim did not restore the exact gate ({:.3e})",
            info_b.gradnorms[i]
        );
    }
    assert!(
        info_b.overlap_dev <= 1e-10,
        "the re-polish did not restore orthogonality after the trim ({:.3e}; run A polished to \
         {:.3e}, and the measured post-trim level on this fixture is 1.8e-14). A residual overlap \
         at the O(1/w) level means the re-polish is not reaching the penalized minimizer — do NOT \
         loosen this bound to admit it",
        info_b.overlap_dev,
        info_a.overlap_dev
    );
    assert!(
        info_b.licence,
        "the trimmed-and-re-polished SU(2) ladder must still be certified"
    );
    let want_licence = info_b.converged.iter().all(|&c| c)
        && info_b.overlap_dev <= opts_b.overlap_tol
        && info_b.swaps == 0;
    assert_eq!(
        info_b.licence, want_licence,
        "`licence` is not `all(converged) ∧ overlap_dev ≤ overlap_tol ∧ swaps == 0`"
    );
}

#[test]
fn tms_d9su2_single_root_reproduces_the_single_state_pipeline() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xD9_1000, 2, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let mut rng = seeded(0xD9_1001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let mut opts = base_opts(space, 1);
    opts.eps = 1e-8;
    opts.max_iter = 20;
    opts.noise = vec![1e-3, 0.0];
    opts.seed = 7;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    let sa_mals = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![1e-3, 0.0],
        noise_type: NoiseType::Perturbative,
        seed: 7,
        max_iter: 20,
        ..Default::default()
    };
    let (mh, xm) = mals_with_history(&h, &x0, &sa_mals);
    assert_eq!(
        info.sa_history.sweeps.len(),
        mh.sweeps.len(),
        "the SA stage ran a different number of sweeps than `mals` on the same schedule"
    );
    assert_f64_bits(
        info.sa_history.lambdas[0],
        mh.lambda,
        "sa_history.lambdas[0] vs mals' λ",
    );
    for (k, (sa, ms)) in info
        .sa_history
        .sweeps
        .iter()
        .zip(mh.sweeps.iter())
        .enumerate()
    {
        assert_f64_bits(
            sa.energies[0],
            ms.energy,
            &format!("SA sweep {k} energies[0]"),
        );
        assert_f64_bits(
            sa.discarded_weight,
            ms.discarded_weight,
            &format!("SA sweep {k} discarded_weight"),
        );
        assert_eq!(
            sa.max_bond_dim, ms.max_bond_dim,
            "SA sweep {k} max_bond_dim"
        );
    }
    let sa_states = info.sa_states.as_ref().expect("keep_sa = true");
    coeffs_bits_eq_su2(&sa_states[0], &xm, "the SA state vs `mals`'s output");

    let ss_mals = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        noise_type: NoiseType::Perturbative,
        seed: 7,
        max_iter: 20,
        ..Default::default()
    };
    let (sh, xs) = mals_with_history(&h, &xm, &ss_mals);
    assert_f64_bits(
        info.ss_energies[0],
        sh.lambda,
        "ss_energies[0] vs the delegate's λ",
    );
    for (k, (ss, ms)) in info.ss_history[0]
        .sweeps
        .iter()
        .zip(sh.sweeps.iter())
        .enumerate()
    {
        assert_f64_bits(ss.energy, ms.energy, &format!("SS sweep {k} energy"));
        assert!(
            ss.overlaps.is_empty(),
            "SS sweep {k}: the delegate path records no overlaps"
        );
        assert_f64_bits(ss.penalty, 0.0, &format!("SS sweep {k} penalty"));
        assert_eq!(
            ss.numops, 0,
            "SS sweep {k}: the delegate reports no applies"
        );
    }
    assert_eq!(
        info.penalties,
        vec![0.0],
        "with no lower states the polish weight is exactly 0"
    );

    let mut cg = RiemannianOpts::new(&space);
    cg.eps = 1e-13;
    cg.max_iter = 600;
    cg.gradtol = 1e-12;
    let gate = |y: &Su2TtVector<f64>| {
        qntt::su2::hess::weingarten_operator(&h, y)
            .gradient()
            .norm()
    };
    let (_j, mut y) = riemannian_cg(&h, &xs, &cg);
    let mut g = gate(&y);
    let mut used = 0usize;
    for _ in 0..opts.cg_restarts {
        if g <= opts.polish_tol {
            break;
        }
        let (_j2, y2) = riemannian_cg(&h, &y, &cg);
        let g2 = gate(&y2);
        if g2 >= g {
            y = y2;
            g = g2;
            used += 1;
            break;
        }
        y = y2;
        g = g2;
        used += 1;
    }
    println!("tms_d9su2: replica used {used} restarts, gate {g:.3e}");
    let (bl, br) = tangent_bases(&y);
    let smin = qntt::su2::weingarten::min_schmidt(&bl, &br);
    assert!(
        smin >= 0.3 * opts.trim_tol,
        "the replica's centerweighted σ_min = {smin:.3e} is under 0.3·trim_tol, so the driver's \
         trim loop WOULD fire and the bitwise comparison below is not the right claim here"
    );

    assert_f64_bits(
        e[0],
        su2_rayleigh_quotient(&h, &y).re_s(),
        "the returned energy vs the replicated polish's Rayleigh quotient",
    );
    coeffs_bits_eq_su2(&roots[0], &y, "the returned root vs the replicated polish");
    assert!(
        (e[0] - levels[0]).abs() <= 1e-8,
        "the bottom root {:.12e} is not the lowest singlet {:.12e}",
        e[0],
        levels[0]
    );
    assert!(
        info.licence,
        "the single-root SU(2) ladder must be certified"
    );
    assert!(
        !info.enriched,
        "at nroots = 1 the enrichment must never fire"
    );
}

fn lih_singlet_levels(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> (
    Vec<f64>,
    Vec<(Vec<bool>, Vec<bool>)>,
    Vec<f64>,
    Vec<Vec<f64>>,
) {
    let (dets22, ev22, evec22) = dense_sector(d, t, v, 2, 2);
    let (_dets31, ev31, _evec31) = dense_sector(d, t, v, 3, 1);
    let idx = singlet_indices(&ev22, &ev31);
    let levels: Vec<f64> = idx.iter().map(|&i| ev22[i]).collect();
    let _ = &evec22;
    (levels, dets22, ev22, evec22)
}

#[test]
fn tms_d10su2_lih_sto3g_ladder_lands_on_singlet_levels() {
    use qntt::integrals::{IntegralsOptions, load_fixture_dir};

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let (d, nelec) = (ints.norb, ints.nelec);
    let (t, v) = (ints.t, ints.v);
    let space = Su2Space::new(d, nelec, 0);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let (levels, dets22, _ev22, _evec22) = lih_singlet_levels(d, &t, &v);
    println!(
        "tms_d10su2: {} singlet levels, s0[0..6] = {:?}",
        levels.len(),
        &levels[..6.min(levels.len())]
    );
    assert!(
        levels.len() >= 6,
        "the S = 0 sub-spectrum must resolve at least six levels"
    );
    assert!(
        (levels[2] - levels[3]).abs() <= 1e-8,
        "the Π singlet pair is not degenerate on this fixture ({:.3e}); the reported negative \
         would not be the documented one",
        (levels[2] - levels[3]).abs()
    );
    let _ = &dets22;

    let codes: Vec<qntt::su2::core::Family> = (0..d)
        .map(|k| {
            if k < nelec / 2 {
                qntt::su2::core::Family::Occ
            } else {
                qntt::su2::core::Family::Empty
            }
        })
        .collect();
    let x0: Su2TtVector<f64> = tt_csf(&codes);

    let mut opts = base_opts(space, 3);
    opts.noise = vec![3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0];
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_d10su2: enriched {}, SA sweeps {} (budget {}), SS sweeps {:?}, gates {:?}, licence {}",
        info.enriched,
        info.sa_history.sweeps.len(),
        opts.max_iter,
        info.ss_history
            .iter()
            .map(|hh| hh.sweeps.len())
            .collect::<Vec<_>>(),
        info.gradnorms,
        info.licence
    );
    assert!(
        info.enriched,
        "an aufbau CSF start cannot host 3 states at any pair; the enrichment must have fired"
    );
    for i in 0..e.len() {
        let (j, dv) = nearest_level(&levels, e[i]);
        println!(
            "tms_d10su2 root {i}: e {:.12e} → singlet level {j} ({:.12e}), Δ {dv:.3e}",
            e[i], levels[j]
        );
        assert!(
            dv <= 1e-8,
            "root {i} = {:.12e} is not an S = 0 level of the dense spectrum (nearest is {j} at \
             {:.12e}, Δ {dv:.3e})",
            e[i],
            levels[j]
        );
    }
    let hit: Vec<usize> = e.iter().map(|&x| nearest_level(&levels, x).0).collect();
    println!(
        "tms_d10su2: the aufbau ladder reached singlet levels {hit:?} — the recorded negative is \
         that the degenerate Π pair (levels 2 and 3) is skipped; REPORTED, never asserted"
    );
    assert!(
        gram_dev_su2(&roots) <= 1e-8,
        "the LiH SU(2) ladder is not orthonormal ({:.3e})",
        gram_dev_su2(&roots)
    );
}

#[test]
fn tms_d10su2_lih_sto3g_random_start_lands_on_singlet_levels() {
    use qntt::integrals::{IntegralsOptions, load_fixture_dir};

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let (d, nelec) = (ints.norb, ints.nelec);
    let (t, v) = (ints.t, ints.v);
    let space = Su2Space::new(d, nelec, 0);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let (levels, _dets22, _ev22, _evec22) = lih_singlet_levels(d, &t, &v);

    let mut rng = seeded(0xDA_0011);
    let mut x0: Su2TtVector<f64> = randomized_su2_state(space, 6, 6, &mut rng);
    normalize_mut(&mut x0);

    let mut opts = base_opts(space, 3);
    opts.noise = vec![3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0];
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    for i in 0..e.len() {
        let (j, dv) = nearest_level(&levels, e[i]);
        println!(
            "tms_d10su2 random root {i}: e {:.12e} → singlet level {j} ({:.12e}), Δ {dv:.3e}",
            e[i], levels[j]
        );
        assert!(
            dv <= 1e-8,
            "root {i} = {:.12e} is not an S = 0 level (nearest {j}, Δ {dv:.3e})",
            e[i]
        );
    }
    let hit: Vec<usize> = e.iter().map(|&x| nearest_level(&levels, x).0).collect();
    println!(
        "tms_d10su2 random: reached singlet levels {hit:?}, licence {}, gates {:?} — which levels \
         a start reaches is a basin claim and is printed, never asserted",
        info.licence, info.gradnorms
    );
    assert!(
        gram_dev_su2(&roots) <= 1e-8,
        "the random-start SU(2) ladder is not orthonormal ({:.3e})",
        gram_dev_su2(&roots)
    );
}

#[test]
fn tms_d1su2_default_opts_are_the_pipeline_schedule() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let o = ExcitedOpts::new(&space);

    let bare = RiemannianOpts::new(&space);
    assert!(
        bare.eps > 1e-6,
        "the premise of the `cg.eps` override is that the bare SU(2) CG default ({:.1e}) is far \
         too loose for `polish_tol`",
        bare.eps
    );
    assert_eq!(o.cg.eps, 1e-13);
    assert_eq!(o.cg.gradtol, 1e-12);
    assert_eq!(o.cg.max_iter, 600);
    assert_eq!(o.nroots, 2);
    assert_eq!(o.rank, 0);
    assert!(o.bond_dims.is_none());
    assert_eq!(o.eps, 1e-10);
    assert_eq!(o.max_iter, 30);
    assert_eq!(
        o.noise,
        vec![3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0],
        "the response ground-state noise schedule; its last entry MUST be 0"
    );
    assert_eq!(o.ss_noise, vec![0.0]);
    assert_eq!(o.noise_type, NoiseType::Perturbative);
    assert_eq!(o.seed, 1234);
    assert_eq!(o.mode, SsMode::Penalty);
    assert_eq!(o.penalty, PenaltyRule::Auto);
    assert_eq!(o.ss_start, SsStart::Sa);
    assert!(o.polish);
    assert_eq!(o.lower_grad_tol, 1e-6);
    assert_eq!(o.trim_tol, 1e-6);
    assert_eq!(o.trim_passes, 3);
    assert_eq!(o.cg_restarts, 10);
    assert_eq!(o.polish_tol, 1e-10);
    assert_eq!(o.grad_gate, 1e-9);
    assert_eq!(o.overlap_tol, 1e-6);
    assert!(!o.keep_sa);
    assert!(!o.verbose);

    let fx = gapped_tv(0xD1_1300, 2, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let mut rng = seeded(0xD1_1301);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (e, _roots, info) = excited_states(&h, &x0, &[], &o);
    for i in 0..o.nroots {
        let (j, dv) = nearest_level(&levels, e[i]);
        println!(
            "tms_d1su2 default-opts root {i}: e {:.12e} → singlet level {j} ({:.12e}), Δ {dv:.3e}",
            e[i], levels[j]
        );
        assert!(
            dv <= 1e-8,
            "root {i} = {:.12e} is not an S = 0 level under the DEFAULT schedule",
            e[i]
        );
    }
    assert!(
        info.licence,
        "the default SU(2) schedule must certify its own ladder at full reduced rank"
    );
    assert!(info.sa_states.is_none(), "`keep_sa` is false by default");
}

#[test]
fn tms_d1su2_csf_start_is_enriched() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xD1_1100, 2, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let codes: Vec<qntt::su2::core::Family> = (0..d)
        .map(|k| {
            if k == 0 {
                qntt::su2::core::Family::Occ
            } else {
                qntt::su2::core::Family::Empty
            }
        })
        .collect();
    let x0: Su2TtVector<f64> = tt_csf(&codes);

    let mut opts = base_opts(space, 2);
    opts.noise = vec![1e-2, 1e-3, 0.0];
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    assert!(
        info.enriched,
        "a rank-1 CSF start must trigger the enrichment iteration (hazard 5)"
    );
    let mut reached: Vec<usize> = Vec::new();
    for i in 0..2 {
        let (j, dv) = nearest_level(&levels, e[i]);
        println!(
            "tms_d1su2 enriched root {i}: e {:.12e} → singlet level {j} ({:.12e}), Δ {dv:.3e}, \
             gate {:.3e}",
            e[i], levels[j], info.gradnorms[i]
        );
        assert!(
            dv <= 1e-8,
            "root {i} = {:.12e} is not an S = 0 level of the dense spectrum after enrichment",
            e[i]
        );
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-10,
            "root {i} is not unit ({:.3e})",
            roots[i].norm()
        );
        reached.push(j);
    }
    println!("tms_d1su2 enriched: singlet levels reached {reached:?} (reported, not asserted)");
    assert_ne!(
        reached[0], reached[1],
        "the two rungs collapsed onto the same singlet level"
    );
    assert!(
        info.licence,
        "the enriched SU(2) ladder must still be certified"
    );
}

#[allow(clippy::too_many_arguments)]
fn e1asu2_certified_ladder(
    h: &Su2SparseHamiltonian<f64>,
    fx: &GappedFixture,
    e: &[f64],
    roots: &[Su2TtVector<f64>],
    info: &qntt::solvers::excited::ExcitedInfo<Su2TtVector<f64>>,
    opts: &ExcitedOpts,
    offset: usize,
    tag: &str,
) {
    let levels = fx.singlet_levels();
    for i in 0..e.len() {
        let want = levels[offset + i];
        println!(
            "{tag} root {i}: e {:.12e} vs singlet {:.12e} (Δ {:.3e}), w {:.6e}, gate {:.3e}, \
             passes {}",
            e[i],
            want,
            (e[i] - want).abs(),
            info.penalties[i],
            info.gradnorms[i],
            info.passes[i]
        );
        assert!(
            (e[i] - want).abs() <= 1e-8,
            "{tag} root {i}: e {:.12e} vs singlet level {want:.12e}",
            e[i]
        );
        let proj = projection_su2(
            &roots[i],
            0,
            &fx.dets11,
            &fx.evec11[fx.singlets[offset + i]],
        );
        assert!(
            proj >= 1.0 - 1e-8,
            "{tag} root {i} is not the dense singlet eigenvector (projection {proj:.12})"
        );
        let res = global_eigen_residual_su2(h, &roots[i], e[i]);
        assert!(res <= 1e-6, "{tag} root {i}: eigen residual {res:.3e}");
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-12,
            "{tag} root {i} is not unit ({:.3e})",
            roots[i].norm()
        );
        assert!(
            info.gradnorms[i] <= opts.grad_gate,
            "{tag} root {i} was returned above the exact gate ({:.3e})",
            info.gradnorms[i]
        );
    }
    let gram = gram_dev_su2(roots);
    assert!(
        gram <= 1e-6,
        "{tag}: the returned SU(2) ladder is not orthonormal ({gram:.3e})"
    );
    assert!(
        info.overlap_dev <= opts.overlap_tol,
        "{tag}: overlap_dev {:.3e} above the gate",
        info.overlap_dev
    );
    assert_eq!(
        info.converged,
        vec![true; e.len()],
        "{tag}: every root must converge"
    );
    assert_eq!(
        info.stalled,
        vec![false; e.len()],
        "{tag}: no root may stall"
    );
    assert_eq!(info.swaps, 0, "{tag}: the ladder is energy-ordered");
    assert!(info.swapped.is_empty(), "{tag}: no index may be flagged");
    assert!(
        info.licence,
        "{tag}: the licence must hold at full reduced rank"
    );
}

#[test]
fn tms_e1asu2_fixed_and_project_drive_the_ladder() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xE1_A100, 3, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    assert!(
        levels.len() >= 4,
        "the fixture must resolve at least four singlet levels"
    );
    let mut rng = seeded(0xE1_A101);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let w_fix = 10.0 * (levels[3] - levels[0]);
    println!(
        "tms_e1asu2: w_fix = 10·(s₃ − s₀) = {w_fix:.6e}, singlet exclusion span {:.6e}",
        levels[3] - levels[0]
    );
    assert!(
        w_fix > 10.0 * (levels[2] - levels[0]),
        "the fixed weight must exceed every rung's own exclusion energy"
    );

    let mut opts1 = base_opts(space, 3);
    opts1.penalty = PenaltyRule::Fixed(w_fix);
    let (e1, roots1, info1) = excited_states(&h, &x0, &[], &opts1);
    e1asu2_certified_ladder(&h, &fx, &e1, &roots1, &info1, &opts1, 0, "tms_e1asu2 run 1");
    assert_eq!(
        info1.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "run 1: the bottom rung has no lower states"
    );
    for i in 1..3 {
        assert_eq!(
            info1.penalties[i].to_bits(),
            w_fix.to_bits(),
            "run 1 root {i}: the polish weight {:.17e} is not `Fixed`'s own {w_fix:.17e}",
            info1.penalties[i]
        );
        for (k, s) in info1.ss_history[i].sweeps.iter().enumerate() {
            assert_eq!(
                s.penalty.to_bits(),
                w_fix.to_bits(),
                "run 1 root {i} sweep {k}: the SU(2) sweep ran at {:.17e}",
                s.penalty
            );
        }
        let last = info1.ss_history[i].sweeps.last().expect("recorded");
        let sig: f64 = last.overlaps.iter().map(|o| o * o).sum();
        println!(
            "tms_e1asu2 run 1 root {i}: θ {:.16e}, λ {:.16e}, w·Σo² {:.3e}",
            last.penalized,
            last.energy,
            w_fix * sig
        );
        assert!(
            (last.penalized - last.energy - w_fix * sig).abs() <= 1e-8,
            "run 1 root {i}: θ − λ = {:.3e} but w·Σ|⟨p,u⟩|² = {:.3e}",
            last.penalized - last.energy,
            w_fix * sig
        );
        let pen: f64 = w_fix
            * (0..i)
                .map(|jx| {
                    let o = qntt::su2::ops::dot(&roots1[jx], &roots1[i]).abs_s();
                    o * o
                })
                .sum::<f64>();
        assert!(
            (info1.j_pol[i] - e1[i] - pen).abs() <= 1e-10,
            "run 1 root {i}: J_pol − λ = {:.3e} but w·Σ|o|² = {pen:.3e}",
            info1.j_pol[i] - e1[i]
        );
    }
    println!("tms_e1asu2 run 1: passes {:?}", info1.passes);

    let mut opts2 = base_opts(space, 3);
    opts2.mode = SsMode::Project;
    let (e2, roots2, info2) = excited_states(&h, &x0, &[], &opts2);
    e1asu2_certified_ladder(&h, &fx, &e2, &roots2, &info2, &opts2, 0, "tms_e1asu2 run 2");
    let mut skipped_total = 0usize;
    for i in 0..3 {
        for (k, s) in info2.ss_history[i].sweeps.iter().enumerate() {
            assert_eq!(
                s.penalty.to_bits(),
                0.0f64.to_bits(),
                "run 2 root {i} sweep {k}: `Project` records no weight"
            );
            assert_eq!(
                s.penalized.to_bits(),
                s.energy.to_bits(),
                "run 2 root {i} sweep {k}: θ must BE λ in `Project`"
            );
            skipped_total += s.skipped;
        }
    }
    println!(
        "tms_e1asu2 run 2: Σ skipped = {skipped_total} (printed, not gated), penalties {:?}",
        info2.penalties
    );
    assert_eq!(
        info2.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "run 2: the bottom rung has no lower states"
    );
    for i in 1..3 {
        assert!(
            info2.penalties[i] >= 1.0,
            "run 2 root {i}: the POLISH weight {:.3e} is below the rule's floor of 1",
            info2.penalties[i]
        );
        let last = info2.ss_history[i].sweeps.last().expect("recorded");
        let want = penalty_weight(info2.ss_energies[i], &last.overlaps, &e2[..i], 0.0);
        assert!(
            (info2.penalties[i] - want).abs() <= 1e-12 * want.abs().max(1.0),
            "run 2 root {i}: the polish weight {:.16e} is not the `Auto` rule replayed from the \
             sweep's final overlaps ({want:.16e})",
            info2.penalties[i]
        );
    }

    let mut opts3 = base_opts(space, 3);
    opts3.mode = SsMode::Project;
    opts3.penalty = PenaltyRule::Fixed(w_fix);
    let (e3, roots3, info3) = excited_states(&h, &x0, &[], &opts3);
    e1asu2_certified_ladder(&h, &fx, &e3, &roots3, &info3, &opts3, 0, "tms_e1asu2 run 3");
    for i in 0..3 {
        for (k, s) in info3.ss_history[i].sweeps.iter().enumerate() {
            assert_eq!(
                s.penalty.to_bits(),
                0.0f64.to_bits(),
                "run 3 root {i} sweep {k}: `Project` overrides the rule inside the sweep"
            );
        }
    }
    assert_eq!(
        info3.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "run 3: the bottom rung has no lower states"
    );
    for i in 1..3 {
        assert_eq!(
            info3.penalties[i].to_bits(),
            w_fix.to_bits(),
            "run 3 root {i}: the polish weight must be `Fixed`'s own, NOT the 0 the `Project` \
             sweep recorded ({:.17e} vs {w_fix:.17e})",
            info3.penalties[i]
        );
    }

    let opts_g = base_opts(space, 1);
    let (_eg, rg, _ig) = excited_states(&h, &x0, &[], &opts_g);
    let psi0 = rg[0].clone();
    let mut opts4 = base_opts(space, 2);
    opts4.penalty = PenaltyRule::Fixed(w_fix);
    let (e4, roots4, info4) = excited_states(&h, &x0, std::slice::from_ref(&psi0), &opts4);
    println!(
        "tms_e1asu2 control: e {e4:?}, penalties {:?}, lower_gradnorms {:?}",
        info4.penalties, info4.lower_gradnorms
    );
    e1asu2_certified_ladder(
        &h,
        &fx,
        &e4,
        &roots4,
        &info4,
        &opts4,
        1,
        "tms_e1asu2 control",
    );
    for i in 0..2 {
        assert_eq!(
            info4.penalties[i].to_bits(),
            w_fix.to_bits(),
            "control root {i}: with an external lower state EVERY rung polishes at {w_fix:.17e}"
        );
    }
    for b in 1..3 {
        assert!(
            info4.overlaps[0][b] <= 1e-6,
            "control: root {} is not orthogonal to the supplied lower state ({:.3e})",
            b - 1,
            info4.overlaps[0][b]
        );
    }
    assert!(
        info4.lower_gradnorms[0] <= 1e-6,
        "control: the supplied lower state is not converged (gate {:.3e})",
        info4.lower_gradnorms[0]
    );
}

#[test]
fn tms_e1bsu2_stall_branch_alone_refuses_the_licence() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xE1_B200, 2, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xE1_B201);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let full = qntt::su2::qn::max_ranks(&space);
    let mut widest = 0usize;
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            widest = widest.max(full[b][s]);
        }
    }
    println!("tms_e1bsu2: widest reduced-multiplet rank at the full grid = {widest}");
    assert!(
        widest > 2,
        "the rank-2 cap does not bind on this SU(2) space (widest {widest})"
    );

    let mut opts = base_opts(space, 1);
    opts.rank = 2;
    opts.bond_dims = Some(vec![2]);
    opts.eps = 1e-6;
    opts.cg.max_iter = 0;
    opts.cg_restarts = 0;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_e1bsu2: gate {:.3e} (grad_gate {:.1e}), stalled {:?}, converged {:?}, ss last \
         converged {}, overlap_dev {:.3e}, swaps {}, passes {:?}, licence {}, e {:.12e}",
        info.gradnorms[0],
        opts.grad_gate,
        info.stalled,
        info.converged,
        info.ss_history[0].sweeps.last().unwrap().converged,
        info.overlap_dev,
        info.swaps,
        info.passes,
        info.licence,
        e[0]
    );
    assert!(
        info.gradnorms[0].is_finite(),
        "the exact penalized gate must still be a number at a starved polish"
    );
    assert!(
        info.gradnorms[0] > 1e-6,
        "the starved polish left a gate of only {:.3e} at a BINDING reduced rank-2 cap; the arm \
         no longer isolates the stall with 1000× headroom over `grad_gate` — escalate it as a \
         fixture finding, do NOT lower the bound",
        info.gradnorms[0]
    );
    assert_eq!(info.stalled, vec![true], "the root must be flagged stalled");
    assert!(
        info.ss_history[0].sweeps.last().unwrap().converged,
        "the SU(2) sweep itself must have converged, or the arm does not isolate the stall"
    );
    assert_eq!(
        info.converged,
        vec![false],
        "converged = ss_conv ∧ !stalled"
    );
    assert_eq!(
        info.overlap_dev, 0.0,
        "a one-rung ladder has no off-diagonal Gram entry"
    );
    assert_eq!(info.swaps, 0, "a one-rung ladder cannot be out of order");
    assert!(
        !info.licence,
        "the stall conjunct alone refuses the licence"
    );
    assert_eq!(info.passes, vec![0], "no trim may fire at trim_tol = 1e-12");
    assert!(
        (roots[0].norm() - 1.0).abs() <= 1e-12,
        "the returned root is not unit"
    );

    let mut ctrl = opts.clone();
    ctrl.polish = false;
    let (_e_c, _r_c, info_c) = excited_states(&h, &x0, &[], &ctrl);
    println!(
        "tms_e1bsu2 control: gradnorms {:?}, stalled {:?}, converged {:?}, licence {}",
        info_c.gradnorms, info_c.stalled, info_c.converged, info_c.licence
    );
    assert!(info_c.gradnorms[0].is_nan(), "no polish, no exact gate");
    assert_eq!(info_c.stalled, vec![false], "no polish, no stall");
    assert_eq!(info_c.converged, vec![true], "the sweep converged");
    assert!(
        info_c.licence,
        "the flip of `polish` alone must flip the licence"
    );
}

#[test]
fn tms_e1fsu2_polish_false_bookkeeping_and_keep_sa() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xE1_F100, 3, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let mut rng = seeded(0xE1_F101);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let mut opts = base_opts(space, 3);
    opts.polish = false;
    opts.keep_sa = true;
    opts.max_iter = 30;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_e1fsu2: e {e:?}, ss_energies {:?}, passes {:?}, stalled {:?}, converged {:?}, \
         penalties {:?}, licence {}",
        info.ss_energies, info.passes, info.stalled, info.converged, info.penalties, info.licence
    );
    assert_eq!(
        info.passes,
        vec![0; 3],
        "no trim loop runs without a polish"
    );
    assert_eq!(info.stalled, vec![false; 3], "no polish, no stall");
    assert!(
        info.gradnorms.iter().all(|g| g.is_nan()),
        "`polish = false` records no exact gate (got {:?})",
        info.gradnorms
    );
    for i in 0..3 {
        assert_eq!(
            info.converged[i],
            info.ss_history[i].sweeps.last().unwrap().converged,
            "root {i}: without a polish `converged[i]` IS the sweep's own stop test"
        );
        println!(
            "tms_e1fsu2 root {i}: e {:.16e}, ss_energies {:.16e} (Δ {:.3e}), singlet {:.12e}",
            e[i],
            info.ss_energies[i],
            e[i] - info.ss_energies[i],
            levels[i]
        );
        assert!(
            (e[i] - info.ss_energies[i]).abs() <= 1e-10,
            "root {i}: `e[i]` and the sweep's own λ differ by {:.3e} at FULL reduced rank",
            e[i] - info.ss_energies[i]
        );
        assert!(
            (e[i] - levels[i]).abs() <= 1e-8,
            "root {i}: the unpolished full-rank SU(2) sweep must still reach the singlet level"
        );
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-12,
            "root {i} is not unit ({:.3e})",
            roots[i].norm()
        );
        let pen: f64 = info.penalties[i]
            * (0..i)
                .map(|jx| {
                    let o = qntt::su2::ops::dot(&roots[jx], &roots[i]).abs_s();
                    o * o
                })
                .sum::<f64>();
        assert!(
            (info.j_pol[i] - e[i] - pen).abs() <= 1e-10,
            "root {i}: J_pol − λ = {:.3e} but w·Σ|o|² = {pen:.3e}",
            info.j_pol[i] - e[i]
        );
        assert!(
            (info.j_ss[i] - info.j_pol[i]).abs() <= 1e-10,
            "root {i}: without a polish J_ss and J_pol are the same functional at the same state"
        );
    }
    assert_eq!(
        info.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "the bottom rung has no lower states"
    );
    for i in 1..3 {
        assert!(
            info.penalties[i] >= 1.0,
            "root {i}: the weight is recorded even when no polish consumes it"
        );
    }
    assert_eq!(
        info.licence,
        info.converged.iter().all(|&c| c)
            && info.overlap_dev <= opts.overlap_tol
            && info.swaps == 0,
        "`licence` is not the D9 conjunction"
    );

    let sa = info.sa_states.as_ref().expect("keep_sa = true");
    for i in 0..3 {
        let ss_opts = StateSpecificOptions {
            mals: MalsOptions {
                eps: opts.eps,
                bond_dims: vec![opts.rank],
                noise: opts.ss_noise.clone(),
                noise_type: opts.noise_type,
                seed: opts.seed,
                max_iter: opts.max_iter,
                verbose: opts.verbose,
                ..Default::default()
            },
            mode: opts.mode,
            penalty: opts.penalty,
            roots: 1,
            lower_energies: Some(e[..i].to_vec()),
            maxsubspace: None,
            restart_keep: None,
        };
        let (hist, mut y) = su2_state_specific_mals_with_history(&h, &sa[i], &roots[..i], &ss_opts);
        normalize_mut(&mut y);
        assert_f64_bits(
            info.ss_energies[i],
            hist.lambda,
            &format!("tms_e1fsu2 root {i}: ss_energies vs the replayed sweep's λ"),
        );
        coeffs_bits_eq_su2(
            &roots[i],
            &y,
            &format!("tms_e1fsu2 root {i}: the returned root vs the replayed sweep's output"),
        );
    }

    let mut opts_n = opts.clone();
    opts_n.keep_sa = false;
    let (e_n, roots_n, info_n) = excited_states(&h, &x0, &[], &opts_n);
    assert!(
        info_n.sa_states.is_none(),
        "`keep_sa = false` must not record the SA ladder"
    );
    for i in 0..3 {
        assert_f64_bits(e_n[i], e[i], &format!("tms_e1fsu2 keep_sa control: e[{i}]"));
        coeffs_bits_eq_su2(
            &roots_n[i],
            &roots[i],
            &format!("tms_e1fsu2 keep_sa control: root {i}"),
        );
    }
    assert_eq!(
        info_n.licence, info.licence,
        "`keep_sa` is a reporting flag; it must have no effect on the computation"
    );
}
