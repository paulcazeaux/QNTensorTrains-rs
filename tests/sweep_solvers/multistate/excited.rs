// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::C64;
use qntt::round::RoundMode;
use qntt::solvers::excited::{ExcitedOpts, SsStart, excited_states, excited_states_c64};
use qntt::solvers::mals::{MalsOptions, NoiseType, mals_with_history};
use qntt::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use qntt::solvers::ssmals::{
    PenaltyRule, SsMode, StateSpecificOptions, penalty_weight, state_specific_mals,
    state_specific_mals_with_history,
};
use qntt::{max_ranks, rayleigh_quotient, tt_randn, tt_state};
use qntt_testkit::randomized_state;

fn state_from_sites<T: Scalar>(d: usize, up_sites: &[usize], dn_sites: &[usize]) -> TtVector<T> {
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in up_sites {
        occ_up[s] = true;
    }
    for &s in dn_sites {
        occ_dn[s] = true;
    }
    tt_state(&occ_up, &occ_dn)
}

fn opts_exact() -> MalsOptions {
    MalsOptions {
        eps: 1e-10,
        conv_tol: Some(1e-12),
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 40,
        verbose: false,
        ..Default::default()
    }
}

fn perturb_center0<T: Scalar>(psi: &TtVector<T>, alpha: f64, seed: u64) -> TtVector<T> {
    let mut y = psi.clone();
    if y.ortho != Some(0) {
        y.right_orthogonalize(true);
    }
    let c0 = y.core(0).clone();
    let mut rng = seeded(seed);
    let mut r = c0.clone();
    r.for_each_coeff_mut(|z| *z = T::sample_standard(&mut rng));
    let rn = r.norm();
    assert!(rn > 0.0, "perturb_center0: zero random fill");
    let mut c = c0.clone();
    c.axpy(T::from_f64(alpha * c0.norm() / rn), &r);
    y.set_core(c);
    y
}

fn grad_exact_at<T: Scalar>(h: &SparseHamiltonian<T>, psi: &TtVector<T>) -> f64 {
    let mut y = psi.clone();
    normalize_mut(&mut y);
    let (bl, br) = tangent_bases(&y);
    let lam = rayleigh_quotient(h, &y).re_s();
    let resid = &(h * &y) - &(&y * T::from_f64(lam));
    qntt::core::tangent::TtTangent::project(&bl, &br, &resid).norm()
}

fn base_opts(space: QnSpace, nroots: usize) -> ExcitedOpts {
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

#[test]
fn tms_d1_full_rank_ladder_matches_dense() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD1_0000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD1_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let opts = base_opts(space, 3);
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    assert_eq!(e.len(), 3, "three energies requested");
    assert_eq!(roots.len(), 3, "three roots requested");
    assert_eq!(info.e, e, "`info.e` must repeat the returned energies");
    for i in 0..3 {
        println!(
            "tms_d1 root {i}: e {:.12e} vs dense {:.12e} (Δ {:.3e}), SA {:.12e}, SS {:.12e}, \
             w {:.3e}, gate {:.3e}, passes {}",
            e[i],
            fx.ev11[i],
            (e[i] - fx.ev11[i]).abs(),
            info.sa_history.lambdas[i],
            info.ss_energies[i],
            info.penalties[i],
            info.gradnorms[i],
            info.passes[i]
        );
        assert!(
            (e[i] - fx.ev11[i]).abs() <= 1e-8,
            "root {i}: e {:.12e} vs dense {:.12e}",
            e[i],
            fx.ev11[i]
        );
        assert!(
            (rayleigh_quotient(&h, &roots[i]).re_s() - e[i]).abs() <= 1e-8,
            "root {i}: `e[i]` is not the Rayleigh quotient of the RETURNED root"
        );
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-12,
            "root {i} is not unit ({:.3e})",
            roots[i].norm()
        );
        assert!(
            projection_on_dense(&roots[i], &fx.dets11, &fx.evec11[i]) >= 1.0 - 1e-8,
            "root {i} is not the dense eigenvector"
        );
        assert!(
            global_eigen_residual_abelian(&h, &roots[i], e[i]) <= 1e-6,
            "root {i} is not an eigenvector of H (residual {:.3e})",
            global_eigen_residual_abelian(&h, &roots[i], e[i])
        );
        assert!(
            (info.ss_energies[i] - fx.ev11[i]).abs() <= 1e-8,
            "the SS stage's own energy for root {i} misses the dense level"
        );
        assert!(
            (info.sa_history.lambdas[i] - fx.ev11[i]).abs() <= 1e-8,
            "the SA stage's energy for root {i} misses the dense level (full rank leaves the \
             state average exact)"
        );
        assert!(
            info.gradnorms[i] <= opts.grad_gate,
            "root {i} was returned above the exact gate ({:.3e})",
            info.gradnorms[i]
        );
    }

    assert!(
        gram_dev_tt(&roots) <= 1e-6,
        "the returned ladder is not orthonormal ({:.3e})",
        gram_dev_tt(&roots)
    );
    assert_eq!(info.overlaps.len(), 3, "the Gram is (m+n)×(m+n)");
    for a in 0..3 {
        assert_eq!(info.overlaps[a].len(), 3, "the Gram is square");
        assert!(
            (info.overlaps[a][a] - 1.0).abs() <= 1e-10,
            "the recorded Gram diagonal is not 1 at {a} ({:.3e})",
            info.overlaps[a][a]
        );
    }
    let off = (0..3)
        .flat_map(|a| (0..3).filter(move |&b| a != b).map(move |b| (a, b)))
        .map(|(a, b)| info.overlaps[a][b])
        .fold(0.0f64, f64::max);
    assert!(
        (info.overlap_dev - off).abs() <= 1e-15,
        "`overlap_dev` {:.3e} is not the max off-diagonal {off:.3e}",
        info.overlap_dev
    );
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
        !info.enriched,
        "a rank-5..10 random start already hosts 3 roots at every pair; the enrichment iteration \
         must not have fired"
    );
    assert!(
        info.lower_gradnorms.is_empty(),
        "there are no lower states, so there is nothing to diagnose"
    );

    assert_eq!(
        info.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "the bottom rung has no lower states, so its polish weight must be exactly 0"
    );
    for i in 1..3 {
        assert!(
            info.penalties[i] >= 1.0,
            "root {i}'s polish weight {:.3e} is below `penalty_weight`'s own floor of 1",
            info.penalties[i]
        );
        let last = info.ss_history[i]
            .sweeps
            .last()
            .expect("a swept root has at least one recorded iteration");
        let want = penalty_weight(info.ss_energies[i], &last.overlaps, &e[..i], 0.0);
        println!(
            "tms_d1 root {i}: penalties {:.6e} vs replay {want:.6e}",
            info.penalties[i]
        );
        assert!(
            (info.penalties[i] - want).abs() <= 1e-12 * want.abs().max(1.0),
            "root {i}'s recorded polish weight {:.16e} is not `penalty_weight` replayed from the \
             SS history's final overlaps and the ladder below it ({want:.16e})",
            info.penalties[i]
        );
    }

    let sa = info
        .sa_states
        .as_ref()
        .expect("`keep_sa = true` must record the SA ladder");
    assert_eq!(sa.len(), 3, "one SA state per root");
    for (i, s) in sa.iter().enumerate() {
        assert!(
            (rayleigh_quotient(&h, s).re_s() - fx.ev11[i]).abs() <= 1e-8,
            "SA state {i} misses the dense level"
        );
    }
}

#[test]
fn tms_d1_determinant_start_is_enriched() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD1_0100, 2, 0.05);
    let h = fx.abelian(space);
    let x0: TtVector<f64> = state_from_sites(d, &[0], &[0]);

    let mut opts = base_opts(space, 2);
    opts.noise = vec![1e-2, 1e-3, 0.0];
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    assert!(
        info.enriched,
        "a rank-1 determinant start must trigger the enrichment iteration (hazard 5)"
    );
    for i in 0..2 {
        println!(
            "tms_d1 enriched root {i}: e {:.12e} vs dense {:.12e}",
            e[i], fx.ev11[i]
        );
        assert!(
            (e[i] - fx.ev11[i]).abs() <= 1e-8,
            "root {i} misses the dense level after enrichment"
        );
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-12,
            "root {i} is not unit"
        );
    }
    assert!(info.licence, "the enriched ladder must still be certified");
}

#[test]
#[should_panic(expected = "excited_states")]
fn tms_d1_refuses_zero_roots() {
    let space = QnSpace::new(6, 1, 1);
    let fx = gapped_tv(0xD1_0200, 2, 0.05);
    let h = fx.abelian(space);
    let x0: TtVector<f64> = state_from_sites(6, &[0], &[0]);
    let opts = base_opts(space, 0);
    let _ = excited_states(&h, &x0, &[], &opts);
}

#[test]
#[should_panic(expected = "norm")]
fn tms_d1_refuses_a_zero_norm_lower_state() {
    let space = QnSpace::new(6, 1, 1);
    let fx = gapped_tv(0xD1_0201, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD1_0202);
    let x0: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    let mut zero = x0.clone();
    zero.scale_mut(0.0);
    let opts = base_opts(space, 1);
    let _ = excited_states(&h, &x0, std::slice::from_ref(&zero), &opts);
}

#[test]
#[should_panic]
fn tms_d1_refuses_a_lower_state_from_another_space() {
    let space = QnSpace::new(6, 1, 1);
    let other = QnSpace::new(6, 2, 0);
    let fx = gapped_tv(0xD1_0203, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD1_0204);
    let x0: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    let alien: TtVector<f64> = randomized_state(other, 2, 4, &mut rng);
    let opts = base_opts(space, 1);
    let _ = excited_states(&h, &x0, std::slice::from_ref(&alien), &opts);
}

#[test]
fn tms_d5_out_of_order_ladder_is_flagged_and_labels_kept() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD5_0000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD5_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let (l1, psi1) = state_specific_mals(
        &h,
        &x0,
        &[],
        &StateSpecificOptions {
            mals: opts_exact(),
            ..Default::default()
        },
    );
    let (l2, psi2) = state_specific_mals(
        &h,
        &x0,
        std::slice::from_ref(&psi1),
        &StateSpecificOptions {
            mals: opts_exact(),
            lower_energies: Some(vec![l1]),
            ..Default::default()
        },
    );
    assert!(
        (l1 - fx.ev11[0]).abs() <= 1e-8 && (l2 - fx.ev11[1]).abs() <= 1e-8,
        "the two reference rungs are not the two lowest dense levels ({l1:.12e}, {l2:.12e})"
    );
    let psi2_before = psi2.clone();

    let start = perturb_center0(&psi1, 1e-2, 0xD5_0002);
    let mut opts = base_opts(space, 1);
    opts.polish = false;
    let (e, _roots, info) = excited_states(&h, &start, std::slice::from_ref(&psi2), &opts);

    println!(
        "tms_d5: e[0] = {:.12e} (dense E₀ {:.12e}), lower E = {:.12e}, swaps {}, swapped {:?}",
        e[0], fx.ev11[0], l2, info.swaps, info.swapped
    );
    assert!(
        (e[0] - fx.ev11[0]).abs() <= 1e-8,
        "the root should sink to the global minimum (it is orthogonal to the lower state)"
    );
    assert_eq!(info.swaps, 1, "the out-of-order rung must be counted");
    assert_eq!(
        info.swapped,
        vec![1],
        "the flagged index is 0-based into Λ = [sort(E_lower); e]"
    );
    assert!(!info.licence, "an out-of-order ladder cannot be certified");
    assert_eq!(info.overlaps.len(), 2, "the Gram covers [lower; roots]");
    assert!(
        info.overlaps[0][1] <= 1e-6,
        "the root should be orthogonal to the lower state ({:.3e})",
        info.overlaps[0][1]
    );
    coeffs_bits_eq_tt(
        &psi2_before,
        &psi2,
        "the driver mutated the caller's lower state",
    );

    let opts2 = base_opts(space, 1);
    let (e2, _r2, info2) = excited_states(&h, &x0, &[psi2.clone(), psi1.clone()], &opts2);
    println!(
        "tms_d5 control: e[0] = {:.12e} vs dense E₂ {:.12e}, swaps {}",
        e2[0], fx.ev11[2], info2.swaps
    );
    assert!(
        (e2[0] - fx.ev11[2]).abs() <= 1e-8,
        "with both lower rungs excluded the root is the third dense level"
    );
    assert_eq!(
        info2.swaps, 0,
        "the swap scan must sort `lower` before comparing — an unsorted argument is not a swap"
    );
    assert!(info2.licence, "the control ladder must be certified");
}

#[test]
fn tms_d6_polish_lowers_j_until_a_trim_fires() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD6_0000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD6_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let mut opts_a = base_opts(space, 3);
    opts_a.trim_passes = 0;
    let (e_a, roots_a, info_a) = excited_states(&h, &x0, &[], &opts_a);
    println!(
        "tms_d6 run A: overlap_dev {:.3e}, gates {:?}, licence {}",
        info_a.overlap_dev, info_a.gradnorms, info_a.licence
    );
    assert_eq!(
        info_a.passes,
        vec![0; 3],
        "at trim_passes = 0 no trim pass may fire"
    );
    for i in 0..3 {
        println!(
            "tms_d6 run A root {i}: J_ss {:.16e}, J_pol {:.16e}, λ {:.16e}, w {:.3e}",
            info_a.j_ss[i], info_a.j_pol[i], e_a[i], info_a.penalties[i]
        );
        assert!(
            info_a.j_pol[i] <= info_a.j_ss[i] + 1e-10,
            "root {i}: the polish RAISED the penalized functional ({:.16e} → {:.16e}); the line \
             search only accepts decreases outside its documented roundoff floor",
            info_a.j_ss[i],
            info_a.j_pol[i]
        );
        assert!(
            info_a.j_ss[i] >= e_a[i] - 1e-12,
            "root {i}: J ≥ λ must hold (the penalty is PSD)"
        );
        let pen: f64 = info_a.penalties[i]
            * (0..i)
                .map(|jx| {
                    let o = qntt::dot(&roots_a[jx], &roots_a[i]).abs_s();
                    o * o
                })
                .sum::<f64>();
        assert!(
            (info_a.j_pol[i] - e_a[i] - pen).abs() <= 1e-10,
            "root {i}: J_pol − λ = {:.3e} but w·Σ|⟨rootⱼ,rootᵢ⟩|² = {pen:.3e}",
            info_a.j_pol[i] - e_a[i]
        );
    }

    let smins: Vec<f64> = roots_a
        .iter()
        .map(|r| {
            let (bl, br) = tangent_bases(r);
            qntt::core::tangent::min_schmidt(&bl, &br)
        })
        .collect();
    println!("tms_d6: σ_min at the polished roots = {smins:?}");
    let trim_tol_b = 1e-2f64;
    let expect_trim: Vec<bool> = smins.iter().map(|&s| s < 0.3 * trim_tol_b).collect();
    assert!(
        expect_trim.iter().any(|&b| b),
        "no polished root carries a Schmidt value below 0.3·{trim_tol_b} ({smins:?}), so run B \
         would not exercise the trim loop at all — this is a FIXTURE finding (pick another seed \
         or engineer a near-null sector), never a reason to drop the arm"
    );

    let mut opts_b = base_opts(space, 3);
    opts_b.trim_tol = trim_tol_b;
    opts_b.trim_passes = 3;
    let (e_b, roots_b, info_b) = excited_states(&h, &x0, &[], &opts_b);
    println!(
        "tms_d6 run B: passes {:?}, gates {:?}, overlap_dev {:.3e}, licence {}",
        info_b.passes, info_b.gradnorms, info_b.overlap_dev, info_b.licence
    );
    for i in 0..3 {
        println!(
            "tms_d6 run B root {i}: J_ss {:.16e}, J_pol {:.16e} (Δ {:.3e}, NOT asserted on a \
             trimmed root)",
            info_b.j_ss[i],
            info_b.j_pol[i],
            info_b.j_pol[i] - info_b.j_ss[i]
        );
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
        assert!(!info_b.stalled[i], "root {i} stalled after the trim");
        for (b, (ra, rb)) in roots_a[i]
            .ranks
            .iter()
            .zip(roots_b[i].ranks.iter())
            .enumerate()
        {
            for s in space.bond_sectors(b) {
                assert!(
                    rb[s] <= ra[s],
                    "root {i}: the trim GREW the rank at bond {b} sector {s:?} ({} → {})",
                    ra[s],
                    rb[s]
                );
            }
        }
    }
    assert!(
        info_b.overlap_dev <= 1e-5,
        "the re-polish did not restore orthogonality after the trim ({:.3e}; run A polished to \
         {:.3e}, and the measured post-trim level on this fixture is 7.6e-7). A residual overlap \
         at the O(1/w) level means the re-polish is not reaching the penalized minimizer — do \
         NOT loosen this bound to admit it",
        info_b.overlap_dev,
        info_a.overlap_dev
    );
    let want_licence = info_b.converged.iter().all(|&c| c)
        && info_b.overlap_dev <= opts_b.overlap_tol
        && info_b.swaps == 0;
    assert_eq!(
        info_b.licence, want_licence,
        "`licence` is not `all(converged) ∧ overlap_dev ≤ overlap_tol ∧ swaps == 0`"
    );
    for i in 0..3 {
        assert_eq!(
            info_b.converged[i],
            !info_b.stalled[i] && info_b.converged[i],
            "`converged[i]` must imply `!stalled[i]`"
        );
        println!(
            "tms_d6 run B root {i}: e {:.12e} vs dense {:.12e}",
            e_b[i], fx.ev11[i]
        );
    }
}

#[test]
fn tms_d7_unconverged_lower_state_is_diagnosed() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD7_0000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD7_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let (l1, psi1) = state_specific_mals(
        &h,
        &x0,
        &[],
        &StateSpecificOptions {
            mals: opts_exact(),
            ..Default::default()
        },
    );
    let (_l2, psi2) = state_specific_mals(
        &h,
        &x0,
        std::slice::from_ref(&psi1),
        &StateSpecificOptions {
            mals: opts_exact(),
            lower_energies: Some(vec![l1]),
            ..Default::default()
        },
    );
    let g1 = grad_exact_at(&h, &psi1);
    assert!(
        g1 <= 1e-8,
        "the reference rung ψ₁ is itself unconverged (gate {g1:.3e}); the contrast this arm \
         measures would be meaningless"
    );

    let mut bad = &psi1 + &(&psi2 * 0.1);
    normalize_mut(&mut bad);
    let oracle = grad_exact_at(&h, &bad);
    println!("tms_d7: ‖P((H−λ)ψ̂_bad)‖ = {oracle:.6e} (ψ₁'s own gate {g1:.3e})");
    assert!(
        oracle >= 0.1,
        "ψ_bad is not far enough from an eigenstate (gate {oracle:.3e}); the arm would not \
         exercise the diagnostic"
    );

    let opts = base_opts(space, 1);
    let (e, _roots, info) = excited_states(&h, &x0, std::slice::from_ref(&bad), &opts);
    assert_eq!(info.lower_gradnorms.len(), 1, "one lower state, one gate");
    println!(
        "tms_d7: lower_gradnorms[0] = {:.6e}, oracle {:.6e}; e[0] = {:.12e}, E₁ = {:.12e}, \
         E₂ = {:.12e}, licence {}",
        info.lower_gradnorms[0], oracle, e[0], fx.ev11[0], fx.ev11[1], info.licence
    );
    assert!(
        (info.lower_gradnorms[0] - oracle).abs() <= 1e-9 * oracle.max(1.0),
        "the recorded lower-state gate {:.12e} is not the exact projected residual {oracle:.12e}",
        info.lower_gradnorms[0]
    );
    assert!(
        info.lower_gradnorms[0] > opts.lower_grad_tol,
        "the diagnostic did not fire on a state {oracle:.3e} away from criticality"
    );

    assert!(
        e[0] <= fx.ev11[1] - 1e-4,
        "the root {:.12e} did not sink below E₂ = {:.12e}; the contaminated lower state's \
         orthogonal complement contains part of ψ₁",
        e[0],
        fx.ev11[1]
    );
    assert!(
        e[0] >= fx.ev11[0] - 1e-10,
        "no state can sit below the global minimum ({:.12e} vs {:.12e})",
        e[0],
        fx.ev11[0]
    );
    let target = (0.01 * fx.ev11[0] + fx.ev11[1]) / 1.01;
    println!(
        "tms_d7: complement-of-ψ_bad energy target {target:.12e}, got {:.12e} (dip {:.3e} bought \
         by the residual overlap {:.3e} at w = {:.3e})",
        e[0],
        target - e[0],
        info.overlaps[0][1],
        info.penalties[0]
    );
    assert!(
        e[0] <= target + 1e-8,
        "the root {:.12e} sits ABOVE the complement energy {target:.12e}, which is a feasible \
         zero-penalty point — the penalized minimization did not reach it",
        e[0]
    );
}

fn decoupled_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    assert_eq!(d, 6, "the decoupled fixture is built for d = 6");
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let diag_a = [-1.5f64, 0.6, 1.5];
    let diag_b = [-0.4f64, 2.1, 3.1];
    let off = [(0usize, 1usize, -0.3f64), (0, 2, -0.1), (1, 2, -0.3)];
    for (i, &val) in diag_a.iter().enumerate() {
        t.set(&[i, i], val);
    }
    for (i, &val) in diag_b.iter().enumerate() {
        t.set(&[3 + i, 3 + i], val);
    }
    for &(i, j, val) in &off {
        t.set(&[i, j], val);
        t.set(&[j, i], val);
        t.set(&[3 + i, 3 + j], val);
        t.set(&[3 + j, 3 + i], val);
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..3 {
        v.set(&[p, p, p, p], 1.0);
    }
    for &(p, q, dd, ex) in &[
        (0usize, 1usize, 0.4f64, 0.5f64),
        (0, 2, 0.2, 0.2),
        (1, 2, 0.3, 0.3),
    ] {
        v.set(&[p, p, q, q], dd);
        v.set(&[q, q, p, p], dd);
        v.set(&[p, q, q, p], ex);
        v.set(&[q, p, p, q], ex);
    }
    for p in 3..6 {
        v.set(&[p, p, p, p], 0.2);
    }
    for &(p, q, dd, ex) in &[
        (3usize, 4usize, 0.2f64, 0.1f64),
        (3, 5, 0.1, 0.05),
        (4, 5, 0.2, 0.15),
    ] {
        v.set(&[p, p, q, q], dd);
        v.set(&[q, q, p, p], dd);
        v.set(&[p, q, q, p], ex);
        v.set(&[q, p, p, q], ex);
    }
    (t, v)
}

fn bb_weight(x: &TtVector<f64>, dets: &[(Vec<bool>, Vec<bool>)]) -> f64 {
    let xd = x.to_dense();
    let total: f64 = xd.iter().map(|c| c * c).sum();
    let mut acc = 0.0f64;
    for (ou, od) in dets {
        let on_b = ou.iter().enumerate().all(|(k, &o)| !o || k >= 3)
            && od.iter().enumerate().all(|(k, &o)| !o || k >= 3);
        if on_b {
            let c = xd[basis_index(ou, od)];
            acc += c * c;
        }
    }
    acc / total.max(1e-300)
}

#[test]
fn tms_d8_decoupled_halves_completeness_negative() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = decoupled_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let (dets, ev, evec) = dense_sector(d, &t, &v, 1, 1);
    println!("tms_d8: dense levels {:?}", &ev[..8]);

    assert!(
        (ev[1] - ev[2]).abs() <= 1e-12,
        "the A↔B pair is not exactly degenerate ({:.3e}); the halves are not decoupled",
        (ev[1] - ev[2]).abs()
    );
    assert!(
        ev[4] - ev[3] >= 0.3 && ev[5] - ev[4] >= 0.3,
        "the hidden level is not isolated (gaps {:.3e}, {:.3e})",
        ev[4] - ev[3],
        ev[5] - ev[4]
    );
    let bb_of_evec = |c: usize| -> f64 {
        let mut acc = 0.0f64;
        let mut tot = 0.0f64;
        for (pos, (ou, od)) in dets.iter().enumerate() {
            let on_b = ou.iter().enumerate().all(|(k, &o)| !o || k >= 3)
                && od.iter().enumerate().all(|(k, &o)| !o || k >= 3);
            tot += evec[c][pos] * evec[c][pos];
            if on_b {
                acc += evec[c][pos] * evec[c][pos];
            }
        }
        acc / tot
    };
    assert!(
        bb_of_evec(4) >= 1.0 - 1e-12,
        "E₄ is not the pure BB state (BB weight {:.12})",
        bb_of_evec(4)
    );
    for c in [0usize, 1, 2, 3, 5] {
        assert!(
            bb_of_evec(c) <= 1e-12,
            "level {c} carries BB weight {:.3e}; the hidden sector is not isolated",
            bb_of_evec(c)
        );
    }

    let mut start: Option<(u64, TtVector<f64>)> = None;
    for attempt in 0..8u64 {
        let seed = 0xD8_0001 + attempt;
        let mut rng = seeded(seed);
        let mut cand: TtVector<f64> = tt_randn(space, &max_ranks(&space), &mut rng);
        cand.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
        normalize_mut(&mut cand);
        if bb_weight(&cand, &dets) >= 0.10 {
            start = Some((seed, cand));
            break;
        }
    }
    let (chosen, x0) = start.expect(
        "no full-rank start in 8 seeds carried ≥ 10 % BB weight (the uniform expectation is \
         9/36 ≈ 0.25) — a FIXTURE finding",
    );
    println!(
        "tms_d8: start seed {chosen:#x}, BB weight {:.4}",
        bb_weight(&x0, &dets)
    );

    let mut opts = base_opts(space, 5);
    opts.rank = 8;
    opts.bond_dims = Some(vec![8]);
    opts.keep_sa = true;
    let (e_sa, roots_sa, info_sa) = excited_states(&h, &x0, &[], &opts);
    let sa_states = info_sa.sa_states.as_ref().expect("keep_sa");
    for (i, s) in sa_states.iter().enumerate() {
        println!(
            "tms_d8 Sa: sa_states[{i}] BB weight {:.3e}",
            bb_weight(s, &dets)
        );
    }
    for (i, r) in roots_sa.iter().enumerate() {
        println!(
            "tms_d8 Sa: root {i} e {:.12e}, BB weight {:.3e}",
            e_sa[i],
            bb_weight(r, &dets)
        );
    }
    for i in 0..4 {
        assert!(
            (e_sa[i] - ev[i]).abs() <= 1e-8,
            "Sa root {i}: {:.12e} vs dense {:.12e}",
            e_sa[i],
            ev[i]
        );
    }
    let sa_bb = sa_states
        .iter()
        .map(|s| bb_weight(s, &dets))
        .fold(0.0f64, f64::max);
    assert!(
        sa_bb <= 1e-8,
        "the shared state-average basis KEPT weight in the BB sector ({sa_bb:.3e}); the recorded \
         completeness negative does not reproduce on this tree — escalate as a finding about the \
         SA truncation, do not retune this fixture"
    );
    assert!(
        (e_sa[4] - ev[5]).abs() <= 1e-8,
        "the top Sa root is {:.12e}, neither the hidden E₄ = {:.12e} nor the recorded E₅ = \
         {:.12e}",
        e_sa[4],
        ev[4],
        ev[5]
    );
    assert!(
        info_sa.licence,
        "the recorded negative is that the licence is TRUE while a level is missing — the \
         licence never certifies completeness (decision 17)"
    );
}

#[test]
fn tms_d9_single_root_reproduces_the_single_state_pipeline() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD9_0000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD9_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

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
        assert_f64_bits(sa.energy, ms.energy, &format!("SA sweep {k} energy"));
        assert_f64_bits(
            sa.discarded_weight,
            ms.discarded_weight,
            &format!("SA sweep {k} discarded_weight"),
        );
        assert_eq!(
            sa.max_bond_dim, ms.max_bond_dim,
            "SA sweep {k} max_bond_dim"
        );
        assert_eq!(sa.bond_dim, ms.bond_dim, "SA sweep {k} bond_dim");
        assert_f64_bits(sa.noise, ms.noise, &format!("SA sweep {k} noise"));
    }
    let sa_states = info.sa_states.as_ref().expect("keep_sa = true");
    coeffs_bits_eq_tt(&sa_states[0], &xm, "the SA state vs `mals`'s output");

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
    assert_eq!(
        info.ss_history[0].sweeps.len(),
        sh.sweeps.len(),
        "the SS stage ran a different number of sweeps than its delegate"
    );
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
        assert_f64_bits(ss.penalized, ms.energy, &format!("SS sweep {k} penalized"));
        assert!(
            ss.overlaps.is_empty(),
            "SS sweep {k}: the delegate path records no overlaps"
        );
        assert_f64_bits(ss.penalty, 0.0, &format!("SS sweep {k} penalty"));
        assert_eq!(
            ss.numops, 0,
            "SS sweep {k}: the delegate reports no applies"
        );
        assert_eq!(ss.skipped, 0, "SS sweep {k}: nothing can be skipped");
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
    let gate = |y: &TtVector<f64>| {
        qntt::solvers::hess::weingarten_operator(&h, y)
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
    println!("tms_d9: replica used {used} restarts, gate {g:.3e}");
    let (bl, br) = tangent_bases(&y);
    let smin = qntt::core::tangent::min_schmidt(&bl, &br);
    assert!(
        smin >= 0.3 * opts.trim_tol,
        "the replica's σ_min = {smin:.3e} is under 0.3·trim_tol, so the driver's trim loop WOULD \
         fire and the bitwise comparison below is not the right claim for this fixture"
    );

    assert_f64_bits(
        e[0],
        rayleigh_quotient(&h, &y).re_s(),
        "the returned energy vs the replicated polish's Rayleigh quotient",
    );
    coeffs_bits_eq_tt(&roots[0], &y, "the returned root vs the replicated polish");
    assert!(
        info.gradnorms[0] <= opts.grad_gate,
        "the bottom root was returned above the exact gate ({:.3e})",
        info.gradnorms[0]
    );
    assert!(
        (e[0] - fx.ev11[0]).abs() <= 1e-8,
        "the bottom root {:.12e} is not the dense ground state {:.12e}",
        e[0],
        fx.ev11[0]
    );
    assert_eq!(info.overlaps.len(), 1, "a one-state Gram");
    assert!(
        (info.overlaps[0][0] - 1.0).abs() <= 1e-12,
        "the one-state Gram is not [[1.0]]"
    );
    assert_eq!(info.swaps, 0, "a one-rung ladder cannot be out of order");
    assert!(info.swapped.is_empty(), "nothing to flag");
    assert!(info.licence, "the single-root ladder must be certified");
    assert!(
        !info.enriched,
        "at nroots = 1 the two-site local space always hosts one state, so the enrichment \
         iteration must never fire"
    );
}

#[test]
fn tms_d10_lih_sto3g_ladder_matches_dense() {
    use qntt::hamiltonian::assembly::Construction;
    use qntt::integrals::{IntegralsOptions, load_fixture_dir};

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let (d, nelec) = (ints.norb, ints.nelec);
    let (nup, ndn) = (nelec / 2, nelec / 2);
    let space = QnSpace::new(d, nup, ndn);
    let (t, v) = (ints.t, ints.v);
    let h = SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Median);

    let (dets, ev, _evec) = dense_sector(d, &t, &v, nup, ndn);
    println!(
        "tms_d10: {} determinants, E[0..4] = {:?}",
        dets.len(),
        &ev[..4]
    );

    let occ: Vec<bool> = (0..d).map(|i| i < nup).collect();
    let x0: TtVector<f64> = tt_state(&occ, &occ);

    let mut opts = base_opts(space, 3);
    opts.noise = vec![3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0];
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_d10: SA sweeps {} (budget {}), SS sweeps {:?}, passes {:?}, gates {:?}",
        info.sa_history.sweeps.len(),
        opts.max_iter,
        info.ss_history
            .iter()
            .map(|hh| hh.sweeps.len())
            .collect::<Vec<_>>(),
        info.passes,
        info.gradnorms
    );
    assert!(
        info.enriched,
        "an HF determinant start cannot host 3 states at any pair; the enrichment iteration must \
         have fired"
    );
    for i in 0..3 {
        println!(
            "tms_d10 root {i}: e {:.12e} vs dense {:.12e} (Δ {:.3e})",
            e[i],
            ev[i],
            (e[i] - ev[i]).abs()
        );
        assert!(
            (e[i] - ev[i]).abs() <= 1e-8,
            "root {i}: {:.12e} vs dense {:.12e}",
            e[i],
            ev[i]
        );
        assert!(
            info.gradnorms[i] <= 1e-9,
            "root {i} was returned above the exact gate ({:.3e})",
            info.gradnorms[i]
        );
        assert!(
            (info.sa_history.lambdas[i] - ev[i]).abs() <= 1e-8,
            "the SA stage's energy for root {i} misses the dense level at full rank"
        );
        assert!(
            (info.ss_energies[i] - ev[i]).abs() <= 1e-8,
            "the SS stage's energy for root {i} misses the dense level at full rank"
        );
    }
    println!(
        "tms_d10: gaps e[1]−e[0] = {:.8e}, e[2]−e[0] = {:.8e} (dense {:.8e}, {:.8e})",
        e[1] - e[0],
        e[2] - e[0],
        ev[1] - ev[0],
        ev[2] - ev[0]
    );
    assert!(
        gram_dev_tt(&roots) <= 1e-8,
        "the LiH ladder is not orthonormal ({:.3e})",
        gram_dev_tt(&roots)
    );
    assert!(info.licence, "the LiH ladder must be certified");
}

#[test]
fn tms_d1_default_opts_are_the_pipeline_schedule() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let o = ExcitedOpts::new(&space);

    let bare = RiemannianOpts::new(&space);
    assert!(
        bare.eps > 1e-6,
        "the premise of the `cg.eps` override is that the bare CG default ({:.1e}) is far too \
         loose for `polish_tol`",
        bare.eps
    );
    assert_eq!(o.cg.eps, 1e-13, "the polish CG truncation tolerance");
    assert_eq!(o.cg.gradtol, 1e-12, "the polish CG gradient stop");
    assert_eq!(o.cg.max_iter, 600, "the polish CG budget");
    assert_eq!(o.nroots, 2);
    assert_eq!(o.rank, 0, "full rank by default");
    assert!(
        o.bond_dims.is_none(),
        "the default ramp is derived from `rank`"
    );
    assert_eq!(o.eps, 1e-10);
    assert_eq!(o.max_iter, 30);
    assert_eq!(
        o.noise,
        vec![3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0],
        "the response ground-state noise schedule (its last entry MUST be 0: a noisy iteration \
         can never satisfy a sweep's stop rule)"
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
    assert!(!o.keep_sa, "the SA ladder is dropped unless asked for");
    assert!(!o.verbose);

    let fx = gapped_tv(0xD1_0300, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD1_0301);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (e, _roots, info) = excited_states(&h, &x0, &[], &o);
    for i in 0..o.nroots {
        println!(
            "tms_d1 default-opts root {i}: e {:.12e} vs dense {:.12e} (gate {:.3e}, passes {})",
            e[i], fx.ev11[i], info.gradnorms[i], info.passes[i]
        );
        assert!(
            (e[i] - fx.ev11[i]).abs() <= 1e-8,
            "root {i}: the DEFAULT schedule misses the dense level ({:.12e} vs {:.12e})",
            e[i],
            fx.ev11[i]
        );
    }
    assert!(
        info.licence,
        "the default schedule must certify its own ladder at full rank"
    );
    assert!(info.sa_states.is_none(), "`keep_sa` is false by default");
}

#[test]
#[should_panic(expected = "after the enrichment iteration")]
fn tms_d1_refuses_a_start_that_is_short_after_enrichment() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD1_0400, 2, 0.05);
    let h = fx.abelian(space);
    let x0: TtVector<f64> = state_from_sites(d, &[0], &[0]);
    let mut opts = base_opts(space, 1000);
    opts.noise = vec![1e-2, 0.0];
    let _ = excited_states(&h, &x0, &[], &opts);
}

#[test]
fn tms_d7_a_zero_iteration_sweep_still_returns_a_unit_root() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD7_0300, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD7_0301);
    let mut x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    normalize_mut(&mut x0);
    x0.scale_mut(3.0);

    let mut opts = base_opts(space, 1);
    opts.max_iter = 0;
    opts.polish = false;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    let mut unit = roots[0].clone();
    normalize_mut(&mut unit);
    let rq = rayleigh_quotient(&h, &unit).re_s();
    println!(
        "tms_d7 zero-iteration: ‖root‖ {:.16e}, e {:.16e}, RQ of the normalized root {rq:.16e}, \
         j_ss {:.16e}",
        roots[0].norm(),
        e[0],
        info.j_ss[0]
    );
    assert!(
        (roots[0].norm() - 1.0).abs() <= 1e-12,
        "the returned root has norm {:.16e}",
        roots[0].norm()
    );
    assert!(
        (e[0] - rq).abs() <= 1e-12 * rq.abs().max(1.0),
        "e[0] = {:.16e} is not the Rayleigh quotient of the returned root ({rq:.16e})",
        e[0]
    );
    assert!(
        (info.j_ss[0] - rq).abs() <= 1e-10 * rq.abs().max(1.0),
        "with no lower states J = λ, but j_ss = {:.16e} against λ = {rq:.16e}",
        info.j_ss[0]
    );
    assert!(
        info.gradnorms[0].is_nan(),
        "`polish = false` records no exact gate"
    );
}

#[test]
fn tms_d7_a_zero_iteration_ladder_penalizes_with_unit_overlaps() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xD7_0300, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xD7_0301);
    let mut x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    normalize_mut(&mut x0);
    x0.scale_mut(3.0);

    let mut opts = base_opts(space, 2);
    opts.max_iter = 0;
    opts.polish = false;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    for i in 0..2 {
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-12,
            "root {i} has norm {:.16e}",
            roots[i].norm()
        );
        let rq = rayleigh_quotient(&h, &roots[i]).re_s();
        assert!(
            (e[i] - rq).abs() <= 1e-12 * rq.abs().max(1.0),
            "e[{i}] = {:.16e} is not the Rayleigh quotient of the returned root ({rq:.16e})",
            e[i]
        );
    }
    let ov = info.overlaps[0][1];
    println!(
        "tms_d7 zero-iteration ladder: ‖roots‖ [{:.16e}, {:.16e}], |⟨0,1⟩| {ov:.16e}, ss_energies \
         {:?}, penalties {:?}, e {:?}, j_ss {:?}",
        roots[0].norm(),
        roots[1].norm(),
        info.ss_energies,
        info.penalties,
        e,
        info.j_ss
    );
    assert!(
        ov <= 1.0 + 1e-12,
        "|⟨root₀, root₁⟩| = {ov:.16e} exceeds 1 — the Gram was not built from unit states"
    );
    assert_eq!(info.penalties[0], 0.0, "root 0 has no lower state");
    let w_want = penalty_weight(info.ss_energies[1], &[ov], &e[0..1], 0.0);
    assert!(
        (info.penalties[1] - w_want).abs() <= 1e-12 * w_want.abs().max(1.0),
        "penalties[1] = {:.16e} but the Auto rule on the returned unit roots gives {w_want:.16e}",
        info.penalties[1]
    );
    let j_want = e[1] + info.penalties[1] * ov * ov;
    assert!(
        (info.j_ss[1] - j_want).abs() <= 1e-10 * j_want.abs().max(1.0),
        "j_ss[1] = {:.16e} but λ + w·o² on the returned unit roots is {j_want:.16e}",
        info.j_ss[1]
    );
}

fn ground<T: Scalar>(h: &SparseHamiltonian<T>, x0: &TtVector<T>) -> (f64, TtVector<T>) {
    state_specific_mals(
        h,
        x0,
        &[],
        &StateSpecificOptions {
            mals: opts_exact(),
            ..Default::default()
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn e1a_certified_ladder(
    h: &SparseHamiltonian<f64>,
    fx: &GappedFixture,
    e: &[f64],
    roots: &[TtVector<f64>],
    info: &qntt::solvers::excited::ExcitedInfo<TtVector<f64>>,
    opts: &ExcitedOpts,
    offset: usize,
    tag: &str,
) {
    for i in 0..e.len() {
        let want = fx.ev11[offset + i];
        println!(
            "{tag} root {i}: e {:.12e} vs dense {:.12e} (Δ {:.3e}), w {:.6e}, gate {:.3e}, \
             passes {}, SS λ {:.12e}",
            e[i],
            want,
            (e[i] - want).abs(),
            info.penalties[i],
            info.gradnorms[i],
            info.passes[i],
            info.ss_energies[i]
        );
        assert!(
            (e[i] - want).abs() <= 1e-8,
            "{tag} root {i}: e {:.12e} vs dense {want:.12e}",
            e[i]
        );
        let proj = projection_on_dense(&roots[i], &fx.dets11, &fx.evec11[offset + i]);
        assert!(
            proj >= 1.0 - 1e-8,
            "{tag} root {i} is not the dense eigenvector (projection {proj:.12})"
        );
        let res = global_eigen_residual_abelian(h, &roots[i], e[i]);
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
    let gram = gram_dev_tt(roots);
    assert!(
        gram <= 1e-6,
        "{tag}: the returned ladder is not orthonormal ({gram:.3e})"
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
    assert!(info.licence, "{tag}: the licence must hold at full rank");
}

#[test]
fn tms_e1a_fixed_and_project_drive_the_ladder() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xE1_A000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xE1_A001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let w_fix = 10.0 * (fx.ev11[3] - fx.ev11[0]);
    println!(
        "tms_e1a: w_fix = 10·(E₃ − E₀) = {w_fix:.6e}, exclusion span E₃ − E₀ = {:.6e}",
        fx.ev11[3] - fx.ev11[0]
    );
    assert!(
        w_fix > 10.0 * (fx.ev11[2] - fx.ev11[0]),
        "the fixed weight must exceed every rung's own exclusion energy"
    );

    let mut opts1 = base_opts(space, 3);
    opts1.mode = SsMode::Penalty;
    opts1.penalty = PenaltyRule::Fixed(w_fix);
    let (e1, roots1, info1) = excited_states(&h, &x0, &[], &opts1);
    e1a_certified_ladder(&h, &fx, &e1, &roots1, &info1, &opts1, 0, "tms_e1a run 1");
    assert_eq!(
        info1.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "run 1: the bottom rung has no lower states, so its polish weight is exactly 0 — the \
         `Fixed` rule is only read where there is something to penalize"
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
                "run 1 root {i} sweep {k}: the sweep ran at {:.17e}, not the fixed {w_fix:.17e}",
                s.penalty
            );
        }
        let last = info1.ss_history[i]
            .sweeps
            .last()
            .expect("a swept root records at least one iteration");
        let sig: f64 = last.overlaps.iter().map(|o| o * o).sum();
        println!(
            "tms_e1a run 1 root {i}: θ {:.16e}, λ {:.16e}, w·Σo² {:.3e}",
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
                    let o = qntt::dot(&roots1[jx], &roots1[i]).abs_s();
                    o * o
                })
                .sum::<f64>();
        assert!(
            (info1.j_pol[i] - e1[i] - pen).abs() <= 1e-10,
            "run 1 root {i}: J_pol − λ = {:.3e} but w·Σ|o|² = {pen:.3e}",
            info1.j_pol[i] - e1[i]
        );
        if info1.passes[i] == 0 {
            assert!(
                info1.j_pol[i] <= info1.j_ss[i] + 1e-10,
                "run 1 root {i}: the polish RAISED J at a fixed weight ({:.16e} → {:.16e})",
                info1.j_ss[i],
                info1.j_pol[i]
            );
        }
    }
    println!("tms_e1a run 1: passes {:?}", info1.passes);

    let mut opts2 = base_opts(space, 3);
    opts2.mode = SsMode::Project;
    opts2.penalty = PenaltyRule::Auto;
    let (e2, roots2, info2) = excited_states(&h, &x0, &[], &opts2);
    e1a_certified_ladder(&h, &fx, &e2, &roots2, &info2, &opts2, 0, "tms_e1a run 2");
    let mut skipped_total = 0usize;
    for i in 0..3 {
        for (k, s) in info2.ss_history[i].sweeps.iter().enumerate() {
            assert_eq!(
                s.penalty.to_bits(),
                0.0f64.to_bits(),
                "run 2 root {i} sweep {k}: `Project` records no weight, it records {:.3e}",
                s.penalty
            );
            assert_eq!(
                s.penalized.to_bits(),
                s.energy.to_bits(),
                "run 2 root {i} sweep {k}: θ must BE λ in `Project` ({:.17e} vs {:.17e})",
                s.penalized,
                s.energy
            );
            skipped_total += s.skipped;
        }
    }
    println!(
        "tms_e1a run 2: Σ skipped = {skipped_total}, penalties {:?}",
        info2.penalties
    );
    assert_eq!(
        skipped_total, 0,
        "run 2: at full rank from a rank-5..10 start every local space hosts one root plus the \
         projected directions, so no bond may be skipped"
    );
    assert_eq!(
        info2.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "run 2: the bottom rung has no lower states"
    );
    for i in 1..3 {
        assert!(
            info2.penalties[i] >= 1.0,
            "run 2 root {i}: the POLISH weight {:.3e} is below `penalty_weight`'s floor of 1 — in \
             `Project` the sweep needs no weight, but the polish still has to hold the root off \
             the ladder below it",
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
    e1a_certified_ladder(&h, &fx, &e3, &roots3, &info3, &opts3, 0, "tms_e1a run 3");
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
            "run 3 root {i}: the polish weight {:.17e} must be `Fixed`'s own {w_fix:.17e}, NOT \
             the 0 the `Project` sweep recorded",
            info3.penalties[i]
        );
    }

    let psi0 = roots1[0].clone();
    assert!(
        (rayleigh_quotient(&h, &psi0).re_s() - fx.ev11[0]).abs() <= 1e-8,
        "the control's lower state is not the dense ground state"
    );
    let mut opts4 = base_opts(space, 2);
    opts4.penalty = PenaltyRule::Fixed(w_fix);
    let (e4, roots4, info4) = excited_states(&h, &x0, std::slice::from_ref(&psi0), &opts4);
    println!(
        "tms_e1a control: e {e4:?}, penalties {:?}, lower_gradnorms {:?}, overlaps[0] {:?}",
        info4.penalties, info4.lower_gradnorms, info4.overlaps[0]
    );
    e1a_certified_ladder(&h, &fx, &e4, &roots4, &info4, &opts4, 1, "tms_e1a control");
    for i in 0..2 {
        assert_eq!(
            info4.penalties[i].to_bits(),
            w_fix.to_bits(),
            "control root {i}: with an external lower state EVERY rung polishes at the fixed \
             weight ({:.17e} vs {w_fix:.17e})",
            info4.penalties[i]
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
        "control: the supplied lower state is not converged (gate {:.3e}); the ladder above it \
         would be built against a wrong direction",
        info4.lower_gradnorms[0]
    );
}

#[test]
fn tms_e1b_stall_branch_alone_refuses_the_licence() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xE1_B000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xE1_B001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let full = max_ranks(&space);
    let widest: usize = space
        .bond_sectors(d / 2)
        .into_iter()
        .map(|s| full[d / 2][s])
        .max()
        .unwrap_or(0);
    println!("tms_e1b: widest mid-bond sector rank at full grid = {widest}");
    assert!(
        widest > 2,
        "the rank-2 cap does not bind on this space (widest {widest}); the arm would measure a \
         converged critical point, not a starved polish"
    );

    let mut opts = base_opts(space, 1);
    opts.rank = 2;
    opts.bond_dims = Some(vec![2]);
    opts.eps = 1e-6;
    opts.cg.max_iter = 0;
    opts.cg_restarts = 0;
    opts.polish = true;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_e1b: gate {:.3e} (grad_gate {:.1e}), stalled {:?}, converged {:?}, ss last \
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
        "the starved polish left a gate of only {:.3e} at a BINDING rank-2 cap; the arm no longer \
         isolates the stall with 1000× headroom over `grad_gate` — escalate this as a fixture \
         finding, do NOT lower the bound",
        info.gradnorms[0]
    );
    assert!(
        info.gradnorms[0] > opts.grad_gate,
        "the stall condition itself is `gradnorms[i] > grad_gate`"
    );
    assert_eq!(info.stalled, vec![true], "the root must be flagged stalled");
    assert!(
        info.ss_history[0].sweeps.last().unwrap().converged,
        "the state-specific sweep itself must have converged — otherwise TWO conjuncts of the \
         licence are false and the arm does not isolate the stall"
    );
    assert_eq!(
        info.converged,
        vec![false],
        "`converged[i] = ss_converged[i] ∧ !stalled[i]`"
    );
    assert_eq!(
        info.overlap_dev, 0.0,
        "a one-rung ladder has no off-diagonal Gram entry, so the overlap conjunct is EXACTLY \
         satisfied and cannot be what refuses the licence"
    );
    assert_eq!(info.swaps, 0, "a one-rung ladder cannot be out of order");
    assert!(info.swapped.is_empty(), "nothing to flag");
    assert!(
        !info.licence,
        "the licence must be refused by the stall conjunct alone"
    );
    assert_eq!(info.passes, vec![0], "no trim may fire at trim_tol = 1e-12");
    assert_eq!(
        info.penalties,
        vec![0.0],
        "with no lower states the polish weight is exactly 0"
    );
    let rq = rayleigh_quotient(&h, &roots[0]).re_s();
    assert!(
        (e[0] - rq).abs() <= 1e-12 * rq.abs().max(1.0),
        "e[0] = {:.16e} is not the Rayleigh quotient of the returned root ({rq:.16e})",
        e[0]
    );
    assert!(
        (roots[0].norm() - 1.0).abs() <= 1e-12,
        "the returned root is not unit ({:.3e})",
        roots[0].norm()
    );

    let mut ctrl = opts.clone();
    ctrl.polish = false;
    let (e_c, roots_c, info_c) = excited_states(&h, &x0, &[], &ctrl);
    println!(
        "tms_e1b control: gradnorms {:?}, stalled {:?}, converged {:?}, licence {}, e {:.12e}",
        info_c.gradnorms, info_c.stalled, info_c.converged, info_c.licence, e_c[0]
    );
    assert!(
        info_c.gradnorms[0].is_nan(),
        "`polish = false` records no exact gate"
    );
    assert_eq!(
        info_c.stalled,
        vec![false],
        "`stall = polish ∧ gate > grad_gate` short-circuits to false without a polish"
    );
    assert_eq!(
        info_c.converged,
        vec![true],
        "the sweep converged; with the stall conjunct gone the root is certified"
    );
    assert!(
        info_c.licence,
        "the flip of `polish` alone must flip the licence — that is what isolates the conjunct"
    );
    assert!(
        (roots_c[0].norm() - 1.0).abs() <= 1e-12,
        "the unpolished root must still be unit"
    );
}

#[test]
fn tms_e1b_stall_branch_with_a_live_gram() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xE1_B100, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xE1_B101);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let mut opts = base_opts(space, 2);
    opts.eps = 1e-6;
    opts.cg.max_iter = 0;
    opts.cg_restarts = 0;
    opts.overlap_tol = 1e-4;
    let (e, _roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_e1b live-Gram: e {e:?}, gates {:?}, overlap_dev {:.3e} (tol {:.1e}), swaps {}, \
         stalled {:?}, converged {:?}, licence {}",
        info.gradnorms,
        info.overlap_dev,
        opts.overlap_tol,
        info.swaps,
        info.stalled,
        info.converged,
        info.licence
    );
    assert_eq!(info.stalled, vec![true; 2], "both roots must stall");
    assert!(
        info.overlap_dev <= opts.overlap_tol,
        "the Gram conjunct must HOLD ({:.3e} vs {:.1e}), otherwise this arm does not isolate the \
         stall from the overlap branch",
        info.overlap_dev,
        opts.overlap_tol
    );
    assert_eq!(info.swaps, 0, "the two rungs are energy-ordered");
    assert!(!info.licence, "the stall alone refuses the licence");
    for i in 0..2 {
        assert!(
            info.gradnorms[i] > opts.grad_gate,
            "root {i}'s gate {:.3e} does not exceed grad_gate",
            info.gradnorms[i]
        );
        assert!(
            (e[i] - fx.ev11[i]).abs() <= 1e-6,
            "root {i}: even unpolished, the full-rank sweep must reach the dense level ({:.12e} \
             vs {:.12e})",
            e[i],
            fx.ev11[i]
        );
    }
}

#[test]
fn tms_e1b_overlap_branch_alone_refuses_the_licence() {
    use qntt::hamiltonian::assembly::Construction;
    use qntt::integrals::{IntegralsOptions, load_fixture_dir};

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let (d, nelec) = (ints.norb, ints.nelec);
    let (nup, ndn) = (nelec / 2, nelec / 2);
    let space = QnSpace::new(d, nup, ndn);
    let (t, v) = (ints.t, ints.v);
    let h = SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Median);
    let (dets, ev, _evec) = dense_sector(d, &t, &v, nup, ndn);
    println!(
        "tms_e1b overlap: {} determinants, E[0..4] = {:?}",
        dets.len(),
        &ev[..4]
    );

    let occ: Vec<bool> = (0..d).map(|i| i < nup).collect();
    let x0: TtVector<f64> = tt_state(&occ, &occ);

    let mut opts = base_opts(space, 3);
    opts.rank = 8;
    opts.bond_dims = None;
    opts.noise = vec![3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 0.0];
    opts.grad_gate = 1e-6;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_e1b overlap: e {e:?}, gates {:?}, passes {:?}, overlap_dev {:.3e} (tol {:.1e}), \
         swaps {}, converged {:?}, stalled {:?}, licence {}",
        info.gradnorms,
        info.passes,
        info.overlap_dev,
        opts.overlap_tol,
        info.swaps,
        info.converged,
        info.stalled,
        info.licence
    );
    for i in 0..3 {
        println!(
            "tms_e1b overlap root {i}: e {:.12e} vs dense {:.12e} (Δ {:.3e}, PRINTED — a \
             rank-8 root is not the dense level and is not asserted to be)",
            e[i],
            ev[i],
            e[i] - ev[i]
        );
        assert!(
            e[i] >= ev[i] - 1e-8,
            "root {i} = {:.12e} sits BELOW the dense level {:.12e}",
            e[i],
            ev[i]
        );
    }
    assert_eq!(info.converged, vec![true; 3], "every root converged");
    assert_eq!(info.stalled, vec![false; 3], "no root stalled");
    assert_eq!(info.swaps, 0, "the ladder is energy-ordered");
    assert!(info.swapped.is_empty(), "no rung is flagged");
    assert!(
        info.overlap_dev > opts.overlap_tol,
        "the binding cap did NOT leave a residual overlap above the {:.1e} gate ({:.3e}) — the \
         O(1/w) fixed-rank residual of decision 17 does not reproduce on this tree; escalate it \
         as a finding about the penalized minimizer, never loosen the gate to match",
        opts.overlap_tol,
        info.overlap_dev
    );
    assert!(
        !info.licence,
        "the licence must be refused by the overlap conjunct alone"
    );
    let gram = gram_dev_tt(&roots);
    assert!(
        (gram - info.overlap_dev).abs() <= 1e-15,
        "`overlap_dev` {:.17e} is not the measured max off-diagonal {gram:.17e}",
        info.overlap_dev
    );
}

#[test]
fn tms_e1f_polish_false_bookkeeping_and_keep_sa() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xE1_F000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xE1_F001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let mut opts = base_opts(space, 3);
    opts.polish = false;
    opts.keep_sa = true;
    opts.max_iter = 30;
    let (e, roots, info) = excited_states(&h, &x0, &[], &opts);

    println!(
        "tms_e1f: e {e:?}, ss_energies {:?}, passes {:?}, stalled {:?}, converged {:?}, \
         penalties {:?}, licence {}",
        info.ss_energies, info.passes, info.stalled, info.converged, info.penalties, info.licence
    );
    assert_eq!(
        info.passes,
        vec![0; 3],
        "no trim loop runs without a polish"
    );
    assert_eq!(
        info.stalled,
        vec![false; 3],
        "`stall = polish ∧ gate > grad_gate` short-circuits to false"
    );
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
            "tms_e1f root {i}: e {:.16e}, ss_energies {:.16e} (Δ {:.3e}), dense {:.12e}",
            e[i],
            info.ss_energies[i],
            e[i] - info.ss_energies[i],
            fx.ev11[i]
        );
        assert!(
            (e[i] - info.ss_energies[i]).abs() <= 1e-10,
            "root {i}: `e[i]` {:.16e} and the sweep's own λ {:.16e} differ by {:.3e} at FULL rank, \
             where the last bond's quotient is the whole train's",
            e[i],
            info.ss_energies[i],
            e[i] - info.ss_energies[i]
        );
        assert!(
            (e[i] - fx.ev11[i]).abs() <= 1e-8,
            "root {i}: the unpolished full-rank sweep must still reach the dense level"
        );
        assert!(
            (roots[i].norm() - 1.0).abs() <= 1e-12,
            "root {i} is not unit ({:.3e}) — nothing else normalizes it on this branch",
            roots[i].norm()
        );
        let pen: f64 = info.penalties[i]
            * (0..i)
                .map(|jx| {
                    let o = qntt::dot(&roots[jx], &roots[i]).abs_s();
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
            "root {i}: without a polish `J_ss` and `J_pol` are the same functional at the same \
             state ({:.16e} vs {:.16e})",
            info.j_ss[i],
            info.j_pol[i]
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
            "root {i}: the weight is COMPUTED and RECORDED even when no polish consumes it ({:.3e})",
            info.penalties[i]
        );
    }
    assert_eq!(
        info.licence,
        info.converged.iter().all(|&c| c)
            && info.overlap_dev <= opts.overlap_tol
            && info.swaps == 0,
        "`licence` is not the D9 conjunction"
    );

    let sa = info
        .sa_states
        .as_ref()
        .expect("`keep_sa = true` must record the SA ladder");
    assert_eq!(sa.len(), 3, "one SA state per root");
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
        let (hist, mut y) = state_specific_mals_with_history(&h, &sa[i], &roots[..i], &ss_opts);
        normalize_mut(&mut y);
        assert_f64_bits(
            info.ss_energies[i],
            hist.lambda,
            &format!("tms_e1f root {i}: ss_energies vs the replayed sweep's λ"),
        );
        coeffs_bits_eq_tt(
            &roots[i],
            &y,
            &format!("tms_e1f root {i}: the returned root vs the replayed sweep's output"),
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
        assert_f64_bits(e_n[i], e[i], &format!("tms_e1f keep_sa control: e[{i}]"));
        assert_f64_bits(
            info_n.ss_energies[i],
            info.ss_energies[i],
            &format!("tms_e1f keep_sa control: ss_energies[{i}]"),
        );
        coeffs_bits_eq_tt(
            &roots_n[i],
            &roots[i],
            &format!("tms_e1f keep_sa control: root {i}"),
        );
    }
    assert_eq!(
        info_n.licence, info.licence,
        "`keep_sa` is a reporting flag; it must have no effect on the computation"
    );
}

#[test]
fn tms_e1c_c64_ladder_matches_the_real_ladder() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xE1_C000, 3, 0.05);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &fx.t, &fx.v, &theta);
    let hc = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
    let mut rngc = seeded(0xE1_C001);
    let x0c: TtVector<C64> = randomized_state(space, 5, 10, &mut rngc);

    let opts = base_opts(space, 3);
    let (ec, rootsc, infoc) = excited_states_c64(&hc, &x0c, &[], &opts);

    let h = fx.abelian(space);
    let mut rngr = seeded(0xE1_C002);
    let x0r: TtVector<f64> = randomized_state(space, 5, 10, &mut rngr);
    let (er, _rootsr, infor) = excited_states(&h, &x0r, &[], &opts);
    assert!(
        infor.licence,
        "the real reference ladder must itself be certified"
    );

    let mut rho_min = f64::INFINITY;
    for i in 0..3 {
        let rho = rho_abelian(&rootsc[i]);
        rho_min = rho_min.min(rho);
        println!(
            "tms_e1c root {i}: C64 e {:.12e}, real e {:.12e}, dense {:.12e}, ρ {rho:.6}, \
             gate {:.3e}, w {:.3e}",
            ec[i], er[i], fx.ev11[i], infoc.gradnorms[i], infoc.penalties[i]
        );
        assert!(
            (ec[i] - fx.ev11[i]).abs() <= 1e-8,
            "C64 root {i}: {:.12e} vs the dense level {:.12e}",
            ec[i],
            fx.ev11[i]
        );
        assert!(
            (ec[i] - er[i]).abs() <= 1e-8,
            "C64 root {i}: {:.12e} vs the real ladder's {:.12e} — the gauge is a unitary, so the \
             two ladders are the same spectrum",
            ec[i],
            er[i]
        );
        let res = global_eigen_residual_abelian(&hc, &rootsc[i], ec[i]);
        assert!(
            res <= 1e-6,
            "C64 root {i} is not an eigenvector of H (residual {res:.3e})"
        );
        assert!(
            (rootsc[i].norm() - 1.0).abs() <= 1e-12,
            "C64 root {i} is not unit ({:.3e})",
            rootsc[i].norm()
        );
        assert!(
            infoc.gradnorms[i] <= opts.grad_gate,
            "C64 root {i} was returned above the exact gate ({:.3e})",
            infoc.gradnorms[i]
        );
    }
    assert!(
        rho_min < 0.99,
        "VACUOUS — every C64 root shares one global phase (min ρ = {rho_min:.6}); the complex \
         ladder is a real computation in disguise"
    );
    let gram = gram_dev_tt(&rootsc);
    assert!(
        gram <= 1e-6,
        "the C64 ladder is not orthonormal ({gram:.3e})"
    );
    assert_eq!(infoc.converged, vec![true; 3], "every C64 root converged");
    assert_eq!(infoc.stalled, vec![false; 3], "no C64 root stalled");
    assert_eq!(infoc.swaps, 0, "the C64 ladder is energy-ordered");
    assert!(infoc.swapped.is_empty(), "no rung flagged");
    assert!(infoc.licence, "the C64 ladder must be certified");
    assert_eq!(
        infoc.penalties[0].to_bits(),
        0.0f64.to_bits(),
        "the bottom rung has no lower states"
    );
    for i in 1..3 {
        let last = infoc.ss_history[i].sweeps.last().expect("recorded");
        let want = penalty_weight(infoc.ss_energies[i], &last.overlaps, &ec[..i], 0.0);
        assert!(
            (infoc.penalties[i] - want).abs() <= 1e-12 * want.abs().max(1.0),
            "C64 root {i}: the polish weight {:.16e} is not the `Auto` rule replayed from the \
             sweep's own (complex-modulus) overlaps ({want:.16e})",
            infoc.penalties[i]
        );
    }

    let (_l0, mut psic0) = ground(&hc, &x0c);
    psic0.scale_mut(C64::from_f64(2.0));
    println!("tms_e1c: lower state norm {:.16e}", psic0.norm());
    assert!(
        (psic0.norm() - 2.0).abs() <= 1e-10,
        "the non-unit lower state must have norm 2 ({:.16e})",
        psic0.norm()
    );
    let opts2 = base_opts(space, 2);
    let (e2, roots2, info2) = excited_states_c64(&hc, &x0c, std::slice::from_ref(&psic0), &opts2);
    println!(
        "tms_e1c non-unit lower: e {e2:?}, penalties {:?}, lower_gradnorms {:?}, Gram row 0 {:?}",
        info2.penalties, info2.lower_gradnorms, info2.overlaps[0]
    );
    for i in 0..2 {
        assert!(
            (e2[i] - fx.ev11[1 + i]).abs() <= 1e-8,
            "C64 root {i} above a supplied ground state: {:.12e} vs {:.12e}",
            e2[i],
            fx.ev11[1 + i]
        );
        assert!(
            (roots2[i].norm() - 1.0).abs() <= 1e-12,
            "C64 root {i} is not unit"
        );
        assert!(
            info2.penalties[i] >= 1.0,
            "root {i}: with an external lower state every rung carries the `Auto` weight ({:.3e})",
            info2.penalties[i]
        );
    }
    assert!(
        (info2.overlaps[0][0] - 1.0).abs() <= 1e-10,
        "the norm-2 lower state's Gram diagonal is {:.16e}, not 1 — `invn` is not folded into \
         both indices of the complex Gram",
        info2.overlaps[0][0]
    );
    for b in 1..3 {
        assert!(
            info2.overlaps[0][b] <= 1e-6,
            "C64 root {} is not orthogonal to the norm-2 lower state ({:.3e})",
            b - 1,
            info2.overlaps[0][b]
        );
    }
    assert!(
        info2.lower_gradnorms[0] <= 1e-6,
        "the supplied complex lower state is not converged (gate {:.3e}) — the gate is computed \
         on the NORMALIZED state, so a norm of 2 must not enter it",
        info2.lower_gradnorms[0]
    );
}

#[test]
fn tms_e1c_c64_phased_lower_state_folds_by_its_modulus() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xE1_C100, 3, 0.05);
    let theta: Vec<f64> = (0..d).map(|p| 0.2 + 0.5 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &fx.t, &fx.v, &theta);
    let hc = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
    let mut rng = seeded(0xE1_C101);
    let x0: TtVector<C64> = randomized_state(space, 5, 10, &mut rng);

    let mut xhat = x0.clone();
    if xhat.ortho != Some(0) {
        xhat.right_orthogonalize(true);
    }
    normalize_mut(&mut xhat);
    let mut psi = xhat.clone();
    psi.scale_mut(C64::i());
    assert!(
        (psi.norm() - 1.0).abs() <= 1e-12,
        "the phased lower state is not unit ({:.16e})",
        psi.norm()
    );

    let mut opts = base_opts(space, 1);
    opts.max_iter = 0;
    opts.polish = false;
    let (e, roots, info) = excited_states_c64(&hc, &x0, std::slice::from_ref(&psi), &opts);
    let same = qntt::dot(&roots[0], &xhat).abs_s();
    assert!(
        same >= 1.0 - 1e-12,
        "tms_e1c phased: the zero-iteration root is not the normalized start (|⟨root, x̂⟩| = \
         {same:.16e})"
    );

    let o = qntt::dot(&psi, &roots[0]);
    println!(
        "tms_e1c phased: ⟨ψ̂, root⟩ = {:.16e} + {:.16e}i (|o| {:.16e}), Gram row 0 {:?}, e {:.16e}, \
         j_ss {:.16e}, w {:.16e}",
        o.re_s(),
        o.im_s(),
        o.abs_s(),
        info.overlaps[0],
        e[0],
        info.j_ss[0],
        info.penalties[0]
    );
    assert!(
        o.abs_s() >= 1.0 - 1e-12,
        "the phased lower state is not the root up to a phase (|o| = {:.16e})",
        o.abs_s()
    );
    assert!(
        o.re_s().abs() <= 1e-10,
        "VACUOUS — the overlap's REAL part is {:.3e}, so `re_s().abs()` and `abs_s()` do not \
         disagree here and the arm discriminates nothing",
        o.re_s()
    );
    assert!(
        info.overlaps[0][1] >= 1.0 - 1e-12,
        "the Gram entry between the lower state and its own phase rotation is {:.16e}, not 1 — \
         the complex overlap was not reduced by its MODULUS",
        info.overlaps[0][1]
    );
    assert!(
        info.overlap_dev >= 1.0 - 1e-12,
        "`overlap_dev` is {:.16e}: a root that duplicates the lower state up to a phase must \
         saturate it",
        info.overlap_dev
    );
    assert!(
        !info.licence,
        "the licence was granted on a root that IS the lower state up to a phase"
    );
    assert!(
        (info.j_ss[0] - e[0] - info.penalties[0]).abs() <= 1e-10 * info.penalties[0].abs().max(1.0),
        "J_ss − λ = {:.16e} but the weight is {:.16e}: `Σ|⟨ψ̂, x⟩|²` is not 1 at a root that is \
         the lower state times `i`",
        info.j_ss[0] - e[0],
        info.penalties[0]
    );
    assert!(
        info.penalties[0] > 0.0,
        "a rung with a lower state must carry a weight ({:.3e})",
        info.penalties[0]
    );
}
