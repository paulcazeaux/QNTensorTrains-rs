// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::C64;
use qntt::core::SparseCore;
use qntt::core::twosite::factor_svd_average;
use qntt::solvers::mals::{MalsOptions, NoiseType, mals, mals_with_history};
use qntt::solvers::multistate::{
    MultiTt, StateAverageOptions, state_average_mals, state_average_mals_mut,
};
use qntt::solvers::ssmals::{
    PenaltyRule, SsHistory, SsMode, StateSpecificOptions, penalty_weight, state_specific_mals,
    state_specific_mals_with_history,
};
use qntt::{RoundMode, rayleigh_quotient, tt_state};
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

fn ss_opts(mals: MalsOptions) -> StateSpecificOptions {
    StateSpecificOptions {
        mals,
        ..Default::default()
    }
}

fn ss_opts_mode(mals: MalsOptions, mode: SsMode, penalty: PenaltyRule) -> StateSpecificOptions {
    StateSpecificOptions {
        mals,
        mode,
        penalty,
        ..Default::default()
    }
}

fn ground<T: Scalar>(h: &SparseHamiltonian<T>, x0: &TtVector<T>) -> (f64, TtVector<T>) {
    state_specific_mals(h, x0, &[], &ss_opts(opts_exact()))
}

fn e_ref<T: Scalar>(h: &SparseHamiltonian<T>, psi: &TtVector<T>) -> f64 {
    let n = psi.norm();
    let rq = rayleigh_quotient(h, psi).re_s();
    if n == 1.0 {
        rq
    } else {
        let invn = 1.0 / n;
        rq * invn * invn
    }
}

fn dense_unit<T: Scalar>(x: &TtVector<T>, dets: &[(Vec<bool>, Vec<bool>)], tag: &str) -> Vec<f64> {
    let xd = x.to_dense();
    let full: f64 = xd.iter().map(|c| c.abs_s().powi(2)).sum::<f64>().sqrt();
    let mut c: Vec<f64> = dets
        .iter()
        .map(|(ou, od)| xd[basis_index(ou, od)].re_s())
        .collect();
    let imax = dets
        .iter()
        .map(|(ou, od)| xd[basis_index(ou, od)].im_s().abs())
        .fold(0.0f64, f64::max);
    assert!(
        imax <= 1e-12 * full.max(1e-300),
        "{tag}: dense_unit is a REAL oracle but the state carries an imaginary part {imax:.3e}"
    );
    let nrm = c.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        nrm >= (1.0 - 1e-10) * full,
        "{tag}: the state leaks outside the determinant sector ({nrm:.12e} of {full:.12e})"
    );
    for v in c.iter_mut() {
        *v /= nrm.max(1e-300);
    }
    c
}

fn penalized_dense(hd: &[Vec<f64>], chats: &[Vec<f64>], w: f64) -> Vec<Vec<f64>> {
    let n = hd.len();
    let mut m = hd.to_vec();
    for c in chats {
        assert_eq!(c.len(), n, "penalized_dense: dimension mismatch");
        for i in 0..n {
            for j in 0..n {
                m[i][j] += w * c[i] * c[j];
            }
        }
    }
    m
}

fn cs_bound<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    theta: f64,
    w: f64,
    psi: &TtVector<T>,
    e: f64,
) -> (f64, f64, f64) {
    let c = qntt::dot(psi, x);
    let y = &(h * x) + &(psi * (T::from_f64(w) * c));
    let rpen = (&y - &(x * T::from_f64(theta))).norm_lr(false);
    let r1 = (&(h * psi) - &(psi * T::from_f64(e))).norm_lr(false);
    let denom = (e + w - theta).abs().max(1e-300);
    ((rpen + r1) / denom, rpen, r1)
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

fn narrow_gap_tv(seed0: u64) -> GappedFixture {
    let d = 6usize;
    for attempt in 0..256u64 {
        let seed = seed0 + attempt;
        let mut rng = seeded(seed);
        let t = sym_t(d, &mut rng);
        let v = sparse_sym_v(d, &mut rng);
        let (dets11, ev11, evec11) = dense_sector(d, &t, &v, 1, 1);
        let (dets20, ev20, evec20) = dense_sector(d, &t, &v, 2, 0);
        let singlets = singlet_indices(&ev11, &ev20);
        if ev11.len() < 4 {
            continue;
        }
        let g1 = ev11[1] - ev11[0];
        let g2 = ev11[2] - ev11[1];
        if g1 >= 0.05 && (0.02..=0.04).contains(&g2) {
            println!(
                "narrow_gap_tv: seed {seed:#x} (attempt {attempt}) — E₀..E₃ = {:?}, \
                 gaps {g1:.4} / {g2:.4}",
                &ev11[..4]
            );
            return GappedFixture {
                seed,
                t,
                v,
                dets11,
                ev11,
                evec11,
                dets20,
                ev20,
                evec20,
                singlets,
            };
        }
    }
    panic!(
        "narrow_gap_tv: no (t,v) with a ≥ 0.05 first gap and a 0.02..0.04 second gap in 256 seeds \
         from {seed0:#x} — a FIXTURE finding; change the generator, never the window"
    );
}

fn multiplet_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 3..6 {
        t.set(&[i, i], 3.0);
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        v.set(&[i, i, i, i], 2.0);
    }
    for i in 0..3 {
        for j in 0..3 {
            if i != j {
                v.set(&[i, j, j, i], 1.0);
            }
        }
    }
    (t, v)
}

#[test]
fn tms_c2_empty_lower_single_root_is_mals_bitwise() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0x7B01);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    assert!(x0.check());

    let noise_free = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 6,
        ..Default::default()
    };
    let ramp = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![4, 0],
        noise: vec![1e-3, 1e-4, 0.0, 0.0],
        max_iter: 6,
        ..Default::default()
    };
    let random = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![1e-4, 1e-4, 0.0],
        noise_type: NoiseType::Random,
        max_iter: 6,
        ..Default::default()
    };

    for (arm, opts) in [
        ("noise-free", noise_free),
        ("perturbative-ramp", ramp),
        ("random", random),
    ] {
        let (mhist, my) = mals_with_history(&h, &x0, &opts);
        let (shist, sy) = state_specific_mals_with_history(&h, &x0, &[], &ss_opts(opts.clone()));

        assert_eq!(
            shist.sweeps.len(),
            mhist.sweeps.len(),
            "{arm}: iteration count (the schedule and the stop test must be bitwise MALS's)"
        );
        assert_f64_bits(shist.lambda, mhist.lambda, &format!("{arm}: final λ"));

        for (it, (s, m)) in shist.sweeps.iter().zip(mhist.sweeps.iter()).enumerate() {
            let tag = |w: &str| format!("{arm}: sweep {it} {w}");
            assert_f64_bits(s.energy, m.energy, &tag("energy"));
            assert_f64_bits(
                s.penalized,
                m.energy,
                &tag("penalized == energy on the delegate"),
            );
            assert_f64_bits(
                s.discarded_weight,
                m.discarded_weight,
                &tag("discarded_weight"),
            );
            assert_eq!(s.max_bond_dim, m.max_bond_dim, "{}", tag("max_bond_dim"));
            assert_eq!(s.bond_dim, m.bond_dim, "{}", tag("bond_dim"));
            assert_f64_bits(s.noise, m.noise, &tag("noise"));
            assert!(s.overlaps.is_empty(), "{}", tag("overlaps must be empty"));
            assert_eq!(s.penalty, 0.0, "{}", tag("penalty must be 0"));
            assert_eq!(s.swaps, 0, "{}", tag("swaps"));
            assert_eq!(s.skipped, 0, "{}", tag("skipped"));
            assert_eq!(
                s.numops,
                0,
                "{}",
                tag("numops is documented as 0 on the delegate path")
            );
        }

        for (it, s) in shist.sweeps.iter().enumerate() {
            if it + 1 < shist.sweeps.len() {
                assert!(
                    !s.converged,
                    "{arm}: sweep {it} must not be marked converged"
                );
            }
        }
        let last = shist.sweeps.last().expect("at least one sweep");
        if shist.sweeps.len() < opts.max_iter {
            assert!(
                last.converged,
                "{arm}: the run stopped early, so the last sweep must record the stop test firing"
            );
        }
        if !last.converged {
            assert_eq!(
                shist.sweeps.len(),
                opts.max_iter,
                "{arm}: a non-converged run must have exhausted max_iter"
            );
        }

        coeffs_bits_eq_tt(&sy, &my, &format!("{arm}: returned state"));

        let (slam, _) = state_specific_mals(&h, &x0, &[], &ss_opts(opts.clone()));
        let (mlam, _) = mals(&h, &x0, &opts);
        assert_f64_bits(slam, mlam, &format!("{arm}: state_specific_mals λ"));
    }
}

#[test]
fn tms_c2_fixed_zero_penalty_sweep_is_mals_bitwise() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC2_0000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC2_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);

    let noise_free = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 5,
        ..Default::default()
    };
    let ramp = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![6, 0],
        noise: vec![1e-3, 1e-4, 0.0, 0.0],
        max_iter: 5,
        ..Default::default()
    };
    let random = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        noise: vec![1e-4, 1e-4, 0.0],
        noise_type: NoiseType::Random,
        max_iter: 5,
        ..Default::default()
    };
    for (arm, opts) in [
        ("noise-free", noise_free),
        ("perturbative-ramp", ramp),
        ("random", random),
    ] {
        let (mhist, my) = mals_with_history(&h, &x0, &opts);
        let o = ss_opts_mode(opts.clone(), SsMode::Penalty, PenaltyRule::Fixed(0.0));
        let (shist, sy) =
            state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);

        assert_eq!(
            shist.sweeps.len(),
            mhist.sweeps.len(),
            "{arm}: iteration count"
        );
        assert_f64_bits(shist.lambda, mhist.lambda, &format!("{arm}: final λ"));
        for (it, (s, m)) in shist.sweeps.iter().zip(mhist.sweeps.iter()).enumerate() {
            assert_f64_bits(s.energy, m.energy, &format!("{arm}: sweep {it} energy"));
            assert_f64_bits(
                s.penalized,
                s.energy,
                &format!("{arm}: sweep {it} θ == λ at w = 0"),
            );
            assert_f64_bits(
                s.discarded_weight,
                m.discarded_weight,
                &format!("{arm}: sweep {it} discarded_weight"),
            );
            assert_eq!(s.penalty, 0.0, "{arm}: sweep {it} penalty");
            assert_eq!(s.swaps, 0, "{arm}: sweep {it} swaps");
            assert_eq!(s.skipped, 0, "{arm}: sweep {it} skipped");
            assert!(
                s.numops >= 1,
                "{arm}: sweep {it} numops must count on the sweep path"
            );
            assert_eq!(s.overlaps.len(), 1, "{arm}: one overlap per lower state");
        }
        coeffs_bits_eq_tt(&sy, &my, &format!("{arm}: returned state"));

        let last = shist.sweeps.last().unwrap();
        let measured = qntt::dot(&psi1, &sy).abs_s() / psi1.norm();
        assert!(
            (last.overlaps[0] - measured).abs() <= 1e-14,
            "{arm}: recorded overlap {:.17e} vs measured {measured:.17e}",
            last.overlaps[0]
        );
        assert!(
            last.overlaps[0] >= 1.0 - 1e-6,
            "{arm}: at w = 0 the state must stay ψ₁ (overlap {:.6})",
            last.overlaps[0]
        );
    }
}

#[test]
fn tms_c3_penalty_and_project_ladders_match_the_dense_spectrum() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC3_0000, 3, 0.05);
    assert_exclusion_gap(&fx.ev11, 3, 0.05, "tms_c3");
    let h = fx.abelian(space);
    let mut rng = seeded(0xC3_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let (lam1, psi1) = ground(&h, &x0);
    assert!(
        (lam1 - fx.ev11[0]).abs() <= 1e-8,
        "rung 0: λ = {lam1:.12e} vs E₀ = {:.12e}",
        fx.ev11[0]
    );

    for mode in [SsMode::Penalty, SsMode::Project] {
        let tag = format!("tms_c3 {mode:?}");
        let mut states = vec![psi1.clone()];
        let mut lambdas = vec![lam1];
        for rung in 1..3usize {
            let e_lower: Vec<f64> = states.iter().map(|p| e_ref(&h, p)).collect();
            let o = StateSpecificOptions {
                mals: opts_exact(),
                mode,
                penalty: PenaltyRule::Auto,
                roots: 1,
                lower_energies: Some(e_lower.clone()),
                ..Default::default()
            };
            let (hist, y) = state_specific_mals_with_history(&h, &x0, &states, &o);
            let tag = format!("{tag} rung {rung}");

            assert!(y.check(), "{tag}: returned state fails check()");
            assert!(
                (y.norm() - 1.0).abs() <= 1e-10,
                "{tag}: ‖x‖ = {:.12e}",
                y.norm()
            );
            assert!(
                (hist.lambda - fx.ev11[rung]).abs() <= 1e-8,
                "{tag}: plain λ = {:.12e} vs E{rung} = {:.12e}",
                hist.lambda,
                fx.ev11[rung]
            );
            let rq = rayleigh_quotient(&h, &y).re_s() / (y.norm() * y.norm());
            assert!(
                (rq - hist.lambda).abs() <= 1e-8,
                "{tag}: RQ {rq:.12e} vs recorded λ {:.12e}",
                hist.lambda
            );
            let res = global_eigen_residual_abelian(&h, &y, hist.lambda);
            let last_w = hist.sweeps.last().unwrap().penalty;
            let sigma: f64 = hist
                .sweeps
                .last()
                .unwrap()
                .overlaps
                .iter()
                .map(|o| o * o)
                .sum();
            println!(
                "{tag}: λ = {:.12e}, residual = {res:.3e} (penalty floor w·(√Σo² + Σo²) = {:.3e})",
                hist.lambda,
                last_w * (sigma.sqrt() + sigma)
            );
            assert!(res <= 5e-6, "{tag}: global eigen residual {res:.3e}");
            let proj = projection_on_dense(&y, &fx.dets11, &fx.evec11[rung]);
            assert!(
                proj >= 1.0 - 1e-8,
                "{tag}: projection on the dense eigenvector {proj:.12e}"
            );

            let last = hist.sweeps.last().expect("at least one sweep recorded");
            assert!(last.converged, "{tag}: the last sweep must be converged");
            for (it, s) in hist.sweeps.iter().enumerate() {
                assert_eq!(s.noise, 0.0, "{tag}: sweep {it} noise");
                assert_eq!(s.bond_dim, 0, "{tag}: sweep {it} bond_dim");
                assert_eq!(s.skipped, 0, "{tag}: sweep {it} skipped (full-rank frames)");
                assert_eq!(s.swaps, 0, "{tag}: sweep {it} swaps at roots = 1");
                assert!(s.numops >= 1, "{tag}: sweep {it} numops");
                assert_eq!(s.overlaps.len(), rung, "{tag}: sweep {it} overlaps length");
                if it + 1 < hist.sweeps.len() {
                    assert!(!s.converged, "{tag}: sweep {it} converged early");
                }
                match mode {
                    SsMode::Penalty => {
                        assert!(
                            s.penalty >= 1.0,
                            "{tag}: sweep {it} penalty {} below the floor 1",
                            s.penalty
                        );
                        assert!(
                            s.penalized >= s.energy - 1e-12,
                            "{tag}: sweep {it} θ {} below λ {}",
                            s.penalized,
                            s.energy
                        );
                    }
                    SsMode::Project => {
                        assert_eq!(s.penalty, 0.0, "{tag}: sweep {it} penalty in Project");
                        assert_f64_bits(
                            s.penalized,
                            s.energy,
                            &format!("{tag}: sweep {it} θ == λ in Project"),
                        );
                    }
                }
            }
            if matches!(mode, SsMode::Penalty) {
                assert_rule_replay(&hist, &e_lower, &tag);
            }

            for (j, p) in states.iter().enumerate() {
                let measured = qntt::dot(p, &y).abs_s() / p.norm();
                assert!(
                    (last.overlaps[j] - measured).abs() <= 1e-14,
                    "{tag}: overlaps[{j}] recorded {:.17e} vs measured {measured:.17e}",
                    last.overlaps[j]
                );
            }

            states.push(y);
            lambdas.push(hist.lambda);
        }

        let gram = gram_dev_tt(&states);
        let floor = if matches!(mode, SsMode::Project) {
            1e-10
        } else {
            1e-6
        };
        println!("{tag}: λ = {lambdas:?}, Gram deviation = {gram:.3e}");
        assert!(
            gram <= floor,
            "{tag}: Gram deviation {gram:.3e} above {floor:.1e}"
        );
    }
}

#[test]
fn tms_c3_c64_second_root_matches_the_dense_spectrum() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC3_1000, 2, 0.05);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &fx.t, &fx.v, &theta);
    let hc = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
    let mut rng = seeded(0xC3_1001);
    let x0: TtVector<C64> = randomized_state(space, 5, 10, &mut rng);

    let (lam1, psi1) = ground(&hc, &x0);
    assert!(
        (lam1 - fx.ev11[0]).abs() <= 1e-8,
        "C64 rung 0: λ = {lam1:.12e} vs E₀ = {:.12e}",
        fx.ev11[0]
    );
    let e_lower = vec![e_ref(&hc, &psi1)];
    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(e_lower.clone()),
        ..Default::default()
    };
    let (hist, y) = state_specific_mals_with_history(&hc, &x0, std::slice::from_ref(&psi1), &o);
    assert!(
        (hist.lambda - fx.ev11[1]).abs() <= 1e-8,
        "C64 rung 1: λ = {:.12e} vs E₁ = {:.12e}",
        hist.lambda,
        fx.ev11[1]
    );
    let rq = rayleigh_quotient(&hc, &y).re_s() / (y.norm() * y.norm());
    assert!((rq - hist.lambda).abs() <= 1e-8, "C64: RQ {rq:.12e}");
    let ov = qntt::dot(&psi1, &y).abs_s();
    assert!(ov <= 1e-6, "C64: |⟨ψ₁,x⟩| = {ov:.3e}");
    let rho = rho_abelian(&y);
    println!("tms_c3 C64: λ = {:.12e}, ρ = {rho:.6}", hist.lambda);
    assert!(rho < 0.99, "C64: VACUOUS — ρ = {rho:.6} (one global phase)");
    assert_rule_replay(&hist, &e_lower, "tms_c3 C64");
}

#[test]
fn tms_c4_degenerate_ground_space_second_root() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let span: [(usize, usize); 4] = [(0, 1), (2, 4), (1, 0), (4, 2)];
    let zs: Vec<TtVector<f64>> = span
        .iter()
        .map(|&(u, dn)| state_from_sites::<f64>(d, &[u], &[dn]))
        .collect();

    let mut last: Option<(SsHistory, TtVector<f64>, TtVector<f64>)> = None;
    for trial in 0..3u64 {
        let mut r1 = seeded(0xC4_0000 + trial);
        let x1: TtVector<f64> = randomized_state(space, 5, 10, &mut r1);
        let (lam1, psi1) = ground(&h, &x1);
        if (lam1 + 0.5).abs() > 1e-6 {
            println!("tms_c4 trial {trial}: ground run reached {lam1:.6}, retrying");
            continue;
        }
        let mut r2 = seeded(0xC4_1000 + trial);
        let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut r2);
        let o = StateSpecificOptions {
            mals: MalsOptions {
                eps: 1e-8,
                conv_tol: Some(1e-12),
                bond_dims: vec![0],
                noise: vec![0.0],
                max_iter: 20,
                ..Default::default()
            },
            lower_energies: Some(vec![e_ref(&h, &psi1)]),
            ..Default::default()
        };
        let (hist, y) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);
        let ok = (hist.lambda + 0.5).abs() <= 1e-6;
        last = Some((hist, y, psi1));
        if ok {
            break;
        }
        println!(
            "tms_c4 trial {trial}: SS root reached {:.6}, retrying",
            last.as_ref().unwrap().0.lambda
        );
    }

    let (hist, y, psi1) = last.expect("tms_c4: no attempt produced a history");
    assert!(
        (hist.lambda + 0.5).abs() <= 1e-8,
        "tms_c4: the penalized second root must sit at the degenerate ground energy (got {:.12e})",
        hist.lambda
    );
    let ov = qntt::dot(&psi1, &y).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c4: |⟨ψ̂₁, x⟩| = {ov:.3e}");
    let mut span_weight = 0.0f64;
    for z in zs.iter() {
        let c = qntt::dot(&y, z).abs_s();
        span_weight += c * c;
    }
    assert!(
        (span_weight - 1.0).abs() <= 1e-8,
        "tms_c4: Σ_z |⟨y, z⟩|² = {span_weight:.12e} — the root left the analytic span"
    );
    let res = global_eigen_residual_abelian(&h, &y, hist.lambda);
    println!(
        "tms_c4: λ = {:.12e}, residual = {res:.3e}, span weight = {span_weight:.12e}",
        hist.lambda
    );
    assert!(res <= 1e-5, "tms_c4: residual {res:.3e}");
    let last_sweep = hist.sweeps.last().unwrap();
    assert_eq!(
        last_sweep.penalty, 1.0,
        "tms_c4: on a degenerate ground space λ⊥ = E₁ and `max(10·0, 1)` must return exactly 1"
    );
}

#[test]
fn tms_c5_truncated_rank_dense_penalized_and_cs_bounds() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC5_0000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC5_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (lam1, psi1) = ground(&h, &x0);
    assert!(
        (lam1 - fx.ev11[0]).abs() <= 1e-8,
        "tms_c5: ψ₁ is not the ground state"
    );
    let e1 = e_ref(&h, &psi1);
    let chat = dense_unit(&psi1, &fx.dets11, "tms_c5 ψ₁");
    let hd = dense_hamiltonian_matrix(d, &fx.t, &fx.v, &fx.dets11);

    for cap in [3usize, 4usize] {
        let tag = format!("tms_c5 cap {cap}");
        let o = StateSpecificOptions {
            mals: MalsOptions {
                eps: 1e-8,
                bond_dims: vec![cap],
                noise: vec![0.0],
                max_iter: 12,
                ..Default::default()
            },
            lower_energies: Some(vec![e1]),
            ..Default::default()
        };
        let (hist, x) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);

        let mut worst_rise = f64::NEG_INFINITY;
        for (it, s) in hist.sweeps.iter().enumerate() {
            assert_eq!(s.noise, 0.0, "{tag}: sweep {it} noise");
            assert_eq!(s.bond_dim, cap, "{tag}: sweep {it} bond_dim");
            assert_eq!(s.swaps, 0, "{tag}: sweep {it} swaps at roots = 1");
            assert!(
                s.max_bond_dim <= cap,
                "{tag}: sweep {it} max_bond_dim {} above the cap",
                s.max_bond_dim
            );
            assert!(
                s.penalized >= s.energy - 1e-12,
                "{tag}: sweep {it} θ {} below λ {}",
                s.penalized,
                s.energy
            );
            let lmin = dense_eigen(&penalized_dense(
                &hd,
                std::slice::from_ref(&chat),
                s.penalty,
            ))
            .0[0];
            assert!(
                s.penalized >= lmin - 1e-10,
                "{tag}: sweep {it} θ = {:.12e} below the DENSE λ_min(H + {}·ψ̂ψ̂ᵀ) = {lmin:.12e}",
                s.penalized,
                s.penalty
            );
            if it > 0 {
                worst_rise = worst_rise.max(s.penalized - hist.sweeps[it - 1].penalized);
            }
        }
        assert_rule_replay(&hist, &[e1], &tag);
        let last = hist.sweeps.last().unwrap();
        assert_f64_bits(hist.lambda, last.energy, &format!("{tag}: hist.lambda"));
        assert_eq!(x.ortho, Some(0), "{tag}: the returned train sits at core 0");
        assert!(
            (x.norm() - 1.0).abs() <= 1e-12,
            "{tag}: ‖x‖ = {:.12e}",
            x.norm()
        );

        let (bound, rpen, r1) = cs_bound(&h, &x, last.penalized, last.penalty, &psi1, e1);
        let measured = qntt::dot(&psi1, &x).abs_s() / psi1.norm();
        println!(
            "{tag}: λ = {:.12e} (E₁ = {:.12e}), w = {:.3}, |⟨ψ̂₁,x⟩| = {measured:.3e}, \
             CS bound = {bound:.3e} (rpen {rpen:.3e}, r₁ {r1:.3e}); max θ rise across sweeps = \
             {worst_rise:+.3e}",
            hist.lambda, fx.ev11[1], last.penalty
        );
        assert!(
            measured <= bound + 1e-12,
            "{tag}: |⟨ψ̂₁,x⟩| = {measured:.6e} exceeds the Cauchy–Schwarz bound {bound:.6e}"
        );
        assert!(
            (last.overlaps[0] - measured).abs() <= 1e-14,
            "{tag}: recorded overlap {:.17e} vs measured {measured:.17e}",
            last.overlaps[0]
        );
    }

    let sa = StateAverageOptions {
        mals: MalsOptions {
            eps: 1e-8,
            bond_dims: vec![4],
            noise: vec![0.0],
            max_iter: 6,
            ..Default::default()
        },
        ..Default::default()
    };
    let (lsa, xsa) = state_average_mals(&h, &x0, 2, &sa);
    let o = StateSpecificOptions {
        mals: MalsOptions {
            eps: 1e-8,
            bond_dims: vec![4],
            noise: vec![0.0],
            max_iter: 12,
            ..Default::default()
        },
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (he, xe) = state_specific_mals_with_history(&h, &xsa[1], std::slice::from_ref(&psi1), &o);
    println!(
        "tms_c5 SA start (cap 4): E_SS = {:.12e}, E_SA[1] = {:.12e}, E₂ = {:.12e} \
         (the SS ≤ SA comparison has no oracle at a binding cap and is NOT asserted)",
        he.lambda, lsa[1], fx.ev11[1]
    );
    let last = he.sweeps.last().unwrap();
    let lmin = dense_eigen(&penalized_dense(
        &hd,
        std::slice::from_ref(&chat),
        last.penalty,
    ))
    .0[0];
    assert!(
        last.penalized >= lmin - 1e-10,
        "tms_c5 SA start: θ = {:.12e} below the dense penalized minimum {lmin:.12e}",
        last.penalized
    );
    let (bound, _, _) = cs_bound(&h, &xe, last.penalized, last.penalty, &psi1, e1);
    let measured = qntt::dot(&psi1, &xe).abs_s() / psi1.norm();
    assert!(
        measured <= bound + 1e-12,
        "tms_c5 SA start: |⟨ψ̂₁,x⟩| = {measured:.6e} exceeds the CS bound {bound:.6e}"
    );
    assert_rule_replay(&he, &[e1], "tms_c5 SA start");
}

#[test]
fn tms_c6_penalty_weight_unit_cases() {
    assert_eq!(penalty_weight(-3.0, &[], &[], 7.0), 0.0, "empty lower list");
    assert_eq!(
        penalty_weight(-3.0, &[0.0], &[-5.0], 0.0),
        20.0,
        "orthogonal start"
    );
    assert_eq!(
        penalty_weight(-5.0, &[0.0], &[-5.0], 0.0),
        1.0,
        "floor at λ⊥ = E"
    );
    assert_eq!(
        penalty_weight(-5.0, &[1.0], &[-5.0], 0.0),
        1.0,
        "σ = 1 fallback"
    );
    assert_eq!(
        penalty_weight(-5.0, &[0.9], &[-5.0], 3.0),
        30.0,
        "collapse escalation ≥ 10·w_prev"
    );
    assert_eq!(
        penalty_weight(-5.0, &[0.5], &[-5.0], 3.0),
        1.0,
        "no escalation at σ = 1/4"
    );
    let got = penalty_weight(-3.0, &[0.6], &[-5.0], 0.0);
    let sigma = 0.6f64 * 0.6;
    let want = (10.0 * ((-3.0 - (-5.0) * sigma) / (1.0 - sigma) + 5.0)).max(1.0);
    assert!(
        (got - want).abs() <= 1e-12 * want.abs(),
        "partial overlap: {got:.17e} vs {want:.17e}"
    );
    let got = penalty_weight(-2.0, &[0.0, 0.0], &[-4.0, -6.0], 0.0);
    assert_eq!(got, 40.0, "min over the lower energies");
}

#[test]
#[should_panic(expected = "penalty_weight")]
fn tms_c6_penalty_weight_refuses_mismatched_lengths() {
    let _ = penalty_weight(-1.0, &[0.1, 0.2], &[-5.0], 0.0);
}

#[test]
fn tms_c6_default_reference_energies_match_lower_energies() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC6_0000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC6_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);
    let e1 = e_ref(&h, &psi1);

    let schedule = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 4,
        ..Default::default()
    };
    let (h_default, _) = state_specific_mals_with_history(
        &h,
        &x0,
        std::slice::from_ref(&psi1),
        &ss_opts(schedule.clone()),
    );
    let o = StateSpecificOptions {
        mals: schedule,
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (h_given, _) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);

    assert_eq!(
        h_default.sweeps.len(),
        h_given.sweeps.len(),
        "the two runs must take the same number of iterations"
    );
    for (it, (a, b)) in h_default
        .sweeps
        .iter()
        .zip(h_given.sweeps.iter())
        .enumerate()
    {
        assert!(
            (a.penalty - b.penalty).abs() <= 1e-12 * b.penalty.abs().max(1.0),
            "sweep {it}: default-E penalty {:.17e} vs lower_energies penalty {:.17e} — the \
             driver's `RQ(ψ)/‖ψ‖²` spelling differs from `e_ref`'s",
            a.penalty,
            b.penalty
        );
        assert!(
            (a.energy - b.energy).abs() <= 1e-10 * b.energy.abs().max(1.0),
            "sweep {it}: energies diverge ({:.12e} vs {:.12e})",
            a.energy,
            b.energy
        );
    }
    assert_rule_replay(&h_given, &[e1], "tms_c6 lower_energies run");
}

#[test]
fn tms_c6_collapse_escalation_from_the_lower_state_itself() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC6_1000, 2, 0.05);
    let s = 3.0 / (fx.ev11[1] - fx.ev11[0]);
    let mut t = fx.t.clone();
    for z in t.data.iter_mut() {
        *z *= s;
    }
    let mut v = fx.v.clone();
    for z in v.data.iter_mut() {
        *z *= s;
    }
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let (e1x, e2x) = (fx.ev11[0] * s, fx.ev11[1] * s);
    println!(
        "tms_c6 collapse: scale {s:.6}, E₁ = {e1x:.12e}, E₂ = {e2x:.12e}, gap = {:.6}",
        e2x - e1x
    );
    assert!(
        (e2x - e1x) > 1.0 && (e2x - e1x) < 10.0,
        "fixture: the scaled gap {:.6} must sit between the weight floor and its first escalation",
        e2x - e1x
    );

    let mut rng = seeded(0xC6_1001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);
    let e1 = e_ref(&h, &psi1);
    assert!(
        (e1 - e1x).abs() <= 1e-8,
        "fixture: the lower state must be the scaled ground state ({e1:.12e} vs {e1x:.12e})"
    );

    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (hist, y) = state_specific_mals_with_history(&h, &psi1, std::slice::from_ref(&psi1), &o);
    let traj: Vec<(f64, f64, bool)> = hist
        .sweeps
        .iter()
        .map(|s| (s.penalty, s.overlaps[0], s.converged))
        .collect();
    println!("tms_c6 collapse: (w, |⟨ψ̂₁,x⟩|, converged) per iteration = {traj:?}");

    let mut collapses = 0usize;
    for it in 0..hist.sweeps.len() {
        let s = &hist.sweeps[it];
        let sigma = s.overlaps[0] * s.overlaps[0];
        if sigma > 0.5 {
            collapses += 1;
            assert!(
                !s.converged,
                "tms_c6: iteration {it} collapsed (Σo² = {sigma:.6}) and must not satisfy the \
                 stop test"
            );
            if it + 1 < hist.sweeps.len() {
                assert!(
                    hist.sweeps[it + 1].penalty >= 10.0 * s.penalty - 1e-12,
                    "tms_c6: after a collapse at iteration {it} (w = {}) the next weight must be \
                     ≥ 10·w_prev (got {})",
                    s.penalty,
                    hist.sweeps[it + 1].penalty
                );
            }
        }
    }
    assert!(
        collapses >= 1,
        "tms_c6: NO recorded iteration ended collapsed — the arm is vacuous and its escalation \
         assertions never ran; fix the fixture, never the assertion"
    );
    assert!(
        hist.sweeps.len() > collapses,
        "tms_c6: the run must escape the collapse before max_iter (all {collapses} iterations \
         ended on ψ₁)"
    );
    assert_rule_replay(&hist, &[e1], "tms_c6 collapse");
    assert_eq!(
        hist.sweeps[0].penalty, 1.0,
        "tms_c6: with Σo² = 1 at entry the auto rule must return exactly 1.0"
    );
    assert!(
        (hist.lambda - e2x).abs() <= 1e-8,
        "tms_c6: the escalation must still reach E₂ (got {:.12e} vs {:.12e})",
        hist.lambda,
        e2x
    );
    let ov = qntt::dot(&psi1, &y).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c6: final |⟨ψ̂₁,x⟩| = {ov:.3e}");
}

#[test]
fn tms_c7_fixed_half_gap_collapses_and_homing_does_not_rescue() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC7_0000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC7_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);
    let e1 = e_ref(&h, &psi1);
    let gap = fx.ev11[1] - fx.ev11[0];
    let w = 0.5 * gap;

    let xs = perturb_center0(&psi1, 0.3, 0xC7_0007);
    let dom = qntt::dot(&psi1, &xs).abs_s() / (psi1.norm() * xs.norm());
    assert!(
        dom > 0.9,
        "tms_c7: the start must be ψ₁-dominated (|⟨ψ̂₁,x̂₀⟩| = {dom:.6})"
    );

    for roots in [1usize, 2usize] {
        let o = StateSpecificOptions {
            mals: MalsOptions {
                eps: 1e-8,
                conv_tol: Some(1e-12),
                bond_dims: vec![0],
                noise: vec![0.0],
                max_iter: 10,
                ..Default::default()
            },
            penalty: PenaltyRule::Fixed(w),
            roots,
            lower_energies: Some(vec![e1]),
            ..Default::default()
        };
        let (hist, y) = state_specific_mals_with_history(&h, &xs, std::slice::from_ref(&psi1), &o);
        let last = hist.sweeps.last().unwrap();
        let swaps: Vec<usize> = hist.sweeps.iter().map(|s| s.swaps).collect();
        println!(
            "tms_c7 roots = {roots}: w = {w:.6} (gap {gap:.6}), final overlap {:.6}, λ = {:.12e} \
             (E₁ = {:.12e}), θ = {:.12e} (E₁ + w = {:.12e}), swaps per iteration = {swaps:?}",
            last.overlaps[0],
            hist.lambda,
            fx.ev11[0],
            last.penalized,
            e1 + w
        );

        assert!(
            last.overlaps[0] > 0.5,
            "tms_c7 roots = {roots}: an under-gap fixed weight must COLLAPSE onto ψ₁ (final \
             overlap {:.6}); if this stops holding the recorded negative has changed and the \
             design note must be revisited",
            last.overlaps[0]
        );
        assert!(
            (last.penalized - (e1 + w)).abs() <= 1e-6,
            "tms_c7 roots = {roots}: θ = {:.12e} vs the exact penalized minimum E₁ + w = {:.12e}",
            last.penalized,
            e1 + w
        );
        assert!(
            (hist.lambda - e1).abs() <= 1e-6,
            "tms_c7 roots = {roots}: the plain energy must be E₁ (got {:.12e})",
            hist.lambda
        );
        assert!(
            (last.energy - (last.penalized - w * last.overlaps[0] * last.overlaps[0])).abs()
                <= 1e-8,
            "tms_c7 roots = {roots}: λ = θ − w·Σ|⟨p,u⟩|² violated"
        );
        assert!(y.check(), "tms_c7 roots = {roots}: returned state");
    }
}

#[test]
fn tms_c8_auto_from_psi1_plus_noise_reaches_the_second_root() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC8_0000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC8_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);
    let e1 = e_ref(&h, &psi1);
    let gap = fx.ev11[1] - fx.ev11[0];

    let xs = perturb_center0(&psi1, 0.1, 0xC8_0003);
    let rq0 = rayleigh_quotient(&h, &xs).re_s() / (xs.norm() * xs.norm());
    println!(
        "tms_c8(a): RQ(x₀)/‖x₀‖² − E₁ = {:.6e}, gap = {gap:.6}, 10·excess = {:.6}",
        rq0 - e1,
        10.0 * (rq0 - e1)
    );

    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (hist, y) = state_specific_mals_with_history(&h, &xs, std::slice::from_ref(&psi1), &o);
    let ws: Vec<f64> = hist.sweeps.iter().map(|s| s.penalty).collect();
    println!(
        "tms_c8(a): weight trajectory = {ws:?}, λ = {:.12e}",
        hist.lambda
    );

    assert!(
        (hist.lambda - fx.ev11[1]).abs() <= 1e-10,
        "tms_c8(a): λ = {:.12e} vs E₂ = {:.12e} (report the measured error, never loosen this \
         silently)",
        hist.lambda,
        fx.ev11[1]
    );
    let ov = qntt::dot(&psi1, &y).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c8(a): |⟨ψ̂₁,x⟩| = {ov:.3e}");
    assert!(
        hist.sweeps[0].penalty > gap,
        "tms_c8(a): the first auto weight {} must already exceed the gap {gap:.6}",
        hist.sweeps[0].penalty
    );
    assert!(
        hist.sweeps.last().unwrap().converged,
        "tms_c8(a): converged"
    );
    for (it, s) in hist.sweeps.iter().enumerate() {
        assert_eq!(s.skipped, 0, "tms_c8(a): sweep {it} skipped at full rank");
    }
    assert_rule_replay(&hist, &[e1], "tms_c8(a)");
}

#[test]
fn tms_c8_auto_from_exactly_psi1_escalates_and_reaches_the_second_root() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC8_1000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC8_1001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);
    let e1 = e_ref(&h, &psi1);

    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (hist, y) = state_specific_mals_with_history(&h, &psi1, std::slice::from_ref(&psi1), &o);
    let traj: Vec<(f64, f64)> = hist
        .sweeps
        .iter()
        .map(|s| (s.penalty, s.overlaps[0]))
        .collect();
    println!(
        "tms_c8(b): (w, overlap) trajectory = {traj:?}, λ = {:.12e}",
        hist.lambda
    );

    assert_eq!(
        hist.sweeps[0].penalty, 1.0,
        "tms_c8(b): with Σo² = 1 at entry λ⊥ = λₓ = E₁ and the rule returns exactly 1.0"
    );
    assert!(
        (hist.lambda - fx.ev11[1]).abs() <= 1e-10,
        "tms_c8(b): λ = {:.12e} vs E₂ = {:.12e}",
        hist.lambda,
        fx.ev11[1]
    );
    let ov = qntt::dot(&psi1, &y).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c8(b): |⟨ψ̂₁,x⟩| = {ov:.3e}");
    assert!(
        hist.sweeps.last().unwrap().converged,
        "tms_c8(b): converged"
    );
    assert_rule_replay(&hist, &[e1], "tms_c8(b)");
}

#[test]
fn tms_c8_sub_unit_lower_state_is_accepted_and_normalized() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC8_2000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC8_2001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);

    let mut chosen: Option<(usize, TtVector<f64>)> = None;
    for cap in [8usize, 6, 5, 4, 3, 2] {
        let y = psi1.round(0.0, qntt::Pooled(cap), RoundMode::Rel);
        let n = y.norm();
        println!("tms_c8(c): cap {cap} gives ‖ψᵗ‖ = {n:.6}");
        if (0.9..=0.9999).contains(&n) {
            chosen = Some((cap, y));
            break;
        }
    }
    let (cap, psit) = chosen.expect(
        "tms_c8(c): no rank cap in [8,6,5,4,3,2] produced a lower state of norm in [0.9, 0.9999] \
         — a FIXTURE finding; widen the cap list, never the window",
    );
    let nt = psit.norm();
    let et = e_ref(&h, &psit);
    println!("tms_c8(c): using cap {cap}, ‖ψᵗ‖ = {nt:.12e}, E(ψᵗ) = {et:.12e}");

    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![et]),
        ..Default::default()
    };
    let (hist, x) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psit), &o);
    let last = hist.sweeps.last().unwrap();
    assert!(last.converged, "tms_c8(c): the run must converge");
    assert_rule_replay(&hist, &[et], "tms_c8(c)");

    let measured = qntt::dot(&psit, &x).abs_s() / nt;
    assert!(
        (last.overlaps[0] - measured).abs() <= 1e-14,
        "tms_c8(c): the recorded overlap must carry the 1/‖ψ‖ ({:.17e} vs {measured:.17e})",
        last.overlaps[0]
    );
    let chat = dense_unit(&psit, &fx.dets11, "tms_c8(c) ψᵗ");
    let hd = dense_hamiltonian_matrix(d, &fx.t, &fx.v, &fx.dets11);
    let lmin = dense_eigen(&penalized_dense(
        &hd,
        std::slice::from_ref(&chat),
        last.penalty,
    ))
    .0[0];
    assert!(
        (last.penalized - lmin).abs() <= 1e-8,
        "tms_c8(c): θ = {:.12e} vs the dense λ_min(H + w·ĉᵗĉᵗᵀ) = {lmin:.12e}",
        last.penalized
    );
    assert!(
        (last.energy - (last.penalized - last.penalty * measured * measured)).abs() <= 1e-12,
        "tms_c8(c): λ = θ − w·Σo² violated ({:.17e} vs {:.17e})",
        last.energy,
        last.penalized - last.penalty * measured * measured
    );
    println!(
        "tms_c8(c): λ − E₂ = {:+.6e} (second-order in the lower state's error, sign-indefinite by \
         decision 15 — recorded, not asserted)",
        hist.lambda - fx.ev11[1]
    );
}

#[test]
fn tms_c8_norm_two_lower_state_matches_the_unit_one() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC8_3000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xC8_3001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = ground(&h, &x0);
    let psin = &psi1 * 2.0;
    assert!(
        (psin.norm() - 2.0 * psi1.norm()).abs() <= 1e-12,
        "fixture: the scaled copy must have twice the norm"
    );
    let e1 = e_ref(&h, &psi1);
    let en = e_ref(&h, &psin);
    assert!(
        (en - e1).abs() <= 1e-10 * e1.abs(),
        "tms_c8(d): RQ/‖ψ‖² must be scale-invariant ({en:.12e} vs {e1:.12e})"
    );

    let schedule = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 8,
        ..Default::default()
    };
    let (ha, xa) = state_specific_mals_with_history(
        &h,
        &x0,
        std::slice::from_ref(&psi1),
        &ss_opts(schedule.clone()),
    );
    let (hn, xn) =
        state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psin), &ss_opts(schedule));

    assert_eq!(
        ha.sweeps.len(),
        hn.sweeps.len(),
        "tms_c8(d): iteration count"
    );
    assert!(
        (ha.lambda - hn.lambda).abs() <= 1e-12 * ha.lambda.abs(),
        "tms_c8(d): λ {:.17e} vs {:.17e}",
        ha.lambda,
        hn.lambda
    );
    for (it, (a, b)) in ha.sweeps.iter().zip(hn.sweeps.iter()).enumerate() {
        for (what, x, y) in [
            ("penalty", a.penalty, b.penalty),
            ("energy", a.energy, b.energy),
            ("overlaps[0]", a.overlaps[0], b.overlaps[0]),
        ] {
            assert!(
                (x - y).abs() <= 1e-12 * x.abs().max(1.0),
                "tms_c8(d): sweep {it} {what}: {x:.17e} vs {y:.17e}"
            );
        }
    }
    let last = hn.sweeps.last().unwrap();
    let measured = qntt::dot(&psin, &xn).abs_s() / psin.norm();
    assert!(
        (last.overlaps[0] - measured).abs() <= 1e-14,
        "tms_c8(d): the recorded overlap must be normalized by ‖ψ‖ = 2"
    );
    let ov = qntt::dot(&psi1, &xn).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c8(d): |⟨ψ̂₁,x⟩| = {ov:.3e}");
    let _ = xa;
}

#[test]
#[should_panic(expected = "norm")]
fn tms_c8_zero_norm_lower_state_is_refused() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xC8_4001);
    let x0: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    let z = &x0 * 0.0;
    let _ = state_specific_mals(&h, &x0, std::slice::from_ref(&z), &ss_opts(opts_exact()));
}

#[test]
#[should_panic]
fn tms_c8_lower_energies_length_mismatch_is_refused() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xC8_5001);
    let x0: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    let psi: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![-1.0, -2.0]),
        ..Default::default()
    };
    let _ = state_specific_mals(&h, &x0, std::slice::from_ref(&psi), &o);
}

#[test]
fn tms_c9_roots_two_follows_the_start_label() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xC9_0000, 3, 0.05);
    assert_exclusion_gap(&fx.ev11, 3, 0.05, "tms_c9");
    let h = fx.abelian(space);
    let mut rng = seeded(0xC9_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let (_, psi1) = ground(&h, &x0);
    let e1 = e_ref(&h, &psi1);
    let o2 = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (h2, psi2) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o2);
    assert!((h2.lambda - fx.ev11[1]).abs() <= 1e-8, "tms_c9: rung 2");
    let lower12 = vec![psi1.clone(), psi2.clone()];
    let e12: Vec<f64> = lower12.iter().map(|p| e_ref(&h, p)).collect();
    let o3 = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(e12.clone()),
        ..Default::default()
    };
    let (h3, psi3) = state_specific_mals_with_history(&h, &x0, &lower12, &o3);
    assert!((h3.lambda - fx.ev11[2]).abs() <= 1e-8, "tms_c9: rung 3");

    let x3 = perturb_center0(&psi3, 0.1, 0xC9_0007);
    let dom = qntt::dot(&psi3, &x3).abs_s() / (psi3.norm() * x3.norm());
    assert!(
        dom > 0.9,
        "tms_c9: the start must be ψ₃-dominated ({dom:.6})"
    );

    let (hi, _) = state_specific_mals_with_history(&h, &x3, std::slice::from_ref(&psi1), &o2);
    assert!(
        (hi.lambda - fx.ev11[1]).abs() <= 1e-8,
        "tms_c9(i): roots = 1 must return E₂ (got {:.12e})",
        hi.lambda
    );
    for (it, s) in hi.sweeps.iter().enumerate() {
        assert_eq!(s.swaps, 0, "tms_c9(i): sweep {it} swaps at roots = 1");
    }

    let o = StateSpecificOptions {
        mals: opts_exact(),
        roots: 2,
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (hii, y) = state_specific_mals_with_history(&h, &x3, std::slice::from_ref(&psi1), &o);
    let swaps: Vec<usize> = hii.sweeps.iter().map(|s| s.swaps).collect();
    println!(
        "tms_c9(ii): λ = {:.12e} (E₃ = {:.12e}), swaps per iteration = {swaps:?}",
        hii.lambda, fx.ev11[2]
    );
    assert!(
        (hii.lambda - fx.ev11[2]).abs() <= 1e-8,
        "tms_c9(ii): roots = 2 from a ψ₃ start must keep the continuation (got {:.12e})",
        hii.lambda
    );
    assert!(
        hii.sweeps.last().unwrap().swaps >= 1,
        "tms_c9(ii): the final iteration must record at least one swap — that is the design's \
         tell that homing, not the energy order, chose the root"
    );
    for (name, p, want_small) in [
        ("ψ₁", &psi1, true),
        ("ψ₂", &psi2, true),
        ("ψ₃", &psi3, false),
    ] {
        let ov = qntt::dot(p, &y).abs_s() / p.norm();
        if want_small {
            assert!(ov <= 1e-6, "tms_c9(ii): |⟨{name},x⟩| = {ov:.3e}");
        } else {
            assert!(
                (ov - 1.0).abs() <= 1e-6,
                "tms_c9(ii): |⟨{name},x⟩| = {ov:.6}"
            );
        }
    }
    assert_eq!(
        hii.sweeps.last().unwrap().overlaps.len(),
        1,
        "tms_c9(ii): one recorded overlap per LOWER state (roots is not a lower state)"
    );

    let (hiii, yiii) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);
    let swaps: Vec<usize> = hiii.sweeps.iter().map(|s| s.swaps).collect();
    let best = (1..4usize)
        .map(|j| (hiii.lambda - fx.ev11[j]).abs())
        .fold(f64::INFINITY, f64::min);
    println!(
        "tms_c9(iii): λ = {:.12e} (E₂ = {:.12e}, E₃ = {:.12e}), closest dense level distance = \
         {best:.3e}, swaps = {swaps:?} (WHICH level a neutral start continues is a coin flip and \
         is not asserted)",
        hiii.lambda, fx.ev11[1], fx.ev11[2]
    );
    assert!(
        best <= 1e-8,
        "tms_c9(iii): λ = {:.12e} is not one of E₂..E₄ (closest {best:.3e})",
        hiii.lambda
    );
    let ov = qntt::dot(&psi1, &yiii).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c9(iii): |⟨ψ̂₁,x⟩| = {ov:.3e}");
    let n = hiii.sweeps.len();
    assert!(
        n >= 2,
        "tms_c9(iii): need two iterations to see a settled label"
    );
    assert_eq!(
        hiii.sweeps[n - 1].swaps,
        hiii.sweeps[n - 2].swaps,
        "tms_c9(iii): the label must be settled by the final iteration (swap counts {swaps:?})"
    );

    let xd = perturb_center0(&psi1, 0.3, 0xC9_0009);
    let (hiv, yd) = state_specific_mals_with_history(&h, &xd, std::slice::from_ref(&psi1), &o);
    let best = (1..4usize)
        .map(|j| (hiv.lambda - fx.ev11[j]).abs())
        .fold(f64::INFINITY, f64::min);
    println!(
        "tms_c9(iv): λ = {:.12e}, closest dense level distance = {best:.3e}, final swaps = {}",
        hiv.lambda,
        hiv.sweeps.last().unwrap().swaps
    );
    assert!(
        best <= 1e-8,
        "tms_c9(iv): λ = {:.12e} is not one of E₂..E₄ (closest {best:.3e})",
        hiv.lambda
    );
    let ov = qntt::dot(&psi1, &yd).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c9(iv): |⟨ψ̂₁,x⟩| = {ov:.3e}");
    let rq = rayleigh_quotient(&h, &yd).re_s() / (yd.norm() * yd.norm());
    assert!((rq - hiv.lambda).abs() <= 1e-8, "tms_c9(iv): RQ vs λ");
}

#[test]
fn tms_c10_threaded_matches_serial_bitwise() {
    let d = 6usize;
    let space = QnSpace::new(d, 2, 2);
    let fx = gapped_tv(0xCA_0000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xCA_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (_, psi1) = mals(
        &h,
        &x0,
        &MalsOptions {
            eps: 1e-8,
            bond_dims: vec![0],
            noise: vec![0.0],
            max_iter: 6,
            ..Default::default()
        },
    );
    let e1 = e_ref(&h, &psi1);

    for mode in [SsMode::Penalty, SsMode::Project] {
        let o = StateSpecificOptions {
            mals: MalsOptions {
                eps: 1e-8,
                bond_dims: vec![3, 4, 0],
                noise: vec![0.0],
                max_iter: 4,
                ..Default::default()
            },
            mode,
            lower_energies: Some(vec![e1]),
            ..Default::default()
        };
        let run = |nt: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap()
                .install(|| {
                    state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o)
                })
        };
        let (h1, y1) = run(1);
        let (h8, y8) = run(8);

        let widest = (0..d - 1)
            .map(|k| contract(y1.core(k), y1.core(k + 1)).mid_qn.len())
            .max()
            .unwrap_or(0);
        println!("tms_c10 {mode:?}: widest visited bond carries {widest} mid sectors");
        assert!(
            widest >= 4,
            "VACUOUS — the widest bond carries {widest} mid sectors, below the kernels' parallel \
             fork threshold (4), so nothing here runs threaded"
        );

        let mut worst = 0.0f64;
        for (a, b) in h1.sweeps.iter().zip(h8.sweeps.iter()) {
            worst = worst.max((a.energy - b.energy).abs());
            worst = worst.max((a.penalized - b.penalized).abs());
            worst = worst.max((a.discarded_weight - b.discarded_weight).abs());
        }
        println!(
            "tms_c10 {mode:?}: max |Δ| between the 1- and 8-thread trajectories = {worst:.3e}"
        );

        assert_eq!(h1.sweeps.len(), h8.sweeps.len(), "{mode:?}: sweep count");
        assert_f64_bits(h1.lambda, h8.lambda, &format!("{mode:?}: final λ"));
        for (it, (a, b)) in h1.sweeps.iter().zip(h8.sweeps.iter()).enumerate() {
            let tag = |w: &str| format!("{mode:?}: sweep {it} {w}");
            assert_f64_bits(a.energy, b.energy, &tag("energy"));
            assert_f64_bits(a.penalized, b.penalized, &tag("penalized"));
            assert_f64_bits(
                a.discarded_weight,
                b.discarded_weight,
                &tag("discarded_weight"),
            );
            assert_f64_bits(a.penalty, b.penalty, &tag("penalty"));
            assert_f64_bits(a.noise, b.noise, &tag("noise"));
            assert_eq!(
                a.overlaps.len(),
                b.overlaps.len(),
                "{}",
                tag("overlaps length")
            );
            for j in 0..a.overlaps.len() {
                assert_f64_bits(
                    a.overlaps[j],
                    b.overlaps[j],
                    &tag(&format!("overlaps[{j}]")),
                );
            }
            assert_eq!(a.max_bond_dim, b.max_bond_dim, "{}", tag("max_bond_dim"));
            assert_eq!(a.bond_dim, b.bond_dim, "{}", tag("bond_dim"));
            assert_eq!(a.swaps, b.swaps, "{}", tag("swaps"));
            assert_eq!(a.numops, b.numops, "{}", tag("numops"));
            assert_eq!(a.skipped, b.skipped, "{}", tag("skipped"));
            assert_eq!(a.converged, b.converged, "{}", tag("converged"));
        }
        coeffs_bits_eq_tt(&y1, &y8, &format!("{mode:?}: returned state"));
    }
}

#[test]
fn tms_c11_max_iter_exhaustion_marks_every_sweep_unconverged() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = narrow_gap_tv(0xCB_0000);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &fx.t, &fx.v, &theta);
    let hc = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
    let mut rng = seeded(0xCB_0001);
    let x0c: TtVector<C64> = randomized_state(space, 5, 10, &mut rng);

    let (lam1, psi1) = ground(&hc, &x0c);
    assert!(
        (lam1 - fx.ev11[0]).abs() <= 1e-8,
        "tms_c11(a): the ground rung must converge at the 0.05 gap (got {lam1:.12e})"
    );
    let o = StateSpecificOptions {
        mals: MalsOptions {
            eps: 1e-10,
            conv_tol: Some(1e-12),
            bond_dims: vec![0],
            noise: vec![0.0],
            max_iter: 2,
            ..Default::default()
        },
        lower_energies: Some(vec![e_ref(&hc, &psi1)]),
        ..Default::default()
    };
    let (hist, _) = state_specific_mals_with_history(&hc, &x0c, std::slice::from_ref(&psi1), &o);
    assert_eq!(hist.sweeps.len(), 2, "tms_c11(a): two iterations recorded");
    for (it, s) in hist.sweeps.iter().enumerate() {
        assert!(
            !s.converged,
            "tms_c11(a): sweep {it} claims convergence after {} iteration(s) at conv_tol 1e-12",
            it + 1
        );
    }
    println!(
        "tms_c11(a): |λ − E₂| after two sweeps = {:.3e} (printed, not asserted)",
        (hist.lambda - fx.ev11[1]).abs()
    );

    let h = fx.abelian(space);
    let mut rng = seeded(0xCB_0002);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let opts = MalsOptions {
        eps: 1e-10,
        conv_tol: Some(1e-12),
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 2,
        ..Default::default()
    };
    let (hd, _) = state_specific_mals_with_history(&h, &x0, &[], &ss_opts(opts));
    assert_eq!(hd.sweeps.len(), 2, "tms_c11(a) delegate: two iterations");
    for (it, s) in hd.sweeps.iter().enumerate() {
        assert!(!s.converged, "tms_c11(a) delegate: sweep {it} converged");
    }
}

#[test]
fn tms_c11_multiplet_fixture_skips_one_dimensional_bonds() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = multiplet_tv(d);
    let (_, ev11, _) = dense_sector(d, &t, &v, 1, 1);
    let (_, ev20, _) = dense_sector(d, &t, &v, 2, 0);
    println!("tms_c11(b): (1,1) levels[..6] = {:?}", &ev11[..6]);
    println!("tms_c11(b): (2,0) levels[..4] = {:?}", &ev20[..4]);
    for i in 0..3 {
        assert!(
            (ev11[i] + 1.0).abs() <= 1e-12,
            "fixture: (1,1) level {i} = {:.12e}, expected the −1 triplet",
            ev11[i]
        );
        assert!(
            (ev20[i] + 1.0).abs() <= 1e-12,
            "fixture: (2,0) level {i} = {:.12e} — the −1 space must be a TRIPLET (it has Sz = 1 \
             partners), so this fixture is never an S = 0 oracle",
            ev20[i]
        );
    }
    assert!(
        (ev11[3] - 1.0).abs() <= 1e-12,
        "fixture: the first level above the triplet must be +1 (got {:.12e})",
        ev11[3]
    );

    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let z01 = state_from_sites::<f64>(d, &[0], &[1]);
    let z10 = state_from_sites::<f64>(d, &[1], &[0]);
    let mut psi1 = &z01 + &z10;
    psi1.scale_mut(1.0 / std::f64::consts::SQRT_2);
    let rq1 = rayleigh_quotient(&h, &psi1).re_s() / (psi1.norm() * psi1.norm());
    assert!(
        (rq1 + 1.0).abs() <= 1e-12,
        "fixture: the constructed lower state must be an EXACT ground-multiplet member (RQ = \
         {rq1:.12e})"
    );

    let x0 = state_from_sites::<f64>(d, &[0], &[1]);
    let dims: Vec<usize> = (0..d - 1)
        .map(|k| contract(x0.core(k), x0.core(k + 1)).num_coeffs())
        .collect();
    println!("tms_c11(b): two-site local dimensions of the start = {dims:?}");
    assert!(
        dims.contains(&1),
        "fixture: no 1-dimensional local space in the start — the SKIP path is not reached"
    );

    let schedule = MalsOptions {
        eps: 1e-10,
        conv_tol: Some(1e-12),
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 40,
        ..Default::default()
    };
    for (label, mode, roots) in [
        ("Project roots = 1", SsMode::Project, 1usize),
        ("Penalty roots = 2", SsMode::Penalty, 2usize),
    ] {
        let o = StateSpecificOptions {
            mals: schedule.clone(),
            mode,
            penalty: PenaltyRule::Auto,
            roots,
            lower_energies: Some(vec![rq1]),
            ..Default::default()
        };
        let (hist, y) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);
        let skipped: usize = hist.sweeps.iter().map(|s| s.skipped).sum();
        let swaps: Vec<usize> = hist.sweeps.iter().map(|s| s.swaps).collect();
        let res = global_eigen_residual_abelian(&h, &y, hist.lambda);
        println!(
            "tms_c11(b) {label}: λ = {:.12e}, Σskipped = {skipped}, swaps = {swaps:?}, \
             residual = {res:.3e} (the reached level is NOT asserted — recorded local-minimum \
             caveat)",
            hist.lambda
        );
        if roots > 1 {
            assert!(
                skipped >= 1,
                "{label}: a bond whose local space cannot host {roots} vectors must be SKIPPED \
                 (Σskipped = 0)"
            );
        }
        assert!(
            hist.lambda >= -1.0 - 1e-8,
            "{label}: the plain energy {:.12e} is below the exact ground level −1 — a Rayleigh \
             quotient cannot be",
            hist.lambda
        );
        let rq = rayleigh_quotient(&h, &y).re_s() / (y.norm() * y.norm());
        assert!(
            (rq - hist.lambda).abs() <= 1e-8,
            "{label}: RQ {rq:.12e} vs recorded λ {:.12e}",
            hist.lambda
        );
        if matches!(mode, SsMode::Project) {
            let ov = qntt::dot(&psi1, &y).abs_s() / (psi1.norm() * y.norm());
            assert!(
                ov <= 1e-6,
                "{label}: the projector must leave |⟨ψ̂₁,x̂⟩| = {ov:.3e} at zero"
            );
        }
    }
}

fn multiplet_tv_mirrored(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..3 {
        t.set(&[i, i], 3.0);
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        v.set(&[i, i, i, i], 2.0);
    }
    for i in 3..6 {
        for j in 3..6 {
            if i != j {
                v.set(&[i, j, j, i], 1.0);
            }
        }
    }
    (t, v)
}

#[test]
fn tms_c11_project_skips_a_blocked_one_dimensional_bond() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = multiplet_tv_mirrored(d);
    let (_, ev11, _) = dense_sector(d, &t, &v, 1, 1);
    println!("tms_c11(c): (1,1) levels[..6] = {:?}", &ev11[..6]);
    for (i, e) in ev11.iter().take(3).enumerate() {
        assert!(
            (e + 1.0).abs() <= 1e-12,
            "fixture: (1,1) level {i} = {e:.12e}, expected the −1 triplet"
        );
    }

    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let z34 = state_from_sites::<f64>(d, &[3], &[4]);
    let z43 = state_from_sites::<f64>(d, &[4], &[3]);
    let mut psi1 = &z34 + &z43;
    psi1.scale_mut(1.0 / std::f64::consts::SQRT_2);
    let rq1 = rayleigh_quotient(&h, &psi1).re_s() / (psi1.norm() * psi1.norm());
    assert!(
        (rq1 + 1.0).abs() <= 1e-12,
        "fixture: the lower state must be an EXACT ground-multiplet member (RQ = {rq1:.12e})"
    );

    let x0 = z34.clone();
    let dims: Vec<usize> = (0..d - 1)
        .map(|k| contract(x0.core(k), x0.core(k + 1)).num_coeffs())
        .collect();
    println!("tms_c11(c): two-site local dimensions of the start = {dims:?}");
    assert_eq!(
        dims[0], 1,
        "fixture: pair (0,1) must be 1-dimensional — the blocked-skip path is not reached"
    );
    let ov0 = qntt::dot(&psi1, &x0).abs_s() / (psi1.norm() * x0.norm());
    assert!(
        (ov0 - 1.0 / std::f64::consts::SQRT_2).abs() <= 1e-12,
        "fixture: the start must carry the 1/√2 overlap the leading bonds see (got {ov0:.12e})"
    );

    let schedule = MalsOptions {
        eps: 1e-10,
        conv_tol: Some(1e-12),
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 40,
        ..Default::default()
    };
    for roots in [1usize, 2usize] {
        let label = format!("Project roots = {roots}");
        let o = StateSpecificOptions {
            mals: schedule.clone(),
            mode: SsMode::Project,
            roots,
            lower_energies: Some(vec![rq1]),
            ..Default::default()
        };
        let (hist, y) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);
        let skipped: usize = hist.sweeps.iter().map(|s| s.skipped).sum();
        let first = hist.sweeps[0].skipped;
        println!(
            "tms_c11(c) {label}: λ = {:.12e}, Σskipped = {skipped} (iteration 0: {first})",
            hist.lambda
        );
        assert!(
            first >= 1,
            "{label}: iteration 0 must SKIP a bond whose local space cannot host roots + the \
             projected directions (skipped = 0)"
        );
        assert!(
            hist.lambda >= -1.0 - 1e-8,
            "{label}: the plain energy {:.12e} is below the exact ground level −1",
            hist.lambda
        );
        let rq = rayleigh_quotient(&h, &y).re_s() / (y.norm() * y.norm());
        assert!(
            (rq - hist.lambda).abs() <= 1e-8,
            "{label}: RQ {rq:.12e} vs recorded λ {:.12e}",
            hist.lambda
        );
        let ov = qntt::dot(&psi1, &y).abs_s() / (psi1.norm() * y.norm());
        assert!(
            ov <= 1e-6,
            "{label}: the projector must leave |⟨ψ̂₁,x̂⟩| = {ov:.3e} at zero"
        );
    }
}

#[test]
fn tms_c12_from_states_round_trip() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xCC_0000, 3, 0.05);
    let h = fx.abelian(space);

    let xs: Vec<TtVector<f64>> = (0..3u64)
        .map(|i| {
            let mut rng = seeded(0xCC_1000 + i);
            let y: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
            &y * (0.5 + i as f64)
        })
        .collect();
    let before = xs.clone();

    let m = MultiTt::from_states(&xs, None);
    m.check();
    assert_eq!(m.nstates(), 3, "three states");
    assert_eq!(m.core_position, 0, "the container sits at core position 0");
    assert_eq!(m.base.ortho, Some(0), "base.ortho");
    for (i, w) in m.weights.iter().enumerate() {
        assert!(
            (w - 1.0 / 3.0).abs() <= 1e-15,
            "weight {i} defaults to 1/n (got {w})"
        );
    }
    for b in 0..=d {
        let total: usize = xs.iter().map(|x| x.ranks[b].total()).sum();
        assert!(
            m.base.ranks[b].total() <= total,
            "bond {b}: the sum's rank {} exceeds Σᵢ rankᵢ = {total}",
            m.base.ranks[b].total()
        );
    }

    let dev = abelian_right_canonicity_dev(&m.base, 1..d);
    println!("tms_c12: base right-canonicity deviation over sites 1..{d} = {dev:.3e}");
    assert!(
        dev <= 1e-12,
        "tms_c12: the container's shared frames are NOT right-canonical (max |Σ_f Q_fQ_fᴴ − I| = \
         {dev:.3e}); the state-average local metric then carries a null space at the next bond. \
         The fix is the orthogonalization used in `from_states`, never this assertion"
    );

    let ys = m.states();
    assert_eq!(ys.len(), 3, "one state per summand");
    for (i, (y, x)) in ys.iter().zip(xs.iter()).enumerate() {
        let nx2 = qntt::dot(x, x).re_s();
        let diff = (y - x).norm_lr(false);
        println!(
            "tms_c12: state {i} ‖y − x‖ = {diff:.3e} (‖x‖ = {:.6})",
            nx2.sqrt()
        );
        assert!(
            diff <= 1e-12 * nx2.sqrt(),
            "state {i}: ‖y − x‖ = {diff:.3e} above 1e-12·‖x‖"
        );
        assert!(
            (qntt::dot(y, x).re_s() - nx2).abs() <= 1e-12 * nx2,
            "state {i}: ⟨y,x⟩ ≠ ⟨x,x⟩"
        );
        assert!(
            (qntt::dot(y, y).re_s() - nx2).abs() <= 1e-12 * nx2,
            "state {i}: ‖y‖² ≠ ‖x‖²"
        );
        assert!(
            (rayleigh_quotient(&h, y).re_s() - rayleigh_quotient(&h, x).re_s()).abs()
                <= 1e-12 * nx2,
            "state {i}: the Rayleigh quotient moved"
        );
    }
    for (i, (a, b)) in xs.iter().zip(before.iter()).enumerate() {
        coeffs_bits_eq_tt(a, b, &format!("input {i} must not be mutated"));
    }

    let mw = MultiTt::from_states(&xs, Some(vec![0.5, 0.3, 0.2]));
    mw.check();
    assert_eq!(mw.weights, vec![0.5, 0.3, 0.2], "explicit weights");

    let mut m2 = MultiTt::from_states(&xs, None);
    let sa = StateAverageOptions {
        mals: MalsOptions {
            eps: 1e-8,
            bond_dims: vec![0],
            noise: vec![0.0],
            max_iter: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let hist = state_average_mals_mut(&h, &mut m2, &sa);
    assert_eq!(hist.sweeps.len(), 1, "one state-average iteration ran");
    m2.check();
}

#[test]
fn tms_c12_from_states_seeds_a_state_average_from_an_ss_ladder() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xCC_2000, 3, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xCC_2001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let (l1, psi1) = ground(&h, &x0);
    let o2 = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![e_ref(&h, &psi1)]),
        ..Default::default()
    };
    let (h2, psi2) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o2);
    let lower = vec![psi1.clone(), psi2.clone()];
    let o3 = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(lower.iter().map(|p| e_ref(&h, p)).collect()),
        ..Default::default()
    };
    let (h3, psi3) = state_specific_mals_with_history(&h, &x0, &lower, &o3);
    println!(
        "tms_c12 ladder: λ = [{l1:.12e}, {:.12e}, {:.12e}] vs E₀..E₂ = {:?}",
        h2.lambda,
        h3.lambda,
        &fx.ev11[..3]
    );

    let xs = vec![psi1, psi2, psi3];
    let mut m = MultiTt::from_states(&xs, None);
    m.check();
    for (i, y) in m.states().iter().enumerate() {
        let rq = rayleigh_quotient(&h, y).re_s() / (y.norm() * y.norm());
        assert!(
            (rq - fx.ev11[i]).abs() <= 1e-8,
            "ladder state {i}: RQ {rq:.12e} vs E{i} = {:.12e}",
            fx.ev11[i]
        );
    }

    let sa = StateAverageOptions {
        mals: MalsOptions {
            eps: 1e-9,
            conv_tol: Some(1e-12),
            bond_dims: vec![0],
            noise: vec![0.0],
            max_iter: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let hist = state_average_mals_mut(&h, &mut m, &sa);
    let ys = m.states();
    for i in 0..3 {
        assert!(
            (hist.lambdas[i] - fx.ev11[i]).abs() <= 1e-8,
            "state average root {i}: λ = {:.12e} vs E{i} = {:.12e}",
            hist.lambdas[i],
            fx.ev11[i]
        );
    }
    let gram = gram_dev_tt(&ys);
    println!(
        "tms_c12 ladder → SA: λ = {:?}, Gram = {gram:.3e}",
        hist.lambdas
    );
    assert!(gram <= 1e-8, "Gram deviation {gram:.3e}");
    assert!(
        hist.sweeps.last().unwrap().overlap_dev <= 1e-6,
        "overlap_dev {:.3e}",
        hist.sweeps.last().unwrap().overlap_dev
    );
}

#[test]
fn tms_c12_from_states_round_trip_c64() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let xs: Vec<TtVector<C64>> = (0..2u64)
        .map(|i| {
            let mut rng = seeded(0xCC_3000 + i);
            randomized_state::<C64, _>(space, 3, 4, &mut rng)
        })
        .collect();
    let m = MultiTt::from_states(&xs, None);
    m.check();
    for (i, (y, x)) in m.states().iter().zip(xs.iter()).enumerate() {
        let nx = qntt::dot(x, x).re_s().sqrt();
        let diff = (y - x).norm_lr(false);
        assert!(diff <= 1e-12 * nx, "C64 state {i}: ‖y − x‖ = {diff:.3e}");
    }
}

#[test]
#[should_panic(expected = "from_states")]
fn tms_c12_from_states_refuses_an_empty_list() {
    let _ = MultiTt::<f64>::from_states(&[], None);
}

#[test]
#[should_panic(expected = "space")]
fn tms_c12_from_states_refuses_mixed_spaces() {
    let d = 6usize;
    let mut r1 = seeded(0xCC_4001);
    let mut r2 = seeded(0xCC_4002);
    let a: TtVector<f64> = randomized_state(QnSpace::new(d, 1, 1), 2, 3, &mut r1);
    let b: TtVector<f64> = randomized_state(QnSpace::new(d, 2, 0), 2, 3, &mut r2);
    let _ = MultiTt::from_states(&[a, b], None);
}

#[test]
#[should_panic(expected = "the two-site local space at")]
fn tms_c12_from_states_refuses_a_local_space_below_n() {
    let d = 6usize;
    let z = state_from_sites::<f64>(d, &[0], &[1]);
    let _ = MultiTt::from_states(&[z.clone(), z], None);
}

#[test]
fn tms_b7k_host_n_states_raises_to_the_next_largest_schmidt_value() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 4, 0xB7_0001);
    let mut rng = seeded(0xB7_0002);
    let cps = unit_blocks_abelian(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    let n = 3usize;

    for (label, r) in [("forced raise", 1usize), ("no raise needed", n)] {
        let (shared, _s, cs, _infos, pooled, _lw) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), true, false);

        let mut entries: Vec<(f64, qntt::qn::Sector)> = Vec::new();
        for (m, evals, _) in pooled_rho_abelian(&cps, &w, true) {
            for l in evals {
                entries.push((l, m));
            }
        }
        entries.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .expect("no NaN in the pooled spectrum")
        });
        let width = |m: qntt::qn::Sector| cps[0].block(m).map_or(0, |b| b.ncols());
        let full = entries.len();
        let mut rmin = r;
        while rmin < full {
            let dim: usize = entries[..rmin].iter().map(|&(_, m)| width(m)).sum();
            if dim >= n {
                break;
            }
            rmin += 1;
        }
        let kept = kept_ranks_local(&shared, true);
        println!(
            "tms_b7k {label}: requested r = {r}, oracle r_min = {rmin} (full {full}), kept total \
             = {}",
            kept.total()
        );
        assert_eq!(
            kept.total(),
            rmin,
            "{label}: the raise must stop at the first cap whose centre hosts {n} states"
        );
        assert_eq!(pooled.kept, rmin, "{label}: pooled TruncInfo.kept");
        for &m in cps[0].mid_qn.iter() {
            let want = entries[..rmin].iter().filter(|&&(_, s)| s == m).count();
            assert_eq!(kept[m], want, "{label}: kept rank in sector {m:?}");
        }
        for (i, c) in cs.iter().enumerate() {
            assert!(
                c.num_coeffs() >= n,
                "{label}: centre {i} holds {} coefficient(s), below {n}",
                c.num_coeffs()
            );
        }
    }
}

#[test]
#[should_panic(expected = "cannot host")]
fn tms_b7k_host_n_states_refuses_when_full_rank_cannot_host_n() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 4, 0xB7_0011);
    let full: usize = cp0
        .mid_qn
        .iter()
        .filter_map(|&m| cp0.block(m))
        .map(|b| b.nrows())
        .sum();
    let n = full + 1;
    let mut rng = seeded(0xB7_0012);
    let cps = unit_blocks_abelian(&cp0, n, &mut rng);
    let w = vec![1.0 / n as f64; n];
    let _ = factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(1), true, false);
}

fn kept_ranks_local<T: Scalar>(shared: &SparseCore<T>, forward: bool) -> qntt::qn::RankGrid {
    if forward {
        shared.col_ranks.clone()
    } else {
        shared.row_ranks.clone()
    }
}

fn abelian_right_canonicity_dev<T: Scalar>(x: &TtVector<T>, range: std::ops::Range<usize>) -> f64 {
    let mut maxdev = 0.0f64;
    for k in range {
        let core = x.core(k);
        for l in core.row_qn.iter().copied() {
            let r = core.row_ranks[l];
            if r == 0 {
                continue;
            }
            let mut g = Mat::<T>::zeros(r, r);
            for f in qntt::Family::ALL {
                if let Some(b) = core.block(f, l) {
                    for i in 0..r {
                        for j in 0..r {
                            let mut acc = T::zero();
                            for c in 0..b.ncols() {
                                acc += *b.get(i, c) * b.get(j, c).conj_s();
                            }
                            *g.get_mut(i, j) += acc;
                        }
                    }
                }
            }
            for i in 0..r {
                for j in 0..r {
                    let target = if i == j { T::one() } else { T::zero() };
                    maxdev = maxdev.max((*g.get(i, j) - target).abs_s());
                }
            }
        }
    }
    maxdev
}

#[test]
fn tms_e1h_block_solver_boundary_bonds_complete_without_stagnating() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xE1_8000, 2, 0.05);
    let h = fx.abelian(space);
    let mut rng = seeded(0xE1_8001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (lam1, psi1) = ground(&h, &x0);
    assert!(
        (lam1 - fx.ev11[0]).abs() <= 1e-8,
        "the lower state is not the dense ground state"
    );
    let e1 = e_ref(&h, &psi1);

    let mut start: TtVector<f64> = state_from_sites(d, &[0], &[3]);
    start.right_orthogonalize(true);
    let dims: Vec<usize> = (0..d - 1)
        .map(|k| contract(start.core(k), start.core(k + 1)).num_coeffs())
        .collect();
    println!("tms_e1h: two-site local dimensions of the start = {dims:?}");
    assert_eq!(
        dims[0], 2,
        "the FIRST pair the sweep solves must have a local space of dimension exactly `roots` = 2 \
         ({dims:?}); anywhere else the rank growth of the earlier bonds moves the pair off the \
         skip test's boundary before it is reached"
    );

    for (tag, mode) in [("penalty", SsMode::Penalty), ("project", SsMode::Project)] {
        let o = StateSpecificOptions {
            mals: MalsOptions {
                eps: 1e-8,
                bond_dims: vec![2],
                noise: vec![0.0],
                max_iter: 3,
                ..Default::default()
            },
            mode,
            penalty: PenaltyRule::Auto,
            roots: 2,
            lower_energies: Some(vec![e1]),
            ..Default::default()
        };
        let (hist, y) =
            state_specific_mals_with_history(&h, &start, std::slice::from_ref(&psi1), &o);
        let skipped: Vec<usize> = hist.sweeps.iter().map(|s| s.skipped).collect();
        let swaps: Vec<usize> = hist.sweeps.iter().map(|s| s.swaps).collect();
        let ws: Vec<f64> = hist.sweeps.iter().map(|s| s.penalty).collect();
        println!(
            "tms_e1h {tag}: λ {:.12e} (E₁ = {:.12e}), skipped per iteration {skipped:?}, swaps \
             {swaps:?}, weights {ws:?}",
            hist.lambda, fx.ev11[0]
        );
        assert!(
            hist.lambda >= fx.ev11[0] - 1e-8,
            "{tag}: λ = {:.12e} sits below the global minimum {:.12e}",
            hist.lambda,
            fx.ev11[0]
        );
        assert!(y.check(), "{tag}: the returned state fails check()");
        let ny = y.norm();
        assert!(
            ny > 0.1 && ny <= 1.0 + 1e-10,
            "{tag}: the returned state has norm {ny:.6e}"
        );
        let total: usize = skipped.iter().sum();
        let per_iter_max = 2 * (d - 1);
        if matches!(mode, SsMode::Penalty) {
            assert_eq!(
                skipped[0], 0,
                "{tag}: the first iteration skipped {} pair(s) although nothing is blocked and \
                 every pair of the start hosts `roots` = 2 — bond 0's dim-2 space was NOT handed \
                 to the block solver",
                skipped[0]
            );
            assert_eq!(
                total, 0,
                "{tag}: {total} skips over the whole run, but the ranks only grow from the start's \
                 {dims:?}"
            );
        } else {
            for (k, &n) in skipped.iter().enumerate() {
                assert_eq!(
                    n, per_iter_max,
                    "{tag} iteration {k}: {n} of the {per_iter_max} pair visits were skipped; on \
                     this start every pair is under `roots + blocked` = 3 ({dims:?}), and a \
                     skipped pair grows no rank"
                );
            }
            let rq0 = e_ref(&h, &start);
            println!(
                "tms_e1h {tag}: λ {:.12e} vs the start's Rayleigh quotient {rq0:.12e} — no pair \
                 was ever solved",
                hist.lambda
            );
            assert!(
                (hist.lambda - rq0).abs() <= 1e-10 * rq0.abs().max(1.0),
                "{tag}: λ = {:.16e} but every pair was skipped, so the recorded energy must be \
                 the start's own Rayleigh quotient {rq0:.16e}",
                hist.lambda
            );
        }
        if matches!(mode, SsMode::Penalty) {
            for (k, w) in ws.iter().enumerate() {
                assert!(
                    *w >= 1.0,
                    "{tag} iteration {k}: the `Auto` weight {w:.3e} is below its floor of 1, so \
                     the `w > 0` guard — and with it the guess discipline the reachability \
                     argument rests on — was never live"
                );
            }
        }
    }
}
