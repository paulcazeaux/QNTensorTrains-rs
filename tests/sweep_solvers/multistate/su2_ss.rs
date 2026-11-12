// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::RoundMode;
use qntt::solvers::mals::{MalsOptions, NoiseType};
use qntt::solvers::multistate::StateAverageOptions;
use qntt::solvers::ssmals::{PenaltyRule, SsMode, StateSpecificOptions};
use qntt::su2::expand::expand;
use qntt::su2::hamiltonian::rayleigh_quotient;
use qntt::su2::solvers::mals::{mals, mals_with_history};
use qntt::su2::solvers::multistate::{
    MultiTt as Su2MultiTt, state_average_mals, state_average_mals_mut,
};
use qntt::su2::solvers::ssmals::{state_specific_mals, state_specific_mals_with_history};
use qntt::su2::twosite::factor_svd_average;
use qntt::su2::vector::tt_csf;
use qntt_testkit::randomized_su2_state;

fn zero_t(d: usize) -> CoeffTensor<f64> {
    CoeffTensor::<f64>::zeros(d, 2)
}

fn hb_tensor(d: usize, entries: &[(usize, usize, usize, usize, f64)]) -> CoeffTensor<f64> {
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for &(i, j, k, l, val) in entries {
        v.set(&[i, j, k, l], val);
    }
    v
}

fn hb_su2(space: Su2Space) -> Su2SparseHamiltonian<f64> {
    let v = hb_tensor(space.d, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)]);
    Su2SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
}

fn hb_polarized_su2(
    space: Su2Space,
    a: usize,
    b: usize,
    c: usize,
    e: usize,
) -> Su2SparseHamiltonian<f64> {
    let v = hb_tensor(space.d, &[(b, a, c, e, -1.0), (a, b, e, c, -1.0)]);
    Su2SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
}

fn two_coupling_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let v = hb_tensor(
        d,
        &[
            (2, 0, 3, 1, -1.0),
            (1, 3, 0, 2, -1.0),
            (0, 4, 2, 5, -1.0),
            (5, 2, 4, 0, -1.0),
        ],
    );
    (zero_t(d), v)
}

const TWO_COUPLING_SPAN: [(usize, usize); 8] = [
    (0, 1),
    (2, 3),
    (1, 0),
    (3, 2),
    (4, 5),
    (0, 2),
    (5, 4),
    (2, 0),
];

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

fn sa_opts(mals: MalsOptions) -> StateAverageOptions {
    StateAverageOptions {
        mals,
        ..Default::default()
    }
}

fn ground_su2<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
) -> (f64, Su2TtVector<T>) {
    state_specific_mals(h, x0, &[], &ss_opts(opts_exact()))
}

fn e_ref_su2<T: Scalar>(h: &Su2SparseHamiltonian<T>, psi: &Su2TtVector<T>) -> f64 {
    let n = psi.norm();
    let rq = rayleigh_quotient(h, psi).re_s();
    if n == 1.0 {
        rq
    } else {
        let invn = 1.0 / n;
        rq * invn * invn
    }
}

fn span_weight_su2(x: &Su2TtVector<f64>, two_sz: i64, span: &[(usize, usize)]) -> f64 {
    let d = x.space.d;
    let e = expand(x, two_sz).to_dense();
    let total: f64 = e.iter().map(|c| c * c).sum();
    let mut acc = 0.0f64;
    for &(u, dn) in span {
        let mut occ_up = vec![false; d];
        let mut occ_dn = vec![false; d];
        occ_up[u] = true;
        occ_dn[dn] = true;
        let c = e[basis_index(&occ_up, &occ_dn)];
        acc += c * c;
    }
    acc / total.max(1e-300)
}

fn assert_two_coupling_spectrum(d: usize) {
    let (t, v) = two_coupling_tv(d);
    let (_, ev11, _) = dense_sector(d, &t, &v, 1, 1);
    let (_, ev20, _) = dense_sector(d, &t, &v, 2, 0);
    let count = |ev: &[f64], e: f64| ev.iter().filter(|x| (**x - e).abs() <= 1e-9).count();
    let n11 = count(&ev11, -0.5);
    let n20 = count(&ev20, -0.5);
    println!(
        "two_coupling_tv: (1,1)[..4] = {:?}, (2,0)[..4] = {:?}; multiplicity of −0.5 is {n11} in \
         (1,1) and {n20} in (2,0) ⇒ {} state(s) with S = 0",
        &ev11[..4],
        &ev20[..4],
        n11 - n20
    );
    assert_eq!(
        n11 - n20,
        2,
        "two_coupling_tv: the S = 0 ground space must be exactly TWO-fold ({n11} − {n20})"
    );
    assert!(
        (ev11[0] + 0.5).abs() <= 1e-12,
        "two_coupling_tv: the ground level must be −0.5 (got {:.12e})",
        ev11[0]
    );
    assert!(
        ev11[n11] >= -0.5 + 0.4,
        "two_coupling_tv: the first level above the degenerate ground space must be well \
         separated (got {:.12e})",
        ev11[n11]
    );

    let vhb = hb_tensor(d, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)]);
    let (_, hb11, _) = dense_sector(d, &zero_t(d), &vhb, 1, 1);
    let (_, hb20, _) = dense_sector(d, &zero_t(d), &vhb, 2, 0);
    assert_eq!(
        count(&hb11, -0.5) - count(&hb20, -0.5),
        1,
        "the single-coupling fixture is expected to carry exactly one S = 0 ground state"
    );
}

#[test]
fn tms_c2su2_empty_lower_single_root_is_mals_bitwise() {
    let d = 6usize;
    let cases: [(Su2Space, &str); 2] = [
        (Su2Space::new(d, 2, 0), "2S=0"),
        (Su2Space::new(d, 2, 2), "2S=2"),
    ];
    for (space, label) in cases {
        let h = if space.two_s == 0 {
            hb_su2(space)
        } else {
            hb_polarized_su2(space, 0, 1, 2, 3)
        };
        let mut rng = seeded(0x5A_1000 + space.two_s as u64);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        assert!(x0.check(), "{label}: start");
        if space.two_s == 2 {
            assert_theta_cw_live(&x0, &format!("{label} start"));
        }

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
            let tag = format!("{label} {arm}");
            let (mhist, my) = mals_with_history(&h, &x0, &opts);
            let (shist, sy) =
                state_specific_mals_with_history(&h, &x0, &[], &ss_opts(opts.clone()));

            assert_eq!(
                shist.sweeps.len(),
                mhist.sweeps.len(),
                "{tag}: iteration count"
            );
            assert_f64_bits(shist.lambda, mhist.lambda, &format!("{tag}: final λ"));
            for (it, (s, m)) in shist.sweeps.iter().zip(mhist.sweeps.iter()).enumerate() {
                let w = |what: &str| format!("{tag}: sweep {it} {what}");
                assert_f64_bits(s.energy, m.energy, &w("energy"));
                assert_f64_bits(s.penalized, m.energy, &w("penalized == energy"));
                assert_f64_bits(
                    s.discarded_weight,
                    m.discarded_weight,
                    &w("discarded_weight"),
                );
                assert_eq!(s.max_bond_dim, m.max_bond_dim, "{}", w("max_bond_dim"));
                assert_eq!(s.bond_dim, m.bond_dim, "{}", w("bond_dim"));
                assert_f64_bits(s.noise, m.noise, &w("noise"));
                assert!(s.overlaps.is_empty(), "{}", w("overlaps"));
                assert_eq!(s.penalty, 0.0, "{}", w("penalty"));
                assert_eq!(s.swaps, 0, "{}", w("swaps"));
                assert_eq!(s.skipped, 0, "{}", w("skipped"));
                assert_eq!(
                    s.numops,
                    0,
                    "{}",
                    w("numops (documented 0 on the delegate)")
                );
                if it + 1 < shist.sweeps.len() {
                    assert!(!s.converged, "{}", w("converged early"));
                }
            }
            let last = shist.sweeps.last().expect("at least one sweep recorded");
            if shist.sweeps.len() < opts.max_iter {
                assert!(last.converged, "{tag}: an early stop must be recorded");
            }
            if !last.converged {
                assert_eq!(shist.sweeps.len(), opts.max_iter, "{tag}: exhaustion");
            }
            coeffs_bits_eq_su2(&sy, &my, &format!("{tag}: returned state"));

            let (slam, _) = state_specific_mals(&h, &x0, &[], &ss_opts(opts.clone()));
            let (mlam, _) = mals(&h, &x0, &opts);
            assert_f64_bits(slam, mlam, &format!("{tag}: state_specific_mals λ"));
        }
    }
}

#[test]
fn tms_c2su2_fixed_zero_penalty_sweep_is_mals_bitwise() {
    let d = 6usize;
    for two_s in [0usize, 2usize] {
        let space = Su2Space::new(d, 2, two_s);
        let label = format!("2S={two_s}");
        let h = if two_s == 0 {
            hb_su2(space)
        } else {
            hb_polarized_su2(space, 0, 1, 2, 3)
        };
        let mut rng = seeded(0x5A_2000 + two_s as u64);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        let (_, psi1) = ground_su2(&h, &x0);
        if two_s == 2 {
            assert_theta_cw_live(&psi1, &format!("{label} ψ₁"));
        }

        let noise_free = MalsOptions {
            eps: 1e-10,
            bond_dims: vec![0],
            noise: vec![0.0],
            max_iter: 5,
            ..Default::default()
        };
        let ramp = MalsOptions {
            eps: 1e-10,
            bond_dims: vec![4, 0],
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
            let label = format!("{label} {arm}");
            let (mhist, my) = mals_with_history(&h, &x0, &opts);
            let o = StateSpecificOptions {
                mals: opts.clone(),
                penalty: PenaltyRule::Fixed(0.0),
                ..Default::default()
            };
            let (shist, sy) =
                state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);

            assert_eq!(
                shist.sweeps.len(),
                mhist.sweeps.len(),
                "{label}: iteration count"
            );
            assert_f64_bits(shist.lambda, mhist.lambda, &format!("{label}: final λ"));
            for (it, (s, m)) in shist.sweeps.iter().zip(mhist.sweeps.iter()).enumerate() {
                assert_f64_bits(s.energy, m.energy, &format!("{label}: sweep {it} energy"));
                assert_f64_bits(
                    s.penalized,
                    s.energy,
                    &format!("{label}: sweep {it} θ == λ"),
                );
                assert_f64_bits(
                    s.discarded_weight,
                    m.discarded_weight,
                    &format!("{label}: sweep {it} discarded_weight"),
                );
                assert_eq!(
                    s.max_bond_dim, m.max_bond_dim,
                    "{label}: sweep {it} max_bond_dim"
                );
                assert_f64_bits(s.noise, m.noise, &format!("{label}: sweep {it} noise"));
                assert_eq!(s.penalty, 0.0, "{label}: sweep {it} penalty");
                assert_eq!(s.skipped, 0, "{label}: sweep {it} skipped");
                assert!(
                    s.numops >= 1,
                    "{label}: sweep {it} numops on the sweep path"
                );
                assert_eq!(s.overlaps.len(), 1, "{label}: sweep {it} overlaps length");
            }
            coeffs_bits_eq_su2(&sy, &my, &format!("{label}: returned state"));
            let last = shist.sweeps.last().unwrap();
            let measured = qntt::su2::ops::dot(&psi1, &sy).abs_s() / psi1.norm();
            assert!(
                (last.overlaps[0] - measured).abs() <= 1e-14,
                "{label}: recorded overlap vs measured ({:.17e} vs {measured:.17e})",
                last.overlaps[0]
            );
        }
    }
}

#[test]
fn tms_c3su2_penalty_and_project_ladders_match_the_dense_spectrum() {
    let d = 6usize;
    let fx = gapped_tv(0xC3_5000, 3, 0.05);
    let slevels = fx.singlet_levels();
    assert_exclusion_gap(&slevels, 3, 0.05, "tms_c3su2 singlets");
    assert_exclusion_gap(&fx.ev20, 3, 0.05, "tms_c3su2 triplets");

    let arms: [(usize, i64, &str); 2] = [(0, 0, "2S=0"), (2, 2, "2S=2")];
    for (two_s, two_sz, label) in arms {
        let space = Su2Space::new(d, 2, two_s);
        let h = fx.su2(space);
        let (levels, dets, evecs): (Vec<f64>, &Vec<(Vec<bool>, Vec<bool>)>, Vec<Vec<f64>>) =
            if two_s == 0 {
                (
                    slevels.clone(),
                    &fx.dets11,
                    fx.singlets.iter().map(|&i| fx.evec11[i].clone()).collect(),
                )
            } else {
                (fx.ev20.clone(), &fx.dets20, fx.evec20.clone())
            };
        let mut rng = seeded(0xC3_6000 + two_s as u64);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

        let (lam1, psi1) = ground_su2(&h, &x0);
        assert!(
            (lam1 - levels[0]).abs() <= 1e-8,
            "{label}: rung 0 λ = {lam1:.12e} vs {:.12e}",
            levels[0]
        );

        for mode in [SsMode::Penalty, SsMode::Project] {
            let mut states = vec![psi1.clone()];
            for rung in 1..3usize {
                let tag = format!("tms_c3su2 {label} {mode:?} rung {rung}");
                let e_lower: Vec<f64> = states.iter().map(|p| e_ref_su2(&h, p)).collect();
                let o = StateSpecificOptions {
                    mals: opts_exact(),
                    mode,
                    penalty: PenaltyRule::Auto,
                    roots: 1,
                    lower_energies: Some(e_lower.clone()),
                    ..Default::default()
                };
                let (hist, y) = state_specific_mals_with_history(&h, &x0, &states, &o);

                assert!(y.check(), "{tag}: returned state fails check()");
                assert!(
                    (hist.lambda - levels[rung]).abs() <= 1e-8,
                    "{tag}: λ = {:.12e} vs the dense level {:.12e}",
                    hist.lambda,
                    levels[rung]
                );
                let nn = qntt::su2::ops::dot(&y, &y).re_s();
                assert!((nn - 1.0).abs() <= 1e-10, "{tag}: ‖x‖² = {nn:.12e}");
                let res = global_eigen_residual_su2(&h, &y, hist.lambda);
                assert!(res <= 1e-6, "{tag}: global eigen residual {res:.3e}");
                let proj = projection_on_dense(&expand(&y, two_sz), dets, &evecs[rung]);
                assert!(proj >= 1.0 - 1e-8, "{tag}: dense projection {proj:.12}");
                let dev = su2_right_canonicity_dev(&y, 1..d);
                println!(
                    "{tag}: λ = {:.12e}, residual {res:.3e}, θ-canonicity {dev:.3e}",
                    hist.lambda
                );
                assert!(dev <= 1e-12, "{tag}: right-canonicity deviation {dev:.3e}");
                if two_s == 2 {
                    assert_theta_cw_live(&y, &tag);
                }

                for (it, s) in hist.sweeps.iter().enumerate() {
                    assert_eq!(s.skipped, 0, "{tag}: sweep {it} skipped at full rank");
                    assert_eq!(s.swaps, 0, "{tag}: sweep {it} swaps at roots = 1");
                    assert_eq!(s.overlaps.len(), rung, "{tag}: sweep {it} overlaps length");
                    match mode {
                        SsMode::Penalty => {
                            assert!(
                                s.penalty >= 1.0,
                                "{tag}: sweep {it} penalty below the floor"
                            );
                            assert!(
                                s.penalized >= s.energy - 1e-12,
                                "{tag}: sweep {it} θ below λ"
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
                assert!(
                    hist.sweeps.last().unwrap().converged,
                    "{tag}: the last sweep must be converged"
                );
                if matches!(mode, SsMode::Penalty) {
                    assert_rule_replay(&hist, &e_lower, &tag);
                }
                let last = hist.sweeps.last().unwrap();
                for (j, p) in states.iter().enumerate() {
                    let measured = qntt::su2::ops::dot(p, &y).abs_s() / p.norm();
                    assert!(
                        (last.overlaps[j] - measured).abs() <= 1e-14,
                        "{tag}: overlaps[{j}] {:.17e} vs measured {measured:.17e}",
                        last.overlaps[j]
                    );
                }
                states.push(y);
            }
            let gram = gram_dev_su2(&states);
            let floor = if matches!(mode, SsMode::Project) {
                1e-10
            } else {
                1e-6
            };
            println!("tms_c3su2 {label} {mode:?}: Gram deviation = {gram:.3e}");
            assert!(gram <= floor, "{label} {mode:?}: Gram deviation {gram:.3e}");
        }
    }
}

#[test]
fn tms_c4su2_degenerate_singlet_ground_space_second_root() {
    let d = 6usize;
    assert_two_coupling_spectrum(d);
    let space = Su2Space::new(d, 2, 0);
    let (t, v) = two_coupling_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);

    let mut chosen: Option<(f64, Su2TtVector<f64>, Su2TtVector<f64>, f64)> = None;
    for trial in 0..3u64 {
        let mut r1 = seeded(0xC4_5000 + trial);
        let x1: Su2TtVector<f64> = randomized_su2_state(space, 5, 10, &mut r1);
        let (lam1, psi1) = ground_su2(&h, &x1);
        if (lam1 + 0.5).abs() > 1e-6 {
            println!("tms_c4su2 trial {trial}: ground run reached {lam1:.6}, retrying");
            continue;
        }
        let mut r2 = seeded(0xC4_6000 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 5, 10, &mut r2);
        let e1 = e_ref_su2(&h, &psi1);
        let o = StateSpecificOptions {
            mals: MalsOptions {
                eps: 1e-8,
                conv_tol: Some(1e-12),
                bond_dims: vec![0],
                noise: vec![0.0],
                max_iter: 20,
                ..Default::default()
            },
            lower_energies: Some(vec![e1]),
            ..Default::default()
        };
        let (hist, y) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);
        let ok = (hist.lambda + 0.5).abs() <= 1e-6;
        chosen = Some((
            hist.lambda,
            y,
            psi1,
            hist.sweeps.last().expect("at least one sweep").penalty,
        ));
        if ok {
            break;
        }
    }

    let (lambda, y, psi1, w_last) = chosen.expect("tms_c4su2: no attempt produced a result");
    assert!(
        (lambda + 0.5).abs() <= 1e-8,
        "tms_c4su2: the penalized second singlet must sit at −0.5 (got {lambda:.12e})"
    );
    let ov = qntt::su2::ops::dot(&psi1, &y).abs_s() / psi1.norm();
    assert!(ov <= 1e-8, "tms_c4su2: |⟨ψ̂₁,x⟩| = {ov:.3e}");
    let sw = span_weight_su2(&y, 0, &TWO_COUPLING_SPAN);
    println!("tms_c4su2: λ = {lambda:.12e}, span weight = {sw:.12e}, final w = {w_last}");
    assert!(
        (sw - 1.0).abs() <= 1e-6,
        "tms_c4su2: the root carries only {sw:.12e} of its weight on the analytic span"
    );
    assert_eq!(
        w_last, 1.0,
        "tms_c4su2: on a degenerate ground space λ⊥ = E₁ and the auto rule returns exactly 1"
    );
}

#[test]
fn tms_c8su2_auto_from_psi1_plus_noise_reaches_the_second_root() {
    let d = 6usize;
    let two_s = 2usize;
    let space = Su2Space::new(d, 2, two_s);
    let fx = gapped_tv(0xC8_5000, 2, 0.05);
    let h = fx.su2(space);
    let levels = fx.ev20.clone();
    let mut rng = seeded(0xC8_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (lam1, psi1) = ground_su2(&h, &x0);
    assert!(
        (lam1 - levels[0]).abs() <= 1e-8,
        "tms_c8su2: ψ₁ is not the ground state ({lam1:.12e} vs {:.12e})",
        levels[0]
    );
    assert_theta_cw_live(&psi1, "tms_c8su2 ψ₁");
    let e1 = e_ref_su2(&h, &psi1);

    let mut xs = psi1.clone();
    if xs.ortho != Some(0) {
        xs.right_orthogonalize(true);
    }
    let c0 = xs.core(0).clone();
    let mut nrng = seeded(0xC8_5003);
    let mut r = c0.clone();
    r.for_each_coeff_mut(|z| *z = f64::sample_standard(&mut nrng));
    let rn = r.norm_center();
    assert!(rn > 0.0, "tms_c8su2: zero random fill");
    let mut c = c0.clone();
    c.axpy(0.1 * c0.norm_center() / rn, &r);
    xs.set_core(c);

    let dom = qntt::su2::ops::dot(&psi1, &xs).abs_s() / (psi1.norm() * xs.norm());
    assert!(
        dom > 0.9,
        "tms_c8su2: the start must be ψ₁-dominated ({dom:.6})"
    );

    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (hist, y) = state_specific_mals_with_history(&h, &xs, std::slice::from_ref(&psi1), &o);
    let ws: Vec<f64> = hist.sweeps.iter().map(|s| s.penalty).collect();
    println!(
        "tms_c8su2: weight trajectory = {ws:?}, λ = {:.12e} (E₂ = {:.12e})",
        hist.lambda, levels[1]
    );
    assert!(
        (hist.lambda - levels[1]).abs() <= 1e-10,
        "tms_c8su2: λ = {:.12e} vs the second level {:.12e}",
        hist.lambda,
        levels[1]
    );
    let ov = qntt::su2::ops::dot(&psi1, &y).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c8su2: |⟨ψ̂₁,x⟩| = {ov:.3e}");
    assert!(
        hist.sweeps.last().unwrap().converged,
        "tms_c8su2: converged"
    );
    assert_theta_cw_live(&y, "tms_c8su2 result");
    assert_rule_replay(&hist, &[e1], "tms_c8su2");
    let proj = projection_on_dense(&expand(&y, 2), &fx.dets20, &fx.evec20[1]);
    assert!(proj >= 1.0 - 1e-8, "tms_c8su2: dense projection {proj:.12}");
}

#[test]
fn tms_c10su2_threaded_matches_serial_bitwise() {
    let d = 6usize;
    let space = Su2Space::new(d, 4, 0);
    let fx = gapped_tv(0xCA_5000, 3, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xCA_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
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
    let e1 = e_ref_su2(&h, &psi1);

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
            .map(|k| su2_contract(y1.core(k), y1.core(k + 1)).mid_qn.len())
            .max()
            .unwrap_or(0);
        println!("tms_c10su2 {mode:?}: widest visited bond carries {widest} mid sectors");
        assert!(
            widest >= 4,
            "VACUOUS — the widest bond carries {widest} mid sectors, below the SU(2) fork \
             threshold (4), so nothing here runs threaded"
        );

        let mut worst = 0.0f64;
        for (a, b) in h1.sweeps.iter().zip(h8.sweeps.iter()) {
            worst = worst.max((a.energy - b.energy).abs());
            worst = worst.max((a.penalized - b.penalized).abs());
        }
        println!(
            "tms_c10su2 {mode:?}: max |Δ| between the 1- and 8-thread trajectories = {worst:.3e}"
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
            for j in 0..a.overlaps.len() {
                assert_f64_bits(
                    a.overlaps[j],
                    b.overlaps[j],
                    &tag(&format!("overlaps[{j}]")),
                );
            }
            assert_eq!(a.max_bond_dim, b.max_bond_dim, "{}", tag("max_bond_dim"));
            assert_eq!(a.swaps, b.swaps, "{}", tag("swaps"));
            assert_eq!(a.numops, b.numops, "{}", tag("numops"));
            assert_eq!(a.skipped, b.skipped, "{}", tag("skipped"));
            assert_eq!(a.converged, b.converged, "{}", tag("converged"));
        }
        coeffs_bits_eq_su2(&y1, &y8, &format!("{mode:?}: returned state"));
    }
}

#[test]
fn tms_c12su2_from_states_round_trip() {
    let d = 6usize;
    for two_s in [0usize, 2usize] {
        let space = Su2Space::new(d, 2, two_s);
        let label = format!("2S={two_s}");
        let h = if two_s == 0 {
            hb_su2(space)
        } else {
            hb_polarized_su2(space, 0, 1, 2, 3)
        };
        let xs: Vec<Su2TtVector<f64>> = (0..3u64)
            .map(|i| {
                let mut rng = seeded(0xCC_5000 + 0x10 * two_s as u64 + i);
                let y: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
                &y * (0.5 + i as f64)
            })
            .collect();
        let before = xs.clone();

        let m = Su2MultiTt::from_states(&xs, None);
        m.check();
        assert_eq!(m.nstates(), 3, "{label}: three states");
        assert_eq!(m.core_position, 0, "{label}: core position");
        assert_eq!(m.base.ortho, Some(0), "{label}: base.ortho");
        for b in 0..=d {
            let total: usize = xs.iter().map(|x| x.ranks[b].total()).sum();
            assert!(
                m.base.ranks[b].total() <= total,
                "{label}: bond {b} rank {} exceeds Σᵢ rankᵢ = {total}",
                m.base.ranks[b].total()
            );
        }
        let dev = su2_right_canonicity_dev(&m.base, 1..d);
        println!("{label}: base θ-right-canonicity deviation = {dev:.3e}");
        assert!(
            dev <= 1e-12,
            "{label}: the container's shared frames are NOT θ-right-canonical (max deviation \
             {dev:.3e}); the state-average local metric then carries a null space at the next \
             bond. The fix is the orthogonalization in `from_states`, never this assertion"
        );
        if two_s == 2 {
            assert_theta_cw_live(&m.base, &format!("{label} base"));
        }

        for (i, (y, x)) in m.states().iter().zip(xs.iter()).enumerate() {
            let nx2 = qntt::su2::ops::dot(x, x).re_s();
            let diff = (y - x).norm_lr(false);
            println!(
                "{label}: state {i} ‖y − x‖ = {diff:.3e} (‖x‖ = {:.6})",
                nx2.sqrt()
            );
            assert!(
                diff <= 1e-12 * nx2.sqrt(),
                "{label}: state {i} ‖y − x‖ = {diff:.3e}"
            );
            assert!(
                (qntt::su2::ops::dot(y, x).re_s() - nx2).abs() <= 1e-12 * nx2,
                "{label}: state {i} ⟨y,x⟩ ≠ ⟨x,x⟩"
            );
            assert!(
                (rayleigh_quotient(&h, y).re_s() - rayleigh_quotient(&h, x).re_s()).abs()
                    <= 1e-12 * nx2,
                "{label}: state {i} Rayleigh quotient moved"
            );
        }
        for (i, (a, b)) in xs.iter().zip(before.iter()).enumerate() {
            coeffs_bits_eq_su2(a, b, &format!("{label}: input {i} must not be mutated"));
        }

        let mw = Su2MultiTt::from_states(&xs, Some(vec![0.5, 0.3, 0.2]));
        mw.check();
        assert_eq!(mw.weights, vec![0.5, 0.3, 0.2], "{label}: explicit weights");

        let mut m2 = Su2MultiTt::from_states(&xs, None);
        let hist = state_average_mals_mut(
            &h,
            &mut m2,
            &sa_opts(MalsOptions {
                eps: 1e-8,
                bond_dims: vec![0],
                noise: vec![0.0],
                max_iter: 1,
                ..Default::default()
            }),
        );
        assert_eq!(
            hist.sweeps.len(),
            1,
            "{label}: one state-average iteration ran"
        );
        m2.check();
    }
}

#[test]
#[should_panic(expected = "from_states")]
fn tms_c12su2_from_states_refuses_an_empty_list() {
    let _ = Su2MultiTt::<f64>::from_states(&[], None);
}

#[test]
#[should_panic(expected = "space")]
fn tms_c12su2_from_states_refuses_mixed_spaces() {
    let d = 6usize;
    let mut r1 = seeded(0xCC_6001);
    let mut r2 = seeded(0xCC_6002);
    let a: Su2TtVector<f64> = randomized_su2_state(Su2Space::new(d, 2, 0), 2, 3, &mut r1);
    let b: Su2TtVector<f64> = randomized_su2_state(Su2Space::new(d, 2, 2), 2, 3, &mut r2);
    let _ = Su2MultiTt::from_states(&[a, b], None);
}

#[test]
fn tms_b8su2_degenerate_singlet_ground_space_state_average() {
    let d = 6usize;
    assert_two_coupling_spectrum(d);
    let space = Su2Space::new(d, 2, 0);
    let (t, v) = two_coupling_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let n = 2usize;

    let mut best: Option<(Vec<f64>, Vec<Su2TtVector<f64>>)> = None;
    for trial in 0..4u64 {
        let mut rng = seeded(0xB8_5000 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 5, 10, &mut rng);
        let (lams, ys) = state_average_mals(&h, &x0, n, &sa_opts(opts_exact()));
        let ok = (0..n).all(|i| (lams[i] + 0.5).abs() <= 1e-8);
        println!("tms_b8su2 trial {trial}: λ = {lams:?}");
        best = Some((lams, ys));
        if ok {
            break;
        }
    }
    let (lams, ys) = best.expect("tms_b8su2: no attempt produced a result");
    for i in 0..n {
        assert!(
            (lams[i] + 0.5).abs() <= 1e-8,
            "tms_b8su2: root {i} λ = {:.12e}, expected the degenerate −0.5 singlet level",
            lams[i]
        );
        assert!(ys[i].check(), "tms_b8su2: root {i} check()");
        let sw = span_weight_su2(&ys[i], 0, &TWO_COUPLING_SPAN);
        assert!(
            (sw - 1.0).abs() <= 1e-8,
            "tms_b8su2: root {i} carries {sw:.12e} of its weight on the analytic span"
        );
        let res = global_eigen_residual_su2(&h, &ys[i], lams[i]);
        assert!(res <= 1e-6, "tms_b8su2: root {i} residual {res:.3e}");
    }
    let gd = gram_dev_su2(&ys);
    println!("tms_b8su2: λ = {lams:?}, Gram deviation = {gd:.3e}");
    assert!(gd <= 1e-8, "tms_b8su2: returned Gram deviation {gd:.3e}");
}

#[test]
fn tms_b7ksu2_weighted_host_n_states_raises_to_the_next_largest_value() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let (x, cp0) = two_site_frame_su2::<f64>(space, 3, 0xB7_5001);
    assert_theta_cw_live(&x, "tms_b7ksu2 frame");
    let mut rng = seeded(0xB7_5002);
    let cps = unit_blocks_su2(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    let n = 3usize;

    let (shared, _s, cs, _infos, pooled, _lw) =
        factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(1), true, false);

    let mut weighted: Vec<(f64, Su2Sector)> = Vec::new();
    let mut raw: Vec<(f64, Su2Sector)> = Vec::new();
    for (m, evals, _) in pooled_rho_su2(&cps, &w, true) {
        let cw = qntt::su2::centerweight(m.two_s, space.two_s);
        for l in evals {
            weighted.push((l, m));
            raw.push((l / cw, m));
        }
    }
    weighted.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .expect("no NaN in the pooled spectrum")
    });
    raw.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .expect("no NaN in the pooled spectrum")
    });
    let width = |m: Su2Sector| cps[0].block(m).map_or(0, |b| b.ncols());
    let full = weighted.len();
    let mut rmin = 1usize;
    while rmin < full {
        let dim: usize = weighted[..rmin].iter().map(|&(_, m)| width(m)).sum();
        if dim >= n {
            break;
        }
        rmin += 1;
    }
    let kept = shared.col_ranks.clone();
    let same_order = weighted.iter().zip(raw.iter()).all(|(a, b)| {
        a.1 == b.1 && (a.0 - b.0 * qntt::su2::centerweight(a.1.two_s, space.two_s)).abs() < 1e-15
    });
    println!(
        "tms_b7ksu2: oracle r_min = {rmin} (full {full}), kept total = {}; the weighted and raw \
         orders {} on this seed",
        kept.total(),
        if same_order {
            "COINCIDE (a fixture note — tms_b5su2 pins the weighting itself)"
        } else {
            "DIFFER"
        }
    );

    assert_eq!(
        kept.total(),
        rmin,
        "the raise must stop at the first cap whose centre hosts {n} states"
    );
    assert_eq!(pooled.kept, rmin, "pooled TruncInfo.kept");
    for &m in cps[0].mid_qn.iter() {
        let want = weighted[..rmin].iter().filter(|&&(_, s)| s == m).count();
        assert_eq!(kept[m], want, "kept rank in sector {m:?}");
    }
    for (i, c) in cs.iter().enumerate() {
        assert!(
            c.num_coeffs() >= n,
            "centre {i} holds {} coefficient(s), below {n}",
            c.num_coeffs()
        );
    }
}

#[test]
#[should_panic(expected = "cannot host")]
fn tms_b7ksu2_host_n_states_refuses_when_full_rank_cannot_host_n() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let (_, cp0) = two_site_frame_su2::<f64>(space, 4, 0xB7_5011);
    let full: usize = cp0
        .mid_qn
        .iter()
        .filter_map(|&m| cp0.block(m))
        .map(|b| b.nrows())
        .sum();
    let n = full + 1;
    let mut rng = seeded(0xB7_5012);
    let cps = unit_blocks_su2(&cp0, n, &mut rng);
    let w = vec![1.0 / n as f64; n];
    let _ = factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(1), true, false);
}

fn perturb_center0_su2(psi: &Su2TtVector<f64>, alpha: f64, seed: u64) -> Su2TtVector<f64> {
    let mut y = psi.clone();
    if y.ortho != Some(0) {
        y.right_orthogonalize(true);
    }
    let c0 = y.core(0).clone();
    let mut rng = seeded(seed);
    let mut r = c0.clone();
    r.for_each_coeff_mut(|z| *z = f64::sample_standard(&mut rng));
    let rn = r.norm_center();
    assert!(rn > 0.0, "perturb_center0_su2: zero random fill");
    let mut c = c0.clone();
    c.axpy(alpha * c0.norm_center() / rn, &r);
    y.set_core(c);
    y
}

#[test]
fn tms_c8su2_sub_unit_lower_state_is_accepted_and_normalized() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xC8_6000, 2, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xC8_6001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (_, psi1) = ground_su2(&h, &x0);
    assert_theta_cw_live(&psi1, "tms_c8su2(c) ψ₁");

    let mut chosen: Option<(usize, Su2TtVector<f64>)> = None;
    for cap in [8usize, 6, 5, 4, 3, 2] {
        let y = psi1.round(0.0, qntt::Pooled(cap), RoundMode::Rel);
        let n = y.norm();
        println!("tms_c8su2(c): cap {cap} gives ‖ψᵗ‖ = {n:.6}");
        if (0.9..=0.9999).contains(&n) {
            chosen = Some((cap, y));
            break;
        }
    }
    let (cap, psit) = chosen.expect(
        "tms_c8su2(c): no rank cap in [8,6,5,4,3,2] produced a lower state of norm in \
         [0.9, 0.9999] — a FIXTURE finding; widen the cap list, never the window",
    );
    let nt = psit.norm();
    let et = e_ref_su2(&h, &psit);
    println!("tms_c8su2(c): using cap {cap}, ‖ψᵗ‖ = {nt:.12e}, E(ψᵗ) = {et:.12e}");
    assert!(nt != 1.0, "fixture: the sub-unit state must not be unit");

    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![et]),
        ..Default::default()
    };
    let (hist, x) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psit), &o);
    let last = hist.sweeps.last().unwrap();
    assert!(last.converged, "tms_c8su2(c): the run must converge");
    assert_rule_replay(&hist, &[et], "tms_c8su2(c)");
    assert_theta_cw_live(&x, "tms_c8su2(c) result");

    let measured = qntt::su2::ops::dot(&psit, &x).abs_s() / nt;
    assert!(
        (last.overlaps[0] - measured).abs() <= 1e-14,
        "tms_c8su2(c): the recorded overlap must carry the 1/‖ψ‖ ({:.17e} vs {measured:.17e})",
        last.overlaps[0]
    );
    assert!(
        (last.energy - (last.penalized - last.penalty * measured * measured)).abs() <= 1e-12,
        "tms_c8su2(c): λ = θ − w·Σo² violated ({:.17e} vs {:.17e})",
        last.energy,
        last.penalized - last.penalty * measured * measured
    );
    println!(
        "tms_c8su2(c): λ − E₂ = {:+.6e} (second-order in the lower state's error, \
         sign-indefinite — recorded, not asserted)",
        hist.lambda - fx.ev20[1]
    );
}

#[test]
fn tms_c8su2_norm_two_lower_state_matches_the_unit_one() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xC8_7000, 2, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xC8_7001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (_, psi1) = ground_su2(&h, &x0);
    let psin = &psi1 * 2.0;
    assert!(
        (psin.norm() - 2.0 * psi1.norm()).abs() <= 1e-12,
        "fixture: the scaled copy must have twice the norm"
    );
    let e1 = e_ref_su2(&h, &psi1);
    let en = e_ref_su2(&h, &psin);
    assert!(
        (en - e1).abs() <= 1e-10 * e1.abs(),
        "tms_c8su2(d): RQ/‖ψ‖² must be scale-invariant ({en:.12e} vs {e1:.12e})"
    );

    let schedule = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 8,
        ..Default::default()
    };
    let (ha, _) = state_specific_mals_with_history(
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
        "tms_c8su2(d): iteration count"
    );
    assert!(
        (ha.lambda - hn.lambda).abs() <= 1e-12 * ha.lambda.abs(),
        "tms_c8su2(d): λ {:.17e} vs {:.17e}",
        ha.lambda,
        hn.lambda
    );
    for (it, (a, b)) in ha.sweeps.iter().zip(hn.sweeps.iter()).enumerate() {
        for (what, u, v) in [
            ("penalty", a.penalty, b.penalty),
            ("energy", a.energy, b.energy),
            ("overlaps[0]", a.overlaps[0], b.overlaps[0]),
        ] {
            assert!(
                (u - v).abs() <= 1e-12 * u.abs().max(1.0),
                "tms_c8su2(d): sweep {it} {what}: {u:.17e} vs {v:.17e}"
            );
        }
    }
    let last = hn.sweeps.last().unwrap();
    let measured = qntt::su2::ops::dot(&psin, &xn).abs_s() / psin.norm();
    assert!(
        (last.overlaps[0] - measured).abs() <= 1e-14,
        "tms_c8su2(d): the recorded overlap must be normalized by ‖ψ‖ = 2"
    );
    let ov = qntt::su2::ops::dot(&psi1, &xn).abs_s() / psi1.norm();
    assert!(ov <= 1e-6, "tms_c8su2(d): |⟨ψ̂₁,x⟩| = {ov:.3e}");
    assert_theta_cw_live(&xn, "tms_c8su2(d) result");
}

#[test]
#[should_panic(expected = "norm")]
fn tms_c8su2_zero_norm_lower_state_is_refused() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0xC8_8001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let z = &x0 * 0.0;
    let _ = state_specific_mals(&h, &x0, std::slice::from_ref(&z), &ss_opts(opts_exact()));
}

#[test]
#[should_panic]
fn tms_c8su2_lower_energies_length_mismatch_is_refused() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0xC8_9001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let psi: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let o = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![-1.0, -2.0]),
        ..Default::default()
    };
    let _ = state_specific_mals(&h, &x0, std::slice::from_ref(&psi), &o);
}

#[test]
fn tms_c6su2_default_reference_energies_match_lower_energies() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xC6_5000, 2, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xC6_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (_, psi1) = ground_su2(&h, &x0);
    let e1 = e_ref_su2(&h, &psi1);

    let schedule = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 6,
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
        "tms_c6su2: the two runs must take the same number of iterations"
    );
    for (it, (a, b)) in h_default
        .sweeps
        .iter()
        .zip(h_given.sweeps.iter())
        .enumerate()
    {
        assert!(
            (a.penalty - b.penalty).abs() <= 1e-12 * b.penalty.abs().max(1.0),
            "tms_c6su2: sweep {it}: default-E penalty {:.17e} vs given {:.17e}",
            a.penalty,
            b.penalty
        );
        assert!(
            (a.energy - b.energy).abs() <= 1e-10 * b.energy.abs().max(1.0),
            "tms_c6su2: sweep {it}: energies diverge ({:.12e} vs {:.12e})",
            a.energy,
            b.energy
        );
    }
    assert_rule_replay(&h_given, &[e1], "tms_c6su2 lower_energies run");
}

#[test]
fn tms_c9su2_roots_two_follows_the_start_label() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xC9_5000, 3, 0.05);
    assert_exclusion_gap(&fx.ev20, 3, 0.05, "tms_c9su2");
    let h = fx.su2(space);
    let mut rng = seeded(0xC9_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let (_, psi1) = ground_su2(&h, &x0);
    let e1 = e_ref_su2(&h, &psi1);
    let o2 = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (h2, psi2) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o2);
    assert!((h2.lambda - fx.ev20[1]).abs() <= 1e-8, "tms_c9su2: rung 2");
    let lower12 = vec![psi1.clone(), psi2.clone()];
    let e12: Vec<f64> = lower12.iter().map(|p| e_ref_su2(&h, p)).collect();
    let o3 = StateSpecificOptions {
        mals: opts_exact(),
        lower_energies: Some(e12),
        ..Default::default()
    };
    let (h3, psi3) = state_specific_mals_with_history(&h, &x0, &lower12, &o3);
    assert!((h3.lambda - fx.ev20[2]).abs() <= 1e-8, "tms_c9su2: rung 3");
    assert_theta_cw_live(&psi3, "tms_c9su2 ψ₃");

    let x3 = perturb_center0_su2(&psi3, 0.1, 0xC9_5007);
    let dom = qntt::su2::ops::dot(&psi3, &x3).abs_s() / (psi3.norm() * x3.norm());
    assert!(
        dom > 0.9,
        "tms_c9su2: the start must be ψ₃-dominated ({dom:.6})"
    );

    let (hi, _) = state_specific_mals_with_history(&h, &x3, std::slice::from_ref(&psi1), &o2);
    assert!(
        (hi.lambda - fx.ev20[1]).abs() <= 1e-8,
        "tms_c9su2(i): roots = 1 must return E₂ (got {:.12e})",
        hi.lambda
    );
    for (it, s) in hi.sweeps.iter().enumerate() {
        assert_eq!(s.swaps, 0, "tms_c9su2(i): sweep {it} swaps at roots = 1");
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
        "tms_c9su2(ii): λ = {:.12e} (E₃ = {:.12e}), swaps per iteration = {swaps:?}",
        hii.lambda, fx.ev20[2]
    );
    assert!(
        (hii.lambda - fx.ev20[2]).abs() <= 1e-8,
        "tms_c9su2(ii): roots = 2 from a ψ₃ start must keep the continuation (got {:.12e})",
        hii.lambda
    );
    assert!(
        hii.sweeps.last().unwrap().swaps >= 1,
        "tms_c9su2(ii): the final iteration must record at least one swap — the design's tell \
         that homing, not the energy order, chose the root"
    );
    for (name, p, want_small) in [
        ("ψ₁", &psi1, true),
        ("ψ₂", &psi2, true),
        ("ψ₃", &psi3, false),
    ] {
        let ov = qntt::su2::ops::dot(p, &y).abs_s() / p.norm();
        if want_small {
            assert!(ov <= 1e-6, "tms_c9su2(ii): |⟨{name},x⟩| = {ov:.3e}");
        } else {
            assert!(
                (ov - 1.0).abs() <= 1e-6,
                "tms_c9su2(ii): |⟨{name},x⟩| = {ov:.6}"
            );
        }
    }
    assert_theta_cw_live(&y, "tms_c9su2(ii) result");
}

#[test]
fn tms_c11su2_max_iter_exhaustion_marks_every_sweep_unconverged() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xCB_5000, 2, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xCB_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (_, psi1) = ground_su2(&h, &x0);
    let e1 = e_ref_su2(&h, &psi1);
    let o = StateSpecificOptions {
        mals: MalsOptions {
            eps: 1e-10,
            conv_tol: Some(1e-14),
            bond_dims: vec![0],
            noise: vec![0.0],
            max_iter: 2,
            ..Default::default()
        },
        lower_energies: Some(vec![e1]),
        ..Default::default()
    };
    let (hist, _) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);
    assert_eq!(
        hist.sweeps.len(),
        2,
        "tms_c11su2: an exhausted run records exactly max_iter iterations"
    );
    for (it, s) in hist.sweeps.iter().enumerate() {
        assert!(
            !s.converged,
            "tms_c11su2: sweep {it} must not be marked converged on an exhausted run"
        );
    }
}

#[test]
fn tms_c11su2_csf_start_skips_one_dimensional_bonds() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xCB_6000, 2, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xCB_6001);
    let xr: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (lam1, psi1) = ground_su2(&h, &xr);
    let e1 = e_ref_su2(&h, &psi1);

    let codes = [
        Su2Family::Up,
        Su2Family::Up,
        Su2Family::Empty,
        Su2Family::Empty,
        Su2Family::Empty,
        Su2Family::Empty,
    ];
    let x0: Su2TtVector<f64> = tt_csf(&codes);
    assert_eq!(
        x0.space, space,
        "fixture: the CSF must live in the arm's space"
    );
    let dims: Vec<usize> = (0..d - 1)
        .map(|k| su2_contract(x0.core(k), x0.core(k + 1)).num_coeffs())
        .collect();
    println!("tms_c11su2: two-site local dimensions of the CSF start = {dims:?}");
    assert!(
        dims.iter().filter(|&&n| n == 1).count() >= 3,
        "fixture: a rank-1 CSF must leave most local spaces 1-dimensional ({dims:?}) — the skip \
         path is not reached otherwise"
    );

    let schedule = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 3,
        ..Default::default()
    };
    for (label, mode, roots) in [
        ("Project roots = 1", SsMode::Project, 1usize),
        ("Penalty roots = 2", SsMode::Penalty, 2usize),
    ] {
        let o = StateSpecificOptions {
            mals: schedule.clone(),
            mode,
            roots,
            lower_energies: Some(vec![e1]),
            ..Default::default()
        };
        let (hist, y) = state_specific_mals_with_history(&h, &x0, std::slice::from_ref(&psi1), &o);
        let skipped: usize = hist.sweeps.iter().map(|s| s.skipped).sum();
        println!(
            "tms_c11su2 {label}: λ = {:.12e}, Σskipped = {skipped} (the reached level is NOT \
             asserted — which level a rank-1 start settles on is a trajectory outcome)",
            hist.lambda
        );
        assert!(
            skipped >= 1,
            "{label}: a 1-dimensional local space that cannot host the requested vectors must be \
             SKIPPED (Σskipped = 0)"
        );
        assert!(
            hist.lambda >= lam1 - 1e-8,
            "{label}: the plain energy {:.12e} is below the ground level {lam1:.12e} — a Rayleigh \
             quotient cannot be",
            hist.lambda
        );
        assert!(y.check(), "{label}: returned state");
    }
}

#[test]
#[should_panic(expected = "the two-site local space at")]
fn tms_c12su2_from_states_refuses_a_local_space_below_n() {
    let codes = [
        Su2Family::Up,
        Su2Family::Up,
        Su2Family::Empty,
        Su2Family::Empty,
        Su2Family::Empty,
        Su2Family::Empty,
    ];
    let a: Su2TtVector<f64> = tt_csf(&codes);
    let b = a.clone();
    let _ = Su2MultiTt::from_states(&[a, b], None);
}

fn penalized_dense_su2(hd: &[Vec<f64>], chat: &[f64], w: f64) -> Vec<Vec<f64>> {
    let n = hd.len();
    let mut m = hd.to_vec();
    assert_eq!(chat.len(), n, "penalized_dense_su2: dimension mismatch");
    for i in 0..n {
        for j in 0..n {
            m[i][j] += w * chat[i] * chat[j];
        }
    }
    m
}

fn expanded_unit(x: &Su2TtVector<f64>, two_sz: i64, dets: &[(Vec<bool>, Vec<bool>)]) -> Vec<f64> {
    let e = expand(x, two_sz).to_dense();
    let full: f64 = e.iter().map(|c| c * c).sum::<f64>().sqrt();
    let mut c: Vec<f64> = dets.iter().map(|(ou, od)| e[basis_index(ou, od)]).collect();
    let nrm = c.iter().map(|a| a * a).sum::<f64>().sqrt();
    assert!(
        nrm >= (1.0 - 1e-10) * full,
        "the expanded state leaks outside the determinant sector ({nrm:.12e} of {full:.12e})"
    );
    for a in c.iter_mut() {
        *a /= nrm.max(1e-300);
    }
    c
}

fn penalized_singlet_min(
    hd11: &[Vec<f64>],
    ev11: &[f64],
    ev20: &[f64],
    chat: &[f64],
    w: f64,
) -> f64 {
    let m = penalized_dense_su2(hd11, chat, w);
    let (ev_pen, _) = dense_eigen(&m);
    let bare = singlet_indices(ev11, ev20);
    let idx = singlet_indices(&ev_pen, ev20);
    assert_eq!(
        idx.len(),
        bare.len(),
        "the penalty changed the singlet/triplet split of the dense spectrum ({} vs {}); the \
         penalty state is not a singlet, or the labelling collapsed on an accidental degeneracy",
        idx.len(),
        bare.len()
    );
    ev_pen[idx[0]]
}

#[test]
fn tms_e1gsu2_fixed_positive_weight_reaches_the_second_singlet() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xE1_7000, 2, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let mut rng = seeded(0xE1_7001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (lam1, psi1) = ground_su2(&h, &x0);
    assert!(
        (lam1 - levels[0]).abs() <= 1e-8,
        "tms_e1gsu2: ψ₁ is not the lowest singlet ({lam1:.12e} vs {:.12e})",
        levels[0]
    );
    let e1 = e_ref_su2(&h, &psi1);
    let gap = levels[1] - levels[0];
    let w = 10.0 * gap;

    let hd11 = dense_hamiltonian_matrix(d, &fx.t, &fx.v, &fx.dets11);
    let chat = expanded_unit(&psi1, 0, &fx.dets11);
    let mu0 = penalized_singlet_min(&hd11, &fx.ev11, &fx.ev20, &chat, w);
    println!(
        "tms_e1gsu2: gap {gap:.6e}, w = 10·gap = {w:.6e}, dense penalized singlet minimum \
         {mu0:.12e} (E₂ = {:.12e})",
        levels[1]
    );
    assert!(
        (mu0 - levels[1]).abs() <= 1e-10,
        "with w > gap the penalized singlet minimum must BE the second singlet ({mu0:.12e} vs \
         {:.12e})",
        levels[1]
    );

    let xs = perturb_center0_su2(&psi1, 0.3, 0xE1_7007);
    let dom = qntt::su2::ops::dot(&psi1, &xs).abs_s() / (psi1.norm() * xs.norm());
    println!("tms_e1gsu2: ψ₁-dominated start |⟨ψ̂₁,x̂ₛ⟩| = {dom:.6}");
    assert!(
        dom > 0.9,
        "the second start must be ψ₁-dominated ({dom:.6}) — that is what makes it a non-trivial \
         start for a penalized run"
    );

    for (tag, start) in [("neutral", &x0), ("psi1-dominated", &xs)] {
        let o = StateSpecificOptions {
            mals: opts_exact(),
            penalty: PenaltyRule::Fixed(w),
            lower_energies: Some(vec![e1]),
            ..Default::default()
        };
        let (hist, y) =
            state_specific_mals_with_history(&h, start, std::slice::from_ref(&psi1), &o);
        let last = hist.sweeps.last().expect("at least one iteration recorded");
        let ov = qntt::su2::ops::dot(&psi1, &y).abs_s() / psi1.norm();
        println!(
            "tms_e1gsu2 {tag}: λ {:.12e} (E₂ = {:.12e}), θ {:.12e}, final |⟨ψ̂₁,y⟩| {ov:.3e}, \
             sweeps {}, converged {}",
            hist.lambda,
            levels[1],
            last.penalized,
            hist.sweeps.len(),
            last.converged
        );
        assert!(
            (hist.lambda - levels[1]).abs() <= 1e-8,
            "{tag}: λ = {:.12e} vs the second singlet {:.12e}",
            hist.lambda,
            levels[1]
        );
        assert!(ov <= 1e-6, "{tag}: final |⟨ψ̂₁,y⟩| = {ov:.3e}");
        assert!(
            last.converged,
            "{tag}: the last iteration must be converged"
        );
        for (k, s) in hist.sweeps.iter().enumerate() {
            assert_eq!(
                s.penalty.to_bits(),
                w.to_bits(),
                "{tag} sweep {k}: `Fixed` is a CONSTANT — the recorded weight is {:.17e}, not \
                 {w:.17e}",
                s.penalty
            );
            assert!(
                s.penalized >= mu0 - 1e-10,
                "{tag} sweep {k}: the penalized Ritz value {:.12e} sits below the DENSE penalized \
                 singlet minimum {mu0:.12e}",
                s.penalized
            );
        }
        let sig: f64 = last.overlaps.iter().map(|q| q * q).sum();
        assert!(
            (last.energy - (last.penalized - w * sig)).abs() <= 1e-8,
            "{tag}: λ = θ − w·Σ|⟨p,u⟩|² violated ({:.12e} vs {:.12e})",
            last.energy,
            last.penalized - w * sig
        );
        let proj = projection_on_dense(&expand(&y, 0), &fx.dets11, &fx.evec11[fx.singlets[1]]);
        assert!(
            proj >= 1.0 - 1e-8,
            "{tag}: the returned state is not the second dense singlet (projection {proj:.12})"
        );
        assert!(y.check(), "{tag}: the returned state fails check()");
        let cdev = su2_right_canonicity_dev(&y, 1..d);
        assert!(
            cdev <= 1e-12,
            "{tag}: the θ-right-canonical law is broken by the penalized write-back ({cdev:.3e})"
        );
    }

    let space_p = Su2Space::new(d, 2, 2);
    let hp = fx.su2(space_p);
    let mut rngp = seeded(0xE1_7011);
    let x0p: Su2TtVector<f64> = randomized_su2_state(space_p, 2, 4, &mut rngp);
    let (lam1p, psi1p) = ground_su2(&hp, &x0p);
    assert!(
        (lam1p - fx.ev20[0]).abs() <= 1e-8,
        "the polarized reference state is not the (2,0) ground level"
    );
    assert_theta_cw_live(&psi1p, "tms_e1gsu2 polarized ψ₁");
    let e1p = e_ref_su2(&hp, &psi1p);
    let gap_p = fx.ev20[1] - fx.ev20[0];
    let wp = 10.0 * gap_p;
    let hd20 = dense_hamiltonian_matrix(d, &fx.t, &fx.v, &fx.dets20);
    let chatp = expanded_unit(&psi1p, 2, &fx.dets20);
    let mp = penalized_dense_su2(&hd20, &chatp, wp);
    let (ev_penp, _) = dense_eigen(&mp);
    println!(
        "tms_e1gsu2 polarized: gap {gap_p:.6e}, w {wp:.6e}, dense penalized minimum {:.12e} \
         (ev20[1] = {:.12e})",
        ev_penp[0], fx.ev20[1]
    );
    assert!(
        (ev_penp[0] - fx.ev20[1]).abs() <= 1e-10,
        "with w > gap the polarized penalized minimum must be the second (2,0) level"
    );
    let op = StateSpecificOptions {
        mals: opts_exact(),
        penalty: PenaltyRule::Fixed(wp),
        lower_energies: Some(vec![e1p]),
        ..Default::default()
    };
    let (histp, yp) =
        state_specific_mals_with_history(&hp, &x0p, std::slice::from_ref(&psi1p), &op);
    let lastp = histp.sweeps.last().expect("recorded");
    println!(
        "tms_e1gsu2 polarized: λ {:.12e} vs ev20[1] {:.12e}, θ {:.12e}",
        histp.lambda, fx.ev20[1], lastp.penalized
    );
    assert!(
        (histp.lambda - fx.ev20[1]).abs() <= 1e-8,
        "polarized: λ = {:.12e} vs {:.12e}",
        histp.lambda,
        fx.ev20[1]
    );
    for (k, s) in histp.sweeps.iter().enumerate() {
        assert_eq!(
            s.penalty.to_bits(),
            wp.to_bits(),
            "polarized sweep {k}: the recorded weight is not the fixed one"
        );
        assert!(
            s.penalized >= ev_penp[0] - 1e-10,
            "polarized sweep {k}: θ {:.12e} below the dense penalized minimum {:.12e}",
            s.penalized,
            ev_penp[0]
        );
    }
    assert_theta_cw_live(&yp, "tms_e1gsu2 polarized result");
    let projp = projection_on_dense(&expand(&yp, 2), &fx.dets20, &fx.evec20[1]);
    assert!(
        projp >= 1.0 - 1e-8,
        "polarized: the returned state is not the second dense (2,0) eigenvector ({projp:.12})"
    );
    assert!(
        su2_right_canonicity_dev(&yp, 1..d) <= 1e-12,
        "polarized: the θ-right-canonical law is broken"
    );
}

#[test]
fn tms_e1gsu2_fixed_half_gap_collapses_and_homing_does_not_rescue() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xE1_7100, 2, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let mut rng = seeded(0xE1_7101);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (lam1, psi1) = ground_su2(&h, &x0);
    assert!(
        (lam1 - levels[0]).abs() <= 1e-8,
        "ψ₁ is not the lowest singlet"
    );
    let e1 = e_ref_su2(&h, &psi1);
    let gap = levels[1] - levels[0];
    let w = 0.5 * gap;

    let hd11 = dense_hamiltonian_matrix(d, &fx.t, &fx.v, &fx.dets11);
    let chat = expanded_unit(&psi1, 0, &fx.dets11);
    let mu0 = penalized_singlet_min(&hd11, &fx.ev11, &fx.ev20, &chat, w);
    println!(
        "tms_e1gsu2 negative: gap {gap:.6e}, w = 0.5·gap = {w:.6e}, dense penalized singlet \
         minimum {mu0:.12e} vs E₁ + w = {:.12e} (E₂ = {:.12e})",
        e1 + w,
        levels[1]
    );
    assert!(
        (mu0 - (e1 + w)).abs() <= 1e-8,
        "for w < gap the penalized singlet minimum must be E₁ + w ({mu0:.12e} vs {:.12e}) — the \
         premise of the recorded negative",
        e1 + w
    );

    let xs = perturb_center0_su2(&psi1, 0.3, 0xE1_7107);
    let dom = qntt::su2::ops::dot(&psi1, &xs).abs_s() / (psi1.norm() * xs.norm());
    assert!(dom > 0.9, "the start must be ψ₁-dominated ({dom:.6})");

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
        let last = hist.sweeps.last().expect("recorded");
        let swaps: Vec<usize> = hist.sweeps.iter().map(|s| s.swaps).collect();
        println!(
            "tms_e1gsu2 negative roots = {roots}: final overlap {:.6}, λ {:.12e} (E₁ = {:.12e}), \
             θ {:.12e} (E₁ + w = {:.12e}), swaps per iteration {swaps:?}",
            last.overlaps[0],
            hist.lambda,
            e1,
            last.penalized,
            e1 + w
        );
        assert!(
            last.overlaps[0] > 0.5,
            "roots = {roots}: an under-gap fixed weight must COLLAPSE onto ψ₁ (final overlap \
             {:.6}); if this stops holding the recorded negative has changed and the design note \
             must be revisited",
            last.overlaps[0]
        );
        assert!(
            (last.penalized - (e1 + w)).abs() <= 1e-6,
            "roots = {roots}: θ = {:.12e} vs the exact penalized minimum E₁ + w = {:.12e}",
            last.penalized,
            e1 + w
        );
        assert!(
            (hist.lambda - e1).abs() <= 1e-6,
            "roots = {roots}: the plain energy must be E₁ (got {:.12e})",
            hist.lambda
        );
        assert!(
            (last.energy - (last.penalized - w * last.overlaps[0] * last.overlaps[0])).abs()
                <= 1e-8,
            "roots = {roots}: λ = θ − w·Σ|⟨p,u⟩|² violated"
        );
        for (k, s) in hist.sweeps.iter().enumerate() {
            assert_eq!(
                s.penalty.to_bits(),
                w.to_bits(),
                "roots = {roots} sweep {k}: `Fixed` is unguarded AND constant — no escalation may \
                 raise it"
            );
        }
        assert!(y.check(), "roots = {roots}: returned state fails check()");
    }
}

#[test]
fn tms_e1hsu2_block_solver_boundary_bonds_complete_without_stagnating() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xE1_8100, 2, 0.05);
    let h = fx.su2(space);
    let mut rng = seeded(0xE1_8101);
    let xr: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (lam1, psi1) = ground_su2(&h, &xr);
    let e1 = e_ref_su2(&h, &psi1);
    assert_theta_cw_live(&psi1, "tms_e1hsu2 ψ₁");

    let patterns: [[Su2Family; 6]; 4] = [
        [
            Su2Family::Up,
            Su2Family::Up,
            Su2Family::Empty,
            Su2Family::Empty,
            Su2Family::Empty,
            Su2Family::Empty,
        ],
        [
            Su2Family::Up,
            Su2Family::Empty,
            Su2Family::Up,
            Su2Family::Empty,
            Su2Family::Empty,
            Su2Family::Empty,
        ],
        [
            Su2Family::Empty,
            Su2Family::Up,
            Su2Family::Up,
            Su2Family::Empty,
            Su2Family::Empty,
            Su2Family::Empty,
        ],
        [
            Su2Family::Up,
            Su2Family::Empty,
            Su2Family::Empty,
            Su2Family::Up,
            Su2Family::Empty,
            Su2Family::Empty,
        ],
    ];
    let mut found: Option<(usize, Su2TtVector<f64>, Vec<usize>)> = None;
    for (idx, codes) in patterns.iter().enumerate() {
        let mut cand: Su2TtVector<f64> = tt_csf(codes);
        assert_eq!(cand.space, space, "the CSF must live in the arm's space");
        cand.right_orthogonalize(true);
        let dims: Vec<usize> = (0..d - 1)
            .map(|k| su2_contract(cand.core(k), cand.core(k + 1)).num_coeffs())
            .collect();
        println!("tms_e1hsu2: CSF pattern {idx} two-site local dimensions {dims:?}");
        if dims[0] == 2 && found.is_none() {
            found = Some((idx, cand, dims));
        }
    }
    let (seed, start, dims) = found.expect(
        "no rank-1 SU(2) CSF among the probed patterns carried a FIRST two-site local space of \
         dimension exactly `roots` = 2 — a FIXTURE finding, never a reason to drop the arm",
    );
    println!("tms_e1hsu2: chosen CSF pattern {seed}, two-site local dimensions {dims:?}");

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
            "tms_e1hsu2 {tag}: λ {:.12e} (ground {lam1:.12e}), skipped per iteration {skipped:?}, \
             swaps {swaps:?}, weights {ws:?}",
            hist.lambda
        );
        assert!(
            hist.lambda >= lam1 - 1e-8,
            "{tag}: λ = {:.12e} sits below the ground level {lam1:.12e}",
            hist.lambda
        );
        assert!(y.check(), "{tag}: the returned state fails check()");
        let ny = y.norm();
        assert!(
            ny > 0.1 && ny <= 1.0 + 1e-10,
            "{tag}: the returned state has norm {ny:.6e}"
        );
        let per_iter_max = 2 * (d - 1);
        if matches!(mode, SsMode::Penalty) {
            assert_eq!(
                skipped[0], 0,
                "{tag}: the first iteration skipped {} pair visit(s) although nothing is blocked \
                 and bond 0 hosts exactly `roots` = 2 vectors ({dims:?}) — the boundary space was \
                 NOT handed to the block solver",
                skipped[0]
            );
            for (k, w) in ws.iter().enumerate() {
                assert!(
                    *w >= 1.0,
                    "{tag} iteration {k}: the `Auto` weight {w:.3e} is below its floor of 1, so \
                     the `w > 0` guard — and with it the guess discipline the reachability \
                     argument rests on — was never live"
                );
            }
        } else {
            for (k, &n) in skipped.iter().enumerate() {
                assert_eq!(
                    n, per_iter_max,
                    "{tag} iteration {k}: {n} of the {per_iter_max} pair visits were skipped; on \
                     this start every pair is under `roots + blocked` = 3 ({dims:?}), and a \
                     skipped pair grows no rank"
                );
            }
            let rq0 = e_ref_su2(&h, &start);
            println!(
                "tms_e1hsu2 {tag}: λ {:.12e} vs the start's Rayleigh quotient {rq0:.12e} — no \
                 pair was ever solved",
                hist.lambda
            );
            assert!(
                (hist.lambda - rq0).abs() <= 1e-10 * rq0.abs().max(1.0),
                "{tag}: λ = {:.16e} but every pair was skipped, so the recorded energy must be \
                 the start's own Rayleigh quotient {rq0:.16e}",
                hist.lambda
            );
        }
    }
}
