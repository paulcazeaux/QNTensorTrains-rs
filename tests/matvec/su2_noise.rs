// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::twosite::{TruncInfo, noise_rng};
use qntt::fcidump::{TwoBodySymmetry, read_fcidump};
use qntt::solvers::mals::NoiseType;
use qntt::su2::core::{Family, Su2Core};
use qntt::su2::frame::Su2Sketch;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hamiltonian::matvec::{h_sketch_mul, noise_density_matrix, sketch_h_mul};
use qntt::su2::qn::{Su2RankGrid, Su2Space};
use qntt::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use qntt::su2::solvers::mals::{MalsOptions, mals, mals_with_history};
use qntt::su2::twosite::{
    Su2ContractedCores, contract, factor_qc, factor_svd_noise, factor_svd_perturbative,
};
use qntt::su2::vector::{Su2TtVector, tt_csf, tt_randn as su2_tt_randn};
use qntt_testkit::randomized_su2_state;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn h4_su2() -> (Su2SparseHamiltonian<f64>, Su2Space) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP");
    let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).expect("read h4.FCIDUMP");
    let space = Su2Space::new(data.norb, data.nelec, 0);
    let h = Su2SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
    (h, space)
}

fn h4_su2_hf(space: Su2Space) -> Su2TtVector<f64> {
    let npairs = space.n / 2;
    let mut codes = vec![Family::Occ; npairs];
    codes.extend(vec![Family::Empty; space.d - npairs]);
    tt_csf(&codes)
}

fn assert_cores_bit_identical(a: &Su2Core<f64>, b: &Su2Core<f64>, ctx: &str) {
    assert_eq!(a.k, b.k, "{ctx}: site mismatch");
    let space = a.space;
    for f in Family::ALL {
        for s in space.bond_sectors(a.k) {
            match (a.block(f, s), b.block(f, s)) {
                (Some(ba), Some(bb)) => {
                    assert_eq!(
                        ba.nrows(),
                        bb.nrows(),
                        "{ctx}: {f:?}@{s:?} row count differs"
                    );
                    assert_eq!(
                        ba.ncols(),
                        bb.ncols(),
                        "{ctx}: {f:?}@{s:?} col count differs"
                    );
                    for j in 0..ba.ncols() {
                        for i in 0..ba.nrows() {
                            assert_eq!(
                                ba[(i, j)],
                                bb[(i, j)],
                                "{ctx}: {f:?}@{s:?}[{i},{j}] differ"
                            );
                        }
                    }
                }
                (None, None) => {}
                _ => panic!("{ctx}: {f:?}@{s:?} block presence differs"),
            }
        }
    }
}

fn assert_sketches_bit_identical(a: &Su2Sketch<f64>, b: &Su2Sketch<f64>, ctx: &str) {
    assert_eq!(a.site(), b.site(), "{ctx}: site mismatch");
    for s in a.space.bond_sectors(a.site()) {
        match (a.block(s), b.block(s)) {
            (Some(ba), Some(bb)) => {
                assert_eq!(ba.nrows(), bb.nrows(), "{ctx}: {s:?} row count differs");
                assert_eq!(ba.ncols(), bb.ncols(), "{ctx}: {s:?} col count differs");
                for j in 0..ba.ncols() {
                    for i in 0..ba.nrows() {
                        assert_eq!(ba[(i, j)], bb[(i, j)], "{ctx}: {s:?}[{i},{j}] differ");
                    }
                }
            }
            (None, None) => {}
            _ => panic!("{ctx}: {s:?} sketch presence differs"),
        }
    }
}

fn assert_trunc_info_eq(a: &TruncInfo, b: &TruncInfo, ctx: &str) {
    assert_eq!(a.kept, b.kept, "{ctx}: kept rank differs");
    assert_eq!(
        a.discarded_weight, b.discarded_weight,
        "{ctx}: discarded weight differs"
    );
}

#[test]
fn factor_svd_perturbative_zero_alpha_is_bit_identical_to_the_noise_free_driver() {
    let (h, space) = h4_su2();
    let mut rng = seeded(0x5b_a1);
    let x = randomized_su2_state(space, 2, 3, &mut rng);
    let d = space.d;

    for k in 0..d - 1 {
        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let fr = right_to_left_sketching(&h, &x)
            .into_iter()
            .nth(k + 1)
            .unwrap();
        let cp: Su2ContractedCores<f64> = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&cp);

        for forward in [true, false] {
            let half = if forward {
                sketch_h_mul(&fl, &h, &xk)
            } else {
                h_sketch_mul(&h, &xk1, &fr)
            };
            let noise_dm = noise_density_matrix(&h, &half, &xk, &xk1, forward);

            let (a0, s0, b0, info0) = factor_svd_perturbative(
                &cp,
                &noise_dm,
                1e-10,
                qntt::Pooled::NONE,
                true,
                0.0,
                forward,
            );
            let (a1, s1, b1, info1) =
                factor_svd_noise(&cp, 1e-10, qntt::Pooled::NONE, true, 0.0, &mut noise_rng(0));

            let ctx = format!("k={k} forward={forward}");
            assert_cores_bit_identical(&a0, &a1, &format!("{ctx} left core"));
            assert_sketches_bit_identical(&s0, &s1, &format!("{ctx} sketch"));
            assert_cores_bit_identical(&b0, &b1, &format!("{ctx} right core"));
            assert_trunc_info_eq(&info0, &info1, &ctx);
        }
    }
}

#[test]
fn zero_noise_mals_matches_the_noise_free_driver_for_every_noise_type() {
    let (h, space) = h4_su2();
    let x0 = randomized_su2_state(space, 3, 4, &mut seeded(0x5b_a2));

    let base = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        max_iter: 15,
        ..MalsOptions::default()
    };
    let (lambda_default, _) = mals(&h, &x0, &base);

    for nt in [NoiseType::Random, NoiseType::Perturbative] {
        for seed in [1u64, 424242] {
            let opts = MalsOptions {
                noise: vec![0.0, 0.0, 0.0],
                noise_type: nt,
                seed,
                ..base.clone()
            };
            let (lambda, _) = mals(&h, &x0, &opts);
            assert_eq!(
                lambda, lambda_default,
                "noise_type={nt:?} seed={seed}: an all-zero noise schedule changed the result"
            );
        }
    }
}

#[test]
fn perturbative_full_rank_reaches_the_noise_free_answer() {
    let (h, space) = h4_su2();
    let x0 = randomized_su2_state(space, 6, 8, &mut seeded(0x5b_f1));
    let quiet = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        max_iter: 30,
        ..MalsOptions::default()
    };
    let (lambda_quiet, _) = mals(&h, &x0, &quiet);

    let sustained = MalsOptions {
        noise: vec![1e-2; 10],
        noise_type: NoiseType::Perturbative,
        max_iter: 10,
        ..quiet.clone()
    };
    let (lambda_pert, _) = mals(&h, &x0, &sustained);
    assert!(
        (lambda_pert - lambda_quiet).abs() <= 1e-9 * lambda_quiet.abs(),
        "full-rank perturbative {lambda_pert} != noise-free {lambda_quiet}"
    );
}

#[test]
fn perturbative_noise_stays_variational_during_the_noisy_sweeps() {
    let (h, space) = h4_su2();
    let x0 = randomized_su2_state(space, 6, 8, &mut seeded(0x5b_f2));
    let quiet = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        max_iter: 30,
        ..MalsOptions::default()
    };
    let (lambda_true, _) = mals(&h, &x0, &quiet);

    let noisy = MalsOptions {
        noise: vec![1e-2; 5],
        noise_type: NoiseType::Perturbative,
        max_iter: 5,
        ..quiet.clone()
    };
    let (hist, _x) = mals_with_history(&h, &x0, &noisy);
    for (i, sw) in hist.sweeps.iter().enumerate() {
        assert!(
            sw.energy >= lambda_true - 1e-8 * lambda_true.abs(),
            "perturbative sweep {i} went below the variational minimum: {} < {lambda_true}",
            sw.energy
        );
    }
}

#[test]
fn perturbative_noise_escapes_a_csf_start() {
    let (h, space) = h4_su2();
    let hf = h4_su2_hf(space);

    let quiet = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        max_iter: 30,
        ..MalsOptions::default()
    };
    let noisy = MalsOptions {
        noise: vec![1e-4, 1e-4, 1e-4, 1e-5, 0.0, 0.0, 0.0],
        noise_type: NoiseType::Perturbative,
        ..quiet.clone()
    };

    let (lambda_quiet, _) = mals(&h, &hf, &quiet);
    let (lambda_noisy, _) = mals(&h, &hf, &noisy);
    let x_true = randomized_su2_state(space, 6, 8, &mut seeded(0x5b_e5));
    let (lambda_true, _) = mals(&h, &x_true, &quiet);

    assert!(
        lambda_quiet > lambda_true + 1e-4,
        "the CSF start should trap the noise-free driver ({lambda_quiet} vs {lambda_true})"
    );
    assert!(
        lambda_noisy <= lambda_true + 1e-6 * lambda_true.abs(),
        "perturbative noise did not escape the trap: {lambda_noisy} vs {lambda_true}"
    );
}

#[test]
fn perturbative_capped_is_variational_and_matches_random() {
    let (h, space) = h4_su2();
    let hf = h4_su2_hf(space);
    let x_true = randomized_su2_state(space, 6, 8, &mut seeded(0x5b_e6));
    let (lambda_true, _) = mals(
        &h,
        &x_true,
        &MalsOptions {
            eps: 1e-12,
            bond_dims: vec![0],
            max_iter: 30,
            ..MalsOptions::default()
        },
    );

    let base = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![4],
        noise: vec![1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 0.0, 0.0],
        seed: 3,
        max_iter: 20,
        ..MalsOptions::default()
    };
    let (rnd, _) = mals(
        &h,
        &hf,
        &MalsOptions {
            noise_type: NoiseType::Random,
            ..base.clone()
        },
    );
    let (hist, _) = mals_with_history(
        &h,
        &hf,
        &MalsOptions {
            noise_type: NoiseType::Perturbative,
            ..base.clone()
        },
    );
    let pert = hist.sweeps.last().unwrap().energy;

    for (i, sw) in hist.sweeps.iter().enumerate() {
        assert!(
            sw.energy >= lambda_true - 1e-8 * lambda_true.abs(),
            "capped perturbative sweep {i} went below the variational minimum: {} < {lambda_true}",
            sw.energy
        );
    }
    assert!(
        pert <= rnd + 1e-6 * rnd.abs().max(1.0),
        "capped perturbative ({pert}) converged worse than capped random ({rnd})"
    );
}

#[test]
fn perturbative_noise_is_deterministic_in_the_seed() {
    let (h, space) = h4_su2();
    let x0 = randomized_su2_state(space, 2, 3, &mut seeded(0x5b_d1));
    let opts = |seed| MalsOptions {
        eps: 1e-10,
        bond_dims: vec![4, 8, 0],
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Perturbative,
        seed,
        max_iter: 6,
        ..MalsOptions::default()
    };
    let (a, _) = mals(&h, &x0, &opts(1));
    let (b, _) = mals(&h, &x0, &opts(999));
    assert_eq!(a, b, "perturbative noise must not depend on the seed");
}

fn c2v_table(d: usize) -> qntt::PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| ((k % 4) + 1) as u8).collect();
    qntt::PointGroup::from_orbsym(qntt::PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}

fn c2v_coeffs(
    pg: &qntt::PointGroup,
    d: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (qntt::coeff::CoeffTensor<f64>, qntt::coeff::CoeffTensor<f64>) {
    use rand_distr::StandardNormal;
    let mut t = qntt::coeff::CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            if t.get(&[i, j]) != 0.0 {
                continue;
            }
            let val = if pg.irrep_of(i) == pg.irrep_of(j) {
                rng.sample::<f64, _>(StandardNormal)
            } else {
                0.0
            };
            t.set(&[i, j], val);
            t.set(&[j, i], val);
        }
    }
    let mut v = qntt::coeff::CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if v.get(&[i, j, k, l]) != 0.0 {
                        continue;
                    }
                    let irrep_ok = pg
                        .irrep_of(i)
                        .fuse(pg.irrep_of(j))
                        .fuse(pg.irrep_of(k))
                        .fuse(pg.irrep_of(l))
                        == qntt::Irrep::IDENTITY;
                    let val = if irrep_ok {
                        rng.sample::<f64, _>(StandardNormal)
                    } else {
                        0.0
                    };
                    for &(a, b, c, dd) in &[
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    ] {
                        v.set(&[a, b, c, dd], val);
                    }
                }
            }
        }
    }
    (t, v)
}

#[test]
fn noise_escapes_a_symmetry_confined_csf_start_at_nontrivial_group() {
    use qntt::su2::qn::max_ranks;
    use qntt::su2::vector::tt_csf_with_point_group;

    let d = 6;
    let n = 4;
    let pg = c2v_table(d);
    let (t, v) = c2v_coeffs(&pg, d, &mut seeded(0xCAFE_0001));
    let mut codes = vec![Family::Occ; n / 2];
    codes.extend(vec![Family::Empty; d - n / 2]);
    let x0: Su2TtVector<f64> = tt_csf_with_point_group(&codes, pg);
    let space = x0.space;
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);

    let caps = max_ranks(&space);
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = caps[b][s].min(16);
            }
            r
        })
        .collect();
    let xr: Su2TtVector<f64> = su2_tt_randn(space, &ranks, &mut seeded(0xCAFE_0002));
    let quiet_opts = MalsOptions {
        eps: 1e-10,
        max_iter: 12,
        bond_dims: vec![0],
        noise: vec![0.0],
        ..MalsOptions::default()
    };
    let (lambda_ref, _) = mals(&h, &xr, &quiet_opts);

    let (lambda_quiet, _) = mals(&h, &x0, &quiet_opts);
    assert!(
        lambda_quiet > lambda_ref + 1.0,
        "fixture no longer traps: quiet={lambda_quiet} vs ref={lambda_ref}"
    );

    for nt in [NoiseType::Perturbative, NoiseType::Random] {
        let (lambda_noisy, x) = mals(
            &h,
            &x0,
            &MalsOptions {
                eps: 1e-10,
                max_iter: 12,
                bond_dims: vec![0],
                noise: vec![1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 0.0, 0.0, 0.0],
                noise_type: nt,
                seed: 3,
                ..MalsOptions::default()
            },
        );
        assert!(x.check(), "escaped train ({nt:?}) failed check()");
        assert!(
            (lambda_noisy - lambda_ref).abs() <= 1e-8 * lambda_ref.abs(),
            "{nt:?} noise failed to escape the symmetry trap: noisy={lambda_noisy} vs \
             ref={lambda_ref} (quiet trap at {lambda_quiet})"
        );
    }
}
