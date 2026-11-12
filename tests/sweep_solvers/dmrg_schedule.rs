// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::fcidump::{TwoBodySymmetry, read_fcidump};
use qntt::solvers::mals::{MalsOptions, NoiseType, mals, mals_with_history};
use qntt::{QnSpace, RankGrid, SparseHamiltonian, TtVector, rayleigh_quotient, tt_randn, tt_state};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn h4() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP")
}

fn h4_system(r: usize) -> (SparseHamiltonian<f64>, TtVector<f64>, QnSpace) {
    let data = read_fcidump::<f64>(&h4(), TwoBodySymmetry::Eightfold).expect("read h4");
    let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
    let h = SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
    let ranks: Vec<RankGrid> = (0..=space.d)
        .map(|b| {
            RankGrid::from_fn(&space, |s| {
                if space.contains(b, s) {
                    (space.max_rank(b, s) as usize).min(r)
                } else {
                    0
                }
            })
        })
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let x0 = tt_randn::<f64, _>(space, &ranks, &mut rng);
    (h, x0, space)
}

#[test]
fn zero_noise_mals_reproduces_the_noise_free_driver_exactly() {
    let (h, x0, _space) = h4_system(4);

    let base = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        max_iter: 15,
        ..MalsOptions::default()
    };
    let (lambda_default, _) = mals(&h, &x0, &base);

    for seed in [1u64, 424242] {
        let opts = MalsOptions {
            noise: vec![0.0, 0.0, 0.0],
            seed,
            ..base.clone()
        };
        let (lambda, _) = mals(&h, &x0, &opts);
        assert_eq!(
            lambda, lambda_default,
            "an all-zero noise schedule changed the result (seed {seed})"
        );
    }
}

#[test]
fn noise_is_deterministic_in_the_seed() {
    let (h, x0, _space) = h4_system(2);
    let opts = |seed| MalsOptions {
        eps: 1e-10,
        bond_dims: vec![4, 8, 0],
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Random,
        seed,
        max_iter: 6,
        ..MalsOptions::default()
    };

    let (a, _) = mals(&h, &x0, &opts(2024));
    let (b, _) = mals(&h, &x0, &opts(2024));
    assert_eq!(a, b, "same seed must reproduce the run exactly");

    let (c, _) = mals(&h, &x0, &opts(2025));
    assert!(
        c.is_finite() && (c - a).abs() < 1.0,
        "a different seed must still converge to the same energy scale: {c} vs {a}"
    );
}

#[test]
fn noise_ends_variational_and_matches_the_noise_free_answer() {
    let (h, x0, _space) = h4_system(8);
    let quiet = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        max_iter: 30,
        ..MalsOptions::default()
    };
    let noisy = MalsOptions {
        noise: vec![1e-2, 1e-3, 1e-4, 0.0, 0.0, 0.0],
        noise_type: NoiseType::Random,
        seed: 5,
        ..quiet.clone()
    };

    let (lambda_quiet, _) = mals(&h, &x0, &quiet);
    let (lambda_noisy, x) = mals(&h, &x0, &noisy);

    assert!(
        (lambda_noisy - lambda_quiet).abs() <= 1e-8 * lambda_quiet.abs(),
        "noisy run {lambda_noisy} vs noise-free {lambda_quiet}"
    );
    let rq = rayleigh_quotient(&h, &x) / qntt::dot(&x, &x);
    assert!(
        rq >= lambda_quiet - 1e-8 * lambda_quiet.abs(),
        "final state is below the variational minimum: {rq} < {lambda_quiet}"
    );
}

#[test]
fn history_records_every_iteration() {
    let (h, x0, _space) = h4_system(3);
    let opts = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![2, 4, 8, 0],
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Random,
        seed: 11,
        max_iter: 5,
        ..MalsOptions::default()
    };
    let (hist, _x) = mals_with_history(&h, &x0, &opts);

    assert!(!hist.sweeps.is_empty());
    assert!(hist.sweeps.len() <= opts.max_iter);
    assert_eq!(hist.lambda, hist.sweeps.last().unwrap().energy);

    for (i, s) in hist.sweeps.iter().enumerate() {
        assert_eq!(s.bond_dim, opts.bond_dims[i.min(opts.bond_dims.len() - 1)]);
        assert_eq!(s.noise, opts.noise[i.min(opts.noise.len() - 1)]);
        assert!(s.discarded_weight >= 0.0 && s.discarded_weight <= 1.0);
        assert!(s.max_bond_dim > 0);
        assert!(s.energy.is_finite());
    }

    for s in &hist.sweeps {
        if s.bond_dim > 0 {
            assert!(
                s.max_bond_dim <= s.bond_dim,
                "M = {} exceeded its cap {}",
                s.max_bond_dim,
                s.bond_dim
            );
        }
    }
}

#[test]
fn empty_or_negative_noise_schedule_is_rejected() {
    let (h, x0, _space) = h4_system(2);

    let empty = MalsOptions {
        noise: vec![],
        ..MalsOptions::default()
    };
    assert!(std::panic::catch_unwind(|| mals(&h, &x0, &empty)).is_err());

    let negative = MalsOptions {
        noise: vec![-1e-3],
        ..MalsOptions::default()
    };
    assert!(std::panic::catch_unwind(|| mals(&h, &x0, &negative)).is_err());
}

#[test]
fn noise_escapes_a_determinant_start() {
    let data = read_fcidump::<f64>(&h4(), TwoBodySymmetry::Eightfold).expect("read h4");
    let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
    let h = SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);

    let occ: Vec<bool> = (0..space.d).map(|k| k < space.nup).collect();
    let hf = tt_state::<f64>(&occ, &occ);

    let quiet = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        max_iter: 30,
        ..MalsOptions::default()
    };
    let noisy = MalsOptions {
        noise: vec![1e-1, 1e-2, 1e-3, 1e-4, 0.0, 0.0, 0.0],
        noise_type: NoiseType::Random,
        seed: 3,
        ..quiet.clone()
    };

    let (lambda_quiet, _) = mals(&h, &hf, &quiet);
    let (lambda_noisy, _) = mals(&h, &hf, &noisy);

    let (_, x0, _) = h4_system(8);
    let (lambda_true, _) = mals(&h, &x0, &quiet);

    assert!(
        lambda_noisy <= lambda_quiet + 1e-12,
        "noise made the determinant start worse: {lambda_noisy} vs {lambda_quiet}"
    );
    assert!(
        lambda_noisy <= lambda_true + 1e-6 * lambda_true.abs(),
        "noisy run from the determinant did not reach the ground state: \
         {lambda_noisy} vs {lambda_true}"
    );
}

#[test]
fn perturbative_zero_noise_matches_the_noise_free_driver() {
    let (h, x0, _space) = h4_system(4);
    let base = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        max_iter: 15,
        ..MalsOptions::default()
    };
    let (lambda_default, _) = mals(&h, &x0, &base);

    let opts = MalsOptions {
        noise: vec![0.0, 0.0],
        noise_type: NoiseType::Perturbative,
        ..base.clone()
    };
    let (lambda, _) = mals(&h, &x0, &opts);
    assert_eq!(
        lambda, lambda_default,
        "an all-zero perturbative schedule changed the result"
    );
}

#[test]
fn perturbative_noise_is_deterministic() {
    let (h, x0, _space) = h4_system(2);
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

#[test]
fn perturbative_noise_stays_variational_during_the_noisy_sweeps() {
    let (h, x0, _space) = h4_system(8);
    let quiet = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        max_iter: 30,
        ..MalsOptions::default()
    };
    let (lambda_true, _) = mals(&h, &x0, &quiet);

    let noisy = MalsOptions {
        noise: vec![1e-2, 1e-2, 1e-2, 1e-2, 1e-2],
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
fn perturbative_full_rank_reaches_the_noise_free_answer() {
    let (h, x0, _space) = h4_system(8);
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
fn perturbative_noise_escapes_a_determinant_start() {
    let data = read_fcidump::<f64>(&h4(), TwoBodySymmetry::Eightfold).expect("read h4");
    let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
    let h = SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
    let occ: Vec<bool> = (0..space.d).map(|k| k < space.nup).collect();
    let hf = tt_state::<f64>(&occ, &occ);

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
    let (_, x0, _) = h4_system(8);
    let (lambda_true, _) = mals(&h, &x0, &quiet);

    assert!(
        lambda_quiet > lambda_true + 1e-4,
        "the determinant should trap the noise-free driver ({lambda_quiet} vs {lambda_true})"
    );
    assert!(
        lambda_noisy <= lambda_true + 1e-6 * lambda_true.abs(),
        "perturbative noise did not escape the trap: {lambda_noisy} vs {lambda_true}"
    );
}

#[test]
fn perturbative_capped_is_variational_and_matches_random() {
    let data = read_fcidump::<f64>(&h4(), TwoBodySymmetry::Eightfold).expect("read h4");
    let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
    let h = SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
    let occ: Vec<bool> = (0..space.d).map(|k| k < space.nup).collect();
    let hf = tt_state::<f64>(&occ, &occ);

    let (_, x0, _) = h4_system(8);
    let (lambda_true, _) = mals(
        &h,
        &x0,
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

fn c2v_table(d: usize) -> qntt::PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| ((k % 4) + 1) as u8).collect();
    qntt::PointGroup::from_orbsym(qntt::PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}

fn c2v_coeffs(
    pg: &qntt::PointGroup,
    d: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (qntt::coeff::CoeffTensor<f64>, qntt::coeff::CoeffTensor<f64>) {
    use rand::RngExt;
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
fn noise_escapes_a_symmetry_confined_determinant_at_nontrivial_group() {
    use qntt::vector::tt_state_in;

    let d = 6;
    let n = 4;
    let pg = c2v_table(d);
    let mut rng_h = Xoshiro256PlusPlus::seed_from_u64(0xCAFE_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng_h);
    let space = QnSpace::with_point_group(d, n / 2, n / 2, pg, qntt::Irrep::IDENTITY);
    let h = SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let occ: Vec<bool> = (0..d).map(|k| k < n / 2).collect();
    let x0: TtVector<f64> = tt_state_in(space, &occ, &occ);

    let caps = qntt::max_ranks(&space);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = caps[b][s].min(16);
            }
            r
        })
        .collect();
    let mut rng_x = Xoshiro256PlusPlus::seed_from_u64(0xCAFE_0002);
    let xr: TtVector<f64> = tt_randn(space, &ranks, &mut rng_x);
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
