// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::tt_state;
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

use qntt::solvers::lanczos::{LanczosOptions, lanczos};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{
    CoeffTensor, QnSpace, Scalar, SparseHamiltonian, TtVector, dot, rayleigh_quotient, tt_zeros,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t7a_mals_onebody_polarized() {
    let d = 6usize;
    let pairs: [(usize, usize); 8] = [
        (0, 0),
        (0, 1),
        (0, 5),
        (1, 4),
        (2, 3),
        (3, 5),
        (1, 3),
        (2, 5),
    ];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let space = QnSpace::from_n_two_sz(d, 2, 2);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[i, j], -1.0);
        t.set(&[j, i], -1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A10 + idx as u64);
        let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        x0.check();
        let opts = MalsOptions {
            eps: 1e-6,
            ..Default::default()
        };
        let (mu, y) = mals(&h, &x0, &opts);
        y.check();
        assert!(
            (rayleigh_quotient(&h, &y) + 1.0).abs() < 1e-4,
            "pair ({i},{j}): energy {mu}"
        );
        let hy = &h * &y;
        let yl = &y * mu;
        let resid = &hy - &yl;
        assert!(resid.norm_lr(false) < 1e-4, "pair ({i},{j}): residual");
    }
}

#[test]
fn t7a_mals_twobody_span_projection() {
    let d = 6usize;
    let cases: [((usize, usize, usize, usize), f64); 3] = [
        ((0, 1, 2, 3), -0.5),
        ((0, 0, 2, 2), -1.0),
        ((0, 0, 2, 3), -std::f64::consts::FRAC_1_SQRT_2),
    ];
    for (ci, &((a, b, c, e), analytic)) in cases.iter().enumerate() {
        let space = QnSpace::new(d, 1, 1);
        let t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[c, a, e, b], -1.0);
        v.set(&[b, e, a, c], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut span: Vec<(usize, usize)> = vec![(a, b), (c, e), (b, a), (e, c)];
        span.sort_unstable();
        span.dedup();
        let zs: Vec<TtVector<f64>> = span
            .iter()
            .map(|&(u, w)| state_from_sites(d, &[u], &[w]))
            .collect();

        let mut mu = f64::INFINITY;
        let mut y = tt_zeros::<f64>(space);
        for trial in 0..4u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A20 + ci as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
            x0.check();
            let opts = MalsOptions {
                eps: 1e-6,
                ..Default::default()
            };
            let (m, yt) = mals(&h, &x0, &opts);
            mu = m;
            y = yt;
            if (mu - analytic).abs() < 1e-4 {
                break;
            }
        }

        y.check();
        assert!(mu <= -0.5 + 1e-4, "case {ci}: energy bound {mu}");
        assert!(
            (mu - analytic).abs() < 1e-4,
            "case {ci}: analytic energy {mu} vs {analytic}"
        );
        let proj: f64 = zs
            .iter()
            .map(|z| {
                let ov = dot(&y, z);
                ov.abs_s() * ov.abs_s()
            })
            .sum();
        assert!(
            (proj - 1.0).abs() < 1e-4,
            "case {ci}: span projection {proj}"
        );
        let hy = &h * &y;
        let yl = &y * mu;
        let resid = &hy - &yl;
        assert!(resid.norm_lr(false) < 1e-4, "case {ci}: residual");
    }
}

#[test]
fn t7a_mals_tight_schedule() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 4, 1], -1.0);
    v.set(&[1, 4, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut mu = f64::INFINITY;
    let mut y = tt_zeros::<f64>(space);
    for trial in 0..3u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A30 + trial);
        let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        x0.check();
        let opts = MalsOptions {
            eps: 1e-6,
            ..Default::default()
        };
        let (m, yt) = mals(&h, &x0, &opts);
        mu = m;
        y = yt;
        if mu <= -0.5 + 1e-4 {
            break;
        }
    }
    y.check();
    assert!(mu <= -0.5 + 1e-4, "tight-schedule energy bound {mu}");
    let hy = &h * &y;
    let yl = &y * mu;
    let resid = &hy - &yl;
    assert!(resid.norm_lr(false) < 1e-4, "tight-schedule residual");
}

#[test]
fn t7a_lanczos_ground() {
    let d = 6usize;
    let pairs: [(usize, usize); 8] = [
        (0, 0),
        (0, 1),
        (0, 5),
        (1, 4),
        (2, 3),
        (3, 5),
        (1, 3),
        (2, 5),
    ];
    for (idx, &(i, j)) in pairs.iter().enumerate() {
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[i, j], -1.0);
        t.set(&[j, i], -1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut lambda = f64::INFINITY;
        let mut x = tt_zeros::<f64>(space);
        for trial in 0..3u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A60 + idx as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 2, 5, &mut rng);
            x0.check();
            let opts = LanczosOptions {
                tol: 1e-4,
                max_iter: 20,
            };
            let out = lanczos(&h, &x0, &opts);
            lambda = out.lambda;
            x = out.x;
            if (lambda + 2.0).abs() < 1e-4 {
                assert!(
                    !out.breakdown,
                    "pair ({i},{j}): converged with breakdown flag set"
                );
                break;
            }
        }
        x.check();
        assert!(
            (lambda + 2.0).abs() < 1e-4,
            "pair ({i},{j}): lanczos energy {lambda}"
        );
        let hx = &h * &x;
        let xl = &x * lambda;
        let resid = &hx - &xl;
        assert!(
            resid.norm_lr(false) < 1e-4,
            "pair ({i},{j}): lanczos residual"
        );
    }
}

#[test]
fn t7a_mals_rank_growth() {
    let d = 6usize;
    let space = QnSpace::from_n_two_sz(d, 2, 2);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[1, 0, 2, 3], -1.0);
    v.set(&[0, 1, 3, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut mu = f64::INFINITY;
    let mut y = tt_zeros::<f64>(space);
    for trial in 0..4u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7AA0 + trial);
        let x0: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
        x0.check();
        let start_max = x0
            .ranks
            .iter()
            .flat_map(|g| g.iter().map(|(_, &r)| r))
            .max()
            .unwrap();
        assert_eq!(start_max, 1, "x0 is not a rank-1 start");
        let opts = MalsOptions {
            eps: 1e-6,
            ..Default::default()
        };
        let (m, yt) = mals(&h, &x0, &opts);
        mu = m;
        y = yt;
        if (mu + 0.5).abs() < 1e-4 {
            break;
        }
    }
    y.check();
    assert!(
        (mu + 0.5).abs() < 1e-4,
        "rank-growth MALS energy {mu} (want -0.5)"
    );
    let grown_max = y
        .ranks
        .iter()
        .flat_map(|g| g.iter().map(|(_, &r)| r))
        .max()
        .unwrap();
    assert!(
        grown_max > 1,
        "MALS did not grow rank from the rank-1 start (max rank {grown_max})"
    );
}

#[test]
fn t7a_lanczos_twobody() {
    let d = 6usize;
    let cases: [(usize, usize, usize, usize); 2] = [(0, 1, 2, 3), (0, 2, 3, 4)];
    for (ci, &(a, b, c, e)) in cases.iter().enumerate() {
        let space = QnSpace::new(d, 1, 1);
        let t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[c, a, e, b], -1.0);
        v.set(&[b, e, a, c], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut span: Vec<(usize, usize)> = vec![(a, b), (c, e), (b, a), (e, c)];
        span.sort_unstable();
        span.dedup();
        let zs: Vec<TtVector<f64>> = span
            .iter()
            .map(|&(u, w)| state_from_sites(d, &[u], &[w]))
            .collect();

        let mut lambda = f64::INFINITY;
        let mut x = tt_zeros::<f64>(space);
        for trial in 0..4u64 {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A70 + ci as u64 * 10 + trial);
            let x0: TtVector<f64> = randomized_state(space, 2, 5, &mut rng);
            x0.check();
            let opts = LanczosOptions {
                tol: 1e-4,
                max_iter: 20,
            };
            let out = lanczos(&h, &x0, &opts);
            lambda = out.lambda;
            x = out.x;
            if (lambda + 0.5).abs() < 1e-4 {
                break;
            }
        }
        x.check();
        assert!(
            lambda <= -0.5 + 1e-4,
            "case {ci}: lanczos two-body energy bound {lambda}"
        );
        assert!(
            (lambda + 0.5).abs() < 1e-4,
            "case {ci}: lanczos two-body energy {lambda} != -0.5"
        );
        let proj: f64 = zs
            .iter()
            .map(|z| {
                let ov = dot(&x, z);
                ov.abs_s() * ov.abs_s()
            })
            .sum();
        assert!(
            (proj - 1.0).abs() < 1e-4,
            "case {ci}: lanczos span projection {proj}"
        );
        let hx = &h * &x;
        let xl = &x * lambda;
        let resid = &hx - &xl;
        assert!(
            resid.norm_lr(false) < 1e-4,
            "case {ci}: lanczos two-body residual"
        );
    }
}

#[test]
fn t7a_mals_threaded_matches_serial() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A99);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    x0.check();
    let opts = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        max_iter: 3,
        verbose: false,
        ..Default::default()
    };
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| mals(&h, &x0, &opts).0)
    };
    let l1 = run(1);
    let l8 = run(8);
    assert!(
        (l1 - l8).abs() < 1e-10 * l1.abs().max(1.0),
        "MALS threaded vs serial energy mismatch: {l1} vs {l8}"
    );
}
