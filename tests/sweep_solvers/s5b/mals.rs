// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5b_m1_mals_singlet_ground() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b10);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let opts = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![0],
        ..Default::default()
    };
    let (lambda, x) = mals(&h, &x0, &opts);
    assert!(x.check());
    assert!(
        (lambda - (-2.0)).abs() <= 1e-4,
        "mals singlet energy {lambda}"
    );
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "mals singlet residual {resid}");
    let (lambda2, _) = mals(&h, &x0, &opts);
    assert_eq!(lambda, lambda2, "mals is not deterministic on a rerun");
}

#[test]
fn s5b_a5t_als_threaded_matches_serial() {
    let space = Su2Space::new(8, 4, 0);
    let t = ha_tensor::<f64>(
        8,
        &[
            (0, 1, -1.0),
            (1, 2, -0.5),
            (2, 3, -0.7),
            (3, 4, -0.4),
            (4, 5, -0.6),
            (5, 6, -0.3),
            (6, 7, -0.5),
        ],
    );
    let v = hb_tensor::<f64>(
        8,
        &[
            (2, 0, 3, 1, -1.0),
            (1, 3, 0, 2, -1.0),
            (5, 3, 6, 4, -0.5),
            (4, 6, 3, 5, -0.5),
        ],
    );
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut rng = seeded(0x5B0D_0A15);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
    let opts = AlsOptions {
        eps: 1e-6,
        max_iter: 2,
        verbose: false,
    };
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| als(&h, &x0, &opts).0)
    };
    let l1 = run(1);
    let l8 = run(8);
    assert!(
        (l1 - l8).abs() <= 1e-10 * l1.abs().max(1.0),
        "als: 1-thread λ={l1} vs 8-thread λ={l8} exceed machine-precision tolerance"
    );
}

#[test]
fn s5b_m4t_mals_threaded_matches_serial() {
    let space = Su2Space::new(8, 4, 0);
    let t = ha_tensor::<f64>(8, &[(0, 1, -1.0), (2, 3, -0.7), (4, 5, -0.6), (6, 7, -0.5)]);
    let v = hb_tensor::<f64>(
        8,
        &[
            (2, 0, 3, 1, -1.0),
            (1, 3, 0, 2, -1.0),
            (5, 3, 6, 4, -0.5),
            (4, 6, 3, 5, -0.5),
        ],
    );
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut rng = seeded(0x5B10_0A15);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![20, 20],
        max_iter: 2,
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
        (l1 - l8).abs() <= 1e-10 * l1.abs().max(1.0),
        "mals: 1-thread λ={l1} vs 8-thread λ={l8} exceed machine-precision tolerance"
    );
}

#[test]
fn s5b_m2_mals_hb_from_rank1() {
    let space = Su2Space::new(6, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0x5b11);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
    assert_eq!(max_rank_of(&x0), 1, "x0 is not a rank-1 start");
    let opts = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![0],
        ..Default::default()
    };
    let (lambda, x) = mals(&h, &x0, &opts);
    assert!(x.check());
    assert!((lambda - (-0.5)).abs() <= 1e-4, "mals HB energy {lambda}");
    let resid = global_eigen_residual(&h, &x, lambda);
    assert!(resid <= 1e-4, "mals HB residual {resid}");
}

#[test]
fn s5b_m3_mals_rank_growth_pathcount_caps() {
    let space = Su2Space::new(6, 2, 2);
    let h = hb_polarized_su2(space, 0, 1, 2, 3);
    let opts = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![0],
        ..Default::default()
    };
    let mut ok = false;
    let mut y = qntt::su2::vector::tt_zeros::<f64>(space);
    let mut last_lambda = f64::INFINITY;
    for trial in 0..4u64 {
        let mut rng = seeded(0x5b12 + trial);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
        assert_eq!(max_rank_of(&x0), 1, "x0 is not a rank-1 start");
        let (lambda, yt) = mals(&h, &x0, &opts);
        last_lambda = lambda;
        if (lambda - (-0.5)).abs() <= 1e-4 {
            y = yt;
            ok = true;
            break;
        }
    }
    assert!(
        ok,
        "mals rank growth: did not reach -0.5, last {last_lambda}"
    );
    assert!(y.check());
    let (n_theta, n_cw) = count_theta_cw(&y);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight trivial (n_theta={n_theta}, n_cw={n_cw})"
    );
    assert!(
        max_rank_of(&y) > 1,
        "mals did not grow rank from the rank-1 start"
    );
    let caps = qntt::su2::qn::max_ranks(&space);
    for (b, cap_grid) in caps.iter().enumerate() {
        for s in space.bond_sectors(b) {
            assert!(
                y.rank(b, s) <= cap_grid[s],
                "bond {b} sector {s:?}: rank {} exceeds pathcount cap {}",
                y.rank(b, s),
                cap_grid[s]
            );
        }
    }
}

#[test]
fn s5b_m4_mals_rank_cap_is_live() {
    let space = Su2Space::new(6, 2, 2);
    let h = hb_polarized_su2(space, 0, 1, 2, 3);
    let mut rng = seeded(0x5b13);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);

    let opts_cap1 = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![1],
        ..Default::default()
    };
    let (lambda1, y1) = mals(&h, &x0, &opts_cap1);
    assert!(y1.check());
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert!(
                y1.rank(b, s) <= 1,
                "bond_dims=[1] cap violated at bond {b} sector {s:?}"
            );
        }
    }
    assert!(
        lambda1 >= -0.5 + 1e-2,
        "bond_dims=[1] should MISS the -0.5 ground by >= 1e-2, got {lambda1}"
    );

    let opts_cap2 = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![2],
        ..Default::default()
    };
    let (lambda2, y2) = mals(&h, &x0, &opts_cap2);
    assert!(y2.check());
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            assert!(
                y2.rank(b, s) <= 2,
                "bond_dims=[2] cap violated at bond {b} sector {s:?}"
            );
        }
    }
    assert!(
        (lambda2 - (-0.5)).abs() <= 1e-4,
        "bond_dims=[2] should reach -0.5, got {lambda2}"
    );
}

#[test]
fn s5b_m5_mals_inner_tol_and_bonddims_guard() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b14);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = MalsOptions {
        eps: 1e-3,
        bond_dims: vec![0],
        ..Default::default()
    };
    let (lambda, _x) = mals(&h, &x0, &opts);
    assert!(
        (lambda - (-2.0)).abs() <= 1e-3,
        "mals loose-eps energy {lambda}"
    );
}

#[test]
#[should_panic(expected = "bond_dims")]
fn s5b_m5b_mals_empty_bonddims_panics() {
    let space = Su2Space::new(6, 2, 0);
    let h = ha_su2(space, 0, 1);
    let mut rng = seeded(0x5b14);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let opts = MalsOptions {
        bond_dims: vec![],
        ..Default::default()
    };
    let _ = mals(&h, &x0, &opts);
}

#[test]
fn s5b_m6_mals_hb_shrink_direction() {
    let space = Su2Space::new(6, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0x5b15);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 5, 10, &mut rng);
    let opts = MalsOptions {
        eps: 1e-6,
        bond_dims: vec![0],
        ..Default::default()
    };
    let (lambda, y) = mals(&h, &x0, &opts);
    assert!(y.check());
    assert!(
        (lambda - (-0.5)).abs() <= 1e-4,
        "mals HB shrink energy {lambda}"
    );
    let resid = global_eigen_residual(&h, &y, lambda);
    assert!(resid <= 1e-4, "mals HB shrink residual {resid}");
    let caps = qntt::su2::qn::max_ranks(&space);
    for (b, cap_grid) in caps.iter().enumerate() {
        for s in space.bond_sectors(b) {
            assert!(
                y.rank(b, s) <= cap_grid[s],
                "bond {b} sector {s:?} exceeds pathcount cap"
            );
        }
    }
}

#[test]
fn s5b_m7_single_site_guard_coverage() {
    let space = Su2Space::new(1, 1, 1);
    assert_eq!(space.d, 1, "M7 must exercise the d==1 branch");
    let h = ha_su2(space, 0, 0);
    let mut rng = seeded(0x5b17);

    let x0m: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
    let (lm, ym) = mals(
        &h,
        &x0m,
        &MalsOptions {
            eps: 1e-6,
            bond_dims: vec![0],
            ..Default::default()
        },
    );
    assert!(lm.is_finite(), "mals d==1 returned non-finite λ={lm}");
    assert!(ym.check(), "mals d==1 result fails check()");

    let x0t: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
    let (lt, yt) = tdvp(
        &h,
        &x0t,
        &TdvpOptions {
            dtau: 1.0,
            eps: 1e-6,
            max_sweeps: 5,
            ..Default::default()
        },
    );
    assert!(lt.is_finite(), "tdvp d==1 returned non-finite λ={lt}");
    assert!(yt.check(), "tdvp d==1 result fails check()");
}
