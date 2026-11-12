// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg5::util::*;
use qntt::fcidump::{TwoBodySymmetry, read_fcidump_with_point_group};
use qntt::hamiltonian::Construction;
use qntt::integrals::load_fcidump;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{CoeffTensor, Irrep, PointGroupKind, SparseHamiltonian, TtVector};
use qntt::{IntegralsOptions, PointGroupPolicy};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn pg5_c2v_fcidump_roundtrip_to_als_energy() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
    let pg = c2v_pg(d);
    let orbsym_molpro = c2v_orbsym_molpro(d);
    let isym_molpro = PointGroupKind::C2v.molpro_from_irrep(target).unwrap();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(70_001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);

    let path = temp_fcidump_path("c2v_roundtrip");
    write_c2v_fcidump(
        &path,
        d,
        nup + ndn,
        nup as i64 - ndn as i64,
        &orbsym_molpro,
        isym_molpro,
        &t,
        &v,
    );

    let (data, pgdata) = read_fcidump_with_point_group::<f64>(&path, TwoBodySymmetry::None, None)
        .unwrap_or_else(|e| panic!("read synthetic C2v fcidump: {e}"));
    let _ = std::fs::remove_file(&path);

    assert_eq!(pgdata.kind, PointGroupKind::C2v);
    assert!(
        pgdata.guessed,
        "order 4 with no explicit kind must be the guessed default"
    );
    assert_eq!(pgdata.target, target);

    let t_diff: f64 = data
        .t
        .data
        .iter()
        .zip(&t.data)
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let v_diff: f64 = data
        .v
        .data
        .iter()
        .zip(&v.data)
        .map(|(&a, &b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(t_diff < 1e-12, "t round-trip Frobenius diff {t_diff}");
    assert!(v_diff < 1e-12, "v round-trip Frobenius diff {v_diff}");

    let space = data
        .qn_space(None)
        .unwrap_or_else(|e| panic!("qn_space: {e}"));
    let h: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space,
        &data.t,
        &data.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );

    let (_dets, lambda_exact) = dense_target_block_ground_state(&pg, d, &t, &v, nup, ndn, target);

    let x0: TtVector<f64> = capped_randn(space, 1, 3, 70_002);
    let (lam_mals, x_mals) = mals(
        &h,
        &x0,
        &MalsOptions {
            eps: 1e-10,
            bond_dims: vec![64],
            max_iter: 30,
            ..Default::default()
        },
    );
    assert!(x_mals.check());
    assert_purity(&pg, d, nup, ndn, target, &x_mals.to_dense());
    assert!(
        (lam_mals - lambda_exact).abs() <= 1e-8 * lambda_exact.abs().max(1.0),
        "MALS must reach the dense target-block ground state through the FCIDUMP path: got \
         {lam_mals}, want {lambda_exact}"
    );

    let (lam_als, x_als) = als(
        &h,
        &x0,
        &AlsOptions {
            eps: 1e-10,
            max_iter: 60,
            ..Default::default()
        },
    );
    assert!(x_als.check());
    assert_purity(&pg, d, nup, ndn, target, &x_als.to_dense());
    assert!(
        (lam_als - lambda_exact).abs() <= 1e-8 * lambda_exact.abs().max(1.0),
        "ALS must ALSO reach the exact ground state at this size: got {lam_als}, want {lambda_exact}"
    );

    let want_yd = dense_apply_hamiltonian_12(d, &t, &v, &x_mals.to_dense());
    let got_yd = h.apply(&x_mals).to_dense();
    assert!(
        rel_diff(&got_yd, &want_yd) < 1e-9,
        "H·x through the FCIDUMP-built Hamiltonian must match the independent dense JW oracle"
    );
}

#[test]
fn pg5_symmetry_forbidden_entries_are_screened_not_uncaught() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let orbsym_molpro = c2v_orbsym_molpro(d);

    let build_and_check = |amp: f64, tag: &str| -> Result<(), String> {
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for i in 0..d {
            t.set(&[i, i], 1.0 + i as f64);
        }
        t.set(&[0, 1], amp);
        t.set(&[1, 0], amp);
        let v = CoeffTensor::<f64>::zeros(d, 4);

        let path = temp_fcidump_path(&format!("o1_forbidden_{tag}"));
        write_c2v_fcidump(
            &path,
            d,
            nup + ndn,
            nup as i64 - ndn as i64,
            &orbsym_molpro,
            1,
            &t,
            &v,
        );
        let opts = IntegralsOptions {
            symmetry: TwoBodySymmetry::Pairswap,
            point_group: PointGroupPolicy::Resolve(None),
            screen: true,
            ..IntegralsOptions::default()
        };
        let result = SparseHamiltonian::<f64>::from_fcidump(&path, None, &opts);
        let _ = std::fs::remove_file(&path);

        match result {
            Ok((h, load)) => {
                assert_eq!(
                    load.ints.t.get(&[0, 1]),
                    0.0,
                    "amp={amp}: the forbidden entry must be screened to exactly zero, not left \
                     to survive at a sub-eps magnitude"
                );
                let space = h.space;
                let x: TtVector<f64> = capped_randn(space, 1, 3, 999_003);
                let y = h.apply(&x);
                assert!(y.to_dense().iter().any(|v| v.is_finite()));
                Ok(())
            }
            Err(e) => Err(e.to_string()),
        }
    };

    build_and_check(0.0, "control").expect("amp=0 must never be rejected");

    build_and_check(1e-15, "noise")
        .expect("a noise-scale forbidden entry must be screened, not rejected or left to panic");

    let err = build_and_check(1e-3, "real")
        .expect_err("a 1e-3 forbidden entry is not noise and must be rejected");
    assert!(err.contains("symmetry"), "got {err:?}");
}

#[test]
fn pg5_explicit_kind_vs_guess_same_energy() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
    let pg = c2v_pg(d);
    let orbsym_molpro = c2v_orbsym_molpro(d);
    let isym_molpro = PointGroupKind::C2v.molpro_from_irrep(target).unwrap();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(70_010);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);

    let path = temp_fcidump_path("c2v_vs_d2");
    write_c2v_fcidump(
        &path,
        d,
        nup + ndn,
        nup as i64 - ndn as i64,
        &orbsym_molpro,
        isym_molpro,
        &t,
        &v,
    );

    let (data_guess, pg_guess) =
        read_fcidump_with_point_group::<f64>(&path, TwoBodySymmetry::None, None).unwrap();
    let (data_d2, pg_d2) = read_fcidump_with_point_group::<f64>(
        &path,
        TwoBodySymmetry::None,
        Some(PointGroupKind::D2),
    )
    .unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(pg_guess.kind, PointGroupKind::C2v);
    assert_eq!(pg_d2.kind, PointGroupKind::D2);
    assert_ne!(
        pg_guess.kind.irrep_name(pg_guess.target).unwrap(),
        pg_d2.kind.irrep_name(pg_d2.target).unwrap(),
        "the two same-order kinds must name the target irrep differently"
    );

    let space_guess = data_guess.qn_space(None).unwrap();
    let space_d2 = data_d2.qn_space(Some(PointGroupKind::D2)).unwrap();

    let h_guess: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_guess,
        &data_guess.t,
        &data_guess.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );
    let h_d2: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_d2,
        &data_d2.t,
        &data_d2.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );

    let x0_guess: TtVector<f64> = capped_randn(space_guess, 1, 3, 70_011);
    let x0_d2: TtVector<f64> = capped_randn(space_d2, 1, 3, 70_011);

    let opts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![64],
        max_iter: 30,
        ..Default::default()
    };
    let (lam_guess, _) = mals(&h_guess, &x0_guess, &opts);
    let (lam_d2, _) = mals(&h_d2, &x0_d2, &opts);

    assert!(
        (lam_guess - lam_d2).abs() <= 1e-10 * lam_guess.abs().max(1.0),
        "the guessed (C2v) and explicit (D2) loads of the SAME file must give the SAME energy: \
         {lam_guess} vs {lam_d2}"
    );
}

#[test]
fn pg5_c2v_isym_selects_a_different_block() {
    let d = 6usize;
    let (nup, ndn) = (2usize, 2usize);
    let pg = c2v_pg(d);
    let orbsym_molpro = c2v_orbsym_molpro(d);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(70_020);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);

    let target_a = Irrep::IDENTITY;
    let target_b = Irrep::new(1);
    let isym_a = 0u8;
    let isym_b = PointGroupKind::C2v.molpro_from_irrep(target_b).unwrap();

    let path_a = temp_fcidump_path("c2v_isym_a");
    let path_b = temp_fcidump_path("c2v_isym_b");
    write_c2v_fcidump(
        &path_a,
        d,
        nup + ndn,
        nup as i64 - ndn as i64,
        &orbsym_molpro,
        isym_a,
        &t,
        &v,
    );
    write_c2v_fcidump(
        &path_b,
        d,
        nup + ndn,
        nup as i64 - ndn as i64,
        &orbsym_molpro,
        isym_b,
        &t,
        &v,
    );

    let (data_a, pgdata_a) =
        read_fcidump_with_point_group::<f64>(&path_a, TwoBodySymmetry::None, None).unwrap();
    let (data_b, pgdata_b) =
        read_fcidump_with_point_group::<f64>(&path_b, TwoBodySymmetry::None, None).unwrap();
    let _ = std::fs::remove_file(&path_a);
    let _ = std::fs::remove_file(&path_b);

    assert_eq!(pgdata_a.target, target_a);
    assert_eq!(pgdata_b.target, target_b);

    let (_dets_a, lambda_exact_a) =
        dense_target_block_ground_state(&pg, d, &t, &v, nup, ndn, target_a);
    let (_dets_b, lambda_exact_b) =
        dense_target_block_ground_state(&pg, d, &t, &v, nup, ndn, target_b);
    assert!(
        (lambda_exact_a - lambda_exact_b).abs() > 1e-6,
        "the fixture must actually distinguish the two target blocks: {lambda_exact_a} vs \
         {lambda_exact_b}"
    );

    let space_a = data_a.qn_space(None).unwrap();
    let space_b = data_b.qn_space(None).unwrap();
    let h_a: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_a,
        &data_a.t,
        &data_a.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );
    let h_b: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_b,
        &data_b.t,
        &data_b.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );

    let opts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![64],
        max_iter: 30,
        ..Default::default()
    };
    let x0_a: TtVector<f64> = capped_randn(space_a, 1, 3, 70_021);
    let x0_b: TtVector<f64> = capped_randn(space_b, 1, 3, 70_022);
    let (lam_a, xa) = mals(&h_a, &x0_a, &opts);
    let (lam_b, xb) = mals(&h_b, &x0_b, &opts);

    assert_purity(&pg, d, nup, ndn, target_a, &xa.to_dense());
    assert_purity(&pg, d, nup, ndn, target_b, &xb.to_dense());
    assert!(
        (lam_a - lambda_exact_a).abs() <= 1e-8 * lambda_exact_a.abs().max(1.0),
        "ISYM=0 (identity target): got {lam_a}, want {lambda_exact_a}"
    );
    assert!(
        (lam_b - lambda_exact_b).abs() <= 1e-8 * lambda_exact_b.abs().max(1.0),
        "ISYM={isym_b} (target irrep 1): got {lam_b}, want {lambda_exact_b}"
    );
}

#[test]
fn pg5_load_pairswap_with_point_group_c2v() {
    let d = 4usize;
    let (nup, ndn) = (2usize, 2usize);
    let target = Irrep::new(2);
    let pg = c2v_pg(d);
    let orbsym_molpro = c2v_orbsym_molpro(d);
    let isym_molpro = PointGroupKind::C2v.molpro_from_irrep(target).unwrap();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(70_030);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);

    let path = temp_fcidump_path("c2v_transcorr");
    write_c2v_fcidump(
        &path,
        d,
        nup + ndn,
        nup as i64 - ndn as i64,
        &orbsym_molpro,
        isym_molpro,
        &t,
        &v,
    );

    let opts = IntegralsOptions {
        symmetry: TwoBodySymmetry::Pairswap,
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let load = load_fcidump::<f64>(&path, None, &opts)
        .unwrap_or_else(|e| panic!("load_fcidump (Pairswap, Resolve): {e}"));
    let _ = std::fs::remove_file(&path);

    let pgdata = load.pg;
    assert_eq!(pgdata.kind, PointGroupKind::C2v);
    assert_eq!(pgdata.target, target);
    assert_eq!((load.space.nup, load.space.ndn), (nup, ndn));
    assert!(load.w.is_none(), "no TCDUMP was given");

    let space = load.ints.qn_space(None).unwrap();
    assert_eq!(
        space, load.space,
        "the resolved space and the raw header's must agree"
    );
    assert_eq!((space.nup, space.ndn), (nup, ndn));
    assert_eq!(space.target, target);
}
