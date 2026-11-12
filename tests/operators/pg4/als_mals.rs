// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg4::util::*;
use qntt::hamiltonian::Construction;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{SparseHamiltonian, TtVector};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn pg4_mals_and_als_ground_state_gb() {
    let (pg, space) = g_b();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(60_001);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);
    let (_dets, lambda_exact) =
        dense_target_block_ground_state(&pg, space.d, &t, &v, space.nup, space.ndn, space.target);

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    let x0: TtVector<f64> = capped_randn(space, 1, 3, 60_002);

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
    assert_purity(
        &pg,
        space.d,
        space.nup,
        space.ndn,
        space.target,
        &x_mals.to_dense(),
    );
    assert!(
        (lam_mals - lambda_exact).abs() <= 1e-8 * lambda_exact.abs().max(1.0),
        "MALS on G-B must reach the dense target-block ground state: got {lam_mals}, want \
         {lambda_exact}"
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
    assert_purity(
        &pg,
        space.d,
        space.nup,
        space.ndn,
        space.target,
        &x_als.to_dense(),
    );
    assert!(
        (lam_als - lambda_exact).abs() <= 1e-8 * lambda_exact.abs().max(1.0),
        "ALS on the small exhaustive G-B must ALSO reach the exact ground state: got {lam_als}, \
         want {lambda_exact}"
    );
}

#[test]
fn pg4_mals_ground_state_and_als_floor_ga() {
    let (pg, space) = g_a();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(60_010);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);
    let (_dets, lambda_exact) =
        dense_target_block_ground_state(&pg, space.d, &t, &v, space.nup, space.ndn, space.target);

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    let x0: TtVector<f64> = capped_randn(space, 1, 3, 60_011);

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
    assert_purity(
        &pg,
        space.d,
        space.nup,
        space.ndn,
        space.target,
        &x_mals.to_dense(),
    );
    assert!(
        (lam_mals - lambda_exact).abs() <= 1e-8 * lambda_exact.abs().max(1.0),
        "MALS on G-A must reach the dense target-block ground state: got {lam_mals}, want \
         {lambda_exact}"
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
    assert_purity(
        &pg,
        space.d,
        space.nup,
        space.ndn,
        space.target,
        &x_als.to_dense(),
    );
    assert!(
        lam_als >= lambda_exact - 1e-9,
        "ALS variational floor violated: lam_als={lam_als} < lambda_exact={lambda_exact} — the \
         operator is leaking outside the target-irrep block"
    );
}
