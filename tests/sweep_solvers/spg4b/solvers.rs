// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::solvers::mals::NoiseType;
use qntt::su2::core::Su2Core;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::mals::{MalsOptions, mals};
use qntt::su2::solvers::tdvp::{TdvpOptions, tdvp};
use qntt::su2::twosite::{contract, factor_qc, factor_svd, factor_svd_noise};
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::{Irrep, PointGroup, QnSpace};

use rand::RngExt;
use rand_distr::StandardNormal;

use crate::spg4b::util::*;

#[test]
fn spg4b_x9_mals_cross_symmetry_c2v_identity_target() {
    let d = 6;
    let pg = c2v_table(d);
    let (i, j) = find_same_nonidentity_irrep_pair(&pg, d);
    assert_ne!(
        pg.irrep_of(i),
        Irrep::IDENTITY,
        "fixture bug: sites must share a NON-identity irrep, else Δ is a no-op along the chain"
    );
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[i, j], -1.0);
    t.set(&[j, i], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let target = Irrep::IDENTITY;

    let su2_space = Su2Space::with_point_group(d, 2, 0, pg, target);
    let h_su2 = Su2SparseHamiltonian::new(su2_space, &t, &v, None, 1e-12);
    let x0_su2 = build_pg_state::<f64>(d, 2, 0, pg, target, 3, 0xB9_0001);
    let (lambda_su2, x_su2) = mals(
        &h_su2,
        &x0_su2,
        &MalsOptions {
            eps: 1e-9,
            bond_dims: vec![4],
            noise: vec![0.0],
            max_iter: 5,
            ..Default::default()
        },
    );
    assert!(x_su2.check());
    assert_expand_purity_chain(&x_su2, &pg);
    assert!(
        (lambda_su2 - (-2.0)).abs() <= 1e-6,
        "su2 mals must reach the analytic singlet ground -2.0: got {lambda_su2}"
    );

    let ab_space = QnSpace::with_point_group(d, 1, 1, pg, target);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-12);
    let x0_ab = capped_randn_ab::<f64>(ab_space, 1, 3, 0xB9_0002);
    let (lambda_ab, x_ab) = qntt::solvers::mals::mals(
        &h_ab,
        &x0_ab,
        &qntt::solvers::mals::MalsOptions {
            eps: 1e-9,
            bond_dims: vec![4],
            noise: vec![0.0],
            max_iter: 5,
            ..Default::default()
        },
    );
    assert!(x_ab.check());
    assert!(
        (lambda_ab - (-2.0)).abs() <= 1e-6,
        "abelian mals must reach the analytic singlet ground -2.0: got {lambda_ab}"
    );

    assert!(
        (lambda_su2 - lambda_ab).abs() <= 1e-6,
        "cross-symmetry mismatch: su2={lambda_su2} vs abelian={lambda_ab}"
    );
}

#[test]
fn spg4b_x10_mals_with_noise_stays_pure_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (i, j) = find_same_nonidentity_irrep_pair(&pg, d);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[i, j], -1.0);
    t.set(&[j, i], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let target = Irrep::IDENTITY;
    let space = Su2Space::with_point_group(d, 2, 0, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);

    for (idx, nt) in [NoiseType::Random, NoiseType::Perturbative]
        .into_iter()
        .enumerate()
    {
        let x0 = build_pg_state::<f64>(d, 2, 0, pg, target, 3, 0xB10_0001 + idx as u64);
        let opts = MalsOptions {
            eps: 1e-9,
            bond_dims: vec![4],
            noise: vec![1e-3, 1e-3, 0.0],
            noise_type: nt,
            max_iter: 3,
            ..Default::default()
        };
        let (lambda, x) = mals(&h, &x0, &opts);
        assert!(x.check(), "mals with noise ({nt:?}) failed check()");
        assert_expand_purity_chain(&x, &pg);
        assert!(
            lambda.is_finite(),
            "mals with noise ({nt:?}) returned non-finite λ={lambda}"
        );
        assert!(
            (lambda + 2.0).abs() <= 1e-4,
            "mals with noise ({nt:?}) energy {lambda} not near the exact singlet ground -2.0"
        );
    }
}

#[test]
fn spg4b_x11_tdvp_cross_symmetry_at_nontrivial_group() {
    let d = 4;
    let pg = c2v_table(d);
    let (n, two_s) = (2, 0);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or(Irrep::IDENTITY);
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng = seeded(0xB11_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let x0 = build_pg_state::<f64>(d, n, two_s, pg, target, 4, 0xB11_0002);

    let mut last_lambda = f64::INFINITY;
    for sweeps in 1..=3usize {
        let opts = TdvpOptions {
            dtau: 5.0,
            eps: 0.0,
            max_sweeps: sweeps,
            chi: qntt::Pooled(4),
            trunc: 1e-10,
            exptol: 1e-12,
            ..Default::default()
        };
        let (lambda, x) = tdvp(&h, &x0, &opts);
        assert!(x.check(), "tdvp sweeps={sweeps} failed check()");
        assert_expand_purity_chain(&x, &pg);
        assert!(
            lambda <= last_lambda + 1e-10,
            "tdvp energy not monotone decreasing: sweeps={sweeps} lambda={lambda} vs previous \
             {last_lambda}"
        );
        last_lambda = lambda;
    }

    let (lambda_mals, x_mals) = mals(
        &h,
        &x0,
        &MalsOptions {
            eps: 1e-9,
            bond_dims: vec![4],
            noise: vec![0.0],
            max_iter: 8,
            ..Default::default()
        },
    );
    assert!(x_mals.check());
    assert!(
        (last_lambda - lambda_mals).abs() <= 1e-4,
        "tdvp converged λ={last_lambda} vs su2 mals λ={lambda_mals} on the same H"
    );

    let ab_space = QnSpace::with_point_group(d, 1, 1, pg, target);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-12);
    let x0_ab = capped_randn_ab::<f64>(ab_space, 1, 3, 0xB11_0003);
    let (lambda_ab, x_ab) = qntt::solvers::mals::mals(
        &h_ab,
        &x0_ab,
        &qntt::solvers::mals::MalsOptions {
            eps: 1e-9,
            bond_dims: vec![8],
            noise: vec![0.0],
            max_iter: 12,
            ..Default::default()
        },
    );
    assert!(x_ab.check());
    assert!(
        (last_lambda - lambda_ab).abs() <= 1e-4,
        "tdvp converged λ={last_lambda} vs independently certified abelian layer λ={lambda_ab}"
    );
}

#[test]
fn spg4b_x12_trivial_group_two_site_path_is_bit_identical() {
    let d = 6;
    let (n, two_s) = (4, 2);
    let space_new = Su2Space::new(d, n, two_s);
    let space_pg = Su2Space::with_point_group(d, n, two_s, PointGroup::TRIVIAL, Irrep::IDENTITY);
    assert_eq!(space_new, space_pg);

    let ranks = su2_capped_ranks(&space_new, 3);
    let mut rng = seeded(0xB12_0001);
    let x_new: Su2TtVector<f64> = su2_tt_randn(space_new, &ranks, &mut rng);
    let cores_pg: Vec<Su2Core<f64>> = x_new
        .cores
        .iter()
        .cloned()
        .map(|c| retag_core(c, space_pg))
        .collect();
    let x_pg = Su2TtVector::new(space_pg, cores_pg);

    let k = 2usize;
    let cp_new = contract(x_new.core(k), x_new.core(k + 1));
    let cp_pg = contract(x_pg.core(k), x_pg.core(k + 1));
    assert_su2contracted_exact_eq(&cp_new, &cp_pg, "contract");

    let (qk_new, ck_new) = factor_qc(&cp_new);
    let (qk_pg, ck_pg) = factor_qc(&cp_pg);
    assert_su2core_exact_eq(&qk_new, &qk_pg, "factor_qc Q");
    assert_su2core_exact_eq(&ck_new, &ck_pg, "factor_qc C");

    let (u_new, s_new, vt_new) = factor_svd(&cp_new, 0.0, qntt::Pooled::NONE, false);
    let (u_pg, s_pg, vt_pg) = factor_svd(&cp_pg, 0.0, qntt::Pooled::NONE, false);
    assert_su2core_exact_eq(&u_new, &u_pg, "factor_svd U");
    assert_su2core_exact_eq(&vt_new, &vt_pg, "factor_svd Vt");
    for m in space_new.bond_sectors(k + 1) {
        assert_eq!(
            s_new.row_ranks()[m],
            s_pg.row_ranks()[m],
            "factor_svd S rank {m:?}"
        );
    }

    let mut rng_noise_a = seeded(0xB12_0002);
    let mut rng_noise_b = seeded(0xB12_0002);
    let (un_new, _sn_new, vn_new, _i1) = factor_svd_noise(
        &cp_new,
        0.0,
        qntt::Pooled::NONE,
        false,
        1e-2,
        &mut rng_noise_a,
    );
    let (un_pg, _sn_pg, vn_pg, _i2) = factor_svd_noise(
        &cp_pg,
        0.0,
        qntt::Pooled::NONE,
        false,
        1e-2,
        &mut rng_noise_b,
    );
    assert_su2core_exact_eq(&un_new, &un_pg, "factor_svd_noise U");
    assert_su2core_exact_eq(&vn_new, &vn_pg, "factor_svd_noise Vt");

    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let mut rng_t = seeded(0xB12_0003);
    for i in 0..d {
        for j in 0..d {
            let val = rng_t.sample::<f64, _>(StandardNormal);
            t.set(&[i, j], val);
            t.set(&[j, i], val);
        }
    }
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let h_new = Su2SparseHamiltonian::new(space_new, &t, &v, None, f64::EPSILON);
    let h_pg = Su2SparseHamiltonian::new(space_pg, &t, &v, None, f64::EPSILON);
    let lchain_new = left_to_right_chain(&h_new, &x_new);
    let rchain_new = right_to_left_chain(&h_new, &x_new);
    let lchain_pg = left_to_right_chain(&h_pg, &x_pg);
    let rchain_pg = right_to_left_chain(&h_pg, &x_pg);
    let heff_new = h_new.two_site_sketched_hamiltonian(&cp_new, &lchain_new[k], &rchain_new[k + 2]);
    let heff_pg = h_pg.two_site_sketched_hamiltonian(&cp_pg, &lchain_pg[k], &rchain_pg[k + 2]);
    assert_su2contracted_exact_eq(&heff_new, &heff_pg, "two_site_sketched_hamiltonian");

    let opts = MalsOptions {
        eps: 1e-9,
        bond_dims: vec![4],
        noise: vec![0.0],
        max_iter: 2,
        ..Default::default()
    };
    let (lambda_new, xf_new) = mals(&h_new, &x_new, &opts);
    let (lambda_pg, xf_pg) = mals(&h_pg, &x_pg, &opts);
    assert_eq!(lambda_new, lambda_pg, "mals λ not bit-identical");
    for kk in 0..d {
        assert_su2core_exact_eq(
            xf_new.core(kk),
            xf_pg.core(kk),
            &format!("mals result core {kk}"),
        );
    }
}
