// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::core::weingarten::FloorPolicy;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::su2::core::Su2Core;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hess::{hessian_grams, symplectic_eigenvalues, weingarten_operator};
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{RiemannianOpts, riemannian_cg, tangent_bases};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::tensorops::{a_i, adag_i, adag_i_a_j, n_i, pair_i, s_i};
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::su2::weingarten::{bond_gauges, tangent_basis, tangent_dim};
use qntt::{Irrep, PointGroup, QnSpace};

use rand::RngExt;
use rand_distr::StandardNormal;

use qntt::solvers::riemannian::{
    RiemannianOpts as AbRiemannianOpts, riemannian_cg as ab_riemannian_cg,
};

use crate::spg5::util::*;

#[test]
fn spg5_y9_riemannian_cg_cross_symmetry_c2v() {
    let d = 6;
    let pg = c2v_table(d);
    let (i, j) = find_same_nonidentity_irrep_pair(&pg, d);
    assert_ne!(
        pg.irrep_of(i),
        Irrep::IDENTITY,
        "fixture bug: sites must share a NON-identity irrep"
    );
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[i, j], -1.0);
    t.set(&[j, i], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let target = Irrep::IDENTITY;

    let su2_space = Su2Space::with_point_group(d, 2, 0, pg, target);
    let h_su2 = Su2SparseHamiltonian::new(su2_space, &t, &v, None, 1e-10);
    let x0_su2 = build_pg_state::<f64>(d, 2, 0, pg, target, 3, 0xC5_0901);
    let mut opts = RiemannianOpts::new(&su2_space);
    opts.eps = 1e-10;
    opts.max_iter = 200;
    opts.gradtol = 1e-10;
    let (lambda_su2, x_su2) = riemannian_cg(&h_su2, &x0_su2, &opts);
    assert!(x_su2.check());
    assert_expand_purity_chain(&x_su2, &pg);
    assert!(
        (lambda_su2 - (-2.0)).abs() <= 1e-6,
        "su2 riemannian_cg must reach the analytic singlet ground -2.0: got {lambda_su2}"
    );

    let ab_space = QnSpace::with_point_group(d, 1, 1, pg, target);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-10);
    let x0_ab = capped_randn_ab::<f64>(ab_space, 1, 3, 0xC5_0902);
    let mut opts_ab = AbRiemannianOpts::new(&ab_space);
    opts_ab.eps = 1e-10;
    opts_ab.max_iter = 200;
    opts_ab.gradtol = 1e-10;
    let (lambda_ab, x_ab) = ab_riemannian_cg(&h_ab, &x0_ab, &opts_ab);
    assert!(x_ab.check());
    assert!(
        (lambda_ab - (-2.0)).abs() <= 1e-6,
        "abelian riemannian_cg must reach the analytic singlet ground -2.0: got {lambda_ab}"
    );

    assert!(
        (lambda_su2 - lambda_ab).abs() <= 1e-6,
        "cross-symmetry mismatch: su2={lambda_su2} vs abelian={lambda_ab}"
    );
}

#[test]
fn spg5_y10_apply_site_tensor_propagates_the_point_group_and_shifts_the_target() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (2usize, 0usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or(Irrep::IDENTITY);
    let i = (0..d)
        .find(|&k| pg.irrep_of(k) != Irrep::IDENTITY)
        .expect("fixture bug: every site totally symmetric");
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0xC5_1001);

    let ya = adag_i(&x, i, two_s as i64 + 1);
    assert_eq!(ya.space.pg, pg, "adag_i dropped the point group");
    assert_eq!(
        ya.space.target,
        target.fuse(pg.irrep_of(i)),
        "adag_i did not shift the target by the site irrep"
    );
    assert!(ya.check());
    assert_expand_purity_chain(&ya, &pg);
    assert!(ya.norm() > 1e-8, "adag_i vacuous");

    let yb = a_i(&x, i, two_s as i64 + 1);
    assert_eq!(yb.space.pg, pg, "a_i dropped the point group");
    assert_eq!(
        yb.space.target,
        target.fuse(pg.irrep_of(i)),
        "a_i did not shift the target by the site irrep"
    );
    assert!(yb.check());
    assert_expand_purity_chain(&yb, &pg);
    assert!(yb.norm() > 1e-8, "a_i vacuous");

    let ys = s_i(&x, i, two_s as i64 + 2);
    assert_eq!(ys.space.pg, pg, "s_i dropped the point group");
    assert_eq!(
        ys.space.target, target,
        "s_i must not shift the target (Γ=IDENTITY)"
    );
    assert!(ys.check());
    assert_expand_purity_chain(&ys, &pg);
    assert!(ys.norm() > 1e-8, "s_i vacuous");

    let yp = pair_i(&x, i);
    assert_eq!(yp.space.pg, pg, "pair_i dropped the point group");
    assert_eq!(
        yp.space.target, target,
        "pair_i must not shift the target (Γ=IDENTITY)"
    );
    assert!(yp.check());
    assert_expand_purity_chain(&yp, &pg);
    assert!(yp.norm() > 1e-8, "pair_i vacuous");

    assert!(pg.order() > 1);
    assert_ne!(pg.irrep_of(i), Irrep::IDENTITY);
}

#[test]
#[should_panic(expected = "different irreps")]
fn spg5_y11_adag_i_a_j_rejects_mixed_irrep_sites() {
    let d = 6;
    let pg = c2v_table(d);
    let (i, j) = find_different_irrep_pair(&pg, d);
    let (n, two_s) = (2usize, 0usize);
    let target = Irrep::IDENTITY;
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0xC5_1101);
    let _ = adag_i_a_j(&x, i, j);
}

#[test]
fn spg5_y12_trivial_group_stack_is_bit_identical() {
    let d = 4usize;
    let (n, two_s) = (3usize, 1usize);
    let space_new = Su2Space::new(d, n, two_s);
    let space_pg = Su2Space::with_point_group(d, n, two_s, PointGroup::TRIVIAL, Irrep::IDENTITY);
    assert_eq!(space_new, space_pg);

    let ranks = su2_capped_ranks(&space_new, 2);
    let mut rng_x = seeded(0xC5_1201);
    let x_new: Su2TtVector<f64> = su2_tt_randn(space_new, &ranks, &mut rng_x);
    let cores_pg: Vec<Su2Core<f64>> = x_new
        .cores
        .iter()
        .cloned()
        .map(|c| retag_core(c, space_pg))
        .collect();
    let x_pg = Su2TtVector::new(space_pg, cores_pg);

    let (bl_new, br_new) = tangent_bases(&x_new);
    let (bl_pg, br_pg) = tangent_bases(&x_pg);

    let mut rng_dir = seeded(0xC5_1202);
    let dir_new: Su2TtVector<f64> = randomized_su2_state(space_new, 1, 2, &mut rng_dir);
    let cores_dir_pg: Vec<Su2Core<f64>> = dir_new
        .cores
        .iter()
        .cloned()
        .map(|c| retag_core(c, space_pg))
        .collect();
    let dir_pg = Su2TtVector::new(space_pg, cores_dir_pg);

    let xi_new = Su2TtTangent::project(&bl_new, &br_new, &dir_new);
    let xi_pg = Su2TtTangent::project(&bl_pg, &br_pg, &dir_pg);
    for k in 0..d {
        assert_su2core_exact_eq(
            xi_new.component(k),
            xi_pg.component(k),
            &format!("project component {k}"),
        );
    }

    let v_new = xi_new.to_vector();
    let v_pg = xi_pg.to_vector();
    assert_su2tt_exact_eq(&v_new, &v_pg, "to_vector");

    let r_new = xi_new.retract_hosvd(0.3);
    let r_pg = xi_pg.retract_hosvd(0.3);
    assert_su2tt_exact_eq(&r_new, &r_pg, "retract_hosvd(0.3)");

    assert_eq!(tangent_dim(&bl_new), tangent_dim(&bl_pg));
    let basis_new = tangent_basis(&bl_new, &br_new, true);
    let basis_pg = tangent_basis(&bl_pg, &br_pg, true);
    assert_eq!(basis_new.len(), basis_pg.len());

    let gauges_new = bond_gauges(&bl_new, &br_new, FloorPolicy::default_hard());
    let gauges_pg = bond_gauges(&bl_pg, &br_pg, FloorPolicy::default_hard());
    assert_eq!(gauges_new.len(), gauges_pg.len());
    for (gn, gp) in gauges_new.iter().zip(gauges_pg.iter()) {
        assert_eq!(
            gn.sigma, gp.sigma,
            "bond gauge sigma mismatch at bond {}",
            gn.k
        );
    }

    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let mut rng_t = seeded(0xC5_1203);
    for p in 0..d {
        for q in 0..d {
            let val = rng_t.sample::<f64, _>(StandardNormal);
            t.set(&[p, q], val);
            t.set(&[q, p], val);
        }
    }
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let h_new = Su2SparseHamiltonian::new(space_new, &t, &v, None, f64::EPSILON);
    let h_pg = Su2SparseHamiltonian::new(space_pg, &t, &v, None, f64::EPSILON);
    let ctx_new = weingarten_operator(&h_new, &x_new);
    let ctx_pg = weingarten_operator(&h_pg, &x_pg);

    let (a_new, b_new) = hessian_grams(&ctx_new, &basis_new);
    let (a_pg, b_pg) = hessian_grams(&ctx_pg, &basis_pg);
    assert_eq!(a_new, a_pg, "hessian_grams A not bit-identical");
    assert_eq!(b_new, b_pg, "hessian_grams B not bit-identical");

    let bapp_new = ctx_new.b_apply(&xi_new);
    let bapp_pg = ctx_pg.b_apply(&xi_pg);
    for k in 0..d {
        assert_su2core_exact_eq(
            bapp_new.component(k),
            bapp_pg.component(k),
            &format!("weingarten_assemble b_apply component {k}"),
        );
    }

    let spec_new = symplectic_eigenvalues(&ctx_new, &basis_new);
    let spec_pg = symplectic_eigenvalues(&ctx_pg, &basis_pg);
    assert_eq!(
        format!("{spec_new:?}"),
        format!("{spec_pg:?}"),
        "symplectic_eigenvalues not bit-identical"
    );

    let i_site = 0usize;
    assert_su2tt_exact_eq(&n_i(&x_new, i_site), &n_i(&x_pg, i_site), "n_i");
    assert_su2tt_exact_eq(
        &adag_i_a_j(&x_new, 0, 1),
        &adag_i_a_j(&x_pg, 0, 1),
        "adag_i_a_j",
    );
    let two_s_out_up = two_s as i64 + 1;
    assert_su2tt_exact_eq(
        &adag_i(&x_new, i_site, two_s_out_up),
        &adag_i(&x_pg, i_site, two_s_out_up),
        "adag_i",
    );
    let two_s_out_dn = (two_s as i64 - 1).max(0);
    assert_su2tt_exact_eq(
        &a_i(&x_new, i_site, two_s_out_dn),
        &a_i(&x_pg, i_site, two_s_out_dn),
        "a_i",
    );
    assert_su2tt_exact_eq(
        &s_i(&x_new, i_site, two_s as i64),
        &s_i(&x_pg, i_site, two_s as i64),
        "s_i",
    );
    assert_su2tt_exact_eq(&pair_i(&x_new, i_site), &pair_i(&x_pg, i_site), "pair_i");
}
