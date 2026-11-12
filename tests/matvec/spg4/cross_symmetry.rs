// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh_quotient};
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::{Irrep, PointGroup, ThreeBody};

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::spg4::util::{
    assert_c2v_coeffs_nonvacuous, assert_expand_purity_chain, assert_su2tt_exact_eq,
    build_pg_state, c2v_coeffs, c2v_table, c2v_w, find_nonidentity_reachable_target,
    max_err_over_two_sz_pg, randn_coeffs, su2_capped_ranks,
};

#[test]
fn spg4_t1_trivial_group_bit_identical_to_new() {
    let d = 5;
    let (n, two_s) = (3, 1);
    let space_new = Su2Space::new(d, n, two_s);
    let space_pg = Su2Space::with_point_group(d, n, two_s, PointGroup::TRIVIAL, Irrep::IDENTITY);
    assert_eq!(space_new, space_pg);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xAAAA_1111);
    let (t, v) = randn_coeffs(d, &mut rng);
    let h_new = Su2SparseHamiltonian::new(space_new, &t, &v, None, f64::EPSILON);
    let h_pg = Su2SparseHamiltonian::new(space_pg, &t, &v, None, f64::EPSILON);

    let ranks = su2_capped_ranks(&space_new, 3);
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0xBBBB_2222);
    let x: Su2TtVector<f64> = su2_tt_randn(space_new, &ranks, &mut rng2);

    let y_new = h_new.apply(&x);
    let y_pg = h_pg.apply(&x);
    assert_su2tt_exact_eq(
        &y_new,
        &y_pg,
        "H·x: ::new vs with_point_group(TRIVIAL, IDENTITY)",
    );

    let rq_new = su2_rayleigh_quotient(&h_new, &x);
    let rq_pg = su2_rayleigh_quotient(&h_pg, &x);
    assert_eq!(
        rq_new, rq_pg,
        "rayleigh_quotient must be bit-identical at |G|=1"
    );
}

#[test]
fn spg4_t2_cross_symmetry_1body_identity_target() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (2, 0);
    let target = Irrep::IDENTITY;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1B0D_0001);
    let (t, _v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);
    assert!(
        (0..d).any(|i| (0..d).any(|j| i != j
            && pg.irrep_of(i) == pg.irrep_of(j)
            && pg.irrep_of(i) != Irrep::IDENTITY)),
        "fixture bug: need an allowed off-diagonal pair with a non-identity irrep, else Δ is a \
         no-op along the whole chain"
    );
    let v = CoeffTensor::<f64>::zeros(d, 4);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0x1B0D_0002);
    let y = h.apply(&x);
    assert!(y.check(), "1-body H·x must pass check()");
    assert_expand_purity_chain(&y, &pg);

    let err = max_err_over_two_sz_pg(&x, &t, &v, None, &y, space);
    assert!(
        err <= 1e-11,
        "1-body identity-target cross-symmetry err={err}"
    );
}

#[test]
fn spg4_t2_cross_symmetry_1body_nonidentity_target() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (2, 0);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1B0D_0010);
    let (t, _v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);
    assert!(
        (0..d).any(|i| (0..d).any(|j| i != j
            && pg.irrep_of(i) == pg.irrep_of(j)
            && pg.irrep_of(i) != Irrep::IDENTITY)),
        "fixture bug: need an allowed off-diagonal pair with a non-identity irrep, else Δ is a \
         no-op along the whole chain"
    );
    let v = CoeffTensor::<f64>::zeros(d, 4);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0x1B0D_0011);
    let y = h.apply(&x);
    assert!(y.check(), "1-body H·x must pass check()");
    assert_expand_purity_chain(&y, &pg);

    let err = max_err_over_two_sz_pg(&x, &t, &v, None, &y, space);
    assert!(
        err <= 1e-11,
        "1-body nonidentity-target cross-symmetry err={err}"
    );
}

#[test]
fn spg4_t2_cross_symmetry_2body_identity_target() {
    let d = 5;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = Irrep::IDENTITY;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x2B0D_0001);
    let (_t, v) = c2v_coeffs(&pg, d, &mut rng);
    let t = CoeffTensor::<f64>::zeros(d, 2);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0x2B0D_0002);
    let y = h.apply(&x);
    assert!(y.check(), "2-body H·x must pass check()");
    assert_expand_purity_chain(&y, &pg);

    let err = max_err_over_two_sz_pg(&x, &t, &v, None, &y, space);
    assert!(
        err <= 1e-11,
        "2-body identity-target cross-symmetry err={err}"
    );
}

#[test]
fn spg4_t2_cross_symmetry_2body_nonidentity_target() {
    let d = 5;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x2B0D_0010);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0x2B0D_0011);
    let y = h.apply(&x);
    assert!(y.check(), "2-body H·x must pass check()");
    assert_expand_purity_chain(&y, &pg);

    let err = max_err_over_two_sz_pg(&x, &t, &v, None, &y, space);
    assert!(
        err <= 1e-11,
        "2-body (1+2-body) nonidentity-target cross-symmetry err={err}"
    );
}

#[test]
fn spg4_t2_cross_symmetry_3body_nonidentity_target() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x3B0D_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);
    let w = c2v_w(&pg, d, 10, &mut rng);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, Some(&w as &dyn ThreeBody<f64>), f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0x3B0D_0002);
    let y = h.apply(&x);
    assert!(y.check(), "3-body H·x must pass check()");
    assert_expand_purity_chain(&y, &pg);

    let err = max_err_over_two_sz_pg(&x, &t, &v, Some(&w as &dyn ThreeBody<f64>), &y, space);
    assert!(
        err <= 1e-10,
        "3-body nonidentity-target cross-symmetry err={err}"
    );
}

#[test]
fn spg4_t4_irrep_axis_nonvacuity_operator_band_expands_nonidentity_plane() {
    let d = 5;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4B0D_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0x4B0D_0002);

    let mut found = false;
    for k in 0..d {
        let core = x.core(k);
        let hrr = h.row_ranks(core);
        for s in space.bond_sectors(k) {
            if s.irrep != Irrep::IDENTITY && hrr[s] > core.row_ranks[s] {
                found = true;
            }
        }
    }
    assert!(
        found,
        "vacuous: the operator band never expanded a non-identity-irrep plane anywhere"
    );
}

#[test]
fn spg4_t2_cross_symmetry_1body_c64() {
    use qntt::{C64, Scalar as _};
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (2, 0);
    let target = Irrep::IDENTITY;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC6_4A01);
    let (tr, _v) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &tr);
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..=i {
            let re = tr.get(&[i, j]);
            if i == j {
                t.set(&[i, i], C64::from_f64(re));
            } else if re != 0.0 {
                let phi = 0.2 + 0.17 * (i + 5 * j) as f64;
                let z = C64::new(re * phi.cos(), re * phi.sin());
                t.set(&[i, j], z);
                t.set(&[j, i], z.conj());
            }
        }
    }
    let v = CoeffTensor::<C64>::zeros(d, 4);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<C64>(d, n, two_s, pg, target, 3, 0xC6_4A02);
    let y = h.apply(&x);
    assert!(y.check(), "C64 1-body H·x must pass check()");
    assert_expand_purity_chain(&y, &pg);

    let err = max_err_over_two_sz_pg(&x, &t, &v, None, &y, space);
    assert!(err <= 1e-11, "C64 1-body cross-symmetry err={err}");
}
