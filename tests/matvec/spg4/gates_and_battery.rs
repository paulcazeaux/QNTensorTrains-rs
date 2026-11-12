// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, assembly};
use qntt::su2::qn::{Su2Sector, Su2Space};
use qntt::su2::solvers::als::{AlsOptions, als};
use qntt::su2::spinalgebra::SpinAlgebra;
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::{Irrep, QnSpace};

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::spg4::util::{
    assert_expand_purity_chain, assert_purity, assert_su2tt_exact_eq, build_pg_state, c2v_coeffs,
    c2v_table, capped_randn_ab, find_different_irrep_pair, find_nonidentity_reachable_target,
    find_same_nonidentity_irrep_pair, max_err_over_two_sz_pg, randn_coeffs, su2_capped_ranks,
};

#[test]
fn spg4_t5_caps_gate_boundary_ranges_single_entry_no_panic() {
    let d = 5;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5B0D_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0x5B0D_0002);

    let _y = h.apply(&x);

    let rr = h.row_ranges(x.core(0));
    let s0 = Su2Sector::new(0, 0);
    let mut nonempty = 0usize;
    let mut found_boundary = false;
    for s in space.bond_sectors(0) {
        if !rr[s].is_empty() {
            nonempty += 1;
            if s == s0 {
                found_boundary = true;
            }
        }
    }
    assert_eq!(
        nonempty, 1,
        "row_ranges(core 0) must carry exactly one non-empty entry"
    );
    assert!(
        found_boundary,
        "row_ranges(core 0)'s sole entry must be (0,0,IDENTITY)"
    );

    let cr = h.col_ranges(x.core(d - 1));
    let last = Su2Sector::with_irrep(n, two_s, target);
    let mut nonempty2 = 0usize;
    let mut found_last = false;
    for s in space.bond_sectors(d) {
        if !cr[s].is_empty() {
            nonempty2 += 1;
            if s == last {
                found_last = true;
            }
        }
    }
    assert_eq!(
        nonempty2, 1,
        "col_ranges(core d-1) must carry exactly one non-empty entry"
    );
    assert!(
        found_last,
        "col_ranges(core d-1)'s sole entry must be (N,2S,target)"
    );

    let mut spin = SpinAlgebra::new();
    let assembled = assembly::assemble(space, &t, &v, None, f64::EPSILON, &mut spin);
    for s in space.bond_sectors(0) {
        if s.irrep != Irrep::IDENTITY {
            assert!(
                assembled.state_qns[0][s].is_empty(),
                "caps gate: state_qns[0][{s:?}] must be empty for every non-identity irrep"
            );
        }
    }
    let target_sector = Su2Sector::with_irrep(n, two_s, target);
    for s in space.bond_sectors(d) {
        if s != target_sector {
            assert!(
                assembled.state_qns[d][s].is_empty(),
                "caps gate: state_qns[{d}][{s:?}] must be empty except at the target sector"
            );
        }
    }
}

#[test]
#[should_panic(expected = "is not totally symmetric")]
fn spg4_t6_symmetry_violating_term_panics_at_nontrivial_group() {
    let d = 5;
    let pg = c2v_table(d);
    let (i, j) = find_different_irrep_pair(&pg, d);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[i, j], 10.0);
    t.set(&[j, i], 10.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);

    let space = Su2Space::with_point_group(d, 2, 0, pg, Irrep::IDENTITY);
    let _h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
}

#[test]
fn spg4_zero_hamiltonian_at_nontrivial_group_yields_zero_no_panic() {
    let d = 5;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let v = CoeffTensor::<f64>::zeros(d, 4);

    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0x6B0D_0001);
    let y = h.apply(&x);
    assert!(y.check(), "zero-Hamiltonian H·x must pass check()");
    for k in 0..d {
        let core = y.core(k);
        for c in 0..core.num_coeffs() {
            assert_eq!(
                core.coeff(c),
                0.0,
                "zero-Hamiltonian H·x must be the exact zero vector (core {k}, coeff {c})"
            );
        }
    }
}

#[test]
fn spg4_als_cross_symmetry_c2v_identity_target() {
    let d = 6;
    let pg = c2v_table(d);
    let (i, j) = find_same_nonidentity_irrep_pair(&pg, d);
    assert_eq!(
        pg.irrep_of(i),
        pg.irrep_of(j),
        "fixture bug: sites must share an irrep"
    );
    assert_ne!(
        pg.irrep_of(i),
        Irrep::IDENTITY,
        "fixture bug: sites must share a NON-identity irrep, else Δ is a no-op along the chain"
    );
    assert!(pg.order() > 1, "fixture group must be nontrivial");

    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    t.set(&[i, j], -1.0);
    t.set(&[j, i], -1.0);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let target = Irrep::IDENTITY;

    let su2_space = Su2Space::with_point_group(d, 2, 0, pg, target);
    let h_su2 = Su2SparseHamiltonian::new(su2_space, &t, &v, None, 1e-12);
    let x0_su2 = build_pg_state::<f64>(d, 2, 0, pg, target, 3, 0x5B44_0001);
    let (lambda_su2, x_su2) = als(
        &h_su2,
        &x0_su2,
        &AlsOptions {
            eps: 1e-9,
            ..Default::default()
        },
    );
    assert!(x_su2.check());
    assert_expand_purity_chain(&x_su2, &pg);
    assert!(
        (lambda_su2 - (-2.0)).abs() <= 1e-6,
        "su2 ALS must reach the analytic singlet ground -2.0: got {lambda_su2}"
    );

    let ab_space = QnSpace::with_point_group(d, 1, 1, pg, target);
    let h_ab = SparseHamiltonian::new(ab_space, &t, &v, None, 1e-12);
    let x0_ab = capped_randn_ab::<f64>(ab_space, 1, 3, 0x5B44_0002);
    let (lambda_ab, x_ab) = qntt::solvers::als::als(
        &h_ab,
        &x0_ab,
        &qntt::solvers::als::AlsOptions {
            eps: 1e-9,
            ..Default::default()
        },
    );
    assert!(x_ab.check());
    assert_purity(&pg, d, 1, 1, target, &x_ab.to_dense());
    assert!(
        (lambda_ab - (-2.0)).abs() <= 1e-6,
        "abelian ALS must reach the analytic singlet ground -2.0: got {lambda_ab}"
    );

    assert!(
        (lambda_su2 - lambda_ab).abs() <= 1e-6,
        "cross-symmetry mismatch: su2={lambda_su2} vs abelian={lambda_ab}"
    );
}

#[test]
fn spg4_nest_g1_apply_determinism_and_populated_cell_sanity() {
    let d = 5;
    let (n, two_s) = (3, 1);
    let space = Su2Space::new(d, n, two_s);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9E57_0001);
    let (t, v) = randn_coeffs(d, &mut rng);

    let h1 = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let h2 = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let ranks = su2_capped_ranks(&space, 3);
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0x9E57_0002);
    let x: Su2TtVector<f64> = su2_tt_randn(space, &ranks, &mut rng2);
    let y1 = h1.apply(&x);
    let y2 = h2.apply(&x);
    assert_su2tt_exact_eq(
        &y1,
        &y2,
        "two independent constructions of the same H must apply bit-identically (R1)",
    );

    let err = max_err_over_two_sz_pg(&x, &t, &v, None, &y1, space);
    assert!(
        err <= 1e-11,
        "Part-A nest sparsification must match the certified abelian cross-symmetry oracle: \
         err={err}"
    );

    let mut spin_a = SpinAlgebra::new();
    let assembled_a = assembly::assemble(space, &t, &v, None, f64::EPSILON, &mut spin_a);
    let mut spin_b = SpinAlgebra::new();
    let assembled_b = assembly::assemble(space, &t, &v, None, f64::EPSILON, &mut spin_b);

    let mut any_populated = false;
    for k in 0..d {
        let pa = assembled_a.coo[k].populated();
        let pb = assembled_b.coo[k].populated();
        assert_eq!(
            pa, pb,
            "site {k}: populated-cell count must be reproducible across independent assemblies"
        );
        if pa > 0 {
            any_populated = true;
        }
    }
    assert!(
        any_populated,
        "vacuous: no populated cell anywhere in the |G|=1 nest"
    );
}
