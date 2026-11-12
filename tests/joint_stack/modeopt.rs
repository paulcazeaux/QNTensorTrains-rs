// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;

use qntt::hamiltonian::{Construction, SparseHamiltonian, TermScreen, rayleigh_quotient};
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::ordering::{irrep_blocked_order, permute_integrals};
use qntt::rotation::{compose_rotations, rotate_integrals};
use qntt::solvers::joint::{JointScfOpts, ModeOptOpts, joint_ground_state, mode_optimize};
use qntt::solvers::mals::{MalsOptions, NoiseType};
use qntt::solvers::moderot::ModeRotSweepOpts;
use qntt::{TtVector, dot};
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn rq(h: &SparseHamiltonian<f64>, x: &TtVector<f64>) -> f64 {
    rayleigh_quotient(h, x) / dot(x, x)
}

fn assert_normalized(what: &str, pre: &qntt::solvers::joint::ModeOptOut) {
    let n2 = dot(&pre.x, &pre.x);
    let q = rq(&pre.h, &pre.x);
    println!(
        "{what}: ‖x‖² − 1 = {:.3e}, λ = {:.12}, RQ = {q:.12} (rel {:.1e})",
        n2 - 1.0,
        pre.lambda,
        (pre.lambda - q).abs() / q.abs()
    );
    assert!(
        (n2 - 1.0).abs() <= 1e-12,
        "{what}: pre.x is not normalized: ‖x‖² − 1 = {:.3e}",
        n2 - 1.0
    );
    assert!(
        (pre.lambda - q).abs() <= 1e-8 * q.abs(),
        "{what}: pre.lambda {} is not the Rayleigh quotient {q}",
        pre.lambda
    );
}

#[test]
fn tmo_handoff_composes_u0_then_u1() {
    let load = load_fixture_dir::<f64>(
        Path::new("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g");
    let (ints, space) = (load.ints, load.space);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7040);
    let x_seed: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    let pre = mode_optimize(
        &ints,
        space,
        &x_seed,
        &ModeOptOpts {
            mals: MalsOptions {
                eps: 1e-10,
                bond_dims: vec![2],
                noise: vec![0.0],
                noise_type: NoiseType::Random,
                max_iter: 4,
                ..MalsOptions::default()
            },
            sweep: ModeRotSweepOpts::default(),
        },
    );
    let gates: usize = pre.history.sweeps.iter().map(|s| s.n_gates).sum();
    assert!(gates > 0, "vacuous: the pre-stage took no rotation");
    assert_normalized("lih cap 2", &pre);
    let plain = |u: &[f64]| {
        let r = rotate_integrals(&ints, u);
        SparseHamiltonian::new_with(space, &r.t, &r.v, None, f64::EPSILON, Construction::Median)
    };
    let q_pre = rayleigh_quotient(&pre.h, &pre.x);
    let q_pre_fresh = rayleigh_quotient(&plain(&pre.u), &pre.x);
    assert!(
        (q_pre - q_pre_fresh).abs() <= 1e-10 * q_pre.abs(),
        "pre.h is not H(u₀)"
    );
    assert!(
        (q_pre - pre.lambda).abs() <= 1e-8 * q_pre.abs(),
        "pre.lambda is not ⟨x, h x⟩"
    );

    let ints_0 = rotate_integrals(&ints, &pre.u);
    let out = joint_ground_state(
        &ints_0,
        space,
        &pre.x,
        &JointScfOpts {
            orb_max_iter: 2,
            construction: Construction::Median,
            ..JointScfOpts::default()
        },
    );
    let moved = out.u.iter().enumerate().any(|(i, &v)| {
        let id = if i % (d + 1) == 0 { 1.0 } else { 0.0 };
        (v - id).abs() > 1e-6
    });
    assert!(moved, "vacuous: the joint stage took no orbital step");

    let u_total = compose_rotations(&pre.u, &out.u, d);
    let u_wrong = compose_rotations(&out.u, &pre.u, d);
    let q = rayleigh_quotient(&out.h, &out.x);
    let q_total = rayleigh_quotient(&plain(&u_total), &out.x);
    let q_wrong = rayleigh_quotient(&plain(&u_wrong), &out.x);
    assert!(
        (q - q_total).abs() <= 1e-10 * q.abs(),
        "out.h is not H(rotate_integrals(ints, u₀·u₁)): {q} vs {q_total}"
    );
    assert!(
        (q - q_wrong).abs() > 1e-7 * q.abs(),
        "the reversed composition also satisfies the identity — the order test is vacuous"
    );
}

#[test]
fn tmo_typed_irrep_blocked_h2o_accepts() {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(Path::new("testdata/h2o_sto3g"), &opts)
        .expect("load h2o_sto3g typed");
    let (pg0, d) = (load.space.pg, load.space.d);
    assert!(!pg0.is_trivial(), "h2o_sto3g must resolve C2v");
    let perm = irrep_blocked_order(pg0, d);
    let ints = permute_integrals(&load.ints, &perm);
    let space = ints
        .qn_space(Some(load.pg.kind))
        .expect("permuted dump resolves");
    let pg = space.pg;
    for (a, &old) in perm.iter().enumerate() {
        assert_eq!(pg.irrep_of(a), pg0.irrep_of(old), "irrep′({a})");
    }
    let same = (0..d - 1)
        .filter(|&k| pg.irrep_of(k) == pg.irrep_of(k + 1))
        .count();
    assert!(same >= 3, "vacuous: {same} same-irrep bonds");
    let x_seed = crate::util::capped_randn(space, 1, 2, 0x7041);

    let pre = mode_optimize(
        &ints,
        space,
        &x_seed,
        &ModeOptOpts {
            mals: MalsOptions {
                eps: 1e-10,
                bond_dims: vec![2],
                noise: vec![0.0],
                noise_type: NoiseType::Random,
                max_iter: 4,
                ..MalsOptions::default()
            },
            sweep: ModeRotSweepOpts::default(),
        },
    );
    let gates: usize = pre.history.sweeps.iter().map(|s| s.n_gates).sum();
    println!("typed mode_optimize: {same} same-irrep bonds, {gates} gates");
    assert!(gates > 0, "vacuous: the pre-stage took no rotation");
    assert_normalized("typed h2o cap 2", &pre);
    assert!(pre.history.u_orth_dev < 1e-12);
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert!(
                    pre.u[p * d + q] == 0.0,
                    "pre.u[{p}][{q}] = {:e} mixes irreps",
                    pre.u[p * d + q]
                );
            }
        }
    }
    let rot = rotate_integrals(&ints, &pre.u);
    let fresh = SparseHamiltonian::new_refillable(
        space,
        &rot.t,
        &rot.v,
        None,
        TermScreen::SymmetryAllowed(pg),
        Construction::Median,
    );
    assert_eq!(
        pre.h.artifact_hash(),
        fresh.artifact_hash(),
        "pre.h is not H(u₀)"
    );
    let plain = SparseHamiltonian::new_with(
        space,
        &rot.t,
        &rot.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );
    let q_pre = rayleigh_quotient(&pre.h, &pre.x);
    let q_plain = rayleigh_quotient(&plain, &pre.x);
    assert!(
        (q_pre - q_plain).abs() <= 1e-10 * q_pre.abs(),
        "pre.h energy {q_pre} vs plain build at pre.u {q_plain}"
    );
    assert!(
        (q_pre - pre.lambda).abs() <= 1e-8 * q_pre.abs(),
        "pre.lambda is not ⟨x, h x⟩"
    );
}
