// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;

use qntt::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::ordering::{irrep_blocked_order, permute_integrals};
use qntt::rotation::{compose_rotations, orthogonality_defect, rotate_integrals};
use qntt::solvers::joint::JointScfOpts;
use qntt::solvers::mals::{MalsOptions, NoiseType};
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, Su2TermScreen, rayleigh_quotient};
use qntt::su2::joint::{ModeOptOpts, su2_joint_ground_state, su2_mode_optimize};
use qntt::su2::ops::dot;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::moderot::ModeRotSweepOpts;
use qntt::su2::vector::Su2TtVector;
use qntt_testkit::{randomized_su2_state, seeded};

fn rq(h: &Su2SparseHamiltonian<f64>, x: &Su2TtVector<f64>) -> f64 {
    rayleigh_quotient(h, x) / dot(x, x)
}

#[test]
fn tsmo_handoff_composes_u0_then_u1() {
    let load = load_fixture_dir::<f64>(
        Path::new("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g");
    let ints: Integrals<f64> = load.ints;
    let space = Su2Space::new(6, 3, 1);
    assert_eq!((ints.norb, ints.nelec), (6, 4));
    let d = space.d;
    let x_seed: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x7040));

    let pre = su2_mode_optimize(
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
    let plain = |u: &[f64]| {
        let r = rotate_integrals(&ints, u);
        Su2SparseHamiltonian::new(space, &r.t, &r.v, None, f64::EPSILON)
    };
    let q_pre = rq(&pre.h, &pre.x);
    let q_pre_fresh = rq(&plain(&pre.u), &pre.x);
    let norm_dev = (dot(&pre.x, &pre.x) - 1.0).abs();
    println!(
        "su2 mode_optimize: {gates} gates, λ = {:.12}, |λ − RQ| = {:.3e}, |pre.h − plain| = \
         {:.3e}, |‖x‖² − 1| = {norm_dev:.3e}, history.lambda = {:.12}",
        pre.lambda,
        (pre.lambda - q_pre).abs(),
        (q_pre - q_pre_fresh).abs(),
        pre.history.lambda
    );
    assert!(gates > 0, "vacuous: the pre-stage took no rotation");
    assert!(
        (q_pre - q_pre_fresh).abs() <= 1e-10 * q_pre.abs(),
        "pre.h is not H(u₀): {q_pre} vs {q_pre_fresh}"
    );
    assert!(
        norm_dev <= 1e-12,
        "pre.x is not normalized: ‖x‖² − 1 = {norm_dev:.3e}"
    );
    assert!(
        (q_pre - pre.lambda).abs() <= 1e-8 * q_pre.abs(),
        "pre.lambda is not the Rayleigh quotient: {} vs {q_pre}",
        pre.lambda
    );

    let ints_0 = rotate_integrals(&ints, &pre.u);
    let out = su2_joint_ground_state(
        &ints_0,
        space,
        &pre.x,
        &JointScfOpts {
            orb_max_iter: 2,
            ..JointScfOpts::default()
        },
    );
    let moved = out
        .u
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let id = if i % (d + 1) == 0 { 1.0 } else { 0.0 };
            (v - id).abs()
        })
        .fold(0.0f64, f64::max);
    let u_total = compose_rotations(&pre.u, &out.u, d);
    let u_wrong = compose_rotations(&out.u, &pre.u, d);
    let q = rq(&out.h, &out.x);
    let q_total = rq(&plain(&u_total), &out.x);
    let q_wrong = rq(&plain(&u_wrong), &out.x);
    println!(
        "su2 joint: max |u₁ − I| = {moved:.3e}, λ = {:.12}, |q − q(u₀u₁)| = {:.3e}, \
         |q − q(u₁u₀)| = {:.3e}",
        out.lambda,
        (q - q_total).abs(),
        (q - q_wrong).abs()
    );
    assert!(
        moved > 1e-6,
        "vacuous: the joint stage took no orbital step"
    );
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
fn tsmo_typed_irrep_blocked_h2o_accepts() {
    let o = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load =
        load_fixture_dir::<f64>(Path::new("testdata/h2o_sto3g"), &o).expect("load h2o_sto3g typed");
    let (pg0, d) = (load.pg.pg, load.ints.norb);
    assert!(!pg0.is_trivial(), "h2o_sto3g must resolve C2v");
    let perm = irrep_blocked_order(pg0, d);
    let ints = permute_integrals(&load.ints, &perm);
    let pgd = ints
        .point_group(Some(load.pg.kind))
        .expect("permuted dump resolves");
    let pg = pgd.pg;
    for (a, &old) in perm.iter().enumerate() {
        assert_eq!(pg.irrep_of(a), pg0.irrep_of(old), "irrep′({a})");
    }
    let same = (0..d - 1)
        .filter(|&k| pg.irrep_of(k) == pg.irrep_of(k + 1))
        .count();
    assert!(same >= 3, "vacuous: {same} same-irrep bonds");

    for two_s in [0usize, 2] {
        let name = format!("typed h2o 2S={two_s}");
        let space = Su2Space::with_point_group(d, ints.nelec, two_s, pg, pgd.target);
        let x_seed: Su2TtVector<f64> =
            randomized_su2_state(space, 1, 2, &mut seeded(0x7042 + two_s as u64));
        let pre = su2_mode_optimize(
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
        let norm_dev = (dot(&pre.x, &pre.x) - 1.0).abs();
        println!(
            "{name}: {same} same-irrep bonds, {gates} gates, λ = {:.12}, |‖x‖² − 1| = \
             {norm_dev:.3e}",
            pre.lambda
        );
        assert!(
            gates > 0,
            "{name}: vacuous — the pre-stage took no rotation"
        );
        assert!(pre.history.u_orth_dev < 1e-12, "{name}: u drifted");
        assert!(orthogonality_defect(&pre.u, d) < 1e-12);
        for p in 0..d {
            for q in 0..d {
                if pg.irrep_of(p) != pg.irrep_of(q) {
                    assert!(
                        pre.u[p * d + q] == 0.0,
                        "{name}: pre.u[{p}][{q}] = {:e} mixes irreps",
                        pre.u[p * d + q]
                    );
                }
            }
        }
        let rot = rotate_integrals(&ints, &pre.u);
        let fresh = Su2SparseHamiltonian::new_refillable(
            space,
            &rot.t,
            &rot.v,
            None,
            Su2TermScreen::SymmetryAllowed(pg),
        );
        assert_eq!(
            pre.h.artifact_hash(),
            fresh.artifact_hash(),
            "{name}: pre.h is not H(u₀)"
        );
        let plain = Su2SparseHamiltonian::new(space, &rot.t, &rot.v, None, f64::EPSILON);
        let (q_pre, q_plain) = (rq(&pre.h, &pre.x), rq(&plain, &pre.x));
        assert!(
            (q_pre - q_plain).abs() <= 1e-10 * q_pre.abs(),
            "{name}: pre.h energy {q_pre} vs plain build at pre.u {q_plain}"
        );
        assert!(
            norm_dev <= 1e-12,
            "{name}: pre.x is not normalized: ‖x‖² − 1 = {norm_dev:.3e}"
        );
        assert!(
            (q_pre - pre.lambda).abs() <= 1e-8 * q_pre.abs(),
            "{name}: pre.lambda is not the Rayleigh quotient: {} vs {q_pre}",
            pre.lambda
        );
    }
}
