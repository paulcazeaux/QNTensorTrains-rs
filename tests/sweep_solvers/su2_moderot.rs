// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::Arc;

use qntt::integrals::{
    Integrals, IntegralsOptions, PointGroupPolicy, load_fcidump, load_fixture_dir,
};
use qntt::ordering::{irrep_blocked_order, permute_integrals};
use qntt::rotation::{orthogonality_defect, rotate_integrals};
use qntt::solvers::mals::NoiseType;
use qntt::su2::fcidump::su2_space;
use qntt::su2::hamiltonian::{
    Su2PatternMask, Su2SparseHamiltonian, Su2TermScreen, rayleigh_quotient,
};
use qntt::su2::ops::dot;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::mals::{MalsOptions, mals_with_history};
use qntt::su2::solvers::moderot::{ModeRotHistory, ModeRotOpts, ModeRotSweepOpts, mals_moderot};
use qntt::su2::vector::Su2TtVector;
use qntt_testkit::{randomized_su2_state, seeded};

const EPS: f64 = f64::EPSILON;

fn lih() -> (Integrals<f64>, Su2Space) {
    let load = load_fixture_dir::<f64>(
        Path::new("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g");
    assert_eq!((load.ints.norb, load.ints.nelec), (6, 4));
    (load.ints, Su2Space::new(6, 3, 1))
}

fn h4() -> (Integrals<f64>, Su2Space) {
    let load = load_fcidump::<f64>(
        Path::new("testdata/h4.FCIDUMP"),
        None,
        &IntegralsOptions::default(),
    )
    .expect("load h4.FCIDUMP");
    assert_eq!((load.ints.norb, load.ints.nelec), (4, 4));
    (load.ints, Su2Space::new(4, 4, 2))
}

fn sa_refillable(ints: &Integrals<f64>, space: Su2Space) -> Su2SparseHamiltonian<f64> {
    Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::SymmetryAllowed(space.pg),
    )
}

fn opts(m: usize, max_iter: usize) -> MalsOptions {
    MalsOptions {
        eps: 1e-10,
        bond_dims: vec![m],
        noise: vec![0.0],
        noise_type: NoiseType::Random,
        max_iter,
        ..MalsOptions::default()
    }
}

fn start(space: Su2Space, seed: u64) -> Su2TtVector<f64> {
    randomized_su2_state(space, 1, 2, &mut seeded(seed))
}

fn gates(hist: &ModeRotHistory) -> usize {
    hist.sweeps.iter().map(|s| s.n_gates).sum()
}

fn rq(h: &Su2SparseHamiltonian<f64>, x: &Su2TtVector<f64>) -> f64 {
    rayleigh_quotient(h, x) / dot(x, x)
}

fn assert_bits_eq(a: &Su2TtVector<f64>, b: &Su2TtVector<f64>, what: &str) {
    assert_eq!(a.ranks, b.ranks, "{what}: rank grids differ");
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        assert_eq!(ca.num_coeffs(), cb.num_coeffs(), "{what}: core {k} size");
        for i in 0..ca.num_coeffs() {
            assert_eq!(
                ca.coeff(i).to_bits(),
                cb.coeff(i).to_bits(),
                "{what}: core {k} coefficient {i} ({} vs {}) not bitwise equal",
                ca.coeff(i),
                cb.coeff(i)
            );
        }
    }
}

fn bitwise_arm(what: &str, o: &MalsOptions, sweep: &ModeRotSweepOpts, zero_angles: bool) {
    let (ints, space) = lih();
    bitwise_arm_on(what, &ints, space, 0x9B_0001, o, sweep, zero_angles);
}

fn bitwise_arm_on(
    what: &str,
    ints: &Integrals<f64>,
    space: Su2Space,
    seed: u64,
    o: &MalsOptions,
    sweep: &ModeRotSweepOpts,
    zero_angles: bool,
) {
    let d = space.d;
    let x0 = start(space, seed);
    let h_ref = sa_refillable(ints, space);
    let (hist_ref, x_ref) = mals_with_history(&h_ref, &x0, o);
    let mut h = sa_refillable(ints, space);
    let mut x = x0;
    let (hist, u) = mals_moderot(&mut h, ints, &mut x, o, sweep);
    println!(
        "{what}: λ {:.12} in {} iterations (MALS {:.12} in {})",
        hist.lambda,
        hist.sweeps.len(),
        hist_ref.lambda,
        hist_ref.sweeps.len()
    );
    assert_eq!(
        hist.lambda.to_bits(),
        hist_ref.lambda.to_bits(),
        "{what}: λ {} vs the MALS driver's {}",
        hist.lambda,
        hist_ref.lambda
    );
    assert_eq!(
        hist.sweeps.len(),
        hist_ref.sweeps.len(),
        "{what}: iteration count"
    );
    for (it, (a, b)) in hist.sweeps.iter().zip(&hist_ref.sweeps).enumerate() {
        assert_eq!(
            (a.energy.to_bits(), a.discarded_weight.to_bits()),
            (b.energy.to_bits(), b.discarded_weight.to_bits()),
            "{what}: iteration {it} energy / discarded weight"
        );
        assert_eq!(
            (a.max_bond_dim, a.bond_dim, a.noise.to_bits()),
            (b.max_bond_dim, b.bond_dim, b.noise.to_bits()),
            "{what}: iteration {it} ranks / schedule"
        );
        assert_eq!(a.n_gates, 0, "{what}: iteration {it} took a gate");
        assert_eq!(
            (a.max_theta, a.functional_drop),
            (0.0, 0.0),
            "{what}: iteration {it} rotation statistics"
        );
    }
    assert_bits_eq(&x, &x_ref, what);
    assert!(
        (0..d * d).all(|i| u[i] == if i % (d + 1) == 0 { 1.0 } else { 0.0 }),
        "{what}: u is not exactly the identity"
    );
    assert_eq!(hist.u_orth_dev, 0.0, "{what}: u_orth_dev of the identity");
    assert_eq!(
        h.artifact_hash(),
        h_ref.artifact_hash(),
        "{what}: the operator was refilled"
    );
    assert_eq!(hist.thetas.len(), hist.sweeps.len(), "{what}: theta slots");
    for (it, th) in hist.thetas.iter().enumerate() {
        if zero_angles {
            assert_eq!(
                th.len(),
                2 * (d - 1),
                "{what}: iteration {it} searched {} bonds, not 2(d − 1)",
                th.len()
            );
            assert!(
                th.iter().all(|t| t.to_bits() == 0),
                "{what}: iteration {it} recorded a non-(+0.0) angle: {th:?}"
            );
        } else {
            assert!(th.is_empty(), "{what}: iteration {it} recorded angles");
        }
    }
}

#[test]
fn su2_mr_t1_without_rotations_is_bitwise_the_mals_driver() {
    let sweep = ModeRotSweepOpts {
        first_rot_iter: usize::MAX,
        ..ModeRotSweepOpts::default()
    };
    bitwise_arm("T1 noise [0]", &opts(4, 3), &sweep, false);
    let o = MalsOptions {
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Perturbative,
        ..opts(4, 3)
    };
    bitwise_arm("T1 perturbative [1e-3, 1e-4, 0]", &o, &sweep, false);
}

#[test]
fn su2_mr_t1p_zero_angles_are_bitwise_the_mals_driver() {
    let sweep = ModeRotSweepOpts {
        rot: ModeRotOpts {
            improve_tol: f64::INFINITY,
            ..ModeRotOpts::default()
        },
        first_rot_iter: 0,
    };
    bitwise_arm("T1′ noise [0]", &opts(4, 3), &sweep, true);
    let o = MalsOptions {
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Perturbative,
        ..opts(4, 3)
    };
    bitwise_arm("T1′ perturbative [1e-3, 1e-4, 0]", &o, &sweep, true);
}

type Loop = (
    Integrals<f64>,
    Su2Space,
    Su2SparseHamiltonian<f64>,
    u64,
    Su2TtVector<f64>,
    Vec<f64>,
    ModeRotHistory,
);

fn closed_loop(fx: fn() -> (Integrals<f64>, Su2Space), seed: u64) -> Loop {
    let (ints, space) = fx();
    let mut h = sa_refillable(&ints, space);
    let mint = h.skeleton_hash();
    let mut x = start(space, seed);
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &opts(0, 6), &sweep);
    (ints, space, h, mint, x, u, hist)
}

#[test]
fn su2_mr_t2_refilled_operator_is_the_fresh_mint_at_u() {
    for (name, fx, seed) in [
        ("lih (6,3,1)", lih as fn() -> _, 0x9B_0201u64),
        ("h4 (4,4,2)", h4, 0x9B_0202),
    ] {
        let (ints, space, h, mint, _x, u, hist) = closed_loop(fx, seed);
        let d = space.d;
        let g = gates(&hist);
        println!(
            "{name}: {g} gates over {} iterations, max|θ| {:.3e}, λ {:.12}, u_orth_dev {:.2e}",
            hist.sweeps.len(),
            hist.sweeps.iter().map(|s| s.max_theta).fold(0.0, f64::max),
            hist.lambda,
            hist.u_orth_dev
        );
        assert!(g > 0, "{name}: vacuous — no rotation was taken");
        assert!(
            hist.u_orth_dev < 1e-12,
            "{name}: u drifted: {:.3e}",
            hist.u_orth_dev
        );
        let dev = orthogonality_defect(&u, d);
        assert!(dev < 1e-12, "{name}: recomputed u defect {dev:.3e}");
        let fresh = sa_refillable(&rotate_integrals(&ints, &u), space);
        assert_eq!(
            h.artifact_hash(),
            fresh.artifact_hash(),
            "{name}: the refilled operator is not the fresh mint at u"
        );
        assert_eq!(
            h.skeleton_hash(),
            mint,
            "{name}: the skeleton moved off the U = I mint's"
        );
    }
}

#[test]
fn su2_mr_t3_sweep_energy_is_the_fresh_operator_energy_in_the_returned_basis() {
    for (name, fx, seed) in [
        ("lih (6,3,1)", lih as fn() -> _, 0x9B_0301u64),
        ("h4 (4,4,2)", h4, 0x9B_0302),
    ] {
        let (ints, space, _h, _mint, x, u, hist) = closed_loop(fx, seed);
        let g = gates(&hist);
        assert!(g > 0, "{name}: vacuous — no rotation was taken");
        let rot = rotate_integrals(&ints, &u);
        let fresh = sa_refillable(&rot, space);
        let plain = Su2SparseHamiltonian::new(space, &rot.t, &rot.v, None, EPS);
        let (e_fresh, e_plain) = (rq(&fresh, &x), rq(&plain, &x));
        println!(
            "{name}: gates {g}, λ_sweep {:.12}, RQ_fresh {e_fresh:.12} (rel {:.1e}), RQ_plain \
             {e_plain:.12} (rel {:.1e})",
            hist.lambda,
            (e_fresh - hist.lambda).abs() / e_fresh.abs(),
            (e_plain - e_fresh).abs() / e_fresh.abs()
        );
        assert!(
            (e_fresh - hist.lambda).abs() <= 1e-8 * e_fresh.abs(),
            "{name}: sweep energy {} vs fresh-operator energy {e_fresh}",
            hist.lambda
        );
        assert!(
            (e_plain - e_fresh).abs() <= 1e-10 * e_fresh.abs(),
            "{name}: plain-build energy {e_plain} vs refillable {e_fresh}"
        );
    }
}

const T4_MARGIN: f64 = 5e-5;

#[test]
fn su2_mr_t4_rotations_lower_the_energy_at_fixed_multiplet_rank_lih() {
    let (ints, space) = lih();
    for m in [2usize, 3] {
        for seed in 0..3u64 {
            let o = opts(m, 8);
            let x0 = start(space, 0x9B_0400 + 16 * m as u64 + seed);
            let h_ref = sa_refillable(&ints, space);
            let (hist_ref, _x_ref) = mals_with_history(&h_ref, &x0, &o);
            let lam_plain = hist_ref.lambda;
            let mut h = sa_refillable(&ints, space);
            let mut x = x0;
            let (hist, _u) = mals_moderot(&mut h, &ints, &mut x, &o, &ModeRotSweepOpts::default());
            let g = gates(&hist);
            println!(
                "cap {m} seed {seed}: plain {lam_plain:.12}  moderot {:.12}  gap {:.3e}  (gates {g}, \
                 kept {:?} / {:?})",
                hist.lambda,
                lam_plain - hist.lambda,
                hist.sweeps
                    .iter()
                    .map(|s| s.max_bond_dim)
                    .collect::<Vec<_>>(),
                hist_ref
                    .sweeps
                    .iter()
                    .map(|s| s.max_bond_dim)
                    .collect::<Vec<_>>()
            );
            assert!(
                g > 0,
                "cap {m} seed {seed}: vacuous — no rotation was taken"
            );
            assert!(
                hist.lambda <= lam_plain - T4_MARGIN,
                "cap {m} seed {seed}: mode rotation did not lower the energy by {T4_MARGIN:.0e}: \
                 {} vs plain {lam_plain} (gap {:.3e})",
                hist.lambda,
                lam_plain - hist.lambda
            );
        }
    }
}

fn run_h4(mut h: Su2SparseHamiltonian<f64>, ints: &Integrals<f64>, space: Su2Space) {
    let mut x = start(space, 0x9B_0601);
    let _ = mals_moderot(
        &mut h,
        ints,
        &mut x,
        &opts(2, 1),
        &ModeRotSweepOpts::default(),
    );
}

#[test]
#[should_panic(expected = "refillable build on Su2TermScreen::SymmetryAllowed")]
fn su2_mr_t6_refuses_a_magnitude_screen() {
    let (ints, space) = h4();
    let h = Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::Magnitude(EPS),
    );
    run_h4(h, &ints, space);
}

#[test]
#[should_panic(expected = "refillable build on Su2TermScreen::SymmetryAllowed")]
fn su2_mr_t6_refuses_a_mask_screen() {
    let (ints, space) = h4();
    let m = Su2PatternMask::from_magnitude(&ints.t, &ints.v, EPS, space.pg);
    let h = Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::Mask(Arc::new(m)),
    );
    run_h4(h, &ints, space);
}

#[test]
#[should_panic(expected = "refillable build on Su2TermScreen::SymmetryAllowed")]
fn su2_mr_t6_refuses_a_plain_build() {
    let (ints, space) = h4();
    let h = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, EPS);
    run_h4(h, &ints, space);
}

#[test]
fn su2_mr_t6_typed_space_is_accepted_and_rotates_same_irrep_bonds_only() {
    let o = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load =
        load_fixture_dir::<f64>(Path::new("testdata/h2o_sto3g"), &o).expect("load h2o_sto3g typed");
    let space = su2_space(&load.ints, &load.pg, Some(0)).expect("h2o SU(2) space");
    assert!(!space.pg.is_trivial(), "h2o_sto3g must resolve C2v");
    let d = space.d;
    assert_eq!(
        same_irrep_bonds(space),
        vec![0],
        "canonical C2v h2o: (0, 1) is the only same-irrep bond"
    );
    let mut h = sa_refillable(&load.ints, space);
    let mut x = start(space, 0x9B_0604);
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let (hist, u) = mals_moderot(&mut h, &load.ints, &mut x, &opts(2, 2), &sweep);
    println!(
        "canonical typed h2o: λ {:.12}, gates {} (bond 0 only), θ {:?}",
        hist.lambda,
        gates(&hist),
        hist.thetas
    );
    assert_eq!(hist.thetas.len(), hist.sweeps.len());
    for (it, th) in hist.thetas.iter().enumerate() {
        assert_eq!(th.len(), 2 * (d - 1), "iteration {it}: slots");
        for k in 1..d - 1 {
            for slot in [k, 2 * d - 3 - k] {
                assert_eq!(
                    th[slot].to_bits(),
                    0,
                    "iteration {it}: cross-irrep bond {k} (slot {slot}) recorded {}",
                    th[slot]
                );
            }
        }
    }
    assert!(
        hist.u_orth_dev < 1e-12,
        "u drifted: {:.3e}",
        hist.u_orth_dev
    );
    for a in 0..d {
        for b in 0..d {
            if a < 2 && b < 2 {
                continue;
            }
            let id = if a == b { 1.0f64 } else { 0.0 };
            assert_eq!(
                u[a * d + b].to_bits(),
                id.to_bits(),
                "u[{a}][{b}] = {:e} outside the (0, 1) block",
                u[a * d + b]
            );
        }
    }
}

fn h2o_typed(
    order: impl Fn(qntt::PointGroup, usize) -> Vec<usize>,
    two_s: usize,
) -> (Integrals<f64>, Su2Space) {
    let o = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load =
        load_fixture_dir::<f64>(Path::new("testdata/h2o_sto3g"), &o).expect("load h2o_sto3g typed");
    let (pg0, d) = (load.pg.pg, load.ints.norb);
    assert!(!pg0.is_trivial(), "h2o_sto3g must resolve C2v");
    let perm = order(pg0, d);
    let ints = permute_integrals(&load.ints, &perm);
    let pgd = ints
        .point_group(Some(load.pg.kind))
        .expect("permuted dump resolves");
    for (a, &old) in perm.iter().enumerate() {
        assert_eq!(pgd.pg.irrep_of(a), pg0.irrep_of(old), "irrep′({a})");
    }
    assert_eq!(pgd.target, load.pg.target, "target irrep");
    let space = Su2Space::with_point_group(d, ints.nelec, two_s, pgd.pg, pgd.target);
    (ints, space)
}

fn h2o_blocked(two_s: usize) -> (Integrals<f64>, Su2Space) {
    h2o_typed(irrep_blocked_order, two_s)
}

const ALL_CROSS: [usize; 7] = [0, 2, 1, 4, 3, 6, 5];

fn same_irrep_bonds(space: Su2Space) -> Vec<usize> {
    (0..space.d - 1)
        .filter(|&k| space.pg.irrep_of(k) == space.pg.irrep_of(k + 1))
        .collect()
}

fn cross_irrep_nonzeros(u: &[f64], space: Su2Space) -> Vec<(usize, usize, f64)> {
    let (pg, d) = (space.pg, space.d);
    let mut out = Vec::new();
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) && u[p * d + q] != 0.0 {
                out.push((p, q, u[p * d + q]));
            }
        }
    }
    out
}

#[test]
fn su2_mr_t7_typed_without_rotations_is_bitwise_the_mals_driver() {
    let sweep = ModeRotSweepOpts {
        first_rot_iter: usize::MAX,
        ..ModeRotSweepOpts::default()
    };
    for two_s in [0usize, 2] {
        let (ints, space) = h2o_blocked(two_s);
        assert_eq!(same_irrep_bonds(space), vec![0, 1, 2, 4]);
        bitwise_arm_on(
            &format!("T7 typed h2o 2S={two_s}"),
            &ints,
            space,
            0x9B_0701 + two_s as u64,
            &opts(4, 3),
            &sweep,
            false,
        );
    }
}

#[test]
fn su2_mr_t7p_typed_all_cross_irrep_order_is_bitwise_the_mals_driver() {
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let pert = MalsOptions {
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Perturbative,
        ..opts(4, 3)
    };
    for two_s in [0usize, 2] {
        let (ints, space) = h2o_typed(|_, _| ALL_CROSS.to_vec(), two_s);
        assert!(
            same_irrep_bonds(space).is_empty(),
            "the order must have no same-irrep adjacent pair"
        );
        for (label, o) in [("noise [0]", opts(4, 3)), ("perturbative", pert.clone())] {
            bitwise_arm_on(
                &format!("T7′ all-cross h2o 2S={two_s} {label}"),
                &ints,
                space,
                0x9B_0711 + two_s as u64,
                &o,
                &sweep,
                true,
            );
        }
    }
}

#[test]
fn su2_mr_t8_typed_irrep_blocked_h2o_closed_loop() {
    for two_s in [0usize, 2] {
        let name = format!("typed h2o 2S={two_s}");
        let (ints, space) = h2o_blocked(two_s);
        let d = space.d;
        let same = same_irrep_bonds(space);
        let cross: Vec<usize> = (0..d - 1).filter(|k| !same.contains(k)).collect();
        assert_eq!(same, vec![0, 1, 2, 4], "{name}: same-irrep bonds");
        assert_eq!(cross, vec![3, 5], "{name}: cross-irrep bonds");

        let mut h = sa_refillable(&ints, space);
        let mint = h.skeleton_hash();
        let mut x = start(space, 0x9B_0801 + two_s as u64);
        let sweep = ModeRotSweepOpts {
            first_rot_iter: 0,
            ..ModeRotSweepOpts::default()
        };
        let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &opts(0, 4), &sweep);

        let g = gates(&hist);
        let nonzero = hist.thetas.iter().flatten().filter(|t| **t != 0.0).count();
        let mut per_bond = vec![0usize; d - 1];
        for th in &hist.thetas {
            for (slot, t) in th.iter().enumerate() {
                if *t != 0.0 {
                    per_bond[if slot < d - 1 { slot } else { 2 * d - 3 - slot }] += 1;
                }
            }
        }
        println!(
            "{name}: {g} gates over {} iterations (per bond {per_bond:?}), max|θ| {:.3e}",
            hist.sweeps.len(),
            hist.sweeps.iter().map(|s| s.max_theta).fold(0.0, f64::max)
        );
        assert!(g > 0, "{name}: vacuous — no same-irrep rotation was taken");
        assert_eq!(
            nonzero, g,
            "{name}: every gate is a θ ≠ 0 slot and vice versa"
        );
        assert_eq!(hist.thetas.len(), hist.sweeps.len());
        for (it, th) in hist.thetas.iter().enumerate() {
            assert_eq!(th.len(), 2 * (d - 1), "{name}: iteration {it} slots");
            for &k in &cross {
                for slot in [k, 2 * d - 3 - k] {
                    assert_eq!(
                        th[slot].to_bits(),
                        0,
                        "{name}: iteration {it}: cross-irrep bond {k} (slot {slot}) recorded {}",
                        th[slot]
                    );
                }
            }
        }

        assert!(
            hist.u_orth_dev < 1e-12,
            "{name}: u drifted: {:.3e}",
            hist.u_orth_dev
        );
        assert!(orthogonality_defect(&u, d) < 1e-12);
        let leak = cross_irrep_nonzeros(&u, space);
        assert!(leak.is_empty(), "{name}: u mixes irreps: {leak:?}");

        let rot = rotate_integrals(&ints, &u);
        for i in 0..d {
            for j in 0..d {
                if space.pg.irrep_of(i) != space.pg.irrep_of(j) {
                    assert!(
                        rot.t.get(&[i, j]) == 0.0,
                        "{name}: forbidden t[{i},{j}] went live"
                    );
                }
            }
        }
        let fresh = sa_refillable(&rot, space);
        assert_eq!(
            h.artifact_hash(),
            fresh.artifact_hash(),
            "{name}: the refilled operator is not the fresh typed mint at u"
        );
        assert_eq!(
            h.skeleton_hash(),
            mint,
            "{name}: the skeleton moved off the U = I mint's"
        );
        let plain = Su2SparseHamiltonian::new(space, &rot.t, &rot.v, None, EPS);
        let (e_fresh, e_plain) = (rq(&fresh, &x), rq(&plain, &x));
        let rel = |a: f64| (a - hist.lambda).abs() / a.abs();
        println!(
            "{name}: λ_sweep {:.14}, RQ_fresh {e_fresh:.14} (rel {:.1e}), RQ_plain {e_plain:.14} \
             (rel {:.1e})",
            hist.lambda,
            rel(e_fresh),
            rel(e_plain)
        );
        assert!(
            rel(e_fresh) <= 1e-10,
            "{name}: sweep energy {} vs fresh-operator energy {e_fresh}",
            hist.lambda
        );
        assert!(
            rel(e_plain) <= 1e-10,
            "{name}: sweep energy {} vs plain-build energy {e_plain}",
            hist.lambda
        );
    }
}

#[test]
#[should_panic(expected = "not in the same site order")]
fn su2_mr_t9_typed_refuses_integrals_in_another_site_order() {
    let (ints_b, space) = h2o_blocked(2);
    let (ints_c, space_c) = h2o_typed(|_, d| (0..d).collect(), 2);
    assert!(
        (0..space.d).any(|k| space.pg.irrep_of(k) != space_c.pg.irrep_of(k)),
        "vacuous: the two orders agree site by site"
    );
    let mut h = sa_refillable(&ints_b, space);
    let mut x = start(space, 0x9B_0901);
    let _ = mals_moderot(
        &mut h,
        &ints_c,
        &mut x,
        &opts(2, 1),
        &ModeRotSweepOpts::default(),
    );
}
