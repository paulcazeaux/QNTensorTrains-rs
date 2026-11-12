// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;

use qntt::hamiltonian::{Construction, SparseHamiltonian, TermScreen, rayleigh_quotient};
use qntt::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::ordering::{irrep_blocked_order, permute_integrals};
use qntt::rotation::{orthogonality_defect, rotate_integrals};
use qntt::solvers::mals::{MalsOptions, NoiseType, mals_mut};
use qntt::solvers::moderot::{ModeRotSweepOpts, mals_moderot};
use qntt::{QnSpace, RankGrid, TtVector, max_ranks, tt_randn};
use qntt_testkit::randomized_state;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn lih() -> (Integrals<f64>, QnSpace) {
    let load = load_fixture_dir::<f64>(
        Path::new("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g");
    (load.ints, load.space)
}

fn median_refillable(ints: &Integrals<f64>, space: QnSpace) -> SparseHamiltonian<f64> {
    SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        TermScreen::SymmetryAllowed(space.pg),
        Construction::Median,
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

fn start(space: QnSpace, seed: u64) -> TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    randomized_state(space, 1, 2, &mut rng)
}

#[test]
fn mr_without_rotations_is_bitwise_the_mals_driver() {
    let (ints, space) = lih();
    let o = opts(4, 3);
    let x0 = start(space, 0x30DE);
    let h_ref = median_refillable(&ints, space);
    let mut x_ref = x0.clone();
    let lam_ref = mals_mut(&h_ref, &mut x_ref, &o);

    let mut h = median_refillable(&ints, space);
    let mut x = x0;
    let sweep = ModeRotSweepOpts {
        first_rot_iter: usize::MAX,
        ..ModeRotSweepOpts::default()
    };
    let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &o, &sweep);
    assert_eq!(
        hist.lambda.to_bits(),
        lam_ref.to_bits(),
        "{} vs {lam_ref}",
        hist.lambda
    );
    let (a, b) = (x.to_dense(), x_ref.to_dense());
    assert!(
        a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
        "state differs"
    );
    assert!(hist.sweeps.iter().all(|s| s.n_gates == 0));
    let d = space.d;
    assert!((0..d * d).all(|i| u[i] == if i % (d + 1) == 0 { 1.0 } else { 0.0 }));
}

#[test]
fn mr_sweep_energy_is_the_fresh_operator_energy_in_the_returned_basis() {
    let (ints, space) = lih();
    let o = opts(0, 6);
    let mut h = median_refillable(&ints, space);
    let mut x = start(space, 0x30DF);
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &o, &sweep);
    let gates: usize = hist.sweeps.iter().map(|s| s.n_gates).sum();
    assert!(gates > 0, "vacuous: no rotation was taken");
    assert!(
        hist.u_orth_dev < 1e-12,
        "u drifted: {:.3e}",
        hist.u_orth_dev
    );
    let rot = rotate_integrals(&ints, &u);
    let fresh = SparseHamiltonian::new_refillable(
        space,
        &rot.t,
        &rot.v,
        None,
        TermScreen::SymmetryAllowed(space.pg),
        Construction::Median,
    );
    assert_eq!(
        h.artifact_hash(),
        fresh.artifact_hash(),
        "refilled operator ≠ fresh at u"
    );
    let rq = rayleigh_quotient(&fresh, &x);
    println!(
        "gates {gates}, λ_sweep {:.12}, RQ_fresh {rq:.12}",
        hist.lambda
    );
    assert!(
        (rq - hist.lambda).abs() <= 1e-8 * rq.abs(),
        "sweep energy {} vs fresh-operator energy {rq}",
        hist.lambda
    );
}

#[test]
fn mr_rotations_lower_the_energy_at_fixed_bond_dimension_lih() {
    let (ints, space) = lih();
    for m in [2usize, 4] {
        let o = opts(m, 8);
        let x0 = start(space, 0x30E0 + m as u64);
        let h_ref = median_refillable(&ints, space);
        let mut x_ref = x0.clone();
        let lam_plain = mals_mut(&h_ref, &mut x_ref, &o);
        let mut h = median_refillable(&ints, space);
        let mut x = x0;
        let (hist, _u) = mals_moderot(&mut h, &ints, &mut x, &o, &ModeRotSweepOpts::default());
        let gates: usize = hist.sweeps.iter().map(|s| s.n_gates).sum();
        println!(
            "M = {m}: plain {lam_plain:.12}  moderot {:.12}  (gates {gates})",
            hist.lambda
        );
        assert!(
            hist.lambda <= lam_plain + 1e-9,
            "M = {m}: mode rotation raised the energy: {} > {lam_plain}",
            hist.lambda
        );
    }
}

#[test]
#[should_panic(expected = "Construction::Median")]
fn mr_refuses_a_bipartite_operator() {
    let (ints, space) = lih();
    let mut h = SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        TermScreen::SymmetryAllowed(space.pg),
        Construction::Bipartite,
    );
    let mut x = start(space, 0x30E9);
    let _ = mals_moderot(
        &mut h,
        &ints,
        &mut x,
        &opts(2, 1),
        &ModeRotSweepOpts::default(),
    );
}

fn h2o_typed(order: impl Fn(qntt::PointGroup, usize) -> Vec<usize>) -> (Integrals<f64>, QnSpace) {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(Path::new("testdata/h2o_sto3g"), &opts)
        .expect("load h2o_sto3g typed");
    let (pg, d) = (load.space.pg, load.space.d);
    assert!(!pg.is_trivial(), "h2o_sto3g must resolve C2v");
    let perm = order(pg, d);
    let ints = permute_integrals(&load.ints, &perm);
    let space = ints
        .qn_space(Some(load.pg.kind))
        .expect("permuted dump resolves");
    for (a, &old) in perm.iter().enumerate() {
        assert_eq!(space.pg.irrep_of(a), pg.irrep_of(old), "irrep′({a})");
    }
    assert_eq!(space.target, load.space.target);
    (ints, space)
}

fn h2o_blocked() -> (Integrals<f64>, QnSpace) {
    h2o_typed(irrep_blocked_order)
}

fn same_irrep_bonds(space: QnSpace) -> Vec<usize> {
    (0..space.d - 1)
        .filter(|&k| space.pg.irrep_of(k) == space.pg.irrep_of(k + 1))
        .collect()
}

fn capped_start(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let d = space.d;
    let caps = max_ranks(&space);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                let draw = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
                r[s] = draw.min(caps[b][s]);
            }
            r
        })
        .collect();
    tt_randn(space, &ranks, &mut rng)
}

fn cross_irrep_nonzeros(u: &[f64], space: QnSpace) -> Vec<(usize, usize, f64)> {
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
fn mr_typed_irrep_blocked_h2o_closed_loop() {
    let (ints, space) = h2o_blocked();
    let d = space.d;
    let same = same_irrep_bonds(space);
    let cross: Vec<usize> = (0..d - 1).filter(|k| !same.contains(k)).collect();
    println!("same-irrep bonds {same:?}, cross-irrep bonds {cross:?}");
    assert!(same.len() >= 3, "vacuous: {same:?}");
    assert_eq!(cross, vec![3, 5]);

    let o = opts(0, 4);
    let mut h = median_refillable(&ints, space);
    let mut x = capped_start(space, 1, 2, 0x30F0);
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &o, &sweep);

    let gates: usize = hist.sweeps.iter().map(|s| s.n_gates).sum();
    let nonzero: usize = hist.thetas.iter().flatten().filter(|t| **t != 0.0).count();
    println!("gates {gates}, nonzero θ slots {nonzero}");
    assert!(gates > 0, "vacuous: no rotation was taken");
    assert_eq!(nonzero, gates, "every gate is a θ ≠ 0 slot and vice versa");
    assert_eq!(hist.thetas.len(), hist.sweeps.len());
    for (it, th) in hist.thetas.iter().enumerate() {
        assert_eq!(
            th.len(),
            2 * (d - 1),
            "iteration {it}: one slot per bond and sweep"
        );
        for &k in &cross {
            for slot in [k, 2 * d - 3 - k] {
                assert!(
                    th[slot] == 0.0,
                    "iteration {it}: cross-irrep bond {k} (slot {slot}) rotated by {}",
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
    assert!(orthogonality_defect(&u, d) < 1e-12);
    let leak = cross_irrep_nonzeros(&u, space);
    assert!(leak.is_empty(), "u mixes irreps: {leak:?}");

    let rot = rotate_integrals(&ints, &u);
    for i in 0..d {
        for j in 0..d {
            if space.pg.irrep_of_sites([i, j]) != qntt::Irrep::IDENTITY {
                assert!(rot.t.get(&[i, j]) == 0.0, "forbidden t[{i},{j}] went live");
            }
        }
    }
    let fresh = median_refillable(&rot, space);
    assert_eq!(
        h.artifact_hash(),
        fresh.artifact_hash(),
        "refilled operator ≠ fresh at u"
    );
    let plain = SparseHamiltonian::new_with(
        space,
        &rot.t,
        &rot.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );
    let rq = rayleigh_quotient(&fresh, &x);
    let rq_plain = rayleigh_quotient(&plain, &x);
    println!(
        "λ_sweep {:.14}, RQ_fresh {rq:.14} (gap {:.2e}), RQ_plain {rq_plain:.14}",
        hist.lambda,
        (rq - hist.lambda).abs()
    );
    assert!(
        (rq - hist.lambda).abs() <= 1e-10 * rq.abs(),
        "sweep energy {} vs fresh-operator energy {rq}",
        hist.lambda
    );
    assert!(
        (rq_plain - hist.lambda).abs() <= 1e-10 * rq_plain.abs(),
        "sweep energy {} vs plain-build energy {rq_plain}",
        hist.lambda
    );
}

#[test]
#[should_panic(expected = "not in the same site order")]
fn mr_typed_refuses_integrals_in_another_site_order() {
    let (ints_b, space) = h2o_blocked();
    let (ints_c, space_c) = h2o_typed(|_, d| (0..d).collect());
    assert!(
        (0..space.d).any(|k| space.pg.irrep_of(k) != space_c.pg.irrep_of(k)),
        "vacuous: the two orders agree site by site"
    );
    let mut h = median_refillable(&ints_b, space);
    let mut x = capped_start(space, 1, 2, 0x30F7);
    let _ = mals_moderot(
        &mut h,
        &ints_c,
        &mut x,
        &opts(2, 1),
        &ModeRotSweepOpts::default(),
    );
}

#[test]
fn mr_typed_without_rotations_is_bitwise_the_mals_driver() {
    let (ints, space) = h2o_blocked();
    assert!(same_irrep_bonds(space).len() >= 3);
    let o = opts(4, 3);
    let x0 = capped_start(space, 1, 2, 0x30F1);
    let h_ref = median_refillable(&ints, space);
    let mut x_ref = x0.clone();
    let lam_ref = mals_mut(&h_ref, &mut x_ref, &o);

    let mut h = median_refillable(&ints, space);
    let mut x = x0;
    let sweep = ModeRotSweepOpts {
        first_rot_iter: usize::MAX,
        ..ModeRotSweepOpts::default()
    };
    let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &o, &sweep);
    assert_eq!(
        hist.lambda.to_bits(),
        lam_ref.to_bits(),
        "{} vs {lam_ref}",
        hist.lambda
    );
    let (a, b) = (x.to_dense(), x_ref.to_dense());
    assert!(
        a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
        "state differs"
    );
    assert!(hist.sweeps.iter().all(|s| s.n_gates == 0));
    assert!(hist.thetas.iter().all(|t| t.is_empty()));
    let d = space.d;
    assert!((0..d * d).all(|i| u[i] == if i % (d + 1) == 0 { 1.0 } else { 0.0 }));
}

#[test]
fn mr_typed_all_cross_irrep_order_is_bitwise_the_mals_driver() {
    let (ints, space) = h2o_typed(|_, _| vec![0, 2, 1, 4, 3, 6, 5]);
    assert!(
        same_irrep_bonds(space).is_empty(),
        "the order must have no same-irrep adjacent pair"
    );
    let d = space.d;
    let o = opts(4, 3);
    let x0 = capped_start(space, 1, 2, 0x30F2);
    let h_ref = median_refillable(&ints, space);
    let mut x_ref = x0.clone();
    let lam_ref = mals_mut(&h_ref, &mut x_ref, &o);

    let mut h = median_refillable(&ints, space);
    let mut x = x0;
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &o, &sweep);
    assert_eq!(
        hist.lambda.to_bits(),
        lam_ref.to_bits(),
        "{} vs {lam_ref}",
        hist.lambda
    );
    let (a, b) = (x.to_dense(), x_ref.to_dense());
    assert!(
        a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()),
        "state differs"
    );
    assert!(hist.sweeps.iter().all(|s| s.n_gates == 0));
    assert!(!hist.thetas.is_empty());
    for (it, th) in hist.thetas.iter().enumerate() {
        assert_eq!(th.len(), 2 * (d - 1), "iteration {it}");
        assert!(
            th.iter().all(|t| t.to_bits() == 0.0f64.to_bits()),
            "iteration {it}: {th:?}"
        );
    }
    assert!(
        (0..d * d).all(|i| u[i].to_bits() == if i % (d + 1) == 0 { 1.0f64 } else { 0.0 }.to_bits())
    );
    assert_eq!(
        hist.gauge_plan_bytes, 0,
        "a cross-irrep pair must not be planned"
    );
    assert_eq!(
        h.artifact_hash(),
        h_ref.artifact_hash(),
        "the operator was refilled"
    );
}

fn typed_dense(
    table: &[u8],
    (nup, ndn): (usize, usize),
    target: u8,
    seed: u64,
) -> (Integrals<f64>, QnSpace) {
    use qntt::{Irrep, PointGroup};
    use qntt_testkit::dense::{random_sym_t, random_sym_v};
    let d = table.len();
    let irs: Vec<Irrep> = table.iter().map(|&i| Irrep::new(i)).collect();
    let pg = PointGroup::new(2, &irs);
    let space = QnSpace::with_point_group(d, nup, ndn, pg, Irrep::new(target));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut t = random_sym_t(d, &mut rng);
    let mut v = random_sym_v(d, &mut rng);
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of_sites([i, j]) != Irrep::IDENTITY {
                t.set(&[i, j], 0.0);
            }
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                        v.set(&[i, j, k, l], 0.0);
                    }
                }
            }
        }
    }
    let ints = Integrals {
        t,
        v,
        e0: 0.0,
        norb: d,
        nelec: nup + ndn,
        ms2: nup as i64 - ndn as i64,
        orbsym: Vec::new(),
        isym: None,
    };
    (ints, space)
}

#[test]
fn mr_typed_dead_end_skeleton_closed_loop() {
    type Case = (&'static str, &'static [u8], (usize, usize), u8, u64);
    let cases: [Case; 3] = [
        ("A", &[0, 1, 1, 1, 2], (2, 2), 2, 0x30FA),
        ("B", &[0, 0, 1, 2, 2, 3], (3, 3), 0, 0x30FB),
        ("C", &[0, 1, 1, 2, 3, 3, 3], (4, 3), 1, 0x30FC),
    ];
    for (label, table, n, target, seed) in cases {
        let (ints, space) = typed_dense(table, n, target, seed);
        let d = space.d;
        let mut h = median_refillable(&ints, space);
        let mut x = capped_start(space, 1, 2, seed + 0x100);
        let sweep = ModeRotSweepOpts {
            first_rot_iter: 0,
            ..ModeRotSweepOpts::default()
        };
        let (hist, u) = mals_moderot(&mut h, &ints, &mut x, &opts(0, 4), &sweep);
        let gates: usize = hist.sweeps.iter().map(|s| s.n_gates).sum();
        assert!(gates > 0, "{label}: vacuous — no rotation was taken");
        assert!(
            hist.u_orth_dev < 1e-12 && orthogonality_defect(&u, d) < 1e-12,
            "{label}: u drifted: {:.3e}",
            hist.u_orth_dev
        );
        let leak = cross_irrep_nonzeros(&u, space);
        assert!(leak.is_empty(), "{label}: u mixes irreps: {leak:?}");
        let rot = rotate_integrals(&ints, &u);
        let fresh = median_refillable(&rot, space);
        assert_eq!(
            h.artifact_hash(),
            fresh.artifact_hash(),
            "{label}: refilled operator ≠ fresh at u"
        );
        let rq = rayleigh_quotient(&fresh, &x);
        println!(
            "{label}: gates {gates}, λ_sweep {:.14}, RQ_fresh {rq:.14} (gap {:.2e})",
            hist.lambda,
            (rq - hist.lambda).abs()
        );
        assert!(
            (rq - hist.lambda).abs() <= 1e-10 * rq.abs(),
            "{label}: sweep energy {} vs fresh-operator energy {rq}",
            hist.lambda
        );
    }
}
