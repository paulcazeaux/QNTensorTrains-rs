// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;

use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::{
    Construction, SparseHamiltonian, TermScreen, rayleigh_quotient, switch_orbits_homogeneous,
};
use qntt::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::ordering::{irrep_blocked_order, permute_integrals};
use qntt::rotation::{expm_kappa_blocked, givens, rotate_integrals};
use qntt::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use qntt::{PointGroup, QnSpace, Sketch, TtVector};
use qntt_testkit::randomized_state;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

const ARMS: [Construction; 2] = [Construction::Median, Construction::Bipartite];

fn generic_rotation(space: QnSpace, scale: f64, rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
    let d = space.d;
    let mut kappa = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in 0..p {
            if space.pg.irrep_of(p) == space.pg.irrep_of(q) {
                let x: f64 = scale * rng.random_range(-1.0..1.0);
                kappa.set(&[p, q], x);
                kappa.set(&[q, p], -x);
            }
        }
    }
    expm_kappa_blocked(&kappa, space.pg)
}

fn classes_at_or_below_eps(ints: &Integrals<f64>, pg: PointGroup, eps: f64) -> usize {
    let d = ints.norb;
    let (t, v) = (&ints.t, &ints.v);
    let ok = |s: [usize; 4]| pg.irrep_of_sites(s) == qntt::Irrep::IDENTITY;
    let mut n = 0usize;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of_sites([i, j]) == qntt::Irrep::IDENTITY && t.get(&[i, j]).abs() <= eps {
                n += 1;
            }
        }
    }
    for i in 0..d {
        for k in i + 1..d {
            for l in 0..d {
                for j in l + 1..d {
                    let w = 0.5
                        * (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j])
                            - v.get(&[i, l, k, j])
                            - v.get(&[k, j, i, l]));
                    if ok([i, j, k, l]) && w.abs() <= eps {
                        n += 1;
                    }
                }
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let w = 0.5 * (v.get(&[i, j, k, l]) + v.get(&[k, l, i, j]));
                    if ok([i, j, k, l]) && w.abs() <= eps {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}

fn forbidden_nonzero_cells(ints: &Integrals<f64>, pg: PointGroup) -> usize {
    let d = ints.norb;
    let mut n = 0usize;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of_sites([i, j]) != qntt::Irrep::IDENTITY && ints.t.get(&[i, j]) != 0.0 {
                n += 1;
            }
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != qntt::Irrep::IDENTITY
                        && ints.v.get(&[i, j, k, l]) != 0.0
                    {
                        n += 1;
                    }
                }
            }
        }
    }
    n
}

fn h2o_typed_blocked() -> (Integrals<f64>, QnSpace) {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(Path::new("testdata/h2o_sto3g"), &opts)
        .expect("load h2o_sto3g typed");
    let (pg, d) = (load.space.pg, load.space.d);
    assert!(!pg.is_trivial(), "h2o_sto3g must resolve C2v");
    let perm = irrep_blocked_order(pg, d);
    let ints_b = permute_integrals(&load.ints, &perm);
    let space_b = ints_b
        .qn_space(Some(load.pg.kind))
        .expect("permuted dump resolves");
    for (a, &old) in perm.iter().enumerate() {
        assert_eq!(space_b.pg.irrep_of(a), pg.irrep_of(old), "irrep′({a})");
    }
    assert_eq!(space_b.target, load.space.target);
    assert_eq!(
        same_irrep_bonds(space_b),
        vec![0, 1, 2, 4],
        "blocked h2o table"
    );
    (ints_b, space_b)
}

fn same_irrep_bonds(space: QnSpace) -> Vec<usize> {
    (0..space.d - 1)
        .filter(|&k| space.pg.irrep_of(k) == space.pg.irrep_of(k + 1))
        .collect()
}

fn refill_matches_rebuild(fixture: &str, seed: u64) {
    let load = load_fixture_dir::<f64>(Path::new(fixture), &IntegralsOptions::default())
        .expect("load fixture");
    let _ = refill_matches_rebuild_on(fixture, &load.ints, load.space, seed);
}

fn refill_matches_rebuild_on(
    fixture: &str,
    ints: &Integrals<f64>,
    space: QnSpace,
    seed: u64,
) -> usize {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let u = generic_rotation(space, 0.1, &mut rng);
    let rot = rotate_integrals(ints, &u);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let n_small = classes_at_or_below_eps(&rot, space.pg, f64::EPSILON);
    println!(
        "{fixture}: d={} rotated classes with |W'| <= eps: {n_small}",
        space.d
    );
    for c in ARMS {
        let screen = TermScreen::SymmetryAllowed(space.pg);
        let mut h =
            SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, screen.clone(), c);
        let skel = h.skeleton_hash();
        h.refill(&rot.t, &rot.v, None);
        assert_eq!(h.skeleton_hash(), skel, "{c:?}: skeleton moved");
        let fresh =
            SparseHamiltonian::new_refillable(space, &rot.t, &rot.v, None, screen.clone(), c);
        assert_eq!(
            h.artifact_hash(),
            fresh.artifact_hash(),
            "{c:?}: refill ≠ fresh build"
        );
        let (q, qf) = (rayleigh_quotient(&h, &x), rayleigh_quotient(&fresh, &x));
        assert!(
            (q - qf).abs() <= 1e-14 * q.abs().max(1.0),
            "{c:?}: RQ after refill {q} vs fresh {qf}"
        );
        if n_small == 0 {
            let plain = SparseHamiltonian::new_with(space, &rot.t, &rot.v, None, f64::EPSILON, c);
            assert_eq!(
                h.artifact_hash(),
                plain.artifact_hash(),
                "{c:?}: refill ≠ plain build"
            );
        }
        let minted =
            SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, screen.clone(), c);
        h.refill(&ints.t, &ints.v, None);
        assert_eq!(
            h.artifact_hash(),
            minted.artifact_hash(),
            "{c:?}: round trip"
        );
    }
    n_small
}

#[test]
fn t8_refill_matches_rebuild_on_rotated_h2o_sto3g() {
    refill_matches_rebuild("testdata/h2o_sto3g", 0x8E_F1_A0);
}

#[test]
fn t8_refill_matches_rebuild_on_typed_irrep_blocked_h2o_sto3g() {
    let (ints, space) = h2o_typed_blocked();
    let (pg, d) = (space.pg, space.d);
    const SEED: u64 = 0x8E_F1_B6;
    let u = generic_rotation(space, 0.1, &mut Xoshiro256PlusPlus::seed_from_u64(SEED));
    let mut mix = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert!(
                    u[p * d + q] == 0.0,
                    "u[{p}][{q}] = {:e} crosses irreps",
                    u[p * d + q]
                );
            } else if p != q {
                mix = mix.max(u[p * d + q].abs());
            }
        }
    }
    assert!(
        mix > 1e-3,
        "vacuous: the blocked rotation does not mix ({mix:e})"
    );
    let rot = rotate_integrals(&ints, &u);
    assert_eq!(forbidden_nonzero_cells(&ints, pg), 0, "the screened dump");
    assert_eq!(
        forbidden_nonzero_cells(&rot, pg),
        0,
        "a blocked rotation must keep every forbidden cell an exact zero"
    );
    for c in ARMS {
        let h = SparseHamiltonian::new_refillable(
            space,
            &ints.t,
            &ints.v,
            None,
            TermScreen::SymmetryAllowed(pg),
            c,
        );
        assert_eq!(h.pattern_leak(&rot.t, &rot.v, f64::EPSILON), 0, "{c:?}");
    }
    let n_small = refill_matches_rebuild_on("h2o_sto3g C2v irrep-blocked", &ints, space, SEED);
    assert_eq!(n_small, 0, "the plain-build pin must run on the typed arm");
}

#[test]
fn t9_env_gauge_h2o_sto3g_median_exact_bipartite_refused() {
    let load = load_fixture_dir::<f64>(
        Path::new("testdata/h2o_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load fixture");
    let (ints, space) = (load.ints, load.space);
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_C9);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let screen = TermScreen::SymmetryAllowed(space.pg);
    for c in ARMS {
        let hom = switch_orbits_homogeneous(space, &ints.t, &ints.v, &screen, c);
        assert!(hom.is_ok(), "{c:?}: {hom:?}");
        let mut h =
            SparseHamiltonian::new_refillable(space, &ints.t, &ints.v, None, screen.clone(), c);
        let closed = h.env_gauge_closed();
        println!("{c:?}: orbits {hom:?}, closed {closed:?}");
        if c == Construction::Bipartite {
            assert!(
                closed.is_err(),
                "bipartite closure now holds — enable the gauge for it"
            );
            continue;
        }
        assert!(closed.is_ok(), "{c:?}: {closed:?}");
        let fr = right_to_left_sketching(&h, &x);
        for i in [0usize, 2, 4] {
            let theta = 0.3 + 0.05 * i as f64;
            let u = qntt::rotation::givens(d, i, theta);
            let g = [
                u[i * d + i],
                u[i * d + i + 1],
                u[(i + 1) * d + i],
                u[(i + 1) * d + i + 1],
            ];
            let rot = rotate_integrals(&ints, &u);
            h.refill(&rot.t, &rot.v, None);
            let fr2 = right_to_left_sketching(&h, &x);
            for j in i + 2..d {
                let mut f = fr[j - 1].clone();
                h.rotate_right_env_adjacent(x.core(j), i, g, &mut f);
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for s in f.sectors() {
                    if let (Some(a), Some(b)) = (f.block(s), fr2[j - 1].block(s)) {
                        for r in 0..a.nrows() {
                            for cc in 0..a.ncols() {
                                num += (a[(r, cc)] - b[(r, cc)]).powi(2);
                                den += b[(r, cc)].powi(2);
                            }
                        }
                    }
                }
                let rel = (num / den.max(1e-300)).sqrt();
                assert!(rel < 1e-12, "{c:?}: pair {i} bond {j}: rel {rel:.3e}");
            }
            h.refill(&ints.t, &ints.v, None);
        }
    }
}

fn sketch_rel_diff(a: &Sketch<f64>, b: &Sketch<f64>) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for s in a.sectors() {
        match (a.block(s), b.block(s)) {
            (Some(x), Some(y)) => {
                assert_eq!(
                    (x.nrows(), x.ncols()),
                    (y.nrows(), y.ncols()),
                    "shape at {s:?}"
                );
                for r in 0..x.nrows() {
                    for c in 0..x.ncols() {
                        num += (x[(r, c)] - y[(r, c)]).powi(2);
                        den += y[(r, c)].powi(2);
                    }
                }
            }
            (None, None) => {}
            _ => panic!("block presence differs at {s:?}"),
        }
    }
    (num / den.max(1e-300)).sqrt()
}

#[test]
fn t9_env_gauge_h2o_sto3g_typed_same_irrep_pairs() {
    let (ints, space) = h2o_typed_blocked();
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_CA);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let screen = TermScreen::SymmetryAllowed(space.pg);
    let hb = SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        screen.clone(),
        Construction::Bipartite,
    );
    println!("typed bipartite: closed {:?}", hb.env_gauge_closed());
    let mut h = SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        screen,
        Construction::Median,
    );
    let closed = h.env_gauge_closed();
    assert!(closed.is_ok(), "typed median skeleton: {closed:?}");
    assert!(
        (0..d - 1).any(|i| space.pg.irrep_of(i) != space.pg.irrep_of(i + 1)),
        "vacuous: no cross-irrep bond"
    );
    let hom = switch_orbits_homogeneous(
        space,
        &ints.t,
        &ints.v,
        &TermScreen::SymmetryAllowed(space.pg),
        Construction::Median,
    );
    assert!(
        matches!(hom, Ok(n) if n > 0),
        "typed median switch orbits: {hom:?}"
    );
    let fr = right_to_left_sketching(&h, &x);
    let fl = left_to_right_sketching(&h, &x);
    let pairs = same_irrep_bonds(space);
    assert!(pairs.len() >= 3, "vacuous: {pairs:?}");
    let (mut n_right, mut n_left, mut moved_right, mut moved_left) = (0usize, 0, 0, 0);
    for &i in &pairs {
        let theta = 0.3 + 0.05 * i as f64;
        let u = givens(d, i, theta);
        let g = [
            u[i * d + i],
            u[i * d + i + 1],
            u[(i + 1) * d + i],
            u[(i + 1) * d + i + 1],
        ];
        let rot = rotate_integrals(&ints, &u);
        assert_eq!(forbidden_nonzero_cells(&rot, space.pg), 0, "pair {i}");
        h.refill(&rot.t, &rot.v, None);
        let fr2 = right_to_left_sketching(&h, &x);
        let fl2 = left_to_right_sketching(&h, &x);
        for j in i + 2..d {
            let mut f = fr[j - 1].clone();
            h.rotate_right_env_adjacent(x.core(j), i, g, &mut f);
            let rel = sketch_rel_diff(&f, &fr2[j - 1]);
            assert!(rel < 1e-12, "right env pair {i} bond {j}: rel {rel:.3e}");
            n_right += 1;
            moved_right += (sketch_rel_diff(&fr[j - 1], &fr2[j - 1]) > 1e-8) as usize;
        }
        for k in 1..=i {
            let mut f = fl[k].clone();
            h.rotate_left_env_adjacent(x.core(k - 1), i, g, &mut f);
            let rel = sketch_rel_diff(&f, &fl2[k]);
            assert!(rel < 1e-12, "left env pair {i} bond {k}: rel {rel:.3e}");
            n_left += 1;
            moved_left += (sketch_rel_diff(&fl[k], &fl2[k]) > 1e-8) as usize;
        }
        h.refill(&ints.t, &ints.v, None);
    }
    println!(
        "pairs {pairs:?}: right {n_right} cases ({moved_right} moved), left {n_left} ({moved_left} moved)"
    );
    assert!(
        moved_right > 0 && moved_left > 0,
        "vacuous: the gauge moved nothing on a stack"
    );
}
