// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::PathBuf;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::su2::core::Family;
use qntt::su2::hess::weingarten_operator;
use qntt::su2::qn::Su2Space;
use qntt::su2::tangent::{Su2TtTangent, project_irrep};
use qntt::su2::vector::{Su2TtVector, apply_group_element, symmetry_defect, tt_csf, tt_randn};

fn testdata(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(sub)
}

fn fixture() -> (PointGroup, Su2Space, qntt::Integrals<f64>) {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(&testdata("lih_sto3g"), &opts).expect("lih_sto3g");
    let space = Su2Space::new(load.ints.norb, load.ints.nelec, 0);
    (load.pg.pg, space, load.ints)
}

fn random_state(space: Su2Space, seed: u64) -> Su2TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let caps = qntt::su2::qn::max_ranks(&space);
    let ranks: Vec<qntt::su2::qn::Su2RankGrid> = (0..=space.d)
        .map(|b| {
            let mut g = qntt::su2::qn::Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                g[s] = (caps[b][s] as usize).min(if b == 0 || b == space.d { 1 } else { 4 });
            }
            g
        })
        .collect();
    tt_randn(space, &ranks, &mut rng)
}

fn converged_base() -> (
    PointGroup,
    Su2Space,
    qntt::su2::hamiltonian::Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
) {
    let (pg, space, ints) = fixture();
    let ham = qntt::su2::hamiltonian::Su2SparseHamiltonian::<f64>::new(
        space,
        &ints.t,
        &ints.v,
        None,
        f64::EPSILON,
    );
    let (_, x) = qntt::su2::solvers::mals::mals(
        &ham,
        &random_state(space, 0x5EED),
        &qntt::solvers::mals::MalsOptions {
            eps: 1e-10,
            bond_dims: vec![12],
            max_iter: 60,
            ..Default::default()
        },
    );
    let mut x = x;
    x.round_mut(1e-8, qntt::Pooled::NONE, qntt::RoundMode::Rel);
    qntt::su2::riemannian::normalize_mut(&mut x);

    let (_, defects) = symmetry_defect(&x, pg);
    let worst = defects.iter().cloned().fold(0.0f64, f64::max);
    let smin = weingarten_operator(&ham, &x)
        .bond_gauges()
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
        .fold(f64::INFINITY, f64::min);
    let dropped: f64 = weingarten_operator(&ham, &x)
        .bond_gauges()
        .iter()
        .map(|g| g.dropped_mass)
        .sum();
    println!("base: symmetry defect {worst:.2e}, sigma_min {smin:.2e}, dropped mass {dropped:.2e}");
    assert!(
        worst <= 1e-6,
        "the converged ground state is symmetry-broken at {worst:e}; the projector precondition \
         does not hold and the tests below would be meaningless"
    );
    (pg, space, ham, x)
}

#[test]
fn pgr_group_action_is_an_exact_involution() {
    let (pg, space, _) = fixture();
    assert!(pg.order() > 1, "fixture must carry a real point group");
    let x = random_state(space, 0xA11);
    for h in 0..pg.order() {
        let h = Irrep::new(h);
        let ux = apply_group_element(&x, pg, h);
        let uux = apply_group_element(&ux, pg, h);
        assert_eq!(x.ranks, ux.ranks, "U_h changed the rank grid (h = {h:?})");
        for k in 0..space.d {
            let (a, b) = (x.core(k), uux.core(k));
            for f in Family::ALL {
                for &l in &space.bond_sectors(k) {
                    match (a.block(f, l), b.block(f, l)) {
                        (Some(ma), Some(mb)) => {
                            for j in 0..ma.ncols() {
                                for i in 0..ma.nrows() {
                                    assert_eq!(ma[(i, j)], mb[(i, j)], "U_h∘U_h ≠ id");
                                }
                            }
                        }
                        (None, None) => {}
                        _ => panic!("U_h changed block presence at site {k}"),
                    }
                }
            }
        }
        assert!(
            (ux.norm() - x.norm()).abs() <= 1e-14 * x.norm(),
            "U_h changed the norm"
        );
    }
}

#[test]
fn pgr_group_action_is_faithful_on_the_fixture() {
    let (pg, space, _) = fixture();
    let x = random_state(space, 0xFA1);
    let id = apply_group_element(&x, pg, Irrep::IDENTITY);
    assert!(
        (&id - &x).norm() <= 1e-14 * x.norm(),
        "identity must act trivially"
    );
    let moved = (1..pg.order())
        .any(|h| (&apply_group_element(&x, pg, Irrep::new(h)) - &x).norm() > 1e-6 * x.norm());
    assert!(
        moved,
        "no group element moved the state — ORBSYM is effectively trivial"
    );
}

#[test]
fn pgr_closed_shell_csf_is_totally_symmetric() {
    let (pg, space, _) = fixture();
    let nocc = space.n / 2;
    let codes: Vec<Family> = (0..space.d)
        .map(|i| if i < nocc { Family::Occ } else { Family::Empty })
        .collect();
    let csf: Su2TtVector<f64> = tt_csf(&codes);
    let (gamma, defects) = symmetry_defect(&csf, pg);
    assert_eq!(
        gamma,
        Irrep::IDENTITY,
        "a closed-shell CSF must be totally symmetric"
    );
    for (j, d) in defects.iter().enumerate() {
        assert!(*d <= 1e-14, "generator {j}: defect {d:e} is not roundoff");
    }
}

#[test]
fn pgr_irrep_projectors_resolve_the_identity() {
    let (pg, space, ham, x) = converged_base();
    let ctx = weingarten_operator(&ham, &x);
    let (base_l, base_r) = {
        let (l, r) = ctx.bases();
        (l.clone(), r.clone())
    };
    let dir = random_state(space, 0xD12);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
    let n = xi.norm();
    assert!(n > 0.0, "degenerate test direction");

    let mut sum: Option<Su2TtTangent<f64>> = None;
    let mut weights = Vec::new();
    for ir in 0..pg.order() {
        let (p, _) = project_irrep(&xi, &base_l, &base_r, pg, Irrep::new(ir));
        weights.push((p.norm() / n).powi(2));
        sum = Some(match sum {
            None => p,
            Some(a) => a.add(&p),
        });
    }
    let resid = xi.sub(&sum.unwrap()).norm() / n;
    assert!(resid <= 1e-10, "Σ_Γ P_Γ ≠ id: residual {resid:e}");

    let total: f64 = weights.iter().sum();
    assert!(
        (total - 1.0).abs() <= 1e-10,
        "Σ_Γ ‖P_Γ ξ‖²/‖ξ‖² = {total}, so the blocks are not orthogonal"
    );

    let live = weights.iter().filter(|w| **w > 1e-6).count();
    assert!(
        live > 1,
        "a random tangent direction landed in one irrep only: {weights:?}"
    );
}

#[test]
fn pgr_irrep_projector_is_idempotent() {
    let (pg, space, ham, x) = converged_base();
    let ctx = weingarten_operator(&ham, &x);
    let (base_l, base_r) = {
        let (l, r) = ctx.bases();
        (l.clone(), r.clone())
    };
    let dir = random_state(space, 0xD13);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);

    for ir in 0..pg.order() {
        let gamma = Irrep::new(ir);
        let (p, _) = project_irrep(&xi, &base_l, &base_r, pg, gamma);
        if p.norm() <= 1e-12 * xi.norm() {
            continue;
        }
        let (pp, leak) = project_irrep(&p, &base_l, &base_r, pg, gamma);
        assert!(
            p.sub(&pp).norm() <= 1e-10 * p.norm(),
            "P_Γ not idempotent for Γ = {gamma:?}"
        );
        assert!(
            leak <= 1e-10,
            "leakage {leak:e} on an already-pure vector (Γ = {gamma:?})"
        );
    }
}

#[test]
fn pgr_symmetry_breaking_degrades_the_resolution() {
    let (pg, space, ints) = fixture();
    let ham = qntt::su2::hamiltonian::Su2SparseHamiltonian::<f64>::new(
        space,
        &ints.t,
        &ints.v,
        None,
        f64::EPSILON,
    );

    let broken = random_state(space, 0x5EED);
    let (_, defects) = symmetry_defect(&broken, pg);
    let worst = defects.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        worst > 1e-3,
        "the random state was accidentally symmetry-pure ({worst:e})"
    );

    let ctx = weingarten_operator(&ham, &broken);
    let (bl, br) = {
        let (l, r) = ctx.bases();
        (l.clone(), r.clone())
    };
    let dir = random_state(space, 0xD12);
    let xi = Su2TtTangent::project(&bl, &br, &dir);
    let n = xi.norm();
    let total: f64 = (0..pg.order())
        .map(|ir| (project_irrep(&xi, &bl, &br, pg, Irrep::new(ir)).0.norm() / n).powi(2))
        .sum();
    assert!(
        (total - 1.0).abs() > 1e-6,
        "Σ_Γ ‖P_Γ ξ‖²/‖ξ‖² = {total} at a symmetry-BROKEN base point — the diagnostic has no teeth"
    );
    println!(
        "symmetry defect {worst:.2e} -> resolution Σ = {total:.6} (1 − Σ = {:.2e})",
        1.0 - total
    );
}
