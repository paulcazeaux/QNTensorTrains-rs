// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeSet;
use std::sync::OnceLock;

use qntt::core::factor::dense_self_adjoint_smallest;
use qntt::core::tangent::{TtTangent, project_irrep, tangent_basis};
use qntt::core::weingarten::{FloorPolicy, bond_gauges};
use qntt::elementaryops::{apply_cross_target, property_flux_classes};
use qntt::integrals::{PropertyOperator, property_hamiltonian};
use qntt::rotation::{PairMask, orbital_pairs};
use qntt::solvers::InnerProductVector;
use qntt::solvers::hess::{SymplecticDavidsonOpts, SymplecticSpectrum};
use qntt::solvers::joint::{
    GramRoute, JointContext, JointCtxBuild, JointGateError, JointOpts, JointScfOpts, JointScfOut,
    JointVector, TransitionMomentOpts, joint_context, joint_context_with, joint_ground_state,
    joint_irrep_basis, joint_irrep_reference, joint_lowest_davidson_irrep, transition_moments,
};
use qntt::solvers::riemannian::normalize_mut;
use qntt::vector::{apply_group_element, symmetry_defect};
use qntt::{
    CoeffTensor, Family, Integrals, Irrep, PointGroup, QnSpace, Sector, SparseHamiltonian, TtVector,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::capped_randn;
use crate::util::*;

fn decode_index(d: usize, idx: usize) -> (Vec<bool>, Vec<bool>) {
    let mut digits = idx;
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for k in 0..d {
        let fam = digits & 3;
        digits >>= 2;
        occ_up[k] = fam & 1 != 0;
        occ_dn[k] = fam & 2 != 0;
    }
    (occ_up, occ_dn)
}

fn det_irrep(pg: PointGroup, occ_up: &[bool], occ_dn: &[bool]) -> Irrep {
    let d = occ_up.len();
    (0..d).fold(Irrep::IDENTITY, |acc, k| {
        acc.fuse(Family::from_occupation(occ_up[k], occ_dn[k]).flux(pg.irrep_of(k)))
    })
}

fn dense_group_action(d: usize, pg: PointGroup, h: Irrep, v: &[f64]) -> Vec<f64> {
    v.iter()
        .enumerate()
        .map(|(idx, &c)| {
            let (ou, od) = decode_index(d, idx);
            c * det_irrep(pg, &ou, &od).character(h)
        })
        .collect()
}

fn dense_sector_vector(dets: &[(Vec<bool>, Vec<bool>)], full: &[f64]) -> Vec<f64> {
    dets.iter()
        .map(|(ou, od)| full[basis_index(ou, od)])
        .collect()
}

fn pure_reference(x: &TtVector<f64>, pg: PointGroup, gamma0: Irrep) -> TtVector<f64> {
    let mut acc: Option<TtVector<f64>> = None;
    for h in 0..pg.order() {
        let h = Irrep::new(h);
        let term = &apply_group_element(x, pg, h) * gamma0.character(h);
        acc = Some(match acc {
            None => term,
            Some(a) => &a + &term,
        });
    }
    let mut out = acc.expect("pg.order() >= 1");
    normalize_mut(&mut out);
    out
}

fn sigma_min_at(base_l: &TtVector<f64>, base_r: &TtVector<f64>) -> f64 {
    bond_gauges(base_l, base_r, FloorPolicy::default_hard())
        .iter()
        .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
        .fold(f64::INFINITY, f64::min)
}

struct PgFixture {
    d: usize,
    pg: PointGroup,
    space_c1: QnSpace,
    gamma0: Irrep,
}

fn pg_fixture() -> &'static PgFixture {
    static CELL: OnceLock<PgFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        let d = 4usize;
        let pg = PointGroup::new(
            2,
            &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
        );
        let space_c1 = QnSpace::new(d, 2, 1);
        PgFixture {
            d,
            pg,
            space_c1,
            gamma0: Irrep::IDENTITY,
        }
    })
}

#[test]
fn tji_pg_group_action_dense_pin() {
    let fx = pg_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE100);
    let x: TtVector<f64> = randomized_state(fx.space_c1, 1, 2, &mut rng);
    let x_dense = x.to_dense();

    for h in 0..fx.pg.order() {
        let h = Irrep::new(h);
        let ux = apply_group_element(&x, fx.pg, h);
        assert_eq!(
            ux.ranks, x.ranks,
            "apply_group_element must be rank-preserving (h={h:?})"
        );
        let ux_dense = ux.to_dense();
        let want = dense_group_action(fx.d, fx.pg, h, &x_dense);
        for i in 0..x_dense.len() {
            assert_close(
                ux_dense[i],
                want[i],
                1e-12 * x_dense.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0),
                &format!("apply_group_element vs dense character action, h={h:?}, idx={i}"),
            );
        }
    }

    for h in 0..fx.pg.order() {
        let h = Irrep::new(h);
        let uux = apply_group_element(&apply_group_element(&x, fx.pg, h), fx.pg, h);
        for k in 0..fx.d {
            let (a, b) = (x.core(k), uux.core(k));
            for f in Family::ALL {
                for &l in &fx.space_c1.bond_sectors(k) {
                    match (a.block(f, l), b.block(f, l)) {
                        (Some(ma), Some(mb)) => {
                            for j in 0..ma.ncols() {
                                for i in 0..ma.nrows() {
                                    assert_eq!(ma[(i, j)], mb[(i, j)], "U_h∘U_h != id (h={h:?})");
                                }
                            }
                        }
                        (None, None) => {}
                        _ => panic!("U_h changed block presence at site {k} (h={h:?})"),
                    }
                }
            }
        }
    }

    let uid = apply_group_element(&x, PointGroup::TRIVIAL, Irrep::IDENTITY);
    let diff = (&uid - &x).norm();
    assert!(
        diff <= 1e-14 * x.norm().max(1.0),
        "TRIVIAL group must act as a clone: {diff:e}"
    );

    let moved = (1..fx.pg.order())
        .any(|h| (&apply_group_element(&x, fx.pg, Irrep::new(h)) - &x).norm() > 1e-6 * x.norm());
    assert!(moved, "anti-vacuity: no group element moved x");
}

#[test]
fn tji_pg_defect_structural_vs_broken() {
    let fx = pg_fixture();

    let space_g = QnSpace::with_point_group(fx.d, 2, 1, fx.pg, fx.gamma0);
    let x_g = capped_randn(space_g, 1, 2, 0xE101);
    let (gamma_g, defects_g) = symmetry_defect(&x_g, fx.pg);
    assert_eq!(
        gamma_g, fx.gamma0,
        "a |G|>1-space train must read back its own space.target"
    );
    let worst_g = defects_g.iter().cloned().fold(0.0f64, f64::max);
    println!("tji_pg_defect_structural_vs_broken: |G|>1 defect = {worst_g:e} (structural)");
    assert!(
        worst_g <= 1e-10,
        "a |G|>1-space train must be EXACTLY symmetric: {worst_g:e}"
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE102);
    let x_c1: TtVector<f64> = randomized_state(fx.space_c1, 1, 2, &mut rng);
    let (_gamma_c1, defects_c1) = symmetry_defect(&x_c1, fx.pg);
    let worst_c1 = defects_c1.iter().cloned().fold(0.0f64, f64::max);
    println!("tji_pg_defect_structural_vs_broken: random C1 defect = {worst_c1:e}");
    assert!(
        worst_c1 > 1e-3,
        "a deliberately mixed C1 state should be broken well above the H7 1e-3 scale: {worst_c1:e}"
    );
}

struct JointPgFixture {
    pg: PointGroup,
    space: QnSpace,
    ham: SparseHamiltonian<f64>,
    x: TtVector<f64>,
    gamma0: Irrep,
}

fn pg_symmetric_tv(d: usize, pg: PointGroup) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let (mut t, mut v) = generic_truncated_tv(d);
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                t.set(&[p, q], 0.0);
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                        v.set(&[i, j, k, l], 0.0);
                    }
                }
            }
        }
    }
    for p in 0..d {
        for q in 0..d {
            if p != q && pg.irrep_of(p) == pg.irrep_of(q) {
                t.set(&[p, q], -0.8 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    (t, v)
}

fn joint_pg_fixture() -> &'static JointPgFixture {
    static CELL: OnceLock<JointPgFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        let fx = pg_fixture();
        let (t, v) = pg_symmetric_tv(fx.d, fx.pg);
        let ham = SparseHamiltonian::new(fx.space_c1, &t, &v, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE200);
        let x_rand: TtVector<f64> = randomized_state(fx.space_c1, 1, 3, &mut rng);
        let x = pure_reference(&x_rand, fx.pg, fx.gamma0);
        JointPgFixture {
            pg: fx.pg,
            space: fx.space_c1,
            ham,
            x,
            gamma0: fx.gamma0,
        }
    })
}

fn diagnosis_only_jctx(fx: &'static JointPgFixture) -> JointContext<'static> {
    let jopts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        pairs: Some(PairMask {
            keep: orbital_pairs(fx.pg, fx.space.d),
        }),
        ..JointOpts::default()
    };
    joint_context(&fx.ham, &fx.x, &jopts).expect("diagnosis-only context must always gate")
}

fn irrep_graded_tangent_basis(
    jctx: &JointContext<'_>,
    pg: PointGroup,
) -> Vec<(Irrep, TtTangent<f64>)> {
    let (base_l, base_r) = jctx.bases();
    let tb = tangent_basis(base_l, base_r, true);
    let mut out: Vec<(Irrep, TtTangent<f64>)> = Vec::new();
    for h in 0..pg.order() {
        let gamma = Irrep::new(h);
        let mut block: Vec<TtTangent<f64>> = Vec::new();
        for xi in &tb {
            let (proj, _leak) = project_irrep(xi, base_l, base_r, pg, gamma);
            let mut v = proj;
            for b in &block {
                let c = b.dot(&v);
                v.axpy(-c, b);
            }
            let n = v.norm();
            if n > 1e-8 {
                v.scale_mut(1.0 / n);
                block.push(v);
            }
        }
        out.extend(block.into_iter().map(|v| (gamma, v)));
    }
    out
}

#[test]
fn tji_perirrep_gram_blocks_vanish() {
    let fx = joint_pg_fixture();
    let jctx = diagnosis_only_jctx(fx);
    let (base_l, base_r) = jctx.bases();
    let sigma_min = sigma_min_at(base_l, base_r);
    let (gamma_x, defects_x) = symmetry_defect(&fx.x, fx.pg);
    let worst_defect = defects_x.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "tji_perirrep_gram_blocks_vanish: base point defect={worst_defect:e} \
         (gamma={gamma_x:?}) sigma_min={sigma_min:e}"
    );
    let pieces = irrep_graded_tangent_basis(&jctx, fx.pg);
    assert!(
        pieces.len() >= 4,
        "anti-vacuity: need a genuinely multi-dimensional tangent space"
    );

    let mut vecs: Vec<JointVector> = pieces
        .iter()
        .map(|(_, xi)| JointVector::from_tangent(jctx.spec(), xi.clone()))
        .collect();
    let n_eff = jctx.n_eff();
    let mut tags: Vec<Irrep> = pieces.iter().map(|(g, _)| *g).collect();
    for j in 0..n_eff {
        let mut c = vec![0.0f64; n_eff];
        c[j] = 1.0;
        vecs.push(JointVector::from_coeffs(jctx.spec(), &jctx, c));
        tags.push(fx.gamma0);
    }
    if n_eff == 0 {
        println!(
            "tji_perirrep_gram_blocks_vanish: WARNING — n_eff=0 at this fixture, the orbital-row \
             fold is VACUOUS (0 seeds folded in). The orbital ⊂ Gamma0 claim through the joint \
             Gram route is unexercised here; see tji_orbital_in_gamma0 for the raw-tangent pin."
        );
    }

    let n = vecs.len();
    let (a, b) = jctx.joint_hessian_grams(&vecs);
    let max_a = a
        .iter()
        .fold(0.0f64, |acc, &v| acc.max(v.abs()))
        .max(1e-300);
    let max_b = b
        .iter()
        .fold(0.0f64, |acc, &v| acc.max(v.abs()))
        .max(1e-300);
    let mut worst_a = 0.0f64;
    let mut worst_b = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if tags[i] != tags[j] {
                worst_a = worst_a.max(a[i * n + j].abs());
                worst_b = worst_b.max(b[i * n + j].abs());
            }
        }
    }
    let scale = max_a.max(max_b);
    println!(
        "tji_perirrep_gram_blocks_vanish: n={n} max_a={max_a:e} max_b={max_b:e} \
         worst_a/scale={:e} worst_b/scale={:e}",
        worst_a / scale,
        worst_b / scale
    );
    assert!(
        max_a > 1e-6,
        "anti-vacuity: the A Gram is itself at roundoff ({max_a:e}) — nothing is being tested"
    );
    assert!(
        worst_a <= 1e-6 * scale,
        "A cross-irrep block did not vanish: {worst_a:e} (scale={scale:e})"
    );
    assert!(
        worst_b <= 1e-6 * scale,
        "B cross-irrep block did not vanish: {worst_b:e} (scale={scale:e})"
    );
    if max_b <= 1e-12 * max_a {
        println!(
            "tji_perirrep_gram_blocks_vanish: WARNING — the B arm is VACUOUS at this fixture \
             (max_b={max_b:e} vs max_a={max_a:e}, i.e. B === 0). The cross-irrep claim is carried \
             by the A block alone here; see this test's own comment for the measured cause."
        );
    }

    let distinct: BTreeSet<u8> = tags.iter().map(|g| g.index()).collect();
    assert!(
        distinct.len() > 1,
        "anti-vacuity: only one irrep present in the graded basis"
    );
}

#[test]
fn tji_perirrep_union() {
    let fx = joint_pg_fixture();
    let jctx = diagnosis_only_jctx(fx);
    let (base_l, base_r) = jctx.bases();
    let sigma_min = sigma_min_at(base_l, base_r);
    let (gamma_x, defects_x) = symmetry_defect(&fx.x, fx.pg);
    let worst_defect = defects_x.iter().cloned().fold(0.0f64, f64::max);
    println!(
        "tji_perirrep_union: base point defect={worst_defect:e} (gamma={gamma_x:?}) \
         sigma_min={sigma_min:e}"
    );
    let pieces = irrep_graded_tangent_basis(&jctx, fx.pg);
    assert!(
        pieces.len() >= 4,
        "anti-vacuity: need a genuinely multi-dimensional tangent space"
    );

    let vecs: Vec<JointVector> = pieces
        .iter()
        .map(|(_, xi)| JointVector::from_tangent(jctx.spec(), xi.clone()))
        .collect();
    let tags: Vec<Irrep> = pieces.iter().map(|(g, _)| *g).collect();
    let n = vecs.len();
    let (a, _b) = jctx.joint_hessian_grams(&vecs);

    let (evals_full, _) = dense_self_adjoint_smallest::<f64>(&a, n, n);

    let mut per_irrep_evals: Vec<f64> = Vec::new();
    for h in 0..fx.pg.order() {
        let gamma = Irrep::new(h);
        let idx: Vec<usize> = (0..n).filter(|&i| tags[i] == gamma).collect();
        if idx.is_empty() {
            continue;
        }
        let m = idx.len();
        let mut a_blk = vec![0.0f64; m * m];
        for (bi, &i) in idx.iter().enumerate() {
            for (bj, &j) in idx.iter().enumerate() {
                a_blk[bi * m + bj] = a[i * n + j];
            }
        }
        let (evals_blk, _) = dense_self_adjoint_smallest::<f64>(&a_blk, m, m);
        per_irrep_evals.extend(evals_blk);
    }
    per_irrep_evals.sort_by(|x, y| x.partial_cmp(y).unwrap());

    assert_eq!(
        per_irrep_evals.len(),
        evals_full.len(),
        "union of per-irrep block sizes must equal the full basis size"
    );
    let scale = evals_full
        .iter()
        .fold(0.0f64, |acc, &v| acc.max(v.abs()))
        .max(1.0);
    let mut worst = 0.0f64;
    for (u, f) in per_irrep_evals.iter().zip(evals_full.iter()) {
        worst = worst.max((u - f).abs());
    }
    println!("tji_perirrep_union: n={n} worst_eigenvalue_diff={worst:e} scale={scale:e}");
    assert!(
        worst <= 1e-4 * scale,
        "union of per-irrep A-spectra must match the full A-spectrum: worst diff {worst:e}"
    );
}

#[test]
fn tji_joint_irrep_basis_matches_dense() {
    let fx = joint_pg_fixture();
    let jctx = diagnosis_only_jctx(fx);

    let iref = joint_irrep_reference(&jctx, fx.pg);
    println!(
        "tji_joint_irrep_basis_matches_dense: reference gamma0={:?} worst_defect={:e} \
         sigma_min={:e}",
        iref.gamma0, iref.worst_defect, iref.sigma_min
    );
    assert_eq!(
        iref.gamma0, fx.gamma0,
        "joint_irrep_reference must find the same gamma0 the fixture was purified to"
    );
    assert!(
        iref.worst_defect <= 1e-8,
        "reference must be symmetry-pure: {:e}",
        iref.worst_defect
    );

    let mut vecs: Vec<JointVector> = Vec::new();
    let mut tags: Vec<Irrep> = Vec::new();
    for h in 0..fx.pg.order() {
        let gamma = Irrep::new(h);
        let (basis, report) = joint_irrep_basis(&jctx, fx.pg, gamma, fx.gamma0);
        println!(
            "tji_joint_irrep_basis_matches_dense: gamma={:?} kept={} dropped={} \
             max_leakage={:e} sigma_min={:e} min_closure={:e} orbital_kept={}",
            report.gamma,
            report.kept,
            report.dropped,
            report.max_leakage,
            report.sigma_min,
            report.min_closure,
            report.orbital_kept
        );
        assert_eq!(
            report.orbital_kept,
            gamma == fx.gamma0,
            "orbital_kept must be structural: gamma==gamma0 exactly"
        );
        for v in basis {
            tags.push(gamma);
            vecs.push(v);
        }
    }
    assert!(
        vecs.len() >= 4,
        "anti-vacuity: need a genuinely multi-dimensional joint basis"
    );

    let n = vecs.len();
    let mut worst_orth = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let dij = vecs[i].dot(&vecs[j]);
            let want = if i == j { 1.0 } else { 0.0 };
            worst_orth = worst_orth.max((dij - want).abs());
        }
    }
    println!("tji_joint_irrep_basis_matches_dense: n={n} worst_orth_error={worst_orth:e}");
    assert!(
        worst_orth <= 1e-8,
        "joint_irrep_basis output is not joint-orthonormal: {worst_orth:e}"
    );

    let (a, _b) = jctx.joint_hessian_grams(&vecs);
    let (evals_full, _) = dense_self_adjoint_smallest::<f64>(&a, n, n);
    let mut per_irrep_evals: Vec<f64> = Vec::new();
    for h in 0..fx.pg.order() {
        let gamma = Irrep::new(h);
        let idx: Vec<usize> = (0..n).filter(|&i| tags[i] == gamma).collect();
        if idx.is_empty() {
            continue;
        }
        let m = idx.len();
        let mut a_blk = vec![0.0f64; m * m];
        for (bi, &i) in idx.iter().enumerate() {
            for (bj, &j) in idx.iter().enumerate() {
                a_blk[bi * m + bj] = a[i * n + j];
            }
        }
        let (evals_blk, _) = dense_self_adjoint_smallest::<f64>(&a_blk, m, m);
        per_irrep_evals.extend(evals_blk);
    }
    per_irrep_evals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(
        per_irrep_evals.len(),
        evals_full.len(),
        "union of per-irrep block sizes must equal the full basis size"
    );
    let scale = evals_full
        .iter()
        .fold(0.0f64, |acc, &v| acc.max(v.abs()))
        .max(1.0);
    let mut worst = 0.0f64;
    for (u, f) in per_irrep_evals.iter().zip(evals_full.iter()) {
        worst = worst.max((u - f).abs());
    }
    println!(
        "tji_joint_irrep_basis_matches_dense: worst_eigenvalue_diff={worst:e} scale={scale:e}"
    );
    assert!(
        worst <= 1e-4 * scale,
        "union via joint_irrep_basis must match the full A-spectrum: worst diff {worst:e}"
    );
}

struct JointCriticalPgFixture {
    pg: PointGroup,
    gamma0: Irrep,
    out: JointScfOut,
}

fn joint_critical_pg_fixture() -> &'static JointCriticalPgFixture {
    static CELL: OnceLock<JointCriticalPgFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        let d = 4usize;
        let pg = PointGroup::new(
            2,
            &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
        );
        let gamma0 = Irrep::IDENTITY;
        let space = QnSpace::new(d, 2, 1);
        let (t, v) = pg_symmetric_tv(d, pg);
        let ints = Integrals::<f64> {
            t,
            v,
            e0: 0.0,
            norb: d,
            nelec: 3,
            ms2: 1,
            orbsym: Vec::new(),
            isym: None,
        };
        let x_rand = capped_randn(space, 1, 1, 0xE300);
        let x0 = pure_reference(&x_rand, pg, gamma0);
        let opts = JointScfOpts {
            orb_max_iter: 40,
            tt_grad_tol: 1e-6,
            pairs: Some(PairMask {
                keep: orbital_pairs(pg, d),
            }),
            ..JointScfOpts::default()
        };
        let out = joint_ground_state(&ints, space, &x0, &opts);
        JointCriticalPgFixture { pg, gamma0, out }
    })
}

fn joint_critical_pg_jctx(fx: &'static JointCriticalPgFixture) -> JointContext<'static> {
    assert!(
        fx.out.info.converged,
        "pg joint fixture must converge: {:?}",
        fx.out.info.refused_reason
    );
    let (gamma_x, defects) = symmetry_defect(&fx.out.x, fx.pg);
    let worst = defects.iter().cloned().fold(0.0f64, f64::max);
    println!("joint_critical_pg_fixture: endpoint gamma={gamma_x:?} worst_defect={worst:e}");
    assert_eq!(gamma_x, fx.gamma0, "endpoint must stay Gamma0-pure");
    assert!(
        worst <= 1e-8,
        "endpoint symmetry defect too large: {worst:e}"
    );
    let jopts = JointOpts {
        pairs: Some(PairMask {
            keep: orbital_pairs(fx.pg, fx.out.x.space.d),
        }),
        ..JointOpts::default()
    };
    joint_context(&fx.out.h, &fx.out.x, &jopts).expect("converged endpoint must be joint-critical")
}

#[test]
fn tji_davidson_matches_dense_gamma0() {
    let fx = joint_critical_pg_fixture();
    let jctx = joint_critical_pg_jctx(fx);
    println!(
        "tji_davidson_matches_dense_gamma0: n_eff={} n_tt={}",
        jctx.n_eff(),
        jctx.joint_basis().0.len() - jctx.n_eff()
    );

    let (basis_g0, report) = joint_irrep_basis(&jctx, fx.pg, fx.gamma0, fx.gamma0);
    println!(
        "tji_davidson_matches_dense_gamma0: gamma0 kept={} dropped={} max_leakage={:e} \
         sigma_min={:e}",
        report.kept, report.dropped, report.max_leakage, report.sigma_min
    );
    assert!(
        basis_g0.len() >= 2,
        "anti-vacuity: the Gamma0 block must be multi-dimensional"
    );

    let SymplecticSpectrum::Stable { omega: omega_dense } =
        jctx.joint_symplectic_eigenvalues(&basis_g0)
    else {
        panic!("expected a stable Gamma0 sub-spectrum at the converged reference");
    };

    let k = 2usize.min(omega_dense.len());
    let opts = SymplecticDavidsonOpts::default();
    let (omega, info, filter) =
        joint_lowest_davidson_irrep(&jctx, &basis_g0, k, fx.pg, fx.gamma0, fx.gamma0, &opts)
            .expect("Gamma0 joint Davidson must not error");
    println!(
        "tji_davidson_matches_dense_gamma0: omega={omega:?} dense={:?} converged={} \
         filter_report={filter:?}",
        &omega_dense[..k],
        info.converged
    );
    assert!(
        info.converged,
        "Gamma0 joint Davidson must converge: {:?}",
        info.residuals
    );
    assert_eq!(omega.len(), k);
    for (i, &w) in omega.iter().enumerate() {
        let want = omega_dense[i];
        assert!(
            (w - want).abs() <= 1e-8 * want.max(1.0),
            "Gamma0 omega[{i}]: Davidson {w} vs dense {want}"
        );
    }
}

struct CrossTargetFixture {
    pg: PointGroup,
    space: QnSpace,
    x: TtVector<f64>,
}

fn cross_target_fixture() -> &'static CrossTargetFixture {
    static CELL: OnceLock<CrossTargetFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        let fx = pg_fixture();
        let space = QnSpace::with_point_group(fx.d, 2, 1, fx.pg, fx.gamma0);
        let x = capped_randn(space, 1, 2, 0xE300);
        CrossTargetFixture {
            pg: fx.pg,
            space,
            x,
        }
    })
}

#[test]
fn tji_cross_target_dense_pin() {
    let fx = cross_target_fixture();
    let (p, q) = (0usize, 1usize);
    assert_ne!(
        fx.pg.irrep_of(p),
        fx.pg.irrep_of(q),
        "fixture precondition: sites 0,1 must carry different irreps"
    );
    let mut m = CoeffTensor::<f64>::zeros(fx.space.d, 2);
    m.set(&[p, q], 1.0);
    m.set(&[q, p], 1.0);

    let classes = property_flux_classes(&m, fx.pg);
    assert_eq!(
        classes.len(),
        1,
        "a single cross-irrep pair must give exactly one flux class"
    );
    let gamma_mu = classes[0];
    assert_eq!(gamma_mu, fx.pg.irrep_of(p).fuse(fx.pg.irrep_of(q)));
    assert_ne!(
        gamma_mu,
        Irrep::IDENTITY,
        "fixture must be genuinely cross-irrep"
    );

    let y = apply_cross_target(&fx.x, &m, fx.pg);
    assert_eq!(y.space.pg, fx.pg);
    assert_eq!(y.space.d, fx.space.d);
    assert_eq!(y.space.nup, fx.space.nup);
    assert_eq!(y.space.ndn, fx.space.ndn);
    assert_eq!(
        y.space.target,
        fx.space.target.fuse(gamma_mu),
        "output target must be Gamma0 (x)Gamma_mu"
    );
    assert!(
        y.check(),
        "apply_cross_target output must satisfy the boundary law"
    );
    assert_eq!(
        y.rank(
            y.space.d,
            Sector::with_irrep(y.space.nup, y.space.ndn, y.space.target)
        ),
        1,
        "boundary rank at bond d must be exactly 1"
    );

    let dets = dets_for(fx.space.d, fx.space.nup, fx.space.ndn);
    let x_full = fx.x.to_dense();
    let y_full = y.to_dense();
    let x_sector = dense_sector_vector(&dets, &x_full);
    let y_sector = dense_sector_vector(&dets, &y_full);
    let mu_x = mat_vec(&dense_kappa_matrix(fx.space.d, &m, &dets), &x_sector);

    let scale = mu_x.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
    for i in 0..dets.len() {
        assert_close(
            y_sector[i],
            mu_x[i],
            1e-10 * scale,
            &format!("apply_cross_target vs dense mu.x at det {i}"),
        );
    }
    assert!(
        y.norm() > 1e-6,
        "anti-vacuity: apply_cross_target must not give the zero train here"
    );
}

#[test]
fn tji_cross_target_trivial_group_degeneration() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE301);
    let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let m = random_sym_t(d, &mut rng);

    let classes = property_flux_classes(&m, PointGroup::TRIVIAL);
    assert_eq!(
        classes,
        vec![Irrep::IDENTITY],
        "TRIVIAL group: everything is IDENTITY flux"
    );

    let y = apply_cross_target(&x, &m, PointGroup::TRIVIAL);
    assert_eq!(
        y.space, space,
        "TRIVIAL group: output space must equal the input space"
    );

    let prop = PropertyOperator::new("trivial_cross_target", m.clone());
    let mu_ham = property_hamiltonian(space, &prop);
    let want = &mu_ham * &x;

    let (yd, wd) = (y.to_dense(), want.to_dense());
    let scale = wd.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
    for (i, (&g, &w)) in yd.iter().zip(wd.iter()).enumerate() {
        assert_close(
            g,
            w,
            1e-12 * scale,
            &format!("TRIVIAL-group apply_cross_target vs property_hamiltonian * x at index {i}"),
        );
    }
    assert!(
        want.norm() > 1e-6,
        "anti-vacuity: the ordinary route must not give the zero train here"
    );
}

#[test]
fn tji_cross_target_flux_classes() {
    let fx = cross_target_fixture();
    let zero_m = CoeffTensor::<f64>::zeros(fx.space.d, 2);
    let classes = property_flux_classes(&zero_m, fx.pg);
    assert!(classes.is_empty(), "m===0 must give no flux classes");

    let y0 = apply_cross_target(&fx.x, &zero_m, fx.pg);
    assert_eq!(
        y0.space.target, fx.space.target,
        "m===0 must leave the target unchanged"
    );
    assert!(y0.norm() <= 1e-14, "m===0 must give the exact zero train");
}

#[test]
#[should_panic]
fn tji_cross_target_multi_flux_panics() {
    let fx = cross_target_fixture();
    assert_ne!(
        fx.pg.irrep_of(0).fuse(fx.pg.irrep_of(1)),
        fx.pg.irrep_of(0).fuse(fx.pg.irrep_of(2)),
        "fixture precondition: two distinct cross-irrep flux classes must be available"
    );
    let mut m = CoeffTensor::<f64>::zeros(fx.space.d, 2);
    m.set(&[0, 1], 1.0);
    m.set(&[1, 0], 1.0);
    m.set(&[0, 2], 1.0);
    m.set(&[2, 0], 1.0);
    let classes = property_flux_classes(&m, fx.pg);
    assert!(
        classes.len() > 1,
        "test setup must actually produce >1 flux class"
    );
    let _ = apply_cross_target(&fx.x, &m, fx.pg);
}

#[test]
fn tji_cross_target_same_irrep_matches_ordinary_route() {
    let fx = cross_target_fixture();
    let d = fx.space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE302);
    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    let mut any = false;
    for &(p, q) in &orbital_pairs(fx.pg, d) {
        use rand::RngExt;
        let val = rng.random_range(-1.0f64..1.0);
        m.set(&[p, q], val);
        m.set(&[q, p], val);
        any = true;
    }
    for i in 0..d {
        m.set(&[i, i], (i as f64) - 1.5);
        any = true;
    }
    assert!(any, "fixture must produce a nonzero same-irrep property");

    let classes = property_flux_classes(&m, fx.pg);
    assert_eq!(
        classes,
        vec![Irrep::IDENTITY],
        "a same-irrep property has exactly one, trivial, flux class"
    );

    let y = apply_cross_target(&fx.x, &m, fx.pg);
    assert_eq!(
        y.space, fx.space,
        "same-irrep route must leave the target unchanged"
    );

    let prop = PropertyOperator::new("cross_target_same_irrep", m.clone());
    let mu_ham = property_hamiltonian(fx.space, &prop);
    let want = &mu_ham * &fx.x;

    let (yd, wd) = (y.to_dense(), want.to_dense());
    let scale = wd.iter().fold(0.0f64, |a, &v| a.max(v.abs())).max(1.0);
    for (i, (&g, &w)) in yd.iter().zip(wd.iter()).enumerate() {
        assert_close(
            g,
            w,
            1e-12 * scale,
            &format!("cross-target route (Gamma_mu=IDENTITY) vs property_hamiltonian * x at {i}"),
        );
    }
    assert!(
        want.norm() > 1e-6,
        "anti-vacuity: the ordinary route must not give the zero train here"
    );
}

#[test]
fn tji_irrep_moments_union_matches_full_basis() {
    let fx = joint_critical_pg_fixture();
    let jctx = joint_critical_pg_jctx(fx);
    let d = fx.out.x.space.d;

    let mut m1 = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        m1.set(&[i, i], (i as f64) - 0.5);
    }
    let mut m2 = CoeffTensor::<f64>::zeros(d, 2);
    m2.set(&[1, 3], 0.7);
    m2.set(&[3, 1], 0.7);
    m2.set(&[0, 0], -0.3);
    let props = [
        PropertyOperator::new("diag", m1),
        PropertyOperator::new("same_irrep_pair", m2),
    ];
    let opts = TransitionMomentOpts::default();

    let (basis_full, _report) = jctx.joint_basis();
    let full = transition_moments(&jctx, &basis_full, &props, &opts)
        .expect("full-basis moments must build at the converged reference");

    let mut union: Vec<(f64, f64, f64)> = Vec::new();
    for h in 0..fx.pg.order() {
        let gamma = Irrep::new(h);
        let (basis_g, report) = joint_irrep_basis(&jctx, fx.pg, gamma, fx.gamma0);
        println!(
            "tji_irrep_moments_union_matches_full_basis: gamma={gamma:?} kept={} max_leakage={:e}",
            report.kept, report.max_leakage
        );
        if basis_g.is_empty() {
            continue;
        }
        let mg = transition_moments(&jctx, &basis_g, &props, &opts)
            .expect("per-irrep moments must build at the converged reference");
        for n in 0..mg.omega.len() {
            union.push((mg.omega[n], mg.moments[0][n], mg.moments[1][n]));
        }
    }
    union.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let n_modes = full.omega.len();
    assert_eq!(
        union.len(),
        n_modes,
        "union of per-irrep mode counts must equal the full basis mode count"
    );
    let wscale = full.omega.last().copied().unwrap_or(1.0).max(1.0);
    for (n, um) in union.iter().enumerate() {
        assert!(
            (um.0 - full.omega[n]).abs() <= 1e-8 * wscale,
            "omega[{n}]: union {} vs full {}",
            um.0,
            full.omega[n]
        );
    }

    let mut clusters: Vec<(usize, usize)> = Vec::new();
    let mut lo = 0usize;
    for n in 1..n_modes {
        if (full.omega[n] - full.omega[n - 1]).abs() > 1e-6 * full.omega[n].max(1.0) {
            clusters.push((lo, n));
            lo = n;
        }
    }
    clusters.push((lo, n_modes));
    let mut max_t = 0.0f64;
    for &(clo, chi) in &clusters {
        let mut want = [0.0f64; 3];
        let mut got = [0.0f64; 3];
        for (n, um) in union.iter().enumerate().take(chi).skip(clo) {
            want[0] += full.moments[0][n] * full.moments[0][n];
            want[1] += full.moments[1][n] * full.moments[1][n];
            want[2] += full.moments[0][n] * full.moments[1][n];
            got[0] += um.1 * um.1;
            got[1] += um.2 * um.2;
            got[2] += um.1 * um.2;
            max_t = max_t.max(um.1.abs()).max(um.2.abs());
        }
        for p in 0..3 {
            assert_close(
                got[p],
                want[p],
                1e-8 * want[p].abs().max(1e-6),
                &format!("cluster [{clo},{chi}) moment product {p}: union vs full"),
            );
        }
    }
    println!(
        "tji_irrep_moments_union_matches_full_basis: n_modes={n_modes} clusters={} max|T|={max_t:e} \
         orth_error={:e}",
        clusters.len(),
        full.orth_error
    );
    assert!(max_t > 1e-6, "anti-vacuity: all moments vanished");
}

#[test]
fn tji_sweep_refused_at_nontrivial_pg() {
    let fx = pg_fixture();
    let space_g = QnSpace::with_point_group(fx.d, 2, 1, fx.pg, fx.gamma0);
    let x_g = capped_randn(space_g, 1, 2, 0xE9EE);
    let (mut t, mut v) = pg_symmetric_tv(fx.d, fx.pg);
    qntt::fcidump::screen_point_group_forbidden(&mut t, &mut v, fx.pg).expect("symmetric tv");
    let ham = SparseHamiltonian::new(space_g, &t, &v, None, f64::EPSILON);
    let jopts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..JointOpts::default()
    };
    let out = joint_context_with(
        &ham,
        &x_g,
        &jopts,
        &JointCtxBuild {
            gram: GramRoute::Sweep,
            lean: false,
        },
    );
    assert!(
        matches!(out, Err(JointGateError::SweepNeedsTrivialPg)),
        "explicit Sweep at |G| > 1 must refuse with SweepNeedsTrivialPg, got {:?}",
        out.err()
    );
}
