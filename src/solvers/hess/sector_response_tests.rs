// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::sector_response::{moments_from_grams, sector_refusal_to_error};
use super::{
    SectorAllOpts, SectorCtx, SectorMomentOpts, SectorObservables, SectorResponseError,
    SectorStaticOpts, property_flux, sector_response_all, weingarten_operator,
};
use crate::coeff::CoeffTensor;
use crate::core::tangent::{TtTangent, tangent_dim_grids};
use crate::core::weingarten::{RpaMetricBlock, WilliamsonError, williamson_reduced_signed_vectors};
use crate::elementaryops::{apply_cross_target, property_flux_classes};
use crate::hamiltonian::SparseHamiltonian;
use crate::integrals::PropertyOperator;
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::QnSpace;
use crate::round::RoundMode;
use crate::solvers::joint::PropRoute;
use crate::solvers::joint::RefusalKind;
use crate::solvers::riemannian::{normalize_mut, tangent_bases};
use crate::test_util::randomized_state;
use crate::vector::TtVector;
use crate::vector::tt_state_in;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const D: usize = 4;

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(3), Irrep::new(2), Irrep::new(1), Irrep::new(0)],
    )
}

fn fx_tv() -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(D, 2);
    for p in 0..D {
        t.set(&[p, p], 0.3 * (p as f64 - 1.5));
    }
    (t, CoeffTensor::<f64>::zeros(D, 4))
}

fn fx() -> (QnSpace, SparseHamiltonian<f64>, TtVector<f64>) {
    let pg = fx_pg();
    let space = QnSpace::with_point_group(D, 2, 2, pg, Irrep::IDENTITY);
    let (t, v) = fx_tv();
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE1E1);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    x.check();
    (space, ham, x)
}

fn live_and_empty(space: QnSpace, x: &TtVector<f64>) -> (Irrep, Irrep) {
    let (bl, _) = tangent_bases(x);
    let dims: Vec<usize> = (0..space.pg.order())
        .map(|p| tangent_dim_grids(&space, &bl.ranks, Irrep::new(p)))
        .collect();
    let live = (1..space.pg.order())
        .map(Irrep::new)
        .find(|p| dims[p.index() as usize] > 0)
        .unwrap_or_else(|| panic!("no live shifted channel: dims {dims:?}"));
    let empty = (1..space.pg.order())
        .map(Irrep::new)
        .find(|p| dims[p.index() as usize] == 0)
        .unwrap_or_else(|| panic!("no empty shifted channel: dims {dims:?}"));
    (live, empty)
}

#[allow(clippy::type_complexity, clippy::needless_range_loop)]
pub(super) fn fx_exact_ground() -> (
    QnSpace,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    SparseHamiltonian<f64>,
    TtVector<f64>,
) {
    use crate::test_util::dense_dot;
    let pg = fx_pg();
    let (mut t, mut v) = fx_tv();
    for p in 0..D {
        for q in 0..D {
            if p != q {
                v.set(&[p, p, q, q], 0.10 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    for idx in [[2, 0, 3, 1], [3, 1, 2, 0], [0, 2, 1, 3], [1, 3, 0, 2]] {
        v.set(&idx, 0.15);
    }
    crate::fcidump::screen_point_group_forbidden(&mut t, &mut v, pg)
        .expect("symmetry-allowed by construction: the screen is a no-op");
    let subsets: Vec<[bool; D]> = (0..(1usize << D))
        .filter(|m| m.count_ones() == 2)
        .map(|m| std::array::from_fn(|k| m & (1 << k) != 0))
        .collect();
    let mut by_irrep: Vec<Vec<([bool; D], [bool; D])>> = vec![Vec::new(); pg.order() as usize];
    for up in &subsets {
        for dn in &subsets {
            let ir = (0..D)
                .filter(|&k| up[k] || dn[k])
                .fold(Irrep::IDENTITY, |acc, k| {
                    if up[k] && dn[k] {
                        acc
                    } else {
                        acc.fuse(pg.irrep_of(k))
                    }
                });
            by_irrep[ir.index() as usize].push((*up, *dn));
        }
    }
    let ground_of = |g: u8| -> (f64, Vec<f64>, Vec<TtVector<f64>>) {
        let space = QnSpace::with_point_group(D, 2, 2, pg, Irrep::new(g));
        let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let dets: Vec<TtVector<f64>> = by_irrep[g as usize]
            .iter()
            .map(|(up, dn)| tt_state_in(space, up, dn))
            .collect();
        let dense: Vec<Vec<f64>> = dets.iter().map(TtVector::to_dense).collect();
        let hd: Vec<Vec<f64>> = dets.iter().map(|d| (&ham * d).to_dense()).collect();
        let n = dets.len();
        let mut m = vec![vec![0.0f64; n]; n];
        for (row, di) in m.iter_mut().zip(&dense) {
            for (cell, hj) in row.iter_mut().zip(&hd) {
                *cell = dense_dot(di, hj);
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let s = 0.5 * (m[i][j] + m[j][i]);
                m[i][j] = s;
                m[j][i] = s;
            }
        }
        let (ev, vecs) = qntt_testkit::dense::dense_eigen(&m);
        (ev[0], vecs[0].clone(), dets)
    };
    let (e0, c, dets) = ground_of(0);
    for g in 1..pg.order() {
        if by_irrep[g as usize].is_empty() {
            continue;
        }
        let (eg, _, _) = ground_of(g);
        assert!(
            eg > e0 + 1e-3,
            "fx_exact_ground premise: irrep {g}'s ground {eg} must lie ABOVE Γ0's {e0}"
        );
    }
    assert!(
        c.iter().filter(|z| z.abs() > 1e-6).count() >= 2,
        "fx_exact_ground premise: the ground must be a superposition, got {c:?}"
    );
    let mut acc: Option<TtVector<f64>> = None;
    for (ci, det) in c.iter().zip(&dets) {
        if *ci == 0.0 {
            continue;
        }
        let term = det * *ci;
        acc = Some(match acc {
            None => term,
            Some(a) => &a + &term,
        });
    }
    let mut x = acc.expect("a nonzero ground eigenvector");
    x.round_mut(1e-13, crate::qn::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    x.check();
    let space = QnSpace::with_point_group(D, 2, 2, pg, Irrep::IDENTITY);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    (space, t, v, ham, x)
}

fn flux_matrix(pg: PointGroup, phi: Irrep, c: f64) -> CoeffTensor<f64> {
    let mut m = CoeffTensor::<f64>::zeros(D, 2);
    let mut k = 0usize;
    for p in 0..D {
        for q in (p + 1)..D {
            if pg.irrep_of(p).fuse(pg.irrep_of(q)) == phi {
                k += 1;
                let w = c * (1.0 + 0.1 * k as f64);
                m.set(&[p, q], w);
                m.set(&[q, p], w);
            }
        }
    }
    assert!(k > 0, "no pair of flux {phi:?} on this table");
    assert_eq!(property_flux_classes(&m, pg), vec![phi]);
    m
}

fn h_out_for(space: QnSpace, phi: Irrep) -> SparseHamiltonian<f64> {
    let (t, v) = fx_tv();
    SparseHamiltonian::<f64>::new(
        space.with_target(space.target.fuse(phi)),
        &t,
        &v,
        None,
        f64::EPSILON,
    )
}

#[test]
fn tse_property_vector_is_the_projected_cross_target_image() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, phi);
    let sc = SectorCtx::new(&ctx, &h_out, phi);
    let p = PropertyOperator::new("mu", flux_matrix(pg, phi, 0.7));
    assert_eq!(property_flux(&p, pg).expect("single class"), Some(phi));

    let got = sc.property_vector(&p).expect("in-channel flux");
    assert_eq!(got.phi(), phi);
    let mx = apply_cross_target(&x, &p.m, pg);
    assert_eq!(mx.space, sc.space_out(), "mu x lives in space_out");
    let want = sc.project(&mx);
    let n = want.norm();
    assert!(n > 1e-6, "anti-vacuity: ||P^phi(mu x)|| = {n:e}");
    let mut diff = got.clone();
    diff.axpy(-1.0, &want);
    assert!(
        diff.norm() <= 1e-14 * n,
        "property_vector differs from project_shifted(apply_cross_target): {:e}",
        diff.norm()
    );
    let p2 = PropertyOperator::new("2mu", flux_matrix(pg, phi, 1.4));
    let got2 = sc.property_vector(&p2).expect("in-channel");
    let mut d2 = got2;
    d2.axpy(-2.0, &got);
    assert!(
        d2.norm() <= 1e-13 * n,
        "P^phi(2 mu x) != 2 P^phi(mu x): {:e}",
        d2.norm()
    );
}

#[test]
fn tse_wrong_and_multi_flux_are_refused_by_name() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let other = (1..pg.order())
        .map(Irrep::new)
        .find(|&q| q != phi)
        .expect("a second non-identity flux");
    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, phi);
    let sc = SectorCtx::new(&ctx, &h_out, phi);

    let wrong = PropertyOperator::new("wrong", flux_matrix(pg, other, 0.5));
    match sc.property_vector(&wrong) {
        Err(SectorResponseError::WrongFlux {
            label,
            classes,
            phi: ph,
        }) => {
            assert_eq!(label, "wrong");
            assert_eq!(classes, vec![other]);
            assert_eq!(ph, Some(phi), "a channel-bound refusal names the channel");
        }
        other => panic!(
            "a wrong single flux must be WrongFlux, got {:?}",
            other.map(|t| t.norm())
        ),
    }

    let mut m = flux_matrix(pg, phi, 0.5);
    let m2 = flux_matrix(pg, other, 0.3);
    for p in 0..D {
        for q in 0..D {
            if m2.get(&[p, q]) != 0.0 {
                m.set(&[p, q], m2.get(&[p, q]));
            }
        }
    }
    let classes = property_flux_classes(&m, pg);
    assert_eq!(classes.len(), 2, "setup: {classes:?}");
    let multi = PropertyOperator::new("multi", m);
    match sc.property_vector(&multi) {
        Err(SectorResponseError::WrongFlux { classes: c, .. }) => assert_eq!(c, classes),
        other => panic!(
            "a multi-class m must be WrongFlux, got {:?}",
            other.map(|t| t.norm())
        ),
    }
    assert!(
        property_flux(&multi, pg).is_err(),
        "property_flux must refuse a multi-class m rather than pick one"
    );
    assert!(matches!(
        sc.polarizability_static(
            std::slice::from_ref(&multi),
            &SectorStaticOpts {
                tt_grad_tol: f64::INFINITY,
                ..SectorStaticOpts::default()
            }
        ),
        Err(SectorResponseError::WrongFlux { .. })
    ));
    assert!(matches!(
        sc.transition_moments_dense(
            std::slice::from_ref(&multi),
            &SectorMomentOpts {
                tt_grad_tol: f64::INFINITY,
                ..SectorMomentOpts::default()
            }
        ),
        Err(SectorResponseError::WrongFlux { .. })
    ));
}

#[test]
fn tse_zero_property_short_circuits_to_the_shaped_zero_of_t_phi() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let zero = PropertyOperator::new("zero", CoeffTensor::<f64>::zeros(D, 2));
    assert!(property_flux_classes(&zero.m, pg).is_empty());
    assert_eq!(
        property_flux(&zero, pg).expect("no class is not an error"),
        None
    );
    assert_eq!(
        apply_cross_target(&x, &zero.m, pg).space,
        space,
        "the sweep returns the INPUT space"
    );

    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, phi);
    let sc = SectorCtx::new(&ctx, &h_out, phi);
    let z = sc
        .property_vector(&zero)
        .expect("m == 0 is a shaped zero, not a refusal");
    assert_eq!(z.phi(), phi);
    assert_eq!(z.norm(), 0.0, "EXACTLY zero");
    assert_eq!(
        z.to_vector().space,
        sc.space_out(),
        "the zero of T^phi embeds in space_out"
    );
    let basis = sc.basis();
    assert!(!basis.is_empty());
    assert_eq!(
        basis[0].dot(&z),
        0.0,
        "dot-compatible with the channel basis"
    );
}

#[test]
fn tse_out_of_channel_component_is_a_structural_zero_row() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let other = (1..pg.order())
        .map(Irrep::new)
        .find(|&q| q != phi)
        .expect("second flux");
    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, phi);
    let sc = SectorCtx::new(&ctx, &h_out, phi);
    let props = [PropertyOperator::new("out", flux_matrix(pg, other, 0.5))];
    let st = sc
        .polarizability_static(
            &props,
            &SectorStaticOpts {
                tt_grad_tol: f64::INFINITY,
                ..SectorStaticOpts::default()
            },
        )
        .expect("an out-of-channel component alone is a zero table, not a refusal");
    assert_eq!(st.alpha, vec![0.0]);
    assert_eq!(st.solves[0].grad_norm, 0.0);
    assert_eq!(st.solves[0].iters, 0);
    assert_eq!(
        st.solves[0].rel_residual, 0.0,
        "0.0, never NaN, when p == 0"
    );
    assert_eq!(
        st.capture_defect[0], 1.0,
        "the channel captures NONE of an out-of-flux image"
    );
    assert_eq!(st.dim, sc.dim());
}

#[test]
fn tse_criticality_gate_refuses_a_non_critical_reference() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, phi);
    let sc = SectorCtx::new(&ctx, &h_out, phi);
    let props = [PropertyOperator::new("mu", flux_matrix(pg, phi, 0.7))];
    let gnorm = ctx.gradient().norm();
    assert!(
        gnorm > 1e-3,
        "fixture premise: a trimmed random point is NOT critical (||grad|| = {gnorm:e})"
    );
    match sc.polarizability_static(&props, &SectorStaticOpts::default()) {
        Err(SectorResponseError::NotTtCritical {
            tt_gradnorm,
            tt_bound,
            lambda,
        }) => {
            assert!(tt_gradnorm > tt_bound, "{tt_gradnorm} vs bound {tt_bound}");
            assert!(
                (tt_gradnorm - gnorm).abs() <= 1e-12 * gnorm,
                "the gate reports the context's own gradient norm"
            );
            assert!((tt_bound - 1e-6 * lambda.abs().max(1.0)).abs() <= 1e-18);
            assert!((lambda - sc.lambda()).abs() <= 1e-14 * lambda.abs().max(1.0));
        }
        other => panic!("expected NotTtCritical, got {other:?}"),
    }
    assert!(matches!(
        sc.transition_moments_dense(&props, &SectorMomentOpts::default()),
        Err(SectorResponseError::NotTtCritical { .. })
    ));
    let other = (1..pg.order())
        .map(Irrep::new)
        .find(|&q| q != phi)
        .expect("second flux");
    let mut m = flux_matrix(pg, phi, 0.5);
    let m2 = flux_matrix(pg, other, 0.3);
    for p in 0..D {
        for q in 0..D {
            if m2.get(&[p, q]) != 0.0 {
                m.set(&[p, q], m2.get(&[p, q]));
            }
        }
    }
    assert!(matches!(
        sc.polarizability_static(
            &[PropertyOperator::new("multi", m)],
            &SectorStaticOpts::default()
        ),
        Err(SectorResponseError::WrongFlux { .. })
    ));
    sc.property_vector(&props[0])
        .expect("property_vector consumes no Hessian and has no gate");
}

#[test]
fn tse_empty_channel_is_an_empty_table_not_a_refusal() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (_, empty) = live_and_empty(space, &x);
    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, empty);
    let sc = SectorCtx::new(&ctx, &h_out, empty);
    assert_eq!(sc.dim(), 0);
    let p = PropertyOperator::new("mu_empty", flux_matrix(pg, empty, 0.7));
    let mx = apply_cross_target(&x, &p.m, pg);
    assert!(
        mx.norm() > 1e-6,
        "anti-vacuity: mu x != 0 on the empty channel"
    );
    let z = sc.property_vector(&p).expect("its own flux is accepted");
    assert_eq!((z.phi(), z.norm()), (empty, 0.0));
    let opts = SectorMomentOpts {
        tt_grad_tol: f64::INFINITY,
        ..SectorMomentOpts::default()
    };
    let tm = sc
        .transition_moments_dense(std::slice::from_ref(&p), &opts)
        .expect("empty channel: empty table");
    assert!(tm.omega.is_empty() && tm.moments[0].is_empty() && tm.oscillator.is_empty());
    assert_eq!((tm.dim, tm.n_neg, tm.complete_basis), (0, 0, true));
    assert_eq!(tm.alpha_sos, Some(vec![0.0]));
    assert_eq!(tm.capture_defect[0], 1.0);
    let st = sc
        .polarizability_static(
            std::slice::from_ref(&p),
            &SectorStaticOpts {
                tt_grad_tol: f64::INFINITY,
                ..SectorStaticOpts::default()
            },
        )
        .expect("empty channel: zero alpha");
    assert_eq!(
        (st.alpha[0], st.solves[0].iters, st.capture_defect[0]),
        (0.0, 0, 1.0)
    );
}

#[test]
fn tse_refusal_map_splits_at_omega_zero() {
    let e = sector_refusal_to_error(RefusalKind::Curvature(-3.5), "x".into(), 0.0, 7, 1e-3, 1e-8)
        .expect("curvature must refuse");
    assert!(
        matches!(e, SectorResponseError::HessianIndefinite { rho, iters, .. } if rho == -3.5 && iters == 7),
        "omega = 0 must map to HessianIndefinite, got {e:?}"
    );
    let e = sector_refusal_to_error(RefusalKind::Curvature(-3.5), "x".into(), 0.4, 7, 1e-3, 1e-8)
        .expect("curvature must refuse");
    assert!(
        matches!(e, SectorResponseError::IndefiniteAbovePole { omega, rho, .. } if omega == 0.4 && rho == -3.5),
        "omega > 0 must map to IndefiniteAbovePole, got {e:?}"
    );
    let e = sector_refusal_to_error(
        RefusalKind::NotConverged(Some(500)),
        "x".into(),
        0.0,
        500,
        2.5e-4,
        1e-8,
    )
    .expect("non-convergence must refuse");
    assert!(
        matches!(e, SectorResponseError::SolveNotConverged { iters, rel_residual, tol, .. } if iters == 500 && rel_residual == 2.5e-4 && tol == 1e-8),
        "got {e:?}"
    );
    assert!(
        sector_refusal_to_error(
            RefusalKind::NotConverged(None),
            "x".into(),
            0.0,
            3,
            1e-12,
            1e-8
        )
        .is_none()
    );
}

fn rotated_diag(v: [f64; 3]) -> Vec<f64> {
    let (c, s) = (0.8f64, 0.6f64);
    let q = [[c, -s * c, s * s], [s, c * c, -c * s], [0.0, s, c]];
    let mut out = vec![0.0f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[i * 3 + j] = (0..3).map(|k| q[i][k] * v[k] * q[j][k]).sum();
        }
    }
    out
}

fn gammas() -> Vec<Vec<f64>> {
    vec![vec![1.0, 0.5, -0.25], vec![-0.3, 0.9, 0.4]]
}

#[test]
fn tse_moments_from_grams_stable_is_the_orthogonal_case() {
    let a = rotated_diag([1.0, 2.0, 3.0]);
    let b = vec![0.0f64; 9];
    let g = gammas();
    let out = moments_from_grams(&a, &b, &g, &SectorMomentOpts::default()).expect("stable");
    assert_eq!(out.n_neg, 0);
    assert!(out.unstable_omega2.is_empty());
    assert_eq!(out.metric, RpaMetricBlock::Rr);
    assert_eq!(out.omega.len(), 3);
    for (w, want) in out.omega.iter().zip([1.0f64, 2.0, 3.0]) {
        assert!((w - want).abs() <= 1e-12, "omega {w} vs {want}");
    }
    let sos = out.alpha_sos.as_ref().expect("n_neg == 0 ⇒ Some");
    assert_eq!(sos.len(), 4);
    let sw = williamson_reduced_signed_vectors(&a, &a, 3, 1e-10).expect("PD");
    for (p, gp) in g.iter().enumerate() {
        let want: Vec<f64> = (0..3)
            .map(|n| (0..3).map(|i| gp[i] * sw.s_r[i * 3 + n]).sum())
            .collect();
        for (n, &wn) in want.iter().enumerate() {
            assert!(
                (out.moments[p][n].abs() - wn.abs()).abs() <= 1e-12,
                "|T_{n}| of property {p}"
            );
        }
        let g2: f64 = gp.iter().map(|z| z * z).sum();
        let t2: f64 = out.moments[p].iter().map(|z| z * z).sum();
        assert!((t2 - g2).abs() <= 1e-12, "sum rule at B == 0: {t2} vs {g2}");
        assert_eq!(
            out.sum_rule[p],
            (t2, g2),
            "the reported pair is the computed pair"
        );
        assert!(out.completeness_defect[p] <= 1e-12);
    }
    let (lam, vecs) = crate::core::factor::dense_self_adjoint_smallest::<f64>(&a, 3, 3);
    for p in 0..2 {
        for q in 0..2 {
            let want: f64 = 2.0
                * (0..3)
                    .map(|k| {
                        let cp: f64 = (0..3).map(|i| vecs[k][i] * g[p][i]).sum();
                        let cq: f64 = (0..3).map(|i| vecs[k][i] * g[q][i]).sum();
                        cp * cq / lam[k]
                    })
                    .sum::<f64>();
            assert!(
                (sos[p * 2 + q] - want).abs() <= 1e-12,
                "alpha_sos[{p},{q}] {} vs {want}",
                sos[p * 2 + q]
            );
        }
    }
    assert_eq!(out.degenerate_clusters, vec![(0, 1), (1, 2), (2, 3)]);
}

#[test]
fn tse_moments_from_grams_resonant_publishes_real_modes_only() {
    let a = rotated_diag([1.0, 2.0, 3.0]);
    let b = rotated_diag([0.5, 0.5, 3.5]);
    let g = gammas();
    let out = moments_from_grams(&a, &b, &g, &SectorMomentOpts::default())
        .expect("resonant is NOT a refusal");
    assert_eq!(out.n_neg, 1);
    assert_eq!(out.metric, RpaMetricBlock::Rr);
    assert_eq!(out.omega.len(), 2, "n − n_neg real modes");
    assert!(
        (out.omega[0] - 0.75f64.sqrt()).abs() <= 1e-12
            && (out.omega[1] - 3.75f64.sqrt()).abs() <= 1e-12,
        "{:?}",
        out.omega
    );
    assert_eq!(out.unstable_omega2.len(), 1);
    assert!(
        (out.unstable_omega2[0] + 3.25).abs() <= 1e-12,
        "{:?}",
        out.unstable_omega2
    );
    assert!(
        out.alpha_sos.is_none(),
        "n_neg > 0 ⇒ alpha_sos is None, never a silent partial sum"
    );
    let hrr: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
    let hii: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p - q).collect();
    let sw = williamson_reduced_signed_vectors(&hrr, &hii, 3, 1e-10).expect("one PD block");
    assert_eq!(sw.n_neg, 1);
    let n_pos = sw.omega.len();
    for (p, gp) in g.iter().enumerate() {
        assert_eq!(out.moments[p].len(), 2, "moments for the REAL modes only");
        for n in 0..n_pos {
            let want: f64 = (0..3).map(|i| gp[i] * sw.s_r[i * n_pos + n]).sum();
            assert!(
                (out.moments[p][n].abs() - want.abs()).abs() <= 1e-12,
                "|T_{n}| of property {p}: {} vs {want}",
                out.moments[p][n]
            );
        }
        let g2: f64 = gp.iter().map(|z| z * z).sum();
        let (t2, rhs) = out.sum_rule[p];
        assert_eq!(rhs, g2);
        assert!(t2.is_finite() && t2 > 0.0);
        assert!(
            (t2 - g2).abs() > 1e-6,
            "on a resonant channel Σ T² over the real modes is NOT the sum rule (got {t2} vs {g2}) — it must not be claimed as one"
        );
    }
    assert_eq!(out.degenerate_clusters, vec![(0, 1), (1, 2)]);
}

#[test]
fn tse_moments_from_grams_on_the_analytic_resonant_family() {
    let a = vec![1.0, 1.0, 1.0, 1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];
    let g = vec![vec![1.0, 1.0], vec![1.0, -1.0]];
    let out = moments_from_grams(&a, &b, &g, &SectorMomentOpts::default())
        .expect("one PD block ⇒ Resonant, not a refusal");

    assert_eq!(out.metric, RpaMetricBlock::Rr, "H^rr is the PD block");
    assert_eq!(out.n_neg, 1, "n_neg = ind(H^ii) = 1, exact by Sylvester");
    assert_eq!(out.unstable_omega2.len(), 1);
    assert!(
        (out.unstable_omega2[0] + 1.0).abs() <= 1e-13,
        "the published resonance is ω² = −1, got {:?}",
        out.unstable_omega2
    );
    assert_eq!(out.omega.len(), 1, "n − n_neg = 1 real mode");
    assert!(
        (out.omega[0] - 3.0f64.sqrt()).abs() <= 1e-14,
        "ω = √3, got {}",
        out.omega[0]
    );
    assert!(
        out.alpha_sos.is_none(),
        "n_neg > 0 ⇒ no α_SOS, never a silent partial sum"
    );
    assert_eq!(out.degenerate_clusters, vec![(0, 1)]);

    let want_t = 2.0f64 / 3.0f64.sqrt();
    assert_eq!(out.moments[0].len(), 1);
    assert!(
        (out.moments[0][0].abs() - want_t.sqrt()).abs() <= 1e-13,
        "|T| = √(2/√3), got {}",
        out.moments[0][0]
    );
    let (t2, rhs) = out.sum_rule[0];
    assert!((t2 - want_t).abs() <= 1e-13, "Σ T² = 2/√3, got {t2}");
    assert_eq!(rhs, 2.0, "‖γ‖² = 2 exactly");
    assert!(
        (t2 - rhs).abs() > 0.8,
        "the real modes of a resonant channel are NOT complete: {t2} vs {rhs}"
    );
    assert!(
        (out.sum_rule_si[0] - 2.0 * 3.0f64.sqrt()).abs() <= 1e-13,
        "the S_i-side diagnostic is 2√3, got {}",
        out.sum_rule_si[0]
    );

    assert!(
        out.moments[1][0].abs() <= 1e-14,
        "the moment along the unstable mode must vanish, got {}",
        out.moments[1][0]
    );
    assert_eq!(out.sum_rule[1], (out.moments[1][0].powi(2), 2.0));
}

#[test]
fn tse_moments_from_grams_refuses_two_indefinite_blocks() {
    let a = rotated_diag([-1.0, 2.0, 3.0]);
    let b = vec![0.0f64; 9];
    match moments_from_grams(&a, &b, &gammas(), &SectorMomentOpts::default()) {
        Err(SectorResponseError::SymplecticUnstable {
            reason: WilliamsonError::HrrNotPd { min_eig },
        }) => {
            assert!((min_eig + 1.0).abs() <= 1e-12, "min_eig {min_eig}");
        }
        other => {
            panic!("both blocks indefinite must be SymplecticUnstable(HrrNotPd), got {other:?}")
        }
    }
}

#[test]
fn tse_moments_from_grams_empty_is_an_empty_table() {
    let out = moments_from_grams(&[], &[], &[vec![], vec![]], &SectorMomentOpts::default())
        .expect("empty");
    assert!(
        out.omega.is_empty()
            && out.moments.iter().all(Vec::is_empty)
            && out.unstable_omega2.is_empty()
    );
    assert_eq!(out.n_neg, 0);
    assert_eq!(out.alpha_sos, Some(vec![0.0; 4]));
    assert_eq!(out.sum_rule, vec![(0.0, 0.0); 2]);
}

#[test]
fn tse_dense_route_is_moments_from_grams_over_the_channel_basis() {
    let (space, t, v, ham, x) = fx_exact_ground();
    let pg = space.pg;
    let (bl, _) = tangent_bases(&x);
    let phi = (1..pg.order())
        .map(Irrep::new)
        .find(|&q| tangent_dim_grids(&space, &bl.ranks, q) > 0)
        .expect("fx_exact_ground: at least one live shifted channel");
    let ctx = weingarten_operator(&ham, &x);
    let h_out = SparseHamiltonian::<f64>::new(
        space.with_target(space.target.fuse(phi)),
        &t,
        &v,
        None,
        f64::EPSILON,
    );
    let sc = SectorCtx::new(&ctx, &h_out, phi);
    let p = PropertyOperator::new("mu", flux_matrix(pg, phi, 0.7));
    let opts = SectorMomentOpts::default();
    let basis = sc.basis();
    let (a, b) = sc.grams(&basis);
    let b_max = b.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    assert!(
        b_max <= 1e-12,
        "fixture premise: B^phi == 0, got max|B| = {b_max:e}"
    );
    let pv = sc.property_vector(&p).expect("in-channel");
    assert!(pv.norm() > 1e-6, "anti-vacuity: ||p|| = {:e}", pv.norm());
    let gam: Vec<f64> = basis
        .iter()
        .map(|bi: &TtTangent<f64>| bi.dot(&pv))
        .collect();
    let g2: f64 = gam.iter().map(|z| z * z).sum();
    let f = moments_from_grams(&a, &b, &[gam], &opts).expect("flat seam: Stable by construction");
    let t = sc
        .transition_moments_dense(std::slice::from_ref(&p), &opts)
        .expect("dense route: Stable by construction, gate armed and passed");
    assert_eq!((f.n_neg, t.n_neg), (0, 0), "Stable by construction");
    assert!(!t.omega.is_empty() && t.omega.len() == basis.len());
    assert!(t.alpha_sos.is_some() && f.alpha_sos.is_some());
    assert!(t.complete_basis);
    println!(
        "seam pin: phi {phi:?} dim {} omega {:?} |T| {:?}",
        t.dim,
        t.omega,
        t.moments[0].iter().map(|z| z.abs()).collect::<Vec<_>>()
    );
    for (u, w) in f.omega.iter().zip(&t.omega) {
        assert!((u - w).abs() <= 1e-12 * u.abs().max(1.0));
    }
    for (u, w) in f.moments[0].iter().zip(&t.moments[0]) {
        assert!((u.abs() - w.abs()).abs() <= 1e-12 * u.abs().max(1.0));
    }
    let (fs, ts) = (f.alpha_sos.unwrap(), t.alpha_sos.unwrap());
    assert!((fs[0] - ts[0]).abs() <= 1e-12 * fs[0].abs().max(1.0));
    for (name, val) in [
        ("flat sum_rule.0", f.sum_rule[0].0),
        ("flat sum_rule_si", f.sum_rule_si[0]),
        ("dense sum_rule.0", t.sum_rule[0].0),
        ("dense sum_rule_si", t.sum_rule_si[0]),
        ("dense sum_rule.1", t.sum_rule[0].1),
    ] {
        assert!(
            (val - g2).abs() <= 1e-10 * g2.max(1.0),
            "{name} = {val:.15e} vs ||gamma||² = {g2:.15e} at B == 0"
        );
    }
}

#[test]
#[should_panic(expected = "at most three Cartesian components")]
fn tse_a_fourth_component_is_a_caller_error_at_the_entry() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, phi);
    let sc = SectorCtx::new(&ctx, &h_out, phi);
    let p = PropertyOperator::new("mu", flux_matrix(pg, phi, 0.7));
    let four = vec![p.clone(), p.clone(), p.clone(), p];
    let _ = sc.polarizability_static(
        &four,
        &SectorStaticOpts {
            tt_grad_tol: f64::INFINITY,
            ..SectorStaticOpts::default()
        },
    );
}

#[test]
#[should_panic(expected = "at most three Cartesian components")]
fn tse_a_fourth_component_is_a_caller_error_at_the_driver() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let (t, v) = fx_tv();
    let p = PropertyOperator::new("mu", flux_matrix(pg, phi, 0.7));
    let four = vec![p.clone(), p.clone(), p.clone(), p];
    let _ = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            properties: Some(four),
            observables: SectorObservables::Alpha,
            ..SectorAllOpts::default()
        },
    );
}

#[test]
#[should_panic(expected = "a channel has only the Materialized route")]
fn tse_shifted_route_is_refused_by_name() {
    let (space, ham, x) = fx();
    let pg = space.pg;
    let (phi, _) = live_and_empty(space, &x);
    let ctx = weingarten_operator(&ham, &x);
    let h_out = h_out_for(space, phi);
    let sc = SectorCtx::new(&ctx, &h_out, phi);
    let p = PropertyOperator::new("mu", flux_matrix(pg, phi, 0.7));
    let _ = sc.transition_moments_dense(
        std::slice::from_ref(&p),
        &SectorMomentOpts {
            route: PropRoute::Shifted,
            tt_grad_tol: f64::INFINITY,
            ..SectorMomentOpts::default()
        },
    );
}
