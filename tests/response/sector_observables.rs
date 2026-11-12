// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::{TtTangent, min_schmidt, project_irrep, tangent_basis};
use qntt::core::weingarten::{RpaMetricBlock, WilliamsonError, williamson_reduced_signed_vectors};
use qntt::elementaryops::{apply_cross_target, property_flux_classes};
use qntt::integrals::property_hamiltonian;
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::solvers::hess::{
    RpaKernel, RpaMetric, RpaModeKind, SectorAllOpts, SectorAllReadout, SectorChannelStatus,
    SectorCtx, SectorDynamicOpts, SectorMomentOpts, SectorObservables, SectorOpts,
    SectorResponseError, SectorStaticOpts, SectorTransitionMoments, SympPrecond,
    SymplecticSpectrum, hessian_grams, property_flux, sector_response_all, weingarten_operator,
};
use qntt::solvers::joint::{
    DynamicSolver, JointOpts, PropRoute, StaticResponseOpts, TransitionMomentOpts, joint_context,
    joint_irrep_basis, oscillator_strengths, polarizability_static, transition_moments,
};
use qntt::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use qntt::vector::perturbation;
use qntt::{
    CoeffTensor, PairMask, PropertyOperator, QnSpace, Scalar, SparseHamiltonian, TtVector, dot,
    rayleigh_quotient,
};
use qntt_testkit::dense::{
    dense_eigen, dense_hamiltonian_matrix, dense_kappa_matrix, dot_vec, flat_to_nested,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::sector_util::*;

fn synthetic_pairs(pg: PointGroup, d: usize, phi: Irrep) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for p in 0..d {
        for q in (p + 1)..d {
            if pg.irrep_of(p).fuse(pg.irrep_of(q)) == phi {
                out.push((p, q));
            }
        }
    }
    out
}

fn synthetic_weight(rng: &mut Xoshiro256PlusPlus) -> f64 {
    use rand::RngExt;
    let mag: f64 = rng.random_range(0.25f64..1.0);
    if rng.random_range(0u8..2) == 0 {
        mag
    } else {
        -mag
    }
}

fn synthetic_property(
    pg: PointGroup,
    d: usize,
    phi: Irrep,
    seed: u64,
    label: &str,
) -> Option<PropertyOperator<f64>> {
    assert_ne!(
        phi,
        Irrep::IDENTITY,
        "synthetic_property: use identity_property for Γ_μ = A"
    );
    let pairs = synthetic_pairs(pg, d, phi);
    if pairs.is_empty() {
        return None;
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q) in &pairs {
        let c = synthetic_weight(&mut rng);
        m.set(&[p, q], c);
        m.set(&[q, p], c);
    }
    let classes = property_flux_classes(&m, pg);
    assert_eq!(
        classes,
        vec![phi],
        "synthetic_property({label}): the matrix's derived flux classes are {classes:?}, not [{phi:?}]"
    );
    Some(PropertyOperator::new(label, m))
}

fn identity_property(pg: PointGroup, d: usize, seed: u64, label: &str) -> PropertyOperator<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        m.set(&[p, p], synthetic_weight(&mut rng));
    }
    for p in 0..d {
        for q in (p + 1)..d {
            if pg.irrep_of(p) == pg.irrep_of(q) {
                let c = synthetic_weight(&mut rng);
                m.set(&[p, q], c);
                m.set(&[q, p], c);
            }
        }
    }
    assert_eq!(property_flux_classes(&m, pg), vec![Irrep::IDENTITY]);
    PropertyOperator::new(label, m)
}

fn multiclass_property(pg: PointGroup, d: usize, label: &str) -> PropertyOperator<f64> {
    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    let mut seen: Vec<Irrep> = Vec::new();
    for p in 0..d {
        for q in (p + 1)..d {
            let g = pg.irrep_of(p).fuse(pg.irrep_of(q));
            if g != Irrep::IDENTITY && !seen.contains(&g) && seen.len() < 2 {
                seen.push(g);
                m.set(&[p, q], 0.5);
                m.set(&[q, p], 0.5);
            }
        }
    }
    let classes = property_flux_classes(&m, pg);
    assert_eq!(
        classes.len(),
        2,
        "multiclass_property: table gives {classes:?}"
    );
    PropertyOperator::new(label, m)
}

fn d6_polarized_ground() -> Fixture {
    let pg = d6_pg();
    let space = QnSpace::with_point_group(D6, 3, 0, pg, Irrep::new(3));
    let (t, v) = d6_tv(pg);
    let (x, e0, _) = exact_typed_ground(space, &t, &v);
    for g in 0..pg.order() {
        let lv = dense_levels(D6, &t, &v, pg, 3, 0, Irrep::new(g));
        if let Some(&low) = lv.first() {
            assert!(
                low >= e0 - 1e-10,
                "d6_polarized_ground: irrep {g}'s ground {low} lies BELOW the reference {e0}"
            );
        }
    }
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    (space, t, v, ham, x)
}

fn synth_exact_ground() -> Fixture {
    let pg = synth_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = synth_tv(pg);
    let (x, e0, _) = exact_typed_ground(space, &t, &v);
    let other = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, Irrep::new(1))[0];
    assert!(
        other > e0,
        "synth_exact_ground: Γ0 = A must be the global ground"
    );
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    (space, t, v, ham, x)
}

struct Channel {
    h_out: SparseHamiltonian<f64>,
}

impl Channel {
    fn new(space: QnSpace, t: &CoeffTensor<f64>, v: &CoeffTensor<f64>, phi: Irrep) -> Self {
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, t, v, None, f64::EPSILON);
        Channel { h_out }
    }
}

fn frobenius(g: &[f64]) -> f64 {
    g.iter().map(|z| z * z).sum::<f64>().sqrt()
}

fn max_abs(v: &[f64], floor: f64) -> f64 {
    v.iter().fold(floor, |m, z| m.max(z.abs()))
}

fn assert_rel(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol * want.abs().max(1.0),
        "{what}: got {got:.15e}, want {want:.15e} (|Δ| = {:.3e}, tol {tol:e}·max(1,|want|))",
        (got - want).abs()
    );
}

fn dense_pd_solve(g: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let (lam, vecs) = dense_eigen(&flat_to_nested(g, n));
    assert!(
        lam[0] > 0.0,
        "dense_pd_solve: the matrix is not PD (min eig {})",
        lam[0]
    );
    let mut x = vec![0.0f64; n];
    for (k, u) in vecs.iter().enumerate() {
        let c = dot_vec(u, b) / lam[k];
        for i in 0..n {
            x[i] += c * u[i];
        }
    }
    x
}

fn clusters_of(omega: &[f64], tol: f64) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut lo = 0usize;
    for n in 1..=omega.len() {
        if n == omega.len() || omega[n] - omega[n - 1] > tol * omega[n].abs().max(1.0) {
            out.push((lo, n));
            lo = n;
        }
    }
    out
}

fn cluster_products(a: &[f64], b: &[f64], clusters: &[(usize, usize)]) -> Vec<f64> {
    clusters
        .iter()
        .map(|&(lo, hi)| (lo..hi).map(|n| a[n] * b[n]).sum())
        .collect()
}

fn assert_moments_agree(
    omega_got: &[f64],
    moments_got: &[Vec<f64>],
    omega_want: &[f64],
    moments_want: &[Vec<f64>],
    tol: f64,
    what: &str,
) {
    assert_multiset_eq(omega_got, omega_want, 1e-8, &format!("{what}: omega"));
    let cl = clusters_of(omega_got, 1e-6);
    assert_eq!(
        moments_got.len(),
        moments_want.len(),
        "{what}: property count"
    );
    for p in 0..moments_got.len() {
        for q in p..moments_got.len() {
            let g = cluster_products(&moments_got[p], &moments_got[q], &cl);
            let w = cluster_products(&moments_want[p], &moments_want[q], &cl);
            let scale = max_abs(&w, 1.0);
            for (c, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= tol * scale,
                    "{what}: cluster {c} ({:?}) product T({p})·T({q}) is {a:.12e} vs oracle \
                     {b:.12e} (scale {scale:e})",
                    cl[c]
                );
            }
        }
    }
}

fn untyped_embedding(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    x: &TtVector<f64>,
) -> (QnSpace, SparseHamiltonian<f64>, TtVector<f64>) {
    let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
    let x_c1 = x.forget_point_group();
    let h_c1 = SparseHamiltonian::<f64>::new(space_c1, t, v, None, f64::EPSILON);
    (space_c1, h_c1, x_c1)
}

fn tt_only_joint_opts(d: usize) -> JointOpts {
    JointOpts {
        pairs: Some(PairMask::with_predicate(d, |_, _| false)),
        ..JointOpts::default()
    }
}

fn tight_static() -> SectorStaticOpts {
    SectorStaticOpts {
        tol: 1e-10,
        ..SectorStaticOpts::default()
    }
}

fn moment_opts() -> SectorMomentOpts {
    SectorMomentOpts {
        route: PropRoute::Materialized,
        ..SectorMomentOpts::default()
    }
}

fn print_moments(tag: &str, tm: &SectorTransitionMoments) {
    println!(
        "  [{tag}] SYNTHETIC operator — dim {} phi {:?} target {:?} sigma_min {:e} orth_error {:e} \
         n_neg {} metric {:?} complete_basis {}",
        tm.dim,
        tm.phi,
        tm.target,
        tm.sigma_min,
        tm.orth_error,
        tm.n_neg,
        tm.metric,
        tm.complete_basis
    );
    for (p, lab) in tm.labels.iter().enumerate() {
        println!(
            "  [{tag}] SYNTHETIC {lab}: |T_n| {:?} capture_defect {:.3e} completeness_defect {:.3e} \
             sum_rule (ΣT², ‖p‖²) = ({:.12e}, {:.12e})",
            tm.moments[p].iter().map(|z| z.abs()).collect::<Vec<_>>(),
            tm.capture_defect[p],
            tm.completeness_defect[p],
            tm.sum_rule[p].0,
            tm.sum_rule[p].1
        );
    }
    println!(
        "  [{tag}] SYNTHETIC omega {:?} f_n {:?} clusters {:?} unstable_omega2 {:?}",
        tm.omega, tm.oscillator, tm.degenerate_clusters, tm.unstable_omega2
    );
}

fn t1_untyped_embedding_oracle(legs: Vec<T1Leg>) {
    for T1Leg {
        label,
        fx: (space, t, v, ham, x),
        curved_required,
    } in legs
    {
        let pg = space.pg;
        let d = space.d;
        let (bl, br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let live: Vec<Irrep> = live_channels(&space, &bl.ranks)
            .into_iter()
            .filter(|&p| p != Irrep::IDENTITY)
            .collect();
        let curved = curved_channels(&space, &bl.ranks);
        println!(
            "tso T1 [{label}]: sigma_min {:e} ||r_perp|| {:e} lambda {} live {live:?} curved {curved:?}",
            min_schmidt(&bl, &br),
            residual_norm(&ham, &x),
            rayleigh_quotient(&ham, &x).re_s()
        );
        assert!(!live.is_empty(), "[{label}] no live shifted channel");
        if curved_required {
            assert!(
                !curved.is_empty(),
                "[{label}] fixture premise: this leg must carry a curved channel"
            );
        }

        let m_a = identity_property(pg, d, 0xE0A0, "mu_A");
        let mut per_channel: Vec<(Irrep, PropertyOperator<f64>, PropertyOperator<f64>)> =
            Vec::new();
        for &phi in &live {
            let p = phi.index() as u64;
            let (Some(pa), Some(pb)) = (
                synthetic_property(pg, d, phi, 0xE100 + p, &format!("mu_{p}_a")),
                synthetic_property(pg, d, phi, 0xE200 + p, &format!("mu_{p}_b")),
            ) else {
                println!(
                    "  [{label}] phi {phi:?}: no representable observable on this table — skipped"
                );
                continue;
            };
            per_channel.push((phi, pa, pb));
        }
        assert!(
            !per_channel.is_empty(),
            "[{label}] anti-vacuity: no live channel carries a synthetic component"
        );
        let mut props_all: Vec<PropertyOperator<f64>> = Vec::new();
        for (_, pa, pb) in &per_channel {
            props_all.push(pa.clone());
            props_all.push(pb.clone());
        }
        props_all.push(m_a.clone());
        let n_all = props_all.len();

        let (space_c1, h_c1, x_c1) = untyped_embedding(space, &t, &v, &x);
        let jctx = joint_context(&h_c1, &x_c1, &tt_only_joint_opts(d))
            .expect("the untyped embedding of a typed critical point is TT-critical");
        assert_eq!(
            jctx.n_eff(),
            0,
            "[{label}] the TT-only recipe must give n_eff = 0"
        );
        let alpha_untyped = polarizability_static(
            &jctx,
            &props_all,
            &StaticResponseOpts {
                tol: 1e-10,
                ..StaticResponseOpts::default()
            },
        )
        .unwrap_or_else(|e| panic!("[{label}] the untyped static solve refused: {e}"))
        .alpha;
        assert_eq!(alpha_untyped.len(), n_all * n_all);
        let jopts = TransitionMomentOpts {
            tt_grad_tol: 1e-6,
            ..TransitionMomentOpts::default()
        };
        let (block_0, rep_0) = joint_irrep_basis(&jctx, pg, space.target, space.target);
        let tm_0 = transition_moments(&jctx, &block_0, &props_all, &jopts)
            .unwrap_or_else(|e| panic!("[{label}] the in-sector joint block refused: {e}"));
        println!(
            "  [{label}] in-sector block: kept {} leakage {:e} sigma_min {:e}",
            rep_0.kept, rep_0.max_leakage, rep_0.sigma_min
        );
        let mut alpha_blocks_sum: Vec<f64> = tm_0.alpha_sos.clone();

        let ctx_c1 = weingarten_operator(&h_c1, &x_c1);
        let (bl_c1, br_c1) = ctx_c1.bases();
        let basis_c1 = tangent_basis(bl_c1, br_c1, true);

        for (ci, (phi, pa, pb)) in per_channel.iter().enumerate() {
            let phi = *phi;
            let target = space.target.fuse(phi);
            let props = [pa.clone(), pb.clone(), m_a.clone()];
            let ch = Channel::new(space, &t, &v, phi);
            let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
            let dim = sc.dim();
            assert!(
                dim > 0,
                "[{label}] phi {phi:?}: empty channel in the live list"
            );
            let basis = sc.basis();
            let (ga, gb) = sc.grams(&basis);
            let ratio = frobenius(&gb) / frobenius(&ga);
            let is_curved = curved.contains(&phi);
            println!(
                "  [{label}] phi {phi:?} -> {target:?}: dim {dim} basis {} ||B||_F/||A||_F {ratio:e} \
                 slots {:?}",
                basis.len(),
                slot_sites(&space, &bl.ranks, phi)
            );
            if is_curved && curved_required {
                assert!(
                    ratio > 1e-3,
                    "[{label}] phi {phi:?}: anti-vacuity, the curved channel has ||B||/||A|| = {ratio:e}"
                );
            }

            assert_eq!(property_flux(pa, pg).expect("single class"), Some(phi));
            assert_eq!(
                property_flux(&m_a, pg).expect("single class"),
                Some(Irrep::IDENTITY)
            );
            let p_a = sc
                .property_vector(pa)
                .expect("in-channel flux must be accepted");
            assert_eq!(p_a.phi(), phi);
            assert!(
                p_a.norm() > 1e-6,
                "[{label}] phi {phi:?}: anti-vacuity, ||p_mu|| = {:e}",
                p_a.norm()
            );

            let st = sc
                .polarizability_static(&props, &tight_static())
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: static refused: {e:?}"));
            assert_eq!(st.alpha.len(), 9);
            assert_eq!(st.dim, dim);
            assert_eq!((st.phi, st.target), (phi, target));
            assert!(
                st.alpha[0] > 1e-4 && st.alpha[4] > 1e-4,
                "[{label}] phi {phi:?}: anti-vacuity, alpha_aa = {:e}, alpha_bb = {:e} (SYNTHETIC)",
                st.alpha[0],
                st.alpha[4]
            );
            for k in 0..3 {
                assert_eq!(
                    (st.alpha[k * 3 + 2], st.alpha[2 * 3 + k]),
                    (0.0, 0.0),
                    "[{label}] phi {phi:?}: the IDENTITY-flux row/col must be a STRUCTURAL zero"
                );
            }
            assert_eq!(
                st.solves[2].grad_norm, 0.0,
                "[{label}] m_A: grad_norm must be exactly 0"
            );
            assert_eq!(
                st.solves[2].iters, 0,
                "[{label}] m_A: no solve may be paid for a zero row"
            );
            let scale_alpha = max_abs(&st.alpha, 1.0);
            assert!(
                st.alpha_defect <= 100.0 * 1e-10 * scale_alpha,
                "[{label}] phi {phi:?}: alpha_defect {:e}",
                st.alpha_defect
            );

            let tm = sc
                .transition_moments_dense(&props, &moment_opts())
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: dense moments refused: {e:?}"));
            print_moments(&format!("{label} phi {phi:?}"), &tm);
            assert_eq!(
                tm.n_neg, 0,
                "[{label}] phi {phi:?}: fixture precondition — Stable"
            );
            assert!(tm.complete_basis);
            assert_eq!(tm.omega.len(), basis.len());
            assert_eq!(tm.moments.len(), 3);
            assert!(
                tm.moments[2].iter().all(|&z| z == 0.0),
                "[{label}] the IDENTITY-flux moment row must be exactly zero"
            );
            let tmax = max_abs(&tm.moments[0], 0.0).max(max_abs(&tm.moments[1], 0.0));
            assert!(
                tmax > 1e-4,
                "[{label}] phi {phi:?}: anti-vacuity, max|T| = {tmax:e}"
            );
            let alpha_sos = tm
                .alpha_sos
                .as_ref()
                .expect("n_neg == 0 ⇒ alpha_sos is Some");
            for (k, (&sos, &a)) in alpha_sos.iter().zip(&st.alpha).enumerate() {
                assert_rel(
                    sos,
                    a,
                    1e-8,
                    &format!("[{label}] phi {phi:?}: alpha_sos[{k}] vs E2"),
                );
            }
            let f_same = oscillator_strengths(&tm.omega, &tm.moments);
            assert_eq!(
                tm.oscillator, f_same,
                "[{label}] oscillator must be the verbatim reuse"
            );

            let (block_j, rep) = joint_irrep_basis(&jctx, pg, target, space.target);
            println!(
                "  [{label}] phi {phi:?} joint block: kept {} dropped {} leakage {:e} closure {}",
                rep.kept, rep.dropped, rep.max_leakage, rep.min_closure
            );
            assert_eq!(
                block_j.len(),
                dim,
                "[{label}] phi {phi:?}: the untyped irrep block and T^phi disagree on the DIMENSION"
            );
            let tmj = transition_moments(&jctx, &block_j, &props_all, &jopts).unwrap_or_else(|e| {
                panic!("[{label}] phi {phi:?}: the joint block moments refused ({e}) — fixture precondition")
            });
            let ia = 2 * ci;
            let ib = 2 * ci + 1;
            let idx = [ia, ib, n_all - 1];
            let scale_u = max_abs(&alpha_untyped, 1.0);
            for (r, &i) in idx.iter().enumerate() {
                for (c, &j) in idx.iter().enumerate() {
                    let want = tmj.alpha_sos[i * n_all + j];
                    let got = st.alpha[r * 3 + c];
                    assert!(
                        (got - want).abs() <= 1e-10 * scale_u,
                        "[{label}] phi {phi:?}: alpha^phi[{r},{c}] = {got:.15e} vs T1a block alpha_SOS \
                         {want:.15e} (scale {scale_u:e})"
                    );
                }
            }
            for i in 0..n_all {
                for j in 0..n_all {
                    if !idx.contains(&i) || !idx.contains(&j) {
                        assert!(
                            tmj.alpha_sos[i * n_all + j].abs() <= 1e-10 * scale_u,
                            "[{label}] phi {phi:?}: T1a block alpha_SOS[{i},{j}] = {:e} for an \
                             out-of-channel pair must be roundoff",
                            tmj.alpha_sos[i * n_all + j]
                        );
                    }
                }
            }
            for (acc, add) in alpha_blocks_sum.iter_mut().zip(&tmj.alpha_sos) {
                *acc += add;
            }
            assert_moments_agree(
                &tm.omega,
                &tm.moments[..2],
                &tmj.omega,
                &[tmj.moments[ia].clone(), tmj.moments[ib].clone()],
                1e-8,
                &format!("[{label}] phi {phi:?}: E3 vs T1a"),
            );
            let f_j = oscillator_strengths(
                &tmj.omega,
                &[
                    tmj.moments[ia].clone(),
                    tmj.moments[ib].clone(),
                    tmj.moments[n_all - 1].clone(),
                ],
            );
            let cl = clusters_of(&tm.omega, 1e-6);
            for (c, &(lo, hi)) in cl.iter().enumerate() {
                let g: f64 = tm.oscillator[lo..hi].iter().sum();
                let w: f64 = f_j[lo..hi].iter().sum();
                assert_rel(
                    g,
                    w,
                    1e-8,
                    &format!("[{label}] phi {phi:?}: f over cluster {c}"),
                );
            }

            let cands: Vec<TtTangent<f64>> = basis_c1
                .iter()
                .map(|b| project_irrep(b, bl_c1, br_c1, pg, target).0)
                .collect();
            let block = mgs_tangents(cands);
            let n = block.len();
            assert_eq!(n, dim, "[{label}] phi {phi:?}: T1b block dimension");
            let (a1, b1) = hessian_grams(&ctx_c1, &block);
            let hrr: Vec<f64> = a1.iter().zip(&b1).map(|(p, q)| p + q).collect();
            let hii: Vec<f64> = a1.iter().zip(&b1).map(|(p, q)| p - q).collect();
            let gamma = |p: &PropertyOperator<f64>| -> Vec<f64> {
                let mx = &property_hamiltonian(space_c1, p) * &x_c1;
                let g = TtTangent::project(bl_c1, br_c1, &mx);
                block.iter().map(|b| b.dot(&g)).collect()
            };
            let (g_a, g_b) = (gamma(pa), gamma(pb));
            let (xi_a, xi_b) = (dense_pd_solve(&hrr, n, &g_a), dense_pd_solve(&hrr, n, &g_b));
            let want = [
                2.0 * dot_vec(&g_a, &xi_a),
                2.0 * dot_vec(&g_a, &xi_b),
                2.0 * dot_vec(&g_b, &xi_a),
                2.0 * dot_vec(&g_b, &xi_b),
            ];
            for (k, (r, c)) in [(0, 0), (0, 1), (1, 0), (1, 1)].into_iter().enumerate() {
                assert!(
                    (st.alpha[r * 3 + c] - want[k]).abs() <= 1e-10 * scale_u,
                    "[{label}] phi {phi:?}: alpha^phi[{r},{c}] = {:.15e} vs T1b dense {:.15e}",
                    st.alpha[r * 3 + c],
                    want[k]
                );
            }
            let sw = williamson_reduced_signed_vectors(&hrr, &hii, n, 1e-10)
                .expect("[T1b] the untyped block must admit a signed Williamson basis");
            assert_eq!(
                sw.n_neg, 0,
                "[{label}] phi {phi:?}: T1b block precondition — Stable"
            );
            let n_pos = sw.omega.len();
            let t_of = |g: &[f64]| -> Vec<f64> {
                (0..n_pos)
                    .map(|m| (0..n).map(|i| g[i] * sw.s_r[i * n_pos + m]).sum())
                    .collect()
            };
            assert_moments_agree(
                &tm.omega,
                &tm.moments[..2],
                &sw.omega,
                &[t_of(&g_a), t_of(&g_b)],
                1e-8,
                &format!("[{label}] phi {phi:?}: E3 vs T1b"),
            );
        }

        let scale_u = max_abs(&alpha_untyped, 1.0);
        let worst = alpha_untyped
            .iter()
            .zip(&alpha_blocks_sum)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        println!(
            "  [{label}] block-diagonality: max|alpha_untyped - sum of blocks| = {worst:e} (scale {scale_u:e})"
        );
        assert!(
            worst <= 1e-10 * scale_u,
            "[{label}] alpha_untyped != sum over irrep blocks: worst {worst:e}"
        );
    }
}

type ChannelSelect<'a> = &'a dyn Fn(Irrep, &[Irrep], &[usize]) -> bool;

struct T1Leg {
    label: &'static str,
    fx: Fixture,
    curved_required: bool,
}

#[test]
fn tso_untyped_embedding_oracle_certifies_exact_grounds() {
    t1_untyped_embedding_oracle(vec![
        T1Leg {
            label: "synth_pg (2,1) exact ground, B == 0, proper-subspace channel",
            fx: synth_exact_ground(),
            curved_required: false,
        },
        T1Leg {
            label: "d6_pg (3,0) polarized exact ground, B == 0, dim == dets",
            fx: d6_polarized_ground(),
            curved_required: false,
        },
    ]);
}

#[test]
fn tso_untyped_embedding_oracle_certifies_the_curved_minimizer() {
    t1_untyped_embedding_oracle(vec![T1Leg {
        label: "d6_minimizer (2,2), curved phi = 3",
        fx: d6_minimizer(),
        curved_required: true,
    }]);
}

#[test]
fn tso_dense_sum_over_states_pins_full_rank_channels() {
    let fx4 = {
        let pg = fx4_pg();
        let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
        let (t, v) = fx4_tv(pg);
        let (x, _e0, _) = exact_typed_ground(space, &t, &v);
        let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        (space, t, v, ham, x)
    };
    let mut identity_channels = 0usize;
    let mut skipped_channels = 0usize;
    for (label, (space, t, v, ham, x)) in [
        ("d6_pg (3,0) polarized exact ground", d6_polarized_ground()),
        ("fx4_pg (2,1) exact ground", fx4),
    ] {
        let pg = space.pg;
        let d = space.d;
        let (bl, br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let rperp = residual_norm(&ham, &x);
        println!(
            "tso T2 [{label}]: sigma_min {:e} ||r_perp|| {rperp:e}",
            min_schmidt(&bl, &br)
        );
        assert!(
            rperp <= 1e-12,
            "[{label}] fixture premise: exact eigenvector, r_perp = {rperp:e}"
        );
        let dets_0 = dets_in_irrep(d, space.nup, space.ndn, pg, space.target);
        let (ev0, vec0) = dense_eigen(&dense_hamiltonian_matrix(d, &t, &v, &dets_0));
        let e0 = ev0[0];
        let psi0 = &vec0[0];

        for phi in live_channels(&space, &bl.ranks) {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let target = space.target.fuse(phi);
            let p = phi.index() as u64;
            let (Some(pa), Some(pb)) = (
                synthetic_property(pg, d, phi, 0xE300 + p, &format!("mu_{p}_a")),
                synthetic_property(pg, d, phi, 0xE400 + p, &format!("mu_{p}_b")),
            ) else {
                continue;
            };
            let props = [pa.clone(), pb.clone()];
            let ch = Channel::new(space, &t, &v, phi);
            let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
            let dim = sc.dim();
            let dets_t = dets_in_irrep(d, space.nup, space.ndn, pg, target);
            let (evt, vect) = dense_eigen(&dense_hamiltonian_matrix(d, &t, &v, &dets_t));
            let tm = sc
                .transition_moments_dense(&props, &moment_opts())
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: dense moments refused: {e:?}"));
            print_moments(&format!("T2 {label} phi {phi:?}"), &tm);
            assert_eq!(tm.n_neg, 0, "[{label}] phi {phi:?}: Stable precondition");
            println!(
                "  [{label}] phi {phi:?}: dim T^phi = {dim}, sector dets = {} -> {}",
                dets_t.len(),
                if dim == dets_t.len() {
                    "IDENTITY leg"
                } else {
                    "SKIP leg (interlacing only)"
                }
            );
            let w_low = evt[0] - e0;
            if dim != dets_t.len() {
                skipped_channels += 1;
                assert!(
                    tm.omega[0] >= w_low - 1e-8,
                    "[{label}] phi {phi:?}: a compression's lowest omega ({}) sits BELOW the sector's \
                     lowest gap ({w_low})",
                    tm.omega[0]
                );
                continue;
            }
            identity_channels += 1;

            let mut union = dets_0.clone();
            union.extend(dets_t.iter().cloned());
            let n0 = dets_0.len();
            let nt = dets_t.len();
            let t_dense = |pr: &PropertyOperator<f64>| -> Vec<f64> {
                let mu = dense_kappa_matrix(d, &pr.m, &union);
                (0..nt)
                    .map(|n| {
                        let psi_n = &vect[n];
                        let mut acc = 0.0;
                        for (r, psi_nr) in psi_n.iter().enumerate() {
                            for (c, psi_0c) in psi0.iter().enumerate() {
                                acc += psi_nr * mu[n0 + r][c] * psi_0c;
                            }
                        }
                        acc
                    })
                    .collect()
            };
            let ta = t_dense(&pa);
            let tb = t_dense(&pb);
            let omega_want: Vec<f64> = evt.iter().map(|e| e - e0).collect();
            assert_moments_agree(
                &tm.omega,
                &tm.moments,
                &omega_want,
                &[ta.clone(), tb.clone()],
                1e-8,
                &format!("[{label}] phi {phi:?}: E3 vs dense CI"),
            );
            let sos = |u: &[f64], w: &[f64]| -> f64 {
                2.0 * (0..nt).map(|n| u[n] * w[n] / omega_want[n]).sum::<f64>()
            };
            let want_alpha = [sos(&ta, &ta), sos(&ta, &tb), sos(&tb, &ta), sos(&tb, &tb)];
            let st = sc
                .polarizability_static(&props, &tight_static())
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: static refused: {e:?}"));
            for (k, &w) in want_alpha.iter().enumerate() {
                assert_rel(
                    st.alpha[k],
                    w,
                    1e-8,
                    &format!("[{label}] phi {phi:?}: alpha[{k}] vs dense SOS"),
                );
            }
            assert!(
                max_abs(&st.alpha, 0.0) > 1e-4,
                "[{label}] phi {phi:?}: anti-vacuity, max|alpha| = {:e}",
                max_abs(&st.alpha, 0.0)
            );
            let f_want: Vec<f64> = (0..nt)
                .map(|n| (2.0 / 3.0) * omega_want[n] * (ta[n] * ta[n] + tb[n] * tb[n]))
                .collect();
            let cl = clusters_of(&tm.omega, 1e-6);
            for &(lo, hi) in &cl {
                let g: f64 = tm.oscillator[lo..hi].iter().sum();
                let w: f64 = f_want[lo..hi].iter().sum();
                assert_rel(
                    g,
                    w,
                    1e-8,
                    &format!("[{label}] phi {phi:?}: f over cluster {lo}..{hi}"),
                );
            }
            for pidx in 0..2 {
                assert!(
                    tm.capture_defect[pidx] <= 1e-6 && tm.completeness_defect[pidx] <= 1e-6,
                    "[{label}] phi {phi:?}: at dim == dets the defects must be roundoff, got \
                     capture {:e} completeness {:e}",
                    tm.capture_defect[pidx],
                    tm.completeness_defect[pidx]
                );
            }
        }
    }
    assert!(
        identity_channels >= 3 && skipped_channels >= 1,
        "anti-vacuity: the dense identity ran on {identity_channels} channels and the skip leg on \
         {skipped_channels} — both legs must be exercised"
    );
}

#[test]
fn tso_iterative_moments_match_the_dense_route() {
    let iter_opts = |tol: f64, max_iter: usize, kernel: RpaKernel| SectorMomentOpts {
        solver: SectorOpts {
            tol,
            max_iter,
            kernel,
            ..SectorOpts::default()
        },
        route: PropRoute::Materialized,
        ..SectorMomentOpts::default()
    };

    {
        let (space, t, v, ham, x) = d6_polarized_ground();
        let pg = space.pg;
        let d = space.d;
        let (bl, _) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let mut channels = 0usize;
        for phi in live_channels(&space, &bl.ranks) {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let p = phi.index() as u64;
            let (Some(pa), Some(pb)) = (
                synthetic_property(pg, d, phi, 0xF100 + p, &format!("mu_{p}_a")),
                synthetic_property(pg, d, phi, 0xF200 + p, &format!("mu_{p}_b")),
            ) else {
                continue;
            };
            let props = [pa, pb];
            let ch = Channel::new(space, &t, &v, phi);
            let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
            let dim = sc.dim();
            let dense = sc
                .transition_moments_dense(&props, &moment_opts())
                .unwrap_or_else(|e| panic!("(a) phi {phi:?}: dense refused {e:?}"));
            let it = sc
                .transition_moments_iterative(
                    &props,
                    dim,
                    &iter_opts(1e-10, 600, RpaKernel::Paired),
                )
                .unwrap_or_else(|e| panic!("(a) phi {phi:?}: iterative refused {e:?}"));
            channels += 1;
            print_moments(&format!("T-iter (a) phi {phi:?} k = dim = {dim}"), &it);
            assert!(
                !it.complete_basis,
                "the iterative route never claims completeness"
            );
            assert_eq!((it.n_neg, it.unstable_omega2.len()), (0, 0));
            assert_eq!(it.omega.len(), dim);
            assert_eq!(it.dim, dense.dim);
            assert_moments_agree(
                &it.omega,
                &it.moments,
                &dense.omega,
                &dense.moments,
                1e-7,
                &format!("(a) phi {phi:?}: iterative vs dense"),
            );
            for (pidx, lab) in ["a", "b"].iter().enumerate() {
                let p2 = it.sum_rule[pidx].1;
                assert!(p2 > 1e-12, "anti-vacuity ||p||² = {p2:e}");
                assert_rel(
                    it.sum_rule[pidx].1,
                    dense.sum_rule[pidx].1,
                    1e-12,
                    &format!("(a) phi {phi:?} {lab}: the rhs is the same ||p||² on both routes"),
                );
                assert_rel(
                    it.sum_rule[pidx].0,
                    p2,
                    1e-8,
                    &format!("(a) phi {phi:?} {lab}: Σ T² = ||p||² at k = dim, B == 0"),
                );
                assert_rel(
                    it.sum_rule_si[pidx],
                    p2,
                    1e-8,
                    &format!("(a) phi {phi:?} {lab}: the S_i-side sum = ||p||² at B == 0"),
                );
            }
            let si = it.alpha_sos.as_ref().expect("n_neg == 0");
            let sd = dense.alpha_sos.as_ref().expect("n_neg == 0");
            let scale = max_abs(sd, 1.0);
            for k in 0..4 {
                assert!(
                    (si[k] - sd[k]).abs() <= 1e-7 * scale,
                    "(a) phi {phi:?}: alpha_sos[{k}] iterative {:.12e} vs dense {:.12e}",
                    si[k],
                    sd[k]
                );
            }
            assert!(
                it.orth_error <= 1e-8,
                "(a) phi {phi:?}: biorthogonality defect {:e}",
                it.orth_error
            );
            for pidx in 0..2 {
                assert!(
                    it.completeness_defect[pidx] <= 1e-7,
                    "(a) phi {phi:?}: at k = dim the oblique reconstruction is complete, defect {:e}",
                    it.completeness_defect[pidx]
                );
            }
            let cl = clusters_of(&dense.omega, 1e-6);
            for &(lo, hi) in &cl {
                assert_rel(
                    it.oscillator[lo..hi].iter().sum::<f64>(),
                    dense.oscillator[lo..hi].iter().sum::<f64>(),
                    1e-7,
                    &format!("(a) phi {phi:?}: f over cluster {lo}..{hi}"),
                );
            }
        }
        assert_eq!(
            channels, 3,
            "measured d6_polarized inventory: three shifted channels"
        );
    }

    {
        let (space, t, v, ham, x) = d6_minimizer();
        let pg = space.pg;
        let (bl, _) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let phi = first_curved_channel(&space, &bl.ranks);
        let p = phi.index() as u64;
        let props = [
            synthetic_property(pg, D6, phi, 0xF300 + p, "mu_a").expect("curved d6 channel"),
            synthetic_property(pg, D6, phi, 0xF400 + p, "mu_b").expect("curved d6 channel"),
        ];
        let ch = Channel::new(space, &t, &v, phi);
        let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
        let basis = sc.basis();
        let (ga, gb) = sc.grams(&basis);
        let ratio = frobenius(&gb) / frobenius(&ga);
        assert!(ratio > 1e-3, "curved premise: ||B||/||A|| = {ratio:e}");
        let dense = sc
            .transition_moments_dense(&props, &moment_opts())
            .expect("dense on the curved channel");
        assert_eq!(dense.n_neg, 0, "Stable premise");
        print_moments("T-iter (b) dense", &dense);
        let cl = clusters_of(&dense.omega, 1e-6);
        let dsos = dense.alpha_sos.as_ref().expect("Stable");
        let mut prev_sum = [0.0f64; 2];
        let mut prev_sos = [0.0f64; 2];
        let mut prev_def = [f64::INFINITY; 2];
        let mut singleton_pins = 0usize;
        for k in 1..=3usize {
            let it = sc
                .transition_moments_iterative(&props, k, &iter_opts(1e-10, 600, RpaKernel::Paired))
                .unwrap_or_else(|e| panic!("(b) k = {k}: iterative refused {e:?}"));
            print_moments(&format!("T-iter (b) k = {k}"), &it);
            assert!(!it.complete_basis);
            assert_eq!(it.n_neg, 0);
            assert_eq!(it.omega.len(), k, "k lowest modes");
            for n in 0..k {
                assert_rel(
                    it.omega[n],
                    dense.omega[n],
                    1e-8,
                    &format!("(b) k = {k}: omega[{n}] vs the dense lowest"),
                );
            }
            for &(lo, hi) in &cl {
                if hi > k {
                    continue;
                }
                for pidx in 0..2 {
                    let scale = max_abs(&dense.moments[pidx], 1.0);
                    if hi - lo == 1 {
                        assert!(
                            (it.moments[pidx][lo].abs() - dense.moments[pidx][lo].abs()).abs()
                                <= 1e-7 * scale,
                            "(b) k = {k}: |T_{lo}| iterative {:.12e} vs dense {:.12e}",
                            it.moments[pidx][lo].abs(),
                            dense.moments[pidx][lo].abs()
                        );
                        singleton_pins += 1;
                    } else {
                        let a: f64 = (lo..hi).map(|n| it.moments[pidx][n].powi(2)).sum();
                        let b: f64 = (lo..hi).map(|n| dense.moments[pidx][n].powi(2)).sum();
                        assert!(
                            (a - b).abs() <= 1e-7 * scale * scale,
                            "(b) k = {k}: cluster {lo}..{hi} Σ T² {a:.12e} vs dense {b:.12e}"
                        );
                    }
                }
            }
            let sos = it.alpha_sos.as_ref().expect("Stable");
            for pidx in 0..2 {
                let s = it.sum_rule[pidx].0;
                let s_dense = dense.sum_rule[pidx].0;
                assert!(
                    s >= prev_sum[pidx] - 1e-12 && s <= s_dense * (1.0 + 1e-9),
                    "(b) k = {k}: partial Σ T² {s:.12e} (previous {:.12e}, dense {s_dense:.12e})",
                    prev_sum[pidx]
                );
                prev_sum[pidx] = s;
                let a = sos[pidx * 2 + pidx];
                let a_dense = dsos[pidx * 2 + pidx];
                assert!(
                    a >= prev_sos[pidx] - 1e-12 && a <= a_dense + 1e-9 * a_dense.abs().max(1.0),
                    "(b) k = {k}: partial alpha_sos {a:.12e} (previous {:.12e}, dense {a_dense:.12e})",
                    prev_sos[pidx]
                );
                prev_sos[pidx] = a;
                let cd = it.completeness_defect[pidx];
                assert!(
                    cd.is_finite() && cd >= 0.0,
                    "(b) k = {k}: completeness_defect {cd:e}"
                );
                println!(
                    "  T-iter (b) k = {k} property {pidx}: completeness_defect {cd:.6e} (previous {:.6e}) — oblique, not monotone",
                    prev_def[pidx]
                );
                prev_def[pidx] = cd;
            }
            assert!(
                it.orth_error <= 1e-8,
                "(b) k = {k}: orth_error {:e}",
                it.orth_error
            );
        }
        assert!(
            singleton_pins >= 2,
            "anti-vacuity: the per-mode |T_n| pin ran on {singleton_pins} singleton modes"
        );

        let rr = sc
            .transition_moments_iterative(
                &props,
                3,
                &iter_opts(1e-10, 600, RpaKernel::signed(RpaMetric::Rr)),
            )
            .expect("(c) Rr");
        let ii = sc
            .transition_moments_iterative(
                &props,
                3,
                &iter_opts(1e-10, 600, RpaKernel::signed(RpaMetric::Ii)),
            )
            .expect("(c) Ii");
        let pa = sc
            .transition_moments_iterative(&props, 3, &iter_opts(1e-10, 600, RpaKernel::Paired))
            .expect("(c) paired");
        assert_eq!(
            (rr.metric, ii.metric, pa.metric),
            (RpaMetricBlock::Rr, RpaMetricBlock::Ii, RpaMetricBlock::Ii)
        );
        for (other, what) in [(&ii, "(c) Rr vs Ii"), (&pa, "(c) Rr vs paired")] {
            assert_moments_agree(
                &rr.omega,
                &rr.moments,
                &other.omega,
                &other.moments,
                1e-7,
                what,
            );
            for pidx in 0..2 {
                assert_rel(
                    rr.sum_rule[pidx].0,
                    other.sum_rule[pidx].0,
                    1e-7,
                    &format!("{what}: Σ T²"),
                );
            }
        }

        match sc.transition_moments_iterative(&props, 1, &iter_opts(1e-10, 4, RpaKernel::Paired)) {
            Err(SectorResponseError::SolveNotConverged {
                label,
                iters,
                rel_residual,
                tol,
            }) => {
                assert!(
                    label.contains("mu_a") && label.contains("mu_b"),
                    "the shared eigen-solve names every property: {label:?}"
                );
                assert!(iters >= 1, "iters {iters}");
                assert!(
                    rel_residual.is_finite() && rel_residual > tol,
                    "rel_residual {rel_residual:e} vs tol {tol:e}"
                );
                assert_eq!(tol, 1e-10);
                println!(
                    "T-iter (e): SolveNotConverged after {iters} vectors, residual {rel_residual:e}"
                );
            }
            other => panic!("(e) max_iter = 4 must refuse SolveNotConverged, got {other:?}"),
        }
    }

    {
        let (space, t, v, ham, x) = d6_polarized_ground();
        let pg = space.pg;
        let (bl, _) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let phi = live_channels(&space, &bl.ranks)
            .into_iter()
            .find(|&p| p != Irrep::IDENTITY)
            .expect("a live shifted channel");
        let pa = synthetic_property(pg, D6, phi, 0xF600, "mu_k0").expect("d6 pairs");
        let ch = Channel::new(space, &t, &v, phi);
        let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
        let p2 = sc.property_vector(&pa).expect("in-channel").norm().powi(2);
        assert!(p2 > 1e-12, "anti-vacuity ||p||² = {p2:e}");
        let it = sc
            .transition_moments_iterative(
                std::slice::from_ref(&pa),
                0,
                &iter_opts(1e-10, 600, RpaKernel::Paired),
            )
            .expect("k = 0: an empty table, not a refusal");
        assert!(it.omega.is_empty() && it.moments[0].is_empty() && it.oscillator.is_empty());
        assert!(!it.complete_basis);
        assert_eq!(it.sum_rule[0].0, 0.0);
        assert_rel(
            it.sum_rule[0].1,
            p2,
            1e-14,
            "(d) k = 0: sum_rule.1 is ||p||²",
        );
        assert_eq!(it.sum_rule_si, vec![0.0]);
        assert_eq!(it.alpha_sos, Some(vec![0.0]));
        assert_eq!((it.n_neg, it.dim), (0, sc.dim()));

        let (space, pg, t, v, ham, x) = lih_typed_fixture();
        let (space, pg) = (*space, *pg);
        let (bl, _) = tangent_bases(x);
        let live = live_channels(&space, &bl.ranks);
        let phi_e = (1..pg.order())
            .map(Irrep::new)
            .find(|p| !live.contains(p))
            .expect("measured LiH inventory: one empty shifted channel");
        let pe = synthetic_property(pg, space.d, phi_e, 0xF6FF, "mu_empty")
            .expect("measured: pairs(Irrep(1)) = [(3, 4)] on LiH's table");
        let ctx = weingarten_operator(ham, x);
        let ch = Channel::new(space, t, v, phi_e);
        let sc = SectorCtx::new(&ctx, &ch.h_out, phi_e);
        assert_eq!(sc.dim(), 0);
        let it = sc
            .transition_moments_iterative(
                std::slice::from_ref(&pe),
                2,
                &iter_opts(1e-10, 600, RpaKernel::Paired),
            )
            .expect("an EMPTY channel is an empty table, not a refusal");
        assert_eq!(it.dim, 0);
        assert!(it.omega.is_empty() && it.moments[0].is_empty());
        assert!(!it.complete_basis);
        assert_eq!(it.sum_rule[0], (0.0, 0.0));
        assert_eq!(it.capture_defect[0], 1.0);
        assert_eq!(it.alpha_sos, Some(vec![0.0]));
    }
}

#[test]
fn tso_iterative_moments_honour_the_sector_precond() {
    let (space, t, v, ham, x) = d6_minimizer();
    let pg = space.pg;
    let (bl, _) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let phi = first_curved_channel(&space, &bl.ranks);
    let p = phi.index() as u64;
    let props = [
        synthetic_property(pg, D6, phi, 0xF300 + p, "mu_a").expect("curved d6 channel"),
        synthetic_property(pg, D6, phi, 0xF400 + p, "mu_b").expect("curved d6 channel"),
    ];
    let ch = Channel::new(space, &t, &v, phi);
    let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
    let k = 3usize;
    let run = |kernel: RpaKernel, precond: Option<SympPrecond>| {
        let mut solver = SectorOpts {
            tol: 1e-10,
            max_iter: 600,
            kernel,
            ..SectorOpts::default()
        };
        if let Some(pc) = precond {
            solver.precond = pc;
        }
        let opts = SectorMomentOpts {
            solver,
            route: PropRoute::Materialized,
            ..SectorMomentOpts::default()
        };
        sc.transition_moments_iterative(&props, k, &opts)
            .unwrap_or_else(|e| panic!("precond {precond:?}: the iterative route refused {e:?}"))
    };
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let dflt = run(kernel, None);
        let off = run(kernel, Some(SympPrecond::Off));
        let d0 = run(kernel, Some(SympPrecond::DiagUnshifted { kappa: 0.0 }));
        print_moments(&format!("T-iter precond {kernel:?}: default (Auto)"), &dflt);
        print_moments(&format!("T-iter precond {kernel:?}: Off"), &off);
        print_moments(&format!("T-iter precond {kernel:?}: D0"), &d0);

        let sums = |s: &[(f64, f64)]| s.iter().flat_map(|&(a, b)| [a, b]).collect::<Vec<f64>>();
        assert!(
            flat_bits(&dflt.omega) == flat_bits(&off.omega),
            "default vs Off: omega {:?} vs {:?}",
            dflt.omega,
            off.omega
        );
        for pidx in 0..2 {
            assert!(
                flat_bits(&dflt.moments[pidx]) == flat_bits(&off.moments[pidx]),
                "default vs Off: moments of property {pidx}"
            );
        }
        assert!(
            flat_bits(&dflt.oscillator) == flat_bits(&off.oscillator),
            "default vs Off: oscillator strengths"
        );
        assert!(
            flat_bits(&sums(&dflt.sum_rule)) == flat_bits(&sums(&off.sum_rule))
                && flat_bits(&dflt.sum_rule_si) == flat_bits(&off.sum_rule_si),
            "default vs Off: sum rules"
        );
        assert_eq!(
            dflt.alpha_sos.as_deref().map(flat_bits),
            off.alpha_sos.as_deref().map(flat_bits),
            "default vs Off: alpha_sos"
        );
        assert_eq!(dflt.metric, off.metric, "default vs Off: metric");

        assert_eq!(
            (d0.omega.len(), off.omega.len()),
            (k, k),
            "k lowest modes on both"
        );
        assert_eq!((d0.n_neg, off.n_neg), (0, 0), "Stable channel");
        assert_moments_agree(
            &d0.omega,
            &d0.moments,
            &off.omega,
            &off.moments,
            1e-7,
            "D0 vs Off",
        );
        for pidx in 0..2 {
            assert_rel(
                d0.sum_rule[pidx].0,
                off.sum_rule[pidx].0,
                1e-7,
                &format!("D0 vs Off: partial Σ T² of property {pidx}"),
            );
        }
        assert!(
            flat_bits(&d0.omega) != flat_bits(&off.omega)
                || (0..2).any(|pidx| flat_bits(&d0.moments[pidx]) != flat_bits(&off.moments[pidx])),
            "non-vacuity: the D0 run is bitwise the Off run — the preconditioner never reached the \
             kernel"
        );
    }
}

#[test]
fn tso_operator_sum_rule_pinned_at_b_zero_and_reported_elsewhere() {
    struct Leg {
        label: &'static str,
        fx: Fixture,
        strong: bool,
    }
    let legs = [
        Leg {
            label: "synth_pg (2,1) exact ground",
            fx: synth_exact_ground(),
            strong: false,
        },
        Leg {
            label: "d6_pg (3,0) polarized exact ground",
            fx: d6_polarized_ground(),
            strong: true,
        },
        Leg {
            label: "d6_minimizer",
            fx: d6_minimizer(),
            strong: false,
        },
    ];
    let mut pinned = 0usize;
    let mut single_slot_at_residual = 0usize;
    let mut curved_defects = 0usize;
    for Leg {
        label,
        fx: (space, t, v, ham, x),
        strong,
    } in legs
    {
        let pg = space.pg;
        let d = space.d;
        let (bl, _br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let rperp = residual_norm(&ham, &x);
        let curved = curved_channels(&space, &bl.ranks);
        let mut f_total = 0.0f64;
        for phi in live_channels(&space, &bl.ranks) {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let p = phi.index() as u64;
            let Some(pa) = synthetic_property(pg, d, phi, 0xE500 + p, &format!("mu_{p}")) else {
                continue;
            };
            let ch = Channel::new(space, &t, &v, phi);
            let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
            let basis = sc.basis();
            let (ga, gb) = sc.grams(&basis);
            let ratio = frobenius(&gb) / frobenius(&ga);
            let slots = slot_sites(&space, &bl.ranks, phi);
            let p_vec = sc.property_vector(&pa).expect("in-channel flux");
            let p2 = p_vec.norm().powi(2);
            assert!(
                p2.sqrt() > 1e-6,
                "[{label}] phi {phi:?}: anti-vacuity ||p|| = {:e}",
                p2.sqrt()
            );
            let tm = sc
                .transition_moments_dense(std::slice::from_ref(&pa), &moment_opts())
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: refused {e:?}"));
            assert_eq!(tm.n_neg, 0, "[{label}] phi {phi:?}: Stable precondition");
            let sum_t2: f64 = tm.moments[0].iter().map(|z| z * z).sum();
            let (rule_lhs, rule_rhs) = tm.sum_rule[0];
            println!(
                "tso T3 [{label}] phi {phi:?}: SYNTHETIC dim {} slots {slots:?} ||B||_F/||A||_F {ratio:e} \
                 ||r_perp|| {rperp:e} sum T^2 {sum_t2:.12e} ||p||^2 {p2:.12e} field ({rule_lhs:.12e}, \
                 {rule_rhs:.12e}) capture_defect {:.3e}",
                tm.dim, tm.capture_defect[0]
            );
            assert_rel(
                rule_lhs,
                sum_t2,
                1e-12,
                &format!("[{label}] phi {phi:?}: sum_rule.0 vs Σ T²"),
            );
            assert_rel(
                rule_rhs,
                p2,
                1e-12,
                &format!("[{label}] phi {phi:?}: sum_rule.1 vs ||p||²"),
            );
            let b_zero = rperp <= 1e-12 || slots.len() == 1;
            if b_zero {
                assert!(
                    ratio <= 1e-12,
                    "[{label}] phi {phi:?}: B must vanish here (r_perp {rperp:e}, slots {slots:?}), ratio {ratio:e}"
                );
                assert_rel(
                    sum_t2,
                    p2,
                    1e-10,
                    &format!("[{label}] phi {phi:?}: operator sum rule at B == 0"),
                );
                pinned += 1;
                if rperp > 1e-3 {
                    single_slot_at_residual += 1;
                }
                if strong {
                    let mx = apply_cross_target(&x, &pa.m, pg);
                    assert_eq!(mx.space, sc.space_out());
                    let mx2 = dot(&mx, &mx);
                    let dets_t =
                        dets_in_irrep(d, space.nup, space.ndn, pg, space.target.fuse(phi)).len();
                    assert_eq!(
                        tm.dim, dets_t,
                        "[{label}] the strong form needs dim == dets"
                    );
                    assert_rel(
                        sum_t2,
                        mx2,
                        1e-10,
                        &format!("[{label}] phi {phi:?}: STRONG sum rule Σ T² = ||mu x||²"),
                    );
                    assert!(
                        tm.capture_defect[0] <= 1e-6,
                        "[{label}] capture_defect {:e}",
                        tm.capture_defect[0]
                    );
                }
            } else {
                assert!(
                    curved.contains(&phi) && ratio > 1e-3,
                    "[{label}] phi {phi:?}: B != 0 expected here"
                );
                assert!(
                    sum_t2.is_finite() && sum_t2 > 0.0,
                    "[{label}] phi {phi:?}: Σ T² = {sum_t2}"
                );
                assert!(
                    sum_t2 <= p2 * 11.0,
                    "[{label}] phi {phi:?}: Σ T² = {sum_t2:e} exceeds the generous bound 11·||p||² = {:e}",
                    11.0 * p2
                );
                println!(
                    "  [{label}] phi {phi:?}: operator sum rule DEFECT (Σ T² − ||p||²)/||p||² = {:e} — an \
                     inequality on a curved channel, reported, not asserted zero",
                    (sum_t2 - p2) / p2
                );
                curved_defects += 1;
            }
            let mut acc = 0.0f64;
            for (n, &f) in tm.oscillator.iter().enumerate() {
                assert!(
                    f >= 0.0 && f.is_finite(),
                    "[{label}] phi {phi:?}: f_{n} = {f}"
                );
                let next = acc + f;
                assert!(next >= acc, "[{label}] partial sums must be non-decreasing");
                acc = next;
            }
            f_total += acc;
        }
        println!(
            "tso T3 [{label}]: Σ_channels Σ_n f_n = {f_total:.6e} beside N_e = {} — not TRK: synthetic \
             operator, incomplete basis, in-sector channel excluded (REPORTED, not asserted)",
            space.nup + space.ndn
        );
    }
    assert!(
        pinned >= 3 && single_slot_at_residual >= 2 && curved_defects >= 1,
        "anti-vacuity: pinned {pinned}, single-slot-at-residual {single_slot_at_residual}, curved \
         defects {curved_defects}"
    );
}

#[test]
fn tso_proper_subspace_channel_reports_defects_and_two_emptinesses() {
    let (space, pg, t, v, ham, x) = lih_typed_fixture();
    let (space, pg) = (*space, *pg);
    let d = space.d;
    let (bl, br) = tangent_bases(x);
    let ctx = weingarten_operator(ham, x);
    let irreps: Vec<Irrep> = (0..d).map(|k| pg.irrep_of(k)).collect();
    println!(
        "tso T5 LiH: site irreps {irreps:?} target {:?} sigma_min {:e} ||r_perp|| {:e}",
        space.target,
        min_schmidt(&bl, &br),
        residual_norm(ham, x)
    );
    let live: Vec<Irrep> = live_channels(&space, &bl.ranks)
        .into_iter()
        .filter(|&p| p != Irrep::IDENTITY)
        .collect();
    let empty: Vec<Irrep> = (1..pg.order())
        .map(Irrep::new)
        .filter(|p| !live.contains(p))
        .collect();
    println!("  live {live:?} empty {empty:?}");
    assert_eq!(
        live.len(),
        2,
        "measured LiH inventory: two live shifted channels"
    );
    assert_eq!(
        empty.len(),
        1,
        "measured LiH inventory: one empty shifted channel"
    );

    for &phi in &live {
        let p = phi.index() as u64;
        let pa = synthetic_property(pg, d, phi, 0xE600 + p, &format!("mu_{p}"))
            .expect("both live LiH channels carry representable pairs (measured)");
        let ch = Channel::new(space, t, v, phi);
        let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
        let dets = dets_in_irrep(d, space.nup, space.ndn, pg, space.target.fuse(phi)).len();
        let dim = sc.dim();
        let slots = slot_sites(&space, &bl.ranks, phi);
        assert!(dim <= dets, "containment: dim {dim} > dets {dets}");
        assert!(
            dim < dets,
            "fixture premise (measured 12/56, 8/56): a PROPER subspace"
        );
        let tm = sc
            .transition_moments_dense(std::slice::from_ref(&pa), &moment_opts())
            .unwrap_or_else(|e| panic!("LiH phi {phi:?}: dense moments refused: {e:?}"));
        print_moments(
            &format!("LiH phi {phi:?} (dim {dim} of {dets} dets, slots {slots:?})"),
            &tm,
        );
        assert_eq!(tm.n_neg, 0, "LiH phi {phi:?}: measured Stable");
        assert_eq!(tm.dim, dim);
        for (name, z) in [
            ("capture", tm.capture_defect[0]),
            ("completeness", tm.completeness_defect[0]),
        ] {
            assert!(
                z.is_finite() && (0.0..=1.0).contains(&z),
                "LiH phi {phi:?}: {name} defect {z}"
            );
        }
        assert!(
            tm.capture_defect[0] > 1e-2,
            "LiH phi {phi:?}: anti-vacuity — the capture defect must be REAL on a proper-subspace \
             channel, got {:e}",
            tm.capture_defect[0]
        );
        assert_eq!(
            slots.len(),
            1,
            "LiH phi {phi:?}: measured single-slot channel"
        );
        let sum_t2: f64 = tm.moments[0].iter().map(|z| z * z).sum();
        let p2 = sc.property_vector(&pa).expect("in-channel").norm().powi(2);
        assert!(p2.sqrt() > 1e-6, "anti-vacuity ||p|| = {:e}", p2.sqrt());
        assert_rel(
            sum_t2,
            p2,
            1e-9,
            &format!("LiH phi {phi:?}: operator sum rule on a single-slot channel"),
        );
        let st = sc
            .polarizability_static(std::slice::from_ref(&pa), &SectorStaticOpts::default())
            .unwrap_or_else(|e| panic!("LiH phi {phi:?}: static refused: {e:?}"));
        println!(
            "  LiH phi {phi:?}: SYNTHETIC alpha^phi = {:.9e} (iters {}, rel_residual {:e})",
            st.alpha[0], st.solves[0].iters, st.solves[0].rel_residual
        );
        assert!(st.alpha[0] > 0.0 && st.alpha[0].is_finite());
        assert_rel(
            st.capture_defect[0],
            tm.capture_defect[0],
            1e-12,
            "the two entry points report ONE capture defect",
        );
    }

    let phi_e = empty[0];
    let pairs = synthetic_pairs(pg, d, phi_e);
    let dets_e = dets_in_irrep(d, space.nup, space.ndn, pg, space.target.fuse(phi_e)).len();
    println!("  empty channel {phi_e:?}: dets {dets_e}, pairs {pairs:?}");
    assert!(
        dets_e > 0,
        "the empty channel must sit over a NON-empty sector to discriminate anything"
    );
    let ch = Channel::new(space, t, v, phi_e);
    let sc = SectorCtx::new(&ctx, &ch.h_out, phi_e);
    assert_eq!(sc.dim(), 0);
    match synthetic_property(pg, d, phi_e, 0xE6FF, "mu_empty") {
        None => {
            panic!(
                "LiH: no representable observable of flux {phi_e:?} — the table {irreps:?} differs \
                 from the measured one; record the new inventory before relying on it"
            );
        }
        Some(pe) => {
            let mx = apply_cross_target(x, &pe.m, pg);
            let mx2 = dot(&mx, &mx);
            assert!(
                mx2 > 1e-8,
                "anti-vacuity: mu x must be nonzero, ||mu x||² = {mx2:e}"
            );
            let p_e = sc
                .property_vector(&pe)
                .expect("an empty channel still accepts its own flux");
            assert_eq!(p_e.phi(), phi_e);
            assert_eq!(
                p_e.norm(),
                0.0,
                "P^phi of anything is the zero of a 0-dim space"
            );
            let tm = sc
                .transition_moments_dense(std::slice::from_ref(&pe), &moment_opts())
                .expect(
                    "an EMPTY channel is a legitimate outcome, not a refusal (row (a) semantics)",
                );
            assert_eq!(tm.dim, 0);
            assert!(tm.omega.is_empty() && tm.moments[0].is_empty() && tm.oscillator.is_empty());
            assert_eq!(tm.n_neg, 0);
            assert!(tm.unstable_omega2.is_empty());
            assert_eq!(
                tm.alpha_sos,
                Some(vec![0.0]),
                "an empty sum over states is zero, not None"
            );
            assert_eq!(tm.sum_rule[0], (0.0, 0.0));
            assert_eq!(
                tm.capture_defect[0], 1.0,
                "an EMPTY channel with a nonzero mu x misses ALL of it: capture defect exactly 1"
            );
            let st = sc
                .polarizability_static(std::slice::from_ref(&pe), &SectorStaticOpts::default())
                .expect("empty channel: alpha == 0 with no apply");
            assert_eq!(st.dim, 0);
            assert_eq!(st.alpha, vec![0.0]);
            assert_eq!(
                st.solves[0].iters, 0,
                "no apply may be paid on an empty channel"
            );
            assert_eq!(st.solves[0].grad_norm, 0.0);
            assert_eq!(st.capture_defect[0], 1.0);
        }
    }
}

fn t6_dynamic_alpha(
    legs: Vec<(&'static str, Fixture)>,
    select: ChannelSelect<'_>,
) -> (usize, usize) {
    let mut channels = 0usize;
    let mut curved_seen = 0usize;
    for (label, (space, t, v, ham, x)) in legs {
        let pg = space.pg;
        let d = space.d;
        let (bl, _br) = tangent_bases(&x);
        let ctx = weingarten_operator(&ham, &x);
        let curved = curved_channels(&space, &bl.ranks);
        for phi in live_channels(&space, &bl.ranks) {
            if phi == Irrep::IDENTITY || !select(phi, &curved, &slot_sites(&space, &bl.ranks, phi))
            {
                continue;
            }
            let p = phi.index() as u64;
            let (Some(pa), Some(pb)) = (
                synthetic_property(pg, d, phi, 0xE700 + p, &format!("mu_{p}_a")),
                synthetic_property(pg, d, phi, 0xE800 + p, &format!("mu_{p}_b")),
            ) else {
                continue;
            };
            let props = [pa.clone(), pb.clone()];
            let ch = Channel::new(space, &t, &v, phi);
            let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
            channels += 1;
            if curved.contains(&phi) {
                curved_seen += 1;
            }
            let st = sc
                .polarizability_static(&props, &tight_static())
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: static refused {e:?}"));
            let tm = sc
                .transition_moments_dense(&props, &moment_opts())
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: moments refused {e:?}"));
            assert_eq!(tm.n_neg, 0, "[{label}] phi {phi:?}: Stable precondition");
            let omega = tm.omega.clone();
            let w1 = omega[0];
            let window = 1e-3;
            let scale = max_abs(&st.alpha, 1.0);
            let dyn_opts = |solver: DynamicSolver| SectorDynamicOpts {
                tol: 1e-10,
                solver,
                spectrum: Some(omega.clone()),
                resonance_window: window,
                unstable_pairs: 0,
                route: PropRoute::Materialized,
                ..SectorDynamicOpts::default()
            };
            let closed = |w: f64| -> Vec<f64> {
                let mut out = vec![0.0f64; 4];
                for r in 0..2 {
                    for c in 0..2 {
                        out[r * 2 + c] = 2.0
                            * (0..omega.len())
                                .map(|n| {
                                    tm.moments[r][n] * tm.moments[c][n] * omega[n]
                                        / (omega[n] * omega[n] - w * w)
                                })
                                .sum::<f64>();
                    }
                }
                out
            };
            println!(
                "tso T6 [{label}] phi {phi:?}: SYNTHETIC dim {} omega_1 {w1:e} curved {} alpha(0) {:?}",
                tm.dim,
                curved.contains(&phi),
                st.alpha
            );

            let r0 = sc
                .linear_response(&props, 0.0, &dyn_opts(DynamicSolver::Auto))
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: omega = 0 refused {e:?}"));
            assert_eq!(r0.omega, 0.0);
            for k in 0..4 {
                assert!(
                    (r0.alpha[k] - st.alpha[k]).abs() <= 1e-12 * scale,
                    "[{label}] phi {phi:?}: alpha(0)[{k}] = {:.15e} vs static {:.15e}",
                    r0.alpha[k],
                    st.alpha[k]
                );
            }
            for s in &r0.solves {
                assert_eq!(
                    s.v_weight, 0.0,
                    "[{label}] v_weight must be EXACTLY 0 at omega = 0"
                );
                assert!(
                    matches!(s.solver_used, DynamicSolver::Pcg),
                    "[{label}] Auto at omega = 0 must resolve to Pcg"
                );
                assert!(
                    s.rel_residual <= 1e-8,
                    "[{label}] fresh residual {:e}",
                    s.rel_residual
                );
            }
            assert!(r0.alpha_defect <= 100.0 * 1e-10 * scale);
            let mut wide = dyn_opts(DynamicSolver::Auto);
            wide.resonance_window = 2.0;
            sc.linear_response(&props, 0.0, &wide)
                .expect("omega = 0 is EXEMPT from the resonance window");

            for w in [0.1 * w1, 0.5 * w1, 0.9 * w1 * (1.0 - window)] {
                let r = sc
                    .linear_response(&props, w, &dyn_opts(DynamicSolver::Auto))
                    .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: omega = {w} refused {e:?}"));
                let want = closed(w);
                let sw = max_abs(&want, 1.0);
                for (k, &wk) in want.iter().enumerate() {
                    assert!(
                        (r.alpha[k] - wk).abs() <= 1e-8 * sw,
                        "[{label}] phi {phi:?}: alpha({w})[{k}] = {:.12e} vs closed form {wk:.12e}",
                        r.alpha[k]
                    );
                }
                assert!(
                    r.solves
                        .iter()
                        .all(|s| matches!(s.solver_used, DynamicSolver::Pcg))
                );
                assert!(
                    r.alpha_defect <= 100.0 * 1e-10 * sw,
                    "[{label}] alpha_defect {:e} at omega {w}",
                    r.alpha_defect
                );
                let p_a = sc.property_vector(&pa).expect("in-channel");
                let (pair, info) = sc
                    .linear_response_solve(&p_a, w, &dyn_opts(DynamicSolver::Auto))
                    .expect("the covector core solves where the entry point does");
                assert_rel(
                    2.0 * p_a.dot(&pair.u),
                    r.alpha[0],
                    1e-8,
                    &format!("[{label}] 2<p,u> vs alpha_aa({w})"),
                );
                assert!(
                    info.v_weight > 0.0,
                    "[{label}] at omega > 0 the v leg is nonzero"
                );
            }

            let w_above = if omega.len() > 1 {
                0.5 * (w1 + omega[1])
            } else {
                1.5 * w1
            };
            let r = sc
                .linear_response(&props, w_above, &dyn_opts(DynamicSolver::Auto))
                .unwrap_or_else(|e| panic!("[{label}] phi {phi:?}: above-pole Auto refused {e:?}"));
            assert!(
                r.solves
                    .iter()
                    .all(|s| matches!(s.solver_used, DynamicSolver::Minres)),
                "[{label}] Auto above omega_1 must resolve to Minres"
            );
            let want = closed(w_above);
            let sw = max_abs(&want, 1.0);
            for (k, &wk) in want.iter().enumerate() {
                assert!(
                    (r.alpha[k] - wk).abs() <= 1e-8 * sw,
                    "[{label}] phi {phi:?}: alpha({w_above})[{k}] above the pole = {:.12e} vs closed \
                     form {wk:.12e}",
                    r.alpha[k]
                );
            }
            match sc.linear_response(&props, w_above, &dyn_opts(DynamicSolver::Pcg)) {
                Err(SectorResponseError::IndefiniteAbovePole { omega: wo, rho, .. }) => {
                    assert_eq!(wo, w_above);
                    assert!(rho.is_finite() && rho <= 0.0, "rho = {rho}");
                }
                other => panic!(
                    "[{label}] phi {phi:?}: Pcg above the pole must refuse IndefiniteAbovePole (the inertia proof), got {other:?}"
                ),
            }

            assert!(matches!(
                sc.linear_response(&props, -0.1, &dyn_opts(DynamicSolver::Auto)),
                Err(SectorResponseError::NegativeOmega { omega: o }) if o == -0.1
            ));
            let w_near = w1 * (1.0 + 0.5 * window);
            match sc.linear_response(&props, w_near, &dyn_opts(DynamicSolver::Minres)) {
                Err(SectorResponseError::NearResonance {
                    omega: o,
                    omega_n,
                    gap,
                    window: wd,
                }) => {
                    assert_eq!(o, w_near);
                    assert_eq!(omega_n, w1);
                    assert_rel(gap, (w_near - w1).abs(), 1e-12, "the measured gap");
                    assert_eq!(wd, window);
                }
                other => panic!(
                    "[{label}] phi {phi:?}: inside the window must refuse NearResonance even under Minres, got {other:?}"
                ),
            }
        }
    }
    (channels, curved_seen)
}

#[test]
fn tso_dynamic_alpha_exact_grounds() {
    let (channels, curved_seen) = t6_dynamic_alpha(
        vec![
            ("synth_pg (2,1) exact ground", synth_exact_ground()),
            ("d6_pg (3,0) polarized exact ground", d6_polarized_ground()),
        ],
        &|_, _, _| true,
    );
    assert_eq!(
        (channels, curved_seen),
        (4, 1),
        "anti-vacuity: measured inventory of the two exact grounds (one structurally curved)"
    );
}

#[test]
fn tso_dynamic_alpha_minimizer_single_slot() {
    let (channels, curved_seen) =
        t6_dynamic_alpha(vec![("d6_minimizer", d6_minimizer())], &|_, _, slots| {
            slots.len() == 1
        });
    assert_eq!(
        (channels, curved_seen),
        (2, 0),
        "anti-vacuity: the minimizer's two single-slot channels"
    );
}

#[test]
fn tso_dynamic_alpha_minimizer_curved() {
    let (channels, curved_seen) =
        t6_dynamic_alpha(vec![("d6_minimizer", d6_minimizer())], &|phi, curved, _| {
            curved.contains(&phi)
        });
    assert_eq!(
        (channels, curved_seen),
        (1, 1),
        "anti-vacuity: the minimizer's one curved channel"
    );
}

#[test]
fn tso_iterative_moments_publish_a_certified_resonance() {
    let (space, t, v, ham, x) = pg4c_random_reference(0x7505);
    let pg = space.pg;
    let ctx = weingarten_operator(&ham, &x);
    let phi = Irrep::new(1);
    let props = [
        synthetic_property(pg, space.d, phi, 0xF701, "mu_a").expect("pg4c pairs of flux 1"),
        synthetic_property(pg, space.d, phi, 0xF702, "mu_b").expect("pg4c pairs of flux 1"),
    ];
    let ch = Channel::new(space, &t, &v, phi);
    let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
    let dim = sc.dim();
    assert_eq!(
        dim, 6,
        "fixture premise: the curved pg4c channel (measured dim 6)"
    );
    let spec = sc.symplectic_eigenvalues(&sc.basis());
    println!("tso M3: channel verdict {}", spectrum_label(&spec));
    assert!(
        matches!(
            spec,
            qntt::solvers::hess::SymplecticSpectrum::Resonant { .. }
        ),
        "fixture premise: Resonant"
    );
    let dense_opts = SectorMomentOpts {
        tt_grad_tol: f64::INFINITY,
        ..moment_opts()
    };
    let dense = sc
        .transition_moments_dense(&props, &dense_opts)
        .expect("the dense route answers a Resonant channel");
    let iter_opts = SectorMomentOpts {
        tt_grad_tol: f64::INFINITY,
        solver: SectorOpts {
            tol: 1e-10,
            max_iter: 600,
            ..SectorOpts::default()
        },
        ..moment_opts()
    };
    let it = sc
        .transition_moments_iterative(&props, dim, &iter_opts)
        .unwrap_or_else(|e| panic!("the paired iterative route refused {e:?}"));
    print_moments("M3 dense", &dense);
    print_moments("M3 iterative", &it);
    assert_eq!(dense.n_neg, 1, "the dense route: one resonance");
    assert_eq!(it.n_neg, 1, "the iterative route: one CERTIFIED resonance");
    assert_rel(
        it.unstable_omega2[0],
        dense.unstable_omega2[0],
        1e-8,
        "unstable omega^2: iterative vs dense",
    );
    assert!(it.unstable_omega2[0] < 0.0);
    assert_moments_agree(
        &it.omega,
        &it.moments,
        &dense.omega,
        &dense.moments,
        1e-7,
        "M3 iterative vs dense",
    );
    assert!(dense.alpha_sos.is_none() && it.alpha_sos.is_none());
    let r = it
        .readout
        .as_ref()
        .expect("the iterative route carries its readout");
    assert!(
        dense.readout.is_none(),
        "the dense route carries no readout"
    );
    let imaginary: Vec<_> = r
        .modes
        .iter()
        .filter(|m| m.kind == RpaModeKind::Imaginary)
        .collect();
    assert_eq!(imaginary.len(), 1, "one imaginary mode in the readout");
    assert!(
        imaginary[0].certificate.is_some_and(|w| w.certified()),
        "the published resonance carries its certified witness"
    );
    assert_eq!(
        it.metric,
        RpaMetricBlock::Ii,
        "the paired readout's convention"
    );
}

#[test]
fn tso_iterative_moments_report_an_indefinite_channel() {
    let pg = synth_pg();
    let gamma0 = Irrep::new(1);
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, gamma0);
    let (t, v) = synth_tv(pg);
    let (x, _e0, _) = exact_typed_ground(space, &t, &v);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let ctx = weingarten_operator(&ham, &x);
    let phi = Irrep::new(1);
    let pa = synthetic_property(pg, D4, phi, 0xE900, "mu_1").expect("pairs(1) on [A,B,B,A]");
    let props = [pa];
    let ch = Channel::new(space, &t, &v, phi);
    let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
    let basis = sc.basis();
    let n = basis.len();
    assert_eq!(n, sc.dim(), "fixture premise: the basis spans T^phi");
    assert!(
        matches!(
            sc.transition_moments_dense(&props, &moment_opts()),
            Err(SectorResponseError::SymplecticUnstable { .. })
        ),
        "fixture premise: the dense route refuses this channel"
    );
    let (a, b) = sc.grams(&basis);
    assert!(
        frobenius(&b) <= 1e-10 * frobenius(&a),
        "fixture premise: B = 0 at the exact reference"
    );
    let (eig_a, u) = dense_eigen(&flat_to_nested(&a, n));
    let p = sc
        .property_vector(&props[0])
        .expect("an in-channel covector");
    let gamma: Vec<f64> = basis.iter().map(|bi| bi.dot(&p)).collect();
    let pos: Vec<(f64, f64)> = eig_a
        .iter()
        .zip(&u)
        .filter(|(z, _)| **z > 0.0)
        .map(|(z, un)| (*z, dot_vec(un, &gamma).abs()))
        .collect();
    let neg: Vec<f64> = eig_a.iter().copied().filter(|z| *z < 0.0).collect();
    assert!(!neg.is_empty(), "fixture premise: A^phi is indefinite");

    let iter_opts = SectorMomentOpts {
        solver: SectorOpts {
            tol: 1e-10,
            max_iter: 600,
            ..SectorOpts::default()
        },
        ..moment_opts()
    };
    let it = sc
        .transition_moments_iterative(&props, n, &iter_opts)
        .unwrap_or_else(|e| panic!("the paired iterative route must answer, refused {e:?}"));
    print_moments("M4 iterative", &it);
    assert_eq!(it.n_neg, 0, "B = 0: no imaginary mode");
    assert!(
        it.alpha_sos.is_none(),
        "a de-excitation withholds alpha_sos"
    );
    let want_w: Vec<f64> = pos.iter().map(|&(z, _)| z).collect();
    assert_multiset_eq(&it.omega, &want_w, 1e-8, "omega vs eig(A) > 0");
    let cl = clusters_of(&want_w, 1e-6);
    let mut singletons = 0usize;
    for &(lo, hi) in &cl {
        if hi - lo == 1 {
            assert_rel(
                it.moments[0][lo].abs(),
                pos[lo].1,
                1e-7,
                &format!("|T_{lo}| vs |<u_n, p>|"),
            );
            singletons += 1;
        }
    }
    assert!(
        singletons >= 2,
        "anti-vacuity: {singletons} singleton moments pinned"
    );
    let r = it
        .readout
        .as_ref()
        .expect("the iterative route carries its readout");
    let mut de: Vec<f64> = r
        .modes
        .iter()
        .filter(|m| m.kind == RpaModeKind::Real && m.krein < 0.0)
        .map(|m| m.omega.re)
        .collect();
    de.sort_by(f64::total_cmp);
    let mut want_de: Vec<f64> = neg.iter().map(|z| z.abs()).collect();
    want_de.sort_by(f64::total_cmp);
    assert_eq!(
        de.len(),
        want_de.len(),
        "one de-excitation per negative eig(A)"
    );
    for (g, w) in de.iter().zip(&want_de) {
        assert_rel(*g, *w, 1e-8, "de-excitation omega vs |eig(A) < 0|");
    }
    println!("tso M4: de-excitations {de:?} (oracle {want_de:?}), {singletons} singleton moments");
}

#[test]
fn tso_instability_refuses_every_observable_without_publishing() {
    let pg = synth_pg();
    let gamma0 = Irrep::new(1);
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, gamma0);
    let (t, v) = synth_tv(pg);
    let (x, e0, _) = exact_typed_ground(space, &t, &v);
    let lower = dense_levels(D4, &t, &v, pg, F_NUP, F_NDN, Irrep::IDENTITY)[0];
    assert!(
        lower < e0 - 1e-3,
        "fixture premise: irrep 0's ground ({lower}) lies BELOW the reference ({e0})"
    );
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let ctx = weingarten_operator(&ham, &x);
    let phi = Irrep::new(1);
    let pa = synthetic_property(pg, D4, phi, 0xE900, "mu_1").expect("pairs(1) on [A,B,B,A]");
    let props = [pa.clone()];
    let ch = Channel::new(space, &t, &v, phi);
    let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
    assert!(sc.dim() > 0);
    let spec = sc.symplectic_eigenvalues(&sc.basis());
    println!(
        "tso T7: channel verdict {} (E_0(lower) - E_0 = {})",
        spectrum_label(&spec),
        lower - e0
    );
    assert!(
        matches!(
            spec,
            qntt::solvers::hess::SymplecticSpectrum::Unstable { .. }
        ),
        "fixture premise: Unstable"
    );

    let p = sc
        .property_vector(&pa)
        .expect("the covector needs no Hessian and no stability");
    assert!(p.norm() > 1e-6, "anti-vacuity ||p|| = {:e}", p.norm());

    match sc.polarizability_static(&props, &SectorStaticOpts::default()) {
        Err(SectorResponseError::HessianIndefinite { label, rho, iters }) => {
            assert_eq!(label, "mu_1");
            assert!(rho.is_finite() && rho <= 0.0, "rho = {rho}");
            println!("  E2 refused HessianIndefinite: rho {rho:e} after {iters} iterations");
        }
        other => panic!("E2 must refuse HessianIndefinite on an indefinite channel, got {other:?}"),
    }
    let dets_lower = dets_in_irrep(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY).len();
    assert_eq!(
        sc.dim(),
        dets_lower,
        "fixture premise: T^phi IS the whole irrep-0 sector (measured 12 == 12)"
    );
    match sc.transition_moments_dense(&props, &moment_opts()) {
        Err(SectorResponseError::SymplecticUnstable { reason }) => {
            println!("  E3 refused SymplecticUnstable: {reason:?}");
            let s = format!("{reason:?}");
            assert!(!s.contains("NaN"), "no NaN in the refusal: {s}");
            match reason {
                WilliamsonError::HrrNotPd { min_eig } => assert_rel(
                    min_eig,
                    lower - e0,
                    1e-10,
                    "HrrNotPd.min_eig vs E_0(irrep 0) − E_0",
                ),
                other => panic!("the both-indefinite refusal is HrrNotPd, got {other:?}"),
            }
        }
        other => panic!(
            "E3 must refuse SymplecticUnstable when BOTH blocks are indefinite, got {other:?}"
        ),
    }
    let dyn_opts = |solver: DynamicSolver| SectorDynamicOpts {
        solver,
        route: PropRoute::Materialized,
        ..SectorDynamicOpts::default()
    };
    assert!(
        matches!(
            sc.linear_response(&props, 0.0, &dyn_opts(DynamicSolver::Pcg)),
            Err(SectorResponseError::HessianIndefinite { .. })
        ),
        "E4 at omega = 0 IS the static system: HessianIndefinite"
    );
    match sc.linear_response(&props, 0.3, &dyn_opts(DynamicSolver::Pcg)) {
        Err(SectorResponseError::IndefiniteAbovePole { omega, rho, .. }) => {
            assert_eq!(omega, 0.3);
            assert!(rho.is_finite());
        }
        other => panic!(
            "E4 Pcg at omega > 0 maps negative curvature to IndefiniteAbovePole (the ω-split), got {other:?}"
        ),
    }
}

#[test]
fn tso_threaded_matches_serial() {
    let (space, t, v, ham, x) = d6_minimizer();
    let pg = space.pg;
    let (bl, _br) = tangent_bases(&x);
    let phi = first_curved_channel(&space, &bl.ranks);
    let p = phi.index() as u64;
    let props = [
        synthetic_property(pg, D6, phi, 0xEA00 + p, "mu_a").expect("curved d6 channel has pairs"),
        synthetic_property(pg, D6, phi, 0xEB00 + p, "mu_b").expect("curved d6 channel has pairs"),
    ];
    let ctx = weingarten_operator(&ham, &x);
    let ch = Channel::new(space, &t, &v, phi);
    let run = || {
        let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
        let (a, b) = sc.grams(&sc.basis());
        let st = sc
            .polarizability_static(&props, &tight_static())
            .expect("static");
        let tm = sc
            .transition_moments_dense(&props, &moment_opts())
            .expect("moments");
        (frobenius(&b) / frobenius(&a), st, tm)
    };
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let (ratio_s, st_s, tm_s) = serial_pool.install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            1,
            "serial arm did not run in the 1-thread pool"
        );
        run()
    });
    let par_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("8-thread pool");
    let (ratio_p, st_p, tm_p) = par_pool.install(|| {
        assert!(
            rayon::current_num_threads() > 1,
            "parallel arm did not run multi-threaded"
        );
        run()
    });
    println!(
        "tso T8: phi {phi:?} dim {} ||B||_F/||A||_F {ratio_s:e} / {ratio_p:e}",
        tm_s.dim
    );
    assert!(
        ratio_s > 1e-3,
        "ANTI-VACUITY: B must be nonzero on the threading channel"
    );
    assert!(tm_s.dim > 1);
    let scale = max_abs(&st_s.alpha, 1.0);
    assert!(scale > 1e-3, "anti-vacuity: alpha is identically zero");
    for (k, (a, b)) in st_s.alpha.iter().zip(&st_p.alpha).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * scale,
            "threaded vs serial alpha[{k}]: {a} vs {b}"
        );
    }
    assert_eq!(tm_s.n_neg, tm_p.n_neg);
    assert_moments_agree(
        &tm_s.omega,
        &tm_s.moments,
        &tm_p.omega,
        &tm_p.moments,
        1e-10,
        "threaded vs serial moments",
    );
    for (k, (a, b)) in tm_s.oscillator.iter().zip(&tm_p.oscillator).enumerate() {
        assert!(
            (a - b).abs() <= 1e-10 * a.abs().max(1.0),
            "threaded vs serial f[{k}]: {a} vs {b}"
        );
    }
    for (k, (a, b)) in tm_s.sum_rule.iter().zip(&tm_p.sum_rule).enumerate() {
        assert!(
            (a.0 - b.0).abs() <= 1e-10 * a.0.abs().max(1.0)
                && (a.1 - b.1).abs() <= 1e-10 * a.1.abs().max(1.0),
            "threaded vs serial sum_rule[{k}]: {a:?} vs {b:?}"
        );
    }
    let sos_s = tm_s.alpha_sos.as_ref().expect("Stable");
    let sos_p = tm_p.alpha_sos.as_ref().expect("Stable");
    for (k, (a, b)) in sos_s.iter().zip(sos_p).enumerate() {
        assert!(
            (a - b).abs() <= 1e-10 * scale,
            "threaded vs serial alpha_sos[{k}]: {a} vs {b}"
        );
    }
}

#[test]
fn tso_driver_composes_per_channel_observables() {
    let (space, t, v, ham, x) = d6_minimizer();
    let pg = space.pg;
    let (bl, _br) = tangent_bases(&x);
    let live: Vec<Irrep> = live_channels(&space, &bl.ranks)
        .into_iter()
        .filter(|&p| p != Irrep::IDENTITY)
        .collect();
    let props: Vec<PropertyOperator<f64>> = live
        .iter()
        .map(|&phi| {
            synthetic_property(
                pg,
                D6,
                phi,
                0xEC00 + phi.index() as u64,
                &format!("mu_{}", phi.index()),
            )
            .expect("d6 live channels have pairs")
        })
        .collect();
    assert_eq!(
        props.len(),
        3,
        "measured d6 inventory: three live shifted channels"
    );
    let n = props.len();
    let opts = SectorAllOpts {
        readout: SectorAllReadout::Rpa,
        k: 2,
        dense_cap: Some(64),
        properties: Some(props.clone()),
        observables: SectorObservables::Both,
        ..SectorAllOpts::default()
    };
    let all = sector_response_all(&ham, &x, &t, &v, None, &opts);
    assert!(
        all.skipped.is_empty(),
        "every component has a live single flux: skipped = {:?}",
        all.skipped
    );
    let ctx = weingarten_operator(&ham, &x);
    let mut alpha_sum = vec![0.0f64; n * n];
    let mut f_sum = 0.0f64;
    for (i, ch) in all.channels.iter().enumerate() {
        let phi = ch.phi;
        assert_eq!(
            ch.status,
            SectorChannelStatus::Ok,
            "row {i} ({phi:?}): {:?}",
            ch.status
        );
        let alpha = ch
            .alpha
            .as_ref()
            .unwrap_or_else(|| panic!("row {i}: alpha requested but absent"));
        let moments = ch
            .moments
            .as_ref()
            .unwrap_or_else(|| panic!("row {i}: moments requested but absent"));
        let own = live
            .iter()
            .position(|&p| p == phi)
            .expect("row flux is live");
        assert_eq!(alpha.alpha.len(), n * n);
        for r in 0..n {
            for c in 0..n {
                if r != own || c != own {
                    assert_eq!(
                        alpha.alpha[r * n + c],
                        0.0,
                        "row {i}: out-of-channel alpha[{r},{c}] must be EXACTLY 0"
                    );
                }
                alpha_sum[r * n + c] += alpha.alpha[r * n + c];
            }
        }
        assert!(
            alpha.alpha[own * n + own] > 1e-4,
            "row {i}: anti-vacuity alpha_own = {:e}",
            alpha.alpha[own * n + own]
        );
        for (r, row) in moments.moments.iter().enumerate() {
            if r != own {
                assert!(
                    row.iter().all(|&z| z == 0.0),
                    "row {i}: out-of-channel moment row {r} must be exactly zero"
                );
            }
        }
        f_sum += moments.oscillator.iter().sum::<f64>();
        let chan = Channel::new(space, &t, &v, phi);
        let sc = SectorCtx::new(&ctx, &chan.h_out, phi);
        let st = sc
            .polarizability_static(&props, &SectorStaticOpts::default())
            .expect("direct static");
        let tm = sc
            .transition_moments_dense(
                &props,
                &SectorMomentOpts {
                    solver: opts.solver.clone(),
                    route: PropRoute::Materialized,
                    ..SectorMomentOpts::default()
                },
            )
            .expect("direct moments");
        let bits = st
            .alpha
            .iter()
            .zip(&alpha.alpha)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && tm
                .omega
                .iter()
                .zip(&moments.omega)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "tso T9 row {i} ({phi:?}): dim {} SYNTHETIC alpha_own {:.9e} bit-identical to the direct call = {bits}",
            ch.dim,
            alpha.alpha[own * n + own]
        );
        let scale = max_abs(&st.alpha, 1.0);
        for k in 0..n * n {
            assert!(
                (st.alpha[k] - alpha.alpha[k]).abs() <= 1e-12 * scale,
                "row {i}: alpha[{k}] driver {} vs direct {}",
                alpha.alpha[k],
                st.alpha[k]
            );
        }
        assert_eq!(tm.dim, moments.dim);
        assert_eq!(tm.n_neg, moments.n_neg);
        assert_moments_agree(
            &moments.omega,
            &moments.moments,
            &tm.omega,
            &tm.moments,
            1e-12,
            &format!("row {i}: driver vs direct moments"),
        );
        for (k, (a, b)) in moments.oscillator.iter().zip(&tm.oscillator).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * a.abs().max(1.0),
                "row {i}: f[{k}] {a} vs {b}"
            );
        }
        assert_eq!(
            moments.labels,
            props.iter().map(|p| p.label.clone()).collect::<Vec<_>>()
        );
    }
    let total = all
        .alpha_total
        .as_ref()
        .expect("alpha requested ⇒ alpha_total present");
    assert_eq!(total.len(), n * n);
    for k in 0..n * n {
        assert_rel(
            total[k],
            alpha_sum[k],
            1e-14,
            &format!("alpha_total[{k}] vs Σ_channels alpha"),
        );
    }
    for r in 0..n {
        for c in 0..n {
            if r != c {
                assert_eq!(
                    total[r * n + c],
                    0.0,
                    "one component per channel ⇒ alpha_total is DIAGONAL"
                );
            }
        }
    }
    let f_total = all
        .oscillator_total
        .expect("moments requested ⇒ oscillator_total present");
    assert_rel(
        f_total,
        f_sum,
        1e-14,
        "oscillator_total vs Σ_channels Σ_n f_n",
    );
    println!(
        "tso T9: SYNTHETIC alpha_total diag {:?}, Σ f = {f_total:.6e} (not TRK: synthetic operator, incomplete basis, in-sector channel excluded)",
        (0..n).map(|r| total[r * n + r]).collect::<Vec<_>>()
    );
}

#[test]
fn tso_driver_skips_by_name_and_empty_or_refused_rows_carry_no_observables() {
    let (space, t, v, ham, x) = d6_minimizer();
    let pg = space.pg;
    let (bl, _br) = tangent_bases(&x);
    let live: Vec<Irrep> = live_channels(&space, &bl.ranks)
        .into_iter()
        .filter(|&p| p != Irrep::IDENTITY)
        .collect();
    let phi0 = live[0];
    let props = vec![
        synthetic_property(pg, D6, phi0, 0xED00, "mu_live").expect("pairs"),
        identity_property(pg, D6, 0xED01, "mu_A"),
        multiclass_property(pg, D6, "mu_multi"),
    ];
    let base = SectorAllOpts {
        readout: SectorAllReadout::Rpa,
        k: 2,
        dense_cap: Some(64),
        ..SectorAllOpts::default()
    };
    let with = sector_response_all(
        &ham,
        &x,
        &t,
        &v,
        None,
        &SectorAllOpts {
            properties: Some(props.clone()),
            observables: SectorObservables::Alpha,
            ..base.clone()
        },
    );
    assert_eq!(
        with.skipped,
        vec!["mu_A".to_string(), "mu_multi".to_string()],
        "skipped names the IDENTITY-flux and the multi-class component, in the caller's order"
    );
    let total = with.alpha_total.as_ref().expect("Alpha requested");
    assert_eq!(total.len(), 9);
    for k in 0..3 {
        for s in [1usize, 2] {
            assert_eq!(total[k * 3 + s], 0.0);
            assert_eq!(total[s * 3 + k], 0.0);
        }
    }
    assert!(
        total[0] > 1e-4,
        "the live component's alpha must be present: {:e}",
        total[0]
    );
    assert!(
        with.oscillator_total.is_none(),
        "Alpha only ⇒ no oscillator total"
    );
    for ch in &with.channels {
        assert!(ch.moments.is_none(), "Alpha only ⇒ no moments row");
        let a = ch
            .alpha
            .as_ref()
            .expect("live channel with Alpha requested");
        for s in [1usize, 2] {
            assert_eq!(
                a.solves[s].grad_norm, 0.0,
                "a skipped component is computed NOWHERE"
            );
            assert_eq!(a.solves[s].iters, 0);
        }
        if ch.phi != phi0 {
            assert!(
                a.alpha.iter().all(|&z| z == 0.0),
                "phi {:?}: no component of its flux ⇒ all-zero block",
                ch.phi
            );
        }
    }
    let none = sector_response_all(&ham, &x, &t, &v, None, &base);
    assert!(
        none.alpha_total.is_none() && none.oscillator_total.is_none() && none.skipped.is_empty()
    );
    assert!(
        none.channels
            .iter()
            .all(|c| c.alpha.is_none() && c.moments.is_none())
    );
    assert_eq!(none.channels.len(), with.channels.len());
    for (a, b) in none.channels.iter().zip(&with.channels) {
        assert_eq!(
            (a.phi, a.dim, a.status.clone()),
            (b.phi, b.dim, b.status.clone())
        );
        let (ra, rb) = (a.rpa.as_ref().unwrap(), b.rpa.as_ref().unwrap());
        assert_eq!(
            ra.omega2()
                .unwrap()
                .iter()
                .map(|z| z.to_bits())
                .collect::<Vec<_>>(),
            rb.omega2()
                .unwrap()
                .iter()
                .map(|z| z.to_bits())
                .collect::<Vec<_>>(),
            "the spectral rows must not change when observables are added"
        );
    }

    let (space4, t4, v4, ham4, x4) = truncated_reference();
    let pg4 = space4.pg;
    let (bl4, _) = tangent_bases(&x4);
    let props4: Vec<PropertyOperator<f64>> = (1..4u8)
        .filter_map(|p| {
            synthetic_property(
                pg4,
                D4,
                Irrep::new(p),
                0xEE00 + p as u64,
                &format!("mu_{p}"),
            )
        })
        .collect();
    assert_eq!(
        props4.len(),
        3,
        "[A,B,C,B] has pairs of every non-identity flux"
    );
    for kernel in [RpaKernel::signed(RpaMetric::Auto), RpaKernel::Paired] {
        let signed = !matches!(kernel, RpaKernel::Paired);
        let out4 = sector_response_all(
            &ham4,
            &x4,
            &t4,
            &v4,
            None,
            &SectorAllOpts {
                properties: Some(props4.clone()),
                observables: SectorObservables::Both,
                dense_cap: Some(64),
                solver: SectorOpts {
                    kernel,
                    ..base.solver.clone()
                },
                ..base.clone()
            },
        );
        let n_live = out4.channels.iter().filter(|c| c.dim > 0).count();
        assert_eq!(out4.built, n_live, "an Empty row builds no operator");
        let mut empties = 0usize;
        let mut refused = 0usize;
        for ch in &out4.channels {
            println!(
                "tso T9b fx4 row phi {:?}: dim {} status {:?} alpha {} moments {}",
                ch.phi,
                ch.dim,
                ch.status,
                ch.alpha.is_some(),
                ch.moments.is_some()
            );
            if ch.dim == 0 {
                assert_eq!(ch.status, SectorChannelStatus::Empty);
                assert!(
                    ch.alpha.is_none() && ch.moments.is_none(),
                    "an Empty row carries no observable"
                );
                assert_eq!(tangent_dim_grids_of(&space4, &bl4.ranks, ch.phi), 0);
                empties += 1;
            } else {
                assert_ne!(
                    ch.status,
                    SectorChannelStatus::Ok,
                    "phi {:?}: a live channel at a non-critical trimmed point cannot be Ok",
                    ch.phi
                );
                assert_eq!(
                    ch.status,
                    SectorChannelStatus::Indefinite {
                        n_negative: 0,
                        certified: true,
                        n_negative_signature: 2,
                        metric_indefinite: signed,
                    },
                    "phi {:?} ({kernel:?}): the flagged signed / anomalous paired verdict",
                    ch.phi
                );
                assert!(
                    ch.alpha.is_none() && ch.moments.is_none(),
                    "phi {:?}: a refused observable is None, recorded in status {:?}",
                    ch.phi,
                    ch.status
                );
                refused += 1;
            }
        }
        assert!(
            empties == 2 && refused == 1,
            "measured fx4 (2,2) inventory: two Empty rows and one live (refused) row, got {empties}/{refused}"
        );
        assert!(
            out4.alpha_total.is_none()
                || out4
                    .alpha_total
                    .as_ref()
                    .is_some_and(|a| a.iter().all(|&z| z == 0.0)),
            "no channel produced an alpha"
        );
    }
}

#[test]
fn tso_driver_reports_a_refused_observable_on_a_stable_channel() {
    let (space, t, v, ham, x) = d6_polarized_ground();
    let pg = space.pg;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xF9C0);
    let mut xp = perturbation(&x, &x.ranks, 1e-3, &mut rng);
    xp.round_ranks_mut(&x.ranks);
    normalize_mut(&mut xp);
    let lambda = rayleigh_quotient(&ham, &xp).re_s();
    let g = grad_norm_at(&ham, &xp);
    println!("tso T9c: nudged reference lambda {lambda} ||P_T grad|| {g:e}");
    assert!(
        g > 1e-6 * lambda.abs().max(1.0),
        "premise: the nudged reference must FAIL the default criticality gate, ||grad|| = {g:e}"
    );
    let (bl, _) = tangent_bases(&xp);
    let props: Vec<PropertyOperator<f64>> = live_channels(&space, &bl.ranks)
        .into_iter()
        .filter(|&p| p != Irrep::IDENTITY)
        .filter_map(|phi| {
            synthetic_property(
                pg,
                D6,
                phi,
                0xF500 + phi.index() as u64,
                &format!("mu_{}", phi.index()),
            )
        })
        .take(3)
        .collect();
    assert!(!props.is_empty(), "anti-vacuity: no synthetic component");
    let all = sector_response_all(
        &ham,
        &xp,
        &t,
        &v,
        None,
        &SectorAllOpts {
            readout: SectorAllReadout::Rpa,
            k: 1,
            dense_cap: Some(64),
            properties: Some(props),
            observables: SectorObservables::Both,
            ..SectorAllOpts::default()
        },
    );
    let mut refused = 0usize;
    for ch in &all.channels {
        if ch.dim == 0 {
            continue;
        }
        let rpa = ch.rpa.as_ref().expect("Rpa requested");
        println!(
            "  T9c phi {:?}: dim {} omega2 {:?} converged {} dense {:?} status {:?}",
            ch.phi,
            ch.dim,
            rpa.omega2(),
            rpa.converged,
            ch.dense.as_ref().map(spectrum_label),
            ch.status
        );
        assert!(
            rpa.converged
                && rpa
                    .omega2()
                    .is_some_and(|w| !w.is_empty() && w.iter().all(|&z| z > 0.0)),
            "premise: the RPA readout is converged and all-positive at a 1e-3 nudge"
        );
        assert!(
            matches!(ch.dense, Some(SymplecticSpectrum::Stable { .. })),
            "premise: the dense verdict is Stable"
        );
        match &ch.status {
            SectorChannelStatus::ObservableRefused { what, why } => {
                assert_eq!(*what, "alpha", "alpha is refused first and wins the slot");
                assert!(
                    why.contains("not TT-critical"),
                    "the reason names the gate: {why}"
                );
            }
            other => panic!(
                "phi {:?}: expected ObservableRefused, got {other:?}",
                ch.phi
            ),
        }
        assert!(
            ch.alpha.is_none() && ch.moments.is_none(),
            "a refused observable publishes nothing"
        );
        refused += 1;
    }
    assert!(refused >= 1, "anti-vacuity: {refused} refused rows");
    let total = all.alpha_total.as_ref().expect("Alpha requested");
    assert!(
        total.iter().all(|&z| z == 0.0),
        "no channel produced an alpha: {total:?}"
    );
    assert_eq!(all.oscillator_total, Some(0.0));
}

fn tangent_dim_grids_of(space: &QnSpace, ranks: &[qntt::RankGrid], phi: Irrep) -> usize {
    qntt::core::tangent::tangent_dim_grids(space, ranks, phi)
}

#[test]
fn tso_field_fd_untyped_embedding_catches_a_shared_sign_error() {
    let (space, t, v, ham, x) = synth_exact_ground();
    let pg = space.pg;
    let d = space.d;
    let (bl, _br) = tangent_bases(&x);
    let phi = live_channels(&space, &bl.ranks)
        .into_iter()
        .find(|&p| p != Irrep::IDENTITY)
        .expect("live shifted channel");
    let pa = synthetic_property(pg, d, phi, 0xEF00, "mu_fd").expect("pairs");
    let ctx = weingarten_operator(&ham, &x);
    let ch = Channel::new(space, &t, &v, phi);
    let sc = SectorCtx::new(&ctx, &ch.h_out, phi);
    let st = sc
        .polarizability_static(std::slice::from_ref(&pa), &tight_static())
        .expect("static on the stable channel");
    let alpha_channel = st.alpha[0];
    assert!(
        alpha_channel > 1e-3,
        "anti-vacuity: SYNTHETIC alpha^phi = {alpha_channel:e}"
    );

    let (space_c1, h_c1, x_c1) = untyped_embedding(space, &t, &v, &x);
    let e0 = rayleigh_quotient(&h_c1, &x_c1).re_s();
    let mut ropts = RiemannianOpts::new(&space_c1);
    ropts.eps = 1e-13;
    ropts.max_iter = 600;
    ropts.gradtol = 1e-13;
    let energy_at = |f: f64| -> (f64, usize) {
        let mut tf = t.clone();
        for p in 0..d {
            for q in 0..d {
                tf.set(&[p, q], t.get(&[p, q]) - f * pa.m.get(&[p, q]));
            }
        }
        let hf = SparseHamiltonian::<f64>::new(space_c1, &tf, &v, None, f64::EPSILON);
        let mut y = x_c1.clone();
        let mut rounds = 0usize;
        loop {
            let (_l, y2) = riemannian_cg(&hf, &y, &ropts);
            y = y2;
            normalize_mut(&mut y);
            rounds += 1;
            let g = grad_norm_at(&hf, &y);
            if g < 1e-11 || rounds >= 20 {
                assert!(
                    g < 1e-9,
                    "field {f:e}: the C1 re-minimization did not converge (||grad|| = {g:e} after {rounds} rounds)"
                );
                break;
            }
        }
        (rayleigh_quotient(&hf, &y).re_s(), rounds)
    };
    let f1 = 2e-3f64;
    let (e_p1, r1) = energy_at(f1);
    let (e_m1, r2) = energy_at(-f1);
    let (e_p2, r3) = energy_at(2.0 * f1);
    let (e_m2, r4) = energy_at(-2.0 * f1);
    println!("tso T4: CG rounds at ±f, ±2f = {r1}/{r2}/{r3}/{r4}; lambda(0) = {e0}");
    for (f, e) in [(f1, e_p1), (-f1, e_m1), (2.0 * f1, e_p2), (-2.0 * f1, e_m2)] {
        let shift = (e - e0).abs();
        println!(
            "  branch continuity: |lambda({f:e}) - lambda(0)| = {shift:e} (O(f^2) budget {:e})",
            10.0 * alpha_channel.max(1.0) * f * f
        );
        assert!(
            shift <= 10.0 * alpha_channel.max(1.0) * f * f + 1e-9,
            "the re-minimization left the branch at f = {f:e}: shift {shift:e}"
        );
    }
    let alpha_3pt_f1 = -(e_p1 - 2.0 * e0 + e_m1) / (f1 * f1);
    let alpha_3pt_2f1 = -(e_p2 - 2.0 * e0 + e_m2) / (4.0 * f1 * f1);
    let alpha_5pt = -(-e_p2 + 16.0 * e_p1 - 30.0 * e0 + 16.0 * e_m1 - e_m2) / (12.0 * f1 * f1);
    let alpha_richardson = (4.0 * alpha_3pt_f1 - alpha_3pt_2f1) / 3.0;
    println!(
        "tso T4: SYNTHETIC alpha_FD 3pt(f) {alpha_3pt_f1:e} 3pt(2f) {alpha_3pt_2f1:e} 5pt {alpha_5pt:e} \
         richardson {alpha_richardson:e} vs channel alpha^phi {alpha_channel:e}"
    );
    assert_rel(
        alpha_5pt,
        alpha_richardson,
        1e-10,
        "Richardson consistency is pure algebra",
    );
    assert!(
        (alpha_5pt - alpha_channel).abs() <= 1e-4 * alpha_channel.abs().max(1.0),
        "field FD on the untyped embedding ({alpha_5pt:e}) disagrees with the channel alpha ({alpha_channel:e}) — \
         a SIGN or factor error shared by the channel path and its algebraic oracle"
    );
    assert!(
        alpha_5pt > 0.0,
        "a stable reference has a POSITIVE polarizability"
    );
}
