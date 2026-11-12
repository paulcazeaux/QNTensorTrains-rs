// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::ImplicitHZ;
use crate::core::border_fixture::{
    BOND, CLUSTER, CLUSTER_BONDS, CLUSTER_FLOOR, ClusterCensus, assert_cluster_premise, base_state,
    cluster_base_state, pg_base_state, sigma_min, unit_tangents, with_schmidt_cluster,
    with_small_schmidt,
};
use crate::core::deflate::ColumnDeflation;
use crate::core::deflate::oracle::{
    asym, congruence, dense_projector, max_abs, max_abs_diff, product_spectrum, radius,
};
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::tangent::{TtTangent, tangent_basis, tangent_dim_grids};
use crate::core::weingarten::right_chain_oracle::weingarten_right_chain_oracle;
use crate::core::weingarten::{
    BondGauge, ExplicitZ, FloorPolicy, Terms, bond_gauges, weingarten_assemble_sources,
};
use crate::hamiltonian::SparseHamiltonian;
use crate::pointgroup::Irrep;
use crate::solvers::hess::{
    CurvatureReg, HessPart, ModeCheckOpts, PdVerdict, RpaKernel, RpaMetric, RpaReport, SectorCtx,
    SymplecticDavidsonOpts, SymplecticSpectrum, WeingartenCtx, hessian_grams, hessian_grams_owned,
    hessian_pd_probe, rpa_mode_checks, select_rpa_metric, symplectic_eigenvalues,
    symplectic_from_grams, symplectic_lowest_signed_davidson, weingarten_operator,
};
use crate::solvers::riemannian::tangent_bases;
use crate::test_util::flat_bits;
use crate::test_util::tangent_bits;
use crate::test_util::{
    generic_truncated_h, pg_symmetric_tv, pg4_curved_state, relabel_sensitive_floor,
};
use crate::vector::{TtVector, tt_randn};

fn fixture_floor(x0: &TtVector<f64>) -> f64 {
    (sigma_min(x0) * 1e-4).sqrt()
}

fn border_point(eps: f64) -> (TtVector<f64>, SparseHamiltonian<f64>, f64) {
    let x0 = base_state();
    let f = fixture_floor(&x0);
    let x = with_small_schmidt(&x0, BOND, eps);
    let h = generic_truncated_h::<f64>(x.space);
    (x, h, f)
}

fn tik_weight(e: f64) -> impl Fn(f64) -> f64 {
    move |s| s * s / (s * s + e * e)
}

fn deflated(p: &ColumnDeflation<f64>) -> usize {
    p.bonds().iter().map(|r| r.deflated).sum()
}

fn seeds(
    ctx: &WeingartenCtx<'_, f64>,
    x: &TtVector<f64>,
    n: usize,
    seed: u64,
) -> Vec<TtTangent<f64>> {
    let (bl, br) = ctx.bases();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n)
        .map(|_| TtTangent::project(bl, br, &tt_randn(x.space, &x.ranks, &mut rng)))
        .collect()
}

fn assert_bitwise(got: &TtTangent<f64>, want: &TtTangent<f64>, what: &str) {
    assert!(
        tangent_bits(got) == tangent_bits(want),
        "{what}: not bitwise equal (‖Δ‖ = {:e})",
        got.sub(want).norm()
    );
}

fn applied(d: &ColumnDeflation<f64>, xi: &TtTangent<f64>) -> TtTangent<f64> {
    let mut v = xi.clone();
    d.apply(&mut v);
    v
}

fn congruent_b(
    ctx: &WeingartenCtx<'_, f64>,
    d: &ColumnDeflation<f64>,
    xi: &TtTangent<f64>,
) -> TtTangent<f64> {
    let mut b = ctx.b_apply(&applied(d, xi));
    d.apply(&mut b);
    b
}

pub(crate) fn williamson_vs_product(
    spec: &SymplecticSpectrum,
    real: &[f64],
    cplx: &[(f64, f64)],
    scale: f64,
    label: &str,
) -> (&'static str, f64) {
    let tol = 1e-8 * scale;
    if let Some(w2) = spec.omega_squared() {
        assert!(
            cplx.is_empty(),
            "{label}: a real Williamson arm with a complex product spectrum"
        );
        assert_eq!(
            w2.len(),
            real.len(),
            "{label}: a real Williamson spectrum is the whole spectrum"
        );
        let gap = w2
            .iter()
            .zip(real)
            .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
        assert!(
            gap <= tol,
            "{label}: Williamson ω² vs the dense product off by {gap:e}"
        );
        let arm = match spec {
            SymplecticSpectrum::Stable { .. } => "Stable",
            _ => "Resonant",
        };
        return (arm, gap);
    }
    let SymplecticSpectrum::Unstable { generator_eigs, .. } = spec else {
        unreachable!("omega_squared is None only on Unstable")
    };
    let prod: Vec<(f64, f64)> = real
        .iter()
        .map(|&r| (r, 0.0))
        .chain(cplx.iter().flat_map(|&(u, v)| [(u, v), (u, -v)]))
        .collect();
    let sq: Vec<(f64, f64)> = generator_eigs
        .iter()
        .map(|z| (z.re * z.re - z.im * z.im, 2.0 * z.re * z.im))
        .collect();
    assert_eq!(
        sq.len(),
        2 * prod.len(),
        "{label}: 2n generator eigenvalues for n product values"
    );
    let near = |p: (f64, f64), set: &[(f64, f64)]| {
        set.iter()
            .map(|q| (p.0 - q.0).hypot(p.1 - q.1))
            .fold(f64::INFINITY, f64::min)
    };
    let gap = prod
        .iter()
        .map(|&p| near(p, &sq))
        .chain(sq.iter().map(|&q| near(q, &prod)))
        .fold(0.0f64, f64::max);
    assert!(
        gap <= tol,
        "{label}: generator eigenvalues² vs the dense product off by {gap:e}"
    );
    ("Unstable", gap)
}

fn gram_arm(
    reg: CurvatureReg,
    d_of: impl Fn(&WeingartenCtx<'_, f64>) -> ColumnDeflation<f64>,
    label: &str,
) {
    let (x, h, f) = border_point(1e-4);
    let ctx_off = weingarten_operator(&h, &x);
    let d = d_of(&ctx_off);
    let hard = ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
    assert_eq!(
        deflated(&hard),
        1,
        "fixture: exactly the planted direction is below the floor"
    );
    let ctx = weingarten_operator(&h, &x).with_curvature(reg);
    assert_eq!(ctx.curvature(), reg);
    let (bl, br) = ctx_off.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a_off, b_off) = hessian_grams(&ctx_off, &basis);
    let (a_reg, b_reg) = hessian_grams(&ctx, &basis);
    let dd = dense_projector(&d, &basis);
    let want = congruence(&dd, &b_off, n);
    let (raw, regm) = (max_abs(&b_off), max_abs(&b_reg));
    let err = max_abs_diff(&b_reg, &want);
    let sym = asym(&b_reg, n);
    println!(
        "{label}: n = {n}, max|B| raw {raw:.3e} regularized {regm:.3e}, |Gram − D·B·D| {err:.2e}, \
         symmetry defect {:.2e} (relative)",
        sym / regm
    );
    assert!(
        err <= 1e-12 * raw.max(1.0),
        "{label}: the regularized B Gram is not D·B·D ({err:e} against max|B| {raw:e})"
    );
    let sym_bound = match reg {
        CurvatureReg::Tikhonov(_) => 1e-14 * regm + 1e-15 * raw,
        _ => 1e-14 * regm + 1e-16 * raw,
    };
    println!("{label}: symmetry defect {sym:.3e} (absolute) vs bound {sym_bound:.3e}");
    assert!(
        sym <= sym_bound,
        "{label}: symmetry defect {sym:e} > {sym_bound:e}"
    );
    assert!(
        flat_bits(&a_reg) == flat_bits(&a_off),
        "{label}: the A Gram must be bitwise the plain context's (A is untouched)"
    );
    assert!(
        raw >= 1e2 * regm,
        "{label}: max|B| raw {raw:e} vs regularized {regm:e}"
    );
    let (_, b_owned) = hessian_grams_owned(&ctx, basis.clone());
    assert!(
        flat_bits(&b_owned) == flat_bits(&b_reg),
        "{label}: hessian_grams_owned"
    );
    let ctx_inner = match reg {
        CurvatureReg::Hard(fh) => weingarten_operator(&h, &x).with_floor(FloorPolicy::Hard(fh)),
        _ => weingarten_operator(&h, &x),
    };
    for xi in seeds(&ctx_off, &x, 3, 0xC0B1) {
        let bt = match reg {
            CurvatureReg::Tikhonov(e) => tik_composition(&ctx_off, e, &xi),
            _ => congruent_b(&ctx_inner, &d, &xi),
        };
        let b_plain = congruent_b(&ctx_off, &d, &xi);
        let dev = bt.sub(&b_plain).norm();
        assert!(
            dev <= 1e-12 * b_plain.norm().max(1.0),
            "{label}: D·B_inner·D vs D·B_plain·D {dev:e} (the same operator)"
        );
        let a = ctx_off.a_apply(&xi);
        assert_bitwise(
            &ctx.b_apply(&xi),
            &bt,
            &format!("{label}: b_apply = D·B_inner·D (Hard) / the P4d composition (Tikhonov)"),
        );
        assert_bitwise(
            &ctx.hpp_apply(&xi),
            &a.add(&bt),
            &format!("{label}: hpp = A + D·B·D"),
        );
        assert_bitwise(
            &ctx.hmm_apply(&xi),
            &a.sub(&bt),
            &format!("{label}: hmm = A − D·B·D"),
        );
        assert_bitwise(
            &ctx.hess_apply(&xi),
            &a.add(&bt),
            &format!("{label}: hess = A + D·B·D"),
        );
    }
}

#[test]
fn cr_hard_b_gram_is_pd_b_pd() {
    let f = fixture_floor(&base_state());
    gram_arm(
        CurvatureReg::Hard(f),
        |ctx| ColumnDeflation::from_gauges(ctx.bond_gauges(), f),
        "T2 Hard",
    );
}

#[test]
fn cr_tikhonov_b_gram_is_dd_b_dd() {
    let f = fixture_floor(&base_state());
    gram_arm(
        CurvatureReg::Tikhonov(f),
        |ctx| ColumnDeflation::from_gauges_weighted(ctx.bond_gauges(), tik_weight(f)),
        "T3 Tikhonov",
    );
}

#[test]
fn cr_kept_block_identity_against_the_in_formula_hard_floor() {
    let (x, h, f) = border_point(1e-4);
    let ctx_off = weingarten_operator(&h, &x);
    let p = ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
    assert_eq!(
        deflated(&p),
        1,
        "fixture: exactly the planted direction is below the floor"
    );
    let (bl, br) = ctx_off.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let pd = dense_projector(&p, &basis);
    let ctx_floor = weingarten_operator(&h, &x).with_floor(FloorPolicy::Hard(f));
    let (_, b_hard) = hessian_grams(&ctx_floor, &basis);
    let ctx_reg = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let (_, b_reg) = hessian_grams(&ctx_reg, &basis);
    let (_, b_off) = hessian_grams(&ctx_off, &basis);
    let want = congruence(&pd, &b_off, n);
    let regm = max_abs(&b_reg);
    let kept = congruence(&pd, &b_hard, n);
    let rel = max_abs_diff(&kept, &want) / regm;
    let rel_reg = max_abs_diff(&b_reg, &want) / regm;
    let asym_hard = asym(&b_hard, n) / max_abs(&b_hard);
    let not_cong = max_abs_diff(&b_hard, &congruence(&pd, &b_hard, n)) / max_abs(&b_hard);
    println!(
        "T4: vs P·B_plain·P, |P·B_hard·P − ·| {rel:.2e}, |Gram(B_reg) − ·| {rel_reg:.2e} (relative to \
         max|B_reg|); B_hard symmetry defect {asym_hard:.2e}, |B_hard − P·B_hard·P| {not_cong:.2e}"
    );
    assert!(
        rel <= 1e-12,
        "the kept block of the in-formula floor is not P·B·P ({rel:e})"
    );
    assert!(
        rel_reg <= 1e-12,
        "Gram(B_reg) is not P·B·P of the plain operator ({rel_reg:e})"
    );
    let ctx_both = weingarten_operator(&h, &x)
        .with_floor(FloorPolicy::Hard(f))
        .with_curvature(CurvatureReg::Hard(f));
    let (_, b_both) = hessian_grams(&ctx_both, &basis);
    assert!(
        flat_bits(&b_both) == flat_bits(&b_reg),
        "with_floor(f) + Hard(f) builds the gauges at max(f, f) = f: Gram(B_reg), bitwise"
    );
    let ctx_rev = weingarten_operator(&h, &x)
        .with_curvature(CurvatureReg::Hard(f))
        .with_floor(FloorPolicy::Hard(f));
    assert_eq!(
        ctx_rev.curvature(),
        CurvatureReg::Hard(f),
        "with_floor keeps the curvature setting"
    );
    for xi in seeds(&ctx_off, &x, 3, 0xC0B4) {
        assert_bitwise(
            &ctx_rev.b_apply(&xi),
            &ctx_both.b_apply(&xi),
            "setter order",
        );
    }
}

fn gauge_bits(gs: &[BondGauge<f64>]) -> (Vec<u64>, Vec<u64>, f64) {
    let (mut rinv, mut sig, mut dropped) = (Vec::new(), Vec::new(), 0.0f64);
    for g in gs {
        for s in g.r_inv.sectors() {
            if let Some(b) = g.r_inv.block(s) {
                for j in 0..b.ncols() {
                    for i in 0..b.nrows() {
                        rinv.push(b.get(i, j).to_bits());
                    }
                }
            }
        }
        for (_, v) in g.svals() {
            sig.extend(v.iter().map(|x| x.to_bits()));
        }
        dropped += g.dropped_mass;
    }
    (rinv, sig, dropped)
}

#[test]
fn cr_hard_builds_the_gauges_floored_at_f() {
    let (x, h, f) = border_point(1e-4);
    let plain = weingarten_operator(&h, &x);
    let cut: Vec<f64> = plain
        .bond_gauges()
        .iter()
        .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
        .filter(|&s| s < f)
        .collect();
    assert_eq!(
        cut.len(),
        1,
        "fixture: exactly the planted direction is below the floor"
    );
    let sp = cut[0];
    let floored = weingarten_operator(&h, &x).with_floor(FloorPolicy::Hard(f));
    let hard = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let hard_rev = weingarten_operator(&h, &x)
        .with_curvature(CurvatureReg::Hard(f))
        .with_floor(FloorPolicy::default_hard());
    let tik = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(f));
    let tik_rev = weingarten_operator(&h, &x)
        .with_curvature(CurvatureReg::Tikhonov(f))
        .with_floor(FloorPolicy::default_hard());
    let (r_p, s_p, m_p) = gauge_bits(plain.bond_gauges());
    let (r_f, s_f, m_f) = gauge_bits(floored.bond_gauges());
    let (r_h, s_h, m_h) = gauge_bits(hard.bond_gauges());
    let (r_r, _, m_r) = gauge_bits(hard_rev.bond_gauges());
    let (r_t, s_t, m_t) = gauge_bits(tik.bond_gauges());
    let (r_tr, _, m_tr) = gauge_bits(tik_rev.bond_gauges());
    let (r_tc, s_tc, m_tc) = gauge_bits(tik.curvature_gauges());
    let (r_trc, _, m_trc) = gauge_bits(tik_rev.curvature_gauges());
    let (bl, br) = plain.bases();
    let (r_d, _, m_d) = gauge_bits(&bond_gauges(bl, br, FloorPolicy::Tikhonov(f)));
    let moved = r_h.iter().zip(&r_p).filter(|(a, b)| a != b).count();
    println!(
        "white-box: σ_p {sp:.4e}, dropped plain {m_p:e} / floored {m_f:e} / Hard {m_h:e}; R⁻¹ \
         coefficients moved by Hard(f): {moved} of {}",
        r_p.len()
    );
    assert_eq!(
        m_p, 0.0,
        "the plain 1e-12 floor drops nothing at this fixture"
    );
    assert_eq!(
        m_f.to_bits(),
        (sp * sp).to_bits(),
        "with_floor(Hard(f)) drops the planted σ_p²"
    );
    assert!(
        r_h == r_f && m_h.to_bits() == m_f.to_bits(),
        "Hard(f) builds the gauges floored at f"
    );
    assert!(
        r_r == r_f && m_r.to_bits() == m_f.to_bits(),
        "Hard(f) then with_floor(1e-12): still max(1e-12, f) = f"
    );
    assert!(
        r_t == r_p && m_t.to_bits() == m_p.to_bits(),
        "Tikhonov(f) leaves bond_gauges() plain"
    );
    assert!(
        r_tr == r_p && m_tr.to_bits() == m_p.to_bits(),
        "Tikhonov(f) then with_floor(1e-12): bond_gauges() still plain"
    );
    assert!(
        r_tc == r_d && m_tc.to_bits() == m_d.to_bits() && m_d == 0.0,
        "Tikhonov(f)'s curvature apply runs on FloorPolicy::Tikhonov(f) gauges (P4d), nothing \
         dropped"
    );
    assert!(
        r_trc == r_d && m_trc.to_bits() == m_d.to_bits(),
        "Tikhonov(f) then with_floor(1e-12): the curvature set is still FloorPolicy::Tikhonov(f) \
         (P4d)"
    );
    assert!(
        r_d != r_p,
        "non-vacuity: the damped inverse differs from the plain one"
    );
    assert!(moved > 0, "non-vacuity: the cut changes R⁻¹");
    assert!(
        s_p == s_f && s_p == s_h && s_p == s_t && s_p == s_tc,
        "the Schmidt values are floor-independent"
    );
}

#[test]
fn cr_signed_solve_matches_the_dense_regularized_pencil() {
    let (x, h, f) = border_point(1e-6);
    let ctx_off = weingarten_operator(&h, &x);
    let (bl, br) = ctx_off.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a, b_off) = hessian_grams(&ctx_off, &basis);
    let hrr_raw: Vec<f64> = a.iter().zip(&b_off).map(|(p, q)| p + q).collect();
    let hii_raw: Vec<f64> = a.iter().zip(&b_off).map(|(p, q)| p - q).collect();
    let (raw_real, raw_cplx) = product_spectrum(&hrr_raw, &hii_raw, n);
    let raw_radius = radius(&raw_real, &raw_cplx);
    let k = 3;
    let opts = SymplecticDavidsonOpts {
        tol: 1.9e-10,
        max_basis: 8 * n,
        max_subspace: n,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let arms: [(&str, CurvatureReg, ColumnDeflation<f64>); 2] = [
        (
            "Hard",
            CurvatureReg::Hard(f),
            ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f),
        ),
        (
            "Tikhonov",
            CurvatureReg::Tikhonov(f),
            ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), tik_weight(f)),
        ),
    ];
    for (label, reg, d) in arms {
        let dd = dense_projector(&d, &basis);
        let b_cong = congruence(&dd, &b_off, n);
        let hrr: Vec<f64> = a.iter().zip(&b_cong).map(|(p, q)| p + q).collect();
        let hii: Vec<f64> = a.iter().zip(&b_cong).map(|(p, q)| p - q).collect();
        let (real, cplx) = product_spectrum(&hrr, &hii, n);
        let reg_radius = radius(&real, &cplx);
        let ctx = weingarten_operator(&h, &x).with_curvature(reg);
        let v0s = seeds(&ctx, &x, k + 2, 0xDEF5);
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        let negatives: Vec<f64> = real.iter().copied().filter(|&w| w < 0.0).collect();
        let positives: Vec<f64> = real.iter().copied().filter(|&w| w >= 0.0).take(k).collect();
        let wanted: Vec<f64> = negatives.iter().chain(&positives).copied().collect();
        let scale = real.iter().fold(1.0f64, |acc, w| acc.max(w.abs()));
        let spec = symplectic_from_grams(&a, &b_cong, n, 1e-10);
        let (arm, wgap) = williamson_vs_product(&spec, &real, &cplx, scale, label);
        println!(
            "T5 {label}: n = {n}, radius raw {raw_radius:.3e} vs regularized {reg_radius:.3e}; dense \
             real {real:?} ({} complex); Williamson arm {arm} (vs dense product {wgap:.2e}); solve \
             ω² {:?} built {} converged {} neg_basis {:?}",
            cplx.len(),
            so.omega2(),
            so.iterations,
            so.converged,
            match &so.report {
                RpaReport::Signed {
                    n_negative_basis, ..
                } => *n_negative_basis,
                RpaReport::Paired { .. } => unreachable!("a signed-kernel pin"),
            }
        );
        assert!(
            raw_radius >= 1e3 * reg_radius,
            "{label}: non-vacuity — the raw spectrum is not border-driven"
        );
        assert_eq!(so.side(), RpaMetric::Ii);
        assert_eq!(
            so.omega2().len(),
            wanted.len(),
            "{label}: every negative + the k lowest"
        );
        assert!(
            so.converged,
            "{label}: the Ii lane is exact at exhaustion AND says so: residuals {:?}",
            so.residuals()
        );
        for (got, want) in so.omega2().iter().zip(&wanted) {
            assert!(
                (got - want).abs() <= 1e-8 * scale,
                "{label}: ω² {got} vs dense {want}"
            );
        }
        if label == "Hard" {
            let spec_ctx = symplectic_eigenvalues(&ctx, &basis);
            assert_eq!(
                std::mem::discriminant(&spec_ctx),
                std::mem::discriminant(&spec),
                "{label}: symplectic_eigenvalues on the regularized context took another arm"
            );
            if let (Some(p), Some(q)) = (spec_ctx.omega_squared(), spec.omega_squared()) {
                for (got, want) in p.iter().zip(&q) {
                    assert!((got - want).abs() <= 1e-8 * scale);
                }
            }
        }
        let rr_opts = SymplecticDavidsonOpts {
            kernel: RpaKernel::signed(RpaMetric::Rr),
            ..opts
        };
        let (so_rr, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &rr_opts);
        println!(
            "T5 {label} Rr lane: ω² {:?} built {} converged {} residuals {:?}",
            so_rr.omega2(),
            so_rr.iterations,
            so_rr.converged,
            so_rr.residuals()
        );
        assert_eq!(so_rr.side(), RpaMetric::Rr);
        assert_eq!(
            so_rr.omega2().len(),
            wanted.len(),
            "{label} Rr: every negative + the k lowest"
        );
        if so_rr.converged {
            for (got, want) in so_rr.omega2().iter().zip(&wanted) {
                assert!(
                    (got - want).abs() <= 1e-8 * scale,
                    "{label}: the Rr lane reports converged, yet ω² {got} vs dense {want}"
                );
            }
        }
    }
}

#[test]
fn cr_mode_checks_read_the_regularized_b() {
    let (x, h, f) = border_point(1e-6);
    let ctx_off = weingarten_operator(&h, &x);
    let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let (bl, br) = ctx.bases();
    let n = tangent_basis(bl, br, true).len();
    let k = 3;
    let v0s = seeds(&ctx, &x, k + 2, 0xDEF5);
    let opts = SymplecticDavidsonOpts {
        tol: 1.9e-10,
        max_basis: 8 * n,
        max_subspace: n,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so, vecs) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
    let with = rpa_mode_checks(&ctx, &so, &vecs, &ModeCheckOpts::default());
    let without = rpa_mode_checks(&ctx_off, &so, &vecs, &ModeCheckOpts::default());
    let (mut eta_with, mut eta_without, mut checked) = (0.0f64, 0.0f64, 0usize);
    for ((c, d), &r) in with.iter().zip(&without).zip(&so.residuals()) {
        if c.omega2.is_nan() || c.omega2 <= 0.0 {
            continue;
        }
        println!(
            "T6 mode {}: ω {:.10} kernel res {r:.3e} | regularized checks: fresh {:.3e} η {:.2e} | \
             plain checks: fresh {:.3e} η {:.2e}",
            c.index, c.omega, c.residual_fresh, c.eta, d.residual_fresh, d.eta
        );
        assert!(
            (c.residual_fresh - r).abs() <= 1e-9 + 1e-3 * r,
            "mode {}: fresh residual {} vs the kernel's {r}",
            c.index,
            c.residual_fresh
        );
        eta_with = eta_with.max(c.eta);
        eta_without = eta_without.max(d.eta);
        checked += 1;
    }
    assert!(checked >= 1, "no positive mode to check");
    assert!(eta_with <= 1e-6, "regularized checks: η {eta_with:e}");
    assert!(
        eta_without >= 1e3 * eta_with.max(1e-12),
        "non-vacuity: the plain context's checks do not see the raw coupling (η {eta_without:e})"
    );
}

#[test]
fn cr_off_and_identity_projector_are_bitwise_today() {
    let (x, h, f) = border_point(1e-4);
    let smin = sigma_min(&x);
    assert!(smin < f, "fixture: the planted value is the point's σ_min");
    assert_eq!(CurvatureReg::default(), CurvatureReg::Off);
    let ctx_plain = weingarten_operator(&h, &x);
    assert_eq!(ctx_plain.curvature(), CurvatureReg::Off);
    let variants: Vec<(&str, CurvatureReg, WeingartenCtx<'_, f64>)> = vec![
        (
            "Off",
            CurvatureReg::Off,
            weingarten_operator(&h, &x).with_curvature(CurvatureReg::Off),
        ),
        (
            "Hard(σ_min/2)",
            CurvatureReg::Hard(0.5 * smin),
            weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(0.5 * smin)),
        ),
        (
            "Hard(σ_min), strict",
            CurvatureReg::Hard(smin),
            weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(smin)),
        ),
        (
            "Tikhonov(1e-12)",
            CurvatureReg::Tikhonov(1e-12),
            weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(1e-12)),
        ),
        (
            "Hard(f) then Off",
            CurvatureReg::Off,
            weingarten_operator(&h, &x)
                .with_curvature(CurvatureReg::Hard(f))
                .with_curvature(CurvatureReg::Off),
        ),
    ];
    let tangents = seeds(&ctx_plain, &x, 3, 0x0FF0);
    for (label, reg, ctx) in &variants {
        assert_eq!(
            ctx.curvature(),
            *reg,
            "{label}: the accessor echoes the setting"
        );
        for xi in &tangents {
            assert_bitwise(
                &ctx.b_apply(xi),
                &ctx_plain.b_apply(xi),
                &format!("{label}: b_apply"),
            );
        }
    }
    let fresh_tik = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(f));
    let used = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let hard_out: Vec<TtTangent<f64>> = tangents.iter().map(|xi| used.b_apply(xi)).collect();
    let used_tik = used.with_curvature(CurvatureReg::Tikhonov(f));
    let used_id = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let _ = used_id.b_apply(&tangents[0]);
    let used_id = used_id.with_curvature(CurvatureReg::Hard(0.5 * smin));
    for (xi, hb) in tangents.iter().zip(&hard_out) {
        let tb = fresh_tik.b_apply(xi);
        assert_bitwise(
            &used_tik.b_apply(xi),
            &tb,
            "used Hard(f) → Tikhonov(f): b_apply",
        );
        assert_bitwise(
            &used_id.b_apply(xi),
            &ctx_plain.b_apply(xi),
            "used Hard(f) → Hard(σ_min/2): b_apply",
        );
        assert!(
            tb.sub(hb).norm() > 1e-8 * hb.norm(),
            "Tikhonov(f) and Hard(f) must give different b_apply"
        );
    }
    let used_tik_f = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(f));
    let tik_f_out: Vec<TtTangent<f64>> = tangents.iter().map(|xi| used_tik_f.b_apply(xi)).collect();
    let used_tik_4f = used_tik_f.with_curvature(CurvatureReg::Tikhonov(4.0 * f));
    let fresh_tik_4f = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(4.0 * f));
    for (xi, b_f) in tangents.iter().zip(&tik_f_out) {
        let want = fresh_tik_4f.b_apply(xi);
        let moved = want.sub(b_f).norm() / b_f.norm();
        println!("T-a2: |B(Tikhonov(4f)) − B(Tikhonov(f))|/|B(Tikhonov(f))| {moved:.3e}");
        assert!(
            moved > 1e-8,
            "non-vacuity: Tikhonov(4f) and Tikhonov(f) must give different b_apply"
        );
        assert_bitwise(
            &used_tik_4f.b_apply(xi),
            &want,
            "used Tikhonov(f) → Tikhonov(4f): b_apply",
        );
    }
    let ctx_cut = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    for xi in &tangents {
        assert_bitwise(
            &ctx_cut.a_apply(xi),
            &ctx_plain.a_apply(xi),
            "Hard(f): a_apply",
        );
        assert_bitwise(
            &ctx_cut.b_apply_damped(xi, 1e-5),
            &ctx_plain.b_apply_damped(xi, 1e-5),
            "Hard(f): b_apply_damped is not a CurvatureReg reader",
        );
    }
    let moved = tangents
        .iter()
        .map(|xi| ctx_cut.b_apply(xi).sub(&ctx_plain.b_apply(xi)).norm())
        .fold(0.0f64, f64::max);
    assert!(moved > 1e-6, "Hard(f) left b_apply unchanged ({moved:e})");
    let opts = SymplecticDavidsonOpts {
        max_basis: 80,
        ..Default::default()
    };
    let v0s = seeds(&ctx_plain, &x, 5, 0x0B17);
    let (so_p, vec_p) = symplectic_lowest_signed_davidson(&ctx_plain, &v0s, 3, &opts);
    for (label, _, ctx) in variants.iter().take(2) {
        let (so, vecs) = symplectic_lowest_signed_davidson(ctx, &v0s, 3, &opts);
        assert!(
            flat_bits(&so.omega2()) == flat_bits(&so_p.omega2()),
            "{label}: ω² bits"
        );
        assert_eq!(so.iterations, so_p.iterations, "{label}: iterations");
        assert_eq!(so.applies, so_p.applies, "{label}: applies");
        assert!(
            flat_bits(&so.residuals()) == flat_bits(&so_p.residuals()),
            "{label}: residuals"
        );
        assert_eq!(vecs.len(), vec_p.len());
        for (v, w) in vecs.iter().zip(&vec_p) {
            assert_bitwise(&v.x, &w.x, &format!("{label}: X"));
            assert_bitwise(&v.y, &w.y, &format!("{label}: Y"));
        }
    }
}

#[test]
fn cr_auto_metric_resolves_to_ii_without_a_probe() {
    let (x, h, f) = border_point(1e-4);
    let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let k = 3;
    let v0s = seeds(&ctx, &x, k + 2, 0xA070);
    let (probed, verdict, _) = select_rpa_metric(&ctx, &v0s[0], 200);
    println!("T8 control: select_rpa_metric → {probed:?} (H^ii verdict {verdict:?})");
    assert_eq!(
        probed,
        RpaMetric::Rr,
        "control: a probe at this point swaps to Rr"
    );
    let (ii_verdict, ii_applies) = hessian_pd_probe(&ctx, HessPart::Ii, &v0s[0], 200);
    let PdVerdict::WitnessNegative { lambda } = ii_verdict else {
        panic!("control: the H^ii probe proves indefiniteness here, got {ii_verdict:?}")
    };
    let (bl, br) = ctx.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a, b_reg) = hessian_grams(&ctx, &basis);
    let (_, b_raw) = hessian_grams(&weingarten_operator(&h, &x), &basis);
    let lmin = |b: &[f64]| {
        let hii: Vec<f64> = a.iter().zip(b).map(|(p, q)| p - q).collect();
        dense_self_adjoint_smallest::<f64>(&hii, n, 0).0[0]
    };
    let (lmin_reg, lmin_raw) = (lmin(&b_reg), lmin(&b_raw));
    println!(
        "T8 probe: H^ii witness λ = {lambda:.6e} after {ii_applies} applies; dense λ_min(A − P·B·P) \
         = {lmin_reg:.6e}, λ_min(A − B_raw) = {lmin_raw:.6e}"
    );
    assert!(
        lambda >= lmin_reg - 1e-8,
        "the probe's witness {lambda:e} is below λ_min(A − P·B·P) = {lmin_reg:e}: it read the raw B"
    );
    assert!(
        lmin_raw < lmin_reg - 1.0,
        "non-vacuity: the raw H^ii reaches below the regularized one"
    );
    let opts_with = |kernel: RpaKernel| SymplecticDavidsonOpts {
        max_basis: 80,
        kernel,
        ..Default::default()
    };
    let (so_auto, _) = symplectic_lowest_signed_davidson(
        &ctx,
        &v0s,
        k,
        &opts_with(RpaKernel::Signed {
            metric: RpaMetric::Auto,
            probe_budget: 200,
        }),
    );
    let (so_ii, _) = symplectic_lowest_signed_davidson(
        &ctx,
        &v0s,
        k,
        &opts_with(RpaKernel::signed(RpaMetric::Ii)),
    );
    assert_eq!(
        so_auto.side(),
        RpaMetric::Ii,
        "Auto on a regularized context runs the Ii lane"
    );
    assert!(
        flat_bits(&so_auto.omega2()) == flat_bits(&so_ii.omega2()),
        "Auto vs Ii: ω² bits"
    );
    assert_eq!(so_auto.applies, so_ii.applies);
    assert_eq!(so_auto.iterations, so_ii.iterations);
}

#[test]
fn cr_shifted_b_gram_is_pd_b_pd() {
    let x = pg4_curved_state(0x7C01);
    let space = x.space;
    let (t, v) = pg_symmetric_tv(&space.pg, space.d, 0xC5B1);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let ctx_off = weingarten_operator(&h, &x);
    let (bl, br) = ctx_off.bases();
    let phi = (1..space.pg.order())
        .map(Irrep::new)
        .find(|&p| crate::core::tangent::tangent_dim_grids(&space, &bl.ranks, p) > 0)
        .expect("fixture premise: pg4c carries a live shifted channel");
    let (f, blocks) = relabel_sensitive_floor(bl, br, ctx_off.bond_gauges(), phi)
        .expect("fixture premise: a relabel-sensitive floor on the curved channel");
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx_off = SectorCtx::new(&ctx_off, &h_out, phi);
    let basis = sctx_off.basis();
    let n = basis.len();
    let (a_off, b_raw) = sctx_off.grams(&basis);
    let raw = max_abs(&b_raw);
    println!(
        "T9 re-host: phi {phi:?}, n = {n}, floor {f:.6e}, premise blocks {blocks:?}, max|B_raw| \
         {raw:.3e}"
    );
    assert!(
        n >= 2,
        "fixture premise: the curved channel carries {n} directions"
    );
    assert!(
        raw > 1e-6,
        "fixture premise: B^phi is not zero here (max|B| {raw:e})"
    );
    let arms: [(&str, CurvatureReg, ColumnDeflation<f64>); 2] = [
        (
            "Hard",
            CurvatureReg::Hard(f),
            ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f),
        ),
        (
            "Tikhonov",
            CurvatureReg::Tikhonov(f),
            ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), tik_weight(f)),
        ),
    ];
    for (label, reg, d) in arms {
        let ctx = weingarten_operator(&h, &x).with_curvature(reg);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let (a_reg, b_reg) = sctx.grams(&basis);
        let dd = dense_projector(&d, &basis);
        let want = congruence(&dd, &b_raw, n);
        let regm = max_abs(&b_reg);
        let err = max_abs_diff(&b_reg, &want);
        let sym = asym(&b_reg, n);
        let moved = max_abs_diff(&b_reg, &b_raw);
        let sym_bound = 1e-14 * regm + 1e-15 * raw;
        println!(
            "T9 re-host [{label}]: max|B_reg| {regm:.3e}, |Gram - D.B.D| {err:.2e}, symmetry \
             {sym:.2e} (bound {sym_bound:.2e}), max|B_reg - B_raw| {moved:.3e}"
        );
        assert!(
            err <= 1e-12 * raw.max(1.0),
            "[{label}] the shifted channel's regularized B Gram is not D.B.D ({err:e})"
        );
        assert!(
            sym <= sym_bound,
            "[{label}] symmetry defect {sym:e} > {sym_bound:e}"
        );
        assert!(
            flat_bits(&a_reg) == flat_bits(&a_off),
            "[{label}] the A Gram must be bitwise the Off channel's (A is never regularized)"
        );
        assert!(
            moved >= 1e-6 * raw,
            "[{label}] non-vacuity: the regularization did not move the shifted B ({moved:e})"
        );
        let ctx_inner = match reg {
            CurvatureReg::Hard(fh) => weingarten_operator(&h, &x).with_floor(FloorPolicy::Hard(fh)),
            _ => weingarten_operator(&h, &x),
        };
        let sctx_inner = SectorCtx::new(&ctx_inner, &h_out, phi);
        for xi in basis.iter().take(3) {
            let conj = |inner: &SectorCtx<'_, '_, f64>| {
                let mut u = applied(&d, xi);
                u = inner.b_apply(&u);
                d.apply(&mut u);
                u
            };
            let bt = match reg {
                CurvatureReg::Tikhonov(e) => tik_composition(&ctx_off, e, xi),
                _ => conj(&sctx_inner),
            };
            let b_plain = conj(&sctx_off);
            let dev = bt.sub(&b_plain).norm();
            assert!(
                dev <= 1e-12 * b_plain.norm().max(1.0),
                "[{label}] D.B_inner.D vs D.B_off.D {dev:e} (the same operator)"
            );
            let a = sctx_off.a_apply(xi);
            assert_bitwise(
                &sctx.b_apply(xi),
                &bt,
                &format!("[{label}] b_apply = D.B_inner.D (Hard) / the P4d composition (Tikhonov)"),
            );
            assert_bitwise(
                &sctx.hpp_apply(xi),
                &a.add(&bt),
                &format!("[{label}] hpp = A + D.B.D"),
            );
            assert_bitwise(
                &sctx.hmm_apply(xi),
                &a.sub(&bt),
                &format!("[{label}] hmm = A - D.B.D"),
            );
        }
    }
}

#[test]
#[should_panic(expected = "phi = IDENTITY")]
fn cr_sector_ctx_identity_refusal_survives_a_regularized_context() {
    let (x, h, f) = border_point(1e-4);
    let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let _ = SectorCtx::new(&ctx, &h, Irrep::IDENTITY);
}

#[test]
#[should_panic(expected = "CurvatureReg")]
fn cr_with_curvature_refuses_a_nonpositive_scale() {
    let x = base_state();
    let h = generic_truncated_h::<f64>(x.space);
    let _ = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(0.0));
}

fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

#[test]
fn cr_with_curvature_validates_the_scale_and_names_it() {
    let x = base_state();
    let h = generic_truncated_h::<f64>(x.space);
    let bad = [
        CurvatureReg::Hard(0.0),
        CurvatureReg::Hard(-0.0),
        CurvatureReg::Hard(-1.0),
        CurvatureReg::Hard(f64::NAN),
        CurvatureReg::Hard(f64::INFINITY),
        CurvatureReg::Tikhonov(0.0),
        CurvatureReg::Tikhonov(-1.0),
        CurvatureReg::Tikhonov(f64::NAN),
        CurvatureReg::Tikhonov(f64::NEG_INFINITY),
    ];
    for reg in bad {
        let v = match reg {
            CurvatureReg::Hard(v) | CurvatureReg::Tikhonov(v) => v,
            CurvatureReg::Off => unreachable!(),
        };
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = weingarten_operator(&h, &x).with_curvature(reg);
        }));
        let payload = caught.expect_err(&format!("{reg:?} must be refused"));
        let msg = panic_text(payload.as_ref());
        assert!(
            msg.contains("CurvatureReg") && msg.contains(&format!("{v}")),
            "{reg:?}: the refusal must name the type and the value, got {msg:?}"
        );
    }
    for ok in [
        CurvatureReg::Off,
        CurvatureReg::Hard(f64::MIN_POSITIVE),
        CurvatureReg::Tikhonov(1e300),
    ] {
        assert_eq!(
            weingarten_operator(&h, &x).with_curvature(ok).curvature(),
            ok
        );
    }
}

const P4D_S1: Terms = Terms {
    diag_s1: true,
    diag_s2: false,
    diag_s3: false,
};
const P4D_S1_S2: Terms = Terms {
    diag_s1: true,
    diag_s2: true,
    diag_s3: false,
};
const P4D_S3: Terms = Terms {
    diag_s1: false,
    diag_s2: false,
    diag_s3: true,
};

const P4D_TV_SEED: u64 = 0xC1A5_7E07;

fn p4d_cluster_values() -> Vec<f64> {
    CLUSTER.iter().map(|m| m * CLUSTER_FLOOR).collect()
}

fn p4d_leak_values() -> Vec<f64> {
    let mut v = p4d_cluster_values();
    v[0] = 1e-10;
    v
}

fn assemble_at(
    ctx: &WeingartenCtx<'_, f64>,
    xi: &TtTangent<f64>,
    gauges: &[BondGauge<f64>],
    terms: Terms,
) -> TtTangent<f64> {
    let e = ctx.envs();
    let zhi = ImplicitHZ {
        h: ctx.h,
        ket: &ctx.base_l,
        ws: &e.ws,
        envs: Some((&e.pz_hi, &e.qz_hi)),
    };
    let zlo = ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
    weingarten_assemble_sources(&ctx.base_l, &ctx.base_r, &[&zhi, &zlo], xi, gauges, terms)
}

fn old_s3_at(
    ctx: &WeingartenCtx<'_, f64>,
    xi: &TtTangent<f64>,
    gauges: &[BondGauge<f64>],
) -> TtTangent<f64> {
    let e = ctx.envs();
    let zhi = ImplicitHZ {
        h: ctx.h,
        ket: &ctx.base_l,
        ws: &e.ws,
        envs: Some((&e.pz_hi, &e.qz_hi)),
    };
    let zlo = ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
    weingarten_right_chain_oracle(&ctx.base_l, &ctx.base_r, &[&zhi, &zlo], xi, gauges)
}

fn tik_composition(ctx: &WeingartenCtx<'_, f64>, eps: f64, xi: &TtTangent<f64>) -> TtTangent<f64> {
    let (bl, br) = ctx.bases();
    let g = bond_gauges(bl, br, FloorPolicy::Tikhonov(eps));
    let d = ColumnDeflation::from_gauges_weighted(&g, tik_weight(eps));
    let mut first = assemble_at(ctx, xi, &g, P4D_S1_S2);
    d.apply(&mut first);
    let second = assemble_at(ctx, &applied(&d, xi), &g, P4D_S3);
    first.add(&second)
}

fn symmetrized_s1_form(
    ctx: &WeingartenCtx<'_, f64>,
    op: &ColumnDeflation<f64>,
    g_reg: &[BondGauge<f64>],
    xis: &[TtTangent<f64>],
) -> Vec<f64> {
    let n = xis.len();
    let s1: Vec<TtTangent<f64>> = xis
        .iter()
        .map(|xi| applied(op, &assemble_at(ctx, xi, g_reg, P4D_S1)))
        .collect();
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = xis[i].dot(&s1[j]) + s1[i].dot(&xis[j]);
        }
    }
    g
}

fn form_from(xis: &[TtTangent<f64>], images: &[TtTangent<f64>]) -> Vec<f64> {
    let n = xis.len();
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = xis[i].dot(&images[j]);
        }
    }
    g
}

struct P4dPoint {
    label: &'static str,
    x: TtVector<f64>,
    h: SparseHamiltonian<f64>,
    channel: Option<(Irrep, SparseHamiltonian<f64>)>,
    f: f64,
    census: ClusterCensus,
}

impl P4dPoint {
    fn flux0(values: &[f64]) -> Self {
        let (x, plant) = with_schmidt_cluster(&cluster_base_state(), &CLUSTER_BONDS, values);
        let census = assert_cluster_premise("P4d flux 0", &x, &plant, CLUSTER_FLOOR, true);
        let h = generic_truncated_h::<f64>(x.space);
        P4dPoint {
            label: "flux 0",
            x,
            h,
            channel: None,
            f: CLUSTER_FLOOR,
            census,
        }
    }

    fn shifted(values: &[f64]) -> Self {
        let (x, plant) = with_schmidt_cluster(&pg_base_state(), &CLUSTER_BONDS, values);
        let census = assert_cluster_premise("P4d shifted", &x, &plant, CLUSTER_FLOOR, false);
        let space = x.space;
        let (t, v) = pg_symmetric_tv(&space.pg, space.d, P4D_TV_SEED);
        let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
        let (bl, br) = tangent_bases(&x);
        let g = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        let p = ColumnDeflation::from_gauges(&g, CLUSTER_FLOOR);
        let removed = |ph: Irrep| p.removed_dim(&unit_tangents(&bl, &br, ph, 1, 0x9D41_0000)[0]);
        let phi = (1..space.pg.order())
            .map(Irrep::new)
            .filter(|&ph| tangent_dim_grids(&space, &bl.ranks, ph) > 0)
            .find(|&ph| removed(ph) >= 2)
            .expect("fixture premise: a live shifted channel the cut reaches");
        println!(
            "P4d shifted: phi {phi:?}, dim T^phi {}, removed by Hard(f) {}",
            tangent_dim_grids(&space, &bl.ranks, phi),
            removed(phi)
        );
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        P4dPoint {
            label: "shifted",
            x,
            h,
            channel: Some((phi, h_out)),
            f: CLUSTER_FLOOR,
            census,
        }
    }

    fn phi(&self) -> Irrep {
        self.channel.as_ref().map_or(Irrep::IDENTITY, |c| c.0)
    }

    fn ctx(&self, reg: CurvatureReg) -> WeingartenCtx<'_, f64> {
        weingarten_operator(&self.h, &self.x).with_curvature(reg)
    }

    fn tangents(&self, ctx: &WeingartenCtx<'_, f64>, n: usize, seed: u64) -> Vec<TtTangent<f64>> {
        let (bl, br) = ctx.bases();
        unit_tangents(bl, br, self.phi(), n, seed)
    }

    fn b<'s>(&'s self, ctx: &WeingartenCtx<'s, f64>, xi: &TtTangent<f64>) -> TtTangent<f64> {
        match &self.channel {
            None => ctx.b_apply(xi),
            Some((phi, h_out)) => SectorCtx::new(ctx, h_out, *phi).b_apply(xi),
        }
    }
}

#[test]
fn cr_p4d_off_b_is_the_pre_p4d_assembly() {
    for point in [
        P4dPoint::flux0(&p4d_cluster_values()),
        P4dPoint::shifted(&p4d_cluster_values()),
    ] {
        let ctx = point.ctx(CurvatureReg::Off);
        let g = ctx.bond_gauges();
        assert!(
            g.iter().all(|gg| gg.dropped_mass == 0.0),
            "[{}] premise: the plain gauges cut nothing",
            point.label
        );
        let mut worst = 0.0f64;
        for xi in point.tangents(&ctx, 4, 0x9D41_0006) {
            let b_new = point.b(&ctx, &xi);
            let b_old = assemble_at(&ctx, &xi, g, P4D_S1_S2).add(&old_s3_at(&ctx, &xi, g));
            assert!(b_old.norm() > 1e-8, "[{}] non-vacuity", point.label);
            worst = worst.max(b_new.sub(&b_old).norm() / b_old.norm());
        }
        println!(
            "P4d T6 [{}]: |B_off − B_pre-P4d|/|B_pre-P4d| {worst:.2e}",
            point.label
        );
        assert!(
            worst <= 1e-13,
            "[{}] T6: the Off B is not the pre-P4d assembly ({worst:e})",
            point.label
        );
    }
}

#[test]
fn cr_p4d_regularized_b_is_symmetric_at_a_cluster() {
    let mut rows = Vec::new();
    for point in [
        P4dPoint::flux0(&p4d_cluster_values()),
        P4dPoint::shifted(&p4d_cluster_values()),
    ] {
        let f = point.f;
        for (label, reg) in [
            ("Hard", CurvatureReg::Hard(f)),
            ("Tikhonov", CurvatureReg::Tikhonov(f)),
        ] {
            let ctx = point.ctx(reg);
            let xis = point.tangents(&ctx, 6, 0x9D41_0007);
            let images: Vec<TtTangent<f64>> = xis.iter().map(|xi| point.b(&ctx, xi)).collect();
            let g = form_from(&xis, &images);
            let scale = max_abs(&g);
            let rel = asym(&g, xis.len()) / scale;
            println!(
                "P4d T-sym [{} {label}]: max|G| {scale:.3e}, relative asymmetry {rel:.2e}",
                point.label
            );
            rows.push((point.label, label, rel, scale));
        }
    }
    for (pl, label, rel, scale) in rows {
        assert!(scale > 1e-8, "[{pl} {label}] non-vacuity: B vanishes");
        assert!(
            rel <= 1e-13,
            "[{pl} {label}] T-sym: the regularized B is not symmetric ({rel:e})"
        );
    }
}

#[test]
fn cr_p4d_regularized_b_is_the_beside_congruence_at_a_cluster() {
    let mut rows = Vec::new();
    for point in [
        P4dPoint::flux0(&p4d_cluster_values()),
        P4dPoint::shifted(&p4d_cluster_values()),
    ] {
        let f = point.f;
        let ctx_off = point.ctx(CurvatureReg::Off);
        let p = ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
        let dt = ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), tik_weight(f));
        let xis = point.tangents(&ctx_off, 4, 0x9D41_0008);
        for (label, reg, op) in [
            ("Hard", CurvatureReg::Hard(f), &p),
            ("Tikhonov", CurvatureReg::Tikhonov(f), &dt),
        ] {
            let ctx = point.ctx(reg);
            let (mut worst, mut moved) = (0.0f64, f64::INFINITY);
            for xi in &xis {
                let want = applied(op, &point.b(&ctx_off, &applied(op, xi)));
                let b = point.b(&ctx_off, xi);
                worst = worst.max(point.b(&ctx, xi).sub(&want).norm() / want.norm());
                moved = moved.min(want.sub(&b).norm() / b.norm());
            }
            println!(
                "P4d T8 [{} {label}]: |B_reg − D·B·D|/|D·B·D| {worst:.2e}; the congruence moves B \
                 by ≥ {moved:.3e}",
                point.label
            );
            rows.push((point.label, label, worst, moved));
        }
    }
    for (pl, label, worst, moved) in rows {
        assert!(
            moved >= 1e-6,
            "[{pl} {label}] non-vacuity: the congruence does not move B ({moved:e})"
        );
        assert!(
            worst <= 1e-12,
            "[{pl} {label}] T8: B_reg is not the congruence D·B_plain·D ({worst:e})"
        );
    }
}

#[test]
fn cr_p4d_no_beside_leakage_at_a_tiny_planted_value() {
    let mut rows = Vec::new();
    for point in [
        P4dPoint::flux0(&p4d_leak_values()),
        P4dPoint::shifted(&p4d_leak_values()),
    ] {
        let tiny = point
            .census
            .cut
            .iter()
            .map(|c| c.2)
            .fold(f64::INFINITY, f64::min);
        println!(
            "P4d leakage [{}]: smallest planted value {tiny:.3e}",
            point.label
        );
        assert!(
            tiny <= 1e-9,
            "[{}] premise: a planted value at ~1e-10 survives the plant ({tiny:e})",
            point.label
        );
        let f = point.f;
        let ctx_off = point.ctx(CurvatureReg::Off);
        let (bl, br) = ctx_off.bases();
        let p = ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
        let dt = ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), tik_weight(f));
        let g_floor = bond_gauges(bl, br, FloorPolicy::Hard(f));
        let g_damp = bond_gauges(bl, br, FloorPolicy::Tikhonov(f));
        let xis = point.tangents(&ctx_off, 5, 0x9D41_0009);
        let n = xis.len();
        for (label, reg, op, g_reg) in [
            ("Hard", CurvatureReg::Hard(f), &p, &g_floor),
            ("Tikhonov", CurvatureReg::Tikhonov(f), &dt, &g_damp),
        ] {
            let ctx = point.ctx(reg);
            let images: Vec<TtTangent<f64>> = xis.iter().map(|xi| point.b(&ctx, xi)).collect();
            let g_ctx = form_from(&xis, &images);
            let g_or = symmetrized_s1_form(&ctx_off, op, g_reg, &xis);
            let scale = max_abs(&g_or);
            let err = max_abs_diff(&g_ctx, &g_or) / scale;
            let sym = asym(&g_ctx, n) / scale;
            let beside: Vec<TtTangent<f64>> = xis
                .iter()
                .map(|xi| applied(op, &point.b(&ctx_off, &applied(op, xi))))
                .collect();
            let err_beside = max_abs_diff(&form_from(&xis, &beside), &g_or) / scale;
            let cut_cols = images
                .iter()
                .map(|b| b.sub(&applied(&p, b)).norm() / b.norm())
                .fold(0.0f64, f64::max);
            println!(
                "P4d leakage [{} {label}]: |G − oracle| {err:.2e}, asymmetry {sym:.2e} (scale \
                 {scale:.3e}); anti-oracle: the beside form misses by {err_beside:.2e}; \
                 |(I − P)B ξ|/|B ξ| {cut_cols:.2e}",
                point.label
            );
            rows.push((point.label, label, err, sym, err_beside, cut_cols));
        }
    }
    for (pl, label, err, sym, err_beside, cut_cols) in rows {
        if label == "Hard" {
            assert!(
                cut_cols <= 1e-14,
                "[{pl} Hard] the cut columns of B_hard ξ are not zero ({cut_cols:e})"
            );
        } else {
            assert!(
                err_beside >= 1e-11,
                "[{pl} Tikhonov] anti-oracle: the beside form shows no residue ({err_beside:e}) — \
                 the pin cannot tell the two forms apart"
            );
        }
        assert!(
            err <= 1e-12,
            "[{pl} {label}] leakage: the regularized form is not the residue-free oracle ({err:e})"
        );
        assert!(
            sym <= 1e-13,
            "[{pl} {label}] leakage: the regularized B is not symmetric ({sym:e})"
        );
    }
}

#[test]
fn cr_p4d_kept_block_identity_at_a_cluster() {
    let point = P4dPoint::flux0(&p4d_cluster_values());
    let f = point.f;
    let ctx_off = point.ctx(CurvatureReg::Off);
    let (bl, br) = ctx_off.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let p = ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
    let pd = dense_projector(&p, &basis);
    let (_, b_off) = hessian_grams(&ctx_off, &basis);
    let ctx_floor = weingarten_operator(&point.h, &point.x).with_floor(FloorPolicy::Hard(f));
    let (_, b_floor) = hessian_grams(&ctx_floor, &basis);
    let (_, b_reg) = hessian_grams(&point.ctx(CurvatureReg::Hard(f)), &basis);
    let want = congruence(&pd, &b_off, n);
    let regm = max_abs(&b_reg);
    let kept = max_abs_diff(&congruence(&pd, &b_floor, n), &want) / regm;
    let reg_err = max_abs_diff(&b_reg, &want) / regm;
    let reg_sym = asym(&b_reg, n) / regm;
    let floor_scale = max_abs(&b_floor);
    let not_cong = max_abs_diff(&b_floor, &congruence(&pd, &b_floor, n)) / floor_scale;
    let floor_sym = asym(&b_floor, n) / floor_scale;
    println!(
        "P4d E8 cluster: n = {n}, max|B_off| {:.3e}, max|B_reg| {regm:.3e}; |P·B_f·P − P·B·P| \
         {kept:.2e}, |Gram(B_reg) − P·B·P| {reg_err:.2e}, Gram(B_reg) asymmetry {reg_sym:.2e} \
         (relative to max|B_reg|); B_f: |B_f − P·B_f·P| {not_cong:.2e}, asymmetry {floor_sym:.2e} \
         (relative to max|B_f|)",
        max_abs(&b_off)
    );
    assert!(
        not_cong >= 1e-8,
        "non-vacuity: the in-formula floor alone is already the congruence ({not_cong:e})"
    );
    assert!(
        kept <= 1e-12,
        "kept-block identity at the cluster: P·B_f·P is not P·B·P ({kept:e})"
    );
    assert!(
        reg_err <= 1e-12,
        "Gram(B_reg) is not P·B·P of the plain Gram at the cluster ({reg_err:e})"
    );
    assert!(
        reg_sym <= 1e-13,
        "Gram(B_reg) is not symmetric at the cluster ({reg_sym:e})"
    );
}
