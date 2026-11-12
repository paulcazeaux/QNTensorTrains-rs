// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::Su2ImplicitHZ;
use crate::core::border_fixture::{CLUSTER, CLUSTER_FLOOR};
use crate::core::deflate::oracle::{
    asym, congruence, max_abs, max_abs_diff, product_spectrum, radius,
};
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::weingarten::{FloorPolicy, Terms};
use crate::solvers::hess::curvature_tests::williamson_vs_product;
use crate::solvers::hess::{
    CurvatureReg, HessPart, PdVerdict, RpaKernel, RpaMetric, SymplecticDavidsonOpts,
    symplectic_from_grams,
};
use crate::su2::border_fixture::{BOND, base_state, dense_form, w_hat_min, with_small_schmidt};
use crate::su2::border_fixture::{
    Su2ClusterCensus, assert_cluster_premise, cluster_state, unit_tangents,
};
use crate::su2::deflate::Su2ColumnDeflation;
use crate::su2::flux::Su2ShiftedFrames;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hess::{
    Su2SectorCtx, Su2WeingartenCtx, hessian_grams, hessian_pd_probe, su2_select_rpa_metric,
    symplectic_lowest_signed_davidson, weingarten_operator,
};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use crate::su2::weingarten::right_chain_oracle::su2_weingarten_right_chain_oracle;
use crate::su2::weingarten::{Su2BondGauge, tangent_basis};
use crate::su2::weingarten::{Su2ExplicitZ, bond_gauges, weingarten_assemble_sources_dressed};
use crate::test_util::flat_bits;
use crate::test_util::generic_su2_h;
use crate::test_util::su2_tangent_bits;

fn border_point(eps: f64) -> (Su2TtVector<f64>, Su2SparseHamiltonian<f64>, f64) {
    let x0 = base_state();
    let f = (w_hat_min(&x0) * 1e-4).sqrt();
    let x = with_small_schmidt(&x0, BOND, 1, eps);
    let h = generic_su2_h::<f64>(x.space);
    (x, h, f)
}

fn raw_sigma_min(ctx: &Su2WeingartenCtx<'_, f64>) -> f64 {
    ctx.bond_gauges()
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
        .fold(f64::INFINITY, f64::min)
}

fn seeds(
    ctx: &Su2WeingartenCtx<'_, f64>,
    x: &Su2TtVector<f64>,
    n: usize,
    seed: u64,
) -> Vec<Su2TtTangent<f64>> {
    let (bl, br) = ctx.bases();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let d: Su2TtVector<f64> = crate::su2::vector::tt_randn(x.space, &x.ranks, &mut rng);
            Su2TtTangent::project(bl, br, &d)
        })
        .collect()
}

fn assert_bitwise(got: &Su2TtTangent<f64>, want: &Su2TtTangent<f64>, what: &str) {
    assert!(
        su2_tangent_bits(got) == su2_tangent_bits(want),
        "{what}: not bitwise equal (‖Δ‖ = {:e})",
        got.sub(want).norm()
    );
}

fn deflated(p: &Su2ColumnDeflation<f64>) -> usize {
    p.bonds().iter().map(|r| r.deflated).sum()
}

#[test]
fn cr_su2_signed_solve_matches_the_dense_regularized_pencil() {
    let k = 3;
    for (label, eps) in [("Hard", 1e-6), ("Tikhonov", 1e-5), ("Tikhonov", 1e-6)] {
        let (x, h, f) = border_point(eps);
        let ctx_off = weingarten_operator(&h, &x);
        let (bl, br) = ctx_off.bases();
        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let (a, b_off) = hessian_grams(&ctx_off, &basis);
        let hrr_raw: Vec<f64> = a.iter().zip(&b_off).map(|(p, q)| p + q).collect();
        let hii_raw: Vec<f64> = a.iter().zip(&b_off).map(|(p, q)| p - q).collect();
        let (raw_real, raw_cplx) = product_spectrum(&hrr_raw, &hii_raw, n);
        let raw_radius = radius(&raw_real, &raw_cplx);
        let p = Su2ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
        assert_eq!(
            deflated(&p),
            1,
            "fixture: the floor cuts the planted multiplet only"
        );
        let opts = SymplecticDavidsonOpts {
            tol: 1.7e-10,
            max_basis: 8 * n,
            max_subspace: n,
            kernel: RpaKernel::signed(RpaMetric::Ii),
            ..Default::default()
        };
        let (reg, d) = if label == "Hard" {
            (CurvatureReg::Hard(f), p)
        } else {
            (
                CurvatureReg::Tikhonov(f),
                Su2ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), |s| {
                    s * s / (s * s + f * f)
                }),
            )
        };
        let dd = dense_form(&basis, |v| {
            let mut w = v.clone();
            d.apply(&mut w);
            w
        });
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
            "S5 {label}: n = {n}, radius raw {raw_radius:.3e} vs regularized {reg_radius:.3e}; dense \
             real {real:?} ({} complex); Williamson arm {arm} (vs dense product {wgap:.2e}); solve \
             ω² {:?} built {} converged {} residuals {:?}",
            cplx.len(),
            so.omega2(),
            so.iterations,
            so.converged,
            so.residuals()
        );
        assert!(raw_radius >= 1e3 * reg_radius, "{label}: non-vacuity");
        assert_eq!(so.side(), RpaMetric::Ii);
        assert_eq!(
            so.omega2().len(),
            wanted.len(),
            "{label}: every negative + the k lowest"
        );
        assert!(
            so.converged,
            "{label}: Ii lane residuals {:?}",
            so.residuals()
        );
        for (got, want) in so.omega2().iter().zip(&wanted) {
            assert!(
                (got - want).abs() <= 1e-8 * scale,
                "{label}: ω² {got} vs dense {want}"
            );
        }
        let rr_opts = SymplecticDavidsonOpts {
            kernel: RpaKernel::signed(RpaMetric::Rr),
            ..opts
        };
        let (so_rr, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &rr_opts);
        println!(
            "S5 {label} Rr lane: ω² {:?} built {} converged {}",
            so_rr.omega2(),
            so_rr.iterations,
            so_rr.converged
        );
        assert_eq!(
            so_rr.omega2().len(),
            wanted.len(),
            "{label} Rr: every negative + the k lowest"
        );
        if so_rr.converged {
            for (got, want) in so_rr.omega2().iter().zip(&wanted) {
                assert!(
                    (got - want).abs() <= 1e-8 * scale,
                    "{label} Rr converged, ω² {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn paired_su2_hard_curvature_matches_the_dense_regularized_pencil() {
    use crate::solvers::hess::{
        PairedDavidsonOpts, PairedHook, PairedPrecond, assert_paired_wanted,
    };
    use crate::su2::hess::paired_lowest_davidson;
    let k = 3;
    let (x, h, f) = border_point(1e-6);
    let ctx_off = weingarten_operator(&h, &x);
    let (bl, br) = ctx_off.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a, b_off) = hessian_grams(&ctx_off, &basis);
    let p = Su2ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
    assert_eq!(
        deflated(&p),
        1,
        "fixture: the floor cuts the planted multiplet only"
    );
    let dd = dense_form(&basis, |v| {
        let mut w = v.clone();
        p.apply(&mut w);
        w
    });
    let b_cong = congruence(&dd, &b_off, n);
    let pm = |s: f64| -> Vec<f64> { a.iter().zip(&b_off).map(|(p, q)| p + s * q).collect() };
    let (raw_real, raw_cplx) = product_spectrum(&pm(1.0), &pm(-1.0), n);
    let pr = |s: f64| -> Vec<f64> { a.iter().zip(&b_cong).map(|(p, q)| p + s * q).collect() };
    let (real, cplx) = product_spectrum(&pr(1.0), &pr(-1.0), n);
    let (raw_radius, reg_radius) = (radius(&raw_real, &raw_cplx), radius(&real, &cplx));
    println!("paired S5: n = {n}, radius raw {raw_radius:.3e} vs regularized {reg_radius:.3e}");
    assert!(raw_radius >= 1e3 * reg_radius, "non-vacuity");
    let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let v0s = seeds(&ctx, &x, k + 2, 0xDEF5);
    let opts = PairedDavidsonOpts {
        tol: 1e-10,
        max_basis: 4 * n,
        max_subspace: n,
        ..PairedDavidsonOpts::default()
    };
    for hook in [PairedHook::Orthogonal, PairedHook::Oblique] {
        let out =
            paired_lowest_davidson(&ctx, &v0s, &[], k, PairedPrecond::Off, false, hook, &opts);
        assert_paired_wanted(
            &format!("paired S5 Hard {hook:?}"),
            &out,
            &a,
            &b_cong,
            n,
            k,
            1e-8,
        );
    }
}

fn su2_gauge_bits(gs: &[Su2BondGauge<f64>]) -> (Vec<u64>, Vec<u64>, f64) {
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
        for (_, v) in &g.sigma {
            sig.extend(v.iter().map(|x| x.to_bits()));
        }
        dropped += g.dropped_mass;
    }
    (rinv, sig, dropped)
}

#[test]
fn cr_su2_hard_builds_the_gauges_floored_at_f() {
    let (x, h, f) = border_point(1e-4);
    let plain = weingarten_operator(&h, &x);
    let cut: Vec<f64> = plain
        .bond_gauges()
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
        .filter(|&s| s < f)
        .collect();
    assert_eq!(
        cut.len(),
        1,
        "fixture: exactly the planted multiplet below the floor"
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
    let (r_p, s_p, m_p) = su2_gauge_bits(plain.bond_gauges());
    let (r_f, s_f, m_f) = su2_gauge_bits(floored.bond_gauges());
    let (r_h, s_h, m_h) = su2_gauge_bits(hard.bond_gauges());
    let (r_r, _, m_r) = su2_gauge_bits(hard_rev.bond_gauges());
    let (r_t, s_t, m_t) = su2_gauge_bits(tik.bond_gauges());
    let (r_tr, _, m_tr) = su2_gauge_bits(tik_rev.bond_gauges());
    let (r_tc, s_tc, m_tc) = su2_gauge_bits(tik.curvature_gauges());
    let (r_trc, _, m_trc) = su2_gauge_bits(tik_rev.curvature_gauges());
    let (bl, br) = plain.bases();
    let (r_d, _, m_d) = su2_gauge_bits(&bond_gauges(bl, br, FloorPolicy::Tikhonov(f)));
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
        "the raw Schmidt values are floor-independent"
    );
}

#[test]
fn cr_su2_off_and_identity_projector_are_bitwise_today() {
    let (x, h, f) = border_point(1e-4);
    let ctx_plain = weingarten_operator(&h, &x);
    let smin = raw_sigma_min(&ctx_plain);
    assert!(
        smin < f,
        "fixture: the planted multiplet carries the raw minimum"
    );
    assert_eq!(ctx_plain.curvature(), CurvatureReg::Off);
    let variants: Vec<(&str, CurvatureReg, Su2WeingartenCtx<'_, f64>)> = vec![
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
    let hard_out: Vec<Su2TtTangent<f64>> = tangents.iter().map(|xi| used.b_apply(xi)).collect();
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
            "non-vacuity: Tikhonov(f) and Hard(f) must give different b_apply"
        );
    }
    let used_tik_f = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(f));
    let tik_f_out: Vec<Su2TtTangent<f64>> =
        tangents.iter().map(|xi| used_tik_f.b_apply(xi)).collect();
    let used_tik_4f = used_tik_f.with_curvature(CurvatureReg::Tikhonov(4.0 * f));
    let fresh_tik_4f = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(4.0 * f));
    for (xi, b_f) in tangents.iter().zip(&tik_f_out) {
        let want = fresh_tik_4f.b_apply(xi);
        let moved = want.sub(b_f).norm() / b_f.norm();
        println!("SU(2) T-a2: |B(Tikhonov(4f)) − B(Tikhonov(f))|/|B(Tikhonov(f))| {moved:.3e}");
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
    assert!(
        moved > 1e-6,
        "non-vacuity: Hard(f) left b_apply unchanged ({moved:e})"
    );
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
        for (v, w) in vecs.iter().zip(&vec_p) {
            assert_bitwise(&v.x, &w.x, &format!("{label}: X"));
            assert_bitwise(&v.y, &w.y, &format!("{label}: Y"));
        }
    }
}

#[test]
fn cr_su2_auto_metric_resolves_to_ii_without_a_probe() {
    let (x, h, f) = border_point(1e-4);
    let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let k = 3;
    let v0s = seeds(&ctx, &x, k + 2, 0xA070);
    let (probed, verdict, _) = su2_select_rpa_metric(&ctx, &v0s[0], 200);
    println!("S6b control: su2_select_rpa_metric → {probed:?} (H^ii verdict {verdict:?})");
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
        "S6b probe: H^ii witness λ = {lambda:.6e} after {ii_applies} applies; dense \
         λ_min(A − P·B·P) = {lmin_reg:.6e}, λ_min(A − B_raw) = {lmin_raw:.6e}"
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
    assert_eq!(so_auto.side(), RpaMetric::Ii);
    assert!(
        flat_bits(&so_auto.omega2()) == flat_bits(&so_ii.omega2()),
        "Auto vs Ii: ω² bits"
    );
    assert_eq!(so_auto.applies, so_ii.applies);
}

fn plant_u() -> crate::su2::qn::Su2Sector {
    crate::su2::border_fixture::plant_sector(&base_state(), BOND, 1)
        .expect("fixture: the spin-½ plant sector")
}

#[test]
fn cr_su2_shifted_b_gram_is_pc_b_pc() {
    let (x, h, f) = border_point(1e-4);
    let sp = crate::su2::border_fixture::planted_raw_sigma(&x, BOND, 1);
    let two_k = 2usize;
    let h_out = generic_su2_h::<f64>(x.space.with_two_s(two_k));
    let ctx_off = weingarten_operator(&h, &x);
    let p = Su2ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
    assert_eq!(
        deflated(&p),
        1,
        "fixture: the floor cuts the planted multiplet only"
    );
    let dt = Su2ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), |s| {
        s * s / (s * s + f * f)
    });
    let sctx_off = Su2SectorCtx::new(&ctx_off, &h_out, two_k);
    let ctx_inner_hard = weingarten_operator(&h, &x).with_floor(FloorPolicy::Hard(f));
    let sctx_inner_hard = Su2SectorCtx::new(&ctx_inner_hard, &h_out, two_k);
    let basis = sctx_off.basis();
    let n = basis.len();
    assert_eq!(n, sctx_off.dim(), "premise: the basis spans T^(S=1)");
    let (a_off, b_off) = sctx_off.grams(&basis);
    let raw = max_abs(&b_off);
    println!("S3: n = {n}, f = {f:.3e}, σ_p = {sp:.3e}, max|B_raw| = {raw:.3e}");
    assert!(raw > 1e-6, "premise: a CURVED triplet channel");
    let apply_op = |op: &Su2ColumnDeflation<f64>, v: &Su2TtTangent<f64>| {
        let mut w = v.clone();
        op.apply(&mut w);
        w
    };
    for (label, reg, op) in [
        ("Hard", CurvatureReg::Hard(f), &p),
        ("Tikhonov", CurvatureReg::Tikhonov(f), &dt),
    ] {
        let ctx = weingarten_operator(&h, &x).with_curvature(reg);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
        let (a_reg, b_reg) = sctx.grams(&basis);
        let dd = dense_form(&basis, |v| apply_op(op, v));
        let want = congruence(&dd, &b_off, n);
        let regm = max_abs(&b_reg);
        let err = max_abs_diff(&b_reg, &want);
        let sym = asym(&b_reg, n);
        let moved = max_abs_diff(&b_reg, &b_off);
        println!(
            "S3 {label}: max|B_reg| {regm:.3e}, |Gram − Pc·B·Pc| {err:.2e}, symmetry defect \
             {sym:.2e}, moved {moved:.3e}"
        );
        assert!(
            err <= 1e-12 * raw.max(1.0),
            "{label}: Gram vs Pc·B·Pc {err:e}"
        );
        assert!(
            sym <= 1e-14 * regm + 1e-15 * raw,
            "{label}: symmetry defect {sym:e}"
        );
        assert!(
            flat_bits(&a_reg) == flat_bits(&a_off),
            "{label}: the A Gram must be bitwise the Off channel's (A is never regularized)"
        );
        assert!(
            moved >= 1e-6 * raw,
            "{label}: non-vacuity, the regularization did not move the channel's B ({moved:e})"
        );
        let inner = if label == "Hard" {
            &sctx_inner_hard
        } else {
            &sctx_off
        };
        let planted: Vec<&Su2TtTangent<f64>> = basis
            .iter()
            .filter(|xi| xi.sub(&apply_op(op, xi)).norm() >= 1e-3)
            .take(3)
            .collect();
        assert!(
            !planted.is_empty(),
            "[{label}] premise: basis vectors with planted content"
        );
        let (bl0, br0) = ctx_off.bases();
        let sf = Su2ShiftedFrames::new(bl0, br0, two_k);
        for xi in basis.iter().take(3).chain(planted) {
            let bt = match reg {
                CurvatureReg::Tikhonov(e) => su2_tik_composition(&ctx_off, e, xi, Some(&sf)),
                _ => apply_op(op, &inner.b_apply(&apply_op(op, xi))),
            };
            let a = sctx_off.a_apply(xi);
            assert_bitwise(
                &sctx.b_apply(xi),
                &bt,
                &format!(
                    "[{label}] b_apply = Pc.B_inner.Pc (Hard) / the P4d composition (Tikhonov)"
                ),
            );
            let b_offc = apply_op(op, &sctx_off.b_apply(&apply_op(op, xi)));
            let dev = sctx.b_apply(xi).sub(&b_offc).norm();
            assert!(
                dev <= 1e-12 * b_offc.norm().max(1.0),
                "[{label}] b_apply vs Pc.B_off.Pc: {dev:e} (the same operator)"
            );
            assert_bitwise(
                &sctx.a_apply(xi),
                &a,
                &format!("[{label}] a_apply untouched"),
            );
            assert_bitwise(
                &sctx.hpp_apply(xi),
                &a.add(&bt),
                &format!("[{label}] hpp = A + Pc.B.Pc"),
            );
            assert_bitwise(
                &sctx.hmm_apply(xi),
                &a.sub(&bt),
                &format!("[{label}] hmm = A - Pc.B.Pc"),
            );
        }
    }
    let f_raw = f / 2f64.sqrt();
    let below: Vec<f64> = ctx_off
        .bond_gauges()
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
        .filter(|&s| s < f_raw)
        .collect();
    assert_eq!(
        below,
        vec![sp],
        "the raw floor cuts exactly the planted value"
    );
    let ctx_floor = weingarten_operator(&h, &x).with_floor(FloorPolicy::Hard(f_raw));
    let (_, b_hard) = Su2SectorCtx::new(&ctx_floor, &h_out, two_k).grams(&basis);
    let ctx_reg = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let (_, b_reg) = Su2SectorCtx::new(&ctx_reg, &h_out, two_k).grams(&basis);
    let ctx_both = weingarten_operator(&h, &x)
        .with_floor(FloorPolicy::Hard(f_raw))
        .with_curvature(CurvatureReg::Hard(f));
    let (_, b_both) = Su2SectorCtx::new(&ctx_both, &h_out, two_k).grams(&basis);
    let pd = dense_form(&basis, |v| apply_op(&p, v));
    let want = congruence(&pd, &b_off, n);
    let regm = max_abs(&b_reg);
    let kept = congruence(&pd, &b_hard, n);
    let rel = max_abs_diff(&kept, &want) / regm;
    let rel_reg = max_abs_diff(&b_reg, &want) / regm;
    let both_bitwise = flat_bits(&b_both) == flat_bits(&b_reg);
    let asym_hard = asym(&b_hard, n) / max_abs(&b_hard);
    println!(
        "S3 E8: vs Pc·B_off·Pc, |Pc·B_hard·Pc − ·| {rel:.2e}, |Gram(B_reg) − ·| {rel_reg:.2e} \
         (relative to max|B_reg|); floor+curvature vs curvature bitwise {both_bitwise}; B_hard \
         symmetry defect {asym_hard:.2e}"
    );
    assert!(
        rel <= 1e-12,
        "kept-block identity on the spin channel {rel:e}"
    );
    assert!(
        rel_reg <= 1e-12,
        "Gram(B_reg) is not Pc·B·Pc of the Off channel ({rel_reg:e})"
    );
    assert!(
        both_bitwise,
        "with_floor(f/√2) + Hard(f) builds the gauges at max(f/√2, f) = f: Gram(B_reg), bitwise"
    );
}

#[test]
#[should_panic(expected = "not a cross-sector channel")]
fn cr_su2_sector_ctx_zero_flux_refusal_survives_a_regularized_context() {
    let (x, h, f) = border_point(1e-4);
    let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let _ = Su2SectorCtx::new(&ctx, &h, 0);
}

#[test]
fn cr_su2_sector_ctx_off_and_identity_regularization_are_bitwise() {
    let (x, h, f) = border_point(1e-4);
    let u_p = plant_u();
    let ctx_plain = weingarten_operator(&h, &x);
    let smin = raw_sigma_min(&ctx_plain);
    let h_tr = generic_su2_h::<f64>(x.space.with_two_s(2));
    let s_plain = Su2SectorCtx::new(&ctx_plain, &h_tr, 2);
    let basis = s_plain.basis();
    let (_, b_plain) = s_plain.grams(&basis);
    for (label, reg) in [
        ("Off", CurvatureReg::Off),
        ("Hard(σ_min/2)", CurvatureReg::Hard(0.5 * smin)),
    ] {
        let ctx = weingarten_operator(&h, &x).with_curvature(reg);
        let sctx = Su2SectorCtx::new(&ctx, &h_tr, 2);
        for xi in basis.iter().take(4) {
            assert_bitwise(
                &sctx.b_apply(xi),
                &s_plain.b_apply(xi),
                &format!("{label}: b_apply"),
            );
        }
        let (_, b) = sctx.grams(&basis);
        assert!(
            flat_bits(&b) == flat_bits(&b_plain),
            "{label}: grams().1 bits"
        );
    }
    let ctx_cut = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    assert_eq!(
        deflated(&Su2ColumnDeflation::from_gauges(ctx_plain.bond_gauges(), f)),
        1,
        "item 5's operator cuts the plant"
    );
    let s_cut_tr = Su2SectorCtx::new(&ctx_cut, &h_tr, 2);
    let moved = basis
        .iter()
        .map(|xi| s_cut_tr.b_apply(xi).sub(&s_plain.b_apply(xi)).norm())
        .fold(0.0f64, f64::max);
    assert!(
        moved > 1e-8,
        "non-vacuity: the triplet channel borrows the plant ({moved:e})"
    );
    let (bl, _) = ctx_plain.bases();
    for two_k in [2usize, 4, 6] {
        if crate::su2::weingarten::tangent_dim_grids(&x.space, &bl.ranks, two_k) == 0 {
            continue;
        }
        let h_out = generic_su2_h::<f64>(x.space.with_two_s(two_k));
        let bk = Su2SectorCtx::new(&ctx_plain, &h_out, two_k).basis();
        let out = bk[0].component(0).space;
        let rd = &bk[0].rank_col()[BOND];
        assert!(
            out.bond_sectors(BOND).into_iter().any(|t| {
                rd[t] > 0
                    && crate::su2::flux::channel_offset(
                        t,
                        u_p,
                        &bl.ranks[BOND],
                        two_k as i64,
                        out.n,
                    )
                    .is_some()
            }),
            "2κ {two_k}: every live channel borrows the spin-½ plant (why item 5 needs another)"
        );
    }
    let u0 = crate::su2::border_fixture::plant_sector(&base_state(), BOND, 0)
        .expect("fixture: the spin-0 plant sector");
    assert_eq!(
        u0.two_s, 0,
        "fixture: the item-5 plant is a singlet multiplet"
    );
    let x0p = with_small_schmidt(&base_state(), BOND, 0, 1e-4);
    let h0 = generic_su2_h::<f64>(x0p.space);
    let ctx0_plain = weingarten_operator(&h0, &x0p);
    let ctx0_cut = weingarten_operator(&h0, &x0p).with_curvature(CurvatureReg::Hard(f));
    let p0 = Su2ColumnDeflation::from_gauges(ctx0_plain.bond_gauges(), f);
    assert!(
        deflated(&p0) == 1 && p0.bonds()[BOND - 1].deflated == 1 && !p0.is_identity(),
        "item 5 PREMISE: a NON-identity operator cutting exactly the spin-0 plant at bond {BOND}"
    );
    let (bl0, _) = ctx0_plain.bases();
    let (mut unborrowed, mut borrowed_moved) = (0usize, 0usize);
    for two_k in [2usize, 4, 6] {
        if crate::su2::weingarten::tangent_dim_grids(&x0p.space, &bl0.ranks, two_k) == 0 {
            println!("S4 item 5, 2κ = {two_k}: empty channel");
            continue;
        }
        let h_out = generic_su2_h::<f64>(x0p.space.with_two_s(two_k));
        let s_p = Su2SectorCtx::new(&ctx0_plain, &h_out, two_k);
        let s_c = Su2SectorCtx::new(&ctx0_cut, &h_out, two_k);
        let bk = s_p.basis();
        let out = bk[0].component(0).space;
        let rd = &bk[0].rank_col()[BOND];
        let borrowed = out.bond_sectors(BOND).into_iter().any(|t| {
            rd[t] > 0
                && crate::su2::flux::channel_offset(t, u0, &bl0.ranks[BOND], two_k as i64, out.n)
                    .is_some()
        });
        let moved = bk
            .iter()
            .map(|xi| s_c.b_apply(xi).sub(&s_p.b_apply(xi)).norm())
            .fold(0.0f64, f64::max);
        println!(
            "S4 item 5, 2κ = {two_k}: dim {}, borrows the spin-0 plant: {borrowed}, b_apply moved \
             {moved:.3e}",
            bk.len()
        );
        if borrowed {
            if moved > 1e-8 {
                borrowed_moved += 1;
            }
            continue;
        }
        for xi in &bk {
            assert_bitwise(
                &s_c.b_apply(xi),
                &s_p.b_apply(xi),
                &format!("2κ {two_k}: a cut multiplet this channel never borrows"),
            );
        }
        let (_, b_c) = s_c.grams(&bk);
        let (_, b_p) = s_p.grams(&bk);
        assert!(
            flat_bits(&b_c) == flat_bits(&b_p),
            "2κ {two_k}: grams().1 bits with a cut multiplet this channel never borrows"
        );
        unborrowed += 1;
    }
    assert!(
        unborrowed >= 1,
        "item 5 COVERAGE: no live channel avoids the spin-0 plant (measured: the quintet does)"
    );
    assert!(
        borrowed_moved >= 1,
        "item 5 non-vacuity: the SAME operator moves a channel that borrows the spin-0 plant \
         (measured: the triplet, by 1.0e2)"
    );
}

#[test]
fn cr_su2_regularized_sector_lowest_matches_the_dense_regularized_pencil() {
    use crate::solvers::hess::{SectorOpts, SectorReadout};
    let (x, h, f) = border_point(1e-6);
    let two_k = 2usize;
    let h_out = generic_su2_h::<f64>(x.space.with_two_s(two_k));
    let ctx_off = weingarten_operator(&h, &x);
    let ctx = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let s_off = Su2SectorCtx::new(&ctx_off, &h_out, two_k);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
    let basis = sctx.basis();
    let n = basis.len();
    let radius_of = |a: &[f64], b: &[f64]| {
        let hrr: Vec<f64> = a.iter().zip(b).map(|(p, q)| p + q).collect();
        let hii: Vec<f64> = a.iter().zip(b).map(|(p, q)| p - q).collect();
        product_spectrum(&hrr, &hii, n)
    };
    let (a_raw, b_raw) = s_off.grams(&basis);
    let (raw_real, raw_cplx) = radius_of(&a_raw, &b_raw);
    let (a, b) = sctx.grams(&basis);
    let (real, cplx) = radius_of(&a, &b);
    let (rr, rg) = (radius(&raw_real, &raw_cplx), radius(&real, &cplx));
    let dense = sctx.symplectic_eigenvalues(&basis);
    let k = 3usize;
    let tol = 1e-9;
    for kernel in [RpaKernel::signed(RpaMetric::Ii), RpaKernel::Paired] {
        let opts = SectorOpts {
            readout: SectorReadout::Rpa,
            kernel,
            tol: 2.6e-10,
            max_iter: 8 * n,
            max_subspace: n,
            ..SectorOpts::default()
        };
        let resp = sctx.sector_lowest(k, &opts);
        let w = resp.omega2().expect("an Rpa readout").to_vec();
        let negatives: Vec<f64> = real.iter().copied().filter(|&z| z < 0.0).collect();
        let positives: Vec<f64> = real.iter().copied().filter(|&z| z >= 0.0).take(k).collect();
        let wanted: Vec<f64> = negatives.iter().chain(&positives).copied().collect();
        let scale = real.iter().fold(1.0f64, |m, z| m.max(z.abs()));
        println!(
            "S5 {kernel:?}: n = {n}, radius raw {rr:.3e} vs regularized {rg:.3e}; dense route {dense:?}; solve ω² \
             {w:?} ({} iterations, {} applies, converged {}); dense wanted {wanted:?} ({} complex)",
            resp.iterations,
            resp.applies,
            resp.converged,
            cplx.len()
        );
        assert!(
            rr >= 1e2 * rg,
            "non-vacuity: the regularization did not tame the channel"
        );
        assert!(
            resp.converged,
            "{kernel:?} residuals {:?}",
            resp.rpa().map(|r| r.residuals())
        );
        assert_eq!(
            w.len(),
            wanted.len(),
            "{kernel:?}: every negative + the k lowest"
        );
        for (got, want) in w.iter().zip(&wanted) {
            assert!(
                (got - want).abs() <= 10.0 * tol * scale,
                "{kernel:?}: ω² {got} vs the dense regularized pencil {want}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "CurvatureReg")]
fn cr_su2_with_curvature_refuses_a_nonpositive_scale() {
    let x = base_state();
    let h = generic_su2_h::<f64>(x.space);
    let _ = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Tikhonov(f64::NAN));
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

fn tik_w(e: f64) -> impl Fn(f64) -> f64 {
    move |s| s * s / (s * s + e * e)
}

fn su2_applied(op: &Su2ColumnDeflation<f64>, v: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
    let mut w = v.clone();
    op.apply(&mut w);
    w
}

fn su2_assemble_at(
    ctx: &Su2WeingartenCtx<'_, f64>,
    xi: &Su2TtTangent<f64>,
    gauges: &[Su2BondGauge<f64>],
    terms: Terms,
    sf: Option<&Su2ShiftedFrames<f64>>,
) -> Su2TtTangent<f64> {
    let e = ctx.envs();
    let zhi = Su2ImplicitHZ {
        h: ctx.h,
        ket: &ctx.base_l,
        ws: &e.ws,
        envs: Some((&e.pz_hi, &e.qz_hi)),
    };
    let zlo = Su2ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
    weingarten_assemble_sources_dressed(
        &ctx.base_l,
        &ctx.base_r,
        &[&zhi, &zlo],
        xi,
        gauges,
        terms,
        sf,
    )
}

fn su2_old_s3_at(
    ctx: &Su2WeingartenCtx<'_, f64>,
    xi: &Su2TtTangent<f64>,
    gauges: &[Su2BondGauge<f64>],
    sf: Option<&Su2ShiftedFrames<f64>>,
) -> Su2TtTangent<f64> {
    let e = ctx.envs();
    let zhi = Su2ImplicitHZ {
        h: ctx.h,
        ket: &ctx.base_l,
        ws: &e.ws,
        envs: Some((&e.pz_hi, &e.qz_hi)),
    };
    let zlo = Su2ExplicitZ::new(&e.z_lo).with_envs(&e.pz_lo, &e.qz_lo);
    su2_weingarten_right_chain_oracle(&ctx.base_l, &ctx.base_r, &[&zhi, &zlo], xi, gauges, sf)
}

fn su2_tik_composition(
    ctx: &Su2WeingartenCtx<'_, f64>,
    eps: f64,
    xi: &Su2TtTangent<f64>,
    sf: Option<&Su2ShiftedFrames<f64>>,
) -> Su2TtTangent<f64> {
    let (bl, br) = ctx.bases();
    let g = bond_gauges(bl, br, FloorPolicy::Tikhonov(eps));
    let d = Su2ColumnDeflation::from_gauges_weighted(&g, tik_w(eps));
    let mut first = su2_assemble_at(ctx, xi, &g, P4D_S1_S2, sf);
    d.apply(&mut first);
    let second = su2_assemble_at(ctx, &su2_applied(&d, xi), &g, P4D_S3, sf);
    first.add(&second)
}

fn su2_symmetrized_s1_form(
    ctx: &Su2WeingartenCtx<'_, f64>,
    op: &Su2ColumnDeflation<f64>,
    g_reg: &[Su2BondGauge<f64>],
    xis: &[Su2TtTangent<f64>],
    sf: Option<&Su2ShiftedFrames<f64>>,
) -> Vec<f64> {
    let n = xis.len();
    let s1: Vec<Su2TtTangent<f64>> = xis
        .iter()
        .map(|xi| su2_applied(op, &su2_assemble_at(ctx, xi, g_reg, P4D_S1, sf)))
        .collect();
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = xis[i].dot(&s1[j]) + s1[i].dot(&xis[j]);
        }
    }
    g
}

fn su2_form_from(xis: &[Su2TtTangent<f64>], images: &[Su2TtTangent<f64>]) -> Vec<f64> {
    let n = xis.len();
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] = xis[i].dot(&images[j]);
        }
    }
    g
}

fn su2_cluster_values() -> Vec<f64> {
    CLUSTER.iter().map(|m| m * CLUSTER_FLOOR).collect()
}

fn su2_leak_values() -> Vec<f64> {
    let mut v = su2_cluster_values();
    for x in v.iter_mut().take(3) {
        *x = 1e-10;
    }
    v
}

struct Su2P4dPoint {
    label: &'static str,
    x: Su2TtVector<f64>,
    h: Su2SparseHamiltonian<f64>,
    h_out: Option<Su2SparseHamiltonian<f64>>,
    two_k: usize,
    f: f64,
    census: Su2ClusterCensus,
}

impl Su2P4dPoint {
    fn new(values: &[f64], two_k: usize) -> Self {
        let (x, plant) = cluster_state(values);
        let label = if two_k == 0 { "flux 0" } else { "2κ = 2" };
        let census = assert_cluster_premise(
            &format!("SU(2) P4d [{label}]"),
            &x,
            &plant,
            CLUSTER_FLOOR,
            true,
        );
        let h = generic_su2_h::<f64>(x.space);
        let h_out = (two_k != 0).then(|| generic_su2_h::<f64>(x.space.with_two_s(two_k)));
        Su2P4dPoint {
            label,
            x,
            h,
            h_out,
            two_k,
            f: CLUSTER_FLOOR,
            census,
        }
    }

    fn ctx(&self, reg: CurvatureReg) -> Su2WeingartenCtx<'_, f64> {
        weingarten_operator(&self.h, &self.x).with_curvature(reg)
    }

    fn sf(&self, ctx: &Su2WeingartenCtx<'_, f64>) -> Option<Su2ShiftedFrames<f64>> {
        let (bl, br) = ctx.bases();
        (self.two_k != 0).then(|| Su2ShiftedFrames::new(bl, br, self.two_k))
    }

    fn tangents(
        &self,
        ctx: &Su2WeingartenCtx<'_, f64>,
        n: usize,
        seed: u64,
    ) -> Vec<Su2TtTangent<f64>> {
        let (bl, br) = ctx.bases();
        unit_tangents(bl, br, self.two_k, n, seed)
    }

    fn b<'s>(
        &'s self,
        ctx: &Su2WeingartenCtx<'s, f64>,
        xi: &Su2TtTangent<f64>,
    ) -> Su2TtTangent<f64> {
        match &self.h_out {
            None => ctx.b_apply(xi),
            Some(h_out) => Su2SectorCtx::new(ctx, h_out, self.two_k).b_apply(xi),
        }
    }
}

#[test]
fn cr_su2_p4d_off_b_is_the_pre_p4d_assembly() {
    for point in [
        Su2P4dPoint::new(&su2_cluster_values(), 0),
        Su2P4dPoint::new(&su2_cluster_values(), 2),
    ] {
        let ctx = point.ctx(CurvatureReg::Off);
        let sf = point.sf(&ctx);
        let g = ctx.bond_gauges();
        assert!(
            g.iter().all(|gg| gg.dropped_mass == 0.0),
            "[{}] premise: the plain gauges cut nothing",
            point.label
        );
        let mut worst = 0.0f64;
        for xi in point.tangents(&ctx, 4, 0x9D43_0001) {
            let b_new = point.b(&ctx, &xi);
            let b_old = su2_assemble_at(&ctx, &xi, g, P4D_S1_S2, sf.as_ref()).add(&su2_old_s3_at(
                &ctx,
                &xi,
                g,
                sf.as_ref(),
            ));
            assert!(b_old.norm() > 1e-8, "[{}] non-vacuity", point.label);
            worst = worst.max(b_new.sub(&b_old).norm() / b_old.norm());
        }
        println!(
            "SU(2) P4d T6 [{}]: |B_off − B_pre-P4d|/|B_pre-P4d| {worst:.2e}",
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
fn cr_su2_p4d_regularized_b_is_symmetric_at_a_cluster() {
    let mut rows = Vec::new();
    for point in [
        Su2P4dPoint::new(&su2_cluster_values(), 0),
        Su2P4dPoint::new(&su2_cluster_values(), 2),
    ] {
        let f = point.f;
        for (label, reg) in [
            ("Hard", CurvatureReg::Hard(f)),
            ("Tikhonov", CurvatureReg::Tikhonov(f)),
        ] {
            let ctx = point.ctx(reg);
            let xis = point.tangents(&ctx, 6, 0x9D43_0002);
            let images: Vec<Su2TtTangent<f64>> = xis.iter().map(|xi| point.b(&ctx, xi)).collect();
            let g = su2_form_from(&xis, &images);
            let scale = max_abs(&g);
            let rel = asym(&g, xis.len()) / scale;
            println!(
                "SU(2) P4d T-sym [{} {label}]: max|G| {scale:.3e}, relative asymmetry {rel:.2e}",
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
fn cr_su2_p4d_regularized_b_is_the_beside_congruence_at_a_cluster() {
    let mut rows = Vec::new();
    for point in [
        Su2P4dPoint::new(&su2_cluster_values(), 0),
        Su2P4dPoint::new(&su2_cluster_values(), 2),
    ] {
        let f = point.f;
        let ctx_off = point.ctx(CurvatureReg::Off);
        let p = Su2ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
        let dt = Su2ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), tik_w(f));
        let xis = point.tangents(&ctx_off, 4, 0x9D43_0003);
        for (label, reg, op) in [
            ("Hard", CurvatureReg::Hard(f), &p),
            ("Tikhonov", CurvatureReg::Tikhonov(f), &dt),
        ] {
            let ctx = point.ctx(reg);
            let (mut worst, mut moved) = (0.0f64, f64::INFINITY);
            for xi in &xis {
                let want = su2_applied(op, &point.b(&ctx_off, &su2_applied(op, xi)));
                let b = point.b(&ctx_off, xi);
                worst = worst.max(point.b(&ctx, xi).sub(&want).norm() / want.norm());
                moved = moved.min(want.sub(&b).norm() / b.norm());
            }
            println!(
                "SU(2) P4d T8 [{} {label}]: |B_reg − D·B·D|/|D·B·D| {worst:.2e}; the congruence \
                 moves B by ≥ {moved:.3e}",
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
fn cr_su2_p4d_no_beside_leakage_at_a_tiny_planted_value() {
    let mut rows = Vec::new();
    for point in [
        Su2P4dPoint::new(&su2_leak_values(), 0),
        Su2P4dPoint::new(&su2_leak_values(), 2),
    ] {
        let tiny = point
            .census
            .cut
            .iter()
            .map(|c| c.2)
            .fold(f64::INFINITY, f64::min);
        println!(
            "SU(2) P4d leakage [{}]: smallest planted raw value {tiny:.3e}",
            point.label
        );
        assert!(
            tiny <= 1e-9,
            "[{}] premise: a planted value at ~1e-10 survives the plant ({tiny:e})",
            point.label
        );
        let f = point.f;
        let ctx_off = point.ctx(CurvatureReg::Off);
        let sf = point.sf(&ctx_off);
        let (bl, br) = ctx_off.bases();
        let p = Su2ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
        let dt = Su2ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), tik_w(f));
        let g_floor = bond_gauges(bl, br, FloorPolicy::Hard(f));
        let g_damp = bond_gauges(bl, br, FloorPolicy::Tikhonov(f));
        let xis = point.tangents(&ctx_off, 5, 0x9D43_0004);
        let n = xis.len();
        for (label, reg, op, g_reg) in [
            ("Hard", CurvatureReg::Hard(f), &p, &g_floor),
            ("Tikhonov", CurvatureReg::Tikhonov(f), &dt, &g_damp),
        ] {
            let ctx = point.ctx(reg);
            let images: Vec<Su2TtTangent<f64>> = xis.iter().map(|xi| point.b(&ctx, xi)).collect();
            let g_ctx = su2_form_from(&xis, &images);
            let g_or = su2_symmetrized_s1_form(&ctx_off, op, g_reg, &xis, sf.as_ref());
            let scale = max_abs(&g_or);
            let err = max_abs_diff(&g_ctx, &g_or) / scale;
            let sym = asym(&g_ctx, n) / scale;
            let beside: Vec<Su2TtTangent<f64>> = xis
                .iter()
                .map(|xi| su2_applied(op, &point.b(&ctx_off, &su2_applied(op, xi))))
                .collect();
            let err_beside = max_abs_diff(&su2_form_from(&xis, &beside), &g_or) / scale;
            let cut_cols = images
                .iter()
                .map(|b| b.sub(&su2_applied(&p, b)).norm() / b.norm())
                .fold(0.0f64, f64::max);
            println!(
                "SU(2) P4d leakage [{} {label}]: |G − oracle| {err:.2e}, asymmetry {sym:.2e} (scale \
                 {scale:.3e}); anti-oracle: the beside form misses by {err_beside:.2e}; \
                 |(I − Pc)B ξ|/|B ξ| {cut_cols:.2e}",
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
                "[{pl} Tikhonov] anti-oracle: the beside form shows no residue ({err_beside:e})"
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
fn cr_su2_p4d_kept_block_identity_at_a_cluster() {
    let mut rows = Vec::new();
    for two_k in [0usize, 2] {
        let point = Su2P4dPoint::new(&su2_cluster_values(), two_k);
        let f = point.f;
        let ctx_off = point.ctx(CurvatureReg::Off);
        let ctx_floor = weingarten_operator(&point.h, &point.x).with_floor(FloorPolicy::Hard(f));
        let ctx_reg = point.ctx(CurvatureReg::Hard(f));
        let p = Su2ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
        let (basis, b_off, b_floor, b_reg) = match &point.h_out {
            None => {
                let (bl, br) = ctx_off.bases();
                let basis = tangent_basis(bl, br, true);
                let (_, b_off) = hessian_grams(&ctx_off, &basis);
                let (_, b_floor) = hessian_grams(&ctx_floor, &basis);
                let (_, b_reg) = hessian_grams(&ctx_reg, &basis);
                (basis, b_off, b_floor, b_reg)
            }
            Some(h_out) => {
                let s_off = Su2SectorCtx::new(&ctx_off, h_out, two_k);
                let basis = s_off.basis();
                let (_, b_off) = s_off.grams(&basis);
                let (_, b_floor) = Su2SectorCtx::new(&ctx_floor, h_out, two_k).grams(&basis);
                let (_, b_reg) = Su2SectorCtx::new(&ctx_reg, h_out, two_k).grams(&basis);
                (basis, b_off, b_floor, b_reg)
            }
        };
        let n = basis.len();
        let pd = dense_form(&basis, |v| su2_applied(&p, v));
        let want = congruence(&pd, &b_off, n);
        let regm = max_abs(&b_reg);
        let kept = max_abs_diff(&congruence(&pd, &b_floor, n), &want) / regm;
        let reg_err = max_abs_diff(&b_reg, &want) / regm;
        let reg_sym = asym(&b_reg, n) / regm;
        let fs = max_abs(&b_floor);
        let not_cong = max_abs_diff(&b_floor, &congruence(&pd, &b_floor, n)) / fs;
        let floor_sym = asym(&b_floor, n) / fs;
        println!(
            "SU(2) P4d E8 cluster [{}]: n = {n}, max|B_off| {:.3e}, max|B_reg| {regm:.3e}; \
             |Pc·B_f·Pc − Pc·B·Pc| {kept:.2e}, |Gram(B_reg) − Pc·B·Pc| {reg_err:.2e}, Gram(B_reg) \
             asymmetry {reg_sym:.2e}; B_f: |B_f − Pc·B_f·Pc| {not_cong:.2e}, asymmetry \
             {floor_sym:.2e}",
            point.label,
            max_abs(&b_off)
        );
        rows.push((point.label, kept, reg_err, reg_sym, not_cong));
    }
    for (pl, kept, reg_err, reg_sym, not_cong) in rows {
        assert!(
            not_cong >= 1e-8,
            "[{pl}] non-vacuity: the in-formula floor alone is already the congruence ({not_cong:e})"
        );
        assert!(
            kept <= 1e-12,
            "[{pl}] kept-block identity at the cluster: Pc·B_f·Pc is not Pc·B·Pc ({kept:e})"
        );
        assert!(
            reg_err <= 1e-12,
            "[{pl}] Gram(B_reg) is not Pc·B·Pc of the plain Gram ({reg_err:e})"
        );
        assert!(
            reg_sym <= 1e-13,
            "[{pl}] Gram(B_reg) is not symmetric ({reg_sym:e})"
        );
    }
}
