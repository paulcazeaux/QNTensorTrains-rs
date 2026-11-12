// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::core::deflate::ColumnDeflation;
use crate::core::deflate::oracle::{asym, congruence, max_abs, max_abs_diff};
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::tangent::TtTangent;
use crate::core::weingarten::FloorPolicy;
use crate::solvers::hess::CurvatureReg;
use crate::su2::border_fixture::{
    BOND, base_state, dense_form, plant_sector, planted_raw_sigma, w_hat_min, with_small_schmidt,
};
use crate::su2::deflate::Su2ColumnDeflation;
use crate::su2::expand::expand;
use crate::su2::hess::{hessian_grams, weingarten_operator};
use crate::su2::qn::Su2Space;
use crate::su2::riemannian::tangent_bases;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use crate::su2::weingarten::{Su2BondGauge, bond_gauges, tangent_basis, tangent_dim};
use crate::test_util::{generic_su2_h, randomized_su2_state};

fn gauges_of(x: &Su2TtVector<f64>) -> (Su2TtVector<f64>, Su2TtVector<f64>, Vec<Su2BondGauge<f64>>) {
    let (bl, br) = tangent_bases(x);
    let g = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    (bl, br, g)
}

fn deflated(p: &Su2ColumnDeflation<f64>) -> usize {
    p.bonds().iter().map(|r| r.deflated).sum()
}

fn raw_sigma_min(gauges: &[Su2BondGauge<f64>]) -> f64 {
    gauges
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
        .fold(f64::INFINITY, f64::min)
}

fn applied(p: &Su2ColumnDeflation<f64>, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
    let mut v = xi.clone();
    p.apply(&mut v);
    v
}

fn su2_seeds(
    x: &Su2TtVector<f64>,
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    n: usize,
    seed: u64,
) -> Vec<Su2TtTangent<f64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let d: Su2TtVector<f64> = crate::su2::vector::tt_randn(x.space, &x.ranks, &mut rng);
            let mut t = Su2TtTangent::project(bl, br, &d);
            let nrm = t.norm();
            t.scale_mut(1.0 / nrm);
            t
        })
        .collect()
}

#[test]
fn cr_su2_projector_is_orthogonal_in_the_centerweighted_metric() {
    let x = with_small_schmidt(&base_state(), BOND, 1, 1e-4);
    let sp = planted_raw_sigma(&x, BOND, 1);
    let (bl, br, gauges) = gauges_of(&x);
    let p = Su2ColumnDeflation::from_gauges(&gauges, 1.6 * sp);
    assert!(
        !p.is_identity(),
        "non-vacuity: the planted multiplet is below the floor"
    );
    assert_eq!(deflated(&p), 1, "exactly the planted multiplet");
    let basis = tangent_basis(&bl, &br, false);
    let n = basis.len();
    assert_eq!(n, tangent_dim(&bl));
    let m = dense_form(&basis, |v| applied(&p, v));
    let (mut idem, mut tr) = (0.0f64, 0.0f64);
    for i in 0..n {
        tr += m[i * n + i];
        for j in 0..n {
            let mm: f64 = (0..n).map(|q| m[i * n + q] * m[q * n + j]).sum();
            idem = idem.max((mm - m[i * n + j]).abs());
        }
    }
    let sym = asym(&m, n);
    let mut leak = 0.0f64;
    for (j, b) in basis.iter().enumerate() {
        let mut r = applied(&p, b);
        for (i, bi) in basis.iter().enumerate() {
            r.axpy(-m[i * n + j], bi);
        }
        leak = leak.max(r.norm());
    }
    let removed = p.removed_dim(&basis[0]);
    let rad0 = Su2TtTangent::project(&bl, &br, &bl);
    let mut dr = applied(&p, &rad0);
    dr.axpy(-1.0, &rad0);
    println!(
        "S1: n = {n}, σ_p = {sp:.4e}, removed {removed}, trace {tr:.12}, sym {sym:.1e}, idem \
         {idem:.1e}, leak {leak:.1e}, radial moved {:.1e}",
        dr.norm() / rad0.norm()
    );
    assert!(removed > 0, "the planted sector has gauge-complement room");
    assert!(
        sym <= 1e-12 && idem <= 1e-12 && leak <= 1e-12,
        "sym {sym:e} idem {idem:e} leak {leak:e}"
    );
    assert!(
        (tr - (n - removed) as f64).abs() <= 1e-9,
        "rank {tr} vs the grid count {}",
        n - removed
    );
    assert!(dr.norm() <= 1e-14 * rad0.norm(), "the radial mode moved");
    let id = Su2ColumnDeflation::from_gauges(&gauges, 0.5 * raw_sigma_min(&gauges));
    assert!(
        id.is_identity(),
        "a floor below every raw σ deflates nothing"
    );
    let v = &basis[n / 2];
    let pv = applied(&id, v);
    for i in 0..v.num_coeffs() {
        assert_eq!(
            pv.coeff(i).to_bits(),
            v.coeff(i).to_bits(),
            "identity is bitwise"
        );
    }
}

fn units_pin(x: &Su2TtVector<f64>, what: &str) {
    assert_eq!(x.space.two_s, 0, "the units pin is a singlet statement");
    let (_, _, g_su2) = gauges_of(x);
    let x_ab = expand(x, 0);
    let (al, ar) = crate::solvers::riemannian::tangent_bases(&x_ab);
    let g_ab = crate::core::weingarten::bond_gauges(&al, &ar, FloorPolicy::default_hard());
    assert_eq!(g_su2.len(), g_ab.len());
    for (gs, ga) in g_su2.iter().zip(&g_ab) {
        let mut want: Vec<f64> = gs
            .sigma
            .iter()
            .flat_map(|(s, v)| {
                v.iter()
                    .flat_map(move |&x| std::iter::repeat_n(x, s.two_s + 1))
            })
            .collect();
        let mut got: Vec<f64> = ga
            .sigma
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        want.sort_by(f64::total_cmp);
        got.sort_by(f64::total_cmp);
        assert_eq!(
            got.len(),
            want.len(),
            "{what} bond {}: multiset sizes",
            gs.k
        );
        let err = got
            .iter()
            .zip(&want)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err <= 1e-12,
            "{what} bond {}: abelian σ vs repeated raw σ off by {err:e}",
            gs.k
        );
    }
}

#[test]
fn cr_su2_floor_is_in_raw_sigma_units() {
    let x0 = base_state();
    assert!(
        plant_sector(&x0, BOND, 2).is_some(),
        "fixture: the seed was chosen so a spin-1 sector with room exists at bond {BOND}"
    );
    for (two_s, w_keep) in [(1usize, 1.2), (2, 1.6)] {
        let x = with_small_schmidt(&x0, BOND, two_s, 1e-4);
        let sp = planted_raw_sigma(&x, BOND, two_s);
        let w_hat = ((two_s + 1) as f64).sqrt() * sp;
        assert!(
            w_keep * sp < w_hat,
            "premise: ŵ would keep the plant at {w_keep}σ_p"
        );
        let (_, _, gauges) = gauges_of(&x);
        let kept = Su2ColumnDeflation::from_gauges(&gauges, 0.9 * sp);
        let gone = Su2ColumnDeflation::from_gauges(&gauges, 1.1 * sp);
        let gone_w = Su2ColumnDeflation::from_gauges(&gauges, w_keep * sp);
        let rep = gone
            .bonds()
            .iter()
            .find(|r| r.bond == BOND)
            .copied()
            .expect("a report per interior bond");
        println!(
            "S2 2s = {two_s}: σ_p {sp:.4e}, ŵ {w_hat:.4e}; Hard(0.9σ_p) identity {}, Hard(1.1σ_p) \
             deflated {}, Hard({w_keep}σ_p) deflated {} (bond {BOND} report {rep:?})",
            kept.is_identity(),
            deflated(&gone),
            deflated(&gone_w)
        );
        assert!(
            kept.is_identity(),
            "2s = {two_s}: σ_p ≥ 0.9σ_p must be kept"
        );
        assert_eq!(
            deflated(&gone),
            1,
            "2s = {two_s}: exactly the planted multiplet below 1.1σ_p"
        );
        assert_eq!(
            deflated(&gone_w),
            1,
            "2s = {two_s}: a raw comparison cuts at {w_keep}σ_p, where ŵ = {w_hat:e} would keep"
        );
        assert_eq!(rep.deflated, 1);
        assert!(
            (rep.largest_deflated - sp).abs() <= 1e-12 * sp,
            "the report is in raw σ units: {} vs {sp:e}",
            rep.largest_deflated
        );
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5D24);
    let mut x4: Su2TtVector<f64> = randomized_su2_state(Su2Space::new(4, 4, 0), 1, 2, &mut rng);
    x4.scale_mut(1.0 / x4.norm());
    units_pin(&x4, "d = 4 random singlet");
    units_pin(
        &with_small_schmidt(&x0, BOND, 1, 1e-4),
        "d = 6 planted fixture",
    );
}

fn embed(
    xi: &Su2TtVector<f64>,
    al: &crate::vector::TtVector<f64>,
    ar: &crate::vector::TtVector<f64>,
    n_xi: f64,
) -> TtTangent<f64> {
    let y = expand(xi, 0);
    let t = TtTangent::project(al, ar, &y);
    assert!(
        (t.norm() - n_xi).abs() <= 1e-12 * n_xi.max(1.0),
        "embedding premise: isometry"
    );
    let back = &t.to_vector() - &y;
    assert!(
        back.norm_lr(false) <= 1e-10 * y.norm_lr(false).max(1.0),
        "embedding premise: expand(ι(ξ), 0) is already tangent at x_ab"
    );
    t
}

#[test]
fn cr_su2_projector_is_the_spin_zero_block_of_the_expanded_abelian_projector() {
    let x = with_small_schmidt(&base_state(), BOND, 1, 1e-4);
    let sp = planted_raw_sigma(&x, BOND, 1);
    let (bl, br, gauges) = gauges_of(&x);
    let x_ab = expand(&x, 0);
    let (al, ar) = crate::solvers::riemannian::tangent_bases(&x_ab);
    let g_ab = crate::core::weingarten::bond_gauges(&al, &ar, FloorPolicy::default_hard());
    let ab_deflated =
        |p: &ColumnDeflation<f64>| -> usize { p.bonds().iter().map(|r| r.deflated).sum() };
    let emb = |xi: &Su2TtTangent<f64>| embed(&xi.to_vector(), &al, &ar, xi.norm());
    let conv = 1.0 / 2f64.sqrt();
    let tangents = su2_seeds(&x, &bl, &br, 4, 0x5E3A);

    let p_su2 = Su2ColumnDeflation::from_gauges(&gauges, 1.6 * sp);
    let p_ab = ColumnDeflation::from_gauges(&g_ab, 1.6 * sp);
    assert_eq!(deflated(&p_su2), 1, "premise: one multiplet");
    assert_eq!(
        ab_deflated(&p_ab),
        2,
        "premise: its 2s_p + 1 = 2 abelian members"
    );
    let mut worst = 0.0f64;
    for xi in &tangents {
        let mut d = emb(&applied(&p_su2, xi));
        let mut r = emb(xi);
        p_ab.apply(&mut r);
        d.axpy(-1.0, &r);
        worst = worst.max(d.norm() / xi.norm());
    }
    let p_su2_12 = Su2ColumnDeflation::from_gauges(&gauges, 1.2 * sp);
    let p_ab_same = ColumnDeflation::from_gauges(&g_ab, 1.2 * sp);
    let p_ab_conv = ColumnDeflation::from_gauges(&g_ab, 1.2 * sp * conv);
    assert_eq!(deflated(&p_su2_12), 1, "a raw 1.2σ_p cuts the plant");
    assert_eq!(
        ab_deflated(&p_ab_same),
        2,
        "the same abelian floor cuts the members"
    );
    assert!(
        p_ab_conv.is_identity(),
        "the ŵ-converted floor keeps the members"
    );
    let mut xi_pl = tangents[0].clone();
    xi_pl.axpy(-1.0, &applied(&p_su2, &tangents[0]));
    let npl = xi_pl.norm();
    assert!(
        npl > 1e-3,
        "non-vacuity: the seed carries planted-column weight ({npl:e})"
    );
    let mut worst_12 = 0.0f64;
    for xi in &tangents {
        let mut d = emb(&applied(&p_su2_12, xi));
        let mut r = emb(xi);
        p_ab_same.apply(&mut r);
        d.axpy(-1.0, &r);
        worst_12 = worst_12.max(d.norm() / xi.norm());
    }
    let mut bad = emb(&applied(&p_su2_12, &xi_pl));
    let mut r = emb(&xi_pl);
    p_ab_conv.apply(&mut r);
    bad.axpy(-1.0, &r);
    let et = 3e-2;
    let d_ab = ColumnDeflation::from_gauges_weighted(&g_ab, |s| s * s / (s * s + et * et));
    let d_su2 = Su2ColumnDeflation::from_gauges_weighted(&gauges, |s| s * s / (s * s + et * et));
    let mut worst_t = 0.0f64;
    for xi in &tangents {
        let mut d = emb(&applied(&d_su2, xi));
        let mut r = emb(xi);
        d_ab.apply(&mut r);
        d.axpy(-1.0, &r);
        worst_t = worst_t.max(d.norm() / xi.norm());
    }
    println!(
        "S3: σ_p {sp:.4e}; |emb P − P_ab emb| {worst:.2e}; at 1.2σ_p same floor {worst_12:.2e} vs \
         ŵ-converted {:.2e} (‖ξ_pl‖ {npl:.3e}); Tikhonov one-ε {worst_t:.2e}",
        bad.norm()
    );
    assert!(
        worst <= 1e-12,
        "Hard: the SU(2) projector is not the spin-0 block ({worst:e})"
    );
    assert!(
        worst_12 <= 1e-12,
        "1.2σ_p with the same floor: {worst_12:e}"
    );
    assert!(
        bad.norm() >= 1e-3 * npl,
        "anti-oracle: the ŵ-converted floor must break it ({:e})",
        bad.norm()
    );
    assert!(worst_t <= 1e-12, "Tikhonov: one-ε weights {worst_t:e}");
}

#[test]
fn cr_su2_b_gram_is_pd_b_pd() {
    let x0 = base_state();
    let f = (w_hat_min(&x0) * 1e-4).sqrt();
    let x = with_small_schmidt(&x0, BOND, 1, 1e-4);
    let sp = planted_raw_sigma(&x, BOND, 1);
    let h = generic_su2_h::<f64>(x.space);
    let ctx_off = weingarten_operator(&h, &x);
    let (bl, br) = ctx_off.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a_off, b_off) = hessian_grams(&ctx_off, &basis);
    let raw = max_abs(&b_off);
    let p = Su2ColumnDeflation::from_gauges(ctx_off.bond_gauges(), f);
    assert_eq!(
        deflated(&p),
        1,
        "fixture: the floor cuts the planted multiplet only"
    );
    let d = Su2ColumnDeflation::from_gauges_weighted(ctx_off.bond_gauges(), |s| {
        s * s / (s * s + f * f)
    });
    for (label, reg, op) in [
        ("Hard", CurvatureReg::Hard(f), &p),
        ("Tikhonov", CurvatureReg::Tikhonov(f), &d),
    ] {
        let ctx = weingarten_operator(&h, &x).with_curvature(reg);
        assert_eq!(ctx.curvature(), reg);
        let (a_reg, b_reg) = hessian_grams(&ctx, &basis);
        let dd = dense_form(&basis, |v| applied(op, v));
        let want = congruence(&dd, &b_off, n);
        let regm = max_abs(&b_reg);
        let err = max_abs_diff(&b_reg, &want);
        let sym = asym(&b_reg, n);
        println!(
            "S4 {label}: n = {n}, max|B| raw {raw:.3e} reg {regm:.3e}, |Gram − D·B·D| {err:.2e}, sym {:.2e}",
            sym / regm
        );
        assert!(
            err <= 1e-12 * raw.max(1.0),
            "{label}: Gram vs D·B·D {err:e}"
        );
        let sym_bound = match reg {
            CurvatureReg::Tikhonov(_) => 1e-14 * regm + 1e-15 * raw,
            _ => 1e-14 * regm + 1e-16 * raw,
        };
        println!("S4 {label}: symmetry defect {sym:.3e} (absolute) vs bound {sym_bound:.3e}");
        assert!(sym <= sym_bound, "{label}: symmetry defect {sym:e}");
        assert!(
            a_reg
                .iter()
                .zip(&a_off)
                .all(|(p, q)| p.to_bits() == q.to_bits()),
            "{label}: A Gram must be bitwise the plain one"
        );
        assert!(
            raw >= 1e2 * regm,
            "{label}: non-vacuity, raw {raw:e} vs regularized {regm:e}"
        );
    }
    let below: Vec<f64> = ctx_off
        .bond_gauges()
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
        .filter(|&s| s < f)
        .collect();
    assert_eq!(
        below,
        vec![sp],
        "the raw floor cuts exactly the planted value"
    );
    let ctx_floor = weingarten_operator(&h, &x).with_floor(FloorPolicy::Hard(f));
    let (_, b_hard) = hessian_grams(&ctx_floor, &basis);
    let ctx_reg = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let (_, b_reg) = hessian_grams(&ctx_reg, &basis);
    let pd = dense_form(&basis, |v| applied(&p, v));
    let off_kept = congruence(&pd, &b_off, n);
    let tol = 1e-12 * max_abs(&b_reg);
    let e_hard = max_abs_diff(&congruence(&pd, &b_hard, n), &off_kept);
    let e_reg = max_abs_diff(&b_reg, &off_kept);
    let asym_hard = asym(&b_hard, n) / max_abs(&b_hard);
    println!(
        "S4 E8: |P·B_hard·P − P·B_off·P| {e_hard:.2e}, |Gram(B_reg) − P·B_off·P| {e_reg:.2e} \
         (bound 1e-12·max|B_reg| = {tol:.2e}); B_hard symmetry defect {asym_hard:.2e}"
    );
    assert!(
        e_hard <= tol,
        "kept-block identity: the in-formula floor's P·B_hard·P vs P·B_off·P {e_hard:e}"
    );
    assert!(
        e_reg <= tol,
        "kept-block identity: Gram(B_reg) vs P·B_off·P {e_reg:e}"
    );
    assert!(dense_self_adjoint_smallest::<f64>(&pd, n, 0).0[0] >= -1e-12);
}

fn col_norm(c: &faer::Mat<f64>, v: &[f64]) -> f64 {
    (0..c.nrows())
        .map(|i| {
            (0..c.ncols())
                .map(|j| c[(i, j)] * v[j])
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn cr_su2_projector_reads_the_right_singular_vectors_not_the_left() {
    use crate::core::factor::svd_uvt_svals;
    use faer::Mat;
    let x0 = base_state();
    let f = (w_hat_min(&x0) * 1e-4).sqrt();
    let t = plant_sector(&x0, BOND, 1).expect("fixture: a spin-½ plant sector");
    let x = with_small_schmidt(&x0, BOND, 1, 1e-4);
    let (bl, br, mut gauges) = gauges_of(&x);
    let g = &mut gauges[BOND - 1];
    let (_, sv, _) = svd_uvt_svals(g.r.block(t).expect("fixture: the planted block"));
    let n = sv.len();
    assert!(n >= 2, "fixture: the planted sector has rank ≥ 2");
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x05EC_72B5);
    let mut gauss = |_: usize, _: usize| rand::RngExt::sample(&mut rng, rand_distr::StandardNormal);
    let g1 = Mat::<f64>::from_fn(n, n, &mut gauss);
    let g2 = Mat::<f64>::from_fn(n, n, &mut gauss);
    let (wq, _, _) = svd_uvt_svals(g1.as_ref());
    let (_, _, vqt) = svd_uvt_svals(g2.as_ref());
    let m = Mat::<f64>::from_fn(n, n, |i, j| {
        (0..n)
            .map(|q| wq[(i, q)] * sv[q] * vqt[(q, j)])
            .sum::<f64>()
    });
    g.r.block_mut(t)
        .expect("the planted block")
        .copy_from(m.as_ref());
    let w_n: Vec<f64> = (0..n).map(|i| wq[(i, n - 1)]).collect();
    let v_n: Vec<f64> = (0..n).map(|j| vqt[(n - 1, j)]).collect();
    let overlap = w_n.iter().zip(&v_n).map(|(a, b)| a * b).sum::<f64>().abs();
    let p = Su2ColumnDeflation::from_gauges(&gauges, f);
    assert_eq!(
        deflated(&p),
        1,
        "the floor still cuts exactly the planted multiplet"
    );
    let (mut worst_v, mut least_w) = (0.0f64, f64::INFINITY);
    for xi in su2_seeds(&x, &bl, &br, 3, 0x05EC_72B6) {
        let c0 = xi.components[BOND - 1].vertical_unfolding(t);
        let scale = c0.norm_l2();
        assert!(
            col_norm(&c0, &v_n) >= 1e-3 * scale,
            "premise: the seed has weight along v_n"
        );
        let y = applied(&p, &xi);
        let c = y.components[BOND - 1].vertical_unfolding(t);
        worst_v = worst_v.max(col_norm(&c, &v_n) / scale);
        least_w = least_w.min(col_norm(&c, &w_n) / scale);
    }
    println!(
        "T-b SU(2): sector {t:?}, n = {n}, |⟨w_n, v_n⟩| = {overlap:.3}, after apply max \
         ‖c·v_n‖/‖c‖ = {worst_v:.2e}, min ‖c·w_n‖/‖c‖ = {least_w:.2e}"
    );
    assert!(
        overlap <= 0.9,
        "premise: W and V differ (|⟨w_n, v_n⟩| = {overlap})"
    );
    assert!(
        worst_v <= 1e-13,
        "the touched slot keeps weight along the RIGHT singular vector ({worst_v:e})"
    );
    assert!(
        least_w >= 1e-3,
        "non-vacuity: the slot lost its weight along the LEFT vector too ({least_w:e}) — a \
         W-built projector would look like this"
    );
}

#[test]
fn cr_su2_floor_is_in_raw_sigma_units_at_a_doublet() {
    let space = Su2Space::new(6, 5, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x00D0_B1E7);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    x.scale_mut(1.0 / x.norm());
    x.sigma_floor_trim_mut(3e-2);
    x.scale_mut(1.0 / x.norm());
    let (_, _, gauges) = gauges_of(&x);
    let smin = raw_sigma_min(&gauges);
    let at = Su2ColumnDeflation::from_gauges(&gauges, smin);
    let above = Su2ColumnDeflation::from_gauges(&gauges, smin * (1.0 + 1e-9));
    let largest = above
        .bonds()
        .iter()
        .map(|r| r.largest_deflated)
        .fold(0.0f64, f64::max);
    let w_of = |two_s: usize, s: f64| crate::su2::centerweight(two_s, 1).sqrt() * s;
    let mults: Vec<(usize, usize, f64)> = gauges
        .iter()
        .flat_map(|g| {
            g.sigma
                .iter()
                .flat_map(move |(s, v)| v.iter().map(move |&sig| (g.k, s.two_s, sig)))
        })
        .collect();
    let carrier = mults
        .iter()
        .copied()
        .min_by(|a, b| a.2.total_cmp(&b.2))
        .expect("a doublet with Schmidt values");
    println!(
        "T-c: 2S = 1, raw σ_min = {smin:.6e} (bond {}, 2s = {}, ŵ {:.6e}); Hard(σ_min) identity \
         {}, Hard(σ_min·(1+1e-9)) cuts {} (largest_deflated {largest:.6e})",
        carrier.0,
        carrier.1,
        w_of(carrier.1, carrier.2),
        at.is_identity(),
        deflated(&above)
    );
    assert!(
        at.is_identity(),
        "strict floor: Hard(σ_min) keeps every multiplet"
    );
    assert!(
        deflated(&above) >= 1,
        "a floor a hair above σ_min must cut its multiplet"
    );
    assert_eq!(
        largest.to_bits(),
        smin.to_bits(),
        "the report is in raw σ units, bit for bit"
    );
    let counts = |f: f64, w_hat: bool| -> Vec<usize> {
        let value = |m: &(usize, usize, f64)| if w_hat { w_of(m.1, m.2) } else { m.2 };
        gauges
            .iter()
            .map(|g| mults.iter().filter(|m| m.0 == g.k && value(m) < f).count())
            .collect()
    };
    let (m, fm) = mults
        .iter()
        .copied()
        .filter(|m| m.1 != 1)
        .map(|m| (m, (m.2 * w_of(m.1, m.2)).sqrt()))
        .filter(|&(_, fm)| counts(fm, false) != counts(fm, true))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .expect("premise: a cw ≠ 1 multiplet whose floor splits the raw and ŵ cut counts");
    let p = Su2ColumnDeflation::from_gauges(&gauges, fm);
    println!(
        "T-c: f_m = {fm:.6e} at multiplet (bond {}, 2s = {}, σ {:.6e}, ŵ {:.6e}): per-bond cut \
         counts raw {:?} vs ŵ {:?}; the kernel's {:?}",
        m.0,
        m.1,
        m.2,
        w_of(m.1, m.2),
        counts(fm, false),
        counts(fm, true),
        p.bonds().iter().map(|r| r.deflated).collect::<Vec<_>>()
    );
    assert_eq!(p.bonds().len(), gauges.len(), "a report per interior bond");
    for (g, rep) in gauges.iter().zip(p.bonds()) {
        let vals: Vec<f64> = g
            .sigma
            .iter()
            .flat_map(|(_, v)| v.iter().copied())
            .collect();
        let cut_max = vals
            .iter()
            .copied()
            .filter(|&s| s < fm)
            .fold(0.0f64, f64::max);
        let kept_min = vals
            .iter()
            .copied()
            .filter(|&s| s >= fm)
            .fold(f64::INFINITY, f64::min);
        assert_eq!(rep.bond, g.k);
        assert_eq!(rep.total, vals.len(), "bond {}: every multiplet seen", g.k);
        assert_eq!(
            rep.deflated,
            vals.iter().filter(|&&s| s < fm).count(),
            "bond {}: the cut count at f_m is the raw one",
            g.k
        );
        assert_eq!(
            rep.largest_deflated.to_bits(),
            cut_max.to_bits(),
            "bond {}: largest_deflated is the raw extreme",
            g.k
        );
        assert_eq!(
            rep.smallest_kept.to_bits(),
            kept_min.to_bits(),
            "bond {}: smallest_kept is the raw extreme",
            g.k
        );
    }
    let h = generic_su2_h::<f64>(space);
    let plain = weingarten_operator(&h, &x);
    let reg = weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(smin));
    let (cl, cr) = plain.bases();
    for xi in su2_seeds(&x, cl, cr, 2, 0xD0B2) {
        let (p, q) = (plain.b_apply(&xi), reg.b_apply(&xi));
        for i in 0..p.num_coeffs() {
            assert_eq!(
                p.coeff(i).to_bits(),
                q.coeff(i).to_bits(),
                "Hard(σ_min) at a doublet: b_apply coefficient {i} is not the plain context's"
            );
        }
    }
}

mod k5_spin_flux {
    use faer::linalg::matmul::matmul;
    use faer::{Accum, Mat, Par};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::{applied, deflated, embed, gauges_of, raw_sigma_min};
    use crate::core::deflate::ColumnDeflation;
    use crate::core::deflate::oracle::asym;
    use crate::core::factor::{dense_self_adjoint_smallest, svd_uvt_svals};
    use crate::core::weingarten::FloorPolicy;
    use crate::su2::border_fixture::{
        BOND, base_state, dense_form, plant_sector, planted_raw_sigma, with_small_schmidt,
    };
    use crate::su2::deflate::{LiftMode, Su2ColumnDeflation};
    use crate::su2::expand::expand;
    use crate::su2::flux::{channel_offset, lift_bond_sketch};
    use crate::su2::frame::Su2Sketch;
    use crate::su2::hamiltonian::assembly::WireGate;
    use crate::su2::qn::{Su2RankGrid, Su2Sector};
    use crate::su2::tangent::Su2TtTangent;
    use crate::su2::tensorops::src_offset;
    use crate::su2::vector::Su2TtVector;
    use crate::su2::weingarten::{Su2BondGauge, tangent_basis_shifted, tangent_dim_grids};

    const FLUXES: [usize; 3] = [2, 4, 6];

    fn plant(two_s: usize) -> (Su2TtVector<f64>, Su2Sector, f64, f64) {
        let x0 = base_state();
        let u = plant_sector(&x0, BOND, two_s).expect("fixture: a plantable sector at bond 3");
        let x = with_small_schmidt(&x0, BOND, two_s, 1e-4);
        let sp = planted_raw_sigma(&x, BOND, two_s);
        let f = if two_s == 1 { 1.6 * sp } else { 1.9 * sp };
        (x, u, sp, f)
    }

    fn combos(basis: &[Su2TtTangent<f64>], n: usize, seed: u64) -> Vec<Su2TtTangent<f64>> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let mut v = basis[0].scale(0.0);
                for b in basis {
                    let c: f64 = rand::RngExt::sample(&mut rng, rand_distr::StandardNormal);
                    v.axpy(c, b);
                }
                let nrm = v.norm();
                v.scale_mut(1.0 / nrm);
                v
            })
            .collect()
    }

    fn apply_mode(
        p: &Su2ColumnDeflation<f64>,
        xi: &Su2TtTangent<f64>,
        mode: LiftMode,
    ) -> Su2TtTangent<f64> {
        let mut v = xi.clone();
        p.apply_lift_mode(&mut v, mode);
        v
    }

    fn null_sketches(
        gauges: &[Su2BondGauge<f64>],
        select: impl Fn(f64) -> Option<f64>,
    ) -> Vec<Su2Sketch<f64>> {
        gauges
            .iter()
            .map(|g| {
                let r = g.r.row_ranks();
                let mut s = Su2Sketch::<f64>::zeros(g.r.space, g.k, &r, &r);
                for u in g.r.sectors() {
                    let Some(blk) = g.r.block(u) else { continue };
                    if blk.nrows() == 0 || blk.ncols() == 0 {
                        continue;
                    }
                    let (_w, sv, vt) = svd_uvt_svals(blk);
                    let n = vt.ncols();
                    let mut m = Mat::<f64>::zeros(n, n);
                    for (j, &sig) in sv.iter().enumerate() {
                        if let Some(c) = select(sig) {
                            for p in 0..n {
                                for q in 0..n {
                                    m[(p, q)] += c * vt[(j, p)] * vt[(j, q)];
                                }
                            }
                        }
                    }
                    if let Some(mut dst) = s.block_mut(u) {
                        dst.copy_from(m.as_ref());
                    }
                }
                s
            })
            .collect()
    }

    fn lifted_reference(
        xi: &Su2TtTangent<f64>,
        x_r: &[Su2RankGrid],
        nulls: &[Su2Sketch<f64>],
    ) -> Su2TtTangent<f64> {
        let mut out = xi.clone();
        let two_k = xi.two_s_flux as i64;
        for nb in nulls {
            let b = nb.k;
            let k = b - 1;
            let comp = &xi.components[k];
            let out_space = comp.space;
            let lift = lift_bond_sketch(nb, &x_r[b], &xi.r_col[b], out_space, two_k);
            for t in out_space.bond_sectors(b) {
                let c = comp.vertical_unfolding(t);
                if c.nrows() == 0 || c.ncols() == 0 {
                    continue;
                }
                let Some(l) = lift.block(t) else { continue };
                assert_eq!(
                    l.nrows(),
                    c.ncols(),
                    "the lift lives on the dressed column grid"
                );
                let mut cl = Mat::<f64>::zeros(c.nrows(), c.ncols());
                matmul(cl.as_mut(), Accum::Replace, c.as_ref(), l, 1.0, Par::Seq);
                let new = Mat::<f64>::from_fn(c.nrows(), c.ncols(), |i, j| c[(i, j)] - cl[(i, j)]);
                out.components[k].scatter_vertical(t, new.as_ref());
            }
        }
        out
    }

    #[derive(Debug, Clone, Copy)]
    struct Window {
        t: Su2Sector,
        off: usize,
        width: usize,
        rd: usize,
        rows: usize,
        rm: usize,
    }

    impl Window {
        fn room(&self) -> usize {
            self.rows.saturating_sub(self.rm)
        }
    }

    fn windows(xi: &Su2TtTangent<f64>, x_r: &[Su2RankGrid], b: usize, u: Su2Sector) -> Vec<Window> {
        let comp = &xi.components[b - 1];
        let out = comp.space;
        let mut w = Vec::new();
        for t in out.bond_sectors(b) {
            let rd = xi.r_col[b][t];
            if rd == 0 {
                continue;
            }
            let Some(off) = channel_offset(t, u, &x_r[b], xi.two_s_flux as i64, out.n) else {
                continue;
            };
            let rows = comp.vertical_unfolding(t).nrows();
            w.push(Window {
                t,
                off,
                width: x_r[b][u],
                rd,
                rows,
                rm: xi.r[b][t],
            });
        }
        w
    }

    fn window_share(xi: &Su2TtTangent<f64>, b: usize, w: &Window) -> f64 {
        let c = xi.components[b - 1].vertical_unfolding(w.t);
        let mut s = 0.0f64;
        for j in w.off..w.off + w.width {
            for i in 0..c.nrows() {
                s += c[(i, j)] * c[(i, j)];
            }
        }
        (crate::su2::centerweight(w.t.two_s, xi.two_s_flux) * s).sqrt()
    }

    fn dist(a: &Su2TtTangent<f64>, b: &Su2TtTangent<f64>) -> f64 {
        a.sub(b).norm()
    }

    #[test]
    fn cr_su2_shifted_projector_is_the_lifted_null_sketch() {
        let mut legs = 0usize;
        for two_s in [1usize, 2] {
            let (x, u_p, sp, f) = plant(two_s);
            let (bl, br, gauges) = gauges_of(&x);
            let x_r = bl.ranks.clone();
            let et = 3e-2;
            let hard = Su2ColumnDeflation::from_gauges(&gauges, f);
            let tik =
                Su2ColumnDeflation::from_gauges_weighted(&gauges, |s| s * s / (s * s + et * et));
            assert_eq!(deflated(&hard), 1, "premise: the floor cuts the plant only");
            let n_hard = null_sketches(&gauges, |s| (s < f).then_some(1.0));
            let n_tik = null_sketches(&gauges, |s| {
                let c = 1.0 - s * s / (s * s + et * et);
                (c > 1e-15).then_some(c)
            });
            for two_k in FLUXES {
                if tangent_dim_grids(&x.space, &x_r, two_k) == 0 {
                    println!("S0 2s_p = {two_s}, 2κ = {two_k}: empty channel, skipped");
                    continue;
                }
                let basis = tangent_basis_shifted(&bl, &br, two_k);
                let xis = combos(&basis, 4, 0x5E50 + two_k as u64);
                let out = basis[0].components[0].space;
                for (b, xrb) in x_r.iter().enumerate().take(x.space.d).skip(1) {
                    for t in out.bond_sectors(b) {
                        for u in x.space.bond_sectors(b) {
                            let ru = xrb[u];
                            let want = if ru == 0
                                || u.n != t.n
                                || u.irrep != t.irrep
                                || !(two_k as i64 >= (t.two_s as i64 - u.two_s as i64).abs()
                                    && two_k as i64 <= (t.two_s + u.two_s) as i64
                                    && (t.two_s + u.two_s + two_k) % 2 == 0)
                            {
                                None
                            } else {
                                let off = src_offset(
                                    t.n as i64,
                                    t.two_s as i64,
                                    u.two_s as i64,
                                    two_k as i64,
                                    0,
                                    t.irrep,
                                    xrb,
                                    out.n,
                                    WireGate::Triangle,
                                );
                                (off >= 0).then_some(off as usize)
                            };
                            assert_eq!(
                                channel_offset(t, u, xrb, two_k as i64, out.n),
                                want,
                                "channel_offset at bond {b}, t {t:?}, u {u:?}, 2κ {two_k}"
                            );
                        }
                    }
                }
                let wins = windows(&basis[0], &x_r, BOND, u_p);
                println!("S0 2s_p = {two_s} (σ_p {sp:.3e}), 2κ = {two_k}: plant windows {wins:?}");
                for (label, p, nulls) in [("Hard", &hard, &n_hard), ("Tikhonov", &tik, &n_tik)] {
                    let mut worst = 0.0f64;
                    for xi in &xis {
                        let got = applied(p, xi);
                        let fan = apply_mode(p, xi, LiftMode::Fan);
                        for i in 0..got.num_coeffs() {
                            assert_eq!(
                                got.coeff(i).to_bits(),
                                fan.coeff(i).to_bits(),
                                "{label} 2κ {two_k}: apply_lift_mode(Fan) is apply, bit for bit"
                            );
                        }
                        let want = lifted_reference(xi, &x_r, nulls);
                        worst = worst.max(dist(&got, &want) / xi.norm());
                        let d = x.space.d;
                        let (gl, xl) = (got.component(d - 1), xi.component(d - 1));
                        assert_eq!(gl.num_coeffs(), xl.num_coeffs());
                        for i in 0..gl.num_coeffs() {
                            assert_eq!(
                                gl.coeff(i).to_bits(),
                                xl.coeff(i).to_bits(),
                                "{label}: the last component has no right bond and is untouched"
                            );
                        }
                    }
                    println!(
                        "S0 2s_p = {two_s}, 2κ = {two_k}, {label}: max ‖apply − (I − lift(N))‖/‖ξ‖ \
                         = {worst:.2e}"
                    );
                    assert!(
                        worst <= 1e-14,
                        "{label} 2κ {two_k}: the factored windows are not I − lift_bond_sketch(N_b) \
                         ({worst:e})"
                    );
                }
                legs += 1;
            }
        }
        assert!(
            legs >= 2,
            "non-vacuity: at least one live channel per plant"
        );
    }

    #[test]
    fn cr_su2_shifted_projector_is_orthogonal_counted_and_gauge_preserving() {
        for two_s in [1usize, 2] {
            let (x, u_p, _sp, f) = plant(two_s);
            let (bl, br, gauges) = gauges_of(&x);
            let hard = Su2ColumnDeflation::from_gauges(&gauges, f);
            let et = 3e-2;
            let tik =
                Su2ColumnDeflation::from_gauges_weighted(&gauges, |s| s * s / (s * s + et * et));
            let s_min = raw_sigma_min(&gauges);
            let ident = Su2ColumnDeflation::from_gauges(&gauges, 0.5 * s_min);
            assert!(
                ident.is_identity(),
                "a floor below every raw σ deflates nothing"
            );
            let et_tiny = s_min * 5e-16f64.sqrt();
            let c_max = gauges
                .iter()
                .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
                .map(|s| 1.0 - s * s / (s * s + et_tiny * et_tiny))
                .fold(0.0f64, f64::max);
            assert!(
                c_max > 1e-16 && c_max <= 1e-15,
                "premise: the largest Tikhonov removed fraction sits just under the rule ({c_max:e})"
            );
            let tik_tiny = Su2ColumnDeflation::from_gauges_weighted(&gauges, |s| {
                s * s / (s * s + et_tiny * et_tiny)
            });
            assert!(
                tik_tiny.is_identity(),
                "Tikhonov(tiny ε) touches no multiplet"
            );
            for two_k in FLUXES {
                let dim = tangent_dim_grids(&x.space, &bl.ranks, two_k);
                if dim == 0 {
                    println!("S1 2s_p = {two_s}, 2κ = {two_k}: empty channel, skipped");
                    continue;
                }
                let basis = tangent_basis_shifted(&bl, &br, two_k);
                let n = basis.len();
                assert_eq!(n, dim, "premise: the basis spans the structural T^S'");
                let m = dense_form(&basis, |v| applied(&hard, v));
                let (mut idem, mut tr) = (0.0f64, 0.0f64);
                for i in 0..n {
                    tr += m[i * n + i];
                    for j in 0..n {
                        let mm: f64 = (0..n).map(|q| m[i * n + q] * m[q * n + j]).sum();
                        idem = idem.max((mm - m[i * n + j]).abs());
                    }
                }
                let sym = asym(&m, n);
                let mut leak = 0.0f64;
                for b in &basis {
                    let pb = applied(&hard, b);
                    let back = Su2TtTangent::project_shifted(&bl, &br, two_k, &pb.to_vector());
                    leak = leak.max(dist(&back, &pb));
                }
                let removed = hard.removed_dim(&basis[0]);
                let mt = dense_form(&basis, |v| applied(&tik, v));
                let sym_t = asym(&mt, n);
                let ev = dense_self_adjoint_smallest::<f64>(&mt, n, 0).0;
                let (lo, hi) = (
                    ev.iter().copied().fold(f64::INFINITY, f64::min),
                    ev.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                );
                let wins = windows(&basis[0], &bl.ranks, BOND, u_p);
                println!(
                    "S1 2s_p = {two_s}, 2κ = {two_k}: n = {n}, removed {removed}, trace {tr:.12}, \
                     sym {sym:.1e}, idem {idem:.1e}, leak {leak:.1e}; Tikhonov sym {sym_t:.1e}, \
                     spectrum [{lo:.3e}, {hi:.3e}]; windows {:?}",
                    wins.iter()
                        .map(|w| (w.t, w.off, w.width, w.rd, w.rows, w.rm, w.room()))
                        .collect::<Vec<_>>()
                );
                assert!(
                    sym <= 1e-12 && idem <= 1e-12 && leak <= 1e-12,
                    "2κ {two_k}: sym {sym:e} idem {idem:e} leak {leak:e}"
                );
                assert!(
                    (tr - (n - removed) as f64).abs() <= 1e-9,
                    "2κ {two_k}: rank {tr} vs the shifted grid count {}",
                    n - removed
                );
                assert!(sym_t <= 1e-12, "Tikhonov symmetry {sym_t:e}");
                assert!(
                    lo >= -1e-12 && hi <= 1.0 + 1e-12,
                    "Tikhonov: a PSD contraction, spectrum [{lo:e}, {hi:e}]"
                );
                if two_k == 2 {
                    assert!(
                        wins.iter().any(|w| w.room() > 0),
                        "premise: the plant has a window with gauge room on the triplet channel"
                    );
                    assert!(
                        removed > 0,
                        "non-vacuity: Pc removes a direction of T^(S=1)"
                    );
                }
                let v = &basis[n / 2];
                let pv = applied(&ident, v);
                for i in 0..v.num_coeffs() {
                    assert_eq!(
                        pv.coeff(i).to_bits(),
                        v.coeff(i).to_bits(),
                        "2κ {two_k}: identity is bitwise on a shifted tangent"
                    );
                }
                for v in &basis {
                    let pv = applied(&tik_tiny, v);
                    for i in 0..v.num_coeffs() {
                        assert_eq!(
                            pv.coeff(i).to_bits(),
                            v.coeff(i).to_bits(),
                            "2κ {two_k}: Tikhonov(tiny ε) is bitwise the identity on a shifted \
                             tangent"
                        );
                    }
                }
            }
        }
        let (_pg, space) = crate::test_util::su2_pg_space(6, 6, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let mut x: Su2TtVector<f64> = crate::su2::vector::tt_randn(
            space,
            &crate::test_util::su2_capped_grids(&space, 1),
            &mut rng,
        );
        x.scale_mut(1.0 / x.norm_lr(false));
        let (bl, br, gauges) = gauges_of(&x);
        let mut ss: Vec<f64> = gauges
            .iter()
            .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
            .collect();
        ss.sort_by(f64::total_cmp);
        let mid = ss.len() / 2;
        assert!(
            ss[mid] >= ss[mid - 1] * (1.0 + 1e-3),
            "premise: a relative raw-σ gap ≥ 1e-3 at the median cut"
        );
        let f = (ss[mid - 1] * ss[mid]).sqrt();
        let hard = Su2ColumnDeflation::from_gauges(&gauges, f);
        let mut discriminated = 0usize;
        for two_k in FLUXES {
            if tangent_dim_grids(&x.space, &bl.ranks, two_k) == 0 {
                continue;
            }
            let basis = tangent_basis_shifted(&bl, &br, two_k);
            let n = basis.len();
            assert_eq!(
                n,
                tangent_dim_grids(&x.space, &bl.ranks, two_k),
                "premise: the basis spans the structural T^S'"
            );
            let tpl = &basis[0];
            let removed = hard.removed_dim(tpl);
            let mut same_n = 0usize;
            for g in &gauges {
                let b = g.k;
                let comp = &tpl.components[b - 1];
                for t in comp.space.bond_sectors(b) {
                    assert_eq!(
                        tpl.r[b][t], bl.ranks[b][t],
                        "the row grid is the masked copy of r at bond {b}, label {t:?}"
                    );
                }
                for (u, v) in &g.sigma {
                    let n_cut = v.iter().filter(|&&s| s < f).count();
                    for t in comp.space.bond_sectors(b) {
                        if tpl.r_col[b][t] > 0 && t.n == u.n {
                            let rows = comp.vertical_unfolding(t).nrows();
                            same_n += rows.saturating_sub(tpl.r[b][t]) * n_cut;
                        }
                    }
                }
            }
            let m = dense_form(&basis, |v| applied(&hard, v));
            let tr: f64 = (0..n).map(|i| m[i * n + i]).sum();
            println!(
                "S1 pg leg 2κ = {two_k}: n = {n}, removed {removed}, trace {tr:.12}, same-n \
                 reading {same_n}"
            );
            assert!(
                (tr - (n - removed) as f64).abs() <= 1e-9,
                "pg 2κ {two_k}: rank {tr} vs the shifted grid count {}",
                n - removed
            );
            if same_n != removed {
                discriminated += 1;
            }
        }
        assert!(
            discriminated >= 1,
            "premise: the same-n reading differs from removed_dim on some pg channel"
        );
    }

    #[test]
    fn cr_su2_shifted_projector_is_the_spin_block_of_the_expanded_abelian_projector() {
        for two_s in [1usize, 2] {
            let (x, u_p, sp, f) = plant(two_s);
            let (bl, br, gauges) = gauges_of(&x);
            let x_ab = expand(&x, 0);
            let (al, ar) = crate::solvers::riemannian::tangent_bases(&x_ab);
            let g_ab = crate::core::weingarten::bond_gauges(&al, &ar, FloorPolicy::default_hard());
            let ab_deflated =
                |p: &ColumnDeflation<f64>| -> usize { p.bonds().iter().map(|r| r.deflated).sum() };
            let emb = |xi: &Su2TtTangent<f64>| embed(&xi.to_vector(), &al, &ar, xi.norm());
            let p_su2 = Su2ColumnDeflation::from_gauges(&gauges, f);
            let p_ab = ColumnDeflation::from_gauges(&g_ab, f);
            assert_eq!(deflated(&p_su2), 1, "premise: one multiplet");
            assert_eq!(
                ab_deflated(&p_ab),
                two_s + 1,
                "premise: its 2s_p + 1 abelian members"
            );
            let et = 3e-2;
            let d_ab = ColumnDeflation::from_gauges_weighted(&g_ab, |s| s * s / (s * s + et * et));
            let d_su2 =
                Su2ColumnDeflation::from_gauges_weighted(&gauges, |s| s * s / (s * s + et * et));
            let mut probes: Vec<(usize, crate::core::tangent::TtTangent<f64>)> = Vec::new();
            {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E2B);
                for _ in 0..2 {
                    let dir: Su2TtVector<f64> =
                        crate::su2::vector::tt_randn(x.space, &x.ranks, &mut rng);
                    let mut t = Su2TtTangent::project(&bl, &br, &dir);
                    let nrm = t.norm();
                    t.scale_mut(1.0 / nrm);
                    probes.push((0, emb(&t)));
                }
            }
            let mut live = Vec::new();
            for two_k in FLUXES {
                if tangent_dim_grids(&x.space, &bl.ranks, two_k) == 0 {
                    continue;
                }
                let basis = tangent_basis_shifted(&bl, &br, two_k);
                for xi in combos(&basis, 2, 0x5E2C + two_k as u64) {
                    probes.push((two_k, emb(&xi)));
                }
                live.push((two_k, basis));
            }
            for (two_k, basis) in &live {
                let two_k = *two_k;
                let xis = combos(basis, 4, 0x5E2D + two_k as u64);
                let (mut worst, mut worst_t, mut moved, mut cross) =
                    (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for xi in &xis {
                    let mut d = emb(&applied(&p_su2, xi));
                    let mut r = emb(xi);
                    p_ab.apply(&mut r);
                    for (s2, e) in &probes {
                        if *s2 != two_k {
                            cross = cross.max(e.dot(&r).abs());
                        }
                    }
                    d.axpy(-1.0, &r);
                    worst = worst.max(d.norm() / xi.norm());
                    moved = moved.max(dist(&applied(&p_su2, xi), xi));
                    let mut dt = emb(&applied(&d_su2, xi));
                    let mut rt = emb(xi);
                    d_ab.apply(&mut rt);
                    dt.axpy(-1.0, &rt);
                    worst_t = worst_t.max(dt.norm() / xi.norm());
                }
                println!(
                    "S2 2s_p = {two_s} (σ_p {sp:.3e}), 2κ = {two_k}: |emb Pc − P_ab emb| {worst:.2e}, \
                     Tikhonov one-ε {worst_t:.2e}, cross-spin {cross:.2e}, max ‖Pc ξ − ξ‖ \
                     {moved:.2e}"
                );
                assert!(
                    worst <= 1e-12,
                    "Hard 2κ {two_k}: not the spin block ({worst:e})"
                );
                assert!(
                    worst_t <= 1e-12,
                    "Tikhonov 2κ {two_k}: one-ε weights {worst_t:e}"
                );
                assert!(
                    cross <= 1e-13,
                    "2κ {two_k}: P_ab mixes spin blocks ({cross:e})"
                );
                if two_k == 2 {
                    assert!(
                        moved >= 1e-3,
                        "non-vacuity: Pc moved no triplet seed ({moved:e})"
                    );
                }
            }
            let (_, basis2) = live
                .iter()
                .find(|(k, _)| *k == 2)
                .expect("fixture premise: the triplet channel is live");
            let xi = &combos(basis2, 1, 0x5E2E)[0];
            let mut xi_pl = xi.clone();
            xi_pl.axpy(-1.0, &applied(&p_su2, xi));
            let npl = xi_pl.norm();
            assert!(
                npl >= 1e-3,
                "premise: the seed carries planted weight ({npl:e})"
            );
            let mut target = emb(&xi_pl);
            p_ab.apply(&mut target);
            let wins = windows(&xi_pl, &bl.ranks, BOND, u_p);
            let shares: Vec<(Su2Sector, usize, f64)> = wins
                .iter()
                .map(|w| (w.t, w.off, window_share(&xi_pl, BOND, w)))
                .collect();
            println!(
                "S2 AO table 2s_p = {two_s}, u = {u_p:?}: (t, off, width, rd, rows, rm, room) {:?}; \
                 planted shares {shares:?} of ‖ξ_pl‖ = {npl:.3e}",
                wins.iter()
                    .map(|w| (w.t, w.off, w.width, w.rd, w.rows, w.rm, w.room()))
                    .collect::<Vec<_>>()
            );
            let ao = |mode: LiftMode| -> f64 {
                let mut e = emb(&apply_mode(&p_su2, &xi_pl, mode));
                e.axpy(-1.0, &target);
                e.norm()
            };
            let (fan_ok, label_only, offset_zero) = (
                ao(LiftMode::Fan),
                ao(LiftMode::LabelOnly),
                ao(LiftMode::OffsetZero),
            );
            println!(
                "S2 AO 2s_p = {two_s}: Fan {fan_ok:.2e}, LabelOnly {label_only:.2e}, OffsetZero \
                 {offset_zero:.2e} (‖ξ_pl‖ {npl:.3e})"
            );
            assert!(
                fan_ok <= 1e-12 * npl.max(1.0),
                "the production lift on ξ_pl ({fan_ok:e})"
            );
            let cross_label = shares
                .iter()
                .filter(|(t, _, _)| *t != u_p)
                .fold(0.0f64, |m, &(_, _, s)| m.max(s));
            assert!(
                cross_label >= 1e-3 * npl,
                "AO-fan premise: ξ_pl has weight {cross_label:e} in the cross-label windows"
            );
            assert!(
                label_only >= 1e-3 * npl,
                "AO-fan: acting only at t = u must break the identity ({label_only:e})"
            );
            if two_s == 2 {
                let shifted = shares
                    .iter()
                    .filter(|(_, off, _)| *off > 0)
                    .fold(0.0f64, |m, &(_, _, s)| m.max(s));
                assert!(
                    shifted >= 1e-3 * npl,
                    "AO-offset premise: ξ_pl has weight {shifted:e} in a window at offset > 0 \
                     (t = (4, 2), offset r[3][(4, 0)])"
                );
                assert!(
                    offset_zero >= 1e-3 * npl,
                    "AO-offset: acting at offset 0 must break the identity ({offset_zero:e})"
                );
            } else {
                assert!(
                    wins.iter().all(|w| w.off == 0),
                    "in-body claim: every spin-½ window sits at offset 0"
                );
                let z = dist(
                    &apply_mode(&p_su2, xi, LiftMode::OffsetZero),
                    &applied(&p_su2, xi),
                );
                assert!(
                    z <= 1e-14,
                    "with every offset 0, OffsetZero is the production lift ({z:e})"
                );
                let p12 = Su2ColumnDeflation::from_gauges(&gauges, 1.2 * sp);
                let p_ab_same = ColumnDeflation::from_gauges(&g_ab, 1.2 * sp);
                let p_ab_conv = ColumnDeflation::from_gauges(&g_ab, 1.2 * sp / 2f64.sqrt());
                assert_eq!(deflated(&p12), 1, "a raw 1.2σ_p cuts the plant");
                assert_eq!(
                    ab_deflated(&p_ab_same),
                    2,
                    "the same abelian floor cuts the members"
                );
                assert!(
                    p_ab_conv.is_identity(),
                    "the ŵ-converted floor keeps the members"
                );
                let mut worst_12 = 0.0f64;
                for xi in &combos(basis2, 4, 0x5E2D + 2) {
                    let mut d = emb(&applied(&p12, xi));
                    let mut r = emb(xi);
                    p_ab_same.apply(&mut r);
                    d.axpy(-1.0, &r);
                    worst_12 = worst_12.max(d.norm() / xi.norm());
                }
                let e_pl = emb(&applied(&p12, &xi_pl));
                let mut ok = e_pl.clone();
                let mut r = emb(&xi_pl);
                p_ab_same.apply(&mut r);
                ok.axpy(-1.0, &r);
                let mut bad = e_pl.clone();
                let mut r = emb(&xi_pl);
                p_ab_conv.apply(&mut r);
                bad.axpy(-1.0, &r);
                println!(
                    "S2 AO-units: same floor {worst_12:.2e} on 4 triplet tangents, {:.2e} on ξ_pl; \
                     ŵ-converted {:.2e} (‖ξ_pl‖ {npl:.3e})",
                    ok.norm(),
                    bad.norm()
                );
                assert!(
                    worst_12 <= 1e-12,
                    "1.2σ_p with the same floor on the triplet tangents: {worst_12:e}"
                );
                assert!(
                    ok.norm() <= 1e-12 * npl,
                    "1.2σ_p with the same floor: {:e}",
                    ok.norm()
                );
                assert!(
                    bad.norm() >= 1e-3 * npl,
                    "AO-units: the ŵ-converted floor must break it ({:e})",
                    bad.norm()
                );
            }
        }
    }
}
