// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::Mat;
use qntt::core::factor::svd_uvt_svals;
use qntt::core::tangent::TtTangent;
use qntt::core::weingarten::FloorPolicy;
use qntt::solvers::hess::{CurvatureReg, weingarten_operator as ab_weingarten_operator};
use qntt::su2::coreops::mul_sketch_left;
use qntt::su2::expand::expand;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh};
use qntt::su2::hess::{
    hessian_grams as su2_hessian_grams, weingarten_operator as su2_weingarten_operator,
};
use qntt::su2::qn::{Su2Sector, Su2Space};
use qntt::su2::riemannian::tangent_bases as su2_tangent_bases;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::{bond_gauges as su2_bond_gauges, tangent_basis as su2_tangent_basis};
use qntt::{QnSpace, Scalar, SparseHamiltonian, TtVector};
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const D: usize = 6;
const BOND: usize = 3;

pub(crate) fn planted_singlet(eps: f64) -> (Su2TtVector<f64>, f64) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5D2F_A7E3);
    let mut x: Su2TtVector<f64> = randomized_su2_state(Su2Space::new(D, 6, 0), 2, 3, &mut rng);
    x.scale_mut(1.0 / x.norm());
    x.sigma_floor_trim_mut(3e-2);
    x.scale_mut(1.0 / x.norm());
    let (bl, br) = su2_tangent_bases(&x);
    let gauges = su2_bond_gauges(&bl, &br, FloorPolicy::default_hard());
    let g = &gauges[BOND - 1];
    let room = |s: Su2Sector| {
        let u = bl.core(BOND - 1).vertical_unfolding(s);
        u.nrows().saturating_sub(u.ncols())
    };
    let t = g
        .r
        .sectors()
        .into_iter()
        .filter(|&s| s.two_s == 1 && g.r.block(s).is_some_and(|m| m.nrows() >= 2) && room(s) > 0)
        .max_by_key(|&s| room(s))
        .expect("fixture: a bond-3 spin-½ sector of rank ≥ 2 with room");
    let (w, mut sv, vt) = svd_uvt_svals(g.r.block(t).expect("block"));
    *sv.last_mut().expect("rank ≥ 2") = eps;
    let n = sv.len();
    let block = Mat::from_fn(n, n, |i, j| {
        sv.iter()
            .enumerate()
            .map(|(q, &s)| w.get(i, q) * s * vt.get(q, j))
            .sum::<f64>()
    });
    let mut r = g.r.clone();
    r.block_mut(t).expect("block").copy_from(block.as_ref());
    let mut cores = Vec::with_capacity(D);
    for k in 0..BOND {
        cores.push(bl.core(k).clone());
    }
    cores.push(mul_sketch_left(&r, br.core(BOND)));
    for k in BOND + 1..D {
        cores.push(br.core(k).clone());
    }
    let mut y = Su2TtVector::new(x.space, cores);
    let nrm = y.norm();
    y.scale_mut(1.0 / nrm);
    let (yl, yr) = su2_tangent_bases(&y);
    let sp = su2_bond_gauges(&yl, &yr, FloorPolicy::default_hard())[BOND - 1]
        .sigma
        .iter()
        .filter(|(s, _)| s.two_s == 1)
        .flat_map(|(_, v)| v.iter().copied())
        .fold(f64::INFINITY, f64::min);
    (y, sp)
}

fn embed(u: &Su2TtTangent<f64>, al: &TtVector<f64>, ar: &TtVector<f64>) -> TtTangent<f64> {
    let y = expand(&u.to_vector(), 0);
    let t = TtTangent::project(al, ar, &y);
    let (n_u, n_t) = (u.norm(), t.norm());
    assert!(
        (n_t - n_u).abs() <= 1e-12 * n_u.max(1.0),
        "premise: expand∘ι is isometric ({n_t} vs {n_u})"
    );
    let back = &t.to_vector() - &y;
    let defect = back.norm_lr(false);
    assert!(
        defect <= 1e-10 * y.norm_lr(false).max(1.0),
        "premise: expand(ι(u), 0) is already tangent at x_ab (moved by {defect:e})"
    );
    t
}

fn cross_gram(
    emb: &[TtTangent<f64>],
    op: impl Fn(&TtTangent<f64>) -> TtTangent<f64>,
) -> (Vec<f64>, f64) {
    let n = emb.len();
    let mut g = vec![0.0; n * n];
    let mut closure = 0.0f64;
    for j in 0..n {
        let img = op(&emb[j]);
        let mut res = img.clone();
        for i in 0..n {
            g[i * n + j] = emb[i].dot(&img);
            res.axpy(-g[i * n + j], &emb[i]);
        }
        closure = closure.max(res.norm() / img.norm().max(1.0));
    }
    (g, closure)
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn crg_su2_regularized_b_is_the_spin_zero_block_of_the_abelian_one() {
    let (t, v) = generic_truncated_tv(D);
    let (x, sp) = planted_singlet(1e-4);
    let h = Su2SparseHamiltonian::new(x.space, &t, &v, None, f64::EPSILON);
    let x_ab = expand(&x, 0);
    let space_ab = QnSpace::new(D, 3, 3);
    assert_eq!(x_ab.space, space_ab);
    let h_ab = SparseHamiltonian::<f64>::new(space_ab, &t, &v, None, f64::EPSILON);
    let lam = su2_rayleigh(&h, &x).re_s();
    let lam_ab = qntt::rayleigh_quotient(&h_ab, &x_ab).re_s();
    assert!(
        (lam - lam_ab).abs() <= 1e-12 * lam.abs().max(1.0),
        "premise: λ agrees across the layers ({lam} vs {lam_ab})"
    );
    let ctx = su2_weingarten_operator(&h, &x);
    let ctx_ab = ab_weingarten_operator(&h_ab, &x_ab);
    let (bl, br) = ctx.bases();
    let basis = su2_tangent_basis(bl, br, false);
    let n = basis.len();
    let (al, ar) = ctx_ab.bases();
    let emb: Vec<TtTangent<f64>> = basis.iter().map(|u| embed(u, al, ar)).collect();

    let (_, b_su2) = su2_hessian_grams(&ctx, &basis);
    let (b_x, closure_off) = cross_gram(&emb, |w| ctx_ab.b_apply(w));
    let scale = max_abs(&b_su2);
    let err_off = max_abs_diff(&b_su2, &b_x);

    let su2_below = |f: f64| -> usize {
        ctx.bond_gauges()
            .iter()
            .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
            .filter(|&s| s < f)
            .count()
    };
    let ab_below = |f: f64| -> usize {
        ctx_ab
            .bond_gauges()
            .iter()
            .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
            .filter(|&s| s < f)
            .count()
    };

    let f = 1.6 * sp;
    assert_eq!(
        (su2_below(f), ab_below(f)),
        (1, 2),
        "(2) premise: 1.6σ_p cuts the planted multiplet and its 2 members only"
    );
    let ctx_reg = su2_weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f));
    let ctx_ab_reg = ab_weingarten_operator(&h_ab, &x_ab).with_curvature(CurvatureReg::Hard(f));
    let (_, b_su2_reg) = su2_hessian_grams(&ctx_reg, &basis);
    let (b_x_reg, closure_reg) = cross_gram(&emb, |w| ctx_ab_reg.b_apply(w));
    let err_reg = max_abs_diff(&b_su2_reg, &b_x_reg);
    let removed = max_abs_diff(&b_su2_reg, &b_su2);

    let f12 = 1.2 * sp;
    let f12_w = f12 / 2f64.sqrt();
    assert_eq!(
        (su2_below(f12), ab_below(f12), ab_below(f12_w)),
        (1, 2, 0),
        "(3) premise: 1.2σ_p cuts the plant and its members, the ŵ-converted 1.2σ_p/√2 nothing"
    );
    let ctx_12 = su2_weingarten_operator(&h, &x).with_curvature(CurvatureReg::Hard(f12));
    let (_, b_su2_12) = su2_hessian_grams(&ctx_12, &basis);
    let ctx_ab_12 = ab_weingarten_operator(&h_ab, &x_ab).with_curvature(CurvatureReg::Hard(f12));
    let (b_x_12, closure_12) = cross_gram(&emb, |w| ctx_ab_12.b_apply(w));
    let err_12 = max_abs_diff(&b_su2_12, &b_x_12);
    let ctx_ab_w = ab_weingarten_operator(&h_ab, &x_ab).with_curvature(CurvatureReg::Hard(f12_w));
    let (b_x_w, _) = cross_gram(&emb, |w| ctx_ab_w.b_apply(w));
    let broken = max_abs_diff(&b_su2_12, &b_x_w);

    println!(
        "I1: n = {n}, σ_p = {sp:.4e}, λ = {lam:.12}; Off |ΔB| {err_off:.2e} (max|B| {scale:.3e}, \
         closure {closure_off:.1e}); Hard(1.6σ_p) on both layers |ΔB| {err_reg:.2e} closure \
         {closure_reg:.1e} (removed {removed:.3e}); Hard(1.2σ_p) on both layers |ΔB| {err_12:.2e} \
         closure {closure_12:.1e}; the ŵ-converted abelian 1.2σ_p/√2 breaks by {broken:.3e}"
    );
    assert!(
        err_off <= 1e-12 * scale,
        "(1) Off: the SU(2) B is not the spin-0 block ({err_off:e})"
    );
    assert!(err_reg <= 1e-12 * scale, "(2) regularized: {err_reg:e}");
    assert!(
        closure_reg <= 1e-10,
        "(2) the abelian regularized B leaves the S' = 0 block ({closure_reg:e})"
    );
    assert!(
        removed >= 1e-3 * scale,
        "(2) non-vacuity: the planted coupling must be removed"
    );
    assert!(
        b_su2_12
            .iter()
            .zip(&b_su2_reg)
            .all(|(p, q)| p.to_bits() == q.to_bits()),
        "(3) the SU(2) Hard(1.2σ_p) cuts the raw σ_p: bitwise the Hard(1.6σ_p) Gram"
    );
    assert!(
        err_12 <= 1e-12 * scale,
        "(3) regularized at 1.2σ_p: {err_12:e}"
    );
    assert!(
        closure_12 <= 1e-10,
        "(3) the abelian Hard(1.2σ_p) B leaves the S' = 0 block ({closure_12:e})"
    );
    assert!(
        broken >= 1e-3 * scale,
        "(3) anti-oracle: the ŵ-converted abelian floor must break the identity ({broken:e})"
    );
}

#[test]
fn crg_su2_q6_planted_fallback_is_a_nonvacuous_regularized_leg() {
    let (t, v) = generic_truncated_tv(D);
    let (x, _sp) = planted_singlet(1e-4);
    let ham = Su2SparseHamiltonian::new(x.space, &t, &v, None, f64::EPSILON);
    let space = x.space;
    let (nonvacuous, _, _) =
        crate::su2_sector::pg_regularized_leg("planted (Q6 fallback)", space, t, v, ham, x, false);
    assert!(nonvacuous, "the Q6 fallback must carry a non-vacuous cut");
}
