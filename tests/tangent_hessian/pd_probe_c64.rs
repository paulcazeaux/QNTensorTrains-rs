// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::integrals::{IntegralsOptions, load_fixture_dir};
use qntt::solvers::hess::{
    HessPart, PdVerdict, hessian_pd_probe, hessian_pd_probe_c64, weingarten_operator,
};
use qntt::solvers::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::su2::hess::{
    hessian_pd_probe as su2_pd_probe, hessian_pd_probe_c64 as su2_pd_probe_c64,
    weingarten_operator as su2_weingarten_operator,
};
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{
    RiemannianOpts as Su2RiemannianOpts, normalize_mut as su2_normalize_mut,
    riemannian_cg as su2_riemannian_cg, tangent_bases as su2_tangent_bases,
};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::{C64, CoeffTensor, QnSpace, RoundMode, Scalar, SparseCore, SparseHamiltonian, TtVector};
use qntt_testkit::generic_su2_h;
use qntt_testkit::generic_truncated_h;
use qntt_testkit::randomized_state;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn cast_tt_c64(x: &TtVector<f64>) -> TtVector<C64> {
    let space = x.space;
    let cores: Vec<SparseCore<C64>> = (0..space.d)
        .map(|k| {
            let c = x.core(k);
            let mut out = SparseCore::<C64>::zeros(space, k, &x.ranks[k], &x.ranks[k + 1]);
            for i in 0..c.num_coeffs() {
                out.set_coeff(i, C64::new(c.coeff(i), 0.0));
            }
            out
        })
        .collect();
    TtVector::new(space, cores)
}

fn cast_su2_c64(x: &Su2TtVector<f64>) -> Su2TtVector<C64> {
    let space = x.space;
    let cores: Vec<qntt::su2::core::Su2Core<C64>> = (0..space.d)
        .map(|k| {
            let c = x.core(k);
            let mut out =
                qntt::su2::core::Su2Core::<C64>::zeros(space, k, &x.ranks[k], &x.ranks[k + 1]);
            for i in 0..c.num_coeffs() {
                out.set_coeff(i, C64::new(c.coeff(i), 0.0));
            }
            out
        })
        .collect();
    Su2TtVector::new(space, cores)
}

fn verdict_value(v: PdVerdict) -> (f64, f64, &'static str) {
    match v {
        PdVerdict::WitnessNegative { lambda } => (lambda, f64::NAN, "WitnessNegative"),
        PdVerdict::LikelyPd {
            lambda_min,
            residual,
        } => (lambda_min, residual, "LikelyPd"),
        PdVerdict::Inconclusive {
            lambda_min,
            residual,
        } => (lambda_min, residual, "Inconclusive"),
    }
}

#[test]
fn pq1_abelian_complex_probe_matches_min_of_the_two_real_parts() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0B1);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let ham: SparseHamiltonian<f64> = generic_truncated_h(space);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
    let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
    normalize_mut(&mut x);
    let ctx = weingarten_operator(&ham, &x);
    let (base_l, base_r) = ctx.bases();
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let v0 = TtTangent::project(base_l, base_r, &dir);

    let (vrr, _) = hessian_pd_probe(&ctx, HessPart::Rr, &v0, 200);
    let (vii, _) = hessian_pd_probe(&ctx, HessPart::Ii, &v0, 200);
    let (lrr, rrr, trr) = verdict_value(vrr);
    let (lii, rii, tii) = verdict_value(vii);
    let want = lrr.min(lii);

    let ham_c: SparseHamiltonian<C64> = generic_truncated_h(space);
    let x_c = cast_tt_c64(&x);
    x_c.check();
    let ctx_c = weingarten_operator(&ham_c, &x_c);
    let (bl_c, br_c) = ctx_c.bases();
    let dir_c = {
        let dr: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let di: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let mut z = cast_tt_c64(&dr);
        let zi = cast_tt_c64(&di);
        z = &z + &(&zi * C64::new(0.0, 1.0));
        z.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
        z
    };
    let v0_c = TtTangent::project(bl_c, br_c, &dir_c);
    let (vc, applies) = hessian_pd_probe_c64(&ctx_c, &v0_c, 400);
    let (lc, resc, tc) = verdict_value(vc);
    println!(
        "pq1: real H^rr {lrr:.9e} ({trr}), H^ii {lii:.9e} ({tii}) -> min {want:.9e}; \
         complex 𝔥 {lc:.9e} ({tc}, res {resc:.2e}, {applies} applies)"
    );
    let fin = |r: f64| if r.is_finite() { r } else { 0.0 };
    let tol_cmp = (fin(resc) + fin(rrr) + fin(rii)).max(1e-6);
    assert!(
        (lc - want).abs() <= tol_cmp,
        "λ_min(𝔥) must equal min(λ_min(H^rr), λ_min(H^ii)) on a real-embedded problem: \
         {lc} vs {want} (tol {tol_cmp:.2e})"
    );
    assert_eq!(
        vc.is_pd(),
        Some(true),
        "this fixture is stable on both real parts, so 𝔥 must be certified PD"
    );
}

#[test]
fn pq2_su2_complex_probe_matches_min_of_the_two_real_parts() {
    let space = Su2Space::new(4, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0C1);
    let ham = generic_su2_h::<f64>(space);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let mut opts = Su2RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    let (_l, mut x) = su2_riemannian_cg(&ham, &x0, &opts);
    for _ in 0..6 {
        let (_l2, y) = su2_riemannian_cg(&ham, &x, &opts);
        x = y;
    }
    su2_normalize_mut(&mut x);
    let (bl, br) = su2_tangent_bases(&x);
    let lam = qntt::su2::hamiltonian::rayleigh_quotient(&ham, &x).re_s();
    let gn = qntt::su2::hess::riemannian_gradient_env(&ham, &bl, &br, lam).norm();
    assert!(gn < 1e-8, "SU(2) fixture not critical (‖grad‖ = {gn:.3e})");

    let ctx = su2_weingarten_operator(&ham, &x);
    let (base_l, base_r) = ctx.bases();
    let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let v0 = Su2TtTangent::project(base_l, base_r, &dir);
    let (vrr, _) = su2_pd_probe(&ctx, HessPart::Rr, &v0, 200);
    let (vii, _) = su2_pd_probe(&ctx, HessPart::Ii, &v0, 200);
    let (lrr, rrr, trr) = verdict_value(vrr);
    let (lii, rii, tii) = verdict_value(vii);
    let want = lrr.min(lii);

    let ham_c = generic_su2_h::<C64>(space);
    let x_c = cast_su2_c64(&x);
    x_c.check();
    let ctx_c = su2_weingarten_operator(&ham_c, &x_c);
    let (bl_c, br_c) = ctx_c.bases();
    let dir_c = {
        let dr: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let di: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let z = cast_su2_c64(&dr);
        let zi = cast_su2_c64(&di);
        let mut s = &z + &(&zi * C64::new(0.0, 1.0));
        s.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
        s
    };
    let v0_c = Su2TtTangent::project(bl_c, br_c, &dir_c);
    let (vc, applies) = su2_pd_probe_c64(&ctx_c, &v0_c, 400);
    let (lc, resc, tc) = verdict_value(vc);
    println!(
        "pq2: real H^rr {lrr:.9e} ({trr}), H^ii {lii:.9e} ({tii}) -> min {want:.9e}; \
         complex 𝔥 {lc:.9e} ({tc}, res {resc:.2e}, {applies} applies)"
    );
    let fin = |r: f64| if r.is_finite() { r } else { 0.0 };
    let tol_cmp = (fin(resc) + fin(rrr) + fin(rii)).max(1e-6);
    assert!(
        (lc - want).abs() <= tol_cmp,
        "λ_min(𝔥) must equal min(λ_min(H^rr), λ_min(H^ii)): {lc} vs {want} (tol {tol_cmp:.2e})"
    );
    assert_eq!(vc.is_pd(), Some(true));
}

#[test]
fn pq3_complex_probe_witnesses_indefiniteness_at_a_noncritical_point() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0B2);
    let mut xr: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    xr.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    xr.scale_mut(1.0 / xr.norm());
    let ham: SparseHamiltonian<C64> = generic_truncated_h(space);
    let x = cast_tt_c64(&xr);
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let v0 = TtTangent::project(bl, br, &cast_tt_c64(&dir));
    let (v, _applies) = hessian_pd_probe_c64(&ctx, &v0, 200);
    match v {
        PdVerdict::WitnessNegative { lambda } => {
            assert!(
                lambda < -1e-6,
                "witness should be O(1) negative, got {lambda}"
            )
        }
        other => {
            let (l, r, t) = verdict_value(other);
            panic!(
                "expected a negative witness at a random non-critical point; got {t}({l}, res {r})"
            )
        }
    }
}

#[test]
fn pq4_starved_budget_is_inconclusive_never_likelypd() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0B3);
    let x0: TtVector<f64> = randomized_state(space, 2, 2, &mut rng);
    let ham: SparseHamiltonian<f64> = generic_truncated_h(space);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
    let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
    normalize_mut(&mut x);
    let (bl, br) = tangent_bases(&x);
    let n = qntt::core::tangent::tangent_dim(&bl);
    assert!(n > 4, "need a non-trivial tangent space, got n = {n}");

    let ctx = weingarten_operator(&ham, &x);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let v0 = TtTangent::project(&bl, &br, &dir);
    for part in [HessPart::Rr, HessPart::Ii] {
        let (v, _) = hessian_pd_probe(&ctx, part, &v0, 1);
        let (l, r, t) = verdict_value(v);
        assert!(
            !matches!(v, PdVerdict::LikelyPd { .. }),
            "{part:?} at budget 1 reported {t}({l}, res {r}) — an unconverged Davidson must not \
             claim positive-definiteness"
        );
    }

    let ham_c: SparseHamiltonian<C64> = generic_truncated_h(space);
    let x_c = cast_tt_c64(&x);
    let ctx_c = weingarten_operator(&ham_c, &x_c);
    let (bl_c, br_c) = ctx_c.bases();
    let v0_c = TtTangent::project(bl_c, br_c, &cast_tt_c64(&dir));
    let (v, _) = hessian_pd_probe_c64(&ctx_c, &v0_c, 1);
    let (l, r, t) = verdict_value(v);
    assert!(
        !matches!(v, PdVerdict::LikelyPd { .. }),
        "complex probe at budget 1 reported {t}({l}, res {r})"
    );

    let (v, _) = hessian_pd_probe_c64(&ctx_c, &v0_c, 400);
    assert_eq!(
        v.is_pd(),
        Some(true),
        "the fixture must be certifiable at a real budget, else pq4 proves nothing"
    );
}

fn load_fixture(name: &str) -> (CoeffTensor<f64>, CoeffTensor<f64>, usize, usize) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name);
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("load {}: {e}", dir.display()));
    let ints = load.ints;
    (ints.t, ints.v, ints.norb, ints.nelec)
}

#[test]
fn pq5_complex_probe_reaches_the_bottom_and_does_not_lock_on_an_interior_pair() {
    let (t, v, d, nelec) = load_fixture("lih_sto3g");
    let space = Su2Space::new(d, nelec, 0);
    let ham =
        qntt::su2::hamiltonian::Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let codes: Vec<qntt::su2::core::Family> = (0..d)
        .map(|i| {
            if i < nelec / 2 {
                qntt::su2::core::Family::Occ
            } else {
                qntt::su2::core::Family::Empty
            }
        })
        .collect();
    let mut x0: Su2TtVector<f64> = qntt::su2::vector::tt_csf(&codes);
    x0.right_orthogonalize(true);
    let mopts = qntt::solvers::mals::MalsOptions {
        eps: 1e-10,
        bond_dims: vec![8],
        noise: vec![0.0],
        max_iter: 30,
        ..Default::default()
    };
    let (_l, xa) = qntt::su2::solvers::mals::mals(&ham, &x0, &mopts);
    let mut opts = Su2RiemannianOpts::new(&space);
    opts.eps = 1e-12;
    opts.max_iter = 600;
    let (_l, mut x) = su2_riemannian_cg(&ham, &xa, &opts);
    for _ in 0..6 {
        let (_l2, y) = su2_riemannian_cg(&ham, &x, &opts);
        x = y;
    }
    su2_normalize_mut(&mut x);
    let (bl, br) = su2_tangent_bases(&x);
    let lam = qntt::su2::hamiltonian::rayleigh_quotient(&ham, &x).re_s();
    let gn = qntt::su2::hess::riemannian_gradient_env(&ham, &bl, &br, lam).norm();
    assert!(gn < 1e-8, "pq5 fixture not critical (‖grad‖ = {gn:.3e})");

    let ctx = su2_weingarten_operator(&ham, &x);
    let (base_l, base_r) = ctx.bases();
    let n = qntt::su2::weingarten::tangent_dim(base_l);
    let budget = 200usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0D5);
    let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let v0 = Su2TtTangent::project(base_l, base_r, &dir);
    let (vrr, _) = su2_pd_probe(&ctx, HessPart::Rr, &v0, budget);
    let (vii, _) = su2_pd_probe(&ctx, HessPart::Ii, &v0, budget);
    let (lrr, _, trr) = verdict_value(vrr);
    let (lii, _, tii) = verdict_value(vii);
    let want = lrr.min(lii);

    let (tc, vc2) = {
        let mut tc = CoeffTensor::<C64>::zeros(d, 2);
        let mut vc = CoeffTensor::<C64>::zeros(d, 4);
        for i in 0..d {
            for j in 0..d {
                let c = t.get(&[i, j]);
                if c != 0.0 {
                    tc.set(&[i, j], C64::new(c, 0.0));
                }
                for k in 0..d {
                    for l in 0..d {
                        let c = v.get(&[i, j, k, l]);
                        if c != 0.0 {
                            vc.set(&[i, j, k, l], C64::new(c, 0.0));
                        }
                    }
                }
            }
        }
        (tc, vc)
    };
    let ham_c = qntt::su2::hamiltonian::Su2SparseHamiltonian::<C64>::new(
        space,
        &tc,
        &vc2,
        None,
        f64::EPSILON,
    );
    let x_c = cast_su2_c64(&x);
    let ctx_c = su2_weingarten_operator(&ham_c, &x_c);
    let (bl_c, br_c) = ctx_c.bases();
    let dir_c = {
        let dr: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let di: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let mut s = &cast_su2_c64(&dr) + &(&cast_su2_c64(&di) * C64::new(0.0, 1.0));
        s.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
        s
    };
    let v0_c = Su2TtTangent::project(bl_c, br_c, &dir_c);
    let (vc, applies) = su2_pd_probe_c64(&ctx_c, &v0_c, budget);
    let (lc, resc, tcs) = verdict_value(vc);
    println!(
        "pq5: n = {n} (deflated 2n−2 = {}); real H^rr {lrr:.9e} ({trr}), H^ii {lii:.9e} ({tii}) \
         -> min {want:.9e}; complex 𝔥 {lc:.9e} ({tcs}, res {resc:.2e}, {applies} applies)",
        2 * n - 2
    );
    assert_eq!(
        vc.is_pd(),
        Some(true),
        "the complex probe must converge on this fixture within {budget} iterations"
    );
    assert!(
        (lc - want).abs() <= 1e-3 * want.abs().max(1.0),
        "the complex probe locked onto an interior eigenpair instead of λ_min: {lc} vs {want} \
         (the pre-fix kernel returns 2.0587 here)"
    );
}
