// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::C64;
use qntt::core::tangent::{TtTangent, tangent_basis, tangent_dim};
use qntt::round::RoundMode;
use qntt::solvers::hess::{
    ModeOpts, ModeResult, PdVerdict, WeingartenCtx, hessian_grams, lowest_hessian_mode,
    weingarten_operator,
};
use qntt::{QnSpace, Scalar, TtVector};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::{
    converged_minimizer, deflate_radial_span, dense_sector_eigen, generic_truncated_h, inf_scale,
    overlap_modulus, radial_mode_at, randomized_state,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Model {
    A,
    Full,
}

#[allow(clippy::too_many_arguments)]
fn run_mode<T: Scalar>(
    ctx: &WeingartenCtx<'_, T>,
    model: Model,
    seed: &TtTangent<T>,
    n_ritz: usize,
    budget: usize,
    warmup: usize,
    neg_early_exit: bool,
    space_dim: usize,
) -> ModeResult<TtTangent<T>> {
    let (bl, br) = ctx.bases();
    let radial = radial_mode_at(bl, br, bl);
    let diag = ctx.a_diagonal();
    let scale = inf_scale(&diag);

    let op = |p: &TtTangent<T>| {
        let mut q = TtTangent::project(bl, br, &p.to_vector());
        deflate_radial_span(&radial, &mut q);
        let mut hq = match model {
            Model::A => ctx.a_apply(&q),
            Model::Full => ctx.hess_apply(&q),
        };
        deflate_radial_span(&radial, &mut hq);
        hq
    };
    let precond =
        |t: &mut TtTangent<T>, theta: f64| qntt::solvers::davidson::precond(t, &diag, theta);
    let clean = |t: &mut TtTangent<T>| {
        let mut q = TtTangent::project(bl, br, &t.to_vector());
        deflate_radial_span(&radial, &mut q);
        *t = q;
    };

    let mut s = TtTangent::project(bl, br, &seed.to_vector());
    deflate_radial_span(&radial, &mut s);

    let opts = ModeOpts {
        budget,
        n_ritz,
        warmup,
        tol_rel: 1e-4,
        neg_early_exit,
        space_dim,
        scale,
    };
    lowest_hessian_mode(op, precond, clean, &s, &opts)
        .expect("lowest_hessian_mode returned None: the cleaned seed collapsed (caller bug)")
}

fn tangent_coords<T: Scalar>(basis: &[TtTangent<T>], v: &TtTangent<T>) -> Vec<T> {
    basis.iter().map(|u| u.dot(v)).collect()
}

fn truncated_fixture() -> (qntt::SparseHamiltonian<f64>, TtVector<f64>, QnSpace) {
    let space = QnSpace::new(4, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0B1);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let ham = generic_truncated_h::<f64>(space);
    let x = converged_minimizer(&ham, &x0);
    (ham, x, space)
}

#[test]
fn lhm_matches_dense_grams_at_truncated_minimizer() {
    let (ham, x, _space) = truncated_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    assert!(
        n >= 4,
        "degenerate fixture: projective tangent dimension {n} is too small to be a test"
    );
    let (a, b) = hessian_grams(&ctx, &basis);
    let hpp: Vec<f64> = a.iter().zip(b.iter()).map(|(p, q)| p + q).collect();
    let bmax = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        bmax > 1e-8,
        "fixture must exercise the Weingarten term (‖B‖_max = {bmax:.3e}); a B ≡ 0 fixture makes \
         this a term-A test only"
    );
    let (evals, evecs) = dense_sector_eigen(&hpp, n);
    assert!(
        evals[1] - evals[0] > 1e-3,
        "lowest pair is degenerate ({:.6e}, {:.6e}); the eigenVECTOR comparison below is then \
         ill-posed — pick another fixture, do not loosen the tolerance",
        evals[0],
        evals[1]
    );

    let diag = ctx.a_diagonal();
    let seed = TtTangent::project(bl, br, &diag.to_vector());
    let space_dim = tangent_dim(bl) - 1;
    let res = run_mode(&ctx, Model::Full, &seed, 2, 120, 8, false, space_dim);

    assert!(res.applies > 0, "helper reported zero applies");
    assert!(
        res.converged,
        "helper did not converge in 120 applies at the campaign's standard d=4 fixture — raise \
         the budget and report, do not loosen the assertions"
    );
    assert!(!res.ritz.is_empty(), "no Ritz pair returned");

    let theta0 = res.ritz[0].theta;
    let r0 = res.ritz[0].residual;
    assert!(
        (theta0 - evals[0]).abs() <= r0 + 1e-6,
        "lowest Ritz value {theta0} vs dense λ_min {} (residual {r0})",
        evals[0]
    );
    assert!(
        res.lambda_fresh >= evals[0] - 1e-10,
        "fresh-apply Rayleigh quotient {} below the dense λ_min {} — the operator the helper \
         applied is not the one the Grams describe",
        res.lambda_fresh,
        evals[0]
    );
    let c = tangent_coords(&basis, &res.ritz[0].vector);
    let ov = overlap_modulus(&c, &evecs[0]);
    assert!(
        ov >= 1.0 - 1e-5,
        "Ritz vector overlaps the dense lowest eigenvector by only {ov:.9}"
    );

    assert_eq!(res.ritz.len(), 2, "n_ritz = 2 must return two pairs");
    assert!(
        res.ritz[1].theta >= res.ritz[0].theta - 1e-12,
        "Ritz pairs not ascending: {} then {}",
        res.ritz[0].theta,
        res.ritz[1].theta
    );
    let r1 = res.ritz[1].residual;
    assert!(
        (res.ritz[1].theta - evals[1]).abs() <= r1 + 1e-3 * evals[1].abs().max(1.0),
        "second Ritz value {} vs dense {} (residual {r1})",
        res.ritz[1].theta,
        evals[1]
    );
    let c1 = tangent_coords(&basis, &res.ritz[1].vector);
    let ov1 = overlap_modulus(&c1, &evecs[1]);
    assert!(
        ov1 >= 1.0 - 1e-3,
        "second Ritz vector overlaps the dense second eigenvector by only {ov1:.9}"
    );

    match res.verdict {
        PdVerdict::LikelyPd { .. } => {}
        other => panic!("expected LikelyPd at a converged fixed-rank minimizer, got {other:?}"),
    }
}

#[test]
fn lhm_a_only_matches_dense_a_gram() {
    let (ham, x, _space) = truncated_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a, _b) = hessian_grams(&ctx, &basis);
    let (evals, evecs) = dense_sector_eigen(&a, n);
    assert!(
        evals[1] - evals[0] > 1e-3,
        "degenerate lowest A-pair; eigenvector comparison ill-posed"
    );

    let diag = ctx.a_diagonal();
    let seed = TtTangent::project(bl, br, &diag.to_vector());
    let res = run_mode(&ctx, Model::A, &seed, 1, 120, 8, false, tangent_dim(bl) - 1);
    assert!(
        res.converged,
        "A-only helper did not converge in 120 applies"
    );
    let theta0 = res.ritz[0].theta;
    assert!(
        (theta0 - evals[0]).abs() <= res.ritz[0].residual + 1e-6,
        "A-only lowest Ritz {theta0} vs dense {}",
        evals[0]
    );
    let ov = overlap_modulus(&tangent_coords(&basis, &res.ritz[0].vector), &evecs[0]);
    assert!(ov >= 1.0 - 1e-5, "A-only eigenvector overlap {ov:.9}");
}

#[test]
fn lhm_warm_start_recovers_the_seeded_eigenvector_in_few_applies() {
    let (ham, x, _space) = truncated_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a, b) = hessian_grams(&ctx, &basis);
    let hpp: Vec<f64> = a.iter().zip(b.iter()).map(|(p, q)| p + q).collect();
    let (evals, evecs) = dense_sector_eigen(&hpp, n);

    let mut seed = basis[0].clone();
    seed.scale_mut(evecs[0][0]);
    for (m, u) in basis.iter().enumerate().skip(1) {
        seed.axpy(evecs[0][m], u);
    }
    let sn = seed.norm();
    seed.scale_mut(1.0 / sn);

    let res = run_mode(
        &ctx,
        Model::Full,
        &seed,
        1,
        2,
        0,
        false,
        tangent_dim(bl) - 1,
    );
    assert!(
        res.applies <= 5,
        "warm start consumed {} applies at budget 2 — the budget is not the matvec count it \
         claims to be",
        res.applies
    );
    assert!(
        (res.ritz[0].theta - evals[0]).abs() <= 1e-8 * evals[0].abs().max(1.0),
        "warm-started Ritz value {} vs dense λ_min {}",
        res.ritz[0].theta,
        evals[0]
    );
    assert!(
        res.ritz[0].residual <= 1e-6 * evals[0].abs().max(1.0),
        "warm-started residual {} is not at eigenvector level",
        res.ritz[0].residual
    );
    assert!(
        res.converged,
        "warm start from an exact eigenvector must converge"
    );
}

#[test]
fn lhm_witness_negative_at_noncritical_point() {
    let space = QnSpace::new(4, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let ham = generic_truncated_h::<f64>(space);
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let seed = TtTangent::project(bl, br, &dir);
    let res = run_mode(
        &ctx,
        Model::Full,
        &seed,
        1,
        60,
        8,
        true,
        tangent_dim(bl) - 1,
    );
    match res.verdict {
        PdVerdict::WitnessNegative { lambda } => {
            assert!(
                lambda < -1e-6,
                "witness should be O(1) negative, got {lambda}"
            );
            assert!(
                (lambda - res.lambda_fresh).abs() < 1e-12,
                "verdict λ {lambda} and lambda_fresh {} must be the SAME fresh apply",
                res.lambda_fresh
            );
        }
        other => panic!(
            "expected a negative witness at a random non-critical point, got {other:?}; a helper \
             that cannot prove indefiniteness cannot serve the index-1 certificate"
        ),
    }
    let wn = res.witness.norm();
    assert!(wn > 0.0, "witness must be a nonzero direction");
}

#[test]
fn lhm_c64_lowest_is_min_of_the_two_real_parts() {
    let (ham, x, space) = truncated_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a, b) = hessian_grams(&ctx, &basis);
    let hpp: Vec<f64> = a.iter().zip(b.iter()).map(|(p, q)| p + q).collect();
    let hmm: Vec<f64> = a.iter().zip(b.iter()).map(|(p, q)| p - q).collect();
    let lrr = dense_sector_eigen(&hpp, n).0[0];
    let lii = dense_sector_eigen(&hmm, n).0[0];
    let want = lrr.min(lii);

    let ham_c: qntt::SparseHamiltonian<C64> = generic_truncated_h::<C64>(space);
    let xc = x.to_c64();
    let ctx_c = weingarten_operator(&ham_c, &xc);
    let (blc, brc) = ctx_c.bases();
    let diag = ctx_c.a_diagonal();
    let seed = TtTangent::project(blc, brc, &diag.to_vector());
    let space_dim = 2 * tangent_dim(blc) - 2;
    let res = run_mode(&ctx_c, Model::Full, &seed, 1, 200, 8, false, space_dim);
    let tol = res.ritz[0].residual + 1e-6;
    assert!(
        (res.lambda_fresh - want).abs() <= tol,
        "C64 λ_min {} vs min(λ_rr {lrr}, λ_ii {lii}) = {want} (tol {tol})",
        res.lambda_fresh
    );
}

#[test]
fn lhm_seed_collapse_returns_none() {
    let (ham, x, _space) = truncated_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let radial = radial_mode_at(bl, br, bl);
    let diag = ctx.a_diagonal();
    let scale = inf_scale(&diag);

    let op = |p: &TtTangent<f64>| {
        let mut q = TtTangent::project(bl, br, &p.to_vector());
        deflate_radial_span(&radial, &mut q);
        let mut hq = ctx.hess_apply(&q);
        deflate_radial_span(&radial, &mut hq);
        hq
    };
    let precond =
        |t: &mut TtTangent<f64>, theta: f64| qntt::solvers::davidson::precond(t, &diag, theta);
    let clean = |t: &mut TtTangent<f64>| {
        let mut q = TtTangent::project(bl, br, &t.to_vector());
        deflate_radial_span(&radial, &mut q);
        *t = q;
    };
    let opts = ModeOpts {
        budget: 20,
        n_ritz: 1,
        warmup: 0,
        tol_rel: 1e-4,
        neg_early_exit: false,
        space_dim: tangent_dim(bl) - 1,
        scale,
    };

    let mut zero_seed = radial.clone();
    zero_seed.scale_mut(0.0);
    assert_eq!(
        zero_seed.norm(),
        0.0,
        "fixture gate: the zero seed must be exact"
    );
    let res = lowest_hessian_mode(op, precond, clean, &zero_seed, &opts);
    assert!(
        res.is_none(),
        "seeding an exact zero must collapse to None, not panic or return a garbage witness"
    );
}
