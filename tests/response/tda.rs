// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::{TtTangent, tangent_basis, tangent_dim};
use qntt::pointgroup::Irrep;
use qntt::solvers::hess::{
    SymplecticSpectrum, TdaDavidsonOpts, WeingartenCtx, hessian_grams, symplectic_from_grams,
    tda_lowest_davidson, weingarten_operator,
};
use qntt::{QnSpace, SparseHamiltonian, TtVector, rayleigh_quotient, tt_randn};
use qntt_testkit::dense::{dense_eigen, flat_to_nested};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::sector_util::{
    D4, F_NDN, F_NUP, d6_minimizer, dense_levels, exact_typed_ground, fx4_pg, fx4_tv,
};

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

fn eigs(m: &[f64], n: usize) -> Vec<f64> {
    let mut e = dense_eigen(&flat_to_nested(m, n)).0;
    e.sort_by(|p, q| p.partial_cmp(q).expect("finite eigenvalue"));
    e
}

#[test]
fn tda_i1_exact_full_rank_ground_gives_the_exact_gaps() {
    let pg = fx4_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = fx4_tv(pg);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (x, e0, levels) = exact_typed_ground(space, &t, &v);
    let ctx = weingarten_operator(&ham, &x);
    let dim = tangent_dim(ctx.bases().0);
    assert_eq!(
        dim,
        levels.len(),
        "fixture precondition: full typed rank (the tangent space is the whole sector)"
    );
    let k = dim - 1;
    let opts = TdaDavidsonOpts {
        tol: 1e-12,
        ..TdaDavidsonOpts::default()
    };
    let out = tda_lowest_davidson(&ctx, &seeds(&ctx, &x, 3, 0x7DA6), k, &opts);
    let gaps: Vec<f64> = levels[1..].iter().map(|l| l - e0).collect();
    println!(
        "i1: dim {dim}, θ {:?}\n    exact gaps {gaps:?}\n    converged {} it {} applies {}",
        out.theta, out.info.converged, out.info.iterations, out.info.applies
    );
    assert!(out.info.converged);
    assert_eq!(out.info.space_dim, dim - 1);
    assert_eq!(out.theta.len(), k);
    for (j, (th, gap)) in out.theta.iter().zip(&gaps).enumerate() {
        assert!(
            (th - gap).abs() <= 1e-9,
            "level {j}: θ {th} vs exact gap {gap}"
        );
        assert!(
            (out.info.rayleigh[j] - gap).abs() <= 1e-9,
            "level {j}: E_ξ − λ {} vs {gap}",
            out.info.rayleigh[j]
        );
    }
    assert!(
        out.info.negative.is_empty(),
        "the exact ground is the lowest state of its sector"
    );
}

#[test]
fn tda_i2_d6_minimizer_matches_the_dense_gram_bounds_the_levels_and_dominates_the_rpa() {
    let (space, t, v, ham, x) = d6_minimizer();
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    let (a, b) = hessian_grams(&ctx, &basis);
    let theta_dense = eigs(&a, n);
    let hrr: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
    let hii: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p - q).collect();
    let (min_rr, min_ii) = (eigs(&hrr, n)[0], eigs(&hii, n)[0]);
    println!("i2: n = {n}, λ_min(A+B) {min_rr:.3e}, λ_min(A−B) {min_ii:.3e}");
    assert!(
        min_rr > 0.0 && min_ii > 0.0,
        "fixture precondition: both RPA blocks positive definite at the minimizer"
    );
    let SymplecticSpectrum::Stable { omega } = symplectic_from_grams(&a, &b, n, 1e-10) else {
        panic!("fixture precondition: the d6 minimizer is Stable")
    };

    let k = 6;
    let opts = TdaDavidsonOpts {
        tol: 1e-11,
        ..TdaDavidsonOpts::default()
    };
    let out = tda_lowest_davidson(&ctx, &seeds(&ctx, &x, k + 2, 0x7DA7), k, &opts);
    println!(
        "i2: θ {:?}\n    dense {:?}\n    ω {:?}\n    converged {} it {} applies {}",
        out.theta,
        &theta_dense[..k],
        &omega[..k],
        out.info.converged,
        out.info.iterations,
        out.info.applies
    );
    assert!(out.info.converged);
    assert_eq!(out.info.space_dim, n);
    assert_eq!(out.theta.len(), k);
    let mut gap_max = 0.0f64;
    for j in 0..k {
        let th = out.theta[j];
        assert!(
            (th - theta_dense[j]).abs() <= 1e-9,
            "root {j}: θ {th} vs dense {}",
            theta_dense[j]
        );
        assert!(th > 1e-6, "root {j}: θ {th} — a radial zero mode leaked in");
        assert!(
            omega[j] <= th + 1e-9,
            "root {j}: ω {} above the TDA θ {th}",
            omega[j]
        );
        gap_max = gap_max.max(th - omega[j]);
    }
    assert!(
        gap_max >= 1e-6,
        "non-vacuity: B must separate θ from ω somewhere (max θ − ω {gap_max:e})"
    );

    let lambda = rayleigh_quotient(&ham, &x) / x.norm().powi(2);
    let levels = dense_levels(
        space.d,
        &t,
        &v,
        space.pg,
        space.nup,
        space.ndn,
        space.target,
    );
    let mut bound: Vec<f64> = std::iter::once(lambda)
        .chain(out.theta.iter().map(|th| lambda + th))
        .collect();
    bound.sort_by(|p, q| p.partial_cmp(q).expect("finite"));
    for (i, (bd, lv)) in bound.iter().zip(&levels).enumerate() {
        println!(
            "    MacDonald {i}: λ+θ {bd:.10} ≥ E {lv:.10} (slack {:.3e})",
            bd - lv
        );
        assert!(
            *bd >= lv - 1e-10,
            "MacDonald {i}: compressed level {bd} below the exact level {lv}"
        );
    }
}
