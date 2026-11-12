// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::solvers::hess::riemannian_gradient_env as ab_gradient_env;
use qntt::solvers::riemannian::{
    RiemannianOpts as AbRiemannianOpts, normalize_mut as ab_normalize_mut,
    riemannian_cg as ab_riemannian_cg, tangent_bases as ab_tangent_bases,
};
use qntt::su2::hamiltonian::{rayleigh_quotient as su2_rq, xt_h_y as su2_xt_h_y};
use qntt::su2::hess::riemannian_gradient_env as su2_gradient_env;
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases as su2_tangent_bases,
};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::{QnSpace, Scalar, TtVector, dot, rayleigh_quotient, xt_h_y};
use qntt_testkit::generic_su2_h;
use qntt_testkit::generic_truncated_h;
use qntt_testkit::randomized_state;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn pc1_abelian_norm_lr_fast_path_matches_the_sweep() {
    let space = QnSpace::new(6, 3, 3);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5701);
    for trial in 0..4 {
        let mut x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let center = trial % space.d;
        x.move_core(center, true);
        assert_eq!(x.ortho, Some(center));
        let fast = x.norm_lr(false);
        let mut y = x.clone();
        y.ortho = None;
        let swept = y.norm_lr(false);
        assert!(
            (fast - swept).abs() <= 1e-12 * swept.max(1.0),
            "trial {trial}: fast {fast:.17e} vs swept {swept:.17e}"
        );
        let d2 = dot(&x, &x).sqrt();
        assert!((fast - d2).abs() <= 1e-12 * d2.max(1.0));
    }
}

#[test]
fn pc3_retract_hosvd_leaves_the_train_right_canonical() {
    let space = Su2Space::new(6, 6, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5703);
    let h = generic_su2_h(space);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    normalize_mut(&mut x);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rq(&h, &x).re_s();
    let eta = su2_gradient_env(&h, &bl, &br, lambda).scale(-1.0);
    let xa = eta.retract_hosvd(0.1);
    assert_eq!(
        xa.ortho,
        Some(0),
        "the `norm_lr` fast path is only a win if the retraction leaves the flag set"
    );
    let fast = xa.norm_lr(false);
    let mut y = xa.clone();
    y.ortho = None;
    let swept = y.norm_lr(false);
    assert!((fast - swept).abs() <= 1e-11 * swept.max(1.0));

    let ab_space = QnSpace::new(6, 3, 3);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5713);
    let hab = generic_truncated_h(ab_space);
    let mut xab: TtVector<f64> = randomized_state(ab_space, 2, 3, &mut rng);
    ab_normalize_mut(&mut xab);
    let (bl, br) = ab_tangent_bases(&xab);
    let lam = rayleigh_quotient(&hab, &xab).re_s();
    let eta = ab_gradient_env(&hab, &bl, &br, lam).scale(-1.0);
    let xa = eta.retract_hosvd(0.1);
    assert_eq!(xa.ortho, Some(0));
}

#[test]
fn pc4_abelian_linesearch_c1_identity() {
    let space = QnSpace::new(6, 3, 3);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5704);
    let h = generic_truncated_h(space);
    let mut x: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    ab_normalize_mut(&mut x);
    let (bl, br) = ab_tangent_bases(&x);
    let lambda = rayleigh_quotient(&h, &x).re_s();
    let grad = ab_gradient_env(&h, &bl, &br, lambda);

    let rnd: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    let etas = [
        grad.scale(-1.0),
        TtTangent::project(&bl, &br, &rnd),
        grad.scale(-1.0).add(&TtTangent::project(&bl, &br, &x)),
    ];
    for (i, eta) in etas.iter().enumerate() {
        let w = eta.to_vector();
        let direct = xt_h_y(&x, &h, &w).re_s();
        let identity = grad.dot(eta).re_s() + lambda * dot(&x, &w).re_s();
        assert!(
            (direct - identity).abs() <= 1e-10 * direct.abs().max(1.0),
            "eta {i}: direct {direct:.17e} vs identity {identity:.17e}"
        );
    }
}

#[test]
fn pc5_su2_linesearch_c1_identity() {
    let space = Su2Space::new(6, 6, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5705);
    let h = generic_su2_h(space);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    normalize_mut(&mut x);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rq(&h, &x).re_s();
    let grad = su2_gradient_env(&h, &bl, &br, lambda);

    let rnd: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    let etas = [
        grad.scale(-1.0),
        Su2TtTangent::project(&bl, &br, &rnd),
        grad.scale(-1.0).add(&Su2TtTangent::project(&bl, &br, &x)),
    ];
    for (i, eta) in etas.iter().enumerate() {
        let w = eta.to_vector();
        let direct = su2_xt_h_y(&x, &h, &w).re_s();
        let identity = grad.dot(eta).re_s() + lambda * qntt::su2::ops::dot(&x, &w).re_s();
        assert!(
            (direct - identity).abs() <= 1e-10 * direct.abs().max(1.0),
            "eta {i}: direct {direct:.17e} vs identity {identity:.17e}"
        );
    }
}

#[test]
fn pc6_cg_still_reaches_a_critical_point_on_both_layers() {
    let space = Su2Space::new(6, 6, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5706);
    let h = generic_su2_h(space);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 3, 3, &mut rng);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-12;
    opts.max_iter = 400;
    let (lam, x) = riemannian_cg(&h, &x0, &opts);
    let (bl, br) = su2_tangent_bases(&x);
    let gn = su2_gradient_env(&h, &bl, &br, lam).norm();
    assert!(gn < 1e-8, "SU(2) CG ‖grad‖ = {gn:.3e}");

    let ab_space = QnSpace::new(6, 3, 3);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0_5716);
    let hab = generic_truncated_h(ab_space);
    let x0: TtVector<f64> = randomized_state(ab_space, 3, 3, &mut rng);
    let mut opts = AbRiemannianOpts::new(&ab_space);
    opts.eps = 1e-12;
    opts.max_iter = 400;
    let (lam, x) = ab_riemannian_cg(&hab, &x0, &opts);
    let (bl, br) = ab_tangent_bases(&x);
    let gn = ab_gradient_env(&hab, &bl, &br, lam).norm();
    assert!(gn < 1e-8, "abelian CG ‖grad‖ = {gn:.3e}");
}
