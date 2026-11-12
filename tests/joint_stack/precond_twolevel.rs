// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::TtVector;
use qntt::core::tangent::TtTangent;
use qntt::solvers::hess::{SympPrecond, SymplecticDavidsonOpts};
use qntt::solvers::joint::{
    JointContext, JointOpts, JointVector, joint_context, joint_lowest_davidson,
};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::optimize::d2_fixture;
use crate::util::*;

fn joint_seed_block(
    jctx: &JointContext<'_>,
    b: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<JointVector> {
    let (base_l, base_r) = jctx.bases();
    let spec = jctx.spec();
    let n_eff = jctx.n_eff();
    (0..b)
        .map(|_| {
            let dir: TtVector<f64> = randomized_state(base_l.space, 1, 2, rng);
            let xi = TtTangent::project(base_l, base_r, &dir);
            let mut v = JointVector::from_tangent(spec, xi);
            v.c = (0..n_eff)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect();
            v
        })
        .collect()
}

#[test]
fn tjp_auto_resolves_off() {
    let fx = d2_fixture();
    let out = &fx.out;
    assert!(out.info.converged, "D2 fixture must converge");
    let jctx = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("the D2 solution must be joint-critical");
    let k = 2usize;
    let run = |precond: SympPrecond| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA070);
        let v0s = joint_seed_block(&jctx, k + 2, &mut rng);
        let opts = SymplecticDavidsonOpts {
            precond,
            ..Default::default()
        };
        joint_lowest_davidson(&jctx, &v0s, k, &opts).expect("stable")
    };
    let (wa, ia) = run(SympPrecond::Auto);
    let (wo, io) = run(SympPrecond::Off);
    assert_eq!(ia.iterations, io.iterations);
    assert_eq!(ia.applies, io.applies);
    for (a, o) in wa.iter().zip(wo.iter()) {
        assert_eq!(
            a.to_bits(),
            o.to_bits(),
            "Auto must BE Off on the joint layer"
        );
    }
}
