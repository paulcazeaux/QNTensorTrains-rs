// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::QnSpace;
use qntt::core::tangent::TtTangent;
use qntt::solvers::hess::{
    RpaKernel, RpaMetric, SympLobpcgOpts, SympPrecond, SymplecticDavidsonOpts,
    symplectic_lowest_lobp4dcg, symplectic_lowest_signed_davidson, weingarten_operator,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn ta_auto_matches_explicit_d0_davidson() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let (k, b) = (2usize, 2usize);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let run = |precond: SympPrecond| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B4);
            let v0s = seed_block::<f64>(space, &base_l, &base_r, b, &mut rng);
            let opts = SymplecticDavidsonOpts {
                max_subspace: k + b + 1,
                precond,
                kernel,
                ..Default::default()
            };
            symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts).0
        };
        let sa = run(SympPrecond::Auto);
        let sd = run(SympPrecond::DiagUnshifted { kappa: 0.0 });
        assert_eq!(sa.iterations, sd.iterations, "{kernel:?}");
        assert_eq!(sa.applies, sd.applies, "{kernel:?}");
        assert_eq!(sa.converged, sd.converged, "{kernel:?}");
        let (wa, wd) = (sa.omega2(), sd.omega2());
        assert_eq!(wa.len(), wd.len(), "{kernel:?}");
        for (a, d) in wa.iter().zip(wd.iter()) {
            assert_eq!(
                a.to_bits(),
                d.to_bits(),
                "{kernel:?}: Auto must BE D0 on the TT layer"
            );
        }
        for (a, d) in sa.residuals().iter().zip(sd.residuals().iter()) {
            assert_eq!(a.to_bits(), d.to_bits(), "{kernel:?}");
        }
    }
}

#[test]
fn ta_auto_matches_explicit_d0_lobp4dcg() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let (k, b) = (2usize, 3usize);
    let run = |precond: SympPrecond| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B4);
        let v0s: Vec<TtTangent<f64>> = seed_block::<f64>(space, &base_l, &base_r, b, &mut rng);
        let opts = SympLobpcgOpts {
            precond,
            ..Default::default()
        };
        symplectic_lowest_lobp4dcg(&ctx, &v0s, k, &opts).expect("stable")
    };
    let (wa, ia) = run(SympPrecond::Auto);
    let (wd, id) = run(SympPrecond::DiagUnshifted { kappa: 0.0 });
    assert_eq!(ia.iterations, id.iterations);
    assert_eq!(ia.applies, id.applies);
    for (a, d) in wa.iter().zip(wd.iter()) {
        assert_eq!(
            a.to_bits(),
            d.to_bits(),
            "Auto must BE D0 on the TT lobp lane"
        );
    }
}
