// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::QnSpace;
use qntt::core::tangent::tangent_basis;
use qntt::solvers::hess::{
    SympLobpcgOpts, SympPrecond, SymplecticSpectrum, hessian_grams, symplectic_from_grams,
    symplectic_lowest_lobp4dcg, weingarten_operator,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t8m_lobp4dcg_matches_dense_truncated() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    let (a, b) = hessian_grams(&ctx, &basis);
    let bmax = b.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(bmax > 1e-8, "B vacuous — fixture proves nothing");
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_from_grams(&a, &b, n, 1e-10)
    else {
        panic!("dense oracle unstable");
    };

    let k = 4usize;
    for (arm, precond, seed_bit) in [
        ("precond-off", SympPrecond::Off, 0u64),
        (
            "precond-on",
            SympPrecond::DiagUnshifted { kappa: 0.0 },
            1u64,
        ),
    ] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B1 ^ seed_bit);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
        let opts = SympLobpcgOpts {
            precond,
            ..SympLobpcgOpts::default()
        };
        let (omega, info) = symplectic_lowest_lobp4dcg(&ctx, &v0s, k, &opts)
            .unwrap_or_else(|e| panic!("[{arm}] must not error: {e:?}"));
        println!(
            "t8m_lobp4dcg_truncated[{arm}]: n={n} built={} applies={} residuals={:?}",
            info.iterations, info.applies, info.residuals
        );
        assert!(info.converged, "[{arm}] must converge at this fixture");
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "[{arm}] ω[{j}]: lobp4dcg {wi} vs dense {wd}"
            );
        }
    }
}
