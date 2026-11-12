// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::solvers::hess::{
    RpaKernel, RpaMetric, RpaReport, SympLobpcgOpts, SympPrecond, SymplecticDavidsonOpts,
    SymplecticSpectrum, hessian_grams, hessian_grams_c64, symplectic_eigenvalues,
    symplectic_from_grams, symplectic_from_grams_c64, symplectic_lowest_lobp4dcg,
    symplectic_lowest_signed_davidson, symplectic_lowest_signed_davidson_c64,
    unshifted_precond_inverse_diagnostic, weingarten_operator,
};
use qntt::{C64, QnSpace, Scalar};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn inverse_extremes(t: &qntt::core::tangent::TtTangent<f64>) -> (f64, f64) {
    let n = t.num_coeffs();
    assert!(n > 0, "empty diagonal model — the fixture proves nothing");
    (0..n).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), i| {
        let v = t.coeff(i);
        (lo.min(v), hi.max(v))
    })
}

#[test]
fn td0_inverse_spd_and_capped() {
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, _base_l, _base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);

    let a = ctx.a_diagonal();
    let n = a.num_coeffs();
    let model: Vec<f64> = (0..n).map(|i| a.coeff(i) * a.coeff(i)).collect();
    let dmax = model.iter().fold(0.0f64, |m, &v| m.max(v.max(0.0)));
    assert!(
        dmax > 0.0,
        "vacuous squared-diagonal model (dmax = {dmax:.3e}) — the cap cannot be tested"
    );

    for kappa in [0.0f64, 1e3, 10.0] {
        let t = unshifted_precond_inverse_diagnostic(&ctx, kappa);
        assert_eq!(
            t.num_coeffs(),
            n,
            "the inverse must share the model's shape (kappa={kappa})"
        );
        let kappa_eff = if kappa > 0.0 { kappa } else { 1e3 };
        let c = dmax.max(f64::MIN_POSITIVE) / kappa_eff;
        for (i, &d_i) in model.iter().enumerate() {
            let want = 1.0 / (d_i.max(0.0) + c);
            let got = t.coeff(i);
            assert!(
                got > 0.0,
                "T[{i}] = {got:.6e} is not strictly positive (kappa={kappa}) — D0 must be SPD in \
                 the coefficient sense; that positivity is what removes the near-singular-shift \
                 collapse channel entirely"
            );
            assert!(
                (got - want).abs() <= 1e-12 * want,
                "T[{i}] (kappa={kappa}): got {got:.17e}, want 1/((Re d)_+ + c) = {want:.17e} \
                 (c = {c:.6e})"
            );
        }
        let (lo, hi) = inverse_extremes(&t);
        let ratio = hi / lo;
        println!(
            "td0_inverse_spd_and_capped: kappa={kappa} kappa_eff={kappa_eff} c={c:.6e} \
             min={lo:.6e} max={hi:.6e} range={ratio:.6e}"
        );
        assert!(
            ratio <= kappa_eff + 1.0 + 1e-9 * (kappa_eff + 1.0),
            "κ(T) = {ratio:.6e} exceeds the R2 bound κ+1 = {:.6e} (kappa={kappa})",
            kappa_eff + 1.0
        );
    }

    let t_auto = unshifted_precond_inverse_diagnostic(&ctx, 0.0);
    let t_1e3 = unshifted_precond_inverse_diagnostic(&ctx, 1e3);
    for i in 0..n {
        assert_eq!(
            t_auto.coeff(i),
            t_1e3.coeff(i),
            "kappa = 0 must route to the auto value 1e3 (coefficient {i})"
        );
    }

    let (lo10, hi10) = inverse_extremes(&unshifted_precond_inverse_diagnostic(&ctx, 10.0));
    let (loa, hia) = inverse_extremes(&t_auto);
    let (r10, ra) = (hi10 / lo10, hia / loa);
    assert!(
        r10 <= 11.0 + 1e-9,
        "kappa = 10 must cap the range at 11 (got {r10:.6e})"
    );
    if ra > 11.0 + 1e-9 {
        assert!(
            r10 < ra,
            "the cap is not binding: range at kappa=10 ({r10:.6e}) not below the auto-lane range \
             ({ra:.6e}), even though the model's own spread exceeds 11"
        );
    } else {
        println!(
            "td0_inverse_spd_and_capped: auto-lane range {ra:.6e} ≤ 11 — the kappa=10 cap is \
             vacuous at this fixture (recorded, not asserted)"
        );
    }
}

#[test]
fn td0_matches_dense_unforced() {
    use qntt::core::tangent::tangent_basis;
    let space = QnSpace::new(4, 2, 1);
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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81D0);
    let k = 4usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
            kernel,
            ..Default::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        println!(
            "td0_matches_dense_unforced {kernel:?}: n={n} built={} applies={} residuals={:?}",
            so.iterations,
            so.applies,
            so.residuals()
        );
        assert!(
            so.converged,
            "{kernel:?}: D0 must converge at the exhaustion-class fixture"
        );
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: D0 davidson {wi} vs dense {wd}"
            );
        }
    }
}

#[test]
fn td0_forced_restart_matches_dense() {
    use qntt::core::tangent::tangent_basis;
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense oracle unstable");
    };

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81D1);
    let (k, b) = (2usize, 2usize);
    let v0s = seed_block::<f64>(space, &base_l, &base_r, b, &mut rng);
    let opts = SymplecticDavidsonOpts {
        max_subspace: k + b + 1,
        precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
        instrument: true,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
    let RpaReport::Signed { diag, .. } = &so.report else {
        panic!("the signed kernel must have run: {:?}", so.report);
    };
    let restarts = diag.as_ref().map(|d| d.restarts).unwrap_or(0);
    println!(
        "td0_forced_restart_matches_dense: built={} applies={} restarts={restarts} residuals={:?}",
        so.iterations,
        so.applies,
        so.residuals()
    );
    assert!(
        restarts >= 1,
        "the cell is only meaningful if restarts actually fire (got {restarts})"
    );
    assert!(
        so.converged,
        "D0 must still converge under forced thick restarts"
    );
    let omega = so
        .omega()
        .expect("a converged stable solve publishes its ω");
    for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
        assert!(
            (wi - wd).abs() < 1e-6 * wd.max(1.0),
            "ω[{j}]: D0 restarted davidson {wi} vs dense {wd}"
        );
    }
}

#[test]
fn td0_lobp_matches_dense() {
    use qntt::core::tangent::tangent_basis;
    let space = QnSpace::new(4, 2, 1);
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

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81D2);
    let k = 4usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    let opts = SympLobpcgOpts {
        precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
        ..SympLobpcgOpts::default()
    };
    let (omega, info) = symplectic_lowest_lobp4dcg(&ctx, &v0s, k, &opts).expect("stable fixture");
    println!(
        "td0_lobp_matches_dense: n={n} built={} applies={} residuals={:?}",
        info.iterations, info.applies, info.residuals
    );
    assert!(
        info.converged,
        "D0 must converge in the restart-free kernel"
    );
    for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
        assert!(
            (wi - wd).abs() < 1e-6 * wd.max(1.0),
            "ω[{j}]: D0 lobp4dcg {wi} vs dense {wd}"
        );
    }
}

#[test]
fn td0_c64_smoke() {
    use qntt::core::tangent::tangent_basis;
    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C5, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    let (a, b) = hessian_grams_c64(&ctx, &basis);
    let bmax = b.iter().fold(0.0f64, |acc, v| acc.max(v.abs_s()));
    assert!(bmax > 1e-8, "B vacuous — fixture proves nothing");
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_from_grams_c64(&a, &b, n, 1e-10)
    else {
        panic!("dense oracle unstable");
    };
    let mut doubled: Vec<f64> = dense.iter().flat_map(|&w| [w, w]).collect();
    doubled.sort_by(|p, q| p.partial_cmp(q).unwrap());

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81D3);
    let k = 6usize;
    let v0s = seed_block::<C64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Rr)] {
        let opts = SymplecticDavidsonOpts {
            precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
            kernel,
            ..Default::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson_c64(&ctx, &v0s, k, &opts);
        println!(
            "td0_c64_smoke {kernel:?}: n={n} built={} applies={} residuals={:?}",
            so.iterations,
            so.applies,
            so.residuals()
        );
        assert!(
            so.converged,
            "{kernel:?}: D0 must converge on the C64 layer"
        );
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        for (j, (wi, wd)) in omega.iter().zip(doubled.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: D0 davidson (C64) {wi} vs doubled dense {wd}"
            );
        }
    }
}
