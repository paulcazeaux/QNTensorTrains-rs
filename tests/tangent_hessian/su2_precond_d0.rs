// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::Scalar;
use qntt::solvers::hess::{
    RpaKernel, RpaMetric, RpaReport, SympDiagnostics, SympPrecond, SymplecticDavidsonOpts,
    SymplecticSpectrum,
};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hess::{
    symplectic_eigenvalues, symplectic_lowest_signed_davidson, weingarten_operator,
};
use qntt::su2::qn::Su2Space;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::tangent_basis;
use qntt_testkit::gad_mol::su2_grad_exact_at;
use qntt_testkit::generic_su2_h;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn su2_minimizer<T: Scalar>(
    ham: Su2SparseHamiltonian<T>,
    space: Su2Space,
    seed: u64,
    lo: usize,
    hi: usize,
) -> (
    Su2SparseHamiltonian<T>,
    Su2TtVector<T>,
    Su2TtVector<T>,
    Su2TtVector<T>,
) {
    use qntt::su2::riemannian::{
        RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x0: Su2TtVector<T> = randomized_su2_state(space, lo, hi, &mut rng);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
    let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
    for _ in 0..8 {
        if su2_grad_exact_at(&ham, &x).norm() < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(&ham, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    x.check();
    let gnorm = su2_grad_exact_at(&ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "SU(2) fixture not critical (‖grad‖={gnorm:.3e})"
    );
    let (base_l, base_r) = tangent_bases(&x);
    (ham, x, base_l, base_r)
}

fn su2_seed_block<T: Scalar>(
    space: Su2Space,
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    b: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<Su2TtTangent<T>> {
    (0..b)
        .map(|_| {
            let dir: Su2TtVector<T> = randomized_su2_state(space, 1, 2, rng);
            Su2TtTangent::project(base_l, base_r, &dir)
        })
        .collect()
}

fn assert_diag_sane(diag: &SympDiagnostics, iterations: usize) {
    assert_eq!(
        diag.survival.len(),
        iterations,
        "one survival ratio per ACCEPTED basis vector: {} vs info.iterations={iterations}",
        diag.survival.len()
    );
    for (i, &s) in diag.survival.iter().enumerate() {
        assert!(
            s.is_finite() && s > 0.0,
            "survival[{i}] = {s:.6e} is not a positive finite ratio"
        );
    }
    for (i, &a) in diag.collapse_angle.iter().enumerate() {
        assert!(
            a.is_finite() && (0.0..=1.0 + 1e-12).contains(&a),
            "collapse_angle[{i}] = {a:.6e} is not a normalized |cos| in [0, 1]"
        );
    }
    assert!(diag.outer_iterations >= 1, "at least one outer iteration");
    assert_eq!(
        diag.gram_min_eig.len(),
        diag.outer_iterations,
        "one basis-Gram λ_min per outer iteration: {} vs outer_iterations={}",
        diag.gram_min_eig.len(),
        diag.outer_iterations
    );
    for (i, &g) in diag.gram_min_eig.iter().enumerate() {
        assert!(
            g > 0.99,
            "gram_min_eig[{i}] = {g:.6e}: the kernel ASSUMES the basis W-Gram is I, so λ_min ≈ 1 \
             on a healthy fixture (signed value — a small or negative one is the degradation \
             signature)"
        );
    }
}

#[test]
fn tsd0_su2_davidson_d0_matches_dense() {
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D3, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense SU(2) oracle unstable");
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71D0);
    let k = 3usize.min(dense.len());
    let v0s = su2_seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
            kernel,
            ..Default::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        println!(
            "tsd0_su2_davidson_d0 {kernel:?}: built={} applies={} residuals={:?}",
            so.iterations,
            so.applies,
            so.residuals()
        );
        assert!(
            so.converged,
            "{kernel:?}: SU(2) D0 arm must converge at this fixture"
        );
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: SU(2) D0 davidson {wi} vs dense {wd}"
            );
        }
    }
}

#[test]
fn tsd0_su2_instrument_smoke() {
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D3, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense SU(2) oracle unstable");
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71D1);
    let k = 3usize.min(dense.len());
    let v0s = su2_seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    let opts = SymplecticDavidsonOpts {
        precond: SympPrecond::Off,
        instrument: true,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
    let RpaReport::Signed { diag, .. } = &so.report else {
        panic!("the signed kernel must have run: {:?}", so.report);
    };
    let diag = diag
        .as_ref()
        .expect("the SU(2) Davidson wrapper inherits the kernel's SympDiagnostics");
    println!(
        "tsd0_su2_instrument_smoke: built={} applies={} outer={} restarts={} drops r/w/z/t={}/{}/{}/{}",
        so.iterations,
        so.applies,
        diag.outer_iterations,
        diag.restarts,
        diag.drops_rank,
        diag.drops_wnull,
        diag.drops_zero,
        diag.drops_triple
    );
    assert!(so.converged, "SU(2) Off lane must converge at this fixture");
    assert_diag_sane(diag, so.iterations);
    assert_eq!(
        diag.restarts, 0,
        "this SU(2) fixture's reduced tangent space is far below the auto subspace bound — no \
         thick restart can fire"
    );
}
