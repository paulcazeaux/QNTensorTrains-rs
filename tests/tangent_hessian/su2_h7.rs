// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::Scalar;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use qntt::su2::hess::weingarten_operator;
use qntt::su2::qn::Su2Space;
use qntt::su2::vector::Su2TtVector;
use qntt_testkit::gad_mol::su2_grad_exact_at;
use qntt_testkit::generic_su2_h;
use qntt_testkit::generic_su2_h_c64;
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
) -> Vec<qntt::su2::tangent::Su2TtTangent<T>> {
    use qntt::su2::tangent::Su2TtTangent;
    (0..b)
        .map(|_| {
            let dir: Su2TtVector<T> = randomized_su2_state(space, 1, 2, rng);
            Su2TtTangent::project(base_l, base_r, &dir)
        })
        .collect()
}

#[test]
fn h7_davidson_matches_dense() {
    use qntt::solvers::hess::{RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum};
    use qntt::su2::hess::{symplectic_eigenvalues, symplectic_lowest_signed_davidson};
    use qntt::su2::tangent::Su2TtTangent;
    use qntt::su2::weingarten::{tangent_basis, tangent_dim};
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D3, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    assert_eq!(
        tangent_dim(&base_l),
        n + 1,
        "SU(2) structural dim vs enumerator span"
    );
    let mut probe_rng = Xoshiro256PlusPlus::seed_from_u64(0xE6A0);
    let probe: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut probe_rng);
    let probe = Su2TtTangent::project(&base_l, &base_r, &probe);
    assert!(
        ctx.b_apply(&probe).norm() > 1e-8 * probe.norm(),
        "B vacuous at the truncated SU(2) minimizer"
    );
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense SU(2) oracle unstable");
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71B1);
    let k = 3usize.min(dense.len());
    let v0s = su2_seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        println!(
            "h7_davidson_f64 {kernel:?}: n={n} built={} applies={} residuals={:?}",
            so.iterations,
            so.applies,
            so.residuals()
        );
        assert!(so.converged, "{kernel:?}: not converged");
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        assert_eq!(omega.len(), k, "{kernel:?}");
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: davidson {wi} vs dense {wd}"
            );
        }
        assert!(
            omega[0] > dense[0] * (1.0 - 1e-6),
            "{kernel:?}: spurious mode below dense ω₁: {} vs {}",
            omega[0],
            dense[0]
        );
    }
}

#[test]
fn h7_davidson_matches_dense_c64() {
    use qntt::C64;
    use qntt::solvers::hess::{RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum};
    use qntt::su2::hess::{symplectic_eigenvalues_c64, symplectic_lowest_signed_davidson_c64};
    use qntt::su2::tangent::Su2TtTangent;
    use qntt::su2::weingarten::tangent_basis;
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h_c64(space), space, 0xD0D6, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues_c64(&ctx, &basis)
    else {
        panic!("dense SU(2) C64 oracle unstable");
    };
    let mut doubled: Vec<f64> = dense.iter().flat_map(|&w| [w, w]).collect();
    doubled.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let mut radial = Su2TtTangent::project(&base_l, &base_r, &x);
    let rn = radial.norm();
    assert!(rn > 0.9, "radial mode must be a unit tangent vector at x");
    radial.scale_mut(C64::from_f64(1.0 / rn));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71B2);
    let k = 4usize;
    let mut v0s = su2_seed_block::<C64>(space, &base_l, &base_r, k + 2, &mut rng);
    for v in &mut v0s {
        v.axpy(C64::new(3.0, 4.0), &radial);
    }
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Rr)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson_c64(&ctx, &v0s, k, &opts);
        println!(
            "h7_davidson_c64 {kernel:?}: built={} applies={} residuals={:?}",
            so.iterations,
            so.applies,
            so.residuals()
        );
        assert!(so.converged, "{kernel:?}: not converged");
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        assert_eq!(omega.len(), k, "{kernel:?}");
        for (j, (wi, wd)) in omega.iter().zip(doubled.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: davidson {wi} vs doubled dense {wd}"
            );
        }
        assert!(
            omega[0] > dense[0] * (1.0 - 1e-6),
            "{kernel:?}: spurious mode below dense ω₁ after radial contamination: {} vs {}",
            omega[0],
            dense[0]
        );
    }
}

#[test]
fn h7_signed_davidson_c64_matches_dense_c64() {
    use qntt::C64;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, RpaModeKind, RpaReport, SymplecticDavidsonOpts, SymplecticSpectrum,
    };
    use qntt::su2::hess::{symplectic_eigenvalues_c64, symplectic_lowest_signed_davidson_c64};
    use qntt::su2::weingarten::tangent_basis;
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h_c64(space), space, 0xD0D6, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues_c64(&ctx, &basis)
    else {
        panic!("dense SU(2) C64 oracle unstable");
    };
    let mut doubled: Vec<f64> = dense.iter().flat_map(|&w| [w * w, w * w]).collect();
    doubled.sort_by(f64::total_cmp);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71B2);
    let k = 4usize;
    let v0s = su2_seed_block::<C64>(space, &base_l, &base_r, k + 2, &mut rng);
    let (i, mi) = (C64::new(0.0, 1.0), C64::new(0.0, -1.0));
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Rr)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, vecs) = symplectic_lowest_signed_davidson_c64(&ctx, &v0s, k, &opts);
        println!(
            "h7_signed_davidson_c64 {kernel:?}: built={} applies={} ω² {:?}",
            so.iterations,
            so.applies,
            so.omega2()
        );
        let paired = matches!(so.report, RpaReport::Paired { .. });
        assert_eq!(
            paired,
            kernel == RpaKernel::Paired,
            "{kernel:?}: the kernel that ran"
        );
        if !paired {
            assert_eq!(so.side(), RpaMetric::Rr, "{kernel:?}");
        }
        assert!(so.converged, "{kernel:?}: not converged");
        assert!(!so.anomalous(), "{kernel:?}: {:?}", so.report);
        let w2s = so.omega2();
        assert_eq!(w2s.len(), k, "{kernel:?}");
        assert!(
            so.modes
                .iter()
                .all(|m| m.kind == RpaModeKind::Real && m.krein > 0.0),
            "{kernel:?}: {:?}",
            so.modes
        );
        for (j, (w2, wd)) in w2s.iter().zip(&doubled).enumerate() {
            assert!(
                (w2 - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω²[{j}] {w2} vs doubled dense {wd}"
            );
        }
        let scale = w2s.iter().fold(0.0f64, |m, w| m.max(w.abs()));
        for (j, (m, &w2)) in vecs.iter().zip(&w2s).enumerate() {
            let q = &m.metric_side(RpaMetric::Rr);
            let w = ctx.hess_apply(q);
            let mut r = ctx.hess_apply(&w.scale(i)).scale(mi);
            r.axpy(C64::from_f64(-w2), q);
            let rel = r.norm() / (scale * q.norm());
            assert!(
                rel <= 1e-6,
                "{kernel:?}: mode {j} ω² {w2}: product residual {rel:.2e}"
            );
            let om = w2.sqrt();
            let mut p = m.x.clone();
            p.axpy(C64::from_f64(-1.0), &m.y);
            let mut rw = w.clone();
            rw.axpy(C64::from_f64(-om), &p);
            let mut rr = ctx.hess_apply(&p.scale(i)).scale(mi);
            rr.axpy(C64::from_f64(-om), q);
            let pq = (p.norm().powi(2) + q.norm().powi(2)).sqrt();
            let eta = (rw.norm().powi(2) + rr.norm().powi(2)).sqrt() / (om * pq);
            let krein = m.x.norm().powi(2) - m.y.norm().powi(2);
            println!("{kernel:?}: mode {j} pencil residual η {eta:.2e}, Krein norm {krein}");
            assert!(m.x_im.is_none(), "{kernel:?}: mode {j} has imaginary parts");
            assert!(
                (krein - 1.0).abs() <= 1e-10,
                "{kernel:?}: mode {j} Krein norm {krein}"
            );
            assert!(
                eta <= 1e-6,
                "{kernel:?}: mode {j} pencil residual {eta:.2e}"
            );
        }
    }
}

#[test]
fn h7_newton_polish_recovers_minimizer() {
    use qntt::solvers::hess::NewtonPolishOpts;
    use qntt::su2::hess::newton_polish;
    use qntt::su2::tangent::Su2TtTangent;
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D3, 1, 2);
    let lam0 = rayleigh_quotient(&ham, &x).re_s();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71D0);
    let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let mut xi = Su2TtTangent::project(&base_l, &base_r, &dir);
    let n = xi.norm();
    xi = xi.scale(3e-2 / n);
    let xp = xi.retract_hosvd(1.0);
    let gp = su2_grad_exact_at(&ham, &xp).norm();
    assert!(gp > 1e-4, "perturbation vacuous (‖grad‖={gp:.1e})");

    let opts_n = NewtonPolishOpts {
        gradtol: 1e-11,
        ..Default::default()
    };
    let (xn, info) = newton_polish(&ham, &xp, &opts_n);
    println!(
        "h7_newton: outer={} applies={} gnorm={:.2e} λ={:.12} (λ₀={lam0:.12})",
        info.outer, info.applies, info.gnorm, info.lambda
    );
    assert!(
        info.gnorm <= 1e-9,
        "polish must reach the 1e-9 class (got {:.2e}; converged flag: {})",
        info.gnorm,
        info.converged
    );
    let gn_check = su2_grad_exact_at(&ham, &xn).norm();
    assert!(
        gn_check <= 1e-9,
        "independent gradient check: {gn_check:.2e}"
    );
    assert!((info.lambda - lam0).abs() < 1e-9);
}
