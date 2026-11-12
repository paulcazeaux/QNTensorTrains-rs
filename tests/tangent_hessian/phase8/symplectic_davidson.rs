// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::core::tangent::TtTangent;
use qntt::round::RoundMode;
use qntt::solvers::riemannian::tangent_bases;
use qntt::{C64, CoeffTensor, QnSpace, Scalar, SparseHamiltonian, TtVector};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn spd_flat(n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
    let a: Vec<f64> = (0..n * n).map(|_| rng.random_range(-1.0..=1.0)).collect();
    let mut s = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = if i == j { n as f64 } else { 0.0 };
            for k in 0..n {
                acc += a[i * n + k] * a[j * n + k];
            }
            s[i * n + j] = acc;
        }
    }
    s
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (&x, &y)| m.max((x - y).abs()))
}

fn mm(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik != 0.0 {
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
    }
    c
}

fn transpose_flat(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

#[test]
fn t8m_skew_canonical_blocks() {
    use qntt::core::weingarten::skew_canonical;
    let two_n = 10usize;
    let n = two_n / 2;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71A0);
    let g: Vec<f64> = (0..two_n * two_n)
        .map(|_| rng.random_range(-1.0..=1.0))
        .collect();
    let mut k = vec![0.0f64; two_n * two_n];
    for i in 0..two_n {
        for j in 0..two_n {
            k[i * two_n + j] = 0.5 * (g[i * two_n + j] - g[j * two_n + i]);
        }
    }
    let (c, q) = skew_canonical(&k, two_n);
    assert_eq!(c.len(), n);
    assert!(c.iter().all(|&v| v > 0.0), "block values positive: {c:?}");
    assert!(c.windows(2).all(|w| w[0] <= w[1]), "ascending: {c:?}");
    let qtq = mm(&transpose_flat(&q, two_n), &q, two_n);
    let mut eye = vec![0.0f64; two_n * two_n];
    for i in 0..two_n {
        eye[i * two_n + i] = 1.0;
    }
    assert!(max_abs_diff(&qtq, &eye) < 1e-12, "QᵀQ = I");
    let qkq = mm(&transpose_flat(&q, two_n), &mm(&k, &q, two_n), two_n);
    let mut want = vec![0.0f64; two_n * two_n];
    for (j, &cj) in c.iter().enumerate() {
        want[j * two_n + (n + j)] = cj;
        want[(n + j) * two_n + j] = -cj;
    }
    assert!(
        max_abs_diff(&qkq, &want) < 1e-12,
        "QᵀKQ = [[0,C],[−C,0]] (defect {:.3e})",
        max_abs_diff(&qkq, &want)
    );
}

#[test]
fn t8m_williamson_reduced_vectors_props() {
    use qntt::core::weingarten::{williamson_reduced, williamson_reduced_vectors};
    let n = 7usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71A1);
    let hrr = spd_flat(n, &mut rng);
    let hii = spd_flat(n, &mut rng);
    let scalar = williamson_reduced(&hrr, &hii, n, 1e-10).expect("SPD fixture");
    let (omega, s_r, s_i) =
        williamson_reduced_vectors(&hrr, &hii, n, 1e-10).expect("SPD fixture (vectors)");
    assert_eq!(omega.len(), n);
    for (j, (&wv, &ws)) in omega.iter().zip(&scalar).enumerate() {
        assert!(
            (wv - ws).abs() < 1e-9 * ws.max(1.0),
            "ω[{j}]: vectors {wv} vs scalar {ws}"
        );
    }
    let mut om = vec![0.0f64; n * n];
    let mut eye = vec![0.0f64; n * n];
    for (j, &w) in omega.iter().enumerate() {
        om[j * n + j] = w;
        eye[j * n + j] = 1.0;
    }
    let srt = transpose_flat(&s_r, n);
    let sit = transpose_flat(&s_i, n);
    let scale = omega.last().copied().unwrap_or(1.0);
    assert!(
        max_abs_diff(&mm(&srt, &mm(&hrr, &s_r, n), n), &om) < 1e-9 * scale,
        "S_rᵀ H^rr S_r = Ω"
    );
    assert!(
        max_abs_diff(&mm(&sit, &mm(&hii, &s_i, n), n), &om) < 1e-9 * scale,
        "S_iᵀ H^ii S_i = Ω"
    );
    assert!(
        max_abs_diff(&mm(&srt, &s_i, n), &eye) < 1e-10,
        "S_rᵀ S_i = I"
    );
}

#[test]
fn t8m_williamson_full_vectors_props() {
    use qntt::core::weingarten::{williamson_full, williamson_full_vectors};
    let n = 5usize;
    let two_n = 2 * n;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x71A2);
    let h = spd_flat(two_n, &mut rng);
    let scalar = williamson_full(&h, two_n, 1e-10).expect("SPD fixture");
    let (omega, s) = williamson_full_vectors(&h, two_n, 1e-10).expect("SPD fixture (vectors)");
    let mut scalar_sorted = scalar.clone();
    scalar_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(omega.len(), n);
    for (j, (&wv, &ws)) in omega.iter().zip(&scalar_sorted).enumerate() {
        assert!(
            (wv - ws).abs() < 1e-9 * ws.max(1.0),
            "ω[{j}]: vectors {wv} vs scalar {ws}"
        );
    }
    let st = transpose_flat(&s, two_n);
    let mut js = vec![0.0f64; two_n * two_n];
    for i in 0..n {
        for j in 0..two_n {
            js[i * two_n + j] = s[(n + i) * two_n + j];
            js[(n + i) * two_n + j] = -s[i * two_n + j];
        }
    }
    let sjs = mm(&st, &js, two_n);
    let mut jmat = vec![0.0f64; two_n * two_n];
    for i in 0..n {
        jmat[i * two_n + (n + i)] = 1.0;
        jmat[(n + i) * two_n + i] = -1.0;
    }
    assert!(
        max_abs_diff(&sjs, &jmat) < 1e-9,
        "SᵀJS = J (defect {:.3e})",
        max_abs_diff(&sjs, &jmat)
    );
    let shs = mm(&st, &mm(&h, &s, two_n), two_n);
    let mut want = vec![0.0f64; two_n * two_n];
    for (j, &w) in omega.iter().enumerate() {
        want[j * two_n + j] = w;
        want[(n + j) * two_n + (n + j)] = w;
    }
    let scale = omega.last().copied().unwrap_or(1.0);
    assert!(
        max_abs_diff(&shs, &want) < 1e-8 * scale,
        "Sᵀ𝔥S = diag(Ω,Ω) (defect {:.3e})",
        max_abs_diff(&shs, &want)
    );
}

#[test]
fn t8m_weingarten_cached_gauges_match() {
    use qntt::core::weingarten::{
        FloorPolicy, Terms, bond_gauges, weingarten_assemble, weingarten_assemble_cached,
    };
    let (_space, _x, base_l, base_r, dir1, dir2) = tangent_test_setup(0x71A3);
    let xi = TtTangent::project(&base_l, &base_r, &dir1);
    let tang = TtTangent::project(&base_l, &base_r, &dir2).to_vector();
    let zn = &dir2 - &tang;
    let floor = FloorPolicy::default_hard();
    let fresh = weingarten_assemble(&base_l, &base_r, &zn, &xi, floor, Terms::all());
    let gauges = bond_gauges(&base_l, &base_r, floor);
    let cached = weingarten_assemble_cached(&base_l, &base_r, &zn, &xi, &gauges, Terms::all());
    let mut diff = cached.clone();
    diff.axpy(-1.0, &fresh);
    assert!(
        diff.norm() <= 1e-14 * fresh.norm().max(1e-300),
        "cached vs fresh assembly: {:.3e}",
        diff.norm()
    );
}

#[test]
fn t8m_davidson_matches_dense_full_rank() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum, symplectic_eigenvalues,
        symplectic_lowest_signed_davidson, weingarten_operator,
    };
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B2, 1, 1);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense oracle unstable");
    };
    let gaps = exact_config_gaps_d3_11(&ham);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B0);
    let k = 3usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        println!(
            "t8m_davidson_full_rank {kernel:?}: built={} applies={} residuals={:?}",
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

        for (j, (wi, wg)) in omega.iter().zip(gaps.iter()).enumerate() {
            assert!(
                (wi - wg).abs() < 1e-6 * wg.max(1.0),
                "{kernel:?}: omega[{j}]: iterative {wi} vs exact configuration gap {wg}"
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
fn t8m_davidson_matches_dense_truncated() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum, hessian_grams,
        symplectic_from_grams, symplectic_lowest_signed_davidson, weingarten_operator,
    };
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
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B1);
    let k = 4usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        println!(
            "t8m_davidson_truncated {kernel:?}: n={n} built={} applies={} residuals={:?}",
            so.iterations,
            so.applies,
            so.residuals()
        );
        assert!(so.converged, "{kernel:?}: not converged");
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: davidson {wi} vs dense {wd}"
            );
        }
    }
}

#[test]
fn t8m_signed_davidson_c64_matches_dense_c64() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, RpaModeKind, RpaReport, SymplecticDavidsonOpts, SymplecticSpectrum,
        hessian_grams_c64, symplectic_from_grams_c64, symplectic_lowest_signed_davidson_c64,
        weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C5, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    let (a, b) = hessian_grams_c64(&ctx, &basis);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_from_grams_c64(&a, &b, n, 1e-10)
    else {
        panic!("dense oracle unstable");
    };
    let bmax = b.iter().fold(0.0f64, |acc, v| acc.max(v.abs_s()));
    assert!(bmax > 1e-8, "B vacuous — fixture proves nothing");
    let mut doubled: Vec<f64> = dense.iter().flat_map(|&w| [w * w, w * w]).collect();
    doubled.sort_by(f64::total_cmp);
    let mut doubled_omega: Vec<f64> = dense.iter().flat_map(|&w| [w, w]).collect();
    doubled_omega.sort_by(f64::total_cmp);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B2);
    let k = 6usize;
    let v0s = seed_block::<C64>(space, &base_l, &base_r, k + 2, &mut rng);
    let (i, mi) = (C64::new(0.0, 1.0), C64::new(0.0, -1.0));
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Rr)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, vecs) = symplectic_lowest_signed_davidson_c64(&ctx, &v0s, k, &opts);
        println!(
            "t8m_signed_davidson_c64 {kernel:?}: n={n} built={} applies={} ω² {:?}",
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
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        for (j, (wi, wd)) in omega.iter().zip(&doubled_omega).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}] {wi} vs doubled dense {wd}"
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
fn t8m_signed_davidson_precond_off_matches_dense() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SympPrecond, SymplecticDavidsonOpts, SymplecticSpectrum,
        symplectic_eigenvalues, symplectic_lowest_signed_davidson, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense oracle unstable");
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B3);
    let k = 4usize;
    let _: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            precond: SympPrecond::Off,
            kernel,
            ..Default::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        assert!(so.converged, "{kernel:?}: not converged");
        let om_d = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        for wd in &om_d {
            assert!(
                dense.iter().any(|wl| (wd - wl).abs() < 1e-6 * wl.max(1.0)),
                "{kernel:?}: davidson ω {wd} not in the dense spectrum {dense:?}"
            );
        }
        assert!(
            (om_d[0] - dense[0]).abs() < 1e-6 * dense[0].max(1.0),
            "{kernel:?}: lowest ω disagrees: {} vs {}",
            om_d[0],
            dense[0]
        );
    }
}

#[test]
fn t8m_davidson_restart_matches_dense() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, RpaReport, SymplecticDavidsonOpts, SymplecticSpectrum,
        symplectic_eigenvalues, symplectic_lowest_signed_davidson, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense oracle unstable");
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B4);
    let k = 2usize;
    let b = 2usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, b, &mut rng);
    let opts = SymplecticDavidsonOpts {
        max_subspace: k + b + 1,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        ..Default::default()
    };
    let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
    let RpaReport::Signed { diag, .. } = &so.report else {
        panic!("the signed kernel must have run: {:?}", so.report);
    };
    println!(
        "t8m_davidson_restart: built={} applies={} restarts={:?} residuals={:?}",
        so.iterations,
        so.applies,
        diag.as_ref().map(|d| d.restarts),
        so.residuals()
    );
    assert!(so.converged, "restarted run must still converge");
    let omega = so
        .omega()
        .expect("a converged stable solve publishes its ω");
    for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
        assert!(
            (wi - wd).abs() < 1e-6 * wd.max(1.0),
            "ω[{j}]: restarted davidson {wi} vs dense {wd}"
        );
    }
}

#[test]
fn t8m_davidson_radial_contamination_guard() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum,
        symplectic_eigenvalues_c64, symplectic_lowest_signed_davidson_c64, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C5, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues_c64(&ctx, &basis)
    else {
        panic!("dense oracle unstable");
    };
    let mut doubled: Vec<f64> = dense.iter().flat_map(|&w| [w, w]).collect();
    doubled.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let radial = {
        let mut r = TtTangent::project(&base_l, &base_r, &base_l);
        let n = r.norm();
        r.scale_mut(C64::new(0.0, 10.0 / n));
        r
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B5);
    let k = 4usize;
    let mut v0s = seed_block::<C64>(space, &base_l, &base_r, k + 2, &mut rng);
    for v in &mut v0s {
        v.axpy(C64::new(1.0, 0.0), &radial);
    }
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Rr)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson_c64(&ctx, &v0s, k, &opts);
        assert!(so.converged, "{kernel:?}: not converged");
        let omega = so.omega().unwrap_or_else(|| {
            panic!(
                "BUG: contaminated seeds must not destabilize a certified-stable fixture \
                 ({kernel:?}): {:?}",
                so.report
            )
        });
        assert!(
            omega[0] > dense[0] * (1.0 - 1e-6),
            "{kernel:?}: spurious mode below dense ω₁ after radial contamination: {} vs {}",
            omega[0],
            dense[0]
        );
        for (j, (wi, wd)) in omega.iter().zip(doubled.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: contaminated davidson {wi} vs doubled dense {wd}"
            );
        }
    }
}

#[test]
fn t8m_davidson_radial_contamination_guard_real() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum, symplectic_eigenvalues,
        symplectic_lowest_signed_davidson, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense oracle unstable");
    };
    let radial = {
        let mut r = TtTangent::project(&base_l, &base_r, &base_l);
        let n = r.norm();
        r.scale_mut(10.0 / n);
        r
    };
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81C3);
    let k = 3usize;
    let mut v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for v in &mut v0s {
        v.axpy(1.0, &radial);
    }
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        assert!(so.converged, "{kernel:?}: not converged");
        let omega = so.omega().unwrap_or_else(|| {
            panic!(
                "BUG: contaminated seeds must not destabilize a certified-stable fixture \
                 ({kernel:?}): {:?}",
                so.report
            )
        });
        assert_eq!(omega.len(), k, "{kernel:?}");
        assert!(
            omega[0] > dense[0] * (1.0 - 1e-6),
            "{kernel:?}: spurious mode below dense ω₁ after radial contamination: {} vs {}",
            omega[0],
            dense[0]
        );
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: contaminated davidson {wi} vs dense {wd}"
            );
        }
    }
}

#[test]
fn t8m_block_resolves_multiplicity() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum, symplectic_eigenvalues,
        symplectic_lowest_signed_davidson, weingarten_operator,
    };
    use qntt::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd};
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q) in &[(0usize, 1usize), (1, 2), (0, 2)] {
        t.set(&[p, q], -1.0);
        t.set(&[q, p], -1.0);
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(&[p, p, q, q], 0.4);
            }
        }
    }
    let ham = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B6);
    let x0: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    let mut opts_r = RiemannianOpts::new(&space);
    opts_r.eps = 1e-13;
    opts_r.max_iter = 600;
    opts_r.gradtol = 1e-13;
    let (_l, xg) = riemannian_gd(&ham, &x0, &opts_r);
    let (_l, mut x) = riemannian_cg(&ham, &xg, &opts_r);
    for _ in 0..5 {
        if grad_exact_at(&ham, &x).norm() < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(&ham, &x, &opts_r);
        x = x2;
    }
    normalize_mut(&mut x);
    let gnorm = grad_exact_at(&ham, &x).norm();
    assert!(gnorm < 1e-9, "not critical: ‖grad‖={gnorm:.3e}");
    let (base_l, base_r) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense oracle unstable");
    };
    assert_eq!(dense.len(), 8, "full-manifold tangent dim");
    let doublet_at = (0..3).find(|&i| (dense[i + 1] - dense[i]).abs() < 1e-8 * dense[i].max(1.0));
    let Some(doublet_at) = doublet_at else {
        panic!("C3v fixture has no degenerate pair in the lowest 4 ω: {dense:?}");
    };
    let _: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    let k = 4usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        assert!(so.converged, "{kernel:?}: not converged");
        let omega = so
            .omega()
            .expect("a converged stable solve publishes its ω");
        assert_eq!(omega.len(), k, "{kernel:?}");
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-7 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: block davidson {wi} vs raw dense {wd} (doublet at \
                 {doublet_at})"
            );
        }
    }
}

#[test]
fn t8m_davidson_ghost_regression_lih() {
    use qntt::core::tangent::{tangent_basis, tangent_dim};
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum, symplectic_eigenvalues,
        symplectic_lowest_signed_davidson, weingarten_operator,
    };

    let (space, ham, x, mut rng) = lih_sto3g_ghost_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let (base_l, base_r) = {
        let (l, r) = ctx.bases();
        (l.clone(), r.clone())
    };
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    assert_eq!(tangent_dim(&base_l), n + 1, "dim formula at the LiH grid");
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense oracle must be stable at the LiH rank-2 minimizer");
    };

    let k = 4usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        println!(
            "t8m_davidson_lih {kernel:?}: n={n} built={} applies={} residuals={:?}",
            so.iterations,
            so.applies,
            so.residuals()
        );
        assert!(so.converged, "{kernel:?}: not converged");
        assert!(
            so.iterations <= n,
            "{kernel:?}: Krylov dimension {} exceeded the {n}-dim tangent space (ghost directions)",
            so.iterations
        );
        assert!(
            !so.anomalous(),
            "BUG REGRESSION ({kernel:?}): anomalous solve at a certified-stable molecular point: \
             {:?}",
            so.report
        );
        let omega = so.omega().unwrap_or_else(|| {
            panic!(
                "BUG REGRESSION ({kernel:?}): ω withheld at a certified-stable molecular point: \
                 ω² {:?}",
                so.omega2()
            )
        });
        for (j, (wi, wd)) in omega.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: ω[{j}]: davidson {wi} vs dense {wd}"
            );
        }
    }
}

#[test]
fn t8l_tangent_dim_matches_enumerator() {
    use qntt::core::tangent::{tangent_basis, tangent_dim};
    for (d, nup, ndn, seed, lo, hi) in [
        (3usize, 1usize, 1usize, 0xB0B2u64, 1usize, 1usize),
        (4, 2, 1, 0xB0B1, 1, 2),
    ] {
        let space = QnSpace::new(d, nup, ndn);
        let (_ham, _x, base_l, base_r) = weingarten_minimizer(space, seed, lo, hi);
        let basis = tangent_basis(&base_l, &base_r, true);
        assert_eq!(
            tangent_dim(&base_l),
            basis.len() + 1,
            "structural dim vs enumerator span at d={d} ({nup},{ndn})"
        );
    }
}

#[test]
fn t8m_williamson_full_refuses_indefinite() {
    use qntt::core::weingarten::{WilliamsonError, williamson_full};
    let h = vec![1.0, 0.0, 0.0, -1.0];
    match williamson_full(&h, 2, 1e-12) {
        Err(WilliamsonError::HLrNotPd { min_eig }) => {
            assert!(
                (min_eig + 1.0).abs() < 1e-12,
                "min_eig should be the -1 eigenvalue, got {min_eig}"
            );
        }
        other => panic!("expected HLrNotPd, got {other:?}"),
    }
}

#[test]
fn t8t_signed_stable_matches_dense() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum, hessian_grams,
        symplectic_from_grams, symplectic_lowest_signed_davidson, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    let (a, b) = hessian_grams(&ctx, &basis);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_from_grams(&a, &b, n, 1e-10)
    else {
        panic!("dense oracle unstable at the certified minimizer");
    };

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x519E);
    let k = 3usize;
    let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
    let opts = SymplecticDavidsonOpts {
        kernel: RpaKernel::Signed {
            metric: RpaMetric::Auto,
            probe_budget: 800,
        },
        ..SymplecticDavidsonOpts::default()
    };
    let (signed, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);

    assert_eq!(
        signed.side(),
        RpaMetric::Ii,
        "Auto at a stable point must keep the historical Ii assignment"
    );
    assert!(
        signed.all_negatives_certified(),
        "vacuous certification must hold with no negatives"
    );
    let w2s = signed.omega2();
    assert_eq!(w2s.len(), k);
    let omega = signed
        .omega()
        .expect("a converged stable solve publishes its ω");
    for (j, &w2) in w2s.iter().enumerate() {
        assert!(
            w2 > 0.0,
            "stable fixture produced a negative ω²[{j}] = {w2}"
        );
        let w = omega[j];
        assert!(
            (w - dense[j]).abs() < 1e-6 * dense[j].max(1.0),
            "ω[{j}]: signed {w} vs dense {}",
            dense[j]
        );
    }
}

#[test]
fn t8t_select_rpa_metric_routing() {
    use qntt::core::tangent::TtTangent;
    use qntt::solvers::hess::{PdVerdict, RpaMetric, select_rpa_metric, weingarten_operator};
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);

    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x51E7);
    let v0 = seed_block::<f64>(space, &base_l, &base_r, 1, &mut rng)
        .pop()
        .expect("one seed");
    let (metric, ii, rr) = select_rpa_metric(&ctx, &v0, 800);
    assert_eq!(metric, RpaMetric::Ii, "stable point must keep Ii");
    assert!(
        !matches!(ii, PdVerdict::WitnessNegative { .. }),
        "no negative witness at a certified minimizer, got {ii:?}"
    );
    assert!(rr.is_none(), "Rr must not be probed when Ii is kept");

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
    let mut xr: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    xr.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    xr.scale_mut(1.0 / xr.norm());
    let ham2 = generic_truncated_h::<f64>(space);
    let ctx2 = weingarten_operator(&ham2, &xr);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let (bl2, br2) = ctx2.bases();
    let v02 = TtTangent::project(bl2, br2, &dir);
    let (metric2, ii2, rr2) = select_rpa_metric(&ctx2, &v02, 800);
    assert!(
        matches!(ii2, PdVerdict::WitnessNegative { .. }),
        "Ii must produce a variational negative witness at a non-critical point, got {ii2:?}"
    );
    assert_eq!(
        metric2,
        RpaMetric::Rr,
        "a proven-indefinite Ii must resolve to the swap"
    );
    assert!(rr2.is_some(), "the swap must carry the Rr probe's verdict");
}

#[test]
fn t8t_williamson_reduced_refuses_by_variant() {
    use qntt::core::weingarten::{WilliamsonError, williamson_reduced};
    match williamson_reduced(&[1.0, 0.0, 0.0, -2.0], &[1.0, 0.0, 0.0, 1.0], 2, 1e-12) {
        Err(WilliamsonError::HrrNotPd { min_eig }) => {
            assert!((min_eig + 2.0).abs() < 1e-12, "got min_eig {min_eig}");
        }
        other => panic!("expected HrrNotPd, got {other:?}"),
    }
    match williamson_reduced(&[1.0, 0.0, 0.0, 1.0], &[4.0, 0.0, 0.0, -3.0], 2, 1e-12) {
        Err(WilliamsonError::OmegaSqNegative { min_eig }) => {
            assert!((min_eig + 3.0).abs() < 1e-12, "got min_eig {min_eig}");
        }
        other => panic!("expected OmegaSqNegative, got {other:?}"),
    }
}

#[test]
fn t8m_signed_davidson_c64_solves_an_indefinite_point() {
    use qntt::core::tangent::TtTangent;
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        RpaModeKind, RpaReport, SymplecticDavidsonOpts, SymplecticSpectrum, hessian_grams_c64,
        symplectic_from_grams_c64, symplectic_lowest_signed_davidson_c64, weingarten_operator,
    };
    let space = QnSpace::new(4, 2, 1);
    let ham = generic_complex_h(space);
    let k = 6usize;
    for state in [0u64, 3] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC1D0 + state);
        let mut x: TtVector<C64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, qntt::qn::Pooled::NONE, RoundMode::Rel);
        let xn = x.norm();
        x.scale_mut(C64::from_f64(1.0 / xn));
        let ctx = weingarten_operator(&ham, &x);
        let (bl, br) = ctx.bases();
        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let (a, b) = hessian_grams_c64(&ctx, &basis);
        let SymplecticSpectrum::Unstable { generator_eigs, .. } =
            symplectic_from_grams_c64(&a, &b, n, 1e-10)
        else {
            panic!("state {state}: fixture premise: both factors indefinite");
        };
        let w2s: Vec<C64> = generator_eigs
            .iter()
            .filter(|z| z.re > 1e-9 || (z.re.abs() <= 1e-9 && z.im > 0.0))
            .map(|z| z * z)
            .collect();
        let scale = w2s.iter().fold(1.0f64, |m, z| m.max(z.norm()));
        let is_real = |z: &C64| z.im.abs() <= 1e-9 * scale;
        let mut negs: Vec<f64> = w2s
            .iter()
            .filter(|z| is_real(z) && z.re < 0.0)
            .map(|z| z.re)
            .collect();
        let mut reals: Vec<f64> = w2s
            .iter()
            .filter(|z| is_real(z) && z.re >= 0.0)
            .map(|z| z.re)
            .collect();
        negs.sort_by(f64::total_cmp);
        reals.sort_by(f64::total_cmp);
        let want: Vec<f64> = negs
            .iter()
            .chain(reals.iter().take(k / 2))
            .flat_map(|&w| [w, w])
            .collect();
        let window = reals[k / 2 - 1];
        let mut cplx: Vec<C64> = w2s
            .iter()
            .filter(|z| !is_real(z) && z.re <= window)
            .copied()
            .collect();
        cplx.sort_by(|p, q| p.re.total_cmp(&q.re));
        assert!(
            !negs.is_empty() && !cplx.is_empty(),
            "state {state}: fixture premise"
        );
        let v0s = seed_block::<C64>(space, bl, br, k + 2, &mut rng);
        let (so, amps) = symplectic_lowest_signed_davidson_c64(
            &ctx,
            &v0s,
            k,
            &SymplecticDavidsonOpts::default(),
        );
        let complex: Vec<C64> = so.complex().map(|m| m.omega2).collect();
        println!(
            "state {state}: n {n} converged {} ω² {:?} vs {want:?}; complex {complex:?} vs {cplx:?}",
            so.converged,
            so.omega2(),
        );
        assert!(matches!(so.report, RpaReport::Paired { .. }));
        assert!(so.converged, "state {state}: not converged");
        let w2 = so.omega2();
        assert_eq!(w2.len(), want.len(), "state {state}: {w2:?} vs {want:?}");
        for (j, (g, w)) in w2.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-6 * w.abs().max(1.0),
                "state {state}: ω²[{j}] {g} vs dense {w}"
            );
        }
        assert!(
            so.all_negatives_certified(),
            "state {state}: an uncertified negative"
        );
        assert_eq!(
            complex.len(),
            cplx.len(),
            "state {state}: complex pairs in the window"
        );
        for c in &complex {
            assert!(
                cplx.iter().any(|z| (c.re - z.re).abs() <= 1e-6 * scale
                    && (c.im - z.im.abs()).abs() <= 1e-6 * scale),
                "state {state}: complex ({}, {}) not in the dense window {cplx:?}",
                c.re,
                c.im
            );
        }
        let mut radial = TtTangent::project(bl, br, bl);
        radial.scale_mut(C64::from_f64(1.0 / radial.norm()));
        let deflate = |mut v: TtTangent<C64>| {
            for _ in 0..2 {
                let c = radial.dot(&v);
                v.axpy(-c, &radial);
            }
            v
        };
        let (i, mi) = (C64::new(0.0, 1.0), C64::new(0.0, -1.0));
        let w_op = |v: &TtTangent<C64>| deflate(ctx.hess_apply(v));
        let r_op = |v: &TtTangent<C64>| deflate(ctx.hess_apply(&v.scale(i)).scale(mi));
        let comb = |terms: &[(f64, &TtTangent<C64>)]| {
            let mut v = terms[0].1.scale(C64::from_f64(terms[0].0));
            for (c, t) in &terms[1..] {
                v.axpy(C64::from_f64(*c), t);
            }
            v
        };
        let mut checked = 0usize;
        for (m, a) in so.modes.iter().zip(&amps) {
            if m.kind == RpaModeKind::Real {
                continue;
            }
            let (Some(x_im), Some(y_im)) = (&a.x_im, &a.y_im) else {
                panic!("state {state}: a {:?} mode without imaginary parts", m.kind)
            };
            let q = (
                comb(&[(1.0, &a.x), (1.0, &a.y)]),
                comb(&[(1.0, x_im), (1.0, y_im)]),
            );
            let p = (
                comb(&[(1.0, &a.x), (-1.0, &a.y)]),
                comb(&[(1.0, x_im), (-1.0, y_im)]),
            );
            let (wr, wi) = (m.omega.re, m.omega.im);
            let half = |op: &dyn Fn(&TtTangent<C64>) -> TtTangent<C64>,
                        u: &(TtTangent<C64>, TtTangent<C64>),
                        v: &(TtTangent<C64>, TtTangent<C64>)| {
                let mut re = op(&u.0);
                re.axpy(C64::from_f64(-wr), &v.0);
                re.axpy(C64::from_f64(wi), &v.1);
                let mut im = op(&u.1);
                im.axpy(C64::from_f64(-wr), &v.1);
                im.axpy(C64::from_f64(-wi), &v.0);
                re.norm().powi(2) + im.norm().powi(2)
            };
            let res = (half(&w_op, &q, &p) + half(&r_op, &p, &q)).sqrt();
            let pq =
                (p.0.norm().powi(2) + p.1.norm().powi(2) + q.0.norm().powi(2) + q.1.norm().powi(2))
                    .sqrt();
            let eta = res / (m.omega.norm() * pq);
            println!(
                "state {state}: {:?} mode ω {:?} pencil residual η {eta:.2e} (reported {:.2e})",
                m.kind, m.omega, m.residual
            );
            assert!(eta <= 1e-6, "state {state}: {:?} mode η {eta:.2e}", m.kind);
            checked += 1;
        }
        assert!(
            checked >= 2,
            "state {state}: no imaginary or complex mode checked"
        );
    }
}
