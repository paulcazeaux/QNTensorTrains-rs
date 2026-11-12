// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s6::util::*;
use qntt::round::RoundMode;
use qntt::solvers::hess::ChannelGsOpts;
use qntt::solvers::mals::MalsOptions;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::hess::{channel_ground_state, weingarten_operator};
use qntt::su2::qn::{Su2RankGrid, Su2Space};
use qntt::su2::riemannian::{
    CgVariant, RiemannianOpts, riemannian_cg, riemannian_gd, riemannian_gradient,
};
use qntt::su2::solvers::mals::mals;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_csf_with_point_group, tt_randn};
use qntt::{PointGroup, Scalar};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[allow(clippy::too_many_arguments)]
mod pre_d {
    use super::*;
    use qntt::su2::hamiltonian::{rayleigh_quotient, xt_h_y};
    use qntt::su2::ops::dot;

    pub(super) const INITIAL_ROUND_EPS: f64 = 1e-14;
    pub(super) const ARMIJO_C: f64 = 1e-4;
    pub(super) const BACKTRACK_RHO: f64 = 0.5;
    pub(super) const MAX_BACKTRACK: usize = 30;
    pub(super) const ENERGY_NOISE_C: f64 = 32.0;
    pub(super) const GRAD_ACCEPT: f64 = 0.9;
    pub(super) const MAX_GRAD_TRIALS: usize = 3;

    pub(super) fn mix_seed(base: u64, k: u64) -> u64 {
        let mut z = base ^ k;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(super) fn tangent_bases<T: Scalar>(x: &Su2TtVector<T>) -> (Su2TtVector<T>, Su2TtVector<T>) {
        let mut base_l = x.clone();
        base_l.left_orthogonalize(true);
        let mut base_r = x.clone();
        base_r.right_orthogonalize(true);
        (base_l, base_r)
    }

    pub(super) fn normalize_mut<T: Scalar>(x: &mut Su2TtVector<T>) {
        let n = x.norm_lr(false);
        if n != 0.0 {
            x.scale_mut(T::from_f64(1.0 / n));
        }
    }

    pub(super) fn riemannian_gd<T: Scalar>(
        h: &Su2SparseHamiltonian<T>,
        x0: &Su2TtVector<T>,
        opts: &RiemannianOpts,
    ) -> (f64, Su2TtVector<T>) {
        let mut x = x0.clone();
        if !opts.preserve_grid {
            x.round_mut(INITIAL_ROUND_EPS, qntt::Pooled::NONE, RoundMode::Abs);
        }
        normalize_mut(&mut x);
        let mut lambda = rayleigh_quotient(h, &x).re_s();
        let mut alpha = 1.0f64;
        let mut lambda_known: Option<f64> = None;
        for it in 0..opts.max_iter {
            let (base_l, base_r) = tangent_bases(&x);
            let seed = mix_seed(opts.seed, it as u64);
            let (grad, l) = gradient_at(h, &x, &base_l, &base_r, opts.gradtol, seed, lambda_known);
            lambda = l;
            let gnorm = grad.norm();
            if gnorm <= opts.eps * lambda.abs() {
                break;
            }
            let eta = grad.scale(T::from_f64(-1.0));
            let slope = -gnorm * gnorm;
            match linesearch(
                h,
                &x,
                &eta,
                lambda,
                slope,
                1.0f64.min(2.0 * alpha),
                gnorm,
                Some(slope),
            ) {
                None => break,
                Some((a_n, x_n, l_n)) => {
                    alpha = a_n;
                    x = x_n;
                    lambda = l_n;
                    lambda_known = Some(l_n);
                }
            }
        }
        (lambda, x)
    }

    pub(super) fn riemannian_cg<T: Scalar>(
        h: &Su2SparseHamiltonian<T>,
        x0: &Su2TtVector<T>,
        opts: &RiemannianOpts,
    ) -> (f64, Su2TtVector<T>) {
        let mut x = x0.clone();
        if !opts.preserve_grid {
            x.round_mut(INITIAL_ROUND_EPS, qntt::Pooled::NONE, RoundMode::Abs);
        }
        normalize_mut(&mut x);

        let (mut base_l, mut base_r) = tangent_bases(&x);
        let (mut grad, mut lambda) = gradient_impl(
            h,
            &x,
            &base_l,
            &base_r,
            opts.gradtol,
            mix_seed(opts.seed, 0),
        );
        let mut gnorm = grad.norm();
        let mut eta = grad.scale(T::from_f64(-1.0));
        let mut alpha = 1.0f64;
        let trace = std::env::var_os("QNTT_CG_TRACE").is_some();
        let mut iters_run = 0usize;
        let mut exit = "max_iter";

        for it in 0..opts.max_iter {
            if trace {
                eprintln!(
                    "CG it={it} λ={lambda:.16e} ‖g‖={gnorm:.6e} stop={:.6e}",
                    opts.eps * lambda.abs()
                );
            }
            if gnorm <= opts.eps * lambda.abs() {
                exit = "gnorm";
                break;
            }
            iters_run = it + 1;

            let mut slope = grad.dot(&eta).re_s();
            if slope >= 0.0 {
                eta = grad.scale(T::from_f64(-1.0));
                slope = -gnorm * gnorm;
            }

            let lambda_known;
            match linesearch(
                h,
                &x,
                &eta,
                lambda,
                slope,
                1.0f64.min(2.0 * alpha),
                gnorm,
                Some(slope),
            ) {
                None => {
                    exit = "linesearch";
                    break;
                }
                Some((a_n, x_n, l_n)) => {
                    alpha = a_n;
                    x = x_n;
                    lambda_known = Some(l_n);
                }
            }

            let (bl_n, br_n) = tangent_bases(&x);
            base_l = bl_n;
            base_r = br_n;
            let (gradn, l) = gradient_at(
                h,
                &x,
                &base_l,
                &base_r,
                opts.gradtol,
                mix_seed(opts.seed, it as u64 + 1),
                lambda_known,
            );
            lambda = l;
            let gnormn = gradn.norm();

            let eta_t = Su2TtTangent::project(&base_l, &base_r, &eta.to_vector());
            let beta = match opts.variant {
                CgVariant::Fr => gnormn * gnormn / (gnorm * gnorm),
                CgVariant::Pr => {
                    let g_t = Su2TtTangent::project(&base_l, &base_r, &grad.to_vector());
                    let num = gradn.dot(&gradn.sub(&g_t)).re_s();
                    (num / (gnorm * gnorm)).max(0.0)
                }
            };
            eta = gradn
                .scale(T::from_f64(-1.0))
                .add(&eta_t.scale(T::from_f64(beta)));
            grad = gradn;
            gnorm = gnormn;
        }
        if trace {
            eprintln!(
                "CG exit={exit} after {iters_run}/{} iterations, ‖g‖={gnorm:.6e}, λ={lambda:.16e}",
                opts.max_iter
            );
        }
        (lambda, x)
    }

    pub(super) fn riemannian_gradient<T: Scalar>(
        h: &Su2SparseHamiltonian<T>,
        x: &Su2TtVector<T>,
        base_l: &Su2TtVector<T>,
        base_r: &Su2TtVector<T>,
        tol: f64,
        seed: u64,
    ) -> (Su2TtTangent<T>, f64) {
        gradient_impl(h, x, base_l, base_r, tol, seed)
    }

    pub(super) fn gradient_impl<T: Scalar>(
        h: &Su2SparseHamiltonian<T>,
        x: &Su2TtVector<T>,
        base_l: &Su2TtVector<T>,
        base_r: &Su2TtVector<T>,
        tol: f64,
        seed: u64,
    ) -> (Su2TtTangent<T>, f64) {
        gradient_at(h, x, base_l, base_r, tol, seed, None)
    }

    pub(super) fn gradient_at<T: Scalar>(
        h: &Su2SparseHamiltonian<T>,
        x: &Su2TtVector<T>,
        base_l: &Su2TtVector<T>,
        base_r: &Su2TtVector<T>,
        tol: f64,
        seed: u64,
        lambda_known: Option<f64>,
    ) -> (Su2TtTangent<T>, f64) {
        let _ = (tol, seed);
        let lambda = match lambda_known {
            Some(l) => l,
            _ => rayleigh_quotient(h, x).re_s(),
        };
        let grad = qntt::su2::hess::riemannian_gradient_env(h, base_l, base_r, lambda);
        (grad, lambda)
    }

    pub(super) fn linesearch<T: Scalar>(
        h: &Su2SparseHamiltonian<T>,
        x: &Su2TtVector<T>,
        eta: &Su2TtTangent<T>,
        lambda0: f64,
        slope: f64,
        alpha0: f64,
        gnorm0: f64,
        gdot: Option<f64>,
    ) -> Option<(f64, Su2TtVector<T>, f64)> {
        let w = eta.to_vector();
        let s1 = dot(x, &w).re_s();
        let s2 = dot(&w, &w).re_s();
        let c1 = match gdot {
            Some(g) => g + lambda0 * s1,
            _ => xt_h_y(x, h, &w).re_s(),
        };
        let c2 = xt_h_y(&w, h, &w).re_s();
        let a = c2 * s1 - c1 * s2;
        let b = c2 - lambda0 * s2;
        let c = c1 - lambda0 * s1;
        let pred = |al: f64| {
            (lambda0 + 2.0 * c1 * al + c2 * al * al) / (1.0 + 2.0 * s1 * al + s2 * al * al)
        };

        let mut cand: Vec<f64> = Vec::new();
        if a.abs() > f64::EPSILON * b.abs().max(c.abs()).max(1.0) {
            let disc = b * b - 4.0 * a * c;
            if disc >= 0.0 {
                cand.push((-b + disc.sqrt()) / (2.0 * a));
                cand.push((-b - disc.sqrt()) / (2.0 * a));
            }
        } else if b.abs() > 0.0 {
            cand.push(-c / b);
        }
        cand.retain(|&al| al > 0.0 && (1.0 + 2.0 * s1 * al + s2 * al * al) > 0.0);
        let mut alpha = if cand.is_empty() {
            alpha0
        } else {
            *cand
                .iter()
                .min_by(|p, q| {
                    pred(**p)
                        .partial_cmp(&pred(**q))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
        };

        let trace = std::env::var_os("QNTT_LS_TRACE").is_some();
        let efloor = ENERGY_NOISE_C * f64::EPSILON * (1.0 + lambda0.abs());
        let mut grad_trials = 0usize;
        let mut best_alpha = 0.0f64;
        let mut best_x: Option<Su2TtVector<T>> = None;
        let mut best_lambda = lambda0;
        for bt in 0..MAX_BACKTRACK {
            let mut xa = eta.retract_hosvd(T::from_f64(alpha));
            normalize_mut(&mut xa);
            let la = rayleigh_quotient(h, &xa).re_s();
            if trace {
                eprintln!(
                    "LS bt={bt} alpha={alpha:.3e} λ0={lambda0:.16e} λa={la:.16e} \
                     dλ={:.3e} armijo_rhs_delta={:.3e} slope={slope:.3e}",
                    la - lambda0,
                    ARMIJO_C * alpha * slope,
                );
            }
            if la <= lambda0 + (ARMIJO_C * alpha * slope).min(-efloor) {
                return Some((alpha, xa, la));
            }
            if lambda0 - la >= efloor && la < best_lambda {
                best_alpha = alpha;
                best_lambda = la;
                best_x = Some(xa.clone());
            }
            if gnorm0 > 0.0 && alpha * slope.abs() < efloor && la - lambda0 <= efloor {
                if bt == 0 {
                    return Some((alpha, xa, la));
                }
                if grad_trials < MAX_GRAD_TRIALS {
                    grad_trials += 1;
                    let (bl, br) = tangent_bases(&xa);
                    let ga = qntt::su2::hess::riemannian_gradient_env(h, &bl, &br, la).norm();
                    if trace {
                        eprintln!("LS grad-trial bt={bt} gnorm0={gnorm0:.3e} gnorm_a={ga:.3e}");
                    }
                    if ga <= GRAD_ACCEPT * gnorm0 {
                        return Some((alpha, xa, la));
                    }
                }
            }
            alpha *= BACKTRACK_RHO;
        }
        if trace && best_x.is_none() {
            eprintln!("LS -> None (no trial decreased below λ0={lambda0:.16e})");
        }
        best_x.map(|xb| (best_alpha, xb, best_lambda))
    }
}

fn generic_tv(d: usize) -> (qntt::CoeffTensor<f64>, qntt::CoeffTensor<f64>) {
    let mut t = qntt::CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            let val = if i == j {
                0.15 * (i as f64 - 3.0)
            } else {
                -1.0 / (1.0 + (i as f64 - j as f64).abs())
            };
            t.set(&[i, j], val);
        }
    }
    let mut v = qntt::CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        v.set(&[p, p, p, p], 0.8);
        for q in 0..d {
            if p == q {
                continue;
            }
            let r = 1.0 + (p as f64 - q as f64).abs();
            v.set(&[p, p, q, q], 0.5 / r);
            v.set(&[p, q, q, p], 0.2 / r);
        }
    }
    (t, v)
}

fn bits_eq<T: Scalar>(a: T, b: T) -> bool {
    a.re_s().to_bits() == b.re_s().to_bits() && a.im_s().to_bits() == b.im_s().to_bits()
}

fn assert_f64_bits(got: f64, want: f64, what: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{what}: {got:.17e} vs {want:.17e} (bit-identity required — the fix is in the driver's \
         operation order, never in this test)"
    );
}

fn coeffs_bits_eq_su2<T: Scalar>(a: &Su2TtVector<T>, b: &Su2TtVector<T>, what: &str) {
    assert_eq!(a.space, b.space, "{what}: space");
    assert_eq!(a.ranks.len(), b.ranks.len(), "{what}: rank grid count");
    for (k, (ra, rb)) in a.ranks.iter().zip(b.ranks.iter()).enumerate() {
        assert_eq!(ra, rb, "{what}: rank grid at bond {k}");
    }
    assert_eq!(a.ortho, b.ortho, "{what}: ortho flag");
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        assert_eq!(
            ca.num_coeffs(),
            cb.num_coeffs(),
            "{what}: num_coeffs core {k}"
        );
        for i in 0..ca.num_coeffs() {
            assert!(
                bits_eq(ca.coeff(i), cb.coeff(i)),
                "{what}: core {k} coefficient {i}"
            );
        }
    }
}

fn tangent_bits_eq<T: Scalar>(a: &Su2TtTangent<T>, b: &Su2TtTangent<T>, what: &str) {
    assert_eq!(
        a.num_coeffs(),
        b.num_coeffs(),
        "{what}: tangent coefficient count"
    );
    for i in 0..a.num_coeffs() {
        assert!(
            bits_eq(a.coeff(i), b.coeff(i)),
            "{what}: tangent coefficient {i}"
        );
    }
}

fn fnv1a64(words: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for byte in w.to_le_bytes() {
            h ^= byte as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn digest_su2<T: Scalar>(lambda: f64, x: &Su2TtVector<T>) -> u64 {
    let mut w = vec![lambda.to_bits()];
    for k in 0..x.space.d {
        let c = x.core(k);
        for i in 0..c.num_coeffs() {
            w.push(c.coeff(i).re_s().to_bits());
            w.push(c.coeff(i).im_s().to_bits());
        }
    }
    fnv1a64(&w)
}

const RECORDED_DIGESTS_S20: [u64; 7] = [
    0xc1d2_4720_af7b_539f,
    0xbc1a_7a3d_1dcb_d4dd,
    0xa5a9_4bf1_9013_3178,
    0x1c87_e288_ccf5_c08a,
    0xa76a_a2aa_2678_b729,
    0xe37b_65ba_c3f3_a0b9,
    0xd4cd_d861_0620_cef3,
];
const RECORDED_DIGESTS_S31: [u64; 7] = [
    0xfa52_7739_25bc_4acc,
    0xe133_4ca5_f157_b762,
    0x070c_02ea_d48a_8ebf,
    0x6d55_bad8_e427_a454,
    0x1dc4_b004_0227_4347,
    0x6cd4_cab4_b7d9_c569,
    0x7c65_5c43_882d_b4d8,
];
const RECORDED_DIGEST_PIPELINE: [u64; 1] = [0x0562_d186_98fb_372d];

fn report_digests(tag: &str, got: &[u64], recorded: &[u64]) {
    for (i, (g, r)) in got.iter().zip(recorded.iter()).enumerate() {
        let verdict = if *r == 0 {
            "not yet recorded"
        } else if g == r {
            "matches the recorded aarch64 value"
        } else {
            "DIFFERS from the recorded aarch64 value (expected across ISAs; diagnostic only)"
        };
        println!("tms_d2su2 digest {tag}[{i}] = {g:#018x} — {verdict}");
    }
}

fn rows_1_to_8(
    h: &Su2SparseHamiltonian<f64>,
    x0: &Su2TtVector<f64>,
    space: Su2Space,
    tag: &str,
) -> Vec<u64> {
    let mut digests = Vec::new();
    let base = RiemannianOpts {
        eps: 1e-6,
        max_iter: 15,
        gradtol: 1e-8,
        variant: CgVariant::Pr,
        ..RiemannianOpts::new(&space)
    };

    let (l_new, x_new) = riemannian_cg(h, x0, &base);
    let (l_old, x_old) = pre_d::riemannian_cg(h, x0, &base);
    assert_f64_bits(l_new, l_old, "row 1: riemannian_cg Pr λ");
    coeffs_bits_eq_su2(&x_new, &x_old, "row 1: riemannian_cg Pr x");
    digests.push(digest_su2(l_new, &x_new));

    let fr = RiemannianOpts {
        variant: CgVariant::Fr,
        ..base.clone()
    };
    let (l_new, x_new) = riemannian_cg(h, x0, &fr);
    let (l_old, x_old) = pre_d::riemannian_cg(h, x0, &fr);
    assert_f64_bits(l_new, l_old, "row 2: riemannian_cg Fr λ");
    coeffs_bits_eq_su2(&x_new, &x_old, "row 2: riemannian_cg Fr x");
    digests.push(digest_su2(l_new, &x_new));

    let pg = RiemannianOpts {
        preserve_grid: true,
        ..base.clone()
    };
    let (l_new, x_new) = riemannian_cg(h, x0, &pg);
    let (l_old, x_old) = pre_d::riemannian_cg(h, x0, &pg);
    assert_f64_bits(l_new, l_old, "row 3: riemannian_cg preserve_grid λ");
    coeffs_bits_eq_su2(&x_new, &x_old, "row 3: riemannian_cg preserve_grid x");
    digests.push(digest_su2(l_new, &x_new));

    let (l_new, x_new) = riemannian_gd(h, x0, &base);
    let (l_old, x_old) = pre_d::riemannian_gd(h, x0, &base);
    assert_f64_bits(l_new, l_old, "row 4: riemannian_gd λ");
    coeffs_bits_eq_su2(&x_new, &x_old, "row 4: riemannian_gd x");
    digests.push(digest_su2(l_new, &x_new));

    let one = RiemannianOpts {
        eps: 0.0,
        max_iter: 1,
        ..base.clone()
    };
    let mut xr = x0.clone();
    xr.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    pre_d::normalize_mut(&mut xr);
    let lambda_start = qntt::su2::hamiltonian::rayleigh_quotient(h, &xr).re_s();
    let (l_new, x_new) = riemannian_gd(h, x0, &one);
    let (l_old, x_old) = pre_d::riemannian_gd(h, x0, &one);
    assert_f64_bits(l_new, l_old, "row 5: one GD step from far λ");
    coeffs_bits_eq_su2(&x_new, &x_old, "row 5: one GD step from far x");
    assert!(
        l_new < lambda_start - 1e-6,
        "row 5 is vacuous at {tag}: one GD step from the far start moved λ by {:.3e} (want a real \
         Armijo-accepted decrease > 1e-6)",
        lambda_start - l_new
    );
    digests.push(digest_su2(l_new, &x_new));

    let tight = RiemannianOpts {
        eps: 1e-13,
        max_iter: 600,
        gradtol: 1e-13,
        ..base.clone()
    };
    let (_lc, xstar) = pre_d::riemannian_cg(h, x0, &tight);
    let gate = weingarten_operator(h, &xstar).gradient().norm();
    println!("tms_d2su2 {tag} row 6: exact gate at the pre-D CG output = {gate:.3e}");
    assert!(
        gate < 1e-9,
        "row 6 is vacuous at {tag}: the pre-D CG did not converge (exact gate {gate:.3e} ≥ 1e-9)"
    );
    let (l_new, x_new) = riemannian_gd(h, &xstar, &one);
    let (l_old, x_old) = pre_d::riemannian_gd(h, &xstar, &one);
    assert_f64_bits(l_new, l_old, "row 6: one GD step from converged λ");
    coeffs_bits_eq_su2(&x_new, &x_old, "row 6: one GD step from converged x");
    digests.push(digest_su2(l_new, &x_new));

    let (bl, br) = pre_d::tangent_bases(&xstar);
    let (g_new, lg_new) = riemannian_gradient(h, &xstar, &bl, &br, 1e-12, 1234);
    let (g_old, lg_old) = pre_d::riemannian_gradient(h, &xstar, &bl, &br, 1e-12, 1234);
    assert_f64_bits(lg_new, lg_old, "row 7: riemannian_gradient λ");
    tangent_bits_eq(&g_new, &g_old, "row 7: riemannian_gradient g");
    let mut gw = vec![lg_new.to_bits()];
    for i in 0..g_new.num_coeffs() {
        gw.push(g_new.coeff(i).re_s().to_bits());
        gw.push(g_new.coeff(i).im_s().to_bits());
    }
    digests.push(fnv1a64(&gw));

    let lam = qntt::su2::hamiltonian::rayleigh_quotient(h, &xstar).re_s();
    let via_env = qntt::su2::hess::riemannian_gradient_env(h, &bl, &br, lam).norm();
    let via_ctx = weingarten_operator(h, &xstar).gradient().norm();
    assert_f64_bits(via_env, via_ctx, "row 8: exact gate env vs weingarten ctx");
    assert!(
        via_env.is_finite() && via_env >= 0.0,
        "row 8: gate is not a finite norm ({via_env:.3e})"
    );

    digests
}

fn rerun_with_sol_d43_off(test: &str) -> bool {
    let off = |k: &str| std::env::var(k).is_ok_and(|v| v == "0");
    if off("QNTT_LS_SECANT_ALL") && off("QNTT_LS_SLOPE_FLOOR") {
        return false;
    }
    let path = module_path!().split_once("::").map_or("", |(_, p)| p);
    let name = format!("{path}::{test}");
    let out = std::process::Command::new(std::env::current_exe().expect("test binary path"))
        .args([name.as_str(), "--exact", "--nocapture", "--test-threads=1"])
        .env("QNTT_LS_SECANT_ALL", "0")
        .env("QNTT_LS_SLOPE_FLOOR", "0")
        .output()
        .expect("spawn the pin's child process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    print!("{stdout}");
    eprint!("{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        out.status.success(),
        "{name} failed with the line-search flags off"
    );
    assert!(
        stdout.contains("test result: ok. 1 passed"),
        "the child ran no test named {name}"
    );
    true
}

#[test]
fn tms_d2su2_no_penalty_path_is_pre_phase_d_singlet() {
    if rerun_with_sol_d43_off("tms_d2su2_no_penalty_path_is_pre_phase_d_singlet") {
        return;
    }
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let (t, v) = generic_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut rng = seeded(0xD2_1001);
    let mut x0: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
    x0.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    let (n_theta, n_cw) = count_theta_cw(&x0);
    println!("tms_d2su2 (6,2,0) start: n_theta = {n_theta}, n_cw = {n_cw}");

    let digests = rows_1_to_8(&h, &x0, space, "(6,2,0)");
    report_digests("(6,2,0)", &digests, &RECORDED_DIGESTS_S20);
}

#[test]
fn tms_d2su2_no_penalty_path_is_pre_phase_d_doublet() {
    if rerun_with_sol_d43_off("tms_d2su2_no_penalty_path_is_pre_phase_d_doublet") {
        return;
    }
    let d = 6usize;
    let space = Su2Space::new(d, 3, 1);
    let (t, v) = generic_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut rng = seeded(0xD2_1002);
    let mut x0: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
    x0.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    let (n_theta, n_cw) = count_theta_cw(&x0);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "the (6,3,1) start does not exercise the weighted paths (n_theta = {n_theta}, \
         n_cw = {n_cw}); this arm would be vacuous as an SU(2) pin"
    );

    let digests = rows_1_to_8(&h, &x0, space, "(6,3,1)");
    report_digests("(6,3,1)", &digests, &RECORDED_DIGESTS_S31);
}

#[test]
fn tms_d2su2_channel_pipeline_is_pre_phase_d() {
    if rerun_with_sol_d43_off("tms_d2su2_channel_pipeline_is_pre_phase_d") {
        return;
    }
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let (t, v) = generic_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let gs = ChannelGsOpts {
        bond_ramp: Some(vec![6, 10, 14, 0, 0, 0]),
        ..ChannelGsOpts::default()
    };

    let (x_new, e_new, g_new) = channel_ground_state(&h, space, &gs);

    let caps = qntt::su2::qn::max_ranks(&space);
    let interior = 1usize;
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut g = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                g[s] = if b == 0 || b == d {
                    caps[b][s].min(1)
                } else {
                    caps[b][s].min(interior)
                };
            }
            g
        })
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(gs.seed);
    let mut r: Su2TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    let codes: Vec<qntt::su2::core::Family> = (0..d)
        .map(|k| {
            if k < 1 {
                qntt::su2::core::Family::Occ
            } else {
                qntt::su2::core::Family::Empty
            }
        })
        .collect();
    let mut csf: Su2TtVector<f64> = tt_csf_with_point_group(&codes, PointGroup::TRIVIAL);
    assert_eq!(csf.space, space, "the replica's CSF start is not a member");
    pre_d::normalize_mut(&mut csf);
    pre_d::normalize_mut(&mut r);
    r.scale_mut(0.2);
    let mut x0 = &csf + &r;
    x0.round_ranks_mut(&ranks);
    pre_d::normalize_mut(&mut x0);
    x0.right_orthogonalize(true);

    let mopts = MalsOptions {
        eps: gs.eps,
        bond_dims: gs.bond_ramp.clone().unwrap(),
        noise: gs.noise.clone(),
        noise_type: gs.noise_type,
        seed: gs.seed,
        max_iter: gs.max_iter,
        ..MalsOptions::default()
    };
    let (_lam, xa) = mals(&h, &x0, &mopts);

    let mut ropts = RiemannianOpts::new(&space);
    ropts.eps = 1e-13;
    ropts.max_iter = 600;
    ropts.gradtol = gs.polish_gradtol;
    let stop = 0.1 * gs.grad_accept;
    let grad = |y: &Su2TtVector<f64>| weingarten_operator(&h, y).gradient().norm();
    let polish = |y0: &Su2TtVector<f64>| -> Su2TtVector<f64> {
        let (_l, mut y) = pre_d::riemannian_cg(&h, y0, &ropts);
        for _ in 0..gs.polish_restarts {
            if grad(&y) < stop {
                break;
            }
            let (_l2, y2) = pre_d::riemannian_cg(&h, &y, &ropts);
            y = y2;
        }
        y
    };
    let mut x = polish(&xa);
    pre_d::normalize_mut(&mut x);
    x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
    pre_d::normalize_mut(&mut x);
    let mut x_old = polish(&x);
    pre_d::normalize_mut(&mut x_old);
    x_old.check();
    let e_old = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x_old);
    let g_old = grad(&x_old);

    assert_f64_bits(e_new, e_old, "channel_ground_state energy");
    assert_f64_bits(g_new, g_old, "channel_ground_state gate");
    coeffs_bits_eq_su2(&x_new, &x_old, "channel_ground_state x");
    println!("tms_d2su2 pipeline: energy {e_new:.16e}, gate {g_new:.3e}");
    assert!(
        g_new < 1e-9,
        "the SU(2) pipeline did not reach a critical point (gate {g_new:.3e}); this arm would not \
         pin the noise-regime branch"
    );
    report_digests(
        "pipeline",
        &[digest_su2(e_new, &x_new)],
        &RECORDED_DIGEST_PIPELINE,
    );
}
