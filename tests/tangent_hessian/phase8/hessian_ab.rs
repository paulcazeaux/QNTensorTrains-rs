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
use qntt::solvers::davidson::DavidsonOpts;
use qntt::solvers::hess::{effham_apply, effham_lowest};
use qntt::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_gd, tangent_bases};
use qntt::{QnSpace, Scalar, TtVector, rayleigh_quotient};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t8h_effham_hermitian() {
    let (_space, x, base_l, base_r, dir1, dir2) = tangent_test_setup(0x8E10);
    let h = two_body_swap_h(_space, 0, 1, 2, 3);
    let lambda = rayleigh_quotient(&h, &x).re_s();
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let dy = TtTangent::project(&base_l, &base_r, &dir2);
    let a_dx = effham_apply(&h, &base_l, &base_r, lambda, &dx);
    let a_dy = effham_apply(&h, &base_l, &base_r, lambda, &dy);
    let lhs = a_dx.dot(&dy);
    let rhs = dx.dot(&a_dy);
    assert!(
        (lhs - rhs).abs() < 1e-10 * lhs.abs().max(1.0),
        "effham Hermitian: ⟨A dx,dy⟩={lhs} vs ⟨dx,A dy⟩={rhs}"
    );
}

#[test]
fn t8h_effham_linear() {
    let (_space, x, base_l, base_r, dir1, dir2) = tangent_test_setup(0x8E20);
    let h = two_body_swap_h(_space, 0, 1, 2, 3);
    let lambda = rayleigh_quotient(&h, &x).re_s();
    let dx = TtTangent::project(&base_l, &base_r, &dir1);
    let dy = TtTangent::project(&base_l, &base_r, &dir2);
    let combo = dx.scale(0.7).add(&dy.scale(-1.3));
    let a_combo = effham_apply(&h, &base_l, &base_r, lambda, &combo);
    let want = effham_apply(&h, &base_l, &base_r, lambda, &dx)
        .scale(0.7)
        .add(&effham_apply(&h, &base_l, &base_r, lambda, &dy).scale(-1.3));
    let diff = &a_combo.to_vector() - &want.to_vector();
    assert!(diff.norm_lr(false) < 1e-10, "effham linearity");
}

#[test]
fn t8h_effham_lowest_smoke() {
    let (_space, x0, _bl, _br, dir1, _dir2) = tangent_test_setup(0x8E30);
    let h = two_body_swap_h(_space, 0, 1, 2, 3);
    let opts = RiemannianOpts::new(&_space);
    let (_lam, x) = riemannian_gd(&h, &x0, &opts);
    let (bl, br) = tangent_bases(&x);
    let v0 = TtTangent::project(&bl, &br, &dir1);
    let dopts = DavidsonOpts::default();
    let (val, vec) = effham_lowest(&h, &x, &v0, &dopts);
    assert!(val.is_finite(), "effham lowest eigenvalue finite");
    assert!(
        (vec.norm() - 1.0).abs() < 1e-6,
        "effham Ritz vector unit-norm: {}",
        vec.norm()
    );
}

fn tangent_ortho_basis(
    base_l: &TtVector<f64>,
    base_r: &TtVector<f64>,
    space: QnSpace,
    seed: u64,
    max_dirs: usize,
) -> Vec<TtTangent<f64>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut basis: Vec<TtTangent<f64>> = Vec::new();
    for _ in 0..max_dirs {
        let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let mut u = TtTangent::project(base_l, base_r, &dir);
        for b in &basis {
            let c = b.dot(&u);
            u = u.sub(&b.scale(c));
        }
        let nrm = u.norm();
        if nrm > 1e-8 {
            u.scale_mut(1.0 / nrm);
            basis.push(u);
        }
    }
    basis
}

#[test]
fn t8h_tangent_basis_complete() {
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E40);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let (base_l, base_r) = tangent_bases(&x);
    let basis = tangent_ortho_basis(&base_l, &base_r, space, 0x9001, 60);
    assert!(!basis.is_empty(), "nonempty tangent basis");
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let xi = TtTangent::project(&base_l, &base_r, &dir);
    let mut recon = xi.scale(0.0);
    for u in &basis {
        let c = u.dot(&xi);
        recon = recon.add(&u.scale(c));
    }
    let err = xi.sub(&recon).norm();
    assert!(
        err < 1e-9,
        "tangent basis completeness: reconstruction err {err}"
    );
}

#[test]
fn t8h_effham_matrix_rep() {
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E50);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let h = two_body_swap_h(space, 0, 1, 1, 0);
    let (base_l, base_r) = tangent_bases(&x);
    let lambda = rayleigh_quotient(&h, &x).re_s();
    let basis = tangent_ortho_basis(&base_l, &base_r, space, 0x9002, 60);
    let n = basis.len();
    assert!(n >= 2, "need a nontrivial tangent space, got {n}");

    let a_basis: Vec<TtTangent<f64>> = basis
        .iter()
        .map(|u| effham_apply(&h, &base_l, &base_r, lambda, u))
        .collect();
    let mat: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| basis[i].dot(&a_basis[j])).collect())
        .collect();
    for (i, row) in mat.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!(
                (val - mat[j][i]).abs() < 1e-9,
                "A matrix symmetric at ({i},{j}): {} vs {}",
                val,
                mat[j][i]
            );
        }
    }
    for (k, ak) in a_basis.iter().enumerate() {
        let mut recon = ak.scale(0.0);
        for (i, ui) in basis.iter().enumerate() {
            recon = recon.add(&ui.scale(mat[i][k]));
        }
        let err = ak.sub(&recon).norm();
        assert!(err < 1e-9, "A matrix-rep consistency at k={k}: err {err}");
    }
}

#[test]
fn t8h_normal_residual_is_normal() {
    use qntt::solvers::hess::normal_residual;
    let d = 6usize;
    let space = QnSpace::new(d, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E60);
    let mut x: TtVector<f64> = randomized_state(space, 2, 2, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let h = two_body_swap_h(space, 0, 1, 2, 3);
    let (base_l, base_r) = tangent_bases(&x);
    let lambda = rayleigh_quotient(&h, &x).re_s();
    let rperp = normal_residual(&h, &base_l, &base_r, &x, lambda);
    let proj = TtTangent::project(&base_l, &base_r, &rperp);
    assert!(
        proj.norm() < 1e-9 * rperp.norm_lr(false).max(1.0),
        "r⊥ not normal: ‖P(r⊥)‖={}, ‖r⊥‖={}",
        proj.norm(),
        rperp.norm_lr(false)
    );
    assert!(
        rperp.norm_lr(false) > 1e-8,
        "expected nonzero normal residual at generic x"
    );
}

#[test]
fn t8i_a_apply_matches_exact_product_reference() {
    use qntt::solvers::hess::{effham_apply, weingarten_operator};
    for (d, nup, ndn, seed, lo, hi) in [
        (4usize, 2usize, 1usize, 0xA1A1u64, 1usize, 2usize),
        (4, 2, 2, 0xA2A2, 1, 2),
        (5, 2, 1, 0xA3A3, 2, 3),
        (5, 3, 2, 0xA4A4, 2, 3),
        (6, 3, 3, 0xA5A5, 2, 4),
    ] {
        let space = QnSpace::new(d, nup, ndn);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: TtVector<f64> = randomized_state(space, lo, hi, &mut rng);
        x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        x.check();
        let ham = generic_truncated_h(space);
        let ctx = weingarten_operator(&ham, &x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let (base_l, base_r) = ctx.bases();
        let basis = tangent_ortho_basis(base_l, base_r, space, seed ^ 0xFFFF, 16);
        assert!(
            basis.len() >= 2,
            "d={d} N=({nup},{ndn}): need a nontrivial tangent space, got {}",
            basis.len()
        );
        let mut worst = 0.0f64;
        for u in &basis {
            let fused = ctx.a_apply_fused(u);
            let exact = effham_apply(&ham, base_l, base_r, lambda, u);
            let mut diff = fused;
            diff.axpy(-1.0, &exact);
            let rel = diff.norm() / exact.norm().max(1.0);
            worst = worst.max(rel);
        }
        assert!(
            worst < 1e-11,
            "d={d} N=({nup},{ndn}): fused a_apply vs exact-product reference, \
             worst relative deviation {worst:.3e} over {} tangents",
            basis.len()
        );
    }
}

#[test]
fn t8i_b_apply_matches_explicit_z_reference() {
    use qntt::core::weingarten::{FloorPolicy, Terms, bond_gauges, weingarten_assemble_cached};
    use qntt::solvers::hess::{normal_residual, weingarten_operator};
    for (d, nup, ndn, seed, lo, hi) in [
        (4usize, 2usize, 1usize, 0xB1B1u64, 1usize, 2usize),
        (4, 2, 2, 0xB2B2, 1, 2),
        (5, 2, 1, 0xB3B3, 2, 3),
        (5, 3, 2, 0xB4B4, 2, 3),
        (6, 3, 3, 0xB5B5, 2, 4),
    ] {
        let space = QnSpace::new(d, nup, ndn);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: TtVector<f64> = randomized_state(space, lo, hi, &mut rng);
        x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        x.check();
        let ham = generic_truncated_h(space);
        let ctx = weingarten_operator(&ham, &x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let (base_l, base_r) = ctx.bases();
        let z = normal_residual(&ham, base_l, base_r, &x, lambda);
        let gauges = bond_gauges(base_l, base_r, FloorPolicy::default_hard());
        let basis = tangent_ortho_basis(base_l, base_r, space, seed ^ 0xFFFF, 16);
        assert!(
            basis.len() >= 2,
            "d={d} N=({nup},{ndn}): need a nontrivial tangent space, got {}",
            basis.len()
        );
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for u in &basis {
            let fused = ctx.b_apply(u);
            let exact = weingarten_assemble_cached(base_l, base_r, &z, u, &gauges, Terms::all());
            let mut diff = fused;
            diff.axpy(-1.0, &exact);
            worst = worst.max(diff.norm());
            scale = scale.max(exact.norm());
        }
        assert!(
            scale > 1e-8,
            "d={d} N=({nup},{ndn}): term B vacuous (max ‖B·ξ‖ = {scale:.3e}) — fixture proves nothing"
        );
        assert!(
            worst < 1e-10 * scale,
            "d={d} N=({nup},{ndn}): split b_apply vs explicit-z reference, worst ‖Δ‖ = {worst:.3e} \
             (max ‖B·ξ‖ = {scale:.3e})"
        );
    }
}

#[test]
fn t8i_ctx_gradient_matches_explicit_projection() {
    use qntt::solvers::hess::weingarten_operator;
    for (d, nup, ndn, seed, lo, hi) in [
        (4usize, 2usize, 1usize, 0xC7C7u64, 1usize, 2usize),
        (5, 3, 2, 0xC8C8, 2, 3),
        (6, 3, 3, 0xC9C9, 2, 4),
    ] {
        let space = QnSpace::new(d, nup, ndn);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: TtVector<f64> = randomized_state(space, lo, hi, &mut rng);
        x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        x.check();
        let ham = generic_truncated_h(space);
        let ctx = weingarten_operator(&ham, &x);
        let (base_l, base_r) = ctx.bases();
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let hx = ham.apply(&x);
        let full = &hx - &(&x * lambda);
        let want = TtTangent::project(base_l, base_r, &full);
        let mut diff = ctx.gradient().clone();
        diff.axpy(-1.0, &want);
        assert!(
            want.norm() > 1e-10,
            "d={d}: vacuous gradient fixture (‖grad‖ = {:.3e})",
            want.norm()
        );
        assert!(
            diff.norm() < 1e-11 * want.norm(),
            "d={d}: ctx.gradient() vs explicit P(Hx−λx): ‖Δ‖ = {:.3e} (‖grad‖ = {:.3e})",
            diff.norm(),
            want.norm()
        );
    }
}

#[test]
fn t8i_hess_apply_matches_fd_of_gradient() {
    use qntt::solvers::hess::weingarten_operator;
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B0, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8F10);
    let mut b_norm_max = 0.0f64;
    for trial in 0..2 {
        let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let xi = TtTangent::project(&base_l, &base_r, &dir);
        let xi = xi.scale(1.0 / xi.norm());
        let fd = hess_fd_central(&ham, &x, &xi, 1e-4);
        let hess = ctx.hess_apply(&xi);
        let rel = fd.sub(&hess).norm() / fd.norm();
        assert!(
            rel < 1e-6,
            "trial {trial}: ‖FD − (A+B)‖/‖FD‖ = {rel:.3e} (analytic hess_apply vs FD-of-gradient)"
        );
        let b_an = ctx.b_apply(&xi);
        let b_fd = fd.sub(&ctx.a_apply(&xi));
        let b_rel = b_fd.sub(&b_an).norm() / b_an.norm().max(1e-30);
        assert!(
            b_rel < 1e-3,
            "trial {trial}: ‖(FD − A) − B_analytic‖/‖B‖ = {b_rel:.3e} (sign/magnitude pinning)"
        );
        b_norm_max = b_norm_max.max(b_an.norm());
    }
    assert!(
        b_norm_max > 1e-8,
        "term B vacuous (‖B·ξ‖={b_norm_max:.3e}) — fixture not a proper truncated minimizer"
    );
}

#[test]
fn t8i_weingarten_dense_matrix_oracle() {
    use qntt::solvers::hess::weingarten_operator;
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_ortho_basis(&base_l, &base_r, space, 0x9003, 80);
    let n = basis.len();
    assert!(n >= 2, "need a nontrivial tangent space, got {n}");

    let a_cols: Vec<TtTangent<f64>> = basis.iter().map(|u| ctx.a_apply(u)).collect();
    let b_cols: Vec<TtTangent<f64>> = basis.iter().map(|u| ctx.b_apply(u)).collect();
    let amat: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| basis[i].dot(&a_cols[j])).collect())
        .collect();
    let bmat: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| basis[i].dot(&b_cols[j])).collect())
        .collect();
    let bmax = bmat
        .iter()
        .flatten()
        .fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax > 1e-8,
        "dense B matrix vacuous (max |B_mn| = {bmax:.3e})"
    );
    for i in 0..n {
        for j in 0..n {
            assert!(
                (bmat[i][j] - bmat[j][i]).abs() < 1e-10 + 1e-6 * bmax,
                "B not symmetric at ({i},{j}): {} vs {}",
                bmat[i][j],
                bmat[j][i]
            );
            let hij = amat[i][j] + bmat[i][j];
            let hji = amat[j][i] + bmat[j][i];
            assert!(
                (hij - hji).abs() < 1e-9,
                "A+B not symmetric at ({i},{j}): {hij} vs {hji}"
            );
        }
    }
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (j, uj) in basis.iter().enumerate() {
        let fd = hess_fd_central(&ham, &x, uj, 1e-4);
        for (i, ui) in basis.iter().enumerate() {
            let fdij = ui.dot(&fd);
            let hij = amat[i][j] + bmat[i][j];
            num += (fdij - hij) * (fdij - hij);
            den += fdij * fdij;
        }
    }
    let rel = (num / den).sqrt();
    assert!(
        rel < 1e-6,
        "dense FD matrix vs A+B: Frobenius-relative error {rel:.3e}"
    );
}

#[test]
fn t8i_weingarten_general_z_matches_b_apply() {
    use qntt::core::weingarten::{FloorPolicy, Terms, weingarten_assemble_general};
    use qntt::solvers::hess::weingarten_operator;
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8F20);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let xi = TtTangent::project(&base_l, &base_r, &dir);
    let xi = xi.scale(1.0 / xi.norm());

    let b_ctx = ctx.b_apply(&xi);
    assert!(b_ctx.norm() > 1e-8, "B vacuous — fixture not truncated");
    let hx = &ham * &x;
    let b_gen = weingarten_assemble_general(
        &base_l,
        &base_r,
        &hx,
        &xi,
        FloorPolicy::default_hard(),
        Terms::all(),
    );
    let rel = b_gen.sub(&b_ctx).norm() / b_ctx.norm();
    assert!(
        rel < 1e-8,
        "general-Z apply vs b_apply: relative diff {rel:.3e}"
    );
}

#[test]
fn t8i_corrected_fd_certifies_hess_at_noncritical_point() {
    use qntt::solvers::hess::weingarten_operator;
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, xstar, bl_star, br_star) = weingarten_minimizer(space, 0xB0B0, 1, 2);

    {
        let ctx = weingarten_operator(&ham, &xstar);
        let (bl, br) = ctx.bases();
        let g_vec = ctx.gradient().to_vector();
        let radial = radial_mode_at(bl, br, &xstar);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8F60);
        let xi = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
        let q_ab = xi.dot(&ctx.a_apply_fused(&xi).add(&ctx.b_apply(&xi)));
        let s = q_ab.abs().max(1.0);
        let (_q, _qu, corr) = corrected_form(&ham, &xstar, &g_vec, &xi, 1e-4);
        assert!(
            corr.abs() < 1e-10,
            "retraction-acceleration bias at a critical point: corr = {corr:.3e} — either the \
             gradient is contaminated or retract_hosvd is no longer second-order"
        );
        let q_rich = corrected_form_richardson(&ham, &xstar, &g_vec, &xi, 2e-4);
        assert!(
            (q_rich - q_ab).abs() < 1e-6 * s,
            "critical point: |q_corr − q_AB|/s = {:.3e}",
            (q_rich - q_ab).abs() / s
        );
    }

    let x = noncritical_point(&xstar, &bl_star, &br_star, 3e-2, 0x81D1);
    let ctx = weingarten_operator(&ham, &x);
    let gnorm = ctx.gradient().norm();
    assert!(
        gnorm > 1e-3,
        "perturbation vacuous: ‖grad‖ = {gnorm:.3e} at α=3e-2"
    );
    let (bl, br) = ctx.bases();
    let g_vec = ctx.gradient().to_vector();
    let radial = radial_mode_at(bl, br, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8F61);

    let mut b_norm_max = 0.0f64;
    let mut vec_bias_max = 0.0f64;
    for trial in 0..3 {
        let xi = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
        let a_xi = ctx.a_apply_fused(&xi);
        let b_xi = ctx.b_apply(&xi);
        let q_ab = xi.dot(&a_xi.add(&b_xi));
        let s = q_ab.abs().max(1.0);
        let q_rich = corrected_form_richardson(&ham, &x, &g_vec, &xi, 2e-4);
        assert!(
            (q_rich - q_ab).abs() < 1e-6 * s,
            "trial {trial}: |q_corr − q_AB|/s = {:.3e} at ‖grad‖ = {gnorm:.3e} — the A+B \
             operator disagrees with the transport-free oracle off-critical",
            (q_rich - q_ab).abs() / s
        );
        let fd_vec = hess_fd_central(&ham, &x, &xi, 1e-4);
        let hess = ctx.hess_apply(&xi);
        vec_bias_max = vec_bias_max.max(fd_vec.sub(&hess).norm() / fd_vec.norm());
        b_norm_max = b_norm_max.max(b_xi.norm());
    }
    assert!(
        b_norm_max > 1e-3,
        "term B vacuous at the perturbed point (max ‖B·ξ‖ = {b_norm_max:.3e})"
    );
    assert!(
        vec_bias_max > 1e-3,
        "vector FD-of-gradient shows no transport bias at ‖grad‖ = {gnorm:.3e} \
         (max rel = {vec_bias_max:.3e}) — the corrected-oracle comparison is then vacuous"
    );

    let xi = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
    let eta = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
    let q_s = corrected_form_richardson(&ham, &x, &g_vec, &xi.add(&eta), 2e-4);
    let q_x = corrected_form_richardson(&ham, &x, &g_vec, &xi, 2e-4);
    let q_e = corrected_form_richardson(&ham, &x, &g_vec, &eta, 2e-4);
    let off_fd = 0.5 * (q_s - q_x - q_e);
    let hx = ctx.a_apply_fused(&xi).add(&ctx.b_apply(&xi));
    let he = ctx.a_apply_fused(&eta).add(&ctx.b_apply(&eta));
    let off_an = 0.5 * (eta.dot(&hx) + xi.dot(&he));
    assert!(
        (off_fd - off_an).abs() < 1e-5,
        "polarized off-diagonal: FD {off_fd:.8e} vs analytic {off_an:.8e}"
    );
}

#[test]
#[should_panic(expected = "Z is not normal")]
#[cfg(debug_assertions)]
fn t8i_weingarten_raw_assembly_rejects_tangential_z() {
    use qntt::core::weingarten::{FloorPolicy, Terms, weingarten_assemble};
    let (_space, x, base_l, base_r, dir1, _dir2) = tangent_test_setup(0x8F30);
    let xi = TtTangent::project(&base_l, &base_r, &dir1);
    let _ = weingarten_assemble(
        &base_l,
        &base_r,
        &x,
        &xi,
        FloorPolicy::default_hard(),
        Terms::all(),
    );
}
