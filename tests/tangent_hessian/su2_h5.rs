// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::shared::su2_minimizer;
use crate::shared::sym_eigvals_jacobi;
use crate::shared::tangent_ortho_basis;
use qntt::core::weingarten::rpa_generator_eigenvalues;
use qntt::round::RoundMode;
use qntt::solvers::hess::SymplecticSpectrum;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient, xt_h_y};
use qntt::su2::hess::{
    hessian_grams, hessian_grams_c64, normal_residual, symplectic_eigenvalues,
    symplectic_eigenvalues_c64, weingarten_operator,
};
use qntt::su2::ops::dot as su2_dot;
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::normalize_mut;
use qntt::su2::tangent::{Su2TtTangent, projective_transport};
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::tangent_basis;
use qntt::{C64, Scalar};
use qntt_testkit::gad_mol::su2_grad_exact_at;
use qntt_testkit::generic_su2_h;
use qntt_testkit::generic_su2_h_c64;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn hess_fd_central<T: Scalar>(
    ham: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    xi: &Su2TtTangent<T>,
    h: f64,
) -> Su2TtTangent<T> {
    let mut xp = xi.retract_hosvd(T::from_f64(h));
    normalize_mut(&mut xp);
    let mut xm = xi.retract_hosvd(T::from_f64(-h));
    normalize_mut(&mut xm);
    let gp = projective_transport(x, &su2_grad_exact_at(ham, &xp));
    let gm = projective_transport(x, &su2_grad_exact_at(ham, &xm));
    gp.sub(&gm).scale(T::from_f64(1.0 / (2.0 * h)))
}

#[test]
fn h5_a_apply_matches_exact_product_reference() {
    use qntt::su2::hess::effham_apply;
    for (d, n, two_s, seed, lo, hi) in [
        (4usize, 2usize, 0usize, 0xC1C1u64, 1usize, 2usize),
        (5, 3, 1, 0xC2C2, 1, 2),
        (5, 4, 0, 0xC3C3, 2, 3),
        (6, 4, 2, 0xC4C4, 2, 3),
    ] {
        let space = Su2Space::new(d, n, two_s);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
        x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        x.check();
        let ham = generic_su2_h::<f64>(space);
        let ctx = weingarten_operator(&ham, &x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let (base_l, base_r) = ctx.bases();
        let basis = tangent_ortho_basis(base_l, base_r, space, seed ^ 0xFFFF, 12);
        assert!(
            basis.len() >= 2,
            "d={d} N={n} 2S={two_s}: need a nontrivial tangent space, got {}",
            basis.len()
        );
        let mut worst = 0.0f64;
        for u in &basis {
            let fused = ctx.a_apply_fused(u);
            let exact = effham_apply(&ham, base_l, base_r, lambda, u);
            let rel = fused.sub(&exact).norm() / exact.norm().max(1.0);
            worst = worst.max(rel);
        }
        assert!(
            worst < 1e-11,
            "d={d} N={n} 2S={two_s}: fused a_apply vs exact-product reference, \
             worst relative deviation {worst:.3e} over {} tangents",
            basis.len()
        );
    }
}

#[test]
fn h5_b_apply_matches_explicit_z_reference() {
    use qntt::core::weingarten::{FloorPolicy, Terms};
    use qntt::su2::weingarten::{bond_gauges, weingarten_assemble_cached};
    for (d, n, two_s, seed, lo, hi) in [
        (5usize, 3usize, 1usize, 0xE2E2u64, 1usize, 2usize),
        (5, 4, 0, 0xE3E3, 1, 2),
        (6, 4, 2, 0xE4E4, 2, 3),
        (6, 6, 0, 0xE5E5, 2, 3),
        (7, 6, 0, 0xE6E6, 2, 3),
    ] {
        let space = Su2Space::new(d, n, two_s);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
        x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        x.check();
        let ham = generic_su2_h::<f64>(space);
        let ctx = weingarten_operator(&ham, &x);
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let (base_l, base_r) = ctx.bases();
        let z = normal_residual(&ham, base_l, base_r, &x, lambda);
        let gauges = bond_gauges(base_l, base_r, FloorPolicy::default_hard());
        let basis = tangent_ortho_basis(base_l, base_r, space, seed ^ 0xFFFF, 12);
        assert!(
            basis.len() >= 2,
            "d={d} N={n} 2S={two_s}: need a nontrivial tangent space, got {}",
            basis.len()
        );
        let mut worst = 0.0f64;
        let mut scale = 0.0f64;
        for u in &basis {
            let fused = ctx.b_apply(u);
            let exact = weingarten_assemble_cached(base_l, base_r, &z, u, &gauges, Terms::all());
            worst = worst.max(fused.sub(&exact).norm());
            scale = scale.max(exact.norm());
        }
        assert!(
            scale > 1e-8,
            "d={d} N={n} 2S={two_s}: term B vacuous (max ‖B·ξ‖ = {scale:.3e})"
        );
        assert!(
            worst < 1e-10 * scale,
            "d={d} N={n} 2S={two_s}: split b_apply vs explicit-z reference, worst ‖Δ‖ = \
             {worst:.3e} (max ‖B·ξ‖ = {scale:.3e})"
        );
    }
}

#[test]
fn h5_ctx_gradient_matches_explicit_projection() {
    for (d, n, two_s, seed, lo, hi) in [
        (5usize, 3usize, 1usize, 0xF1F1u64, 1usize, 2usize),
        (6, 4, 2, 0xF2F2, 2, 3),
        (6, 6, 0, 0xF3F3, 2, 3),
    ] {
        let space = Su2Space::new(d, n, two_s);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
        x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        x.check();
        let ham = generic_su2_h::<f64>(space);
        let ctx = weingarten_operator(&ham, &x);
        let (base_l, base_r) = ctx.bases();
        let lambda = rayleigh_quotient(&ham, &x).re_s();
        let hx = ham.apply(&x);
        let full = &hx - &(&x * lambda);
        let want = Su2TtTangent::project(base_l, base_r, &full);
        assert!(
            want.norm() > 1e-10,
            "d={d}: vacuous gradient fixture (‖grad‖ = {:.3e})",
            want.norm()
        );
        let rel = ctx.gradient().sub(&want).norm() / want.norm();
        assert!(
            rel < 1e-11,
            "d={d} N={n} 2S={two_s}: ctx.gradient() vs explicit P(Hx−λx): rel {rel:.3e}"
        );
    }
}

#[test]
fn h5_hess_apply_matches_fd_of_gradient() {
    let d = 5usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0E1);
    let mut b_norm_max = 0.0f64;
    for trial in 0..2 {
        let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
        let xi = xi.scale(1.0 / xi.norm());
        let fd = hess_fd_central(&ham, &x, &xi, 1e-5);
        let hess = ctx.hess_apply(&xi);
        let rel = fd.sub(&hess).norm() / fd.norm();
        assert!(
            rel < 1e-6,
            "trial {trial}: ‖FD − (A+B)‖/‖FD‖ = {rel:.3e} (SU(2) analytic vs FD-of-gradient)"
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
fn h5_normal_residual_is_normal() {
    let d = 5usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D2, 1, 2);
    let lam = rayleigh_quotient(&ham, &x).re_s();
    let rperp = normal_residual(&ham, &base_l, &base_r, &x, lam);
    let proj = Su2TtTangent::project(&base_l, &base_r, &rperp);
    assert!(
        proj.norm() < 1e-9 * rperp.norm().max(1.0),
        "r⊥ not normal: ‖P(r⊥)‖={}, ‖r⊥‖={}",
        proj.norm(),
        rperp.norm()
    );
    assert!(
        rperp.norm() > 1e-8,
        "expected nonzero normal residual at a truncated SU(2) minimizer"
    );
}

#[test]
fn h5_weingarten_dense_matrix_oracle() {
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D3, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_ortho_basis(&base_l, &base_r, space, 0xD0E3, 80);
    let n = basis.len();
    assert!(n >= 2, "need a nontrivial tangent space, got {n}");

    let a_cols: Vec<Su2TtTangent<f64>> = basis.iter().map(|u| ctx.a_apply(u)).collect();
    let b_cols: Vec<Su2TtTangent<f64>> = basis.iter().map(|u| ctx.b_apply(u)).collect();
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
fn h5_tangent_basis_complete() {
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (_ham, _x, base_l, base_r) =
        su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D4, 1, 2);
    let basis = tangent_basis(&base_l, &base_r, false);
    assert!(!basis.is_empty(), "nonempty tangent basis");
    for (i, ui) in basis.iter().enumerate() {
        for (j, uj) in basis.iter().enumerate() {
            let g = ui.dot(uj);
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (g - want).abs() < 1e-9,
                "⟨u_{i},u_{j}⟩ = {g} (centerweighted orthonormality)"
            );
        }
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0E4);
    let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
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
fn h5_symplectic_linear_energy_full_rank() {
    let d = 3usize;
    let space = Su2Space::new(d, 2, 0);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D5, 1, 1);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0E5);
    let mut vbasis: Vec<Su2TtVector<f64>> = Vec::new();
    for _ in 0..30 {
        let mut v: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
        for _pass in 0..2 {
            for b in &vbasis {
                let c = su2_dot(b, &v);
                v = &v - &(b * c);
            }
            v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        }
        let nrm = v.norm();
        if nrm > 1e-6 {
            v.scale_mut(1.0 / nrm);
            vbasis.push(v);
        }
    }
    let nd = vbasis.len();
    assert_eq!(nd, 6, "CSF dimension at (3,2,0)");
    let mut hmat = vec![vec![0.0f64; nd]; nd];
    for (j, sj) in vbasis.iter().enumerate() {
        let hs = ham.apply(sj);
        for (i, si) in vbasis.iter().enumerate() {
            hmat[i][j] = su2_dot(si, &hs);
        }
    }
    let evals = sym_eigvals_jacobi(hmat);
    let lam = rayleigh_quotient(&ham, &x).re_s();
    assert!(
        (lam - evals[0]).abs() < 1e-8,
        "manifold minimizer vs dense CSF ground energy: {lam} vs {}",
        evals[0]
    );

    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    assert_eq!(basis.len(), nd - 1, "projective SU(2) tangent dimension");
    let (_a, b) = hessian_grams(&ctx, &basis);
    let bmax = b.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax < 1e-10,
        "B must vanish on the full manifold: {bmax:.3e}"
    );
    match symplectic_eigenvalues(&ctx, &basis) {
        SymplecticSpectrum::Stable { omega } => {
            assert_eq!(omega.len(), nd - 1);
            for (k, w) in omega.iter().enumerate() {
                let want = evals[k + 1] - evals[0];
                assert!(
                    (w - want).abs() < 1e-6 * want.max(1.0),
                    "ω[{k}] = {w} vs E_{} − E_0 = {want}",
                    k + 1
                );
            }
        }
        SymplecticSpectrum::Resonant { signed } => panic!(
            "expected a stable spectrum at the ground state, got Resonant: {} unstable pair(s), \
             metric {:?}, lowest ω² = {:?}",
            signed.n_neg,
            signed.metric,
            signed.omega2.first()
        ),
        SymplecticSpectrum::Unstable { reason, .. } => {
            panic!("expected a stable spectrum at the ground state, got {reason:?}")
        }
    }
}

#[test]
fn h5_symplectic_generator_cross_check() {
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D3, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    assert!(n >= 2, "need a nontrivial tangent space, got {n}");
    let (a, b) = hessian_grams(&ctx, &basis);
    let bmax = b.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax > 1e-8,
        "B vacuous at the truncated minimizer ({bmax:.3e})"
    );

    let SymplecticSpectrum::Stable { omega } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("expected a stable RPA spectrum at a converged minimizer");
    };
    assert_eq!(omega.len(), n);
    assert!(omega[0] > 0.0, "ω must be positive (Williamson)");

    let eigs = rpa_generator_eigenvalues(&a, &b, n);
    assert_eq!(eigs.len(), 2 * n);
    let scale = omega.last().copied().unwrap_or(1.0);
    let mut pos: Vec<f64> = Vec::new();
    for e in &eigs {
        assert!(
            e.im.abs() < 1e-8 * scale.max(1.0),
            "generator eigenvalue not real at a stable minimum: {e:?}"
        );
        if e.re > 0.0 {
            pos.push(e.re);
        }
    }
    assert_eq!(pos.len(), n, "generator eigenvalues must come in ±ω pairs");
    pos.sort_by(|x, y| x.partial_cmp(y).unwrap());
    for (k, (w, g)) in omega.iter().zip(pos.iter()).enumerate() {
        assert!(
            (w - g).abs() < 1e-7 * w.max(1.0),
            "ω[{k}]: Williamson {w} vs generator {g}"
        );
    }
}

#[test]
fn h5_c64_structure_fd_and_driver() {
    let d = 4usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h_c64(space), space, 0xD0D6, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0E6);
    let dir: Su2TtVector<C64> = randomized_su2_state(space, 1, 2, &mut rng);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
    let xi = xi.scale(C64::from_f64(1.0 / xi.norm()));
    let dir2: Su2TtVector<C64> = randomized_su2_state(space, 1, 2, &mut rng);
    let eta = Su2TtTangent::project(&base_l, &base_r, &dir2);
    let eta = eta.scale(C64::from_f64(1.0 / eta.norm()));

    let a_xi = ctx.a_apply(&xi);
    let b_xi = ctx.b_apply(&xi);
    assert!(b_xi.norm() > 1e-8, "B vacuous over C64 SU(2)");

    let c = C64::new(0.7, -1.3);
    let b_cxi = ctx.b_apply(&xi.scale(c));
    let b_anti = b_cxi.sub(&b_xi.scale(c.conj_s())).norm() / b_xi.norm();
    assert!(
        b_anti < 1e-10,
        "B not antilinear over C64 SU(2): {b_anti:.3e}"
    );
    let b_lin_hyp = b_cxi.sub(&b_xi.scale(c)).norm() / b_xi.norm();
    assert!(
        b_lin_hyp > 1e-2,
        "linear-B hypothesis did not fail ({b_lin_hyp:.3e})"
    );
    let b_eta = ctx.b_apply(&eta);
    let b_sym = (eta.dot(&b_xi) - xi.dot(&b_eta)).abs_s() / b_xi.norm();
    assert!(
        b_sym < 1e-10,
        "B not complex-symmetric over C64 SU(2): {b_sym:.3e}"
    );
    assert!(
        eta.dot(&b_xi).im_s().abs() > 1e-8,
        "Im⟨η,Bξ⟩ vacuous — complex SU(2) layer untested"
    );

    let fd = hess_fd_central(&ham, &x, &xi, 1e-4);
    let rel = fd.sub(&a_xi.add(&b_xi)).norm() / fd.norm();
    assert!(
        rel < 1e-6,
        "‖FD[ξ] − (A+B)ξ‖/‖FD‖ = {rel:.3e} over C64 SU(2)"
    );
    let i = C64::new(0.0, 1.0);
    let fd_i = hess_fd_central(&ham, &x, &xi.scale(i), 1e-4);
    let anti = fd_i.sub(&a_xi.scale(i).sub(&b_xi.scale(i))).norm() / fd_i.norm();
    let lin = fd_i.sub(&a_xi.scale(i).add(&b_xi.scale(i))).norm() / fd_i.norm();
    assert!(anti < 1e-6, "FD[iξ] ≠ i·Aξ − i·Bξ over SU(2): {anti:.3e}");
    assert!(
        lin > 100.0 * anti.max(1e-8),
        "wrong-sign combination did not fail (anti {anti:.3e} vs lin {lin:.3e})"
    );

    let basis = tangent_basis(&base_l, &base_r, true);
    let (a, b) = hessian_grams_c64(&ctx, &basis);
    let im_max = a
        .iter()
        .chain(b.iter())
        .fold(0.0f64, |acc, v| acc.max(v.im_s().abs()));
    assert!(
        im_max > 1e-6,
        "H^ri vacuous ({im_max:.3e}) on the complex SU(2) layer"
    );
    let SymplecticSpectrum::Stable { omega } = symplectic_eigenvalues_c64(&ctx, &basis) else {
        panic!("expected a stable complex SU(2) RPA spectrum");
    };
    assert_eq!(omega.len(), basis.len());
    assert!(omega[0] > 0.0);
}

#[test]
fn h5_c64_structure_and_fd_see_the_conjugate_conversion_at_d5() {
    let d = 5usize;
    let space = Su2Space::new(d, 4, 0);
    let (ham, x, base_l, base_r) = su2_minimizer(generic_su2_h_c64(space), space, 0xD1D6, 1, 3);
    let ctx = weingarten_operator(&ham, &x);
    let mut live_complex = Vec::new();
    for g in ctx.bond_gauges() {
        for t in g.r_inv.sectors() {
            let Some(m) = g.r_inv.block(t) else {
                continue;
            };
            let (mut im, mut mx) = (0.0f64, 0.0f64);
            for j in 0..m.ncols() {
                for i in 0..m.nrows() {
                    let z: C64 = m[(i, j)];
                    im = im.max(z.im.abs());
                    mx = mx.max(z.norm());
                }
            }
            let right = base_r.core(g.k).horizontal_unfolding(t);
            let left = base_l.core(g.k - 1).vertical_unfolding(t);
            if right.ncols() > right.nrows() && left.nrows() > left.ncols() && im > 1e-8 * mx {
                live_complex.push((g.k, t, m.nrows(), im));
            }
        }
    }
    println!(
        "h5 d5: live complex conversion blocks (bond, sector, size, max|Im|) {live_complex:?}"
    );
    assert!(
        !live_complex.is_empty(),
        "premise: no complex conversion block with room on both sides — the conjugate is untested"
    );
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0E6);
    let dir: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
    let xi = xi.scale(C64::from_f64(1.0 / xi.norm()));
    let dir2: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
    let eta = Su2TtTangent::project(&base_l, &base_r, &dir2);
    let eta = eta.scale(C64::from_f64(1.0 / eta.norm()));

    let a_xi = ctx.a_apply(&xi);
    let b_xi = ctx.b_apply(&xi);
    let b_eta = ctx.b_apply(&eta);
    assert!(b_xi.norm() > 1e-8, "B vacuous over C64 SU(2) at d = 5");
    let b_sym = (eta.dot(&b_xi) - xi.dot(&b_eta)).abs_s() / b_xi.norm();
    let fd = hess_fd_central(&ham, &x, &xi, 1e-4);
    let rel = fd.sub(&a_xi.add(&b_xi)).norm() / fd.norm();
    let b_rel = fd.sub(&a_xi).sub(&b_xi).norm() / b_xi.norm().max(1e-30);
    println!(
        "h5 d5: complex symmetry {b_sym:.3e}, FD rel {rel:.3e}, B-isolated {b_rel:.3e} \
         (‖Bξ‖/‖FD‖ {:.3e})",
        b_xi.norm() / fd.norm()
    );
    assert!(
        b_sym < 1e-10,
        "B not complex-symmetric over C64 SU(2) at d = 5: {b_sym:.3e}"
    );
    assert!(
        eta.dot(&b_xi).im_s().abs() > 1e-8,
        "Im⟨η,Bξ⟩ vacuous — complex SU(2) layer untested at d = 5"
    );
    assert!(
        rel < 1e-6,
        "‖FD[ξ] − (A+B)ξ‖/‖FD‖ = {rel:.3e} over C64 SU(2) at d = 5"
    );
    assert!(
        b_rel < 1e-3,
        "‖(FD − Aξ) − Bξ‖/‖Bξ‖ = {b_rel:.3e} over C64 SU(2) at d = 5"
    );
    assert!(
        b_xi.norm() > 1e-4 * fd.norm(),
        "B too small to pin at d = 5 (‖Bξ‖/‖FD‖ = {:.3e})",
        b_xi.norm() / fd.norm()
    );
}

fn rayleigh_tilde(ham: &Su2SparseHamiltonian<f64>, y: &Su2TtVector<f64>) -> f64 {
    xt_h_y(y, ham, y).re_s() / (2.0 * su2_dot(y, y).re_s())
}

fn radial_mode_at(
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    x: &Su2TtVector<f64>,
) -> Su2TtTangent<f64> {
    let mut r = Su2TtTangent::project(bl, br, x);
    let n = r.norm();
    r.scale_mut(1.0 / n);
    r
}

fn deflate_radial_span(radial: &Su2TtTangent<f64>, v: &mut Su2TtTangent<f64>) {
    for _ in 0..2 {
        let c = radial.dot(v);
        v.axpy(-c, radial);
    }
}

fn horizontal_unit_tangent(
    space: Su2Space,
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    radial: &Su2TtTangent<f64>,
    rng: &mut Xoshiro256PlusPlus,
) -> Su2TtTangent<f64> {
    let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, rng);
    let mut xi = Su2TtTangent::project(bl, br, &dir);
    deflate_radial_span(radial, &mut xi);
    let n = xi.norm();
    xi.scale_mut(1.0 / n);
    xi
}

fn corrected_form(
    ham: &Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    g_vec: &Su2TtVector<f64>,
    xi: &Su2TtTangent<f64>,
    h: f64,
) -> (f64, f64, f64) {
    let cp = xi.retract_hosvd(h);
    let cm = xi.retract_hosvd(-h);
    let m0 = rayleigh_tilde(ham, x);
    let mp = rayleigh_tilde(ham, &cp);
    let mm = rayleigh_tilde(ham, &cm);
    let q_uncorr = (mp - 2.0 * m0 + mm) / (h * h);
    let corr = (su2_dot(g_vec, &cp) + su2_dot(g_vec, &cm) - 2.0 * su2_dot(g_vec, x)) / (h * h);
    (q_uncorr - corr, q_uncorr, corr)
}

fn corrected_form_richardson(
    ham: &Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    g_vec: &Su2TtVector<f64>,
    xi: &Su2TtTangent<f64>,
    h: f64,
) -> f64 {
    let (q1, _, _) = corrected_form(ham, x, g_vec, xi, h);
    let (q2, _, _) = corrected_form(ham, x, g_vec, xi, h / 2.0);
    (4.0 * q2 - q1) / 3.0
}

fn noncritical_point(
    xstar: &Su2TtVector<f64>,
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    alpha: f64,
    seed: u64,
) -> Su2TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let dir: Su2TtVector<f64> = randomized_su2_state(xstar.space, 1, 2, &mut rng);
    let mut xi = Su2TtTangent::project(bl, br, &dir);
    let n = xi.norm();
    xi.scale_mut(alpha / n);
    let mut xp = xi.retract_hosvd(1.0);
    normalize_mut(&mut xp);
    xp.check();
    xp
}

#[test]
fn h5_corrected_fd_certifies_hess_at_noncritical_point() {
    let d = 5usize;
    let space = Su2Space::new(d, 3, 1);
    let (ham, xstar, bl_star, br_star) =
        su2_minimizer(generic_su2_h::<f64>(space), space, 0xD0D1, 1, 2);

    {
        let ctx = weingarten_operator(&ham, &xstar);
        let (bl, br) = ctx.bases();
        let g_vec = ctx.gradient().to_vector();
        let radial = radial_mode_at(bl, br, &xstar);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0EA);
        let xi = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
        let q_ab = xi.dot(&ctx.a_apply_fused(&xi).add(&ctx.b_apply(&xi)));
        let s = q_ab.abs().max(1.0);
        let (_q, _qu, corr) = corrected_form(&ham, &xstar, &g_vec, &xi, 1e-4);
        assert!(
            corr.abs() < 1e-10,
            "SU(2) critical point: corr = {corr:.3e} — retraction no longer second-order?"
        );
        let q_rich = corrected_form_richardson(&ham, &xstar, &g_vec, &xi, 2e-4);
        assert!(
            (q_rich - q_ab).abs() < 1e-6 * s,
            "critical point: |q_corr − q_AB|/s = {:.3e}",
            (q_rich - q_ab).abs() / s
        );
    }

    let x = noncritical_point(&xstar, &bl_star, &br_star, 3e-2, 0xD0EB);
    let ctx = weingarten_operator(&ham, &x);
    let grad = ctx.gradient().clone();
    let gnorm = grad.norm();
    assert!(
        gnorm > 1e-3,
        "perturbation vacuous: ‖grad‖ = {gnorm:.3e} at α=3e-2"
    );
    let (bl, br) = ctx.bases();
    let g_vec = grad.to_vector();
    let radial = radial_mode_at(bl, br, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0EC);

    let mut b_norm_max = 0.0f64;
    let mut vec_bias_max = 0.0f64;
    for trial in 0..3 {
        let xi = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
        let iso = (su2_dot(&g_vec, &xi.to_vector()) - grad.dot(&xi)).abs();
        assert!(
            iso < 1e-12,
            "trial {trial}: ι-isometry violated: |⟨ι(grad),ι(ξ)⟩_amb − ⟨grad,ξ⟩_cw| = {iso:.3e}"
        );
        let a_xi = ctx.a_apply_fused(&xi);
        let b_xi = ctx.b_apply(&xi);
        let q_ab = xi.dot(&a_xi.add(&b_xi));
        let s = q_ab.abs().max(1.0);
        let q_rich = corrected_form_richardson(&ham, &x, &g_vec, &xi, 2e-4);
        assert!(
            (q_rich - q_ab).abs() < 1e-6 * s,
            "trial {trial}: |q_corr − q_AB|/s = {:.3e} at ‖grad‖ = {gnorm:.3e}",
            (q_rich - q_ab).abs() / s
        );
        let fd_vec = hess_fd_central(&ham, &x, &xi, 1e-4);
        let hess = a_xi.add(&b_xi);
        vec_bias_max = vec_bias_max.max(fd_vec.sub(&hess).norm() / fd_vec.norm());
        b_norm_max = b_norm_max.max(b_xi.norm());
    }
    assert!(
        b_norm_max > 1e-3,
        "term B vacuous at the perturbed point (max ‖B·ξ‖ = {b_norm_max:.3e})"
    );
    assert!(
        vec_bias_max > 1e-4,
        "vector FD-of-gradient shows no transport bias at ‖grad‖ = {gnorm:.3e} \
         (max rel = {vec_bias_max:.3e}) — the corrected-oracle comparison is then vacuous"
    );
}
