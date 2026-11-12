// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::core::tangent::TtTangent;
use qntt::{C64, QnSpace, Scalar, SparseHamiltonian, TtVector, dot, rayleigh_quotient};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn t8j_symplectic_linear_energy_full_rank() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        SymplecticSpectrum, hessian_grams, symplectic_from_grams, weingarten_operator,
    };
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B2, 1, 1);

    let mut states: Vec<TtVector<f64>> = Vec::new();
    for up in 0..d {
        for dn in 0..d {
            states.push(state_from_sites(d, &[up], &[dn]));
        }
    }
    let nd = states.len();
    let mut hmat = vec![vec![0.0f64; nd]; nd];
    for (j, sj) in states.iter().enumerate() {
        let hs = &ham * sj;
        for (i, si) in states.iter().enumerate() {
            hmat[i][j] = dot(si, &hs);
        }
    }
    let evals = sym_eigvals_jacobi(hmat);
    let lam = rayleigh_quotient(&ham, &x).re_s();
    assert!(
        (lam - evals[0]).abs() < 1e-8,
        "manifold minimizer vs dense ground energy: {lam} vs {}",
        evals[0]
    );

    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    assert_eq!(
        basis.len(),
        nd - 1,
        "projective tangent dimension (radial excluded)"
    );
    let (a, b) = hessian_grams(&ctx, &basis);
    let bmax = b.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax < 1e-10,
        "B must vanish on the full manifold: {bmax:.3e}"
    );
    match symplectic_from_grams(&a, &b, basis.len(), 1e-10) {
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
fn t8j_symplectic_generator_cross_check() {
    use qntt::core::tangent::tangent_basis;
    use qntt::core::weingarten::rpa_generator_eigenvalues;
    use qntt::solvers::hess::{
        SymplecticSpectrum, hessian_grams, symplectic_from_grams, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
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

    let SymplecticSpectrum::Stable { omega } = symplectic_from_grams(&a, &b, n, 1e-10) else {
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

fn cast_tt_c64(x: &TtVector<f64>) -> TtVector<C64> {
    let space = x.space;
    let cores: Vec<qntt::SparseCore<C64>> = (0..space.d)
        .map(|k| {
            let c = x.core(k);
            let mut out = qntt::SparseCore::<C64>::zeros(space, k, &x.ranks[k], &x.ranks[k + 1]);
            for i in 0..c.num_coeffs() {
                out.set_coeff(i, C64::new(c.coeff(i), 0.0));
            }
            out
        })
        .collect();
    TtVector::new(space, cores)
}

#[test]
fn t8k_weingarten_structure_c64() {
    use qntt::solvers::hess::weingarten_operator;
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0D1);
    let dir1: TtVector<C64> = randomized_state(space, 1, 2, &mut rng);
    let dir2: TtVector<C64> = randomized_state(space, 1, 2, &mut rng);
    let xi = TtTangent::project(&base_l, &base_r, &dir1);
    let xi = xi.scale(C64::from_f64(1.0 / xi.norm()));
    let eta = TtTangent::project(&base_l, &base_r, &dir2);
    let eta = eta.scale(C64::from_f64(1.0 / eta.norm()));

    let a_xi = ctx.a_apply(&xi);
    let b_xi = ctx.b_apply(&xi);
    let a_eta = ctx.a_apply(&eta);
    let b_eta = ctx.b_apply(&eta);
    assert!(
        b_xi.norm() > 1e-8,
        "B vacuous over C64 (‖B·ξ‖ = {:.3e}) — fixture not truncated",
        b_xi.norm()
    );

    let c = C64::new(0.7, -1.3);
    let a_lin = ctx.a_apply(&xi.scale(c)).sub(&a_xi.scale(c)).norm() / a_xi.norm();
    assert!(a_lin < 1e-12, "A not ℂ-linear over C64: {a_lin:.3e}");
    let a_herm = (eta.dot(&a_xi) - a_eta.dot(&xi)).abs_s() / a_xi.norm();
    assert!(a_herm < 1e-10, "A not Hermitian over C64: {a_herm:.3e}");
    let b_cxi = ctx.b_apply(&xi.scale(c));
    let b_anti = b_cxi.sub(&b_xi.scale(c.conj_s())).norm() / b_xi.norm();
    assert!(b_anti < 1e-10, "B not antilinear over C64: {b_anti:.3e}");
    let b_lin_hyp = b_cxi.sub(&b_xi.scale(c)).norm() / b_xi.norm();
    assert!(
        b_lin_hyp > 1e-2,
        "linear-B hypothesis did not fail ({b_lin_hyp:.3e}) — antilinearity untestable at this ξ"
    );
    let b_sym = (eta.dot(&b_xi) - xi.dot(&b_eta)).abs_s() / b_xi.norm();
    assert!(
        b_sym < 1e-10,
        "B not complex-symmetric over C64: {b_sym:.3e}"
    );
    assert!(
        eta.dot(&b_xi).im_s().abs() > 1e-8,
        "Im⟨η,Bξ⟩ = {:.3e} — complex layer trivial, fixture proves nothing",
        eta.dot(&b_xi).im_s()
    );
}

#[test]
fn t8k_hess_apply_matches_fd_of_gradient_c64() {
    use qntt::solvers::hess::weingarten_operator;
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C2, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0D2);
    let dir: TtVector<C64> = randomized_state(space, 1, 2, &mut rng);
    let xi = TtTangent::project(&base_l, &base_r, &dir);
    let xi = xi.scale(C64::from_f64(1.0 / xi.norm()));

    let a_xi = ctx.a_apply(&xi);
    let b_xi = ctx.b_apply(&xi);
    let fd = hess_fd_central(&ham, &x, &xi, 1e-4);
    let rel = fd.sub(&a_xi.add(&b_xi)).norm() / fd.norm();
    assert!(rel < 1e-6, "‖FD[ξ] − (A+B)ξ‖/‖FD‖ = {rel:.3e} over C64");
    let b_rel = fd.sub(&a_xi).sub(&b_xi).norm() / b_xi.norm().max(1e-30);
    assert!(b_rel < 1e-3, "‖(FD − Aξ) − Bξ‖/‖Bξ‖ = {b_rel:.3e} over C64");
    assert!(
        b_xi.norm() > 1e-4 * fd.norm(),
        "B too small to pin signs (‖Bξ‖/‖FD‖ = {:.3e})",
        b_xi.norm() / fd.norm()
    );
    let i = C64::new(0.0, 1.0);
    let fd_i = hess_fd_central(&ham, &x, &xi.scale(i), 1e-4);
    let anti = fd_i.sub(&a_xi.scale(i).sub(&b_xi.scale(i))).norm() / fd_i.norm();
    let lin = fd_i.sub(&a_xi.scale(i).add(&b_xi.scale(i))).norm() / fd_i.norm();
    assert!(anti < 1e-6, "FD[iξ] ≠ i·Aξ − i·Bξ: {anti:.3e}");
    assert!(
        lin > 100.0 * anti.max(1e-8),
        "linear-B combination did not fail (anti {anti:.3e} vs lin {lin:.3e}) — sign pin vacuous"
    );
}

#[test]
fn t8k_structure_and_fd_see_the_conjugate_conversion_at_d5_c64() {
    use qntt::solvers::hess::weingarten_operator;
    let d = 5usize;
    let space = QnSpace::new(d, 2, 2);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0F5, 1, 3);
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
        "t8k d5: live complex conversion blocks (bond, sector, size, max|Im|) {live_complex:?}"
    );
    assert!(
        !live_complex.is_empty(),
        "premise: no complex conversion block with room on both sides — the conjugate is untested"
    );
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0D2);
    let dir: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
    let xi = TtTangent::project(&base_l, &base_r, &dir);
    let xi = xi.scale(C64::from_f64(1.0 / xi.norm()));
    let dir2: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
    let eta = TtTangent::project(&base_l, &base_r, &dir2);
    let eta = eta.scale(C64::from_f64(1.0 / eta.norm()));

    let a_xi = ctx.a_apply(&xi);
    let b_xi = ctx.b_apply(&xi);
    let b_eta = ctx.b_apply(&eta);
    assert!(
        b_xi.norm() > 1e-8,
        "B vacuous over C64 at d = 5 (‖B·ξ‖ = {:.3e})",
        b_xi.norm()
    );
    let b_sym = (eta.dot(&b_xi) - xi.dot(&b_eta)).abs_s() / b_xi.norm();
    let fd = hess_fd_central(&ham, &x, &xi, 1e-4);
    let rel = fd.sub(&a_xi.add(&b_xi)).norm() / fd.norm();
    let b_rel = fd.sub(&a_xi).sub(&b_xi).norm() / b_xi.norm().max(1e-30);
    println!(
        "t8k d5: complex symmetry {b_sym:.3e}, FD rel {rel:.3e}, B-isolated {b_rel:.3e} \
         (‖Bξ‖/‖FD‖ {:.3e})",
        b_xi.norm() / fd.norm()
    );
    assert!(
        b_sym < 1e-10,
        "B not complex-symmetric over C64 at d = 5: {b_sym:.3e}"
    );
    assert!(
        eta.dot(&b_xi).im_s().abs() > 1e-8,
        "Im⟨η,Bξ⟩ = {:.3e} — complex layer trivial at d = 5",
        eta.dot(&b_xi).im_s()
    );
    assert!(
        rel < 1e-6,
        "‖FD[ξ] − (A+B)ξ‖/‖FD‖ = {rel:.3e} over C64 at d = 5"
    );
    assert!(
        b_rel < 1e-3,
        "‖(FD − Aξ) − Bξ‖/‖Bξ‖ = {b_rel:.3e} over C64 at d = 5"
    );
    assert!(
        b_xi.norm() > 1e-4 * fd.norm(),
        "B too small to pin at d = 5 (‖Bξ‖/‖FD‖ = {:.3e})",
        b_xi.norm() / fd.norm()
    );
}

#[test]
fn t8k_tangent_basis_c64_complex_span() {
    use qntt::core::tangent::tangent_basis;
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (_ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C3, 1, 1);
    let basis = tangent_basis(&base_l, &base_r, true);
    assert_eq!(basis.len(), 8, "complex projective tangent dimension");
    for (i, ui) in basis.iter().enumerate() {
        for (j, uj) in basis.iter().enumerate() {
            let g = ui.dot(uj);
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (g - C64::from_f64(want)).abs_s() < 1e-10,
                "⟨u_{i},u_{j}⟩ = {g:?}"
            );
        }
    }
    let mut radial = TtTangent::project(&base_l, &base_r, &x);
    let rn = radial.norm();
    assert!(rn > 0.9, "radial mode should be a unit tangent vector at x");
    radial.scale_mut(C64::from_f64(1.0 / rn));
    for (j, uj) in basis.iter().enumerate() {
        let overlap = radial.dot(uj).abs_s();
        assert!(overlap < 1e-10, "|⟨radial, u_{j}⟩| = {overlap:.3e}");
    }
}

#[test]
fn t8k_symplectic_linear_energy_full_rank_c64() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        SymplecticSpectrum, hessian_grams_c64, symplectic_from_grams_c64, weingarten_operator,
    };
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C4, 1, 1);

    let mut states: Vec<TtVector<C64>> = Vec::new();
    for up in 0..d {
        for dn in 0..d {
            states.push(state_from_sites(d, &[up], &[dn]));
        }
    }
    let nd = states.len();
    let mut hc = vec![vec![C64::from_f64(0.0); nd]; nd];
    for (j, sj) in states.iter().enumerate() {
        let hs = &ham * sj;
        for (i, si) in states.iter().enumerate() {
            hc[i][j] = dot(si, &hs);
        }
    }
    let mut doubled = vec![vec![0.0f64; 2 * nd]; 2 * nd];
    for i in 0..nd {
        for j in 0..nd {
            doubled[i][j] = hc[i][j].re_s();
            doubled[i][nd + j] = -hc[i][j].im_s();
            doubled[nd + i][j] = hc[i][j].im_s();
            doubled[nd + i][nd + j] = hc[i][j].re_s();
        }
    }
    let ev2 = sym_eigvals_jacobi(doubled);
    let evals: Vec<f64> = (0..nd)
        .map(|k| {
            assert!(
                (ev2[2 * k] - ev2[2 * k + 1]).abs() < 1e-9,
                "doubling pair broken at {k}"
            );
            0.5 * (ev2[2 * k] + ev2[2 * k + 1])
        })
        .collect();
    let lam = rayleigh_quotient(&ham, &x).re_s();
    assert!(
        (lam - evals[0]).abs() < 1e-8,
        "manifold minimizer vs dense ground energy: {lam} vs {}",
        evals[0]
    );

    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    assert_eq!(basis.len(), nd - 1, "complex projective tangent dimension");
    let (a, b) = hessian_grams_c64(&ctx, &basis);
    let bmax = b.iter().fold(0.0f64, |acc, v| acc.max(v.abs_s()));
    assert!(
        bmax < 1e-10,
        "B must vanish on the full manifold: {bmax:.3e}"
    );
    let im_a_max = a.iter().fold(0.0f64, |acc, v| acc.max(v.im_s().abs()));
    assert!(
        im_a_max > 1e-6,
        "Im A vacuous ({im_a_max:.3e}) — fixture not genuinely complex"
    );
    match symplectic_from_grams_c64(&a, &b, basis.len(), 1e-10) {
        SymplecticSpectrum::Stable { omega } => {
            assert_eq!(omega.len(), nd - 1, "one ω per complex tangent dimension");
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
fn t8k_symplectic_generator_cross_check_c64() {
    use qntt::core::tangent::tangent_basis;
    use qntt::core::weingarten::symplectic_generator_full;
    use qntt::solvers::hess::{
        SymplecticSpectrum, hessian_grams_c64, hll_from_grams_c64, symplectic_from_grams_c64,
        weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, x, base_l, base_r) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C5, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let n = basis.len();
    assert!(n >= 2, "need a nontrivial tangent space, got {n}");
    let (a, b) = hessian_grams_c64(&ctx, &basis);
    let bmax = b.iter().fold(0.0f64, |acc, v| acc.max(v.abs_s()));
    assert!(
        bmax > 1e-8,
        "B vacuous at the truncated minimizer ({bmax:.3e})"
    );
    let hll = hll_from_grams_c64(&a, &b, n);
    let two_n = 2 * n;
    let mut hri_max = 0.0f64;
    for j in 0..n {
        for k in 0..n {
            hri_max = hri_max.max(hll[j * two_n + (n + k)].abs());
        }
    }
    assert!(
        hri_max > 1e-6,
        "H^ri vacuous ({hri_max:.3e}) — complex layer untested"
    );

    let SymplecticSpectrum::Stable { omega } = symplectic_from_grams_c64(&a, &b, n, 1e-10) else {
        panic!("expected a stable RPA spectrum at a converged minimizer");
    };
    assert_eq!(omega.len(), n);
    assert!(omega[0] > 0.0, "ω must be positive (Williamson)");

    let eigs = symplectic_generator_full(&hll, two_n);
    assert_eq!(eigs.len(), two_n);
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
            "ω[{k}]: full Williamson {w} vs eig(J𝔥) generator {g}"
        );
    }
}

#[test]
fn t8k_symplectic_real_embedded_matches_f64() {
    use qntt::core::tangent::tangent_basis;
    use qntt::solvers::hess::{
        SymplecticSpectrum, hessian_grams, hessian_grams_c64, symplectic_from_grams,
        symplectic_from_grams_c64, weingarten_operator,
    };
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham_f, x_f, base_l_f, base_r_f) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx_f = weingarten_operator(&ham_f, &x_f);
    let basis_f = tangent_basis(&base_l_f, &base_r_f, true);
    let n = basis_f.len();
    let (a_f, b_f) = hessian_grams(&ctx_f, &basis_f);
    let SymplecticSpectrum::Stable { omega: omega_f } = symplectic_from_grams(&a_f, &b_f, n, 1e-10)
    else {
        panic!("f64 side unstable");
    };

    let ham_c: SparseHamiltonian<C64> = generic_truncated_h(space);
    let x_c = cast_tt_c64(&x_f);
    x_c.check();
    let ctx_c = weingarten_operator(&ham_c, &x_c);
    let (base_l_c, base_r_c) = ctx_c.bases();
    let basis_c = tangent_basis(base_l_c, base_r_c, true);
    assert_eq!(
        basis_c.len(),
        n,
        "complex tangent dimension must equal the real dimension on a real problem"
    );
    let (a_c, b_c) = hessian_grams_c64(&ctx_c, &basis_c);
    let im_max = a_c
        .iter()
        .chain(b_c.iter())
        .fold(0.0f64, |acc, v| acc.max(v.im_s().abs()));
    assert!(
        im_max < 1e-10,
        "H^ri must vanish on a real-embedded problem (max |Im| = {im_max:.3e})"
    );
    let SymplecticSpectrum::Stable { omega: omega_c } =
        symplectic_from_grams_c64(&a_c, &b_c, n, 1e-10)
    else {
        panic!("C64 side unstable");
    };
    assert_eq!(omega_c.len(), omega_f.len());
    for (k, (wc, wf)) in omega_c.iter().zip(omega_f.iter()).enumerate() {
        assert!(
            (wc - wf).abs() < 1e-8 * wf.max(1.0),
            "ω[{k}]: C64 full Williamson {wc} vs f64 Eq.-20 {wf}"
        );
    }
}

#[test]
fn t8k_corrected_fd_certifies_hess_at_noncritical_point_c64() {
    use qntt::solvers::hess::weingarten_operator;
    let d = 4usize;
    let space = QnSpace::new(d, 2, 1);
    let (ham, xstar, bl_star, br_star) =
        weingarten_minimizer_c64(generic_complex_h(space), space, 0xC0C2, 1, 2);

    {
        let ctx = weingarten_operator(&ham, &xstar);
        let (bl, br) = ctx.bases();
        let g_vec = ctx.gradient().to_vector();
        let radial = radial_mode_at(bl, br, &xstar);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0D8);
        let xi = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
        let (_q, _qu, corr) = corrected_form(&ham, &xstar, &g_vec, &xi, 1e-4);
        assert!(
            corr.abs() < 1e-10,
            "C64 critical point: corr = {corr:.3e} — retraction no longer second-order?"
        );
    }

    let x = noncritical_point(&xstar, &bl_star, &br_star, 3e-2, 0xC0D5);
    let ctx = weingarten_operator(&ham, &x);
    let gnorm = ctx.gradient().norm();
    assert!(
        gnorm > 1e-3,
        "perturbation vacuous: ‖grad‖ = {gnorm:.3e} at α=3e-2"
    );
    let (bl, br) = ctx.bases();
    let g_vec = ctx.gradient().to_vector();
    let radial = radial_mode_at(bl, br, &x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0D9);

    let mut re_b_max = 0.0f64;
    let mut vec_bias_max = 0.0f64;
    for trial in 0..2 {
        let xi = horizontal_unit_tangent(space, bl, br, &radial, &mut rng);
        let ixi = xi.scale(C64::new(0.0, 1.0));
        let a_xi = ctx.a_apply_fused(&xi);
        let b_xi = ctx.b_apply(&xi);
        let re_a = xi.dot(&a_xi).re_s();
        let re_b = xi.dot(&b_xi).re_s();
        let q_ab = re_a + re_b;
        let s = q_ab.abs().max(1.0);
        let q_p = corrected_form_richardson(&ham, &x, &g_vec, &xi, 2e-4);
        let q_m = corrected_form_richardson(&ham, &x, &g_vec, &ixi, 2e-4);
        assert!(
            (q_p - q_ab).abs() < 1e-6 * s,
            "trial {trial}: |q(ξ) − (ReA + ReB)|/s = {:.3e} off-critical",
            (q_p - q_ab).abs() / s
        );
        assert!(
            (0.5 * (q_p - q_m) - re_b).abs() < 1e-6 * s,
            "trial {trial}: sign pin |½(q(ξ)−q(iξ)) − Re⟨ξ,Bξ⟩| = {:.3e}",
            (0.5 * (q_p - q_m) - re_b).abs()
        );
        assert!(
            (0.5 * (q_p + q_m) - re_a).abs() < 1e-6 * s,
            "trial {trial}: |½(q(ξ)+q(iξ)) − Re⟨ξ,Aξ⟩| = {:.3e}",
            (0.5 * (q_p + q_m) - re_a).abs()
        );
        let fd_vec = hess_fd_central(&ham, &x, &xi, 1e-4);
        let hess = ctx.hess_apply(&xi);
        vec_bias_max = vec_bias_max.max(fd_vec.sub(&hess).norm() / fd_vec.norm());
        re_b_max = re_b_max.max(re_b.abs());
    }
    assert!(
        re_b_max > 1e-3,
        "sign pin vacuous: max |Re⟨ξ,Bξ⟩| = {re_b_max:.3e} (need ≫ the 1e-6·s gate)"
    );
    assert!(
        vec_bias_max > 1e-3,
        "vector FD-of-gradient shows no transport bias at ‖grad‖ = {gnorm:.3e} \
         (max rel = {vec_bias_max:.3e})"
    );
}
