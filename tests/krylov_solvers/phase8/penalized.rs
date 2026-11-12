// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::penalized_pins::{gauge_integrals, generic_tv, rho_abelian};
use faer::Mat;
use qntt::C64;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::core::tangent::{TtTangent, tangent_dim};
use qntt::qn::QnSpace;
use qntt::round::RoundMode;
use qntt::solvers::penalty::Penalty;
use qntt::solvers::riemannian::{
    CgVariant, RiemannianOpts, normalize_mut, penalized_exact_gradnorm, riemannian_cg,
    riemannian_cg_penalized, riemannian_gradient, riemannian_gradient_penalized, tangent_bases,
};
use qntt::{Scalar, SparseHamiltonian, TtVector, max_ranks, tt_randn, tt_state};
use qntt_testkit::dense::{basis_index, dense_eigen, dense_hamiltonian_matrix, dets_for};
use qntt_testkit::{randomized_state, seeded};

fn bits_eq<T: Scalar>(a: T, b: T) -> bool {
    a.re_s().to_bits() == b.re_s().to_bits() && a.im_s().to_bits() == b.im_s().to_bits()
}

fn assert_f64_bits(got: f64, want: f64, what: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{what}: {got:.17e} vs {want:.17e} (bit-identity required — an inactive penalty must take \
         the bare path, not a numerically-equal one)"
    );
}

fn coeffs_bits_eq_tt<T: Scalar>(a: &TtVector<T>, b: &TtVector<T>, what: &str) {
    assert_eq!(a.space, b.space, "{what}: space");
    assert_eq!(a.ortho, b.ortho, "{what}: ortho flag");
    for (k, (ra, rb)) in a.ranks.iter().zip(b.ranks.iter()).enumerate() {
        assert_eq!(ra, rb, "{what}: rank grid at bond {k}");
    }
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

fn tt_coeffs_identical<T: Scalar>(a: &TtVector<T>, b: &TtVector<T>) -> bool {
    if a.ranks != b.ranks {
        return false;
    }
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        if ca.num_coeffs() != cb.num_coeffs() {
            return false;
        }
        for i in 0..ca.num_coeffs() {
            if !bits_eq(ca.coeff(i), cb.coeff(i)) {
                return false;
            }
        }
    }
    true
}

fn tangent_bits_eq<T: Scalar>(a: &TtTangent<T>, b: &TtTangent<T>, what: &str) {
    assert_eq!(a.num_coeffs(), b.num_coeffs(), "{what}: coefficient count");
    for i in 0..a.num_coeffs() {
        assert!(bits_eq(a.coeff(i), b.coeff(i)), "{what}: coefficient {i}");
    }
}

fn unit_state<T: Scalar>(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<T> {
    let mut rng = seeded(seed);
    let mut x: TtVector<T> = randomized_state(space, lo, hi, &mut rng);
    x.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    normalize_mut(&mut x);
    x
}

fn j_functional<T: Scalar>(
    h: &SparseHamiltonian<T>,
    y: &TtVector<T>,
    psis: &[TtVector<T>],
    w: f64,
) -> f64 {
    let num_h = qntt::xt_h_y(y, h, y).re_s();
    let mut pen = 0.0f64;
    for p in psis {
        let invn = 1.0 / p.norm();
        let o = qntt::dot(p, y) * T::from_f64(invn);
        pen += o.re_s() * o.re_s() + o.im_s() * o.im_s();
    }
    (num_h + w * pen) / qntt::dot(y, y).re_s()
}

fn overlaps_ref<T: Scalar>(psis: &[TtVector<T>], y: &TtVector<T>) -> Vec<T> {
    psis.iter()
        .map(|p| qntt::dot(p, y) * T::from_f64(1.0 / p.norm()))
        .collect()
}

fn richardson_dj<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    v: &TtVector<T>,
    psis: &[TtVector<T>],
    w: f64,
    step: f64,
) -> f64 {
    let d = |t: f64| {
        let plus = x + &(v * T::from_f64(t));
        let minus = x + &(v * T::from_f64(-t));
        (j_functional(h, &plus, psis, w) - j_functional(h, &minus, psis, w)) / (2.0 * t)
    };
    (4.0 * d(step / 2.0) - d(step)) / 3.0
}

fn penalized_dense(hd: &[Vec<f64>], chats: &[Vec<f64>], w: f64) -> Vec<Vec<f64>> {
    let n = hd.len();
    let mut m = hd.to_vec();
    for c in chats {
        assert_eq!(c.len(), n, "penalized_dense: dimension mismatch");
        for i in 0..n {
            for j in 0..n {
                m[i][j] += w * c[i] * c[j];
            }
        }
    }
    m
}

fn dense_unit<T: Scalar>(x: &TtVector<T>, dets: &[(Vec<bool>, Vec<bool>)], tag: &str) -> Vec<f64> {
    let xd = x.to_dense();
    let full: f64 = xd.iter().map(|c| c.abs_s().powi(2)).sum::<f64>().sqrt();
    let mut c: Vec<f64> = dets
        .iter()
        .map(|(ou, od)| xd[basis_index(ou, od)].re_s())
        .collect();
    let imax = dets
        .iter()
        .map(|(ou, od)| xd[basis_index(ou, od)].im_s().abs())
        .fold(0.0f64, f64::max);
    assert!(
        imax <= 1e-12 * full.max(1e-300),
        "{tag}: dense_unit is a REAL oracle but the state carries an imaginary part {imax:.3e}"
    );
    let nrm = c.iter().map(|val| val * val).sum::<f64>().sqrt();
    assert!(
        nrm >= (1.0 - 1e-10) * full,
        "{tag}: the state leaks outside the determinant sector ({nrm:.12e} of {full:.12e})"
    );
    for val in c.iter_mut() {
        *val /= nrm.max(1e-300);
    }
    c
}

fn projection_on_dense<T: Scalar>(
    x: &TtVector<T>,
    dets: &[(Vec<bool>, Vec<bool>)],
    evec: &[f64],
) -> f64 {
    let xd = x.to_dense();
    let mut acc = T::zero();
    let mut ne = 0.0f64;
    for (pos, (ou, od)) in dets.iter().enumerate() {
        acc += T::from_f64(evec[pos]) * xd[basis_index(ou, od)];
        ne += evec[pos] * evec[pos];
    }
    let nx: f64 = xd.iter().map(|c| c.abs_s().powi(2)).sum::<f64>().sqrt();
    acc.abs_s() / (nx * ne.sqrt()).max(1e-300)
}

fn d2b_rows<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
    space: QnSpace,
    psi: &TtVector<T>,
) {
    let opts = RiemannianOpts {
        eps: 1e-6,
        max_iter: 12,
        gradtol: 1e-8,
        variant: CgVariant::Pr,
        ..RiemannianOpts::new(&space)
    };
    let empty: Penalty<TtVector<T>> = Penalty::new(0.0, &[]);
    let zero_w: Penalty<TtVector<T>> = Penalty::new(0.0, std::slice::from_ref(psi));
    let no_states: Penalty<TtVector<T>> = Penalty::new(3.0, &[]);
    for p in [&empty, &zero_w, &no_states] {
        assert!(
            !p.is_active(),
            "an inactive penalty (w = {}, {} states) reports itself active; the `w > 0 ∧ \
             !states.is_empty()` guard is what makes the fold below exact",
            p.w,
            p.states.len()
        );
    }

    let (l_bare, x_bare) = riemannian_cg(h, x0, &opts);
    for (tag, p) in [
        ("empty", &empty),
        ("w=0", &zero_w),
        ("no states", &no_states),
    ] {
        let (j, xj) = riemannian_cg_penalized(h, x0, p, &opts);
        assert_f64_bits(
            j,
            l_bare,
            &format!("riemannian_cg_penalized ({tag}) J vs λ"),
        );
        coeffs_bits_eq_tt(&xj, &x_bare, &format!("riemannian_cg_penalized ({tag}) x"));
    }

    let fr = RiemannianOpts {
        variant: CgVariant::Fr,
        ..opts.clone()
    };
    let (l_fr, x_fr) = riemannian_cg(h, x0, &fr);
    let (j_fr, xj_fr) = riemannian_cg_penalized(h, x0, &empty, &fr);
    assert_f64_bits(j_fr, l_fr, "riemannian_cg_penalized Fr J vs λ");
    coeffs_bits_eq_tt(&xj_fr, &x_fr, "riemannian_cg_penalized Fr x");

    let (bl, br) = tangent_bases(&x_bare);
    let (g_bare, l_g) = riemannian_gradient(h, &x_bare, &bl, &br, 1e-12, 1234);
    let (g_pen, j_g) = riemannian_gradient_penalized(h, &x_bare, &bl, &br, &empty, 1e-12, 1234);
    assert_f64_bits(j_g, l_g, "riemannian_gradient_penalized (empty) J vs λ");
    tangent_bits_eq(&g_pen, &g_bare, "riemannian_gradient_penalized (empty) g");

    let gate_pen = penalized_exact_gradnorm(h, &x_bare, &empty);
    let gate_bare = qntt::solvers::hess::weingarten_operator(h, &x_bare)
        .gradient()
        .norm();
    assert_f64_bits(
        gate_pen,
        gate_bare,
        "penalized_exact_gradnorm (empty) vs the response pipeline's gate",
    );
}

#[test]
fn tms_d2b_inactive_penalty_is_the_bare_path_f64() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x0: TtVector<f64> = unit_state(space, 3, 5, 0xD2B_0001);
    let psi: TtVector<f64> = unit_state(space, 2, 4, 0xD2B_0002);
    d2b_rows(&h, &x0, space, &psi);
}

#[test]
fn tms_d2b_inactive_penalty_is_the_bare_path_c64() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
    let h = SparseHamiltonian::new(space, &tc, &vc, None, f64::EPSILON);
    let x0: TtVector<C64> = unit_state(space, 3, 5, 0xD2B_0003);
    let psi: TtVector<C64> = unit_state(space, 2, 4, 0xD2B_0004);
    let rho = rho_abelian(&x0);
    assert!(rho < 0.99, "the C64 start is phase-trivial (ρ = {rho:.6})");
    d2b_rows(&h, &x0, space, &psi);
}

fn d3_rows<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    psi1: &TtVector<T>,
    psi2: &TtVector<T>,
    space: QnSpace,
    seed: u64,
    tag: &str,
) {
    assert!(
        (x.norm() - 1.0).abs() <= 1e-12,
        "{tag}: the base point must be unit ({:.3e})",
        x.norm()
    );
    assert!(
        (psi2.norm() - 2.0).abs() <= 1e-12,
        "{tag}: ψ₂ must have norm 2 so the `invn` fold is exercised (got {:.6})",
        psi2.norm()
    );
    let (bl, br) = tangent_bases(x);
    let lambda = qntt::rayleigh_quotient(h, x).re_s();
    let psis = [psi1.clone(), psi2.clone()];

    let mut rng = seeded(seed);
    let mut accepted = 0usize;
    let mut dirs: Vec<TtTangent<T>> = Vec::new();
    for _ in 0..12 {
        let mut dir: TtVector<T> = randomized_state(space, 1, 3, &mut rng);
        dir.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
        let mut xi = TtTangent::project(&bl, &br, &dir);
        let n = xi.norm();
        if n <= 1e-8 {
            continue;
        }
        xi.scale_mut(T::from_f64(1.0 / n));
        dirs.push(xi);
        accepted += 1;
        if accepted >= 6 {
            break;
        }
    }
    assert!(
        dirs.len() >= 3,
        "{tag}: fewer than 3 usable tangent directions were drawn"
    );

    for &w in &[1e-2f64, 1.0, 1e3] {
        let p = Penalty::new(w, &psis);
        assert!(p.is_active(), "{tag}: w = {w} with 2 states must be active");
        let (g, j) = riemannian_gradient_penalized(h, x, &bl, &br, &p, 1e-12, 1234);

        let os = overlaps_ref(&psis, x);
        let pen: f64 = w * os
            .iter()
            .map(|o| o.re_s() * o.re_s() + o.im_s() * o.im_s())
            .sum::<f64>();
        let j_ref = j_functional(h, x, &psis, w);
        assert!(
            (j - j_ref).abs() <= 1e-12 * j.abs().max(1.0),
            "{tag} w={w}: the driver's J {j:.16e} differs from the package-verb functional \
             {j_ref:.16e} by {:.3e}",
            (j - j_ref).abs()
        );
        assert!(
            (j - lambda - pen).abs() <= 1e-12 * j.abs().max(1.0),
            "{tag} w={w}: J − λ = {:.16e} but w·Σ|oⱼ|² = {pen:.16e}",
            j - lambda
        );
        assert!(
            j > lambda,
            "{tag} w={w}: the penalty must RAISE the functional (J = {j:.6e}, λ = {lambda:.6e}); \
             a nonzero overlap with the lower states is what makes this arm non-vacuous"
        );
        assert!(
            pen >= 1e-6 * w,
            "{tag} w={w}: the base point is (near-)orthogonal to both penalty states \
             (pen = {pen:.3e}); the arm would not discriminate the penalty term"
        );

        let mut checked = 0usize;
        for xi in &dirs {
            let vv = xi.to_vector();
            let r = richardson_dj(h, x, &vv, &psis, w, 1e-3);
            if r.abs() < 1e-2 {
                continue;
            }
            let analytic = 2.0 * g.dot(xi).re_s();
            let rel = (r - analytic).abs() / r.abs();
            println!(
                "tms_d3 {tag} w={w}: FD {r:.12e} vs 2·Re⟨g,ξ⟩ {analytic:.12e} (rel {rel:.3e})"
            );
            assert!(
                rel <= 1e-8,
                "{tag} w={w}: the penalized gradient does not differentiate J (rel {rel:.3e}; \
                 Richardson truncation at h = 1e-3 is ~1e-13, so 1e-8 carries decades of headroom)"
            );
            checked += 1;
            if checked >= 3 {
                break;
            }
        }
        assert!(
            checked >= 3,
            "{tag} w={w}: only {checked} directions produced a resolvable FD signal"
        );

        let xir = TtTangent::project(&bl, &br, x);
        assert!(
            (xir.norm() - 1.0).abs() <= 1e-12,
            "{tag}: P(x) is not unit ({:.3e}) — x is not on the manifold",
            xir.norm()
        );
        let vr = xir.to_vector();
        let rr = richardson_dj(h, x, &vr, &psis, w, 1e-3);
        assert!(
            rr.abs() <= 1e-7,
            "{tag} w={w}: J is not radially flat (FD along P(x) = {rr:.3e}); either the fixture \
             or the functional is wrong"
        );
        let radial = 2.0 * g.dot(&xir).re_s();
        assert!(
            radial.abs() <= 1e-9,
            "{tag} w={w}: the J-shifted gradient has a spurious radial component {radial:.3e} \
             (a λ-shift leaves 2·pen = {:.3e} there — see the anti-oracle below)",
            2.0 * pen
        );

        let hx = h * x;
        let mut glam = TtTangent::project(&bl, &br, &hx);
        glam.axpy(T::from_f64(-lambda), &TtTangent::project(&bl, &br, x));
        for (o, psi) in os.iter().zip(psis.iter()) {
            let invn = 1.0 / psi.norm();
            glam.axpy(
                T::from_f64(w * invn) * *o,
                &TtTangent::project(&bl, &br, psi),
            );
        }
        assert!(
            2.0 * pen >= 1e-3,
            "{tag} w={w}: 2·pen = {:.3e} is below the anti-oracle's own resolution; this arm \
             would not discriminate a λ-shift",
            2.0 * pen
        );
        let radial_lambda = 2.0 * glam.dot(&xir).re_s();
        assert!(
            (radial_lambda - 2.0 * pen).abs() <= 1e-9,
            "{tag} w={w}: the λ-shifted gradient's radial component is {radial_lambda:.6e}, not \
             2·pen = {:.6e}",
            2.0 * pen
        );

        let mut psi2_big = psi2.clone();
        psi2_big.scale_mut(T::from_f64(3.0));
        let scaled = [psi1.clone(), psi2_big];
        let p2 = Penalty::new(w, &scaled);
        let (g2, j2) = riemannian_gradient_penalized(h, x, &bl, &br, &p2, 1e-12, 1234);
        assert!(
            (j2 - j).abs() <= 1e-12 * j.abs().max(1.0),
            "{tag} w={w}: rescaling ψ₂ by 3 moved J by {:.3e}; `invn` is not folded",
            (j2 - j).abs()
        );
        let mut dev = 0.0f64;
        for i in 0..g.num_coeffs() {
            dev = dev.max((g2.coeff(i) - g.coeff(i)).abs_s());
        }
        assert!(
            dev <= 1e-12 * g.norm().max(1.0),
            "{tag} w={w}: rescaling ψ₂ by 3 moved the gradient by {dev:.3e}"
        );
    }
}

#[test]
fn tms_d3_penalized_gradient_matches_richardson_fd_f64() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x: TtVector<f64> = unit_state(space, 3, 5, 0xD3_0001);
    let psi1: TtVector<f64> = unit_state(space, 2, 4, 0xD3_0002);
    let mut psi2: TtVector<f64> = unit_state(space, 2, 4, 0xD3_0003);
    psi2.scale_mut(2.0);
    d3_rows(&h, &x, &psi1, &psi2, space, 0xD3_0010, "f64");
}

#[test]
fn tms_d3_penalized_gradient_matches_richardson_fd_c64() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
    let h = SparseHamiltonian::new(space, &tc, &vc, None, f64::EPSILON);
    let x: TtVector<C64> = unit_state(space, 3, 5, 0xD3_0004);
    let psi1: TtVector<C64> = unit_state(space, 2, 4, 0xD3_0005);
    let mut psi2: TtVector<C64> = unit_state(space, 2, 4, 0xD3_0006);
    psi2.scale_mut(C64::from_f64(2.0));

    let rho = rho_abelian(&x);
    assert!(
        rho < 0.99,
        "the C64 base point is phase-trivial (ρ = {rho:.6})"
    );
    let o1 = qntt::dot(&psi1, &x);
    assert!(
        o1.im_s().abs() > 1e-3,
        "the C64 arm needs a genuinely complex overlap to discriminate `oⱼ` from `conj(oⱼ)` \
         (Im⟨ψ̂₁,x⟩ = {:.3e})",
        o1.im_s()
    );
    d3_rows(&h, &x, &psi1, &psi2, space, 0xD3_0011, "c64");
}

#[test]
fn tms_d4_penalized_cg_reaches_the_dense_minimizer() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let psi: TtVector<f64> = unit_state(space, 2, 4, 0xD4_0001);
    let w = 3.0f64;

    let mut rng = seeded(0xD4_0002);
    let mut x0: TtVector<f64> = tt_randn(space, &max_ranks(&space), &mut rng);
    normalize_mut(&mut x0);
    let (bl0, _br0) = tangent_bases(&x0);
    let dets = dets_for(d, 1, 1);
    let tdim = tangent_dim(&bl0);
    println!(
        "tms_d4: tangent_dim = {tdim}, determinant count = {}",
        dets.len()
    );
    assert_eq!(
        tdim,
        dets.len(),
        "the start is not at full representing rank, so the manifold is not the whole sector and \
         'the dense minimizer' is not the right oracle"
    );

    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let chat = dense_unit(&psi, &dets, "tms_d4 ψ̂");
    let m = penalized_dense(&hd, std::slice::from_ref(&chat), w);
    let (mu, mvec) = dense_eigen(&m);
    println!("tms_d4: μ₀ = {:.12e}, μ₁ = {:.12e}", mu[0], mu[1]);
    assert!(
        mu[1] - mu[0] >= 1e-3,
        "the penalized dense spectrum is not resolved at the bottom (μ₁ − μ₀ = {:.3e}); the \
         minimizer claim would be ill-posed",
        mu[1] - mu[0]
    );

    let p = Penalty::new(w, std::slice::from_ref(&psi));
    let opts = RiemannianOpts {
        eps: 1e-13,
        max_iter: 600,
        gradtol: 1e-12,
        preserve_grid: true,
        ..RiemannianOpts::new(&space)
    };
    let (mut j, mut x) = riemannian_cg_penalized(&h, &x0, &p, &opts);
    let mut gate = penalized_exact_gradnorm(&h, &x, &p);
    let mut restarts = 0usize;
    while gate > 1e-10 && restarts < 10 {
        let (j2, x2) = riemannian_cg_penalized(&h, &x, &p, &opts);
        j = j2;
        x = x2;
        gate = penalized_exact_gradnorm(&h, &x, &p);
        restarts += 1;
    }
    println!("tms_d4: restarts used {restarts}/10, gate {gate:.3e}, J {j:.16e}");
    assert!(
        restarts <= 3,
        "the polish used {restarts} restarts of a 10-restart budget — under the 3× headroom a \
         count needs before it is trustworthy; report it rather than raising the budget"
    );
    assert!(
        gate <= 1e-10,
        "the polish did not converge (gate {gate:.3e})"
    );

    assert!(
        (j - mu[0]).abs() <= 1e-10,
        "the penalized CG's J {j:.16e} misses the dense penalized minimum {:.16e} by {:.3e}",
        mu[0],
        (j - mu[0]).abs()
    );
    let proj = projection_on_dense(&x, &dets, &mvec[0]);
    assert!(
        proj >= 1.0 - 1e-8,
        "the polished state is not the dense penalized minimizer (projection {proj:.12})"
    );
    assert!(
        (x.norm() - 1.0).abs() <= 1e-12,
        "the returned state is not unit ({:.3e})",
        x.norm()
    );

    let lam = qntt::rayleigh_quotient(&h, &x).re_s();
    let o = qntt::dot(&psi, &x).re_s();
    assert!(
        (lam + w * o * o - j).abs() <= 1e-12 * j.abs().max(1.0),
        "J is not λ + w|⟨ψ̂,x⟩|² at the returned root (λ = {lam:.16e}, o² = {:.3e}, J = {j:.16e})",
        o * o
    );
}

fn quotient_rows<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    psis: &[TtVector<T>],
    space: QnSpace,
    seed: u64,
    tag: &str,
) {
    let w = 3.0f64;
    let p = Penalty::new(w, psis);
    let (bl, br) = tangent_bases(x);
    let mut rng = seeded(seed);
    let mut dir: TtVector<T> = randomized_state(space, 1, 3, &mut rng);
    dir.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    let mut eta = TtTangent::project(&bl, &br, &dir);
    let n = eta.norm();
    assert!(n > 1e-8, "{tag}: degenerate line-search direction");
    eta.scale_mut(T::from_f64(1.0 / n));
    let vv = eta.to_vector();

    let s1 = qntt::dot(x, &vv).re_s();
    let s2 = qntt::dot(&vv, &vv).re_s();
    let (dc1, dc2) = p.linesearch_coeffs(x, &vv);
    let c1 = qntt::xt_h_y(x, h, &vv).re_s() + dc1;
    let c2 = qntt::xt_h_y(&vv, h, &vv).re_s() + dc2;
    let lambda0 = j_functional(h, x, psis, w);

    assert!(
        dc2 > 1e-6,
        "{tag}: Δc₂ = {dc2:.3e} — the search direction barely touches the penalty states, so \
         this arm would not discriminate the coefficients"
    );
    let ov = overlaps_ref(psis, &vv);
    let ox = overlaps_ref(psis, x);
    let dc1_ref: f64 = w * ox
        .iter()
        .zip(ov.iter())
        .map(|(a, b)| a.re_s() * b.re_s() + a.im_s() * b.im_s())
        .sum::<f64>();
    let dc2_ref: f64 = w * ov
        .iter()
        .map(|b| b.re_s() * b.re_s() + b.im_s() * b.im_s())
        .sum::<f64>();
    assert!(
        (dc1 - dc1_ref).abs() <= 1e-12 * dc1.abs().max(1.0),
        "{tag}: Δc₁ = {dc1:.16e} but w·Σ Re(conj(oˣ)·oᵛ) = {dc1_ref:.16e} (a conj swap shows up \
         here first on a complex fixture)"
    );
    assert!(
        (dc2 - dc2_ref).abs() <= 1e-12 * dc2.abs().max(1.0),
        "{tag}: Δc₂ = {dc2:.16e} but w·Σ|oᵛ|² = {dc2_ref:.16e}"
    );

    for &al in &[0.3f64, 0.05] {
        let pred = (lambda0 + 2.0 * c1 * al + c2 * al * al) / (1.0 + 2.0 * s1 * al + s2 * al * al);
        let y = x + &(&vv * T::from_f64(al));
        let exact = j_functional(h, &y, psis, w);
        println!("tms_d4 {tag} α={al}: pred {pred:.16e} vs J(x+αv) {exact:.16e}");
        assert!(
            (pred - exact).abs() <= 1e-12 * exact.abs().max(1.0),
            "{tag} α={al}: the quartic model is not the exact quotient of J along the embedded \
             line ({:.3e})",
            (pred - exact).abs()
        );
    }
}

#[test]
fn tms_d4_linesearch_quotient_is_exact_f64() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x: TtVector<f64> = unit_state(space, 3, 5, 0xD4_0011);
    let psi1: TtVector<f64> = unit_state(space, 2, 4, 0xD4_0012);
    let mut psi2: TtVector<f64> = unit_state(space, 2, 4, 0xD4_0013);
    psi2.scale_mut(2.0);
    quotient_rows(&h, &x, &[psi1, psi2], space, 0xD4_0014, "f64");
}

#[test]
fn tms_d4_linesearch_quotient_is_exact_c64() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
    let h = SparseHamiltonian::new(space, &tc, &vc, None, f64::EPSILON);
    let x: TtVector<C64> = unit_state(space, 3, 5, 0xD4_0015);
    let psi1: TtVector<C64> = unit_state(space, 2, 4, 0xD4_0016);
    let mut psi2: TtVector<C64> = unit_state(space, 2, 4, 0xD4_0017);
    psi2.scale_mut(C64::from_f64(2.0));
    let o1 = qntt::dot(&psi1, &x);
    assert!(
        o1.im_s().abs() > 1e-3,
        "the C64 quotient arm needs a complex overlap to discriminate the conjugation \
         (Im⟨ψ̂₁,x⟩ = {:.3e})",
        o1.im_s()
    );
    quotient_rows(&h, &x, &[psi1, psi2], space, 0xD4_0018, "c64");
}

#[test]
#[should_panic(expected = "norm")]
fn tms_d3_penalty_refuses_a_zero_norm_state() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut zero: TtVector<f64> = unit_state(space, 2, 4, 0xD3_0020);
    zero.scale_mut(0.0);
    let _ = Penalty::new(1.0, std::slice::from_ref(&zero));
}

#[test]
#[should_panic]
fn tms_d3_penalty_refuses_a_negative_weight() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let psi: TtVector<f64> = unit_state(space, 2, 4, 0xD3_0021);
    let _ = Penalty::new(-1.0, std::slice::from_ref(&psi));
}

#[test]
fn tms_e1c_c64_penalized_cg_reaches_the_dense_minimizer() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
    let h = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
    let w = 3.0f64;

    let psi: TtVector<C64> = unit_state(space, 2, 4, 0xE1_C010);
    let rho_p = rho_abelian(&psi);
    assert!(
        rho_p < 0.99,
        "the penalty state is phase-trivial (ρ = {rho_p:.6}); `ĉĉᴴ` and `ĉĉᵀ` would agree on it"
    );

    let mut rng = seeded(0xE1_C011);
    let mut x0: TtVector<C64> = tt_randn(space, &max_ranks(&space), &mut rng);
    normalize_mut(&mut x0);
    let (bl0, _br0) = tangent_bases(&x0);
    let dets = dets_for(d, 1, 1);
    let tdim = tangent_dim(&bl0);
    println!(
        "tms_e1c C64 polish: tangent_dim = {tdim}, determinant count = {}, ρ(ψ) = {rho_p:.6}",
        dets.len()
    );
    assert_eq!(
        tdim,
        dets.len(),
        "the start is not at full representing rank, so 'the dense minimizer' is not the oracle"
    );
    let o0 = qntt::dot(&psi, &x0);
    println!(
        "tms_e1c C64 polish: ⟨ψ̂, x₀⟩ = {:.6e} + {:.6e}i",
        o0.re_s(),
        o0.im_s()
    );
    assert!(
        o0.im_s().abs() > 1e-3,
        "the arm needs a genuinely complex overlap to discriminate `oⱼ` from `conj(oⱼ)` \
         (Im⟨ψ̂,x₀⟩ = {:.3e})",
        o0.im_s()
    );

    let n = dets.len();
    let mut hc = Mat::<C64>::zeros(n, n);
    for (b, (ou, od)) in dets.iter().enumerate() {
        let col = (&h * &tt_state::<C64>(ou, od)).to_dense();
        for (i, (ru, rd)) in dets.iter().enumerate() {
            *hc.get_mut(i, b) = col[basis_index(ru, rd)];
        }
    }
    let mut herm_dev = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            herm_dev = herm_dev.max((*hc.get(i, j) - hc.get(j, i).conj_s()).abs_s());
        }
    }
    assert!(
        herm_dev <= 1e-12,
        "the assembled complex operator is not Hermitian ({herm_dev:.3e})"
    );
    let (mu_bare, _) = dense_self_adjoint_eigen::<C64>(hc.as_ref());
    let hd_real = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let (ev_real, _) = dense_eigen(&hd_real);
    let spec_dev = (0..n)
        .map(|i| (mu_bare[i] - ev_real[i]).abs())
        .fold(0.0f64, f64::max);
    println!("tms_e1c C64 polish: |spec(Hc) − spec(H)|_max = {spec_dev:.3e}");
    assert!(
        spec_dev <= 1e-10,
        "the phase gauge is not a similarity of the real operator ({spec_dev:.3e}); the complex \
         oracle is not the same physics"
    );

    let pd = psi.to_dense();
    let mut chat: Vec<C64> = dets
        .iter()
        .map(|(ou, od)| pd[basis_index(ou, od)])
        .collect();
    let cn: f64 = chat.iter().map(|c| c.abs_s().powi(2)).sum::<f64>().sqrt();
    let full: f64 = pd.iter().map(|c| c.abs_s().powi(2)).sum::<f64>().sqrt();
    assert!(
        cn >= (1.0 - 1e-10) * full,
        "the penalty state leaks outside the determinant sector ({cn:.12e} of {full:.12e})"
    );
    for c in chat.iter_mut() {
        *c *= C64::from_f64(1.0 / cn);
    }
    let mut m = hc.clone();
    for i in 0..n {
        for j in 0..n {
            *m.get_mut(i, j) += C64::from_f64(w) * chat[i] * chat[j].conj_s();
        }
    }
    let (mu, mvec) = dense_self_adjoint_eigen::<C64>(m.as_ref());
    println!(
        "tms_e1c C64 polish: μ₀ = {:.12e}, μ₁ = {:.12e}",
        mu[0], mu[1]
    );
    assert!(
        mu[1] - mu[0] >= 1e-3,
        "the penalized complex spectrum is not resolved at the bottom ({:.3e}); the minimizer \
         claim would be ill-posed",
        mu[1] - mu[0]
    );

    let p = Penalty::new(w, std::slice::from_ref(&psi));
    let opts = RiemannianOpts {
        eps: 1e-13,
        max_iter: 600,
        gradtol: 1e-12,
        preserve_grid: true,
        ..RiemannianOpts::new(&space)
    };
    let (mut j, mut x) = riemannian_cg_penalized(&h, &x0, &p, &opts);
    let mut gate = penalized_exact_gradnorm(&h, &x, &p);
    let mut restarts = 0usize;
    while gate > 1e-10 && restarts < 10 {
        let (j2, x2) = riemannian_cg_penalized(&h, &x, &p, &opts);
        j = j2;
        x = x2;
        gate = penalized_exact_gradnorm(&h, &x, &p);
        restarts += 1;
    }
    println!("tms_e1c C64 polish: restarts used {restarts}/10, gate {gate:.3e}, J {j:.16e}");
    assert!(
        restarts <= 3,
        "the polish used {restarts} restarts of a 10-restart budget — under the 3× headroom a \
         count needs before it is trustworthy; report it rather than raising the budget"
    );
    assert!(
        gate <= 1e-10,
        "the polish did not converge (gate {gate:.3e})"
    );
    assert!(
        (j - mu[0]).abs() <= 1e-10,
        "the complex penalized CG's J {j:.16e} misses the dense penalized minimum {:.16e} by \
         {:.3e}",
        mu[0],
        (j - mu[0]).abs()
    );

    let xd = x.to_dense();
    let mut acc = C64::zero();
    let mut ne = 0.0f64;
    for (a, (ou, od)) in dets.iter().enumerate() {
        acc += mvec.get(a, 0).conj_s() * xd[basis_index(ou, od)];
        ne += mvec.get(a, 0).abs_s().powi(2);
    }
    let nx: f64 = xd.iter().map(|c| c.abs_s().powi(2)).sum::<f64>().sqrt();
    let proj = acc.abs_s() / (nx * ne.sqrt()).max(1e-300);
    let rho_x = rho_abelian(&x);
    println!("tms_e1c C64 polish: projection {proj:.12}, ρ(x) = {rho_x:.6}");
    assert!(
        proj >= 1.0 - 1e-8,
        "the polished complex state is not the dense penalized minimizer (projection {proj:.12})"
    );
    assert!(
        rho_x < 0.99,
        "VACUOUS — the polished state shares one global phase (ρ = {rho_x:.6})"
    );
    assert!(
        (x.norm() - 1.0).abs() <= 1e-12,
        "the returned state is not unit ({:.3e})",
        x.norm()
    );

    let lam = qntt::rayleigh_quotient(&h, &x).re_s();
    let o = qntt::dot(&psi, &x);
    let osq = o.abs_s() * o.abs_s();
    assert!(
        (lam + w * osq - j).abs() <= 1e-12 * j.abs().max(1.0),
        "J is not λ + w|⟨ψ̂,x⟩|² at the returned root (λ = {lam:.16e}, |o|² = {osq:.3e}, \
         J = {j:.16e})"
    );
}

#[test]
fn tms_e1d_penalized_linesearch_scores_by_j() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = generic_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let psi: TtVector<f64> = unit_state(space, 2, 4, 0xE1_D001);
    let w = 3.0f64;
    let p = Penalty::new(w, std::slice::from_ref(&psi));
    let psis = std::slice::from_ref(&psi);

    let mut x: TtVector<f64> = unit_state(space, 2, 2, 0xE1_D002);
    let dets = dets_for(d, 1, 1);
    let (bl0, _br0) = tangent_bases(&x);
    let tdim = tangent_dim(&bl0);
    println!(
        "tms_e1d: tangent_dim {tdim} of {} determinants — the cap binds",
        dets.len()
    );
    assert!(
        tdim < dets.len(),
        "the cap does not bind (tangent_dim {tdim} = the whole sector); at full rank the quartic \
         model is the exact quotient and no backtracking branch is ever reached"
    );

    let opts = RiemannianOpts {
        eps: 1e-13,
        max_iter: 1,
        gradtol: 1e-12,
        preserve_grid: true,
        ..RiemannianOpts::new(&space)
    };

    let n_calls = 160usize;
    let mut ran = 0usize;
    let mut flat = 0usize;
    let mut j_first = f64::NAN;
    for it in 0..n_calls {
        assert!(
            (x.norm() - 1.0).abs() <= 1e-12,
            "call {it}: the iterate is not unit ({:.3e})",
            x.norm()
        );
        let o_in = qntt::dot(&psi, &x).abs_s();
        let pen_in = w * o_in * o_in;
        assert!(
            pen_in >= 1e-9,
            "call {it}: the penalty term has fallen to {pen_in:.3e}; `J = λ` there and the arm no \
             longer tests the penalized line search"
        );
        let j_in = j_functional(&h, &x, psis, w);
        if it == 0 {
            j_first = j_in;
        }
        let efloor = 32.0 * f64::EPSILON * (1.0 + j_in.abs());

        let mut x_entry = x.clone();
        normalize_mut(&mut x_entry);

        let (j, x2) = riemannian_cg_penalized(&h, &x, &p, &opts);

        let j_out = j_functional(&h, &x2, psis, w);
        assert!(
            (j - j_out).abs() <= 1e-12 * j_out.abs().max(1.0),
            "call {it}: the penalized CG returned {j:.17e} but `J` at the returned point is \
             {j_out:.17e} — the accepted trial was not scored by `J`"
        );
        assert!(
            j <= j_in + efloor,
            "call {it}: `J` ROSE from {j_in:.17e} to {j:.17e} (Δ {:.3e}) past the roundoff floor \
             {efloor:.3e}",
            j - j_in
        );
        if !tt_coeffs_identical(&x2, &x_entry) {
            ran += 1;
            if (j - j_in).abs() <= efloor {
                flat += 1;
            }
        }
        x = x2;
    }
    let gate = penalized_exact_gradnorm(&h, &x, &p);
    let j_last = j_functional(&h, &x, psis, w);
    println!(
        "tms_e1d: {n_calls} CG calls, {ran} of them ran a line search, J {j_first:.16e} → \
         {j_last:.16e}, exact gate {gate:.3e}, {flat} moved call(s) inside the roundoff floor"
    );
    assert!(
        j_last <= j_first + 1e-12,
        "the chain of line searches did not decrease `J` overall"
    );
    assert!(
        gate <= 1e-9,
        "the chain did not approach a critical point of the penalized functional (gate \
         {gate:.3e}); the late calls — the ones that exercise the roundoff-floor branches — never \
         happened"
    );
    assert!(
        ran >= 20,
        "only {ran} of {n_calls} calls ran a line search at all; the arm is (nearly) vacuous — \
         every claim above is about calls that MOVED the iterate"
    );
    assert!(
        flat >= 5,
        "only {flat} of the {ran} moved calls landed inside the roundoff floor, so the branches \
         that key on `efloor` are (nearly) unreached; lengthen the chain or tighten the cap"
    );
}
