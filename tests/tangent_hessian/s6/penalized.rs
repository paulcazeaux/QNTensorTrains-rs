// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s6::util::*;
use qntt::round::RoundMode;
use qntt::solvers::penalty::Penalty;
use qntt::su2::core::{Family, Su2Core};
use qntt::su2::expand::expand;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{
    CgVariant, RiemannianOpts, normalize_mut, penalized_exact_gradnorm, riemannian_cg,
    riemannian_cg_penalized, riemannian_gradient, riemannian_gradient_penalized, tangent_bases,
};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_randn};
use qntt::su2::weingarten::tangent_dim;
use qntt::{CoeffTensor, Scalar};
use qntt_testkit::dense::{basis_index, dense_eigen, dense_hamiltonian_matrix, dets_for};
use qntt_testkit::gad_mol::singlet_indices;

fn generic_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
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
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
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

fn assert_f64_bits(got: f64, want: f64, what: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{what}: {got:.17e} vs {want:.17e} (bit-identity required — an inactive penalty must take \
         the bare path, not a numerically-equal one)"
    );
}

fn coeffs_bits_eq_su2<T: Scalar>(a: &Su2TtVector<T>, b: &Su2TtVector<T>, what: &str) {
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
            let (x, y) = (ca.coeff(i), cb.coeff(i));
            assert!(
                x.re_s().to_bits() == y.re_s().to_bits()
                    && x.im_s().to_bits() == y.im_s().to_bits(),
                "{what}: core {k} coefficient {i}"
            );
        }
    }
}

fn su2_coeffs_identical<T: Scalar>(a: &Su2TtVector<T>, b: &Su2TtVector<T>) -> bool {
    if a.ranks != b.ranks {
        return false;
    }
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        if ca.num_coeffs() != cb.num_coeffs() {
            return false;
        }
        for i in 0..ca.num_coeffs() {
            let (x, y) = (ca.coeff(i), cb.coeff(i));
            if x.re_s().to_bits() != y.re_s().to_bits() || x.im_s().to_bits() != y.im_s().to_bits()
            {
                return false;
            }
        }
    }
    true
}

fn tangent_bits_eq<T: Scalar>(a: &Su2TtTangent<T>, b: &Su2TtTangent<T>, what: &str) {
    assert_eq!(a.num_coeffs(), b.num_coeffs(), "{what}: coefficient count");
    for i in 0..a.num_coeffs() {
        let (x, y) = (a.coeff(i), b.coeff(i));
        assert!(
            x.re_s().to_bits() == y.re_s().to_bits() && x.im_s().to_bits() == y.im_s().to_bits(),
            "{what}: coefficient {i}"
        );
    }
}

fn unit_su2(space: Su2Space, lo: usize, hi: usize, seed: u64) -> Su2TtVector<f64> {
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
    x.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    normalize_mut(&mut x);
    x
}

fn expanded(y: &Su2TtVector<f64>, two_sz: i64, dets: &[(Vec<bool>, Vec<bool>)]) -> Vec<f64> {
    let e = expand(y, two_sz).to_dense();
    let full: f64 = e.iter().map(|c| c * c).sum::<f64>().sqrt();
    let c: Vec<f64> = dets.iter().map(|(ou, od)| e[basis_index(ou, od)]).collect();
    let nrm = c.iter().map(|val| val * val).sum::<f64>().sqrt();
    assert!(
        nrm >= (1.0 - 1e-10) * full,
        "the expanded state leaks outside the determinant sector ({nrm:.12e} of {full:.12e})"
    );
    c
}

fn j_dense(hd: &[Vec<f64>], chats: &[Vec<f64>], c: &[f64], w: f64) -> f64 {
    let n = c.len();
    let mut num = 0.0f64;
    for i in 0..n {
        let mut acc = 0.0f64;
        for j in 0..n {
            acc += hd[i][j] * c[j];
        }
        num += c[i] * acc;
    }
    for chat in chats {
        let o: f64 = chat.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        num += w * o * o;
    }
    let den: f64 = c.iter().map(|a| a * a).sum();
    num / den
}

#[allow(clippy::too_many_arguments)]
fn richardson_dj(
    hd: &[Vec<f64>],
    chats: &[Vec<f64>],
    x: &Su2TtVector<f64>,
    v: &Su2TtVector<f64>,
    two_sz: i64,
    dets: &[(Vec<bool>, Vec<bool>)],
    w: f64,
    step: f64,
) -> f64 {
    let d = |t: f64| {
        let plus = x + &(v * t);
        let minus = x + &(v * (-t));
        (j_dense(hd, chats, &expanded(&plus, two_sz, dets), w)
            - j_dense(hd, chats, &expanded(&minus, two_sz, dets), w))
            / (2.0 * t)
    };
    (4.0 * d(step / 2.0) - d(step)) / 3.0
}

fn flat_component_dot(a: &Su2Core<f64>, b: &Su2Core<f64>) -> f64 {
    let mut acc = 0.0f64;
    for f in Family::ALL {
        for l in a.transitions(f).collect::<Vec<_>>() {
            let (Some(ba), Some(bb)) = (a.block(f, l), b.block(f, l)) else {
                continue;
            };
            for j in 0..ba.ncols() {
                for i in 0..ba.nrows() {
                    acc += *ba.get(i, j) * *bb.get(i, j);
                }
            }
        }
    }
    acc
}

fn flat_tangent_dot(a: &Su2TtTangent<f64>, b: &Su2TtTangent<f64>, d: usize) -> f64 {
    (0..d)
        .map(|k| flat_component_dot(a.component(k), b.component(k)))
        .sum()
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

fn projection_on_dense(c: &[f64], evec: &[f64]) -> f64 {
    let num: f64 = c.iter().zip(evec.iter()).map(|(a, b)| a * b).sum();
    let nc: f64 = c.iter().map(|a| a * a).sum::<f64>().sqrt();
    let ne: f64 = evec.iter().map(|a| a * a).sum::<f64>().sqrt();
    num.abs() / (nc * ne).max(1e-300)
}

#[test]
fn tms_d2bsu2_inactive_penalty_is_the_bare_path() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let (t, v) = generic_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let x0 = unit_su2(space, 3, 5, 0xD2B_1001);
    let psi = unit_su2(space, 2, 4, 0xD2B_1002);

    let opts = RiemannianOpts {
        eps: 1e-6,
        max_iter: 12,
        gradtol: 1e-8,
        variant: CgVariant::Pr,
        ..RiemannianOpts::new(&space)
    };
    let empty: Penalty<Su2TtVector<f64>> = Penalty::new(0.0, &[]);
    let zero_w: Penalty<Su2TtVector<f64>> = Penalty::new(0.0, std::slice::from_ref(&psi));
    let no_states: Penalty<Su2TtVector<f64>> = Penalty::new(3.0, &[]);
    for p in [&empty, &zero_w, &no_states] {
        assert!(
            !p.is_active(),
            "an inactive penalty (w = {}, {} states) reports itself active",
            p.w,
            p.states.len()
        );
    }

    let (l_bare, x_bare) = riemannian_cg(&h, &x0, &opts);
    for (tag, p) in [
        ("empty", &empty),
        ("w=0", &zero_w),
        ("no states", &no_states),
    ] {
        let (j, xj) = riemannian_cg_penalized(&h, &x0, p, &opts);
        assert_f64_bits(
            j,
            l_bare,
            &format!("riemannian_cg_penalized ({tag}) J vs λ"),
        );
        coeffs_bits_eq_su2(&xj, &x_bare, &format!("riemannian_cg_penalized ({tag}) x"));
    }

    let fr = RiemannianOpts {
        variant: CgVariant::Fr,
        ..opts.clone()
    };
    let (l_fr, x_fr) = riemannian_cg(&h, &x0, &fr);
    let (j_fr, xj_fr) = riemannian_cg_penalized(&h, &x0, &empty, &fr);
    assert_f64_bits(j_fr, l_fr, "riemannian_cg_penalized Fr J vs λ");
    coeffs_bits_eq_su2(&xj_fr, &x_fr, "riemannian_cg_penalized Fr x");

    let (bl, br) = tangent_bases(&x_bare);
    let (g_bare, l_g) = riemannian_gradient(&h, &x_bare, &bl, &br, 1e-12, 1234);
    let (g_pen, j_g) = riemannian_gradient_penalized(&h, &x_bare, &bl, &br, &empty, 1e-12, 1234);
    assert_f64_bits(j_g, l_g, "riemannian_gradient_penalized (empty) J vs λ");
    tangent_bits_eq(&g_pen, &g_bare, "riemannian_gradient_penalized (empty) g");

    let gate_pen = penalized_exact_gradnorm(&h, &x_bare, &empty);
    let gate_bare = qntt::su2::hess::weingarten_operator(&h, &x_bare)
        .gradient()
        .norm();
    assert_f64_bits(
        gate_pen,
        gate_bare,
        "penalized_exact_gradnorm (empty) vs the response pipeline's gate",
    );
}

#[allow(clippy::too_many_arguments)]
fn d3su2_rows(space: Su2Space, two_sz: i64, nup: usize, ndn: usize, seed0: u64, tag: &str) {
    let d = space.d;
    let (t, v) = generic_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let dets = dets_for(d, nup, ndn);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);

    let x = unit_su2(space, 3, 5, seed0);
    let psi1 = unit_su2(space, 2, 4, seed0 + 1);
    let mut psi2 = unit_su2(space, 2, 4, seed0 + 2);
    psi2.scale_mut(2.0);
    assert!(
        (psi2.norm() - 2.0).abs() <= 1e-12,
        "{tag}: ψ₂ must have norm 2 so the `invn` fold is exercised ({:.6})",
        psi2.norm()
    );
    let psis = [psi1.clone(), psi2.clone()];

    let (n_theta, n_cw) = count_theta_cw(&x);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "{tag}: the base point does not exercise the weighted paths (n_theta = {n_theta}, \
         n_cw = {n_cw}); the SU(2) arms would be vacuous"
    );

    let chats: Vec<Vec<f64>> = psis
        .iter()
        .map(|p| {
            let mut c = expanded(p, two_sz, &dets);
            let n: f64 = c.iter().map(|a| a * a).sum::<f64>().sqrt();
            for a in c.iter_mut() {
                *a /= n;
            }
            c
        })
        .collect();
    let cx = expanded(&x, two_sz, &dets);

    let (bl, br) = tangent_bases(&x);
    let lambda = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x).re_s();

    let mut rng = seeded(seed0 + 10);
    let mut dirs: Vec<Su2TtTangent<f64>> = Vec::new();
    for _ in 0..12 {
        let mut dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        dir.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
        let mut xi = Su2TtTangent::project(&bl, &br, &dir);
        let n = xi.norm();
        if n <= 1e-8 {
            continue;
        }
        xi.scale_mut(1.0 / n);
        dirs.push(xi);
        if dirs.len() >= 6 {
            break;
        }
    }
    assert!(
        dirs.len() >= 3,
        "{tag}: fewer than 3 usable tangent directions were drawn"
    );

    let mut flat_separations: Vec<f64> = Vec::new();
    for &w in &[1e-2f64, 1.0, 1e3] {
        let p = Penalty::new(w, &psis);
        assert!(p.is_active(), "{tag}: w = {w} with 2 states must be active");
        let (g, j) = riemannian_gradient_penalized(&h, &x, &bl, &br, &p, 1e-12, 1234);

        let os: Vec<f64> = psis
            .iter()
            .map(|q| qntt::su2::ops::dot(q, &x).re_s() / q.norm())
            .collect();
        let pen: f64 = w * os.iter().map(|o| o * o).sum::<f64>();
        assert!(
            (j - lambda - pen).abs() <= 1e-11 * j.abs().max(1.0),
            "{tag} w={w}: J − λ = {:.16e} but w·Σ|oⱼ|² = {pen:.16e}",
            j - lambda
        );
        let j_ab = j_dense(&hd, &chats, &cx, w);
        assert!(
            (j - j_ab).abs() <= 1e-11 * j.abs().max(1.0),
            "{tag} w={w}: the driver's J {j:.16e} differs from the METRIC-FREE expanded \
             functional {j_ab:.16e} by {:.3e} — either `expand` is not an isometry for this \
             space or the penalty is measured in the wrong metric",
            (j - j_ab).abs()
        );
        assert!(
            j > lambda && pen >= 1e-6 * w,
            "{tag} w={w}: the base point is (near-)orthogonal to the penalty states \
             (pen = {pen:.3e}); the arm would not discriminate the penalty term"
        );

        let mut checked = 0usize;
        for xi in &dirs {
            let vv = xi.to_vector();
            let r = richardson_dj(&hd, &chats, &x, &vv, two_sz, &dets, w, 1e-3);
            if r.abs() < 1e-2 {
                continue;
            }
            let analytic = 2.0 * g.dot(xi);
            let rel = (r - analytic).abs() / r.abs();
            println!(
                "tms_d3su2 {tag} w={w}: FD {r:.12e} vs 2·Re⟨g,ξ⟩ {analytic:.12e} (rel {rel:.3e})"
            );
            assert!(
                rel <= 1e-8,
                "{tag} w={w}: the penalized SU(2) gradient does not differentiate J (rel \
                 {rel:.3e})"
            );
            let flat = 2.0 * flat_tangent_dot(&g, xi, d);
            let sep = (r - flat).abs() / r.abs();
            flat_separations.push(sep);
            checked += 1;
            if checked >= 3 {
                break;
            }
        }
        assert!(
            checked >= 3,
            "{tag} w={w}: only {checked} directions produced a resolvable FD signal"
        );

        let xir = Su2TtTangent::project(&bl, &br, &x);
        assert!(
            (xir.norm() - 1.0).abs() <= 1e-12,
            "{tag}: P(x) is not unit ({:.3e}) — x is not on the manifold",
            xir.norm()
        );
        let rr = richardson_dj(&hd, &chats, &x, &xir.to_vector(), two_sz, &dets, w, 1e-3);
        assert!(
            rr.abs() <= 1e-7,
            "{tag} w={w}: J is not radially flat (FD along P(x) = {rr:.3e})"
        );
        let radial = 2.0 * g.dot(&xir);
        assert!(
            radial.abs() <= 1e-9,
            "{tag} w={w}: the J-shifted gradient has a spurious radial component {radial:.3e} \
             (a λ-shift would leave 2·pen = {:.3e} there)",
            2.0 * pen
        );

        let hx = qntt::solvers::TtHamiltonian::matvec(&h, &x);
        let mut glam = Su2TtTangent::project(&bl, &br, &hx);
        glam.axpy(-lambda, &Su2TtTangent::project(&bl, &br, &x));
        for (o, q) in os.iter().zip(psis.iter()) {
            let invn = 1.0 / q.norm();
            glam.axpy(w * invn * o, &Su2TtTangent::project(&bl, &br, q));
        }
        assert!(
            2.0 * pen >= 1e-3,
            "{tag} w={w}: 2·pen = {:.3e} is below the anti-oracle's resolution",
            2.0 * pen
        );
        let radial_lambda = 2.0 * glam.dot(&xir);
        assert!(
            (radial_lambda - 2.0 * pen).abs() <= 1e-9,
            "{tag} w={w}: the λ-shifted gradient's radial component is {radial_lambda:.6e}, not \
             2·pen = {:.6e}",
            2.0 * pen
        );

        let mut psi2_big = psi2.clone();
        psi2_big.scale_mut(3.0);
        let p2 = Penalty::new(w, &[psi1.clone(), psi2_big]);
        let (g2, j2) = riemannian_gradient_penalized(&h, &x, &bl, &br, &p2, 1e-12, 1234);
        assert!(
            (j2 - j).abs() <= 1e-12 * j.abs().max(1.0),
            "{tag} w={w}: rescaling ψ₂ by 3 moved J by {:.3e}; `invn` is not folded",
            (j2 - j).abs()
        );
        let mut dev = 0.0f64;
        for i in 0..g.num_coeffs() {
            dev = dev.max((g2.coeff(i) - g.coeff(i)).abs());
        }
        assert!(
            dev <= 1e-12 * g.norm().max(1.0),
            "{tag} w={w}: rescaling ψ₂ by 3 moved the gradient by {dev:.3e}"
        );
    }

    let worst = flat_separations.iter().fold(0.0f64, |acc, &s| acc.max(s));
    println!("tms_d3su2 {tag}: centerweight-dropped separations {flat_separations:?}");
    assert!(
        worst > 0.10,
        "{tag}: a PLAIN Frobenius tangent dot reproduces the FD to within {:.1} % on every \
         direction — this suite is then not testing the SU(2) metric at all",
        100.0 * worst
    );
}

#[test]
fn tms_d3su2_penalized_gradient_matches_richardson_fd_singlet() {
    d3su2_rows(Su2Space::new(6, 2, 0), 0, 1, 1, 0xD3_1001, "(6,2,0)");
}

#[test]
fn tms_d3su2_penalized_gradient_matches_richardson_fd_doublet() {
    d3su2_rows(Su2Space::new(6, 3, 1), 1, 2, 1, 0xD3_1002, "(6,3,1)");
}

#[test]
fn tms_d4su2_penalized_cg_reaches_the_dense_minimizer() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let (t, v) = generic_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let psi = unit_su2(space, 2, 4, 0xD4_1001);
    let w = 3.0f64;

    let dets11 = dets_for(d, 1, 1);
    let dets20 = dets_for(d, 2, 0);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets11);
    let hd20 = dense_hamiltonian_matrix(d, &t, &v, &dets20);
    let (ev20, _) = dense_eigen(&hd20);

    let mut chat = expanded(&psi, 0, &dets11);
    let cn: f64 = chat.iter().map(|a| a * a).sum::<f64>().sqrt();
    for a in chat.iter_mut() {
        *a /= cn;
    }
    let m = penalized_dense(&hd, &[chat.clone()], w);
    let (ev_pen, evec_pen) = dense_eigen(&m);

    let (ev11, _) = dense_eigen(&hd);
    let s_idx_bare = singlet_indices(&ev11, &ev20);
    let s_idx = singlet_indices(&ev_pen, &ev20);
    assert_eq!(
        s_idx.len(),
        s_idx_bare.len(),
        "the penalty changed the singlet/triplet split of the dense spectrum ({} vs {}); the \
         penalty state is not a singlet, or the labelling collapsed on an accidental degeneracy",
        s_idx.len(),
        s_idx_bare.len()
    );
    let mu0 = ev_pen[s_idx[0]];
    let mu1 = ev_pen[s_idx[1]];
    println!("tms_d4su2: penalized singlet levels μ₀ = {mu0:.12e}, μ₁ = {mu1:.12e}");
    assert!(
        mu1 - mu0 >= 1e-3,
        "the penalized singlet spectrum is not resolved at the bottom ({:.3e})",
        mu1 - mu0
    );

    let mut rng = seeded(0xD4_1002);
    let mut x0: Su2TtVector<f64> = tt_randn(space, &qntt::su2::qn::max_ranks(&space), &mut rng);
    normalize_mut(&mut x0);
    let (bl0, _br0) = tangent_bases(&x0);
    let tdim = tangent_dim(&bl0);
    println!(
        "tms_d4su2: tangent_dim = {tdim}, singlet count = {}",
        s_idx.len()
    );
    assert_eq!(
        tdim,
        s_idx.len(),
        "the start is not at full reduced-multiplet rank, so the manifold is not the whole S = 0 \
         sector and 'the dense minimizer' is not the right oracle"
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
    println!("tms_d4su2: restarts used {restarts}/10, gate {gate:.3e}, J {j:.16e}");
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
        (j - mu0).abs() <= 1e-10,
        "the penalized SU(2) CG's J {j:.16e} misses the dense penalized singlet minimum \
         {mu0:.16e} by {:.3e}",
        (j - mu0).abs()
    );
    let cx = expanded(&x, 0, &dets11);
    let proj = projection_on_dense(&cx, &evec_pen[s_idx[0]]);
    assert!(
        proj >= 1.0 - 1e-8,
        "the polished SU(2) state is not the dense penalized minimizer (projection {proj:.12})"
    );
    assert!(
        (x.norm() - 1.0).abs() <= 1e-12,
        "the returned state is not unit ({:.3e})",
        x.norm()
    );

    let lam = qntt::su2::hamiltonian::rayleigh_quotient(&h, &x).re_s();
    let o = qntt::su2::ops::dot(&psi, &x).re_s();
    assert!(
        (lam + w * o * o - j).abs() <= 1e-11 * j.abs().max(1.0),
        "J is not λ + w|⟨ψ̂,x⟩|² at the returned root (λ = {lam:.16e}, o² = {:.3e}, J = {j:.16e})",
        o * o
    );
}

#[test]
fn tms_e1dsu2_penalized_linesearch_scores_by_j() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let (t, v) = generic_tv(d);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let w = 3.0f64;
    let psi = unit_su2(space, 2, 4, 0xE1_D101);

    let dets11 = dets_for(d, 1, 1);
    let dets20 = dets_for(d, 2, 0);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets11);
    let hd20 = dense_hamiltonian_matrix(d, &t, &v, &dets20);
    let (ev20, _) = dense_eigen(&hd20);
    let (ev11, _) = dense_eigen(&hd);
    let n_singlets = singlet_indices(&ev11, &ev20).len();

    let mut chat = expanded(&psi, 0, &dets11);
    let cn: f64 = chat.iter().map(|a| a * a).sum::<f64>().sqrt();
    for a in chat.iter_mut() {
        *a /= cn;
    }
    let chats = [chat];

    let mut x = unit_su2(space, 1, 2, 0xE1_D102);
    let (bl0, _br0) = tangent_bases(&x);
    let tdim = tangent_dim(&bl0);
    let (n_theta, n_cw) = count_theta_cw(&x);
    println!(
        "tms_e1dsu2: tangent_dim {tdim} of {n_singlets} singlets — the cap binds; n_theta \
         {n_theta}, n_cw {n_cw}"
    );
    assert!(
        tdim < n_singlets,
        "the cap does not bind (tangent_dim {tdim} = the whole S = 0 sector); at full reduced rank \
         the quartic model is the exact quotient and no backtracking branch is ever reached"
    );

    let p = Penalty::new(w, std::slice::from_ref(&psi));
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
        let o_in = qntt::su2::ops::dot(&psi, &x).abs_s();
        let pen_in = w * o_in * o_in;
        assert!(
            pen_in >= 1e-9,
            "call {it}: the penalty term has fallen to {pen_in:.3e}; `J = λ` there and the arm no \
             longer tests the penalized line search"
        );
        let j_in = j_dense(&hd, &chats, &expanded(&x, 0, &dets11), w);
        if it == 0 {
            j_first = j_in;
        }
        let efloor = 32.0 * f64::EPSILON * (1.0 + j_in.abs());

        let mut x_entry = x.clone();
        normalize_mut(&mut x_entry);

        let (j, x2) = riemannian_cg_penalized(&h, &x, &p, &opts);

        let j_out = j_dense(&hd, &chats, &expanded(&x2, 0, &dets11), w);
        assert!(
            (j - j_out).abs() <= 1e-11 * j_out.abs().max(1.0),
            "call {it}: the penalized SU(2) CG returned {j:.17e} but `J_ab` at the returned point \
             is {j_out:.17e} — the accepted trial was not scored by `J`"
        );
        assert!(
            j <= j_in + efloor,
            "call {it}: `J` ROSE from {j_in:.17e} to {j:.17e} (Δ {:.3e}) past the roundoff floor \
             {efloor:.3e}",
            j - j_in
        );
        if !su2_coeffs_identical(&x2, &x_entry) {
            ran += 1;
            if (j - j_in).abs() <= efloor {
                flat += 1;
            }
        }
        x = x2;
    }
    let gate = penalized_exact_gradnorm(&h, &x, &p);
    let j_last = j_dense(&hd, &chats, &expanded(&x, 0, &dets11), w);
    println!(
        "tms_e1dsu2: {n_calls} CG calls, {ran} of them ran a line search, J {j_first:.16e} → \
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
