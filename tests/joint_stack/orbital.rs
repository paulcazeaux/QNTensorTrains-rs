// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::panic::AssertUnwindSafe;
use std::path::Path;

use qntt::integrals::{IntegralsOptions, load_fixture_dir};
use qntt::rdm::transition_1rdm_bruteforce;
use qntt::rotation::{
    e_pq, expm_antisym, expm_kappa_blocked, freeze_core, g_pq, kappa_apply, kappa_operator,
    orbital_gradient, rotate_integrals,
};
use qntt::{
    CoeffTensor, Integrals, Orbital, PointGroup, QnSpace, SparseHamiltonian, Spin, TtVector, dot,
    rayleigh_quotient, tt_state,
};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::*;

const A5_SEED: u64 = 0xA5A5_5A5A;

fn energy_at(fx: &A5Fixture, hh: f64) -> f64 {
    let u = expm_antisym(&scale_flat(&fx.k.data, -hh), fx.d);
    let rotated = rotate_integrals(&fx.ints, &u);
    let h = SparseHamiltonian::new(fx.space, &rotated.t, &rotated.v, None, f64::EPSILON);
    rayleigh_quotient(&h, &fx.x)
}

fn fd4(f: impl Fn(f64) -> f64, h: f64) -> f64 {
    (f(-2.0 * h) - 8.0 * f(-h) + 8.0 * f(h) - f(2.0 * h)) / (12.0 * h)
}

fn directional_derivative(d: usize, g: &CoeffTensor<f64>, k: &CoeffTensor<f64>) -> f64 {
    let mut dir = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            dir += 0.5 * g.get(&[p, q]) * k.get(&[p, q]);
        }
    }
    dir
}

fn naive_rotate_tv(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    u: &[f64],
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut tp = CoeffTensor::<f64>::zeros(d, 2);
    for a in 0..d {
        for b in 0..d {
            let mut s = 0.0f64;
            for i in 0..d {
                for j in 0..d {
                    s += u[i * d + a] * u[j * d + b] * t.get(&[i, j]);
                }
            }
            tp.set(&[a, b], s);
        }
    }
    let mut vp = CoeffTensor::<f64>::zeros(d, 4);
    for a in 0..d {
        for b in 0..d {
            for c in 0..d {
                for e in 0..d {
                    let mut s = 0.0f64;
                    for i in 0..d {
                        for j in 0..d {
                            for k in 0..d {
                                for l in 0..d {
                                    s += u[i * d + a]
                                        * u[j * d + b]
                                        * u[k * d + c]
                                        * u[l * d + e]
                                        * v.get(&[i, j, k, l]);
                                }
                            }
                        }
                    }
                    vp.set(&[a, b, c, e], s);
                }
            }
        }
    }
    (tp, vp)
}

#[test]
fn tja_a1_e_pq_dense_oracle() {
    let d = 4;
    let spaces = [
        QnSpace::new(d, 2, 2),
        QnSpace::new(d, 3, 1),
        QnSpace::new(d, 4, 0),
        QnSpace::new(d, 0, 4),
    ];
    let pairs: [(usize, usize); 9] = [
        (0, 0),
        (2, 2),
        (3, 3),
        (0, 1),
        (1, 0),
        (0, 3),
        (3, 0),
        (1, 2),
        (2, 1),
    ];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA1A1);
    for &space in &spaces {
        let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let x0d = x0.to_dense();
        for &(p, q) in &pairs {
            let t_pq: f64 = rng.sample::<f64, _>(StandardNormal);
            let y = e_pq(&x0, p, q, t_pq);
            let yd = y.to_dense();
            let up = dense_apply_string(
                d,
                &[
                    (Orbital::new(p, Spin::Up), true),
                    (Orbital::new(q, Spin::Up), false),
                ],
                &x0d,
            );
            let dn = dense_apply_string(
                d,
                &[
                    (Orbital::new(p, Spin::Dn), true),
                    (Orbital::new(q, Spin::Dn), false),
                ],
                &x0d,
            );
            let reference: Vec<f64> = up.iter().zip(&dn).map(|(a, b)| t_pq * (a + b)).collect();
            assert_dense_close(
                &yd,
                &reference,
                1e-9,
                &format!(
                    "e_pq dense oracle: space=({},{}) p={p} q={q}",
                    space.nup, space.ndn
                ),
            );
        }
    }
}

#[test]
fn tja_a1_e_pq_diagonal_is_occupation_number() {
    let d = 6;
    let s = det_tt(d, &[0, 2, 4], &[0, 3]);
    for &(p, want) in &[(0usize, 2.0f64), (2, 1.0), (d - 1, 0.0)] {
        let y = e_pq(&s, p, p, 1.0);
        let val = dot(&s, &y);
        assert_close(
            val,
            want,
            1e-13,
            "e_pq diagonal must equal orbital occupation number",
        );
    }
}

#[test]
fn tja_a1_g_pq_dense_oracle_and_antisymmetry() {
    let d = 5;
    let space = QnSpace::new(d, 3, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA1A2);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let x0d = x0.to_dense();
    let pairs: [(usize, usize); 3] = [(1, 3), (2, 4), (0, 4)];
    for &(p, q) in &pairs {
        let y = g_pq(&x0, p, q);
        let yd = y.to_dense();
        let up_pq = dense_apply_string(
            d,
            &[
                (Orbital::new(p, Spin::Up), true),
                (Orbital::new(q, Spin::Up), false),
            ],
            &x0d,
        );
        let dn_pq = dense_apply_string(
            d,
            &[
                (Orbital::new(p, Spin::Dn), true),
                (Orbital::new(q, Spin::Dn), false),
            ],
            &x0d,
        );
        let up_qp = dense_apply_string(
            d,
            &[
                (Orbital::new(q, Spin::Up), true),
                (Orbital::new(p, Spin::Up), false),
            ],
            &x0d,
        );
        let dn_qp = dense_apply_string(
            d,
            &[
                (Orbital::new(q, Spin::Dn), true),
                (Orbital::new(p, Spin::Dn), false),
            ],
            &x0d,
        );
        let reference: Vec<f64> = (0..yd.len())
            .map(|i| (up_pq[i] + dn_pq[i]) - (up_qp[i] + dn_qp[i]))
            .collect();
        assert_dense_close(
            &yd,
            &reference,
            1e-9,
            &format!("g_pq dense oracle p={p} q={q}"),
        );

        let y_rev = g_pq(&x0, q, p);
        let sum = &y + &y_rev;
        let scale = y.norm().max(1e-3);
        assert!(
            sum.norm() <= 1e-12 * scale,
            "g_pq exact antisymmetry violated at p={p} q={q}: ‖g_pq+g_qp‖={} scale={scale}",
            sum.norm()
        );
    }
}

#[test]
#[should_panic(expected = "G_pp")]
fn tja_a1_g_pq_panics_on_equal_indices() {
    let x0 = det_tt(4, &[0, 1], &[0, 1]);
    let _ = g_pq(&x0, 1, 1);
}

#[test]
fn tja_a2_kappa_apply_matches_generator_sum() {
    let d = 5;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2A1);
    let kappa = random_antisym_coeff(d, &mut rng);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    let y = kappa_apply(&kappa, &x0);

    let mut reference: Option<TtVector<f64>> = None;
    for p in 1..d {
        for q in 0..p {
            let kpq = kappa.get(&[p, q]);
            if kpq == 0.0 {
                continue;
            }
            let term = &g_pq(&x0, p, q) * kpq;
            reference = Some(match reference {
                None => term,
                Some(acc) => &acc + &term,
            });
        }
    }
    let reference = reference.expect("random antisymmetric kappa draw must be nonzero");
    let mut diff = &y - &reference;
    diff.round_mut(1e-14, qntt::Pooled::NONE, qntt::round::RoundMode::Rel);

    assert!(
        diff.norm() <= 1e-11 * reference.norm().max(1e-8),
        "kappa_apply vs Σ κ[p,q]·g_pq(x,p,q): ‖diff‖={} ‖ref‖={}",
        diff.norm(),
        reference.norm()
    );

    let op = kappa_operator(space, &kappa);
    let y2 = &op * &x0;
    let diff2 = &y - &y2;
    assert!(
        diff2.norm() <= 1e-12 * y.norm().max(1e-8),
        "kappa_apply and kappa_operator(...).apply(x) must agree (kappa_apply is a pure alias)"
    );
}

#[test]
fn tja_a2_half_matrix_rejected_at_any_magnitude() {
    let d = 5;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2A3);
    let kfull = random_antisym_coeff(d, &mut rng);
    let khalf = strict_lower_only(&kfull);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    for &s in &[1.0f64, 1e-15, 1e-30] {
        let scaled = scale_coeff(&khalf, s);
        let r1 = std::panic::catch_unwind(AssertUnwindSafe(|| kappa_operator(space, &scaled)));
        assert!(
            r1.is_err(),
            "kappa_operator must reject the p>q half matrix at scale {s:e}"
        );

        let scaled2 = scale_coeff(&khalf, s);
        let x0c = x0.clone();
        let r2 = std::panic::catch_unwind(AssertUnwindSafe(move || kappa_apply(&scaled2, &x0c)));
        assert!(
            r2.is_err(),
            "kappa_apply must reject the p>q half matrix at scale {s:e}"
        );
    }
}

#[test]
fn tja_a2_nonantisymmetric_kappa_panics() {
    let d = 4;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2A4);
    let mut bad = random_antisym_coeff(d, &mut rng);
    let v = bad.get(&[0, 1]);
    bad.set(&[0, 1], v + 1.0);
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let r = std::panic::catch_unwind(AssertUnwindSafe(|| kappa_operator(space, &bad)));
    assert!(
        r.is_err(),
        "kappa_operator must reject a non-antisymmetric kappa"
    );
}

#[test]
fn tja_a2_small_scale_kappa_with_roundoff_asymmetry_is_accepted() {
    let d = 5;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2A5);
    let kfull = random_antisym_coeff(d, &mut rng);
    let mut kappa = scale_coeff(&kfull, 1e-4);
    let v = kappa.get(&[0, 1]);
    kappa.set(&[0, 1], v + 1e-15);
    let space = QnSpace::from_n_two_sz(d, 3, 1);

    let op = kappa_operator(space, &kappa);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let y = &op * &x0;
    assert!(
        y.check(),
        "accepted small-scale kappa must still apply cleanly"
    );
}

#[test]
fn tja_lin_exact_linearity_in_kappa() {
    let d = 5;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA10A1);
    let scales = [1e-3f64, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0];
    let mut kappa = random_antisym_coeff(d, &mut rng);
    for p in 1..d {
        for q in 0..p {
            let idx = rng.random_range(0..scales.len());
            let v = kappa.get(&[p, q]) * scales[idx];
            kappa.set(&[p, q], v);
            kappa.set(&[q, p], -v);
        }
    }
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let y1 = kappa_apply(&kappa, &x0);
    let n1 = y1.norm();
    assert!(
        n1 > 1e-6,
        "anti-vacuity: kappa_apply(kappa, x0) must be genuinely nonzero"
    );

    for &s in &[1e-8f64, 1e-15, 1e-16, 1e-20, 1e-40] {
        let scaled = scale_coeff(&kappa, s);
        let ys = kappa_apply(&scaled, &x0);
        let ratio = ys.norm() / (s.abs() * n1);
        assert!(
            (ratio - 1.0).abs() <= 1e-13,
            "exact linearity failed at s={s:e}: ‖kappa_apply(s·k,x)‖/(s·‖y1‖) = {ratio}, want 1 ± 1e-13"
        );
    }
}

#[test]
fn tja_a3_expm_antisym_orthogonal_and_involution() {
    let n = 6;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA3A1);
    let k = random_antisym_flat(n, &mut rng);
    let w = flat_to_nested(&expm_antisym(&k, n), n);
    let wt = mat_transpose(&w);
    let wtw = mat_mul(&wt, &w);
    assert!(max_abs_diff(&wtw, &mat_identity(n)) <= 1e-12, "WᵀW != I");

    let det = mat_det(&w);
    assert!((det - 1.0).abs() <= 1e-9, "det(W) != +1, got {det}");

    let wneg = flat_to_nested(&expm_antisym(&neg_flat(&k), n), n);
    let prod = mat_mul(&w, &wneg);
    assert!(
        max_abs_diff(&prod, &mat_identity(n)) <= 1e-12,
        "W · exp(-K) != I"
    );
}

#[test]
fn tja_a3_expm_antisym_matches_series_at_small_norm() {
    let n = 5;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA3A2);
    let kraw = random_antisym_flat(n, &mut rng);
    let maxabs = kraw.iter().fold(0.0f64, |a, &x| a.max(x.abs())).max(1e-12);
    let k: Vec<f64> = kraw.iter().map(|x| x * 0.05 / maxabs).collect();

    let w = flat_to_nested(&expm_antisym(&k, n), n);
    let series = mat_expm(&flat_to_nested(&k, n));
    assert!(
        max_abs_diff(&w, &series) <= 1e-11,
        "expm_antisym disagrees with the independent series reference at small ‖K‖"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn tja_a3_expm_antisym_handles_kernel_and_odd_dimension() {
    let n = 5;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA3A3);
    let k = random_antisym_flat(n, &mut rng);
    let w = flat_to_nested(&expm_antisym(&k, n), n);
    let wt = mat_transpose(&w);
    assert!(
        max_abs_diff(&mat_mul(&wt, &w), &mat_identity(n)) <= 1e-12,
        "odd-n orthogonality"
    );
    let det = mat_det(&w);
    assert!(
        (det.abs() - 1.0).abs() <= 1e-9,
        "odd-n |det W| != 1, got {det}"
    );

    let n2 = 6;
    let mut kraw = random_antisym_flat(n2, &mut rng);
    for j in 0..n2 {
        kraw[j] = 0.0;
        kraw[j * n2] = 0.0;
    }
    let w2 = flat_to_nested(&expm_antisym(&kraw, n2), n2);
    for j in 0..n2 {
        let want = if j == 0 { 1.0 } else { 0.0 };
        assert_close(
            w2[0][j],
            want,
            1e-11,
            "fixed kernel direction not preserved in row 0",
        );
        assert_close(
            w2[j][0],
            want,
            1e-11,
            "fixed kernel direction not preserved in col 0",
        );
    }
}

#[test]
fn tja_a3_expm_kappa_blocked_matches_full_at_trivial_group() {
    let d = 6;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA3A4);
    let kappa = random_antisym_coeff(d, &mut rng);
    let u_blocked = expm_kappa_blocked(&kappa, PointGroup::TRIVIAL);
    let u_full = expm_antisym(&kappa.data, d);
    assert_eq!(
        u_blocked, u_full,
        "expm_kappa_blocked must be bit-identical to expm_antisym at |G|=1"
    );
}

#[test]
fn tja_a3_dense_kappa_matches_state_coefficients() {
    let d = 4;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA3A5);
    let kappa = random_antisym_coeff(d, &mut rng);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let dets = dets_for(d, space.nup, space.ndn);
    let kd = dense_kappa_matrix(d, &kappa, &dets);

    let c: Vec<f64> = dets
        .iter()
        .map(|(ou, od)| dot(&x0, &tt_state::<f64>(ou, od)))
        .collect();
    let kd_c = mat_vec(&kd, &c);

    let y = kappa_apply(&kappa, &x0);
    let cy: Vec<f64> = dets
        .iter()
        .map(|(ou, od)| dot(&y, &tt_state::<f64>(ou, od)))
        .collect();

    let maxdiff = kd_c
        .iter()
        .zip(&cy)
        .fold(0.0f64, |a, (p, q)| a.max((p - q).abs()));
    assert!(
        maxdiff <= 1e-11,
        "dense Kd·c vs kappa_apply state coefficients mismatch: {maxdiff:e}"
    );
}

#[test]
fn tja_a5_orbital_gradient_matches_tt_fd() {
    let fx = build_a5_fixture(A5_SEED);
    let g = orbital_gradient(&fx.ham, &fx.x);

    let mut maxg = 0.0f64;
    for p in 0..fx.d {
        for q in 0..fx.d {
            maxg = maxg.max(g.get(&[p, q]).abs());
        }
    }
    for p in 0..fx.d {
        for q in 0..fx.d {
            assert_close(
                g.get(&[p, q]),
                -g.get(&[q, p]),
                1e-12 * maxg.max(1.0),
                "orbital_gradient must be exactly antisymmetric",
            );
        }
    }
    assert!(
        maxg > 1e-6,
        "anti-vacuity: orbital_gradient must be genuinely nonzero"
    );

    let dir = directional_derivative(fx.d, &g, &fx.k);
    assert!(
        dir.abs() > 1e-6,
        "anti-vacuity: directional derivative must be genuinely nonzero"
    );

    let h = 1e-3;
    let fd = fd4(|hh| energy_at(&fx, hh), h);
    assert!(
        (fd - dir).abs() <= 1e-9 * dir.abs().max(1.0),
        "A5 convention pin (TT FD arm): fd={fd} dir={dir} |diff|={:e}",
        (fd - dir).abs()
    );
}

#[test]
fn tja_a5_dense_commutator_matches_gradient() {
    let fx = build_a5_fixture(A5_SEED);
    let g = orbital_gradient(&fx.ham, &fx.x);
    let dir = directional_derivative(fx.d, &g, &fx.k);

    let hd = dense_hamiltonian_matrix(fx.d, &fx.t, &fx.v, &fx.dets);
    let kd = dense_kappa_matrix(fx.d, &fx.k, &fx.dets);
    let kh = mat_mul(&kd, &hd);
    let hk = mat_mul(&hd, &kd);
    let comm: Vec<Vec<f64>> = kh
        .iter()
        .zip(&hk)
        .map(|(r1, r2)| r1.iter().zip(r2).map(|(a, b)| a - b).collect())
        .collect();
    let commc = mat_vec(&comm, &fx.c);
    let dir2 = dot_vec(&fx.c, &commc);

    assert!(
        (dir2 - dir).abs() <= 1e-9 * dir.abs().max(1.0),
        "A5 dense commutator arm: dir2={dir2} dir={dir} |diff|={:e}",
        (dir2 - dir).abs()
    );
}

#[test]
fn tja_a5_dense_exponential_flow_fd() {
    let fx = build_a5_fixture(A5_SEED);
    let g = orbital_gradient(&fx.ham, &fx.x);
    let dir = directional_derivative(fx.d, &g, &fx.k);

    let hd = dense_hamiltonian_matrix(fx.d, &fx.t, &fx.v, &fx.dets);
    let kd = dense_kappa_matrix(fx.d, &fx.k, &fx.dets);

    let dense_energy = |hh: f64| -> f64 {
        let w = mat_expm(&mat_scale(&kd, -hh));
        let wc = mat_vec(&w, &fx.c);
        dot_vec(&wc, &mat_vec(&hd, &wc))
    };
    let h = 1e-3;
    let fd3 = fd4(dense_energy, h);
    assert!(
        (fd3 - dir).abs() <= 1e-9 * dir.abs().max(1.0),
        "A5 dense exponential-flow FD arm: fd3={fd3} dir={dir} |diff|={:e}",
        (fd3 - dir).abs()
    );
}

#[test]
fn tja_a5_finite_u_operator_identity() {
    let fx = build_a5_fixture(A5_SEED);
    let hd = dense_hamiltonian_matrix(fx.d, &fx.t, &fx.v, &fx.dets);
    let kd = dense_kappa_matrix(fx.d, &fx.k, &fx.dets);
    let hd_scale = max_abs(&hd).max(1.0);

    for &hh in &[0.05f64, 0.4, 1.5] {
        let u = expm_antisym(&scale_flat(&fx.k.data, -hh), fx.d);
        let rotated = rotate_integrals(&fx.ints, &u);
        let hrot_dense = dense_hamiltonian_matrix(fx.d, &rotated.t, &rotated.v, &fx.dets);

        let w = mat_expm(&mat_scale(&kd, -hh));
        let wt = mat_transpose(&w);
        let predicted = mat_mul(&mat_mul(&wt, &hd), &w);

        let diff = max_abs_diff(&hrot_dense, &predicted);
        assert!(
            diff <= 1e-8 * hd_scale,
            "A5 finite-U operator identity failed at h={hh}: max|Hrot - WᵀHW|={diff:e}"
        );
    }
}

#[test]
fn tja_a5_rotate_integrals_matches_naive_quadruple_loop() {
    let fx = build_a5_fixture(A5_SEED);
    let u = expm_antisym(&scale_flat(&fx.k.data, -0.3), fx.d);
    let rotated = rotate_integrals(&fx.ints, &u);
    let (tp_ref, vp_ref) = naive_rotate_tv(fx.d, &fx.t, &fx.v, &u);

    let mut maxdiff_t = 0.0f64;
    for i in 0..fx.d {
        for j in 0..fx.d {
            maxdiff_t = maxdiff_t.max((rotated.t.get(&[i, j]) - tp_ref.get(&[i, j])).abs());
        }
    }
    assert!(
        maxdiff_t <= 1e-10,
        "rotate_integrals t' disagrees with the independent quadruple-loop reference: {maxdiff_t:e}"
    );

    let mut maxdiff_v = 0.0f64;
    for i in 0..fx.d {
        for j in 0..fx.d {
            for k in 0..fx.d {
                for l in 0..fx.d {
                    maxdiff_v = maxdiff_v
                        .max((rotated.v.get(&[i, j, k, l]) - vp_ref.get(&[i, j, k, l])).abs());
                }
            }
        }
    }
    assert!(
        maxdiff_v <= 1e-9,
        "rotate_integrals v' disagrees with the independent quadruple-loop reference: {maxdiff_v:e}"
    );

    assert_eq!(
        rotated.e0, fx.ints.e0,
        "rotate_integrals must pass e0 through unchanged"
    );
    assert_eq!(
        rotated.norb, fx.ints.norb,
        "rotate_integrals must pass norb through unchanged"
    );
    assert_eq!(
        rotated.nelec, fx.ints.nelec,
        "rotate_integrals must pass nelec through unchanged"
    );
    assert_eq!(
        rotated.ms2, fx.ints.ms2,
        "rotate_integrals must pass ms2 through unchanged"
    );
}

#[test]
fn tja_a5_negative_arms_fail_at_o1() {
    let fx = build_a5_fixture(A5_SEED);
    let g = orbital_gradient(&fx.ham, &fx.x);
    let dir = directional_derivative(fx.d, &g, &fx.k);
    let scale = dir.abs().max(1.0);
    let h = 1e-3;

    let fd_good = fd4(|hh| energy_at(&fx, hh), h);
    let good_residual = (fd_good - dir).abs();

    let energy_sign_flip = |hh: f64| -> f64 {
        let u = expm_antisym(&scale_flat(&fx.k.data, hh), fx.d);
        let rotated = rotate_integrals(&fx.ints, &u);
        let h_op = SparseHamiltonian::new(fx.space, &rotated.t, &rotated.v, None, f64::EPSILON);
        rayleigh_quotient(&h_op, &fx.x)
    };
    let fd_sign = fd4(energy_sign_flip, h);
    let residual_sign = (fd_sign - dir).abs();
    assert!(
        residual_sign >= 1e3 * good_residual.max(1e-14) && residual_sign >= 1e-3 * scale,
        "sign-flip negative arm did not fail at O(1): residual={residual_sign:e} vs good={good_residual:e}"
    );

    let energy_transposed = |hh: f64| -> f64 {
        let u = expm_antisym(&scale_flat(&fx.k.data, -hh), fx.d);
        let ut = transpose_flat(&u, fx.d);
        let rotated = rotate_integrals(&fx.ints, &ut);
        let h_op = SparseHamiltonian::new(fx.space, &rotated.t, &rotated.v, None, f64::EPSILON);
        rayleigh_quotient(&h_op, &fx.x)
    };
    let fd_t = fd4(energy_transposed, h);
    let residual_t = (fd_t - dir).abs();
    assert!(
        residual_t >= 1e3 * good_residual.max(1e-14) && residual_t >= 1e-3 * scale,
        "transpose negative arm did not fail at O(1): residual={residual_t:e} vs good={good_residual:e}"
    );

    let energy_permuted = |hh: f64| -> f64 {
        let u = expm_antisym(&scale_flat(&fx.k.data, -hh), fx.d);
        let (tp, vp) = naive_rotate_tv(fx.d, &fx.t, &fx.v, &u);
        let mut vp_wrong = CoeffTensor::<f64>::zeros(fx.d, 4);
        for a in 0..fx.d {
            for b in 0..fx.d {
                for c in 0..fx.d {
                    for e in 0..fx.d {
                        vp_wrong.set(&[a, b, c, e], vp.get(&[a, b, e, c]));
                    }
                }
            }
        }
        let h_op = SparseHamiltonian::new(fx.space, &tp, &vp_wrong, None, f64::EPSILON);
        rayleigh_quotient(&h_op, &fx.x)
    };
    let fd_perm = fd4(energy_permuted, h);
    let residual_perm = (fd_perm - dir).abs();
    assert!(
        residual_perm >= 1e3 * good_residual.max(1e-14) && residual_perm >= 1e-3 * scale,
        "permuted-v negative arm did not fail at O(1): residual={residual_perm:e} vs good={good_residual:e}"
    );
}

#[test]
fn tja_a7_nonhermitian_h_breaks_the_a5_pin() {
    let d = 4usize;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA7A7);
    let t = random_asym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    let kfull = random_antisym_coeff(d, &mut rng);
    let k = scale_coeff(&kfull, 0.1);

    let ham = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-13, qntt::Pooled::NONE, qntt::round::RoundMode::Rel);
    let nx = x.norm();
    x.scale_mut(1.0 / nx);

    let mut y: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    y.round_mut(1e-13, qntt::Pooled::NONE, qntt::round::RoundMode::Rel);
    let ny = y.norm();
    y.scale_mut(1.0 / ny);

    let hx = ham.apply(&x);
    let hy = ham.apply(&y);
    let xhy = dot(&x, &hy);
    let yhx = dot(&y, &hx);
    assert!(
        (xhy - yhx).abs() >= 1e-3,
        "A7 fixture must be genuinely non-Hermitian: |<x,Hy>-<y,Hx>|={:e}",
        (xhy - yhx).abs()
    );

    let ints = Integrals {
        t: t.clone(),
        v: v.clone(),
        e0: 0.0,
        norb: d,
        nelec: space.n(),
        ms2: 1,
        orbsym: Vec::new(),
        isym: None,
    };
    let g = orbital_gradient(&ham, &x);
    let dir = directional_derivative(d, &g, &k);

    let energy_at_asym = |hh: f64| -> f64 {
        let u = expm_antisym(&scale_flat(&k.data, -hh), d);
        let rotated = rotate_integrals(&ints, &u);
        let h_op = SparseHamiltonian::new(space, &rotated.t, &rotated.v, None, f64::EPSILON);
        rayleigh_quotient(&h_op, &x)
    };
    let h = 1e-3;
    let fd = fd4(energy_at_asym, h);
    let residual = (fd - dir).abs();
    let scale = dir.abs().max(1.0);
    assert!(
        residual >= 1e-3 * scale,
        "A7: orbital_gradient's collapsed 2(γ-γᵀ) form must NOT match the true FD directional \
         derivative once H is non-Hermitian (its ⟨x,HE_pq x⟩=⟨Hx,E_pq x⟩ derivation step is \
         invalid there): residual={residual:e} dir={dir} fd={fd}"
    );
}

#[test]
fn tja_a6_gradient_vanishes_at_dense_exact_eigenstate() {
    let d = 4;
    let space = QnSpace::new(d, 1, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA6A6);
    let t = random_sym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    let ham = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let dets = dets_for(d, 1, 1);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let (evals, evecs) = dense_eigen(&hd);

    for &idx in &[0usize, 2] {
        let lambda = evals[idx];
        let coeffs = &evecs[idx];
        let mut x: Option<TtVector<f64>> = None;
        for (i, (ou, od)) in dets.iter().enumerate() {
            let term = &tt_state::<f64>(ou, od) * coeffs[i];
            x = Some(match x {
                None => term,
                Some(acc) => &acc + &term,
            });
        }
        let mut x = x.expect("nonempty determinant basis");
        let nrm = x.norm();
        x.scale_mut(1.0 / nrm);

        let hx = ham.apply(&x);
        let mut resid = &hx - &(&x * lambda);
        resid.round_mut(1e-14, qntt::Pooled::NONE, qntt::round::RoundMode::Abs);
        let scale = lambda.abs().max(1.0);
        assert!(
            resid.norm() <= 1e-10 * scale,
            "lifted state is not an eigenstate of H at idx={idx}: ‖Hx-λx‖={}",
            resid.norm()
        );

        let g = orbital_gradient(&ham, &x);
        let maxg = (0..d)
            .flat_map(|p| (0..d).map(move |q| (p, q)))
            .map(|(p, q)| g.get(&[p, q]).abs())
            .fold(0.0f64, f64::max);
        assert!(
            maxg <= 1e-10 * scale,
            "orbital_gradient does not vanish at an exact eigenstate (idx={idx}): max|g|={maxg:e}"
        );
    }
}

#[test]
fn tja_a6_orbital_gradient_matches_rdm_route() {
    let fx = build_a5_fixture(A5_SEED);
    let g_direct = orbital_gradient(&fx.ham, &fx.x);
    let hx = fx.ham.apply(&fx.x);
    let gamma = transition_1rdm_bruteforce(&fx.x, &hx);

    let mut maxdiff = 0.0f64;
    let mut maxg = 0.0f64;
    for p in 0..fx.d {
        for q in 0..fx.d {
            let g2 = 2.0 * (gamma.get(&[p, q]) - gamma.get(&[q, p]));
            let g1 = g_direct.get(&[p, q]);
            maxg = maxg.max(g1.abs());
            maxdiff = maxdiff.max((g1 - g2).abs());
        }
    }
    assert!(
        maxg > 1e-6,
        "anti-vacuity: gradient must be genuinely nonzero"
    );
    assert!(
        maxdiff <= 1e-12 * maxg.max(1.0),
        "orbital_gradient disagrees with the 2(γ-γᵀ)/transition_1rdm_bruteforce route: {maxdiff:e}"
    );
}

#[test]
fn tja_a4_transition_1rdm_bruteforce_properties() {
    let d = 5;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA4A1);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let y: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);

    let gamma_xx = transition_1rdm_bruteforce(&x, &x);
    let mut maxsym = 0.0f64;
    let mut maxg = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            let a = gamma_xx.get(&[p, q]);
            let b = gamma_xx.get(&[q, p]);
            maxg = maxg.max(a.abs());
            maxsym = maxsym.max((a - b).abs());
        }
    }
    assert!(
        maxsym <= 1e-12 * maxg.max(1.0),
        "gamma(x,x) is not symmetric: max|diff|={maxsym:e}"
    );

    let mut tr = 0.0f64;
    for p in 0..d {
        tr += gamma_xx.get(&[p, p]);
    }
    let xx = dot(&x, &x);
    let n = (space.nup + space.ndn) as f64;
    assert_close(
        tr,
        n * xx,
        1e-12 * (n * xx).abs().max(1.0),
        "tr gamma(x,x) != N·<x,x>",
    );

    let gamma_xy = transition_1rdm_bruteforce(&x, &y);
    let gamma_yx = transition_1rdm_bruteforce(&y, &x);
    let mut maxt = 0.0f64;
    let mut maxxy = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            maxxy = maxxy.max(gamma_xy.get(&[p, q]).abs());
            maxt = maxt.max((gamma_xy.get(&[p, q]) - gamma_yx.get(&[q, p])).abs());
        }
    }
    assert!(
        maxt <= 1e-13 * maxxy.max(1.0),
        "gamma(x,y) != gamma(y,x)ᵀ: max|diff|={maxt:e}"
    );

    let t = random_sym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    let ham = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let hx = ham.apply(&x);
    let gamma_x_hx = transition_1rdm_bruteforce(&x, &hx);
    let mut tr2 = 0.0f64;
    for p in 0..d {
        tr2 += gamma_x_hx.get(&[p, p]);
    }
    let xhx = dot(&x, &hx);
    assert_close(
        tr2,
        n * xhx,
        1e-10 * (n * xhx).abs().max(1.0),
        "tr gamma(x,Hx) != N·<x,Hx>",
    );
}

#[test]
fn tja_a4_dense_cross_check() {
    let d = 4;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA4A2);
    let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let y: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let gamma = transition_1rdm_bruteforce(&x, &y);
    let dets = dets_for(d, space.nup, space.ndn);
    let cx: Vec<f64> = dets
        .iter()
        .map(|(ou, od)| dot(&x, &tt_state::<f64>(ou, od)))
        .collect();
    let cy: Vec<f64> = dets
        .iter()
        .map(|(ou, od)| dot(&y, &tt_state::<f64>(ou, od)))
        .collect();
    let scale = (x.norm() * y.norm()).max(1.0);
    for p in 0..d {
        for q in 0..d {
            let epq = dense_epq_matrix(d, p, q, &dets);
            let epqcy = mat_vec(&epq, &cy);
            let want = dot_vec(&cx, &epqcy);
            let got = gamma.get(&[p, q]);
            assert_close(
                got,
                want,
                1e-12 * scale,
                &format!("dense cross-check p={p} q={q}"),
            );
        }
    }
}

fn lih_sto3g_ints() -> Integrals<f64> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load =
        load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g loads");
    load.ints
}

#[test]
fn tja_fc_n_core_zero_is_identity() {
    let ints = lih_sto3g_ints();
    let fc0 = freeze_core(&ints, 0);
    assert_eq!(
        fc0.t.data, ints.t.data,
        "freeze_core(_, 0) must be bit-identical on t"
    );
    assert_eq!(
        fc0.v.data, ints.v.data,
        "freeze_core(_, 0) must be bit-identical on v"
    );
    assert_eq!(fc0.e0, ints.e0);
    assert_eq!(fc0.norb, ints.norb);
    assert_eq!(fc0.nelec, ints.nelec);
    assert_eq!(fc0.ms2, ints.ms2);
    assert_eq!(fc0.orbsym, ints.orbsym);
}

#[test]
fn tja_fc_e_core_additivity() {
    let ints = lih_sto3g_ints();
    let n_core = 1;
    let fc = freeze_core(&ints, n_core);

    let mut e_core_naive = 0.0f64;
    for i in 0..n_core {
        e_core_naive += 2.0 * ints.t.get(&[i, i]);
    }
    for i in 0..n_core {
        for j in 0..n_core {
            e_core_naive += 2.0 * ints.v.get(&[i, i, j, j]) - ints.v.get(&[i, j, j, i]);
        }
    }
    assert_close(
        fc.e0,
        ints.e0 + e_core_naive,
        1e-11 * (ints.e0 + e_core_naive).abs().max(1.0),
        "E_core additivity",
    );

    assert_eq!(fc.norb, ints.norb - n_core);
    assert_eq!(fc.nelec, ints.nelec - 2 * n_core);
    assert_eq!(fc.ms2, ints.ms2);
    if !ints.orbsym.is_empty() {
        assert_eq!(fc.orbsym, ints.orbsym[n_core..].to_vec());
    }
}

#[test]
fn tja_fc_dense_oracle_active_matches_restricted_full() {
    let ints = lih_sto3g_ints();
    let n_core = 1;
    let fc = freeze_core(&ints, n_core);
    let d_full = ints.norb;
    let d_act = fc.norb;
    let nup_full = ints.nelec / 2;
    let ndn_full = ints.nelec / 2;

    let dets_full_core_occ: Vec<(Vec<bool>, Vec<bool>)> = dets_for(d_full, nup_full, ndn_full)
        .into_iter()
        .filter(|(ou, od)| ou[0] && od[0])
        .collect();
    assert!(
        !dets_full_core_occ.is_empty(),
        "no core-doubly-occupied determinants — vacuous fixture"
    );
    let hd_full_restricted =
        dense_hamiltonian_matrix(d_full, &ints.t, &ints.v, &dets_full_core_occ);
    let (evals_full, _) = dense_eigen(&hd_full_restricted);
    let lambda_full = evals_full[0];

    let nup_act = fc.nelec / 2;
    let ndn_act = fc.nelec / 2;
    let dets_act = dets_for(d_act, nup_act, ndn_act);
    let hd_act = dense_hamiltonian_matrix(d_act, &fc.t, &fc.v, &dets_act);
    let (evals_act, _) = dense_eigen(&hd_act);
    let lambda_act = evals_act[0];

    assert_close(
        lambda_act + fc.e0,
        lambda_full + ints.e0,
        1e-10 * (lambda_full + ints.e0).abs().max(1.0),
        "freeze_core dense oracle: active ground state + E_core != full core-restricted ground state",
    );
}
