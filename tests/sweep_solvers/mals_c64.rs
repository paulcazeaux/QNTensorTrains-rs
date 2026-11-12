// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::mals::{MalsOptions, mals};
use qntt::su2::core::{Family as Su2Family, Su2Core};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::solvers::mals::mals as su2_mals;
use qntt::su2::vector::{Su2TtVector, tt_csf};
use qntt::{C64, CoeffTensor, QnSpace, Scalar, SparseCore, SparseHamiltonian, TtVector};
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn cast_tt_c64(x: &TtVector<f64>) -> TtVector<C64> {
    let space = x.space;
    let cores: Vec<SparseCore<C64>> = (0..space.d)
        .map(|k| {
            let c = x.core(k);
            let mut out = SparseCore::<C64>::zeros(space, k, &x.ranks[k], &x.ranks[k + 1]);
            for i in 0..c.num_coeffs() {
                out.set_coeff(i, C64::new(c.coeff(i), 0.0));
            }
            out
        })
        .collect();
    TtVector::new(space, cores)
}

fn cast_su2_c64(x: &Su2TtVector<f64>) -> Su2TtVector<C64> {
    let space = x.space;
    let cores: Vec<Su2Core<C64>> = (0..space.d)
        .map(|k| {
            let c = x.core(k);
            let mut out = Su2Core::<C64>::zeros(space, k, &x.ranks[k], &x.ranks[k + 1]);
            for i in 0..c.num_coeffs() {
                out.set_coeff(i, C64::new(c.coeff(i), 0.0));
            }
            out
        })
        .collect();
    Su2TtVector::new(space, cores)
}

fn rho_abelian(x: &TtVector<C64>) -> f64 {
    let mut xb = x.clone();
    for k in 0..x.space.d {
        let n = xb.cores[k].num_coeffs();
        for i in 0..n {
            let c = xb.cores[k].coeff(i);
            xb.cores[k].set_coeff(i, c.conj_s());
        }
    }
    qntt::dot(&xb, x).abs_s() / qntt::dot(x, x).re_s().max(1e-300)
}

fn rho_su2(x: &Su2TtVector<C64>) -> f64 {
    let mut xb = x.clone();
    for k in 0..x.space.d {
        let n = xb.cores[k].num_coeffs();
        for i in 0..n {
            let c = xb.cores[k].coeff(i);
            xb.cores[k].set_coeff(i, c.conj_s());
        }
    }
    qntt::su2::ops::dot(&xb, x).abs_s() / qntt::su2::ops::dot(x, x).re_s().max(1e-300)
}

fn gauge_integrals(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    theta: &[f64],
) -> (CoeffTensor<C64>, CoeffTensor<C64>) {
    let ph = |a: f64| C64::new(a.cos(), a.sin());
    let mut tc = CoeffTensor::<C64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c != 0.0 {
                tc.set(&[i, j], ph(theta[i] - theta[j]) * C64::new(c, 0.0));
            }
        }
    }
    let mut vc = CoeffTensor::<C64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c != 0.0 {
                        vc.set(
                            &[i, j, k, l],
                            ph(theta[i] - theta[j] + theta[k] - theta[l]) * C64::new(c, 0.0),
                        );
                    }
                }
            }
        }
    }
    (tc, vc)
}

fn opts_exact() -> MalsOptions {
    MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 40,
        verbose: false,
        ..Default::default()
    }
}

#[test]
fn mc1_su2_mals_c64_real_embedding_matches_f64_and_stays_real() {
    let d = 6usize;
    let space = Su2Space::new(d, 6, 0);
    let (t, v) = generic_truncated_tv(d);
    let ham_r = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (tc, vc) = gauge_integrals(d, &t, &v, &vec![0.0; d]);
    let ham_c = Su2SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);

    let codes: Vec<Su2Family> = (0..d)
        .map(|i| {
            if i < 3 {
                Su2Family::Occ
            } else {
                Su2Family::Empty
            }
        })
        .collect();
    let mut x0: Su2TtVector<f64> = tt_csf(&codes);
    x0.right_orthogonalize(true);
    let opts = opts_exact();
    let (lr, _xr) = su2_mals(&ham_r, &x0, &opts);
    let (lc, xc) = su2_mals(&ham_c, &cast_su2_c64(&x0), &opts);
    let rho = rho_su2(&xc);
    println!("mc1: λ_f64 = {lr:.12}, λ_C64 = {lc:.12}, ρ = {rho:.15}");
    assert!(
        (lc - lr).abs() <= 1e-10 * lr.abs().max(1.0),
        "C64 MALS must reproduce the f64 energy on a real-embedded problem: {lc} vs {lr}"
    );
    assert!(
        (1.0 - rho).abs() < 1e-10,
        "the real embedding must stay real (ρ = {rho})"
    );
}

#[test]
fn mc2_su2_mals_c64_is_gauge_covariant_and_genuinely_complex() {
    let d = 6usize;
    let space = Su2Space::new(d, 6, 0);
    let (t, v) = generic_truncated_tv(d);
    let ham_r = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let theta: Vec<f64> = (0..d).map(|p| 0.37 + 0.91 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
    let ham_c = Su2SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);

    let codes: Vec<Su2Family> = (0..d)
        .map(|i| {
            if i < 3 {
                Su2Family::Occ
            } else {
                Su2Family::Empty
            }
        })
        .collect();
    let mut x0: Su2TtVector<f64> = tt_csf(&codes);
    x0.right_orthogonalize(true);
    let opts = opts_exact();
    let (lr, _xr) = su2_mals(&ham_r, &x0, &opts);
    let (lc, xc) = su2_mals(&ham_c, &cast_su2_c64(&x0), &opts);
    let rho = rho_su2(&xc);
    println!("mc2: λ_f64 = {lr:.12}, λ_C64(gauge) = {lc:.12}, ρ = {rho:.6}");
    assert!(
        (lc - lr).abs() <= 1e-9 * lr.abs().max(1.0),
        "a diagonal orbital gauge is unitary, so the SU(2) C64 MALS energy must equal the real \
         one exactly: {lc} vs {lr}"
    );
    assert!(
        rho < 0.99,
        "the gauge-transformed ground state must be genuinely complex, got ρ = {rho}"
    );
}

#[test]
fn mc3_abelian_mals_c64_is_gauge_covariant_and_genuinely_complex() {
    let d = 6usize;
    let space = QnSpace::new(d, 3, 3);
    let (t, v) = generic_truncated_tv(d);
    let ham_r = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let theta: Vec<f64> = (0..d).map(|p| 0.41 + 0.77 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &t, &v, &theta);
    let ham_c = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0FFEE);
    let x0: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    let opts = opts_exact();
    let (lr, _xr) = mals(&ham_r, &x0, &opts);
    let (lc, xc) = mals(&ham_c, &cast_tt_c64(&x0), &opts);
    let rho = rho_abelian(&xc);
    println!("mc3: λ_f64 = {lr:.12}, λ_C64(gauge) = {lc:.12}, ρ = {rho:.6}");
    assert!(
        (lc - lr).abs() <= 1e-9 * lr.abs().max(1.0),
        "abelian C64 MALS must be gauge covariant: {lc} vs {lr}"
    );
    assert!(
        rho < 0.99,
        "the gauge-transformed ground state must be genuinely complex, got ρ = {rho}"
    );
}

#[test]
fn mc4_abelian_mals_c64_real_embedding_matches_f64_and_stays_real() {
    let d = 6usize;
    let space = QnSpace::new(d, 3, 3);
    let (t, v) = generic_truncated_tv(d);
    let ham_r = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let (tc, vc) = gauge_integrals(d, &t, &v, &vec![0.0; d]);
    let ham_c = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF01);
    let x0: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    let opts = opts_exact();
    let (lr, _xr) = mals(&ham_r, &x0, &opts);
    let (lc, xc) = mals(&ham_c, &cast_tt_c64(&x0), &opts);
    let rho = rho_abelian(&xc);
    println!("mc4: λ_f64 = {lr:.12}, λ_C64 = {lc:.12}, ρ = {rho:.15}");
    assert!((lc - lr).abs() <= 1e-10 * lr.abs().max(1.0), "{lc} vs {lr}");
    assert!(
        (1.0 - rho).abs() < 1e-10,
        "the real embedding must stay real (ρ = {rho})"
    );
}
