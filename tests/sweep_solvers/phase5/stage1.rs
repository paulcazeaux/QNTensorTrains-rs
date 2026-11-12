// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase5::*;
use qntt_testkit::dense_apply_string;

fn dense_norm<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt()
}

fn dense_apply_hamiltonian_full<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: &CoeffTensor<T>,
    vec: &[T],
) -> Vec<T> {
    let len = vec.len();
    let mut out = vec![T::zero(); len];
    let spins = [Spin::Up, Spin::Dn];
    let accumulate = |ops: &[(Orbital, bool)], coeff: T, out: &mut [T]| {
        let term = dense_apply_string(d, ops, vec);
        for (o, tm) in out.iter_mut().zip(term) {
            *o += tm * coeff;
        }
    };

    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c == T::zero() {
                continue;
            }
            for &s in &spins {
                accumulate(
                    &[(Orbital::new(i, s), true), (Orbital::new(j, s), false)],
                    c,
                    &mut out,
                );
            }
        }
    }

    let half = T::from_f64(0.5);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c == T::zero() {
                        continue;
                    }
                    for &s1 in &spins {
                        for &s2 in &spins {
                            let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                            let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                            if oi != ok && oj != ol {
                                accumulate(
                                    &[(oi, true), (ok, true), (ol, false), (oj, false)],
                                    c * half,
                                    &mut out,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    let sixth = T::from_f64(1.0 / 6.0);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    for m in 0..d {
                        for n in 0..d {
                            let c = w.get(&[i, j, k, l, m, n]);
                            if c == T::zero() {
                                continue;
                            }
                            for &s1 in &spins {
                                for &s2 in &spins {
                                    for &s3 in &spins {
                                        let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                                        let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                                        let (om, on) = (Orbital::new(m, s3), Orbital::new(n, s3));
                                        let cd = oi != ok && oi != om && ok != om;
                                        let ad = oj != ol && oj != on && ol != on;
                                        if cd && ad {
                                            accumulate(
                                                &[
                                                    (oi, true),
                                                    (ok, true),
                                                    (om, true),
                                                    (on, false),
                                                    (ol, false),
                                                    (oj, false),
                                                ],
                                                c * sixth,
                                                &mut out,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    out
}

#[test]
fn t1_sketched_hamiltonian_onesite_rayleigh_identity() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A02);
    let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    x.right_orthogonalize(true);
    let k = d / 2;
    x.move_core(k, true);
    x.check();

    let fr = right_to_left_sketching(&h, &x);
    let fl = if k == 0 {
        boundary_idsketch(&x, 0, 1)
    } else {
        left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap()
    };
    let ws = onesite_workspace(&h, x.core(k));
    let yk = h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws);
    let lhs = x.core(k).dot_cores(&yk);

    let xd = x.to_dense();
    let w0 = CoeffTensor::<f64>::zeros(d, 6);
    let hxd = dense_apply_hamiltonian_full(d, &t, &v, &w0, &xd);
    let oracle: f64 = xd.iter().zip(&hxd).map(|(&a, &b)| a * b).sum();
    let tol = 1e-9 * oracle.abs().max(1.0);
    assert!(
        (lhs - oracle).abs() < tol,
        "onesite sketched Rayleigh identity mismatch: {lhs} vs {oracle}"
    );
}

#[test]
fn t1_als_sweep_threaded_matches_serial() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A02);
    let x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    let opts = AlsOptions {
        eps: 1e-8,
        max_iter: 3,
        verbose: false,
    };
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| als(&h, &x, &opts).0)
    };
    let l1 = run(1);
    let l8 = run(8);
    assert!(
        (l1 - l8).abs() < 1e-10 * l1.abs().max(1.0),
        "ALS threaded vs serial energy mismatch: {l1} vs {l8}"
    );
}

#[test]
fn t1_env_sweep_reproduces_rayleigh() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A04);
    let x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    x.check();

    let mut p = boundary_idsketch(&x, 0, 1);
    for k in 0..d {
        let ws = onesite_workspace(&h, x.core(k));
        p = h.sketching_step_right(&x, &p, &ws);
    }
    let last = Sector::new(space.nup, space.ndn);
    let scalar = *p.block(last).unwrap().get(0, 0);

    let xd = x.to_dense();
    let w0 = CoeffTensor::<f64>::zeros(d, 6);
    let hxd = dense_apply_hamiltonian_full(d, &t, &v, &w0, &xd);
    let oracle: f64 = xd.iter().zip(&hxd).map(|(&a, &b)| a * b).sum();
    let tol = 1e-9 * oracle.abs().max(1.0);
    assert!(
        (scalar - oracle).abs() < tol,
        "env sweep vs dense Rayleigh mismatch: {scalar} vs {oracle}"
    );
}

#[test]
fn t1_rayleigh_fused_regression() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 4, 1], -1.0);
    v.set(&[1, 4, 0, 2], -1.0);
    t.set(&[0, 1], 0.3);
    t.set(&[1, 0], 0.3);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng_x = Xoshiro256PlusPlus::seed_from_u64(0x5A05);
    let x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng_x);
    let mut rng_y = Xoshiro256PlusPlus::seed_from_u64(0x5A06);
    let y: TtVector<f64> = randomized_state(space, 2, 4, &mut rng_y);
    x.check();
    y.check();

    let xd = x.to_dense();
    let yd = y.to_dense();
    let w0 = CoeffTensor::<f64>::zeros(d, 6);
    let hxd = dense_apply_hamiltonian_full(d, &t, &v, &w0, &xd);

    let oracle_rq: f64 = xd.iter().zip(&hxd).map(|(&a, &b)| a * b).sum();
    let rq = rayleigh_quotient(&h, &x);
    let scale_rq = (dense_norm(&xd) * dense_norm(&hxd)).max(1e-300);
    assert!(
        (rq - oracle_rq).abs() <= 1e-10 * scale_rq,
        "rayleigh_quotient regression: {rq} vs {oracle_rq}"
    );

    let oracle_xthy: f64 = yd.iter().zip(&hxd).map(|(&a, &b)| a * b).sum();
    let xthy = xt_h_y(&y, &h, &x);
    let scale_xthy = (dense_norm(&yd) * dense_norm(&hxd)).max(1e-300);
    assert!(
        (xthy - oracle_xthy).abs() <= 1e-10 * scale_xthy,
        "xt_h_y regression: {xthy} vs {oracle_xthy}"
    );
}

#[test]
fn t1_env_sweep_reproduces_rayleigh_complex() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    for &(i, j, c) in &complex_hermitian_t() {
        t.set(&[i, j], c);
    }
    let v = CoeffTensor::<C64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A0B);
    let x: TtVector<C64> = randomized_state(space, 2, 4, &mut rng);
    x.check();

    let xd = x.to_dense();
    let w0 = CoeffTensor::<C64>::zeros(d, 6);
    let hxd = dense_apply_hamiltonian_full(d, &t, &v, &w0, &xd);
    let oracle: C64 = xd.iter().zip(&hxd).map(|(&a, &b)| a.conj_s() * b).sum();
    assert!(
        oracle.abs_s() > 1e-6,
        "vacuous complex oracle (‖⟨x,Hx⟩‖ == 0)"
    );

    let mut p = boundary_idsketch(&x, 0, 1);
    for k in 0..d {
        let ws = onesite_workspace(&h, x.core(k));
        p = h.sketching_step_right(&x, &p, &ws);
    }
    let last = Sector::new(space.nup, space.ndn);
    let scalar_lr = *p.block(last).unwrap().get(0, 0);

    let mut q = boundary_idsketch(&x, d, 1);
    for k in (0..d).rev() {
        let ws = onesite_workspace(&h, x.core(k));
        q = h.sketching_step_left(&x, &q, &ws);
    }
    let scalar_rl = *q.block(Sector::new(0, 0)).unwrap().get(0, 0);

    let rq = rayleigh_quotient(&h, &x);

    let tol = 1e-10 * oracle.abs_s().max(1.0);
    assert!(
        (scalar_lr - oracle).abs_s() < tol,
        "L→R env sweep vs complex Rayleigh oracle: {scalar_lr} vs {oracle}"
    );
    assert!(
        (scalar_rl - oracle).abs_s() < tol,
        "R→L env sweep vs complex Rayleigh oracle: {scalar_rl} vs {oracle}"
    );
    assert!(
        (rq - oracle).abs_s() < tol,
        "fused rayleigh_quotient vs complex oracle: {rq} vs {oracle}"
    );
}
