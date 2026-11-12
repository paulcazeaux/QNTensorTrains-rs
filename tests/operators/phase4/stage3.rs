// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase4::*;

fn complex_state(
    space: QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> TtVector<C64> {
    let mut x = randomized_state::<C64, _>(space, lo, hi, rng);
    x.round_mut(1e-14, qntt::Pooled::NONE, RoundMode::Abs);
    x
}

fn dense_apply_hamiltonian_1body<T: Scalar>(
    d: usize,
    t_entries: &[(usize, usize, T)],
    vec: &[T],
) -> Vec<T> {
    let len = vec.len();
    let mut out = vec![T::zero(); len];
    for &(i, j, coeff) in t_entries {
        for spin in [Spin::Up, Spin::Dn] {
            let term = dense_apply_string(
                d,
                &[
                    (Orbital::new(i, spin), true),
                    (Orbital::new(j, spin), false),
                ],
                vec,
            );
            for (o, t) in out.iter_mut().zip(term) {
                *o += t * coeff;
            }
        }
    }
    out
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
fn t3_elem_3body_dense_oracle() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x3E3);
    let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let x0d = x0.to_dense();

    let i = Orbital::new(0, Spin::Up);
    let j = Orbital::new(1, Spin::Up);
    let k = Orbital::new(2, Spin::Dn);
    let l = Orbital::new(3, Spin::Dn);
    let m = Orbital::new(4, Spin::Up);
    let n = Orbital::new(2, Spin::Up);
    let coeff = 0.31_f64;
    let x = adag_i_adag_k_adag_m_a_n_a_l_a_j(&x0, i, j, k, l, m, n, coeff);

    let refr: Vec<f64> = dense_apply_string(
        d,
        &[
            (i, true),
            (k, true),
            (m, true),
            (n, false),
            (l, false),
            (j, false),
        ],
        &x0d,
    )
    .into_iter()
    .map(|v| v * coeff)
    .collect();
    let got = x.to_dense();
    let tol = 50.0 * d as f64 * f64::EPSILON;
    assert!(rel_err(&got, &refr) < tol, "3-body dense oracle mismatch");
}

#[test]
fn t3_ham_rayleigh() {
    let d = 6usize;
    let space = QnSpace::from_n_two_sz(d, 6, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x139);
    let mut x: TtVector<f64> = randomized_state(space, 1, 6, &mut rng);
    x.round_mut(1e-10, qntt::Pooled::NONE, RoundMode::Rel);

    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                t.set(&[p, q], 1.0 / (p as f64 - q as f64).abs());
            }
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d - 1 {
        for q in p + 1..d {
            for r in 0..d - 1 {
                for s in r + 1..d {
                    v.set(
                        &[p, q, r, s],
                        2.0 / (p as f64 + r as f64 - q as f64 - s as f64 + 0.5).abs(),
                    );
                }
            }
        }
    }
    let mut w = CoeffTensor::<f64>::zeros(d, 6);
    for _ in 0..50 {
        let idx = [
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
        ];
        w.set(&idx, rng.random_range(-1.0..1.0));
    }
    let h = SparseHamiltonian::new(space, &t, &v, Some(&w), f64::EPSILON);

    let xd = x.to_dense();
    let hxd = dense_apply_hamiltonian_full(d, &t, &v, &w, &xd);
    let oracle: f64 = xd.iter().zip(&hxd).map(|(&a, &b)| a * b).sum();
    let rq = rayleigh_quotient(&h, &x);
    let scale = (dense_norm(&xd) * dense_norm(&hxd)).max(1e-300);
    assert!(
        (rq - oracle).abs() <= 1e-10 * scale,
        "RayleighQuotient(H,x)={rq} != dense ⟨x,Hx⟩={oracle}"
    );
}

fn complex_hermitian_t() -> [(usize, usize, C64); 2] {
    [
        (1, 3, -C64::from_polar(1.0, 0.9)),
        (3, 1, -C64::from_polar(1.0, -0.9)),
    ]
}

#[test]
fn t3_ham_complex_dense_oracle() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t_entries = complex_hermitian_t();
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    for &(i, j, c) in &t_entries {
        t.set(&[i, j], c);
    }
    let v = CoeffTensor::<C64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC04);
    let x = complex_state(space, 2, 4, &mut rng);
    let y = &h * &x;

    let xd = x.to_dense();
    let expect = dense_apply_hamiltonian_1body(d, &t_entries, &xd);
    let got = y.to_dense();
    let tol = 50.0 * d as f64 * f64::EPSILON;
    assert!(rel_err(&got, &expect) < tol, "H*x dense oracle mismatch");
}

#[test]
fn t3_ham_complex_ground() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    for &(i, j, c) in &complex_hermitian_t() {
        t.set(&[i, j], c);
    }
    let v = CoeffTensor::<C64>::zeros(d, 4);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let basis: Vec<(usize, usize)> = (0..d).flat_map(|u| (0..d).map(move |w| (u, w))).collect();
    let n = basis.len();
    let mut hd = faer::Mat::<C64>::zeros(n, n);
    for (col, &(u, w)) in basis.iter().enumerate() {
        let x0: TtVector<C64> = state_from_sites(d, &[u], &[w]);
        let y = &h * &x0;
        let yd = y.to_dense();
        for (row, &(ur, wr)) in basis.iter().enumerate() {
            *hd.get_mut(row, col) = yd[det_index(d, ur, wr)];
        }
    }
    let eigs = hd
        .as_ref()
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .expect("self-adjoint eigendecomposition");
    let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        (min_eig - (-2.0)).abs() < 1e-4,
        "complex Hermitian 1-body ground energy mismatch: min_eig={min_eig}"
    );
}

#[test]
fn t3_ham_matvec_threaded_bit_identical() {
    let d = 6usize;
    let space = QnSpace::from_n_two_sz(d, 6, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7DE);
    let x0: TtVector<f64> = randomized_state(space, 1, 6, &mut rng);

    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                t.set(&[p, q], 1.0 / (p as f64 - q as f64).abs());
            }
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[0, 1, 2, 3], 0.7);
    v.set(&[2, 3, 0, 1], 0.7);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build 1-thread rayon pool");
    let y_serial = serial_pool.install(|| &h * &x0);
    let y_parallel = &h * &x0;

    assert_eq!(
        y_serial.ranks, y_parallel.ranks,
        "threaded vs serial H*x: rank grids differ"
    );
    for k in 0..d {
        for f in Family::ALL {
            for s in y_serial.cores[k].transitions(f) {
                let a = y_serial.cores[k].block(f, s).expect("serial block present");
                let b = y_parallel.cores[k]
                    .block(f, s)
                    .expect("parallel block present");
                assert_eq!(a.nrows(), b.nrows());
                assert_eq!(a.ncols(), b.ncols());
                for col in 0..a.ncols() {
                    for row in 0..a.nrows() {
                        assert_eq!(
                            *a.get(row, col),
                            *b.get(row, col),
                            "core {k} family {f:?} sector {s:?} entry ({row},{col}) differs"
                        );
                    }
                }
            }
        }
    }
}
