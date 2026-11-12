// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

pub mod dense;
pub mod gad_mol;
pub mod joint_d2;
pub mod pg;

use qntt::qn::{QnSpace, RankGrid};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::{Su2RankGrid, Su2Space};
use qntt::su2::vector::Su2TtVector;
use qntt::{C64, CoeffTensor, Orbital, Scalar, SparseHamiltonian, Spin, TtVector};
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub fn generic_truncated_tv<T: Scalar>(d: usize) -> (CoeffTensor<T>, CoeffTensor<T>) {
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for p in 0..d - 1 {
        t.set(&[p, p + 1], T::from_f64(-1.0));
        t.set(&[p + 1, p], T::from_f64(-1.0));
    }
    for p in 0..d {
        t.set(&[p, p], T::from_f64(0.15 * (p as f64 - 0.5 * d as f64)));
    }
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(
                    &[p, p, q, q],
                    T::from_f64(0.5 / (1.0 + (p as f64 - q as f64).abs())),
                );
            }
        }
    }
    (t, v)
}

pub fn generic_truncated_h<T: Scalar>(space: QnSpace) -> SparseHamiltonian<T> {
    let (t, v) = generic_truncated_tv::<T>(space.d);
    SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub fn generic_su2_h<T: Scalar>(space: Su2Space) -> Su2SparseHamiltonian<T> {
    let (t, v) = generic_truncated_tv::<T>(space.d);
    Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub fn generic_complex_tv(d: usize) -> (CoeffTensor<C64>, CoeffTensor<C64>) {
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    for p in 0..d - 1 {
        let phi = 0.3 + 0.4 * p as f64;
        t.set(&[p, p + 1], -C64::new(phi.cos(), phi.sin()));
        t.set(&[p + 1, p], -C64::new(phi.cos(), -phi.sin()));
    }
    for p in 0..d {
        t.set(&[p, p], C64::from_f64(0.15 * (p as f64 - 0.5 * d as f64)));
    }
    let mut v = CoeffTensor::<C64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(
                    &[p, p, q, q],
                    C64::from_f64(0.5 / (1.0 + (p as f64 - q as f64).abs())),
                );
            }
        }
    }
    (t, v)
}

pub fn generic_complex_h(space: QnSpace) -> SparseHamiltonian<C64> {
    let (t, v) = generic_complex_tv(space.d);
    SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub fn generic_su2_h_c64(space: Su2Space) -> Su2SparseHamiltonian<C64> {
    let (t, v) = generic_complex_tv(space.d);
    Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub fn randomized_state<T: Scalar, R: Rng + ?Sized>(
    space: QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut R,
) -> TtVector<T> {
    let d = space.d;
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    qntt::tt_randn(space, &ranks, rng)
}

pub fn randomized_su2_state<T: Scalar, R: Rng + ?Sized>(
    space: Su2Space,
    lo: usize,
    hi: usize,
    rng: &mut R,
) -> Su2TtVector<T> {
    let d = space.d;
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    qntt::su2::vector::tt_randn(space, &ranks, rng)
}

pub fn dense_apply_string<T: Scalar>(d: usize, ops: &[(Orbital, bool)], vec: &[T]) -> Vec<T> {
    let len = 1usize << (2 * d);
    assert_eq!(vec.len(), len);
    let mut out = vec![T::zero(); len];
    for (idx, &coeff) in vec.iter().enumerate() {
        if coeff == T::zero() {
            continue;
        }
        let mut digs = idx;
        let mut list: Vec<Orbital> = Vec::new();
        for k in 0..d {
            let fam = digs & 3;
            digs >>= 2;
            if fam & 1 != 0 {
                list.push(Orbital::new(k, Spin::Up));
            }
            if fam & 2 != 0 {
                list.push(Orbital::new(k, Spin::Dn));
            }
        }
        let mut sign: f64 = 1.0;
        let mut ok = true;
        for &(orb, creation) in ops.iter().rev() {
            if creation {
                match list.binary_search(&orb) {
                    Ok(_) => {
                        ok = false;
                        break;
                    }
                    Err(pos) => {
                        if pos % 2 == 1 {
                            sign = -sign;
                        }
                        list.insert(pos, orb);
                    }
                }
            } else {
                match list.binary_search(&orb) {
                    Ok(pos) => {
                        if pos % 2 == 1 {
                            sign = -sign;
                        }
                        list.remove(pos);
                    }
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if !ok {
            continue;
        }
        let mut occ_up = vec![false; d];
        let mut occ_dn = vec![false; d];
        for o in &list {
            match o.spin {
                Spin::Up => occ_up[o.site] = true,
                Spin::Dn => occ_dn[o.site] = true,
            }
        }
        let mut out_idx = 0usize;
        for k in (0..d).rev() {
            let fam = (occ_up[k] as usize) | ((occ_dn[k] as usize) << 1);
            out_idx = (out_idx << 2) | fam;
        }
        out[out_idx] += T::from_f64(sign) * coeff;
    }
    out
}
