// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Dense 0-based coefficient tensor for the Hamiltonian's one-, two-, and three-body coefficients:
//! a flat row-major `d^rank` array. Fields are deliberately `pub`: `SparseHamiltonian::new`
//! reads `d`/`rank` to shape-validate against the space and scans `data` to screen the `v`/`w`
//! channels, so the shape metadata and buffer are part of the type's contract.

use crate::scalar::Scalar;

/// `rank`-dimensional (2, 4, or 6) dense coefficient tensor over `d` sites, row-major, 0-based.
#[derive(Clone, Debug)]
pub struct CoeffTensor<T: Scalar> {
    pub d: usize,
    pub rank: usize,
    pub data: Vec<T>,
}

impl<T: Scalar> CoeffTensor<T> {
    /// All-zero tensor of the given `rank` over `d` sites (`d.pow(rank)` entries).
    pub fn zeros(d: usize, rank: usize) -> Self {
        let n = d.pow(rank as u32);
        CoeffTensor {
            d,
            rank,
            data: vec![T::zero(); n],
        }
    }

    #[inline]
    fn flat(&self, idx: &[usize]) -> usize {
        debug_assert_eq!(idx.len(), self.rank, "index arity must equal tensor rank");
        let mut off = 0usize;
        for &i in idx {
            debug_assert!(i < self.d, "index {i} out of range 0..{}", self.d);
            off = off * self.d + i;
        }
        off
    }

    /// Entry at 0-based multi-index `idx` (`idx.len() == rank`).
    pub fn get(&self, idx: &[usize]) -> T {
        self.data[self.flat(idx)]
    }

    /// Set entry at 0-based multi-index `idx`.
    pub fn set(&mut self, idx: &[usize], val: T) {
        let off = self.flat(idx);
        self.data[off] = val;
    }
}

/// Read surface `enumerate_3body` needs from a three-body coefficient tensor.
/// Implemented by the dense [`CoeffTensor`] (small-`d` tests) and [`Sparse3Body`] (FCIDUMP
/// scale). Consumed as `&dyn ThreeBody<T>` so `SparseHamiltonian::new` keeps one non-generic
/// signature and both backings coerce at the call site. Dispatch cost is construction-only,
/// never on the matvec path.
pub trait ThreeBody<T: Scalar> {
    /// Orbital count `d`.
    fn norb(&self) -> usize;
    /// Entry at the 6 axis indices `[c0,a0,c1,a1,c2,a2]` (creation/annihilation slot order — the
    /// SAME raw axis convention [`CoeffTensor::flat`]/`read_tcdump`'s slot map use). Missing/unset
    /// ⇒ `T::zero()`.
    fn entry(&self, idx: &[usize; 6]) -> T;
    /// The nonzero support above `tau`: every 6-index with stored magnitude `> tau`. Output order
    /// is irrelevant (candidates pass through sorted `BTreeSet`s downstream); a `Vec` in whatever
    /// order the backing scans.
    fn support_above(&self, tau: f64) -> Vec<[usize; 6]>;
}

impl<T: Scalar> ThreeBody<T> for CoeffTensor<T> {
    fn norb(&self) -> usize {
        debug_assert_eq!(self.rank, 6, "ThreeBody::norb: tensor is not rank-6");
        self.d
    }

    fn entry(&self, idx: &[usize; 6]) -> T {
        self.get(idx)
    }

    fn support_above(&self, tau: f64) -> Vec<[usize; 6]> {
        let d = self.d;
        let mut out = Vec::new();
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        for m in 0..d {
                            for n in 0..d {
                                let idx = [i, j, k, l, m, n];
                                if self.get(&idx).abs_s() > tau {
                                    out.push(idx);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

/// Sparse three-body coefficient tensor: `FxHashMap` from the raw 6-axis key to the value, storing
/// ONLY explicitly-set (post-symmetry-completion) slots. A dense `d⁶` tensor `w`
/// plus a separate occupancy mask collapse to one map: key-presence IS `wset` (the
/// `d^6`-byte mask vanishes). 0-based sites; `d ≤ 126` (the 7-bit label-packing cap already
/// asserted in `assembly::assemble`) so each site fits `u8`.
#[derive(Clone, Debug)]
pub struct Sparse3Body<T: Scalar> {
    pub d: usize,
    data: rustc_hash::FxHashMap<[u8; 6], T>,
}

#[inline]
fn key(idx: &[usize; 6]) -> [u8; 6] {
    let mut k = [0u8; 6];
    for (ki, &i) in k.iter_mut().zip(idx.iter()) {
        debug_assert!(
            i < 128,
            "Sparse3Body: site index {i} out of the u8/d≤126 range"
        );
        *ki = i as u8;
    }
    k
}

impl<T: Scalar> Sparse3Body<T> {
    /// Empty sparse tensor over `d` orbitals (no stored slots).
    pub fn zeros(d: usize) -> Self {
        Sparse3Body {
            d,
            data: rustc_hash::FxHashMap::default(),
        }
    }

    /// Unconditional set — the explicit-entry pass.
    pub fn set(&mut self, idx: &[usize; 6], val: T) {
        self.data.insert(key(idx), val);
    }

    /// Insert only if unset; returns whether inserted — the first-write-wins rule the
    /// symmetry-completion pass needs.
    pub fn set_if_absent(&mut self, idx: &[usize; 6], val: T) -> bool {
        use std::collections::hash_map::Entry;
        match self.data.entry(key(idx)) {
            Entry::Occupied(_) => false,
            Entry::Vacant(e) => {
                e.insert(val);
                true
            }
        }
    }

    /// Entry or zero (dense read of an unset slot).
    pub fn get(&self, idx: &[usize; 6]) -> T {
        self.data.get(&key(idx)).copied().unwrap_or(T::zero())
    }

    /// Stored-slot count; the round-trip tests' `count_nonzero`.
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Scale every stored entry. `O(nnz)`.
    pub fn scale_all(&mut self, factor: T) {
        for v in self.data.values_mut() {
            *v *= factor;
        }
    }

    /// Iterate stored support as `(0-based 6-index, value)` — test hook (`frob_diff` etc.).
    pub fn support(&self) -> impl Iterator<Item = ([usize; 6], T)> + '_ {
        self.data.iter().map(|(k, &v)| {
            let mut idx = [0usize; 6];
            for (i, &ki) in idx.iter_mut().zip(k.iter()) {
                *i = ki as usize;
            }
            (idx, v)
        })
    }

    /// Sparse copy of a dense rank-6 tensor (test ergonomic; asserts `rank == 6`).
    pub fn from_dense(w: &CoeffTensor<T>) -> Self {
        assert_eq!(
            w.rank, 6,
            "Sparse3Body::from_dense: source tensor is not rank-6"
        );
        let mut out = Sparse3Body::zeros(w.d);
        let d = w.d;
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        for m in 0..d {
                            for n in 0..d {
                                let idx = [i, j, k, l, m, n];
                                let v = w.get(&idx);
                                if v != T::zero() {
                                    out.set(&idx, v);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

impl<T: Scalar> ThreeBody<T> for Sparse3Body<T> {
    fn norb(&self) -> usize {
        self.d
    }

    fn entry(&self, idx: &[usize; 6]) -> T {
        self.get(idx)
    }

    fn support_above(&self, tau: f64) -> Vec<[usize; 6]> {
        self.support()
            .filter(|(_, v)| v.abs_s() > tau)
            .map(|(idx, _)| idx)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn sparse3body_set_get_missing() {
        let mut w = Sparse3Body::<f64>::zeros(4);
        w.set(&[0, 1, 2, 3, 1, 0], 2.5);
        assert_eq!(w.get(&[0, 1, 2, 3, 1, 0]), 2.5);
        assert_eq!(w.get(&[3, 3, 3, 3, 3, 3]), 0.0);
        assert_eq!(
            <Sparse3Body<f64> as ThreeBody<f64>>::entry(&w, &[0, 1, 2, 3, 1, 0]),
            2.5
        );
        assert_eq!(w.norb(), 4);
    }

    #[test]
    fn sparse3body_set_if_absent_first_write_wins() {
        let mut w = Sparse3Body::<f64>::zeros(3);
        assert!(w.set_if_absent(&[0, 0, 1, 1, 2, 2], 1.0));
        assert!(!w.set_if_absent(&[0, 0, 1, 1, 2, 2], 9.0));
        assert_eq!(w.get(&[0, 0, 1, 1, 2, 2]), 1.0);
        w.set(&[0, 0, 1, 1, 2, 2], 3.0);
        assert_eq!(w.get(&[0, 0, 1, 1, 2, 2]), 3.0);
    }

    #[test]
    fn sparse3body_nnz_scale_support() {
        let mut w = Sparse3Body::<f64>::zeros(5);
        w.set(&[0, 1, 2, 3, 4, 0], 2.0);
        w.set(&[1, 1, 1, 1, 1, 1], -4.0);
        assert_eq!(w.nnz(), 2);
        w.scale_all(-0.5);
        assert_eq!(w.get(&[0, 1, 2, 3, 4, 0]), -1.0);
        assert_eq!(w.get(&[1, 1, 1, 1, 1, 1]), 2.0);
        let support: HashSet<[usize; 6]> = w.support().map(|(k, _)| k).collect();
        assert_eq!(support.len(), 2);
        assert!(support.contains(&[0, 1, 2, 3, 4, 0]));
        assert!(support.contains(&[1, 1, 1, 1, 1, 1]));
    }

    #[test]
    fn sparse3body_from_dense_roundtrip_and_support_crosscheck() {
        let d = 3usize;
        let mut dense = CoeffTensor::<f64>::zeros(d, 6);
        dense.set(&[0, 0, 0, 0, 0, 0], 1.0);
        dense.set(&[1, 2, 0, 1, 2, 0], 1e-4);
        dense.set(&[2, 2, 2, 2, 2, 2], 1e-12);
        let sparse = Sparse3Body::from_dense(&dense);
        assert_eq!(sparse.nnz(), 3);
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        for m in 0..d {
                            for n in 0..d {
                                let idx = [i, j, k, l, m, n];
                                assert_eq!(sparse.get(&idx), dense.get(&idx));
                            }
                        }
                    }
                }
            }
        }
        let tau = 1e-8;
        let dset: HashSet<[usize; 6]> = ThreeBody::support_above(&dense, tau).into_iter().collect();
        let sset: HashSet<[usize; 6]> =
            ThreeBody::support_above(&sparse, tau).into_iter().collect();
        assert_eq!(dset, sset);
        assert_eq!(dset.len(), 2);
        assert!(dset.contains(&[0, 0, 0, 0, 0, 0]));
        assert!(dset.contains(&[1, 2, 0, 1, 2, 0]));
    }
}
