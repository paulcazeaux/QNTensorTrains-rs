// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::prelude::ReborrowMut;

use crate::scalar::Scalar;
use crate::su2::qn::Symmetry;

use super::Su2Core;

impl<T: Scalar, S: Symmetry> Su2Core<T, S> {
    /// Number of coefficients (sum of present-block sizes), canonical order.
    pub fn num_coeffs(&self) -> usize {
        let mut n = 0usize;
        for &fam in S::STEPS {
            for s in self.transitions(fam) {
                if let Some(b) = self.stored_at(fam, s) {
                    n += b.nrows() * b.ncols();
                }
            }
        }
        n
    }

    /// Visit every coefficient by value in the canonical order — the read-only twin of
    /// [`Self::for_each_coeff_mut`], walking exactly the blocks [`Self::num_coeffs`] counts and in
    /// the order [`Self::coeff`] indexes. `O(num_coeffs)`, unlike a `coeff(i)` loop, which re-walks
    /// every block per index; that is what makes it usable for serialization
    /// ([`crate::tt_io`]) at solver-scale ranks.
    pub fn for_each_coeff<F: FnMut(T)>(&self, mut f: F) {
        for &fam in S::STEPS {
            for s in self.transitions(fam) {
                if let Some(b) = self.stored_at(fam, s) {
                    let (m, n) = (b.nrows(), b.ncols());
                    for j in 0..n {
                        for i in 0..m {
                            f(*b.get(i, j));
                        }
                    }
                }
            }
        }
    }

    /// Visit every coefficient mutably in the canonical order (step outer, row label inner,
    /// column-major within a block).
    pub fn for_each_coeff_mut<F: FnMut(&mut T)>(&mut self, mut f: F) {
        for &fam in S::STEPS {
            let secs: Vec<S::Label> = self.transitions(fam).collect();
            for s in secs {
                let mut b = match self.stored_at_mut(fam, s) {
                    Some(b) => b,
                    None => continue,
                };
                let (m, n) = (b.nrows(), b.ncols());
                for j in 0..n {
                    for i in 0..m {
                        f(b.rb_mut().get_mut(i, j));
                    }
                }
            }
        }
    }

    /// Zip two same-shaped cores' coefficients in the canonical order: `f(&mut self_i, other_i)`.
    /// Precondition: `self`/`other` share the same rank grids (debug_assert-able by the caller); a
    /// slot stored in one core and empty in the other panics.
    pub fn zip_coeffs_mut<F: FnMut(&mut T, T)>(&mut self, other: &Self, mut f: F) {
        for &fam in S::STEPS {
            let secs: Vec<S::Label> = self.transitions(fam).collect();
            for s in secs {
                let (mut b, ob) = match (self.stored_at_mut(fam, s), other.stored_at(fam, s)) {
                    (Some(b), Some(ob)) => (b, ob),
                    (None, None) => continue,
                    _ => panic!("zip_coeffs_mut: shape — {fam:?} at {s:?} is empty in one core"),
                };
                let (m, n) = (b.nrows(), b.ncols());
                debug_assert_eq!((m, n), (ob.nrows(), ob.ncols()), "zip_coeffs_mut: shape");
                for j in 0..n {
                    for i in 0..m {
                        let ov = *ob.get(i, j);
                        f(b.rb_mut().get_mut(i, j), ov);
                    }
                }
            }
        }
    }

    /// Read the coefficient at a flat canonical index (`0..num_coeffs()`).
    pub fn coeff(&self, idx: usize) -> T {
        let mut off = 0usize;
        for &fam in S::STEPS {
            for s in self.transitions(fam) {
                if let Some(b) = self.stored_at(fam, s) {
                    let (m, n) = (b.nrows(), b.ncols());
                    let sz = m * n;
                    if idx < off + sz {
                        let local = idx - off;
                        return *b.get(local % m, local / m);
                    }
                    off += sz;
                }
            }
        }
        panic!("coeff: index {idx} out of range 0..{off}");
    }

    /// Write the coefficient at a flat canonical index.
    pub fn set_coeff(&mut self, idx: usize, val: T) {
        let mut off = 0usize;
        for &fam in S::STEPS {
            let secs: Vec<S::Label> = self.transitions(fam).collect();
            for s in secs {
                let (m, n) = match self.stored_at(fam, s) {
                    Some(b) => (b.nrows(), b.ncols()),
                    None => continue,
                };
                let sz = m * n;
                if idx < off + sz {
                    let local = idx - off;
                    *self
                        .stored_at_mut(fam, s)
                        .expect("stored block")
                        .get_mut(local % m, local / m) = val;
                    return;
                }
                off += sz;
            }
        }
        panic!("set_coeff: index {idx} out of range 0..{off}");
    }
}
