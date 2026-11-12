// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Fermionic orbital labels and the sign of reordering a creation/annihilation operator string into
//! canonical order.

/// Electron spin. Declaration order fixes the derived [`Ord`] (`Up < Dn`), which `Orbital`'s own
/// ordering (site first, then `Up < Dn`) is built on.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Spin {
    Up,
    Dn,
}

/// A single-particle orbital: 0-based lattice site + spin. Field declaration order (`site` then
/// `spin`) fixes the derived [`Ord`] to order by site first, then `Up` before `Dn` — the order
/// [`fermionic_sign`] counts inversions against.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Orbital {
    /// 0-based lattice site.
    pub site: usize,
    /// Spin.
    pub spin: Spin,
}

impl Orbital {
    /// Convenience constructor.
    #[inline]
    pub const fn new(site: usize, spin: Spin) -> Self {
        Orbital { site, spin }
    }
}

/// Fermionic sign of the permutation that sorts `ops` into ascending [`Orbital`] order —
/// strict-inversion-count parity: `for a<b { inv += (ops[a] > ops[b]) as
/// usize }`, sign = `inv & 1 == 1 ? -1 : 1`. Strict `>` (never `>=`) is load-bearing: equal
/// orbitals tie and add no inversion, matching stable-sort parity. Callers build the fixed operator
/// sequences verbatim — 1-body `[i,j]`, 2-body `[i,k,l,j]`, 3-body `[i,k,m,n,l,j]` — slice order is
/// load-bearing physics.
pub fn fermionic_sign(ops: &[Orbital]) -> i8 {
    let mut inv = 0usize;
    for a in 0..ops.len() {
        for b in (a + 1)..ops.len() {
            if ops[a] > ops[b] {
                inv += 1;
            }
        }
    }
    if inv & 1 == 1 { -1 } else { 1 }
}
