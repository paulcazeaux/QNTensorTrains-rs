// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Spin algebra: exact-rational Clebsch–Gordan generator, brute-force 6-CG recoupling factor,
//! fixed single-site fermionic algebra, the Wigner–Eckart site-operator catalogue, sparse-COO
//! dense reference builders, and the runtime least-squares channel-weight fit.

mod cg;
mod channels;
mod compose;
mod couple;
mod fit;
mod helpers;
mod recouple;
mod siteops;

#[cfg(test)]
mod compose_tests;

pub use compose::ComposeTable;
pub use siteops::{
    accumulate_term, dense_term, elementary_tensor, kron_accum, occ_count, site_op_matrix,
    tensor_a, tensor_cdag, tensor_identity,
};

use std::rc::Rc;

use num_bigint::BigInt;
use rustc_hash::FxHashMap;

/// Basis-index alias: `0=|0⟩, 1=|↑⟩, 2=|↓⟩, 3=|↑↓⟩`
/// (`|↑↓⟩ = a†↑a†↓|0⟩`). Matrix convention `m[dout][din]` (row = output determinant, col = input).
pub type Mat4 = [[f64; 4]; 4];

/// One irreducible-rank site tensor. `components` has length `two_k+1`, indexed **μ-descending**
/// (`components[0]` ↔ `twoμ = two_k`, …, `components[two_k]` ↔ `twoμ = -two_k`).
/// `flux` = Δn at this site; `odd` = fermionic parity.
#[derive(Clone, Debug)]
pub struct TensorOp {
    pub two_k: i32,
    pub flux: i32,
    pub odd: bool,
    pub components: Vec<Mat4>,
}

/// One irreducible-rank channel of a term's operator content at one site. `red[occ_x][occ_y]` is the 3×3 Wigner–Eckart reduced element over
/// occupations `0=|0⟩, 1=|1e⟩, 2=|2e⟩`.
#[derive(Clone, Debug, PartialEq)]
pub struct SiteOp {
    pub two_k: i32,
    pub flux: i32,
    pub odd: bool,
    pub red: [[f64; 3]; 3],
}

/// Interned index into [`SpinAlgebra::siteops`] — identity-shared `SiteOp`
/// instances: two `site_channel_ops` calls on equal token groups yield
/// the SAME ids, so downstream site-op dedup is `id == other_id`.
pub type SiteOpId = usize;

/// One spin-coupled chain channel of a term. `ranks` is the operator-bond rank path (length `q+1`,
/// `ranks[0] = ranks[q] = 0`); `siteops` is one [`SiteOpId`] per DISTINCT site, in site order;
/// `weight` is the least-squares-fitted channel weight.
#[derive(Clone, Debug)]
pub struct TermChannel {
    pub ranks: Vec<i32>,
    pub siteops: Vec<SiteOpId>,
    pub weight: f64,
}

/// Sparse COO dense-reference accumulator: `(dout, din) -> amplitude`, 0-based determinant indices.
/// A `BTreeMap` (not `FxHashMap`) so `.keys()` iterates in a FIXED ascending order — the
/// deterministic-row-assembly hardening of the channel-fit's `rows` table.
pub type Coo = std::collections::BTreeMap<(usize, usize), f64>;

/// Canonical (position/d-independent) pattern key for the `term_channels` memoization:
/// the token sequence with sites remapped to `0..q` ordinal positions,
/// role order preserved (NEVER sorted — sorting collides pair-role-swapped terms).
pub type PatternKey = Vec<(usize, bool)>;

/// Per-construction cache/interning context, in place of four global dictionaries
/// (`cg_cache`/`recouple_cache`/`siteops_cache`/`channel_cache`).
/// Threaded `&mut` through Hamiltonian assembly; **no global mutable state** — the abelian
/// `Automaton` interner precedent.
#[allow(dead_code)]
#[derive(Default)]
pub struct SpinAlgebra {
    cg_cache: FxHashMap<[i32; 6], f64>,
    recouple_cache: FxHashMap<[i32; 9], f64>,
    fact_cache: Vec<BigInt>,
    siteops: Vec<SiteOp>,
    siteops_cache: FxHashMap<u64, Rc<Vec<SiteOpId>>>,
    channel_cache: FxHashMap<PatternKey, Rc<Vec<TermChannel>>>,
}

#[allow(dead_code)]
impl SpinAlgebra {
    /// Fresh, empty caches.
    pub fn new() -> Self {
        Self::default()
    }

    /// Read the interned [`SiteOp`] pool (for the Hamiltonian assembler, which stores
    /// [`SiteOpId`]s and resolves them against this pool at the COO fill).
    pub fn siteop(&self, id: SiteOpId) -> &SiteOp {
        &self.siteops[id]
    }
}
