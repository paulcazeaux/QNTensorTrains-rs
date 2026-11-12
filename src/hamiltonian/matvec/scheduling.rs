// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::core::{Family, SparseCore};
use crate::hamiltonian::SparseHamiltonian;
use crate::qn::Sector;
use crate::scalar::Scalar;

pub(crate) const PAR_MIN_TASKS: usize = 8;

pub(crate) static LOCAL_MATVEC_APPLIES: AtomicU64 = AtomicU64::new(0);

/// Read the accumulated local-matvec apply count (see [`LOCAL_MATVEC_APPLIES`]).
pub fn local_matvec_applies() -> u64 {
    LOCAL_MATVEC_APPLIES.load(Ordering::Relaxed)
}
/// Reset the local-matvec apply counter to zero.
pub fn reset_local_matvec_applies() {
    LOCAL_MATVEC_APPLIES.store(0, Ordering::Relaxed);
}

pub(crate) static BLOCK_FORK: AtomicBool = AtomicBool::new(true);

/// Read the fork-axis switch (see [`BLOCK_FORK`]).
#[inline]
pub fn block_fork() -> bool {
    BLOCK_FORK.load(Ordering::Relaxed)
}
/// Set the fork-axis switch (see [`BLOCK_FORK`]); returns the previous value.
pub fn set_block_fork(on: bool) -> bool {
    BLOCK_FORK.swap(on, Ordering::Relaxed)
}

pub(crate) static BLOCK_SPLIT: AtomicBool = AtomicBool::new(true);

pub(crate) const BLOCK_SPLIT_TARGET: u64 = 32;

pub(crate) static ENV_PARTIALS_BYTES: AtomicU64 = AtomicU64::new(u64::MAX);
const ENV_PARTIALS_BYTES_DEFAULT: u64 = 1 << 30;

/// The current env-step partial-set budget (see [`ENV_PARTIALS_BYTES`]).
pub fn env_partials_bytes() -> u64 {
    use std::sync::LazyLock;
    static FROM_ENV: LazyLock<u64> = LazyLock::new(|| {
        std::env::var("QNTT_ENV_PARTIALS_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(ENV_PARTIALS_BYTES_DEFAULT)
    });
    let v = ENV_PARTIALS_BYTES.load(Ordering::Relaxed);
    if v == u64::MAX { *FROM_ENV } else { v }
}

/// Set the env-step partial-set budget (tests and instruments); returns the previous value.
pub fn set_env_partials_bytes(bytes: u64) -> u64 {
    ENV_PARTIALS_BYTES.swap(bytes, Ordering::Relaxed)
}

pub(crate) static BLOCK_SPLIT_EVENTS: AtomicU64 = AtomicU64::new(0);

/// Read [`BLOCK_SPLIT_EVENTS`].
pub fn block_split_events() -> u64 {
    BLOCK_SPLIT_EVENTS.load(Ordering::Relaxed)
}
/// Reset [`BLOCK_SPLIT_EVENTS`].
pub fn reset_block_split_events() {
    BLOCK_SPLIT_EVENTS.store(0, Ordering::Relaxed);
}

/// Read the block-splitting switch (see [`BLOCK_SPLIT`]).
#[inline]
pub fn block_split() -> bool {
    BLOCK_SPLIT.load(Ordering::Relaxed)
}
/// Set the block-splitting switch (see [`BLOCK_SPLIT`]); returns the previous value.
pub fn set_block_split(on: bool) -> bool {
    BLOCK_SPLIT.swap(on, Ordering::Relaxed)
}

pub(super) fn est_cost<T: Scalar>(
    h: &SparseHamiltonian<T>,
    k: usize,
    fl_rows: &crate::qn::RankGrid,
    x: &SparseCore<T>,
    sectors_k: &[Sector],
    fam: Family,
    l: Sector,
) -> u64 {
    let rows = fl_rows[l] as u64;
    let mut w = 0u64;
    for rec in h.nest_slot(k, fam, l) {
        let x_occ = Family::from_physical_index(rec.x_occ as usize);
        let m = sectors_k[rec.m_ord as usize];
        let nnz = rec.csr.nnz() as u64;
        let r_in = x.row_ranks[m] as u64;
        let r_out = x.col_ranks[x_occ.apply(m, h.space.pg.irrep_of(k))] as u64;
        let ncol = rec.csc.ids.len() as u64;
        w += nnz * r_in + ncol * r_in * r_out;
    }
    rows * w
}

#[allow(clippy::too_many_arguments)]
pub(super) fn est_cost_right<T: Scalar>(
    h: &SparseHamiltonian<T>,
    k: usize,
    f_cols: usize,
    x: &SparseCore<T>,
    sectors_k: &[Sector],
    fam: Family,
    n1: Sector,
) -> u64 {
    let mut w = 0u64;
    for rec in h.nest_slot(k, fam, n1) {
        let x_occ = Family::from_physical_index(rec.x_occ as usize);
        let m = sectors_k[rec.m_ord as usize];
        let nnz = rec.csr.nnz() as u64;
        let r_in = x.row_ranks[m] as u64;
        let r_out = x.col_ranks[x_occ.apply(m, h.space.pg.irrep_of(k))] as u64;
        let ncol = rec.csc.ids.len() as u64;
        w += nnz * r_in + ncol * r_in * r_out;
    }
    f_cols as u64 * w
}
