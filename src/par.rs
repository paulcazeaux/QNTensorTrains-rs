// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Threading policy for the solver-layer **vector algebra** (`dot`/`norm`/`axpy`/`scale` on
//! cores, contracted core pairs and tangent vectors) — the O(m²·n) basis work that dominates the
//! RPA/Krylov drivers once the tangent dimension passes ~10⁴: the parallel `H`-applies are a
//! minority of the wall, and everything between them — two-pass MGS, the m×m projected matrix, the
//! restart recombination — runs serial unless the knobs below fire.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

pub(crate) static VECTOR_PAR: AtomicBool = AtomicBool::new(true);

/// Read the vector-algebra parallelism switch (see [`VECTOR_PAR`]).
#[inline]
pub fn vector_par() -> bool {
    VECTOR_PAR.load(Ordering::Relaxed)
}

/// Set the vector-algebra parallelism switch (see [`VECTOR_PAR`]); returns the previous value.
pub fn set_vector_par(on: bool) -> bool {
    VECTOR_PAR.swap(on, Ordering::Relaxed)
}

pub(crate) static BLOCK_SLICE: AtomicBool = AtomicBool::new(true);

/// Read the block-kernel inner-loop switch (see [`BLOCK_SLICE`]).
#[inline]
pub fn block_slice() -> bool {
    BLOCK_SLICE.load(Ordering::Relaxed)
}

/// Set the block-kernel inner-loop switch (see [`BLOCK_SLICE`]); returns the previous value.
pub fn set_block_slice(on: bool) -> bool {
    BLOCK_SLICE.swap(on, Ordering::Relaxed)
}

pub(crate) static PAR_MIN_COEFFS: AtomicUsize = AtomicUsize::new(1 << 19);

/// Read the per-op size floor (see [`PAR_MIN_COEFFS`]).
#[inline]
pub fn par_min_coeffs() -> usize {
    PAR_MIN_COEFFS.load(Ordering::Relaxed)
}

/// Set the per-op size floor (see [`PAR_MIN_COEFFS`]); returns the previous value.
pub fn set_par_min_coeffs(v: usize) -> usize {
    PAR_MIN_COEFFS.swap(v, Ordering::Relaxed)
}

pub(crate) static PAR_MIN_FORK_WORK: AtomicUsize = AtomicUsize::new(4096);

/// Read the driver-fork work floor (see [`PAR_MIN_FORK_WORK`]).
#[inline]
pub fn par_min_fork_work() -> usize {
    PAR_MIN_FORK_WORK.load(Ordering::Relaxed)
}

/// Set the driver-fork work floor (see [`PAR_MIN_FORK_WORK`]); returns the previous value.
pub fn set_par_min_fork_work(v: usize) -> usize {
    PAR_MIN_FORK_WORK.swap(v, Ordering::Relaxed)
}

thread_local! {
    static NEST_DEPTH: Cell<u32> = const { Cell::new(0) };
}

pub(crate) struct NestGuard(());

impl NestGuard {
    #[inline]
    pub(crate) fn new() -> Self {
        NEST_DEPTH.with(|c| c.set(c.get() + 1));
        NestGuard(())
    }
}

impl Drop for NestGuard {
    #[inline]
    fn drop(&mut self) {
        NEST_DEPTH.with(|c| c.set(c.get() - 1));
    }
}

/// Run `f` with this thread marked as already inside a driver-level parallel task: the vector
/// ops `f` performs take their serial paths (see the module docs). Purely a scheduling hint —
/// results are unaffected either way.
#[inline]
pub fn nested_scope<R>(f: impl FnOnce() -> R) -> R {
    let _guard = NestGuard::new();
    f()
}

#[inline]
pub(crate) fn par_vector_ops(coeffs: usize) -> bool {
    coeffs >= par_min_coeffs() && par_enabled()
}

#[inline]
pub(crate) fn par_driver_fork(total_work: usize) -> bool {
    total_work >= par_min_fork_work() && par_enabled()
}

#[inline]
fn par_enabled() -> bool {
    NEST_DEPTH.with(|c| c.get()) == 0 && vector_par() && rayon::current_num_threads() > 1
}

#[cfg(test)]
pub(crate) static PAR_SWITCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn lock_par_switches() -> std::sync::MutexGuard<'static, ()> {
    PAR_SWITCH_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nest_guard_suppresses_and_restores() {
        let _lock = lock_par_switches();
        let big = par_min_coeffs().saturating_mul(4).max(1 << 22);
        let outside = par_vector_ops(big);
        assert_eq!(outside, vector_par() && rayon::current_num_threads() > 1);
        {
            let _g = NestGuard::new();
            assert!(!par_vector_ops(big), "nested tasks must run serial");
            assert!(!par_driver_fork(big), "nested driver forks must run serial");
            {
                let _g2 = NestGuard::new();
                assert!(!par_vector_ops(big));
            }
            assert!(!par_vector_ops(big), "inner drop must not clear the outer");
        }
        assert_eq!(par_vector_ops(big), outside, "guard drop must restore");
    }

    #[test]
    fn size_floor_keeps_small_sweeps_serial() {
        let _lock = lock_par_switches();
        assert!(!par_vector_ops(0));
        assert!(!par_driver_fork(0));
        assert!(par_min_fork_work() < par_min_coeffs());
    }

    #[test]
    fn switch_round_trips() {
        let _lock = lock_par_switches();
        let prev = set_vector_par(true);
        assert!(vector_par());
        assert!(set_vector_par(prev));
    }
}
