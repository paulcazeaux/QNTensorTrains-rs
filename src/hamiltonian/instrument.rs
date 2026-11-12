// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Feature-gated cost-attribution counters for the matvec kernels (`cell_apply_*`).

/// Snapshot of the accumulated counters (element counts, not bytes — multiply by
/// `size_of::<T>()` at report time; the kernels are generic over the scalar).
#[derive(Debug, Default, Clone, Copy)]
pub struct Counters {
    /// Total multiply-add flops (`2·m·n·k` per GEMM).
    pub flops: u64,
    /// Operator (`fblk`) elements read into GEMMs or gathers — the operator DRAM demand.
    pub op_elems: u64,
    /// Input (`xb`) elements read.
    pub x_elems: u64,
    /// Output (`yb`) elements written (GEMM `Add`/`Replace` and scatter `+=`).
    pub y_elems: u64,
    /// Intermediate (`fj`/`fix` panel) elements read+written in the gather/scatter zips.
    pub tmp_elems: u64,
    /// Number of GEMM calls (matmul dispatches).
    pub gemm_calls: u64,
    /// Number of `cell_apply_*` invocations.
    pub cell_calls: u64,
    /// Largest full operator block (`fblk`) elements seen — the operator working set reused across
    /// a cell's inner loop. Compared to cache sizes it tells whether the operator re-reads hit DRAM
    /// (block > L2/SLC) or stay resident (the cache-overlay that separates DRAM from cache traffic).
    pub max_fblk_elems: u64,
    /// Largest single intermediate buffer (`fj`/`fix`/`xfj`) elements — the per-cell gather panel.
    /// If this is ≤ L1, the large `tmp` movement is cache-resident, not DRAM.
    pub max_tmp_block: u64,
    /// Largest single output submatrix (`yb` block) elements written by one scatter/GEMM.
    pub max_out_block: u64,
}

mod imp {
    use super::Counters;
    #[inline(always)]
    pub fn gemm(_m: usize, _n: usize, _k: usize) {}
    #[inline(always)]
    pub fn footprint(_fblk_elems: usize, _tmp_block: usize) {}
    #[inline(always)]
    pub fn op_read(_elems: usize) {}
    #[inline(always)]
    pub fn x_read(_elems: usize) {}
    #[inline(always)]
    pub fn y_write(_elems: usize) {}
    #[inline(always)]
    pub fn tmp(_elems: usize) {}
    #[inline(always)]
    pub fn cell() {}
    #[inline(always)]
    pub fn panel(_m: usize, _n: usize, _k: usize) {}
    pub fn reset() {}
    pub fn snapshot() -> Counters {
        Counters::default()
    }
}

pub(crate) use imp::panel;
pub use imp::{cell, footprint, gemm, op_read, reset, snapshot, tmp, x_read, y_write};

/// Phase slots of the one-site sketched matvec. The serial slots are what cap the speedup:
/// Amdahl's non-scaling fraction is `(out_alloc + tasklist + merge) / total` plus whatever of the
/// parallel region fails to scale (tail imbalance + rayon overhead, exposed by `task_sum`/`task_max`).
#[derive(Debug, Default, Clone, Copy)]
pub struct Phases {
    /// SERIAL: `SparseCore::zeros` for the output core — allocation + zeroing of every block.
    pub out_alloc_ns: u64,
    /// SERIAL: building the `(q, fam, l)` task list.
    pub tasklist_ns: u64,
    /// Wall time of the whole rayon region (fork + tasks + join).
    pub parallel_ns: u64,
    /// SERIAL: the collect-then-write merge back into the output core.
    pub merge_ns: u64,
    /// Sum over tasks of their own wall time — the useful work. `task_sum / parallel` is the
    /// achieved occupancy of the parallel region.
    pub task_sum_ns: u64,
    /// Longest single task — the critical path. `task_max · P / task_sum` is the achieved
    /// imbalance factor (1.0 = perfect).
    pub task_max_ns: u64,
    /// Number of tasks (live output blocks).
    pub tasks: u64,
    /// Within a task, SERIAL-per-task: zeroing the `(rows × rq)` band buffer. The band-expanded
    /// `rq` makes this a large bandwidth-bound memset (the known `Mat::zeros` churn lever).
    pub blk_zero_ns: u64,
    /// Within a task: the actual `sketch_h_mul_block_into` cell work.
    pub mul_ns: u64,
    /// Within a task: the final `band · Fᴿ` GEMM (including its own buffer alloc).
    pub gemm_ns: u64,
    /// Elements zeroed across all band buffers — pairs with `blk_zero_ns` to give a memset rate.
    pub blk_zero_elems: u64,
    /// Wall of the SERIAL arm of `sketch_h_mul_into`/`h_sketch_mul_into` (`tasks < PAR_MIN_TASKS`).
    pub shm_serial_ns: u64,
    /// Wall of the PARALLEL arm of the same two primitives (fork + join + collect merge).
    pub shm_par_ns: u64,
    /// Calls taking the serial arm.
    pub shm_serial_calls: u64,
    /// Calls taking the parallel arm.
    pub shm_par_calls: u64,
    /// The left closure (`Fᴸ·H·Xₖ`) of the two-site matvec's `rayon::join` — arm asymmetry at
    /// 16 threads is the flatten-the-join gate.
    pub ts_left_ns: u64,
    /// The right closure (`H·Xₖ₊₁·Fᴿ`) of the same join.
    pub ts_right_ns: u64,
    /// Thread-summed wall per `cell_apply_*` branch, indexed `[CSC_L, CSR_L, COO_L, CSC_R, CSR_R,
    /// COO_R]`. NOTE the left/right kernels swap gather/scatter roles: left CSC gathers then GEMMs,
    /// left CSR GEMMs then scatters; the right kernel is reversed.
    pub cell_ns: [u64; 6],
    /// Invocation count per `cell_apply_*` branch, same indexing as `cell_ns`.
    pub cell_counts: [u64; 6],
    /// Σ over closed [`Region`]s of that region's max task wall — the critical-path lower bound on
    /// total parallel wall. `sum_region_max · P / task_sum` is the per-region imbalance factor,
    /// which the global `task_max` cannot give (it maxes across all regions/applies).
    pub sum_region_max_ns: u64,
    /// Number of [`Region`]s closed (parallel regions — one per apply for the one-site kernel).
    pub regions: u64,
    /// Wall per MALS sweep stage, indexed by [`stage`] (`CONTRACT`, `DAVIDSON`, `SPLIT`, `ENV`).
    pub stage_ns: [u64; stage::N],
    /// Process RSS high-water (bytes) sampled at the END of each MALS stage, indexed by [`stage`].
    /// Linux only (`/proc/self/status` VmRSS); zero elsewhere. Which stage carries the peak is
    /// the question this answers; the peak's size is the RSS reading.
    pub stage_rss_hw: [u64; stage::N],
    /// Largest transient logical byte count seen at each [`memslot`] site: the two wide two-site
    /// halves (+ the QC factors) at the apply, the env-step partial set, the noise half-matvec.
    pub mem_hw: [u64; memslot::N],
}

/// MALS sweep stages for [`stage_end`] / `Phases::stage_ns` / `Phases::stage_rss_hw`.
pub mod stage {
    pub const CONTRACT: usize = 0;
    pub const DAVIDSON: usize = 1;
    pub const SPLIT: usize = 2;
    pub const ENV: usize = 3;
    pub const N: usize = 4;
    pub const NAMES: [&str; N] = ["contract", "davidson", "split", "env_step"];
}

/// Transient-bytes high-water sites for [`mem_hw`] / `Phases::mem_hw`.
pub mod memslot {
    /// `l + r + xk + xk1` at the two-site apply (both wide halves live at once).
    pub const APPLY: usize = 0;
    /// Σ bytes of the collected partial set in the block-fork env step.
    pub const ENV_PARTIALS: usize = 1;
    /// The perturbative-noise half-matvec `half` at the split.
    pub const NOISE_HALF: usize = 2;
    pub const N: usize = 3;
    pub const NAMES: [&str; N] = ["apply(l+r+qc)", "env_partials", "noise_half"];
}

/// Whether instrumentation is compiled in; a `const`, so the timing calls fold away entirely
/// when off and the shipped/timed kernel keeps its byte-for-byte guarantee.
pub const ENABLED: bool = false;

/// Start a phase timer — `None` (and no clock read) when instrumentation is compiled out.
#[inline(always)]
pub fn tic() -> Option<std::time::Instant> {
    if ENABLED {
        Some(std::time::Instant::now())
    } else {
        None
    }
}

/// Phase slot ids for [`toc`]. Keep ids 0–6 stable so instrumented phase breakdowns stay
/// comparable across sessions.
pub mod slot {
    pub const OUT_ALLOC: usize = 0;
    pub const TASKLIST: usize = 1;
    pub const PARALLEL: usize = 2;
    pub const MERGE: usize = 3;
    pub const BLK_ZERO: usize = 4;
    pub const MUL: usize = 5;
    pub const GEMM: usize = 6;
    pub const SHM_SERIAL: usize = 7;
    pub const SHM_PAR: usize = 8;
    pub const TS_LEFT: usize = 9;
    pub const TS_RIGHT: usize = 10;
    pub const CELL_CSC_L: usize = 11;
    pub const CELL_CSR_L: usize = 12;
    pub const CELL_COO_L: usize = 13;
    pub const CELL_CSC_R: usize = 14;
    pub const CELL_CSR_R: usize = 15;
    pub const CELL_COO_R: usize = 16;
    /// Base of the `stage::N` MALS stage timers (`STAGE0 + stage::CONTRACT`, …).
    pub const STAGE0: usize = 17;
    pub const N: usize = STAGE0 + super::stage::N;
}

mod phase_imp {
    use super::Phases;
    #[inline(always)]
    pub fn toc(_s: usize, _t: Option<std::time::Instant>) {}
    #[inline(always)]
    pub fn task(_t: Option<std::time::Instant>, _blk_elems: usize) {}
    #[inline(always)]
    pub fn cell_done(_s: usize, _t: Option<std::time::Instant>) {}
    #[inline(always)]
    pub fn cell_timers() -> bool {
        false
    }
    #[inline(always)]
    pub fn shm_call(_par: bool) {}
    /// Zero-sized no-op stand-in — closures capturing `&Region` capture nothing real.
    pub struct Region;
    impl Region {
        pub fn begin() -> Region {
            Region
        }
        #[inline(always)]
        pub fn task(&self, _t: Option<std::time::Instant>, _blk_elems: usize) {}
        pub fn stats(&self) -> (u64, u64) {
            (0, 0)
        }
        pub fn end(self) {}
    }
    #[inline(always)]
    pub fn mem_hw(_which: usize, _bytes: u64) {}
    pub fn rss_bytes() -> u64 {
        0
    }
    #[inline(always)]
    pub fn stage_end(_stage: usize, _t: Option<std::time::Instant>) {}
    pub fn reset_phases() {}
    pub fn phases() -> Phases {
        Phases::default()
    }
}

pub use phase_imp::{
    Region, cell_done, cell_timers, mem_hw, phases, reset_phases, rss_bytes, shm_call, stage_end,
    task, toc,
};
