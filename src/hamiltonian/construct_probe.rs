// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Feature-gated stage timers + counters for the **construction** paths (`assembly::assemble_with`
//! and the `bipartite` sweep) — the twin of [`super::instrument`], which attributes the matvec.

/// One flat, non-overlapping cost bucket of a construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Stage {
    /// Phase-A term enumeration + screening (`enumerate_1body`/`2body`/`3body`).
    PhaseA = 0,
    /// `run_term`'s κ-sweep: `vertex` + the `set_edges` sector loop + interning.
    SetEdges = 1,
    /// `finalize`: per-`(bond, sector)` bucket sort of interned vertices.
    FinalizeSort = 2,
    /// `finalize`: the edge-key sort + COO fill.
    FinalizeCoo = 3,
    /// Step 1: per-ride edge build (U-key/V-key interning, `adm_mask`).
    GraphBuild = 4,
    /// Step 2: per-`(sector, flux)` admissibility blocking + δ-masks.
    Blocking = 5,
    /// Step 3 prep: U/V vertex sets (sort/dedup) + adjacency build.
    CoverPrep = 6,
    /// Step 3: Hopcroft–Karp matching + König cover.
    CoverHk = 7,
    /// Step 4: routing (channel interning, `emits` accumulation).
    Route = 8,
    /// Carrier update: forward each ride to its bond-`(k+1)` channel.
    Carrier = 9,
    /// Phase-B analogue: channel instantiation (sort + `pos` + `states`/`state_qns`).
    Instantiate = 10,
    /// Step 5: COO emission.
    CooEmit = 11,
    /// The switch-oracle post-pass (`cover_switch_bonds` reading deposit bonds off the sweep).
    SwitchScan = 12,
    /// `finalize`: the edge-key sort alone (split out of [`Stage::FinalizeCoo`], which then holds
    /// the COO fill).
    EdgeKeySort = 13,
    /// `HNest::from_coo` per site: the COO → CSR/CSC compression (`csx_pair`'s two sorts + the
    /// index/value gathers). Runs in `SparseHamiltonian::new_with` AFTER `assemble_with`, so it
    /// was outside every recorded construction total before this bucket existed.
    NestCompress = 14,
}

pub const N_STAGES: usize = 15;

/// A counted quantity (all monotone sums unless noted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Counter {
    /// Screened terms handed to the sink.
    Terms = 0,
    /// `set_edges` calls (median path) — i.e. (term, bond) pairs swept.
    SetEdgesCalls = 1,
    /// `Automaton::emit` calls (median path) — (term, bond, sector, δ) quadruples.
    Emits = 2,
    /// Interned vertices (median) / instantiated `(channel, sector)` pairs (bipartite).
    Vertices = 3,
    /// Interned automaton edges (median).
    Edges = 4,
    /// Rides processed, summed over sites (bipartite).
    Rides = 5,
    /// Graph edges built, summed over sites (bipartite).
    GraphEdges = 6,
    /// Cover subproblems solved (per flux, or per `(sector, flux)`).
    CoverBlocks = 7,
    /// Σ U vertices and Σ V vertices over all cover subproblems.
    CoverU = 8,
    CoverV = 9,
    /// `emits` map entries (bipartite routing) — the COO pre-image.
    EmitEntries = 10,
    /// COO triples pushed.
    CooTriples = 11,
    /// COO triples whose value is an integral coefficient `±γ` (the term's switch-bond deposit),
    /// as opposed to a rotation-invariant transport constant `±1`. Median: edges emitted by the
    /// `bond == switch` `set_edges` call; bipartite: triples of a `Normal → Comp` routing entry.
    /// The ratio to [`Counter::CooTriples`] sizes a coefficient-only refill.
    CooTriplesWeighted = 12,
    /// Routing entries (bipartite) / `set_edges` emits (median) carrying `±γ` — the pre-image of
    /// [`Counter::CooTriplesWeighted`].
    EmitEntriesWeighted = 13,
}

pub const N_COUNTERS: usize = 14;

/// A snapshot of the probe state: nanoseconds per [`Stage`], raw [`Counter`] sums, and the total
/// wall the caller measured around the whole construction (`0` if unset).
#[derive(Clone, Copy, Debug, Default)]
pub struct Report {
    pub nanos: [u64; N_STAGES],
    pub counts: [u64; N_COUNTERS],
    pub total_nanos: u64,
}

impl Report {
    pub fn stage_names() -> [&'static str; N_STAGES] {
        [
            "phase_a",
            "set_edges",
            "finalize_sort",
            "finalize_coo",
            "graph_build",
            "blocking",
            "cover_prep",
            "cover_hk",
            "route",
            "carrier",
            "instantiate",
            "coo_emit",
            "switch_scan",
            "edge_key_sort",
            "nest_compress",
        ]
    }

    pub fn counter_names() -> [&'static str; N_COUNTERS] {
        [
            "terms",
            "set_edges_calls",
            "emits",
            "vertices",
            "edges",
            "rides",
            "graph_edges",
            "cover_blocks",
            "cover_u",
            "cover_v",
            "emit_entries",
            "coo_triples",
            "coo_triples_weighted",
            "emit_entries_weighted",
        ]
    }

    /// Stage times made **exclusive**. Two nestings exist, both on the median path: the Phase-A
    /// enumeration *is* the driver of `run_term`, so [`Stage::SetEdges`] accumulates inside
    /// [`Stage::PhaseA`] (subtracting it leaves enumeration+screening alone); and the edge-key
    /// sort runs inside `finalize`'s COO span, so [`Stage::EdgeKeySort`] accumulates inside
    /// [`Stage::FinalizeCoo`] (subtracting it leaves the fill alone). Every other stage is
    /// already disjoint.
    pub fn exclusive(&self) -> [u64; N_STAGES] {
        let mut n = self.nanos;
        n[Stage::PhaseA as usize] =
            n[Stage::PhaseA as usize].saturating_sub(n[Stage::SetEdges as usize]);
        n[Stage::FinalizeCoo as usize] =
            n[Stage::FinalizeCoo as usize].saturating_sub(n[Stage::EdgeKeySort as usize]);
        n
    }

    /// Human-readable breakdown, stages sorted by cost. Percentages are of `total_nanos` when the
    /// caller supplied one, else of the attributed sum.
    pub fn format(&self) -> String {
        let excl = self.exclusive();
        let attributed: u64 = excl.iter().sum();
        let denom = if self.total_nanos > 0 {
            self.total_nanos
        } else {
            attributed.max(1)
        };
        let mut out = String::new();
        out.push_str(&format!(
            "CONSTRUCT_PROBE total={:.3}s attributed={:.3}s ({:.1}%)\n",
            denom as f64 / 1e9,
            attributed as f64 / 1e9,
            100.0 * attributed as f64 / denom as f64
        ));
        let names = Self::stage_names();
        let mut rows: Vec<(usize, u64)> = excl
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, n)| n > 0)
            .collect();
        rows.sort_unstable_by_key(|&(_, n)| std::cmp::Reverse(n));
        for (i, n) in rows {
            out.push_str(&format!(
                "  {:<14} {:>9.3}s  {:>5.1}%\n",
                names[i],
                n as f64 / 1e9,
                100.0 * n as f64 / denom as f64
            ));
        }
        if self.total_nanos > attributed {
            out.push_str(&format!(
                "  {:<14} {:>9.3}s  {:>5.1}%\n",
                "(unattributed)",
                (self.total_nanos - attributed) as f64 / 1e9,
                100.0 * (self.total_nanos - attributed) as f64 / denom as f64
            ));
        }
        let cnames = Self::counter_names();
        let counts: Vec<String> = self
            .counts
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, c)| c > 0)
            .map(|(i, c)| format!("{}={}", cnames[i], c))
            .collect();
        if !counts.is_empty() {
            out.push_str(&format!("  counts: {}\n", counts.join(" ")));
        }
        out
    }
}

mod imp {
    use super::{Counter, Report, Stage};

    /// Zero-sized no-op guard: `timer` compiles away entirely. The empty `Drop` impl is
    /// deliberate — call sites end a stage early with `drop(_t)`, which must type-check (and lint)
    /// identically whether or not the feature is on.
    pub struct Timer;

    impl Drop for Timer {
        fn drop(&mut self) {}
    }

    #[inline(always)]
    pub fn timer(_stage: Stage) -> Timer {
        Timer
    }
    #[inline(always)]
    pub fn bump(_c: Counter, _n: u64) {}
    #[inline(always)]
    pub fn set_total(_nanos: u64) {}
    pub fn reset() {}
    pub fn snapshot() -> Report {
        Report::default()
    }
    pub const ENABLED: bool = false;
}

pub use imp::{ENABLED, Timer, bump, reset, set_total, snapshot, timer};
