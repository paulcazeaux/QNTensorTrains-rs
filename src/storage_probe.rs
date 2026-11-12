// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Feature-gated census of the dense label-grid storage, both layers — the instrument of the
//! core-storage campaign (before/after measurements).

/// The label table a counted core or grid belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Table {
    /// `crate::qn::SectorGrid` / `SparseCore`.
    Abelian = 0,
    /// `crate::su2::qn::Su2SectorGrid` / `Su2Core<_, StateSym>`.
    State = 1,
    /// The SU(2) operator label table.
    Op = 2,
}

pub const N_TABLES: usize = 3;

impl Table {
    /// The SU(2) table of a core with `steps` block families: 4 is the state table, any other
    /// count (16) the operator table.
    #[inline(always)]
    pub fn su2(steps: usize) -> Table {
        if steps == 4 { Table::State } else { Table::Op }
    }

    pub fn name(self) -> &'static str {
        ["abelian", "state", "op"][self as usize]
    }
}

/// How a counted core was made: assembled slot by slot (`zeros`, `from_fn`), or handed its block
/// grids (`from_blocks_sym`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum CoreKind {
    Assembled = 0,
    FromBlocks = 1,
}

/// Element sizes above this share the last bucket.
pub const MAX_VSIZE: usize = 256;
/// Site indices at or above `MAX_SITE` share the last bucket.
pub const MAX_SITE: usize = 128;

/// Per-table core counts of one [`CoreKind`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoreCounts {
    pub cores: u64,
    /// Σ dense block-grid slots (`steps × grid length`) — the `Option<Mat>` headers allocated.
    pub grid_slots: u64,
    /// Σ admissible blocks (`Some` slots).
    pub admissible: u64,
    /// Σ admissible blocks with `rows·cols > 0`.
    pub nonempty: u64,
    /// Σ NON-EMPTY blocks the assembly's `fill` supplied (assembled cores only; an empty one is
    /// never stored, and the non-empty constructors never ask for one — core-storage 1c).
    pub filled: u64,
    /// Σ `rows·cols` over the blocks.
    pub coeffs: u64,
    /// Σ blocks the core's block store holds (`crate::blockstore`; equals `nonempty`).
    pub stored: u64,
    /// Σ lengths of the stores' dense key indexes (4 B each; 0 for a hash index).
    pub index_keys: u64,
}

/// Per-(table, element size) grid counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GridCounts {
    pub builds: u64,
    pub build_slots: u64,
    pub clones: u64,
    pub clone_slots: u64,
}

#[derive(Clone, Debug)]
struct GridSite {
    table: Table,
    vsize: usize,
    site: String,
    scope: String,
    counts: GridCounts,
}

impl GridSite {
    fn bytes(&self) -> u64 {
        (self.counts.build_slots + self.counts.clone_slots) * self.vsize as u64
    }
}

/// A snapshot of the census.
#[derive(Clone, Debug)]
pub struct Report {
    /// `[kind][table]`.
    pub cores: [[CoreCounts; N_TABLES]; 2],
    /// `[table][site]`: cores of either kind built for site `k`.
    pub sites: [[u64; MAX_SITE]; N_TABLES],
    /// `[table][size_of::<V>()]`.
    pub grids: [[GridCounts; MAX_VSIZE + 1]; N_TABLES],
    grid_sites: Vec<GridSite>,
}

impl Default for Report {
    fn default() -> Self {
        Report {
            cores: [[CoreCounts::default(); N_TABLES]; 2],
            sites: [[0; MAX_SITE]; N_TABLES],
            grids: [[GridCounts::default(); MAX_VSIZE + 1]; N_TABLES],
            grid_sites: Vec::new(),
        }
    }
}

impl Report {
    /// Dense grid bytes allocated by builds and clones of `table` (slots × element size).
    pub fn grid_bytes(&self, table: Table) -> u64 {
        self.grids[table as usize]
            .iter()
            .enumerate()
            .map(|(v, g)| (g.build_slots + g.clone_slots) * v as u64)
            .sum()
    }

    /// `CS:` lines, one block per table with anything counted: cores by kind, cores per site,
    /// then grids by element size (`v<bytes>`: builds/slots, clones/slots).
    pub fn format(&self, tag: &str) -> String {
        let mut out = String::new();
        for table in [Table::Abelian, Table::State, Table::Op] {
            let t = table as usize;
            let any_core = self.cores.iter().any(|k| k[t].cores > 0);
            let any_grid = self.grids[t].iter().any(|g| g.builds + g.clones > 0);
            if !any_core && !any_grid {
                continue;
            }
            for (kind, name) in [(CoreKind::Assembled, "asm"), (CoreKind::FromBlocks, "fb")] {
                let c = &self.cores[kind as usize][t];
                if c.cores == 0 {
                    continue;
                }
                out.push_str(&format!(
                    "CS: tag={tag} table={} {name} cores={} grid_slots={} admissible={} \
                     nonempty={} filled={} coeffs={} stored={} index_keys={}\n",
                    table.name(),
                    c.cores,
                    c.grid_slots,
                    c.admissible,
                    c.nonempty,
                    c.filled,
                    c.coeffs,
                    c.stored,
                    c.index_keys
                ));
            }
            let sites: Vec<String> = self.sites[t]
                .iter()
                .enumerate()
                .filter(|&(_, &n)| n > 0)
                .map(|(k, n)| format!("{k}:{n}"))
                .collect();
            if !sites.is_empty() {
                out.push_str(&format!(
                    "CS: tag={tag} table={} sites {}\n",
                    table.name(),
                    sites.join(",")
                ));
            }
            let grids: Vec<String> = self.grids[t]
                .iter()
                .enumerate()
                .filter(|(_, g)| g.builds + g.clones > 0)
                .map(|(v, g)| {
                    format!(
                        "v{v}={}/{}+{}/{}",
                        g.builds, g.build_slots, g.clones, g.clone_slots
                    )
                })
                .collect();
            if !grids.is_empty() {
                out.push_str(&format!(
                    "CS: tag={tag} table={} grids(builds/slots+clones/slots) {} dense_bytes={}\n",
                    table.name(),
                    grids.join(" "),
                    self.grid_bytes(table)
                ));
            }
            let mut sites: Vec<&GridSite> = self
                .grid_sites
                .iter()
                .filter(|s| s.table == table)
                .collect();
            sites.sort_by_key(|s| std::cmp::Reverse(s.bytes()));
            for s in sites {
                let g = &s.counts;
                out.push_str(&format!(
                    "CSS: tag={tag} table={} v{} {}/{}+{}/{} bytes={} site={} scope={}\n",
                    table.name(),
                    s.vsize,
                    g.builds,
                    g.build_slots,
                    g.clones,
                    g.clone_slots,
                    s.bytes(),
                    s.site,
                    if s.scope.is_empty() { "-" } else { &s.scope }
                ));
            }
        }
        out
    }
}

mod imp {
    use super::{CoreKind, Report, Table};

    /// Zero-sized no-op tally: every method compiles away.
    #[derive(Default)]
    pub struct CoreTally;

    impl CoreTally {
        #[inline(always)]
        pub fn new() -> Self {
            CoreTally
        }
        #[inline(always)]
        pub fn block(&mut self, _rows: usize, _cols: usize) {}
        #[inline(always)]
        pub fn filled(&mut self) {}
        #[inline(always)]
        pub fn finish(
            self,
            _kind: CoreKind,
            _table: Table,
            _k: usize,
            _grid_slots: usize,
            _stored: usize,
            _index_keys: usize,
        ) {
        }
    }

    #[inline(always)]
    pub fn grid_built(_table: Table, _vsize: usize, _slots: usize) {}
    #[inline(always)]
    pub fn grid_cloned(_table: Table, _vsize: usize, _slots: usize) {}
    pub fn reset() {}
    pub fn snapshot() -> Report {
        Report::default()
    }
    pub const ENABLED: bool = false;
}

pub use imp::{CoreTally, ENABLED, grid_built, grid_cloned, reset, snapshot};
