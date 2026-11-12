// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::Family;
use crate::orbital::Orbital;
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, Sector, SectorGrid};
use crate::scalar::Scalar;

use super::super::csx::CooCell;

/// Per-spin operator-state shift `(flux, n_left, n_right)`. Signed small ints; derives
/// lexicographic `Ord` for the Phase-B flatten sort.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct SpinShift {
    pub flux: i8,
    pub nl: i8,
    pub nr: i8,
}

/// Two-spin operator state plus the accumulated irrep flux.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct StateShift {
    pub up: SpinShift,
    pub dn: SpinShift,
    pub delta: Irrep,
}

/// An operator channel's flux across a bond: the per-spin particle flux plus the accumulated
/// irrep flux `Δ`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Flux {
    pub up: i8,
    pub dn: i8,
    pub irrep: Irrep,
}

/// The ⦿-padded open-orbital list (`None` is the ⦿ placeholder). Derived `Ord` puts `None < Some`,
/// giving the ⦿-sorts-first flatten order with zero custom `impl`.
pub type OpenIdx = [Option<Orbital>; 6];

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct Label {
    pub(super) bond: usize,
    pub(super) sector: Sector,
    pub(super) state: StateShift,
    pub(super) open: OpenIdx,
}

pub(super) struct EdgeData<T> {
    pub(super) m: Sector,
    pub(super) x_occ: Family,
    pub(super) alpha: T,
    pub(super) src: u32,
}

pub(crate) const SRC_NONE: u32 = u32::MAX;
pub(crate) const SRC_NEG: u32 = 1 << 31;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Op {
    Id,
    A,
    Adag,
    N,
}

/// One populated COO cell of a site's nest, keyed by its x-side coordinates.
#[derive(Clone, Debug)]
pub struct CooEntry<T: Scalar> {
    /// `x_occ as u8` (the x-transition [`Family`]).
    pub x_occ: u8,
    /// Position of the x-row sector in `QnSpace::bond_sectors(k)`.
    pub m_ord: u32,
    pub cell: CooCell<T>,
    /// Per-value provenance, parallel to `cell.vals` when recorded (a refillable build), else
    /// empty. Each word is [`SRC_NONE`] or `term_ordinal | (SRC_NEG if the JW sign is −)`.
    pub src: Vec<u32>,
}

/// COO block storage for one site: a **populated-cell list** with a dense outer index.
pub struct CooNest<T: Scalar> {
    pub(super) offsets: Vec<u32>,
    pub(super) cells: Vec<CooEntry<T>>,
    pub(super) geom: GridGeom,
    pub(super) sectors: Vec<Sector>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct GridGeom {
    nup1: usize,
    ndn1: usize,
    nir: usize,
}

impl GridGeom {
    pub(super) fn of(space: &QnSpace) -> Self {
        GridGeom {
            nup1: space.nup + 1,
            ndn1: space.ndn + 1,
            nir: space.pg.order() as usize,
        }
    }

    #[inline]
    pub(super) fn nsec(&self) -> usize {
        self.nir * self.nup1 * self.ndn1
    }

    #[inline]
    pub(super) fn idx(&self, s: Sector) -> usize {
        debug_assert!(
            s.nup < self.nup1 && s.ndn < self.ndn1 && (s.irrep.index() as usize) < self.nir,
            "sector {s:?} out of grid"
        );
        (s.irrep.index() as usize) * self.nup1 * self.ndn1 + s.nup * self.ndn1 + s.ndn
    }

    #[inline]
    fn sector(&self, g: usize) -> Sector {
        let plane = self.nup1 * self.ndn1;
        let (ir, p) = (g / plane, g % plane);
        Sector::with_irrep(p / self.ndn1, p % self.ndn1, Irrep::new(ir as u8))
    }
}

impl<T: Scalar> CooNest<T> {
    /// The populated cells of the outer slot `(y_occ, n1)`, in kernel visitation order.
    #[inline]
    pub fn slot(&self, y_occ: Family, n1: Sector) -> &[CooEntry<T>] {
        let o = (y_occ as usize) * self.geom.nsec() + self.geom.idx(n1);
        let lo = self.offsets[o] as usize;
        let hi = self.offsets[o + 1] as usize;
        &self.cells[lo..hi]
    }

    /// `QnSpace::bond_sectors(k)` for this site — `m_ord` indexes it.
    #[inline]
    pub fn sectors(&self) -> &[Sector] {
        &self.sectors
    }

    /// Total populated cells — the nest-storage census surface.
    pub fn populated(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub(crate) fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub(crate) fn into_cells(self) -> impl Iterator<Item = CooEntry<T>> {
        self.cells.into_iter()
    }

    /// Every populated cell as `(y_occ, n1, x_occ, m, &cell)` in **`SectorGrid` order** on both
    /// sector axes (`irrep` slowest, `ndn` fastest) rather than storage order.
    pub fn iter_grid_order(&self) -> Vec<(usize, Sector, usize, Sector, &CooCell<T>)> {
        let nsec = self.geom.nsec();
        let mut out = Vec::with_capacity(self.cells.len());
        let mut buf: Vec<(usize, usize, usize)> = Vec::new();
        for y_occ in 0..4usize {
            for n1g in 0..nsec {
                let o = y_occ * nsec + n1g;
                let (lo, hi) = (self.offsets[o] as usize, self.offsets[o + 1] as usize);
                if lo == hi {
                    continue;
                }
                let n1 = self.geom.sector(n1g);
                buf.clear();
                for i in lo..hi {
                    let e = &self.cells[i];
                    buf.push((
                        e.x_occ as usize,
                        self.geom.idx(self.sectors[e.m_ord as usize]),
                        i,
                    ));
                }
                buf.sort_unstable();
                for &(xf, _, i) in &buf {
                    let e = &self.cells[i];
                    out.push((y_occ, n1, xf, self.sectors[e.m_ord as usize], &e.cell));
                }
            }
        }
        out
    }
}

/// Construction algorithm selector (design record `docs/bipartite-design.md` §0): the
/// median-threaded automaton and the bipartite/vertex-cover sweep produce the *same*
/// exact-integer operator with different (equally valid) channel layouts. Both stay
/// first-class and user-selectable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Construction {
    /// The original median-bond path-threading automaton (`run_term`/`set_edges`/`finalize`) —
    /// the t7b bit-identity layout certificates are pinned to this arm.
    Median,
    /// The streaming per-bond bipartite/vertex-cover sweep ([`crate::hamiltonian::bipartite`]),
    /// settling the cover once per **flux** (union over sectors) with reachability read off the
    /// instantiation map — the shipped bipartite algorithm. Certified by the dense/cross oracles,
    /// the N2 energy gate, and the bit-identity suite in the `operators` binary; the matvec
    /// matches `BipartitePerSector` to machine precision.
    Bipartite,
    /// The per-`(sector, flux)` cover (the original bipartite graft) — a tighter per-sector layout
    /// (sector-summed bond dim ~3% below [`Self::Bipartite`]) at ~4–5× the construction wall. Kept
    /// first-class and selectable as the certified reference; not the default.
    BipartitePerSector,
    /// **The reconciliation arm** (`docs/median-bipartite-reconciliation.md`): the *median*
    /// machinery — same vertex labels `(StateShift, OpenIdx)`, same `set_edges`, same `finalize` —
    /// driven by the *bipartite cover's* weighted edge instead of the positional median. Isolates
    /// the two axes on which the two constructions actually differ: switch placement (this arm
    /// changes it) and channel-identity granularity (this arm keeps the median's, which refines
    /// the bipartite key by `(nl, nr)`). Diagnostic/first-class-selectable; not a production
    /// default.
    MedianCoverSwitch,
    /// The complementary reconciliation arm: the *bipartite* machinery — same channel keys, same
    /// Comp merging, same reachability instantiation — with the *positional median* weighted edge
    /// substituted for the cover's. With [`Construction::MedianCoverSwitch`] it completes the 2×2
    /// (switch placement × channel-identity granularity) that decomposes where the bipartite
    /// construction's compression actually comes from. Diagnostic; not a production default.
    BipartiteMedianSwitch,
}

/// The construction [`crate::hamiltonian::SparseHamiltonian::new`] delegates to.
/// Bipartite is the production default: vs Median it costs slightly more COO nnz and a one-time
/// construction wall, and buys a much lower construction peak RSS, a halved peak-site nnz and a
/// smaller sector-summed rank driver. Median stays first-class.
pub const DEFAULT_CONSTRUCTION: Construction = Construction::Bipartite;

/// One per-bond operator channel of the assembled artifact. External consumers read only the
/// list length (`automaton_bond_dims`) and the per-spin flux (`channel_fluxes`); the payload
/// shape is construction-specific (design §1.4).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum ChannelDesc {
    /// Median-path vertex: operator state shift + ⦿-padded open-orbital triple.
    Median(StateShift, OpenIdx),
    /// Bipartite-path channel: the packed total-order `(Flux, ChanKind, OpString)` key.
    Bipartite(u64),
}

impl ChannelDesc {
    /// The channel's flux — per-spin particle flux plus the accumulated irrep flux Δ, the piece
    /// the `state_qns` formula and the metric harness consume.
    pub fn flux(&self) -> Flux {
        match self {
            ChannelDesc::Median(st, _) => Flux {
                up: st.up.flux,
                dn: st.dn.flux,
                irrep: st.delta,
            },
            ChannelDesc::Bipartite(key) => crate::hamiltonian::bipartite::key_flux(*key),
        }
    }
}

/// The Phase-B/C output consumed by `SparseHamiltonian`.
pub struct Assembled<T: Scalar> {
    pub states: Vec<SectorGrid<Vec<ChannelDesc>>>,
    pub state_qns: Vec<SectorGrid<Vec<Sector>>>,
    pub coo: Vec<CooNest<T>>,
    /// Terms the Phase-A stream emitted — the provenance ordinal range of a refillable build.
    pub n_terms: usize,
}
