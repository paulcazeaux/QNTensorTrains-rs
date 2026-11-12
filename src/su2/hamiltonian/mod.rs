// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The SU(2) sparse Hamiltonian: term-enumeration constructor, `H·x` matvec, and the
//! Rayleigh-quotient / `xᵀHy` boundary contractions, plus the CSR/CSC freeze consumption side.

pub mod assembly;
pub(crate) mod channel;
pub mod matvec;

pub use assembly::{Su2PatternMask, Su2TermScreen};

use crate::coeff::{CoeffTensor, ThreeBody};
use crate::hamiltonian::csx::{CooCell, CsxBlock};
use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::su2::core::Family;
use crate::su2::coreops::adjoint_core_times_core;
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::spinalgebra::SpinAlgebra;
use crate::su2::vector::Su2TtVector;
use assembly::{SRC_NONE, Su2CellSrc, Su2Factor, Su2RunSrc};
use rayon::prelude::*;

/// The spin-free reduced chain-state vertex label (`StateT`).
/// `flux`/`mask`/`twok`/`delta`/`chanid` together identify a vertex at a bond; see the module docs
/// for the field-by-field abelian contrast.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct StateT {
    /// Scalar total particle flux `Δn` accumulated left of the bond (`+1` per creation, `-1`
    /// per annihilation with `site < κ`).
    pub flux: i8,
    /// Bitmask of applied-token POSITIONS in operator order —
    /// resolves pair role so role-swapped terms with the same site set land on distinct
    /// vertices.
    pub mask: u8,
    /// Accumulated operator spherical rank (two-units) at this bond, `= ranks[nleft]` where `nleft`
    /// = number of distinct sites left of the bond (0-based `ranks` index, dropping the
    /// `ranks[nleft+1]` `+1` once).
    pub twok: u8,
    /// Accumulated operator irrep flux `Δ` at this bond: `⊗ g_{site(p)}` over EVERY token `p`
    /// applied left of the bond — creations AND annihilations alike (`Z2^m` is self-inverse, so a
    /// token contributes its site's irrep regardless of direction). Note the ASYMMETRY with
    /// [`Self::flux`], which is signed by token kind: copying the `flux` branch here yields a `Δ`
    /// that is wrong on every annihilation and invisible at `|G| = 1` (the `set_edges`
    /// Δ-increment `debug_assert` is the tripwire). [`Irrep::IDENTITY`] in a
    /// trivial-group space.
    pub delta: Irrep,
    /// Content-hash channel identity — `0` for 1-/2-body terms (channels never share a median
    /// vertex there); nonzero for 3-body multi-channel terms, interned to a small ordinal by
    /// [`assembly::Automaton`] purely to fit the packed vertex-label key.
    pub chanid: u64,
}

impl StateT {
    /// The trivial (transport) vertex state: no tokens applied yet, `chanid = 0`.
    pub const ZERO: StateT = StateT {
        flux: 0,
        mask: 0,
        twok: 0,
        delta: Irrep::IDENTITY,
        chanid: 0,
    };
}

/// A [`StateT`] paired with its role-ordered open-site pair `(i1, i2)`. Sites are stored
/// offset-by-one (`0 = ⦿` absent, `s ↦ (s+1) as u16`), the abelian `pack_orb`-style sentinel
/// encoding dropping the spin bit the abelian `OpenIdx` needs (SU2 open indices are spin-free site
/// integers). **NEVER sorted** — role (token-position) order distinguishes terms with the same site
/// set but swapped pair roles (the exact abelian `vertex()` role-order rule).
pub type Ts = (StateT, [u16; 2]);

#[derive(Clone, Copy, Debug)]
pub(crate) struct Su2GridGeom {
    n1: usize,
    s1: usize,
    nir: usize,
}

impl Su2GridGeom {
    pub(crate) fn of(space: &Su2Space) -> Self {
        Su2GridGeom {
            n1: space.n + 1,
            s1: space.n + 1,
            nir: space.pg.order() as usize,
        }
    }

    #[inline]
    pub(crate) fn nsec(&self) -> usize {
        self.nir * self.n1 * self.s1
    }

    #[inline]
    pub(crate) fn idx(&self, s: Su2Sector) -> usize {
        debug_assert!(
            s.n < self.n1 && s.two_s < self.s1 && (s.irrep.index() as usize) < self.nir,
            "sector {s:?} out of grid"
        );
        (s.irrep.index() as usize * self.n1 + s.n) * self.s1 + s.two_s
    }

    #[inline]
    pub(crate) fn sector(&self, g: usize) -> Su2Sector {
        let plane = self.n1 * self.s1;
        let (ir, p) = (g / plane, g % plane);
        Su2Sector::with_irrep(p / self.s1, p % self.s1, Irrep::new(ir as u8))
    }
}

/// One populated COO cell of a site's nest, keyed by its x-side coordinates (SU(2) twin of the
/// abelian `assembly::CooEntry`)./// `m_ord` is the position of the x-row sector `m` in
/// [`Su2Space::bond_sectors`] — the order the matvec kernels visit input sectors in, so a slot's
/// entries are already stored in visitation order and the kernels iterate them directly.
#[derive(Clone, Debug)]
pub struct Su2CooEntry<T: Scalar> {
    /// `x_occ as u8` (the x-transition [`Family`]).
    pub x_occ: u8,
    /// Position of the x-row sector in `Su2Space::bond_sectors(k)`.
    pub m_ord: u32,
    pub cell: CooCell<T>,
}

/// COO block storage for one site: a **populated-cell list** with a dense outer index (mirroring
/// the abelian `HNest`).
pub struct Su2CooNest<T: Scalar> {
    offsets: Vec<u32>,
    cells: Vec<Su2CooEntry<T>>,
    geom: Su2GridGeom,
    sectors: Vec<Su2Sector>,
}

impl<T: Scalar> Su2CooNest<T> {
    pub(crate) fn from_parts(
        offsets: Vec<u32>,
        cells: Vec<Su2CooEntry<T>>,
        geom: Su2GridGeom,
        sectors: Vec<Su2Sector>,
    ) -> Self {
        debug_assert_eq!(offsets.len(), 4 * geom.nsec() + 1);
        Su2CooNest {
            offsets,
            cells,
            geom,
            sectors,
        }
    }

    /// The populated cells of the outer slot `(y_occ, n1)`, in kernel visitation order.
    #[inline]
    pub fn slot(&self, y_occ: Family, n1: Su2Sector) -> &[Su2CooEntry<T>] {
        let o = (y_occ as usize) * self.geom.nsec() + self.geom.idx(n1);
        let lo = self.offsets[o] as usize;
        let hi = self.offsets[o + 1] as usize;
        &self.cells[lo..hi]
    }

    /// `Su2Space::bond_sectors(k)` for this site — `m_ord` indexes it.
    #[inline]
    pub fn sectors(&self) -> &[Su2Sector] {
        &self.sectors
    }

    /// Total populated cells (the census surface).
    pub fn populated(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub(crate) fn offsets(&self) -> &[u32] {
        &self.offsets
    }

    pub(crate) fn into_cells(self) -> impl Iterator<Item = Su2CooEntry<T>> {
        self.cells.into_iter()
    }

    /// Random access to one cell, `None` when it carries no triple (test/diff surface — the
    /// production consumers all iterate [`Self::slot`]).
    pub fn cell(
        &self,
        y_occ: usize,
        n1: Su2Sector,
        x_occ: usize,
        m: Su2Sector,
    ) -> Option<&CooCell<T>> {
        let o = y_occ * self.geom.nsec() + self.geom.idx(n1);
        let (lo, hi) = (self.offsets[o] as usize, self.offsets[o + 1] as usize);
        self.cells[lo..hi]
            .iter()
            .find(|e| e.x_occ as usize == x_occ && self.sectors[e.m_ord as usize] == m)
            .map(|e| &e.cell)
    }

    /// Every populated cell as `(y_occ, n1, x_occ, m, &cell)` in **`Su2SectorGrid` order** on both
    /// sector axes (`irrep` slowest, `2s` fastest) rather than storage order. Cell
    /// *visitation* order is a storage artifact and carries no numerical meaning; only the
    /// intra-cell `(rows, cols, vals)` sequence is the determinism claim. No SU(2) pinned artifact
    /// hash exists (unlike the abelian `PINNED_COO_HASH`), so this view is a census/diff surface
    /// only. It genuinely differs from storage order: `bond_sectors` runs `2s` OUTER / `n` INNER
    /// (qn.rs, the column-major order), the grid `n` outer / `2s` inner.
    pub fn iter_grid_order(&self) -> Vec<(usize, Su2Sector, usize, Su2Sector, &CooCell<T>)> {
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

pub(crate) struct Su2HCell<T: Scalar> {
    pub x_occ: u8,
    pub m_ord: u32,
    pub csr: CsxBlock<T>,
    pub csc: CsxBlock<T>,
    pub(crate) src: Option<Box<Su2CellSrc>>,
}

impl<T: Scalar> Su2HCell<T> {
    #[inline]
    pub(crate) fn triples(&self) -> impl Iterator<Item = (i32, i32, T)> + '_ {
        self.csr.entries()
    }

    fn refill(
        &mut self,
        gamma: &[T],
        factors: &[Su2Factor],
        changed: &[u64],
        chunk: usize,
    ) -> usize {
        let Su2HCell { csr, csc, src, .. } = self;
        let s = src
            .as_deref()
            .expect("Su2HNest::refill on a cell built without provenance (not a refillable build)");
        let mut mask = vec![0u8; csr.vals.len()];
        let written = if csr.vals.len() <= chunk {
            s.refill_csr(&mut csr.vals, 0, gamma, factors, changed, &mut mask)
        } else {
            csr.vals
                .par_chunks_mut(chunk)
                .zip(mask.par_chunks_mut(chunk))
                .enumerate()
                .map(|(i, (v, m))| s.refill_csr(v, i * chunk, gamma, factors, changed, m))
                .sum()
        };
        if written == 0 {
            return 0;
        }
        let (csr_vals, mask) = (&csr.vals, &mask);
        written
            + if csc.vals.len() <= chunk {
                s.refill_csc(&mut csc.vals, 0, csr_vals, mask)
            } else {
                csc.vals
                    .par_chunks_mut(chunk)
                    .enumerate()
                    .map(|(i, v)| s.refill_csc(v, i * chunk, csr_vals, mask))
                    .sum()
            }
    }
}

pub(crate) struct Su2HNest<T: Scalar> {
    offsets: Vec<u32>,
    diag: Vec<u32>,
    cells: Vec<Su2HCell<T>>,
    sectors: Vec<Su2Sector>,
    geom: Su2GridGeom,
}

impl<T: Scalar> Su2HNest<T> {
    #[inline]
    fn oslot(&self, y_occ: Family, n1: Su2Sector) -> usize {
        (y_occ as usize) * self.geom.nsec() + self.geom.idx(n1)
    }

    #[inline]
    pub(crate) fn slot(&self, y_occ: Family, n1: Su2Sector) -> &[Su2HCell<T>] {
        let o = self.oslot(y_occ, n1);
        &self.cells[self.offsets[o] as usize..self.offsets[o + 1] as usize]
    }

    #[inline]
    pub(crate) fn sector_of(&self, c: &Su2HCell<T>) -> Su2Sector {
        self.sectors[c.m_ord as usize]
    }

    #[inline]
    fn diag_cell(&self, y_occ: Family, n1: Su2Sector) -> Option<&Su2HCell<T>> {
        let i = self.diag[self.oslot(y_occ, n1)];
        (i != u32::MAX).then(|| &self.cells[i as usize])
    }

    fn from_coo(coo: Su2CooNest<T>, runs: Option<(Vec<Su2RunSrc>, &[Su2Factor])>) -> Self {
        let geom = coo.geom;
        let nsec = geom.nsec();
        let sectors = coo.sectors().to_vec();
        let offsets = coo.offsets().to_vec();
        let mut runs = runs.map(|(r, factors)| {
            assert_eq!(
                r.len(),
                coo.populated(),
                "recording: run provenance not parallel to the nest's cells"
            );
            (r.into_iter(), factors)
        });
        let cells: Vec<Su2HCell<T>> = coo
            .into_cells()
            .map(|e| {
                let (csr, csc, src) = match runs.as_mut() {
                    None => {
                        let (csr, csc) = crate::hamiltonian::csx::csx_pair(&e.cell);
                        (csr, csc, None)
                    }
                    Some((runs, factors)) => {
                        let run = runs.next().expect("one run record per cell");
                        let slots: Vec<u32> = (0..e.cell.vals.len() as u32).collect();
                        let (csr, csc, perm) =
                            crate::hamiltonian::csx::csx_pair_with_src(&e.cell, &slots);
                        debug_assert!(
                            perm.csr_src
                                .iter()
                                .enumerate()
                                .all(|(i, &p)| p as usize == i),
                            "a frozen cell's CSR permutation is not the identity"
                        );
                        let src = Box::new(Su2CellSrc::new(run, perm.csc_src, factors));
                        (csr, csc, Some(src))
                    }
                };
                Su2HCell {
                    x_occ: e.x_occ,
                    m_ord: e.m_ord,
                    csr,
                    csc,
                    src,
                }
            })
            .collect();
        let mut diag = vec![u32::MAX; 4 * nsec];
        for y_occ in 0..4usize {
            for n1g in 0..nsec {
                let o = y_occ * nsec + n1g;
                let (lo, hi) = (offsets[o] as usize, offsets[o + 1] as usize);
                let n1 = geom.sector(n1g);
                for (i, c) in cells[lo..hi].iter().enumerate() {
                    if c.x_occ as usize == y_occ && sectors[c.m_ord as usize] == n1 {
                        diag[o] = (lo + i) as u32;
                        break;
                    }
                }
            }
        }
        Su2HNest {
            offsets,
            diag,
            cells,
            sectors,
            geom,
        }
    }

    fn refill(
        &mut self,
        gamma: &[T],
        factors: &[Su2Factor],
        changed: &[u64],
        chunk: usize,
    ) -> usize {
        self.cells
            .par_iter_mut()
            .map(|c| c.refill(gamma, factors, changed, chunk))
            .sum()
    }
}

/// The assembled SU(2) operator-action automaton for a screened one-/two-/three-body
/// Hamiltonian./// `new` runs the full [`assembly::assemble`] pipeline and
/// `apply`/`row_ranks`/`col_ranks`/ `row_ranges`/`col_ranges` read `state_qns`/`nests`; `states` is
/// read by [`Self::channel_descs`] (the perturbative-noise channel grouping). The struct-level
/// `#[allow(dead_code)]` matches the abelian precedent this struct otherwise mirrors.
#[allow(dead_code)]
pub struct Su2SparseHamiltonian<T: Scalar> {
    /// Symmetry descriptor shared with the trains it acts on.
    pub space: Su2Space,
    states: Vec<Su2SectorGrid<Vec<Ts>>>,
    state_qns: Vec<Su2SectorGrid<Vec<Su2Sector>>>,
    pub(crate) nests: Vec<Su2HNest<T>>,
    empty_csx: CsxBlock<T>,
    prov: Option<Su2Provenance<T>>,
    skeleton: Option<Su2Skeleton>,
}

struct Su2Skeleton {
    screen: assembly::Su2TermScreen,
    #[cfg_attr(not(test), allow(dead_code))]
    debris_drops: usize,
    term_hash: Option<u64>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Su2RefillStats {
    pub(crate) changed_terms: usize,
    pub(crate) slots_written: usize,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Su2RefillCensus {
    pub(crate) multi_contrib_runs: usize,
    pub(crate) multi_term_runs: usize,
    pub(crate) transport_term_runs: usize,
    pub(crate) debris_drops: usize,
    pub(crate) factors: usize,
    pub(crate) escape_blocks: usize,
    pub(crate) wide_perm_cells: usize,
    pub(crate) fid_fallback_cells: usize,
    pub(crate) zero_values: usize,
}

struct Su2Provenance<T: Scalar> {
    n_terms: usize,
    refill_gamma: Vec<T>,
    factors: Vec<Su2Factor>,
}

impl<T: Scalar> Su2SparseHamiltonian<T> {
    /// Screen and assemble the one-/two-/(optional) three-body term automaton. `w = None` omits the
    /// three-body channel entirely. Shape asserts mirror the abelian constructor: `t` is `d×d`
    /// (rank 2), `v` is `d×d×d×d` (rank 4), `w` (if present) has `norb() == d`.
    pub fn new(
        space: Su2Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        eps: f64,
    ) -> Self {
        Self::with_screen(space, t, v, w, assembly::Su2TermScreen::Magnitude(eps))
    }

    pub(crate) fn with_screen(
        space: Su2Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        screen: assembly::Su2TermScreen,
    ) -> Self {
        Self::check_shapes(space, t, v, w);
        let mut spin = SpinAlgebra::new();
        let assembly::Assembled {
            states,
            state_qns,
            coo,
        } = assembly::assemble_screened(space, t, v, w, &screen, &mut spin);
        let nests: Vec<Su2HNest<T>> = coo
            .into_iter()
            .map(|c| Su2HNest::from_coo(c, None))
            .collect();
        Su2SparseHamiltonian {
            space,
            states,
            state_qns,
            nests,
            empty_csx: CsxBlock::empty(),
            prov: None,
            skeleton: None,
        }
    }

    /// A **refillable** build (orbital-rotation R1b, the SU(2) twin of the abelian
    /// `SparseHamiltonian::new_refillable`): the operator's SKELETON — vertex lists,
    /// coupled x-sector lists, every index array, which cells, runs and entries exist — is a
    /// function of `space` and the Phase-A term set ONLY, never of the values beyond it, and every
    /// stored value carries its provenance (the Phase-A term stream, an interned fixed-factor
    /// table, per entry its contributions). Under `SymmetryAllowed` and `Mask` the term set is a
    /// function of `(space, screen)`; under `Magnitude` it is the mint-time values' (`|w| > eps`),
    /// so a refill must not add a class. Built under the frozen build rules (the
    /// `assembly::provenance` module docs): the fill skips a deposit only when its fixed factor is
    /// zero, and the freeze drops a run only when every term group cancels structurally — so
    /// zero-weight classes are stored as explicit `+0` entries instead of dropped.
    ///
    /// # Panics
    ///
    /// On ANY three-body `w`, under every screen (the orbital-rotation driver has none, and that
    /// channel's candidate screen is value-driven); on a `SymmetryAllowed` or `Mask` point group
    /// that is not `space.pg`; on a `Mask` over another `d`.
    pub fn new_refillable(
        space: Su2Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        screen: Su2TermScreen,
    ) -> Self {
        assert!(
            w.is_none(),
            "Su2SparseHamiltonian::new_refillable refuses a three-body channel (its candidate \
             screen is value-driven, so its term set is not a pattern) — build it with `new`"
        );
        match &screen {
            Su2TermScreen::SymmetryAllowed(pg) => assert!(
                *pg == space.pg,
                "new_refillable: the SymmetryAllowed screen's point group must be the space's"
            ),
            Su2TermScreen::Mask(m) => {
                assert!(
                    m.pg() == space.pg,
                    "new_refillable: the Mask's point group must be the space's"
                );
                assert!(
                    m.d() == space.d,
                    "new_refillable: the Mask's d must be the space's ({} vs {})",
                    m.d(),
                    space.d
                );
            }
            Su2TermScreen::Magnitude(_) => {}
        }
        Self::build_recorded(space, t, v, screen, true)
    }

    /// Was this operator built by [`Self::new_refillable`]?
    pub fn is_refillable(&self) -> bool {
        self.skeleton.is_some()
    }

    /// The screen a refillable build was minted with (`None` on a plain build).
    pub fn term_screen(&self) -> Option<Su2TermScreen> {
        self.skeleton.as_ref().map(|s| s.screen.clone())
    }

    /// Classes of `(t, v)` that a fresh `new(.., eps)` would carry and this skeleton cannot (the
    /// SU(2) twin of the abelian `SparseHamiltonian::pattern_leak`): `0` means a
    /// [`Self::refill`] with `(t, v)` acts as that operator, any mask class whose `|w| ≤ eps` being
    /// stored explicitly (bitwise it when there is none, up to cancellation debris); a positive
    /// count means the pattern has genuinely grown and the skeleton must be re-minted: a refill
    /// under it would SILENTLY omit the leaked classes, and the result would not be `H(t, v)`.
    /// Always `0` for a `SymmetryAllowed` skeleton.
    ///
    /// # Panics
    ///
    /// On a build that is not refillable (a plain or a value-rule recording build), on a
    /// `Magnitude` skeleton (whose only guards are `refill`'s term-count and token-hash asserts),
    /// and as [`Su2PatternMask::leak`].
    pub fn pattern_leak(&self, t: &CoeffTensor<T>, v: &CoeffTensor<T>, eps: f64) -> usize {
        let sk = self
            .skeleton
            .as_ref()
            .expect("Su2SparseHamiltonian::pattern_leak: not a refillable build");
        match &sk.screen {
            Su2TermScreen::Mask(m) => m.leak(t, v, eps, self.space.pg),
            Su2TermScreen::SymmetryAllowed(_) => 0,
            Su2TermScreen::Magnitude(_) => panic!(
                "Su2SparseHamiltonian::pattern_leak: a Magnitude skeleton has no frozen class set \
                 — mint with Su2TermScreen::Mask(Su2PatternMask::from_magnitude(..)) instead"
            ),
        }
    }

    /// Rewrite the values of a [`Self::new_refillable`] build from new integrals over the same
    /// space, keeping the skeleton — `states`, `state_qns`, every nest's offsets, diagonal index
    /// and CSR/CSC index arrays, and the provenance — bitwise unchanged (orbital-rotation R2, the
    /// SU(2) twin of the abelian `SparseHamiltonian::refill`). Bitwise
    /// `new_refillable(space, t, v, None, screen)` with the minted screen: Phase A is re-run under
    /// that screen for the new term stream, and each stored value is one run sum
    /// (`+0` start, run order) over α recomputed from the stream and the fixed factors — the
    /// build's own expressions.
    ///
    /// # Panics
    ///
    /// If the operator is not a refillable build (a plain or a value-rule recording build); on any
    /// `Some(w)`; on a shape mismatch; if the Phase-A term count differs from the minted one, or
    /// the count matches but the emitted term SET (the token sequence, hashed at mint under
    /// `Magnitude`) does not — a `Magnitude` screen whose surviving set moved with the values (use
    /// `SymmetryAllowed` or a `Mask`). Under a pattern screen the term set is fixed, so both guards
    /// are vacuous and the token sequence is not hashed (the count is still checked); a `Mask`
    /// refill never checks the leak ([`Self::pattern_leak`] is the caller's) and never reshapes.
    /// Values and the stored γ are written only after every guard has passed, and no panic source
    /// is reachable in the write phase — so a refused refill leaves the operator untouched (were
    /// one reachable, a caught panic would leave values half-written against a stale γ; the
    /// abelian refill is built the same way).
    pub fn refill(&mut self, t: &CoeffTensor<T>, v: &CoeffTensor<T>, w: Option<&dyn ThreeBody<T>>) {
        self.refill_chunked(t, v, w, crate::hamiltonian::REFILL_CHUNK);
    }

    pub(crate) fn refill_chunked(
        &mut self,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        chunk: usize,
    ) -> Su2RefillStats {
        assert!(chunk >= 1, "Su2SparseHamiltonian::refill: a zero task size");
        let Su2SparseHamiltonian {
            space,
            nests,
            prov,
            skeleton,
            ..
        } = self;
        let sk = skeleton
            .as_ref()
            .expect("Su2SparseHamiltonian::refill: not a refillable build (use `new_refillable`)");
        assert!(
            w.is_none(),
            "Su2SparseHamiltonian::refill: refuses a three-body channel (a refillable build has \
             none)"
        );
        Self::check_shapes(*space, t, v, None);
        let (gamma, term_hash) = assembly::phase_a_stream(space.d, t, v, &sk.screen);
        let prov = prov
            .as_mut()
            .expect("a refillable build is a recording build");
        assert_eq!(
            gamma.len(),
            prov.n_terms,
            "Su2SparseHamiltonian::refill: the Phase-A term count changed ({} vs {} at build) — \
             the skeleton is not frozen for these integrals (a `Su2TermScreen::Magnitude` screen \
             moves with the values; use `SymmetryAllowed`)",
            gamma.len(),
            prov.n_terms
        );
        assert_eq!(
            term_hash, sk.term_hash,
            "Su2SparseHamiltonian::refill: the Phase-A term set changed (the same {} terms by \
             count, a different token sequence) — the skeleton is not frozen for these integrals \
             (a `Su2TermScreen::Magnitude` screen moves with the values; use `SymmetryAllowed`)",
            prov.n_terms
        );
        let changed: Vec<u64> = gamma
            .par_chunks(64)
            .zip(prov.refill_gamma.par_chunks(64))
            .map(|(new, old)| {
                let mut w = 0u64;
                for (b, (x, y)) in new.iter().zip(old).enumerate() {
                    let same = x.re_s().to_bits() == y.re_s().to_bits()
                        && x.im_s().to_bits() == y.im_s().to_bits();
                    w |= u64::from(!same) << b;
                }
                w
            })
            .collect();
        let mut stats = Su2RefillStats {
            changed_terms: changed.iter().map(|w| w.count_ones() as usize).sum(),
            ..Su2RefillStats::default()
        };
        if stats.changed_terms > 0 {
            let factors = &prov.factors;
            stats.slots_written = nests
                .par_iter_mut()
                .map(|n| n.refill(&gamma, factors, &changed, chunk))
                .sum();
        }
        prov.refill_gamma = gamma;
        stats
    }

    /// A 64-bit hash of the WHOLE operator, any build: per-bond vertex lists and coupled x-sector
    /// lists, every nest's offsets / diagonal index / sector list, and every cell's
    /// `x_occ`/`m_ord`, CSR/CSC `ids`/`starts`/`index` AND values (IEEE bit patterns, real and
    /// imaginary parts). Provenance-free, so a refillable build and a plain one compare. The SU(2)
    /// twin of the abelian `SparseHamiltonian::artifact_hash`, with its hasher: `FxHasher` over the
    /// derived `Hash` impls in a fixed field order — deterministic for one build of the crate, NOT
    /// a constant to pin (the hasher's algorithm is a dependency's, and the channel fit runs
    /// through faer, so the values may differ across ISAs).
    pub fn artifact_hash(&self) -> u64 {
        self.hash_artifact(true)
    }

    /// [`Self::artifact_hash`] without the values: the skeleton.
    pub fn skeleton_hash(&self) -> u64 {
        self.hash_artifact(false)
    }

    fn hash_artifact(&self, with_vals: bool) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        for b in 0..=self.space.d {
            for s in self.space.bond_sectors(b) {
                self.states[b][s].hash(&mut h);
                self.state_qns[b][s].hash(&mut h);
            }
        }
        for n in &self.nests {
            n.offsets.hash(&mut h);
            n.diag.hash(&mut h);
            n.sectors.hash(&mut h);
            for c in &n.cells {
                c.x_occ.hash(&mut h);
                c.m_ord.hash(&mut h);
                for blk in [&c.csr, &c.csc] {
                    blk.ids.hash(&mut h);
                    blk.starts.hash(&mut h);
                    blk.index.hash(&mut h);
                    if with_vals {
                        for &v in &blk.vals {
                            v.re_s().to_bits().hash(&mut h);
                            v.im_s().to_bits().hash(&mut h);
                        }
                    }
                }
            }
        }
        h.finish()
    }

    fn build_recorded(
        space: Su2Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        screen: Su2TermScreen,
        frozen: bool,
    ) -> Self {
        Self::check_shapes(space, t, v, None);
        let mut spin = SpinAlgebra::new();
        let (
            assembly::Assembled {
                states,
                state_qns,
                coo,
            },
            rec,
        ) = assembly::assemble_recorded(space, t, v, &screen, frozen, &mut spin);
        let debris_drops = rec.debris_drops();
        let (refill_gamma, factors, sites) = rec.into_parts();
        assert_eq!(
            sites.len(),
            coo.len(),
            "recording: one run-provenance site per nest"
        );
        let term_hash = frozen.then(|| {
            let (gamma, hash) = assembly::phase_a_stream(space.d, t, v, &screen);
            assert!(
                gamma.len() == refill_gamma.len()
                    && gamma.iter().zip(&refill_gamma).all(|(a, b)| {
                        a.re_s().to_bits() == b.re_s().to_bits()
                            && a.im_s().to_bits() == b.im_s().to_bits()
                    }),
                "new_refillable: the Phase-A stream disagrees with the recorded term stream"
            );
            hash
        });
        let nests: Vec<Su2HNest<T>> = coo
            .into_iter()
            .zip(sites)
            .map(|(c, runs)| Su2HNest::from_coo(c, Some((runs, &factors))))
            .collect();
        Su2SparseHamiltonian {
            space,
            states,
            state_qns,
            nests,
            empty_csx: CsxBlock::empty(),
            prov: Some(Su2Provenance {
                n_terms: refill_gamma.len(),
                refill_gamma,
                factors,
            }),
            skeleton: term_hash.map(|term_hash| Su2Skeleton {
                screen,
                debris_drops,
                term_hash,
            }),
        }
    }

    fn check_shapes(
        space: Su2Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
    ) {
        assert!(
            t.rank == 2 && t.d == space.d,
            "one-body tensor must be rank-2 over d={} sites (got rank {}, d {})",
            space.d,
            t.rank,
            t.d
        );
        assert!(
            v.rank == 4 && v.d == space.d,
            "two-body tensor must be rank-4 over d={} sites (got rank {}, d {})",
            space.d,
            v.rank,
            v.d
        );
        if let Some(w) = w {
            assert!(
                w.norb() == space.d,
                "three-body tensor orbital count must equal d={} sites (got {})",
                space.d,
                w.norb()
            );
        }
    }

    pub(crate) fn self_cell(&self, k: usize, occ: Family, l: Su2Sector) -> &CsxBlock<T> {
        match self.nests[k].diag_cell(occ, l) {
            Some(c) => &c.csr,
            None => &self.empty_csx,
        }
    }

    pub(crate) fn state_qns_at(&self, b: usize, s: Su2Sector) -> &[Su2Sector] {
        &self.state_qns[b][s]
    }

    pub(crate) fn channel_descs(&self, b: usize, s: Su2Sector) -> &[Ts] {
        &self.states[b][s]
    }
}

#[cfg_attr(not(test), allow(dead_code))]
impl<T: Scalar> Su2SparseHamiltonian<T> {
    pub(crate) fn with_screen_recorded(
        space: Su2Space,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        screen: assembly::Su2TermScreen,
    ) -> Self {
        assert!(
            w.is_none(),
            "Su2SparseHamiltonian::with_screen_recorded: recording refuses a three-body channel \
             (its candidate screen is value-driven, so its term set is not a pattern)"
        );
        Self::build_recorded(space, t, v, screen, false)
    }

    pub(crate) fn refill_gamma(&self) -> Option<&[T]> {
        self.prov.as_ref().map(|p| p.refill_gamma.as_slice())
    }

    pub(crate) fn n_terms(&self) -> Option<usize> {
        self.prov.as_ref().map(|p| p.n_terms)
    }

    pub(crate) fn factors(&self) -> Option<&[Su2Factor]> {
        self.prov.as_ref().map(|p| p.factors.as_slice())
    }

    pub(crate) fn refill_census(&self) -> Option<Su2RefillCensus> {
        let sk = self.skeleton.as_ref()?;
        let factors = self
            .factors()
            .expect("a refillable build is a recording build");
        let mut c = Su2RefillCensus {
            debris_drops: sk.debris_drops,
            factors: factors.len(),
            ..Su2RefillCensus::default()
        };
        let mut terms: Vec<u32> = Vec::new();
        for cell in self.nests.iter().flat_map(|n| n.cells.iter()) {
            let src = cell
                .src
                .as_ref()
                .expect("a recording build carries provenance on every cell");
            let lay = src.layout_census();
            c.escape_blocks += lay.escape_blocks;
            c.wide_perm_cells += usize::from(lay.wide_perm);
            c.fid_fallback_cells += usize::from(lay.fid_fallback);
            for p in 0..cell.csr.nnz() {
                let (mut len, mut transport) = (0usize, false);
                terms.clear();
                for (term, _) in src.run(p) {
                    len += 1;
                    if term == SRC_NONE {
                        transport = true;
                    } else if !terms.contains(&term) {
                        terms.push(term);
                    }
                }
                c.multi_contrib_runs += usize::from(len > 1);
                c.multi_term_runs += usize::from(terms.len() > 1);
                c.transport_term_runs += usize::from(transport && !terms.is_empty());
                let x = cell.csr.vals[p];
                c.zero_values += usize::from(x.re_s().to_bits() == 0 && x.im_s().to_bits() == 0);
            }
        }
        Some(c)
    }

    pub(crate) fn src_hash(&self) -> Option<u64> {
        use std::hash::{Hash, Hasher};
        let p = self.prov.as_ref()?;
        let mut h = rustc_hash::FxHasher::default();
        p.n_terms.hash(&mut h);
        for g in &p.refill_gamma {
            g.re_s().to_bits().hash(&mut h);
            g.im_s().to_bits().hash(&mut h);
        }
        for f in &p.factors {
            f.cw.to_bits().hash(&mut h);
            f.fr.to_bits().hash(&mut h);
            f.neg.hash(&mut h);
            f.transport.hash(&mut h);
        }
        for n in &self.nests {
            n.cells.len().hash(&mut h);
            for c in &n.cells {
                c.src
                    .as_ref()
                    .expect("a recording build carries provenance on every cell")
                    .hash_into(&mut h);
            }
        }
        Some(h.finish())
    }
}

impl<T: Scalar> std::ops::Mul<&Su2TtVector<T>> for &Su2SparseHamiltonian<T> {
    type Output = Su2TtVector<T>;
    fn mul(self, x: &Su2TtVector<T>) -> Su2TtVector<T> {
        self.apply(x)
    }
}

/// `⟨x, Hx⟩`. NOT divided by
/// `⟨x,x⟩` despite the name; same quirk as the abelian twin
/// `rayleigh_quotient`. Realized through [`xt_h_y`]'s fused identity-sketch env sweep; certified
/// through the cross-symmetry `expand` gate.
pub fn rayleigh_quotient<T: Scalar>(h: &Su2SparseHamiltonian<T>, x: &Su2TtVector<T>) -> T {
    xt_h_y(x, h, x)
}

/// `⟨x, Hy⟩`, the bilinear twin of [`rayleigh_quotient`] (`xᵀHy`).
pub fn xt_h_y<T: Scalar>(x: &Su2TtVector<T>, h: &Su2SparseHamiltonian<T>, y: &Su2TtVector<T>) -> T {
    assert!(
        x.boundary_width() == 1 && y.boundary_width() == 1,
        "xt_h_y is the scalar ⟨x, Hy⟩ of two unit-boundary trains (widths {} and {}); an \
         open block goes through xt_h_y_block",
        x.boundary_width(),
        y.boundary_width()
    );
    *fold_x_h_y(x, h, y).get(0, 0)
}

/// Operator Gram of two OPEN BLOCKS: `G[i, j] = ⟨X_i, H·Y_j⟩` (conjugation on `x`), of shape
/// `x.boundary_width() × y.boundary_width()` — the sketched-Gram kernel of block trace
/// estimation. It is [`xt_h_y`]'s own LEFT fold (unweighted on both layers; `H·Y` is never
/// materialized) with the whole bond-`d` block read out instead of one entry, so the cost does
/// not grow with the widths beyond the last core's. On a unit-boundary pair it is the `1×1`
/// matrix holding [`xt_h_y`], bit for bit.
pub fn xt_h_y_block<T: Scalar>(
    x: &Su2TtVector<T>,
    h: &Su2SparseHamiltonian<T>,
    y: &Su2TtVector<T>,
) -> faer::Mat<T> {
    fold_x_h_y(x, h, y)
}

fn fold_x_h_y<T: Scalar>(
    x: &Su2TtVector<T>,
    h: &Su2SparseHamiltonian<T>,
    y: &Su2TtVector<T>,
) -> faer::Mat<T> {
    let space = x.space;
    let d = space.d;
    let mut p = Su2Sketch::identity(space, 0, 1);
    for k in 0..d {
        let l = matvec::sketch_h_mul(&p, h, y.core(k));
        p = adjoint_core_times_core(x.core(k), &l);
    }
    let last = Su2Sector::with_irrep(space.n, space.two_s, space.target);
    p.block(last)
        .expect("final bond has the top (n, 2S) sector")
        .to_owned()
}

impl<T: Scalar> crate::mem::Footprint for Su2SparseHamiltonian<T> {
    fn bytes(&self) -> u64 {
        let blocks: u64 = self
            .nests
            .iter()
            .flat_map(|n| n.cells.iter())
            .map(|c| c.csr.bytes() + c.csc.bytes() + c.src.as_ref().map_or(0, |s| s.bytes()))
            .sum();
        let states: u64 = self
            .states
            .iter()
            .flat_map(|g| g.iter())
            .map(|(_, v)| (v.capacity() * std::mem::size_of::<Ts>()) as u64)
            .sum();
        let state_qns: u64 = self
            .state_qns
            .iter()
            .flat_map(|g| g.iter())
            .map(|(_, v)| (v.capacity() * std::mem::size_of::<Su2Sector>()) as u64)
            .sum();
        let prov = self.prov.as_ref().map_or(0, |p| {
            (p.refill_gamma.capacity() * std::mem::size_of::<T>()
                + p.factors.capacity() * std::mem::size_of::<Su2Factor>()) as u64
        });
        blocks + states + state_qns + prov
    }
}

impl<T: Scalar> Su2SparseHamiltonian<T> {
    /// Predicted logical bytes of the environment `Su2Sketch` at every bond `0..=d` for a train
    /// with the given `d+1` reduced-rank grids — the SU(2) twin of the abelian
    /// `SparseHamiltonian::predicted_env_bytes`: interior bond block at sector `s` is
    /// `ranks[b][s] × Σ_{m ∈ state_qns[b][s]} ranks[b][m]`, boundary bonds are the identity frame.
    pub fn predicted_env_bytes(&self, ranks: &[Su2RankGrid]) -> Vec<u64> {
        let d = self.space.d;
        assert_eq!(
            ranks.len(),
            d + 1,
            "predicted_env_bytes: need d+1 rank grids"
        );
        (0..=d)
            .map(|b| {
                self.space
                    .bond_sectors(b)
                    .iter()
                    .map(|&s| {
                        let rows = ranks[b][s] as u64;
                        let cols: u64 = if b == 0 || b == d {
                            rows
                        } else {
                            self.state_qns[b][s]
                                .iter()
                                .map(|&m| ranks[b][m] as u64)
                                .sum()
                        };
                        rows * cols * std::mem::size_of::<T>() as u64
                    })
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod block_tests;
