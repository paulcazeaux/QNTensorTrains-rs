// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Block-sparse TT core.

use std::ops::Range;

use faer::prelude::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use rand::Rng;

use crate::blockstore::{BlockStore, Lookup};
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

pub(crate) mod algebra;
#[cfg(test)]
pub(crate) mod border_fixture;
pub(crate) mod deflate;
pub mod factor;
pub mod tangent;
pub mod twosite;
pub mod weingarten;

/// The four block families of a core, by site occupation. The discriminant is the physical index: 0
/// = empty, 1 = up, 2 = dn, 3 = doubly occupied.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Family {
    /// Site unoccupied (`○○`): column sector = row sector.
    Empty = 0,
    /// One spin-up electron (`up`): column sector = row + (1, 0).
    Up = 1,
    /// One spin-down electron (`dn`): column sector = row + (0, 1).
    Dn = 2,
    /// Both electrons (`●●`): column sector = row + (1, 1).
    Occ = 3,
}

impl Family {
    /// All four families in physical-index order.
    pub const ALL: [Family; 4] = [Family::Empty, Family::Up, Family::Dn, Family::Occ];

    /// Family from a physical index `0..4`.
    #[inline]
    pub fn from_physical_index(i: usize) -> Family {
        Self::ALL[i]
    }

    /// Family occupying the site with the given spins.
    #[inline]
    pub fn from_occupation(occ_up: bool, occ_dn: bool) -> Family {
        match (occ_up, occ_dn) {
            (false, false) => Family::Empty,
            (true, false) => Family::Up,
            (false, true) => Family::Dn,
            (true, true) => Family::Occ,
        }
    }

    /// Sector shift `(Δnup, Δndn)` this family applies across the site.
    #[inline]
    pub fn delta(self) -> (usize, usize) {
        match self {
            Family::Empty => (0, 0),
            Family::Up => (1, 0),
            Family::Dn => (0, 1),
            Family::Occ => (1, 1),
        }
    }

    /// Point-group irrep flux across a site whose orbital irrep is `g`: nothing for an empty
    /// site, `g` for each singly-occupied spin-orbital, `g ⊗ g` (= [`Irrep::IDENTITY`] for the
    /// `Z2^m` groups we support) for a doubly-occupied one.
    #[inline]
    pub const fn flux(self, g: Irrep) -> Irrep {
        match self {
            Family::Empty => Irrep::IDENTITY,
            Family::Up | Family::Dn => g,
            Family::Occ => g.fuse(g),
        }
    }

    /// Column sector reached from row sector `s` across a site of irrep `g`.
    #[inline]
    pub fn apply(self, s: Sector, g: Irrep) -> Sector {
        let (du, dd) = self.delta();
        Sector::with_irrep(s.nup + du, s.ndn + dd, s.irrep.fuse(self.flux(g)))
    }

    /// Inverse of [`apply`](Self::apply): the row sector that reaches `t` across a site of irrep
    /// `g`. `None` exactly when the particle counts underflow, i.e. this family cannot enter `t`.
    #[inline]
    pub fn unapply(self, t: Sector, g: Irrep) -> Option<Sector> {
        let (du, dd) = self.delta();
        Some(Sector::with_irrep(
            t.nup.checked_sub(du)?,
            t.ndn.checked_sub(dd)?,
            t.irrep.fuse(self.flux(g).inverse()),
        ))
    }
}

/// A dense per-row-sector block grid of one family: `None` where the transition leaves the
/// admissible sector cone, `Some` (possibly with a zero dimension) where it does not. The layout
/// `SparseCore` stored before it moved onto its block store (`crate::blockstore`); no longer used
/// inside the crate.
pub type BlockGrid<T> = SectorGrid<Option<Mat<T>>>;

#[inline]
fn storage_len(space: &QnSpace) -> usize {
    SectorGrid::<()>::storage_len(space)
}

#[inline]
fn block_key(space: &QnSpace, len: usize, f: Family, s: Sector) -> u32 {
    (f as usize * len + SectorGrid::<()>::storage_index(space, s)) as u32
}

/// One block-sparse TT core.
#[derive(Clone, Debug)]
pub struct SparseCore<T: Scalar> {
    /// Symmetry descriptor shared by the whole train.
    pub space: QnSpace,
    /// Site index, `0..d`.
    pub k: usize,
    /// Admissible row sectors (bond `k`), canonical order.
    pub row_qn: Vec<Sector>,
    /// Admissible column sectors (bond `k+1`), canonical order.
    pub col_qn: Vec<Sector>,
    /// Per-sector row ranks (zero at inadmissible sectors).
    pub row_ranks: RankGrid,
    /// Per-sector column ranks (zero at inadmissible sectors).
    pub col_ranks: RankGrid,
    store: BlockStore<T>,
}

impl<T: Scalar> SparseCore<T> {
    /// Zero-initialized core with the given bond ranks. Positive ranks are only permitted on
    /// admissible sectors.
    pub fn zeros(space: QnSpace, k: usize, row_ranks: &RankGrid, col_ranks: &RankGrid) -> Self {
        assert!(k < space.d, "site index {k} out of range 0..{}", space.d);
        let row_qn = space.bond_sectors(k);
        let col_qn = space.bond_sectors(k + 1);

        for (s, &r) in row_ranks.iter() {
            assert!(
                r == 0 || space.contains(k, s),
                "positive row rank {r} at inadmissible sector {s:?} of bond {k}"
            );
        }
        for (s, &r) in col_ranks.iter() {
            assert!(
                r == 0 || space.contains(k + 1, s),
                "positive column rank {r} at inadmissible sector {s:?} of bond {}",
                k + 1
            );
        }

        let g = space.pg.irrep_of(k);
        let len = storage_len(&space);
        let mut tally = crate::storage_probe::CoreTally::new();
        let mut entries: Vec<(u32, Mat<T>)> = Vec::new();
        let mut empties: Vec<(u32, usize, usize)> = Vec::new();
        for f in Family::ALL {
            for i in 0..len {
                let s = SectorGrid::<()>::storage_label(&space, i);
                let t = f.apply(s, g);
                if space.contains(k, s) && space.contains(k + 1, t) {
                    let (r, c) = (row_ranks[s], col_ranks[t]);
                    tally.block(r, c);
                    let key = (f as usize * len + i) as u32;
                    if r * c > 0 {
                        entries.push((key, Mat::zeros(r, c)));
                    } else {
                        empties.push((key, r, c));
                    }
                }
            }
        }
        let store = BlockStore::from_sorted_recording_empty(4 * len, entries, empties);
        if crate::storage_probe::ENABLED {
            let (kind, table) = (
                crate::storage_probe::CoreKind::Assembled,
                crate::storage_probe::Table::Abelian,
            );
            tally.finish(kind, table, k, 0, store.len(), store.dense_index_len());
        }

        SparseCore {
            space,
            k,
            row_qn,
            col_qn,
            row_ranks: row_ranks.clone(),
            col_ranks: col_ranks.clone(),
            store,
        }
    }

    pub(crate) fn restamp(&self, space: QnSpace) -> Self {
        assert!(
            space.same_lattice(&self.space),
            "restamp: {space:?} is not the same lattice as {:?}",
            self.space
        );
        let mut out = self.clone();
        out.space = space;
        out.row_qn = space.bond_sectors(self.k);
        out.col_qn = space.bond_sectors(self.k + 1);
        out
    }

    #[allow(dead_code)]
    pub(crate) fn relabel_irrep(&self, phi: Irrep) -> Self {
        self.clone().relabel_irrep_into(phi, self.space)
    }

    pub(crate) fn relabel_irrep_into(mut self, phi: Irrep, space: QnSpace) -> Self {
        assert!(
            space.same_lattice(&self.space),
            "relabel_irrep_into: {space:?} is not the same lattice as {:?}",
            self.space
        );
        if phi != Irrep::IDENTITY {
            self.row_ranks.relabel_irrep_in_place(phi);
            self.col_ranks.relabel_irrep_in_place(phi);
            let (old, len) = (self.space, storage_len(&self.space));
            self.store.rekey(|key| {
                let (f, i) = (key as usize / len, key as usize % len);
                let s = SectorGrid::<()>::storage_label(&old, i);
                let s = Sector::with_irrep(s.nup, s.ndn, s.irrep.fuse(phi));
                (f * len + SectorGrid::<()>::storage_index(&old, s)) as u32
            });
        }
        self.space = space;
        self.row_qn = space.bond_sectors(self.k);
        self.col_qn = space.bond_sectors(self.k + 1);
        debug_assert!(self.check_invariants());
        self
    }

    /// Core of a pure (non-entangled) determinant state: rank-1 through the single sector path
    /// entering at row sector `left` (the determinant's occupation of sites `0..k`: electron
    /// counts *and* accumulated point-group irrep) with site occupation `(occ_up, occ_dn)`. The
    /// selected block holds the scalar 1.
    pub fn pure_state(space: QnSpace, k: usize, occ_up: bool, occ_dn: bool, left: Sector) -> Self {
        let s = left;
        let f = Family::from_occupation(occ_up, occ_dn);
        let t = f.apply(s, space.pg.irrep_of(k));
        assert!(
            space.contains(k, s) && space.contains(k + 1, t),
            "pure-state path {s:?} -> {t:?} inadmissible at site {k}"
        );

        let mut row_ranks = RankGrid::zeros(&space);
        let mut col_ranks = RankGrid::zeros(&space);
        row_ranks[s] = 1;
        col_ranks[t] = 1;

        let mut core = Self::zeros(space, k, &row_ranks, &col_ranks);
        let b = core.block_mut(f, s).expect("selected block admissible");
        *b.get_mut(0, 0) = T::one();
        core
    }

    /// The block of `family` at row sector `s`: the stored block, or a zero-size view of its
    /// recorded shape `(row_ranks[s], col_ranks[t])` when the ranks make it empty; `None` when the
    /// transition is inadmissible or `s` is off the sector grid. One index read classifies the slot
    /// (the store records the empty slots, [`Self::zeros`]).
    #[inline]
    pub fn block(&self, family: Family, s: Sector) -> Option<MatRef<'_, T>> {
        if !self.in_grid(s) {
            return None;
        }
        let key = block_key(&self.space, storage_len(&self.space), family, s);
        let (r, c) = match self.store.lookup(key) {
            Lookup::Stored(p) => return Some(self.store.blocks()[p].as_ref()),
            Lookup::Empty(r, c) => (r, c),
            Lookup::Absent => return None,
            Lookup::Unrecorded => self.empty_shape(family, s)?,
        };
        Some(MatRef::from_column_major_slice(&[], r, c))
    }

    /// Mutable block access (the same answers as [`Self::block`]; an empty block's view has no
    /// element to write).
    #[inline]
    pub fn block_mut(&mut self, family: Family, s: Sector) -> Option<MatMut<'_, T>> {
        if !self.in_grid(s) {
            return None;
        }
        let key = block_key(&self.space, storage_len(&self.space), family, s);
        let (r, c) = match self.store.lookup(key) {
            Lookup::Stored(p) => return Some(self.store.blocks_mut()[p].as_mut()),
            Lookup::Empty(r, c) => (r, c),
            Lookup::Absent => return None,
            Lookup::Unrecorded => self.empty_shape(family, s)?,
        };
        Some(MatMut::from_column_major_slice_mut(&mut [], r, c))
    }

    #[inline]
    fn in_grid(&self, s: Sector) -> bool {
        s.nup <= self.space.nup
            && s.ndn <= self.space.ndn
            && s.irrep.index() < self.space.pg.order()
    }

    #[inline]
    fn empty_shape(&self, family: Family, s: Sector) -> Option<(usize, usize)> {
        let t = family.apply(s, self.space.pg.irrep_of(self.k));
        if !(self.space.contains(self.k, s) && self.space.contains(self.k + 1, t)) {
            return None;
        }
        let (r, c) = (self.row_ranks[s], self.col_ranks[t]);
        assert!(
            r * c == 0,
            "site {}: the {r}×{c} block {family:?} at {s:?} is not stored — a rank grid was \
             changed without its blocks",
            self.k
        );
        Some((r, c))
    }

    #[inline]
    pub(crate) fn store(&self) -> &BlockStore<T> {
        &self.store
    }

    #[inline]
    pub(crate) fn store_mut(&mut self) -> &mut BlockStore<T> {
        &mut self.store
    }

    #[inline]
    pub(crate) fn step_range(&self, family: Family) -> Range<usize> {
        self.store
            .step_range(family as usize, storage_len(&self.space))
    }

    #[inline]
    pub(crate) fn stored_label(&self, i: usize) -> (Family, Sector) {
        let len = storage_len(&self.space);
        let key = self.store.keys()[i] as usize;
        (
            Family::ALL[key / len],
            SectorGrid::<()>::storage_label(&self.space, key % len),
        )
    }

    #[inline]
    pub(crate) fn coeff_hint(&self) -> usize {
        let rows: usize = self.row_ranks.values().iter().sum();
        let cmax: usize = self.col_ranks.values().iter().copied().max().unwrap_or(0);
        rows * cmax
    }

    /// Dimensions of the `family` block at row sector `s`, if present.
    #[inline]
    pub fn block_dims(&self, family: Family, s: Sector) -> Option<(usize, usize)> {
        self.block(family, s).map(|m| (m.nrows(), m.ncols()))
    }

    /// Copy the `family` block at row sector `s` into `dst`, **column-major**. Returns the
    /// dimensions, or `None` when the block is absent. Panics if `dst` is shorter than
    /// `rows·cols` (the caller queries [`Self::block_dims`] first).
    pub fn block_copy_out(
        &self,
        family: Family,
        s: Sector,
        dst: &mut [T],
    ) -> Option<(usize, usize)> {
        let m = self.block(family, s)?;
        let (rows, cols) = (m.nrows(), m.ncols());
        assert!(
            dst.len() >= rows * cols,
            "block_copy_out: destination holds {} elements, block is {rows}×{cols}",
            dst.len()
        );
        for j in 0..cols {
            for i in 0..rows {
                dst[j * rows + i] = m[(i, j)];
            }
        }
        Some((rows, cols))
    }

    /// Overwrite the `family` block at row sector `s` from a **column-major** `src`. Returns the
    /// dimensions, or `None` when the block is absent. Panics on a length mismatch.
    pub fn block_copy_in(
        &mut self,
        family: Family,
        s: Sector,
        src: &[T],
    ) -> Option<(usize, usize)> {
        let (rows, cols) = self.block_dims(family, s)?;
        assert_eq!(
            src.len(),
            rows * cols,
            "block_copy_in: source holds {} elements, block is {rows}×{cols}",
            src.len()
        );
        let mut m = self.block_mut(family, s).expect("dims implied presence");
        for j in 0..cols {
            for i in 0..rows {
                m[(i, j)] = src[j * rows + i];
            }
        }
        Some((rows, cols))
    }

    pub(crate) fn copy_block_into(
        &mut self,
        family: Family,
        s: Sector,
        i0: usize,
        j0: usize,
        src: MatRef<'_, T>,
    ) {
        let Some(mut dst) = self.block_mut(family, s) else {
            return;
        };
        assert!(
            i0 + src.nrows() <= dst.nrows() && j0 + src.ncols() <= dst.ncols(),
            "copy_block_into: a {}×{} band at ({i0}, {j0}) does not fit the {}×{} block at \
             {family:?}, {s:?}",
            src.nrows(),
            src.ncols(),
            dst.nrows(),
            dst.ncols()
        );
        for j in 0..src.ncols() {
            for i in 0..src.nrows() {
                *dst.rb_mut().get_mut(i0 + i, j0 + j) = *src.get(i, j);
            }
        }
    }

    /// Row sectors with an admissible `family` transition,
    /// in canonical row-sector order.
    pub fn transitions(&self, family: Family) -> impl Iterator<Item = Sector> + '_ {
        let space = self.space;
        let k = self.k;
        let g = space.pg.irrep_of(k);
        self.row_qn
            .iter()
            .copied()
            .filter(move |&s| space.contains(k + 1, family.apply(s, g)))
    }

    /// Fill every non-degenerate block with i.i.d. draws from the field's standard normal, scaled
    /// by `row_rank^(-1/2)` so each block is an expected isometry `E[BᴴB] = I` (the
    /// non-orthogonalized draw; the per-path second moment then telescopes to 1 along the chain).
    pub fn fill_randn<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let g = self.space.pg.irrep_of(self.k);
        for f in Family::ALL {
            for s in self.transitions(f).collect::<Vec<_>>() {
                let rr = self.row_ranks[s];
                let cr = self.col_ranks[f.apply(s, g)];
                if rr == 0 || cr == 0 {
                    continue;
                }
                let sigma = T::from_f64(1.0 / (rr as f64).sqrt());
                let mut b = self.block_mut(f, s).expect("transition admissible");
                for j in 0..cr {
                    for i in 0..rr {
                        *b.rb_mut().get_mut(i, j) = T::sample_standard(rng) * sigma;
                    }
                }
            }
        }
    }

    /// `orthogonal = :left` Stewart-frame fill. Per **column**
    /// sector `t` at bond `k+1`: draws an `m×c` Stewart-Haar frame (`m` = `vertical_height` at
    /// `t`, zero-padded if infeasible) rescaled by `√(m/c)` and scatters it vertically. Result
    /// is `Vᴴ·V = (m/c)·I` (an "expected-isometry" sketch, NOT bilinear/Hermitian-orthonormal) and
    /// is never flagged orthogonal by the caller.
    pub fn fill_stewart_left<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) {
        let space = self.space;
        let k = self.k;
        for t in space.bond_sectors(k + 1) {
            let c = self.col_ranks[t];
            if c == 0 {
                continue;
            }
            let m = self.vertical_height(t);
            if m == 0 {
                continue;
            }
            let scale = T::from_f64((m as f64 / c as f64).sqrt());
            let cols = c.min(m);
            let frame = crate::core::factor::stewart_frame::<T, R>(rng, m, cols);
            let mut f = Mat::<T>::zeros(m, c);
            for jj in 0..cols {
                for ii in 0..m {
                    *f.get_mut(ii, jj) = *frame.get(ii, jj) * scale;
                }
            }
            self.scatter_vertical(t, f.as_ref());
        }
    }

    /// `orthogonal = :right` Stewart-frame fill. Per **row** sector `l` at bond `k`: draws an `n×r`
    /// Stewart-Haar frame transposed to `r×n` (`n` = `horizontal_width` at `l`, zero-padded if
    /// infeasible) rescaled by `√(n/r)` and scatters it horizontally.
    pub fn fill_stewart_right<R: rand::Rng + ?Sized>(&mut self, rng: &mut R) {
        let space = self.space;
        let k = self.k;
        for l in space.bond_sectors(k) {
            let r = self.row_ranks[l];
            if r == 0 {
                continue;
            }
            let n = self.horizontal_width(l);
            if n == 0 {
                continue;
            }
            let scale = T::from_f64((n as f64 / r as f64).sqrt());
            let cols = r.min(n);
            let frame = crate::core::factor::stewart_frame::<T, R>(rng, n, cols);
            let mut f = Mat::<T>::zeros(r, n);
            for ii in 0..cols {
                for jj in 0..n {
                    *f.get_mut(ii, jj) = *frame.get(jj, ii) * scale;
                }
            }
            self.scatter_horizontal(l, f.as_ref());
        }
    }

    /// Structural invariants: block presence matches admissible transitions and block shapes match
    /// the rank grids. Panics on violation; intended for `debug_assert!(core.check_invariants())`
    /// and tests.
    pub fn check_invariants(&self) -> bool {
        assert_eq!(self.row_qn, self.space.bond_sectors(self.k));
        assert_eq!(self.col_qn, self.space.bond_sectors(self.k + 1));
        let g = self.space.pg.irrep_of(self.k);
        for (i, m) in self.store.blocks().iter().enumerate() {
            let (f, s) = self.stored_label(i);
            let t = f.apply(s, g);
            let admissible = self.space.contains(self.k, s) && self.space.contains(self.k + 1, t);
            assert!(admissible, "site {}: spurious block {f:?} at {s:?}", self.k);
            assert_eq!(
                (m.nrows(), m.ncols()),
                (self.row_ranks[s], self.col_ranks[t]),
                "site {}: block {f:?} at {s:?} has wrong shape",
                self.k
            );
            assert!(
                m.nrows() * m.ncols() > 0,
                "site {}: block {f:?} at {s:?} stores an empty block",
                self.k
            );
        }
        let len = storage_len(&self.space);
        let recording = self.store.records_empty();
        for f in Family::ALL {
            for s in self.transitions(f) {
                let t = f.apply(s, g);
                let (r, c) = (self.row_ranks[s], self.col_ranks[t]);
                let found = self.store.lookup(block_key(&self.space, len, f, s));
                let ok = if r * c > 0 {
                    matches!(found, Lookup::Stored(_))
                } else {
                    !recording || found == Lookup::Empty(r, c)
                };
                assert!(ok, "site {}: block {f:?} at {s:?} has wrong shape", self.k);
            }
        }
        for (key, _, _) in self.store.recorded_empties() {
            let (f, s) = (
                Family::ALL[key as usize / len],
                SectorGrid::<()>::storage_label(&self.space, key as usize % len),
            );
            let t = f.apply(s, g);
            assert!(
                self.space.contains(self.k, s) && self.space.contains(self.k + 1, t),
                "site {}: spurious block {f:?} at {s:?}",
                self.k
            );
        }
        true
    }
}

impl<T: Scalar> crate::mem::Footprint for SparseCore<T> {
    fn bytes(&self) -> u64 {
        self.store
            .blocks()
            .iter()
            .map(|m| crate::mem::block_bytes::<T>(m.nrows(), m.ncols()))
            .sum()
    }
}

#[cfg(test)]
mod pg2_tests {
    use super::Family;
    use crate::core::factor::adjoint_core_times_core;
    use crate::core::twosite::contract;
    use crate::pointgroup::Irrep;
    use crate::qn::{QnSpace, RankGrid, Sector};
    use crate::{Sketch, TtVector, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn family_flux_matches_campaign_design_per_family() {
        for gi in 0u8..8 {
            let g = Irrep::new(gi);
            assert_eq!(
                Family::Empty.flux(g),
                Irrep::IDENTITY,
                "Empty site: no flux"
            );
            assert_eq!(
                Family::Occ.flux(g),
                Irrep::IDENTITY,
                "doubly-occupied site: flux g.fuse(g) must cancel to IDENTITY"
            );
            assert_eq!(
                Family::Up.flux(g),
                g,
                "singly-occupied (up) site: flux is g"
            );
            assert_eq!(
                Family::Dn.flux(g),
                g,
                "singly-occupied (dn) site: flux is g"
            );
        }
    }

    #[test]
    fn family_apply_unapply_round_trip_all_families_all_irreps() {
        for f in Family::ALL {
            let (du, dd) = f.delta();
            for gi in 0u8..8 {
                let g = Irrep::new(gi);
                for nup in 0..=4usize {
                    for ndn in 0..=4usize {
                        for iri in 0u8..8 {
                            let s = Sector::with_irrep(nup, ndn, Irrep::new(iri));
                            let t = f.apply(s, g);
                            assert_eq!(
                                (t.nup, t.ndn),
                                (s.nup + du, s.ndn + dd),
                                "f={f:?} g={g:?} s={s:?}: apply's particle shift must match delta()"
                            );
                            assert_eq!(
                                f.unapply(t, g),
                                Some(s),
                                "f={f:?} g={g:?} s={s:?}: unapply(apply(s,g),g) must round-trip to s"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn family_unapply_returns_none_on_particle_underflow() {
        let t = Sector::with_irrep(0, 0, Irrep::new(2));
        for f in [Family::Up, Family::Dn, Family::Occ] {
            for gi in 0u8..8 {
                let g = Irrep::new(gi);
                assert_eq!(
                    f.unapply(t, g),
                    None,
                    "f={f:?} g={g:?}: unapply must return None on particle underflow, not panic or \
                     wrap"
                );
            }
        }
    }

    #[test]
    fn family_apply_reduces_to_pre_pg2_behavior_at_trivial_irrep() {
        for f in Family::ALL {
            let (du, dd) = f.delta();
            for nup in 0..=5usize {
                for ndn in 0..=5usize {
                    let s = Sector::new(nup, ndn);
                    let t = f.apply(s, Irrep::IDENTITY);
                    assert_eq!(
                        t,
                        Sector::new(nup + du, ndn + dd),
                        "f={f:?} s={s:?}: apply at |G|=1 must reduce exactly to the pre-PG2 result"
                    );
                }
            }
        }
    }

    #[test]
    fn block_accessors_reject_out_of_range_irrep_even_at_trivial_group() {
        let d = 4;
        let space = QnSpace::new(d, 2, 1);
        let out_of_range = Sector::with_irrep(0, 0, Irrep::new(1));

        let mut ranks = Vec::with_capacity(d + 1);
        for b in 0..=d {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = 1;
            }
            ranks.push(r);
        }
        let x: TtVector<f64> = tt_randn(space, &ranks, &mut Xoshiro256PlusPlus::seed_from_u64(1));

        assert_eq!(
            x.core(0).block(Family::Empty, out_of_range),
            None,
            "SparseCore::block must reject an out-of-range irrep even at |G|=1"
        );

        let sketch: Sketch<f64> = Sketch::identity(space, 0, 1);
        assert_eq!(
            sketch.block(out_of_range),
            None,
            "Sketch::block must reject an out-of-range irrep even at |G|=1"
        );

        let sk2 = adjoint_core_times_core(x.core(0), x.core(0));
        assert_eq!(
            sk2.block(out_of_range),
            None,
            "adjoint_core_times_core's Sketch::block must reject an out-of-range irrep even at |G|=1"
        );

        let tc = contract(x.core(0), x.core(1));
        assert_eq!(
            tc.block(out_of_range),
            None,
            "ContractedSparseCores::block must reject an out-of-range irrep even at |G|=1"
        );
    }
}
