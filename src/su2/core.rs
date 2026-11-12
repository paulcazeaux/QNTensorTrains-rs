// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) block-sparse TT core.

mod coeffs;
#[cfg(test)]
mod from_fn_tests;
mod unfold;

use std::borrow::Cow;
use std::ops::Range;

use faer::prelude::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use rand::Rng;

use crate::blockstore::BlockStore;
use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::storage_probe;
use crate::su2::qn::{
    LabelTables, Slot, StateSym, Su2RankGrid, Su2Sector, Su2Space, SymGrid, SymSpace, SymStep,
    Symmetry, for_each_nonempty_slot, for_each_slot, storage_labels,
};

#[inline]
fn storage_len<S: Symmetry>(space: &S::Space) -> usize {
    <S::Grid<()> as SymGrid<()>>::storage_len(space)
}

#[inline]
fn block_key<S: Symmetry>(space: &S::Space, len: usize, f: S::Step, s: S::Label) -> u32 {
    (f.index() * len + <S::Grid<()> as SymGrid<()>>::storage_index(space, s)) as u32
}

fn check_admissible_ranks<S: Symmetry>(
    space: &S::Space,
    b: usize,
    labels: &[S::Label],
    ranks: &S::Grid<usize>,
    side: &str,
) {
    let all = ranks.values().iter().filter(|&&r| r > 0).count();
    let on = labels.iter().filter(|&&l| ranks[l] > 0).count();
    if all != on {
        for (s, &r) in ranks.iter() {
            assert!(
                r == 0 || space.admissible(b, s),
                "positive {side} rank {r} at inadmissible sector {s:?} of bond {b}"
            );
        }
    }
}

fn add_rank_grids<G: SymGrid<usize> + Clone>(a: &G, b: &G) -> G {
    let mut r = a.clone();
    for (s, &v) in b.iter() {
        r[s] += v;
    }
    r
}

/// The four block families of an SU(2) core, by genealogical step. The discriminant is the physical
/// index: 0 = empty, 1 = up (S+½ branch), 2 = dn (S−½ branch), 3 = doubly occupied (on-site
/// singlet). **`Up`/`Dn` are S±½ coupling BRANCHES of one electron, not electron spins** — see the
/// module docs above.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Family {
    /// Site unoccupied (`○○`): column sector = row sector.
    Empty = 0,
    /// One electron, **S+½ branch** (`up`): column sector = row + `(+1, +1)`.
    Up = 1,
    /// One electron, **S−½ branch** (`dn`): column sector = row + `(+1, −1)`.
    Dn = 2,
    /// On-site singlet pair (`●●`): column sector = row + `(+2, 0)`.
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

    /// Sector shift `(Δn, Δ(2s))` this family applies across the site. `Δ(2s)` is signed: `Dn`
    /// is the only family with a negative component.
    #[inline]
    pub fn delta(self) -> (usize, i64) {
        match self {
            Family::Empty => (0, 0),
            Family::Up => (1, 1),
            Family::Dn => (1, -1),
            Family::Occ => (2, 0),
        }
    }

    /// Point-group irrep flux across a site whose orbital irrep is `g`. **`Up`/`Dn` are the S±½
    /// coupling BRANCHES of adding ONE electron, not electron spins** (module docs): both occupy
    /// the site with exactly one electron, so both carry `g` — not the abelian "one spin-orbital
    /// each" reading, but the same electron counted twice by coupling channel. `Occ` is the
    /// on-site singlet pair, `g ⊗ g` (= [`Irrep::IDENTITY`] for the `Z2^m` groups we support).
    #[inline]
    pub const fn flux(self, g: Irrep) -> Irrep {
        match self {
            Family::Empty => Irrep::IDENTITY,
            Family::Up | Family::Dn => g,
            Family::Occ => g.fuse(g),
        }
    }

    /// Column sector reached from row sector `s` across a site of irrep `g`, or `None` if the
    /// doubled-spin axis would go negative (the `Dn` branch out of a spin-0 sector, `two_s == 0`)
    /// — the underflow guard. The irrep half is
    /// `s.irrep ⊗ flux(g)`; `Up` and `Dn` both fuse `g` because both are one-electron COUPLING
    /// BRANCHES (module docs), not spin channels. Does NOT check grid bounds or
    /// [`Su2Space::contains`]; callers combine this with an admissibility check.
    #[inline]
    pub fn apply(self, s: Su2Sector, g: Irrep) -> Option<Su2Sector> {
        let (dn, dt) = self.delta();
        let two_s = s.two_s as i64 + dt;
        if two_s < 0 {
            return None;
        }
        Some(Su2Sector::with_irrep(
            s.n + dn,
            two_s as usize,
            s.irrep.fuse(self.flux(g)),
        ))
    }

    /// Inverse of [`apply`](Self::apply) ACROSS A SITE OF IRREP `g`: the row sector that reaches
    /// column sector `t` under this family, or `None` when the inversion underflows — `t.n` below
    /// the family's `Δn` (no incoming transition of this family), or `t.two_s == 0` for `Up` (the
    /// `S+½` branch cannot land on a spin-0 multiplet). The `None` never means irrep
    /// unreachability: the irrep half is total, `t.irrep ⊗ flux(g)⁻¹`.
    #[inline]
    pub fn unapply(self, t: Su2Sector, g: Irrep) -> Option<Su2Sector> {
        let (dn, dt) = self.delta();
        let two_s = t.two_s as i64 - dt;
        if two_s < 0 {
            return None;
        }
        Some(Su2Sector::with_irrep(
            t.n.checked_sub(dn)?,
            two_s as usize,
            t.irrep.fuse(self.flux(g).inverse()),
        ))
    }
}

/// Per-row-sector block storage of one family: `None` where the transition leaves the
/// admissible sector cone (or underflows, `Family::apply` returns `None`), `Some` (possibly with
/// a zero dimension) where it does not. `S` is the label table ([`StateSym`] by default, i.e.
/// `Su2SectorGrid<Option<Mat<T>>>`).
pub type BlockGrid<T, S = StateSym> = <S as Symmetry>::Grid<Option<Mat<T>>>;

macro_rules! assert_grid_eq {
    ($a:expr, $b:expr, $($arg:tt)+) => {{
        let (left, right) = (&$a, &$b);
        if !$crate::su2::qn::SymGrid::eq_grid(left, right) {
            panic!(
                "assertion `left == right` failed: {}\n  left: {:?}\n right: {:?}",
                format_args!($($arg)+),
                left,
                right
            );
        }
    }};
}
pub(crate) use assert_grid_eq;

fn owned_grid<G: Clone>(g: Cow<'_, G>) -> G {
    match g {
        Cow::Borrowed(g) => g.clone(),
        Cow::Owned(g) => g,
    }
}

/// One SU(2) block-sparse TT core. Stores Wigner-Eckart REDUCED matrix elements: the block of
/// `(family, s)` is the reduced block of the fusion `row sector s ⊗ site → col sector
/// family.apply(s, g_k)`, `g_k` the site's orbital irrep.
#[derive(Clone, Debug)]
pub struct Su2Core<T: Scalar, S: Symmetry = StateSym> {
    /// Symmetry descriptor shared by the whole train.
    pub space: S::Space,
    /// Site index, `0..d`.
    pub k: usize,
    /// Admissible row sectors (bond `k`), canonical order.
    pub row_qn: Vec<S::Label>,
    /// Admissible column sectors (bond `k+1`), canonical order.
    pub col_qn: Vec<S::Label>,
    /// Per-sector row ranks (zero at inadmissible sectors). Reduced rank = number of
    /// multiplets, NOT number of abelian states.
    pub row_ranks: S::Grid<usize>,
    /// Per-sector column ranks (zero at inadmissible sectors).
    pub col_ranks: S::Grid<usize>,
    store: BlockStore<T>,
}

impl<T: Scalar> Su2Core<T> {
    /// Zero-initialized core with the given bond ranks. Positive ranks are only permitted on
    /// admissible sectors.
    #[inline]
    pub fn zeros(
        space: Su2Space,
        k: usize,
        row_ranks: &Su2RankGrid,
        col_ranks: &Su2RankGrid,
    ) -> Self {
        Self::zeros_sym(space, k, row_ranks, col_ranks)
    }

    /// Construct a core directly from its four block grids (`Family::ALL` order) — the state
    /// façade of [`Self::from_blocks_sym`], which documents the derivation.
    #[inline]
    pub fn from_blocks(space: Su2Space, k: usize, blocks: [BlockGrid<T>; 4]) -> Self {
        Self::from_blocks_sym(space, k, Vec::from(blocks))
    }

    /// Single genealogical coupling step: the sole `1×1` block of `family` at `entering` (the
    /// sector ENTERING site `k`) is set to `1` — the exact reduced matrix element of a genealogical
    /// step in the orthonormal-CG convention. Building block of `tt_csf`. Both `entering` and its
    /// `family`-shifted destination must be admissible. `entering` carries the CUMULATIVE
    /// irrep of sites `0..k` — caller-threaded, since the core never sees those sites; the
    /// destination irrep is `entering.irrep ⊗ family.flux(pg.irrep_of(k))`. Riding the whole sector
    /// rather than a separate irrep argument makes it unforgettable (abelian `pure_state`
    /// precedent).
    pub fn csf_step(space: Su2Space, k: usize, family: Family, entering: Su2Sector) -> Self {
        let s = entering;
        let t = family
            .apply(s, space.pg.irrep_of(k))
            .filter(|&t| space.contains(k, s) && space.contains(k + 1, t))
            .unwrap_or_else(|| {
                panic!("csf step {family:?} from {s:?} inadmissible at site {k}");
            });

        let mut row_ranks = Su2RankGrid::zeros(&space);
        let mut col_ranks = Su2RankGrid::zeros(&space);
        row_ranks[s] = 1;
        col_ranks[t] = 1;

        let mut core = Self::zeros(space, k, &row_ranks, &col_ranks);
        let mut b = core
            .block_mut(family, s)
            .expect("selected block admissible");
        *b.rb_mut().get_mut(0, 0) = T::one();
        core
    }
}

impl<T: Scalar, S: Symmetry> Su2Core<T, S> {
    /// Zero-initialized core with the given bond ranks (generic twin of [`Su2Core::zeros`]).
    /// Positive ranks are only permitted on admissible sectors.
    pub fn zeros_sym(
        space: S::Space,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
    ) -> Self {
        Self::from_fn_nonempty_sym(space, k, row_ranks, col_ranks, |_, _, _| None)
    }

    #[allow(dead_code)]
    pub(crate) fn from_fn_sym(
        space: S::Space,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
        fill: impl FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>,
    ) -> Self {
        let d = space.sites();
        assert!(k < d, "site index {k} out of range 0..{d}");
        let row_qn = space.bond_labels(k);
        let col_qn = space.bond_labels(k + 1);
        let rows = storage_labels::<S>(&space, k);
        let g = space.site_ctx(k);
        let slots = S::STEPS.iter().flat_map(|&f| {
            rows.iter().filter_map(move |&s| {
                f.step_apply(s, g)
                    .filter(|&t| space.admissible(k + 1, t))
                    .map(|t| (f, s, t))
            })
        });
        Self::assemble(space, k, row_qn, col_qn, row_ranks, col_ranks, slots, fill)
    }

    #[allow(dead_code)]
    #[inline]
    pub(crate) fn zeros_in(
        tables: &LabelTables<S>,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
    ) -> Self {
        Self::from_fn_nonempty_in(tables, k, row_ranks, col_ranks, |_, _, _| None)
    }

    #[allow(dead_code)]
    pub(crate) fn from_fn_nonempty_in(
        tables: &LabelTables<S>,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
        fill: impl FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>,
    ) -> Self {
        Self::from_fn_nonempty_in_cow(
            tables,
            k,
            Cow::Borrowed(row_ranks),
            Cow::Borrowed(col_ranks),
            fill,
        )
    }

    pub(crate) fn from_fn_nonempty_in_cow(
        tables: &LabelTables<S>,
        k: usize,
        row_ranks: Cow<'_, S::Grid<usize>>,
        col_ranks: Cow<'_, S::Grid<usize>>,
        fill: impl FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>,
    ) -> Self {
        let space = *tables.space();
        let d = space.sites();
        assert!(k < d, "site index {k} out of range 0..{d}");
        Self::assemble_nonempty(
            space,
            k,
            tables.labels(k).to_vec(),
            tables.labels(k + 1).to_vec(),
            tables.storage_labels(k),
            row_ranks,
            col_ranks,
            fill,
        )
    }

    pub(crate) fn from_fn_nonempty_sym(
        space: S::Space,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
        fill: impl FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>,
    ) -> Self {
        Self::from_fn_nonempty_sym_cow(
            space,
            k,
            Cow::Borrowed(row_ranks),
            Cow::Borrowed(col_ranks),
            fill,
        )
    }

    pub(crate) fn from_fn_nonempty_sym_cow(
        space: S::Space,
        k: usize,
        row_ranks: Cow<'_, S::Grid<usize>>,
        col_ranks: Cow<'_, S::Grid<usize>>,
        fill: impl FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>,
    ) -> Self {
        let d = space.sites();
        assert!(k < d, "site index {k} out of range 0..{d}");
        let rows = storage_labels::<S>(&space, k);
        Self::assemble_nonempty(
            space,
            k,
            space.bond_labels(k),
            space.bond_labels(k + 1),
            &rows,
            row_ranks,
            col_ranks,
            fill,
        )
    }

    pub(crate) fn from_entries_sym(
        space: S::Space,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
        entries: Vec<(S::Step, S::Label, Mat<T>)>,
    ) -> Self {
        let mut it = Self::keyed_entries(&space, k, row_ranks, col_ranks, entries);
        let core = Self::from_fn_nonempty_sym(space, k, row_ranks, col_ranks, |f, s, _| {
            Self::take_entry(&space, &mut it, f, s)
        });
        Self::assert_entries_placed(&space, k, it);
        core
    }

    fn keyed_entries(
        space: &S::Space,
        k: usize,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
        entries: Vec<(S::Step, S::Label, Mat<T>)>,
    ) -> std::iter::Peekable<std::vec::IntoIter<(u32, Mat<T>)>> {
        let (len, g) = (storage_len::<S>(space), space.site_ctx(k));
        let mut keyed: Vec<(u32, Mat<T>)> = Vec::with_capacity(entries.len());
        for (f, s, m) in entries {
            let t = (space.in_grid(s) && space.admissible(k, s))
                .then(|| f.step_apply(s, g))
                .flatten()
                .filter(|&t| space.admissible(k + 1, t));
            let Some(t) = t else {
                panic!(
                    "Su2Core::from_entries: the {}×{} block {f:?} at {s:?} is no admissible block \
                     of site {k}",
                    m.nrows(),
                    m.ncols()
                );
            };
            assert_eq!(
                (m.nrows(), m.ncols()),
                (row_ranks[s], col_ranks[t]),
                "Su2Core::from_entries: block {f:?} at {s:?} has the wrong shape"
            );
            if m.nrows() * m.ncols() > 0 {
                keyed.push((block_key::<S>(space, len, f, s), m));
            }
        }
        keyed.sort_unstable_by_key(|&(key, _)| key);
        for w in keyed.windows(2) {
            assert!(
                w[0].0 != w[1].0,
                "Su2Core::from_entries: two blocks for one slot (key {}) at site {k}",
                w[0].0
            );
        }
        keyed.into_iter().peekable()
    }

    #[inline]
    fn take_entry(
        space: &S::Space,
        it: &mut std::iter::Peekable<std::vec::IntoIter<(u32, Mat<T>)>>,
        f: S::Step,
        s: S::Label,
    ) -> Option<Mat<T>> {
        let key = block_key::<S>(space, storage_len::<S>(space), f, s);
        if it.peek().is_some_and(|&(k2, _)| k2 == key) {
            it.next().map(|(_, m)| m)
        } else {
            None
        }
    }

    fn assert_entries_placed(
        space: &S::Space,
        k: usize,
        mut it: std::iter::Peekable<std::vec::IntoIter<(u32, Mat<T>)>>,
    ) {
        if let Some((key, m)) = it.next() {
            let len = storage_len::<S>(space);
            panic!(
                "Su2Core::from_entries: the {}×{} block {:?} at {:?} is no non-empty admissible \
                 block of site {k}",
                m.nrows(),
                m.ncols(),
                S::STEPS[key as usize / len],
                <S::Grid<()> as SymGrid<()>>::storage_label(space, key as usize % len),
            );
        }
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        space: S::Space,
        k: usize,
        row_qn: Vec<S::Label>,
        col_qn: Vec<S::Label>,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
        slots: impl IntoIterator<Item = Slot<S>>,
        mut fill: impl FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>,
    ) -> Self {
        check_admissible_ranks::<S>(&space, k, &row_qn, row_ranks, "row");
        check_admissible_ranks::<S>(&space, k + 1, &col_qn, col_ranks, "column");
        let len = storage_len::<S>(&space);
        let mut entries: Vec<(u32, Mat<T>)> = Vec::new();
        let mut tally = storage_probe::CoreTally::new();
        for (f, s, t) in slots {
            let (r, c) = (row_ranks[s], col_ranks[t]);
            let m = match fill(f, s, t) {
                Some(m) => {
                    assert_eq!(
                        (m.nrows(), m.ncols()),
                        (r, c),
                        "Su2Core::from_fn: block {f:?} at {s:?} has the wrong shape"
                    );
                    if r * c > 0 {
                        tally.filled();
                    }
                    Some(m)
                }
                None => (r * c > 0).then(|| Mat::zeros(r, c)),
            };
            tally.block(r, c);
            if r * c > 0 {
                entries.push((
                    block_key::<S>(&space, len, f, s),
                    m.expect("non-empty block"),
                ));
            }
        }
        let store = BlockStore::from_sorted(S::STEPS.len() * len, entries);
        if storage_probe::ENABLED {
            let table = storage_probe::Table::su2(S::STEPS.len());
            let kind = storage_probe::CoreKind::Assembled;
            tally.finish(kind, table, k, 0, store.len(), store.dense_index_len());
        }
        Su2Core {
            space,
            k,
            row_qn,
            col_qn,
            row_ranks: row_ranks.clone(),
            col_ranks: col_ranks.clone(),
            store,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn assemble_nonempty(
        space: S::Space,
        k: usize,
        row_qn: Vec<S::Label>,
        col_qn: Vec<S::Label>,
        rows_storage: &[S::Label],
        row_ranks: Cow<'_, S::Grid<usize>>,
        col_ranks: Cow<'_, S::Grid<usize>>,
        mut fill: impl FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>,
    ) -> Self {
        let (row_ranks, col_ranks) = (owned_grid(row_ranks), owned_grid(col_ranks));
        check_admissible_ranks::<S>(&space, k, &row_qn, &row_ranks, "row");
        check_admissible_ranks::<S>(&space, k + 1, &col_qn, &col_ranks, "column");
        let len = storage_len::<S>(&space);
        let mut entries: Vec<(u32, Mat<T>)> = Vec::new();
        let mut tally = storage_probe::CoreTally::new();
        for_each_nonempty_slot::<S>(
            &space,
            k,
            rows_storage,
            &row_ranks,
            &col_ranks,
            |f, s, t| {
                let (r, c) = (row_ranks[s], col_ranks[t]);
                let m = match fill(f, s, t) {
                    Some(m) => {
                        assert_eq!(
                            (m.nrows(), m.ncols()),
                            (r, c),
                            "Su2Core::from_fn: block {f:?} at {s:?} has the wrong shape"
                        );
                        tally.filled();
                        m
                    }
                    None => Mat::zeros(r, c),
                };
                entries.push((block_key::<S>(&space, len, f, s), m));
            },
        );
        let store = BlockStore::from_sorted(S::STEPS.len() * len, entries);
        if storage_probe::ENABLED {
            for_each_slot::<S>(&space, k, rows_storage, |_, s, t| {
                tally.block(row_ranks[s], col_ranks[t])
            });
            let table = storage_probe::Table::su2(S::STEPS.len());
            let kind = storage_probe::CoreKind::Assembled;
            tally.finish(kind, table, k, 0, store.len(), store.dense_index_len());
        }
        Su2Core {
            space,
            k,
            row_qn,
            col_qn,
            row_ranks,
            col_ranks,
            store,
        }
    }

    /// Fill every non-degenerate block with i.i.d. standard-normal draws scaled by
    /// `row_rank^(-1/2)` so each block is an expected isometry `E[BᴴB] = I` (the
    /// the Gaussian fill). The unweighted reduced metric here IS the fixed-Sz inner
    /// product — no `θ`/`centerweight` (that is the whole point of the orthonormal-CG convention).
    /// The Stewart-frame modes are [`Su2Core::fill_stewart_left`]/[`Su2Core::fill_stewart_right`].
    pub fn fill_randn<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            for s in self.transitions(f).collect::<Vec<_>>() {
                let rr = self.row_ranks[s];
                let t = f.step_apply(s, g).expect("transition admissible");
                let cr = self.col_ranks[t];
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

    /// Construct a core directly from its block grids (one per `S::STEPS` entry, in that order),
    /// deriving `row_ranks`/`col_ranks` from block shapes and asserting block presence exactly
    /// matches the admissible-transition sets (generic twin of [`Su2Core::from_blocks`]).
    /// **The column-rank derivation uses the shifted-sector ADJOINT lookups: `up` from
    /// `(n−1, 2s−1)`, `dn` from `(n−1, 2s+1)` — note the `+1` on `2s`, the inverse of `dn`'s own
    /// `−1` shift. Getting this sign wrong is a silent structural bug, and
    /// the irrep half is fused back out with `Family::flux(g).inverse()`.**
    pub fn from_blocks_sym(space: S::Space, k: usize, blocks: Vec<BlockGrid<T, S>>) -> Self {
        let d = space.sites();
        assert!(k < d, "site index {k} out of range 0..{d}");
        assert_eq!(
            blocks.len(),
            S::STEPS.len(),
            "Su2Core::from_blocks: one block grid per step"
        );
        let g = space.site_ctx(k);
        let row_qn = space.bond_labels(k);
        let col_qn = space.bond_labels(k + 1);

        for &f in S::STEPS {
            for (s, b) in blocks[f.index()].iter() {
                let admissible = space.admissible(k, s)
                    && f.step_apply(s, g)
                        .is_some_and(|t| space.admissible(k + 1, t));
                assert_eq!(
                    b.is_some(),
                    admissible,
                    "site {k}: block {f:?} presence at {s:?} disagrees with admissibility"
                );
            }
        }

        let mut row_ranks = <S::Grid<usize>>::filled(&space, 0);
        for &s in &row_qn {
            let mut found: Option<usize> = None;
            for &f in S::STEPS {
                if let Some(m) = blocks[f.index()][s].as_ref() {
                    match found {
                        None => {
                            found = Some(m.nrows());
                            row_ranks[s] = m.nrows();
                        }
                        Some(r) => assert_eq!(
                            r,
                            m.nrows(),
                            "site {k}: row-rank disagreement at {s:?} across families"
                        ),
                    }
                }
            }
        }

        let mut col_ranks = <S::Grid<usize>>::filled(&space, 0);
        for &t in &col_qn {
            let mut found: Option<usize> = None;
            for &f in S::STEPS {
                let Some(src) = f.step_unapply(t, g) else {
                    continue;
                };
                let grid = &blocks[f.index()];
                if !grid.in_bounds(src) {
                    continue;
                }
                if let Some(m) = grid[src].as_ref() {
                    match found {
                        None => {
                            found = Some(m.ncols());
                            col_ranks[t] = m.ncols();
                        }
                        Some(c) => assert_eq!(
                            c,
                            m.ncols(),
                            "site {k}: col-rank disagreement at {t:?} across incoming families"
                        ),
                    }
                }
            }
        }

        let len = storage_len::<S>(&space);
        let mut tally = storage_probe::CoreTally::new();
        let mut entries: Vec<(u32, Mat<T>)> = Vec::new();
        for (fi, mut grid) in blocks.into_iter().enumerate() {
            for (l, slot) in grid.iter_mut() {
                if let Some(m) = slot.take() {
                    tally.block(m.nrows(), m.ncols());
                    if m.nrows() * m.ncols() > 0 {
                        let f = S::STEPS[fi];
                        entries.push((block_key::<S>(&space, len, f, l), m));
                    }
                }
            }
        }
        let store = BlockStore::from_sorted(S::STEPS.len() * len, entries);
        if storage_probe::ENABLED {
            let table = storage_probe::Table::su2(S::STEPS.len());
            let kind = storage_probe::CoreKind::FromBlocks;
            tally.finish(kind, table, k, 0, store.len(), store.dense_index_len());
        }
        Su2Core {
            space,
            k,
            row_qn,
            col_qn,
            row_ranks,
            col_ranks,
            store,
        }
    }

    /// The block of `family` at row sector `s`, if the transition is admissible: the stored block,
    /// or a zero-size view of shape `(row_ranks[s], col_ranks[t])` when the ranks make it empty;
    /// `None` when the transition is inadmissible or `s` is off the label grid (`up`/`●●` can shift
    /// `n` past `N`, `dn` can drop `2s` below 0, a foreign irrep can exceed the group).
    #[inline]
    pub fn block(&self, family: S::Step, s: S::Label) -> Option<MatRef<'_, T>> {
        if !self.space.in_grid(s) {
            return None;
        }
        let len = storage_len::<S>(&self.space);
        if let Some(m) = self.store.get(block_key::<S>(&self.space, len, family, s)) {
            return Some(m);
        }
        let (r, c) = self.empty_shape(family, s)?;
        Some(MatRef::from_column_major_slice(&[], r, c))
    }

    /// Mutable block access (the same answers as [`Self::block`]; an empty slot's view has no
    /// element to write).
    #[inline]
    pub fn block_mut(&mut self, family: S::Step, s: S::Label) -> Option<MatMut<'_, T>> {
        if !self.space.in_grid(s) {
            return None;
        }
        let len = storage_len::<S>(&self.space);
        if let Some(p) = self
            .store
            .position(block_key::<S>(&self.space, len, family, s))
        {
            return Some(self.store.blocks_mut()[p].as_mut());
        }
        let (r, c) = self.empty_shape(family, s)?;
        Some(MatMut::from_column_major_slice_mut(&mut [], r, c))
    }

    #[inline]
    fn empty_shape(&self, family: S::Step, s: S::Label) -> Option<(usize, usize)> {
        let t = family.step_apply(s, self.space.site_ctx(self.k))?;
        if !(self.space.admissible(self.k, s) && self.space.admissible(self.k + 1, t)) {
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
    pub(crate) fn stored_at(&self, family: S::Step, s: S::Label) -> Option<MatRef<'_, T>> {
        debug_assert!(self.space.in_grid(s), "stored_at: {s:?} off the label grid");
        let len = storage_len::<S>(&self.space);
        self.store.get(block_key::<S>(&self.space, len, family, s))
    }

    #[inline]
    pub(crate) fn stored_at_mut(&mut self, family: S::Step, s: S::Label) -> Option<MatMut<'_, T>> {
        debug_assert!(
            self.space.in_grid(s),
            "stored_at_mut: {s:?} off the label grid"
        );
        let len = storage_len::<S>(&self.space);
        let p = self
            .store
            .position(block_key::<S>(&self.space, len, family, s))?;
        Some(self.store.blocks_mut()[p].as_mut())
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
    pub(crate) fn step_range(&self, family: S::Step) -> Range<usize> {
        self.store
            .step_range(family.index(), storage_len::<S>(&self.space))
    }

    #[inline]
    pub(crate) fn stored_label(&self, i: usize) -> (S::Step, S::Label) {
        let len = storage_len::<S>(&self.space);
        let key = self.store.keys()[i] as usize;
        (
            S::STEPS[key / len],
            <S::Grid<()> as SymGrid<()>>::storage_label(&self.space, key % len),
        )
    }

    pub(crate) fn stored_mut(&mut self) -> Vec<(S::Step, S::Label, MatMut<'_, T>)> {
        let len = storage_len::<S>(&self.space);
        let space = self.space;
        let keys: Vec<u32> = self.store.keys().to_vec();
        keys.into_iter()
            .zip(self.store.blocks_mut().iter_mut())
            .map(|(key, m)| {
                let key = key as usize;
                (
                    S::STEPS[key / len],
                    <S::Grid<()> as SymGrid<()>>::storage_label(&space, key % len),
                    m.as_mut(),
                )
            })
            .collect()
    }

    #[inline]
    pub(crate) fn coeff_hint(&self) -> usize {
        let rows: usize = self.row_ranks.values().iter().sum();
        let cmax: usize = self.col_ranks.values().iter().copied().max().unwrap_or(0);
        rows * cmax
    }

    /// Dimensions of the `family` block at row sector `s`, if present (C-ABI twin of
    /// [`crate::core::SparseCore::block_dims`]).
    #[inline]
    pub fn block_dims(&self, family: S::Step, s: S::Label) -> Option<(usize, usize)> {
        self.block(family, s).map(|m| (m.nrows(), m.ncols()))
    }

    /// Copy the `family` block at row sector `s` into `dst`, **column-major** (C-ABI twin of
    /// [`crate::core::SparseCore::block_copy_out`]).
    pub fn block_copy_out(
        &self,
        family: S::Step,
        s: S::Label,
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

    /// Overwrite the `family` block at row sector `s` from a **column-major** `src` (C-ABI twin
    /// of [`crate::core::SparseCore::block_copy_in`]).
    pub fn block_copy_in(
        &mut self,
        family: S::Step,
        s: S::Label,
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
        family: S::Step,
        s: S::Label,
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
                dst[(i0 + i, j0 + j)] = src[(i, j)];
            }
        }
    }

    /// Row sectors with an admissible `family` transition, in canonical row-sector order. Guards
    /// the `Dn` doubled-spin underflow via [`Family::apply`] before checking
    /// [`Su2Space::contains`].
    pub fn transitions(&self, family: S::Step) -> impl Iterator<Item = S::Label> + '_ {
        let space = self.space;
        let k = self.k;
        let g = space.site_ctx(k);
        self.row_qn.iter().copied().filter(move |&s| {
            family
                .step_apply(s, g)
                .is_some_and(|t| space.admissible(k + 1, t))
        })
    }

    /// Structural invariants: block presence matches admissible transitions (including the `Dn`
    /// underflow guard) and block shapes match the rank grids. Panics on violation; intended for
    /// `debug_assert!(core.check_invariants())` and tests. The point-group flux law needs no
    /// assertion of its own: the column rank is read at the FUSED sector `t = f.apply(s, g_k)`, so
    /// a block whose columns came from the wrong irrep plane fails the shape assert below.
    pub fn check_invariants(&self) -> bool {
        assert_eq!(self.row_qn, self.space.bond_labels(self.k));
        assert_eq!(self.col_qn, self.space.bond_labels(self.k + 1));
        let g = self.space.site_ctx(self.k);
        for (i, m) in self.store.blocks().iter().enumerate() {
            let (f, s) = self.stored_label(i);
            let t = f.step_apply(s, g).filter(|&t| {
                self.space.admissible(self.k, s) && self.space.admissible(self.k + 1, t)
            });
            let Some(t) = t else {
                panic!("site {}: spurious block {f:?} at {s:?}", self.k);
            };
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
        let len = storage_len::<S>(&self.space);
        for &f in S::STEPS {
            for s in self.transitions(f) {
                let t = f.step_apply(s, g).expect("transition admissible");
                let (r, c) = (self.row_ranks[s], self.col_ranks[t]);
                if r * c > 0 {
                    assert!(
                        self.store
                            .position(block_key::<S>(&self.space, len, f, s))
                            .is_some(),
                        "site {}: block {f:?} at {s:?} has wrong shape",
                        self.k
                    );
                }
            }
        }
        true
    }

    /// Block-concatenation add `A ⊕ B`: UNWEIGHTED — pure block concatenation, no θ/centerweight,
    /// identical shape to the abelian `oplus`. Boundary cores concatenate only along the
    /// non-boundary bond.
    pub fn oplus(&self, other: &Self) -> Self {
        assert_eq!(self.space, other.space, "oplus: Su2Space mismatch");
        assert_eq!(self.k, other.k, "oplus: site mismatch");
        let space = self.space;
        let d = space.sites();
        let k = self.k;

        let (row_ranks, col_ranks) = if d == 1 {
            assert_grid_eq!(
                self.row_ranks,
                other.row_ranks,
                "oplus d==1: row ranks differ"
            );
            assert_grid_eq!(
                self.col_ranks,
                other.col_ranks,
                "oplus d==1: col ranks differ"
            );
            (self.row_ranks.clone(), self.col_ranks.clone())
        } else if k == 0 {
            assert_grid_eq!(
                self.row_ranks,
                other.row_ranks,
                "oplus k==0: row ranks differ"
            );
            (
                self.row_ranks.clone(),
                add_rank_grids(&self.col_ranks, &other.col_ranks),
            )
        } else if k == d - 1 {
            assert_grid_eq!(
                self.col_ranks,
                other.col_ranks,
                "oplus k==d-1: col ranks differ"
            );
            (
                add_rank_grids(&self.row_ranks, &other.row_ranks),
                self.col_ranks.clone(),
            )
        } else {
            (
                add_rank_grids(&self.row_ranks, &other.row_ranks),
                add_rank_grids(&self.col_ranks, &other.col_ranks),
            )
        };

        let mut out = Self::zeros_sym(space, k, &row_ranks, &col_ranks);

        for &f in S::STEPS {
            for s in self.transitions(f) {
                let a = self.block(f, s).expect("A transition present");
                let (ma, na) = (a.nrows(), a.ncols());
                let b = other.block(f, s).expect("B transition present");
                let (mb, nb) = (b.nrows(), b.ncols());
                let (row_off, col_off) = if d == 1 {
                    (0, 0)
                } else if k == 0 {
                    (0, na)
                } else if k == d - 1 {
                    (ma, 0)
                } else {
                    (ma, na)
                };
                let mut dst = out.block_mut(f, s).unwrap();
                for j in 0..na {
                    for i in 0..ma {
                        *dst.rb_mut().get_mut(i, j) = *a.get(i, j);
                    }
                }
                for j in 0..nb {
                    for i in 0..mb {
                        let v = *b.get(i, j);
                        if d == 1 {
                            *dst.rb_mut().get_mut(i, j) += v;
                        } else {
                            *dst.rb_mut().get_mut(row_off + i, col_off + j) = v;
                        }
                    }
                }
            }
        }
        out
    }

    /// Scale every block by `a` in place: unweighted, scales every non-empty block (the stored
    /// ones; an empty block has nothing to scale).
    pub fn scale_mut(&mut self, a: T) {
        for m in self.store.blocks_mut() {
            crate::core::algebra::block_scale(m.as_mut(), a);
        }
    }

    /// `-A`: a scaled clone.
    pub fn neg(&self) -> Self {
        let mut out = self.clone();
        out.scale_mut(T::from_f64(-1.0));
        out
    }
}

impl<T: Scalar, S: Symmetry> crate::mem::Footprint for Su2Core<T, S> {
    fn bytes(&self) -> u64 {
        self.store
            .blocks()
            .iter()
            .map(|m| crate::mem::block_bytes::<T>(m.nrows(), m.ncols()))
            .sum()
    }
}

#[cfg(test)]
mod spg2_tests {
    use super::*;

    #[test]
    fn spg2_family_flux_table_and_cross_pin_against_delta_parity() {
        for gi in 0u8..8 {
            let g = Irrep::new(gi);
            assert_eq!(Family::Empty.flux(g), Irrep::IDENTITY, "Empty: no flux");
            assert_eq!(
                Family::Up.flux(g),
                g,
                "Up: flux is g (S+1/2 branch, one electron)"
            );
            assert_eq!(
                Family::Dn.flux(g),
                g,
                "Dn: flux is g (S-1/2 branch, one electron)"
            );
            assert_eq!(
                Family::Occ.flux(g),
                g.fuse(g),
                "Occ: flux is g.fuse(g) (written as the product, not folded to IDENTITY)"
            );
            assert_eq!(
                Family::Occ.flux(g),
                Irrep::IDENTITY,
                "Occ: g.fuse(g) must equal IDENTITY on Z2^m (self-inverse), the value \
                 ODD_ELECTRON cross-pins against"
            );
            for f in Family::ALL {
                let (dn, _) = f.delta();
                let odd = dn % 2 == 1;
                let expected = if odd { g } else { Irrep::IDENTITY };
                assert_eq!(
                    f.flux(g),
                    expected,
                    "f={f:?} g={g:?}: flux disagrees with the delta().0 % 2 parity cross-pin"
                );
            }
        }
    }

    #[test]
    fn spg2_family_apply_unapply_round_trip_at_nontrivial_irrep() {
        for f in Family::ALL {
            for gi in 0u8..4 {
                let g = Irrep::new(gi);
                for n in 0..=4usize {
                    for two_s in 0..=4usize {
                        for iri in 0u8..4 {
                            let s = Su2Sector::with_irrep(n, two_s, Irrep::new(iri));
                            if let Some(t) = f.apply(s, g) {
                                assert_eq!(
                                    f.unapply(t, g),
                                    Some(s),
                                    "f={f:?} g={g:?} s={s:?}: unapply(apply(s,g),g) must \
                                     round-trip"
                                );
                                assert_eq!(t.irrep, s.irrep.fuse(f.flux(g)));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn spg2_family_apply_reduces_to_pre_spg2_at_identity_irrep() {
        for f in Family::ALL {
            for n in 0..=5usize {
                for two_s in 0..=5usize {
                    let s = Su2Sector::with_irrep(n, two_s, Irrep::new(2));
                    let expect_shift = f.apply(s, Irrep::IDENTITY).map(|t| (t.n, t.two_s));
                    if let Some((tn, tts)) = expect_shift {
                        let t = f.apply(s, Irrep::IDENTITY).unwrap();
                        assert_eq!((t.n, t.two_s), (tn, tts));
                        assert_eq!(
                            t.irrep, s.irrep,
                            "IDENTITY flux must leave the irrep unchanged"
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod s1_tests {
    use super::{BlockGrid, Family as Su2Family, Su2Core};
    use crate::pointgroup::Irrep;
    use crate::scalar::Scalar;
    use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
    use crate::su2::vector::{Su2TtVector, tt_zeros as su2_tt_zeros};
    use crate::test_util::randomized_su2_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn family_delta_and_apply_sign_table_with_dn_underflow_guard() {
        assert_eq!(Su2Family::Empty.delta(), (0, 0));
        assert_eq!(Su2Family::Up.delta(), (1, 1));
        assert_eq!(Su2Family::Dn.delta(), (1, -1));
        assert_eq!(Su2Family::Occ.delta(), (2, 0));

        let s = Su2Sector::new(3, 2);
        assert_eq!(
            Su2Family::Empty.apply(s, Irrep::IDENTITY),
            Some(Su2Sector::new(3, 2))
        );
        assert_eq!(
            Su2Family::Up.apply(s, Irrep::IDENTITY),
            Some(Su2Sector::new(4, 3))
        );
        assert_eq!(
            Su2Family::Dn.apply(s, Irrep::IDENTITY),
            Some(Su2Sector::new(4, 1))
        );
        assert_eq!(
            Su2Family::Occ.apply(s, Irrep::IDENTITY),
            Some(Su2Sector::new(5, 2))
        );

        let singlet = Su2Sector::new(4, 0);
        assert_eq!(Su2Family::Dn.apply(singlet, Irrep::IDENTITY), None);
        assert!(Su2Family::Empty.apply(singlet, Irrep::IDENTITY).is_some());
        assert!(Su2Family::Up.apply(singlet, Irrep::IDENTITY).is_some());
        assert!(Su2Family::Occ.apply(singlet, Irrep::IDENTITY).is_some());
    }

    #[test]
    fn ctor_invariants_zeros_and_csf_step_pass_check() {
        for &(d, n, two_s) in &[(4usize, 2usize, 0usize), (4, 3, 1)] {
            let space = Su2Space::new(d, n, two_s);
            let row_ranks = Su2RankGrid::zeros(&space);
            let col_ranks = Su2RankGrid::zeros(&space);
            let core: Su2Core<f64> = Su2Core::zeros(space, 0, &row_ranks, &col_ranks);
            assert!(core.check_invariants());

            let entering = Su2Sector::new(0, 0);
            let step: Su2Core<f64> = Su2Core::csf_step(space, 0, Su2Family::Up, entering);
            assert!(step.check_invariants());
        }
    }

    fn clone_core_blocks<T: Scalar>(core: &Su2Core<T>) -> [BlockGrid<T>; 4] {
        let families = [
            Su2Family::Empty,
            Su2Family::Up,
            Su2Family::Dn,
            Su2Family::Occ,
        ];
        let mut blocks: [BlockGrid<T>; 4] =
            std::array::from_fn(|_| Su2SectorGrid::filled(&core.space, None));
        for (fi, &f) in families.iter().enumerate() {
            for s in core.transitions(f) {
                let m = core.block(f, s).expect("admissible transition has a block");
                blocks[fi][s] = Some(m.to_owned());
            }
        }
        blocks
    }

    #[test]
    fn from_blocks_roundtrip_rederives_rank_grids_including_dn_adjoint() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xF00D_B10C);
        let mut dn_adjoint_seen = false;
        for &(d, n, two_s) in &[(6usize, 4usize, 2usize), (5, 3, 1), (6, 4, 0)] {
            let space = Su2Space::new(d, n, two_s);
            let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
            for k in 0..d {
                let core = x.core(k);
                let blocks = clone_core_blocks(core);
                let rebuilt: Su2Core<f64> = Su2Core::from_blocks(space, k, blocks);
                assert!(rebuilt.check_invariants());
                for s in space.bond_sectors(k) {
                    assert_eq!(rebuilt.row_ranks[s], core.row_ranks[s], "row k={k} s={s:?}");
                }
                for t in space.bond_sectors(k + 1) {
                    assert_eq!(rebuilt.col_ranks[t], core.col_ranks[t], "col k={k} t={t:?}");
                    if t.n >= 1 {
                        let src = Su2Sector::new(t.n - 1, t.two_s + 1);
                        if let Some(m) = core.block(Su2Family::Dn, src) {
                            assert_eq!(m.ncols(), core.col_ranks[t], "Dn +1 adjoint k={k} t={t:?}");
                            dn_adjoint_seen = true;
                        }
                    }
                }
            }
        }
        assert!(
            dn_adjoint_seen,
            "Dn shifted-adjoint column-rank lookup was never exercised — coverage gap"
        );
    }

    #[test]
    fn zero_dim_admissible_blocks_are_some_not_none() {
        let space = Su2Space::new(6, 4, 0);
        let x: Su2TtVector<f64> = su2_tt_zeros(space);
        let mut saw_zero_dim = false;
        for k in 0..space.d {
            let core = x.core(k);
            for f in [
                Su2Family::Empty,
                Su2Family::Up,
                Su2Family::Dn,
                Su2Family::Occ,
            ] {
                for s in core.transitions(f) {
                    assert!(
                        core.block(f, s).is_some(),
                        "admissible {f:?} at {s:?} (k={k}) must be Some, not None"
                    );
                    let t = f
                        .apply(s, core.space.pg.irrep_of(k))
                        .expect("admissible transition");
                    if core.row_ranks[s] == 0 || core.col_ranks[t] == 0 {
                        saw_zero_dim = true;
                    }
                }
            }
        }
        assert!(saw_zero_dim, "no zero-dim admissible block was produced");
    }
}
