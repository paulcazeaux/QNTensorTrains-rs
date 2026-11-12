// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Quantum-number bookkeeping: the space descriptor, sector enumeration, and dense
//! per-sector grids.

use crate::pointgroup::{Irrep, PointGroup};
use std::ops::RangeInclusive;

const _: () = assert!(size_of::<QnSpace>() == 96);
const _: () = assert!(size_of::<Sector>() == 24);

/// Global symmetry descriptor: `d` lattice sites, `Nup`/`Ndn` conserved spin-up/spin-down electron
/// counts, and the abelian point group with the state's total irrep. Every TT object stores one and
/// binary operations assert equality (so two trains carrying different point groups can never be
/// combined).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QnSpace {
    /// Total number of spin-up electrons.
    pub nup: usize,
    /// Total number of spin-down electrons.
    pub ndn: usize,
    /// Number of lattice sites (tensor order).
    pub d: usize,
    /// Per-site irrep table; [`PointGroup::TRIVIAL`] (order 1) means "no point-group symmetry".
    pub pg: PointGroup,
    /// The state's total irrep, i.e. the [`Sector`] irrep required at the right boundary (bond
    /// `d`). [`Irrep::IDENTITY`] for a trivial group. There is no left-boundary counterpart:
    /// nothing is occupied left of bond `0`, so it is always [`Irrep::IDENTITY`].
    pub target: Irrep,
}

/// Bond quantum-number label: electrons of each spin to the left of the bond, plus the point-group
/// irrep of that occupation pattern. 0-based.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Sector {
    /// Spin-up electron count left of the bond.
    pub nup: usize,
    /// Spin-down electron count left of the bond.
    pub ndn: usize,
    /// Point-group irrep of the occupation pattern left of the bond. [`Irrep::IDENTITY`] in a
    /// trivial-group ([`PointGroup::TRIVIAL`]) space, which is every space built by
    /// [`QnSpace::new`] / [`QnSpace::from_n_two_sz`].
    pub irrep: Irrep,
}

impl Sector {
    /// Convenience constructor, trivial irrep ([`Irrep::IDENTITY`]).
    #[inline]
    pub const fn new(nup: usize, ndn: usize) -> Self {
        Sector {
            nup,
            ndn,
            irrep: Irrep::IDENTITY,
        }
    }

    /// Convenience constructor carrying an explicit point-group irrep.
    #[inline]
    pub const fn with_irrep(nup: usize, ndn: usize, irrep: Irrep) -> Self {
        Sector { nup, ndn, irrep }
    }
}

#[inline]
fn spin_bounds(total: usize, d: usize, b: usize) -> RangeInclusive<usize> {
    (total + b).saturating_sub(d)..=b.min(total)
}

impl QnSpace {
    /// New space without point-group symmetry ([`PointGroup::TRIVIAL`], identity target);
    /// requires `Nup ≤ d` and `Ndn ≤ d`.
    pub fn new(d: usize, nup: usize, ndn: usize) -> Self {
        assert!(
            nup <= d && ndn <= d,
            "electron counts Nup={nup}, Ndn={ndn} cannot exceed number of sites d={d}"
        );
        QnSpace {
            nup,
            ndn,
            d,
            pg: PointGroup::TRIVIAL,
            target: Irrep::IDENTITY,
        }
    }

    /// From particle number `N` and doubled spin projection `2·Sz`. No point-group symmetry — see
    /// [`from_n_two_sz_with_point_group`](Self::from_n_two_sz_with_point_group).
    pub fn from_n_two_sz(d: usize, n: usize, two_sz: i64) -> Self {
        let n = n as i64;
        assert!(
            (n + two_sz) % 2 == 0 && n + two_sz >= 0 && n - two_sz >= 0,
            "invalid N={n}, 2Sz={two_sz}"
        );
        Self::new(d, ((n + two_sz) / 2) as usize, ((n - two_sz) / 2) as usize)
    }

    /// New space with abelian point-group symmetry: per-site irrep table `pg` and total (right
    /// boundary) irrep `target`. Panics unless `Nup ≤ d`, `Ndn ≤ d`, `target ∈ 0..pg.order()`
    /// and `d ≤ PointGroup::MAX_SITES` (a site past the packed table would silently read back as
    /// [`Irrep::IDENTITY`], which is a wrong answer rather than an error).
    pub fn with_point_group(
        d: usize,
        nup: usize,
        ndn: usize,
        pg: PointGroup,
        target: Irrep,
    ) -> Self {
        assert!(
            d <= PointGroup::MAX_SITES,
            "d={d} exceeds the {}-site point-group table limit",
            PointGroup::MAX_SITES
        );
        assert!(
            target.index() < pg.order(),
            "target irrep {} is out of range 0..{}",
            target.index(),
            pg.order()
        );
        QnSpace {
            pg,
            target,
            ..Self::new(d, nup, ndn)
        }
    }

    /// The same lattice with a different boundary irrep — the one-line move the cross-sector
    /// response is built on.
    pub fn with_target(self, target: Irrep) -> Self {
        assert!(
            target.index() < self.pg.order(),
            "target irrep {} is out of range 0..{}",
            target.index(),
            self.pg.order()
        );
        QnSpace { target, ..self }
    }

    /// Do the two spaces describe the same *lattice* — equal `nup`, `ndn`, `d` and per-site irrep
    /// table — **ignoring `target`**?
    pub fn same_lattice(&self, other: &QnSpace) -> bool {
        self.nup == other.nup && self.ndn == other.ndn && self.d == other.d && self.pg == other.pg
    }

    /// [`from_n_two_sz`](Self::from_n_two_sz) with point-group symmetry (the FCIDUMP-boundary
    /// shape: `N`, `2Sz`, ORBSYM table, total irrep).
    pub fn from_n_two_sz_with_point_group(
        d: usize,
        n: usize,
        two_sz: i64,
        pg: PointGroup,
        target: Irrep,
    ) -> Self {
        let s = Self::from_n_two_sz(d, n, two_sz);
        Self::with_point_group(d, s.nup, s.ndn, pg, target)
    }

    /// Total particle number `N = Nup + Ndn`.
    #[inline]
    pub fn n(&self) -> usize {
        self.nup + self.ndn
    }

    /// Admissible sectors at bond `b ∈ 0..=d`, in canonical order (`nup` fastest, `ndn` next,
    /// `irrep` outermost).
    pub fn bond_sectors(&self, b: usize) -> Vec<Sector> {
        assert!(b <= self.d, "bond index {b} out of range 0..={}", self.d);
        let mut out = Vec::new();
        for ir in 0..self.pg.order() {
            for ndn in spin_bounds(self.ndn, self.d, b) {
                for nup in spin_bounds(self.nup, self.d, b) {
                    out.push(Sector::with_irrep(nup, ndn, Irrep::new(ir)));
                }
            }
        }
        out
    }

    /// Is `s` an admissible sector at bond `b`? (0-based, including the irrep bound.)
    #[inline]
    pub fn contains(&self, b: usize, s: Sector) -> bool {
        b <= self.d
            && spin_bounds(self.nup, self.d, b).contains(&s.nup)
            && spin_bounds(self.ndn, self.d, b).contains(&s.ndn)
            && s.irrep.index() < self.pg.order()
    }

    /// Maximum possible TT rank of sector `s` at bond `b`: the number of left paths
    /// `C(b, nup)·C(b, ndn)` capped by the number of right paths
    /// `C(d−b, Nup−nup)·C(d−b, Ndn−ndn)`. Saturates at `u64::MAX`.
    pub fn max_rank(&self, b: usize, s: Sector) -> u64 {
        if !self.contains(b, s) {
            return 0;
        }
        let left = binomial(b, s.nup).saturating_mul(binomial(b, s.ndn));
        let right = binomial(self.d - b, self.nup - s.nup)
            .saturating_mul(binomial(self.d - b, self.ndn - s.ndn));
        u64::try_from(left.min(right)).unwrap_or(u64::MAX)
    }
}

fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc.saturating_mul((n - i) as u128) / (i as u128 + 1);
    }
    acc
}

/// Dense `|G| × (Nup+1) × (Ndn+1)` grid indexed by [`Sector`] (row-major: `irrep` slowest, then
/// `nup`, `ndn` fastest).
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct SectorGrid<V> {
    nup1: usize,
    ndn1: usize,
    nir: usize,
    data: Vec<V>,
}

impl<V> SectorGrid<V> {
    /// Build a full grid for `space` from a function of the sector. `f` is called in storage
    /// order: `irrep` outermost, then `nup`, `ndn` innermost (note this is the *opposite* nesting
    /// of `nup`/`ndn` from [`QnSpace::bond_sectors`] — both orders are load-bearing).
    pub fn from_fn(space: &QnSpace, mut f: impl FnMut(Sector) -> V) -> Self {
        let (nup1, ndn1) = (space.nup + 1, space.ndn + 1);
        let nir = space.pg.order() as usize;
        let mut data = Vec::with_capacity(nir * nup1 * ndn1);
        for ir in 0..nir {
            for nup in 0..nup1 {
                for ndn in 0..ndn1 {
                    data.push(f(Sector::with_irrep(nup, ndn, Irrep::new(ir as u8))));
                }
            }
        }
        crate::storage_probe::grid_built(
            crate::storage_probe::Table::Abelian,
            std::mem::size_of::<V>(),
            data.len(),
        );
        SectorGrid {
            nup1,
            ndn1,
            nir,
            data,
        }
    }

    /// Particle-window grid extents `(Nup+1, Ndn+1)`. Deliberately still a 2-tuple (the C ABI
    /// marshals a `rows × cols` block); the irrep extent is [`nirrep`](Self::nirrep).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.nup1, self.ndn1)
    }

    /// Extent of the irrep axis, i.e. the point-group order the grid was built for.
    #[inline]
    pub fn nirrep(&self) -> usize {
        self.nir
    }

    #[inline]
    pub(crate) fn storage_len(space: &QnSpace) -> usize {
        space.pg.order() as usize * (space.nup + 1) * (space.ndn + 1)
    }

    #[inline]
    pub(crate) fn storage_index(space: &QnSpace, s: Sector) -> usize {
        let (nup1, ndn1) = (space.nup + 1, space.ndn + 1);
        debug_assert!(
            s.nup < nup1 && s.ndn < ndn1 && (s.irrep.index() as usize) < space.pg.order() as usize,
            "sector {s:?} out of grid"
        );
        (s.irrep.index() as usize) * nup1 * ndn1 + s.nup * ndn1 + s.ndn
    }

    #[inline]
    pub(crate) fn storage_label(space: &QnSpace, i: usize) -> Sector {
        let (nup1, ndn1) = (space.nup + 1, space.ndn + 1);
        let (ir, p) = (i / (nup1 * ndn1), i % (nup1 * ndn1));
        Sector::with_irrep(p / ndn1, p % ndn1, Irrep::new(ir as u8))
    }

    #[inline]
    fn idx(&self, s: Sector) -> usize {
        debug_assert!(
            s.nup < self.nup1 && s.ndn < self.ndn1 && (s.irrep.index() as usize) < self.nir,
            "sector {s:?} out of grid"
        );
        (s.irrep.index() as usize) * self.nup1 * self.ndn1 + s.nup * self.ndn1 + s.ndn
    }

    /// Iterate `(sector, &value)` over the full grid (admissible or not), `ndn` fastest,
    /// `irrep` slowest.
    pub fn iter(&self) -> impl Iterator<Item = (Sector, &V)> + '_ {
        let plane = self.nup1 * self.ndn1;
        self.data.iter().enumerate().map(move |(i, v)| {
            let (ir, p) = (i / plane, i % plane);
            (
                Sector::with_irrep(p / self.ndn1, p % self.ndn1, Irrep::new(ir as u8)),
                v,
            )
        })
    }

    #[inline]
    pub(crate) fn values(&self) -> &[V] {
        &self.data
    }

    #[inline]
    pub(crate) fn values_mut(&mut self) -> &mut [V] {
        &mut self.data
    }

    pub(crate) fn relabel_irrep_in_place(&mut self, phi: Irrep) {
        if phi == Irrep::IDENTITY {
            return;
        }
        let plane = self.nup1 * self.ndn1;
        for ir in 0..self.nir {
            let jr = usize::from(Irrep::new(ir as u8).fuse(phi).index());
            debug_assert!(
                jr < self.nir,
                "relabel_irrep: phi {} leaves the grid's irrep axis (order {})",
                phi.index(),
                self.nir
            );
            debug_assert_eq!(
                usize::from(Irrep::new(jr as u8).fuse(phi).index()),
                ir,
                "relabel_irrep assumes an involution on the irrep axis"
            );
            if jr > ir {
                for p in 0..plane {
                    self.data.swap(ir * plane + p, jr * plane + p);
                }
            }
        }
    }
}

impl<V: Clone> Clone for SectorGrid<V> {
    fn clone(&self) -> Self {
        crate::storage_probe::grid_cloned(
            crate::storage_probe::Table::Abelian,
            std::mem::size_of::<V>(),
            self.data.len(),
        );
        SectorGrid {
            nup1: self.nup1,
            ndn1: self.ndn1,
            nir: self.nir,
            data: self.data.clone(),
        }
    }
}

impl<V: Clone> SectorGrid<V> {
    /// Grid with every entry equal to `v`.
    pub fn filled(space: &QnSpace, v: V) -> Self {
        Self::from_fn(space, |_| v.clone())
    }

    /// The irrep RELABELLING `out[s] = self[s ⊕ φ]` — the single primitive the shifted tangent
    /// space is built from.
    pub fn relabel_irrep(&self, phi: Irrep) -> Self {
        let mut out = self.clone();
        out.relabel_irrep_in_place(phi);
        out
    }
}

impl<V> std::ops::Index<Sector> for SectorGrid<V> {
    type Output = V;
    #[inline]
    fn index(&self, s: Sector) -> &V {
        &self.data[self.idx(s)]
    }
}

impl<V> std::ops::IndexMut<Sector> for SectorGrid<V> {
    #[inline]
    fn index_mut(&mut self, s: Sector) -> &mut V {
        let i = self.idx(s);
        &mut self.data[i]
    }
}

/// Per-sector bond ranks.
pub type RankGrid = SectorGrid<usize>;

impl RankGrid {
    /// All-zero rank grid.
    pub fn zeros(space: &QnSpace) -> Self {
        Self::filled(space, 0)
    }

    /// Sum of all sector ranks (the "total" bond dimension).
    pub fn total(&self) -> usize {
        self.data.iter().sum()
    }
}

/// Forward/backward occupation-path counts, one grid per bond (abelian twin of
/// [`crate::su2::qn::PathCountGrid`]). Saturating `u64` — path counts grow combinatorially.
pub type PathCountGrid = SectorGrid<u64>;

const MOVES: [(usize, usize, bool); 4] = [(0, 0, false), (1, 0, true), (0, 1, true), (1, 1, false)];

/// PER-SECTOR rank cap: every sector rank `r[q] ≤ .0`. `0` = no cap. Parameters of this type are
/// named `rmax`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct PerSector(pub usize);

/// POOLED cap on the TOTAL bond dimension: `Σ_q r[q] ≤ .0`, the conventional DMRG `χ` / `M`, with
/// the allocation across sectors decided by pooling (all sectors' singular values sorted together).
/// `0` = no cap. Parameters of this type are named `chi`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Pooled(pub usize);

impl PerSector {
    /// No per-sector cap.
    pub const NONE: PerSector = PerSector(0);
    /// Whether this is the "no cap" sentinel.
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

impl Pooled {
    /// No pooled cap.
    pub const NONE: Pooled = Pooled(0);
    /// Whether this is the "no cap" sentinel.
    pub const fn is_none(self) -> bool {
        self.0 == 0
    }
}

/// Forward (`L`) and backward (`R`) occupation-path counts over `(nup, ndn, irrep)`, one grid per
/// bond `0..=d` (the abelian analogue of the spin-adapted [`crate::su2::qn::pathcounts`]).
pub fn path_counts(space: &QnSpace) -> (Vec<PathCountGrid>, Vec<PathCountGrid>) {
    let d = space.d;
    let mut l: Vec<PathCountGrid> = (0..=d).map(|_| PathCountGrid::filled(space, 0)).collect();
    l[0][Sector::new(0, 0)] = 1;
    for b in 0..d {
        let g = space.pg.irrep_of(b);
        for s in space.bond_sectors(b) {
            let c = l[b][s];
            if c == 0 {
                continue;
            }
            for &(du, dd, odd) in &MOVES {
                let t = child(s, du, dd, odd, g);
                if space.contains(b + 1, t) {
                    l[b + 1][t] = l[b + 1][t].saturating_add(c);
                }
            }
        }
    }
    let mut r: Vec<PathCountGrid> = (0..=d).map(|_| PathCountGrid::filled(space, 0)).collect();
    r[d][Sector::with_irrep(space.nup, space.ndn, space.target)] = 1;
    for b in (0..d).rev() {
        let g = space.pg.irrep_of(b);
        for s in space.bond_sectors(b) {
            let mut c: u64 = 0;
            for &(du, dd, odd) in &MOVES {
                let t = child(s, du, dd, odd, g);
                if space.contains(b + 1, t) {
                    c = c.saturating_add(r[b + 1][t]);
                }
            }
            r[b][s] = c;
        }
    }
    (l, r)
}

#[inline]
fn child(s: Sector, du: usize, dd: usize, odd: bool, g: Irrep) -> Sector {
    Sector::with_irrep(
        s.nup + du,
        s.ndn + dd,
        if odd { s.irrep.fuse(g) } else { s.irrep },
    )
}

/// Per-bond, per-sector maximum TT rank `min(L[b][s], R[b][s])` (the abelian twin of
/// [`crate::su2::qn::max_ranks`]). Unreachable sectors — no left path, or none
/// that completes to `space.target` — get `0`, which is what makes D3's `contains` superset
/// harmless.
pub fn max_ranks(space: &QnSpace) -> Vec<RankGrid> {
    let (l, r) = path_counts(space);
    (0..=space.d)
        .map(|b| {
            RankGrid::from_fn(space, |s| {
                usize::try_from(l[b][s].min(r[b][s])).unwrap_or(usize::MAX)
            })
        })
        .collect()
}

/// [`max_ranks`] for an OPEN BLOCK of width `rho_d` at bond `d`: `min(L[b][s], ρ_d·R[b][s])`.
pub fn max_ranks_open(space: &QnSpace, rho_d: usize) -> Vec<RankGrid> {
    assert!(rho_d >= 1, "max_ranks_open: a block has width ≥ 1");
    let (l, r) = path_counts(space);
    (0..=space.d)
        .map(|b| {
            RankGrid::from_fn(space, |s| {
                let open = r[b][s].saturating_mul(rho_d as u64);
                usize::try_from(l[b][s].min(open)).unwrap_or(usize::MAX)
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{PathCountGrid, QnSpace, RankGrid, Sector, SectorGrid, max_ranks, path_counts};
    use crate::pointgroup::{Irrep, PointGroup, PointGroupKind};
    use std::collections::{HashMap, HashSet};

    fn assert_max_ranks_matches_binomial(space: &QnSpace) {
        let caps = max_ranks(space);
        assert_eq!(caps.len(), space.d + 1, "max_ranks must return d+1 grids");
        for (b, caps_b) in caps.iter().enumerate() {
            for s in space.bond_sectors(b) {
                assert_eq!(
                    caps_b[s] as u64,
                    space.max_rank(b, s),
                    "d={} Nup={} Ndn={} bond={b} sector={s:?}: DP cap != binomial max_rank",
                    space.d,
                    space.nup,
                    space.ndn
                );
            }
        }
    }

    #[test]
    fn max_ranks_matches_binomial_max_rank_sweep() {
        for d in 1..=12usize {
            for nup in 0..=d.min(5) {
                for ndn in 0..=d.min(5) {
                    let space = QnSpace::new(d, nup, ndn);
                    assert_max_ranks_matches_binomial(&space);
                }
            }
        }
    }

    #[test]
    fn max_ranks_sweep_is_nonvacuous() {
        let space = QnSpace::new(8, 3, 2);
        let caps = max_ranks(&space);
        let mut max_seen = 0u64;
        for (b, caps_b) in caps.iter().enumerate() {
            for s in space.bond_sectors(b) {
                max_seen = max_seen.max(caps_b[s] as u64);
            }
        }
        assert!(max_seen > 1, "max_ranks table is vacuously <= 1 everywhere");
    }

    #[test]
    fn max_ranks_matches_binomial_max_rank_at_u64_saturation_scale() {
        assert_max_ranks_matches_binomial(&QnSpace::new(70, 35, 35));
    }

    #[test]
    fn sector_new_defaults_to_identity_irrep() {
        let s = Sector::new(3, 2);
        assert_eq!(s.irrep, Irrep::IDENTITY);
        assert_eq!(s, Sector::with_irrep(3, 2, Irrep::IDENTITY));
    }

    #[test]
    fn bond_sectors_canonical_order_pin_with_identity_irrep() {
        let space = QnSpace::new(4, 2, 2);
        let got = space.bond_sectors(2);
        let expected_pairs = [
            (0, 0),
            (1, 0),
            (2, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        ];
        assert_eq!(got.len(), expected_pairs.len());
        for (s, &(nup, ndn)) in got.iter().zip(expected_pairs.iter()) {
            assert_eq!((s.nup, s.ndn), (nup, ndn));
            assert_eq!(
                s.irrep,
                Irrep::IDENTITY,
                "sector {s:?} must carry the trivial irrep at |G|=1"
            );
        }
    }

    #[test]
    fn sector_grid_from_fn_visitation_order_unchanged_at_trivial_group() {
        let space = QnSpace::new(3, 2, 1);
        let mut visited: Vec<(usize, usize)> = Vec::new();
        let _grid: SectorGrid<usize> = SectorGrid::from_fn(&space, |s| {
            assert_eq!(s.irrep, Irrep::IDENTITY);
            visited.push((s.nup, s.ndn));
            visited.len()
        });
        let mut expected = Vec::new();
        for nup in 0..=space.nup {
            for ndn in 0..=space.ndn {
                expected.push((nup, ndn));
            }
        }
        assert_eq!(visited, expected);
    }

    #[test]
    fn sector_grid_index_and_iter_reduce_to_current_layout_at_trivial_group() {
        let space = QnSpace::new(3, 2, 1);
        let ndn1 = space.ndn + 1;
        let grid: RankGrid = SectorGrid::from_fn(&space, |s| s.nup * ndn1 + s.ndn);

        for nup in 0..=space.nup {
            for ndn in 0..=space.ndn {
                assert_eq!(grid[Sector::new(nup, ndn)], nup * ndn1 + ndn);
            }
        }

        let seq: Vec<(usize, usize)> = grid.iter().map(|(s, _)| (s.nup, s.ndn)).collect();
        let mut expected = Vec::new();
        for nup in 0..=space.nup {
            for ndn in 0..=space.ndn {
                expected.push((nup, ndn));
            }
        }
        assert_eq!(seq, expected);
        for (s, &v) in grid.iter() {
            assert_eq!(s.irrep, Irrep::IDENTITY);
            assert_eq!(v, s.nup * ndn1 + s.ndn);
        }
    }

    #[test]
    fn sector_grid_iter_irrep_is_slowest_stride_at_nontrivial_group() {
        let d = 5;
        let (nup, ndn) = (3, 2);
        let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k % 4) as u8)).collect();
        let pg = PointGroup::new(2, &irreps);
        let space = QnSpace::with_point_group(d, nup, ndn, pg, Irrep::new(0));
        let nup1 = space.nup + 1;
        let ndn1 = space.ndn + 1;
        let plane = nup1 * ndn1;
        let order = pg.order() as usize;

        let grid: SectorGrid<usize> = SectorGrid::filled(&space, 0);
        let seq: Vec<Sector> = grid.iter().map(|(s, _)| s).collect();
        assert_eq!(seq.len(), plane * order);

        for (i, s) in seq.iter().enumerate() {
            assert_eq!(
                s.irrep,
                Irrep::new((i / plane) as u8),
                "iter() position {i}: irrep must be the SLOWEST stride (constant for {plane} \
                 consecutive elements, then advancing)"
            );
        }

        let first_block: Vec<(usize, usize)> =
            seq[0..plane].iter().map(|s| (s.nup, s.ndn)).collect();
        for ir in 1..order {
            let block: Vec<(usize, usize)> = seq[ir * plane..(ir + 1) * plane]
                .iter()
                .map(|s| (s.nup, s.ndn))
                .collect();
            assert_eq!(
                block, first_block,
                "irrep block {ir}: (nup,ndn) sub-order must reproduce irrep 0's exactly"
            );
        }
    }

    #[test]
    fn bond_sectors_irrep_outermost_restricts_to_trivial_group_relative_order() {
        let d = 5;
        let (nup, ndn) = (3, 2);
        let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k % 4) as u8)).collect();
        let pg = PointGroup::new(2, &irreps);
        let target = Irrep::new(1);
        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
        let trivial_space = QnSpace::new(d, nup, ndn);
        let order = pg.order() as usize;

        for b in 0..=d {
            let got = space.bond_sectors(b);
            let want_pairs: Vec<(usize, usize)> = trivial_space
                .bond_sectors(b)
                .iter()
                .map(|s| (s.nup, s.ndn))
                .collect();
            assert_eq!(
                got.len(),
                want_pairs.len() * order,
                "bond {b}: sector count must scale by |G|={order} over the particle-window count"
            );

            let chunk_len = want_pairs.len();
            assert!(chunk_len > 0, "bond {b}: vacuous particle window");
            for (chunk_idx, chunk) in got.chunks(chunk_len).enumerate() {
                for s in chunk {
                    assert_eq!(
                        s.irrep,
                        Irrep::new(chunk_idx as u8),
                        "bond {b} chunk {chunk_idx}: irrep must be constant within an irrep-block \
                         (irrep is the OUTERMOST/slowest axis)"
                    );
                }
                let pairs: Vec<(usize, usize)> = chunk.iter().map(|s| (s.nup, s.ndn)).collect();
                assert_eq!(
                    pairs, want_pairs,
                    "bond {b} chunk {chunk_idx}: restricting to one irrep must reproduce the |G|=1 \
                     relative (nup, ndn) order exactly"
                );
            }
        }
    }

    #[test]
    fn bond_sectors_equals_contains_set_at_various_group_orders() {
        let d = 5;
        let (nup, ndn) = (3, 2);
        for nbits in 0..=3u8 {
            let order = 1u8 << nbits;
            let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k as u8) % order)).collect();
            let pg = PointGroup::new(nbits, &irreps);
            let space = QnSpace::with_point_group(d, nup, ndn, pg, Irrep::new(0));
            for b in 0..=d {
                let got: HashSet<Sector> = space.bond_sectors(b).into_iter().collect();
                let mut want: HashSet<Sector> = HashSet::new();
                for u in 0..=space.nup {
                    for v in 0..=space.ndn {
                        for ir in 0..pg.order() {
                            let s = Sector::with_irrep(u, v, Irrep::new(ir));
                            if space.contains(b, s) {
                                want.insert(s);
                            }
                        }
                    }
                }
                assert_eq!(
                    got, want,
                    "nbits={nbits} bond={b}: bond_sectors(b) != {{s : contains(b,s)}}"
                );
            }
        }
    }

    type BruteGrid = HashMap<(usize, usize, Irrep), u64>;

    fn range_occupation_counts(pg: &PointGroup, sites: &[usize]) -> BruteGrid {
        let n = sites.len();
        let mut counts: BruteGrid = HashMap::new();
        for up_mask in 0u32..(1 << n) {
            for dn_mask in 0u32..(1 << n) {
                let nu = up_mask.count_ones() as usize;
                let nd = dn_mask.count_ones() as usize;
                let mut irrep = Irrep::IDENTITY;
                for (i, &k) in sites.iter().enumerate() {
                    if (up_mask >> i) & 1 == 1 {
                        irrep = irrep.fuse(pg.irrep_of(k));
                    }
                    if (dn_mask >> i) & 1 == 1 {
                        irrep = irrep.fuse(pg.irrep_of(k));
                    }
                }
                *counts.entry((nu, nd, irrep)).or_insert(0) += 1;
            }
        }
        counts
    }

    fn brute_force_l_r(
        pg: &PointGroup,
        d: usize,
        nup: usize,
        ndn: usize,
        target: Irrep,
    ) -> (Vec<BruteGrid>, Vec<BruteGrid>) {
        let mut l = Vec::with_capacity(d + 1);
        let mut r = Vec::with_capacity(d + 1);
        for b in 0..=d {
            let prefix = range_occupation_counts(pg, &(0..b).collect::<Vec<_>>());
            let suffix = range_occupation_counts(pg, &(b..d).collect::<Vec<_>>());
            let mut r_b: BruteGrid = HashMap::new();
            for u in 0..=nup {
                for v in 0..=ndn {
                    for ir in 0..pg.order() {
                        let irrep = Irrep::new(ir);
                        let need = suffix
                            .get(&(nup - u, ndn - v, irrep.fuse(target)))
                            .copied()
                            .unwrap_or(0);
                        if need > 0 {
                            r_b.insert((u, v, irrep), need);
                        }
                    }
                }
            }
            l.push(prefix);
            r.push(r_b);
        }
        (l, r)
    }

    #[test]
    fn max_ranks_matches_brute_force_reachability_nontrivial_target() {
        let d = 6;
        let orbsym: [u8; 6] = [1, 2, 3, 4, 2, 3];
        let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table");
        let (nup, ndn) = (3, 2);

        let full = range_occupation_counts(&pg, &(0..d).collect::<Vec<_>>());
        let target = (1..pg.order())
            .map(Irrep::new)
            .find(|&ir| full.get(&(nup, ndn, ir)).copied().unwrap_or(0) > 0)
            .expect(
                "PG1 fixture must have a reachable non-identity target — fixture bug if this fires",
            );

        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
        let (l_dp, r_dp): (Vec<PathCountGrid>, Vec<PathCountGrid>) = path_counts(&space);
        let caps = max_ranks(&space);
        let (l_brute, r_brute) = brute_force_l_r(&pg, d, nup, ndn, target);

        let mut any_reachable = false;
        let mut any_unreachable = false;
        for b in 0..=d {
            for s in space.bond_sectors(b) {
                let key = (s.nup, s.ndn, s.irrep);
                let l_expected = l_brute[b].get(&key).copied().unwrap_or(0);
                let r_expected = r_brute[b].get(&key).copied().unwrap_or(0);
                assert_eq!(
                    l_dp[b][s], l_expected,
                    "L mismatch at bond {b} sector {s:?}",
                );
                assert_eq!(
                    r_dp[b][s], r_expected,
                    "R mismatch at bond {b} sector {s:?}",
                );
                assert_eq!(
                    caps[b][s],
                    l_dp[b][s].min(r_dp[b][s]) as usize,
                    "internal inconsistency at bond {b} sector {s:?}: max_ranks != min(L, R)"
                );
                let cap_expected = l_expected.min(r_expected);
                assert_eq!(
                    caps[b][s] as u64, cap_expected,
                    "cap mismatch at bond {b} sector {s:?}"
                );
                if cap_expected > 0 {
                    any_reachable = true;
                } else {
                    any_unreachable = true;
                }
            }
        }
        assert!(any_reachable, "vacuous: no reachable sector found at all");
        assert!(
            any_unreachable,
            "vacuous: every admissible (D3-superset) sector is reachable — the D3 relaxation is not \
             exercised by this fixture"
        );
    }

    #[test]
    fn max_ranks_all_zero_when_target_is_unreachable() {
        let d = 6;
        let orbsym: [u8; 6] = [1, 2, 3, 4, 2, 3];
        let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table");
        let space = QnSpace::with_point_group(d, 0, 0, pg, Irrep::new(1));

        let caps = max_ranks(&space);
        for (b, caps_b) in caps.iter().enumerate() {
            for s in space.bond_sectors(b) {
                assert_eq!(
                    caps_b[s], 0,
                    "bond {b} sector {s:?}: unreachable-target space must have every cap 0"
                );
            }
        }
        let (l, r): (Vec<PathCountGrid>, Vec<PathCountGrid>) = path_counts(&space);
        for b in 0..=d {
            for s in space.bond_sectors(b) {
                assert_eq!(
                    l[b][s].min(r[b][s]),
                    0,
                    "bond {b} sector {s:?}: L.min(R) must be 0 too"
                );
            }
        }
    }

    #[test]
    fn contains_holds_on_every_brute_force_reachable_sector() {
        let d = 6;
        let orbsym: [u8; 6] = [1, 2, 3, 4, 2, 3];
        let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table");
        let (nup, ndn) = (3, 2);
        let full = range_occupation_counts(&pg, &(0..d).collect::<Vec<_>>());
        let target = (1..pg.order())
            .map(Irrep::new)
            .find(|&ir| full.get(&(nup, ndn, ir)).copied().unwrap_or(0) > 0)
            .expect("PG1 fixture must have a reachable non-identity target");
        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
        let (l_brute, r_brute) = brute_force_l_r(&pg, d, nup, ndn, target);

        let mut checked = 0usize;
        for b in 0..=d {
            for (&(u, v, ir), &lc) in l_brute[b].iter() {
                let rc = r_brute[b].get(&(u, v, ir)).copied().unwrap_or(0);
                if lc > 0 && rc > 0 {
                    let s = Sector::with_irrep(u, v, ir);
                    assert!(
                        space.contains(b, s),
                        "bond {b} sector {s:?}: independently brute-force-reachable but `contains` \
                         rejects it — the D3 superset property is violated"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "vacuous: no reachable sector found to check");
    }

    #[test]
    fn contains_admits_a_reachability_false_witness_at_bond_zero() {
        let d = 4;
        let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k % 4) as u8)).collect();
        let pg = PointGroup::new(2, &irreps);
        let space = QnSpace::with_point_group(d, 2, 1, pg, Irrep::new(0));

        let witness = Sector::with_irrep(0, 0, Irrep::new(1));
        assert!(
            space.contains(0, witness),
            "D3: contains must accept every irrep at the empty-prefix boundary"
        );
        let (l, _r): (Vec<PathCountGrid>, Vec<PathCountGrid>) = path_counts(&space);
        assert_eq!(
            l[0][witness], 0,
            "D3 witness: {witness:?} must be contains=true but forward-unreachable (L=0)"
        );
        let caps = max_ranks(&space);
        assert_eq!(
            caps[0][witness], 0,
            "the reachability-false witness must have cap 0"
        );
    }

    use crate::core::Family;

    #[test]
    fn family_apply_matches_qn_path_counts_reachability_at_single_site() {
        for f in Family::ALL {
            let (du, dd) = f.delta();
            for gi in 0u8..8 {
                let g = Irrep::new(gi);
                let pg = PointGroup::new(3, &[g]);
                let expected_t = f.apply(Sector::new(0, 0), g);
                let space = QnSpace::with_point_group(1, du, dd, pg, expected_t.irrep);
                let caps = max_ranks(&space);
                assert_eq!(
                    caps[1][expected_t], 1,
                    "f={f:?} g={g:?}: Family::apply's target sector must be exactly the one \
                     qn::path_counts (its own independent MOVES table) finds reachable"
                );
            }
        }
    }

    use std::collections::BTreeSet;

    fn oracle_spin_counts(total: usize, d: usize, b: usize) -> BTreeSet<usize> {
        let mut counts = BTreeSet::new();
        for mask in 0u32..(1 << d) {
            if mask.count_ones() as usize == total {
                counts.insert((mask & ((1 << b) - 1)).count_ones() as usize);
            }
        }
        counts
    }

    #[test]
    fn bond_sectors_match_bitmask_oracle() {
        for d in 1..=8usize {
            for nup in 0..=d.min(4) {
                for ndn in 0..=d.min(4) {
                    let space = QnSpace::new(d, nup, ndn);
                    for b in 0..=d {
                        let ups = oracle_spin_counts(nup, d, b);
                        let dns = oracle_spin_counts(ndn, d, b);
                        let expected: BTreeSet<(usize, usize)> = ups
                            .iter()
                            .flat_map(|&u| dns.iter().map(move |&v| (u, v)))
                            .collect();
                        let got: BTreeSet<(usize, usize)> = space
                            .bond_sectors(b)
                            .iter()
                            .map(|s| (s.nup, s.ndn))
                            .collect();
                        assert_eq!(got, expected, "d={d} Nup={nup} Ndn={ndn} bond={b}");
                        for &(u, v) in &expected {
                            assert!(space.contains(b, Sector::new(u, v)));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn max_rank_boundaries_and_symmetry() {
        let space = QnSpace::new(8, 3, 2);
        assert_eq!(space.max_rank(0, Sector::new(0, 0)), 1);
        assert_eq!(space.max_rank(8, Sector::new(3, 2)), 1);
        assert_eq!(space.max_rank(0, Sector::new(1, 0)), 0);
        assert_eq!(space.max_rank(4, Sector::new(2, 1)), 16);
    }
}

#[cfg(test)]
mod qn_relabel_tests;

#[cfg(test)]
mod open_caps_tests;
