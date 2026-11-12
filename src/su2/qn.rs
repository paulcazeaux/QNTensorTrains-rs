// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) quantum-number bookkeeping: the space descriptor, bond-sector enumeration, the dense
//! per-sector grid, and genealogical `pathcounts` (the reduced-rank cap).

use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::ops::{Index, IndexMut};

use crate::pointgroup::{Irrep, PointGroup};
use crate::su2::core::Family;

const _: () = assert!(size_of::<Su2Space>() == 96);
const _: () = assert!(size_of::<Su2Sector>() == 24);

/// Global SU(2) symmetry descriptor: `d` lattice sites, total particle number `n`, total doubled
/// spin `two_s`, plus the abelian point group with the state's total irrep. Carried as a value, not
/// as `{N, twoS, d}` type parameters; every SU(2) TT object stores one and binary operations assert
/// equality (so two trains carrying different point groups can never be combined).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Su2Space {
    /// Total particle number.
    pub n: usize,
    /// Total doubled spin `2S`.
    pub two_s: usize,
    /// Number of lattice sites (tensor order).
    pub d: usize,
    /// Per-site irrep table; [`PointGroup::TRIVIAL`] (order 1) means "no point-group symmetry".
    pub pg: PointGroup,
    /// The state's total irrep, i.e. the [`Su2Sector`] irrep required at the right boundary (bond
    /// `d`). [`Irrep::IDENTITY`] for a trivial group. There is no left-boundary counterpart:
    /// nothing is occupied left of bond `0`, so that end is always [`Irrep::IDENTITY`].
    pub target: Irrep,
}

impl Su2Space {
    /// New space without point-group symmetry ([`PointGroup::TRIVIAL`], identity target); requires
    /// `0 ≤ N ≤ 2d`, `0 ≤ 2S ≤ min(N, 2d−N)`, and `N − 2S` even (the admissibility
    /// boundscheck), not the sector-enumeration algorithm itself.
    pub fn new(d: usize, n: usize, two_s: usize) -> Self {
        assert!(n <= 2 * d, "N={n} cannot exceed 2d={}", 2 * d);
        let cap = n.min(2 * d - n);
        assert!(
            two_s <= cap,
            "2S={two_s} exceeds min(N, 2d-N)={cap} (N={n}, d={d})"
        );
        assert!(
            (n + two_s).is_multiple_of(2),
            "N-2S must be even (N={n}, 2S={two_s})"
        );
        Su2Space {
            n,
            two_s,
            d,
            pg: PointGroup::TRIVIAL,
            target: Irrep::IDENTITY,
        }
    }

    /// New space with abelian point-group symmetry: per-site irrep table `pg` and total (right
    /// boundary) irrep `target`. Panics unless the three [`Su2Space::new`] invariants hold, plus
    /// `target ∈ 0..pg.order()` and `d ≤ PointGroup::MAX_SITES` (a site past the packed table
    /// would silently read back as [`Irrep::IDENTITY`], which is a wrong answer rather than an
    /// error, so the limit is checked even for a trivial group).
    pub fn with_point_group(
        d: usize,
        n: usize,
        two_s: usize,
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
        Su2Space {
            pg,
            target,
            ..Self::new(d, n, two_s)
        }
    }

    /// The same lattice with the total doubled spin replaced: `n`, `d`, `pg` and `target` all
    /// unchanged, `two_s` set to `two_s`. The SHIFTED channel's space:
    /// a singlet reference's triplet channel
    /// is `space.with_two_s(2)`.
    pub fn with_two_s(self, two_s: usize) -> Su2Space {
        Su2Space {
            pg: self.pg,
            target: self.target,
            ..Su2Space::new(self.d, self.n, two_s)
        }
    }

    pub(crate) fn same_orbital_lattice(&self, other: &Su2Space) -> bool {
        self.n == other.n && self.d == other.d && self.pg == other.pg
    }

    /// Admissible bond sectors at bond `b ∈ 0..=d`, in canonical order (`n` fastest, `2s` next,
    /// `irrep` outermost, see module docs).
    pub fn bond_sectors(&self, b: usize) -> Vec<Su2Sector> {
        assert!(b <= self.d, "bond index {b} out of range 0..={}", self.d);
        let lo = self.n.saturating_sub(2 * (self.d - b));
        let hi = (2 * b).min(self.n);
        let s_hi = b.min(self.n);
        let mut out = Vec::new();
        for ir in 0..self.pg.order() {
            for two_s in 0..=s_hi {
                for n in lo..=hi {
                    let s = Su2Sector::with_irrep(n, two_s, Irrep::new(ir));
                    if self.admissible_clauses(b, s) {
                        out.push(s);
                    }
                }
            }
        }
        out
    }

    /// Is `s` an admissible sector at bond `b`? The four admissibility clauses
    /// plus the irrep bound.
    #[inline]
    pub fn contains(&self, b: usize, s: Su2Sector) -> bool {
        b <= self.d && self.admissible_clauses(b, s) && s.irrep.index() < self.pg.order()
    }

    #[inline]
    fn admissible_clauses(&self, b: usize, s: Su2Sector) -> bool {
        let nn = s.n as i64;
        let twos = s.two_s as i64;
        let big_n = self.n as i64;
        let big_s = self.two_s as i64;
        let d = self.d as i64;
        let b = b as i64;
        let cap_right = 2 * (d - b);
        let lo_n = (big_n - cap_right).max(0);
        let hi_n = (2 * b).min(big_n);
        if nn < lo_n || nn > hi_n {
            return false;
        }
        if twos < 0 || twos > nn.min(2 * b - nn) {
            return false;
        }
        if (nn - twos).rem_euclid(2) != 0 {
            return false;
        }
        let rem = big_n - nn;
        (twos - big_s).abs() <= rem.min(cap_right - rem)
    }
}

/// SU(2) bond quantum-number label: total electrons `n`, total doubled coupled spin `2s`, and the
/// point-group irrep accumulated on the sites left of the bond. 0-based./// The irrep is the XOR
/// ([`Irrep::fuse`]) of the site irreps over every **singly** occupied site left of the bond — a
/// doubly-occupied site contributes `g ⊗ g = IDENTITY` and an empty site nothing, so the flux rides
/// on the odd-electron families [`Family::Up`]/[`Family::Dn`] only (which are the S±½ coupling
/// BRANCHES of one electron, not electron spins).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Su2Sector {
    /// Electron count left of the bond.
    pub n: usize,
    /// Doubled coupled spin `2s` left of the bond.
    pub two_s: usize,
    /// Point-group irrep of the occupation pattern left of the bond. [`Irrep::IDENTITY`] in a
    /// trivial-group ([`PointGroup::TRIVIAL`]) space, which is every space built by
    /// [`Su2Space::new`].
    pub irrep: Irrep,
}

impl Su2Sector {
    /// Convenience constructor, trivial irrep ([`Irrep::IDENTITY`]).
    #[inline]
    pub const fn new(n: usize, two_s: usize) -> Self {
        Su2Sector {
            n,
            two_s,
            irrep: Irrep::IDENTITY,
        }
    }

    /// Convenience constructor carrying an explicit point-group irrep.
    #[inline]
    pub const fn with_irrep(n: usize, two_s: usize, irrep: Irrep) -> Self {
        Su2Sector { n, two_s, irrep }
    }
}

/// Dense `|G| × (N+1) × (N+1)` grid indexed by [`Su2Sector`] (row-major: `irrep` slowest, then
/// `n`, `2s` fastest).
#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Su2SectorGrid<V> {
    n1: usize,
    s1: usize,
    nir: usize,
    data: Vec<V>,
}

impl<V> Su2SectorGrid<V> {
    /// Build a full grid for `space` from a function of the sector. `f` is called in storage
    /// order: `irrep` outermost, then `n`, `2s` innermost (note this is the *opposite* nesting of
    /// `n`/`2s` from [`Su2Space::bond_sectors`] — both orders are load-bearing).
    pub fn from_fn(space: &Su2Space, mut f: impl FnMut(Su2Sector) -> V) -> Self {
        let (n1, s1) = (space.n + 1, space.n + 1);
        let nir = space.pg.order() as usize;
        let mut data = Vec::with_capacity(nir * n1 * s1);
        for ir in 0..nir {
            for n in 0..n1 {
                for two_s in 0..s1 {
                    data.push(f(Su2Sector::with_irrep(n, two_s, Irrep::new(ir as u8))));
                }
            }
        }
        crate::storage_probe::grid_built(
            crate::storage_probe::Table::State,
            std::mem::size_of::<V>(),
            data.len(),
        );
        Su2SectorGrid { n1, s1, nir, data }
    }

    /// Particle/spin grid extents `(N+1, N+1)` (both axes have extent `N+1`: `n ∈ 0..=N`,
    /// `2s ∈ 0..=N`). Deliberately still a 2-tuple (the C ABI marshals a `rows × cols` block);
    /// the irrep extent is [`nirrep`](Self::nirrep).
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        (self.n1, self.s1)
    }

    /// Extent of the irrep axis, i.e. the point-group order the grid was built for.
    #[inline]
    pub fn nirrep(&self) -> usize {
        self.nir
    }

    #[inline]
    fn idx(&self, s: Su2Sector) -> usize {
        debug_assert!(
            s.n < self.n1 && s.two_s < self.s1 && (s.irrep.index() as usize) < self.nir,
            "sector {s:?} out of grid"
        );
        (s.irrep.index() as usize * self.n1 + s.n) * self.s1 + s.two_s
    }

    /// Iterate `(sector, &value)` over the full grid (admissible or not), `2s` fastest, `irrep`
    /// slowest.
    pub fn iter(&self) -> impl Iterator<Item = (Su2Sector, &V)> + '_ {
        let plane = self.n1 * self.s1;
        self.data.iter().enumerate().map(move |(i, v)| {
            let (ir, p) = (i / plane, i % plane);
            (
                Su2Sector::with_irrep(p / self.s1, p % self.s1, Irrep::new(ir as u8)),
                v,
            )
        })
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (Su2Sector, &mut V)> + '_ {
        let (n1, s1) = (self.n1, self.s1);
        let plane = n1 * s1;
        self.data.iter_mut().enumerate().map(move |(i, v)| {
            let (ir, p) = (i / plane, i % plane);
            (
                Su2Sector::with_irrep(p / s1, p % s1, Irrep::new(ir as u8)),
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

    #[inline]
    pub(crate) fn sector_at(&self, i: usize) -> Su2Sector {
        let two_s = i % self.s1;
        let rest = i / self.s1;
        Su2Sector {
            n: rest % self.n1,
            two_s,
            irrep: Irrep::new((rest / self.n1) as u8),
        }
    }
}

impl<V: Clone> Clone for Su2SectorGrid<V> {
    fn clone(&self) -> Self {
        crate::storage_probe::grid_cloned(
            crate::storage_probe::Table::State,
            std::mem::size_of::<V>(),
            self.data.len(),
        );
        Su2SectorGrid {
            n1: self.n1,
            s1: self.s1,
            nir: self.nir,
            data: self.data.clone(),
        }
    }
}

impl<V: Clone> Su2SectorGrid<V> {
    /// Grid with every entry equal to `v`.
    pub fn filled(space: &Su2Space, v: V) -> Self {
        Self::from_fn(space, |_| v.clone())
    }
}

impl<V> std::ops::Index<Su2Sector> for Su2SectorGrid<V> {
    type Output = V;
    #[inline]
    fn index(&self, s: Su2Sector) -> &V {
        &self.data[self.idx(s)]
    }
}

impl<V> std::ops::IndexMut<Su2Sector> for Su2SectorGrid<V> {
    #[inline]
    fn index_mut(&mut self, s: Su2Sector) -> &mut V {
        let i = self.idx(s);
        &mut self.data[i]
    }
}

/// Per-sector bond ranks. Reduced rank =
/// number of SU(2) multiplets, NOT number of abelian states.
pub type Su2RankGrid = Su2SectorGrid<usize>;

impl Su2RankGrid {
    /// All-zero rank grid.
    pub fn zeros(space: &Su2Space) -> Self {
        Self::filled(space, 0)
    }

    /// Sum of all sector ranks.
    pub fn total(&self) -> usize {
        self.data.iter().sum()
    }
}

/// Genealogical forward/backward path-count grid. Saturating `u64` — path counts grow combinatorially.
pub type PathCountGrid = Su2SectorGrid<u64>;

const ODD_ELECTRON: [bool; 4] = [false, true, true, false];

#[inline]
fn child(f: Family, s: Su2Sector, g: Irrep) -> Option<Su2Sector> {
    let t = f.apply(s, g)?;
    let irrep = if ODD_ELECTRON[f as usize] {
        s.irrep.fuse(g)
    } else {
        s.irrep
    };
    debug_assert_eq!(
        t.irrep, irrep,
        "ODD_ELECTRON disagrees with Family::flux for {f:?}"
    );
    Some(Su2Sector::with_irrep(t.n, t.two_s, irrep))
}

/// Forward (`L`) and backward (`R`) genealogical path-count tables over `(n, 2s, irrep)`, one grid
/// per bond `0..=d`. `L[0][(0,0,IDENTITY)] = 1`, `R[d][(N,twoS,target)] = 1`; each interior grid
/// accumulates counts along the four moves `○○, up, dn, ●●`, guarding the
/// `2s−1` underflow on the `dn` move and admissible bounds via [`Su2Space::contains`]./// Site `b`
/// is the site crossed between bonds `b` and `b+1`, so its irrep is `pg.irrep_of(b)`; the fuse is
/// on the CHILD and only for the odd-electron families ([`ODD_ELECTRON`]). XOR is self-inverse, so
/// the backward sweep reuses the same forward [`child`] relation and needs no inverse.
pub fn pathcounts(space: &Su2Space) -> (Vec<PathCountGrid>, Vec<PathCountGrid>) {
    let d = space.d;

    let mut l: Vec<PathCountGrid> = (0..=d).map(|_| PathCountGrid::filled(space, 0)).collect();
    l[0][Su2Sector::new(0, 0)] = 1;
    for b in 0..d {
        let g = space.pg.irrep_of(b);
        for s in space.bond_sectors(b) {
            let c = l[b][s];
            if c == 0 {
                continue;
            }
            for f in Family::ALL {
                if let Some(t) = child(f, s, g)
                    && space.contains(b + 1, t)
                {
                    l[b + 1][t] = l[b + 1][t].saturating_add(c);
                }
            }
        }
    }

    let mut r: Vec<PathCountGrid> = (0..=d).map(|_| PathCountGrid::filled(space, 0)).collect();
    r[d][Su2Sector::with_irrep(space.n, space.two_s, space.target)] = 1;
    for b in (0..d).rev() {
        let g = space.pg.irrep_of(b);
        for s in space.bond_sectors(b) {
            let mut c: u64 = 0;
            for f in Family::ALL {
                if let Some(t) = child(f, s, g)
                    && space.contains(b + 1, t)
                {
                    c = c.saturating_add(r[b + 1][t]);
                }
            }
            r[b][s] = c;
        }
    }

    (l, r)
}

/// Per-bond maximum reduced rank `min(L[b][s], R[b][s])` per `(n, 2s, irrep)` (the `min.(L[k],
/// R[k])`) — REPLACES the abelian binomial `QnSpace::max_rank`; do not use a
/// binomial cap for SU(2) sectors. Unreachable sectors — no left path, or none that completes to
/// `space.target` — get `0`, which is what makes D3's `contains` superset harmless.
pub fn max_ranks(space: &Su2Space) -> Vec<Su2RankGrid> {
    StateSym::max_ranks(space)
}

/// [`max_ranks`] for an OPEN BLOCK of width `rho_d` at bond `d`: `min(L[b][s], ρ_d·R[b][s])` —
/// SU(2) twin of [`crate::qn::max_ranks_open`], in REDUCED ranks. The right genealogical count
/// is multiplied by the width (one column per right path AND block column); the left one and the
/// zero mask are untouched. `max_ranks_open(space, 1)` is [`max_ranks`].
pub fn max_ranks_open(space: &Su2Space, rho_d: usize) -> Vec<Su2RankGrid> {
    StateSym::max_ranks_open(space, rho_d)
}

/// A bond LABEL of a symmetry-label table — the key of every rank grid, path-count table and block
/// grid. The state implementor is [`Su2Sector`] `(n, 2s, irrep)`.
pub trait SymLabel: Copy + Eq + Hash + Debug + Send + Sync + 'static {
    /// The doubled angular momentum `2j` the metric reads. [`crate::su2::theta`] and
    /// [`crate::su2::centerweight`] take this and nothing else of a label, so the θ and
    /// center-weight placement is the same code for every table. `2s` for a state sector.
    fn two_j(self) -> usize;

    /// The integer handed to the abelian chain law's `Sharing::class` — a KEY, not a class (the law
    /// applies its own folding; returning a class here would apply it twice). `n` for a state
    /// sector: the sketch law shares factors by total electron count only, never by spin.
    fn sharing_key(self) -> usize;
}

/// A site STEP of a symmetry-label table: one block family of a core. The state implementor is
/// [`Family`] (`Empty`, `Up`, `Dn`, `Occ`).
pub trait SymStep: Copy + Eq + Debug + Send + Sync + 'static {
    /// The label a step maps between.
    type Label: SymLabel;
    /// The per-site datum a step needs, hoisted once per bond by the kernels
    /// ([`SymSpace::site_ctx`]): the crossed site's orbital irrep for a state step.
    type Ctx: Copy + Debug + Send + Sync + 'static;

    /// Every step, in block-index order. **Load-bearing**: it is the block axis of a core, the
    /// slice order of every unfolding, the summand order of every fold and the ω-factor binding of
    /// the chain draws. For [`Family`]: `Empty, Up, Dn, Occ` (the physical index).
    const ALL: &'static [Self];

    /// Position of `self` in [`ALL`](Self::ALL) (`ALL[s.index()] == s`) — the block index. O(1):
    /// it sits on the block accessors' hot path.
    fn index(self) -> usize;

    /// The column label reached from row label `l` across a site with context `ctx`, or `None`
    /// when the step has no image (a state step: only the `Dn` spin underflow out of `2s = 0`).
    /// Arithmetic only — no admissibility and no grid test; callers filter with
    /// [`SymSpace::admissible`].
    fn step_apply(self, l: Self::Label, ctx: Self::Ctx) -> Option<Self::Label>;

    /// Inverse of [`step_apply`](Self::step_apply): the row label that reaches column label `t`,
    /// or `None` when the inversion underflows. Arithmetic only, and the result may lie OFF the
    /// label grid — the state `Dn` adjoint INCREMENTS `2s`, so from `2s = N` it leaves the
    /// `(N+1)²` plane. Callers keep their grid guard ([`SymGrid::in_bounds`] with a grid in hand,
    /// [`SymSpace::in_grid`] without) as well as their admissibility filter.
    fn step_unapply(self, t: Self::Label, ctx: Self::Ctx) -> Option<Self::Label>;

    /// Doubled spin `2j` of the site factor the step carries: `0` for `Empty`/`Occ`, `1` for the
    /// one-electron branches `Up`/`Dn`.
    fn site_two_j(self) -> usize;

    /// Does the step change the electron count by an odd amount? (The fermionic parity and the
    /// point-group flux ride on it.)
    fn odd(self) -> bool;

    /// The chain-factor slot this step draws from: `< chain_plan().slots()` of the table's space
    /// (`SymSpace::chain_plan`, checked at every draw entry point). Default: the step index — the
    /// identity onto `ω₁…ω₄` for the four state steps (Families4), one slot per step under
    /// PerStep. Slot `0` is the implicit identity under `Omega1::Identity`, so only an
    /// identity-like empty step may take it.
    #[inline]
    fn chain_factor(self) -> usize {
        self.index()
    }
}

/// A dense grid over EVERY label of a space (admissible or not) — the storage of rank grids,
/// path-count tables and block grids. The state implementor is [`Su2SectorGrid`].
pub trait SymGrid<V>: Index<Self::Label, Output = V> + IndexMut<Self::Label> + Sized {
    /// The space a grid is built for.
    type Space;
    /// The label a grid is indexed by.
    type Label: SymLabel;

    /// Build the full grid, calling `f` once per label in storage order.
    fn from_fn(space: &Self::Space, f: impl FnMut(Self::Label) -> V) -> Self;

    /// The grid with every entry equal to `v`.
    fn filled(space: &Self::Space, v: V) -> Self
    where
        V: Clone,
    {
        Self::from_fn(space, |_| v.clone())
    }

    /// `(label, &value)` over the full grid, storage order.
    fn iter<'a>(&'a self) -> impl Iterator<Item = (Self::Label, &'a V)> + 'a
    where
        V: 'a;

    /// `(label, &mut value)` over the full grid, storage order. The slots are disjoint, so a
    /// block-parallel kernel may hand them to separate tasks.
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (Self::Label, &'a mut V)> + 'a
    where
        V: 'a;

    /// The flat storage, storage order.
    fn values(&self) -> &[V];

    /// Mutable twin of [`values`](Self::values).
    fn values_mut(&mut self) -> &mut [V];

    /// The label stored at flat index `i` of [`values`](Self::values) — the exact inverse of the
    /// storage index on every axis.
    fn label_at(&self, i: usize) -> Self::Label;

    /// The length of [`values`](Self::values) of a grid built for `space`, without building one —
    /// the per-step key span of the cores' block store (`crate::blockstore`).
    fn storage_len(space: &Self::Space) -> usize;

    /// The storage-order index of `l` (on the grid, [`SymSpace::in_grid`]) in a grid built for
    /// `space`, without building one: `values()[storage_index(space, l)]` is the entry at `l`.
    /// Part of the storage-order contract above; [`storage_label`](Self::storage_label) inverts it.
    fn storage_index(space: &Self::Space, l: Self::Label) -> usize;

    /// The label at storage index `i` — [`label_at`](Self::label_at) of any grid built for
    /// `space`, without building one.
    fn storage_label(space: &Self::Space, i: usize) -> Self::Label;

    /// Is `l` addressable in this grid? The guard for a label that may come from outside the grid
    /// (a foreign space, or an off-grid [`SymStep::step_unapply`] result): indexing such a label
    /// is a debug-only assertion and, in release, may alias another slot.
    fn in_bounds(&self, l: Self::Label) -> bool;

    /// Feed the grid to `h` exactly as the concrete grid's derived `Hash` would (header and data),
    /// so a fingerprint composed from it keeps its value.
    fn hash_grid<H: Hasher>(&self, h: &mut H)
    where
        V: Hash;

    /// Grid equality exactly as the concrete grid's derived `PartialEq` (header and data). A
    /// method rather than a bound because a block grid of faer matrices has no `PartialEq`.
    fn eq_grid(&self, other: &Self) -> bool
    where
        V: PartialEq;
}

/// A symmetry SPACE: the descriptor every core, sketch and train of one table carries, and the
/// source of its canonical label order. The state implementor is [`Su2Space`].
pub trait SymSpace: Copy + Eq + Hash + Debug + Send + Sync + 'static {
    /// The label of this space's bonds.
    type Label: SymLabel;
    /// The per-site step context ([`SymStep::Ctx`]).
    type Ctx: Copy + Debug + Send + Sync + 'static;

    /// Number of sites `d`: cores `0..d`, bonds `0..=d`.
    fn sites(&self) -> usize;

    /// The total doubled angular momentum `2J` the center weight is normalized by
    /// (`centerweight(2j, 2J)`): `2S` for a state.
    fn total_two_j(&self) -> usize;

    /// The step context of site `k` (the site's orbital irrep for a state), hoisted once per bond.
    fn site_ctx(&self, k: usize) -> Self::Ctx;

    /// Is `l` admissible at bond `b`? Total: `false` for `b > d` and for any off-grid label.
    /// Admissible is a SUPERSET of reachable — unreachable labels carry cap `0` from
    /// [`Symmetry::max_ranks`]. For a state this is [`Su2Space::contains`].
    fn admissible(&self, b: usize, l: Self::Label) -> bool;

    /// Does `l` lie on this space's label grid, i.e. inside every [`SymGrid`] built for it? The
    /// grid guard for an [`SymStep::step_unapply`] result where no grid is in hand.
    fn in_grid(&self, l: Self::Label) -> bool;

    /// The admissible labels at bond `b ∈ 0..=d` in CANONICAL order: exactly
    /// `{l : admissible(b, l)}`. **Load-bearing**: every per-sector loop, first-seen tie-break,
    /// stable sort and RNG stream rides on it. For a state: [`Su2Space::bond_sectors`] — `irrep`
    /// outermost, then `2s`, `n` fastest.
    fn bond_labels(&self, b: usize) -> Vec<Self::Label>;

    /// The left-boundary label at bond `0`: `(0, 0, IDENTITY)` for a state.
    fn root(&self) -> Self::Label;

    /// The right-boundary label at bond `d`, the only one carrying a path: `(N, 2S, target)` for a
    /// state.
    fn top(&self) -> Self::Label;

    /// The `n_total` of the chain draws: `N` for a state. (The default chain RANK is
    /// [`default_chain_rank`](Self::default_chain_rank), which defaults to this value.)
    fn chain_total(&self) -> usize;

    /// The default chain rank `R` of this symmetry's sketch chains. Default
    /// [`chain_total`](Self::chain_total) (`N` for a state).
    #[inline]
    fn default_chain_rank(&self) -> usize {
        self.chain_total()
    }

    /// Is every admissible label reachable (positive cap)? When it is, the kernels skip the
    /// path-count DP that masks unreachable labels — a pure fast path, the values are the same
    /// either way. For a state: the trivial point group.
    fn all_admissible_reachable(&self) -> bool;

    /// May blocks stamped `self` and `other` meet in the plain block folds? For a state: the same
    /// ORBITAL lattice (`n`, `d`, point group), blind to `2S` and to the target irrep — the one
    /// relaxation the shifted folds need, and the guard of
    /// [`crate::su2::coreops::core_times_adjoint_core`] only.
    fn same_lattice(&self, other: &Self) -> bool;
}

/// A symmetry-label table: what the shared SU(2) kernels are generic over.
pub trait Symmetry: Copy + Debug + Send + Sync + 'static {
    /// The bond label.
    type Label: SymLabel;
    /// The per-site step context.
    type Ctx: Copy + Debug + Send + Sync + 'static;
    /// The space descriptor.
    type Space: SymSpace<Label = Self::Label, Ctx = Self::Ctx>;
    /// The site step (block family).
    type Step: SymStep<Label = Self::Label, Ctx = Self::Ctx>;
    /// The dense label grid. The bound on `V` is what keeps `#[derive(Clone, Debug)]` working on
    /// the generic structs; equality and hashing are the methods [`SymGrid::eq_grid`] /
    /// [`SymGrid::hash_grid`], not bounds, because a block grid of faer matrices has neither.
    type Grid<V>: SymGrid<V, Space = Self::Space, Label = Self::Label> + Clone + Debug + Send + Sync
    where
        V: Clone + Debug + Send + Sync;

    /// Every step in block-index order — [`SymStep::ALL`] of the step type, restated so a kernel
    /// writes `S::STEPS`. Never override it.
    const STEPS: &'static [Self::Step] = <Self::Step as SymStep>::ALL;

    /// `θ²(l → t) = (2j_t + 1)/(2j_l + 1)`, the right-canonical dual weight of the step `l → t`
    /// (`rproject`), formed as the INTEGER RATIO. Bit-identical to the state's step-keyed closed
    /// forms (`Up` `(2s+2)/(2s+1)`, `Dn` `2s/(2s+1)`, `Empty`/`Occ` exactly `1`): same
    /// small-integer operands, one rounding. Never `theta(..).powi(2)` — a square root then a
    /// square is not the ratio and differs in the last place.
    #[inline]
    fn theta_sq(l: Self::Label, t: Self::Label) -> f64 {
        (t.two_j() + 1) as f64 / (l.two_j() + 1) as f64
    }

    /// Forward (`L`) and backward (`R`) genealogical path counts over bonds `0..=d`, saturating
    /// `u64`. Provided: [`derived_pathcounts`].
    fn pathcounts(space: &Self::Space) -> PathCounts<Self> {
        derived_pathcounts::<Self>(space)
    }

    /// Per-bond maximum reduced rank `min(L, R)` — the cap every rank grid respects; `0` exactly
    /// on the unreachable labels. An override must keep it `min(L, R)` of
    /// [`pathcounts`](Self::pathcounts): the memoized label tables read the caps off
    /// their path counts by this formula.
    fn max_ranks(space: &Self::Space) -> Vec<Self::Grid<usize>> {
        max_ranks_from::<Self>(space, &Self::pathcounts(space), 1)
    }

    /// [`max_ranks`](Self::max_ranks) for an OPEN BLOCK of width `rho_d` at bond `d`:
    /// `min(L, ρ_d·R)` (one right path per block column). `max_ranks_open(space, 1)` is
    /// `max_ranks(space)`. An override must keep the formula, as for `max_ranks`.
    fn max_ranks_open(space: &Self::Space, rho_d: usize) -> Vec<Self::Grid<usize>> {
        assert!(rho_d >= 1, "max_ranks_open: a block has width ≥ 1");
        max_ranks_from::<Self>(space, &Self::pathcounts(space), rho_d)
    }

    /// The step from the bond-`0` root to the bond-`1` label `q`. Provided: the first of
    /// [`STEPS`](Self::STEPS) whose [`step_apply`](SymStep::step_apply) from
    /// [`root`](SymSpace::root) at site `0` lands on `q`; panics if none does.
    fn boundary_step(space: &Self::Space, q: Self::Label) -> Self::Step {
        let (root, ctx) = (space.root(), space.site_ctx(0));
        Self::STEPS
            .iter()
            .copied()
            .find(|s| s.step_apply(root, ctx) == Some(q))
            .unwrap_or_else(|| {
                panic!("boundary target {q:?} is not a single-site transition from {root:?}")
            })
    }
}

/// The forward (`L`) and backward (`R`) path-count tables of label table `S`, one grid per bond
/// `0..=d` ([`Symmetry::pathcounts`]); `PathCounts<StateSym>` is the pair [`pathcounts`] returns.
pub type PathCounts<S> = (
    Vec<<S as Symmetry>::Grid<u64>>,
    Vec<<S as Symmetry>::Grid<u64>>,
);

fn max_ranks_from<S: Symmetry>(
    space: &S::Space,
    (l, r): &PathCounts<S>,
    rho_d: usize,
) -> Vec<S::Grid<usize>> {
    (0..=space.sites())
        .map(|b| {
            <S::Grid<usize> as SymGrid<usize>>::from_fn(space, |s| {
                l[b][s].min(r[b][s].saturating_mul(rho_d as u64)) as usize
            })
        })
        .collect()
}

/// The path-count DP of [`Symmetry::pathcounts`], written against the table alone: `L[0][root] =
/// 1`, `R[d][top] = 1`, and each bond accumulates along every step whose image is admissible at
/// the next bond, saturating. Walks [`SymSpace::bond_labels`] × [`Symmetry::STEPS`] — the same
/// visit order as the state's [`pathcounts`], to which it is equal at [`StateSym`] (pinned).
pub fn derived_pathcounts<S: Symmetry>(space: &S::Space) -> PathCounts<S> {
    let d = space.sites();
    let zeros = || <S::Grid<u64> as SymGrid<u64>>::filled(space, 0);

    let mut l: Vec<S::Grid<u64>> = (0..=d).map(|_| zeros()).collect();
    l[0][space.root()] = 1;
    for b in 0..d {
        let ctx = space.site_ctx(b);
        for s in space.bond_labels(b) {
            let c = l[b][s];
            if c == 0 {
                continue;
            }
            for &f in S::STEPS {
                if let Some(t) = f.step_apply(s, ctx)
                    && space.admissible(b + 1, t)
                {
                    l[b + 1][t] = l[b + 1][t].saturating_add(c);
                }
            }
        }
    }

    let mut r: Vec<S::Grid<u64>> = (0..=d).map(|_| zeros()).collect();
    r[d][space.top()] = 1;
    for b in (0..d).rev() {
        let ctx = space.site_ctx(b);
        for s in space.bond_labels(b) {
            let mut c: u64 = 0;
            for &f in S::STEPS {
                if let Some(t) = f.step_apply(s, ctx)
                    && space.admissible(b + 1, t)
                {
                    c = c.saturating_add(r[b + 1][t]);
                }
            }
            r[b][s] = c;
        }
    }

    (l, r)
}

#[inline]
pub(crate) fn for_each_nonempty_slot<S: Symmetry>(
    space: &S::Space,
    k: usize,
    rows_storage: &[S::Label],
    row_ranks: &S::Grid<usize>,
    col_ranks: &S::Grid<usize>,
    mut visit: impl FnMut(S::Step, S::Label, S::Label),
) {
    let live: Vec<S::Label> = rows_storage
        .iter()
        .copied()
        .filter(|&s| row_ranks[s] > 0)
        .collect();
    let g = space.site_ctx(k);
    for &f in S::STEPS {
        for &s in &live {
            if let Some(t) = f.step_apply(s, g)
                && space.admissible(k + 1, t)
                && col_ranks[t] > 0
            {
                visit(f, s, t);
            }
        }
    }
}

pub(crate) type Slot<S> = (
    <S as Symmetry>::Step,
    <S as Symmetry>::Label,
    <S as Symmetry>::Label,
);

pub(crate) fn storage_labels<S: Symmetry>(space: &S::Space, b: usize) -> Vec<S::Label> {
    <S::Grid<()> as SymGrid<()>>::filled(space, ())
        .iter()
        .map(|(l, _)| l)
        .filter(|&l| space.admissible(b, l))
        .collect()
}

#[inline]
pub(crate) fn for_each_slot<S: Symmetry>(
    space: &S::Space,
    k: usize,
    rows_storage: &[S::Label],
    mut visit: impl FnMut(S::Step, S::Label, S::Label),
) {
    let g = space.site_ctx(k);
    for &f in S::STEPS {
        for &s in rows_storage {
            if let Some(t) = f.step_apply(s, g)
                && space.admissible(k + 1, t)
            {
                visit(f, s, t);
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) struct LabelTables<S: Symmetry> {
    space: S::Space,
    labels: Vec<Vec<S::Label>>,
    rows: Vec<std::sync::OnceLock<Vec<S::Label>>>,
    slots: Vec<std::sync::OnceLock<Vec<Slot<S>>>>,
    pathcounts: std::sync::OnceLock<PathCounts<S>>,
    max_ranks: std::sync::OnceLock<Vec<S::Grid<usize>>>,
    positions: Vec<std::sync::OnceLock<Vec<u32>>>,
}

#[allow(dead_code)]
impl<S: Symmetry> LabelTables<S> {
    pub(crate) fn new(space: &S::Space) -> Self {
        let d = space.sites();
        LabelTables {
            space: *space,
            labels: (0..=d).map(|b| space.bond_labels(b)).collect(),
            rows: (0..=d).map(|_| std::sync::OnceLock::new()).collect(),
            slots: (0..d).map(|_| std::sync::OnceLock::new()).collect(),
            pathcounts: std::sync::OnceLock::new(),
            max_ranks: std::sync::OnceLock::new(),
            positions: (0..=d).map(|_| std::sync::OnceLock::new()).collect(),
        }
    }

    #[inline]
    pub(crate) fn space(&self) -> &S::Space {
        &self.space
    }

    #[inline]
    pub(crate) fn labels(&self, b: usize) -> &[S::Label] {
        &self.labels[b]
    }

    #[inline]
    pub(crate) fn pos(&self, b: usize, l: S::Label) -> Option<usize> {
        if !self.space.in_grid(l) {
            return None;
        }
        let i = self.positions(b)[<S::Grid<()> as SymGrid<()>>::storage_index(&self.space, l)];
        (i != u32::MAX).then_some(i as usize)
    }

    fn positions(&self, b: usize) -> &[u32] {
        self.positions[b].get_or_init(|| {
            let space = &self.space;
            let mut at = vec![u32::MAX; <S::Grid<()> as SymGrid<()>>::storage_len(space)];
            for (i, &l) in self.labels[b].iter().enumerate() {
                at[<S::Grid<()> as SymGrid<()>>::storage_index(space, l)] = i as u32;
            }
            at
        })
    }

    pub(crate) fn pathcounts(&self) -> &PathCounts<S> {
        self.pathcounts.get_or_init(|| S::pathcounts(&self.space))
    }

    pub(crate) fn max_ranks(&self) -> &[S::Grid<usize>] {
        self.max_ranks
            .get_or_init(|| max_ranks_from::<S>(&self.space, self.pathcounts(), 1))
    }

    pub(crate) fn max_ranks_open(&self, rho_d: usize) -> std::borrow::Cow<'_, [S::Grid<usize>]> {
        assert!(rho_d >= 1, "max_ranks_open: a block has width ≥ 1");
        if rho_d == 1 {
            std::borrow::Cow::Borrowed(self.max_ranks())
        } else {
            std::borrow::Cow::Owned(max_ranks_from::<S>(&self.space, self.pathcounts(), rho_d))
        }
    }

    pub(crate) fn storage_labels(&self, b: usize) -> &[S::Label] {
        self.rows[b].get_or_init(|| {
            let mut rows = self.labels[b].clone();
            rows.sort_unstable_by_key(|&l| {
                <S::Grid<()> as SymGrid<()>>::storage_index(&self.space, l)
            });
            rows
        })
    }

    pub(crate) fn slots(&self, k: usize) -> &[Slot<S>] {
        self.slots[k].get_or_init(|| {
            let mut out = Vec::new();
            for_each_slot::<S>(&self.space, k, self.storage_labels(k), |f, s, t| {
                out.push((f, s, t))
            });
            out
        })
    }
}

impl SymLabel for Su2Sector {
    #[inline]
    fn two_j(self) -> usize {
        self.two_s
    }

    #[inline]
    fn sharing_key(self) -> usize {
        self.n
    }
}

impl SymStep for Family {
    type Label = Su2Sector;
    type Ctx = Irrep;

    const ALL: &'static [Family] = &Family::ALL;

    #[inline]
    fn index(self) -> usize {
        self as usize
    }

    #[inline]
    fn step_apply(self, l: Su2Sector, g: Irrep) -> Option<Su2Sector> {
        self.apply(l, g)
    }

    #[inline]
    fn step_unapply(self, t: Su2Sector, g: Irrep) -> Option<Su2Sector> {
        self.unapply(t, g)
    }

    #[inline]
    fn site_two_j(self) -> usize {
        match self {
            Family::Empty | Family::Occ => 0,
            Family::Up | Family::Dn => 1,
        }
    }

    #[inline]
    fn odd(self) -> bool {
        ODD_ELECTRON[self as usize]
    }
}

impl SymSpace for Su2Space {
    type Label = Su2Sector;
    type Ctx = Irrep;

    #[inline]
    fn sites(&self) -> usize {
        self.d
    }

    #[inline]
    fn total_two_j(&self) -> usize {
        self.two_s
    }

    #[inline]
    fn site_ctx(&self, k: usize) -> Irrep {
        self.pg.irrep_of(k)
    }

    #[inline]
    fn admissible(&self, b: usize, l: Su2Sector) -> bool {
        self.contains(b, l)
    }

    #[inline]
    fn in_grid(&self, l: Su2Sector) -> bool {
        l.n <= self.n && l.two_s <= self.n && l.irrep.index() < self.pg.order()
    }

    #[inline]
    fn bond_labels(&self, b: usize) -> Vec<Su2Sector> {
        self.bond_sectors(b)
    }

    #[inline]
    fn root(&self) -> Su2Sector {
        Su2Sector::new(0, 0)
    }

    #[inline]
    fn top(&self) -> Su2Sector {
        Su2Sector::with_irrep(self.n, self.two_s, self.target)
    }

    #[inline]
    fn chain_total(&self) -> usize {
        self.n
    }

    #[inline]
    fn all_admissible_reachable(&self) -> bool {
        self.pg.is_trivial()
    }

    #[inline]
    fn same_lattice(&self, other: &Su2Space) -> bool {
        self.same_orbital_lattice(other)
    }
}

impl<V> SymGrid<V> for Su2SectorGrid<V> {
    type Space = Su2Space;
    type Label = Su2Sector;

    #[inline]
    fn from_fn(space: &Su2Space, f: impl FnMut(Su2Sector) -> V) -> Self {
        Su2SectorGrid::from_fn(space, f)
    }

    #[inline]
    fn iter<'a>(&'a self) -> impl Iterator<Item = (Su2Sector, &'a V)> + 'a
    where
        V: 'a,
    {
        Su2SectorGrid::iter(self)
    }

    #[inline]
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = (Su2Sector, &'a mut V)> + 'a
    where
        V: 'a,
    {
        Su2SectorGrid::iter_mut(self)
    }

    #[inline]
    fn values(&self) -> &[V] {
        Su2SectorGrid::values(self)
    }

    #[inline]
    fn values_mut(&mut self) -> &mut [V] {
        Su2SectorGrid::values_mut(self)
    }

    #[inline]
    fn label_at(&self, i: usize) -> Su2Sector {
        self.sector_at(i)
    }

    #[inline]
    fn storage_len(space: &Su2Space) -> usize {
        space.pg.order() as usize * (space.n + 1) * (space.n + 1)
    }

    #[inline]
    fn storage_index(space: &Su2Space, l: Su2Sector) -> usize {
        debug_assert!(space.in_grid(l), "sector {l:?} out of grid");
        let n1 = space.n + 1;
        (l.irrep.index() as usize * n1 + l.n) * n1 + l.two_s
    }

    #[inline]
    fn storage_label(space: &Su2Space, i: usize) -> Su2Sector {
        let n1 = space.n + 1;
        let rest = i / n1;
        Su2Sector {
            n: rest % n1,
            two_s: i % n1,
            irrep: Irrep::new((rest / n1) as u8),
        }
    }

    #[inline]
    fn in_bounds(&self, l: Su2Sector) -> bool {
        l.n < self.n1 && l.two_s < self.s1 && (l.irrep.index() as usize) < self.nir
    }

    #[inline]
    fn hash_grid<H: Hasher>(&self, h: &mut H)
    where
        V: Hash,
    {
        Hash::hash(self, h)
    }

    #[inline]
    fn eq_grid(&self, other: &Self) -> bool
    where
        V: PartialEq,
    {
        self == other
    }
}

/// The STATE label table: labels [`Su2Sector`] `(n, 2s, irrep)`, steps [`Family`], space
/// [`Su2Space`], grid [`Su2SectorGrid`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StateSym;

impl Symmetry for StateSym {
    type Label = Su2Sector;
    type Ctx = Irrep;
    type Space = Su2Space;
    type Step = Family;
    type Grid<V>
        = Su2SectorGrid<V>
    where
        V: Clone + Debug + Send + Sync;

    fn pathcounts(space: &Su2Space) -> (Vec<PathCountGrid>, Vec<PathCountGrid>) {
        crate::su2::qn::pathcounts(space)
    }

    fn boundary_step(_space: &Su2Space, q: Su2Sector) -> Family {
        match (q.n, q.two_s) {
            (0, 0) => Family::Empty,
            (1, 1) => Family::Up,
            (2, 0) => Family::Occ,
            _ => panic!("boundary target {q:?} is not a single-site transition from (0,0)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pointgroup::PointGroupKind;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randomized_su2_state, seeded};
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::hash::{Hash, Hasher};

    #[test]
    fn csu2_same_orbital_lattice_is_two_s_and_target_blind() {
        let pg = build_pg(1, 4);
        let base = Su2Space::with_point_group(4, 4, 0, pg, Irrep::IDENTITY);

        for two_s in [0usize, 2, 4] {
            let other = base.with_two_s(two_s);
            assert!(
                base.same_orbital_lattice(&other),
                "same_orbital_lattice must ignore two_s (2S = {two_s})"
            );
            assert_eq!(
                (other.n, other.d, other.pg, other.target),
                (base.n, base.d, base.pg, base.target),
                "with_two_s must change ONLY two_s"
            );
            assert_eq!(other.two_s, two_s);
        }
        let retargeted = Su2Space::with_point_group(4, 4, 0, pg, Irrep::new(1)).with_two_s(2);
        assert!(base.same_orbital_lattice(&retargeted));

        let triplet = base.with_two_s(2);
        assert_ne!(base, triplet, "PartialEq must still compare two_s");
        assert_ne!(
            base.bond_sectors(3),
            triplet.bond_sectors(3),
            "ANTI-VACUITY: the two spaces must admit DIFFERENT bond-3 sectors, or the predicate is \
             as inert here as the abelian same_lattice is there"
        );

        assert!(!base.same_orbital_lattice(&Su2Space::with_point_group(
            4,
            2,
            0,
            pg,
            Irrep::IDENTITY
        )));
        assert!(!base.same_orbital_lattice(&Su2Space::with_point_group(
            5,
            4,
            0,
            build_pg(1, 5),
            Irrep::IDENTITY
        )));
        assert!(!base.same_orbital_lattice(&Su2Space::new(4, 4, 0)));

        let mut rng = seeded(0xC5_0A);
        let x: Su2TtVector<f64> = randomized_su2_state(base, 2, 2, &mut rng);
        let restamped = crate::su2::flux::dress_train(
            &x,
            triplet,
            0,
            &mut crate::su2::spinalgebra::SpinAlgebra::new(),
        );
        assert_eq!(restamped.space, triplet);
        let sketch = crate::su2::coreops::core_times_adjoint_core(x.core(1), restamped.core(1));
        assert_eq!(
            sketch.space, base,
            "the sketch takes the FIRST operand's stamp, as every other fold does"
        );
        let touched: usize = sketch
            .sectors()
            .iter()
            .filter_map(|&l| sketch.block(l).map(|b| b.nrows() * b.ncols()))
            .sum();
        assert!(
            touched > 0,
            "ANTI-VACUITY: the cross-two_s fold produced no block at all, so the relaxed guard was \
             never exercised"
        );
    }

    fn build_pg(nbits: u8, d: usize) -> PointGroup {
        let order = 1u8 << nbits;
        let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k as u8) % order)).collect();
        PointGroup::new(nbits, &irreps)
    }

    fn family_deltas() -> [(usize, i64, bool); 4] {
        let mut out = [(0usize, 0i64, false); 4];
        for (i, f) in Family::ALL.into_iter().enumerate() {
            let (dn, dt) = f.delta();
            out[i] = (dn, dt, dn % 2 == 1);
        }
        out
    }

    fn admissible_nt(
        n_total: usize,
        two_s_total: usize,
        d: usize,
    ) -> Vec<BTreeSet<(usize, usize)>> {
        let deltas: [(i64, i64); 4] = [(0, 0), (1, 1), (1, -1), (2, 0)];
        let mut f: Vec<BTreeSet<(usize, usize)>> = vec![BTreeSet::new(); d + 1];
        f[0].insert((0, 0));
        for k in 0..d {
            let cur: Vec<(usize, usize)> = f[k].iter().copied().collect();
            for (nn, tt) in cur {
                for (dn, dt) in deltas {
                    let n2 = nn as i64 + dn;
                    let t2 = tt as i64 + dt;
                    if t2 < 0 || n2 > n_total as i64 || t2 > n2 {
                        continue;
                    }
                    f[k + 1].insert((n2 as usize, t2 as usize));
                }
            }
        }
        let mut b: Vec<BTreeSet<(usize, usize)>> = vec![BTreeSet::new(); d + 1];
        b[d].insert((n_total, two_s_total));
        for k in (0..d).rev() {
            let cur: Vec<(usize, usize)> = b[k + 1].iter().copied().collect();
            for (nn, tt) in cur {
                for (dn, dt) in deltas {
                    let pn = nn as i64 - dn;
                    let pt = tt as i64 - dt;
                    if pn < 0 || pt < 0 || pt > pn {
                        continue;
                    }
                    b[k].insert((pn as usize, pt as usize));
                }
            }
        }
        (0..=d)
            .map(|k| f[k].intersection(&b[k]).copied().collect())
            .collect()
    }

    type IrGrid = HashMap<(usize, usize, u8), u64>;

    fn brute_force_walks(
        pg: PointGroup,
        n: usize,
        two_s: usize,
        target: Irrep,
        d: usize,
    ) -> (Vec<IrGrid>, Vec<IrGrid>) {
        let admissible = admissible_nt(n, two_s, d);
        let deltas = family_deltas();

        let mut l: Vec<IrGrid> = vec![HashMap::new(); d + 1];
        l[0].insert((0, 0, 0u8), 1);
        for b in 0..d {
            let g = pg.irrep_of(b).index();
            let cur: Vec<((usize, usize, u8), u64)> = l[b].iter().map(|(&k, &c)| (k, c)).collect();
            for ((nn, tt, ir), c) in cur {
                for (dn, dt, odd) in deltas {
                    let n2 = nn as i64 + dn as i64;
                    let t2 = tt as i64 + dt;
                    if t2 < 0 {
                        continue;
                    }
                    let key_nt = (n2 as usize, t2 as usize);
                    if !admissible[b + 1].contains(&key_nt) {
                        continue;
                    }
                    let ir2 = if odd { ir ^ g } else { ir };
                    *l[b + 1].entry((key_nt.0, key_nt.1, ir2)).or_insert(0) += c;
                }
            }
        }

        let mut r: Vec<IrGrid> = vec![HashMap::new(); d + 1];
        r[d].insert((n, two_s, target.index()), 1);
        for b in (0..d).rev() {
            let g = pg.irrep_of(b).index();
            let cur: Vec<((usize, usize, u8), u64)> =
                r[b + 1].iter().map(|(&k, &c)| (k, c)).collect();
            for ((nn, tt, ir), c) in cur {
                for (dn, dt, odd) in deltas {
                    let pn = nn as i64 - dn as i64;
                    let pt = tt as i64 - dt;
                    if pn < 0 || pt < 0 {
                        continue;
                    }
                    let key_nt = (pn as usize, pt as usize);
                    if !admissible[b].contains(&key_nt) {
                        continue;
                    }
                    let ir_parent = if odd { ir ^ g } else { ir };
                    *r[b].entry((key_nt.0, key_nt.1, ir_parent)).or_insert(0) += c;
                }
            }
        }
        (l, r)
    }

    fn check_pathcounts_case(pg: PointGroup, d: usize, n: usize, two_s: usize, target: Irrep) {
        let space = Su2Space::with_point_group(d, n, two_s, pg, target);
        let (l_dp, r_dp) = pathcounts(&space);
        let caps = max_ranks(&space);
        let (l_brute, r_brute) = brute_force_walks(pg, n, two_s, target, d);

        let mut any_reachable = false;
        for b in 0..=d {
            for s in space.bond_sectors(b) {
                let key = (s.n, s.two_s, s.irrep.index());
                let lb = l_brute[b].get(&key).copied().unwrap_or(0);
                let rb = r_brute[b].get(&key).copied().unwrap_or(0);
                assert_eq!(
                    l_dp[b][s], lb,
                    "L mismatch d={d} n={n} two_s={two_s} target={target:?} b={b} s={s:?}"
                );
                assert_eq!(
                    r_dp[b][s], rb,
                    "R mismatch d={d} n={n} two_s={two_s} target={target:?} b={b} s={s:?}"
                );
                let expected_cap = lb.min(rb);
                assert_eq!(
                    caps[b][s] as u64, expected_cap,
                    "cap mismatch d={d} n={n} two_s={two_s} target={target:?} b={b} s={s:?}"
                );
                assert_eq!(
                    caps[b][s] as u64,
                    l_dp[b][s].min(r_dp[b][s]),
                    "internal min(L,R) inconsistency b={b} s={s:?}"
                );
                if expected_cap > 0 {
                    any_reachable = true;
                }
            }
        }
        assert!(
            any_reachable,
            "vacuous case: nothing reachable d={d} n={n} two_s={two_s} target={target:?}"
        );
    }

    #[test]
    fn spg1_sector_new_defaults_to_identity_irrep_and_pins_layout() {
        let s = Su2Sector::new(3, 2);
        assert_eq!(s, Su2Sector::with_irrep(3, 2, Irrep::IDENTITY));
        assert_eq!(s.irrep, Irrep::IDENTITY);
        assert_eq!(std::mem::size_of::<Su2Sector>(), 24);
        assert_eq!(std::mem::size_of::<Su2Space>(), 96);
    }

    #[test]
    fn spg1_space_new_is_trivial_group() {
        let space = Su2Space::new(4, 2, 0);
        assert_eq!(space.pg, PointGroup::TRIVIAL);
        assert_eq!(space.pg.order(), 1);
        assert_eq!(space.target, Irrep::IDENTITY);
    }

    #[test]
    #[should_panic(expected = "out of range 0..")]
    fn spg1_with_point_group_panics_on_target_out_of_range() {
        let _ = Su2Space::with_point_group(1, 0, 0, PointGroup::TRIVIAL, Irrep::new(1));
    }

    #[test]
    #[should_panic(expected = "exceeds the")]
    fn spg1_with_point_group_panics_on_d_exceeds_max_sites_even_at_trivial() {
        let _ = Su2Space::with_point_group(129, 0, 0, PointGroup::TRIVIAL, Irrep::IDENTITY);
    }

    #[test]
    #[should_panic(expected = "cannot exceed 2d")]
    fn spg1_with_point_group_panics_on_n_exceeds_2d() {
        let _ = Su2Space::with_point_group(1, 3, 1, PointGroup::TRIVIAL, Irrep::IDENTITY);
    }

    #[test]
    #[should_panic(expected = "exceeds min(N, 2d-N)")]
    fn spg1_with_point_group_panics_on_two_s_exceeds_cap() {
        let _ = Su2Space::with_point_group(2, 2, 3, PointGroup::TRIVIAL, Irrep::IDENTITY);
    }

    #[test]
    #[should_panic(expected = "must be even")]
    fn spg1_with_point_group_panics_on_odd_n_minus_two_s() {
        let _ = Su2Space::with_point_group(2, 2, 1, PointGroup::TRIVIAL, Irrep::IDENTITY);
    }

    #[test]
    fn spg1_bond_sectors_pin_unchanged_at_trivial() {
        let expected_interior = vec![
            Su2Sector::new(0, 0),
            Su2Sector::new(2, 0),
            Su2Sector::new(1, 1),
        ];
        for space in [
            Su2Space::new(3, 2, 0),
            Su2Space::with_point_group(3, 2, 0, PointGroup::TRIVIAL, Irrep::IDENTITY),
        ] {
            assert_eq!(space.bond_sectors(0), vec![Su2Sector::new(0, 0)]);
            assert_eq!(space.bond_sectors(1), expected_interior);
            assert_eq!(space.bond_sectors(2), expected_interior);
            assert_eq!(space.bond_sectors(3), vec![Su2Sector::new(2, 0)]);
        }
    }

    #[test]
    fn spg1_bond_sectors_irrep_outermost() {
        let d = 5;
        let (n, two_s) = (3, 1);
        let trivial = Su2Space::new(d, n, two_s);
        for nbits in [1u8, 2, 3] {
            let pg = build_pg(nbits, d);
            let order = pg.order();
            let space = Su2Space::with_point_group(d, n, two_s, pg, Irrep::new(0));
            for b in 0..=d {
                let got = space.bond_sectors(b);
                let want: Vec<(usize, usize)> = trivial
                    .bond_sectors(b)
                    .iter()
                    .map(|s| (s.n, s.two_s))
                    .collect();
                assert!(!want.is_empty(), "vacuous particle window at bond {b}");
                assert_eq!(
                    got.len(),
                    want.len() * order as usize,
                    "bond {b} nbits={nbits}: count must scale by |G|"
                );
                for (chunk_idx, chunk) in got.chunks(want.len()).enumerate() {
                    for s in chunk {
                        assert_eq!(
                            s.irrep,
                            Irrep::new(chunk_idx as u8),
                            "bond {b} chunk {chunk_idx}: irrep must be constant within a block \
                             (irrep is the OUTERMOST/slowest axis)"
                        );
                    }
                    let pairs: Vec<(usize, usize)> = chunk.iter().map(|s| (s.n, s.two_s)).collect();
                    assert_eq!(pairs, want, "bond {b} chunk {chunk_idx} nbits={nbits}");
                }
                let irreps: Vec<u8> = got.iter().map(|s| s.irrep.index()).collect();
                assert!(
                    irreps.windows(2).all(|w| w[0] <= w[1]),
                    "bond {b}: irrep sequence must be non-decreasing"
                );
            }
        }
    }

    #[test]
    fn spg1_bond_sectors_equals_contains_set() {
        let d = 5;
        let (n, two_s) = (3, 1);
        let trivial = Su2Space::new(d, n, two_s);
        for nbits in 0u8..=3 {
            let pg = build_pg(nbits, d);
            let order = pg.order();
            let space = Su2Space::with_point_group(d, n, two_s, pg, Irrep::new(0));
            for b in 0..=d {
                let got: HashSet<Su2Sector> = space.bond_sectors(b).into_iter().collect();
                let mut want: HashSet<Su2Sector> = HashSet::new();
                for u in 0..=space.n {
                    for t in 0..=space.n {
                        for ir in 0..order {
                            let s = Su2Sector::with_irrep(u, t, Irrep::new(ir));
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
                assert_eq!(got.len(), trivial.bond_sectors(b).len() * order as usize);
            }
        }
    }

    #[test]
    fn spg1_contains_rejects_out_of_range_irrep_but_accepts_admissible_unreachable() {
        let d = 4;
        let pg = build_pg(2, d);
        let space = Su2Space::with_point_group(d, 2, 0, pg, Irrep::new(0));

        let out_of_range = Su2Sector::with_irrep(0, 0, Irrep::new(4));
        assert!(
            !space.contains(0, out_of_range),
            "irrep out of 0..order() must be rejected"
        );

        let witness = Su2Sector::with_irrep(0, 0, Irrep::new(1));
        assert!(
            space.contains(0, witness),
            "D3: contains must accept every irrep at bond 0"
        );
        let caps = max_ranks(&space);
        assert_eq!(
            caps[0][witness], 0,
            "the reachability-false witness must have cap 0"
        );
    }

    #[test]
    fn spg1_grid_layout_reduces_to_current_at_trivial() {
        let space = Su2Space::new(3, 2, 0);
        let mut visited: Vec<(usize, usize)> = Vec::new();
        let _grid: Su2SectorGrid<usize> = Su2SectorGrid::from_fn(&space, |s| {
            assert_eq!(s.irrep, Irrep::IDENTITY);
            visited.push((s.n, s.two_s));
            visited.len()
        });
        let mut expected = Vec::new();
        for n in 0..=space.n {
            for two_s in 0..=space.n {
                expected.push((n, two_s));
            }
        }
        assert_eq!(visited, expected);

        let n1 = space.n + 1;
        let grid: Su2SectorGrid<usize> = Su2SectorGrid::from_fn(&space, |s| s.n * n1 + s.two_s);
        assert_eq!(grid.shape(), (n1, n1));
        assert_eq!(grid.nirrep(), 1);
        assert_eq!(grid.iter().count(), n1 * n1);
        for n in 0..=space.n {
            for two_s in 0..=space.n {
                assert_eq!(grid[Su2Sector::new(n, two_s)], n * n1 + two_s);
            }
        }
        let seq: Vec<(usize, usize)> = grid.iter().map(|(s, _)| (s.n, s.two_s)).collect();
        assert_eq!(seq, expected);
    }

    #[test]
    fn spg1_grid_irrep_is_slowest_stride() {
        let d = 5;
        let (n, two_s) = (3, 1);
        let pg = build_pg(2, d);
        let space = Su2Space::with_point_group(d, n, two_s, pg, Irrep::new(0));
        let order = space.pg.order() as usize;
        assert_eq!(order, 4);
        let n1 = space.n + 1;
        let plane = n1 * n1;

        let grid: Su2SectorGrid<(u8, usize, usize)> =
            Su2SectorGrid::from_fn(&space, |s| (s.irrep.index(), s.n, s.two_s));
        assert_eq!(grid.nirrep(), order);

        let seq: Vec<Su2Sector> = grid.iter().map(|(s, _)| s).collect();
        assert_eq!(seq.len(), plane * order);
        for (i, s) in seq.iter().enumerate() {
            assert_eq!(
                s.irrep,
                Irrep::new((i / plane) as u8),
                "position {i}: irrep must be the SLOWEST stride"
            );
            assert_eq!(
                grid[*s],
                (s.irrep.index(), s.n, s.two_s),
                "idx round-trip at {s:?}"
            );
        }
        let first_block: Vec<(usize, usize)> =
            seq[0..plane].iter().map(|s| (s.n, s.two_s)).collect();
        for ir in 1..order {
            let block: Vec<(usize, usize)> = seq[ir * plane..(ir + 1) * plane]
                .iter()
                .map(|s| (s.n, s.two_s))
                .collect();
            assert_eq!(
                block, first_block,
                "irrep block {ir} must reproduce block 0's (n,2s) order"
            );
        }
    }

    #[test]
    fn spg1_odd_electron_matches_family_delta_one_site_probe() {
        let d = 2;
        let pg = PointGroup::new(1, &[Irrep::new(1), Irrep::new(0)]);
        let space = Su2Space::with_point_group(d, 2, 0, pg, Irrep::IDENTITY);
        let (l, _r) = pathcounts(&space);

        for f in Family::ALL {
            let (dn, _dt) = f.delta();
            let Some(t) = f.apply(Su2Sector::new(0, 0), space.pg.irrep_of(0)) else {
                continue;
            };
            if !space.contains(1, t) {
                continue;
            }
            let want_irrep = if dn % 2 == 1 {
                Irrep::new(1)
            } else {
                Irrep::IDENTITY
            };
            let other = if dn % 2 == 1 {
                Irrep::IDENTITY
            } else {
                Irrep::new(1)
            };
            assert_eq!(
                l[1][Su2Sector::with_irrep(t.n, t.two_s, want_irrep)],
                1,
                "family {f:?}: expected exactly one path landing with irrep {want_irrep:?}"
            );
            assert_eq!(
                l[1][Su2Sector::with_irrep(t.n, t.two_s, other)],
                0,
                "family {f:?}: must not also land with irrep {other:?}"
            );
        }
    }

    #[test]
    fn spg1_pathcounts_unchanged_at_trivial() {
        for &(d, n, two_s) in &[
            (4usize, 3usize, 1usize),
            (6, 4, 0),
            (6, 4, 2),
            (5, 5, 3),
            (8, 6, 0),
        ] {
            let a = Su2Space::new(d, n, two_s);
            let b = Su2Space::with_point_group(d, n, two_s, PointGroup::TRIVIAL, Irrep::IDENTITY);
            let (la, ra) = pathcounts(&a);
            let (lb, rb) = pathcounts(&b);
            let capsa = max_ranks(&a);
            let capsb = max_ranks(&b);
            for k in 0..=d {
                for s in a.bond_sectors(k) {
                    assert_eq!(
                        la[k][s], lb[k][s],
                        "L d={d} n={n} two_s={two_s} bond={k} s={s:?}"
                    );
                    assert_eq!(
                        ra[k][s], rb[k][s],
                        "R d={d} n={n} two_s={two_s} bond={k} s={s:?}"
                    );
                    assert_eq!(
                        capsa[k][s], capsb[k][s],
                        "caps d={d} n={n} two_s={two_s} bond={k} s={s:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn spg1_target_reachable_only_via_odd_number_of_singly_occupied_sites() {
        let pg = PointGroup::new(1, &[Irrep::new(1), Irrep::new(1)]);
        check_pathcounts_case(pg, 2, 1, 1, Irrep::new(1));
    }

    #[test]
    fn spg1_max_ranks_zero_on_unreachable_target() {
        let pg = PointGroup::new(2, &[Irrep::IDENTITY; 4]);
        let space = Su2Space::with_point_group(4, 2, 0, pg, Irrep::new(1));
        let caps = max_ranks(&space);
        for (b, caps_b) in caps.iter().enumerate() {
            let sectors = space.bond_sectors(b);
            assert!(
                !sectors.is_empty(),
                "bond {b}: D3 superset must still be non-empty"
            );
            for s in sectors {
                assert_eq!(
                    caps_b[s], 0,
                    "bond {b} sector {s:?}: unreachable target must cap at 0"
                );
            }
        }
        let (l, r) = pathcounts(&space);
        for b in 0..=space.d {
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
    fn spg1_irrep_axis_partitions_paths() {
        let d = 4;
        let (n, two_s) = (2, 0);
        let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &[1, 2, 3, 4]).expect("valid orbsym");
        let order = pg.order();
        let trivial = Su2Space::new(d, n, two_s);
        let (l_trivial, r_trivial) = pathcounts(&trivial);

        let l_space = Su2Space::with_point_group(d, n, two_s, pg, Irrep::IDENTITY);
        let (l_pg, _) = pathcounts(&l_space);
        for b in 0..=d {
            for s in trivial.bond_sectors(b) {
                let sum: u64 = (0..order)
                    .map(|ir| l_pg[b][Su2Sector::with_irrep(s.n, s.two_s, Irrep::new(ir))])
                    .sum();
                assert_eq!(sum, l_trivial[b][s], "L sum bond={b} s={s:?}");
            }
        }

        let mut r_per_target = Vec::with_capacity(order as usize);
        for t in 0..order {
            let space_t = Su2Space::with_point_group(d, n, two_s, pg, Irrep::new(t));
            let (_, r_t) = pathcounts(&space_t);
            r_per_target.push(r_t);
        }
        for b in 0..=d {
            for s in trivial.bond_sectors(b) {
                let sum: u64 = r_per_target
                    .iter()
                    .map(|r_t| r_t[b][Su2Sector::with_irrep(s.n, s.two_s, Irrep::IDENTITY)])
                    .sum();
                assert_eq!(sum, r_trivial[b][s], "R sum bond={b} s={s:?}");
            }
        }
    }

    #[test]
    fn spg1_space_eq_hash_discriminates_point_group() {
        let pg_a = PointGroup::new(1, &[Irrep::new(0), Irrep::new(1)]);
        let pg_b = PointGroup::new(1, &[Irrep::new(1), Irrep::new(0)]);
        let space_a = Su2Space::with_point_group(2, 2, 0, pg_a, Irrep::new(0));
        let space_b = Su2Space::with_point_group(2, 2, 0, pg_b, Irrep::new(0));
        let space_c = Su2Space::with_point_group(2, 2, 0, pg_a, Irrep::new(1));

        assert_ne!(space_a, space_b, "differing pg must compare unequal");
        assert_ne!(space_a, space_c, "differing target must compare unequal");
        assert_eq!(space_a, space_a);

        fn hash_of(s: &Su2Space) -> u64 {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            s.hash(&mut h);
            h.finish()
        }
        assert_ne!(
            hash_of(&space_a),
            hash_of(&space_b),
            "differing pg should hash differently"
        );
        assert_ne!(
            hash_of(&space_a),
            hash_of(&space_c),
            "differing target should hash differently"
        );

        let set: HashSet<Su2Space> = [space_a, space_b, space_c].into_iter().collect();
        assert_eq!(
            set.len(),
            3,
            "three distinct spaces must remain distinct in a HashSet"
        );
    }

    #[test]
    fn spg1_edge_case_zero_electrons_grid_shape_and_no_panic() {
        let d = 3;
        let pg = build_pg(2, d);
        let space = Su2Space::with_point_group(d, 0, 0, pg, Irrep::IDENTITY);
        let caps = max_ranks(&space);
        for caps_b in &caps {
            assert_eq!(caps_b.shape(), (1, 1));
            assert_eq!(caps_b.nirrep(), 4);
            for ir in 0..4u8 {
                let s = Su2Sector::with_irrep(0, 0, Irrep::new(ir));
                let _ = caps_b[s];
            }
        }
        assert_eq!(caps[0][Su2Sector::new(0, 0)], 1);
        assert_eq!(caps[d][Su2Sector::new(0, 0)], 1);
    }

    #[test]
    fn spg1_edge_case_high_spin_two_s_equals_n_no_dn_out_of_bond_zero() {
        let d = 3;
        let pg = build_pg(1, d);
        check_pathcounts_case(pg, d, 3, 3, Irrep::new(1));
    }

    #[test]
    fn spg1_edge_case_d_one() {
        let pg = PointGroup::new(1, &[Irrep::new(1)]);
        let space = Su2Space::with_point_group(1, 1, 1, pg, Irrep::new(1));
        let caps = max_ranks(&space);
        assert_eq!(caps[1][Su2Sector::with_irrep(1, 1, Irrep::new(1))], 1);
        assert_eq!(caps[1][Su2Sector::with_irrep(1, 1, Irrep::IDENTITY)], 0);
    }

    #[test]
    fn spg2_max_ranks_positive_on_every_admissible_sector_at_trivial_group() {
        for d in 2..=6usize {
            for n in 0..=2 * d {
                for two_s in 0..=n.min(2 * d - n) {
                    if (n + two_s) % 2 != 0 {
                        continue;
                    }
                    let space = Su2Space::new(d, n, two_s);
                    let caps = max_ranks(&space);
                    for (b, cap) in caps.iter().enumerate() {
                        for s in space.bond_sectors(b) {
                            assert!(
                                cap[s] > 0,
                                "d={d} n={n} 2s={two_s} bond={b} sector={s:?}: admissible but \
                                 cap 0 at |G|=1"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn spg3_h1_max_ranks_positive_lemma_trivial_d1_to_8() {
        for d in 1..=8usize {
            for n in 0..=2 * d {
                for two_s in 0..=n.min(2 * d - n) {
                    if (n + two_s) % 2 != 0 {
                        continue;
                    }
                    let space = Su2Space::new(d, n, two_s);
                    let caps = max_ranks(&space);
                    for (b, cap) in caps.iter().enumerate() {
                        for s in space.bond_sectors(b) {
                            assert!(
                                cap[s] > 0,
                                "d={d} n={n} 2s={two_s} bond={b} sector={s:?}: admissible but \
                                 cap 0 at |G|=1 — the mask-never-fires certificate is FALSE"
                            );
                        }
                    }
                }
            }
        }
    }

    fn forward_reachable(n_total: usize, d: usize) -> Vec<BTreeSet<(usize, usize)>> {
        let deltas: [(i64, i64); 4] = [(0, 0), (1, 1), (1, -1), (2, 0)];
        let mut f: Vec<BTreeSet<(usize, usize)>> = vec![BTreeSet::new(); d + 1];
        f[0].insert((0, 0));
        for k in 0..d {
            let cur: Vec<(usize, usize)> = f[k].iter().copied().collect();
            for (nn, tt) in cur {
                for (dn, dt) in deltas {
                    let n2 = nn as i64 + dn;
                    let t2 = tt as i64 + dt;
                    if t2 < 0 || n2 > n_total as i64 || t2 > n2 {
                        continue;
                    }
                    f[k + 1].insert((n2 as usize, t2 as usize));
                }
            }
        }
        f
    }

    fn backward_reachable(
        n_total: usize,
        two_s_total: usize,
        d: usize,
    ) -> Vec<BTreeSet<(usize, usize)>> {
        let deltas: [(i64, i64); 4] = [(0, 0), (1, 1), (1, -1), (2, 0)];
        let mut b: Vec<BTreeSet<(usize, usize)>> = vec![BTreeSet::new(); d + 1];
        b[d].insert((n_total, two_s_total));
        for k in (0..d).rev() {
            let cur: Vec<(usize, usize)> = b[k + 1].iter().copied().collect();
            for (nn, tt) in cur {
                for (dn, dt) in deltas {
                    let pn = nn as i64 - dn;
                    let pt = tt as i64 - dt;
                    if pn < 0 || pt < 0 || pt > pn {
                        continue;
                    }
                    b[k].insert((pn as usize, pt as usize));
                }
            }
        }
        b
    }

    fn admissible_grid_oracle(n: usize, two_s: usize, d: usize) -> Vec<BTreeSet<(usize, usize)>> {
        let f = forward_reachable(n, d);
        let b = backward_reachable(n, two_s, d);
        (0..=d)
            .map(|k| f[k].intersection(&b[k]).copied().collect())
            .collect()
    }

    type BondCounts = HashMap<(usize, usize), u64>;

    fn brute_force_pathcounts(
        n: usize,
        two_s: usize,
        d: usize,
    ) -> (Vec<BondCounts>, Vec<BondCounts>) {
        let admissible = admissible_grid_oracle(n, two_s, d);
        let deltas: [(i64, i64); 4] = [(0, 0), (1, 1), (1, -1), (2, 0)];

        let mut l: Vec<HashMap<(usize, usize), u64>> = vec![HashMap::new(); d + 1];
        l[0].insert((0, 0), 1);
        for k in 0..d {
            let cur: Vec<((usize, usize), u64)> = l[k].iter().map(|(&s, &c)| (s, c)).collect();
            for ((nn, tt), c) in cur {
                for (dn, dt) in deltas {
                    let n2 = nn as i64 + dn;
                    let t2 = tt as i64 + dt;
                    if t2 < 0 {
                        continue;
                    }
                    let key = (n2 as usize, t2 as usize);
                    if admissible[k + 1].contains(&key) {
                        *l[k + 1].entry(key).or_insert(0) += c;
                    }
                }
            }
        }

        let mut r: Vec<HashMap<(usize, usize), u64>> = vec![HashMap::new(); d + 1];
        r[d].insert((n, two_s), 1);
        for k in (0..d).rev() {
            let cur: Vec<((usize, usize), u64)> = r[k + 1].iter().map(|(&s, &c)| (s, c)).collect();
            for ((nn, tt), c) in cur {
                for (dn, dt) in deltas {
                    let pn = nn as i64 - dn;
                    let pt = tt as i64 - dt;
                    if pn < 0 || pt < 0 {
                        continue;
                    }
                    let key = (pn as usize, pt as usize);
                    if admissible[k].contains(&key) {
                        *r[k].entry(key).or_insert(0) += c;
                    }
                }
            }
        }
        (l, r)
    }

    fn choose(n: usize, k: usize) -> u128 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut acc = 1u128;
        for i in 0..k {
            acc = acc * (n - i) as u128 / (i as u128 + 1);
        }
        acc
    }

    fn lfree_counts(n_cap: usize, d: usize) -> Vec<HashMap<(usize, usize), u64>> {
        let deltas: [(i64, i64); 4] = [(0, 0), (1, 1), (1, -1), (2, 0)];
        let mut l: Vec<HashMap<(usize, usize), u64>> = vec![HashMap::new(); d + 1];
        l[0].insert((0, 0), 1);
        for k in 0..d {
            let cur: Vec<((usize, usize), u64)> = l[k].iter().map(|(&s, &c)| (s, c)).collect();
            for ((nn, tt), c) in cur {
                for (dn, dt) in deltas {
                    let n2 = nn as i64 + dn;
                    let t2 = tt as i64 + dt;
                    if n2 > n_cap as i64 || t2 < 0 || t2 > n_cap as i64 {
                        continue;
                    }
                    *l[k + 1].entry((n2 as usize, t2 as usize)).or_insert(0) += c;
                }
            }
        }
        l
    }

    #[test]
    fn s1_1_boundary_sectors_unique() {
        for &(d, n, two_s) in &[(4, 3, 1), (6, 4, 0), (6, 4, 2), (5, 5, 3), (8, 6, 0)] {
            let space = Su2Space::new(d, n, two_s);
            assert_eq!(space.bond_sectors(0), vec![Su2Sector::new(0, 0)]);
            assert_eq!(space.bond_sectors(d), vec![Su2Sector::new(n, two_s)]);
        }
    }

    #[test]
    fn s1_2_parity_and_admissibility_invariants() {
        for &(d, n, two_s) in &[(6, 4, 2), (5, 5, 3)] {
            let space = Su2Space::new(d, n, two_s);
            for k in 0..=d {
                for s in space.bond_sectors(k) {
                    assert!((s.n + s.two_s).is_multiple_of(2), "parity: {s:?}");
                    assert!(s.two_s <= s.n, "0<=2s<=n: {s:?}");
                    assert!(s.n <= 2 * k, "n<=2k: {s:?} k={k}");
                    assert!(space.contains(k, s), "contains must agree: {s:?} k={k}");
                }
            }
        }
    }

    #[test]
    fn new1_bond_sectors_match_independent_genealogical_walk_oracle() {
        for &(d, n, two_s) in &[(4, 3, 1), (6, 4, 0), (6, 4, 2), (5, 5, 3), (8, 6, 0)] {
            let space = Su2Space::new(d, n, two_s);
            let expected_by_bond = admissible_grid_oracle(n, two_s, d);
            for (k, expected) in expected_by_bond.iter().enumerate() {
                let got: BTreeSet<(usize, usize)> = space
                    .bond_sectors(k)
                    .iter()
                    .map(|s| (s.n, s.two_s))
                    .collect();
                assert_eq!(&got, expected, "d={d} N={n} 2S={two_s} bond={k}");
            }
        }
    }

    #[test]
    fn s1_3_pathcount_binomial_identity() {
        for &d in &[4usize, 6] {
            for &n_cap in &[2usize, 3, 4] {
                let lfree = lfree_counts(n_cap, d);
                for (k, lfree_k) in lfree.iter().enumerate() {
                    for nup in 0..=k.min(n_cap) {
                        for ndn in 0..=k.min(n_cap) {
                            let tot = nup + ndn;
                            if tot > n_cap {
                                continue;
                            }
                            let two_m = nup as i64 - ndn as i64;
                            let lo = two_m.unsigned_abs() as usize;
                            let sum: u64 = (lo..=n_cap)
                                .map(|t| *lfree_k.get(&(tot, t)).unwrap_or(&0))
                                .sum();
                            let want = choose(k, nup) * choose(k, ndn);
                            assert_eq!(
                                sum as u128, want,
                                "d={d} N={n_cap} k={k} nup={nup} ndn={ndn}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn s1_4_pathcounts_boundary_and_bondwise_invariants() {
        for &(d, n, two_s) in &[(6, 4, 2), (5, 3, 1)] {
            let space = Su2Space::new(d, n, two_s);
            let (l, r) = pathcounts(&space);
            assert_eq!(l[0][Su2Sector::new(0, 0)], 1);
            assert_eq!(r[d][Su2Sector::new(n, two_s)], 1);
            let total = l[d][Su2Sector::new(n, two_s)];
            assert_eq!(r[0][Su2Sector::new(0, 0)], total);
            for k in 0..=d {
                let sum: u64 = space
                    .bond_sectors(k)
                    .iter()
                    .map(|&s| l[k][s] * r[k][s])
                    .sum();
                assert_eq!(sum, total, "bond {k}");
            }
        }
    }

    #[test]
    fn new2_pathcounts_match_brute_force_path_enumeration() {
        for &d in &[3usize, 4] {
            for &n in &[2usize, 3, 4] {
                if n > 2 * d {
                    continue;
                }
                let cap = n.min(2 * d - n);
                for two_s in (0..=cap).filter(|t| (n + t).is_multiple_of(2)) {
                    let space = Su2Space::new(d, n, two_s);
                    let (bl, br) = brute_force_pathcounts(n, two_s, d);
                    let (l, r) = pathcounts(&space);
                    for k in 0..=d {
                        for s in space.bond_sectors(k) {
                            let want_l = *bl[k].get(&(s.n, s.two_s)).unwrap_or(&0);
                            let want_r = *br[k].get(&(s.n, s.two_s)).unwrap_or(&0);
                            assert_eq!(
                                l[k][s], want_l,
                                "L mismatch d={d} N={n} 2S={two_s} k={k} s={s:?}"
                            );
                            assert_eq!(
                                r[k][s], want_r,
                                "R mismatch d={d} N={n} 2S={two_s} k={k} s={s:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod memo_tests;
#[cfg(test)]
mod sym_dummy_tests;
#[cfg(test)]
mod sym_tests;
