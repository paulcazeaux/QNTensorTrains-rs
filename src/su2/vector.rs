// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) quantum-number-preserving tensor train vector.

use std::ops::{Add, Div, Mul, Neg, Sub};

use rand::Rng;

use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core, assert_grid_eq};
use crate::su2::qn::{StateSym, Su2RankGrid, Su2Sector, Su2Space, SymGrid, SymSpace, Symmetry};

/// An SU(2) TT-format quantum state: `d` sparse cores with `d+1` per-sector bond rank grids.
/// Boundary conditions: bond 0 carries only sector `(n=0, 2s=0)` and bond `d` only `(n=N,
/// 2s=twoS)`, both at rank 1 for state vectors.
#[derive(Clone, Debug)]
pub struct Su2TtVector<T: Scalar, S: Symmetry = StateSym> {
    /// Symmetry descriptor (carries `N`, `twoS`, `d`).
    pub space: S::Space,
    /// `d+1` bond rank grids (`ranks[b]` = bond `b`). Reduced rank = number of multiplets.
    pub ranks: Vec<S::Grid<usize>>,
    /// The `d` cores.
    pub cores: Vec<Su2Core<T, S>>,
    /// `Some(k)` when the train is orthogonal with center at core `k`; `None` otherwise.
    pub ortho: Option<usize>,
}

impl<T: Scalar, S: Symmetry> Su2TtVector<T, S> {
    /// Assemble a train from cores, deriving the bond rank grids and validating consistency. The
    /// bond rank grids are explicit clones of the cores' row/col ranks, checked for agreement.
    pub fn new(space: S::Space, cores: Vec<Su2Core<T, S>>) -> Self {
        let d = space.sites();
        assert_eq!(cores.len(), d, "expected {d} cores");
        for (k, c) in cores.iter().enumerate() {
            assert_eq!(c.space, space, "core {k} belongs to a different Su2Space");
            assert_eq!(c.k, k, "core at position {k} labeled {}", c.k);
        }
        for k in 1..d {
            assert_grid_eq!(
                cores[k - 1].col_ranks,
                cores[k].row_ranks,
                "rank mismatch between cores {} and {k}",
                k - 1
            );
        }
        let mut ranks: Vec<S::Grid<usize>> = cores.iter().map(|c| c.row_ranks.clone()).collect();
        ranks.push(cores[d - 1].col_ranks.clone());
        Su2TtVector {
            space,
            ranks,
            cores,
            ortho: None,
        }
    }

    /// The core at site `k`.
    #[inline]
    pub fn core(&self, k: usize) -> &Su2Core<T, S> {
        &self.cores[k]
    }

    /// Rank of sector `s` at bond `b`.
    #[inline]
    pub fn rank(&self, b: usize, s: S::Label) -> usize {
        self.ranks[b][s]
    }

    /// Number of lattice sites `d`.
    #[inline]
    pub fn d(&self) -> usize {
        self.space.sites()
    }

    /// `self *= a`: scales ONE core (the standard TT trick), the ortho center if flagged else core
    /// 0 — mirror the abelian `TtVector::scale_mut` (`src/ops.rs`) EXACTLY. Do NOT
    /// implement as `*self = &*self * a` — `Mul<T> for &Su2TtVector` scales core `d-1`,
    /// whereas `scale_mut` scales core `0`; both are mathematically equal but differ in
    /// which core's blocks change, and the sketch caches are keyed on core content.
    pub fn scale_mut(&mut self, a: T) {
        let idx = self.ortho.unwrap_or(0);
        self.cores[idx].scale_mut(a);
    }

    /// Largest single per-sector rank over all bonds (`maximum(maximum.(rank(x)))`;
    /// `TtState::max_sector_rank`, mirror `src/solvers/layer.rs`).
    pub fn max_sector_rank(&self) -> usize {
        self.ranks
            .iter()
            .flat_map(|g| g.iter().map(|(_, &r)| r))
            .max()
            .unwrap_or(0)
    }

    /// Max over bonds of the summed-over-sector bond dimension
    /// (`TtState::max_total_bond_rank`, mirror `src/solvers/layer.rs`).
    pub fn max_total_bond_rank(&self) -> usize {
        self.ranks
            .iter()
            .map(|g| g.iter().map(|(_, &r)| r).sum::<usize>())
            .max()
            .unwrap_or(0)
    }

    /// Consistency invariants (`check`): core/train rank agreement, per-block shape checks (which
    /// also validate the per-site irrep flux, see [`Su2Core::check_invariants`]), and the boundary
    /// law. Panics on violation.
    pub fn check(&self) -> bool {
        assert_eq!(self.ranks.len(), self.space.sites() + 1);
        let d = self.space.sites();
        let first = self.space.root();
        let last = self.space.top();
        for (s, &r) in self.ranks[0].iter() {
            assert!(
                r == 0 || s == first,
                "bond 0: positive rank {r} at {s:?}, but the left boundary sector is {first:?}"
            );
        }
        for (s, &r) in self.ranks[d].iter() {
            assert!(
                r == 0 || s == last,
                "bond {d}: positive rank {r} at {s:?}, but the right boundary sector is {last:?}"
            );
        }
        for (k, c) in self.cores.iter().enumerate() {
            assert_grid_eq!(
                c.row_ranks,
                self.ranks[k],
                "row ranks of core {k} disagree with train"
            );
            assert_grid_eq!(
                c.col_ranks,
                self.ranks[k + 1],
                "column ranks of core {k} disagree with train"
            );
            c.check_invariants();
        }
        true
    }
}

impl<T: Scalar> Su2TtVector<T> {
    /// Total particle number `N`. Used by tests that reformulate the compile-time
    /// `TTvector{T,N,twoS,d}` type assertions as runtime field checks.
    #[inline]
    pub fn n_electrons(&self) -> usize {
        self.space.n
    }

    /// Total doubled spin `2S`.
    #[inline]
    pub fn two_s(&self) -> usize {
        self.space.two_s
    }

    /// The DIRECT-SUM embedding of a typed train into the point-group-free space
    /// `Su2Space::new(d, N, 2S)`: the same state, its `|G|` irrep blocks pooled into one REDUCED
    /// multiplet bond index per `(n, 2s)` sector.
    pub fn forget_point_group(&self) -> Su2TtVector<T> {
        let space = self.space;
        if space.pg.is_trivial() {
            return self.clone();
        }
        let d = space.d;
        let order = usize::from(space.pg.order());
        let out_space = Su2Space::new(d, space.n, space.two_s);

        let mut pooled: Vec<Su2RankGrid> =
            (0..=d).map(|_| Su2RankGrid::zeros(&out_space)).collect();
        let mut offset: Vec<Su2RankGrid> = (0..=d).map(|_| Su2RankGrid::zeros(&space)).collect();
        for b in 0..=d {
            for s in out_space.bond_sectors(b) {
                let mut acc = 0usize;
                for ir in 0..order {
                    let ts = Su2Sector::with_irrep(s.n, s.two_s, Irrep::new(ir as u8));
                    offset[b][ts] = acc;
                    acc += self.ranks[b][ts];
                }
                pooled[b][s] = acc;
            }
        }

        let cores = (0..d)
            .map(|k| {
                let mut core = Su2Core::zeros(out_space, k, &pooled[k], &pooled[k + 1]);
                let g = space.pg.irrep_of(k);
                for f in Family::ALL {
                    for l in space.bond_sectors(k) {
                        let Some(t) = f.apply(l, g) else {
                            continue;
                        };
                        if !space.contains(k + 1, t) {
                            continue;
                        }
                        let Some(src) = self.cores[k].block(f, l) else {
                            continue;
                        };
                        let (i0, j0) = (offset[k][l], offset[k + 1][t]);
                        let lu = Su2Sector::new(l.n, l.two_s);
                        core.copy_block_into(f, lu, i0, j0, src);
                    }
                }
                core
            })
            .collect();

        let mut out = Su2TtVector::new(out_space, cores);
        out.ortho = self.ortho;
        debug_assert!(out.check());
        out
    }
}

impl Su2TtVector<f64> {
    /// Real → complex embedding — SU(2) twin of [`crate::TtVector::to_c64`]. Same space, same
    /// reduced rank grid, every coefficient copied into the real part; linear in the coefficient
    /// count (one [`Su2Core::for_each_coeff`] walk against one
    /// [`Su2Core::for_each_coeff_mut`] walk, not the quadratic `coeff`/`set_coeff` pair), and
    /// `ortho` carried over because it stays true. See the abelian docs for both.
    pub fn to_c64(&self) -> Su2TtVector<crate::C64> {
        let space = self.space;
        let cores = (0..space.d)
            .map(|k| {
                let src = &self.cores[k];
                let mut dst =
                    Su2Core::<crate::C64>::zeros(space, k, &self.ranks[k], &self.ranks[k + 1]);
                let mut vals = Vec::with_capacity(src.num_coeffs());
                src.for_each_coeff(|c| vals.push(crate::C64::new(c, 0.0)));
                let mut it = vals.into_iter();
                dst.for_each_coeff_mut(|c| *c = it.next().expect("identical coefficient walk"));
                debug_assert!(it.next().is_none(), "coefficient walk length");
                dst
            })
            .collect();
        let mut out = Su2TtVector::new(space, cores);
        out.ortho = self.ortho;
        out
    }
}

/// The all-zero state: unit boundary ranks, every interior rank 0. (A rank-vector form with a
/// user-supplied per-bond grid is not implemented; no caller needs it.) The state façade of
/// [`Su2TtVector::zeros_sym`].
#[inline]
pub fn tt_zeros<T: Scalar>(space: Su2Space) -> Su2TtVector<T> {
    Su2TtVector::<T, StateSym>::zeros_sym(space)
}

impl<T: Scalar, S: Symmetry> Su2TtVector<T, S> {
    /// The all-zero train: unit boundary ranks at [`SymSpace::root`] and [`SymSpace::top`], every
    /// interior rank 0 (generic twin of [`tt_zeros`]).
    pub fn zeros_sym(space: S::Space) -> Self {
        let d = space.sites();
        let mut ranks: Vec<S::Grid<usize>> = (0..=d)
            .map(|_| <S::Grid<usize>>::filled(&space, 0))
            .collect();
        ranks[0][space.root()] = 1;
        ranks[d][space.top()] = 1;
        let cores = (0..d)
            .map(|k| Su2Core::<T, S>::zeros_sym(space, k, &ranks[k], &ranks[k + 1]))
            .collect();
        let tt = Self::new(space, cores);
        debug_assert!(tt.check());
        tt
    }

    /// Gaussian random train with prescribed per-sector bond ranks (generic twin of
    /// [`tt_randn`], which documents it).
    pub fn randn_sym<R: Rng + ?Sized>(
        space: S::Space,
        ranks: &[S::Grid<usize>],
        rng: &mut R,
    ) -> Self {
        let d = space.sites();
        assert_eq!(ranks.len(), d + 1, "need d+1 bond rank grids");
        let masked = mask_to_caps::<S>(&space, ranks);
        let grids: &[S::Grid<usize>] = masked.as_deref().unwrap_or(ranks);
        let cores = (0..d)
            .map(|k| {
                let mut c = Su2Core::<T, S>::zeros_sym(space, k, &grids[k], &grids[k + 1]);
                c.fill_randn(rng);
                c
            })
            .collect();
        let tt = Self::new(space, cores);
        debug_assert!(tt.check());
        tt
    }

    /// Gaussian/Stewart-frame random train with a selectable fill mode (generic twin of
    /// [`tt_randn_orth`], which documents it).
    pub fn randn_orth_sym<R: Rng + ?Sized>(
        space: S::Space,
        ranks: &[S::Grid<usize>],
        orthogonal: Orthogonal,
        rng: &mut R,
    ) -> Self {
        let d = space.sites();
        assert_eq!(ranks.len(), d + 1, "need d+1 bond rank grids");
        let masked = mask_to_caps::<S>(&space, ranks);
        let grids: &[S::Grid<usize>] = masked.as_deref().unwrap_or(ranks);
        let cores = (0..d)
            .map(|k| {
                let mut c = Su2Core::<T, S>::zeros_sym(space, k, &grids[k], &grids[k + 1]);
                match orthogonal {
                    Orthogonal::None => c.fill_randn(rng),
                    Orthogonal::Left => c.fill_stewart_left(rng),
                    Orthogonal::Right => c.fill_stewart_right(rng),
                }
                c
            })
            .collect();
        let tt = Self::new(space, cores);
        debug_assert!(tt.check());
        tt
    }
}

/// Rank-1 configuration state function (CSF) from a genealogical coupling path. `codes[k]` is the
/// [`Family`] of the genealogical step at site `k`. Walks `(n,2s)` from `(0,0)` at bond 0;
/// `N`/`twoS`/`d` are INFERRED from the walk endpoint and `codes.len()`. **Panics** if a `Dn` step
/// is taken from a `two_s == 0` sector (spin-lowering from a spin-0 multiplet is invalid).
pub fn tt_csf<T: Scalar>(codes: &[Family]) -> Su2TtVector<T> {
    let (n, two_s) = csf_endpoint(codes);
    tt_csf_in(Su2Space::new(codes.len(), n, two_s), codes)
}

/// Rank-1 CSF on a POINT-GROUP-SYMMETRIC space (sibling of the abelian `crate::vector::tt_state`).
/// `N`/`twoS`/`d` are inferred from the genealogical walk exactly as in
/// [`tt_csf`]; the total irrep is **derived**, never taken from the caller, as the fuse of
/// [`Family::flux`] over the walk. `Up`/`Dn` are the `S±½` COUPLING BRANCHES of ONE electron — both
/// are singly occupied and both carry the site irrep `g_k`; `Occ` carries `g_k ⊗ g_k` and `Empty`
/// nothing. **Panics** on the same `Dn`-from-spin-0 path as [`tt_csf`], and on
/// [`Su2Space::with_point_group`]'s range checks.
pub fn tt_csf_with_point_group<T: Scalar>(codes: &[Family], pg: PointGroup) -> Su2TtVector<T> {
    let (n, two_s) = csf_endpoint(codes);
    let target = codes
        .iter()
        .enumerate()
        .fold(Irrep::IDENTITY, |acc, (k, &f)| {
            acc.fuse(f.flux(pg.irrep_of(k)))
        });
    tt_csf_in(
        Su2Space::with_point_group(codes.len(), n, two_s, pg, target),
        codes,
    )
}

fn csf_endpoint(codes: &[Family]) -> (usize, usize) {
    let (mut n, mut two_s) = (0i64, 0i64);
    for (k, f) in codes.iter().enumerate() {
        let (dn, dt) = f.delta();
        n += dn as i64;
        two_s += dt;
        assert!(
            two_s >= 0,
            "invalid genealogical path: spin-lowering step at site {k} from a spin-0 sector"
        );
    }
    (n as usize, two_s as usize)
}

fn tt_csf_in<T: Scalar>(space: Su2Space, codes: &[Family]) -> Su2TtVector<T> {
    let d = space.d;
    debug_assert_eq!(codes.len(), d);
    let mut cores = Vec::with_capacity(d);
    let mut s = Su2Sector::new(0, 0);
    for (k, &f) in codes.iter().enumerate() {
        cores.push(Su2Core::csf_step(space, k, f, s));
        s = f
            .apply(s, space.pg.irrep_of(k))
            .expect("csf_step accepted this step");
    }
    debug_assert_eq!(
        s,
        Su2Sector::with_irrep(space.n, space.two_s, space.target),
        "CSF walk endpoint disagrees with the derived (N, 2S, target)"
    );
    let tt = Su2TtVector::new(space, cores);
    debug_assert!(tt.check());
    tt
}

fn mask_to_caps<S: Symmetry>(
    space: &S::Space,
    ranks: &[S::Grid<usize>],
) -> Option<Vec<S::Grid<usize>>> {
    (!space.all_admissible_reachable()).then(|| {
        let caps = S::max_ranks(space);
        (0..=space.sites())
            .map(|b| {
                let mut g = ranks[b].clone();
                for s in space.bond_labels(b) {
                    if caps[b][s] == 0 {
                        g[s] = 0;
                    }
                }
                g
            })
            .collect()
    })
}

/// Gaussian random train with prescribed per-sector bond ranks (`ranks.len() == d+1`, positive
/// entries only on admissible sectors). Gaussian fill ONLY — the Stewart-frame modes are
/// [`tt_randn_orth`]. (An interior-only `d-1`-length rank-vector form is not implemented.)
#[inline]
pub fn tt_randn<T: Scalar, R: Rng + ?Sized>(
    space: Su2Space,
    ranks: &[Su2RankGrid],
    rng: &mut R,
) -> Su2TtVector<T> {
    Su2TtVector::<T, StateSym>::randn_sym(space, ranks, rng)
}

/// Stewart-frame mode selector for [`tt_randn_orth`]. `None` is the Gaussian fill of the plain
/// [`tt_randn`]; `Left`/`Right` are the Stewart-frame modes
/// (`Su2Core::fill_stewart_left`/`fill_stewart_right`, `src/su2/core.rs`). A train built by ANY
/// mode is **never** flagged orthogonal (`ortho: None`) — the Stewart fan-in rescale deliberately
/// breaks isometry (oblivious-subspace-injection sketch, not a canonical orthogonal TT).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Orthogonal {
    /// Gaussian fill, `σ = row_rank^{-1/2}` (the [`tt_randn`] path).
    None,
    /// Vertical-unfolding Stewart frame, UNWEIGHTED (`Su2Core::fill_stewart_left`).
    Left,
    /// Horizontal-unfolding Stewart frame, θ-WEIGHTED via `unscale_horizontal`
    /// (`Su2Core::fill_stewart_right`).
    Right,
}

/// Gaussian/Stewart-frame random train with a selectable fill mode, alongside the plain
/// [`tt_randn`] (which keeps its 3-argument signature); `Orthogonal::None` here is equivalent to
/// calling [`tt_randn`]. Result is never flagged orthogonal (`ortho: None`) regardless of mode.
#[inline]
pub fn tt_randn_orth<T: Scalar, R: Rng + ?Sized>(
    space: Su2Space,
    ranks: &[Su2RankGrid],
    orthogonal: Orthogonal,
    rng: &mut R,
) -> Su2TtVector<T> {
    Su2TtVector::<T, StateSym>::randn_orth_sym(space, ranks, orthogonal, rng)
}

impl<T: Scalar, S: Symmetry> Add for &Su2TtVector<T, S> {
    type Output = Su2TtVector<T, S>;
    fn add(self, rhs: &Su2TtVector<T, S>) -> Su2TtVector<T, S> {
        assert_eq!(self.space, rhs.space, "add: Su2Space mismatch");
        let cores = (0..self.space.sites())
            .map(|k| self.cores[k].oplus(&rhs.cores[k]))
            .collect();
        Su2TtVector::new(self.space, cores)
    }
}

impl<T: Scalar, S: Symmetry> Sub for &Su2TtVector<T, S> {
    type Output = Su2TtVector<T, S>;
    fn sub(self, rhs: &Su2TtVector<T, S>) -> Su2TtVector<T, S> {
        assert_eq!(self.space, rhs.space, "sub: Su2Space mismatch");
        let cores = (0..self.space.sites())
            .map(|k| {
                if k == 0 {
                    self.cores[0].oplus(&rhs.cores[0].neg())
                } else {
                    self.cores[k].oplus(&rhs.cores[k])
                }
            })
            .collect();
        Su2TtVector::new(self.space, cores)
    }
}

impl<T: Scalar, S: Symmetry> Neg for &Su2TtVector<T, S> {
    type Output = Su2TtVector<T, S>;
    fn neg(self) -> Su2TtVector<T, S> {
        let mut out = self.clone();
        let idx = out.ortho.unwrap_or(0);
        out.cores[idx].scale_mut(T::from_f64(-1.0));
        out
    }
}

impl<T: Scalar, S: Symmetry> Mul<T> for &Su2TtVector<T, S> {
    type Output = Su2TtVector<T, S>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, b: T) -> Su2TtVector<T, S> {
        let mut out = self.clone();
        let idx = out.ortho.unwrap_or(out.space.sites() - 1);
        out.cores[idx].scale_mut(b);
        out
    }
}

impl<T: Scalar, S: Symmetry> Div<T> for &Su2TtVector<T, S> {
    type Output = Su2TtVector<T, S>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, b: T) -> Su2TtVector<T, S> {
        self * b.recip()
    }
}

impl<S: Symmetry> Mul<&Su2TtVector<f64, S>> for f64 {
    type Output = Su2TtVector<f64, S>;
    fn mul(self, x: &Su2TtVector<f64, S>) -> Su2TtVector<f64, S> {
        let mut out = x.clone();
        let idx = out.ortho.unwrap_or(0);
        out.cores[idx].scale_mut(self);
        out
    }
}

impl<S: Symmetry> Mul<&Su2TtVector<crate::scalar::C64, S>> for crate::scalar::C64 {
    type Output = Su2TtVector<crate::scalar::C64, S>;
    fn mul(self, x: &Su2TtVector<crate::scalar::C64, S>) -> Su2TtVector<crate::scalar::C64, S> {
        let mut out = x.clone();
        let idx = out.ortho.unwrap_or(0);
        out.cores[idx].scale_mut(self);
        out
    }
}

/// `U_h x`: the abelian point-group element `h` acting on a train, as one sign per
/// `(site, Family)`. Exact and rank-preserving — no rounding, no reallocation of the rank grid.
pub fn apply_group_element<T: Scalar>(
    x: &Su2TtVector<T>,
    pg: PointGroup,
    h: Irrep,
) -> Su2TtVector<T> {
    let space = x.space;
    let mut out = x.clone();
    let minus = T::from_f64(-1.0);
    for k in 0..space.d {
        if Family::Up.flux(pg.irrep_of(k)).character(h) > 0.0 {
            continue;
        }
        let sectors = space.bond_sectors(k);
        let core = &mut out.cores[k];
        for f in [Family::Up, Family::Dn] {
            for &l in &sectors {
                if let Some(mut b) = core.block_mut(f, l) {
                    for j in 0..b.ncols() {
                        for i in 0..b.nrows() {
                            b[(i, j)] *= minus;
                        }
                    }
                }
            }
        }
    }
    out
}

/// The train's total irrep, and how far it is from being a symmetry eigenstate.
pub fn symmetry_defect<T: Scalar>(x: &Su2TtVector<T>, pg: PointGroup) -> (Irrep, Vec<f64>) {
    let order = pg.order();
    let nrm = x.norm();
    let mut gamma = 0u8;
    let mut defects = Vec::new();
    let mut j = 0u8;
    while (1u8 << j) < order {
        let h = Irrep::new(1u8 << j);
        let ux = apply_group_element(x, pg, h);
        let plus = (&ux - x).norm() / nrm.max(f64::MIN_POSITIVE);
        let minus = (&ux + x).norm() / nrm.max(f64::MIN_POSITIVE);
        if minus < plus {
            gamma |= 1u8 << j;
            defects.push(minus);
        } else {
            defects.push(plus);
        }
        j += 1;
    }
    (Irrep::new(gamma), defects)
}

impl<T: Scalar, S: Symmetry> crate::mem::Footprint for Su2TtVector<T, S> {
    fn bytes(&self) -> u64 {
        self.cores.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::C64;
    use crate::su2::qn::Su2SectorGrid;
    use crate::test_util::{seeded, su2_capped_grids, su2_coeffs};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn build_pg(nbits: u8, d: usize) -> PointGroup {
        let order = 1u8 << nbits;
        let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k as u8) % order)).collect();
        PointGroup::new(nbits, &irreps)
    }

    #[test]
    fn spg2_tt_csf_target_matches_manual_xor_over_singly_occupied_sites() {
        let d = 4;
        let pg = build_pg(2, d);
        let codes = [Family::Up, Family::Up, Family::Dn, Family::Occ];
        let manual_target =
            codes
                .iter()
                .enumerate()
                .fold(Irrep::IDENTITY, |acc, (k, &f)| match f {
                    Family::Up | Family::Dn => acc.fuse(pg.irrep_of(k)),
                    Family::Empty | Family::Occ => acc,
                });
        assert_ne!(
            manual_target,
            Irrep::IDENTITY,
            "fixture bug: expected a nonidentity target"
        );

        let x: Su2TtVector<f64> = tt_csf_with_point_group(&codes, pg);
        assert_eq!(x.space.target, manual_target);
        assert!(x.check());
    }

    #[test]
    fn spg2_tt_csf_target_all_paired_sites_is_identity() {
        let d = 3;
        let pg = build_pg(2, d);
        let codes = vec![Family::Occ; d];
        let x: Su2TtVector<f64> = tt_csf_with_point_group(&codes, pg);
        assert_eq!(x.space.target, Irrep::IDENTITY);
        assert!(x.check());
    }

    #[test]
    fn spg2_tt_csf_unchanged_at_trivial_group() {
        let codes = [
            Family::Up,
            Family::Dn,
            Family::Up,
            Family::Occ,
            Family::Empty,
        ];
        let via_old: Su2TtVector<f64> = tt_csf(&codes);
        let via_new: Su2TtVector<f64> = tt_csf_with_point_group(&codes, PointGroup::TRIVIAL);
        assert_eq!(via_old.space, via_new.space);
        assert_eq!(via_old.ranks, via_new.ranks);
        for k in 0..codes.len() {
            let (co, cn) = (via_old.core(k), via_new.core(k));
            assert_eq!(co.num_coeffs(), cn.num_coeffs());
            for i in 0..co.num_coeffs() {
                assert_eq!(
                    co.coeff(i),
                    cn.coeff(i),
                    "core {k} coeff {i} differs at |G|=1"
                );
            }
        }
    }

    #[test]
    fn spg2_tt_randn_matches_across_new_and_with_point_group_trivial_at_fixed_seed() {
        let d = 5;
        let space_new = Su2Space::new(d, 4, 2);
        let space_pg = Su2Space::with_point_group(d, 4, 2, PointGroup::TRIVIAL, Irrep::IDENTITY);
        assert_eq!(space_new, space_pg);

        let ranks: Vec<Su2RankGrid> = (0..=d)
            .map(|b| {
                Su2SectorGrid::from_fn(&space_new, |s| if space_new.contains(b, s) { 2 } else { 0 })
            })
            .collect();

        let mut rng_a = Xoshiro256PlusPlus::seed_from_u64(0xC0FF_EE01);
        let mut rng_b = Xoshiro256PlusPlus::seed_from_u64(0xC0FF_EE01);
        let xa: Su2TtVector<f64> = tt_randn(space_new, &ranks, &mut rng_a);
        let xb: Su2TtVector<f64> = tt_randn(space_pg, &ranks, &mut rng_b);

        for k in 0..d {
            assert_eq!(xa.core(k).num_coeffs(), xb.core(k).num_coeffs());
            for i in 0..xa.core(k).num_coeffs() {
                assert_eq!(
                    xa.core(k).coeff(i),
                    xb.core(k).coeff(i),
                    "core {k} coeff {i}"
                );
            }
        }
        assert_eq!(
            rng_a.next_u64(),
            rng_b.next_u64(),
            "RNG end state must match: draw COUNT, not just values, must be identical"
        );
    }

    #[test]
    fn su2_cast_preserves_structure_and_coefficients() {
        let space = Su2Space::new(6, 6, 0);
        let x = tt_randn::<f64, _>(space, &su2_capped_grids(&space, 4), &mut seeded(0xBEEF));
        assert!(x.max_sector_rank() > 1, "vacuous fixture");

        let y = x.to_c64();
        assert_eq!(y.space, x.space);
        for b in 0..=space.d {
            for s in space.bond_sectors(b) {
                assert_eq!(y.ranks[b][s], x.ranks[b][s], "rank at bond {b}");
            }
        }
        assert!(y.check());

        for k in 0..space.d {
            let a = su2_coeffs(&x.cores[k]);
            let b = su2_coeffs(&y.cores[k]);
            assert_eq!(a.len(), b.len(), "coefficient count at site {k}");
            for (i, (&re, &c)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(c, C64::new(re, 0.0), "site {k} coefficient {i}");
            }
        }
    }

    #[test]
    fn su2_cast_carries_a_true_orthogonality_flag() {
        let space = Su2Space::new(6, 6, 0);
        let mut x = tt_randn::<f64, _>(space, &su2_capped_grids(&space, 4), &mut seeded(0xBEE));
        x.left_orthogonalize(true);
        assert_eq!(x.ortho, Some(space.d - 1));

        let y = x.to_c64();
        assert_eq!(y.ortho, x.ortho);

        let mut z = y.clone();
        z.ortho = None;
        let (shortcut, contraction) = (y.norm(), z.norm());
        assert!(shortcut > 0.0, "vacuous fixture");
        assert!(
            (shortcut - contraction).abs() <= 1e-12 * shortcut,
            "the inherited flag is false: centerweighted centre-core norm {shortcut} against \
             full contraction {contraction}"
        );
        assert!((shortcut - x.norm()).abs() <= 1e-12 * shortcut);
    }
}

#[cfg(test)]
mod s1_tests {
    use super::{Su2TtVector, tt_csf, tt_zeros as su2_tt_zeros};
    use crate::su2::core::Family as Su2Family;
    use crate::su2::expand::expand;
    use crate::su2::ops::dot as su2_dot;
    use crate::su2::qn::{Su2Sector, Su2Space};
    use crate::test_util::randomized_su2_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn s1_7_tt_zeros_valid_zero_norm_expands_to_zero() {
        let space = Su2Space::new(6, 4, 2);
        let x: Su2TtVector<f64> = su2_tt_zeros(space);
        assert!(x.check());
        assert_eq!(x.norm(), 0.0);
        let e = expand(&x, x.two_s() as i64);
        assert_eq!(e.norm(), 0.0);
    }

    #[test]
    #[should_panic(expected = "spin-lowering")]
    fn s1_9_tt_csf_invalid_path_panics() {
        let _: Su2TtVector<f64> = tt_csf(&[Su2Family::Dn, Su2Family::Up]);
    }

    #[test]
    fn s1_11_tt_randn_boundary_ranks_and_norm_via_dot() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1234_5678);
        let space = Su2Space::new(6, 4, 0);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 0, 3, &mut rng);
        assert!(x.check());
        assert_eq!(x.rank(0, Su2Sector::new(0, 0)), 1);
        assert_eq!(x.rank(6, Su2Sector::new(4, 0)), 1);
        let via_dot = su2_dot(&x, &x).abs().sqrt();
        assert!((x.norm() - via_dot).abs() < 1e-12 * via_dot.max(1.0));
    }

    #[test]
    fn s1_12_rank_accessors_agree_with_grid() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xABCD_EF01);
        let space = Su2Space::new(5, 3, 1);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        for k in 0..=space.d {
            for s in space.bond_sectors(k) {
                assert_eq!(x.rank(k, s), x.ranks[k][s]);
                if k < space.d {
                    assert_eq!(x.ranks[k][s], x.core(k).row_ranks[s], "row k={k} s={s:?}");
                }
                if k > 0 {
                    assert_eq!(
                        x.ranks[k][s],
                        x.core(k - 1).col_ranks[s],
                        "col k={k} s={s:?}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod su2_forget_pg_tests;
