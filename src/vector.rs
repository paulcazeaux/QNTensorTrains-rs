// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Quantum-number-preserving tensor train vector.

use rand::Rng;

use crate::core::{Family, SparseCore};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{QnSpace, RankGrid, Sector};
use crate::scalar::Scalar;

/// A TT-format quantum state: `d` sparse cores with `d+1` per-sector bond rank grids.
/// Boundary conditions: bond 0 carries only sector `(0,0)` and bond `d` only `(Nup,Ndn)`,
/// both at rank 1 for state vectors.
#[derive(Clone, Debug)]
pub struct TtVector<T: Scalar> {
    /// Symmetry descriptor.
    pub space: QnSpace,
    /// `d+1` bond rank grids (`ranks[b]` = bond `b`).
    pub ranks: Vec<RankGrid>,
    /// The `d` cores.
    pub cores: Vec<SparseCore<T>>,
    /// `Some(k)` when the train is orthogonal with center at core `k`; `None` otherwise.
    pub ortho: Option<usize>,
}

impl<T: Scalar> TtVector<T> {
    /// Assemble a train from cores, deriving the bond rank grids and validating consistency.
    pub fn new(space: QnSpace, cores: Vec<SparseCore<T>>) -> Self {
        assert_eq!(cores.len(), space.d, "expected {} cores", space.d);
        for (k, c) in cores.iter().enumerate() {
            assert_eq!(c.space, space, "core {k} belongs to a different QnSpace");
            assert_eq!(c.k, k, "core at position {k} labeled {}", c.k);
        }
        for k in 1..space.d {
            assert_eq!(
                cores[k - 1].col_ranks,
                cores[k].row_ranks,
                "rank mismatch between cores {} and {k}",
                k - 1
            );
        }
        let mut ranks: Vec<RankGrid> = cores.iter().map(|c| c.row_ranks.clone()).collect();
        ranks.push(cores[space.d - 1].col_ranks.clone());
        TtVector {
            space,
            ranks,
            cores,
            ortho: None,
        }
    }

    /// The core at site `k`.
    #[inline]
    pub fn core(&self, k: usize) -> &SparseCore<T> {
        &self.cores[k]
    }

    /// Rank of sector `s` at bond `b`.
    #[inline]
    pub fn rank(&self, b: usize, s: Sector) -> usize {
        self.ranks[b][s]
    }

    /// Largest per-sector rank over every bond (the quantity an `rmax: PerSector` cap bounds).
    pub fn max_sector_rank(&self) -> usize {
        self.ranks
            .iter()
            .flat_map(|g| g.iter().map(|(_, &r)| r))
            .max()
            .unwrap_or(0)
    }

    /// Max over bonds of the summed-over-sector bond dimension (the quantity a `chi: Pooled` cap
    /// bounds).
    pub fn max_total_bond_rank(&self) -> usize {
        self.ranks
            .iter()
            .map(|g| g.iter().map(|(_, &r)| r).sum::<usize>())
            .max()
            .unwrap_or(0)
    }

    /// Consistency invariants: core/train rank agreement, per-block shape checks (which also
    /// validate the per-site irrep flux, see `SparseCore::check_invariants`), and the boundary law.
    /// Panics on violation.
    pub fn check(&self) -> bool {
        assert_eq!(self.ranks.len(), self.space.d + 1);
        let d = self.space.d;
        let first = Sector::new(0, 0);
        let last = Sector::with_irrep(self.space.nup, self.space.ndn, self.space.target);
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
            assert_eq!(
                c.row_ranks, self.ranks[k],
                "row ranks of core {k} disagree with train"
            );
            assert_eq!(
                c.col_ranks,
                self.ranks[k + 1],
                "column ranks of core {k} disagree with train"
            );
            c.check_invariants();
        }
        true
    }

    /// Dense materialization as a length-`4^d` coefficient vector — the small-`d` test oracle.
    /// Requires unit boundary ranks. The physical index at site `k` is `0` = empty, `1` = up, `2` =
    /// dn, `3` = doubly occupied, and site 0 varies fastest in the linear index. The indexing is
    /// unchanged by the point group.
    pub fn to_dense(&self) -> Vec<T> {
        let d = self.space.d;
        assert!(d <= 16, "to_dense is a small-d test oracle");
        let first = Sector::new(0, 0);
        let last = Sector::with_irrep(self.space.nup, self.space.ndn, self.space.target);
        assert_eq!(self.rank(0, first), 1, "boundary rank at bond 0 must be 1");
        assert_eq!(self.rank(d, last), 1, "boundary rank at bond d must be 1");

        let len = 1usize << (2 * d);
        let mut out = vec![T::zero(); len];
        'config: for (idx, out_v) in out.iter_mut().enumerate() {
            let mut digits = idx;
            let mut s = first;
            let mut v: Vec<T> = vec![T::one()];
            for k in 0..d {
                let f = Family::from_physical_index(digits & 3);
                digits >>= 2;
                let Some(b) = self.cores[k].block(f, s) else {
                    continue 'config;
                };
                debug_assert_eq!(b.nrows(), v.len());
                if b.ncols() == 0 {
                    continue 'config;
                }
                let mut w = vec![T::zero(); b.ncols()];
                for (j, w_j) in w.iter_mut().enumerate() {
                    for (i, &v_i) in v.iter().enumerate() {
                        *w_j += v_i * *b.get(i, j);
                    }
                }
                v = w;
                s = f.apply(s, self.space.pg.irrep_of(k));
            }
            debug_assert_eq!(s, last);
            debug_assert_eq!(v.len(), 1);
            *out_v = v[0];
        }
        out
    }

    /// The DIRECT-SUM embedding of a typed train into the point-group-free space
    /// `QnSpace::new(d, Nup, Ndn)`: the same state, its `|G|` irrep blocks pooled into one bond
    /// index per particle sector.
    pub fn forget_point_group(&self) -> TtVector<T> {
        let space = self.space;
        if space.pg.is_trivial() {
            return self.clone();
        }
        let d = space.d;
        let order = usize::from(space.pg.order());
        let out_space = QnSpace::new(d, space.nup, space.ndn);

        let mut pooled: Vec<RankGrid> = (0..=d).map(|_| RankGrid::zeros(&out_space)).collect();
        let mut offset: Vec<RankGrid> = (0..=d).map(|_| RankGrid::zeros(&space)).collect();
        for b in 0..=d {
            for s in out_space.bond_sectors(b) {
                let mut acc = 0usize;
                for ir in 0..order {
                    let ts = Sector::with_irrep(s.nup, s.ndn, Irrep::new(ir as u8));
                    offset[b][ts] = acc;
                    acc += self.ranks[b][ts];
                }
                pooled[b][s] = acc;
            }
        }

        let cores = (0..d)
            .map(|k| {
                let mut core = SparseCore::zeros(out_space, k, &pooled[k], &pooled[k + 1]);
                let g = space.pg.irrep_of(k);
                for f in Family::ALL {
                    for l in space.bond_sectors(k) {
                        let t = f.apply(l, g);
                        if !space.contains(k + 1, t) {
                            continue;
                        }
                        let Some(src) = self.cores[k].block(f, l) else {
                            continue;
                        };
                        let (i0, j0) = (offset[k][l], offset[k + 1][t]);
                        let lu = Sector::new(l.nup, l.ndn);
                        core.copy_block_into(f, lu, i0, j0, src);
                    }
                }
                core
            })
            .collect();

        let mut out = TtVector::new(out_space, cores);
        out.ortho = self.ortho;
        debug_assert!(out.check());
        out
    }
}

impl TtVector<f64> {
    /// Real → complex embedding: the same space, the same rank grid, every coefficient copied
    /// into the real part.
    pub fn to_c64(&self) -> TtVector<crate::C64> {
        let space = self.space;
        let cores = (0..space.d)
            .map(|k| {
                let src = &self.cores[k];
                let mut dst =
                    SparseCore::<crate::C64>::zeros(space, k, &self.ranks[k], &self.ranks[k + 1]);
                let mut vals = Vec::with_capacity(src.num_coeffs());
                src.for_each_coeff(|c| vals.push(crate::C64::new(c, 0.0)));
                let mut it = vals.into_iter();
                dst.for_each_coeff_mut(|c| *c = it.next().expect("identical coefficient walk"));
                debug_assert!(it.next().is_none(), "coefficient walk length");
                dst
            })
            .collect();
        let mut out = TtVector::new(space, cores);
        out.ortho = self.ortho;
        out
    }
}

/// The all-zero state: unit boundary ranks, every interior rank 0.
pub fn tt_zeros<T: Scalar>(space: QnSpace) -> TtVector<T> {
    let d = space.d;
    let mut ranks: Vec<RankGrid> = (0..=d).map(|_| RankGrid::zeros(&space)).collect();
    ranks[0][Sector::new(0, 0)] = 1;
    ranks[d][Sector::with_irrep(space.nup, space.ndn, space.target)] = 1;
    let cores = (0..d)
        .map(|k| SparseCore::zeros(space, k, &ranks[k], &ranks[k + 1]))
        .collect();
    let tt = TtVector::new(space, cores);
    debug_assert!(tt.check());
    tt
}

fn unit_ranks(space: &QnSpace, caps: &[RankGrid], b: usize) -> RankGrid {
    let mut r = RankGrid::zeros(space);
    for s in space.bond_sectors(b) {
        r[s] = usize::from(caps[b][s] > 0);
    }
    r
}

/// The all-ones coefficient train on the conserved-QN sector: rank 1 on every reachable sector, all
/// block entries 1. Its dense form is the indicator of the `(Nup, Ndn, space.target)` configuration
/// subspace — 1 on every determinant with the right electron counts *and* total irrep
/// `space.target`, 0 elsewhere.
pub fn tt_ones<T: Scalar>(space: QnSpace) -> TtVector<T> {
    let d = space.d;
    let caps = crate::qn::max_ranks(&space);
    assert!(
        caps[d][Sector::with_irrep(space.nup, space.ndn, space.target)] > 0,
        "no determinant has ({}, {}) electrons and irrep {:?} in this space — the target-irrep \
         indicator is the zero vector and has no rank-1 representation",
        space.nup,
        space.ndn,
        space.target
    );
    let ranks: Vec<RankGrid> = (0..=d).map(|b| unit_ranks(&space, &caps, b)).collect();
    let mut cores: Vec<SparseCore<T>> = (0..d)
        .map(|k| SparseCore::zeros(space, k, &ranks[k], &ranks[k + 1]))
        .collect();
    for core in cores.iter_mut() {
        for f in Family::ALL {
            for s in core.transitions(f).collect::<Vec<_>>() {
                if let Some(b) = core.block_mut(f, s)
                    && b.nrows() == 1
                    && b.ncols() == 1
                {
                    *b.get_mut(0, 0) = T::one();
                }
            }
        }
    }
    let tt = TtVector::new(space, cores);
    debug_assert!(tt.check());
    tt
}

/// A single determinant (occupation-basis) state from per-site occupation flags. The `QnSpace` is
/// derived from the flag counts, so it is point-group-free; use [`tt_state_in`] to place a
/// determinant in a space with point-group symmetry.
pub fn tt_state<T: Scalar>(occ_up: &[bool], occ_dn: &[bool]) -> TtVector<T> {
    let d = occ_up.len();
    assert_eq!(occ_dn.len(), d, "occupation vectors must share length d");
    let nup: usize = occ_up.iter().filter(|&&o| o).count();
    let ndn: usize = occ_dn.iter().filter(|&&o| o).count();
    tt_state_in(QnSpace::new(d, nup, ndn), occ_up, occ_dn)
}

/// A single determinant placed in a given [`QnSpace`] — the point-group-aware sibling of
/// [`tt_state`]. The determinant's cumulative bond sector is folded left to right with
/// [`Family::apply`] itself (one source of truth for the flux), so the bond irrep at bond `b` is
/// the XOR of the site irreps over every occupied spin-orbital on sites `0..b`; a doubly-occupied
/// site contributes `g ⊗ g = IDENTITY`.
pub fn tt_state_in<T: Scalar>(space: QnSpace, occ_up: &[bool], occ_dn: &[bool]) -> TtVector<T> {
    let d = space.d;
    assert_eq!(
        occ_up.len(),
        d,
        "occupation vectors must have length d = {d}"
    );
    assert_eq!(
        occ_dn.len(),
        d,
        "occupation vectors must have length d = {d}"
    );
    let nup: usize = occ_up.iter().filter(|&&o| o).count();
    let ndn: usize = occ_dn.iter().filter(|&&o| o).count();
    assert_eq!(
        (nup, ndn),
        (space.nup, space.ndn),
        "determinant occupies ({nup}, {ndn}) electrons, space wants ({}, {})",
        space.nup,
        space.ndn
    );

    let mut cores = Vec::with_capacity(d);
    let mut s = Sector::new(0, 0);
    for k in 0..d {
        let f = Family::from_occupation(occ_up[k], occ_dn[k]);
        cores.push(SparseCore::pure_state(space, k, occ_up[k], occ_dn[k], s));
        s = f.apply(s, space.pg.irrep_of(k));
    }
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    assert_eq!(
        s, last,
        "determinant's total irrep {:?} differs from the space target {:?}",
        s.irrep, space.target
    );

    let tt = TtVector::new(space, cores);
    debug_assert!(tt.check());
    tt
}

/// Gaussian random train with prescribed per-sector bond ranks (`ranks.len() == d+1`, positive
/// entries only on admissible sectors). Blocks are scaled `row_rank^(-1/2)` (expected-isometry
/// convention, see `SparseCore::fill_randn`).
pub fn tt_randn<T: Scalar, R: Rng + ?Sized>(
    space: QnSpace,
    ranks: &[RankGrid],
    rng: &mut R,
) -> TtVector<T> {
    let d = space.d;
    assert_eq!(ranks.len(), d + 1, "need d+1 bond rank grids");
    let caps = crate::qn::max_ranks(&space);
    let masked: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut g = ranks[b].clone();
            for s in space.bond_sectors(b) {
                if caps[b][s] == 0 {
                    g[s] = 0;
                }
            }
            g
        })
        .collect();
    let cores = (0..d)
        .map(|k| {
            let mut c = SparseCore::zeros(space, k, &masked[k], &masked[k + 1]);
            c.fill_randn(rng);
            c
        })
        .collect();
    let tt = TtVector::new(space, cores);
    debug_assert!(tt.check());
    tt
}

/// `U_h x`: the abelian point-group element `h` acting on a train, as one sign per
/// `(site, Family)`. Exact and rank-preserving — no rounding, no reallocation of the rank grid,
/// and `ortho` survives (a per-`(site, Family)` sign is a diagonal orthogonal factor on either
/// unfolding, so `DL` and `HD` are as orthonormal as `L` and `H`).
pub fn apply_group_element<T: Scalar>(x: &TtVector<T>, pg: PointGroup, h: Irrep) -> TtVector<T> {
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
pub fn symmetry_defect<T: Scalar>(x: &TtVector<T>, pg: PointGroup) -> (Irrep, Vec<f64>) {
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

impl<T: Scalar> crate::mem::Footprint for TtVector<T> {
    fn bytes(&self) -> u64 {
        self.cores.bytes()
    }
}

/// `perturbation(tt, r, ϵ)`: builds a random low-rank train `p` at `rp[b][s] = max(1, r[b][s] −
/// tt.ranks[b][s])` at every point-group-REACHABLE sector (`0` at the unreachable ones; accepts
/// `r.len() ∈ {d-1, d+1}`, boundary-completed), scales it to `ϵ·‖tt‖/‖p‖`,
/// and returns `tt + p`. The `d-1`-length completed-rank indexing (two independent 1→0-based
/// shifts) is the highest silent-bug surface in this file.
pub fn perturbation<T: Scalar, R: Rng + ?Sized>(
    x: &TtVector<T>,
    ranks: &[RankGrid],
    eps: f64,
    rng: &mut R,
) -> TtVector<T> {
    let space = x.space;
    let d = space.d;
    assert!(
        ranks.len() == d - 1 || ranks.len() == d + 1,
        "perturbation ranks length must be d-1 or d+1"
    );
    let mut rp = x.ranks.clone();
    let caps = crate::qn::max_ranks(&space);
    let sub1 = |b: usize, s: Sector, target: usize, have: usize| -> usize {
        if caps[b][s] == 0 {
            0
        } else {
            target.saturating_sub(have).max(1)
        }
    };
    if ranks.len() == d + 1 {
        for b in 0..=d {
            for s in space.bond_sectors(b) {
                rp[b][s] = sub1(b, s, ranks[b][s], x.ranks[b][s]);
            }
        }
    } else {
        for b in 1..=d - 1 {
            for s in space.bond_sectors(b) {
                rp[b][s] = sub1(b, s, ranks[b - 1][s], x.ranks[b][s]);
            }
        }
    }
    let p: TtVector<T> = tt_randn(space, &rp, rng);
    let np = p.norm();
    let nx = x.norm();
    let scale = if np > 0.0 { eps * nx / np } else { 0.0 };
    let p_scaled = &p * T::from_f64(scale);
    x + &p_scaled
}

/// `perturbation(tt, R, ϵ)`: every interior bond sector gets rank `r`, boundaries `0` (restored to
/// `1` by [`perturbation`]'s reachability-gated `max(1, ·)` — both boundary bonds have exactly one
/// sector of positive cap), then delegates.
pub fn perturbation_uniform<T: Scalar, R: Rng + ?Sized>(
    x: &TtVector<T>,
    r: usize,
    eps: f64,
    rng: &mut R,
) -> TtVector<T> {
    let space = x.space;
    let d = space.d;
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut g = RankGrid::zeros(&space);
            if b != 0 && b != d {
                for s in space.bond_sectors(b) {
                    g[s] = r;
                }
            }
            g
        })
        .collect();
    perturbation(x, &ranks, eps, rng)
}

#[cfg(test)]
mod pg2_tests {
    use super::{TtVector, tt_randn, tt_state, tt_state_in, tt_zeros};
    use crate::C64;
    use crate::qn::{QnSpace, RankGrid};
    use crate::scalar::Scalar;
    use crate::test_util::{
        assert_purity, c2v_table, config_irrep, find_determinant, irrep_census,
        pick_nontrivial_target,
    };
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    #[should_panic]
    fn check_rejects_row_rank_mismatch_between_core_and_train() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut x: TtVector<f64> = tt_zeros(space);
        let s = space.bond_sectors(2)[0];
        x.cores[2].row_ranks[s] += 1;
        x.check();
    }

    fn tt_state_purity_case<T: Scalar>(d: usize) {
        let pg = c2v_table(d);
        let census = irrep_census(&pg, d);
        let (nup, ndn, target) = pick_nontrivial_target(&census, d);
        let (occ_up, occ_dn) = find_determinant(&pg, d, nup, ndn, target);
        assert_eq!(
            config_irrep(&pg, &occ_up, &occ_dn),
            target,
            "d={d}: find_determinant/config_irrep_from_digits and config_irrep must agree \
             (both independent XOR paths)"
        );

        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
        let x: TtVector<T> = tt_state_in(space, &occ_up, &occ_dn);
        assert!(x.check());
        assert_purity(&pg, d, nup, ndn, target, &x.to_dense());
    }

    fn tt_randn_purity_case<T: Scalar>(d: usize, seed: u64) {
        let pg = c2v_table(d);
        let census = irrep_census(&pg, d);
        let (nup, ndn, target) = pick_nontrivial_target(&census, d);
        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let ranks: Vec<RankGrid> = (0..=d)
            .map(|b| {
                let mut r = RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    r[s] = if b == 0 || b == d {
                        1
                    } else {
                        rng.random_range(1..=3)
                    };
                }
                r
            })
            .collect();
        let x: TtVector<T> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        assert_purity(&pg, d, nup, ndn, target, &x.to_dense());
    }

    #[test]
    fn to_dense_purity_oracle_tt_state_f64() {
        for d in [4usize, 6, 8] {
            tt_state_purity_case::<f64>(d);
        }
    }

    #[test]
    fn to_dense_purity_oracle_tt_state_complex() {
        for d in [4usize, 6, 8] {
            tt_state_purity_case::<C64>(d);
        }
    }

    #[test]
    fn to_dense_purity_oracle_tt_randn_f64() {
        for d in [4usize, 6, 8] {
            tt_randn_purity_case::<f64>(d, 1000 + d as u64);
        }
    }

    #[test]
    fn to_dense_purity_oracle_tt_randn_complex() {
        for d in [4usize, 6, 8] {
            tt_randn_purity_case::<C64>(d, 2000 + d as u64);
        }
    }

    #[test]
    fn tt_state_delegates_to_tt_state_in_at_trivial_group() {
        let d = 5;
        let occ_up = [true, false, false, true, false];
        let occ_dn = [false, true, false, false, true];
        let nup = occ_up.iter().filter(|&&x| x).count();
        let ndn = occ_dn.iter().filter(|&&x| x).count();

        let via_old: TtVector<f64> = tt_state(&occ_up, &occ_dn);
        let space = QnSpace::new(d, nup, ndn);
        let via_new: TtVector<f64> = tt_state_in(space, &occ_up, &occ_dn);

        assert_eq!(via_old.to_dense(), via_new.to_dense());
    }
}

#[cfg(test)]
mod phase1_tests {
    use super::{TtVector, tt_state, tt_zeros};
    use crate::qn::QnSpace;

    #[test]
    fn tt_state_dense_is_a_single_determinant() {
        let d = 8;
        let s_up = [1usize, 3, 5];
        let s_dn = [6usize, 7];
        let occ_up: Vec<bool> = (0..d).map(|k| s_up.contains(&k)).collect();
        let occ_dn: Vec<bool> = (0..d).map(|k| s_dn.contains(&k)).collect();
        let x: TtVector<f64> = tt_state(&occ_up, &occ_dn);
        x.check();

        let expected_idx: usize = (0..d)
            .map(|k| {
                let digit = match (occ_up[k], occ_dn[k]) {
                    (false, false) => 0usize,
                    (true, false) => 1,
                    (false, true) => 2,
                    (true, true) => 3,
                };
                digit << (2 * k)
            })
            .sum();

        let dense = x.to_dense();
        for (idx, &v) in dense.iter().enumerate() {
            if idx == expected_idx {
                assert_eq!(v, 1.0, "determinant coefficient");
            } else {
                assert_eq!(v, 0.0, "config {idx} should be zero");
            }
        }
    }

    #[test]
    fn tt_zeros_dense_is_zero() {
        let x: TtVector<f64> = tt_zeros(QnSpace::from_n_two_sz(8, 4, -2));
        x.check();
        assert!(x.to_dense().iter().all(|&v| v == 0.0));
    }
}

#[cfg(test)]
mod cast_tests {
    use super::{TtVector, tt_randn};
    use crate::C64;
    use crate::pointgroup::{Irrep, PointGroup};
    use crate::qn::{QnSpace, RankGrid, max_ranks};
    use crate::scalar::Scalar;
    use crate::test_util::seeded;

    fn capped_grids(space: &QnSpace, m: usize) -> Vec<RankGrid> {
        max_ranks(space)
            .into_iter()
            .map(|g| {
                let mut out = RankGrid::zeros(space);
                for (s, &c) in g.iter() {
                    out[s] = c.min(m);
                }
                out
            })
            .collect()
    }

    fn coeffs<T: Scalar>(c: &crate::core::SparseCore<T>) -> Vec<T> {
        let mut out = Vec::with_capacity(c.num_coeffs());
        c.for_each_coeff(|v| out.push(v));
        out
    }

    fn assert_nonvacuous(x: &TtVector<f64>) {
        let d = x.space.d;
        let multi = (0..=d).any(|b| {
            x.space
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| x.ranks[b][s] > 0)
                .count()
                > 1
        });
        assert!(multi, "vacuous fixture: no bond carries several sectors");
        assert!(
            (0..d).map(|k| x.cores[k].num_coeffs()).sum::<usize>() > 0,
            "vacuous fixture: no coefficients"
        );
    }

    #[test]
    fn abelian_cast_preserves_structure_and_coefficients() {
        let space = QnSpace::new(6, 3, 3);
        let x = tt_randn::<f64, _>(space, &capped_grids(&space, 4), &mut seeded(0xC0FFEE));
        assert_nonvacuous(&x);

        let y = x.to_c64();
        assert_eq!(y.space, x.space);
        assert_eq!(y.ranks.len(), x.ranks.len());
        for b in 0..=space.d {
            for s in space.bond_sectors(b) {
                assert_eq!(y.ranks[b][s], x.ranks[b][s], "rank at bond {b}");
            }
        }
        assert!(y.check());

        for k in 0..space.d {
            let a = coeffs(&x.cores[k]);
            let b = coeffs(&y.cores[k]);
            assert_eq!(a.len(), b.len(), "coefficient count at site {k}");
            for (i, (&re, &c)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(c.re, re, "site {k} coefficient {i}: real part");
                assert_eq!(c.im, 0.0, "site {k} coefficient {i}: imaginary part");
            }
        }

        let xd = x.to_dense();
        let yd = y.to_dense();
        assert_eq!(xd.len(), yd.len());
        for (i, (&a, &b)) in xd.iter().zip(yd.iter()).enumerate() {
            assert_eq!(b, C64::new(a, 0.0), "dense amplitude {i}");
        }
    }

    #[test]
    fn abelian_cast_carries_a_point_group() {
        let irreps: Vec<Irrep> = [0u8, 1, 2, 3, 1, 2]
            .iter()
            .map(|&i| Irrep::new(i))
            .collect();
        let pg = PointGroup::try_new(3, &irreps).expect("valid table");
        let space = QnSpace::with_point_group(6, 3, 3, pg, Irrep::IDENTITY);
        let x = tt_randn::<f64, _>(space, &capped_grids(&space, 3), &mut seeded(0x5EED));
        assert_nonvacuous(&x);

        let y = x.to_c64();
        assert_eq!(y.space, x.space);
        assert!(y.check());
        for k in 0..space.d {
            let a = coeffs(&x.cores[k]);
            let b = coeffs(&y.cores[k]);
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter()
                    .zip(b.iter())
                    .all(|(&re, &c)| c == C64::new(re, 0.0))
            );
        }
        assert!(space.pg.order() > 1);
    }

    #[test]
    fn cast_carries_a_true_orthogonality_flag() {
        let space = QnSpace::new(6, 3, 3);

        for centre in [space.d - 1, 2] {
            let mut x = tt_randn::<f64, _>(space, &capped_grids(&space, 4), &mut seeded(0xA11CE));
            x.left_orthogonalize(true);
            x.move_core(centre, true);
            assert_eq!(
                x.ortho,
                Some(centre),
                "fixture must be canonical at {centre}"
            );

            let y = x.to_c64();
            assert_eq!(y.ortho, x.ortho, "the flag must survive the embedding");

            let mut z = y.clone();
            z.ortho = None;
            let (shortcut, contraction) = (y.norm(), z.norm());
            assert!(shortcut > 0.0, "vacuous fixture");
            assert!(
                (shortcut - contraction).abs() <= 1e-12 * shortcut,
                "centre {centre}: the inherited flag is false — centre-core shortcut {shortcut} \
                 against full contraction {contraction}"
            );
            assert!(
                (shortcut - x.norm()).abs() <= 1e-12 * shortcut,
                "the embedding must not move the norm"
            );
        }

        let w = tt_randn::<f64, _>(space, &capped_grids(&space, 4), &mut seeded(7));
        assert_eq!(w.ortho, None, "tt_randn is never flagged canonical");
        assert_eq!(w.to_c64().ortho, None);
    }

    #[test]
    fn cast_cost_is_linear_in_the_coefficient_count() {
        let space = QnSpace::new(12, 6, 6);
        let x = tt_randn::<f64, _>(space, &capped_grids(&space, 48), &mut seeded(1));
        let n: usize = (0..space.d).map(|k| x.cores[k].num_coeffs()).sum();
        assert!(
            n > 100_000,
            "fixture is too small to separate n from n²: {n}"
        );

        let t0 = std::time::Instant::now();
        let y = x.to_c64();
        let elapsed = t0.elapsed().as_secs_f64();
        assert_eq!(y.cores[0].num_coeffs(), x.cores[0].num_coeffs());
        assert!(
            elapsed < 5.0,
            "to_c64 took {elapsed:.2} s on {n} coefficients — a linear walk is milliseconds here, so \
             this is the quadratic `coeff`/`set_coeff` shape"
        );
    }
}

#[cfg(test)]
mod vector_forget_pg_tests;
