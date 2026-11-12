// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! **CAS(n,m) as a sector-pinned tensor train** — the substrate the joint (TT ⊕ orbital) stack
//! needs in order to run on a complete-active-space manifold.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
use crate::pointgroup::PointGroup;
use crate::qn::{PerSector, QnSpace, RankGrid, Sector, max_ranks};
use crate::rotation::PairMask;
use crate::solvers::als::{AlsOptions, als};
use crate::solvers::hess::{NewtonPolishOpts, newton_polish};
use crate::solvers::riemannian::normalize_mut;
use crate::vector::perturbation;
use crate::vector::{TtVector, tt_state_in};

const SEED_EPS: f64 = 0.2;

/// The three-way split of an ordered orbital chain into core / active / virtual windows.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CasPartition {
    /// Number of doubly-occupied core orbitals (chain prefix).
    pub n_core: usize,
    /// Number of active orbitals (the free window).
    pub n_active: usize,
    /// Number of empty virtual orbitals (chain suffix).
    pub n_virt: usize,
}

impl CasPartition {
    /// Total orbital count `n_core + n_active + n_virt`; must equal `space.d`.
    #[inline]
    pub fn d(&self) -> usize {
        self.n_core + self.n_active + self.n_virt
    }

    /// Panic unless this partition admits the sector pins of §1 in `space` (crate style:
    /// panic-on-impossible, cf. [`QnSpace::new`]). Every message names the offending field.
    pub fn validate(&self, space: &QnSpace) {
        assert_eq!(
            self.d(),
            space.d,
            "CasPartition n_core={} + n_active={} + n_virt={} = {} must cover the chain (d = {})",
            self.n_core,
            self.n_active,
            self.n_virt,
            self.d(),
            space.d
        );
        assert!(
            self.n_core <= space.nup && self.n_core <= space.ndn,
            "CasPartition n_core = {} exceeds an electron count (N↑ = {}, N↓ = {}): a \
             doubly-occupied core prefix needs n_core electrons of EACH spin, so the (b,b) prefix \
             pin has no representation",
            self.n_core,
            space.nup,
            space.ndn
        );
        assert!(
            space.nup - self.n_core <= self.n_active && space.ndn - self.n_core <= self.n_active,
            "CasPartition n_active = {} cannot hold the {} up / {} down active electrons left by \
             n_core = {} (the virtual suffix is empty by construction)",
            self.n_active,
            space.nup - self.n_core,
            space.ndn - self.n_core,
            self.n_core
        );
        assert!(
            self.n_active >= 1,
            "CasPartition n_active = 0 leaves no free bond: the pinned state would be a single \
             determinant, which is a different object (escalate if that is wanted)"
        );
        if space.pg.order() > 1 {
            let act = active_space(space, self);
            let caps = max_ranks(&act);
            let last = Sector::with_irrep(act.nup, act.ndn, act.target);
            assert!(
                caps[self.n_active][last] > 0,
                "CasPartition: target irrep {:?} is unreachable from the active window \
                 (n_core = {}, n_active = {}) — the core prefix and virtual suffix both flux to \
                 IDENTITY, so the whole target must be carried by the active orbitals",
                space.target,
                self.n_core,
                self.n_active
            );
        }
    }
}

fn active_space(space: &QnSpace, part: &CasPartition) -> QnSpace {
    let (nup, ndn) = (space.nup - part.n_core, space.ndn - part.n_core);
    if space.pg.order() <= 1 {
        return QnSpace::new(part.n_active, nup, ndn);
    }
    let irreps: Vec<_> = (part.n_core..part.n_core + part.n_active)
        .map(|k| space.pg.irrep_of(k))
        .collect();
    let pg_act = PointGroup::new(space.pg.nbits(), &irreps);
    QnSpace::with_point_group(part.n_active, nup, ndn, pg_act, space.target)
}

/// The **pinned rank grid** of §1: `space.d + 1` per-bond grids (bond `b` at index `b`, the
/// convention [`TtVector::ranks`](crate::vector::TtVector::ranks),
/// [`round_ranks_mut`](crate::vector::TtVector::round_ranks_mut) and [`max_ranks`] share).
pub fn cas_rank_grid(space: &QnSpace, part: &CasPartition, m_active: PerSector) -> Vec<RankGrid> {
    part.validate(space);
    let d = space.d;
    let (ni, na) = (part.n_core, part.n_active);
    let mut out: Vec<RankGrid> = (0..=d).map(|_| RankGrid::zeros(space)).collect();

    for (b, grid) in out.iter_mut().enumerate().take(ni + 1) {
        grid[Sector::new(b, b)] = 1;
    }
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    for grid in out.iter_mut().skip(ni + na) {
        grid[last] = 1;
    }

    let act = active_space(space, part);
    let caps = max_ranks(&act);
    for j in 0..=na {
        for sa in act.bond_sectors(j) {
            let cap = caps[j][sa];
            if cap == 0 {
                continue;
            }
            let s = Sector::with_irrep(sa.nup + ni, sa.ndn + ni, sa.irrep);
            debug_assert!(
                space.contains(ni + j, s),
                "active sector {sa:?} shifted to {s:?} is inadmissible at bond {}",
                ni + j
            );
            out[ni + j][s] = if m_active.is_none() {
                cap
            } else {
                m_active.0.min(cap)
            };
        }
    }

    debug_assert_eq!(out.len(), d + 1);
    debug_assert_eq!(out[0].total(), 1, "boundary law at bond 0");
    debug_assert_eq!(out[d].total(), 1, "boundary law at bond d");
    debug_assert!(
        (0..=d).all(|b| b <= ni || b >= ni + na || out[b].total() >= 1),
        "an active-window bond ended up empty"
    );
    out
}

/// The CASSCF **non-redundant** orbital-rotation pairs: `{core–active, core–virtual,
/// active–virtual}`, i.e. every pair whose two orbitals sit in different windows.
pub fn cas_pairs(part: &CasPartition, d: usize) -> PairMask {
    assert_eq!(
        part.d(),
        d,
        "cas_pairs: partition covers {} orbitals, caller passed d = {d}",
        part.d()
    );
    let window = |i: usize| -> u8 {
        if i < part.n_core {
            0
        } else if i < part.n_core + part.n_active {
            1
        } else {
            2
        }
    };
    PairMask::with_predicate(d, |p, q| window(p) != window(q))
}

fn grid_le(space: &QnSpace, got: &[RankGrid], want: &[RankGrid]) -> bool {
    got.len() == want.len()
        && (0..=space.d).all(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .all(|s| got[b][s] <= want[b][s])
        })
}

/// A converged ground state **on the pinned CAS manifold**: returns `(electronic λ, unit-norm
/// state)`. The caller adds `ints.e0` for a total energy (the standing convention).
pub fn cas_reference_state(
    h: &SparseHamiltonian<f64>,
    space: &QnSpace,
    part: &CasPartition,
    m_active: usize,
    seed: u64,
) -> (f64, TtVector<f64>) {
    let grid = cas_rank_grid(space, part, PerSector(m_active));
    let d = space.d;

    let occ_up: Vec<bool> = (0..d).map(|i| i < space.nup).collect();
    let occ_dn: Vec<bool> = (0..d).map(|i| i < space.ndn).collect();
    let hf: TtVector<f64> = tt_state_in(*space, &occ_up, &occ_dn);
    debug_assert!(
        grid_le(space, &hf.ranks, &grid),
        "the HF determinant must live inside the pinned grid"
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x = perturbation(&hf, &grid, SEED_EPS, &mut rng);
    x.round_ranks_mut(&grid);
    normalize_mut(&mut x);

    let (_lam_als, x_als) = als(
        h,
        &x,
        &AlsOptions {
            eps: 1e-12,
            max_iter: 200,
            ..AlsOptions::default()
        },
    );
    let (mut x, _info) = newton_polish(
        h,
        &x_als,
        &NewtonPolishOpts {
            gradtol: 1e-11,
            ..NewtonPolishOpts::default()
        },
    );
    normalize_mut(&mut x);
    debug_assert!(x.check());
    debug_assert!(
        grid_le(space, &x.ranks, &grid),
        "cas_reference_state left the pinned grid from ABOVE — round_ranks_mut and the HOSVD \
         retraction are both '≤ target' operations, so only shrinkage is reachable"
    );

    let lambda = rayleigh_quotient(h, &x);
    (lambda, x)
}

#[cfg(test)]
mod tjcas_tests {
    use super::CasPartition;
    use crate::qn::QnSpace;

    #[test]
    #[should_panic(expected = "n_core")]
    fn tjcas_a0_partition_validation_rejects_overfull_core() {
        let space = QnSpace::new(6, 2, 2);
        let bad = CasPartition {
            n_core: 3,
            n_active: 2,
            n_virt: 1,
        };
        bad.validate(&space);
    }
}
