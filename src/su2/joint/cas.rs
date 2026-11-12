// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! **CAS(n,m) as a sector-pinned SU(2) tensor train** — the spin-adapted twin of
//! [`crate::solvers::joint::cas`], and the substrate the SU(2) joint (TT ⊕ orbital) stack needs to
//! run on a complete-active-space manifold.

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::PerSector;
use crate::solvers::hess::NewtonPolishOpts;
use crate::solvers::joint::cas::CasPartition;
use crate::su2::core::Family;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use crate::su2::hess::newton_polish;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space, max_ranks};
use crate::su2::riemannian::normalize_mut;
use crate::su2::solvers::als::{AlsOptions, als};
use crate::su2::vector::{Su2TtVector, tt_csf, tt_csf_with_point_group, tt_randn};

const SEED_EPS: f64 = 0.2;

fn active_space(space: &Su2Space, part: &CasPartition) -> Su2Space {
    let (n, two_s) = (space.n - 2 * part.n_core, space.two_s);
    if space.pg.order() <= 1 {
        return Su2Space::new(part.n_active, n, two_s);
    }
    let irreps: Vec<Irrep> = (part.n_core..part.n_core + part.n_active)
        .map(|k| space.pg.irrep_of(k))
        .collect();
    let pg_act = PointGroup::new(space.pg.nbits(), &irreps);
    Su2Space::with_point_group(part.n_active, n, two_s, pg_act, space.target)
}

fn validate(part: &CasPartition, space: &Su2Space) {
    assert_eq!(
        part.d(),
        space.d,
        "CasPartition n_core={} + n_active={} + n_virt={} = {} must cover the chain (d = {})",
        part.n_core,
        part.n_active,
        part.n_virt,
        part.d(),
        space.d
    );
    assert!(
        2 * part.n_core <= space.n,
        "CasPartition n_core = {} needs {} electrons (a doubly-occupied core prefix spends TWO per \
         orbital) but the space carries only N = {}, so the (2b, 0) prefix pin has no \
         representation",
        part.n_core,
        2 * part.n_core,
        space.n
    );
    let n_act = space.n - 2 * part.n_core;
    assert!(
        n_act <= 2 * part.n_active,
        "CasPartition n_active = {} cannot hold the {n_act} active electrons left by n_core = {} \
         (the virtual suffix is empty by construction)",
        part.n_active,
        part.n_core
    );
    assert!(
        space.two_s <= n_act && space.two_s + n_act <= 2 * part.n_active,
        "CasPartition: 2S = {} is not carryable by the active window alone — with n_active = {} \
         and {n_act} active electrons the reduced space admits only 2S ≤ min({n_act}, {}). The \
         core prefix is singlet-coupled, so it contributes no spin",
        space.two_s,
        part.n_active,
        2 * part.n_active - n_act
    );
    assert!(
        part.n_active >= 1,
        "CasPartition n_active = 0 leaves no free bond: the pinned state would be a single CSF, \
         which is a different object (escalate if that is wanted)"
    );
    debug_assert!(
        (n_act + space.two_s).is_multiple_of(2),
        "reduced parity broken: (N − 2·n_core) − 2S must be even"
    );
    if space.pg.order() > 1 {
        let act = active_space(space, part);
        let caps = max_ranks(&act);
        let last = Su2Sector::with_irrep(act.n, act.two_s, act.target);
        assert!(
            caps[part.n_active][last] > 0,
            "CasPartition: target irrep {:?} is unreachable from the active window (n_core = {}, \
             n_active = {}) — the core prefix and virtual suffix both flux to IDENTITY, so the \
             whole target must be carried by the active orbitals",
            space.target,
            part.n_core,
            part.n_active
        );
    }
}

/// The **pinned reduced-rank grid** of §1: `space.d + 1` per-bond grids (bond `b` at index `b`, the
/// convention [`Su2TtVector::ranks`], [`Su2TtVector::round_ranks_mut`] and [`max_ranks`] share).
pub fn su2_cas_rank_grid(
    space: &Su2Space,
    part: &CasPartition,
    m_active: PerSector,
) -> Vec<Su2RankGrid> {
    validate(part, space);
    let (d, ni, na) = (space.d, part.n_core, part.n_active);
    let mut out: Vec<Su2RankGrid> = (0..=d).map(|_| Su2RankGrid::zeros(space)).collect();

    for (b, grid) in out.iter_mut().enumerate().take(ni + 1) {
        grid[Su2Sector::new(2 * b, 0)] = 1;
    }
    let last = Su2Sector::with_irrep(space.n, space.two_s, space.target);
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
            let s = Su2Sector::with_irrep(sa.n + 2 * ni, sa.two_s, sa.irrep);
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

fn cas_csf_codes(space: &Su2Space, part: &CasPartition) -> Vec<Family> {
    let n_act = space.n - 2 * part.n_core;
    let n_single = space.two_s;
    let n_pair = (n_act - n_single) / 2;
    debug_assert!(
        n_single + n_pair <= part.n_active,
        "the seed occupancy spills past the active window — validate should have refused"
    );
    (0..space.d)
        .map(|k| {
            if k < part.n_core {
                Family::Occ
            } else if k < part.n_core + n_single {
                Family::Up
            } else if k < part.n_core + n_single + n_pair {
                Family::Occ
            } else {
                Family::Empty
            }
        })
        .collect()
}

fn grid_le(space: &Su2Space, got: &[Su2RankGrid], want: &[Su2RankGrid]) -> bool {
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
///
/// # Panics
///
/// Through [`validate`] on an inadmissible partition, and through [`tt_csf`] on a genealogical path
/// the space cannot carry.
pub fn su2_cas_reference_state(
    h: &Su2SparseHamiltonian<f64>,
    space: &Su2Space,
    part: &CasPartition,
    m_active: usize,
    seed: u64,
) -> (f64, Su2TtVector<f64>) {
    let grid = su2_cas_rank_grid(space, part, PerSector(m_active));
    let codes = cas_csf_codes(space, part);
    let mut csf: Su2TtVector<f64> = if space.pg.order() <= 1 {
        tt_csf(&codes)
    } else {
        tt_csf_with_point_group(&codes, space.pg)
    };
    debug_assert_eq!(
        csf.space, *space,
        "the seed CSF's inferred space disagrees with the caller's"
    );
    debug_assert!(
        grid_le(space, &csf.ranks, &grid),
        "the seed CSF must live inside the pinned grid"
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut r: Su2TtVector<f64> = tt_randn(*space, &grid, &mut rng);
    normalize_mut(&mut csf);
    normalize_mut(&mut r);
    r.scale_mut(SEED_EPS);
    let mut x = &csf + &r;
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
        "su2_cas_reference_state left the pinned grid from ABOVE — round_ranks_mut and the HOSVD \
         retraction are both '≤ target' operations, so only shrinkage is reachable"
    );

    let lambda = rayleigh_quotient(h, &x);
    (lambda, x)
}

#[cfg(test)]
mod a0_tests {
    use super::su2_cas_rank_grid;
    use crate::qn::PerSector;
    use crate::solvers::joint::cas::CasPartition;
    use crate::su2::qn::Su2Space;

    #[test]
    #[should_panic(expected = "n_core = 3 needs 6 electrons")]
    fn tsc_a0_overfull_core_panics_naming_n_core() {
        let space = Su2Space::new(6, 4, 0);
        let part = CasPartition {
            n_core: 3,
            n_active: 2,
            n_virt: 1,
        };
        let _ = su2_cas_rank_grid(&space, &part, PerSector::NONE);
    }

    #[test]
    #[should_panic(expected = "cannot hold the 4 active electrons")]
    fn tsc_a0_active_window_too_small_panics_naming_n_active() {
        let space = Su2Space::new(6, 4, 0);
        let part = CasPartition {
            n_core: 0,
            n_active: 1,
            n_virt: 5,
        };
        let _ = su2_cas_rank_grid(&space, &part, PerSector::NONE);
    }

    #[test]
    #[should_panic(expected = "2S = 4 is not carryable by the active window")]
    fn tsc_a0_spin_uncarryable_panics_naming_2s() {
        let space = Su2Space::new(6, 4, 4);
        let part = CasPartition {
            n_core: 1,
            n_active: 2,
            n_virt: 3,
        };
        let _ = su2_cas_rank_grid(&space, &part, PerSector::NONE);
    }

    #[test]
    #[should_panic(expected = "n_active = 0 leaves no free bond")]
    fn tsc_a0_zero_active_panics_naming_the_free_bond() {
        let space = Su2Space::new(6, 4, 0);
        let part = CasPartition {
            n_core: 2,
            n_active: 0,
            n_virt: 4,
        };
        let _ = su2_cas_rank_grid(&space, &part, PerSector::NONE);
    }
}
