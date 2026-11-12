// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::coeff::CoeffTensor;
use crate::qn::{PerSector, Pooled};
use crate::round::RoundMode;
use crate::solvers::hess::{
    ChannelGsOpts, ChannelOpts, ChannelPlan, ChannelResult, ChannelSpectrum, ChannelStatus,
    START_PERTURBATION, channel_plan, merge_levels, start_rank,
};
use crate::solvers::mals::MalsOptions;
use crate::su2::core::Family;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use crate::su2::qn::{Su2RankGrid, Su2Space};
use crate::su2::riemannian::{RiemannianOpts, exact_gradnorm, normalize_mut, riemannian_cg};
use crate::su2::solvers::mals::mals;
use crate::su2::vector::{Su2TtVector, tt_csf_with_point_group, tt_randn};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::try_cone_response;

fn su2_derived_ramp(m: usize) -> Vec<usize> {
    vec![
        (m / 4).max(6),
        (m / 2).max(10),
        (3 * m / 4).max(14),
        m,
        m,
        m,
    ]
}

fn su2_csf_start(space: Su2Space) -> Option<Su2TtVector<f64>> {
    let d = space.d;
    let n_occ = (space.n - space.two_s) / 2;
    if n_occ + space.two_s > d {
        return None;
    }
    let base: Vec<Family> = (0..d)
        .map(|k| {
            if k < n_occ {
                Family::Occ
            } else if k < n_occ + space.two_s {
                Family::Up
            } else {
                Family::Empty
            }
        })
        .collect();
    let mut cands = vec![base.clone()];
    for i in n_occ..n_occ + space.two_s {
        for a in n_occ + space.two_s..d {
            let mut c = base.clone();
            c.swap(i, a);
            cands.push(c);
        }
    }
    cands.into_iter().find_map(|codes| {
        let x: Su2TtVector<f64> = tt_csf_with_point_group(&codes, space.pg);
        (x.space == space).then_some(x)
    })
}

fn su2_channel_start(space: Su2Space, gs: &ChannelGsOpts) -> Su2TtVector<f64> {
    let d = space.d;
    let caps = crate::su2::qn::max_ranks(&space);
    let interior = start_rank(PerSector(1));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(gs.seed);
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut g = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                g[s] = if b == 0 || b == d {
                    caps[b][s].min(1)
                } else {
                    caps[b][s].min(interior)
                };
            }
            g
        })
        .collect();
    let mut r: Su2TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    let Some(mut csf) = su2_csf_start(space) else {
        normalize_mut(&mut r);
        return r;
    };
    normalize_mut(&mut csf);
    normalize_mut(&mut r);
    r.scale_mut(START_PERTURBATION);
    let mut x = &csf + &r;
    x.round_ranks_mut(&ranks);
    normalize_mut(&mut x);
    x
}

fn su2_mals_start(space: Su2Space, gs: &ChannelGsOpts) -> Su2TtVector<f64> {
    let mut x0 = su2_channel_start(space, gs);
    x0.right_orthogonalize(true);
    x0
}

fn su2_channel_space(base: Su2Space, entry: &ChannelPlan) -> Option<Su2Space> {
    let d = base.d;
    let two_s = base.two_s + 2 * entry.spin_delta;
    if base.n > 2 * d || two_s > base.n.min(2 * d - base.n) {
        return None;
    }
    Some(Su2Space::with_point_group(
        d,
        base.n,
        two_s,
        base.pg,
        entry.irrep,
    ))
}

crate::solvers::hess::impl_channel!(
    ham: Su2SparseHamiltonian<f64>,
    vec: Su2TtVector<f64>,
    space: Su2Space,
    ropts: RiemannianOpts,
    channel_start: su2_mals_start,
    derived_ramp: su2_derived_ramp,
    channel_space: su2_channel_space,
    max_ranks: crate::su2::qn::max_ranks,
    mals: mals,
    riemannian_cg: riemannian_cg,
    exact_gradnorm: exact_gradnorm,
    normalize_mut: normalize_mut,
    rayleigh_quotient: rayleigh_quotient,
    merge_degenerate: false,
    channel_ground_state,
    channel_response,
);
