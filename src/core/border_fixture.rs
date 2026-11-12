// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::Mat;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::core::factor::{mul_sketch_left, svd_uvt_svals};
use crate::core::weingarten::{FloorPolicy, bond_gauges};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{Pooled, QnSpace, Sector};
use crate::round::RoundMode;
use crate::solvers::riemannian::tangent_bases;
use crate::test_util::randomized_state;
use crate::vector::TtVector;

pub(crate) const BOND: usize = 2;

pub(crate) fn base_state() -> TtVector<f64> {
    let space = QnSpace::new(6, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEF1_A7E0);
    let mut x: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    x.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    x.sigma_floor_trim_mut(3e-2);
    x.scale_mut(1.0 / x.norm());
    x
}

pub(crate) fn sigma_min(x: &TtVector<f64>) -> f64 {
    let (bl, br) = tangent_bases(x);
    bond_gauges(&bl, &br, FloorPolicy::default_hard())
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, sv)| sv.iter().copied()))
        .fold(f64::INFINITY, f64::min)
}

pub(crate) fn with_small_schmidt(x: &TtVector<f64>, b: usize, eps: f64) -> TtVector<f64> {
    let (bl, br) = tangent_bases(x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    let g = &gauges[b - 1];
    let room = |s| {
        let u = bl.core(b - 1).vertical_unfolding(s);
        u.nrows().saturating_sub(u.ncols())
    };
    let t =
        g.r.sectors()
            .into_iter()
            .filter(|&s| g.r.block(s).is_some_and(|m| m.nrows() >= 2) && room(s) > 0)
            .max_by_key(|&s| room(s))
            .expect("fixture: a bond sector of rank ≥ 2 with gauge-complement room");
    let (w, mut sv, vt) = svd_uvt_svals(g.r.block(t).expect("block"));
    *sv.last_mut().expect("rank ≥ 2") = eps;
    let n = sv.len();
    let block = Mat::from_fn(n, n, |i, j| {
        sv.iter()
            .enumerate()
            .map(|(q, &s)| w.get(i, q) * s * vt.get(q, j))
            .sum::<f64>()
    });
    let mut r = g.r.clone();
    r.block_mut(t).expect("block").copy_from(block.as_ref());
    let d = x.space.d;
    let mut cores = Vec::with_capacity(d);
    for k in 0..b {
        cores.push(bl.core(k).clone());
    }
    cores.push(mul_sketch_left(&r, br.core(b)));
    for k in b + 1..d {
        cores.push(br.core(k).clone());
    }
    let mut y = TtVector::new(x.space, cores);
    let nrm = y.norm();
    y.scale_mut(1.0 / nrm);
    y
}

pub(crate) const CLUSTER: [f64; 5] = [0.3, 0.6, 0.9, 1.1, 1.5];

pub(crate) const CLUSTER_BONDS: [usize; 3] = [2, 3, 4];

pub(crate) const CLUSTER_FLOOR: f64 = 1e-3;

pub(crate) struct ClusterPlant {
    pub(crate) slots: Vec<(usize, Sector, f64)>,
    pub(crate) passes: usize,
    pub(crate) deviation: f64,
}

pub(crate) const CLUSTER_PASSES: usize = 8;

pub(crate) fn with_schmidt_cluster(
    x: &TtVector<f64>,
    bonds: &[usize],
    values: &[f64],
) -> (TtVector<f64>, ClusterPlant) {
    assert!(!values.is_empty(), "with_schmidt_cluster: no values");
    let mut y = x.clone();
    for _ in 0..CLUSTER_PASSES {
        y = plant_sweep(&y, bonds, values, true).0;
    }
    let (_, slots, deviation) = plant_sweep(&y, bonds, values, false);
    assert!(
        !slots.is_empty(),
        "with_schmidt_cluster: no rank-≥ 2 sector at bonds {bonds:?}"
    );
    (
        y,
        ClusterPlant {
            slots,
            passes: CLUSTER_PASSES,
            deviation,
        },
    )
}

fn plant_sweep(
    x: &TtVector<f64>,
    bonds: &[usize],
    values: &[f64],
    replant: bool,
) -> (TtVector<f64>, Vec<(usize, Sector, f64)>, f64) {
    let mut y = x.clone();
    let mut slots = Vec::new();
    let mut cursor = 0usize;
    let mut deviation = 0.0f64;
    for &b in bonds {
        let (bl, br) = tangent_bases(&y);
        let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        let g = &gauges[b - 1];
        let mut r = g.r.clone();
        for t in g.r.sectors() {
            let Some(blk) = g.r.block(t) else { continue };
            let n = blk.nrows();
            if n < 2 {
                continue;
            }
            let (w, mut sv, vt) = svd_uvt_svals(blk);
            let mut targets: Vec<f64> = (1..n)
                .map(|_| {
                    let v = values[cursor % values.len()];
                    cursor += 1;
                    v
                })
                .collect();
            targets.sort_by(|p, q| q.total_cmp(p));
            assert!(
                targets[0] < sv[0],
                "with_schmidt_cluster: bond {b} sector {t:?} keeps σ = {:e}, not above the planted \
                 {:e}",
                sv[0],
                targets[0]
            );
            for (i, &v) in targets.iter().enumerate() {
                deviation = deviation.max((sv[i + 1] - v).abs() / v);
                sv[i + 1] = v;
                slots.push((b, t, v));
            }
            let block = Mat::from_fn(n, n, |i, j| {
                sv.iter()
                    .enumerate()
                    .map(|(q, &s)| w.get(i, q) * s * vt.get(q, j))
                    .sum::<f64>()
            });
            r.block_mut(t).expect("block").copy_from(block.as_ref());
        }
        if !replant {
            continue;
        }
        let d = y.space.d;
        let mut cores = Vec::with_capacity(d);
        for k in 0..b {
            cores.push(bl.core(k).clone());
        }
        cores.push(mul_sketch_left(&r, br.core(b)));
        for k in b + 1..d {
            cores.push(br.core(k).clone());
        }
        let mut z = TtVector::new(y.space, cores);
        let nrm = z.norm();
        z.scale_mut(1.0 / nrm);
        y = z;
    }
    (y, slots, deviation)
}

pub(crate) struct ClusterCensus {
    pub(crate) cut: Vec<(usize, Sector, f64, usize)>,
    pub(crate) near: Vec<(usize, Sector, f64)>,
    pub(crate) smallest_kept: f64,
    pub(crate) largest_cut: f64,
}

impl ClusterCensus {
    pub(crate) fn cut_bonds(&self) -> Vec<usize> {
        let mut b: Vec<usize> = self.cut.iter().map(|c| c.0).collect();
        b.dedup();
        b
    }
    pub(crate) fn cut_blocks(&self) -> usize {
        let mut s: Vec<(usize, Sector)> = self.cut.iter().map(|c| (c.0, c.1)).collect();
        s.dedup();
        s.len()
    }
    pub(crate) fn cut_with_room(&self) -> usize {
        self.cut.iter().filter(|c| c.3 > 0).count()
    }
}

pub(crate) fn assert_cluster_premise(
    label: &str,
    x: &TtVector<f64>,
    plant: &ClusterPlant,
    f: f64,
    room: bool,
) -> ClusterCensus {
    let c = cluster_census(x, f);
    println!(
        "{label}: plant {} slots in {} sweeps (deviation from target {:.1e}); at f = {f:e}: {} cut \
         in {} blocks at bonds {:?} ({} with room), {} kept in [f, 2f), largest cut {:.4e}, \
         smallest kept {:.4e}, smallest cut {:.4e}",
        plant.slots.len(),
        plant.passes,
        plant.deviation,
        c.cut.len(),
        c.cut_blocks(),
        c.cut_bonds(),
        c.cut_with_room(),
        c.near.len(),
        c.largest_cut,
        c.smallest_kept,
        c.cut.iter().map(|q| q.2).fold(f64::INFINITY, f64::min)
    );
    assert!(
        c.cut_bonds().len() >= 2 && c.cut_blocks() >= 3,
        "{label}: premise — the floor must cut several values in several sectors and bonds"
    );
    assert!(
        !room || c.cut_with_room() >= 3,
        "{label}: premise — cut blocks with gauge-complement room"
    );
    assert!(
        c.near.len() >= 2,
        "{label}: premise — kept values just above the floor"
    );
    assert!(
        c.largest_cut <= 0.95 * f && c.smallest_kept >= 1.05 * f,
        "{label}: premise — no value within 5 % of the floor"
    );
    c
}

pub(crate) fn cluster_census(x: &TtVector<f64>, f: f64) -> ClusterCensus {
    let (bl, br) = tangent_bases(x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    let (mut cut, mut near) = (Vec::new(), Vec::new());
    let (mut smallest_kept, mut largest_cut) = (f64::INFINITY, 0.0f64);
    for g in &gauges {
        for (t, sv) in g.svals() {
            let u = bl.core(g.k - 1).vertical_unfolding(*t);
            let room = u.nrows().saturating_sub(u.ncols());
            for &s in sv {
                if s < f {
                    cut.push((g.k, *t, s, room));
                    largest_cut = largest_cut.max(s);
                } else {
                    smallest_kept = smallest_kept.min(s);
                    if s < 2.0 * f {
                        near.push((g.k, *t, s));
                    }
                }
            }
        }
    }
    ClusterCensus {
        cut,
        near,
        smallest_kept,
        largest_cut,
    }
}

pub(crate) fn cluster_base_state() -> TtVector<f64> {
    let space = QnSpace::new(6, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEF1_A7E0);
    let mut x: TtVector<f64> = randomized_state(space, 3, 4, &mut rng);
    x.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    x.sigma_floor_trim_mut(3e-2);
    x.scale_mut(1.0 / x.norm());
    x
}

pub(crate) fn cluster_state() -> (TtVector<f64>, ClusterPlant) {
    let values: Vec<f64> = CLUSTER.iter().map(|m| m * CLUSTER_FLOOR).collect();
    with_schmidt_cluster(&cluster_base_state(), &CLUSTER_BONDS, &values)
}

pub(crate) fn pg6_table() -> PointGroup {
    PointGroup::new(
        2,
        &[
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
        ],
    )
}

pub(crate) fn pg_base_state() -> TtVector<f64> {
    let space = QnSpace::with_point_group(6, 3, 3, pg6_table(), Irrep::IDENTITY);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC1A5_7E06);
    let mut x: TtVector<f64> = randomized_state(space, 3, 4, &mut rng);
    x.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    x.sigma_floor_trim_mut(3e-2);
    x.scale_mut(1.0 / x.norm());
    x
}

pub(crate) fn pg_cluster_state() -> (TtVector<f64>, ClusterPlant) {
    let values: Vec<f64> = CLUSTER.iter().map(|m| m * CLUSTER_FLOOR).collect();
    with_schmidt_cluster(&pg_base_state(), &CLUSTER_BONDS, &values)
}

pub(crate) fn unit_tangents(
    bl: &TtVector<f64>,
    br: &TtVector<f64>,
    phi: Irrep,
    n: usize,
    seed: u64,
) -> Vec<crate::core::tangent::TtTangent<f64>> {
    use crate::core::tangent::TtTangent;
    let space = bl.space;
    let space_out = space.with_target(space.target.fuse(phi));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let mut xi = if phi == Irrep::IDENTITY {
            TtTangent::project(bl, br, &y)
        } else {
            TtTangent::project_shifted(bl, br, phi, &y)
        };
        let nrm = xi.norm();
        if nrm > 1e-8 {
            xi.scale_mut(1.0 / nrm);
            out.push(xi);
        }
    }
    out
}
