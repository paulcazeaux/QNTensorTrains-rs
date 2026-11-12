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

use crate::core::factor::svd_uvt_svals;
use crate::core::weingarten::FloorPolicy;
use crate::su2::coreops::mul_sketch_left;
use crate::su2::qn::{Su2Sector, Su2Space};
use crate::su2::riemannian::tangent_bases;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use crate::su2::weingarten::{bond_gauges, min_schmidt};
use crate::test_util::randomized_su2_state;

pub(crate) const BOND: usize = 3;

pub(crate) fn base_state() -> Su2TtVector<f64> {
    let space = Su2Space::new(6, 6, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5D2F_A7E3);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    x.scale_mut(1.0 / x.norm());
    x.sigma_floor_trim_mut(3e-2);
    x.scale_mut(1.0 / x.norm());
    x
}

pub(crate) fn w_hat_min(x: &Su2TtVector<f64>) -> f64 {
    let (bl, br) = tangent_bases(x);
    min_schmidt(&bl, &br)
}

pub(crate) fn planted_raw_sigma(x: &Su2TtVector<f64>, b: usize, two_s: usize) -> f64 {
    let (bl, br) = tangent_bases(x);
    bond_gauges(&bl, &br, FloorPolicy::default_hard())[b - 1]
        .sigma
        .iter()
        .filter(|(s, _)| s.two_s == two_s)
        .flat_map(|(_, v)| v.iter().copied())
        .fold(f64::INFINITY, f64::min)
}

pub(crate) fn room(bl: &Su2TtVector<f64>, b: usize, t: Su2Sector) -> usize {
    let u = bl.core(b - 1).vertical_unfolding(t);
    u.nrows().saturating_sub(u.ncols())
}

pub(crate) fn plant_sector(x: &Su2TtVector<f64>, b: usize, two_s: usize) -> Option<Su2Sector> {
    let (bl, br) = tangent_bases(x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    let g = &gauges[b - 1];
    g.r.sectors()
        .into_iter()
        .filter(|&s| {
            s.two_s == two_s && g.r.block(s).is_some_and(|m| m.nrows() >= 2) && room(&bl, b, s) > 0
        })
        .max_by_key(|&s| room(&bl, b, s))
}

fn resynth(x: &Su2TtVector<f64>, b: usize, t: Su2Sector, eps: Option<f64>) -> Su2TtVector<f64> {
    let (bl, br) = tangent_bases(x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    let g = &gauges[b - 1];
    let (w, mut sv, vt) = svd_uvt_svals(g.r.block(t).expect("fixture: the planted block"));
    if let Some(e) = eps {
        *sv.last_mut().expect("rank ≥ 2") = e;
    }
    let n = sv.len();
    let block = Mat::from_fn(n, n, |i, j| {
        sv.iter()
            .enumerate()
            .map(|(q, &s)| w.get(i, q) * s * vt.get(q, j))
            .sum::<f64>()
    });
    let mut r = g.r.clone();
    r.block_mut(t)
        .expect("fixture: the planted block")
        .copy_from(block.as_ref());
    let d = x.space.d;
    let mut cores = Vec::with_capacity(d);
    for k in 0..b {
        cores.push(bl.core(k).clone());
    }
    cores.push(mul_sketch_left(&r, br.core(b)));
    for k in b + 1..d {
        cores.push(br.core(k).clone());
    }
    let mut y = Su2TtVector::new(x.space, cores);
    let nrm = y.norm();
    y.scale_mut(1.0 / nrm);
    y
}

pub(crate) fn with_small_schmidt(
    x: &Su2TtVector<f64>,
    b: usize,
    two_s: usize,
    eps: f64,
) -> Su2TtVector<f64> {
    let t = plant_sector(x, b, two_s).unwrap_or_else(|| {
        panic!("fixture: no bond-{b} sector of spin 2s = {two_s} with rank ≥ 2 and room")
    });
    let same = resynth(x, b, t, None);
    let defect = (&same - x).norm_lr(false);
    assert!(
        defect <= 1e-12,
        "fixture PREMISE: re-synthesis with the original σ moved x by {defect:e}"
    );
    resynth(x, b, t, Some(eps))
}

pub(crate) fn dense_form(
    basis: &[Su2TtTangent<f64>],
    op: impl Fn(&Su2TtTangent<f64>) -> Su2TtTangent<f64>,
) -> Vec<f64> {
    let n = basis.len();
    let images: Vec<Su2TtTangent<f64>> = basis.iter().map(op).collect();
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = basis[i].dot(&images[j]);
        }
    }
    m
}

pub(crate) const CLUSTER_BONDS: [usize; 3] = [2, 3, 4];

pub(crate) struct Su2ClusterPlant {
    pub(crate) slots: Vec<(usize, Su2Sector, f64)>,
    pub(crate) passes: usize,
    pub(crate) deviation: f64,
}

pub(crate) fn with_schmidt_cluster(
    x: &Su2TtVector<f64>,
    bonds: &[usize],
    values: &[f64],
) -> (Su2TtVector<f64>, Su2ClusterPlant) {
    use crate::core::border_fixture::CLUSTER_PASSES;
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
        Su2ClusterPlant {
            slots,
            passes: CLUSTER_PASSES,
            deviation,
        },
    )
}

fn plant_sweep(
    x: &Su2TtVector<f64>,
    bonds: &[usize],
    values: &[f64],
    replant: bool,
) -> (Su2TtVector<f64>, Vec<(usize, Su2Sector, f64)>, f64) {
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
            r.block_mut(t)
                .expect("fixture: the planted block")
                .copy_from(block.as_ref());
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
        let mut z = Su2TtVector::new(y.space, cores);
        let nrm = z.norm();
        z.scale_mut(1.0 / nrm);
        y = z;
    }
    (y, slots, deviation)
}

pub(crate) struct Su2ClusterCensus {
    pub(crate) cut: Vec<(usize, Su2Sector, f64, usize)>,
    pub(crate) near: Vec<(usize, Su2Sector, f64)>,
    pub(crate) smallest_kept: f64,
    pub(crate) largest_cut: f64,
}

impl Su2ClusterCensus {
    pub(crate) fn cut_bonds(&self) -> Vec<usize> {
        let mut b: Vec<usize> = self.cut.iter().map(|c| c.0).collect();
        b.dedup();
        b
    }
    pub(crate) fn cut_blocks(&self) -> usize {
        let mut s: Vec<(usize, Su2Sector)> = self.cut.iter().map(|c| (c.0, c.1)).collect();
        s.dedup();
        s.len()
    }
    pub(crate) fn cut_with_room(&self) -> usize {
        self.cut.iter().filter(|c| c.3 > 0).count()
    }
}

pub(crate) fn cluster_census(x: &Su2TtVector<f64>, f: f64) -> Su2ClusterCensus {
    let (bl, br) = tangent_bases(x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    let (mut cut, mut near) = (Vec::new(), Vec::new());
    let (mut smallest_kept, mut largest_cut) = (f64::INFINITY, 0.0f64);
    for g in &gauges {
        for (t, sv) in &g.sigma {
            let rm = room(&bl, g.k, *t);
            for &s in sv {
                if s < f {
                    cut.push((g.k, *t, s, rm));
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
    Su2ClusterCensus {
        cut,
        near,
        smallest_kept,
        largest_cut,
    }
}

pub(crate) fn assert_cluster_premise(
    label: &str,
    x: &Su2TtVector<f64>,
    plant: &Su2ClusterPlant,
    f: f64,
    room: bool,
) -> Su2ClusterCensus {
    let c = cluster_census(x, f);
    println!(
        "{label}: plant {} slots in {} sweeps (deviation from target {:.1e}); at raw f = {f:e}: {} \
         cut in {} blocks at bonds {:?} ({} with room), {} kept in [f, 2f), largest cut {:.4e}, \
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

pub(crate) fn cluster_state(values: &[f64]) -> (Su2TtVector<f64>, Su2ClusterPlant) {
    with_schmidt_cluster(&base_state(), &CLUSTER_BONDS, values)
}

pub(crate) fn unit_tangents(
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    two_k: usize,
    n: usize,
    seed: u64,
) -> Vec<Su2TtTangent<f64>> {
    use crate::su2::flux::{flux_ranks, shifted_out_space};
    use crate::su2::qn::Su2RankGrid;
    use crate::su2::vector::tt_randn;
    let space = bl.space;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut xi = if two_k == 0 {
            let y: Su2TtVector<f64> = tt_randn(space, &bl.ranks, &mut rng);
            Su2TtTangent::project(bl, br, &y)
        } else {
            let outs = shifted_out_space(space, two_k);
            let grids: Vec<Su2RankGrid> = (0..=space.d)
                .map(|b| {
                    let row = flux_ranks(&bl.ranks[b], 0, space, outs, b);
                    let col = flux_ranks(&bl.ranks[b], two_k as i64, space, outs, b);
                    let mut g = Su2RankGrid::zeros(&outs);
                    for s in outs.bond_sectors(b) {
                        g[s] = match b {
                            0 => usize::from(row[s] > 0),
                            _ if b == space.d => usize::from(col[s] > 0),
                            _ => row[s] + col[s],
                        };
                    }
                    g
                })
                .collect();
            let y: Su2TtVector<f64> = tt_randn(outs, &grids, &mut rng);
            Su2TtTangent::project_shifted(bl, br, two_k, &y)
        };
        let nrm = xi.norm();
        if nrm > 1e-8 {
            xi.scale_mut(1.0 / nrm);
            out.push(xi);
        }
    }
    out
}
