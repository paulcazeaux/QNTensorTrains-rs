// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::qn::{QnSpace, RankGrid, max_ranks};
use qntt::su2::qn::{Su2RankGrid, Su2SectorGrid, Su2Space, max_ranks as su2_max_ranks};
use qntt::{Irrep, PointGroup, PointGroupKind, Scalar};
use rand::RngExt;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn digits(idx: usize, d: usize) -> Vec<usize> {
    (0..d).map(|k| (idx >> (2 * k)) & 3).collect()
}

pub fn config_counts(dig: &[usize]) -> (usize, usize) {
    let nup = dig.iter().filter(|&&x| x == 1 || x == 3).count();
    let ndn = dig.iter().filter(|&&x| x == 2 || x == 3).count();
    (nup, ndn)
}

pub fn config_irrep_from_digits(pg: &PointGroup, dig: &[usize]) -> Irrep {
    let mut irrep = Irrep::IDENTITY;
    for (k, &x) in dig.iter().enumerate() {
        if x == 1 || x == 3 {
            irrep = irrep.fuse(pg.irrep_of(k));
        }
        if x == 2 || x == 3 {
            irrep = irrep.fuse(pg.irrep_of(k));
        }
    }
    irrep
}

pub fn irrep_census(
    pg: &PointGroup,
    d: usize,
) -> std::collections::HashMap<(usize, usize, Irrep), u64> {
    let mut counts = std::collections::HashMap::new();
    for idx in 0..(1usize << (2 * d)) {
        let dig = digits(idx, d);
        let (nup, ndn) = config_counts(&dig);
        let irrep = config_irrep_from_digits(pg, &dig);
        *counts.entry((nup, ndn, irrep)).or_insert(0u64) += 1;
    }
    counts
}

pub fn assert_irrep_split(pg: &PointGroup, d: usize, nup: usize, ndn: usize) {
    let census = irrep_census(pg, d);
    let distinct: std::collections::HashSet<Irrep> = census
        .iter()
        .filter(|&(&(u, v, _), &c)| u == nup && v == ndn && c > 0)
        .map(|(&(_, _, ir), _)| ir)
        .collect();
    assert!(
        distinct.len() >= 2,
        "vacuous fixture: (nup={nup}, ndn={ndn}) at d={d} spans only {} irrep(s)",
        distinct.len()
    );
}

pub fn assert_purity<T: Scalar>(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    target: Irrep,
    dense: &[T],
) {
    assert_eq!(dense.len(), 1usize << (2 * d));
    let mut any_nonzero_in_target = false;
    for (idx, v) in dense.iter().enumerate() {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            assert!(
                *v == T::zero(),
                "d={d}: config {idx} outside (Nup={nup},Ndn={ndn}) must be exactly zero, got {v:?}"
            );
            continue;
        }
        let irrep = config_irrep_from_digits(pg, &dig);
        if irrep == target {
            if *v != T::zero() {
                any_nonzero_in_target = true;
            }
        } else {
            assert!(
                *v == T::zero(),
                "campaign oracle: d={d} config {idx} has irrep {irrep:?} != target {target:?} \
                 but amplitude is {v:?} (not exactly zero)"
            );
        }
    }
    assert!(
        any_nonzero_in_target,
        "d={d}: oracle vacuous — no nonzero amplitude found inside the target irrep {target:?}"
    );
}

pub fn c2v_table(d: usize) -> PointGroup {
    let orbsym: Vec<u8> = (0..d).map(|k| ((k % 4) + 1) as u8).collect();
    PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table")
}

pub fn c2v_coeffs(
    pg: &PointGroup,
    d: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            if t.get(&[i, j]) != 0.0 {
                continue;
            }
            let val = if pg.irrep_of(i) == pg.irrep_of(j) {
                rng.sample::<f64, _>(StandardNormal)
            } else {
                0.0
            };
            t.set(&[i, j], val);
            t.set(&[j, i], val);
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if v.get(&[i, j, k, l]) != 0.0 {
                        continue;
                    }
                    let irrep_ok = pg
                        .irrep_of(i)
                        .fuse(pg.irrep_of(j))
                        .fuse(pg.irrep_of(k))
                        .fuse(pg.irrep_of(l))
                        == Irrep::IDENTITY;
                    let val = if irrep_ok {
                        rng.sample::<f64, _>(StandardNormal)
                    } else {
                        0.0
                    };
                    for &(a, b, c, dd) in &[
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    ] {
                        v.set(&[a, b, c, dd], val);
                    }
                }
            }
        }
    }
    (t, v)
}

pub fn capped_ranks(
    space: &QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<RankGrid> {
    let d = space.d;
    let caps = max_ranks(space);
    (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(space);
            for s in space.bond_sectors(b) {
                let draw = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
                r[s] = draw.min(caps[b][s]);
            }
            r
        })
        .collect()
}

pub fn su2_capped_ranks(space: &Su2Space, rmax: usize) -> Vec<Su2RankGrid> {
    let caps = su2_max_ranks(space);
    (0..=space.d)
        .map(|b| Su2SectorGrid::from_fn(space, |s| caps[b][s].min(rmax)))
        .collect()
}
