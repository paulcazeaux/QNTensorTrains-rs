// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::sketch::Sketch;
use crate::qn::{Pooled, RankGrid, Sector};
use crate::scalar::Scalar;

fn per_sector_svals<T: Scalar>(sv: &Sketch<T>) -> Vec<(Sector, Vec<f64>)> {
    let mut out = Vec::new();
    for s in sv.sectors() {
        if let Some(b) = sv.block(s) {
            let len = b.nrows().min(b.ncols());
            let vals: Vec<f64> = (0..len).map(|i| b.get(i, i).re_s()).collect();
            out.push((s, vals));
        }
    }
    out
}

fn dropped_count(sorted_asc: &[f64], eps: f64) -> usize {
    let thresh = eps * eps;
    let mut cum = 0.0f64;
    let mut dropped = 0usize;
    for &s in sorted_asc {
        cum += s * s;
        if cum <= thresh {
            dropped += 1;
        } else {
            break;
        }
    }
    dropped
}

/// Keep-one fallback: the sector with the largest leading singular value, scanning canonical order;
/// `None` if no sector carries any value. Never selects a zero-dimensional sector (those have an
/// empty singular-value list).
pub fn keep_one_sector<T: Scalar>(sv: &Sketch<T>) -> Option<Sector> {
    let mut best: Option<(Sector, f64)> = None;
    for (s, vals) in per_sector_svals(sv) {
        if let Some(&lead) = vals.first() {
            match best {
                Some((_, b)) if b >= lead => {}
                _ => best = Some((s, lead)),
            }
        }
    }
    best.map(|(s, _)| s)
}

fn apply_keep_one<T: Scalar>(sv: &Sketch<T>, ranks: &mut RankGrid) {
    for s in sv.sectors() {
        ranks[s] = 0;
    }
    if let Some(n0) = keep_one_sector(sv) {
        ranks[n0] = 1;
    }
}

fn count_above<T: Scalar>(sv: &Sketch<T>, cutoff: f64) -> RankGrid {
    let mut ranks = RankGrid::zeros(&sv.space);
    for (s, vals) in per_sector_svals(sv) {
        ranks[s] = vals.iter().filter(|&&x| x > cutoff).count();
    }
    ranks
}

/// ε + POOLED-rank energy chop used by the rounding drivers: pool all singular values of the bond,
/// keep the fewest largest whose dropped Frobenius energy `≤ ϵ²`, capped at `chi` total kept
/// values across all sectors ([`Pooled::NONE`] = uncapped). Degeneracy cutoff is
/// `prevfloat(all_svals[r])`; keep-one fallback when everything would be chopped.
pub fn chop_energy<T: Scalar>(sv: &Sketch<T>, eps: f64, chi: Pooled) -> RankGrid {
    if eps > 0.0 {
        let mut all: Vec<f64> = per_sector_svals(sv)
            .into_iter()
            .flat_map(|(_, v)| v)
            .collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = all.len();
        let dropped = dropped_count(&all, eps);
        let idx = if chi.is_none() {
            dropped
        } else {
            dropped.max(n.saturating_sub(chi.0))
        };
        if idx + 1 < n {
            let cutoff = all[idx].next_down();
            count_above(sv, cutoff)
        } else {
            let mut ranks = RankGrid::zeros(&sv.space);
            apply_keep_one(sv, &mut ranks);
            ranks
        }
    } else if chi.is_none() {
        sv.row_ranks()
    } else {
        chop_rank(sv, chi)
    }
}

/// Fixed POOLED-rank chop: tag values with their sector, sort descending, take `min(chi, total)`
/// across the pool, count per sector.
pub fn chop_rank<T: Scalar>(sv: &Sketch<T>, chi: Pooled) -> RankGrid {
    let mut tagged: Vec<(f64, Sector)> = Vec::new();
    for (s, vals) in per_sector_svals(sv) {
        for v in vals {
            tagged.push((v, s));
        }
    }
    tagged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let take = chi.0.min(tagged.len());
    let mut ranks = RankGrid::zeros(&sv.space);
    for &(_, s) in tagged.iter().take(take) {
        ranks[s] += 1;
    }
    ranks
}

/// Global vector chop: pool singular values across ALL bonds and sectors, one global energy
/// threshold, distribute per bond with keep-one.
pub fn chop_global<T: Scalar>(svs: &[Sketch<T>], eps: f64) -> Vec<RankGrid> {
    let mut ranks: Vec<RankGrid> = svs.iter().map(|s| RankGrid::zeros(&s.space)).collect();
    if eps <= 0.0 {
        for (k, sv) in svs.iter().enumerate() {
            ranks[k] = sv.row_ranks();
        }
        return ranks;
    }
    let mut all: Vec<f64> = Vec::new();
    for sv in svs {
        for (_, v) in per_sector_svals(sv) {
            all.extend(v);
        }
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = all.len();
    let dropped = dropped_count(&all, eps);
    if dropped < n {
        let cutoff = all[dropped].next_down();
        for (k, sv) in svs.iter().enumerate() {
            ranks[k] = count_above(sv, cutoff);
            if ranks[k].total() == 0 {
                apply_keep_one(sv, &mut ranks[k]);
            }
        }
    } else {
        for (k, sv) in svs.iter().enumerate() {
            apply_keep_one(sv, &mut ranks[k]);
        }
    }
    ranks
}

#[cfg(test)]
mod chop_scale_invariance_tests {
    use super::{Sketch, chop_global};
    use crate::qn::{QnSpace, RankGrid, Sector};
    use faer::prelude::ReborrowMut;

    fn diag_sketch(vals: &[f64]) -> Sketch<f64> {
        let space = QnSpace::new(1, 1, 0);
        let s = Sector::new(0, 0);
        let mut ranks = RankGrid::zeros(&space);
        ranks[s] = vals.len();
        let mut sk = Sketch::zeros(space, 0, &ranks, &ranks);
        let mut b = sk.block_mut(s).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            *b.rb_mut().get_mut(i, i) = v;
        }
        sk
    }

    #[test]
    fn chop_global_keeps_boundary_value_above_magnitude_two() {
        let sk = diag_sketch(&[3.0, 6.0, 10.0]);
        let ranks = chop_global(std::slice::from_ref(&sk), 3.5);
        assert_eq!(
            ranks[0].total(),
            2,
            "boundary singular value 6.0 must be kept, not chopped by an absolute-epsilon tie"
        );
    }
}

#[cfg(test)]
mod phase3_tests {
    use crate::qn::QnSpace;
    use faer::Mat;

    #[test]
    fn chop_energy_keep_one_on_tied_max() {
        use crate::SectorGrid;
        use crate::core::factor::{Sketch, chop_energy};

        let space = QnSpace::from_n_two_sz(4, 2, 0);
        let k = 2;
        let secs = space.bond_sectors(k);
        assert!(secs.len() >= 2, "need ≥2 admissible sectors at bond {k}");
        let (s0, s1) = (secs[0], secs[1]);
        let blocks: SectorGrid<Option<Mat<f64>>> = SectorGrid::from_fn(&space, |s| {
            if s == s0 || s == s1 {
                let mut m = Mat::<f64>::zeros(1, 1);
                *m.get_mut(0, 0) = 1.0;
                Some(m)
            } else {
                None
            }
        });
        let sv = Sketch::new(space, k, blocks);
        let ranks = chop_energy(&sv, 1e-6, crate::Pooled(1));
        assert_eq!(
            ranks.total(),
            1,
            "keep exactly one on a tied max with maxRank=1"
        );
        assert_eq!(
            ranks[s0], 1,
            "the survivor sits in the first (canonical) argmax sector"
        );
        assert_eq!(ranks[s1], 0, "the tied duplicate is dropped");
    }
}
