// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Degeneracy-weighted `chop` truncation-rule family. Mirrors the abelian
//! `crate::core::factor::chop_*` family (`core/factor/`) with ONE SU(2) addition: ranks are
//! counted on WEIGHTED singular values `√centerweight(2s,2S)·σ`, while the SVD's stored
//! `U`/`S`/`Vt` factors that the chop truncates stay RAW (unscaled). The ε budget is then a true
//! fixed-Sz error norm, because `Σ_dropped centerweight·σ² = Σ_dropped (√cw·σ)²` is exactly the
//! fixed-Sz energy of what is discarded.

use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{SymGrid, SymLabel, SymSpace, Symmetry};

/// The sector with the largest leading (weighted) singular value, among sectors that have any
/// values; `None` if no sector qualifies. Input is
/// conceptually the WEIGHTED svals (`√centerweight·σ`, not raw σ) — mirror of the abelian
/// `keep_one_sector` (`core/factor/chop.rs`) but scanning the weighted leading
/// value. Tie-break: first-seen strict maximum in canonical (`bond_sectors`) order, matching
/// `argmax`'s first-maximizer semantics.
pub fn keep_one_sector<T: Scalar, S: Symmetry>(sv: &Su2Sketch<T, S>) -> Option<S::Label> {
    let mut best: Option<(S::Label, f64)> = None;
    for (s, vals) in weighted_svals(sv) {
        if let Some(&lead) = vals.first() {
            match best {
                Some((_, b)) if b >= lead => {}
                _ => best = Some((s, lead)),
            }
        }
    }
    best.map(|(s, _)| s)
}

/// ε + POOLED-rank energy chop: pools ALL sectors' WEIGHTED singular values, keeps the fewest
/// largest whose dropped fixed-Sz energy `≤ ε²`, capped at `chi` total kept REDUCED MULTIPLETS
/// across all sectors ([`Pooled::NONE`] = uncapped). Degeneracy cutoff
/// is `f64::next_down(all_svals[r])` (NOT `-eps`). Used by `round`/`round_lr`.
/// Mirror of the abelian `chop_energy` with [`weighted_svals`]-style weighting substituted for
/// the pool.
pub fn chop_energy<T: Scalar, S: Symmetry>(
    sv: &Su2Sketch<T, S>,
    eps: f64,
    chi: Pooled,
) -> S::Grid<usize> {
    if eps > 0.0 {
        let mut all: Vec<f64> = weighted_svals(sv)
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
            let mut ranks = <S::Grid<usize>>::filled(&sv.space, 0);
            apply_keep_one(sv, &mut ranks);
            ranks
        }
    } else if chi.is_none() {
        sv.row_ranks()
    } else {
        chop_rank(sv, chi)
    }
}

/// Fixed POOLED-rank chop: tag each WEIGHTED value with its sector, stable-sort descending, take
/// `min(chi, total)` across the pool of REDUCED MULTIPLETS, count per sector. Mirror of
/// `chop_rank` (`core/factor/`).
pub fn chop_rank<T: Scalar, S: Symmetry>(sv: &Su2Sketch<T, S>, chi: Pooled) -> S::Grid<usize> {
    let mut tagged: Vec<(f64, S::Label)> = Vec::new();
    for (s, vals) in weighted_svals(sv) {
        for v in vals {
            tagged.push((v, s));
        }
    }
    tagged.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let take = chi.0.min(tagged.len());
    let mut ranks = <S::Grid<usize>>::filled(&sv.space, 0);
    for &(_, s) in tagged.iter().take(take) {
        ranks[s] += 1;
    }
    ranks
}

/// Global pooled chop across ALL bonds and sectors: one global WEIGHTED energy threshold,
/// distributed per bond via `count(>cutoff)`, with a per-bond keep-one fallback whenever a bond's
/// ranks would come out all-zero. Used by `round_global_mut`. Mirror of the abelian
/// `chop_global` (`core/factor/`). Returned `Vec` index `i` corresponds to bond `i+1`
/// (matching the abelian convention).
pub fn chop_global<T: Scalar, S: Symmetry>(
    svs: &[Su2Sketch<T, S>],
    eps: f64,
) -> Vec<S::Grid<usize>> {
    let mut ranks: Vec<S::Grid<usize>> = svs
        .iter()
        .map(|s| <S::Grid<usize>>::filled(&s.space, 0))
        .collect();
    if eps <= 0.0 {
        for (k, sv) in svs.iter().enumerate() {
            ranks[k] = sv.row_ranks();
        }
        return ranks;
    }
    let mut all: Vec<f64> = Vec::new();
    for sv in svs {
        for (_, v) in weighted_svals(sv) {
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
            if ranks[k].values().iter().sum::<usize>() == 0 {
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

pub(crate) fn pool_weight<Sp: SymSpace>(space: &Sp, s: Sp::Label) -> f64 {
    crate::su2::centerweight(s.two_j(), space.total_two_j()).sqrt()
}

fn weighted_svals<T: Scalar, S: Symmetry>(sv: &Su2Sketch<T, S>) -> Vec<(S::Label, Vec<f64>)> {
    let mut out = Vec::new();
    for s in sv.sectors() {
        if let Some(b) = sv.block(s) {
            let w = pool_weight(&sv.space, s);
            let len = b.nrows().min(b.ncols());
            let vals: Vec<f64> = (0..len).map(|i| w * b.get(i, i).re_s()).collect();
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

fn count_above<T: Scalar, S: Symmetry>(sv: &Su2Sketch<T, S>, cutoff: f64) -> S::Grid<usize> {
    let mut ranks = <S::Grid<usize>>::filled(&sv.space, 0);
    for (s, vals) in weighted_svals(sv) {
        ranks[s] = vals.iter().filter(|&&x| x > cutoff).count();
    }
    ranks
}

fn apply_keep_one<T: Scalar, S: Symmetry>(sv: &Su2Sketch<T, S>, ranks: &mut S::Grid<usize>) {
    for s in sv.sectors() {
        ranks[s] = 0;
    }
    if let Some(n0) = keep_one_sector(sv) {
        ranks[n0] = 1;
    }
}

#[cfg(test)]
mod s2_tests {
    use super::{chop_energy, chop_global};
    use crate::qn::Pooled;
    use crate::su2::centerweight;
    use crate::su2::frame::Su2Sketch;
    use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
    use faer::prelude::ReborrowMut;

    #[test]
    fn tst1_chop_weighting_fixture_two_sector_exact_grid() {
        let space = Su2Space::new(6, 5, 1);
        let k = 2usize;
        let sa = Su2Sector::new(2, 0);
        let sb = Su2Sector::new(2, 2);
        assert!(space.contains(k, sa) && space.contains(k, sb));
        assert!((centerweight(sa.two_s, space.two_s) - 0.5).abs() < 1e-15);
        assert!((centerweight(sb.two_s, space.two_s) - 1.5).abs() < 1e-15);

        let mut row = Su2RankGrid::zeros(&space);
        let mut col = Su2RankGrid::zeros(&space);
        row[sa] = 2;
        col[sa] = 2;
        row[sb] = 2;
        col[sb] = 2;
        let mut sk = Su2Sketch::<f64>::zeros(space, k, &row, &col);
        {
            let mut ba = sk.block_mut(sa).unwrap();
            *ba.rb_mut().get_mut(0, 0) = 3.0;
            *ba.rb_mut().get_mut(1, 1) = 2.0;
        }
        {
            let mut bb = sk.block_mut(sb).unwrap();
            *bb.rb_mut().get_mut(0, 0) = 3.0;
            *bb.rb_mut().get_mut(1, 1) = 1.5;
        }

        let eps = 3.0f64.sqrt();

        let g = chop_energy(&sk, eps, Pooled::NONE);
        assert_eq!(g[sa], 1, "weighted chop must keep 1 in the cw<1 sector");
        assert_eq!(g[sb], 2, "weighted chop must keep 2 in the cw>1 sector");
        for s in space.bond_sectors(k) {
            if s != sa && s != sb {
                assert_eq!(g[s], 0, "no rank on empty sector {s:?}");
            }
        }

        let g3 = chop_energy(&sk, eps, Pooled(2));
        assert_eq!(g3[sa] + g3[sb], 2, "rmax=2 must cap the pooled kept count");
        assert_eq!(g3[sa], 1);
        assert_eq!(g3[sb], 1);
    }

    fn diag_sketch(vals: &[f64]) -> Su2Sketch<f64> {
        let space = Su2Space::new(6, 4, 0);
        let (k, s) = (3usize, Su2Sector::new(2, 0));
        assert!(space.contains(k, s));
        assert_eq!(centerweight(s.two_s, space.two_s), 1.0);
        let mut ranks = Su2RankGrid::zeros(&space);
        ranks[s] = vals.len();
        let mut sk = Su2Sketch::<f64>::zeros(space, k, &ranks, &ranks);
        let mut b = sk.block_mut(s).unwrap();
        for (i, &v) in vals.iter().enumerate() {
            *b.rb_mut().get_mut(i, i) = v;
        }
        sk
    }

    #[test]
    fn chop_global_keeps_boundary_value_above_magnitude_two() {
        let ranks = chop_global(std::slice::from_ref(&diag_sketch(&[3.0, 6.0, 10.0])), 3.5);
        assert_eq!(ranks[0].values().iter().sum::<usize>(), 2);
    }
}
