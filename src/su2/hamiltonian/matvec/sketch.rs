// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use faer::prelude::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use rayon::prelude::*;

use crate::hamiltonian::matvec::{
    BandKeep, BandMask, ColShard, RowShard, cell_apply_left, cell_apply_right,
    cell_apply_right_shard,
};
use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::shard::{Region, Splitter, band_cuts, count_split};

use super::PAR_MIN_TASKS;

fn scaled_adjoint<T: Scalar>(b: MatRef<'_, T>, w: f64) -> Mat<T> {
    let a = T::from_f64(w);
    let mut out = Mat::<T>::zeros(b.ncols(), b.nrows());
    for j in 0..b.nrows() {
        for i in 0..b.ncols() {
            *out.get_mut(i, j) = b.get(j, i).conj_s() * a;
        }
    }
    out
}

impl<T: Scalar> Su2SparseHamiltonian<T> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sketch_h_mul_block_into(
        &self,
        yb: MatMut<'_, T>,
        fl: &Su2Sketch<T>,
        x: &Su2Core<T>,
        y_occ: Family,
        n1: Su2Sector,
        i_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        j_tab: &Su2SectorGrid<Vec<Range<usize>>>,
    ) {
        self.sketch_h_mul_block_shard_into(yb, fl, x, y_occ, n1, i_tab, j_tab, ColShard::ALL);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sketch_h_mul_block_shard_into<K: BandKeep>(
        &self,
        mut yb: MatMut<'_, T>,
        fl: &Su2Sketch<T>,
        x: &Su2Core<T>,
        y_occ: Family,
        n1: Su2Sector,
        i_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        j_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        shard: K,
    ) {
        let k = x.k;
        let g = self.space.pg.irrep_of(k);
        let nest = &self.nests[k];
        let Some(q) = y_occ.apply(n1, g) else { return };
        let fblk = match fl.block(n1) {
            Some(b) if b.nrows() > 0 => b,
            _ => return,
        };
        let i_ranges = &i_tab[n1];
        let j_ranges = &j_tab[q];
        for rec in nest.slot(y_occ, n1) {
            let x_occ = Family::from_physical_index(rec.x_occ as usize);
            let m = nest.sector_of(rec);
            let xb = match x.block(x_occ, m) {
                Some(b) => b,
                None => continue,
            };
            cell_apply_left(
                yb.rb_mut(),
                fblk,
                xb,
                i_ranges,
                j_ranges,
                &rec.csr,
                &rec.csc,
                shard,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h_sketch_mul_block_into(
        &self,
        yb: MatMut<'_, T>,
        x: &Su2Core<T>,
        fr: &Su2Sketch<T>,
        y_occ: Family,
        n1: Su2Sector,
        i_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        j_tab: &Su2SectorGrid<Vec<Range<usize>>>,
    ) {
        self.h_sketch_mul_block_shard_into(yb, x, fr, y_occ, n1, i_tab, j_tab, RowShard::ALL);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h_sketch_mul_block_shard_into<K: BandKeep>(
        &self,
        mut yb: MatMut<'_, T>,
        x: &Su2Core<T>,
        fr: &Su2Sketch<T>,
        y_occ: Family,
        n1: Su2Sector,
        i_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        j_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        shard: K,
    ) {
        let k = x.k;
        let g = self.space.pg.irrep_of(k);
        let nest = &self.nests[k];
        let Some(q) = y_occ.apply(n1, g) else { return };
        let fblk = match fr.block(q) {
            Some(b) => b,
            None => return,
        };
        let i_ranges = &i_tab[n1];
        let j_ranges = &j_tab[q];
        for rec in nest.slot(y_occ, n1) {
            let x_occ = Family::from_physical_index(rec.x_occ as usize);
            let m = nest.sector_of(rec);
            let xb = match x.block(x_occ, m) {
                Some(b) => b,
                None => continue,
            };
            cell_apply_right_shard(
                yb.rb_mut(),
                xb,
                fblk,
                i_ranges,
                j_ranges,
                &rec.csr,
                &rec.csc,
                shard,
            );
        }
    }

    fn nest_work(&self, x: &Su2Core<T>, y_occ: Family, n1: Su2Sector) -> u64 {
        let nest = &self.nests[x.k];
        nest.slot(y_occ, n1)
            .iter()
            .filter_map(|rec| {
                let x_occ = Family::from_physical_index(rec.x_occ as usize);
                let xb = x.block(x_occ, nest.sector_of(rec))?;
                Some(rec.csr.nnz() as u64 * xb.nrows() as u64 * xb.ncols() as u64)
            })
            .sum()
    }

    pub(crate) fn sketch_h_mul_into(
        &self,
        y: &mut Su2Core<T>,
        fl: &Su2Sketch<T>,
        x: &Su2Core<T>,
        keep: impl Fn(Su2Sector) -> bool,
        i_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        j_tab: &Su2SectorGrid<Vec<Range<usize>>>,
    ) {
        let space = self.space;
        let k = x.k;
        let g = space.pg.irrep_of(k);

        let apply_nest = |yb: MatMut<'_, T>, y_occ: Family, n1: Su2Sector| {
            self.sketch_h_mul_block_into(yb, fl, x, y_occ, n1, i_tab, j_tab);
        };

        let tasks: Vec<(Family, Su2Sector)> = Family::ALL
            .into_iter()
            .flat_map(|y_occ| y.transitions(y_occ).map(move |n1| (y_occ, n1)))
            .filter(|&(y_occ, n1)| y_occ.apply(n1, g).is_some_and(&keep))
            .collect();

        if tasks.len() < PAR_MIN_TASKS {
            for (y_occ, n1) in tasks {
                if let Some(yb) = y.block_mut(y_occ, n1) {
                    apply_nest(yb, y_occ, n1);
                }
            }
        } else {
            let views = task_views(y, &tasks);
            let costs: Vec<u64> = views
                .iter()
                .map(|(o, n1, yb)| yb.nrows() as u64 * self.nest_work(x, *o, *n1))
                .collect();
            let sp = Splitter::new(costs.iter().sum());
            let mut pieces: Vec<(Family, Su2Sector, ColShard, MatMut<'_, T>)> =
                Vec::with_capacity(views.len());
            for ((o, n1, yb), cost) in views.into_iter().zip(costs) {
                let bands = o.apply(n1, g).map_or(&[][..], |q| &j_tab[q][..]);
                let n = sp.count(cost, yb.ncols());
                let cuts = if n > 1 && bands.last().is_some_and(|r| r.end == yb.ncols()) {
                    band_cuts(bands, n)
                } else {
                    Vec::new()
                };
                if cuts.len() < 2 {
                    pieces.push((o, n1, ColShard::ALL, yb));
                    continue;
                }
                count_split(Region::SketchHMul);
                let mut rest = yb;
                let mut at = 0usize;
                for (o0, o1) in cuts {
                    let (lo, hi) = (bands[o0 as usize].start, bands[o1 as usize - 1].end);
                    let (head, tail) = rest.split_at_col_mut(hi - at);
                    pieces.push((o, n1, ColShard::new(o0, o1, lo), head));
                    rest = tail;
                    at = hi;
                }
            }
            pieces.into_par_iter().for_each(|(o, n1, sh, yb)| {
                self.sketch_h_mul_block_shard_into(yb, fl, x, o, n1, i_tab, j_tab, sh);
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h_sketch_mul_into(
        &self,
        y: &mut Su2Core<T>,
        x: &Su2Core<T>,
        fr: &Su2Sketch<T>,
        keep: impl Fn(Su2Sector) -> bool,
        i_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        j_tab: &Su2SectorGrid<Vec<Range<usize>>>,
    ) {
        let apply_nest = |yb: MatMut<'_, T>, y_occ: Family, n1: Su2Sector| {
            self.h_sketch_mul_block_into(yb, x, fr, y_occ, n1, i_tab, j_tab);
        };

        let tasks: Vec<(Family, Su2Sector)> = Family::ALL
            .into_iter()
            .flat_map(|y_occ| y.transitions(y_occ).map(move |n1| (y_occ, n1)))
            .filter(|&(_, n1)| keep(n1))
            .collect();

        if tasks.len() < PAR_MIN_TASKS {
            for (y_occ, n1) in tasks {
                if let Some(yb) = y.block_mut(y_occ, n1) {
                    apply_nest(yb, y_occ, n1);
                }
            }
        } else {
            let views = task_views(y, &tasks);
            let costs: Vec<u64> = views
                .iter()
                .map(|(o, n1, yb)| yb.ncols() as u64 * self.nest_work(x, *o, *n1))
                .collect();
            let sp = Splitter::new(costs.iter().sum());
            let mut pieces: Vec<(Family, Su2Sector, RowShard, MatMut<'_, T>)> =
                Vec::with_capacity(views.len());
            for ((o, n1, yb), cost) in views.into_iter().zip(costs) {
                let bands = &i_tab[n1][..];
                let n = sp.count(cost, yb.nrows());
                let cuts = if n > 1 && bands.last().is_some_and(|r| r.end == yb.nrows()) {
                    band_cuts(bands, n)
                } else {
                    Vec::new()
                };
                if cuts.len() < 2 {
                    pieces.push((o, n1, RowShard::ALL, yb));
                    continue;
                }
                count_split(Region::HSketchMul);
                let mut rest = yb;
                let mut at = 0usize;
                for (o0, o1) in cuts {
                    let (lo, hi) = (bands[o0 as usize].start, bands[o1 as usize - 1].end);
                    let (head, tail) = rest.split_at_row_mut(hi - at);
                    pieces.push((o, n1, RowShard::new(o0, o1, lo), head));
                    rest = tail;
                    at = hi;
                }
            }
            pieces.into_par_iter().for_each(|(o, n1, sh, yb)| {
                self.h_sketch_mul_block_shard_into(yb, x, fr, o, n1, i_tab, j_tab, sh);
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sketch_h_mul_masked_into(
        &self,
        y: &mut Su2Core<T>,
        fl: &Su2Sketch<T>,
        x: &Su2Core<T>,
        keep: &[(Su2Sector, Vec<usize>)],
        i_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        out_tab: &Su2SectorGrid<Vec<Range<usize>>>,
    ) {
        let g = self.space.pg.irrep_of(x.k);
        let masks = band_masks(&self.space, keep);
        let words = |o: Family, n1: Su2Sector| o.apply(n1, g).map(|q| &masks[q][..]);
        let tasks = masked_tasks(y, |o, n1| words(o, n1).is_some_and(|w| !w.is_empty()));
        let run = |yb: MatMut<'_, T>, o: Family, n1: Su2Sector| {
            let mask = BandMask::new(words(o, n1).expect("a kept target has a column sector"));
            self.sketch_h_mul_block_shard_into(yb, fl, x, o, n1, i_tab, out_tab, mask);
        };
        if tasks.len() < PAR_MIN_TASKS {
            for (o, n1) in tasks {
                if let Some(yb) = y.block_mut(o, n1) {
                    run(yb, o, n1);
                }
            }
        } else {
            task_views(y, &tasks)
                .into_par_iter()
                .for_each(|(o, n1, yb)| run(yb, o, n1));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h_sketch_mul_masked_into(
        &self,
        y: &mut Su2Core<T>,
        x: &Su2Core<T>,
        fr: &Su2Sketch<T>,
        keep: &[(Su2Sector, Vec<usize>)],
        j_tab: &Su2SectorGrid<Vec<Range<usize>>>,
        out_tab: &Su2SectorGrid<Vec<Range<usize>>>,
    ) {
        let masks = band_masks(&self.space, keep);
        let tasks = masked_tasks(y, |_, n1| !masks[n1].is_empty());
        let run = |yb: MatMut<'_, T>, o: Family, n1: Su2Sector| {
            let mask = BandMask::new(&masks[n1]);
            self.h_sketch_mul_block_shard_into(yb, x, fr, o, n1, out_tab, j_tab, mask);
        };
        if tasks.len() < PAR_MIN_TASKS {
            for (o, n1) in tasks {
                if let Some(yb) = y.block_mut(o, n1) {
                    run(yb, o, n1);
                }
            }
        } else {
            task_views(y, &tasks)
                .into_par_iter()
                .for_each(|(o, n1, yb)| run(yb, o, n1));
        }
    }
}

const UNKEPT_BAND: Range<usize> = usize::MAX..usize::MAX;

pub(crate) fn compact_band_table(
    space: &Su2Space,
    full: &Su2SectorGrid<Vec<Range<usize>>>,
    keep: &[(Su2Sector, Vec<usize>)],
) -> (Su2RankGrid, Su2SectorGrid<Vec<Range<usize>>>) {
    let mut ranks = Su2RankGrid::zeros(space);
    let mut table = Su2SectorGrid::from_fn(space, |s| vec![UNKEPT_BAND; full[s].len()]);
    for (s, cs) in keep {
        let (bands, slots) = (&full[*s], &mut table[*s]);
        let mut off = 0usize;
        for &c in cs {
            let w = bands[c].len();
            slots[c] = off..off + w;
            off += w;
        }
        ranks[*s] = off;
    }
    (ranks, table)
}

fn band_masks(space: &Su2Space, keep: &[(Su2Sector, Vec<usize>)]) -> Su2SectorGrid<Vec<u64>> {
    let mut masks = Su2SectorGrid::filled(space, Vec::new());
    for (s, cs) in keep {
        masks[*s] = BandMask::words_of(cs);
    }
    masks
}

fn masked_tasks<T: Scalar>(
    y: &Su2Core<T>,
    kept: impl Fn(Family, Su2Sector) -> bool,
) -> Vec<(Family, Su2Sector)> {
    Family::ALL
        .into_iter()
        .flat_map(|o| y.transitions(o).map(move |n1| (o, n1)))
        .filter(|&(o, n1)| kept(o, n1))
        .collect()
}

pub(crate) fn task_views<'y, T: Scalar>(
    y: &'y mut Su2Core<T>,
    tasks: &[(Family, Su2Sector)],
) -> Vec<(Family, Su2Sector, MatMut<'y, T>)> {
    let space = y.space;
    let mut want: [Su2SectorGrid<bool>; 4] =
        std::array::from_fn(|_| Su2SectorGrid::filled(&space, false));
    for &(o, n1) in tasks {
        want[o as usize][n1] = true;
    }
    y.stored_mut()
        .into_iter()
        .filter(|&(f, s, _)| want[f as usize][s])
        .collect()
}

/// `Fᴸ · (H·x)` at bond `x.k+1`, unfiltered. Allocating wrapper over
/// [`Su2SparseHamiltonian::sketch_h_mul_into`] (mirrors the abelian
/// `crate::hamiltonian::matvec::sketch_h_mul`).
pub fn sketch_h_mul<T: Scalar>(
    fl: &Su2Sketch<T>,
    h: &Su2SparseHamiltonian<T>,
    x: &Su2Core<T>,
) -> Su2Core<T> {
    let i_tab = h.row_ranges(x);
    let j_tab = h.col_ranges(x);
    let mut y = Su2Core::zeros(x.space, x.k, &fl.row_ranks(), &h.col_ranks(x));
    h.sketch_h_mul_into(&mut y, fl, x, |_| true, &i_tab, &j_tab);
    y
}

#[doc(hidden)]
pub fn sketch_h_mul_keep<T: Scalar>(
    fl: &Su2Sketch<T>,
    h: &Su2SparseHamiltonian<T>,
    x: &Su2Core<T>,
    keep: impl Fn(Su2Sector) -> bool,
) -> Su2Core<T> {
    let i_tab = h.row_ranges(x);
    let j_tab = h.col_ranges(x);
    let full_cols = h.col_ranks(x);
    let mut cols = Su2RankGrid::zeros(&x.space);
    for s in x.space.bond_sectors(x.k + 1) {
        if keep(s) {
            cols[s] = full_cols[s];
        }
    }
    let mut y = Su2Core::zeros(x.space, x.k, &fl.row_ranks(), &cols);
    h.sketch_h_mul_into(&mut y, fl, x, keep, &i_tab, &j_tab);
    y
}

/// `(H·x)·Fᴿ` at bond `x.k`, unfiltered. Allocating wrapper over
/// [`Su2SparseHamiltonian::h_sketch_mul_into`] (mirrors the abelian
/// `crate::hamiltonian::matvec::h_sketch_mul`).
pub fn h_sketch_mul<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2Core<T>,
    fr: &Su2Sketch<T>,
) -> Su2Core<T> {
    let i_tab = h.row_ranges(x);
    let j_tab = h.col_ranges(x);
    let mut y = Su2Core::zeros(x.space, x.k, &h.row_ranks(x), &fr.col_ranks());
    h.h_sketch_mul_into(&mut y, x, fr, |_| true, &i_tab, &j_tab);
    y
}

/// **THE θ²-weight-bearing kernel** — `H·x·r†`. The up/dn contributions carry the θ²-dual
/// storage-index ratio weight `(sl±1)/sl` = [`crate::su2::coreops::theta_sq`]`(f, l)`
/// (factored out of [`crate::su2::coreops::rproject`]: `(2S+2)/(2S+1)` for `Up`, `2S/(2S+1)`
/// for `Dn`, `1` for `Empty`/`Occ`). This kernel is the non-streamed reference form of
/// [`Su2SparseHamiltonian::sketching_step_left`]; it has NO caller. (Every consumer-side
/// environment fold uses the left form `adjoint(core(Ω,k)) · H · x`, and the right-environment
/// update inlines its own θ² product rather than calling this one.)
pub fn h_x_adjoint_r<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2Core<T>,
    r: &Su2Core<T>,
) -> Su2Sketch<T> {
    let space = h.space;
    let k = x.k;
    let g = space.pg.irrep_of(k);
    debug_assert_eq!(k, r.k, "h_x_adjoint_r: site mismatch");
    debug_assert_eq!(
        r.col_ranks,
        h.col_ranks(x),
        "h_x_adjoint_r: row_ranks(r†) must equal col_ranks(H,x)"
    );

    let i_tab = h.row_ranges(x);
    let j_tab = h.col_ranges(x);
    let mut y = Su2Sketch::zeros(space, k, &h.row_ranks(x), &r.row_ranks);
    let nest = &h.nests[k];

    for f in Family::ALL {
        let rows: Vec<Su2Sector> = r.transitions(f).collect();
        for l in rows {
            let Some(q) = f.apply(l, g) else { continue };
            if !space.contains(k + 1, q) {
                continue;
            }
            let Some(rb) = r.block(f, l) else { continue };
            let w = crate::su2::coreops::theta_sq(f, l);
            let rw = scaled_adjoint(rb, w);
            let i_ranges = &i_tab[l];
            let j_ranges = &j_tab[q];
            let Some(mut yb) = y.block_mut(l) else {
                continue;
            };
            for rec in nest.slot(f, l) {
                let x_occ = Family::from_physical_index(rec.x_occ as usize);
                let m = nest.sector_of(rec);
                let xb = match x.block(x_occ, m) {
                    Some(b) => b,
                    None => continue,
                };
                cell_apply_right(
                    yb.rb_mut(),
                    xb,
                    rw.as_ref(),
                    i_ranges,
                    j_ranges,
                    &rec.csr,
                    &rec.csc,
                );
            }
        }
    }
    y
}

#[cfg(test)]
mod shard_tests;

#[cfg(test)]
mod masked_tests;
