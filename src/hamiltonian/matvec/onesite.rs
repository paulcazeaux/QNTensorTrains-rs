// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use faer::prelude::ReborrowMut;
use faer::{Mat, MatMut, MatRef, unzip, zip};
use rayon::prelude::*;

use super::cell::{ColShard, cell_apply_left, cell_apply_right};
use super::scheduling::PAR_MIN_TASKS;
use crate::core::factor::Sketch;
use crate::core::{Family, SparseCore};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::instrument;
use crate::qn::{Sector, SectorGrid};
use crate::scalar::Scalar;

/// Cached H-extended range tables for repeated one-site matvecs at a fixed site (fixed for the
/// duration of a site solve — keep-rank moves don't change them).
pub struct OneSiteWorkspace {
    /// [`SparseHamiltonian::row_ranges`]`(Xₖ)`.
    pub i_tab: SectorGrid<Vec<Range<usize>>>,
    /// [`SparseHamiltonian::col_ranges`]`(Xₖ)`.
    pub j_tab: SectorGrid<Vec<Range<usize>>>,
}

/// Build the range tables cached by a site solve.
pub fn onesite_workspace<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &SparseCore<T>,
) -> OneSiteWorkspace {
    OneSiteWorkspace {
        i_tab: h.row_ranges(x),
        j_tab: h.col_ranges(x),
    }
}

/// `y = Fᴸ·H·x`, over every output sector (`keep = |_| true`). Allocating wrapper over
/// [`SparseHamiltonian::sketch_h_mul_into`]. Drives the fused
/// [`crate::hamiltonian::rayleigh_quotient`]/[`crate::hamiltonian::xt_h_y`] env sweep; also exercised directly by the sketched-apply invariant tests.
pub fn sketch_h_mul<T: Scalar>(
    fl: &Sketch<T>,
    h: &SparseHamiltonian<T>,
    x: &SparseCore<T>,
) -> SparseCore<T> {
    let i_tab = h.row_ranges(x);
    let j_tab = h.col_ranges(x);
    let mut y = SparseCore::zeros(x.space, x.k, &fl.row_ranks(), &h.col_ranks(x));
    h.sketch_h_mul_into(&mut y, fl, x, |_| true, &i_tab, &j_tab);
    y
}

/// `y = H·x·Fᴿ`, over every output sector. Column-side mirror of [`sketch_h_mul`]. The streamed ALS
/// path uses the private `h_sketch_mul_into` band form; this full-sector wrapper is pinned by the
/// sketched-apply invariant tests (the left/right contraction-equivalence check).
pub fn h_sketch_mul<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &SparseCore<T>,
    fr: &Sketch<T>,
) -> SparseCore<T> {
    let i_tab = h.row_ranges(x);
    let j_tab = h.col_ranges(x);
    let mut y = SparseCore::zeros(x.space, x.k, &h.row_ranks(x), &fr.col_ranks());
    h.h_sketch_mul_into(&mut y, x, fr, |_| true, &i_tab, &j_tab);
    y
}

impl<T: Scalar> SparseHamiltonian<T> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sketch_h_mul_into(
        &self,
        y: &mut SparseCore<T>,
        fl: &Sketch<T>,
        x: &SparseCore<T>,
        keep: impl Fn(Sector) -> bool,
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
    ) {
        let space = self.space;
        let k = x.k;
        let sectors_k = space.bond_sectors(k);

        let apply_nest = |yb: MatMut<'_, T>, y_occ: Family, n1: Sector| {
            self.sketch_h_mul_block_into(yb, fl, x, y_occ, n1, &sectors_k, i_tab, j_tab);
        };

        let tasks: Vec<(Family, Sector)> = Family::ALL
            .into_iter()
            .flat_map(|y_occ| y.transitions(y_occ).map(move |n1| (y_occ, n1)))
            .filter(|&(y_occ, n1)| keep(y_occ.apply(n1, space.pg.irrep_of(k))))
            .collect();

        if tasks.len() < PAR_MIN_TASKS {
            instrument::shm_call(false);
            let _t = instrument::tic();
            for (y_occ, n1) in tasks {
                if let Some(yb) = y.block_mut(y_occ, n1) {
                    apply_nest(yb, y_occ, n1);
                }
            }
            instrument::toc(instrument::slot::SHM_SERIAL, _t);
        } else {
            instrument::shm_call(true);
            let _t = instrument::tic();
            let region = instrument::Region::begin();
            let dims: Vec<(Family, Sector, usize, usize)> = tasks
                .iter()
                .filter_map(|&(o, n1)| y.block(o, n1).map(|b| (o, n1, b.nrows(), b.ncols())))
                .collect();
            let results: Vec<(Family, Sector, Mat<T>)> = dims
                .into_par_iter()
                .map(|(o, n1, r, c)| {
                    let _tt = instrument::tic();
                    let mut m = Mat::<T>::zeros(r, c);
                    apply_nest(m.as_mut(), o, n1);
                    region.task(_tt, r * c);
                    (o, n1, m)
                })
                .collect();
            for (o, n1, m) in results {
                let mut yb = y
                    .block_mut(o, n1)
                    .expect("sketch_h_mul_into: computed block has no output slot");
                zip!(yb.rb_mut(), m.as_ref()).for_each(|unzip!(d, s)| *d += *s);
            }
            region.end();
            instrument::toc(instrument::slot::SHM_PAR, _t);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sketch_h_mul_block_into(
        &self,
        yb: MatMut<'_, T>,
        fl: &Sketch<T>,
        x: &SparseCore<T>,
        y_occ: Family,
        n1: Sector,
        sectors_k: &[Sector],
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
    ) {
        self.sketch_h_mul_block_shard_into(
            yb,
            fl,
            x,
            y_occ,
            n1,
            sectors_k,
            i_tab,
            j_tab,
            ColShard::ALL,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sketch_h_mul_block_shard_into(
        &self,
        mut yb: MatMut<'_, T>,
        fl: &Sketch<T>,
        x: &SparseCore<T>,
        y_occ: Family,
        n1: Sector,
        sectors_k: &[Sector],
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
        shard: ColShard,
    ) {
        let k = x.k;
        let q = y_occ.apply(n1, self.space.pg.irrep_of(k));
        let fblk = match fl.block(n1) {
            Some(b) if b.nrows() > 0 => b,
            _ => return,
        };
        let i_ranges = &i_tab[n1];
        let j_ranges = &j_tab[q];
        for rec in self.nest_slot(k, y_occ, n1) {
            let x_occ = Family::from_physical_index(rec.x_occ as usize);
            let m = sectors_k[rec.m_ord as usize];
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
    pub(crate) fn h_sketch_mul_into(
        &self,
        y: &mut SparseCore<T>,
        x: &SparseCore<T>,
        fr: &Sketch<T>,
        keep: impl Fn(Sector) -> bool,
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
    ) {
        let space = self.space;
        let k = x.k;
        let sectors_k = space.bond_sectors(k);

        let apply_nest = |yb: MatMut<'_, T>, y_occ: Family, n1: Sector| {
            self.h_sketch_mul_block_into(yb, x, fr, y_occ, n1, &sectors_k, i_tab, j_tab);
        };

        let tasks: Vec<(Family, Sector)> = Family::ALL
            .into_iter()
            .flat_map(|y_occ| y.transitions(y_occ).map(move |n1| (y_occ, n1)))
            .filter(|&(_, n1)| keep(n1))
            .collect();

        if tasks.len() < PAR_MIN_TASKS {
            instrument::shm_call(false);
            let _t = instrument::tic();
            for (y_occ, n1) in tasks {
                if let Some(yb) = y.block_mut(y_occ, n1) {
                    apply_nest(yb, y_occ, n1);
                }
            }
            instrument::toc(instrument::slot::SHM_SERIAL, _t);
        } else {
            instrument::shm_call(true);
            let _t = instrument::tic();
            let region = instrument::Region::begin();
            let dims: Vec<(Family, Sector, usize, usize)> = tasks
                .iter()
                .filter_map(|&(o, n1)| y.block(o, n1).map(|b| (o, n1, b.nrows(), b.ncols())))
                .collect();
            let results: Vec<(Family, Sector, Mat<T>)> = dims
                .into_par_iter()
                .map(|(o, n1, r, c)| {
                    let _tt = instrument::tic();
                    let mut m = Mat::<T>::zeros(r, c);
                    apply_nest(m.as_mut(), o, n1);
                    region.task(_tt, r * c);
                    (o, n1, m)
                })
                .collect();
            for (o, n1, m) in results {
                let mut yb = y
                    .block_mut(o, n1)
                    .expect("h_sketch_mul_into: computed block has no output slot");
                zip!(yb.rb_mut(), m.as_ref()).for_each(|unzip!(d, s)| *d += *s);
            }
            region.end();
            instrument::toc(instrument::slot::SHM_PAR, _t);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h_sketch_mul_block_into(
        &self,
        yb: MatMut<'_, T>,
        x: &SparseCore<T>,
        fr: &Sketch<T>,
        y_occ: Family,
        n1: Sector,
        sectors_k: &[Sector],
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
    ) {
        let q = y_occ.apply(n1, self.space.pg.irrep_of(x.k));
        let fblk = match fr.block(q) {
            Some(b) => b,
            None => return,
        };
        self.h_sketch_mul_cells_into(yb, x, fblk, y_occ, n1, sectors_k, i_tab, j_tab);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn h_sketch_mul_block_cols_into(
        &self,
        yb: MatMut<'_, T>,
        x: &SparseCore<T>,
        fr: &Sketch<T>,
        y_occ: Family,
        n1: Sector,
        sectors_k: &[Sector],
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
        cols: Range<usize>,
    ) {
        let q = y_occ.apply(n1, self.space.pg.irrep_of(x.k));
        let fblk = match fr.block(q) {
            Some(b) => b.subcols(cols.start, cols.len()),
            None => return,
        };
        self.h_sketch_mul_cells_into(yb, x, fblk, y_occ, n1, sectors_k, i_tab, j_tab);
    }

    #[allow(clippy::too_many_arguments)]
    fn h_sketch_mul_cells_into(
        &self,
        mut yb: MatMut<'_, T>,
        x: &SparseCore<T>,
        fblk: MatRef<'_, T>,
        y_occ: Family,
        n1: Sector,
        sectors_k: &[Sector],
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
    ) {
        let k = x.k;
        let q = y_occ.apply(n1, self.space.pg.irrep_of(k));
        let i_ranges = &i_tab[n1];
        let j_ranges = &j_tab[q];
        for rec in self.nest_slot(k, y_occ, n1) {
            let x_occ = Family::from_physical_index(rec.x_occ as usize);
            let m = sectors_k[rec.m_ord as usize];
            let xb = match x.block(x_occ, m) {
                Some(b) => b,
                None => continue,
            };
            cell_apply_right(
                yb.rb_mut(),
                xb,
                fblk,
                i_ranges,
                j_ranges,
                &rec.csr,
                &rec.csc,
            );
        }
    }

    pub(super) fn left_band(
        &self,
        fl: &Sketch<T>,
        x: &SparseCore<T>,
        q: Sector,
        rq: usize,
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
    ) -> SparseCore<T> {
        let mut col_band = crate::qn::RankGrid::zeros(&x.space);
        col_band[q] = rq;
        let mut band = SparseCore::zeros(x.space, x.k, &fl.row_ranks(), &col_band);
        self.sketch_h_mul_into(&mut band, fl, x, |s| s == q, i_tab, j_tab);
        band
    }

    pub(super) fn right_band(
        &self,
        x: &SparseCore<T>,
        fr: &Sketch<T>,
        q: Sector,
        rq: usize,
        i_tab: &SectorGrid<Vec<Range<usize>>>,
        j_tab: &SectorGrid<Vec<Range<usize>>>,
    ) -> SparseCore<T> {
        let mut row_band = crate::qn::RankGrid::zeros(&x.space);
        row_band[q] = rq;
        let mut band = SparseCore::zeros(x.space, x.k, &row_band, &fr.col_ranks());
        self.h_sketch_mul_into(&mut band, x, fr, |s| s == q, i_tab, j_tab);
        band
    }
}

#[cfg(test)]
mod phase5_tests {
    use super::{h_sketch_mul, onesite_workspace, sketch_h_mul};
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::test_util::randomized_state;
    use crate::{Family, TtVector};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t1_sketch_h_mul_contraction_equivalence() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[0, 1], -0.7);
        t.set(&[1, 0], -0.7);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 4, 1], -1.0);
        v.set(&[1, 4, 0, 2], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A0C);
        let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);
        let k = 3usize;
        x.move_core(k, true);
        x.check();

        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let fr = right_to_left_sketching(&h, &x);
        let ws = onesite_workspace(&h, x.core(k));
        let xk = x.core(k).clone();

        let expected = h.sketched_hamiltonian(&xk, &fl, &fr[k], &ws);
        assert!(
            expected.norm() > 0.0,
            "vacuous contraction-equivalence test"
        );
        let y_left = sketch_h_mul(&fl, &h, &xk);
        let y_right = h_sketch_mul(&h, &xk, &fr[k]);

        let mut max_left = 0.0f64;
        let mut max_right = 0.0f64;
        for fam in Family::ALL {
            for &l in &expected.row_qn.clone() {
                let q = fam.apply(l, h.space.pg.irrep_of(k));
                let Some(eb) = expected.block(fam, l) else {
                    continue;
                };
                let (m, n) = (eb.nrows(), eb.ncols());
                if let (Some(yb), Some(fb)) = (y_left.block(fam, l), fr[k].block(q)) {
                    for a in 0..m {
                        for c in 0..n {
                            let mut acc = 0.0f64;
                            for b in 0..yb.ncols() {
                                acc += *yb.get(a, b) * *fb.get(b, c);
                            }
                            max_left = max_left.max((acc - *eb.get(a, c)).abs());
                        }
                    }
                }
                if let (Some(flb), Some(yb)) = (fl.block(l), y_right.block(fam, l)) {
                    for a in 0..m {
                        for c in 0..n {
                            let mut acc = 0.0f64;
                            for b in 0..flb.ncols() {
                                acc += *flb.get(a, b) * *yb.get(b, c);
                            }
                            max_right = max_right.max((acc - *eb.get(a, c)).abs());
                        }
                    }
                }
            }
        }
        assert!(
            max_left < 1e-12,
            "sketch_h_mul contraction mismatch: {max_left}"
        );
        assert!(
            max_right < 1e-12,
            "h_sketch_mul contraction mismatch: {max_right}"
        );
    }
}
