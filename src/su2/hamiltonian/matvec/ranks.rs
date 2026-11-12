// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use faer::prelude::ReborrowMut;
use faer::{MatMut, MatRef, unzip, zip};

use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid};
use crate::su2::vector::Su2TtVector;

fn cumulative_ranges(widths: impl Iterator<Item = usize>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for w in widths {
        ranges.push(start..start + w);
        start += w;
    }
    ranges
}

fn axpy_block<T: Scalar>(
    mut dst: MatMut<'_, T>,
    i0: usize,
    j0: usize,
    nrows: usize,
    ncols: usize,
    alpha: T,
    src: MatRef<'_, T>,
) {
    let d_sub = dst.rb_mut().submatrix_mut(i0, j0, nrows, ncols);
    let s_sub = src.submatrix(0, 0, nrows, ncols);
    zip!(d_sub, s_sub).for_each(|unzip!(d, s)| *d += alpha * *s);
}

impl<T: Scalar> Su2SparseHamiltonian<T> {
    /// Operator-band-expanded row ranks of the output core replacing `x`'s core at site `k`.
    /// Boundary `k==0` passes through `x.row_ranks`; else sums `x.row_rank` over each output
    /// sector's coupled x-row sub-sectors (`state_qns[k]`).
    pub fn row_ranks(&self, x: &Su2Core<T>) -> Su2RankGrid {
        let space = self.space;
        let k = x.k;
        if k == 0 {
            return x.row_ranks.clone();
        }
        let sq = &self.state_qns[k];
        let mut ranks = Su2RankGrid::zeros(&space);
        for &s in &x.row_qn {
            ranks[s] = sq[s].iter().map(|&m| x.row_ranks[m]).sum();
        }
        ranks
    }

    /// Operator-band-expanded column ranks. Mirror of [`Self::row_ranks`] on the column side (boundary `k==d-1`).
    pub fn col_ranks(&self, x: &Su2Core<T>) -> Su2RankGrid {
        let space = self.space;
        let k = x.k;
        if k == space.d - 1 {
            return x.col_ranks.clone();
        }
        let sq = &self.state_qns[k + 1];
        let mut ranks = Su2RankGrid::zeros(&space);
        for &s in &x.col_qn {
            ranks[s] = sq[s].iter().map(|&m| x.col_ranks[m]).sum();
        }
        ranks
    }

    /// Per-sector contiguous row-index ranges partitioning the H-extended row block (the
    /// `row_ranges(H,x)`) — the block-offset table every matvec kernel below
    /// consumes.
    pub fn row_ranges(&self, x: &Su2Core<T>) -> Su2SectorGrid<Vec<Range<usize>>> {
        let space = self.space;
        let k = x.k;
        let mut out = Su2SectorGrid::filled(&space, Vec::new());
        if k == 0 {
            let s0 = Su2Sector::new(0, 0);
            #[allow(clippy::single_range_in_vec_init)]
            {
                out[s0] = vec![0..x.row_ranks[s0]];
            }
        } else {
            let sq = &self.state_qns[k];
            for &s in &x.row_qn {
                out[s] = cumulative_ranges(sq[s].iter().map(|&m| x.row_ranks[m]));
            }
        }
        out
    }

    /// Per-sector contiguous column-index ranges. Mirror of [`Self::row_ranges`].
    pub fn col_ranges(&self, x: &Su2Core<T>) -> Su2SectorGrid<Vec<Range<usize>>> {
        let space = self.space;
        let k = x.k;
        let mut out = Su2SectorGrid::filled(&space, Vec::new());
        if k == space.d - 1 {
            let last = Su2Sector::with_irrep(space.n, space.two_s, space.target);
            #[allow(clippy::single_range_in_vec_init)]
            {
                out[last] = vec![0..x.col_ranks[last]];
            }
        } else {
            let sq = &self.state_qns[k + 1];
            for &s in &x.col_qn {
                out[s] = cumulative_ranges(sq[s].iter().map(|&m| x.col_ranks[m]));
            }
        }
        out
    }

    /// The full materialized `H·x`. **THE cross-symmetry gate**: `expand(H·x, two_sz) ≈
    /// H_abelian·expand(x, two_sz)` staged per arity (1-body → 2-body → 3-body incl. rank-3/2
    /// fully-site-distinct patterns) is the phase spine every other kernel is ultimately validated
    /// through. A pure per-triple axpy over the output-family × input-family nest, weight-free
    /// (`α` is the raw operator coefficient); iterate the fixed cell order, and within a cell the
    /// CSR entry order (determinism-load-bearing, mirrors
    /// `crate::hamiltonian::SparseHamiltonian::apply`). Serial; a per-output-block spawn is a
    /// benchmark-milestone deferral.
    pub fn apply(&self, x: &Su2TtVector<T>) -> Su2TtVector<T> {
        assert_eq!(self.space, x.space, "apply: Su2Space mismatch");
        let space = self.space;
        let d = space.d;
        let cores: Vec<Su2Core<T>> = (0..d).map(|k| self.build_core(x.core(k))).collect();
        Su2TtVector::new(space, cores)
    }

    pub(crate) fn build_core(&self, x: &Su2Core<T>) -> Su2Core<T> {
        let space = self.space;
        let k = x.k;
        let g = space.pg.irrep_of(k);
        let rr = self.row_ranks(x);
        let cr = self.col_ranks(x);
        let mut y = Su2Core::zeros(space, k, &rr, &cr);
        let i_tab = self.row_ranges(x);
        let j_tab = self.col_ranges(x);
        let nest = &self.nests[k];

        for occ_n in Family::ALL {
            let out_secs: Vec<Su2Sector> = y.transitions(occ_n).collect();
            for n1 in out_secs {
                let qout = occ_n.apply(n1, g).expect("output transition admissible");
                let i_ranges = &i_tab[n1];
                let j_ranges = &j_tab[qout];
                let mut yb = match y.block_mut(occ_n, n1) {
                    Some(b) => b,
                    None => continue,
                };
                if yb.nrows() == 0 || yb.ncols() == 0 {
                    continue;
                }
                for rec in nest.slot(occ_n, n1) {
                    let occ_m = Family::from_physical_index(rec.x_occ as usize);
                    let m = nest.sector_of(rec);
                    let xb = match x.block(occ_m, m) {
                        Some(b) => b,
                        None => continue,
                    };
                    for (row, col, alpha) in rec.triples() {
                        let ir = &i_ranges[row as usize];
                        let jr = &j_ranges[col as usize];
                        axpy_block(
                            yb.rb_mut(),
                            ir.start,
                            jr.start,
                            ir.len(),
                            jr.len(),
                            alpha,
                            xb,
                        );
                    }
                }
            }
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use crate::su2::hamiltonian::Su2SparseHamiltonian;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randn_4tensor, randn_matrix, randomized_su2_state, seeded};

    #[test]
    fn s2_b5_hook_consistency_ranks() {
        let d = 5;
        let n = 3;
        let two_s = 1;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(555);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let t = randn_matrix::<f64, _>(d, &mut rng);
        let v = randn_4tensor::<f64, _>(d, &mut rng);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let z = h.apply(&x);
        for k in 0..d {
            let rr = h.row_ranks(x.core(k));
            let cr = h.col_ranks(x.core(k));
            assert_eq!(rr, z.core(k).row_ranks);
            assert_eq!(cr, z.core(k).col_ranks);
        }
    }
}
