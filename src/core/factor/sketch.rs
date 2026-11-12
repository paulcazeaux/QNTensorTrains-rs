// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;

use super::gemm;
use crate::core::{Family, SparseCore};
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

const PAR_MIN_FOLD_SECTORS: usize = 2;

/// Block-diagonal factor object: one block per sector
/// (no family shifts). The triangular/diagonal special cases (`R`/`L`/`S`) are stored as plain
/// [`Mat`]s — the structure tag is not carried, and the invariant tests never inspect it.
#[derive(Clone)]
pub struct Sketch<T: Scalar> {
    /// Symmetry descriptor (needed for canonical-order sector iteration).
    pub space: QnSpace,
    /// Bond index this sketch lives on.
    pub k: usize,
    /// One optional block per sector, keyed by [`Sector`]; `None` = structurally absent.
    pub blocks: SectorGrid<Option<Mat<T>>>,
}

impl<T: Scalar> Sketch<T> {
    /// Adopt a block grid at bond `k`; ranks are inferred from the block dimensions.
    pub fn new(space: QnSpace, k: usize, blocks: SectorGrid<Option<Mat<T>>>) -> Self {
        Sketch { space, k, blocks }
    }

    /// Block-diagonal zeros with the given per-sector row/col ranks (one block per admissible
    /// sector at bond `k`).
    pub fn zeros(space: QnSpace, k: usize, row_ranks: &RankGrid, col_ranks: &RankGrid) -> Self {
        let blocks = SectorGrid::from_fn(&space, |s| {
            space
                .contains(k, s)
                .then(|| Mat::<T>::zeros(row_ranks[s], col_ranks[s]))
        });
        Sketch { space, k, blocks }
    }

    /// Rank-`r` identity frame at bond `k`: the `r×r` identity at every **reachable** sector
    /// (`qn::max_ranks[k][s] > 0`), a `0×0` block at admissible-but-unreachable ones, `None` off
    /// the admissible cone.
    pub fn identity(space: QnSpace, k: usize, r: usize) -> Self {
        let caps = (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space));
        let blocks = SectorGrid::from_fn(&space, |s| {
            space.contains(k, s).then(|| {
                let n = if caps.as_ref().is_none_or(|c| c[k][s] > 0) {
                    r
                } else {
                    0
                };
                let mut m = Mat::<T>::zeros(n, n);
                for i in 0..n {
                    *m.get_mut(i, i) = T::one();
                }
                m
            })
        });
        Sketch { space, k, blocks }
    }

    /// The bond index.
    #[inline]
    pub fn site(&self) -> usize {
        self.k
    }

    /// Admissible sectors of this bond, canonical order (`nup` fastest).
    #[inline]
    pub fn sectors(&self) -> Vec<Sector> {
        self.space.bond_sectors(self.k)
    }

    /// The block at sector `s`, if present.
    #[inline]
    pub fn block(&self, s: Sector) -> Option<MatRef<'_, T>> {
        let (n1, d1) = self.blocks.shape();
        if s.nup >= n1 || s.ndn >= d1 || usize::from(s.irrep.index()) >= self.blocks.nirrep() {
            return None;
        }
        self.blocks[s].as_ref().map(|m| m.as_ref())
    }

    /// Mutable block access.
    #[inline]
    pub fn block_mut(&mut self, s: Sector) -> Option<MatMut<'_, T>> {
        let (n1, d1) = self.blocks.shape();
        if s.nup >= n1 || s.ndn >= d1 || usize::from(s.irrep.index()) >= self.blocks.nirrep() {
            return None;
        }
        self.blocks[s].as_mut().map(|m| m.as_mut())
    }

    /// Per-sector row ranks (block row counts).
    pub fn row_ranks(&self) -> RankGrid {
        let mut r = RankGrid::zeros(&self.space);
        for s in self.sectors() {
            if let Some(b) = self.block(s) {
                r[s] = b.nrows();
            }
        }
        r
    }

    /// Per-sector column ranks (block column counts).
    pub fn col_ranks(&self) -> RankGrid {
        let mut r = RankGrid::zeros(&self.space);
        for s in self.sectors() {
            if let Some(b) = self.block(s) {
                r[s] = b.ncols();
            }
        }
        r
    }

    /// Frobenius norm over all blocks — used by the norm-orthonormalization sweeps to normalize
    /// `C`/`L`/`R` before folding.
    pub fn norm(&self) -> f64 {
        let mut s = 0.0f64;
        for (_, b) in self.blocks.iter() {
            if let Some(m) = b {
                for j in 0..m.ncols() {
                    for i in 0..m.nrows() {
                        let v = *m.get(i, j);
                        s += v.abs_s() * v.abs_s();
                    }
                }
            }
        }
        s.sqrt()
    }

    /// `self ← self + other` per sector — the accumulator the tangent-space Hessian environments
    /// need (`solvers::hess`: each `L̃`/`R̃` step sums a transported partial with a freshly
    /// inserted component term).
    pub fn add_assign_sketch(&mut self, other: &Sketch<T>) {
        debug_assert_eq!(self.k, other.k, "add_assign_sketch: bond mismatch");
        for s in self.sectors() {
            let Some(ob) = other.block(s) else {
                continue;
            };
            let Some(mut sb) = self.block_mut(s) else {
                continue;
            };
            debug_assert_eq!(
                (sb.nrows(), sb.ncols()),
                (ob.nrows(), ob.ncols()),
                "add_assign_sketch: block shape mismatch at sector {s:?}"
            );
            for j in 0..sb.ncols() {
                for i in 0..sb.nrows() {
                    *sb.rb_mut().get_mut(i, j) += *ob.get(i, j);
                }
            }
        }
    }

    pub(crate) fn relabel_irrep(&self, phi: Irrep) -> Self {
        let mut blocks = self.blocks.clone();
        blocks.relabel_irrep_in_place(phi);
        Sketch {
            space: self.space,
            k: self.k,
            blocks,
        }
    }

    /// `self ← self · b` per sector: each block is right-multiplied by `b`'s same-sector block.
    /// Used by `round_mut` (`U ← U·S`).
    pub fn rmul_sketch_mut(&mut self, b: &Sketch<T>) {
        debug_assert_eq!(self.k, b.k, "rmul_sketch_mut: bond mismatch");
        for s in self.sectors() {
            let (Some(self_blk), Some(bb)) = (self.block(s), b.block(s)) else {
                continue;
            };
            debug_assert!(
                bb.nrows() == self_blk.ncols() && bb.ncols() == self_blk.ncols(),
                "rmul_sketch_mut: B block at {s:?} must be square, dims == col_rank(self)"
            );
            let prod = gemm(self_blk, bb);
            let mut dst = self.block_mut(s).unwrap();
            let (m, n) = (prod.nrows(), prod.ncols());
            for j in 0..n {
                for i in 0..m {
                    *dst.rb_mut().get_mut(i, j) = *prod.get(i, j);
                }
            }
        }
    }
}

/// `C = adjoint(a) · b` (`a`, `b` at site `k`) → [`Sketch`] at bond `k+1`, shape `(col_ranks(a),
/// col_ranks(b))`. Zero-init + `Accum::Add` over the four families (every sector has ≥1
/// contributing family, matching the `dot` fold idiom).
pub fn adjoint_core_times_core<T: Scalar>(a: &SparseCore<T>, b: &SparseCore<T>) -> Sketch<T> {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(k, b.k, "adjoint_core_times_core: site mismatch");
    let mut c = Sketch::zeros(space, k + 1, &a.col_ranks, &b.col_ranks);
    let g = space.pg.irrep_of(k);
    for t in space.bond_sectors(k + 1) {
        let Some(mut dst) = c.block_mut(t) else {
            continue;
        };
        for f in Family::ALL {
            let Some(s) = f.unapply(t, g) else {
                continue;
            };
            if !space.contains(k, s) {
                continue;
            }
            let (Some(ab), Some(bb)) = (a.block(f, s), b.block(f, s)) else {
                continue;
            };
            matmul(
                dst.rb_mut(),
                Accum::Add,
                ab.adjoint(),
                bb,
                T::one(),
                Par::Seq,
            );
        }
    }
    c
}

/// `C = a · adjoint(b)` (`a`, `b` at site `k`) → [`Sketch`] at bond `k` (the row-bond mirror of
/// [`adjoint_core_times_core`], which folds at bond `k+1`): shape `(row_ranks(a),
/// row_ranks(b))`. Zero-init + `Accum::Add` over the four families out
/// of each row sector `l`: `block(C, l) = Σ_fam a[fam,l] · adjoint(b[fam,l])`. `debug_assert`s
/// `a.k == b.k` and matching col ranks per family (the contracted index).
pub fn core_times_adjoint_core<T: Scalar>(a: &SparseCore<T>, b: &SparseCore<T>) -> Sketch<T> {
    core_times_adjoint_core_impl(a, b, true)
}

pub(crate) fn core_times_adjoint_core_seq<T: Scalar>(
    a: &SparseCore<T>,
    b: &SparseCore<T>,
) -> Sketch<T> {
    core_times_adjoint_core_impl(a, b, false)
}

fn core_times_adjoint_core_impl<T: Scalar>(
    a: &SparseCore<T>,
    b: &SparseCore<T>,
    allow_par: bool,
) -> Sketch<T> {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(k, b.k, "core_times_adjoint_core: site mismatch");
    debug_assert!(
        a.space.same_lattice(&b.space),
        "core_times_adjoint_core: lattice mismatch, {:?} vs {:?}",
        a.space,
        b.space
    );
    let mut c = Sketch::zeros(space, k, &a.row_ranks, &b.row_ranks);
    let g = space.pg.irrep_of(k);

    let fold_one = |l: Sector, mut dst: MatMut<'_, T>| {
        for f in Family::ALL {
            let t = f.apply(l, g);
            if !space.contains(k + 1, t) {
                continue;
            }
            let (Some(ab), Some(bb)) = (a.block(f, l), b.block(f, l)) else {
                continue;
            };
            debug_assert_eq!(
                ab.ncols(),
                bb.ncols(),
                "core_times_adjoint_core: col-rank mismatch at contracted (family, col)"
            );
            matmul(
                dst.rb_mut(),
                Accum::Add,
                ab,
                bb.adjoint(),
                T::one(),
                Par::Seq,
            );
        }
    };
    let tasks: Vec<Sector> = space
        .bond_sectors(k)
        .into_iter()
        .filter(|&l| {
            a.row_ranks[l] > 0
                && b.row_ranks[l] > 0
                && Family::ALL.into_iter().any(|f| {
                    space.contains(k + 1, f.apply(l, g))
                        && a.block(f, l).is_some()
                        && b.block(f, l).is_some()
                })
        })
        .collect();

    if !allow_par || tasks.len() < PAR_MIN_FOLD_SECTORS {
        for l in tasks {
            if let Some(dst) = c.block_mut(l) {
                fold_one(l, dst);
            }
        }
    } else {
        let results: Vec<(Sector, Mat<T>)> = tasks
            .par_iter()
            .map(|&l| {
                let mut m = Mat::<T>::zeros(a.row_ranks[l], b.row_ranks[l]);
                fold_one(l, m.as_mut());
                (l, m)
            })
            .collect();
        for (l, m) in results {
            c.blocks[l] = Some(m);
        }
    }
    c
}

/// Core-producing `Sketch × Core`: result at `site(b)` with `(row_ranks(a), col_ranks(b))`; `a`'s
/// block folds into the row side of each family of `b`.
pub fn mul_sketch_left<T: Scalar>(a: &Sketch<T>, b: &SparseCore<T>) -> SparseCore<T> {
    let space = b.space;
    let k = b.k;
    debug_assert_eq!(a.k, k, "mul_sketch_left: bond mismatch");
    let mut out = SparseCore::zeros(space, k, &a.row_ranks(), &b.col_ranks);
    let g = space.pg.irrep_of(k);
    for &l in &b.row_qn {
        let Some(ablk) = a.block(l) else { continue };
        for f in Family::ALL {
            let t = f.apply(l, g);
            if !space.contains(k + 1, t) {
                continue;
            }
            let Some(bblk) = b.block(f, l) else { continue };
            let dst = out.block_mut(f, l).unwrap();
            if ablk.nrows() > 0 && ablk.ncols() > 0 && bblk.ncols() > 0 {
                matmul(dst, Accum::Replace, ablk, bblk, T::one(), Par::Seq);
            }
        }
    }
    out
}

/// Core-producing `Core × Sketch`: result at `site(a)` with `(row_ranks(a), col_ranks(c))`, `c`'s
/// block selected by the shifted target column sector.
pub fn mul_sketch_right<T: Scalar>(a: &SparseCore<T>, c: &Sketch<T>) -> SparseCore<T> {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(c.k, k + 1, "mul_sketch_right: bond mismatch");
    let mut out = SparseCore::zeros(space, k, &a.row_ranks, &c.col_ranks());
    let g = space.pg.irrep_of(k);
    for &l in &a.row_qn {
        for f in Family::ALL {
            let t = f.apply(l, g);
            if !space.contains(k + 1, t) {
                continue;
            }
            let (Some(ablk), Some(cblk)) = (a.block(f, l), c.block(t)) else {
                continue;
            };
            let dst = out.block_mut(f, l).unwrap();
            if ablk.nrows() > 0 && ablk.ncols() > 0 && cblk.ncols() > 0 {
                matmul(dst, Accum::Replace, ablk, cblk, T::one(), Par::Seq);
            }
        }
    }
    out
}

impl<T: Scalar> crate::mem::Footprint for Sketch<T> {
    fn bytes(&self) -> u64 {
        self.blocks
            .iter()
            .filter_map(|(_, b)| b.as_ref())
            .map(|m| crate::mem::block_bytes::<T>(m.nrows(), m.ncols()))
            .sum()
    }
}

#[cfg(test)]
mod pg3_tests {
    use crate::Sketch;
    use crate::qn::{QnSpace, Sector, max_ranks};
    use crate::test_util::pg3_space_d6;

    #[test]
    fn pg3_sketch_identity_boundary_is_single_sector_g1() {
        let space = QnSpace::new(6, 2, 2);
        let id0: Sketch<f64> = Sketch::identity(space, 0, 1);
        let rr0 = id0.row_ranks();
        for s in space.bond_sectors(0) {
            let want = if s == Sector::new(0, 0) { 1 } else { 0 };
            assert_eq!(rr0[s], want);
        }
        let idd: Sketch<f64> = Sketch::identity(space, space.d, 1);
        let last = Sector::with_irrep(space.nup, space.ndn, space.target);
        for s in space.bond_sectors(space.d) {
            assert_eq!(idd.row_ranks()[s], if s == last { 1 } else { 0 });
        }
    }

    #[test]
    fn pg3_sketch_identity_boundary_is_single_sector() {
        let (_pg, space) = pg3_space_d6();
        let id0: Sketch<f64> = Sketch::identity(space, 0, 1);
        let rr0 = id0.row_ranks();
        for s in space.bond_sectors(0) {
            let want = if s == Sector::new(0, 0) { 1 } else { 0 };
            assert_eq!(
                rr0[s], want,
                "bond 0 boundary sketch must live only at (0,0,IDENTITY)"
            );
        }

        let idd: Sketch<f64> = Sketch::identity(space, space.d, 1);
        let last = Sector::with_irrep(space.nup, space.ndn, space.target);
        for s in space.bond_sectors(space.d) {
            assert_eq!(
                idd.row_ranks()[s],
                if s == last { 1 } else { 0 },
                "bond d {s:?}"
            );
        }

        let caps = max_ranks(&space);
        let k = 3;
        let idk: Sketch<f64> = Sketch::identity(space, k, 3);
        let mut any_live = false;
        let mut any_dead = false;
        for s in space.bond_sectors(k) {
            let block = idk
                .block(s)
                .expect("admissible sector must carry a (possibly 0x0) block");
            if caps[k][s] > 0 {
                any_live = true;
                assert_eq!((block.nrows(), block.ncols()), (3, 3));
                for i in 0..3 {
                    for j in 0..3 {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert_eq!(*block.get(i, j), want);
                    }
                }
            } else {
                any_dead = true;
                assert_eq!(
                    (block.nrows(), block.ncols()),
                    (0, 0),
                    "bond {k} {s:?} must be 0x0"
                );
            }
        }
        assert!(any_live, "vacuous: no reachable sector at bond {k}");
        assert!(any_dead, "vacuous: no unreachable sector at bond {k}");
    }
}

#[cfg(test)]
mod seq_twin_tests {
    use super::{
        PAR_MIN_FOLD_SECTORS, Sketch, core_times_adjoint_core, core_times_adjoint_core_seq,
    };
    use crate::core::Family;
    use crate::qn::QnSpace;
    use crate::test_util::{randomized_state, seeded};
    use crate::vector::TtVector;

    fn sketch_bits_eq(a: &Sketch<f64>, b: &Sketch<f64>, what: &str) {
        assert_eq!(a.k, b.k, "{what}: bond");
        for (s, x) in a.blocks.iter() {
            match (x.as_ref(), b.blocks[s].as_ref()) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(
                        (x.nrows(), x.ncols()),
                        (y.nrows(), y.ncols()),
                        "{what}: shape at {s:?}"
                    );
                    for j in 0..x.ncols() {
                        for i in 0..x.nrows() {
                            assert_eq!(
                                x[(i, j)].to_bits(),
                                y[(i, j)].to_bits(),
                                "{what}: entry ({i},{j}) at {s:?}"
                            );
                        }
                    }
                }
                _ => panic!("{what}: block presence differs at {s:?}"),
            }
        }
    }

    #[test]
    fn seq_twin_matches_forking_fold_bitwise() {
        let space = QnSpace::new(6, 2, 2);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut seeded(0xC0DE_F017));
        let y: TtVector<f64> = crate::vector::tt_randn(space, &x.ranks, &mut seeded(0xC0DE_F018));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("rayon pool");
        for k in 1..space.d - 1 {
            let a = x.core(k);
            let g = space.pg.irrep_of(k);
            let live = space
                .bond_sectors(k)
                .into_iter()
                .filter(|&l| {
                    a.row_ranks[l] > 0
                        && Family::ALL.into_iter().any(|f| {
                            space.contains(k + 1, f.apply(l, g)) && a.block(f, l).is_some()
                        })
                })
                .count();
            assert!(
                live >= PAR_MIN_FOLD_SECTORS,
                "site {k}: {live} live sectors — the forking branch would not run (vacuous)"
            );
            for (b, what) in [(a, "(x, x)"), (y.core(k), "(x, y)")] {
                let (forked, seq) = pool.install(|| {
                    (
                        core_times_adjoint_core(a, b),
                        core_times_adjoint_core_seq(a, b),
                    )
                });
                sketch_bits_eq(&seq, &forked, &format!("site {k}, {what}"));
            }
        }
    }
}
