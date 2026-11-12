// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) core-level block algebra + θ-weighted factorization wrappers. Folds
//! [`crate::su2::theta`]/[`crate::su2::centerweight`] AROUND the shared sector-agnostic Mat-level
//! kernels in `src/core/factor.rs` (`my_qc`, `my_cq`, `thin_q`, `svd_uvt_svals`/`svd_u_svals`,
//! `gemm`) — **never modify factor.rs, never call faer outside the `Mat<T>` block type**.

use std::borrow::Cow;

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, MatMut, MatRef, Par};

use rayon::prelude::*;

use crate::core::algebra::{block_abs2, block_axpy, block_dot, block_scale};
use crate::core::factor::{gemm, my_cq, my_qc, svd_uvt_svals, thin_q};
use crate::knobs;
use crate::par;
use crate::scalar::{Scalar, gemm_par};
use crate::su2::core::{BlockGrid, Family, Su2Core};
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{
    LabelTables, Slot, Su2Sector, SymGrid, SymLabel, SymSpace, SymStep, Symmetry,
    for_each_nonempty_slot, storage_labels,
};

pub(crate) const PAR_MIN_COREOPS_BLOCKS: usize = 2;

pub(crate) const PAR_MIN_COREOPS_WORK: usize = 1 << 20;

pub(crate) const PAR_MIN_COREOPS_TASK_WORK: usize = 1 << 14;

pub(crate) fn step_dot_center<T: Scalar, S: Symmetry>(
    a: &Su2Core<T, S>,
    b: &Su2Core<T, S>,
    family: S::Step,
) -> T {
    let (g, two_j) = (a.space.site_ctx(a.k), a.space.total_two_j());
    let (ab, bb) = (a.store().blocks(), b.store().blocks());
    let mut acc = T::zero();
    for i in a.step_range(family) {
        let (_, l) = a.stored_label(i);
        let t = family
            .step_apply(l, g)
            .expect("stored block implies admissible transition");
        let cw = T::from_f64(crate::su2::centerweight(t.two_j(), two_j));
        acc += cw * block_dot(ab[i].as_ref(), bb[i].as_ref());
    }
    acc
}

pub(crate) fn step_abs2_center<T: Scalar, S: Symmetry>(a: &Su2Core<T, S>, family: S::Step) -> f64 {
    let (g, two_j) = (a.space.site_ctx(a.k), a.space.total_two_j());
    let ab = a.store().blocks();
    let mut acc = 0.0f64;
    for i in a.step_range(family) {
        let (_, l) = a.stored_label(i);
        let t = family
            .step_apply(l, g)
            .expect("stored block implies admissible transition");
        let cw = crate::su2::centerweight(t.two_j(), two_j);
        acc += cw * block_abs2(ab[i].as_ref());
    }
    acc
}

#[inline]
pub(crate) fn assert_same_blocks<T: Scalar, S: Symmetry>(
    a: &Su2Core<T, S>,
    b: &Su2Core<T, S>,
    what: &str,
) {
    assert!(
        a.store().same_keys(b.store()),
        "{what}: the two cores store different blocks at site {} (their rank grids differ)",
        a.k
    );
}

/// `C = adjoint(a) · b` (`a`, `b` at site `k`) → [`Su2Sketch`] at bond `k+1`, shape
/// `(col_ranks(a), col_ranks(b))`. PLAIN adjoint contraction (all weights 1) — this is
/// EXACTLY the "outer op" fold `su2::ops::dot` already does inline; mirror of the abelian
/// `adjoint_core_times_core` (`core/factor/`) with SU(2) 4-family connectivity + the
/// shifted-sector (`dn`: `two_s+1`) incoming lookup substituted for the abelian 2-family one.
pub fn adjoint_core_times_core<T: Scalar, S: Symmetry>(
    a: &Su2Core<T, S>,
    b: &Su2Core<T, S>,
) -> Su2Sketch<T, S> {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(k, b.k, "adjoint_core_times_core: site mismatch");
    let g = space.site_ctx(k);
    let mut c = Su2Sketch::<T, S>::zeros_sym(space, k + 1, &a.col_ranks, &b.col_ranks);
    for t in space.bond_labels(k + 1) {
        let Some(mut dst) = c.block_mut(t) else {
            continue;
        };
        for &f in S::STEPS {
            let Some(s) = f.step_unapply(t, g) else {
                continue;
            };
            if !space.admissible(k, s) {
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

/// `C = a · adjoint(b)` (`a`, `b` at site `k`) → [`Su2Sketch`] at bond `k`, shape `(row_ranks(a),
/// row_ranks(b))`. PLAIN adjoint contraction (all weights 1) — mirror of the abelian
/// `core_times_adjoint_core` (`core/factor/`).
pub fn core_times_adjoint_core<T: Scalar, S: Symmetry>(
    a: &Su2Core<T, S>,
    b: &Su2Core<T, S>,
) -> Su2Sketch<T, S> {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(k, b.k, "core_times_adjoint_core: site mismatch");
    debug_assert!(
        a.space.same_lattice(&b.space),
        "core_times_adjoint_core: orbital-lattice mismatch ({:?} vs {:?})",
        a.space,
        b.space
    );
    let g = space.site_ctx(k);
    let fold_one = |l: S::Label, mut dst: MatMut<'_, T>| {
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
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
            let par = gemm_par(ab.nrows(), bb.nrows(), ab.ncols());
            matmul(dst.rb_mut(), Accum::Add, ab, bb.adjoint(), T::one(), par);
        }
    };
    let labels = space.bond_labels(k);
    let (tasks, work) = fold_load(a, b, &labels);
    if fork_blocks(tasks, work) {
        return fold_fresh(space, k, &labels, &a.row_ranks, &b.row_ranks, fold_one);
    }
    let mut c = Su2Sketch::<T, S>::zeros_sym(space, k, &a.row_ranks, &b.row_ranks);
    for &l in &labels {
        let Some(dst) = c.block_mut(l) else {
            continue;
        };
        fold_one(l, dst);
    }
    c
}

/// `rproject(A, Q) -> Su2Sketch @ site(A)` — the θ²-weighted dual pairing: `block(C,l) = Σ_fam
/// weight_f · A[fam,l] · adjoint(Q[fam,l])` over the four admissible families out of row sector
/// `l`, with `weight_f ∈ {1, (l.two_s+2)/(l.two_s+1), l.two_s/(l.two_s+1), 1}` for `{Empty, Up, Dn,
/// Occ}` — EXACTLY `theta(l.two_s, l.two_s±1)²`. Reconstructs `A ≈ C·Q` from a θ-right-canonical
/// `Q` (`Σ_r θ²·Q·Qᴴ =
/// I`, e.g. the `Q` returned by [`Su2Core::cq`]) and makes the ALS/MALS local effective Hamiltonian
/// self-adjoint in the θ-weighted core inner product. **A plain adjoint contraction
/// ([`core_times_adjoint_core`], weight ≡ 1) is a SILENT O(1) error** whenever `Q` carries
/// `up`/`dn` transition rank — the dedicated anti-oracle in the `factorization_rounding` binary (O1) pins this.
pub fn rproject<T: Scalar, S: Symmetry>(a: &Su2Core<T, S>, q: &Su2Core<T, S>) -> Su2Sketch<T, S> {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(k, q.k, "rproject: site mismatch");
    debug_assert!(
        a.col_ranks.eq_grid(&q.col_ranks),
        "rproject: col-rank mismatch"
    );
    let g = space.site_ctx(k);
    let fold_one = |l: S::Label, mut dst: MatMut<'_, T>| {
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
                continue;
            }
            let (Some(ab), Some(qb)) = (a.block(f, l), q.block(f, l)) else {
                continue;
            };
            let weight = S::theta_sq(l, t);
            let par = gemm_par(ab.nrows(), qb.nrows(), ab.ncols());
            matmul(
                dst.rb_mut(),
                Accum::Add,
                ab,
                qb.adjoint(),
                T::from_f64(weight),
                par,
            );
        }
    };
    let labels = space.bond_labels(k);
    let (tasks, work) = fold_load(a, q, &labels);
    if fork_blocks(tasks, work) {
        return fold_fresh(space, k, &labels, &a.row_ranks, &q.row_ranks, fold_one);
    }
    let mut c = Su2Sketch::<T, S>::zeros_sym(space, k, &a.row_ranks, &q.row_ranks);
    for &l in &labels {
        let Some(dst) = c.block_mut(l) else {
            continue;
        };
        fold_one(l, dst);
    }
    c
}

#[inline]
fn fork_blocks(tasks: usize, work: usize) -> bool {
    tasks >= knobs::COREOPS_BLOCKS.get()
        && work >= knobs::COREOPS_WORK.get()
        && work / tasks.max(1) >= knobs::COREOPS_TASK_WORK.get()
        && rayon::current_num_threads() > 1
}

fn fold_load<T: Scalar, S: Symmetry>(
    a: &Su2Core<T, S>,
    b: &Su2Core<T, S>,
    labels: &[S::Label],
) -> (usize, usize) {
    let (space, k) = (a.space, a.k);
    let g = space.site_ctx(k);
    let (mut tasks, mut work) = (0, 0);
    for &l in labels {
        if a.row_ranks[l] == 0 || b.row_ranks[l] == 0 {
            continue;
        }
        tasks += 1;
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
                continue;
            }
            if let (Some(ab), Some(bb)) = (a.block(f, l), b.block(f, l)) {
                work += ab.nrows() * bb.nrows() * ab.ncols();
            }
        }
    }
    (tasks, work)
}

fn fold_fresh<T: Scalar, S: Symmetry>(
    space: S::Space,
    k: usize,
    labels: &[S::Label],
    ra: &S::Grid<usize>,
    rb: &S::Grid<usize>,
    fold_one: impl Fn(S::Label, MatMut<'_, T>) + Sync,
) -> Su2Sketch<T, S> {
    let tasks: Vec<S::Label> = labels
        .iter()
        .copied()
        .filter(|&l| ra[l] > 0 && rb[l] > 0)
        .collect();
    let blocks: Vec<Mat<T>> = tasks
        .par_iter()
        .map(|&l| {
            let mut m = Mat::<T>::zeros(ra[l], rb[l]);
            fold_one(l, m.as_mut());
            m
        })
        .collect();
    let mut grid = <S::Grid<Option<Mat<T>>>>::from_fn(&space, |_| None);
    for &l in labels {
        if ra[l] == 0 || rb[l] == 0 {
            grid[l] = Some(Mat::<T>::zeros(ra[l], rb[l]));
        }
    }
    for (l, m) in tasks.into_iter().zip(blocks) {
        grid[l] = Some(m);
    }
    Su2Sketch {
        space,
        k,
        blocks: grid,
    }
}

/// `rproject`'s ANTI-ORACLE baseline: weights ALL transitions `1` — the
/// naive `A * adjoint(Q)` a careless implementation would write. Structurally identical to
/// [`core_times_adjoint_core`] (same computation, kept as a SEPARATELY NAMED entry point so the
/// O1 anti-oracle test reads as "the wrong implementation of `rproject`", not "an unrelated
/// function"). MUST FAIL the `A ≈ C·Q` reconstruction whenever `Q` carries `up`/`dn` transition
/// rank at a `two_s > 0` row sector (`θ² ≠ 1` there) — see the `factorization_rounding` binary O1.
pub fn rproject_plain<T: Scalar, S: Symmetry>(
    a: &Su2Core<T, S>,
    q: &Su2Core<T, S>,
) -> Su2Sketch<T, S> {
    core_times_adjoint_core(a, q)
}

/// Core-producing `Sketch × Core`: result at `site(b)` with `(row_ranks(a), col_ranks(b))`; `a`'s
/// block folds into the row side of each family of `b`. UNWEIGHTED (Sketch folds carry no θ).
/// Mirror of the abelian `mul_sketch_left` (`core/factor/`).
pub fn mul_sketch_left<T: Scalar, S: Symmetry>(
    a: &Su2Sketch<T, S>,
    b: &Su2Core<T, S>,
) -> Su2Core<T, S> {
    let space = b.space;
    let k = b.k;
    debug_assert_eq!(a.k, k, "mul_sketch_left: bond mismatch");
    let g = space.site_ctx(k);
    let row_ranks = a.row_ranks();
    let mut tasks: Vec<(S::Step, S::Label)> = Vec::new();
    let mut work = 0;
    for l in b.row_qn.iter().copied() {
        let Some(ablk) = a.block(l) else { continue };
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
                continue;
            }
            if let Some(bblk) = b.block(f, l) {
                tasks.push((f, l));
                work += ablk.nrows() * ablk.ncols() * bblk.ncols();
            }
        }
    }
    if fork_blocks(tasks.len(), work) {
        let blocks: Vec<Mat<T>> = tasks
            .par_iter()
            .map(|&(f, l)| {
                let (ablk, bblk) = (a.block(l).unwrap(), b.block(f, l).unwrap());
                let mut m = Mat::<T>::zeros(ablk.nrows(), bblk.ncols());
                let par = gemm_par(ablk.nrows(), bblk.ncols(), ablk.ncols());
                matmul(m.as_mut(), Accum::Replace, ablk, bblk, T::one(), par);
                m
            })
            .collect();
        return core_from_tasks::<T, S>(space, k, &row_ranks, &b.col_ranks, &tasks, blocks);
    }
    let mut out = Su2Core::<T, S>::zeros_sym(space, k, &row_ranks, &b.col_ranks);
    for &l in &b.row_qn {
        let Some(ablk) = a.block(l) else { continue };
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
                continue;
            }
            let Some(bblk) = b.block(f, l) else { continue };
            let mut dst = out.block_mut(f, l).unwrap();
            let par = gemm_par(ablk.nrows(), bblk.ncols(), ablk.ncols());
            matmul(dst.rb_mut(), Accum::Replace, ablk, bblk, T::one(), par);
        }
    }
    out
}

fn core_from_tasks<T: Scalar, S: Symmetry>(
    space: S::Space,
    k: usize,
    row_ranks: &S::Grid<usize>,
    col_ranks: &S::Grid<usize>,
    tasks: &[(S::Step, S::Label)],
    blocks: Vec<Mat<T>>,
) -> Su2Core<T, S> {
    let entries = tasks
        .iter()
        .zip(blocks)
        .map(|(&(f, l), m)| (f, l, m))
        .collect();
    Su2Core::<T, S>::from_entries_sym(space, k, row_ranks, col_ranks, entries)
}

/// Core-producing `Core × Sketch`: result at `site(a)` with `(row_ranks(a), col_ranks(c))`, `c`'s
/// block selected by the FORWARD target column sector `f.apply(l, g)`. UNWEIGHTED. Mirror of the
/// abelian `mul_sketch_right` (`core/factor/`).
pub fn mul_sketch_right<T: Scalar, S: Symmetry>(
    a: &Su2Core<T, S>,
    c: &Su2Sketch<T, S>,
) -> Su2Core<T, S> {
    mul_sketch_right_with(None, a, c)
}

#[allow(dead_code)]
pub(crate) fn mul_sketch_right_in<T: Scalar, S: Symmetry>(
    tables: &LabelTables<S>,
    a: &Su2Core<T, S>,
    c: &Su2Sketch<T, S>,
) -> Su2Core<T, S> {
    mul_sketch_right_with(Some(tables), a, c)
}

#[cfg(test)]
thread_local! {
    static MSR_FORKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn mul_sketch_right_with<T: Scalar, S: Symmetry>(
    tables: Option<&LabelTables<S>>,
    a: &Su2Core<T, S>,
    c: &Su2Sketch<T, S>,
) -> Su2Core<T, S> {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(c.k, k + 1, "mul_sketch_right: bond mismatch");
    let tables = tables.filter(|t| *t.space() == space);
    let col_ranks = match tables {
        Some(tb) => c.col_ranks_in(tb),
        None => c.col_ranks(),
    };
    let owned: Vec<S::Label>;
    let rows: &[S::Label] = match tables {
        Some(tb) => tb.storage_labels(k),
        None => {
            owned = storage_labels::<S>(&space, k);
            &owned
        }
    };
    let mut slots: Vec<Slot<S>> = Vec::new();
    for_each_nonempty_slot::<S>(&space, k, rows, &a.row_ranks, &col_ranks, |f, l, t| {
        slots.push((f, l, t))
    });
    let operands = |f: S::Step, l: S::Label, t: S::Label| match (a.block(f, l), c.block(t)) {
        (Some(ablk), Some(cblk)) if ablk.nrows() * ablk.ncols() * cblk.ncols() > 0 => {
            Some((ablk, cblk))
        }
        _ => None,
    };
    let (mut n_tasks, mut work) = (0usize, 0usize);
    for &(f, l, t) in &slots {
        if let Some((ablk, cblk)) = operands(f, l, t) {
            n_tasks += 1;
            work += ablk.nrows() * ablk.ncols() * cblk.ncols();
        }
    }
    let mul = |ablk: MatRef<'_, T>, cblk: MatRef<'_, T>| -> Mat<T> {
        let mut m = Mat::<T>::zeros(ablk.nrows(), cblk.ncols());
        let par = gemm_par(ablk.nrows(), cblk.ncols(), ablk.ncols());
        matmul(m.as_mut(), Accum::Replace, ablk, cblk, T::one(), par);
        m
    };
    let assemble = move |fill: &mut dyn FnMut(S::Step, S::Label, S::Label) -> Option<Mat<T>>| {
        let (rows, cols) = (Cow::Borrowed(&a.row_ranks), Cow::Owned(col_ranks));
        match tables {
            Some(tb) => Su2Core::<T, S>::from_fn_nonempty_in_cow(tb, k, rows, cols, fill),
            None => Su2Core::<T, S>::from_fn_nonempty_sym_cow(space, k, rows, cols, fill),
        }
    };
    if !fork_blocks(n_tasks, work) {
        return assemble(&mut |f, l, t| operands(f, l, t).map(|(ablk, cblk)| mul(ablk, cblk)));
    }
    #[cfg(test)]
    MSR_FORKS.with(|n| n.set(n.get() + 1));
    let tasks: Vec<u32> = (0..slots.len() as u32)
        .filter(|&i| {
            let (f, l, t) = slots[i as usize];
            operands(f, l, t).is_some()
        })
        .collect();
    let blocks: Vec<Mat<T>> = tasks
        .par_iter()
        .map(|&i| {
            let (f, l, t) = slots[i as usize];
            let (ablk, cblk) = operands(f, l, t).unwrap();
            mul(ablk, cblk)
        })
        .collect();
    let mut done = tasks.iter().zip(blocks).peekable();
    let mut i = 0u32;
    assemble(&mut |_, _, _| {
        let hit = done.peek().is_some_and(|(ti, _)| **ti == i);
        i += 1;
        hit.then(|| done.next().unwrap().1)
    })
}

/// Accumulating form of [`mul_sketch_left`]: `out += a · b`: needed for the multi-summand
/// `FᴸSₖFᴿ = Σₛ Fᴸ[s]·SₖFᴿ[s]` accumulation — mirror of the abelian
/// `crate::core::factor::mul_sketch_left_into`. UNWEIGHTED, same as [`mul_sketch_left`].
pub fn mul_sketch_left_into<T: Scalar, S: Symmetry>(
    out: &mut Su2Core<T, S>,
    a: &Su2Sketch<T, S>,
    b: &Su2Core<T, S>,
) {
    let space = b.space;
    let k = b.k;
    debug_assert_eq!(a.k, k, "mul_sketch_left_into: bond mismatch");
    debug_assert_eq!(out.k, k, "mul_sketch_left_into: out bond mismatch");
    let g = space.site_ctx(k);
    for &l in &b.row_qn {
        let Some(ablk) = a.block(l) else { continue };
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
                continue;
            }
            let Some(bblk) = b.block(f, l) else { continue };
            let Some(mut dst) = out.block_mut(f, l) else {
                continue;
            };
            matmul(dst.rb_mut(), Accum::Add, ablk, bblk, T::one(), Par::Seq);
        }
    }
}

/// Accumulating form of [`mul_sketch_right`]: `out += a · c` (twin of [`mul_sketch_left_into`]).
pub fn mul_sketch_right_into<T: Scalar, S: Symmetry>(
    out: &mut Su2Core<T, S>,
    a: &Su2Core<T, S>,
    c: &Su2Sketch<T, S>,
) {
    let space = a.space;
    let k = a.k;
    debug_assert_eq!(c.k, k + 1, "mul_sketch_right_into: bond mismatch");
    debug_assert_eq!(out.k, k, "mul_sketch_right_into: out bond mismatch");
    let g = space.site_ctx(k);
    for &l in &a.row_qn {
        for &f in S::STEPS {
            let Some(t) = f.step_apply(l, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
                continue;
            }
            let (Some(ablk), Some(cblk)) = (a.block(f, l), c.block(t)) else {
                continue;
            };
            let Some(mut dst) = out.block_mut(f, l) else {
                continue;
            };
            matmul(dst.rb_mut(), Accum::Add, ablk, cblk, T::one(), Par::Seq);
        }
    }
}

pub(crate) fn theta_sq(f: Family, l: Su2Sector) -> f64 {
    match f {
        Family::Empty | Family::Occ => 1.0,
        Family::Up => (l.two_s as f64 + 2.0) / (l.two_s as f64 + 1.0),
        Family::Dn => l.two_s as f64 / (l.two_s as f64 + 1.0),
    }
}

impl<T: Scalar, S: Symmetry> Su2Core<T, S> {
    /// `self ← α·v + self`: per block, all 4 families, guarded by admissibility; asserts matching
    /// row/col ranks with `v`. Distinct from `⊕` (block concatenation) — this is the in-place
    /// elementwise accumulation the rounding sweeps use.
    pub fn axpy(&mut self, alpha: T, v: &Su2Core<T, S>) {
        for &f in S::STEPS {
            for s in self.transitions(f).collect::<Vec<_>>() {
                let vb = v.block(f, s).expect("v transition present");
                debug_assert_eq!(
                    (vb.nrows(), vb.ncols()),
                    self.block(f, s).map(|b| (b.nrows(), b.ncols())).unwrap(),
                    "axpy: block shape mismatch"
                );
                let sb = self.block_mut(f, s).expect("transition present");
                block_axpy(sb, vb, alpha);
            }
        }
    }

    /// `self ← α·v + β·self`.
    pub fn axpby(&mut self, alpha: T, v: &Su2Core<T, S>, beta: T) {
        for &f in S::STEPS {
            for s in self.transitions(f).collect::<Vec<_>>() {
                let vb = v.block(f, s).expect("v transition present");
                let mut sb = self.block_mut(f, s).expect("transition present");
                for j in 0..vb.ncols() {
                    for i in 0..vb.nrows() {
                        let c = sb.rb_mut().get_mut(i, j);
                        *c = *c * beta + alpha * *vb.get(i, j);
                    }
                }
            }
        }
    }

    pub(crate) fn dot_center_par(&self, other: &Su2Core<T, S>) -> T {
        assert_same_blocks(self, other, "dot_center_par");
        if !par::par_vector_ops(self.coeff_hint()) {
            return self.dot_center(other);
        }
        let parts: Vec<T> = (0..S::STEPS.len())
            .into_par_iter()
            .map(|i| step_dot_center(self, other, S::STEPS[i]))
            .collect();
        parts.into_iter().fold(T::zero(), |acc, p| acc + p)
    }

    pub(crate) fn abs2_center_par(&self) -> f64 {
        if !par::par_vector_ops(self.coeff_hint()) {
            return self.abs2_center();
        }
        let parts: Vec<f64> = (0..S::STEPS.len())
            .into_par_iter()
            .map(|i| step_abs2_center(self, S::STEPS[i]))
            .collect();
        parts.into_iter().sum()
    }

    pub(crate) fn norm_center_par(&self) -> f64 {
        self.abs2_center_par().sqrt()
    }

    pub(crate) fn axpy_par(&mut self, alpha: T, v: &Su2Core<T, S>) {
        assert_same_blocks(self, v, "axpy_par");
        if !par::par_vector_ops(self.coeff_hint()) {
            self.axpy(alpha, v);
            return;
        }
        self.store_mut()
            .blocks_mut()
            .par_iter_mut()
            .zip(v.store().blocks().par_iter())
            .for_each(|(sb, ob)| block_axpy(sb.as_mut(), ob.as_ref(), alpha));
    }

    pub(crate) fn scale_mut_par(&mut self, alpha: T) {
        if !par::par_vector_ops(self.coeff_hint()) {
            self.scale_mut(alpha);
            return;
        }
        self.store_mut()
            .blocks_mut()
            .par_iter_mut()
            .for_each(|sb| block_scale(sb.as_mut(), alpha));
    }

    /// Centerweighted core inner product: `Σ_{row sector} Σ_fam centerweight(2S_col, 2S)·dot(fam(V), fam(W))`,
    /// where `2S_col` is the family's target column-sector spin (`○○→l.two_s`, `up→l.two_s+1`,
    /// `dn→l.two_s−1`, `●●→l.two_s`). Reduces to the unweighted sum at bond `d` (sole col sector
    /// `(N,2S)` has `centerweight=1`).
    pub fn dot_center(&self, other: &Su2Core<T, S>) -> T {
        let mut acc = T::zero();
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            for l in self.transitions(f) {
                let t = f.step_apply(l, g).expect("transition admissible");
                let cw = T::from_f64(crate::su2::centerweight(
                    t.two_j(),
                    self.space.total_two_j(),
                ));
                let vb = self.block(f, l).expect("self transition present");
                let wb = other.block(f, l).expect("other transition present");
                acc += cw * block_dot(vb, wb);
            }
        }
        acc
    }

    /// Centerweighted squared norm: same weighted sum as [`Self::dot_center`] with `Σ|·|²` per block; real accumulator.
    pub fn abs2_center(&self) -> f64 {
        let mut acc = 0.0f64;
        let g = self.space.site_ctx(self.k);
        for &f in S::STEPS {
            for l in self.transitions(f) {
                let t = f.step_apply(l, g).expect("transition admissible");
                let cw = crate::su2::centerweight(t.two_j(), self.space.total_two_j());
                let vb = self.block(f, l).expect("self transition present");
                acc += cw * block_abs2(vb);
            }
        }
        acc
    }

    /// `sqrt(abs2_center)`.
    pub fn norm_center(&self) -> f64 {
        self.abs2_center().sqrt()
    }

    /// `self ← A · self`: fold the block-diagonal sketch into the row side of all four families.
    /// Requires `A` square with `row_ranks(A) == col_ranks(A) == row_ranks(self)`. UNWEIGHTED.
    /// Mirror of the abelian `SparseCore::fold_sketch_left_mut` (`core/factor/`).
    pub fn fold_sketch_left_mut(&mut self, a: &Su2Sketch<T, S>) {
        let k = self.k;
        let g = self.space.site_ctx(k);
        for l in self.row_qn.clone() {
            let Some(ablk) = a.block(l) else { continue };
            for &f in S::STEPS {
                let Some(t) = f.step_apply(l, g) else {
                    continue;
                };
                if !self.space.admissible(k + 1, t) {
                    continue;
                }
                let Some(blk) = self.block(f, l) else {
                    continue;
                };
                debug_assert!(
                    ablk.nrows() == blk.nrows() && ablk.ncols() == blk.nrows(),
                    "fold_sketch_left_mut: A block at {l:?} must be square, dims == row_rank(self)"
                );
                let prod = gemm(ablk, blk);
                let mut dst = self.block_mut(f, l).unwrap();
                let (m, n) = (prod.nrows(), prod.ncols());
                for j in 0..n {
                    for i in 0..m {
                        *dst.rb_mut().get_mut(i, j) = *prod.get(i, j);
                    }
                }
            }
        }
    }

    /// `self ← self · B`: fold into the column side, `B`'s block selected by the FORWARD target
    /// column sector `f.apply(l, g)`. UNWEIGHTED. Mirror of the abelian
    /// `SparseCore::fold_sketch_right_mut` (`core/factor/`).
    pub fn fold_sketch_right_mut(&mut self, b: &Su2Sketch<T, S>) {
        let k = self.k;
        let g = self.space.site_ctx(k);
        for l in self.row_qn.clone() {
            for &f in S::STEPS {
                let Some(t) = f.step_apply(l, g) else {
                    continue;
                };
                if !self.space.admissible(k + 1, t) {
                    continue;
                }
                let (Some(blk), Some(bblk)) = (self.block(f, l), b.block(t)) else {
                    continue;
                };
                debug_assert!(
                    bblk.nrows() == blk.ncols() && bblk.ncols() == blk.ncols(),
                    "fold_sketch_right_mut: B block at {t:?} must be square, dims == col_rank(self)"
                );
                let prod = gemm(blk, bblk);
                let mut dst = self.block_mut(f, l).unwrap();
                let (m, n) = (prod.nrows(), prod.ncols());
                for j in 0..n {
                    for i in 0..m {
                        *dst.rb_mut().get_mut(i, j) = *prod.get(i, j);
                    }
                }
            }
        }
    }

    /// Rank-preserving vertical QR, UNWEIGHTED. Mutates `self` in place to hold `Q` (zero-padded to keep col ranks fixed); returns
    /// the upper-triangular `R` as a [`Su2Sketch`] at bond `site+1`. Built from
    /// [`crate::core::factor::thin_q`] + [`gemm`] (no pub `(Q,R)` free fn exists — mirror the
    /// abelian `SparseCore::qr_fixed`, `core/factor/`, exactly).
    pub fn qr_fixed(&mut self) -> Su2Sketch<T, S> {
        let space = self.space;
        let k = self.k;
        let mut r_sketch =
            Su2Sketch::<T, S>::zeros_sym(space, k + 1, &self.col_ranks, &self.col_ranks);
        for r in space.bond_labels(k + 1) {
            let n = self.col_ranks[r];
            let av = self.vertical_unfolding(r);
            let m = av.nrows();
            if m == 0 || n == 0 {
                continue;
            }
            let q = thin_q(av.as_ref());
            let rank = q.ncols();
            let mut rr = Mat::<T>::zeros(rank, n);
            if rank > 0 {
                matmul(
                    rr.as_mut(),
                    Accum::Replace,
                    q.as_ref().adjoint(),
                    av.as_ref(),
                    T::one(),
                    Par::Seq,
                );
            }
            {
                let mut rb = r_sketch.block_mut(r).unwrap();
                for j in 0..n {
                    for i in 0..rank {
                        *rb.rb_mut().get_mut(i, j) = *rr.get(i, j);
                    }
                }
            }
            let mut padded = Mat::<T>::zeros(m, n);
            for j in 0..rank {
                for i in 0..m {
                    *padded.get_mut(i, j) = *q.get(i, j);
                }
            }
            self.scatter_vertical(r, padded.as_ref());
        }
        r_sketch
    }

    /// Rank-preserving horizontal LQ, θ-WEIGHTED. Per row sector:
    /// `scale_horizontal` before the plain-transpose QR construction (matching `my_cq`, NOT
    /// adjoint), `unscale_horizontal` on `Q` write-back; `L` stays RAW. Returns `L` as a
    /// [`Su2Sketch`] at bond `site`. Mirror of the abelian `SparseCore::lq_fixed`
    /// (`core/factor/`) with the θ fold inserted.
    pub fn lq_fixed(&mut self) -> Su2Sketch<T, S> {
        let space = self.space;
        let k = self.k;
        let mut l_sketch = Su2Sketch::<T, S>::zeros_sym(space, k, &self.row_ranks, &self.row_ranks);
        for l in space.bond_labels(k) {
            let m = self.row_ranks[l];
            let mut ah = self.horizontal_unfolding(l);
            self.scale_horizontal(&mut ah, l);
            let width = ah.ncols();
            if m == 0 || width == 0 {
                continue;
            }
            let t = ah.transpose().to_owned();
            let qt = thin_q(t.as_ref());
            let rank = qt.ncols();
            let mut rt = Mat::<T>::zeros(rank, m);
            if rank > 0 {
                matmul(
                    rt.as_mut(),
                    Accum::Replace,
                    qt.as_ref().adjoint(),
                    t.as_ref(),
                    T::one(),
                    Par::Seq,
                );
            }
            {
                let mut lb = l_sketch.block_mut(l).unwrap();
                for j in 0..rank {
                    for i in 0..m {
                        *lb.rb_mut().get_mut(i, j) = *rt.get(j, i);
                    }
                }
            }
            let mut padded = Mat::<T>::zeros(m, width);
            for j in 0..width {
                for i in 0..rank {
                    *padded.get_mut(i, j) = *qt.get(j, i);
                }
            }
            self.unscale_horizontal(&mut padded, l);
            self.scatter_horizontal(l, padded.as_ref());
        }
        l_sketch
    }

    /// Rank-revealing vertical QC, UNWEIGHTED. Per column sector,
    /// [`crate::core::factor::my_qc`] of `vertical_unfolding`. Returns `(Q core, C sketch @ site+1,
    /// reduced col_ranks)`, `A ≈ Q·C`. Mirror of the abelian `SparseCore::qc`
    /// (`core/factor/`).
    pub fn qc(&self) -> (Su2Core<T, S>, Su2Sketch<T, S>, S::Grid<usize>) {
        let space = self.space;
        let k = self.k;
        let mut col_ranks = <S::Grid<usize>>::filled(&space, 0);
        let mut qmats: Vec<(S::Label, Mat<T>)> = Vec::new();
        let mut cblocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);

        for r in space.bond_labels(k + 1) {
            let av = self.vertical_unfolding(r);
            let (q, c, rank) = my_qc(av.as_ref());
            col_ranks[r] = rank;
            qmats.push((r, q));
            cblocks[r] = Some(c);
        }

        let mut q_core = Su2Core::<T, S>::zeros_sym(space, k, &self.row_ranks, &col_ranks);
        for (r, q) in qmats {
            q_core.scatter_vertical(r, q.as_ref());
        }
        let c_sketch = Su2Sketch::<T, S>::new_sym(space, k + 1, cblocks);
        (q_core, c_sketch, col_ranks)
    }

    /// Rank-revealing horizontal CQ, θ-WEIGHTED. Per row sector:
    /// `scale_horizontal` before [`crate::core::factor::my_cq`], `unscale_horizontal` on `Q`
    /// write-back (using the DESTINATION `Q` core's `col_ranks`, as required); `C` stays RAW.
    /// Returns `(Q core, C sketch @ site, reduced row_ranks)`, `A ≈ C·Q`. Mirror of the abelian
    /// `SparseCore::cq` (`core/factor/`).
    pub fn cq(&self) -> (Su2Core<T, S>, Su2Sketch<T, S>, S::Grid<usize>) {
        let space = self.space;
        let k = self.k;
        let mut row_ranks = <S::Grid<usize>>::filled(&space, 0);
        let mut qmats: Vec<(S::Label, Mat<T>)> = Vec::new();
        let mut cblocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);

        for l in space.bond_labels(k) {
            let mut ah = self.horizontal_unfolding(l);
            self.scale_horizontal(&mut ah, l);
            let (q, c, rank) = my_cq(ah.as_ref());
            row_ranks[l] = rank;
            qmats.push((l, q));
            cblocks[l] = Some(c);
        }

        let mut q_core = Su2Core::<T, S>::zeros_sym(space, k, &row_ranks, &self.col_ranks);
        for (l, mut q) in qmats {
            q_core.unscale_horizontal(&mut q, l);
            q_core.scatter_horizontal(l, q.as_ref());
        }
        let c_sketch = Su2Sketch::<T, S>::new_sym(space, k, cblocks);
        (q_core, c_sketch, row_ranks)
    }

    /// Per-row-sector horizontal SVD, θ-WEIGHTED (`svd_horizontal`). Per row sector:
    /// `scale_horizontal` before
    /// [`crate::core::factor::svd_uvt_svals`]; `U` RAW, `S` diagonal sketch (the returned σ are
    /// the WEIGHTED-in-the-bond-metric fixed-Sz Schmidt values — [`crate::su2::chop`] pools them
    /// correctly), `Vt` RAW but still θ-SCALED (consumers unscale on write-back, see round.rs).
    /// Zero-rank row sector: dimensioned empties (`U: m×0`, `S: 0×0`, `Vt: 0×horizontal_width`).
    /// Returns `(U sketch, S sketch, raw Vt grid)` both @ bond `site`. Mirror of the abelian
    /// `SparseCore::svd_horizontal` (`core/factor/`).
    #[allow(clippy::type_complexity)]
    pub fn svd_horizontal(&self) -> (Su2Sketch<T, S>, Su2Sketch<T, S>, BlockGrid<T, S>) {
        let space = self.space;
        let k = self.k;
        let mut ublocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);
        let mut sblocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);
        let mut vtblocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);

        for l in space.bond_labels(k) {
            let m = self.row_ranks[l];
            let width = self.horizontal_width(l);
            let size = m.min(width);
            if size > 0 {
                let mut ah = self.horizontal_unfolding(l);
                self.scale_horizontal(&mut ah, l);
                let (u, svals, vt) = svd_uvt_svals(ah.as_ref());
                let mut s = Mat::<T>::zeros(size, size);
                for (i, &sval) in svals.iter().enumerate().take(size) {
                    *s.get_mut(i, i) = T::from_f64(sval);
                }
                ublocks[l] = Some(u);
                sblocks[l] = Some(s);
                vtblocks[l] = Some(vt);
            } else {
                ublocks[l] = Some(Mat::<T>::zeros(m, 0));
                sblocks[l] = Some(Mat::<T>::zeros(0, 0));
                vtblocks[l] = Some(Mat::<T>::zeros(0, width));
            }
        }
        (
            Su2Sketch::<T, S>::new_sym(space, k, ublocks),
            Su2Sketch::<T, S>::new_sym(space, k, sblocks),
            vtblocks,
        )
    }

    /// Per-column-sector vertical SVD, `√centerweight`-SCALAR-weighted. Per column sector:
    /// `scale_vertical` (the uniform scalar) before `svd_uvt_svals`; `U` is the UNWEIGHTED
    /// left-canonical factor (RAW, scale-invariant); `Vt` RAW but still scaled. Zero-rank column
    /// sector: dimensioned empties (`U: vertical_height×0`, `S: 0×0`, `Vt: 0×n`). Returns `(raw U
    /// grid, S sketch, Vt sketch)` both @ bond `site+1`. Mirror of the abelian
    /// `SparseCore::svd_vertical` (`core/factor/`).
    #[allow(clippy::type_complexity)]
    pub fn svd_vertical(&self) -> (BlockGrid<T, S>, Su2Sketch<T, S>, Su2Sketch<T, S>) {
        let space = self.space;
        let k = self.k;
        let mut ublocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);
        let mut sblocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);
        let mut vtblocks: BlockGrid<T, S> = <BlockGrid<T, S>>::from_fn(&space, |_| None);

        for r in space.bond_labels(k + 1) {
            let n = self.col_ranks[r];
            let height = self.vertical_height(r);
            let size = height.min(n);
            if size > 0 {
                let mut av = self.vertical_unfolding(r);
                self.scale_vertical(&mut av, r);
                let (u, svals, vt) = svd_uvt_svals(av.as_ref());
                let mut s = Mat::<T>::zeros(size, size);
                for (i, &sval) in svals.iter().enumerate().take(size) {
                    *s.get_mut(i, i) = T::from_f64(sval);
                }
                ublocks[r] = Some(u);
                sblocks[r] = Some(s);
                vtblocks[r] = Some(vt);
            } else {
                ublocks[r] = Some(Mat::<T>::zeros(height, 0));
                sblocks[r] = Some(Mat::<T>::zeros(0, 0));
                vtblocks[r] = Some(Mat::<T>::zeros(0, n));
            }
        }
        (
            ublocks,
            Su2Sketch::<T, S>::new_sym(space, k + 1, sblocks),
            Su2Sketch::<T, S>::new_sym(space, k + 1, vtblocks),
        )
    }

    /// [`Self::svd_horizontal`] packaged back into a core, used by `svdvals`. Builds `C`
    /// with `row_ranks = col_ranks(S)`, `col_ranks = self.col_ranks`; each row sector's `Vt` rows
    /// are `unscale_horizontal`'d HERE on write-back (the θ hook lands in this function, not
    /// `svd_horizontal` itself). Returns `(U sketch, S sketch, core carrying raw right-canonical
    /// Vt)`. Mirror of the abelian `SparseCore::svd_horizontal_to_core` (`core/factor/`).
    pub fn svd_horizontal_to_core(&self) -> (Su2Sketch<T, S>, Su2Sketch<T, S>, Su2Core<T, S>) {
        let space = self.space;
        let k = self.k;
        let (u, s, mut vt) = self.svd_horizontal();
        let mut c = Su2Core::<T, S>::zeros_sym(space, k, &s.col_ranks(), &self.col_ranks);
        for l in space.bond_labels(k) {
            if let Some(mut vtb) = vt[l].take() {
                c.unscale_horizontal(&mut vtb, l);
                c.scatter_horizontal(l, vtb.as_ref());
            }
        }
        (u, s, c)
    }

    /// New core with the top-left sub-blocks of `self` truncated to `row_ranks ≤` / `col_ranks ≤`
    /// the current ranks, used by `round_global_mut`. UNWEIGHTED copy over all 4 families. Asserts `0
    /// ≤ new ≤ current`. Mirror of the abelian `SparseCore::reduce_ranks` (`core/factor/`).
    pub fn reduce_ranks(
        &self,
        row_ranks: &S::Grid<usize>,
        col_ranks: &S::Grid<usize>,
    ) -> Su2Core<T, S> {
        let space = self.space;
        let k = self.k;
        for s in space.bond_labels(k) {
            debug_assert!(row_ranks[s] <= self.row_ranks[s], "reduce_ranks: row grows");
        }
        for s in space.bond_labels(k + 1) {
            debug_assert!(col_ranks[s] <= self.col_ranks[s], "reduce_ranks: col grows");
        }
        let g = space.site_ctx(k);
        let mut b = Su2Core::<T, S>::zeros_sym(space, k, row_ranks, col_ranks);
        for l in space.bond_labels(k) {
            for &f in S::STEPS {
                let Some(t) = f.step_apply(l, g) else {
                    continue;
                };
                if !space.admissible(k + 1, t) {
                    continue;
                }
                let (rr, cc) = (row_ranks[l], col_ranks[t]);
                let (Some(src), Some(mut dst)) = (self.block(f, l), b.block_mut(f, l)) else {
                    continue;
                };
                for j in 0..cc {
                    for i in 0..rr {
                        *dst.rb_mut().get_mut(i, j) = *src.get(i, j);
                    }
                }
            }
        }
        b
    }
}

#[cfg(test)]
mod par_tests;
#[cfg(test)]
mod ref_tests;
