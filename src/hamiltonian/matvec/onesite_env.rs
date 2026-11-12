// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::sync::atomic::Ordering;

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, Par, unzip, zip};
use rayon::prelude::*;

use super::cell::ColShard;
use super::onesite::OneSiteWorkspace;
use super::scheduling::{
    BLOCK_SPLIT_EVENTS, BLOCK_SPLIT_TARGET, LOCAL_MATVEC_APPLIES, block_fork, block_split,
    env_partials_bytes, est_cost,
};
use crate::core::factor::Sketch;
use crate::core::{Family, SparseCore};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::instrument;
use crate::qn::Sector;
use crate::scalar::Scalar;
use crate::vector::TtVector;

impl<T: Scalar> SparseHamiltonian<T> {
    /// Davidson local operator `(Fᴸ·H·Xₖ)·Fᴿ`, streamed per mid QN-sector `q`. This IS the abelian
    /// one-site matvec closure `op` passed to [`crate::solvers::davidson::davidson`].
    pub fn sketched_hamiltonian(
        &self,
        x: &SparseCore<T>,
        fl: &Sketch<T>,
        fr: &Sketch<T>,
        ws: &OneSiteWorkspace,
    ) -> SparseCore<T> {
        LOCAL_MATVEC_APPLIES.fetch_add(1, Ordering::Relaxed);
        let space = self.space;
        let k = x.k;
        let _t = instrument::tic();
        let mut out = SparseCore::zeros(space, k, &fl.row_ranks(), &fr.col_ranks());
        instrument::toc(instrument::slot::OUT_ALLOC, _t);

        let fl_rows = fl.row_ranks();
        let results: Vec<(Family, Sector, Mat<T>)> = if block_fork() {
            let _tl = instrument::tic();
            let sectors_k = space.bond_sectors(k);
            let mut tasks: Vec<(Sector, Family, Sector, usize, u32, u32)> = Vec::new();
            for &q in &out.col_qn {
                let rq = ws.j_tab[q].last().map_or(0, |r| r.end);
                if rq == 0 || fr.block(q).is_none() {
                    continue;
                }
                for fam in Family::ALL {
                    let Some(l) = fam.unapply(q, space.pg.irrep_of(k)) else {
                        continue;
                    };
                    if !space.contains(k, l) {
                        continue;
                    }
                    tasks.push((q, fam, l, rq, 0, ws.j_tab[q].len() as u32));
                }
            }
            if block_split() {
                let costs: Vec<u64> = tasks
                    .iter()
                    .map(|&(_, fam, l, _, _, _)| est_cost(self, k, &fl_rows, x, &sectors_k, fam, l))
                    .collect();
                let total: u64 = costs.iter().sum();
                let target = (total / BLOCK_SPLIT_TARGET).max(1);
                let mut split: Vec<(Sector, Family, Sector, usize, u32, u32)> =
                    Vec::with_capacity(tasks.len());
                for (t, &c) in tasks.iter().zip(&costs) {
                    let n = c.div_ceil(target).clamp(1, 16) as usize;
                    let jt = &ws.j_tab[t.0];
                    if n == 1 || jt.len() < 2 {
                        split.push(*t);
                        continue;
                    }
                    BLOCK_SPLIT_EVENTS.fetch_add(1, Ordering::Relaxed);
                    let total_w = t.3;
                    let mut o0 = 0u32;
                    for sh in 1..=n {
                        let want = total_w * sh / n;
                        let mut o1 = o0;
                        while (o1 as usize) < jt.len() && jt[o1 as usize].end <= want {
                            o1 += 1;
                        }
                        if sh == n {
                            o1 = jt.len() as u32;
                        }
                        if o1 > o0 {
                            split.push((t.0, t.1, t.2, t.3, o0, o1));
                            o0 = o1;
                        }
                    }
                    if (o0 as usize) < jt.len() {
                        split.push((t.0, t.1, t.2, t.3, o0, jt.len() as u32));
                    }
                }
                tasks = split;
            }
            instrument::toc(instrument::slot::TASKLIST, _tl);
            let _tp = instrument::tic();
            let region = instrument::Region::begin();
            let collected: Vec<(Family, Sector, Mat<T>)> = tasks
                .into_par_iter()
                .map(|(q, fam, l, _rq, o0, o1)| {
                    let _tt = instrument::tic();
                    let rows = fl_rows[l];
                    let jt = &ws.j_tab[q];
                    let off = jt[o0 as usize].start;
                    let width = jt[o1 as usize - 1].end - off;
                    let _tz = instrument::tic();
                    let mut blk = Mat::<T>::zeros(rows, width);
                    instrument::toc(instrument::slot::BLK_ZERO, _tz);
                    let _tm = instrument::tic();
                    self.sketch_h_mul_block_shard_into(
                        blk.as_mut(),
                        fl,
                        x,
                        fam,
                        l,
                        &sectors_k,
                        &ws.i_tab,
                        &ws.j_tab,
                        ColShard::new(o0, o1, off),
                    );
                    instrument::toc(instrument::slot::MUL, _tm);
                    let _tg = instrument::tic();
                    let fq = fr
                        .block(q)
                        .expect("sketched_hamiltonian: fr block vanished after filtering");
                    let mut ob = Mat::<T>::zeros(rows, fq.ncols());
                    matmul(
                        ob.as_mut(),
                        Accum::Replace,
                        blk.as_ref(),
                        fq.submatrix(off, 0, width, fq.ncols()),
                        T::one(),
                        Par::Seq,
                    );
                    instrument::toc(instrument::slot::GEMM, _tg);
                    region.task(_tt, rows * width);
                    (fam, l, ob)
                })
                .collect();
            region.end();
            instrument::toc(instrument::slot::PARALLEL, _tp);
            collected
        } else {
            out.col_qn
                .clone()
                .into_par_iter()
                .flat_map_iter(|q| {
                    let mut local: Vec<(Family, Sector, Mat<T>)> = Vec::new();
                    let rq = ws.j_tab[q].last().map_or(0, |r| r.end);
                    if rq == 0 {
                        return local.into_iter();
                    }
                    let lband = self.left_band(fl, x, q, rq, &ws.i_tab, &ws.j_tab);
                    let fq = match fr.block(q) {
                        Some(b) => b,
                        None => return local.into_iter(),
                    };
                    for fam in Family::ALL {
                        let Some(l) = fam.unapply(q, space.pg.irrep_of(k)) else {
                            continue;
                        };
                        if !space.contains(k, l) {
                            continue;
                        }
                        if let Some(lb) = lband.block(fam, l) {
                            let mut ob = Mat::<T>::zeros(lb.nrows(), fq.ncols());
                            matmul(ob.as_mut(), Accum::Replace, lb, fq, T::one(), Par::Seq);
                            local.push((fam, l, ob));
                        }
                    }
                    local.into_iter()
                })
                .collect()
        };
        let _tm = instrument::tic();
        for (fam, l, m) in results {
            let mut ob = out
                .block_mut(fam, l)
                .expect("sketched_hamiltonian: computed block has no output slot");
            zip!(ob.rb_mut(), m.as_ref()).for_each(|unzip!(d, s)| *d += *s);
        }
        instrument::toc(instrument::slot::MERGE, _tm);
        out
    }

    /// Left-environment update `C = adjoint(Xₖ)·(Fᴸ·H·Xₖ)` at bond `k+1` — the ALS forward-sweep
    /// and `left_to_right_sketching` environment build. `k = site(Fᴸ)`, `Xₖ = x.core(k)`.
    pub fn sketching_step_right(
        &self,
        x: &TtVector<T>,
        fl: &Sketch<T>,
        ws: &OneSiteWorkspace,
    ) -> Sketch<T> {
        let xk = x.core(fl.site());
        self.sketching_step_right_mixed(xk, xk, fl, ws)
    }

    /// [`Self::sketching_step_right`] with DISTINCT bra and ket cores:
    /// `C = adjoint(Bₖ)·(Fᴸ·H·Kₖ)` at bond `k+1`, `k = site(Fᴸ)`.
    pub fn sketching_step_right_mixed(
        &self,
        bra_k: &SparseCore<T>,
        ket_k: &SparseCore<T>,
        fl: &Sketch<T>,
        ws: &OneSiteWorkspace,
    ) -> Sketch<T> {
        let space = self.space;
        let k = fl.site();
        debug_assert_eq!(
            bra_k.k, k,
            "sketching_step_right_mixed: bra core site mismatch"
        );
        debug_assert_eq!(
            ket_k.k, k,
            "sketching_step_right_mixed: ket core site mismatch"
        );
        let ch = self.col_ranks(ket_k);
        let mut c = Sketch::zeros(space, k + 1, &bra_k.col_ranks, &ch);

        if block_fork() {
            let sectors_k = space.bond_sectors(k);
            let fl_rows = fl.row_ranks();
            let mut tasks: Vec<(Sector, Family, Sector, usize)> = Vec::new();
            for &q in &space.bond_sectors(k + 1) {
                let rq = ws.j_tab[q].last().map_or(0, |r| r.end);
                if rq == 0 || bra_k.col_ranks[q] == 0 || ch[q] == 0 {
                    continue;
                }
                for fam in Family::ALL {
                    let Some(l) = fam.unapply(q, space.pg.irrep_of(k)) else {
                        continue;
                    };
                    if !space.contains(k, l) {
                        continue;
                    }
                    if bra_k.block(fam, l).is_none() {
                        continue;
                    }
                    tasks.push((q, fam, l, rq));
                }
            }
            let budget = env_partials_bytes();
            let p_bytes = |q: Sector| crate::mem::block_bytes::<T>(bra_k.col_ranks[q], ch[q]);
            let mut start = 0usize;
            while start < tasks.len() {
                let mut end = start;
                let mut bytes = 0u64;
                while end < tasks.len() {
                    let next = p_bytes(tasks[end].0);
                    if end > start && budget != 0 && bytes + next > budget {
                        break;
                    }
                    bytes += next;
                    end += 1;
                }
                let partials: Vec<(Sector, Mat<T>)> = tasks[start..end]
                    .par_iter()
                    .map(|&(q, fam, l, rq)| {
                        let mut blk = Mat::<T>::zeros(fl_rows[l], rq);
                        self.sketch_h_mul_block_into(
                            blk.as_mut(),
                            fl,
                            ket_k,
                            fam,
                            l,
                            &sectors_k,
                            &ws.i_tab,
                            &ws.j_tab,
                        );
                        let xb = bra_k
                            .block(fam, l)
                            .expect("sketching_step_right: bra block vanished after filtering");
                        let mut p = Mat::<T>::zeros(bra_k.col_ranks[q], ch[q]);
                        matmul(
                            p.as_mut(),
                            Accum::Replace,
                            xb.adjoint(),
                            blk.as_ref(),
                            T::one(),
                            Par::Seq,
                        );
                        (q, p)
                    })
                    .collect();
                if instrument::ENABLED {
                    instrument::mem_hw(instrument::memslot::ENV_PARTIALS, bytes);
                }
                for (q, p) in partials {
                    let mut cq = c
                        .block_mut(q)
                        .expect("sketching_step: computed block has no output slot");
                    zip!(cq.rb_mut(), p.as_ref()).for_each(|unzip!(d, s)| *d += *s);
                }
                start = end;
            }
            return c;
        }

        let results: Vec<(Sector, Mat<T>)> = space
            .bond_sectors(k + 1)
            .into_par_iter()
            .filter_map(|q| {
                let rq = ws.j_tab[q].last().map_or(0, |r| r.end);
                if rq == 0 {
                    return None;
                }
                let (rows, cols) = (bra_k.col_ranks[q], ch[q]);
                if rows == 0 || cols == 0 {
                    return None;
                }
                let lband = self.left_band(fl, ket_k, q, rq, &ws.i_tab, &ws.j_tab);
                let mut cq = Mat::<T>::zeros(rows, cols);
                let mut any = false;
                for fam in Family::ALL {
                    let Some(l) = fam.unapply(q, space.pg.irrep_of(k)) else {
                        continue;
                    };
                    if !space.contains(k, l) {
                        continue;
                    }
                    let (Some(xb), Some(lb)) = (bra_k.block(fam, l), lband.block(fam, l)) else {
                        continue;
                    };
                    matmul(
                        cq.as_mut(),
                        Accum::Add,
                        xb.adjoint(),
                        lb,
                        T::one(),
                        Par::Seq,
                    );
                    any = true;
                }
                any.then_some((q, cq))
            })
            .collect();
        for (q, m) in results {
            let mut cq = c
                .block_mut(q)
                .expect("sketching_step: computed block has no output slot");
            cq.copy_from(m.as_ref());
        }
        c
    }

    /// Right-environment update `C = (H·Xₖ·Fᴿ)·adjoint(Xₖ)` at bond `k` — the ALS back-sweep and
    /// `right_to_left_sketching` per-step update. `k = site(Fᴿ) - 1`, `Xₖ = x.core(k)`. The range
    /// tables `ws` cache `row_ranges(Xₖ)`/`col_ranges(Xₖ)` and are direction-agnostic, so the
    /// back-sweep passes the one-site `ws` it already holds for `Xₖ` here.
    pub fn sketching_step_left(
        &self,
        x: &TtVector<T>,
        fr: &Sketch<T>,
        ws: &OneSiteWorkspace,
    ) -> Sketch<T> {
        let xk = x.core(fr.site() - 1);
        self.sketching_step_left_mixed(xk, xk, fr, ws)
    }

    /// [`Self::sketching_step_left`] with DISTINCT bra and ket cores:
    /// `C = (H·Kₖ·Fᴿ)·adjoint(Bₖ)` at bond `k`, `k = site(Fᴿ) - 1`. Mirror of
    /// [`Self::sketching_step_right_mixed`] — same contract, including that `ws` must be the
    /// **ket's** `onesite_workspace`.
    pub fn sketching_step_left_mixed(
        &self,
        bra_k: &SparseCore<T>,
        ket_k: &SparseCore<T>,
        fr: &Sketch<T>,
        ws: &OneSiteWorkspace,
    ) -> Sketch<T> {
        let space = self.space;
        let k = fr.site() - 1;
        debug_assert_eq!(
            bra_k.k, k,
            "sketching_step_left_mixed: bra core site mismatch"
        );
        debug_assert_eq!(
            ket_k.k, k,
            "sketching_step_left_mixed: ket core site mismatch"
        );
        let i_tab = &ws.i_tab;
        let j_tab = &ws.j_tab;
        let rh = self.row_ranks(ket_k);
        let mut c = Sketch::zeros(space, k, &rh, &bra_k.row_ranks);

        if block_fork() {
            let sectors_k = space.bond_sectors(k);
            let mut tasks: Vec<(Sector, Family, usize)> = Vec::new();
            for &q in &space.bond_sectors(k) {
                let rq = i_tab[q].last().map_or(0, |r| r.end);
                if rq == 0 || rh[q] == 0 || bra_k.row_ranks[q] == 0 {
                    continue;
                }
                for fam in Family::ALL {
                    let r = fam.apply(q, space.pg.irrep_of(k));
                    if !space.contains(k + 1, r) {
                        continue;
                    }
                    if bra_k.block(fam, q).is_none() {
                        continue;
                    }
                    tasks.push((q, fam, rq));
                }
            }
            let region = instrument::Region::begin();
            let budget = env_partials_bytes();
            let fr_cols = fr.col_ranks();
            let p_bytes = |q: Sector| crate::mem::block_bytes::<T>(rh[q], bra_k.row_ranks[q]);
            let mut start = 0usize;
            while start < tasks.len() {
                let mut end = start;
                let mut bytes = 0u64;
                while end < tasks.len() {
                    let next = p_bytes(tasks[end].0);
                    if end > start && budget != 0 && bytes + next > budget {
                        break;
                    }
                    bytes += next;
                    end += 1;
                }
                let partials: Vec<(Sector, Mat<T>)> = tasks[start..end]
                    .par_iter()
                    .map(|&(q, fam, rq)| {
                        let _tt = instrument::tic();
                        let cols = fr_cols[fam.apply(q, space.pg.irrep_of(k))];
                        let _tz = instrument::tic();
                        let mut blk = Mat::<T>::zeros(rq, cols);
                        instrument::toc(instrument::slot::BLK_ZERO, _tz);
                        let _tm = instrument::tic();
                        self.h_sketch_mul_block_into(
                            blk.as_mut(),
                            ket_k,
                            fr,
                            fam,
                            q,
                            &sectors_k,
                            i_tab,
                            j_tab,
                        );
                        instrument::toc(instrument::slot::MUL, _tm);
                        let _tg = instrument::tic();
                        let xb = bra_k
                            .block(fam, q)
                            .expect("sketching_step_left: bra block vanished after filtering");
                        let mut p = Mat::<T>::zeros(rh[q], bra_k.row_ranks[q]);
                        matmul(
                            p.as_mut(),
                            Accum::Replace,
                            blk.as_ref(),
                            xb.adjoint(),
                            T::one(),
                            Par::Seq,
                        );
                        instrument::toc(instrument::slot::GEMM, _tg);
                        region.task(_tt, rq * cols);
                        (q, p)
                    })
                    .collect();
                if instrument::ENABLED {
                    instrument::mem_hw(instrument::memslot::ENV_PARTIALS, bytes);
                }
                for (q, p) in partials {
                    let mut cq = c
                        .block_mut(q)
                        .expect("sketching_step: computed block has no output slot");
                    zip!(cq.rb_mut(), p.as_ref()).for_each(|unzip!(d, s)| *d += *s);
                }
                start = end;
            }
            region.end();
            return c;
        }

        let results: Vec<(Sector, Mat<T>)> = space
            .bond_sectors(k)
            .into_par_iter()
            .filter_map(|q| {
                let rq = i_tab[q].last().map_or(0, |r| r.end);
                if rq == 0 {
                    return None;
                }
                let (rows, cols) = (rh[q], bra_k.row_ranks[q]);
                if rows == 0 || cols == 0 {
                    return None;
                }
                let rband = self.right_band(ket_k, fr, q, rq, i_tab, j_tab);
                let mut cq = Mat::<T>::zeros(rows, cols);
                let mut any = false;
                for fam in Family::ALL {
                    let r = fam.apply(q, space.pg.irrep_of(k));
                    if !space.contains(k + 1, r) {
                        continue;
                    }
                    let (Some(rb), Some(xb)) = (rband.block(fam, q), bra_k.block(fam, q)) else {
                        continue;
                    };
                    matmul(
                        cq.as_mut(),
                        Accum::Add,
                        rb,
                        xb.adjoint(),
                        T::one(),
                        Par::Seq,
                    );
                    any = true;
                }
                any.then_some((q, cq))
            })
            .collect();
        for (q, m) in results {
            let mut cq = c
                .block_mut(q)
                .expect("sketching_step: computed block has no output slot");
            cq.copy_from(m.as_ref());
        }
        c
    }

    /// Exact diagonal of `H_eff = (Fᴸ·H·Xₖ)·Fᴿ` in the coefficient basis — the Davidson
    /// preconditioner diagonal. The matvec restricted to identical in/out coefficients factorizes
    /// through the operator bond: for output occupation `occ` from row sector `l` into col sector
    /// `q = l + occ_shift`, `D_occ[l] = FLdiag_l · Mᵒᶜᶜ_l · transpose(FRdiag_q)`, where `M =
    /// coo[k].occ[l].occ[l]` is the on-site SELF coupling. Uses the PLAIN transpose of `FRdiag`
    /// (NOT adjoint) — a deliberate, metric-agnostic choice, load-bearing for complex `T`.
    pub fn effective_diagonal(
        &self,
        fl: &Sketch<T>,
        fr: &Sketch<T>,
        x: &SparseCore<T>,
        ws: &OneSiteWorkspace,
    ) -> SparseCore<T> {
        let space = self.space;
        let k = x.k;
        let mut dcore = SparseCore::zeros(space, k, &x.row_ranks, &x.col_ranks);
        let i_tab = &ws.i_tab;
        let j_tab = &ws.j_tab;
        let ch = self.col_ranks(x);

        let tasks: Vec<(Family, Sector)> = Family::ALL
            .into_iter()
            .flat_map(|occ| x.row_qn.iter().map(move |&l| (occ, l)))
            .collect();
        let results: Vec<(Family, Sector, Mat<T>)> = tasks
            .into_par_iter()
            .filter_map(|(occ, l)| {
                let q = occ.apply(l, space.pg.irrep_of(k));
                if !space.contains(k + 1, q) {
                    return None;
                }
                let r_l = x.row_ranks[l];
                let r_q = x.col_ranks[q];
                if !(r_l > 0 && r_q > 0 && ch[q] > 0) {
                    return None;
                }
                let il = &i_tab[l];
                let jl = &j_tab[q];
                let (n_i, n_j) = (il.len(), jl.len());
                if n_i == 0 || n_j == 0 {
                    return None;
                }
                let fl_blk = fl.block(l)?;
                let fr_blk = fr.block(q)?;

                let mut fl_diag = Mat::<T>::zeros(r_l, n_i);
                for (i, rng) in il.iter().enumerate() {
                    if rng.len() != r_l {
                        continue;
                    }
                    for a in 0..r_l {
                        *fl_diag.get_mut(a, i) = *fl_blk.get(a, rng.start + a);
                    }
                }
                let mut fr_diag = Mat::<T>::zeros(r_q, n_j);
                for (j, rng) in jl.iter().enumerate() {
                    if rng.len() != r_q {
                        continue;
                    }
                    for b in 0..r_q {
                        *fr_diag.get_mut(b, j) = *fr_blk.get(rng.start + b, b);
                    }
                }

                let cell = self.self_cell(k, occ, l);
                let mut g = Mat::<T>::zeros(r_l, n_j);
                for (it, jt, vt) in cell.entries() {
                    let (it, jt) = (it as usize, jt as usize);
                    for a in 0..r_l {
                        *g.get_mut(a, jt) += vt * *fl_diag.get(a, it);
                    }
                }
                let mut db = Mat::<T>::zeros(r_l, r_q);
                matmul(
                    db.as_mut(),
                    Accum::Replace,
                    g.as_ref(),
                    fr_diag.as_ref().transpose(),
                    T::one(),
                    Par::Seq,
                );
                Some((occ, l, db))
            })
            .collect();
        for (occ, l, m) in results {
            let mut db = dcore
                .block_mut(occ, l)
                .expect("effective_diagonal: occ block admissible");
            db.copy_from(m.as_ref());
        }
        dcore
    }
}

#[cfg(test)]
mod phase5_tests {
    use crate::coeff::CoeffTensor;
    use crate::core::SparseCore;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::hamiltonian::matvec::onesite_workspace;
    use crate::qn::QnSpace;
    use crate::scalar::Scalar;
    use crate::solvers::InnerProductVector;
    use crate::solvers::als::{
        boundary_idsketch, left_to_right_sketching, right_to_left_sketching,
    };
    use crate::solvers::davidson::effective_diagonal_bruteforce;
    use crate::test_util::randomized_state;
    use crate::{C64, Family, TtVector};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn complex_hermitian_t() -> [(usize, usize, C64); 2] {
        [
            (1, 3, -C64::from_polar(1.0, 0.9)),
            (3, 1, -C64::from_polar(1.0, -0.9)),
        ]
    }

    fn assert_coeffs_close(a: &SparseCore<f64>, b: &SparseCore<f64>, tol: f64, msg: &str) {
        let n = <SparseCore<f64> as InnerProductVector>::num_coeffs(a);
        assert_eq!(n, <SparseCore<f64> as InnerProductVector>::num_coeffs(b));
        let mut diff2 = 0.0f64;
        let mut bref2 = 0.0f64;
        for i in 0..n {
            let av = <SparseCore<f64> as InnerProductVector>::coeff(a, i);
            let bv = <SparseCore<f64> as InnerProductVector>::coeff(b, i);
            diff2 += (av - bv) * (av - bv);
            bref2 += bv * bv;
        }
        let bound = tol * bref2.sqrt().max(1.0);
        assert!(
            diff2.sqrt() <= bound,
            "{msg}: diff {} vs bound {}",
            diff2.sqrt(),
            bound
        );
    }

    fn assert_coeffs_close_c64(a: &SparseCore<C64>, b: &SparseCore<C64>, tol: f64, msg: &str) {
        let n = <SparseCore<C64> as InnerProductVector>::num_coeffs(a);
        assert_eq!(n, <SparseCore<C64> as InnerProductVector>::num_coeffs(b));
        let mut diff2 = 0.0f64;
        let mut bref2 = 0.0f64;
        for i in 0..n {
            let av = <SparseCore<C64> as InnerProductVector>::coeff(a, i);
            let bv = <SparseCore<C64> as InnerProductVector>::coeff(b, i);
            let d = av - bv;
            diff2 += d.abs_s() * d.abs_s();
            bref2 += bv.abs_s() * bv.abs_s();
        }
        let bound = tol * bref2.sqrt().max(1.0);
        assert!(
            diff2.sqrt() <= bound,
            "{msg}: diff {} vs bound {}",
            diff2.sqrt(),
            bound
        );
    }

    #[test]
    fn t1_sketched_hamiltonian_threaded_matches_serial() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], -1.0);
        v.set(&[1, 3, 0, 2], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A02);
        let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);
        let k = d / 2;
        x.move_core(k, true);

        let fr = right_to_left_sketching(&h, &x);
        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let ws = onesite_workspace(&h, x.core(k));

        let run = |nt: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap()
                .install(|| h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws))
        };
        let y1 = run(1);
        let y8 = run(8);

        let mut maxdiff = 0.0f64;
        for fam in Family::ALL {
            for s in space.bond_sectors(k) {
                match (y1.block(fam, s), y8.block(fam, s)) {
                    (Some(a), Some(b)) => {
                        assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
                        for j in 0..a.ncols() {
                            for i in 0..a.nrows() {
                                maxdiff = maxdiff.max((*a.get(i, j) - *b.get(i, j)).abs());
                            }
                        }
                    }
                    (None, None) => {}
                    _ => panic!("threaded vs serial block presence mismatch at {fam:?} {s:?}"),
                }
            }
        }
        assert!(
            maxdiff < 1e-12,
            "threaded vs serial sketched_hamiltonian max block diff {maxdiff:e}"
        );
    }

    #[test]
    fn t1_sketched_hamiltonian_ws_reuse_linearity() {
        let _g = crate::par::lock_par_switches();
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[0, 1], -1.0);
        t.set(&[1, 0], -1.0);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 4, 1], -1.0);
        v.set(&[1, 4, 0, 2], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A03);
        let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);
        let fr = right_to_left_sketching(&h, &x);
        let fl = left_to_right_sketching(&h, &x);
        let k = 2usize;

        let ws = onesite_workspace(&h, x.core(k));
        let xa = x.core(k).clone();
        let mut xb = xa.clone();
        xb.scale_mut(-0.7);

        let ya1 = h.sketched_hamiltonian(&xa, &fl[k], &fr[k], &ws);
        let yb = h.sketched_hamiltonian(&xb, &fl[k], &fr[k], &ws);
        let ya2 = h.sketched_hamiltonian(&xa, &fl[k], &fr[k], &ws);

        assert!(ya1.norm() > 0.0, "vacuous linearity test (‖y(Xa)‖ == 0)");

        let mut lin = ya1.clone();
        lin.scale_mut(-0.7);
        let mut dlin = yb.clone();
        dlin.axpy(-1.0, &lin);
        assert!(
            dlin.norm() <= 1e-13 * ya1.norm().max(1.0),
            "ws-reuse linearity broken: {}",
            dlin.norm()
        );

        let mut didem = ya2.clone();
        didem.axpy(-1.0, &ya1);
        assert!(
            didem.norm() == 0.0,
            "ws reuse not idempotent (stale state?): {}",
            didem.norm()
        );
    }

    #[test]
    fn t2_effective_diagonal_matches_bruteforce_real() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for i in 0..d {
            t.set(&[i, i], 1.0 + 0.3 * i as f64);
        }
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], -1.0);
        v.set(&[1, 3, 0, 2], -1.0);
        v.set(&[0, 0, 3, 3], 0.5);
        v.set(&[3, 3, 0, 0], 0.5);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A07);
        let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);

        for &kk in &[0usize, 2, 4] {
            let mut x2 = x.clone();
            if kk > 0 {
                x2.move_core(kk, true);
            }
            x2.check();
            let fl = if kk == 0 {
                boundary_idsketch(&x2, 0, 1)
            } else {
                left_to_right_sketching(&h, &x2)
                    .into_iter()
                    .nth(kk)
                    .unwrap()
            };
            let fr = right_to_left_sketching(&h, &x2);
            let ws = onesite_workspace(&h, x2.core(kk));
            let xk = x2.core(kk).clone();

            let mut op = |a: &SparseCore<f64>| h.sketched_hamiltonian(a, &fl, &fr[kk], &ws);
            let d_diag = h.effective_diagonal(&fl, &fr[kk], &xk, &ws);
            let o_diag = effective_diagonal_bruteforce(&mut op, &xk);
            assert!(
                o_diag.norm() > 1e-6,
                "vacuous effective_diagonal at kk={kk} (‖O‖={})",
                o_diag.norm()
            );
            assert_coeffs_close(
                &d_diag,
                &o_diag,
                1e-10,
                &format!("effective_diagonal at kk={kk}"),
            );
        }
    }

    #[test]
    fn t2_effective_diagonal_matches_bruteforce_complex() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<C64>::zeros(d, 2);
        for &(i, j, c) in &complex_hermitian_t() {
            t.set(&[i, j], c);
        }
        for i in 0..d {
            t.set(&[i, i], C64::new(1.0 + 0.3 * i as f64, 0.0));
        }
        let v = CoeffTensor::<C64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5A08);
        let mut x: TtVector<C64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);

        for &kk in &[0usize, 2, 4] {
            let mut x2 = x.clone();
            if kk > 0 {
                x2.move_core(kk, true);
            }
            x2.check();
            let fl = if kk == 0 {
                boundary_idsketch(&x2, 0, 1)
            } else {
                left_to_right_sketching(&h, &x2)
                    .into_iter()
                    .nth(kk)
                    .unwrap()
            };
            let fr = right_to_left_sketching(&h, &x2);
            let ws = onesite_workspace(&h, x2.core(kk));
            let xk = x2.core(kk).clone();

            let mut op = |a: &SparseCore<C64>| h.sketched_hamiltonian(a, &fl, &fr[kk], &ws);
            let d_diag = h.effective_diagonal(&fl, &fr[kk], &xk, &ws);
            let o_diag = effective_diagonal_bruteforce(&mut op, &xk);
            assert!(
                o_diag.norm() > 1e-6,
                "vacuous complex effective_diagonal at kk={kk} (‖O‖={})",
                o_diag.norm()
            );
            assert_coeffs_close_c64(
                &d_diag,
                &o_diag,
                1e-10,
                &format!("effective_diagonal (complex) at kk={kk}"),
            );
        }
    }
}
