// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, Par};
use rayon::prelude::*;

use super::scheduling::{
    BLOCK_SPLIT_EVENTS, BLOCK_SPLIT_TARGET, LOCAL_MATVEC_APPLIES, block_split, est_cost,
    est_cost_right,
};
use crate::core::factor::Sketch;
use crate::core::twosite::{ContractedSparseCores, contract_mid_block};
use crate::core::{Family, SparseCore};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::instrument;
use crate::mem::Footprint as _;
use crate::qn::{RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

pub(crate) static TWO_SITE_STREAM_BYTES: AtomicU64 = AtomicU64::new(u64::MAX);

pub(crate) static SPLIT_MIN_WIDTH: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(8);
const TWO_SITE_STREAM_BYTES_DEFAULT: u64 = 16 << 30;

/// The current wave budget (see [`TWO_SITE_STREAM_BYTES`]).
pub fn two_site_stream_bytes() -> u64 {
    use std::sync::LazyLock;
    static FROM_ENV: LazyLock<u64> = LazyLock::new(|| {
        std::env::var("QNTT_TWO_SITE_STREAM_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(TWO_SITE_STREAM_BYTES_DEFAULT)
    });
    let v = TWO_SITE_STREAM_BYTES.load(Ordering::Relaxed);
    if v == u64::MAX { *FROM_ENV } else { v }
}

/// Set the wave budget (tests and instruments); returns the previous value. `u64::MAX` restores
/// the env/default resolution.
pub fn set_two_site_stream_bytes(bytes: u64) -> u64 {
    TWO_SITE_STREAM_BYTES.swap(bytes, Ordering::Relaxed)
}

impl<T: Scalar> SparseHamiltonian<T> {
    /// Two-site Davidson local operator `(Fᴸ·H·Xₖ)·(H·Xₖ₊₁·Fᴿ)`, contracted through
    /// `factor_qc(cp)`. `fl` @ bond `k`, `cp` @ site `k` (spanning `k,k+1`), `fr` @ bond `k+2`.
    /// This IS the abelian two-site matvec closure `op` passed to
    /// [`crate::solvers::davidson::davidson`] by the MALS driver.
    pub fn two_site_sketched_hamiltonian(
        &self,
        cp: &ContractedSparseCores<T>,
        fl: &Sketch<T>,
        fr: &Sketch<T>,
    ) -> ContractedSparseCores<T> {
        LOCAL_MATVEC_APPLIES.fetch_add(1, Ordering::Relaxed);
        debug_assert_eq!(fl.site(), cp.site(), "two_site matvec: Fᴸ @ bond k");
        debug_assert_eq!(fr.site(), cp.site() + 2, "two_site matvec: Fᴿ @ bond k+2");
        let (xk, xk1) = crate::core::twosite::factor_qc(cp);
        let il = self.row_ranges(&xk);
        let jr = self.col_ranges(&xk1);
        self.two_site_apply_streamed(&xk, &xk1, fl, fr, &il, &jr)
    }

    pub(crate) fn two_site_sketched_hamiltonian_cached(
        &self,
        cp: &ContractedSparseCores<T>,
        fl: &Sketch<T>,
        fr: &Sketch<T>,
        il: &SectorGrid<Vec<Range<usize>>>,
        jr: &SectorGrid<Vec<Range<usize>>>,
    ) -> ContractedSparseCores<T> {
        LOCAL_MATVEC_APPLIES.fetch_add(1, Ordering::Relaxed);
        debug_assert_eq!(fl.site(), cp.site(), "two_site matvec: Fᴸ @ bond k");
        debug_assert_eq!(fr.site(), cp.site() + 2, "two_site matvec: Fᴿ @ bond k+2");
        let (xk, xk1) = crate::core::twosite::factor_qc(cp);
        self.two_site_apply_streamed(&xk, &xk1, fl, fr, il, jr)
    }

    pub(crate) fn two_site_apply_streamed(
        &self,
        xk: &SparseCore<T>,
        xk1: &SparseCore<T>,
        fl: &Sketch<T>,
        fr: &Sketch<T>,
        il: &SectorGrid<Vec<Range<usize>>>,
        jr: &SectorGrid<Vec<Range<usize>>>,
    ) -> ContractedSparseCores<T> {
        let space = self.space;
        let k = xk.k;
        debug_assert_eq!(xk1.k, k + 1);
        let gl = space.pg.irrep_of(k);
        let gr = space.pg.irrep_of(k + 1);
        let j_tab_l = self.col_ranges(xk);
        let i_tab_r = self.row_ranges(xk1);
        let l_rows: RankGrid = fl.row_ranks();
        let l_cols: RankGrid = self.col_ranks(xk);
        let r_rows: RankGrid = self.row_ranks(xk1);
        let r_cols: RankGrid = fr.col_ranks();
        let sectors_k = space.bond_sectors(k);
        let sectors_k1 = space.bond_sectors(k + 1);
        let sz = std::mem::size_of::<T>() as u64;

        let mut out = ContractedSparseCores::zeros(space, k, &l_rows, &r_cols);

        #[derive(Clone, Copy)]
        enum Side {
            L,
            R,
        }
        struct Task {
            side: Side,
            fam: Family,
            sec: Sector,
            rows: usize,
            cols: usize,
            cuts: Vec<Range<usize>>,
        }
        let mut per_mid: Vec<(Sector, Vec<Task>, u64)> = xk
            .col_qn
            .iter()
            .filter(|&&m| out.block(m).is_some_and(|b| b.nrows() > 0 && b.ncols() > 0))
            .map(|&m| {
                let mut tasks = Vec::with_capacity(8);
                let mut bytes = 0u64;
                for fi in Family::ALL {
                    let Some(l) = fi.unapply(m, gl) else {
                        continue;
                    };
                    if !space.contains(k, l) {
                        continue;
                    }
                    let (rows, cols) = (l_rows[l], l_cols[m]);
                    if rows > 0 && cols > 0 {
                        bytes += rows as u64 * cols as u64 * sz;
                        tasks.push(Task {
                            side: Side::L,
                            fam: fi,
                            sec: l,
                            rows,
                            cols,
                            cuts: Vec::new(),
                        });
                    }
                }
                for fj in Family::ALL {
                    let r = fj.apply(m, gr);
                    if !space.contains(k + 2, r) {
                        continue;
                    }
                    let (rows, cols) = (r_rows[m], r_cols[r]);
                    if rows > 0 && cols > 0 {
                        bytes += rows as u64 * cols as u64 * sz;
                        tasks.push(Task {
                            side: Side::R,
                            fam: fj,
                            sec: m,
                            rows,
                            cols,
                            cuts: Vec::new(),
                        });
                    }
                }
                (m, tasks, bytes)
            })
            .collect();

        if block_split() {
            let min_w = SPLIT_MIN_WIDTH.load(Ordering::Relaxed).max(2);
            let costs: Vec<Vec<u64>> = per_mid
                .iter()
                .map(|(_, tasks, _)| {
                    tasks
                        .iter()
                        .map(|t| match t.side {
                            Side::L => est_cost(self, k, &l_rows, xk, &sectors_k, t.fam, t.sec),
                            Side::R => {
                                est_cost_right(self, k + 1, t.cols, xk1, &sectors_k1, t.fam, t.sec)
                            }
                        })
                        .collect()
                })
                .collect();
            let total: u64 = costs.iter().flatten().sum();
            let target = (total / BLOCK_SPLIT_TARGET).max(1);
            for ((_, tasks, _), cs) in per_mid.iter_mut().zip(&costs) {
                for (t, &c) in tasks.iter_mut().zip(cs) {
                    if !matches!(t.side, Side::R) {
                        continue;
                    }
                    let n = (c.div_ceil(target).clamp(1, 16) as usize).min(t.cols / min_w);
                    if n > 1 {
                        t.cuts = (0..n)
                            .map(|s| t.cols * s / n..t.cols * (s + 1) / n)
                            .collect();
                        BLOCK_SPLIT_EVENTS.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        let budget = two_site_stream_bytes();
        let mut waves: Vec<Vec<usize>> = Vec::new();
        let mut cur: Vec<usize> = Vec::new();
        let mut cur_bytes = 0u64;
        for (i, (_, _, b)) in per_mid.iter().enumerate() {
            if !cur.is_empty() && budget > 0 && cur_bytes + b > budget {
                waves.push(std::mem::take(&mut cur));
                cur_bytes = 0;
            }
            cur.push(i);
            cur_bytes += b;
        }
        if !cur.is_empty() {
            waves.push(cur);
        }

        for wave in waves {
            let _t = instrument::tic();
            let refs: Vec<&Task> = wave.iter().flat_map(|&i| per_mid[i].1.iter()).collect();
            let blocks: Vec<Mat<T>> = refs
                .par_iter()
                .map(|t| {
                    let mut blk = Mat::<T>::zeros(t.rows, t.cols);
                    if t.cuts.is_empty() {
                        match t.side {
                            Side::L => self.sketch_h_mul_block_into(
                                blk.as_mut(),
                                fl,
                                xk,
                                t.fam,
                                t.sec,
                                &sectors_k,
                                il,
                                &j_tab_l,
                            ),
                            Side::R => self.h_sketch_mul_block_into(
                                blk.as_mut(),
                                xk1,
                                fr,
                                t.fam,
                                t.sec,
                                &sectors_k1,
                                &i_tab_r,
                                jr,
                            ),
                        }
                    } else {
                        let mut views = Vec::with_capacity(t.cuts.len());
                        let mut rest = blk.as_mut();
                        let mut base = 0usize;
                        for cut in &t.cuts {
                            let (head, tail) = rest.split_at_col_mut(cut.end - base);
                            views.push((head, cut.clone()));
                            rest = tail;
                            base = cut.end;
                        }
                        views.into_par_iter().for_each(|(v, cut)| {
                            self.h_sketch_mul_block_cols_into(
                                v,
                                xk1,
                                fr,
                                t.fam,
                                t.sec,
                                &sectors_k1,
                                &i_tab_r,
                                jr,
                                cut,
                            )
                        });
                    }
                    blk
                })
                .collect();
            instrument::toc(instrument::slot::TS_LEFT, _t);
            if instrument::ENABLED {
                let wave_bytes: u64 = wave.iter().map(|&i| per_mid[i].2).sum();
                instrument::mem_hw(
                    instrument::memslot::APPLY,
                    wave_bytes + xk.bytes() + xk1.bytes(),
                );
            }
            let _t = instrument::tic();
            let mut offs = Vec::with_capacity(wave.len());
            let mut o = 0usize;
            for &i in &wave {
                offs.push(o);
                o += per_mid[i].1.len();
            }
            let results: Vec<(Sector, Mat<T>)> = wave
                .par_iter()
                .zip(offs.par_iter())
                .map(|(&i, &o)| {
                    let (m, tasks, _) = &per_mid[i];
                    let b = out.block(*m).expect("filtered above");
                    let mut local = Mat::<T>::zeros(b.nrows(), b.ncols());
                    let find = |side: Side, fam: Family, sec: Sector| {
                        tasks
                            .iter()
                            .position(|t| {
                                matches!((t.side, side), (Side::L, Side::L) | (Side::R, Side::R))
                                    && t.fam == fam
                                    && t.sec == sec
                            })
                            .map(|j| blocks[o + j].as_ref())
                    };
                    contract_mid_block(
                        space,
                        k,
                        *m,
                        &l_rows,
                        &r_cols,
                        &|fi, l| find(Side::L, fi, l),
                        &|fj, mm| find(Side::R, fj, mm),
                        local.as_mut(),
                    );
                    (*m, local)
                })
                .collect();
            for (m, local) in results {
                out.block_mut(m)
                    .expect("filtered above")
                    .copy_from(local.as_ref());
            }
            instrument::toc(instrument::slot::TS_RIGHT, _t);
        }
        out
    }

    /// Exact diagonal of `H_eff = (Fᴸ·H_k·H_{k+1}·Fᴿ)` on `cp`, in the coefficient basis — the MALS Davidson preconditioner diagonal.
    /// Factorizes through the mid operator bond `γ`; structurally the one-site
    /// [`Self::effective_diagonal`] run twice and joined. Uses the
    /// PLAIN transpose of the site-`(k+1)` gather (NOT adjoint) — load-bearing for complex `T`,
    /// same metric-agnostic choice as the one-site kernel.
    pub fn two_site_effective_diagonal(
        &self,
        fl: &Sketch<T>,
        fr: &Sketch<T>,
        cp: &ContractedSparseCores<T>,
    ) -> ContractedSparseCores<T> {
        let space = self.space;
        let k = cp.k;
        let (xk, xk1) = crate::core::twosite::factor_qc(cp);
        let ik = self.row_ranges(&xk);
        let jk1 = self.col_ranges(&xk1);
        let jk = self.col_ranges(&xk);
        let ik1 = self.row_ranges(&xk1);
        let mut dcore = cp.zeros_like();

        let results: Vec<(Sector, Mat<T>)> = cp
            .mid_qn
            .par_iter()
            .filter_map(|&m| {
                let (nr, nc) = {
                    let b = dcore.block(m)?;
                    (b.nrows(), b.ncols())
                };
                if nr == 0 || nc == 0 {
                    return None;
                }
                let mut local = Mat::<T>::zeros(nr, nc);
                let mut blk = local.as_mut();
                let n_gamma = jk[m].len().min(ik1[m].len());

                let gr_cols: Vec<(usize, Option<Mat<T>>)> = Family::ALL
                    .into_iter()
                    .map(|fj| {
                        let r = fj.apply(m, space.pg.irrep_of(k + 1));
                        let r_r = if space.contains(k + 2, r) {
                            cp.col_ranks[r]
                        } else {
                            0
                        };
                        if r_r == 0 {
                            return (0, None);
                        }
                        let jr = &jk1[r];
                        let n_jk1 = jr.len();
                        let mut fr_diag = Mat::<T>::zeros(r_r, n_jk1);
                        if let Some(fr_blk) = fr.block(r) {
                            for (j, rng) in jr.iter().enumerate() {
                                if rng.len() != r_r {
                                    continue;
                                }
                                for b in 0..r_r {
                                    *fr_diag.get_mut(b, j) = *fr_blk.get(rng.start + b, b);
                                }
                            }
                        }
                        let cell2 = self.self_cell(k + 1, fj, m);
                        let mut gr = Mat::<T>::zeros(r_r, n_gamma);
                        for (it2, jt2, vt2) in cell2.entries() {
                            let (it2, jt2) = (it2 as usize, jt2 as usize);
                            if it2 < n_gamma && jt2 < n_jk1 {
                                for b in 0..r_r {
                                    *gr.get_mut(b, it2) += vt2 * *fr_diag.get(b, jt2);
                                }
                            }
                        }
                        (r_r, Some(gr))
                    })
                    .collect();

                let mut roff = 0usize;
                for fi in Family::ALL {
                    let Some(l) = fi.unapply(m, space.pg.irrep_of(k)) else {
                        continue;
                    };
                    let r_l = if space.contains(k, l) {
                        cp.row_ranks[l]
                    } else {
                        0
                    };
                    if r_l == 0 {
                        continue;
                    }
                    let il = &ik[l];
                    let n_ik = il.len();
                    let mut fl_diag = Mat::<T>::zeros(r_l, n_ik);
                    if let Some(fl_blk) = fl.block(l) {
                        for (i, rng) in il.iter().enumerate() {
                            if rng.len() != r_l {
                                continue;
                            }
                            for a in 0..r_l {
                                *fl_diag.get_mut(a, i) = *fl_blk.get(a, rng.start + a);
                            }
                        }
                    }
                    let cell = self.self_cell(k, fi, l);
                    let mut gl = Mat::<T>::zeros(r_l, n_gamma);
                    for (it, jt, vt) in cell.entries() {
                        let (it, jt) = (it as usize, jt as usize);
                        if it < n_ik && jt < n_gamma {
                            for a in 0..r_l {
                                *gl.get_mut(a, jt) += vt * *fl_diag.get(a, it);
                            }
                        }
                    }

                    let mut coff = 0usize;
                    for (r_r, gr_opt) in &gr_cols {
                        let r_r = *r_r;
                        if let Some(gr) = gr_opt {
                            let dst = blk.rb_mut().submatrix_mut(roff, coff, r_l, r_r);
                            matmul(
                                dst,
                                Accum::Replace,
                                gl.as_ref(),
                                gr.as_ref().transpose(),
                                T::one(),
                                Par::Seq,
                            );
                        }
                        coff += r_r;
                    }
                    roff += r_l;
                }
                Some((m, local))
            })
            .collect();
        for (m, mres) in results {
            let mut blk = dcore
                .block_mut(m)
                .expect("two_site_effective_diagonal: mid block admissible");
            blk.copy_from(mres.as_ref());
        }
        dcore
    }
}

#[cfg(test)]
mod cached_oracle_tests;

#[cfg(test)]
mod phase7_tests {
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::scalar::Scalar;
    use crate::solvers::als::{
        boundary_idsketch, left_to_right_sketching, right_to_left_sketching,
    };
    use crate::solvers::davidson::effective_diagonal_bruteforce;
    use crate::solvers::mals::{ContractedSparseCores, contract};
    use crate::test_util::randomized_state;
    use crate::{C64, TtVector};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn assert_contracted_coeffs_close(
        a: &ContractedSparseCores<f64>,
        b: &ContractedSparseCores<f64>,
        tol: f64,
        msg: &str,
    ) {
        let n = a.num_coeffs();
        assert_eq!(n, b.num_coeffs(), "{msg}: coeff-count mismatch");
        let mut diff2 = 0.0f64;
        let mut bref2 = 0.0f64;
        for i in 0..n {
            let av = a.coeff(i);
            let bv = b.coeff(i);
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

    fn assert_contracted_coeffs_close_c64(
        a: &ContractedSparseCores<C64>,
        b: &ContractedSparseCores<C64>,
        tol: f64,
        msg: &str,
    ) {
        let n = a.num_coeffs();
        assert_eq!(n, b.num_coeffs(), "{msg}: coeff-count mismatch");
        let mut diff2 = 0.0f64;
        let mut bref2 = 0.0f64;
        for i in 0..n {
            let d = a.coeff(i) - b.coeff(i);
            diff2 += d.abs_s() * d.abs_s();
            bref2 += b.coeff(i).abs_s() * b.coeff(i).abs_s();
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
    fn t7a_twosite_effective_diagonal() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for i in 0..d {
            t.set(&[i, i], -0.5 - 0.1 * i as f64);
        }
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], -1.0);
        v.set(&[1, 3, 0, 2], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A50);
        let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);

        for &kk in &[0usize, 3] {
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
            let fr_all = right_to_left_sketching(&h, &x2);
            let fr = &fr_all[kk + 1];

            let cp = contract(x2.core(kk), x2.core(kk + 1));
            let mut op =
                |c: &ContractedSparseCores<f64>| h.two_site_sketched_hamiltonian(c, &fl, fr);
            let d2 = h.two_site_effective_diagonal(&fl, fr, &cp);
            let o2 = effective_diagonal_bruteforce(&mut op, &cp);

            assert!(o2.norm() > 1e-6, "vacuous two-site diagonal at kk={kk}");
            assert_contracted_coeffs_close(
                &d2,
                &o2,
                1e-10,
                &format!("two-site effective_diagonal at kk={kk}"),
            );
        }
        let _ = x.check();
    }

    #[test]
    fn t7a_twosite_effective_diagonal_complex() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<C64>::zeros(d, 2);
        for i in 0..d {
            t.set(&[i, i], C64::new(-0.5 - 0.1 * i as f64, 0.0));
        }
        let z = C64::from_polar(1.0, 0.7);
        let mut v = CoeffTensor::<C64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], z);
        v.set(&[1, 3, 0, 2], z.conj_s());
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7A5C);
        let mut x: TtVector<C64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);

        for &kk in &[0usize, 3] {
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
            let fr_all = right_to_left_sketching(&h, &x2);
            let fr = &fr_all[kk + 1];

            let cp = contract(x2.core(kk), x2.core(kk + 1));
            let mut op =
                |c: &ContractedSparseCores<C64>| h.two_site_sketched_hamiltonian(c, &fl, fr);
            let d2 = h.two_site_effective_diagonal(&fl, fr, &cp);
            let o2 = effective_diagonal_bruteforce(&mut op, &cp);

            assert!(
                o2.norm() > 1e-6,
                "vacuous complex two-site diagonal at kk={kk}"
            );
            assert_contracted_coeffs_close_c64(
                &d2,
                &o2,
                1e-10,
                &format!("complex two-site effective_diagonal at kk={kk}"),
            );
        }
        let _ = x.check();
    }
}

#[cfg(test)]
mod threaded_tests {
    use crate::TtVector;
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::solvers::mals::contract;
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn two_site_sketched_hamiltonian_threaded_matches_serial() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], -1.0);
        v.set(&[1, 3, 0, 2], -1.0);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x2A02_B1AC);
        let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
        x.right_orthogonalize(true);
        let k = 2usize;
        x.move_core(k, true);
        x.check();

        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let fr_all = right_to_left_sketching(&h, &x);
        let fr = &fr_all[k + 1];
        let cp = contract(x.core(k), x.core(k + 1));

        let run = |nt: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(nt)
                .build()
                .unwrap()
                .install(|| h.two_site_sketched_hamiltonian(&cp, &fl, fr))
        };
        let y1 = run(1);
        let y8 = run(8);

        let mut maxdiff = 0.0f64;
        for &m in &cp.mid_qn {
            match (y1.block(m), y8.block(m)) {
                (Some(a), Some(b)) => {
                    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
                    for j in 0..a.ncols() {
                        for i in 0..a.nrows() {
                            maxdiff = maxdiff.max((*a.get(i, j) - *b.get(i, j)).abs());
                        }
                    }
                }
                (None, None) => {}
                _ => panic!("threaded vs serial block presence mismatch at mid {m:?}"),
            }
        }
        assert!(
            maxdiff < 1e-12,
            "threaded vs serial two_site_sketched_hamiltonian max block diff {maxdiff:e}"
        );
    }
}

#[cfg(test)]
mod l2_streamed_apply_tests {
    use super::{set_two_site_stream_bytes, two_site_stream_bytes};
    use crate::TtVector;
    use crate::coeff::CoeffTensor;
    use crate::core::twosite::{contract, factor_qc};
    use crate::hamiltonian::SparseHamiltonian;
    use crate::hamiltonian::matvec::{h_sketch_mul, sketch_h_mul};
    use crate::qn::QnSpace;
    use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn streamed_apply_is_bit_identical_to_full_cores() {
        let d = 6usize;
        let space = QnSpace::new(d, 2, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    t.set(&[i, j], 0.3 / (1.0 + (i as f64 - j as f64).abs()));
                }
            }
        }
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], -1.0);
        v.set(&[1, 3, 0, 2], -1.0);
        v.set(&[0, 1, 4, 5], 0.5);
        v.set(&[5, 4, 1, 0], 0.5);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x12_0002);
        let mut x: TtVector<f64> = randomized_state(space, 2, 5, &mut rng);
        x.right_orthogonalize(true);
        let k = 2usize;
        x.move_core(k, true);
        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let fr_all = right_to_left_sketching(&h, &x);
        let fr = &fr_all[k + 1];
        let cp = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&cp);
        let reference = contract(&sketch_h_mul(&fl, &h, &xk), &h_sketch_mul(&h, &xk1, fr));
        let il = h.row_ranges(&xk);
        let jr = h.col_ranges(&xk1);
        let check = |label: &str| {
            let got = h.two_site_apply_streamed(&xk, &xk1, &fl, fr, &il, &jr);
            assert_eq!(got.mid_qn, reference.mid_qn);
            for &m in &reference.mid_qn {
                match (reference.block(m), got.block(m)) {
                    (Some(a), Some(b)) => {
                        assert_eq!(
                            (a.nrows(), a.ncols()),
                            (b.nrows(), b.ncols()),
                            "{label} {m:?}"
                        );
                        for j in 0..a.ncols() {
                            for i in 0..a.nrows() {
                                assert!(
                                    a.get(i, j).to_bits() == b.get(i, j).to_bits()
                                        || (*a.get(i, j) == 0.0 && *b.get(i, j) == 0.0),
                                    "{label}: block {m:?} entry ({i},{j}) {} vs {}",
                                    a.get(i, j),
                                    b.get(i, j)
                                );
                            }
                        }
                    }
                    (None, None) => {}
                    _ => panic!("{label}: presence mismatch at {m:?}"),
                }
            }
        };
        let prev = set_two_site_stream_bytes(1);
        assert_eq!(two_site_stream_bytes(), 1);
        for threads in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| check(&format!("budget=1 threads={threads}")));
        }
        set_two_site_stream_bytes(0);
        for threads in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| check(&format!("budget=0 threads={threads}")));
        }
        set_two_site_stream_bytes(prev);
        assert!(reference.mid_qn.len() >= 3);
    }

    #[test]
    fn split_blocks_are_bitwise_the_unsplit_apply() {
        use crate::hamiltonian::matvec::scheduling::{
            block_split_events, reset_block_split_events, set_block_split,
        };
        let _g = crate::par::lock_par_switches();
        let d = 6usize;
        let space = QnSpace::new(d, 2, 1);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for i in 0..d {
            for j in 0..d {
                if i != j {
                    t.set(&[i, j], 0.3 / (1.0 + (i as f64 - j as f64).abs()));
                }
            }
        }
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        v.set(&[2, 0, 3, 1], -1.0);
        v.set(&[1, 3, 0, 2], -1.0);
        v.set(&[0, 1, 4, 5], 0.5);
        v.set(&[5, 4, 1, 0], 0.5);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x12_0003);
        let mut x: TtVector<f64> = randomized_state(space, 2, 5, &mut rng);
        x.right_orthogonalize(true);
        let bits = |c: &crate::core::twosite::ContractedSparseCores<f64>| -> Vec<u64> {
            c.mid_qn
                .iter()
                .flat_map(|&m| {
                    let b = c.block(m).expect("present");
                    (0..b.ncols())
                        .flat_map(move |j| (0..b.nrows()).map(move |i| b.get(i, j).to_bits()))
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let prev = set_block_split(true);
        let prev_w = super::SPLIT_MIN_WIDTH.load(std::sync::atomic::Ordering::Relaxed);
        reset_block_split_events();
        for (min_w, k) in [2usize, 3]
            .into_iter()
            .flat_map(|w| (1..d - 2).map(move |k| (w, k)))
        {
            super::SPLIT_MIN_WIDTH.store(min_w, std::sync::atomic::Ordering::Relaxed);
            let mut xk = x.clone();
            xk.move_core(k, true);
            let fl = left_to_right_sketching(&h, &xk).into_iter().nth(k).unwrap();
            let fr = &right_to_left_sketching(&h, &xk)[k + 1];
            let cp = contract(xk.core(k), xk.core(k + 1));
            for threads in [1usize, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                set_block_split(false);
                let unsplit = pool.install(|| h.two_site_sketched_hamiltonian(&cp, &fl, fr));
                set_block_split(true);
                let split = pool.install(|| h.two_site_sketched_hamiltonian(&cp, &fl, fr));
                assert!(
                    bits(&split) == bits(&unsplit),
                    "pair ({k}, {}) threads {threads} min width {min_w}: split ≠ unsplit",
                    k + 1
                );
            }
        }
        let events = block_split_events();
        set_block_split(prev);
        super::SPLIT_MIN_WIDTH.store(prev_w, std::sync::atomic::Ordering::Relaxed);
        println!("two-site blocks split: {events}");
        assert!(events > 0, "no block was split — the pin is vacuous");
    }
}
