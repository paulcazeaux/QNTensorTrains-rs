// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Mat, Par};
use rayon::prelude::*;

use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid};
use crate::su2::twosite::{Su2ContractedCores, factor_qc};
use crate::su2::vector::Su2TtVector;

use super::PAR_MIN_TASKS;
use super::sketch::compact_band_table;

/// Cached H-extended range tables for repeated one-site matvecs at a fixed site
/// (mirrors the abelian `crate::hamiltonian::matvec::OneSiteWorkspace`; bands are per-call
/// owned `Mat`s, so there is no band buffer and no `Fᴸ` ctor parameter).
pub struct Su2OneSiteWorkspace {
    /// `Su2SparseHamiltonian::row_ranges(H, Xₖ)`.
    pub i_tab: Su2SectorGrid<Vec<Range<usize>>>,
    /// `Su2SparseHamiltonian::col_ranges(H, Xₖ)`.
    pub j_tab: Su2SectorGrid<Vec<Range<usize>>>,
}

/// Build the range tables cached by a site solve (`onesite_workspace` with `Fᴸ` already dropped).
pub fn su2_onesite_workspace<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2Core<T>,
) -> Su2OneSiteWorkspace {
    Su2OneSiteWorkspace {
        i_tab: h.row_ranges(x),
        j_tab: h.col_ranges(x),
    }
}

/// Two-site `factor_qc` result cache keyed on the mid-bond rank grid (HAZARD H4: the SU(2)
/// two-site workspace carries mutable state the abelian one does not — do NOT assume abelian
/// shapes transfer). Fields: `mid_rank` (the rebuild key), `xk`/`xk1` (the two reused factor
/// cores), `il`/`jl`/`ir`/`jr` (the four range tables — consumed by
/// [`Su2SparseHamiltonian::two_site_sketched_hamiltonian_ws`]'s `_into` folds; that reuse IS the
/// workspace's whole service, exactly as for the one-site sibling). No scratch pools, no
/// panelized band engine, no gather panels: bands are per-call owned `Mat`s and faer's pivoted
/// QR owns its own scratch.
pub struct Su2TwoSiteWorkspace<T: Scalar> {
    /// The mid-bond rank grid `factor_qc` last produced — the rebuild key.
    pub mid_rank: Su2RankGrid,
    /// The left factor core from the last `factor_qc`.
    pub xk: Su2Core<T>,
    /// The right factor core from the last `factor_qc`.
    pub xk1: Su2Core<T>,
    /// `row_ranges(H, xk)`.
    pub il: Su2SectorGrid<Vec<Range<usize>>>,
    /// `col_ranges(H, xk)`.
    pub jl: Su2SectorGrid<Vec<Range<usize>>>,
    /// `row_ranges(H, xk1)`.
    pub ir: Su2SectorGrid<Vec<Range<usize>>>,
    /// `col_ranges(H, xk1)`.
    pub jr: Su2SectorGrid<Vec<Range<usize>>>,
}

/// The empty cell the sweep owns. **Ownership contract**: the sweep driver owns the
/// `Option<Su2TwoSiteWorkspace<T>>` as a LOCAL; the two-site
/// `sketched_hamiltonian` takes it `&mut`. NOT stored in the Hamiltonian, NOT `'static`, NOT
/// shared. Serial-only, so no `Sync` burden.
pub fn su2_twosite_workspace<T: Scalar>() -> Option<Su2TwoSiteWorkspace<T>> {
    None
}

/// Lazily creates/rebuilds `ws` from `tc` via [`factor_qc`], only recomputing the range tables when
/// the mid-bond rank grid actually changed. Returns a live `&mut` borrow of the (now populated)
/// workspace.
pub fn factor_qc_into<'w, T: Scalar>(
    ws: &'w mut Option<Su2TwoSiteWorkspace<T>>,
    h: &Su2SparseHamiltonian<T>,
    tc: &Su2ContractedCores<T>,
    fl: &Su2Sketch<T>,
    fr: &Su2Sketch<T>,
) -> &'w mut Su2TwoSiteWorkspace<T> {
    let _ = (fl, fr);
    let (xk, xk1) = factor_qc(tc);
    let mid_rank = xk.col_ranks.clone();

    let stale = ws.as_ref().is_none_or(|w| {
        w.mid_rank != mid_rank || w.xk.row_ranks != xk.row_ranks || w.xk1.col_ranks != xk1.col_ranks
    });
    if stale {
        let il = h.row_ranges(&xk);
        let jl = h.col_ranges(&xk);
        let ir = h.row_ranges(&xk1);
        let jr = h.col_ranges(&xk1);
        *ws = Some(Su2TwoSiteWorkspace {
            mid_rank,
            xk,
            xk1,
            il,
            jl,
            ir,
            jr,
        });
    } else {
        let w = ws.as_mut().expect("not stale ⇒ populated");
        w.xk = xk;
        w.xk1 = xk1;
    }
    ws.as_mut().expect("populated above")
}

impl<T: Scalar> Su2SparseHamiltonian<T> {
    /// Davidson one-site local operator `(Fᴸ·H·Xₖ)·Fᴿ`. Mirrors the abelian `sketched_hamiltonian`
    /// (`src/hamiltonian/matvec/`) — the one-site matvec closure `op` the shared
    /// [`crate::solvers::davidson::davidson`] receives once `TtState`/`InnerProductVector` are
    /// wired ([`crate::su2::layer`]).
    pub fn sketched_hamiltonian(
        &self,
        x: &Su2Core<T>,
        fl: &Su2Sketch<T>,
        fr: &Su2Sketch<T>,
        ws: &Su2OneSiteWorkspace,
    ) -> Su2Core<T> {
        let mut y = Su2Core::zeros(x.space, x.k, &fl.row_ranks(), &self.col_ranks(x));
        self.sketch_h_mul_into(&mut y, fl, x, |_| true, &ws.i_tab, &ws.j_tab);
        crate::su2::coreops::mul_sketch_right(&y, fr)
    }

    /// Left-environment update `C = adjoint(Xₖ)·(Fᴸ·H·Xₖ)` at bond `k+1` (streamed,
    /// no workspace). **UNWEIGHTED in SU2**: reuses the same
    /// unweighted streamed step the one-site env update uses — NO θ, NO θ² here. `k = fl.site()`,
    /// `xk = x.core(k)`.
    pub fn sketching_step_right(
        &self,
        x: &Su2TtVector<T>,
        fl: &Su2Sketch<T>,
        ws: &Su2OneSiteWorkspace,
    ) -> Su2Sketch<T> {
        let xk = x.core(fl.site());
        self.sketching_step_right_mixed(xk, xk, fl, ws)
    }

    /// [`Self::sketching_step_right`] with DISTINCT bra and ket cores:
    /// `C = adjoint(Bₖ)·(Fᴸ·H·Kₖ)` at bond `k+1`, `k = site(Fᴸ)`. The same-vector entry point
    /// above is the `bra_k == ket_k` case — one implementation, two entry points.
    pub fn sketching_step_right_mixed(
        &self,
        bra_k: &Su2Core<T>,
        ket_k: &Su2Core<T>,
        fl: &Su2Sketch<T>,
        ws: &Su2OneSiteWorkspace,
    ) -> Su2Sketch<T> {
        let k = fl.site();
        debug_assert_eq!(
            bra_k.k, k,
            "sketching_step_right_mixed: bra core site mismatch"
        );
        debug_assert_eq!(
            ket_k.k, k,
            "sketching_step_right_mixed: ket core site mismatch"
        );
        let mut y = Su2Core::zeros(ket_k.space, k, &fl.row_ranks(), &self.col_ranks(ket_k));
        self.sketch_h_mul_into(&mut y, fl, ket_k, |_| true, &ws.i_tab, &ws.j_tab);
        crate::su2::coreops::adjoint_core_times_core(bra_k, &y)
    }

    /// Right-environment update `C = (H·Xₖ·Fᴿ)·adjoint(Xₖ)` at bond `k` (the
    /// streamed no-workspace form). **THE θ²-WEIGHTED `rproject`** (the
    /// adjoint-core form that is dead on the abelian layer is LIVE here) — the R band itself is
    /// built PLAIN
    /// (unweighted `h_sketch_mul` restricted to the row sector `q`), the θ² weights enter ONLY
    /// in the dual contraction. Per-family weight `theta_sq` (`sl`-storage-index ratio:
    /// `○○→1, up→(2s+2)/(2s+1), dn→2s/(2s+1), ●●→1`) — **[`crate::su2::coreops::theta_sq`] has
    /// exactly ONE definition; do not open-code the ratios or write a second one.** A plain
    /// adjoint here is a silent O(1) error (the mandatory anti-oracle arms pin it).
    /// `k = fr.site() - 1`, `xk = x.core(k)`. Uses the ROW range table (`ws.i_tab`), NOT `j_tab`
    /// — same trap the abelian kernel flags.
    pub fn sketching_step_left(
        &self,
        x: &Su2TtVector<T>,
        fr: &Su2Sketch<T>,
        ws: &Su2OneSiteWorkspace,
    ) -> Su2Sketch<T> {
        let xk = x.core(fr.site() - 1);
        self.sketching_step_left_mixed(xk, xk, fr, ws)
    }

    /// [`Self::sketching_step_left`] with DISTINCT bra and ket cores:
    /// `C = (H·Kₖ·Fᴿ)·adjoint(Bₖ)` at bond `k`, `k = site(Fᴿ) - 1`. Mirror of
    /// [`Self::sketching_step_right_mixed`] — same bra/ket contract, same `ws`-from-the-ket rule.
    pub fn sketching_step_left_mixed(
        &self,
        bra_k: &Su2Core<T>,
        ket_k: &Su2Core<T>,
        fr: &Su2Sketch<T>,
        ws: &Su2OneSiteWorkspace,
    ) -> Su2Sketch<T> {
        let k = fr.site() - 1;
        debug_assert_eq!(
            bra_k.k, k,
            "sketching_step_left_mixed: bra core site mismatch"
        );
        debug_assert_eq!(
            ket_k.k, k,
            "sketching_step_left_mixed: ket core site mismatch"
        );
        let xk = bra_k;
        let mut y = Su2Core::zeros(ket_k.space, k, &self.row_ranks(ket_k), &fr.col_ranks());
        self.h_sketch_mul_into(&mut y, ket_k, fr, |_| true, &ws.i_tab, &ws.j_tab);
        crate::su2::coreops::rproject(&y, xk)
    }

    pub(crate) fn recontract_right_rows(
        &self,
        xk: &Su2Core<T>,
        fr_next: &Su2Sketch<T>,
        keep: &[(Su2Sector, Vec<usize>)],
        ws: &Su2OneSiteWorkspace,
        fr_j: &mut Su2Sketch<T>,
    ) {
        if keep.iter().all(|(_, cs)| cs.is_empty()) {
            return;
        }
        let (space, k) = (self.space, xk.k);
        debug_assert_eq!(fr_next.site(), k + 1, "recontract_right_rows: Fᴿ bond");
        debug_assert_eq!(fr_j.site(), k, "recontract_right_rows: destination bond");
        let (ranks, tab) = compact_band_table(&space, &ws.i_tab, keep);
        let mut y = Su2Core::zeros(space, k, &ranks, &fr_next.col_ranks());
        self.h_sketch_mul_masked_into(&mut y, xk, fr_next, keep, &ws.j_tab, &tab);
        let c = crate::su2::coreops::rproject(&y, xk);
        scatter_bands(fr_j, &c, keep, &tab, &ws.i_tab, Along::Rows);
    }

    pub(crate) fn recontract_left_cols(
        &self,
        xk: &Su2Core<T>,
        fl_prev: &Su2Sketch<T>,
        keep: &[(Su2Sector, Vec<usize>)],
        ws: &Su2OneSiteWorkspace,
        fl_b: &mut Su2Sketch<T>,
    ) {
        if keep.iter().all(|(_, cs)| cs.is_empty()) {
            return;
        }
        let (space, k) = (self.space, xk.k);
        debug_assert_eq!(fl_prev.site(), k, "recontract_left_cols: Fᴸ bond");
        debug_assert_eq!(fl_b.site(), k + 1, "recontract_left_cols: destination bond");
        let (ranks, tab) = compact_band_table(&space, &ws.j_tab, keep);
        let mut y = Su2Core::zeros(space, k, &fl_prev.row_ranks(), &ranks);
        self.sketch_h_mul_masked_into(&mut y, fl_prev, xk, keep, &ws.i_tab, &tab);
        let c = crate::su2::coreops::adjoint_core_times_core(xk, &y);
        scatter_bands(fl_b, &c, keep, &tab, &ws.j_tab, Along::Cols);
    }

    /// Exact diagonal of `H_eff = (Fᴸ·H·Xₖ)·Fᴿ` in the coefficient basis. **NO θ, NO
    /// centerweight anywhere** ("the metric centerweight CANCELS in the
    /// Rayleigh-quotient diagonal — so there is NO θ in the diagonal"; validated against the
    /// layer-agnostic brute-force oracle [`crate::solvers::davidson::effective_diagonal_bruteforce`]).
    /// Uses the PLAIN transpose of `FRdiag`, NOT the adjoint (deliberate, metric-agnostic,
    /// load-bearing for complex `T` — mirrors the abelian `hamiltonian/matvec/` doc note). 0-based
    /// sector-bound check `q.n <= N && q.two_s <= N` — "the single highest-value
    /// line to review in this kernel".
    pub fn effective_diagonal(
        &self,
        fl: &Su2Sketch<T>,
        fr: &Su2Sketch<T>,
        x: &Su2Core<T>,
        ws: &Su2OneSiteWorkspace,
    ) -> Su2Core<T> {
        let space = self.space;
        let k = x.k;
        let g = space.pg.irrep_of(k);
        let mut dcore = Su2Core::zeros(space, k, &x.row_ranks, &x.col_ranks);
        let i_tab = &ws.i_tab;
        let j_tab = &ws.j_tab;
        let ch = self.col_ranks(x);

        let tasks: Vec<(Family, Su2Sector)> = Family::ALL
            .into_iter()
            .flat_map(|occ| x.row_qn.iter().map(move |&l| (occ, l)))
            .collect();
        let compute = |occ: Family, l: Su2Sector| -> Option<(Family, Su2Sector, Mat<T>)> {
            let q = occ.apply(l, g)?;
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
            let (Some(fl_blk), Some(fr_blk)) = (fl.block(l), fr.block(q)) else {
                return None;
            };

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
            let mut out = Mat::<T>::zeros(r_l, r_q);
            matmul(
                out.as_mut(),
                Accum::Replace,
                g.as_ref(),
                fr_diag.as_ref().transpose(),
                T::one(),
                Par::Seq,
            );
            Some((occ, l, out))
        };
        let results: Vec<(Family, Su2Sector, Mat<T>)> = if tasks.len() >= PAR_MIN_TASKS {
            tasks
                .into_par_iter()
                .filter_map(|(occ, l)| compute(occ, l))
                .collect()
        } else {
            tasks
                .into_iter()
                .filter_map(|(occ, l)| compute(occ, l))
                .collect()
        };
        for (occ, l, m) in results {
            if let Some(mut db) = dcore.block_mut(occ, l) {
                db.copy_from(m.as_ref());
            }
        }
        dcore
    }

    /// Two-site Davidson local operator `(Fᴸ·H·Xₖ)·(H·Xₖ₊₁·Fᴿ)`, contracted through
    /// [`factor_qc`] (the two-site [`Self::sketched_hamiltonian`] without workspace).
    /// `fl` @ bond `cp.site()`, `cp` @ site `k` spanning `(k,k+1)`, `fr` @
    /// bond `cp.site()+2` — `debug_assert!` both. Body = `factor_qc(cp)` + the streamed apply.
    pub fn two_site_sketched_hamiltonian(
        &self,
        cp: &Su2ContractedCores<T>,
        fl: &Su2Sketch<T>,
        fr: &Su2Sketch<T>,
    ) -> Su2ContractedCores<T> {
        debug_assert_eq!(fl.site(), cp.site(), "two_site matvec: Fᴸ @ bond k");
        debug_assert_eq!(fr.site(), cp.site() + 2, "two_site matvec: Fᴿ @ bond k+2");
        let (xk, xk1) = factor_qc(cp);
        let il = self.row_ranges(&xk);
        let jl = self.col_ranges(&xk);
        let ir = self.row_ranges(&xk1);
        let jr = self.col_ranges(&xk1);
        self.two_site_apply_streamed(&xk, &xk1, fl, fr, &il, &jl, &ir, &jr)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn two_site_apply_streamed(
        &self,
        xk: &Su2Core<T>,
        xk1: &Su2Core<T>,
        fl: &Su2Sketch<T>,
        fr: &Su2Sketch<T>,
        il: &Su2SectorGrid<Vec<Range<usize>>>,
        jl: &Su2SectorGrid<Vec<Range<usize>>>,
        ir: &Su2SectorGrid<Vec<Range<usize>>>,
        jr: &Su2SectorGrid<Vec<Range<usize>>>,
    ) -> Su2ContractedCores<T> {
        use crate::hamiltonian::matvec::two_site_stream_bytes;
        use crate::mem::Footprint as _;
        use crate::su2::twosite::{contract_mid_block, mpred, msucc};
        let space = xk.space;
        let k = xk.k;
        debug_assert_eq!(xk1.k, k + 1);
        let l_rows: Su2RankGrid = fl.row_ranks();
        let l_cols: Su2RankGrid = self.col_ranks(xk);
        let r_rows: Su2RankGrid = self.row_ranks(xk1);
        let r_cols: Su2RankGrid = fr.col_ranks();
        let sz = std::mem::size_of::<T>() as u64;
        let mut out = Su2ContractedCores::zeros(space, k, &l_rows, &r_cols);

        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Side {
            L,
            R,
        }
        struct Task {
            side: Side,
            fam: Family,
            sec: Su2Sector,
            rows: usize,
            cols: usize,
        }
        let per_mid: Vec<(Su2Sector, Vec<Task>, u64)> = xk
            .col_qn
            .iter()
            .filter(|&&m| out.block(m).is_some_and(|b| b.nrows() > 0 && b.ncols() > 0))
            .map(|&m| {
                let mut tasks = Vec::with_capacity(8);
                let mut bytes = 0u64;
                for (fi, l) in mpred(&space, k, m) {
                    let (rows, cols) = (l_rows[l], l_cols[m]);
                    if rows > 0 && cols > 0 {
                        bytes += rows as u64 * cols as u64 * sz;
                        tasks.push(Task {
                            side: Side::L,
                            fam: fi,
                            sec: l,
                            rows,
                            cols,
                        });
                    }
                }
                for (fj, r) in msucc(&space, k, m) {
                    let (rows, cols) = (r_rows[m], r_cols[r]);
                    if rows > 0 && cols > 0 {
                        bytes += rows as u64 * cols as u64 * sz;
                        tasks.push(Task {
                            side: Side::R,
                            fam: fj,
                            sec: m,
                            rows,
                            cols,
                        });
                    }
                }
                (m, tasks, bytes)
            })
            .collect();

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
            let refs: Vec<&Task> = wave.iter().flat_map(|&i| per_mid[i].1.iter()).collect();
            let blocks: Vec<Mat<T>> = refs
                .par_iter()
                .map(|t| {
                    let mut blk = Mat::<T>::zeros(t.rows, t.cols);
                    match t.side {
                        Side::L => {
                            self.sketch_h_mul_block_into(blk.as_mut(), fl, xk, t.fam, t.sec, il, jl)
                        }
                        Side::R => self.h_sketch_mul_block_into(
                            blk.as_mut(),
                            xk1,
                            fr,
                            t.fam,
                            t.sec,
                            ir,
                            jr,
                        ),
                    }
                    blk
                })
                .collect();
            if crate::hamiltonian::instrument::ENABLED {
                let wave_bytes: u64 = wave.iter().map(|&i| per_mid[i].2).sum();
                crate::hamiltonian::instrument::mem_hw(
                    crate::hamiltonian::instrument::memslot::APPLY,
                    wave_bytes + xk.bytes() + xk1.bytes(),
                );
            }
            let mut offs = Vec::with_capacity(wave.len());
            let mut o = 0usize;
            for &i in &wave {
                offs.push(o);
                o += per_mid[i].1.len();
            }
            let results: Vec<(Su2Sector, Mat<T>)> = wave
                .par_iter()
                .zip(offs.par_iter())
                .map(|(&i, &o)| {
                    let (m, tasks, _) = &per_mid[i];
                    let b = out.block(*m).expect("filtered above");
                    let mut local = Mat::<T>::zeros(b.nrows(), b.ncols());
                    let find = |side: Side, fam: Family, sec: Su2Sector| {
                        tasks
                            .iter()
                            .position(|t| t.side == side && t.fam == fam && t.sec == sec)
                            .map(|j| blocks[o + j].as_ref())
                    };
                    contract_mid_block(
                        &space,
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
        }
        out
    }

    /// Two-site sketched-Hamiltonian variant threading the [`Su2TwoSiteWorkspace`] cache.
    /// Identical result to [`Self::two_site_sketched_hamiltonian`] (bit-comparable gate) —
    /// only the range-table recomputation is cached across mid-rank-stable Krylov iterates via
    /// [`factor_qc_into`]. Like the one-site [`Self::sketched_hamiltonian`], this MUST route
    /// through the `_into` kernels consuming `ws.il/jl/ir/jr` (the alternative passes the four cached
    /// tables straight into the sketch-mul kernels); the allocating
    /// `sketch_h_mul`/`h_sketch_mul` wrappers rebuild the tables internally, which would make the
    /// workspace cache nothing and this form strictly slower than the no-workspace one.
    pub fn two_site_sketched_hamiltonian_ws(
        &self,
        ws: &mut Option<Su2TwoSiteWorkspace<T>>,
        cp: &Su2ContractedCores<T>,
        fl: &Su2Sketch<T>,
        fr: &Su2Sketch<T>,
    ) -> Su2ContractedCores<T> {
        debug_assert_eq!(fl.site(), cp.site(), "two_site matvec: Fᴸ @ bond k");
        debug_assert_eq!(fr.site(), cp.site() + 2, "two_site matvec: Fᴿ @ bond k+2");
        let w = factor_qc_into(ws, self, cp, fl, fr);
        self.two_site_apply_streamed(&w.xk, &w.xk1, fl, fr, &w.il, &w.jl, &w.ir, &w.jr)
    }

    /// Exact diagonal of `H_eff = (Fᴸ·H_k·H_{k+1}·Fᴿ)` on `cp`, in the coefficient basis.
    /// Factorizes through the mid operator bond `γ`; structurally the one-site
    /// [`Self::effective_diagonal`] run twice and joined. **NO θ, NO centerweight** (same
    /// verdict as the one-site form). The row arm shifts `l = m ⊖
    /// {(○○,0,0),(up,1,1),(dn,1,−1),(●●,2,0)}`, the col arm shifts `r = m ⊕ {same deltas}` — do NOT
    /// reuse one tuple for both. `coff += r_r` / `roff += r_l` happen even when the arm's rank is 0
    /// (the offset bookkeeping is NOT symmetric between row and col — the sharpest
    /// silent-corruption trap in this kernel). Uses the PLAIN transpose of the site-`(k+1)` gather
    /// (metric-agnostic, load-bearing for complex `T`).
    pub fn two_site_effective_diagonal(
        &self,
        fl: &Su2Sketch<T>,
        fr: &Su2Sketch<T>,
        cp: &Su2ContractedCores<T>,
    ) -> Su2ContractedCores<T> {
        let space = self.space;
        let k = cp.site();
        let (gl, gr) = (space.pg.irrep_of(k), space.pg.irrep_of(k + 1));
        let (xk, xk1) = factor_qc(cp);
        let ik = self.row_ranges(&xk);
        let jk1 = self.col_ranges(&xk1);
        let jk = self.col_ranges(&xk);
        let ik1 = self.row_ranges(&xk1);
        let mut dcore = cp.zeros_like();

        let mid_block = |m: Su2Sector| -> Option<(Su2Sector, Mat<T>)> {
            let (nr, nc) = {
                let b = cp.block(m)?;
                (b.nrows(), b.ncols())
            };
            if nr == 0 || nc == 0 {
                return None;
            }
            let n_gamma = jk[m].len().min(ik1[m].len());

            let gr_cols: Vec<(usize, Option<Mat<T>>)> = Family::ALL
                .into_iter()
                .map(|fj| {
                    let Some(r) = fj.apply(m, gr) else {
                        return (0, None);
                    };
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

            let mut local = Mat::<T>::zeros(nr, nc);
            let mut blk = local.as_mut();
            let mut roff = 0usize;
            for fi in Family::ALL {
                let Some(l) = fi.unapply(m, gl) else { continue };
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
        };
        let results: Vec<(Su2Sector, Mat<T>)> = if cp.mid_qn.len() >= PAR_MIN_TASKS {
            cp.mid_qn.par_iter().filter_map(|&m| mid_block(m)).collect()
        } else {
            cp.mid_qn.iter().filter_map(|&m| mid_block(m)).collect()
        };
        for (m, local) in results {
            if let Some(mut db) = dcore.block_mut(m) {
                db.copy_from(local.as_ref());
            }
        }
        dcore
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Along {
    Rows,
    Cols,
}

pub(crate) fn scatter_bands<T: Scalar>(
    dst: &mut Su2Sketch<T>,
    src: &Su2Sketch<T>,
    keep: &[(Su2Sector, Vec<usize>)],
    from: &Su2SectorGrid<Vec<Range<usize>>>,
    to: &Su2SectorGrid<Vec<Range<usize>>>,
    along: Along,
) {
    for (s, cs) in keep.iter().filter(|(_, cs)| !cs.is_empty()) {
        let b = src
            .block(*s)
            .expect("a kept sector is admissible at the bond");
        let mut d = dst
            .block_mut(*s)
            .expect("the destination holds every admissible sector");
        debug_assert!(match along {
            Along::Rows => d.ncols() == b.ncols(),
            Along::Cols => d.nrows() == b.nrows(),
        });
        for &c in cs {
            let (f, t) = (&from[*s][c], &to[*s][c]);
            match along {
                Along::Rows => d
                    .rb_mut()
                    .submatrix_mut(t.start, 0, t.len(), b.ncols())
                    .copy_from(b.submatrix(f.start, 0, f.len(), b.ncols())),
                Along::Cols => d
                    .rb_mut()
                    .submatrix_mut(0, t.start, b.nrows(), t.len())
                    .copy_from(b.submatrix(0, f.start, b.nrows(), f.len())),
            }
        }
    }
}

pub(crate) fn mix_bands<T: Scalar>(
    dst: &mut Su2Sketch<T>,
    pairs: &[(Su2Sector, Vec<[u32; 2]>)],
    tab: &Su2SectorGrid<Vec<Range<usize>>>,
    along: Along,
    c: f64,
    s: f64,
) {
    mix_bands_as::<T, false>(dst, pairs, tab, along, c, s);
}

#[cfg(test)]
pub(crate) fn mix_bands_swapped<T: Scalar>(
    dst: &mut Su2Sketch<T>,
    pairs: &[(Su2Sector, Vec<[u32; 2]>)],
    tab: &Su2SectorGrid<Vec<Range<usize>>>,
    along: Along,
    c: f64,
    s: f64,
) {
    mix_bands_as::<T, true>(dst, pairs, tab, along, c, s);
}

fn mix_bands_as<T: Scalar, const SWAP: bool>(
    dst: &mut Su2Sketch<T>,
    pairs: &[(Su2Sector, Vec<[u32; 2]>)],
    tab: &Su2SectorGrid<Vec<Range<usize>>>,
    along: Along,
    c: f64,
    s: f64,
) {
    let (ct, st) = (T::from_f64(c), T::from_f64(s));
    let rotate = |lo: &mut [T], hi: &mut [T]| {
        for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
            let (x, y) = (*a, *b);
            let (nl, nh) = (ct * x - st * y, st * x + ct * y);
            if SWAP {
                (*a, *b) = (nh, nl);
            } else {
                (*a, *b) = (nl, nh);
            }
        }
    };
    for (sec, ps) in pairs.iter().filter(|(_, ps)| !ps.is_empty()) {
        let bands = &tab[*sec];
        let mut blk = dst
            .block_mut(*sec)
            .expect("a mixed sector is admissible at the bond");
        for &[lo, hi] in ps {
            assert_eq!(
                bands[lo as usize].len(),
                bands[hi as usize].len(),
                "mix_bands: sector {sec:?} twin bands [{lo}, {hi}] of unequal widths"
            );
        }
        match along {
            Along::Rows => {
                for j in 0..blk.ncols() {
                    let col = blk
                        .rb_mut()
                        .col_mut(j)
                        .try_as_col_major_mut()
                        .expect("a sketch block column is contiguous")
                        .as_slice_mut();
                    for &[lo, hi] in ps {
                        let (l, h) = (&bands[lo as usize], &bands[hi as usize]);
                        if l.start < h.start {
                            let (a, b) = col.split_at_mut(h.start);
                            rotate(&mut a[l.clone()], &mut b[..h.len()]);
                        } else {
                            let (b, a) = col.split_at_mut(l.start);
                            rotate(&mut a[..l.len()], &mut b[h.clone()]);
                        }
                    }
                }
            }
            Along::Cols => {
                for &[lo, hi] in ps {
                    let (l, h) = (&bands[lo as usize], &bands[hi as usize]);
                    for (pl, ph) in l.clone().zip(h.clone()) {
                        let (a, b) = blk.rb_mut().two_cols_mut(pl, ph);
                        rotate(
                            a.try_as_col_major_mut()
                                .expect("a sketch block column is contiguous")
                                .as_slice_mut(),
                            b.try_as_col_major_mut()
                                .expect("a sketch block column is contiguous")
                                .as_slice_mut(),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod l2_streamed_apply_tests {
    use crate::hamiltonian::matvec::set_two_site_stream_bytes;
    use crate::su2::hamiltonian::matvec::{h_sketch_mul, sketch_h_mul};
    use crate::su2::qn::Su2Space;
    use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::su2::twosite::{contract, factor_qc};
    use crate::test_util::{generic_su2_h, randomized_su2_state, seeded};

    #[test]
    fn streamed_apply_is_bit_identical_to_full_cores() {
        let space = Su2Space::new(6, 4, 0);
        let h = generic_su2_h::<f64>(space);
        let mut rng = seeded(0x12_0004);
        let mut x = randomized_su2_state::<f64, _>(space, 2, 4, &mut rng);
        x.right_orthogonalize(true);
        let k = 2usize;
        x.move_core(k, true);
        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let fr_all = right_to_left_sketching(&h, &x);
        let fr = &fr_all[k + 1];
        let cp = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&cp);
        let reference = contract(&sketch_h_mul(&fl, &h, &xk), &h_sketch_mul(&h, &xk1, fr));
        let (il, jl, ir, jr) = (
            h.row_ranges(&xk),
            h.col_ranges(&xk),
            h.row_ranges(&xk1),
            h.col_ranges(&xk1),
        );
        let check = |label: &str| {
            let got = h.two_site_apply_streamed(&xk, &xk1, &fl, fr, &il, &jl, &ir, &jr);
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
                                let (p, q) = (*a.get(i, j), *b.get(i, j));
                                assert!(
                                    p.to_bits() == q.to_bits() || (p == 0.0 && q == 0.0),
                                    "{label}: {m:?} ({i},{j}) {p} vs {q}"
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
}
