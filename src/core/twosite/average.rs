// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::SparseCore;
use crate::core::factor::{Sketch, chop_energy, dense_self_adjoint_eigen};
use crate::qn::{Pooled, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

use super::ContractedSparseCores;
use super::noise::{
    TruncInfo, factor_svd_perturbative, frob_norm, frob_sq, rho_prime_eig, scale_mat, svals_sketch,
};

/// Diagnostics of the center Gram `G = [⟨cᵢ, cⱼ⟩]` the Löwdin step of [`factor_svd_average`]
/// inverts — reported whether or not the re-orthonormalization actually ran, because the
/// interesting number is how far the truncated states had drifted from orthogonality.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LoewdinInfo {
    /// `max_{i≠j} |Gᵢⱼ|` BEFORE the Löwdin step (0 at `n = 1`).
    pub overlap_dev: f64,
    /// Smallest eigenvalue of `G` (1 at `n = 1`).
    pub gram_min: f64,
    /// Largest eigenvalue of `G` (1 at `n = 1`).
    pub gram_max: f64,
}

/// [`super::factor_svd_perturbative`] for `n` states at once: ONE pooled chop of the weighted
/// density matrix `ρ = Σᵢ wᵢ BᵢBᵢᴴ`, ONE shared basis, `n` per-state centers.
///
/// # Panics
/// On any violated precondition of the module header, and when the Gram is singular under
/// `reorthonormalize` (two roots collapsed onto the same kept directions).
/// `chi` is a POOLED cap on the split bond's TOTAL dimension `Σ_q r[q]` ([`Pooled::NONE`] =
/// uncapped).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn factor_svd_average<T: Scalar>(
    cps: &[ContractedSparseCores<T>],
    w: &[f64],
    noise_dm: Option<&SectorGrid<Mat<T>>>,
    alpha: f64,
    eps: f64,
    chi: Pooled,
    forward: bool,
    reorthonormalize: bool,
) -> (
    SparseCore<T>,
    Sketch<T>,
    Vec<SparseCore<T>>,
    Vec<TruncInfo>,
    TruncInfo,
    LoewdinInfo,
) {
    let n = cps.len();
    validate(cps, w);

    let space = cps[0].space;
    let k = cps[0].k;

    let nd_empty;
    let nd: &SectorGrid<Mat<T>> = match noise_dm {
        Some(g) => g,
        None => {
            nd_empty = SectorGrid::from_fn(&space, |_| Mat::<T>::zeros(0, 0));
            &nd_empty
        }
    };

    if n == 1 {
        let (xk, s, xk1, info) =
            factor_svd_perturbative(&cps[0], nd, eps, chi, true, alpha, forward);
        let (shared, center) = if forward { (xk, xk1) } else { (xk1, xk) };
        return (
            shared,
            s,
            vec![center],
            vec![info],
            info,
            LoewdinInfo {
                overlap_dev: 0.0,
                gram_min: 1.0,
                gram_max: 1.0,
            },
        );
    }

    let nd_norm = frob_norm(cps[0].mid_qn.iter().map(|&m| nd[m].as_ref()));
    let alpha_eff = if alpha > 0.0 && nd_norm > 0.0 {
        alpha / nd_norm
    } else {
        0.0
    };
    let alpha_t = T::from_f64(alpha_eff);
    let cap = (alpha > 0.0 && !space.pg.is_trivial())
        .then(|| crate::qn::max_ranks(&space).swap_remove(k + 1));

    type EigA<T> = (Sector, Mat<T>, Vec<f64>);
    let gram_eig = |m: Sector| -> EigA<T> {
        let dims = cps[0].block(m).map(|b| (b.nrows(), b.ncols()));
        rho_prime_eig(
            m,
            dims,
            &nd[m],
            forward,
            alpha > 0.0,
            cap.as_ref(),
            alpha_t,
            |rho| {
                for (i, cp) in cps.iter().enumerate() {
                    let b = cp
                        .block(m)
                        .expect("equal shapes: block present at every state");
                    let wi = T::from_f64(w[i]);
                    if forward {
                        matmul(rho.as_mut(), Accum::Add, b, b.adjoint(), wi, Par::Seq);
                    } else {
                        matmul(rho.as_mut(), Accum::Add, b.adjoint(), b, wi, Par::Seq);
                    }
                }
            },
        )
    };
    let mid_qn = &cps[0].mid_qn;
    let adata: Vec<EigA<T>> = if mid_qn.len() >= 4 {
        mid_qn.par_iter().map(|&m| gram_eig(m)).collect()
    } else {
        mid_qn.iter().map(|&m| gram_eig(m)).collect()
    };
    let adata_idx: FxHashMap<Sector, usize> = adata
        .iter()
        .enumerate()
        .map(|(i, (s, ..))| (*s, i))
        .collect();

    let s_full = svals_sketch::<T>(space, k + 1, |m| {
        adata_idx.get(&m).map(|&i| adata[i].2.as_slice())
    });
    let mut mid_rank = chop_energy(&s_full, eps, chi);

    let center_dim = |mr: &RankGrid| -> usize {
        mid_qn
            .iter()
            .map(|&m| {
                mr[m]
                    * cps[0]
                        .block(m)
                        .map_or(0, |b| if forward { b.ncols() } else { b.nrows() })
            })
            .sum()
    };
    let full = s_full.row_ranks().total();
    let mut cap_total = mid_rank.total();
    while center_dim(&mid_rank) < n && cap_total < full {
        cap_total += 1;
        mid_rank = chop_energy(&s_full, 0.0, Pooled(cap_total));
    }
    assert!(
        center_dim(&mid_rank) >= n,
        "factor_svd_average: bond {} holds only {} center coefficient(s) even at full rank — it \
         cannot host {n} states; enrich the start (the local space is too small for this many \
         roots)",
        k + 1,
        center_dim(&mid_rank),
    );

    let inv_norm: Vec<T> = cps
        .iter()
        .map(|cp| {
            let s = frob_norm(mid_qn.iter().filter_map(|&m| cp.block(m)));
            T::from_f64(if s > 0.0 { 1.0 / s } else { 1.0 })
        })
        .collect();

    let (mut shared, center_proto) = if forward {
        (
            SparseCore::zeros(space, k, &cps[0].row_ranks, &mid_rank),
            SparseCore::zeros(space, k + 1, &mid_rank, &cps[0].col_ranks),
        )
    } else {
        (
            SparseCore::zeros(space, k + 1, &mid_rank, &cps[0].col_ranks),
            SparseCore::zeros(space, k, &cps[0].row_ranks, &mid_rank),
        )
    };
    let mut centers: Vec<SparseCore<T>> = (0..n).map(|_| center_proto.clone()).collect();

    let s_out_blocks = SectorGrid::from_fn(&space, |m| {
        space.contains(k + 1, m).then(|| {
            let r = mid_rank[m];
            let mut mat = Mat::<T>::zeros(r, r);
            for i in 0..r {
                *mat.get_mut(i, i) = T::one();
            }
            mat
        })
    });
    let s_out = Sketch::new(space, k + 1, s_out_blocks);

    let (mut dropped2, mut total2, mut kept) = (vec![0.0f64; n], vec![0.0f64; n], 0usize);
    for (m, basis, _s) in &adata {
        let r = mid_rank[*m];
        kept += r;
        let Some(b0) = cps[0].block(*m) else { continue };
        let (rows, cols) = (b0.nrows(), b0.ncols());
        let has_state = rows > 0 && cols > 0;
        if has_state {
            for (i, cp) in cps.iter().enumerate() {
                total2[i] += frob_sq(cp.block(*m).expect("equal shapes"));
            }
        }
        if r == 0 {
            if has_state {
                for (i, cp) in cps.iter().enumerate() {
                    dropped2[i] += frob_sq(cp.block(*m).expect("equal shapes"));
                }
            }
            continue;
        }
        let rr = r.min(basis.ncols());
        if forward {
            let u = basis.submatrix(0, 0, rows, rr);
            shared.scatter_vertical(*m, u);
            if has_state {
                for (i, cp) in cps.iter().enumerate() {
                    let b = cp.block(*m).expect("equal shapes");
                    let mut center = Mat::<T>::zeros(rr, cols);
                    matmul(
                        center.as_mut(),
                        Accum::Replace,
                        u.adjoint(),
                        b,
                        T::one(),
                        Par::Seq,
                    );
                    dropped2[i] += (frob_sq(b) - frob_sq(center.as_ref())).max(0.0);
                    scale_mat(&mut center, inv_norm[i]);
                    centers[i].scatter_horizontal(*m, center.as_ref());
                }
            }
        } else {
            let v = basis.submatrix(0, 0, cols, rr);
            let mut vt_out = Mat::<T>::zeros(rr, cols);
            for i in 0..rr {
                for j in 0..cols {
                    *vt_out.get_mut(i, j) = v[(j, i)].conj_s();
                }
            }
            shared.scatter_horizontal(*m, vt_out.as_ref());
            if has_state {
                for (i, cp) in cps.iter().enumerate() {
                    let b = cp.block(*m).expect("equal shapes");
                    let mut center = Mat::<T>::zeros(rows, rr);
                    matmul(center.as_mut(), Accum::Replace, b, v, T::one(), Par::Seq);
                    dropped2[i] += (frob_sq(b) - frob_sq(center.as_ref())).max(0.0);
                    scale_mat(&mut center, inv_norm[i]);
                    centers[i].scatter_vertical(*m, center.as_ref());
                }
            }
        }
    }
    let infos: Vec<TruncInfo> = (0..n)
        .map(|i| TruncInfo {
            discarded_weight: if total2[i] > 0.0 {
                dropped2[i] / total2[i]
            } else {
                0.0
            },
            kept,
        })
        .collect();
    let pooled = TruncInfo {
        discarded_weight: (0..n).map(|i| w[i] * infos[i].discarded_weight).sum(),
        kept,
    };

    let mut gram = Mat::<T>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            *gram.get_mut(i, j) = centers[i].dot_cores(&centers[j]);
        }
    }
    let (loewdin, transform) = loewdin_dense(gram.as_ref(), reorthonormalize, k);
    if let Some(mt) = transform {
        let mut fresh: Vec<SparseCore<T>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut c = centers[0].clone();
            c.scale_mut(mt[(0, i)]);
            for j in 1..n {
                c.axpy(mt[(j, i)], &centers[j]);
            }
            let nrm = c.norm();
            if nrm > 0.0 {
                c.scale_mut(T::from_f64(1.0 / nrm));
            }
            fresh.push(c);
        }
        centers = fresh;
    }

    (shared, s_out, centers, infos, pooled, loewdin)
}

fn validate<T: Scalar>(cps: &[ContractedSparseCores<T>], w: &[f64]) {
    assert!(
        !cps.is_empty() && cps.len() == w.len(),
        "factor_svd_average: {} local solutions but {} weights (both must be ≥ 1 and equal)",
        cps.len(),
        w.len(),
    );
    assert!(
        w.iter().all(|a| a.is_finite() && *a > 0.0),
        "factor_svd_average: weights must be finite and strictly positive (got {w:?}) — a \
         zero-weight state is truncated away with no trace in the ledger",
    );
    assert!(
        w.windows(2).all(|p| p[0] >= p[1]),
        "factor_svd_average: weights must be non-increasing in label order (got {w:?})",
    );
    let sum: f64 = w.iter().sum();
    assert!(
        (sum - 1.0).abs() <= 1e-12,
        "factor_svd_average: weights must sum to 1 (got {sum}) — the chop tolerance is ABSOLUTE \
         and `Σwᵢ‖Bᵢ‖²` is the trace of ρ",
    );
    for (i, cp) in cps.iter().enumerate().skip(1) {
        assert!(
            cp.space == cps[0].space
                && cp.k == cps[0].k
                && cp.row_ranks == cps[0].row_ranks
                && cp.col_ranks == cps[0].col_ranks,
            "factor_svd_average: state {i} has a different two-site shape than state 0 (space, \
             site or bond ranks) — every state must be measured in the SAME local frames",
        );
    }
    for (i, cp) in cps.iter().enumerate() {
        let nrm = cp.norm();
        assert!(
            (nrm - 1.0).abs() <= 1e-8,
            "factor_svd_average: state {i} has norm {nrm} — the driver must hand UNIT-norm local \
             solutions (they are NOT normalized here: the chop tolerance is absolute)",
        );
    }
}

pub(crate) fn loewdin_dense<T: Scalar>(
    gram: MatRef<'_, T>,
    reorthonormalize: bool,
    k: usize,
) -> (LoewdinInfo, Option<Mat<T>>) {
    let n = gram.nrows();
    let mut overlap_dev = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                overlap_dev = overlap_dev.max(gram[(i, j)].abs_s());
            }
        }
    }
    let (evals, u) = dense_self_adjoint_eigen(gram);
    let info = LoewdinInfo {
        overlap_dev,
        gram_min: evals.first().copied().unwrap_or(0.0),
        gram_max: evals.last().copied().unwrap_or(0.0),
    };
    if !reorthonormalize {
        return (info, None);
    }
    assert!(
        info.gram_min > 1e-12 * info.gram_max,
        "factor_svd_average: the center Gram at bond {} is singular (λmin/λmax = {:.3e}) — two \
         state-average roots collapsed onto the same kept directions, so `G^-1/2` is undefined; \
         raise the rank/tolerance or give the roots distinct starts",
        k + 1,
        info.gram_min / info.gram_max.max(f64::MIN_POSITIVE),
    );
    let mut mt = Mat::<T>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let mut acc = T::zero();
            for p in 0..n {
                acc += u[(j, p)] * T::from_f64(evals[p].powf(-0.5)) * u[(i, p)].conj_s();
            }
            *mt.get_mut(j, i) = acc;
        }
    }
    (info, Some(mt))
}

pub(crate) fn weighted_noise_sum<T: Scalar>(
    nds: &[SectorGrid<Mat<T>>],
    w: &[f64],
    template: &ContractedSparseCores<T>,
    forward: bool,
) -> SectorGrid<Mat<T>> {
    let space = template.space;
    let mut out = SectorGrid::from_fn(&space, |_| Mat::<T>::zeros(0, 0));
    for &m in template.mid_qn.iter() {
        let Some(b) = template.block(m) else { continue };
        let g = if forward { b.nrows() } else { b.ncols() };
        if g == 0 {
            continue;
        }
        let mut acc = Mat::<T>::zeros(g, g);
        let mut contributed = false;
        for (i, nd) in nds.iter().enumerate() {
            let blk = &nd[m];
            if blk.nrows() == 0 && blk.ncols() == 0 {
                continue;
            }
            assert!(
                blk.nrows() == g && blk.ncols() == g,
                "noise_dm[state {i}][{m:?}] is {}x{}, want {g}x{g} or absent (forward = {forward})",
                blk.nrows(),
                blk.ncols()
            );
            let wi = T::from_f64(w[i]);
            for j in 0..g {
                for r in 0..g {
                    let v = acc[(r, j)] + wi * blk[(r, j)];
                    *acc.get_mut(r, j) = v;
                }
            }
            contributed = true;
        }
        if contributed {
            out[m] = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{QnSpace, TtVector, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn two_site(seed: u64) -> (ContractedSparseCores<f64>, QnSpace) {
        let space = QnSpace::new(6, 1, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: TtVector<f64> = tt_randn(space, &crate::max_ranks(&space), &mut rng);
        x.right_orthogonalize(false);
        x.move_core(2, false);
        (super::super::contract(x.core(2), x.core(3)), space)
    }

    fn eye(g: usize, s: f64) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(g, g);
        for i in 0..g {
            *m.get_mut(i, i) = s;
        }
        m
    }

    fn two_sectors(tc: &ContractedSparseCores<f64>) -> (Sector, usize, Sector, usize) {
        let live: Vec<(Sector, usize)> = tc
            .mid_qn
            .iter()
            .filter_map(|&m| tc.block(m).map(|b| (m, b.nrows())))
            .filter(|&(_, g)| g > 0)
            .collect();
        assert!(live.len() >= 2, "fixture needs two populated mid sectors");
        (live[0].0, live[0].1, live[1].0, live[1].1)
    }

    #[test]
    fn weighted_noise_sum_applies_each_state_its_own_weight() {
        let (tc, space) = two_site(0xB3_2001);
        let (a, ga, b, gb) = two_sectors(&tc);
        let nd0 = SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let nd1 = SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga, 2.0)
            } else if m == b {
                eye(gb, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let w = [0.6, 0.4];
        let out = weighted_noise_sum(&[nd0, nd1], &w, &tc, true);

        for i in 0..ga {
            for j in 0..ga {
                let want = if i == j { 0.6 * 1.0 + 0.4 * 2.0 } else { 0.0 };
                assert!(
                    (out[a][(i, j)] - want).abs() <= 1e-15,
                    "both states present: entry ({i},{j}) = {} want {want}",
                    out[a][(i, j)]
                );
            }
        }
        for i in 0..gb {
            for j in 0..gb {
                let want = if i == j { 0.4 } else { 0.0 };
                assert!(
                    (out[b][(i, j)] - want).abs() <= 1e-15,
                    "only state 1 present: entry ({i},{j}) = {} want {want} (state 1's OWN \
                     weight, never state 0's)",
                    out[b][(i, j)]
                );
            }
        }
        for &m in tc.mid_qn.iter() {
            if m != a && m != b {
                assert_eq!(
                    (out[m].nrows(), out[m].ncols()),
                    (0, 0),
                    "sector {m:?}: no state contributed, so the sum must stay ABSENT — a zeroed \
                     `g×g` block would enter the `noise_dm` grid as live noise"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "want")]
    fn weighted_noise_sum_refuses_a_mis_shaped_block() {
        let (tc, space) = two_site(0xB3_2011);
        let (a, ga, _b, _gb) = two_sectors(&tc);
        let nd0 = SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let nd1 = SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga + 1, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let _ = weighted_noise_sum(&[nd0, nd1], &[0.5, 0.5], &tc, true);
    }
}
