// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::twosite::{LoewdinInfo, TruncInfo, loewdin_dense};
use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, SymSpace};

use super::noise::{ThetaMetric, frob_sq2, rho_prime_eig, svals_sketch};
use super::{PAR_MIN_MID_SECTORS, Su2ContractedCores, factor_svd_perturbative};

/// [`crate::core::twosite::factor_svd_average`] for [`super::Su2ContractedCores`]: ONE pooled chop
/// of the weighted density matrix `ρ = Σᵢ wᵢ B_θᵢB_θᵢᴴ`, ONE shared basis, `n` per-state centers,
/// with θ at the three sites of the module header.
///
/// # Panics
/// On any violated precondition of the abelian module header, and when the center Gram is singular
/// under `reorthonormalize` (two roots collapsed onto the same kept directions).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn factor_svd_average<T: Scalar>(
    cps: &[Su2ContractedCores<T>],
    w: &[f64],
    noise_dm: Option<&Su2SectorGrid<Mat<T>>>,
    alpha: f64,
    eps: f64,
    chi: Pooled,
    forward: bool,
    reorthonormalize: bool,
) -> (
    Su2Core<T>,
    Su2Sketch<T>,
    Vec<Su2Core<T>>,
    Vec<TruncInfo>,
    TruncInfo,
    LoewdinInfo,
) {
    let n = cps.len();
    validate(cps, w);

    let space = cps[0].space;
    let k = cps[0].k;

    let nd_empty;
    let nd: &Su2SectorGrid<Mat<T>> = match noise_dm {
        Some(g) => g,
        None => {
            nd_empty = Su2SectorGrid::from_fn(&space, |_| Mat::<T>::zeros(0, 0));
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

    let mid_qn = &cps[0].mid_qn;
    let metric = ThetaMetric::new(&cps[0]);
    let b_theta = |cp: &Su2ContractedCores<T>, m: Su2Sector| -> Option<Mat<T>> {
        metric.scaled(cp.block(m)?, m)
    };
    let bthetas: Vec<Vec<Option<Mat<T>>>> = mid_qn
        .iter()
        .map(|&m| cps.iter().map(|cp| b_theta(cp, m)).collect())
        .collect();

    let mut nd_norm2 = 0.0f64;
    for &m in mid_qn {
        nd_norm2 += frob_sq2(nd[m].as_ref());
    }
    let nd_norm = nd_norm2.sqrt();
    let alpha_eff = if alpha > 0.0 && nd_norm > 0.0 {
        alpha / nd_norm
    } else {
        0.0
    };
    let alpha_t = T::from_f64(alpha_eff);
    let cap = (alpha > 0.0 && !space.all_admissible_reachable())
        .then(|| crate::su2::qn::max_ranks(&space).swap_remove(k + 1));

    type EigA<T> = (Su2Sector, Mat<T>, Vec<f64>);
    let gram_eig = |pos: usize| -> EigA<T> {
        let m = mid_qn[pos];
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
                for (i, bt) in bthetas[pos].iter().enumerate() {
                    let bt = bt
                        .as_ref()
                        .expect("has_state implies a nonempty θ-scaled block");
                    let wi = T::from_f64(w[i]);
                    if forward {
                        matmul(
                            rho.as_mut(),
                            Accum::Add,
                            bt.as_ref(),
                            bt.adjoint(),
                            wi,
                            Par::Seq,
                        );
                    } else {
                        matmul(
                            rho.as_mut(),
                            Accum::Add,
                            bt.adjoint(),
                            bt.as_ref(),
                            wi,
                            Par::Seq,
                        );
                    }
                }
            },
        )
    };
    let adata: Vec<EigA<T>> = if mid_qn.len() >= PAR_MIN_MID_SECTORS {
        (0..mid_qn.len()).into_par_iter().map(gram_eig).collect()
    } else {
        (0..mid_qn.len()).map(gram_eig).collect()
    };
    let adata_idx: FxHashMap<Su2Sector, usize> = adata
        .iter()
        .enumerate()
        .map(|(i, (s, ..))| (*s, i))
        .collect();

    let s_full = svals_sketch::<T>(space, k + 1, |m| {
        adata_idx.get(&m).map(|&i| adata[i].2.as_slice())
    });
    let mut mid_rank = crate::su2::chop::chop_energy(&s_full, eps, chi);

    let center_dim = |mr: &Su2RankGrid| -> usize {
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
        mid_rank = crate::su2::chop::chop_energy(&s_full, 0.0, Pooled(cap_total));
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
            let s = cp.norm();
            T::from_f64(if s > 0.0 { 1.0 / s } else { 1.0 })
        })
        .collect();

    let (mut shared, center_proto) = if forward {
        (
            Su2Core::zeros(space, k, &cps[0].row_ranks, &mid_rank),
            Su2Core::zeros(space, k + 1, &mid_rank, &cps[0].col_ranks),
        )
    } else {
        (
            Su2Core::zeros(space, k + 1, &mid_rank, &cps[0].col_ranks),
            Su2Core::zeros(space, k, &cps[0].row_ranks, &mid_rank),
        )
    };
    let mut centers: Vec<Su2Core<T>> = (0..n).map(|_| center_proto.clone()).collect();

    let s_out_blocks = Su2SectorGrid::from_fn(&space, |m| {
        space.contains(k + 1, m).then(|| {
            let r = mid_rank[m];
            let mut mat = Mat::<T>::zeros(r, r);
            for i in 0..r {
                *mat.get_mut(i, i) = T::one();
            }
            mat
        })
    });
    let s_out = Su2Sketch::new(space, k + 1, s_out_blocks);

    let (mut dropped2, mut total2, mut kept) = (vec![0.0f64; n], vec![0.0f64; n], 0usize);
    for (pos, (m, basis, _s)) in adata.iter().enumerate() {
        let r = mid_rank[*m];
        kept += r;
        let Some(b0) = cps[0].block(*m) else { continue };
        let (rows, cols) = (b0.nrows(), b0.ncols());
        let has_state = rows > 0 && cols > 0;
        let cw = crate::su2::centerweight(m.two_s, space.two_s);
        if has_state {
            for i in 0..n {
                let bt = bthetas[pos][i].as_ref().expect("has_state");
                total2[i] += cw * frob_sq2(bt.as_ref());
            }
        }
        if r == 0 {
            if has_state {
                for i in 0..n {
                    let bt = bthetas[pos][i].as_ref().expect("has_state");
                    dropped2[i] += cw * frob_sq2(bt.as_ref());
                }
            }
            continue;
        }
        let rr = r.min(basis.ncols());
        if forward {
            let u = basis.submatrix(0, 0, rows, rr);
            shared.scatter_vertical(*m, u);
            if has_state {
                for i in 0..n {
                    let bt = bthetas[pos][i].as_ref().expect("has_state");
                    let mut center = Mat::<T>::zeros(rr, cols);
                    matmul(
                        center.as_mut(),
                        Accum::Replace,
                        u.adjoint(),
                        bt.as_ref(),
                        T::one(),
                        Par::Seq,
                    );
                    dropped2[i] +=
                        cw * (frob_sq2(bt.as_ref()) - frob_sq2(center.as_ref())).max(0.0);
                    scale_mat(&mut center, inv_norm[i]);
                    centers[i].unscale_horizontal(&mut center, *m);
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
            shared.unscale_horizontal(&mut vt_out, *m);
            shared.scatter_horizontal(*m, vt_out.as_ref());
            if has_state {
                for i in 0..n {
                    let bt = bthetas[pos][i].as_ref().expect("has_state");
                    let mut center = Mat::<T>::zeros(rows, rr);
                    matmul(
                        center.as_mut(),
                        Accum::Replace,
                        bt.as_ref(),
                        v,
                        T::one(),
                        Par::Seq,
                    );
                    dropped2[i] +=
                        cw * (frob_sq2(bt.as_ref()) - frob_sq2(center.as_ref())).max(0.0);
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
            *gram.get_mut(i, j) = centers[i].dot_center(&centers[j]);
        }
    }
    let (loewdin, transform) = loewdin_dense(gram.as_ref(), reorthonormalize, k);
    if let Some(mt) = transform {
        let mut fresh: Vec<Su2Core<T>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut c = centers[0].clone();
            c.scale_mut(mt[(0, i)]);
            for j in 1..n {
                c.axpy(mt[(j, i)], &centers[j]);
            }
            let nrm = c.norm_center();
            if nrm > 0.0 {
                c.scale_mut(T::from_f64(1.0 / nrm));
            }
            fresh.push(c);
        }
        centers = fresh;
    }

    (shared, s_out, centers, infos, pooled, loewdin)
}

fn validate<T: Scalar>(cps: &[Su2ContractedCores<T>], w: &[f64]) {
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

fn scale_mat<T: Scalar>(m: &mut Mat<T>, a: T) {
    let (nr, nc) = (m.nrows(), m.ncols());
    for j in 0..nc {
        for i in 0..nr {
            let v = m[(i, j)] * a;
            *m.get_mut(i, j) = v;
        }
    }
}

pub(crate) fn weighted_noise_sum<T: Scalar>(
    nds: &[Su2SectorGrid<Mat<T>>],
    w: &[f64],
    template: &Su2ContractedCores<T>,
    forward: bool,
) -> Su2SectorGrid<Mat<T>> {
    let space = template.space;
    let mut out = Su2SectorGrid::from_fn(&space, |_| Mat::<T>::zeros(0, 0));
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
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::{Su2TtVector, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn two_site(seed: u64) -> (Su2ContractedCores<f64>, Su2Space) {
        let space = Su2Space::new(6, 2, 2);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: Su2TtVector<f64> = tt_randn(space, &crate::su2::qn::max_ranks(&space), &mut rng);
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

    fn two_sectors(tc: &Su2ContractedCores<f64>) -> (Su2Sector, usize, Su2Sector, usize) {
        let live: Vec<(Su2Sector, usize)> = tc
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
        let (tc, space) = two_site(0xB3_2101);
        let (a, ga, b, gb) = two_sectors(&tc);
        let nd0 = Su2SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let nd1 = Su2SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga, 2.0)
            } else if m == b {
                eye(gb, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let out = weighted_noise_sum(&[nd0, nd1], &[0.6, 0.4], &tc, true);

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
                    "only state 1 present: entry ({i},{j}) = {} want {want}",
                    out[b][(i, j)]
                );
            }
        }
        for &m in tc.mid_qn.iter() {
            if m != a && m != b {
                assert_eq!(
                    (out[m].nrows(), out[m].ncols()),
                    (0, 0),
                    "sector {m:?}: no state contributed, so the sum must stay ABSENT"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "want")]
    fn weighted_noise_sum_refuses_a_mis_shaped_block() {
        let (tc, space) = two_site(0xB3_2111);
        let (a, ga, _b, _gb) = two_sectors(&tc);
        let nd0 = Su2SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let nd1 = Su2SectorGrid::from_fn(&space, |m| {
            if m == a {
                eye(ga + 1, 1.0)
            } else {
                Mat::<f64>::zeros(0, 0)
            }
        });
        let _ = weighted_noise_sum(&[nd0, nd1], &[0.5, 0.5], &tc, true);
    }
}
