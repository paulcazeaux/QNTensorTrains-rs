// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::factor::{dense_self_adjoint_eigen, svd_uvt_svals};
use crate::core::twosite::{TruncInfo, noise_rng};
use crate::qn::Pooled;
use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space, SymSpace};

use super::{PAR_MIN_MID_SECTORS, Su2ContractedCores};

/// [`super::factor_svd`] with DMRG noise — the SU(2) twin of
/// [`crate::core::twosite::factor_svd_noise`], and see that function for the semantics, the
/// non-variational caveat, and the one-sided-seeding rationale (`|G| > 1` escape).
pub fn factor_svd_noise<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    eps: f64,
    chi: Pooled,
    normalize: bool,
    alpha: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> (Su2Core<T>, Su2Sketch<T>, Su2Core<T>, TruncInfo) {
    let space = tc.space;
    let k = tc.k;

    let metric = ThetaMetric::new(tc);

    let cap = (alpha > 0.0 && !space.all_admissible_reachable())
        .then(|| crate::su2::qn::max_ranks(&space).swap_remove(k + 1));
    let draw_dims = |m: Su2Sector| -> (usize, usize) {
        let (nr, nc) = tc.block(m).map_or((0, 0), |b| (b.nrows(), b.ncols()));
        let seedable = alpha > 0.0 && cap.as_ref().is_none_or(|c| c[m] > 0);
        match (nr, nc) {
            (r, 0) if r > 0 && seedable => (r, r),
            (0, c) if c > 0 && seedable => (c, c),
            _ => (nr, nc),
        }
    };

    let noise: Option<Vec<(Su2Sector, Mat<T>)>> = (alpha > 0.0).then(|| {
        let scaled_block = |m: Su2Sector| -> Option<Mat<T>> { metric.scaled(tc.block(m)?, m) };
        let seeds: Vec<(Su2Sector, u64)> = tc.mid_qn.iter().map(|&m| (m, rng.next_u64())).collect();
        let mut blocks: Vec<(Su2Sector, Mat<T>)> = seeds
            .iter()
            .map(|&(m, seed)| {
                let (nr, nc) = draw_dims(m);
                let mut r = Mat::<T>::zeros(nr, nc);
                let mut g = noise_rng(seed);
                for j in 0..nc {
                    for i in 0..nr {
                        *r.get_mut(i, j) = T::sample_standard(&mut g);
                    }
                }
                (m, r)
            })
            .collect();
        let mut b2 = 0.0f64;
        for &m in &tc.mid_qn {
            if let Some(sc) = scaled_block(m) {
                for j in 0..sc.ncols() {
                    for i in 0..sc.nrows() {
                        let z = sc[(i, j)];
                        b2 += (z.conj_s() * z).re_s();
                    }
                }
            }
        }
        let mut r2 = 0.0f64;
        for (_, r) in &blocks {
            for j in 0..r.ncols() {
                for i in 0..r.nrows() {
                    let z = r[(i, j)];
                    r2 += (z.conj_s() * z).re_s();
                }
            }
        }
        if r2 > 0.0 && b2 > 0.0 {
            let scale = T::from_f64(alpha * b2.sqrt() / r2.sqrt());
            for (_, r) in blocks.iter_mut() {
                for j in 0..r.ncols() {
                    for i in 0..r.nrows() {
                        let v = r[(i, j)] * scale;
                        *r.get_mut(i, j) = v;
                    }
                }
            }
        }
        blocks
    });

    let noise_idx: Option<FxHashMap<Su2Sector, usize>> = noise
        .as_ref()
        .map(|ns| ns.iter().enumerate().map(|(i, &(s, _))| (s, i)).collect());

    type SvdData<T> = Vec<(Su2Sector, Mat<T>, Vec<f64>, Mat<T>)>;
    let svd_of = |m: Su2Sector| -> (Su2Sector, Mat<T>, Vec<f64>, Mat<T>) {
        let nsm = noise_idx
            .as_ref()
            .and_then(|idx| idx.get(&m))
            .map(|&i| &noise.as_ref().unwrap()[i].1);
        let bdims = tc.block(m).map(|b| (b.nrows(), b.ncols()));
        let (u, svals, vt) = match (bdims, nsm) {
            (Some((nr, nc)), _) if nr > 0 && nc > 0 => {
                let b = tc.block(m).expect("dims came from this block");
                let mut scaled = b.to_owned();
                metric.scale_in_place(&mut scaled, m);
                if let Some(r) = nsm {
                    for j in 0..scaled.ncols() {
                        for i in 0..scaled.nrows() {
                            let v = scaled[(i, j)] + r[(i, j)];
                            *scaled.get_mut(i, j) = v;
                        }
                    }
                }
                svd_uvt_svals(scaled.as_ref())
            }
            (Some((nr, 0)), Some(r)) if nr > 0 && r.ncols() == nr => {
                let (u, svals, _vt) = svd_uvt_svals(r.as_ref());
                (u, svals, Mat::<T>::zeros(nr, 0))
            }
            (Some((0, nc)), Some(r)) if nc > 0 && r.nrows() == nc => {
                let (_u, svals, vt) = svd_uvt_svals(r.as_ref());
                (Mat::<T>::zeros(0, nc), svals, vt)
            }
            _ => (Mat::<T>::zeros(0, 0), Vec::new(), Mat::<T>::zeros(0, 0)),
        };
        (m, u, svals, vt)
    };
    let data: SvdData<T> = if tc.mid_qn.len() >= PAR_MIN_MID_SECTORS {
        tc.mid_qn.par_iter().map(|&m| svd_of(m)).collect()
    } else {
        tc.mid_qn.iter().map(|&m| svd_of(m)).collect()
    };
    let data_idx: FxHashMap<Su2Sector, usize> = data
        .iter()
        .enumerate()
        .map(|(i, (s, ..))| (*s, i))
        .collect();

    let s_full = svals_sketch::<T>(space, k + 1, |m| {
        data_idx.get(&m).map(|&i| data[i].2.as_slice())
    });
    let mid_rank = crate::su2::chop::chop_energy(&s_full, eps, chi);
    let s_norm = if normalize { s_full.norm() } else { 1.0 };

    let mut xk = Su2Core::zeros(space, k, &tc.row_ranks, &mid_rank);
    let mut xk1 = Su2Core::zeros(space, k + 1, &mid_rank, &tc.col_ranks);

    let s_out_blocks = Su2SectorGrid::from_fn(&space, |m| {
        space.contains(k + 1, m).then(|| {
            let r = mid_rank[m];
            let sv = data_idx.get(&m).map(|&i| &data[i].2);
            let mut mat = Mat::<T>::zeros(r, r);
            if let Some(v) = sv {
                for (i, &val) in v.iter().enumerate().take(r) {
                    *mat.get_mut(i, i) = T::from_f64(val / s_norm);
                }
            }
            mat
        })
    });
    let s_out = Su2Sketch::new(space, k + 1, s_out_blocks);

    for (m, u, _svals, vt) in &data {
        let r = mid_rank[*m];
        if r > 0 {
            xk.scatter_vertical(*m, u.submatrix(0, 0, u.nrows(), r));
            let mut vt_r = vt.submatrix(0, 0, r, vt.ncols()).to_owned();
            xk1.unscale_horizontal(&mut vt_r, *m);
            xk1.scatter_horizontal(*m, vt_r.as_ref());
        }
    }

    let (mut dropped2, mut total2, mut kept) = (0.0f64, 0.0f64, 0usize);
    for (m, _u, svals, _vt) in &data {
        let r = mid_rank[*m];
        kept += r;
        for (i, &s) in svals.iter().enumerate() {
            total2 += s * s;
            if i >= r {
                dropped2 += s * s;
            }
        }
    }
    let info = TruncInfo {
        discarded_weight: if total2 > 0.0 { dropped2 / total2 } else { 0.0 },
        kept,
    };

    (xk, s_out, xk1, info)
}

/// [`super::factor_svd`] with **perturbative (White density-matrix) noise** — the SU(2) twin of
/// [`crate::core::twosite::factor_svd_perturbative`], and the variational, operator-targeted
/// alternative to [`factor_svd_noise`]'s isotropic random perturbation.
#[allow(clippy::too_many_arguments)]
pub fn factor_svd_perturbative<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    noise_dm: &Su2SectorGrid<Mat<T>>,
    eps: f64,
    chi: Pooled,
    normalize: bool,
    alpha: f64,
    forward: bool,
) -> (Su2Core<T>, Su2Sketch<T>, Su2Core<T>, TruncInfo) {
    if alpha <= 0.0 {
        return factor_svd_noise(tc, eps, chi, normalize, 0.0, &mut noise_rng(0));
    }
    let space = tc.space;
    let k = tc.k;

    let metric = ThetaMetric::new(tc);
    let b_theta = |m: Su2Sector| -> Option<Mat<T>> { metric.scaled(tc.block(m)?, m) };

    let mut nd_norm2 = 0.0f64;
    for &m in &tc.mid_qn {
        let nd = &noise_dm[m];
        for j in 0..nd.ncols() {
            for i in 0..nd.nrows() {
                let z = nd[(i, j)];
                nd_norm2 += (z.conj_s() * z).re_s();
            }
        }
    }
    let nd_norm = nd_norm2.sqrt();
    let alpha_t = T::from_f64(if nd_norm > 0.0 { alpha / nd_norm } else { 0.0 });
    let cap = (!space.all_admissible_reachable())
        .then(|| crate::su2::qn::max_ranks(&space).swap_remove(k + 1));

    type EigA<T> = (Su2Sector, Mat<T>, Vec<f64>);
    let gram_eig = |m: Su2Sector| -> EigA<T> {
        let dims = tc.block(m).map(|b| (b.nrows(), b.ncols()));
        rho_prime_eig(
            m,
            dims,
            &noise_dm[m],
            forward,
            true,
            cap.as_ref(),
            alpha_t,
            |rho| {
                let bt = b_theta(m).expect("has_state implies a nonempty θ-scaled block");
                if forward {
                    matmul(
                        rho.as_mut(),
                        Accum::Replace,
                        bt.as_ref(),
                        bt.adjoint(),
                        T::one(),
                        Par::Seq,
                    );
                } else {
                    matmul(
                        rho.as_mut(),
                        Accum::Replace,
                        bt.adjoint(),
                        bt.as_ref(),
                        T::one(),
                        Par::Seq,
                    );
                }
            },
        )
    };
    let adata: Vec<EigA<T>> = if tc.mid_qn.len() >= PAR_MIN_MID_SECTORS {
        tc.mid_qn.par_iter().map(|&m| gram_eig(m)).collect()
    } else {
        tc.mid_qn.iter().map(|&m| gram_eig(m)).collect()
    };
    let adata_idx: FxHashMap<Su2Sector, usize> = adata
        .iter()
        .enumerate()
        .map(|(i, (s, ..))| (*s, i))
        .collect();

    let s_full = svals_sketch::<T>(space, k + 1, |m| {
        adata_idx.get(&m).map(|&i| adata[i].2.as_slice())
    });
    let mid_rank = crate::su2::chop::chop_energy(&s_full, eps, chi);

    let s_norm = if normalize { tc.norm() } else { 1.0 };
    let inv_norm = T::from_f64(if normalize && s_norm > 0.0 {
        1.0 / s_norm
    } else {
        1.0
    });

    let mut xk = Su2Core::zeros(space, k, &tc.row_ranks, &mid_rank);
    let mut xk1 = Su2Core::zeros(space, k + 1, &mid_rank, &tc.col_ranks);

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

    let (mut dropped2, mut total2, mut kept) = (0.0f64, 0.0f64, 0usize);
    for (m, basis, _s) in &adata {
        let r = mid_rank[*m];
        kept += r;
        let (rows, cols) = match tc.block(*m) {
            Some(b) => (b.nrows(), b.ncols()),
            None => continue,
        };
        let has_state = rows > 0 && cols > 0;
        let bt = if has_state { b_theta(*m) } else { None };
        let bt2 = bt.as_ref().map_or(0.0, |bt| frob_sq2(bt.as_ref()));
        total2 += bt2;
        if r == 0 {
            dropped2 += bt2;
            continue;
        }
        let rr = r.min(basis.ncols());
        if forward {
            let u = basis.submatrix(0, 0, rows, rr);
            xk.scatter_vertical(*m, u);
            if let Some(bt) = bt {
                let mut center = Mat::<T>::zeros(rr, cols);
                matmul(
                    center.as_mut(),
                    Accum::Replace,
                    u.adjoint(),
                    bt.as_ref(),
                    T::one(),
                    Par::Seq,
                );
                dropped2 += (bt2 - frob_sq2(center.as_ref())).max(0.0);
                for j in 0..cols {
                    for i in 0..rr {
                        let v = center[(i, j)] * inv_norm;
                        *center.get_mut(i, j) = v;
                    }
                }
                xk1.unscale_horizontal(&mut center, *m);
                xk1.scatter_horizontal(*m, center.as_ref());
            }
        } else {
            let v = basis.submatrix(0, 0, cols, rr);
            let mut vt_out = Mat::<T>::zeros(rr, cols);
            for i in 0..rr {
                for j in 0..cols {
                    *vt_out.get_mut(i, j) = v[(j, i)].conj_s();
                }
            }
            xk1.unscale_horizontal(&mut vt_out, *m);
            xk1.scatter_horizontal(*m, vt_out.as_ref());
            if let Some(bt) = bt {
                let mut center = Mat::<T>::zeros(rows, rr);
                matmul(
                    center.as_mut(),
                    Accum::Replace,
                    bt.as_ref(),
                    v,
                    T::one(),
                    Par::Seq,
                );
                dropped2 += (bt2 - frob_sq2(center.as_ref())).max(0.0);
                for j in 0..rr {
                    for i in 0..rows {
                        let val = center[(i, j)] * inv_norm;
                        *center.get_mut(i, j) = val;
                    }
                }
                xk.scatter_vertical(*m, center.as_ref());
            }
        }
    }
    let info = TruncInfo {
        discarded_weight: if total2 > 0.0 { dropped2 / total2 } else { 0.0 },
        kept,
    };

    (xk, s_out, xk1, info)
}

pub(super) struct ThetaMetric<T: Scalar> {
    shell: Su2Core<T>,
}

impl<T: Scalar> ThetaMetric<T> {
    pub(super) fn new(tc: &Su2ContractedCores<T>) -> Self {
        let space = tc.space;
        ThetaMetric {
            shell: Su2Core::<T>::zeros(space, tc.k + 1, &Su2RankGrid::zeros(&space), &tc.col_ranks),
        }
    }

    pub(super) fn scale_in_place(&self, b: &mut Mat<T>, m: Su2Sector) {
        self.shell.scale_horizontal(b, m);
    }

    pub(super) fn scaled(&self, b: MatRef<'_, T>, m: Su2Sector) -> Option<Mat<T>> {
        if b.nrows() == 0 || b.ncols() == 0 {
            return None;
        }
        let mut sc = b.to_owned();
        self.scale_in_place(&mut sc, m);
        Some(sc)
    }
}

pub(super) fn svals_sketch<'a, T: Scalar>(
    space: Su2Space,
    bond: usize,
    svals: impl Fn(Su2Sector) -> Option<&'a [f64]>,
) -> Su2Sketch<T> {
    let blocks = Su2SectorGrid::from_fn(&space, |m| {
        space.contains(bond, m).then(|| {
            let sv = svals(m);
            let size = sv.map_or(0, |v| v.len());
            let mut mat = Mat::<T>::zeros(size, size);
            if let Some(v) = sv {
                for (i, &val) in v.iter().enumerate() {
                    *mat.get_mut(i, i) = T::from_f64(val);
                }
            }
            mat
        })
    });
    Su2Sketch::new(space, bond, blocks)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rho_prime_eig<T: Scalar>(
    m: Su2Sector,
    dims: Option<(usize, usize)>,
    nd: &Mat<T>,
    forward: bool,
    noise_on: bool,
    cap: Option<&Su2RankGrid>,
    alpha_t: T,
    state_gram: impl FnOnce(&mut Mat<T>),
) -> (Su2Sector, Mat<T>, Vec<f64>) {
    let empty = |m: Su2Sector| (m, Mat::<T>::zeros(0, 0), Vec::new());
    let Some((rows, cols)) = dims else {
        assert!(
            nd.nrows() == 0 && nd.ncols() == 0,
            "noise_dm[{m:?}] is {}x{} but the two-site block is absent: the enrichment \
             would be silently dropped and `alpha_eff` diluted for every other sector",
            nd.nrows(),
            nd.ncols(),
        );
        return empty(m);
    };
    let g = if forward { rows } else { cols };
    if g == 0 {
        assert!(
            nd.nrows() == 0 && nd.ncols() == 0,
            "noise_dm[{m:?}] is {}x{} but the split side has no basis room (g = 0)",
            nd.nrows(),
            nd.ncols(),
        );
        return empty(m);
    }
    let has_state = rows > 0 && cols > 0;
    assert!(
        (nd.nrows() == 0 && nd.ncols() == 0) || (nd.nrows() == g && nd.ncols() == g),
        "noise_dm[{m:?}] is {}x{}, want {g}x{g} or absent (forward = {forward})",
        nd.nrows(),
        nd.ncols(),
    );
    let has_noise = noise_on && nd.nrows() == g && nd.ncols() == g;
    if !has_state && (!has_noise || cap.is_some_and(|c| c[m] == 0)) {
        return empty(m);
    }
    let mut rho = Mat::<T>::zeros(g, g);
    if has_state {
        state_gram(&mut rho);
    }
    if has_noise {
        for j in 0..g {
            for i in 0..g {
                let v = rho[(i, j)] + alpha_t * nd[(i, j)];
                *rho.get_mut(i, j) = v;
            }
        }
    }
    let (basis, svals) = descending_eig_floored(rho.as_ref());
    (m, basis, svals)
}

fn descending_eig_floored<T: Scalar>(rho: MatRef<'_, T>) -> (Mat<T>, Vec<f64>) {
    let g = rho.nrows();
    let (evals, u) = dense_self_adjoint_eigen(rho);
    let lam_max = evals.last().copied().unwrap_or(0.0).max(0.0);
    let floor = lam_max * (g as f64) * f64::EPSILON;
    let mut basis = Mat::<T>::zeros(g, g);
    let mut svals = Vec::with_capacity(g);
    for c in 0..g {
        let src = g - 1 - c;
        for r in 0..g {
            *basis.get_mut(r, c) = u[(r, src)];
        }
        svals.push(if evals[src] > floor {
            evals[src].sqrt()
        } else {
            0.0
        });
    }
    (basis, svals)
}

pub(super) fn frob_sq2<T: Scalar>(a: MatRef<'_, T>) -> f64 {
    let mut acc = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            let z = a[(i, j)];
            acc += (z.conj_s() * z).re_s();
        }
    }
    acc
}
