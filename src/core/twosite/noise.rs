// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::SparseCore;
use crate::core::factor::{Sketch, chop_energy, dense_self_adjoint_eigen, svd_uvt_svals};
use crate::qn::{Pooled, QnSpace, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;

use super::ContractedSparseCores;

/// What a truncation discarded, for the caller's convergence bookkeeping (DMRG's *discarded
/// weight*, the standard rank-convergence diagnostic: it is the quantity a production run
/// extrapolates to zero).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TruncInfo {
    /// `Σ_{dropped} σ² / Σ_{all} σ²` at this bond — relative, so it is comparable across bonds
    /// and sweeps. Zero when nothing was dropped.
    pub discarded_weight: f64,
    /// Total kept rank summed over sectors (the conventional DMRG bond dimension at this bond).
    pub kept: usize,
}

/// The RNG the noise path draws from. A concrete type (not a generic `R: Rng`) because the noise
/// has to be pre-drawn *serially* to stay independent of rayon's scheduling — see the seed comment
/// in [`factor_svd_noise`]. Matches the crate's settled RNG convention: explicit seeds, never a
/// global generator.
pub fn noise_rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

/// [`super::factor_svd`] with DMRG *noise*: before the SVD, each mid-sector block is perturbed by a
/// random matrix scaled so the whole perturbation has Frobenius norm `alpha · ‖B‖_F`, where
/// `‖B‖_F` is the norm of the **entire** two-site block. Scaling by the global norm rather than
/// each sector's own is deliberate: per-sector scaling would give a nearly-empty sector a
/// relatively enormous perturbation, which is the opposite of the intent.
pub fn factor_svd_noise<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    eps: f64,
    chi: Pooled,
    normalize: bool,
    alpha: f64,
    rng: &mut Xoshiro256PlusPlus,
) -> (SparseCore<T>, Sketch<T>, SparseCore<T>, TruncInfo) {
    let space = tc.space;
    let k = tc.k;

    let cap = (alpha > 0.0 && !space.pg.is_trivial())
        .then(|| crate::qn::max_ranks(&space).swap_remove(k + 1));
    let draw_dims = |m: Sector| -> (usize, usize) {
        let (nr, nc) = tc.block(m).map_or((0, 0), |b| (b.nrows(), b.ncols()));
        let seedable = alpha > 0.0 && cap.as_ref().is_none_or(|c| c[m] > 0);
        match (nr, nc) {
            (r, 0) if r > 0 && seedable => (r, r),
            (0, c) if c > 0 && seedable => (c, c),
            _ => (nr, nc),
        }
    };

    let noise: Option<Vec<(Sector, Mat<T>)>> = (alpha > 0.0).then(|| {
        let seeds: Vec<(Sector, u64)> = tc.mid_qn.iter().map(|&m| (m, rng.next_u64())).collect();
        let mut blocks: Vec<(Sector, Mat<T>)> = seeds
            .par_iter()
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

        let b_norm = frob_norm(tc.mid_qn.iter().filter_map(|&m| tc.block(m)));
        let r_norm = frob_norm(blocks.iter().map(|(_, r)| r.as_ref()));
        if r_norm > 0.0 && b_norm > 0.0 {
            let scale = T::from_f64(alpha * b_norm / r_norm);
            for (_, r) in blocks.iter_mut() {
                let (nr, nc) = (r.nrows(), r.ncols());
                for j in 0..nc {
                    for i in 0..nr {
                        let v = r[(i, j)] * scale;
                        *r.get_mut(i, j) = v;
                    }
                }
            }
        }
        blocks
    });
    let noise_idx: Option<FxHashMap<Sector, usize>> = noise
        .as_ref()
        .map(|ns| ns.iter().enumerate().map(|(i, &(s, _))| (s, i)).collect());

    type SvdData<T> = Vec<(Sector, Mat<T>, Vec<f64>, Mat<T>)>;
    let data: SvdData<T> = tc
        .mid_qn
        .par_iter()
        .map(|&m| {
            let empty = || (Mat::<T>::zeros(0, 0), Vec::new(), Mat::<T>::zeros(0, 0));
            let bdims = tc.block(m).map(|b| (b.nrows(), b.ncols()));
            let nsm = noise_idx
                .as_ref()
                .and_then(|idx| idx.get(&m))
                .map(|&i| &noise.as_ref().unwrap()[i].1);
            let (u, svals, vt) = match (bdims, nsm) {
                (Some((nr, nc)), Some(r)) if nr > 0 && nc > 0 => {
                    let b = tc.block(m).expect("dims came from this block");
                    let mut p = b.to_owned();
                    for j in 0..nc {
                        for i in 0..nr {
                            let v = p[(i, j)] + r[(i, j)];
                            *p.get_mut(i, j) = v;
                        }
                    }
                    svd_uvt_svals(p.as_ref())
                }
                (Some((nr, 0)), Some(r)) if nr > 0 && r.ncols() == nr => {
                    let (u, svals, _vt) = svd_uvt_svals(r.as_ref());
                    (u, svals, Mat::<T>::zeros(nr, 0))
                }
                (Some((0, nc)), Some(r)) if nc > 0 && r.nrows() == nc => {
                    let (_u, svals, vt) = svd_uvt_svals(r.as_ref());
                    (Mat::<T>::zeros(0, nc), svals, vt)
                }
                (Some((nr, nc)), _) if nr > 0 && nc > 0 => {
                    svd_uvt_svals(tc.block(m).expect("dims came from this block"))
                }
                _ => empty(),
            };
            (m, u, svals, vt)
        })
        .collect();
    let data_idx: FxHashMap<Sector, usize> = data
        .iter()
        .enumerate()
        .map(|(i, (s, ..))| (*s, i))
        .collect();

    let s_full = svals_sketch::<T>(space, k + 1, |m| {
        data_idx.get(&m).map(|&i| data[i].2.as_slice())
    });
    let mid_rank = chop_energy(&s_full, eps, chi);
    let s_norm = if normalize { s_full.norm() } else { 1.0 };

    let mut xk = SparseCore::zeros(space, k, &tc.row_ranks, &mid_rank);
    let mut xk1 = SparseCore::zeros(space, k + 1, &mid_rank, &tc.col_ranks);

    let s_out_blocks = SectorGrid::from_fn(&space, |m| {
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
    let s_out = Sketch::new(space, k + 1, s_out_blocks);

    for (m, u, _svals, vt) in &data {
        let r = mid_rank[*m];
        if r > 0 {
            xk.scatter_vertical(*m, u.submatrix(0, 0, u.nrows(), r));
            xk1.scatter_horizontal(*m, vt.submatrix(0, 0, r, vt.ncols()));
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

pub(super) fn frob_norm<'a, T: Scalar>(blocks: impl Iterator<Item = MatRef<'a, T>>) -> f64 {
    blocks.map(frob_sq).sum::<f64>().sqrt()
}

pub(super) fn frob_sq<T: Scalar>(b: MatRef<'_, T>) -> f64 {
    let mut acc = 0.0f64;
    for j in 0..b.ncols() {
        for i in 0..b.nrows() {
            let z = b[(i, j)];
            acc += (z.conj_s() * z).re_s();
        }
    }
    acc
}

/// [`super::factor_svd`] with **perturbative (White-style) noise** via the true density-matrix
/// noise recipe: the variational, operator-targeted alternative to [`factor_svd_noise`]'s isotropic
/// random perturbation.
#[allow(clippy::too_many_arguments)]
pub fn factor_svd_perturbative<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    noise_dm: &SectorGrid<Mat<T>>,
    eps: f64,
    chi: Pooled,
    normalize: bool,
    alpha: f64,
    forward: bool,
) -> (SparseCore<T>, Sketch<T>, SparseCore<T>, TruncInfo) {
    if alpha <= 0.0 {
        return factor_svd_noise(tc, eps, chi, normalize, 0.0, &mut noise_rng(0));
    }
    let space = tc.space;
    let k = tc.k;
    let nd_norm = frob_norm(tc.mid_qn.iter().map(|&m| noise_dm[m].as_ref()));
    let alpha_t = T::from_f64(if nd_norm > 0.0 { alpha / nd_norm } else { 0.0 });
    let cap = (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space).swap_remove(k + 1));

    type EigA<T> = (Sector, Mat<T>, Vec<f64>);
    let gram_eig = |m: Sector| -> EigA<T> {
        let block = tc.block(m);
        let dims = block.map(|b| (b.nrows(), b.ncols()));
        rho_prime_eig(
            m,
            dims,
            &noise_dm[m],
            forward,
            true,
            cap.as_ref(),
            alpha_t,
            |rho| {
                let b = block.expect("has_state implies the block is present");
                if forward {
                    matmul(
                        rho.as_mut(),
                        Accum::Replace,
                        b,
                        b.adjoint(),
                        T::one(),
                        Par::Seq,
                    );
                } else {
                    matmul(
                        rho.as_mut(),
                        Accum::Replace,
                        b.adjoint(),
                        b,
                        T::one(),
                        Par::Seq,
                    );
                }
            },
        )
    };
    let adata: Vec<EigA<T>> = if tc.mid_qn.len() >= 4 {
        tc.mid_qn.par_iter().map(|&m| gram_eig(m)).collect()
    } else {
        tc.mid_qn.iter().map(|&m| gram_eig(m)).collect()
    };
    let adata_idx: FxHashMap<Sector, usize> = adata
        .iter()
        .enumerate()
        .map(|(i, (s, ..))| (*s, i))
        .collect();

    let s_full = svals_sketch::<T>(space, k + 1, |m| {
        adata_idx.get(&m).map(|&i| adata[i].2.as_slice())
    });
    let mid_rank = chop_energy(&s_full, eps, chi);

    let s_norm = if normalize {
        frob_norm(tc.mid_qn.iter().filter_map(|&m| tc.block(m)))
    } else {
        1.0
    };
    let inv_norm = T::from_f64(if normalize && s_norm > 0.0 {
        1.0 / s_norm
    } else {
        1.0
    });

    let mut xk = SparseCore::zeros(space, k, &tc.row_ranks, &mid_rank);
    let mut xk1 = SparseCore::zeros(space, k + 1, &mid_rank, &tc.col_ranks);

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

    let (mut dropped2, mut total2, mut kept) = (0.0f64, 0.0f64, 0usize);
    for (m, basis, _s) in &adata {
        let r = mid_rank[*m];
        kept += r;
        let Some(b) = tc.block(*m) else { continue };
        let (rows, cols) = (b.nrows(), b.ncols());
        let has_state = rows > 0 && cols > 0;
        if has_state {
            total2 += frob_sq(b);
        }
        if r == 0 {
            if has_state {
                dropped2 += frob_sq(b);
            }
            continue;
        }
        let rr = r.min(basis.ncols());
        if forward {
            let u = basis.submatrix(0, 0, rows, rr);
            xk.scatter_vertical(*m, u);
            if has_state {
                let mut center = Mat::<T>::zeros(rr, cols);
                matmul(
                    center.as_mut(),
                    Accum::Replace,
                    u.adjoint(),
                    b,
                    T::one(),
                    Par::Seq,
                );
                dropped2 += (frob_sq(b) - frob_sq(center.as_ref())).max(0.0);
                scale_mat(&mut center, inv_norm);
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
            xk1.scatter_horizontal(*m, vt_out.as_ref());
            if has_state {
                let mut center = Mat::<T>::zeros(rows, rr);
                matmul(center.as_mut(), Accum::Replace, b, v, T::one(), Par::Seq);
                dropped2 += (frob_sq(b) - frob_sq(center.as_ref())).max(0.0);
                scale_mat(&mut center, inv_norm);
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

pub(super) fn scale_mat<T: Scalar>(m: &mut Mat<T>, a: T) {
    let (nr, nc) = (m.nrows(), m.ncols());
    for j in 0..nc {
        for i in 0..nr {
            let v = m[(i, j)] * a;
            *m.get_mut(i, j) = v;
        }
    }
}

pub(super) fn svals_sketch<'a, T: Scalar>(
    space: QnSpace,
    bond: usize,
    svals: impl Fn(Sector) -> Option<&'a [f64]>,
) -> Sketch<T> {
    let blocks = SectorGrid::from_fn(&space, |m| {
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
    Sketch::new(space, bond, blocks)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn rho_prime_eig<T: Scalar>(
    m: Sector,
    dims: Option<(usize, usize)>,
    nd: &Mat<T>,
    forward: bool,
    noise_on: bool,
    cap: Option<&RankGrid>,
    alpha_t: T,
    state_gram: impl FnOnce(&mut Mat<T>),
) -> (Sector, Mat<T>, Vec<f64>) {
    let empty = |m: Sector| (m, Mat::<T>::zeros(0, 0), Vec::new());
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

#[cfg(test)]
mod shape_invariant_tests {

    use crate::core::twosite::{contract, factor_svd_perturbative};
    use crate::fcidump::{TwoBodySymmetry, read_fcidump};
    use crate::qn::SectorGrid;
    use crate::{QnSpace, RankGrid, TtVector, tt_randn};
    use faer::Mat;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn h4_two_site() -> (crate::core::twosite::ContractedSparseCores<f64>, QnSpace) {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP");
        let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).expect("read h4");
        let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
        let ranks: Vec<RankGrid> = (0..=space.d)
            .map(|b| {
                RankGrid::from_fn(&space, |s| {
                    if space.contains(b, s) {
                        (space.max_rank(b, s) as usize).min(3)
                    } else {
                        0
                    }
                })
            })
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let x: TtVector<f64> = tt_randn::<f64, _>(space, &ranks, &mut rng);
        (contract(x.core(0), x.core(1)), space)
    }

    #[test]
    fn correctly_shaped_noise_dm_is_accepted() {
        let (tc, space) = h4_two_site();
        let nd = SectorGrid::from_fn(&space, |m| match tc.block(m) {
            Some(b) if b.nrows() > 0 => Mat::<f64>::zeros(b.nrows(), b.nrows()),
            _ => Mat::<f64>::zeros(0, 0),
        });
        let (_a, _s, _b, info) =
            factor_svd_perturbative(&tc, &nd, 1e-10, crate::qn::Pooled::NONE, true, 1e-4, true);
        assert!(info.kept > 0, "the split kept no rank at all");
    }

    #[test]
    #[should_panic(expected = "want")]
    fn mis_sized_noise_dm_trips_the_invariant() {
        let (tc, space) = h4_two_site();
        let mut corrupted = false;
        let nd = SectorGrid::from_fn(&space, |m| match tc.block(m) {
            Some(b) if b.nrows() > 0 && !corrupted => {
                corrupted = true;
                Mat::<f64>::zeros(b.nrows() + 1, b.nrows() + 1)
            }
            Some(b) if b.nrows() > 0 => Mat::<f64>::zeros(b.nrows(), b.nrows()),
            _ => Mat::<f64>::zeros(0, 0),
        });
        assert!(
            corrupted,
            "fixture produced no populated mid sector to corrupt"
        );
        let _ = factor_svd_perturbative(&tc, &nd, 1e-10, crate::qn::Pooled::NONE, true, 1e-4, true);
    }
}

#[cfg(test)]
mod schedule_tests {
    use crate::core::twosite::{contract, factor_svd, factor_svd_noise, noise_rng};
    use crate::fcidump::{TwoBodySymmetry, read_fcidump};
    use crate::{QnSpace, RankGrid, SparseHamiltonian, TtVector, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn h4() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP")
    }

    fn h4_system(r: usize) -> (SparseHamiltonian<f64>, TtVector<f64>, QnSpace) {
        let data = read_fcidump::<f64>(&h4(), TwoBodySymmetry::Eightfold).expect("read h4");
        let space = QnSpace::from_n_two_sz(data.norb, data.nelec, data.ms2);
        let h = SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
        let ranks: Vec<RankGrid> = (0..=space.d)
            .map(|b| {
                RankGrid::from_fn(&space, |s| {
                    if space.contains(b, s) {
                        (space.max_rank(b, s) as usize).min(r)
                    } else {
                        0
                    }
                })
            })
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let x0 = tt_randn::<f64, _>(space, &ranks, &mut rng);
        (h, x0, space)
    }

    fn sec_list(x: &TtVector<f64>, bond: usize) -> Vec<crate::Sector> {
        x.space.bond_sectors(bond)
    }

    #[test]
    fn zero_noise_factor_svd_is_bit_identical() {
        let (_h, x, _space) = h4_system(3);
        let mut rng = noise_rng(99);

        for k in 0..(x.space.d - 1) {
            let cp = contract(x.core(k), x.core(k + 1));
            let (a0, s0, b0) = factor_svd(&cp, 1e-10, crate::qn::Pooled::NONE, true);
            let (a1, s1, b1, info) =
                factor_svd_noise(&cp, 1e-10, crate::qn::Pooled::NONE, true, 0.0, &mut rng);

            for (f, sec) in [
                (crate::Family::Empty, ()),
                (crate::Family::Up, ()),
                (crate::Family::Dn, ()),
                (crate::Family::Occ, ()),
            ]
            .map(|(f, _)| (f, sec_list(&x, k)))
            {
                for s in sec {
                    match (a0.block(f, s), a1.block(f, s)) {
                        (Some(m0), Some(m1)) => {
                            assert_eq!(m0.nrows(), m1.nrows());
                            assert_eq!(m0.ncols(), m1.ncols());
                            for j in 0..m0.ncols() {
                                for i in 0..m0.nrows() {
                                    assert_eq!(
                                        m0[(i, j)],
                                        m1[(i, j)],
                                        "site {k} left core differs"
                                    );
                                }
                            }
                        }
                        (None, None) => {}
                        _ => panic!("block presence differs at site {k}"),
                    }
                }
            }
            for s in sec_list(&x, k + 1) {
                match (s0.block(s), s1.block(s)) {
                    (Some(m0), Some(m1)) => {
                        for j in 0..m0.ncols() {
                            for i in 0..m0.nrows() {
                                assert_eq!(m0[(i, j)], m1[(i, j)], "singular values differ");
                            }
                        }
                    }
                    (None, None) => {}
                    _ => panic!("sketch presence differs"),
                }
            }
            assert_eq!(b0.row_ranks, b1.row_ranks, "right core ranks differ");

            assert!(
                info.discarded_weight < 1e-12,
                "unexpected discard: {}",
                info.discarded_weight
            );
            assert!(info.kept > 0);
        }
    }
}
