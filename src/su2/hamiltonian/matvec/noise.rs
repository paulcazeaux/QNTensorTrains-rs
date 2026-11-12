// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use rustc_hash::FxHashMap;

use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, Ts};
use crate::su2::qn::{Su2Sector, Su2SectorGrid};

/// Per-mid-sector **noise density matrix** `ρ_noise(s) = Σ_w (L_w·B_θ)(L_w·B_θ)ᴴ / ‖L_w·B_θ‖²` —
/// the SU(2) twin of [`crate::hamiltonian::matvec::noise_density_matrix`], built entirely in the
/// θ-scaled metric so [`crate::su2::twosite::factor_svd_perturbative`] can form
/// `ρ' = B_θ·B_θᴴ + α_eff·ρ_noise` directly.
pub fn noise_density_matrix<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    half: &Su2Core<T>,
    xk: &Su2Core<T>,
    xk1: &Su2Core<T>,
    forward: bool,
) -> Su2SectorGrid<Mat<T>> {
    if forward {
        noise_density_matrix_from(h, xk, xk1, true, &half.col_qn, &|s| {
            half.vertical_unfolding(s)
        })
    } else {
        noise_density_matrix_from(h, xk, xk1, false, &half.row_qn, &|s| {
            let mut e = half.horizontal_unfolding(s);
            half.scale_horizontal(&mut e, s);
            e
        })
    }
}

/// [`noise_density_matrix`] WITHOUT the wide half — SU(2) twin of the abelian
/// `noise_density_matrix_streamed` (memory campaign, L2b). Each output sector's slab is the
/// `Family::ALL`-ordered concatenation of the ≤ 4 blocks the block kernels produce for it (the
/// same blocks `sketch_h_mul` / `h_sketch_mul` fill), rebuilt from the environment on demand —
/// twice per sector, one pass each; the back slab carries the same θ column scaling
/// `Su2Core::scale_horizontal` applies to the materialized unfolding. Bit-identical to the
/// materialized route (pinned by `l2b_streamed_noise_tests`). `env` is `Fᴸ` at bond `k`
/// (forward) or `Fᴿ` at bond `k+2` (back).
pub fn noise_density_matrix_streamed<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    env: &crate::su2::frame::Su2Sketch<T>,
    xk: &Su2Core<T>,
    xk1: &Su2Core<T>,
    forward: bool,
) -> Su2SectorGrid<Mat<T>> {
    use crate::su2::core::Family;
    let space = xk.space;
    let k = xk.k;
    if forward {
        debug_assert_eq!(env.site(), k);
        let g = space.pg.irrep_of(k);
        let il = h.row_ranges(xk);
        let jl = h.col_ranges(xk);
        let rows = env.row_ranks();
        let cols = h.col_ranks(xk);
        let sectors = space.bond_sectors(k + 1);
        let slab = |s: Su2Sector| -> Mat<T> {
            let n = cols[s];
            let preds: Vec<(Family, Su2Sector)> = Family::ALL
                .into_iter()
                .filter_map(|f| f.unapply(s, g).map(|l| (f, l)))
                .filter(|&(_, l)| space.contains(k, l))
                .collect();
            let height: usize = preds.iter().map(|&(_, l)| rows[l]).sum();
            let mut out = Mat::<T>::zeros(height, n);
            let mut off = 0usize;
            for (f, l) in preds {
                let hgt = rows[l];
                if hgt > 0 && n > 0 {
                    h.sketch_h_mul_block_into(
                        out.as_mut().submatrix_mut(off, 0, hgt, n),
                        env,
                        xk,
                        f,
                        l,
                        &il,
                        &jl,
                    );
                }
                off += hgt;
            }
            if crate::hamiltonian::instrument::ENABLED {
                crate::hamiltonian::instrument::mem_hw(
                    crate::hamiltonian::instrument::memslot::NOISE_HALF,
                    crate::mem::block_bytes::<T>(height, n),
                );
            }
            out
        };
        noise_density_matrix_from(h, xk, xk1, true, &sectors, &slab)
    } else {
        let k1 = xk1.k;
        debug_assert_eq!(env.site(), k1 + 1);
        let g = space.pg.irrep_of(k1);
        let ir = h.row_ranges(xk1);
        let jr = h.col_ranges(xk1);
        let rows = h.row_ranks(xk1);
        let cols = env.col_ranks();
        let sectors = space.bond_sectors(k1);
        let slab = |s: Su2Sector| -> Mat<T> {
            let m = rows[s];
            let succs: Vec<(Family, Su2Sector)> = Family::ALL
                .into_iter()
                .filter_map(|f| f.apply(s, g).map(|t| (f, t)))
                .filter(|&(_, t)| space.contains(k1 + 1, t))
                .collect();
            let width: usize = succs.iter().map(|&(_, t)| cols[t]).sum();
            let mut out = Mat::<T>::zeros(m, width);
            let mut off = 0usize;
            for (f, t) in succs {
                let w = cols[t];
                if m > 0 && w > 0 {
                    h.h_sketch_mul_block_into(
                        out.as_mut().submatrix_mut(0, off, m, w),
                        xk1,
                        env,
                        f,
                        s,
                        &ir,
                        &jr,
                    );
                }
                let theta = crate::su2::theta(s.two_s, t.two_s);
                if theta != 1.0 {
                    let factor = T::from_f64(theta);
                    for j in off..off + w {
                        for i in 0..m {
                            *out.get_mut(i, j) *= factor;
                        }
                    }
                }
                off += w;
            }
            if crate::hamiltonian::instrument::ENABLED {
                crate::hamiltonian::instrument::mem_hw(
                    crate::hamiltonian::instrument::memslot::NOISE_HALF,
                    crate::mem::block_bytes::<T>(m, width),
                );
            }
            out
        };
        noise_density_matrix_from(h, xk, xk1, false, &sectors, &slab)
    }
}

fn noise_density_matrix_from<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    xk: &Su2Core<T>,
    xk1: &Su2Core<T>,
    forward: bool,
    sectors: &[Su2Sector],
    slab: &dyn Fn(Su2Sector) -> Mat<T>,
) -> Su2SectorGrid<Mat<T>> {
    let space = xk.space;
    let mut out = Su2SectorGrid::from_fn(&space, |_| Mat::<T>::zeros(0, 0));
    let mut pgram: Su2SectorGrid<Option<Mat<T>>> = Su2SectorGrid::from_fn(&space, |_| None);
    let mut group_sq: FxHashMap<Ts, f64> = FxHashMap::default();
    if forward {
        let k = xk.k;
        let ranges = h.col_ranges(xk);
        for &s in sectors {
            let e_raw = slab(s);
            let rows = e_raw.nrows();
            if rows == 0 || e_raw.ncols() == 0 {
                continue;
            }
            let m_list = h.state_qns_at(k + 1, s);
            let descs = h.channel_descs(k + 1, s);
            debug_assert_eq!(
                m_list.len(),
                descs.len(),
                "state_qns_at/channel_descs desync at interior bond {}",
                k + 1
            );
            let rs = &ranges[s];
            for (c, &m_c) in m_list.iter().enumerate() {
                let rng = &rs[c];
                let w = rng.len();
                if w == 0 {
                    continue;
                }
                if pgram[m_c].is_none() {
                    let mut r = xk1.horizontal_unfolding(m_c);
                    let mut p = Mat::<T>::zeros(w, w);
                    if r.nrows() == w && r.ncols() > 0 {
                        xk1.scale_horizontal(&mut r, m_c);
                        matmul(
                            p.as_mut(),
                            Accum::Replace,
                            r.as_ref(),
                            r.adjoint(),
                            T::one(),
                            Par::Seq,
                        );
                    }
                    pgram[m_c] = Some(p);
                }
                let p = pgram[m_c].as_ref().unwrap();
                if p.nrows() != w {
                    continue;
                }
                let sigma_sc = (m_c.two_s as f64 + 1.0) / (s.two_s as f64 + 1.0);
                let slab = e_raw.submatrix(0, rng.start, rows, w);
                let mut tmp = Mat::<T>::zeros(rows, w);
                matmul(
                    tmp.as_mut(),
                    Accum::Replace,
                    slab,
                    p.as_ref(),
                    T::one(),
                    Par::Seq,
                );
                *group_sq.entry(descs[c]).or_insert(0.0) +=
                    sigma_sc * channel_sq(tmp.as_ref(), slab);
            }
        }
        let floor = group_floor(&group_sq);
        for &s in sectors {
            let e_raw = slab(s);
            let rows = e_raw.nrows();
            if rows == 0 || e_raw.ncols() == 0 {
                continue;
            }
            let m_list = h.state_qns_at(k + 1, s);
            let descs = h.channel_descs(k + 1, s);
            let rs = &ranges[s];
            let mut rho = Mat::<T>::zeros(rows, rows);
            let mut any = false;
            for (c, &m_c) in m_list.iter().enumerate() {
                let rng = &rs[c];
                let w = rng.len();
                if w == 0 {
                    continue;
                }
                let gs = group_sq.get(&descs[c]).copied().unwrap_or(0.0);
                if gs <= floor {
                    continue;
                }
                let p = match pgram[m_c].as_ref() {
                    Some(p) if p.nrows() == w => p,
                    _ => continue,
                };
                let sigma_sc = (m_c.two_s as f64 + 1.0) / (s.two_s as f64 + 1.0);
                let slab = e_raw.submatrix(0, rng.start, rows, w);
                let mut tmp = Mat::<T>::zeros(rows, w);
                matmul(
                    tmp.as_mut(),
                    Accum::Replace,
                    slab,
                    p.as_ref(),
                    T::one(),
                    Par::Seq,
                );
                matmul(
                    rho.as_mut(),
                    Accum::Add,
                    tmp.as_ref(),
                    slab.adjoint(),
                    T::from_f64(sigma_sc / gs),
                    Par::Seq,
                );
                any = true;
            }
            if any {
                out[s] = rho;
            }
        }
    } else {
        let k1 = xk1.k;
        let ranges = h.row_ranges(xk1);
        for &s in sectors {
            let e_raw = slab(s);
            let cols = e_raw.ncols();
            if cols == 0 || e_raw.nrows() == 0 {
                continue;
            }
            let m_list = h.state_qns_at(k1, s);
            let descs = h.channel_descs(k1, s);
            debug_assert_eq!(
                m_list.len(),
                descs.len(),
                "state_qns_at/channel_descs desync at interior bond {k1}"
            );
            let rs = &ranges[s];
            for (c, &m_c) in m_list.iter().enumerate() {
                let rng = &rs[c];
                let w = rng.len();
                if w == 0 {
                    continue;
                }
                if pgram[m_c].is_none() {
                    let l = xk.vertical_unfolding(m_c);
                    let mut p = Mat::<T>::zeros(w, w);
                    if l.ncols() == w && l.nrows() > 0 {
                        matmul(
                            p.as_mut(),
                            Accum::Replace,
                            l.adjoint(),
                            l.as_ref(),
                            T::one(),
                            Par::Seq,
                        );
                    }
                    pgram[m_c] = Some(p);
                }
                let p = pgram[m_c].as_ref().unwrap();
                if p.nrows() != w {
                    continue;
                }
                let slab = e_raw.submatrix(rng.start, 0, w, cols);
                let mut tmp = Mat::<T>::zeros(w, cols);
                matmul(
                    tmp.as_mut(),
                    Accum::Replace,
                    p.as_ref(),
                    slab,
                    T::one(),
                    Par::Seq,
                );
                *group_sq.entry(descs[c]).or_insert(0.0) += channel_sq(tmp.as_ref(), slab);
            }
        }
        let floor = group_floor(&group_sq);
        for &s in sectors {
            let e_raw = slab(s);
            let cols = e_raw.ncols();
            if cols == 0 || e_raw.nrows() == 0 {
                continue;
            }
            let m_list = h.state_qns_at(k1, s);
            let descs = h.channel_descs(k1, s);
            let rs = &ranges[s];
            let mut rho = Mat::<T>::zeros(cols, cols);
            let mut any = false;
            for (c, &m_c) in m_list.iter().enumerate() {
                let rng = &rs[c];
                let w = rng.len();
                if w == 0 {
                    continue;
                }
                let gs = group_sq.get(&descs[c]).copied().unwrap_or(0.0);
                if gs <= floor {
                    continue;
                }
                let p = match pgram[m_c].as_ref() {
                    Some(p) if p.nrows() == w => p,
                    _ => continue,
                };
                let slab = e_raw.submatrix(rng.start, 0, w, cols);
                let mut tmp = Mat::<T>::zeros(w, cols);
                matmul(
                    tmp.as_mut(),
                    Accum::Replace,
                    p.as_ref(),
                    slab,
                    T::one(),
                    Par::Seq,
                );
                matmul(
                    rho.as_mut(),
                    Accum::Add,
                    slab.adjoint(),
                    tmp.as_ref(),
                    T::from_f64(1.0 / gs),
                    Par::Seq,
                );
                any = true;
            }
            if any {
                out[s] = rho;
            }
        }
    }
    out
}

#[inline]
fn channel_sq<T: Scalar>(tmp: MatRef<'_, T>, slab: MatRef<'_, T>) -> f64 {
    let mut acc = 0.0;
    for j in 0..tmp.ncols() {
        for i in 0..tmp.nrows() {
            acc += (tmp[(i, j)] * slab[(i, j)].conj_s()).re_s();
        }
    }
    acc
}

#[inline]
fn group_floor(group_sq: &FxHashMap<Ts, f64>) -> f64 {
    let max_g = group_sq.values().copied().fold(0.0, f64::max);
    max_g * f64::EPSILON
}

#[cfg(test)]
mod l2b_streamed_noise_tests {
    use super::{noise_density_matrix, noise_density_matrix_streamed};
    use crate::su2::hamiltonian::matvec::{h_sketch_mul, sketch_h_mul};
    use crate::su2::qn::Su2Space;
    use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::su2::twosite::{contract, factor_qc};
    use crate::test_util::{generic_su2_h, randomized_su2_state, seeded};

    #[test]
    fn streamed_noise_dm_is_bit_identical() {
        let space = Su2Space::new(6, 4, 0);
        let h = generic_su2_h::<f64>(space);
        let mut rng = seeded(0x12_0005);
        let mut x = randomized_su2_state::<f64, _>(space, 2, 4, &mut rng);
        x.right_orthogonalize(true);
        let k = 2usize;
        x.move_core(k, true);
        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let fr_all = right_to_left_sketching(&h, &x);
        let fr = &fr_all[k + 1];
        let cp = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&cp);
        let same = |a: &crate::su2::qn::Su2SectorGrid<faer::Mat<f64>>,
                    b: &crate::su2::qn::Su2SectorGrid<faer::Mat<f64>>,
                    label: &str| {
            for (s, ma) in a.iter() {
                let mb = &b[s];
                assert_eq!(
                    (ma.nrows(), ma.ncols()),
                    (mb.nrows(), mb.ncols()),
                    "{label} {s:?}"
                );
                for j in 0..ma.ncols() {
                    for i in 0..ma.nrows() {
                        let (p, q) = (*ma.get(i, j), *mb.get(i, j));
                        assert!(
                            p.to_bits() == q.to_bits() || (p == 0.0 && q == 0.0),
                            "{label}: {s:?} ({i},{j}) {p} vs {q}"
                        );
                    }
                }
            }
        };
        let mut nonzero = 0usize;
        for threads in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let half_f = sketch_h_mul(&fl, &h, &xk);
                let ref_f = noise_density_matrix(&h, &half_f, &xk, &xk1, true);
                let got_f = noise_density_matrix_streamed(&h, &fl, &xk, &xk1, true);
                same(&ref_f, &got_f, &format!("forward threads={threads}"));
                let half_b = h_sketch_mul(&h, &xk1, fr);
                let ref_b = noise_density_matrix(&h, &half_b, &xk, &xk1, false);
                let got_b = noise_density_matrix_streamed(&h, fr, &xk, &xk1, false);
                same(&ref_b, &got_b, &format!("back threads={threads}"));
                nonzero += ref_f.iter().filter(|(_, m)| m.nrows() > 0).count()
                    + ref_b.iter().filter(|(_, m)| m.nrows() > 0).count();
            });
        }
        assert!(nonzero >= 4, "fixture produced no noise blocks");
    }
}
