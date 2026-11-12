// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::{h_sketch_mul, noise_density_matrix, sketch_h_mul};
use crate::core::factor::{dense_self_adjoint_eigen, svd_uvt_svals};
use crate::fcidump::{TwoBodySymmetry, read_fcidump};
use crate::su2::chop::chop_energy;
use crate::su2::core::Su2Core;
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, Ts};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::su2::twosite::{contract, factor_qc, factor_svd_perturbative};
use crate::su2::vector::tt_randn;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rustc_hash::FxHashMap;
use std::collections::HashSet;
use std::path::Path;

mod chop;
mod placement;

fn h4_fixture() -> (Su2SparseHamiltonian<f64>, Su2Space) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP");
    let data = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).expect("read h4.FCIDUMP");
    let space = Su2Space::new(data.norb, data.nelec, 0);
    let h = Su2SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
    (h, space)
}

fn pg_fixture() -> (Su2SparseHamiltonian<f64>, Su2Space) {
    use crate::{Irrep, PointGroup, PointGroupKind};

    let d = 6usize;
    let orbsym: Vec<u8> = (0..d).map(|k| ((k % 4) + 1) as u8).collect();
    let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table");
    let (n, two_s) = (4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5bc0_de02);

    let mut t = crate::coeff::CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            if t.get(&[i, j]) != 0.0 {
                continue;
            }
            let val = if pg.irrep_of(i) == pg.irrep_of(j) {
                rng.sample::<f64, _>(rand_distr::StandardNormal)
            } else {
                0.0
            };
            t.set(&[i, j], val);
            t.set(&[j, i], val);
        }
    }
    let mut v = crate::coeff::CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if v.get(&[i, j, k, l]) != 0.0 {
                        continue;
                    }
                    let irrep_ok = pg
                        .irrep_of(i)
                        .fuse(pg.irrep_of(j))
                        .fuse(pg.irrep_of(k))
                        .fuse(pg.irrep_of(l))
                        == Irrep::IDENTITY;
                    let val = if irrep_ok {
                        rng.sample::<f64, _>(rand_distr::StandardNormal)
                    } else {
                        0.0
                    };
                    for &(a, b, c, dd) in &[
                        (i, j, k, l),
                        (j, i, k, l),
                        (i, j, l, k),
                        (j, i, l, k),
                        (k, l, i, j),
                        (l, k, i, j),
                        (k, l, j, i),
                        (l, k, j, i),
                    ] {
                        v.set(&[a, b, c, dd], val);
                    }
                }
            }
        }
    }

    let target = (1..pg.order())
        .map(Irrep::new)
        .find(|&ir| {
            let space = Su2Space::with_point_group(d, n, two_s, pg, ir);
            crate::su2::qn::max_ranks(&space)[d][Su2Sector::with_irrep(n, two_s, ir)] > 0
        })
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    (h, space)
}

fn randomized_state(
    space: Su2Space,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> crate::su2::vector::Su2TtVector<f64> {
    let d = space.d;
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    tt_randn(space, &ranks, rng)
}

fn oracle_channel_sq(tmp: MatRef<'_, f64>, slab: MatRef<'_, f64>) -> f64 {
    let mut acc = 0.0;
    for j in 0..tmp.ncols() {
        for i in 0..tmp.nrows() {
            acc += tmp[(i, j)] * slab[(i, j)];
        }
    }
    acc
}

fn oracle_forward(
    h: &Su2SparseHamiltonian<f64>,
    half: &Su2Core<f64>,
    xk: &Su2Core<f64>,
    xk1: &Su2Core<f64>,
) -> Su2SectorGrid<Mat<f64>> {
    let space = half.space;
    let k = xk.k;
    let ranges = h.col_ranges(xk);
    let mut out = Su2SectorGrid::from_fn(&space, |_| Mat::<f64>::zeros(0, 0));
    let mut group_sq: FxHashMap<Ts, f64> = FxHashMap::default();
    let build_p = |m_c: Su2Sector, w: usize| -> Option<Mat<f64>> {
        let mut r = xk1.horizontal_unfolding(m_c);
        if r.nrows() != w {
            return None;
        }
        xk1.scale_horizontal(&mut r, m_c);
        let mut p = Mat::<f64>::zeros(w, w);
        matmul(
            p.as_mut(),
            Accum::Replace,
            r.as_ref(),
            r.adjoint(),
            1.0,
            Par::Seq,
        );
        Some(p)
    };
    for &s in &half.col_qn {
        let e_raw = half.vertical_unfolding(s);
        let rows = e_raw.nrows();
        if rows == 0 || e_raw.ncols() == 0 {
            continue;
        }
        let m_list = h.state_qns_at(k + 1, s);
        let descs = h.channel_descs(k + 1, s);
        let rs = &ranges[s];
        for (c, &m_c) in m_list.iter().enumerate() {
            let rng = &rs[c];
            let w = rng.len();
            if w == 0 {
                continue;
            }
            let Some(p) = build_p(m_c, w) else { continue };
            let sigma_sc = (m_c.two_s as f64 + 1.0) / (s.two_s as f64 + 1.0);
            let slab = e_raw.submatrix(0, rng.start, rows, w);
            let mut tmp = Mat::<f64>::zeros(rows, w);
            matmul(
                tmp.as_mut(),
                Accum::Replace,
                slab,
                p.as_ref(),
                1.0,
                Par::Seq,
            );
            *group_sq.entry(descs[c]).or_insert(0.0) +=
                sigma_sc * oracle_channel_sq(tmp.as_ref(), slab);
        }
    }
    let floor = group_sq.values().copied().fold(0.0, f64::max) * f64::EPSILON;
    for &s in &half.col_qn {
        let e_raw = half.vertical_unfolding(s);
        let rows = e_raw.nrows();
        if rows == 0 || e_raw.ncols() == 0 {
            continue;
        }
        let m_list = h.state_qns_at(k + 1, s);
        let descs = h.channel_descs(k + 1, s);
        let rs = &ranges[s];
        let mut rho = Mat::<f64>::zeros(rows, rows);
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
            let Some(p) = build_p(m_c, w) else { continue };
            let sigma_sc = (m_c.two_s as f64 + 1.0) / (s.two_s as f64 + 1.0);
            let slab = e_raw.submatrix(0, rng.start, rows, w);
            let mut tmp = Mat::<f64>::zeros(rows, w);
            matmul(
                tmp.as_mut(),
                Accum::Replace,
                slab,
                p.as_ref(),
                1.0,
                Par::Seq,
            );
            matmul(
                rho.as_mut(),
                Accum::Add,
                tmp.as_ref(),
                slab.adjoint(),
                sigma_sc / gs,
                Par::Seq,
            );
            any = true;
        }
        if any {
            out[s] = rho;
        }
    }
    out
}

fn oracle_back(
    h: &Su2SparseHamiltonian<f64>,
    half: &Su2Core<f64>,
    xk: &Su2Core<f64>,
    xk1: &Su2Core<f64>,
) -> Su2SectorGrid<Mat<f64>> {
    let space = half.space;
    let k1 = xk1.k;
    let ranges = h.row_ranges(xk1);
    let mut out = Su2SectorGrid::from_fn(&space, |_| Mat::<f64>::zeros(0, 0));
    let mut group_sq: FxHashMap<Ts, f64> = FxHashMap::default();
    let build_p = |m_c: Su2Sector, w: usize| -> Option<Mat<f64>> {
        let l = xk.vertical_unfolding(m_c);
        if l.ncols() != w {
            return None;
        }
        let mut p = Mat::<f64>::zeros(w, w);
        matmul(
            p.as_mut(),
            Accum::Replace,
            l.adjoint(),
            l.as_ref(),
            1.0,
            Par::Seq,
        );
        Some(p)
    };
    for &s in &half.row_qn {
        let mut e_raw = half.horizontal_unfolding(s);
        let cols = e_raw.ncols();
        if cols == 0 || e_raw.nrows() == 0 {
            continue;
        }
        half.scale_horizontal(&mut e_raw, s);
        let m_list = h.state_qns_at(k1, s);
        let descs = h.channel_descs(k1, s);
        let rs = &ranges[s];
        for (c, &m_c) in m_list.iter().enumerate() {
            let rng = &rs[c];
            let w = rng.len();
            if w == 0 {
                continue;
            }
            let Some(p) = build_p(m_c, w) else { continue };
            let slab = e_raw.submatrix(rng.start, 0, w, cols);
            let mut tmp = Mat::<f64>::zeros(w, cols);
            matmul(
                tmp.as_mut(),
                Accum::Replace,
                p.as_ref(),
                slab,
                1.0,
                Par::Seq,
            );
            *group_sq.entry(descs[c]).or_insert(0.0) += oracle_channel_sq(tmp.as_ref(), slab);
        }
    }
    let floor = group_sq.values().copied().fold(0.0, f64::max) * f64::EPSILON;
    for &s in &half.row_qn {
        let mut e_raw = half.horizontal_unfolding(s);
        let cols = e_raw.ncols();
        if cols == 0 || e_raw.nrows() == 0 {
            continue;
        }
        half.scale_horizontal(&mut e_raw, s);
        let m_list = h.state_qns_at(k1, s);
        let descs = h.channel_descs(k1, s);
        let rs = &ranges[s];
        let mut rho = Mat::<f64>::zeros(cols, cols);
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
            let Some(p) = build_p(m_c, w) else { continue };
            let slab = e_raw.submatrix(rng.start, 0, w, cols);
            let mut tmp = Mat::<f64>::zeros(w, cols);
            matmul(
                tmp.as_mut(),
                Accum::Replace,
                p.as_ref(),
                slab,
                1.0,
                Par::Seq,
            );
            matmul(
                rho.as_mut(),
                Accum::Add,
                slab.adjoint(),
                tmp.as_ref(),
                1.0 / gs,
                Par::Seq,
            );
            any = true;
        }
        if any {
            out[s] = rho;
        }
    }
    out
}

fn block_rel_diff(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    let mut diff = 0.0;
    let mut refn = 0.0;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            diff += (a[(i, j)] - b[(i, j)]).powi(2);
            refn += b[(i, j)] * b[(i, j)];
        }
    }
    diff.sqrt() / refn.sqrt().max(1.0)
}

fn assert_non_vacuous(h: &Su2SparseHamiltonian<f64>, bond: usize, sectors: &[Su2Sector]) {
    let mut groups: HashSet<Ts> = HashSet::default();
    let mut cross_spin = false;
    for &s in sectors {
        for (c, &desc) in h.channel_descs(bond, s).iter().enumerate() {
            groups.insert(desc);
            if h.state_qns_at(bond, s)[c].two_s != s.two_s {
                cross_spin = true;
            }
        }
    }
    assert!(
        groups.len() >= 2,
        "fixture too poor: need >=2 operator channels threading bond {bond}, got {}",
        groups.len()
    );
    assert!(
        cross_spin,
        "fixture too poor: need a channel with m_c.two_s != s.two_s at bond {bond} to \
         distinguish the reference-s θ placement from the rejected reference-m_c alternative"
    );
}
