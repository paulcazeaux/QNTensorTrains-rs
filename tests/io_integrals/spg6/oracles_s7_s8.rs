// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(unused_imports)]

use qntt::coeff::CoeffTensor;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::fcidump::{
    FcidumpError, PointGroupData, SYMMETRY_FORBIDDEN_TOL, TwoBodySymmetry, read_fcidump,
    read_fcidump_with_point_group, screen_point_group_forbidden,
};
use qntt::su2::core::Family;
use qntt::su2::expand::expand;
use qntt::su2::fcidump::{Su2FcidumpLoad, read_screened, su2_space};
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use qntt::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space, max_ranks};
use qntt::su2::solvers::mals::{MalsOptions, mals};
use qntt::su2::vector::{Su2TtVector, tt_csf, tt_csf_with_point_group, tt_randn as su2_tt_randn};
use qntt::vector::TtVector;
use qntt::{Irrep, PointGroup, PointGroupKind, QnSpace, Scalar};

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rustc_hash::FxHashMap;

use faer::Mat;
use qntt::integrals::read_meta;
use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::spg6::util::*;

fn screen_forbidden(
    t: &mut CoeffTensor<f64>,
    v: &mut CoeffTensor<f64>,
    pg: PointGroup,
) -> (usize, f64, usize, f64) {
    let d = t.d;
    let mut n1 = 0usize;
    let mut max1 = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of_sites([i, j]) == Irrep::IDENTITY {
                continue;
            }
            let val = t.get(&[i, j]).abs();
            if val == 0.0 {
                continue;
            }
            n1 += 1;
            max1 = max1.max(val);
            t.set(&[i, j], 0.0);
        }
    }
    let dv = v.d;
    let mut n2 = 0usize;
    let mut max2 = 0.0f64;
    for i in 0..dv {
        for j in 0..dv {
            for k in 0..dv {
                for l in 0..dv {
                    if pg.irrep_of_sites([i, j, k, l]) == Irrep::IDENTITY {
                        continue;
                    }
                    let val = v.get(&[i, j, k, l]).abs();
                    if val == 0.0 {
                        continue;
                    }
                    n2 += 1;
                    max2 = max2.max(val);
                    v.set(&[i, j, k, l], 0.0);
                }
            }
        }
    }
    (n1, max1, n2, max2)
}

fn basis_index(occ_up: &[bool], occ_dn: &[bool]) -> usize {
    let d = occ_up.len();
    let mut idx = 0usize;
    for k in (0..d).rev() {
        let fam = (occ_up[k] as usize) | ((occ_dn[k] as usize) << 1);
        idx = (idx << 2) | fam;
    }
    idx
}
fn determinants_with_irrep(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    irrep: Irrep,
) -> Vec<(Vec<bool>, Vec<bool>)> {
    let mut out = Vec::new();
    for idx in 0..(1usize << (2 * d)) {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            continue;
        }
        if config_irrep_from_digits(pg, &dig) == irrep {
            out.push((
                dig.iter().map(|&x| x == 1 || x == 3).collect(),
                dig.iter().map(|&x| x == 2 || x == 3).collect(),
            ));
        }
    }
    out
}
fn apply_string_to_det(
    d: usize,
    ops: &[(qntt::Orbital, bool)],
    occ_up: &[bool],
    occ_dn: &[bool],
) -> Option<(f64, Vec<bool>, Vec<bool>)> {
    let mut list: Vec<qntt::Orbital> = Vec::new();
    for k in 0..d {
        if occ_up[k] {
            list.push(qntt::Orbital::new(k, qntt::Spin::Up));
        }
        if occ_dn[k] {
            list.push(qntt::Orbital::new(k, qntt::Spin::Dn));
        }
    }
    let mut sign: f64 = 1.0;
    for &(orb, creation) in ops.iter().rev() {
        if creation {
            match list.binary_search(&orb) {
                Ok(_) => return None,
                Err(pos) => {
                    if pos % 2 == 1 {
                        sign = -sign;
                    }
                    list.insert(pos, orb);
                }
            }
        } else {
            match list.binary_search(&orb) {
                Ok(pos) => {
                    if pos % 2 == 1 {
                        sign = -sign;
                    }
                    list.remove(pos);
                }
                Err(_) => return None,
            }
        }
    }
    let mut out_up = vec![false; d];
    let mut out_dn = vec![false; d];
    for o in &list {
        match o.spin {
            qntt::Spin::Up => out_up[o.site] = true,
            qntt::Spin::Dn => out_dn[o.site] = true,
        }
    }
    Some((sign, out_up, out_dn))
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn target_block_ground_state_sparse(
    pg: &PointGroup,
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    nup: usize,
    ndn: usize,
    target: Irrep,
) -> (Vec<(Vec<bool>, Vec<bool>)>, f64) {
    let dets = determinants_with_irrep(pg, d, nup, ndn, target);
    let n = dets.len();
    assert!(n > 0, "empty target-irrep determinant block");
    let n_particle_block = (0..(1usize << (2 * d)))
        .filter(|&idx| config_counts(&digits(idx, d)) == (nup, ndn))
        .count();
    assert!(
        n < n_particle_block,
        "vacuous: target-irrep block is the WHOLE particle block"
    );

    let index: FxHashMap<usize, usize> = dets
        .iter()
        .enumerate()
        .map(|(row, (ou, od))| (basis_index(ou, od), row))
        .collect();

    let spins = [qntt::Spin::Up, qntt::Spin::Dn];
    let mut hblock = Mat::<f64>::zeros(n, n);
    for (col, (ou, od)) in dets.iter().enumerate() {
        for i in 0..d {
            for j in 0..d {
                let c = t.get(&[i, j]);
                if c == 0.0 {
                    continue;
                }
                for &s in &spins {
                    let ops = [
                        (qntt::Orbital::new(i, s), true),
                        (qntt::Orbital::new(j, s), false),
                    ];
                    if let Some((sign, ru, rd)) = apply_string_to_det(d, &ops, ou, od)
                        && let Some(&row) = index.get(&basis_index(&ru, &rd))
                    {
                        *hblock.get_mut(row, col) += sign * c;
                    }
                }
            }
        }
        let half = 0.5;
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        let c = v.get(&[i, j, k, l]);
                        if c == 0.0 {
                            continue;
                        }
                        for &s1 in &spins {
                            for &s2 in &spins {
                                let oi = qntt::Orbital::new(i, s1);
                                let oj = qntt::Orbital::new(j, s1);
                                let ok = qntt::Orbital::new(k, s2);
                                let ol = qntt::Orbital::new(l, s2);
                                if oi != ok && oj != ol {
                                    let ops = [(oi, true), (ok, true), (ol, false), (oj, false)];
                                    if let Some((sign, ru, rd)) =
                                        apply_string_to_det(d, &ops, ou, od)
                                        && let Some(&row) = index.get(&basis_index(&ru, &rd))
                                    {
                                        *hblock.get_mut(row, col) += sign * c * half;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut asym = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            asym = asym.max((hblock.get(i, j) - hblock.get(j, i)).abs());
        }
    }
    assert!(asym < 1e-9, "H_block not Hermitian to 1e-9 (max {asym})");
    let (evals, _u) = dense_self_adjoint_eigen(hblock.as_ref());
    (dets, evals[0])
}

#[allow(clippy::too_many_arguments)]
fn full_rank_su2_ground(
    pg: &PointGroup,
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    n: usize,
    two_s: usize,
    target: Irrep,
    seed: u64,
) -> f64 {
    let space = Su2Space::with_point_group(d, n, two_s, *pg, target);
    let caps = max_ranks(&space);
    let caps_ref = &caps;
    let cap_max = (0..=d)
        .flat_map(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .map(move |s| caps_ref[b][s])
        })
        .max()
        .unwrap_or(0);
    assert!(
        cap_max > 0,
        "n={n} 2s={two_s} target={target:?}: empty block"
    );
    let ranks = su2_capped_ranks(&space, cap_max);
    let mut rng = seeded(seed);
    let x0: Su2TtVector<f64> = su2_tt_randn(space, &ranks, &mut rng);
    assert!(x0.check());
    let h = Su2SparseHamiltonian::new(space, t, v, None, 1e-12);
    let opts = MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        max_iter: 40,
        noise: vec![0.0],
        ..Default::default()
    };
    let (lam, x) = mals(&h, &x0, &opts);
    assert!(x.check());
    lam
}

#[test]
fn spg6_s7_synthetic_nonidentity_target() {
    let d = 4usize;
    let pg = c2v_table(d);
    let mut rng = seeded(76_700);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);

    for &(n, two_s) in &[(4usize, 0usize), (4usize, 2usize)] {
        let id_target = Irrep::IDENTITY;
        let nonid_target =
            find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or_else(|| {
                panic!("fixture bug: no nonidentity-reachable target at n={n} 2s={two_s}")
            });

        let lam_id = full_rank_su2_ground(&pg, d, &t, &v, n, two_s, id_target, 76_701);
        let lam_nonid = full_rank_su2_ground(&pg, d, &t, &v, n, two_s, nonid_target, 76_702);
        assert!(
            (lam_id - lam_nonid).abs() > 1e-6,
            "n={n} 2s={two_s}: identity vs nonidentity target energies must differ: {lam_id} vs \
             {lam_nonid}"
        );

        for &(target, lam_su2) in &[(id_target, lam_id), (nonid_target, lam_nonid)] {
            let nup = (n + two_s) / 2;
            let ndn = (n - two_s) / 2;
            let (dets, lam_dense) =
                target_block_ground_state_sparse(&pg, d, &t, &v, nup, ndn, target);
            assert!(!dets.is_empty());
            assert!(
                lam_dense <= lam_su2 + 1e-8 * lam_su2.abs().max(1.0),
                "n={n} 2s={two_s} target={target:?}: the abelian dense block ({lam_dense}) must \
                 lower-bound the SU(2) full-rank ground energy ({lam_su2}) — it is a superspace"
            );
        }
    }
}

#[test]
fn spg6_s8_screening_census_matches_meta() {
    let h2o_meta_map = h2o_meta();
    let (mut h2o_data, h2o_pg) =
        read_fcidump_with_point_group::<f64>(&h2o_dump(), TwoBodySymmetry::Eightfold, None)
            .unwrap_or_else(|e| panic!("h2o_c2v fixture unreadable: {e}"));
    let h2o_census = screen_forbidden(&mut h2o_data.t, &mut h2o_data.v, h2o_pg.pg);
    assert_eq!(h2o_census, (0, 0.0, 0, 0.0));
    assert_eq!(meta_usize(&h2o_meta_map, "forbidden_1body"), 0);
    assert_eq!(meta_usize(&h2o_meta_map, "forbidden_2body"), 0);

    let n2_meta_map = n2_meta();
    let (mut n2_data, n2_pg) =
        read_fcidump_with_point_group::<f64>(&n2_dump(), TwoBodySymmetry::Eightfold, None)
            .unwrap_or_else(|e| panic!("n2_d2h fixture unreadable: {e}"));
    assert_eq!(n2_pg.kind, PointGroupKind::D2h);

    let (n1, max1, n2, max2) = screen_forbidden(&mut n2_data.t, &mut n2_data.v, n2_pg.pg);
    assert_eq!(n1, meta_usize(&n2_meta_map, "forbidden_1body"));
    assert_eq!(n2, meta_usize(&n2_meta_map, "forbidden_2body"));
    let observed_max = max1.max(max2);
    assert!((observed_max - meta_f64(&n2_meta_map, "forbidden_max")).abs() < 1e-16);
    assert!(max1 < SYMMETRY_FORBIDDEN_TOL);
    assert!(max2 < SYMMETRY_FORBIDDEN_TOL);

    let load = read_screened::<f64>(&n2_dump(), TwoBodySymmetry::Eightfold, None, Some(0))
        .unwrap_or_else(|e| panic!("read_screened must succeed on n2_d2h: {e}"));
    let mut t_post = load.data.t.clone();
    let mut v_post = load.data.v.clone();
    let post_census = screen_forbidden(&mut t_post, &mut v_post, load.pg.pg);
    assert_eq!(
        post_census,
        (0, 0.0, 0, 0.0),
        "the loader's own output must carry ZERO forbidden entries after screening"
    );
}
