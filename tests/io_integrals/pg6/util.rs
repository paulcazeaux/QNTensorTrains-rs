// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::Mat;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::fcidump::{
    FcidumpData, PointGroupData, TwoBodySymmetry, read_fcidump, read_fcidump_with_point_group,
};
use qntt::integrals::read_meta;
use qntt::{CoeffTensor, Irrep, PointGroup, PointGroupKind, QnSpace, Scalar, TtVector, tt_randn};
pub(crate) use qntt_testkit::pg::capped_ranks;
pub(crate) use qntt_testkit::pg::config_counts;
pub(crate) use qntt_testkit::pg::config_irrep_from_digits;
pub(crate) use qntt_testkit::pg::digits;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rustc_hash::FxHashMap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub(crate) fn testdata_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}
pub(crate) fn h2o_dir() -> PathBuf {
    testdata_path("h2o_c2v")
}
pub(crate) fn n2_dir() -> PathBuf {
    testdata_path("n2_d2h")
}
pub(crate) fn h2o_dump_path() -> PathBuf {
    h2o_dir().join("H2O.FCIDUMP")
}
pub(crate) fn h2o_meta_path() -> PathBuf {
    h2o_dir().join("meta.txt")
}
pub(crate) fn n2_dump_path() -> PathBuf {
    n2_dir().join("N2.FCIDUMP")
}
pub(crate) fn n2_meta_path() -> PathBuf {
    n2_dir().join("meta.txt")
}

pub(crate) fn load_h2o(kind: Option<PointGroupKind>) -> (FcidumpData<f64>, PointGroupData) {
    let path = h2o_dump_path();
    read_fcidump_with_point_group::<f64>(&path, TwoBodySymmetry::Eightfold, kind).unwrap_or_else(
        |e| {
            panic!(
                "missing or unreadable PG6 fixture {path:?}: {e}\n\
                 Generate it first: ~/miniforge3/envs/pyscf/bin/python {}",
                h2o_dir().join("generate.py").display()
            )
        },
    )
}
pub(crate) fn load_n2(kind: Option<PointGroupKind>) -> (FcidumpData<f64>, PointGroupData) {
    let path = n2_dump_path();
    read_fcidump_with_point_group::<f64>(&path, TwoBodySymmetry::Eightfold, kind).unwrap_or_else(
        |e| {
            panic!(
                "missing or unreadable PG6 fixture {path:?}: {e}\n\
                 Generate it first: ~/miniforge3/envs/pyscf/bin/python {}",
                n2_dir().join("generate.py").display()
            )
        },
    )
}
pub(crate) fn load_h2o_c1() -> FcidumpData<f64> {
    let path = h2o_dump_path();
    read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold).unwrap_or_else(|e| {
        panic!(
            "missing or unreadable PG6 fixture {path:?}: {e}\n\
             Generate it first: ~/miniforge3/envs/pyscf/bin/python {}",
            h2o_dir().join("generate.py").display()
        )
    })
}

pub(crate) fn parse_meta(path: &Path) -> HashMap<String, String> {
    read_meta(path)
        .unwrap_or_else(|e| {
            panic!(
                "missing or unreadable PG6 meta file {path:?}: {e}\n\
                 Generate it (and its FCIDUMP sibling) first: \
                 ~/miniforge3/envs/pyscf/bin/python {}",
                path.parent().unwrap().join("generate.py").display()
            )
        })
        .keys
}
pub(crate) fn h2o_meta() -> HashMap<String, String> {
    parse_meta(&h2o_meta_path())
}
pub(crate) fn n2_meta() -> HashMap<String, String> {
    parse_meta(&n2_meta_path())
}
pub(crate) fn meta_get<'a>(meta: &'a HashMap<String, String>, key: &str) -> &'a str {
    meta.get(key)
        .unwrap_or_else(|| panic!("meta file missing key {key:?}"))
}
pub(crate) fn meta_f64(meta: &HashMap<String, String>, key: &str) -> f64 {
    meta_get(meta, key)
        .parse()
        .unwrap_or_else(|e| panic!("meta key {key:?}: {e}"))
}
pub(crate) fn meta_usize(meta: &HashMap<String, String>, key: &str) -> usize {
    meta_get(meta, key)
        .parse()
        .unwrap_or_else(|e| panic!("meta key {key:?}: {e}"))
}
pub(crate) fn meta_i64(meta: &HashMap<String, String>, key: &str) -> i64 {
    meta_get(meta, key)
        .parse()
        .unwrap_or_else(|e| panic!("meta key {key:?}: {e}"))
}
pub(crate) fn meta_vec_i64(meta: &HashMap<String, String>, key: &str) -> Vec<i64> {
    meta_get(meta, key)
        .split_whitespace()
        .map(|s| {
            s.parse()
                .unwrap_or_else(|e| panic!("meta key {key:?}: {e}"))
        })
        .collect()
}
pub(crate) fn meta_vec_str<'a>(meta: &'a HashMap<String, String>, key: &str) -> Vec<&'a str> {
    meta_get(meta, key).split_whitespace().collect()
}

pub(crate) fn config_occ(dig: &[usize]) -> (Vec<bool>, Vec<bool>) {
    let occ_up = dig.iter().map(|&x| x == 1 || x == 3).collect();
    let occ_dn = dig.iter().map(|&x| x == 2 || x == 3).collect();
    (occ_up, occ_dn)
}
pub(crate) fn determinants_with_irrep(
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
            out.push(config_occ(&dig));
        }
    }
    out
}
pub(crate) fn assert_purity<T: Scalar>(
    pg: &PointGroup,
    d: usize,
    nup: usize,
    ndn: usize,
    target: Irrep,
    dense: &[T],
) {
    assert_eq!(dense.len(), 1usize << (2 * d));
    let mut any_nonzero_in_target = false;
    for (idx, v) in dense.iter().enumerate() {
        let dig = digits(idx, d);
        let (cu, cd) = config_counts(&dig);
        if cu != nup || cd != ndn {
            assert!(
                *v == T::zero(),
                "d={d}: config {idx} outside (Nup,Ndn) must be exactly zero"
            );
            continue;
        }
        let irrep = config_irrep_from_digits(pg, &dig);
        if irrep == target {
            if *v != T::zero() {
                any_nonzero_in_target = true;
            }
        } else {
            assert!(
                *v == T::zero(),
                "d={d}: config {idx} has irrep {irrep:?} != target {target:?}"
            );
        }
    }
    assert!(
        any_nonzero_in_target,
        "d={d}: oracle vacuous for target {target:?}"
    );
}
pub(crate) fn capped_randn<T: Scalar>(
    space: QnSpace,
    lo: usize,
    hi: usize,
    seed: u64,
) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks = capped_ranks(&space, lo, hi, &mut rng);
    let x: TtVector<T> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    x
}
pub(crate) fn basis_index(occ_up: &[bool], occ_dn: &[bool]) -> usize {
    let d = occ_up.len();
    let mut idx = 0usize;
    for k in (0..d).rev() {
        let fam = (occ_up[k] as usize) | ((occ_dn[k] as usize) << 1);
        idx = (idx << 2) | fam;
    }
    idx
}

pub(crate) fn screen_forbidden(
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

pub(crate) fn apply_string_to_det(
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
pub(crate) fn target_block_ground_state_sparse(
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
    assert!(
        asym < 1e-9,
        "H_block is not Hermitian to 1e-9 (max asymmetry {asym})"
    );

    let (evals, _u) = dense_self_adjoint_eigen(hblock.as_ref());
    (dets, evals[0])
}
