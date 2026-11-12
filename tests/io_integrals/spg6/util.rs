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
pub(crate) use qntt_testkit::pg::assert_irrep_split;
pub(crate) use qntt_testkit::pg::c2v_table;
pub(crate) use qntt_testkit::pg::config_counts;
pub(crate) use qntt_testkit::pg::config_irrep_from_digits;
pub(crate) use qntt_testkit::pg::digits;
pub(crate) use qntt_testkit::pg::irrep_census;
pub(crate) use qntt_testkit::pg::su2_capped_ranks;
use std::collections::HashMap;
use std::io::Write as _;
use std::path::{Path, PathBuf};

pub(crate) fn testdata_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}
pub(crate) fn h2o_dir() -> PathBuf {
    testdata_path("h2o_c2v")
}
pub(crate) fn h2o_dump() -> PathBuf {
    h2o_dir().join("H2O.FCIDUMP")
}
pub(crate) fn ch2_dir() -> PathBuf {
    testdata_path("ch2_triplet_c2v")
}
pub(crate) fn ch2_dump() -> PathBuf {
    ch2_dir().join("CH2.FCIDUMP")
}
pub(crate) fn n2_dir() -> PathBuf {
    testdata_path("n2_d2h")
}
pub(crate) fn n2_dump() -> PathBuf {
    n2_dir().join("N2.FCIDUMP")
}

pub(crate) fn load_h2o(kind: Option<PointGroupKind>, two_s: Option<usize>) -> Su2FcidumpLoad<f64> {
    let path = h2o_dump();
    read_screened::<f64>(&path, TwoBodySymmetry::Eightfold, kind, two_s).unwrap_or_else(|e| {
        panic!(
            "missing or unreadable SPG6 fixture {path:?}: {e}\n\
             Generate it first: ~/miniforge3/envs/pyscf/bin/python {}",
            h2o_dir().join("generate.py").display()
        )
    })
}

pub(crate) fn ch2_available() -> bool {
    ch2_dump().is_file() && ch2_dir().join("meta.txt").is_file()
}

pub(crate) fn require_ch2() {
    if !ch2_available() {
        panic!(
            "missing SPG6 triplet fixture {:?}: this is a NEW SPG6 deliverable (Part 3), not yet \
             generated on this worktree. Generate it first: \
             ~/miniforge3/envs/pyscf/bin/python {}",
            ch2_dump(),
            ch2_dir().join("generate.py").display()
        );
    }
}

pub(crate) fn load_ch2(two_s: Option<usize>) -> Su2FcidumpLoad<f64> {
    require_ch2();
    let path = ch2_dump();
    read_screened::<f64>(&path, TwoBodySymmetry::Eightfold, None, two_s)
        .unwrap_or_else(|e| panic!("SPG6 triplet fixture {path:?} failed to load: {e}"))
}

pub(crate) fn parse_meta(path: &Path) -> HashMap<String, String> {
    read_meta(path)
        .unwrap_or_else(|e| {
            panic!(
                "missing or unreadable SPG6 meta file {path:?}: {e}\n\
                 Generate it (and its FCIDUMP sibling) first: \
                 ~/miniforge3/envs/pyscf/bin/python {}",
                path.parent().unwrap().join("generate.py").display()
            )
        })
        .keys
}
pub(crate) fn h2o_meta() -> HashMap<String, String> {
    parse_meta(&h2o_dir().join("meta.txt"))
}
pub(crate) fn ch2_meta() -> HashMap<String, String> {
    require_ch2();
    parse_meta(&ch2_dir().join("meta.txt"))
}
pub(crate) fn n2_meta() -> HashMap<String, String> {
    parse_meta(&n2_dir().join("meta.txt"))
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

pub(crate) fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

pub(crate) fn find_nonidentity_reachable_target(
    pg: PointGroup,
    n: usize,
    two_s: usize,
    d: usize,
) -> Option<Irrep> {
    (1..pg.order()).map(Irrep::new).find(|&ir| {
        let space = Su2Space::with_point_group(d, n, two_s, pg, ir);
        let caps = max_ranks(&space);
        caps[d][Su2Sector::with_irrep(n, two_s, ir)] > 0
    })
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
                "d={d}: config {idx} outside (Nup={nup},Ndn={ndn}) must be exactly zero, got {v:?}"
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
                "d={d} config {idx} has irrep {irrep:?} != target {target:?} but amplitude is \
                 {v:?} (not exactly zero)"
            );
        }
    }
    assert!(
        any_nonzero_in_target,
        "d={d}: oracle vacuous — no nonzero amplitude found inside the target irrep {target:?}"
    );
}
pub(crate) fn assert_expand_purity_chain<T: Scalar>(x: &Su2TtVector<T>, pg: &PointGroup) {
    let d = x.space.d;
    let two_s = x.space.two_s as i64;
    let target = x.space.target;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let nup = ((x.space.n as i64 + two_sz) / 2) as usize;
            let ndn = ((x.space.n as i64 - two_sz) / 2) as usize;
            let y: TtVector<T> = expand(x, two_sz);
            assert_eq!(
                y.space,
                QnSpace::with_point_group(d, nup, ndn, x.space.pg, target),
                "expand(x, two_sz={two_sz}): abelian space mismatch"
            );
            assert!(
                y.check(),
                "expand(x, two_sz={two_sz}): abelian check() failed"
            );
            assert_irrep_split(pg, d, nup, ndn);
            assert_purity(pg, d, nup, ndn, target, &y.to_dense());
        }
        two_sz += 1;
    }
}

pub(crate) fn c2v_coeffs(
    pg: &PointGroup,
    d: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    use rand::RngExt;
    use rand_distr::StandardNormal;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            if t.get(&[i, j]) != 0.0 {
                continue;
            }
            let val = if pg.irrep_of(i) == pg.irrep_of(j) {
                rng.sample::<f64, _>(StandardNormal)
            } else {
                0.0
            };
            t.set(&[i, j], val);
            t.set(&[j, i], val);
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
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
                        rng.sample::<f64, _>(StandardNormal)
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
    (t, v)
}

pub(crate) fn mals_opts() -> MalsOptions {
    MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        max_iter: 30,
        noise: vec![0.0],
        ..Default::default()
    }
}

pub(crate) fn mals_min_over_seeds(
    h: &Su2SparseHamiltonian<f64>,
    space: Su2Space,
    rmax: usize,
    opts: &MalsOptions,
) -> (f64, Su2TtVector<f64>) {
    const SEEDS: [u64; 3] = [96_101, 96_102, 96_103];
    let caps = max_ranks(&space);
    let target_sector = Su2Sector::with_irrep(space.n, space.two_s, space.target);
    assert!(
        caps[space.d][target_sector] > 0,
        "requested (N={}, 2S={}, target={:?}) block is EMPTY — no CSF has this quantum number",
        space.n,
        space.two_s,
        space.target
    );

    let mut results: Vec<(f64, Su2TtVector<f64>)> = Vec::new();
    for &seed in &SEEDS {
        let ranks = su2_capped_ranks(&space, rmax);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let x0: Su2TtVector<f64> = su2_tt_randn(space, &ranks, &mut rng);
        assert!(x0.check(), "seed {seed}: start state failed check()");
        let (lam, x) = mals(h, &x0, opts);
        assert!(x.check(), "seed {seed}: converged state failed check()");
        results.push((lam, x));
    }
    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let lam_min = results[0].0;
    let agree = results[1..]
        .iter()
        .filter(|(l, _)| (l - lam_min).abs() <= 1e-8 * lam_min.abs().max(1.0))
        .count();
    assert!(
        agree >= 1,
        "at least two of three MALS seeds must agree to 1e-8, got {:?}",
        results.iter().map(|r| r.0).collect::<Vec<_>>()
    );
    results.remove(0)
}
