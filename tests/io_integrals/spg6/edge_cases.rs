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

#[test]
fn spg6_l2_g1_bit_identical_to_plain_construction() {
    let path = testdata_path("h4.FCIDUMP");
    let (h_pg, load) = Su2SparseHamiltonian::<f64>::from_fcidump_with_point_group(
        &path,
        TwoBodySymmetry::Eightfold,
        None,
        None,
        f64::EPSILON,
    )
    .unwrap_or_else(|e| panic!("h4.FCIDUMP must be readable: {e}"));
    assert!(
        load.pg.pg.is_trivial(),
        "h4.FCIDUMP carries no ORBSYM: must resolve the trivial group"
    );
    assert_eq!(
        load.space,
        Su2Space::new(load.space.d, load.space.n, load.space.two_s)
    );

    let data_plain = read_fcidump::<f64>(&path, TwoBodySymmetry::Eightfold)
        .unwrap_or_else(|e| panic!("h4.FCIDUMP must be readable: {e}"));
    assert_eq!(
        load.data.t.data, data_plain.t.data,
        "|G|=1 must be bit-identical on t (screening is a documented no-op there)"
    );
    assert_eq!(
        load.data.v.data, data_plain.v.data,
        "|G|=1 must be bit-identical on v"
    );

    let space_plain = Su2Space::new(data_plain.norb, data_plain.nelec, load.space.two_s);
    let h_plain = Su2SparseHamiltonian::new(
        space_plain,
        &data_plain.t,
        &data_plain.v,
        None,
        f64::EPSILON,
    );

    let codes = [Family::Occ, Family::Occ, Family::Empty, Family::Empty];
    let x_pg: Su2TtVector<f64> = tt_csf(&codes);
    let x_plain: Su2TtVector<f64> = tt_csf(&codes);
    assert_eq!(x_pg.space, load.space);
    assert_eq!(x_plain.space, space_plain);

    let rq_pg = rayleigh_quotient(&h_pg, &x_pg);
    let rq_plain = rayleigh_quotient(&h_plain, &x_plain);
    assert_eq!(
        rq_pg, rq_plain,
        "|G|=1: the point-group-aware and plain construction paths must give a BIT-IDENTICAL \
         Rayleigh quotient (screening is a no-op, not a float reduction)"
    );
}

fn temp_fcidump_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("qntt_spg6_{tag}_{}.FCIDUMP", std::process::id()));
    p
}

#[allow(clippy::too_many_arguments)]
fn write_synthetic_fcidump(
    path: &Path,
    norb: usize,
    nelec: usize,
    ms2: i64,
    orbsym_molpro: &[u8],
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) {
    let mut f = std::fs::File::create(path).expect("create temp fcidump");
    let orbsym_str: String = orbsym_molpro.iter().map(|x| format!("{x},")).collect();
    writeln!(f, " &FCI NORB=  {norb} , NELEC=  {nelec} , MS2=  {ms2} ,").unwrap();
    writeln!(f, " ORBSYM={orbsym_str}").unwrap();
    writeln!(f, " &end").unwrap();
    for i in 0..norb {
        for j in 0..norb {
            for k in 0..norb {
                for l in 0..norb {
                    let x = v.get(&[i, j, k, l]);
                    if x != 0.0 {
                        writeln!(f, " {x:.17e}  {} {} {} {}", i + 1, j + 1, k + 1, l + 1).unwrap();
                    }
                }
            }
        }
    }
    for i in 0..norb {
        for j in 0..norb {
            let x = t.get(&[i, j]);
            if x != 0.0 {
                writeln!(f, " {x:.17e}  {} {}  0  0", i + 1, j + 1).unwrap();
            }
        }
    }
    writeln!(f, " 0.0  0  0  0  0").unwrap();
}

#[test]
fn spg6_l4_injected_forbidden_entry_is_rejected_not_silently_zeroed() {
    let d = 4usize;
    let orbsym: Vec<u8> = vec![1, 3, 1, 3];
    let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table");
    assert_ne!(
        pg.irrep_of(0),
        pg.irrep_of(1),
        "fixture bug: need a genuinely symmetry-forbidden pair"
    );

    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        t.set(&[i, i], 1.0 + i as f64 * 0.1);
        v.set(&[i, i, i, i], 0.5);
    }
    t.set(&[0, 1], 1e-3);
    t.set(&[1, 0], 1e-3);

    let path = temp_fcidump_path("l4_forbidden");
    write_synthetic_fcidump(&path, d, 4, 0, &orbsym, &t, &v);
    let result = read_screened::<f64>(&path, TwoBodySymmetry::Eightfold, None, Some(0));
    let _ = std::fs::remove_file(&path);

    match result {
        Err(FcidumpError::ArgumentError(msg)) => {
            assert!(
                msg.contains("violates the point group"),
                "unexpected rejection message: {msg}"
            );
        }
        other => panic!(
            "expected a rejected ArgumentError (an injected 1e-3 entry is far above the \
             {SYMMETRY_FORBIDDEN_TOL:e} noise-floor tolerance), got {other:?}"
        ),
    }
}
