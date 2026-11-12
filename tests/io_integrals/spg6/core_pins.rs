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
fn spg6_s0_h2o_resolution() {
    let meta = h2o_meta();
    let auto = load_h2o(None, Some(0));
    let explicit = load_h2o(Some(PointGroupKind::C2v), Some(0));

    assert_eq!(auto.pg.kind, PointGroupKind::C2v);
    assert!(
        auto.pg.guessed,
        "no explicit kind was given: guessed must be set"
    );
    assert!(!explicit.pg.guessed);
    assert_eq!(auto.pg.pg, explicit.pg.pg);
    assert_eq!(
        auto.pg.target,
        Irrep::IDENTITY,
        "ISYM=1 must resolve to the identity irrep"
    );
    assert_eq!(
        auto.space, explicit.space,
        "AUTO and explicit C2v must build the IDENTICAL space"
    );
    assert_eq!(auto.pg.pg.order(), 4);
    assert_eq!(
        auto.space,
        Su2Space::with_point_group(7, 10, 0, auto.pg.pg, Irrep::IDENTITY)
    );
    assert_eq!(auto.data.orbsym, meta_vec_i64(&meta, "orbsym_molpro"));
    assert!(
        (auto.data.e0 - meta_f64(&meta, "e_nuc")).abs() < 1e-12,
        "e0={} vs meta e_nuc={}",
        auto.data.e0,
        meta_f64(&meta, "e_nuc")
    );
}

#[test]
fn spg6_s1_h2o_singlet_fci_pin() {
    let meta = h2o_meta();
    let load = load_h2o(None, Some(0));
    assert_eq!(load.space.target, Irrep::IDENTITY);

    let h = Su2SparseHamiltonian::new(load.space, &load.data.t, &load.data.v, None, f64::EPSILON);
    let (lam, x) = mals_min_over_seeds(&h, load.space, 6, &mals_opts());
    let e_total = lam + load.data.e0;
    let e_fci = meta_f64(&meta, "e_fci");
    assert!(
        (e_total - e_fci).abs() < 1e-8,
        "SU(2) MALS ground energy {e_total} (λ={lam} + e0={}) vs PySCF's own e_fci {e_fci} — the \
         cross-code physics pin of the whole SU(2) point-group campaign",
        load.data.e0
    );
    assert!(x.check());
    assert_expand_purity_chain(&x, &load.pg.pg);
}

#[test]
fn spg6_s2_c2v_vs_c1_agreement() {
    let meta = h2o_meta();
    let load = load_h2o(None, Some(0));
    let h_pg =
        Su2SparseHamiltonian::new(load.space, &load.data.t, &load.data.v, None, f64::EPSILON);

    let data_c1 = read_fcidump::<f64>(&h2o_dump(), TwoBodySymmetry::Eightfold)
        .unwrap_or_else(|e| panic!("h2o_c2v fixture unreadable: {e}"));
    let space_c1 = Su2Space::new(data_c1.norb, data_c1.nelec, 0);
    let h_c1 = Su2SparseHamiltonian::new(space_c1, &data_c1.t, &data_c1.v, None, f64::EPSILON);

    let opts = mals_opts();
    let (lam_pg, x_pg) = mals_min_over_seeds(&h_pg, load.space, 6, &opts);
    let (lam_c1, _x_c1) = mals_min_over_seeds(&h_c1, space_c1, 6, &opts);
    assert!(x_pg.check());

    let e_fci = meta_f64(&meta, "e_fci");
    let e_pg = lam_pg + load.data.e0;
    let e_c1 = lam_c1 + data_c1.e0;
    assert!(
        (e_pg - e_c1).abs() <= 1e-8 * e_pg.abs().max(1.0),
        "the point-group-labeled and C1 (unscreened, ORBSYM-ignoring) arms must agree: {e_pg} vs \
         {e_c1}"
    );
    assert!(
        (e_pg - e_fci).abs() < 1e-8,
        "C2v arm vs PySCF e_fci: {e_pg} vs {e_fci}"
    );
    assert!(
        (e_c1 - e_fci).abs() < 1e-8,
        "C1 arm vs PySCF e_fci: {e_c1} vs {e_fci}"
    );
}

#[test]
fn spg6_s3_ch2_triplet_identity_pin() {
    let meta = ch2_meta();
    let load = load_ch2(Some(2));
    assert_eq!(load.space.two_s, 2);
    assert_eq!(
        load.space.target,
        Irrep::IDENTITY,
        "the shipped dump's ISYM=1 (PySCF's write_head hardcodes it) must resolve to IDENTITY"
    );

    let h = Su2SparseHamiltonian::new(load.space, &load.data.t, &load.data.v, None, f64::EPSILON);
    let (lam, x) = mals_min_over_seeds(&h, load.space, 6, &mals_opts());
    let e_total = lam + load.data.e0;
    let e_want = meta_f64(&meta, "e_fci_identity");
    assert!(
        (e_total - e_want).abs() < 1e-8,
        "S3 (identity/ISYM block): {e_total} vs meta e_fci_identity {e_want}"
    );
    assert!(x.check());
    assert_expand_purity_chain(&x, &load.pg.pg);
}

#[test]
fn spg6_s3b_ch2_two_s_default_is_ms2_derived() {
    let load = load_ch2(None);
    assert_eq!(
        load.space.two_s, 2,
        "two_s=None must default to the dump's |MS2|=2, not 0"
    );
}

#[test]
fn spg6_s4_ch2_triplet_gs_pin() {
    require_ch2();
    let meta = ch2_meta();
    let path = ch2_dump();

    let (mut data, pgdata) =
        read_fcidump_with_point_group::<f64>(&path, TwoBodySymmetry::Eightfold, None)
            .unwrap_or_else(|e| panic!("SPG6 triplet fixture {path:?} failed to read: {e}"));
    screen_point_group_forbidden(&mut data.t, &mut data.v, pgdata.pg)
        .unwrap_or_else(|e| panic!("SPG6 triplet fixture {path:?} failed to screen: {e}"));

    let gs_irrep = Irrep::new(meta_usize(&meta, "gs_irrep_xor") as u8);
    assert_ne!(
        gs_irrep,
        Irrep::IDENTITY,
        "meta bug: gs_irrep_xor must be non-identity (that is this fixture's whole point)"
    );
    let two_s = 2usize;
    let space = Su2Space::with_point_group(data.norb, data.nelec, two_s, pgdata.pg, gs_irrep);

    let h = Su2SparseHamiltonian::new(space, &data.t, &data.v, None, f64::EPSILON);
    let (lam, x) = mals_min_over_seeds(&h, space, 6, &mals_opts());
    let e_total = lam + data.e0;
    let e_want = meta_f64(&meta, "e_fci");
    assert!(
        (e_total - e_want).abs() < 1e-8,
        "S4 (target-override ground-state block): {e_total} vs meta e_fci {e_want}"
    );
    assert!(x.check());
    assert_expand_purity_chain(&x, &pgdata.pg);

    let e_identity = meta_f64(&meta, "e_fci_identity");
    assert!(
        e_total < e_identity - 1e-4,
        "the true (target-override) ground state must sit strictly below the identity block: \
         {e_total} vs {e_identity}"
    );
}

#[test]
fn spg6_s6_meta_certificates() {
    let meta = ch2_meta();
    assert_eq!(
        meta_i64(&meta, "ms2"),
        2,
        "MS2 must be 2 in the shipped bytes"
    );
    assert!(
        (meta_f64(&meta, "fci_s_squared") - 2.0).abs() < 1e-6,
        "⟨S²⟩ must be 2.0 (a true triplet, not spin-contaminated)"
    );
    assert!((meta_f64(&meta, "fci_multiplicity") - 3.0).abs() < 1e-6);
    assert_eq!(
        meta_get(&meta, "fix_spin_agrees"),
        "yes",
        "the fix_spin_ control must agree with the unprojected FCI (a `no` never ships)"
    );
    assert_ne!(meta_usize(&meta, "gs_irrep_xor"), 0);
    let dets_particle = meta_usize(&meta, "dets_particle");
    let dets_identity = meta_usize(&meta, "dets_identity");
    let dets_gs = meta_usize(&meta, "dets_gs");
    assert!(
        dets_identity > 0 && dets_identity < dets_particle,
        "identity block must be a PROPER nonempty subspace: {dets_identity} of {dets_particle}"
    );
    assert!(
        dets_gs > 0 && dets_gs < dets_particle,
        "ground-state block must be a PROPER nonempty subspace: {dets_gs} of {dets_particle}"
    );
    assert_eq!(
        meta_i64(&meta, "isym"),
        1,
        "PySCF's write_head hardcodes ISYM=1 (fixture-measurement.md §1.6)"
    );
}
