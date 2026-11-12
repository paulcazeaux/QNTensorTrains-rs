// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg6::util::*;
use qntt::hamiltonian::Construction;
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{Irrep, PointGroupKind, SparseHamiltonian, TtVector};

#[test]
fn pg6_h2o_auto_guess_resolves_c2v() {
    let meta = h2o_meta();
    let (data, pg) = load_h2o(None);

    assert_eq!(pg.kind, PointGroupKind::C2v);
    assert!(
        pg.guessed,
        "no explicit kind was given: guessed must be set"
    );
    assert_eq!(
        pg.target,
        Irrep::IDENTITY,
        "ISYM=1 (PySCF's write_head hardcodes it) must resolve to the identity irrep"
    );

    let irrep_names = meta_vec_str(&meta, "irrep_names");
    assert_eq!(irrep_names.len(), data.norb);
    for (k, &want_name) in irrep_names.iter().enumerate() {
        let name = pg
            .kind
            .irrep_name(pg.pg.irrep_of(k))
            .unwrap_or_else(|e| panic!("site {k}: {e}"));
        assert_eq!(name, want_name, "site {k}: irrep name mismatch");
    }

    assert_eq!(data.orbsym, meta_vec_i64(&meta, "orbsym_molpro"));
    assert_eq!(data.isym, Some(1));
    assert_eq!(data.norb, meta_usize(&meta, "norb"));
    assert_eq!(data.nelec, meta_usize(&meta, "nelec"));
    assert_eq!(data.ms2, meta_i64(&meta, "ms2"));
    assert!(
        (data.e0 - meta_f64(&meta, "e_nuc")).abs() < 1e-12,
        "e0={} vs meta e_nuc={}",
        data.e0,
        meta_f64(&meta, "e_nuc")
    );
}

#[test]
fn pg6_h2o_explicit_c2v_equals_auto() {
    let (data_auto, pg_auto) = load_h2o(None);
    let (data_explicit, pg_explicit) = load_h2o(Some(PointGroupKind::C2v));

    assert!(pg_auto.guessed);
    assert!(!pg_explicit.guessed);
    assert_eq!(pg_explicit.pg, pg_auto.pg);
    assert_eq!(pg_explicit.target, pg_auto.target);
    assert_eq!(
        data_auto.t.data, data_explicit.t.data,
        "AUTO vs explicit C2v must be bit-identical on t"
    );
    assert_eq!(
        data_auto.v.data, data_explicit.v.data,
        "AUTO vs explicit C2v must be bit-identical on v"
    );

    let space_auto = data_auto
        .qn_space(None)
        .unwrap_or_else(|e| panic!("qn_space (auto): {e}"));
    let space_explicit = data_explicit
        .qn_space(Some(PointGroupKind::C2v))
        .unwrap_or_else(|e| panic!("qn_space (explicit): {e}"));
    assert_eq!(space_auto, space_explicit);

    let h_auto: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_auto,
        &data_auto.t,
        &data_auto.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );
    let h_explicit: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_explicit,
        &data_explicit.t,
        &data_explicit.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );

    let opts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![64],
        max_iter: 20,
        ..Default::default()
    };
    let x0_auto: TtVector<f64> = capped_randn(space_auto, 1, 4, 76_001);
    let x0_explicit: TtVector<f64> = capped_randn(space_explicit, 1, 4, 76_001);
    let (lam_auto, _) = mals(&h_auto, &x0_auto, &opts);
    let (lam_explicit, _) = mals(&h_explicit, &x0_explicit, &opts);

    assert!(
        (lam_auto - lam_explicit).abs() <= 1e-12 * lam_auto.abs().max(1.0),
        "the guessed (C2v) and explicit (C2v) loads of the SAME file must give the IDENTICAL \
         energy — only names could ever differ, and here they don't even do that: {lam_auto} vs \
         {lam_explicit}"
    );
}
