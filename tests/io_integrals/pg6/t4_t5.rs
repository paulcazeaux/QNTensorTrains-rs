// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg6::util::*;
use qntt::hamiltonian::Construction;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{QnSpace, SparseHamiltonian, TtVector};

#[test]
fn pg6_h2o_mals_matches_c1_load() {
    let meta = h2o_meta();
    let (mut data_pg, pgdata) = load_h2o(None);
    screen_forbidden(&mut data_pg.t, &mut data_pg.v, pgdata.pg);
    let space_pg = data_pg
        .qn_space(None)
        .unwrap_or_else(|e| panic!("qn_space: {e}"));

    let data_c1 = load_h2o_c1();
    let space_c1 = QnSpace::from_n_two_sz(data_c1.norb, data_c1.nelec, data_c1.ms2);

    let h_pg: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_pg,
        &data_pg.t,
        &data_pg.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );
    let h_c1: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_c1,
        &data_c1.t,
        &data_c1.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );

    let opts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![64],
        max_iter: 30,
        ..Default::default()
    };
    let x0_pg: TtVector<f64> = capped_randn(space_pg, 1, 4, 76_004);
    let x0_c1: TtVector<f64> = capped_randn(space_c1, 1, 4, 76_004);
    let (lam_pg, x_pg) = mals(&h_pg, &x0_pg, &opts);
    let (lam_c1, _x_c1) = mals(&h_c1, &x0_c1, &opts);
    assert!(x_pg.check());

    let e_exact = meta_f64(&meta, "e_fci") - meta_f64(&meta, "e_nuc");
    assert!(
        (lam_pg - lam_c1).abs() <= 1e-8 * lam_pg.abs().max(1.0),
        "the |G|>1 (point-group) and |G|=1 (legacy C1) MALS arms must agree: {lam_pg} vs {lam_c1}"
    );
    assert!(
        (lam_pg - e_exact).abs() <= 1e-8 * e_exact.abs().max(1.0),
        "|G|>1 MALS vs the exact FCI electronic energy: {lam_pg} vs {e_exact}"
    );
    assert!(
        (lam_c1 - e_exact).abs() <= 1e-8 * e_exact.abs().max(1.0),
        "|G|=1 MALS vs the exact FCI electronic energy: {lam_c1} vs {e_exact}"
    );
}

#[test]
fn pg6_h2o_als_matches_c1_load() {
    let meta = h2o_meta();
    let (mut data_pg, pgdata) = load_h2o(None);
    screen_forbidden(&mut data_pg.t, &mut data_pg.v, pgdata.pg);
    let space_pg = data_pg
        .qn_space(None)
        .unwrap_or_else(|e| panic!("qn_space: {e}"));

    let data_c1 = load_h2o_c1();
    let space_c1 = QnSpace::from_n_two_sz(data_c1.norb, data_c1.nelec, data_c1.ms2);

    let h_pg: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_pg,
        &data_pg.t,
        &data_pg.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );
    let h_c1: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space_c1,
        &data_c1.t,
        &data_c1.v,
        None,
        f64::EPSILON,
        Construction::Bipartite,
    );

    let opts = AlsOptions {
        eps: 1e-10,
        max_iter: 60,
        ..Default::default()
    };
    let x0_pg: TtVector<f64> = capped_randn(space_pg, 1024, 1024, 76_005);
    let x0_c1: TtVector<f64> = capped_randn(space_c1, 1024, 1024, 76_005);
    let (lam_pg, x_pg) = als(&h_pg, &x0_pg, &opts);
    let (lam_c1, _x_c1) = als(&h_c1, &x0_c1, &opts);
    assert!(x_pg.check());

    let e_exact = meta_f64(&meta, "e_fci") - meta_f64(&meta, "e_nuc");
    assert!(
        lam_pg >= e_exact - 1e-8,
        "|G|>1 ALS must not undershoot the exact energy: {lam_pg} < {e_exact}"
    );
    assert!(
        lam_c1 >= e_exact - 1e-8,
        "|G|=1 ALS must not undershoot the exact energy: {lam_c1} < {e_exact}"
    );
    assert!(
        (lam_pg - lam_c1).abs() <= 1e-8 * lam_pg.abs().max(1.0),
        "the |G|>1 and |G|=1 ALS arms must agree (whether or not they hit the exact floor): \
         {lam_pg} vs {lam_c1}"
    );
    assert!(
        (lam_pg - e_exact).abs() <= 1e-8 * e_exact.abs().max(1.0),
        "|G|>1 full-rank ALS vs the exact FCI electronic energy: {lam_pg} vs {e_exact}"
    );
    assert!(
        (lam_c1 - e_exact).abs() <= 1e-8 * e_exact.abs().max(1.0),
        "|G|=1 full-rank ALS vs the exact FCI electronic energy: {lam_c1} vs {e_exact}"
    );
}
