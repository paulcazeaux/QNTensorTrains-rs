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
use qntt::{SparseHamiltonian, TtVector};

#[test]
fn pg6_h2o_converged_state_is_exactly_pure() {
    let (mut data, pgdata) = load_h2o(None);
    screen_forbidden(&mut data.t, &mut data.v, pgdata.pg);
    let space = data
        .qn_space(None)
        .unwrap_or_else(|e| panic!("qn_space: {e}"));
    let h: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
        space,
        &data.t,
        &data.v,
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
    let x0: TtVector<f64> = capped_randn(space, 1, 4, 76_006);
    let (_lam, x_mals) = mals(&h, &x0, &opts);
    assert!(x_mals.check());

    let nup = ((data.nelec as i64 + data.ms2) / 2) as usize;
    let ndn = data.nelec - nup;
    assert_purity(
        &pgdata.pg,
        data.norb,
        nup,
        ndn,
        pgdata.target,
        &x_mals.to_dense(),
    );
}
