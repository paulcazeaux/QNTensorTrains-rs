// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg6::util::*;

#[test]
fn pg6_h2o_fci_matches_pyscf() {
    let meta = h2o_meta();
    let (mut data, pg) = load_h2o(None);
    let census = screen_forbidden(&mut data.t, &mut data.v, pg.pg);
    assert_eq!(
        census,
        (0, 0.0, 0, 0.0),
        "H2O fixture unexpectedly carries forbidden entries (see T7's negative result)"
    );

    let d = data.norb;
    let nup = ((data.nelec as i64 + data.ms2) / 2) as usize;
    let ndn = data.nelec - nup;

    let (dets, lambda_block) =
        target_block_ground_state_sparse(&pg.pg, d, &data.t, &data.v, nup, ndn, pg.target);
    assert_eq!(
        dets.len(),
        133,
        "the A1 target-irrep block must have exactly 133 of the 441 (Nup,Ndn)=(5,5) determinants \
         (measured by the probe; 0 < 133 < 441 pins both non-vacuity and properness)"
    );

    let e_total = lambda_block + data.e0;
    let e_fci = meta_f64(&meta, "e_fci");
    assert!(
        (e_total - e_fci).abs() < 1e-8,
        "sparse-oracle FCI energy {e_total} (block {lambda_block} + e0 {}) vs PySCF's own e_fci \
         {e_fci} — this is the cross-code physics pin of the whole campaign",
        data.e0
    );
}
