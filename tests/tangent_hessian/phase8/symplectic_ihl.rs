// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase8::util::*;
use qntt::QnSpace;
use qntt::solvers::hess::{
    BasisSelection, SympLobpcgOpts, SympPrecond, SymplecticLowestInfo, symplectic_lowest_lobp4dcg,
    weingarten_operator,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn assert_run_identical(
    tag: &str,
    (om_a, info_a): (&[f64], &SymplecticLowestInfo),
    (om_b, info_b): (&[f64], &SymplecticLowestInfo),
) {
    assert_eq!(om_a.len(), om_b.len(), "[{tag}] ω length");
    for (j, (x, y)) in om_a.iter().zip(om_b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "[{tag}] ω[{j}] not bit-identical: {x:.17e} vs {y:.17e}"
        );
    }
    assert_eq!(
        info_a.residuals.len(),
        info_b.residuals.len(),
        "[{tag}] residual history length"
    );
    for (j, (x, y)) in info_a
        .residuals
        .iter()
        .zip(info_b.residuals.iter())
        .enumerate()
    {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "[{tag}] residual[{j}] not bit-identical: {x:.17e} vs {y:.17e}"
        );
    }
    assert_eq!(info_a.iterations, info_b.iterations, "[{tag}] built count");
    assert_eq!(info_a.applies, info_b.applies, "[{tag}] apply count");
    assert_eq!(info_a.converged, info_b.converged, "[{tag}] converged flag");
    if let (Some(da), Some(db)) = (info_a.diag.as_ref(), info_b.diag.as_ref()) {
        assert_eq!(da.outer_iterations, db.outer_iterations, "[{tag}] outer");
        assert_eq!(da.drops_rank, db.drops_rank, "[{tag}] drops_rank");
        assert_eq!(da.drops_wnull, db.drops_wnull, "[{tag}] drops_wnull");
        assert_eq!(da.drops_zero, db.drops_zero, "[{tag}] drops_zero");
        assert_eq!(da.drops_triple, db.drops_triple, "[{tag}] drops_triple");
        assert_eq!(da.survival, db.survival, "[{tag}] survival history");
        assert_eq!(
            da.collapse_angle, db.collapse_angle,
            "[{tag}] collapse angles"
        );
    }
}

#[test]
fn tihl_legacy_default_unchanged() {
    assert_eq!(
        BasisSelection::default(),
        BasisSelection::Legacy,
        "the certified pre-phase basis selection is the enum's own default"
    );
    assert_eq!(
        SympLobpcgOpts::default().basis,
        BasisSelection::Legacy,
        "D8 ships OPT-IN: promoting Ihl to default is a deliberate edit here, with the measured \
         table that promotion requires, never a silent drift"
    );

    let space = QnSpace::new(4, 2, 1);
    let (ham, x, base_l, base_r) = weingarten_minimizer(space, 0xB0B1, 1, 2);
    let ctx = weingarten_operator(&ham, &x);
    let k = 4usize;

    let run = |basis: Option<BasisSelection>| {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81B1);
        let v0s = seed_block::<f64>(space, &base_l, &base_r, k + 2, &mut rng);
        let opts = match basis {
            None => SympLobpcgOpts {
                precond: SympPrecond::Off,
                ..SympLobpcgOpts::default()
            },
            Some(b) => SympLobpcgOpts {
                precond: SympPrecond::Off,
                basis: b,
                ..SympLobpcgOpts::default()
            },
        };
        symplectic_lowest_lobp4dcg(&ctx, &v0s, k, &opts).expect("stable d=4 fixture")
    };

    let (om_1, info_1) = run(Some(BasisSelection::Legacy));
    let (om_2, info_2) = run(Some(BasisSelection::Legacy));
    let (om_d, info_d) = run(None);
    println!(
        "tihl_legacy_default_unchanged: built={} applies={} converged={}",
        info_1.iterations, info_1.applies, info_1.converged
    );
    assert!(
        info_1.converged,
        "the d=4 truncated fixture converges on the Legacy/Off lane (certified by \
         t8m_lobp4dcg_matches_dense_truncated)"
    );
    assert_run_identical("legacy-repeat", (&om_1, &info_1), (&om_2, &info_2));
    assert_run_identical("legacy-vs-omitted", (&om_1, &info_1), (&om_d, &info_d));
}
