// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::panic::AssertUnwindSafe;
use std::path::Path;

use qntt::hamiltonian::Construction;
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::{SparseHamiltonian, TtVector, rayleigh_quotient, tt_state_in};

const H2O: &str = "testdata/h2o_c2v";
const N2: &str = "testdata/n2_d2h";

fn resolve_opts(screen: bool) -> IntegralsOptions {
    IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen,
        ..IntegralsOptions::default()
    }
}

#[test]
#[should_panic(expected = "totally symmetric")]
fn pg_gate_poisoned_forbidden_cell_panics_at_construction() {
    let mut load =
        load_fixture_dir::<f64>(Path::new(H2O), &resolve_opts(true)).expect("fixture loads");
    let pg = load.pg.pg;
    let d = load.ints.norb;

    let (i, j) = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .find(|&(i, j)| pg.irrep_of(i) != pg.irrep_of(j))
        .expect("a |G| > 1 fixture has at least one forbidden one-body cell");
    load.ints.t.set(&[i, j], 1e-8);

    let _h: SparseHamiltonian<f64> =
        SparseHamiltonian::new(load.space, &load.ints.t, &load.ints.v, None, f64::EPSILON);
}

#[test]
fn pg_gate_unscreened_real_dump_panics_at_construction() {
    let load = load_fixture_dir::<f64>(Path::new(N2), &resolve_opts(false)).expect("fixture loads");
    let pg = load.pg.pg;
    let d = load.ints.norb;

    let mut residue = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of(i) != pg.irrep_of(j) {
                residue = residue.max(load.ints.t.get(&[i, j]).abs());
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != qntt::Irrep::IDENTITY {
                        residue = residue.max(load.ints.v.get(&[i, j, k, l]).abs());
                    }
                }
            }
        }
    }
    if residue == 0.0 {
        eprintln!("skip: {N2} carries no forbidden residue — nothing for the gate to catch");
        return;
    }

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _h: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
            load.space,
            &load.ints.t,
            &load.ints.v,
            None,
            f64::EPSILON,
            Construction::Bipartite,
        );
    }));
    match result {
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic payload>");
            assert!(
                msg.contains("totally symmetric"),
                "construction must fail through the symmetry gate, got: {msg}"
            );
        }
        Ok(()) => panic!(
            "unscreened dump (forbidden residue {residue:.3e} > eps) must fail at construction"
        ),
    }
}

#[test]
fn pg_gate_screened_build_and_matvec_unchanged() {
    let load = load_fixture_dir::<f64>(Path::new(H2O), &resolve_opts(true)).expect("fixture loads");
    let meta = load.meta.as_ref().expect("fixture dirs carry meta");
    let d = load.ints.norb;
    let nocc = load.ints.nelec / 2;

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new(load.space, &load.ints.t, &load.ints.v, None, f64::EPSILON);

    let mut occ = vec![false; d];
    occ[..nocc].fill(true);
    let x: TtVector<f64> = tt_state_in(load.space, &occ, &occ);
    assert!(x.check());

    let rq = rayleigh_quotient(&h, &x);
    let get = |k: &str| -> f64 { meta.keys[k].trim().parse().expect("numeric meta key") };
    let want = get("e_hf") - get("e_nuc");
    assert!(
        (rq - want).abs() <= 1e-6 * want.abs().max(1.0),
        "HF Rayleigh quotient {rq} vs meta e_hf - e_nuc = {want}"
    );
}

#[test]
fn pg_gate_vacuous_at_trivial_group() {
    let load = load_fixture_dir::<f64>(
        Path::new(H2O),
        &IntegralsOptions {
            point_group: PointGroupPolicy::Ignore,
            ..IntegralsOptions::default()
        },
    )
    .expect("fixture loads");
    let d = load.ints.norb;
    let nocc = load.ints.nelec / 2;

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new(load.space, &load.ints.t, &load.ints.v, None, f64::EPSILON);
    let mut occ = vec![false; d];
    occ[..nocc].fill(true);
    let x: TtVector<f64> = tt_state_in(load.space, &occ, &occ);
    let rq = rayleigh_quotient(&h, &x);
    let meta = load.meta.as_ref().expect("fixture dirs carry meta");
    let get = |k: &str| -> f64 { meta.keys[k].trim().parse().expect("numeric meta key") };
    let want = get("e_hf") - get("e_nuc");
    assert!(
        (rq - want).abs() <= 1e-6 * want.abs().max(1.0),
        "HF Rayleigh quotient {rq} vs meta e_hf - e_nuc = {want}"
    );
}
