// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;

use qntt::fcidump::FcidumpError;
use qntt::integrals::{
    PropertyOperator, load_property, parse_property, property_hamiltonian, su2_property_hamiltonian,
};
use qntt::su2::expand::expand;
use qntt::su2::ops::dot as su2_dot;
use qntt::su2::qn::{Su2RankGrid, Su2Space};
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::{CoeffTensor, QnSpace, TtVector, dot as ab_dot, tt_state};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::load_h2o;
use crate::util::*;

#[test]
fn tjq_property_roundtrip() {
    let text = "\
label dipole_z
# a comment line, ignored
1.5 2 1
1.5 1 2
-0.25 4 4
";
    let p = parse_property::<f64>(text, 4).expect("a well-formed .prop text must parse");
    assert_eq!(p.label, "dipole_z", "label must round-trip verbatim");
    assert_eq!(p.m.d, 4);
    assert_eq!(p.m.rank, 2);

    assert_close(p.m.get(&[1, 0]), 1.5, 1e-15, "record `1.5 2 1` -> m[1][0]");
    assert_close(p.m.get(&[0, 1]), 1.5, 1e-15, "record `1.5 1 2` -> m[0][1]");
    assert_close(
        p.m.get(&[3, 3]),
        -0.25,
        1e-15,
        "record `-0.25 4 4` -> m[3][3]",
    );

    for i in 0..4 {
        for j in 0..4 {
            if (i, j) == (1, 0) || (i, j) == (0, 1) || (i, j) == (3, 3) {
                continue;
            }
            assert_eq!(
                p.m.get(&[i, j]),
                0.0,
                "unset cell [{i}][{j}] must stay zero"
            );
        }
    }
}

#[test]
fn tjq_property_malformed_lines_skipped() {
    let text = "label ok\nnotanumber 2 1\n1.0 2 1 3\n1.0 2 1\n1.0 1 2\n";
    let p = parse_property::<f64>(text, 4).expect("malformed lines must be skipped, not fatal");
    assert_close(
        p.m.get(&[1, 0]),
        1.0,
        1e-15,
        "only the well-formed record must land",
    );
    assert_close(
        p.m.get(&[0, 1]),
        1.0,
        1e-15,
        "the symmetric partner record must also land",
    );
}

#[test]
fn tjq_dense_oracle() {
    let d = 4;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD0D0_0001);
    let m = random_sym_t(d, &mut rng);
    let p = PropertyOperator {
        label: "test_dense".to_string(),
        m: m.clone(),
    };
    let h = property_hamiltonian(space, &p);

    let dets = dets_for(d, space.nup, space.ndn);
    let zeros4 = CoeffTensor::<f64>::zeros(d, 4);
    let dense = dense_hamiltonian_matrix(d, &m, &zeros4, &dets);

    for (col, (ou, od)) in dets.iter().enumerate() {
        let x: TtVector<f64> = tt_state(ou, od);
        let hx = h.apply(&x);
        for (row, (ru, rd)) in dets.iter().enumerate() {
            let got = ab_dot(&tt_state::<f64>(ru, rd), &hx);
            let want = dense[row][col];
            assert_close(
                got,
                want,
                1e-12 * want.abs().max(1.0),
                &format!("property_hamiltonian dense cross-check row={row} col={col}"),
            );
        }
    }
}

#[test]
fn tjq_exact_linearity_eps_zero_pin() {
    let d = 4;
    let space = QnSpace::from_n_two_sz(d, 3, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x11EA_0001);
    let m = random_sym_t(d, &mut rng);
    let s = 1e-17;
    let ms = scale_coeff(&m, s);

    let p1 = PropertyOperator {
        label: "m".to_string(),
        m,
    };
    let p2 = PropertyOperator {
        label: "s_m".to_string(),
        m: ms,
    };
    let h1 = property_hamiltonian(space, &p1);
    let h2 = property_hamiltonian(space, &p2);

    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0x11EA_0002);
    let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng2);
    let y1 = h1.apply(&x);
    let y2 = h2.apply(&x);
    let e1 = ab_dot(&x, &y1);
    let e2 = ab_dot(&x, &y2);
    assert_close(
        e2,
        s * e1,
        1e-6 * (s * e1).abs().max(1e-25),
        "property_hamiltonian(space, s*m) must be exactly s * property_hamiltonian(space, m) at \
         eps=0.0, even for s below f64::EPSILON",
    );
}

#[test]
fn tjq_su2_twin() {
    let d = 5;
    let n = 5;
    let two_s = 1usize;
    let space = Su2Space::new(d, n, two_s);
    let ab_space = QnSpace::from_n_two_sz(d, n, two_s as i64);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5502_0001);
    let m = random_sym_t(d, &mut rng);
    let p = PropertyOperator {
        label: "su2_test".to_string(),
        m,
    };
    let h_su2 = su2_property_hamiltonian(space, &p);
    let h_ab = property_hamiltonian(ab_space, &p);

    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(1..=3)
                };
            }
            r
        })
        .collect();
    let x: Su2TtVector<f64> = su2_tt_randn(space, &ranks, &mut rng);
    let x_exp: TtVector<f64> = expand(&x, two_s as i64);

    let hx_su2 = &h_su2 * &x;
    let e_su2 = su2_dot(&x, &hx_su2);
    let hx_ab = &h_ab * &x_exp;
    let e_ab = ab_dot(&x_exp, &hx_ab);

    assert_close(
        e_su2,
        e_ab,
        1e-10 * e_ab.abs().max(1.0),
        "su2_property_hamiltonian expectation vs abelian property_hamiltonian through expand",
    );
}

#[test]
#[should_panic(expected = "cross_irrep_probe")]
fn tjq_pg_refusal() {
    let load = load_h2o();
    let pg = load.pg.pg;
    let d = load.ints.norb;
    let space = load.space;
    let (p, q) = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .find(|&(i, j)| pg.irrep_of(i) != pg.irrep_of(j))
        .expect("a |G| > 1 fixture has at least one cross-irrep pair");

    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    m.set(&[p, q], 1.0);
    m.set(&[q, p], 1.0);
    let prop = PropertyOperator {
        label: "cross_irrep_probe".to_string(),
        m,
    };
    let _ = property_hamiltonian(space, &prop);
}

#[test]
fn tjq_load_property_missing_file_is_io_error() {
    match load_property::<f64>(Path::new("/nonexistent/property.txt"), 4) {
        Err(FcidumpError::Io(_)) => {}
        other => panic!("expected FcidumpError::Io, got {other:?}"),
    }
}

#[test]
fn tjq_property_index_past_norb_is_dimension_mismatch() {
    let text = "label big\n1.0 5 1\n1.0 1 5\n";
    match parse_property::<f64>(text, 4) {
        Err(FcidumpError::DimensionMismatch(msg)) => {
            assert!(msg.contains("norb=4"), "message should name norb: {msg}");
        }
        other => panic!("expected FcidumpError::DimensionMismatch, got {other:?}"),
    }
}

#[test]
fn tjq_property_missing_label_key_is_argument_error() {
    match parse_property::<f64>("1.0 2 1\n1.0 1 2\n", 4) {
        Err(FcidumpError::ArgumentError(msg)) => {
            assert!(msg.contains("label"), "message should name the key: {msg}");
        }
        other => panic!("expected FcidumpError::ArgumentError, got {other:?}"),
    }
}

#[test]
#[should_panic(expected = "must be rank-2")]
fn tjq_property_operator_refuses_non_rank2() {
    let m = CoeffTensor::<f64>::zeros(4, 4);
    let _ = PropertyOperator::new("bad_rank", m);
}
