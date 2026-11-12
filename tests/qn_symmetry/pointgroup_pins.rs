// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::{Irrep, PointGroupKind};
use std::path::{Path, PathBuf};

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/pointgroups/tables.txt")
}

#[derive(Debug)]
struct FixtureGroup {
    name: String,
    order: usize,
    names_molpro: Vec<String>,
    molpro_to_xor: Vec<u8>,
    product: Vec<Vec<u8>>,
}

fn parse_fixture(text: &str) -> Vec<FixtureGroup> {
    let mut lines = text
        .lines()
        .filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        })
        .peekable();

    let mut groups = Vec::new();
    while let Some(line) = lines.next() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(
            toks.first().copied(),
            Some("group"),
            "expected a 'group' line, got: {line:?}"
        );
        assert_eq!(toks.len(), 4, "malformed group line: {line:?}");
        let name = toks[1].to_string();
        assert_eq!(toks[2], "order", "malformed group line: {line:?}");
        let order: usize = toks[3]
            .parse()
            .unwrap_or_else(|e| panic!("bad order in {line:?}: {e}"));

        let names_line = lines
            .next()
            .unwrap_or_else(|| panic!("missing names_molpro_order line after group {name}"));
        let names_toks: Vec<&str> = names_line.split_whitespace().collect();
        assert_eq!(
            names_toks.first().copied(),
            Some("names_molpro_order"),
            "expected names_molpro_order line for group {name}, got: {names_line:?}"
        );
        let names_molpro: Vec<String> = names_toks[1..].iter().map(|s| s.to_string()).collect();
        assert_eq!(
            names_molpro.len(),
            order,
            "group {name}: names_molpro_order count mismatch"
        );

        let m2x_line = lines
            .next()
            .unwrap_or_else(|| panic!("missing molpro_to_xor line after group {name}"));
        let m2x_toks: Vec<&str> = m2x_line.split_whitespace().collect();
        assert_eq!(
            m2x_toks.first().copied(),
            Some("molpro_to_xor"),
            "expected molpro_to_xor line for group {name}, got: {m2x_line:?}"
        );
        let molpro_to_xor: Vec<u8> = m2x_toks[1..]
            .iter()
            .map(|s| {
                s.parse()
                    .unwrap_or_else(|e| panic!("bad molpro_to_xor entry for group {name}: {e}"))
            })
            .collect();
        assert_eq!(
            molpro_to_xor.len(),
            order,
            "group {name}: molpro_to_xor count mismatch"
        );

        let header_line = lines
            .next()
            .unwrap_or_else(|| panic!("missing product_table_molpro marker after group {name}"));
        assert_eq!(
            header_line.trim(),
            "product_table_molpro",
            "expected product_table_molpro marker for group {name}, got: {header_line:?}"
        );

        let mut product = Vec::with_capacity(order);
        for r in 0..order {
            let row_line = lines
                .next()
                .unwrap_or_else(|| panic!("missing product_table_molpro row {r} for group {name}"));
            let row: Vec<u8> = row_line
                .split_whitespace()
                .map(|s| {
                    s.parse().unwrap_or_else(|e| {
                        panic!("bad product entry for group {name} row {r}: {e}")
                    })
                })
                .collect();
            assert_eq!(
                row.len(),
                order,
                "group {name}: product_table_molpro row {r} length mismatch"
            );
            product.push(row);
        }

        let mut sorted_perm = molpro_to_xor.clone();
        sorted_perm.sort_unstable();
        let expected_perm: Vec<u8> = (0..order as u8).collect();
        assert_eq!(
            sorted_perm, expected_perm,
            "group {name}: molpro_to_xor is not a permutation of 0..{order}"
        );
        for (i, row) in product.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert!(
                    v >= 1 && (v as usize) <= order,
                    "group {name}: product_table_molpro[{i}][{j}]={v} out of range 1..={order}"
                );
            }
        }

        groups.push(FixtureGroup {
            name,
            order,
            names_molpro,
            molpro_to_xor,
            product,
        });
    }

    assert_eq!(
        groups.len(),
        8,
        "expected exactly 8 groups in the fixture, found {}",
        groups.len()
    );
    groups
}

fn load_fixture() -> Vec<FixtureGroup> {
    let path = fixture_path();
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    parse_fixture(&text)
}

#[test]
fn pin_fixture_coverage_and_ordering() {
    let fixture = load_fixture();
    assert_eq!(fixture.len(), 8, "fixture must contain exactly 8 groups");

    for (kind, fx) in PointGroupKind::ALL.into_iter().zip(fixture.iter()) {
        assert_eq!(
            kind.name(),
            fx.name,
            "PointGroupKind::ALL order must match the fixture block order"
        );
        assert_eq!(
            PointGroupKind::from_name(&fx.name),
            Some(kind),
            "from_name({:?})",
            fx.name
        );
        assert_eq!(
            PointGroupKind::from_name(&fx.name.to_lowercase()),
            Some(kind),
            "from_name (lowercase) {:?}",
            fx.name.to_lowercase()
        );
        assert_eq!(
            PointGroupKind::from_name(&fx.name.to_uppercase()),
            Some(kind),
            "from_name (uppercase) {:?}",
            fx.name.to_uppercase()
        );
    }
}

#[test]
fn pin_molpro_to_xor_permutation() {
    let fixture = load_fixture();
    for (kind, fx) in PointGroupKind::ALL.into_iter().zip(fixture.iter()) {
        assert_eq!(
            kind.order() as usize,
            fx.order,
            "{}: order mismatch",
            kind.name()
        );
        assert_eq!(
            kind.order(),
            1u8 << kind.nbits(),
            "{}: order != 1 << nbits",
            kind.name()
        );

        for m in 1..=fx.order {
            let expected = Irrep::new(fx.molpro_to_xor[m - 1]);
            let got = kind
                .irrep_from_molpro(m as u8)
                .unwrap_or_else(|e| panic!("{}: molpro id {m}: {e}", kind.name()));
            assert_eq!(got, expected, "{}: molpro id {m}", kind.name());

            let back = kind
                .molpro_from_irrep(expected)
                .unwrap_or_else(|e| panic!("{}: xor {expected:?}: {e}", kind.name()));
            assert_eq!(
                back as usize,
                m,
                "{}: molpro_from_irrep is not the inverse of molpro_to_xor at molpro id {m}",
                kind.name()
            );
        }
    }
}

#[test]
fn pin_product_table_molpro() {
    let fixture = load_fixture();
    let mut checks = 0usize;
    for (kind, fx) in PointGroupKind::ALL.into_iter().zip(fixture.iter()) {
        for i in 1..=fx.order {
            for j in 1..=fx.order {
                let a = kind.irrep_from_molpro(i as u8).unwrap();
                let b = kind.irrep_from_molpro(j as u8).unwrap();
                let got = kind.molpro_from_irrep(a.fuse(b)).unwrap();
                assert_eq!(
                    got as usize,
                    fx.product[i - 1][j - 1] as usize,
                    "{}: {} x {}",
                    kind.name(),
                    i,
                    j
                );
                checks += 1;
            }
        }
    }
    assert_eq!(
        checks, 125,
        "non-vacuity guard: expected 125 product-table checks across all 8 groups, got {checks}"
    );
}

#[test]
fn pin_irrep_names_and_negative_control() {
    let fixture = load_fixture();

    for (kind, fx) in PointGroupKind::ALL.into_iter().zip(fixture.iter()) {
        let names_molpro = kind.irrep_names_molpro();
        assert_eq!(
            names_molpro.len(),
            fx.order,
            "{}: irrep_names_molpro length",
            kind.name()
        );
        for m in 1..=fx.order {
            assert_eq!(
                names_molpro[m - 1],
                fx.names_molpro[m - 1],
                "{}: name at molpro id {m}",
                kind.name()
            );
            let ir = kind.irrep_from_molpro(m as u8).unwrap();
            let name = kind
                .irrep_name(ir)
                .unwrap_or_else(|e| panic!("{}: irrep_name({ir:?}): {e}", kind.name()));
            assert_eq!(
                name,
                fx.names_molpro[m - 1],
                "{}: irrep_name(irrep_from_molpro({m})) name",
                kind.name()
            );
        }

        let names_xor = kind.irrep_names_xor();
        assert_eq!(
            names_xor.len(),
            fx.order,
            "{}: irrep_names_xor length",
            kind.name()
        );
        for m in 1..=fx.order {
            let x = fx.molpro_to_xor[m - 1] as usize;
            assert_eq!(
                names_xor[x],
                fx.names_molpro[m - 1],
                "{}: irrep_names_xor[{x}] vs molpro id {m}",
                kind.name()
            );
        }
    }

    let d2h_fx = fixture
        .iter()
        .find(|fx| fx.name == "D2h")
        .expect("D2h block in fixture");
    let id_perm: Vec<u8> = (0..d2h_fx.order as u8).collect();
    assert_ne!(
        id_perm, d2h_fx.molpro_to_xor,
        "D2h's molpro_to_xor must differ from the identity permutation for the negative control to mean anything"
    );

    let d2h = PointGroupKind::D2h;
    let names_xor = d2h.irrep_names_xor();
    assert_eq!(
        d2h_fx.names_molpro[1], "B3u",
        "D2h Molpro id 2 sanity anchor"
    );
    assert_ne!(
        d2h_fx.names_molpro[1], names_xor[1],
        "identity-permutation naming must NOT match the true fixture naming"
    );
}

#[test]
fn pin_fixture_molpro_to_xor_is_gf2_linear() {
    let fixture = load_fixture();
    for fx in &fixture {
        let n = fx.order as u8;
        for i in 0..n {
            for j in 0..n {
                let lhs = fx.molpro_to_xor[(i ^ j) as usize];
                let rhs = fx.molpro_to_xor[i as usize] ^ fx.molpro_to_xor[j as usize];
                assert_eq!(
                    lhs, rhs,
                    "{}: molpro_to_xor GF(2)-linearity at i={i} j={j}",
                    fx.name
                );
            }
        }
    }
}
