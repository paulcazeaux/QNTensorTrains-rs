// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::{Path, PathBuf};

use qntt::integrals::{
    FixtureMeta, IntegralsOptions, load_fixture_dir, load_property_matching, read_meta,
};
use qntt::qn::QnSpace;
use qntt::solvers::joint::cas::{CasPartition, cas_rank_grid};

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

fn meta_usize(meta: &FixtureMeta, key: &str) -> usize {
    meta.keys
        .get(key)
        .unwrap_or_else(|| panic!("meta is missing `{key}`"))
        .parse()
        .unwrap_or_else(|e| panic!("meta `{key}` does not parse as usize: {e}"))
}

#[test]
fn tcf_cas_meta_partition_and_grid_roundtrip() {
    for (name, d, nelec, cas, n_core) in [
        ("h2o_sto3g_cas", 7usize, 10usize, (8usize, 6usize), 1usize),
        ("h2o_631g_cas", 13, 10, (8, 6), 1),
        ("hcho_631g_cas", 22, 16, (4, 3), 6),
        ("c2h4_631g_cas", 26, 16, (2, 2), 7),
    ] {
        let load = load_fixture_dir::<f64>(&fixture_dir(name), &IntegralsOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let meta = load
            .meta
            .as_ref()
            .expect("fixture dirs always carry a meta");
        assert_eq!((load.ints.norb, load.ints.nelec), (d, nelec), "{name}");

        assert_eq!(
            meta.keys.get("mo_source").map(String::as_str),
            Some("casscf"),
            "{name}"
        );
        let fp = meta
            .keys
            .get("phase_fingerprint")
            .unwrap_or_else(|| panic!("{name}: meta has no phase_fingerprint"));
        assert_eq!(
            fp.len(),
            16,
            "{name}: fingerprint is 16 hex chars, got {fp:?}"
        );
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()), "{name}: {fp:?}");

        assert_eq!(meta_usize(meta, "cas_nelec"), cas.0, "{name}");
        assert_eq!(meta_usize(meta, "cas_norb"), cas.1, "{name}");
        let part = CasPartition {
            n_core: meta_usize(meta, "n_core"),
            n_active: meta_usize(meta, "n_active"),
            n_virt: meta_usize(meta, "n_virt"),
        };
        assert_eq!(part.n_core, n_core, "{name}");
        assert_eq!(part.n_active, cas.1, "{name}");
        assert_eq!(part.d(), d, "{name}: partition must cover the chain");

        let casscf_gs: f64 = meta
            .keys
            .get("casscf_gs")
            .unwrap_or_else(|| panic!("{name}: meta has no casscf_gs"))
            .parse()
            .unwrap();
        assert!(
            casscf_gs.is_finite() && casscf_gs < 0.0,
            "{name}: casscf_gs = {casscf_gs}"
        );
        let e_hf = meta.e_hf.expect("keyed metas carry e_hf");
        assert!(
            casscf_gs < e_hf && casscf_gs > 2.0 * e_hf,
            "{name}: casscf_gs = {casscf_gs} vs e_hf = {e_hf}"
        );

        let space: QnSpace = load.space;
        part.validate(&space);
        let grid = cas_rank_grid(&space, &part, qntt::PerSector::NONE);
        assert_eq!(grid.len(), d + 1, "{name}");
        assert_eq!(grid[0].total(), 1, "{name}: boundary law at bond 0");
        assert_eq!(grid[d].total(), 1, "{name}: boundary law at bond d");
    }
}

#[test]
fn tcf_c2h4_sto3g_loads() {
    let load = load_fixture_dir::<f64>(&fixture_dir("c2h4_sto3g"), &IntegralsOptions::default())
        .expect("c2h4_sto3g");
    let meta = load.meta.as_ref().unwrap();
    assert_eq!((load.ints.norb, load.ints.nelec), (14, 16));
    assert_eq!(meta.keys.get("mo_source").map(String::as_str), Some("rhf"));
    assert!(
        meta.keys.contains_key("e_ccsd_t"),
        "d=14 is above the FCI cap"
    );
    let distinct: std::collections::HashSet<i64> = meta.orbsym.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "ORBSYM must be nontrivial: {:?}",
        meta.orbsym
    );
}

#[test]
fn tcf_prop_fingerprint_match_loads() {
    for (name, d) in [
        ("h2o_sto3g_cas", 7usize),
        ("h2o_631g_cas", 13),
        ("hcho_631g_cas", 22),
        ("c2h4_631g_cas", 26),
    ] {
        let dir = fixture_dir(name);
        let meta = read_meta(&dir.join("meta.txt")).unwrap();
        for comp in ["x", "y", "z"] {
            let p =
                load_property_matching::<f64>(&dir.join(format!("dipole_{comp}.prop")), d, &meta)
                    .unwrap_or_else(|e| panic!("{name} dipole_{comp}: {e}"));
            assert_eq!(p.label, format!("dipole_{comp}"), "{name}");
            let max = (0..d)
                .flat_map(|i| (0..d).map(move |j| (i, j)))
                .map(|(i, j)| p.m.get(&[i, j]).abs())
                .fold(0.0f64, f64::max);
            assert!(max > 1e-6, "{name} dipole_{comp}: max|m| = {max:e}");
        }
    }
}

#[test]
fn tcf_prop_fingerprint_mismatch_refused() {
    let dir = fixture_dir("h2o_sto3g_cas");
    let meta = read_meta(&dir.join("meta.txt")).unwrap();
    let text = std::fs::read_to_string(dir.join("dipole_z.prop")).unwrap();
    let fp = meta.keys.get("phase_fingerprint").unwrap();
    assert!(
        text.contains(fp.as_str()),
        "the real pair shares the fingerprint"
    );

    let tmp = std::env::temp_dir().join(format!("qntt_tcf_prop_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    let bad = tmp.join("tampered.prop");
    std::fs::write(&bad, text.replace(fp.as_str(), "deadbeefdeadbeef")).unwrap();
    let err = load_property_matching::<f64>(&bad, 7, &meta)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("deadbeefdeadbeef") && err.contains(fp.as_str()),
        "mismatch refusal must name both fingerprints: {err}"
    );

    let stripped = tmp.join("stripped.prop");
    let no_fp: String = text
        .lines()
        .filter(|l| !l.starts_with("phase_fingerprint"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(&stripped, no_fp).unwrap();
    let err = load_property_matching::<f64>(&stripped, 7, &meta)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("no phase_fingerprint"),
        "missing-fingerprint refusal: {err}"
    );

    let lih_meta = read_meta(&fixture_dir("lih_sto3g").join("meta.txt")).unwrap();
    assert!(
        !lih_meta.keys.contains_key("phase_fingerprint"),
        "lih_sto3g predates fingerprints; if this changed, re-point arm (c)"
    );
    let tiny = tmp.join("tiny.prop");
    std::fs::write(&tiny, "label t\n0.5 1 2\n0.5 2 1\n").unwrap();
    let p = load_property_matching::<f64>(&tiny, 6, &lih_meta).expect("plain load");
    assert_eq!(p.label, "t");

    std::fs::remove_dir_all(&tmp).ok();
}
