// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::PathBuf;

use qntt::fcidump::TwoBodySymmetry;
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::{QnSpace, SparseHamiltonian, rayleigh_quotient};

fn testdata(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(sub)
}

fn hf_determinant(space: QnSpace) -> qntt::TtVector<f64> {
    let occ_up: Vec<bool> = (0..space.d).map(|k| k < space.nup).collect();
    let occ_dn: Vec<bool> = (0..space.d).map(|k| k < space.ndn).collect();
    qntt::tt_state::<f64>(&occ_up, &occ_dn)
}

fn hf_energy(h: &SparseHamiltonian<f64>, space: QnSpace, e_core: f64) -> f64 {
    let hf = hf_determinant(space);
    rayleigh_quotient(h, &hf) / qntt::dot(&hf, &hf) + e_core
}

fn load_and_build(name: &str) -> (SparseHamiltonian<f64>, QnSpace, f64, Option<f64>) {
    let opts = IntegralsOptions::default();
    let load = load_fixture_dir::<f64>(&testdata(name), &opts).unwrap_or_else(|e| panic!("{e}"));
    let meta = load
        .meta
        .clone()
        .expect("every fixture dir ships a meta.txt");
    let e_core = load.ints.e0;
    let mut ints = load.ints;
    let (h, space, _) = ints
        .hamiltonian(None, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    (h, space, e_core, meta.e_hf)
}

const FIXTURES_FAST: [&str; 6] = [
    "lih_sto3g",
    "h2o_sto3g",
    "h2o_c2v",
    "ch2_triplet_c2v",
    "hcho_sto3g",
    "h2o_631g",
];
const FIXTURES_SLOW: [&str; 4] = ["hcho_631g", "c2h4_631g", "n2_ccpvdz", "n2_d2h"];

const FIXTURES_NOT_ENERGY_GATED: [&str; 11] = [
    "c2h4_631g_cas",
    "c2h4_sto3g",
    "h2o_631g_cas",
    "h2o_631g_stretch",
    "h2o_sto3g_cas",
    "h2o_sto3g_stretch",
    "h4_chain_stretch",
    "hcho_631g_cas",
    "joint_d2",
    "joint_d2_r22",
    "lih_sto3g_tr_legacy",
];

fn walk_fixture_dirs() -> Vec<String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata");
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .expect("testdata/ exists")
        .filter_map(|e| {
            let e = e.ok()?;
            let p = e.path();
            (p.is_dir() && p.join("meta.txt").is_file())
                .then(|| e.file_name().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    names
}

#[test]
fn i0_fixture_lists_cover_the_walk() {
    let mut curated: Vec<String> = FIXTURES_FAST
        .iter()
        .chain(FIXTURES_SLOW.iter())
        .chain(FIXTURES_NOT_ENERGY_GATED.iter())
        .map(|s| s.to_string())
        .collect();
    curated.sort();
    let walked = walk_fixture_dirs();
    assert_eq!(
        curated, walked,
        "the curated fixture lists and the directory walk disagree — classify the new (or          removed) fixture into FIXTURES_FAST, FIXTURES_SLOW or FIXTURES_NOT_ENERGY_GATED"
    );
    let mut dedup = curated.clone();
    dedup.dedup();
    assert_eq!(curated, dedup, "a fixture appears in two lists");
}

#[test]
fn i1_fixtures_reproduce_their_reference_hf_energy() {
    for name in FIXTURES_FAST {
        let (h, space, e_core, e_hf) = load_and_build(name);
        assert_eq!(space.d, h.space.d);
        let Some(e_hf) = e_hf else {
            println!("{name}: no e_hf in meta — build-only");
            continue;
        };
        let got = hf_energy(&h, space, e_core);
        assert!(
            (got - e_hf).abs() < 1e-8,
            "{name}: HF determinant energy {got} vs reference {e_hf} (Δ {:e})",
            got - e_hf
        );
    }
}

#[test]
fn i2_pairswap_on_a_packed_dump_is_not_the_same_hamiltonian() {
    let opts = IntegralsOptions {
        symmetry: TwoBodySymmetry::Pairswap,
        ..IntegralsOptions::default()
    };
    let load =
        load_fixture_dir::<f64>(&testdata("lih_sto3g"), &opts).unwrap_or_else(|e| panic!("{e}"));
    let e_hf = load
        .meta
        .as_ref()
        .and_then(|m| m.e_hf)
        .expect("lih_sto3g meta carries e_hf");
    let e_core = load.ints.e0;
    let mut ints = load.ints;
    let (h, space, _) = ints.hamiltonian(None, &opts).expect("build");
    let got = hf_energy(&h, space, e_core);
    assert!(
        (got - e_hf).abs() > 1e-3,
        "Pairswap on a packed dump produced HF energy {got}, indistinguishable from the correct \
         {e_hf} — the negative control has lost its teeth"
    );
}

#[test]
fn i5_every_fixture_directory_is_self_consistent() {
    let names = walk_fixture_dirs();
    assert!(names.len() >= 18, "fixture walk found only {names:?}");
    for name in names.iter() {
        let load = load_fixture_dir::<f64>(&testdata(name), &IntegralsOptions::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let meta = load.meta.expect("meta.txt");
        assert_eq!(meta.norb, load.ints.norb, "{name}: norb");
        assert_eq!(meta.nelec, load.ints.nelec, "{name}: nelec");
        assert_eq!(meta.ms2, load.ints.ms2, "{name}: ms2");
        assert!(
            (meta.e_core - load.ints.e0).abs() < 1e-12,
            "{name}: meta e_nuc {} vs the dump's core-energy sentinel {}",
            meta.e_core,
            load.ints.e0
        );

        let declared = meta.keys.get("point_group").map(String::as_str);
        if declared.is_some_and(|g| g != "C1") {
            let pg_opts = IntegralsOptions {
                point_group: PointGroupPolicy::Resolve(None),
                ..IntegralsOptions::default()
            };
            let pg = load_fixture_dir::<f64>(&testdata(name), &pg_opts)
                .unwrap_or_else(|e| panic!("{name} (point group): {e}"));
            assert!(
                pg.pg.pg.order() > 1,
                "{name} declares {declared:?} but resolved to |G| = 1"
            );
        }
    }
}

fn scratch_dir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("qntt_io_integrals_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).expect("scratch dir");
    p
}

#[test]
fn i6_write_fixture_dir_round_trips_lih_exactly() {
    use qntt::integrals::write_fixture_dir;
    let opts = IntegralsOptions::default();
    let src =
        load_fixture_dir::<f64>(&testdata("lih_sto3g"), &opts).unwrap_or_else(|e| panic!("{e}"));
    let dir = scratch_dir("i6");
    write_fixture_dir(
        &dir,
        "LIH_COPY.FCIDUMP",
        &src.ints,
        1e-12,
        &[("parent_fixture", "lih_sto3g"), ("note", "i6 round trip")],
    )
    .unwrap_or_else(|e| panic!("write_fixture_dir: {e}"));
    let back = load_fixture_dir::<f64>(&dir, &opts).unwrap_or_else(|e| panic!("reload: {e}"));
    let (a, b) = (&src.ints, &back.ints);
    assert_eq!((a.norb, a.nelec, a.ms2), (b.norb, b.nelec, b.ms2));
    assert_eq!(a.e0, b.e0, "e0 must round-trip exactly");
    assert_eq!(a.t.data, b.t.data, "t must round-trip bit for bit");
    let dv =
        a.v.data
            .iter()
            .zip(&b.v.data)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
    println!("i6: max |Δv| against the PySCF source = {dv:.3e} (its redundant orbit lines)");
    assert!(
        dv <= 1e-15,
        "v reload deviates from the source beyond roundoff: {dv:e}"
    );
    let dir2 = scratch_dir("i6b");
    write_fixture_dir(&dir2, "LIH_COPY.FCIDUMP", &back.ints, 1e-12, &[])
        .unwrap_or_else(|e| panic!("second write: {e}"));
    assert_eq!(
        std::fs::read_to_string(dir.join("LIH_COPY.FCIDUMP")).unwrap(),
        std::fs::read_to_string(dir2.join("LIH_COPY.FCIDUMP")).unwrap(),
        "write → read → write must reproduce the dump byte for byte"
    );
    let _ = std::fs::remove_dir_all(&dir2);
    let meta = back.meta.expect("meta.txt");
    assert_eq!(
        meta.keys.get("parent_fixture").map(String::as_str),
        Some("lih_sto3g")
    );
    assert_eq!(
        meta.keys.get("note").map(String::as_str),
        Some("i6 round trip")
    );
    assert!(
        (meta.e_core - b.e0).abs() == 0.0,
        "meta e_nuc must equal the dump's sentinel"
    );
    assert_ne!(
        std::fs::read_to_string(dir.join("LIH_COPY.FCIDUMP")).unwrap(),
        std::fs::read_to_string(testdata("lih_sto3g").join("LIH.FCIDUMP")).unwrap(),
        "the writer's byte layout differs from PySCF's (no write_tol, shortest round-trip floats)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn i7_write_fixture_dir_ships_a_rotated_basis_and_refuses_asymmetry() {
    use qntt::integrals::{write_fcidump, write_fixture_dir};
    use qntt::rotation::expm_antisym;
    let opts = IntegralsOptions::default();
    let src =
        load_fixture_dir::<f64>(&testdata("lih_sto3g"), &opts).unwrap_or_else(|e| panic!("{e}"));
    let d = src.ints.norb;
    let mut k = vec![0.0f64; d * d];
    for p in 0..d {
        for q in 0..p {
            let a = 0.1 * ((p * 7 + q * 3) % 5) as f64 - 0.2;
            k[p * d + q] = a;
            k[q * d + p] = -a;
        }
    }
    let u = expm_antisym(&k, d);
    let rot = qntt::rotate_integrals(&src.ints, &u);
    let sym_atol = 1e-12;
    let dir = scratch_dir("i7");
    write_fixture_dir(
        &dir,
        "LIH_ROT.FCIDUMP",
        &rot,
        sym_atol,
        &[("parent_fixture", "lih_sto3g")],
    )
    .unwrap_or_else(|e| panic!("write_fixture_dir: {e}"));
    let back = load_fixture_dir::<f64>(&dir, &opts).unwrap_or_else(|e| panic!("reload: {e}"));
    let maxdev = |x: &[f64], y: &[f64]| {
        x.iter()
            .zip(y)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max)
    };
    let dt = maxdev(&rot.t.data, &back.ints.t.data);
    let dv = maxdev(&rot.v.data, &back.ints.v.data);
    println!("i7: rotated-basis round trip max |Δt| = {dt:.3e}, max |Δv| = {dv:.3e}");
    assert!(
        dt <= sym_atol && dv <= sym_atol,
        "reload deviates beyond sym_atol: {dt:e} / {dv:e}"
    );
    assert!(
        maxdev(&rot.t.data, &src.ints.t.data) > 1e-2,
        "u did not rotate anything"
    );
    let (mut r1, mut r2) = (rot, back.ints);
    let (h1, sp, _) = r1
        .hamiltonian(None, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    let (h2, _, _) = r2
        .hamiltonian(None, &opts)
        .unwrap_or_else(|e| panic!("{e}"));
    let (e1, e2) = (hf_energy(&h1, sp, r1.e0), hf_energy(&h2, sp, r2.e0));
    assert!((e1 - e2).abs() < 1e-13, "operators differ: {e1} vs {e2}");
    let mut broken = qntt::rotate_integrals(&src.ints, &u);
    let x = broken.v.get(&[1, 0, 0, 0]);
    broken.v.set(&[1, 0, 0, 0], x + 1e-3);
    let err = write_fcidump(&dir.join("BROKEN.FCIDUMP"), &broken, sym_atol)
        .expect_err("an asymmetric v must be refused");
    assert!(
        format!("{err}").contains("not eightfold-symmetric"),
        "wrong refusal: {err}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
