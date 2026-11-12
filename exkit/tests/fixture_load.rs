// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::integrals::IntegralsOptions;
use qntt_exkit::repo::Repo;

fn repo() -> Repo {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("exkit sits under the checkout")
        .to_string_lossy()
        .into_owned();
    // SAFETY: single-threaded test setup, before any other thread reads the environment.
    unsafe { std::env::set_var("QNTT_REPO_ROOT", &root) };
    Repo::from_manifest_dir("unused-because-the-override-wins")
}

#[test]
fn t_load_lih_sto3g_exposes_every_field_the_old_tuples_carried() {
    let fx = repo().load("lih_sto3g");

    assert_eq!(fx.d(), 6);
    assert_eq!(fx.nelec(), 4);
    assert_eq!((fx.t().d, fx.t().rank), (6, 2));

    assert_eq!(fx.ms2(), 0);
    assert_eq!((fx.nup(), fx.ndn()), (2, 2));
    assert_eq!(fx.space().d, 6);

    assert!(fx.try_e_hf().is_some(), "lih_sto3g carries an e_hf");
    assert!(fx.e_core().is_finite());
}

#[test]
fn t_hf_determinant_is_normalized_and_lives_in_the_fixture_space() {
    let fx = repo().load("lih_sto3g");
    let x = fx.hf();
    assert_eq!(x.space.d, fx.d());
    assert!((x.norm() - 1.0).abs() < 1e-12, "‖HF‖ = {}", x.norm());
}

#[test]
fn t_a_verbatim_path_loads_the_same_fixture_as_the_bare_name() {
    let r = repo();
    let by_name = r.load("lih_sto3g");
    let by_path = r.load(&r.testdata().join("lih_sto3g").to_string_lossy());
    assert_eq!(by_name.d(), by_path.d());
    assert_eq!(by_name.dir, by_path.dir);
}

#[test]
fn t_try_load_reports_a_missing_fixture_instead_of_panicking() {
    let e = repo().try_load("no_such_fixture_anywhere");
    assert!(e.is_err(), "a missing fixture must be an Err, not a panic");
}

#[test]
fn t_load_with_honours_explicit_options() {
    let opts = IntegralsOptions::default();
    let fx = repo().load_with("lih_sto3g", &opts);
    assert_eq!(fx.d(), 6);
}
