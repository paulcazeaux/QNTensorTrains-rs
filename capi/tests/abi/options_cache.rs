// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn option_defaults_match_the_core_crate() {
    let mut als = QnttAlsOptions::defaults();
    assert_eq!(unsafe { qntt_als_options_default(&mut als) }, QNTT_OK);
    let core = qntt::solvers::als::AlsOptions::default();
    assert_eq!((als.eps, als.max_iter), (core.eps, core.max_iter));

    let mut mals = QnttMalsOptions::defaults();
    assert_eq!(unsafe { qntt_mals_options_default(&mut mals) }, QNTT_OK);
    let core = qntt::solvers::mals::MalsOptions::default();
    assert_eq!(
        (mals.eps, mals.max_iter, mals.seed),
        (core.eps, core.max_iter, core.seed)
    );
    assert_eq!(
        mals.noise_type, 1,
        "the recorded MALS noise default is Perturbative (wire code 1)"
    );
    assert_eq!(core.noise_type, NoiseType::Perturbative);
    assert!(
        matches!(
            qntt_capi::enums::noise_type(mals.noise_type),
            Ok(nt) if nt == core.noise_type
        ),
        "the ABI noise code must round-trip back to the core default"
    );

    let mut tdvp = QnttTdvpOptions::defaults();
    assert_eq!(unsafe { qntt_tdvp_options_default(&mut tdvp) }, QNTT_OK);
    let core = qntt::solvers::tdvp::TdvpOptions::default();
    assert_eq!(
        (tdvp.dtau, tdvp.eps, tdvp.trunc),
        (core.dtau, core.eps, core.trunc)
    );

    let mut lan = QnttLanczosOptions::defaults();
    assert_eq!(unsafe { qntt_lanczos_options_default(&mut lan) }, QNTT_OK);
    let core = qntt::solvers::lanczos::LanczosOptions::default();
    assert_eq!((lan.tol, lan.max_iter), (core.tol, core.max_iter));

    let mut sym = QnttSymplecticLowestOpts::defaults();
    assert_eq!(
        unsafe { qntt_symplectic_options_default(&mut sym) },
        QNTT_OK
    );
    let core = qntt::solvers::hess::SymplecticDavidsonOpts::default();
    assert_eq!((sym.max_iter, sym.tol), (core.max_basis, core.tol));
}

fn cache_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("qntt_abi_{}_{name}.tt", std::process::id()))
}

fn c_path(p: &std::path::Path) -> CString {
    CString::new(p.to_str().expect("utf-8 temp path")).expect("no interior NUL")
}

#[test]
fn state_cache_round_trips_on_both_layers() {
    let space = QnSpace::new(6, 3, 3);
    let flat = flat_ranks(&space, &uniform_ranks(space, 3));
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                20260729,
                &mut x,
            )
        },
        QNTT_OK
    );

    let path = cache_path("abelian_f64");
    let cpath = c_path(&path);
    assert_eq!(
        unsafe { qntt_f64_vec_save(x, cpath.as_ptr()) },
        QNTT_OK,
        "vec_save failed: {}",
        last_error()
    );
    let mut y: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_load(cpath.as_ptr(), &mut y) },
        QNTT_OK,
        "vec_load failed: {}",
        last_error()
    );

    let (mut d, mut a, mut b) = (0usize, 0usize, 0usize);
    assert_eq!(
        unsafe { qntt_f64_vec_space(y, &mut d, &mut a, &mut b) },
        QNTT_OK
    );
    assert_eq!((d, a, b), (space.d, space.nup, space.ndn));

    let mut got = vec![0usize; flat.len()];
    let mut len = 0usize;
    assert_eq!(
        unsafe { qntt_f64_vec_ranks(y, got.as_mut_ptr(), got.len(), &mut len) },
        QNTT_OK
    );
    assert_eq!(len, flat.len());
    assert_eq!(got, flat, "rank grids must survive the cache");

    let mut diff: *mut _ = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_add(x, y, -1, &mut diff) }, QNTT_OK);
    let (mut nd, mut nx) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_f64_vec_norm(diff, &mut nd) }, QNTT_OK);
    assert_eq!(unsafe { qntt_f64_vec_norm(x, &mut nx) }, QNTT_OK);
    assert!(
        nd <= 1e-13 * nx,
        "‖x − load(save(x))‖ = {nd} against ‖x‖ = {nx}"
    );

    let mut wrong: *mut _ = ptr::null_mut();
    let code = unsafe { qntt_su2_f64_vec_load(cpath.as_ptr(), &mut wrong) };
    assert_eq!(code, qntt_capi::err::QNTT_IO, "got: {}", last_error());
    assert!(last_error().contains("SU(2)"), "message: {}", last_error());
    let mut wrong2: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_c64_vec_load(cpath.as_ptr(), &mut wrong2) },
        qntt_capi::err::QNTT_IO
    );

    for hv in [x, y, diff] {
        unsafe { qntt_f64_vec_free(hv) };
    }
    let _ = std::fs::remove_file(&path);

    let (dd, nn, ts) = (4usize, 4usize, 0usize);
    let sflat = su2_capped_ranks(dd, nn, ts, 4);
    let mut sx: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_c64_vec_randn(dd, nn, ts, sflat.as_ptr(), sflat.len(), 7, &mut sx) },
        QNTT_OK
    );
    let spath = cache_path("su2_c64");
    let scpath = c_path(&spath);
    assert_eq!(
        unsafe { qntt_su2_c64_vec_save(sx, scpath.as_ptr()) },
        QNTT_OK
    );
    let mut sy: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_c64_vec_load(scpath.as_ptr(), &mut sy) },
        QNTT_OK,
        "su2 c64 load failed: {}",
        last_error()
    );
    let mut sdiff: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_su2_c64_vec_add(sx, sy, -1, &mut sdiff) },
        QNTT_OK
    );
    let (mut snd, mut snx) = (0.0f64, 0.0f64);
    assert_eq!(unsafe { qntt_su2_c64_vec_norm(sdiff, &mut snd) }, QNTT_OK);
    assert_eq!(unsafe { qntt_su2_c64_vec_norm(sx, &mut snx) }, QNTT_OK);
    assert!(snx > 0.0, "vacuous fixture");
    assert!(snd <= 1e-13 * snx, "SU(2)/C64 round trip: {snd} vs {snx}");

    for hv in [sx, sy, sdiff] {
        unsafe { qntt_su2_c64_vec_free(hv) };
    }
    let _ = std::fs::remove_file(&spath);
}

#[test]
fn state_cache_rejects_bad_paths_without_dereferencing() {
    let mut out: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_load(ptr::null(), &mut out) },
        qntt_capi::err::QNTT_NULL_ARG
    );
    let missing = c_path(&cache_path("definitely_absent"));
    assert_eq!(
        unsafe { qntt_f64_vec_load(missing.as_ptr(), &mut out) },
        qntt_capi::err::QNTT_IO
    );

    let junk = cache_path("junk");
    std::fs::write(&junk, b"this is not a qntt state file, not even close").unwrap();
    let cjunk = c_path(&junk);
    assert_eq!(
        unsafe { qntt_f64_vec_load(cjunk.as_ptr(), &mut out) },
        qntt_capi::err::QNTT_IO
    );
    assert!(last_error().contains("magic"), "message: {}", last_error());
    let _ = std::fs::remove_file(&junk);

    let space = QnSpace::new(4, 2, 2);
    let flat = flat_ranks(&space, &uniform_ranks(space, 2));
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn(
                space.d,
                space.nup,
                space.ndn,
                flat.as_ptr(),
                flat.len(),
                1,
                &mut x,
            )
        },
        QNTT_OK
    );
    assert_eq!(
        unsafe { qntt_f64_vec_save(x, ptr::null()) },
        qntt_capi::err::QNTT_NULL_ARG
    );
    assert_eq!(
        unsafe { qntt_f64_vec_save(ptr::null(), c_path(&cache_path("nope")).as_ptr()) },
        qntt_capi::err::QNTT_NULL_ARG
    );
    unsafe { qntt_f64_vec_free(x) };
}
