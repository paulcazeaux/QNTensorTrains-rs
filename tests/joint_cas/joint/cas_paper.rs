// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashSet;
use std::path::Path;

use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use qntt::integrals::{
    Integrals, IntegralsOptions, load_fixture_dir, load_property_matching, read_meta,
};
use qntt::qn::QnSpace;
use qntt::solvers::joint::cas::CasPartition;

struct CasFixture {
    ints: Integrals<f64>,
    space: QnSpace,
    part: CasPartition,
    dirname: &'static str,
}

fn load_cas_fixture(dirname: &'static str) -> CasFixture {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(dirname);
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{dirname}: {e:?}"));
    let meta = load.meta.as_ref().unwrap();
    let get = |k: &str| -> usize {
        meta.keys
            .get(k)
            .unwrap_or_else(|| panic!("{dirname}: meta missing `{k}`"))
            .parse()
            .unwrap()
    };
    let part = CasPartition {
        n_core: get("n_core"),
        n_active: get("n_active"),
        n_virt: get("n_virt"),
    };
    let _casscf_gs: f64 = meta.keys.get("casscf_gs").unwrap().parse().unwrap();
    let ints = load.ints;
    let nup = (ints.nelec + ints.ms2.unsigned_abs() as usize) / 2;
    let ndn = ints.nelec - nup;
    let space = QnSpace::new(ints.norb, nup, ndn);
    CasFixture {
        ints,
        space,
        part,
        dirname,
    }
}

#[test]
fn tjcp_p6_dense_arbiter_endpoint_classification() {
    use crate::joint::cas_dense::{DenseCasscfOpts, dense_casscf};
    let fx = load_cas_fixture("h2o_sto3g_cas");
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(fx.dirname);
    let meta = read_meta(&dir.join("meta.txt")).unwrap();
    let pz = load_property_matching::<f64>(&dir.join("dipole_z.prop"), fx.space.d, &meta).unwrap();
    let opts = |g_tol: f64, max_iter: usize| DenseCasscfOpts {
        g_tol,
        max_iter,
        trust0: 0.5,
        shift_floor: 1e-8,
        verbose: false,
    };

    let ordinary = dense_casscf(&fx.ints, &fx.part, &opts(1e-9, 200));
    let forced = dense_casscf(&fx.ints, &fx.part, &opts(1e-30, 200));
    eprintln!(
        "tjcp_p6 (i): g_tol 1e-30 ⇒ converged = {}, stalled_at_floor = {}, ‖g‖ = {:.3e}, \
         ΔE vs the 1e-9 run = {:.3e}",
        forced.converged,
        forced.stalled_at_floor,
        forced.g_norm,
        (forced.e_total - ordinary.e_total).abs()
    );
    assert!(
        forced.converged && forced.stalled_at_floor,
        "an unsatisfiable g_tol must exit through the √ε floor branch, not as a failure \
         (converged = {}, stalled = {}, ‖g‖ = {:.3e})",
        forced.converged,
        forced.stalled_at_floor,
        forced.g_norm
    );
    assert!(
        forced.g_norm < 1e-5,
        "the floor endpoint's gradient must be IN the √ε band, not merely somewhere: {:.3e}",
        forced.g_norm
    );
    assert!(
        (forced.e_total - ordinary.e_total).abs() < 1e-11,
        "a floor endpoint must carry the SAME energy as a g_tol endpoint: {:.12} vs {:.12}",
        forced.e_total,
        ordinary.e_total
    );

    let cut = dense_casscf(&fx.ints, &fx.part, &opts(1e-9, 0));
    eprintln!(
        "tjcp_p6 (ii): max_iter = 0 ⇒ converged = {}, stalled_at_floor = {}, ‖g‖ = {:.3e}",
        cut.converged, cut.stalled_at_floor, cut.g_norm
    );
    assert!(
        !cut.converged && !cut.stalled_at_floor,
        "a max_iter cut-off must stay an honest failure and must NOT be labelled a floor endpoint"
    );

    let delta = 2e-5f64;
    let e_at = |sfield: f64| -> f64 {
        let ints_s = qntt::solvers::joint::with_field(&fx.ints, &pz.m, -sfield);
        let o = dense_casscf(&ints_s, &fx.part, &opts(1e-9, 200));
        assert!(
            o.converged,
            "field {sfield:e}: ‖g‖ = {:.3e} after {} iters, stalled_at_floor = {}",
            o.g_norm, o.iters, o.stalled_at_floor
        );
        o.e_total
    };
    let alpha0 = -(e_at(delta) - 2.0 * e_at(0.0) + e_at(-delta)) / (delta * delta);
    eprintln!(
        "tjcp_p6 (iii-a): zero-field α_zz (3-point, δ = {delta:.0e}) = {alpha0:.6} vs p3's 5-point \
         field-FD 2.366533 (dev {:+.3e}, stencil noise floor η/δ² ≈ 2.5e-4)",
        alpha0 - 2.366533
    );
    assert!(
        (alpha0 - 2.366533).abs() < 1e-3,
        "the ladder's own stencil must reproduce the certified α_zz at zero field: {alpha0:.6}"
    );

    let energies: Vec<(f64, f64, bool)> = (0..21)
        .map(|k| {
            let sfield = 3.8e-3 + delta * (k as f64);
            let ints_s = qntt::solvers::joint::with_field(&fx.ints, &pz.m, -sfield);
            let o = dense_casscf(&ints_s, &fx.part, &opts(1e-9, 200));
            assert!(
                o.converged,
                "ladder field {sfield:e}: ‖g‖ = {:.3e} after {} iters, stalled_at_floor = {}",
                o.g_norm, o.iters, o.stalled_at_floor
            );
            (sfield, o.e_total, o.stalled_at_floor)
        })
        .collect();
    let stalls = energies.iter().filter(|e| e.2).count();
    let alphas: Vec<f64> = energies
        .windows(3)
        .map(|w| -(w[2].1 - 2.0 * w[1].1 + w[0].1) / (delta * delta))
        .collect();
    let (amin, amax) = alphas
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), &a| (lo.min(a), hi.max(a)));
    let spread = (amax - amin) / amax.abs().max(1.0);
    eprintln!(
        "tjcp_p6 (iii-b): 21 fields at +3.8e-3…+4.2e-3, {stalls} at the √ε floor; α_zz(f) from \
         second differences ∈ [{amin:.6}, {amax:.6}] (spread {spread:.3e}); finite-field drift vs \
         α(0) = {:+.3e}, the hyperpolarizability term",
        0.5 * (amin + amax) - alpha0
    );
    assert!(
        spread < 1e-2,
        "the ladder's α_zz estimates must agree — a stalled endpoint off the smooth energy curve \
         shows up here: spread {spread:.3e} over [{amin:.6}, {amax:.6}]"
    );
    assert!(
        (0.5 * (amin + amax) - alpha0).abs() < 5e-2,
        "the ladder's α_zz(f) must stay near the zero-field {alpha0:.6}: got [{amin:.6}, {amax:.6}]"
    );
}

#[allow(dead_code)]
fn _seed_hygiene() -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(0)
}

#[allow(dead_code)]
fn _distinct(v: &[usize]) -> usize {
    v.iter().copied().collect::<HashSet<_>>().len()
}
