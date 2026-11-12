// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::coeff::CoeffTensor;
use qntt::core::weingarten::FloorPolicy;
use qntt::solvers::hess::{RpaKernel, RpaMetric, SymplecticDavidsonOpts, SymplecticSpectrum};
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use qntt::su2::hess::{
    hessian_grams, normal_residual, symplectic_eigenvalues, symplectic_lowest_signed_davidson,
    weingarten_operator,
};
use qntt::su2::qn::{Su2Space, max_ranks};
use qntt::su2::riemannian::{normalize_mut, tangent_bases};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::{bond_gauges, tangent_basis};
use qntt::{Irrep, Scalar};

use crate::spg5::util::*;

#[test]
fn spg5_y5_hess_apply_matches_fd_of_gradient_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (4usize, 2usize);
    let target = Irrep::IDENTITY;
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng_h = seeded(0xC5_0501);
    let (t, _) = c2v_coeffs(&pg, d, &mut rng_h);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);
    let v = CoeffTensor::<f64>::zeros(d, 4);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let (ham, x, base_l, base_r) = su2_minimizer(ham, space, 0xC5_0502, 1, 2);
    let ctx = weingarten_operator(&ham, &x);

    let mut rng = seeded(0xC5_0503);
    for trial in 0..3 {
        let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
        let n = xi.norm();
        assert!(n > 1e-8, "trial {trial}: vacuous tangent direction");
        let xi = xi.scale(1.0 / n);
        let fd = hess_fd_central(&ham, &x, &xi, 1e-5);
        let hess = ctx.hess_apply(&xi);
        let rel = fd.sub(&hess).norm() / fd.norm().max(1e-30);
        assert!(
            rel < 1e-5,
            "trial {trial}: ‖FD − hess_apply‖/‖FD‖ = {rel:.3e}"
        );
    }

    let lam = rayleigh_quotient(&ham, &x).re_s();
    let rperp = normal_residual(&ham, &base_l, &base_r, &x, lam);
    let proj = Su2TtTangent::project(&base_l, &base_r, &rperp);
    assert!(
        proj.norm() < 1e-9 * rperp.norm().max(1.0),
        "P(Z) not ~0: ‖P(Z)‖={} ‖Z‖={}",
        proj.norm(),
        rperp.norm()
    );

    assert!(pg.order() > 1);
}

#[test]
fn spg5_y6_weingarten_and_hessian_grams_are_symmetric_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (4usize, 2usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: no nonidentity reachable target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng_h = seeded(0xC5_0601);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng_h);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut x = build_pg_state::<f64>(d, n, two_s, pg, target, 1, 0xC5_0602);
    normalize_mut(&mut x);
    let (base_l, base_r) = tangent_bases(&x);
    let ctx = weingarten_operator(&ham, &x);

    let basis = tangent_ortho_basis(&base_l, &base_r, space, 0xC5_0603, 40);
    let nb = basis.len();
    assert!(nb >= 2, "need a nontrivial tangent space, got {nb}");
    for i in 0..nb {
        for j in 0..nb {
            let g = basis[i].dot(&basis[j]);
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (g - want).abs() < 1e-9,
                "basis not orthonormal at ({i},{j}): {g}"
            );
        }
    }

    let (a, b) = hessian_grams(&ctx, &basis);
    let amax = a.iter().fold(0.0f64, |acc, &val| acc.max(val.abs()));
    let bmax = b.iter().fold(0.0f64, |acc, &val| acc.max(val.abs()));
    assert!(amax > 1e-8, "A vacuous");
    assert!(bmax > 1e-8, "B vacuous");
    for i in 0..nb {
        for j in 0..nb {
            assert!(
                (a[i * nb + j] - a[j * nb + i]).abs() < 1e-9 * amax.max(1.0),
                "A not symmetric at ({i},{j})"
            );
            assert!(
                (b[i * nb + j] - b[j * nb + i]).abs() < 1e-9 * bmax.max(1.0),
                "B not symmetric at ({i},{j})"
            );
        }
    }

    let mut rng = seeded(0xC5_0604);
    let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
    let mut recon = xi.scale(0.0);
    for u in &basis {
        let c = u.dot(&xi);
        recon = recon.add(&u.scale(c));
    }
    let err = xi.sub(&recon).norm();
    assert!(
        err < 1e-9 * xi.norm().max(1.0),
        "basis not complete: err={err}"
    );

    let gauges = bond_gauges(&base_l, &base_r, FloorPolicy::default_hard());
    assert_eq!(gauges.len(), d - 1);
    for gauge in &gauges {
        for (_, sig) in &gauge.sigma {
            for w in sig.windows(2) {
                assert!(w[0] + 1e-12 >= w[1], "sigma not descending: {sig:?}");
            }
        }
        assert!(gauge.dropped_mass.is_finite());
    }

    assert!(pg.order() > 1);
}

#[test]
fn spg5_y7_symplectic_iterative_matches_dense_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (4usize, 2usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: no nonidentity reachable target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng_h = seeded(0xC5_0701);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng_h);
    assert_c2v_coeffs_nonvacuous(&pg, d, &t);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let (ham, x, base_l, base_r) = su2_minimizer(ham, space, 0xC5_0702, 1, 1);
    let ctx = weingarten_operator(&ham, &x);
    let basis = tangent_basis(&base_l, &base_r, true);
    let SymplecticSpectrum::Stable { omega: dense } = symplectic_eigenvalues(&ctx, &basis) else {
        panic!("dense SU(2) oracle unstable at |G|>1");
    };
    let dense = dedup_sorted(&dense, 1e-9);
    assert!(!dense.is_empty(), "vacuous dense spectrum");

    let mut rng = seeded(0xC5_0703);
    let _: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let k = 2usize.min(dense.len());

    let v0s: Vec<Su2TtTangent<f64>> = (0..k + 2)
        .map(|_| {
            let dir: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
            Su2TtTangent::project(&base_l, &base_r, &dir)
        })
        .collect();
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let opts = SymplecticDavidsonOpts {
            kernel,
            ..SymplecticDavidsonOpts::default()
        };
        let (so, _) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        assert!(so.converged, "{kernel:?}: not converged at |G|>1");
        let omega_d = so
            .omega()
            .expect("a converged stable SU(2) solve at |G|>1 publishes its ω");
        assert_eq!(omega_d.len(), k, "{kernel:?}");
        for (j, (wi, wd)) in omega_d.iter().zip(dense.iter()).enumerate() {
            assert!(
                (wi - wd).abs() < 1e-6 * wd.max(1.0),
                "{kernel:?}: davidson ω[{j}]: {wi} vs dense {wd}"
            );
        }
    }

    assert!(pg.order() > 1);
}

#[test]
#[allow(clippy::needless_range_loop)]
fn spg5_y8_max_rank_accessors_count_only_reachable_sectors() {
    let d = 6;
    let pg = c2v_table(d);
    let rmax = 3usize;
    let (n, two_s) = (3usize, 1usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d).unwrap_or(Irrep::IDENTITY);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, rmax, 0xC5_0801);

    let space = x.space;
    let caps = max_ranks(&space);
    let mut oracle_max_sector = 0usize;
    let mut oracle_max_total = 0usize;
    let mut saw_cap_zero_admissible = false;
    for b in 0..=d {
        let mut bond_total = 0usize;
        for s in space.bond_sectors(b) {
            if caps[b][s] == 0 {
                saw_cap_zero_admissible = true;
                assert_eq!(
                    x.ranks[b][s], 0,
                    "rank positive at cap-0 (unreachable) sector {s:?} at bond {b}"
                );
                continue;
            }
            oracle_max_sector = oracle_max_sector.max(x.ranks[b][s]);
            bond_total += x.ranks[b][s];
        }
        oracle_max_total = oracle_max_total.max(bond_total);
    }
    assert!(
        saw_cap_zero_admissible,
        "vacuous fixture: no cap-0 admissible sector found (need |G|>1 boundary evidence)"
    );
    assert_eq!(x.max_sector_rank(), oracle_max_sector);
    assert_eq!(x.max_total_bond_rank(), oracle_max_total);
    assert!(pg.order() > 1);
}
