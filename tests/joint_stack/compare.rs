// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::tangent_basis;
use qntt::rotation::{PairMask, orbital_pairs};
use qntt::solvers::hess::{
    ConeOpts, SymplecticSpectrum, cone_response, hessian_grams, symplectic_from_grams,
    weingarten_operator,
};
use qntt::solvers::joint::{JointOpts, joint_context};
use qntt::vector::symmetry_defect;
use qntt::{Irrep, PointGroup, QnSpace};

use crate::optimize::d2_fixture;
use crate::util::*;

#[test]
fn tjc_fixed_reference_contains_tangent() {
    let fx = d2_fixture();
    let out = &fx.out;
    assert!(
        out.info.converged,
        "D2 fixture must converge for this pin to be legal: {:?}",
        out.info.refused_reason
    );

    let jctx = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("joint_ground_state's own output must be a certified joint-critical point");
    let wctx = weingarten_operator(&out.h, &out.x);
    let (bl, br) = jctx.bases();
    let tb = tangent_basis(bl, br, true);

    let n_eff = jctx.n_eff();
    assert!(
        n_eff >= 1,
        "D2 fixture must be genuinely truncated for this pin to be non-vacuous: n_eff={n_eff}"
    );

    let (basis, report) = jctx.joint_basis();
    let m = report.n_tt;
    let n = basis.len();
    assert_eq!(
        m,
        tb.len(),
        "n_tt must equal the wrapped tangent basis size"
    );
    assert!(
        report.n_orb_kept >= 1,
        "anti-vacuity: some orbital seed must survive MGS"
    );
    assert_eq!(
        n,
        m + report.n_orb_kept,
        "joint basis size must be exactly tangent + kept orbital seeds (modes are ADDED)"
    );

    let (a_tt, b_tt) = hessian_grams(&wctx, &tb);
    let (a_joint, b_joint) = jctx.joint_hessian_grams(&basis);
    let mut worst_a = 0.0f64;
    let mut worst_b = 0.0f64;
    for i in 0..m {
        for j in 0..m {
            let at = a_tt[i * m + j];
            let bt = b_tt[i * m + j];
            let aj = a_joint[i * n + j];
            let bj = b_joint[i * n + j];
            worst_a = worst_a.max((aj - at).abs() / at.abs().max(1.0));
            worst_b = worst_b.max((bj - bt).abs() / bt.abs().max(1.0));
        }
    }
    assert!(
        worst_a <= 1e-12,
        "leading A block must equal the TT-only A Gram to ~roundoff: worst rel dev {worst_a:e}"
    );
    assert!(
        worst_b <= 1e-12,
        "leading B block must equal the TT-only B Gram to ~roundoff: worst rel dev {worst_b:e}"
    );

    let bmax_tt = b_tt.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
    assert!(
        bmax_tt > 1e-8,
        "fixture must be genuinely truncated (B_tt vacuous otherwise)"
    );
    let SymplecticSpectrum::Stable { omega: omega_tt } =
        symplectic_from_grams(&a_tt, &b_tt, m, 1e-10)
    else {
        panic!("expected a Stable tangent-only spectrum at a converged D2 minimizer");
    };
    let spectrum_joint = jctx.joint_symplectic_eigenvalues(&basis);
    let SymplecticSpectrum::Stable { omega: omega_joint } = spectrum_joint else {
        panic!(
            "expected a Stable joint spectrum at a converged D2 minimizer (orth_error={:e}, \
             sigma_min={:e})",
            report.orth_error,
            jctx.sigma_min()
        );
    };
    assert_eq!(omega_tt.len(), m);
    assert_eq!(omega_joint.len(), n);

    let mut moved = false;
    for k in 0..m {
        let wt = omega_tt[k];
        let wj = omega_joint[k];
        let tol = 1e-9 * wt.max(1.0);
        assert!(
            wj <= wt + tol,
            "ω_joint[{k}]={wj} must be a Ritz value FROM ABOVE relative to ω_tt[{k}]={wt} \
             (subspace enlargement can only tighten a min-max upper bound); orth_error={:e}",
            report.orth_error
        );
        if wt - wj > 1e-6 * wt.max(1.0) {
            moved = true;
        }
    }
    assert!(
        moved,
        "anti-vacuity: at least one ω_joint[k] (k<{m}) must move measurably below ω_tt[k] — \
         otherwise the added orbital directions did nothing observable. omega_tt={omega_tt:?} \
         omega_joint (first {m})={:?}",
        &omega_joint[..m]
    );
}

#[test]
fn tjc_irrep_wall_gamma0_confined() {
    let d = 5usize;
    let space = QnSpace::new(d, 2, 1);
    let pg = PointGroup::new(
        2,
        &[
            Irrep::new(0),
            Irrep::new(1),
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
        ],
    );
    let (ham, x, t, v) = weingarten_minimizer_pg(space, pg, 0xB0C0, 2, 2);
    let (gamma_x, defects) = symmetry_defect(&x, pg);
    let worst_defect = defects.iter().cloned().fold(0.0f64, f64::max);
    assert_eq!(gamma_x, Irrep::IDENTITY, "endpoint must stay Gamma0-pure");
    assert!(worst_defect <= 1e-8, "endpoint defect: {worst_defect:e}");

    let dets = dets_for(d, space.nup, space.ndn);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let (evals, _evecs) = dense_eigen(&hd);
    let gap1 = evals[1] - evals[0];
    let base = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 0,
            ..Default::default()
        },
    );
    let e0 = (base.gaps[0] - gap1).abs();
    let aug = cone_response(
        &ham,
        &x,
        &ConeOpts {
            targets: 2,
            max_added: 2,
            ..Default::default()
        },
    );
    let et = (aug.gaps[0] - gap1).abs();

    assert!(
        e0 > 1e-3,
        "precondition: tangent-only CI must miss the first gap materially: {e0:e}"
    );
    assert!(
        et < e0 / 5.0,
        "cone-CI must close the first gap by >=5x at this fixture: {e0:e} -> {et:e}"
    );
    let jopts = JointOpts {
        pairs: Some(PairMask {
            keep: orbital_pairs(pg, d),
        }),
        ..JointOpts::default()
    };
    let jctx = joint_context(&ham, &x, &jopts).expect("pg fixture must be TT-critical");
    let n_eff = jctx.n_eff();
    let n_orb = jctx.orbital_directions().len();
    assert_eq!(n_orb, 2, "two same-irrep pairs at this group");
    let z = jctx.normal_residual();
    let z_norm2 = qntt::dot(z, z);
    eprintln!(
        "tjc_irrep_wall_gamma0_confined: gap1={gap1} tangent-CI err={e0:e} cone-CI(k=2) \
         err={et:e} ||z||^2={z_norm2:e} n_eff={n_eff} n_orb={n_orb} \
         worst_defect={worst_defect:e}"
    );
    assert!(
        z_norm2 <= 1e-12,
        "the Gamma0-confined problem must be exactly solved at this fixture (measured ~1e-17): \
         ||z||^2={z_norm2:e}"
    );
    assert_eq!(
        n_eff, 0,
        "every Gamma0 orbital direction must be tangential here — the wall's second half"
    );
}
