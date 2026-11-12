// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::C64;
use qntt::core::tangent::{TtTangent, tangent_dim};
use qntt::solvers::gad::{GadOpts, gad, gad_c64};
use qntt::solvers::hess::{HessPart, PdVerdict, hessian_pd_probe_c64_witness, weingarten_operator};
use qntt::su2::qn::Su2Space;
use qntt::su2::tangent::Su2TtTangent;
use qntt::{QnSpace, Scalar, SparseHamiltonian, TtVector, rayleigh_quotient};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::{
    coeff_vector, converged_minimizer, deflate_radial_span, dense_h_in_sector, dense_sector_eigen,
    det_states, f64_full_rank_arm, full_rank_state, generic_complex_h, generic_truncated_tv,
    grad_exact_at, kick_along, overlap_modulus, radial_mode_at, singlet_indices,
    su2_converged_minimizer, su2_deflate_radial_span, su2_full_rank_state, su2_grad_exact_at,
    su2_kick_along, su2_radial_mode_at,
};

fn dense_arm_opts() -> GadOpts {
    GadOpts {
        gradtol: 1e-8,
        max_steps: 600,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    }
}

#[test]
fn gad_f64_full_rank_reaches_first_excited_dense() {
    let arm = f64_full_rank_arm();
    let opts = dense_arm_opts();
    let (xe, info) = gad(&arm.ham, &arm.x0, None, &opts);
    xe.check();

    assert!(
        info.converged,
        "GAD did not converge: gnorm {:.3e} after {} steps ({} rejected, {} applies)",
        info.gnorm, info.steps, info.rejected_steps, info.applies
    );
    assert!(info.index1_certified, "endpoint not certified index-1");
    assert_eq!(info.index_estimate, Some(1));
    assert!(info.gnorm <= opts.gradtol);
    assert!(info.applies > 0);

    let lam = rayleigh_quotient(&arm.ham, &xe);
    assert!(
        (lam - arm.evals[1]).abs() < 1e-7,
        "endpoint energy {lam} vs dense E1 {}",
        arm.evals[1]
    );
    assert!(
        (info.lambda - lam).abs() < 1e-9,
        "info.lambda disagrees with a fresh Rayleigh quotient"
    );
    assert!(
        (lam - arm.evals[0]).abs() > 1e-3,
        "endpoint fell back onto the GROUND state — the radial mode was not deflated somewhere"
    );

    let ov = overlap_modulus(&coeff_vector(&arm.dets, &xe), &arm.evecs[1]);
    assert!(ov >= 1.0 - 1e-6, "|<x_gad, psi1>| = {ov:.12}");

    let eig1 = info.eig1.expect("a certified endpoint must report eig1");
    let want = arm.evals[0] - arm.evals[1];
    assert!(
        eig1 < 0.0,
        "eig1 {eig1} must be negative at an index-1 saddle"
    );
    assert!(
        (eig1 - want).abs() <= 1e-3 * want.abs(),
        "eig1 {eig1} vs dense E0 - E1 = {want}"
    );
    assert!(
        info.v_overlap >= 1.0 - 1e-6,
        "v_overlap {} did not reach 1",
        info.v_overlap
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "<V, x> reached {:.3e} along the trajectory",
        info.radial_overlap_max
    );
}

#[test]
fn gad_c64_full_rank_reaches_first_excited_dense() {
    let space = QnSpace::new(4, 2, 1);
    let ham = generic_complex_h(space);
    let dets = det_states::<C64>(4, 2, 1);
    let ndet = dets.len();
    assert_eq!(ndet, 24, "sector enumeration changed");
    let (evals, evecs) = dense_sector_eigen(&dense_h_in_sector(&dets, &ham), ndet);
    assert!(
        evals[1] - evals[0] > 1e-3 && evals[2] - evals[1] > 1e-3,
        "fixture gate: E₁ degenerate or unseparated ({:?})",
        &evals[..3]
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6AD1);
    let seed_state: TtVector<C64> = full_rank_state(space, &mut rng);
    let xgs = converged_minimizer(&ham, &seed_state);
    let lam0 = rayleigh_quotient(&ham, &xgs).re_s();
    assert!(
        (lam0 - evals[0]).abs() < 1e-8,
        "C64 ground state {lam0} vs {}",
        evals[0]
    );

    let v0 = {
        let ctx = weingarten_operator(&ham, &xgs);
        let (bl, br) = ctx.bases();
        assert_eq!(tangent_dim(bl), ndet, "not at full representing rank");
        let diag = ctx.a_diagonal();
        let seed = TtTangent::project(bl, br, &diag.to_vector());
        let (verdict, w, _a) = hessian_pd_probe_c64_witness(&ctx, &seed, 250);
        assert!(
            matches!(verdict, PdVerdict::LikelyPd { .. }),
            "the C64 ground state must probe PD, got {verdict:?}"
        );
        let radial = radial_mode_at(bl, br, bl);
        let mut v = w;
        deflate_radial_span(&radial, &mut v);
        let n = v.norm();
        v.scale_mut(C64::from_f64(1.0 / n));
        v
    };

    let x0 = kick_along(&v0, 5e-2);
    let g0 = grad_exact_at(&ham, &x0).norm();
    assert!(
        g0 > 1e-4,
        "C64 kick left the state critical (‖grad‖={g0:.3e})"
    );

    let opts = dense_arm_opts();
    let (xe, info) = gad_c64(&ham, &x0, None, &opts);
    xe.check();

    assert!(
        info.converged,
        "C64 GAD did not converge: gnorm {:.3e} after {} steps",
        info.gnorm, info.steps
    );
    assert!(
        info.index1_certified,
        "C64 endpoint not certified index-1 (complex span)"
    );
    assert_eq!(info.index_estimate, Some(1));

    let lam = rayleigh_quotient(&ham, &xe).re_s();
    assert!(
        (lam - evals[1]).abs() < 1e-7,
        "C64 endpoint {lam} vs dense E₁ {}",
        evals[1]
    );
    assert!(
        (lam - evals[0]).abs() > 1e-3,
        "C64 endpoint fell back onto the ground state"
    );
    let ov = overlap_modulus(&coeff_vector(&dets, &xe), &evecs[1]);
    assert!(ov >= 1.0 - 1e-6, "C64 |⟨x_gad, ψ₁⟩| = {ov:.12}");
    let eig1 = info.eig1.expect("certified C64 endpoint must report eig1");
    let want = evals[0] - evals[1];
    assert!(
        (eig1 - want).abs() <= 1e-3 * want.abs(),
        "C64 eig1 {eig1} vs {want}"
    );
    assert!(
        info.v_overlap >= 1.0 - 1e-6,
        "C64 v_overlap {}",
        info.v_overlap
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "C64 ⟨V,x⟩ {:.3e}",
        info.radial_overlap_max
    );
}

#[test]
fn gad_su2_full_rank_reaches_first_excited_singlet_dense() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv::<f64>(d);

    let sp22 = QnSpace::new(d, 2, 2);
    let h22 = SparseHamiltonian::<f64>::new(sp22, &t, &v, None, f64::EPSILON);
    let dets22 = det_states::<f64>(d, 2, 2);
    let (ev22, evec22) = dense_sector_eigen(&dense_h_in_sector(&dets22, &h22), dets22.len());
    let sp31 = QnSpace::new(d, 3, 1);
    let h31 = SparseHamiltonian::<f64>::new(sp31, &t, &v, None, f64::EPSILON);
    let dets31 = det_states::<f64>(d, 3, 1);
    let (ev31, _) = dense_sector_eigen(&dense_h_in_sector(&dets31, &h31), dets31.len());
    assert_eq!(ev22.len(), 36);
    assert_eq!(ev31.len(), 16);
    let sidx = singlet_indices(&ev22, &ev31);
    assert_eq!(
        sidx.len(),
        20,
        "sector-degeneracy spin labelling failed (accidental degeneracy at this fixture?)"
    );
    let i0 = sidx[0];
    let i1 = sidx[1];
    let e0 = ev22[i0];
    let e1 = ev22[i1];
    assert!(
        ev22.iter()
            .enumerate()
            .all(|(j, e)| j == i1 || (e - e1).abs() > 1e-3),
        "the target singlet is degenerate inside the (2,2) sector — its abelian eigenvector is \
         not uniquely the singlet and the overlap assertion is ill posed"
    );
    assert!(
        ev31.iter().all(|p| (p - e1).abs() > 1e-6),
        "the target singlet coincides with a triplet level — spin labelling is not decisive here"
    );

    let space = Su2Space::new(d, 4, 0);
    let hsu =
        qntt::su2::hamiltonian::Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6AD2);
    let seed_state = su2_full_rank_state::<f64>(space, &mut rng);
    let xgs = su2_converged_minimizer(&hsu, &seed_state);
    let lam0 = qntt::su2::hamiltonian::rayleigh_quotient(&hsu, &xgs);
    assert!(
        (lam0 - e0).abs() < 1e-8,
        "SU(2) full-rank ground state {lam0} vs singlet E₀ {e0}"
    );

    let v0 = {
        let ctx = qntt::su2::hess::weingarten_operator(&hsu, &xgs);
        let (bl, br) = ctx.bases();
        assert_eq!(
            qntt::su2::weingarten::tangent_dim(bl),
            20,
            "not at full SU(2) representing rank: tangent_dim must be the Weyl–Paldus count \
             f(4,4,0) = 20 (radial included)"
        );
        let diag = ctx.a_diagonal();
        let seed = Su2TtTangent::project(bl, br, &diag.to_vector());
        let (verdict, w, _a) = qntt::su2::hess::hessian_pd_probe_witness_with(
            &ctx,
            HessPart::Rr,
            std::slice::from_ref(&seed),
            200,
            crate::util::FIXTURE_PROBE_SEED,
            &crate::util::fixture_probe(),
        );
        assert!(
            matches!(verdict, PdVerdict::LikelyPd { .. }),
            "the SU(2) ground state must probe PD, got {verdict:?}"
        );
        let radial = su2_radial_mode_at(bl, br, bl);
        let mut vv = w;
        su2_deflate_radial_span(&radial, &mut vv);
        let n = vv.norm();
        vv.scale_mut(1.0 / n);
        crate::util::su2_canonicalize_witness_sign(&mut vv, &seed);
        vv
    };

    let x0 = su2_kick_along(&v0, 5e-2);
    let g0 = su2_grad_exact_at(&hsu, &x0).norm();
    assert!(
        g0 > 1e-4,
        "SU(2) kick left the state critical (‖grad‖={g0:.3e})"
    );

    let opts = dense_arm_opts();
    let (xe, info) = qntt::su2::hess::gad(&hsu, &x0, None, &opts);
    xe.check();

    assert!(
        info.converged,
        "SU(2) GAD did not converge: gnorm {:.3e} after {} steps",
        info.gnorm, info.steps
    );
    assert!(
        info.index1_certified,
        "SU(2) endpoint not certified index-1"
    );
    assert_eq!(info.index_estimate, Some(1));

    let lam = qntt::su2::hamiltonian::rayleigh_quotient(&hsu, &xe);
    assert!(
        (lam - e1).abs() < 1e-7,
        "SU(2) endpoint {lam} vs first excited SINGLET {e1} (the abelian E₁ is {}, which may be a \
         triplet component)",
        ev22[1]
    );
    assert!(
        (lam - e0).abs() > 1e-3,
        "SU(2) endpoint fell back onto the ground state"
    );

    let xab = qntt::su2::expand(&xe, 0);
    let ov = overlap_modulus(&coeff_vector(&dets22, &xab), &evec22[i1]);
    assert!(ov >= 1.0 - 1e-6, "SU(2) |⟨expand(x_gad), ψ₁⟩| = {ov:.12}");

    let eig1 = info
        .eig1
        .expect("certified SU(2) endpoint must report eig1");
    let want = e0 - e1;
    assert!(
        (eig1 - want).abs() <= 1e-3 * want.abs(),
        "SU(2) eig1 {eig1} vs {want}"
    );
    assert!(
        info.v_overlap >= 1.0 - 1e-6,
        "SU(2) v_overlap {}",
        info.v_overlap
    );
    assert!(
        info.radial_overlap_max <= 1e-8,
        "SU(2) ⟨V,x⟩ {:.3e}",
        info.radial_overlap_max
    );
}
