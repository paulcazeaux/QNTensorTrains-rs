// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::OnceLock;

use qntt::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::integrals::{
    Integrals, IntegralsOptions, PropertyOperator, load_fixture_dir, load_property_matching,
    read_meta,
};
use qntt::qn::QnSpace;
use qntt::rotation::rotate_integrals;
use qntt::solvers::hess::SymplecticSpectrum;
use qntt::solvers::joint::cas::{CasPartition, cas_pairs, cas_reference_state};
use qntt::solvers::joint::{
    JointOpts, JointScfOpts, JointScfOut, JointSolver, PropRoute, StaticResponseOpts, ZRoute,
    joint_ground_state,
};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::joint::cas::su2_cas_reference_state;
use qntt::su2::joint::{
    Su2JointScfOut, su2_joint_context, su2_joint_ground_state, su2_polarizability_static,
};
use qntt::su2::qn::Su2Space;

use crate::su2_cas::util::H2O_PART;

pub(crate) struct Endpoint {
    ints: Integrals<f64>,
    d: usize,
    part: CasPartition,
    su2: Su2JointScfOut,
    ab: JointScfOut,
    casscf_gs: f64,
    dir: std::path::PathBuf,
}

fn scf_opts(part: &CasPartition, d: usize) -> JointScfOpts {
    JointScfOpts {
        solver: JointSolver::DenseSchur,
        pairs: Some(cas_pairs(part, d)),
        orb_max_iter: 40,
        orb_grad: 1e-6,
        orb_grad_full: Some(1e-6),
        tt_grad_tol: 1e-6,
        orb_floor: 1e-8,
        tt_preserve_grid: true,
        ..JointScfOpts::default()
    }
}

pub(crate) fn endpoint() -> &'static Endpoint {
    static E: OnceLock<Endpoint> = OnceLock::new();
    E.get_or_init(|| build_endpoint("h2o_sto3g_cas", H2O_PART))
}

fn build_endpoint(dirname: &str, part: CasPartition) -> Endpoint {
    {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(dirname);
        let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
            .unwrap_or_else(|e| panic!("{dirname}: {e:?}"));
        let casscf_gs: f64 = load
            .meta
            .as_ref()
            .and_then(|m| m.keys.get("casscf_gs"))
            .expect("a _cas fixture meta carries casscf_gs")
            .parse()
            .unwrap();
        let ints = load.ints;
        let d = ints.norb;

        let space = Su2Space::new(d, ints.nelec, 0);
        let ham = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
        let (_l0, x0) = su2_cas_reference_state(&ham, &space, &part, 0, 0xCA5D);
        let su2 = su2_joint_ground_state(&ints, space, &x0, &scf_opts(&part, d));
        assert!(
            su2.info.converged,
            "{dirname}: su2 pinned SCF did not converge — no ω may be published. iters = {}, \
             ‖g_kept‖ = {:.3e}, refusal = {:?}",
            su2.info.iters,
            su2.info.g_orb_kept.last().copied().unwrap_or(f64::NAN),
            su2.info.refused_reason
        );

        let nup = (ints.nelec + ints.ms2.unsigned_abs() as usize) / 2;
        let ab_space = QnSpace::new(d, nup, ints.nelec - nup);
        let ab_ham = SparseHamiltonian::new(ab_space, &ints.t, &ints.v, None, f64::EPSILON);
        let (_abl0, abx0) = cas_reference_state(&ab_ham, &ab_space, &part, 0, 0xCA5D);
        let ab = joint_ground_state(&ints, ab_space, &abx0, &scf_opts(&part, d));
        assert!(
            ab.info.converged,
            "{dirname}: abelian pinned SCF did not converge: iters = {}, ‖g_kept‖ = {:.3e}, \
             refusal = {:?}",
            ab.info.iters,
            ab.info.g_orb_kept.last().copied().unwrap_or(f64::NAN),
            ab.info.refused_reason
        );

        Endpoint {
            ints,
            d,
            part,
            su2,
            ab,
            casscf_gs,
            dir,
        }
    }
}

impl Endpoint {
    fn joint_opts(&self) -> JointOpts {
        JointOpts {
            tt_grad_tol: 1e-6,
            orb_floor: 1e-8,
            pairs: Some(cas_pairs(&self.part, self.d)),
            z_route: ZRoute::Explicit,
            b_general: false,
        }
    }

    fn props(&self, u: &[f64]) -> Vec<PropertyOperator<f64>> {
        let meta = read_meta(&self.dir.join("meta.txt")).expect("meta");
        ["x", "y", "z"]
            .iter()
            .map(|c| {
                let p = load_property_matching::<f64>(
                    &self.dir.join(format!("dipole_{c}.prop")),
                    self.d,
                    &meta,
                )
                .unwrap_or_else(|e| panic!("dipole_{c}: {e}"));
                let wrapped = Integrals::<f64> {
                    t: p.m.clone(),
                    v: CoeffTensor::<f64>::zeros(self.d, 4),
                    e0: 0.0,
                    norb: self.d,
                    nelec: 0,
                    ms2: 0,
                    orbsym: Vec::new(),
                    isym: None,
                };
                PropertyOperator::new(p.label.clone(), rotate_integrals(&wrapped, u).t)
            })
            .collect()
    }
}

fn omega_or_panic(spec: &SymplecticSpectrum, layer: &str) -> Vec<f64> {
    match spec {
        SymplecticSpectrum::Stable { omega } => omega.clone(),
        SymplecticSpectrum::Resonant { signed } => panic!(
            "{layer} endpoint is symplectically RESONANT — {} unstable pair(s), metric {:?} \
             (λ_min = {:.6e}), lowest signed ω² = {:?}. Criticality is not stability; no real ω \
             exists here.",
            signed.n_neg,
            signed.metric,
            signed.metric_min_eig,
            signed.omega2.first()
        ),
        SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } => panic!(
            "{layer} endpoint is symplectically UNSTABLE — reason = {reason:?}, generator has {} \
             eigenvalues with max |Im| = {:.6e}. Criticality is not stability; no ω exists here.",
            generator_eigs.len(),
            generator_eigs
                .iter()
                .map(|z| z.im.abs())
                .fold(0.0f64, f64::max)
        ),
    }
}

#[test]
fn tscr_b0_endpoint_and_stability_gate() {
    let e = endpoint();
    let dev = e.su2.lambda + e.ints.e0 - e.casscf_gs;
    assert!(
        dev.abs() <= 1e-8,
        "su2 pinned endpoint energy: λ+e0 = {:.12} vs meta casscf_gs = {:.12}, Δ = {dev:.3e}",
        e.su2.lambda + e.ints.e0,
        e.casscf_gs
    );
    let dlam = e.su2.lambda - e.ab.lambda;
    eprintln!(
        "tscr_b0: cross-layer Δλ = {dlam:.3e} (su2 {:.12} vs ab {:.12})",
        e.su2.lambda, e.ab.lambda
    );
    assert!(
        dlam.abs() <= 1e-10,
        "the two layers' pinned optima disagree: Δλ = {dlam:.3e}"
    );

    let ctx = su2_joint_context(&e.su2.h, &e.su2.x, &e.joint_opts())
        .unwrap_or_else(|err| panic!("su2 joint context refused at the endpoint: {err:?}"));
    let (basis, report) = ctx.joint_basis();
    assert_eq!(ctx.n_eff(), 6, "n_eff at the pinned CASSCF endpoint");
    assert_eq!(report.dropped, 0, "no orbital seed should be dropped here");
    assert_eq!(
        basis.len(),
        110,
        "joint basis = (CSF dim − 1) + n_eff = (105 − 1) + 6; got n_tt = {}, n_orb_kept = {}",
        report.n_tt,
        report.n_orb_kept
    );

    let omega = omega_or_panic(&ctx.joint_symplectic_eigenvalues(&basis), "su2");
    assert_eq!(omega.len(), 110, "one ω per joint basis direction");
    assert!(
        omega[0] > 1e-3,
        "ω₁ = {:.6e} is at the floor — the endpoint is marginal, not stable (iso_defect = {:.3e}, \
         sigma_min = {:.3e})",
        omega[0],
        ctx.iso_defect(),
        ctx.sigma_min()
    );
    assert!(
        omega.windows(2).all(|w| w[0] <= w[1]),
        "the symplectic spectrum must come back ascending"
    );
    eprintln!(
        "tscr_b0: STABLE, ω₁ = {:.9}, ω_max = {:.9}; iso_defect = {:.3e}, sigma_min = {:.3e}, \
         orth_error = {:.3e}",
        omega[0],
        omega[omega.len() - 1],
        ctx.iso_defect(),
        ctx.sigma_min(),
        ctx.orth_error()
    );
}

#[test]
#[should_panic(expected = "PropRoute::Shifted is unavailable on the SU(2) layer")]
fn tscr_b5_shifted_route_is_refused() {
    let e = endpoint();
    let ctx = su2_joint_context(&e.su2.h, &e.su2.x, &e.joint_opts()).expect("su2 context");
    let _ = su2_polarizability_static(
        &ctx,
        &e.props(&e.su2.u),
        &StaticResponseOpts {
            route: PropRoute::Shifted,
            ..StaticResponseOpts::default()
        },
    );
}
