// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::OnceLock;

use qntt::core::tangent::min_schmidt;
use qntt::core::weingarten::{FloorPolicy, bond_gauges};
use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::pointgroup::Irrep;
use qntt::rotation::{orbital_gradient, rotate_integrals};
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::hess::{NewtonPolishOpts, SymplecticSpectrum, newton_polish};
use qntt::solvers::joint::cas::{CasPartition, cas_pairs, cas_rank_grid, cas_reference_state};
use qntt::solvers::joint::{
    JointContext, JointOpts, JointScfOpts, JointScfOut, joint_context, joint_ground_state,
    joint_solver_diagnostics,
};
use qntt::solvers::riemannian::{normalize_mut, tangent_bases};
use qntt::vector::perturbation;
use qntt::{QnSpace, RankGrid, SparseHamiltonian, TtVector, dot, rayleigh_quotient, tt_state_in};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::joint::cas::{fmt_grid_diff, grid_diff, grid_le};
use crate::joint::cas_dense::*;
use crate::joint::util::*;

pub(crate) struct AnchorRun {
    pub(crate) label: &'static str,
    pub(crate) ints: Integrals<f64>,
    pub(crate) space: QnSpace,
    pub(crate) part: CasPartition,
    pub(crate) ci_irreps: Option<Vec<usize>>,
    pub(crate) out: JointScfOut,
}

fn deficiency_aware_grid(
    untyped: &QnSpace,
    dirname: &str,
    part: &CasPartition,
    m_active: usize,
) -> Vec<RankGrid> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(dirname);
    let typed = load_fixture_dir::<f64>(
        &dir,
        &IntegralsOptions {
            point_group: PointGroupPolicy::Resolve(None),
            ..IntegralsOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{dirname} typed: {e:?}"));
    assert!(
        typed.space.pg.order() > 1,
        "{dirname}: the typed space must actually carry a point group, or the deficiency-aware \
         grid is just the untyped one and route (a) is vacuous"
    );
    let typed_grid = cas_rank_grid(&typed.space, part, qntt::PerSector(m_active));
    let mut out: Vec<RankGrid> = Vec::with_capacity(typed_grid.len());
    for (b, tg) in typed_grid.iter().enumerate() {
        let mut g = RankGrid::zeros(untyped);
        for s in untyped.bond_sectors(b) {
            g[s] = typed
                .space
                .bond_sectors(b)
                .into_iter()
                .filter(|t| t.nup == s.nup && t.ndn == s.ndn)
                .map(|t| tg[t])
                .sum();
        }
        out.push(g);
    }
    out
}

fn reference_state_on_grid(
    ham: &SparseHamiltonian<f64>,
    space: &QnSpace,
    grid: &[RankGrid],
    seed: u64,
) -> (f64, TtVector<f64>) {
    let d = space.d;
    let occ_up: Vec<bool> = (0..d).map(|i| i < space.nup).collect();
    let occ_dn: Vec<bool> = (0..d).map(|i| i < space.ndn).collect();
    let hf: TtVector<f64> = tt_state_in(*space, &occ_up, &occ_dn);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x = perturbation(&hf, grid, 0.2, &mut rng);
    x.round_ranks_mut(grid);
    normalize_mut(&mut x);
    let (_l, x_als) = als(
        ham,
        &x,
        &AlsOptions {
            eps: 1e-12,
            max_iter: 200,
            ..AlsOptions::default()
        },
    );
    let (mut x, _info) = newton_polish(
        ham,
        &x_als,
        &NewtonPolishOpts {
            gradtol: 1e-11,
            ..NewtonPolishOpts::default()
        },
    );
    normalize_mut(&mut x);
    x.check();
    assert!(
        grid_le(space, &x.ranks, grid),
        "reference_state_on_grid left the pinned grid from ABOVE: {}",
        fmt_grid_diff(&grid_diff(space, &x.ranks, grid))
    );
    let lam = rayleigh_quotient(ham, &x);
    (lam, x)
}

fn anchor_scf_opts(part: &CasPartition, d: usize) -> JointScfOpts {
    JointScfOpts {
        pairs: Some(cas_pairs(part, d)),
        orb_max_iter: 40,
        orb_grad: 1e-6,
        orb_grad_full: Some(1e-6),
        tt_grad_tol: 1e-6,
        orb_floor: 1e-8,
        construction: Construction::Median,
        tt_preserve_grid: true,
        ..JointScfOpts::default()
    }
}

fn load_untyped(dirname: &str) -> (Integrals<f64>, QnSpace) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(dirname);
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{dirname}: {e:?}"));
    let ints = load.ints;
    let nup = (ints.nelec + ints.ms2.unsigned_abs() as usize) / 2;
    let ndn = ints.nelec - nup;
    let space = QnSpace::new(ints.norb, nup, ndn);
    (ints, space)
}

pub(crate) fn lih_anchor() -> &'static AnchorRun {
    static CELL: OnceLock<AnchorRun> = OnceLock::new();
    CELL.get_or_init(|| {
        let (ints, space) = load_untyped("lih_sto3g");
        let part = CasPartition {
            n_core: 1,
            n_active: 2,
            n_virt: 3,
        };
        let _grid = cas_rank_grid(&space, &part, qntt::PerSector::NONE);
        let ham = SparseHamiltonian::new_with(
            space,
            &ints.t,
            &ints.v,
            None,
            f64::EPSILON,
            Construction::Median,
        );
        let (_lam0, x0) = cas_reference_state(&ham, &space, &part, 0, 0xCA22);
        let out = joint_ground_state(&ints, space, &x0, &anchor_scf_opts(&part, space.d));
        AnchorRun {
            label: "lih_cas22",
            ints,
            space,
            part,
            ci_irreps: None,
            out,
        }
    })
}

pub(crate) fn h2o_anchor_route_a() -> &'static AnchorRun {
    static CELL: OnceLock<AnchorRun> = OnceLock::new();
    CELL.get_or_init(|| {
        let (ints, space) = load_untyped("h2o_sto3g");
        let part = CasPartition {
            n_core: 1,
            n_active: 6,
            n_virt: 0,
        };
        let grid = deficiency_aware_grid(&space, "h2o_sto3g", &part, 0);
        let ham = SparseHamiltonian::new_with(
            space,
            &ints.t,
            &ints.v,
            None,
            f64::EPSILON,
            Construction::Median,
        );
        let (_lam0, x0) = reference_state_on_grid(&ham, &space, &grid, 0xCA86);
        let out = joint_ground_state(&ints, space, &x0, &anchor_scf_opts(&part, space.d));
        let ci_irreps = Some(cas_determinant_irreps("h2o_sto3g", &part, &space));
        AnchorRun {
            label: "h2o_cas86_route_a",
            ints,
            space,
            part,
            ci_irreps,
            out,
        }
    })
}

pub(crate) fn h2o_anchor_full() -> &'static AnchorRun {
    static CELL: OnceLock<AnchorRun> = OnceLock::new();
    CELL.get_or_init(|| {
        let stage_a = h2o_anchor_route_a();
        assert!(
            stage_a.out.info.converged,
            "h2o full-grid anchor: stage (a) must have converged before its orbitals may be used \
             as a warm start ({:?})",
            stage_a.out.info.refused_reason
        );
        let ints = rotate_integrals(&stage_a.ints, &stage_a.out.u);
        let space = stage_a.space;
        let part = stage_a.part;
        let _grid = cas_rank_grid(&space, &part, qntt::PerSector::NONE);
        let ham = SparseHamiltonian::new_with(
            space,
            &ints.t,
            &ints.v,
            None,
            f64::EPSILON,
            Construction::Median,
        );
        let (_lam0, x0) = cas_reference_state(&ham, &space, &part, 0, 0xCA86);
        let out = joint_ground_state(&ints, space, &x0, &anchor_scf_opts(&part, space.d));
        AnchorRun {
            label: "h2o_cas86_full",
            ints,
            space,
            part,
            ci_irreps: None,
            out,
        }
    })
}

fn cas_determinant_irreps(dirname: &str, part: &CasPartition, untyped: &QnSpace) -> Vec<usize> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(dirname);
    let typed = load_fixture_dir::<f64>(
        &dir,
        &IntegralsOptions {
            point_group: PointGroupPolicy::Resolve(None),
            ..IntegralsOptions::default()
        },
    )
    .unwrap_or_else(|e| panic!("{dirname} typed: {e:?}"));
    let pg = typed.space.pg;
    let sp = DetSpace::new(untyped.d, untyped.nup, untyped.ndn);
    let (_full_pos, sp_act) = sp.cas_subset(part);
    sp_act
        .dets
        .iter()
        .map(|(ou, od)| {
            let sites = ou
                .iter()
                .enumerate()
                .filter(|(_, b)| **b)
                .map(|(i, _)| i + part.n_core)
                .chain(
                    od.iter()
                        .enumerate()
                        .filter(|(_, b)| **b)
                        .map(|(i, _)| i + part.n_core),
                );
            let g: Irrep = pg.irrep_of_sites(sites);
            g.index() as usize
        })
        .collect()
}

fn report_conditioning(run: &AnchorRun, jctx: &JointContext<'_>) -> f64 {
    let diags = joint_solver_diagnostics(jctx);
    let (bl, br) = tangent_bases(&run.out.x);
    let smin = min_schmidt(&bl, &br);
    let dropped: f64 = bond_gauges(&bl, &br, FloorPolicy::default_hard())
        .iter()
        .map(|g| g.dropped_mass)
        .sum();
    eprintln!(
        "tjcan {} CONDITIONING: n_tt = {}, n_eff = {}, space_dim = {}, μ ∈ [{:.3e}, {:.3e}], \
         dropped = {}, orth_error = {:.3e}, sigma_min = {:.3e}, min_schmidt = {:.3e}, \
         gauge_dropped_mass = {:.3e}, ‖g_L‖ = {:.3e}, ‖g_tt‖ = {:.3e}, converged = {}",
        run.label,
        diags.n_tt,
        diags.n_eff,
        diags.space_dim,
        diags.mu_min,
        diags.mu_max,
        diags.dropped,
        diags.orth_error,
        jctx.sigma_min(),
        smin,
        dropped,
        run.out.info.g_orb_full.last().copied().unwrap_or(f64::NAN),
        run.out
            .info
            .tt_gradnorms
            .last()
            .copied()
            .unwrap_or(f64::NAN),
        run.out.info.converged,
    );
    diags.orth_error
}

fn anchor_context(run: &AnchorRun) -> JointContext<'_> {
    assert!(
        run.out.info.converged,
        "{}: ★ PUBLISH GATE — no ω may be read from an unconverged SCF. refused_reason = {:?}, \
         ‖g_L‖ history = {:?}, ‖g_tt‖ history = {:?}",
        run.label, run.out.info.refused_reason, run.out.info.g_orb_full, run.out.info.tt_gradnorms,
    );
    let jctx = joint_context(
        &run.out.h,
        &run.out.x,
        &JointOpts {
            pairs: Some(cas_pairs(&run.part, run.space.d)),
            ..JointOpts::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!(
            "{}: the accepted optimum must be a certified joint-critical point: {e}",
            run.label
        )
    });
    let n_orb = cas_pairs(&run.part, run.space.d).keep.len();
    assert_eq!(
        jctx.n_eff(),
        n_orb,
        "{}: n_eff must equal the non-redundant pair count. A DROPPED orbital direction means the \
         two charts span different spaces and the entrywise comparison below is meaningless.",
        run.label
    );
    let max_tau = jctx
        .orbital_projections()
        .iter()
        .fold(0.0f64, |a, t| a.max(t.norm()));
    eprintln!(
        "tjcan {}: max_a ‖τ_a‖ = {max_tau:.3e} (must vanish: it is what collapses N = S − τᵀτ to \
         N = S and makes G = diag(I,S) on BOTH sides)",
        run.label
    );
    assert!(
        max_tau <= 1e-12,
        "{}: τ_a = P(u_a) must vanish on the non-redundant set (got {max_tau:.3e}); otherwise the \
         joint metric is not the dense reference's metric and the anchor equality compares two \
         different quadratic forms",
        run.label
    );
    jctx
}

fn omega_dev(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "ω list lengths differ: {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
}

fn dense_opts_for(run: &AnchorRun) -> DenseLrOpts {
    DenseLrOpts {
        ci_irreps: run.ci_irreps.clone(),
        ..DenseLrOpts::default()
    }
}

#[test]
fn tjcan_c2_anchor_equality_lih() {
    let run = lih_anchor();
    let jctx = anchor_context(run);
    let orth = report_conditioning(run, &jctx);

    let (basis, report) = jctx.joint_basis();
    assert_eq!(
        basis.len(),
        report.n_tt + report.n_orb_kept,
        "lih: joint basis bookkeeping"
    );
    assert_eq!(
        report.dropped, 0,
        "lih: the MGS dropped {} orbital seeds — the two charts no longer span the same space (ratios {:?})",
        report.dropped, report.ratios
    );
    assert_eq!(
        (report.n_tt, report.n_orb_kept),
        (3, 11),
        "lih: chart dimension must be 3 + 11 = 14 (the brief's '~231' arithmetic at h2o was off by \
         one for the same reason: the radial mode is excluded)"
    );

    let SymplecticSpectrum::Stable { omega } = jctx.joint_symplectic_eigenvalues(&basis) else {
        panic!(
            "lih: the joint spectrum at a converged CAS optimum must be Stable; an Unstable result \
             here is the FINDING (record min_eig, orth_error = {orth:.3e}, sigma_min = {:.3e}, \
             ‖g_L‖ = {:?}) and is NOT gate-loosened",
            jctx.sigma_min(),
            run.out.info.g_orb_full.last()
        );
    };

    let rotated = rotate_integrals(&run.ints, &run.out.u);
    let lr = dense_lr_full_with(&rotated, &run.part, &dense_opts_for(run));
    let omega_dense = lr.omega();

    assert_eq!(
        (lr.dim_ci, lr.dim_orb),
        (report.n_tt, report.n_orb_kept),
        "lih: the two charts must have the same block dimensions"
    );
    let dev = omega_dev(&omega, &omega_dense);
    let scale = omega.iter().fold(0.0f64, |a, &x| a.max(x)).max(1.0);
    eprintln!(
        "tjcan_c2 lih: ω_joint = {:?}\n            ω_dense = {:?}\n            max|Δω| = \
         {dev:.3e} (relative {:.3e}); u_orth_dev = {:.3e}, μ(S) ∈ [{:.3e}, {:.3e}]",
        omega,
        omega_dense,
        dev / scale,
        run.out.info.u_orth_dev,
        lr.diag.mu_min,
        lr.diag.mu_max,
    );

    let (mu_dense, _v) = dense_eigen(&lr.s);
    let mu_joint = jctx.orb_spectrum();
    let mu_dev = mu_dense
        .iter()
        .zip(mu_joint)
        .fold(0.0f64, |a, (&x, &y)| a.max((x - y).abs()));
    eprintln!("tjcan_c2 lih: eig(S_dense) vs jctx.orb_spectrum(): max Δ = {mu_dev:.3e}");
    assert!(
        mu_dev <= 1e-8 * lr.diag.mu_max.max(1.0),
        "lih: the two sides' orbital metrics differ (max Δμ = {mu_dev:.3e}); the ω comparison \
         below would then be between two different inner products"
    );

    assert!(
        dev <= 1e-11 * scale,
        "lih ANCHOR EQUALITY: max|Δω| = {dev:.3e} against ω scale {scale:.3e}. Quote it beside \
         ‖g_L‖ = {:?}, ‖g_tt‖ = {:?}, orth_error = {orth:.3e}, sigma_min = {:.3e}.",
        run.out.info.g_orb_full.last(),
        run.out.info.tt_gradnorms.last(),
        jctx.sigma_min(),
    );
}

#[test]
fn tjcan_c4_anti_vacuity_floors_at_the_optimum() {
    for cold in [lih_anchor(), h2o_anchor_route_a()] {
        let g0 = cold.out.info.g_orb_full.first().copied().unwrap_or(0.0);
        eprintln!("tjcan_c4 {}: start-of-SCF ‖g_L‖ = {g0:.3e}", cold.label);
        assert!(
            g0 >= 1e-5,
            "{}: the orbital gradient was already at the gate when the SCF STARTED — the run had \
             nothing to do and the energy pin in tjcan_c1 is vacuous",
            cold.label
        );
    }

    for run in [lih_anchor(), h2o_anchor_full()] {
        let jctx = anchor_context(run);
        let _orth = report_conditioning(run, &jctx);
        let rotated = rotate_integrals(&run.ints, &run.out.u);
        let lr = dense_lr_full_with(&rotated, &run.part, &dense_opts_for(run));

        let z = jctx.normal_residual();
        let z_norm = dot(z, z).sqrt();
        let mut max_b_orb = 0.0f64;
        let mut max_b_cross = 0.0f64;
        let mut max_b_tt = 0.0f64;
        for i in 0..lr.dim_ci {
            for j in 0..lr.dim_ci {
                max_b_tt = max_b_tt.max(lr.b[i][j].abs());
            }
            for a in 0..lr.dim_orb {
                max_b_cross = max_b_cross.max(lr.b[i][lr.dim_ci + a].abs());
            }
        }
        for a in 0..lr.dim_orb {
            for b in 0..lr.dim_orb {
                max_b_orb = max_b_orb.max(lr.b[lr.dim_ci + a][lr.dim_ci + b].abs());
            }
        }
        let g_start = {
            let g = orbital_gradient(&run.out.h, &run.out.x);
            cas_pairs(&run.part, run.space.d)
                .keep
                .iter()
                .fold(0.0f64, |m, &(p, q)| m.max(g.get(&[p, q]).abs()))
        };
        eprintln!(
            "tjcan_c4 {}: ‖z‖ = {z_norm:.3e} (dense side {:.3e}), max|B_orb| = {max_b_orb:.3e}, \
             max|B_cross| = {max_b_cross:.3e}, max|B_tt| = {max_b_tt:.3e} (PRINTED ONLY — the flat \
             corollary is tjcas_a5's), n_eff = {}, max|g_pq| at the optimum = {g_start:.3e}",
            run.label,
            lr.diag.z_norm,
            jctx.n_eff(),
        );

        assert!(
            z_norm >= 5e-3,
            "{}: ‖z‖ = {z_norm:.3e} is vacuous — B_orb and B_cross both contract z, so a small \
             ‖z‖ means the anchor has no RPA character left to certify",
            run.label
        );
        assert!(
            (z_norm - lr.diag.z_norm).abs() <= 1e-6 * z_norm.max(1.0),
            "{}: the two sides disagree on ‖z‖ ({z_norm:.3e} vs {:.3e}) — they are not at the same \
             point, which would invalidate every comparison in this file",
            run.label,
            lr.diag.z_norm
        );
        assert!(
            max_b_orb >= 5e-3,
            "{}: max|B_orb| = {max_b_orb:.3e} is vacuous; the ω-visible negative arms of tjcd_b8 \
             have nothing to bite on",
            run.label
        );
        assert!(
            max_b_cross >= 5e-4,
            "{}: max|B_cross| = {max_b_cross:.3e} is vacuous",
            run.label
        );
        assert!(
            jctx.n_eff() >= 1,
            "{}: no orbital redundancy — the joint stack degenerates to TT-RPA exactly (which is \
             not an error, but this anchor then certifies nothing about the orbital sector)",
            run.label
        );
    }
}
