// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::OnceLock;

use qntt::core::tangent::tangent_basis;
use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::load_fixture_dir;
use qntt::solvers::InnerProductVector;
use qntt::solvers::hess::{NewtonPolishOpts, newton_polish};
use qntt::solvers::joint::{
    FcRoute, JointOpts, JointScfOpts, JointScfOut, JointSolver, JointVector, SchurRoute,
    joint_context, joint_ground_state,
};
use qntt::solvers::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg};
use qntt::{
    Integrals, IntegralsOptions, Irrep, PointGroup, QnSpace, SparseHamiltonian, TtVector,
    freeze_core, orbital_gradient, rayleigh_quotient, rotate_integrals, tt_randn,
};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use qntt_testkit::joint_d2::{D2Problem, d2_problem, d2_scf_opts};

use crate::util::capped_ranks;
use crate::util::*;

pub(crate) fn capped_randn(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks = capped_ranks(&space, lo, hi, &mut rng);
    let mut x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    normalize_mut(&mut x);
    x
}

fn u_orth_dev(u: &[f64], d: usize) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0f64;
            for k in 0..d {
                s += u[k * d + i] * u[k * d + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((s - want).abs());
        }
    }
    worst
}

fn tt_only_converge(ham: &SparseHamiltonian<f64>, x0: &TtVector<f64>) -> f64 {
    let space = x0.space;
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-10;
    opts.max_iter = 300;
    let (_l, mut x) = riemannian_cg(ham, x0, &opts);
    for _ in 0..10 {
        if grad_exact_at(ham, &x).norm() < 1e-9 {
            break;
        }
        let (_l2, x2) = riemannian_cg(ham, &x, &opts);
        x = x2;
    }
    let (xp, _info) = newton_polish(
        ham,
        &x,
        &NewtonPolishOpts {
            gradtol: 1e-9,
            ..Default::default()
        },
    );
    rayleigh_quotient(ham, &xp)
}

#[test]
fn tjo_d1_fullrank_one_step() {
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, xstar, _bl, _br) = weingarten_minimizer(space, 0xD100, 1, 1);
    let (t, v) = generic_truncated_tv(d);
    let ints = Integrals::<f64> {
        t,
        v,
        e0: 0.0,
        norb: d,
        nelec: space.n(),
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    };
    let lambda0 = rayleigh_quotient(&ham, &xstar);

    let opts = JointScfOpts {
        orb_max_iter: 5,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&ints, space, &xstar, &opts);

    assert!(
        out.info.converged,
        "full-rank fixture must converge: {:?}",
        out.info.refused_reason
    );
    assert_eq!(
        out.info.n_eff, 0,
        "full rank ⇒ every orbital direction is tangential ⇒ n_eff == 0"
    );
    assert_eq!(
        out.info.iters, 1,
        "full rank ⇒ ~1 orbital iteration (structural, per the translation spec's D1)"
    );
    assert!(
        out.info.delta_k_norms.is_empty(),
        "no orbital step should ever be taken at n_eff==0: {:?}",
        out.info.delta_k_norms
    );
    assert!(
        (out.lambda - lambda0).abs() < 1e-10,
        "energy must be unchanged to 1e-10: {} vs {}",
        out.lambda,
        lambda0
    );

    for i in 0..d {
        for j in 0..d {
            let want = if i == j { 1.0 } else { 0.0 };
            assert_eq!(
                out.u[i * d + j],
                want,
                "U must stay exactly I at n_eff==0 (row {i}, col {j})"
            );
        }
    }

    assert!(!out.info.e_history.is_empty());
    let e0 = out.info.e_history[0];
    assert!(
        out.info.e_history.iter().all(|&e| (e - e0).abs() < 1e-9),
        "energy history must be flat: {:?}",
        out.info.e_history
    );
    assert!(
        out.info.g_orb_full.last().copied().unwrap_or(f64::INFINITY) <= 1e-6,
        "full-pair orbital gradient must already be tiny at an exact eigenstate: {:?}",
        out.info.g_orb_full
    );
}

pub(crate) use qntt_testkit::joint_d2::D2_ORB_GRAD;

pub(crate) struct D2Fixture {
    pub(crate) space: QnSpace,
    pub(crate) ints: Integrals<f64>,
    pub(crate) e_fci: f64,
    pub(crate) lambda_tt_only: f64,
    pub(crate) out: JointScfOut,
}

fn build_d2_fixture() -> D2Fixture {
    let D2Problem {
        t,
        v,
        ints,
        space,
        x0,
    } = d2_problem();
    let d = space.d;
    let dets = dets_for(d, 2, 1);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let (evals, _evecs) = dense_eigen(&hd);
    let e_fci = evals[0];

    let ham = SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    let lambda_tt_only = tt_only_converge(&ham, &x0);

    let out = joint_ground_state(&ints, space, &x0, &d2_scf_opts());

    D2Fixture {
        space,
        ints,
        e_fci,
        lambda_tt_only,
        out,
    }
}

pub(crate) fn d2_fixture() -> &'static D2Fixture {
    static CELL: OnceLock<D2Fixture> = OnceLock::new();
    CELL.get_or_init(build_d2_fixture)
}

pub(crate) struct D2Point {
    pub(crate) space: QnSpace,
    pub(crate) h: SparseHamiltonian<f64>,
    pub(crate) x: TtVector<f64>,
}

fn load_d2_point() -> D2Point {
    load_tracked_d2("testdata/joint_d2")
}

fn load_tracked_d2(rel: &str) -> D2Point {
    use qntt::tt_io::load_tt;

    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    assert!(
        dir.join("x.tt").is_file(),
        "the tracked D2 endpoint is not present at {dir:?}; it is a testdata/ fixture, so its \
         absence is a real failure, not an environmental skip"
    );
    let ints = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{dir:?}: {e}"))
        .ints;
    assert_eq!(
        (ints.norb, ints.nelec, ints.ms2),
        (6, 3, 1),
        "testdata/joint_d2 is not the D2 problem"
    );
    let space = QnSpace::new(ints.norb, 2, 1);
    let x = load_tt::<f64>(dir.join("x.tt")).unwrap_or_else(|e| panic!("x.tt: {e:?}"));
    x.check();
    assert_eq!(x.space, space, "x.tt is not a (2,1) state over d = 6");
    let opts = d2_scf_opts();
    let h = SparseHamiltonian::new_with(
        space,
        &ints.t,
        &ints.v,
        None,
        opts.ham_eps,
        opts.construction,
    );
    D2Point { space, h, x }
}

pub(crate) fn d2_point() -> &'static D2Point {
    static CELL: OnceLock<D2Point> = OnceLock::new();
    CELL.get_or_init(load_d2_point)
}

pub(crate) fn d2r22_point() -> &'static D2Point {
    static CELL: OnceLock<D2Point> = OnceLock::new();
    CELL.get_or_init(|| load_tracked_d2("testdata/joint_d2_r22"))
}

#[test]
fn tjo_d2_tracked_endpoint_is_joint_critical() {
    assert_tracked_endpoint_is_joint_critical(d2_point(), "testdata/joint_d2");
}

#[test]
fn tjo_d2r22_tracked_endpoint_is_joint_critical() {
    assert_tracked_endpoint_is_joint_critical(d2r22_point(), "testdata/joint_d2_r22");
}

fn assert_tracked_endpoint_is_joint_critical(p: &D2Point, rel: &str) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let meta = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .expect("reload")
        .meta
        .expect("meta.txt");
    let lambda_rec: f64 = meta
        .keys
        .get("lambda")
        .and_then(|v| v.parse().ok())
        .expect("meta.txt carries the run's `lambda`");
    let lam = rayleigh_quotient(&p.h, &p.x);
    assert!(
        (lam - lambda_rec).abs() <= 1e-12 * lambda_rec.abs(),
        "{rel}: RQ {lam} vs recorded λ {lambda_rec}"
    );
    let jctx = joint_context(&p.h, &p.x, &JointOpts::default())
        .unwrap_or_else(|e| panic!("{rel}: the tracked endpoint must be joint-critical: {e:?}"));
    assert!(jctx.n_eff() >= 1, "{rel} must be genuinely truncated");
    let g = orbital_gradient(&p.h, &p.x);
    let d = p.space.d;
    let gnorm = (0..d)
        .flat_map(|i| (0..d).map(move |j| (i, j)))
        .map(|(i, j)| g.get(&[i, j]).powi(2))
        .sum::<f64>()
        .sqrt();
    eprintln!(
        "{rel}: λ {lam:.15} n_eff {} σ_min {:.3e} ‖g_orb‖ {gnorm:.3e}",
        jctx.n_eff(),
        jctx.sigma_min()
    );
    assert!(
        gnorm <= D2_ORB_GRAD,
        "{rel}: full orbital gradient {gnorm:.3e} above the run's gate {D2_ORB_GRAD:e}"
    );
}

#[test]
fn tjo_d2_refill_reproduces_the_rebuild_path() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);
    let opts = JointScfOpts {
        orb_max_iter: 40,
        orb_grad: D2_ORB_GRAD,
        tt_grad_tol: 1e-6,
        construction: Construction::Median,
        refill_h: true,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&fx.ints, fx.space, &x0, &opts);
    assert!(
        out.info.converged,
        "refill-driven run must certify like the rebuild-driven one: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.h.is_refillable(),
        "the returned operator must be the refillable skeleton"
    );
    let scale = fx.out.lambda.abs().max(1.0);
    assert!(
        (out.lambda - fx.out.lambda).abs() <= 1e-9 * scale,
        "refill λ {} vs rebuild λ {} (rel {:.3e})",
        out.lambda,
        fx.out.lambda,
        (out.lambda - fx.out.lambda).abs() / scale
    );
    assert!(
        out.info.ls_build_count >= 1,
        "at least one trial must have been refilled: {}",
        out.info.ls_build_count
    );
    let rot = rotate_integrals(&fx.ints, &out.u);
    let plain = SparseHamiltonian::new_with(
        fx.space,
        &rot.t,
        &rot.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );
    let (q, qp) = (
        rayleigh_quotient(&out.h, &out.x),
        rayleigh_quotient(&plain, &out.x),
    );
    assert!(
        (q - qp).abs() <= 1e-12 * q.abs().max(1.0),
        "returned operator is not H(rotate_integrals(ints, u)): RQ {q} vs {qp}"
    );
}

#[test]
fn tjo_d2_truncated_beats_tt() {
    let fx = d2_fixture();
    let out = &fx.out;

    assert!(
        out.info.g_orb_full.first().copied().unwrap_or(0.0) > 1e-3,
        "orbital gradient at the START must be non-vacuous: {:?}",
        out.info.g_orb_full.first()
    );

    assert!(
        out.info.converged,
        "D2 fixture must converge: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.info.g_orb_kept.last().copied().unwrap_or(f64::INFINITY) <= D2_ORB_GRAD,
        "kept-space orbital gradient must have met its own gate: {:?}",
        out.info.g_orb_kept
    );
    let tt_bound = 1e-6 * out.lambda.abs().max(1.0);
    assert!(
        out.info
            .tt_gradnorms
            .last()
            .copied()
            .unwrap_or(f64::INFINITY)
            <= tt_bound,
        "TT leg must still be polished at its own gate: {:?}",
        out.info.tt_gradnorms
    );
    assert!(
        out.info.g_orb_full.last().copied().unwrap_or(f64::INFINITY) <= 1e-6,
        "full-pair orbital gradient must be converged: {:?}",
        out.info.g_orb_full
    );
    assert!(
        out.info
            .g_orb_dropped
            .last()
            .copied()
            .unwrap_or(f64::INFINITY)
            <= 1e-6,
        "dropped-component orbital gradient must be converged: {:?}",
        out.info.g_orb_dropped
    );

    assert!(
        out.info.delta_k_norms.iter().any(|&dk| dk > 0.0),
        "an orbital step must have happened: {:?}",
        out.info.delta_k_norms
    );

    assert!(
        out.lambda >= fx.e_fci - 1e-10,
        "variational floor: joint λ={} below dense FCI={}",
        out.lambda,
        fx.e_fci
    );

    let margin = fx.lambda_tt_only - out.lambda;
    assert!(
        margin >= 1e-5,
        "joint must beat the TT-only reference by a measurable margin: {margin:e}"
    );

    let dev = u_orth_dev(&out.u, fx.space.d);
    assert!(dev <= 1e-12, "U not orthogonal (recomputed): {dev:e}");
    assert!(
        out.info.u_orth_dev <= 1e-12,
        "U not orthogonal (self-reported): {:e}",
        out.info.u_orth_dev
    );

    assert!(out.info.ls_build_count > 0, "no H rebuilds were counted");
    assert!(
        out.info.ls_build_count < 100_000,
        "sanity bound on the rebuild count: {}",
        out.info.ls_build_count
    );

    eprintln!(
        "tjo_d2: iters={} trials_per_iter={:?} terminal ‖gk‖={:e} last δK={:?} lambda={} \
         lambda_tt_only={} e_fci={} margin={margin:e}",
        out.info.iters,
        out.info.trials_per_iter,
        out.info.g_orb_kept.last().copied().unwrap_or(f64::NAN),
        out.info.delta_k_norms.last(),
        out.lambda,
        fx.lambda_tt_only,
        fx.e_fci,
    );
}

#[test]
fn tjo_d3_central_validity() {
    let fx = d2_fixture();
    let out = &fx.out;
    assert!(
        out.info.converged,
        "D3 requires a converged D2 point to be legal: {:?}",
        out.info.refused_reason
    );

    let jctx = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("the joint_ground_state output must be a certified joint-critical point");

    let (bl, br) = jctx.bases();
    let tb = tangent_basis(bl, br, true);
    assert!(
        tb.len() >= 9,
        "need >=9 tangent directions for the D3 direction recipe: got {}",
        tb.len()
    );
    let n_eff = jctx.n_eff();
    assert!(
        n_eff >= 1,
        "D2 fixture must be genuinely truncated (n_eff>=1): got {n_eff}"
    );
    let z = jctx.normal_residual();
    assert!(
        z.norm() > 1e-4,
        "z vacuous -- the FD arm would collapse to the numerically-blind full-rank case: {:e}",
        z.norm()
    );

    let gl_full: Vec<f64> = {
        let g = orbital_gradient(&out.h, &out.x);
        jctx.pairs().iter().map(|&(p, q)| g.get(&[p, q])).collect()
    };
    let gl_norm = gl_full.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        gl_norm <= D2_ORB_GRAD,
        "fixture gate: full-pair orbital gradient must be converged at the D2 solution \
         (same quantity, same number as the library's own D2 orb_grad): {gl_norm:e}"
    );

    let spec = jctx.spec();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD300);
    let mut vecs: Vec<JointVector> = Vec::new();
    for i in 0..2 {
        let mut vv = JointVector::from_tangent(spec, tb[i % tb.len()].clone());
        vv.c = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
        vecs.push(vv);
    }
    vecs.push(JointVector::from_tangent(spec, tb[0].clone()));
    let c_only: Vec<f64> = (0..n_eff).map(|_| rng.random_range(-1.0..1.0)).collect();
    vecs.push(JointVector::from_coeffs(spec, &jctx, c_only));
    assert_c1_selfadjoint(&jctx, &vecs);

    let mut xi_dir = tb[0].clone();
    xi_dir.axpy(0.7, &tb[3]);
    xi_dir.axpy(-0.4, &tb[8]);
    let xn = xi_dir.norm();
    xi_dir.scale_mut(1.0 / xn);

    let n_orb = jctx.pairs().len();
    let mut a_dir: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    let an = a_dir.iter().map(|val| val * val).sum::<f64>().sqrt();
    for val in a_dir.iter_mut() {
        *val /= an;
    }

    let ints_cur = rotate_integrals(&fx.ints, &out.u);

    let h_step = 1e-4;
    let (fxi_fd, fc_fd) = joint_covector_fd_central(
        &ints_cur.t,
        &ints_cur.v,
        fx.space,
        jctx.pairs(),
        &a_dir,
        &out.x,
        &xi_dir,
        h_step,
    );

    let (a_xi, a_c) = jctx.a_covector(&xi_dir, &a_dir, FcRoute::Sweep);
    let (b_xi, b_c) = jctx.b_covector(&xi_dir, &a_dir, FcRoute::Sweep);
    let mut fxi_ref = a_xi;
    fxi_ref.axpy(1.0, &b_xi);
    let fc_ref: Vec<f64> = a_c.iter().zip(&b_c).map(|(&a, &b)| a + b).collect();

    let max_bc = b_c.iter().fold(0.0f64, |acc, &val| acc.max(val.abs()));
    assert!(
        max_bc > 2e-3,
        "B term vacuous at the D2 point -- the FD arm would be toothless: {max_bc:e}"
    );
    assert!(
        fxi_ref.norm() > 1e-3,
        "reference covector's tangent leg vacuous: {:e}",
        fxi_ref.norm()
    );
    assert!(
        b_xi.norm() > 1e-3,
        "B term's own tangent leg vacuous at the D2 point -- b_xi would contribute nothing for \
         the FD arm to catch: {:e}",
        b_xi.norm()
    );
    let max_fc_ref = fc_ref.iter().fold(0.0f64, |acc, &val| acc.max(val.abs()));
    assert!(
        max_fc_ref > 1e-6,
        "reference covector's orbital leg vacuous: {max_fc_ref:e}"
    );

    let mut dxi = fxi_fd.clone();
    dxi.axpy(-1.0, &fxi_ref);
    let dev_xi = dxi.norm();
    let tol_xi = 1e-6 * fxi_ref.norm().max(1.0);
    assert!(
        dev_xi <= tol_xi,
        "FD-of-gradient tangent leg vs reference: {dev_xi:e} > tol {tol_xi:e}"
    );

    let dev_c = fc_fd
        .iter()
        .zip(&fc_ref)
        .fold(0.0f64, |acc, (&f, &r)| acc.max((f - r).abs()));
    let tol_c = 1e-6 * max_fc_ref.max(1.0);
    assert!(
        dev_c <= tol_c,
        "FD-of-gradient orbital leg vs reference: {dev_c:e} > tol {tol_c:e}"
    );

    eprintln!("tjo_d3: dev_xi={dev_xi:e} (tol {tol_xi:e}), dev_c={dev_c:e} (tol {tol_c:e})");
}

#[test]
fn tjo_dfc_frozen_core_composes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints_full = load.ints;
    let ints_act = freeze_core(&ints_full, 1);
    let d_act = ints_act.norb;
    assert_eq!(d_act, ints_full.norb - 1, "one core orbital folded away");
    assert_eq!(
        ints_act.nelec,
        ints_full.nelec - 2,
        "one doubly-occupied core folded into e0'"
    );

    let space_act = QnSpace::from_n_two_sz(d_act, ints_act.nelec, ints_act.ms2);

    let dets = dets_for(d_act, space_act.nup, space_act.ndn);
    let hd = dense_hamiltonian_matrix(d_act, &ints_act.t, &ints_act.v, &dets);
    let (evals, _evecs) = dense_eigen(&hd);
    let e_fci_active = evals[0] + ints_act.e0;

    let x0 = capped_randn(space_act, 1, 2, 0xDFC0);

    let opts = JointScfOpts {
        orb_max_iter: 15,
        construction: Construction::Median,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&ints_act, space_act, &x0, &opts);

    assert!(
        out.info.converged,
        "frozen-core run must actually certify -- a refusal (e.g. a stall or veto specific to \
         the active-space pair list) would still leave `lambda` satisfying the variational floor \
         below via the plain envelope-theorem bound, so that assertion alone cannot catch it: \
         {:?}",
        out.info.refused_reason
    );
    assert_eq!(
        out.u.len(),
        d_act * d_act,
        "u must be sized to the ACTIVE d, not the full-space d"
    );
    let max_pairs = d_act * d_act.saturating_sub(1) / 2;
    assert!(
        out.info.orb_spectrum.len() <= max_pairs,
        "orb_spectrum must live on the active pair list (filtered by PG): {} > {max_pairs}",
        out.info.orb_spectrum.len()
    );

    assert!(
        out.lambda + ints_act.e0 >= e_fci_active - 1e-9,
        "joint energy below the active-space FCI floor: {} < {}",
        out.lambda + ints_act.e0,
        e_fci_active
    );
}

#[test]
fn tjo_refuse() {
    let (space, _ham, x, t, v) = lih_cation_rank1_fixture();
    let ints = Integrals::<f64> {
        t,
        v,
        e0: 0.0,
        norb: space.d,
        nelec: space.n(),
        ms2: 1,
        orbsym: Vec::new(),
        isym: None,
    };

    let opts = JointScfOpts {
        orb_max_iter: 30,
        orb_grad: 1e-6,
        orb_grad_full: Some(1e-9),
        orb_floor: 1e-2,
        construction: Construction::Median,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&ints, space, &x, &opts);

    assert!(
        out.info.converged,
        "the kept gradient and TT gate are both met, so this must certify: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.info.refused_reason.is_none(),
        "a rank-limited point is not a refusal: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.info.rank_limited,
        "the dropped residual is above orb_grad_full, so this point MUST be flagged rank-limited"
    );

    let gk_last = out.info.g_orb_kept.last().copied().unwrap_or(f64::INFINITY);
    let gl_last = out.info.g_orb_full.last().copied().unwrap_or(0.0);
    assert!(
        gk_last <= opts.orb_grad,
        "the KEPT gradient must have met its own gate -- the refusal is about the DROPPED part: \
         {gk_last:e}"
    );
    assert!(
        gl_last > opts.orb_grad_full.unwrap(),
        "the FULL gradient must NOT have met the tight full threshold: {gl_last:e}"
    );
    assert!(
        out.info.g_dropped_final > opts.orb_grad_full.unwrap(),
        "rank_limited must be driven by the dropped component: {:e}",
        out.info.g_dropped_final
    );
    assert!(
        out.info.dropped_angle.is_finite() && out.info.dropped_angle > 0.0,
        "a dropped mode exists, so its angle must be reported: {}",
        out.info.dropped_angle
    );
    assert!(
        out.info.dropped_leak_bound > 0.0,
        "the Cauchy-Schwarz leak bound must be stamped alongside"
    );

    assert!(
        out.info.iters <= 10,
        "the two-sided-gate refusal should fire promptly once the kept gradient converges, not \
         after burning most of orb_max_iter: iters={}",
        out.info.iters
    );
}

#[test]
fn tjo_basis_identity_contract() {
    let fx = d2_fixture();
    let out = &fx.out;
    let d = fx.space.d;

    let mut orth_dev = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..d {
                s += out.u[k * d + i] * out.u[k * d + j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            orth_dev = orth_dev.max((s - want).abs());
        }
    }
    assert!(
        orth_dev <= 1e-12,
        "u must be orthogonal: dev = {orth_dev:e}"
    );

    let norm2 = qntt::dot(&out.x, &out.x);
    assert!(
        (norm2 - 1.0).abs() <= 1e-12,
        "the returned x must be unit-norm: ⟨x,x⟩ = {norm2}"
    );
    let rq = rayleigh_quotient(&out.h, &out.x);
    assert!(
        (rq - out.lambda).abs() <= 1e-10 * out.lambda.abs().max(1.0),
        "lambda must be the Rayleigh value of the returned (h, x): {rq:.15e} vs {:.15e}",
        out.lambda
    );

    let rot = rotate_integrals(&fx.ints, &out.u);
    let h2 = SparseHamiltonian::new_with(
        fx.space,
        &rot.t,
        &rot.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );
    for seed in [11u64, 12, 13] {
        let a = capped_randn(fx.space, 1, 2, seed);
        let b = capped_randn(fx.space, 1, 2, seed ^ 0xFF);
        let lhs = qntt::xt_h_y(&a, &out.h, &b);
        let rhs = qntt::xt_h_y(&a, &h2, &b);
        assert!(
            (lhs - rhs).abs() <= 1e-11 * lhs.abs().max(1.0),
            "the returned h must be the rotated-basis Hamiltonian: ⟨a,Hb⟩ = {lhs:.15e} vs \
             fresh-build {rhs:.15e} (seed {seed})"
        );
    }
}

#[test]
fn tjo_pg_zero_cells_loop() {
    let d = 6usize;
    let pg = PointGroup::new(
        2,
        &[
            Irrep::new(0),
            Irrep::new(1),
            Irrep::new(2),
            Irrep::new(0),
            Irrep::new(1),
            Irrep::new(2),
        ],
    );
    let space = QnSpace::with_point_group(d, 2, 2, pg, Irrep::IDENTITY);
    let (t, v) = pg_symmetric_truncated_tv(d, pg);
    let ints = Integrals::<f64> {
        t,
        v,
        e0: 0.0,
        norb: d,
        nelec: 4,
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    };
    assert!(
        space.pg.order() > 1,
        "the synthetic table must give a nontrivial point group"
    );

    let x0 = capped_randn(space, 1, 1, 0x9A70);

    let opts = JointScfOpts {
        orb_max_iter: 2,
        orb_grad: 1e-9,
        tt_grad_tol: 1e-6,
        construction: Construction::Median,
        ..JointScfOpts::default()
    };
    let out = joint_ground_state(&ints, space, &x0, &opts);

    assert!(
        out.info.iters >= 1,
        "expected at least one orbital iteration to run"
    );
    assert!(
        out.info.ls_build_count >= 1,
        "expected at least one line-search H rebuild -- otherwise the bitwise-zero forbidden-\
         cell check below never exercises a ROTATED rebuild: ls_build_count={}",
        out.info.ls_build_count
    );
    assert!(
        out.info.delta_k_norms.iter().any(|&dk| dk > 0.0),
        "expected at least one accepted orbital step (nonzero ‖δK‖): {:?}",
        out.info.delta_k_norms
    );

    for p in 0..d {
        for q in 0..d {
            if space.pg.irrep_of_sites([p, q]) != Irrep::IDENTITY {
                assert_eq!(
                    out.u[p * d + q],
                    0.0,
                    "u[{p},{q}] must stay exactly 0 across irreps"
                );
            }
        }
    }

    let ints_final = rotate_integrals(&ints, &out.u);
    for p in 0..d {
        for q in 0..d {
            if space.pg.irrep_of_sites([p, q]) != Irrep::IDENTITY {
                assert_eq!(
                    ints_final.t.get(&[p, q]),
                    0.0,
                    "forbidden t cell not exactly zero at [{p},{q}]"
                );
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if space.pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                        assert_eq!(
                            ints_final.v.get(&[i, j, k, l]),
                            0.0,
                            "forbidden v cell not exactly zero at [{i},{j},{k},{l}]"
                        );
                    }
                }
            }
        }
    }
}

fn d2_schur_opts(route: SchurRoute) -> JointScfOpts {
    JointScfOpts {
        solver: JointSolver::DenseSchur,
        schur_route: route,
        orb_max_iter: 2,
        orb_grad: D2_ORB_GRAD,
        tt_grad_tol: 1e-6,
        construction: Construction::Median,
        ..JointScfOpts::default()
    }
}

#[test]
fn tjo_sc1_schur_routes_drive_the_same_step() {
    let fx = d2_fixture();
    let x0 = capped_randn(fx.space, 1, 1, 57);

    let dense = joint_ground_state(&fx.ints, fx.space, &x0, &d2_schur_opts(SchurRoute::Dense));
    let mf = joint_ground_state(
        &fx.ints,
        fx.space,
        &x0,
        &d2_schur_opts(SchurRoute::MatrixFree),
    );

    for (label, out) in [("dense", &dense), ("matrix-free", &mf)] {
        assert_eq!(
            out.info.solver,
            JointSolver::DenseSchur,
            "{label}: the nested branch must be the one that ran"
        );
        assert!(
            out.info.g_orb_full.first().copied().unwrap_or(0.0) > 1e-3,
            "{label}: anti-vacuity — the starting point must have a real orbital gradient, got \
             {:?}",
            out.info.g_orb_full.first()
        );
        assert_eq!(
            out.info.delta_k_norms.len(),
            1,
            "{label}: exactly one orbital step must have been ACCEPTED (a refused line search \
             would make both routes agree trivially at u = I); trials = {:?}, refused = {:?}",
            out.info.trials_per_iter,
            out.info.refused_reason
        );
        assert!(
            out.info.delta_k_norms[0] > 1e-4,
            "{label}: anti-vacuity — the accepted step must be a real rotation, ‖δK‖_F = {:.3e}",
            out.info.delta_k_norms[0]
        );
        let d = fx.ints.norb;
        let off_identity = (0..d)
            .flat_map(|i| (0..d).map(move |j| (i, j)))
            .map(|(i, j)| (out.u[i * d + j] - if i == j { 1.0 } else { 0.0 }).abs())
            .fold(0.0f64, f64::max);
        assert!(
            off_identity > 1e-4,
            "{label}: anti-vacuity — u must be far from the identity, max|u − I| = \
             {off_identity:.3e}"
        );
    }

    let d = fx.ints.norb;
    let du = (0..d * d)
        .map(|k| (dense.u[k] - mf.u[k]).abs())
        .fold(0.0f64, f64::max);
    let dlam = (dense.lambda - mf.lambda).abs();
    let ddk = (dense.info.delta_k_norms[0] - mf.info.delta_k_norms[0]).abs();
    eprintln!(
        "tjo_sc1: max|Δu| = {du:.3e}, |Δλ| = {dlam:.3e}, |Δ‖δK‖_F| = {ddk:.3e}; \
         ‖δK‖_F = {:.6e} / {:.6e}, trials = {:?} / {:?}, λ = {:.12} / {:.12}",
        dense.info.delta_k_norms[0],
        mf.info.delta_k_norms[0],
        dense.info.trials_per_iter,
        mf.info.trials_per_iter,
        dense.lambda,
        mf.lambda,
    );
    assert!(
        du <= 2e-8,
        "the two SchurRoutes must drive the same rotation (measured max|Δu| = 1.356e-9): got \
         {du:.3e}"
    );
    assert!(
        dlam <= 5e-11,
        "...and land at the same energy (measured |Δλ| = 2.560e-12): got {dlam:.3e}"
    );
    assert!(
        ddk <= 5e-9,
        "...with the same step length (measured |Δ‖δK‖_F| = 3.633e-10 on ‖δK‖_F = 4.8557e-1): \
         got {ddk:.3e}"
    );
}

fn sc_fro(m: &[f64]) -> f64 {
    m.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn sc_cg_dense(a: &[f64], n: usize, b: &[f64], tol: f64, cap: usize) -> (Vec<f64>, usize, f64) {
    let matvec = |v: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|i| (0..n).map(|j| a[i * n + j] * v[j]).sum())
            .collect()
    };
    let bn = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mut x = vec![0.0f64; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rz: f64 = r.iter().map(|v| v * v).sum();
    for it in 0..cap {
        let ap = matvec(&p);
        let pap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        if pap <= 0.0 {
            let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            return (x, it, rn / bn.max(1e-300));
        }
        let alpha = rz / pap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
        if rn / bn.max(1e-300) <= tol {
            return (x, it + 1, rn / bn.max(1e-300));
        }
        let rz_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = rz_new / rz;
        rz = rz_new;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
    }
    let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
    (x, cap, rn / bn.max(1e-300))
}

#[test]
fn tjo_sc2_shear_leaves_heff_invariant() {
    let fx = d2_fixture();
    let jctx = joint_context(&fx.out.h, &fx.out.x, &JointOpts::default())
        .expect("the D2 endpoint must re-gate");
    let (basis, report) = jctx.joint_basis();
    let n_tt = report.n_tt;
    let n_eff = jctx.n_eff();
    assert_eq!(
        report.dropped, 0,
        "this arm indexes the orbital block by kept index, so it needs a drop-free basis; \
         dropped = {:?}",
        report.dropped_idx
    );
    assert!(
        n_tt > 10 && n_eff > 5,
        "anti-vacuity: the D2 endpoint must have a rich joint basis, got n_tt = {n_tt}, \
         n_eff = {n_eff}"
    );

    let tangent_block: Vec<JointVector> = basis[..n_tt].to_vec();

    let mut raw_orb: Vec<JointVector> = Vec::with_capacity(n_eff);
    let mut sheared_orb: Vec<JointVector> = Vec::with_capacity(n_eff);
    let mut min_shift = f64::INFINITY;
    for j in 0..n_eff {
        let mut c = vec![0.0f64; n_eff];
        c[j] = 1.0;
        let seed = JointVector::from_coeffs(jctx.spec(), &jctx, c);
        let mut sh = seed.clone();
        for _pass in 0..2 {
            for b in tangent_block.iter() {
                let cf = b.dot(&sh);
                sh.axpy(-cf, b);
            }
        }
        let mut delta = sh.clone();
        delta.axpy(-1.0, &seed);
        min_shift = min_shift.min(delta.norm() / seed.norm().max(1e-300));
        raw_orb.push(seed);
        sheared_orb.push(sh);
    }
    assert!(
        min_shift > 0.5,
        "anti-vacuity: the MGS must actually shear every direction — the kept orbital seeds lie \
         almost entirely inside the tangent space, so a relative move below 0.5 means the \
         orthogonalization stopped subtracting; min = {min_shift:.3e}"
    );

    let heff_of = |orb: &[JointVector]| -> (Vec<f64>, f64, f64) {
        let mut bs: Vec<JointVector> = tangent_block.clone();
        bs.extend_from_slice(orb);
        let nn = bs.len();
        let (ga, gb) = jctx.joint_hessian_grams(&bs);
        let m: Vec<f64> = ga.iter().zip(gb.iter()).map(|(x, y)| x + y).collect();
        let htt: Vec<f64> = (0..n_tt)
            .flat_map(|i| (0..n_tt).map(move |j| (i, j)))
            .map(|(i, j)| 0.5 * (m[i * nn + j] + m[j * nn + i]))
            .collect();
        let htc: Vec<f64> = (0..n_tt)
            .flat_map(|i| (0..n_eff).map(move |j| (i, j)))
            .map(|(i, j)| m[i * nn + (n_tt + j)])
            .collect();
        let hcc: Vec<f64> = (0..n_eff)
            .flat_map(|i| (0..n_eff).map(move |j| (i, j)))
            .map(|(i, j)| m[(n_tt + i) * nn + (n_tt + j)])
            .collect();
        let mut schur = vec![0.0f64; n_eff * n_eff];
        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(n_eff);
        for j in 0..n_eff {
            let col: Vec<f64> = (0..n_tt).map(|i| htc[i * n_eff + j]).collect();
            let (y, its, res) = sc_cg_dense(&htt, n_tt, &col, 1e-14, 4000);
            assert!(
                its < 4000,
                "the ORACLE inner solve must converge — {its} iterations at rel-res {res:.3e}; \
                 a capped oracle would make the invariance claim meaningless"
            );
            ys.push(y);
        }
        for i in 0..n_eff {
            for j in 0..n_eff {
                schur[i * n_eff + j] = (0..n_tt).map(|k| htc[k * n_eff + i] * ys[j][k]).sum();
            }
        }
        let heff: Vec<f64> = (0..n_eff * n_eff).map(|k| hcc[k] - schur[k]).collect();
        let ratio = sc_fro(&hcc) / sc_fro(&heff).max(1e-300);
        (heff, ratio, sc_fro(&schur))
    };

    let (heff_raw, ratio_raw, schur_raw) = heff_of(&raw_orb);
    let (heff_sh, ratio_sh, schur_sh) = heff_of(&sheared_orb);
    let gap = sc_fro(
        &(0..n_eff * n_eff)
            .map(|k| heff_raw[k] - heff_sh[k])
            .collect::<Vec<f64>>(),
    ) / sc_fro(&heff_raw).max(1e-300);
    eprintln!(
        "tjo_sc2: ‖Schur‖ {schur_raw:.4e} → {schur_sh:.4e}, cancellation ratio \
         {ratio_raw:.3e} → {ratio_sh:.3e}, ‖ΔHeff‖/‖Heff‖ = {gap:.3e} \
         (n_tt = {n_tt}, n_eff = {n_eff}, min relative shear move = {min_shift:.3e})"
    );

    assert!(
        ratio_raw > 8.0,
        "the RAW coordinates must genuinely cancel (measured 1.607e1) or this arm is testing \
         nothing; got ‖Hcc‖/‖Heff‖ = {ratio_raw:.3e}"
    );
    assert!(
        ratio_sh < 1.2,
        "the SHEARED coordinates must collapse the cancellation to O(1) (measured 1.046e0); got \
         ‖Hcc‖/‖Heff‖ = {ratio_sh:.3e} against raw {ratio_raw:.3e}"
    );
    assert!(
        schur_sh <= schur_raw / 50.0,
        "the condensed term must collapse with it (measured 9.6138e0 → 3.2846e-2, 293×); got \
         {schur_raw:.3e} → {schur_sh:.3e}"
    );

    assert!(
        gap <= 5e-13,
        "the shear must leave Heff INVARIANT (measured 5.524e-14): ‖Heff_raw − Heff_sheared‖ / \
         ‖Heff‖ = {gap:.3e}. A congruence — a renormalized shear — fails here, and so does a \
         shear that drops one of its three Hcc terms."
    );
}
