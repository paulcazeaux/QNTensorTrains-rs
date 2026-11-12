// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::sync::OnceLock;

use qntt::core::tangent::TtTangent;
use qntt::integrals::{PropertyOperator, property_hamiltonian};
use qntt::rotation::PairMask;
use qntt::solvers::joint::{
    JointOpts, PropRoute, ResponseError, StaticResponseOpts, joint_context, polarizability_static,
    property_gradient, property_gradient_with, property_value,
};
use qntt::{CoeffTensor, QnSpace, TtVector, dot, xt_h_y};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::optimize::d2_fixture;
use crate::util::*;

fn local_radial_mode(bl: &TtVector<f64>, br: &TtVector<f64>) -> TtTangent<f64> {
    let mut r = TtTangent::project(bl, br, bl);
    let n = r.norm();
    assert!(
        n > 0.0,
        "local_radial_mode: base point projects to zero (degenerate fixture)"
    );
    r.scale_mut(1.0 / n);
    r
}

struct SosFixture {
    space: QnSpace,
    ham: qntt::SparseHamiltonian<f64>,
    x: TtVector<f64>,
    dets: Vec<(Vec<bool>, Vec<bool>)>,
    evals: Vec<f64>,
    evecs: Vec<Vec<f64>>,
    m1: CoeffTensor<f64>,
    m2: CoeffTensor<f64>,
}

fn build_sos_fixture() -> SosFixture {
    let d = 3usize;
    let space = QnSpace::new(d, 1, 1);
    let (ham, x, _bl, _br) = weingarten_minimizer(space, 0xABCD, 1, 1);
    let dets = dets_for(d, 1, 1);
    let (t, v) = generic_truncated_tv(d);
    let hd = dense_hamiltonian_matrix(d, &t, &v, &dets);
    let (evals, evecs) = dense_eigen(&hd);

    let mut m1 = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        m1.set(&[i, i], (i as f64) - 1.0);
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9002);
    let mut m2 = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in i..d {
            use rand::RngExt;
            let val = rng.random_range(-1.0f64..1.0);
            m2.set(&[i, j], val);
            m2.set(&[j, i], val);
        }
    }
    SosFixture {
        space,
        ham,
        x,
        dets,
        evals,
        evecs,
        m1,
        m2,
    }
}

fn sos_fixture() -> &'static SosFixture {
    static CELL: OnceLock<SosFixture> = OnceLock::new();
    CELL.get_or_init(build_sos_fixture)
}

fn dense_sos_alpha(evals: &[f64], evecs: &[Vec<f64>], mu_a: &[Vec<f64>], mu_b: &[Vec<f64>]) -> f64 {
    let v0 = &evecs[0];
    let mu_a_v0 = mat_vec(mu_a, v0);
    let mu_b_v0 = mat_vec(mu_b, v0);
    let mut acc = 0.0;
    for n in 1..evals.len() {
        let an = dot_vec(&evecs[n], &mu_a_v0);
        let bn = dot_vec(&evecs[n], &mu_b_v0);
        acc += an * bn / (evals[n] - evals[0]);
    }
    2.0 * acc
}

#[test]
fn tjs_gradient_pins() {
    let fx = d2_fixture();
    let out = &fx.out;
    let x = &out.x;
    let d = x.space.d;

    let jctx =
        joint_context(&out.h, x, &JointOpts::default()).expect("D2 fixture must be TT-critical");

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7701);
    let m = random_sym_t(d, &mut rng);
    let prop = PropertyOperator::new("tjs_gradient_pins", m);
    let mu_ham = property_hamiltonian(x.space, &prop);

    let want_m0 = xt_h_y(x, &mu_ham, x) / dot(x, x);
    let got_m0 = property_value(&jctx, &mu_ham);
    assert_close(
        got_m0,
        want_m0,
        1e-12 * want_m0.abs().max(1.0),
        "property_value vs xt_h_y",
    );

    let (bl, br) = jctx.bases();
    let mu_x = &mu_ham * x;
    let mut f_xi = TtTangent::project(bl, br, &mu_x);
    f_xi.axpy(-want_m0, &TtTangent::project(bl, br, x));
    let f_c_full: Vec<f64> = jctx
        .orbital_directions()
        .iter()
        .map(|u_a| xt_h_y(u_a, &mu_ham, x))
        .collect();
    let f_c_kept = jctx.vt_times(&f_c_full);
    let g_want = jctx.joint_raise(f_xi, f_c_kept);

    let g_got = property_gradient(&jctx, &mu_ham);

    let mut diff = g_got.xi.clone();
    diff.axpy(-1.0, &g_want.xi);
    let scale = g_want.xi.norm().max(1.0);
    assert!(
        diff.norm() <= 1e-9 * scale,
        "tangent leg of property_gradient vs bruteforce covector: diff={:e} scale={:e}",
        diff.norm(),
        scale
    );
    for (i, (&got, &want)) in g_got.c.iter().zip(g_want.c.iter()).enumerate() {
        assert_close(
            got,
            want,
            1e-9 * want.abs().max(1.0),
            &format!("orbital coeff {i}"),
        );
    }

    let radial = local_radial_mode(bl, br);
    let rad_dot = radial.dot(&g_got.xi);
    assert!(
        rad_dot.abs() <= 1e-9 * g_got.xi.norm().max(1.0),
        "raised gradient must be radial-deflated: ⟨radial,g⟩ = {rad_dot:e}"
    );

    let max_fc = f_c_full.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    assert!(max_fc > 1e-6, "anti-vacuity: max|F_c full| = {max_fc:e}");
}

#[test]
fn tjs_sos_fullrank() {
    let fx = sos_fixture();
    let jctx = joint_context(&fx.ham, &fx.x, &JointOpts::default())
        .expect("SOS fixture must be TT-critical");
    assert_eq!(
        jctx.n_eff(),
        0,
        "SOS fixture must be full rank (n_eff == 0)"
    );
    let anchor = jctx
        .orb_spectrum()
        .iter()
        .cloned()
        .fold(0.0f64, f64::max)
        .max(1.0);
    assert!(
        jctx.orb_spectrum()
            .iter()
            .all(|&mu| mu.abs() <= 1e-8 * anchor),
        "orb_spectrum must be a pure-roundoff cluster at full rank: {:?}",
        jctx.orb_spectrum()
    );

    let props = [
        PropertyOperator::new("diag", fx.m1.clone()),
        PropertyOperator::new("offdiag", fx.m2.clone()),
    ];
    let mus: Vec<_> = props
        .iter()
        .map(|p| property_hamiltonian(fx.space, p))
        .collect();

    let opts = StaticResponseOpts {
        tol: 1e-10,
        ..Default::default()
    };
    let resp = polarizability_static(&jctx, &props, &opts).expect("full-rank SOS arm must solve");

    let mu_d: Vec<Vec<Vec<f64>>> = props
        .iter()
        .map(|p| dense_kappa_matrix(fx.space.d, &p.m, &fx.dets))
        .collect();
    let n = props.len();
    let mut want = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            want[i * n + j] = dense_sos_alpha(&fx.evals, &fx.evecs, &mu_d[i], &mu_d[j]);
        }
    }
    for i in 0..n {
        for j in 0..n {
            assert_close(
                resp.alpha[i * n + j],
                want[i * n + j],
                1e-8 * want[i * n + j].abs().max(1.0),
                &format!("alpha[{i},{j}] vs dense SOS"),
            );
        }
    }
    let max_alpha = resp.alpha.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
    assert!(max_alpha > 1e-3, "anti-vacuity: max|alpha| = {max_alpha:e}");
    assert!(
        resp.alpha[1].abs() > 1e-4,
        "anti-vacuity: off-diagonal alpha[0,1] must be non-negligible: {:e}",
        resp.alpha[1]
    );
    assert!(
        resp.alpha_defect <= 1e-10 * max_alpha.max(1.0),
        "alpha_defect too large: {:e}",
        resp.alpha_defect
    );

    let x_d = dense_state_vector(&fx.dets, &fx.x);
    let xnorm2 = dot_vec(&x_d, &x_d);
    for (i, mu) in mus.iter().enumerate() {
        let want_val = dot_vec(&x_d, &mat_vec(&mu_d[i], &x_d)) / xnorm2;
        assert_close(
            resp.values[i],
            want_val,
            1e-12 * want_val.abs().max(1.0),
            "values vs dense",
        );
        let via_property_value = property_value(&jctx, mu);
        assert_close(
            resp.values[i],
            via_property_value,
            1e-12 * via_property_value.abs().max(1.0),
            "StaticResponse::values vs property_value",
        );
    }
}

#[test]
fn tjs_orbital_rows_twin() {
    let fx = d2_fixture();
    let out = &fx.out;
    let x = &out.x;
    let d = x.space.d;

    let jctx =
        joint_context(&out.h, x, &JointOpts::default()).expect("D2 fixture must be TT-critical");

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7702);
    let m = random_sym_t(d, &mut rng);
    let prop = PropertyOperator::new("tjs_orbital_rows_twin", m);
    let mu_ham = property_hamiltonian(x.space, &prop);

    let g = qntt::rotation::orbital_gradient(&mu_ham, x);
    let gam = qntt::rdm::transition_1rdm_h(x, &mu_ham, x);
    let mut max_gc = 0.0f64;
    for &(p, q) in jctx.pairs() {
        let gc = gam.get(&[p, q]) - gam.get(&[q, p]);
        let twin = 0.5 * g.get(&[p, q]);
        assert_close(
            gc,
            twin,
            1e-12 * twin.abs().max(1.0),
            "orbital row vs ½·orbital_gradient",
        );
        max_gc = max_gc.max(gc.abs());
    }
    assert!(max_gc > 1e-6, "anti-vacuity: max|g_c| = {max_gc:e}");

    let sfx = sos_fixture();
    let sjctx = joint_context(&sfx.ham, &sfx.x, &JointOpts::default())
        .expect("SOS fixture must be TT-critical");
    assert_eq!(sjctx.n_eff(), 0, "SOS fixture must stay full rank");
    let sprop = PropertyOperator::new("route_twin", sfx.m2.clone());
    let smu = property_hamiltonian(sfx.space, &sprop);
    let g_shift = property_gradient_with(&sjctx, &smu, PropRoute::Shifted);
    let g_mat = property_gradient_with(&sjctx, &smu, PropRoute::Materialized);
    let mut diff = g_shift.xi.clone();
    diff.axpy(-1.0, &g_mat.xi);
    let scale = g_shift.xi.norm().max(g_mat.xi.norm()).max(1.0);
    assert!(
        diff.norm() <= 1e-13 * scale,
        "Shifted vs Materialized covector-level agreement: diff={:e} scale={:e}",
        diff.norm(),
        scale
    );
}

#[test]
fn tjs_refusals() {
    let space3 = QnSpace::new(3, 1, 1);
    let (ham3, xstar, bl3, br3) = weingarten_minimizer(space3, 0xF001, 1, 1);
    let xnc = noncritical_point(&xstar, &bl3, &br3, 0.3, 0xF002);
    let diag_opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..JointOpts::default()
    };
    let diag_ctx =
        joint_context(&ham3, &xnc, &diag_opts).expect("diagnosis-only context always gates");
    let mut rng1 = Xoshiro256PlusPlus::seed_from_u64(0xF003);
    let m1 = random_sym_t(3, &mut rng1);
    let prop1 = PropertyOperator::new("diag_only", m1);
    match polarizability_static(&diag_ctx, &[prop1], &StaticResponseOpts::default()) {
        Err(ResponseError::NotJointCritical {
            tt_gradnorm,
            tt_bound,
            ..
        }) => {
            assert!(
                tt_gradnorm > tt_bound,
                "measured must exceed bound: {tt_gradnorm:e} vs {tt_bound:e}"
            );
        }
        other => panic!("expected NotJointCritical (diagnosis-only), got {other:?}"),
    }

    let (space_li, ham_li, x_li, _t, _v) = lih_cation_rank1_fixture();
    let jctx_li = joint_context(&ham_li, &x_li, &JointOpts::default())
        .expect("LiH+ fixture must be TT-critical");
    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0xF004);
    let m2 = random_sym_t(space_li.d, &mut rng2);
    let prop2 = PropertyOperator::new("lih_orbital_noncritical", m2);
    match polarizability_static(&jctx_li, &[prop2], &StaticResponseOpts::default()) {
        Err(ResponseError::NotJointCritical {
            orb_grad_kept,
            orb_grad_full,
            orb_bound,
            ..
        }) => {
            assert!(
                orb_grad_full > orb_bound || orb_grad_kept > orb_bound,
                "measured orbital gradient must exceed the bound: kept={orb_grad_kept:e} \
                 full={orb_grad_full:e} bound={orb_bound:e}"
            );
        }
        other => panic!("expected NotJointCritical (orbital), got {other:?}"),
    }

    let fx = d2_fixture();
    let out = &fx.out;
    let jctx_d2 = joint_context(&out.h, &out.x, &JointOpts::default())
        .expect("D2 fixture must be TT-critical");
    let mut rng3 = Xoshiro256PlusPlus::seed_from_u64(0xF005);
    let m3 = random_sym_t(fx.space.d, &mut rng3);
    let prop3 = PropertyOperator::new("d2_max_iter1", m3);
    let tight_opts = StaticResponseOpts {
        tol: 1e-14,
        max_iter: 1,
        ..StaticResponseOpts::default()
    };
    match polarizability_static(&jctx_d2, &[prop3], &tight_opts) {
        Err(ResponseError::SolveNotConverged {
            iters,
            rel_residual,
            tol,
            ..
        }) => {
            assert_eq!(iters, 1);
            assert!(
                rel_residual > tol,
                "reported residual {rel_residual:e} must genuinely exceed tol {tol:e}"
            );
        }
        other => panic!("expected SolveNotConverged, got {other:?}"),
    }
}

#[test]
fn tjs_edge_cases() {
    let fx = sos_fixture();
    let jctx = joint_context(&fx.ham, &fx.x, &JointOpts::default())
        .expect("SOS fixture must be TT-critical");
    assert_eq!(jctx.n_eff(), 0);
    let opts = StaticResponseOpts::default();

    let zero_m = CoeffTensor::<f64>::zeros(fx.space.d, 2);
    let zero_prop = PropertyOperator::new("zero", zero_m);
    let resp0 = polarizability_static(&jctx, &[zero_prop], &opts)
        .expect("zero property must solve trivially");
    assert_eq!(resp0.alpha[0], 0.0);
    assert_eq!(resp0.solves[0].rel_residual, 0.0);
    assert!(!resp0.alpha[0].is_nan());

    let mut const_m = CoeffTensor::<f64>::zeros(fx.space.d, 2);
    for i in 0..fx.space.d {
        const_m.set(&[i, i], 2.5);
    }
    let const_prop = PropertyOperator::new("const", const_m.clone());
    let mu_const = property_hamiltonian(fx.space, &const_prop);
    let g_const = property_gradient(&jctx, &mu_const);
    assert!(
        g_const.xi.norm() <= 1e-11,
        "constant property must give a near-zero gradient: {:e}",
        g_const.xi.norm()
    );
    let resp_const =
        polarizability_static(&jctx, &[const_prop], &opts).expect("const property must solve");
    assert!(
        resp_const.alpha[0].abs() <= 1e-9,
        "constant property must give alpha ≈ 0: {:e}",
        resp_const.alpha[0]
    );

    let empty = JointOpts {
        pairs: Some(PairMask::with_predicate(fx.space.d, |_, _| false)),
        ..JointOpts::default()
    };
    let jctx_tt = joint_context(&fx.ham, &fx.x, &empty).expect("empty-pair-list context");
    assert_eq!(jctx_tt.n_eff(), 0);
    let props = [
        PropertyOperator::new("d1", fx.m1.clone()),
        PropertyOperator::new("d2", fx.m2.clone()),
    ];
    let resp_default = polarizability_static(&jctx, &props, &opts).expect("default-pairs solve");
    let resp_empty = polarizability_static(&jctx_tt, &props, &opts).expect("empty-pairs solve");
    for (a, b) in resp_default.alpha.iter().zip(resp_empty.alpha.iter()) {
        assert_close(
            *a,
            *b,
            1e-14 * a.abs().max(1.0),
            "n_eff==0 must forward literally regardless of pair-list content",
        );
    }
}
