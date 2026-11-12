// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use std::ffi::c_char;
use std::mem::{align_of, offset_of, size_of};

use qntt::elementaryops::property_flux_classes;
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::solvers::hess::{
    SectorCtx, SectorMomentOpts, SectorStaticOpts, SectorTransitionMoments, weingarten_operator,
};
use qntt::{CoeffTensor, PropertyOperator, TtVector};
use qntt_capi::err::{QNTT_NOT_CONVERGED, QNTT_NULL_ARG};
use qntt_capi::handle::{Handle, borrow};
use qntt_capi::opts::{
    QNTT_PROP_ROUTE_MATERIALIZED, QNTT_PROP_ROUTE_SHIFTED, QnttSectorAlphaInfo,
    QnttSectorAlphaOpts, QnttSectorMomentOpts, QnttSectorMomentsInfo,
};
use qntt_capi::rpa::*;

const REFUSAL_NONE: i32 = 0;
const REFUSAL_NOT_TT_CRITICAL: i32 = 1;
const REFUSAL_HESSIAN_INDEFINITE: i32 = 2;
const REFUSAL_SYMPLECTIC_UNSTABLE: i32 = 3;
const REFUSAL_SOLVE_NOT_CONVERGED: i32 = 4;
const REFUSAL_UNCERTIFIED_NEGATIVE: i32 = 5;
const REFUSAL_NEGATIVE_OMEGA: i32 = 7;
const REFUSAL_NEAR_RESONANCE: i32 = 8;
const REFUSAL_INDEFINITE_ABOVE_POLE: i32 = 9;

const PHI_LIVE: u8 = 3;
const PHI_EMPTY: u8 = 1;

type HamHandle = Handle<qntt::SparseHamiltonian<f64>>;
type VecHandle = Handle<TtVector<f64>>;

fn fx4_point_group() -> PointGroup {
    PointGroup::new(
        2,
        &SITE_IRREPS.map(|ir| Irrep::new(u8::try_from(ir).expect("a 0-based XOR index"))),
    )
}

fn synthetic_pairs(phi: u8) -> Vec<(usize, usize)> {
    let pg = fx4_point_group();
    let mut out = Vec::new();
    for p in 0..D {
        for q in (p + 1)..D {
            if pg.irrep_of(p).fuse(pg.irrep_of(q)) == Irrep::new(phi) {
                out.push((p, q));
            }
        }
    }
    out
}

fn matrix_on_pairs(pairs: &[(usize, usize)], weights: &[f64]) -> Vec<f64> {
    assert_eq!(pairs.len(), weights.len());
    let mut m = vec![0.0f64; D * D];
    for (&(p, q), &w) in pairs.iter().zip(weights) {
        assert_ne!(
            p, q,
            "a diagonal cell is same-irrep and would make m multi-class"
        );
        m[p * D + q] = w;
        m[q * D + p] = w;
    }
    m
}

fn synthetic_m(phi: u8, subset: Option<&[usize]>) -> Vec<f64> {
    let all = synthetic_pairs(phi);
    assert!(
        !all.is_empty(),
        "pairs({phi}) is empty on [A, B, C, B]: no representable observable of that flux"
    );
    let pairs: Vec<(usize, usize)> = match subset {
        None => all,
        Some(idx) => idx.iter().map(|&i| all[i]).collect(),
    };
    let weights: Vec<f64> = (0..pairs.len())
        .map(|i| if i % 2 == 0 { 0.75 } else { -0.4375 })
        .collect();
    let m = matrix_on_pairs(&pairs, &weights);
    let classes = property_flux_classes(&as_coeff(&m), fx4_point_group());
    assert_eq!(
        classes,
        vec![Irrep::new(phi)],
        "synthetic_m({phi}, {subset:?}): derived flux classes {classes:?}"
    );
    m
}

fn multiclass_m() -> Vec<f64> {
    let mut pairs = vec![synthetic_pairs(1)[0]];
    pairs.push(synthetic_pairs(2)[0]);
    let m = matrix_on_pairs(&pairs, &[0.75, -0.5]);
    let classes = property_flux_classes(&as_coeff(&m), fx4_point_group());
    assert_eq!(
        classes.len(),
        2,
        "multiclass_m: the table gives {classes:?}, not two classes"
    );
    m
}

fn as_coeff(m: &[f64]) -> CoeffTensor<f64> {
    assert_eq!(m.len(), D * D);
    let mut t = CoeffTensor::<f64>::zeros(D, 2);
    for p in 0..D {
        for q in 0..D {
            t.set(&[p, q], m[p * D + q]);
        }
    }
    t
}

fn prop(m: &[f64], label: &str) -> PropertyOperator<f64> {
    PropertyOperator::new(label, as_coeff(m))
}

fn concat(blocks: &[Vec<f64>]) -> Vec<f64> {
    blocks.iter().flat_map(|b| b.iter().copied()).collect()
}

fn alpha_opts() -> QnttSectorAlphaOpts {
    let mut o = QnttSectorAlphaOpts::defaults();
    o.max_iter = 500;
    o
}

fn moment_opts() -> QnttSectorMomentOpts {
    QnttSectorMomentOpts::defaults()
}

fn alpha_info_sentinel() -> QnttSectorAlphaInfo {
    QnttSectorAlphaInfo {
        alpha_defect: f64::NAN,
        sigma_min: f64::NAN,
        dim: usize::MAX,
        phi: -99,
        target: -99,
    }
}

fn moments_info_sentinel() -> QnttSectorMomentsInfo {
    QnttSectorMomentsInfo {
        orth_error: f64::NAN,
        sigma_min: f64::NAN,
        dim: usize::MAX,
        n_modes: usize::MAX,
        n_neg: usize::MAX,
        n_clusters: usize::MAX,
        metric: -99,
        has_alpha_sos: -99,
        complete_basis: -99,
        phi: -99,
        target: -99,
        reserved: -99,
    }
}

struct AlphaOut {
    rc: i32,
    alpha: Vec<f64>,
    props: Vec<f64>,
    iters: Vec<usize>,
    lens: (usize, usize, usize),
    info: QnttSectorAlphaInfo,
    refusal: i32,
}

#[allow(clippy::too_many_arguments)]
fn call_alpha(
    h_ref: *const HamHandle,
    h_out: *const HamHandle,
    x: *const VecHandle,
    phi: u8,
    m: &[f64],
    n_props: usize,
    norb: usize,
    labels: *const *const c_char,
    opts: &QnttSectorAlphaOpts,
    caps: [Option<usize>; 3],
) -> AlphaOut {
    let mut abuf = vec![f64::NAN; caps[0].unwrap_or(0)];
    let mut pbuf = vec![f64::NAN; caps[1].unwrap_or(0)];
    let mut ibuf = vec![usize::MAX; caps[2].unwrap_or(0)];
    let aptr = if caps[0].is_some() {
        abuf.as_mut_ptr()
    } else {
        ptr::null_mut()
    };
    let pptr = if caps[1].is_some() {
        pbuf.as_mut_ptr()
    } else {
        ptr::null_mut()
    };
    let iptr = if caps[2].is_some() {
        ibuf.as_mut_ptr()
    } else {
        ptr::null_mut()
    };
    let (mut la, mut lp, mut li) = (usize::MAX, usize::MAX, usize::MAX);
    let mut info = alpha_info_sentinel();
    let mut refusal = -99i32;
    let rc = unsafe {
        qntt_f64_sector_alpha(
            h_ref,
            h_out,
            x,
            phi,
            m.as_ptr(),
            m.len(),
            n_props,
            norb,
            labels,
            opts,
            aptr,
            abuf.len(),
            &mut la,
            pptr,
            pbuf.len(),
            &mut lp,
            iptr,
            ibuf.len(),
            &mut li,
            &mut info,
            &mut refusal,
        )
    };
    let (alpha, props, iters) = if rc == QNTT_OK {
        (
            abuf[..la.min(abuf.len())].to_vec(),
            pbuf[..lp.min(pbuf.len())].to_vec(),
            ibuf[..li.min(ibuf.len())].to_vec(),
        )
    } else {
        (abuf.clone(), pbuf.clone(), ibuf.clone())
    };
    AlphaOut {
        rc,
        alpha,
        props,
        iters,
        lens: (la, lp, li),
        info,
        refusal,
    }
}

fn call_alpha_sized(
    h_ref: *const HamHandle,
    h_out: *const HamHandle,
    x: *const VecHandle,
    phi: u8,
    m: &[f64],
    n_props: usize,
    opts: &QnttSectorAlphaOpts,
) -> AlphaOut {
    call_alpha(
        h_ref,
        h_out,
        x,
        phi,
        m,
        n_props,
        D,
        ptr::null(),
        opts,
        [
            Some(n_props * n_props),
            Some(n_props * 3),
            Some(n_props.max(1)),
        ],
    )
}

struct MomentsOut {
    rc: i32,
    omega: Vec<f64>,
    moments: Vec<f64>,
    oscillator: Vec<f64>,
    alpha_sos: Vec<f64>,
    props: Vec<f64>,
    unstable: Vec<f64>,
    clusters: Vec<usize>,
    lens: [usize; 7],
    info: QnttSectorMomentsInfo,
    refusal: i32,
}

#[allow(clippy::too_many_arguments)]
fn call_moments(
    h_ref: *const HamHandle,
    h_out: *const HamHandle,
    x: *const VecHandle,
    phi: u8,
    m: &[f64],
    n_props: usize,
    norb: usize,
    labels: *const *const c_char,
    opts: &QnttSectorMomentOpts,
    caps: [Option<usize>; 7],
) -> MomentsOut {
    let mut om = vec![f64::NAN; caps[0].unwrap_or(0)];
    let mut mo = vec![f64::NAN; caps[1].unwrap_or(0)];
    let mut os = vec![f64::NAN; caps[2].unwrap_or(0)];
    let mut so = vec![f64::NAN; caps[3].unwrap_or(0)];
    let mut pr = vec![f64::NAN; caps[4].unwrap_or(0)];
    let mut un = vec![f64::NAN; caps[5].unwrap_or(0)];
    let mut cl = vec![usize::MAX; caps[6].unwrap_or(0)];
    macro_rules! p {
        ($buf:expr, $i:expr) => {
            if caps[$i].is_some() {
                $buf.as_mut_ptr()
            } else {
                ptr::null_mut()
            }
        };
    }
    let (omp, mop, osp, sop, prp, unp) = (
        p!(om, 0),
        p!(mo, 1),
        p!(os, 2),
        p!(so, 3),
        p!(pr, 4),
        p!(un, 5),
    );
    let clp = p!(cl, 6);
    let mut lens = [usize::MAX; 7];
    let mut info = moments_info_sentinel();
    let mut refusal = -99i32;
    let rc = unsafe {
        qntt_f64_sector_transition_moments(
            h_ref,
            h_out,
            x,
            phi,
            m.as_ptr(),
            m.len(),
            n_props,
            norb,
            labels,
            opts,
            omp,
            om.len(),
            &mut lens[0],
            mop,
            mo.len(),
            &mut lens[1],
            osp,
            os.len(),
            &mut lens[2],
            sop,
            so.len(),
            &mut lens[3],
            prp,
            pr.len(),
            &mut lens[4],
            unp,
            un.len(),
            &mut lens[5],
            clp,
            cl.len(),
            &mut lens[6],
            &mut info,
            &mut refusal,
        )
    };
    let ok = rc == QNTT_OK;
    let cut = |b: &Vec<f64>, n: usize| {
        if ok {
            b[..n.min(b.len())].to_vec()
        } else {
            b.clone()
        }
    };
    MomentsOut {
        rc,
        omega: cut(&om, if ok { lens[0] } else { 0 }),
        moments: cut(&mo, if ok { lens[1] } else { 0 }),
        oscillator: cut(&os, if ok { lens[2] } else { 0 }),
        alpha_sos: cut(&so, if ok { lens[3] } else { 0 }),
        props: cut(&pr, if ok { lens[4] } else { 0 }),
        unstable: cut(&un, if ok { lens[5] } else { 0 }),
        clusters: if ok {
            cl[..lens[6].min(cl.len())].to_vec()
        } else {
            cl.clone()
        },
        lens,
        info,
        refusal,
    }
}

#[allow(clippy::too_many_arguments)]
fn call_moments_at_dim(
    h_ref: *const HamHandle,
    h_out: *const HamHandle,
    x: *const VecHandle,
    phi: u8,
    m: &[f64],
    n_props: usize,
    dim: usize,
    opts: &QnttSectorMomentOpts,
) -> MomentsOut {
    call_moments(
        h_ref,
        h_out,
        x,
        phi,
        m,
        n_props,
        D,
        ptr::null(),
        opts,
        [
            Some(dim),
            Some(n_props * dim),
            Some(dim),
            Some(n_props * n_props),
            Some(n_props * 5),
            Some(dim),
            Some(2 * dim),
        ],
    )
}

fn channel_dim(
    h_ref: *const HamHandle,
    h_out: *const HamHandle,
    x: *const VecHandle,
    phi: u8,
) -> usize {
    let opts = QnttSectorOpts::defaults();
    let mut dim = usize::MAX;
    let mut applies = usize::MAX;
    let rc = unsafe {
        qntt_f64_sector_rpa(
            h_ref,
            h_out,
            x,
            phi,
            0,
            &opts,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            QNTT_SECTOR_PRECOND_OFF,
            0.0,
            0x5EC7,
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
            &mut dim,
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            &mut applies,
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, QNTT_OK, "{}", last_error());
    assert_eq!(applies, 0, "the k = 0 dim query must spend no applies");
    dim
}

fn rust_moments(
    h_ref: *const HamHandle,
    h_out: *const HamHandle,
    x: *const VecHandle,
    phi: u8,
    props: &[PropertyOperator<f64>],
    opts: &SectorMomentOpts,
) -> SectorTransitionMoments {
    unsafe {
        let hr = borrow(h_ref).unwrap();
        let ho = borrow(h_out).unwrap();
        let xr = borrow(x).unwrap();
        let ctx = weingarten_operator(hr, xr);
        let sc = SectorCtx::new(&ctx, ho, Irrep::new(phi));
        sc.transition_moments_dense(props, opts)
            .expect("the oracle must not refuse here")
    }
}

fn rust_static_opts(o: &QnttSectorAlphaOpts) -> SectorStaticOpts {
    SectorStaticOpts {
        tol: o.tol,
        max_iter: o.max_iter,
        tt_grad_tol: o.tt_grad_tol,
        sym_assert_rel: if o.has_sym_assert_rel != 0 {
            Some(o.sym_assert_rel)
        } else {
            None
        },
    }
}

fn rust_moment_opts(o: &QnttSectorMomentOpts) -> SectorMomentOpts {
    SectorMomentOpts {
        tt_grad_tol: o.tt_grad_tol,
        degeneracy_tol: o.degeneracy_tol,
        psd_tol: o.psd_tol,
        ..SectorMomentOpts::default()
    }
}

#[test]
fn sector_obs_options_default_mirrors_the_rust_defaults() {
    let mut a = QnttSectorAlphaOpts {
        tol: 0.0,
        max_iter: 0,
        tt_grad_tol: 0.0,
        sym_assert_rel: -7.0,
        has_sym_assert_rel: -7,
        route: -7,
    };
    assert_eq!(
        unsafe { qntt_sector_alpha_options_default(&mut a) },
        QNTT_OK
    );
    let sa = SectorStaticOpts::default();
    assert_eq!(a.tol, sa.tol);
    assert_eq!(a.max_iter, sa.max_iter);
    assert_eq!(a.tt_grad_tol, sa.tt_grad_tol);
    assert_eq!(sa.sym_assert_rel, None);
    assert_eq!(a.has_sym_assert_rel, 0);
    assert_eq!(a.route, QNTT_PROP_ROUTE_MATERIALIZED);

    let mut m = QnttSectorMomentOpts {
        tt_grad_tol: 0.0,
        degeneracy_tol: 0.0,
        psd_tol: 0.0,
        route: -7,
        ..QnttSectorMomentOpts::defaults()
    };
    assert_eq!(
        unsafe { qntt_sector_moment_options_default(&mut m) },
        QNTT_OK
    );
    let sm = SectorMomentOpts::default();
    assert_eq!(m.tt_grad_tol, sm.tt_grad_tol);
    assert_eq!(m.degeneracy_tol, sm.degeneracy_tol);
    assert_eq!(m.psd_tol, sm.psd_tol);
    assert_eq!(m.route, QNTT_PROP_ROUTE_MATERIALIZED);

    assert_eq!(
        unsafe { qntt_sector_alpha_options_default(ptr::null_mut()) },
        QNTT_NULL_ARG
    );
    assert_eq!(
        unsafe { qntt_sector_moment_options_default(ptr::null_mut()) },
        QNTT_NULL_ARG
    );
}

#[test]
fn sector_alpha_matches_the_rust_api() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let opts = alpha_opts();

    let m0 = synthetic_m(PHI_LIVE, Some(&[0]));
    let m1 = synthetic_m(PHI_LIVE, Some(&[1]));

    for (n_props, blocks) in [
        (1usize, vec![m0.clone()]),
        (2, vec![m0.clone(), m1.clone()]),
    ] {
        let flat = concat(&blocks);
        let got = call_alpha_sized(h, h_out, x, PHI_LIVE, &flat, n_props, &opts);
        assert_eq!(got.rc, QNTT_OK, "n_props = {n_props}: {}", last_error());
        assert_eq!(got.refusal, REFUSAL_NONE);
        assert_eq!(got.lens, (n_props * n_props, n_props * 3, n_props));

        let props: Vec<PropertyOperator<f64>> = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| prop(b, &format!("p{i}")))
            .collect();
        let want = unsafe {
            let hr = borrow(h as *const _).unwrap();
            let ho = borrow(h_out as *const _).unwrap();
            let xr = borrow(x as *const _).unwrap();
            let ctx = weingarten_operator(hr, xr);
            let sc = SectorCtx::new(&ctx, ho, Irrep::new(PHI_LIVE));
            sc.polarizability_static(&props, &rust_static_opts(&opts))
                .expect("a critical reference on a stable live channel")
        };

        assert_eq!(
            got.alpha, want.alpha,
            "the ABI alpha must be the Rust alpha VERBATIM (n_props = {n_props})"
        );
        assert_eq!(got.info.alpha_defect, want.alpha_defect);
        assert_eq!(got.info.dim, want.dim);
        assert_eq!(got.info.sigma_min, want.sigma_min);
        assert_eq!(got.info.phi, i32::from(want.phi.index()));
        assert_eq!(got.info.target, i32::from(want.target.index()));
        for u in 0..n_props {
            for v in 0..n_props {
                assert_eq!(got.alpha[u * n_props + v], got.alpha[v * n_props + u]);
            }
        }
        for i in 0..n_props {
            assert_eq!(
                got.props[i * 3],
                want.capture_defect[i],
                "capture_defect[{i}]"
            );
            assert_eq!(
                got.props[i * 3 + 1],
                want.solves[i].grad_norm,
                "grad_norm[{i}]"
            );
            assert_eq!(
                got.props[i * 3 + 2],
                want.solves[i].rel_residual,
                "rel_residual[{i}]"
            );
            assert_eq!(got.iters[i], want.solves[i].iters, "iters[{i}]");
        }
        assert!(got.info.dim > 0);
        assert!(
            got.props[1] > 1e-8,
            "||p_0|| = {} is not a visible component",
            got.props[1]
        );
        assert!(
            got.iters[0] > 0,
            "an in-channel component must spend CG iterations"
        );
        println!(
            "sector_alpha n_props = {n_props}: dim {}, alpha {:?}, defect {:e}, iters {:?}",
            got.info.dim, got.alpha, got.info.alpha_defect, got.iters
        );
    }

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_transition_moments_matches_the_rust_api() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let mopts = moment_opts();
    let m0 = synthetic_m(PHI_LIVE, Some(&[0]));
    let m1 = synthetic_m(PHI_LIVE, Some(&[1]));
    let n_props = 2usize;
    let flat = concat(&[m0.clone(), m1.clone()]);
    let dim = channel_dim(h, h_out, x, PHI_LIVE);
    assert!(dim > 0, "the live channel must be live");

    let got = call_moments_at_dim(h, h_out, x, PHI_LIVE, &flat, n_props, dim, &mopts);
    assert_eq!(got.rc, QNTT_OK, "{}", last_error());
    assert_eq!(got.refusal, REFUSAL_NONE);
    assert_eq!(
        got.info.complete_basis, 1,
        "the dense route IS a complete basis"
    );
    assert_eq!(got.info.dim, dim);
    assert!(got.info.n_modes + got.info.n_neg <= got.info.dim);
    assert!(got.info.n_clusters <= got.info.n_modes);

    let props = [prop(&m0, "p0"), prop(&m1, "p1")];
    let want = rust_moments(h, h_out, x, PHI_LIVE, &props, &rust_moment_opts(&mopts));

    assert_eq!(
        got.omega, want.omega,
        "out_omega must be the Rust omega verbatim"
    );
    assert!(
        got.omega.windows(2).all(|w| w[0] <= w[1]),
        "out_omega is ASCENDING: {:?}",
        got.omega
    );
    assert_eq!(got.info.n_modes, want.omega.len());
    assert_eq!(got.info.n_neg, want.n_neg);
    assert_eq!(got.unstable, want.unstable_omega2);
    assert_eq!(got.oscillator, want.oscillator);
    assert_eq!(got.info.orth_error, want.orth_error);
    assert_eq!(got.info.sigma_min, want.sigma_min);
    assert_eq!(got.info.phi, i32::from(want.phi.index()));
    assert_eq!(got.info.target, i32::from(want.target.index()));

    let n_modes = got.info.n_modes;
    assert_eq!(got.lens[1], n_props * n_modes);
    for i in 0..n_props {
        for n in 0..n_modes {
            assert_eq!(
                got.moments[i * n_modes + n].abs(),
                want.moments[i][n].abs(),
                "|T_{n}| of component {i}"
            );
        }
    }
    for n in 0..n_modes {
        assert_eq!(
            got.moments[n] * got.moments[n_modes + n],
            want.moments[0][n] * want.moments[1][n],
            "the same-n two-component product at mode {n} is gauge-invariant and must match"
        );
    }

    assert_eq!(got.lens[4], n_props * 5);
    for i in 0..n_props {
        assert_eq!(
            got.props[i * 5],
            want.capture_defect[i],
            "capture_defect[{i}]"
        );
        assert_eq!(
            got.props[i * 5 + 1],
            want.completeness_defect[i],
            "completeness_defect[{i}]"
        );
        assert_eq!(got.props[i * 5 + 2], want.sum_rule[i].0, "sum T^2 [{i}]");
        assert_eq!(
            got.props[i * 5 + 3],
            want.sum_rule[i].1,
            "||P^phi(mu x)||^2 [{i}]"
        );
        assert_eq!(
            got.props[i * 5 + 4],
            want.sum_rule_si[i],
            "sum_rule_si[{i}]"
        );
    }

    assert_eq!(got.lens[6], 2 * got.info.n_clusters);
    assert_eq!(
        got.clusters.len(),
        2 * want.degenerate_clusters.len(),
        "one [lo,hi) pair per cluster"
    );
    let mut cursor = 0usize;
    for (c, &(lo, hi)) in want.degenerate_clusters.iter().enumerate() {
        assert_eq!((got.clusters[2 * c], got.clusters[2 * c + 1]), (lo, hi));
        assert_eq!(lo, cursor, "the clusters must partition 0..n_modes");
        assert!(hi > lo);
        cursor = hi;
    }
    assert_eq!(cursor, n_modes);

    assert_eq!(
        got.info.has_alpha_sos != 0,
        want.alpha_sos.is_some(),
        "has_alpha_sos <=> n_neg == 0"
    );
    assert_eq!(got.info.has_alpha_sos != 0, got.info.n_neg == 0);
    if let Some(sos) = want.alpha_sos.as_ref() {
        assert_eq!(
            got.alpha_sos, *sos,
            "out_alpha_sos must be the Rust sum verbatim"
        );
        let aopts = alpha_opts();
        let stat = call_alpha_sized(h, h_out, x, PHI_LIVE, &flat, n_props, &aopts);
        assert_eq!(stat.rc, QNTT_OK, "{}", last_error());
        let budget = 1e-6
            + 100.0
                * (got.props[1]
                    + got.props[6]
                    + got.info.orth_error
                    + stat.props[2]
                    + stat.props[5]);
        println!(
            "alpha_sos vs alpha: budget {budget:e} (completeness {:e}/{:e}, orth {:e}, residual \
             {:e}/{:e})",
            got.props[1], got.props[6], got.info.orth_error, stat.props[2], stat.props[5]
        );
        for k in 0..n_props * n_props {
            let d = (got.alpha_sos[k] - stat.alpha[k]).abs();
            let scale = stat.alpha[k].abs().max(1.0);
            assert!(
                d <= budget * scale,
                "the SOS reconstruction is an ALGEBRAIC identity: alpha_sos[{k}] = {} vs alpha \
                 {} (|d| = {d:e}, budget {:e})",
                got.alpha_sos[k],
                stat.alpha[k],
                budget * scale
            );
        }
    } else {
        assert!(
            got.info.n_neg > 0,
            "alpha_sos is None only when imaginary modes deplete the resolution of the identity"
        );
    }

    println!(
        "sector_transition_moments: dim {}, n_modes {}, n_neg {}, n_clusters {}, metric {}, \
         omega {:?}, f {:?}",
        got.info.dim,
        got.info.n_modes,
        got.info.n_neg,
        got.info.n_clusters,
        got.info.metric,
        got.omega,
        got.oscillator
    );

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn the_length_probe_sizes_every_moment_buffer() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let mopts = moment_opts();
    let n_props = 1usize;
    let flat = synthetic_m(PHI_LIVE, None);

    let probe = call_moments(
        h,
        h_out,
        x,
        PHI_LIVE,
        &flat,
        n_props,
        D,
        ptr::null(),
        &mopts,
        [None; 7],
    );
    assert_eq!(probe.rc, QNTT_OK, "{}", last_error());
    assert_ne!(
        probe.info.dim,
        usize::MAX,
        "out_info is written on the probe pass"
    );
    assert_ne!(probe.info.n_modes, usize::MAX);
    assert_ne!(probe.info.n_neg, usize::MAX);
    assert_ne!(probe.info.n_clusters, usize::MAX);
    assert!(probe.info.orth_error.is_finite());
    assert!(probe.info.sigma_min.is_finite());
    assert_eq!(probe.refusal, REFUSAL_NONE);
    let (n_modes, n_neg, n_clusters, dim) = (
        probe.info.n_modes,
        probe.info.n_neg,
        probe.info.n_clusters,
        probe.info.dim,
    );
    assert!(n_modes + n_neg <= dim, "{n_modes} + {n_neg} > {dim}");
    assert!(n_clusters <= n_modes);
    assert_eq!(
        probe.lens[0], n_modes,
        "the probe reports the real omega length"
    );
    assert_eq!(probe.lens[1], n_props * n_modes);
    assert_eq!(probe.lens[2], n_modes);
    assert_eq!(probe.lens[4], n_props * 5);
    assert_eq!(probe.lens[5], n_neg);
    assert_eq!(probe.lens[6], 2 * n_clusters);
    assert_eq!(
        probe.lens[3],
        if probe.info.has_alpha_sos != 0 {
            n_props * n_props
        } else {
            0
        },
        "out_alpha_sos_len is 0 exactly when the flag says nothing was computed"
    );

    let exact = call_moments(
        h,
        h_out,
        x,
        PHI_LIVE,
        &flat,
        n_props,
        D,
        ptr::null(),
        &mopts,
        [
            Some(n_modes.max(1)),
            Some((n_props * n_modes).max(1)),
            Some(n_modes.max(1)),
            Some(n_props * n_props),
            Some(n_props * 5),
            Some(n_neg.max(1)),
            Some((2 * n_clusters).max(1)),
        ],
    );
    assert_eq!(exact.rc, QNTT_OK, "{}", last_error());
    assert_eq!(
        exact.lens, probe.lens,
        "the probe's lengths ARE the answer's"
    );
    assert!(exact.omega.iter().all(|v| v.is_finite()));

    assert_eq!(dim, channel_dim(h, h_out, x, PHI_LIVE));
    let one = call_moments_at_dim(h, h_out, x, PHI_LIVE, &flat, n_props, dim, &mopts);
    assert_eq!(one.rc, QNTT_OK, "{}", last_error());
    assert_eq!(
        one.omega, exact.omega,
        "the one-solve recipe is the same answer"
    );
    assert_eq!(one.moments, exact.moments);
    assert_eq!(one.props, exact.props);
    assert_eq!(one.clusters, exact.clusters);

    if n_modes > 0 {
        let short = call_moments(
            h,
            h_out,
            x,
            PHI_LIVE,
            &flat,
            n_props,
            D,
            ptr::null(),
            &mopts,
            [
                Some(n_modes - 1),
                Some(n_props * n_modes),
                Some(n_modes),
                Some(n_props * n_props),
                Some(n_props * 5),
                Some(n_neg.max(1)),
                Some((2 * n_clusters).max(1)),
            ],
        );
        assert_eq!(short.rc, QNTT_INVALID_ARG, "{}", last_error());
        assert!(!last_error().is_empty());
        assert_eq!(
            short.lens[0], n_modes,
            "out_len IS written on the short-buffer path — that is the length protocol"
        );
        assert!(
            short.omega.iter().all(|v| v.is_nan()),
            "nothing may be written into a refused buffer"
        );
    }

    let aopts = alpha_opts();
    let aprobe = call_alpha(
        h,
        h_out,
        x,
        PHI_LIVE,
        &flat,
        n_props,
        D,
        ptr::null(),
        &aopts,
        [None; 3],
    );
    assert_eq!(aprobe.rc, QNTT_OK, "{}", last_error());
    assert_eq!(
        aprobe.lens,
        (n_props * n_props, n_props * 3, n_props),
        "every alpha length is a function of the caller's own n_props"
    );
    let ashort = call_alpha(
        h,
        h_out,
        x,
        PHI_LIVE,
        &flat,
        n_props,
        D,
        ptr::null(),
        &aopts,
        [
            Some(n_props * n_props - 1),
            Some(n_props * 3),
            Some(n_props),
        ],
    );
    assert_eq!(ashort.rc, QNTT_INVALID_ARG, "{}", last_error());
    assert!(!last_error().is_empty());
    assert_eq!(ashort.lens.0, n_props * n_props);

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn labels_default_and_round_trip_in_messages() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let opts = alpha_opts();
    let m0 = synthetic_m(PHI_LIVE, Some(&[0]));
    let m1 = synthetic_m(PHI_LIVE, Some(&[1]));
    let flat2 = concat(&[m0.clone(), m1.clone()]);

    let a = call_alpha_sized(h, h_out, x, PHI_LIVE, &flat2, 2, &opts);
    assert_eq!(a.rc, QNTT_OK, "{}", last_error());

    let named = CString::new("dipole_z").unwrap();
    let pair: Vec<*const c_char> = vec![named.as_ptr(), ptr::null()];
    let b = call_alpha(
        h,
        h_out,
        x,
        PHI_LIVE,
        &flat2,
        2,
        D,
        pair.as_ptr(),
        &opts,
        [Some(4), Some(6), Some(2)],
    );
    assert_eq!(b.rc, QNTT_OK, "{}", last_error());
    assert_eq!(b.alpha, a.alpha, "a label may not change a number");

    let bad = multiclass_m();
    let one: Vec<*const c_char> = vec![named.as_ptr()];
    let r = call_alpha(
        h,
        h_out,
        x,
        PHI_LIVE,
        &bad,
        1,
        D,
        one.as_ptr(),
        &opts,
        [Some(1), Some(3), Some(1)],
    );
    assert_eq!(r.rc, QNTT_INVALID_ARG, "{}", last_error());
    let msg = last_error();
    assert!(
        msg.contains("dipole_z"),
        "the refusal must name the component the caller named: {msg:?}"
    );

    let bytes: Vec<c_char> = vec![0x66, -0x01, 0x00];
    let utf8: Vec<*const c_char> = vec![bytes.as_ptr()];
    let e1 = call_alpha(
        h,
        h_out,
        x,
        PHI_LIVE,
        &m0,
        1,
        D,
        utf8.as_ptr(),
        &opts,
        [Some(1), Some(3), Some(1)],
    );
    assert_eq!(e1.rc, QNTT_INVALID_ARG, "{}", last_error());
    assert!(!last_error().is_empty());

    let long = CString::new("x".repeat(300)).unwrap();
    let over: Vec<*const c_char> = vec![long.as_ptr()];
    let e2 = call_alpha(
        h,
        h_out,
        x,
        PHI_LIVE,
        &m0,
        1,
        D,
        over.as_ptr(),
        &opts,
        [Some(1), Some(3), Some(1)],
    );
    assert_eq!(e2.rc, QNTT_INVALID_ARG, "{}", last_error());
    assert!(!last_error().is_empty());

    let dim = channel_dim(h, h_out, x, PHI_LIVE);
    let mopts = moment_opts();
    let mnamed = call_moments(
        h,
        h_out,
        x,
        PHI_LIVE,
        &flat2,
        2,
        D,
        pair.as_ptr(),
        &mopts,
        [
            Some(dim),
            Some(2 * dim),
            Some(dim),
            Some(4),
            Some(10),
            Some(dim),
            Some(2 * dim),
        ],
    );
    assert_eq!(mnamed.rc, QNTT_OK, "{}", last_error());
    let mbad = call_moments(
        h,
        h_out,
        x,
        PHI_LIVE,
        &bad,
        1,
        D,
        one.as_ptr(),
        &mopts,
        [
            Some(dim),
            Some(dim),
            Some(dim),
            Some(1),
            Some(5),
            Some(dim),
            Some(2 * dim),
        ],
    );
    assert_eq!(mbad.rc, QNTT_INVALID_ARG, "{}", last_error());
    assert!(last_error().contains("dipole_z"), "{}", last_error());

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn a_multiclass_matrix_is_a_caller_error() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let bad = multiclass_m();

    let a = call_alpha_sized(h, h_out, x, PHI_LIVE, &bad, 1, &alpha_opts());
    assert_eq!(a.rc, QNTT_INVALID_ARG, "{}", last_error());
    assert_eq!(
        a.refusal, REFUSAL_NONE,
        "a caller error is not a physics refusal: out_refusal stays NONE"
    );
    let msg = last_error();
    assert!(
        msg.contains("flux") && msg.contains("class"),
        "the message must name the derived classes: {msg:?}"
    );

    let dim = channel_dim(h, h_out, x, PHI_LIVE);
    let m = call_moments_at_dim(h, h_out, x, PHI_LIVE, &bad, 1, dim, &moment_opts());
    assert_eq!(m.rc, QNTT_INVALID_ARG, "{}", last_error());
    assert_eq!(m.refusal, REFUSAL_NONE);
    assert!(!last_error().is_empty());

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn an_out_of_flux_component_is_a_structural_zero_row() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let inch = synthetic_m(PHI_LIVE, Some(&[0]));
    let out = synthetic_m(1, None);
    let flat = concat(&[inch, out]);
    let got = call_alpha_sized(h, h_out, x, PHI_LIVE, &flat, 2, &alpha_opts());
    assert_eq!(
        got.rc,
        QNTT_OK,
        "an out-of-flux component is an ANSWER: {}",
        last_error()
    );
    assert_eq!(got.refusal, REFUSAL_NONE);

    for v in [got.alpha[1], got.alpha[2], got.alpha[3]] {
        assert_eq!(
            v, 0.0,
            "the out-of-flux row and column are EXACTLY zero: {:?}",
            got.alpha
        );
    }
    assert_ne!(
        got.alpha[0], 0.0,
        "anti-vacuity: the in-channel component is visible"
    );
    let col = |i: usize, k: usize| got.props[i * 3 + k];
    assert_eq!(col(1, 1), 0.0, "grad_norm of an invisible component");
    assert_eq!(col(1, 0), 1.0, "capture_defect is exactly 1 out of flux");
    assert_eq!(got.iters[1], 0, "no apply is paid on a structural zero row");
    assert!(got.iters[0] > 0);

    let dim = channel_dim(h, h_out, x, PHI_LIVE);
    let mom = call_moments_at_dim(h, h_out, x, PHI_LIVE, &flat, 2, dim, &moment_opts());
    assert_eq!(mom.rc, QNTT_OK, "{}", last_error());
    let n_modes = mom.info.n_modes;
    for n in 0..n_modes {
        assert_eq!(
            mom.moments[n_modes + n],
            0.0,
            "T_{n} of an out-of-flux component"
        );
    }
    assert_eq!(mom.props[5], 1.0, "capture_defect is exactly 1 out of flux");

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn caller_errors_never_panic() {
    let (h, x) = reference();
    let h1 = ham_at_target(1);
    let h2 = ham_at_target(2);
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let good = synthetic_m(PHI_LIVE, None);
    let opts = alpha_opts();

    let expect_invalid = |rc: i32, needle: &str, what: &str| {
        assert_eq!(rc, QNTT_INVALID_ARG, "{what}: {}", last_error());
        let msg = last_error();
        assert!(
            msg.contains(needle),
            "{what}: message {msg:?} does not mention {needle:?}"
        );
        assert!(rc != QNTT_PANIC);
    };

    expect_invalid(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &good, 0, &opts).rc,
        "n_props",
        "n_props = 0",
    );
    let four = concat(&[good.clone(), good.clone(), good.clone(), good.clone()]);
    expect_invalid(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &four, 4, &opts).rc,
        "n_props",
        "n_props = 4",
    );
    expect_invalid(
        call_alpha(
            h,
            h_out,
            x,
            PHI_LIVE,
            &good,
            1,
            3,
            ptr::null(),
            &opts,
            [Some(1), Some(3), Some(1)],
        )
        .rc,
        "norb",
        "norb = 3",
    );
    let mut off = good.clone();
    off.push(0.0);
    expect_invalid(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &off, 1, &opts).rc,
        "m_len",
        "m_len off by one",
    );
    let at = |p: usize, q: usize| p * D + q;
    let mut nan = good.clone();
    nan[at(1, 2)] = f64::NAN;
    nan[at(2, 1)] = f64::NAN;
    expect_invalid(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &nan, 1, &opts).rc,
        "finite",
        "a NaN entry",
    );
    let mut asym = vec![0.0f64; D * D];
    asym[at(1, 0)] = 1.0;
    asym[at(0, 1)] = 0.5;
    expect_invalid(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &asym, 1, &opts).rc,
        "is not symmetric",
        "an asymmetric block",
    );
    let mut half = vec![0.0f64; D * D];
    half[at(1, 0)] = 1e-20;
    expect_invalid(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &half, 1, &opts).rc,
        "one nonzero",
        "a half-triangle block",
    );

    let shifted = QnttSectorAlphaOpts {
        route: QNTT_PROP_ROUTE_SHIFTED,
        ..alpha_opts()
    };
    expect_invalid(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &good, 1, &shifted).rc,
        "MATERIALIZED",
        "route = SHIFTED (alpha)",
    );
    let mshifted = QnttSectorMomentOpts {
        route: QNTT_PROP_ROUTE_SHIFTED,
        ..moment_opts()
    };
    let dim = channel_dim(h, h_out, x, PHI_LIVE);
    expect_invalid(
        call_moments_at_dim(h, h_out, x, PHI_LIVE, &good, 1, dim, &mshifted).rc,
        "MATERIALIZED",
        "route = SHIFTED (moments)",
    );
    for bad in [
        QnttSectorAlphaOpts {
            tol: 0.0,
            ..alpha_opts()
        },
        QnttSectorAlphaOpts {
            tol: f64::NAN,
            ..alpha_opts()
        },
        QnttSectorAlphaOpts {
            max_iter: 0,
            ..alpha_opts()
        },
        QnttSectorAlphaOpts {
            tt_grad_tol: -1.0,
            ..alpha_opts()
        },
        QnttSectorAlphaOpts {
            has_sym_assert_rel: 1,
            sym_assert_rel: 0.0,
            ..alpha_opts()
        },
        QnttSectorAlphaOpts {
            route: 7,
            ..alpha_opts()
        },
    ] {
        let got = call_alpha_sized(h, h_out, x, PHI_LIVE, &good, 1, &bad);
        assert_eq!(got.rc, QNTT_INVALID_ARG, "{}", last_error());
        assert!(!last_error().is_empty());
    }
    let escape = QnttSectorAlphaOpts {
        tt_grad_tol: f64::INFINITY,
        ..alpha_opts()
    };
    assert_eq!(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &good, 1, &escape).rc,
        QNTT_OK,
        "{}",
        last_error()
    );

    let two = 2.0f64;
    let mut x2: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_scaled(x, &two, &mut x2) },
        QNTT_OK,
        "{}",
        last_error()
    );
    expect_invalid(
        call_alpha_sized(h, h_out, x2, PHI_LIVE, &good, 1, &opts).rc,
        "normal",
        "a non-unit reference (alpha)",
    );
    expect_invalid(
        call_moments_at_dim(h, h_out, x2, PHI_LIVE, &good, 1, dim, &moment_opts()).rc,
        "normal",
        "a non-unit reference (moments)",
    );
    unsafe { qntt_f64_vec_free(x2) };

    let m1 = synthetic_m(1, None);
    expect_invalid(
        call_alpha_sized(h, h1, x, 0, &m1, 1, &opts).rc,
        "rpa_signed",
        "phi = 0",
    );
    expect_invalid(
        call_alpha_sized(h, h1, x, 7, &m1, 1, &opts).rc,
        "point group",
        "phi out of range",
    );
    expect_invalid(
        call_alpha_sized(h, h, x, 1, &m1, 1, &opts).rc,
        "BUILT",
        "h_out = h_ref",
    );
    expect_invalid(
        call_alpha_sized(h, h2, x, 1, &m1, 1, &opts).rc,
        "BUILT",
        "h_out at the wrong target",
    );
    expect_invalid(
        call_alpha_sized(h1, h2, x, 1, &m1, 1, &opts).rc,
        "h_ref",
        "h_ref off x's space",
    );
    expect_invalid(
        call_moments_at_dim(h, h1, x, 0, &m1, 1, 1, &moment_opts()).rc,
        "rpa_signed",
        "phi = 0 (moments)",
    );
    expect_invalid(
        call_moments_at_dim(h, h, x, 1, &m1, 1, 1, &moment_opts()).rc,
        "BUILT",
        "h_out = h_ref (moments)",
    );

    let null_h = call_alpha_sized(ptr::null(), h_out, x, PHI_LIVE, &good, 1, &opts);
    assert_eq!(null_h.rc, QNTT_NULL_ARG);
    assert_eq!(
        null_h.refusal, REFUSAL_NONE,
        "a null handle is QNTT_NULL_ARG, and out_refusal is still written: NONE"
    );
    let mut info = alpha_info_sentinel();
    let mut refusal = -99i32;
    let (mut la, mut lp, mut li) = (0usize, 0usize, 0usize);
    assert_eq!(
        unsafe {
            qntt_f64_sector_alpha(
                h,
                h_out,
                x,
                PHI_LIVE,
                good.as_ptr(),
                good.len(),
                1,
                D,
                ptr::null(),
                ptr::null(),
                ptr::null_mut(),
                0,
                &mut la,
                ptr::null_mut(),
                0,
                &mut lp,
                ptr::null_mut(),
                0,
                &mut li,
                &mut info,
                &mut refusal,
            )
        },
        QNTT_NULL_ARG,
        "a null options pointer"
    );
    assert_eq!(
        refusal, REFUSAL_NONE,
        "a null opts is QNTT_NULL_ARG, and out_refusal is still written: NONE"
    );
    refusal = -99;
    assert_eq!(
        unsafe {
            qntt_f64_sector_alpha(
                h,
                h_out,
                x,
                PHI_LIVE,
                good.as_ptr(),
                good.len(),
                1,
                D,
                ptr::null(),
                &opts,
                ptr::null_mut(),
                0,
                &mut la,
                ptr::null_mut(),
                0,
                &mut lp,
                ptr::null_mut(),
                0,
                &mut li,
                ptr::null_mut(),
                &mut refusal,
            )
        },
        QNTT_NULL_ARG,
        "out_info is required"
    );
    assert_eq!(
        refusal, REFUSAL_NONE,
        "a null out_info is QNTT_NULL_ARG, and out_refusal is written FIRST: NONE"
    );
    assert_eq!(
        unsafe {
            qntt_f64_sector_alpha(
                h,
                h_out,
                x,
                PHI_LIVE,
                good.as_ptr(),
                good.len(),
                1,
                D,
                ptr::null(),
                &opts,
                ptr::null_mut(),
                0,
                &mut la,
                ptr::null_mut(),
                0,
                &mut lp,
                ptr::null_mut(),
                0,
                &mut li,
                &mut info,
                ptr::null_mut(),
            )
        },
        QNTT_NULL_ARG,
        "out_refusal is required"
    );

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_ham_free(h2);
        qntt_f64_ham_free(h1);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn an_empty_channel_answers_both_observables() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_EMPTY));
    assert_eq!(
        channel_dim(h, h_out, x, PHI_EMPTY),
        0,
        "fixture premise: phi = {PHI_EMPTY} is structurally EMPTY on fx4 (2,1)"
    );
    let m = synthetic_m(PHI_EMPTY, None);

    let a = call_alpha_sized(h, h_out, x, PHI_EMPTY, &m, 1, &alpha_opts());
    assert_eq!(a.rc, QNTT_OK, "{}", last_error());
    assert_eq!(a.refusal, REFUSAL_NONE);
    assert_eq!(a.info.dim, 0);
    assert_eq!(a.alpha, vec![0.0], "an empty channel's alpha is all zeros");
    assert_eq!(a.iters, vec![0], "no apply is spent on an empty channel");
    assert_eq!(
        a.props[0], 1.0,
        "capture_defect is exactly 1 on an empty channel"
    );
    assert_eq!(
        a.props[1], 0.0,
        "grad_norm is 0: the covector is the zero vector"
    );

    let mo = call_moments(
        h,
        h_out,
        x,
        PHI_EMPTY,
        &m,
        1,
        D,
        ptr::null(),
        &moment_opts(),
        [
            Some(1),
            Some(1),
            Some(1),
            Some(1),
            Some(5),
            Some(1),
            Some(2),
        ],
    );
    assert_eq!(mo.rc, QNTT_OK, "{}", last_error());
    assert_eq!(mo.refusal, REFUSAL_NONE);
    assert_eq!(mo.info.dim, 0);
    assert_eq!(
        (mo.info.n_modes, mo.info.n_neg, mo.info.n_clusters),
        (0, 0, 0)
    );
    assert_eq!(
        (mo.lens[0], mo.lens[1], mo.lens[2], mo.lens[5], mo.lens[6]),
        (0, 0, 0, 0, 0),
        "every mode-indexed length is 0"
    );
    assert_eq!(
        mo.info.metric, METRIC_AUTO,
        "no metric is resolved on an empty channel — AUTO, never a placeholder Rr"
    );
    assert_ne!(
        mo.info.has_alpha_sos, 0,
        "an empty sum is a sum: the flag is set"
    );
    assert_eq!(
        mo.alpha_sos,
        vec![0.0],
        "and the tensor it wrote is all zeros"
    );
    assert_eq!(
        mo.props[0], 1.0,
        "capture_defect is exactly 1 on an empty channel"
    );

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn a_refusal_writes_a_code_and_no_buffer() {
    let (h, x) = reference();
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let m = synthetic_m(PHI_LIVE, None);
    let dim = channel_dim(h, h_out, x, PHI_LIVE);

    let tight = QnttSectorAlphaOpts {
        tt_grad_tol: 1e-300,
        ..alpha_opts()
    };
    let a = call_alpha_sized(h, h_out, x, PHI_LIVE, &m, 1, &tight);
    assert_eq!(a.rc, QNTT_NOT_CONVERGED, "{}", last_error());
    assert_eq!(a.refusal, REFUSAL_NOT_TT_CRITICAL);
    let msg = last_error();
    assert!(msg.contains("not TT-critical"), "{msg:?}");
    assert!(
        a.alpha.iter().all(|v| v.is_nan()) && a.props.iter().all(|v| v.is_nan()),
        "no out-buffer may be written on a refusal: {:?} {:?}",
        a.alpha,
        a.props
    );
    assert!(a.iters.iter().all(|&v| v == usize::MAX));
    assert_eq!(
        a.lens,
        (usize::MAX, usize::MAX, usize::MAX),
        "out_len is not written on a refusal either — there is no length to report"
    );
    assert_eq!(
        a.info.dim,
        usize::MAX,
        "out_info is left untouched: no half-written diagnostics"
    );
    assert!(a.info.sigma_min.is_nan() && a.info.alpha_defect.is_nan());
    assert_eq!((a.info.phi, a.info.target), (-99, -99));

    let mtight = QnttSectorMomentOpts {
        tt_grad_tol: 1e-300,
        ..moment_opts()
    };
    let mo = call_moments_at_dim(h, h_out, x, PHI_LIVE, &m, 1, dim, &mtight);
    assert_eq!(mo.rc, QNTT_NOT_CONVERGED, "{}", last_error());
    assert_eq!(mo.refusal, REFUSAL_NOT_TT_CRITICAL);
    assert!(last_error().contains("not TT-critical"), "{}", last_error());
    assert!(mo.omega.iter().all(|v| v.is_nan()));
    assert!(mo.moments.iter().all(|v| v.is_nan()));
    assert!(mo.oscillator.iter().all(|v| v.is_nan()));
    assert!(mo.alpha_sos.iter().all(|v| v.is_nan()));
    assert!(mo.props.iter().all(|v| v.is_nan()));
    assert!(mo.clusters.iter().all(|&v| v == usize::MAX));
    assert_eq!(mo.lens, [usize::MAX; 7]);
    assert_eq!(mo.info.dim, usize::MAX);
    assert_eq!(mo.info.metric, -99);

    let escape = QnttSectorAlphaOpts {
        tt_grad_tol: f64::INFINITY,
        ..alpha_opts()
    };
    assert_eq!(
        call_alpha_sized(h, h_out, x, PHI_LIVE, &m, 1, &escape).rc,
        QNTT_OK,
        "{}",
        last_error()
    );

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn an_unstable_channel_refuses_both_observables() {
    let h = ham_at_target(0);
    let ob = orbsym();
    let flat = capped_ranks_pg(0, 1);
    let mut x: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn_pg(
                D,
                NUP,
                NDN,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                flat.as_ptr(),
                flat.len(),
                7,
                &mut x,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut nrm = 0.0f64;
    assert_eq!(
        unsafe { qntt_f64_vec_norm(x, &mut nrm) },
        QNTT_OK,
        "{}",
        last_error()
    );
    let inv = 1.0 / nrm;
    let mut xn: *mut _ = ptr::null_mut();
    assert_eq!(
        unsafe { qntt_f64_vec_scaled(x, &inv, &mut xn) },
        QNTT_OK,
        "{}",
        last_error()
    );
    unsafe { qntt_f64_vec_free(x) };
    let x = xn;
    let h_out = ham_at_target(i32::from(PHI_LIVE));
    let dim = channel_dim(h, h_out, x, PHI_LIVE);
    assert!(dim > 0);

    let mut w2 = vec![f64::NAN; dim];
    let (mut len, mut ddim) = (usize::MAX, usize::MAX);
    let (mut stable, mut metric) = (-99i32, -99i32);
    let rc = unsafe {
        qntt_f64_sector_rpa_dense(
            h,
            h_out,
            x,
            PHI_LIVE,
            w2.as_mut_ptr(),
            w2.len(),
            &mut len,
            &mut ddim,
            &mut stable,
            &mut metric,
        )
    };
    println!("unstable fixture: rc {rc}, stable {stable}, metric {metric}, dim {ddim}");
    assert_eq!(
        stable, -1,
        "this fixture was chosen for its indefinite pair (measured: fx4 (2,1), rank cap 1, seed \
         7 -> Unstable); it now reports arm {stable}, so the SYMPLECTIC_UNSTABLE path is no \
         longer covered"
    );

    let m = synthetic_m(PHI_LIVE, None);
    let mopts = QnttSectorMomentOpts {
        tt_grad_tol: f64::INFINITY,
        ..moment_opts()
    };
    let mo = call_moments_at_dim(h, h_out, x, PHI_LIVE, &m, 1, dim, &mopts);
    assert_eq!(mo.rc, QNTT_NOT_CONVERGED, "{}", last_error());
    assert_eq!(mo.refusal, REFUSAL_SYMPLECTIC_UNSTABLE);
    let msg = last_error();
    assert!(msg.contains("no real symplectic basis"), "{msg:?}");
    assert!(!msg.contains("NaN"), "no NaN in the refusal: {msg:?}");
    assert!(mo.omega.iter().all(|v| v.is_nan()), "nothing is published");
    assert_eq!(mo.lens, [usize::MAX; 7]);

    let aopts = QnttSectorAlphaOpts {
        tt_grad_tol: f64::INFINITY,
        ..alpha_opts()
    };
    let a = call_alpha_sized(h, h_out, x, PHI_LIVE, &m, 1, &aopts);
    assert_eq!(a.rc, QNTT_NOT_CONVERGED, "{}", last_error());
    println!(
        "  sector_alpha refusal code on the same point: {}",
        a.refusal
    );
    assert!(
        [REFUSAL_HESSIAN_INDEFINITE, REFUSAL_SOLVE_NOT_CONVERGED].contains(&a.refusal),
        "the static solve on an indefinite H^rr is HESSIAN_INDEFINITE (PCG met negative \
         curvature) or, if the covector misses every negative direction, SOLVE_NOT_CONVERGED; \
         got {}",
        a.refusal
    );
    assert!(a.alpha.iter().all(|v| v.is_nan()), "nothing is published");
    assert_eq!(a.info.dim, usize::MAX);

    assert_ne!(a.refusal, REFUSAL_UNCERTIFIED_NEGATIVE);
    assert_ne!(mo.refusal, REFUSAL_UNCERTIFIED_NEGATIVE);

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn the_resonant_arm_has_no_fixture_in_the_tree() {
    let (h, x) = reference();
    let mut verdicts = Vec::new();
    for phi in 1u8..4 {
        let h_out = ham_at_target(i32::from(phi));
        let dim = channel_dim(h, h_out, x, phi);
        let mut w2 = vec![f64::NAN; dim.max(1)];
        let (mut len, mut ddim) = (usize::MAX, usize::MAX);
        let (mut stable, mut metric) = (-99i32, -99i32);
        unsafe {
            qntt_f64_sector_rpa_dense(
                h,
                h_out,
                x,
                phi,
                w2.as_mut_ptr(),
                w2.len(),
                &mut len,
                &mut ddim,
                &mut stable,
                &mut metric,
            )
        };
        verdicts.push((phi, dim, stable));
        unsafe { qntt_f64_ham_free(h_out) };
    }
    println!("fx4 (2,1) channel verdicts (phi, dim, out_stable): {verdicts:?}");
    assert!(
        verdicts.iter().all(|&(_, _, stable)| stable != 0),
        "a RESONANT channel has appeared ({verdicts:?}): the resonant contract in this arm's doc \
         is now testable and must be written as real arms — this pin is not coverage"
    );
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

fn header_struct_fields(hdr: &str, name: &str) -> Vec<(String, String)> {
    let close = format!("}} {name};");
    let end = hdr
        .find(&close)
        .unwrap_or_else(|| panic!("qntt.h declares no `{close}`"));
    let head = hdr[..end]
        .rfind("typedef struct {")
        .expect("a typedef struct opener before the closer");
    let body = &hdr[head + "typedef struct {".len()..end];
    let mut clean = String::new();
    let mut rest = body;
    while let Some(i) = rest.find("/*") {
        clean.push_str(&rest[..i]);
        let j = rest[i..].find("*/").expect("an unterminated comment") + i + 2;
        clean.push(' ');
        rest = &rest[j..];
    }
    clean.push_str(rest);
    clean
        .split(';')
        .filter_map(|decl| {
            let toks: Vec<&str> = decl.split_whitespace().collect();
            if toks.len() < 2 {
                return None;
            }
            Some((toks[0].to_string(), toks[toks.len() - 1].to_string()))
        })
        .collect()
}

fn c_size_align(ctype: &str) -> (usize, usize) {
    match ctype {
        "double" | "size_t" | "uint64_t" => (8, 8),
        "int32_t" => (4, 4),
        other => panic!("unexpected C type {other:?} in a sector-obs struct"),
    }
}

fn c_layout(fields: &[(String, String)]) -> (Vec<usize>, usize, usize) {
    let mut off = Vec::new();
    let (mut cur, mut align, mut sum) = (0usize, 1usize, 0usize);
    for (ty, _) in fields {
        let (sz, al) = c_size_align(ty);
        cur = cur.next_multiple_of(al);
        off.push(cur);
        cur += sz;
        sum += sz;
        align = align.max(al);
    }
    (off, cur.next_multiple_of(align), sum)
}

fn check_layout(
    hdr: &str,
    name: &str,
    rust_size: usize,
    rust_align: usize,
    rust: &[(&str, usize)],
) {
    let fields = header_struct_fields(hdr, name);
    let (offsets, size, sum) = c_layout(&fields);
    assert_eq!(rust_align, 8, "{name}: the Rust mirror must be 8-aligned");
    assert_eq!(
        size, rust_size,
        "{name}: the header lays out {size} bytes, the Rust struct {rust_size} ({fields:?})"
    );
    assert_eq!(
        size,
        sum,
        "{name}: {} bytes of IMPLICIT PADDING — F-spec §2.7 requires none (eight-byte fields \
         first, then an EVEN number of int32_t); add an explicit `reserved` field",
        size - sum
    );
    let declared: Vec<&str> = fields.iter().map(|(_, f)| f.as_str()).collect();
    let expected: Vec<&str> = rust.iter().map(|&(f, _)| f).collect();
    let named: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|f| !f.starts_with("reserved"))
        .collect();
    assert_eq!(
        named, expected,
        "{name}: the header's field order is {declared:?}, the contract's {expected:?}"
    );
    for (field, rust_off) in rust {
        let i = declared
            .iter()
            .position(|d| d == field)
            .unwrap_or_else(|| panic!("{name}: no field {field:?} in {declared:?}"));
        assert_eq!(
            offsets[i], *rust_off,
            "{name}.{field}: header offset {} vs Rust {rust_off}",
            offsets[i]
        );
    }
}

#[test]
fn the_header_and_the_rust_structs_agree_on_layout() {
    let hdr = include_str!("../../include/qntt.h");
    check_layout(
        hdr,
        "QnttSectorOpts",
        size_of::<QnttSectorOpts>(),
        align_of::<QnttSectorOpts>(),
        &[
            ("max_iter", offset_of!(QnttSectorOpts, max_iter)),
            ("tol", offset_of!(QnttSectorOpts, tol)),
            ("max_subspace", offset_of!(QnttSectorOpts, max_subspace)),
            ("readout", offset_of!(QnttSectorOpts, readout)),
            ("metric", offset_of!(QnttSectorOpts, metric)),
            ("kernel", offset_of!(QnttSectorOpts, kernel)),
        ],
    );
    check_layout(
        hdr,
        "QnttSectorAlphaOpts",
        size_of::<QnttSectorAlphaOpts>(),
        align_of::<QnttSectorAlphaOpts>(),
        &[
            ("tol", offset_of!(QnttSectorAlphaOpts, tol)),
            ("max_iter", offset_of!(QnttSectorAlphaOpts, max_iter)),
            ("tt_grad_tol", offset_of!(QnttSectorAlphaOpts, tt_grad_tol)),
            (
                "sym_assert_rel",
                offset_of!(QnttSectorAlphaOpts, sym_assert_rel),
            ),
            (
                "has_sym_assert_rel",
                offset_of!(QnttSectorAlphaOpts, has_sym_assert_rel),
            ),
            ("route", offset_of!(QnttSectorAlphaOpts, route)),
        ],
    );
    check_layout(
        hdr,
        "QnttSectorMomentOpts",
        size_of::<QnttSectorMomentOpts>(),
        align_of::<QnttSectorMomentOpts>(),
        &[
            ("tt_grad_tol", offset_of!(QnttSectorMomentOpts, tt_grad_tol)),
            (
                "degeneracy_tol",
                offset_of!(QnttSectorMomentOpts, degeneracy_tol),
            ),
            ("psd_tol", offset_of!(QnttSectorMomentOpts, psd_tol)),
            ("route", offset_of!(QnttSectorMomentOpts, route)),
        ],
    );
    check_layout(
        hdr,
        "QnttSectorAlphaInfo",
        size_of::<QnttSectorAlphaInfo>(),
        align_of::<QnttSectorAlphaInfo>(),
        &[
            (
                "alpha_defect",
                offset_of!(QnttSectorAlphaInfo, alpha_defect),
            ),
            ("sigma_min", offset_of!(QnttSectorAlphaInfo, sigma_min)),
            ("dim", offset_of!(QnttSectorAlphaInfo, dim)),
            ("phi", offset_of!(QnttSectorAlphaInfo, phi)),
            ("target", offset_of!(QnttSectorAlphaInfo, target)),
        ],
    );
    check_layout(
        hdr,
        "QnttSectorMomentsInfo",
        size_of::<QnttSectorMomentsInfo>(),
        align_of::<QnttSectorMomentsInfo>(),
        &[
            ("orth_error", offset_of!(QnttSectorMomentsInfo, orth_error)),
            ("sigma_min", offset_of!(QnttSectorMomentsInfo, sigma_min)),
            ("dim", offset_of!(QnttSectorMomentsInfo, dim)),
            ("n_modes", offset_of!(QnttSectorMomentsInfo, n_modes)),
            ("n_neg", offset_of!(QnttSectorMomentsInfo, n_neg)),
            ("n_clusters", offset_of!(QnttSectorMomentsInfo, n_clusters)),
            ("metric", offset_of!(QnttSectorMomentsInfo, metric)),
            (
                "has_alpha_sos",
                offset_of!(QnttSectorMomentsInfo, has_alpha_sos),
            ),
            (
                "complete_basis",
                offset_of!(QnttSectorMomentsInfo, complete_basis),
            ),
            ("phi", offset_of!(QnttSectorMomentsInfo, phi)),
            ("target", offset_of!(QnttSectorMomentsInfo, target)),
        ],
    );
}

#[test]
fn the_header_declares_the_sector_obs_family() {
    let hdr = include_str!("../../include/qntt.h");
    let i = hdr
        .find("#define QNTT_DECLARE_SECTOR_OBS(SFX)")
        .expect("qntt.h defines no QNTT_DECLARE_SECTOR_OBS(SFX)");
    let mut body = String::new();
    for line in hdr[i..].lines() {
        body.push_str(line);
        body.push('\n');
        if !line.trim_end().ends_with('\\') {
            break;
        }
    }
    for sym in [
        "qntt_##SFX##_sector_alpha",
        "qntt_##SFX##_sector_transition_moments",
    ] {
        assert!(
            body.contains(sym),
            "the macro body declares no {sym}: {body}"
        );
    }
    let invocations: Vec<&str> = hdr
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("QNTT_DECLARE_SECTOR_OBS("))
        .collect();
    assert_eq!(
        invocations,
        vec!["QNTT_DECLARE_SECTOR_OBS(f64)"],
        "the cross-sector observable family is f64-only today"
    );
    for sym in [
        "qntt_sector_alpha_options_default",
        "qntt_sector_moment_options_default",
    ] {
        assert!(hdr.contains(sym), "qntt.h declares no {sym}");
    }
}

#[test]
fn the_header_defines_match_the_wire_codes() {
    let hdr = include_str!("../../include/qntt.h");
    let define = |name: &str| -> i32 {
        let pat = format!("#define {name} ");
        let i = hdr
            .find(&pat)
            .unwrap_or_else(|| panic!("qntt.h defines no {name}"));
        hdr[i + pat.len()..]
            .split_whitespace()
            .next()
            .expect("a value")
            .parse()
            .expect("an integer literal")
    };
    assert_eq!(define("QNTT_SECTOR_REFUSAL_NONE"), REFUSAL_NONE);
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_NOT_TT_CRITICAL"),
        REFUSAL_NOT_TT_CRITICAL
    );
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_HESSIAN_INDEFINITE"),
        REFUSAL_HESSIAN_INDEFINITE
    );
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_SYMPLECTIC_UNSTABLE"),
        REFUSAL_SYMPLECTIC_UNSTABLE
    );
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_SOLVE_NOT_CONVERGED"),
        REFUSAL_SOLVE_NOT_CONVERGED
    );
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_UNCERTIFIED_NEGATIVE"),
        REFUSAL_UNCERTIFIED_NEGATIVE
    );
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_NEGATIVE_OMEGA"),
        REFUSAL_NEGATIVE_OMEGA
    );
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_NEAR_RESONANCE"),
        REFUSAL_NEAR_RESONANCE
    );
    assert_eq!(
        define("QNTT_SECTOR_REFUSAL_INDEFINITE_ABOVE_POLE"),
        REFUSAL_INDEFINITE_ABOVE_POLE
    );
    let mut declared: Vec<i32> = hdr
        .lines()
        .filter_map(|l| l.strip_prefix("#define QNTT_SECTOR_REFUSAL_"))
        .map(|rest| {
            rest.split_whitespace()
                .nth(1)
                .expect("a value")
                .parse()
                .expect("an integer literal")
        })
        .collect();
    declared.sort_unstable();
    assert_eq!(
        declared,
        vec![0, 1, 2, 3, 4, 5, 7, 8, 9],
        "the wire codes are frozen: nine names, 6 reserved and unnamed"
    );
    assert_eq!(define("QNTT_PROP_ROUTE_SHIFTED"), QNTT_PROP_ROUTE_SHIFTED);
    assert_eq!(
        define("QNTT_PROP_ROUTE_MATERIALIZED"),
        QNTT_PROP_ROUTE_MATERIALIZED
    );
    assert_eq!(
        (QNTT_PROP_ROUTE_SHIFTED, QNTT_PROP_ROUTE_MATERIALIZED),
        (0, 1),
        "PropRoute's own #[default] is Shifted, which is why it is named on the wire and REFUSED"
    );
    assert_eq!(define("QNTT_CURVATURE_REG_OFF"), QNTT_CURVATURE_REG_OFF);
    assert_eq!(define("QNTT_CURVATURE_REG_HARD"), QNTT_CURVATURE_REG_HARD);
    assert_eq!(
        define("QNTT_CURVATURE_REG_TIKHONOV"),
        QNTT_CURVATURE_REG_TIKHONOV
    );
    assert_eq!(define("QNTT_RPA_KERNEL_PAIRED"), QNTT_RPA_KERNEL_PAIRED);
    assert_eq!(define("QNTT_RPA_KERNEL_SIGNED"), QNTT_RPA_KERNEL_SIGNED);
    assert_eq!((QNTT_RPA_KERNEL_PAIRED, QNTT_RPA_KERNEL_SIGNED), (0, 1));
    assert_eq!(define("QNTT_SECTOR_PRECOND_OFF"), QNTT_SECTOR_PRECOND_OFF);
    assert_eq!(define("QNTT_SECTOR_PRECOND_D0"), QNTT_SECTOR_PRECOND_D0);
    assert_eq!(define("QNTT_SECTOR_PRECOND_AUTO"), QNTT_SECTOR_PRECOND_AUTO);
    assert_eq!(
        (
            QNTT_SECTOR_PRECOND_OFF,
            QNTT_SECTOR_PRECOND_D0,
            QNTT_SECTOR_PRECOND_AUTO
        ),
        (0, 1, 2)
    );
}
