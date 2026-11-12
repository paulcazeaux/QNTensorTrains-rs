// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::pointgroup::Irrep;
use qntt::solvers::hess::{
    CurvatureReg, RpaKernel, RpaMetric, RpaReport, SectorCtx, SectorOpts, SectorReadout,
    SectorResponse, SympPrecond, SymplecticSpectrum, weingarten_operator,
};
use qntt_capi::err::{QNTT_NOT_CONVERGED, QNTT_NULL_ARG};
use qntt_capi::handle::borrow;
use qntt_capi::rpa::*;

type HamH = qntt_capi::handle::Handle<SparseHamiltonian<f64>>;
type VecH = qntt_capi::handle::Handle<qntt::TtVector<f64>>;

struct SectorOut {
    rc: i32,
    vals: Vec<f64>,
    fw: Vec<f64>,
    residual: Vec<f64>,
    krein: Vec<f64>,
    complex: Vec<f64>,
    len: usize,
    dim: usize,
    metric: i32,
    converged: i32,
    iters: usize,
    applies: usize,
    certified: i32,
    report: Vec<i64>,
}

impl SectorOut {
    fn indefinite(&self) -> bool {
        self.report.get(2).is_some_and(|&e| e > 0)
    }
}

fn n_max(opts: &QnttSectorOpts, k: usize) -> usize {
    opts.max_iter.max(k + 2)
}

#[allow(clippy::too_many_arguments)]
fn call_sector_with(
    h_ref: *const HamH,
    h_out: *const HamH,
    x: *const VecH,
    phi: u8,
    k: usize,
    opts: &QnttSectorOpts,
    curvature_kind: i32,
    curvature_value: f64,
    precond: i32,
    precond_kappa: f64,
    seed: u64,
) -> SectorOut {
    let nm = n_max(opts, k);
    let (mut vals, mut fw, mut res) = (vec![f64::NAN; nm], vec![f64::NAN; nm], vec![f64::NAN; nm]);
    let (mut krein, mut cx, mut report) =
        (vec![f64::NAN; nm], vec![f64::NAN; 3 * nm], vec![-99i64; 5]);
    let (mut len, mut fwlen, mut reslen, mut kreinlen, mut cxlen, mut replen) = (
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
        usize::MAX,
    );
    let mut dim = usize::MAX;
    let (mut metric, mut converged, mut certified) = (-99i32, -99i32, -99i32);
    let (mut iters, mut applies) = (usize::MAX, usize::MAX);
    let rc = unsafe {
        qntt_f64_sector_rpa(
            h_ref,
            h_out,
            x,
            phi,
            k,
            opts,
            curvature_kind,
            curvature_value,
            precond,
            precond_kappa,
            seed,
            vals.as_mut_ptr(),
            vals.len(),
            &mut len,
            fw.as_mut_ptr(),
            fw.len(),
            &mut fwlen,
            res.as_mut_ptr(),
            res.len(),
            &mut reslen,
            krein.as_mut_ptr(),
            krein.len(),
            &mut kreinlen,
            cx.as_mut_ptr(),
            cx.len(),
            &mut cxlen,
            &mut dim,
            &mut metric,
            &mut converged,
            &mut iters,
            &mut applies,
            &mut certified,
            report.as_mut_ptr(),
            report.len(),
            &mut replen,
        )
    };
    let ok = rc == QNTT_OK;
    let cut = |n: usize, cap: usize| {
        if !ok || n == usize::MAX {
            0
        } else {
            n.min(cap)
        }
    };
    if ok {
        assert!(len <= nm, "out_len = {len} exceeds the n_max bound {nm}");
        assert!(cxlen <= 3 * nm, "out_cx_len = {cxlen} exceeds 3·n_max");
        if opts.readout == READOUT_TDA {
            assert!(len <= k, "TDA out_len = {len} exceeds the cap k = {k}");
        }
    }
    SectorOut {
        rc,
        vals: vals[..cut(len, nm)].to_vec(),
        fw: fw[..cut(fwlen, nm)].to_vec(),
        residual: res[..cut(reslen, nm)].to_vec(),
        krein: krein[..cut(kreinlen, nm)].to_vec(),
        complex: cx[..cut(cxlen, 3 * nm)].to_vec(),
        len,
        dim,
        metric,
        converged,
        iters,
        applies,
        certified,
        report: report[..cut(replen, 5)].to_vec(),
    }
}

fn call_sector(
    h_ref: *const HamH,
    h_out: *const HamH,
    x: *const VecH,
    phi: u8,
    k: usize,
    opts: &QnttSectorOpts,
    seed: u64,
) -> SectorOut {
    call_sector_with(
        h_ref,
        h_out,
        x,
        phi,
        k,
        opts,
        QNTT_CURVATURE_REG_OFF,
        0.0,
        QNTT_SECTOR_PRECOND_OFF,
        0.0,
        seed,
    )
}

fn channel_dims(h_ref: *const HamH, x: *const VecH) -> Vec<(u8, usize)> {
    let opts = QnttSectorOpts::defaults();
    (1u8..4)
        .map(|phi| {
            let h_out = ham_at_target(i32::from(phi));
            let got = call_sector(h_ref, h_out, x, phi, 1, &opts, 0x5EC7);
            unsafe { qntt_f64_ham_free(h_out) };
            assert_eq!(got.rc, QNTT_OK, "phi = {phi}: {}", last_error());
            (phi, got.dim)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn rust_sector(
    h_ref: *const HamH,
    h_out: *const HamH,
    x: *const VecH,
    phi: u8,
    k: usize,
    opts: &QnttSectorOpts,
    reg: CurvatureReg,
    precond: SympPrecond,
    seed: u64,
) -> (SectorResponse, Vec<f64>) {
    let (hr, ho, xr) = unsafe {
        (
            borrow(h_ref).unwrap(),
            borrow(h_out).unwrap(),
            borrow(x).unwrap(),
        )
    };
    let ctx = weingarten_operator(hr, xr).with_curvature(reg);
    let sc = SectorCtx::new(&ctx, ho, Irrep::new(phi));
    let (core, requested) = opts.to_core(seed).expect("valid option codes");
    let core = SectorOpts { precond, ..core };
    match core.readout {
        SectorReadout::Tda => (sc.sector_lowest_tda(k, &core), Vec::new()),
        SectorReadout::Rpa => {
            let (resp, vecs) = sc.sector_lowest_vectors(k, &core);
            let readout = resp.rpa().expect("an Rpa readout");
            let lane = match &readout.report {
                RpaReport::Signed { metric, .. } => *metric,
                RpaReport::Paired { .. } if requested == RpaMetric::Rr => RpaMetric::Rr,
                RpaReport::Paired { .. } => RpaMetric::Ii,
            };
            let n2 = readout.omega2().len();
            let fw = vecs
                .iter()
                .take(n2)
                .map(|m| {
                    let v = &m.metric_side(lane);
                    let qa = v.dot(&sc.a_apply(v));
                    let qb = v.dot(&sc.b_apply(v));
                    if qa == 0.0 {
                        f64::INFINITY
                    } else {
                        qb.abs() / qa.abs()
                    }
                })
                .collect();
            (resp, fw)
        }
    }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn assert_abi_is_rust(
    out: &SectorOut,
    resp: &SectorResponse,
    fw: &[f64],
    requested: i32,
    what: &str,
) {
    use qntt::solvers::hess::{RpaModeKind, RpaReport, SectorSpectrum};
    assert_eq!(out.rc, QNTT_OK, "{what}: {}", last_error());
    type Want = (Vec<f64>, i32, Vec<f64>, Vec<f64>, Vec<f64>, Vec<i64>);
    let (vals, metric, residual, krein, cx, report): Want = match &resp.spectrum {
        SectorSpectrum::Tda { theta, .. } => (
            theta.clone(),
            METRIC_AUTO,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            vec![-1; 5],
        ),
        SectorSpectrum::Rpa(r) => {
            let real: Vec<_> = r
                .modes
                .iter()
                .filter(|m| m.kind != RpaModeKind::Complex)
                .collect();
            let (metric, report) = match &r.report {
                RpaReport::Signed {
                    events,
                    n_negative_basis,
                    general_path,
                    ..
                } => (
                    rpa_metric_code(r.side()),
                    vec![
                        -1,
                        -1,
                        events.len() as i64,
                        *n_negative_basis as i64,
                        i64::from(*general_path),
                    ],
                ),
                RpaReport::Paired { offset, .. } => (
                    if requested == METRIC_RR {
                        METRIC_RR
                    } else {
                        METRIC_II
                    },
                    vec![offset.neg_apb as i64, offset.neg_amb as i64, -1, -1, -1],
                ),
            };
            (
                r.omega2(),
                metric,
                real.iter().map(|m| m.residual).collect(),
                real.iter().map(|m| m.krein).collect(),
                r.complex()
                    .flat_map(|m| [m.omega2.re, m.omega2.im.abs(), m.residual])
                    .collect(),
                report,
            )
        }
    };
    assert_eq!(bits(&out.vals), bits(&vals), "{what}: values");
    assert_eq!(bits(&out.fw), bits(fw), "{what}: f_W");
    assert_eq!(bits(&out.residual), bits(&residual), "{what}: residuals");
    assert_eq!(bits(&out.krein), bits(&krein), "{what}: Krein margins");
    assert_eq!(bits(&out.complex), bits(&cx), "{what}: complex rows");
    assert_eq!(out.dim, resp.dim, "{what}: dim");
    assert_eq!(out.metric, metric, "{what}: metric echo");
    assert_eq!(
        out.converged,
        i32::from(resp.converged),
        "{what}: converged"
    );
    assert_eq!(out.iters, resp.iterations, "{what}: iterations");
    assert_eq!(
        out.applies, resp.applies,
        "{what}: applies (f_W is not counted)"
    );
    assert_eq!(
        out.certified,
        i32::from(resp.all_negatives_certified()),
        "{what}: certified"
    );
    assert_eq!(out.report, report, "{what}: report");
    assert_eq!(
        out.indefinite(),
        resp.metric_indefinite(),
        "{what}: metric_indefinite"
    );
}

const PG4C_SITE_IRREPS: [i32; 4] = [3, 2, 1, 0];

fn pg4c_orbsym() -> Vec<u8> {
    PG4C_SITE_IRREPS
        .iter()
        .map(|&ir| {
            let mut id = 0i32;
            assert_eq!(
                unsafe { qntt_pg_molpro_from_irrep(PG_C2V, ir, &mut id) },
                QNTT_OK,
                "{}",
                last_error()
            );
            u8::try_from(id).expect("a Molpro id fits a byte")
        })
        .collect()
}

fn pg4c_coeffs() -> (Vec<f64>, Vec<f64>) {
    let d = 4usize;
    let mut t = vec![0.0f64; d * d];
    let mut v = vec![0.0f64; d * d * d * d];
    let ti = |i: usize, j: usize| i * d + j;
    let vi = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    for p in 0..d {
        t[ti(p, p)] = 0.17 * (p as f64 - 0.5 * d as f64);
    }
    for p in 0..d {
        for q in 0..d {
            if p != q {
                let w = 1.0 + (p as f64 - q as f64).abs();
                v[vi(p, p, q, q)] = 0.5 / w;
                v[vi(p, q, q, p)] = 0.23 / w;
            }
        }
    }
    for idx in [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 0, 1],
        [2, 3, 1, 0],
        [3, 2, 1, 0],
    ] {
        v[vi(idx[0], idx[1], idx[2], idx[3])] = 0.19;
    }
    (t, v)
}

fn pg4c_ham(target: i32) -> *mut HamH {
    let (t, v) = pg4c_coeffs();
    let ob = pg4c_orbsym();
    let mut h: *mut HamH = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new_pg(
                4,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                target,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "pg4c ham_new_pg at target {target} failed: {}",
        last_error()
    );
    h
}

fn pg4c_reference(seed: u64) -> (*mut HamH, *mut VecH) {
    let h = pg4c_ham(0);
    let ob = pg4c_orbsym();
    let mut len = 0usize;
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                4,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                ptr::null_mut(),
                0,
                &mut len,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut flat = vec![0usize; len];
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                4,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                flat.as_mut_ptr(),
                len,
                &mut len,
            )
        },
        QNTT_OK
    );
    for r in &mut flat {
        *r = (*r).min(1);
    }
    let mut raw: *mut VecH = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn_pg(
                4,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                flat.as_ptr(),
                flat.len(),
                seed,
                &mut raw,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut nrm = 0.0f64;
    assert_eq!(unsafe { qntt_f64_vec_norm(raw, &mut nrm) }, QNTT_OK);
    let inv = 1.0 / nrm;
    let mut x: *mut VecH = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_scaled(raw, &inv, &mut x) }, QNTT_OK);
    unsafe { qntt_f64_vec_free(raw) };
    (h, x)
}

const D6_SITE_IRREPS: [i32; 6] = [2, 1, 0, 2, 1, 0];

fn d6_orbsym() -> Vec<u8> {
    D6_SITE_IRREPS
        .iter()
        .map(|&ir| {
            let mut id = 0i32;
            assert_eq!(
                unsafe { qntt_pg_molpro_from_irrep(PG_C2V, ir, &mut id) },
                QNTT_OK,
                "{}",
                last_error()
            );
            u8::try_from(id).expect("a Molpro id fits a byte")
        })
        .collect()
}

fn d6_coeffs() -> (Vec<f64>, Vec<f64>) {
    let d = 6usize;
    let mut t = vec![0.0f64; d * d];
    let mut v = vec![0.0f64; d * d * d * d];
    let ti = |i: usize, j: usize| i * d + j;
    let vi = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    for p in 0..d {
        t[ti(p, p)] = 0.13 * (p as f64 - 0.5 * d as f64);
    }
    for (p, q) in [(0usize, 3usize), (1, 4), (2, 5)] {
        let h = -0.8 + 0.1 * p as f64;
        t[ti(p, q)] = h;
        t[ti(q, p)] = h;
    }
    for p in 0..d {
        for q in 0..d {
            if p != q {
                let w = 1.0 + (p as f64 - q as f64).abs();
                v[vi(p, p, q, q)] = 0.5 / w;
                v[vi(p, q, q, p)] = 0.23 / w;
            }
        }
    }
    for idx in [[0, 3, 1, 4], [1, 4, 0, 3], [3, 0, 4, 1], [4, 1, 3, 0]] {
        v[vi(idx[0], idx[1], idx[2], idx[3])] = 0.19;
    }
    (t, v)
}

fn d6_ham(target: i32) -> *mut HamH {
    let (t, v) = d6_coeffs();
    let ob = d6_orbsym();
    let mut h: *mut HamH = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_ham_new_pg(
                6,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                target,
                t.as_ptr(),
                t.len(),
                v.as_ptr(),
                v.len(),
                ptr::null(),
                f64::EPSILON,
                0,
                &mut h,
            )
        },
        QNTT_OK,
        "d6 ham_new_pg at target {target} failed: {}",
        last_error()
    );
    h
}

fn d6_reference(seed: u64) -> (*mut HamH, *mut VecH) {
    let h = d6_ham(0);
    let ob = d6_orbsym();
    let mut len = 0usize;
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                6,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                ptr::null_mut(),
                0,
                &mut len,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut flat = vec![0usize; len];
    assert_eq!(
        unsafe {
            qntt_space_max_ranks_pg(
                6,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                flat.as_mut_ptr(),
                len,
                &mut len,
            )
        },
        QNTT_OK
    );
    for r in &mut flat {
        *r = (*r).min(1);
    }
    let mut raw: *mut VecH = ptr::null_mut();
    assert_eq!(
        unsafe {
            qntt_f64_vec_randn_pg(
                6,
                2,
                2,
                PG_C2V,
                ob.as_ptr(),
                ob.len(),
                0,
                flat.as_ptr(),
                flat.len(),
                seed,
                &mut raw,
            )
        },
        QNTT_OK,
        "{}",
        last_error()
    );
    let mut nrm = 0.0f64;
    assert_eq!(unsafe { qntt_f64_vec_norm(raw, &mut nrm) }, QNTT_OK);
    let inv = 1.0 / nrm;
    let mut x: *mut VecH = ptr::null_mut();
    assert_eq!(unsafe { qntt_f64_vec_scaled(raw, &inv, &mut x) }, QNTT_OK);
    unsafe { qntt_f64_vec_free(raw) };
    (h, x)
}

fn schmidt_values(x: *const VecH) -> Vec<f64> {
    let (mut len, mut nb) = (0usize, 0usize);
    assert_eq!(
        unsafe {
            qntt_f64_vec_schmidt(
                x,
                ptr::null_mut(),
                0,
                &mut len,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                0,
                &mut nb,
            )
        },
        QNTT_OK
    );
    let (mut vals, mut wts, mut counts) = (vec![0.0f64; len], vec![0.0f64; len], vec![0usize; nb]);
    assert_eq!(
        unsafe {
            qntt_f64_vec_schmidt(
                x,
                vals.as_mut_ptr(),
                vals.len(),
                &mut len,
                wts.as_mut_ptr(),
                wts.len(),
                counts.as_mut_ptr(),
                counts.len(),
                &mut nb,
            )
        },
        QNTT_OK
    );
    let mut w: Vec<f64> = vals.iter().zip(&wts).map(|(s, w)| w.sqrt() * s).collect();
    w.sort_by(f64::total_cmp);
    w
}

#[test]
fn sector_options_default_mirrors_the_rust_defaults() {
    let mut o = QnttSectorOpts {
        max_iter: 0,
        tol: 0.0,
        max_subspace: 7,
        readout: -7,
        metric: -7,
        kernel: -7,
        reserved: -7,
    };
    assert_eq!(unsafe { qntt_sector_options_default(&mut o) }, QNTT_OK);
    let core = SectorOpts::default();
    assert_eq!(o.max_iter, core.max_iter);
    assert_eq!(o.tol, core.tol);
    assert_eq!(
        o.max_subspace, core.max_subspace,
        "0 = the kernel's auto bound"
    );
    assert_eq!(o.readout, READOUT_TDA);
    assert_eq!(o.metric, METRIC_AUTO);
    assert!(matches!(core.kernel, RpaKernel::Paired));
    assert_eq!(o.kernel, QNTT_RPA_KERNEL_PAIRED);
    assert_eq!(o.reserved, 0);
    assert_eq!(
        unsafe { qntt_sector_options_default(ptr::null_mut()) },
        QNTT_NULL_ARG
    );

    let (h, x) = reference();
    let h_out = ham_at_target(1);
    for bad in [
        QnttSectorOpts {
            readout: 2,
            ..QnttSectorOpts::defaults()
        },
        QnttSectorOpts {
            metric: 3,
            ..QnttSectorOpts::defaults()
        },
        QnttSectorOpts {
            metric: -1,
            ..QnttSectorOpts::defaults()
        },
        QnttSectorOpts {
            readout: READOUT_RPA,
            kernel: 2,
            ..QnttSectorOpts::defaults()
        },
        QnttSectorOpts {
            readout: READOUT_RPA,
            kernel: -1,
            ..QnttSectorOpts::defaults()
        },
    ] {
        let got = call_sector(h, h_out, x, 1, 1, &bad, 1);
        assert_eq!(got.rc, QNTT_INVALID_ARG, "{}", last_error());
        assert!(!last_error().is_empty());
    }
    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_rpa_both_readouts_match_the_rust_api() {
    let (h, x) = reference();
    let dims = channel_dims(h, x);
    println!("fx4 (2,1) channel dims: {dims:?}");
    let (phi, dim) = *dims
        .iter()
        .find(|&&(_, dim)| dim > 0)
        .expect("the fixture must have at least one live shifted channel");
    let h_out = ham_at_target(i32::from(phi));
    let k = 2usize;

    let mut opts = QnttSectorOpts::defaults();
    opts.readout = READOUT_TDA;
    opts.max_iter = 200;
    let got = call_sector(h, h_out, x, phi, k, &opts, 0x5EC7);
    assert_eq!(got.rc, QNTT_OK, "{}", last_error());
    assert_eq!(got.dim, dim);
    assert!(got.vals.len() <= k.min(dim));
    assert!(got.vals.windows(2).all(|w| w[0] <= w[1]), "{:?}", got.vals);
    assert!(got.applies > 0, "a live channel must spend applies");
    assert!(got.converged == 0 || got.converged == 1);
    assert!(got.certified == 0 || got.certified == 1);
    assert_eq!(got.metric, METRIC_AUTO);

    let (rt, _) = rust_sector(
        h,
        h_out,
        x,
        phi,
        k,
        &opts,
        CurvatureReg::Off,
        SympPrecond::Off,
        0x5EC7,
    );
    assert_eq!(rt.dim, got.dim);
    assert_eq!(
        rt.theta().unwrap().to_vec(),
        got.vals,
        "the ABI TDA arm must be the Rust arm verbatim"
    );
    assert_abi_is_rust(&got, &rt, &[], opts.metric, "TDA");

    opts.readout = READOUT_RPA;
    opts.max_iter = 400;
    let rpa = call_sector(h, h_out, x, phi, k, &opts, 0x5EC7);
    assert_eq!(rpa.rc, QNTT_OK, "{}", last_error());
    assert_eq!(rpa.dim, dim);
    assert!(rpa.vals.windows(2).all(|w| w[0] <= w[1]), "{:?}", rpa.vals);
    assert!(
        [METRIC_II, METRIC_RR, METRIC_AUTO].contains(&rpa.metric),
        "out_metric = {}",
        rpa.metric
    );
    if rpa.applies > 0 {
        assert!(
            rpa.metric == METRIC_II || rpa.metric == METRIC_RR,
            "a run that spent applies resolved its metric; got {}",
            rpa.metric
        );
    }
    println!(
        "phi = {phi}: tda = {:?} (conv {}), rpa omega2 = {:?} (metric {}, conv {}, applies {}, \
         iters {}, certified {}, f_W {:?}, residuals {:?}, krein {:?}, report {:?})",
        got.vals,
        got.converged,
        rpa.vals,
        rpa.metric,
        rpa.converged,
        rpa.applies,
        rpa.iters,
        rpa.certified,
        rpa.fw,
        rpa.residual,
        rpa.krein,
        rpa.report
    );
    let (rr, rfw) = rust_sector(
        h,
        h_out,
        x,
        phi,
        k,
        &opts,
        CurvatureReg::Off,
        SympPrecond::Off,
        0x5EC7,
    );
    assert_eq!(
        rr.omega2().unwrap().to_vec(),
        rpa.vals,
        "the ABI RPA arm must be the Rust arm verbatim"
    );
    assert_abi_is_rust(&rpa, &rr, &rfw, opts.metric, "RPA");
    assert_eq!(rpa.fw.len(), rpa.vals.len(), "one f_W per mode");

    let mut len = usize::MAX;
    let rc = unsafe {
        qntt_f64_sector_rpa(
            h,
            h_out,
            x,
            phi,
            k,
            &opts,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            QNTT_SECTOR_PRECOND_OFF,
            0.0,
            0x5EC7,
            ptr::null_mut(),
            0,
            &mut len,
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
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            0,
            ptr::null_mut(),
        )
    };
    assert_eq!(rc, QNTT_OK, "{}", last_error());
    assert_eq!(len, rpa.vals.len(), "the probe must report the real length");
    if len > 0 {
        let mut short = vec![f64::NAN; len - 1];
        let mut len2 = 0usize;
        let (mut sdim, mut smetric) = (usize::MAX, -99i32);
        let (mut sconv, mut scert) = (-99i32, -99i32);
        let (mut siters, mut sapplies) = (usize::MAX, usize::MAX);
        let (mut srep, mut srepl) = (vec![-99i64; 5], usize::MAX);
        let rc = unsafe {
            qntt_f64_sector_rpa(
                h,
                h_out,
                x,
                phi,
                k,
                &opts,
                QNTT_CURVATURE_REG_OFF,
                0.0,
                QNTT_SECTOR_PRECOND_OFF,
                0.0,
                0x5EC7,
                short.as_mut_ptr(),
                short.len(),
                &mut len2,
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
                &mut sdim,
                &mut smetric,
                &mut sconv,
                &mut siters,
                &mut sapplies,
                &mut scert,
                srep.as_mut_ptr(),
                srep.len(),
                &mut srepl,
            )
        };
        assert_eq!(rc, QNTT_INVALID_ARG);
        assert!(last_error().contains("need"), "{}", last_error());
        assert_eq!(
            (sdim, smetric, sconv, siters, sapplies, scert),
            (usize::MAX, -99i32, -99i32, usize::MAX, usize::MAX, -99i32),
            "a refused call must not have written any scalar out-parameter"
        );
        assert_eq!(
            (srep, srepl),
            (vec![-99i64; 5], usize::MAX),
            "nor the report"
        );
        assert_eq!(
            len2, len,
            "out_len IS written on the refusal — that is the two-pass length protocol, and it is \
             what tells the caller how large the buffer must be"
        );
    }

    let none = call_sector(h, h_out, x, phi, 0, &opts, 0x5EC7);
    assert_eq!(none.rc, QNTT_OK, "{}", last_error());
    assert!(none.vals.is_empty());
    assert_eq!(none.dim, dim, "the channel's true dim is still reported");
    assert_eq!(none.applies, 0);
    assert_eq!(none.converged, 1);

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn a_structurally_empty_channel_is_an_answer() {
    let (h, x) = reference();
    let dims = channel_dims(h, x);
    let empty: Vec<u8> = dims
        .iter()
        .filter(|&&(_, dim)| dim == 0)
        .map(|&(phi, _)| phi)
        .collect();
    assert_eq!(
        empty,
        vec![1u8, 2],
        "fx4 (2,1) at the rank-2 typed grid: fluxes 1 and 2 must be the EMPTY channels and flux \
         3 live — bond 1's (0,0) sector has typed cap 0 in every irrep. Measured inventory \
         {dims:?}"
    );
    for phi in empty {
        let h_out = ham_at_target(i32::from(phi));
        for (readout, kernel) in [
            (READOUT_TDA, QNTT_RPA_KERNEL_PAIRED),
            (READOUT_RPA, QNTT_RPA_KERNEL_PAIRED),
            (READOUT_RPA, QNTT_RPA_KERNEL_SIGNED),
        ] {
            let opts = QnttSectorOpts {
                readout,
                kernel,
                ..QnttSectorOpts::defaults()
            };
            let got = call_sector(h, h_out, x, phi, 2, &opts, 0x5EC7);
            assert_eq!(got.rc, QNTT_OK, "{}", last_error());
            assert!(got.vals.is_empty());
            assert_eq!(got.dim, 0);
            assert_eq!(got.applies, 0);
            assert_eq!(got.converged, 1);
            assert!(
                got.fw.is_empty()
                    && got.residual.is_empty()
                    && got.krein.is_empty()
                    && got.complex.is_empty(),
                "an empty channel writes empty per-mode buffers"
            );
            let (want, metric): (&[i64], i32) = match (readout, kernel) {
                (READOUT_TDA, _) => (&[-1; 5], METRIC_AUTO),
                (_, QNTT_RPA_KERNEL_PAIRED) => (&[0, 0, -1, -1, -1], METRIC_II),
                _ => (&[-1, -1, 0, 0, 0], METRIC_II),
            };
            assert_eq!(
                (got.indefinite(), &got.report[..], got.metric),
                (false, want, metric),
                "readout {readout}, kernel {kernel}: an empty channel met no metric"
            );
        }
        unsafe { qntt_f64_ham_free(h_out) };
    }
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_rpa_dense_is_tri_state_and_probes() {
    let (h, x) = reference();
    let dims = channel_dims(h, x);
    let (phi, dim) = *dims
        .iter()
        .find(|&&(_, dim)| dim > 0)
        .expect("the fixture must have at least one live shifted channel");
    let h_out = ham_at_target(i32::from(phi));

    let (mut len, mut ddim) = (usize::MAX, usize::MAX);
    let (mut stable, mut metric) = (-99i32, -99i32);
    let rc0 = unsafe {
        qntt_f64_sector_rpa_dense(
            h,
            h_out,
            x,
            phi,
            ptr::null_mut(),
            0,
            &mut len,
            &mut ddim,
            &mut stable,
            &mut metric,
        )
    };
    assert_eq!(ddim, dim, "out_dim must agree with the iterative arm's");
    assert!(
        [1i32, 0, -1].contains(&stable),
        "out_stable must be the tri-state, got {stable}"
    );
    assert!([METRIC_II, METRIC_RR, METRIC_AUTO].contains(&metric));
    println!("dense phi = {phi}: stable = {stable}, metric = {metric}, len = {len}");

    if stable == -1 {
        assert_eq!(rc0, QNTT_NOT_CONVERGED);
        assert_eq!(len, 0);
        assert!(!last_error().is_empty());
    } else {
        assert_eq!(rc0, QNTT_OK, "{}", last_error());
        let mut w2 = vec![f64::NAN; len];
        let mut len2 = usize::MAX;
        let rc = unsafe {
            qntt_f64_sector_rpa_dense(
                h,
                h_out,
                x,
                phi,
                w2.as_mut_ptr(),
                w2.len(),
                &mut len2,
                ptr::null_mut(),
                &mut stable,
                ptr::null_mut(),
            )
        };
        assert_eq!(rc, QNTT_OK, "{}", last_error());
        assert_eq!(len2, len);
        assert!(w2.windows(2).all(|w| w[0] <= w[1]), "{w2:?}");
        if stable == 1 {
            assert!(w2.iter().all(|&z| z > 0.0), "{w2:?}");
        } else {
            assert!(
                w2.iter().any(|&z| z < 0.0),
                "resonant with no negative: {w2:?}"
            );
        }
        if len > 0 {
            let mut short = vec![f64::NAN; len - 1];
            let mut st2 = -99i32;
            let rc = unsafe {
                qntt_f64_sector_rpa_dense(
                    h,
                    h_out,
                    x,
                    phi,
                    short.as_mut_ptr(),
                    short.len(),
                    &mut len2,
                    ptr::null_mut(),
                    &mut st2,
                    ptr::null_mut(),
                )
            };
            assert_eq!(rc, QNTT_INVALID_ARG, "{}", last_error());
            assert_eq!(
                unsafe {
                    qntt_f64_sector_rpa_dense(
                        h,
                        h_out,
                        x,
                        phi,
                        w2.as_mut_ptr(),
                        w2.len(),
                        &mut len2,
                        ptr::null_mut(),
                        ptr::null_mut(),
                        ptr::null_mut(),
                    )
                },
                QNTT_NULL_ARG,
                "out_stable is a required out-parameter"
            );
        }

        let want = unsafe {
            let hr = borrow(h as *const _).unwrap();
            let ho = borrow(h_out as *const _).unwrap();
            let xr = borrow(x as *const _).unwrap();
            let ctx = weingarten_operator(hr, xr);
            let sc = SectorCtx::new(&ctx, ho, Irrep::new(phi));
            let basis = sc.basis();
            let spec = sc.symplectic_eigenvalues(&basis);
            assert_eq!(
                matches!(spec, SymplecticSpectrum::Stable { .. }),
                stable == 1
            );
            spec.omega_squared().expect("not the Unstable arm here")
        };
        assert_eq!(
            want, w2,
            "the ABI dense arm must be the Rust oracle verbatim"
        );
    }

    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_rpa_dense_reports_an_unstable_channel_as_a_status() {
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
    let (phi, dim) = *channel_dims(h, x)
        .iter()
        .find(|&&(_, dim)| dim > 0)
        .expect("the fixture must have at least one live shifted channel");
    let h_out = ham_at_target(i32::from(phi));

    let mut w2 = vec![f64::NAN; dim];
    let (mut len, mut ddim) = (usize::MAX, usize::MAX);
    let (mut stable, mut metric) = (-99i32, -99i32);
    let rc = unsafe {
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
    println!("random reference, phi = {phi}: stable = {stable}, metric = {metric}, rc = {rc}");
    assert_eq!(ddim, dim);
    match stable {
        -1 => {
            assert_eq!(rc, QNTT_NOT_CONVERGED, "{}", last_error());
            assert_eq!(len, 0, "nothing may be written on the unstable arm");
            let msg = last_error();
            assert!(msg.contains("indefinite"), "{msg:?}");
            assert_eq!(
                metric, METRIC_AUTO,
                "no metric is resolved when neither block is PD"
            );
        }
        0 | 1 => {
            assert_eq!(rc, QNTT_OK, "{}", last_error());
            assert!(len > 0);
        }
        v => panic!("out_stable = {v} is not the documented tri-state"),
    }
    assert_eq!(
        stable, -1,
        "this fixture was chosen for its indefinite pair (measured: fx4 (2,1), rank cap 1, \
         seed 7 -> Unstable); it now reports arm {stable}, so the QNTT_NOT_CONVERGED path is \
         no longer covered"
    );
    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_rpa_refuses_caller_errors_without_panicking() {
    let (h, x) = reference();
    let h1 = ham_at_target(1);
    let h2 = ham_at_target(2);
    let opts = QnttSectorOpts::defaults();

    let expect_invalid = |rc: i32, needle: &str, what: &str| {
        assert_eq!(rc, QNTT_INVALID_ARG, "{what}: {}", last_error());
        let msg = last_error();
        assert!(
            msg.contains(needle),
            "{what}: message {msg:?} does not mention {needle:?}"
        );
    };

    expect_invalid(
        call_sector(h, h1, x, 0, 2, &opts, 1).rc,
        "rpa_signed",
        "phi = 0",
    );
    let mut ddim = 0usize;
    let mut st = 0i32;
    expect_invalid(
        unsafe {
            qntt_f64_sector_rpa_dense(
                h,
                h1,
                x,
                0,
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                &mut ddim,
                &mut st,
                ptr::null_mut(),
            )
        },
        "rpa_signed",
        "phi = 0 (dense)",
    );

    expect_invalid(
        call_sector(h, h1, x, 7, 2, &opts, 1).rc,
        "point group",
        "phi out of range",
    );

    expect_invalid(
        call_sector(h, h, x, 1, 2, &opts, 1).rc,
        "BUILT",
        "h_out = h_ref",
    );
    expect_invalid(
        call_sector(h, h2, x, 1, 2, &opts, 1).rc,
        "BUILT",
        "h_out at the wrong target",
    );

    expect_invalid(
        call_sector(h1, h2, x, 1, 2, &opts, 1).rc,
        "h_ref",
        "h_ref off x's space",
    );

    assert_eq!(
        call_sector(ptr::null(), h1, x, 1, 2, &opts, 1).rc,
        QNTT_NULL_ARG
    );
    let mut buf = [f64::NAN; 2];
    let mut len = 0usize;
    assert_eq!(
        unsafe {
            qntt_f64_sector_rpa(
                h,
                h1,
                x,
                1,
                2,
                ptr::null(),
                QNTT_CURVATURE_REG_OFF,
                0.0,
                QNTT_SECTOR_PRECOND_OFF,
                0.0,
                1,
                buf.as_mut_ptr(),
                buf.len(),
                &mut len,
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
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
            )
        },
        QNTT_NULL_ARG,
    );

    unsafe {
        qntt_f64_ham_free(h2);
        qntt_f64_ham_free(h1);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_rpa_off_pair_and_off_precond_are_the_rust_defaults() {
    let (h, x) = reference();
    let (phi, _) = *channel_dims(h, x)
        .iter()
        .find(|&&(_, dim)| dim > 0)
        .expect("a live shifted channel");
    let h_out = ham_at_target(i32::from(phi));
    for readout in [READOUT_RPA, READOUT_TDA] {
        let opts = QnttSectorOpts {
            readout,
            max_iter: if readout == READOUT_RPA { 400 } else { 200 },
            ..QnttSectorOpts::defaults()
        };
        let k = 2usize;
        let base = call_sector(h, h_out, x, phi, k, &opts, 0x5EC7);
        assert_eq!(base.rc, QNTT_OK, "{}", last_error());
        let variants = [
            (
                "curvature OFF(NaN)",
                QNTT_CURVATURE_REG_OFF,
                f64::NAN,
                QNTT_SECTOR_PRECOND_OFF,
                0.0,
            ),
            (
                "precond AUTO",
                QNTT_CURVATURE_REG_OFF,
                0.0,
                QNTT_SECTOR_PRECOND_AUTO,
                0.0,
            ),
            (
                "precond OFF(NaN kappa)",
                QNTT_CURVATURE_REG_OFF,
                0.0,
                QNTT_SECTOR_PRECOND_OFF,
                f64::NAN,
            ),
        ];
        for (what, ck, cv, pc, pk) in variants {
            let got = call_sector_with(h, h_out, x, phi, k, &opts, ck, cv, pc, pk, 0x5EC7);
            assert_eq!(got.rc, QNTT_OK, "{what}: {}", last_error());
            assert_eq!(
                bits(&got.vals),
                bits(&base.vals),
                "readout {readout} {what}: values"
            );
            assert_eq!(
                bits(&got.fw),
                bits(&base.fw),
                "readout {readout} {what}: f_W"
            );
            assert_eq!(bits(&got.residual), bits(&base.residual));
            assert_eq!(bits(&got.krein), bits(&base.krein));
            assert_eq!(
                (got.iters, got.applies, got.converged, got.metric),
                (base.iters, base.applies, base.converged, base.metric),
                "readout {readout} {what}: counts and echo"
            );
        }
        let (resp, fw) = rust_sector(
            h,
            h_out,
            x,
            phi,
            k,
            &opts,
            CurvatureReg::default(),
            SectorOpts::default().precond,
            0x5EC7,
        );
        assert_abi_is_rust(
            &base,
            &resp,
            &fw,
            opts.metric,
            &format!("readout {readout}: the Rust defaults"),
        );
    }
    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_rpa_curvature_and_precond_reach_the_rust_api() {
    let (h, x) = pg4c_reference(7);
    let phi = 1u8;
    let h_out = pg4c_ham(i32::from(phi));
    let opts = QnttSectorOpts {
        readout: READOUT_RPA,
        metric: METRIC_II,
        max_iter: 400,
        ..QnttSectorOpts::defaults()
    };
    let k = 2usize;
    let probe = call_sector(h, h_out, x, phi, 0, &opts, 0x5EC7);
    assert_eq!(probe.rc, QNTT_OK, "{}", last_error());
    assert!(
        probe.dim > 0,
        "fixture premise: the pg4c curved channel phi = 1 is live"
    );
    let w = schmidt_values(x);
    let top = w[w.len() - 1];
    let next = w
        .iter()
        .rev()
        .copied()
        .find(|&s| (top - s) / top >= 1e-3)
        .expect("fixture premise: two distinct Schmidt values");
    let f = (top * next).sqrt();
    assert!(
        w.iter().all(|s| (s - f).abs() > 1e-6 * f),
        "floor rule: no Schmidt value within 1e-6 of the floor"
    );
    println!(
        "C2 pg4c: dim {}, Schmidt in [{:.4e}, {top:.4e}], floor {f:.4e}",
        probe.dim, w[0]
    );
    let off = call_sector(h, h_out, x, phi, k, &opts, 0x5EC7);
    assert_eq!(off.rc, QNTT_OK, "{}", last_error());
    let scale = off
        .vals
        .iter()
        .fold(0.0f64, |m, z| m.max(z.abs()))
        .max(f64::MIN_POSITIVE);
    for (ck, reg) in [
        (QNTT_CURVATURE_REG_HARD, CurvatureReg::Hard(f)),
        (QNTT_CURVATURE_REG_TIKHONOV, CurvatureReg::Tikhonov(f)),
    ] {
        for (pc, precond) in [
            (QNTT_SECTOR_PRECOND_OFF, SympPrecond::Off),
            (
                QNTT_SECTOR_PRECOND_D0,
                SympPrecond::DiagUnshifted { kappa: 0.0 },
            ),
        ] {
            let out = call_sector_with(h, h_out, x, phi, k, &opts, ck, f, pc, 0.0, 0x5EC7);
            let (resp, fw) = rust_sector(h, h_out, x, phi, k, &opts, reg, precond, 0x5EC7);
            println!(
                "C2 {reg:?} {precond:?}: omega2 {:?} (OFF {:?}), flagged {}, applies {}",
                out.vals,
                off.vals,
                out.indefinite(),
                out.applies
            );
            assert_abi_is_rust(
                &out,
                &resp,
                &fw,
                opts.metric,
                &format!("C2 {reg:?} {precond:?}"),
            );
            assert_eq!(
                resp.rpa().map(|r| r.side()),
                Some(RpaMetric::Ii),
                "the II request is echoed"
            );
            if ck == QNTT_CURVATURE_REG_HARD && pc == QNTT_SECTOR_PRECOND_OFF {
                let moved = out
                    .vals
                    .iter()
                    .zip(&off.vals)
                    .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
                    / scale;
                assert!(
                    out.vals.len() != off.vals.len() || moved >= 1e-8,
                    "non-vacuity: HARD between two Schmidt values must move omega^2 ({moved:e})"
                );
            }
        }
    }
    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }

    let (h6, x6) = d6_reference(0x9002);
    let mut premise_hit = 0usize;
    for phi6 in 1u8..4 {
        if premise_hit > 0 {
            break;
        }
        let h6_out = d6_ham(i32::from(phi6));
        for readout in [READOUT_RPA, READOUT_TDA] {
            let o6 = QnttSectorOpts {
                readout,
                metric: METRIC_II,
                max_iter: if readout == READOUT_RPA { 120 } else { 60 },
                ..QnttSectorOpts::defaults()
            };
            let off6 = call_sector(h6, h6_out, x6, phi6, 2, &o6, 0x5EC7);
            assert_eq!(off6.rc, QNTT_OK, "{}", last_error());
            if off6.dim < 20 {
                continue;
            }
            let d06 = call_sector_with(
                h6,
                h6_out,
                x6,
                phi6,
                2,
                &o6,
                QNTT_CURVATURE_REG_OFF,
                0.0,
                QNTT_SECTOR_PRECOND_D0,
                0.0,
                0x5EC7,
            );
            let (resp, fw) = rust_sector(
                h6,
                h6_out,
                x6,
                phi6,
                2,
                &o6,
                CurvatureReg::Off,
                SympPrecond::DiagUnshifted { kappa: 0.0 },
                0x5EC7,
            );
            println!(
                "C2 d6 premise phi {phi6} readout {readout} dim {}: OFF {:?} ({} iters, {} \
                 applies) | D0 {:?} ({} iters, {} applies)",
                off6.dim, off6.vals, off6.iters, off6.applies, d06.vals, d06.iters, d06.applies
            );
            assert_abi_is_rust(&d06, &resp, &fw, o6.metric, &format!("C2 d6 phi {phi6} D0"));
            assert!(
                bits(&d06.vals) != bits(&off6.vals)
                    || bits(&d06.residual) != bits(&off6.residual)
                    || (d06.iters, d06.applies) != (off6.iters, off6.applies),
                "PREMISE: phi {phi6} readout {readout}: D0 is bitwise OFF on a {}-wide channel — \
                 the precond code never reached the solve",
                off6.dim
            );
            premise_hit += 1;
        }
        unsafe { qntt_f64_ham_free(h6_out) };
    }
    assert!(
        premise_hit >= 2,
        "fixture premise: a d6 channel with dim >= 20 on both readouts ({premise_hit} legs)"
    );
    unsafe {
        qntt_f64_vec_free(x6);
        qntt_f64_ham_free(h6);
    }

    let (h5, x5) = pg4c_reference(0x7505);
    let h5_out = pg4c_ham(1);
    let orr = QnttSectorOpts {
        readout: READOUT_RPA,
        metric: METRIC_RR,
        kernel: QNTT_RPA_KERNEL_SIGNED,
        max_iter: 400,
        ..QnttSectorOpts::defaults()
    };
    let rr = call_sector(h5, h5_out, x5, 1, 2, &orr, 0x5EC7);
    let (resp, fw) = rust_sector(
        h5,
        h5_out,
        x5,
        1,
        2,
        &orr,
        CurvatureReg::Off,
        SympPrecond::Off,
        0x5EC7,
    );
    println!(
        "C2 0x7505 RR: omega2 {:?}, signature {:?}, flagged {}, general {}",
        rr.vals,
        rr.krein,
        rr.indefinite(),
        rr.report[4]
    );
    assert_abi_is_rust(&rr, &resp, &fw, orr.metric, "C2 0x7505 RR");
    assert_eq!(rr.metric, METRIC_RR, "the RR lane is echoed");
    assert_eq!(
        (rr.indefinite(), rr.report[4]),
        (true, 0),
        "RR at 0x7505: W = A + B indefinite (flagged), R = A - B PD (definite path)"
    );
    unsafe {
        qntt_f64_ham_free(h5_out);
        qntt_f64_vec_free(x5);
        qntt_f64_ham_free(h5);
    }
}

#[test]
fn sector_rpa_max_subspace_reaches_the_rust_api() {
    let (h, x) = d6_reference(0x9002);
    let k = 1usize;
    let tight = 2 * (k + k + 2);
    let base = QnttSectorOpts {
        readout: READOUT_RPA,
        metric: METRIC_II,
        max_iter: 120,
        ..QnttSectorOpts::defaults()
    };
    assert_eq!(
        base.max_subspace, 0,
        "the default is the kernel's auto bound"
    );
    let mut hit = 0usize;
    for phi in 1u8..4 {
        let h_out = d6_ham(i32::from(phi));
        let dflt = call_sector(h, h_out, x, phi, k, &base, 0x5EC7);
        assert_eq!(dflt.rc, QNTT_OK, "{}", last_error());
        if dflt.dim < 20 {
            unsafe { qntt_f64_ham_free(h_out) };
            continue;
        }
        hit += 1;
        let cut = call_sector(
            h,
            h_out,
            x,
            phi,
            k,
            &QnttSectorOpts {
                max_subspace: tight,
                ..base
            },
            0x5EC7,
        );
        let auto = (8 * (k + k + 2)).max(32).min(dflt.dim);
        let explicit = call_sector(
            h,
            h_out,
            x,
            phi,
            k,
            &QnttSectorOpts {
                max_subspace: auto,
                ..base
            },
            0x5EC7,
        );
        let (hr, ho, xr) = unsafe {
            (
                borrow(h).unwrap(),
                borrow(h_out).unwrap(),
                borrow(x).unwrap(),
            )
        };
        let ctx = weingarten_operator(hr, xr);
        let sc = SectorCtx::new(&ctx, ho, Irrep::new(phi));
        let core = SectorOpts {
            tol: base.tol,
            max_iter: base.max_iter,
            readout: SectorReadout::Rpa,
            seed: 0x5EC7,
            kernel: RpaKernel::Paired,
            precond: SympPrecond::Off,
            max_subspace: tight,
        };
        let (resp, vecs) = sc.sector_lowest_vectors(k, &core);
        let n2 = resp.omega2().map_or(0, |w| w.len());
        let fw: Vec<f64> = vecs
            .iter()
            .take(n2)
            .map(|m| {
                let v = &m.metric_side(RpaMetric::Ii);
                let qa = v.dot(&sc.a_apply(v));
                let qb = v.dot(&sc.b_apply(v));
                if qa == 0.0 {
                    f64::INFINITY
                } else {
                    qb.abs() / qa.abs()
                }
            })
            .collect();
        println!(
            "P3 max_subspace d6 phi {phi} dim {}: default {:?} ({} iters, {} applies) | tight \
             {tight}: {:?} ({} iters, {} applies)",
            dflt.dim, dflt.vals, dflt.iters, dflt.applies, cut.vals, cut.iters, cut.applies
        );
        assert_abi_is_rust(&cut, &resp, &fw, base.metric, "max_subspace = tight");
        assert!(
            bits(&cut.vals) != bits(&dflt.vals)
                || (cut.iters, cut.applies) != (dflt.iters, dflt.applies),
            "non-vacuity: the tight-bound call is bitwise the default one — the field never reached \
             the solve"
        );
        assert!(
            bits(&explicit.vals) == bits(&dflt.vals)
                && (explicit.iters, explicit.applies) == (dflt.iters, dflt.applies)
                && bits(&explicit.residual) == bits(&dflt.residual),
            "the auto bound {auto} passed explicitly must be the default (0) call bit for bit"
        );
        unsafe { qntt_f64_ham_free(h_out) };
        break;
    }
    assert!(hit > 0, "fixture premise: a d6 channel with dim >= 20");
    unsafe {
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

fn check_sector_refused(out: &SectorOut, what: &str) {
    assert_eq!(out.rc, QNTT_INVALID_ARG, "{what}: {}", last_error());
    assert!(
        out.vals.is_empty()
            && out.len == usize::MAX
            && out.applies == usize::MAX
            && out.metric == -99
            && out.dim == usize::MAX
            && out.report.is_empty(),
        "{what}: no solve ran and nothing was written"
    );
}

#[test]
fn sector_rpa_refuses_bad_curvature_and_precond_codes() {
    let (h, x) = reference();
    let (phi, _) = *channel_dims(h, x)
        .iter()
        .find(|&&(_, dim)| dim > 0)
        .expect("a live shifted channel");
    let h_out = ham_at_target(i32::from(phi));
    let opts = QnttSectorOpts {
        readout: READOUT_RPA,
        ..QnttSectorOpts::defaults()
    };
    for kind in [3, -1] {
        let out = call_sector_with(
            h,
            h_out,
            x,
            phi,
            2,
            &opts,
            kind,
            1e-3,
            QNTT_SECTOR_PRECOND_OFF,
            0.0,
            1,
        );
        check_sector_refused(&out, &format!("curvature kind {kind}"));
        let msg = last_error();
        assert!(
            msg.contains("0 = off") && msg.contains("1 = hard") && msg.contains("2 = tikhonov"),
            "kind {kind}: the message names the codes, got {msg:?}"
        );
    }
    for (kind, value) in [
        (QNTT_CURVATURE_REG_HARD, 0.0),
        (QNTT_CURVATURE_REG_HARD, -1.0),
        (QNTT_CURVATURE_REG_HARD, f64::NAN),
        (QNTT_CURVATURE_REG_HARD, f64::INFINITY),
        (QNTT_CURVATURE_REG_TIKHONOV, 0.0),
    ] {
        let out = call_sector_with(
            h,
            h_out,
            x,
            phi,
            2,
            &opts,
            kind,
            value,
            QNTT_SECTOR_PRECOND_OFF,
            0.0,
            1,
        );
        check_sector_refused(&out, &format!("curvature kind {kind} value {value}"));
    }
    for code in [3, -1] {
        let out = call_sector_with(
            h,
            h_out,
            x,
            phi,
            2,
            &opts,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            code,
            0.0,
            1,
        );
        check_sector_refused(&out, &format!("precond code {code}"));
        let msg = last_error().to_lowercase();
        assert!(
            msg.contains("precond")
                && msg.contains("off")
                && msg.contains("d0")
                && msg.contains("auto"),
            "precond {code}: the message names the codes, got {msg:?}"
        );
    }
    for (ck, cv, pc) in [
        (3, 1e-3, QNTT_SECTOR_PRECOND_OFF),
        (QNTT_CURVATURE_REG_OFF, 0.0, 3),
    ] {
        let out = call_sector_with(ptr::null(), h_out, x, phi, 2, &opts, ck, cv, pc, 0.0, 1);
        assert_ne!(
            out.rc, QNTT_NULL_ARG,
            "the selectors are decoded before any handle borrow (kind {ck}, precond {pc})"
        );
        check_sector_refused(&out, &format!("null h_ref, kind {ck}, precond {pc}"));
    }
    let auto_cap = call_sector_with(
        h,
        h_out,
        x,
        phi,
        2,
        &opts,
        QNTT_CURVATURE_REG_OFF,
        0.0,
        QNTT_SECTOR_PRECOND_D0,
        0.0,
        1,
    );
    assert_eq!(auto_cap.rc, QNTT_OK, "{}", last_error());
    for kappa in [f64::NAN, -1.0, f64::INFINITY] {
        let got = call_sector_with(
            h,
            h_out,
            x,
            phi,
            2,
            &opts,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            QNTT_SECTOR_PRECOND_D0,
            kappa,
            1,
        );
        assert_eq!(got.rc, QNTT_OK, "D0 kappa {kappa}: {}", last_error());
        assert_eq!(
            bits(&got.vals),
            bits(&auto_cap.vals),
            "D0 kappa {kappa} is the auto cap, bit for bit"
        );
        assert_eq!(got.applies, auto_cap.applies);
    }
    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}

#[test]
fn sector_precond_from_code_decodes_and_validates() {
    assert_eq!(
        (
            QNTT_SECTOR_PRECOND_OFF,
            QNTT_SECTOR_PRECOND_D0,
            QNTT_SECTOR_PRECOND_AUTO
        ),
        (0, 1, 2)
    );
    assert_eq!(
        sector_precond_from_code(QNTT_SECTOR_PRECOND_OFF, f64::NAN).ok(),
        Some(SympPrecond::Off)
    );
    assert_eq!(
        sector_precond_from_code(QNTT_SECTOR_PRECOND_D0, 7.0).ok(),
        Some(SympPrecond::DiagUnshifted { kappa: 7.0 })
    );
    let auto = sector_precond_from_code(QNTT_SECTOR_PRECOND_AUTO, 0.0).expect("AUTO decodes");
    assert_eq!(
        auto.resolve_sector(),
        SympPrecond::Off,
        "AUTO is the sector default"
    );
    for kappa in [f64::NAN, 0.0, -3.0, f64::INFINITY] {
        assert!(
            matches!(
                sector_precond_from_code(QNTT_SECTOR_PRECOND_D0, kappa),
                Ok(SympPrecond::DiagUnshifted { .. })
            ),
            "D0 with kappa {kappa} is D0 at the auto cap, not a refusal"
        );
    }
    for code in [3, -1, 7] {
        let err = sector_precond_from_code(code, 1.0).expect_err("an unknown code is refused");
        assert_eq!(err.code, QNTT_INVALID_ARG, "code {code}: {}", err.msg);
        let msg = err.msg.to_lowercase();
        assert!(
            msg.contains("off") && msg.contains("d0") && msg.contains("auto"),
            "code {code}: the message names the codes, got {:?}",
            err.msg
        );
    }
}

#[test]
fn the_sector_rpa_declaration_carries_the_p2_arguments() {
    let hdr = include_str!("../../include/qntt.h");
    let flat = hdr
        .replace('\\', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let start = flat
        .find("int32_t qntt_##SFX##_sector_rpa(")
        .expect("qntt.h declares qntt_##SFX##_sector_rpa");
    let end = start + flat[start..].find(");").expect("the declaration ends");
    let decl = &flat[start..end];
    for piece in [
        "const QnttSectorOpts *opts, int32_t curvature_kind, double curvature_value, int32_t \
         precond, double precond_kappa, uint64_t seed",
        "double *out_fw, size_t fw_cap, size_t *out_fw_len, double *out_residual, size_t res_cap, \
         size_t *out_res_len, double *out_krein, size_t krein_cap, size_t *out_krein_len, double \
         *out_complex, size_t cx_cap, size_t *out_cx_len",
        "int32_t *out_negatives_certified, int64_t *out_report, size_t report_cap, size_t \
         *out_report_len",
    ] {
        assert!(
            decl.contains(piece),
            "the sector_rpa declaration does not carry `{piece}`: {decl}"
        );
    }
    assert!(
        decl.ends_with("size_t *out_report_len"),
        "out_report_len is the LAST parameter: {decl}"
    );
}

#[test]
fn sector_rpa_n_max_buffers_suffice_and_tda_writes_empty_rpa_buffers() {
    let (h, x) = pg4c_reference(7);
    let phi = 1u8;
    let h_out = pg4c_ham(i32::from(phi));
    let k = 2usize;
    let opts = QnttSectorOpts {
        readout: READOUT_RPA,
        metric: METRIC_II,
        max_iter: 40,
        ..QnttSectorOpts::defaults()
    };
    let nm = n_max(&opts, k);
    assert_eq!(nm, 40);
    let out = call_sector(h, h_out, x, phi, k, &opts, 0x5EC7);
    assert_eq!(
        out.rc,
        QNTT_OK,
        "n_max buffers never fail: {}",
        last_error()
    );
    let n_neg = out.vals.iter().filter(|&&z| z < 0.0).count();
    println!(
        "C5 pg4c: omega2 {:?} ({} entries, {n_neg} negative), signature {:?}, complex {:?}, flagged \
         {}, general {}, n_neg_basis {}",
        out.vals,
        out.len,
        out.krein,
        out.complex,
        out.indefinite(),
        out.report[4],
        out.report[3]
    );
    assert!(
        out.vals.len() > k,
        "fixture premise: the signed list exceeds k ({} entries) — a k-sized buffer would fail",
        out.vals.len()
    );
    for (what, n) in [
        ("f_W", out.fw.len()),
        ("residual", out.residual.len()),
        ("krein", out.krein.len()),
    ] {
        assert_eq!(n, out.vals.len(), "{what}: one entry per omega^2");
    }
    assert_eq!(
        out.complex.len() % 3,
        0,
        "complex rows are (re, im, residual)"
    );
    assert_eq!(out.report.len(), 5, "the report has five entries");
    let mut short = vec![f64::NAN; k];
    let mut len = 0usize;
    let (mut dim, mut metric, mut conv) = (usize::MAX, -99i32, -99i32);
    let (mut iters, mut applies, mut cert) = (usize::MAX, usize::MAX, -99i32);
    let (mut rep, mut replen) = (vec![-99i64; 5], usize::MAX);
    let rc = unsafe {
        qntt_f64_sector_rpa(
            h,
            h_out,
            x,
            phi,
            k,
            &opts,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            QNTT_SECTOR_PRECOND_OFF,
            0.0,
            0x5EC7,
            short.as_mut_ptr(),
            short.len(),
            &mut len,
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
            &mut metric,
            &mut conv,
            &mut iters,
            &mut applies,
            &mut cert,
            rep.as_mut_ptr(),
            rep.len(),
            &mut replen,
        )
    };
    assert_eq!(rc, QNTT_INVALID_ARG, "a k-sized buffer is too short here");
    assert!(last_error().contains("need"), "{}", last_error());
    assert_eq!(
        len,
        out.vals.len(),
        "out_len is written on the refusal (two-pass protocol)"
    );
    assert_eq!(
        (dim, metric, conv, iters, applies, cert, replen),
        (
            usize::MAX,
            -99,
            -99,
            usize::MAX,
            usize::MAX,
            -99,
            usize::MAX
        ),
        "a refused call writes no scalar"
    );
    assert_eq!(rep, vec![-99i64; 5], "nor the report");

    assert_eq!(
        out.vals.len(),
        3,
        "fixture premise: the seed-7 list has 3 entries"
    );
    let canary = f64::from_bits(0x7FF8_DEAD_BEEF_0001);
    let mut vals = vec![f64::NAN; nm];
    let mut fw = vec![canary; 2];
    let mut res = vec![f64::NAN; nm];
    let mut krein = vec![f64::NAN; nm];
    let mut cx = vec![f64::NAN; 3 * nm];
    let (mut len, mut fwlen, mut reslen, mut kreinlen, mut cxlen) =
        (usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX);
    let (mut dim, mut metric, mut conv) = (usize::MAX, -99i32, -99i32);
    let (mut iters, mut applies, mut cert) = (usize::MAX, usize::MAX, -99i32);
    let (mut rep, mut replen) = (vec![-99i64; 5], usize::MAX);
    let rc = unsafe {
        qntt_f64_sector_rpa(
            h,
            h_out,
            x,
            phi,
            k,
            &opts,
            QNTT_CURVATURE_REG_OFF,
            0.0,
            QNTT_SECTOR_PRECOND_OFF,
            0.0,
            0x5EC7,
            vals.as_mut_ptr(),
            vals.len(),
            &mut len,
            fw.as_mut_ptr(),
            1,
            &mut fwlen,
            res.as_mut_ptr(),
            res.len(),
            &mut reslen,
            krein.as_mut_ptr(),
            krein.len(),
            &mut kreinlen,
            cx.as_mut_ptr(),
            cx.len(),
            &mut cxlen,
            &mut dim,
            &mut metric,
            &mut conv,
            &mut iters,
            &mut applies,
            &mut cert,
            rep.as_mut_ptr(),
            rep.len(),
            &mut replen,
        )
    };
    assert_eq!(
        rc, QNTT_INVALID_ARG,
        "fw_cap = 1 is too short for 3 entries"
    );
    let msg = last_error();
    assert!(msg.contains("f_w") && msg.contains("need"), "{msg}");
    assert_eq!(
        (len, fwlen),
        (3, 3),
        "out_len and out_fw_len are written (two-pass)"
    );
    assert!(
        fw.iter().all(|z| z.to_bits() == canary.to_bits()),
        "the short f_w buffer — and the canary past its cap — is never written: {fw:?}"
    );
    assert_eq!(
        (reslen, kreinlen, cxlen, replen),
        (usize::MAX, usize::MAX, usize::MAX, usize::MAX),
        "the copy-outs after the failing one never run"
    );
    assert!(
        res.iter().all(|z| z.is_nan()) && krein.iter().all(|z| z.is_nan()),
        "nothing written past the failing buffer"
    );
    assert_eq!(
        (dim, metric, conv, iters, applies, cert, replen),
        (
            usize::MAX,
            -99,
            -99,
            usize::MAX,
            usize::MAX,
            -99,
            usize::MAX
        ),
        "a refused call writes no scalar"
    );
    assert_eq!(rep, vec![-99i64; 5], "nor the report");

    let topts = QnttSectorOpts {
        readout: READOUT_TDA,
        ..opts
    };
    let tda = call_sector(h, h_out, x, phi, k, &topts, 0x5EC7);
    assert_eq!(tda.rc, QNTT_OK, "{}", last_error());
    assert!(tda.vals.len() <= k, "TDA writes at most k entries");
    assert!(
        tda.fw.is_empty()
            && tda.residual.is_empty()
            && tda.krein.is_empty()
            && tda.complex.is_empty(),
        "the RPA-only buffers are written with length 0 on TDA"
    );
    assert_eq!(
        (tda.indefinite(), &tda.report[..]),
        (false, &[-1i64; 5][..]),
        "the report is all -1 on TDA"
    );
    let w = schmidt_values(x);
    let f = (w[w.len() - 1] * w[w.len() - 2]).sqrt();
    let hard = call_sector_with(
        h,
        h_out,
        x,
        phi,
        k,
        &topts,
        QNTT_CURVATURE_REG_HARD,
        f,
        QNTT_SECTOR_PRECOND_OFF,
        0.0,
        0x5EC7,
    );
    assert_eq!(hard.rc, QNTT_OK, "{}", last_error());
    assert_eq!(
        bits(&hard.vals),
        bits(&tda.vals),
        "the curvature pair has no effect on TDA: A is never regularized"
    );
    assert_eq!((hard.iters, hard.applies), (tda.iters, tda.applies));
    let bad = call_sector_with(
        h,
        h_out,
        x,
        phi,
        k,
        &topts,
        QNTT_CURVATURE_REG_HARD,
        f64::NAN,
        QNTT_SECTOR_PRECOND_OFF,
        0.0,
        0x5EC7,
    );
    check_sector_refused(&bad, "TDA validates the curvature pair");
    unsafe {
        qntt_f64_ham_free(h_out);
        qntt_f64_vec_free(x);
        qntt_f64_ham_free(h);
    }
}
