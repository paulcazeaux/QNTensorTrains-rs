// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::{Path, PathBuf};

use qntt::hamiltonian::SparseHamiltonian;
use qntt::integrals::{Integrals, IntegralsOptions, load_fixture_dir};
use qntt::qn::QnSpace;
use qntt::solvers::lanczos::{LanczosOptions, lanczos};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::solvers::pinvit::{
    CalibrateOpts, ExpSumPrecond, OpinvitOpts, PinvitConsts, PinvitOpts, choose_gamma,
    diagonal_one_particle, gamma_for_positive_diagonal, opinvit_ground_state, pinvit_ground_state,
};
use qntt::vector::{TtVector, tt_state_in};

fn fixture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name)
}

fn load(name: &str) -> (SparseHamiltonian<f64>, QnSpace, Integrals<f64>) {
    let opts = IntegralsOptions::default();
    let ld = load_fixture_dir::<f64>(&fixture_dir(name), &opts)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let h = ld.hamiltonian(opts.eps);
    (h, ld.space, ld.ints)
}

fn gamma_both(
    ints: &Integrals<f64>,
    h: &SparseHamiltonian<f64>,
    x0: &TtVector<f64>,
    space: QnSpace,
) -> f64 {
    let nelec = space.nup + space.ndn;
    let g_pd = choose_gamma(h, x0, 2.0);
    let g_d = gamma_for_positive_diagonal(ints, space.nup, space.ndn, nelec, 5.0);
    g_pd.max(g_d)
}

fn hf_determinant(space: QnSpace) -> TtVector<f64> {
    let d = space.d;
    let occ_up: Vec<bool> = (0..d).map(|i| i < space.nup).collect();
    let occ_dn: Vec<bool> = (0..d).map(|i| i < space.ndn).collect();
    tt_state_in(space, &occ_up, &occ_dn)
}

const C0: f64 = 0.6;

fn make_precond(ints: &Integrals<f64>, gamma: f64, space: QnSpace) -> ExpSumPrecond {
    let nelec = space.nup + space.ndn;
    let d = diagonal_one_particle(ints, gamma, nelec);
    ExpSumPrecond::from_diagonal(&d, space.nup, space.ndn, C0)
}

fn lanczos_reference(h: &SparseHamiltonian<f64>, x0: &TtVector<f64>) -> f64 {
    let lz = lanczos(
        h,
        x0,
        &LanczosOptions {
            tol: 1e-11,
            max_iter: 80,
        },
    );
    lz.lambda
}

#[test]
fn tpv_inner_matches_lanczos_lih() {
    let (h, space, ints) = load("lih_sto3g");
    let x0 = hf_determinant(space);
    let gamma = gamma_both(&ints, &h, &x0, space);
    let precond = make_precond(&ints, gamma, space);

    let opts = PinvitOpts {
        consts: PinvitConsts {
            gamma,
            delta: 50.0,
            c: Some(0.5),
        },
        tau: 1e-7,
        max_iter: 300,
        ..PinvitOpts::default()
    };
    let (lam, _y, info) = pinvit_ground_state(&h, &precond, &x0, &opts);
    assert!(
        info.converged,
        "PINVIT did not converge in {} iters",
        info.iters
    );

    let e_pinvit = lam - gamma + ints.e0;
    let e_ref = lanczos_reference(&h, &x0) + ints.e0;
    assert!(
        (e_pinvit - e_ref).abs() < 1e-5,
        "PINVIT E={e_pinvit:.10} vs Lanczos E={e_ref:.10} (Δ={:.2e})",
        (e_pinvit - e_ref).abs()
    );
}

#[test]
fn tpv_inner_matches_mals_lih() {
    let (h, space, ints) = load("lih_sto3g");
    let x0 = hf_determinant(space);
    let gamma = gamma_both(&ints, &h, &x0, space);
    let precond = make_precond(&ints, gamma, space);

    let opts = PinvitOpts {
        consts: PinvitConsts {
            gamma,
            delta: 50.0,
            c: Some(0.5),
        },
        tau: 1e-7,
        max_iter: 300,
        ..PinvitOpts::default()
    };
    let (lam, _y, _) = pinvit_ground_state(&h, &precond, &x0, &opts);
    let e_pinvit = lam - gamma + ints.e0;

    let mopts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        noise: vec![1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 0.0, 0.0],
        max_iter: 30,
        ..MalsOptions::default()
    };
    let (lam_mals, _) = mals(&h, &x0, &mopts);
    let e_mals = lam_mals + ints.e0;
    assert!(
        (e_pinvit - e_mals).abs() < 1e-5,
        "PINVIT E={e_pinvit:.10} vs MALS E={e_mals:.10} (Δ={:.2e})",
        (e_pinvit - e_mals).abs()
    );
}

#[test]
fn tpv_outer_matches_lanczos_lih() {
    let (h, space, ints) = load("lih_sto3g");
    let x0 = hf_determinant(space);
    let gamma = gamma_both(&ints, &h, &x0, space);
    let precond = make_precond(&ints, gamma, space);

    let inner = PinvitOpts {
        consts: PinvitConsts {
            gamma,
            delta: 50.0,
            c: Some(0.5),
        },
        max_iter: 100,
        ..PinvitOpts::default()
    };
    let opts = OpinvitOpts {
        inner,
        tau_outer: 1e-7,
        max_outer: 30,
        ..OpinvitOpts::default()
    };
    let (lam, _y, info) = opinvit_ground_state(&h, &precond, &x0, &opts);
    assert!(
        info.converged,
        "OPINVIT did not converge in {} outer steps",
        info.outer_iters
    );

    let e_out = lam - gamma + ints.e0;
    let e_ref = lanczos_reference(&h, &x0) + ints.e0;
    assert!(
        (e_out - e_ref).abs() < 1e-5,
        "OPINVIT E={e_out:.10} vs Lanczos E={e_ref:.10} (Δ={:.2e})",
        (e_out - e_ref).abs()
    );
}

fn jacobi_spectrum(a0: &[f64], n: usize) -> Vec<f64> {
    let mut a = a0.to_vec();
    for _ in 0..200 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[i * n + j] * a[i * n + j];
            }
        }
        if off.sqrt() < 1e-13 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p * n + q].abs() < 1e-15 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * a[p * n + q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let sn = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - sn * akq;
                    a[k * n + q] = sn * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - sn * aqk;
                    a[q * n + k] = sn * apk + c * aqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    ev.sort_by(|x, y| x.partial_cmp(y).expect("finite spectrum"));
    ev
}

fn sector_configs(space: QnSpace) -> Vec<usize> {
    (0..4usize.pow(space.d as u32))
        .filter(|cfg| {
            let (mut nu, mut nd) = (0usize, 0usize);
            for k in 0..space.d {
                match (cfg / 4usize.pow(k as u32)) % 4 {
                    0 => {}
                    1 => nu += 1,
                    2 => nd += 1,
                    _ => {
                        nu += 1;
                        nd += 1;
                    }
                }
            }
            nu == space.nup && nd == space.ndn
        })
        .collect()
}

fn cfg_occ(cfg: usize, d: usize) -> (Vec<bool>, Vec<bool>) {
    let (mut ou, mut od) = (vec![false; d], vec![false; d]);
    for k in 0..d {
        match (cfg / 4usize.pow(k as u32)) % 4 {
            0 => {}
            1 => ou[k] = true,
            2 => od[k] = true,
            _ => {
                ou[k] = true;
                od[k] = true;
            }
        }
    }
    (ou, od)
}

fn dense_spectrum_a(
    h: &SparseHamiltonian<f64>,
    space: QnSpace,
    p: &ExpSumPrecond,
    gamma: f64,
) -> (f64, f64) {
    let cfgs = sector_configs(space);
    let n = cfgs.len();
    let pos: std::collections::HashMap<usize, usize> =
        cfgs.iter().enumerate().map(|(i, &c)| (c, i)).collect();

    let mut m = vec![0.0f64; n * n];
    for (j, &cfg) in cfgs.iter().enumerate() {
        let (ou, od) = cfg_occ(cfg, space.d);
        let e: TtVector<f64> = tt_state_in(space, &ou, &od);
        let he = h.apply(&e).to_dense();
        for (&c2, &i) in pos.iter() {
            m[i * n + j] = he[c2];
        }
        m[j * n + j] += gamma;
    }
    let sdiag: Vec<f64> = cfgs
        .iter()
        .map(|&cfg| {
            let (ou, od) = cfg_occ(cfg, space.d);
            let lam: f64 = (0..space.d)
                .map(|k| p.d[k] * ((ou[k] as usize + od[k] as usize) as f64))
                .sum();
            p.scalar(lam)
        })
        .collect();
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] *= sdiag[i] * sdiag[j];
        }
    }
    let ev = jacobi_spectrum(&m, n);
    (ev[0], ev[n - 1])
}

#[test]
fn tpv_calibrate_matches_dense_spectrum() {
    let (h, space, ints) = load("lih_sto3g");
    let nelec = space.nup + space.ndn;
    let x0 = hf_determinant(space);
    let gamma = gamma_both(&ints, &h, &x0, space);
    let d = diagonal_one_particle(&ints, gamma, nelec);
    let mut p = ExpSumPrecond::from_diagonal(&d, space.nup, space.ndn, 0.1);

    let (lo0, hi0) = dense_spectrum_a(&h, space, &p, gamma);
    let norm_i_minus_a = (1.0 - lo0).abs().max((1.0 - hi0).abs());

    let cal = p.calibrate(&h, gamma, &x0, &CalibrateOpts::default());
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-30);
    assert!(
        rel(cal.lambda_min, lo0) < 5e-2 && rel(cal.lambda_max, hi0) < 5e-2,
        "calibrate [{:.6e}, {:.6e}] vs dense [{lo0:.6e}, {hi0:.6e}]",
        cal.lambda_min,
        cal.lambda_max
    );
    assert!(
        cal.lambda_min >= lo0 * (1.0 - 1e-12) && cal.lambda_max <= hi0 * (1.0 + 1e-12),
        "the estimate escaped the true spectrum: [{:.9e}, {:.9e}] vs dense [{lo0:.9e}, {hi0:.9e}]",
        cal.lambda_min,
        cal.lambda_max
    );

    let c_dense = (hi0 - lo0) / (hi0 + lo0);
    assert!(
        cal.c_raw <= c_dense + 1e-12,
        "c_raw {:.9} should not exceed the dense c {c_dense:.9}",
        cal.c_raw
    );
    assert!(
        cal.c > c_dense,
        "the margin failed to cover the inward bias: c = {:.6} but the dense c is {c_dense:.6}",
        cal.c
    );

    let (lo1, hi1) = dense_spectrum_a(&h, space, &p, gamma);
    let mid_before = 0.5 * (lo0 + hi0);
    let mid_after = 0.5 * (lo1 + hi1);
    assert!(
        (mid_after - 1.0).abs() < 2e-2,
        "post-calibration spec = [{lo1:.6e}, {hi1:.6e}]; midpoint {mid_after:.6} should be ~1"
    );
    assert!(
        (mid_after - 1.0).abs() * 20.0 < (mid_before - 1.0).abs(),
        "the eq-4.11 rescale barely moved the centre: {mid_before:.4} -> {mid_after:.4}"
    );
    assert!(
        rel(0.5 * (hi1 - lo1), cal.c_raw) < 2e-2,
        "post-calibration half-width {:.6e} should equal c_raw {:.6e}",
        0.5 * (hi1 - lo1),
        cal.c_raw
    );
    assert!(
        cal.c_raw < norm_i_minus_a,
        "the eq-4.11 rescale did not improve conditioning: c_raw {:.4} vs uncentred ‖I−A‖ \
         {norm_i_minus_a:.4}",
        cal.c_raw
    );
}
