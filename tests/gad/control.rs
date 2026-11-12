// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::C64;
use qntt::core::tangent::{TtTangent, tangent_basis};
use qntt::solvers::gad::{GadOpts, gad_c64};
use qntt::solvers::hess::{
    ModeOpts, hessian_grams_c64, hll_from_grams_c64, lowest_hessian_mode, weingarten_operator,
};
use qntt::{QnSpace, Scalar, TtVector};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::{
    deflate_radial_span, dense_sector_eigen, generic_complex_h, grad_exact_at, inf_scale,
    kick_along, radial_mode_at, randomized_state,
};

fn realified_coords(basis: &[TtTangent<C64>], w: &TtTangent<C64>) -> Vec<f64> {
    let n = basis.len();
    let mut y = vec![0.0f64; 2 * n];
    for (j, u) in basis.iter().enumerate() {
        let c = u.dot(w);
        y[j] = c.re;
        y[n + j] = -c.im;
    }
    y
}

fn times_i(y: &[f64]) -> Vec<f64> {
    let n = y.len() / 2;
    let mut z = vec![0.0f64; 2 * n];
    for j in 0..n {
        z[j] = y[n + j];
        z[n + j] = -y[j];
    }
    z
}

fn quad(m: &[f64], y: &[f64]) -> f64 {
    let n = y.len();
    let mut acc = 0.0;
    for i in 0..n {
        let mut row = 0.0;
        for j in 0..n {
            row += m[i * n + j] * y[j];
        }
        acc += y[i] * row;
    }
    acc
}

fn neg_count_after_deflation(m: &[f64], dim: usize, dirs: &[Vec<f64>], tol: f64) -> usize {
    let mut p: Vec<f64> = vec![0.0; dim * dim];
    for i in 0..dim {
        p[i * dim + i] = 1.0;
    }
    for d in dirs {
        for i in 0..dim {
            for j in 0..dim {
                p[i * dim + j] -= d[i] * d[j];
            }
        }
    }
    let mut pm = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let pik = p[i * dim + k];
            if pik == 0.0 {
                continue;
            }
            for j in 0..dim {
                pm[i * dim + j] += pik * m[k * dim + j];
            }
        }
    }
    let mut pmp = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let v = pm[i * dim + k];
            if v == 0.0 {
                continue;
            }
            for j in 0..dim {
                pmp[i * dim + j] += v * p[k * dim + j];
            }
        }
    }
    for i in 0..dim {
        for j in 0..i {
            let s = 0.5 * (pmp[i * dim + j] + pmp[j * dim + i]);
            pmp[i * dim + j] = s;
            pmp[j * dim + i] = s;
        }
    }
    let (evals, _) = dense_sector_eigen::<f64>(&pmp, dim);
    evals.iter().filter(|e| **e < -tol).count()
}

#[test]
fn gad_ctl_c64_truncated_rank_certificate_reading() {
    let space = QnSpace::new(4, 2, 1);
    let ham = generic_complex_h(space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC64B);
    let seed_state: TtVector<C64> = randomized_state(space, 1, 2, &mut rng);
    let xstar = crate::util::converged_minimizer(&ham, &seed_state);

    let v0 = {
        let ctx = weingarten_operator(&ham, &xstar);
        let (bl, br) = ctx.bases();
        let diag = ctx.a_diagonal();
        let seed = TtTangent::project(bl, br, &diag.to_vector());
        let (_verdict, w, _a) = qntt::solvers::hess::hessian_pd_probe_c64_witness(&ctx, &seed, 200);
        let radial = radial_mode_at(bl, br, bl);
        let mut v = w;
        deflate_radial_span(&radial, &mut v);
        let n = v.norm();
        v.scale_mut(C64::from_f64(1.0 / n));
        v
    };
    let x0 = kick_along(&v0, 5e-2);
    assert!(
        grad_exact_at(&ham, &x0).norm() > 1e-5,
        "the C64 truncated kick left the state critical — vacuous arm"
    );

    let opts = GadOpts {
        gradtol: 1e-9,
        max_steps: 300,
        k_davidson: 8,
        certify: true,
        probe_budget: 200,
        preserve_grid: true,
        ..GadOpts::default()
    };
    let (xe, info) = gad_c64(&ham, &x0, None, &opts);
    xe.check();

    let ctx = weingarten_operator(&ham, &xe);
    let (bl, br) = ctx.bases();
    let basis = tangent_basis(bl, br, true);
    let n = basis.len();
    assert!(n > 1, "degenerate tangent space at the endpoint (n = {n})");
    let (a, b) = hessian_grams_c64(&ctx, &basis);
    let hll = hll_from_grams_c64(&a, &b, n);
    let dim = 2 * n;

    let radial = radial_mode_at(bl, br, bl);
    let diag = ctx.a_diagonal();
    let scale = inf_scale(&diag);
    let op = |p: &TtTangent<C64>| {
        let mut q = TtTangent::project(bl, br, &p.to_vector());
        deflate_radial_span(&radial, &mut q);
        let mut hq = ctx.hess_apply(&q);
        deflate_radial_span(&radial, &mut hq);
        hq
    };
    let precond =
        |t: &mut TtTangent<C64>, theta: f64| qntt::solvers::davidson::precond(t, &diag, theta);
    let clean = |t: &mut TtTangent<C64>| {
        let mut q = TtTangent::project(bl, br, &t.to_vector());
        deflate_radial_span(&radial, &mut q);
        *t = q;
    };
    let mut s = TtTangent::project(bl, br, &diag.to_vector());
    deflate_radial_span(&radial, &mut s);
    let res = lowest_hessian_mode(
        op,
        precond,
        clean,
        &s,
        &ModeOpts {
            budget: 200,
            n_ritz: 1,
            warmup: 2,
            tol_rel: 1e-6,
            neg_early_exit: false,
            space_dim: 2 * n,
            scale,
        },
    )
    .expect("the deterministic seed collapsed");
    let mut w = res.witness;
    let wn = w.norm();
    w.scale_mut(C64::from_f64(1.0 / wn));

    let y = realified_coords(&basis, &w);
    let z = times_i(&y);
    let ynorm = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        (ynorm - 1.0).abs() <= 1e-8,
        "the tracked mode is not inside the radial-free tangent basis (‖coords‖ = {ynorm:.12}) — \
         either it carries radial content or the basis is not orthonormal"
    );
    let dot_yz: f64 = y.iter().zip(z.iter()).map(|(p, q)| p * q).sum();
    assert!(
        dot_yz.abs() <= 1e-12,
        "V and iV are not orthogonal in the realified metric (⟨y,z⟩ = {dot_yz:.3e})"
    );
    let qw = quad(&hll, &y);
    let qiw = quad(&hll, &z);
    let fresh_w = w.dot(&ctx.hess_apply(&w)).re;
    let mut iw = w.clone();
    iw.scale_mut(C64::new(0.0, 1.0));
    let fresh_iw = iw.dot(&ctx.hess_apply(&iw)).re;
    assert!(
        (qw - fresh_w).abs() <= 1e-8 * fresh_w.abs().max(1.0),
        "realified quadratic form on V ({qw:.12e}) disagrees with a fresh Re⟨V, (A+B)V⟩ \
         ({fresh_w:.12e}) — the coordinate convention, not the measurement, is what this arm \
         would otherwise be reporting"
    );
    assert!(
        (qiw - fresh_iw).abs() <= 1e-8 * fresh_iw.abs().max(1.0),
        "realified quadratic form on iV ({qiw:.12e}) disagrees with a fresh Re⟨iV, (A+B)iV⟩ \
         ({fresh_iw:.12e})"
    );

    let tol = 1e-9 * scale;
    let neg_all = neg_count_after_deflation(&hll, dim, &[], tol);
    let neg_v = neg_count_after_deflation(&hll, dim, std::slice::from_ref(&y), tol);
    let neg_viv = neg_count_after_deflation(&hll, dim, &[y.clone(), z.clone()], tol);
    println!(
        "GAD C64 TRUNCATED READING: ranks 1..=2, tangent dim_C {n} (dim_R {dim}) | negatives: \
         undeflated {neg_all}, after {{V}} {neg_v}, after {{V,iV}} {neg_viv} | Re⟨V,𝔥V⟩ \
         {fresh_w:+.6e} Re⟨iV,𝔥 iV⟩ {fresh_iw:+.6e} | driver: converged {} certified {} index \
         {:?} eig1 {:?} steps {} ‖grad‖ {:.2e} b_gain_max {:.3e}",
        info.converged,
        info.index1_certified,
        info.index_estimate,
        info.eig1,
        info.steps,
        info.gnorm,
        info.b_gain_max
    );

    assert!(
        neg_all >= neg_v && neg_all - neg_v <= 1,
        "interlacing violated deflating {{V}}: {neg_all} -> {neg_v}"
    );
    assert!(
        neg_v >= neg_viv && neg_v - neg_viv <= 1,
        "interlacing violated deflating iV on top of V: {neg_v} -> {neg_viv}"
    );

    if info.index1_certified {
        assert_eq!(
            info.index_estimate,
            Some(1),
            "certified index-1 without the matching index estimate"
        );
        assert_eq!(
            neg_viv, 0,
            "the driver certified index 1 at a truncated COMPLEX endpoint where the realified \
             dense Hessian still has {neg_viv} negative direction(s) after removing the full \
             complex span {{V, iV}} — probe 2 is reporting PD on a subspace that is not PD"
        );
    }
    if neg_v != neg_viv {
        println!(
            "GAD C64 TRUNCATED READING: the two readings DISCRIMINATE at this point ({{V}} leaves \
             {neg_v}, {{V,iV}} leaves {neg_viv}) — the open phase-1 question has a live \
             fixture here"
        );
    } else {
        println!(
            "GAD C64 TRUNCATED READING: the two readings agree at this point ({neg_v}); this \
             fixture does not discriminate them"
        );
    }
}
