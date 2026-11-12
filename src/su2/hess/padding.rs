// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::integrals::Integrals;
use crate::qn::Pooled;
use crate::round::RoundMode;
use crate::solvers::hess::{CurvatureReg, PadOpts, Padded, pad_weights, singles_pairs};
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::ops::dot;
use crate::su2::orbital::e_pq;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;

use super::{Su2WeingartenCtx, weingarten_operator};

fn same_irrep(x: &Su2TtVector<f64>) -> impl Fn(usize, usize) -> bool + '_ {
    move |i, a| x.space.pg.irrep_of(i) == x.space.pg.irrep_of(a)
}

/// The SU(2) twin of [`crate::solvers::hess::pad_singles`]: `y′ = normalize(x + ε·ẑ)`, `ẑ` the
/// singlet-CIS-weighted sum of the singlet images `E_ai x`, capped (pooled, reduced multiplets) and
/// normalized. Read the result through [`padded_context`].
///
/// # Panics
///
/// On a non-positive or non-finite `ε`, or an `occupied` list that is not ascending and in range.
pub fn pad_singles(
    x: &Su2TtVector<f64>,
    ints: &Integrals<f64>,
    occupied: &[usize],
    opts: &PadOpts,
) -> Padded<Su2TtVector<f64>> {
    assert!(
        opts.eps.is_finite() && opts.eps > 0.0,
        "su2 pad_singles: ε must be finite and > 0 (got {})",
        opts.eps
    );
    let pairs = singles_pairs(x.space.d, occupied, same_irrep(x));
    let w = pad_weights(ints, occupied, &pairs, opts.roots, true, opts.seed);
    let mut z: Option<Su2TtVector<f64>> = None;
    for (&(i, a, _), wi) in pairs.iter().zip(&w) {
        let img = e_pq(x, a, i, wi[0]);
        let mut acc = match z.take() {
            Some(z0) => &z0 + &img,
            None => img,
        };
        acc.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
        z = Some(acc);
    }
    let mut z = z.expect("su2 pad_singles: no representable single excitation");
    if !opts.cap.is_none() {
        z.round_mut(1e-12, opts.cap, RoundMode::Rel);
    }
    let nz = z.norm();
    assert!(
        nz > 0.0,
        "su2 pad_singles: every single-excitation image vanished"
    );
    z.scale_mut(opts.eps / nz);
    let mut y = x + &z;
    let ny = y.norm();
    y.scale_mut(1.0 / ny);
    Padded {
        state: y,
        eps: opts.eps,
    }
}

/// The SU(2) twin of [`crate::solvers::hess::padded_context`]: [`weingarten_operator`] at `y′`
/// with [`CurvatureReg::Hard`]`(max(f, 1e3·ε))`; returns the context and the floor it set.
pub fn padded_context<'h>(
    h: &'h Su2SparseHamiltonian<f64>,
    padded: &Padded<Su2TtVector<f64>>,
    f: f64,
) -> (Su2WeingartenCtx<'h, f64>, f64) {
    let floor = f.max(1e3 * padded.eps);
    (
        weingarten_operator(h, &padded.state).with_curvature(CurvatureReg::Hard(floor)),
        floor,
    )
}

/// The SU(2) twin of [`crate::solvers::hess::singles_completeness`], over the `nroots` lowest
/// SINGLET CIS roots: the minimum of `‖P Ê x‖ / ‖Ê x‖`, `Ê = Σ X_ai E_ai`.
pub fn singles_completeness(
    ctx: &Su2WeingartenCtx<'_, f64>,
    ints: &Integrals<f64>,
    occupied: &[usize],
    nroots: usize,
) -> f64 {
    let (bl, br) = ctx.shared_bases();
    let x: &Su2TtVector<f64> = bl;
    let pairs = singles_pairs(x.space.d, occupied, same_irrep(x));
    let raw: Vec<Su2TtVector<f64>> = pairs.iter().map(|&(i, a, _)| e_pq(x, a, i, 1.0)).collect();
    let proj: Vec<Su2TtTangent<f64>> = raw
        .iter()
        .map(|t| Su2TtTangent::project(bl, br, t))
        .collect();
    let n = raw.len();
    let mut gram = vec![0.0; n * n];
    for p in 0..n {
        for q in p..n {
            let g = dot(&raw[p], &raw[q]);
            gram[p * n + q] = g;
            gram[q * n + p] = g;
        }
    }
    crate::cis::cis_roots_occ(ints, occupied)
        .iter()
        .filter(|r| !r.triplet)
        .take(nroots)
        .map(|r| {
            let w: Vec<f64> = pairs.iter().map(|&(_, _, ia)| r.x[ia]).collect();
            let full: f64 = (0..n)
                .flat_map(|p| (0..n).map(move |q| (p, q)))
                .map(|(p, q)| w[p] * w[q] * gram[p * n + q])
                .sum::<f64>()
                .max(0.0)
                .sqrt();
            let mut v = proj[0].scale(0.0);
            for (k, t) in proj.iter().enumerate() {
                v.axpy(w[k], t);
            }
            v.norm() / full
        })
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factor::dense_self_adjoint_smallest;
    use crate::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
    use crate::solvers::hess::TdaDavidsonOpts;
    use crate::su2::core::Family;
    use crate::su2::hess::tda_lowest_davidson;
    use crate::su2::vector::{tt_csf, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::path::Path;

    fn seeds(ctx: &Su2WeingartenCtx<'_, f64>, n: usize, seed: u64) -> Vec<Su2TtTangent<f64>> {
        let (bl, br) = ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .map(|_| Su2TtTangent::project(bl, br, &tt_randn(bl.space, &bl.ranks, &mut rng)))
            .collect()
    }

    fn singles_ritz_min(h: &Su2SparseHamiltonian<f64>, x: &Su2TtVector<f64>, occ: &[usize]) -> f64 {
        let lambda = dot(x, &h.apply(x));
        let b: Vec<Su2TtVector<f64>> = singles_pairs(x.space.d, occ, |_, _| true)
            .iter()
            .map(|&(i, a, _)| {
                let e = e_pq(x, a, i, 1.0);
                &e - &(x * dot(x, &e))
            })
            .collect();
        let n = b.len();
        let hb: Vec<Su2TtVector<f64>> = b.iter().map(|v| h.apply(v)).collect();
        let mut g = vec![0.0; n * n];
        let mut m = vec![0.0; n * n];
        for p in 0..n {
            for q in 0..n {
                g[p * n + q] = dot(&b[p], &b[q]);
                m[p * n + q] = dot(&b[p], &hb[q]) - lambda * g[p * n + q];
            }
        }
        let (ge, gu) = dense_self_adjoint_smallest(&g, n, n);
        let gmax = ge.iter().fold(0.0f64, |a, &e| a.max(e));
        let w: Vec<Vec<f64>> = (0..n)
            .filter(|&j| ge[j] > 1e-10 * gmax)
            .map(|j| gu[j].iter().map(|u| u / ge[j].sqrt()).collect())
            .collect();
        let r = w.len();
        let mut mw = vec![0.0; r * r];
        for (a, wa) in w.iter().enumerate() {
            for (c, wc) in w.iter().enumerate() {
                mw[a * r + c] = (0..n)
                    .flat_map(|p| (0..n).map(move |q| (p, q)))
                    .map(|(p, q)| wa[p] * m[p * n + q] * wc[q])
                    .sum();
            }
        }
        dense_self_adjoint_smallest(&mw, r, 1).0[0]
    }

    #[test]
    fn su2_pad_singles_restores_the_singles_of_a_planted_reference() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
        let opts = IntegralsOptions {
            point_group: PointGroupPolicy::Ignore,
            ..IntegralsOptions::default()
        };
        let ints = load_fixture_dir::<f64>(&dir, &opts)
            .expect("lih_sto3g loads")
            .ints;
        use Family::{Empty as E, Occ as O};
        let hf: Su2TtVector<f64> = tt_csf(&[O, O, E, E, E, E]);
        let h = Su2SparseHamiltonian::new(hf.space, &ints.t, &ints.v, None, f64::EPSILON);
        let occ = [0usize, 1];
        let double = |a: usize| {
            let mut v = e_pq(&e_pq(&hf, a, 1, 1.0), a, 1, 1.0);
            let n = v.norm();
            v.scale_mut(1.0 / n);
            v
        };
        let mut x = &(&hf + &(&double(2) * 0.3)) + &(&double(3) * 0.2);
        x.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
        let nx = x.norm();
        x.scale_mut(1.0 / nx);
        let theta_s = singles_ritz_min(&h, &x, &occ);

        let ctx0 = weingarten_operator(&h, &x);
        let s0 = singles_completeness(&ctx0, &ints, &occ, 8);
        let tda = TdaDavidsonOpts::default();
        let t0 = tda_lowest_davidson(&ctx0, &seeds(&ctx0, 3, 0xBAD0), 1, &tda);
        let padded = pad_singles(
            &x,
            &ints,
            &occ,
            &PadOpts {
                cap: Pooled::NONE,
                ..PadOpts::default()
            },
        );
        let (ctx1, floor) = padded_context(&h, &padded, 0.0);
        let s1 = singles_completeness(&ctx1, &ints, &occ, 8);
        let t1 = tda_lowest_davidson(&ctx1, &seeds(&ctx1, 3, 0xBAD1), 1, &tda);
        println!(
            "su2 planted lih: completeness {s0:.3e} → {s1:.9}; TDA θ₁ {:.9} → {:.9}; singles Ritz \
             bound θ_S {theta_s:.9}; floor {floor:e}",
            t0.theta[0], t1.theta[0]
        );
        assert!(s0 <= 1e-10, "the planted reference holds singles: {s0:e}");
        assert_eq!(floor, 1e-3, "the padded context's floor");
        assert!(
            s1 >= 0.999,
            "padding left the singles out: completeness {s1}"
        );
        assert!(
            t1.theta[0] <= theta_s + 1e-6,
            "padded TDA θ₁ {} above the singles' Ritz bound {theta_s}",
            t1.theta[0]
        );
        assert!(
            t0.theta[0] > theta_s + 1e-3,
            "the unpadded TDA already reached the singles: θ₁ {} vs {theta_s}",
            t0.theta[0]
        );
    }
}
