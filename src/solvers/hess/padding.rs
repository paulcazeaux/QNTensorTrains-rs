// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::cis::{CisRoot, cis_roots_occ, virtuals};
use crate::core::tangent::TtTangent;
use crate::elementaryops::adag_i_a_j;
use crate::integrals::Integrals;
use crate::ops::dot;
use crate::orbital::{Orbital, Spin};
use crate::qn::Pooled;
use crate::round::RoundMode;
use crate::vector::TtVector;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::hamiltonian::SparseHamiltonian;

use super::{CurvatureReg, WeingartenCtx, weingarten_operator};

/// Options of [`pad_singles`] (both layers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PadOpts {
    /// The padding's weight `ε` in `y′ = normalize(y + ε·ẑ)`, `‖ẑ‖ = 1`.
    pub eps: f64,
    /// Pooled cap on the image combination before it is added (the padded width).
    /// [`Pooled::NONE`] keeps every image direction (exact, expensive).
    pub cap: Pooled,
    /// The number of lowest CIS roots weighting the images (each a random sign); `0` = one random
    /// weight per elementary image. Measured (c2h4 STO-3G, cap 64, pooled padded cap 64): CIS
    /// weighting puts the lowest TDA root 4.4 mHa above FCI, random weights 21 mHa.
    pub roots: usize,
    /// Seed of the random signs / weights.
    pub seed: u64,
}

impl Default for PadOpts {
    fn default() -> Self {
        PadOpts {
            eps: 1e-6,
            cap: Pooled(64),
            roots: 16,
            seed: 0x9AD5,
        }
    }
}

/// A padded reference: `y′` and the `ε` it was padded with (read by [`padded_context`]).
#[derive(Clone, Debug)]
pub struct Padded<V> {
    pub state: V,
    pub eps: f64,
}

pub(crate) fn singles_pairs(
    d: usize,
    occ: &[usize],
    same_irrep: impl Fn(usize, usize) -> bool,
) -> Vec<(usize, usize, usize)> {
    let virt = virtuals(d, occ);
    let nv = virt.len();
    occ.iter()
        .enumerate()
        .flat_map(|(ik, &i)| {
            virt.iter()
                .enumerate()
                .map(move |(ak, &a)| (i, a, ik * nv + ak))
        })
        .filter(|&(i, a, _)| same_irrep(i, a))
        .collect()
}

pub(crate) fn pad_weights(
    ints: &Integrals<f64>,
    occ: &[usize],
    pairs: &[(usize, usize, usize)],
    roots: usize,
    singlet_only: bool,
    seed: u64,
) -> Vec<[f64; 2]> {
    use rand::RngExt;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    if roots == 0 {
        return pairs
            .iter()
            .map(|_| [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)])
            .collect();
    }
    let cis: Vec<CisRoot> = cis_roots_occ(ints, occ)
        .into_iter()
        .filter(|r| !(singlet_only && r.triplet))
        .take(roots)
        .collect();
    let g: Vec<f64> = cis
        .iter()
        .map(|_| if rng.random_bool(0.5) { 1.0 } else { -1.0 })
        .collect();
    pairs
        .iter()
        .map(|&(_, _, ia)| {
            let mut w = [0.0; 2];
            for (r, gj) in cis.iter().zip(&g) {
                w[0] += gj * r.x[ia];
                w[1] += gj * if r.triplet { -r.x[ia] } else { r.x[ia] };
            }
            w
        })
        .collect()
}

fn same_irrep(x: &TtVector<f64>) -> impl Fn(usize, usize) -> bool + '_ {
    move |i, a| x.space.pg.irrep_of(i) == x.space.pg.irrep_of(a)
}

/// Pad `x` with its single excitations from the determinant doubly occupying `occupied` (module
/// doc): `y′ = normalize(x + ε·ẑ)`, `ẑ` the [`PadOpts::roots`]-CIS-weighted sum of the images
/// `E_ai x` (both spins), rounded at `1e-12` relative as it accumulates, then capped at
/// [`PadOpts::cap`] and normalized. Read the result through [`padded_context`].
///
/// # Panics
///
/// On a non-positive or non-finite `ε`, or an `occupied` list that is not ascending and in range.
pub fn pad_singles(
    x: &TtVector<f64>,
    ints: &Integrals<f64>,
    occupied: &[usize],
    opts: &PadOpts,
) -> Padded<TtVector<f64>> {
    assert!(
        opts.eps.is_finite() && opts.eps > 0.0,
        "pad_singles: ε must be finite and > 0 (got {})",
        opts.eps
    );
    let d = x.space.d;
    let pairs = singles_pairs(d, occupied, same_irrep(x));
    let w = pad_weights(ints, occupied, &pairs, opts.roots, false, opts.seed);
    let mut z: Option<TtVector<f64>> = None;
    for (&(i, a, _), wi) in pairs.iter().zip(&w) {
        for (s, &ws) in [Spin::Up, Spin::Dn].into_iter().zip(wi) {
            let img = adag_i_a_j(x, Orbital::new(a, s), Orbital::new(i, s), ws);
            let mut acc = match z.take() {
                Some(z0) => &z0 + &img,
                None => img,
            };
            acc.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
            z = Some(acc);
        }
    }
    let mut z = z.expect("pad_singles: no representable single excitation");
    if !opts.cap.is_none() {
        z.round_mut(1e-12, opts.cap, RoundMode::Rel);
    }
    let nz = z.norm();
    assert!(
        nz > 0.0,
        "pad_singles: every single-excitation image vanished"
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

/// The context to read a padded reference through (module doc): [`weingarten_operator`] at `y′`
/// with [`CurvatureReg::Hard`]`(max(f, 1e3·ε))`, which cuts `B` on the padded columns. Returns the
/// context and the floor it set.
pub fn padded_context<'h>(
    h: &'h SparseHamiltonian<f64>,
    padded: &Padded<TtVector<f64>>,
    f: f64,
) -> (WeingartenCtx<'h, f64>, f64) {
    let floor = f.max(1e3 * padded.eps);
    (
        weingarten_operator(h, &padded.state).with_curvature(CurvatureReg::Hard(floor)),
        floor,
    )
}

/// The singles completeness of `ctx`'s tangent space: the minimum over the `nroots` lowest CIS
/// roots (both spins) of `‖P Ê x‖ / ‖Ê x‖`, `Ê x` the root's excitation of `ctx`'s point and `P`
/// the tangent projector (`‖Ê x‖` from the Gram of the elementary images, no TT sum). `0` means
/// a CIS state that no tangent vector reaches; values near `1` that every root is represented
/// (necessary, not sufficient, for the TDA/RPA to resolve the singly excited states). Costs
/// `2·|occ|·nvirt` image builds, projections and their Gram.
pub fn singles_completeness(
    ctx: &WeingartenCtx<'_, f64>,
    ints: &Integrals<f64>,
    occupied: &[usize],
    nroots: usize,
) -> f64 {
    let (bl, br) = ctx.shared_bases();
    let x: &TtVector<f64> = bl;
    let d = x.space.d;
    let pairs = singles_pairs(d, occupied, same_irrep(x));
    let raw: Vec<[TtVector<f64>; 2]> = pairs
        .iter()
        .map(|&(i, a, _)| {
            [Spin::Up, Spin::Dn].map(|s| adag_i_a_j(x, Orbital::new(a, s), Orbital::new(i, s), 1.0))
        })
        .collect();
    let proj: Vec<[TtTangent<f64>; 2]> = raw
        .iter()
        .map(|p| p.each_ref().map(|t| TtTangent::project(bl, br, t)))
        .collect();
    let flat: Vec<&TtVector<f64>> = raw.iter().flatten().collect();
    let n = flat.len();
    let mut gram = vec![0.0; n * n];
    for p in 0..n {
        for q in p..n {
            let g = dot(flat[p], flat[q]);
            gram[p * n + q] = g;
            gram[q * n + p] = g;
        }
    }
    let roots = cis_roots_occ(ints, occupied);
    roots
        .iter()
        .take(nroots)
        .map(|r| {
            let sign = if r.triplet { -1.0 } else { 1.0 };
            let w: Vec<f64> = pairs
                .iter()
                .flat_map(|&(_, _, ia)| [r.x[ia], sign * r.x[ia]])
                .collect();
            let full: f64 = (0..n)
                .flat_map(|p| (0..n).map(move |q| (p, q)))
                .map(|(p, q)| w[p] * w[q] * gram[p * n + q])
                .sum::<f64>()
                .max(0.0)
                .sqrt();
            let mut v = proj[0][0].scale(0.0);
            for (k, pq) in proj.iter().enumerate() {
                v.axpy(w[2 * k], &pq[0]);
                v.axpy(w[2 * k + 1], &pq[1]);
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
    use crate::solvers::hess::{TdaDavidsonOpts, tda_lowest_davidson};
    use crate::vector::{tt_randn, tt_state_in};
    use std::path::Path;

    fn sites(d: usize, occ: &[usize]) -> Vec<bool> {
        (0..d).map(|k| occ.contains(&k)).collect()
    }

    fn seeds(ctx: &WeingartenCtx<'_, f64>, n: usize, seed: u64) -> Vec<TtTangent<f64>> {
        let (bl, br) = ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .map(|_| TtTangent::project(bl, br, &tt_randn(bl.space, &bl.ranks, &mut rng)))
            .collect()
    }

    fn singles_ritz_min(h: &SparseHamiltonian<f64>, x: &TtVector<f64>, occ: &[usize]) -> f64 {
        let lambda = dot(x, &(h * x));
        let pairs = singles_pairs(x.space.d, occ, |_, _| true);
        let b: Vec<TtVector<f64>> = pairs
            .iter()
            .flat_map(|&(i, a, _)| {
                [Spin::Up, Spin::Dn].map(|s| {
                    let e = adag_i_a_j(x, Orbital::new(a, s), Orbital::new(i, s), 1.0);
                    &e - &(x * dot(x, &e))
                })
            })
            .collect();
        let n = b.len();
        let hb: Vec<TtVector<f64>> = b.iter().map(|v| h * v).collect();
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
        let keep: Vec<usize> = (0..n).filter(|&j| ge[j] > 1e-10 * gmax).collect();
        let r = keep.len();
        let w: Vec<Vec<f64>> = keep
            .iter()
            .map(|&j| gu[j].iter().map(|u| u / ge[j].sqrt()).collect())
            .collect();
        let mut mw = vec![0.0; r * r];
        for (a, wa) in w.iter().enumerate() {
            for (c, wc) in w.iter().enumerate() {
                let mut s = 0.0;
                for p in 0..n {
                    for q in 0..n {
                        s += wa[p] * m[p * n + q] * wc[q];
                    }
                }
                mw[a * r + c] = s;
            }
        }
        dense_self_adjoint_smallest(&mw, r, 1).0[0]
    }

    #[test]
    fn pad_singles_restores_the_singles_of_a_planted_reference() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
        let opts = IntegralsOptions {
            point_group: PointGroupPolicy::Ignore,
            ..IntegralsOptions::default()
        };
        let load = load_fixture_dir::<f64>(&dir, &opts).expect("lih_sto3g loads");
        let h = load.hamiltonian(opts.eps);
        let space = load.space;
        let d = space.d;
        let occ = [0usize, 1];
        let det = |o: &[usize]| tt_state_in::<f64>(space, &sites(d, o), &sites(d, o));
        let mut x = &(&det(&[0, 1]) + &(&det(&[0, 2]) * 0.3)) + &(&det(&[0, 3]) * 0.2);
        x.round_mut(1e-12, Pooled::NONE, RoundMode::Rel);
        let nx = x.norm();
        x.scale_mut(1.0 / nx);
        let theta_s = singles_ritz_min(&h, &x, &occ);

        let ctx0 = weingarten_operator(&h, &x);
        let s0 = singles_completeness(&ctx0, &load.ints, &occ, 16);
        let tda = TdaDavidsonOpts::default();
        let t0 = tda_lowest_davidson(&ctx0, &seeds(&ctx0, 3, 0xBAD0), 1, &tda);

        let padded = pad_singles(
            &x,
            &load.ints,
            &occ,
            &PadOpts {
                cap: Pooled::NONE,
                ..PadOpts::default()
            },
        );
        let (ctx1, floor) = padded_context(&h, &padded, 0.0);
        let s1 = singles_completeness(&ctx1, &load.ints, &occ, 16);
        let t1 = tda_lowest_davidson(&ctx1, &seeds(&ctx1, 3, 0xBAD1), 1, &tda);
        println!(
            "planted lih: completeness {s0:.3e} → {s1:.9}; TDA θ₁ {:.9} → {:.9}; singles Ritz bound \
             θ_S {theta_s:.9}; floor {floor:e}",
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
