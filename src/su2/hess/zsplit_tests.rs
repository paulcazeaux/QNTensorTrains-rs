// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::{project_hmul_env, project_kappa_hmul_env};
use crate::coeff::CoeffTensor;
use crate::rotation::orbital_pairs;
use crate::round::RoundMode;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::rayleigh_quotient as su2_rayleigh;
use crate::su2::orbital::kappa_operator;
use crate::su2::qn::{Su2RankGrid, Su2Space};
use crate::su2::riemannian::{normalize_mut, tangent_bases};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::theta;
use crate::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use crate::test_util::generic_truncated_tv;

fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn randomized_su2_state<R: Rng + ?Sized>(
    space: Su2Space,
    lo: usize,
    hi: usize,
    rng: &mut R,
) -> Su2TtVector<f64> {
    let d = space.d;
    let ranks: Vec<Su2RankGrid> = (0..=d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    su2_tt_randn(space, &ranks, rng)
}

fn count_theta(x: &Su2TtVector<f64>) -> usize {
    let space = x.space;
    let mut n = 0usize;
    for b in 0..space.d {
        for m in space.bond_sectors(b) {
            if x.rank(b, m) == 0 {
                continue;
            }
            for succ in space.bond_sectors(b + 1) {
                if x.rank(b + 1, succ) == 0 {
                    continue;
                }
                if (theta(m.two_s, succ.two_s) - 1.0).abs() > 1e-12 {
                    n += 1;
                }
            }
        }
    }
    n
}

fn random_kappa(d: usize, pg: crate::pointgroup::PointGroup, seed: u64) -> CoeffTensor<f64> {
    let mut rng = seeded(seed);
    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q) in &orbital_pairs(pg, d) {
        let val: f64 = rng.random_range(-0.3..0.3);
        k.set(&[p, q], val);
        k.set(&[q, p], -val);
    }
    k
}

struct Fixture {
    h: Su2SparseHamiltonian<f64>,
    x: Su2TtVector<f64>,
    base_l: std::sync::Arc<Su2TtVector<f64>>,
    base_r: std::sync::Arc<Su2TtVector<f64>>,
    lambda: f64,
}

fn fixture(d: usize, n: usize, two_s: usize, seed: u64) -> Fixture {
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, n, two_s);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    assert!(x.check(), "fixture: check() failed");
    let (base_l, base_r) = tangent_bases(&x);
    let lambda = su2_rayleigh(&h, &x);
    Fixture {
        h,
        x,
        base_l: std::sync::Arc::new(base_l),
        base_r: std::sync::Arc::new(base_r),
        lambda,
    }
}

fn worst_and_scale(got: &Su2TtTangent<f64>, want: &Su2TtTangent<f64>) -> (f64, f64) {
    assert_eq!(
        got.num_coeffs(),
        want.num_coeffs(),
        "tangent shape mismatch — a rank-grid bug, not a tolerance question"
    );
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for i in 0..want.num_coeffs() {
        let (a, b) = (got.coeff(i), want.coeff(i));
        worst = worst.max((a - b).abs());
        scale = scale.max(b.abs());
    }
    (worst, scale)
}

fn assert_frames_distinguishable(fx: &Fixture, dir: &Su2TtVector<f64>, label: &str) {
    let straight = Su2TtTangent::project(&fx.base_l, &fx.base_r, dir);
    let swapped = Su2TtTangent::project(&fx.base_r, &fx.base_l, dir);
    let (delta, scale) = worst_and_scale(&swapped, &straight);
    assert!(
        scale > 1e-8,
        "{label}: the reference projection is numerically zero — the whole arm is vacuous"
    );
    assert!(
        delta > 1e-6 * scale,
        "{label}: base_l/base_r are NOT metric-distinguishable here (worst |Δ| = {delta:e} vs \
         scale {scale:e}); a frame swap inside the kernel would be invisible and this arm \
         certifies nothing about the θ² placement"
    );
}

fn project_hmul_explicit_twin(
    h: &Su2SparseHamiltonian<f64>,
    base_l: &Su2TtVector<f64>,
    base_r: &Su2TtVector<f64>,
    lambda: f64,
    w: &Su2TtVector<f64>,
) -> Su2TtTangent<f64> {
    let hw = h.apply(w);
    let mut out = Su2TtTangent::project(base_l, base_r, &hw);
    out.axpy(-lambda, &Su2TtTangent::project(base_l, base_r, w));
    out
}

#[test]
fn zsplit_t2_project_hmul_env_matches_explicit() {
    for &(d, n, two_s, seed) in &[(5usize, 3usize, 1usize, 0x1201u64), (6, 4, 2, 0x1202)] {
        let fx = fixture(d, n, two_s, seed);
        let label = format!("d{d}/N{n}/2S{two_s}");

        let k_mat = random_kappa(d, fx.x.space.pg, seed ^ 0xBEEF);
        let kop = kappa_operator(fx.x.space, &k_mat);
        let w = &kop * &fx.x;
        assert!(w.norm() > 1e-8, "{label}: κ̂x is vacuous");

        let mut grids_differ = 0usize;
        for b in 1..d {
            for s in fx.x.space.bond_sectors(b) {
                if w.rank(b, s) != fx.base_l.rank(b, s) {
                    grids_differ += 1;
                }
            }
        }
        assert!(
            grids_differ > 0,
            "{label}: κ̂x shares base_l's interior rank grid — a workspace keyed on base_l would \
             slice identically and the C2.1 arm has NO teeth"
        );

        assert!(
            count_theta(&fx.x) > 0,
            "{label}: θ ≡ 1 on the base point — vacuous metric arm"
        );
        assert!(
            count_theta(&w) > 0,
            "{label}: θ ≡ 1 on the ambient ket — vacuous metric arm"
        );
        assert_frames_distinguishable(&fx, &fx.h.apply(&w), &format!("{label} T2"));

        for (arm, lambda) in [("lambda=0", 0.0f64), ("lambda!=0", fx.lambda)] {
            let got = project_hmul_env(&fx.h, &fx.base_l, &fx.base_r, lambda, &w);
            let want = project_hmul_explicit_twin(&fx.h, &fx.base_l, &fx.base_r, lambda, &w);
            let (worst, scale) = worst_and_scale(&got, &want);
            assert!(
                scale > 1e-8,
                "{label}/{arm}: the explicit twin is numerically zero — vacuous arm"
            );
            assert!(
                worst <= 1e-11 * scale,
                "{label}/{arm}: project_hmul_env vs explicit worst |Δ| = {worst:e} > {:e}",
                1e-11 * scale
            );
        }

        let with_tail = project_hmul_env(&fx.h, &fx.base_l, &fx.base_r, fx.lambda, &w);
        let no_tail = project_hmul_env(&fx.h, &fx.base_l, &fx.base_r, 0.0, &w);
        let (tail_delta, tail_scale) = worst_and_scale(&with_tail, &no_tail);
        assert!(
            fx.lambda.abs() > 1e-6,
            "{label}: λ is numerically zero — the tail arm is vacuous"
        );
        assert!(
            tail_delta > 1e-4 * tail_scale.max(1.0),
            "{label}: the `−λ·P(w)` tail barely moves the result (worst |Δ| = {tail_delta:e}) — \
             the λ ≠ 0 arm would pass with the tail dropped entirely"
        );
    }
}

#[test]
fn zsplit_t2_small_d_smoke() {
    for &(d, n, two_s) in &[(1usize, 1usize, 1usize), (2, 2, 0)] {
        let space = Su2Space::new(d, n, two_s);
        let (mut t, v) = generic_truncated_tv(d);
        t.set(&[0, 0], -0.7);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = seeded(0x1210 + d as u64);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        normalize_mut(&mut x);
        let (bl, br) = tangent_bases(&x);
        let (bl, br) = (std::sync::Arc::new(bl), std::sync::Arc::new(br));
        let lambda = su2_rayleigh(&h, &x);
        let got = project_hmul_env(&h, &bl, &br, lambda, &x);
        let want = project_hmul_explicit_twin(&h, &bl, &br, lambda, &x);
        let (worst, scale) = worst_and_scale(&got, &want);
        if d == 1 {
            assert_eq!(
                want.num_coeffs(),
                1,
                "d=1: expected a single tangent coefficient, got {}",
                want.num_coeffs()
            );
            assert_eq!(
                scale, 0.0,
                "d=1: the materialized twin is no longer structurally zero — re-derive what this \
                 arm certifies"
            );
            assert_eq!(
                worst, 0.0,
                "d=1: the two routes differ on a structurally zero arm"
            );
            continue;
        }
        assert!(
            scale > 1e-8,
            "d={d} smoke: the materialized twin is numerically zero (scale {scale:e}) — vacuous arm"
        );
        assert!(
            worst <= 1e-11 * scale,
            "d={d} smoke: worst |Δ| = {worst:e}, scale {scale:e}"
        );
    }
}

#[test]
fn zsplit_t3_project_kappa_hmul_env_matches_explicit() {
    for &(d, n, two_s, seed) in &[(5usize, 3usize, 1usize, 0x1301u64), (6, 4, 2, 0x1302)] {
        let fx = fixture(d, n, two_s, seed);
        let label = format!("d{d}/N{n}/2S{two_s}");

        let k_mat = random_kappa(d, fx.x.space.pg, seed ^ 0xC0DE);
        let kop = kappa_operator(fx.x.space, &k_mat);

        let hx = fx.h.apply(&fx.x);
        let khx = &kop * &hx;
        assert!(khx.norm() > 1e-8, "{label}: κ̂·H·x is vacuous");

        assert!(
            count_theta(&fx.x) > 0,
            "{label}: θ ≡ 1 on the base point — vacuous metric arm"
        );
        assert!(
            count_theta(&khx) > 0,
            "{label}: θ ≡ 1 on κ̂·H·x — vacuous metric arm"
        );
        assert_frames_distinguishable(&fx, &khx, &format!("{label} T3"));

        let want = Su2TtTangent::project(&fx.base_l, &fx.base_r, &khx);
        let got = project_kappa_hmul_env(&kop, &fx.h, &fx.base_l, &fx.base_r, &fx.x);
        let (worst, scale) = worst_and_scale(&got, &want);
        assert!(
            scale > 1e-8,
            "{label}: the explicit κ̂·H·x projection is numerically zero — vacuous arm"
        );
        assert!(
            worst <= 1e-10 * scale,
            "{label}: project_kappa_hmul_env vs explicit worst |Δ| = {worst:e} > {:e}",
            1e-10 * scale
        );

        let kx = &kop * &fx.x;
        let wrong = project_hmul_env(&fx.h, &fx.base_l, &fx.base_r, 0.0, &kx);
        let (comm, comm_scale) = worst_and_scale(&wrong, &want);
        assert!(
            comm > 1e-4 * comm_scale.max(1.0),
            "{label}: P(κ̂·H·x) and P(H·κ̂·x) agree to {comm:e} at this fixture — the commutator is \
             degenerate here and the C4.1 arm has no teeth"
        );
    }
}
