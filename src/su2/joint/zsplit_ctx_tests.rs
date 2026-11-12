// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::{HMUL_EXPLICIT_CALLS, Su2JointContext, su2_joint_context};
use crate::coeff::CoeffTensor;
use crate::round::RoundMode;
use crate::solvers::joint::{FcRoute, JointOpts, ZRoute};
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::{Su2RankGrid, Su2Space};
use crate::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::theta;
use crate::su2::transition1rdm::su2_transition_1rdm;
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

fn fixture() -> (Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
    let d = 5usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 3, 1);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0x2C_0500);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let (_l, mut x) = riemannian_cg(&h, &x0, &opts);
    for _ in 0..8 {
        let (bl, br) = tangent_bases(&x);
        let hx = h.apply(&x);
        let lam = crate::su2::hamiltonian::rayleigh_quotient(&h, &x);
        let g = Su2TtTangent::project(&bl, &br, &(&hx - &(&x * lam))).norm();
        if g < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(&h, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    assert!(x.check(), "zsplit ctx fixture failed check()");
    assert!(
        count_theta(&x) > 0,
        "θ ≡ 1 at this base point — every metric arm built on it is vacuous"
    );
    (h, x)
}

fn ctx_at<'h>(
    h: &'h Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    z_route: ZRoute,
) -> Su2JointContext<'h> {
    let opts = JointOpts {
        tt_grad_tol: 1e-4,
        orb_floor: 1e-8,
        pairs: None,
        z_route,
        b_general: false,
    };
    su2_joint_context(h, x, &opts).unwrap_or_else(|e| panic!("su2_joint_context refused: {e}"))
}

fn worst_and_scale(got: &CoeffTensor<f64>, want: &CoeffTensor<f64>, d: usize) -> (f64, f64) {
    let mut worst = 0.0f64;
    let mut scale = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            let (a, b) = (got.get(&[p, q]), want.get(&[p, q]));
            worst = worst.max((a - b).abs());
            scale = scale.max(b.abs());
        }
    }
    (worst, scale)
}

#[test]
fn zsplit_c5_z_lo_completes_the_residual() {
    let (h, x) = fixture();
    let ctx = ctx_at(&h, &x, ZRoute::Split);

    let z = ctx.normal_residual().clone();
    let z_lo = ctx.z_lo().clone();
    let hx = h.apply(&x);

    let scale = z.norm();
    assert!(scale > 1e-8, "the normal residual is zero — vacuous arm");
    assert!(z_lo.norm() > 1e-8, "z_lo is zero — vacuous arm");
    assert!(
        (&z_lo - &z).norm() > 1e-3 * scale,
        "z_lo coincides with z — the H·x half carries no weight here and the identity is trivial"
    );

    let recon = &hx + &z_lo;
    let err = (&recon - &z).norm();
    assert!(
        err <= 1e-11 * scale,
        "z ≠ H·x + z_lo: ‖Δ‖ = {err:e} > {:e} (‖z‖ = {scale:e})",
        1e-11 * scale
    );
}

#[test]
fn zsplit_c6_gamma_z_routes_agree_entrywise() {
    let (h, x) = fixture();
    let d = x.space.d;
    let split = ctx_at(&h, &x, ZRoute::Split);
    let explicit = ctx_at(&h, &x, ZRoute::Explicit);

    assert!(
        count_theta(explicit.normal_residual()) > 0,
        "θ ≡ 1 on the normal residual — this metric arm is vacuous"
    );

    let n_orb = split.pairs().len();
    assert!(n_orb > 0, "vacuous pair list");
    let mut rng = seeded(0xC6_01);
    let a: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    let im = split
        .kappa_images(&a)
        .expect("a nonzero pair direction must build κ̂ images");

    let (bl, br) = split.bases();
    let xi_amb = {
        let mut v: Su2TtVector<f64> = randomized_su2_state(x.space, 1, 2, &mut rng);
        v.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        Su2TtTangent::project(bl, br, &v).to_vector()
    };

    for (label, ket) in [("x", &x), ("kappa_x", &im.kx), ("iota_xi", &xi_amb)] {
        assert!(ket.norm() > 1e-8, "{label}: ket is zero — vacuous arm");
        let got = split.gamma_z(ket);
        let want = explicit.gamma_z(ket);
        let (worst, scale) = worst_and_scale(&got, &want, d);
        assert!(
            scale > 1e-6,
            "{label}: γ(z, ket) is numerically zero — vacuous arm"
        );
        let mut asym = 0.0f64;
        for p in 0..d {
            for q in 0..d {
                asym = asym.max((want.get(&[p, q]) - want.get(&[q, p])).abs());
            }
        }
        assert!(
            asym > 1e-3 * scale,
            "{label}: γ(z, ket) is near-symmetric (asym {asym:e} vs scale {scale:e}) — a \
             transposed H half would be invisible and this arm has no teeth"
        );
        assert!(
            worst <= 1e-10 * scale,
            "{label}: gamma_z Split vs Explicit worst |Δ| = {worst:e} > {:e}",
            1e-10 * scale
        );
    }

    let cached = split.gamma_zx().clone();
    let direct = split.gamma_z(&x);
    let (worst, scale) = worst_and_scale(&cached, &direct, d);
    assert!(scale > 1e-6, "γ(z,x) is numerically zero — vacuous arm");
    assert!(
        worst == 0.0,
        "gamma_zx() is not a faithful cache of gamma_z(&x): worst |Δ| = {worst:e}"
    );
}

#[test]
fn zsplit_c6_3_gamma_kappa_z_x_matches_materialized_twin() {
    let (h, x) = fixture();
    let d = x.space.d;
    let split = ctx_at(&h, &x, ZRoute::Split);
    let explicit = ctx_at(&h, &x, ZRoute::Explicit);
    let n_orb = split.pairs().len();
    assert!(n_orb > 0, "vacuous pair list");

    assert!(
        count_theta(explicit.normal_residual()) > 0,
        "θ ≡ 1 on the normal residual — this metric arm is vacuous"
    );

    let mut rng = seeded(0xC6_03);
    let mut dirs: Vec<Vec<f64>> = (0..n_orb)
        .map(|i| {
            let mut a = vec![0.0f64; n_orb];
            a[i] = 1.0;
            a
        })
        .collect();
    dirs.push((0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect());

    let mut nonvacuous = 0usize;
    for (idx, a) in dirs.iter().enumerate() {
        let im = split
            .kappa_images(a)
            .expect("a nonzero pair direction must build κ̂ images");
        let g1 = split.gamma_z(&im.kx);
        let got = split.gamma_kappa_z_x(&im, &g1);

        let kz = &im.kop * explicit.normal_residual();
        let want = su2_transition_1rdm(&kz, &x);

        let (worst, scale) = worst_and_scale(&got, &want, d);
        if scale > 1e-8 {
            nonvacuous += 1;
        }
        assert!(
            worst <= 1e-10 * scale.max(1e-8),
            "dir {idx}: gamma_kappa_z_x vs materialized γ(κ̂z, x) worst |Δ| = {worst:e} > {:e} \
             (scale {scale:e})",
            1e-10 * scale.max(1e-8)
        );
    }
    assert!(
        nonvacuous >= n_orb / 2,
        "γ(κ̂z, x) is numerically zero over most directions ({nonvacuous}/{}) — the arm is vacuous",
        dirs.len()
    );
}

#[test]
fn zsplit_site2_split_never_calls_project_hmul_explicit() {
    use std::sync::atomic::Ordering;

    let (h, x) = fixture();
    let split = ctx_at(&h, &x, ZRoute::Split);
    let explicit = ctx_at(&h, &x, ZRoute::Explicit);
    let n_orb = split.pairs().len();
    assert!(n_orb > 0, "vacuous pair list");

    let (bl, br) = split.bases();
    let mut rng = seeded(0x51_7E2);
    let xi = {
        let mut v: Su2TtVector<f64> = randomized_su2_state(x.space, 1, 2, &mut rng);
        v.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        Su2TtTangent::project(bl, br, &v)
    };
    assert!(xi.norm() > 1e-8, "the tangent leg of the probe is zero");
    let a: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    assert!(
        a.iter().any(|v| v.abs() > 1e-3),
        "the orbital leg of the probe is zero — `kappa_images` would return None and the site \
         would never be reached on EITHER route"
    );

    let before = HMUL_EXPLICIT_CALLS.load(Ordering::Relaxed);
    let (a_xi, a_c) = split.a_covector(&xi, &a, FcRoute::Sweep);
    let (b_xi, b_c) = split.b_covector(&xi, &a, FcRoute::Sweep);
    assert!(
        a_xi.norm() > 1e-8 && b_xi.norm() > 1e-8,
        "covector ξ-legs are vacuous — the applies may be short-circuiting before the site"
    );
    assert!(
        a_c.iter().chain(b_c.iter()).any(|v| v.abs() > 1e-8),
        "F_c rows are vacuous — nothing was exercised"
    );
    let after_split = HMUL_EXPLICIT_CALLS.load(Ordering::Relaxed);
    assert_eq!(
        after_split, before,
        "★ STANDING RULE VIOLATED: a ZRoute::Split covector called `project_hmul_explicit`, which \
         materializes the `H·κ̂x` train. No `z`-based probe can see this — that is why the arm \
         exists."
    );

    let _ = explicit.a_covector(&xi, &a, FcRoute::Sweep);
    assert!(
        HMUL_EXPLICIT_CALLS.load(Ordering::Relaxed) > after_split,
        "the counter never moves even on the Explicit route — the probe is broken and the Split \
         assertion above proves nothing"
    );
}
