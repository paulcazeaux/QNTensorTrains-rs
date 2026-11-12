// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::panic::AssertUnwindSafe;

use qntt::integrals::Integrals;
use qntt::orbopt::{
    FrozenOrbExit, FrozenOrbInfo, FrozenOrbMethod, FrozenOrbOpts, FrozenTrOpts, Rdm12,
    energy_from_rdms, frozen_orbital_gradient, frozen_orbital_hessian, frozen_orbital_step,
    generalized_fock,
};
use qntt::rotation::{
    compose_rotations, expm_antisym, givens, identity_matrix, orbital_gradient,
    orthogonality_defect, rotate_integrals,
};
use qntt::solvers::riemannian::normalize_mut;
use qntt::{
    CoeffTensor, Irrep, PointGroup, QnSpace, SparseHamiltonian, TtVector, max_ranks,
    rayleigh_quotient, tt_randn, tt_state,
};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::optimize::capped_randn;
use crate::util::*;

const E0: f64 = 0.37;

fn ints_of(space: QnSpace, t: CoeffTensor<f64>, v: CoeffTensor<f64>) -> Integrals<f64> {
    Integrals::<f64> {
        t,
        v,
        e0: E0,
        norb: space.d,
        nelec: space.n(),
        ms2: space.nup as i64 - space.ndn as i64,
        orbsym: Vec::new(),
        isym: None,
    }
}

struct Fx {
    label: &'static str,
    space: QnSpace,
    ints: Integrals<f64>,
    x: TtVector<f64>,
    k: Vec<f64>,
    kappa: Vec<f64>,
}

fn fixture(label: &'static str, d: usize, nup: usize, ndn: usize, seed: u64) -> Fx {
    let space = QnSpace::new(d, nup, ndn);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let t = random_sym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    let k = random_antisym_flat(d, &mut rng);
    let kappa = scale_flat(&random_antisym_flat(d, &mut rng), 0.3);
    let x = capped_randn(space, 1, 2, seed ^ 0x0DD5_EED5);
    Fx {
        label,
        space,
        ints: ints_of(space, t, v),
        x,
        k,
        kappa,
    }
}

fn fixtures() -> Vec<Fx> {
    vec![
        fixture("d=4 (2,1)", 4, 2, 1, 0x6_0421),
        fixture("d=4 (2,0)", 4, 2, 0, 0x6_0420),
        fixture("d=5 (2,2)", 5, 2, 2, 0x6_0522),
    ]
}

fn asym_fixture() -> Fx {
    let (d, nup, ndn, seed) = (4usize, 2usize, 1usize, 0x6_A5A5u64);
    let space = QnSpace::new(d, nup, ndn);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let t = random_asym_t(d, &mut rng);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for e in v.data.iter_mut() {
        *e = rng.sample::<f64, _>(StandardNormal);
    }
    let k = random_antisym_flat(d, &mut rng);
    let kappa = scale_flat(&random_antisym_flat(d, &mut rng), 0.3);
    let x = capped_randn(space, 1, 2, seed ^ 0x0DD5_EED5);
    Fx {
        label: "d=4 (2,1) asymmetric (t, v)",
        space,
        ints: ints_of(space, t, v),
        x,
        k,
        kappa,
    }
}

fn e_rq(space: QnSpace, ints: &Integrals<f64>, u: &[f64], x: &TtVector<f64>) -> f64 {
    let r = rotate_integrals(ints, u);
    let h = SparseHamiltonian::new(space, &r.t, &r.v, None, f64::EPSILON);
    rayleigh_quotient(&h, x)
}

fn sector_amplitudes(x: &TtVector<f64>, dets: &[(Vec<bool>, Vec<bool>)]) -> (Vec<f64>, f64) {
    let xd = x.to_dense();
    let c: Vec<f64> = dets.iter().map(|(u, d)| xd[basis_index(u, d)]).collect();
    let n2 = dot_vec(&c, &c);
    (c, n2)
}

fn e_jw(
    d: usize,
    ints: &Integrals<f64>,
    u: &[f64],
    dets: &[(Vec<bool>, Vec<bool>)],
    c: &[f64],
) -> f64 {
    let r = rotate_integrals(ints, u);
    let hd = dense_hamiltonian_matrix(d, &r.t, &r.v, dets);
    dot_vec(c, &mat_vec(&hd, c))
}

fn fro(g: &CoeffTensor<f64>) -> f64 {
    g.data.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn pair_norm(g: &CoeffTensor<f64>) -> f64 {
    let d = g.d;
    let mut s = 0.0f64;
    for p in 0..d {
        for q in 0..p {
            s += g.data[p * d + q] * g.data[p * d + q];
        }
    }
    s.sqrt()
}

fn max_abs_t(g: &CoeffTensor<f64>) -> f64 {
    g.data.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn dir_of(d: usize, g: &CoeffTensor<f64>, kappa: &[f64]) -> f64 {
    let mut s = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            s += 0.5 * g.get(&[p, q]) * kappa[p * d + q];
        }
    }
    s
}

fn dist_from_identity(u: &[f64], d: usize) -> f64 {
    let id = identity_matrix(d);
    u.iter()
        .zip(&id)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

fn truncated_cells(x: &TtVector<f64>) -> usize {
    let caps = max_ranks(&x.space);
    (1..x.space.d)
        .map(|b| {
            x.space
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| x.ranks[b][s] < caps[b][s])
                .count()
        })
        .sum()
}

fn exit_label(e: FrozenOrbExit) -> &'static str {
    match e {
        FrozenOrbExit::GradTol => "grad-tol",
        FrozenOrbExit::MaxIter => "max-iter",
        FrozenOrbExit::LineSearch => "line-search",
        FrozenOrbExit::AlreadyCritical => "already-critical",
    }
}

fn is_already_critical(info: &FrozenOrbInfo) -> bool {
    matches!(info.exit, FrozenOrbExit::AlreadyCritical)
}

fn print_info(tag: &str, info: &FrozenOrbInfo) {
    println!(
        "{tag}: exit {}, iters {}, n_evals {}, E {:.15} → {:.15} (δ = {:.3e}), ‖g‖ {:.3e} → \
         {:.3e}, kappa_norm {:.3e}",
        exit_label(info.exit),
        info.iters,
        info.n_evals,
        info.e_before,
        info.e_after,
        info.e_before - info.e_after,
        info.g_before,
        info.g_after,
        info.kappa_norm
    );
}

fn panic_text(f: impl FnOnce()) -> String {
    let payload = std::panic::catch_unwind(AssertUnwindSafe(f))
        .expect_err("expected a refusal panic, the call returned normally");
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        String::from("<non-string panic payload>")
    }
}

fn assert_refused(f: impl FnOnce(), needles: &[&str], what: &str) {
    let msg = panic_text(f);
    for n in needles {
        assert!(
            msg.contains(n),
            "{what}: the refusal message must contain {n:?}, got: {msg}"
        );
    }
}

fn assert_fixture_floors(fx: &Fx, rdm: &Rdm12) {
    let d = fx.space.d;
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;

    let trunc = truncated_cells(&fx.x);
    let h = SparseHamiltonian::new(fx.space, &fx.ints.t, &fx.ints.v, None, f64::EPSILON);
    let g_ref = orbital_gradient(&h, &fx.x);
    let mut delta = 0.0f64;
    for a in 0..d {
        for b in 0..d {
            for e in 0..d {
                delta += 0.5 * fx.ints.v.get(&[a, b, b, e]) * rdm.gamma1.get(&[a, e]);
            }
        }
    }
    let docc = (0..d)
        .map(|i| rdm.gamma2[idx(i, i, i, i)] - rdm.gamma1.get(&[i, i]))
        .fold(f64::NEG_INFINITY, f64::max);
    let docc_abs = (0..d)
        .map(|i| (rdm.gamma2[idx(i, i, i, i)] - rdm.gamma1.get(&[i, i])).abs())
        .fold(0.0f64, f64::max);
    println!(
        "{}: {trunc} truncated interior cells; ‖g‖_F = {:.3e}; δ-term = {delta:.3e}; \
         max_i(Γ₂[iiii] − γ[ii]) = {docc:.3e}",
        fx.label,
        fro(&g_ref)
    );
    assert!(
        trunc > 0,
        "{}: the state is not truncated anywhere",
        fx.label
    );
    assert!(
        fro(&g_ref) > 1e-3,
        "{}: the state is (near-)critical, ‖g‖_F = {:e}: arm is vacuous",
        fx.label,
        fro(&g_ref)
    );
    assert!(
        delta.abs() > 1e-3,
        "{}: the δ_bc term contributes only {delta:e}: a missing δ correction would pass",
        fx.label
    );
    if fx.space.nup >= 1 && fx.space.ndn >= 1 {
        assert!(
            docc > 1e-3,
            "{}: no double occupancy (max 2⟨n↑n↓⟩ = {docc:e}): the on-site channel is not exercised",
            fx.label
        );
    } else {
        assert!(
            docc_abs <= 1e-13,
            "{}: a single-channel state carries double occupancy {docc_abs:e}",
            fx.label
        );
    }
}

fn check_energy_three_ways(fx: &Fx, rdm: &Rdm12, u: &[f64], ulabel: &str) -> f64 {
    let d = fx.space.d;
    let dets = dets_for(d, fx.space.nup, fx.space.ndn);
    let (c, n2) = sector_amplitudes(&fx.x, &dets);
    assert!(
        (n2 - 1.0).abs() <= 1e-13,
        "{}: the determinant list does not carry x (Σ c² = {n2})",
        fx.label
    );
    let e_rdm = energy_from_rdms(&rotate_integrals(&fx.ints, u), rdm);
    let e_tt = e_rq(fx.space, &fx.ints, u, &fx.x);
    let e_dense = e_jw(d, &fx.ints, u, &dets, &c);
    let tol = 1e-12 * e_rdm.abs().max(1.0);
    let devs = [
        (e_rdm - e_tt).abs(),
        (e_rdm - e_dense).abs(),
        (e_tt - e_dense).abs(),
    ];
    let worst = devs.iter().fold(0.0f64, |m, v| m.max(*v));
    println!(
        "tord_g1 {} U={ulabel}: E_rdm = {e_rdm:.15}, E_rq = {e_tt:.15}, E_jw = {e_dense:.15}; \
         worst {worst:.1e} (tol {tol:.1e}, margin {:.1}×)",
        fx.label,
        tol / worst.max(1e-300)
    );
    assert!(
        worst <= tol,
        "{} U={ulabel}: energy identity fails: |rdm−rq| = {:e}, |rdm−jw| = {:e}, |rq−jw| = {:e} \
         (tol {tol:e})",
        fx.label,
        devs[0],
        devs[1],
        devs[2]
    );
    e_rdm
}

fn check_traces(fx: &Fx, rdm: &Rdm12) {
    let d = fx.space.d;
    let n = fx.space.n() as f64;
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    assert_eq!(rdm.d, d, "{}: rdm.d", fx.label);
    assert_eq!(rdm.n, fx.space.nup + fx.space.ndn, "{}: rdm.n", fx.label);
    assert_eq!(rdm.gamma1.rank, 2, "{}: gamma1 is rank 2", fx.label);
    assert_eq!(rdm.gamma1.d, d, "{}: gamma1 is d×d", fx.label);
    assert_eq!(
        rdm.gamma2.len(),
        d * d * d * d,
        "{}: gamma2 is d⁴",
        fx.label
    );

    let tr: f64 = (0..d).map(|i| rdm.gamma1.get(&[i, i])).sum();
    assert!(
        (tr - n).abs() <= 1e-13 * n,
        "{}: tr γ = {tr} ≠ N = {n}",
        fx.label
    );
    let gscale = max_abs_t(&rdm.gamma1).max(1.0);
    let mut worst = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            let s: f64 = (0..d).map(|k| rdm.gamma2[idx(i, j, k, k)]).sum();
            let dev = (s - n * rdm.gamma1.get(&[i, j])).abs();
            worst = worst.max(dev);
            assert!(
                dev <= 1e-12 * n * gscale,
                "{}: Σ_k Γ₂[{i},{j},k,k] = {s:e} ≠ N·γ[{i},{j}] = {:e}",
                fx.label,
                n * rdm.gamma1.get(&[i, j])
            );
        }
    }
    println!(
        "tord_g1 {}: tr γ − N = {:.1e}; worst |Σ_k Γ₂[i,j,k,k] − Nγ[i,j]| = {worst:.1e}",
        fx.label,
        tr - n
    );
}

#[test]
fn tord_g1_energy_identity_three_ways() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        assert_fixture_floors(&fx, &rdm);
        check_traces(&fx, &rdm);

        let u_rand = expm_antisym(&fx.k, d);
        let u_giv = givens(d, 1, 0.7);
        let e_i = check_energy_three_ways(&fx, &rdm, &identity_matrix(d), "I");
        let e_r = check_energy_three_ways(&fx, &rdm, &u_rand, "expm(K)");
        let e_g = check_energy_three_ways(&fx, &rdm, &u_giv, "givens(1,0.7)");
        assert!(
            (e_i - e_r).abs() > 1e-3 && (e_i - e_g).abs() > 1e-3,
            "{}: the rotations do not move the energy (|ΔE| = {:e}, {:e}): the U arms are vacuous",
            fx.label,
            (e_i - e_r).abs(),
            (e_i - e_g).abs()
        );
    }
}

#[test]
fn tord_g1_zero_rank_interior_sectors() {
    let space = QnSpace::new(4, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6_1020);
    let t = random_sym_t(4, &mut rng);
    let v = random_sym_v(4, &mut rng);
    let mut xrng = Xoshiro256PlusPlus::seed_from_u64(0xA1_0201);
    let mut x: TtVector<f64> = randomized_state(space, 0, 2, &mut xrng);
    let zero_sectors: usize = (1..space.d)
        .map(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| x.ranks[b][s] == 0)
                .count()
        })
        .sum();
    let n0 = x.norm();
    println!("tord_g1 zero-rank: {zero_sectors} zero-rank interior sectors, ‖x‖ before = {n0:e}");
    assert!(
        zero_sectors > 0,
        "no zero-rank interior sector: arm is vacuous"
    );
    assert!(n0 > 1e-3, "state is (numerically) zero: ‖x‖ = {n0:e}");
    normalize_mut(&mut x);
    let fx = Fx {
        label: "d=4 (2,1) zero-rank sectors",
        space,
        ints: ints_of(space, t, v),
        x,
        k: vec![0.0; 16],
        kappa: vec![0.0; 16],
    };
    let rdm = Rdm12::of(&fx.x);
    check_traces(&fx, &rdm);
    check_energy_three_ways(&fx, &rdm, &identity_matrix(4), "I");
}

#[test]
fn tord_g1_asymmetric_integrals() {
    let fx = asym_fixture();
    let d = fx.space.d;
    let mut asym_v = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    asym_v = asym_v
                        .max((fx.ints.v.get(&[i, j, k, l]) - fx.ints.v.get(&[k, l, i, j])).abs())
                        .max((fx.ints.v.get(&[i, j, k, l]) - fx.ints.v.get(&[j, i, l, k])).abs());
                }
            }
        }
    }
    assert!(
        asym_v > 0.1,
        "the asymmetric fixture is symmetric ({asym_v:e})"
    );
    let rdm = Rdm12::of(&fx.x);
    check_traces(&fx, &rdm);
    let e_i = check_energy_three_ways(&fx, &rdm, &identity_matrix(d), "I");
    let e_r = check_energy_three_ways(&fx, &rdm, &expm_antisym(&fx.k, d), "expm(K)");
    let e_g = check_energy_three_ways(&fx, &rdm, &givens(d, 1, 0.7), "givens(1,0.7)");
    assert!(
        (e_i - e_r).abs() > 1e-3 && (e_i - e_g).abs() > 1e-3,
        "asymmetric fixture: the rotations do not move the energy"
    );
}

const H_LADDER: [f64; 5] = [1.6e-2, 8e-3, 4e-3, 2e-3, 1e-3];

fn fd_ladder(e: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
    let dd: Vec<f64> = H_LADDER
        .iter()
        .map(|&h| (e(h) - e(-h)) / (2.0 * h))
        .collect();
    let rr: Vec<f64> = (1..dd.len())
        .map(|i| (4.0 * dd[i] - dd[i - 1]) / 3.0)
        .collect();
    (dd, rr)
}

fn assert_fd_matches(tag: &str, dir: f64, dd: &[f64], rr: &[f64]) {
    let scale = dir.abs().max(1.0);
    for (i, &h) in H_LADDER.iter().enumerate() {
        let r = if i == 0 {
            String::from("—")
        } else {
            format!("{:.2e}", (rr[i - 1] - dir).abs())
        };
        println!(
            "{tag}: h = {h:.1e}: |D − dir| = {:.2e}, |R − dir| = {r}",
            (dd[i] - dir).abs()
        );
    }
    for i in 2..H_LADDER.len() {
        let dev = (rr[i - 1] - dir).abs();
        assert!(
            dev <= 1e-6 * scale,
            "{tag}: Richardson rung h = {:e} misses dir = {dir} by {dev:e} (plateau: a wrong \
             gradient, not a stencil error)",
            H_LADDER[i]
        );
    }
    let last = rr[rr.len() - 1];
    assert!(
        (last - dir).abs() <= 1e-9 * scale,
        "{tag}: Richardson FD at h = 1e-3 = {last} vs ½Σgκ = {dir}: |diff| = {:e}",
        (last - dir).abs()
    );
}

fn check_gradient_at(fx: &Fx, rdm: &Rdm12, u: &[f64], ulabel: &str, left_negative: bool) {
    let d = fx.space.d;
    let tag = format!("tord_g2 {} U={ulabel}", fx.label);
    let ints_u = rotate_integrals(&fx.ints, u);
    let g = frozen_orbital_gradient(&ints_u, rdm);
    assert_eq!((g.rank, g.d), (2, d), "{tag}: g is d×d");

    for p in 0..d {
        assert!(
            g.get(&[p, p]) == 0.0,
            "{tag}: diagonal g[{p},{p}] = {:e} ≠ 0",
            g.get(&[p, p])
        );
        for q in 0..d {
            assert!(
                g.get(&[p, q]) == -g.get(&[q, p]),
                "{tag}: g not exactly antisymmetric at ({p},{q}): {:e} vs {:e}",
                g.get(&[p, q]),
                g.get(&[q, p])
            );
        }
    }

    let y = generalized_fock(&ints_u, rdm);
    assert_eq!(y.len(), d * d, "{tag}: Y is d×d");
    let yscale = y.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for p in 0..d {
        for q in 0..d {
            if p == q {
                continue;
            }
            let want = y[q * d + p] - y[p * d + q];
            assert!(
                (g.get(&[p, q]) - want).abs() <= 1e-13 * yscale,
                "{tag}: g[{p},{q}] = {:e} ≠ Y[{q},{p}] − Y[{p},{q}] = {want:e}",
                g.get(&[p, q])
            );
        }
    }

    let h = SparseHamiltonian::new(fx.space, &ints_u.t, &ints_u.v, None, f64::EPSILON);
    let g_ref = orbital_gradient(&h, &fx.x);
    let gscale = max_abs_t(&g_ref).max(1.0);
    let mut worst = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            worst = worst.max((g.get(&[p, q]) - g_ref.get(&[p, q])).abs());
        }
    }
    println!(
        "{tag}: ‖g‖_F = {:.3e}; max|g − orbital_gradient| = {worst:.1e} (tol {:.1e})",
        fro(&g),
        1e-11 * gscale
    );
    assert!(
        fro(&g_ref) > 1e-3,
        "{tag}: ‖g_ref‖_F = {:e}: arm is vacuous",
        fro(&g_ref)
    );
    assert!(
        worst <= 1e-11 * gscale,
        "{tag}: frozen_orbital_gradient vs orbital_gradient: max dev {worst:e} (scale {gscale:e})"
    );

    let dir = dir_of(d, &g, &fx.kappa);
    assert!(
        dir.abs() > 1e-3,
        "{tag}: |dir| = {dir:e}: FD arm is vacuous"
    );
    let body = |hh: f64| {
        let w = expm_antisym(&scale_flat(&fx.kappa, -hh), d);
        e_rq(fx.space, &fx.ints, &compose_rotations(u, &w, d), &fx.x)
    };
    let (dd, rr) = fd_ladder(body);
    assert_fd_matches(&tag, dir, &dd, &rr);

    if left_negative {
        let left = |hh: f64| {
            let w = expm_antisym(&scale_flat(&fx.kappa, -hh), d);
            e_rq(fx.space, &fx.ints, &compose_rotations(&w, u, d), &fx.x)
        };
        let (_, rl) = fd_ladder(left);
        let miss = (rl[rl.len() - 1] - dir).abs();
        println!("{tag}: left-frame FD misses the body-frame dir by {miss:.3e}");
        assert!(
            miss > 1e-3 * dir.abs().max(1.0),
            "{tag}: the left-frame derivative matches the body-frame one ({miss:e}): the U_r rung \
             does not discriminate the frame"
        );
    }
}

#[test]
fn tord_g2_gradient_matches_orbital_gradient_and_fd() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        assert_fixture_floors(&fx, &rdm);
        check_gradient_at(&fx, &rdm, &identity_matrix(d), "I", false);
        let u_r = expm_antisym(&scale_flat(&fx.k, 0.5), d);
        check_gradient_at(&fx, &rdm, &u_r, "U_r", true);
    }
}

#[test]
fn tord_g2_asymmetric_integrals_fd() {
    let fx = asym_fixture();
    let d = fx.space.d;
    let rdm = Rdm12::of(&fx.x);
    let u_r = expm_antisym(&scale_flat(&fx.k, 0.5), d);
    for (u, ulabel) in [(identity_matrix(d), "I"), (u_r, "U_r")] {
        let tag = format!("tord_g2 {} U={ulabel}", fx.label);
        let ints_u = rotate_integrals(&fx.ints, &u);
        let g = frozen_orbital_gradient(&ints_u, &rdm);
        for p in 0..d {
            assert!(g.get(&[p, p]) == 0.0, "{tag}: non-zero diagonal");
            for q in 0..d {
                assert!(
                    g.get(&[p, q]) == -g.get(&[q, p]),
                    "{tag}: not exactly antisymmetric"
                );
            }
        }
        let dir = dir_of(d, &g, &fx.kappa);
        assert!(dir.abs() > 1e-3, "{tag}: |dir| = {dir:e}: vacuous");
        let body = |hh: f64| {
            let w = expm_antisym(&scale_flat(&fx.kappa, -hh), d);
            e_rq(fx.space, &fx.ints, &compose_rotations(&u, &w, d), &fx.x)
        };
        let (dd, rr) = fd_ladder(body);
        assert_fd_matches(&tag, dir, &dd, &rr);
    }
}

fn check_step_common(
    tag: &str,
    fx: &Fx,
    ints: &Integrals<f64>,
    rdm: &Rdm12,
    opts: &FrozenOrbOpts,
    u: &[f64],
    info: &FrozenOrbInfo,
) -> f64 {
    let d = fx.space.d;
    assert_eq!(u.len(), d * d, "{tag}: U is d×d");

    let e0_rdm = energy_from_rdms(ints, rdm);
    assert!(
        info.e_before.to_bits() == e0_rdm.to_bits(),
        "{tag}: e_before = {} is not energy_from_rdms(ints) = {e0_rdm} bitwise",
        info.e_before
    );
    let e1_rdm = energy_from_rdms(&rotate_integrals(ints, u), rdm);
    assert!(
        info.e_after.to_bits() == e1_rdm.to_bits(),
        "{tag}: e_after = {} is not energy_from_rdms(rotate_integrals(ints, U)) = {e1_rdm} bitwise",
        info.e_after
    );
    let e0_tt = e_rq(fx.space, ints, &identity_matrix(d), &fx.x);
    let e1_tt = e_rq(fx.space, ints, u, &fx.x);
    let tol = 1e-12 * info.e_before.abs().max(1.0);
    assert!(
        (info.e_before - e0_tt).abs() <= tol,
        "{tag}: e_before vs ⟨x|H(ints)|x⟩: {:e}",
        (info.e_before - e0_tt).abs()
    );
    assert!(
        (info.e_after - e1_tt).abs() <= tol,
        "{tag}: e_after vs ⟨x|H(rotate(ints,U))|x⟩: {:e} (a Uᵀ/u[new][old] return misses here)",
        (info.e_after - e1_tt).abs()
    );
    assert!(
        info.e_after <= info.e_before,
        "{tag}: the step raised the energy: {} → {}",
        info.e_before,
        info.e_after
    );
    if info.iters >= 1 {
        assert!(
            info.e_after < info.e_before,
            "{tag}: {} accepted steps but no strict decrease",
            info.iters
        );
    }

    let orth = orthogonality_defect(u, d);
    let det = mat_det(&flat_to_nested(u, d));
    assert!(orth <= 1e-13, "{tag}: max|UᵀU − I| = {orth:e}");
    assert!((det - 1.0).abs() <= 1e-12, "{tag}: det U = {det}");

    let dist = dist_from_identity(u, d);
    assert!(
        info.kappa_norm.is_finite() && info.kappa_norm >= 0.0,
        "{tag}: kappa_norm = {}",
        info.kappa_norm
    );
    assert!(
        dist <= std::f64::consts::SQRT_2 * info.kappa_norm * (1.0 + 1e-10) + 1e-14,
        "{tag}: ‖U − I‖_F = {dist:e} exceeds √2·kappa_norm = {:e}: kappa_norm is not the path length",
        std::f64::consts::SQRT_2 * info.kappa_norm
    );
    if info.iters >= 1 {
        assert!(dist > 0.0, "{tag}: iters ≥ 1 but U == I");
        assert!(info.kappa_norm > 0.0, "{tag}: iters ≥ 1 but kappa_norm = 0");
    }

    let gp0 = pair_norm(&frozen_orbital_gradient(ints, rdm));
    let gp1 = pair_norm(&frozen_orbital_gradient(&rotate_integrals(ints, u), rdm));
    for (name, got, gp) in [
        ("g_before", info.g_before, gp0),
        ("g_after", info.g_after, gp1),
    ] {
        assert!(
            (got - gp).abs() <= 1e-14 * gp.max(f64::MIN_POSITIVE),
            "{tag}: {name} = {got:e} is not the pair norm √(Σ_{{p>q}} g²) = {gp:e} \
             (a full-Frobenius reading sits at √2×)"
        );
    }
    if matches!(info.exit, FrozenOrbExit::GradTol) {
        assert!(
            info.g_after <= opts.grad_tol,
            "{tag}: exit GradTol with g_after = {:e} > grad_tol = {:e}",
            info.g_after,
            opts.grad_tol
        );
    }
    if info.g_after <= opts.grad_tol {
        assert!(
            matches!(
                info.exit,
                FrozenOrbExit::GradTol | FrozenOrbExit::AlreadyCritical
            ),
            "{tag}: g_after = {:e} ≤ grad_tol = {:e} but exit {}: the grad_tol stop did not fire",
            info.g_after,
            opts.grad_tol,
            exit_label(info.exit)
        );
    }

    assert!(
        info.iters <= opts.max_iter,
        "{tag}: iters {} > max_iter {}",
        info.iters,
        opts.max_iter
    );
    dist
}

fn check_step_result(
    tag: &str,
    fx: &Fx,
    ints: &Integrals<f64>,
    rdm: &Rdm12,
    opts: &FrozenOrbOpts,
    u: &[f64],
    info: &FrozenOrbInfo,
) {
    print_info(tag, info);
    let dist = check_step_common(tag, fx, ints, rdm, opts, u, info);

    let bound = std::f64::consts::SQRT_2 * opts.max_step * info.iters as f64;
    assert!(
        dist <= bound * (1.0 + 1e-10) + 1e-14,
        "{tag}: ‖U − I‖_F = {dist:e} exceeds √2·max_step·iters = {bound:e}"
    );
    let pair_bound = opts.max_step * info.iters as f64;
    assert!(
        info.kappa_norm <= pair_bound * (1.0 + 1e-10) + 1e-14,
        "{tag}: kappa_norm = {:e} exceeds max_step·iters = {pair_bound:e} (pair norm)",
        info.kappa_norm
    );

    let cap = 1 + (info.iters + 1) * (opts.max_backtracks + 1);
    assert!(
        info.n_evals > info.iters && info.n_evals <= cap,
        "{tag}: n_evals = {} outside [1 + iters, 1 + (iters+1)(max_backtracks+1)] = [{}, {cap}]",
        info.n_evals,
        1 + info.iters
    );
}

#[test]
fn tord_g3_frozen_step_contract() {
    let opts = FrozenOrbOpts::default();
    for fx in fixtures() {
        let rdm = Rdm12::of(&fx.x);
        assert_fixture_floors(&fx, &rdm);
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        let tag = format!("tord_g3 {} default", fx.label);
        check_step_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
        let drop = info.e_before - info.e_after;
        println!(
            "{tag}: decrease {drop:.3e} (floor 1e-6, margin {:.1e}×)",
            drop / 1e-6
        );
        assert!(
            drop > 1e-6,
            "{tag}: the step lowered E by only {drop:e} (≤ 1e-6) from a non-critical point"
        );
        assert!(
            !is_already_critical(&info),
            "{tag}: AlreadyCritical at a point with ‖g‖_F > 1e-3"
        );
        assert!(info.iters >= 1, "{tag}: no accepted step");
    }
}

#[test]
fn tord_g3_step_cap_respected() {
    let cap = 1e-3;
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        let gf = fro(&frozen_orbital_gradient(&fx.ints, &rdm));
        assert!(
            gf >= 100.0 * cap,
            "{}: ‖g‖_F = {gf:e} is not large against the cap: arm is vacuous",
            fx.label
        );
        for k in [1usize, 5] {
            let opts = FrozenOrbOpts {
                max_step: cap,
                max_iter: k,
                ..FrozenOrbOpts::default()
            };
            let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
            let tag = format!("tord_g3 {} cap {cap:e} max_iter {k}", fx.label);
            check_step_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
            assert!(info.iters >= 1, "{tag}: a tiny capped step was refused");
            println!(
                "{tag}: ‖U − I‖_F = {:.3e} vs bound {:.3e}",
                dist_from_identity(&u, d),
                std::f64::consts::SQRT_2 * cap * info.iters as f64
            );
        }
        let wide = FrozenOrbOpts {
            max_step: 1e3,
            max_iter: 1,
            ..FrozenOrbOpts::default()
        };
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &wide);
        let tag = format!("tord_g3 {} uncapped max_iter 1", fx.label);
        check_step_result(&tag, &fx, &fx.ints, &rdm, &wide, &u, &info);
        let dist = dist_from_identity(&u, d);
        let floor = 10.0 * std::f64::consts::SQRT_2 * cap;
        println!("{tag}: ‖U − I‖_F = {dist:.3e} (anti-vacuity floor {floor:.3e})");
        assert!(
            dist > floor,
            "{tag}: the uncapped step moves only {dist:e}: the cap arm cannot discriminate"
        );
    }
}

#[test]
fn tord_g3_right_trivialized_steps() {
    let cap = 5e-3;
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        let opts = FrozenOrbOpts {
            max_iter: 2,
            memory: 0,
            max_step: cap,
            ..FrozenOrbOpts::default()
        };
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        let tag = format!("tord_g3 {} two SD steps", fx.label);
        check_step_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
        assert_eq!(
            info.iters, 2,
            "{tag}: two tiny capped steps must both be accepted"
        );

        let max_dev = |a: &[f64], b: &[f64]| {
            a.iter()
                .zip(b)
                .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        };
        let mut devs: Vec<(f64, f64, &str)> = Vec::new();
        for (reading, scale) in [
            ("pair", std::f64::consts::FRAC_1_SQRT_2),
            ("frobenius", 1.0),
        ] {
            let g1 = frozen_orbital_gradient(&fx.ints, &rdm);
            let a1 = cap / (scale * fro(&g1));
            assert!(a1 < 1.0, "{tag}: the cap does not bind (α₁ = {a1})");
            let w1 = expm_antisym(&scale_flat(&g1.data, a1), d);
            let g2 = frozen_orbital_gradient(&rotate_integrals(&fx.ints, &w1), &rdm);
            let a2 = cap / (scale * fro(&g2));
            let w2 = expm_antisym(&scale_flat(&g2.data, a2), d);
            let right = compose_rotations(&w1, &w2, d);
            let left = compose_rotations(&w2, &w1, d);
            let dev = max_dev(&u, &right);
            let gap = max_dev(&right, &left);
            println!(
                "{tag}: {reading} reading: max|U − W₁W₂| = {dev:.1e}, max|W₁W₂ − W₂W₁| = {gap:.1e}"
            );
            devs.push((dev, gap, reading));
        }
        let (dev, gap, _) = devs[0];
        let (dev_f, _, _) = devs[1];
        assert!(
            dev <= 1e-12,
            "{tag}: U does not match the pair-norm right-trivialized steepest-descent pair \
             ({dev:e}; frobenius reading {dev_f:e})"
        );
        assert!(
            dev_f > 1e-10,
            "{tag}: the frobenius reading also matches ({dev_f:e}): the norm is not discriminated"
        );
        assert!(
            gap > 1e-10,
            "{tag}: W₁W₂ and W₂W₁ agree to {gap:e}: the frame is not discriminated"
        );
    }
}

fn lower_half(g: &CoeffTensor<f64>) -> Vec<f64> {
    let d = g.d;
    let mut h = Vec::with_capacity(d * (d - 1) / 2);
    for p in 0..d {
        for q in 0..p {
            h.push(g.data[p * d + q]);
        }
    }
    h
}

fn from_lower_half(h: &[f64], d: usize) -> Vec<f64> {
    let mut k = vec![0.0f64; d * d];
    let mut i = 0;
    for p in 0..d {
        for q in 0..p {
            k[p * d + q] = h[i];
            k[q * d + p] = -h[i];
            i += 1;
        }
    }
    k
}

fn vdot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn dense_lbfgs_h(pairs: &[(Vec<f64>, Vec<f64>)], n: usize) -> Vec<f64> {
    let gamma = pairs.last().map_or(1.0, |(s, y)| vdot(s, y) / vdot(y, y));
    let mut h = vec![0.0f64; n * n];
    for i in 0..n {
        h[i * n + i] = gamma;
    }
    for (s, y) in pairs {
        let rho = 1.0 / vdot(s, y);
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                v[i * n + j] = f64::from(u8::from(i == j)) - rho * y[i] * s[j];
            }
        }
        let mut hv = vec![0.0f64; n * n];
        for i in 0..n {
            for k in 0..n {
                let a = h[i * n + k];
                for j in 0..n {
                    hv[i * n + j] += a * v[k * n + j];
                }
            }
        }
        let mut next = vec![0.0f64; n * n];
        for i in 0..n {
            for k in 0..n {
                let a = v[k * n + i];
                for j in 0..n {
                    next[i * n + j] += a * hv[k * n + j];
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                next[i * n + j] += rho * s[i] * s[j];
            }
        }
        h = next;
    }
    h
}

#[test]
fn tord_g3_lbfgs_direction() {
    let cap = 5e-3;
    let mut skip_consumed = false;
    for fx in fixtures() {
        let d = fx.space.d;
        let n = d * (d - 1) / 2;
        let rdm = Rdm12::of(&fx.x);
        for (memory, steps) in [(1usize, 2usize), (1, 3), (2, 3), (2, 4)] {
            let opts = FrozenOrbOpts {
                max_iter: steps,
                memory,
                max_step: cap,
                ..FrozenOrbOpts::default()
            };
            let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
            let tag = format!("tord_g3 {} L-BFGS memory {memory} steps {steps}", fx.label);
            check_step_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
            assert_eq!(
                info.iters, steps,
                "{tag}: every capped step must be accepted"
            );
            assert_eq!(
                info.n_evals,
                1 + steps,
                "{tag}: a capped step backtracked, so α is not the first trial"
            );

            let mut uk = identity_matrix(d);
            let mut gk = lower_half(&frozen_orbital_gradient(&fx.ints, &rdm));
            let mut mem: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
            let mut pairs_seen = 0usize;
            let mut two_pairs_at_last = false;
            for k in 0..steps {
                if k + 1 == steps {
                    two_pairs_at_last = pairs_seen >= 2;
                }
                let h = dense_lbfgs_h(&mem, n);
                let r: Vec<f64> = (0..n).map(|i| vdot(&h[i * n..(i + 1) * n], &gk)).collect();
                let slope = -vdot(&gk, &r);
                assert!(
                    slope < 0.0,
                    "{tag}: step {k}: the model direction is not descent"
                );
                let kn = vdot(&r, &r).sqrt();
                let alpha = (cap / kn).min(1.0);
                let step: Vec<f64> = r.iter().map(|ri| alpha * ri).collect();
                let w = expm_antisym(&from_lower_half(&step, d), d);
                uk = compose_rotations(&uk, &w, d);
                let gn = lower_half(&frozen_orbital_gradient(
                    &rotate_integrals(&fx.ints, &uk),
                    &rdm,
                ));
                let sv: Vec<f64> = step.iter().map(|x| -x).collect();
                let yv: Vec<f64> = gn.iter().zip(&gk).map(|(a, b)| a - b).collect();
                let cos = vdot(&sv, &yv) / (vdot(&sv, &sv).sqrt() * vdot(&yv, &yv).sqrt());
                println!("{tag}: step {k}: α = {alpha:.3e}, cos(s, y) = {cos:+.3e}");
                assert!(
                    cos.abs() > 1e-6,
                    "{tag}: step {k}: cos(s, y) = {cos:e} is at the skip threshold"
                );
                if cos > 1e-12 {
                    mem.push((sv, yv));
                    pairs_seen += 1;
                    if mem.len() > memory {
                        mem.remove(0);
                    }
                } else if steps == 4 && k + 1 < steps {
                    skip_consumed = true;
                }
                gk = gn;
            }
            let dev = u
                .iter()
                .zip(&uk)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            let sd = FrozenOrbOpts { memory: 0, ..opts };
            let (u_sd, _) = frozen_orbital_step(&fx.ints, &rdm, &sd);
            let gap = u_sd
                .iter()
                .zip(&uk)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            println!(
                "{tag}: max|U − U_rebuilt| = {dev:.1e}, max|U_sd − U_rebuilt| = {gap:.1e}, \
                 pairs stored {pairs_seen}, two pairs before the last step: {two_pairs_at_last}"
            );
            assert!(
                dev <= 1e-12,
                "{tag}: U does not match the dense-BFGS rebuild ({dev:e})"
            );
            assert!(
                gap > 1e-8,
                "{tag}: the memory run matches steepest descent ({gap:e}): the arm is vacuous"
            );
            if steps == 3 {
                assert!(
                    two_pairs_at_last,
                    "{tag}: fewer than two pairs before the last step: eviction/loop order untested"
                );
            }
        }
    }
    assert!(
        skip_consumed,
        "tord_g3 L-BFGS: no fixture skips a negative-curvature pair before the last step of the \
         4-step cell: the skip rule is untested"
    );
}

#[test]
fn tord_g3_armijo_slope_pairing() {
    let cap = 5e-3;
    for fx in fixtures() {
        let rdm = Rdm12::of(&fx.x);
        let opts = FrozenOrbOpts {
            max_iter: 1,
            memory: 0,
            max_step: cap,
            armijo_c1: 0.6,
            ..FrozenOrbOpts::default()
        };
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        let tag = format!("tord_g3 {} armijo_c1 0.6", fx.label);
        check_step_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
        let gp = pair_norm(&frozen_orbital_gradient(&fx.ints, &rdm));
        let predicted = cap * gp;
        let ratio = (info.e_before - info.e_after) / predicted;
        println!("{tag}: actual/predicted decrease = {ratio:.4}");
        assert!(
            ratio > 0.7 && ratio < 1.1,
            "{tag}: decrease ratio {ratio} outside (0.7, 1.1): the arm does not discriminate"
        );
        assert_eq!(
            info.iters, 1,
            "{tag}: the first capped trial was not accepted"
        );
        assert_eq!(info.n_evals, 2, "{tag}: the step backtracked");
    }
}

#[test]
fn tord_g3_monotone_prefixes() {
    for fx in fixtures() {
        let rdm = Rdm12::of(&fx.x);
        let mut prev = f64::INFINITY;
        for k in 1..=5usize {
            let opts = FrozenOrbOpts {
                max_iter: k,
                ..FrozenOrbOpts::default()
            };
            let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
            let tag = format!("tord_g3 {} max_iter {k}", fx.label);
            check_step_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
            assert!(
                info.e_after <= prev,
                "{tag}: e_after = {} rose above the max_iter = {} value {prev}",
                info.e_after,
                k - 1
            );
            prev = info.e_after;
        }
    }
}

#[test]
fn tord_g3_caller_composes() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        let u0 = expm_antisym(&scale_flat(&fx.k, 0.5), d);
        let ints0 = rotate_integrals(&fx.ints, &u0);
        let opts = FrozenOrbOpts::default();
        let (u1, info) = frozen_orbital_step(&ints0, &rdm, &opts);
        let tag = format!("tord_g3 {} from U0", fx.label);
        check_step_result(&tag, &fx, &ints0, &rdm, &opts, &u1, &info);
        let u01 = compose_rotations(&u0, &u1, d);
        let e = e_rq(fx.space, &fx.ints, &u01, &fx.x);
        let dev = (e - info.e_after).abs();
        println!("{tag}: |⟨x|H(rotate(ints, U0·U1))|x⟩ − e_after| = {dev:.1e}");
        assert!(
            dev <= 1e-12 * e.abs().max(1.0),
            "{tag}: the composed rotation does not reproduce e_after: {dev:e}"
        );
        assert!(
            orthogonality_defect(&u01, d) <= 1e-13,
            "{tag}: U0·U1 not orthogonal"
        );
        assert!(
            info.e_before - info.e_after > 1e-6,
            "{tag}: no decrease from U0 ({:e})",
            info.e_before - info.e_after
        );
    }
}

#[test]
fn tord_g3_orthogonality_does_not_compound() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        let opts = FrozenOrbOpts {
            max_step: 1e-2,
            grad_tol: 0.0,
            max_iter: 200,
            ..FrozenOrbOpts::default()
        };
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        let tag = format!("tord_g3 {} long run", fx.label);
        let orth = orthogonality_defect(&u, d);
        let floor = 4.0 * f64::EPSILON * d as f64;
        let mut u_total = identity_matrix(d);
        for _ in 0..40 {
            u_total = compose_rotations(&u_total, &u, d);
        }
        let orth40 = orthogonality_defect(&u_total, d);
        println!(
            "{tag}: {} accepted steps, exit {}; max|UᵀU − I| = {orth:.1e} (floor {floor:.1e}); \
             after 40 compositions {orth40:.1e}",
            info.iters,
            exit_label(info.exit)
        );
        assert!(
            info.iters >= 50,
            "{tag}: only {} accepted steps — the run is too short to discriminate compounding",
            info.iters
        );
        assert!(
            orth <= floor,
            "{tag}: returned U has max|UᵀU − I| = {orth:e} > {floor:e}: roundoff compounds"
        );
        assert!(
            orth40 <= 5e-14,
            "{tag}: 40 compositions reach max|UᵀU − I| = {orth40:e}"
        );
    }
}

fn lifted_eigenstate(space: QnSpace, ints: &Integrals<f64>, idx: usize) -> (TtVector<f64>, f64) {
    let d = space.d;
    let dets = dets_for(d, space.nup, space.ndn);
    let hd = dense_hamiltonian_matrix(d, &ints.t, &ints.v, &dets);
    let (evals, evecs) = dense_eigen(&hd);
    let lambda = evals[idx];
    let mut x: Option<TtVector<f64>> = None;
    for (i, (ou, od)) in dets.iter().enumerate() {
        let term = &tt_state::<f64>(ou, od) * evecs[idx][i];
        x = Some(match x {
            None => term,
            Some(acc) => &acc + &term,
        });
    }
    let mut x = x.expect("non-empty determinant basis");
    x.round_mut(1e-14, qntt::Pooled::NONE, qntt::round::RoundMode::Rel);
    normalize_mut(&mut x);
    let h = SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    let hx = h.apply(&x);
    let mut resid = &hx - &(&x * lambda);
    resid.round_mut(1e-14, qntt::Pooled::NONE, qntt::round::RoundMode::Abs);
    let r = resid.norm();
    println!(
        "tord_g4 d={d} ({}, {}) idx {idx}: λ = {lambda:.15}, ‖Hx − λx‖ = {r:.1e}",
        space.nup, space.ndn
    );
    assert!(
        r <= 1e-10 * lambda.abs().max(1.0),
        "lifted state is not an eigenstate at idx {idx}: ‖Hx − λx‖ = {r:e}"
    );
    (x, lambda)
}

#[test]
fn tord_g4_eigenstate_already_critical() {
    for (d, nup, ndn, seed, idxs) in [
        (4usize, 2usize, 1usize, 0x6_4421u64, vec![0usize, 2]),
        (4, 2, 0, 0x6_4420, vec![0]),
    ] {
        let space = QnSpace::new(d, nup, ndn);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let t = random_sym_t(d, &mut rng);
        let v = random_sym_v(d, &mut rng);
        let ints = ints_of(space, t, v);
        for idx in idxs {
            let tag = format!("tord_g4 d={d} ({nup},{ndn}) idx {idx}");
            let (x, lambda) = lifted_eigenstate(space, &ints, idx);
            let rdm = Rdm12::of(&x);
            let scale = lambda.abs().max(1.0);
            let gmax = max_abs_t(&frozen_orbital_gradient(&ints, &rdm));
            assert!(
                gmax <= 1e-10 * scale,
                "{tag}: frozen gradient does not vanish at an eigenstate: max|g| = {gmax:e}"
            );
            let (u, info) = frozen_orbital_step(&ints, &rdm, &FrozenOrbOpts::default());
            print_info(&tag, &info);
            assert!(
                is_already_critical(&info),
                "{tag}: exit {} at an exact eigenstate (g_before = {:e})",
                exit_label(info.exit),
                info.g_before
            );
            let id = identity_matrix(d);
            assert_eq!(u.len(), d * d, "{tag}: U is d×d");
            assert!(
                u.iter().zip(&id).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{tag}: AlreadyCritical must return U = I bitwise, got {u:?}"
            );
            assert_eq!(info.iters, 0, "{tag}: iters");
            assert!(
                info.e_after.to_bits() == info.e_before.to_bits(),
                "{tag}: e_after ≠ e_before bitwise"
            );
            assert!(
                info.g_after.to_bits() == info.g_before.to_bits(),
                "{tag}: g_after ≠ g_before bitwise"
            );
            assert!(
                info.kappa_norm == 0.0,
                "{tag}: kappa_norm = {:e}",
                info.kappa_norm
            );
            assert!(info.n_evals <= 1, "{tag}: n_evals = {}", info.n_evals);
            assert!(
                info.g_before <= 1e-7,
                "{tag}: g_before = {:e}",
                info.g_before
            );
            assert!(
                (info.e_before - lambda).abs() <= 1e-12 * scale,
                "{tag}: e_before = {} vs λ = {lambda}",
                info.e_before
            );

            let y = capped_randn(space, 1, 2, seed ^ (idx as u64 + 1));
            let mut xp = &x + &(&y * 1e-3);
            normalize_mut(&mut xp);
            let rdm_p = Rdm12::of(&xp);
            let gp = fro(&frozen_orbital_gradient(&ints, &rdm_p));
            let (_, info_p) = frozen_orbital_step(&ints, &rdm_p, &FrozenOrbOpts::default());
            print_info(&format!("{tag} perturbed"), &info_p);
            assert!(gp > 1e-5, "{tag}: the perturbation left ‖g‖_F = {gp:e}");
            assert!(
                !is_already_critical(&info_p),
                "{tag}: a perturbed eigenstate (‖g‖_F = {gp:e}) was declared AlreadyCritical"
            );
        }
    }
}

#[test]
fn tord_g4_gate_is_grad_tol() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        let gf = fro(&frozen_orbital_gradient(&fx.ints, &rdm));
        assert!(
            gf > 1e-3 && gf < 1e2,
            "{}: ‖g‖_F = {gf:e} outside (1e-3, 1e2)",
            fx.label
        );
        let opts = FrozenOrbOpts {
            grad_tol: 1e3,
            ..FrozenOrbOpts::default()
        };
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        let tag = format!("tord_g4 {} grad_tol 1e3", fx.label);
        print_info(&tag, &info);
        assert!(is_already_critical(&info), "{tag}: not AlreadyCritical");
        let id = identity_matrix(d);
        assert!(
            u.iter().zip(&id).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{tag}: U ≠ I bitwise"
        );
        assert_eq!(info.iters, 0, "{tag}: iters");
        assert!(
            info.e_after.to_bits() == info.e_before.to_bits(),
            "{tag}: e_after ≠ e_before bitwise"
        );
        assert!(
            info.e_before.to_bits() == energy_from_rdms(&fx.ints, &rdm).to_bits(),
            "{tag}: e_before ≠ energy_from_rdms(ints) bitwise"
        );
    }
}

fn unit_state_421(seed: u64) -> TtVector<f64> {
    capped_randn(QnSpace::new(4, 2, 1), 1, 2, seed)
}

#[test]
fn tord_g5_rdm_refuses_norm_just_outside_band() {
    let mut x = unit_state_421(0x6_5001);
    x.scale_mut(1.0 + 1e-9);
    assert_refused(
        || {
            let _ = Rdm12::of(&x);
        },
        &["Rdm12::of", "norm"],
        "‖x‖ = 1 + 1e-9",
    );
}

#[test]
fn tord_g5_rdm_refuses_norm_far_outside_band() {
    let mut x = unit_state_421(0x6_5002);
    x.scale_mut(1.5);
    assert_refused(
        || {
            let _ = Rdm12::of(&x);
        },
        &["Rdm12::of", "norm"],
        "‖x‖ = 1.5",
    );
}

#[test]
fn tord_g5_rdm_accepts_norm_inside_band() {
    let mut x = unit_state_421(0x6_5003);
    x.scale_mut(1.0 + 1e-11);
    let rdm = Rdm12::of(&x);
    assert_eq!(rdm.n, 3);
    assert_eq!(rdm.d, 4);
}

#[test]
fn tord_g5_rdm_refuses_point_group() {
    let pg = PointGroup::new(1, &[0u8, 1, 1, 0].map(Irrep::new));
    let space = QnSpace::with_point_group(4, 2, 1, pg, Irrep::new(1));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6_5004);
    let mut x: TtVector<f64> = tt_randn(space, &max_ranks(&space), &mut rng);
    normalize_mut(&mut x);
    assert!(
        (x.norm() - 1.0).abs() <= 1e-13,
        "the |G| = 2 state is not unit-norm"
    );
    assert_refused(
        || {
            let _ = Rdm12::of(&x);
        },
        &["Rdm12::of", "point"],
        "|G| = 2",
    );
}

#[test]
fn tord_g5_dimension_mismatch_refused() {
    let x = unit_state_421(0x6_5005);
    let rdm = Rdm12::of(&x);
    let space5 = QnSpace::new(5, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6_5006);
    let ints5 = ints_of(space5, random_sym_t(5, &mut rng), random_sym_v(5, &mut rng));
    assert_refused(
        || {
            let _ = energy_from_rdms(&ints5, &rdm);
        },
        &["norb"],
        "energy_from_rdms d mismatch",
    );
    assert_refused(
        || {
            let _ = generalized_fock(&ints5, &rdm);
        },
        &["norb"],
        "generalized_fock d mismatch",
    );
    assert_refused(
        || {
            let _ = frozen_orbital_gradient(&ints5, &rdm);
        },
        &["norb"],
        "frozen_orbital_gradient d mismatch",
    );
    assert_refused(
        || {
            let _ = frozen_orbital_step(&ints5, &rdm, &FrozenOrbOpts::default());
        },
        &["norb"],
        "frozen_orbital_step d mismatch",
    );
}

#[test]
fn tord_g5_invalid_options_refused() {
    let space = QnSpace::new(4, 2, 1);
    let x = unit_state_421(0x6_5007);
    let rdm = Rdm12::of(&x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6_5008);
    let ints = ints_of(space, random_sym_t(4, &mut rng), random_sym_v(4, &mut rng));
    let cases: Vec<(FrozenOrbOpts, &str)> = vec![
        (
            FrozenOrbOpts {
                max_step: 0.0,
                ..FrozenOrbOpts::default()
            },
            "max_step",
        ),
        (
            FrozenOrbOpts {
                max_step: -0.1,
                ..FrozenOrbOpts::default()
            },
            "max_step",
        ),
        (
            FrozenOrbOpts {
                max_step: f64::NAN,
                ..FrozenOrbOpts::default()
            },
            "max_step",
        ),
        (
            FrozenOrbOpts {
                max_step: f64::INFINITY,
                ..FrozenOrbOpts::default()
            },
            "max_step",
        ),
        (
            FrozenOrbOpts {
                armijo_c1: 0.0,
                ..FrozenOrbOpts::default()
            },
            "armijo_c1",
        ),
        (
            FrozenOrbOpts {
                armijo_c1: 1.0,
                ..FrozenOrbOpts::default()
            },
            "armijo_c1",
        ),
        (
            FrozenOrbOpts {
                armijo_c1: f64::NAN,
                ..FrozenOrbOpts::default()
            },
            "armijo_c1",
        ),
        (
            FrozenOrbOpts {
                grad_tol: -1e-7,
                ..FrozenOrbOpts::default()
            },
            "grad_tol",
        ),
        (
            FrozenOrbOpts {
                grad_tol: f64::NAN,
                ..FrozenOrbOpts::default()
            },
            "grad_tol",
        ),
    ];
    for (opts, field) in cases {
        assert_refused(
            || {
                let _ = frozen_orbital_step(&ints, &rdm, &opts);
            },
            &["frozen_orbital_step", field],
            &format!("invalid {field}"),
        );
    }
}

#[test]
fn tord_g5_defaults() {
    let o = FrozenOrbOpts::default();
    assert_eq!(o.max_iter, 30);
    assert_eq!(o.grad_tol, 1e-7);
    assert_eq!(o.max_step, 0.2);
    assert_eq!(o.memory, 8);
    assert_eq!(o.armijo_c1, 1e-4);
    assert_eq!(o.max_backtracks, 20);
}

fn pair_list(d: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(d * d.saturating_sub(1) / 2);
    for p in 0..d {
        for q in 0..p {
            v.push((p, q));
        }
    }
    v
}

fn lower_half_flat(m: &[f64], d: usize) -> Vec<f64> {
    pair_list(d).iter().map(|&(p, q)| m[p * d + q]).collect()
}

fn neg_vec(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| -x).collect()
}

fn max_abs_vec(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn max_dev_vec(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "max_dev_vec: length mismatch");
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn hess_vec(h: &[f64], n: usize, x: &[f64]) -> Vec<f64> {
    (0..n).map(|a| vdot(&h[a * n..(a + 1) * n], x)).collect()
}

fn chart_energy(ints: &Integrals<f64>, rdm: &Rdm12, u: &[f64], k: &[f64], d: usize) -> f64 {
    let w = expm_antisym(&from_lower_half(&neg_vec(k), d), d);
    energy_from_rdms(&rotate_integrals(ints, &compose_rotations(u, &w, d)), rdm)
}

fn left_chart_energy(ints: &Integrals<f64>, rdm: &Rdm12, u: &[f64], k: &[f64], d: usize) -> f64 {
    let w = expm_antisym(&from_lower_half(&neg_vec(k), d), d);
    energy_from_rdms(&rotate_integrals(ints, &compose_rotations(&w, u, d)), rdm)
}

const Q_LADDER: [f64; 3] = [8e-3, 4e-3, 2e-3];

fn q_ladder(f: impl Fn(f64) -> f64, f0: f64) -> (Vec<f64>, Vec<f64>) {
    let qq: Vec<f64> = Q_LADDER
        .iter()
        .map(|&h| (f(h) - 2.0 * f0 + f(-h)) / (h * h))
        .collect();
    let rr: Vec<f64> = (1..qq.len())
        .map(|i| (4.0 * qq[i] - qq[i - 1]) / 3.0)
        .collect();
    (qq, rr)
}

fn assert_hessian_structure(tag: &str, h: &[f64], n: usize) {
    assert_eq!(
        h.len(),
        n * n,
        "{tag}: H must hold n_pairs² = {} entries",
        n * n
    );
    for a in 0..n {
        for b in 0..n {
            assert!(h[a * n + b].is_finite(), "{tag}: H[{a},{b}] is not finite");
            assert!(
                h[a * n + b].to_bits() == h[b * n + a].to_bits(),
                "{tag}: H not exactly symmetric at ({a},{b}): {:e} vs {:e}",
                h[a * n + b],
                h[b * n + a]
            );
        }
    }
}

fn check_hessian_chart_fd(
    fx: &Fx,
    rdm: &Rdm12,
    u: &[f64],
    ulabel: &str,
    left_negative: bool,
) -> Vec<f64> {
    let d = fx.space.d;
    let n = d * (d - 1) / 2;
    let tag = format!("tord_n1 {} U={ulabel}", fx.label);
    let ints_u = rotate_integrals(&fx.ints, u);
    let h = frozen_orbital_hessian(&ints_u, rdm);
    assert_hessian_structure(&tag, &h, n);
    let hmax = max_abs_vec(&h);
    assert!(hmax > 1e-2, "{tag}: max|H| = {hmax:e}: arm is vacuous");
    let scale = hmax.max(1.0);

    let zero = vec![0.0f64; n];
    let f0 = chart_energy(&fx.ints, rdm, u, &zero, d);
    let q_of = |dir: &[f64]| {
        q_ladder(
            |hh| chart_energy(&fx.ints, rdm, u, &scale_flat(dir, hh), d),
            f0,
        )
    };
    let unit = |a: usize| {
        let mut e = vec![0.0f64; n];
        e[a] = 1.0;
        e
    };
    let mut fd_last = vec![0.0f64; n * n];
    let mut fd_mid = vec![0.0f64; n * n];
    for a in 0..n {
        let (_, ra) = q_of(&unit(a));
        fd_last[a * n + a] = ra[ra.len() - 1];
        fd_mid[a * n + a] = ra[0];
        for b in a + 1..n {
            let plus: Vec<f64> = (0..n).map(|i| unit(a)[i] + unit(b)[i]).collect();
            let minus: Vec<f64> = (0..n).map(|i| unit(a)[i] - unit(b)[i]).collect();
            let (_, rp) = q_of(&plus);
            let (_, rm) = q_of(&minus);
            let last = (rp[rp.len() - 1] - rm[rm.len() - 1]) / 4.0;
            let mid = (rp[0] - rm[0]) / 4.0;
            fd_last[a * n + b] = last;
            fd_last[b * n + a] = last;
            fd_mid[a * n + b] = mid;
            fd_mid[b * n + a] = mid;
        }
    }
    let worst = max_dev_vec(&fd_last, &h);
    let worst_mid = max_dev_vec(&fd_mid, &h);
    let tol = 1e-7 * scale;
    println!(
        "{tag}: max|H| = {hmax:.3e}; polarized FD: max|R(4e-3) − H| = {worst_mid:.2e}, \
         max|R(2e-3) − H| = {worst:.2e} (tol {tol:.1e}, margin {:.1}×)",
        tol / worst.max(1e-300)
    );
    assert!(
        worst_mid <= 1e-5 * scale,
        "{tag}: the h = 4e-3 Richardson rung misses H by {worst_mid:e} (plateau: a wrong Hessian, \
         not a stencil error)"
    );
    assert!(
        worst <= tol,
        "{tag}: frozen_orbital_hessian vs the polarized chart FD: max dev {worst:e} (tol {tol:e})"
    );

    let k = lower_half_flat(&fx.kappa, d);
    let want = vdot(&k, &hess_vec(&h, n, &k));
    let (_, rk) = q_of(&k);
    let got = rk[rk.len() - 1];
    let ktol = 1e-7 * want.abs().max(1.0);
    println!(
        "{tag}: random direction: κᵀHκ = {want:.12e}, R(2e-3) = {got:.12e}, |diff| = {:.2e} \
         (tol {ktol:.1e})",
        (got - want).abs()
    );
    assert!(
        want.abs() > 1e-3,
        "{tag}: κᵀHκ = {want:e}: random-direction arm is vacuous"
    );
    assert!(
        (got - want).abs() <= ktol,
        "{tag}: κᵀHκ = {want} vs the chart FD {got}: |diff| = {:e}",
        (got - want).abs()
    );

    if left_negative {
        let (_, rl) = q_ladder(
            |hh| left_chart_energy(&fx.ints, rdm, u, &scale_flat(&k, hh), d),
            f0,
        );
        let miss = (rl[rl.len() - 1] - want).abs();
        println!("{tag}: the left-chart second difference misses κᵀHκ by {miss:.3e}");
        assert!(
            miss > 1e-3 * want.abs().max(1.0),
            "{tag}: the left chart exp(−κ)·U matches the body-frame κᵀHκ ({miss:e}): the U_r rung \
             does not discriminate the frame"
        );
    }
    h
}

#[test]
fn tord_n1_hessian_matches_chart_fd() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        assert_fixture_floors(&fx, &rdm);
        let h_i = check_hessian_chart_fd(&fx, &rdm, &identity_matrix(d), "I", false);
        let u_r = expm_antisym(&scale_flat(&fx.k, 0.5), d);
        let h_r = check_hessian_chart_fd(&fx, &rdm, &u_r, "U_r", true);
        let moved = max_dev_vec(&h_i, &h_r);
        assert!(
            moved > 1e-3,
            "{}: H(U_r) equals H(I) to {moved:e}: the U_r rung is vacuous",
            fx.label
        );
    }
}

#[test]
fn tord_n1_asymmetric_integrals_fd() {
    let fx = asym_fixture();
    let d = fx.space.d;
    let rdm = Rdm12::of(&fx.x);
    let h_i = check_hessian_chart_fd(&fx, &rdm, &identity_matrix(d), "I", false);
    let u_r = expm_antisym(&scale_flat(&fx.k, 0.5), d);
    let h_r = check_hessian_chart_fd(&fx, &rdm, &u_r, "U_r", true);
    assert!(
        max_dev_vec(&h_i, &h_r) > 1e-3,
        "asymmetric fixture: H(U_r) equals H(I)"
    );
}

#[test]
fn tord_n1_small_d_edges() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6_7001);
    let space1 = QnSpace::new(1, 1, 0);
    let ints1 = ints_of(space1, random_sym_t(1, &mut rng), random_sym_v(1, &mut rng));
    let x1: TtVector<f64> = tt_state(&[true], &[false]);
    let rdm1 = Rdm12::of(&x1);
    let h1 = frozen_orbital_hessian(&ints1, &rdm1);
    assert!(h1.is_empty(), "d = 1: H must be empty, got {h1:?}");

    let space2 = QnSpace::new(2, 1, 1);
    let ints2 = ints_of(space2, random_sym_t(2, &mut rng), random_sym_v(2, &mut rng));
    let x2 = capped_randn(space2, 1, 2, 0x6_7002);
    let rdm2 = Rdm12::of(&x2);
    let h2 = frozen_orbital_hessian(&ints2, &rdm2);
    assert_hessian_structure("tord_n1 d = 2", &h2, 1);
    let id = identity_matrix(2);
    let f0 = chart_energy(&ints2, &rdm2, &id, &[0.0], 2);
    let (_, r) = q_ladder(|hh| chart_energy(&ints2, &rdm2, &id, &[hh], 2), f0);
    let got = r[r.len() - 1];
    println!(
        "tord_n1 d = 2 (1,1): H = {:.12e}, chart FD = {got:.12e}",
        h2[0]
    );
    assert!(
        h2[0].abs() > 1e-3,
        "d = 2: H = {:e} is (near) zero: the arm is vacuous",
        h2[0]
    );
    assert!(
        (got - h2[0]).abs() <= 1e-7 * h2[0].abs().max(1.0),
        "d = 2: H = {} vs the chart FD {got}",
        h2[0]
    );
}

fn check_body_gradient_fd(fx: &Fx, rdm: &Rdm12, u: &[f64], ulabel: &str) {
    let d = fx.space.d;
    let n = d * (d - 1) / 2;
    let tag = format!("tord_n2 {} U={ulabel}", fx.label);
    let ints_u = rotate_integrals(&fx.ints, u);
    let h = frozen_orbital_hessian(&ints_u, rdm);
    let gm = frozen_orbital_gradient(&ints_u, rdm);
    let k = lower_half_flat(&fx.kappa, d);
    let kk = &fx.kappa;
    let mut comm = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0f64;
            for l in 0..d {
                s += kk[i * d + l] * gm.data[l * d + j] - gm.data[i * d + l] * kk[l * d + j];
            }
            comm[i * d + j] = s;
        }
    }
    let c = lower_half_flat(&comm, d);
    let hk = hess_vec(&h, n, &k);
    let rhs: Vec<f64> = hk.iter().zip(&c).map(|(a, b)| a + 0.5 * b).collect();

    let gfun = |hh: f64| {
        let w = expm_antisym(&scale_flat(kk, -hh), d);
        lower_half(&frozen_orbital_gradient(
            &rotate_integrals(&fx.ints, &compose_rotations(u, &w, d)),
            rdm,
        ))
    };
    let dd: Vec<Vec<f64>> = H_LADDER
        .iter()
        .map(|&hh| {
            let gp = gfun(hh);
            let gn = gfun(-hh);
            gp.iter()
                .zip(&gn)
                .map(|(a, b)| (a - b) / (2.0 * hh))
                .collect()
        })
        .collect();
    let rr: Vec<Vec<f64>> = (1..dd.len())
        .map(|i| {
            dd[i]
                .iter()
                .zip(&dd[i - 1])
                .map(|(a, b)| (4.0 * a - b) / 3.0)
                .collect()
        })
        .collect();
    let scale = max_abs_vec(&rhs).max(1.0);
    let tol = 1e-9 * scale;
    for i in 2..H_LADDER.len() {
        let dev = max_dev_vec(&rr[i - 1], &rhs);
        assert!(
            dev <= 1e-6 * scale,
            "{tag}: Richardson rung h = {:e} misses H·k + ½[K,G] by {dev:e} (plateau: a wrong \
             relation, not a stencil error)",
            H_LADDER[i]
        );
    }
    let fd = &rr[rr.len() - 1];
    let dev = max_dev_vec(fd, &rhs);
    let cmax = max_abs_vec(&c);
    println!(
        "{tag}: max|½[K,G]| = {:.3e}; max|FD − (H·k + ½[K,G])| = {dev:.2e} (tol {tol:.1e}, \
         margin {:.1}×)",
        0.5 * cmax,
        tol / dev.max(1e-300)
    );
    assert!(
        cmax > 1e-3,
        "{tag}: max|[K,G]| = {cmax:e}: the commutator correction is not live, the arm is vacuous"
    );
    assert!(
        dev <= tol,
        "{tag}: d/dh g(U·e^(−hK)) vs H·k + ½[K,G]: max dev {dev:e} (tol {tol:e})"
    );
    let wrong: [(&str, Vec<f64>); 3] = [
        ("H·k", hk.clone()),
        (
            "H·k − ½[K,G]",
            hk.iter().zip(&c).map(|(a, b)| a - 0.5 * b).collect(),
        ),
        (
            "H·k + [K,G]",
            hk.iter().zip(&c).map(|(a, b)| a + b).collect(),
        ),
    ];
    for (name, w) in &wrong {
        let miss = max_dev_vec(fd, w);
        println!(
            "{tag}: the {name} variant misses the FD by {miss:.3e} ({:.1e}× tol)",
            miss / tol
        );
        assert!(
            miss >= 10.0 * tol,
            "{tag}: the wrong variant {name} matches the FD to {miss:e} (< 10× tol {tol:e}): the \
             arm does not discriminate"
        );
    }
}

#[test]
fn tord_n2_body_gradient_fd() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        assert_fixture_floors(&fx, &rdm);
        check_body_gradient_fd(&fx, &rdm, &identity_matrix(d), "I");
        let u_r = expm_antisym(&scale_flat(&fx.k, 0.5), d);
        check_body_gradient_fd(&fx, &rdm, &u_r, "U_r");
    }
    let fx = asym_fixture();
    let d = fx.space.d;
    let rdm = Rdm12::of(&fx.x);
    let g = frozen_orbital_gradient(&fx.ints, &rdm);
    assert!(
        fro(&g) > 1e-3,
        "asymmetric fixture: ‖g‖_F = {:e}: vacuous",
        fro(&g)
    );
    check_body_gradient_fd(&fx, &rdm, &identity_matrix(d), "I");
    let u_r = expm_antisym(&scale_flat(&fx.k, 0.5), d);
    check_body_gradient_fd(&fx, &rdm, &u_r, "U_r");
}

fn dense_commutator_oracle(
    d: usize,
    ints_u: &Integrals<f64>,
    dets: &[(Vec<bool>, Vec<bool>)],
    c: &[f64],
) -> (Vec<f64>, Vec<f64>, f64) {
    let hd = dense_hamiltonian_matrix(d, &ints_u.t, &ints_u.v, dets);
    let hdt = mat_transpose(&hd);
    let hc = mat_vec(&hd, c);
    let htc = mat_vec(&hdt, c);
    let pairs = pair_list(d);
    let n = pairs.len();
    let mut gv = Vec::with_capacity(n);
    let mut gtv = Vec::with_capacity(n);
    let mut crc = Vec::with_capacity(n);
    let mut crtc = Vec::with_capacity(n);
    for &(p, q) in &pairs {
        let mut a = vec![0.0f64; d * d];
        a[p * d + q] = 1.0;
        a[q * d + p] = -1.0;
        let gp = dense_kappa_matrix(d, &flat_to_coeff(d, &a), dets);
        let gpt = mat_transpose(&gp);
        let g_c = mat_vec(&gp, c);
        let gt_c = mat_vec(&gpt, c);
        let h_gc = mat_vec(&hd, &g_c);
        let cc: Vec<f64> = mat_vec(&gp, &hc)
            .iter()
            .zip(&h_gc)
            .map(|(x, y)| x - y)
            .collect();
        let ht_gtc = mat_vec(&hdt, &gt_c);
        let gt_htc = mat_vec(&gpt, &htc);
        let cct: Vec<f64> = ht_gtc.iter().zip(&gt_htc).map(|(x, y)| x - y).collect();
        gv.push(g_c);
        gtv.push(gt_c);
        crc.push(cc);
        crtc.push(cct);
    }
    let g_or: Vec<f64> = (0..n)
        .map(|a| dot_vec(&gtv[a], &hc) - dot_vec(&htc, &gv[a]))
        .collect();
    let mut m = vec![0.0f64; n * n];
    for a in 0..n {
        for b in 0..n {
            m[a * n + b] = dot_vec(&gtv[a], &crc[b]) - dot_vec(&crtc[b], &gv[a]);
        }
    }
    let mut h_or = vec![0.0f64; n * n];
    let mut asym = 0.0f64;
    for a in 0..n {
        for b in 0..n {
            h_or[a * n + b] = 0.5 * (m[a * n + b] + m[b * n + a]);
            asym = asym.max((m[a * n + b] - m[b * n + a]).abs());
        }
    }
    (h_or, g_or, asym)
}

#[test]
fn tord_n3_dense_commutator_oracle() {
    let mut all = fixtures();
    all.push(asym_fixture());
    for fx in &all {
        let d = fx.space.d;
        let n = d * (d - 1) / 2;
        let rdm = Rdm12::of(&fx.x);
        let dets = dets_for(d, fx.space.nup, fx.space.ndn);
        let (c, n2) = sector_amplitudes(&fx.x, &dets);
        assert!(
            (n2 - 1.0).abs() <= 1e-13,
            "{}: the determinant list does not carry x (Σ c² = {n2})",
            fx.label
        );
        let u_r = expm_antisym(&scale_flat(&fx.k, 0.5), d);
        for (u, ulabel) in [(identity_matrix(d), "I"), (u_r, "U_r")] {
            let tag = format!("tord_n3 {} U={ulabel}", fx.label);
            let ints_u = rotate_integrals(&fx.ints, &u);
            let h = frozen_orbital_hessian(&ints_u, &rdm);
            assert_hessian_structure(&tag, &h, n);
            let (h_or, g_or, asym) = dense_commutator_oracle(d, &ints_u, &dets, &c);
            let g = lower_half(&frozen_orbital_gradient(&ints_u, &rdm));
            let hmax = max_abs_vec(&h_or);
            let hdev = max_dev_vec(&h, &h_or);
            let gdev = max_dev_vec(&g, &g_or);
            let htol = 1e-11 * hmax.max(1.0);
            let gtol = 1e-11 * max_abs_vec(&g_or).max(1.0);
            println!(
                "{tag}: max|H_or| = {hmax:.3e}, max|M − Mᵀ| = {asym:.3e}; max|H − H_or| = \
                 {hdev:.2e} (tol {htol:.1e}); max|g − g_or| = {gdev:.2e} (tol {gtol:.1e})"
            );
            assert!(hmax > 1e-2, "{tag}: max|H_or| = {hmax:e}: vacuous");
            assert!(
                max_abs_vec(&g_or) > 1e-3,
                "{tag}: the oracle gradient vanishes: the point is critical, the arm is vacuous"
            );
            assert!(
                asym > 1e-3,
                "{tag}: the double commutator is symmetric to {asym:e}: the symmetrization is \
                 not exercised"
            );
            assert!(
                gdev <= gtol,
                "{tag}: frozen_orbital_gradient vs ⟨[Ĝ_P, Ĥ]⟩: max dev {gdev:e} (the sign pin)"
            );
            assert!(
                hdev <= htol,
                "{tag}: frozen_orbital_hessian vs the dense commutator oracle: max dev {hdev:e} \
                 (tol {htol:e})"
            );
        }
    }
}

fn newton_tr(opts: &FrozenOrbOpts) -> FrozenTrOpts {
    match opts.method {
        FrozenOrbMethod::Newton(tr) => tr,
        _ => panic!("check_newton_result called on a non-Newton run"),
    }
}

fn newton_opts(max_iter: usize) -> FrozenOrbOpts {
    FrozenOrbOpts {
        max_iter,
        method: FrozenOrbMethod::Newton(FrozenTrOpts::default()),
        ..FrozenOrbOpts::default()
    }
}

fn print_newton_info(tag: &str, info: &FrozenOrbInfo) {
    print_info(tag, info);
    println!(
        "{tag}: n_hessians {}, min_curvature {:.6e}",
        info.n_hessians, info.min_curvature
    );
}

fn check_newton_result(
    tag: &str,
    fx: &Fx,
    ints: &Integrals<f64>,
    rdm: &Rdm12,
    opts: &FrozenOrbOpts,
    u: &[f64],
    info: &FrozenOrbInfo,
) {
    print_newton_info(tag, info);
    let tr = newton_tr(opts);
    let dist = check_step_common(tag, fx, ints, rdm, opts, u, info);
    let top = opts.max_step.max(tr.radius_max);
    let pair_bound = top * info.iters as f64;
    assert!(
        info.kappa_norm <= pair_bound * (1.0 + 1e-10) + 1e-14,
        "{tag}: kappa_norm = {:e} exceeds max(max_step, radius_max)·iters = {pair_bound:e}",
        info.kappa_norm
    );
    assert!(
        dist <= std::f64::consts::SQRT_2 * pair_bound * (1.0 + 1e-10) + 1e-14,
        "{tag}: ‖U − I‖_F = {dist:e} exceeds √2·max(max_step, radius_max)·iters"
    );
    let per_iter = ((top / tr.min_radius).ln() / 4f64.ln()).floor() as usize + 2;
    let cap = 1 + (info.iters + 1) * per_iter;
    assert!(
        info.n_evals > info.iters && info.n_evals <= cap,
        "{tag}: n_evals = {} outside [1 + iters, 1 + (iters+1)(⌊log₄(Δ_top/min_radius)⌋+2)] = \
         [{}, {cap}]",
        info.n_evals,
        1 + info.iters
    );
    assert_eq!(
        info.min_curvature.is_nan(),
        info.n_hessians == 0,
        "{tag}: min_curvature = {} with n_hessians = {}: NaN must mean exactly 'no Hessian built'",
        info.min_curvature,
        info.n_hessians
    );
    if info.n_hessians >= 1 {
        assert!(
            info.min_curvature.is_finite(),
            "{tag}: min_curvature = {} is not finite",
            info.min_curvature
        );
    }
    match info.exit {
        FrozenOrbExit::AlreadyCritical => {
            assert_eq!(info.n_hessians, 0, "{tag}: AlreadyCritical built a Hessian");
            assert_eq!(info.n_evals, 1, "{tag}: AlreadyCritical n_evals");
        }
        FrozenOrbExit::GradTol | FrozenOrbExit::MaxIter => assert_eq!(
            info.n_hessians,
            info.iters,
            "{tag}: exit {} with n_hessians {} ≠ iters {}",
            exit_label(info.exit),
            info.n_hessians,
            info.iters
        ),
        FrozenOrbExit::LineSearch => assert!(
            info.n_hessians == info.iters || info.n_hessians == info.iters + 1,
            "{tag}: exit line-search with n_hessians {} ∉ {{iters, iters + 1}} = {{{}, {}}}",
            info.n_hessians,
            info.iters,
            info.iters + 1
        ),
    }
}

fn check_newton_per_step(
    tag: &str,
    ints: &Integrals<f64>,
    rdm: &Rdm12,
    opts: &FrozenOrbOpts,
    info: &FrozenOrbInfo,
) -> Vec<(f64, f64)> {
    let tr = newton_tr(opts);
    let delta0 = opts.max_step.min(tr.radius_max);
    let (mut prev_e, mut prev_path) = (info.e_before, 0.0f64);
    let mut hist = Vec::with_capacity(info.iters);
    for k in 1..=info.iters {
        let ok = FrozenOrbOpts {
            max_iter: k,
            ..*opts
        };
        let (_, ik) = frozen_orbital_step(ints, rdm, &ok);
        assert_eq!(
            ik.iters, k,
            "{tag}: prefix max_iter {k} accepted {} steps",
            ik.iters
        );
        assert!(
            ik.e_after < prev_e,
            "{tag}: step {k} did not lower E strictly: {} → {} (Δ = {:e})",
            prev_e,
            ik.e_after,
            ik.e_after - prev_e
        );
        let step = ik.kappa_norm - prev_path;
        let slack = 1e-12 * (1.0 + ik.kappa_norm);
        assert!(
            step <= tr.radius_max * (1.0 + 1e-12) + slack,
            "{tag}: step {k} has ‖k‖ = {step:e} > radius_max = {:e}",
            tr.radius_max
        );
        if k == 1 {
            assert!(
                step <= delta0 * (1.0 + 1e-12),
                "{tag}: the first step has ‖k‖ = {step:e} > Δ₀ = min(max_step, radius_max) = \
                 {delta0:e}"
            );
        }
        hist.push((ik.g_after, step));
        prev_e = ik.e_after;
        prev_path = ik.kappa_norm;
        if k == info.iters {
            assert_eq!(
                ik.e_after.to_bits(),
                info.e_after.to_bits(),
                "{tag}: the max_iter {k} prefix does not reproduce e_after"
            );
            assert_eq!(
                ik.kappa_norm.to_bits(),
                info.kappa_norm.to_bits(),
                "{tag}: the max_iter {k} prefix does not reproduce kappa_norm"
            );
        }
    }
    hist
}

#[test]
fn tord_n4_newton_step_contract() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        assert_fixture_floors(&fx, &rdm);
        let mut prev = f64::INFINITY;
        for k in 1..=3usize {
            let opts = newton_opts(k);
            let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
            let tag = format!("tord_n4 {} Newton max_iter {k}", fx.label);
            check_newton_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
            assert!(
                info.iters >= 1,
                "{tag}: no accepted step from a non-critical start"
            );
            assert!(
                info.n_hessians >= 1,
                "{tag}: no Hessian built: the Newton method was not dispatched"
            );
            assert!(
                info.e_before - info.e_after > 1e-6,
                "{tag}: the step lowered E by only {:e}",
                info.e_before - info.e_after
            );
            assert!(
                info.e_after <= prev,
                "{tag}: e_after = {} rose above the max_iter = {} value {prev}",
                info.e_after,
                k - 1
            );
            prev = info.e_after;
            if k == 1 {
                assert!(
                    info.kappa_norm <= opts.max_step * (1.0 + 1e-12),
                    "{tag}: the first step has ‖k‖ = {:e} > Δ₀ = max_step = {:e}",
                    info.kappa_norm,
                    opts.max_step
                );
            }
        }

        let u0 = expm_antisym(&scale_flat(&fx.k, 0.5), d);
        let ints0 = rotate_integrals(&fx.ints, &u0);
        let opts = newton_opts(3);
        let (u1, info) = frozen_orbital_step(&ints0, &rdm, &opts);
        let tag = format!("tord_n4 {} Newton from U0", fx.label);
        check_newton_result(&tag, &fx, &ints0, &rdm, &opts, &u1, &info);
        let e = e_rq(fx.space, &fx.ints, &compose_rotations(&u0, &u1, d), &fx.x);
        let dev = (e - info.e_after).abs();
        println!("{tag}: |⟨x|H(rotate(ints, U0·U1))|x⟩ − e_after| = {dev:.1e}");
        assert!(
            dev <= 1e-12 * e.abs().max(1.0),
            "{tag}: the composed rotation does not reproduce e_after: {dev:e}"
        );
        assert!(info.iters >= 1, "{tag}: no accepted step from U0");
    }
}

fn tr_boundary_reference(h: &[f64], n: usize, g: &[f64], delta: f64) -> (Vec<f64>, f64, f64) {
    let (lam, q) = dense_eigen(&flat_to_nested(h, n));
    let ghat: Vec<f64> = (0..n).map(|c| dot_vec(&q[c], g)).collect();
    let gn = dot_vec(g, g).sqrt();
    assert!(
        ghat[0].abs() > 1e-3 * gn,
        "tr_boundary_reference: |q₀ᵀg| = {:e} ≤ 1e-3‖g‖: near the hard case, the reference is \
         ill-conditioned",
        ghat[0].abs()
    );
    let knorm = |mu: f64| {
        (0..n)
            .map(|c| (ghat[c] / (lam[c] + mu)).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let mut lo = (-lam[0]).max(0.0);
    let mut hi = lo + gn / delta + 1.0;
    assert!(
        knorm(hi) < delta,
        "tr_boundary_reference: bad upper bracket"
    );
    for _ in 0..400 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if knorm(mid) > delta {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mu = hi;
    let mut k = vec![0.0f64; n];
    for c in 0..n {
        let coef = -ghat[c] / (lam[c] + mu);
        for r in 0..n {
            k[r] += coef * q[c][r];
        }
    }
    (k, mu, lam[0])
}

fn dev_from_step(u: &[f64], k: &[f64], d: usize) -> f64 {
    max_dev_vec(u, &expm_antisym(&from_lower_half(&neg_vec(k), d), d))
}

#[test]
fn tord_n4_boundary_first_step() {
    let delta0 = 1e-3;
    for fx in fixtures() {
        let d = fx.space.d;
        let n = d * (d - 1) / 2;
        let rdm = Rdm12::of(&fx.x);
        let tag = format!("tord_n4 {} boundary first step", fx.label);
        let h = frozen_orbital_hessian(&fx.ints, &rdm);
        let g = lower_half(&frozen_orbital_gradient(&fx.ints, &rdm));
        let (lam, _) = dense_eigen(&flat_to_nested(&h, n));
        let ghat_norm_over_lam = if lam[0] > 0.0 {
            vdot(&g, &g).sqrt() / lam[n - 1]
        } else {
            f64::INFINITY
        };
        assert!(
            ghat_norm_over_lam > 10.0 * delta0,
            "{tag}: the Newton step may be interior (‖g‖/λ_max = {ghat_norm_over_lam:e}): vacuous"
        );
        let opts = FrozenOrbOpts {
            max_step: delta0,
            ..newton_opts(1)
        };
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        check_newton_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
        assert_eq!(info.iters, 1, "{tag}: the boundary step was not accepted");
        assert_eq!(
            info.n_evals, 2,
            "{tag}: the first boundary trial at Δ₀ = {delta0:e} was rejected"
        );
        let len_dev = (info.kappa_norm - delta0).abs() / delta0;
        assert!(
            len_dev <= 1e-12,
            "{tag}: ‖k‖ = {:e} is not Δ₀ = {delta0:e} (rel dev {len_dev:e}): not a boundary step",
            info.kappa_norm
        );
        let (k_ref, mu, lmin) = tr_boundary_reference(&h, n, &g, delta0);
        let dev = dev_from_step(&u, &k_ref, d);
        let gn = vdot(&g, &g).sqrt();
        let k_sd: Vec<f64> = g.iter().map(|x| -delta0 * x / gn).collect();
        let dev_sd = dev_from_step(&u, &k_sd, d);
        println!(
            "{tag}: λ_min = {lmin:.4e}, μ = {mu:.4e}; max|U − exp(−K_ref)| = {dev:.2e}; the \
             steepest-descent step of the same length misses by {dev_sd:.2e}"
        );
        assert!(
            mu > (-lmin).max(0.0),
            "{tag}: reference μ = {mu:e} not above max(0, −λ_min)"
        );
        assert!(
            dev <= 1e-10,
            "{tag}: U is not the Moré–Sorensen boundary step exp(−K_ref): max dev {dev:e}"
        );
        assert!(
            dev_sd > 1e-8,
            "{tag}: the steepest-descent step matches U to {dev_sd:e}: the arm does not \
             discriminate the model"
        );
    }
}

fn isolated_minimum_fixtures() -> Vec<Fx> {
    let mut v: Vec<Fx> = fixtures()
        .into_iter()
        .filter(|fx| !(fx.space.d == 4 && fx.space.ndn == 0))
        .collect();
    v.push(asym_fixture());
    v
}

#[test]
fn tord_n4_interior_newton_step() {
    for fx in isolated_minimum_fixtures() {
        let d = fx.space.d;
        let n = d * (d - 1) / 2;
        let rdm = Rdm12::of(&fx.x);
        let tag = format!("tord_n4 {} interior Newton step", fx.label);
        let opts = newton_opts(60);
        let (u_star, info_star) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        print_newton_info(&format!("{tag} (preparation)"), &info_star);
        assert!(
            matches!(info_star.exit, FrozenOrbExit::GradTol),
            "{tag}: the preparatory Newton run did not reach GradTol (exit {}, {} iters; see \
             tord_n5)",
            exit_label(info_star.exit),
            info_star.iters
        );
        let u_p = compose_rotations(&u_star, &expm_antisym(&scale_flat(&fx.k, 0.01), d), d);
        let ints_p = rotate_integrals(&fx.ints, &u_p);
        let h = frozen_orbital_hessian(&ints_p, &rdm);
        let g = lower_half(&frozen_orbital_gradient(&ints_p, &rdm));
        let (lam, q) = dense_eigen(&flat_to_nested(&h, n));
        let mut k_n = vec![0.0f64; n];
        for c in 0..n {
            let coef = -dot_vec(&q[c], &g) / lam[c];
            for r in 0..n {
                k_n[r] += coef * q[c][r];
            }
        }
        let kn = vdot(&k_n, &k_n).sqrt();
        println!(
            "{tag}: at the displaced start λ_min = {:.4e}, ‖g‖ = {:.3e}, ‖k_N‖ = {kn:.4e}",
            lam[0],
            vdot(&g, &g).sqrt()
        );
        assert!(
            lam[0] > 1e-3 * lam[n - 1].abs().max(1.0),
            "{tag}: the displaced endpoint is not positive definite (λ_min = {:e}): vacuous",
            lam[0]
        );
        assert!(
            kn < 0.5 * opts.max_step && kn > 1e-6,
            "{tag}: ‖k_N‖ = {kn:e} not safely inside Δ₀ = {}",
            opts.max_step
        );
        let one = newton_opts(1);
        let (u, info) = frozen_orbital_step(&ints_p, &rdm, &one);
        check_newton_result(&tag, &fx, &ints_p, &rdm, &one, &u, &info);
        assert_eq!(
            info.iters, 1,
            "{tag}: the interior Newton step was not accepted"
        );
        assert_eq!(
            info.n_evals, 2,
            "{tag}: the interior Newton step was not the first trial"
        );
        let len_dev = (info.kappa_norm - kn).abs() / kn;
        let dev = dev_from_step(&u, &k_n, d);
        let gn = vdot(&g, &g).sqrt();
        let k_sd: Vec<f64> = g.iter().map(|x| -kn * x / gn).collect();
        let dev_sd = dev_from_step(&u, &k_sd, d);
        println!(
            "{tag}: max|U − exp(−K_N)| = {dev:.2e}, |kappa_norm/‖k_N‖ − 1| = {len_dev:.2e}; \
             steepest descent of the same length misses by {dev_sd:.2e}"
        );
        assert!(
            dev <= 1e-10,
            "{tag}: U is not the Newton step exp(−K_N): max dev {dev:e}"
        );
        assert!(
            len_dev <= 1e-10,
            "{tag}: kappa_norm = {:e} is not ‖k_N‖ = {kn:e}",
            info.kappa_norm
        );
        assert!(
            dev_sd > 1e-6,
            "{tag}: the steepest-descent step matches to {dev_sd:e}: the arm does not discriminate"
        );
    }
}

#[test]
fn tord_n4_newton_bookkeeping() {
    for fx in fixtures() {
        let d = fx.space.d;
        let rdm = Rdm12::of(&fx.x);
        let id = identity_matrix(d);

        let (_, info) = frozen_orbital_step(&fx.ints, &rdm, &FrozenOrbOpts::default());
        let tag = format!("tord_n4 {} default (L-BFGS)", fx.label);
        print_newton_info(&tag, &info);
        assert!(info.iters >= 1, "{tag}: no step");
        assert_eq!(info.n_hessians, 0, "{tag}: L-BFGS reports a Hessian");
        assert!(
            info.min_curvature.is_nan(),
            "{tag}: L-BFGS min_curvature = {} (must be NaN)",
            info.min_curvature
        );

        let crit = FrozenOrbOpts {
            grad_tol: 1e3,
            ..newton_opts(30)
        };
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &crit);
        let tag = format!("tord_n4 {} Newton grad_tol 1e3", fx.label);
        check_newton_result(&tag, &fx, &fx.ints, &rdm, &crit, &u, &info);
        assert!(is_already_critical(&info), "{tag}: not AlreadyCritical");
        assert!(
            u.iter().zip(&id).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{tag}: U ≠ I bitwise"
        );
        assert!(
            info.kappa_norm == 0.0,
            "{tag}: kappa_norm = {:e}",
            info.kappa_norm
        );
        assert!(
            info.e_after.to_bits() == info.e_before.to_bits(),
            "{tag}: e_after ≠ e_before bitwise"
        );

        let zero = newton_opts(0);
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &zero);
        let tag = format!("tord_n4 {} Newton max_iter 0", fx.label);
        check_newton_result(&tag, &fx, &fx.ints, &rdm, &zero, &u, &info);
        assert!(
            matches!(info.exit, FrozenOrbExit::MaxIter),
            "{tag}: exit {} (want max-iter)",
            exit_label(info.exit)
        );
        assert_eq!(info.iters, 0, "{tag}: iters");
        assert_eq!(
            info.n_hessians, 0,
            "{tag}: a Hessian was built at max_iter = 0"
        );
        assert!(
            u.iter().zip(&id).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{tag}: U ≠ I bitwise"
        );
        assert!(
            info.e_after.to_bits() == info.e_before.to_bits(),
            "{tag}: e_after ≠ e_before bitwise"
        );
    }
}

#[test]
fn tord_n4_newton_options_refused() {
    let space = QnSpace::new(4, 2, 1);
    let x = unit_state_421(0x6_7101);
    let rdm = Rdm12::of(&x);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6_7102);
    let ints = ints_of(space, random_sym_t(4, &mut rng), random_sym_v(4, &mut rng));
    let tr = FrozenTrOpts::default();
    let with = |t: FrozenTrOpts| FrozenOrbOpts {
        method: FrozenOrbMethod::Newton(t),
        ..FrozenOrbOpts::default()
    };
    let mut cases: Vec<(FrozenOrbOpts, &str)> = Vec::new();
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        cases.push((
            with(FrozenTrOpts {
                radius_max: bad,
                ..tr
            }),
            "radius_max",
        ));
    }
    for bad in [-1e-3, 0.25, 0.5, f64::NAN] {
        cases.push((with(FrozenTrOpts { eta: bad, ..tr }), "eta"));
    }
    for bad in [0.0, -1e-12, f64::NAN, f64::INFINITY] {
        cases.push((
            with(FrozenTrOpts {
                min_radius: bad,
                ..tr
            }),
            "min_radius",
        ));
    }
    cases.push((
        FrozenOrbOpts {
            max_step: 0.0,
            ..with(tr)
        },
        "max_step",
    ));
    cases.push((
        FrozenOrbOpts {
            armijo_c1: 0.0,
            ..with(tr)
        },
        "armijo_c1",
    ));
    cases.push((
        FrozenOrbOpts {
            grad_tol: f64::NAN,
            ..with(tr)
        },
        "grad_tol",
    ));
    for (opts, field) in cases {
        assert_refused(
            || {
                let _ = frozen_orbital_step(&ints, &rdm, &opts);
            },
            &["frozen_orbital_step", field],
            &format!("Newton, invalid {field}"),
        );
        if field != "grad_tol" {
            let at_gate = FrozenOrbOpts {
                grad_tol: 1e3,
                ..opts
            };
            assert_refused(
                || {
                    let _ = frozen_orbital_step(&ints, &rdm, &at_gate);
                },
                &["frozen_orbital_step", field],
                &format!("Newton, invalid {field} at a gate-critical point"),
            );
        }
    }
    let (_, info) = frozen_orbital_step(&ints, &rdm, &with(FrozenTrOpts { eta: 0.0, ..tr }));
    assert!(info.iters >= 1, "eta = 0: no step taken");
}

#[test]
fn tord_n4_hessian_refusal_and_defaults() {
    let x = unit_state_421(0x6_7103);
    let rdm = Rdm12::of(&x);
    let space5 = QnSpace::new(5, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6_7104);
    let ints5 = ints_of(space5, random_sym_t(5, &mut rng), random_sym_v(5, &mut rng));
    assert_refused(
        || {
            let _ = frozen_orbital_hessian(&ints5, &rdm);
        },
        &["frozen_orbital_hessian", "norb"],
        "frozen_orbital_hessian d mismatch",
    );
    assert!(
        matches!(FrozenOrbOpts::default().method, FrozenOrbMethod::Lbfgs),
        "FrozenOrbOpts::default().method must be Lbfgs"
    );
    let tr = FrozenTrOpts::default();
    assert_eq!(tr.radius_max, 1.0);
    assert_eq!(tr.eta, 1e-4);
    assert_eq!(tr.min_radius, 1e-10);
}

fn lambda_min_at(ints: &Integrals<f64>, rdm: &Rdm12, u: &[f64], n: usize) -> f64 {
    let h = frozen_orbital_hessian(&rotate_integrals(ints, u), rdm);
    dense_eigen(&flat_to_nested(&h, n)).0[0]
}

#[test]
fn tord_n5_newton_converges() {
    for fx in &isolated_minimum_fixtures() {
        let d = fx.space.d;
        let n = d * (d - 1) / 2;
        let rdm = Rdm12::of(&fx.x);
        let opts = newton_opts(60);
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        let tag = format!("tord_n5 {} Newton", fx.label);
        check_newton_result(&tag, fx, &fx.ints, &rdm, &opts, &u, &info);
        let hist: Vec<String> = check_newton_per_step(&tag, &fx.ints, &rdm, &opts, &info)
            .iter()
            .map(|(g, _)| format!("{g:.2e}"))
            .collect();
        let lmin_end = lambda_min_at(&fx.ints, &rdm, &u, n);
        let (_, lb) = frozen_orbital_step(&fx.ints, &rdm, &FrozenOrbOpts::default());
        println!(
            "{tag}: {} iters (budget {}, headroom {:.1}×), E {:.15}, min_curvature {:.4e}, \
             endpoint λ_min {lmin_end:.4e}; ‖g‖ history [{:.2e}, {}]; L-BFGS default: exit {}, \
             E {:.15}, ‖g‖ {:.2e}",
            info.iters,
            opts.max_iter,
            opts.max_iter as f64 / info.iters.max(1) as f64,
            info.e_after,
            info.min_curvature,
            info.g_before,
            hist.join(", "),
            exit_label(lb.exit),
            lb.e_after,
            lb.g_after
        );
        assert!(
            matches!(info.exit, FrozenOrbExit::GradTol),
            "{tag}: exit {} after {} iters (g_after = {:e})",
            exit_label(info.exit),
            info.iters,
            info.g_after
        );
        assert!(
            3 * info.iters <= opts.max_iter,
            "{tag}: {} iterations leave less than 3× headroom in max_iter = {}",
            info.iters,
            opts.max_iter
        );
    }
}

#[test]
fn tord_n5_degenerate_minimum_contract() {
    let fx = fixtures()
        .into_iter()
        .find(|fx| fx.space.d == 4 && fx.space.ndn == 0)
        .expect("the spinless design fixture");
    let rdm = Rdm12::of(&fx.x);
    let opts = newton_opts(30);
    let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
    let tag = format!("tord_n5 {} Newton (degenerate minimum)", fx.label);
    check_newton_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
    assert!(info.iters >= 1, "{tag}: no step");
    check_newton_per_step(&tag, &fx.ints, &rdm, &opts, &info);
    let (_, lb) = frozen_orbital_step(&fx.ints, &rdm, &FrozenOrbOpts::default());
    println!(
        "{tag}: exit {}, {} iters, E {:.15}, ‖g‖ {:.2e}; L-BFGS default: exit {}, E {:.15}, ‖g‖ \
         {:.2e}",
        exit_label(info.exit),
        info.iters,
        info.e_after,
        info.g_after,
        exit_label(lb.exit),
        lb.e_after,
        lb.g_after
    );
}

fn indefinite_start(fx: &Fx, rdm: &Rdm12) -> (Vec<f64>, &'static str) {
    let d = fx.space.d;
    let n = d * (d - 1) / 2;
    let cands = [
        (identity_matrix(d), "I"),
        (expm_antisym(&scale_flat(&fx.k, 0.5), d), "expm(½K)"),
        (expm_antisym(&fx.k, d), "expm(K)"),
    ];
    for (u0, lab) in cands {
        let lmin = lambda_min_at(&fx.ints, rdm, &u0, n);
        let gn = pair_norm(&frozen_orbital_gradient(
            &rotate_integrals(&fx.ints, &u0),
            rdm,
        ));
        println!(
            "tord_n6 {}: candidate {lab}: λ_min = {lmin:.4e}, ‖g‖ = {gn:.3e}",
            fx.label
        );
        if lmin < -1e-2 && gn > 1e-3 {
            return (u0, lab);
        }
    }
    panic!(
        "tord_n6 {}: no candidate start is indefinite: the arm is vacuous",
        fx.label
    )
}

#[test]
fn tord_n6_indefinite_start_descends() {
    for fx in fixtures() {
        let d = fx.space.d;
        let n = d * (d - 1) / 2;
        let rdm = Rdm12::of(&fx.x);
        let (u0, lab) = indefinite_start(&fx, &rdm);
        let tag = format!("tord_n6 {} from {lab}", fx.label);
        let ints0 = rotate_integrals(&fx.ints, &u0);
        let h0 = frozen_orbital_hessian(&ints0, &rdm);
        let g0 = lower_half(&frozen_orbital_gradient(&ints0, &rdm));
        let (lam, _) = dense_eigen(&flat_to_nested(&h0, n));
        let l_start = lam[0];
        let lscale = lam[0].abs().max(lam[n - 1].abs()).max(1.0);

        let one = newton_opts(1);
        let (u, info) = frozen_orbital_step(&ints0, &rdm, &one);
        check_newton_result(&tag, &fx, &ints0, &rdm, &one, &u, &info);
        assert_eq!(
            info.iters, 1,
            "{tag}: no step accepted from the indefinite start"
        );
        assert!(
            info.e_after < info.e_before,
            "{tag}: no strict descent from the indefinite start"
        );
        assert_eq!(info.n_hessians, 1, "{tag}: n_hessians");
        assert!(
            (info.min_curvature - l_start).abs() <= 1e-12 * lscale && info.min_curvature < 0.0,
            "{tag}: min_curvature = {:e} is not λ_min(H at the start) = {l_start:e}",
            info.min_curvature
        );
        let j = info.n_evals - 2;
        let delta_j = one.max_step * 0.25f64.powi(j as i32);
        let len_dev = (info.kappa_norm - delta_j).abs() / delta_j;
        let (k_ref, mu, _) = tr_boundary_reference(&h0, n, &g0, delta_j);
        let dev = dev_from_step(&u, &k_ref, d);
        let (lam_f, q_f) = dense_eigen(&flat_to_nested(&h0, n));
        let mut k_floor = vec![0.0f64; n];
        for c in 0..n {
            let coef = -dot_vec(&q_f[c], &g0) / lam_f[c].abs().max(1e-12);
            for r in 0..n {
                k_floor[r] += coef * q_f[c][r];
            }
        }
        let kfn = vdot(&k_floor, &k_floor).sqrt();
        let k_floor: Vec<f64> = k_floor.iter().map(|x| x * delta_j / kfn).collect();
        let gn = vdot(&g0, &g0).sqrt();
        let k_sd: Vec<f64> = g0.iter().map(|x| -delta_j * x / gn).collect();
        let dev_floor = dev_from_step(&u, &k_floor, d);
        let dev_sd = dev_from_step(&u, &k_sd, d);
        println!(
            "{tag}: λ_start = {l_start:.4e}; {j} rejection(s), Δ = {delta_j:.4e}, ‖k‖ rel dev \
             {len_dev:.1e}; μ_ref = {mu:.4e}; max|U − exp(−K_ref)| = {dev:.2e}; flooring misses by \
             {dev_floor:.2e}, steepest descent by {dev_sd:.2e}"
        );
        assert!(
            len_dev <= 1e-10,
            "{tag}: kappa_norm = {:e} is not max_step·4^(−{j}) = {delta_j:e}: not a boundary step \
             after {j} quartered radii",
            info.kappa_norm
        );
        assert!(
            mu > -l_start && -l_start > 0.0,
            "{tag}: μ_ref = {mu:e} not above −λ_min = {:e} > 0",
            -l_start
        );
        assert!(
            dev <= 1e-9,
            "{tag}: U is not the level-shifted boundary step exp(−K_ref): max dev {dev:e}"
        );
        assert!(
            dev_floor > 1e-6 && dev_sd > 1e-6,
            "{tag}: an alternative step matches U (flooring {dev_floor:e}, steepest descent \
             {dev_sd:e}): the arm does not see the level shift"
        );

        let full = newton_opts(60);
        let (u, info) = frozen_orbital_step(&ints0, &rdm, &full);
        let tag = format!("{tag}, max_iter 60");
        check_newton_result(&tag, &fx, &ints0, &rdm, &full, &u, &info);
        check_newton_per_step(&tag, &ints0, &rdm, &full, &info);
        println!(
            "{tag}: exit {}, {} iters, E {:.15}, min_curvature {:.4e}",
            exit_label(info.exit),
            info.iters,
            info.e_after,
            info.min_curvature
        );
        assert!(info.e_after < info.e_before, "{tag}: no strict descent");
        assert!(
            info.min_curvature <= l_start + 1e-12 * lscale,
            "{tag}: min_curvature = {:e} above λ_start = {l_start:e}: not the minimum over every \
             Hessian built",
            info.min_curvature
        );
        let degenerate = fx.space.d == 4 && fx.space.ndn == 0;
        if !degenerate {
            assert!(
                matches!(info.exit, FrozenOrbExit::GradTol) && 3 * info.iters <= full.max_iter,
                "{tag}: exit {} after {} iters: GradTol with 3× headroom not reached",
                exit_label(info.exit),
                info.iters
            );
        }
    }
}

#[test]
fn tord_n6_rejection_quarters_radius() {
    const R: f64 = 3.0;
    for fx in fixtures() {
        let d = fx.space.d;
        let n = d * (d - 1) / 2;
        let rdm = Rdm12::of(&fx.x);
        let (u0, lab) = indefinite_start(&fx, &rdm);
        let tag = format!("tord_n6 {} from {lab}, Δ₀ = {R}", fx.label);
        let ints0 = rotate_integrals(&fx.ints, &u0);
        let h0 = frozen_orbital_hessian(&ints0, &rdm);
        let g0 = lower_half(&frozen_orbital_gradient(&ints0, &rdm));
        let e0 = energy_from_rdms(&ints0, &rdm);
        let opts = FrozenOrbOpts {
            max_step: R,
            method: FrozenOrbMethod::Newton(FrozenTrOpts {
                radius_max: R,
                ..FrozenTrOpts::default()
            }),
            ..newton_opts(1)
        };
        let (u, info) = frozen_orbital_step(&ints0, &rdm, &opts);
        check_newton_result(&tag, &fx, &ints0, &rdm, &opts, &u, &info);
        assert_eq!(info.iters, 1, "{tag}: no step accepted");
        let j = info.n_evals - 2;
        let mut rhos = Vec::new();
        for i in 0..=j {
            let di = R * 0.25f64.powi(i as i32);
            let (k, _, _) = tr_boundary_reference(&h0, n, &g0, di);
            let mut hk = vec![0.0f64; n];
            for r in 0..n {
                for c in 0..n {
                    hk[r] += h0[r * n + c] * k[c];
                }
            }
            let pred = -(vdot(&g0, &k) + 0.5 * vdot(&k, &hk));
            let w = expm_antisym(&from_lower_half(&neg_vec(&k), d), d);
            let et = energy_from_rdms(&rotate_integrals(&ints0, &w), &rdm);
            rhos.push(format!("Δ {di:.4e}: ρ {:.3e}", (e0 - et) / pred));
        }
        println!("{tag}: {j} rejection(s); {}", rhos.join("; "));
        assert!(
            j >= 1,
            "{tag}: the first trial at Δ₀ = {R} was accepted: the rejection rule is not exercised"
        );
        let delta_j = R * 0.25f64.powi(j as i32);
        let len_dev = (info.kappa_norm - delta_j).abs() / delta_j;
        assert!(
            len_dev <= 1e-10,
            "{tag}: kappa_norm = {:e} is not R·4^(−{j}) = {delta_j:e} after {j} rejection(s): the \
             radius is not quartered",
            info.kappa_norm
        );
        let (k_ref, _, _) = tr_boundary_reference(&h0, n, &g0, delta_j);
        let dev = dev_from_step(&u, &k_ref, d);
        assert!(
            dev <= 1e-9,
            "{tag}: U is not the boundary step exp(−K_ref(Δ_{j})): max dev {dev:e}"
        );
    }
}

#[test]
fn tord_n4_radius_cap() {
    for fx in fixtures() {
        let rdm = Rdm12::of(&fx.x);
        let tag = format!("tord_n4 {} radius_max 0.05", fx.label);
        let opts = FrozenOrbOpts {
            method: FrozenOrbMethod::Newton(FrozenTrOpts {
                radius_max: 0.05,
                ..FrozenTrOpts::default()
            }),
            ..newton_opts(8)
        };
        assert!(
            opts.max_step > 0.05,
            "{tag}: max_step must exceed radius_max"
        );
        let (u, info) = frozen_orbital_step(&fx.ints, &rdm, &opts);
        check_newton_result(&tag, &fx, &fx.ints, &rdm, &opts, &u, &info);
        let hist = check_newton_per_step(&tag, &fx.ints, &rdm, &opts, &info);
        let lens: Vec<String> = hist.iter().map(|(_, l)| format!("{l:.4e}")).collect();
        println!(
            "{tag}: exit {}, {} iters, step lengths [{}]",
            exit_label(info.exit),
            info.iters,
            lens.join(", ")
        );
        assert!(
            hist.iter().skip(1).any(|&(_, l)| l >= 0.99 * 0.05),
            "{tag}: no step after the first reaches the cap: the growth cap is not exercised"
        );
    }
}
