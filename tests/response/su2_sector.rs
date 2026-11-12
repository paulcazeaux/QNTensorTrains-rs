// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::sector_util::{krein_signs, metric_report, residuals_of};

use qntt::core::tangent::{
    TtTangent, tangent_basis as ab_tangent_basis, tangent_dim as ab_tangent_dim,
};
use qntt::core::weingarten::{FloorPolicy, Terms};
use qntt::fcidump::screen_point_group_forbidden;
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::solvers::hess::{
    ModeCheckOpts, NegativeWitness, RpaKernel, RpaMetric, SectorOpts, SectorReadout,
    SectorSpectrum, SympPrecond, SymplecticSpectrum, hessian_grams as ab_hessian_grams,
    symplectic_eigenvalues as ab_symplectic_eigenvalues,
    weingarten_operator as ab_weingarten_operator,
};
use qntt::solvers::riemannian::tangent_bases as ab_tangent_bases;
use qntt::su2::expand::expand;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh};
use qntt::su2::hess::{
    Su2SectorCtx, effham_apply_shifted, effham_gram_shifted, hessian_grams as su2_hessian_grams,
    normal_residual as su2_normal_residual, symplectic_eigenvalues as su2_symplectic_eigenvalues,
    weingarten_operator as su2_weingarten_operator,
};
use qntt::su2::ops::dot as su2_dot;
use qntt::su2::qn::{Su2RankGrid, Su2Space, max_ranks as su2_max_ranks};
use qntt::su2::riemannian::{
    RiemannianOpts, normalize_mut as su2_normalize_mut, riemannian_cg, riemannian_gd,
    tangent_bases as su2_tangent_bases,
};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_csf, tt_randn as su2_tt_randn};
use qntt::su2::weingarten::{
    min_schmidt as su2_min_schmidt, tangent_basis as su2_tangent_basis, tangent_basis_shifted,
    tangent_dim_grids, weingarten_assemble as su2_weingarten_assemble,
};
use qntt::{
    CoeffTensor, Orbital, QnSpace, Scalar, SparseHamiltonian, Spin, TtVector, adag_i_a_j,
    adag_i_adag_k_a_l_a_j, dot as ab_dot, tt_state, xt_h_y as ab_xt_h_y,
};
use qntt_testkit::dense::{dense_eigen, flat_to_nested};
use qntt_testkit::generic_truncated_tv;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const D4: usize = 4;
const NEL: usize = 4;
const TRIPLET: usize = 2;
const QUINTET: usize = 4;
const CHANNELS: [usize; 3] = [0, TRIPLET, QUINTET];

fn pg_tv(pg: PointGroup) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let d = D4;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        t.set(&[p, p], 0.15 * (p as f64 - 0.5 * d as f64));
    }
    t.set(&[0, 3], -0.9);
    t.set(&[3, 0], -0.9);
    t.set(&[1, 2], -1.1);
    t.set(&[2, 1], -1.1);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    for idx in [[0, 3, 1, 2], [1, 2, 0, 3], [3, 0, 2, 1], [2, 1, 3, 0]] {
        v.set(&idx, 0.21);
    }
    screen_point_group_forbidden(&mut t, &mut v, pg)
        .expect("the fixture is symmetry-allowed by construction — the screen must be a no-op");
    (t, v)
}

fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur: Vec<usize> = (0..n).collect();
    if n == 0 {
        return vec![vec![]];
    }
    if n > d {
        return out;
    }
    loop {
        out.push(cur.clone());
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if cur[i] != i + d - n {
                break;
            }
            if i == 0 {
                return out;
            }
        }
        cur[i] += 1;
        for j in i + 1..n {
            cur[j] = cur[j - 1] + 1;
        }
    }
}

fn exact_config_spectrum(space: QnSpace, ham: &SparseHamiltonian<f64>) -> Vec<f64> {
    let d = space.d;
    let states: Vec<TtVector<f64>> = combinations(d, space.nup)
        .into_iter()
        .flat_map(|up| {
            combinations(d, space.ndn)
                .into_iter()
                .map(move |dn| (up.clone(), dn))
        })
        .map(|(up, dn)| {
            let mut occ_up = vec![false; d];
            let mut occ_dn = vec![false; d];
            for &s in &up {
                occ_up[s] = true;
            }
            for &s in &dn {
                occ_dn[s] = true;
            }
            tt_state(&occ_up, &occ_dn)
        })
        .collect();
    let n = states.len();
    let mut hmat = vec![vec![0.0f64; n]; n];
    for (j, sj) in states.iter().enumerate() {
        let hs = ham * sj;
        for (i, si) in states.iter().enumerate() {
            hmat[i][j] = ab_dot(si, &hs);
        }
    }
    dense_eigen(&hmat).0
}

fn spin_labelled_levels(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let half = NEL / 2;
    let sp22 = QnSpace::new(d, half, half);
    let sp31 = QnSpace::new(d, half + 1, half - 1);
    let sp40 = QnSpace::new(d, half + 2, half - 2);
    let ev22 = exact_config_spectrum(
        sp22,
        &SparseHamiltonian::new(sp22, t, v, None, f64::EPSILON),
    );
    let ev31 = exact_config_spectrum(
        sp31,
        &SparseHamiltonian::new(sp31, t, v, None, f64::EPSILON),
    );
    let ev40 = exact_config_spectrum(
        sp40,
        &SparseHamiltonian::new(sp40, t, v, None, f64::EPSILON),
    );
    let near = |e: f64, set: &[f64]| set.iter().any(|p| (e - p).abs() <= 1e-8 * e.abs().max(1.0));
    let singlets: Vec<f64> = ev22.iter().copied().filter(|&e| !near(e, &ev31)).collect();
    let triplets: Vec<f64> = ev31.iter().copied().filter(|&e| !near(e, &ev40)).collect();
    (singlets, triplets, ev40)
}

fn capped_grids(space: &Su2Space, cap: usize) -> Vec<Su2RankGrid> {
    let caps = su2_max_ranks(space);
    (0..=space.d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(space);
            for s in space.bond_sectors(b) {
                r[s] = if cap == 0 {
                    caps[b][s]
                } else {
                    caps[b][s].min(cap)
                };
            }
            r
        })
        .collect()
}

fn su2_grad_norm(ham: &Su2SparseHamiltonian<f64>, y: &Su2TtVector<f64>) -> f64 {
    let lam = su2_rayleigh(ham, y).re_s();
    let res = &ham.apply(y) - &(y * lam);
    let (bl, br) = su2_tangent_bases(y);
    Su2TtTangent::project(&bl, &br, &res).norm()
}

#[allow(clippy::type_complexity)]
fn truncated_reference(
    cap: usize,
    seed: u64,
) -> (
    Su2Space,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
) {
    let (t, v) = generic_truncated_tv(D4);
    let space = Su2Space::new(D4, NEL, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, cap), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check(), "the truncated reference must be a legal train");
    (space, t, v, ham, x)
}

fn full_rank_singlet_ground() -> (
    Su2Space,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
) {
    let (t, v) = generic_truncated_tv(D4);
    let space = Su2Space::new(D4, NEL, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let grids = capped_grids(&space, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_A0);
    let mut x0: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng);
    x0.scale_mut(1.0 / x0.norm_lr(false));

    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-14;
    opts.gradtol = 1e-14;
    opts.max_iter = 600;
    opts.preserve_grid = true;
    let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
    let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
    let mut rounds = 0usize;
    for _ in 0..12 {
        if su2_grad_norm(&ham, &x) < 1e-12 {
            break;
        }
        let (_l2, x2) = riemannian_cg(&ham, &x, &opts);
        x = x2;
        rounds += 1;
    }
    su2_normalize_mut(&mut x);
    assert!(x.check());
    println!(
        "csr full-rank ground: {rounds} polish rounds (budget 12), ||grad|| = {:e}",
        su2_grad_norm(&ham, &x)
    );
    (space, t, v, ham, x)
}

fn number_op<T: Scalar>(v: &TtVector<T>, site: usize, spin: Spin) -> TtVector<T> {
    let o = Orbital::new(site, spin);
    adag_i_a_j(v, o, o, T::one())
}

fn sz_apply<T: Scalar>(v: &TtVector<T>) -> TtVector<T> {
    let d = v.space.d;
    let mut terms: Vec<TtVector<T>> = Vec::with_capacity(2 * d);
    for p in 0..d {
        terms.push(number_op(v, p, Spin::Up));
    }
    for p in 0..d {
        terms.push(&number_op(v, p, Spin::Dn) * T::from_f64(-1.0));
    }
    let mut acc = terms[0].clone();
    for t in &terms[1..] {
        acc = &acc + t;
    }
    &acc * T::from_f64(0.5)
}

fn s_minus_s_plus<T: Scalar>(v: &TtVector<T>) -> TtVector<T> {
    let d = v.space.d;
    let mut terms: Vec<TtVector<T>> = Vec::new();
    for p in 0..d {
        let ndn_v = number_op(v, p, Spin::Dn);
        terms.push(ndn_v.clone());
        let nup_ndn_v = number_op(&ndn_v, p, Spin::Up);
        terms.push(&nup_ndn_v * T::from_f64(-1.0));
    }
    for p in 0..d {
        for q in 0..d {
            if p == q {
                continue;
            }
            let i = Orbital::new(p, Spin::Up);
            let j = Orbital::new(q, Spin::Up);
            let k = Orbital::new(q, Spin::Dn);
            let l = Orbital::new(p, Spin::Dn);
            terms.push(adag_i_adag_k_a_l_a_j(v, i, j, k, l, T::from_f64(-1.0)));
        }
    }
    let mut acc = terms[0].clone();
    for t in &terms[1..] {
        acc = &acc + t;
    }
    acc
}

fn s_squared<T: Scalar>(v: &TtVector<T>) -> TtVector<T> {
    let smsp = s_minus_s_plus(v);
    let sz = sz_apply(v);
    let sz2 = sz_apply(&sz);
    let sum1 = &smsp + &sz2;
    &sum1 + &sz
}

fn rand_coef(rng: &mut Xoshiro256PlusPlus) -> f64 {
    use rand::RngExt;
    rng.random_range(-1.0..1.0)
}

fn s2_residual(v: &TtVector<f64>) -> f64 {
    let n = v.norm_lr(false);
    assert!(n > 0.0, "s2_residual: the state must be nonzero");
    (&s_squared(v) - &(v * 2.0)).norm_lr(false) / n
}

fn random_spin_state(space: Su2Space, two_s: usize, r: &[Su2RankGrid], seed: u64) -> TtVector<f64> {
    let out_space = space.with_two_s(two_s);
    let caps = su2_max_ranks(&out_space);
    let widest = (0..=space.d)
        .flat_map(|b| space.bond_sectors(b).into_iter().map(move |s| r[b][s]))
        .max()
        .unwrap_or(1)
        .max(1);
    let grids: Vec<Su2RankGrid> = (0..=space.d)
        .map(|b| {
            let mut g = Su2RankGrid::zeros(&out_space);
            for s in out_space.bond_sectors(b) {
                g[s] = caps[b][s].min(widest);
            }
            g
        })
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut y: Su2TtVector<f64> = su2_tt_randn(out_space, &grids, &mut rng);
    y.scale_mut(1.0 / y.norm_lr(false));
    assert!(y.check(), "the control train must be legal in space_out");
    expand(&y, 0)
}

fn gram_eigs(g: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(g.len(), n * n, "gram is not n x n");
    dense_eigen(&flat_to_nested(g, n)).0
}

fn assert_multiset_eq(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: level COUNT {} != {}\n got  {got:?}\n want {want:?}",
        got.len(),
        want.len()
    );
    for (j, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "{what}: level {j} is {a} but the oracle says {b} (tol {tol:e})"
        );
    }
}

fn iota_gram(
    h_out: &Su2SparseHamiltonian<f64>,
    lambda: f64,
    basis: &[Su2TtTangent<f64>],
) -> Vec<f64> {
    let n = basis.len();
    let iv: Vec<Su2TtVector<f64>> = basis.iter().map(|b| b.to_vector()).collect();
    for (i, w) in iv.iter().enumerate() {
        assert_eq!(
            w.space, h_out.space,
            "O1: iota(b{i}) must live in h_out's own space, or `apply` is answering for a \
             different target"
        );
        assert!(w.check(), "O1: iota(b{i}) must be a legal train");
    }
    let mut g = vec![0.0f64; n * n];
    for j in 0..n {
        let hw = h_out.apply(&iv[j]);
        for i in 0..n {
            g[i * n + j] = su2_dot(&iv[i], &hw) - lambda * su2_dot(&iv[i], &iv[j]);
        }
    }
    g
}

#[test]
#[should_panic(expected = "not a cross-sector channel")]
fn csr_zero_flux_channel_is_refused_at_construction() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_01);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let _ = Su2SectorCtx::new(&ctx, &h_out, 0);
}

#[test]
#[should_panic(expected = "singlet")]
fn csr_non_singlet_reference_is_refused_at_construction() {
    let (t, v) = generic_truncated_tv(D4);
    let base = Su2Space::new(D4, NEL, TRIPLET);
    let ham = Su2SparseHamiltonian::new(base, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_02);
    let mut x: Su2TtVector<f64> = su2_tt_randn(base, &capped_grids(&base, 2), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    let ctx = su2_weingarten_operator(&ham, &x);
    let out = base.with_two_s(QUINTET);
    let h_out = Su2SparseHamiltonian::new(out, &t, &v, None, f64::EPSILON);
    let _ = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
}

#[test]
#[should_panic(expected = "sector_lowest_tda")]
fn csr_rpa_readout_is_refused_in_sector_lowest_tda() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_03);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        ..Default::default()
    };
    let _ = sctx.sector_lowest_tda(2, &opts);
}

#[test]
fn csr_effham_gram_shifted_matches_the_iota_and_h_out_construction() {
    for &(label, cap) in &[("cap 2 (truncated)", 2usize), ("full rank", 0usize)] {
        let (space, t, v, ham, x) = truncated_reference(cap, 0xC5_10 + cap as u64);
        let (bl, br) = su2_tangent_bases(&x);
        let lambda = su2_rayleigh(&ham, &x).re_s();
        let out_space = space.with_two_s(TRIPLET);
        let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
        assert_eq!(h_out.space.two_s, TRIPLET);

        let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
        let n = basis.len();
        println!("csr O1 [{label}]: dim = {n}, lambda = {lambda}");
        assert!(
            n > 1,
            "ANTI-VACUITY [{label}]: the triplet channel has {n} directions"
        );

        let got = effham_gram_shifted(&h_out, &bl, &br, TRIPLET, lambda, &basis);
        let want = iota_gram(&h_out, lambda, &basis);
        assert_eq!(got.len(), n * n);
        let scale = want.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        assert!(
            scale > 1e-3,
            "ANTI-VACUITY [{label}]: the Gram is identically zero (scale {scale:e})"
        );
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (got[i * n + j] - want[i * n + j]).abs() <= 1e-12 * scale,
                    "[{label}] O1 entry ({i},{j}): effham_gram_shifted {} vs the iota-and-H_out \
                     construction {}",
                    got[i * n + j],
                    want[i * n + j]
                );
                assert!(
                    (got[i * n + j] - got[j * n + i]).abs() <= 1e-11 * scale,
                    "[{label}] the A Gram must be SYMMETRIC ({i},{j}): {} vs {} — asymmetry is \
                     the H9-a signature (plain adjoint substituted for the θ²-weighted rproject), \
                     which no shape assertion sees",
                    got[i * n + j],
                    got[j * n + i]
                );
            }
        }

        let ctx = su2_weingarten_operator(&ham, &x);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        assert_eq!(sctx.two_s_out(), TRIPLET);
        assert_eq!(sctx.space_out(), out_space);
        assert_eq!(
            sctx.lambda(),
            lambda,
            "the wrapper must bind ctx's own lambda"
        );
        let via_exact = sctx.a_gram_exact(&basis);
        for (k, (a, b)) in via_exact.iter().zip(got.iter()).enumerate() {
            assert_eq!(
                a, b,
                "[{label}] Su2SectorCtx::a_gram_exact must be effham_gram_shifted at the SAME \
                 frames and lambda, bit for bit (entry {k}) — a re-derived context would silently \
                 decouple the applies from the caller's basis"
            );
        }
        let via_ctx = sctx.a_gram(&basis);
        assert_eq!(via_ctx.len(), got.len());
        let mut worst_gram = 0.0f64;
        for (k, (a, b)) in via_ctx.iter().zip(got.iter()).enumerate() {
            worst_gram = worst_gram.max((a - b).abs());
            assert!(
                (a - b).abs() <= 1e-11 * scale,
                "[{label}] Su2SectorCtx::a_gram (FUSED) entry {k}: {a} vs effham_gram_shifted {b} \
                 (scale {scale:e})"
            );
        }
        let mut worst_ag = 0.0f64;
        for (i, bi) in basis.iter().enumerate() {
            let ab = sctx.a_apply(bi);
            for (j, bj) in basis.iter().enumerate() {
                worst_ag = worst_ag.max((bj.dot(&ab) - got[j * n + i]).abs());
                assert!(
                    (bj.dot(&ab) - got[j * n + i]).abs() <= 1e-11 * scale,
                    "[{label}] a_apply and a_gram disagree at ({j},{i})"
                );
            }
        }
        println!(
            "csr O1 [{label}]: fused-vs-exact worst |dG| = {worst_gram:e}, worst \
             |<bj, A bi> - G| = {worst_ag:e} (scale {scale:e})"
        );
    }
}

#[test]
fn csr_sector_ctx_is_a_consistent_wrapper() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_15);
    let (bl, br) = su2_tangent_bases(&x);
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);

    let (cbl, cbr) = sctx.bases();
    assert_eq!(
        (cbl.space, cbr.space),
        (space, space),
        "bases() must return the ORIGINAL frames in x's own space; the shifted pair is built \
         internally where it is needed"
    );
    assert_eq!(
        cbl.ranks, bl.ranks,
        "bases() must be the same gauge, not a re-orthogonalization"
    );
    assert_eq!(cbr.ranks, br.ranks);
    for k in 0..space.d {
        for (name, a, b) in [
            ("base_l", cbl.core(k), bl.core(k)),
            ("base_r", cbr.core(k), br.core(k)),
        ] {
            assert_eq!(a.num_coeffs(), b.num_coeffs(), "{name} core {k}");
            for i in 0..a.num_coeffs() {
                assert_eq!(
                    a.coeff(i),
                    b.coeff(i),
                    "{name} core {k} coeff {i}: the context must BIND the frames it was built on, \
                     bit for bit — a second tangent_bases(x) is a different gauge for the same point"
                );
            }
        }
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC517);
    let dressed = tangent_basis_shifted(&bl, &br, TRIPLET);
    assert!(
        !dressed.is_empty(),
        "ANTI-VACUITY: the triplet channel is empty"
    );
    let mut y = dressed[0].to_vector();
    for b in dressed.iter().skip(1) {
        y = &y + &(&b.to_vector() * rand_coef(&mut rng));
    }
    let p_ctx = sctx.project(&y);
    let p_free = Su2TtTangent::project_shifted(&bl, &br, TRIPLET, &y);
    let n = p_free.norm();
    assert!(n > 1e-6, "ANTI-VACUITY: the projection vanished");
    assert!(
        p_ctx.sub(&p_free).norm() <= 1e-12 * n,
        "Su2SectorCtx::project != project_shifted at the bound frames"
    );
    assert_eq!(p_ctx.two_s_flux(), TRIPLET);
}

#[test]
fn csr_a_apply_exact_is_the_free_function_bit_for_bit() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_12);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);

    let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
    assert!(basis.len() > 1, "ANTI-VACUITY: dim {}", basis.len());
    let mut worst = 0.0f64;
    for (b, xi) in basis.iter().enumerate() {
        let via_ctx = sctx.a_apply_exact(xi);
        let via_free = effham_apply_shifted(&h_out, &bl, &br, TRIPLET, lambda, xi);
        assert_eq!(
            via_ctx.two_s_flux(),
            TRIPLET,
            "vector {b}: the image's flux"
        );
        assert_eq!(
            via_ctx.rank_col(),
            xi.rank_col(),
            "vector {b}: the image must live at the channel's own column grids, or A's range is not \
             T^(S=1)"
        );
        assert_eq!(
            via_ctx.num_coeffs(),
            via_free.num_coeffs(),
            "vector {b}: coefficient count"
        );
        worst = worst.max(via_ctx.norm());
        for i in 0..via_ctx.num_coeffs() {
            assert_eq!(
                via_ctx.coeff(i),
                via_free.coeff(i),
                "vector {b}, coeff {i}: Su2SectorCtx::a_apply_exact must BE effham_apply_shifted at the \
                 bound frames, flux and lambda — bit for bit; a re-derived context or a rebound \
                 lambda is invisible to a Gram comparison in the directions the basis does not \
                 resolve"
            );
        }
    }
    assert!(
        worst > 1e-6,
        "ANTI-VACUITY: A annihilated every basis direction (worst image norm {worst:e})"
    );
    println!(
        "csr a_apply_exact: {} directions, worst |A xi| = {worst:e}",
        basis.len()
    );
}

#[test]
#[should_panic(expected = "shifted tangent")]
fn csr_unshifted_apply_refuses_a_shifted_tangent() {
    let (_space, _t, _v, ham, x) = truncated_reference(2, 0xC5_14);
    let (bl, br) = su2_tangent_bases(&x);
    let ctx = su2_weingarten_operator(&ham, &x);
    let shifted = tangent_basis_shifted(&bl, &br, TRIPLET);
    assert!(
        !shifted.is_empty(),
        "premise: the triplet channel must be non-empty, or the refusal could pass vacuously"
    );
    assert_eq!(shifted[0].two_s_flux(), TRIPLET, "premise: the flux");
    let _ = ctx.a_apply(&shifted[0]);
}

#[test]
#[should_panic(expected = "spin flux")]
fn csr_effham_apply_shifted_refuses_the_wrong_flux() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_13);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let quintet = tangent_basis_shifted(&bl, &br, QUINTET);
    assert_eq!(
        quintet.len(),
        1,
        "premise: the quintet channel is 1-dimensional"
    );
    let _ = effham_apply_shifted(&h_out, &bl, &br, TRIPLET, lambda, &quintet[0]);
}

#[test]
fn csr_full_rank_triplet_levels_are_the_abelian_triplets() {
    let (space, t, v, ham, x) = full_rank_singlet_ground();
    let (singlets, triplets, quintets) = spin_labelled_levels(D4, &t, &v);
    assert_eq!(
        (singlets.len(), triplets.len(), quintets.len()),
        (20, 15, 1),
        "the sector-degeneracy labelling is measured 20 / 15 / 1 at d = 4, N = 4 (accidental \
         degeneracy at this fixture would break the labelling, not the layer)"
    );
    let e0 = singlets[0];
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let g = su2_grad_norm(&ham, &x);
    let (bl, br) = su2_tangent_bases(&x);
    let sigma_min = su2_min_schmidt(&bl, &br);
    println!(
        "csr full-rank: lambda = {lambda}, E0 = {e0}, ||grad|| = {g:e}, sigma_min = {sigma_min:e}"
    );

    assert!(
        g < 1e-10,
        "PRECONDITION: the reference must be critical (||grad|| = {g:e}); B is linear in r_perp \
         and the A-only claim rests on r_perp = 0"
    );
    assert!(
        (lambda - e0).abs() <= 1e-9 * e0.abs().max(1.0),
        "PRECONDITION: lambda = {lambda} must be the exact singlet GROUND E_0 = {e0}"
    );
    let caps = capped_grids(&space, 0);
    for (b, cap) in caps.iter().enumerate() {
        assert_eq!(
            bl.ranks[b], *cap,
            "PRECONDITION: the reference must sit at the max_ranks grid at bond {b} — that is \
             what makes T^{{S=1}} the whole triplet sector"
        );
    }

    let dim = tangent_dim_grids(&space, &bl.ranks, TRIPLET);
    let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
    println!("csr full-rank: dim T^(S=1) = {dim}, basis {}", basis.len());
    assert_eq!(
        (dim, basis.len()),
        (15, 15),
        "PRECONDITION: dim T^{{S=1}} must equal the triplet sector's 15 multiplets, or eig(A) is a \
         Cauchy upper bound and this arm's equality is not a theorem"
    );

    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    assert_eq!(
        sctx.dim(),
        dim,
        "the readout must report the structural dim"
    );
    let got: Vec<f64> = gram_eigs(&sctx.a_gram(&basis), 15)
        .into_iter()
        .map(|z| z + e0)
        .collect();
    println!("csr full-rank triplets:\n got  {got:?}\n want {triplets:?}");
    assert_multiset_eq(
        &got,
        &triplets,
        1e-8,
        "eig(A^{S=1}) + E_0 vs the 15 abelian triplet levels",
    );

    assert!(
        triplets[0] > e0 + 1e-6,
        "ANTI-VACUITY: the lowest triplet ({}) must sit strictly above the singlet ground ({e0}), \
         or the channel is not a genuine wall",
        triplets[0]
    );
    assert!(
        got[0] - e0 > 1e-6,
        "the lowest triplet EXCITATION energy must be positive at the singlet ground (got \
         eig(A)[0] = {})",
        got[0] - e0
    );
}

#[test]
fn csr_whole_sector_channels_are_exact_at_a_truncated_non_critical_reference() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_20);
    let (_singlets, triplets, quintets) = spin_labelled_levels(D4, &t, &v);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let g = su2_grad_norm(&ham, &x);
    println!(
        "csr truncated: lambda = {lambda}, ||grad|| = {g:e}, sigma_min = {:e}",
        su2_min_schmidt(&bl, &br)
    );
    assert!(
        g > 1e-2,
        "ANTI-VACUITY: this arm must run at a NON-critical point, or it is the full-rank arm in \
         disguise and the base-point-freedom of the corollary is untested (||grad|| = {g:e})"
    );
    let full = capped_grids(&space, 0);
    assert!(
        (0..=D4).any(|b| bl.ranks[b] != full[b]),
        "ANTI-VACUITY: the cap-2 grid coincides with max_ranks, so nothing is truncated"
    );

    let ctx = su2_weingarten_operator(&ham, &x);
    for (two_s_flux, want_levels) in [(TRIPLET, &triplets), (QUINTET, &quintets)] {
        let dim = tangent_dim_grids(&space, &bl.ranks, two_s_flux);
        let basis = tangent_basis_shifted(&bl, &br, two_s_flux);
        println!(
            "csr truncated 2k = {two_s_flux}: dim {dim}, basis {}, sector {}",
            basis.len(),
            want_levels.len()
        );
        assert_eq!(
            (dim, basis.len()),
            (want_levels.len(), want_levels.len()),
            "PRECONDITION (2k = {two_s_flux}): the corollary applies only where T^{{S'}} is the \
             WHOLE sector — dim {dim} vs {} multiplets",
            want_levels.len()
        );
        let out_space = space.with_two_s(two_s_flux);
        let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_s_flux);
        let got: Vec<f64> = gram_eigs(&sctx.a_gram(&basis), dim)
            .into_iter()
            .map(|z| z + lambda)
            .collect();
        assert_multiset_eq(
            &got,
            want_levels,
            1e-10,
            &format!("2k = {two_s_flux}: eig(A) + lambda vs the exact sector levels"),
        );
    }
}

#[test]
fn csr_proper_subspace_channel_reports_dim_and_only_interlaces() {
    let (space, t, v, ham, x) = truncated_reference(1, 0xC5_30);
    let (_singlets, triplets, _q) = spin_labelled_levels(D4, &t, &v);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();

    let dim = tangent_dim_grids(&space, &bl.ranks, TRIPLET);
    let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
    println!(
        "csr cap 1: dim T^(S=1) = {dim}, span {}, sector 15, lambda = {lambda}",
        basis.len()
    );
    assert_eq!(
        dim, 13,
        "the cap-1 triplet channel is measured 13-dimensional (C-spec §1.5)"
    );
    assert!(
        dim < triplets.len(),
        "ANTI-VACUITY: this arm needs a channel STRICTLY inside its sector, or the interlacing \
         assertions are the equality ones in disguise"
    );
    assert!(basis.len() <= dim, "span {} vs dim {dim}", basis.len());
    assert!(
        basis.len() > 1,
        "ANTI-VACUITY: the channel must be non-trivial"
    );

    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    assert_eq!(sctx.dim(), dim);

    let ritz: Vec<f64> = gram_eigs(&sctx.a_gram(&basis), basis.len())
        .into_iter()
        .map(|z| z + lambda)
        .collect();
    let (lo, hi) = (triplets[0], triplets[triplets.len() - 1]);
    println!("csr cap 1 ritz: {ritz:?} in [{lo}, {hi}]");
    for (j, w) in ritz.iter().enumerate() {
        assert!(
            *w >= lo - 1e-8 && *w <= hi + 1e-8,
            "A^{{S=1}} is a COMPRESSION of H|_{{2S=2}} - lambda to a proper subspace, so Ritz \
             value {j} ({w}) must lie in [{lo}, {hi}]"
        );
    }

    let opts = SectorOpts {
        tol: 1e-9,
        max_iter: 300,
        readout: SectorReadout::Tda,
        seed: 0xC5_31,
        ..Default::default()
    };
    let res = sctx.sector_lowest_tda(3, &opts);
    println!(
        "csr cap 1 readout: dim {}, two_s_out {}, converged {}, iters {}, applies {}, \
         sigma_min {:e}, theta {:?}",
        res.dim,
        res.two_s_out,
        res.converged,
        res.iterations,
        res.applies,
        res.sigma_min,
        res.theta()
    );
    assert_eq!(res.dim, dim, "the readout must report the STRUCTURAL dim");
    assert_eq!(res.two_s_out, TRIPLET);
    assert_eq!(res.space, out_space);
    assert!(res.omega2().is_none(), "a Tda readout carries no omega^2");
    let theta = res.theta().expect("a Tda readout carries theta");
    assert!(
        theta.len() <= 3,
        "the readout must never return MORE roots than requested"
    );
    assert!(
        theta.iter().all(|z| z.is_finite()),
        "a short-and-unconverged readout is a SPAN statement, never a NaN"
    );
    assert!(
        (res.sigma_min - su2_min_schmidt(&bl, &br)).abs() <= 1e-14,
        "sigma_min must be the CENTERWEIGHTED min_schmidt of the base point (SU(2)-specific — \
         never a raw sigma)"
    );
}

#[test]
fn csr_expanded_abelian_spin_block_certifies_a() {
    for cap in [2usize, 1] {
        let (space, t, v, ham, x) = truncated_reference(cap, 0xC5_40 + cap as u64);
        let (bl, br) = su2_tangent_bases(&x);
        let lambda = su2_rayleigh(&ham, &x).re_s();

        let x_ab = expand(&x, 0);
        let space_ab = QnSpace::new(D4, NEL / 2, NEL / 2);
        assert_eq!(
            x_ab.space, space_ab,
            "expand(x, 0) must land in the (N/2, N/2) abelian sector"
        );
        assert!(x_ab.check());
        let h_ab = SparseHamiltonian::<f64>::new(space_ab, &t, &v, None, f64::EPSILON);
        let lam_ab = qntt::rayleigh_quotient(&h_ab, &x_ab).re_s();
        assert!(
            (lam_ab - lambda).abs() <= 1e-12 * lambda.abs().max(1.0),
            "PREMISE: expand preserves inner products exactly, so the two layers' lambda must \
             agree: SU(2) {lambda} vs abelian {lam_ab}"
        );
        let (bl_ab, br_ab) = ab_tangent_bases(&x_ab);
        let basis_ab: Vec<TtTangent<f64>> = ab_tangent_basis(&bl_ab, &br_ab, false);
        let n = basis_ab.len();
        let dims: Vec<usize> = CHANNELS
            .iter()
            .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
            .collect();
        println!("csr O2 [cap {cap}]: abelian tangent dim {n}, per-channel {dims:?}");
        assert_eq!(
            n,
            ab_tangent_dim(&bl_ab),
            "the abelian enumerator must fill its own structural dimension at this point"
        );
        assert_eq!(
            dims.iter().sum::<usize>(),
            n,
            "★ the sum identity must hold before the spin split can be a partition"
        );

        let basis_su2_probe = tangent_basis_shifted(&bl, &br, TRIPLET);
        assert!(
            !basis_su2_probe.is_empty(),
            "ANTI-VACUITY [cap {cap}]: the triplet channel is empty"
        );
        let control = s2_residual(&random_spin_state(
            space,
            TRIPLET,
            &bl.ranks,
            0xC5_4C + cap as u64,
        ));
        let mut worst = 0.0f64;
        for (i, xi) in basis_su2_probe.iter().enumerate() {
            let w = expand(&xi.to_vector(), 0);
            let nw = w.norm_lr(false);
            assert!(
                nw > 1e-6,
                "[cap {cap}] iota(b{i}) must have a nonzero Sz = 0 slice"
            );
            let rel = s2_residual(&w);
            worst = worst.max(rel);
            assert!(
                rel <= 1e-11,
                "[cap {cap}] expand(iota(b{i}), 0) must be an S^2 eigenstate with eigenvalue \
                 S(S+1) = 2: ||S^2 v - 2 v|| / ||v|| = {rel:e} (wire-free control at the same \
                 ranks: {control:e}). A failure here says ι(ξ) is not a LEGAL spin-1 train in \
                 space_out, or that expand/s_squared do not reproduce S(S+1) on it — not that a \
                 recoupling coefficient is wrong (see this leg's note: the eigenvalue is carried by \
                 the representation, and the coefficient certificates are R1, P's self-adjointness \
                 and the CAP-1 comparison below). Do NOT relax it: measured 1.9e-15 / 1.2e-15 with \
                 norm_lr, four orders below the bound"
            );
        }
        println!(
            "csr O2 [cap {cap}]: S^2 residual worst {worst:e} over the shifted basis, control \
             (wire-free spin-1 train) {control:e}"
        );

        let iv: Vec<TtVector<f64>> = basis_ab.iter().map(|b| b.to_vector()).collect();
        for (i, a) in iv.iter().enumerate() {
            for (j, b) in iv.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ab_dot(a, b) - want).abs() <= 1e-10,
                    "PREMISE [cap {cap}]: the abelian tangent basis must be orthonormal in the \
                     embedded dot: <iota(b{i}), iota(b{j})> = {}",
                    ab_dot(a, b)
                );
            }
        }
        let mut sg = vec![vec![0.0f64; n]; n];
        for j in 0..n {
            let w = s_squared(&iv[j]);
            for i in 0..n {
                sg[i][j] = ab_dot(&iv[i], &w);
            }
        }
        let (sev, svec) = dense_eigen(&sg);
        println!(
            "csr O2 [cap {cap}]: S^2 Gram spectrum {:?}",
            sev.iter()
                .map(|z| (z * 1e6).round() / 1e6)
                .collect::<Vec<_>>()
        );
        let mut want_spec: Vec<f64> = Vec::new();
        for (k, &dk) in dims.iter().enumerate() {
            let s = k as f64;
            want_spec.extend(std::iter::repeat_n(s * (s + 1.0), dk));
        }
        want_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_multiset_eq(
            &sev,
            &want_spec,
            1e-10,
            &format!(
                "[cap {cap}] the S^2 Gram spectrum must be {{0, 2, 6}} with multiplicities \
                 {dims:?} — this single assertion IS the theorem T_x M_ab = (+)_S' expand(T^S')"
            ),
        );

        let idx: Vec<usize> = (0..n).filter(|&c| (sev[c] - 2.0).abs() <= 1e-8).collect();
        assert_eq!(
            idx.len(),
            dims[1],
            "[cap {cap}] the eigenvalue-2 eigenspace must have dimension dim T^{{S=1}}"
        );
        assert!(
            !idx.is_empty(),
            "ANTI-VACUITY [cap {cap}]: the triplet block is empty"
        );
        let basis_s1: Vec<TtTangent<f64>> = idx
            .iter()
            .map(|&c| {
                let coef = &svec[c];
                let mut acc = basis_ab[0].scale(coef[0]);
                for (i, b) in basis_ab.iter().enumerate().skip(1) {
                    acc.axpy(coef[i], b);
                }
                acc
            })
            .collect();
        for (i, a) in basis_s1.iter().enumerate() {
            for (j, b) in basis_s1.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (a.dot(b) - want).abs() <= 1e-9,
                    "[cap {cap}] the S=1 block basis must stay orthonormal: <c{i}, c{j}> = {}",
                    a.dot(b)
                );
            }
        }
        let ctx_ab = ab_weingarten_operator(&h_ab, &x_ab);
        let (a1, _b1) = ab_hessian_grams(&ctx_ab, &basis_s1);
        let got_ab = gram_eigs(&a1, basis_s1.len());

        let out_space = space.with_two_s(TRIPLET);
        let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
        let basis_su2 = tangent_basis_shifted(&bl, &br, TRIPLET);
        assert_eq!(
            basis_su2.len(),
            basis_s1.len(),
            "[cap {cap}] the two layers must be describing the same space"
        );
        let got_su2 = gram_eigs(
            &effham_gram_shifted(&h_out, &bl, &br, TRIPLET, lambda, &basis_su2),
            basis_su2.len(),
        );
        println!("csr O2 [cap {cap}]:\n abelian S=1 {got_ab:?}\n su2 A^S=1   {got_su2:?}");
        let scale = got_su2.iter().fold(1.0f64, |m, z| m.max(z.abs()));
        assert!(
            scale > 1e-3,
            "ANTI-VACUITY [cap {cap}]: the A block is identically zero"
        );
        assert_multiset_eq(
            &got_su2,
            &got_ab,
            1e-10 * scale,
            &format!(
                "[cap {cap}] eig(A^{{S=1}}) vs the S=1 block of the expanded abelian A-Gram (NO \
                 metric conversion enters: both bases are orthonormal in their own layer's metric \
                 and expand∘iota is an isometry between them)"
            ),
        );
    }
}

#[test]
fn csr_sector_lowest_tda_matches_the_dense_a_gram() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_50);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let sigma_min = su2_min_schmidt(&bl, &br);

    let basis = sctx.basis();
    assert!(
        basis.len() > 3,
        "ANTI-VACUITY: dim {} is too small to restart",
        basis.len()
    );
    let dense = gram_eigs(&sctx.a_gram(&basis), basis.len());

    let k = 3usize;
    let opts = SectorOpts {
        tol: 1e-10,
        max_iter: 300,
        readout: SectorReadout::Tda,
        seed: 0xC5_51,
        ..Default::default()
    };
    let res = sctx.sector_lowest_tda(k, &opts);
    println!(
        "csr tda: dim {}, converged {}, iters {}, applies {}, theta {:?}\n     dense {:?}",
        res.dim,
        res.converged,
        res.iterations,
        res.applies,
        res.theta(),
        &dense[..k.min(dense.len())]
    );
    assert!(
        res.converged,
        "the readout must converge on a live 15-dimensional channel with a 300-iteration budget \
         (short-and-unconverged there would be a span statement about a space that is not short)"
    );
    let theta = res.theta().expect("Tda readout");
    assert_eq!(theta.len(), k, "all {k} requested roots must come back");
    for (j, (got, want)) in theta.iter().zip(dense.iter()).enumerate() {
        assert!(
            (got - want).abs() <= 1e-7 * want.abs().max(1.0),
            "root {j}: the iterative readout {got} vs the dense a_gram {want}"
        );
    }
    assert!(
        theta[0] >= dense[0] - 1e-8,
        "a projected Davidson inside range(P) cannot go BELOW the dense spectrum's minimum — a \
         Ritz value under {} is a leaking projector, not a converged root",
        dense[0]
    );
    assert!(res.applies > 0, "a live channel must cost applies");
    assert!(
        (res.sigma_min - sigma_min).abs() <= 1e-14,
        "the readout must quote the base point's CENTERWEIGHTED min_schmidt ({sigma_min:e}), \
         which is base-point data identical across channels — never a raw sigma"
    );
    let _ = lambda;
}

#[test]
fn csr_readout_contracts_mirror_the_abelian_semantics() {
    let (t, v) = generic_truncated_tv(D4);

    let x: Su2TtVector<f64> = tt_csf(&[
        qntt::su2::core::Family::Up,
        qntt::su2::core::Family::Dn,
        qntt::su2::core::Family::Up,
        qntt::su2::core::Family::Dn,
    ]);
    let space = x.space;
    assert_eq!((space.d, space.n, space.two_s), (D4, NEL, 0));
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let (bl, br) = su2_tangent_bases(&x);
    let (_s, _tr, quintets) = spin_labelled_levels(D4, &t, &v);
    assert_eq!(
        quintets.len(),
        1,
        "premise: the QUINTET sector must be non-empty (1 state), or the arm tests an empty \
         Hilbert space rather than an empty CHANNEL"
    );
    assert!(
        !tangent_basis_shifted(&bl, &br, TRIPLET).is_empty(),
        "premise: the CSF must have a LIVE triplet channel, so leg 2 has somewhere to run"
    );
    let dim_q = tangent_dim_grids(&space, &bl.ranks, QUINTET);
    assert_eq!(
        dim_q, 0,
        "premise: the CSF's quintet channel must be structurally empty (no bond can feed a 2k = 4 \
         wire from a label the masked row grid keeps: tri(1,1,4) and tri(2,0,4) are both false, and \
         at 2S = 4 bonds 2, 3, 4 admit only (2,2), (3,3), (4,4), which this CSF does not carry), \
         which is WHAT this leg tests"
    );
    let out_q = space.with_two_s(QUINTET);
    let h_q = Su2SparseHamiltonian::new(out_q, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx_q = Su2SectorCtx::new(&ctx, &h_q, QUINTET);
    assert_eq!(sctx_q.dim(), 0);
    assert!(
        sctx_q.basis().is_empty(),
        "an empty channel has an empty basis"
    );
    let opts = SectorOpts {
        readout: SectorReadout::Tda,
        seed: 0xC5_61,
        ..Default::default()
    };
    let empty = sctx_q.sector_lowest_tda(3, &opts);
    println!(
        "csr empty channel: dim {}, converged {}, applies {}, theta {:?}",
        empty.dim,
        empty.converged,
        empty.applies,
        empty.theta()
    );
    assert_eq!(empty.dim, 0);
    assert_eq!(empty.two_s_out, QUINTET);
    assert_eq!(empty.theta().expect("Tda readout").len(), 0);
    assert!(
        empty.converged,
        "a structurally unreachable channel is CONVERGED — there is nothing left to converge"
    );
    assert_eq!(empty.applies, 0, "and it must cost nothing");
    assert!(empty.negatives().is_empty());

    let dim_t = tangent_dim_grids(&space, &bl.ranks, TRIPLET);
    assert!(
        dim_t > 0,
        "premise: the CSF's TRIPLET channel must be live, or leg 2 has no live channel to use"
    );
    let out_t = space.with_two_s(TRIPLET);
    let h_t = Su2SparseHamiltonian::new(out_t, &t, &v, None, f64::EPSILON);
    let sctx_t = Su2SectorCtx::new(&ctx, &h_t, TRIPLET);
    assert_eq!(sctx_t.dim(), dim_t);
    assert!(
        sctx_t.basis().len() <= dim_t,
        "dim is STRUCTURAL and bounds the numerical span from above"
    );
    let zero = sctx_t.sector_lowest_tda(0, &opts);
    println!(
        "csr k=0: dim {}, converged {}, applies {}, theta {:?}",
        zero.dim,
        zero.converged,
        zero.applies,
        zero.theta()
    );
    assert_eq!(
        zero.dim, dim_t,
        "k = 0 must still report the channel's true dim"
    );
    assert_eq!(zero.theta().expect("Tda readout").len(), 0);
    assert!(
        zero.converged,
        "nothing was requested, so nothing is outstanding"
    );
    assert_eq!(zero.applies, 0);
}

#[test]
fn csr_quintet_channel_and_the_sum_identity() {
    let (tv_t, tv_v) = generic_truncated_tv(D4);
    let (_singlets, _triplets, quintets) = spin_labelled_levels(D4, &tv_t, &tv_v);
    assert_eq!(
        quintets.len(),
        1,
        "the 2S = 4 sector holds exactly one state"
    );
    for &(cap, want) in &[(0usize, 36usize), (2, 35), (1, 28)] {
        let (space, t, v, ham, x) = truncated_reference(cap, 0xC5_70 + cap as u64);
        let (bl, br) = su2_tangent_bases(&x);
        let lambda = su2_rayleigh(&ham, &x).re_s();
        let dims: Vec<usize> = CHANNELS
            .iter()
            .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
            .collect();
        let total = ab_tangent_dim(&expand(&x, 0));
        println!(
            "csr sum identity [cap {cap}]: {dims:?} sum {} vs expand {total}",
            dims.iter().sum::<usize>()
        );
        assert_eq!(
            dims.iter().sum::<usize>(),
            total,
            "★ [cap {cap}] Sum_S' dim T^S' must equal tangent_dim(expand(x, 0)) exactly"
        );
        assert_eq!(
            total, want,
            "[cap {cap}] tangent_dim(expand) is measured {want}"
        );
        assert_eq!(dims[2], 1, "[cap {cap}] dim T^(S=2) is measured 1");

        let out_space = space.with_two_s(QUINTET);
        let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
        let basis = tangent_basis_shifted(&bl, &br, QUINTET);
        assert_eq!(
            basis.len(),
            1,
            "[cap {cap}] the quintet channel must span its whole 1-state sector"
        );
        let e = gram_eigs(
            &effham_gram_shifted(&h_out, &bl, &br, QUINTET, lambda, &basis),
            1,
        )[0] + lambda;
        println!(
            "csr quintet [cap {cap}]: eig(A) + lambda = {e}, exact {}",
            quintets[0]
        );
        assert!(
            (e - quintets[0]).abs() <= 1e-10 * quintets[0].abs().max(1.0),
            "[cap {cap}] the quintet channel is the WHOLE sector (1 = 1), so eig(A) + lambda must \
             be its single exact level {} at ANY base point, got {e}",
            quintets[0]
        );
    }
}

#[test]
fn csr_instability_reports_a_certified_negative() {
    let (space, t, v, ham, x) = truncated_reference(1, 0xC5_80);
    let (singlets, triplets, _q) = spin_labelled_levels(D4, &t, &v);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    println!(
        "csr instability: lambda = {lambda}, singlet ground {}, lowest triplet {}, \
         sigma_min = {:e}",
        singlets[0],
        triplets[0],
        su2_min_schmidt(&bl, &br)
    );
    assert!(
        triplets[0] < lambda - 1e-3,
        "PRECONDITION: the triplet sector must hold a state BELOW the reference's lambda, or there \
         is no instability to report (lowest triplet {} vs lambda {lambda})",
        triplets[0]
    );

    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let basis = sctx.basis();
    let dense = gram_eigs(&sctx.a_gram(&basis), basis.len());
    println!(
        "csr instability: dense eig(A) lowest {:?}",
        &dense[..2.min(dense.len())]
    );
    assert!(
        dense[0] < -1e-6,
        "PRECONDITION: the dense A block must be indefinite at this reference (lowest eig {}) — a \
         compression can sit above the sector's ground, so the dense check is what makes the \
         iterative claim non-vacuous",
        dense[0]
    );

    let opts = SectorOpts {
        tol: 1e-9,
        max_iter: 300,
        readout: SectorReadout::Tda,
        seed: 0xC5_81,
        ..Default::default()
    };
    let res = sctx.sector_lowest_tda(2, &opts);
    let theta = res.theta().expect("Tda readout");
    println!(
        "csr instability readout: theta {theta:?}, negatives {:?}",
        res.negatives()
            .iter()
            .map(|w: &NegativeWitness| (w.index, w.theta, w.fresh, w.certified()))
            .collect::<Vec<_>>()
    );
    assert!(
        theta.iter().all(|z| z.is_finite()),
        "an indefinite channel must report finite numbers, never NaN"
    );
    assert!(
        theta[0] < -1e-6,
        "the readout must SURFACE the negative eigenvalue ({}), not clamp it to zero — \
         criticality is not stability, and a state below lambda in another channel is physics",
        theta[0]
    );
    assert!(
        !res.negatives().is_empty(),
        "every reported negative value must carry a witness"
    );
    assert!(
        res.all_negatives_certified(),
        "the witnesses must be CERTIFIED (the fresh-apply quadratic form negative too), or the \
         claim is refused rather than published: {:?}",
        res.negatives()
    );
    for w in &res.negatives() {
        assert!(
            (w.theta - theta[w.index]).abs() <= 1e-8 * theta[w.index].abs().max(1.0),
            "the witness theta must be the reported eig(A) value in ENERGY units, not omega^2"
        );
    }
}

#[test]
fn csr_threaded_matches_serial() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_90);
    let (bl, br) = su2_tangent_bases(&x);
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);

    let run = || {
        let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let gram = sctx.a_gram(&basis);
        let norms: Vec<f64> = basis.iter().map(|b| b.norm()).collect();
        let fused: Vec<f64> = basis.iter().map(|b| sctx.a_apply_fused(b).norm()).collect();
        let nc = sctx.a_apply_fused(&basis[0]).num_coeffs();
        (basis.len(), gram, norms, fused, nc)
    };

    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("build 1-thread rayon pool");
    let (ns, gs, nrms, fs, ncs) = serial_pool.install(|| {
        assert_eq!(
            rayon::current_num_threads(),
            1,
            "serial arm did not run in the 1-thread pool"
        );
        run()
    });
    let par_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("build 8-thread rayon pool");
    let (np, gp, nrmp, fp, ncp) = par_pool.install(|| {
        assert!(
            rayon::current_num_threads() > 1,
            "parallel arm did not run multi-threaded ({} threads)",
            rayon::current_num_threads()
        );
        run()
    });

    assert!(ns > 1, "ANTI-VACUITY: the shifted basis has {ns} vectors");
    assert_eq!(
        ns, np,
        "threaded vs serial: the MGS cull accepted a different number of directions"
    );
    let scale = gs.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    assert!(scale > 1e-3, "ANTI-VACUITY: the A Gram is identically zero");
    for (j, (a, b)) in gs.iter().zip(gp.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * scale,
            "threaded vs serial A gram entry {j}: {a} vs {b}"
        );
    }
    for (j, (a, b)) in nrms.iter().zip(nrmp.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-13 * a.max(1.0),
            "threaded vs serial centerweighted norm of basis vector {j}: {a} vs {b} — a \
             pool-dependent norm is the R4 metric error (the parallel arm reading a different \
             space's two_s than the serial one)"
        );
    }
    assert_eq!(
        fs.len(),
        fp.len(),
        "the fused shifted apply was run over a different number of directions across pools"
    );
    let fscale = fs.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
    println!("csr threading: fused |A xi| scale = {fscale:e}, num_coeffs = {ncs}");
    assert!(
        fscale > 1e-3,
        "ANTI-VACUITY: the fused shifted apply annihilated every direction (scale {fscale:e}) — a \
         matches-serial pin over an annihilated direction forks nothing and proves nothing"
    );
    for (j, (a, b)) in fs.iter().zip(fp.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-12 * fscale,
            "threaded vs serial fused |A^(S') xi| for direction {j}: {a} vs {b}"
        );
    }
    assert_eq!(
        ncs, ncp,
        "threaded vs serial: the fused shifted apply returned a different coefficient count \
         ({ncs} vs {ncp}) — a pool-dependent zero-width component passes every norm bound"
    );
}

#[test]
fn csr_shifted_frames_carry_the_space_out_stamp() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC5_A1);
    let (bl, br) = su2_tangent_bases(&x);
    let _ = (&t, &v, &ham);
    for two_s_flux in [TRIPLET, QUINTET] {
        let out_space = space.with_two_s(two_s_flux);
        let basis = tangent_basis_shifted(&bl, &br, two_s_flux);
        assert!(
            !basis.is_empty(),
            "ANTI-VACUITY: 2k = {two_s_flux} is empty at this reference"
        );
        for (i, xi) in basis.iter().enumerate() {
            assert_eq!(
                xi.two_s_flux(),
                two_s_flux,
                "basis vector {i} must carry the channel's flux"
            );
            for k in 0..space.d {
                assert_eq!(
                    xi.component(k).space,
                    out_space,
                    "2k = {two_s_flux}, vector {i}, component {k}: every component must be \
                     stamped space_out, or dot_center weights with the WRONG total spin — a \
                     (2S+1) metric error that only appears above par::PAR_MIN_COEFFS"
                );
            }
            assert_eq!(xi.to_vector().space, out_space);
        }
    }
}

#[test]
fn csr_point_group_channel_dims_and_gram() {
    let pg = PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(1), Irrep::new(0)],
    );
    let (t, v) = pg_tv(pg);
    let space = Su2Space::with_point_group(D4, NEL, 0, pg, Irrep::IDENTITY);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_B0);
    let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, 0), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();

    let mut multiplane = false;
    for b in 0..=D4 {
        let planes: std::collections::BTreeSet<usize> = space
            .bond_sectors(b)
            .into_iter()
            .filter(|&s| bl.ranks[b][s] > 0)
            .map(|s| s.irrep.index() as usize)
            .collect();
        if planes.len() > 1 {
            multiplane = true;
        }
    }
    assert!(
        multiplane,
        "ANTI-VACUITY: no bond carried rank in more than one irrep plane, so the per-plane rank \
         sums are untested and this is the trivial-group arm again"
    );

    let dims: Vec<usize> = CHANNELS
        .iter()
        .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
        .collect();
    let total = ab_tangent_dim(&expand(&x, 0));
    println!("csr pg [A,B,B,A]: per-channel {dims:?}, expand {total}, lambda = {lambda}");
    assert!(
        dims[1] > 1,
        "ANTI-VACUITY: the spin flux does not move the irrep, so the triplet channel cannot be \
         irrep-empty — got dim T^(S=1) = {}, which means the fan lost the plane",
        dims[1]
    );
    assert_eq!(
        dims.iter().sum::<usize>(),
        total,
        "★ the sum identity must hold with a point group too (per-channel {dims:?})"
    );

    let out_space = space.with_two_s(TRIPLET);
    assert_eq!(
        (out_space.pg, out_space.target),
        (space.pg, space.target),
        "with_two_s must change ONLY two_s: the spin flux carries no irrep"
    );
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
    let got = effham_gram_shifted(&h_out, &bl, &br, TRIPLET, lambda, &basis);
    let want = iota_gram(&h_out, lambda, &basis);
    let n = basis.len();
    let scale = want.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    assert!(
        scale > 1e-3,
        "ANTI-VACUITY: the pg A Gram is identically zero"
    );
    for k in 0..n * n {
        assert!(
            (got[k] - want[k]).abs() <= 1e-12 * scale,
            "pg O1 entry {k}: {} vs {}",
            got[k],
            want[k]
        );
    }
}

#[test]
fn csr_d6_shifted_gram_and_sum_identity() {
    const D6: usize = 6;
    const N6: usize = 6;
    let (t, v) = generic_truncated_tv(D6);
    let space = Su2Space::new(D6, N6, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let caps = su2_max_ranks(&space);
    let grids: Vec<Su2RankGrid> = (0..=D6)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = caps[b][s].min(2);
            }
            r
        })
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_C0);
    let mut x: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();

    let channels: Vec<usize> = (0..=6).step_by(2).collect();
    let dims: Vec<usize> = channels
        .iter()
        .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
        .collect();
    let total = ab_tangent_dim(&expand(&x, 0));
    println!("csr d6: channels {channels:?}, dims {dims:?}, expand {total}");
    assert_eq!(
        dims.iter().sum::<usize>(),
        total,
        "★ d = 6: Sum_S' dim T^S' must equal tangent_dim(expand(x, 0)) exactly"
    );
    assert_eq!(
        dims,
        vec![95, 121, 31, 1],
        "the d = 6 channel dims are grid-determined and measured [95, 121, 31, 1] over \
         2S' in {{0, 2, 4, 6}}; got {dims:?}"
    );

    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
    let n = basis.len();
    let got = effham_gram_shifted(&h_out, &bl, &br, TRIPLET, lambda, &basis);
    let want = iota_gram(&h_out, lambda, &basis);
    let scale = want.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    assert!(
        scale > 1e-3,
        "ANTI-VACUITY: the d6 A Gram is identically zero"
    );
    for k in 0..n * n {
        assert!(
            (got[k] - want[k]).abs() <= 1e-11 * scale,
            "d6 O1 entry {k}: {} vs {}",
            got[k],
            want[k]
        );
    }
}

fn su2_residual_norm(
    ham: &Su2SparseHamiltonian<f64>,
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    x: &Su2TtVector<f64>,
) -> f64 {
    let lambda = su2_rayleigh(ham, x).re_s();
    su2_normal_residual(ham, bl, br, x, lambda).norm_lr(false)
}

fn gram_spectral_norm(g: &[f64], n: usize) -> f64 {
    gram_eigs(g, n).iter().fold(0.0f64, |m, z| m.max(z.abs()))
}

fn gram_asymmetry(g: &[f64], n: usize) -> f64 {
    let mut m = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            m = m.max((g[i * n + j] - g[j * n + i]).abs());
        }
    }
    m
}

fn dense_matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

fn dense_trace(a: &[f64], n: usize) -> f64 {
    (0..n).map(|i| a[i * n + i]).sum()
}

fn signed_omega2_or_fail(spec: &SymplecticSpectrum, what: &str) -> Vec<f64> {
    match spec {
        SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } => panic!(
            "{what}: the Williamson reduction came back Unstable ({reason:?}) — BOTH H^rr and \
             H^ii are indefinite there, so no signed omega^2 exists and the identity is not \
             stated. generator eigs = {generator_eigs:?}"
        ),
        _ => {
            let mut w = spec
                .omega_squared()
                .expect("omega_squared is Some on Stable and Resonant");
            w.sort_by(|p, q| p.partial_cmp(q).expect("finite omega^2"));
            w
        }
    }
}

fn generator_eigs(a: &[f64], b: &[f64], n: usize) -> Vec<qntt::C64> {
    qntt::core::weingarten::rpa_generator_eigenvalues(a, b, n)
}

fn assert_complex_multiset_eq(got: &[qntt::C64], want: &[qntt::C64], tol: f64, what: &str) -> f64 {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: generator eigenvalue COUNT {} != {}",
        got.len(),
        want.len()
    );
    let mut used = vec![false; want.len()];
    let mut worst = 0.0f64;
    for (j, p) in got.iter().enumerate() {
        let mut best = (f64::INFINITY, usize::MAX);
        for (i, q) in want.iter().enumerate() {
            if used[i] {
                continue;
            }
            let d = (p - q).norm();
            if d < best.0 {
                best = (d, i);
            }
        }
        assert!(
            best.0 <= tol,
            "{what}: generator eigenvalue {j} = {p} has no unmatched partner in the oracle within \
             {tol:e} (the closest unmatched value is {} at distance {:e})",
            want[best.1],
            best.0
        );
        used[best.1] = true;
        worst = worst.max(best.0);
    }
    worst
}

fn spectrum_label(spec: &SymplecticSpectrum) -> String {
    match spec {
        SymplecticSpectrum::Stable { omega } => format!("Stable(n = {})", omega.len()),
        SymplecticSpectrum::Resonant { signed } => format!(
            "Resonant(n = {}, n_neg = {})",
            signed.omega2.len(),
            signed.n_neg
        ),
        SymplecticSpectrum::Unstable { reason, .. } => format!("Unstable({reason:?})"),
    }
}

fn rayleigh_tilde_ab(ham: &SparseHamiltonian<f64>, y: &TtVector<f64>) -> f64 {
    ab_xt_h_y(y, ham, y).re_s() / (2.0 * ab_dot(y, y).re_s())
}

fn corrected_form_ab(
    ham: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    g_vec: &TtVector<f64>,
    xi: &TtTangent<f64>,
    h: f64,
) -> (f64, f64, f64) {
    let cp = xi.retract_hosvd(h);
    let cm = xi.retract_hosvd(-h);
    let m0 = rayleigh_tilde_ab(ham, x);
    let mp = rayleigh_tilde_ab(ham, &cp);
    let mm = rayleigh_tilde_ab(ham, &cm);
    let q_uncorr = (mp - 2.0 * m0 + mm) / (h * h);
    let gp = ab_dot(g_vec, &cp).re_s();
    let gm = ab_dot(g_vec, &cm).re_s();
    let g0 = ab_dot(g_vec, x).re_s();
    let corr = (gp + gm - 2.0 * g0) / (h * h);
    (q_uncorr - corr, q_uncorr, corr)
}

const FD_H: f64 = 5e-5;

fn corrected_form_richardson_ab(
    ham: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    g_vec: &TtVector<f64>,
    xi: &TtTangent<f64>,
    h: f64,
) -> f64 {
    let (q1, _, _) = corrected_form_ab(ham, x, g_vec, xi, h);
    let (q2, _, _) = corrected_form_ab(ham, x, g_vec, xi, h / 2.0);
    (4.0 * q2 - q1) / 3.0
}

fn three_band_grid(xi: &Su2TtTangent<f64>, space_out: Su2Space) -> Vec<Su2RankGrid> {
    let d = space_out.d;
    let rows = xi.rank();
    let cols = xi.rank_col();
    (0..=d)
        .map(|b| {
            let mut g = Su2RankGrid::zeros(&space_out);
            for s in space_out.bond_sectors(b) {
                g[s] = match b {
                    0 => rows[b][s],
                    _ if b == d => cols[b][s],
                    _ => rows[b][s] + cols[b][s],
                };
            }
            g
        })
        .collect()
}

fn shifted_unit_tangents(
    sctx: &Su2SectorCtx<'_, '_, f64>,
    seed: u64,
    n: usize,
) -> Vec<Su2TtTangent<f64>> {
    let space_out = sctx.space_out();
    let template = sctx
        .basis()
        .into_iter()
        .next()
        .expect("shifted_unit_tangents: the channel must be non-empty");
    let grid = three_band_grid(&template, space_out);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    for _ in 0..(8 * n) {
        if out.len() == n {
            break;
        }
        let y: Su2TtVector<f64> = su2_tt_randn(space_out, &grid, &mut rng);
        let mut xi = sctx.project(&y);
        let nrm = xi.norm();
        if nrm <= 1e-8 {
            continue;
        }
        xi.scale_mut(1.0 / nrm);
        out.push(xi);
    }
    assert_eq!(
        out.len(),
        n,
        "shifted_unit_tangents: only {} of {n} draws survived P at seed {seed:#x}",
        out.len()
    );
    out
}

fn capped_singlet_minimizer(
    cap: usize,
) -> (
    Su2Space,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
) {
    static MEMO: OnceLock<Mutex<HashMap<usize, Su2TtVector<f64>>>> = OnceLock::new();
    assert!(cap > 0, "cap 0 is full_rank_singlet_ground's job");
    let (t, v) = generic_truncated_tv(D4);
    let space = Su2Space::new(D4, NEL, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let memo = MEMO.get_or_init(|| Mutex::new(HashMap::new()));
    let x = {
        let mut m = memo.lock().expect("minimizer memo poisoned");
        m.entry(cap)
            .or_insert_with(|| {
                let grids = capped_grids(&space, cap);
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_B0 + cap as u64);
                let mut x0: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng);
                x0.scale_mut(1.0 / x0.norm_lr(false));
                let mut opts = RiemannianOpts::new(&space);
                opts.eps = 1e-14;
                opts.gradtol = 1e-14;
                opts.max_iter = 600;
                opts.preserve_grid = true;
                let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
                let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
                let mut rounds = 0usize;
                for _ in 0..12 {
                    if su2_grad_norm(&ham, &x) < 1e-11 {
                        break;
                    }
                    let (_l2, x2) = riemannian_cg(&ham, &x, &opts);
                    x = x2;
                    rounds += 1;
                }
                su2_normalize_mut(&mut x);
                assert!(x.check());
                let g = su2_grad_norm(&ham, &x);
                println!(
                    "csr cap-{cap} minimizer: {rounds} polish rounds (budget 12), ||grad|| = {g:e}, \
                     lambda = {}, sigma_min = {:e}",
                    su2_rayleigh(&ham, &x).re_s(),
                    {
                        let (bl, br) = su2_tangent_bases(&x);
                        su2_min_schmidt(&bl, &br)
                    }
                );
                assert!(
                    g < 1e-8,
                    "cap-{cap} minimizer PRECONDITION: ||grad|| = {g:e} after the polish budget — \
                     the Williamson arms need a critical reference (H^rr is PSD only at a minimum)"
                );
                x
            })
            .clone()
    };
    (space, t, v, ham, x)
}

struct O2Block {
    space: Su2Space,
    t: CoeffTensor<f64>,
    v: CoeffTensor<f64>,
    ham: Su2SparseHamiltonian<f64>,
    x: Su2TtVector<f64>,
    bl: Su2TtVector<f64>,
    br: Su2TtVector<f64>,
    lambda: f64,
    h_ab: SparseHamiltonian<f64>,
    x_ab: TtVector<f64>,
    dims: Vec<usize>,
    basis_s1: Vec<TtTangent<f64>>,
}

fn o2_spin_one_block(d: usize, nel: usize, cap: usize, seed: u64) -> O2Block {
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, nel, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, cap), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();

    let x_ab = expand(&x, 0);
    let space_ab = QnSpace::new(d, nel / 2, nel / 2);
    assert_eq!(x_ab.space, space_ab);
    assert!(x_ab.check());
    let h_ab = SparseHamiltonian::<f64>::new(space_ab, &t, &v, None, f64::EPSILON);
    let lam_ab = qntt::rayleigh_quotient(&h_ab, &x_ab).re_s();
    assert!(
        (lam_ab - lambda).abs() <= 1e-12 * lambda.abs().max(1.0),
        "O2 PREMISE: the two layers' lambda must agree ({lambda} vs {lam_ab})"
    );
    let (bl_ab, br_ab) = ab_tangent_bases(&x_ab);
    let basis_ab: Vec<TtTangent<f64>> = ab_tangent_basis(&bl_ab, &br_ab, false);
    let n = basis_ab.len();
    let channels: Vec<usize> = (0..=nel.min(2 * d - nel)).step_by(2).collect();
    let dims: Vec<usize> = channels
        .iter()
        .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
        .collect();
    assert_eq!(n, ab_tangent_dim(&bl_ab));
    assert_eq!(
        dims.iter().sum::<usize>(),
        n,
        "O2 PREMISE: the sum identity must hold before the spin split is a partition"
    );
    let iv: Vec<TtVector<f64>> = basis_ab.iter().map(|b| b.to_vector()).collect();
    for (i, a) in iv.iter().enumerate() {
        for (j, b) in iv.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (ab_dot(a, b) - want).abs() <= 1e-10,
                "O2 PREMISE: abelian basis orthonormal ({i},{j}) = {}",
                ab_dot(a, b)
            );
        }
    }
    let mut sg = vec![vec![0.0f64; n]; n];
    for j in 0..n {
        let w = s_squared(&iv[j]);
        for i in 0..n {
            sg[i][j] = ab_dot(&iv[i], &w);
        }
    }
    let (sev, svec) = dense_eigen(&sg);
    let mut want_spec: Vec<f64> = Vec::new();
    for (k, &dk) in dims.iter().enumerate() {
        let s = (channels[k] / 2) as f64;
        want_spec.extend(std::iter::repeat_n(s * (s + 1.0), dk));
    }
    want_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_multiset_eq(
        &sev,
        &want_spec,
        1e-10,
        &format!("O2 [d {d} cap {cap}] the S^2 Gram spectrum with multiplicities {dims:?}"),
    );
    let idx: Vec<usize> = (0..n).filter(|&c| (sev[c] - 2.0).abs() <= 1e-8).collect();
    assert_eq!(
        idx.len(),
        dims[1],
        "the eigenvalue-2 eigenspace has dimension dim T^(S=1)"
    );
    let basis_s1: Vec<TtTangent<f64>> = idx
        .iter()
        .map(|&c| {
            let coef = &svec[c];
            let mut acc = basis_ab[0].scale(coef[0]);
            for (i, b) in basis_ab.iter().enumerate().skip(1) {
                acc.axpy(coef[i], b);
            }
            acc
        })
        .collect();
    for (i, a) in basis_s1.iter().enumerate() {
        for (j, b) in basis_s1.iter().enumerate() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (a.dot(b) - want).abs() <= 1e-9,
                "the S=1 block basis must stay orthonormal ({i},{j}) = {}",
                a.dot(b)
            );
        }
    }
    O2Block {
        space,
        t,
        v,
        ham,
        x,
        bl,
        br,
        lambda,
        h_ab,
        x_ab,
        dims,
        basis_s1,
    }
}

fn odd_trace_words(a: &[f64], b: &[f64], n: usize) -> [f64; 4] {
    let ab = dense_matmul(a, b, n);
    let a2b = dense_matmul(a, &ab, n);
    let b2 = dense_matmul(b, b, n);
    let b3 = dense_matmul(&b2, b, n);
    [
        dense_trace(b, n),
        dense_trace(&ab, n),
        dense_trace(&a2b, n),
        dense_trace(&b3, n),
    ]
}

fn hrr_hii(a: &[f64], b: &[f64]) -> (Vec<f64>, Vec<f64>) {
    (
        a.iter().zip(b).map(|(p, q)| p + q).collect(),
        a.iter().zip(b).map(|(p, q)| p - q).collect(),
    )
}

#[test]
fn csr_expanded_abelian_spin_block_certifies_b() {
    for cap in [2usize, 1] {
        let blk = o2_spin_one_block(D4, NEL, cap, 0xC5_40 + cap as u64);
        assert_eq!(
            blk.dims,
            if cap == 2 {
                vec![19usize, 15, 1]
            } else {
                vec![14usize, 13, 1]
            },
            "[cap {cap}] the per-channel dims are grid-determined"
        );
        let rperp = su2_residual_norm(&blk.ham, &blk.bl, &blk.br, &blk.x);
        assert!(
            rperp > 1e-6,
            "[cap {cap}] premise: B is linear in r_perp, so a zero residual would certify A alone \
             (||r_perp|| = {rperp:e})"
        );
        let ctx_ab = ab_weingarten_operator(&blk.h_ab, &blk.x_ab);
        let n = blk.basis_s1.len();
        assert!(n > 1, "[cap {cap}] ANTI-VACUITY: the S=1 block is trivial");
        let (a1, b1) = ab_hessian_grams(&ctx_ab, &blk.basis_s1);

        let out_space = blk.space.with_two_s(TRIPLET);
        let h_out = Su2SparseHamiltonian::new(out_space, &blk.t, &blk.v, None, f64::EPSILON);
        let ctx = su2_weingarten_operator(&blk.ham, &blk.x);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let basis = sctx.basis();
        assert_eq!(
            basis.len(),
            n,
            "[cap {cap}] the two layers describe the same space"
        );
        let (a, b) = sctx.grams(&basis);

        let (na, nb) = (gram_spectral_norm(&a, n), gram_spectral_norm(&b, n));
        let scale = na.max(1.0);
        println!(
            "csr O2-B [cap {cap}]: dim {n}, lambda = {}, ||r_perp|| = {rperp:e}, sigma_min = {:e}, \
             ||A|| = {na:e}, ||B|| = {nb:e}, ★ ||B^(S=1)||/||A^(S=1)|| = {:.4e} (abelian block: \
             ||B_ab|| = {:e})",
            blk.lambda,
            su2_min_schmidt(&blk.bl, &blk.br),
            nb / na.max(f64::MIN_POSITIVE),
            gram_spectral_norm(&b1, n)
        );
        assert!(
            nb > 1e-3 * na,
            "[cap {cap}] ANTI-VACUITY: ||B|| = {nb:e} vs ||A|| = {na:e} — the SU(2) B is (nearly) \
             zero at a truncated non-critical reference on a CURVED channel: the silent-collapse \
             signature (pZ/qZ seeded from a dressed frame, Cb-spec §1.6)"
        );
        assert_multiset_eq(
            &gram_eigs(&b, n),
            &gram_eigs(&b1, n),
            1e-10 * scale,
            &format!("[cap {cap}] eig(B^{{S=1}}) vs eig(S=1 block of the abelian B Gram)"),
        );
        let (hrr, hii) = hrr_hii(&a, &b);
        let (hrr1, hii1) = hrr_hii(&a1, &b1);
        assert_multiset_eq(
            &gram_eigs(&hrr, n),
            &gram_eigs(&hrr1, n),
            1e-10 * scale,
            &format!(
                "[cap {cap}] eig(A + B) vs the abelian block — pins the relative eigenvector alignment of A and B"
            ),
        );
        assert_multiset_eq(
            &gram_eigs(&hii, n),
            &gram_eigs(&hii1, n),
            1e-10 * scale,
            &format!("[cap {cap}] eig(A − B) vs the abelian block"),
        );
        assert_multiset_eq(
            &gram_eigs(&a, n),
            &gram_eigs(&a1, n),
            1e-10 * scale,
            &format!("[cap {cap}] eig(A^{{S=1}}) vs the abelian block"),
        );
        let spec = sctx.symplectic_eigenvalues(&basis);
        let spec_ab = qntt::solvers::hess::symplectic_from_grams(&a1, &b1, n, 1e-10);
        println!(
            "csr O2-B [cap {cap}]: dense arm su2 {} vs abelian block {}",
            spectrum_label(&spec),
            spectrum_label(&spec_ab)
        );
        match (spec.omega_squared(), spec_ab.omega_squared()) {
            (Some(mut w), Some(mut w1)) => {
                w.sort_by(|p, q| p.partial_cmp(q).unwrap());
                w1.sort_by(|p, q| p.partial_cmp(q).unwrap());
                let ws = w1.iter().fold(1.0f64, |m, z| m.max(z.abs()));
                assert_multiset_eq(
                    &w,
                    &w1,
                    1e-8 * ws,
                    &format!("[cap {cap}] signed omega^2 vs the abelian block"),
                );
            }
            (None, None) => {
                let (
                    SymplecticSpectrum::Unstable {
                        generator_eigs: g, ..
                    },
                    SymplecticSpectrum::Unstable {
                        generator_eigs: g1, ..
                    },
                ) = (&spec, &spec_ab)
                else {
                    panic!(
                        "[cap {cap}] omega_squared() is None only on the Unstable arm: su2 {} vs \
                         abelian {}",
                        spectrum_label(&spec),
                        spectrum_label(&spec_ab)
                    )
                };
                let gs = g1.iter().fold(0.0f64, |m, z| m.max(z.norm())).max(1.0);
                let worst = assert_complex_multiset_eq(
                    g,
                    g1,
                    1e-8 * gs,
                    &format!("[cap {cap}] eig(𝓜^{{S=1}}) vs the abelian block's generator"),
                );
                println!(
                    "csr O2-B [cap {cap}]: both Unstable — {} generator eigenvalues matched, worst \
                     {worst:e} (scale {gs:e})",
                    g.len()
                );
            }
            (p, q) => panic!(
                "[cap {cap}] the two layers routed the SAME pair differently: su2 {} vs abelian {} \
                 (omega^2 {p:?} vs {q:?})",
                spectrum_label(&spec),
                spectrum_label(&spec_ab)
            ),
        }
    }
}

#[test]
fn csr_fd_oracle_certifies_a_plus_b_on_the_expanded_curve() {
    for cap in [2usize, 1] {
        let blk = o2_spin_one_block(D4, NEL, cap, 0xC5_F0 + cap as u64);
        let ctx_ab = ab_weingarten_operator(&blk.h_ab, &blk.x_ab);
        let (bl_ab, br_ab) = ctx_ab.bases();
        let g_vec = ctx_ab.gradient().to_vector();
        let rperp = su2_residual_norm(&blk.ham, &blk.bl, &blk.br, &blk.x);
        println!(
            "csr fd [cap {cap}]: sigma_min = {:e}, sigma_min(ab) = {:e}, ||r_perp|| = {rperp:e}, \
             ||grad_ab|| = {:e}",
            su2_min_schmidt(&blk.bl, &blk.br),
            qntt::core::tangent::min_schmidt(bl_ab, br_ab),
            ctx_ab.gradient().norm()
        );
        assert!(
            rperp > 1e-6,
            "[cap {cap}] premise: r_perp = 0 would certify A alone"
        );

        let out_space = blk.space.with_two_s(TRIPLET);
        let h_out = Su2SparseHamiltonian::new(out_space, &blk.t, &blk.v, None, f64::EPSILON);
        let ctx = su2_weingarten_operator(&blk.ham, &blk.x);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        assert!(sctx.dim() >= 2);
        let dirs = shifted_unit_tangents(&sctx, 0xC5_F1 + cap as u64, 4);

        let embed = |xi: &Su2TtTangent<f64>, what: &str| -> TtTangent<f64> {
            let y = expand(&xi.to_vector(), 0);
            let xi_ab = TtTangent::project(bl_ab, br_ab, &y);
            let (n_xi, n_ab) = (xi.norm(), xi_ab.norm());
            assert!(
                (n_ab - n_xi).abs() <= 1e-12 * n_xi.max(1.0),
                "{what}: expand∘iota is not isometric across the embedding ({n_ab} vs {n_xi})"
            );
            let back = &xi_ab.to_vector() - &y;
            let (ny, defect) = (y.norm_lr(false), back.norm_lr(false));
            assert!(
                defect <= 1e-10 * ny.max(1.0),
                "{what}: expand(iota(xi), 0) is not ALREADY tangent at x_ab — the abelian \
                 projection moved it by {defect:e} (||y|| = {ny:e}); the shifted gauge condition \
                 is the S=1 block of the abelian one, so this must be the identity"
            );
            xi_ab
        };

        for (j, xi) in dirs.iter().take(3).enumerate() {
            let a_xi = sctx.a_apply(xi);
            let b_xi = sctx.b_apply(xi);
            let h_xi = sctx.hess_apply(xi);
            let q_ab = xi.dot(&h_xi);
            let s = q_ab.abs().max(1.0);
            let xi_ab = embed(xi, &format!("[cap {cap}] dir {j}"));
            let (_qc, _qu, corr) = corrected_form_ab(&blk.h_ab, &blk.x_ab, &g_vec, &xi_ab, FD_H);
            let q_rich = corrected_form_richardson_ab(&blk.h_ab, &blk.x_ab, &g_vec, &xi_ab, FD_H);
            let q_alg = xi_ab.dot(&ctx_ab.a_apply_fused(&xi_ab).add(&ctx_ab.b_apply(&xi_ab)));
            println!(
                "csr fd [cap {cap}] dir {j}: q_su2 = {q_ab:.12} q_rich = {q_rich:.12} (|d| = {:e}) \
                 q_alg = {q_alg:.12} (|d| = {:e}) corr = {corr:e} ||A xi|| = {:e} ||B xi|| = {:e} \
                 ratio = {:.3e}",
                (q_rich - q_ab).abs(),
                (q_alg - q_ab).abs(),
                a_xi.norm(),
                b_xi.norm(),
                b_xi.norm() / a_xi.norm().max(f64::MIN_POSITIVE)
            );
            assert!(
                (q_alg - q_ab).abs() <= 1e-10 * s,
                "[cap {cap}] dir {j}: <xi,(A+B)xi> = {q_ab} disagrees with the abelian algebraic \
                 value {q_alg} at 1e-10 — this is the SU(2) assembly, not the FD"
            );
            assert!(
                (q_rich - q_ab).abs() <= 1e-6 * s,
                "[cap {cap}] dir {j}: the corrected Richardson FD says {q_rich} but \
                 <xi,(A+B)xi> = {q_ab} (scale {s:e}); the algebraic cross-check passed, so this is \
                 the FD or the embedding"
            );
            assert!(
                b_xi.norm() > 1e-3 * a_xi.norm(),
                "[cap {cap}] dir {j}: ANTI-VACUITY — ||B xi|| = {:e} under 1e-3 of ||A xi|| = {:e} \
                 on the curved triplet channel; this arm would be an FD check on A alone",
                b_xi.norm(),
                a_xi.norm()
            );
        }
        let (xi, eta) = (&dirs[0], &dirs[3]);
        let xi_ab = embed(xi, &format!("[cap {cap}] polarized xi"));
        let eta_ab = embed(eta, &format!("[cap {cap}] polarized eta"));
        let sum_ab = xi_ab.add(&eta_ab);
        let q_x = corrected_form_richardson_ab(&blk.h_ab, &blk.x_ab, &g_vec, &xi_ab, FD_H);
        let q_e = corrected_form_richardson_ab(&blk.h_ab, &blk.x_ab, &g_vec, &eta_ab, FD_H);
        let q_s = corrected_form_richardson_ab(&blk.h_ab, &blk.x_ab, &g_vec, &sum_ab, FD_H);
        let fd_off = 0.5 * (q_s - q_x - q_e);
        let op_off = 0.5 * (eta.dot(&sctx.hess_apply(xi)) + xi.dot(&sctx.hess_apply(eta)));
        println!(
            "csr fd [cap {cap}] polarized: FD {fd_off:.12} vs operator {op_off:.12} (|d| = {:e}, \
             <xi,eta> = {:e})",
            (fd_off - op_off).abs(),
            xi.dot(eta)
        );
        assert!(
            (fd_off - op_off).abs() <= 1e-5,
            "[cap {cap}] the polarized second difference {fd_off} disagrees with \
             1/2(<eta,(A+B)xi> + <xi,(A+B)eta>) = {op_off}"
        );
    }
}

#[test]
fn csr_b_is_self_adjoint_and_hpp_hmm_are_a_pm_b() {
    for cap in [2usize, 1] {
        let (space, t, v, ham, x) = truncated_reference(cap, 0xC6_00 + cap as u64);
        let (bl, br) = su2_tangent_bases(&x);
        let rperp = su2_residual_norm(&ham, &bl, &br, &x);
        assert!(
            rperp > 1e-6,
            "[cap {cap}] premise: symmetry at r_perp = 0 is that of the zero operator"
        );
        let out_space = space.with_two_s(TRIPLET);
        let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
        let ctx = su2_weingarten_operator(&ham, &x);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let dirs = shifted_unit_tangents(&sctx, 0xC6_01 + cap as u64, 6);

        for j in 0..3 {
            let (xi, eta) = (&dirs[2 * j], &dirs[2 * j + 1]);
            let b_xi = sctx.b_apply(xi);
            let b_eta = sctx.b_apply(eta);
            assert_eq!(b_xi.two_s_flux(), TRIPLET);
            assert_eq!(
                b_xi.rank_col(),
                xi.rank_col(),
                "B's image lives at the channel's column grids"
            );
            let lhs = xi.dot(&b_eta);
            let rhs = b_xi.dot(eta);
            let s = lhs
                .abs()
                .max(rhs.abs())
                .max(b_xi.norm() * eta.norm())
                .max(1e-12);
            println!(
                "csr b-props [cap {cap}] pair {j}: <xi,B eta> = {lhs:.15e} <B xi,eta> = {rhs:.15e} \
                 (|d| = {:e}, ||B xi|| = {:e})",
                (lhs - rhs).abs(),
                b_xi.norm()
            );
            assert!(
                b_xi.norm() > 1e-8,
                "[cap {cap}] pair {j}: ANTI-VACUITY, B annihilated the direction"
            );
            assert!(
                (lhs - rhs).abs() <= 1e-12 * s,
                "[cap {cap}] pair {j}: B^(S=1) is not self-adjoint ({lhs} vs {rhs}) — an S1 or S3 \
                 chain carries an UNLIFTED R^-1 (both act on a column axis, lifted), or the final \
                 gauge is on base_l instead of the masked restamp (F8)"
            );
        }

        let basis = sctx.basis();
        let n = basis.len();
        assert!(n >= 2);
        let (a, b) = sctx.grams(&basis);
        let (na, nb) = (gram_spectral_norm(&a, n), gram_spectral_norm(&b, n));
        let (hrr, hii) = hrr_hii(&a, &b);
        println!(
            "csr b-props [cap {cap}] grams: n = {n}, ||A|| = {na:e}, ||B|| = {nb:e}, ||B||/||A|| = {:.3e}, \
             asym(A) = {:e}, asym(B) = {:e}",
            nb / na.max(f64::MIN_POSITIVE),
            gram_asymmetry(&a, n),
            gram_asymmetry(&b, n)
        );
        assert!(
            nb > 1e-3 * na,
            "[cap {cap}] ANTI-VACUITY: ||B|| = {nb:e} vs ||A|| = {na:e}"
        );
        assert!(
            gram_asymmetry(&b, n) <= 1e-12 * nb,
            "[cap {cap}] the dense B Gram is not symmetric"
        );
        assert!(
            gram_asymmetry(&a, n) <= 1e-12 * na,
            "[cap {cap}] the dense A Gram is not symmetric"
        );
        assert!(
            gram_asymmetry(&hrr, n) <= 1e-12 * (na + nb),
            "[cap {cap}] A + B not symmetric"
        );
        assert!(
            gram_asymmetry(&hii, n) <= 1e-12 * (na + nb),
            "[cap {cap}] A − B not symmetric"
        );
        for (i, g) in [&a, &b].into_iter().enumerate() {
            assert!(
                g.iter().all(|z| z.is_finite()),
                "[cap {cap}] Gram {i} carries a non-finite entry"
            );
        }

        for (j, xi) in dirs.iter().take(3).enumerate() {
            let a_xi = sctx.a_apply(xi);
            let b_xi = sctx.b_apply(xi);
            let scale = a_xi.norm().max(b_xi.norm()).max(1.0);
            let hpp = sctx.hpp_apply(xi);
            let hmm = sctx.hmm_apply(xi);
            let hess = sctx.hess_apply(xi);
            assert!(
                hpp.sub(&a_xi.add(&b_xi)).norm() <= 1e-12 * scale,
                "[cap {cap}] dir {j}: hpp_apply must be A + B (= H^rr)"
            );
            assert!(
                hmm.sub(&a_xi.sub(&b_xi)).norm() <= 1e-12 * scale,
                "[cap {cap}] dir {j}: hmm_apply must be A − B (= H^ii)"
            );
            assert!(
                hess.sub(&hpp).norm() <= 1e-12 * scale,
                "[cap {cap}] dir {j}: hess_apply and hpp_apply are the same operator"
            );
            assert!(
                hpp.sub(&hmm).norm() > 1e-6 * scale,
                "[cap {cap}] dir {j}: ANTI-VACUITY — H^rr and H^ii coincide, so B is zero here"
            );
        }
    }
}

#[test]
fn csr_b_vanishes_at_the_exact_singlet_ground() {
    let (space, t, v, ham, x) = full_rank_singlet_ground();
    let (bl, br) = su2_tangent_bases(&x);
    let rperp = su2_residual_norm(&ham, &bl, &br, &x);
    let g = su2_grad_norm(&ham, &x);
    println!(
        "csr b-zero: ||r_perp|| = {rperp:e} (norm_lr), ||grad|| = {g:e}, sigma_min = {:e}",
        su2_min_schmidt(&bl, &br)
    );
    assert!(
        rperp <= 1e-12,
        "PREMISE: the exact singlet ground must sit at the residual floor (||r_perp|| = {rperp:e} via \
         norm_lr); without it 'B vanishes' is not the claim being tested"
    );
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    assert_eq!(
        sctx.dim(),
        15,
        "the full-rank triplet channel is the whole sector"
    );
    for (j, xi) in shifted_unit_tangents(&sctx, 0xC6_10, 3).iter().enumerate() {
        let a_xi = sctx.a_apply(xi);
        let b_xi = sctx.b_apply(xi);
        println!(
            "csr b-zero dir {j}: ||A xi|| = {:e}, ||B xi|| = {:e}",
            a_xi.norm(),
            b_xi.norm()
        );
        assert!(
            a_xi.norm() > 1e-3,
            "dir {j}: ANTI-VACUITY — a vanishing A satisfies the bound for any B"
        );
        assert!(
            b_xi.norm() <= 1e-9 * a_xi.norm(),
            "dir {j}: ||B xi|| = {:e} at r_perp = 0 — B is LINEAR in the normal residual and must \
             vanish here (||A xi|| = {:e})",
            b_xi.norm(),
            a_xi.norm()
        );
    }
    let basis = sctx.basis();
    let n = basis.len();
    let (a, b) = sctx.grams(&basis);
    let (na, nb) = (gram_spectral_norm(&a, n), gram_spectral_norm(&b, n));
    println!("csr b-zero grams: n = {n}, ||A|| = {na:e}, ||B|| = {nb:e}");
    assert!(na > 1e-3);
    assert!(
        nb <= 1e-9 * na,
        "the whole B Gram is {nb:e} at r_perp = 0 (||A|| = {na:e})"
    );
    let spec = sctx.symplectic_eigenvalues(&basis);
    let w = signed_omega2_or_fail(&spec, "b-zero: the dense readout at the exact ground");
    let want: Vec<f64> = {
        let mut e: Vec<f64> = gram_eigs(&a, n).iter().map(|z| z * z).collect();
        e.sort_by(|p, q| p.partial_cmp(q).unwrap());
        e
    };
    assert_multiset_eq(
        &w,
        &want,
        1e-8 * na * na,
        "b-zero: omega^2 = eig(A)^2 where B = 0",
    );
}

#[test]
fn csr_b_is_linear_in_z_through_the_public_assembly() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_20);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let xi = shifted_unit_tangents(&sctx, 0xC6_21, 1).remove(0);

    let z1 = su2_normal_residual(&ham, &bl, &br, &x, lambda);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC6_22);
    let w: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, 0), &mut rng);
    let z2 = &w - &Su2TtTangent::project(&bl, &br, &w).to_vector();
    for (name, z) in [("r_perp", &z1), ("w - iota(P w)", &z2)] {
        let zn = z.norm_lr(false);
        let tang = Su2TtTangent::project(&bl, &br, z).norm();
        assert!(
            zn > 1e-6 && tang <= 1e-6 * zn,
            "{name}: not a non-degenerate normal vector ({zn:e}, {tang:e})"
        );
    }
    let b = |z: &Su2TtVector<f64>| {
        su2_weingarten_assemble(&bl, &br, z, &xi, FloorPolicy::default_hard(), Terms::all())
    };
    let (b1, b2, bs) = (b(&z1), b(&z2), b(&(&z1 + &z2)));
    assert_eq!(
        b1.two_s_flux(),
        TRIPLET,
        "the public assembly carries the flux through"
    );
    let scale = b1.norm().max(b2.norm());
    println!(
        "csr linearity: ||B(r_perp) xi|| = {:e}, ||B(z2) xi|| = {:e}, |additivity| = {:e}",
        b1.norm(),
        b2.norm(),
        bs.sub(&b1.add(&b2)).norm()
    );
    assert!(
        b1.norm() > 1e-8,
        "ANTI-VACUITY: B(r_perp) xi = 0 on the curved triplet channel"
    );
    assert!(
        b1.sub(&b2).norm() > 1e-3 * scale,
        "ANTI-VACUITY: the two normal z's give the same image"
    );
    assert!(
        bs.sub(&b1.add(&b2)).norm() <= 1e-12 * scale,
        "B(z1 + z2) != B(z1) + B(z2)"
    );
    for alpha in [-2.5f64, 0.5] {
        let ba = b(&(&z1 * alpha));
        assert!(
            ba.sub(&b1.scale(alpha)).norm() <= 1e-12 * (alpha.abs() * b1.norm()).max(1.0),
            "B(alpha z) != alpha B(z) at alpha = {alpha}"
        );
    }
    let via_ctx = sctx.b_apply(&xi);
    let d = via_ctx.sub(&b1).norm();
    println!("csr linearity: |Su2SectorCtx::b_apply − weingarten_assemble(r_perp)| = {d:e}");
    assert!(
        d <= 1e-10 * b1.norm(),
        "the split-Z context route and the explicit normal residual must agree on the triplet \
         channel ({d:e} vs ||B xi|| = {:e}) — Cb-spec §1.6's cached-envs reuse",
        b1.norm()
    );
}

#[test]
fn csr_williamson_union_matches_the_expanded_abelian_spectrum() {
    let mut signed_legs: Vec<(String, f64)> = Vec::new();
    for &(label, cap, want_dims, want_ab) in &[
        ("full-rank ground", 0usize, [20usize, 15, 1], 36usize),
        ("cap-2 minimizer", 2, [19, 15, 1], 35),
        ("cap-1 minimizer", 1, [14, 13, 1], 28),
    ] {
        let (space, t, v, ham, x) = if cap == 0 {
            full_rank_singlet_ground()
        } else {
            capped_singlet_minimizer(cap)
        };
        let (bl, br) = su2_tangent_bases(&x);
        let ctx = su2_weingarten_operator(&ham, &x);
        let g = ctx.gradient().norm();
        let rperp = su2_residual_norm(&ham, &bl, &br, &x);
        println!(
            "csr williamson [{label}]: ||grad|| = {g:e}, ||r_perp|| = {rperp:e}, sigma_min = {:e}, \
             lambda = {}",
            su2_min_schmidt(&bl, &br),
            su2_rayleigh(&ham, &x).re_s()
        );
        assert!(
            g < 1e-8,
            "[{label}] PRECONDITION: the reference must be critical (||grad|| = {g:e})"
        );
        let caps = capped_grids(&space, cap);
        for (b, cap_b) in caps.iter().enumerate() {
            assert_eq!(
                bl.ranks[b], *cap_b,
                "[{label}] PRECONDITION: the minimizer must sit at the declared cap-{cap} grid at bond \
                 {b}, or the literal dimensions below are not this fixture's"
            );
        }

        let x_ab = expand(&x, 0);
        let h_ab = SparseHamiltonian::<f64>::new(x_ab.space, &t, &v, None, f64::EPSILON);
        let ctx_ab = ab_weingarten_operator(&h_ab, &x_ab);
        let (bl_ab, br_ab) = ctx_ab.bases();

        let dims: Vec<usize> = CHANNELS
            .iter()
            .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
            .collect();
        let dim_ab = ab_tangent_dim(bl_ab);
        println!("csr williamson [{label}]: dims {dims:?}, tangent_dim(expand) {dim_ab}");
        assert_eq!(
            dims,
            want_dims.to_vec(),
            "[{label}] the per-channel dims are measured (Cb-spec §0 F9)"
        );
        assert_eq!(
            dim_ab, want_ab,
            "[{label}] tangent_dim(expand(x, 0)) is measured"
        );
        assert_eq!(
            (dims[0] - 1) + dims[1] + dims[2],
            dim_ab - 1,
            "[{label}] dimension identity: (dim T^0 - 1) + Σ_{{S'≥1}} dim T^S' = tangent_dim(expand) - 1"
        );
        assert_eq!(
            (dims[0] - 1) + dims[1] + dims[2],
            want_ab - 1,
            "[{label}] = 35 / 34 / 27"
        );

        let basis_ab = ab_tangent_basis(bl_ab, br_ab, true);
        let basis_0 = su2_tangent_basis(&bl, &br, true);
        let shifted: Vec<(usize, Vec<Su2TtTangent<f64>>)> = [TRIPLET, QUINTET]
            .iter()
            .map(|&k| (k, tangent_basis_shifted(&bl, &br, k)))
            .collect();
        let n_shifted: usize = shifted.iter().map(|(_, b)| b.len()).sum();
        assert_eq!(
            basis_ab.len(),
            basis_0.len() + n_shifted,
            "[{label}] the abelian radial-deflated basis has {} directions but the blocks sum to {} \
             ({} in S'=0 + {n_shifted} shifted)",
            basis_ab.len(),
            basis_0.len() + n_shifted,
            basis_0.len()
        );

        let spec_ab = ab_symplectic_eigenvalues(&ctx_ab, &basis_ab);
        let spec_0 = su2_symplectic_eigenvalues(&ctx, &basis_0);
        let (a_ab, b_ab) = ab_hessian_grams(&ctx_ab, &basis_ab);
        let n_ab = basis_ab.len();
        let (hrr_ab, hii_ab) = hrr_hii(&a_ab, &b_ab);
        let (a_0, b_0) = su2_hessian_grams(&ctx, &basis_0);
        let n_0 = basis_0.len();
        let (hrr_0, hii_0) = hrr_hii(&a_0, &b_0);
        let mut union_hrr = gram_eigs(&hrr_0, n_0);
        let mut union_hii = gram_eigs(&hii_0, n_0);
        let mut union_gen = generator_eigs(&a_0, &b_0, n_0);
        let mut max_b_ratio = 0.0f64;
        println!(
            "csr williamson [{label}]: abelian {} (n = {n_ab}), S'=0 {} (n = {n_0})",
            spectrum_label(&spec_ab),
            spectrum_label(&spec_0),
        );
        let signed_or_none = |sp: &SymplecticSpectrum| -> Option<Vec<f64>> {
            sp.omega_squared().map(|mut w| {
                w.sort_by(|p, q| p.partial_cmp(q).expect("finite omega^2"));
                w
            })
        };
        let mut union_w = signed_or_none(&spec_0);
        let mut all_blocks_stable = matches!(spec_0, SymplecticSpectrum::Stable { .. });
        for (k, basis_k) in &shifted {
            let out_space = space.with_two_s(*k);
            let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
            let sctx = Su2SectorCtx::new(&ctx, &h_out, *k);
            let (a, b) = sctx.grams(basis_k);
            let n = basis_k.len();
            let spec = sctx.symplectic_eigenvalues(basis_k);
            println!(
                "csr williamson [{label}] 2S'={k}: {} (n = {n}), ||A|| = {:e}, ||B|| = {:e}, ||B||/||A|| = {:.3e}",
                spectrum_label(&spec),
                gram_spectral_norm(&a, n),
                gram_spectral_norm(&b, n),
                gram_spectral_norm(&b, n) / gram_spectral_norm(&a, n).max(f64::MIN_POSITIVE)
            );
            if *k == QUINTET {
                assert!(
                    gram_spectral_norm(&b, n) <= 1e-12 * gram_spectral_norm(&a, n).max(1.0),
                    "[{label}] the quintet channel is single-slot: B^(S=2) ≡ 0 structurally"
                );
            }
            let (hrr, hii) = hrr_hii(&a, &b);
            union_hrr.extend(gram_eigs(&hrr, n));
            union_hii.extend(gram_eigs(&hii, n));
            union_gen.extend(generator_eigs(&a, &b, n));
            max_b_ratio = max_b_ratio
                .max(gram_spectral_norm(&b, n) / gram_spectral_norm(&a, n).max(f64::MIN_POSITIVE));
            all_blocks_stable &= matches!(spec, SymplecticSpectrum::Stable { .. });
            union_w = match (union_w, signed_or_none(&spec)) {
                (Some(mut acc), Some(w)) => {
                    acc.extend(w);
                    Some(acc)
                }
                _ => None,
            };
        }
        for (what, mut got, want) in [
            ("H^rr", union_hrr, gram_eigs(&hrr_ab, n_ab)),
            ("H^ii", union_hii, gram_eigs(&hii_ab, n_ab)),
        ] {
            got.sort_by(|p, q| p.partial_cmp(q).expect("finite eigenvalue"));
            let sc = want.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            assert!(
                sc > 1e-6,
                "[{label}] ANTI-VACUITY: the abelian {what} spectrum is identically zero"
            );
            println!(
                "csr williamson [{label}]: |union {what}| = {}, |abelian| = {}, worst = {:e}",
                got.len(),
                want.len(),
                got.iter()
                    .zip(want.iter())
                    .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()))
            );
            assert_multiset_eq(
                &got,
                &want,
                1e-8 * sc,
                &format!(
                    "[{label}] union_S' eig({what}^S') vs the expanded abelian eig({what}) — the                      block-diagonality the omega^2 identity rests on, stateable at EVERY base point"
                ),
            );
        }
        let ab_stable = matches!(spec_ab, SymplecticSpectrum::Stable { .. });
        assert_eq!(
            ab_stable,
            all_blocks_stable,
            "[{label}] H^rr and H^ii are PD on the abelian side exactly when they are PD on every              block (block diagonality), so the arms must agree: abelian {} vs blocks all-Stable              {all_blocks_stable}",
            spectrum_label(&spec_ab)
        );
        match (union_w, signed_or_none(&spec_ab)) {
            (Some(mut union), Some(w_ab)) => {
                union.sort_by(|p, q| p.partial_cmp(q).expect("finite omega^2"));
                let scale = w_ab.iter().fold(0.0f64, |m, z| m.max(z.abs()));
                println!(
                    "csr williamson [{label}]: |union omega^2| = {}, |abelian| = {}, max|omega^2| =                      {scale:e}, worst = {:e}",
                    union.len(),
                    w_ab.len(),
                    union
                        .iter()
                        .zip(w_ab.iter())
                        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()))
                );
                assert!(
                    scale > 1e-6,
                    "[{label}] ANTI-VACUITY: the abelian omega^2 spectrum is identically zero"
                );
                assert_multiset_eq(
                    &union,
                    &w_ab,
                    1e-8 * scale.max(1.0),
                    &format!(
                        "[{label}] union_S' omega^2_S' vs the expanded abelian signed spectrum"
                    ),
                );
                signed_legs.push((label.to_string(), max_b_ratio));
            }
            (u, w) => {
                println!(
                    "csr williamson [{label}]: omega^2 route NOT STATED (abelian {}, blocks                      all-Stable {all_blocks_stable}) — the H^rr/H^ii spectral union and the arm                      cross-check carry this leg",
                    spectrum_label(&spec_ab)
                );
                assert_eq!(
                    u.is_some(),
                    w.is_some(),
                    "[{label}] one side has a Williamson reduced form and the other does not, which                      block diagonality forbids"
                );
                let gen_ab = generator_eigs(&a_ab, &b_ab, n_ab);
                let gs = gen_ab.iter().fold(0.0f64, |m, z| m.max(z.norm()));
                assert!(
                    gs > 1e-6,
                    "[{label}] ANTI-VACUITY: the abelian generator spectrum is identically zero"
                );
                let worst = assert_complex_multiset_eq(
                    &union_gen,
                    &gen_ab,
                    1e-8 * gs,
                    &format!(
                        "[{label}] union_S' eig(M^S') vs the expanded abelian eig(M) — the \
                         always-stateable form of the omega^2 identity"
                    ),
                );
                println!(
                    "csr williamson [{label}]: |union eig(M)| = {}, |abelian| = {}, max|eig(M)| = \
                     {gs:e}, worst matched = {worst:e}",
                    union_gen.len(),
                    gen_ab.len()
                );
            }
        }
    }
    println!("csr williamson: signed-omega^2 legs {signed_legs:?}");
    assert!(
        signed_legs.iter().any(|(_, r)| *r > 1e-3),
        "COVERAGE: the signed omega^2 union identity was never stated on a leg with a non-negligible \
         B (legs that reached it, with their max ||B||/||A||: {signed_legs:?}). The identity would \
         then be certified only where TDA = RPA — add or repair a curved CRITICAL leg rather than \
         letting the arm pass as an A-only statement"
    );
}

#[test]
fn csr_isospectrality_watch_and_four_odd_trace_words() {
    fn watch_numbers(a: &[f64], b: &[f64], n: usize) -> (f64, f64, f64, [f64; 4]) {
        let (hrr, hii) = hrr_hii(a, b);
        let (ep, em) = (gram_eigs(&hrr, n), gram_eigs(&hii, n));
        let sep = ep
            .iter()
            .zip(em.iter())
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        (
            gram_spectral_norm(a, n),
            gram_spectral_norm(b, n),
            sep,
            odd_trace_words(a, b, n),
        )
    }
    let report = |label: &str, a: &[f64], b: &[f64], n: usize| -> (f64, [f64; 4], [f64; 4]) {
        let (na, nb, sep, [tb, tab, ta2b, tb3]) = watch_numbers(a, b, n);
        let (hrr, hii) = hrr_hii(a, b);
        let nf = n as f64;
        let scales = [nf * nb, nf * na * nb, nf * na * na * nb, nf * nb * nb * nb];
        println!(
            "csr isospectrality [{label}]: n = {n}, ||A|| = {na:e}, ||B|| = {nb:e} (ratio {:.3e})\n   \
             spec(A+B) = {:?}\n   spec(A-B) = {:?}\n   max separation = {sep:e}\n   tr B = {tb:e} \
             (scale {:e})  tr AB = {tab:e} (scale {:e})  tr A^2B = {ta2b:e} (scale {:e})  \
             tr B^3 = {tb3:e} (scale {:e})",
            nb / na.max(f64::MIN_POSITIVE),
            gram_eigs(&hrr, n),
            gram_eigs(&hii, n),
            scales[0],
            scales[1],
            scales[2],
            scales[3]
        );
        for z in [tb, tab, ta2b, tb3, sep] {
            assert!(z.is_finite(), "[{label}] a non-finite diagnostic");
        }
        assert!(
            nb > 1e-3 * na,
            "[{label}] ANTI-VACUITY: B is (nearly) zero, the watch is trivial"
        );
        (sep, [tb, tab, ta2b, tb3], scales)
    };

    for cap in [2usize, 1] {
        let (space, t, v, ham, x) = capped_singlet_minimizer(cap);
        let (bl, br) = su2_tangent_bases(&x);
        let ctx = su2_weingarten_operator(&ham, &x);
        println!(
            "csr isospectrality [cap-{cap} minimizer, CRITICAL]: ||grad|| = {:e}, ||r_perp|| = {:e}",
            ctx.gradient().norm(),
            su2_residual_norm(&ham, &bl, &br, &x)
        );
        let out_space = space.with_two_s(TRIPLET);
        let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let basis = sctx.basis();
        let n = basis.len();
        let (a, b) = sctx.grams(&basis);
        let label = format!("cap-{cap} minimizer, CRITICAL");
        let (sep, words, scales) = report(&label, &a, &b, n);
        let na = gram_spectral_norm(&a, n);

        for (i, name) in [(0usize, "tr B"), (3, "tr B^3")] {
            assert!(
                words[i].abs() <= 1e-8 * scales[i].max(1e-300),
                "[{label}] {name} = {:e} against scale {:e} — MEASURED to vanish at both caps AND \
                 at the non-critical reference (the odd words in B ALONE do; the ones carrying an A \
                 do not, see below). There is NO theorem behind it, so a failure here is either a \
                 change in one of the two assemblies or a change of basin — REPORT EITHER rather \
                 than adjusting the bound",
                words[i],
                scales[i]
            );
        }
        println!(
            "csr isospectrality [{label}]: tr AB = {:e} at relative {:.3e} — grid-dependent, printed \
             (cap 2: 4.4e-16, cap 1: 2.5e-5; it vanishes at the cap-2 random reference too)",
            words[1],
            words[1].abs() / scales[1].max(1e-300)
        );
        assert!(
            words[2].abs() > 1e-5 * scales[2],
            "[{label}] tr A^2B = {:e} against scale {:e} came out VANISHING — Cb-spec §1.9's \
             prediction would then hold after all, which would be a new result and must go to the \
             user rather than being absorbed silently (measured 1.1e-3 relative at cap 2)",
            words[2],
            scales[2]
        );
        assert!(
            sep > 1e-3 * na,
            "[{label}] max |spec(A+B) - spec(A-B)| = {sep:e} against ||A|| = {na:e} came out \
             VANISHING — see the tr A^2B message: a confirmation of the refuted prediction is itself \
             a result for the user"
        );

        let x_ab = expand(&x, 0);
        let h_ab = SparseHamiltonian::<f64>::new(x_ab.space, &t, &v, None, f64::EPSILON);
        let ctx_ab = ab_weingarten_operator(&h_ab, &x_ab);
        let (bl_ab, br_ab) = ctx_ab.bases();
        let basis_ab: Vec<TtTangent<f64>> = ab_tangent_basis(bl_ab, br_ab, false);
        let m = basis_ab.len();
        let iv: Vec<TtVector<f64>> = basis_ab.iter().map(|u| u.to_vector()).collect();
        let mut sg = vec![vec![0.0f64; m]; m];
        for j in 0..m {
            let w = s_squared(&iv[j]);
            for i in 0..m {
                sg[i][j] = ab_dot(&iv[i], &w);
            }
        }
        let (sev, svec) = dense_eigen(&sg);
        let idx: Vec<usize> = (0..m).filter(|&c| (sev[c] - 2.0).abs() <= 1e-8).collect();
        assert_eq!(
            idx.len(),
            n,
            "[{label}] the abelian S=1 eigenspace must be the SU(2) triplet channel"
        );
        let basis_s1: Vec<TtTangent<f64>> = idx
            .iter()
            .map(|&c| {
                let coef = &svec[c];
                let mut acc = basis_ab[0].scale(coef[0]);
                for (i, u) in basis_ab.iter().enumerate().skip(1) {
                    acc.axpy(coef[i], u);
                }
                acc
            })
            .collect();
        let (a1, b1) = ab_hessian_grams(&ctx_ab, &basis_s1);
        let (sep1, words1, _) = report(&format!("cap-{cap} ABELIAN S=1 control"), &a1, &b1, n);
        let tol = 1e-9 * na.max(1.0) * na.max(1.0) * gram_spectral_norm(&b1, n).max(1.0);
        assert!(
            (sep1 - sep).abs() <= 1e-9 * na,
            "[{label}] CONTROL: the abelian S=1 block separates by {sep1:e} but the SU(2) channel by \
             {sep:e} — the two layers must agree at the same point"
        );
        for (i, name) in [
            (0usize, "tr B"),
            (1, "tr AB"),
            (2, "tr A^2B"),
            (3, "tr B^3"),
        ] {
            assert!(
                (words1[i] - words[i]).abs() <= tol,
                "[{label}] CONTROL: {name} is {:e} on the abelian S=1 block and {:e} on the SU(2) \
                 channel (tol {tol:e}) — the refutation must be the POINT's, not a layer's",
                words1[i],
                words[i]
            );
        }
    }

    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_30);
    let (bl, br) = su2_tangent_bases(&x);
    let ctx = su2_weingarten_operator(&ham, &x);
    println!(
        "csr isospectrality [cap-2 random, NON-critical]: ||grad|| = {:e}, ||r_perp|| = {:e}",
        ctx.gradient().norm(),
        su2_residual_norm(&ham, &bl, &br, &x)
    );
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let basis = sctx.basis();
    let n = basis.len();
    let (a, b) = sctx.grams(&basis);
    let label = "cap-2 random, NON-critical";
    let (_sep, words, scales) = report(label, &a, &b, n);
    for (i, name) in [(0usize, "tr B"), (3, "tr B^3")] {
        assert!(
            words[i].abs() <= 1e-8 * scales[i].max(1e-300),
            "[{label}] {name} = {:e} against scale {:e} — measured to vanish at the non-critical \
             reference as well (2.0e-15 / 1.5e-17); no theorem, so report an assembly change or a \
             fixture change rather than loosening this",
            words[i],
            scales[i]
        );
    }
}

#[test]
fn csr_sector_lowest_honours_max_subspace() {
    let (space, t, v, ham, x) = capped_singlet_minimizer(2);
    let ctx = su2_weingarten_operator(&ham, &x);
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let k = 2usize;
    csr_max_subspace_leg(
        "cap-2 triplet",
        &sctx,
        RpaKernel::signed(RpaMetric::Ii),
        k,
        2 * (k + (k + 2)),
    );
    let (label, table, nel, g0, cap, seed, ..) = PG6_FIXTURES[0];
    assert_eq!(label, "smallest");
    let (space6, t6, v6, ham6, x6) = pg6_reference(table, nel, g0, cap, seed);
    let ctx6 = su2_weingarten_operator(&ham6, &x6);
    let h6_out =
        Su2SparseHamiltonian::new(space6.with_two_s(TRIPLET), &t6, &v6, None, f64::EPSILON);
    let sctx6 = Su2SectorCtx::new(&ctx6, &h6_out, TRIPLET);
    csr_max_subspace_leg("smallest triplet", &sctx6, RpaKernel::Paired, 1, 16);
}

fn csr_max_subspace_leg(
    label: &str,
    sctx: &Su2SectorCtx<'_, '_, f64>,
    kernel: RpaKernel,
    k: usize,
    tight: usize,
) {
    let dim = sctx.dim();
    assert!(
        dim > tight,
        "premise [{label}]: dim {dim} leaves room below it for the tight bound {tight}"
    );
    let tol = 1e-10;
    let run = |max_subspace: usize| {
        sctx.sector_lowest(
            k,
            &SectorOpts {
                tol,
                readout: SectorReadout::Rpa,
                kernel,
                max_subspace,
                ..SectorOpts::default()
            },
        )
    };
    let (dflt, wide, cut) = (run(0), run(dim + 10), run(tight));
    let w = |r: &qntt::su2::hess::Su2SectorResponse| -> Vec<f64> {
        r.omega2().expect("an Rpa readout").to_vec()
    };
    println!(
        "csr max_subspace [{label}, {kernel:?}, k {k}]: dim {dim}; default {:?} ({} iters, {} \
         applies, converged {}) | tight {tight}: {:?} ({} iters, {} applies, converged {})",
        w(&dflt),
        dflt.iterations,
        dflt.applies,
        dflt.converged,
        w(&cut),
        cut.iterations,
        cut.applies,
        cut.converged
    );
    let bits = crate::sector_util::flat_bits;
    assert!(
        bits(&w(&wide)) == bits(&w(&dflt))
            && (wide.iterations, wide.applies) == (dflt.iterations, dflt.applies)
            && bits(&residuals_of(wide.rpa())) == bits(&residuals_of(dflt.rpa())),
        "[{label}] a bound above dim clamps to dim, as the auto bound does here: the same solve \
         bit for bit"
    );
    assert!(
        bits(&w(&cut)) != bits(&w(&dflt))
            || (cut.iterations, cut.applies) != (dflt.iterations, dflt.applies),
        "[{label}] non-vacuity: the tight-bound run is bitwise the default one — the bound never \
         reached the kernel"
    );
    assert!(
        dflt.converged && cut.converged,
        "[{label}] both runs converge"
    );
    let (wd, wc) = (w(&dflt), w(&cut));
    assert_eq!(wd.len(), wc.len(), "[{label}] the same number of roots");
    let scale = wd.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
    for (j, (a, b)) in wd.iter().zip(&wc).enumerate() {
        assert!(
            (a - b).abs() <= 10.0 * tol * scale,
            "[{label}] root {j}: default {a} vs tight bound {b}"
        );
    }
}

#[test]
fn csr_sector_lowest_matches_the_dense_symplectic_driver() {
    let (space, t, v, ham, x) = capped_singlet_minimizer(2);
    let (bl, br) = su2_tangent_bases(&x);
    let ctx = su2_weingarten_operator(&ham, &x);
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let basis = sctx.basis();
    let n = basis.len();
    assert!(n >= 3, "dim {n} is too small to iterate on");
    let dense = sctx.symplectic_eigenvalues(&basis);
    println!(
        "csr iterative: n = {n}, dim = {}, sigma_min = {:e}, ||r_perp|| = {:e}, dense {}",
        sctx.dim(),
        su2_min_schmidt(&bl, &br),
        su2_residual_norm(&ham, &bl, &br, &x),
        spectrum_label(&dense)
    );
    let w_dense = signed_omega2_or_fail(
        &dense,
        "the dense readout on the cap-2 minimizer's triplet channel",
    );
    let scale = w_dense.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    assert!(
        scale > 1e-6,
        "ANTI-VACUITY: the dense omega^2 spectrum is zero"
    );

    let k = 2usize;
    let run = |metric: RpaMetric| {
        let opts = SectorOpts {
            readout: SectorReadout::Rpa,
            kernel: RpaKernel::signed(metric),
            ..SectorOpts::default()
        };
        sctx.sector_lowest(k, &opts)
    };
    let auto = run(RpaMetric::Auto);
    let resolved = match &auto.spectrum {
        SectorSpectrum::Rpa(r) => r.side(),
        SectorSpectrum::Tda { .. } => panic!("sector_lowest returned a Tda spectrum"),
    };
    println!(
        "csr iterative Auto: resolved {resolved:?}, converged {}, iters {}, applies {}, omega2 {:?}, \
         negatives {} (all certified {}), sigma_min {:e}",
        auto.converged,
        auto.iterations,
        auto.applies,
        auto.omega2(),
        auto.negatives().len(),
        auto.all_negatives_certified(),
        auto.sigma_min
    );
    assert_eq!(auto.dim, sctx.dim());
    assert_eq!(auto.two_s_out, TRIPLET);
    assert_eq!(auto.space, out_space);
    assert!(
        (auto.sigma_min - su2_min_schmidt(&bl, &br)).abs() <= 1e-14,
        "sigma_min is the CENTERWEIGHTED min_schmidt"
    );
    assert!(
        auto.theta().is_none(),
        "an Rpa readout publishes no theta()"
    );
    assert_ne!(
        resolved,
        RpaMetric::Auto,
        "the RESOLVED metric must be Ii or Rr, never Auto"
    );
    assert!(
        auto.converged,
        "the Auto run did not converge in {} iterations / {} applies (omega2 = {:?}); every \
         assertion below is gated on convergence",
        auto.iterations,
        auto.applies,
        auto.omega2()
    );
    let w_it = auto.omega2().expect("an Rpa readout publishes omega2()");
    println!(
        "csr iterative: dense label {} (recorded, Q12); signature {:?}, report {:?}, index {:?}",
        spectrum_label(&dense),
        krein_signs(auto.rpa()),
        metric_report(auto.rpa()).map(|(n, g, _)| (n, g)),
        auto.index_lower_bound()
    );
    match &dense {
        SymplecticSpectrum::Stable { .. } => {
            assert!(
                !w_it.is_empty() && w_it.len() <= k,
                "expected 1..={k} roots, got {}",
                w_it.len()
            );
            assert!(
                !auto.metric_indefinite() && !auto.metric_indefinite(),
                "a Stable channel's signed solve meets no indefinite metric"
            );
            assert_eq!(
                krein_signs(auto.rpa()),
                vec![1i8; w_it.len()],
                "every s = +1"
            );
            assert_eq!(
                auto.index_lower_bound(),
                Some(0),
                "no inertia on a Stable channel"
            );
        }
        SymplecticSpectrum::Resonant { signed } => assert_eq!(
            w_it.len(),
            signed.n_neg + k,
            "a Resonant leg lists every negative plus the k lowest (Q12)"
        ),
        SymplecticSpectrum::Unstable { .. } => {
            unreachable!("signed_omega2_or_fail refused the Unstable arm above")
        }
    }
    assert_eq!(
        residuals_of(auto.rpa()).len(),
        w_it.len(),
        "one residual per omega^2 entry"
    );
    let tol = 10.0 * SectorOpts::default().tol * scale.max(1.0);
    for (j, w) in w_it.iter().enumerate() {
        assert!(
            (w - w_dense[j]).abs() <= tol,
            "iterative omega^2[{j}] = {w} vs dense {} (tol {tol:e})",
            w_dense[j]
        );
    }
    assert!(
        auto.omega().is_some() == (w_it.iter().all(|&z| z > 0.0) && auto.negatives().is_empty()),
        "omega() is Some exactly when every omega^2 is positive and no witness is outstanding"
    );
    let ii = run(RpaMetric::Ii);
    let rr = run(RpaMetric::Rr);
    println!(
        "csr iterative forced: Ii converged {} ({} applies) omega2 {:?} | Rr converged {} ({} applies) omega2 {:?}",
        ii.converged,
        ii.applies,
        ii.omega2(),
        rr.converged,
        rr.applies,
        rr.omega2()
    );
    assert!(
        ii.converged || rr.converged,
        "NEITHER forced metric converged"
    );
    if matches!(dense, SymplecticSpectrum::Stable { .. }) {
        assert!(
            crate::sector_util::flat_bits(&w_it)
                == crate::sector_util::flat_bits(&ii.omega2().expect("Rpa"))
                && (auto.iterations, auto.applies) == (ii.iterations, ii.applies)
                && crate::sector_util::flat_bits(&residuals_of(auto.rpa()))
                    == crate::sector_util::flat_bits(&residuals_of(ii.rpa())),
            "Auto must be the Ii call bit for bit on a Stable channel"
        );
    }
    if ii.converged && rr.converged {
        assert_multiset_eq(
            &ii.omega2().expect("Rpa"),
            &rr.omega2().expect("Rpa"),
            tol,
            "forced Ii vs forced Rr — which block carries the metric is a free choice where both are PD",
        );
    }
}

#[test]
fn csr_rpa_readout_routes_instability_without_publishing_omega() {
    let (space, t, v, ham, x) = truncated_reference(1, 0xC5_80);
    let (_singlets, triplets, _q) = spin_labelled_levels(D4, &t, &v);
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    assert!(
        triplets[0] < lambda - 1e-3,
        "PRECONDITION: the triplet sector holds a state below lambda"
    );
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let basis = sctx.basis();
    let n = basis.len();
    let (a, b) = sctx.grams(&basis);
    let dense = sctx.symplectic_eigenvalues(&basis);
    let lowest_a = gram_eigs(&a, n)[0];
    println!(
        "csr rpa-instability: dim {} n {n} sigma_min {:e} ||r_perp|| {:e} ||A|| {:e} ||B|| {:e} \
         dense {} lowest eig(A) {lowest_a}",
        sctx.dim(),
        su2_min_schmidt(&bl, &br),
        su2_residual_norm(&ham, &bl, &br, &x),
        gram_spectral_norm(&a, n),
        gram_spectral_norm(&b, n),
        spectrum_label(&dense)
    );
    assert!(
        lowest_a < -1e-6,
        "PRECONDITION: A^(S=1) must be indefinite here (lowest eig {lowest_a})"
    );
    match &dense {
        SymplecticSpectrum::Stable { omega } => panic!(
            "the dense readout came back Stable ({omega:?}) at a reference that is NOT the lowest \
             state of its target channel — an indefinite A cannot leave both H^rr and H^ii PD"
        ),
        SymplecticSpectrum::Resonant { signed } => {
            assert!(
                signed.n_neg > 0,
                "Resonant with n_neg = 0 is Stable by another name"
            );
            assert!(signed.omega2.iter().all(|z| z.is_finite()));
        }
        SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } => {
            println!(
                "csr rpa-instability: Unstable({reason:?}), 2n = {}",
                generator_eigs.len()
            );
            assert_eq!(
                generator_eigs.len(),
                2 * n,
                "the Unstable arm reports all 2n generator eigenvalues"
            );
            assert!(dense.omega_squared().is_none() && dense.unstable_pairs().is_none());
        }
    }

    let inertia = crate::sector_util::pencil_inertia(&a, &b, n);
    println!("csr rpa-instability: dense inertia {inertia:?}");
    assert!(
        inertia.min_abs_eig >= 1e-8,
        "PRECONDITION: the pencil is nonsingular (min |eig(A +- B)| = {:e})",
        inertia.min_abs_eig
    );
    assert_eq!(
        inertia.lhs(),
        inertia.rhs(),
        "the dense counting convention (pencil_inertia): {inertia:?}"
    );

    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        kernel: RpaKernel::signed(RpaMetric::Auto),
        ..SectorOpts::default()
    };
    let out = sctx.sector_lowest(sctx.dim(), &opts);
    println!(
        "csr rpa-instability iterative: dim {} omega2 {:?} signature {:?} converged {} iters {} \
         applies {} negatives {} all certified {} flagged {} report {:?} index {:?}",
        out.dim,
        out.omega2(),
        krein_signs(out.rpa()),
        out.converged,
        out.iterations,
        out.applies,
        out.negatives().len(),
        out.all_negatives_certified(),
        out.metric_indefinite(),
        metric_report(out.rpa()),
        out.index_lower_bound()
    );
    assert_eq!(out.two_s_out, TRIPLET);
    assert_eq!(out.space, out_space);
    assert!(out.dim > 0);
    assert!(
        out.theta().is_none(),
        "an Rpa readout must not publish theta()"
    );
    assert!(
        out.omega().is_none(),
        "no real omega may be published at a reference whose target channel holds a lower state: {:?}",
        out.omega()
    );
    let w = out
        .omega2()
        .expect("an Rpa readout reports in Rpa units: an omega2, never None");
    assert!(w.iter().all(|z| z.is_finite()), "never NaN: {w:?}");
    assert!(
        !w.is_empty(),
        "the signed readout answers on a live channel: a non-empty omega2"
    );
    assert_eq!(
        krein_signs(out.rpa()).len(),
        w.len(),
        "one Krein signature per entry"
    );
    assert_eq!(
        residuals_of(out.rpa()).len(),
        w.len(),
        "one residual per entry"
    );
    assert!(
        out.applies > 0,
        "the solve spent applies and must report them"
    );
    let n_neg = w.iter().filter(|&&z| z < 0.0).count();
    assert_eq!(
        out.negatives().len(),
        n_neg,
        "one NegativeWitness per negative value"
    );
    for wit in &out.negatives() {
        assert!(wit.index < w.len());
        assert!(
            (wit.theta - w[wit.index]).abs() <= 1e-8 * w[wit.index].abs().max(1.0),
            "the witness theta must be the published omega^2 (omega^2 UNITS on the Rpa arm, the \
             opposite of the Tda arm)"
        );
        assert!(
            wit.fresh.is_finite(),
            "the fresh-apply quadratic form must be finite"
        );
    }
    assert!(out.all_negatives_certified(), "{:?}", out.negatives());
    let report = metric_report(out.rpa());
    match &dense {
        SymplecticSpectrum::Stable { .. } => unreachable!("the Stable arm panicked above"),
        SymplecticSpectrum::Unstable { .. } => {
            println!(
                "csr rpa-instability: branch Unstable — both blocks indefinite: flagged, general path"
            );
            assert!(
                out.metric_indefinite(),
                "W = H^ii is indefinite: an exhaustive Ii solve must be FLAGGED"
            );
            assert!(
                report.is_some_and(|(_, general_path, _)| general_path),
                "R = H^rr is indefinite too: the projected pencil leaves the definite path"
            );
        }
        SymplecticSpectrum::Resonant { signed } => match signed.metric {
            qntt::core::weingarten::RpaMetricBlock::Ii => {
                println!("csr rpa-instability: branch Resonant(PD = H^ii) — unflagged, negatives");
                assert!(
                    !out.metric_indefinite(),
                    "W = H^ii is PD: the Ii lane meets no indefinite metric"
                );
                assert!(n_neg >= 1, "Resonant(PD = H^ii) carries a negative omega^2");
            }
            qntt::core::weingarten::RpaMetricBlock::Rr => {
                println!(
                    "csr rpa-instability: branch Resonant(PD = H^rr) — flagged, definite path"
                );
                assert!(
                    out.metric_indefinite(),
                    "W = H^ii is indefinite: the Ii solve is FLAGGED"
                );
                assert!(
                    report.is_some_and(|(_, general_path, _)| !general_path),
                    "R = H^rr is PD: the projected pencil stays on the definite path"
                );
                assert!(n_neg >= 1, "Resonant(PD = H^rr) carries a negative omega^2");
            }
        },
    }
    assert_eq!(
        out.index_lower_bound(),
        Some(inertia.rhs()),
        "exhaustion: the projected pencil's inertia is n-(A+B) + n-(A-B) ({inertia:?})"
    );
    println!(
        "csr rpa-instability: negative omega^2 on a live B != 0 tangent: {}",
        if n_neg > 0 {
            "YES (the gap closed on this route)"
        } else {
            "no"
        }
    );
}

#[test]
fn csr_rpa_readout_contracts_mirror_the_abelian_route_table() {
    let (t, v) = generic_truncated_tv(D4);
    let x: Su2TtVector<f64> = tt_csf(&[
        qntt::su2::core::Family::Up,
        qntt::su2::core::Family::Dn,
        qntt::su2::core::Family::Up,
        qntt::su2::core::Family::Dn,
    ]);
    let space = x.space;
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let (bl, _br) = su2_tangent_bases(&x);
    let (_s, _tr, quintets) = spin_labelled_levels(D4, &t, &v);
    assert_eq!(
        quintets.len(),
        1,
        "premise: the quintet SECTOR is non-empty"
    );
    assert_eq!(
        tangent_dim_grids(&space, &bl.ranks, QUINTET),
        0,
        "premise: the quintet CHANNEL is empty"
    );
    assert!(
        tangent_dim_grids(&space, &bl.ranks, TRIPLET) > 0,
        "premise: the triplet channel is live"
    );
    let ctx = su2_weingarten_operator(&ham, &x);

    for metric in [RpaMetric::Auto, RpaMetric::Ii, RpaMetric::Rr] {
        let opts = SectorOpts {
            readout: SectorReadout::Rpa,
            kernel: RpaKernel::signed(metric),
            seed: 0xC6_61,
            ..SectorOpts::default()
        };
        let h_q = Su2SparseHamiltonian::new(space.with_two_s(QUINTET), &t, &v, None, f64::EPSILON);
        let sctx_q = Su2SectorCtx::new(&ctx, &h_q, QUINTET);
        let empty = sctx_q.sector_lowest(3, &opts);
        println!(
            "csr routes [{metric:?}] empty: dim {} converged {} applies {} iters {} spectrum {:?}",
            empty.dim, empty.converged, empty.applies, empty.iterations, empty.spectrum
        );
        assert_eq!(empty.dim, 0);
        assert_eq!(empty.two_s_out, QUINTET);
        assert!(
            empty.converged,
            "a structurally unreachable channel is CONVERGED — nothing left to converge"
        );
        assert_eq!(empty.applies, 0);
        assert_eq!(empty.iterations, 0);
        assert!(empty.negatives().is_empty());
        match &empty.spectrum {
            SectorSpectrum::Rpa(r) => {
                assert!(
                    r.modes.is_empty(),
                    "an empty channel has an empty, REAL spectrum"
                );
                assert_eq!(r.side(), metric, "route (a) reports opts.metric unresolved");
            }
            SectorSpectrum::Tda { .. } => panic!("an Rpa readout must carry an Rpa spectrum"),
        }
        assert_eq!(empty.omega2(), Some(Vec::new()));
        assert!(empty.theta().is_none());
        assert!(
            empty.omega().is_none(),
            "an EMPTY omega2 publishes no omega"
        );
        assert!(
            krein_signs(empty.rpa()).is_empty() && residuals_of(empty.rpa()).is_empty(),
            "route (a): no per-entry field"
        );
        assert!(!empty.metric_indefinite());
        assert_eq!(empty.index_lower_bound(), Some(0));

        let h_t = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
        let sctx_t = Su2SectorCtx::new(&ctx, &h_t, TRIPLET);
        let zero = sctx_t.sector_lowest(0, &opts);
        assert!(zero.dim > 0, "k = 0 still reports the channel's true dim");
        assert!(zero.converged, "nothing requested, nothing outstanding");
        assert_eq!(zero.applies, 0);
        assert_eq!(zero.omega2(), Some(Vec::new()));
        assert!(zero.omega().is_none());
        assert!(
            krein_signs(zero.rpa()).is_empty()
                && residuals_of(zero.rpa()).is_empty()
                && !zero.metric_indefinite(),
            "route (a'): nothing solved"
        );
        assert_eq!(zero.index_lower_bound(), Some(0));
        if let SectorSpectrum::Rpa(r) = &zero.spectrum {
            assert_eq!(r.side(), metric);
        } else {
            panic!("Rpa spectrum expected");
        }
    }
}

#[test]
#[should_panic(expected = "sector_lowest:")]
fn csr_tda_readout_is_refused_in_sector_lowest() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_70);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let opts = SectorOpts {
        readout: SectorReadout::Tda,
        ..SectorOpts::default()
    };
    let _ = sctx.sector_lowest(1, &opts);
}

#[test]
fn csr_sector_lowest_tda_refusal_takes_the_abelian_wording() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_71);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        ..SectorOpts::default()
    };
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        sctx.sector_lowest_tda(2, &opts)
    }));
    let payload = res.expect_err("sector_lowest_tda must refuse an Rpa readout");
    let msg = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a string panic payload");
    println!("csr wording: {msg}");
    assert!(
        msg.contains("sector_lowest_tda"),
        "the refusal must name its entry point: {msg}"
    );
    assert!(
        !msg.contains("does not exist yet") && !msg.contains("C-b"),
        "the C-a 'Phase C-b … does not exist yet' clause must be gone once sector_lowest exists: {msg}"
    );
}

#[test]
#[should_panic(expected = "spin flux")]
fn csr_grams_refuses_a_wrong_flux_basis_vector() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_72);
    let (bl, br) = su2_tangent_bases(&x);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let mut basis = sctx.basis();
    assert!(!basis.is_empty());
    let quintet = tangent_basis_shifted(&bl, &br, QUINTET);
    assert_eq!(
        quintet.len(),
        1,
        "premise: the quintet channel is 1-dimensional"
    );
    assert!(quintet[0].norm() > 1e-8);
    basis.push(quintet[0].clone());
    let _ = sctx.grams(&basis);
}

#[test]
#[should_panic(expected = "spin flux")]
fn csr_b_apply_refuses_the_wrong_flux() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_73);
    let (bl, br) = su2_tangent_bases(&x);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let quintet = tangent_basis_shifted(&bl, &br, QUINTET);
    assert_eq!(quintet.len(), 1);
    let _ = sctx.b_apply(&quintet[0]);
}

#[test]
#[should_panic(expected = "spin flux")]
fn csr_a_apply_fused_refuses_the_wrong_flux() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_74);
    let (bl, br) = su2_tangent_bases(&x);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let quintet = tangent_basis_shifted(&bl, &br, QUINTET);
    assert_eq!(quintet.len(), 1);
    assert!(quintet[0].norm() > 1e-8);
    let _ = sctx.a_apply_fused(&quintet[0]);
}

#[test]
#[should_panic(expected = "spin flux")]
fn csr_a_gram_refuses_a_wrong_flux_basis_vector() {
    let (space, t, v, ham, x) = truncated_reference(2, 0xC6_75);
    let (bl, br) = su2_tangent_bases(&x);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let mut basis = sctx.basis();
    assert!(!basis.is_empty());
    let quintet = tangent_basis_shifted(&bl, &br, QUINTET);
    assert_eq!(quintet.len(), 1);
    assert!(quintet[0].norm() > 1e-8);
    basis.push(quintet[0].clone());
    let _ = sctx.a_gram(&basis);
}

#[test]
fn csr_threaded_matches_serial_for_grams_and_williamson() {
    let (space, t, v, ham, x) = capped_singlet_minimizer(2);
    let (bl, br) = su2_tangent_bases(&x);
    let out_space = space.with_two_s(TRIPLET);
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let run = || {
        let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let (a, b) = sctx.grams(&basis);
        let (hrr, _) = hrr_hii(&a, &b);
        let w = signed_omega2_or_fail(
            &sctx.symplectic_eigenvalues(&basis),
            "threading: the shifted dense driver",
        );
        let probe = &basis[0];
        let b_out = sctx.b_apply(probe);
        let bx: Vec<f64> = (0..b_out.num_coeffs()).map(|i| b_out.coeff(i)).collect();
        let h_out_xi = sctx.hess_apply(probe);
        let hx: Vec<f64> = (0..h_out_xi.num_coeffs())
            .map(|i| h_out_xi.coeff(i))
            .collect();
        (basis.len(), a, b, hrr, w, bx, hx)
    };
    let serial_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let (ns, as_, bs, hrrs, ws, bxs, hxs) = serial_pool.install(|| {
        assert_eq!(rayon::current_num_threads(), 1);
        run()
    });
    let par_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .expect("8-thread pool");
    let (np, ap, bp, hrrp, wp, bxp, hxp) = par_pool.install(|| {
        assert!(rayon::current_num_threads() > 1);
        run()
    });
    assert!(ns > 1);
    assert_eq!(
        ns, np,
        "the MGS cull accepted a different number of directions across pools"
    );
    let na = as_.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    let nb = bs.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    println!(
        "csr threading: n = {ns}, ||A|| = {na:e}, ||B|| = {nb:e}, |omega^2| = {}",
        ws.len()
    );
    assert!(
        nb > 1e-3 * na,
        "ANTI-VACUITY: a matches-serial pin over a zero B forks nothing"
    );
    for (name, s, q, sc) in [
        ("A", &as_, &ap, na),
        ("B", &bs, &bp, nb),
        ("H^rr", &hrrs, &hrrp, na + nb),
    ] {
        assert_eq!(s.len(), q.len());
        for (j, (a, b)) in s.iter().zip(q.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * sc.max(1.0),
                "threaded vs serial {name} gram entry {j}: {a} vs {b}"
            );
        }
    }
    for (name, s, q) in [("B xi", &bxs, &bxp), ("(A+B) xi", &hxs, &hxp)] {
        assert_eq!(s.len(), q.len());
        let sc = s.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
        for (j, (a, b)) in s.iter().zip(q.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-12 * sc,
                "threaded vs serial {name} coeff {j}: {a} vs {b}"
            );
        }
    }
    assert_eq!(ws.len(), wp.len());
    let wscale = ws.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
    for (j, (a, b)) in ws.iter().zip(wp.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-10 * wscale,
            "threaded vs serial signed omega^2[{j}]: {a} vs {b}"
        );
    }
}

#[test]
fn csr_quintet_channel_b_is_structurally_zero() {
    for cap in [2usize, 1] {
        let (space, t, v, ham, x) = truncated_reference(cap, 0xC6_80 + cap as u64);
        let (bl, br) = su2_tangent_bases(&x);
        let rperp = su2_residual_norm(&ham, &bl, &br, &x);
        assert!(rperp > 1e-6, "[cap {cap}] premise: r_perp ≠ 0");
        let ctx = su2_weingarten_operator(&ham, &x);
        let h_t = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
        let h_q = Su2SparseHamiltonian::new(space.with_two_s(QUINTET), &t, &v, None, f64::EPSILON);
        let sctx_t = Su2SectorCtx::new(&ctx, &h_t, TRIPLET);
        let sctx_q = Su2SectorCtx::new(&ctx, &h_q, QUINTET);
        let bt = sctx_t.basis();
        let bq = sctx_q.basis();
        assert_eq!(
            bq.len(),
            1,
            "[cap {cap}] the quintet channel is 1-dimensional"
        );
        let (at, btg) = sctx_t.grams(&bt);
        let (aq, bqg) = sctx_q.grams(&bq);
        let (nat, nbt) = (
            gram_spectral_norm(&at, bt.len()),
            gram_spectral_norm(&btg, bt.len()),
        );
        println!(
            "csr quintet [cap {cap}]: ||r_perp|| = {rperp:e}, triplet ||B||/||A|| = {:.3e}, quintet A = {:e}, B = {:e}",
            nbt / nat.max(f64::MIN_POSITIVE),
            aq[0],
            bqg[0]
        );
        assert!(
            nbt > 1e-3 * nat,
            "[cap {cap}] ANTI-VACUITY: the triplet B must be large at this reference"
        );
        assert!(
            bqg[0].abs() <= 1e-12 * aq[0].abs().max(1.0),
            "[cap {cap}] B^(S=2) = {} on a single-slot channel — structurally zero (the affine-curve theorem)",
            bqg[0]
        );
        let xi = &bq[0];
        assert!(sctx_q.b_apply(xi).norm() <= 1e-12 * sctx_q.a_apply(xi).norm().max(1.0));
        let spec = sctx_q.symplectic_eigenvalues(&bq);
        let w = signed_omega2_or_fail(&spec, &format!("[cap {cap}] quintet dense"));
        assert_eq!(w.len(), 1);
        assert!(
            (w[0] - aq[0] * aq[0]).abs() <= 1e-10 * (aq[0] * aq[0]).max(1.0),
            "[cap {cap}] omega^2 = eig(A)^2 where B = 0"
        );
        for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
            let opts = SectorOpts {
                readout: SectorReadout::Rpa,
                seed: 0xC6_81,
                kernel,
                ..SectorOpts::default()
            };
            let res = sctx_q.sector_lowest(1, &opts);
            println!(
                "csr quintet [cap {cap}] iterative {kernel:?}: converged {}, applies {}, omega2 {:?}",
                res.converged,
                res.applies,
                res.omega2()
            );
            assert!(
                res.converged,
                "[cap {cap}] a 1-dimensional channel converges in one block"
            );
            let w_it = res.omega2().expect("Rpa");
            assert_eq!(w_it.len(), 1);
            assert!(
                (w_it[0] - w[0]).abs() <= 10.0 * opts.tol * w[0].abs().max(1.0),
                "[cap {cap}] iterative {} vs dense {}",
                w_it[0],
                w[0]
            );
        }
    }
}

#[test]
fn csr_point_group_b_gram_is_symmetric_and_consistent() {
    let pg = PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(1), Irrep::new(0)],
    );
    let (t, v) = pg_tv(pg);
    let space = Su2Space::with_point_group(D4, NEL, 0, pg, Irrep::IDENTITY);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC6_90);
    let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, 1), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let rperp = su2_residual_norm(&ham, &bl, &br, &x);
    let out_space = space.with_two_s(TRIPLET);
    assert_eq!((out_space.pg, out_space.target), (space.pg, space.target));
    let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham, &x);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let basis = sctx.basis();
    let n = basis.len();
    assert!(n > 1, "ANTI-VACUITY: the pg triplet channel is trivial");
    let (a, b) = sctx.grams(&basis);
    let (na, nb) = (gram_spectral_norm(&a, n), gram_spectral_norm(&b, n));
    let (hrr, hii) = hrr_hii(&a, &b);
    println!(
        "csr pg-B: dim {n}, ||r_perp|| = {rperp:e}, ||A|| = {na:e}, ||B|| = {nb:e}, ||B||/||A|| = {:.3e}",
        nb / na.max(f64::MIN_POSITIVE)
    );
    assert!(rperp > 1e-6, "premise: r_perp ≠ 0 at a truncated reference");
    assert!(a.iter().chain(b.iter()).all(|z| z.is_finite()));
    assert!(
        nb <= 1e-12 * na,
        "the pg triplet channel is single-slot, so B^(S=1) must be STRUCTURALLY zero at any base \
         point (measured 5.0e-15 relative at cap 1 with ||r_perp|| = 0.76): got ||B|| = {nb:e} \
         against ||A|| = {na:e}"
    );
    assert!(
        gram_asymmetry(&b, n) <= 1e-12 * na,
        "pg B Gram not symmetric at the scale of the operator that exists (||A|| = {na:e})"
    );
    assert!(
        gram_asymmetry(&hrr, n) <= 1e-12 * (na + nb),
        "pg A + B not symmetric"
    );
    assert!(
        gram_asymmetry(&hii, n) <= 1e-12 * (na + nb),
        "pg A − B not symmetric"
    );
    let z = su2_normal_residual(&ham, &bl, &br, &x, lambda);
    let xi = &basis[0];
    let via_ctx = sctx.b_apply(xi);
    let via_pub =
        su2_weingarten_assemble(&bl, &br, &z, xi, FloorPolicy::default_hard(), Terms::all());
    assert!(
        via_ctx.sub(&via_pub).norm() <= 1e-10 * na,
        "pg: the split-Z route and the explicit normal residual disagree ({:e} vs ||A|| = {na:e}, \
         ||B xi|| = {:e})",
        via_ctx.sub(&via_pub).norm(),
        via_pub.norm()
    );
}

#[test]
fn csr_d6_shifted_b_gram_matches_the_expanded_abelian_block() {
    const D6: usize = 6;
    const N6: usize = 6;
    let (t, v) = generic_truncated_tv(D6);
    let space = Su2Space::new(D6, N6, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_C0);
    let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, 2), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check(), "the d = 6 reference must be a legal train");
    let (bl, br) = su2_tangent_bases(&x);
    let lambda = su2_rayleigh(&ham, &x).re_s();
    let rperp = su2_residual_norm(&ham, &bl, &br, &x);
    let channels: Vec<usize> = (0..=N6.min(2 * D6 - N6)).step_by(2).collect();
    let dims: Vec<usize> = channels
        .iter()
        .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
        .collect();
    let x_ab = expand(&x, 0);
    let h_ab = SparseHamiltonian::<f64>::new(x_ab.space, &t, &v, None, f64::EPSILON);
    let lam_ab = qntt::rayleigh_quotient(&h_ab, &x_ab).re_s();
    let ctx_ab = ab_weingarten_operator(&h_ab, &x_ab);
    let (bl_ab, br_ab) = ctx_ab.bases();
    let dim_ab = ab_tangent_dim(bl_ab);
    println!(
        "csr d6-B: lambda = {lambda}, ||r_perp|| = {rperp:e}, sigma_min = {:e}, dims {dims:?}, \
         tangent_dim(expand) {dim_ab}",
        su2_min_schmidt(&bl, &br)
    );
    assert_eq!(
        dims,
        vec![95, 121, 31, 1],
        "the d = 6 channel dims are grid-determined"
    );
    assert_eq!(
        dims.iter().sum::<usize>(),
        dim_ab,
        "the channels must PARTITION the expanded abelian tangent space before a per-channel \
         comparison means anything"
    );
    assert!(
        (lam_ab - lambda).abs() <= 1e-12 * lambda.abs().max(1.0),
        "PREMISE: the two layers' lambda must agree ({lambda} vs {lam_ab})"
    );
    assert!(
        rperp > 1e-6,
        "premise: B is linear in r_perp, so a zero residual would certify A alone"
    );
    let ctx = su2_weingarten_operator(&ham, &x);

    let mut curved = 0usize;
    for &two_s in &channels[1..] {
        let h_out = Su2SparseHamiltonian::new(space.with_two_s(two_s), &t, &v, None, f64::EPSILON);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_s);
        let basis = sctx.basis();
        let n = basis.len();
        assert!(n > 0, "d6 2S'={two_s}: the channel must be live");
        let emb: Vec<TtTangent<f64>> = basis
            .iter()
            .map(|u| {
                let y = expand(&u.to_vector(), 0);
                let e = TtTangent::project(bl_ab, br_ab, &y);
                assert!(
                    (e.norm() - u.norm()).abs() <= 1e-12 * u.norm().max(1.0),
                    "d6 2S'={two_s}: expand∘iota is not isometric ({} vs {})",
                    e.norm(),
                    u.norm()
                );
                let defect = (&e.to_vector() - &y).norm_lr(false);
                assert!(
                    defect <= 1e-10 * y.norm_lr(false).max(1.0),
                    "d6 2S'={two_s}: expand(iota(xi), 0) is not ALREADY tangent at x_ab ({defect:e})"
                );
                e
            })
            .collect();
        let (a, b) = sctx.grams(&basis);
        let mut a1 = vec![0.0f64; n * n];
        let mut b1 = vec![0.0f64; n * n];
        let mut worst_inv = 0.0f64;
        for (j, ej) in emb.iter().enumerate() {
            let (av, bv) = (ctx_ab.a_apply_fused(ej), ctx_ab.b_apply(ej));
            for (i, ei) in emb.iter().enumerate() {
                a1[i * n + j] = ei.dot(&av);
                b1[i * n + j] = ei.dot(&bv);
            }
            if j < 3 {
                let mut resid = bv.clone();
                for (i, ei) in emb.iter().enumerate() {
                    resid.axpy(-b1[i * n + j], ei);
                }
                worst_inv = worst_inv.max(resid.norm());
            }
        }
        let (na, nb) = (gram_spectral_norm(&a, n), gram_spectral_norm(&b, n));
        let worst_a = a
            .iter()
            .zip(a1.iter())
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        let worst_b = b
            .iter()
            .zip(b1.iter())
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        println!(
            "csr d6-B 2S'={two_s}: n = {n}, ||A|| = {na:e}, ||B|| = {nb:e}, ratio {:.3e}, \
             max |A_su2 - A_ab| = {worst_a:e}, max |B_su2 - B_ab| = {worst_b:e}, \
             span-invariance defect = {worst_inv:e} (absolute)",
            nb / na.max(f64::MIN_POSITIVE)
        );
        let scale = na.max(nb).max(1.0);
        let (scale_a, scale_b) = (na.max(1.0), nb.max(1.0));
        assert!(
            worst_inv <= 1e-10 * scale,
            "d6 2S'={two_s}: B_ab maps the embedded channel OUT of itself by {worst_inv:e} against \
             scale {scale:e} — the spin block-diagonality the entrywise comparison rests on"
        );
        assert!(
            worst_a <= 1e-9 * scale_a,
            "d6 2S'={two_s}: A^S' disagrees with the abelian block ENTRYWISE by {worst_a:e} \
             (against ||A|| = {na:e})"
        );
        assert!(
            worst_b <= 1e-9 * scale_b,
            "d6 2S'={two_s}: B^S' disagrees with the abelian block ENTRYWISE by {worst_b:e} \
             (against ||B|| = {nb:e}) — at 2S' = 4 this is the campaign's ONLY test of the \
             crossing's phase at |s_t - s_u| = 2"
        );
        let (hrr, hii) = hrr_hii(&a, &b);
        let (hrr1, hii1) = hrr_hii(&a1, &b1);
        for (what, got, want, sc) in [
            ("B", &b, &b1, scale_b),
            ("A", &a, &a1, scale_a),
            ("A+B", &hrr, &hrr1, (na + nb).max(1.0)),
            ("A-B", &hii, &hii1, (na + nb).max(1.0)),
        ] {
            assert_multiset_eq(
                &gram_eigs(got, n),
                &gram_eigs(want, n),
                1e-9 * sc,
                &format!("d6 2S'={two_s}: eig({what}) vs the abelian block"),
            );
        }
        if n == 1 {
            assert!(
                nb <= 1e-12 * na,
                "d6 2S'={two_s}: a 1-dimensional channel varies ONE core slot, so B ≡ 0 \
                 structurally; got ||B|| = {nb:e} against ||A|| = {na:e}"
            );
        } else {
            assert!(
                nb > 1e-3 * na,
                "d6 2S'={two_s} ANTI-VACUITY: ||B|| = {nb:e} vs ||A|| = {na:e}"
            );
            curved += 1;
        }
        let spec = sctx.symplectic_eigenvalues(&basis);
        let spec_ab = qntt::solvers::hess::symplectic_from_grams(&a1, &b1, n, 1e-10);
        println!(
            "csr d6-B 2S'={two_s}: dense su2 {} vs abelian block {}",
            spectrum_label(&spec),
            spectrum_label(&spec_ab)
        );
        match (spec.omega_squared(), spec_ab.omega_squared()) {
            (Some(mut w), Some(mut w1)) => {
                w.sort_by(|p, q| p.partial_cmp(q).unwrap());
                w1.sort_by(|p, q| p.partial_cmp(q).unwrap());
                let ws = w1.iter().fold(1.0f64, |m, z| m.max(z.abs()));
                assert_multiset_eq(
                    &w,
                    &w1,
                    1e-8 * ws,
                    &format!("d6 2S'={two_s}: signed omega^2 vs the abelian block"),
                );
            }
            (None, None) => {
                let (
                    SymplecticSpectrum::Unstable {
                        generator_eigs: g, ..
                    },
                    SymplecticSpectrum::Unstable {
                        generator_eigs: g1, ..
                    },
                ) = (&spec, &spec_ab)
                else {
                    panic!(
                        "d6 2S'={two_s}: omega_squared() is None only on the Unstable arm (su2 {}, \
                         abelian {})",
                        spectrum_label(&spec),
                        spectrum_label(&spec_ab)
                    )
                };
                let gs = g1.iter().fold(0.0f64, |m, z| m.max(z.norm())).max(1.0);
                let worst = assert_complex_multiset_eq(
                    g,
                    g1,
                    1e-8 * gs,
                    &format!("d6 2S'={two_s}: eig(𝓜) vs the abelian block's generator"),
                );
                println!(
                    "csr d6-B 2S'={two_s}: both Unstable — {} generator eigenvalues matched, worst \
                     {worst:e} (scale {gs:e})",
                    g.len()
                );
            }
            _ => panic!(
                "d6 2S'={two_s}: the two layers routed the SAME pair differently (su2 {}, abelian {})",
                spectrum_label(&spec),
                spectrum_label(&spec_ab)
            ),
        }
    }
    assert!(
        curved >= 2,
        "d6: both the 2S' = 2 and the 2S' = 4 channel must be CURVED, or the phase at \
         |s_t - s_u| = 2 is untested anywhere in the campaign"
    );
}

const D6PG: usize = 6;

fn pg6_tv(pg: PointGroup) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let d = D6PG;
    let allowed = |sites: [usize; 2]| pg.irrep_of_sites(sites) == Irrep::IDENTITY;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        t.set(&[p, p], 0.13 * (p as f64 - 0.5 * d as f64));
    }
    for p in 0..d {
        for q in p + 1..d {
            if allowed([p, q]) {
                let h = -0.8 + 0.07 * p as f64;
                t.set(&[p, q], h);
                t.set(&[q, p], h);
            }
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
                v.set(&[p, q, q, p], 0.23 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    'outer: for p in 0..d {
        for q in p + 1..d {
            if !allowed([p, q]) {
                continue;
            }
            for r in 0..d {
                for s in r + 1..d {
                    if (r, s) == (p, q) || !allowed([r, s]) {
                        continue;
                    }
                    for idx in [[p, q, r, s], [r, s, p, q], [q, p, s, r], [s, r, q, p]] {
                        v.set(&idx, 0.19);
                    }
                    break 'outer;
                }
            }
        }
    }
    screen_point_group_forbidden(&mut t, &mut v, pg)
        .expect("the fixture is symmetry-allowed by construction — the screen must be a no-op");
    (t, v)
}

const PG6_TABLES: [(u8, [u8; 6]); 3] = [
    (2, [2, 1, 0, 2, 1, 0]),
    (2, [0, 1, 1, 2, 2, 0]),
    (1, [0, 0, 0, 1, 1, 1]),
];

const PG6_SCAN_SEED: u64 = 0xD6_00;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pg6Cand {
    table: usize,
    nel: usize,
    g0: u8,
    cap: usize,
    dim: usize,
    planes: usize,
    curved: bool,
}

fn pg6_scan() -> Vec<Pg6Cand> {
    let mut out = Vec::new();
    for (ti, (nbits, orb)) in PG6_TABLES.iter().enumerate() {
        let irreps: Vec<Irrep> = orb.iter().map(|&i| Irrep::new(i)).collect();
        let pg = PointGroup::new(*nbits, &irreps);
        let (t, v) = pg6_tv(pg);
        for nel in [4usize, 6] {
            for g0 in 0..(1u8 << *nbits) {
                let space = Su2Space::with_point_group(D6PG, nel, 0, pg, Irrep::new(g0));
                for cap in [1usize, 2] {
                    let mut rng = Xoshiro256PlusPlus::seed_from_u64(PG6_SCAN_SEED);
                    let mut x: Su2TtVector<f64> =
                        su2_tt_randn(space, &capped_grids(&space, cap), &mut rng);
                    x.scale_mut(1.0 / x.norm_lr(false));
                    assert!(
                        x.check(),
                        "{orb:?} N={nel} g0={g0} cap={cap}: illegal train"
                    );
                    let (bl, br) = su2_tangent_bases(&x);
                    let dim = tangent_dim_grids(&space, &bl.ranks, TRIPLET);
                    let planes = (0..=D6PG)
                        .map(|b| {
                            space
                                .bond_sectors(b)
                                .into_iter()
                                .filter(|&s| bl.ranks[b][s] > 0)
                                .map(|s| s.irrep.index())
                                .collect::<std::collections::BTreeSet<u8>>()
                                .len()
                        })
                        .max()
                        .unwrap_or(0);
                    if dim == 0 {
                        println!("csr pg6-scan {orb:?} N={nel} g0={g0} cap={cap}: dim 0 (EMPTY)");
                        out.push(Pg6Cand {
                            table: ti,
                            nel,
                            g0,
                            cap,
                            dim,
                            planes,
                            curved: false,
                        });
                        continue;
                    }
                    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
                    let h_out = Su2SparseHamiltonian::new(
                        space.with_two_s(TRIPLET),
                        &t,
                        &v,
                        None,
                        f64::EPSILON,
                    );
                    let ctx = su2_weingarten_operator(&ham, &x);
                    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
                    let basis = sctx.basis();
                    let n = basis.len();
                    let mut xi = basis[0].scale(0.0);
                    for (i, u) in basis.iter().enumerate() {
                        let sgn = if (i * 7 + 3) % 5 < 2 { -1.0 } else { 1.0 };
                        xi.axpy(sgn / (n as f64).sqrt(), u);
                    }
                    let (bn, an) = (sctx.b_apply(&xi).norm(), sctx.a_apply(&xi).norm());
                    let probe = bn / an.max(f64::MIN_POSITIVE);
                    let curved = probe > 1e-3;
                    println!(
                        "csr pg6-scan {orb:?} N={nel} g0={g0} cap={cap}: dim {dim} n {n} planes \
                         {planes} ||r_perp|| {:.2e} probe ||B xi||/||A xi|| {probe:.3e} => {}",
                        su2_residual_norm(&ham, &bl, &br, &x),
                        if curved { "CURVED" } else { "flat" }
                    );
                    out.push(Pg6Cand {
                        table: ti,
                        nel,
                        g0,
                        cap,
                        dim,
                        planes,
                        curved,
                    });
                }
            }
        }
    }
    out
}

#[allow(clippy::type_complexity)]
const PG6_FIXTURES: [(&str, usize, usize, u8, usize, u64, &[usize], usize, usize); 2] = [
    ("smallest", 1, 4, 3, 1, 0xD6_06, &[16, 17, 3], 36, 2),
    ("four-plane", 0, 6, 1, 1, 0xD6_01, &[32, 32, 6, 0], 70, 4),
];

fn pg6_reference(
    table: usize,
    nel: usize,
    g0: u8,
    cap: usize,
    seed: u64,
) -> (
    Su2Space,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
) {
    let (nbits, orb) = PG6_TABLES[table];
    let irreps: Vec<Irrep> = orb.iter().map(|&i| Irrep::new(i)).collect();
    let pg = PointGroup::new(nbits, &irreps);
    let (t, v) = pg6_tv(pg);
    let space = Su2Space::with_point_group(D6PG, nel, 0, pg, Irrep::new(g0));
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, cap), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check(), "the d = 6 pg reference must be a legal train");
    (space, t, v, ham, x)
}

fn pg6_widest_plane_count(space: &Su2Space, ranks: &[Su2RankGrid]) -> usize {
    (0..=space.d)
        .map(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| ranks[b][s] > 0)
                .map(|s| s.irrep.index())
                .collect::<std::collections::BTreeSet<u8>>()
                .len()
        })
        .max()
        .unwrap_or(0)
}

#[test]
fn csr_d6_pg_curved_channel_scan_and_dimension_identity() {
    let cands = pg6_scan();
    let curved: Vec<Pg6Cand> = cands.iter().copied().filter(|c| c.curved).collect();
    let flat: Vec<Pg6Cand> = cands.iter().copied().filter(|c| !c.curved).collect();
    println!(
        "csr pg6-scan: {} candidates, {} CURVED, {} flat",
        cands.len(),
        curved.len(),
        flat.len()
    );
    assert!(
        !curved.is_empty(),
        "the SEARCH FAILED: no point-group-typed d = 6 SU(2) triplet channel came back curved, so \
         arm 24 has no fixture and the curved-channel gap is not closed. Widen \
         PG6_TABLES (or go to d = 8) rather than relaxing the 1e-3 curvature threshold"
    );
    assert!(
        !flat.is_empty(),
        "ANTI-VACUITY of the SEARCH: every candidate was curved, which means the probe is not \
         discriminating (a flat channel must exist — the single-slot zero — and the \
         d = 4 table's N = 4 rows are it)"
    );
    let pick = |want_planes: usize| -> Pg6Cand {
        *curved
            .iter()
            .filter(|c| c.planes >= want_planes)
            .min_by_key(|c| (c.dim, c.table, c.nel, c.g0, c.cap))
            .expect("a curved candidate at this plane count")
    };
    for &(label, table, nel, g0, cap, _seed, want_dims, _want_ab, want_planes) in &PG6_FIXTURES {
        let got = pick(want_planes);
        println!(
            "csr pg6-scan [{label}]: minimal curved candidate at planes >= {want_planes} is {got:?}"
        );
        assert_eq!(
            (got.table, got.nel, got.g0, got.cap),
            (table, nel, g0, cap),
            "[{label}] PG6_FIXTURES no longer names the minimal curved candidate of the scan — \
             re-choose it from the printed scan rather than keeping a stale literal"
        );
        assert_eq!(
            got.dim, want_dims[1],
            "[{label}] dim T^(S=1) is grid-determined"
        );
    }

    for &(label, table, nel, g0, cap, seed, want_dims, want_ab, want_planes) in &PG6_FIXTURES {
        let (space, _t, _v, _ham, x) = pg6_reference(table, nel, g0, cap, seed);
        let (bl, br) = su2_tangent_bases(&x);
        let planes = pg6_widest_plane_count(&space, &bl.ranks);
        let channels: Vec<usize> = (0..=nel.min(2 * D6PG - nel)).step_by(2).collect();
        let dims: Vec<usize> = channels
            .iter()
            .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
            .collect();
        let x_ab = expand(&x, 0);
        let dim_ab = ab_tangent_dim(&x_ab);
        println!(
            "csr pg6-dims [{label}]: channels {channels:?}, dims {dims:?}, sum {}, \
             tangent_dim(expand) {dim_ab}, widest plane count {planes}, sigma_min {:e}",
            dims.iter().sum::<usize>(),
            su2_min_schmidt(&bl, &br)
        );
        assert_eq!(
            planes, want_planes,
            "[{label}] ANTI-VACUITY: the widest bond plane count is the whole point of a \
             point-group fixture"
        );
        assert_eq!(dims, want_dims.to_vec(), "[{label}] the dims are measured");
        assert_eq!(
            dims.iter().sum::<usize>(),
            dim_ab,
            "[{label}] ★ the SU(2) spin channels must partition the Γ0-TYPED expanded abelian \
             tangent space: Σ_S' dim T^S' = tangent_dim(expand(x, 0)) (with_two_s moves no irrep, \
             so no forget_point_group anywhere in this identity)"
        );
        assert_eq!(dim_ab, want_ab, "[{label}] tangent_dim(expand) is measured");
        let (bl_ab, br_ab) = ab_tangent_bases(&x_ab);
        let basis_ab = ab_tangent_basis(&bl_ab, &br_ab, true);
        let basis_0 = su2_tangent_basis(&bl, &br, true);
        let n_shifted: usize = channels[1..]
            .iter()
            .map(|&k| tangent_basis_shifted(&bl, &br, k).len())
            .sum();
        assert_eq!(
            basis_ab.len(),
            basis_0.len() + n_shifted,
            "[{label}] basis lengths: {} abelian vs {} + {n_shifted} shifted",
            basis_ab.len(),
            basis_0.len()
        );
        assert_eq!(
            (dims[0] - 1) + dims[1..].iter().sum::<usize>(),
            dim_ab - 1,
            "[{label}] the radial-deflated form of the same identity (one radial mode per side, \
             absent from every shifted channel)"
        );
        if label == "four-plane" {
            assert_eq!(
                dims[3], 0,
                "[four-plane] the 2S' = 6 channel is EMPTY at this grid (a borrowed-suffix \
                 outcome, not a failure)"
            );
        }
    }
}

#[test]
fn csr_d6_pg_curved_triplet_matches_the_expanded_abelian_block() {
    let mut curved_legs: Vec<(String, usize, f64, usize)> = Vec::new();
    for &(label, table, nel, g0, cap, seed, want_dims, want_ab, want_planes) in &PG6_FIXTURES {
        let (space, t, v, ham, x) = pg6_reference(table, nel, g0, cap, seed);
        let (bl, br) = su2_tangent_bases(&x);
        let lambda = su2_rayleigh(&ham, &x).re_s();
        let rperp = su2_residual_norm(&ham, &bl, &br, &x);
        let planes = pg6_widest_plane_count(&space, &bl.ranks);
        let channels: Vec<usize> = (0..=nel.min(2 * D6PG - nel)).step_by(2).collect();
        let dims: Vec<usize> = channels
            .iter()
            .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
            .collect();

        let x_ab = expand(&x, 0);
        let h_ab = SparseHamiltonian::<f64>::new(x_ab.space, &t, &v, None, f64::EPSILON);
        let lam_ab = qntt::rayleigh_quotient(&h_ab, &x_ab).re_s();
        let ctx_ab = ab_weingarten_operator(&h_ab, &x_ab);
        let (bl_ab, br_ab) = ctx_ab.bases();
        let dim_ab = ab_tangent_dim(bl_ab);
        println!(
            "csr pg6-B [{label}]: lambda = {lambda}, ||r_perp|| = {rperp:e}, sigma_min = {:e}, \
             dims {dims:?}, tangent_dim(expand) {dim_ab}, widest plane count {planes}",
            su2_min_schmidt(&bl, &br)
        );
        assert_eq!(
            (space.pg, space.target),
            (x_ab.space.pg, x_ab.space.target),
            "[{label}] expand must preserve the point group and the target irrep"
        );
        assert_eq!(planes, want_planes, "[{label}] widest plane count");
        assert_eq!(dims, want_dims.to_vec(), "[{label}] the dims are measured");
        assert_eq!(dims.iter().sum::<usize>(), dim_ab, "[{label}] sum identity");
        assert_eq!(dim_ab, want_ab, "[{label}] tangent_dim(expand)");
        assert!(
            (lam_ab - lambda).abs() <= 1e-12 * lambda.abs().max(1.0),
            "[{label}] PREMISE: the two layers' lambda must agree ({lambda} vs {lam_ab})"
        );
        assert!(
            rperp > 1e-6,
            "[{label}] premise: B is linear in r_perp, so a zero residual would certify A alone"
        );

        let ctx = su2_weingarten_operator(&ham, &x);
        let basis_ab = ab_tangent_basis(bl_ab, br_ab, true);
        let n_ab = basis_ab.len();
        let (a_ab, b_ab) = ab_hessian_grams(&ctx_ab, &basis_ab);
        let (hrr_ab, hii_ab) = hrr_hii(&a_ab, &b_ab);
        let basis_0 = su2_tangent_basis(&bl, &br, true);
        let n_0 = basis_0.len();
        let (a_0, b_0) = su2_hessian_grams(&ctx, &basis_0);
        let (hrr_0, hii_0) = hrr_hii(&a_0, &b_0);
        let mut union_hrr = gram_eigs(&hrr_0, n_0);
        let mut union_hii = gram_eigs(&hii_0, n_0);
        let mut union_gen = generator_eigs(&a_0, &b_0, n_0);

        for (ci, &two_s) in channels.iter().enumerate() {
            if two_s == 0 {
                continue;
            }
            let h_out =
                Su2SparseHamiltonian::new(space.with_two_s(two_s), &t, &v, None, f64::EPSILON);
            assert_eq!(
                (h_out.space.pg, h_out.space.target),
                (space.pg, space.target),
                "[{label}] with_two_s must change ONLY two_s: the spin flux carries no irrep"
            );
            let sctx = Su2SectorCtx::new(&ctx, &h_out, two_s);
            let basis = sctx.basis();
            let n = basis.len();
            assert_eq!(
                n, dims[ci],
                "[{label}] 2S'={two_s}: the enumerator must reach the structural dim"
            );
            if n == 0 {
                println!("csr pg6-B [{label}] 2S'={two_s}: EMPTY channel");
                continue;
            }
            let emb: Vec<TtTangent<f64>> = basis
                .iter()
                .map(|u| {
                    let y = expand(&u.to_vector(), 0);
                    assert_eq!(
                        (y.space.pg, y.space.target),
                        (space.pg, space.target),
                        "[{label}] 2S'={two_s}: expand(iota(xi), 0) left the irrep plane"
                    );
                    let e = TtTangent::project(bl_ab, br_ab, &y);
                    assert!(
                        (e.norm() - u.norm()).abs() <= 1e-12 * u.norm().max(1.0),
                        "[{label}] 2S'={two_s}: expand∘iota is not isometric ({} vs {})",
                        e.norm(),
                        u.norm()
                    );
                    let defect = (&e.to_vector() - &y).norm_lr(false);
                    assert!(
                        defect <= 1e-10 * y.norm_lr(false).max(1.0),
                        "[{label}] 2S'={two_s}: expand(iota(xi), 0) is not ALREADY tangent at \
                         x_ab ({defect:e})"
                    );
                    e
                })
                .collect();
            let (a, b) = sctx.grams(&basis);
            let mut a1 = vec![0.0f64; n * n];
            let mut b1 = vec![0.0f64; n * n];
            let mut worst_inv = 0.0f64;
            for (j, ej) in emb.iter().enumerate() {
                let (av, bv) = (ctx_ab.a_apply_fused(ej), ctx_ab.b_apply(ej));
                for (i, ei) in emb.iter().enumerate() {
                    a1[i * n + j] = ei.dot(&av);
                    b1[i * n + j] = ei.dot(&bv);
                }
                if j < 3 {
                    let mut resid = bv.clone();
                    for (i, ei) in emb.iter().enumerate() {
                        resid.axpy(-b1[i * n + j], ei);
                    }
                    worst_inv = worst_inv.max(resid.norm());
                }
            }
            let (na, nb) = (gram_spectral_norm(&a, n), gram_spectral_norm(&b, n));
            let ratio = nb / na.max(f64::MIN_POSITIVE);
            let worst_a = a
                .iter()
                .zip(a1.iter())
                .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
            let worst_b = b
                .iter()
                .zip(b1.iter())
                .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
            println!(
                "csr pg6-B [{label}] 2S'={two_s}: n = {n}, ||A|| = {na:e}, ||B|| = {nb:e}, ratio \
                 {ratio:.3e}, max |A_su2 - A_ab| = {worst_a:e}, max |B_su2 - B_ab| = {worst_b:e}, \
                 span-invariance defect = {worst_inv:e} (absolute)"
            );
            let (scale_a, scale_b) = (na.max(1.0), nb.max(1.0));
            assert!(
                worst_inv <= 1e-10 * na.max(nb).max(1.0),
                "[{label}] 2S'={two_s}: B_ab maps the embedded channel OUT of itself by \
                 {worst_inv:e} — the spin block-diagonality the entrywise comparison rests on"
            );
            assert!(
                worst_a <= 1e-9 * scale_a,
                "[{label}] 2S'={two_s}: A^S' disagrees with the abelian block ENTRYWISE by \
                 {worst_a:e} (against ||A|| = {na:e})"
            );
            assert!(
                worst_b <= 1e-9 * scale_b,
                "[{label}] 2S'={two_s}: B^S' disagrees with the abelian block ENTRYWISE by \
                 {worst_b:e} (against ||B|| = {nb:e}) — on a CURVED point-group-typed channel this \
                 is the certificate the curved-channel gap was missing"
            );
            let (hrr, hii) = hrr_hii(&a, &b);
            let (hrr1, hii1) = hrr_hii(&a1, &b1);
            for (what, got, want, sc) in [
                ("B", &b, &b1, scale_b),
                ("A", &a, &a1, scale_a),
                ("A+B", &hrr, &hrr1, (na + nb).max(1.0)),
                ("A-B", &hii, &hii1, (na + nb).max(1.0)),
            ] {
                assert_multiset_eq(
                    &gram_eigs(got, n),
                    &gram_eigs(want, n),
                    1e-9 * sc,
                    &format!("[{label}] 2S'={two_s}: eig({what}) vs the abelian block"),
                );
            }
            if n == 1 || ratio <= 1e-3 {
                assert!(
                    nb <= 1e-12 * na,
                    "[{label}] 2S'={two_s}: a FLAT channel must be structurally zero, not merely \
                     small: ||B|| = {nb:e} against ||A|| = {na:e}"
                );
            } else {
                curved_legs.push((label.to_string(), two_s, ratio, planes));
            }
            let spec = sctx.symplectic_eigenvalues(&basis);
            let spec_ab = qntt::solvers::hess::symplectic_from_grams(&a1, &b1, n, 1e-10);
            println!(
                "csr pg6-B [{label}] 2S'={two_s}: dense su2 {} vs abelian block {}",
                spectrum_label(&spec),
                spectrum_label(&spec_ab)
            );
            match (spec.omega_squared(), spec_ab.omega_squared()) {
                (Some(mut w), Some(mut w1)) => {
                    w.sort_by(|p, q| p.partial_cmp(q).expect("finite omega^2"));
                    w1.sort_by(|p, q| p.partial_cmp(q).expect("finite omega^2"));
                    let ws = w1.iter().fold(1.0f64, |m, z| m.max(z.abs()));
                    assert_multiset_eq(
                        &w,
                        &w1,
                        1e-8 * ws,
                        &format!("[{label}] 2S'={two_s}: signed omega^2 vs the abelian block"),
                    );
                }
                (None, None) => {
                    let (
                        SymplecticSpectrum::Unstable {
                            generator_eigs: g, ..
                        },
                        SymplecticSpectrum::Unstable {
                            generator_eigs: g1, ..
                        },
                    ) = (&spec, &spec_ab)
                    else {
                        panic!(
                            "[{label}] 2S'={two_s}: omega_squared() is None only on the Unstable \
                             arm (su2 {}, abelian {})",
                            spectrum_label(&spec),
                            spectrum_label(&spec_ab)
                        )
                    };
                    let gs = g1.iter().fold(0.0f64, |m, z| m.max(z.norm())).max(1.0);
                    let worst = assert_complex_multiset_eq(
                        g,
                        g1,
                        1e-8 * gs,
                        &format!("[{label}] 2S'={two_s}: eig(𝓜) vs the abelian block's generator"),
                    );
                    println!(
                        "csr pg6-B [{label}] 2S'={two_s}: both Unstable — {} generator eigenvalues \
                         matched, worst {worst:e} (scale {gs:e})",
                        g.len()
                    );
                }
                _ => panic!(
                    "[{label}] 2S'={two_s}: the two layers routed the SAME pair differently (su2 \
                     {}, abelian {})",
                    spectrum_label(&spec),
                    spectrum_label(&spec_ab)
                ),
            }
            union_hrr.extend(gram_eigs(&hrr, n));
            union_hii.extend(gram_eigs(&hii, n));
            union_gen.extend(generator_eigs(&a, &b, n));
        }

        assert_eq!(
            n_ab,
            union_gen.len() / 2,
            "[{label}] the union accumulators must cover the abelian basis exactly: {n_ab} \
             abelian directions vs {} over the blocks ({n_0} in S' = 0)",
            union_gen.len() / 2
        );
        for (what, mut got, want) in [
            ("H^rr", union_hrr, gram_eigs(&hrr_ab, n_ab)),
            ("H^ii", union_hii, gram_eigs(&hii_ab, n_ab)),
        ] {
            got.sort_by(|p, q| p.partial_cmp(q).expect("finite eigenvalue"));
            let sc = want.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            assert!(
                sc > 1e-6,
                "[{label}] ANTI-VACUITY: the abelian {what} spectrum is identically zero"
            );
            println!(
                "csr pg6-B [{label}]: |union {what}| = {}, |abelian| = {}, worst = {:e}",
                got.len(),
                want.len(),
                got.iter()
                    .zip(want.iter())
                    .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()))
            );
            assert_multiset_eq(
                &got,
                &want,
                1e-8 * sc,
                &format!(
                    "[{label}] union_S' eig({what}^S') vs the expanded abelian eig({what}) — the \
                     spin block-diagonality, stateable at EVERY base point"
                ),
            );
        }
        let gen_ab = generator_eigs(&a_ab, &b_ab, n_ab);
        let gs = gen_ab.iter().fold(0.0f64, |m, z| m.max(z.norm()));
        assert!(
            gs > 1e-6,
            "[{label}] ANTI-VACUITY: the abelian generator spectrum is identically zero"
        );
        let worst = assert_complex_multiset_eq(
            &union_gen,
            &gen_ab,
            1e-8 * gs,
            &format!(
                "[{label}] union_S' eig(M^S') vs the expanded abelian eig(M) — the \
                 always-stateable form of the omega^2 union identity"
            ),
        );
        println!(
            "csr pg6-B [{label}]: |union eig(M)| = {}, |abelian| = {}, max|eig(M)| = {gs:e}, \
             worst matched = {worst:e}",
            union_gen.len(),
            gen_ab.len()
        );
    }
    println!("csr pg6-B: curved legs {curved_legs:?}");
    assert!(
        curved_legs.iter().any(|&(_, _, r, p)| r >= 1e-3 && p >= 2),
        "COVERAGE: the curved-channel gap is NOT closed — no channel of any fixture \
         was both CURVED (||B||/||A|| >= 1e-3) and carried rank in more than one irrep plane, so \
         the crossing's coefficients are still certified only at |G| = 1 (legs: {curved_legs:?})"
    );
    assert!(
        curved_legs.iter().any(|&(_, k, _, _)| k == QUINTET),
        "COVERAGE: no CURVED point-group-typed channel at 2k = 4, so the crossing's phase at \
         |s_t - s_u| = 2 is still uncertified under a non-trivial group (legs: {curved_legs:?})"
    );
    assert!(
        curved_legs.iter().any(|&(_, _, _, p)| p == 4),
        "COVERAGE: no CURVED channel at a bond carrying all FOUR irrep planes, which is the \
         configuration where a per-plane rank sum in src_offset can be wrong without cancelling \
         (legs: {curved_legs:?})"
    );
}

#[test]
fn csr_fused_a_apply_matches_the_dressed_unfused_reference() {
    struct Leg {
        label: String,
        space: Su2Space,
        t: CoeffTensor<f64>,
        v: CoeffTensor<f64>,
        ham: Su2SparseHamiltonian<f64>,
        x: Su2TtVector<f64>,
        channels: Vec<usize>,
        full_basis: bool,
    }

    let mut legs: Vec<Leg> = Vec::new();
    for &(cap, seed) in &[(2usize, 0xC5_20u64), (1, 0xC5_21)] {
        let (space, t, v, ham, x) = truncated_reference(cap, seed);
        legs.push(Leg {
            label: format!("d = 4 trivial pg, cap {cap}"),
            space,
            t,
            v,
            ham,
            x,
            channels: vec![TRIPLET, QUINTET],
            full_basis: true,
        });
    }
    {
        const D6: usize = 6;
        const N6: usize = 6;
        let (t, v) = generic_truncated_tv(D6);
        let space = Su2Space::new(D6, N6, 0);
        let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let caps = su2_max_ranks(&space);
        let grids: Vec<Su2RankGrid> = (0..=D6)
            .map(|b| {
                let mut r = Su2RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    r[s] = caps[b][s].min(2);
                }
                r
            })
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_C0);
        let mut x: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        assert!(x.check());
        legs.push(Leg {
            label: "d = 6, N = 6, cap 2".to_string(),
            space,
            t,
            v,
            ham,
            x,
            channels: vec![TRIPLET, QUINTET],
            full_basis: false,
        });
    }
    for &(label, table, nel, g0, cap, seed, _dims, _ab, _planes) in &PG6_FIXTURES {
        let (space, t, v, ham, x) = pg6_reference(table, nel, g0, cap, seed);
        let channels: Vec<usize> = (2..=nel.min(2 * D6PG - nel)).step_by(2).collect();
        legs.push(Leg {
            label: format!("d = 6 pg [{label}]"),
            space,
            t,
            v,
            ham,
            x,
            channels,
            full_basis: false,
        });
    }

    let mut kappa4_reached = 0usize;
    let mut pg_reached = 0usize;
    let mut empty_reached = 0usize;
    for leg in legs {
        let Leg {
            label,
            space,
            t,
            v,
            ham,
            x,
            channels,
            full_basis,
        } = leg;
        let d = space.d;
        let (bl, br) = su2_tangent_bases(&x);
        let lambda = su2_rayleigh(&ham, &x).re_s();
        let ctx = su2_weingarten_operator(&ham, &x);
        let mut live = 0usize;
        for two_k in channels {
            let dim = tangent_dim_grids(&space, &bl.ranks, two_k);
            if dim == 0 {
                let out_space = space.with_two_s(two_k);
                let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
                let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
                assert_eq!(sctx.dim(), 0, "[{label}] 2k={two_k}: dim must be 0");
                assert!(
                    tangent_basis_shifted(&bl, &br, two_k).is_empty(),
                    "[{label}] 2k={two_k}: a zero-width channel carries no basis"
                );
                let (ga, gb) = sctx.grams(&[]);
                assert!(
                    ga.is_empty()
                        && gb.is_empty()
                        && sctx.a_gram(&[]).is_empty()
                        && sctx.a_gram_exact(&[]).is_empty(),
                    "[{label}] 2k={two_k}: every driver must answer the empty basis with an empty \
                     Gram, on a zero-width grid, without panicking"
                );
                empty_reached += 1;
                println!("csr fused apply [{label}] 2k={two_k}: EMPTY (feasible, zero-width)");
                continue;
            }
            live += 1;
            if two_k == QUINTET {
                kappa4_reached += 1;
            }
            if space.pg.order() > 1 {
                pg_reached += 1;
            }
            let out_space = space.with_two_s(two_k);
            let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
            let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
            let basis = tangent_basis_shifted(&bl, &br, two_k);
            assert!(
                !basis.is_empty(),
                "[{label}] 2k={two_k}: a channel with dim {dim} must carry a basis"
            );
            let step = if full_basis {
                1
            } else {
                basis.len().div_ceil(8)
            };
            let idx: Vec<usize> = (0..basis.len()).step_by(step).collect();
            assert!(
                !idx.is_empty(),
                "[{label}] 2k={two_k}: no direction would be checked"
            );

            let mut worst = 0.0f64;
            let mut weakest = f64::INFINITY;
            for &j in &idx {
                let xi = &basis[j];
                let fused = sctx.a_apply_fused(xi);
                let exact = sctx.a_apply_exact(xi);
                let free = effham_apply_shifted(&h_out, &bl, &br, two_k, lambda, xi);

                assert_eq!(exact.num_coeffs(), free.num_coeffs());
                for i in 0..free.num_coeffs() {
                    assert_eq!(
                        exact.coeff(i),
                        free.coeff(i),
                        "[{label}] 2k={two_k}, direction {j}, coeff {i}: a_apply_exact must BE \
                         effham_apply_shifted"
                    );
                }

                let en = exact.norm();
                weakest = weakest.min(en);
                worst = worst.max(fused.sub(&exact).norm() / en.max(1.0));

                assert_eq!(
                    fused.two_s_flux(),
                    two_k,
                    "[{label}] 2k={two_k}, direction {j}: the image's flux"
                );
                assert_eq!(
                    fused.rank(),
                    xi.rank(),
                    "[{label}] 2k={two_k}, direction {j}: the image must live on the channel's \
                     MASKED row grids"
                );
                assert_eq!(
                    fused.rank_col(),
                    xi.rank_col(),
                    "[{label}] 2k={two_k}, direction {j}: the image must live on the channel's \
                     DRESSED column grids, or A's range is not T^(S')"
                );
                assert_eq!(
                    fused.num_coeffs(),
                    exact.num_coeffs(),
                    "[{label}] 2k={two_k}, direction {j}: coefficient count"
                );
                for k in 0..d {
                    let (cf, ce) = (fused.component(k), exact.component(k));
                    assert_eq!(
                        cf.space, out_space,
                        "[{label}] 2k={two_k}: component {k} must come out stamped space_out, or \
                         dot_center weights with the WRONG total spin (the R4 metric error)"
                    );
                    assert_eq!(
                        cf.row_ranks, ce.row_ranks,
                        "[{label}] 2k={two_k}: component {k} row grid"
                    );
                    assert_eq!(
                        cf.col_ranks, ce.col_ranks,
                        "[{label}] 2k={two_k}: component {k} col grid — a 0-column block at one \
                         sector passes every norm bound and is visible only here"
                    );
                    for f in qntt::su2::core::Family::ALL {
                        for s in out_space.bond_sectors(k) {
                            assert_eq!(
                                cf.block_dims(f, s),
                                ce.block_dims(f, s),
                                "[{label}] 2k={two_k}: component {k} block dims at ({f:?}, {s:?})"
                            );
                        }
                    }
                }
            }
            println!(
                "csr fused apply [{label}] 2k={two_k}: dim {dim}, basis {}, {} directions, \
                 weakest ||A xi|| = {weakest:e}, worst rel = {worst:e}",
                basis.len(),
                idx.len()
            );
            assert!(
                weakest > 1e-6,
                "[{label}] 2k={two_k}: ANTI-VACUITY — A^(S') annihilated a checked direction \
                 (||A xi|| = {weakest:e}); a relative bound over a zero image proves nothing"
            );
            assert!(
                worst < 1e-11,
                "[{label}] 2k={two_k}: fused vs dressed-unfused shifted A-apply, worst relative \
                 deviation {worst:e} over {} directions. The contract is reassociation (the \
                 su2_h5.rs model), so anything above 1e-11 is a different operator — and a theta^2 \
                 inserted or dropped on either chain is an O(1) error, not a near miss",
                idx.len()
            );
        }
        assert!(
            live >= 1,
            "[{label}] no live shifted channel — this leg would be vacuous"
        );
    }
    assert!(
        kappa4_reached >= 1,
        "ANTI-VACUITY: no leg reached 2k = 4, where |s_t - s_u| = 2 and the reduced-storage phase \
         (-1)^(s_t - s_u) separates from the other functions of that difference"
    );
    assert!(
        pg_reached >= 1,
        "ANTI-VACUITY: no leg ran under a NON-TRIVIAL point group, which is the only place \
         h_out's automaton vertex set genuinely differs from h's — i.e. the only runnable \
         discriminator against a route that reused ctx.envs()"
    );
    assert!(
        empty_reached >= 1,
        "ANTI-VACUITY: no leg reached a FEASIBLE zero-width channel (measured: four-plane's \
         2S' = 6, the only one in this inventory — every QUINTET channel here is live at \
         dim 1), so the no-early-return-needed claim was never exercised from the integration \
         surface"
    );
}

#[test]
fn csr_fused_a_apply_at_the_minimal_feasible_shapes() {
    for &(d, nel, seed) in &[
        (2usize, 2usize, 0xC5_D2u64),
        (3, 2, 0xC5_D3),
        (3, 4, 0xC5_D4),
    ] {
        let (t, v) = generic_truncated_tv(d);
        let space = Su2Space::new(d, nel, 0);
        let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, 0), &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        assert!(x.check(), "d = {d}: the reference must be a legal train");
        let (bl, br) = su2_tangent_bases(&x);
        let lambda = su2_rayleigh(&ham, &x).re_s();
        let ctx = su2_weingarten_operator(&ham, &x);

        let dim = tangent_dim_grids(&space, &bl.ranks, TRIPLET);
        let out_space = space.with_two_s(TRIPLET);
        let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let basis = tangent_basis_shifted(&bl, &br, TRIPLET);
        let coeffs: Vec<usize> = (0..d).map(|k| basis[0].component(k).num_coeffs()).collect();
        println!(
            "csr fused minimal [d = {d}, N = {nel}]: dim = {dim}, basis = {}, per-component \
             coeffs = {coeffs:?}",
            basis.len()
        );
        assert!(
            !basis.is_empty(),
            "d = {d}, N = {nel}: fixture premise — the triplet channel must be NON-empty (the \
             minimal-shape claim is about a channel that carries a direction, not about an empty \
             one)"
        );
        if d >= 3 {
            assert!(
                coeffs.iter().filter(|&&c| c > 0).count() >= 2,
                "d = {d}: fixture premise — at least two components must be non-empty, or the \
                 TRANSPORT step this leg exists for folds structurally empty data (measured \
                 [1, 3, 1])"
            );
        }

        let mut worst = 0.0f64;
        let mut weakest = f64::INFINITY;
        for (j, xi) in basis.iter().enumerate() {
            let fused = sctx.a_apply_fused(xi);
            let exact = sctx.a_apply_exact(xi);
            let free = effham_apply_shifted(&h_out, &bl, &br, TRIPLET, lambda, xi);
            assert_eq!(exact.num_coeffs(), free.num_coeffs());
            for i in 0..free.num_coeffs() {
                assert_eq!(
                    exact.coeff(i),
                    free.coeff(i),
                    "d = {d}, direction {j}, coeff {i}: a_apply_exact must BE effham_apply_shifted"
                );
            }
            weakest = weakest.min(exact.norm());
            worst = worst.max(fused.sub(&exact).norm() / exact.norm().max(1.0));

            assert_eq!(fused.two_s_flux(), TRIPLET, "d = {d}, direction {j}: flux");
            assert_eq!(fused.rank(), xi.rank(), "d = {d}, direction {j}: row grids");
            assert_eq!(
                fused.rank_col(),
                xi.rank_col(),
                "d = {d}, direction {j}: column grids"
            );
            assert_eq!(
                fused.num_coeffs(),
                exact.num_coeffs(),
                "d = {d}, direction {j}: coefficient count"
            );
            for k in 0..d {
                let (cf, ce) = (fused.component(k), exact.component(k));
                assert_eq!(cf.space, out_space, "d = {d}: component {k} stamp");
                assert_eq!(
                    cf.row_ranks, ce.row_ranks,
                    "d = {d}: component {k} row grid"
                );
                assert_eq!(
                    cf.col_ranks, ce.col_ranks,
                    "d = {d}: component {k} col grid"
                );
            }
        }
        println!(
            "csr fused minimal [d = {d}, N = {nel}]: weakest ||A xi|| = {weakest:e}, worst rel = \
             {worst:e}"
        );
        assert!(
            weakest > 1e-6,
            "d = {d}, N = {nel}: ANTI-VACUITY — A^(S=1) annihilated a direction (||A xi|| = \
             {weakest:e})"
        );
        assert!(
            worst < 1e-11,
            "d = {d}, N = {nel}: fused vs dressed-unfused shifted A-apply, worst relative \
             deviation {worst:e}"
        );
    }
}

#[test]
fn csr_fused_a_gram_is_symmetric_and_agrees_with_the_paired_driver() {
    let mut checked = 0usize;
    for &(label, table, nel, g0, cap, seed, _dims, _ab, _planes) in &PG6_FIXTURES {
        let (space, t, v, ham, x) = pg6_reference(table, nel, g0, cap, seed);
        let (bl, br) = su2_tangent_bases(&x);
        let ctx = su2_weingarten_operator(&ham, &x);
        for two_k in [TRIPLET, QUINTET] {
            if tangent_dim_grids(&space, &bl.ranks, two_k) == 0 {
                continue;
            }
            let out_space = space.with_two_s(two_k);
            let h_out = Su2SparseHamiltonian::new(out_space, &t, &v, None, f64::EPSILON);
            let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
            let basis = tangent_basis_shifted(&bl, &br, two_k);
            let n = basis.len();
            assert!(
                n > 0,
                "[{label}] 2k={two_k}: a live channel carries a basis"
            );
            checked += 1;

            let g_fused = sctx.a_gram(&basis);
            let g_exact = sctx.a_gram_exact(&basis);
            let (g_paired, _b) = sctx.grams(&basis);
            assert_eq!(g_fused.len(), n * n);
            assert_eq!(g_paired.len(), n * n);
            let scale = g_exact.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            assert!(
                scale > 1e-3,
                "[{label}] 2k={two_k}: ANTI-VACUITY, the A Gram is identically zero"
            );

            let worst_fx = g_fused
                .iter()
                .zip(g_exact.iter())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            let worst_pair = g_fused
                .iter()
                .zip(g_paired.iter())
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            let asym = gram_asymmetry(&g_fused, n);
            let na = gram_spectral_norm(&g_fused, n);
            println!(
                "csr fused gram [{label}] 2k={two_k}: n = {n}, scale = {scale:e}, ||A|| = {na:e}, \
                 worst |fused - exact| = {worst_fx:e}, worst |a_gram - grams| = {worst_pair:e}, \
                 ||G - G^T|| = {asym:e}"
            );
            assert!(
                worst_fx <= 1e-11 * scale,
                "[{label}] 2k={two_k}: the FUSED A Gram differs from the exact one by \
                 {worst_fx:e} (scale {scale:e}) — above the reassociation class, so it is a \
                 different operator"
            );
            assert!(
                worst_pair <= 1e-12 * scale,
                "[{label}] 2k={two_k}: a_gram and grams().0 disagree by {worst_pair:e} — one of \
                 the two shifted A drivers is not on the fused route"
            );
            assert!(
                asym <= 1e-11 * na.max(1.0),
                "[{label}] 2k={two_k}: the FUSED A^(S') Gram is not symmetric ({asym:e} at \
                 ||A|| = {na:e}) — the H9-a signature (a plain adjoint where the theta^2-weighted \
                 rproject belongs, or the two mixed folds' bra/ket roles swapped), which no \
                 eigenvalue comparison against a symmetrized reference sees"
            );
        }
    }
    assert!(
        checked >= 2,
        "ANTI-VACUITY: only {checked} point-group channels were reached (measured inventory: \
         smallest 2k in {{2, 4}}, four-plane 2k in {{2, 4}})"
    );
}

fn feasible_fluxes(space: &Su2Space) -> Vec<usize> {
    [TRIPLET, QUINTET]
        .into_iter()
        .filter(|&k| k <= space.n.min(2 * space.d - space.n) && (space.n + k).is_multiple_of(2))
        .collect()
}

#[test]
fn csr_shifted_diagonal_is_the_embedded_rayleigh_quotient() {
    #[allow(clippy::type_complexity)]
    let mut cases: Vec<(
        String,
        Su2Space,
        CoeffTensor<f64>,
        CoeffTensor<f64>,
        Su2SparseHamiltonian<f64>,
        Su2TtVector<f64>,
    )> = Vec::new();
    for &(d, nel, seed) in &[
        (2usize, 2usize, 0xC5_D2u64),
        (3, 2, 0xC5_D3),
        (3, 4, 0xC5_D4),
    ] {
        let (t, v) = generic_truncated_tv(d);
        let space = Su2Space::new(d, nel, 0);
        let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: Su2TtVector<f64> = su2_tt_randn(space, &capped_grids(&space, 0), &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        assert!(x.check());
        cases.push((format!("max-ranks d={d} N={nel}"), space, t, v, ham, x));
    }
    for (cap, seed) in [(2usize, 0xC6_80u64), (1, 0xC6_81)] {
        let (space, t, v, ham, x) = truncated_reference(cap, seed);
        cases.push((format!("truncated cap {cap}"), space, t, v, ham, x));
    }
    let (mut checked, mut far) = (0usize, 0usize);
    let (mut first_checked, mut last_checked) = (0usize, 0usize);
    for (label, space, t, v, ham, x) in &cases {
        let d = space.d;
        let (bl, _br) = su2_tangent_bases(x);
        let ctx = su2_weingarten_operator(ham, x);
        for two_k in feasible_fluxes(space) {
            if tangent_dim_grids(space, &bl.ranks, two_k) == 0 {
                println!("csr K4 rayleigh [{label}] 2κ={two_k}: empty");
                continue;
            }
            let h_out =
                Su2SparseHamiltonian::new(space.with_two_s(two_k), t, v, None, f64::EPSILON);
            let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
            let lambda = sctx.lambda();
            let diag = sctx.a_diagonal();
            assert_eq!(diag.two_s_flux(), two_k);
            let counts: Vec<usize> = (0..d).map(|k| diag.component(k).num_coeffs()).collect();
            let mut start = vec![0usize; d + 1];
            for k in 0..d {
                start[k + 1] = start[k] + counts[k];
            }
            let mut seen = vec![false; d];
            let mut worst = 0.0f64;
            for i in 0..diag.num_coeffs() {
                let k = (0..d)
                    .find(|&k| i < start[k + 1])
                    .expect("coefficient owner");
                let mut e = diag.scale(0.0);
                e.set_coeff(i, 1.0);
                let y = e.to_vector();
                assert_eq!(
                    y.space, h_out.space,
                    "[{label}] 2κ={two_k}: iota(e_i) in space_out"
                );
                assert!(y.check());
                let yy = su2_dot(&y, &y);
                let en = e.norm();
                assert!(
                    (yy - en * en).abs() <= 1e-12 * yy.max(1.0),
                    "[{label}] 2κ={two_k} coeff {i}: isometry premise ({yy} vs {})",
                    en * en
                );
                let rq = iota_gram(&h_out, lambda, std::slice::from_ref(&e))[0] / yy;
                let got = diag.coeff(i);
                worst = worst.max((rq - got).abs() / (1.0 + rq.abs()));
                assert!(
                    (rq - got).abs() <= 1e-10 * (1.0 + rq.abs()),
                    "[{label}] 2κ={two_k} coeff {i} (k = {k}): a_diagonal {got} vs the embedded RQ {rq}"
                );
                if (yy - 1.0).abs() >= 0.1 {
                    far += 1;
                }
                seen[k] = true;
                checked += 1;
            }
            println!(
                "csr K4 rayleigh [{label}] 2κ={two_k}: per-component coefficients {counts:?}, worst \
                 rel {worst:.2e}"
            );
            assert!(
                (counts[0] == 0 || seen[0]) && (counts[d - 1] == 0 || seen[d - 1]),
                "[{label}] 2κ={two_k}: a live boundary component went unchecked ({counts:?})"
            );
            first_checked += usize::from(seen[0]);
            last_checked += usize::from(seen[d - 1]);
        }
    }
    println!(
        "csr K4 rayleigh: {checked} coefficients checked, {far} with |w − 1| ≥ 0.1; boundary \
         checked on {first_checked} (k = 0) / {last_checked} (k = d − 1) channels"
    );
    assert!(
        first_checked >= 1 && last_checked >= 1,
        "the SU(2) boundary is live: k = 0 and k = d − 1 must each be checked on some channel"
    );
    assert!(checked >= 8, "non-vacuity: {checked} coefficients checked");
    assert!(
        far >= 1,
        "discrimination premise: no coefficient carried a metric weight w ≠ 1"
    );
}

#[test]
fn csr_sector_lowest_d0_reaches_the_same_spectrum() {
    let (space, t, v, ham, x) = capped_singlet_minimizer(2);
    let ctx = su2_weingarten_operator(&ham, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let d0 = SympPrecond::DiagUnshifted { kappa: 0.0 };
    println!("csr D0: triplet dim {}", sctx.dim());
    for (readout, kernel, k) in [
        (SectorReadout::Rpa, RpaKernel::signed(RpaMetric::Ii), 3usize),
        (SectorReadout::Rpa, RpaKernel::Paired, 1),
        (SectorReadout::Tda, RpaKernel::Paired, 3),
    ] {
        let run = |precond: Option<SympPrecond>| {
            let mut opts = SectorOpts {
                readout,
                kernel,
                ..SectorOpts::default()
            };
            if let Some(p) = precond {
                opts.precond = p;
            }
            match readout {
                SectorReadout::Rpa => sctx.sector_lowest(k, &opts),
                SectorReadout::Tda => sctx.sector_lowest_tda(k, &opts),
            }
        };
        let vals = |r: &qntt::su2::hess::Su2SectorResponse| -> Vec<f64> {
            r.omega2()
                .or_else(|| r.theta().map(<[f64]>::to_vec))
                .expect("one arm")
                .to_vec()
        };
        let off = run(Some(SympPrecond::Off));
        let with = run(Some(d0));
        let dflt = run(None);
        let (wo, wd) = (vals(&off), vals(&with));
        println!(
            "csr D0 [{readout:?} {kernel:?} k {k}]: Off {wo:?} ({} iters, {} applies, converged {}) \
             | D0 {wd:?} ({} iters, {} applies, converged {})",
            off.iterations,
            off.applies,
            off.converged,
            with.iterations,
            with.applies,
            with.converged
        );
        assert!(
            off.converged && with.converged,
            "[{readout:?}] both runs converge"
        );
        assert_eq!(wo.len(), wd.len(), "[{readout:?}] the same number of roots");
        let scale = wo.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
        let tol = 10.0 * SectorOpts::default().tol * scale;
        for (j, (a, b)) in wo.iter().zip(&wd).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "[{readout:?}] root {j}: Off {a} vs D0 {b} (tol {tol:e})"
            );
        }
        assert!(
            crate::sector_util::flat_bits(&vals(&dflt)) == crate::sector_util::flat_bits(&wo)
                && (dflt.iterations, dflt.applies) == (off.iterations, off.applies)
                && crate::sector_util::flat_bits(&residuals_of(dflt.rpa()))
                    == crate::sector_util::flat_bits(&residuals_of(off.rpa())),
            "[{readout:?}] the default (Auto -> Off) must be the explicit Off run bit for bit"
        );
        assert!(
            crate::sector_util::flat_bits(&wd) != crate::sector_util::flat_bits(&wo)
                || (with.iterations, with.applies) != (off.iterations, off.applies),
            "[{readout:?}] non-vacuity (re-hosted P4 pins): the D0 run is bitwise the Off run — \
             the request was silently ignored"
        );
    }
}

#[test]
fn csr_sector_d0_survives_a_collapsed_tda_model() {
    let (space, t, v, _ham, x) = full_rank_singlet_ground();
    let neg = |c: &CoeffTensor<f64>, rank: usize| {
        let d = space.d;
        let mut out = CoeffTensor::<f64>::zeros(d, rank);
        let mut idx = vec![0usize; rank];
        loop {
            let val = c.get(&idx);
            if val != 0.0 {
                out.set(&idx, -val);
            }
            let mut p = 0;
            loop {
                if p == rank {
                    return out;
                }
                idx[p] += 1;
                if idx[p] < d {
                    break;
                }
                idx[p] = 0;
                p += 1;
            }
        }
    };
    let (tn, vn) = (neg(&t, 2), neg(&v, 4));
    let ham_n = Su2SparseHamiltonian::new(space, &tn, &vn, None, f64::EPSILON);
    let ctx = su2_weingarten_operator(&ham_n, &x);
    let h_out = Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &tn, &vn, None, f64::EPSILON);
    let sctx = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
    let dg = sctx.a_diagonal();
    let (dmin, dmax) = (0..dg.num_coeffs())
        .map(|i| dg.coeff(i))
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), z| {
            (lo.min(z), hi.max(z))
        });
    println!(
        "csr D0 collapsed: λ(−H) = {:.6}, dim {}, a_diagonal in [{dmin:.4e}, {dmax:.4e}]",
        sctx.lambda(),
        sctx.dim()
    );
    assert!(
        dmax <= 0.0,
        "premise: the TDA model is collapsed (max {dmax:e})"
    );
    assert!(dmin < 0.0, "premise: the RPA model (D_A)² is live");
    let k = 2usize;
    let d0 = SympPrecond::DiagUnshifted { kappa: 0.0 };
    let tda = |pc: SympPrecond| {
        sctx.sector_lowest_tda(
            k,
            &SectorOpts {
                readout: SectorReadout::Tda,
                precond: pc,
                ..SectorOpts::default()
            },
        )
    };
    let (to, tw) = (tda(SympPrecond::Off), tda(d0));
    println!(
        "csr D0 collapsed [Tda]: Off {:?} ({} iters) | D0 {:?} ({} iters)",
        to.theta(),
        to.iterations,
        tw.theta(),
        tw.iterations
    );
    assert!(
        crate::sector_util::flat_bits(to.theta().expect("Tda"))
            == crate::sector_util::flat_bits(tw.theta().expect("Tda"))
            && (to.iterations, to.applies, to.converged)
                == (tw.iterations, tw.applies, tw.converged),
        "[Tda] a collapsed model skips D0: bitwise the Off run"
    );
    for kernel in [RpaKernel::Paired, RpaKernel::signed(RpaMetric::Ii)] {
        let rpa = |pc: SympPrecond| {
            sctx.sector_lowest(
                k,
                &SectorOpts {
                    readout: SectorReadout::Rpa,
                    precond: pc,
                    kernel,
                    ..SectorOpts::default()
                },
            )
        };
        let (ro, rw) = (rpa(SympPrecond::Off), rpa(d0));
        let (wo, wd) = (
            ro.omega2().expect("Rpa").to_vec(),
            rw.omega2().expect("Rpa").to_vec(),
        );
        println!(
            "csr D0 collapsed [Rpa {kernel:?}]: Off {wo:?} ({} iters, conv {}) | D0 {wd:?} ({} \
             iters, conv {})",
            ro.iterations, ro.converged, rw.iterations, rw.converged
        );
        assert!(
            ro.converged && rw.converged,
            "[Rpa {kernel:?}] both runs converge"
        );
        assert_eq!(wo.len(), wd.len(), "[Rpa {kernel:?}] root count");
        let scale = wo.iter().fold(0.0f64, |m, z| m.max(z.abs())).max(1.0);
        let tol = 10.0 * SectorOpts::default().tol * scale;
        for (j, (a, b)) in wo.iter().zip(&wd).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "[Rpa {kernel:?}] root {j}: Off {a} vs D0 {b}"
            );
        }
        if matches!(kernel, RpaKernel::Paired) {
            assert!(
                crate::sector_util::flat_bits(&wo) == crate::sector_util::flat_bits(&wd)
                    && (ro.iterations, ro.applies) == (rw.iterations, rw.applies),
                "[Rpa paired] a collapsed D_A model skips D0: bitwise the Off run"
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Multiplet {
    b: usize,
    u: qntt::su2::qn::Su2Sector,
    sigma: f64,
    w_hat: f64,
}

#[allow(clippy::type_complexity)]
fn rule_f_candidates(
    ctx: &qntt::su2::hess::Su2WeingartenCtx<'_, f64>,
) -> (Vec<Multiplet>, Vec<(usize, f64, Vec<Multiplet>)>) {
    let mut m: Vec<Multiplet> = Vec::new();
    for g in ctx.bond_gauges() {
        for (u, vals) in &g.sigma {
            for &sigma in vals {
                m.push(Multiplet {
                    b: g.k,
                    u: *u,
                    sigma,
                    w_hat: qntt::su2::centerweight(u.two_s, 0).sqrt() * sigma,
                });
            }
        }
    }
    m.sort_by(|p, q| p.sigma.total_cmp(&q.sigma));
    let mut out = Vec::new();
    for j in 1..=4usize.min(m.len().saturating_sub(1)) {
        let (s0, s1) = (m[j - 1].sigma, m[j].sigma);
        if s1 >= s0 * (1.0 + 1e-3) {
            out.push((j, (s0 * s1).sqrt(), m[..j].to_vec()));
        }
    }
    (m, out)
}

#[test]
fn csr_d6_pg_curved_regularized_channels_are_the_expanded_abelian_blocks() {
    let mut plane_cuts: Vec<(String, usize, usize)> = Vec::new();
    let mut vacuous: Vec<&str> = Vec::new();
    let mut units_checked: Vec<&str> = Vec::new();
    for &(label, table, nel, g0, cap, seed, _dims, _ab, _planes) in &PG6_FIXTURES {
        let (space, t, v, ham, x) = pg6_reference(table, nel, g0, cap, seed);
        let (nonvacuous, cuts, units) = pg_regularized_leg(label, space, t, v, ham, x, true);
        if nonvacuous {
            plane_cuts.extend(cuts);
        } else {
            vacuous.push(label);
        }
        if units {
            units_checked.push(label);
        }
    }
    if !vacuous.is_empty() {
        println!(
            "csr pg6-reg: the Q6 fallback (curvature_reg) covers the vacuous PG6 legs {vacuous:?}"
        );
    }
    println!("csr pg6-reg: cut multiplets (fixture, bond, planes at the bond) {plane_cuts:?}");
    assert!(
        plane_cuts.iter().any(|&(_, _, p)| p >= 2),
        "COVERAGE: no cut multiplet sits on a bond carrying ≥ 2 irrep planes ({plane_cuts:?})"
    );
    println!("csr pg6-reg: the units anti-oracle ran on {units_checked:?}");
    assert!(
        units_checked.contains(&"smallest"),
        "COVERAGE: the units anti-oracle must run (measured on `smallest`; ran on \
         {units_checked:?})"
    );
}

#[allow(clippy::type_complexity)]
pub(crate) fn pg_regularized_leg(
    label: &str,
    space: Su2Space,
    t: CoeffTensor<f64>,
    v: CoeffTensor<f64>,
    ham: Su2SparseHamiltonian<f64>,
    x: Su2TtVector<f64>,
    allow_vacuous: bool,
) -> (bool, Vec<(String, usize, usize)>, bool) {
    use qntt::solvers::hess::CurvatureReg;
    let mut plane_cuts: Vec<(String, usize, usize)> = Vec::new();
    let (bl, br) = su2_tangent_bases(&x);
    let ctx_off = su2_weingarten_operator(&ham, &x);
    let nel = space.n;
    let channels: Vec<usize> = (2..=nel.min(2 * D6PG - nel)).step_by(2).collect();
    let h_outs: Vec<(usize, Su2SparseHamiltonian<f64>)> = channels
        .iter()
        .filter(|&&k| tangent_dim_grids(&space, &bl.ranks, k) > 0)
        .map(|&k| {
            (
                k,
                Su2SparseHamiltonian::new(space.with_two_s(k), &t, &v, None, f64::EPSILON),
            )
        })
        .collect();
    let (mults, cands) = rule_f_candidates(&ctx_off);
    let mut chosen = None;
    for (j, f, cut) in &cands {
        let ctx_try = su2_weingarten_operator(&ham, &x).with_curvature(CurvatureReg::Hard(*f));
        let mut moved_any = false;
        for (two_s, h_out) in &h_outs {
            let s_off = Su2SectorCtx::new(&ctx_off, h_out, *two_s);
            let basis = s_off.basis();
            let n = basis.len();
            if n == 0 {
                continue;
            }
            let (_, b_raw) = s_off.grams(&basis);
            let (_, b_reg) = Su2SectorCtx::new(&ctx_try, h_out, *two_s).grams(&basis);
            let raw = b_raw.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            let moved = b_raw
                .iter()
                .zip(&b_reg)
                .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
            println!(
                "csr pg6-reg [{label}] rule F J={j}: 2S'={two_s} max|B_raw| {raw:.3e} moved \
                 {moved:.3e}"
            );
            if raw > 1e-6 && moved >= 1e-6 * raw {
                moved_any = true;
            }
        }
        if moved_any {
            chosen = Some((*j, *f, cut.clone()));
            break;
        }
    }
    let nonvacuous = chosen.is_some();
    let chosen = chosen.or_else(|| {
        allow_vacuous
            .then(|| cands.last().map(|(j, f, cut)| (*j, *f, cut.clone())))
            .flatten()
    });
    let Some((j, f, cut)) = chosen else {
        panic!(
            "[{label}] RULE F found no non-vacuous cut at J ≤ 4 (candidates {:?}; {} \
             multiplets) — re-host on the planted public fixture of \
             tests/response/curvature_reg.rs (spec Q6)",
            cands.iter().map(|c| c.0).collect::<Vec<_>>(),
            mults.len()
        )
    };
    if !nonvacuous {
        println!(
            "csr pg6-reg [{label}]: RULE F found no non-vacuous cut at J ≤ 4 (candidates {:?}) — \
             the identity runs at the largest order candidate J = {j} (VACUOUS on every spin \
             channel); the spin-channel non-vacuity moves to the Q6 fallback",
            cands.iter().map(|c| c.0).collect::<Vec<_>>()
        );
    }
    let want_members: usize = cut.iter().map(|c| c.u.two_s + 1).sum();
    println!("csr pg6-reg [{label}]: J = {j}, f = {f:.4e} (raw σ, both layers), cut {cut:?}");
    for c in &cut {
        let planes = space
            .bond_sectors(c.b)
            .into_iter()
            .filter(|&s| bl.ranks[c.b][s] > 0)
            .map(|s| s.irrep.index())
            .collect::<std::collections::BTreeSet<u8>>()
            .len();
        plane_cuts.push((label.to_string(), c.b, planes));
    }
    let x_ab = expand(&x, 0);
    let h_ab = SparseHamiltonian::<f64>::new(x_ab.space, &t, &v, None, f64::EPSILON);
    let ctx_ab = ab_weingarten_operator(&h_ab, &x_ab).with_curvature(CurvatureReg::Hard(f));
    let ab_below = |floor: f64| -> usize {
        ctx_ab
            .bond_gauges()
            .iter()
            .flat_map(|g| g.sigma.iter().flat_map(|(_, s)| s.iter().copied()))
            .filter(|&s| s < floor)
            .count()
    };
    assert_eq!(
        ab_below(f),
        want_members,
        "[{label}] the same abelian floor cuts exactly the members of the SU(2) cut multiplets"
    );
    let k_w = mults.iter().filter(|m| m.w_hat < f).count();
    let ctx_ab_w = if k_w < j {
        assert!(
            mults[..k_w].iter().all(|m| m.w_hat < f),
            "[{label}] premise: the ŵ cut at f is a raw-σ prefix (one abelian floor realizes it)"
        );
        let f_w = if k_w == 0 {
            0.5 * mults[0].sigma
        } else {
            let (s0, s1) = (mults[k_w - 1].sigma, mults[k_w].sigma);
            assert!(
                s1 >= s0 * (1.0 + 1e-3),
                "[{label}] premise: a relative raw-σ gap ≥ 1e-3 under the ŵ cut"
            );
            (s0 * s1).sqrt()
        };
        let members_w: usize = mults[..k_w].iter().map(|m| m.u.two_s + 1).sum();
        assert_eq!(
            ab_below(f_w),
            members_w,
            "[{label}] premise: the abelian floor f_w cuts exactly the members of {{ŵ < f}}"
        );
        println!(
            "csr pg6-reg [{label}]: the units differ at f — ŵ cuts {k_w} of the {j} multiplets; \
             the ŵ-paired abelian floor f_w = {f_w:.4e} cuts {members_w} of {want_members} members"
        );
        Some(ab_weingarten_operator(&h_ab, &x_ab).with_curvature(CurvatureReg::Hard(f_w)))
    } else {
        println!(
            "csr pg6-reg [{label}]: the units agree at f (ŵ < f on every cut multiplet) — no units \
             anti-oracle here"
        );
        None
    };
    let mut units_miss = 0.0f64;
    let ctx = su2_weingarten_operator(&ham, &x).with_curvature(CurvatureReg::Hard(f));
    let (bl_ab, br_ab) = ctx_ab.bases();
    let basis_ab = ab_tangent_basis(bl_ab, br_ab, true);
    let n_ab = basis_ab.len();
    let (a_ab, b_ab) = ab_hessian_grams(&ctx_ab, &basis_ab);
    let (hrr_ab, hii_ab) = hrr_hii(&a_ab, &b_ab);
    let basis_0 = su2_tangent_basis(&bl, &br, true);
    let n_0 = basis_0.len();
    let (a_0, b_0) = su2_hessian_grams(&ctx, &basis_0);
    let (hrr_0, hii_0) = hrr_hii(&a_0, &b_0);
    let mut union_hrr = gram_eigs(&hrr_0, n_0);
    let mut union_hii = gram_eigs(&hii_0, n_0);
    let mut union_gen = generator_eigs(&a_0, &b_0, n_0);
    let embed = |u: &Su2TtTangent<f64>, what: &str| -> TtTangent<f64> {
        let y = expand(&u.to_vector(), 0);
        let e = TtTangent::project(bl_ab, br_ab, &y);
        assert!(
            (e.norm() - u.norm()).abs() <= 1e-12 * u.norm().max(1.0),
            "[{label}] {what}: expand∘iota is not isometric"
        );
        let defect = (&e.to_vector() - &y).norm_lr(false);
        assert!(
            defect <= 1e-10 * y.norm_lr(false).max(1.0),
            "[{label}] {what}: expand(iota(xi), 0) is not already tangent ({defect:e})"
        );
        e
    };
    let s2_defect = |y: &TtVector<f64>, two_s: usize| -> f64 {
        let ss = (two_s * (two_s + 2)) as f64 / 4.0;
        let yy = ab_dot(y, y);
        (ab_dot(y, &s_squared(y)) - ss * yy).abs() / (ss * yy).max(1.0)
    };
    let mut probes: Vec<(usize, TtTangent<f64>)> = basis_0
        .iter()
        .take(3)
        .map(|u| (0usize, embed(u, "S'=0 probe")))
        .collect();
    #[allow(clippy::type_complexity)]
    let mut chan: Vec<(usize, Vec<Su2TtTangent<f64>>, Vec<TtTangent<f64>>)> = Vec::new();
    for (two_s, h_out) in &h_outs {
        let basis = Su2SectorCtx::new(&ctx_off, h_out, *two_s).basis();
        let emb: Vec<TtTangent<f64>> = basis
            .iter()
            .map(|u| embed(u, &format!("2S'={two_s}")))
            .collect();
        for e in emb.iter().take(3) {
            probes.push((*two_s, e.clone()));
        }
        chan.push((*two_s, basis, emb));
    }
    for ((two_s, basis, emb), (_, h_out)) in chan.iter().zip(&h_outs) {
        let two_s = *two_s;
        let n = basis.len();
        let s_off = Su2SectorCtx::new(&ctx_off, h_out, two_s);
        let sctx = Su2SectorCtx::new(&ctx, h_out, two_s);
        let (a_off, b_off) = s_off.grams(basis);
        let (a, b) = sctx.grams(basis);
        assert!(
            crate::sector_util::flat_bits(&a) == crate::sector_util::flat_bits(&a_off),
            "[{label}] 2S'={two_s}: A must be bitwise the Off channel's"
        );
        if let Some(c_w) = &ctx_ab_w {
            let raw = b_off.iter().fold(0.0f64, |m, z| m.max(z.abs()));
            let mut miss = 0.0f64;
            for (jj, ej) in emb.iter().enumerate() {
                let bw = c_w.b_apply(ej);
                for (i, ei) in emb.iter().enumerate() {
                    miss = miss.max((b[i * n + jj] - ei.dot(&bw)).abs());
                }
            }
            println!(
                "csr pg6-reg [{label}] 2S'={two_s}: the ŵ-paired abelian block misses B^S'_reg by \
                 {miss:.3e} (max|B_raw| {raw:.3e})"
            );
            if raw > 1e-6 {
                units_miss = units_miss.max(miss / raw);
            }
        }
        let mut b1 = vec![0.0f64; n * n];
        let (mut worst_inv, mut worst_s2, mut worst_cross) = (0.0f64, 0.0f64, 0.0f64);
        for (jj, ej) in emb.iter().enumerate() {
            let bv = ctx_ab.b_apply(ej);
            for (i, ei) in emb.iter().enumerate() {
                b1[i * n + jj] = ei.dot(&bv);
            }
            if jj < 3 {
                let mut resid = bv.clone();
                for (i, ei) in emb.iter().enumerate() {
                    resid.axpy(-b1[i * n + jj], ei);
                }
                worst_inv = worst_inv.max(resid.norm());
                for (s2, e) in &probes {
                    if *s2 != two_s {
                        worst_cross = worst_cross.max(e.dot(&bv).abs());
                    }
                }
            }
            if jj < 2 {
                worst_s2 = worst_s2
                    .max(s2_defect(&ej.to_vector(), two_s))
                    .max(s2_defect(&bv.to_vector(), two_s));
            }
        }
        let nb = gram_spectral_norm(&b, n);
        let worst_b = b
            .iter()
            .zip(&b1)
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        println!(
            "csr pg6-reg [{label}] 2S'={two_s}: n = {n}, ||B_reg|| = {nb:e}, max|B_su2 − B_ab| \
             {worst_b:e}, closure {worst_inv:e}, S² {worst_s2:e}, cross-channel {worst_cross:e}"
        );
        assert!(
            worst_b <= 1e-9 * nb.max(1.0),
            "[{label}] 2S'={two_s}: regularized B^S' vs the regularized abelian block {worst_b:e}"
        );
        assert!(
            worst_inv <= 1e-10 * nb.max(1.0),
            "[{label}] 2S'={two_s}: B_ab,reg maps the embedded channel out of itself {worst_inv:e}"
        );
        assert!(
            worst_s2 <= 1e-10,
            "[{label}] 2S'={two_s}: S² eigen-defect {worst_s2:e}"
        );
        assert!(
            worst_cross <= 1e-12 * nb.max(1.0),
            "[{label}] 2S'={two_s}: B_ab,reg couples spin channels {worst_cross:e}"
        );
        let spec = sctx.symplectic_eigenvalues(basis);
        let spec_ab = qntt::solvers::hess::symplectic_from_grams(&a, &b1, n, 1e-10);
        assert!(
            std::mem::discriminant(&spec) == std::mem::discriminant(&spec_ab),
            "[{label}] 2S'={two_s}: the layers route the same regularized pair differently \
             ({} vs {})",
            spectrum_label(&spec),
            spectrum_label(&spec_ab)
        );
        let (hrr, hii) = hrr_hii(&a, &b);
        union_hrr.extend(gram_eigs(&hrr, n));
        union_hii.extend(gram_eigs(&hii, n));
        union_gen.extend(generator_eigs(&a, &b, n));
    }
    if ctx_ab_w.is_some() {
        assert!(
            units_miss >= 1e-3,
            "[{label}] units anti-oracle: the ŵ-paired abelian floor reproduces every regularized \
             spin channel (worst miss {units_miss:e} of max|B_raw|)"
        );
    }
    assert_eq!(
        n_ab,
        union_gen.len() / 2,
        "[{label}] the union covers the abelian basis"
    );
    for (what, mut got, want) in [
        ("H^rr", union_hrr, gram_eigs(&hrr_ab, n_ab)),
        ("H^ii", union_hii, gram_eigs(&hii_ab, n_ab)),
    ] {
        got.sort_by(|p, q| p.partial_cmp(q).expect("finite eigenvalue"));
        let sc = want.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        assert!(sc > 1e-6, "[{label}] ANTI-VACUITY: {what} is zero");
        assert_multiset_eq(
            &got,
            &want,
            1e-8 * sc,
            &format!("[{label}] regularized union eig({what}) vs the regularized abelian"),
        );
    }
    let gen_ab = generator_eigs(&a_ab, &b_ab, n_ab);
    let gs = gen_ab.iter().fold(0.0f64, |m, z| m.max(z.norm()));
    assert!(
        gs > 1e-6,
        "[{label}] ANTI-VACUITY: the abelian generator spectrum is zero"
    );
    let worst = assert_complex_multiset_eq(
        &union_gen,
        &gen_ab,
        1e-8 * gs,
        &format!("[{label}] regularized union eig(M) vs the regularized abelian eig(M)"),
    );
    println!("csr pg6-reg [{label}]: union eig(M) worst {worst:e} (scale {gs:e})");
    (nonvacuous, plane_cuts, ctx_ab_w.is_some())
}

#[test]
fn csr_tda_full_rank_singlet_ground_gives_the_exact_singlet_gaps() {
    use qntt::solvers::hess::TdaDavidsonOpts;
    use qntt::su2::hess::tda_lowest_davidson;
    let (space, t, v, ham, x) = full_rank_singlet_ground();
    let ctx = su2_weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let dim = qntt::su2::weingarten::tangent_dim(bl);
    let (singlets, _, _) = spin_labelled_levels(space.d, &t, &v);
    assert_eq!(
        dim,
        singlets.len(),
        "fixture precondition: the manifold is the whole singlet subspace"
    );
    let lambda = su2_rayleigh(&ham, &x).re_s();
    assert!(
        (lambda - singlets[0]).abs() <= 1e-10,
        "fixture precondition: the reference is the singlet ground (λ {lambda} vs {})",
        singlets[0]
    );
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7DA8);
    let v0s: Vec<Su2TtTangent<f64>> = (0..3)
        .map(|_| Su2TtTangent::project(bl, br, &su2_tt_randn(x.space, &x.ranks, &mut rng)))
        .collect();
    let k = dim - 1;
    let out = tda_lowest_davidson(
        &ctx,
        &v0s,
        k,
        &TdaDavidsonOpts {
            tol: 1e-12,
            ..TdaDavidsonOpts::default()
        },
    );
    println!(
        "csr tda: θ {:?}\n    converged {} it {} applies {}",
        out.theta, out.info.converged, out.info.iterations, out.info.applies
    );
    assert!(out.info.converged);
    assert_eq!(out.theta.len(), k);
    for (j, th) in out.theta.iter().enumerate() {
        let gap = singlets[j + 1] - lambda;
        assert!(
            (th - gap).abs() <= 1e-9,
            "singlet level {}: θ {th} vs exact gap {gap}",
            j + 1
        );
    }
}

fn paired_su2_sector_leg(
    label: &str,
    sc: &Su2SectorCtx<'_, '_, f64>,
    k: usize,
    stable: bool,
    cross: bool,
) -> usize {
    use crate::sector_util::{dense_pencil, pencil_inertia};
    use qntt::solvers::hess::{RpaModeKind, RpaReport};
    let basis = sc.basis();
    let n = basis.len();
    let (a, b) = sc.grams(&basis);
    let dense = dense_pencil(&a, &b, n);
    let (want_r, want_c) = dense.wanted(k);
    let opts = |kernel: RpaKernel| SectorOpts {
        readout: SectorReadout::Rpa,
        tol: 1e-10,
        kernel,
        ..SectorOpts::default()
    };
    let (resp, amps) = sc.sector_lowest_vectors(k, &opts(RpaKernel::Paired));
    let r = resp.rpa().expect("an Rpa readout");
    let got_r = r.omega2();
    let got_c: Vec<(f64, f64)> = {
        let mut c: Vec<_> = r
            .complex()
            .map(|m| (m.omega2.re, m.omega2.im.abs()))
            .collect();
        c.sort_by(|p, q| p.0.total_cmp(&q.0));
        c
    };
    let rel = |g: f64, w: f64| (g - w).abs() / w.abs().max(f64::MIN_POSITIVE);
    let worst = got_r
        .iter()
        .zip(&want_r)
        .map(|(&g, &w)| rel(g, w))
        .chain(
            got_c
                .iter()
                .zip(&want_c)
                .flat_map(|(g, w)| [rel(g.0, w.0), rel(g.1, w.1)]),
        )
        .fold(0.0f64, f64::max);
    println!(
        "csr paired [{label}] k {k}: n {n}, dim {}, built {}, pairs {}, converged {}, omega2 \
         {got_r:?} (dense {want_r:?}), complex {got_c:?} (dense {want_c:?}), worst rel \
         {worst:.2e}, index {:?}",
        sc.dim(),
        resp.iterations,
        resp.applies,
        resp.converged,
        resp.index_lower_bound()
    );
    assert!(
        resp.converged,
        "[{label}] k {k}: the paired solve converges"
    );
    assert!(
        matches!(r.report, RpaReport::Paired { .. }),
        "[{label}] k {k}: the paired kernel ran ({:?})",
        r.report
    );
    assert_eq!(
        got_r.len(),
        want_r.len(),
        "[{label}] k {k}: real/imaginary mode count"
    );
    assert_eq!(
        got_c.len(),
        want_c.len(),
        "[{label}] k {k}: complex pair count"
    );
    assert!(
        worst <= 1e-8,
        "[{label}] k {k}: omega^2 off the dense pencil by {worst:e} (relative)"
    );
    assert!(
        r.modes
            .iter()
            .filter(|m| m.kind == RpaModeKind::Imaginary)
            .all(|m| m.certificate.is_some_and(|w| w.certified())),
        "[{label}] k {k}: every imaginary mode carries a certified witness"
    );
    if k >= sc.dim() {
        assert_eq!(
            resp.index_lower_bound(),
            Some(pencil_inertia(&a, &b, n).rhs()),
            "[{label}] k = dim: the projected inertia IS n-(A+B) + n-(A-B)"
        );
    }
    assert_eq!(
        resp.omega().is_some(),
        stable,
        "[{label}] k {k}: omega() is published on a Stable channel only"
    );
    let checks = sc.rpa_mode_checks(&resp, &amps, &ModeCheckOpts::default());
    assert_eq!(checks.len(), r.modes.len());
    for (j, c) in checks.iter().enumerate() {
        let res = r.modes[j].residual;
        if c.eta.is_finite() {
            assert!(c.eta <= 1e-6, "[{label}] k {k} mode {j}: eta {}", c.eta);
        }
        if c.residual_fresh.is_finite() {
            assert!(
                (c.residual_fresh - res).abs() <= 1e-9 + 1e-3 * res,
                "[{label}] k {k} mode {j}: fresh residual {} vs the readout's {res}",
                c.residual_fresh
            );
        }
    }
    assert!(
        checks
            .iter()
            .filter(|c| c.omega2 > 0.0)
            .all(|c| c.eta.is_finite() && c.residual_fresh.is_finite()),
        "[{label}] k {k}: every positive mode is checked"
    );
    if cross {
        let signed = sc.sector_lowest(k, &opts(RpaKernel::signed(RpaMetric::Ii)));
        let ws = signed.omega2().expect("Rpa");
        assert!(
            signed.converged,
            "[{label}] k {k}: the signed solve converges"
        );
        assert_eq!(ws.len(), got_r.len(), "[{label}] k {k}: cross-kernel count");
        for (j, (p, s)) in got_r.iter().zip(&ws).enumerate() {
            assert!(
                rel(*s, *p) <= 1e-8,
                "[{label}] k {k} mode {j}: paired {p} vs signed {s}"
            );
        }
    }
    got_r.len() + got_c.len()
}

#[test]
fn csr_paired_sector_readout_matches_the_dense_pencil() {
    let mut compared = 0usize;
    {
        let (space, t, v, ham, x) = capped_singlet_minimizer(2);
        let ctx = su2_weingarten_operator(&ham, &x);
        let h_out =
            Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
        let sc = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let spec = sc.symplectic_eigenvalues(&sc.basis());
        assert!(
            matches!(spec, SymplecticSpectrum::Stable { .. }),
            "premise: the cap-2 triplet is Stable, got {}",
            spectrum_label(&spec)
        );
        for k in [3, sc.dim()] {
            compared += paired_su2_sector_leg("cap-2 triplet", &sc, k, true, true);
        }
    }
    {
        let (space, t, v, ham, x) = truncated_reference(1, 0xC5_80);
        let ctx = su2_weingarten_operator(&ham, &x);
        let h_out =
            Su2SparseHamiltonian::new(space.with_two_s(TRIPLET), &t, &v, None, f64::EPSILON);
        let sc = Su2SectorCtx::new(&ctx, &h_out, TRIPLET);
        let spec = sc.symplectic_eigenvalues(&sc.basis());
        let label = format!("cap-1 random triplet {}", spectrum_label(&spec));
        assert!(
            matches!(spec, SymplecticSpectrum::Unstable { .. }),
            "[{label}] premise Unstable"
        );
        compared += paired_su2_sector_leg(&label, &sc, sc.dim(), false, false);
    }
    println!("csr paired: {compared} modes compared");
}
