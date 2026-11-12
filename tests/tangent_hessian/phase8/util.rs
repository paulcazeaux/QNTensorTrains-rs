// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

pub(crate) use crate::shared::sym_eigvals_jacobi;
use qntt::core::tangent::{TtTangent, projective_transport};
use qntt::integrals::{IntegralsOptions, load_fixture_dir};
use qntt::round::RoundMode;
use qntt::solvers::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::{
    C64, CoeffTensor, QnSpace, RankGrid, Scalar, SparseHamiltonian, TtVector, dot, tt_state, xt_h_y,
};
pub(crate) use qntt_testkit::gad_mol::grad_exact_at;
pub(crate) use qntt_testkit::generic_complex_h;
pub(crate) use qntt_testkit::generic_truncated_h;
pub(crate) use qntt_testkit::generic_truncated_tv;
pub(crate) use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) fn state_from_sites<T: Scalar>(
    d: usize,
    up_sites: &[usize],
    dn_sites: &[usize],
) -> TtVector<T> {
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in up_sites {
        occ_up[s] = true;
    }
    for &s in dn_sites {
        occ_dn[s] = true;
    }
    tt_state(&occ_up, &occ_dn)
}

pub(crate) fn two_body_swap_h(
    space: QnSpace,
    a: usize,
    b: usize,
    c: usize,
    e: usize,
) -> SparseHamiltonian<f64> {
    let d = space.d;
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[c, a, e, b], -1.0);
    v.set(&[b, e, a, c], -1.0);
    SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON)
}

pub(crate) fn span_states(d: usize, a: usize, b: usize, c: usize, e: usize) -> Vec<TtVector<f64>> {
    let mut span: Vec<(usize, usize)> = vec![(a, b), (c, e), (b, a), (e, c)];
    span.sort_unstable();
    span.dedup();
    span.iter()
        .map(|&(u, w)| state_from_sites(d, &[u], &[w]))
        .collect()
}

#[allow(clippy::type_complexity)]
pub(crate) fn tangent_test_setup(
    seed: u64,
) -> (
    QnSpace,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
) {
    let d = 6usize;
    let space = QnSpace::new(d, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    let mut dir1: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    dir1.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    dir1.scale_mut(1.0 / dir1.norm());
    let mut dir2: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    dir2.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    dir2.scale_mut(1.0 / dir2.norm());
    (space, x, base_l, base_r, dir1, dir2)
}

pub(crate) fn hess_fd_central<T: Scalar>(
    ham: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    xi: &TtTangent<T>,
    h: f64,
) -> TtTangent<T> {
    let mut xp = xi.retract_hosvd(T::from_f64(h));
    normalize_mut(&mut xp);
    let mut xm = xi.retract_hosvd(T::from_f64(-h));
    normalize_mut(&mut xm);
    let gp = projective_transport(x, &grad_exact_at(ham, &xp));
    let gm = projective_transport(x, &grad_exact_at(ham, &xm));
    gp.sub(&gm).scale(T::from_f64(1.0 / (2.0 * h)))
}

pub(crate) fn weingarten_minimizer(
    space: QnSpace,
    seed: u64,
    lo: usize,
    hi: usize,
) -> (
    SparseHamiltonian<f64>,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    #[allow(clippy::type_complexity)]
    static CACHE: Mutex<Option<HashMap<(QnSpace, u64, usize, usize), TtVector<f64>>>> =
        Mutex::new(None);

    let ham = generic_truncated_h(space);
    let key = (space, seed, lo, hi);
    let cached: Option<TtVector<f64>> = CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .get(&key)
        .cloned();
    let x = match cached {
        Some(x) => x,
        None => {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let x0: TtVector<f64> = randomized_state(space, lo, hi, &mut rng);
            let mut opts = RiemannianOpts::new(&space);
            opts.eps = 1e-13;
            opts.max_iter = 600;
            opts.gradtol = 1e-13;
            let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
            let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
            for _ in 0..5 {
                if grad_exact_at(&ham, &x).norm() < 1e-11 {
                    break;
                }
                let (_l2, x2) = riemannian_cg(&ham, &x, &opts);
                x = x2;
            }
            normalize_mut(&mut x);
            CACHE
                .lock()
                .unwrap()
                .get_or_insert_with(HashMap::new)
                .insert(key, x.clone());
            x
        }
    };
    x.check();
    let gnorm = grad_exact_at(&ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "fixture not a Riemannian critical point (‖grad‖={gnorm:.3e}) — FD-of-gradient would be biased"
    );
    let (base_l, base_r) = tangent_bases(&x);
    (ham, x, base_l, base_r)
}

pub(crate) fn weingarten_minimizer_c64(
    ham: SparseHamiltonian<C64>,
    space: QnSpace,
    seed: u64,
    lo: usize,
    hi: usize,
) -> (
    SparseHamiltonian<C64>,
    TtVector<C64>,
    TtVector<C64>,
    TtVector<C64>,
) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x0: TtVector<C64> = randomized_state(space, lo, hi, &mut rng);
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let (_l, xg) = riemannian_gd(&ham, &x0, &opts);
    let (_l, mut x) = riemannian_cg(&ham, &xg, &opts);
    for _ in 0..8 {
        if grad_exact_at(&ham, &x).norm() < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(&ham, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    x.check();
    let gnorm = grad_exact_at(&ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "C64 fixture not a Riemannian critical point (‖grad‖={gnorm:.3e})"
    );
    let (base_l, base_r) = tangent_bases(&x);
    (ham, x, base_l, base_r)
}

pub(crate) fn lih_sto3g_ghost_fixture() -> (
    QnSpace,
    SparseHamiltonian<f64>,
    TtVector<f64>,
    Xoshiro256PlusPlus,
) {
    use qntt::solvers::als::{AlsOptions, als};
    use std::sync::{Mutex, OnceLock};

    #[allow(clippy::type_complexity)]
    static CACHE: OnceLock<Mutex<Option<(TtVector<f64>, Xoshiro256PlusPlus)>>> = OnceLock::new();
    let cached = CACHE
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap()
        .clone();

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let (d, nelec) = (ints.norb, ints.nelec);
    let (nup, ndn) = (nelec / 2, nelec / 2);
    let space = QnSpace::new(d, nup, ndn);
    let (t, v) = (ints.t, ints.v);
    let ham = SparseHamiltonian::new_with(
        space,
        &t,
        &v,
        None,
        f64::EPSILON,
        qntt::hamiltonian::assembly::Construction::Median,
    );

    if let Some((x, rng)) = cached {
        x.check();
        let gnorm = grad_exact_at(&ham, &x).norm();
        assert!(
            gnorm < 1e-9,
            "cached LiH fixture not critical (‖grad‖={gnorm:.3e})"
        );
        return (space, ham, x, rng);
    }

    let rank = 2usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x11A2);
    let occ: Vec<bool> = (0..d).map(|i| i < nup).collect();
    let hf: TtVector<f64> = tt_state(&occ, &occ);
    let grid: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut g = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                g[s] = if b == 0 || b == d {
                    1
                } else {
                    rank.min(space.max_rank(b, s) as usize)
                };
            }
            g
        })
        .collect();
    let mut x0 = qntt::vector::perturbation(&hf, &grid, 0.2, &mut rng);
    x0.round_ranks_mut(&grid);
    normalize_mut(&mut x0);
    let (_lals, xa) = als(
        &ham,
        &x0,
        &AlsOptions {
            eps: 1e-10,
            max_iter: 60,
            ..AlsOptions::default()
        },
    );
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    let polish = |y0: &TtVector<f64>| -> TtVector<f64> {
        let (_l, mut y) = riemannian_cg(&ham, y0, &opts);
        for _ in 0..30 {
            if grad_exact_at(&ham, &y).norm() < 1e-10 {
                break;
            }
            let (_l2, y2) = riemannian_cg(&ham, &y, &opts);
            y = y2;
        }
        y
    };
    let mut x = polish(&xa);
    normalize_mut(&mut x);
    x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    x = polish(&x);
    normalize_mut(&mut x);
    x.check();
    let gnorm = grad_exact_at(&ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "LiH fixture not critical (‖grad‖={gnorm:.3e})"
    );
    *CACHE.get_or_init(|| Mutex::new(None)).lock().unwrap() = Some((x.clone(), rng.clone()));
    (space, ham, x, rng)
}

pub(crate) fn rayleigh_tilde<T: Scalar>(ham: &SparseHamiltonian<T>, y: &TtVector<T>) -> f64 {
    xt_h_y(y, ham, y).re_s() / (2.0 * dot(y, y).re_s())
}

pub(crate) fn radial_mode_at<T: Scalar>(
    bl: &TtVector<T>,
    br: &TtVector<T>,
    x: &TtVector<T>,
) -> TtTangent<T> {
    let mut r = TtTangent::project(bl, br, x);
    let n = r.norm();
    r.scale_mut(T::from_f64(1.0 / n));
    r
}

pub(crate) fn deflate_radial_span<T: Scalar>(radial: &TtTangent<T>, v: &mut TtTangent<T>) {
    for _ in 0..2 {
        let c = radial.dot(v);
        v.axpy(-c, radial);
    }
}

pub(crate) fn horizontal_unit_tangent<T: Scalar>(
    space: QnSpace,
    bl: &TtVector<T>,
    br: &TtVector<T>,
    radial: &TtTangent<T>,
    rng: &mut Xoshiro256PlusPlus,
) -> TtTangent<T> {
    let dir: TtVector<T> = randomized_state(space, 1, 2, rng);
    let mut xi = TtTangent::project(bl, br, &dir);
    deflate_radial_span(radial, &mut xi);
    let n = xi.norm();
    xi.scale_mut(T::from_f64(1.0 / n));
    xi
}

pub(crate) fn corrected_form<T: Scalar>(
    ham: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    g_vec: &TtVector<T>,
    xi: &TtTangent<T>,
    h: f64,
) -> (f64, f64, f64) {
    let cp = xi.retract_hosvd(T::from_f64(h));
    let cm = xi.retract_hosvd(T::from_f64(-h));
    let m0 = rayleigh_tilde(ham, x);
    let mp = rayleigh_tilde(ham, &cp);
    let mm = rayleigh_tilde(ham, &cm);
    let q_uncorr = (mp - 2.0 * m0 + mm) / (h * h);
    let gp = dot(g_vec, &cp).re_s();
    let gm = dot(g_vec, &cm).re_s();
    let g0 = dot(g_vec, x).re_s();
    let corr = (gp + gm - 2.0 * g0) / (h * h);
    (q_uncorr - corr, q_uncorr, corr)
}

pub(crate) fn corrected_form_richardson<T: Scalar>(
    ham: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    g_vec: &TtVector<T>,
    xi: &TtTangent<T>,
    h: f64,
) -> f64 {
    let (q1, _, _) = corrected_form(ham, x, g_vec, xi, h);
    let (q2, _, _) = corrected_form(ham, x, g_vec, xi, h / 2.0);
    (4.0 * q2 - q1) / 3.0
}

pub(crate) fn noncritical_point<T: Scalar>(
    xstar: &TtVector<T>,
    bl_star: &TtVector<T>,
    br_star: &TtVector<T>,
    alpha: f64,
    seed: u64,
) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let dir: TtVector<T> = randomized_state(xstar.space, 1, 2, &mut rng);
    let mut xi = TtTangent::project(bl_star, br_star, &dir);
    let n = xi.norm();
    xi.scale_mut(T::from_f64(alpha / n));
    let mut xp = xi.retract_hosvd(T::from_f64(1.0));
    normalize_mut(&mut xp);
    xp.check();
    xp
}

pub(crate) fn seed_block<T: Scalar>(
    space: QnSpace,
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    b: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<TtTangent<T>> {
    (0..b)
        .map(|_| {
            let dir: TtVector<T> = randomized_state(space, 1, 2, rng);
            TtTangent::project(base_l, base_r, &dir)
        })
        .collect()
}

pub(crate) fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
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

pub(crate) fn exact_config_spectrum(space: QnSpace, ham: &SparseHamiltonian<f64>) -> Vec<f64> {
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
            hmat[i][j] = dot(si, &hs);
        }
    }
    sym_eigvals_jacobi(hmat)
}

pub(crate) fn exact_config_gaps_d3_11(ham: &SparseHamiltonian<f64>) -> Vec<f64> {
    let d = 3usize;
    let mut states: Vec<TtVector<f64>> = Vec::new();
    for up in 0..d {
        for dn in 0..d {
            states.push(state_from_sites(d, &[up], &[dn]));
        }
    }
    let nd = states.len();
    let mut hmat = vec![vec![0.0f64; nd]; nd];
    for (j, sj) in states.iter().enumerate() {
        let hs = ham * sj;
        for (i, si) in states.iter().enumerate() {
            hmat[i][j] = qntt::dot(si, &hs);
        }
    }
    let evals = sym_eigvals_jacobi(hmat);
    evals[1..].iter().map(|e| e - evals[0]).collect()
}
