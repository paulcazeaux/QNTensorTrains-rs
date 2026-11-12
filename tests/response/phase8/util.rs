// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::integrals::{IntegralsOptions, load_fixture_dir};
use qntt::round::RoundMode;
use qntt::solvers::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::{QnSpace, RankGrid, SparseHamiltonian, TtVector, dot, tt_state};
pub(crate) use qntt_testkit::gad_mol::grad_exact_at;
pub(crate) use qntt_testkit::generic_truncated_h;
pub(crate) use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

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

#[allow(clippy::needless_range_loop)]
pub(crate) fn sym_eigvals_jacobi(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _sweep in 0..200 {
        let mut off = 0.0f64;
        for i in 0..n {
            for j in i + 1..n {
                off += a[i][j] * a[i][j];
            }
        }
        if off < 1e-28 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-15 {
                    continue;
                }
                let theta = 0.5 * (a[q][q] - a[p][p]) / a[p][q];
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut v: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v
}

pub(crate) fn lih_sto3g_ghost_fixture() -> (
    QnSpace,
    SparseHamiltonian<f64>,
    TtVector<f64>,
    Xoshiro256PlusPlus,
) {
    use qntt::solvers::als::{AlsOptions, als};

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
    (space, ham, x, rng)
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

pub(crate) fn normal_project_ambient(
    bl: &TtVector<f64>,
    br: &TtVector<f64>,
    v: &TtVector<f64>,
) -> TtVector<f64> {
    let tang = TtTangent::project(bl, br, v).to_vector();
    let mut n = v - &tang;
    n.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    n
}

pub(crate) fn mgs_accept_ambient(
    refs: &[TtVector<f64>],
    v: TtVector<f64>,
) -> Option<TtVector<f64>> {
    let mut v = v;
    let n0 = v.norm();
    if n0 < 1e-10 {
        return None;
    }
    v.scale_mut(1.0 / n0);
    for _pass in 0..2 {
        for b in refs {
            let c = dot(b, &v);
            v = &v - &(b * c);
        }
        v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    }
    let nrm = v.norm();
    if nrm > 1e-6 {
        v.scale_mut(1.0 / nrm);
        Some(v)
    } else {
        None
    }
}
