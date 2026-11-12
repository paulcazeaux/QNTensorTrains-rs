// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::{TtTangent, projective_transport};
use qntt::integrals::Integrals;
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::rotation::{expm_antisym, orbital_gradient, rotate_integrals};
use qntt::round::RoundMode;
use qntt::solvers::InnerProductVector;
use qntt::solvers::joint::{JointContext, JointVector};
use qntt::solvers::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::vector::apply_group_element;
use qntt::{
    CoeffTensor, Irrep, PointGroup, QnSpace, RankGrid, SparseHamiltonian, TtVector, dot,
    rayleigh_quotient, tt_state,
};
use qntt::{max_ranks, tt_randn};
pub(crate) use qntt_testkit::dense_apply_string;
pub(crate) use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::path::Path;

pub(crate) use qntt_testkit::dense::*;
pub(crate) use qntt_testkit::gad_mol::grad_exact_at;
pub(crate) use qntt_testkit::generic_truncated_h;
pub(crate) use qntt_testkit::generic_truncated_tv;
pub(crate) use qntt_testkit::pg::capped_ranks;

pub(crate) fn hess_fd_central(
    ham: &SparseHamiltonian<f64>,
    x: &TtVector<f64>,
    xi: &TtTangent<f64>,
    h: f64,
) -> TtTangent<f64> {
    let mut xp = xi.retract_hosvd(h);
    normalize_mut(&mut xp);
    let mut xm = xi.retract_hosvd(-h);
    normalize_mut(&mut xm);
    let gp = projective_transport(x, &grad_exact_at(ham, &xp));
    let gm = projective_transport(x, &grad_exact_at(ham, &xm));
    gp.sub(&gm).scale(1.0 / (2.0 * h))
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
        "fixture not a Riemannian critical point (‖grad‖={gnorm:.3e})"
    );
    let (base_l, base_r) = tangent_bases(&x);
    (ham, x, base_l, base_r)
}

pub(crate) fn pg_symmetric_truncated_tv(
    d: usize,
    pg: PointGroup,
) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let (mut t, mut v) = generic_truncated_tv(d);
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                t.set(&[p, q], 0.0);
            } else if p != q {
                t.set(&[p, q], -0.8 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                        v.set(&[i, j, k, l], 0.0);
                    }
                }
            }
        }
    }
    (t, v)
}

pub(crate) fn weingarten_minimizer_pg(
    space: QnSpace,
    pg: PointGroup,
    seed: u64,
    lo: usize,
    hi: usize,
) -> (
    SparseHamiltonian<f64>,
    TtVector<f64>,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
) {
    let (t, v) = pg_symmetric_truncated_tv(space.d, pg);
    let ham = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x_rand: TtVector<f64> = randomized_state(space, lo, hi, &mut rng);
    let mut acc: Option<TtVector<f64>> = None;
    for h in 0..pg.order() {
        let term = apply_group_element(&x_rand, pg, Irrep::new(h));
        acc = Some(match acc {
            None => term,
            Some(a) => &a + &term,
        });
    }
    let mut x0 = acc.expect("pg.order() >= 1");
    normalize_mut(&mut x0);

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
    x.check();
    let gnorm = grad_exact_at(&ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "pg fixture not a Riemannian critical point (‖grad‖={gnorm:.3e})"
    );
    (ham, x, t, v)
}

pub(crate) fn noncritical_point(
    xstar: &TtVector<f64>,
    bl_star: &TtVector<f64>,
    br_star: &TtVector<f64>,
    alpha: f64,
    seed: u64,
) -> TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let dir: TtVector<f64> = randomized_state(xstar.space, 1, 2, &mut rng);
    let mut xi = TtTangent::project(bl_star, br_star, &dir);
    let n = xi.norm();
    xi.scale_mut(alpha / n);
    let mut xp = xi.retract_hosvd(1.0);
    normalize_mut(&mut xp);
    xp.check();
    xp
}

pub(crate) fn lih_cation_rank1_fixture() -> (
    QnSpace,
    SparseHamiltonian<f64>,
    TtVector<f64>,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
) {
    use qntt::hamiltonian::assembly::Construction;
    use qntt::integrals::{IntegralsOptions, load_fixture_dir};
    use qntt::solvers::als::{AlsOptions, als};
    use qntt::solvers::hess::{NewtonPolishOpts, newton_polish};
    use qntt::vector::perturbation;

    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let d = ints.norb;
    let (nup, ndn) = (2usize, 1usize);
    let space = QnSpace::new(d, nup, ndn);
    let ham = SparseHamiltonian::new_with(
        space,
        &ints.t,
        &ints.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );

    let rank = 1usize;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x10E4);
    let occ_up: Vec<bool> = (0..d).map(|i| i < nup).collect();
    let occ_dn: Vec<bool> = (0..d).map(|i| i < ndn).collect();
    let hf: TtVector<f64> = tt_state(&occ_up, &occ_dn);
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
    if let Some(x) = LIH_CATION_X.lock().unwrap().clone() {
        x.check();
        let gnorm = grad_exact_at(&ham, &x).norm();
        assert!(
            gnorm < 1e-9,
            "LiH+ fixture not TT-critical (‖grad‖={gnorm:.3e})"
        );
        return (space, ham, x, ints.t, ints.v);
    }
    let mut x0 = perturbation(&hf, &grid, 0.2, &mut rng);
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
    let (x, _info) = newton_polish(
        &ham,
        &x,
        &NewtonPolishOpts {
            gradtol: 1e-11,
            ..Default::default()
        },
    );
    x.check();
    let gnorm = grad_exact_at(&ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "LiH+ fixture not TT-critical (‖grad‖={gnorm:.3e})"
    );
    *LIH_CATION_X.lock().unwrap() = Some(x.clone());
    (space, ham, x, ints.t, ints.v)
}

static LIH_CATION_X: std::sync::Mutex<Option<TtVector<f64>>> = std::sync::Mutex::new(None);

pub(crate) fn dense_gpq(
    d: usize,
    p: usize,
    q: usize,
    dets: &[(Vec<bool>, Vec<bool>)],
) -> Vec<Vec<f64>> {
    let epq = dense_epq_matrix(d, p, q, dets);
    let eqp = dense_epq_matrix(d, q, p, dets);
    mat_add(&epq, &mat_scale(&eqp, -1.0))
}

pub(crate) fn dense_state_vector(dets: &[(Vec<bool>, Vec<bool>)], x: &TtVector<f64>) -> Vec<f64> {
    dets.iter()
        .map(|(ou, od)| {
            let s: TtVector<f64> = tt_state(ou, od);
            dot(x, &s)
        })
        .collect()
}

pub(crate) fn joint_energy_dense(
    hd: &[Vec<f64>],
    c: &[f64],
    kd: &[Vec<f64>],
    zeta: &[f64],
    s: f64,
) -> f64 {
    let expo = mat_expm(&mat_scale(kd, -s));
    let disp: Vec<f64> = c.iter().zip(zeta).map(|(&ci, &zi)| ci + s * zi).collect();
    let psi = mat_vec(&expo, &disp);
    let hpsi = mat_vec(hd, &psi);
    dot_vec(&psi, &hpsi) / dot_vec(&psi, &psi)
}

pub(crate) fn q2_dense(
    hd: &[Vec<f64>],
    c: &[f64],
    kd: &[Vec<f64>],
    zeta: &[f64],
    e0: f64,
    h: f64,
) -> f64 {
    let ep2 = joint_energy_dense(hd, c, kd, zeta, 2.0 * h);
    let ep1 = joint_energy_dense(hd, c, kd, zeta, h);
    let em1 = joint_energy_dense(hd, c, kd, zeta, -h);
    let em2 = joint_energy_dense(hd, c, kd, zeta, -2.0 * h);
    (-ep2 + 16.0 * ep1 - 30.0 * e0 + 16.0 * em1 - em2) / (24.0 * h * h)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn q2_polarized(
    hd: &[Vec<f64>],
    c: &[f64],
    kd_i: &[Vec<f64>],
    zeta_i: &[f64],
    kd_j: &[Vec<f64>],
    zeta_j: &[f64],
    e0: f64,
    h: f64,
) -> f64 {
    let kd_sum = mat_add(kd_i, kd_j);
    let zeta_sum: Vec<f64> = zeta_i.iter().zip(zeta_j).map(|(&a, &b)| a + b).collect();
    let q_sum = q2_dense(hd, c, &kd_sum, &zeta_sum, e0, h);
    let q_i = q2_dense(hd, c, kd_i, zeta_i, e0, h);
    let q_j = q2_dense(hd, c, kd_j, zeta_j, e0, h);
    0.5 * (q_sum - q_i - q_j)
}

pub(crate) fn dense_a_entry(vi: &[f64], vj: &[f64], hd: &[Vec<f64>], lambda: f64) -> f64 {
    let hvj = mat_vec(hd, vj);
    dot_vec(vi, &hvj) - lambda * dot_vec(vi, vj)
}

pub(crate) fn dense_b_orb(zd: &[f64], ga: &[Vec<f64>], gb: &[Vec<f64>], c: &[f64]) -> f64 {
    let gbc = mat_vec(gb, c);
    let gagbc = mat_vec(ga, &gbc);
    let gac = mat_vec(ga, c);
    let gbgac = mat_vec(gb, &gac);
    let sum: Vec<f64> = gagbc.iter().zip(&gbgac).map(|(&a, &b)| a + b).collect();
    0.5 * dot_vec(zd, &sum)
}

pub(crate) fn dense_b_cross(zd: &[f64], ga: &[Vec<f64>], zeta_i: &[f64]) -> f64 {
    let g_zeta = mat_vec(ga, zeta_i);
    -dot_vec(zd, &g_zeta)
}

pub(crate) fn assert_c1_selfadjoint(jctx: &JointContext<'_>, vecs: &[JointVector]) {
    let mut worst_a = 0.0f64;
    let mut scale_a = 0.0f64;
    let mut worst_b = 0.0f64;
    let mut scale_b = 0.0f64;
    for i in 0..vecs.len() {
        for j in (i + 1)..vecs.len() {
            let aij = vecs[i].dot(&jctx.a_apply(&vecs[j]));
            let aji = vecs[j].dot(&jctx.a_apply(&vecs[i]));
            worst_a = worst_a.max((aij - aji).abs());
            scale_a = scale_a.max(aij.abs()).max(aji.abs());
            let bij = vecs[i].dot(&jctx.b_apply(&vecs[j]));
            let bji = vecs[j].dot(&jctx.b_apply(&vecs[i]));
            worst_b = worst_b.max((bij - bji).abs());
            scale_b = scale_b.max(bij.abs()).max(bji.abs());
        }
    }
    assert!(
        scale_b > 1e-8,
        "B vacuous in self-adjointness arm: scale {scale_b:.3e}"
    );
    assert!(
        worst_a <= 1e-9 * scale_a.max(1.0),
        "A not self-adjoint in the joint dot: {worst_a:.3e}"
    );
    assert!(
        worst_b <= 1e-8 * scale_b.max(1.0),
        "B not self-adjoint in the joint dot: {worst_b:.3e}"
    );
}

pub(crate) fn coeff2_flat(c: &CoeffTensor<f64>, d: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            out[i * d + j] = c.get(&[i, j]);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_covector_fd_central(
    t_cur: &CoeffTensor<f64>,
    v_cur: &CoeffTensor<f64>,
    space: QnSpace,
    pairs: &[(usize, usize)],
    a_dir: &[f64],
    x_base: &TtVector<f64>,
    xi_dir: &TtTangent<f64>,
    h: f64,
) -> (TtTangent<f64>, Vec<f64>) {
    let d = space.d;
    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    for (&(p, q), &ai) in pairs.iter().zip(a_dir) {
        k.set(&[p, q], ai);
        k.set(&[q, p], -ai);
    }
    let kflat = coeff2_flat(&k, d);

    let one = |s: f64| -> (TtTangent<f64>, Vec<f64>) {
        let step = scale_flat(&kflat, -s);
        let u = expm_antisym(&step, d);
        let ints_s = Integrals::<f64> {
            t: t_cur.clone(),
            v: v_cur.clone(),
            e0: 0.0,
            norb: d,
            nelec: space.n(),
            ms2: space.nup as i64 - space.ndn as i64,
            orbsym: Vec::new(),
            isym: None,
        };
        let rotated = rotate_integrals(&ints_s, &u);
        let hs = SparseHamiltonian::new(space, &rotated.t, &rotated.v, None, f64::EPSILON);
        let mut xs = xi_dir.retract_hosvd(s);
        normalize_mut(&mut xs);
        let lam_s = rayleigh_quotient(&hs, &xs);
        let hx = hs.apply(&xs);
        let res = &hx - &(&xs * lam_s);
        let (bl_s, br_s) = tangent_bases(&xs);
        let g_own = TtTangent::project(&bl_s, &br_s, &res);
        let g_xi = projective_transport(x_base, &g_own);
        let g = orbital_gradient(&hs, &xs);
        let gc: Vec<f64> = pairs.iter().map(|&(p, q)| 0.5 * g.get(&[p, q])).collect();
        (g_xi, gc)
    };

    let (mut gxi_p, gc_p) = one(h);
    let (gxi_m, gc_m) = one(-h);
    gxi_p.axpy(-1.0, &gxi_m);
    gxi_p.scale_mut(1.0 / (2.0 * h));
    let gc: Vec<f64> = gc_p
        .iter()
        .zip(&gc_m)
        .map(|(&p, &m)| (p - m) / (2.0 * h))
        .collect();
    (gxi_p, gc)
}

pub(crate) fn load_h2o() -> qntt::IntegralsLoad<f64> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h2o_c2v");
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    load_fixture_dir::<f64>(&dir, &opts).expect("h2o_c2v fixture loads")
}

pub(crate) fn capped_randn(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
    use rand::RngExt;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let d = space.d;
    let caps = max_ranks(&space);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                let draw = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
                r[s] = draw.min(caps[b][s]);
            }
            r
        })
        .collect();
    tt_randn(space, &ranks, &mut rng)
}

pub(crate) fn panic_message(f: impl FnOnce()) -> String {
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    let payload = res.expect_err("expected a panic");
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .expect("a string panic payload")
}
