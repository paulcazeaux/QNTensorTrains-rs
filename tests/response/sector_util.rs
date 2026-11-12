// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use qntt::core::Family;
use qntt::core::tangent::{TtTangent, project_irrep, tangent_basis, tangent_dim_grids};
use qntt::fcidump::screen_point_group_forbidden;
use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::{IntegralsLoad, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::pointgroup::{Irrep, PointGroup};
use qntt::round::RoundMode;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::hess::{
    CurvatureReg, SymplecticSpectrum, WeingartenCtx, hessian_grams, normal_residual,
    weingarten_operator,
};
use qntt::solvers::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::{
    CoeffTensor, QnSpace, RankGrid, Scalar, SparseHamiltonian, TtVector, max_ranks,
    rayleigh_quotient, tt_state_in,
};
use qntt_testkit::dense::{dense_eigen, dense_hamiltonian_matrix, dets_for, flat_to_nested};
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) const D4: usize = 4;

pub(crate) const F_NUP: usize = 2;
pub(crate) const F_NDN: usize = 1;

pub(crate) const T_NUP: usize = 2;
pub(crate) const T_NDN: usize = 2;

pub(crate) fn synth_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(1), Irrep::new(0)],
    )
}

pub(crate) fn fx4_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
    )
}

pub(crate) fn synth_tv(pg: PointGroup) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
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

pub(crate) fn fx4_tv(pg: PointGroup) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let d = D4;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        t.set(&[p, p], 0.17 * (p as f64 - 0.5 * d as f64));
    }
    t.set(&[1, 3], -0.85);
    t.set(&[3, 1], -0.85);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
                v.set(&[p, q, q, p], 0.23 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    for idx in [[1, 3, 0, 0], [3, 1, 0, 0], [0, 0, 1, 3], [0, 0, 3, 1]] {
        v.set(&idx, 0.19);
    }
    screen_point_group_forbidden(&mut t, &mut v, pg)
        .expect("the fixture is symmetry-allowed by construction — the screen must be a no-op");
    (t, v)
}

pub(crate) const D6: usize = 6;

pub(crate) fn d6_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
        ],
    )
}

pub(crate) fn d6_tv(pg: PointGroup) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let d = D6;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        t.set(&[p, p], 0.13 * (p as f64 - 0.5 * d as f64));
    }
    for (p, q) in [(0usize, 3usize), (1, 4), (2, 5)] {
        let h = -0.8 + 0.1 * p as f64;
        t.set(&[p, q], h);
        t.set(&[q, p], h);
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
    for idx in [[0, 3, 1, 4], [1, 4, 0, 3], [3, 0, 4, 1], [4, 1, 3, 0]] {
        v.set(&idx, 0.19);
    }
    screen_point_group_forbidden(&mut t, &mut v, pg)
        .expect("the fixture is symmetry-allowed by construction — the screen must be a no-op");
    (t, v)
}

pub(crate) type Fixture = (
    QnSpace,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    SparseHamiltonian<f64>,
    TtVector<f64>,
);

pub(crate) fn live_channels(space: &QnSpace, ranks: &[RankGrid]) -> Vec<Irrep> {
    (0..space.pg.order())
        .map(Irrep::new)
        .filter(|&phi| tangent_dim_grids(space, ranks, phi) > 0)
        .collect()
}

pub(crate) fn first_shifting_channel(space: &QnSpace, ranks: &[RankGrid]) -> Irrep {
    let live = live_channels(space, ranks);
    live.iter()
        .copied()
        .find(|&phi| phi != Irrep::IDENTITY)
        .unwrap_or_else(|| panic!("no live shifted channel at this base point (live = {live:?})"))
}

pub(crate) fn slot_sites(space: &QnSpace, ranks: &[RankGrid], phi: Irrep) -> Vec<usize> {
    let shift = |t: qntt::qn::Sector| qntt::qn::Sector::with_irrep(t.nup, t.ndn, t.irrep.fuse(phi));
    (0..space.d)
        .filter(|&k| {
            let g = space.pg.irrep_of(k);
            let mut raw = 0usize;
            for l in space.bond_sectors(k) {
                for f in Family::ALL {
                    let t = f.apply(l, g);
                    if space.contains(k + 1, t) {
                        raw += ranks[k][l] * ranks[k + 1][shift(t)];
                    }
                }
            }
            let gauge = if k < space.d - 1 {
                space
                    .bond_sectors(k + 1)
                    .into_iter()
                    .map(|t| ranks[k + 1][t] * ranks[k + 1][shift(t)])
                    .sum::<usize>()
            } else {
                0
            };
            raw > gauge
        })
        .collect()
}

pub(crate) fn curved_channels(space: &QnSpace, ranks: &[RankGrid]) -> Vec<Irrep> {
    live_channels(space, ranks)
        .into_iter()
        .filter(|&phi| phi != Irrep::IDENTITY && slot_sites(space, ranks, phi).len() >= 2)
        .collect()
}

pub(crate) fn first_curved_channel(space: &QnSpace, ranks: &[RankGrid]) -> Irrep {
    curved_channels(space, ranks)
        .first()
        .copied()
        .unwrap_or_else(|| {
            panic!(
                "no live shifted channel reaches two sites at this base point, so B^phi is \
             identically zero on every one of them (slot_sites per live channel: {:?}) — an arm \
             about B cannot run here",
                live_channels(space, ranks)
                    .iter()
                    .map(|&p| (p, slot_sites(space, ranks, p)))
                    .collect::<Vec<_>>()
            )
        })
}

pub(crate) fn dets_in_irrep(
    d: usize,
    nup: usize,
    ndn: usize,
    pg: PointGroup,
    target: Irrep,
) -> Vec<(Vec<bool>, Vec<bool>)> {
    dets_for(d, nup, ndn)
        .into_iter()
        .filter(|(ou, od)| {
            let mut ir = Irrep::IDENTITY;
            for (k, &o) in ou.iter().enumerate() {
                if o {
                    ir = ir.fuse(pg.irrep_of(k));
                }
            }
            for (k, &o) in od.iter().enumerate() {
                if o {
                    ir = ir.fuse(pg.irrep_of(k));
                }
            }
            ir == target
        })
        .collect()
}

pub(crate) fn dense_levels(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    pg: PointGroup,
    nup: usize,
    ndn: usize,
    target: Irrep,
) -> Vec<f64> {
    let dets = dets_in_irrep(d, nup, ndn, pg, target);
    if dets.is_empty() {
        return Vec::new();
    }
    dense_eigen(&dense_hamiltonian_matrix(d, t, v, &dets)).0
}

pub(crate) fn gram_eigs(g: &[f64], n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    assert_eq!(g.len(), n * n, "gram is not n x n");
    dense_eigen(&flat_to_nested(g, n)).0
}

pub(crate) fn assert_multiset_eq(got: &[f64], want: &[f64], tol: f64, what: &str) {
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

pub(crate) fn exact_typed_ground(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
) -> (TtVector<f64>, f64, Vec<f64>) {
    exact_typed_eigvec(space, t, v, 0)
}

pub(crate) fn exact_typed_eigvec(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    which: usize,
) -> (TtVector<f64>, f64, Vec<f64>) {
    let d = space.d;
    let dets = dets_in_irrep(d, space.nup, space.ndn, space.pg, space.target);
    assert!(
        !dets.is_empty(),
        "exact_typed_ground: the target irrep {:?} is empty",
        space.target
    );
    let mat = dense_hamiltonian_matrix(d, t, v, &dets);
    let (evals, evecs) = dense_eigen(&mat);
    let mut acc: Option<TtVector<f64>> = None;
    for (i, (ou, od)) in dets.iter().enumerate() {
        let c = evecs[which][i];
        if c == 0.0 {
            continue;
        }
        let term = &tt_state_in::<f64>(space, ou, od) * c;
        acc = Some(match acc {
            None => term,
            Some(a) => &a + &term,
        });
    }
    let mut x = acc.expect("the eigenvector is not identically zero");
    x.round_mut(1e-13, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    x.check();
    (x, evals[which], evals)
}

pub(crate) fn truncated_reference() -> Fixture {
    let pg = fx4_pg();
    let space = QnSpace::with_point_group(D4, T_NUP, T_NDN, pg, Irrep::IDENTITY);
    let (t, v) = fx4_tv(pg);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7103);
    let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x);
    x.check();
    (space, t, v, ham, x)
}

pub(crate) fn d6_minimizer() -> Fixture {
    let pg = d6_pg();
    let space = QnSpace::with_point_group(D6, 2, 2, pg, Irrep::IDENTITY);
    let (t, v) = d6_tv(pg);
    let (ham, x) = typed_minimizer(space, &t, &v, 0x9002, 1, 2);
    (space, t, v, ham, x)
}

pub(crate) fn fx4_minimizer() -> Fixture {
    let pg = fx4_pg();
    let space = QnSpace::with_point_group(D4, F_NUP, F_NDN, pg, Irrep::IDENTITY);
    let (t, v) = fx4_tv(pg);
    let (ham, x) = typed_minimizer(space, &t, &v, 0x7505, 1, 2);
    (space, t, v, ham, x)
}

pub(crate) fn residual_norm(ham: &SparseHamiltonian<f64>, x: &TtVector<f64>) -> f64 {
    let lambda = rayleigh_quotient(ham, x).re_s();
    let (bl, br) = tangent_bases(x);
    normal_residual(ham, &bl, &br, x, lambda).norm_lr(false)
}

pub(crate) fn typed_minimizer(
    space: QnSpace,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    seed: u64,
    lo: usize,
    hi: usize,
) -> (SparseHamiltonian<f64>, TtVector<f64>) {
    #[allow(clippy::type_complexity)]
    static CACHE: Mutex<Option<HashMap<(QnSpace, u64, usize, usize), TtVector<f64>>>> =
        Mutex::new(None);
    let ham = SparseHamiltonian::<f64>::new(space, t, v, None, f64::EPSILON);
    let key = (space, seed, lo, hi);
    let cached = CACHE
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
            for _ in 0..10 {
                if grad_norm_at(&ham, &x) < 1e-11 {
                    break;
                }
                let (_l2, x2) = riemannian_cg(&ham, &x, &opts);
                x = x2;
            }
            normalize_mut(&mut x);
            x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
            normalize_mut(&mut x);
            let (_l3, mut x) = riemannian_cg(&ham, &x, &opts);
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
    let g = grad_norm_at(&ham, &x);
    println!("typed_minimizer(seed {seed:#x}): ||grad|| = {g:e}");
    assert!(g < 1e-8, "typed fixture is not critical (||grad|| = {g:e})");
    (ham, x)
}

pub(crate) fn grad_norm_at(ham: &SparseHamiltonian<f64>, x: &TtVector<f64>) -> f64 {
    let lam = rayleigh_quotient(ham, x).re_s();
    let res = &ham.apply(x) - &(x * lam);
    let (bl, br) = tangent_bases(x);
    TtTangent::project(&bl, &br, &res).norm()
}

pub(crate) fn mgs_tangents(cands: Vec<TtTangent<f64>>) -> Vec<TtTangent<f64>> {
    let mut out: Vec<TtTangent<f64>> = Vec::new();
    for mut u in cands {
        let n0 = u.norm();
        if n0 <= 1e-8 {
            continue;
        }
        for _ in 0..2 {
            for b in &out {
                let c = b.dot(&u);
                u.axpy(-c, b);
            }
        }
        let nrm = u.norm();
        if nrm > 1e-8 * n0 && nrm > 1e-10 {
            u.scale_mut(1.0 / nrm);
            out.push(u);
        }
    }
    out
}

pub(crate) fn gram_spectral_norm(g: &[f64], n: usize) -> f64 {
    gram_eigs(g, n).iter().fold(0.0f64, |m, z| m.max(z.abs()))
}

pub(crate) fn signed_omega2_or_fail(spec: &SymplecticSpectrum, what: &str) -> Vec<f64> {
    match spec {
        SymplecticSpectrum::Unstable {
            reason,
            generator_eigs,
        } => panic!(
            "{what}: the Williamson reduction came back Unstable ({reason:?}) — BOTH H^rr and \
             H^ii are indefinite there, so no signed omega^2 exists and the union identity is \
             not stated. generator eigs = {generator_eigs:?}"
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

pub(crate) fn spectrum_label(spec: &SymplecticSpectrum) -> String {
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

pub(crate) type LihFixture = (
    QnSpace,
    PointGroup,
    CoeffTensor<f64>,
    CoeffTensor<f64>,
    SparseHamiltonian<f64>,
    TtVector<f64>,
);

pub(crate) fn lih_typed_fixture() -> &'static LihFixture {
    static CELL: OnceLock<LihFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        let load = lih_load(true);
        let space = load.space;
        let pg = load.pg.pg;
        let (t, v) = (load.ints.t, load.ints.v);
        let d = space.d;
        let ham = SparseHamiltonian::<f64>::new_with(
            space,
            &t,
            &v,
            None,
            f64::EPSILON,
            Construction::Median,
        );

        let rank = 2usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x11A2);
        let occ: Vec<bool> = (0..d).map(|i| i < space.nup).collect();
        let hf: TtVector<f64> = tt_state_in(space, &occ, &occ);
        let caps = max_ranks(&space);
        let grid: Vec<RankGrid> = (0..=d)
            .map(|b| {
                let mut g = RankGrid::zeros(&space);
                for s in space.bond_sectors(b) {
                    g[s] = if b == 0 || b == d {
                        caps[b][s].min(1)
                    } else {
                        rank.min(caps[b][s])
                    };
                }
                g
            })
            .collect();
        let mut x0 = qntt::vector::perturbation(&hf, &grid, 0.2, &mut rng);
        x0.round_ranks_mut(&grid);
        normalize_mut(&mut x0);
        let (_l, xa) = als(
            &ham,
            &x0,
            &AlsOptions {
                eps: 1e-10,
                max_iter: 60,
                ..AlsOptions::default()
            },
        );
        let mut ropts = RiemannianOpts::new(&space);
        ropts.eps = 1e-13;
        ropts.max_iter = 600;
        ropts.gradtol = 1e-13;
        let polish = |y0: &TtVector<f64>| -> TtVector<f64> {
            let (_l, mut y) = riemannian_cg(&ham, y0, &ropts);
            for _ in 0..30 {
                if grad_norm_at(&ham, &y) < 1e-10 {
                    break;
                }
                let (_l2, y2) = riemannian_cg(&ham, &y, &ropts);
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
        (space, pg, t, v, ham, x)
    })
}

pub(crate) fn lih_load(screen: bool) -> IntegralsLoad<f64> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen,
        ..IntegralsOptions::default()
    };
    load_fixture_dir::<f64>(&dir, &opts).expect("lih_sto3g")
}

pub(crate) fn embed_untyped(
    xi: &TtTangent<f64>,
    bl_c1: &TtVector<f64>,
    br_c1: &TtVector<f64>,
    what: &str,
) -> TtTangent<f64> {
    let y = xi.to_vector().forget_point_group();
    let xi_c1 = TtTangent::project(bl_c1, br_c1, &y);
    let n_xi = xi.norm();
    let n_c1 = xi_c1.norm();
    assert!(
        (n_c1 - n_xi).abs() <= 1e-12 * n_xi.max(1.0),
        "{what}: iota^phi is not isometric across the untyped embedding ({n_c1} vs {n_xi})"
    );
    let back = &xi_c1.to_vector() - &y;
    let ny = y.norm_lr(false);
    let defect = back.norm_lr(false);
    assert!(
        defect <= 1e-10 * ny.max(1.0),
        "{what}: iota^phi(xi) is not ALREADY a tangent of M'' — the untyped projection moved \
         it by {defect:e} (||y|| = {ny:e}). The phi-gauge condition is the (gamma, \
         gamma+phi) block of the untyped one, so this must be the identity"
    );
    xi_c1
}

pub(crate) fn d3_minimal_channel() -> Fixture {
    let pg = PointGroup::new(1, &[Irrep::new(0), Irrep::new(1), Irrep::new(0)]);
    let mut t = CoeffTensor::<f64>::zeros(3, 2);
    for p in 0..3 {
        t.set(&[p, p], 0.17 * (p as f64 - 1.0));
    }
    t.set(&[0, 2], -0.85);
    t.set(&[2, 0], -0.85);
    let mut v = CoeffTensor::<f64>::zeros(3, 4);
    for p in 0..3 {
        for q in 0..3 {
            if p != q {
                let w = 1.0 + (p as f64 - q as f64).abs();
                v.set(&[p, p, q, q], 0.5 / w);
                v.set(&[p, q, q, p], 0.23 / w);
            }
        }
    }
    for idx in [[0, 2, 1, 1], [1, 1, 0, 2], [2, 0, 1, 1], [1, 1, 2, 0]] {
        v.set(&idx, 0.19);
    }
    screen_point_group_forbidden(&mut t, &mut v, pg)
        .expect("the d = 3 fixture is symmetry-allowed by construction — the screen is a no-op");
    let space = QnSpace::with_point_group(3, 2, 2, pg, Irrep::IDENTITY);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let caps = max_ranks(&space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7D30);
    let mut x: TtVector<f64> = qntt::vector::tt_randn(space, &caps, &mut rng);
    normalize_mut(&mut x);
    x.check();
    (space, t, v, ham, x)
}

pub(crate) fn pooled_schmidt(ctx: &WeingartenCtx<'_, f64>) -> Vec<f64> {
    ctx.bond_gauges()
        .iter()
        .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
        .collect()
}

pub(crate) fn schmidt_floor_candidates(values: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = values
        .iter()
        .copied()
        .filter(|s| s.is_finite() && *s > 0.0)
        .collect();
    v.sort_by(f64::total_cmp);
    v.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * b.abs());
    let mut out = Vec::new();
    for w in v.windows(2) {
        if (w[1] - w[0]) / w[1] >= 1e-3 {
            let f = (w[0] * w[1]).sqrt();
            assert!(
                values.iter().all(|s| (s - f).abs() > 1e-6 * f),
                "floor rule: a Schmidt value lies within 1e-6 of the floor {f:e}"
            );
            out.push(f);
        }
    }
    out
}

pub(crate) fn partial_hard_floor(
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    x: &TtVector<f64>,
    phi: Irrep,
) -> Option<(f64, f64, f64)> {
    let space = x.space;
    let space_c1 = QnSpace::new(space.d, space.nup, space.ndn);
    let x_c1 = x.forget_point_group();
    let h_c1 = SparseHamiltonian::<f64>::new(space_c1, t, v, None, f64::EPSILON);
    let ctx_c1 = weingarten_operator(&h_c1, &x_c1);
    let (bl_c1, br_c1) = ctx_c1.bases();
    let gamma = space.target.fuse(phi);
    let cands: Vec<TtTangent<f64>> = tangent_basis(bl_c1, br_c1, false)
        .iter()
        .map(|b| project_irrep(b, bl_c1, br_c1, space.pg, gamma).0)
        .collect();
    let block = mgs_tangents(cands);
    let (_, b_raw) = hessian_grams(&ctx_c1, &block);
    let raw = b_raw.iter().fold(0.0f64, |m, z| m.max(z.abs()));
    for f in schmidt_floor_candidates(&pooled_schmidt(&ctx_c1)) {
        let reg = weingarten_operator(&h_c1, &x_c1).with_curvature(CurvatureReg::Hard(f));
        let (_, b_reg) = hessian_grams(&reg, &block);
        let regm = b_reg.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        let moved = b_reg
            .iter()
            .zip(&b_raw)
            .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
        if moved >= 1e-6 * raw && regm >= 1e-3 * raw {
            return Some((f, raw, regm));
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PencilInertia {
    pub(crate) n_neg_rr: usize,
    pub(crate) n_neg_ii: usize,
    pub(crate) neg: usize,
    pub(crate) complex: usize,
    pub(crate) pos_minus: usize,
    pub(crate) min_abs_eig: f64,
}

impl PencilInertia {
    pub(crate) fn rhs(&self) -> usize {
        self.n_neg_rr + self.n_neg_ii
    }
    pub(crate) fn lhs(&self) -> usize {
        self.neg + self.complex + 2 * self.pos_minus
    }
}

pub(crate) fn pencil_inertia(a: &[f64], b: &[f64], n: usize) -> PencilInertia {
    let hrr: Vec<f64> = a.iter().zip(b).map(|(p, q)| p + q).collect();
    let hii: Vec<f64> = a.iter().zip(b).map(|(p, q)| p - q).collect();
    let er = gram_eigs(&hrr, n);
    let ei = gram_eigs(&hii, n);
    let min_abs_eig = er
        .iter()
        .chain(ei.iter())
        .fold(f64::INFINITY, |m, z| m.min(z.abs()));
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        for l in 0..n {
            let ril = hrr[i * n + l];
            if ril != 0.0 {
                for j in 0..n {
                    m[i * n + j] += ril * hii[l * n + j];
                }
            }
        }
    }
    let (z, q) = qntt::core::factor::dense_eigen_general::<f64>(&m, n);
    let scale = z
        .iter()
        .fold(0.0f64, |acc, c| acc.max(c.norm()))
        .max(1e-300);
    let krein = |qa: &[qntt::C64], qb: &[qntt::C64]| -> qntt::C64 {
        let mut s = qntt::C64::new(0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                s += qa[i].conj() * hii[i * n + j] * qb[j];
            }
        }
        s
    };
    let (mut neg, mut complex, mut pos_minus) = (0usize, 0usize, 0usize);
    let mut done = vec![false; n];
    for i in 0..n {
        if done[i] {
            continue;
        }
        if z[i].im.abs() > 1e-10 * scale {
            complex += 1;
            done[i] = true;
            continue;
        }
        if z[i].re < 0.0 {
            neg += 1;
            done[i] = true;
            continue;
        }
        let cluster: Vec<usize> = (i..n)
            .filter(|&j| {
                !done[j] && z[j].im.abs() <= 1e-10 * scale && (z[j] - z[i]).norm() <= 1e-8 * scale
            })
            .collect();
        let c = cluster.len();
        let mut g = vec![qntt::C64::new(0.0, 0.0); c * c];
        for (p, &jp) in cluster.iter().enumerate() {
            for (r, &jr) in cluster.iter().enumerate() {
                g[p * c + r] = krein(&q[jp], &q[jr]);
            }
        }
        let ev = qntt::core::factor::dense_self_adjoint_smallest::<qntt::C64>(&g, c, 0).0;
        pos_minus += ev.iter().filter(|e| **e < 0.0).count();
        for &j in &cluster {
            done[j] = true;
        }
    }
    PencilInertia {
        n_neg_rr: er.iter().filter(|e| **e < 0.0).count(),
        n_neg_ii: ei.iter().filter(|e| **e < 0.0).count(),
        neg,
        complex,
        pos_minus,
        min_abs_eig,
    }
}

pub(crate) fn pg4c_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(3), Irrep::new(2), Irrep::new(1), Irrep::new(0)],
    )
}

pub(crate) fn pg4c_tv() -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let d = D4;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        t.set(&[p, p], 0.17 * (p as f64 - 0.5 * d as f64));
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            if p != q {
                let w = 1.0 + (p as f64 - q as f64).abs();
                v.set(&[p, p, q, q], 0.5 / w);
                v.set(&[p, q, q, p], 0.23 / w);
            }
        }
    }
    for idx in [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 0, 1],
        [2, 3, 1, 0],
        [3, 2, 1, 0],
    ] {
        v.set(&idx, 0.19);
    }
    screen_point_group_forbidden(&mut t, &mut v, pg4c_pg()).expect(
        "the pg4c fixture is symmetry-allowed by construction — the screen must be a no-op",
    );
    (t, v)
}

pub(crate) fn pg4c_random_reference(seed: u64) -> Fixture {
    let pg = pg4c_pg();
    let space = QnSpace::with_point_group(D4, 2, 2, pg, Irrep::IDENTITY);
    let (t, v) = pg4c_tv();
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let caps = max_ranks(&space);
    let grid: Vec<RankGrid> = (0..=D4)
        .map(|b| {
            let mut g = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                g[s] = caps[b][s].min(1);
            }
            g
        })
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = qntt::vector::tt_randn(space, &grid, &mut rng);
    x.scale_mut(1.0 / x.norm());
    x.check();
    (space, t, v, ham, x)
}

pub(crate) fn flat_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

pub fn krein_signs(r: Option<&qntt::solvers::hess::RpaReadout>) -> Vec<i8> {
    use qntt::solvers::hess::RpaModeKind;
    r.map_or_else(Vec::new, |r| {
        r.modes
            .iter()
            .filter(|m| m.kind != RpaModeKind::Complex)
            .map(|m| {
                if m.kind == RpaModeKind::Real && m.krein != 0.0 {
                    m.krein.signum() as i8
                } else {
                    0
                }
            })
            .collect()
    })
}

pub fn residuals_of(r: Option<&qntt::solvers::hess::RpaReadout>) -> Vec<f64> {
    r.map_or_else(Vec::new, |r| r.residuals())
}

pub fn metric_report(r: Option<&qntt::solvers::hess::RpaReadout>) -> Option<(usize, bool, usize)> {
    use qntt::solvers::hess::RpaReport;
    let r = r?;
    match &r.report {
        RpaReport::Signed {
            events,
            n_negative_basis,
            general_path,
            ..
        } if !events.is_empty() => Some((*n_negative_basis, *general_path, r.complex().count())),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DensePencil {
    pub(crate) real: Vec<f64>,
    pub(crate) complex: Vec<(f64, f64)>,
}

pub(crate) fn dense_pencil(a: &[f64], b: &[f64], n: usize) -> DensePencil {
    let m2 = 2 * n;
    let mut sh = vec![0.0f64; m2 * m2];
    for i in 0..n {
        for j in 0..n {
            let (aij, bij) = (a[i * n + j], b[i * n + j]);
            sh[i * m2 + j] = aij;
            sh[i * m2 + n + j] = bij;
            sh[(n + i) * m2 + j] = -bij;
            sh[(n + i) * m2 + n + j] = -aij;
        }
    }
    let (z, _) = qntt::core::factor::dense_eigen_general::<f64>(&sh, m2);
    let scale = z.iter().fold(0.0f64, |m, c| m.max(c.norm())).max(1e-300);
    let floor = 1e-9 * scale;
    let (mut real, mut complex) = (Vec::new(), Vec::new());
    for l in &z {
        if l.im.abs() <= floor {
            if l.re > 0.0 {
                real.push(l.re * l.re);
            }
        } else if l.re.abs() <= floor {
            if l.im > 0.0 {
                real.push(-l.im * l.im);
            }
        } else if l.re > 0.0 && l.im > 0.0 {
            let w2 = *l * *l;
            complex.push((w2.re, w2.im.abs()));
        }
    }
    real.sort_by(f64::total_cmp);
    complex.sort_by(|p, q| p.0.total_cmp(&q.0));
    assert_eq!(
        real.len() + 2 * complex.len(),
        n,
        "dense_pencil: one representative per ±ω pair (a zero mode or a misclassified λ breaks it)"
    );
    DensePencil { real, complex }
}

impl DensePencil {
    pub(crate) fn wanted(&self, k: usize) -> (Vec<f64>, Vec<(f64, f64)>) {
        let neg: Vec<f64> = self.real.iter().copied().filter(|&z| z < 0.0).collect();
        let pos: Vec<f64> = self.real.iter().copied().filter(|&z| z >= 0.0).collect();
        let take = k.min(pos.len());
        let all_real = take == pos.len();
        let top = pos
            .get(take.wrapping_sub(1))
            .copied()
            .unwrap_or(f64::NEG_INFINITY);
        let real: Vec<f64> = neg.iter().chain(&pos[..take]).copied().collect();
        let complex = self
            .complex
            .iter()
            .copied()
            .filter(|c| all_real || c.0 <= top)
            .collect();
        (real, complex)
    }
}
