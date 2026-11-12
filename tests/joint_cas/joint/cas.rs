// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::OnceLock;

use qntt::core::tangent::{
    TtTangent, min_schmidt, projective_transport, tangent_basis, tangent_dim,
};
use qntt::core::weingarten::{FloorPolicy, bond_gauges};
use qntt::hamiltonian::assembly::Construction;
use qntt::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::rotation::{
    PairMask, e_pq, expm_antisym, g_pq, orbital_gradient, orbital_pairs, rotate_integrals,
};
use qntt::round::RoundMode;
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::hess::{
    NewtonPolishOpts, WeingartenCtx, newton_polish, normal_residual, weingarten_operator,
};
use qntt::solvers::joint::cas::{CasPartition, cas_pairs, cas_rank_grid, cas_reference_state};
use qntt::solvers::joint::{FcRoute, JointOpts, JointScfOpts, joint_context, joint_ground_state};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::solvers::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::vector::perturbation;
use qntt::{
    CoeffTensor, Family, QnSpace, RankGrid, Sector, SparseCore, SparseHamiltonian, TtVector, dot,
    max_ranks, rayleigh_quotient, tt_state_in,
};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::joint::util::*;

pub(crate) fn grid_diff(
    space: &QnSpace,
    got: &[RankGrid],
    want: &[RankGrid],
) -> Vec<(usize, Sector, usize, usize)> {
    assert_eq!(got.len(), space.d + 1, "got: expected d+1 rank grids");
    assert_eq!(want.len(), space.d + 1, "want: expected d+1 rank grids");
    let mut out = Vec::new();
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            if got[b][s] != want[b][s] {
                out.push((b, s, want[b][s], got[b][s]));
            }
        }
    }
    out
}

pub(crate) fn fmt_grid_diff(diff: &[(usize, Sector, usize, usize)]) -> String {
    diff.iter()
        .map(|(b, s, want, got)| format!("b{b} ({},{}) {want}->{got}", s.nup, s.ndn))
        .collect::<Vec<_>>()
        .join("; ")
}

fn regrown_zero_sectors(
    space: &QnSpace,
    got: &[RankGrid],
    want: &[RankGrid],
) -> Vec<(usize, Sector, usize)> {
    let mut out = Vec::new();
    for b in 0..=space.d {
        for s in space.bond_sectors(b) {
            if want[b][s] == 0 && got[b][s] > 0 {
                out.push((b, s, got[b][s]));
            }
        }
    }
    out
}

pub(crate) fn grid_le(space: &QnSpace, got: &[RankGrid], want: &[RankGrid]) -> bool {
    (0..=space.d).all(|b| {
        space
            .bond_sectors(b)
            .into_iter()
            .all(|s| got[b][s] <= want[b][s])
    })
}

pub(crate) fn grid_totals(grids: &[RankGrid]) -> Vec<usize> {
    grids.iter().map(|g| g.total()).collect()
}

fn binom(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc = 1usize;
    for i in 0..k {
        acc = acc * (n - i) / (i + 1);
    }
    acc
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SiteClass {
    Core,
    Active,
    Virt,
}

fn site_class(part: &CasPartition, i: usize) -> SiteClass {
    if i < part.n_core {
        SiteClass::Core
    } else if i < part.n_core + part.n_active {
        SiteClass::Active
    } else {
        SiteClass::Virt
    }
}

fn pair_label(part: &CasPartition, p: usize, q: usize) -> &'static str {
    let (a, b) = (site_class(part, p), site_class(part, q));
    let mut cls = [a, b];
    cls.sort_by_key(|c| match c {
        SiteClass::Core => 0,
        SiteClass::Active => 1,
        SiteClass::Virt => 2,
    });
    match (cls[0], cls[1]) {
        (SiteClass::Core, SiteClass::Core) => "CC",
        (SiteClass::Core, SiteClass::Active) => "CA",
        (SiteClass::Core, SiteClass::Virt) => "CV",
        (SiteClass::Active, SiteClass::Active) => "AA",
        (SiteClass::Active, SiteClass::Virt) => "AV",
        (SiteClass::Virt, SiteClass::Virt) => "VV",
        _ => unreachable!("classes sorted"),
    }
}

fn is_nonredundant(part: &CasPartition, p: usize, q: usize) -> bool {
    site_class(part, p) != site_class(part, q)
}

pub(crate) fn cas_ci_dim(space: &QnSpace, part: &CasPartition) -> usize {
    binom(part.n_active, space.nup - part.n_core) * binom(part.n_active, space.ndn - part.n_core)
}

pub(crate) fn expected_n_eff(part: &CasPartition) -> usize {
    part.n_active * part.n_core + part.n_core * part.n_virt + part.n_active * part.n_virt
}

fn component_norm(space: &QnSpace, core: &SparseCore<f64>, k: usize) -> f64 {
    let mut acc = 0.0f64;
    let mut buf = Vec::new();
    for f in Family::ALL {
        for s in space.bond_sectors(k) {
            let Some((rows, cols)) = core.block_dims(f, s) else {
                continue;
            };
            if rows == 0 || cols == 0 {
                continue;
            }
            buf.resize(rows * cols, 0.0);
            let _ = core.block_copy_out(f, s, &mut buf);
            for v in &buf {
                acc += v * v;
            }
        }
    }
    acc.sqrt()
}

fn vmax(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &x| a.max(x.abs()))
}

fn vmax_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FdFrames {
    TransportedToState,
    FrozenBaseFrames,
    TransportedToGaugeFrame,
}

#[allow(clippy::too_many_arguments)]
fn covector_fd_variant(
    t_cur: &CoeffTensor<f64>,
    v_cur: &CoeffTensor<f64>,
    space: QnSpace,
    pairs: &[(usize, usize)],
    a_dir: &[f64],
    x_base: &TtVector<f64>,
    bl0: &TtVector<f64>,
    br0: &TtVector<f64>,
    xi_dir: &TtTangent<f64>,
    h: f64,
    frames: FdFrames,
) -> (TtTangent<f64>, Vec<f64>) {
    let d = space.d;
    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    for (&(p, q), &ai) in pairs.iter().zip(a_dir) {
        k.set(&[p, q], ai);
        k.set(&[q, p], -ai);
    }
    let kflat = coeff2_flat(&k, d);

    let one = |s: f64| -> (TtTangent<f64>, Vec<f64>) {
        let u = expm_antisym(&scale_flat(&kflat, -s), d);
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
        let g_xi = match frames {
            FdFrames::TransportedToState => {
                let (bl_s, br_s) = tangent_bases(&xs);
                projective_transport(x_base, &TtTangent::project(&bl_s, &br_s, &res))
            }
            FdFrames::FrozenBaseFrames => TtTangent::project(bl0, br0, &res),
            FdFrames::TransportedToGaugeFrame => {
                let (bl_s, br_s) = tangent_bases(&xs);
                projective_transport(bl0, &TtTangent::project(&bl_s, &br_s, &res))
            }
        };
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

pub(crate) struct CasFixture {
    pub(crate) label: &'static str,
    pub(crate) space: QnSpace,
    pub(crate) ints: Integrals<f64>,
    pub(crate) ham: SparseHamiltonian<f64>,
    pub(crate) part: CasPartition,
    pub(crate) m_active: usize,
    pub(crate) grid: Vec<RankGrid>,
    pub(crate) lambda: f64,
    pub(crate) x: TtVector<f64>,
}

impl CasFixture {
    pub(crate) fn frames(&self) -> (TtVector<f64>, TtVector<f64>) {
        tangent_bases(&self.x)
    }

    pub(crate) fn z_norm(&self) -> f64 {
        let (bl, br) = self.frames();
        let z = normal_residual(&self.ham, &bl, &br, &self.x, self.lambda);
        dot(&z, &z).sqrt()
    }

    pub(crate) fn orb_grad_max(&self) -> f64 {
        let g = orbital_gradient(&self.ham, &self.x);
        cas_pairs(&self.part, self.space.d)
            .keep
            .iter()
            .fold(0.0f64, |acc, &(p, q)| acc.max(g.get(&[p, q]).abs()))
    }
}

fn build_cas_fixture(
    label: &'static str,
    dirname: &str,
    part: CasPartition,
    m_active: usize,
    seed: u64,
) -> CasFixture {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(dirname);
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{label}: loading {dirname} failed: {e:?}"));
    let ints = load.ints;
    let d = ints.norb;
    assert_eq!(
        ints.ms2, 0,
        "{label}: both CAS fixtures are Sz = 0 singlets"
    );
    let nup = (ints.nelec + ints.ms2.unsigned_abs() as usize) / 2;
    let ndn = ints.nelec - nup;
    let space = QnSpace::new(d, nup, ndn);
    assert_eq!(part.d(), d, "{label}: partition must cover the chain");
    let ham = SparseHamiltonian::new_with(
        space,
        &ints.t,
        &ints.v,
        None,
        f64::EPSILON,
        Construction::Median,
    );

    let grid = cas_rank_grid(&space, &part, qntt::PerSector(m_active));
    assert_eq!(
        grid.len(),
        d + 1,
        "{label}: cas_rank_grid must return d+1 grids"
    );
    let (lambda, x) = cas_reference_state(&ham, &space, &part, m_active, seed);
    x.check();

    let diff = grid_diff(&space, &x.ranks, &grid);
    assert!(
        grid_le(&space, &x.ranks, &grid),
        "{label}: cas_reference_state left the pinned grid from ABOVE (round_ranks_mut is a \
         '<= target' operation, so only shrinkage is legal): {}",
        fmt_grid_diff(&diff)
    );
    assert!(
        diff.is_empty(),
        "{label}: realized grid != pinned grid: {}",
        fmt_grid_diff(&diff)
    );

    let (bl, br) = tangent_bases(&x);
    let g_tt = TtTangent::project(&bl, &br, &{
        let hx = ham.apply(&x);
        &hx - &(&x * lambda)
    })
    .norm();
    let bound = JointOpts::default().tt_grad_tol * lambda.abs().max(1.0);
    assert!(
        g_tt <= bound,
        "{label}: reference state is not TT-critical: ‖g_tt‖ = {g_tt:.3e} > {bound:.3e} \
         (JointOpts::default().tt_grad_tol · max(1,|λ|))"
    );
    let smin = min_schmidt(&bl, &br);
    let dropped: f64 = bond_gauges(&bl, &br, FloorPolicy::default_hard())
        .iter()
        .map(|g| g.dropped_mass)
        .sum();
    eprintln!(
        "tjcas fixture {label}: d={d} N=({nup},{ndn}) part={}/{}/{} m_active={m_active} \
         λ={lambda:.10} λ+e0={:.10} totals={:?} tangent_dim={} ‖g_tt‖={g_tt:.3e} \
         min_schmidt={smin:.3e} gauge_dropped_mass={dropped:.3e}",
        part.n_core,
        part.n_active,
        part.n_virt,
        lambda + ints.e0,
        grid_totals(&grid),
        tangent_dim(&bl),
    );

    CasFixture {
        label,
        space,
        ints,
        ham,
        part,
        m_active,
        grid,
        lambda,
        x,
    }
}

pub(crate) fn lih_cas22() -> &'static CasFixture {
    static CELL: OnceLock<CasFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        build_cas_fixture(
            "lih_cas22",
            "lih_sto3g",
            CasPartition {
                n_core: 1,
                n_active: 2,
                n_virt: 3,
            },
            0,
            0xCA22,
        )
    })
}

pub(crate) fn h2o_cas86() -> &'static CasFixture {
    static CELL: OnceLock<CasFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        build_cas_fixture(
            "h2o_cas86",
            "h2o_sto3g",
            CasPartition {
                n_core: 1,
                n_active: 6,
                n_virt: 0,
            },
            0,
            0xCA86,
        )
    })
}

pub(crate) fn h2o_cas86_m4() -> &'static CasFixture {
    static CELL: OnceLock<CasFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        build_cas_fixture(
            "h2o_cas86_m4",
            "h2o_sto3g",
            CasPartition {
                n_core: 1,
                n_active: 6,
                n_virt: 0,
            },
            4,
            0xCA84,
        )
    })
}

struct FullRankFixture {
    space: QnSpace,
    ham: SparseHamiltonian<f64>,
    x: TtVector<f64>,
}

fn lih_fullrank() -> &'static FullRankFixture {
    static CELL: OnceLock<FullRankFixture> = OnceLock::new();
    CELL.get_or_init(|| {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
        let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
        let ints = load.ints;
        let d = ints.norb;
        let space = QnSpace::new(d, 2, 2);
        let ham = SparseHamiltonian::new_with(
            space,
            &ints.t,
            &ints.v,
            None,
            f64::EPSILON,
            Construction::Median,
        );
        let full = max_ranks(&space);
        let occ: Vec<bool> = (0..d).map(|i| i < 2).collect();
        let hf: TtVector<f64> = tt_state_in(space, &occ, &occ);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xF011);
        let mut x0 = perturbation(&hf, &full, 0.2, &mut rng);
        x0.round_ranks_mut(&full);
        normalize_mut(&mut x0);
        let (_l, xa) = als(
            &ham,
            &x0,
            &AlsOptions {
                eps: 1e-12,
                max_iter: 200,
                ..AlsOptions::default()
            },
        );
        let (mut x, _info) = newton_polish(
            &ham,
            &xa,
            &NewtonPolishOpts {
                gradtol: 1e-11,
                ..Default::default()
            },
        );
        normalize_mut(&mut x);
        x.check();
        FullRankFixture { space, ham, x }
    })
}

fn check_gpq_against_dense(fx: &CasFixture, p: usize, q: usize) {
    let d = fx.space.d;
    let dets = dets_for(d, fx.space.nup, fx.space.ndn);
    let c = dense_state_vector(&dets, &fx.x);
    let gd = dense_gpq(d, p, q, &dets);
    let want = mat_vec(&gd, &c);
    let label = pair_label(&fx.part, p, q);

    for &(a, b) in &[(p, q), (q, p)] {
        let ed = dense_epq_matrix(d, a, b, &dets);
        let want_e = mat_vec(&ed, &c);
        let got_e = dense_state_vector(&dets, &e_pq(&fx.x, a, b, 1.0f64));
        let dev_e = vmax_diff(&got_e, &want_e);
        let tol_e = 1e-10 * vmax(&want_e).max(1.0);
        eprintln!(
            "tjcas_a1 {}: e_pq({a},{b}) [{label}] max|E·c|={:.3e} dev={dev_e:.3e}",
            fx.label,
            vmax(&want_e)
        );
        assert!(
            dev_e <= tol_e,
            "{}: e_pq({a},{b}) [{label}] vs dense oracle: {dev_e:.3e} > {tol_e:.3e}",
            fx.label
        );
    }

    let gtt = g_pq(&fx.x, p, q);
    let got = dense_state_vector(&dets, &gtt);
    let dev = vmax_diff(&got, &want);
    let scale = vmax(&want);
    eprintln!(
        "tjcas_a1 {}: ({p},{q}) [{label}] max|G·c|={scale:.3e} dev={dev:.3e}",
        fx.label
    );

    let tol = 1e-10 * scale.max(1.0);
    assert!(
        dev <= tol,
        "{}: g_pq({p},{q}) [{label}] vs dense oracle: {dev:.3e} > {tol:.3e}",
        fx.label
    );

    if matches!(label, "CC" | "VV") {
        assert!(
            scale <= 1e-14,
            "{}: the dense oracle says ({p},{q}) [{label}] is NOT structurally zero \
             (max|G·c| = {scale:.3e}) — the class prediction is wrong, not the TT",
            fx.label
        );
        assert!(
            vmax(&got) <= 1e-14,
            "{}: g_pq({p},{q}) [{label}] must be the exact-zero train, got max {:.3e}",
            fx.label,
            vmax(&got)
        );
    } else {
        assert!(
            scale > 1e-3,
            "{}: ({p},{q}) [{label}] is vacuous — max|G·c| = {scale:.3e}",
            fx.label
        );
    }
}

#[test]
fn tjcas_a1_epq_gpq_across_pin() {
    let lih = lih_cas22();
    for &(p, q) in &[(1usize, 0usize), (3, 0), (3, 1), (2, 1), (4, 3)] {
        check_gpq_against_dense(lih, p, q);
    }

    let h2o = h2o_cas86();
    for &(p, q) in &[(1usize, 0usize), (2, 1)] {
        check_gpq_against_dense(h2o, p, q);
    }

    let d = lih.space.d;
    let dets = dets_for(d, lih.space.nup, lih.space.ndn);
    let c = dense_state_vector(&dets, &lih.x);
    let (p, q) = (3usize, 0usize);
    let gd = dense_gpq(d, p, q, &dets);
    let gc = mat_vec(&gd, &c);
    let want_minus: Vec<f64> = gc.iter().map(|v| -v).collect();
    let ua = &g_pq(&lih.x, p, q) * (-1.0f64);
    let got = dense_state_vector(&dets, &ua);
    let dev_minus = vmax_diff(&got, &want_minus);
    let dev_plus = vmax_diff(&got, &gc);
    let teeth = 0.5 * vmax(&gc);
    eprintln!(
        "tjcas_a1 sign pin: dev(−G·c)={dev_minus:.3e} dev(+G·c)={dev_plus:.3e} teeth={teeth:.3e}"
    );
    assert!(
        teeth > 1e3 * 1e-10 * vmax(&gc).max(1.0),
        "sign-pin teeth are too small to have meaning: teeth={teeth:.3e}"
    );
    assert!(
        dev_minus <= 1e-10 * vmax(&gc).max(1.0),
        "u_a = −Ĝ_a·x must reproduce the dense oracle: {dev_minus:.3e}"
    );
    assert!(
        dev_plus > teeth,
        "negative arm (+Ĝ_a·x) did not fail by a margin: {dev_plus:.3e} <= teeth {teeth:.3e}"
    );
}

fn check_tangent_keystone(fx: &CasFixture, gram_tol: f64) {
    let (bl, br) = fx.frames();
    let dim = tangent_dim(&bl);

    for b in 0..=fx.space.d {
        if b <= fx.part.n_core || b >= fx.part.n_core + fx.part.n_active {
            assert_eq!(
                fx.grid[b].total(),
                1,
                "{}: pinned bond {b} must carry total rank exactly 1, got {}",
                fx.label,
                fx.grid[b].total()
            );
        }
    }

    if fx.m_active == 0 {
        assert_eq!(
            dim,
            cas_ci_dim(&fx.space, &fx.part),
            "{}: at FULL active rank the TT tangent space IS the CAS-CI tangent space",
            fx.label
        );
    } else {
        assert!(
            dim < cas_ci_dim(&fx.space, &fx.part),
            "{}: a truncated active rank must give a STRICTLY smaller tangent space \
             (dim={dim}, CAS-CI={})",
            fx.label,
            cas_ci_dim(&fx.space, &fx.part)
        );
    }

    let tb = tangent_basis(&bl, &br, true);
    assert_eq!(
        tb.len(),
        dim - 1,
        "{}: tangent_basis(.., exclude_radial=true) must be exactly tangent_dim − 1 long \
         (over-long ⇒ singular Grams; short ⇒ span loss — both historical F7 modes)",
        fx.label
    );

    let n = tb.len();
    let strides: &[usize] = if n <= 80 { &[] } else { &[1, 7, 31] };
    let mut probes: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        probes.push((i, i));
        if strides.is_empty() {
            for j in i + 1..n {
                probes.push((i, j));
            }
        } else {
            for &st in strides {
                if i + st < n {
                    probes.push((i, i + st));
                }
            }
        }
    }
    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for &(i, j) in &probes {
        let g = tb[i].dot(&tb[j]);
        let want = if i == j { 1.0 } else { 0.0 };
        let dev = (g - want).abs();
        if dev > worst {
            worst = dev;
            worst_at = (i, j);
        }
    }
    let smin = min_schmidt(&bl, &br);
    eprintln!(
        "tjcas_a2 {}: tangent_dim={dim} basis={n} worst|G−I|={worst:.3e} at {worst_at:?} \
         min_schmidt={smin:.3e}",
        fx.label
    );
    assert!(
        worst < gram_tol,
        "{}: tangent basis not orthonormal: worst |G−I| = {worst:.3e} > {gram_tol:.3e} \
         (min_schmidt = {smin:.3e})",
        fx.label
    );
}

#[test]
fn tjcas_a2_tangent_dim_and_gram() {
    check_tangent_keystone(lih_cas22(), 1e-10);
    check_tangent_keystone(h2o_cas86(), 1e-6);
}

fn check_normal_classification(fx: &CasFixture, normal_tol: f64, tangential_tol: f64) {
    let (bl, br) = fx.frames();
    let d = fx.space.d;
    let mut u_by_pair: Vec<(usize, usize, &'static str, f64, f64)> = Vec::new();
    let mut us: Vec<(usize, usize, TtVector<f64>)> = Vec::new();

    for &(p, q) in orbital_pairs(fx.space.pg, d).iter() {
        let ua = &g_pq(&fx.x, p, q) * (-1.0f64);
        let n_u = dot(&ua, &ua).sqrt();
        let tau = TtTangent::project(&bl, &br, &ua);
        let n_tau = tau.norm();
        u_by_pair.push((p, q, pair_label(&fx.part, p, q), n_u, n_tau));
        us.push((p, q, ua));
    }

    for &(p, q, label, n_u, n_tau) in &u_by_pair {
        let ratio = if n_u > 0.0 { n_tau / n_u } else { 0.0 };
        eprintln!(
            "tjcas_a3 {}: ({p},{q}) [{label}] ‖u‖={n_u:.6e} ‖P(u)‖={n_tau:.6e} ratio={ratio:.3e}",
            fx.label
        );
        match label {
            "CC" | "VV" => {
                assert!(
                    n_u <= 1e-14,
                    "{}: ({p},{q}) [{label}] must be the exact-zero train, ‖u‖={n_u:.3e}",
                    fx.label
                );
            }
            "CA" | "CV" | "AV" => {
                assert!(
                    n_u > 1e-2,
                    "{}: ({p},{q}) [{label}] ‖u‖={n_u:.3e} too small for the ratio test to mean \
                     anything",
                    fx.label
                );
                assert!(
                    ratio <= normal_tol,
                    "{}: ({p},{q}) [{label}] must be PURELY NORMAL at a pinned point: \
                     ‖P(u)‖/‖u‖ = {ratio:.3e} > {normal_tol:.3e}",
                    fx.label
                );
            }
            "AA" => {
                assert!(
                    n_u > 1e-6,
                    "{}: ({p},{q}) [{label}] ‖u‖={n_u:.3e} vacuous",
                    fx.label
                );
                if fx.m_active == 0 {
                    assert!(
                        (1.0 - ratio).abs() <= tangential_tol,
                        "{}: ({p},{q}) [{label}] must be TANGENTIAL at full active rank: \
                         1 − ‖P(u)‖/‖u‖ = {:.3e} > {tangential_tol:.3e}",
                        fx.label,
                        (1.0 - ratio).abs()
                    );
                }
            }
            _ => unreachable!(),
        }
    }

    let mut worst_cross = 0.0f64;
    let mut worst_at = ((0usize, 0usize), (0usize, 0usize));
    for (i, (pa, qa, ua)) in us.iter().enumerate() {
        for (pb, qb, ub) in us.iter().skip(i + 1) {
            let la = pair_label(&fx.part, *pa, *qa);
            let lb = pair_label(&fx.part, *pb, *qb);
            if la == lb {
                continue;
            }
            let s = dot(ua, ub).abs();
            if s > worst_cross {
                worst_cross = s;
                worst_at = ((*pa, *qa), (*pb, *qb));
            }
        }
    }
    eprintln!(
        "tjcas_a3 {}: worst cross-class |S_ab| = {worst_cross:.3e} at {worst_at:?}",
        fx.label
    );
    assert!(
        worst_cross <= 1e-12,
        "{}: cross-class |S_ab| = {worst_cross:.3e} at {worst_at:?} — N is not class-block-diagonal",
        fx.label
    );
}

#[test]
fn tjcas_a3_normal_classification() {
    check_normal_classification(lih_cas22(), 1e-12, 1e-10);
    check_normal_classification(h2o_cas86(), 1e-12, 1e-8);
}

fn n_eff_with(fx: &CasFixture, pairs: Option<PairMask>, orb_floor: f64) -> (usize, Vec<f64>) {
    let opts = JointOpts {
        orb_floor,
        pairs,
        ..JointOpts::default()
    };
    let ctx = match joint_context(&fx.ham, &fx.x, &opts) {
        Ok(c) => c,
        Err(e) => panic!(
            "{}: joint_context refused at the pinned reference (orb_floor={orb_floor:e}): {e:?}",
            fx.label
        ),
    };
    (ctx.n_eff(), ctx.orb_spectrum().to_vec())
}

fn check_n_eff_pin(fx: &CasFixture) {
    let want = expected_n_eff(&fx.part);
    let d = fx.space.d;

    let zn = fx.z_norm();
    let og = fx.orb_grad_max();
    eprintln!(
        "tjcas_a4 {}: ‖z‖={zn:.3e} max|orbital_gradient| over cas_pairs={og:.3e} \
         (TT-critical, orbital-NON-critical, quoted per the criticality gate)",
        fx.label
    );
    assert!(
        zn > 1e-3,
        "{}: ‖z‖ = {zn:.3e} — the pinned manifold is supposed to carry a LARGE normal residual; \
         a small one means the reference collapsed onto an exact eigenstate and every curvature \
         arm downstream is vacuous",
        fx.label
    );

    let mask = cas_pairs(&fx.part, d);
    assert_eq!(
        mask.keep.len(),
        want,
        "{}: cas_pairs must keep exactly N_A·N_I + N_I·N_E + N_A·N_E pairs",
        fx.label
    );
    assert!(
        mask.keep.windows(2).all(|w| w[0] < w[1]),
        "{}: cas_pairs must preserve with_predicate's canonical (p asc, q asc) order: {:?}",
        fx.label,
        mask.keep
    );
    assert!(
        mask.keep
            .iter()
            .all(|&(p, q)| p > q && is_nonredundant(&fx.part, p, q)),
        "{}: cas_pairs kept a redundant (same-class) pair: {:?}",
        fx.label,
        mask.keep
    );

    let (n_cas, spec_cas) = n_eff_with(fx, Some(cas_pairs(&fx.part, d)), 1e-8);
    assert_eq!(
        n_cas, want,
        "{}: n_eff under the CAS mask must be the analytic non-redundant count",
        fx.label
    );

    let (n_full, spec_full) = n_eff_with(fx, None, 1e-8);
    assert_eq!(
        n_full, want,
        "{}: at full active rank the FULL triangle must auto-classify to the same n_eff \
         (spectrum: {spec_full:?})",
        fx.label
    );
    assert_eq!(
        spec_full.len(),
        orbital_pairs(fx.space.pg, d).len(),
        "{}: orb_spectrum must carry every pair",
        fx.label
    );

    let n_dropped = spec_full.len() - n_full;
    let max_dropped = spec_full[..n_dropped]
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs()));
    let min_kept = spec_full[n_dropped..]
        .iter()
        .fold(f64::INFINITY, |a, &v| a.min(v));
    eprintln!(
        "tjcas_a4 {}: n_eff={n_full}/{} max|μ dropped|={max_dropped:.3e} min μ kept={min_kept:.3e} \
         gap={:.3e} decades  (cas-mask spectrum: {spec_cas:?})",
        fx.label,
        spec_full.len(),
        (min_kept / max_dropped.max(f64::MIN_POSITIVE)).log10()
    );
    assert!(
        min_kept > 1e6 * max_dropped.max(1e-300),
        "{}: the keep rule is marginal — min kept μ = {min_kept:.3e}, max dropped |μ| = \
         {max_dropped:.3e}",
        fx.label
    );

    for &floor in &[1e-10f64, 1e-8, 1e-6] {
        let (n, _) = n_eff_with(fx, None, floor);
        assert_eq!(
            n, want,
            "{}: n_eff changed at orb_floor = {floor:e} (full triangle)",
            fx.label
        );
        let (n, _) = n_eff_with(fx, Some(cas_pairs(&fx.part, d)), floor);
        assert_eq!(
            n, want,
            "{}: n_eff changed at orb_floor = {floor:e} (cas mask)",
            fx.label
        );
    }

    let ctx = joint_context(&fx.ham, &fx.x, &JointOpts::default()).expect("full-triangle context");
    let pairs = ctx.pairs().to_vec();
    let mut worst_leak = 0.0f64;
    let mut worst_row = (0usize, 0usize);
    for j in 0..ctx.n_eff() {
        let mut e = vec![0.0f64; ctx.n_eff()];
        e[j] = 1.0;
        let col = ctx.v_times(&e);
        for (a, &(p, q)) in pairs.iter().enumerate() {
            if !is_nonredundant(&fx.part, p, q) && col[a].abs() > worst_leak {
                worst_leak = col[a].abs();
                worst_row = (p, q);
            }
        }
    }
    eprintln!(
        "tjcas_a4 {}: worst kept-eigenvector leak onto a REDUNDANT pair row = {worst_leak:.3e} at \
         {worst_row:?}",
        fx.label
    );
    assert!(
        worst_leak <= 1e-6,
        "{}: a kept orbital direction leaks onto the redundant pair {worst_row:?}: \
         {worst_leak:.3e}",
        fx.label
    );
}

#[test]
fn tjcas_a4_neff_pins_and_floor_sweep() {
    check_n_eff_pin(lih_cas22());
    check_n_eff_pin(h2o_cas86());

    let fr = lih_fullrank();
    let ctx = joint_context(&fr.ham, &fr.x, &JointOpts::default())
        .expect("full-rank reference must be TT-critical");
    let mu_max = ctx
        .orb_spectrum()
        .iter()
        .fold(0.0f64, |a, &v| a.max(v.abs()));
    eprintln!(
        "tjcas_a4 negative arm (lih FULL rank): n_eff={} max|μ|={mu_max:.3e} totals={:?}",
        ctx.n_eff(),
        grid_totals(&fr.x.ranks)
    );
    assert_eq!(
        ctx.n_eff(),
        0,
        "at full TT rank the joint stack must degenerate to TT-RPA exactly (n_eff == 0)"
    );
    assert!(
        mu_max <= 1e-9,
        "full-rank orb_spectrum is not a roundoff cluster: max|μ| = {mu_max:.3e}"
    );
    let pinned = cas_rank_grid(
        &fr.space,
        &CasPartition {
            n_core: 1,
            n_active: 2,
            n_virt: 3,
        },
        qntt::PerSector::NONE,
    );
    assert!(
        !grid_diff(&fr.space, &fr.x.ranks, &pinned).is_empty(),
        "the full-rank negative arm accidentally sits on the pinned grid — it has no teeth"
    );
}

#[test]
fn tjcas_a4b_truncated_rank_divergence() {
    let fx = h2o_cas86_m4();
    let d = fx.space.d;
    let (n_cas, _) = n_eff_with(fx, Some(cas_pairs(&fx.part, d)), 1e-8);
    let (n_full, spec_full) = n_eff_with(fx, None, 1e-8);
    let n_pairs_full = orbital_pairs(fx.space.pg, d).len();
    eprintln!(
        "tjcas_a4b {} (m_active={}): n_eff(cas mask)={n_cas}  n_eff(full triangle)={n_full}/\
         {n_pairs_full}  spectrum={spec_full:?}",
        fx.label, fx.m_active
    );

    assert_eq!(
        n_cas,
        expected_n_eff(&fx.part),
        "the CAS mask's own kept count is a property of the mask, not of the rank"
    );
    assert!(
        n_full > n_cas,
        "at truncated active rank the full triangle must keep MORE directions than the CAS mask: \
         {n_full} vs {n_cas}"
    );
    assert!(
        n_full <= n_pairs_full,
        "n_eff cannot exceed the pair count: {n_full} > {n_pairs_full}"
    );
    let extra = n_full - n_cas;
    let min_extra = spec_full[n_pairs_full - n_full..n_pairs_full - n_cas]
        .iter()
        .fold(f64::INFINITY, |a, &v| a.min(v));
    eprintln!("tjcas_a4b: {extra} extra kept directions, smallest μ = {min_extra:.3e}");
    assert!(
        min_extra > 1e-10,
        "the extra kept directions are floor-marginal (min μ = {min_extra:.3e}) — the divergence \
         claim has no teeth"
    );
}

fn strongest_b_pair(
    wctx: &WeingartenCtx<'_, f64>,
    tb: &[TtTangent<f64>],
) -> (usize, f64, usize, f64) {
    let mut ib = 0usize;
    let mut nb = 0.0f64;
    for (i, v) in tb.iter().enumerate() {
        let n = wctx.b_apply(v).norm();
        if n > nb {
            nb = n;
            ib = i;
        }
    }
    let b_xi = wctx.b_apply(&tb[ib]);
    let mut jb = if ib == 0 { 1 } else { 0 };
    let mut cb = 0.0f64;
    for (j, v) in tb.iter().enumerate() {
        if j == ib {
            continue;
        }
        let c = v.dot(&b_xi).abs();
        if c > cb {
            cb = c;
            jb = j;
        }
    }
    (ib, nb, jb, cb)
}

#[allow(clippy::too_many_arguments)]
fn covector_dev(
    fx: &CasFixture,
    pairs: &[(usize, usize)],
    a_dir: &[f64],
    xi: &TtTangent<f64>,
    ref_xi: &TtTangent<f64>,
    ref_c: &[f64],
    h: f64,
    frames: FdFrames,
) -> (f64, f64) {
    let (bl0, br0) = fx.frames();
    let (fd_xi, fd_c) = covector_fd_variant(
        &fx.ints.t, &fx.ints.v, fx.space, pairs, a_dir, &fx.x, &bl0, &br0, xi, h, frames,
    );
    let mut dxi = fd_xi.clone();
    dxi.axpy(-1.0, ref_xi);
    let dev_xi = dxi.norm() / ref_xi.norm().max(1.0);
    let dev_c = vmax_diff(&fd_c, ref_c) / vmax(ref_c).max(1.0);
    (dev_xi, dev_c)
}

fn check_covector_fd(fx: &CasFixture, tol: f64, gauge_frame_bites: bool, full_joint_tol: f64) {
    let d = fx.space.d;
    let opts = JointOpts {
        pairs: Some(cas_pairs(&fx.part, d)),
        ..JointOpts::default()
    };
    let jctx = joint_context(&fx.ham, &fx.x, &opts).expect("pinned reference must be TT-critical");
    let pairs = jctx.pairs().to_vec();
    let n_orb = pairs.len();
    assert!(n_orb >= 1, "{}: need orbital pairs for this arm", fx.label);

    let (bl, br) = fx.frames();
    let tb = tangent_basis(&bl, &br, true);
    assert!(!tb.is_empty());
    let xi = tb[0].clone();

    let og = fx.orb_grad_max();
    eprintln!(
        "tjcas_a6 {}: n_orb={n_orb} tangent basis {} max|orbital_gradient|={og:.3e} \
         (orbital-NON-critical, quoted per the criticality gate)",
        fx.label,
        tb.len()
    );

    let zeros = vec![0.0f64; n_orb];
    let (mut ref_xi, mut ref_c) = jctx.a_covector(&xi, &zeros, FcRoute::Sweep);
    {
        let (bxi, bc) = jctx.b_covector(&xi, &zeros, FcRoute::Sweep);
        ref_xi.axpy(1.0, &bxi);
        for (r, b) in ref_c.iter_mut().zip(&bc) {
            *r += b;
        }
    }
    assert!(
        ref_xi.norm() > 1e-6 && vmax(&ref_c) > 1e-6,
        "{}: the analytic covector is vacuous (‖F_ξ‖={:.3e}, max|F_c|={:.3e})",
        fx.label,
        ref_xi.norm(),
        vmax(&ref_c)
    );

    {
        let h = 1e-3;
        let (a_xi, a_c) = joint_covector_fd_central(
            &fx.ints.t, &fx.ints.v, fx.space, &pairs, &zeros, &fx.x, &xi, h,
        );
        let (bl0, br0) = fx.frames();
        let (b_xi, b_c) = covector_fd_variant(
            &fx.ints.t,
            &fx.ints.v,
            fx.space,
            &pairs,
            &zeros,
            &fx.x,
            &bl0,
            &br0,
            &xi,
            h,
            FdFrames::TransportedToState,
        );
        let mut dd = a_xi.clone();
        dd.axpy(-1.0, &b_xi);
        assert!(
            dd.norm() <= 1e-12 * a_xi.norm().max(1.0),
            "{}: the local FD variant must reproduce util::joint_covector_fd_central exactly \
             (ξ leg): {:.3e}",
            fx.label,
            dd.norm()
        );
        assert!(
            vmax_diff(&a_c, &b_c) <= 1e-12 * vmax(&a_c).max(1.0),
            "{}: the local FD variant must reproduce util::joint_covector_fd_central exactly \
             (c leg): {:.3e}",
            fx.label,
            vmax_diff(&a_c, &b_c)
        );
    }

    let hs = [9e-3f64, 3e-3, 1e-3];
    let mut devs = Vec::new();
    for &h in &hs {
        let dev = covector_dev(
            fx,
            &pairs,
            &zeros,
            &xi,
            &ref_xi,
            &ref_c,
            h,
            FdFrames::TransportedToState,
        );
        eprintln!(
            "tjcas_a6 {} transported: h={h:.1e} dev_ξ={:.3e} dev_c={:.3e}",
            fx.label, dev.0, dev.1
        );
        devs.push(dev);
    }
    for k in 0..hs.len() - 1 {
        let r_xi = devs[k].0 / devs[k + 1].0.max(f64::MIN_POSITIVE);
        let r_c = devs[k].1 / devs[k + 1].1.max(f64::MIN_POSITIVE);
        assert!(
            (4.0..=25.0).contains(&r_xi) || devs[k + 1].0 <= tol,
            "{}: transported ξ leg is not second order (h {:.1e}->{:.1e}, ratio {r_xi:.2})",
            fx.label,
            hs[k],
            hs[k + 1]
        );
        assert!(
            (4.0..=25.0).contains(&r_c) || devs[k + 1].1 <= tol,
            "{}: transported c leg is not second order (h {:.1e}->{:.1e}, ratio {r_c:.2})",
            fx.label,
            hs[k],
            hs[k + 1]
        );
    }
    let (dev_xi, dev_c) = devs[devs.len() - 1];
    assert!(
        dev_xi <= tol && dev_c <= tol,
        "{}: FD-of-gradient vs A+B at h={:.1e}: dev_ξ={dev_xi:.3e}, dev_c={dev_c:.3e} > {tol:.3e}",
        fx.label,
        hs[hs.len() - 1]
    );

    let (bl0, br0) = fx.frames();
    let g_tr = covector_fd_variant(
        &fx.ints.t,
        &fx.ints.v,
        fx.space,
        &pairs,
        &zeros,
        &fx.x,
        &bl0,
        &br0,
        &xi,
        1e-3,
        FdFrames::TransportedToState,
    )
    .0;
    let g_fr = covector_fd_variant(
        &fx.ints.t,
        &fx.ints.v,
        fx.space,
        &pairs,
        &zeros,
        &fx.x,
        &bl0,
        &br0,
        &xi,
        1e-3,
        FdFrames::FrozenBaseFrames,
    )
    .0;
    let mut d_fr = g_fr.clone();
    d_fr.axpy(-1.0, &g_tr);
    let b_xi = weingarten_operator(&fx.ham, &fx.x).b_apply(&xi);
    eprintln!(
        "tjcas_a6 {}: NOT-A-NEGATIVE-ARM [frozen base frames] at h=1e-3: \
         ‖frozen−transported‖={:.3e} beside ‖Bξ‖={:.3e} — the frozen bias IS the Weingarten term, \
         and B ≡ 0 at full active rank",
        fx.label,
        d_fr.norm(),
        b_xi.norm()
    );
    assert!(
        d_fr.norm() <= 1e-10 * ref_xi.norm().max(1.0),
        "{}: at full active rank the frozen-frame and transported conventions must COINCIDE \
         (their difference is the Weingarten term, which vanishes on a linear slice): {:.3e}",
        fx.label,
        d_fr.norm()
    );

    if gauge_frame_bites {
        let mut wrong = Vec::new();
        for &h in &hs {
            let dev = covector_dev(
                fx,
                &pairs,
                &zeros,
                &xi,
                &ref_xi,
                &ref_c,
                h,
                FdFrames::TransportedToGaugeFrame,
            );
            eprintln!(
                "tjcas_a6 {} NEGATIVE [transported to a gauge FRAME]: h={h:.1e} dev_ξ={:.3e} \
                 dev_c={:.3e}",
                fx.label, dev.0, dev.1
            );
            wrong.push(dev);
        }
        let last = wrong[wrong.len() - 1].0;
        let ratio = wrong[0].0 / last.max(f64::MIN_POSITIVE);
        assert!(
            last > 1e3 * dev_xi.max(1e-14),
            "{}: negative arm [gauge FRAME] did not fail by a margin: {last:.3e} vs transported \
             {dev_xi:.3e}",
            fx.label
        );
        assert!(
            ratio < 3.0,
            "{}: negative arm [gauge FRAME] is CONVERGING (ratio {ratio:.2}) — it should be a flat \
             bias, so this arm is not testing what it claims",
            fx.label
        );
    } else {
        let dev = covector_dev(
            fx,
            &pairs,
            &zeros,
            &xi,
            &ref_xi,
            &ref_c,
            hs[hs.len() - 1],
            FdFrames::TransportedToGaugeFrame,
        );
        let mut dbl = bl0.clone();
        dbl = &dbl - &fx.x;
        let bl0_dev = dot(&dbl, &dbl).sqrt();
        eprintln!(
            "tjcas_a6 {}: NOT-A-NEGATIVE-ARM [gauge FRAME] dev_ξ={:.3e} vs transported \
             {dev_xi:.3e} — ‖bl0−x‖={bl0_dev:.3e}, so the left- and right-orthogonal gauges \
             coincide here",
            fx.label, dev.0
        );
        assert!(
            bl0_dev <= 1e-12,
            "{}: the gauge-FRAME negative arm is demoted here on the assumption that the left- \
             and right-orthogonal gauges coincide bit-for-bit (‖bl0−x‖ small) — that assumption \
             just broke (‖bl0−x‖={bl0_dev:.3e}); re-enable `gauge_frame_bites` for this fixture \
             instead of silently skipping a negative arm that would now have teeth",
            fx.label
        );
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCA06);
    let a_dir: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();
    let (mut rj_xi, mut rj_c) = jctx.a_covector(&xi, &a_dir, FcRoute::Sweep);
    {
        let (bxi, bc) = jctx.b_covector(&xi, &a_dir, FcRoute::Sweep);
        rj_xi.axpy(1.0, &bxi);
        for (r, b) in rj_c.iter_mut().zip(&bc) {
            *r += b;
        }
    }
    let (jd_xi, jd_c) = covector_dev(
        fx,
        &pairs,
        &a_dir,
        &xi,
        &rj_xi,
        &rj_c,
        1e-3,
        FdFrames::TransportedToState,
    );
    eprintln!(
        "tjcas_a6 {} FULL joint direction (orbital-non-critical, max|g_orb|={og:.3e}): \
         dev_ξ={jd_xi:.3e} dev_c={jd_c:.3e}  [full_joint_tol={full_joint_tol:.1e}]",
        fx.label
    );
    assert!(
        jd_xi <= full_joint_tol && jd_c <= full_joint_tol,
        "{}: full-joint-direction FD-of-gradient vs A+B: dev_ξ={jd_xi:.3e}, dev_c={jd_c:.3e} > \
         {full_joint_tol:.1e}",
        fx.label
    );
}

fn check_covector_fd_negative(fx: &CasFixture) {
    assert!(
        fx.m_active > 0,
        "{}: the negative arms need a CURVED pinned manifold (truncated active rank)",
        fx.label
    );
    let d = fx.space.d;
    let opts = JointOpts {
        pairs: Some(cas_pairs(&fx.part, d)),
        ..JointOpts::default()
    };
    let jctx = joint_context(&fx.ham, &fx.x, &opts).expect("pinned reference must be TT-critical");
    let pairs = jctx.pairs().to_vec();
    let zeros = vec![0.0f64; pairs.len()];
    let (bl0, br0) = fx.frames();
    let wctx = weingarten_operator(&fx.ham, &fx.x);
    let tb = tangent_basis(&bl0, &br0, true);

    let (ib, nb, _jb, _cb) = strongest_b_pair(&wctx, &tb);
    assert!(
        nb > 1e-3,
        "{}: no direction carries enough Weingarten content ({nb:.3e}) for the frozen-frame arm",
        fx.label
    );
    let xi = tb[ib].clone();
    let b_xi = wctx.b_apply(&xi);

    let (mut ref_xi, mut ref_c) = jctx.a_covector(&xi, &zeros, FcRoute::Sweep);
    {
        let (bxi, bc) = jctx.b_covector(&xi, &zeros, FcRoute::Sweep);
        ref_xi.axpy(1.0, &bxi);
        for (r, b) in ref_c.iter_mut().zip(&bc) {
            *r += b;
        }
    }

    let hs = [9e-3f64, 3e-3, 1e-3];
    let mut dev_tr = Vec::new();
    for &h in &hs {
        dev_tr.push(
            covector_dev(
                fx,
                &pairs,
                &zeros,
                &xi,
                &ref_xi,
                &ref_c,
                h,
                FdFrames::TransportedToState,
            )
            .0,
        );
    }
    let dev_xi = dev_tr[dev_tr.len() - 1];
    eprintln!(
        "tjcas_a6b {}: direction = basis vector {ib} (‖Bξ‖={nb:.3e}); transported dev_ξ {:.3e} / \
         {:.3e} / {:.3e} over h = 9e-3 / 3e-3 / 1e-3",
        fx.label, dev_tr[0], dev_tr[1], dev_tr[2]
    );
    assert!(
        dev_xi <= 1e-4,
        "{}: the transported (certified) leg must still converge here: {dev_xi:.3e}",
        fx.label
    );

    for (frames, name) in [
        (FdFrames::FrozenBaseFrames, "frozen base frames"),
        (
            FdFrames::TransportedToGaugeFrame,
            "transported to a gauge FRAME",
        ),
    ] {
        let mut wrong = Vec::new();
        for &h in &hs {
            let dev = covector_dev(fx, &pairs, &zeros, &xi, &ref_xi, &ref_c, h, frames).0;
            eprintln!(
                "tjcas_a6b {} NEGATIVE [{name}]: h={h:.1e} dev_ξ={dev:.3e}",
                fx.label
            );
            wrong.push(dev);
        }
        let last = wrong[wrong.len() - 1];
        let ratio = wrong[0] / last.max(f64::MIN_POSITIVE);
        assert!(
            last > 1e2 * dev_xi.max(1e-14),
            "{}: negative arm [{name}] did not fail by a margin: {last:.3e} vs transported \
             {dev_xi:.3e}",
            fx.label
        );
        assert!(
            ratio < 3.0,
            "{}: negative arm [{name}] is CONVERGING (ratio {ratio:.2}) — it should be a flat \
             bias, so this arm is not testing what it claims",
            fx.label
        );
    }

    let mut ids = Vec::new();
    for &h in &hs {
        let one = |fr| {
            covector_fd_variant(
                &fx.ints.t, &fx.ints.v, fx.space, &pairs, &zeros, &fx.x, &bl0, &br0, &xi, h, fr,
            )
            .0
        };
        let mut diff = one(FdFrames::FrozenBaseFrames);
        diff.axpy(-1.0, &one(FdFrames::TransportedToState));
        diff.axpy(1.0, &b_xi);
        eprintln!(
            "tjcas_a6b {}: identity h={h:.1e}: ‖(frozen−transported) + Bξ‖={:.3e} \
             (‖Bξ‖={:.3e})",
            fx.label,
            diff.norm(),
            b_xi.norm()
        );
        ids.push(diff.norm());
    }
    assert!(
        ids[2] <= 2e-2 * b_xi.norm(),
        "{}: (frozen − transported) does not converge to −Bξ: {:.3e} vs ‖Bξ‖={:.3e}",
        fx.label,
        ids[2],
        b_xi.norm()
    );
    for k in 0..ids.len() - 1 {
        let r = ids[k] / ids[k + 1].max(f64::MIN_POSITIVE);
        assert!(
            (4.0..=25.0).contains(&r),
            "{}: the frozen-frame identity is not second order (h {:.1e}->{:.1e}, ratio {r:.2})",
            fx.label,
            hs[k],
            hs[k + 1]
        );
    }
}

#[test]
fn tjcas_a6_covector_fd_transport() {
    check_covector_fd(lih_cas22(), 1e-5, false, 5e-3);
    check_covector_fd(h2o_cas86(), 1e-4, true, 5e-4);
}

#[test]
fn tjcas_a6b_covector_fd_negative_arms_at_truncated_rank() {
    check_covector_fd_negative(h2o_cas86_m4());
}

fn check_retraction_grid(fx: &CasFixture) {
    let (bl, br) = fx.frames();
    let tb = tangent_basis(&bl, &br, true);
    assert!(!tb.is_empty());
    let xi = &tb[0];

    let y0 = xi.retract_hosvd(0.0);
    y0.check();
    let d0 = grid_diff(&fx.space, &y0.ranks, &fx.grid);
    assert!(
        d0.is_empty(),
        "{}: retract_hosvd(0) left the pinned grid: {}",
        fx.label,
        fmt_grid_diff(&d0)
    );

    for &alpha in &[1e-3f64, -1e-3, 0.1, -0.1] {
        let mut y = xi.retract_hosvd(alpha);
        y.check();
        assert!(
            grid_le(&fx.space, &y.ranks, &fx.grid),
            "{}: retract_hosvd({alpha}) GREW a pinned-zero sector: {}",
            fx.label,
            fmt_grid_diff(&grid_diff(&fx.space, &y.ranks, &fx.grid))
        );
        assert!(
            regrown_zero_sectors(&fx.space, &y.ranks, &fx.grid).is_empty(),
            "{}: retract_hosvd({alpha}) regrew a pinned-ZERO sector",
            fx.label
        );
        let dd = grid_diff(&fx.space, &y.ranks, &fx.grid);
        assert!(
            dd.is_empty(),
            "{}: retract_hosvd({alpha}) shrank the grid: {}",
            fx.label,
            fmt_grid_diff(&dd)
        );
        normalize_mut(&mut y);
        y.check();
    }
}

#[test]
fn tjcas_a7_retraction_grid_roundtrip() {
    check_retraction_grid(lih_cas22());
    check_retraction_grid(h2o_cas86());
}

fn raw_gauge_census(space: &QnSpace, r: &[RankGrid]) -> (usize, usize, Vec<usize>) {
    let d = space.d;
    let mut raw = 0usize;
    let mut gauge = 0usize;
    let mut per_site = vec![0usize; d];
    for k in 0..d {
        let g = space.pg.irrep_of(k);
        for l in space.bond_sectors(k) {
            for f in Family::ALL {
                let t = f.apply(l, g);
                if space.contains(k + 1, t) {
                    per_site[k] += r[k][l] * r[k + 1][t];
                }
            }
        }
        raw += per_site[k];
        if k < d - 1 {
            for s in space.bond_sectors(k + 1) {
                gauge += r[k + 1][s] * r[k + 1][s];
            }
        }
    }
    (raw, gauge, per_site)
}

fn check_mgs_census(fx: &CasFixture, support_tol: f64) {
    let (bl, br) = fx.frames();
    let dim = tangent_dim(&bl);
    let (raw, gauge, per_site) = raw_gauge_census(&fx.space, &fx.grid);
    assert_eq!(
        raw - gauge,
        dim,
        "{}: the structural raw−gauge count must reproduce tangent_dim",
        fx.label
    );

    let mut pinned_sites = Vec::new();
    for (k, &raw_k) in per_site.iter().enumerate() {
        let cls = site_class(&fx.part, k);
        if cls != SiteClass::Active {
            assert_eq!(
                raw_k, 1,
                "{}: pinned site {k} ({cls:?}) must be a single 1×1×1 scalar block, raw={raw_k}",
                fx.label
            );
            pinned_sites.push(k);
        }
    }
    assert_eq!(
        pinned_sites.len(),
        fx.part.n_core + fx.part.n_virt,
        "{}: every core and virtual site must be pinned",
        fx.label
    );

    let tb = tangent_basis(&bl, &br, true);
    assert_eq!(
        tb.len(),
        dim - 1,
        "{}: exactly dim−1 accepted — over-acceptance turns into singular Grams, \
         under-acceptance into silent span loss",
        fx.label
    );

    let mut worst = 0.0f64;
    let mut worst_at = (0usize, 0usize);
    for (i, v) in tb.iter().enumerate() {
        for &k in &pinned_sites {
            let n = component_norm(&fx.space, v.component(k), k);
            if n > worst {
                worst = n;
                worst_at = (i, k);
            }
        }
    }
    let smin = min_schmidt(&bl, &br);
    let dropped: f64 = bond_gauges(&bl, &br, FloorPolicy::default_hard())
        .iter()
        .map(|g| g.dropped_mass)
        .sum();
    eprintln!(
        "tjcas_a8 {}: raw={raw} gauge={gauge} dim={dim} accepted={} pinned_sites={pinned_sites:?} \
         worst pinned-site component norm={worst:.3e} at {worst_at:?} min_schmidt={smin:.3e} \
         gauge dropped_mass={dropped:.3e}  [floors: MGS_EMPTY_SLOT=1e-12, MGS_REL_TOL=1e-8, \
         MGS_NOISE_FLOOR=1e-13]",
        fx.label,
        tb.len()
    );
    assert!(
        worst <= support_tol,
        "{}: an accepted tangent direction carries weight {worst:.3e} on PINNED site {} — the \
         empty-slot cull did not fire (min_schmidt={smin:.3e}, gauge dropped_mass={dropped:.3e})",
        fx.label,
        worst_at.1
    );

    let mut relaxed = fx.grid.clone();
    let b_relax = fx.part.n_core;
    let mut added = 0usize;
    for s in fx.space.bond_sectors(b_relax) {
        if relaxed[b_relax][s] == 0 && max_ranks(&fx.space)[b_relax][s] > 0 {
            relaxed[b_relax][s] = 1;
            added += 1;
            break;
        }
    }
    assert_eq!(
        added, 1,
        "{}: could not find a pinned-zero sector to relax at bond {b_relax} — the control has no \
         teeth",
        fx.label
    );
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCA08);
    let mut xr: TtVector<f64> = qntt::tt_randn(fx.space, &relaxed, &mut rng);
    normalize_mut(&mut xr);
    xr.check();
    let dq = grid_diff(&fx.space, &xr.ranks, &relaxed);
    assert!(
        dq.is_empty(),
        "{}: the control state must sit on the RELAXED grid exactly, else it is measuring \
         something else: {}",
        fx.label,
        fmt_grid_diff(&dq)
    );
    let (blr, brr) = tangent_bases(&xr);
    let dim_r = tangent_dim(&blr);
    let tbr = tangent_basis(&blr, &brr, true);
    eprintln!(
        "tjcas_a8 {} CONTROL: relaxed bond {b_relax} totals {:?} -> dim {dim_r} accepted {} \
         min_schmidt={:.3e}",
        fx.label,
        grid_totals(&xr.ranks),
        tbr.len(),
        min_schmidt(&blr, &brr)
    );
    assert!(
        dim_r > dim,
        "{}: relaxing a pinned-zero sector must ENLARGE the tangent space ({dim_r} vs {dim})",
        fx.label
    );
    assert_eq!(
        tbr.len(),
        dim_r - 1,
        "{}: the relaxed point's genuinely-populated slots must survive the cull — accepted {} \
         vs dim−1 {}",
        fx.label,
        tbr.len(),
        dim_r - 1
    );
}

#[test]
fn tjcas_a8_mgs_empty_slot_census() {
    check_mgs_census(lih_cas22(), 1e-10);
    check_mgs_census(h2o_cas86(), 1e-8);
}

#[test]
fn tjcas_a9_mals_regrowth_antioracle() {
    let fx = lih_cas22();
    let opts = MalsOptions {
        eps: 1e-10,
        max_iter: 3,
        ..MalsOptions::default()
    };
    let (lam_mals, xm) = mals(&fx.ham, &fx.x, &opts);
    xm.check();

    let regrown = regrown_zero_sectors(&fx.space, &xm.ranks, &fx.grid);
    let (blm, _brm) = tangent_bases(&xm);
    let dim_pinned = tangent_dim(&fx.frames().0);
    let dim_mals = tangent_dim(&blm);
    eprintln!(
        "tjcas_a9 {}: pinned totals {:?} -> MALS totals {:?}; tangent_dim {dim_pinned} -> \
         {dim_mals}; λ {:.10} -> {lam_mals:.10}; regrown pinned-zero sectors: {}",
        fx.label,
        grid_totals(&fx.grid),
        grid_totals(&xm.ranks),
        fx.lambda,
        regrown
            .iter()
            .map(|(b, s, r)| format!("b{b} ({},{}) 0->{r}", s.nup, s.ndn))
            .collect::<Vec<_>>()
            .join(", ")
    );

    assert!(
        !regrown.is_empty(),
        "{}: MALS left the pinned grid intact — the anti-oracle has no teeth, and either MALS \
         changed or the fixture is not really pinned",
        fx.label
    );
    assert!(
        !grid_diff(&fx.space, &xm.ranks, &fx.grid).is_empty(),
        "{}: MALS must not preserve the pinned grid",
        fx.label
    );
    assert!(
        dim_mals > dim_pinned,
        "{}: MALS must ENLARGE the tangent space ({dim_mals} vs {dim_pinned})",
        fx.label
    );
    let named = Sector::new(0, 0);
    assert!(
        regrown.iter().any(|&(b, s, _)| b == 1 && s == named),
        "{}: expected bond 1 sector (0,0) among the regrown pinned-zero sectors, got {regrown:?}",
        fx.label
    );
    let margin = fx.lambda - lam_mals;
    assert!(
        margin > 1e-6,
        "{}: MALS did not lower the energy below the CAS reference by a measurable margin \
         ({margin:.3e}) — the regrowth is not carrying dynamic correlation and the arm is weak",
        fx.label
    );
}

fn check_perturbation_composition(fx: &CasFixture) {
    let d = fx.space.d;
    let occ_up: Vec<bool> = (0..d).map(|i| i < fx.space.nup).collect();
    let occ_dn: Vec<bool> = (0..d).map(|i| i < fx.space.ndn).collect();
    let hf: TtVector<f64> = tt_state_in(fx.space, &occ_up, &occ_dn);
    hf.check();

    assert!(
        grid_le(&fx.space, &hf.ranks, &fx.grid),
        "{}: the HF determinant must live inside the pinned grid: {}",
        fx.label,
        fmt_grid_diff(&grid_diff(&fx.space, &hf.ranks, &fx.grid))
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCA10);
    let mut p = perturbation(&hf, &fx.grid, 0.2, &mut rng);
    let raised = regrown_zero_sectors(&fx.space, &p.ranks, &fx.grid);
    eprintln!(
        "tjcas_a10 {}: pinned totals {:?} -> post-perturbation {:?} ({} pinned-zero sectors \
         RAISED by the max(1,·) floor)",
        fx.label,
        grid_totals(&fx.grid),
        grid_totals(&p.ranks),
        raised.len()
    );
    assert!(
        !raised.is_empty(),
        "{}: perturbation did NOT raise any pinned-zero sector — either the max(1,·) floor \
         changed or the grid is not really pinned; the mandatory round below would then be \
         certifying nothing",
        fx.label
    );
    let named = Sector::new(0, 0);
    assert!(
        raised.iter().any(|&(b, s, _)| b == 1 && s == named),
        "{}: expected bond 1 sector (0,0) among the raised pinned-zero sectors, got {raised:?}",
        fx.label
    );

    p.round_ranks_mut(&fx.grid);
    let diff = grid_diff(&fx.space, &p.ranks, &fx.grid);
    assert!(
        diff.is_empty(),
        "{}: round_ranks_mut(&pinned) must restore the pinned grid EXACTLY, sector by sector, at \
         all d+1 bonds: {}",
        fx.label,
        fmt_grid_diff(&diff)
    );
    normalize_mut(&mut p);
    p.check();

    let dets = dets_for(d, fx.space.nup, fx.space.ndn);
    let c_hf = dense_state_vector(&dets, &hf);
    let c_p = dense_state_vector(&dets, &p);
    let num: f64 = c_p
        .iter()
        .zip(&c_hf)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let den: f64 = c_hf.iter().map(|b| b * b).sum::<f64>().sqrt();
    eprintln!(
        "tjcas_a10 {}: rounded perturbed state vs HF, determinant-basis relative distance = {:.3e}",
        fx.label,
        num / den
    );
    assert!(
        num / den > 1e-3,
        "{}: the rounded perturbation collapsed back onto HF (relative distance {:.3e}) — the \
         seed carries no enrichment",
        fx.label,
        num / den
    );

    let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0xCA11);
    let q: TtVector<f64> = qntt::tt_randn(fx.space, &fx.grid, &mut rng2);
    let dq = grid_diff(&fx.space, &q.ranks, &fx.grid);
    assert!(
        dq.is_empty(),
        "{}: tt_randn must honour a pinned grid exactly: {}",
        fx.label,
        fmt_grid_diff(&dq)
    );
}

#[test]
fn tjcas_a10_perturbation_round_composition() {
    check_perturbation_composition(lih_cas22());
    check_perturbation_composition(h2o_cas86());
}

#[test]
fn tjcas_a11_riemannian_cg_unpins_newton_polish_does_not() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let (bl, br) = fx.frames();
        let tb = tangent_basis(&bl, &br, true);
        assert!(!tb.is_empty(), "{}: empty tangent basis", fx.label);
        let mut x0 = tb[0].retract_hosvd(0.3);
        normalize_mut(&mut x0);
        x0.check();
        let d0 = grid_diff(&fx.space, &x0.ranks, &fx.grid);
        assert!(
            d0.is_empty(),
            "{}: the displaced starting point must itself land on the pinned grid exactly \
             (retract_hosvd closes onto it): {}",
            fx.label,
            fmt_grid_diff(&d0)
        );

        let (xp, info) = newton_polish(
            &fx.ham,
            &x0,
            &NewtonPolishOpts {
                gradtol: 1e-11,
                ..Default::default()
            },
        );
        xp.check();
        eprintln!(
            "tjcas_a11 {}: newton_polish from a displaced (non-critical) start totals {:?} \
             (pinned {:?}), outer {}, applies {}, gnorm {:.3e}, converged {}",
            fx.label,
            grid_totals(&xp.ranks),
            grid_totals(&fx.grid),
            info.outer,
            info.applies,
            info.gnorm,
            info.converged
        );
        assert!(
            info.outer >= 1,
            "{}: the displaced start must force at least one real Newton step, or this arm is \
             vacuous again (see the comment above) — got outer = 0",
            fx.label
        );
        let diff = grid_diff(&fx.space, &xp.ranks, &fx.grid);
        assert!(
            diff.is_empty(),
            "{}: newton_polish must preserve the pinned grid (its retraction closes with \
             round_ranks_mut(&base grid), which cannot grow and must not shrink here): {}",
            fx.label,
            fmt_grid_diff(&diff)
        );
    }

    let fx = h2o_cas86();
    let mut opts = RiemannianOpts::new(&fx.space);
    opts.eps = 1e-13;
    opts.max_iter = 100;
    opts.gradtol = 1e-13;
    opts.preserve_grid = false;
    let (_l, xc) = riemannian_cg(&fx.ham, &fx.x, &opts);
    xc.check();
    let diff = grid_diff(&fx.space, &xc.ranks, &fx.grid);
    let dim_pinned = tangent_dim(&fx.frames().0);
    let (blc, _brc) = tangent_bases(&xc);
    let dim_cg = tangent_dim(&blc);
    eprintln!(
        "tjcas_a11 {}: riemannian_cg totals {:?} (pinned {:?}); tangent_dim {dim_pinned} -> \
         {dim_cg}; grid diff: {}",
        fx.label,
        grid_totals(&xc.ranks),
        grid_totals(&fx.grid),
        fmt_grid_diff(&diff)
    );
    assert!(
        grid_le(&fx.space, &xc.ranks, &fx.grid),
        "{}: riemannian_cg's entry round is shrink-only; a GROWN sector means something else \
         changed: {}",
        fx.label,
        fmt_grid_diff(&diff)
    );
    assert!(
        !diff.is_empty() && dim_cg < dim_pinned,
        "{}: riemannian_cg was expected to UN-PIN this state via its entry \
         round_mut(1e-14, Abs) — it did not (tangent_dim {dim_pinned} -> {dim_cg}). If CG has \
         been made pin-aware, this arm and escalation E2 both need updating; do NOT simply \
         re-wire cas_reference_state to use it without a fresh certification.",
        fx.label
    );

    let typed_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h2o_sto3g");
    let typed = load_fixture_dir::<f64>(
        &typed_dir,
        &IntegralsOptions {
            point_group: PointGroupPolicy::Resolve(None),
            ..IntegralsOptions::default()
        },
    )
    .expect("h2o_sto3g, C2v-typed");
    assert!(
        typed.space.pg.order() > 1,
        "the typed h2o space must actually carry a point group"
    );
    let typed_grid = cas_rank_grid(&typed.space, &fx.part, qntt::PerSector(fx.m_active));
    for (b, tg) in typed_grid.iter().enumerate() {
        for s in fx.space.bond_sectors(b) {
            let want: usize = typed
                .space
                .bond_sectors(b)
                .into_iter()
                .filter(|t| t.nup == s.nup && t.ndn == s.ndn)
                .map(|t| tg[t])
                .sum();
            assert_eq!(
                xc.ranks[b][s], want,
                "{}: riemannian_cg's rank-revealing round at bond {b} sector ({},{}) should land \
                 exactly on the C2v-typed CAS cap (got {}, typed cap {want}) — if this drifts, the \
                 h2o deficiency is NOT the point-group block structure and the comment above is \
                 wrong",
                fx.label, s.nup, s.ndn, xc.ranks[b][s]
            );
        }
    }
    eprintln!(
        "tjcas_a11 {}: CG-revealed grid == C2v-typed CAS cap, sector by sector (untyped totals \
         {:?}, typed {:?})",
        fx.label,
        grid_totals(&fx.grid),
        grid_totals(&typed_grid)
    );
}

const A12_ENTRY_ROUND_EPS: f64 = 1e-14;

fn tt_bit_diff(space: &QnSpace, a: &TtVector<f64>, b: &TtVector<f64>) -> Result<(), String> {
    if a.ranks.len() != b.ranks.len() {
        return Err(format!(
            "rank-grid length {} vs {}",
            a.ranks.len(),
            b.ranks.len()
        ));
    }
    for b_idx in 0..a.ranks.len() {
        for s in space.bond_sectors(b_idx) {
            if a.ranks[b_idx][s] != b.ranks[b_idx][s] {
                return Err(format!(
                    "bond {b_idx} sector ({},{}) rank {} vs {}",
                    s.nup, s.ndn, a.ranks[b_idx][s], b.ranks[b_idx][s]
                ));
            }
        }
    }
    let mut buf_a = Vec::new();
    let mut buf_b = Vec::new();
    for (k, (ca, cb)) in a.cores.iter().zip(&b.cores).enumerate() {
        for f in Family::ALL {
            for s in space.bond_sectors(k) {
                let (da, db) = (ca.block_dims(f, s), cb.block_dims(f, s));
                if da != db {
                    return Err(format!(
                        "site {k} family {f:?} sector ({},{}) dims {da:?} vs {db:?}",
                        s.nup, s.ndn
                    ));
                }
                let Some((rows, cols)) = da else { continue };
                if rows == 0 || cols == 0 {
                    continue;
                }
                buf_a.resize(rows * cols, 0.0);
                buf_b.resize(rows * cols, 0.0);
                let _ = ca.block_copy_out(f, s, &mut buf_a);
                let _ = cb.block_copy_out(f, s, &mut buf_b);
                for (i, (&va, &vb)) in buf_a.iter().zip(&buf_b).enumerate() {
                    if va.to_bits() != vb.to_bits() {
                        return Err(format!(
                            "site {k} family {f:?} sector ({},{}) slot {i}: {va:e} vs {vb:e} \
                             (bits {:#x} vs {:#x})",
                            s.nup,
                            s.ndn,
                            va.to_bits(),
                            vb.to_bits()
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn a12_opts(space: &QnSpace, max_iter: usize, preserve_grid: bool) -> RiemannianOpts {
    RiemannianOpts {
        eps: 1e-13,
        max_iter,
        gradtol: 1e-13,
        preserve_grid,
        ..RiemannianOpts::new(space)
    }
}

#[test]
fn tjcas_a12b_preserve_grid_keeps_the_pinned_grid() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let (bl0, br0) = fx.frames();
        let dim_pinned = tangent_dim(&bl0);
        let smin = min_schmidt(&bl0, &br0);
        let dropped: f64 = bond_gauges(&bl0, &br0, FloorPolicy::default_hard())
            .iter()
            .map(|g| g.dropped_mass)
            .sum();

        for (name, run) in [("cg", 0usize), ("gd", 1usize)] {
            let opts = a12_opts(&fx.space, 20, true);
            let (lam, xp) = if run == 0 {
                riemannian_cg(&fx.ham, &fx.x, &opts)
            } else {
                riemannian_gd(&fx.ham, &fx.x, &opts)
            };
            xp.check();
            let diff = grid_diff(&fx.space, &xp.ranks, &fx.grid);
            let (blp, _brp) = tangent_bases(&xp);
            let dim_p = tangent_dim(&blp);
            eprintln!(
                "tjcas_a12b {} {name}(preserve_grid=true): totals {:?} (pinned {:?}), \
                 tangent_dim {dim_pinned} -> {dim_p}, λ = {lam:.12}, min_schmidt = {smin:.3e}, \
                 gauge_dropped_mass = {dropped:.3e}",
                fx.label,
                grid_totals(&xp.ranks),
                grid_totals(&fx.grid),
            );
            assert!(
                diff.is_empty(),
                "{} {name}: preserve_grid must carry the DECLARED grid through the whole loop \
                 unchanged (the entry round is skipped and every downstream primitive is \
                 grid-closing): {}",
                fx.label,
                fmt_grid_diff(&diff)
            );
            assert_eq!(
                dim_p, dim_pinned,
                "{} {name}: preserve_grid kept the rank grid but the tangent dimension moved \
                 ({dim_pinned} -> {dim_p}) — that means a sector's realized rank fell below its \
                 declared one somewhere inside the loop",
                fx.label
            );
        }
    }
}

#[test]
fn tjcas_a12c_preserve_grid_removes_exactly_the_entry_round() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let opts_def = a12_opts(&fx.space, 8, false);
        let (l_a, x_a) = riemannian_cg(&fx.ham, &fx.x, &opts_def);
        let (l_b, x_b) = riemannian_cg(&fx.ham, &fx.x, &opts_def);
        assert_eq!(
            l_a.to_bits(),
            l_b.to_bits(),
            "{}: determinism control failed inside tjcas_a12c — see tjcas_a12a",
            fx.label
        );
        assert!(tt_bit_diff(&fx.space, &x_a, &x_b).is_ok());

        let x_pre =
            fx.x.round(A12_ENTRY_ROUND_EPS, qntt::Pooled::NONE, RoundMode::Abs);
        let opts_pre = a12_opts(&fx.space, 8, true);
        let (l_pre, x_p) = riemannian_cg(&fx.ham, &x_pre, &opts_pre);
        eprintln!(
            "tjcas_a12c {}: default λ = {l_a:.17e} (bits {:#x}); pre-rounded+preserve λ = \
             {l_pre:.17e} (bits {:#x}); pinned totals {:?}, pre-rounded totals {:?}",
            fx.label,
            l_a.to_bits(),
            l_pre.to_bits(),
            grid_totals(&fx.grid),
            grid_totals(&x_pre.ranks),
        );
        assert_eq!(
            l_pre.to_bits(),
            l_a.to_bits(),
            "{}: cg(pre-rounded x, preserve_grid = true) must be BIT-identical to cg(x, default) \
             — the two runs receive identical input and must execute the same instruction stream. \
             A difference means `preserve_grid` changed something OTHER than the entry round \
             (a reordering, a moved `normalize_mut`, a `round()`-vs-`round_mut()` swap).",
            fx.label
        );
        if let Err(e) = tt_bit_diff(&fx.space, &x_p, &x_a) {
            panic!(
                "{}: cg(pre-rounded x, preserve) vs cg(x, default) differ in the state: {e}",
                fx.label
            );
        }
    }
}

#[test]
fn tjcas_a12d_default_contract_gd_and_unpinned_noop() {
    let fx = h2o_cas86();
    let opts = a12_opts(&fx.space, 20, false);
    let (_l, xg) = riemannian_gd(&fx.ham, &fx.x, &opts);
    xg.check();
    let diff = grid_diff(&fx.space, &xg.ranks, &fx.grid);
    let dim_pinned = tangent_dim(&fx.frames().0);
    let dim_gd = tangent_dim(&tangent_bases(&xg).0);
    eprintln!(
        "tjcas_a12d {}: riemannian_gd(default) totals {:?} (pinned {:?}), tangent_dim \
         {dim_pinned} -> {dim_gd}, diff {}",
        fx.label,
        grid_totals(&xg.ranks),
        grid_totals(&fx.grid),
        fmt_grid_diff(&diff)
    );
    assert!(
        grid_le(&fx.space, &xg.ranks, &fx.grid),
        "{}: gd's entry round is shrink-only; a GROWN sector means something else changed: {}",
        fx.label,
        fmt_grid_diff(&diff)
    );
    assert!(
        !diff.is_empty() && dim_gd < dim_pinned,
        "{}: riemannian_gd's DEFAULT is expected to un-pin this genuinely rank-deficient state \
         via its entry round_mut(1e-14, Abs) — it did not (tangent_dim {dim_pinned} -> {dim_gd}). \
         If the default changed, this arm, tjcas_a11 and the deliverable-A escalation all need \
         updating together.",
        fx.label
    );

    let fr = lih_fullrank();
    let o_def = a12_opts(&fr.space, 5, false);
    let o_pre = a12_opts(&fr.space, 5, true);
    let (l_def, x_def) = riemannian_cg(&fr.ham, &fr.x, &o_def);
    let (l_pre, x_pre) = riemannian_cg(&fr.ham, &fr.x, &o_pre);
    let declared = grid_totals(&fr.x.ranks);
    eprintln!(
        "tjcas_a12d lih_fullrank: declared {declared:?}; totals default {:?} preserve {:?}; \
         λ {l_def:.14} vs {l_pre:.14} (|Δλ| = {:.3e})",
        grid_totals(&x_def.ranks),
        grid_totals(&x_pre.ranks),
        (l_def - l_pre).abs()
    );
    assert!(
        grid_le(&fr.space, &x_def.ranks, &fr.x.ranks),
        "lih_fullrank: the default's entry round is shrink-only"
    );
    assert!(
        !grid_diff(&fr.space, &x_def.ranks, &fr.x.ranks).is_empty(),
        "lih_fullrank: the default is expected to REVEAL this fixture's over-declared ranks \
         (measured: bond 3 29 -> 8, bond 4 16 -> 9). If it no longer does, the fixture build \
         changed and this half of the arm must be re-pointed — it is not a no-op control."
    );
    for b in 0..x_pre.ranks.len() {
        for s in fr.space.bond_sectors(b) {
            assert_eq!(
                x_pre.ranks[b][s], fr.x.ranks[b][s],
                "lih_fullrank bond {b} sector ({},{}): preserve_grid must carry the DECLARED grid \
                 through unchanged even when it is over-declared — that is the whole point of it",
                s.nup, s.ndn
            );
        }
    }
    assert!(
        (l_def - l_pre).abs() <= 1e-9 * l_def.abs().max(1.0),
        "lih_fullrank: the two modes must reach the same λ to roundoff — carrying the null \
         declared directions must not change the answer: {l_def:.17e} vs {l_pre:.17e}"
    );
}

#[test]
fn tjcas_a12e_joint_scf_tt_preserve_grid_plumbing() {
    let fx = h2o_cas86();
    let base = JointScfOpts {
        pairs: Some(cas_pairs(&fx.part, fx.space.d)),
        orb_max_iter: 1,
        cg_max_iter_initial: 5,
        cg_skip_factor: 0.0,
        construction: Construction::Median,
        ..JointScfOpts::default()
    };
    let out_def = joint_ground_state(
        &fx.ints,
        fx.space,
        &fx.x,
        &JointScfOpts {
            tt_preserve_grid: false,
            ..base.clone()
        },
    );
    let out_pre = joint_ground_state(
        &fx.ints,
        fx.space,
        &fx.x,
        &JointScfOpts {
            tt_preserve_grid: true,
            ..base.clone()
        },
    );
    let diff_def = grid_diff(&fx.space, &out_def.x.ranks, &fx.grid);
    let diff_pre = grid_diff(&fx.space, &out_pre.x.ranks, &fx.grid);
    eprintln!(
        "tjcas_a12e h2o cold leg: default totals {:?} (diff {}), preserve totals {:?} (diff {}), \
         pinned {:?}; iters {} / {}, tt_gradnorms {:?} / {:?}",
        grid_totals(&out_def.x.ranks),
        fmt_grid_diff(&diff_def),
        grid_totals(&out_pre.x.ranks),
        fmt_grid_diff(&diff_pre),
        grid_totals(&fx.grid),
        out_def.info.iters,
        out_pre.info.iters,
        out_def.info.tt_gradnorms,
        out_pre.info.tt_gradnorms,
    );
    assert!(
        !diff_def.is_empty(),
        "h2o: with tt_preserve_grid = false the TT leg's CG must still drop the deficient bond-4 \
         ranks (this is the default's contract, tjcas_a11) — if it does not, the knob's default \
         changed or the leg stopped calling CG at all (check cg_skip_factor)"
    );
    assert!(
        diff_pre.is_empty(),
        "h2o: with tt_preserve_grid = true the TT leg must carry the declared grid through the \
         cold-leg riemannian_cg site in tt_leg: {}. This half only certifies that site — \
         `orb_max_iter = 1` means no orbital step runs, so the mid-line-search re-entry \
         (tt_leg's second riemannian_cg call) is never reached here; that forward remains wired but structurally \
         unexercised by this suite (only fires when a warm leg's polish misses `polish_tol`).",
        fmt_grid_diff(&diff_pre)
    );

    let fl = lih_cas22();
    let out = joint_ground_state(
        &fl.ints,
        fl.space,
        &fl.x,
        &JointScfOpts {
            pairs: Some(cas_pairs(&fl.part, fl.space.d)),
            orb_max_iter: 4,
            cg_skip_factor: 0.0,
            construction: Construction::Median,
            tt_preserve_grid: true,
            ..JointScfOpts::default()
        },
    );
    out.x.check();
    let diff = grid_diff(&fl.space, &out.x.ranks, &fl.grid);
    eprintln!(
        "tjcas_a12e lih: iters {} converged {} refused {:?} λ+e0 = {:.10} totals {:?} \
         trials_per_iter {:?} ls_build_count {}",
        out.info.iters,
        out.info.converged,
        out.info.refused_reason,
        out.lambda + fl.ints.e0,
        grid_totals(&out.x.ranks),
        out.info.trials_per_iter,
        out.info.ls_build_count,
    );
    assert!(
        out.info.iters >= 1,
        "lih: the SCF must take at least one orbital iteration, or the knob was never exercised \
         past the cold leg"
    );
    assert!(
        !out.info.tt_gradnorms.is_empty(),
        "lih: no TT leg ran — this arm is vacuous"
    );
    assert!(
        diff.is_empty(),
        "lih: joint_ground_state under tt_preserve_grid must return a grid-exact state: {}",
        fmt_grid_diff(&diff)
    );
}
