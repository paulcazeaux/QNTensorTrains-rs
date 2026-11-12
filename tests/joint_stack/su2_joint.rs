// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::Arc;

use qntt::coeff::CoeffTensor;
use qntt::hamiltonian::SparseHamiltonian;
use qntt::integrals::{Integrals, IntegralsOptions, PropertyOperator, load_fixture_dir};
use qntt::pointgroup::{Irrep, PointGroup, PointGroupKind};
use qntt::qn::QnSpace;
use qntt::rdm::transition_1rdm;
use qntt::rotation::{
    PairMask, expm_antisym, orbital_gradient as orbital_gradient_ab, orbital_pairs,
};
use qntt::round::RoundMode;
use qntt::scalar::Scalar;
use qntt::solvers::InnerProductVector;
use qntt::solvers::hess::{NewtonPolishOpts, SymplecticDavidsonOpts, SymplecticSpectrum};
use qntt::solvers::joint::{
    DynamicResponseOpts, FcRoute, JointOpts, JointScfOpts, JointSolver, PropRoute, ResponseError,
    StaticResponseOpts, TransitionMomentOpts, TransitionMoments, ZRoute,
};
use qntt::su2::core::{Family, Su2Core};
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh};
use qntt::su2::hess::{newton_polish, symplectic_eigenvalues};
use qntt::su2::joint::{
    Su2JointContext, Su2JointGateError, Su2JointScfOut, Su2JointSpec, Su2JointVector,
    su2_joint_context, su2_joint_ground_state, su2_joint_lowest_davidson, su2_linear_response,
    su2_polarizability_static, su2_property_gradient, su2_property_value, su2_transition_moments,
};
use qntt::su2::ops::dot as su2_dot;
use qntt::su2::orbital::{e_pq, kappa_apply, kappa_operator, orbital_gradient};
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks};
use qntt::su2::riemannian::{RiemannianOpts, normalize_mut, riemannian_cg, tangent_bases};
use qntt::su2::solvers::als::als;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::transition1rdm::{su2_transition_1rdm, su2_transition_1rdm_h};
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::tangent_basis;
use qntt::su2::{centerweight, expand, theta};
use qntt::vector::TtVector;

use crate::su2_util::c2v_table;
use crate::su2_util::spin_free_tv_symmetric;
use crate::su2_util::su2_grad_norm;
use crate::su2_util::su2_minimizer;
use crate::util::coeff2_flat;
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::pg::su2_capped_ranks;
use qntt_testkit::randomized_su2_state;
use qntt_testkit::seeded;
use rand::RngExt;

fn count_theta_cw<T: Scalar>(x: &Su2TtVector<T>) -> (usize, usize) {
    let space = x.space;
    let d = space.d;
    let mut n_theta = 0usize;
    let mut n_cw = 0usize;
    for b in 0..d {
        for m in space.bond_sectors(b) {
            if x.rank(b, m) == 0 {
                continue;
            }
            for succ in space.bond_sectors(b + 1) {
                if x.rank(b + 1, succ) == 0 {
                    continue;
                }
                if (theta(m.two_s, succ.two_s) - 1.0).abs() > 1e-12 {
                    n_theta += 1;
                }
            }
        }
    }
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            if x.rank(b, s) == 0 {
                continue;
            }
            if (centerweight(s.two_s, space.two_s) - 1.0).abs() > 1e-12 {
                n_cw += 1;
            }
        }
    }
    (n_theta, n_cw)
}

fn assert_su2core_exact_eq<T: Scalar>(a: &Su2Core<T>, b: &Su2Core<T>, label: &str) {
    assert_eq!(
        a.num_coeffs(),
        b.num_coeffs(),
        "{label}: num_coeffs differs"
    );
    for i in 0..a.num_coeffs() {
        assert_eq!(a.coeff(i), b.coeff(i), "{label}: coeff {i} differs");
    }
}

fn assert_su2tt_exact_eq<T: Scalar>(a: &Su2TtVector<T>, b: &Su2TtVector<T>, label: &str) {
    assert_eq!(a.space.d, b.space.d, "{label}: d differs");
    for k in 0..a.space.d {
        assert_su2core_exact_eq(a.core(k), b.core(k), &format!("{label} core {k}"));
    }
}

fn flat_component_dot<T: Scalar>(a: &Su2Core<T>, b: &Su2Core<T>) -> T {
    let mut acc = T::zero();
    for f in Family::ALL {
        for l in a.transitions(f).collect::<Vec<_>>() {
            let (Some(ba), Some(bb)) = (a.block(f, l), b.block(f, l)) else {
                continue;
            };
            for j in 0..ba.ncols().min(bb.ncols()) {
                for i in 0..ba.nrows().min(bb.nrows()) {
                    acc += ba.get(i, j).conj_s() * *bb.get(i, j);
                }
            }
        }
    }
    acc
}

fn flat_tangent_dot(a: &Su2TtTangent<f64>, b: &Su2TtTangent<f64>) -> f64 {
    let d = a.rank().len().saturating_sub(1);
    (0..d)
        .map(|k| flat_component_dot(a.component(k), b.component(k)))
        .sum()
}

fn su2_default_joint_opts() -> JointOpts {
    JointOpts {
        tt_grad_tol: 1e-4,
        orb_floor: 1e-8,
        pairs: None,
        z_route: ZRoute::Explicit,
        b_general: false,
    }
}

struct LihFixture {
    space: Su2Space,
    h: Su2SparseHamiltonian<f64>,
    x: Su2TtVector<f64>,
}

static LIH_WORKHORSE_X: std::sync::Mutex<Option<Su2TtVector<f64>>> = std::sync::Mutex::new(None);

fn su2_lih_workhorse_fixture() -> LihFixture {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let d = ints.norb;
    let n = 3usize;
    let two_s = 1usize;
    let space = Su2Space::new(d, n, two_s);
    let h = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    let cached = LIH_WORKHORSE_X.lock().unwrap().clone();

    let msu2 = 2usize;
    let ranks = su2_capped_ranks(&space, msu2);
    let mut rng = seeded(0x5117_1157);
    let x = if let Some(x) = cached {
        x
    } else {
        let x = {
            let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, msu2, &mut rng);
            let (_lals, xa) = als(
                &h,
                &x0,
                &qntt::solvers::als::AlsOptions {
                    eps: 1e-10,
                    max_iter: 60,
                    ..qntt::solvers::als::AlsOptions::default()
                },
            );
            let mut x = xa;
            x.round_ranks_mut(&ranks);
            normalize_mut(&mut x);

            let mut opts = RiemannianOpts::new(&space);
            opts.eps = 1e-13;
            opts.max_iter = 600;
            opts.gradtol = 1e-13;
            let polish = |y0: &Su2TtVector<f64>| -> Su2TtVector<f64> {
                let (_l, mut y) = riemannian_cg(&h, y0, &opts);
                for _ in 0..30 {
                    if su2_grad_norm(&h, &y) < 1e-10 {
                        break;
                    }
                    let (_l2, y2) = riemannian_cg(&h, &y, &opts);
                    y = y2;
                }
                y
            };
            let mut x = polish(&x);
            x.round_mut(1e-8, qntt::Pooled::NONE, RoundMode::Rel);
            x.round_ranks_mut(&ranks);
            normalize_mut(&mut x);
            let x = polish(&x);
            let (mut x, _info) = newton_polish(
                &h,
                &x,
                &NewtonPolishOpts {
                    gradtol: 1e-11,
                    ..NewtonPolishOpts::default()
                },
            );
            normalize_mut(&mut x);
            x
        };
        *LIH_WORKHORSE_X.lock().unwrap() = Some(x.clone());
        x
    };
    assert!(x.check(), "su2_lih_workhorse_fixture: check() failed");

    let gnorm = su2_grad_norm(&h, &x);
    assert!(
        gnorm < 1e-6,
        "su2_lih_workhorse_fixture: not TT-critical enough (‖grad‖={gnorm:.3e}); \
         tighten the polish loop or lower msu2"
    );
    let g_orb = orbital_gradient(&h, &x);
    let g_orb_norm: f64 = orbital_pairs(space.pg, d)
        .iter()
        .map(|&(p, q)| g_orb.get(&[p, q]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        g_orb_norm > 1e-6,
        "su2_lih_workhorse_fixture: orbital gradient is VACUOUS (‖g_orb‖={g_orb_norm:.3e}) — \
         msu2={msu2} likely collapsed to (near-)an exact eigenstate; lower msu2, per the msu2 \
         re-measurement note in the module doc"
    );

    LihFixture { space, h, x }
}

fn kappa_gamma_fixture(
    two_s: usize,
    seed: u64,
) -> (Su2Space, Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
    use std::collections::HashMap;
    use std::sync::Mutex;
    #[allow(clippy::type_complexity)]
    static CACHE: Mutex<Option<HashMap<(usize, u64), Su2TtVector<f64>>>> = Mutex::new(None);

    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 3, two_s);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    if let Some(x) = CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .get(&(two_s, seed))
        .cloned()
    {
        return (space, h, x);
    }
    let x = su2_minimizer(&h, space, seed, 1, 2);
    CACHE
        .lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert((two_s, seed), x.clone());
    (space, h, x)
}

fn random_kappa(d: usize, pg: PointGroup, seed: u64) -> CoeffTensor<f64> {
    let mut rng = seeded(seed);
    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q) in &orbital_pairs(pg, d) {
        let val: f64 = rng.random_range(-0.3..0.3);
        k.set(&[p, q], val);
        k.set(&[q, p], -val);
    }
    k
}

#[test]
fn tsu_kappa_expand_matches_abelian_apply() {
    let (space, _h, x) = kappa_gamma_fixture(1, 0xCA5A);
    let d = space.d;
    let k = random_kappa(d, space.pg, 0xCA5B);

    let su2_result = kappa_apply(&k, &x);

    let two_s = space.two_s as i64;
    let mut two_sz = -two_s;
    let mut checked = 0usize;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let ab_x: TtVector<f64> = expand(&x, two_sz);
            let ab_space = ab_x.space;
            let ab_result = qntt::rotation::kappa_apply(&k, &ab_x);
            let su2_expanded: TtVector<f64> = expand(&su2_result, two_sz);
            assert_eq!(
                su2_expanded.space, ab_space,
                "expand space mismatch at sz={two_sz}"
            );

            let a = su2_expanded.to_dense();
            let b = ab_result.to_dense();
            let mut num = 0.0f64;
            let mut den = 0.0f64;
            for i in 0..a.len() {
                num += (a[i] - b[i]).powi(2);
                den += b[i].powi(2);
            }
            let rel = (num / den.max(1e-300)).sqrt();
            assert!(
                rel < 1e-12,
                "kappa expand mismatch at two_sz={two_sz}: rel={rel:e}"
            );
            checked += 1;
        }
        two_sz += 1;
    }
    assert_eq!(
        checked, 2,
        "twoS=1 must yield exactly two admissible sz arms"
    );
}

#[test]
fn tsu_kappa_exact_linearity_at_s_1e_minus_15() {
    let (space, _h, x) = kappa_gamma_fixture(1, 0xCA5A);
    let k = random_kappa(space.d, space.pg, 0xCA5C);
    let s = 1e-15f64;

    let mut ks = CoeffTensor::<f64>::zeros(space.d, 2);
    for p in 0..space.d {
        for q in 0..space.d {
            ks.set(&[p, q], s * k.get(&[p, q]));
        }
    }
    let lhs = kappa_apply(&ks, &x);
    let mut rhs = kappa_apply(&k, &x);
    rhs.scale_mut(s);

    let da = expand(&lhs, space.two_s as i64).to_dense();
    let db = expand(&rhs, space.two_s as i64).to_dense();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..da.len() {
        num += (da[i] - db[i]).powi(2);
        den += db[i].powi(2);
    }
    let rel = (num / den.max(1e-300)).sqrt();
    assert!(
        rel < 1e-12,
        "kappa_apply not exactly linear at s=1e-15: rel={rel:e} (eps must be EXACTLY 0.0, not \
         f64::EPSILON — the latter would drop the whole 1e-15-scaled generator silently)"
    );
}

#[test]
fn tsu_kappa_zero_term_free_build() {
    let (space, _h, x) = kappa_gamma_fixture(1, 0xCA5A);
    let zero_k = CoeffTensor::<f64>::zeros(space.d, 2);
    let result = kappa_apply(&zero_k, &x);
    let dense = expand(&result, space.two_s as i64).to_dense();
    assert!(
        dense.iter().all(|&v| v == 0.0),
        "kappa_apply(0, x) must be the EXACT zero train"
    );
    let op = kappa_operator(space, &zero_k);
    let via_op = op.apply(&x);
    assert_su2tt_exact_eq(&via_op, &result, "kappa_operator vs kappa_apply at kappa=0");
}

#[test]
fn tsu_kappa_e_pq_matches_adag_i_a_j() {
    let (_space, _h, x) = kappa_gamma_fixture(1, 0xCA5A);
    let direct = e_pq(&x, 0, 1, 1.0);
    let via_tensorops = qntt::su2::tensorops::adag_i_a_j(&x, 0, 1);
    assert_su2tt_exact_eq(&direct, &via_tensorops, "e_pq vs adag_i_a_j");
}

#[test]
#[should_panic(expected = "kappa has a nonzero entry k[")]
fn tsu_kappa_cross_irrep_entry_refused() {
    let d = 4usize;
    let pg = c2v_table(d);
    let space = Su2Space::with_point_group(d, 3, 1, pg, Irrep::IDENTITY);
    assert_ne!(pg.irrep_of(0), pg.irrep_of(1));
    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    k.set(&[0, 1], 0.3);
    k.set(&[1, 0], -0.3);
    let _ = kappa_operator(space, &k);
}

#[test]
fn tsu_gamma_matches_abelian_transition_1rdm() {
    let (space, _h, x) = kappa_gamma_fixture(1, 0xC000);
    let y = {
        let mut rng = seeded(0xC001);
        let mut y: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        y.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        let n = y.norm_lr(false);
        y.scale_mut(1.0 / n);
        y
    };
    let gamma_su2 = su2_transition_1rdm(&x, &y);

    let two_s = space.two_s as i64;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let ax: TtVector<f64> = expand(&x, two_sz);
            let ay: TtVector<f64> = expand(&y, two_sz);
            let gamma_ab = transition_1rdm(&ax, &ay);
            for p in 0..space.d {
                for q in 0..space.d {
                    let a = gamma_su2.get(&[p, q]);
                    let b = gamma_ab.get(&[p, q]);
                    assert!(
                        (a - b).abs() < 1e-12 * b.abs().max(1.0),
                        "gamma[{p},{q}] mismatch at sz={two_sz}: su2={a} abelian={b}"
                    );
                }
            }
        }
        two_sz += 1;
    }
}

#[test]
fn tsu_gamma_h_matches_abelian_transition_1rdm_h() {
    let (space, h, x) = kappa_gamma_fixture(1, 0xC010);
    let w0 = {
        let mut rng = seeded(0xC011);
        let mut w: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        w.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        w
    };
    let gamma_su2 = su2_transition_1rdm_h(&x, &h, &w0);

    let two_s = space.two_s as i64;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let ax: TtVector<f64> = expand(&x, two_sz);
            let aw0: TtVector<f64> = expand(&w0, two_sz);
            let h_ab = SparseHamiltonian::new(
                ax.space,
                &kappa_gamma_ints_t(space.d),
                &kappa_gamma_ints_v(space.d),
                None,
                f64::EPSILON,
            );
            let aw: TtVector<f64> = h_ab.apply(&aw0);
            let gamma_ab = transition_1rdm(&ax, &aw);

            for p in 0..space.d {
                for q in 0..space.d {
                    let a = gamma_su2.get(&[p, q]);
                    let b = gamma_ab.get(&[p, q]);
                    assert!(
                        (a - b).abs() < 1e-12 * b.abs().max(1.0),
                        "gamma_h[{p},{q}] mismatch at sz={two_sz}: su2={a} abelian={b}"
                    );
                }
            }
        }
        two_sz += 1;
    }
}

#[test]
fn tsu_orbital_gradient_matches_abelian_through_expand() {
    let (space, h, x) = kappa_gamma_fixture(1, 0xC030);
    let g_su2 = orbital_gradient(&h, &x);

    let two_s = space.two_s as i64;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let ax: TtVector<f64> = expand(&x, two_sz);
            let ab_space = ax.space;
            let h_ab = SparseHamiltonian::new(
                ab_space,
                &kappa_gamma_ints_t(space.d),
                &kappa_gamma_ints_v(space.d),
                None,
                f64::EPSILON,
            );
            let g_ab = orbital_gradient_ab(&h_ab, &ax);
            for &(p, q) in &orbital_pairs(space.pg, space.d) {
                let a = g_su2.get(&[p, q]);
                let b = g_ab.get(&[p, q]);
                assert!(
                    (a - b).abs() < 1e-10 * b.abs().max(1.0),
                    "orbital_gradient[{p},{q}] mismatch at sz={two_sz}: su2={a} abelian={b}"
                );
            }
        }
        two_sz += 1;
    }
}

fn kappa_gamma_ints_t(d: usize) -> CoeffTensor<f64> {
    generic_truncated_tv(d).0
}
fn kappa_gamma_ints_v(d: usize) -> CoeffTensor<f64> {
    generic_truncated_tv(d).1
}

#[test]
fn tsu_orbital_gradient_structural_antisymmetry_and_zero_diagonal() {
    let (space, h, x) = kappa_gamma_fixture(1, 0xC040);
    let g = orbital_gradient(&h, &x);
    for p in 0..space.d {
        assert_eq!(
            g.get(&[p, p]),
            0.0,
            "diagonal must be exactly zero at p={p}"
        );
        for q in 0..space.d {
            if p == q {
                continue;
            }
            assert_eq!(
                g.get(&[p, q]),
                -g.get(&[q, p]),
                "g[{p},{q}] must be the EXACT fp negative of g[{q},{p}]"
            );
        }
    }
    let norm: f64 = orbital_pairs(space.pg, space.d)
        .iter()
        .map(|&(p, q)| g.get(&[p, q]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(norm > 1e-4, "orbital gradient vacuous at fixture: {norm:e}");
}

#[test]
fn tsu_orbital_gradient_cross_irrep_exact_zero() {
    let d = 6usize;
    let pg = c2v_table(d);
    let (t, v) = spin_free_tv_symmetric(d, pg);
    let space = Su2Space::with_point_group(d, 3, 1, pg, Irrep::IDENTITY);
    let caps = max_ranks(&space);
    if caps[d][Su2Sector::with_irrep(3, 1, Irrep::IDENTITY)] == 0 {
        return;
    }
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&h, space, 0xC041, 1, 2);
    let g = orbital_gradient(&h, &x);
    let pairs = orbital_pairs(pg, d);
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert_eq!(
                    g.get(&[p, q]),
                    0.0,
                    "cross-irrep g[{p},{q}] must be exactly 0.0 (Γ_{p}={:?} != Γ_{q}={:?})",
                    pg.irrep_of(p),
                    pg.irrep_of(q)
                );
            }
        }
    }
    assert!(
        !pairs.is_empty(),
        "orbital_pairs must be non-vacuous under c2v_table"
    );
}

#[test]
fn tsu_apply_context_builds_and_iso_certificate_holds() {
    let fx = su2_lih_workhorse_fixture();
    let ctx: Su2JointContext<'_> = su2_joint_context(&fx.h, &fx.x, &su2_default_joint_opts())
        .unwrap_or_else(|e| panic!("su2_joint_context refused: {e}"));

    assert!(ctx.n_eff() >= 1, "workhorse fixture must have n_eff >= 1");
    assert!(
        ctx.n_eff() < ctx.pairs().len(),
        "workhorse fixture must NOT be full-rank (that is the §6 degeneration arm)"
    );
    let iso = ctx.iso_defect();
    assert!(
        iso < 1e-8,
        "ι-isometry defect too large: {iso:e} (ISO_TOL=1e-8)"
    );

    let tau = ctx.orbital_projections();
    let n = tau.len().min(12);
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let cw = tau[i].dot(&tau[j]);
            let amb = su2_dot(&tau[i].to_vector(), &tau[j].to_vector());
            worst = worst.max((cw - amb).abs());
        }
    }
    assert!(
        worst < 1e-6,
        "independently recomputed ι-isometry defect over orbital_projections() too large: \
         {worst:e}"
    );

    let (_n_theta, n_cw) = count_theta_cw(&fx.x);
    assert!(
        n_cw >= 2,
        "workhorse fixture must be centerweight-non-vacuous"
    );
}

#[test]
fn tsu_apply_a_b_covectors_match_expand_of_abelian() {
    let (space, h, x) = kappa_gamma_fixture(1, 0xC050);
    let ctx = su2_joint_context(&h, &x, &su2_default_joint_opts())
        .unwrap_or_else(|e| panic!("su2_joint_context refused: {e}"));

    let two_s = space.two_s as i64;
    let ax: TtVector<f64> = expand(&x, two_s);
    let ab_space = ax.space;
    let h_ab = SparseHamiltonian::new(
        ab_space,
        &kappa_gamma_ints_t(space.d),
        &kappa_gamma_ints_v(space.d),
        None,
        f64::EPSILON,
    );
    let ab_opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..su2_default_joint_opts()
    };
    let ab_ctx = qntt::solvers::joint::joint_context(&h_ab, &ax, &ab_opts)
        .unwrap_or_else(|e| panic!("abelian joint_context refused: {e}"));

    assert_eq!(
        ctx.pairs(),
        ab_ctx.pairs(),
        "pair lists must match (same pg, same d)"
    );

    let n_orb = ctx.pairs().len();
    let mut rng = seeded(0xC051);
    let a_dir: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();

    let (bl, br) = ctx.bases();
    let xi_dir = {
        let mut d: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        d.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        Su2TtTangent::project(bl, br, &d)
    };
    let (bl_ab, br_ab) = ab_ctx.bases();
    let xi_dir_ab = {
        let d_expanded: TtVector<f64> = expand(&xi_dir.to_vector(), two_s);
        qntt::core::tangent::TtTangent::project(bl_ab, br_ab, &d_expanded)
    };

    let (f_xi_su2, f_c_su2) = ctx.a_covector(&xi_dir, &a_dir, FcRoute::Dots);
    let (f_xi_ab, f_c_ab) =
        ab_ctx.a_covector(&xi_dir_ab, &a_dir, qntt::solvers::joint::FcRoute::Dots);

    let su2_expanded: TtVector<f64> = expand(&f_xi_su2.to_vector(), two_s);
    let a = su2_expanded.to_dense();
    let b = f_xi_ab.to_vector().to_dense();
    let mut ip = 0.0f64;
    let mut na = 0.0f64;
    for i in 0..a.len() {
        ip += a[i] * b[i];
        na += a[i] * a[i];
    }
    let nesting_rel = (ip - na).abs() / na.max(1e-300);
    assert!(
        nesting_rel < 1e-8,
        "A-covector F_xi nesting defect: |<iota(F_xi_su2),F_xi_ab> - ||iota(F_xi_su2)||^2| / \
         ||iota(F_xi_su2)||^2 = {nesting_rel:e}"
    );

    for i in 0..n_orb {
        assert!(
            (f_c_su2[i] - f_c_ab[i]).abs() < 1e-13 * f_c_ab[i].abs().max(1.0),
            "A-covector F_c[{i}] mismatch: su2={} abelian={}",
            f_c_su2[i],
            f_c_ab[i]
        );
    }
}

#[test]
fn tsu_apply_c1_selfadjoint() {
    let fx = su2_lih_workhorse_fixture();
    let ctx = su2_joint_context(&fx.h, &fx.x, &su2_default_joint_opts())
        .unwrap_or_else(|e| panic!("su2_joint_context refused: {e}"));
    let spec: &Arc<Su2JointSpec> = ctx.spec();
    let n = ctx.n_eff();
    assert!(n >= 1, "workhorse must have n_eff >= 1 for a mixed vector");

    let mut rng = seeded(0xC060);
    let mut vecs = Vec::new();
    for _ in 0..2 {
        let mut d: Su2TtVector<f64> = randomized_su2_state(fx.space, 1, 2, &mut rng);
        d.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        let (bl, br) = ctx.bases();
        let xi = Su2TtTangent::project(bl, br, &d);
        vecs.push(Su2JointVector::from_tangent(spec, xi));
    }
    for _ in 0..2 {
        let c: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        vecs.push(Su2JointVector::from_coeffs(spec, &ctx, c));
    }
    for _ in 0..2 {
        let mut d: Su2TtVector<f64> = randomized_su2_state(fx.space, 1, 2, &mut rng);
        d.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
        let (bl, br) = ctx.bases();
        let xi = Su2TtTangent::project(bl, br, &d);
        let mut v = Su2JointVector::from_tangent(spec, xi);
        let c: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
        v.c = c;
        vecs.push(v);
    }

    let mut worst_a = 0.0f64;
    let mut scale_a = 0.0f64;
    let mut worst_b = 0.0f64;
    let mut scale_b = 0.0f64;
    for i in 0..vecs.len() {
        for j in (i + 1)..vecs.len() {
            let aij = vecs[i].dot(&ctx.a_apply(&vecs[j]));
            let aji = vecs[j].dot(&ctx.a_apply(&vecs[i]));
            worst_a = worst_a.max((aij - aji).abs());
            scale_a = scale_a.max(aij.abs()).max(aji.abs());
            let bij = vecs[i].dot(&ctx.b_apply(&vecs[j]));
            let bji = vecs[j].dot(&ctx.b_apply(&vecs[i]));
            worst_b = worst_b.max((bij - bji).abs());
            scale_b = scale_b.max(bij.abs()).max(bji.abs());
        }
    }
    assert!(
        scale_b > 1e-8,
        "B vacuous in self-adjointness arm: scale {scale_b:.3e}"
    );
    assert!(
        worst_a <= 1e-8 * scale_a.max(1.0),
        "A not self-adjoint: {worst_a:.3e}"
    );
    assert!(
        worst_b <= 1e-6 * scale_b.max(1.0),
        "B not self-adjoint: {worst_b:.3e}"
    );
}

#[test]
fn tsu_apply_singlet_spectrum_cross_check() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);

    let sp22 = QnSpace::new(d, 2, 2);
    let h22 = SparseHamiltonian::new(sp22, &t, &v, None, f64::EPSILON);
    let sp31 = QnSpace::new(d, 3, 1);
    let h31 = SparseHamiltonian::new(sp31, &t, &v, None, f64::EPSILON);

    let ev22 = exact_config_spectrum(sp22, &h22);
    let ev31 = exact_config_spectrum(sp31, &h31);
    let singlets: Vec<f64> = ev22
        .iter()
        .copied()
        .filter(|e| ev31.iter().all(|p| (e - p).abs() > 1e-8 * e.abs().max(1.0)))
        .collect();
    assert_eq!(
        singlets.len(),
        20,
        "singlet-labeling sanity (same as su2_h8)"
    );

    let space = Su2Space::new(d, 4, 0);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&h, space, 0xC070, 1, 2);
    let ctx = su2_joint_context(
        &h,
        &x,
        &JointOpts {
            tt_grad_tol: f64::INFINITY,
            ..su2_default_joint_opts()
        },
    )
    .unwrap_or_else(|e| panic!("su2_joint_context refused: {e}"));

    let (basis, _report) = ctx.joint_basis();
    assert!(!basis.is_empty(), "joint basis must be non-vacuous at d=4");
    let spectrum = ctx.joint_symplectic_eigenvalues(&basis);
    match spectrum {
        SymplecticSpectrum::Stable { omega } => {
            assert!(
                omega.iter().all(|&w| w > -1e-6),
                "stable spectrum must be non-negative: {omega:?}"
            );
        }
        SymplecticSpectrum::Resonant { .. } | SymplecticSpectrum::Unstable { .. } => {}
    }
    assert!(
        su2_rayleigh(&h, &x) > singlets[0],
        "fixed-rank energy must sit above the exact singlet ground state"
    );
}

fn exact_config_spectrum(space: QnSpace, ham: &SparseHamiltonian<f64>) -> Vec<f64> {
    let d = space.d;
    let combos = |n: usize| -> Vec<Vec<usize>> { combinations(d, n) };
    let states: Vec<TtVector<f64>> = combos(space.nup)
        .into_iter()
        .flat_map(|up| {
            combos(space.ndn)
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
            qntt::tt_state(&occ_up, &occ_dn)
        })
        .collect();
    let n = states.len();
    let mut hmat = vec![vec![0.0f64; n]; n];
    for (j, sj) in states.iter().enumerate() {
        let hs = ham * sj;
        for (i, si) in states.iter().enumerate() {
            hmat[i][j] = qntt::dot(si, &hs);
        }
    }
    sym_eigvals_jacobi(hmat)
}

fn combinations(d: usize, n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    if n > d {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut idx: Vec<usize> = (0..n).collect();
    loop {
        out.push(idx.clone());
        let mut i = n;
        loop {
            if i == 0 {
                return out;
            }
            i -= 1;
            if idx[i] != i + d - n {
                break;
            }
        }
        idx[i] += 1;
        for j in (i + 1)..n {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

#[allow(clippy::needless_range_loop)]
fn sym_eigvals_jacobi(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < 1e-13 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];
                a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
            }
        }
    }
    let mut evs: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    evs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    evs
}

#[test]
fn tsu_apply_davidson_matches_dense() {
    let (space, h, x) = kappa_gamma_fixture(1, 0xDA71);
    let ctx = su2_joint_context(
        &h,
        &x,
        &JointOpts {
            tt_grad_tol: f64::INFINITY,
            ..su2_default_joint_opts()
        },
    )
    .unwrap_or_else(|e| panic!("context refused: {e}"));

    let (basis, _report) = ctx.joint_basis();
    assert!(!basis.is_empty(), "basis must be non-vacuous");
    let dense = ctx.joint_symplectic_eigenvalues(&basis);
    let SymplecticSpectrum::Stable { omega: dense_omega } = dense else {
        return;
    };
    let k = dense_omega.len().clamp(1, 3);

    let spec = ctx.spec();
    let mut rng = seeded(0xDA72);
    let v0s: Vec<Su2JointVector> = (0..(k + 2).min(basis.len()))
        .map(|_| {
            let mut d: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
            d.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
            let (bl, br) = ctx.bases();
            let xi = Su2TtTangent::project(bl, br, &d);
            let mut v = Su2JointVector::from_tangent(spec, xi);
            for c in v.c.iter_mut() {
                *c = rng.random_range(-1.0..1.0);
            }
            v
        })
        .collect();

    let davidson_opts = SymplecticDavidsonOpts::default();
    match su2_joint_lowest_davidson(&ctx, &v0s, k, &davidson_opts) {
        Ok((omega, info)) => {
            assert!(
                info.converged,
                "su2_joint_lowest_davidson must report converged"
            );
            for i in 0..k {
                assert!(
                    (omega[i] - dense_omega[i]).abs() < 1e-6 * dense_omega[i].abs().max(1.0),
                    "davidson vs dense omega[{i}]: {} vs {}",
                    omega[i],
                    dense_omega[i]
                );
            }
        }
        Err(e) => panic!("su2_joint_lowest_davidson failed: {e:?}"),
    }
}

fn dense_rayleigh(h_ab: &SparseHamiltonian<f64>, e: &TtVector<f64>) -> f64 {
    let hy = (h_ab * e).to_dense();
    let ee = e.to_dense();
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..ee.len() {
        num += ee[i] * hy[i];
        den += ee[i] * ee[i];
    }
    num / den
}

#[test]
fn tsu_mutation_fd_through_expand_at_joint_direction() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 3, 1);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xC080);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    x.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let nx = x.norm_lr(false);
    x.scale_mut(1.0 / nx);

    let pairs = orbital_pairs(space.pg, d);
    let n_orb = pairs.len();

    let mut rng = seeded(0xC084);
    let a_dir: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-0.5..0.5)).collect();
    let (bl, br) = tangent_bases(&x);
    let mut xi_dir_vec: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    xi_dir_vec.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let n_norm = xi_dir_vec.norm_lr(false);
    xi_dir_vec.scale_mut(1.0 / n_norm);
    let xi_dir = Su2TtTangent::project(&bl, &br, &xi_dir_vec);

    let (n_theta, n_cw) = count_theta_cw(&xi_dir.to_vector());
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous: theta/centerweight trivial (n_theta={n_theta}, n_cw={n_cw})"
    );

    let mut kappa = CoeffTensor::<f64>::zeros(d, 2);
    for (&(p, q), &ai) in pairs.iter().zip(&a_dir) {
        kappa.set(&[p, q], ai);
        kappa.set(&[q, p], -ai);
    }
    let kflat = coeff2_flat(&kappa, d);

    let (t0, v0) = (kappa_gamma_ints_t(d), kappa_gamma_ints_v(d));
    let two_sz = space.two_s as i64;

    let energy = |s: f64| -> f64 {
        let step: Vec<f64> = kflat.iter().map(|&v| -s * v).collect();
        let u = expm_antisym(&step, d);
        let ints_s = Integrals::<f64> {
            t: t0.clone(),
            v: v0.clone(),
            e0: 0.0,
            norb: d,
            nelec: space.n,
            ms2: 0,
            orbsym: Vec::new(),
            isym: None,
        };
        let rotated = qntt::rotation::rotate_integrals(&ints_s, &u);
        let mut xs = xi_dir.retract_hosvd(s);
        normalize_mut(&mut xs);
        let axs: TtVector<f64> = expand(&xs, two_sz);
        let ab_space = axs.space;
        let h_ab_s = SparseHamiltonian::new(ab_space, &rotated.t, &rotated.v, None, f64::EPSILON);
        dense_rayleigh(&h_ab_s, &axs)
    };

    let hstep = 1e-5f64;
    let slope_fd = (energy(hstep) - energy(-hstep)) / (2.0 * hstep);

    let (bl0, br0) = tangent_bases(&x);
    let lam0 = su2_rayleigh(&h, &x);
    let hx0 = h.apply(&x);
    let res0 = &hx0 - &(&x * lam0);
    let g0 = Su2TtTangent::project(&bl0, &br0, &res0);
    let slope_ana_xi = 2.0 * g0.dot(&xi_dir);
    let g_orb0 = orbital_gradient(&h, &x);
    let gc0: Vec<f64> = pairs.iter().map(|&(p, q)| g_orb0.get(&[p, q])).collect();
    let slope_ana_c: f64 = gc0.iter().zip(&a_dir).map(|(&g, &a)| g * a).sum();
    let slope_ana = slope_ana_xi + slope_ana_c;

    assert!(
        slope_ana.abs() > 0.05,
        "non-vacuity: slope_ana = {slope_ana}"
    );
    assert!(
        (slope_fd - slope_ana).abs() < 1e-3 * slope_ana.abs().max(1.0),
        "joint metric FD linchpin: slope_fd={slope_fd} vs slope_ana={slope_ana} — a wrong \
         rproject (H9-a) or dropped centerweight (H9-b) throws this off by O(1)"
    );
}

#[test]
fn tsu_mutation_dropped_centerweight_breaks_tangent_dot() {
    let (space, _h, x) = kappa_gamma_fixture(1, 0xC090);
    let (bl, br) = tangent_bases(&x);
    let mut rng = seeded(0xC091);
    let mut d1: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    d1.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let mut d2: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    d2.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let xi1 = Su2TtTangent::project(&bl, &br, &d1);
    let xi2 = Su2TtTangent::project(&bl, &br, &d2);

    let (n_theta, n_cw) = count_theta_cw(&xi1.to_vector());
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "vacuous fixture for the centerweight mutation"
    );

    let correct = xi1.dot(&xi2);
    let mutant = flat_tangent_dot(&xi1, &xi2);
    let scale = correct.abs().max(mutant.abs()).max(1e-3);
    assert!(
        (correct - mutant).abs() > 1e-3 * scale,
        "dropped-centerweight mutation did NOT bite: correct={correct} mutant={mutant} \
         (the anti-oracle is blind at this fixture — enlarge ranks/twoS, do not loosen this bound)"
    );
}

#[test]
fn tsu_g1_kappa_apply_bit_identical() {
    let d = 4usize;
    let trivial_space = Su2Space::new(d, 3, 1);
    let explicit_pg = PointGroup::from_orbsym(PointGroupKind::C1, &vec![1u8; d]).expect("C1 table");
    let explicit_space = Su2Space::with_point_group(d, 3, 1, explicit_pg, Irrep::IDENTITY);

    let mut rng = seeded(0xC1DE);
    let x_trivial: Su2TtVector<f64> = randomized_su2_state(trivial_space, 1, 2, &mut rng);
    let mut rng2 = seeded(0xC1DE);
    let x_explicit: Su2TtVector<f64> = randomized_su2_state(explicit_space, 1, 2, &mut rng2);
    assert_su2tt_exact_eq(
        &x_trivial,
        &x_explicit,
        "trivial vs explicit-identity fixture build",
    );

    let k = random_kappa(d, trivial_space.pg, 0xC1DF);
    let y_trivial = kappa_apply(&k, &x_trivial);
    let y_explicit = kappa_apply(&k, &x_explicit);
    assert_su2tt_exact_eq(&y_trivial, &y_explicit, "kappa_apply |G|=1 bit identity");
}

#[test]
fn tsu_g1_orbital_gradient_bit_identical() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let trivial_space = Su2Space::new(d, 3, 1);
    let explicit_pg = PointGroup::from_orbsym(PointGroupKind::C1, &vec![1u8; d]).expect("C1 table");
    let explicit_space = Su2Space::with_point_group(d, 3, 1, explicit_pg, Irrep::IDENTITY);

    let h_trivial = Su2SparseHamiltonian::new(trivial_space, &t, &v, None, f64::EPSILON);
    let h_explicit = Su2SparseHamiltonian::new(explicit_space, &t, &v, None, f64::EPSILON);

    let mut rng_t = seeded(0xC1E0);
    let mut x_trivial: Su2TtVector<f64> = randomized_su2_state(trivial_space, 1, 2, &mut rng_t);
    x_trivial.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let mut rng_e = seeded(0xC1E0);
    let mut x_explicit: Su2TtVector<f64> = randomized_su2_state(explicit_space, 1, 2, &mut rng_e);
    x_explicit.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    assert_su2tt_exact_eq(&x_trivial, &x_explicit, "fixture rebuild sanity");

    let g_trivial = orbital_gradient(&h_trivial, &x_trivial);
    let g_explicit = orbital_gradient(&h_explicit, &x_explicit);
    for p in 0..d {
        for q in 0..d {
            assert_eq!(
                g_trivial.get(&[p, q]),
                g_explicit.get(&[p, q]),
                "orbital_gradient |G|=1 bit identity mismatch at [{p},{q}]"
            );
        }
    }
}

#[test]
fn tsu_g1_joint_context_spectrum_bit_identical() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let trivial_space = Su2Space::new(d, 3, 1);
    let explicit_pg = PointGroup::from_orbsym(PointGroupKind::C1, &vec![1u8; d]).expect("C1 table");
    let explicit_space = Su2Space::with_point_group(d, 3, 1, explicit_pg, Irrep::IDENTITY);

    let h_trivial = Su2SparseHamiltonian::new(trivial_space, &t, &v, None, f64::EPSILON);
    let h_explicit = Su2SparseHamiltonian::new(explicit_space, &t, &v, None, f64::EPSILON);

    let mut rng_t = seeded(0xC1F0);
    let mut x_trivial: Su2TtVector<f64> = randomized_su2_state(trivial_space, 1, 2, &mut rng_t);
    x_trivial.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x_trivial);
    let mut rng_e = seeded(0xC1F0);
    let mut x_explicit: Su2TtVector<f64> = randomized_su2_state(explicit_space, 1, 2, &mut rng_e);
    x_explicit.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    normalize_mut(&mut x_explicit);
    assert_su2tt_exact_eq(&x_trivial, &x_explicit, "fixture rebuild sanity");

    let opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..su2_default_joint_opts()
    };
    let ctx_t = su2_joint_context(&h_trivial, &x_trivial, &opts)
        .unwrap_or_else(|e| panic!("trivial context refused: {e}"));
    let ctx_e = su2_joint_context(&h_explicit, &x_explicit, &opts)
        .unwrap_or_else(|e| panic!("explicit context refused: {e}"));

    assert_eq!(ctx_t.orb_spectrum().len(), ctx_e.orb_spectrum().len());
    for (a, b) in ctx_t.orb_spectrum().iter().zip(ctx_e.orb_spectrum()) {
        assert_eq!(a, b, "orb_spectrum |G|=1 bit identity");
    }
    for (a, b) in ctx_t.orbital_gram().iter().zip(ctx_e.orbital_gram()) {
        assert_eq!(a, b, "orbital_gram (S) |G|=1 bit identity");
    }
}

#[test]
fn tsu_neff_zero_forwards_to_tt() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 3, 1);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let caps = max_ranks(&space);
    let rmax = caps
        .iter()
        .flat_map(|g| g.iter().map(|(_, &r)| r))
        .max()
        .unwrap_or(1);
    let x = su2_minimizer(&h, space, 0xF011, rmax, rmax);

    let opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        orb_floor: 1e-8,
        pairs: None,
        z_route: ZRoute::Explicit,
        b_general: false,
    };
    let ctx = su2_joint_context(&h, &x, &opts).unwrap_or_else(|e| panic!("context refused: {e}"));
    assert_eq!(
        ctx.n_eff(),
        0,
        "full-rank fixture must degenerate to n_eff = 0"
    );
    let anchor = ctx.orb_spectrum().last().copied().unwrap_or(0.0).max(1.0);
    assert!(
        ctx.orb_spectrum().iter().all(|&m| m < 1e-8 * anchor),
        "orb_spectrum must be pure roundoff at n_eff=0: {:?}",
        ctx.orb_spectrum()
    );

    let spec: &Arc<Su2JointSpec> = ctx.spec();
    let (bl, br) = ctx.bases();
    let mut rng = seeded(0xF012);
    let mut d1: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    d1.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
    let xi = Su2TtTangent::project(bl, br, &d1);
    let v = Su2JointVector::from_tangent(spec, xi.clone());

    let applied = ctx.a_apply(&v);
    let direct = ctx.weingarten().a_apply_fused(&xi);
    assert_su2core_exact_eq(
        applied.xi.component(0),
        direct.component(0),
        "a_apply must forward LITERALLY (bit-identical) to a_apply_fused at n_eff=0",
    );
    assert_eq!(applied.c.len(), 0);

    let (basis, _report) = ctx.joint_basis();
    let joint_spec = ctx.joint_symplectic_eigenvalues(&basis);
    let tt_basis = tangent_basis(bl, br, true);
    let tt_spec = symplectic_eigenvalues(ctx.weingarten(), &tt_basis);
    match (joint_spec, tt_spec) {
        (SymplecticSpectrum::Stable { omega: jo }, SymplecticSpectrum::Stable { omega: to }) => {
            assert_eq!(jo.len(), to.len(), "spectrum size must match at n_eff=0");
            for (a, b) in jo.iter().zip(&to) {
                assert!(
                    (a - b).abs() < 1e-6 * b.abs().max(1.0),
                    "joint vs TT-only symplectic spectrum mismatch at n_eff=0: {a} vs {b}"
                );
            }
        }
        _ => panic!("both spectra must be Stable at a converged fixed-rank minimizer"),
    }
}

#[test]
fn tsu_resp_property_gradient_expand_cross_check() {
    let (space, h, x) = kappa_gamma_fixture(1, 0xC0A0);
    let mut m = CoeffTensor::<f64>::zeros(space.d, 2);
    for p in 0..space.d - 1 {
        m.set(&[p, p + 1], 0.35);
        m.set(&[p + 1, p], 0.35);
    }
    let mu_su2 =
        Su2SparseHamiltonian::new(space, &m, &CoeffTensor::<f64>::zeros(space.d, 4), None, 0.0);
    let g_su2 = orbital_gradient(&mu_su2, &x);

    let two_s = space.two_s as i64;
    let mut two_sz = -two_s;
    while two_sz <= two_s {
        if (two_s - two_sz).rem_euclid(2) == 0 {
            let ax: TtVector<f64> = expand(&x, two_sz);
            let ab_space = ax.space;
            let mu_ab = SparseHamiltonian::new(
                ab_space,
                &m,
                &CoeffTensor::<f64>::zeros(space.d, 4),
                None,
                0.0,
            );
            let g_ab = orbital_gradient_ab(&mu_ab, &ax);
            for &(p, q) in &orbital_pairs(space.pg, space.d) {
                let a = g_su2.get(&[p, q]);
                let b = g_ab.get(&[p, q]);
                assert!(
                    (a - b).abs() < 1e-10 * b.abs().max(1.0),
                    "property gradient[{p},{q}] mismatch at sz={two_sz}: su2={a} abelian={b}"
                );
            }
        }
        two_sz += 1;
    }

    let ctx = su2_joint_context(&h, &x, &su2_default_joint_opts())
        .unwrap_or_else(|e| panic!("context refused: {e}"));
    let val_su2 = su2_property_value(&ctx, &mu_su2);
    let ax0: TtVector<f64> = expand(&x, two_s);
    let ab_space0 = ax0.space;
    let mu_ab0 = SparseHamiltonian::new(
        ab_space0,
        &m,
        &CoeffTensor::<f64>::zeros(space.d, 4),
        None,
        0.0,
    );
    let ab_opts = JointOpts {
        tt_grad_tol: f64::INFINITY,
        ..su2_default_joint_opts()
    };
    let h_ab_full = SparseHamiltonian::new(
        ab_space0,
        &kappa_gamma_ints_t(space.d),
        &kappa_gamma_ints_v(space.d),
        None,
        f64::EPSILON,
    );
    let ab_ctx = qntt::solvers::joint::joint_context(&h_ab_full, &ax0, &ab_opts)
        .unwrap_or_else(|e| panic!("abelian context refused: {e}"));
    let val_ab = qntt::solvers::joint::property_value(&ab_ctx, &mu_ab0);
    assert!(
        (val_su2 - val_ab).abs() < 1e-10 * val_ab.abs().max(1.0),
        "property_value mismatch: su2={val_su2} abelian={val_ab}"
    );

    let raised = su2_property_gradient(&ctx, &mu_su2);
    assert!(
        raised.norm() > 1e-6,
        "property gradient must be non-vacuous"
    );
}

#[test]
fn tsu_resp_alpha_sos_matches_static_algebraic_identity() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 4, 0);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let x = su2_minimizer(&h, space, 0x50505, 1, 2);

    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d - 1 {
        m.set(&[p, p + 1], 0.3);
        m.set(&[p + 1, p], 0.3);
    }
    let prop = PropertyOperator::new("tsu_resp_sos", m);

    let ctx = su2_joint_context(
        &h,
        &x,
        &JointOpts {
            tt_grad_tol: f64::INFINITY,
            pairs: Some(PairMask::with_predicate(d, |_, _| false)),
            ..su2_default_joint_opts()
        },
    )
    .unwrap_or_else(|e| panic!("context refused: {e}"));

    let response = su2_polarizability_static(
        &ctx,
        std::slice::from_ref(&prop),
        &StaticResponseOpts {
            route: PropRoute::Materialized,
            tt_grad_tol: f64::INFINITY,
            orb_grad_tol: f64::INFINITY,
            ..StaticResponseOpts::default()
        },
    );
    let (basis, _report) = ctx.joint_basis();
    let moments: Result<TransitionMoments, ResponseError> = su2_transition_moments(
        &ctx,
        &basis,
        &[prop],
        &TransitionMomentOpts {
            route: PropRoute::Materialized,
            tt_grad_tol: f64::INFINITY,
            orb_grad_tol: f64::INFINITY,
            ..TransitionMomentOpts::default()
        },
    );

    match (response, moments) {
        (Ok(r), Ok(mm)) => {
            let budget = mm.completeness_defect[0].max(mm.orth_error).max(1e-6);
            let rel = (mm.alpha_sos[0] - r.alpha[0]).abs() / r.alpha[0].abs().max(1.0);
            assert!(
                rel < 20.0 * budget,
                "alpha_sos {} vs static alpha {} (rel={rel:e}, budget={budget:e})",
                mm.alpha_sos[0],
                r.alpha[0]
            );
            for w in mm.omega.windows(2) {
                assert!(
                    w[0] <= w[1] + 1e-10,
                    "omega must be ascending: {:?}",
                    mm.omega
                );
            }
            let mut covered = 0usize;
            for &(lo, hi) in &mm.degenerate_clusters {
                assert_eq!(
                    lo, covered,
                    "clusters must partition 0..omega.len() contiguously"
                );
                covered = hi;
            }
            assert_eq!(
                covered,
                mm.omega.len(),
                "clusters must cover the full omega range"
            );
        }
        (Err(e), _) => panic!("static response refused: {e:?}"),
        (_, Err(e)) => panic!("transition_moments refused: {e:?}"),
    }
}

#[test]
fn tsu_resp_omega_zero_degenerates() {
    let (space, h, x) = kappa_gamma_fixture(1, 0xD0A1);
    let d = space.d;
    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d - 1 {
        m.set(&[p, p + 1], 0.3);
        m.set(&[p + 1, p], 0.3);
    }
    let props = [PropertyOperator::new("tsu_resp_omega0", m)];

    let ctx = su2_joint_context(
        &h,
        &x,
        &JointOpts {
            pairs: Some(PairMask::with_predicate(d, |_, _| false)),
            ..su2_default_joint_opts()
        },
    )
    .unwrap_or_else(|e| panic!("omega0 context refused: {e}"));

    let sresp = su2_polarizability_static(
        &ctx,
        &props,
        &StaticResponseOpts {
            route: PropRoute::Materialized,
            tol: 1e-9,
            ..StaticResponseOpts::default()
        },
    )
    .unwrap_or_else(|e| panic!("static solve refused: {e:?}"));

    let dopts = DynamicResponseOpts {
        tol: 1e-9,
        route: PropRoute::Materialized,
        ..DynamicResponseOpts::default()
    };
    let dresp = su2_linear_response(&ctx, &props, 0.0, &dopts)
        .unwrap_or_else(|e| panic!("omega=0 solve refused: {e:?}"));

    let rel = (dresp.alpha[0] - sresp.alpha[0]).abs() / sresp.alpha[0].abs().max(1.0);
    assert!(
        rel <= 1e-10,
        "omega=0 must degenerate to the static path: dyn={} static={} rel={rel:e}",
        dresp.alpha[0],
        sresp.alpha[0]
    );
    assert!(
        dresp.solves[0].v_weight <= 1e-8,
        "the v-block must stay ~0 at omega=0: v_weight={:e}",
        dresp.solves[0].v_weight
    );
    assert!(
        dresp.solves[0].rel_residual <= dopts.tol * 10.0,
        "achieved residual must be reported fresh and small: {:e}",
        dresp.solves[0].rel_residual
    );
}

fn omega_route_fixture() -> (
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
    [PropertyOperator<f64>; 1],
) {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 3, 1);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0x5D1F);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    normalize_mut(&mut x);
    let mut m = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d - 1 {
        m.set(&[p, p + 1], 0.3);
        m.set(&[p + 1, p], 0.3);
    }
    (h, x, [PropertyOperator::new("tsu_resp_shifted", m)])
}

#[test]
#[should_panic(expected = "PropRoute::Shifted is unavailable")]
fn tsu_resp_omega_shifted_route_refused() {
    let (h, x, props) = omega_route_fixture();
    let ctx = su2_joint_context(
        &h,
        &x,
        &JointOpts {
            tt_grad_tol: f64::INFINITY,
            ..su2_default_joint_opts()
        },
    )
    .unwrap_or_else(|e| panic!("context construction refused: {e}"));
    let opts = DynamicResponseOpts {
        route: PropRoute::Shifted,
        ..DynamicResponseOpts::default()
    };
    let _ = su2_linear_response(&ctx, &props, 0.25, &opts);
}

#[test]
fn tsu_resp_omega_default_route_reaches_the_gate() {
    let (h, x, props) = omega_route_fixture();
    let ctx = su2_joint_context(
        &h,
        &x,
        &JointOpts {
            tt_grad_tol: f64::INFINITY,
            ..su2_default_joint_opts()
        },
    )
    .unwrap_or_else(|e| panic!("context construction refused: {e}"));
    let r = su2_linear_response(&ctx, &props, 0.25, &DynamicResponseOpts::default());
    assert!(
        matches!(r, Err(ResponseError::NotJointCritical { .. })),
        "default route must reach the gate and be refused there, got {r:?}"
    );
}

#[test]
fn tsu_opt_d4_tt_preserve_grid_plumbing() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let d = ints.norb;
    let space = Su2Space::new(d, 3, 1);
    let h_orig = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);

    let msu2 = 2usize;
    let ranks = su2_capped_ranks(&space, msu2);
    let mut rng = seeded(0xD4D4);
    let x0_raw: Su2TtVector<f64> = randomized_su2_state(space, 1, msu2, &mut rng);
    let (_lals, mut x0) = als(
        &h_orig,
        &x0_raw,
        &qntt::solvers::als::AlsOptions {
            eps: 1e-10,
            max_iter: 60,
            ..qntt::solvers::als::AlsOptions::default()
        },
    );
    x0.round_ranks_mut(&ranks);
    normalize_mut(&mut x0);
    let pinned = x0.ranks.clone();

    let scf_opts = JointScfOpts {
        orb_max_iter: 4,
        orb_grad: 1e-4,
        tt_grad_tol: 1e-4,
        orb_floor: 1e-8,
        max_ls_trials: 4,
        cg_skip_factor: 0.0,
        tt_preserve_grid: true,
        ..JointScfOpts::default()
    };
    let out: Su2JointScfOut = su2_joint_ground_state(&ints, space, &x0, &scf_opts);
    assert!(
        out.info.iters >= 1,
        "the SCF must take at least one orbital iteration, or the knob was never exercised past \
         the cold leg"
    );
    assert!(
        !out.info.tt_gradnorms.is_empty(),
        "no TT leg ran — this arm is vacuous"
    );
    assert_eq!(
        out.x.ranks, pinned,
        "tt_preserve_grid = true must carry the caller's declared SU(2) reduced-multiplet grid \
         through BOTH su2 riemannian_cg sites (the cold leg and the mid-line-search re-entry in \
         su2 tt_leg) unchanged"
    );
}

#[test]
fn tsu_tr_sp2_matches_dense() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let d = ints.norb;
    let space = Su2Space::new(d, 3, 1);
    let h_orig = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);

    let msu2 = 2usize;
    let ranks = su2_capped_ranks(&space, msu2);
    let mut rng = seeded(0x7261);
    let x0_raw: Su2TtVector<f64> = randomized_su2_state(space, 1, msu2, &mut rng);
    let (_lals, mut x0) = als(
        &h_orig,
        &x0_raw,
        &qntt::solvers::als::AlsOptions {
            eps: 1e-10,
            max_iter: 60,
            ..qntt::solvers::als::AlsOptions::default()
        },
    );
    x0.round_ranks_mut(&ranks);
    normalize_mut(&mut x0);

    let base = JointScfOpts {
        orb_max_iter: 20,
        orb_grad: 1e-4,
        tt_grad_tol: 1e-4,
        orb_floor: 1e-8,
        max_ls_trials: 4,
        ..JointScfOpts::default()
    };
    let dense: Su2JointScfOut = su2_joint_ground_state(&ints, space, &x0, &base);
    assert!(
        dense.info.converged,
        "the dense oracle run must certify: {:?}",
        dense.info.refused_reason
    );
    assert!(
        dense.info.g_orb_full.first().copied().unwrap_or(0.0) > 1e-3,
        "initial orbital gradient must be non-vacuous: {:?}",
        dense.info.g_orb_full.first()
    );

    let tr_opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        ..base.clone()
    };
    let tr: Su2JointScfOut = su2_joint_ground_state(&ints, space, &x0, &tr_opts);
    assert!(
        tr.info.converged,
        "the TR run must certify: {:?}",
        tr.info.refused_reason
    );
    assert_eq!(tr.info.solver, JointSolver::TrKrylov);

    assert!(
        (tr.lambda - dense.lambda).abs() <= 1e-6,
        "TR endpoint λ = {:.12e} vs dense λ = {:.12e} — same-critical-point parity at these \
         gates",
        tr.lambda,
        dense.lambda
    );
    assert_eq!(
        tr.info.n_eff, dense.info.n_eff,
        "n_eff parity with the dense run — a mismatch falsifies the same-point claim; \
         investigate, never relax"
    );

    let last_kept = *tr.info.g_orb_kept.last().unwrap();
    let last_tt = *tr.info.tt_gradnorms.last().unwrap();
    assert!(
        last_kept <= base.orb_grad * 1.01,
        "kept gate: {last_kept:e}"
    );
    assert!(
        last_tt <= base.tt_grad_tol * tr.lambda.abs().max(1.0) * 1.01,
        "TT gate: {last_tt:e}"
    );

    assert!(!tr.info.tr_radii.is_empty());
    assert!(!tr.info.cg_iters.is_empty());
    assert_eq!(tr.info.cg_statuses.len(), tr.info.cg_iters.len());
    assert!(tr.info.rho_history.len() >= tr.info.tr_radii.len());
    assert!(
        tr.info.delta_k_norms.iter().any(|&dk| dk > 0.0),
        "at least one accepted step must have moved the orbitals: {:?}",
        tr.info.delta_k_norms
    );
}

#[test]
#[should_panic(expected = "tr_init")]
fn tsu_tr_door_tr_init_zero() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let ints = Integrals::<f64> {
        t,
        v,
        e0: 0.0,
        norb: d,
        nelec: 2,
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    };
    let space = Su2Space::new(d, 2, 0);
    let mut rng = seeded(0xD00A);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        tr_init: 0.0,
        ..JointScfOpts::default()
    };
    let _ = su2_joint_ground_state(&ints, space, &x0, &opts);
}

fn tsu_probe_coeffs(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = seeded(seed);
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn tsu_combine(basis: &[Su2JointVector], c: &[f64]) -> Su2JointVector {
    let mut v = basis[0].clone();
    v.scale(0.0);
    for (b, &ci) in basis.iter().zip(c.iter()) {
        v.axpy(ci, b);
    }
    v
}

fn tsu_image_dev(a: &Su2JointVector, b: &Su2JointVector) -> f64 {
    let mut dxi = a.xi.clone();
    dxi.axpy(-1.0, &b.xi);
    let dc: f64 =
        a.c.iter()
            .zip(b.c.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum();
    let num = (dxi.dot(&dxi) + dc).sqrt();
    let den = (a.xi.dot(&a.xi) + a.c.iter().map(|x| x * x).sum::<f64>()).sqrt();
    num / den.max(1e-30)
}

#[allow(clippy::needless_range_loop)]
fn sym_eigen_jacobi(mut a: Vec<Vec<f64>>) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p][q] * a[p][q];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = t * c;
                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];
                a[p][p] = c * c * app - 2.0 * sn * c * apq + sn * sn * aqq;
                a[q][q] = sn * sn * app + 2.0 * sn * c * apq + c * c * aqq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for i in 0..n {
                    if i == p || i == q {
                        continue;
                    }
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip - sn * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = sn * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - sn * viq;
                    v[i][q] = sn * vip + c * viq;
                }
            }
        }
    }
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| a[i][i].partial_cmp(&a[j][j]).unwrap());
    let evals: Vec<f64> = idx.iter().map(|&i| a[i][i]).collect();
    let evecs: Vec<Vec<f64>> = idx
        .iter()
        .map(|&col| (0..n).map(|row| v[row][col]).collect())
        .collect();
    (evals, evecs)
}

#[test]
fn tsu_tr_wless_gate_removes_exactly_the_weingarten_term() {
    let fx = su2_lih_workhorse_fixture();
    let cb = su2_joint_context(&fx.h, &fx.x, &su2_default_joint_opts())
        .expect("the su2 workhorse point must gate");
    let (bb, _) = cb.joint_basis();
    let n = bb.len();
    assert!(
        n > 10,
        "anti-vacuity: the workhorse point must have a rich joint basis, got {n}"
    );
    let v = tsu_combine(&bb, &tsu_probe_coeffs(n, 0x5EED_5EED));
    let full = cb.hess_apply_deflated(&v);
    let wless = cb.hess_apply_deflated_wless(&v);
    let live = tsu_image_dev(&full, &wless);
    assert!(
        live > 1e-3,
        "anti-vacuity: the tt-Weingarten term must be live here — dev(Full, WLess) = {live:e}"
    );
    let wxi = cb.weingarten().b_apply(&v.xi);
    let mut delta_wless = full.clone();
    delta_wless.axpy(-1.0, &wless);
    let scale = full.norm().max(1e-30);
    let mut pure = 0usize;
    let mut biggest = 0.0f64;
    for (i, b) in bb.iter().enumerate() {
        if !b.c.iter().all(|&x| x == 0.0) {
            continue;
        }
        pure += 1;
        let target_w = b.xi.dot(&wxi);
        biggest = biggest.max(target_w.abs());
        let dev_w = (b.dot(&delta_wless) - target_w).abs();
        assert!(
            dev_w <= 1e-10 * scale,
            "the W-LESS gate must remove EXACTLY the tt-Weingarten term at basis vector {i}: \
             ⟨b, Full − WLess⟩ = {:e} vs ⟨ξ_b, W ξ_v⟩ = {target_w:e} (scale {scale:e})",
            b.dot(&delta_wless)
        );
    }
    assert!(
        pure > 0 && biggest > 1e-8 * scale,
        "anti-vacuity: the identity must be checked on pure-tangent basis vectors against a \
         resolvable W pairing — {pure} such vectors, largest |⟨ξ_b, W ξ_v⟩| = {biggest:e}"
    );
}

#[test]
fn tsu_tr_fw_separates_the_artifact_from_the_physical_modes() {
    let fx = su2_lih_workhorse_fixture();
    let jctx = su2_joint_context(&fx.h, &fx.x, &su2_default_joint_opts())
        .expect("the su2 workhorse point must gate");
    let (basis, _report) = jctx.joint_basis();
    let n = basis.len();
    let (ga, gb) = jctx.joint_hessian_grams(&basis);
    let model: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| 0.5 * ((ga[i * n + j] + gb[i * n + j]) + (ga[j * n + i] + gb[j * n + i])))
                .collect()
        })
        .collect();
    let (evals, evecs) = sym_eigen_jacobi(model);
    let wctx = jctx.weingarten();
    let wxi: Vec<Su2TtTangent<f64>> = basis.iter().map(|b| wctx.b_apply(&b.xi)).collect();
    let mut wg = vec![0.0f64; n * n];
    for (i, bi) in basis.iter().enumerate() {
        for (j, wj) in wxi.iter().enumerate() {
            wg[i * n + j] = bi.xi.dot(wj);
        }
    }
    let quad = |m: &[f64], c: &[f64]| -> f64 {
        (0..n)
            .map(|i| c[i] * (0..n).map(|j| m[i * n + j] * c[j]).sum::<f64>())
            .sum::<f64>()
    };
    let fracs: Vec<f64> = evecs
        .iter()
        .map(|c| quad(&wg, c).abs() / quad(&ga, c).abs())
        .collect();
    let diag = (0..n)
        .map(|i| wg[i * n + i].abs() / ga[i * n + i].abs())
        .fold(0.0f64, f64::max);
    let quietest = fracs[1..].iter().cloned().fold(f64::INFINITY, f64::min);
    let tail_max = fracs[ARTIFACT_SCREEN_MODES..]
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);
    println!(
        "tsu_tr_fw: n = {n}, σ_min = {:.3e}, λ_min = {:+.4e}, f_W(screened) = {:?}, quietest \
         f_W = {quietest:.3e}, unscreened tail max = {tail_max:.3e}, diagonal proxy = {diag:.3e}",
        jctx.sigma_min(),
        evals[0],
        &fracs[..ARTIFACT_SCREEN_MODES]
    );
    assert!(
        evals[0] < -1.0,
        "the premise of the artifact is a large spurious negative mode: λ_min = {:e}",
        evals[0]
    );
    assert!(
        fracs[0] > 2.0,
        "the artifact mode must clear the shipped τ = 1 with margin: f_W = {:e} (measured \
         1.857e1)",
        fracs[0]
    );
    assert!(
        quietest < 1e-10,
        "the detector must reach decades below τ on the same point: quietest f_W = {quietest:e} \
         (measured 2.316e-15 already on mode 3)"
    );
    assert!(
        fracs[0] / quietest > 1e10,
        "artifact-vs-physical dynamic range must be many orders: {:e} / {quietest:e}",
        fracs[0]
    );
    assert!(
        tail_max > 1.0,
        "the screen must stay restricted to the lowest {ARTIFACT_SCREEN_MODES} modes: this point \
         measures {tail_max:e} (1.013e1) among modes {ARTIFACT_SCREEN_MODES}.., so a \
         spectrum-wide τ = 1 filter would delete real curvature"
    );
    assert!(
        diag < fracs[0],
        "the diagonal proxy must under-report the collective artifact: {diag:e} against f_W = \
         {:e}",
        fracs[0]
    );
}

const ARTIFACT_SCREEN_MODES: usize = 4;

#[test]
fn tsu_tr_knobs_off_short_run_matches_the_recorded_trace() {
    let (ints, space, x0) = tsu_tr_start();
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        orb_max_iter: 6,
        orb_grad: 1e-4,
        tt_grad_tol: 1e-4,
        orb_floor: 1e-8,
        max_ls_trials: 4,
        ..JointScfOpts::default()
    };
    let out: Su2JointScfOut = su2_joint_ground_state(&ints, space, &x0, &opts);
    println!(
        "tsu_tr_knobs_off: λ = {:.17e}, radii = {:?}, ρ = {:?}, cg = {:?}",
        out.lambda, out.info.tr_radii, out.info.rho_history, out.info.cg_iters
    );

    const LAMBDA: f64 = TSU_OFF_LAMBDA;
    assert!(
        (out.lambda - LAMBDA).abs() <= 1e-12 * LAMBDA.abs(),
        "the knobs-off trajectory moved: λ = {:.17e} against the recorded {LAMBDA:.17e}",
        out.lambda
    );
    assert_eq!(
        out.info.tr_radii.len(),
        TSU_OFF_RADII.len(),
        "iteration count moved: {:?}",
        out.info.tr_radii
    );
    for (i, (&got, &want)) in out
        .info
        .tr_radii
        .iter()
        .zip(TSU_OFF_RADII.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs(),
            "the radius trace moved at iteration {i}: {got:e} against the recorded {want:e}"
        );
    }
    assert_eq!(
        out.info.rho_history.len(),
        TSU_OFF_RHO.len(),
        "trial count moved: {:?}",
        out.info.rho_history
    );
    for (i, (&got, &want)) in out
        .info
        .rho_history
        .iter()
        .zip(TSU_OFF_RHO.iter())
        .enumerate()
    {
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "the ρ history moved at trial {i}: {got:e} against the recorded {want:e}"
        );
    }
    assert_eq!(
        out.info.cg_iters.as_slice(),
        TSU_OFF_CG,
        "the Steihaug iteration counts moved"
    );
}

const TSU_OFF_LAMBDA: f64 = -8.609_191_128_169_206;
const TSU_OFF_RADII: &[f64] = &[0.1, 0.2, 0.2, 0.4, 0.2];
const TSU_OFF_RHO: &[f64] = &[
    1.0033928100348373,
    1.0055801508700524,
    0.561041024344998,
    0.9012736551280807,
    -0.5017704545881888,
    0.9293648305768918,
];
const TSU_OFF_CG: &[usize] = &[1, 1, 1, 1, 1, 1];

fn tsu_tr_start() -> (Integrals<f64>, Su2Space, Su2TtVector<f64>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g");
    let ints = load.ints;
    let d = ints.norb;
    let space = Su2Space::new(d, 3, 1);
    let h_orig = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    let msu2 = 2usize;
    let ranks = su2_capped_ranks(&space, msu2);
    let mut rng = seeded(0x7261);
    let x0_raw: Su2TtVector<f64> = randomized_su2_state(space, 1, msu2, &mut rng);
    let (_lals, mut x0) = als(
        &h_orig,
        &x0_raw,
        &qntt::solvers::als::AlsOptions {
            eps: 1e-10,
            max_iter: 60,
            ..qntt::solvers::als::AlsOptions::default()
        },
    );
    x0.round_ranks_mut(&ranks);
    normalize_mut(&mut x0);
    (ints, space, x0)
}

#[test]
#[should_panic(expected = "cached the damped gauge set")]
fn tsu_tr_damped_gauge_cache_is_one_s_per_context() {
    let fx = su2_lih_workhorse_fixture();
    let jctx = su2_joint_context(&fx.h, &fx.x, &su2_default_joint_opts())
        .expect("the su2 workhorse point must gate");
    let (basis, _) = jctx.joint_basis();
    let v = tsu_combine(&basis, &tsu_probe_coeffs(basis.len(), 11));
    let wctx = jctx.weingarten();
    let _ = wctx.b_apply_damped(&v.xi, 1e-6);
    let _ = wctx.b_apply_damped(&v.xi, 1e-5);
}

#[test]
fn tsu_rank_limited_diagnostic() {
    let (ints, space, x0) = tsu_tr_start();
    let opts = JointScfOpts {
        orb_max_iter: 30,
        orb_grad: 1e-6,
        orb_grad_full: Some(1e-9),
        orb_floor: 1e-2,
        tt_grad_tol: 1e-4,
        ..JointScfOpts::default()
    };
    let out = su2_joint_ground_state(&ints, space, &x0, &opts);
    println!(
        "tsu_rank_limited: converged={} rank_limited={} gk={:?} gL={:?} dropped={:.3e} \
         angle={:.3e} leak={:.3e} iters={}",
        out.info.converged,
        out.info.rank_limited,
        out.info.g_orb_kept.last(),
        out.info.g_orb_full.last(),
        out.info.g_dropped_final,
        out.info.dropped_angle,
        out.info.dropped_leak_bound,
        out.info.iters
    );

    assert!(
        out.info.converged,
        "the kept gradient and TT gate are both met, so this must certify: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.info.refused_reason.is_none(),
        "a rank-limited point is not a refusal: {:?}",
        out.info.refused_reason
    );
    assert!(
        out.info.rank_limited,
        "the dropped residual is above orb_grad_full, so this point MUST be flagged rank-limited"
    );

    let gk_last = out.info.g_orb_kept.last().copied().unwrap_or(f64::INFINITY);
    assert!(
        gk_last <= opts.orb_grad,
        "the KEPT gradient must have met its own gate — the flag is about the DROPPED part: \
         {gk_last:e}"
    );
    assert!(
        out.info.g_dropped_final > opts.orb_grad_full.unwrap(),
        "rank_limited must be driven by the DROPPED component: {:e}",
        out.info.g_dropped_final
    );
    assert!(
        out.info.dropped_angle.is_finite() && out.info.dropped_angle > 0.0,
        "a dropped mode exists, so its angle must be reported: {}",
        out.info.dropped_angle
    );
    assert!(
        out.info.dropped_leak_bound > 0.0,
        "the Cauchy-Schwarz leak bound must be stamped alongside"
    );
    assert!(
        out.info.iters < opts.orb_max_iter,
        "the rank-limited exit must fire before the iteration cap: {} vs {}",
        out.info.iters,
        opts.orb_max_iter
    );
}

#[test]
#[should_panic(expected = "tr_legs_every is a period in accepted steps")]
fn tsu_tr_door_legs_every_zero() {
    let (ints, space, x0) = tsu_tr_start();
    let opts = JointScfOpts {
        solver: JointSolver::TrKrylov,
        tr_legs_every: 0,
        ..JointScfOpts::default()
    };
    let _ = su2_joint_ground_state(&ints, space, &x0, &opts);
}

#[test]
#[should_panic(expected = "moving-quotient protocol")]
fn tsu_tr_door_legs_every_needs_trkrylov() {
    let (ints, space, x0) = tsu_tr_start();
    let opts = JointScfOpts {
        solver: JointSolver::DenseSchur,
        tr_legs_every: 2,
        ..JointScfOpts::default()
    };
    let _ = su2_joint_ground_state(&ints, space, &x0, &opts);
}

#[test]
fn tsu_tr_moving_quotient_certifies_with_fewer_legs() {
    let (ints, space, x0) = tsu_tr_start();
    let opts1 = JointScfOpts {
        solver: JointSolver::TrKrylov,
        orb_max_iter: 30,
        orb_grad: 1e-4,
        tt_grad_tol: 1e-4,
        orb_floor: 1e-8,
        max_ls_trials: 4,
        ..JointScfOpts::default()
    };
    let out1 = su2_joint_ground_state(&ints, space, &x0, &opts1);
    let out2 = su2_joint_ground_state(
        &ints,
        space,
        &x0,
        &JointScfOpts {
            tr_legs_every: 2,
            ..opts1.clone()
        },
    );
    println!(
        "tsu_tr_mq: k=1 λ = {:.15e} legs = {} iters = {} | k=2 λ = {:.15e} legs = {} iters = {} \
         | Δλ = {:.3e}",
        out1.lambda,
        out1.info.tt_legs,
        out1.info.iters,
        out2.lambda,
        out2.info.tt_legs,
        out2.info.iters,
        out2.lambda - out1.lambda
    );
    assert!(
        out1.info.converged && out2.info.converged,
        "both protocols must certify: k=1 {:?}, k=2 {:?}",
        out1.info.refused_reason,
        out2.info.refused_reason
    );
    assert!(
        out2.info.tt_legs < out1.info.tt_legs,
        "the deferred protocol must run fewer driver legs: {} vs {}",
        out2.info.tt_legs,
        out1.info.tt_legs
    );
    assert!(
        (out2.lambda - out1.lambda).abs() <= 1e-8 * out1.lambda.abs(),
        "the two protocols must land on the same certified endpoint: Δλ = {:.3e}",
        out2.lambda - out1.lambda
    );
}

#[test]
fn tsu_f1_sweep_and_dots_covectors_agree() {
    for (two_s, seed, b_general) in [(1usize, 0xF1_00u64, false), (1, 0xF1_02, true)] {
        let (space, h, x) = kappa_gamma_fixture(two_s, seed);
        let opts = JointOpts {
            b_general,
            ..su2_default_joint_opts()
        };
        let ctx = su2_joint_context(&h, &x, &opts)
            .unwrap_or_else(|e| panic!("su2_joint_context refused: {e}"));

        let n_orb = ctx.pairs().len();
        assert!(n_orb > 0, "vacuous pair list");
        let mut rng = seeded(seed ^ 0xABCD);
        let (bl, br) = ctx.bases();
        let xi = {
            let mut v: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
            v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
            Su2TtTangent::project(bl, br, &v)
        };
        let zeros = vec![0.0f64; n_orb];
        let a_dir: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();

        for (label, a) in [("kappa=0", &zeros), ("kappa!=0", &a_dir)] {
            for (which, sw, dt) in [
                (
                    "A",
                    ctx.a_covector(&xi, a, FcRoute::Sweep),
                    ctx.a_covector(&xi, a, FcRoute::Dots),
                ),
                (
                    "B",
                    ctx.b_covector(&xi, a, FcRoute::Sweep),
                    ctx.b_covector(&xi, a, FcRoute::Dots),
                ),
            ] {
                let scale = dt.1.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
                let worst =
                    sw.1.iter()
                        .zip(dt.1.iter())
                        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                assert!(
                    worst <= 1e-9 * scale,
                    "{which} covector F_c: Sweep vs Dots worst |Δ| = {worst:e} > {:e} \
                     ({label}, b_general={b_general})",
                    1e-9 * scale
                );
                assert!(
                    scale > 1e-6,
                    "{which} covector F_c is numerically zero ({label}) — the arm is vacuous"
                );
                let dxi = sw.0.add(&dt.0.scale(-1.0));
                assert!(
                    dxi.norm() <= 1e-12 * dt.0.norm().max(1.0),
                    "{which} covector F_ξ must be route-independent ({label}): ‖Δ‖ = {:e}",
                    dxi.norm()
                );
            }
        }
    }
}

#[test]
fn tsu_z1_split_and_explicit_covectors_agree() {
    for (two_s, seed, b_general) in [(1usize, 0xF1_10u64, false), (1, 0xF1_12, true)] {
        let (space, h, x) = kappa_gamma_fixture(two_s, seed);

        let (n_theta, n_cw) = count_theta_cw(&x);
        assert!(
            n_theta > 0,
            "θ ≡ 1 at this base point (n_theta={n_theta}, n_cw={n_cw}) — the H12 metric arms are \
             vacuous here"
        );

        let mk = |z_route: ZRoute| JointOpts {
            b_general,
            z_route,
            ..su2_default_joint_opts()
        };
        let split = su2_joint_context(&h, &x, &mk(ZRoute::Split))
            .unwrap_or_else(|e| panic!("su2_joint_context (Split) refused: {e}"));
        let explicit = su2_joint_context(&h, &x, &mk(ZRoute::Explicit))
            .unwrap_or_else(|e| panic!("su2_joint_context (Explicit) refused: {e}"));
        assert_eq!(
            split.n_eff(),
            explicit.n_eff(),
            "z_route must not change n_eff"
        );
        assert_eq!(
            split.pairs().len(),
            explicit.pairs().len(),
            "z_route must not change the pair list"
        );

        let n_orb = split.pairs().len();
        assert!(n_orb > 0, "vacuous pair list");
        let mut rng = seeded(seed ^ 0xABCD);
        let (bl, br) = split.bases();
        let tb = tangent_basis(bl, br, true);
        assert!(tb.len() >= 3, "need a nontrivial tangent basis");
        let xi = {
            let mut v: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
            v.round_mut(1e-12, qntt::Pooled::NONE, RoundMode::Rel);
            Su2TtTangent::project(bl, br, &v)
        };
        let zeros = vec![0.0f64; n_orb];
        let a_dir: Vec<f64> = (0..n_orb).map(|_| rng.random_range(-1.0..1.0)).collect();

        for (label, a) in [("kappa=0", &zeros), ("kappa!=0", &a_dir)] {
            for (which, sp, ex) in [
                (
                    "A",
                    split.a_covector(&xi, a, FcRoute::Sweep),
                    explicit.a_covector(&xi, a, FcRoute::Sweep),
                ),
                (
                    "B",
                    split.b_covector(&xi, a, FcRoute::Sweep),
                    explicit.b_covector(&xi, a, FcRoute::Sweep),
                ),
            ] {
                let scale_fc = ex.1.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let worst_fc =
                    sp.1.iter()
                        .zip(ex.1.iter())
                        .fold(0.0f64, |m, (s, e)| m.max((s - e).abs()));
                assert!(
                    scale_fc > 1e-6,
                    "{which} covector F_c is numerically zero ({label}, b_general={b_general}) — \
                     the arm is vacuous"
                );
                assert!(
                    worst_fc <= 1e-11 * scale_fc,
                    "{which} covector F_c: Split vs Explicit worst |Δ| = {worst_fc:e} > {:e} \
                     ({label}, b_general={b_general})",
                    1e-11 * scale_fc
                );

                let mut scale_fxi = 0.0f64;
                let mut worst_fxi = 0.0f64;
                for slot in tb.iter().take(8) {
                    let got = slot.dot(&sp.0);
                    let want = slot.dot(&ex.0);
                    scale_fxi = scale_fxi.max(got.abs()).max(want.abs());
                    worst_fxi = worst_fxi.max((got - want).abs());
                }
                assert!(
                    scale_fxi > 1e-6,
                    "{which} covector F_ξ is numerically zero ({label}, b_general={b_general}) — \
                     the arm is vacuous"
                );
                assert!(
                    worst_fxi <= 1e-11 * scale_fxi,
                    "{which} covector F_ξ: Split vs Explicit worst |Δ| = {worst_fxi:e} > {:e} \
                     ({label}, b_general={b_general})",
                    1e-11 * scale_fxi
                );
            }
        }

        for (which, k0, k1) in [
            (
                "A",
                split.a_covector(&xi, &zeros, FcRoute::Sweep),
                split.a_covector(&xi, &a_dir, FcRoute::Sweep),
            ),
            (
                "B",
                split.b_covector(&xi, &zeros, FcRoute::Sweep),
                split.b_covector(&xi, &a_dir, FcRoute::Sweep),
            ),
        ] {
            let leg = k1.0.add(&k0.0.scale(-1.0)).norm();
            assert!(
                leg > 1e-6 * k0.0.norm().max(1.0),
                "{which}: the κ-dependent ξ-leg is numerically absent (‖Δ‖ = {leg:e}, \
                 b_general={b_general}) — the route-equivalence assertions above have no teeth"
            );
        }
    }
}

#[test]
fn tsu_gate_orb_floor_negative_refused() {
    let (_space, h, x) = kappa_gamma_fixture(1, 0xF10A);
    let opts = JointOpts {
        orb_floor: -1e-3,
        ..su2_default_joint_opts()
    };
    match su2_joint_context(&h, &x, &opts) {
        Err(Su2JointGateError::OrbFloorNegative { got }) => {
            assert_eq!(got, -1e-3, "the refusal must carry the offending value");
        }
        Ok(_) => panic!("a negative orb_floor must refuse"),
        Err(e) => panic!("expected OrbFloorNegative, got {e}"),
    }
}

#[test]
fn tsu_gate_not_tt_critical_refused() {
    let d = 4usize;
    let (t, v) = generic_truncated_tv(d);
    let space = Su2Space::new(d, 3, 1);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0x0F1B);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    normalize_mut(&mut x);
    match su2_joint_context(&h, &x, &su2_default_joint_opts()) {
        Err(Su2JointGateError::NotTtCritical {
            measured,
            bound,
            lambda,
        }) => {
            assert!(
                measured > bound,
                "refusal fields incoherent: measured {measured} <= bound {bound}"
            );
            assert!(
                lambda.is_finite(),
                "λ must be reported finite, got {lambda}"
            );
        }
        Ok(_) => panic!("a random state must not gate as TT-critical"),
        Err(e) => panic!("expected NotTtCritical, got {e}"),
    }
}
