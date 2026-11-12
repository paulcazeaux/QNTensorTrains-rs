// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::tangent::TtTangent;
use qntt::core::weingarten::FloorPolicy;
use qntt::solvers::InnerProductVector;
use qntt::solvers::riemannian::{
    RiemannianOpts, normalize_mut, riemannian_cg, riemannian_gd, tangent_bases,
};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::{Su2RankGrid, Su2Space};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::{QnSpace, RankGrid, Scalar, SparseHamiltonian, TtVector, rayleigh_quotient};
pub(crate) use qntt_testkit::randomized_state;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) use qntt_testkit::gad_mol::grad_exact_at;
pub(crate) use qntt_testkit::gad_mol::{
    coeff_vector, dense_h_in_sector, dense_sector_eigen, det_states, overlap_modulus,
    singlet_indices, su2_deflate_radial_span, su2_grad_exact_at, su2_kick_along,
    su2_radial_mode_at,
};
pub(crate) use qntt_testkit::generic_complex_h;
pub(crate) use qntt_testkit::generic_truncated_h;
pub(crate) use qntt_testkit::generic_truncated_tv;

pub(crate) fn full_rank_state<T: Scalar>(
    space: QnSpace,
    rng: &mut Xoshiro256PlusPlus,
) -> TtVector<T> {
    qntt::tt_randn(space, &qntt::max_ranks(&space), rng)
}

pub(crate) fn su2_randomized_state<T: Scalar>(
    space: Su2Space,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Su2TtVector<T> {
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
    qntt::su2::vector::tt_randn(space, &ranks, rng)
}

pub(crate) fn su2_full_rank_state<T: Scalar>(
    space: Su2Space,
    rng: &mut Xoshiro256PlusPlus,
) -> Su2TtVector<T> {
    qntt::su2::vector::tt_randn(space, &qntt::su2::qn::max_ranks(&space), rng)
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

pub(crate) fn inf_scale<V: InnerProductVector>(d: &V) -> f64 {
    let mut m = 0.0f64;
    let mut c = d.clone();
    c.for_each_coeff_mut(|x| m = m.max(x.abs_s()));
    m.max(1.0)
}

pub(crate) fn su2_min_schmidt_raw<T: Scalar>(bl: &Su2TtVector<T>, br: &Su2TtVector<T>) -> f64 {
    qntt::su2::weingarten::bond_gauges(bl, br, FloorPolicy::default_hard())
        .iter()
        .flat_map(|g| g.sigma.iter().flat_map(|(_, v)| v.iter().copied()))
        .fold(f64::INFINITY, f64::min)
}

pub(crate) fn converged_minimizer<T: Scalar>(
    ham: &SparseHamiltonian<T>,
    x0: &TtVector<T>,
) -> TtVector<T> {
    let space = x0.space;
    let mut opts = RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    opts.preserve_grid = true;
    let (_l, xg) = riemannian_gd(ham, x0, &opts);
    let (_l, mut x) = riemannian_cg(ham, &xg, &opts);
    for _ in 0..8 {
        if grad_exact_at(ham, &x).norm() < 1e-11 {
            break;
        }
        let (_l2, x2) = riemannian_cg(ham, &x, &opts);
        x = x2;
    }
    normalize_mut(&mut x);
    x.check();
    let gnorm = grad_exact_at(ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "GAD launch fixture is not a Riemannian critical point (‖grad‖={gnorm:.3e}) — the \
         index-0 refusal arm and the kick recipe both assume a converged ground state"
    );
    x
}

pub(crate) fn su2_converged_minimizer<T: Scalar>(
    ham: &Su2SparseHamiltonian<T>,
    x0: &Su2TtVector<T>,
) -> Su2TtVector<T> {
    use qntt::su2::riemannian::{
        RiemannianOpts as Su2RiemannianOpts, normalize_mut as su2_normalize_mut,
        riemannian_cg as su2_cg, riemannian_gd as su2_gd,
    };
    let space = x0.space;
    let mut opts = Su2RiemannianOpts::new(&space);
    opts.eps = 1e-13;
    opts.max_iter = 600;
    opts.gradtol = 1e-13;
    opts.preserve_grid = true;
    let (_l, xg) = su2_gd(ham, x0, &opts);
    let (_l, mut x) = su2_cg(ham, &xg, &opts);
    for _ in 0..8 {
        if su2_grad_exact_at(ham, &x).norm() < 1e-11 {
            break;
        }
        let (_l2, x2) = su2_cg(ham, &x, &opts);
        x = x2;
    }
    su2_normalize_mut(&mut x);
    x.check();
    let gnorm = su2_grad_exact_at(ham, &x).norm();
    assert!(
        gnorm < 1e-9,
        "SU(2) GAD launch fixture is not critical (‖grad‖={gnorm:.3e})"
    );
    x
}

pub(crate) fn kick_along<T: Scalar>(v0: &TtTangent<T>, eps: f64) -> TtVector<T> {
    let mut x = v0.retract_hosvd(T::from_f64(eps));
    normalize_mut(&mut x);
    x.check();
    x
}

pub(crate) struct FullRankArm {
    pub(crate) ham: SparseHamiltonian<f64>,
    pub(crate) dets: Vec<TtVector<f64>>,
    pub(crate) evals: Vec<f64>,
    pub(crate) evecs: Vec<Vec<f64>>,
    pub(crate) xgs: TtVector<f64>,
    pub(crate) x0: TtVector<f64>,
}

pub(crate) fn fixture_probe() -> qntt::solvers::hess::ProbeDavidson {
    qntt::solvers::hess::ProbeDavidson::default()
}

pub(crate) const FIXTURE_PROBE_SEED: u64 = 0xC3;

const SIGN_REF_MIN_COS: f64 = 1e-3;

pub(crate) fn canonicalize_witness_sign(w: &mut TtTangent<f64>, seed: &TtTangent<f64>) {
    let c = w.dot(seed) / (w.norm() * seed.norm());
    assert!(
        c.abs() > SIGN_REF_MIN_COS,
        "the witness sign reference is degenerate (|cos| = {:.3e} ≤ {SIGN_REF_MIN_COS:.0e}): the \
         launch point is no longer a property of the fixture",
        c.abs()
    );
    if c < 0.0 {
        w.scale_mut(-1.0);
    }
}

pub(crate) fn su2_canonicalize_witness_sign(w: &mut Su2TtTangent<f64>, seed: &Su2TtTangent<f64>) {
    let c = w.dot(seed) / (w.norm() * seed.norm());
    assert!(
        c.abs() > SIGN_REF_MIN_COS,
        "the SU(2) witness sign reference is degenerate (|cos| = {:.3e} ≤ \
         {SIGN_REF_MIN_COS:.0e}): the launch point is no longer a property of the fixture",
        c.abs()
    );
    if c < 0.0 {
        w.scale_mut(-1.0);
    }
}

pub(crate) fn f64_full_rank_arm() -> FullRankArm {
    use qntt::core::tangent::tangent_dim;
    use qntt::solvers::hess::{
        HessPart, PdVerdict, hessian_pd_probe_witness_with, weingarten_operator,
    };

    let space = QnSpace::new(4, 2, 2);
    let ham = generic_truncated_h::<f64>(space);
    let dets = det_states::<f64>(4, 2, 2);
    let ndet = dets.len();
    assert_eq!(ndet, 36, "sector enumeration changed");
    let (evals, evecs) = dense_sector_eigen(&dense_h_in_sector(&dets, &ham), ndet);
    assert!(
        evals[1] - evals[0] > 1e-3 && evals[2] - evals[1] > 1e-3,
        "fixture gate: E₁ must be non-degenerate and separated ({:?})",
        &evals[..3]
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6AD0);
    let seed_state: TtVector<f64> = full_rank_state(space, &mut rng);
    let xgs = converged_minimizer(&ham, &seed_state);
    let lam0 = rayleigh_quotient(&ham, &xgs);
    assert!(
        (lam0 - evals[0]).abs() < 1e-8,
        "full-rank ground state {lam0} vs dense E₀ {}",
        evals[0]
    );

    let v0 = {
        let ctx = weingarten_operator(&ham, &xgs);
        let (bl, br) = ctx.bases();
        assert_eq!(
            tangent_dim(bl),
            ndet,
            "not at full representing rank: tangent_dim {} vs sector dimension {ndet}",
            tangent_dim(bl)
        );
        let diag = ctx.a_diagonal();
        let seed = TtTangent::project(bl, br, &diag.to_vector());
        let (verdict, w, _a) = hessian_pd_probe_witness_with(
            &ctx,
            HessPart::Rr,
            std::slice::from_ref(&seed),
            200,
            FIXTURE_PROBE_SEED,
            &fixture_probe(),
        );
        assert!(
            matches!(verdict, PdVerdict::LikelyPd { .. }),
            "the ground state must probe PD, got {verdict:?}"
        );
        let radial = radial_mode_at(bl, br, bl);
        let mut v = w;
        deflate_radial_span(&radial, &mut v);
        let n = v.norm();
        v.scale_mut(1.0 / n);
        canonicalize_witness_sign(&mut v, &seed);
        v
    };

    let x0 = kick_along(&v0, 5e-2);
    let g0 = grad_exact_at(&ham, &x0).norm();
    assert!(
        g0 > 1e-4,
        "kick left the state critical (‖grad‖={g0:.3e}) — vacuous arm"
    );

    FullRankArm {
        ham,
        dets,
        evals,
        evecs,
        xgs,
        x0,
    }
}

pub(crate) fn f64_truncated_arm() -> (SparseHamiltonian<f64>, TtVector<f64>, TtVector<f64>) {
    use qntt::solvers::hess::{HessPart, hessian_pd_probe_witness_with, weingarten_operator};

    let space = QnSpace::new(4, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB0B1);
    let seed_state: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let ham = generic_truncated_h::<f64>(space);
    let x = converged_minimizer(&ham, &seed_state);
    let v0 = {
        let ctx = weingarten_operator(&ham, &x);
        let (bl, br) = ctx.bases();
        let diag = ctx.a_diagonal();
        let seed = TtTangent::project(bl, br, &diag.to_vector());
        let (_verdict, w, _a) = hessian_pd_probe_witness_with(
            &ctx,
            HessPart::Rr,
            std::slice::from_ref(&seed),
            150,
            FIXTURE_PROBE_SEED,
            &fixture_probe(),
        );
        let radial = radial_mode_at(bl, br, bl);
        let mut v = w;
        deflate_radial_span(&radial, &mut v);
        let n = v.norm();
        v.scale_mut(1.0 / n);
        canonicalize_witness_sign(&mut v, &seed);
        v
    };
    let x0 = kick_along(&v0, 5e-2);
    (ham, x, x0)
}

pub(crate) fn f64_deficient_state() -> (SparseHamiltonian<f64>, TtVector<f64>, usize, qntt::Sector)
{
    use qntt::Family;

    let space = QnSpace::new(4, 2, 1);
    let ham = generic_truncated_h::<f64>(space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEAD_BEEF);
    let mut x: TtVector<f64> = randomized_state(space, 2, 2, &mut rng);

    let bond = 2usize;
    let sector = space
        .bond_sectors(bond)
        .into_iter()
        .find(|&s| x.ranks[bond][s] >= 2)
        .expect("fixture gate: no rank-≥2 sector at bond 2 — the (2,1) space changed shape");

    let k = bond - 1;
    let g = space.pg.irrep_of(k);
    let row_qn = x.cores[k].row_qn.clone();
    {
        let core = &mut x.cores[k];
        for f in Family::ALL {
            for s in row_qn.iter().copied() {
                if f.apply(s, g) != sector {
                    continue;
                }
                if let Some(mut m) = core.block_mut(f, s) {
                    let nc = m.ncols();
                    if nc > 1 {
                        for i in 0..m.nrows() {
                            use faer::prelude::ReborrowMut;
                            let v = *m.rb_mut().get_mut(i, 1);
                            *m.rb_mut().get_mut(i, 1) = v * 1e-16;
                        }
                    }
                }
            }
        }
    }
    normalize_mut(&mut x);
    x.check();
    (ham, x, bond, sector)
}

pub(crate) fn f64_deficient_declarable_state(
    scale: f64,
) -> (SparseHamiltonian<f64>, TtVector<f64>, usize, qntt::Sector) {
    use qntt::Family;

    let space = QnSpace::new(4, 2, 1);
    let ham = generic_truncated_h::<f64>(space);
    let caps = qntt::max_ranks(&space);
    let d = space.d;
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    caps[b][s].min(2)
                };
            }
            r
        })
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0DEF_1C17);
    let mut x: TtVector<f64> = qntt::tt_randn(space, &ranks, &mut rng);

    let bond = 2usize;
    let sector = space
        .bond_sectors(bond)
        .into_iter()
        .find(|&s| caps[bond][s] >= 2)
        .expect("fixture gate: no cap-2 sector at bond 2 — the (2,1) space changed shape");

    let k = bond - 1;
    let g = space.pg.irrep_of(k);
    let row_qn = x.cores[k].row_qn.clone();
    {
        let core = &mut x.cores[k];
        for f in Family::ALL {
            for s in row_qn.iter().copied() {
                if f.apply(s, g) != sector {
                    continue;
                }
                if let Some(mut m) = core.block_mut(f, s) {
                    let nc = m.ncols();
                    if nc > 1 {
                        for i in 0..m.nrows() {
                            use faer::prelude::ReborrowMut;
                            let v = *m.rb_mut().get_mut(i, 1);
                            *m.rb_mut().get_mut(i, 1) = v * scale;
                        }
                    }
                }
            }
        }
    }
    normalize_mut(&mut x);
    x.check();
    (ham, x, bond, sector)
}

pub(crate) fn su2_truncated_arm() -> (
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
) {
    use qntt::solvers::hess::HessPart;

    let space = Su2Space::new(4, 4, 0);
    let (t, v) = generic_truncated_tv::<f64>(4);
    let ham = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0A1);
    let seed_state = su2_randomized_state::<f64>(space, 1, 2, &mut rng);
    let x = su2_converged_minimizer(&ham, &seed_state);
    let v0 = {
        let ctx = qntt::su2::hess::weingarten_operator(&ham, &x);
        let (bl, br) = ctx.bases();
        let diag = ctx.a_diagonal();
        let seed = Su2TtTangent::project(bl, br, &diag.to_vector());
        let (_verdict, w, _a) = qntt::su2::hess::hessian_pd_probe_witness_with(
            &ctx,
            HessPart::Rr,
            std::slice::from_ref(&seed),
            150,
            FIXTURE_PROBE_SEED,
            &fixture_probe(),
        );
        let radial = su2_radial_mode_at(bl, br, bl);
        let mut vv = w;
        su2_deflate_radial_span(&radial, &mut vv);
        let n = vv.norm();
        vv.scale_mut(1.0 / n);
        su2_canonicalize_witness_sign(&mut vv, &seed);
        vv
    };
    let x0 = su2_kick_along(&v0, 5e-2);
    (ham, x, x0)
}

pub(crate) fn generic_kick_from<T: Scalar>(x: &TtVector<T>, eps: f64, seed: u64) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let r: TtVector<T> = qntt::tt_randn(x.space, &x.ranks, &mut rng);
    let (bl, br) = tangent_bases(x);
    let mut v = TtTangent::project(&bl, &br, &r);
    let radial = radial_mode_at(&bl, &br, &bl);
    deflate_radial_span(&radial, &mut v);
    let n = v.norm();
    v.scale_mut(T::from_f64(1.0 / n));
    kick_along(&v, eps)
}

pub(crate) fn su2_generic_kick_from<T: Scalar>(
    x: &Su2TtVector<T>,
    eps: f64,
    seed: u64,
) -> Su2TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let r: Su2TtVector<T> = qntt::su2::vector::tt_randn(x.space, &x.ranks, &mut rng);
    let (bl, br) = qntt::su2::riemannian::tangent_bases(x);
    let mut v = Su2TtTangent::project(&bl, &br, &r);
    let radial = su2_radial_mode_at(&bl, &br, &bl);
    su2_deflate_radial_span(&radial, &mut v);
    let n = v.norm();
    assert!(
        n > 1e-8,
        "SU(2) generic kick collapsed under radial deflation (‖v‖ = {n:.3e}) — the declared grid \
         has no non-radial tangent content at this seed"
    );
    v.scale_mut(T::from_f64(1.0 / n));
    su2_kick_along(&v, eps)
}

pub(crate) fn longest_frozen_streak(gnorms: &[f64], rel: f64) -> usize {
    let mut best = 0usize;
    let mut cur = 0usize;
    for k in 1..gnorms.len() {
        let prev = gnorms[k - 1];
        let d = (gnorms[k] - prev).abs();
        if d <= rel * prev.abs().max(f64::MIN_POSITIVE) {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
    }
    best
}

pub(crate) fn su2_full_rank_arm() -> (
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
) {
    use qntt::solvers::hess::HessPart;

    let space = Su2Space::new(4, 4, 0);
    let (t, v) = generic_truncated_tv::<f64>(4);
    let ham = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6AD2);
    let seed_state = su2_full_rank_state::<f64>(space, &mut rng);
    let xgs = su2_converged_minimizer(&ham, &seed_state);
    let v0 = {
        let ctx = qntt::su2::hess::weingarten_operator(&ham, &xgs);
        let (bl, br) = ctx.bases();
        assert_eq!(
            qntt::su2::weingarten::tangent_dim(bl),
            20,
            "not at full SU(2) representing rank (the Weyl–Paldus count f(4,4,0) = 20)"
        );
        let diag = ctx.a_diagonal();
        let seed = Su2TtTangent::project(bl, br, &diag.to_vector());
        let (_verdict, w, _a) = qntt::su2::hess::hessian_pd_probe_witness_with(
            &ctx,
            HessPart::Rr,
            std::slice::from_ref(&seed),
            200,
            FIXTURE_PROBE_SEED,
            &fixture_probe(),
        );
        let radial = su2_radial_mode_at(bl, br, bl);
        let mut vv = w;
        su2_deflate_radial_span(&radial, &mut vv);
        let n = vv.norm();
        vv.scale_mut(1.0 / n);
        su2_canonicalize_witness_sign(&mut vv, &seed);
        vv
    };
    let x0 = su2_kick_along(&v0, 5e-2);
    let g0 = su2_grad_exact_at(&ham, &x0).norm();
    assert!(
        g0 > 1e-4,
        "SU(2) full-rank kick left the state critical (‖grad‖ = {g0:.3e})"
    );
    (ham, xgs, x0)
}
