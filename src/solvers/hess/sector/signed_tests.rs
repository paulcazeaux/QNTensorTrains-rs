// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::cell::Cell;

use super::{SectorCtx, SectorOpts, SectorReadout};
use crate::core::SparseCore;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::tangent::{TtTangent, shifted_channel_is_empty, tangent_dim_grids};
use crate::hamiltonian::SparseHamiltonian;
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{QnSpace, RankGrid, max_ranks};
use crate::solvers::davidson::effective_diagonal_bruteforce;
use crate::solvers::hess::sector_response_tests::fx_exact_ground;
use crate::solvers::hess::{
    RpaKernel, RpaMetric, SympPrecond, SymplecticDavidsonOpts, omega_davidson_vectors,
    weingarten_operator,
};
use crate::solvers::riemannian::normalize_mut;
use crate::test_util::tangent_bits;
use crate::test_util::{pg_symmetric_tv, pg4_curved_state, seeded};
use crate::vector::{TtVector, tt_randn};

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn sector_lowest_on_a_pd_channel_is_bitwise_the_refusing_kernel() {
    let (space, t, v, ham, x) = fx_exact_ground();
    let ctx = weingarten_operator(&ham, &x);
    let (bl, _) = ctx.bases();
    let phi = (1..space.pg.order())
        .map(Irrep::new)
        .find(|&p| tangent_dim_grids(&space, &bl.ranks, p) > 0)
        .expect("fixture premise: fx_exact_ground carries a live shifted channel");
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let dim = sctx.dim();
    let k = 2usize;
    assert!(
        dim >= k + 2,
        "fixture premise: dim {dim} must leave room beyond the k = {k} roots and the seed block"
    );

    let basis = sctx.basis();
    let n = basis.len();
    let (a, b) = sctx.grams(&basis);
    let hrr: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
    let hii: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p - q).collect();
    let lmin_rr = dense_self_adjoint_smallest::<f64>(&hrr, n, 0).0[0];
    let lmin_ii = dense_self_adjoint_smallest::<f64>(&hii, n, 0).0[0];
    println!(
        "signed bitwise: phi {phi:?}, dim {dim}, n {n}, lambda_min(A+B) {lmin_rr:.6e}, \
         lambda_min(A-B) {lmin_ii:.6e}"
    );
    assert!(
        lmin_rr > 1e-3 && lmin_ii > 1e-3,
        "fixture premise: both blocks PD (A+B: {lmin_rr:e}, A-B: {lmin_ii:e}) — the bitwise \
         contract is stated only where no signature -1 vector and no negative Ritz value exists"
    );

    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        tol: 0.0,
        max_iter: 600,
        ..SectorOpts::default()
    };
    let nroots = k.min(dim);
    let v0s = sctx.seed_block(nroots + 2, opts.seed);
    assert!(
        !v0s.is_empty(),
        "fixture premise: the seed block survives P^phi"
    );
    let dopts = SymplecticDavidsonOpts {
        tol: opts.tol,
        max_basis: opts.max_iter,
        precond: SympPrecond::Off,
        ..SymplecticDavidsonOpts::default()
    };
    let applies = Cell::new(0usize);
    let bump = || applies.set(applies.get() + 1);
    let replay = omega_davidson_vectors(
        |w: &TtTangent<f64>| {
            bump();
            sctx.hpp_apply(w)
        },
        |v: &TtTangent<f64>| {
            bump();
            sctx.hmm_apply(v)
        },
        |_v: &mut TtTangent<f64>| {},
        |v: &mut TtTangent<f64>| v.gauge_fix_orthogonal_mut(),
        None::<fn(&mut TtTangent<f64>, f64)>,
        &v0s,
        nroots,
        dim,
        &dopts,
    );
    let (w2_old, vecs_old, info_old) = match replay {
        Ok(out) => out,
        Err(_) => panic!("fixture premise: the refusing kernel refused a PD channel"),
    };
    let applies_old = applies.get();

    let (resp, vecs) = sctx.sector_lowest_vectors(k, &opts);
    let w2 = resp
        .omega2()
        .expect("an Rpa readout publishes omega2()")
        .to_vec();
    println!(
        "signed bitwise: pre-P2 omega2 {w2_old:?} ({} iters, {applies_old} applies) | P2 omega2 \
         {w2:?} ({} iters, {} applies)",
        info_old.iterations, resp.iterations, resp.applies
    );
    assert!(
        bits(&w2) == bits(&w2_old),
        "omega^2 must be the refusing kernel's bit for bit"
    );
    assert_eq!(resp.iterations, info_old.iterations, "iterations");
    assert_eq!(
        resp.applies, applies_old,
        "applies: the counter wraps the two kernel closures only (f_W and mode checks are not \
         readout applies)"
    );
    assert_eq!(resp.converged, info_old.converged, "converged");
    let rpa = resp.rpa().expect("an Rpa readout");
    assert_eq!(
        vecs.len(),
        vecs_old.len(),
        "one Ritz vector per omega^2 entry"
    );
    for (j, (m, q)) in vecs.iter().zip(&vecs_old).enumerate() {
        let side = m.metric_side(RpaMetric::Ii);
        let mut d = side.clone();
        d.axpy(-side.dot(q) / q.dot(q), q);
        assert!(
            d.norm() <= 1e-12 * side.norm(),
            "mode {j}'s metric side is not the refusing kernel's Ritz vector (‖Δ‖/‖q‖ = {:e})",
            d.norm() / side.norm()
        );
    }
    let certificates = rpa.modes.iter().filter(|m| m.certificate.is_some()).count();
    assert_eq!(certificates, info_old.negative.len());
    assert_eq!(certificates, 0, "PD channel: no negative Ritz value");
    assert!(
        rpa.modes
            .iter()
            .all(|m| m.kind == crate::solvers::hess::RpaModeKind::Real && m.krein > 0.0),
        "PD channel: every Krein margin positive"
    );
    assert!(
        rpa.metric_events() == 0 && !resp.metric_indefinite(),
        "PD channel: no metric event"
    );
    assert_eq!(resp.index_lower_bound(), Some(0));
    assert_eq!(rpa.side(), RpaMetric::Ii, "the lane is reported");

    let (auto, vecs_auto) = sctx.sector_lowest_vectors(
        k,
        &SectorOpts {
            kernel: RpaKernel::signed(RpaMetric::Auto),
            ..opts.clone()
        },
    );
    let auto_rpa = auto.rpa().expect("Rpa");
    assert_eq!(
        auto_rpa.side(),
        RpaMetric::Ii,
        "Auto resolves to the Ii lane on the sector layer"
    );
    assert!(bits(&auto_rpa.omega2()) == bits(&w2), "Auto vs Ii: omega^2");
    assert_eq!(
        (auto.iterations, auto.applies),
        (resp.iterations, resp.applies)
    );
    assert!(bits(&auto_rpa.residuals()) == bits(&rpa.residuals()));
    for (p, q) in vecs_auto.iter().zip(&vecs) {
        assert!(
            tangent_bits(&p.x) == tangent_bits(&q.x) && tangent_bits(&p.y) == tangent_bits(&q.y),
            "Auto vs Ii: amplitudes"
        );
    }

    let plain = sctx.sector_lowest(k, &opts);
    let plain_rpa = plain.rpa().expect("Rpa");
    assert!(bits(&plain_rpa.omega2()) == bits(&w2));
    assert_eq!(
        (plain.iterations, plain.applies, plain.converged),
        (resp.iterations, resp.applies, resp.converged)
    );
    assert!(bits(&plain_rpa.residuals()) == bits(&rpa.residuals()));
    let kreins = |r: &crate::solvers::hess::RpaReadout| -> Vec<u64> {
        r.modes.iter().map(|m| m.krein.to_bits()).collect()
    };
    assert_eq!(kreins(plain_rpa), kreins(rpa));
}

#[test]
fn t8m_shifted_diagonal_matches_bruteforce() {
    let x = pg4_curved_state(0x7C01);
    let space = x.space;
    let d = space.d;
    let (t, v) = pg_symmetric_tv(&space.pg, d, 0x8D1A);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let ctx = weingarten_operator(&h, &x);
    let mut interior = 0usize;
    for p in 1..space.pg.order() {
        let phi = Irrep::new(p);
        let space_out = space.with_target(space.target.fuse(phi));
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        let lambda = sctx.lambda();
        let diag = sctx.a_diagonal();
        let (bl, _) = sctx.bases();
        let r: Vec<RankGrid> = bl.ranks.clone();
        let r_col: Vec<RankGrid> = r.iter().map(|g| g.relabel_irrep(phi)).collect();
        assert_eq!(diag.phi(), phi, "phi {p}: the diagonal's flux");
        assert_eq!(diag.rank(), &r[..], "phi {p}: row grids");
        assert_eq!(diag.rank_col(), &r_col[..], "phi {p}: column grids");
        let e = sctx.envs();
        assert!(
            !shifted_channel_is_empty(&e.r_phi[d - 1]),
            "phi {p}: pg4c's empty channels are kind B (the bond-d identity has a column)"
        );
        let counts: Vec<usize> = (0..d).map(|k| diag.component(k).num_coeffs()).collect();
        println!(
            "t8m shifted: phi {p}, dim {}, lambda {lambda:.6}, per-component coefficients {counts:?}",
            sctx.dim()
        );
        for k in 0..d {
            let comp = diag.component(k);
            assert_eq!(comp.space, space_out, "phi {p}: component {k} stamp");
            assert_eq!(comp.row_ranks, r[k], "phi {p}: component {k} row grid");
            assert_eq!(
                comp.col_ranks,
                r_col[k + 1],
                "phi {p}: component {k} column grid (r_col[k+1] = r[k+1] relabelled by phi)"
            );
            let template = SparseCore::<f64>::zeros(space_out, k, &r[k], &r_col[k + 1]);
            let mut op = |a: &SparseCore<f64>| {
                h_out.sketched_hamiltonian(a, &e.l[k], &e.r_phi[k], &e.ws_du[k])
            };
            let o = effective_diagonal_bruteforce(&mut op, &template);
            assert_eq!(
                comp.num_coeffs(),
                o.num_coeffs(),
                "phi {p}: shape at k = {k}"
            );
            for i in 0..o.num_coeffs() {
                let want = o.coeff(i) - lambda;
                let got = comp.coeff(i);
                assert!(
                    (got - want).abs() <= 1e-10 * (1.0 + want.abs()),
                    "phi {p}: diagonal coeff {i} at k = {k}: got {got}, want {want}"
                );
            }
            if o.num_coeffs() > 0 {
                assert!(
                    o.norm() > 1e-6,
                    "phi {p}: vacuous effective diagonal at k = {k} (|O| = {})",
                    o.norm()
                );
                if k > 0 && k + 1 < d {
                    interior += 1;
                }
            }
        }
    }
    assert!(
        interior >= 2,
        "non-vacuity: only {interior} interior components carried a diagonal"
    );
}

#[test]
fn t8m_shifted_diagonal_on_a_kind_a_channel_keeps_the_declared_shape() {
    let pg = PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(1), Irrep::new(0)],
    );
    let space = QnSpace::with_point_group(4, 2, 1, pg, Irrep::IDENTITY);
    let d = space.d;
    let (t, v) = pg_symmetric_tv(&pg, d, 0x8D1B);
    let h = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let caps = max_ranks(&space);
    let mut x: TtVector<f64> = tt_randn(space, &caps, &mut seeded(0x8D1C));
    normalize_mut(&mut x);
    let ctx = weingarten_operator(&h, &x);
    let mut reached = 0usize;
    for p in 1..pg.order() {
        let phi = Irrep::new(p);
        let space_out = space.with_target(space.target.fuse(phi));
        if max_ranks(&space_out)[d].total() != 0 {
            continue;
        }
        let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
        let sctx = SectorCtx::new(&ctx, &h_out, phi);
        assert_eq!(sctx.dim(), 0, "phi {p}: kind A is empty");
        let e = sctx.envs();
        assert!(
            shifted_channel_is_empty(&e.r_phi[d - 1]),
            "phi {p}: kind-A premise — the bond-d identity has no columns"
        );
        let lambda = sctx.lambda();
        let diag = sctx.a_diagonal();
        let (bl, _) = sctx.bases();
        let r: Vec<RankGrid> = bl.ranks.clone();
        let r_col: Vec<RankGrid> = r.iter().map(|g| g.relabel_irrep(phi)).collect();
        assert_eq!(diag.rank(), &r[..]);
        assert_eq!(diag.rank_col(), &r_col[..]);
        for k in 0..d {
            let comp = diag.component(k);
            assert_eq!(comp.row_ranks, r[k], "phi {p}: component {k} row grid");
            assert_eq!(
                comp.col_ranks,
                r_col[k + 1],
                "phi {p}: component {k} column grid — the DECLARED shape, not the degenerate \
                 sketch chain's"
            );
        }
        assert_eq!(
            diag.num_coeffs(),
            0,
            "phi {p}: phi lies outside the subgroup the site irreps generate, so every declared \
             block of a kind-A channel has a zero dimension (doc above)"
        );
        let _ = lambda;
        reached += 1;
    }
    assert!(
        reached >= 1,
        "fixture premise: [A, B, B, A] (2, 1) must carry a kind-A empty channel (phi = 2, 3)"
    );
}

fn d6_random_channel() -> (
    SparseHamiltonian<f64>,
    TtVector<f64>,
    SparseHamiltonian<f64>,
    Irrep,
) {
    let pg = PointGroup::new(
        2,
        &[
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
        ],
    );
    let space = QnSpace::with_point_group(6, 2, 2, pg, Irrep::IDENTITY);
    let mut t = crate::CoeffTensor::<f64>::zeros(6, 2);
    for p in 0..6 {
        t.set(&[p, p], 0.13 * (p as f64 - 3.0));
    }
    for (p, q) in [(0usize, 3usize), (1, 4), (2, 5)] {
        let h = -0.8 + 0.1 * p as f64;
        t.set(&[p, q], h);
        t.set(&[q, p], h);
    }
    let mut v = crate::CoeffTensor::<f64>::zeros(6, 4);
    for p in 0..6 {
        for q in 0..6 {
            if p != q {
                v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
                v.set(&[p, q, q, p], 0.23 / (1.0 + (p as f64 - q as f64).abs()));
            }
        }
    }
    for idx in [[0, 3, 1, 4], [1, 4, 0, 3], [3, 0, 4, 1], [4, 1, 3, 0]] {
        v.set(&idx, 0.19);
    }
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xD06A);
    let mut x: TtVector<f64> = crate::test_util::randomized_state(space, 1, 1, &mut rng);
    x.round_mut(1e-8, crate::qn::Pooled::NONE, crate::round::RoundMode::Rel);
    normalize_mut(&mut x);
    let ctx = weingarten_operator(&ham, &x);
    let (bl, _) = ctx.bases();
    let phi = (1..space.pg.order())
        .map(Irrep::new)
        .find(|&p| tangent_dim_grids(&space, &bl.ranks, p) >= 20)
        .expect("fixture premise: a live channel of dim >= 20");
    let space_out = space.with_target(space.target.fuse(phi));
    let h_out = SparseHamiltonian::<f64>::new(space_out, &t, &v, None, f64::EPSILON);
    (ham, x, h_out, phi)
}

#[test]
fn sector_lowest_d0_model_is_the_squared_shifted_diagonal() {
    use crate::solvers::davidson::precond_fixed_inverse;
    use crate::solvers::hess::davidson::REFUSE_TAU;
    use crate::solvers::hess::davidson_signed::omega_davidson_signed;
    let (ham, x, h_out, phi) = d6_random_channel();
    let ctx = weingarten_operator(&ham, &x);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let dim = sctx.dim();
    let k = 2usize;
    let opts = SectorOpts {
        readout: SectorReadout::Rpa,
        kernel: RpaKernel::signed(RpaMetric::Ii),
        precond: SympPrecond::DiagUnshifted { kappa: 0.0 },
        max_iter: 32,
        ..SectorOpts::default()
    };
    let (resp, vecs) = sctx.sector_lowest_vectors(k, &opts);
    let w2 = resp.omega2().expect("Rpa").to_vec();
    let off = sctx.sector_lowest(
        k,
        &SectorOpts {
            precond: SympPrecond::Off,
            ..opts.clone()
        },
    );

    let replay = |square: bool| {
        let mut model = sctx.a_diagonal();
        if square {
            model.for_each_coeff_mut(|c| *c = *c * *c);
        }
        precond_fixed_inverse(&mut model, 0.0);
        let precond = Some(move |t: &mut TtTangent<f64>, _theta: f64| {
            t.zip_coeffs_mut(&model, |ti, di| *ti *= di);
        });
        let v0s = sctx.seed_block(k.min(dim) + 2, opts.seed);
        let dopts = SymplecticDavidsonOpts {
            tol: opts.tol,
            max_basis: opts.max_iter,
            precond: opts.precond,
            ..SymplecticDavidsonOpts::default()
        };
        let applies = Cell::new(0usize);
        let bump = || applies.set(applies.get() + 1);
        let out = omega_davidson_signed(
            |w: &TtTangent<f64>| {
                bump();
                sctx.hpp_apply(w)
            },
            |v: &TtTangent<f64>| {
                bump();
                sctx.hmm_apply(v)
            },
            |_v: &mut TtTangent<f64>| {},
            |v: &mut TtTangent<f64>| v.gauge_fix_orthogonal_mut(),
            precond,
            &v0s,
            k.min(dim),
            dim,
            &dopts,
            REFUSE_TAU,
        );
        (out, applies.get())
    };
    let (sq, sq_applies) = replay(true);
    let (lin, lin_applies) = replay(false);
    println!(
        "D0 model pin: phi {phi:?}, dim {dim}: production {w2:?} ({} iters, {} applies) | Off {:?} \
         ({} iters, {} applies) | replay D_A^2 {:?} ({} applies) | replay D_A {:?} ({} applies)",
        resp.iterations,
        resp.applies,
        off.omega2(),
        off.iterations,
        off.applies,
        sq.omega2,
        sq_applies,
        lin.omega2,
        lin_applies
    );
    assert!(
        bits(&w2) != bits(&off.omega2().expect("Rpa"))
            || (resp.iterations, resp.applies) != (off.iterations, off.applies),
        "premise: D0 is bitwise Off on this channel — the model never reached the kernel"
    );
    let resp_r = resp.rpa().expect("Rpa");
    let (lin, _) = crate::solvers::hess::readout_from_signed(lin, RpaMetric::Ii, lin_applies);
    let (sq, sq_amps) = crate::solvers::hess::readout_from_signed(sq, RpaMetric::Ii, sq_applies);
    assert!(
        bits(&lin.omega2()) != bits(&w2)
            || lin_applies != resp.applies
            || bits(&lin.residuals()) != bits(&resp_r.residuals()),
        "premise: the unsquared model is bitwise the production run — this channel cannot tell \
         D_A from D_A^2"
    );
    assert!(
        bits(&sq.omega2()) == bits(&w2),
        "omega^2: the production D0 is not precond_fixed_inverse(a_diagonal^2)"
    );
    assert_eq!(sq_applies, resp.applies, "applies");
    assert_eq!(sq.iterations, resp.iterations, "iterations");
    assert!(
        bits(&sq.residuals()) == bits(&resp_r.residuals()),
        "residuals"
    );
    assert_eq!(sq_amps.len(), vecs.len());
    for (j, (p, q)) in sq_amps.iter().zip(&vecs).enumerate() {
        assert!(
            tangent_bits(&p.x) == tangent_bits(&q.x) && tangent_bits(&p.y) == tangent_bits(&q.y),
            "mode {j}'s amplitudes differ from the replay's"
        );
    }
}

#[test]
fn sector_lowest_paired_is_the_kernel() {
    use crate::solvers::davidson::precond_fixed_inverse;
    use crate::solvers::hess::{
        PairedHook, PairedPrecond, omega_davidson_paired, paired_image, paired_mu, paired_route,
        readout_from_paired,
    };
    let (ham, x, h_out, phi) = d6_random_channel();
    let ctx = weingarten_operator(&ham, &x);
    let sctx = SectorCtx::new(&ctx, &h_out, phi);
    let dim = sctx.dim();
    let k = 2usize;
    let d_a = sctx.a_diagonal();
    let dmax = (0..d_a.num_coeffs())
        .map(|i| d_a.coeff(i))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (1e-100..=1e100).contains(&dmax),
        "premise: D_A^phi peaks inside the guard's untouched band ({dmax:e})"
    );
    let mut runs = Vec::new();
    for precond in [SympPrecond::Off, SympPrecond::DiagUnshifted { kappa: 0.0 }] {
        let opts = SectorOpts {
            readout: SectorReadout::Rpa,
            precond,
            max_iter: 32,
            ..SectorOpts::default()
        };
        assert!(
            matches!(opts.kernel, RpaKernel::Paired),
            "the sector default"
        );
        let (resp, vecs) = sctx.sector_lowest_vectors(k, &opts);
        let rpa = resp.rpa().expect("Rpa").clone();
        let v0s = sctx.seed_block(k.min(dim) + 2, opts.seed);
        let dopts = SymplecticDavidsonOpts {
            tol: opts.tol,
            max_basis: opts.max_iter,
            max_subspace: opts.max_subspace,
            precond,
            kernel: RpaKernel::Paired,
            ..SymplecticDavidsonOpts::default()
        };
        let (pp, popts) = paired_route(&dopts).expect("the paired route");
        let model: Option<TtTangent<f64>> = match pp {
            PairedPrecond::Off => None,
            PairedPrecond::Unshifted { kappa } => {
                let mut m = sctx.a_diagonal();
                precond_fixed_inverse(&mut m, kappa);
                Some(m)
            }
            PairedPrecond::Shifted => unreachable!("no shifted arm on the sector layer"),
        };
        let pc = model.map(|m| {
            move |t: &mut TtTangent<f64>, _s: f64| t.zip_coeffs_mut(&m, |ti, di| *ti *= di)
        });
        let mu = paired_mu(PairedHook::Orthogonal, &d_a);
        let clean = |v: &mut TtTangent<f64>| v.gauge_fix_orthogonal_mut();
        let out = omega_davidson_paired(
            |v| paired_image(clean, |q| sctx.a_apply(q), mu, v, true),
            |v| paired_image(clean, |q| sctx.b_apply(q), mu, v, false),
            clean,
            pc,
            false,
            &v0s,
            &[],
            k.min(dim),
            dim,
            &popts,
        );
        let (rep, rep_amps) = readout_from_paired(out);
        println!(
            "paired replay [{precond:?}]: phi {phi:?}, dim {dim}: production {:?} ({} built, {} \
             pairs) | replay {:?} ({} built, {} pairs)",
            rpa.omega2(),
            rpa.iterations,
            rpa.applies,
            rep.omega2(),
            rep.iterations,
            rep.applies
        );
        assert!(
            bits(&rep.omega2()) == bits(&rpa.omega2()),
            "[{precond:?}] omega^2"
        );
        assert_eq!(rep.iterations, rpa.iterations, "[{precond:?}] built");
        assert_eq!(rep.applies, rpa.applies, "[{precond:?}] apply pairs");
        assert!(
            bits(&rep.residuals()) == bits(&rpa.residuals()),
            "[{precond:?}] residuals"
        );
        assert_eq!(rep_amps.len(), vecs.len());
        for (j, (p, q)) in rep_amps.iter().zip(&vecs).enumerate() {
            assert!(
                tangent_bits(&p.x) == tangent_bits(&q.x)
                    && tangent_bits(&p.y) == tangent_bits(&q.y),
                "[{precond:?}] mode {j}: amplitudes"
            );
        }
        runs.push(rpa);
    }
    assert!(
        bits(&runs[0].omega2()) != bits(&runs[1].omega2())
            || (runs[0].iterations, runs[0].applies) != (runs[1].iterations, runs[1].applies),
        "premise: D0 is bitwise Off on this channel — the model never reached the kernel"
    );
}
