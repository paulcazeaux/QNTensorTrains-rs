// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::cell::Cell;

use super::Su2SectorCtx;
use super::csa_bundle_tests::{live_channel_fixture, zero_width_channel_fixture};
use crate::solvers::InnerProductVector;
use crate::solvers::davidson::{effective_diagonal_bruteforce, precond_fixed_inverse};
use crate::solvers::hess::{
    REFUSE_TAU, RpaKernel, RpaMetric, SectorOpts, SectorReadout, SympPrecond,
    SymplecticDavidsonOpts, build_symp_precond, omega_davidson_signed, sector_d0_precond,
};
use crate::su2::core::Su2Core;
use crate::su2::hess::weingarten_operator;
use crate::su2::qn::Su2Space;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use crate::test_util::su2_tangent_bits;
use crate::test_util::{generic_su2_h, randomized_su2_state, seeded};

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn run<F: FnMut(&mut Su2TtTangent<f64>, f64)>(
    p: Option<F>,
    v: &Su2TtTangent<f64>,
) -> Option<Su2TtTangent<f64>> {
    p.map(|mut f| {
        let mut w = v.clone();
        f(&mut w, 0.37);
        w
    })
}

fn dmax_of(m: &Su2TtTangent<f64>) -> f64 {
    let mut v = m.clone();
    let mut dm = 0.0f64;
    v.for_each_coeff_mut(|c| dm = dm.max(c.max(0.0)));
    dm
}

#[test]
fn t8m_su2_shifted_diagonal_matches_bruteforce() {
    let (space, ham, x) = live_channel_fixture();
    let d = space.d;
    let ctx = weingarten_operator(&ham, &x);
    let mut live = 0usize;
    for two_k in [2usize, 4] {
        let out_space = space.with_two_s(two_k);
        let h_out = generic_su2_h::<f64>(out_space);
        let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
        let dim = sctx.dim();
        if dim == 0 {
            println!("t8m su2: 2κ = {two_k}: empty channel");
            continue;
        }
        live += 1;
        let lambda = sctx.lambda();
        let diag = sctx.a_diagonal();
        let rm = sctx.dressed.rm.clone();
        let rd = sctx.dressed.rd.clone();
        assert_eq!(diag.two_s_flux(), two_k, "2κ {two_k}: the diagonal's flux");
        assert_eq!(diag.rank(), &rm[..], "2κ {two_k}: row grids = rm");
        assert_eq!(diag.rank_col(), &rd[..], "2κ {two_k}: column grids = rd");
        let e = sctx.envs();
        let before = std::ptr::from_ref(e);
        let mut nonvacuous = 0usize;
        let mut counts = Vec::with_capacity(d);
        for k in 0..d {
            let comp = diag.component(k);
            assert_eq!(comp.space, out_space, "2κ {two_k}: component {k} stamp");
            assert_eq!(comp.row_ranks, rm[k], "2κ {two_k}: component {k} row grid");
            assert_eq!(
                comp.col_ranks,
                rd[k + 1],
                "2κ {two_k}: component {k} column grid"
            );
            let template = Su2Core::<f64>::zeros(out_space, k, &rm[k], &rd[k + 1]);
            let mut op = |a: &Su2Core<f64>| {
                h_out.sketched_hamiltonian(a, &e.l[k], &e.r_kappa[k], &e.ws_du[k])
            };
            let o = effective_diagonal_bruteforce(&mut op, &template);
            assert_eq!(
                comp.num_coeffs(),
                o.num_coeffs(),
                "2κ {two_k}: shape at k = {k}"
            );
            counts.push(o.num_coeffs());
            for i in 0..o.num_coeffs() {
                let want = o.coeff(i) - lambda;
                let got = comp.coeff(i);
                assert!(
                    (got - want).abs() <= 1e-10 * (1.0 + want.abs()),
                    "2κ {two_k}: diagonal coeff {i} at k = {k}: got {got}, want {want}"
                );
            }
            if o.norm() > 1e-6 {
                nonvacuous += 1;
            }
        }
        println!(
            "t8m su2: 2κ = {two_k}, dim {dim}, λ {lambda:.6}, per-component coefficients {counts:?}, \
             non-vacuous components {nonvacuous}"
        );
        if two_k == 2 {
            assert!(
                nonvacuous >= d - 1,
                "non-vacuity: only {nonvacuous} components carried a diagonal"
            );
            assert!(
                counts[0] >= 1 && counts[d - 1] >= 1,
                "non-vacuity: the SU(2) boundary components are live ({counts:?})"
            );
        }
        let basis = sctx.basis();
        let _ = sctx.a_apply(&basis[0]);
        assert_eq!(
            std::ptr::from_ref(sctx.envs.get().expect("built")),
            before,
            "2κ {two_k}: a_diagonal must share the ONE channel bundle the applies use"
        );
    }
    assert!(live >= 1, "fixture premise: a live channel");
}

#[test]
fn t8m_su2_shifted_diagonal_on_a_zero_width_channel_keeps_the_declared_shape() {
    let (space, ham, x) = zero_width_channel_fixture();
    let two_k = 2usize;
    let ctx = weingarten_operator(&ham, &x);
    let h_out = generic_su2_h::<f64>(space.with_two_s(two_k));
    let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
    assert_eq!(sctx.dim(), 0, "fixture premise: zero-width");
    let diag = sctx.a_diagonal();
    assert_eq!(
        diag.num_coeffs(),
        0,
        "a zero-width channel carries no coefficient"
    );
    assert_eq!(diag.two_s_flux(), two_k);
    assert_eq!(diag.rank(), &sctx.dressed.rm[..], "declared row grids");
    assert_eq!(
        diag.rank_col(),
        &sctx.dressed.rd[..],
        "declared column grids"
    );
    for k in 0..space.d {
        assert_eq!(
            diag.component(k).row_ranks,
            sctx.dressed.rm[k],
            "component {k} rows"
        );
        assert_eq!(
            diag.component(k).col_ranks,
            sctx.dressed.rd[k + 1],
            "component {k} cols"
        );
    }
}

#[test]
fn sector_d0_guard_skips_collapsed_and_rescales_extreme_models() {
    let (space, ham, x) = live_channel_fixture();
    let ctx = weingarten_operator(&ham, &x);
    let h_out = generic_su2_h::<f64>(space.with_two_s(2));
    let sctx = Su2SectorCtx::new(&ctx, &h_out, 2);
    let base = sctx.a_diagonal();
    let dm0 = dmax_of(&base);
    assert!(
        (1e-100..=1e100).contains(&dm0),
        "premise: the live model is in band with a positive part ({dm0:e})"
    );
    let choice = SympPrecond::DiagUnshifted { kappa: 0.0 };
    let calls = Cell::new(0usize);
    let ones = {
        let mut o = base.clone();
        o.for_each_coeff_mut(|c| *c = 1.0);
        o
    };
    let r = run(
        sector_d0_precond(SympPrecond::Off, || {
            calls.set(calls.get() + 1);
            base.clone()
        }),
        &ones,
    );
    assert!(r.is_none(), "(i) Off is None");
    assert_eq!(calls.get(), 0, "(i) Off never builds the model");
    let mut neg = base.clone();
    neg.for_each_coeff_mut(|c| *c = -c.abs() - 1.0);
    let r = run(
        sector_d0_precond(choice, || {
            calls.set(calls.get() + 1);
            neg.clone()
        }),
        &ones,
    );
    assert!(r.is_none(), "(ii) a collapsed model skips D0");
    assert_eq!(calls.get(), 1, "(ii) the model was read once");
    let tiny = base.scale(1e-120 / dm0);
    let dmt = dmax_of(&tiny);
    assert!(dmt < 1e-100, "premise: below the band ({dmt:e})");
    let got = run(sector_d0_precond(choice, || tiny.clone()), &ones).expect("(iii) Some");
    let mut m = tiny.clone();
    let s = 1.0 / dmt;
    m.for_each_coeff_mut(|c| *c *= s);
    precond_fixed_inverse(&mut m, 0.0);
    let mut want = ones.clone();
    want.zip_coeffs_mut(&m, |ti, di| *ti *= di);
    let mut m_raw = tiny.clone();
    precond_fixed_inverse(&mut m_raw, 0.0);
    let mut unscaled = ones.clone();
    unscaled.zip_coeffs_mut(&m_raw, |ti, di| *ti *= di);
    assert!(
        su2_tangent_bits(&got) == su2_tangent_bits(&want),
        "(iii) the rescaled recipe, bit for bit"
    );
    assert!(
        su2_tangent_bits(&got) != su2_tangent_bits(&unscaled),
        "(iii) non-vacuity: the rescale changed the preconditioner"
    );
    let huge = base.scale(1e120 / dm0);
    let dmh = dmax_of(&huge);
    assert!(dmh > 1e100, "premise: above the band ({dmh:e})");
    let got = run(sector_d0_precond(choice, || huge.clone()), &ones).expect("(iii') Some");
    let mut m = huge.clone();
    let s = 1.0 / dmh;
    m.for_each_coeff_mut(|c| *c *= s);
    precond_fixed_inverse(&mut m, 0.0);
    let mut want = ones.clone();
    want.zip_coeffs_mut(&m, |ti, di| *ti *= di);
    let mut m_raw = huge.clone();
    precond_fixed_inverse(&mut m_raw, 0.0);
    let mut unscaled = ones.clone();
    unscaled.zip_coeffs_mut(&m_raw, |ti, di| *ti *= di);
    assert!(
        su2_tangent_bits(&got) == su2_tangent_bits(&want),
        "(iii') above the band: the rescaled recipe, bit for bit"
    );
    assert!(
        su2_tangent_bits(&got) != su2_tangent_bits(&unscaled),
        "(iii') non-vacuity: the rescale changed the preconditioner"
    );
    let got = run(sector_d0_precond(choice, || base.clone()), &ones).expect("(iv) Some");
    let want = run(build_symp_precond(choice, || base.clone()), &ones).expect("(iv) Some");
    assert!(
        su2_tangent_bits(&got) == su2_tangent_bits(&want),
        "(iv) in band: the plain factory recipe"
    );
    println!(
        "d0 guard: dmax {dm0:.3e}, tiny dmax {dmt:.3e}, model calls {}",
        calls.get()
    );
}

#[test]
fn sector_lowest_d0_model_is_the_squared_shifted_diagonal_su2() {
    let space = Su2Space::new(6, 6, 0);
    let ham = generic_su2_h::<f64>(space);
    let mut rng = seeded(0xD06B);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let ctx = weingarten_operator(&ham, &x);
    let two_k = 2usize;
    let h_out = generic_su2_h::<f64>(space.with_two_s(two_k));
    let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
    let dim = sctx.dim();
    println!("d0 model pin su2: triplet dim {dim}");
    assert!(dim >= 20, "fixture premise: triplet dim {dim} ≥ 20");
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
        let precond = Some(move |t: &mut Su2TtTangent<f64>, _theta: f64| {
            t.zip_coeffs_mut(&model, |ti, di| *ti *= di);
        });
        let v0s = sctx.seed_block(k.min(dim) + 2, opts.seed);
        let dopts = SymplecticDavidsonOpts {
            tol: opts.tol,
            max_basis: opts.max_iter,
            max_subspace: opts.max_subspace,
            precond: opts.precond,
            ..SymplecticDavidsonOpts::default()
        };
        let applies = Cell::new(0usize);
        let bump = || applies.set(applies.get() + 1);
        let out = omega_davidson_signed(
            |w: &Su2TtTangent<f64>| {
                bump();
                sctx.hpp_apply(w)
            },
            |v: &Su2TtTangent<f64>| {
                bump();
                sctx.hmm_apply(v)
            },
            |_v: &mut Su2TtTangent<f64>| {},
            |v: &mut Su2TtTangent<f64>| v.gauge_fix_orthogonal_mut(),
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
        "d0 model pin su2: production {w2:?} ({} iters, {} applies) | Off {:?} ({} iters, {} \
         applies) | replay D_A^2 {:?} ({sq_applies} applies) | replay D_A {:?} ({lin_applies} \
         applies)",
        resp.iterations,
        resp.applies,
        off.omega2(),
        off.iterations,
        off.applies,
        sq.omega2,
        lin.omega2
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
        "premise: the unsquared model is bitwise the production run"
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
            su2_tangent_bits(&p.x) == su2_tangent_bits(&q.x)
                && su2_tangent_bits(&p.y) == su2_tangent_bits(&q.y),
            "mode {j}'s amplitudes differ from the replay's"
        );
    }
}

#[test]
fn sector_lowest_paired_is_the_kernel_su2() {
    use crate::solvers::davidson::precond_fixed_inverse;
    use crate::solvers::hess::{
        PairedHook, PairedPrecond, RpaKernel, omega_davidson_paired, paired_image, paired_mu,
        paired_route, readout_from_paired,
    };
    let space = Su2Space::new(6, 6, 0);
    let ham = generic_su2_h::<f64>(space);
    let mut rng = seeded(0xD06B);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let ctx = weingarten_operator(&ham, &x);
    let two_k = 2usize;
    let h_out = generic_su2_h::<f64>(space.with_two_s(two_k));
    let sctx = Su2SectorCtx::new(&ctx, &h_out, two_k);
    let dim = sctx.dim();
    assert!(dim >= 20, "fixture premise: triplet dim {dim} ≥ 20");
    let k = 2usize;
    let d_a = sctx.a_diagonal();
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
        let model: Option<Su2TtTangent<f64>> = match pp {
            PairedPrecond::Off => None,
            PairedPrecond::Unshifted { kappa } => {
                let mut m = sctx.a_diagonal();
                precond_fixed_inverse(&mut m, kappa);
                Some(m)
            }
            PairedPrecond::Shifted => unreachable!("no shifted arm on the sector layer"),
        };
        let pc = model.map(|m| {
            move |t: &mut Su2TtTangent<f64>, _s: f64| t.zip_coeffs_mut(&m, |ti, di| *ti *= di)
        });
        let mu = paired_mu(PairedHook::Orthogonal, &d_a);
        let clean = |v: &mut Su2TtTangent<f64>| v.gauge_fix_orthogonal_mut();
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
            "paired replay su2 [{precond:?}]: dim {dim}: production {:?} ({} built, {} pairs) | \
             replay {:?} ({} built, {} pairs)",
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
                su2_tangent_bits(&p.x) == su2_tangent_bits(&q.x)
                    && su2_tangent_bits(&p.y) == su2_tangent_bits(&q.y),
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
