// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::scalar::Scalar;
use crate::solvers::hess::{
    PairedDavidsonOpts, PairedDavidsonOut, PairedHook, PairedPrecond, omega_davidson_paired,
    paired_image, paired_mu, paired_precond, realified,
};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use std::sync::Arc;

use super::{Su2WeingartenCtx, clean_tangent, radial_mode, reproject_tangent};

fn su2_clean<T: Scalar>(
    hook: PairedHook,
    base_l: &Arc<Su2TtVector<T>>,
    base_r: &Arc<Su2TtVector<T>>,
    radial: Option<&Su2TtTangent<T>>,
    v: &mut Su2TtTangent<T>,
) {
    match hook {
        PairedHook::Oblique => reproject_tangent(base_l, base_r, radial, v),
        PairedHook::Orthogonal => clean_tangent(radial, v),
    }
}

fn su2_image<T: Scalar>(
    ctx: &Su2WeingartenCtx<'_, T>,
    hook: PairedHook,
    radial: Option<&Su2TtTangent<T>>,
    mu: f64,
    v: &Su2TtTangent<T>,
    a_side: bool,
) -> Su2TtTangent<T> {
    let (base_l, base_r) = ctx.shared_bases();
    paired_image(
        |w: &mut Su2TtTangent<T>| su2_clean(hook, base_l, base_r, radial, w),
        |q| {
            if a_side {
                ctx.a_apply(q)
            } else {
                ctx.b_apply(q)
            }
        },
        mu,
        v,
        a_side,
    )
}

/// The lowest RPA modes at `ctx`'s point by the paired-basis Davidson on the SU(2) layer — see
/// [`crate::solvers::hess::paired_lowest_davidson`] for the contract and the options, all shared.
/// Seeds and target are reduced-multiplet tangents at `ctx`'s frames (typically `k + 2` random
/// ones built by `Su2TtTangent::project`).
///
/// # Panics
///
/// On a spin-flux seed or target (`two_s_flux != 0` — the cross-spin channels are
/// `Su2SectorCtx`), and on every refusal of the shared kernel.
#[allow(clippy::too_many_arguments)]
pub fn paired_lowest_davidson<T: Scalar>(
    ctx: &Su2WeingartenCtx<'_, T>,
    v0s: &[Su2TtTangent<T>],
    target: &[Su2TtTangent<T>],
    k: usize,
    precond: PairedPrecond,
    tda: bool,
    hook: PairedHook,
    opts: &PairedDavidsonOpts,
) -> PairedDavidsonOut<Su2TtTangent<T>> {
    for (i, v) in v0s.iter().enumerate() {
        v.refuse_if_shifted(&format!("su2 paired_lowest_davidson (seed {i})"));
    }
    for (i, v) in target.iter().enumerate() {
        v.refuse_if_shifted(&format!("su2 paired_lowest_davidson (target {i})"));
    }
    let (base_l, base_r) = ctx.shared_bases();
    let radial = radial_mode(ctx);
    let radial = radial.as_ref();
    let space_dim = realified(T::IS_COMPLEX)
        * (crate::su2::weingarten::tangent_dim(base_l) - radial.map_or(0, |_| 1));
    ctx.warm_a_caches();
    if !tda {
        ctx.warm_b_caches();
    }
    let d_a = ctx.a_diagonal();
    let mu = paired_mu(hook, &d_a);
    let precond_fn = paired_precond(precond, d_a);
    let clean = |v: &mut Su2TtTangent<T>| su2_clean(hook, base_l, base_r, radial, v);
    omega_davidson_paired(
        |v| su2_image(ctx, hook, radial, mu, v, true),
        |v| su2_image(ctx, hook, radial, mu, v, false),
        clean,
        precond_fn,
        tda,
        v0s,
        target,
        k,
        space_dim,
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::hess::{AnomalyRule, assert_paired_wanted};
    use crate::su2::hamiltonian::Su2SparseHamiltonian;
    use crate::su2::hess::{deflate_radial, hessian_grams, weingarten_operator};
    use crate::su2::qn::Su2Space;
    use crate::su2::weingarten::tangent_basis;
    use crate::test_util::{generic_su2_h, randomized_su2_state};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn point() -> (Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
        let space = Su2Space::new(4, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5DA1);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
        x.scale_mut(1.0 / x.norm_lr(false));
        (generic_su2_h::<f64>(space), x)
    }

    fn seeds(ctx: &Su2WeingartenCtx<'_, f64>, n: usize, seed: u64) -> Vec<Su2TtTangent<f64>> {
        let (bl, br) = ctx.bases();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..n)
            .map(|_| {
                let d: Su2TtVector<f64> = randomized_su2_state(bl.space, 1, 2, &mut rng);
                Su2TtTangent::project(bl, br, &d)
            })
            .collect()
    }

    fn grams(ctx: &Su2WeingartenCtx<'_, f64>) -> (Vec<f64>, Vec<f64>, usize) {
        let (bl, br) = ctx.bases();
        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let (a, b) = hessian_grams(ctx, &basis);
        (a, b, n)
    }

    #[test]
    fn paired_su2_live_tangent_space_matches_the_dense_grams() {
        let (h, x) = point();
        let ctx = weingarten_operator(&h, &x);
        let (a, b, n) = grams(&ctx);
        let k = 3;
        let v0s = seeds(&ctx, k + 2, 0x5DA2);
        let opts = PairedDavidsonOpts {
            tol: 1e-11,
            max_basis: 400,
            max_subspace: n,
            ..PairedDavidsonOpts::default()
        };
        for hook in [PairedHook::Orthogonal, PairedHook::Oblique] {
            for pre in [
                PairedPrecond::Off,
                PairedPrecond::Unshifted { kappa: 1e3 },
                PairedPrecond::Shifted,
            ] {
                let out = paired_lowest_davidson(&ctx, &v0s, &[], k, pre, false, hook, &opts);
                assert_paired_wanted(
                    &format!("su2 d4 {hook:?} {pre:?}"),
                    &out,
                    &a,
                    &b,
                    n,
                    k,
                    1e-8,
                );
            }
        }
        let tda = paired_lowest_davidson(
            &ctx,
            &v0s,
            &[],
            k,
            PairedPrecond::Off,
            true,
            PairedHook::Orthogonal,
            &opts,
        );
        let zero = vec![0.0; n * n];
        assert_paired_wanted("su2 d4 tda", &tda, &a, &zero, n, k, 1e-8);
    }

    #[test]
    fn paired_su2_exhausts_the_realized_space_below_the_structural_dim() {
        let (h, x) = point();
        let ctx = weingarten_operator(&h, &x);
        let (a, b, n) = grams(&ctx);
        let structural = crate::su2::weingarten::tangent_dim(ctx.bases().0);
        assert!(
            n + 1 < structural,
            "fixture premise: realized {n} + radial below structural {structural}"
        );
        let k = 3;
        let v0s = seeds(&ctx, k + 2, 0x5DA2);
        let opts = PairedDavidsonOpts {
            tol: 1e-17,
            max_basis: 400,
            max_subspace: structural,
            ..PairedDavidsonOpts::default()
        };
        for pre in [PairedPrecond::Off, PairedPrecond::Unshifted { kappa: 1e3 }] {
            let out = paired_lowest_davidson(
                &ctx,
                &v0s,
                &[],
                k,
                pre,
                false,
                PairedHook::Orthogonal,
                &opts,
            );
            let i = &out.info;
            println!(
                "su2 d4 {pre:?}: n {n} structural {structural} built {} exhausted {} converged {} \
                 stalled {}",
                i.built, i.exhausted, i.converged, i.stalled
            );
            assert!(
                i.exhausted && i.converged && !i.stalled,
                "{pre:?}: not exhausted"
            );
            assert!(i.built < 400, "{pre:?}: ran to the budget");
            assert_paired_wanted(
                &format!("su2 d4 exhausted {pre:?}"),
                &out,
                &a,
                &b,
                n,
                k,
                1e-8,
            );
        }
    }

    #[test]
    fn selective_ranks_behind_the_su2_signed_entries_are_the_kernel() {
        use crate::solvers::hess::{SymplecticDavidsonOpts, WantedRank, paired_route};
        use crate::su2::hess::{
            symplectic_lowest_signed_davidson, symplectic_lowest_signed_davidson_targeted,
        };
        let (h, x) = point();
        let ctx = weingarten_operator(&h, &x);
        let k = 3;
        let v0s = seeds(&ctx, k + 2, 0x5DA2);
        let target = seeds(&ctx, 2, 0x5DA3);
        let base = SymplecticDavidsonOpts {
            tol: 1e-10,
            max_basis: 400,
            ..SymplecticDavidsonOpts::default()
        };
        let mut nonempty = 0;
        for (label, rank, tgt) in [
            ("AExcess(1.05)", WantedRank::AExcess(1.05), &[][..]),
            ("target, Omega", WantedRank::Omega, &target[..]),
            (
                "target, TargetGateOrExcess(0.9, 1.05)",
                WantedRank::TargetGateOrExcess(0.9, 1.05),
                &target[..],
            ),
        ] {
            let opts = SymplecticDavidsonOpts { rank, ..base };
            let (pre, popts) = paired_route(&opts).expect("a paired route");
            let direct = paired_lowest_davidson(
                &ctx,
                &v0s,
                tgt,
                k,
                pre,
                false,
                PairedHook::Orthogonal,
                &popts,
            );
            let (so, _) = if tgt.is_empty() {
                symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts)
            } else {
                symplectic_lowest_signed_davidson_targeted(&ctx, &v0s, tgt, k, &opts)
            };
            assert_eq!(so.omega2(), direct.omega2(), "{label}: entry ≠ kernel");
            assert_eq!(so.applies, direct.info.apply_pairs, "{label}");
            assert_eq!(
                so.passed_over.len(),
                direct.untargeted_below.len(),
                "{label}"
            );
            for (p, m) in so.passed_over.iter().zip(&direct.untargeted_below) {
                assert_eq!(
                    (p.omega, p.krein, p.rho_a_over_omega, p.residual),
                    (m.omega.re, m.krein, m.rho_a / m.omega.re, m.residual),
                    "{label}"
                );
            }
            println!("{label}: passed over {:?}", so.passed_over);
            nonempty += usize::from(!so.passed_over.is_empty());
        }
        assert!(nonempty > 0, "vacuous: no arm passed a pair over");
    }

    #[test]
    fn paired_su2_kernel_behind_the_signed_entry_is_the_kernel() {
        use crate::solvers::hess::{RpaMetric, RpaReport, SymplecticDavidsonOpts, paired_route};
        use crate::su2::hess::symplectic_lowest_signed_davidson;
        let (h, x) = point();
        let ctx = weingarten_operator(&h, &x);
        let k = 3;
        let v0s = seeds(&ctx, k + 2, 0x5DA2);
        let opts = SymplecticDavidsonOpts {
            tol: 1e-10,
            max_basis: 400,
            ..SymplecticDavidsonOpts::default()
        };
        let (pre, popts) = paired_route(&opts).expect("a paired route");
        let direct = paired_lowest_davidson(
            &ctx,
            &v0s,
            &[],
            k,
            pre,
            false,
            PairedHook::Orthogonal,
            &popts,
        );
        let (so, vecs) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
        assert!(matches!(so.report, RpaReport::Paired { .. }));
        assert_eq!(so.omega2(), direct.omega2(), "entry ≠ kernel");
        assert_eq!(so.applies, direct.info.apply_pairs);
        let radial = radial_mode(&ctx);
        let scale = so.omega2().iter().fold(0.0f64, |m, w| m.max(w.abs()));
        for metric in [RpaMetric::Ii, RpaMetric::Rr] {
            for (j, (m, &w2)) in vecs.iter().zip(&so.omega2()).enumerate() {
                let q = &m.metric_side(metric);
                let rr = metric == RpaMetric::Rr;
                let mut w = if rr {
                    ctx.hpp_apply(q)
                } else {
                    ctx.hmm_apply(q)
                };
                deflate_radial(radial.as_ref(), &mut w);
                let mut r = if rr {
                    ctx.hmm_apply(&w)
                } else {
                    ctx.hpp_apply(&w)
                };
                deflate_radial(radial.as_ref(), &mut r);
                r.axpy(-w2, q);
                let rel = r.norm() / (scale * q.norm());
                assert!(
                    rel <= 1e-6,
                    "{metric:?}: mode {j} ω² {w2}: product residual {rel:.2e}"
                );
            }
        }
    }

    #[test]
    fn paired_su2_border_point_exhaustion_is_the_dense_pencil() {
        use crate::su2::border_fixture::{BOND, base_state, with_small_schmidt};
        let x = with_small_schmidt(&base_state(), BOND, 1, 1e-6);
        let h = generic_su2_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let (a, b, n) = grams(&ctx);
        let k = 3;
        let v0s = seeds(&ctx, k + 2, 0xDEF5);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 4 * n,
            max_subspace: n,
            ..PairedDavidsonOpts::default()
        };
        for hook in [PairedHook::Orthogonal, PairedHook::Oblique] {
            let out =
                paired_lowest_davidson(&ctx, &v0s, &[], k, PairedPrecond::Off, false, hook, &opts);
            assert_paired_wanted(
                &format!("su2 border 1e-6 {hook:?}"),
                &out,
                &a,
                &b,
                n,
                k,
                1e-8,
            );
        }
    }

    #[test]
    fn paired_su2_orthogonal_hook_is_one_symmetric_operator() {
        use crate::su2::border_fixture::base_state;
        let x = base_state();
        let h = generic_su2_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let radial = radial_mode(&ctx);
        let radial = radial.as_ref();
        let mu = paired_mu(PairedHook::Orthogonal, &ctx.a_diagonal());
        let qs = seeds(&ctx, 2, 0x0B11);
        let off = |q: &Su2TtTangent<f64>, f: f64| {
            let mut w = q.clone();
            w.for_each_coeff_mut(|c| *c = (*c * f).sin());
            w
        };
        let (u, w) = (off(&qs[0], 7.3), off(&qs[1], 3.1));
        let asym = |hook: PairedHook, a_side: bool| {
            let m = paired_mu(hook, &ctx.a_diagonal());
            let iu = su2_image(&ctx, hook, radial, m, &u, a_side);
            let iw = su2_image(&ctx, hook, radial, m, &w, a_side);
            let (p, q) = (u.dot(&iw), iu.dot(&w));
            (p - q).abs() / (iu.norm() * w.norm()).max(f64::MIN_POSITIVE)
        };
        for a_side in [true, false] {
            let (so, sb) = (
                asym(PairedHook::Orthogonal, a_side),
                asym(PairedHook::Oblique, a_side),
            );
            println!(
                "su2 {} asymmetry: orthogonal {so:.2e}, oblique {sb:.2e}",
                if a_side { "A" } else { "B" }
            );
            assert!(
                so <= 1e-12,
                "orthogonal-hook operator not symmetric: {so:e}"
            );
            assert!(
                sb >= 1e-6,
                "oblique-hook operator came out symmetric ({sb:e}) — vacuous"
            );
        }
        let mut g = u.clone();
        clean_tangent(radial, &mut g);
        let mut c = u.clone();
        c.axpy(-1.0, &g);
        let mut ac = su2_image(&ctx, PairedHook::Orthogonal, radial, mu, &c, true);
        ac.axpy(-mu, &c);
        let bc = su2_image(&ctx, PairedHook::Orthogonal, radial, mu, &c, false);
        assert!(
            ac.norm() <= 1e-12 * mu * c.norm(),
            "A′c ≠ μc: {:e}",
            ac.norm()
        );
        assert!(
            bc.norm() <= 1e-12 * mu * c.norm(),
            "B′c ≠ 0: {:e}",
            bc.norm()
        );
        let mut q = qs[1].clone();
        deflate_radial(radial, &mut q);
        assert!(c.norm() >= 1e-2 * u.norm(), "u was already clean");
        assert!(
            (q.dot(&c) / (q.norm() * c.norm())).abs() <= 1e-14,
            "the removed part overlaps a tangent"
        );
    }

    #[test]
    fn paired_su2_entry_is_the_kernel() {
        let (h, x) = point();
        let ctx = weingarten_operator(&h, &x);
        let k = 3;
        let v0s = seeds(&ctx, k + 2, 0x5DA3);
        let opts = PairedDavidsonOpts {
            tol: 1e-10,
            max_basis: 200,
            max_subspace: 24,
            positive_window: true,
            anomalies: AnomalyRule::Freeze,
            ..PairedDavidsonOpts::default()
        };
        let pre = PairedPrecond::Unshifted { kappa: 1e3 };
        let hook = PairedHook::Orthogonal;
        let out = paired_lowest_davidson(&ctx, &v0s, &[], k, pre, false, hook, &opts);
        let (bl, _) = ctx.shared_bases();
        let radial = radial_mode(&ctx);
        let radial = radial.as_ref();
        let d_a = ctx.a_diagonal();
        let mu = paired_mu(hook, &d_a);
        let space_dim = crate::su2::weingarten::tangent_dim(bl) - radial.map_or(0, |_| 1);
        let by_hand = omega_davidson_paired(
            |v| su2_image(&ctx, hook, radial, mu, v, true),
            |v| su2_image(&ctx, hook, radial, mu, v, false),
            |v: &mut Su2TtTangent<f64>| clean_tangent(radial, v),
            paired_precond(pre, d_a),
            false,
            &v0s,
            &[],
            k,
            space_dim,
            &opts,
        );
        let bits = |o: &PairedDavidsonOut<Su2TtTangent<f64>>| -> Vec<u64> {
            o.real
                .iter()
                .flat_map(|m| [m.omega.re.to_bits(), m.residual.to_bits()])
                .collect()
        };
        println!(
            "su2 entry: converged {} built {} ω {:?}",
            out.info.converged,
            out.info.built,
            out.real.iter().map(|m| m.omega.re).collect::<Vec<_>>()
        );
        assert!(out.info.converged, "fixture: the solve converges");
        assert_eq!(out.info.built, by_hand.info.built, "same trajectory");
        assert_eq!(
            bits(&out),
            bits(&by_hand),
            "entry is not bitwise the kernel"
        );
    }
}
