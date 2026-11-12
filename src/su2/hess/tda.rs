// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::solvers::hess::{
    TdaDavidsonOpts, TdaHooks, TdaModes, complement_shift, diag_inf_scale, tda_lowest_core,
};
use crate::su2::tangent::Su2TtTangent;

use super::{Su2WeingartenCtx, clean_tangent, deflate_radial, radial_mode};

/// The `k` lowest TDA pairs at `ctx`'s base point on the SU(2) layer — see
/// [`crate::solvers::hess::tda_lowest_davidson`] for the contract, the options and the empty and
/// short returns, all shared. Seeds are reduced-multiplet tangents at `ctx`'s frames (typically
/// `k + 2` random ones built by `Su2TtTangent::project`).
///
/// # Panics
///
/// On an empty `v0s` (for `k ≥ 1` on a non-empty space), on a spin-flux seed (`two_s_flux != 0` —
/// the cross-spin TDA is `Su2SectorCtx::sector_lowest_tda`), and on a negative or non-finite `tol`.
pub fn tda_lowest_davidson(
    ctx: &Su2WeingartenCtx<'_, f64>,
    v0s: &[Su2TtTangent<f64>],
    k: usize,
    opts: &TdaDavidsonOpts,
) -> TdaModes<Su2TtTangent<f64>> {
    for (i, v) in v0s.iter().enumerate() {
        v.refuse_if_shifted(&format!("su2 tda_lowest_davidson (seed {i})"));
    }
    let (base_l, _) = ctx.bases();
    let radial = radial_mode(ctx);
    let space_dim = crate::su2::weingarten::tangent_dim(base_l) - radial.as_ref().map_or(0, |_| 1);
    ctx.warm_a_caches();
    let radial = radial.as_ref();
    let d_a = ctx.a_diagonal();
    let mu = complement_shift(diag_inf_scale(&d_a));
    tda_lowest_core(
        TdaHooks {
            a: |v: &Su2TtTangent<f64>| ctx.a_apply(v),
            reproject: |v: &mut Su2TtTangent<f64>| clean_tangent(radial, v),
            deflate: |v: &mut Su2TtTangent<f64>| deflate_radial(radial, v),
            diag: move || d_a,
            complement_shift: mu,
        },
        v0s,
        k,
        space_dim,
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::factor::dense_self_adjoint_smallest;
    use crate::solvers::hess::TdaPrecond;
    use crate::su2::hamiltonian::Su2SparseHamiltonian;
    use crate::su2::hess::{hessian_grams, weingarten_operator};
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::su2::weingarten::{tangent_basis, tangent_dim};
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

    #[test]
    fn su2_tda_s1_orthogonal_hook_and_the_dense_centerweighted_a_gram() {
        let (h, x) = point();
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.bases();

        let qs = seeds(&ctx, 4, 0x5DA2);
        let mut w = qs[0].clone();
        w.for_each_coeff_mut(|c| *c = (*c * 7.3).sin());
        let mut g = w.clone();
        g.gauge_fix_orthogonal_mut();
        let mut e_g = w.clone();
        e_g.axpy(-1.0, &g);
        let p = Su2TtTangent::project(bl, br, &w.to_vector());
        let mut e_p = w.clone();
        e_p.axpy(-1.0, &p);
        assert!(
            e_g.norm() > 1e-2 && e_p.norm() > 1e-2,
            "w is off the subspace"
        );
        let (mut worst_g, mut worst_p) = (0.0f64, 0.0f64);
        for q in &qs[1..] {
            worst_g = worst_g.max((q.dot(&e_g) / (q.norm() * e_g.norm())).abs());
            worst_p = worst_p.max((q.dot(&e_p) / (q.norm() * e_p.norm())).abs());
        }
        println!("S1: off-subspace overlap: orthogonal {worst_g:.2e}, project {worst_p:.2e}");
        assert!(
            worst_g <= 1e-14,
            "centerweighted-orthogonal gauge fix ({worst_g:e})"
        );
        assert!(
            worst_p >= 1e-4,
            "project∘to_vector is oblique ({worst_p:e})"
        );

        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let space_dim = tangent_dim(bl) - 1;
        println!("S1: reachable n = {n}, structural tangent_dim − 1 = {space_dim}");
        assert!(
            n < space_dim,
            "fixture precondition: a rank-padded point ({n} vs {space_dim})"
        );
        let (a, _b) = hessian_grams(&ctx, &basis);
        let dense = dense_self_adjoint_smallest(&a, n, n).0;
        println!("S1: dense eig(A) = {dense:?}");
        assert!(
            dense[0] < -1e-3,
            "fixture precondition: A is indefinite at this non-critical point ({})",
            dense[0]
        );
        let k = 3;
        let tol = 1e-11;
        let n_neg = dense[..k].iter().filter(|&&t| t < 0.0).count();
        let radial = radial_mode(&ctx).expect("non-degenerate point");
        for precond in [
            TdaPrecond::Off,
            TdaPrecond::Diag { kappa: 10.0 },
            TdaPrecond::Shifted,
        ] {
            let opts = TdaDavidsonOpts {
                tol,
                max_iter: 2000,
                max_subspace: 2 * k,
                restart_keep: k,
                precond,
                ..TdaDavidsonOpts::default()
            };
            let out = tda_lowest_davidson(&ctx, &seeds(&ctx, k + 2, 0x5DA3), k, &opts);
            println!(
                "S1 {precond:?}: converged {} it {} applies {} θ {:?} res {:?}",
                out.info.converged,
                out.info.iterations,
                out.info.applies,
                out.theta,
                out.info.residuals
            );
            assert!(out.info.converged, "{precond:?}: not converged");
            assert_eq!(out.theta.len(), k);
            let kernel_applies = out.info.applies - k - out.info.negative.len();
            assert!(
                kernel_applies > 2 * k,
                "{precond:?}: the forced restarts never ran"
            );
            for (j, want) in dense.iter().enumerate().take(k) {
                let th = out.theta[j];
                assert!(
                    (th - want).abs() <= 1e-9,
                    "{precond:?} root {j}: θ {th} vs dense {want}"
                );
                assert!(
                    out.info.residuals[j] <= 10.0 * tol,
                    "{precond:?} root {j}: fresh residual {}",
                    out.info.residuals[j]
                );
            }
            assert_eq!(out.info.negative.len(), n_neg, "{precond:?}: witnesses");
            assert!(out.all_negatives_certified());
            for (i, u) in out.vectors.iter().enumerate() {
                assert!(
                    radial.dot(u).abs() <= 1e-10,
                    "{precond:?}: radial weight of ξ{i}"
                );
                let mut pu = Su2TtTangent::project(bl, br, &u.to_vector());
                pu.axpy(-1.0, u);
                assert!(pu.norm() <= 1e-10, "{precond:?}: tangency defect of ξ{i}");
                for (j, v) in out.vectors.iter().enumerate() {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (u.dot(v) - want).abs() <= 1e-9,
                        "{precond:?}: centerweighted ⟨ξ{i}, ξ{j}⟩"
                    );
                }
            }
        }

        let out = tda_lowest_davidson(
            &ctx,
            &seeds(&ctx, 6, 0x5DA3),
            space_dim,
            &TdaDavidsonOpts {
                tol: 1e-10,
                ..TdaDavidsonOpts::default()
            },
        );
        println!(
            "S1 whole space: {} roots of {space_dim} requested, converged {}",
            out.theta.len(),
            out.info.converged
        );
        assert_eq!(out.info.space_dim, space_dim);
        assert_eq!(out.theta.len(), n, "the reachable span, exactly");
        assert!(
            !out.info.converged,
            "a short return is never reported converged"
        );
        for (j, (th, v)) in out.theta.iter().zip(&out.vectors).enumerate() {
            assert!(
                (th - dense[j]).abs() <= 1e-9,
                "root {j}: θ {th} vs dense {}",
                dense[j]
            );
            let e = v.to_vector().norm();
            assert!(
                (e - 1.0).abs() <= 1e-12,
                "root {j}: embedding norm {e} — an ι-null root"
            );
        }
    }
}
