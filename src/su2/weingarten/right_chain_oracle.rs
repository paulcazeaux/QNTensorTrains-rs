// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;
use crate::su2::flux::regrid_train;

pub(crate) fn su2_weingarten_right_chain_oracle<T: Scalar>(
    base_l: &Su2TtVector<T>,
    base_r: &Su2TtVector<T>,
    sources: &[&dyn Su2ZSource<T>],
    xi: &Su2TtTangent<T>,
    gauges: &[Su2BondGauge<T>],
    sf: Option<&Su2ShiftedFrames<T>>,
) -> Su2TtTangent<T> {
    let space = base_l.space;
    let d = space.d;
    let r = base_l.ranks.clone();
    let two_s_flux = xi.two_s_flux();
    let two_kappa = two_s_flux as i64;
    let shifted = two_s_flux != 0;
    assert!(
        (shifted && sf.is_some_and(|s| s.two_s_flux == two_s_flux)) || (!shifted && sf.is_none()),
        "su2_weingarten_right_chain_oracle: the bundle must be xi's channel's"
    );
    let out_space = sf.map_or(space, |s| s.out_space);
    let rows: &[Su2RankGrid] = sf.map_or(r.as_slice(), |s| s.rm.as_slice());
    let cols: &[Su2RankGrid] = sf.map_or(r.as_slice(), |s| s.rd.as_slice());
    let (fl, fr) = sf.map_or((base_l, base_r), |s| (&s.frames.0, &s.frames.1));
    let vz: Option<Su2TtVector<T>> = sf.map(|s| regrid_train(&s.frames.1, space, &s.rdz));
    let zr = vz.as_ref().unwrap_or(base_r);
    let finish = |cores: Vec<Su2Core<T>>| match sf {
        Some(s) => Su2TtTangent::from_parts_shifted(
            s.rm.clone(),
            s.rd.clone(),
            two_s_flux,
            s.frames.0.clone(),
            s.frames.1.clone(),
            cores,
        ),
        None => Su2TtTangent::from_parts(r.clone(), base_l.clone(), base_r.clone(), cores),
    };
    let mut out_cores: Vec<Su2Core<T>> = (0..d)
        .map(|k| Su2Core::<T>::zeros(out_space, k, &rows[k], &cols[k + 1]))
        .collect();
    if d == 1 {
        return finish(out_cores);
    }
    if let Some(s) = sf {
        let empty =
            s.rd[d].total() == 0 || (1..d).any(|b| s.rm[b].total() == 0 && s.rd[b].total() == 0);
        if empty {
            return finish(out_cores);
        }
    }
    let mut spin = SpinAlgebra::new();
    let comps_r = right_project_components(fl, fr, &xi.to_vector());
    let comps_r_scaled: Vec<Su2Core<T>> = (1..d)
        .map(|b| match sf {
            Some(s) => {
                let masked = lift_bond_sketch(&gauges[b - 1].r_inv, &r[b], &s.rm[b], out_space, 0);
                mul_sketch_left(&masked, &comps_r[b])
            }
            None => mul_sketch_left(&gauges[b - 1].r_inv, &comps_r[b]),
        })
        .collect();
    let comps_r_scaled_z: Vec<Su2Core<T>> = match sf {
        Some(s) => (1..d)
            .map(|b| {
                cross_wire(
                    &comps_r_scaled[b - 1],
                    CrossDir::AtoB,
                    two_kappa,
                    (&s.rm[b], &s.rd[b + 1]),
                    (&s.rdz[b], &r[b + 1]),
                    out_space,
                    space,
                    (&r[b], &r[b + 1]),
                    &mut spin,
                )
            })
            .collect(),
        None => Vec::new(),
    };
    let evr_comps = if shifted {
        &comps_r_scaled_z
    } else {
        &comps_r_scaled
    };
    for &z in sources {
        let folded;
        let (pz, qz): (&[Su2Sketch<T>], &[Su2Sketch<T>]) = match z.cached_envs() {
            Some(pq) => pq,
            None => {
                folded = (left_env_z(base_l, z), right_env_z(base_r, z));
                (&folded.0, &folded.1)
            }
        };
        let evr = var_right_env_oracle(zr, z, evr_comps, qz);
        for (k, middle) in out_cores.iter_mut().enumerate().take(d - 1) {
            let s3 = mul_sketch_left(&pz[k], &z.right_mul(k, &evr[k + 1]));
            let s3 = match sf {
                Some(s) => regrid_core(&s3, out_space, &s.rm[k], &s.rd[k + 1]),
                None => s3,
            };
            middle.axpy(T::one(), &s3);
        }
    }
    for (k, core) in out_cores.iter_mut().enumerate().take(d - 1) {
        gauge_complement_mut(core, fl.core(k));
    }
    finish(out_cores)
}

fn var_right_env_oracle<T: Scalar>(
    base_r: &Su2TtVector<T>,
    z: &dyn Su2ZSource<T>,
    comps_r_scaled: &[Su2Core<T>],
    qz: &[Su2Sketch<T>],
) -> Vec<Su2Sketch<T>> {
    let space = base_r.space;
    let d = space.d;
    let mut evr: Vec<Option<Su2Sketch<T>>> = (0..=d).map(|_| None).collect();
    evr[0] = Some(Su2Sketch::zeros(
        space,
        0,
        &z.bond_ranks(0),
        &base_r.ranks[0],
    ));
    evr[d] = Some(Su2Sketch::zeros(
        space,
        d,
        &z.bond_ranks(d),
        &base_r.ranks[d],
    ));
    for b in (1..d).rev() {
        let t1a = z.right_mul(b, evr[b + 1].as_ref().unwrap());
        let mut cur = rproject(&t1a, base_r.core(b));
        let t2a = z.right_mul(b, &qz[b + 1]);
        let t2 = rproject(&t2a, &comps_r_scaled[b - 1]);
        cur.add_assign_sketch(&t2);
        evr[b] = Some(cur);
    }
    evr.into_iter().map(|x| x.unwrap()).collect()
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::border_fixture::{CLUSTER, CLUSTER_FLOOR};
    use crate::su2::border_fixture::{assert_cluster_premise, cluster_state, unit_tangents};
    use crate::su2::deflate::Su2ColumnDeflation;
    use crate::su2::qn::max_ranks;
    use crate::su2::riemannian::tangent_bases;
    use crate::su2::vector::tt_randn;
    use crate::test_util::seeded;

    const S1_ONLY: Terms = Terms {
        diag_s1: true,
        diag_s2: false,
        diag_s3: false,
    };
    const S1_S2: Terms = Terms {
        diag_s1: true,
        diag_s2: true,
        diag_s3: false,
    };
    const S3_ONLY: Terms = Terms {
        diag_s1: false,
        diag_s2: false,
        diag_s3: true,
    };

    struct Kfx {
        label: &'static str,
        bl: std::sync::Arc<Su2TtVector<f64>>,
        br: std::sync::Arc<Su2TtVector<f64>>,
        z: Su2TtVector<f64>,
        g_plain: Vec<Su2BondGauge<f64>>,
        f: f64,
        two_k: usize,
        sf: Option<Su2ShiftedFrames<f64>>,
        xis: Vec<Su2TtTangent<f64>>,
    }

    fn normal_z(bl: &Su2TtVector<f64>, br: &Su2TtVector<f64>, seed: u64) -> Su2TtVector<f64> {
        let space = bl.space;
        let mut rng = seeded(seed);
        let w: Su2TtVector<f64> = tt_randn(space, &max_ranks(&space), &mut rng);
        let z = &w - &Su2TtTangent::project(bl, br, &w).to_vector();
        let zn = z.norm_lr(false);
        let tang = Su2TtTangent::project(bl, br, &z).norm();
        assert!(
            zn > 1e-6 && tang <= 1e-6 * zn,
            "normal_z: ‖z‖ {zn:e}, ‖P(z)‖ {tang:e}"
        );
        z
    }

    fn fx(two_k: usize) -> Kfx {
        let values: Vec<f64> = CLUSTER.iter().map(|m| m * CLUSTER_FLOOR).collect();
        let (x, plant) = cluster_state(&values);
        let f = CLUSTER_FLOOR;
        let label = if two_k == 0 { "flux 0" } else { "2κ = 2" };
        assert_cluster_premise(&format!("SU(2) cluster [{label}]"), &x, &plant, f, true);
        let (bl, br) = tangent_bases(&x);
        let g_plain = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        let z = normal_z(&bl, &br, 0x9D42_0001);
        let xis = unit_tangents(&bl, &br, two_k, 4, 0x9D42_0002 + two_k as u64);
        let sf = (two_k != 0).then(|| Su2ShiftedFrames::new(&bl, &br, two_k));
        if two_k != 0 {
            let removed = Su2ColumnDeflation::from_gauges(&g_plain, f).removed_dim(&xis[0]);
            println!("SU(2) cluster [{label}]: the lifted Hard(f) operator removes {removed}");
            assert!(
                removed >= 2,
                "fixture premise: the cut reaches the spin channel"
            );
        }
        Kfx {
            label,
            bl: std::sync::Arc::new(bl),
            br: std::sync::Arc::new(br),
            z,
            g_plain,
            f,
            two_k,
            sf,
            xis,
        }
    }

    fn asm(
        k: &Kfx,
        xi: &Su2TtTangent<f64>,
        g: &[Su2BondGauge<f64>],
        terms: Terms,
    ) -> Su2TtTangent<f64> {
        weingarten_assemble_sources_dressed(
            &k.bl,
            &k.br,
            &[&Su2ExplicitZ::new(&k.z)],
            xi,
            g,
            terms,
            k.sf.as_ref(),
        )
    }

    fn old_s3(k: &Kfx, xi: &Su2TtTangent<f64>, g: &[Su2BondGauge<f64>]) -> Su2TtTangent<f64> {
        su2_weingarten_right_chain_oracle(
            &k.bl,
            &k.br,
            &[&Su2ExplicitZ::new(&k.z)],
            xi,
            g,
            k.sf.as_ref(),
        )
    }

    fn applied(p: &Su2ColumnDeflation<f64>, xi: &Su2TtTangent<f64>) -> Su2TtTangent<f64> {
        let mut v = xi.clone();
        p.apply(&mut v);
        v
    }

    #[test]
    fn rco_su2_t6_the_right_chain_is_one_form_at_the_plain_inverse() {
        for k in [fx(0), fx(2)] {
            let (mut worst_s3, mut worst_b, mut share) = (0.0f64, 0.0f64, f64::INFINITY);
            for xi in &k.xis {
                assert_eq!(xi.two_s_flux(), k.two_k);
                let s3_new = asm(&k, xi, &k.g_plain, S3_ONLY);
                let s3_old = old_s3(&k, xi, &k.g_plain);
                let b_new = asm(&k, xi, &k.g_plain, Terms::all());
                let b_old = asm(&k, xi, &k.g_plain, S1_S2).add(&s3_old);
                worst_s3 = worst_s3.max(s3_new.sub(&s3_old).norm() / s3_old.norm());
                worst_b = worst_b.max(b_new.sub(&b_old).norm() / b_old.norm());
                share = share.min(s3_old.norm() / b_old.norm());
            }
            println!(
                "SU(2) T6 [{}]: |S3_new − S3_old|/|S3_old| {worst_s3:.2e}, |B_new − B_old|/|B_old| \
                 {worst_b:.2e}; min ‖S3‖/‖B‖ {share:.3}",
                k.label
            );
            assert!(share >= 1e-2, "[{}] non-vacuity", k.label);
            assert!(
                worst_s3 <= 1e-13,
                "[{}] T6: the right-frame summand is not the pre-P4d chain ({worst_s3:e})",
                k.label
            );
            assert!(
                worst_b <= 1e-13,
                "[{}] T6: B is not the pre-P4d assembly ({worst_b:e})",
                k.label
            );
        }
    }

    #[test]
    fn rco_su2_t7_the_regularized_right_chain_cuts_the_output_columns() {
        let mut rows = Vec::new();
        for k in [fx(0), fx(2)] {
            let f = k.f;
            let g_hard = bond_gauges(&k.bl, &k.br, FloorPolicy::Hard(f));
            let g_tik = bond_gauges(&k.bl, &k.br, FloorPolicy::Tikhonov(f));
            let p = Su2ColumnDeflation::from_gauges(&k.g_plain, f);
            let dt =
                Su2ColumnDeflation::from_gauges_weighted(&k.g_plain, |s| s * s / (s * s + f * f));
            let (mut s3_hard, mut s3_tik, mut s1_hard, mut s1_tik, mut moved) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for xi in &k.xis {
                let s3 = old_s3(&k, xi, &k.g_plain);
                let n3 = s3.norm();
                let want_h = applied(&p, &s3);
                let want_t = applied(&dt, &s3);
                moved = moved.max(s3.sub(&want_h).norm() / n3);
                s3_hard = s3_hard.max(asm(&k, xi, &g_hard, S3_ONLY).sub(&want_h).norm() / n3);
                s3_tik = s3_tik.max(asm(&k, xi, &g_tik, S3_ONLY).sub(&want_t).norm() / n3);
                let s1 = asm(&k, xi, &k.g_plain, S1_ONLY).norm();
                s1_hard = s1_hard.max(
                    asm(&k, xi, &g_hard, S1_ONLY)
                        .sub(&asm(&k, &applied(&p, xi), &k.g_plain, S1_ONLY))
                        .norm()
                        / s1,
                );
                s1_tik = s1_tik.max(
                    asm(&k, xi, &g_tik, S1_ONLY)
                        .sub(&asm(&k, &applied(&dt, xi), &k.g_plain, S1_ONLY))
                        .norm()
                        / s1,
                );
            }
            println!(
                "SU(2) T7 [{}]: |S3'(floored) − Pc·S3|/|S3| {s3_hard:.2e}, |S3'(damped) − Dc·S3|/|S3| \
                 {s3_tik:.2e}; control |S1(floored) − S1(Pc·)|/|S1| {s1_hard:.2e}, |S1(damped) − \
                 S1(Dc·)|/|S1| {s1_tik:.2e}; Pc moves S3 by {moved:.3e}",
                k.label
            );
            rows.push((k.label, s3_hard, s3_tik, s1_hard, s1_tik, moved));
        }
        for (label, s3_hard, s3_tik, s1_hard, s1_tik, moved) in rows {
            assert!(
                moved >= 1e-6,
                "[{label}] non-vacuity: the cut does not reach the right-frame summand's output"
            );
            assert!(
                s1_hard <= 1e-13 && s1_tik <= 1e-13,
                "[{label}] control: the floored/damped left-frame summand is not S1 of the projected \
                 input ({s1_hard:e}, {s1_tik:e})"
            );
            assert!(
                s3_hard <= 1e-13,
                "[{label}] T7 Hard: the floored right-frame summand is not Pc·S3 ({s3_hard:e})"
            );
            assert!(
                s3_tik <= 1e-13,
                "[{label}] T7 Tikhonov: the damped right-frame summand is not Dc·S3 ({s3_tik:e})"
            );
        }
    }

    #[test]
    fn rco_su2_t1_the_two_frame_terms_are_adjoint_at_the_plain_inverse() {
        for k in [fx(0), fx(2)] {
            let s1: Vec<Su2TtTangent<f64>> = k
                .xis
                .iter()
                .map(|xi| asm(&k, xi, &k.g_plain, S1_ONLY))
                .collect();
            let s3: Vec<Su2TtTangent<f64>> =
                k.xis.iter().map(|xi| old_s3(&k, xi, &k.g_plain)).collect();
            let (mut worst, mut scale) = (0.0f64, 0.0f64);
            for (i, eta) in k.xis.iter().enumerate() {
                for (j, xi) in k.xis.iter().enumerate() {
                    let lhs = eta.dot(&s3[j]);
                    let rhs = s1[i].dot(xi);
                    worst = worst.max((lhs - rhs).abs());
                    scale = scale.max(lhs.abs()).max(rhs.abs());
                }
            }
            println!(
                "SU(2) T1 [{}]: max |<eta, S3 xi> − <S1 eta, xi>| {worst:.2e} (scale {scale:.3e})",
                k.label
            );
            assert!(scale > 1e-8, "[{}] non-vacuity", k.label);
            assert!(
                worst <= 1e-12 * scale,
                "[{}] T1: S3 is not S1ᴴ ({worst:e} at scale {scale:e})",
                k.label
            );
        }
    }
}
