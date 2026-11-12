// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

pub(crate) fn weingarten_right_chain_oracle<T: Scalar>(
    base_l: &TtVector<T>,
    base_r: &TtVector<T>,
    sources: &[&dyn ZSource<T>],
    xi: &TtTangent<T>,
    gauges: &[BondGauge<T>],
) -> TtTangent<T> {
    let space = base_l.space;
    let d = space.d;
    let r = base_l.ranks.clone();
    let phi = xi.phi();
    let shifted = phi != Irrep::IDENTITY;
    let space_out = if shifted {
        space.with_target(space.target.fuse(phi))
    } else {
        space
    };
    let r_col: Option<Vec<RankGrid>> =
        shifted.then(|| r.iter().map(|g| g.relabel_irrep(phi)).collect());
    let rc = |b: usize| -> &RankGrid { r_col.as_ref().map_or(&r[b], |v| &v[b]) };
    let shifted_pair = shifted.then(|| shifted_frames(base_l, base_r, phi));
    let (fl, fr) = shifted_pair
        .as_ref()
        .map_or((base_l, base_r), |(u, v)| (u, v));
    let finish = |cores: Vec<SparseCore<T>>| {
        if shifted {
            TtTangent::from_parts_shifted(
                r.clone(),
                r_col.clone().expect("shifted"),
                phi,
                fl.clone(),
                fr.clone(),
                cores,
            )
        } else {
            TtTangent::from_parts(r.clone(), base_l.clone(), base_r.clone(), cores)
        }
    };
    let mut out_cores: Vec<SparseCore<T>> = (0..d)
        .map(|k| SparseCore::<T>::zeros(space_out, k, &r[k], rc(k + 1)))
        .collect();
    if d == 1 {
        return finish(out_cores);
    }
    if shifted {
        let q = Sketch::<T>::identity(space_out, d, 1);
        let empty_channel = q
            .blocks
            .iter()
            .all(|(_, b)| b.as_ref().is_none_or(|m| m.ncols() == 0));
        if empty_channel {
            return finish(out_cores);
        }
    }
    let comps_r = right_project_components(fl, fr, &xi.to_vector());
    let comps_r_scaled: Vec<SparseCore<T>> = (1..d)
        .map(|b| {
            let c = mul_sketch_left(&gauges[b - 1].r_inv, &comps_r[b]);
            if shifted {
                c.relabel_irrep_into(phi, space_out)
            } else {
                c
            }
        })
        .collect();
    for &z in sources {
        let folded;
        let (pz, qz): (&[Sketch<T>], &[Sketch<T>]) = match z.cached_envs() {
            Some(pq) => pq,
            None => {
                folded = (left_env_z(base_l, z), right_env_z(base_r, z));
                (&folded.0, &folded.1)
            }
        };
        let evr = var_right_env_oracle(fr, z, &comps_r_scaled, qz);
        for (k, middle) in out_cores.iter_mut().enumerate().take(d - 1) {
            let s3 = mul_sketch_left(&pz[k], &z.right_mul(k, &evr[k + 1]));
            core_add_assign(middle, T::one(), &s3);
        }
    }
    for (k, core) in out_cores.iter_mut().enumerate().take(d - 1) {
        gauge_complement_mut(core, base_l.core(k));
    }
    finish(out_cores)
}

fn var_right_env_oracle<T: Scalar>(
    base_r: &TtVector<T>,
    z: &dyn ZSource<T>,
    comps_r_scaled: &[SparseCore<T>],
    qz: &[Sketch<T>],
) -> Vec<Sketch<T>> {
    let space = base_r.space;
    let d = space.d;
    let mut evr: Vec<Option<Sketch<T>>> = (0..=d).map(|_| None).collect();
    evr[0] = Some(Sketch::zeros(space, 0, &z.bond_ranks(0), &base_r.ranks[0]));
    evr[d] = Some(Sketch::zeros(space, d, &z.bond_ranks(d), &base_r.ranks[d]));
    for b in (1..d).rev() {
        let t1a = z.right_mul(b, evr[b + 1].as_ref().unwrap());
        let mut cur = core_times_adjoint_core(&t1a, base_r.core(b));
        let t2a = z.right_mul(b, &qz[b + 1]);
        let t2 = core_times_adjoint_core(&t2a, &comps_r_scaled[b - 1]);
        cur.add_assign_sketch(&t2);
        evr[b] = Some(cur);
    }
    evr.into_iter().map(|x| x.unwrap()).collect()
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::core::border_fixture::{
        CLUSTER_FLOOR, assert_cluster_premise, cluster_state, pg_cluster_state, unit_tangents,
    };
    use crate::core::deflate::ColumnDeflation;
    use crate::core::tangent::tangent_dim_grids;
    use crate::solvers::riemannian::tangent_bases;
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

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
        bl: TtVector<f64>,
        br: TtVector<f64>,
        z: TtVector<f64>,
        g_plain: Vec<BondGauge<f64>>,
        f: f64,
        phi: Irrep,
        xis: Vec<TtTangent<f64>>,
    }

    fn normal_z(bl: &TtVector<f64>, br: &TtVector<f64>, seed: u64) -> TtVector<f64> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let w: TtVector<f64> = randomized_state(bl.space, 1, 2, &mut rng);
        let z = &w - &TtTangent::project(bl, br, &w).to_vector();
        let zn = z.norm_lr(false);
        let tang = TtTangent::project(bl, br, &z).norm();
        assert!(
            zn > 1e-6 && tang <= 1e-6 * zn,
            "normal_z: ‖z‖ {zn:e}, ‖P(z)‖ {tang:e}"
        );
        z
    }

    fn fx_flux0() -> Kfx {
        let (x, plant) = cluster_state();
        let f = CLUSTER_FLOOR;
        assert_cluster_premise("flux 0 cluster", &x, &plant, f, true);
        let (bl, br) = tangent_bases(&x);
        let z = normal_z(&bl, &br, 0x9D40_0001);
        let g_plain = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        let xis = unit_tangents(&bl, &br, Irrep::IDENTITY, 4, 0x9D40_0002);
        Kfx {
            label: "flux 0",
            bl,
            br,
            z,
            g_plain,
            f,
            phi: Irrep::IDENTITY,
            xis,
        }
    }

    fn fx_shifted() -> Kfx {
        let (x, plant) = pg_cluster_state();
        let f = CLUSTER_FLOOR;
        assert_cluster_premise("typed cluster", &x, &plant, f, false);
        let (bl, br) = tangent_bases(&x);
        let g_plain = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        let space = x.space;
        let p = ColumnDeflation::from_gauges(&g_plain, f);
        let phi = (1..space.pg.order())
            .map(Irrep::new)
            .filter(|&p| tangent_dim_grids(&space, &bl.ranks, p) > 0)
            .find(|&ph| {
                let t = unit_tangents(&bl, &br, ph, 1, 0x9D40_0003);
                p.removed_dim(&t[0]) >= 2
            })
            .expect("fixture premise: a live shifted channel the cut reaches");
        let t = unit_tangents(&bl, &br, phi, 1, 0x9D40_0003);
        println!(
            "typed cluster: phi {phi:?}, dim T^phi {}, removed by Hard(f) {}",
            tangent_dim_grids(&space, &bl.ranks, phi),
            p.removed_dim(&t[0])
        );
        let z = normal_z(&bl, &br, 0x9D40_0004);
        let xis = unit_tangents(&bl, &br, phi, 4, 0x9D40_0005);
        Kfx {
            label: "shifted",
            bl,
            br,
            z,
            g_plain,
            f,
            phi,
            xis,
        }
    }

    fn asm(k: &Kfx, xi: &TtTangent<f64>, g: &[BondGauge<f64>], terms: Terms) -> TtTangent<f64> {
        weingarten_assemble_cached(&k.bl, &k.br, &k.z, xi, g, terms)
    }

    fn old_s3(k: &Kfx, xi: &TtTangent<f64>, g: &[BondGauge<f64>]) -> TtTangent<f64> {
        weingarten_right_chain_oracle(&k.bl, &k.br, &[&ExplicitZ::new(&k.z)], xi, g)
    }

    fn applied(p: &ColumnDeflation<f64>, xi: &TtTangent<f64>) -> TtTangent<f64> {
        let mut v = xi.clone();
        p.apply(&mut v);
        v
    }

    fn rel(a: &TtTangent<f64>, b: &TtTangent<f64>) -> f64 {
        a.sub(b).norm() / b.norm().max(f64::MIN_POSITIVE)
    }

    #[test]
    fn rco_t6_the_right_chain_is_one_form_at_the_plain_inverse() {
        for k in [fx_flux0(), fx_shifted()] {
            let (mut worst_s3, mut worst_b, mut share) = (0.0f64, 0.0f64, f64::INFINITY);
            for xi in &k.xis {
                let s3_new = asm(&k, xi, &k.g_plain, S3_ONLY);
                let s3_old = old_s3(&k, xi, &k.g_plain);
                let b_new = asm(&k, xi, &k.g_plain, Terms::all());
                let b_old = asm(&k, xi, &k.g_plain, S1_S2).add(&s3_old);
                worst_s3 = worst_s3.max(rel(&s3_new, &s3_old));
                worst_b = worst_b.max(rel(&b_new, &b_old));
                share = share.min(s3_old.norm() / b_old.norm());
            }
            println!(
                "T6 [{}]: |S3_new − S3_old|/|S3_old| {worst_s3:.2e}, |B_new − B_old|/|B_old| \
                 {worst_b:.2e}; min ‖S3‖/‖B‖ {share:.3}",
                k.label
            );
            assert!(
                share >= 1e-2,
                "[{}] non-vacuity: the right-frame summand carries no share of B",
                k.label
            );
            assert!(
                worst_s3 <= 1e-13,
                "[{}] T6: the right-frame summand is not the pre-P4d right chain ({worst_s3:e})",
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
    fn rco_t7_the_regularized_right_chain_cuts_the_output_columns() {
        let mut rows = Vec::new();
        for k in [fx_flux0(), fx_shifted()] {
            let f = k.f;
            let g_hard = bond_gauges(&k.bl, &k.br, FloorPolicy::Hard(f));
            let g_tik = bond_gauges(&k.bl, &k.br, FloorPolicy::Tikhonov(f));
            let p = ColumnDeflation::from_gauges(&k.g_plain, f);
            let dt = ColumnDeflation::from_gauges_weighted(&k.g_plain, |s| s * s / (s * s + f * f));
            let (mut s3_hard, mut s3_tik, mut s1_hard, mut s1_tik, mut moved) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for xi in &k.xis {
                assert_eq!(xi.phi(), k.phi);
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
                "T7 [{}]: |S3'(floored) − P·S3|/|S3| {s3_hard:.2e}, |S3'(damped) − D·S3|/|S3| \
                 {s3_tik:.2e}; control |S1(floored) − S1(P·)|/|S1| {s1_hard:.2e}, \
                 |S1(damped) − S1(D·)|/|S1| {s1_tik:.2e}; P moves S3 by {moved:.3e}",
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
                "[{label}] control T2: the floored/damped left-frame summand is not S1 of the \
                 projected input ({s1_hard:e}, {s1_tik:e})"
            );
            assert!(
                s3_hard <= 1e-13,
                "[{label}] T7 Hard: the floored right-frame summand is not P·S3 ({s3_hard:e}) — the \
                 cut sits on the input's right-gauge rows, not on the output columns"
            );
            assert!(
                s3_tik <= 1e-13,
                "[{label}] T7 Tikhonov: the damped right-frame summand is not D·S3 ({s3_tik:e})"
            );
        }
    }

    #[test]
    fn rco_t1_the_two_frame_terms_are_adjoint_at_the_plain_inverse() {
        for k in [fx_flux0(), fx_shifted()] {
            let s1: Vec<TtTangent<f64>> = k
                .xis
                .iter()
                .map(|xi| asm(&k, xi, &k.g_plain, S1_ONLY))
                .collect();
            let s3: Vec<TtTangent<f64>> =
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
                "T1 [{}]: max |<eta, S3 xi> − <S1 eta, xi>| {worst:.2e} (scale {scale:.3e})",
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
