// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::C64;
use qntt::core::factor::Sketch;
use qntt::solvers::InnerProductVector;
use qntt::solvers::overlap_env::{
    OverlapEnvs, local_overlap_vector, overlap_envs, overlap_step_left, overlap_step_right,
};
use qntt_testkit::randomized_state;

fn full_envs_abelian<T: Scalar>(x: &TtVector<T>, psi: &TtVector<T>) -> OverlapEnvs<T> {
    let d = x.space.d;
    let mut l: Vec<Option<Sketch<T>>> = vec![None; d + 1];
    let mut r: Vec<Option<Sketch<T>>> = vec![None; d + 1];
    l[0] = Some(Sketch::identity(x.space, 0, 1));
    for k in 0..d {
        let prev = l[k]
            .clone()
            .expect("full_envs_abelian: l[k] built in order");
        l[k + 1] = Some(overlap_step_right(x, psi, &prev, k));
    }
    r[d] = Some(Sketch::identity(x.space, d, 1));
    for k in (1..d).rev() {
        let nxt = r[k + 1]
            .clone()
            .expect("full_envs_abelian: r[k+1] built in order");
        r[k] = Some(overlap_step_left(x, psi, &nxt, k));
    }
    OverlapEnvs { l, r }
}

fn sketch_bits_eq<T: Scalar>(a: &Sketch<T>, b: &Sketch<T>, what: &str) {
    assert_eq!(a.k, b.k, "{what}: bond index");
    for s in a.sectors() {
        match (a.block(s), b.block(s)) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(
                    (x.nrows(), x.ncols()),
                    (y.nrows(), y.ncols()),
                    "{what}: block {s:?} dims"
                );
                for j in 0..x.ncols() {
                    for i in 0..x.nrows() {
                        assert!(
                            bits_eq(*x.get(i, j), *y.get(i, j)),
                            "{what}: block {s:?} entry ({i},{j}) differs (bit-identity required — \
                             the fix is in the fold's operation order, never in this test)"
                        );
                    }
                }
            }
            _ => panic!("{what}: block {s:?} is present on exactly one side"),
        }
    }
}

fn assert_env_shapes<T: Scalar>(
    env: &OverlapEnvs<T>,
    x: &TtVector<T>,
    psi: &TtVector<T>,
    tag: &str,
) {
    let space = x.space;
    let d = space.d;
    for b in 0..=d {
        if let Some(lb) = env.l[b].as_ref() {
            assert_eq!(lb.k, b, "{tag}: l[{b}] lives on bond {}", lb.k);
            let (rr, cc) = (lb.row_ranks(), lb.col_ranks());
            for s in space.bond_sectors(b) {
                assert_eq!(rr[s], x.rank(b, s), "{tag}: l[{b}] rows at {s:?} (rank_x)");
                assert_eq!(
                    cc[s],
                    psi.rank(b, s),
                    "{tag}: l[{b}] cols at {s:?} (rank_ψ)"
                );
            }
        }
        if let Some(rb) = env.r[b].as_ref() {
            assert_eq!(rb.k, b, "{tag}: r[{b}] lives on bond {}", rb.k);
            let (rr, cc) = (rb.row_ranks(), rb.col_ranks());
            for s in space.bond_sectors(b) {
                assert_eq!(
                    rr[s],
                    psi.rank(b, s),
                    "{tag}: r[{b}] rows at {s:?} (rank_ψ)"
                );
                assert_eq!(cc[s], x.rank(b, s), "{tag}: r[{b}] cols at {s:?} (rank_x)");
            }
        }
    }
}

fn assert_local_shape<T: Scalar>(
    p: &ContractedSparseCores<T>,
    b: &ContractedSparseCores<T>,
    tag: &str,
) {
    assert_eq!(p.k, b.k, "{tag}: site index");
    assert_eq!(p.row_ranks, b.row_ranks, "{tag}: row ranks (bond k)");
    assert_eq!(p.col_ranks, b.col_ranks, "{tag}: col ranks (bond k+2)");
    assert_eq!(p.mid_qn, b.mid_qn, "{tag}: mid sector list");
    for &m in b.mid_qn.iter() {
        match (p.block(m), b.block(m)) {
            (None, None) => {}
            (Some(pb), Some(bb)) => assert_eq!(
                (pb.nrows(), pb.ncols()),
                (bb.nrows(), bb.ncols()),
                "{tag}: mid block {m:?} dims"
            ),
            _ => panic!("{tag}: mid block {m:?} present on exactly one side"),
        }
    }
    assert_eq!(p.num_coeffs(), b.num_coeffs(), "{tag}: coefficient count");
}

fn pair_values<T: Scalar>(
    env: &OverlapEnvs<T>,
    x: &TtVector<T>,
    psi: &TtVector<T>,
    k: usize,
    tag: &str,
) -> (T, T) {
    let p = local_overlap_vector(env, psi, k);
    let b = contract(x.core(k), x.core(k + 1));
    assert_local_shape(&p, &b, &format!("{tag}: pair ({k}, {})", k + 1));
    (p.dot(&b), qntt::dot(psi, x))
}

#[test]
fn tms_c1_fold_identity_at_every_pair() {
    let d = 6usize;
    for (space, label) in [
        (QnSpace::new(d, 1, 1), "(6,1,1)"),
        (QnSpace::new(d, 2, 2), "(6,2,2)"),
    ] {
        let mut prng = seeded(0xC1_0001);
        let psi: TtVector<f64> = randomized_state(space, 2, 3, &mut prng);
        assert!(psi.check(), "{label}: psi");
        assert_eq!(psi.ortho, None, "{label}: psi must be non-canonical");

        let mut xrng = seeded(0xC1_0002);
        let xraw: TtVector<f64> = randomized_state(space, 2, 4, &mut xrng);
        let (xf, _) = two_site_frame_abelian::<f64>(space, 0, 0xC1_0003);
        let mut xmid = xf.clone();
        xmid.move_core(3, false);

        for (x, gauge) in [
            (xraw.clone(), "raw (non-canonical, padded grid)"),
            (xf.clone(), "canonical about pair (0,1)"),
            (xmid.clone(), "canonical about site 3"),
        ] {
            let tag = format!("tms_c1 {label} x = {gauge}");
            let scale = psi.norm() * x.norm();
            let reference = qntt::dot(&psi, &x);
            assert!(
                reference.abs_s() > 1e-3 * scale,
                "{tag}: ⟨ψ,x⟩ = {:.3e} is below 1e-3·‖ψ‖‖x‖ = {:.3e} — the identity would be \
                 vacuous; change the seed, never the tolerance",
                reference.abs_s(),
                1e-3 * scale
            );

            let env = full_envs_abelian(&x, &psi);
            assert_env_shapes(&env, &x, &psi, &tag);

            let mut worst = 0.0f64;
            for k in 0..d - 1 {
                let (got, want) = pair_values(&env, &x, &psi, k, &tag);
                let err = (got - want).abs_s();
                worst = worst.max(err);
                assert!(
                    err <= 1e-14 * scale,
                    "{tag}: pair ({k}, {}) gives dot_local(p, B) = {:.17e} against ⟨ψ,x⟩ = \
                     {:.17e} (|Δ| = {err:.3e} > 1e-14·‖ψ‖‖x‖ = {:.3e})",
                    k + 1,
                    got.re_s(),
                    want.re_s(),
                    1e-14 * scale
                );
            }
            println!("{tag}: worst |dot_local(p,B) − ⟨ψ,x⟩| over the 5 pairs = {worst:.3e}");
        }
    }
}

#[test]
fn tms_c1_overlap_envs_fill_matches_the_steps() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut prng = seeded(0xC1_0011);
    let psi: TtVector<f64> = randomized_state(space, 2, 3, &mut prng);
    let (x, _) = two_site_frame_abelian::<f64>(space, 0, 0xC1_0012);

    let fe = overlap_envs(&x, &psi);
    let full = full_envs_abelian(&x, &psi);

    assert_eq!(fe.l.len(), d + 1, "l has d+1 slots (bonds 0..=d)");
    assert_eq!(fe.r.len(), d + 1, "r has d+1 slots (bonds 0..=d)");
    assert!(fe.l[0].is_some(), "l[0] is the boundary identity");
    for b in 1..=d {
        assert!(
            fe.l[b].is_none(),
            "l[{b}] must be left unbuilt (the sweep grows it)"
        );
    }
    assert!(fe.r[0].is_none(), "r[0] is never needed and never built");
    for b in 1..=d {
        assert!(fe.r[b].is_some(), "r[{b}] must be built by overlap_envs");
        sketch_bits_eq(
            fe.r[b].as_ref().unwrap(),
            full.r[b].as_ref().unwrap(),
            &format!("overlap_envs r[{b}] vs the right-to-left loop"),
        );
    }
    sketch_bits_eq(
        fe.l[0].as_ref().unwrap(),
        full.l[0].as_ref().unwrap(),
        "overlap_envs l[0] vs Sketch::identity",
    );
    assert_env_shapes(&fe, &x, &psi, "tms_c1 overlap_envs");

    let (got, want) = pair_values(&fe, &x, &psi, 0, "tms_c1 overlap_envs pair (0,1)");
    let scale = psi.norm() * x.norm();
    assert!(
        (got - want).abs_s() <= 1e-14 * scale,
        "overlap_envs pair (0,1): |Δ| = {:.3e}",
        (got - want).abs_s()
    );
}

#[test]
fn tms_c1_incremental_stack_is_bitwise_the_fresh_one() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut prng = seeded(0xC1_0021);
    let psi: TtVector<f64> = randomized_state(space, 2, 3, &mut prng);
    let (x0, _) = two_site_frame_abelian::<f64>(space, 0, 0xC1_0022);
    let scale = psi.norm() * x0.norm();

    let mut x = x0.clone();
    let mut env = overlap_envs(&x, &psi);
    for k in 0..d - 2 {
        x.move_core(k + 1, false);
        let lk = env.l[k]
            .clone()
            .expect("l[k] present from the previous bond");
        env.l[k + 1] = Some(overlap_step_right(&x, &psi, &lk, k));
        let fresh = full_envs_abelian(&x, &psi);
        sketch_bits_eq(
            env.l[k + 1].as_ref().unwrap(),
            fresh.l[k + 1].as_ref().unwrap(),
            &format!("forward: incremental l[{}] vs fresh", k + 1),
        );
        let mixed = OverlapEnvs {
            l: env.l.clone(),
            r: fresh.r.clone(),
        };
        let (got, want) = pair_values(&mixed, &x, &psi, k + 1, "forward mixed stack");
        assert!(
            (got - want).abs_s() <= 1e-14 * scale,
            "forward pair ({}, {}): |Δ| = {:.3e}",
            k + 1,
            k + 2,
            (got - want).abs_s()
        );
    }

    let mut x = x0.clone();
    x.move_core(d - 1, false);
    let mut env = full_envs_abelian(&x, &psi);
    for k in (1..d - 1).rev() {
        x.move_core(k, false);
        let rk2 = env.r[k + 2]
            .clone()
            .expect("r[k+2] present from the previous bond");
        env.r[k + 1] = Some(overlap_step_left(&x, &psi, &rk2, k + 1));
        let fresh = full_envs_abelian(&x, &psi);
        sketch_bits_eq(
            env.r[k + 1].as_ref().unwrap(),
            fresh.r[k + 1].as_ref().unwrap(),
            &format!("back: incremental r[{}] vs fresh", k + 1),
        );
        let mixed = OverlapEnvs {
            l: fresh.l.clone(),
            r: env.r.clone(),
        };
        let (got, want) = pair_values(&mixed, &x, &psi, k - 1, "back mixed stack");
        assert!(
            (got - want).abs_s() <= 1e-14 * scale,
            "back pair ({}, {k}): |Δ| = {:.3e}",
            k - 1,
            (got - want).abs_s()
        );
    }
}

#[test]
fn tms_c1_c64_local_image_is_antilinear_in_x() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let mut prng = seeded(0xC1_0031);
    let psi: TtVector<C64> = randomized_state(space, 2, 3, &mut prng);
    let mut xrng = seeded(0xC1_0032);
    let xraw: TtVector<C64> = randomized_state(space, 2, 4, &mut xrng);
    let mut xcan = xraw.clone();
    xcan.right_orthogonalize(true);

    for (x, gauge, use_shipped) in [
        (xraw.clone(), "raw", false),
        (xcan.clone(), "right-orthogonalized", true),
    ] {
        let tag = format!("tms_c1 C64 x = {gauge}");
        let scale = psi.norm() * x.norm();
        let reference = qntt::dot(&psi, &x);
        assert!(
            reference.abs_s() > 1e-3 * scale,
            "{tag}: ⟨ψ,x⟩ too small ({:.3e}) — change the seed",
            reference.abs_s()
        );
        assert!(
            reference.im_s().abs() > 1e-3 * scale,
            "{tag}: VACUOUS — |Im ⟨ψ,x⟩| = {:.3e} is below 1e-3·‖ψ‖‖x‖; the complex path is a \
             real computation in disguise",
            reference.im_s().abs()
        );
        assert!(
            rho_abelian(&x) < 0.99,
            "{tag}: VACUOUS — x has a single global phase (ρ = {:.6})",
            rho_abelian(&x)
        );

        let env = full_envs_abelian(&x, &psi);
        assert_env_shapes(&env, &x, &psi, &tag);
        let pairs: Vec<usize> = if use_shipped {
            vec![0]
        } else {
            (0..d - 1).collect()
        };
        let shipped = overlap_envs(&x, &psi);
        for k in pairs {
            let e = if use_shipped { &shipped } else { &env };
            let (got, want) = pair_values(e, &x, &psi, k, &tag);
            assert!(
                (got - want).abs_s() <= 1e-14 * scale,
                "{tag}: pair ({k}, {}) |Δ| = {:.3e}",
                k + 1,
                (got - want).abs_s()
            );
            assert!(
                (got - want.conj_s()).abs_s() > 1e-3 * scale,
                "{tag}: pair ({k}, {}) ALSO matches conj(⟨ψ,x⟩) to {:.3e} — a conjugation is on \
                 the wrong operand (p must be antilinear in x, linear in ψ)",
                k + 1,
                (got - want.conj_s()).abs_s()
            );
        }
    }
}
