// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::C64;
use qntt::solvers::InnerProductVector;
use qntt::su2::coreops::{core_times_adjoint_core, mul_sketch_left, mul_sketch_right};
use qntt::su2::frame::Su2Sketch;
use qntt::su2::solvers::overlap_env::{
    OverlapEnvs as Su2OverlapEnvs, local_overlap_vector, overlap_envs, overlap_step_left,
    overlap_step_right,
};
use qntt_testkit::randomized_su2_state;

fn full_envs_su2<T: Scalar>(x: &Su2TtVector<T>, psi: &Su2TtVector<T>) -> Su2OverlapEnvs<T> {
    let d = x.space.d;
    let mut l: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    let mut r: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    l[0] = Some(Su2Sketch::identity(x.space, 0, 1));
    for k in 0..d {
        let prev = l[k].clone().expect("full_envs_su2: l[k] built in order");
        l[k + 1] = Some(overlap_step_right(x, psi, &prev, k));
    }
    r[d] = Some(Su2Sketch::identity(x.space, d, 1));
    for k in (1..d).rev() {
        let nxt = r[k + 1]
            .clone()
            .expect("full_envs_su2: r[k+1] built in order");
        r[k] = Some(overlap_step_left(x, psi, &nxt, k));
    }
    Su2OverlapEnvs { l, r }
}

fn wrong_envs_su2<T: Scalar>(x: &Su2TtVector<T>, psi: &Su2TtVector<T>) -> Su2OverlapEnvs<T> {
    let d = x.space.d;
    let mut l: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    let mut r: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    l[0] = Some(Su2Sketch::identity(x.space, 0, 1));
    for k in 0..d {
        let prev = l[k].clone().expect("wrong_envs_su2: l[k] built in order");
        l[k + 1] = Some(overlap_step_right(x, psi, &prev, k));
    }
    r[d] = Some(Su2Sketch::identity(x.space, d, 1));
    for k in (1..d).rev() {
        let nxt = r[k + 1]
            .clone()
            .expect("wrong_envs_su2: r[k+1] built in order");
        let t = mul_sketch_right(psi.core(k), &nxt);
        r[k] = Some(core_times_adjoint_core(&t, x.core(k)));
    }
    Su2OverlapEnvs { l, r }
}

fn sketch_bits_eq_su2<T: Scalar>(a: &Su2Sketch<T>, b: &Su2Sketch<T>, what: &str) {
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

fn assert_env_shapes_su2<T: Scalar>(
    env: &Su2OverlapEnvs<T>,
    x: &Su2TtVector<T>,
    psi: &Su2TtVector<T>,
    tag: &str,
) {
    let space = x.space;
    let d = space.d;
    for b in 0..=d {
        if let Some(lb) = env.l[b].as_ref() {
            assert_eq!(lb.k, b, "{tag}: l[{b}] bond index");
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
            assert_eq!(rb.k, b, "{tag}: r[{b}] bond index");
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

fn assert_local_shape_su2<T: Scalar>(
    p: &Su2ContractedCores<T>,
    b: &Su2ContractedCores<T>,
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

fn pair_values_su2<T: Scalar>(
    env: &Su2OverlapEnvs<T>,
    x: &Su2TtVector<T>,
    psi: &Su2TtVector<T>,
    k: usize,
    tag: &str,
) -> (T, T) {
    let p = local_overlap_vector(env, psi, k);
    let b = su2_contract(x.core(k), x.core(k + 1));
    assert_local_shape_su2(&p, &b, &format!("{tag}: pair ({k}, {})", k + 1));
    (p.dot(&b), qntt::su2::ops::dot(psi, x))
}

fn wrong_pair_value_su2<T: Scalar>(
    wrong: &Su2OverlapEnvs<T>,
    psi: &Su2TtVector<T>,
    b: &Su2ContractedCores<T>,
    k: usize,
) -> T {
    let lk = wrong.l[k].as_ref().expect("anti-oracle: l[k]");
    let rk2 = wrong.r[k + 2].as_ref().expect("anti-oracle: r[k+2]");
    let left = mul_sketch_left(lk, psi.core(k));
    let right = mul_sketch_right(psi.core(k + 1), rk2);
    let pbar = su2_contract(&left, &right);
    pbar.dot(b)
}

fn weighted_successor_sectors<T: Scalar>(
    x: &Su2TtVector<T>,
    psi: &Su2TtVector<T>,
    b: usize,
) -> usize {
    let space = x.space;
    space
        .bond_sectors(b)
        .into_iter()
        .filter(|&s| {
            x.rank(b, s) > 0
                && psi.rank(b, s) > 0
                && (qntt::su2::centerweight(s.two_s, space.two_s) - 1.0).abs() > 1e-12
        })
        .count()
}

#[test]
fn tms_c1su2_fold_identity_at_every_pair() {
    let d = 6usize;
    for two_s in [0usize, 2usize] {
        let space = Su2Space::new(d, 2, two_s);
        let label = format!("2S={two_s}");
        let mut prng = seeded(0xC1_5000 + two_s as u64);
        let psi: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut prng);
        assert!(psi.check(), "{label}: psi");
        assert_eq!(psi.ortho, None, "{label}: psi must be non-canonical");

        let mut xrng = seeded(0xC1_5100 + two_s as u64);
        let xraw: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut xrng);
        let (xf, _) = two_site_frame_su2::<f64>(space, 0, 0xC1_5200 + two_s as u64);
        let mut xmid = xf.clone();
        xmid.move_core(3, false);

        for (x, gauge) in [
            (xraw.clone(), "raw (non-canonical, padded grid)"),
            (xf.clone(), "θ-right-canonical about pair (0,1)"),
            (xmid.clone(), "θ-canonical about site 3"),
        ] {
            let tag = format!("tms_c1su2 {label} x = {gauge}");
            if two_s == 2 {
                assert_theta_cw_live(&x, &format!("{tag} (x)"));
                assert_theta_cw_live(&psi, &format!("{tag} (psi)"));
            }
            let scale = psi.norm() * x.norm();
            let reference = qntt::su2::ops::dot(&psi, &x);
            assert!(
                reference.abs_s() > 1e-3 * scale,
                "{tag}: ⟨ψ,x⟩ = {:.3e} below 1e-3·‖ψ‖‖x‖ — change the seed, never the tolerance",
                reference.abs_s()
            );

            let env = full_envs_su2(&x, &psi);
            assert_env_shapes_su2(&env, &x, &psi, &tag);

            let mut worst = 0.0f64;
            for k in 0..d - 1 {
                let (got, want) = pair_values_su2(&env, &x, &psi, k, &tag);
                let err = (got - want).abs_s();
                worst = worst.max(err);
                assert!(
                    err <= 1e-14 * scale,
                    "{tag}: pair ({k}, {}) gives dot_local(p, B) = {:.17e} against ⟨ψ,x⟩ = \
                     {:.17e} (|Δ| = {err:.3e} > 1e-14·‖ψ‖‖x‖ = {:.3e}) — the left step's weight \
                     is the suspect (R1 `rproject`, not the R2 closure)",
                    k + 1,
                    got.re_s(),
                    want.re_s(),
                    1e-14 * scale
                );
            }
            println!("{tag}: worst |dot_local(p,B) − ⟨ψ,x⟩| = {worst:.3e}");
        }
    }
}

#[test]
fn tms_c1su2_overlap_envs_fill_matches_the_steps() {
    let d = 6usize;
    for two_s in [0usize, 2usize] {
        let space = Su2Space::new(d, 2, two_s);
        let mut prng = seeded(0xC1_5300 + two_s as u64);
        let psi: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut prng);
        let (x, _) = two_site_frame_su2::<f64>(space, 0, 0xC1_5400 + two_s as u64);
        if two_s == 2 {
            assert_theta_cw_live(&x, "tms_c1su2 fill (x)");
        }

        let fe = overlap_envs(&x, &psi);
        let full = full_envs_su2(&x, &psi);
        assert_eq!(fe.l.len(), d + 1, "l has d+1 slots");
        assert_eq!(fe.r.len(), d + 1, "r has d+1 slots");
        assert!(fe.l[0].is_some(), "l[0] is the boundary identity");
        for b in 1..=d {
            assert!(fe.l[b].is_none(), "l[{b}] must be left unbuilt");
            assert!(fe.r[b].is_some(), "r[{b}] must be built");
            sketch_bits_eq_su2(
                fe.r[b].as_ref().unwrap(),
                full.r[b].as_ref().unwrap(),
                &format!("2S={two_s} overlap_envs r[{b}]"),
            );
        }
        assert!(fe.r[0].is_none(), "r[0] is never built");
        assert_env_shapes_su2(&fe, &x, &psi, &format!("2S={two_s} overlap_envs"));
    }
}

#[test]
fn tms_c1su2_incremental_stack_is_bitwise_the_fresh_one() {
    let d = 6usize;
    for two_s in [0usize, 2usize] {
        let space = Su2Space::new(d, 2, two_s);
        let mut prng = seeded(0xC1_5500 + two_s as u64);
        let psi: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut prng);
        let (x0, _) = two_site_frame_su2::<f64>(space, 0, 0xC1_5600 + two_s as u64);
        let scale = psi.norm() * x0.norm();

        let mut x = x0.clone();
        let mut env = overlap_envs(&x, &psi);
        for k in 0..d - 2 {
            x.move_core(k + 1, false);
            let lk = env.l[k].clone().expect("l[k] present");
            env.l[k + 1] = Some(overlap_step_right(&x, &psi, &lk, k));
            let fresh = full_envs_su2(&x, &psi);
            sketch_bits_eq_su2(
                env.l[k + 1].as_ref().unwrap(),
                fresh.l[k + 1].as_ref().unwrap(),
                &format!("2S={two_s} forward incremental l[{}]", k + 1),
            );
            let mixed = Su2OverlapEnvs {
                l: env.l.clone(),
                r: fresh.r.clone(),
            };
            let (got, want) = pair_values_su2(&mixed, &x, &psi, k + 1, "forward mixed stack");
            assert!(
                (got - want).abs_s() <= 1e-14 * scale,
                "2S={two_s} forward pair ({}, {}): |Δ| = {:.3e}",
                k + 1,
                k + 2,
                (got - want).abs_s()
            );
        }

        let mut x = x0.clone();
        x.move_core(d - 1, false);
        let mut env = full_envs_su2(&x, &psi);
        for k in (1..d - 1).rev() {
            x.move_core(k, false);
            let rk2 = env.r[k + 2].clone().expect("r[k+2] present");
            env.r[k + 1] = Some(overlap_step_left(&x, &psi, &rk2, k + 1));
            let fresh = full_envs_su2(&x, &psi);
            sketch_bits_eq_su2(
                env.r[k + 1].as_ref().unwrap(),
                fresh.r[k + 1].as_ref().unwrap(),
                &format!("2S={two_s} back incremental r[{}]", k + 1),
            );
            let mixed = Su2OverlapEnvs {
                l: fresh.l.clone(),
                r: env.r.clone(),
            };
            let (got, want) = pair_values_su2(&mixed, &x, &psi, k - 1, "back mixed stack");
            assert!(
                (got - want).abs_s() <= 1e-14 * scale,
                "2S={two_s} back pair ({}, {k}): |Δ| = {:.3e}",
                k - 1,
                (got - want).abs_s()
            );
        }
    }
}

#[test]
fn tms_c1su2_r2_closure_stack_breaks_the_identity() {
    let d = 6usize;
    for two_s in [0usize, 2usize] {
        let space = Su2Space::new(d, 2, two_s);
        let label = format!("2S={two_s}");
        let mut prng = seeded(0xC1_5700 + two_s as u64);
        let psi: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut prng);
        let (x, _) = two_site_frame_su2::<f64>(space, 0, 0xC1_5800 + two_s as u64);
        if two_s == 2 {
            assert_theta_cw_live(&x, &format!("{label} anti-oracle (x)"));
            assert_theta_cw_live(&psi, &format!("{label} anti-oracle (psi)"));
        }

        let scale = psi.norm() * x.norm();
        let reference = qntt::su2::ops::dot(&psi, &x);
        assert!(
            reference.abs_s() > 1e-3 * scale,
            "{label}: ⟨ψ,x⟩ too small — change the seed"
        );
        let right = full_envs_su2(&x, &psi);
        let wrong = wrong_envs_su2(&x, &psi);

        let mut rel = vec![0.0f64; d - 1];
        let mut live = vec![0usize; d - 1];
        for k in 0..d - 1 {
            let b = su2_contract(x.core(k), x.core(k + 1));
            let good = pair_values_su2(&right, &x, &psi, k, &label).0;
            assert!(
                (good - reference).abs_s() <= 1e-14 * scale,
                "{label}: the SHIPPED stack must satisfy the identity at pair {k} first"
            );
            let bad = wrong_pair_value_su2(&wrong, &psi, &b, k);
            rel[k] = (bad - reference).abs_s() / scale;
            live[k] = weighted_successor_sectors(&x, &psi, k + 2);
        }
        println!(
            "{label} R2 anti-oracle: relative deviation per pair = {:?}\n  \
             weighted (cw≠1) bond-(k+2) sectors carrying rank on both trains = {live:?}",
            rel.iter().map(|v| format!("{v:.3e}")).collect::<Vec<_>>()
        );

        for k in 0..d - 2 {
            assert!(
                live[k] > 0,
                "{label}: VACUOUS at pair {k} — bond {} carries no populated sector with \
                 centerweight ≠ 1, so the R2 stack cannot differ there",
                k + 2
            );
            assert!(
                rel[k] > 1e-3,
                "{label}: the R2 closure stack reproduces the identity at pair ({k}, {}) to \
                 {:.3e} relative — either `overlap_step_left` no longer uses the θ²-weighted \
                 frame dual, or this fixture cannot discriminate (change the SEED, never this \
                 threshold)",
                k + 1,
                rel[k]
            );
        }
        assert!(
            rel[d - 2] <= 1e-13,
            "{label}: the wrong stack must still be exact at the BLIND last pair (got {:.3e}) — \
             a failure here is a boundary-handling bug, not the weight",
            rel[d - 2]
        );
    }
}

#[test]
fn tms_c1su2_r2_deviation_is_sign_definite_at_a_canonical_frame() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    for k in 0..d - 2 {
        let (x, b) = two_site_frame_su2::<f64>(space, k, 0xC1_5900 + k as u64);
        let psi = x.clone();
        let scale = psi.norm() * x.norm();
        let live = weighted_successor_sectors(&x, &psi, k + 2);
        assert!(
            live > 0,
            "pair {k}: VACUOUS — bond {} carries no populated centerweighted sector",
            k + 2
        );

        let right = full_envs_su2(&x, &psi);
        let wrong = wrong_envs_su2(&x, &psi);
        let reference = qntt::su2::ops::dot(&psi, &x);
        let good = pair_values_su2(&right, &x, &psi, k, "ψ = x").0;
        assert!(
            (good - reference).abs_s() <= 1e-14 * scale,
            "pair {k}: the shipped stack must be exact first"
        );
        let bad = wrong_pair_value_su2(&wrong, &psi, &b, k);
        let dev = bad.re_s() - reference.re_s();
        println!(
            "ψ = x at pair {k}: ⟨x,x⟩ = {:.12e}, R2 value = {:.12e}, deviation = {dev:+.6e} \
             ({live} weighted successor sectors)",
            reference.re_s(),
            bad.re_s()
        );
        assert!(
            dev > 1e-3 * scale,
            "pair {k}: the R2 deviation Σ_r cw(r)(cw(r)−1)‖B_r‖² must be strictly POSITIVE and \
             O(1) here (got {dev:+.3e}); a null value means `overlap_step_left` folds with the \
             closure, or `two_site_frame_su2` is no longer θ-right-canonical"
        );
    }
}

#[test]
fn tms_c1su2_c64_antilinear_in_x_and_anti_oracle() {
    let d = 6usize;
    for two_s in [0usize, 2usize] {
        let space = Su2Space::new(d, 2, two_s);
        let label = format!("2S={two_s} C64");
        let mut prng = seeded(0xC1_5A00 + two_s as u64);
        let psi: Su2TtVector<C64> = randomized_su2_state(space, 2, 3, &mut prng);
        let mut xrng = seeded(0xC1_5B00 + two_s as u64);
        let x: Su2TtVector<C64> = randomized_su2_state(space, 2, 4, &mut xrng);
        if two_s == 2 {
            assert_theta_cw_live(&x, &format!("{label} (x)"));
        }

        let scale = psi.norm() * x.norm();
        let reference = qntt::su2::ops::dot(&psi, &x);
        assert!(
            reference.abs_s() > 1e-3 * scale,
            "{label}: ⟨ψ,x⟩ too small — change the seed"
        );
        assert!(
            reference.im_s().abs() > 1e-3 * scale,
            "{label}: VACUOUS — |Im ⟨ψ,x⟩| = {:.3e}; the complex path is a real computation in \
             disguise",
            reference.im_s().abs()
        );

        let env = full_envs_su2(&x, &psi);
        let wrong = wrong_envs_su2(&x, &psi);
        let mut worst_wrong = 0.0f64;
        for k in 0..d - 1 {
            let (got, want) = pair_values_su2(&env, &x, &psi, k, &label);
            assert!(
                (got - want).abs_s() <= 1e-14 * scale,
                "{label}: pair {k} |Δ| = {:.3e}",
                (got - want).abs_s()
            );
            assert!(
                (got - want.conj_s()).abs_s() > 1e-3 * scale,
                "{label}: pair {k} ALSO matches conj(⟨ψ,x⟩) — a conjugation sits on the wrong \
                 operand (p is antilinear in x, linear in ψ)"
            );
            if k + 2 < d {
                let b = su2_contract(x.core(k), x.core(k + 1));
                let bad = wrong_pair_value_su2(&wrong, &psi, &b, k);
                worst_wrong = worst_wrong.max((bad - want).abs_s() / scale);
            }
        }
        println!(
            "{label}: worst R2 relative deviation over pairs 0..{} = {worst_wrong:.3e}",
            d - 2
        );
        assert!(
            worst_wrong > 1e-3,
            "{label}: the R2 closure stack is indistinguishable on this fixture (worst \
             {worst_wrong:.3e}) — change the seed, never the threshold"
        );
    }
}
