// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use super::MultiTt;
use crate::C64;
use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::expand::expand;
use crate::su2::ops::block_tests::su2_open_block;
use crate::su2::qn::{Su2Sector, Su2Space};
use crate::su2::vector::Su2TtVector;
use crate::test_util::{
    dense_diff_norm, dense_dot, dense_norm, randomized_su2_state, seeded, su2_pg_space,
};

fn members(space: &Su2Space) -> Vec<i64> {
    let two_s = space.two_s as i64;
    (-two_s..=two_s).step_by(2).collect()
}

fn dense_columns<T: Scalar>(x: &Su2TtVector<T>, two_sz: i64) -> Vec<Vec<T>> {
    (0..x.boundary_width())
        .map(|j| expand(&x.column(j), two_sz).to_dense())
        .collect()
}

fn assert_dense_close<T: Scalar>(got: &[T], want: &[T], tol: f64, what: &str) {
    let err = dense_diff_norm(got, want);
    let scale = dense_norm(want).max(1.0);
    assert!(
        err <= tol * scale,
        "{what}: ‖Δ‖ = {err:.3e} > {tol:.0e}·{scale:.3e}"
    );
}

fn container_at<T: Scalar>(space: Su2Space, pos: usize, n: usize, seed: u64) -> MultiTt<T> {
    let mut rng = seeded(seed);
    let mut base: Su2TtVector<T> = randomized_su2_state(space, 2, 3, &mut rng);
    base.move_core(pos, false);
    let c0 = base.core(pos).clone();
    let mut centers = vec![c0.clone()];
    for _ in 1..n {
        let mut c = c0.clone();
        c.for_each_coeff_mut(|z| *z = T::sample_standard(&mut rng));
        centers.push(c);
    }
    let w: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
    let tot: f64 = w.iter().sum();
    let m = MultiTt {
        base,
        centers,
        weights: w.iter().map(|x| x / tot).collect(),
        core_position: pos,
    };
    m.check();
    m
}

fn coeffs_and_ranks_bits_eq<T: Scalar>(a: &Su2TtVector<T>, b: &Su2TtVector<T>, what: &str) {
    assert_eq!(a.space, b.space, "{what}: space");
    assert_eq!(a.ranks, b.ranks, "{what}: rank grids");
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        assert_eq!(ca.num_coeffs(), cb.num_coeffs(), "{what}: core {k} size");
        for i in 0..ca.num_coeffs() {
            let (x, y) = (ca.coeff(i), cb.coeff(i));
            assert!(
                x.re_s().to_bits() == y.re_s().to_bits()
                    && x.im_s().to_bits() == y.im_s().to_bits(),
                "{what}: core {k} coefficient {i}: {x:?} vs {y:?}"
            );
        }
    }
}

fn uniform(n: usize) -> Vec<f64> {
    vec![1.0 / n as f64; n]
}

fn round_trip<T: Scalar>(omega: &Su2TtVector<T>, w: Vec<f64>, tag: &str) -> MultiTt<T> {
    let d = omega.space.d;
    let r = omega.boundary_width();
    let m = MultiTt::from_open_block(omega, w.clone());
    m.check();
    assert_eq!(m.core_position, d - 1, "{tag}: container position");
    assert_eq!(m.nstates(), r, "{tag}: one state per column");
    assert_eq!(
        m.weights, w,
        "{tag}: the caller's weights, unchanged and in label order"
    );
    let back = m.to_open_block();
    assert!(back.check());
    assert_eq!(back.space, omega.space, "{tag}: space");
    assert_eq!(back.boundary_width(), r, "{tag}: width");
    for two_sz in members(&omega.space) {
        let want = dense_columns(omega, two_sz);
        let got = dense_columns(&back, two_sz);
        for j in 0..r {
            assert_dense_close(
                &expand(&m.state(j), two_sz).to_dense(),
                &want[j],
                1e-12,
                &format!("{tag}: 2Sz = {two_sz}: state {j} vs column {j}"),
            );
            assert_dense_close(
                &got[j],
                &want[j],
                1e-12,
                &format!("{tag}: 2Sz = {two_sz}: round-trip column {j}"),
            );
        }
    }
    m
}

#[test]
fn su2_bridge_round_trip_doublet() {
    let omega: Su2TtVector<f64> = su2_open_block(Su2Space::new(5, 3, 1), 3, 3, 0x5B_0001);
    round_trip(&omega, vec![0.5, 0.3, 0.2], "2S=1");
}

#[test]
fn su2_bridge_round_trip_triplet() {
    let omega: Su2TtVector<f64> = su2_open_block(Su2Space::new(5, 4, 2), 3, 3, 0x5B_0002);
    round_trip(&omega, uniform(3), "2S=2");
    let omega_c: Su2TtVector<C64> = su2_open_block(Su2Space::new(5, 4, 2), 3, 2, 0x5B_0003);
    round_trip(&omega_c, uniform(2), "2S=2 c64");
}

#[test]
fn su2_bridge_round_trip_singlet_control() {
    let omega: Su2TtVector<f64> = su2_open_block(Su2Space::new(4, 4, 0), 3, 3, 0x5B_0004);
    round_trip(&omega, uniform(3), "2S=0");
}

fn column_is_state<T: Scalar>(m: &MultiTt<T>, tag: &str) {
    let d = m.base.space.d;
    let b = m.to_open_block();
    assert!(b.check());
    assert_eq!(b.boundary_width(), m.nstates(), "{tag}: width");
    assert_eq!(b.ortho, Some(d - 1), "{tag}: block gauge flag");
    for j in 0..m.nstates() {
        let col = b.column(j);
        let st = m.state(j);
        coeffs_and_ranks_bits_eq(&col, &st, &format!("{tag}: column {j} vs state {j}"));
        assert_eq!(col.ortho, st.ortho, "{tag}: column {j} ortho flag");
    }
}

#[test]
fn su2_bridge_column_of_to_open_block_is_the_state() {
    let m: MultiTt<f64> = container_at(Su2Space::new(5, 3, 1), 4, 3, 0x5B_0101);
    column_is_state(&m, "2S=1");
    let m2: MultiTt<C64> = container_at(Su2Space::new(5, 4, 2), 4, 2, 0x5B_0102);
    column_is_state(&m2, "2S=2 c64");
}

#[test]
fn su2_bridge_from_to_reproduces_the_container() {
    let m: MultiTt<f64> = container_at(Su2Space::new(5, 4, 2), 4, 3, 0x5B_0201);
    let b = m.to_open_block();
    let m2 = MultiTt::from_open_block(&b, m.weights.clone());
    m2.check();
    assert_eq!(m2.core_position, m.core_position);
    assert_eq!(m2.weights, m.weights);
    assert_eq!(m2.nstates(), m.nstates());
    for two_sz in members(&m.base.space) {
        for j in 0..m.nstates() {
            assert_dense_close(
                &expand(&m2.state(j), two_sz).to_dense(),
                &expand(&m.state(j), two_sz).to_dense(),
                1e-12,
                &format!("2Sz = {two_sz}: state {j}"),
            );
        }
    }
}

#[test]
fn su2_bridge_shared_cores_are_isometries_on_an_over_ranked_block() {
    let space = Su2Space::new(5, 4, 2);
    let omega: Su2TtVector<f64> = su2_open_block(space, 3, 3, 0x5B_0401);
    let over = space
        .bond_sectors(1)
        .into_iter()
        .any(|s| omega.rank(1, s) >= 2);
    assert!(
        over,
        "VACUOUS — the fixture is not over-ranked at bond 1: {:?}",
        omega.ranks[1]
    );
    let m = MultiTt::from_open_block(&omega, uniform(3));
    let g = m.gram(&m);
    for i in 0..3 {
        for j in 0..3 {
            let (mut u, mut v) = (Vec::new(), Vec::new());
            m.centers[i].for_each_coeff(|z| u.push(z));
            m.centers[j].for_each_coeff(|z| v.push(z));
            let want = dense_dot(&u, &v);
            assert!(
                (*g.get(i, j) - want).abs() <= 1e-12 * want.abs().max(1.0),
                "G[{i},{j}] = {}, centre coefficient Gram {want}",
                g.get(i, j)
            );
        }
    }
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn su2_bridge_refuses_a_block_with_two_boundary_sectors() {
    let (_pg, space) = su2_pg_space(5, 3, 1);
    let d = space.d;
    assert_ne!(
        space.target,
        Irrep::IDENTITY,
        "fixture: non-identity target"
    );
    let omega: Su2TtVector<f64> = su2_open_block(space, 2, 2, 0x5B_0501);
    let other = Su2Sector::with_irrep(space.n, space.two_s, Irrep::IDENTITY);
    let mut col = omega.ranks[d].clone();
    col[other] = 1;
    let src = omega.core(d - 1);
    let mut core = Su2Core::<f64>::zeros(space, d - 1, &omega.ranks[d - 1], &col);
    for f in Family::ALL {
        let secs: Vec<Su2Sector> = src.transitions(f).collect();
        for s in secs {
            let Some((rows, cols)) = src.block_dims(f, s) else {
                continue;
            };
            if core.block_dims(f, s) != Some((rows, cols)) {
                continue;
            }
            let mut buf = vec![0.0f64; rows * cols];
            src.block_copy_out(f, s, &mut buf);
            core.block_copy_in(f, s, &buf);
        }
    }
    let mut cores = omega.cores.clone();
    cores[d - 1] = core;
    let bad = Su2TtVector::new(space, cores);
    assert_eq!(
        bad.rank(d, other),
        1,
        "fixture: the second boundary sector carries rank"
    );
    let _ = MultiTt::from_open_block(&bad, uniform(2));
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn su2_bridge_refuses_weights_of_the_wrong_length() {
    let omega: Su2TtVector<f64> = su2_open_block(Su2Space::new(5, 3, 1), 3, 3, 0x5B_0502);
    let _ = MultiTt::from_open_block(&omega, vec![0.5, 0.5]);
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn su2_bridge_refuses_increasing_weights() {
    let omega: Su2TtVector<f64> = su2_open_block(Su2Space::new(5, 3, 1), 3, 3, 0x5B_0503);
    let _ = MultiTt::from_open_block(&omega, vec![0.2, 0.3, 0.5]);
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn su2_bridge_refuses_a_zero_width_block() {
    let omega: Su2TtVector<f64> = su2_open_block(Su2Space::new(5, 3, 1), 3, 0, 0x5B_0504);
    assert_eq!(omega.boundary_width(), 0);
    let _ = MultiTt::from_open_block(&omega, Vec::new());
}

#[test]
#[should_panic(expected = "MultiTt::to_open_block")]
fn su2_bridge_to_open_block_refuses_a_container_at_site_zero() {
    let m: MultiTt<f64> = container_at(Su2Space::new(5, 3, 1), 0, 2, 0x5B_0505);
    let _ = m.to_open_block();
}
