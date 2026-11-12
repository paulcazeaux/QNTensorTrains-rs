// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use super::MultiTt;
use crate::core::{Family, SparseCore};
use crate::ops::block_tests::open_block;
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, Sector};
use crate::scalar::Scalar;
use crate::test_util::{
    dense_diff_norm, dense_dot, dense_norm, pg3_space_d6, randomized_state, seeded,
};
use crate::{C64, TtVector};

fn dense_columns<T: Scalar>(x: &TtVector<T>) -> Vec<Vec<T>> {
    (0..x.boundary_width())
        .map(|j| x.column(j).to_dense())
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

fn container_at<T: Scalar>(space: QnSpace, pos: usize, n: usize, seed: u64) -> MultiTt<T> {
    let mut rng = seeded(seed);
    let mut base: TtVector<T> = randomized_state(space, 2, 3, &mut rng);
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

fn coeffs_and_ranks_bits_eq<T: Scalar>(a: &TtVector<T>, b: &TtVector<T>, what: &str) {
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

fn round_trip<T: Scalar>(omega: &TtVector<T>, w: Vec<f64>, tag: &str) -> MultiTt<T> {
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
    let want = dense_columns(omega);
    for j in 0..r {
        assert_dense_close(
            &m.state(j).to_dense(),
            &want[j],
            1e-12,
            &format!("{tag}: state {j} vs column {j}"),
        );
    }
    let back = m.to_open_block();
    assert!(back.check());
    assert_eq!(back.space, omega.space, "{tag}: space");
    assert_eq!(back.boundary_width(), r, "{tag}: width");
    let got = dense_columns(&back);
    for j in 0..r {
        assert_dense_close(
            &got[j],
            &want[j],
            1e-12,
            &format!("{tag}: round-trip column {j}"),
        );
    }
    m
}

#[test]
fn bridge_round_trip_open_block_f64() {
    let omega: TtVector<f64> = open_block(QnSpace::new(4, 2, 1), 3, 3, 0xB0_0001);
    round_trip(&omega, vec![0.5, 0.3, 0.2], "open_block f64");
}

#[test]
fn bridge_round_trip_open_block_c64() {
    let omega: TtVector<C64> = open_block(QnSpace::new(4, 2, 1), 3, 3, 0xB0_0002);
    round_trip(&omega, uniform(3), "open_block c64");
}

#[test]
fn bridge_round_trip_width_one() {
    let omega: TtVector<f64> = open_block(QnSpace::new(4, 2, 1), 3, 1, 0xB0_0008);
    assert_eq!(omega.boundary_width(), 1);
    round_trip(&omega, vec![1.0], "width 1");
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
fn bridge_column_of_to_open_block_is_the_state() {
    let m: MultiTt<f64> = container_at(QnSpace::new(5, 2, 2), 4, 3, 0xB0_0101);
    column_is_state(&m, "f64");
    let mc: MultiTt<C64> = container_at(QnSpace::new(4, 2, 1), 3, 2, 0xB0_0102);
    column_is_state(&mc, "c64");
}

#[test]
fn bridge_from_to_reproduces_the_container() {
    let m: MultiTt<f64> = container_at(QnSpace::new(5, 2, 2), 4, 3, 0xB0_0201);
    let b = m.to_open_block();
    let m2 = MultiTt::from_open_block(&b, m.weights.clone());
    m2.check();
    assert_eq!(m2.core_position, m.core_position);
    assert_eq!(m2.weights, m.weights);
    assert_eq!(m2.nstates(), m.nstates());
    for j in 0..m.nstates() {
        assert_dense_close(
            &m2.state(j).to_dense(),
            &m.state(j).to_dense(),
            1e-12,
            &format!("state {j}"),
        );
    }
}

#[test]
fn bridge_shared_cores_are_isometries_on_an_over_ranked_block() {
    let space = QnSpace::new(4, 2, 1);
    let omega: TtVector<f64> = open_block(space, 3, 3, 0xB0_0401);
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
fn bridge_refuses_a_block_with_two_boundary_sectors() {
    let (_pg, space) = pg3_space_d6();
    let d = space.d;
    let omega: TtVector<f64> = open_block(space, 2, 2, 0xB0_0501);
    let other = Sector::with_irrep(space.nup, space.ndn, Irrep::IDENTITY);
    assert_ne!(
        space.target,
        Irrep::IDENTITY,
        "fixture: the target must not be the identity"
    );
    let mut col = omega.ranks[d].clone();
    col[other] = 1;
    let src = omega.core(d - 1);
    let mut core = SparseCore::<f64>::zeros(space, d - 1, &omega.ranks[d - 1], &col);
    for f in Family::ALL {
        let secs: Vec<Sector> = src.transitions(f).collect();
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
    let bad = TtVector::new(space, cores);
    assert_eq!(
        bad.rank(d, other),
        1,
        "fixture: the second boundary sector carries rank"
    );
    let _ = MultiTt::from_open_block(&bad, uniform(2));
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn bridge_refuses_weights_of_the_wrong_length() {
    let omega: TtVector<f64> = open_block(QnSpace::new(4, 2, 1), 3, 3, 0xB0_0502);
    let _ = MultiTt::from_open_block(&omega, vec![0.5, 0.5]);
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn bridge_refuses_weights_that_do_not_sum_to_one() {
    let omega: TtVector<f64> = open_block(QnSpace::new(4, 2, 1), 3, 3, 0xB0_0503);
    let _ = MultiTt::from_open_block(&omega, vec![0.5, 0.3, 0.1]);
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn bridge_refuses_increasing_weights() {
    let omega: TtVector<f64> = open_block(QnSpace::new(4, 2, 1), 3, 3, 0xB0_0504);
    let _ = MultiTt::from_open_block(&omega, vec![0.2, 0.3, 0.5]);
}

#[test]
#[should_panic(expected = "MultiTt::from_open_block")]
fn bridge_refuses_a_zero_width_block() {
    let omega: TtVector<f64> = open_block(QnSpace::new(4, 2, 1), 3, 0, 0xB0_0505);
    assert_eq!(omega.boundary_width(), 0);
    let _ = MultiTt::from_open_block(&omega, Vec::new());
}

#[test]
#[should_panic(expected = "MultiTt::to_open_block")]
fn bridge_to_open_block_refuses_a_container_at_site_zero() {
    let m: MultiTt<f64> = container_at(QnSpace::new(4, 2, 1), 0, 2, 0xB0_0506);
    let _ = m.to_open_block();
}

#[test]
#[should_panic(expected = "MultiTt::to_open_block")]
fn bridge_to_open_block_refuses_an_interior_container() {
    let m: MultiTt<f64> = container_at(QnSpace::new(5, 2, 2), 2, 2, 0xB0_0507);
    let _ = m.to_open_block();
}
