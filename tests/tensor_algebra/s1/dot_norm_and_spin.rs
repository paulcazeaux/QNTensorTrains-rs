// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s1::util::*;
use crate::s1::*;

#[test]
fn new3_norm_and_dot_through_expand_at_every_admissible_two_sz() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFEED_BEEF);
    for &(d, n, two_s) in &[(6usize, 4usize, 2usize), (7, 5, 1), (5, 4, 0)] {
        let space = Su2Space::new(d, n, two_s);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let y: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let cx: Su2TtVector<f64> = random_csf(d, n, two_s, &mut rng);
        let s_xy = su2_dot(&x, &y);
        let nx = x.norm();
        let mut ts = two_s as i64;
        while ts >= -(two_s as i64) {
            let ex = expand(&x, ts);
            let ey = expand(&y, ts);
            let ecx = expand(&cx, ts);
            assert!(
                (nx - ex.norm()).abs() < 1e-12 * nx.max(1.0),
                "norm 2Sz={ts}"
            );
            assert!((ecx.norm() - 1.0).abs() < 1e-12, "csf norm 2Sz={ts}");
            assert!(
                (s_xy - ab_dot(&ex, &ey)).abs() < 1e-12 * s_xy.abs().max(1.0),
                "dot 2Sz={ts}"
            );
            ts -= 2;
        }
    }
}

fn number_op<T: Scalar>(v: &TtVector<T>, site: usize, spin: Spin) -> TtVector<T> {
    let o = Orbital::new(site, spin);
    adag_i_a_j(v, o, o, T::one())
}

fn sz_apply<T: Scalar>(v: &TtVector<T>) -> TtVector<T> {
    let d = v.space.d;
    let mut terms: Vec<TtVector<T>> = Vec::with_capacity(2 * d);
    for p in 0..d {
        terms.push(number_op(v, p, Spin::Up));
    }
    for p in 0..d {
        terms.push(&number_op(v, p, Spin::Dn) * T::from_f64(-1.0));
    }
    let mut acc = terms[0].clone();
    for t in &terms[1..] {
        acc = &acc + t;
    }
    &acc * T::from_f64(0.5)
}

fn s_minus_s_plus<T: Scalar>(v: &TtVector<T>) -> TtVector<T> {
    let d = v.space.d;
    let mut terms: Vec<TtVector<T>> = Vec::new();
    for p in 0..d {
        let ndn_v = number_op(v, p, Spin::Dn);
        terms.push(ndn_v.clone());
        let nup_ndn_v = number_op(&ndn_v, p, Spin::Up);
        terms.push(&nup_ndn_v * T::from_f64(-1.0));
    }
    for p in 0..d {
        for q in 0..d {
            if p == q {
                continue;
            }
            let i = Orbital::new(p, Spin::Up);
            let j = Orbital::new(q, Spin::Up);
            let k = Orbital::new(q, Spin::Dn);
            let l = Orbital::new(p, Spin::Dn);
            terms.push(adag_i_adag_k_a_l_a_j(v, i, j, k, l, T::from_f64(-1.0)));
        }
    }
    let mut acc = terms[0].clone();
    for t in &terms[1..] {
        acc = &acc + t;
    }
    acc
}

fn s_squared<T: Scalar>(v: &TtVector<T>) -> TtVector<T> {
    let smsp = s_minus_s_plus(v);
    let sz = sz_apply(v);
    let sz2 = sz_apply(&sz);
    let sum1 = &smsp + &sz2;
    &sum1 + &sz
}

#[test]
fn new4_expand_tt_csf_is_an_s_squared_eigenstate() {
    let triplet: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Up]);
    for &two_sz in &[2i64, 0, -2] {
        let v = expand(&triplet, two_sz);
        let target_s = 1.0_f64;
        let w = s_squared(&v);
        assert_eq!(w.space, v.space, "S^2 must be number/space conserving");
        let want = target_s * (target_s + 1.0);
        let scaled = want * &v;
        let diff = &w - &scaled;
        let tol = 1e-11 * v.norm().max(1.0);
        assert!(
            diff.norm() <= tol,
            "twoSz={two_sz}: |S^2 v - S(S+1) v| = {} > tol {tol}",
            diff.norm()
        );
    }

    let singlet: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Dn]);
    let v = expand(&singlet, 0);
    let w = s_squared(&v);
    assert_eq!(w.space, v.space);
    assert!(
        w.norm() <= 1e-11 * v.norm().max(1.0),
        "singlet S^2 eigenvalue should be 0"
    );
}
