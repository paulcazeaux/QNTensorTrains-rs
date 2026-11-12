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
fn s1_13_two_electron_singlet_expand() {
    let csf: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Dn]);
    let e = expand(&csf, 0);
    let a: TtVector<f64> = tt_state(&[true, false], &[false, true]);
    let b: TtVector<f64> = tt_state(&[false, true], &[true, false]);
    let scale = 1.0 / 2.0_f64.sqrt();
    let want: Vec<f64> = a
        .to_dense()
        .iter()
        .zip(b.to_dense())
        .map(|(x, y)| (x - y) * scale)
        .collect();
    assert_dense_close(&e.to_dense(), &want, 1e-14);
}

#[test]
fn s1_14_two_electron_triplet_expand_two_members() {
    let csf: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Up]);

    let e2 = expand(&csf, 2);
    let want2: TtVector<f64> = tt_state(&[true, true], &[false, false]);
    assert_dense_close(&e2.to_dense(), &want2.to_dense(), 1e-14);

    let e0 = expand(&csf, 0);
    let a: TtVector<f64> = tt_state(&[true, false], &[false, true]);
    let b: TtVector<f64> = tt_state(&[false, true], &[true, false]);
    let scale = 1.0 / 2.0_f64.sqrt();
    let want0: Vec<f64> = a
        .to_dense()
        .iter()
        .zip(b.to_dense())
        .map(|(x, y)| (x + y) * scale)
        .collect();
    assert_dense_close(&e0.to_dense(), &want0, 1e-14);
}

#[test]
fn s1_15_onsite_pair_expand_phase_pinned_plus_one() {
    let csf: Su2TtVector<f64> = tt_csf(&[Su2Family::Occ, Su2Family::Empty]);
    let e = expand(&csf, 0);
    let want: TtVector<f64> = tt_state(&[true, false], &[true, false]);
    assert_dense_close(&e.to_dense(), &want.to_dense(), 1e-14);
}

#[test]
fn s1_16_high_spin_determinant_expand() {
    let csf: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Up, Su2Family::Up]);
    let e = expand(&csf, 3);
    let want: TtVector<f64> = tt_state(&[true, true, true], &[false, false, false]);
    assert_dense_close(&e.to_dense(), &want.to_dense(), 1e-14);
}

#[test]
fn s1_18_hand_checked_doublet_expansion_cg_magnitude_anchor() {
    let csf: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Up, Su2Family::Dn]);
    let e = expand(&csf, 1);
    let uud: TtVector<f64> = tt_state(&[true, true, false], &[false, false, true]);
    let udu: TtVector<f64> = tt_state(&[true, false, true], &[false, true, false]);
    let duu: TtVector<f64> = tt_state(&[false, true, true], &[true, false, false]);
    let c1 = (2.0_f64 / 3.0).sqrt();
    let c2 = (1.0_f64 / 6.0).sqrt();
    let want: Vec<f64> = uud
        .to_dense()
        .iter()
        .zip(udu.to_dense())
        .zip(duu.to_dense())
        .map(|((u, v), w)| c1 * u - c2 * v - c2 * w)
        .collect();
    assert_dense_close(&e.to_dense(), &want, 1e-14);
}

#[test]
fn s1_19_random_csf_unit_norm_at_every_admissible_two_sz() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9EA1_9EA1);
    for &(d, n, two_s) in &[(6usize, 4usize, 2usize), (7, 5, 1), (5, 4, 0)] {
        let x: Su2TtVector<f64> = random_csf(d, n, two_s, &mut rng);
        let mut ts = two_s as i64;
        while ts >= -(two_s as i64) {
            let e = expand(&x, ts);
            assert!(e.check());
            assert!(
                (e.norm() - 1.0).abs() < 1e-12,
                "d={d} N={n} 2S={two_s} 2Sz={ts}"
            );
            ts -= 2;
        }
    }
}

#[test]
fn s1_20_distinct_csfs_orthonormal_through_expansion() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0FED_C0DE);
    let (d, n, two_s) = (6, 4, 2);
    let csfs: Vec<Su2TtVector<f64>> = (0..6).map(|_| random_csf(d, n, two_s, &mut rng)).collect();
    for i in 0..csfs.len() {
        for j in i..csfs.len() {
            let s_su2 = su2_dot(&csfs[i], &csfs[j]);
            let ei = expand(&csfs[i], 0);
            let ej = expand(&csfs[j], 0);
            let s_ab = ab_dot(&ei, &ej);
            assert!((s_su2 - s_ab).abs() < 1e-12, "i={i} j={j}");
            assert!(
                s_su2.abs() < 1e-9 || (s_su2 - 1.0).abs() < 1e-9,
                "overlap should be (near) 0 or 1, got {s_su2} at i={i} j={j}"
            );
        }
    }
}
