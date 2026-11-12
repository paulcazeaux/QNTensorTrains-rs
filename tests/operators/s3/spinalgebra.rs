// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::s3::*;

#[test]
fn s1_a1_cg_generator_matches_shipped_closed_forms() {
    use qntt::su2::expand::{cg_dn_lower, cg_dn_raise, cg_up_lower, cg_up_raise};
    let mut spin = SpinAlgebra::new();
    let mut two_j = 0i32;
    while two_j <= 14 {
        let mut two_m = -two_j;
        while two_m <= two_j {
            let up_raise = spin.clebschgordan(two_j, two_m, 1, 1, two_j + 1, two_m + 1);
            assert!((up_raise - cg_up_raise(two_j as i64, two_m as i64)).abs() < 1e-14);
            let dn_raise = spin.clebschgordan(two_j, two_m, 1, -1, two_j + 1, two_m - 1);
            assert!((dn_raise - cg_dn_raise(two_j as i64, two_m as i64)).abs() < 1e-14);
            if two_j > 0 {
                let up_lower = spin.clebschgordan(two_j, two_m, 1, 1, two_j - 1, two_m + 1);
                assert!((up_lower - cg_up_lower(two_j as i64, two_m as i64)).abs() < 1e-14);
                let dn_lower = spin.clebschgordan(two_j, two_m, 1, -1, two_j - 1, two_m - 1);
                assert!((dn_lower - cg_dn_lower(two_j as i64, two_m as i64)).abs() < 1e-14);
            }
            two_m += 2;
        }
        two_j += 2;
    }
}

#[test]
fn s1_a2_cg_orthogonality_column_and_completeness() {
    let mut spin = SpinAlgebra::new();
    let pairs: [(i32, i32); 6] = [(1, 1), (2, 1), (12, 1), (12, 2), (7, 2), (19, 1)];
    for &(two_j1, two_j2) in &pairs {
        let mut two_bigj = (two_j1 - two_j2).abs();
        while two_bigj <= two_j1 + two_j2 {
            let mut two_bigm = -two_bigj;
            while two_bigm <= two_bigj {
                let mut sum = 0.0f64;
                let mut two_m1 = -two_j1;
                while two_m1 <= two_j1 {
                    let two_m2 = two_bigm - two_m1;
                    if two_m2.abs() <= two_j2 {
                        let cg =
                            spin.clebschgordan(two_j1, two_m1, two_j2, two_m2, two_bigj, two_bigm);
                        sum += cg * cg;
                    }
                    two_m1 += 2;
                }
                assert!(
                    (sum - 1.0).abs() < 1e-12,
                    "column-orthonormality failed: sum={sum}"
                );
                two_bigm += 2;
            }
            two_bigj += 2;
        }
        let mut two_m1 = -two_j1;
        while two_m1 <= two_j1 {
            let mut two_m2 = -two_j2;
            while two_m2 <= two_j2 {
                let two_bigm = two_m1 + two_m2;
                let mut sum = 0.0f64;
                let mut two_bigj = (two_j1 - two_j2).abs();
                while two_bigj <= two_j1 + two_j2 {
                    if two_bigm.abs() <= two_bigj {
                        let cg =
                            spin.clebschgordan(two_j1, two_m1, two_j2, two_m2, two_bigj, two_bigm);
                        sum += cg * cg;
                    }
                    two_bigj += 2;
                }
                assert!((sum - 1.0).abs() < 1e-12, "completeness failed: sum={sum}");
                two_m2 += 2;
            }
            two_m1 += 2;
        }
    }
}

#[test]
fn s1_cg_selection_symmetry_and_singlet() {
    let mut spin = SpinAlgebra::new();
    assert_eq!(spin.clebschgordan(2, 0, 2, 0, 2, 0), 0.0);
    assert_eq!(spin.clebschgordan(1, 1, 1, 1, 0, 0), 0.0);
    assert!(
        (spin.clebschgordan(1, 1, 1, -1, 0, 0) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-13
    );
    assert!(
        (spin.clebschgordan(1, -1, 1, 1, 0, 0) + std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-13
    );
    for &(two_j1, two_j2) in &[(1i32, 1i32), (2, 1), (2, 2), (3, 1)] {
        let mut two_bigj = (two_j1 - two_j2).abs();
        while two_bigj <= two_j1 + two_j2 {
            let mut two_m1 = -two_j1;
            while two_m1 <= two_j1 {
                let mut two_m2 = -two_j2;
                while two_m2 <= two_j2 {
                    let two_bigm = two_m1 + two_m2;
                    if two_bigm.abs() <= two_bigj {
                        let lhs =
                            spin.clebschgordan(two_j1, two_m1, two_j2, two_m2, two_bigj, two_bigm);
                        let rhs =
                            spin.clebschgordan(two_j2, two_m2, two_j1, two_m1, two_bigj, two_bigm);
                        let sign = if ((two_j1 + two_j2 - two_bigj) / 2) % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                        assert!((lhs - sign * rhs).abs() < 1e-13, "CG swap symmetry failed");
                    }
                    two_m2 += 2;
                }
                two_m1 += 2;
            }
            two_bigj += 2;
        }
    }
}

#[test]
fn s1_a3_recouple_trivial_limits() {
    let mut spin = SpinAlgebra::new();
    let mut two_j = 0i32;
    while two_j <= 8 {
        for &two_k in &[0i32, 2] {
            let mut two_bigj = (two_j - two_k).abs();
            while two_bigj <= two_j + two_k {
                let f = spin.recouple(two_j, two_j, 0, two_k, two_k, 0, two_bigj, two_bigj, 0);
                assert!(
                    (f - 1.0).abs() < 1e-12,
                    "identity transport recouple != 1: {f}"
                );
                two_bigj += 2;
            }
        }
        two_j += 2;
    }
    let mut two_j = 0i32;
    while two_j <= 6 {
        for &two_jp in &[two_j - 1, two_j + 1] {
            if two_jp < 0 {
                continue;
            }
            let f = spin.recouple(two_j, two_jp, 1, 0, 0, 0, two_j, two_jp, 1);
            assert!(
                (f - 1.0).abs() < 1e-12,
                "k=0 no-operator recouple != 1: {f}"
            );
        }
        two_j += 2;
    }
}

#[test]
fn s1_a4_site_channel_ops_single_creation() {
    let mut spin = SpinAlgebra::new();
    let ids = spin.site_channel_ops(&[true]);
    assert_eq!(ids.len(), 1);
    let op = spin.siteop(ids[0]).clone();
    assert_eq!(op.two_k, 1);
    assert!(op.odd);
    assert_eq!(op.flux, 1);
    assert!((op.red[0][1].abs() - 1.0).abs() < 1e-13);
    assert!(op.red[1][2].abs() > 0.5);
}

#[test]
fn s1_a5_composite_on_site_adag_a_tilde() {
    let mut spin = SpinAlgebra::new();
    let ids = spin.site_channel_ops(&[true, false]);
    let mut two_ks: Vec<i32> = ids.iter().map(|&id| spin.siteop(id).two_k).collect();
    two_ks.sort();
    assert_eq!(two_ks, vec![0, 2]);
    let o0 = spin
        .siteop(*ids.iter().find(|&&id| spin.siteop(id).two_k == 0).unwrap())
        .clone();
    let o2 = spin
        .siteop(*ids.iter().find(|&&id| spin.siteop(id).two_k == 2).unwrap())
        .clone();
    assert!(!o0.odd && !o2.odd && o0.flux == 0);
    assert!(o0.red[1][1].abs() > 0.1 && o0.red[2][2].abs() > 0.1 && o0.red[0][0].abs() < 1e-13);
    assert!(o2.red[1][1].abs() > 0.1 && o2.red[2][2].abs() < 1e-13 && o2.red[0][0].abs() < 1e-13);
}

#[test]
fn s1_a6_on_site_pair_channel() {
    let mut spin = SpinAlgebra::new();
    let ids = spin.site_channel_ops(&[true, true]);
    assert_eq!(ids.len(), 1);
    let op = spin.siteop(ids[0]).clone();
    assert_eq!(op.two_k, 0);
    assert!(!op.odd);
    assert_eq!(op.flux, 2);
    assert!(op.red[0][2].abs() > 0.5);
    assert!(op.red[1][1].abs() < 1e-13);
}

#[test]
fn s1_a7_channel_fit_gate_pattern_classes() {
    let mut spin = SpinAlgebra::new();
    let mut patterns: Vec<Vec<(usize, bool)>> = vec![
        vec![(0, true), (1, false)],
        vec![(1, true), (0, false)],
        vec![(0, true), (0, false)],
    ];
    for a in 0..4usize {
        for b in 0..4usize {
            for c in 0..4usize {
                for d in 0..4usize {
                    let mut u = [a, b, c, d];
                    u.sort_unstable();
                    if u.windows(2).any(|w| w[0] == w[1]) {
                        continue;
                    }
                    patterns.push(vec![(a, true), (b, true), (c, false), (d, false)]);
                }
            }
        }
    }
    for &(i, k, l, j) in &[
        (0usize, 0usize, 1usize, 2usize),
        (0, 1, 0, 2),
        (0, 1, 2, 0),
        (1, 0, 0, 2),
        (1, 0, 2, 0),
        (1, 2, 0, 0),
        (0, 1, 1, 2),
        (1, 0, 1, 2),
        (1, 1, 0, 2),
        (0, 1, 2, 1),
        (1, 0, 2, 1),
        (0, 2, 1, 1),
        (0, 1, 2, 2),
        (2, 0, 1, 2),
        (2, 2, 0, 1),
        (0, 2, 1, 2),
        (2, 0, 2, 1),
        (2, 1, 2, 0),
    ] {
        patterns.push(vec![(i, true), (k, true), (l, false), (j, false)]);
    }
    for &(i, k, l, j) in &[
        (0usize, 0usize, 1usize, 1usize),
        (1, 1, 0, 0),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (0, 1, 0, 1),
        (1, 0, 1, 0),
        (0, 0, 0, 1),
        (0, 0, 1, 0),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        (0, 1, 1, 1),
        (1, 0, 1, 1),
        (1, 1, 0, 1),
        (1, 1, 1, 0),
    ] {
        patterns.push(vec![(i, true), (k, true), (l, false), (j, false)]);
    }
    patterns.push(vec![(0, true), (0, true), (0, false), (0, false)]);

    for tk in &patterns {
        let chans = spin.term_channels(tk);
        if chans.is_empty() {
            let mut sites: Vec<usize> = tk.iter().map(|t| t.0).collect();
            sites.sort_unstable();
            sites.dedup();
            let remapped: Vec<(usize, bool)> = tk
                .iter()
                .map(|&(s, c)| (sites.iter().position(|&x| x == s).unwrap(), c))
                .collect();
            assert!(dense_term_norm(sites.len(), &remapped) < 1e-13);
        }
    }
}

#[test]
#[should_panic(expected = "tokens must have length 2, 4, or 6")]
fn dense_term_refuses_bad_token_length() {
    let _ = dense_term(1, &[(0, true)]);
}

#[test]
fn s3_rme_absolute_sign_pins() {
    use qntt::su2::spinalgebra::{tensor_a, tensor_cdag};
    let mut sa = SpinAlgebra::new();
    let rc = sa.reduce_siteop(&tensor_cdag());
    let ra = sa.reduce_siteop(&tensor_a());
    let sq2 = std::f64::consts::SQRT_2;
    let tol = 1e-14;

    assert!(
        (rc[0][1] - 1.0).abs() < tol,
        "⟨1e‖c†‖0⟩ = +1, got {}",
        rc[0][1]
    );
    assert!(
        (rc[1][2] + sq2).abs() < tol,
        "⟨2e‖c†‖1e⟩ = −√2, got {}",
        rc[1][2]
    );
    assert!(
        (ra[1][0] - sq2).abs() < tol,
        "⟨0‖ã‖1e⟩ = +√2, got {}",
        ra[1][0]
    );
    assert!(
        (ra[2][1] - 1.0).abs() < tol,
        "⟨1e‖ã‖2e⟩ = +1, got {}",
        ra[2][1]
    );

    for (name, red, nz) in [
        ("c†", &rc, [(0usize, 1usize), (1, 2)]),
        ("ã", &ra, [(1, 0), (2, 1)]),
    ] {
        for (ox, row) in red.iter().enumerate() {
            for (oy, &val) in row.iter().enumerate() {
                if nz.contains(&(ox, oy)) {
                    continue;
                }
                assert_eq!(
                    val, 0.0,
                    "{name} red[{ox}][{oy}] must be an exact structural zero"
                );
            }
        }
    }
}
