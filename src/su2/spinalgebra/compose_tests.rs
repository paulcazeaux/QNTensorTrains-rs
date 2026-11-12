// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use super::compose::ComposeTable;
use super::siteops::{OCC_SPIN, OCC_STATES};
use super::{Mat4, SpinAlgebra, TensorOp};

fn tri_range(a: i32, b: i32) -> impl Iterator<Item = i32> {
    ((a - b).abs()..=a + b).step_by(2)
}

fn tri(a: i32, b: i32, c: i32) -> bool {
    (a - b).abs() <= c && c <= a + b && (a + b + c) % 2 == 0
}

fn projections(two_j: i32) -> impl Iterator<Item = i32> {
    (-two_j..=two_j).step_by(2)
}

const SITE_OPS: [(&str, usize, usize, i32); 10] = [
    ("E00", 0, 0, 0),
    ("N11", 1, 1, 0),
    ("E22", 2, 2, 0),
    ("P20", 0, 2, 0),
    ("P02", 2, 0, 0),
    ("S11", 1, 1, 2),
    ("C10", 0, 1, 1),
    ("C21", 1, 2, 1),
    ("A01", 1, 0, 1),
    ("A12", 2, 1, 1),
];

fn site_op(occ_in: usize, occ_out: usize, two_r: i32) -> Option<&'static str> {
    SITE_OPS
        .iter()
        .find(|o| o.1 == occ_in && o.2 == occ_out && o.3 == two_r)
        .map(|o| o.0)
}

#[allow(clippy::too_many_arguments)]
fn s_proj(sp: &mut SpinAlgebra, sx: i32, sm: i32, sy: i32, rb: i32, ra: i32, kap: i32) -> f64 {
    let mut tot = 0.0;
    for mua in projections(ra) {
        for mub in projections(rb) {
            let mu = mua + mub;
            if mu.abs() > kap {
                continue;
            }
            let c1 = sp.clebschgordan(rb, mub, ra, mua, kap, mu);
            if c1 == 0.0 {
                continue;
            }
            for mx in projections(sx) {
                for my in projections(sy) {
                    let c2 = sp.clebschgordan(sx, mx, kap, mu, sy, my);
                    if c2 == 0.0 {
                        continue;
                    }
                    for mm in projections(sm) {
                        tot += c1
                            * c2
                            * sp.clebschgordan(sm, mm, ra, mua, sy, my)
                            * sp.clebschgordan(sx, mx, rb, mub, sm, mm);
                    }
                }
            }
        }
    }
    tot / (sy + 1) as f64
}

fn unit_tensor(sp: &mut SpinAlgebra, occ_in: usize, occ_out: usize, two_r: i32) -> TensorOp {
    let (sx, sy) = (OCC_SPIN[occ_in], OCC_SPIN[occ_out]);
    let mut comps = vec![[[0.0f64; 4]; 4]; (two_r + 1) as usize];
    for (imu, mu) in (0..).zip((-two_r..=two_r).rev().step_by(2)) {
        for (ix, mx) in (0..).zip((-sx..=sx).rev().step_by(2)) {
            for (iy, my) in (0..).zip((-sy..=sy).rev().step_by(2)) {
                let v = sp.clebschgordan(sx, mx, two_r, mu, sy, my);
                comps[imu][OCC_STATES[occ_out][iy]][OCC_STATES[occ_in][ix]] = v;
            }
        }
    }
    TensorOp {
        two_k: two_r,
        flux: occ_out as i32 - occ_in as i32,
        odd: (occ_in + occ_out) % 2 == 1,
        components: comps,
    }
}

fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut c = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for k in 0..4 {
            for j in 0..4 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn composable() -> Vec<(usize, usize, i32)> {
    let mut out = Vec::new();
    for (ia, a) in SITE_OPS.iter().enumerate() {
        for (ib, b) in SITE_OPS.iter().enumerate() {
            if a.1 != b.2 {
                continue;
            }
            for kap in tri_range(a.3, b.3) {
                if tri(OCC_SPIN[b.1], kap, OCC_SPIN[a.2]) {
                    out.push((ia, ib, kap));
                }
            }
        }
    }
    out
}

fn g5a_residual(
    sp: &mut SpinAlgebra,
    max_two: i32,
    g: impl Fn(&mut SpinAlgebra, [i32; 9]) -> f64,
) -> (f64, usize) {
    let (mut worst, mut count) = (0.0f64, 0usize);
    for kb in 0..=max_two {
        for rb in 0..=2 {
            for kb2 in tri_range(kb, rb).filter(|&x| x <= max_two) {
                for ka in 0..=max_two {
                    for ra in 0..=2 {
                        for ka2 in tri_range(ka, ra).filter(|&x| x <= max_two) {
                            for kc2 in tri_range(kb2, ka2).filter(|&x| x <= max_two) {
                                let mut gs: Vec<(i32, i32, f64)> = Vec::new();
                                for kc in tri_range(kb, ka) {
                                    for kap in tri_range(rb, ra) {
                                        if !tri(kc, kap, kc2) {
                                            continue;
                                        }
                                        let v = g(sp, [kb, kb2, rb, ka, ka2, ra, kc, kc2, kap]);
                                        gs.push((kc, kap, v));
                                    }
                                }
                                for q2 in projections(kc2) {
                                    for qb in projections(kb) {
                                        for mb in projections(rb) {
                                            for qa in projections(ka) {
                                                let ma = q2 - qb - mb - qa;
                                                if ma.abs() > ra || (ma - ra) % 2 != 0 {
                                                    continue;
                                                }
                                                let (qb2, qa2) = (qb + mb, qa + ma);
                                                let lhs = sp
                                                    .clebschgordan(kb2, qb2, ka2, qa2, kc2, q2)
                                                    * sp.clebschgordan(ka, qa, ra, ma, ka2, qa2)
                                                    * sp.clebschgordan(kb, qb, rb, mb, kb2, qb2);
                                                let (qc, mu) = (qb + qa, mb + ma);
                                                let mut rhs = 0.0;
                                                for &(kc, kap, gv) in &gs {
                                                    rhs += gv
                                                        * sp.clebschgordan(kb, qb, ka, qa, kc, qc)
                                                        * sp.clebschgordan(rb, mb, ra, ma, kap, mu)
                                                        * sp.clebschgordan(
                                                            kc, qc, kap, mu, kc2, q2,
                                                        );
                                                }
                                                worst = worst.max((lhs - rhs).abs());
                                                count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (worst, count)
}

#[test]
fn g5a_bond_recoupling_is_recouple_with_b_in_the_x_slots() {
    let mut sp = SpinAlgebra::new();
    let (worst, count) = g5a_residual(&mut sp, 8, |sp, a| {
        sp.recouple(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
    });
    println!("G5a (B in x): {count} projection tuples, max |LHS − RHS| = {worst:e}");
    assert!(count > 100_000, "anti-vacuity: only {count} tuples");
    assert!(worst <= 1e-13, "G5a: max residual {worst:e} > 1e-13");
}

#[test]
fn g5a_rejects_a_in_the_x_slots_under_b_first_couplings() {
    let mut sp = SpinAlgebra::new();
    let (worst, count) = g5a_residual(&mut sp, 4, |sp, a| {
        sp.recouple(a[3], a[4], a[5], a[0], a[1], a[2], a[6], a[7], a[8])
    });
    println!("G5a anti-oracle (A in x): {count} tuples, max |LHS − RHS| = {worst:e}");
    assert!(worst > 0.1, "the mixed convention passed G5a ({worst:e})");
}

#[test]
fn compose_table_values_are_the_b_first_projection() {
    fn plain_data<T: Send + Sync + Clone + 'static>(_: &T) {}
    let table = ComposeTable::new();
    plain_data(&table);
    let mut sp = SpinAlgebra::new();
    let mut nonzero = 0usize;
    for oib in 0..3usize {
        for om in 0..3usize {
            for ooa in 0..3usize {
                for rb in 0..=2i32 {
                    for ra in 0..=2i32 {
                        for kap in 0..=4i32 {
                            let got =
                                table.element(oib, om, ooa, rb as usize, ra as usize, kap as usize);
                            let exists = site_op(oib, om, rb).is_some()
                                && site_op(om, ooa, ra).is_some()
                                && tri(ra, rb, kap)
                                && tri(OCC_SPIN[oib], kap, OCC_SPIN[ooa]);
                            let key = format!("({oib},{om},{ooa}; 2r_B={rb}, 2r_A={ra}, 2κ={kap})");
                            if !exists {
                                assert_eq!(got, 0.0, "{key}: no composition, got {got}");
                                continue;
                            }
                            let want = s_proj(
                                &mut sp,
                                OCC_SPIN[oib],
                                OCC_SPIN[om],
                                OCC_SPIN[ooa],
                                rb,
                                ra,
                                kap,
                            );
                            assert!(
                                (got - want).abs() <= 1e-14,
                                "{key}: table {got} vs projection {want}"
                            );
                            assert!(got.abs() >= 0.5 - 1e-14, "{key}: |S| = {got} < ½");
                            nonzero += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        nonzero, 37,
        "37 composed elements over the 34 composable pairs"
    );
    assert_eq!(
        composable().len(),
        37,
        "the site-op enumeration agrees with the table's support"
    );
    let (h, r3, r23) = (0.5, 3f64.sqrt() / 2.0, (2.0f64 / 3.0).sqrt());
    let named: [(&str, &str, i32, f64); 8] = [
        ("S11", "S11", 0, -1.0 / 3f64.sqrt()),
        ("S11", "S11", 2, r23),
        ("C10", "A01", 0, -h),
        ("C10", "A01", 2, r3),
        ("A12", "C21", 0, -h),
        ("A12", "C21", 2, r3),
        ("C21", "C10", 0, 1.0),
        ("A01", "A12", 0, 1.0),
    ];
    for (a, b, kap, want) in named {
        let oa = SITE_OPS.iter().find(|o| o.0 == a).unwrap();
        let ob = SITE_OPS.iter().find(|o| o.0 == b).unwrap();
        let got = table.element(ob.1, ob.2, oa.2, ob.3 as usize, oa.3 as usize, kap as usize);
        assert!(
            (got - want).abs() <= 1e-15,
            "{a}∘{b} at 2κ={kap}: {got} vs closed form {want}"
        );
    }
    for (a, b, kap) in [("N11", "N11", 0), ("P20", "P02", 0)] {
        let oa = SITE_OPS.iter().find(|o| o.0 == a).unwrap();
        let ob = SITE_OPS.iter().find(|o| o.0 == b).unwrap();
        let got = table.element(ob.1, ob.2, oa.2, ob.3 as usize, oa.3 as usize, kap as usize);
        assert!((got - 1.0).abs() <= 1e-15, "{a}∘{b}: {got} vs 1");
    }
}

#[test]
fn g5b_unit_tensor_products_expand_with_the_table() {
    let table = ComposeTable::new();
    let mut sp = SpinAlgebra::new();
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for a in SITE_OPS {
        for b in SITE_OPS {
            if a.1 != b.2 {
                continue;
            }
            let ta = unit_tensor(&mut sp, a.1, a.2, a.3);
            let tb = unit_tensor(&mut sp, b.1, b.2, b.3);
            let (sx, sy) = (OCC_SPIN[b.1], OCC_SPIN[a.2]);
            for (ia, mua) in (0..).zip((-a.3..=a.3).rev().step_by(2)) {
                for (ib, mub) in (0..).zip((-b.3..=b.3).rev().step_by(2)) {
                    let prod = mat4_mul(&ta.components[ia], &tb.components[ib]);
                    let mu = mua + mub;
                    let mut want = [[0.0f64; 4]; 4];
                    for kap in tri_range(a.3, b.3) {
                        if mu.abs() > kap {
                            continue;
                        }
                        let s =
                            table.element(b.1, b.2, a.2, b.3 as usize, a.3 as usize, kap as usize);
                        let c = sp.clebschgordan(b.3, mub, a.3, mua, kap, mu);
                        for (ix, mx) in (0..).zip((-sx..=sx).rev().step_by(2)) {
                            for (iy, my) in (0..).zip((-sy..=sy).rev().step_by(2)) {
                                want[OCC_STATES[a.2][iy]][OCC_STATES[b.1][ix]] +=
                                    c * s * sp.clebschgordan(sx, mx, kap, mu, sy, my);
                            }
                        }
                    }
                    for i in 0..4 {
                        for j in 0..4 {
                            worst = worst.max((prod[i][j] - want[i][j]).abs());
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    println!("G5b: {checked} (pair, μ_A, μ_B) products, max residual {worst:e}");
    assert!(checked > 50, "anti-vacuity: {checked}");
    assert!(worst <= 1e-14, "G5b: max residual {worst:e}");
}

#[test]
fn compose_table_is_couple_times_the_coupling_order_phase() {
    let table = ComposeTable::new();
    let mut sp = SpinAlgebra::new();
    let mut flipped = 0usize;
    for (ia, ib, kap) in composable() {
        let (a, b) = (SITE_OPS[ia], SITE_OPS[ib]);
        let ta = unit_tensor(&mut sp, a.1, a.2, a.3);
        let tb = unit_tensor(&mut sp, b.1, b.2, b.3);
        let coupled = sp.couple(&ta, &tb, kap);
        let red = sp.reduce_siteop(&coupled);
        let s_couple = red[b.1][a.2];
        let phase = if ((a.3 + b.3 - kap) / 2) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let s = table.element(b.1, b.2, a.2, b.3 as usize, a.3 as usize, kap as usize);
        assert!(
            (s - phase * s_couple).abs() <= 1e-14,
            "{}∘{} at 2κ={kap}: table {s} vs phase·couple {}",
            a.0,
            b.0,
            phase * s_couple
        );
        flipped += usize::from(phase < 0.0 && s_couple != 0.0);
    }
    assert!(flipped > 0, "anti-vacuity: the phase never acts");
}

fn g5c_residual(
    sp: &mut SpinAlgebra,
    max_two: i32,
    s_of: impl Fn(&mut SpinAlgebra, i32, i32, i32, i32, i32, i32) -> f64,
) -> (f64, usize) {
    fn f(sp: &mut SpinAlgebra, a: [i32; 9]) -> f64 {
        sp.recouple(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8])
    }
    let u = |sp: &mut SpinAlgebra, j: i32, kb: i32, ka: i32, j1: i32, kc: i32, jj: i32| {
        sp.recouple(j, j, 0, kb, kc, ka, j1, jj, ka)
    };
    let (mut worst, mut cases) = (0.0f64, 0usize);
    for sx in 0..=1 {
        for sm in 0..=1 {
            for sy in 0..=1 {
                for rb in tri_range(sx, sm).filter(|&r| r <= 2) {
                    for ra in tri_range(sm, sy).filter(|&r| r <= 2) {
                        for j in 0..=max_two {
                            for j2 in tri_range(j, sx).filter(|&x| x <= max_two) {
                                for kb in 0..=max_two {
                                    for kb2 in tri_range(kb, rb).filter(|&x| x <= max_two) {
                                        for ka in 0..=max_two {
                                            for ka2 in tri_range(ka, ra).filter(|&x| x <= max_two) {
                                                for kc in tri_range(kb, ka) {
                                                    for kc2 in tri_range(kb2, ka2) {
                                                        if !tri_range(ra, rb)
                                                            .any(|kap| tri(kc, kap, kc2))
                                                        {
                                                            continue;
                                                        }
                                                        for jj in tri_range(j, kc) {
                                                            for jj2 in tri_range(j2, kc2) {
                                                                if !tri(jj, sy, jj2) {
                                                                    continue;
                                                                }
                                                                let mut lhs = 0.0;
                                                                for j1 in tri_range(j, kb)
                                                                    .filter(|&x| tri(x, ka, jj))
                                                                {
                                                                    let u0 = u(
                                                                        sp, j, kb, ka, j1, kc, jj,
                                                                    );
                                                                    if u0 == 0.0 {
                                                                        continue;
                                                                    }
                                                                    for j12 in tri_range(j2, kb2)
                                                                        .filter(|&x| {
                                                                            tri(x, ka2, jj2)
                                                                        })
                                                                    {
                                                                        let u1 = u(
                                                                            sp, j2, kb2, ka2, j12,
                                                                            kc2, jj2,
                                                                        );
                                                                        let fb = f(
                                                                            sp,
                                                                            [
                                                                                j, j2, sx, kb, kb2,
                                                                                rb, j1, j12, sm,
                                                                            ],
                                                                        );
                                                                        let fa = f(
                                                                            sp,
                                                                            [
                                                                                j1, j12, sm, ka,
                                                                                ka2, ra, jj, jj2,
                                                                                sy,
                                                                            ],
                                                                        );
                                                                        lhs += u1 * fb * fa * u0;
                                                                    }
                                                                }
                                                                let mut rhs = 0.0;
                                                                for kap in tri_range(ra, rb) {
                                                                    if !tri(kc, kap, kc2) {
                                                                        continue;
                                                                    }
                                                                    let g = f(
                                                                        sp,
                                                                        [
                                                                            kb, kb2, rb, ka, ka2,
                                                                            ra, kc, kc2, kap,
                                                                        ],
                                                                    );
                                                                    let s = s_of(
                                                                        sp, sx, sm, sy, rb, ra, kap,
                                                                    );
                                                                    let fc = f(
                                                                        sp,
                                                                        [
                                                                            j, j2, sx, kc, kc2,
                                                                            kap, jj, jj2, sy,
                                                                        ],
                                                                    );
                                                                    rhs += g * s * fc;
                                                                }
                                                                worst =
                                                                    worst.max((lhs - rhs).abs());
                                                                cases += 1;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (worst, cases)
}

#[test]
fn g5c_two_applies_equal_one_apply_of_the_product() {
    let mut sp = SpinAlgebra::new();
    let (worst, cases) = g5c_residual(&mut sp, 6, s_proj);
    println!("G5c: {cases} cases, max |LHS − RHS| = {worst:e}");
    assert!(cases > 100_000, "anti-vacuity: only {cases} cases");
    assert!(worst <= 1e-13, "G5c: max residual {worst:e} > 1e-13");
}

#[test]
fn g5c_rejects_the_a_first_element_without_its_phase() {
    let mut sp = SpinAlgebra::new();
    let (worst, cases) = g5c_residual(&mut sp, 4, |sp, sx, sm, sy, rb, ra, kap| {
        let phase = if ((ra + rb - kap) / 2) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        phase * s_proj(sp, sx, sm, sy, rb, ra, kap)
    });
    println!("G5c anti-oracle (couple without phase): {cases} cases, max |LHS − RHS| = {worst:e}");
    assert!(
        worst > 0.1,
        "the unphased A-first element passed G5c ({worst:e})"
    );
}
