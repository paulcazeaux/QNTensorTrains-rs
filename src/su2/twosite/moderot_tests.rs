// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::collections::{HashMap, HashSet};
use std::f64::consts::{FRAC_PI_2, PI};

use faer::Mat;
use rand::RngExt;

use super::{
    ClassGenerator, ThetaMetric, apply_mode_rotation, apply_mode_rotation_cs,
    apply_mode_rotation_serial, apply_mode_swap, class_gate_cs, class_generators, gate_tables,
    pool_weight, rotated_bond_svals, rotated_bond_wvals_tiled, search_angle,
};
use crate::core::factor::svals_qr;
use crate::core::twosite as ab;
use crate::core::twosite::{ModeRotOpts, minimize_angle, noise_rng};
use crate::entropy::{BondFunctional, bond_functional};
use crate::pointgroup::{Irrep, PointGroup, PointGroupKind};
use crate::qn::Pooled;
use crate::rotation::{expm_antisym, givens_cs};
use crate::scalar::C64;
use crate::su2::core::Family;
use crate::su2::expand::expand;
use crate::su2::orbital::g_pq;
use crate::su2::qn::{Su2Sector, Su2Space};
use crate::su2::twosite::{
    PAR_MIN_MID_SECTORS, Su2ContractedCores, contract, factor_qc, factor_svd_noise, mpred, msucc,
};
use crate::su2::vector::{Su2TtVector, tt_csf};
use crate::test_util::{dense_diff_norm, dense_norm, randomized_su2_state, rel_err, seeded};
use crate::vector::TtVector;

fn lih_pg() -> PointGroup {
    PointGroup::from_orbsym(PointGroupKind::C2v, &[1, 1, 1, 2, 3, 1]).expect("valid orbsym table")
}

fn bpair4_pg() -> PointGroup {
    PointGroup::from_orbsym(PointGroupKind::C2v, &[1, 2, 2, 1]).expect("valid orbsym table")
}

fn bpair5_pg() -> PointGroup {
    PointGroup::from_orbsym(PointGroupKind::C2v, &[2, 3, 3, 1, 1]).expect("valid orbsym table")
}

fn fixtures() -> Vec<(&'static str, Su2Space, Vec<usize>)> {
    vec![
        ("h4 (4,4,0)", Su2Space::new(4, 4, 0), (0..3).collect()),
        ("h4 (4,4,2)", Su2Space::new(4, 4, 2), (0..3).collect()),
        ("lih (6,3,1)", Su2Space::new(6, 3, 1), (0..5).collect()),
        (
            "lih C2v (6,3,1)",
            Su2Space::with_point_group(6, 3, 1, lih_pg(), Irrep::IDENTITY),
            vec![0, 1],
        ),
        (
            "[1,2,2,1] (4,4,2,A1)",
            Su2Space::with_point_group(4, 4, 2, bpair4_pg(), Irrep::IDENTITY),
            vec![1],
        ),
        (
            "[1,2,2,1] (4,3,1,Γ2)",
            Su2Space::with_point_group(4, 3, 1, bpair4_pg(), Irrep::new(2)),
            vec![1],
        ),
        (
            "[2,3,3,1,1] (5,3,1,Γ1)",
            Su2Space::with_point_group(5, 3, 1, bpair5_pg(), Irrep::new(1)),
            vec![1],
        ),
    ]
}

fn members(space: &Su2Space) -> Vec<i64> {
    let ts = space.two_s as i64;
    (-ts..=ts).step_by(2).collect()
}

#[derive(Clone, Copy, Debug)]
enum Arm {
    Theta(f64),
    Cs(f64),
}

fn su2_arm(tc: &Su2ContractedCores<f64>, arm: Arm) -> Su2ContractedCores<f64> {
    match arm {
        Arm::Theta(t) => apply_mode_rotation(tc, t),
        Arm::Cs(t) => {
            let (c, s) = givens_cs(t);
            apply_mode_rotation_cs(tc, c, s)
        }
    }
}

fn ab_arm(tc: &ab::ContractedSparseCores<f64>, arm: Arm) -> ab::ContractedSparseCores<f64> {
    match arm {
        Arm::Theta(t) => ab::apply_mode_rotation(tc, t),
        Arm::Cs(t) => {
            let (c, s) = givens_cs(t);
            ab::apply_mode_rotation_cs(tc, c, s)
        }
    }
}

fn su2_resplit(x: &Su2TtVector<f64>, tc: &Su2ContractedCores<f64>) -> Su2TtVector<f64> {
    let (a, b) = factor_qc(tc);
    let mut y = x.clone();
    y.set_cores(a, b);
    y.ortho = None;
    y
}

fn su2_gated(x: &Su2TtVector<f64>, k: usize, arm: Arm) -> Su2TtVector<f64> {
    su2_resplit(x, &su2_arm(&contract(x.core(k), x.core(k + 1)), arm))
}

fn ab_gated(xa: &TtVector<f64>, k: usize, arm: Arm) -> TtVector<f64> {
    let tc = ab_arm(&ab::contract(xa.core(k), xa.core(k + 1)), arm);
    let (a, b) = ab::factor_qc(&tc);
    let mut y = xa.clone();
    y.set_cores(a, b);
    y
}

fn entries(tc: &Su2ContractedCores<f64>) -> Vec<f64> {
    let mut v = Vec::new();
    for &m in &tc.mid_qn {
        if let Some(b) = tc.block(m) {
            for j in 0..b.ncols() {
                for i in 0..b.nrows() {
                    v.push(b[(i, j)]);
                }
            }
        }
    }
    v
}

fn assert_same_shape(a: &Su2ContractedCores<f64>, b: &Su2ContractedCores<f64>, ctx: &str) {
    assert_eq!(a.k, b.k, "{ctx}: site");
    assert_eq!(a.mid_qn, b.mid_qn, "{ctx}: mid sectors");
    assert!(a.row_ranks == b.row_ranks, "{ctx}: row ranks");
    assert!(a.col_ranks == b.col_ranks, "{ctx}: col ranks");
    for &m in &a.mid_qn {
        let sa = a.block(m).map(|x| (x.nrows(), x.ncols()));
        let sb = b.block(m).map(|x| (x.nrows(), x.ncols()));
        assert_eq!(sa, sb, "{ctx}: block {m:?} presence/shape");
    }
}

fn assert_bitwise(a: &Su2ContractedCores<f64>, b: &Su2ContractedCores<f64>, ctx: &str) {
    assert_same_shape(a, b, ctx);
    let (ea, eb) = (entries(a), entries(b));
    assert!(
        ea.iter().zip(&eb).all(|(x, y)| x.to_bits() == y.to_bits()),
        "{ctx}: not bitwise identical"
    );
}

fn max_diff(a: &Su2ContractedCores<f64>, b: &Su2ContractedCores<f64>) -> f64 {
    entries(a)
        .iter()
        .zip(&entries(b))
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn scale(tc: &Su2ContractedCores<f64>) -> f64 {
    entries(tc).iter().fold(1.0f64, |m, v| m.max(v.abs()))
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    m: Su2Sector,
    l: Su2Sector,
    r: Su2Sector,
    ro: usize,
    co: usize,
    rl: usize,
    rr: usize,
}

fn tile_at(tc: &Su2ContractedCores<f64>, m: Su2Sector, l: Su2Sector, r: Su2Sector) -> Tile {
    let (sp, k) = (&tc.space, tc.k);
    let mut ro = 0;
    let mut found_l = false;
    for (_, l2) in mpred(sp, k, m) {
        if l2 == l {
            found_l = true;
            break;
        }
        ro += tc.row_ranks[l2];
    }
    let mut co = 0;
    let mut found_r = false;
    for (_, r2) in msucc(sp, k, m) {
        if r2 == r {
            found_r = true;
            break;
        }
        co += tc.col_ranks[r2];
    }
    assert!(found_l && found_r, "tile ({l:?}, {r:?}) not in block {m:?}");
    Tile {
        m,
        l,
        r,
        ro,
        co,
        rl: tc.row_ranks[l],
        rr: tc.col_ranks[r],
    }
}

fn tiles(tc: &Su2ContractedCores<f64>) -> Vec<Tile> {
    let mut out = Vec::new();
    for &m in &tc.mid_qn {
        if tc.block(m).is_none() {
            continue;
        }
        for (_, l) in mpred(&tc.space, tc.k, m) {
            for (_, r) in msucc(&tc.space, tc.k, m) {
                let t = tile_at(tc, m, l, r);
                if t.rl > 0 && t.rr > 0 {
                    out.push(t);
                }
            }
        }
    }
    out
}

fn expected_classes(space: &Su2Space, k: usize) -> HashMap<(Su2Sector, Su2Sector), Vec<Su2Sector>> {
    let mut out: HashMap<(Su2Sector, Su2Sector), Vec<Su2Sector>> = HashMap::new();
    for m in space.bond_sectors(k + 1) {
        if !space.contains(k + 1, m) {
            continue;
        }
        for (_, l) in mpred(space, k, m) {
            for (_, r) in msucc(space, k, m) {
                out.entry((l, r)).or_default().push(m);
            }
        }
    }
    out
}

fn fam_k(space: &Su2Space, k: usize, l: Su2Sector, m: Su2Sector) -> Family {
    let g = space.pg.irrep_of(k);
    let hits: Vec<Family> = Family::ALL
        .into_iter()
        .filter(|f| f.apply(l, g) == Some(m))
        .collect();
    assert_eq!(hits.len(), 1, "path {l:?} → {m:?}: family not unique");
    hits[0]
}

fn apply_table(
    tc: &Su2ContractedCores<f64>,
    classes: &[ClassGenerator],
    mats: &[Vec<f64>],
) -> Su2ContractedCores<f64> {
    let mut out = tc.zeros_like();
    for (cl, a) in classes.iter().zip(mats) {
        let n = cl.mids.len();
        if tc.row_ranks[cl.l] == 0 || tc.col_ranks[cl.r] == 0 {
            continue;
        }
        for i in 0..n {
            let ti = tile_at(tc, cl.mids[i], cl.l, cl.r);
            for j in 0..n {
                let coef = a[i * n + j];
                if coef == 0.0 {
                    continue;
                }
                let tj = tile_at(tc, cl.mids[j], cl.l, cl.r);
                let src = tc.block(tj.m).expect("source block present");
                let mut dst = out.block_mut(ti.m).expect("destination block present");
                for c in 0..ti.rr {
                    for r in 0..ti.rl {
                        dst[(ti.ro + r, ti.co + c)] += coef * src[(tj.ro + r, tj.co + c)];
                    }
                }
            }
        }
    }
    out
}

fn mm(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..n {
                c[i * n + j] += a[i * n + l] * b[l * n + j];
            }
        }
    }
    c
}

fn eye(n: usize) -> Vec<f64> {
    let mut e = vec![0.0f64; n * n];
    for i in 0..n {
        e[i * n + i] = 1.0;
    }
    e
}

fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn orth_defect(g: &[f64], n: usize) -> f64 {
    let mut gt = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            gt[i * n + j] = g[j * n + i];
        }
    }
    max_abs(&sub(&mm(&gt, g, n), &eye(n)))
}

fn scaled(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|v| s * v).collect()
}

fn t2_check(label: &str, x: &Su2TtVector<f64>, ks: &[usize], arms: &[Arm], pi_moves: bool) -> f64 {
    let mut worst = 0.0f64;
    let mut moved_pi = 0.0f64;
    let has_pi = arms.iter().any(|a| matches!(a, Arm::Theta(t) if *t == PI));
    for &k in ks {
        for &arm in arms {
            let y = su2_gated(x, k, arm);
            for (idx, tsz) in members(&x.space).into_iter().enumerate() {
                let xa = expand(x, tsz);
                let want = ab_gated(&xa, k, arm).to_dense();
                let got = expand(&y, tsz).to_dense();
                let rel = rel_err(&got, &want);
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-13,
                    "{label} k={k} {arm:?} 2Sz={tsz}: SU(2) gate vs abelian gate rel {rel:.3e}"
                );
                if idx == 0 {
                    let x0 = xa.to_dense();
                    let moved = dense_diff_norm(&got, &x0) / dense_norm(&x0);
                    if matches!(arm, Arm::Theta(t) if t == PI) {
                        moved_pi = moved_pi.max(moved);
                    } else {
                        assert!(
                            moved > 1e-3,
                            "{label} k={k} {arm:?}: anti-vacuity, moved only {moved:.3e}"
                        );
                    }
                }
            }
        }
    }
    if has_pi && pi_moves {
        assert!(
            moved_pi > 1e-3,
            "{label}: anti-vacuity, θ = π moved only {moved_pi:.3e}"
        );
    }
    worst
}

#[test]
fn smr_t2_matches_abelian_gate_through_expand() {
    let arms = [
        Arm::Theta(0.3),
        Arm::Theta(-0.3),
        Arm::Cs(FRAC_PI_2),
        Arm::Cs(-FRAC_PI_2),
        Arm::Theta(PI),
    ];
    for (i, (label, space, ks)) in fixtures().into_iter().take(3).enumerate() {
        let x: Su2TtVector<f64> =
            randomized_su2_state(space, 1, 3, &mut seeded(0x052A_7200 + i as u64));
        let worst = t2_check(label, &x, &ks, &arms, true);
        println!("smr_t2 {label}: worst rel {worst:.3e}");
    }
}

#[test]
fn smr_t2_typed_same_irrep() {
    let arms = [
        Arm::Theta(0.3),
        Arm::Theta(-0.3),
        Arm::Cs(FRAC_PI_2),
        Arm::Cs(-FRAC_PI_2),
        Arm::Theta(PI),
    ];
    for (i, (label, space, ks)) in fixtures().into_iter().skip(3).enumerate() {
        for &k in &ks {
            assert_eq!(
                space.pg.irrep_of(k),
                space.pg.irrep_of(k + 1),
                "{label}: fixture bond {k} must be same-irrep"
            );
        }
        if label.starts_with('[') {
            assert_ne!(
                space.pg.irrep_of(ks[0]),
                Irrep::IDENTITY,
                "{label}: fixture must exercise a non-A1 pair"
            );
        }
        let x: Su2TtVector<f64> =
            randomized_su2_state(space, 1, 3, &mut seeded(0x052A_7300 + i as u64));
        let pi_moves = label != "[1,2,2,1] (4,4,2,A1)";
        let worst = t2_check(label, &x, &ks, &arms, pi_moves);
        println!("smr_t2_typed {label}: worst rel {worst:.3e}");
    }
}

#[test]
#[should_panic(expected = "different irreps")]
fn smr_refuses_cross_irrep() {
    let space = Su2Space::with_point_group(6, 3, 1, lih_pg(), Irrep::IDENTITY);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x052A_C200));
    let _ = apply_mode_rotation(&contract(x.core(2), x.core(3)), 0.3);
}

#[test]
#[should_panic(expected = "different irreps")]
fn smr_swap_refuses_cross_irrep() {
    let space = Su2Space::with_point_group(6, 3, 1, lih_pg(), Irrep::IDENTITY);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x052A_C201));
    let _ = apply_mode_swap(&contract(x.core(2), x.core(3)));
}

#[test]
fn smr_zero_angle_clone_precedes_irrep_refusal() {
    let space = Su2Space::with_point_group(6, 3, 1, lih_pg(), Irrep::IDENTITY);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x052A_C202));
    let tc = contract(x.core(2), x.core(3));
    assert_bitwise(&apply_mode_rotation(&tc, 0.0), &tc, "cross-irrep θ=0");
    assert_bitwise(&apply_mode_rotation(&tc, -0.0), &tc, "cross-irrep θ=−0");
    assert_bitwise(
        &apply_mode_rotation_serial(&tc, 0.0),
        &tc,
        "cross-irrep θ=0 serial",
    );
}

type FamTable = [[Option<u64>; 4]; 4];

#[test]
fn smr_class_generators_antisymmetric_orthogonal_and_keyed() {
    let angles = [0.3, -1.1, 0.8, FRAC_PI_2, 3.0, PI];
    let mut census: HashMap<usize, usize> = HashMap::new();
    let mut keyed: HashMap<(usize, usize, usize), FamTable> = HashMap::new();
    let mut racah_seen = false;
    let (mut worst_anti, mut worst_orth, mut worst_expm) = (0.0f64, 0.0f64, 0.0f64);
    for (label, space, ks) in fixtures() {
        for &k in &ks {
            let classes = class_generators(&space, k);
            let want = expected_classes(&space, k);
            assert_eq!(
                classes.len(),
                want.len(),
                "{label} k={k}: class count {} vs {} from the windows",
                classes.len(),
                want.len()
            );
            let mut seen = HashSet::new();
            for cl in &classes {
                let ctx = format!("{label} k={k} l={:?} r={:?}", cl.l, cl.r);
                assert!(seen.insert((cl.l, cl.r)), "{ctx}: duplicate class");
                let w = want
                    .get(&(cl.l, cl.r))
                    .unwrap_or_else(|| panic!("{ctx}: not a class"));
                assert_eq!(&cl.mids, w, "{ctx}: paths");
                let n = cl.mids.len();
                assert!((1..=4).contains(&n), "{ctx}: {n} paths");
                assert_eq!(cl.generator.len(), n * n, "{ctx}: generator size");
                *census.entry(n).or_default() += 1;
                let kk = &cl.generator;
                let fam: Vec<Family> = cl.mids.iter().map(|&m| fam_k(&space, k, cl.l, m)).collect();
                let nf = |i: usize| fam[i].delta().0 as i64;

                for i in 0..n {
                    for j in 0..n {
                        let a = (kk[i * n + j] + kk[j * n + i]).abs();
                        worst_anti = worst_anti.max(a);
                        assert!(a <= 1e-14, "{ctx}: |K+Kᵀ|[{i}][{j}] = {a:e}");
                        if (nf(i) - nf(j)).abs() != 1 {
                            assert_eq!(kk[i * n + j], 0.0, "{ctx}: K[{i}][{j}] must be exactly 0");
                        }
                    }
                }

                let k2 = mm(kk, kk, n);
                let p1: Vec<f64> = k2.iter().zip(eye(n)).map(|(a, b)| a + b).collect();
                let p4: Vec<f64> = k2.iter().zip(eye(n)).map(|(a, b)| a + 4.0 * b).collect();
                let spec = max_abs(&mm(&mm(kk, &p1, n), &p4, n));
                assert!(spec <= 1e-13, "{ctx}: K(K²+I)(K²+4I) = {spec:e}");
                let dn = cl.r.n - cl.l.n;
                if n == 1 {
                    assert!(kk[0] == 0.0, "{ctx}: single path, K must be exactly 0");
                }
                if dn % 2 == 1 {
                    assert_eq!(n, 2, "{ctx}: odd Δn must have 2 paths");
                }
                if n == 2 {
                    assert!(dn % 2 == 1, "{ctx}: 2 paths only at odd Δn");
                    let d = max_abs(&p1);
                    assert!(d <= 1e-14, "{ctx}: 2-path K² ≠ −I ({d:e})");
                }
                if n >= 3 {
                    assert!(
                        dn == 2 && cl.l.two_s == cl.r.two_s,
                        "{ctx}: {n} paths off the Δn = 2, Δ2s = 0 class"
                    );
                    assert_eq!(n == 3, cl.l.two_s == 0, "{ctx}: 3 vs 4 paths");
                }

                for &th in &angles {
                    let (s, c) = th.sin_cos();
                    let g = class_gate_cs(kk, n, c, s);
                    assert_eq!(g.len(), n * n);
                    let o = orth_defect(&g, n);
                    let e = max_abs(&sub(&g, &expm_antisym(&scaled(kk, th), n)));
                    worst_orth = worst_orth.max(o);
                    worst_expm = worst_expm.max(e);
                    assert!(o <= 1e-14, "{ctx} θ={th}: GᵀG−I = {o:e}");
                    assert!(e <= 1e-14, "{ctx} θ={th}: builder vs expm_antisym {e:e}");
                }
                let id = class_gate_cs(kk, n, 1.0, 0.0);
                assert!(
                    id.iter().zip(eye(n)).all(|(a, b)| *a == b),
                    "{ctx}: (c,s)=(1,0) must be the identity"
                );
                for sgn in [1.0f64, -1.0] {
                    let sw = class_gate_cs(kk, n, 0.0, sgn);
                    let o = orth_defect(&sw, n);
                    let e = max_abs(&sub(&sw, &expm_antisym(&scaled(kk, sgn * FRAC_PI_2), n)));
                    worst_orth = worst_orth.max(o);
                    worst_expm = worst_expm.max(e);
                    assert!(o <= 1e-14, "{ctx} (0,{sgn}): GᵀG−I = {o:e}");
                    assert!(e <= 1e-14, "{ctx} (0,{sgn}): vs expm_antisym(±π/2·K) {e:e}");
                    if n == 1 {
                        assert!(sw[0] == 1.0, "{ctx}: single-path swap must be exactly 1");
                    }
                    if n == 2 {
                        for i in 0..2 {
                            assert!(
                                sw[i * 2 + i] == 0.0,
                                "{ctx} (0,{sgn}): swap diagonal {} must be exactly 0",
                                sw[i * 2 + i]
                            );
                        }
                        for (i, j) in [(0usize, 1usize), (1, 0)] {
                            let v = sw[i * 2 + j];
                            assert!(
                                (v.abs() - 1.0).abs() <= 4.0 * f64::EPSILON,
                                "{ctx} (0,{sgn}): |S[{i}][{j}]| = {v}"
                            );
                            assert!(
                                v * sgn * kk[i * 2 + j] > 0.0,
                                "{ctx} (0,{sgn}): S[{i}][{j}] sign vs s·K"
                            );
                        }
                    }
                    if n == 4 && sw.iter().any(|v| v.abs() > 1e-3 && v.abs() < 1.0 - 1e-3) {
                        racah_seen = true;
                    }
                }
                if n == 1 {
                    for &th in &angles {
                        let (s, c) = th.sin_cos();
                        assert!(
                            class_gate_cs(kk, 1, c, s)[0] == 1.0,
                            "{ctx} θ={th}: single-path gate must be exactly 1"
                        );
                    }
                }

                let mut canon = [[None; 4]; 4];
                for i in 0..n {
                    for j in 0..n {
                        canon[fam[i] as usize][fam[j] as usize] = Some(kk[i * n + j].to_bits());
                    }
                }
                let key = (cl.l.two_s, dn, cl.r.two_s);
                match keyed.get(&key) {
                    Some(prev) => assert_eq!(
                        prev, &canon,
                        "{ctx}: K differs from an earlier class with key (l.2s, Δn, r.2s) = {key:?}"
                    ),
                    None => {
                        keyed.insert(key, canon);
                    }
                }
            }
        }
    }
    assert!(
        racah_seen,
        "anti-claim: no 4-path class swap mixes paths — the fixtures lost the recoupling class"
    );
    for n in 1..=4 {
        assert!(
            census.get(&n).copied().unwrap_or(0) > 0,
            "census: no {n}-path class"
        );
    }
    let mut c: Vec<_> = census.into_iter().collect();
    c.sort_unstable();
    println!(
        "smr_class census (paths, classes) = {c:?}; keys {}; worst |K+Kᵀ| {worst_anti:.2e}, \
         GᵀG−I {worst_orth:.2e}, builder−expm {worst_expm:.2e}",
        keyed.len()
    );
}

#[test]
fn smr_applied_gate_is_the_class_table() {
    for (i, (label, space, ks)) in fixtures().into_iter().enumerate() {
        let x: Su2TtVector<f64> =
            randomized_su2_state(space, 1, 3, &mut seeded(0x052A_7A00 + i as u64));
        for &k in &ks {
            let tc = contract(x.core(k), x.core(k + 1));
            let sc = scale(&tc);
            let classes = class_generators(&space, k);
            for (c, s) in [(0.0, 1.0), (0.0, -1.0), (0.3f64.cos(), 0.3f64.sin())] {
                let got = apply_mode_rotation_cs(&tc, c, s);
                let mats: Vec<Vec<f64>> = classes
                    .iter()
                    .map(|cl| class_gate_cs(&cl.generator, cl.mids.len(), c, s))
                    .collect();
                let want = apply_table(&tc, &classes, &mats);
                assert_same_shape(&got, &tc, &format!("{label} k={k} cs=({c},{s})"));
                let gap = max_diff(&got, &want);
                assert!(
                    gap <= 1e-14 * sc,
                    "{label} k={k} (c,s)=({c},{s}): applier vs class table {gap:e}"
                );
            }
            for th in [0.3, -1.1, PI] {
                let got = apply_mode_rotation(&tc, th);
                let mats: Vec<Vec<f64>> = classes
                    .iter()
                    .map(|cl| expm_antisym(&scaled(&cl.generator, th), cl.mids.len()))
                    .collect();
                let want = apply_table(&tc, &classes, &mats);
                assert_same_shape(&got, &tc, &format!("{label} k={k} θ={th}"));
                let gap = max_diff(&got, &want);
                assert!(
                    gap <= 1e-14 * sc,
                    "{label} k={k} θ={th}: applier vs expm_antisym table {gap:e}"
                );
            }
        }
    }
}

#[test]
fn smr_generator_table_is_g_pq() {
    let cases: Vec<(&str, Su2Space, Vec<usize>)> = vec![
        ("(5,4,2)", Su2Space::new(5, 4, 2), (0..4).collect()),
        ("(6,3,1)", Su2Space::new(6, 3, 1), (0..5).collect()),
        (
            "lih C2v (6,3,1)",
            Su2Space::with_point_group(6, 3, 1, lih_pg(), Irrep::IDENTITY),
            vec![0, 1],
        ),
        (
            "[2,3,3,1,1] (5,3,1,Γ1)",
            Su2Space::with_point_group(5, 3, 1, bpair5_pg(), Irrep::new(1)),
            vec![1],
        ),
    ];
    for (i, (label, space, ks)) in cases.into_iter().enumerate() {
        let x: Su2TtVector<f64> =
            randomized_su2_state(space, 1, 3, &mut seeded(0x052A_6E00 + i as u64));
        for &k in &ks {
            let tc = contract(x.core(k), x.core(k + 1));
            let classes = class_generators(&space, k);
            let gens: Vec<Vec<f64>> = classes.iter().map(|cl| cl.generator.clone()).collect();
            let kx = su2_resplit(&x, &apply_table(&tc, &classes, &gens));
            let oracle = g_pq(&x, k + 1, k);
            for tsz in members(&space) {
                let want = expand(&oracle, tsz).to_dense();
                let got = expand(&kx, tsz).to_dense();
                assert!(dense_norm(&want) > 1e-3, "{label} k={k}: vacuous oracle");
                let rel = rel_err(&got, &want);
                assert!(
                    rel <= 1e-12,
                    "{label} k={k} 2Sz={tsz}: class generators vs g_pq rel {rel:.3e}"
                );
            }
        }
    }
}

#[test]
fn smr_generator_matches_g_pq() {
    let space = Su2Space::new(5, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut seeded(0x052A_6F00));
    let h = 1e-4;
    for k in 0..4 {
        let plus = su2_gated(&x, k, Arm::Theta(h));
        let minus = su2_gated(&x, k, Arm::Theta(-h));
        let oracle = g_pq(&x, k + 1, k);
        for tsz in members(&space) {
            let (p, m) = (
                expand(&plus, tsz).to_dense(),
                expand(&minus, tsz).to_dense(),
            );
            let want = expand(&oracle, tsz).to_dense();
            let fd: Vec<f64> = p.iter().zip(&m).map(|(a, b)| (a - b) / (2.0 * h)).collect();
            let rel = rel_err(&fd, &want);
            assert!(
                rel < 1e-6,
                "pair ({k}, {}) 2Sz={tsz}: generator mismatch rel {rel:.3e}",
                k + 1
            );
        }
    }
}

#[test]
fn smr_t3_algebra() {
    let cases = [
        ("(6,3,1)", Su2Space::new(6, 3, 1), 0x052A_7301u64),
        ("(4,4,2)", Su2Space::new(4, 4, 2), 0x052A_7302),
    ];
    for (label, space, seed) in cases {
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut seeded(seed));
        for k in 1..space.d - 2 {
            let ctx = format!("{label} k={k}");
            let tc = contract(x.core(k), x.core(k + 1));
            let sc = scale(&tc);
            let classes = expected_classes(&space, k);
            let npaths = |t: &Tile| classes[&(t.l, t.r)].len();

            assert_bitwise(&apply_mode_rotation(&tc, 0.0), &tc, &format!("{ctx} θ=0"));
            assert_bitwise(&apply_mode_rotation(&tc, -0.0), &tc, &format!("{ctx} θ=−0"));

            let lhs = apply_mode_rotation(&apply_mode_rotation(&tc, -1.1), 0.3);
            let rhs = apply_mode_rotation(&tc, -0.8);
            let gap = max_diff(&lhs, &rhs);
            assert!(gap <= 1e-14 * sc, "{ctx}: G(0.3)G(−1.1) vs G(−0.8) {gap:e}");

            for th in [0.1f64, 0.37, -0.83, 3.0] {
                let (s, c) = th.sin_cos();
                assert_bitwise(
                    &apply_mode_rotation(&tc, th),
                    &apply_mode_rotation_cs(&tc, c, s),
                    &format!("{ctx} θ={th}: θ path vs cs path"),
                );
            }

            let sw = apply_mode_swap(&tc);
            assert_bitwise(
                &sw,
                &apply_mode_rotation_cs(&tc, 0.0, 1.0),
                &format!("{ctx} swap"),
            );
            for sgn in [1.0f64, -1.0] {
                let exact = apply_mode_rotation_cs(&tc, 0.0, sgn);
                let near = apply_mode_rotation(&tc, sgn * FRAC_PI_2);
                let gap = max_diff(&exact, &near);
                println!("smr_t3 {ctx}: max|cs(0,{sgn}) − G({sgn}·π/2)| = {gap:e}");
                assert!(gap <= 1e-15 * sc, "{ctx}: cs(0,{sgn}) vs G(±π/2) {gap:e}");
            }

            let s2 = apply_mode_swap(&sw);
            let s4 = apply_mode_swap(&apply_mode_swap(&s2));
            let gpi = apply_mode_rotation(&tc, PI);
            let g03 = apply_mode_rotation(&tc, 0.3);
            for (name, y) in [
                ("swap", &sw),
                ("S²", &s2),
                ("S⁴", &s4),
                ("G(π)", &gpi),
                ("G(0.3)", &g03),
            ] {
                assert_same_shape(y, &tc, &format!("{ctx} {name}"));
            }
            for t in tiles(&tc) {
                let a = tc.block(t.m).unwrap();
                let parity = if (t.r.n - t.l.n) % 2 == 1 { -1.0 } else { 1.0 };
                for (name, y, sign, tol) in [
                    ("G(π)", &gpi, parity, 1e-14 * sc),
                    ("S²", &s2, parity, 1e-14 * sc),
                    ("S⁴", &s4, 1.0, 1e-14 * sc),
                ] {
                    let b = y.block(t.m).unwrap();
                    for c in 0..t.rr {
                        for r in 0..t.rl {
                            let (i, j) = (t.ro + r, t.co + c);
                            let gap = (b[(i, j)] - sign * a[(i, j)]).abs();
                            assert!(gap <= tol, "{ctx} {name} tile {t:?}: {gap:e}");
                        }
                    }
                }
                if npaths(&t) == 1 {
                    for (name, y) in [("swap", &sw), ("G(π)", &gpi), ("G(0.3)", &g03)] {
                        let b = y.block(t.m).unwrap();
                        for c in 0..t.rr {
                            for r in 0..t.rl {
                                let (i, j) = (t.ro + r, t.co + c);
                                assert!(
                                    b[(i, j)] == a[(i, j)],
                                    "{ctx} {name}: single-path tile {t:?} must be copied exactly"
                                );
                            }
                        }
                    }
                }
            }

            let y2 = su2_resplit(&x, &s2);
            for tsz in members(&space) {
                let xa = expand(&x, tsz);
                let want = ab_gated(&ab_gated(&xa, k, Arm::Cs(FRAC_PI_2)), k, Arm::Cs(FRAC_PI_2))
                    .to_dense();
                let rel = rel_err(&expand(&y2, tsz).to_dense(), &want);
                assert!(
                    rel <= 1e-13,
                    "{ctx} 2Sz={tsz}: S² vs abelian swap² rel {rel:.3e}"
                );
            }

            let n0 = tc.dot_cores(&tc);
            for (name, y) in [("swap", &sw), ("G(π)", &gpi), ("G(0.3)", &g03)] {
                let n1 = y.dot_cores(y);
                let rel = (n1 - n0).abs() / n0;
                assert!(
                    rel <= 1e-14,
                    "{ctx} {name}: dot_cores {n0} → {n1} (rel {rel:e})"
                );
            }
            let n1 = apply_mode_rotation(&tc, -1.1).dot_cores(&apply_mode_rotation(&tc, -1.1));
            assert!(
                (n1 - n0).abs() <= 1e-14 * n0,
                "{ctx} G(−1.1): dot_cores {n0} → {n1}"
            );
        }
    }
}

#[test]
fn smr_c64_gate_acts_on_real_and_imaginary_parts_separately() {
    let space = Su2Space::new(5, 3, 1);
    let mut rng = seeded(0x052A_C640);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 2, &mut rng);
    let cr = contract(x.core(1), x.core(2));
    let mut ci = cr.clone();
    for &m in &cr.mid_qn {
        if let Some(mut b) = ci.block_mut(m) {
            for j in 0..b.ncols() {
                for i in 0..b.nrows() {
                    b[(i, j)] = rng.random_range(-1.0..1.0);
                }
            }
        }
    }
    let mut cz = Su2ContractedCores::<C64>::zeros(space, 1, &cr.row_ranks, &cr.col_ranks);
    for &m in &cr.mid_qn {
        if let (Some(a), Some(b), Some(mut z)) = (cr.block(m), ci.block(m), cz.block_mut(m)) {
            for j in 0..a.ncols() {
                for i in 0..a.nrows() {
                    z[(i, j)] = C64::new(a[(i, j)], b[(i, j)]);
                }
            }
        }
    }
    for th in [0.83, PI] {
        let (gr, gi, gz) = (
            apply_mode_rotation(&cr, th),
            apply_mode_rotation(&ci, th),
            apply_mode_rotation(&cz, th),
        );
        let (sr, si, sz) = (
            apply_mode_swap(&cr),
            apply_mode_swap(&ci),
            apply_mode_swap(&cz),
        );
        for ((a, b), z) in [((&gr, &gi), &gz), ((&sr, &si), &sz)] {
            for &m in &cr.mid_qn {
                if let (Some(a), Some(b), Some(z)) = (a.block(m), b.block(m), z.block(m)) {
                    for j in 0..a.ncols() {
                        for i in 0..a.nrows() {
                            assert!((z[(i, j)].re - a[(i, j)]).abs() < 1e-15, "θ={th}: re");
                            assert!((z[(i, j)].im - b[(i, j)]).abs() < 1e-15, "θ={th}: im");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn smr_t4_threaded_matches_serial_bitwise() {
    let space = Su2Space::new(6, 3, 1);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut seeded(0x052A_7400));
    let k = 2;
    let tc = contract(x.core(k), x.core(k + 1));
    let live = tc.mid_qn.iter().filter(|&&m| tc.block(m).is_some()).count();
    assert!(
        live >= PAR_MIN_MID_SECTORS,
        "fixture: {live} present mid sectors < PAR_MIN_MID_SECTORS = {PAR_MIN_MID_SECTORS}"
    );
    for th in [0.3, -0.9, 1.4, FRAC_PI_2, PI] {
        let par = apply_mode_rotation(&tc, th);
        let ser = apply_mode_rotation_serial(&tc, th);
        assert_bitwise(&par, &ser, &format!("θ={th}: threaded vs serial"));
        assert!(max_diff(&par, &tc) > 1e-3, "θ={th}: anti-vacuity");
    }
}

fn centred(x: &Su2TtVector<f64>, k: usize) -> Su2TtVector<f64> {
    let mut y = x.clone();
    y.move_core(k, true);
    y
}

fn with_spins(tc: &Su2ContractedCores<f64>, vals: &[f64]) -> Vec<(f64, usize)> {
    let mut out = Vec::with_capacity(vals.len());
    let mut it = vals.iter();
    for &m in &tc.mid_qn {
        let Some(b) = tc.block(m) else { continue };
        for _ in 0..b.nrows().min(b.ncols()) {
            out.push((*it.next().expect("too few values for the blocks"), m.two_s));
        }
    }
    assert!(it.next().is_none(), "more values than the blocks hold");
    out
}

fn averaged_spectrum(pairs: &[(f64, usize)]) -> Vec<f64> {
    let mut out = Vec::new();
    for &(w, ts) in pairs {
        let g = (ts + 1) as f64;
        out.extend(std::iter::repeat_n(w * w / g, ts + 1));
    }
    out
}

fn spectrum_gap(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().max(b.len());
    let prep = |v: &[f64]| {
        let mut v = v.to_vec();
        v.resize(n, 0.0);
        v.sort_by(|p, q| q.total_cmp(p));
        v
    };
    let (a, b) = (prep(a), prep(b));
    a.iter()
        .zip(&b)
        .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()))
}

fn member_matrix(y: &Su2TtVector<f64>, tsz: i64, k: usize) -> Mat<f64> {
    let v = expand(y, tsz).to_dense();
    let rows = 4usize.pow(k as u32 + 1);
    let cols = v.len() / rows;
    Mat::from_fn(rows, cols, |r, c| v[r + c * rows])
}

fn per_mid(rot: &Su2ContractedCores<f64>, theta_scale: bool, weight: bool) -> Vec<(f64, usize)> {
    let met = ThetaMetric::new(rot);
    let mut out = Vec::new();
    for &m in &rot.mid_qn {
        let Some(b) = rot.block(m) else { continue };
        if b.nrows() == 0 || b.ncols() == 0 {
            continue;
        }
        let mut bb = b.to_owned();
        if theta_scale {
            met.scale_in_place(&mut bb, m);
        }
        let w = if weight {
            pool_weight(&rot.space, m)
        } else {
            1.0
        };
        out.extend(svals_qr(bb.as_ref()).into_iter().map(|s| (w * s, m.two_s)));
    }
    out
}

#[test]
fn smr_k1_wvals_are_the_sz_averaged_schmidt_spectrum() {
    let thetas = [0.0, 0.37, -1.1, FRAC_PI_2];
    let (mut worst_spec, mut worst_norm, mut worst_vn) = (0.0f64, 0.0f64, 0.0f64);
    let mut f1 = (0.0f64, 0usize, 0.0f64, 0i64, 0.0f64, 0.0f64);
    for (i, (label, space)) in [
        ("h4 (4,4,2)", Su2Space::new(4, 4, 2)),
        ("h4 (4,4,0)", Su2Space::new(4, 4, 0)),
    ]
    .into_iter()
    .enumerate()
    {
        let x0: Su2TtVector<f64> =
            randomized_su2_state(space, 1, 3, &mut seeded(0x052A_9A10 + i as u64));
        let mems = members(&space);
        let gdeg = mems.len() as f64;
        let (mut no_theta, mut no_cw) = (0.0f64, 0.0f64);
        for k in 0..space.d - 1 {
            let x = centred(&x0, k);
            let tc = contract(x.core(k), x.core(k + 1));
            for &th in &thetas {
                let ctx = format!("{label} k={k} θ={th}");
                let w = rotated_bond_svals(&tc, th);
                let rot = apply_mode_rotation(&tc, th);
                let pairs = with_spins(&rot, &w);
                let total: f64 = w.iter().map(|v| v * v).sum();
                assert!(total > 1e-3, "{ctx}: vacuous spectrum");

                let y = su2_resplit(&x, &rot);
                let blocks: Vec<Mat<f64>> = mems.iter().map(|&t| member_matrix(&y, t, k)).collect();
                let (rows, cols) = (blocks[0].nrows(), blocks[0].ncols());
                let c = Mat::from_fn(rows, cols * blocks.len(), |r, j| {
                    blocks[j / cols][(r, j % cols)] / gdeg.sqrt()
                });
                let mu2: Vec<f64> = svals_qr(c.as_ref()).iter().map(|v| v * v).collect();

                let gap = spectrum_gap(&mu2, &averaged_spectrum(&pairs)) / total;
                worst_spec = worst_spec.max(gap);
                assert!(gap <= 1e-12, "{ctx}: σ(C)² vs ŵ²/(2s+1) rel {gap:.3e}");

                for (bm, &tsz) in blocks.iter().zip(&mems) {
                    let n2: f64 = (0..bm.ncols())
                        .flat_map(|j| (0..bm.nrows()).map(move |r| (r, j)))
                        .map(|(r, j)| bm[(r, j)] * bm[(r, j)])
                        .sum();
                    let rel = (n2 - total).abs() / total;
                    worst_norm = worst_norm.max(rel);
                    assert!(
                        rel <= 1e-12,
                        "{ctx} 2Sz={tsz}: ‖expand‖² {n2} vs Σŵ² {total}"
                    );
                }

                let s_bar = {
                    let tr: f64 = mu2.iter().sum();
                    -mu2.iter()
                        .filter(|&&v| v > 0.0)
                        .map(|&v| (v / tr) * (v / tr).ln())
                        .sum::<f64>()
                };
                let mult: f64 = pairs
                    .iter()
                    .map(|&(v, ts)| (v * v / total) * ((ts + 1) as f64).ln())
                    .sum();
                let vn = (s_bar - (bond_functional(&w, BondFunctional::VonNeumann) + mult)).abs();
                worst_vn = worst_vn.max(vn);
                assert!(vn <= 1e-12, "{ctx}: VN identity off by {vn:.3e}");

                no_theta = no_theta.max(
                    spectrum_gap(&mu2, &averaged_spectrum(&per_mid(&rot, false, true))) / total,
                );
                no_cw = no_cw.max(
                    spectrum_gap(&mu2, &averaged_spectrum(&per_mid(&rot, true, false))) / total,
                );

                if space.two_s == 2 {
                    let fw = bond_functional(&w, BondFunctional::default());
                    for (bm, &tsz) in blocks.iter().zip(&mems) {
                        let fm = bond_functional(&svals_qr(bm.as_ref()), BondFunctional::default());
                        if (fm - fw).abs() > f1.0 {
                            f1 = ((fm - fw).abs(), k, th, tsz, fm, fw);
                        }
                    }
                }
            }
        }
        println!("smr_k1 {label}: anti-oracles no θ {no_theta:.3e}, no √cw {no_cw:.3e} (·Σŵ²)");
        assert!(
            no_theta > 1e-6,
            "{label}: anti-oracle — dropping θ went unseen ({no_theta:.3e})"
        );
        assert!(
            no_cw > 1e-6,
            "{label}: anti-oracle — dropping √cw went unseen ({no_cw:.3e})"
        );
    }
    println!(
        "smr_k1: worst σ(C)² gap {worst_spec:.2e}·Σŵ², ‖expand_M‖² {worst_norm:.2e}, VN identity \
         {worst_vn:.2e}; F1 (4,4,2): max |f(member) − f(ŵ)| = {:.4} at k={} θ={} 2Sz={} \
         (member {:.6}, ŵ {:.6})",
        f1.0, f1.1, f1.2, f1.3, f1.4, f1.5
    );
    assert!(
        f1.0 > 1e-6,
        "F1: a single member's functional matched ŵ's everywhere ({:.3e}) — the fixture lost \
         the case that makes cross-layer θ comparisons invalid",
        f1.0
    );
}

#[test]
fn smr_k2_tiled_scorer_is_bitwise_rotated_bond_svals() {
    let space = Su2Space::new(6, 3, 1);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut seeded(0x052A_9A20));
    let k = 2;
    let x = centred(&x0, k);
    let tc = contract(x.core(k), x.core(k + 1));
    let (tab, met) = (gate_tables(&tc), ThetaMetric::new(&tc));
    for th in [0.0, 0.37, -0.81, 0.2, FRAC_PI_2, -1e-9] {
        let a = rotated_bond_svals(&tc, th);
        let b = rotated_bond_wvals_tiled(&tc, &tab, &met, th);
        assert_eq!(a.len(), b.len(), "θ = {th}: spectrum length");
        assert!(!a.is_empty(), "θ = {th}: vacuous");
        for (i, (u, v)) in a.iter().zip(&b).enumerate() {
            assert_eq!(u.to_bits(), v.to_bits(), "θ = {th}, ŵ[{i}]: {u} vs {v}");
        }
    }
}

#[test]
fn smr_k3_zero_angle_scorer_is_the_weighted_svdvals() {
    for (i, space) in [Su2Space::new(4, 4, 2), Su2Space::new(6, 3, 1)]
        .into_iter()
        .enumerate()
    {
        let x0: Su2TtVector<f64> =
            randomized_su2_state(space, 1, 3, &mut seeded(0x052A_9A30 + i as u64));
        let sv = x0.clone().svdvals();
        for k in 0..space.d - 1 {
            let x = centred(&x0, k);
            let w = rotated_bond_svals(&contract(x.core(k), x.core(k + 1)), 0.0);
            let mut want = Vec::new();
            for s in sv[k].sectors() {
                if let Some(b) = sv[k].block(s) {
                    let pw = pool_weight(&space, s);
                    want.extend((0..b.nrows().min(b.ncols())).map(|j| pw * b[(j, j)]));
                }
            }
            let sc = w.iter().fold(0.0f64, |m, v| m.max(*v));
            let gap = spectrum_gap(&w, &want);
            assert!(sc > 1e-3, "space {i} k={k}: vacuous");
            assert!(
                gap <= 1e-12 * sc,
                "space {i} k={k}: ŵ vs svdvals·√cw {gap:.3e}"
            );
        }
    }
}

#[test]
fn smr_k4_search_beats_the_planted_inverse() {
    let space = Su2Space::new(6, 3, 1);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x052A_9A40));
    let k = 2;
    let x = centred(&x0, k);
    let opts = ModeRotOpts::default();
    let raw = contract(x.core(k), x.core(k + 1));
    let base = apply_mode_rotation(&raw, search_angle(&raw, &opts).0);
    let f_origin = bond_functional(&rotated_bond_svals(&base, 0.0), opts.functional);
    let cp = apply_mode_rotation(&base, 0.37);
    let (theta, f0, ft) = search_angle(&cp, &opts);
    println!("smr_k4: θ = {theta}, f0 = {f0}, f(θ) = {ft}, f(undo) = {f_origin}");
    assert!(
        f0 > f_origin + 1e-3,
        "anti-vacuity: the plant must raise the functional ({f0} vs {f_origin})"
    );
    assert!(ft <= f0, "search returned a worse point: {ft} > {f0}");
    assert!(
        ft <= f_origin + 1e-3 * f_origin.abs().max(1.0),
        "search (θ={theta}, f={ft}) did worse than undoing the plant (f={f_origin})"
    );
    let a = bond_functional(&rotated_bond_svals(&cp, 0.2), opts.functional);
    let b = bond_functional(&rotated_bond_svals(&cp, 0.2 + PI), opts.functional);
    assert!((a - b).abs() < 1e-12, "π-periodicity: {a} vs {b}");
}

#[test]
fn smr_k5_no_rotation_exits_are_exactly_zero() {
    use Family::{Empty, Occ, Up};
    let csf: Su2TtVector<f64> = tt_csf(&[Up, Occ, Occ, Empty]);
    let tc = contract(csf.core(1), csf.core(2));
    let (t, f0, ft) = search_angle(&tc, &ModeRotOpts::default());
    assert_eq!(t.to_bits(), 0.0f64.to_bits(), "flat grid: θ = {t}");
    assert_eq!(
        ft.to_bits(),
        f0.to_bits(),
        "flat grid: f(θ) {ft} vs f0 {f0}"
    );

    let space = Su2Space::new(6, 3, 1);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x052A_9A50));
    let x = centred(&x0, 2);
    let cp = apply_mode_rotation(&contract(x.core(2), x.core(3)), 0.37);
    let (moved, _, _) = search_angle(&cp, &ModeRotOpts::default());
    assert!(
        moved != 0.0,
        "anti-vacuity: the default search must rotate this pair"
    );
    let opts = ModeRotOpts {
        improve_tol: f64::INFINITY,
        ..ModeRotOpts::default()
    };
    let (t, f0, ft) = search_angle(&cp, &opts);
    assert_eq!(t.to_bits(), 0.0f64.to_bits(), "improve_tol = ∞: θ = {t}");
    assert_eq!(
        ft.to_bits(),
        f0.to_bits(),
        "improve_tol = ∞: f(θ) {ft} vs f0 {f0}"
    );
}

#[test]
fn smr_k6_search_matches_serial_bitwise() {
    let space = Su2Space::new(6, 3, 1);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut seeded(0x052A_9A60));
    let k = 2;
    let x = centred(&x0, k);
    let tc = apply_mode_rotation(&contract(x.core(k), x.core(k + 1)), -0.6);
    let live = tc.mid_qn.iter().filter(|&&m| tc.block(m).is_some()).count();
    assert!(live >= PAR_MIN_MID_SECTORS, "fixture: {live} mid sectors");
    let run = |n: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap();
        pool.install(|| {
            let (tab, met) = (gate_tables(&tc), ThetaMetric::new(&tc));
            let mut bits = Vec::new();
            let (t, f0, ft) = search_angle(&tc, &ModeRotOpts::default());
            bits.extend([t.to_bits(), f0.to_bits(), ft.to_bits()]);
            for th in [0.0, 0.3, -1.2, FRAC_PI_2] {
                bits.extend(rotated_bond_svals(&tc, th).iter().map(|v| v.to_bits()));
                bits.extend(
                    rotated_bond_wvals_tiled(&tc, &tab, &met, th)
                        .iter()
                        .map(|v| v.to_bits()),
                );
            }
            bits
        })
    };
    let (one, many) = (run(1), run(8));
    assert!(one.len() > 3, "vacuous");
    assert_eq!(one, many, "1-thread vs 8-thread bits");
}

#[test]
fn smr_k5_exits_are_told_apart_by_the_evaluation_count() {
    use Family::{Empty, Occ, Up};
    use std::cell::Cell;
    let counted = |tc: &Su2ContractedCores<f64>, opts: &ModeRotOpts| -> (f64, usize) {
        let (tab, met) = (gate_tables(tc), ThetaMetric::new(tc));
        let n = Cell::new(0usize);
        let got = minimize_angle(
            |th| {
                n.set(n.get() + 1);
                bond_functional(
                    &rotated_bond_wvals_tiled(tc, &tab, &met, th),
                    opts.functional,
                )
            },
            opts,
        );
        let want = search_angle(tc, opts);
        assert_eq!(
            [got.0.to_bits(), got.1.to_bits(), got.2.to_bits()],
            [want.0.to_bits(), want.1.to_bits(), want.2.to_bits()],
            "the counted minimization is not search_angle's"
        );
        (got.0, n.get())
    };
    let def = ModeRotOpts::default();
    assert_eq!(
        (def.n_grid, def.n_refine),
        (8, 8),
        "the counts are the defaults'"
    );

    let csf: Su2TtVector<f64> = tt_csf(&[Up, Occ, Occ, Empty]);
    let (t, n_flat) = counted(&contract(csf.core(1), csf.core(2)), &def);
    assert_eq!(t.to_bits(), 0.0f64.to_bits(), "flat grid: θ = {t}");
    assert_eq!(n_flat, 8, "flat-grid exit: f(0) + 7 grid points");

    let space = Su2Space::new(6, 3, 1);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x052A_9A50));
    let x = centred(&x0, 2);
    let cp = apply_mode_rotation(&contract(x.core(2), x.core(3)), 0.37);
    let (moved, n_acc) = counted(&cp, &def);
    assert!(
        moved != 0.0,
        "anti-vacuity: the default search must rotate this pair"
    );
    assert_eq!(
        n_acc, 18,
        "accepting search: 8 grid + 2 seeds + 8 refinements"
    );
    let inf = ModeRotOpts {
        improve_tol: f64::INFINITY,
        ..def
    };
    let (t, n_rej) = counted(&cp, &inf);
    assert_eq!(t.to_bits(), 0.0f64.to_bits(), "improve_tol = ∞: θ = {t}");
    assert_eq!(n_rej, 18, "improve_tol exit: the whole golden section ran");
}

#[test]
fn smr_k7_wvals_are_the_split_sketch_times_pool_weight() {
    let space = Su2Space::new(6, 3, 1);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut seeded(0x052A_9A70));
    let (mut worst, mut anti, mut n_vals) = (0.0f64, 0.0f64, 0usize);
    for k in 0..space.d - 1 {
        let x = centred(&x0, k);
        let raw = contract(x.core(k), x.core(k + 1));
        for th in [0.0, 0.37, -1.1] {
            let w = rotated_bond_svals(&raw, th);
            let tc = apply_mode_rotation(&raw, th);
            let (_, s, _, _) =
                factor_svd_noise(&tc, 0.0, Pooled::NONE, false, 0.0, &mut noise_rng(0));
            let (mut want, mut unweighted) = (Vec::new(), Vec::new());
            for &m in &tc.mid_qn {
                let Some(b) = tc.block(m) else { continue };
                let n = b.nrows().min(b.ncols());
                if n == 0 {
                    continue;
                }
                let pw = pool_weight(&space, m);
                let sb = s
                    .block(m)
                    .expect("keep-all: every live mid sector has a sketch block");
                assert_eq!(
                    sb.nrows(),
                    n,
                    "k={k} θ={th}: keep-all kept {} of {n}",
                    sb.nrows()
                );
                want.extend((0..n).map(|j| pw * sb[(j, j)]));
                unweighted.extend((0..n).map(|j| sb[(j, j)]));
            }
            assert_eq!(w.len(), want.len(), "k={k} θ={th}: spectrum length");
            let sc = w.iter().fold(0.0f64, |a, v| a.max(*v));
            assert!(sc > 1e-3, "k={k} θ={th}: vacuous");
            for ((a, b), c) in w.iter().zip(&want).zip(&unweighted) {
                worst = worst.max((a - b).abs() / sc);
                anti = anti.max((a - c).abs() / sc);
            }
            n_vals += w.len();
        }
    }
    println!(
        "smr_k7: {n_vals} values, worst |ŵ − √cw·σ_split| / max ŵ = {worst:.3e} \
         (raw σ_split: {anti:.3e})"
    );
    assert!(
        anti > 1e-3,
        "anti-oracle: the unweighted σ also matches ({anti:.3e})"
    );
    assert!(worst <= 1e-12, "ŵ vs the split's √cw·σ: {worst:.3e}");
}
