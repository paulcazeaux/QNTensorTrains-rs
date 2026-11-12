// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;
use std::panic::{AssertUnwindSafe, catch_unwind};

use qntt::integrals::Integrals;
use qntt::ordering::{
    SwapLayerInfo, apply_swap_layer, permutation_matrix, permute_integrals, swap_matrix,
    swap_network_layer,
};
use qntt::rdm::orbital_rdms;
use qntt::rotation::{
    compose_rotations, expm_antisym, givens, identity_matrix, orthogonality_defect,
    rotate_integrals,
};
use qntt::solvers::riemannian::normalize_mut;
use qntt::{
    C64, Irrep, PointGroup, PointGroupKind, Pooled, QnSpace, SparseHamiltonian, TtVector, dot,
    max_ranks, rayleigh_quotient, tt_randn,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::{
    basis_index, dense_gpq, dense_state_vector, dets_for, flat_to_nested, mat_vec, random_sym_t,
    random_sym_v, randomized_state,
};

type Dets = Vec<(Vec<bool>, Vec<bool>)>;

#[derive(Clone, Debug, PartialEq)]
struct Signed {
    arr: Vec<usize>,
    sgn: Vec<f64>,
}

impl Signed {
    fn id(d: usize) -> Self {
        Signed {
            arr: (0..d).collect(),
            sgn: vec![1.0; d],
        }
    }

    fn swap(&mut self, k: usize) {
        self.arr.swap(k, k + 1);
        self.sgn.swap(k, k + 1);
        self.sgn[k] = -self.sgn[k];
    }

    fn apply_layer(&mut self, bonds: &[usize]) {
        for &k in bonds {
            self.swap(k);
        }
    }

    fn matrix(&self) -> Vec<f64> {
        let d = self.arr.len();
        let mut u = vec![0.0f64; d * d];
        for new in 0..d {
            u[self.arr[new] * d + new] = self.sgn[new];
        }
        u
    }
}

fn signed_relabel(ints: &Integrals<f64>, s: &Signed) -> (Vec<f64>, Vec<f64>) {
    let d = ints.norb;
    let per = permute_integrals(ints, &s.arr);
    let mut t = per.t.data.clone();
    for a in 0..d {
        for b in 0..d {
            t[a * d + b] *= s.sgn[a] * s.sgn[b];
        }
    }
    let mut v = per.v.data.clone();
    for a in 0..d {
        for b in 0..d {
            for c in 0..d {
                for e in 0..d {
                    v[((a * d + b) * d + c) * d + e] *= s.sgn[a] * s.sgn[b] * s.sgn[c] * s.sgn[e];
                }
            }
        }
    }
    (t, v)
}

fn swap_ints(d: usize, seed: u64) -> Integrals<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    Integrals {
        t: random_sym_t(d, &mut rng),
        v: random_sym_v(d, &mut rng),
        e0: 0.3,
        norb: d,
        nelec: 4,
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    }
}

fn energy(ints: &Integrals<f64>, x: &TtVector<f64>) -> f64 {
    let h = SparseHamiltonian::new(x.space, &ints.t, &ints.v, None, f64::EPSILON);
    rayleigh_quotient(&h, x)
}

struct DenseGates {
    d: usize,
    dets: Dets,
    cache: HashMap<usize, Vec<Vec<f64>>>,
}

impl DenseGates {
    fn new(d: usize, dets: &Dets) -> Self {
        DenseGates {
            d,
            dets: dets.clone(),
            cache: HashMap::new(),
        }
    }

    fn gate(&mut self, k: usize) -> &Vec<Vec<f64>> {
        if !self.cache.contains_key(&k) {
            let n = self.dets.len();
            let g = dense_gpq(self.d, k + 1, k, &self.dets);
            let flat: Vec<f64> = g.iter().flatten().map(|&v| FRAC_PI_2 * v).collect();
            self.cache
                .insert(k, flat_to_nested(&expm_antisym(&flat, n), n));
        }
        &self.cache[&k]
    }

    fn apply(&mut self, bonds: &[usize], mut c: Vec<f64>) -> Vec<f64> {
        for &k in bonds {
            c = mat_vec(self.gate(k), &c);
        }
        c
    }
}

struct Fix {
    name: &'static str,
    space: QnSpace,
    dets: Dets,
    x: TtVector<f64>,
}

fn fixture(
    name: &'static str,
    d: usize,
    nup: usize,
    ndn: usize,
    lohi: (usize, usize),
    seed: u64,
) -> Fix {
    let space = QnSpace::new(d, nup, ndn);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = randomized_state(space, lohi.0, lohi.1, &mut rng);
    let raw = dot(&x, &x);
    assert!(raw > 1e-3, "{name}: raw ⟨x,x⟩ = {raw:e} is negligible");
    normalize_mut(&mut x);
    assert!((dot(&x, &x) - 1.0).abs() <= 1e-13, "{name}: not normalized");
    Fix {
        name,
        space,
        dets: dets_for(d, nup, ndn),
        x,
    }
}

fn run_layer(
    x: &TtVector<f64>,
    bonds: &[usize],
    chi: Pooled,
    eps: f64,
) -> (TtVector<f64>, Vec<f64>, SwapLayerInfo) {
    let mut y = x.clone();
    let (u, info) = apply_swap_layer(&mut y, bonds, chi, eps);
    (y, u, info)
}

fn split_bonds(bonds: &[usize]) -> Vec<usize> {
    bonds.iter().map(|&k| k + 1).collect()
}

fn assert_layer_postconditions(
    tag: &str,
    x: &TtVector<f64>,
    y: &TtVector<f64>,
    bonds: &[usize],
    cap: Option<usize>,
) {
    assert!(y.check(), "{tag}: output train inconsistent");
    let d = x.space.d;
    let n_dot = dot(y, y).sqrt();
    assert!(
        (n_dot - 1.0).abs() <= 1e-12,
        "{tag}: ‖y‖ = {n_dot:.16} (unit input)"
    );
    let last = *bonds.last().expect("non-empty layer");
    assert_eq!(y.ortho, Some(last + 1), "{tag}: orthogonality centre");
    assert!(
        (y.norm() - n_dot).abs() <= 1e-12,
        "{tag}: centre norm {} vs gauge-free {n_dot} (stale ortho?)",
        y.norm()
    );
    let split = split_bonds(bonds);
    for b in 0..=d {
        if !split.contains(&b) {
            assert_eq!(y.ranks[b], x.ranks[b], "{tag}: untouched bond {b} changed");
        }
    }
    if let Some(dcap) = cap {
        for &b in &split {
            let r = y.ranks[b].total();
            assert!(
                r <= dcap,
                "{tag}: split bond {b} pooled rank {r} > cap {dcap}"
            );
        }
    }
}

fn max_abs_diff_vec(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn infidelity(a: &TtVector<f64>, b: &TtVector<f64>) -> f64 {
    let ov = dot(a, b) / (dot(a, a) * dot(b, b)).sqrt();
    1.0 - ov * ov
}

fn panic_message(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("expected a panic");
    if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        (*s).to_string()
    } else {
        String::from("<non-string panic payload>")
    }
}

fn assert_refused(tag: &str, f: impl FnOnce(), phrases: &[&str]) {
    let msg = panic_message(f);
    println!("{tag} refusal: {msg}");
    for p in phrases {
        assert!(msg.contains(p), "{tag}: message lacks {p:?}: {msg}");
    }
}

#[test]
fn tord_s1_schedule_covers_every_pair() {
    for d in [2usize, 4, 6, 16, 36] {
        for l in 0..d {
            let b = swap_network_layer(d, l);
            let want: Vec<usize> = (l % 2..d - 1).step_by(2).collect();
            assert_eq!(b, want, "d={d}: layer {l}");
            assert_eq!(b.len(), if l % 2 == 0 { d / 2 } else { d / 2 - 1 });
            assert!(
                b.windows(2).all(|w| w[1] >= w[0] + 2),
                "d={d} l={l}: overlap"
            );
            assert!(b.iter().all(|&k| k + 1 < d), "d={d} l={l}: out of range");
        }
        for l in 0..4 {
            assert_eq!(swap_network_layer(d, l), swap_network_layer(d, l + 2));
            assert_eq!(
                swap_network_layer(d, l),
                swap_network_layer(d, l + 2 * d + 2)
            );
        }

        let mut arr: Vec<usize> = (0..d).collect();
        let mut tr = Signed::id(d);
        let mut u = identity_matrix(d);
        let mut swapped = vec![0usize; d * d];
        let mut adjacent = vec![0usize; d * d];
        for l in 0..2 * d {
            if l % 2 == 0 && l / 2 < d / 2 {
                for w in arr.windows(2) {
                    let (a, b) = (w[0].min(w[1]), w[0].max(w[1]));
                    adjacent[a * d + b] += 1;
                }
            }
            if l == d {
                let rev: Vec<usize> = (0..d).rev().collect();
                assert_eq!(arr, rev, "d={d}: P_(d/2) is not the reversal");
                for o in 0..d {
                    for new in 0..d {
                        let want = if new == d - 1 - o {
                            if o % 2 == 0 { 1.0 } else { -1.0 }
                        } else {
                            0.0
                        };
                        assert!(u[o * d + new] == want, "d={d}: period U[{o}][{new}]");
                    }
                }
            }
            for &k in &swap_network_layer(d, l) {
                if l < d {
                    let (a, b) = (arr[k].min(arr[k + 1]), arr[k].max(arr[k + 1]));
                    swapped[a * d + b] += 1;
                }
                arr.swap(k, k + 1);
                tr.swap(k);
                u = compose_rotations(&u, &swap_matrix(d, k), d);
            }
            assert_eq!(tr.arr, arr);
            let want = tr.matrix();
            let pm = permutation_matrix(&arr);
            for i in 0..d * d {
                assert!(
                    u[i] == want[i],
                    "d={d} after layer {l}: U cell {i} {} vs {}",
                    u[i],
                    want[i]
                );
                assert!(
                    u[i].abs() == pm[i],
                    "d={d} after layer {l}: support cell {i}"
                );
            }
            assert_eq!(orthogonality_defect(&u, d), 0.0, "d={d} after layer {l}");
        }
        for a in 0..d {
            for b in a + 1..d {
                assert_eq!(swapped[a * d + b], 1, "d={d}: pair ({a},{b}) swap count");
                assert_eq!(
                    adjacent[a * d + b],
                    1,
                    "d={d}: pair ({a},{b}) Walecki count"
                );
            }
        }
        assert_eq!(swapped.iter().sum::<usize>(), d * (d - 1) / 2);
        assert_eq!(
            arr,
            (0..d).collect::<Vec<_>>(),
            "d={d}: P_d is not the identity"
        );
        for i in 0..d {
            for j in 0..d {
                let want = if i == j { -1.0 } else { 0.0 };
                assert!(
                    u[i * d + j] == want,
                    "d={d}: two periods U[{i}][{j}] (want −I)"
                );
            }
        }
        println!(
            "tord_s1 d={d}: {} swaps per period, schedule verified",
            d * (d - 1) / 2
        );
    }
}

#[test]
fn tord_s1_bad_d_refused() {
    for d in [1usize, 5, 7] {
        let ds = format!("{d}");
        assert_refused(
            "swap_network_layer odd d",
            || {
                let _ = swap_network_layer(d, 0);
            },
            &["swap_network_layer", "odd", &ds],
        );
    }
    assert_refused(
        "swap_network_layer d=0",
        || {
            let _ = swap_network_layer(0, 0);
        },
        &["swap_network_layer", "0"],
    );
}

#[test]
fn tord_s2_swap_matrix_is_the_signed_permutation() {
    let mut worst = 0.0f64;
    for d in [2usize, 5, 6] {
        for k in 0..d - 1 {
            let s = swap_matrix(d, k);
            assert_eq!(s.len(), d * d);
            let mut tau: Vec<usize> = (0..d).collect();
            tau.swap(k, k + 1);
            let mut diag = identity_matrix(d);
            diag[k * d + k] = -1.0;
            let want = compose_rotations(&permutation_matrix(&tau), &diag, d);
            let mut tr = Signed::id(d);
            tr.swap(k);
            let want2 = tr.matrix();
            let g = givens(d, k, FRAC_PI_2);
            for i in 0..d * d {
                assert!(
                    s[i] == want[i],
                    "d={d} k={k}: cell {i}: {} vs P·D {}",
                    s[i],
                    want[i]
                );
                assert!(s[i] == want2[i], "d={d} k={k}: cell {i} vs F2 tracker");
                assert!(
                    s[i] == g[i].round(),
                    "d={d} k={k}: cell {i} vs round(givens)"
                );
                worst = worst.max((s[i] - g[i]).abs());
            }
            assert_eq!(orthogonality_defect(&s, d), 0.0, "d={d} k={k}");
            let s2 = compose_rotations(&s, &s, d);
            let mut minus_pair = identity_matrix(d);
            minus_pair[k * d + k] = -1.0;
            minus_pair[(k + 1) * d + k + 1] = -1.0;
            let s4 = compose_rotations(&s2, &s2, d);
            let id = identity_matrix(d);
            for i in 0..d * d {
                assert!(s2[i] == minus_pair[i], "d={d} k={k}: S² cell {i}");
                assert!(s4[i] == id[i], "d={d} k={k}: S⁴ cell {i}");
            }
        }
    }
    println!("tord_s2: max|swap_matrix − givens(π/2)| = {worst:e}");
    assert!(worst <= 1e-16 && worst > 0.0, "givens(π/2) gap {worst:e}");
}

#[test]
fn tord_s2_swap_matrix_refuses_out_of_range() {
    assert_refused(
        "swap_matrix",
        || {
            let _ = swap_matrix(4, 3);
        },
        &["swap_matrix", "3"],
    );
}

#[test]
fn tord_s2_rotate_integrals_is_a_signed_relabelling() {
    let mut flips = 0usize;
    for (d, seed) in [(5usize, 0x52_0005u64), (6, 0x52_0006)] {
        let ints = swap_ints(d, seed);
        let mut cases: Vec<(String, Vec<usize>)> =
            (0..d - 1).map(|k| (format!("k={k}"), vec![k])).collect();
        if d % 2 == 0 {
            for l in 0..2 {
                cases.push((format!("layer {l}"), swap_network_layer(d, l)));
            }
        }
        for (tag, bonds) in cases {
            let mut u = identity_matrix(d);
            for &k in &bonds {
                u = compose_rotations(&u, &swap_matrix(d, k), d);
            }
            let mut tr = Signed::id(d);
            tr.apply_layer(&bonds);
            let rot = rotate_integrals(&ints, &u);
            let (t, v) = signed_relabel(&ints, &tr);
            let per = permute_integrals(&ints, &tr.arr);
            for i in 0..d * d {
                assert!(
                    rot.t.data[i] == t[i],
                    "d={d} {tag}: t cell {i}: {} vs {}",
                    rot.t.data[i],
                    t[i]
                );
                if t[i] != 0.0 && t[i] == -per.t.data[i] {
                    flips += 1;
                }
            }
            for i in 0..d * d * d * d {
                assert!(
                    rot.v.data[i] == v[i],
                    "d={d} {tag}: v cell {i}: {} vs {}",
                    rot.v.data[i],
                    v[i]
                );
            }
            assert_eq!(rot.e0, ints.e0, "d={d} {tag}: e0");
            assert_eq!(
                (rot.norb, rot.nelec, rot.ms2, rot.isym),
                (ints.norb, ints.nelec, ints.ms2, ints.isym)
            );
            assert!(
                rot.orbsym.is_empty(),
                "d={d} {tag}: empty orbsym stays empty"
            );
        }
    }
    println!("tord_s2: {flips} t cells where σ flips a nonzero value");
    assert!(flips > 0, "anti-vacuity: σ never flipped a nonzero t cell");
}

fn s3_fixtures() -> Vec<(Fix, Vec<Vec<usize>>)> {
    let a = fixture("A d=6 (3,0)", 6, 3, 0, (1, 2), 0x53_000A);
    let b = fixture("B d=5 (2,2)", 5, 2, 2, (1, 2), 0x53_000B);
    let bp = fixture("B′ d=5 (2,2) zero-rank", 5, 2, 2, (0, 2), 0x53_00B1);
    let zeros: usize = (1..bp.space.d)
        .map(|k| bp.x.ranks[k].iter().filter(|(_, r)| **r == 0).count())
        .sum();
    println!("tord_s3 {}: {zeros} zero-rank interior sectors", bp.name);
    assert!(zeros > 0, "B′ anti-vacuity: no zero-rank sector");
    let c = fixture("C d=6 (3,2)", 6, 3, 2, (1, 2), 0x53_000C);
    let net6 = vec![swap_network_layer(6, 0), swap_network_layer(6, 1)];
    let hand5 = vec![vec![0, 2], vec![1, 3]];
    vec![
        (a, net6.clone()),
        (b, hand5.clone()),
        (bp, hand5),
        (c, net6),
    ]
}

#[test]
fn tord_s3_dense_oracle_one_layer() {
    for (fix, layers) in s3_fixtures() {
        let d = fix.space.d;
        let mut gates = DenseGates::new(d, &fix.dets);
        let c0 = dense_state_vector(&fix.dets, &fix.x);
        let mut runs: Vec<(String, Vec<Vec<usize>>)> = layers
            .iter()
            .map(|b| (format!("bonds {b:?}"), vec![b.clone()]))
            .collect();
        if d % 2 == 0 {
            runs.push(("double layer 0,1".into(), layers.clone()));
        }
        for (tag, seq) in runs {
            let tag = format!("{} {tag}", fix.name);
            let mut y = fix.x.clone();
            let mut c = c0.clone();
            let mut tr = Signed::id(d);
            let mut u_all = identity_matrix(d);
            for bonds in &seq {
                let before = y.clone();
                let (u, info) = apply_swap_layer(&mut y, bonds, Pooled::NONE, 0.0);
                assert_layer_postconditions(&tag, &before, &y, bonds, None);
                assert_eq!(info.discarded, 0.0, "{tag}: discarded at NONE/eps 0");
                let mbd = split_bonds(bonds)
                    .iter()
                    .map(|&b| y.ranks[b].total())
                    .max()
                    .unwrap();
                assert_eq!(info.max_bond_dim, mbd, "{tag}: max_bond_dim");
                let mut ltr = Signed::id(d);
                ltr.apply_layer(bonds);
                let want = ltr.matrix();
                for i in 0..d * d {
                    assert!(u[i] == want[i], "{tag}: layer u cell {i}");
                }
                u_all = compose_rotations(&u_all, &u, d);
                tr.apply_layer(bonds);
                c = gates.apply(bonds, c);
            }
            let got = dense_state_vector(&fix.dets, &y);
            let err = max_abs_diff_vec(&got, &c);
            let moved = max_abs_diff_vec(&c, &c0);
            println!("tord_s3 {tag}: max|Δ| vs dense = {err:e} (state moved {moved:.3e})");
            assert!(
                moved > 1e-2,
                "{tag}: anti-vacuity: the layer did not move the state"
            );
            assert!(err <= 1e-12, "{tag}: dense mismatch {err:e}");
            let want = tr.matrix();
            for i in 0..d * d {
                assert!(u_all[i] == want[i], "{tag}: composed u cell {i}");
            }
        }
    }
}

#[test]
fn tord_s3_energy_invariance() {
    for (fix, layers) in s3_fixtures() {
        if fix.name.starts_with("B′") {
            continue;
        }
        let d = fix.space.d;
        let ints = swap_ints(d, 0x53_E000 + d as u64);
        let e0 = energy(&ints, &fix.x);
        let scale = e0.abs().max(1.0);
        for bonds in &layers {
            let tag = format!("{} bonds {bonds:?}", fix.name);
            let (y, u, _) = run_layer(&fix.x, bonds, Pooled::NONE, 0.0);
            let e1 = energy(&rotate_integrals(&ints, &u), &y);
            println!("tord_s3 {tag}: E0 = {e0:.12}, |ΔE| = {:e}", (e1 - e0).abs());
            assert!((e1 - e0).abs() <= 1e-12 * scale, "{tag}: E {e0} → {e1}");

            let mut tr = Signed::id(d);
            tr.apply_layer(bonds);
            let unsigned = energy(&rotate_integrals(&ints, &permutation_matrix(&tr.arr)), &y);
            let mut wrong = tr.matrix();
            let k0 = bonds[0];
            for new in [k0, k0 + 1] {
                for old in 0..d {
                    wrong[old * d + new] = -wrong[old * d + new];
                }
            }
            let wrong_site = energy(&rotate_integrals(&ints, &wrong), &y);
            let (m1, m2) = ((unsigned - e0).abs(), (wrong_site - e0).abs());
            println!("tord_s3 {tag}: negative-arm margins unsigned {m1:.3e}, wrong site {m2:.3e}");
            assert!(
                m1 > 1e-6,
                "{tag}: unsigned relabel is indistinguishable ({m1:e})"
            );
            assert!(
                m2 > 1e-6,
                "{tag}: wrong-site sign is indistinguishable ({m2:e})"
            );
        }
    }
}

#[test]
fn tord_s3_binding_cap() {
    let fixes = [
        (
            fixture("A d=6 (3,0)", 6, 3, 0, (2, 3), 0x53_CA0A),
            vec![0usize, 2, 4],
        ),
        (
            fixture("A d=6 (3,0)", 6, 3, 0, (2, 3), 0x53_CA0A),
            vec![1usize, 3],
        ),
        (
            fixture("B d=5 (2,2)", 5, 2, 2, (2, 3), 0x53_CA0B),
            vec![0usize, 2],
        ),
        (
            fixture("B d=5 (2,2)", 5, 2, 2, (2, 3), 0x53_CA0B),
            vec![1usize, 3],
        ),
    ];
    for (fix, bonds) in fixes {
        let tag = format!("{} bonds {bonds:?}", fix.name);
        let (y_num, _, _) = run_layer(&fix.x, &bonds, Pooled::NONE, 1e-10);
        let schmidt: Vec<(usize, usize)> = bonds
            .iter()
            .map(|&k| (k, y_num.ranks[k + 1].total()))
            .collect();
        let m = schmidt.iter().map(|&(_, r)| r).max().unwrap();
        let dcap = m / 2;
        println!("tord_s3 {tag}: numerical Schmidt ranks {schmidt:?}, cap D = {dcap}");
        assert!(dcap >= 1 && dcap < m, "{tag}: anti-vacuity: M = {m}");

        let (y_e, _, info_e) = run_layer(&fix.x, &bonds, Pooled::NONE, 0.0);
        assert_eq!(info_e.discarded, 0.0);
        let (y_c, u_c, info_c) = run_layer(&fix.x, &bonds, Pooled(dcap), 0.0);
        assert_layer_postconditions(&tag, &fix.x, &y_c, &bonds, Some(dcap));
        assert!(
            info_c.max_bond_dim <= dcap,
            "{tag}: max_bond_dim {} > D",
            info_c.max_bond_dim
        );
        assert!(
            info_c.discarded > 0.0 && info_c.discarded < 1.0,
            "{tag}: discarded {}",
            info_c.discarded
        );
        let mut tr = Signed::id(fix.space.d);
        tr.apply_layer(&bonds);
        assert_eq!(u_c, tr.matrix(), "{tag}: u does not depend on the cap");
        let inf = infidelity(&y_e, &y_c);
        println!(
            "tord_s3 {tag}: discarded = {:.6e}, 1 − fidelity = {inf:.6e}, |Δ| = {:e}",
            info_c.discarded,
            (inf - info_c.discarded).abs()
        );
        assert!(
            (inf - info_c.discarded).abs() <= 1e-12,
            "{tag}: layer fidelity identity"
        );

        let mut singles = Vec::new();
        for &(k, r) in &schmidt {
            if r <= dcap {
                continue;
            }
            let (y_e1, _, _) = run_layer(&fix.x, &[k], Pooled::NONE, 0.0);
            let (y_c1, _, info1) = run_layer(&fix.x, &[k], Pooled(dcap), 0.0);
            assert_layer_postconditions(
                &format!("{tag} single k={k}"),
                &fix.x,
                &y_c1,
                &[k],
                Some(dcap),
            );
            let inf1 = infidelity(&y_e1, &y_c1);
            println!(
                "tord_s3 {tag} single k={k}: w = {:.6e}, 1 − fidelity = {inf1:.6e}",
                info1.discarded
            );
            assert!(
                info1.discarded > 0.0,
                "{tag} k={k}: cap bound but nothing discarded"
            );
            assert!(
                (inf1 - info1.discarded).abs() <= 1e-12,
                "{tag} k={k}: single-bond identity"
            );
            singles.push(info1.discarded);
        }
        assert!(!singles.is_empty(), "{tag}: no binding bond");
    }
}

#[test]
fn tord_s3_scaled_input_and_absolute_eps() {
    const S: f64 = 3.0;
    let fixes = [
        fixture("A d=6 (3,0)", 6, 3, 0, (2, 3), 0x53_CA0A),
        fixture("B d=5 (2,2)", 5, 2, 2, (2, 3), 0x53_CA0B),
    ];
    for fix in fixes {
        let d = fix.space.d;
        let (k, m) = (0..d - 1)
            .map(|k| {
                let (y, _, _) = run_layer(&fix.x, &[k], Pooled::NONE, 1e-10);
                (k, y.ranks[k + 1].total())
            })
            .max_by_key(|&(k, m)| (m, std::cmp::Reverse(k)))
            .unwrap();
        let tag = format!("{} k={k}", fix.name);
        assert!(m >= 3, "{tag}: anti-vacuity: numerical Schmidt rank {m}");
        let mut tau = vec![1.0f64; m + 1];
        for dd in 1..m {
            tau[dd] = run_layer(&fix.x, &[k], Pooled(dd), 0.0).2.discarded;
        }
        tau[m] = 0.0;
        for dd in 1..m {
            assert!(
                tau[dd] > 1e-12 && tau[dd] < tau[dd - 1],
                "{tag}: tail τ({dd}) = {:e} not strictly decreasing above roundoff",
                tau[dd]
            );
        }
        println!("tord_s3 {tag}: m = {m}, τ = {tau:?}");
        let sig: Vec<f64> = (1..=m).map(|i| (tau[i - 1] - tau[i]).sqrt()).collect();
        let kept_sq = |th: f64| (1..=m).find(|&dd| tau[dd] <= th).unwrap();
        let kept_lin = |e: f64| {
            let mut cum = 0.0;
            let mut dropped = 0;
            for &v in sig.iter().rev() {
                cum += v;
                if cum <= e {
                    dropped += 1;
                } else {
                    break;
                }
            }
            (m - dropped).max(1)
        };
        let clear_sq = |th: f64| {
            tau.iter()
                .all(|&t| t == 0.0 || !(t / 1.1..=t * 1.1).contains(&th))
        };
        let clear_lin = |e: f64| {
            let mut cum = 0.0;
            sig.iter().rev().all(|&v| {
                cum += v;
                !(cum / 1.1..=cum * 1.1).contains(&e)
            })
        };
        let eps = (16..320)
            .map(|j| 10f64.powf(-(j as f64) / 32.0))
            .find(|&e| {
                let (e2, s2) = (e * e, S * S);
                clear_sq(e2)
                    && clear_sq(e2 / s2)
                    && clear_lin(e)
                    && kept_sq(e2) < m
                    && kept_sq(e2 / s2) != kept_sq(e2)
                    && kept_lin(e) != kept_sq(e2)
            })
            .unwrap_or_else(|| panic!("{tag}: no discriminating eps on the grid"));
        let d_unit = kept_sq(eps * eps);
        let d_scaled = kept_sq(eps * eps / (S * S));
        println!(
            "tord_s3 {tag}: eps = {eps:.3e}: absolute keeps {d_unit} (unit) / {d_scaled} (‖x‖ = \
             {S}); relative-to-‖x‖ would keep {d_unit} at ‖x‖ = {S}; Σσ reading keeps {}",
            kept_lin(eps)
        );

        let mut xs = fix.x.clone();
        xs.scale_mut(S);
        assert!((dot(&xs, &xs).sqrt() - S).abs() <= 1e-12 * S);
        let check_norm = |t: &str, y: &TtVector<f64>| {
            assert!(y.check(), "{t}: output train inconsistent");
            let n_dot = dot(y, y).sqrt();
            assert!(
                (n_dot - S).abs() <= 1e-12 * S,
                "{t}: ‖y‖ = {n_dot:.16}, input norm {S}"
            );
            assert!(
                (y.norm() - n_dot).abs() <= 1e-12 * S,
                "{t}: centre norm {} vs gauge-free {n_dot}",
                y.norm()
            );
        };
        let scaled_match = |t: &str, ys: &TtVector<f64>, y1: &TtVector<f64>| {
            let a = ys.to_dense();
            let b: Vec<f64> = y1.to_dense().iter().map(|v| S * v).collect();
            let gap = max_abs_diff_vec(&a, &b);
            println!("tord_s3 {tag} {t}: max|y_s − {S}·y_1| = {gap:.2e}");
            assert!(
                gap <= 1e-12 * S,
                "{tag} {t}: not {S}× the unit-input layer ({gap:e})"
            );
        };

        let (y1, _, info1) = run_layer(&fix.x, &[k], Pooled::NONE, eps);
        assert_layer_postconditions(&format!("{tag} eps"), &fix.x, &y1, &[k], None);
        assert_eq!(info1.max_bond_dim, d_unit, "{tag}: unit input kept count");
        assert_eq!(
            y1.ranks[k + 1].total(),
            d_unit,
            "{tag}: unit input bond rank"
        );
        assert!(
            (info1.discarded - tau[d_unit]).abs() <= 1e-12,
            "{tag}: discarded {:e} vs τ({d_unit}) = {:e}",
            info1.discarded,
            tau[d_unit]
        );

        let (ys, _, infos) = run_layer(&xs, &[k], Pooled::NONE, eps);
        check_norm(&format!("{tag} scaled eps"), &ys);
        assert_eq!(
            infos.max_bond_dim, d_scaled,
            "{tag}: scaled input at eps kept {} (absolute reading {d_scaled}, relative {d_unit})",
            infos.max_bond_dim
        );
        assert!(
            (infos.discarded - tau[d_scaled]).abs() <= 1e-12,
            "{tag}: scaled discarded {:e} vs τ({d_scaled}) = {:e}",
            infos.discarded,
            tau[d_scaled]
        );

        let (ys2, _, infos2) = run_layer(&xs, &[k], Pooled::NONE, S * eps);
        check_norm(&format!("{tag} scaled S·eps"), &ys2);
        assert_eq!(infos2.max_bond_dim, d_unit, "{tag}: scaled input at S·eps");
        assert!((infos2.discarded - info1.discarded).abs() <= 1e-12);
        scaled_match("S·eps", &ys2, &y1);

        let dcap = m / 2;
        let (yc1, _, _) = run_layer(&fix.x, &[k], Pooled(dcap), 0.0);
        let (ycs, _, infocs) = run_layer(&xs, &[k], Pooled(dcap), 0.0);
        check_norm(&format!("{tag} scaled cap {dcap}"), &ycs);
        assert!(infocs.discarded > 0.0, "{tag}: cap {dcap} does not bind");
        assert!((infocs.discarded - tau[dcap]).abs() <= 1e-12);
        scaled_match("cap", &ycs, &yc1);
    }
}

fn swap_tail_energies(x: &TtVector<f64>, k: usize) -> Vec<f64> {
    let (y, _, _) = run_layer(x, &[k], Pooled::NONE, 1e-10);
    let m = y.ranks[k + 1].total();
    let mut tau = vec![1.0f64; m + 1];
    for dd in 1..m {
        tau[dd] = run_layer(x, &[k], Pooled(dd), 0.0).2.discarded;
    }
    tau[m] = 0.0;
    tau
}

#[test]
fn tord_s3_layer_splits_share_the_full_eps() {
    let fix = fixture("B d=5 (2,2)", 5, 2, 2, (2, 3), 0x53_CA0B);
    let mut discriminated = 0;
    for bonds in [[0usize, 2], [1, 3]] {
        let tag = format!("{} bonds {bonds:?}", fix.name);
        let tau1 = swap_tail_energies(&fix.x, bonds[0]);
        let kept = |tau: &[f64], th: f64| (1..tau.len()).find(|&dd| tau[dd] <= th).unwrap();
        let clear = |tau: &[f64], th: f64| {
            tau.iter()
                .all(|&t| t == 0.0 || !(t / 1.1..=t * 1.1).contains(&th))
        };
        let pick = (16..320)
            .map(|j| 10f64.powf(-(j as f64) / 32.0))
            .find_map(|e| {
                let e2 = e * e;
                if !(clear(&tau1, e2) && clear(&tau1, e2 / 2.0)) {
                    return None;
                }
                let (ya, _, _) = run_layer(&fix.x, &bonds[..1], Pooled::NONE, e);
                let tau2 = swap_tail_energies(&ya, bonds[1]);
                if !(clear(&tau2, e2) && clear(&tau2, e2 / 2.0)) {
                    return None;
                }
                let disc = kept(&tau1, e2) != kept(&tau1, e2 / 2.0)
                    || kept(&tau2, e2) != kept(&tau2, e2 / 2.0);
                let binding = kept(&tau1, e2) < tau1.len() - 1 && kept(&tau2, e2) < tau2.len() - 1;
                (disc && binding).then_some((e, ya, tau2))
            });
        let Some((eps, ya, tau2)) = pick else {
            println!("tord_s3 {tag}: no discriminating eps on the grid");
            continue;
        };
        discriminated += 1;
        let (k1, k2) = (bonds[0], bonds[1]);
        let d1 = kept(&tau1, eps * eps);
        let d2 = kept(&tau2, eps * eps);
        println!(
            "tord_s3 {tag}: eps = {eps:.3e}: full eps keeps ({d1}, {d2}); eps/√2 would keep ({}, \
             {}); τ₁ = {tau1:?}, τ₂ = {tau2:?}",
            kept(&tau1, eps * eps / 2.0),
            kept(&tau2, eps * eps / 2.0)
        );
        assert_eq!(
            ya.ranks[k1 + 1].total(),
            d1,
            "{tag}: single-pair [{k1}] kept"
        );
        let (yb, _, _) = run_layer(&ya, &bonds[1..], Pooled::NONE, eps);
        assert_eq!(
            yb.ranks[k2 + 1].total(),
            d2,
            "{tag}: single-pair [{k2}] kept"
        );
        let (y, _, info) = run_layer(&fix.x, &bonds, Pooled::NONE, eps);
        assert_layer_postconditions(&tag, &fix.x, &y, &bonds, None);
        assert_eq!(
            y.ranks[k1 + 1].total(),
            d1,
            "{tag}: layer split at bond {} kept",
            k1 + 1
        );
        assert_eq!(
            y.ranks[k2 + 1].total(),
            d2,
            "{tag}: layer split at bond {} kept",
            k2 + 1
        );
        assert_eq!(info.max_bond_dim, d1.max(d2), "{tag}: max_bond_dim");
        let gap = max_abs_diff_vec(&y.to_dense(), &yb.to_dense());
        println!("tord_s3 {tag}: max|y_layer − y_composed| = {gap:.2e}");
        assert!(
            gap <= 1e-12,
            "{tag}: layer ≠ composed single-pair layers ({gap:e})"
        );
    }
    assert!(
        discriminated >= 1,
        "anti-vacuity: no bond pair of fixture B separates eps from eps/√2"
    );
}

#[test]
fn tord_s3_c64_layer_matches_dense() {
    let space = QnSpace::new(5, 2, 2);
    let dets = dets_for(5, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x53_0C64);
    let mut x: TtVector<C64> = randomized_state(space, 1, 2, &mut rng);
    normalize_mut(&mut x);
    let gather = |z: &TtVector<C64>| -> (Vec<f64>, Vec<f64>) {
        let full = z.to_dense();
        dets.iter()
            .map(|(ou, od)| {
                let v = full[basis_index(ou, od)];
                (v.re, v.im)
            })
            .unzip()
    };
    let (re0, im0) = gather(&x);
    assert!(
        im0.iter().any(|v| v.abs() > 1e-3),
        "anti-vacuity: imaginary part is empty"
    );
    let mut gates = DenseGates::new(5, &dets);
    for bonds in [vec![0usize, 2], vec![1usize, 3]] {
        let mut y = x.clone();
        let (u, info) = apply_swap_layer(&mut y, &bonds, Pooled::NONE, 0.0);
        assert!(y.check());
        assert_eq!(info.discarded, 0.0);
        let mut tr = Signed::id(5);
        tr.apply_layer(&bonds);
        assert_eq!(u, tr.matrix(), "C64 u is the f64 u");
        let (re, im) = gather(&y);
        let want_re = gates.apply(&bonds, re0.clone());
        let want_im = gates.apply(&bonds, im0.clone());
        let err = max_abs_diff_vec(&re, &want_re).max(max_abs_diff_vec(&im, &want_im));
        println!("tord_s3 C64 bonds {bonds:?}: max|Δ| = {err:e}");
        assert!(err <= 1e-12, "C64 bonds {bonds:?}: dense mismatch {err:e}");
    }
}

#[test]
fn tord_s3_thread_count_parity() {
    let fix = fixture("C d=6 (3,2)", 6, 3, 2, (2, 3), 0x53_7EAD);
    let pool = |n: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("pool")
    };
    let (p1, p4) = (pool(1), pool(4));
    for chi in [Pooled::NONE, Pooled(4)] {
        for l in 0..2 {
            let bonds = swap_network_layer(6, l);
            let (y1, u1, i1) = p1.install(|| run_layer(&fix.x, &bonds, chi, 0.0));
            let (y4, u4, i4) = p4.install(|| run_layer(&fix.x, &bonds, chi, 0.0));
            let (d1, d4) = (y1.to_dense(), y4.to_dense());
            let err = max_abs_diff_vec(&d1, &d4);
            let bitwise = d1.iter().zip(&d4).all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "tord_s3 threads chi={} layer {l}: max|Δ| = {err:e}, bitwise = {bitwise}, discarded {:e} / {:e}",
                chi.0, i1.discarded, i4.discarded
            );
            assert_eq!(u1, u4);
            assert_eq!(i1.max_bond_dim, i4.max_bond_dim);
            assert_eq!(y1.ranks, y4.ranks);
            assert!(
                err <= 1e-14,
                "chi={} layer {l}: 1 vs 4 threads {err:e}",
                chi.0
            );
            assert!((i1.discarded - i4.discarded).abs() <= 1e-14);
        }
    }
}

#[test]
fn tord_s3_empty_layer_is_the_identity() {
    let fix = fixture("A d=6 (3,0)", 6, 3, 0, (1, 2), 0x53_E4B7);
    let mut y = fix.x.clone();
    let (u, info) = apply_swap_layer(&mut y, &[], Pooled(3), 0.0);
    assert_eq!(u, identity_matrix(6));
    assert_eq!(
        info,
        SwapLayerInfo {
            discarded: 0.0,
            max_bond_dim: 0
        }
    );
    assert_eq!(y.ranks, fix.x.ranks);
    assert_eq!(y.ortho, fix.x.ortho);
    let bits = |z: &TtVector<f64>| z.to_dense().iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&y), bits(&fix.x), "empty layer touched x");
}

#[test]
fn tord_s3_apply_swap_layer_refusals() {
    let fix = fixture("A d=6 (3,0)", 6, 3, 0, (1, 2), 0x53_4EF0);
    let bad_bonds: [(&[usize], &str); 4] = [
        (&[0, 1], "bonds[1]"),
        (&[2, 0], "bonds[1]"),
        (&[0, 0], "bonds[1]"),
        (&[1, 5], "bonds[1]"),
    ];
    for (bonds, phrase) in bad_bonds {
        let mut y = fix.x.clone();
        assert_refused(
            &format!("apply_swap_layer {bonds:?}"),
            || {
                let _ = apply_swap_layer(&mut y, bonds, Pooled::NONE, 0.0);
            },
            &["apply_swap_layer", phrase],
        );
    }
    for eps in [-1e-3, f64::NAN, f64::INFINITY] {
        let mut y = fix.x.clone();
        assert_refused(
            &format!("apply_swap_layer eps={eps}"),
            || {
                let _ = apply_swap_layer(&mut y, &[0], Pooled::NONE, eps);
            },
            &["apply_swap_layer", "eps"],
        );
    }
    let mut z = fix.x.clone();
    z.scale_mut(0.0);
    assert_refused(
        "apply_swap_layer zero state",
        || {
            let _ = apply_swap_layer(&mut z, &[0], Pooled::NONE, 0.0);
        },
        &["apply_swap_layer", "norm"],
    );
}

#[test]
#[should_panic(expected = "different irreps")]
fn tord_s3_cross_irrep_pair_refused() {
    let orbsym: Vec<u8> = (0..6).map(|k| [2u8, 3, 4, 1][k % 4]).collect();
    let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym");
    let space = QnSpace::with_point_group(6, 3, 3, pg, Irrep::IDENTITY);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x53_C2F0);
    let mut x: TtVector<f64> = tt_randn(space, &max_ranks(&space), &mut rng);
    normalize_mut(&mut x);
    let _ = apply_swap_layer(&mut x, &[0], Pooled::NONE, 0.0);
}

#[test]
fn tord_s4_full_period_d6() {
    let d = 6;
    for fix in [
        fixture("A d=6 (3,0)", 6, 3, 0, (1, 2), 0x54_000A),
        fixture("C d=6 (3,2)", 6, 3, 2, (1, 2), 0x54_000C),
    ] {
        let ints = swap_ints(d, 0x54_1000);
        let e0 = energy(&ints, &fix.x);
        let scale = e0.abs().max(1.0);
        let mut x = fix.x.clone();
        let mut u = identity_matrix(d);
        let mut tr = Signed::id(d);
        let mut gates = DenseGates::new(d, &fix.dets);
        let mut c = dense_state_vector(&fix.dets, &fix.x);
        let mut layer_us: Vec<Vec<f64>> = Vec::new();
        for l in 0..d {
            let bonds = swap_network_layer(d, l);
            let before = x.clone();
            let (s_l, info) = apply_swap_layer(&mut x, &bonds, Pooled::NONE, 0.0);
            let tag = format!("{} layer {l}", fix.name);
            assert_layer_postconditions(&tag, &before, &x, &bonds, None);
            assert_eq!(info.discarded, 0.0, "{tag}");
            u = compose_rotations(&u, &s_l, d);
            tr.apply_layer(&bonds);
            c = gates.apply(&bonds, c);
            assert_eq!(orthogonality_defect(&u, d), 0.0, "{tag}");
            assert!(
                u.iter().all(|&v| v == 0.0 || v == 1.0 || v == -1.0),
                "{tag}: entries"
            );
            assert_eq!(u, tr.matrix(), "{tag}: U vs F2 tracker");
            if l % 2 == 1 {
                let r = l.div_ceil(2);
                let pm = permutation_matrix(&tr.arr);
                for i in 0..d * d {
                    assert!(u[i].abs() == pm[i], "{tag}: |U| support cell {i}");
                }
                let rot = rotate_integrals(&ints, &u);
                let e1 = energy(&rot, &x);
                println!(
                    "tord_s4 {} r={r}: arr {:?}, |ΔE| = {:e}",
                    fix.name,
                    tr.arr,
                    (e1 - e0).abs()
                );
                assert!(
                    (e1 - e0).abs() <= 1e-12 * scale,
                    "{tag}: closed loop {e0} → {e1}"
                );
                let (t, v) = signed_relabel(&ints, &tr);
                assert!(
                    rot.t.data.iter().zip(&t).all(|(a, b)| a == b),
                    "{tag}: t relabel"
                );
                assert!(
                    rot.v.data.iter().zip(&v).all(|(a, b)| a == b),
                    "{tag}: v relabel"
                );
                if r == 1 {
                    let rev = compose_rotations(&s_l, &layer_us[0], d);
                    let miss = (energy(&rotate_integrals(&ints, &rev), &x) - e0).abs();
                    println!("tord_s4 {}: reversed-order margin {miss:.3e}", fix.name);
                    assert!(miss > 1e-6, "{tag}: reversed composition indistinguishable");
                }
            }
            layer_us.push(s_l);
        }
        assert_eq!(tr.arr, vec![5, 4, 3, 2, 1, 0], "{}: arrangement", fix.name);
        for o in 0..d {
            let want = if o % 2 == 0 { 1.0 } else { -1.0 };
            assert!(
                u[o * d + d - 1 - o] == want,
                "{}: U[{o}][{}]",
                fix.name,
                d - 1 - o
            );
        }
        let err = max_abs_diff_vec(&dense_state_vector(&fix.dets, &x), &c);
        println!("tord_s4 {}: six layers vs dense max|Δ| = {err:e}", fix.name);
        assert!(err <= 1e-11, "{}: dense period mismatch {err:e}", fix.name);
        let (r_in, r_out) = (orbital_rdms(&fix.x), orbital_rdms(&x));
        let mut worst = 0.0f64;
        for k in 0..d {
            let (a, b) = (r_out.rho1(k), r_in.rho1(d - 1 - k));
            worst = worst.max(max_abs_diff_vec(&a, &b));
        }
        println!("tord_s4 {}: max|ρ₁(k) − ρ₁⁰(5−k)| = {worst:e}", fix.name);
        assert!(
            worst <= 1e-12,
            "{}: one-orbital RDMs not reversed",
            fix.name
        );
    }
}
