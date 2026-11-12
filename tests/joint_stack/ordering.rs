// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::path::Path;

use qntt::core::factor::dense_self_adjoint_smallest;
use qntt::integrals::{
    Integrals, IntegralsLoad, IntegralsOptions, PointGroupPolicy, load_fixture_dir,
};
use qntt::ordering::{
    FiedlerComponent, FiedlerOrder, assert_permutation, exchange_affinity, fiedler_order,
    inverse_permutation, irrep_blocked_order, permutation_matrix, permute_integrals,
};
use qntt::rotation::{compose_rotations, expm_antisym, identity_matrix, rotate_integrals};
use qntt::{C64, SparseHamiltonian, rayleigh_quotient, tt_state};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::panic_message;
use crate::util::{random_antisym_flat, random_sym_t, random_sym_v};

fn f4_order(f: &[f64], members: &[usize]) -> Vec<usize> {
    assert!(
        members.windows(2).all(|w| w[0] < w[1]),
        "members must be ascending"
    );
    let m = f.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let k = f.iter().position(|v| v.abs() == m).expect("non-empty");
    let s = if f[k] < 0.0 { -1.0 } else { 1.0 };
    let g: Vec<f64> = f.iter().map(|v| s * v).collect();
    let mut p: Vec<usize> = (0..f.len()).collect();
    p.sort_by(|&a, &b| g[a].total_cmp(&g[b]).then(members[a].cmp(&members[b])));
    p.into_iter().map(|a| members[a]).collect()
}

fn laplacian(w: &[f64], d: usize, members: &[usize]) -> Vec<f64> {
    let m = members.len();
    let mut lap = vec![0.0f64; m * m];
    for a in 0..m {
        let mut deg = 0.0;
        for b in 0..m {
            if a != b {
                let x = w[members[a] * d + members[b]];
                lap[a * m + b] = -x;
                deg += x;
            }
        }
        lap[a * m + a] = deg;
    }
    lap
}

fn margins(f: &[f64]) -> (f64, f64) {
    let mut g = f.to_vec();
    g.sort_by(|a, b| a.total_cmp(b));
    let spacing = g
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);
    let mut a: Vec<f64> = f.iter().map(|v| v.abs()).collect();
    a.sort_by(|x, y| y.total_cmp(x));
    (spacing, a[0] - a[1])
}

fn affinity(d: usize, edges: &[(usize, usize, f64)]) -> Vec<f64> {
    let mut w = vec![0.0f64; d * d];
    for &(a, b, x) in edges {
        w[a * d + b] = x;
        w[b * d + a] = x;
    }
    w
}

fn expected_component(
    w: &[f64],
    d: usize,
    members: &[usize],
    label: &str,
) -> (Vec<usize>, Vec<f64>) {
    let m = members.len();
    let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&laplacian(w, d, members), m, m.min(3));
    let (spacing, top) = margins(&evecs[1]);
    println!(
        "{label}: Fiedler margins: sorted spacing {spacing:e}, top-|f| gap {top:e}; λ = {evals:?}"
    );
    assert!(
        spacing > 1e-6 && top > 1e-6,
        "{label}: order is not roundoff-robust"
    );
    (f4_order(&evecs[1], members), evals)
}

fn assert_component(c: &FiedlerComponent, size: usize, evals: &[f64], label: &str) {
    assert_eq!(c.size, size, "{label}: component size");
    assert!(
        (c.lambda2 - evals[1]).abs() <= 1e-12,
        "{label}: λ2 {} vs {}",
        c.lambda2,
        evals[1]
    );
    if size >= 3 {
        assert!(
            (c.lambda3 - evals[2]).abs() <= 1e-12,
            "{label}: λ3 {} vs {}",
            c.lambda3,
            evals[2]
        );
    } else {
        assert!(c.lambda3.is_nan(), "{label}: m < 3 must give λ3 = NaN (F5)");
        assert!(
            !c.degenerate,
            "{label}: m < 3 must give degenerate = false (F5)"
        );
    }
}

fn assert_same_order(a: &FiedlerOrder, b: &FiedlerOrder, label: &str) {
    assert_eq!(a.perm, b.perm, "{label}: perm differs");
    assert_eq!(
        a.components.len(),
        b.components.len(),
        "{label}: component count differs"
    );
    for (x, y) in a.components.iter().zip(&b.components) {
        assert_eq!(x.size, y.size, "{label}");
        assert_eq!(x.lambda2.to_bits(), y.lambda2.to_bits(), "{label}: λ2 bits");
        assert_eq!(x.lambda3.to_bits(), y.lambda3.to_bits(), "{label}: λ3 bits");
        assert_eq!(x.degenerate, y.degenerate, "{label}");
    }
}

fn assert_f1_refusal(w: Vec<f64>, d: usize, phrase: &str, i: usize, j: usize) {
    let msg = panic_message(|| {
        let _ = fiedler_order(&w, d);
    });
    println!("F1 refusal: {msg}");
    assert!(msg.contains(phrase), "message lacks {phrase:?}: {msg}");
    let (sq, par) = (format!("[{i},{j}]"), format!("({i}, {j})"));
    assert!(
        msg.contains(&sq) || msg.contains(&par),
        "message does not name the offending pair ({i}, {j}): {msg}"
    );
}

fn load(dir: &str, opts: &IntegralsOptions) -> IntegralsLoad<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(dir);
    load_fixture_dir::<f64>(&path, opts).unwrap_or_else(|e| panic!("{dir} loads: {e:?}"))
}

fn mem_ints(d: usize, seed: u64) -> Integrals<f64> {
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

const P5: [usize; 5] = [3, 0, 4, 1, 2];
const P6: [usize; 6] = [4, 2, 0, 5, 1, 3];
const P7: [usize; 7] = [5, 2, 6, 0, 3, 1, 4];

#[test]
fn tord_a6_weighted_path_recovers_order() {
    let d = 6;
    let sigma = [3usize, 0, 5, 1, 4, 2];
    let wts = [1.0, 2.0, 0.5, 3.0, 1.5];
    let edges: Vec<_> = (0..5).map(|k| (sigma[k], sigma[k + 1], wts[k])).collect();
    let w = affinity(d, &edges);
    let all: Vec<usize> = (0..d).collect();
    let (want, evals) = expected_component(&w, d, &all, "tord_a6 path");
    let ord = fiedler_order(&w, d);
    println!("tord_a6 path: perm {:?}", ord.perm);
    assert_permutation(&ord.perm, d);
    assert_eq!(ord.perm, want, "F4 order of the in-test Fiedler vector");
    let rev: Vec<usize> = sigma.iter().rev().copied().collect();
    assert!(
        ord.perm == sigma || ord.perm == rev,
        "path order not recovered: {:?}",
        ord.perm
    );
    assert_eq!(ord.components.len(), 1);
    assert_component(&ord.components[0], d, &evals, "tord_a6 path");
    assert!(
        !ord.components[0].degenerate,
        "a weighted path has a simple λ2"
    );
}

fn all_perms(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut out = Vec::new();
    for rest in all_perms(n - 1) {
        for pos in 0..=rest.len() {
            let mut p: Vec<usize> = rest.iter().map(|&x| x + 1).collect();
            p.insert(pos, 0);
            out.push(p);
        }
    }
    out.sort();
    out
}

#[test]
fn tord_a6_relabelling_equivariance() {
    let d = 6;
    let sigma = [3usize, 0, 5, 1, 4, 2];
    let wts = [1.0, 2.0, 0.5, 3.0, 1.5];
    let edges: Vec<_> = (0..5).map(|k| (sigma[k], sigma[k + 1], wts[k])).collect();
    let w = affinity(d, &edges);
    let base = fiedler_order(&w, d).perm;
    let all: Vec<usize> = (0..d).collect();
    let mut negative_pivots = 0;
    let perms = all_perms(d);
    assert_eq!(perms.len(), 720);
    for q in &perms {
        let mut wq = vec![0.0f64; d * d];
        for a in 0..d {
            for b in 0..d {
                wq[a * d + b] = w[q[a] * d + q[b]];
            }
        }
        let (_, evecs) = dense_self_adjoint_smallest::<f64>(&laplacian(&wq, d, &all), d, 3);
        let f = &evecs[1];
        let m = f.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        if f[f.iter().position(|v| v.abs() == m).unwrap()] < 0.0 {
            negative_pivots += 1;
        }
        let pq = fiedler_order(&wq, d).perm;
        let mapped: Vec<usize> = pq.iter().map(|&a| q[a]).collect();
        assert_eq!(mapped, base, "relabelling {q:?}: order is not equivariant");
    }
    println!("tord_a6 equivariance: {negative_pivots} of 720 relabellings had a negative pivot");
}

#[test]
fn tord_a6_components_interleaved_singletons() {
    let d = 8;
    let w = affinity(d, &[(6, 1, 1.0), (1, 7, 2.5), (7, 3, 0.7), (4, 5, 1.3)]);
    let block_a = [1usize, 3, 6, 7];
    let (want_a, evals_a) = expected_component(&w, d, &block_a, "tord_a6 block A");
    let ord = fiedler_order(&w, d);
    println!("tord_a6 components: perm {:?}", ord.perm);
    assert_permutation(&ord.perm, d);
    assert_eq!(ord.perm[0], 0, "singleton {{0}} first");
    assert_eq!(&ord.perm[1..5], &want_a[..], "block A in F4 order, second");
    assert_eq!(ord.perm[5], 2, "singleton {{2}} third");
    let mut tail = ord.perm[6..].to_vec();
    tail.sort_unstable();
    assert_eq!(tail, vec![4, 5], "block B last");
    assert_eq!(
        ord.components.len(),
        2,
        "only components of size ≥ 2 are listed"
    );
    assert_component(&ord.components[0], 4, &evals_a, "tord_a6 block A");
    assert!(!ord.components[0].degenerate);
    assert_component(&ord.components[1], 2, &[0.0, 2.6], "tord_a6 block B");
}

#[test]
fn tord_a6_empty_graph_is_identity() {
    for d in [0usize, 1, 5] {
        let ord = fiedler_order(&vec![0.0; d * d], d);
        assert_eq!(
            ord.perm,
            (0..d).collect::<Vec<_>>(),
            "d={d}: empty graph must give identity"
        );
        assert!(
            ord.components.is_empty(),
            "d={d}: singletons are not listed"
        );
    }
}

#[test]
fn tord_a6_degenerate_cycle_and_complete_graph() {
    let c6: Vec<_> = (0..6).map(|i| (i, (i + 1) % 6, 1.0)).collect();
    let k5: Vec<_> = (0..5)
        .flat_map(|i| ((i + 1)..5).map(move |j| (i, j, 1.0)))
        .collect();
    for (name, d, edges, lam) in [("C6", 6usize, c6, 1.0f64), ("K5", 5, k5, 5.0)] {
        let ord = fiedler_order(&affinity(d, &edges), d);
        println!("tord_a6 {name}: perm {:?} (not asserted)", ord.perm);
        assert_permutation(&ord.perm, d);
        assert_eq!(ord.components.len(), 1);
        let c = &ord.components[0];
        assert_eq!(c.size, d);
        assert!(
            (c.lambda2 - lam).abs() <= 1e-12 && (c.lambda3 - lam).abs() <= 1e-12,
            "{name}: λ"
        );
        assert!(
            c.degenerate,
            "{name}: λ2 = λ3 must be flagged degenerate (F5)"
        );
    }
}

#[test]
fn tord_a6_degeneracy_flag_is_scale_covariant() {
    let sigma = [3usize, 0, 5, 1, 4, 2];
    let wts = [1.0, 2.0, 0.5, 3.0, 1.5];
    let path: Vec<_> = (0..5).map(|k| (sigma[k], sigma[k + 1], wts[k])).collect();
    let c6: Vec<_> = (0..6).map(|i| (i, (i + 1) % 6, 1.0)).collect();
    for (name, edges, perm_stable) in [("path", path, true), ("C6", c6, false)] {
        let w = affinity(6, &edges);
        let base = fiedler_order(&w, 6);
        for c in [1e-9f64, 1e9] {
            let wc: Vec<f64> = w.iter().map(|x| c * x).collect();
            let ord = fiedler_order(&wc, 6);
            let (b, o) = (&base.components[0], &ord.components[0]);
            println!(
                "tord_a6 scale {name} c={c:e}: λ2 {:e}, λ3 {:e}, λ3−λ2 {:e}, degenerate {}",
                o.lambda2,
                o.lambda3,
                o.lambda3 - o.lambda2,
                o.degenerate
            );
            assert_eq!(ord.components.len(), 1, "{name} c={c:e}");
            assert_eq!(
                o.degenerate, b.degenerate,
                "{name} c={c:e}: degenerate flag not scale-invariant (F5)"
            );
            for (got, want, what) in [(o.lambda2, b.lambda2, "λ2"), (o.lambda3, b.lambda3, "λ3")]
            {
                assert!(
                    (got - c * want).abs() <= 1e-12 * c * want.abs().max(1.0),
                    "{name} c={c:e}: {what} {got:e} vs c·{want:e}"
                );
            }
            if perm_stable {
                assert_eq!(
                    ord.perm, base.perm,
                    "{name} c={c:e}: perm changed under scaling"
                );
            }
        }
    }
}

#[test]
fn tord_a6_degeneracy_is_relative_to_lambda_max() {
    let d = 12;
    let (eps, eta) = (1e-4f64, 3e-6f64);
    let mut edges = Vec::new();
    for blk in 0..3 {
        for a in 0..4 {
            for b in (a + 1)..4 {
                edges.push((4 * blk + a, 4 * blk + b, 1.0));
            }
        }
    }
    edges.push((3, 4, eps));
    edges.push((7, 8, eps));
    edges.push((11, 0, eps * (1.0 + eta)));
    let w = affinity(d, &edges);
    let all: Vec<usize> = (0..d).collect();
    let (evals, _) = dense_self_adjoint_smallest::<f64>(&laplacian(&w, d, &all), d, 0);
    let (l2, l3, lmax) = (evals[1], evals[2], evals[d - 1]);
    let gap = l3 - l2;
    println!(
        "tord_a6 λmax-relative: λ2 {l2:e}, λ3 {l3:e}, gap {gap:e}, 1e-8·λ2 {:e}, 1e-8·λmax {:e}",
        1e-8 * l2,
        1e-8 * lmax
    );
    assert!(
        gap > 100.0 * 1e-8 * l3 && gap < 1e-8 * lmax / 100.0,
        "fixture gap {gap:e} is not inside the discriminating window"
    );
    let ord = fiedler_order(&w, d);
    assert_eq!(ord.components.len(), 1);
    assert!(
        ord.components[0].degenerate,
        "gap {gap:e} ≤ 1e-8·λ_max = {:e} must be flagged degenerate (F5)",
        1e-8 * lmax
    );
}

#[test]
fn tord_a6_rejects_asymmetric_w() {
    let mut w = affinity(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
    w[2 * 3 + 1] = 0.5;
    assert_f1_refusal(w, 3, "is not symmetric", 1, 2);
}

#[test]
fn tord_a6_rejects_negative_w() {
    let w = affinity(3, &[(0, 1, 1.0), (1, 2, -1.0)]);
    assert_f1_refusal(w, 3, "is negative", 1, 2);
}

#[test]
fn tord_a6_roundoff_negatives_are_no_edge() {
    let d = 6;
    let sigma = [3usize, 0, 5, 1, 4, 2];
    let wts = [1.0, 2.0, 0.5, 3.0, 1.5];
    let edges: Vec<_> = (0..5).map(|k| (sigma[k], sigma[k + 1], wts[k])).collect();
    let base = affinity(d, &edges);
    let reference = fiedler_order(&base, d);

    let mut w = base.clone();
    w[3 * d + 2] = -3e-13;
    w[2 * d + 3] = -3e-13;
    w[1] = -3e-13;
    assert_same_order(&reference, &fiedler_order(&w, d), "in-band negatives");

    let two = affinity(d, &[(0, 1, 1.0), (1, 2, 2.0), (3, 4, 1.5), (4, 5, 0.5)]);
    let mut w = two.clone();
    w[2 * d + 3] = -1e-13;
    w[3 * d + 2] = -1e-13;
    let ord = fiedler_order(&w, d);
    assert_eq!(
        ord.components.len(),
        2,
        "an in-band negative joined two components"
    );
    assert_same_order(&fiedler_order(&two, d), &ord, "in-band cross entry");

    let mut w = base.clone();
    w[3 * d + 2] = -3e-11;
    w[2 * d + 3] = -3e-11;
    assert_f1_refusal(w, d, "is negative", 2, 3);
}

#[test]
fn tord_a6_accepts_orbital_entanglement_mutual() {
    use qntt::entropy::orbital_entanglement;
    use qntt::rdm::orbital_rdms;
    use rand::RngExt;

    let d = 6;
    let flags = |k: usize, l: usize| {
        let mut f = vec![false; d];
        f[k] = true;
        f[l] = true;
        f
    };
    let none = vec![false; d];
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA6_E2F1);
    let (mut negatives, mut min_cross) = (0usize, f64::INFINITY);
    for _ in 0..20 {
        let alpha: Vec<f64> = (0..3).map(|_| rng.random_range(0.2..1.0)).collect();
        let beta: Vec<f64> = (0..3).map(|_| rng.random_range(0.2..1.0)).collect();
        let mut x = &tt_state::<f64>(&flags(0, 3), &none) * (alpha[0] * beta[0]);
        for a in 0..3 {
            for b in 0..3 {
                if (a, b) != (0, 0) {
                    x = &x + &(&tt_state::<f64>(&flags(a, 3 + b), &none) * (alpha[a] * beta[b]));
                }
            }
        }
        let ent = orbital_entanglement(&orbital_rdms(&x));
        for i in 0..d {
            for j in 0..d {
                if i == j {
                    continue;
                }
                let v = ent.mutual[i * d + j];
                if (i < 3) == (j < 3) {
                    assert!(v > 1e-3, "within-block I[{i},{j}] = {v:e} not positive");
                } else {
                    assert!(v.abs() <= 1e-12, "cross-block I[{i},{j}] = {v:e} not ~0");
                    negatives += usize::from(v < 0.0);
                    min_cross = min_cross.min(v);
                }
            }
        }
        let ord = fiedler_order(&ent.mutual, d);
        assert_permutation(&ord.perm, d);
    }
    println!(
        "tord_a6 E2→F1: {negatives} negative cross-block entries over 20 states, min {min_cross:e}"
    );
}

#[test]
fn tord_a6_rejects_non_finite_w() {
    let w = affinity(3, &[(0, 1, 1.0), (1, 2, f64::NAN)]);
    assert_f1_refusal(w, 3, "is not finite", 1, 2);
    let w = affinity(3, &[(0, 1, 1.0), (1, 2, f64::INFINITY)]);
    assert_f1_refusal(w, 3, "is not finite", 1, 2);
}

#[test]
#[should_panic]
fn tord_a6_rejects_wrong_length_w() {
    let _ = fiedler_order(&[0.0; 8], 3);
}

#[test]
fn tord_a6_tolerance_and_ignored_diagonal() {
    let d = 6;
    let sigma = [3usize, 0, 5, 1, 4, 2];
    let wts = [1.0, 2.0, 0.5, 3.0, 1.5];
    let edges: Vec<_> = (0..5).map(|k| (sigma[k], sigma[k + 1], wts[k])).collect();
    let base = affinity(d, &edges);
    let mut near = base.clone();
    near[sigma[3] * d + sigma[4]] += 1e-14 * 3.0;
    let ord = fiedler_order(&near, d);
    assert_permutation(&ord.perm, d);

    let ints = mem_ints(5, 0xA6_0005);
    let wx = exchange_affinity(&ints);
    let asym = (0..5)
        .flat_map(|i| (0..5).map(move |j| (i, j)))
        .fold(0.0f64, |a, (i, j)| {
            a.max((wx[i * 5 + j] - wx[j * 5 + i]).abs())
        });
    println!("tord_a6: exchange_affinity(random_sym_v) max asymmetry {asym:e}");
    let ord = fiedler_order(&wx, 5);
    assert_permutation(&ord.perm, 5);

    let reference = fiedler_order(&base, d);
    for diag in [7.0, -3.0, f64::NAN] {
        let mut w = base.clone();
        for i in 0..d {
            w[i * d + i] = diag;
        }
        assert_same_order(
            &reference,
            &fiedler_order(&w, d),
            &format!("diagonal {diag}"),
        );
    }
}

#[test]
fn tord_a7_exchange_graph_h2o_matches_recipe() {
    let ld = load("h2o_sto3g", &IntegralsOptions::default());
    let ints = &ld.ints;
    let d = ints.norb;
    assert_eq!(d, 7);
    let w = exchange_affinity(ints);
    assert_eq!(w.len(), d * d);
    for i in 0..d {
        for j in 0..d {
            let want = if i == j {
                0.0
            } else {
                ints.v.get(&[i, j, j, i]).abs()
            };
            assert_eq!(w[i * d + j], want, "exchange_affinity[{i}][{j}]");
        }
    }
    let mut lap = vec![0.0f64; d * d];
    for i in 0..d {
        let mut rowsum = 0.0;
        for j in 0..d {
            if i != j {
                let kij = ints.v.get(&[i, j, j, i]).abs();
                lap[i * d + j] = -kij;
                rowsum += kij;
            }
        }
        lap[i * d + i] = rowsum;
    }
    let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&lap, d, 3);
    let fied = &evecs[1];
    let (spacing, top) = margins(fied);
    println!("tord_a7: margins {spacing:e} / {top:e}; λ = {evals:?}");
    assert!(
        spacing > 1e-6 && top > 1e-6,
        "h2o exchange-graph order is not roundoff-robust"
    );
    let mut recipe: Vec<usize> = (0..d).collect();
    recipe.sort_by(|&a, &b| fied[a].partial_cmp(&fied[b]).unwrap());
    let rev: Vec<usize> = recipe.iter().rev().copied().collect();
    let all: Vec<usize> = (0..d).collect();

    let ord = fiedler_order(&w, d);
    println!("tord_a7: perm {:?}, recipe {recipe:?}", ord.perm);
    assert_permutation(&ord.perm, d);
    assert_eq!(ord.components.len(), 1);
    assert_component(&ord.components[0], d, &evals, "tord_a7");
    assert!(!ord.components[0].degenerate);
    assert_eq!(
        ord.perm,
        f4_order(fied, &all),
        "F4 sort of the recipe's Fiedler vector"
    );
    assert!(
        ord.perm == recipe || ord.perm == rev,
        "not ± the recipe order"
    );
}

#[test]
fn tord_a7_exchange_affinity_c64_real_data() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h2o_sto3g");
    let lc =
        load_fixture_dir::<C64>(&path, &IntegralsOptions::default()).expect("h2o_sto3g as C64");
    let lf = load("h2o_sto3g", &IntegralsOptions::default());
    let (wc, wf) = (exchange_affinity(&lc.ints), exchange_affinity(&lf.ints));
    assert_eq!(wc.len(), wf.len());
    for k in 0..wf.len() {
        assert!(
            (wc[k] - wf[k]).abs() <= 2.0 * f64::EPSILON * wf[k],
            "C64 vs f64 at {k}"
        );
    }
}

#[test]
fn tord_a8_permute_integrals_is_a_bitwise_relabelling() {
    let ld = load("lih_sto3g", &IntegralsOptions::default());
    let ints = &ld.ints;
    let d = ints.norb;
    let p = P6;
    assert_ne!(
        inverse_permutation(&p),
        p.to_vec(),
        "P6 must not be an involution"
    );
    assert_eq!(ints.orbsym.len(), d, "lih_sto3g carries a raw ORBSYM");
    let out = permute_integrals(ints, &p);
    assert_eq!((out.t.d, out.t.rank, out.v.d, out.v.rank), (d, 2, d, 4));
    for a in 0..d {
        for b in 0..d {
            assert_eq!(
                out.t.get(&[a, b]).to_bits(),
                ints.t.get(&[p[a], p[b]]).to_bits(),
                "t′[{a},{b}]"
            );
            for c in 0..d {
                for e in 0..d {
                    assert_eq!(
                        out.v.get(&[a, b, c, e]).to_bits(),
                        ints.v.get(&[p[a], p[b], p[c], p[e]]).to_bits(),
                        "v′[{a},{b},{c},{e}]"
                    );
                }
            }
        }
    }
    let want_sym: Vec<i64> = p.iter().map(|&o| ints.orbsym[o]).collect();
    assert_eq!(out.orbsym, want_sym, "orbsym′[a] = orbsym[perm[a]]");
    assert_ne!(
        out.orbsym, ints.orbsym,
        "relabelling must be visible in orbsym"
    );
    assert_eq!(out.e0.to_bits(), ints.e0.to_bits());
    assert_eq!(
        (out.norb, out.nelec, out.ms2, out.isym),
        (ints.norb, ints.nelec, ints.ms2, ints.isym)
    );

    let mem = mem_ints(5, 0xA8_0001);
    assert!(
        permute_integrals(&mem, &P5).orbsym.is_empty(),
        "empty orbsym stays empty"
    );
}

#[test]
fn tord_a8_permute_integrals_typed_orbsym() {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let ld = load("h2o_c2v", &opts);
    let ints = &ld.ints;
    let d = ints.norb;
    let p = P7;
    let out = permute_integrals(ints, &p);
    let pg = ints.point_group(None).expect("original resolves").pg;
    let pg_out = out.point_group(None).expect("permuted resolves").pg;
    for a in 0..d {
        assert_eq!(
            pg_out.irrep_of(a),
            pg.irrep_of(p[a]),
            "irrep′({a}) = irrep(perm[{a}])"
        );
        assert_eq!(out.orbsym[a], ints.orbsym[p[a]]);
    }
    assert_ne!(out.orbsym, ints.orbsym, "P7 must move irreps");
    let stale = rotate_integrals(ints, &permutation_matrix(&p));
    assert_eq!(
        stale.orbsym, ints.orbsym,
        "rotate_integrals passes orbsym through verbatim"
    );
}

#[test]
fn tord_a8_permutation_matrix_matches_rotate_integrals() {
    let d = 5;
    let p = P5;
    assert_ne!(inverse_permutation(&p), p.to_vec());
    let pm = permutation_matrix(&p);
    assert_eq!(pm.len(), d * d);
    for old in 0..d {
        for new in 0..d {
            let want = if old == p[new] { 1.0 } else { 0.0 };
            assert_eq!(pm[old * d + new], want, "P[{old}][{new}]");
        }
    }
    assert_eq!(permutation_matrix(&[0, 1, 2, 3, 4]), identity_matrix(d));
    let ints = mem_ints(d, 0xA8_0003);
    let rot = rotate_integrals(&ints, &pm);
    let per = permute_integrals(&ints, &p);
    for k in 0..d * d {
        assert!(
            rot.t.data[k] == per.t.data[k],
            "t cell {k}: {} vs {}",
            rot.t.data[k],
            per.t.data[k]
        );
    }
    for k in 0..d * d * d * d {
        assert!(
            rot.v.data[k] == per.v.data[k],
            "v cell {k}: {} vs {}",
            rot.v.data[k],
            per.v.data[k]
        );
    }
    assert_eq!(rot.e0, per.e0);
    assert_eq!(
        (rot.norb, rot.nelec, rot.ms2, rot.isym),
        (per.norb, per.nelec, per.ms2, per.isym)
    );
    assert!(rot.orbsym.is_empty() && per.orbsym.is_empty());
}

#[test]
fn tord_a8_composition_with_rotation() {
    let d = 5;
    let p = P5;
    let ints = mem_ints(d, 0xA8_0004);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA8_0044);
    let u = expm_antisym(&random_antisym_flat(d, &mut rng), d);
    let rotated = rotate_integrals(&ints, &u);
    let moved = (0..d * d).fold(0.0f64, |a, k| {
        a.max((rotated.t.data[k] - ints.t.data[k]).abs())
    });
    assert!(
        moved > 1e-3,
        "u is (numerically) the identity: max|Δt| = {moved:e}"
    );
    let lhs = rotate_integrals(&ints, &compose_rotations(&u, &permutation_matrix(&p), d));
    let rhs = permute_integrals(&rotated, &p);
    let mut worst = 0.0f64;
    for (a, b) in lhs
        .t
        .data
        .iter()
        .zip(&rhs.t.data)
        .chain(lhs.v.data.iter().zip(&rhs.v.data))
    {
        worst = worst.max((a - b).abs());
    }
    println!("tord_a8 P4: max |Δ| = {worst:e}");
    assert!(worst <= 1e-12, "P4 composition defect {worst:e}");
}

#[test]
fn tord_a8_inverse_and_round_trip() {
    let d = 6;
    let p = P6;
    let inv = inverse_permutation(&p);
    assert_permutation(&inv, d);
    for k in 0..d {
        assert_eq!(inv[p[k]], k, "inv[p[{k}]]");
        assert_eq!(p[inv[k]], k, "p[inv[{k}]]");
    }
    assert_eq!(inverse_permutation(&[]), Vec::<usize>::new());
    assert_eq!(
        compose_rotations(&permutation_matrix(&p), &permutation_matrix(&inv), d),
        identity_matrix(d)
    );

    let ld = load("lih_sto3g", &IntegralsOptions::default());
    let ints = &ld.ints;
    let back = permute_integrals(&permute_integrals(ints, &p), &inv);
    let bits = |x: &[f64]| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&back.t.data), bits(&ints.t.data), "round trip t");
    assert_eq!(bits(&back.v.data), bits(&ints.v.data), "round trip v");
    assert_eq!(back.orbsym, ints.orbsym, "round trip orbsym");

    let p2 = [1usize, 5, 3, 0, 4, 2];
    let comp: Vec<usize> = (0..d).map(|a| p[p2[a]]).collect();
    let twice = permute_integrals(&permute_integrals(ints, &p), &p2);
    let once = permute_integrals(ints, &comp);
    assert_eq!(bits(&twice.t.data), bits(&once.t.data), "composition law t");
    assert_eq!(bits(&twice.v.data), bits(&once.v.data), "composition law v");
    assert_eq!(twice.orbsym, once.orbsym, "composition law orbsym");
}

#[test]
fn tord_a8_hf_rayleigh_quotient_invariant() {
    let ld = load("lih_sto3g", &IntegralsOptions::default());
    let meta = ld.meta.as_ref().expect("lih_sto3g ships meta.txt");
    let ints = &ld.ints;
    let space = ld.space;
    let d = ints.norb;
    let h = SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    let occ: Vec<bool> = (0..d).map(|k| k < space.nup).collect();
    let e_ref = rayleigh_quotient(&h, &tt_state::<f64>(&occ, &occ));
    let e_hf = meta.e_hf.expect("lih_sto3g meta carries e_hf");
    assert!(
        (e_ref + meta.e_core - e_hf).abs() <= 1e-8,
        "anchor: {} vs {e_hf}",
        e_ref + meta.e_core
    );

    let fiedler = fiedler_order(&exchange_affinity(ints), d).perm;
    println!("tord_a8: lih exchange-graph Fiedler perm {fiedler:?} (not asserted)");
    for (name, p) in [("P6", P6.to_vec()), ("fiedler", fiedler)] {
        let pi = permute_integrals(ints, &p);
        let hp = SparseHamiltonian::new(space, &pi.t, &pi.v, None, f64::EPSILON);
        let occp: Vec<bool> = (0..d).map(|k| occ[p[k]]).collect();
        let e = rayleigh_quotient(&hp, &tt_state::<f64>(&occp, &occp));
        println!("tord_a8 {name}: E_HF′ − E_HF = {:e}", e - e_ref);
        assert!(
            (e - e_ref).abs() <= 1e-10,
            "{name}: HF energy {e} vs {e_ref}"
        );
    }
}

#[test]
#[should_panic(expected = "has length 2")]
fn tord_a8_assert_permutation_rejects_length() {
    assert_permutation(&[0, 1], 3);
}

#[test]
#[should_panic(expected = "perm[1] = 3 is out of range")]
fn tord_a8_assert_permutation_rejects_out_of_range() {
    assert_permutation(&[0, 3, 1], 3);
}

#[test]
#[should_panic(expected = "perm[2] = 1 duplicates perm[1]")]
fn tord_a8_assert_permutation_rejects_duplicate() {
    assert_permutation(&[0, 1, 1], 3);
}

#[test]
#[should_panic(expected = "perm[1] = 0 duplicates perm[0]")]
fn tord_a8_assert_permutation_first_offender_duplicate() {
    assert_permutation(&[0, 0, 5], 3);
}

#[test]
#[should_panic(expected = "perm[1] = 5 is out of range")]
fn tord_a8_assert_permutation_first_offender_range() {
    assert_permutation(&[0, 5, 0], 3);
}

#[test]
fn tord_a8_assert_permutation_accepts() {
    assert_permutation(&[2, 0, 1], 3);
    assert_permutation(&[], 0);
    assert_permutation(&P7, 7);
}

#[test]
#[should_panic(expected = "perm[1] = 0 duplicates perm[0]")]
fn tord_a8_permute_integrals_refuses_duplicate() {
    let _ = permute_integrals(&mem_ints(3, 0xA8_0001), &[0, 0, 2]);
}

#[test]
#[should_panic(expected = "perm has length 2, expected d = 3")]
fn tord_a8_permute_integrals_refuses_length() {
    let _ = permute_integrals(&mem_ints(3, 0xA8_0002), &[1, 0]);
}

#[test]
#[should_panic(expected = "perm[2] = 3 is out of range")]
fn tord_a8_permutation_matrix_refuses_out_of_range() {
    let _ = permutation_matrix(&[1, 0, 3]);
}

#[test]
#[should_panic(expected = "perm[1] = 0 duplicates perm[0]")]
fn tord_a8_inverse_permutation_refuses_duplicate() {
    let _ = inverse_permutation(&[0, 0, 2]);
}

#[test]
#[should_panic(expected = "orbsym has 2 entries, expected 0 or d = 3")]
fn tord_a8_permute_integrals_refuses_short_orbsym() {
    let mut ints = mem_ints(3, 0xA8_0003);
    ints.orbsym = vec![1, 1];
    let _ = permute_integrals(&ints, &[2, 0, 1]);
}

#[test]
#[should_panic(expected = "orbsym has 4 entries, expected 0 or d = 3")]
fn tord_a8_permute_integrals_refuses_long_orbsym() {
    let mut ints = mem_ints(3, 0xA8_0004);
    ints.orbsym = vec![1, 1, 1, 1];
    let _ = permute_integrals(&ints, &[2, 0, 1]);
}

#[test]
#[should_panic(expected = "t must be rank-2 over d=3")]
fn tord_a8_permute_integrals_refuses_mismatched_t() {
    let mut ints = mem_ints(3, 0xA8_0005);
    ints.t = mem_ints(4, 0xA8_0006).t;
    let _ = permute_integrals(&ints, &[2, 0, 1]);
}

#[test]
#[should_panic(expected = "v must be rank-4 over d=3")]
fn tord_a8_permute_integrals_refuses_mismatched_v() {
    let mut ints = mem_ints(3, 0xA8_0007);
    ints.v = mem_ints(4, 0xA8_0008).v;
    let _ = permute_integrals(&ints, &[2, 0, 1]);
}

#[test]
fn tord_a9_irrep_blocked_order_on_typed_h2o_sto3g() {
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let ld = load("h2o_sto3g", &opts);
    let (ints, pg, kind) = (&ld.ints, ld.space.pg, ld.pg.kind);
    let d = ints.norb;
    assert!(!pg.is_trivial(), "h2o_sto3g must resolve a point group");
    let xor = |p: qntt::PointGroup, perm: &[usize]| -> Vec<u8> {
        perm.iter().map(|&o| p.irrep_of(o).index()).collect()
    };
    let same_bonds = |p: qntt::PointGroup| {
        (0..d - 1)
            .filter(|&k| p.irrep_of(k) == p.irrep_of(k + 1))
            .count()
    };
    let id: Vec<usize> = (0..d).collect();
    assert_eq!(xor(pg, &id), vec![0, 0, 2, 0, 3, 0, 2], "fixture table");

    let perm = irrep_blocked_order(pg, d);
    assert_eq!(perm, vec![0, 1, 3, 5, 2, 6, 4]);
    let ints_b = permute_integrals(ints, &perm);
    let space_b = ints_b.qn_space(Some(kind)).expect("permuted dump resolves");
    for a in 0..d {
        assert_eq!(space_b.pg.irrep_of(a), pg.irrep_of(perm[a]), "irrep′({a})");
    }
    assert_eq!(xor(space_b.pg, &id), vec![0, 0, 0, 0, 2, 2, 3]);
    assert_eq!(
        (space_b.d, space_b.nup, space_b.ndn, space_b.target),
        (ld.space.d, ld.space.nup, ld.space.ndn, ld.space.target)
    );
    let (n0, nb) = (same_bonds(pg), same_bonds(space_b.pg));
    println!("same-irrep adjacent bonds: canonical {n0}, irrep-blocked {nb}");
    assert_eq!((n0, nb), (1, 4));
    assert_eq!(
        irrep_blocked_order(space_b.pg, d),
        id,
        "blocking the blocked table is the identity"
    );

    let pf = fiedler_order(&exchange_affinity(ints), d).perm;
    let ints_f = permute_integrals(ints, &pf);
    let pg_f = ints_f.point_group(Some(kind)).expect("resolves").pg;
    let pb = irrep_blocked_order(pg_f, d);
    let tot: Vec<usize> = (0..d).map(|a| pf[pb[a]]).collect();
    assert_permutation(&tot, d);
    let k = xor(pg, &tot);
    assert!(
        k.windows(2).all(|w| w[0] <= w[1]),
        "blocks not contiguous: {k:?}"
    );
    for r in 0..4u8 {
        let sub = |seq: &[usize]| -> Vec<usize> {
            seq.iter()
                .copied()
                .filter(|&o| pg.irrep_of(o).index() == r)
                .collect()
        };
        assert_eq!(sub(&tot), sub(&pf), "irrep {r}: Fiedler order not kept");
    }
    let twice = permute_integrals(&ints_f, &pb);
    let once = permute_integrals(ints, &tot);
    assert!(
        twice
            .t
            .data
            .iter()
            .zip(&once.t.data)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            && twice
                .v
                .data
                .iter()
                .zip(&once.v.data)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
        "relabel by pf then pb ≠ relabel by pf∘pb"
    );
    assert_eq!(twice.orbsym, once.orbsym);
}
