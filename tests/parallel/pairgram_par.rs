// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use qntt::pairgram::{gamma2_mirror, generalized_gamma2, set_gamma2_mirror};
use qntt::par::{
    nested_scope, par_min_fork_work, set_par_min_fork_work, set_vector_par, vector_par,
};
use qntt::rdm::transition_1rdm;
use qntt::solvers::riemannian::normalize_mut;
use qntt::{QnSpace, RankGrid, TtVector, e_pq, max_ranks, tt_randn};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lock;
use qntt_testkit::pg::capped_ranks;

fn capped_randn_with(
    space: QnSpace,
    lo: usize,
    hi: usize,
    seed: u64,
    edit: impl FnOnce(&mut [RankGrid]),
) -> TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut ranks = capped_ranks(&space, lo, hi, &mut rng);
    edit(&mut ranks);
    let mut x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    normalize_mut(&mut x);
    x
}

fn capped_randn(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
    capped_randn_with(space, lo, hi, seed, |_| {})
}

fn is_truncated(x: &TtVector<f64>) -> bool {
    let caps = max_ranks(&x.space);
    (0..=x.space.d).any(|b| {
        x.space
            .bond_sectors(b)
            .into_iter()
            .any(|s| x.ranks[b][s] < caps[b][s])
    })
}

struct ForceParallel {
    fork: usize,
    on: bool,
}

impl ForceParallel {
    fn new() -> Self {
        ForceParallel {
            fork: set_par_min_fork_work(0),
            on: set_vector_par(true),
        }
    }
}

impl Drop for ForceParallel {
    fn drop(&mut self) {
        set_par_min_fork_work(self.fork);
        set_vector_par(self.on);
    }
}

struct MirrorMode {
    prev: bool,
}

impl MirrorMode {
    fn new(on: bool) -> Self {
        let prev = set_gamma2_mirror(on);
        assert_eq!(gamma2_mirror(), on, "GAMMA2_MIRROR not set to {on}");
        MirrorMode { prev }
    }
}

impl Drop for MirrorMode {
    fn drop(&mut self) {
        set_gamma2_mirror(self.prev);
    }
}

const MODES: [(bool, &str); 2] = [(false, "Full"), (true, "Mirrored")];

fn pool(n: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .unwrap_or_else(|e| panic!("{n}-thread pool: {e}"))
}

fn timed<R>(f: impl FnOnce() -> R) -> (R, Duration) {
    let t = Instant::now();
    let r = f();
    (r, t.elapsed())
}

fn route(i: usize, j: usize, k: usize, l: usize) -> &'static str {
    let mt = i.max(j).max(k).max(l);
    let side = |a: usize, b: usize| match (a == b, (a < mt) as u8 + (b < mt) as u8) {
        (true, 0) => 0,
        (true, _) => 2,
        (false, 1) => 1,
        (false, 2) => 3,
        s => unreachable!("side {s:?}"),
    };
    let below = |a: usize, b: usize| {
        if a == b {
            (a < mt) as usize
        } else {
            (a < mt) as usize + (b < mt) as usize
        }
    };
    if below(i, j) + below(k, l) <= 2 {
        return match (side(i, j), side(k, l)) {
            (0, 0) => "E0",
            (1, 0) => "Bo",
            (0, 1) => "Ko",
            (2, 0) => "Bd1",
            (0, 2) => "Kd1",
            (3, 0) => "Bd2",
            (0, 3) => "Kd2",
            (1, 1) => "BoKo",
            (2, 1) => "Bd1Ko",
            (1, 2) => "BoKd1",
            s => unreachable!("terminal class {s:?} at {:?}", (i, j, k, l)),
        };
    }
    let ket_top = k == mt || l == mt;
    let (own_lo, other_hi) = if ket_top {
        (k.min(l), i.max(j))
    } else {
        (i.min(j), k.max(l))
    };
    let m = own_lo.max(other_hi);
    match (ket_top, own_lo == m, other_hi == m) {
        (true, true, false) => "T:Bd2xKopen",
        (true, true, true) => "T:BoCloseKopen",
        (true, false, true) => "T:BoKoBclose",
        (false, true, false) => "T:Kd2xBopen",
        (false, true, true) => "T:KoCloseBopen",
        (false, false, true) => "T:BoKoKclose",
        s => unreachable!("transient class {s:?} at {:?}", (i, j, k, l)),
    }
}

const ALL_CLASSES: [&str; 16] = [
    "E0",
    "Bo",
    "Ko",
    "Bd1",
    "Kd1",
    "Bd2",
    "Kd2",
    "BoKo",
    "Bd1Ko",
    "BoKd1",
    "T:Bd2xKopen",
    "T:BoCloseKopen",
    "T:BoKoBclose",
    "T:Kd2xBopen",
    "T:KoCloseBopen",
    "T:BoKoKclose",
];

fn census(g: &[f64], d: usize, label: &str) -> BTreeMap<&'static str, (usize, usize)> {
    let mut c: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let e = c.entry(route(i, j, k, l)).or_default();
                    e.0 += 1;
                    if g[((i * d + j) * d + k) * d + l] != 0.0 {
                        e.1 += 1;
                    }
                }
            }
        }
    }
    let total: usize = c.values().map(|v| v.0).sum();
    assert_eq!(total, d.pow(4), "{label}: classifier must cover d⁴ cells");
    eprintln!("{label}: census (class: cells, nonzero) = {c:?}");
    if d >= 4 {
        for class in ALL_CLASSES {
            let (cells, nz) = c.get(class).copied().unwrap_or((0, 0));
            assert!(
                nz > 0,
                "{label}: class {class} has {cells} cells but no nonzero content — the \
                 bit comparison would be vacuous for it"
            );
        }
    }
    c
}

fn assert_bits(want: &[f64], got: &[f64], d: usize, what: &str) {
    assert_eq!(want.len(), d.pow(4), "{what}: reference length");
    assert_eq!(got.len(), want.len(), "{what}: length");
    let unflat = |n: usize| (n / (d * d * d), (n / (d * d)) % d, (n / d) % d, n % d);
    let neg_zero = got.iter().position(|v| *v == 0.0 && v.is_sign_negative());
    if let Some(n) = neg_zero {
        let (i, j, k, l) = unflat(n);
        panic!(
            "{what}: cell {:?} (class {}) is −0.0 — the serial sweep only ever `+=`s into +0.0 \
             and never stores −0.0",
            (i, j, k, l),
            route(i, j, k, l)
        );
    }
    let mut first = None;
    let mut count = 0usize;
    for (n, (a, b)) in want.iter().zip(got).enumerate() {
        if a.to_bits() != b.to_bits() {
            count += 1;
            first.get_or_insert(n);
        }
    }
    if let Some(n) = first {
        let (i, j, k, l) = unflat(n);
        panic!(
            "{what}: {count} of {} cells differ in bits; first at {:?} (class {}): \
             reference {:e} ({:#018x}) vs {:e} ({:#018x})",
            want.len(),
            (i, j, k, l),
            route(i, j, k, l),
            want[n],
            want[n].to_bits(),
            got[n],
            got[n].to_bits()
        );
    }
}

fn assert_oracle(x: &TtVector<f64>, g: &[f64], label: &str) {
    let d = x.space.d;
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    let mut reference = vec![0.0f64; d.pow(4)];
    for k in 0..d {
        for l in 0..d {
            let y = e_pq(x, k, l, 1.0);
            let gam = transition_1rdm(x, &y);
            for i in 0..d {
                for j in 0..d {
                    reference[idx(i, j, k, l)] = gam.get(&[i, j]);
                }
            }
        }
    }
    let scale = reference
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1e-30);
    let (worst, at) = g
        .iter()
        .zip(&reference)
        .enumerate()
        .map(|(n, (a, b))| ((a - b).abs(), n))
        .fold((0.0f64, 0usize), |w, v| if v.0 > w.0 { v } else { w });
    eprintln!("{label}: oracle worst rel = {:e}", worst / scale);
    assert!(
        worst <= 1e-12 * scale,
        "{label}: worst |sweep − oracle| = {worst:e} at flat {at} (scale {scale:e}); \
         sweep = {:e}, oracle = {:e}",
        g[at],
        reference[at]
    );
}

struct Arms {
    reference: Vec<f64>,
    forced16: Vec<f64>,
}

fn check_pools(x: &TtVector<f64>, label: &str) -> Arms {
    let d = x.space.d;
    let (reference, w_r) = timed(|| {
        pool(1).install(|| {
            assert_eq!(rayon::current_num_threads(), 1);
            generalized_gamma2(x)
        })
    });
    assert_eq!(reference.len(), d.pow(4), "{label}: Γ₂ length");
    assert!(
        reference.iter().any(|v| *v != 0.0),
        "{label}: reference Γ₂ is identically zero — the comparison would be vacuous"
    );
    assert_bits(&reference, &reference, d, &format!("{label} R (self)"));

    let (s, w_s) = timed(|| {
        pool(16).install(|| {
            assert_eq!(rayon::current_num_threads(), 16);
            generalized_gamma2(x)
        })
    });
    assert_bits(
        &reference,
        &s,
        d,
        &format!("{label} S (16 thr, shipped floor)"),
    );

    let _force = ForceParallel::new();
    let (r0, w_r0) = timed(|| {
        pool(1).install(|| {
            assert_eq!(rayon::current_num_threads(), 1);
            generalized_gamma2(x)
        })
    });
    assert_bits(&reference, &r0, d, &format!("{label} R0 (1 thr, floor 0)"));

    let mut forced = Vec::new();
    let mut walls = Vec::new();
    for nt in [4usize, 16] {
        let (g, w) = timed(|| {
            pool(nt).install(|| {
                assert_eq!(rayon::current_num_threads(), nt, "{label}: pool size");
                assert_eq!(
                    par_min_fork_work(),
                    0,
                    "{label}: driver-fork floor not forced — the forced arm would be vacuous"
                );
                assert!(
                    vector_par(),
                    "{label}: VECTOR_PAR off — the forced arm would be vacuous"
                );
                generalized_gamma2(x)
            })
        });
        assert_bits(&reference, &g, d, &format!("{label} F{nt} (floor 0)"));
        forced.push(g);
        walls.push(w);
    }

    let (n, w_n) = timed(|| pool(16).install(|| nested_scope(|| generalized_gamma2(x))));
    assert_bits(&reference, &n, d, &format!("{label} N (16 thr, nested)"));

    eprintln!(
        "{label}: walls R {w_r:?} | S {w_s:?} | R0 {w_r0:?} | F4 {:?} | F16 {:?} | N {w_n:?}",
        walls[0], walls[1]
    );
    Arms {
        reference,
        forced16: forced.pop().expect("F16 arm"),
    }
}

#[test]
fn tpp1_d6_spinful_truncated_bitwise_across_pools() {
    let _g = lock();
    let space = QnSpace::new(6, 3, 2);
    let a = capped_randn(space, 1, 2, 0xF3_0601);
    assert!(is_truncated(&a), "A must be truncated below max_ranks");

    let mut zeroed = None;
    let b = capped_randn_with(space, 1, 2, 0xF3_0601, |ranks| {
        let secs = space.bond_sectors(2);
        let s = secs[secs.len() / 2];
        assert!(
            ranks[2][s] > 0,
            "B: the zeroed sector must be reachable (rank > 0 before the edit)"
        );
        ranks[2][s] = 0;
        zeroed = Some(s);
    });
    let s = zeroed.expect("edit ran");
    assert_eq!(b.ranks[2][s], 0, "B: sector {s:?} at bond 2 must be rank 0");
    assert!(
        b.ranks.iter().all(|r| r.iter().any(|(_, &n)| n > 0)),
        "B: every bond keeps a live sector"
    );

    for (mirror, tag) in MODES {
        let _mode = MirrorMode::new(mirror);
        let arms_a = check_pools(&a, &format!("tpp1 A d=6 (3,2) truncated [{tag}]"));
        census(&arms_a.reference, 6, &format!("tpp1 A [{tag}]"));

        let arms_b = check_pools(&b, &format!("tpp1 B d=6 (3,2) zero-rank sector [{tag}]"));
        census(&arms_b.reference, 6, &format!("tpp1 B [{tag}]"));
        assert_oracle(&b, &arms_b.forced16, &format!("tpp1 B F16 [{tag}]"));
    }
}

#[test]
fn tpp2_single_channel_bitwise_across_pools() {
    let _g = lock();
    let a = capped_randn(QnSpace::new(8, 4, 0), 1, 2, 0xF3_0801);
    let b = capped_randn(QnSpace::new(4, 0, 2), 1, 2, 0xF3_0402);

    for (mirror, tag) in MODES {
        let _mode = MirrorMode::new(mirror);
        let arms_a = check_pools(&a, &format!("tpp2 A d=8 (4,0) [{tag}]"));
        census(&arms_a.reference, 8, &format!("tpp2 A [{tag}]"));
        let zeros = arms_a.reference.iter().filter(|v| **v == 0.0).count();
        eprintln!(
            "tpp2 A [{tag}]: {zeros} exact-zero cells of {}",
            8usize.pow(4)
        );
        assert!(
            zeros > 0,
            "tpp2 A [{tag}]: no exact-zero cell — the signed-zero check would be vacuous"
        );

        let arms_b = check_pools(&b, &format!("tpp2 B d=4 (0,2) [{tag}]"));
        census(&arms_b.reference, 4, &format!("tpp2 B [{tag}]"));
    }
}

#[test]
fn tpp3_d5_full_rank_bitwise_and_oracle() {
    let _g = lock();
    let space = QnSpace::new(5, 2, 2);
    let x = capped_randn(space, 8, 8, 0xF3_0502);
    let caps = max_ranks(&space);
    assert_eq!(caps.len(), x.ranks.len(), "one cap grid per bond");
    for (b, (have, cap)) in x.ranks.iter().zip(&caps).enumerate() {
        for s in space.bond_sectors(b) {
            assert_eq!(
                have[s], cap[s],
                "bond {b} sector {s:?} must sit at its max_ranks cap"
            );
        }
    }
    for (mirror, tag) in MODES {
        let _mode = MirrorMode::new(mirror);
        let arms = check_pools(&x, &format!("tpp3 d=5 (2,2) full rank [{tag}]"));
        census(&arms.reference, 5, &format!("tpp3 [{tag}]"));
        assert_oracle(&x, &arms.forced16, &format!("tpp3 F16 [{tag}]"));
    }
}

#[test]
fn tpp4_small_d_edges_bitwise_across_pools() {
    let _g = lock();
    for (d, nup, ndn, seed) in [
        (1usize, 1usize, 1usize, 0xF3_0101u64),
        (2, 1, 1, 0xF3_0201),
        (3, 2, 1, 0xF3_0301),
    ] {
        let x = capped_randn(QnSpace::new(d, nup, ndn), 8, 8, seed);
        for (mirror, tag) in MODES {
            let _mode = MirrorMode::new(mirror);
            let label = format!("tpp4 d={d} ({nup},{ndn}) [{tag}]");
            let arms = check_pools(&x, &label);
            census(&arms.reference, d, &label);
        }
    }
}

#[test]
fn tpp5_full_path_adjoint_symmetry() {
    let _g = lock();
    let _mode = MirrorMode::new(false);
    let space = QnSpace::new(5, 3, 1);
    let x = capped_randn(space, 1, 2, 91);
    let d = space.d;
    assert!(!gamma2_mirror(), "tpp5 must run the Full walk");
    let g2 = generalized_gamma2(&x);
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    let scale = g2.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    let mut asymmetric_bits = 0usize;
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let a = g2[idx(i, j, k, l)];
                    let b = g2[idx(l, k, j, i)];
                    if a.to_bits() != b.to_bits() {
                        asymmetric_bits += 1;
                    }
                    assert!(
                        (a - b).abs() <= 1e-12 * scale,
                        "adjoint symmetry failed: Γ₂[{i},{j},{k},{l}] = {a:e} vs \
                         Γ₂[{l},{k},{j},{i}] = {b:e}"
                    );
                }
            }
        }
    }
    eprintln!(
        "tpp5: Full-path Γ₂ bitwise asymmetric in {asymmetric_bits} of {} cells",
        d.pow(4)
    );
    assert!(
        asymmetric_bits > 0,
        "tpp5: the Full output is bitwise symmetric — indistinguishable from the Mirrored walk, \
         so the switch may not be routed"
    );
}
