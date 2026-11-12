// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::prelude::ReborrowMut;
use faer::{Mat, MatRef};
use rand::{Rng, RngExt};

use super::{
    PAR_MIN_COREOPS_BLOCKS, PAR_MIN_COREOPS_TASK_WORK, PAR_MIN_COREOPS_WORK,
    core_times_adjoint_core, mul_sketch_left, mul_sketch_right, rproject, rproject_plain,
};
use crate::scalar::Scalar;
use crate::scalar::gemm_par_hook::{Hook, pool};
use crate::su2::core::{Family, Su2Core};
use crate::su2::expand::expand;
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::vector::Su2TtVector;
use crate::test_util::in_pool;
use crate::test_util::{randomized_su2_state, seeded};

const POOLS: [usize; 3] = [1, 8, 18];

fn rand_grid<R: Rng + ?Sized>(space: &Su2Space, b: usize, rng: &mut R) -> Su2RankGrid {
    let mut g = Su2RankGrid::zeros(space);
    for s in space.bond_sectors(b) {
        g[s] = rng.random_range(48..=80);
    }
    g
}

fn rand_core<R: Rng + ?Sized>(
    space: Su2Space,
    k: usize,
    rows: &Su2RankGrid,
    cols: &Su2RankGrid,
    rng: &mut R,
) -> Su2Core<f64> {
    let mut c = Su2Core::zeros(space, k, rows, cols);
    c.fill_randn(rng);
    c
}

fn rand_sketch<R: Rng + ?Sized>(
    space: Su2Space,
    b: usize,
    rows: &Su2RankGrid,
    cols: &Su2RankGrid,
    rng: &mut R,
) -> Su2Sketch<f64> {
    let mut s = Su2Sketch::zeros(space, b, rows, cols);
    for l in space.bond_sectors(b) {
        let mut m = s.block_mut(l).unwrap();
        for j in 0..m.ncols() {
            for i in 0..m.nrows() {
                *m.rb_mut().get_mut(i, j) = f64::sample_standard(rng);
            }
        }
    }
    s
}

fn assert_mat_bits(x: MatRef<'_, f64>, y: MatRef<'_, f64>, what: &str) {
    assert_eq!(
        (x.nrows(), x.ncols()),
        (y.nrows(), y.ncols()),
        "{what}: block shape"
    );
    for j in 0..x.ncols() {
        for i in 0..x.nrows() {
            assert_eq!(
                x[(i, j)].to_bits(),
                y[(i, j)].to_bits(),
                "{what}: entry ({i}, {j}) differs"
            );
        }
    }
}

fn assert_opt_bits(x: &Option<Mat<f64>>, y: &Option<Mat<f64>>, what: &str) {
    match (x, y) {
        (None, None) => {}
        (Some(a), Some(b)) => assert_mat_bits(a.as_ref(), b.as_ref(), what),
        _ => panic!("{what}: block presence differs"),
    }
}

fn assert_sketch_bits(x: &Su2Sketch<f64>, y: &Su2Sketch<f64>, what: &str) {
    assert_eq!((x.space, x.k), (y.space, y.k), "{what}: header");
    for ((s, bx), (_, by)) in x.blocks.iter().zip(y.blocks.iter()) {
        assert_opt_bits(bx, by, &format!("{what} @ {s:?}"));
    }
}

fn assert_core_bits(x: &Su2Core<f64>, y: &Su2Core<f64>, what: &str) {
    assert_eq!((x.space, x.k), (y.space, y.k), "{what}: header");
    assert_eq!(x.row_qn, y.row_qn, "{what}: row_qn");
    assert_eq!(x.col_qn, y.col_qn, "{what}: col_qn");
    assert_eq!(x.row_ranks, y.row_ranks, "{what}: row_ranks");
    assert_eq!(x.col_ranks, y.col_ranks, "{what}: col_ranks");
    let labels = Su2SectorGrid::<()>::filled(&x.space, ());
    for f in Family::ALL {
        for (s, _) in labels.iter() {
            let (bx, by) = (
                x.block(f, s).map(|m| m.to_owned()),
                y.block(f, s).map(|m| m.to_owned()),
            );
            assert_opt_bits(&bx, &by, &format!("{what} {f:?} @ {s:?}"));
        }
    }
}

fn fold_load(
    space: &Su2Space,
    k: usize,
    ra: &Su2RankGrid,
    rb: &Su2RankGrid,
    rc: &Su2RankGrid,
) -> (usize, usize) {
    let g = space.pg.irrep_of(k);
    let (mut tasks, mut work) = (0, 0);
    for l in space.bond_sectors(k) {
        if ra[l] == 0 || rb[l] == 0 {
            continue;
        }
        tasks += 1;
        for f in Family::ALL {
            if let Some(t) = f.apply(l, g)
                && space.contains(k + 1, t)
            {
                work += ra[l] * rb[l] * rc[t];
            }
        }
    }
    (tasks, work)
}

fn core_load(
    space: &Su2Space,
    k: usize,
    with_empty: bool,
    mnk: impl Fn(Su2Sector, Su2Sector) -> usize,
) -> (usize, usize) {
    let g = space.pg.irrep_of(k);
    let (mut tasks, mut work) = (0, 0);
    for l in space.bond_sectors(k) {
        for f in Family::ALL {
            if let Some(t) = f.apply(l, g)
                && space.contains(k + 1, t)
            {
                let w = mnk(l, t);
                tasks += usize::from(with_empty || w > 0);
                work += w;
            }
        }
    }
    (tasks, work)
}

fn assert_forks(what: &str, (tasks, work): (usize, usize)) {
    let per_task = work / tasks;
    eprintln!("{what}: {tasks} tasks, Σ m·n·k = {work}, per task = {per_task}");
    assert!(
        tasks > PAR_MIN_COREOPS_BLOCKS,
        "{what}: fixture must clear the task gate: {tasks} tasks"
    );
    assert!(
        work >= PAR_MIN_COREOPS_WORK,
        "{what}: fixture must clear the work gate: Σ m·n·k = {work}"
    );
    assert!(
        per_task >= PAR_MIN_COREOPS_TASK_WORK,
        "{what}: fixture must clear the per-task gate: {per_task}"
    );
}

struct Fixture {
    space: Su2Space,
    k: usize,
    ra: Su2RankGrid,
    rq: Su2RankGrid,
    rc: Su2RankGrid,
}

fn fixture(seed: u64) -> (Fixture, rand_xoshiro::Xoshiro256PlusPlus) {
    let mut rng = seeded(seed);
    let space = Su2Space::new(8, 4, 0);
    let k = 4;
    let mut ra = rand_grid(&space, k, &mut rng);
    ra[space.bond_sectors(k)[0]] = 0;
    let rq = rand_grid(&space, k, &mut rng);
    let rc = rand_grid(&space, k + 1, &mut rng);
    (
        Fixture {
            space,
            k,
            ra,
            rq,
            rc,
        },
        rng,
    )
}

#[test]
fn coreops_rproject_threaded_matches_serial_exact() {
    let (fx, mut rng) = fixture(0xC0E0_0001);
    let a = rand_core(fx.space, fx.k, &fx.ra, &fx.rc, &mut rng);
    let q = rand_core(fx.space, fx.k, &fx.rq, &fx.rc, &mut rng);
    assert_forks(
        "rproject",
        fold_load(&fx.space, fx.k, &fx.ra, &fx.rq, &fx.rc),
    );
    let runs: Vec<_> = POOLS
        .iter()
        .map(|&nt| in_pool(nt, || rproject(&a, &q)))
        .collect();
    for (nt, c) in POOLS.iter().zip(&runs).skip(1) {
        assert_sketch_bits(&runs[0], c, &format!("rproject 1 vs {nt} threads"));
    }
}

#[test]
fn coreops_core_times_adjoint_core_threaded_matches_serial_exact() {
    let (fx, mut rng) = fixture(0xC0E0_0002);
    let a = rand_core(fx.space, fx.k, &fx.ra, &fx.rc, &mut rng);
    let b = rand_core(fx.space, fx.k, &fx.rq, &fx.rc, &mut rng);
    assert_forks(
        "core_times_adjoint_core",
        fold_load(&fx.space, fx.k, &fx.ra, &fx.rq, &fx.rc),
    );
    let runs: Vec<_> = POOLS
        .iter()
        .map(|&nt| in_pool(nt, || core_times_adjoint_core(&a, &b)))
        .collect();
    for (nt, c) in POOLS.iter().zip(&runs).skip(1) {
        assert_sketch_bits(
            &runs[0],
            c,
            &format!("core_times_adjoint_core 1 vs {nt} threads"),
        );
    }
    let plain = in_pool(8, || rproject_plain(&a, &b));
    assert_sketch_bits(
        &runs[0],
        &plain,
        "rproject_plain vs core_times_adjoint_core",
    );
}

#[test]
fn coreops_mul_sketch_right_threaded_matches_serial_exact() {
    let (fx, mut rng) = fixture(0xC0E0_0003);
    let a = rand_core(fx.space, fx.k, &fx.ra, &fx.rc, &mut rng);
    let rd = rand_grid(&fx.space, fx.k + 1, &mut rng);
    let c = rand_sketch(fx.space, fx.k + 1, &fx.rc, &rd, &mut rng);
    assert_forks(
        "mul_sketch_right",
        core_load(&fx.space, fx.k, false, |l, t| fx.ra[l] * fx.rc[t] * rd[t]),
    );
    let runs: Vec<_> = POOLS
        .iter()
        .map(|&nt| in_pool(nt, || mul_sketch_right(&a, &c)))
        .collect();
    for (nt, y) in POOLS.iter().zip(&runs).skip(1) {
        assert_core_bits(&runs[0], y, &format!("mul_sketch_right 1 vs {nt} threads"));
    }
}

#[test]
fn coreops_mul_sketch_left_threaded_matches_serial_exact() {
    let (fx, mut rng) = fixture(0xC0E0_0004);
    let b = rand_core(fx.space, fx.k, &fx.ra, &fx.rc, &mut rng);
    let re = rand_grid(&fx.space, fx.k, &mut rng);
    let s = rand_sketch(fx.space, fx.k, &re, &fx.ra, &mut rng);
    assert_forks(
        "mul_sketch_left",
        core_load(&fx.space, fx.k, true, |l, t| re[l] * fx.ra[l] * fx.rc[t]),
    );
    let runs: Vec<_> = POOLS
        .iter()
        .map(|&nt| in_pool(nt, || mul_sketch_left(&s, &b)))
        .collect();
    for (nt, y) in POOLS.iter().zip(&runs).skip(1) {
        assert_core_bits(&runs[0], y, &format!("mul_sketch_left 1 vs {nt} threads"));
    }
}

fn sketch_flat(x: &Su2Sketch<f64>) -> Vec<f64> {
    let mut v = Vec::new();
    for (_, b) in x.blocks.iter() {
        if let Some(m) = b {
            for j in 0..m.ncols() {
                v.extend((0..m.nrows()).map(|i| m[(i, j)]));
            }
        }
    }
    v
}

fn core_flat(x: &Su2Core<f64>) -> Vec<f64> {
    let mut v = Vec::new();
    for m in x.store().blocks() {
        for j in 0..m.ncols() {
            v.extend((0..m.nrows()).map(|i| m[(i, j)]));
        }
    }
    v
}

#[test]
fn coreops_gemm_gate_forced_matches_serial() {
    let (fx, mut rng) = fixture(0xC0E0_0007);
    let a = rand_core(fx.space, fx.k, &fx.ra, &fx.rc, &mut rng);
    let q = rand_core(fx.space, fx.k, &fx.rq, &fx.rc, &mut rng);
    let rd = rand_grid(&fx.space, fx.k + 1, &mut rng);
    let c = rand_sketch(fx.space, fx.k + 1, &fx.rc, &rd, &mut rng);
    let re = rand_grid(&fx.space, fx.k, &mut rng);
    let s = rand_sketch(fx.space, fx.k, &re, &fx.ra, &mut rng);
    let all = || {
        [
            sketch_flat(&rproject(&a, &q)),
            sketch_flat(&core_times_adjoint_core(&a, &q)),
            core_flat(&mul_sketch_right(&a, &c)),
            core_flat(&mul_sketch_left(&s, &a)),
        ]
    };
    let names = [
        "rproject",
        "core_times_adjoint_core",
        "mul_sketch_right",
        "mul_sketch_left",
    ];
    let one = Hook::forced();
    let y1 = pool(1, &one).install(all);
    let y1r = pool(1, &one).install(all);
    assert!(
        one.calls() > 0 && one.taken() == 0,
        "a 1-thread pool threaded a GEMM"
    );
    for (n, (u, v)) in names.iter().zip(y1.iter().zip(&y1r)) {
        assert!(
            u.iter().zip(v).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{n}: 1-thread re-run differs"
        );
    }
    for nt in [8usize, 18] {
        let hook = Hook::forced();
        let yn = pool(nt, &hook).install(all);
        assert!(
            hook.taken() > 0,
            "{nt} thr: no gated GEMM threaded ({} of {} calls)",
            hook.taken(),
            hook.calls()
        );
        for (n, (u, v)) in names.iter().zip(y1.iter().zip(&yn)) {
            assert_eq!(u.len(), v.len(), "{n}: shape");
            let num: f64 = u.iter().zip(v).map(|(x, y)| (x - y).powi(2)).sum();
            let den: f64 = u.iter().map(|x| x * x).sum();
            let rel = (num / den).sqrt();
            println!("forced {n}, 1 vs {nt} threads: ‖Δ‖/‖y₁‖ = {rel:.3e}");
            assert!(rel <= 1e-12, "forced {n}, 1 vs {nt}: {rel:.3e} > 1e-12");
        }
    }
}

fn dense_all_sz(x: &Su2TtVector<f64>) -> Vec<Vec<f64>> {
    let two_s = x.space.two_s as i64;
    (-two_s..=two_s)
        .step_by(2)
        .map(|ts| expand(x, ts).to_dense())
        .collect()
}

fn rel_dev(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in x.iter().zip(y) {
        for (u, v) in a.iter().zip(b) {
            num += (u - v).powi(2);
            den += v.powi(2);
        }
    }
    (num / den).sqrt()
}

fn with_core(x: &Su2TtVector<f64>, edits: Vec<(usize, Su2Core<f64>)>) -> Su2TtVector<f64> {
    let mut cores = x.cores.clone();
    for (k, c) in edits {
        cores[k] = c;
    }
    Su2TtVector::new(x.space, cores)
}

#[test]
fn coreops_mul_sketch_left_right_insertion_dense_oracle() {
    let mut rng = seeded(0xC0E0_0005);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let b = 3;
    let r = x.ranks[b].clone();
    let s = rand_sketch(space, b, &r, &r, &mut rng);
    let xr = with_core(&x, vec![(b - 1, mul_sketch_right(x.core(b - 1), &s))]);
    let xl = with_core(&x, vec![(b, mul_sketch_left(&s, x.core(b)))]);
    let (dr, dl, d0) = (dense_all_sz(&xr), dense_all_sz(&xl), dense_all_sz(&x));
    let dev = rel_dev(&dr, &dl);
    assert!(dev < 1e-13, "right vs left insertion: rel dev {dev}");
    let moved = rel_dev(&dr, &d0);
    assert!(moved > 1e-2, "insertion is vacuous: moved {moved}");
}

#[test]
fn coreops_rproject_reconstruction_dense_oracle() {
    let mut rng = seeded(0xC0E0_0006);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let k = 3;
    let a = x.core(k).clone();
    let (q, _c, _) = a.cq();
    let d0 = dense_all_sz(&x);
    let rebuild = |c: &Su2Sketch<f64>| {
        with_core(
            &x,
            vec![(k - 1, mul_sketch_right(x.core(k - 1), c)), (k, q.clone())],
        )
    };
    let dev = rel_dev(&dense_all_sz(&rebuild(&rproject(&a, &q))), &d0);
    assert!(dev < 1e-12, "rproject reconstruction: rel dev {dev}");
    let dev_plain = rel_dev(&dense_all_sz(&rebuild(&rproject_plain(&a, &q))), &d0);
    assert!(
        dev_plain > 1e-3,
        "plain-adjoint anti-oracle is vacuous: rel dev {dev_plain}"
    );
}
