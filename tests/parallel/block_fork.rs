// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::Family;
use qntt::hamiltonian::matvec::{
    block_split_events, onesite_workspace, reset_block_split_events, set_block_fork,
    set_block_split,
};
use qntt::solvers::als::{AlsOptions, als, left_to_right_sketching, right_to_left_sketching};
use qntt::{CoeffTensor, QnSpace, RankGrid, SparseHamiltonian, TtVector};
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lock;

fn fixture(seed: u64) -> (SparseHamiltonian<f64>, TtVector<f64>, usize, QnSpace) {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    x.right_orthogonalize(true);
    let k = d / 2;
    x.move_core(k, true);
    x.check();
    (h, x, k, space)
}

fn skewed_fixture(seed: u64) -> (SparseHamiltonian<f64>, TtVector<f64>, usize, QnSpace) {
    let d = 8usize;
    let space = QnSpace::new(d, 2, 2);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            if i != j {
                v.set(&[i, j, j, i], -0.5 / (1.0 + (i as f64 - j as f64).abs()));
            }
        }
    }
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut g = RankGrid::zeros(&space);
            let secs = space.bond_sectors(b);
            let mid = secs.len() / 2;
            for (i, &sq) in secs.iter().enumerate() {
                let want = if b == 0 || b == d {
                    1
                } else if i == mid {
                    24
                } else {
                    2
                };
                g[sq] = want.min(space.max_rank(b, sq) as usize).max(1);
            }
            g
        })
        .collect();
    let mut x: TtVector<f64> = qntt::tt_randn(space, &ranks, &mut rng);
    x.right_orthogonalize(true);
    let k = d / 2;
    x.move_core(k, true);
    x.check();
    (h, x, k, space)
}

#[test]
fn sketched_hamiltonian_block_fork_bit_identical() {
    let _g = lock();
    let (h, x, k, space) = fixture(0x5A02);

    let fr = right_to_left_sketching(&h, &x);
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let ws = onesite_workspace(&h, x.core(k));

    let prev = set_block_fork(false);
    let y_old = h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws);
    set_block_fork(true);
    let y_new = h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws);
    set_block_fork(prev);

    let mut compared = 0usize;
    let mut nonzero = 0usize;
    for fam in Family::ALL {
        for s in space.bond_sectors(k) {
            match (y_old.block(fam, s), y_new.block(fam, s)) {
                (Some(a), Some(b)) => {
                    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
                    for j in 0..a.ncols() {
                        for i in 0..a.nrows() {
                            let (p, q) = (*a.get(i, j), *b.get(i, j));
                            assert_eq!(
                                p.to_bits(),
                                q.to_bits(),
                                "block fork not bit-identical at {fam:?} {s:?} ({i},{j}): \
                                 {p:e} vs {q:e}"
                            );
                            compared += 1;
                            if p != 0.0 {
                                nonzero += 1;
                            }
                        }
                    }
                }
                (None, None) => {}
                _ => panic!("block presence mismatch across the fork switch at {fam:?} {s:?}"),
            }
        }
    }
    assert!(compared > 0, "no blocks compared — fixture is vacuous");
    assert!(
        nonzero > 0,
        "every compared entry was zero — fixture is vacuous"
    );
}

#[test]
fn sketched_hamiltonian_block_fork_thread_count_independent() {
    let _g = lock();
    let (h, x, k, space) = fixture(0x5A03);

    let fr = right_to_left_sketching(&h, &x);
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let ws = onesite_workspace(&h, x.core(k));

    let prev = set_block_fork(true);
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws))
    };
    let y1 = run(1);
    let y8 = run(8);
    let y18 = run(18);
    set_block_fork(prev);

    let mut nonzero = 0usize;
    for fam in Family::ALL {
        for s in space.bond_sectors(k) {
            let (a, b, c) = (y1.block(fam, s), y8.block(fam, s), y18.block(fam, s));
            match (a, b, c) {
                (Some(a), Some(b), Some(c)) => {
                    for j in 0..a.ncols() {
                        for i in 0..a.nrows() {
                            let v = *a.get(i, j);
                            assert_eq!(v.to_bits(), b.get(i, j).to_bits(), "1 vs 8 threads");
                            assert_eq!(v.to_bits(), c.get(i, j).to_bits(), "1 vs 18 threads");
                            if v != 0.0 {
                                nonzero += 1;
                            }
                        }
                    }
                }
                (None, None, None) => {}
                _ => panic!("block presence differs across thread counts at {fam:?} {s:?}"),
            }
        }
    }
    assert!(
        nonzero > 0,
        "every compared entry was zero — fixture is vacuous"
    );
}

fn sketch_maxdiff(
    a: &qntt::Sketch<f64>,
    b: &qntt::Sketch<f64>,
    space: QnSpace,
    bond: usize,
) -> (f64, usize) {
    let mut maxdiff = 0.0f64;
    let mut nonzero = 0usize;
    for s in space.bond_sectors(bond) {
        match (a.block(s), b.block(s)) {
            (Some(p), Some(q)) => {
                assert_eq!((p.nrows(), p.ncols()), (q.nrows(), q.ncols()));
                for j in 0..p.ncols() {
                    for i in 0..p.nrows() {
                        let (u, v) = (*p.get(i, j), *q.get(i, j));
                        maxdiff = maxdiff.max((u - v).abs());
                        if u != 0.0 {
                            nonzero += 1;
                        }
                    }
                }
            }
            (None, None) => {}
            _ => panic!("sketch block presence mismatch at {s:?}"),
        }
    }
    (maxdiff, nonzero)
}

#[test]
fn sketching_step_right_block_fork_matches_to_machine_precision() {
    let _g = lock();
    let (h, x, k, space) = fixture(0x5A05);
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let ws = onesite_workspace(&h, x.core(k));

    let prev = set_block_fork(false);
    let c_old = h.sketching_step_right(&x, &fl, &ws);
    set_block_fork(true);
    let c_new = h.sketching_step_right(&x, &fl, &ws);
    set_block_fork(prev);

    let (maxdiff, nonzero) = sketch_maxdiff(&c_old, &c_new, space, k + 1);
    assert!(
        nonzero > 0,
        "every compared entry was zero — fixture is vacuous"
    );
    assert!(
        maxdiff < 1e-12,
        "sketching_step_right differs across the fork switch by {maxdiff:e} (> machine precision)"
    );
}

#[test]
fn sketching_step_right_block_fork_thread_count_independent() {
    let _g = lock();
    let (h, x, k, space) = fixture(0x5A06);
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let ws = onesite_workspace(&h, x.core(k));

    let prev = set_block_fork(true);
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| h.sketching_step_right(&x, &fl, &ws))
    };
    let c1 = run(1);
    let c8 = run(8);
    let c18 = run(18);
    set_block_fork(prev);

    let mut nonzero = 0usize;
    for s in space.bond_sectors(k + 1) {
        match (c1.block(s), c8.block(s), c18.block(s)) {
            (Some(a), Some(b), Some(c)) => {
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        let v = *a.get(i, j);
                        assert_eq!(
                            v.to_bits(),
                            b.get(i, j).to_bits(),
                            "1 vs 8 threads at {s:?}"
                        );
                        assert_eq!(
                            v.to_bits(),
                            c.get(i, j).to_bits(),
                            "1 vs 18 threads at {s:?}"
                        );
                        if v != 0.0 {
                            nonzero += 1;
                        }
                    }
                }
            }
            (None, None, None) => {}
            _ => panic!("block presence differs across thread counts at {s:?}"),
        }
    }
    assert!(
        nonzero > 0,
        "every compared entry was zero — fixture is vacuous"
    );
}

#[test]
fn sketching_step_left_block_fork_matches_to_machine_precision() {
    let _g = lock();
    let (h, x, k, space) = fixture(0x5A07);
    let fr_all = right_to_left_sketching(&h, &x);
    let fr = &fr_all[k];
    let ws = onesite_workspace(&h, x.core(k));

    let prev = set_block_fork(false);
    let c_old = h.sketching_step_left(&x, fr, &ws);
    set_block_fork(true);
    let c_new = h.sketching_step_left(&x, fr, &ws);
    set_block_fork(prev);

    let (maxdiff, nonzero) = sketch_maxdiff(&c_old, &c_new, space, k);
    assert!(
        nonzero > 0,
        "every compared entry was zero — fixture is vacuous"
    );
    assert!(
        maxdiff < 1e-12,
        "sketching_step_left differs across the fork switch by {maxdiff:e} (> machine precision)"
    );
}

#[test]
fn als_energy_matches_across_block_fork() {
    let _g = lock();
    let (h, x, _k, _space) = fixture(0x5A04);
    let opts = AlsOptions {
        eps: 1e-8,
        max_iter: 2,
        ..Default::default()
    };

    let prev = set_block_fork(false);
    let l_old = als(&h, &x, &opts).0;
    set_block_fork(true);
    let l_new = als(&h, &x, &opts).0;
    set_block_fork(prev);

    assert!(
        l_old.is_finite() && l_old.abs() > 1e-8,
        "λ echo is vacuous: {l_old}"
    );
    assert!(
        (l_old - l_new).abs() < 1e-10 * l_old.abs().max(1.0),
        "ALS λ drifted across the fork switch: {l_old} vs {l_new} (Δ={:e})",
        (l_old - l_new).abs()
    );
}

#[test]
fn sketched_hamiltonian_block_split_matches_to_machine_precision() {
    let _g = lock();
    let (h, x, k, space) = skewed_fixture(0x5A09);
    let fr = right_to_left_sketching(&h, &x);
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let ws = onesite_workspace(&h, x.core(k));

    let prev = set_block_fork(true);
    let prev_split = set_block_split(false);
    let y_whole = h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws);
    set_block_split(true);
    reset_block_split_events();
    let y_split = h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws);
    let events = block_split_events();
    set_block_split(prev_split);
    set_block_fork(prev);
    assert!(
        events > 0,
        "no block was actually split — the test is vacuous, raise the fixture size"
    );

    let mut maxdiff = 0.0f64;
    let mut scale = 0.0f64;
    for fam in Family::ALL {
        for s in space.bond_sectors(k) {
            if let (Some(a), Some(b)) = (y_whole.block(fam, s), y_split.block(fam, s)) {
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        let (u, v) = (*a.get(i, j), *b.get(i, j));
                        maxdiff = maxdiff.max((u - v).abs());
                        scale = scale.max(u.abs());
                    }
                }
            }
        }
    }
    assert!(scale > 0.0, "fixture is vacuous");
    assert!(
        maxdiff <= 1e-12 * scale.max(1.0),
        "block split differs from the whole-block path by {maxdiff:e} (scale {scale:e})"
    );
}

#[test]
fn sketched_hamiltonian_block_split_thread_count_independent() {
    let _g = lock();
    let (h, x, k, space) = skewed_fixture(0x5A0A);
    let fr = right_to_left_sketching(&h, &x);
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let ws = onesite_workspace(&h, x.core(k));

    let prev = set_block_fork(true);
    let prev_split = set_block_split(true);
    reset_block_split_events();
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| h.sketched_hamiltonian(x.core(k), &fl, &fr[k], &ws))
    };
    let y1 = run(1);
    let y8 = run(8);
    let y18 = run(18);
    let events = block_split_events();
    set_block_split(prev_split);
    set_block_fork(prev);
    assert!(
        events > 0,
        "no block was actually split — the test is vacuous"
    );

    let mut nonzero = 0usize;
    for fam in Family::ALL {
        for s in space.bond_sectors(k) {
            if let (Some(a), Some(b), Some(c)) =
                (y1.block(fam, s), y8.block(fam, s), y18.block(fam, s))
            {
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        let v = *a.get(i, j);
                        assert_eq!(v.to_bits(), b.get(i, j).to_bits(), "1 vs 8 threads");
                        assert_eq!(v.to_bits(), c.get(i, j).to_bits(), "1 vs 18 threads");
                        if v != 0.0 {
                            nonzero += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(nonzero > 0, "fixture is vacuous");
}

#[test]
fn sketching_step_partials_batching_is_bit_identical() {
    use qntt::hamiltonian::matvec::set_env_partials_bytes;
    let _g = lock();
    let (h, x, k, space) = fixture(0x5A0B);
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let fr_all = right_to_left_sketching(&h, &x);
    let fr = &fr_all[k];
    let ws = onesite_workspace(&h, x.core(k));
    let prev_fork = set_block_fork(true);

    let prev_b = set_env_partials_bytes(0);
    let r_ref = h.sketching_step_right(&x, &fl, &ws);
    let l_ref = h.sketching_step_left(&x, fr, &ws);

    set_env_partials_bytes(1);
    for threads in [1usize, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let r = h.sketching_step_right(&x, &fl, &ws);
            let l = h.sketching_step_left(&x, fr, &ws);
            let (dr, nr) = sketch_maxdiff(&r_ref, &r, space, k + 1);
            let (dl, nl) = sketch_maxdiff(&l_ref, &l, space, k);
            assert!(nr > 0 && nl > 0, "vacuous fixture");
            assert_eq!(
                dr, 0.0,
                "right step: batching moved a value at {threads} threads"
            );
            assert_eq!(
                dl, 0.0,
                "left step: batching moved a value at {threads} threads"
            );
        });
    }
    set_env_partials_bytes(prev_b);
    set_block_fork(prev_fork);
}
