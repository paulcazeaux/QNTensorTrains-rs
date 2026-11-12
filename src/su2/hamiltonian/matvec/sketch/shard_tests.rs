// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::sync::atomic::Ordering;

use faer::MatRef;
use faer::prelude::ReborrowMut;
use rand::{Rng, RngExt};

use crate::hamiltonian::matvec::PAR_MIN_TASKS;
use crate::scalar::Scalar;
use crate::scalar::gemm_par_hook::{Hook, pool};
use crate::su2::core::{Family, Su2Core};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::matvec::{h_sketch_mul, sketch_h_mul};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::shard::{FORCE_SPLIT, Region, take_split_counts};
use crate::su2::vector::Su2TtVector;
use crate::test_util::in_pool;
use crate::test_util::{generic_su2_h, randomized_su2_state, seeded};

const POOLS: [usize; 3] = [1, 8, 18];

fn rand_sketch(
    space: Su2Space,
    k: usize,
    rows: Option<&Su2RankGrid>,
    cols: Option<&Su2RankGrid>,
    rng: &mut impl Rng,
) -> Su2Sketch<f64> {
    let mut draw = |given: Option<&Su2RankGrid>| {
        let mut g = Su2RankGrid::zeros(&space);
        for s in space.bond_sectors(k) {
            g[s] = given.map_or_else(|| rng.random_range(2..=6), |r| r[s]);
        }
        g
    };
    let (r, c) = (draw(rows), draw(cols));
    let mut f = Su2Sketch::zeros(space, k, &r, &c);
    for s in space.bond_sectors(k) {
        if let Some(mut b) = f.block_mut(s) {
            for j in 0..b.ncols() {
                for i in 0..b.nrows() {
                    *b.rb_mut().get_mut(i, j) = f64::sample_standard(rng);
                }
            }
        }
    }
    f
}

fn assert_block_bits(a: MatRef<'_, f64>, b: MatRef<'_, f64>, what: &str) {
    assert_eq!(
        (a.nrows(), a.ncols()),
        (b.nrows(), b.ncols()),
        "{what}: shape"
    );
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            assert_eq!(
                a.get(i, j).to_bits(),
                b.get(i, j).to_bits(),
                "{what}: entry ({i}, {j})"
            );
        }
    }
}

fn assert_core_bits(a: &Su2Core<f64>, b: &Su2Core<f64>, what: &str) {
    assert_eq!(a.row_ranks, b.row_ranks, "{what}: row ranks");
    assert_eq!(a.col_ranks, b.col_ranks, "{what}: col ranks");
    for f in Family::ALL {
        let ta: Vec<Su2Sector> = a.transitions(f).collect();
        let tb: Vec<Su2Sector> = b.transitions(f).collect();
        assert_eq!(ta, tb, "{what}: {f:?} transitions");
        for s in ta {
            assert_block_bits(
                a.block(f, s).unwrap(),
                b.block(f, s).unwrap(),
                &format!("{what}: block ({f:?}, {s:?})"),
            );
        }
    }
}

fn serial_left(
    h: &Su2SparseHamiltonian<f64>,
    fl: &Su2Sketch<f64>,
    x: &Su2Core<f64>,
    cols: &Su2RankGrid,
    keep: impl Fn(Su2Sector) -> bool,
) -> (Su2Core<f64>, usize) {
    let (i_tab, j_tab) = (h.row_ranges(x), h.col_ranges(x));
    let g = x.space.pg.irrep_of(x.k);
    let mut y = Su2Core::zeros(x.space, x.k, &fl.row_ranks(), cols);
    let tasks: Vec<(Family, Su2Sector)> = Family::ALL
        .into_iter()
        .flat_map(|o| y.transitions(o).map(move |n1| (o, n1)))
        .filter(|&(o, n1)| o.apply(n1, g).is_some_and(&keep))
        .collect();
    for &(o, n1) in &tasks {
        if let Some(yb) = y.block_mut(o, n1) {
            h.sketch_h_mul_block_into(yb, fl, x, o, n1, &i_tab, &j_tab);
        }
    }
    (y, tasks.len())
}

fn serial_right(
    h: &Su2SparseHamiltonian<f64>,
    x: &Su2Core<f64>,
    fr: &Su2Sketch<f64>,
    rows: &Su2RankGrid,
    keep: impl Fn(Su2Sector) -> bool,
) -> (Su2Core<f64>, usize) {
    let (i_tab, j_tab) = (h.row_ranges(x), h.col_ranges(x));
    let mut y = Su2Core::zeros(x.space, x.k, rows, &fr.col_ranks());
    let tasks: Vec<(Family, Su2Sector)> = Family::ALL
        .into_iter()
        .flat_map(|o| y.transitions(o).map(move |n1| (o, n1)))
        .filter(|&(_, n1)| keep(n1))
        .collect();
    for &(o, n1) in &tasks {
        if let Some(yb) = y.block_mut(o, n1) {
            h.h_sketch_mul_block_into(yb, x, fr, o, n1, &i_tab, &j_tab);
        }
    }
    (y, tasks.len())
}

fn fixture() -> (Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
    let space = Su2Space::new(8, 4, 0);
    let x = randomized_su2_state::<f64, _>(space, 2, 5, &mut seeded(0x5_4A2D));
    (generic_su2_h::<f64>(space), x)
}

struct Forced(bool);
impl Forced {
    fn on() -> Self {
        Forced(FORCE_SPLIT.swap(true, Ordering::Relaxed))
    }
}
impl Drop for Forced {
    fn drop(&mut self) {
        FORCE_SPLIT.store(self.0, Ordering::Relaxed);
    }
}

#[test]
fn sketch_kernels_band_shards_bit_identical_across_pools() {
    let _lock = crate::par::lock_par_switches();
    let _forced = Forced::on();
    let (h, x) = fixture();
    let space = x.space;
    let mut rng = seeded(0x5_4A2E);
    for k in [3usize, 4] {
        let xk = x.core(k);
        let fl = rand_sketch(space, k, None, Some(&h.row_ranks(xk)), &mut rng);
        let fr = rand_sketch(space, k + 1, Some(&h.col_ranks(xk)), None, &mut rng);
        let (left0, nl) = serial_left(&h, &fl, xk, &h.col_ranks(xk), |_| true);
        let (right0, nr) = serial_right(&h, xk, &fr, &h.row_ranks(xk), |_| true);
        assert!(
            nl >= PAR_MIN_TASKS && nr >= PAR_MIN_TASKS,
            "site {k}: {nl}/{nr} targets — the threaded arm would not run"
        );
        for nt in POOLS {
            let (l, r, c) = in_pool(nt, || {
                let _ = take_split_counts();
                let l = sketch_h_mul(&fl, &h, xk);
                let r = h_sketch_mul(&h, xk, &fr);
                (l, r, take_split_counts())
            });
            assert_core_bits(&left0, &l, &format!("sketch_h_mul site {k}, {nt} thr"));
            assert_core_bits(&right0, &r, &format!("h_sketch_mul site {k}, {nt} thr"));
            let (cl, cr) = (
                c[Region::SketchHMul as usize],
                c[Region::HSketchMul as usize],
            );
            println!("site {k}, {nt} threads: blocks split left {cl}, right {cr}");
            if nt == 1 {
                assert_eq!((cl, cr), (0, 0), "site {k}: split in a 1-thread pool");
            } else {
                assert!(
                    cl > 0 && cr > 0,
                    "site {k}, {nt} thr: no block split: {c:?}"
                );
            }
        }
    }
}

#[test]
fn sketch_kernels_keep_one_sector_stay_serial_bit_identical() {
    let _lock = crate::par::lock_par_switches();
    let _forced = Forced::on();
    let (h, x) = fixture();
    let space = x.space;
    let mut rng = seeded(0x5_4A2F);
    let k = 3usize;
    let xk = x.core(k);
    let (i_tab, j_tab) = (h.row_ranges(xk), h.col_ranges(xk));
    let fl = rand_sketch(space, k, None, Some(&h.row_ranks(xk)), &mut rng);
    let fr = rand_sketch(space, k + 1, Some(&h.col_ranks(xk)), None, &mut rng);
    let mut checked = 0usize;
    for q in space.bond_sectors(k + 1) {
        let mut cols = Su2RankGrid::zeros(&space);
        cols[q] = h.col_ranks(xk)[q];
        let (y0, n) = serial_left(&h, &fl, xk, &cols, |s| s == q);
        assert!(n <= 4, "one column sector admits ≤ 4 targets");
        for nt in POOLS {
            let (y, c) = in_pool(nt, || {
                let _ = take_split_counts();
                let mut y = Su2Core::zeros(space, k, &fl.row_ranks(), &cols);
                h.sketch_h_mul_into(&mut y, &fl, xk, |s| s == q, &i_tab, &j_tab);
                (y, take_split_counts())
            });
            assert_core_bits(&y0, &y, &format!("keep {q:?}, {nt} thr"));
            assert_eq!(c, [0; 5], "keep {q:?}, {nt} thr: the serial arm sharded");
            checked += 1;
        }
    }
    for n0 in space.bond_sectors(k) {
        let mut rows = Su2RankGrid::zeros(&space);
        rows[n0] = h.row_ranks(xk)[n0];
        let (y0, n) = serial_right(&h, xk, &fr, &rows, |s| s == n0);
        assert!(n <= 4, "one row sector admits ≤ 4 targets");
        for nt in POOLS {
            let (y, c) = in_pool(nt, || {
                let _ = take_split_counts();
                let mut y = Su2Core::zeros(space, k, &rows, &fr.col_ranks());
                h.h_sketch_mul_into(&mut y, xk, &fr, |s| s == n0, &i_tab, &j_tab);
                (y, take_split_counts())
            });
            assert_core_bits(&y0, &y, &format!("keep {n0:?}, {nt} thr"));
            assert_eq!(c, [0; 5], "keep {n0:?}, {nt} thr: the serial arm sharded");
            checked += 1;
        }
    }
    assert!(checked > 0);
}

fn core_rel(a: &Su2Core<f64>, b: &Su2Core<f64>) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for f in Family::ALL {
        for s in a.transitions(f) {
            let (x, y) = (a.block(f, s).unwrap(), b.block(f, s).unwrap());
            for j in 0..x.ncols() {
                for i in 0..x.nrows() {
                    num += (x[(i, j)] - y[(i, j)]).powi(2);
                    den += x[(i, j)].powi(2);
                }
            }
        }
    }
    (num / den).sqrt()
}

#[test]
fn sketch_kernels_band_shards_bit_identical_gemm_gate_forced() {
    let _lock = crate::par::lock_par_switches();
    let _forced = Forced::on();
    let (h, x) = fixture();
    let space = x.space;
    let mut rng = seeded(0x5_4A2E);
    for k in [3usize, 4] {
        let xk = x.core(k);
        let fl = rand_sketch(space, k, None, Some(&h.row_ranks(xk)), &mut rng);
        let fr = rand_sketch(space, k + 1, Some(&h.col_ranks(xk)), None, &mut rng);
        let mut ref1: Option<(Su2Core<f64>, Su2Core<f64>)> = None;
        for nt in POOLS {
            let hook = Hook::forced();
            let (l0, r0, l, r) = pool(nt, &hook).install(|| {
                let (l0, _) = serial_left(&h, &fl, xk, &h.col_ranks(xk), |_| true);
                let (r0, _) = serial_right(&h, xk, &fr, &h.row_ranks(xk), |_| true);
                (l0, r0, sketch_h_mul(&fl, &h, xk), h_sketch_mul(&h, xk, &fr))
            });
            assert_core_bits(&l0, &l, &format!("forced sketch_h_mul site {k}, {nt} thr"));
            assert_core_bits(&r0, &r, &format!("forced h_sketch_mul site {k}, {nt} thr"));
            if nt == 1 {
                assert_eq!(hook.taken(), 0, "site {k}: a 1-thread pool threaded a GEMM");
                ref1 = Some((l0, r0));
                continue;
            }
            assert!(
                hook.taken() > 0,
                "site {k}, {nt} thr: no gated GEMM threaded ({} of {} calls)",
                hook.taken(),
                hook.calls()
            );
            let (l1, r1) = ref1.as_ref().unwrap();
            for (what, y1, yn) in [("sketch_h_mul", l1, &l), ("h_sketch_mul", r1, &r)] {
                let rel = core_rel(y1, yn);
                println!("forced {what} site {k}, 1 vs {nt} threads: ‖Δ‖/‖y₁‖ = {rel:.3e}");
                assert!(rel <= 1e-12, "forced {what} site {k}, 1 vs {nt}: {rel:.3e}");
            }
        }
    }
}
