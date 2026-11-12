// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;

use qntt::hamiltonian::xt_h_y;
use qntt::integrals::{IntegralsOptions, load_fcidump, load_fixture_dir};
use qntt::rdm::{transition_1rdm, transition_1rdm_bruteforce, transition_1rdm_h};
use qntt::rotation::{e_pq, orbital_gradient};
use qntt::{CoeffTensor, QnSpace, SparseHamiltonian, TtVector, dot};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::load_h2o;
use crate::util::*;

fn h4_fixture() -> (QnSpace, SparseHamiltonian<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP");
    let load =
        load_fcidump::<f64>(&path, None, &IntegralsOptions::default()).expect("h4 fcidump loads");
    let ham = load.hamiltonian(f64::EPSILON);
    (load.space, ham)
}

fn lih_fixture() -> (QnSpace, SparseHamiltonian<f64>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih loads");
    let ham = load.hamiltonian(f64::EPSILON);
    (load.space, ham)
}

fn maxg_maxdiff(a: &CoeffTensor<f64>, b: &CoeffTensor<f64>, d: usize) -> (f64, f64) {
    let mut maxg = 0.0f64;
    let mut maxdiff = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            let av = a.get(&[p, q]);
            let bv = b.get(&[p, q]);
            maxg = maxg.max(av.abs()).max(bv.abs());
            maxdiff = maxdiff.max((av - bv).abs());
        }
    }
    (maxg, maxdiff)
}

#[test]
fn tjr_hermitian_trace() {
    let d = 6;
    let space = QnSpace::from_n_two_sz(d, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE3E3_0001);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);

    let gamma = transition_1rdm(&x, &x);
    let mut maxg = 0.0f64;
    let mut maxsym = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            let a = gamma.get(&[p, q]);
            let b = gamma.get(&[q, p]);
            maxg = maxg.max(a.abs());
            maxsym = maxsym.max((a - b).abs());
        }
    }
    assert!(
        maxsym <= 1e-13 * maxg.max(1.0),
        "gamma(x,x) (sweep) is not symmetric: max|diff|={maxsym:e}"
    );

    let mut tr = 0.0f64;
    for p in 0..d {
        tr += gamma.get(&[p, p]);
    }
    let n = (space.nup + space.ndn) as f64;
    let xx = dot(&x, &x);
    assert_close(
        tr,
        n * xx,
        1e-12 * (n * xx).abs().max(1.0),
        "tr gamma(x,x) (sweep) != N·<x,x>",
    );

    let y: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let gamma_xy = transition_1rdm(&x, &y);
    let gamma_yx = transition_1rdm(&y, &x);
    let mut maxxy = 0.0f64;
    let mut maxt = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            maxxy = maxxy.max(gamma_xy.get(&[p, q]).abs());
            maxt = maxt.max((gamma_xy.get(&[p, q]) - gamma_yx.get(&[q, p])).abs());
        }
    }
    assert!(
        maxt <= 1e-13 * maxxy.max(1.0),
        "gamma(x,y) (sweep) != gamma(y,x)^T: max|diff|={maxt:e}"
    );
}

#[test]
fn tjr_dressed_empty_spin_channel() {
    let d = 4;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE4E4_0001);
    let t = random_sym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    for &two_sz in &[2i64, -2] {
        let space = QnSpace::from_n_two_sz(d, 2, two_sz);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng2 = Xoshiro256PlusPlus::seed_from_u64(0xE4E4_1000 + two_sz.unsigned_abs());
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng2);
        let w: TtVector<f64> = randomized_state(space, 1, 3, &mut rng2);
        let hw = h.apply(&w);
        let g_dressed = transition_1rdm_h(&x, &h, &w);
        let g_materialized = transition_1rdm_bruteforce(&x, &hw);
        let (maxg, maxdiff) = maxg_maxdiff(&g_dressed, &g_materialized, d);
        assert!(
            maxdiff <= 1e-12 * maxg.max(1.0),
            "empty-channel (2Sz={two_sz}) transition_1rdm_h vs bruteforce: max|diff|={maxdiff:e}"
        );
    }
}

#[test]
fn tjr_dressed_trace_identity() {
    let (space, h) = h4_fixture();
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7B0C_0001);
    let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let y: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);

    let gamma = transition_1rdm_h(&x, &h, &y);
    let mut tr = 0.0f64;
    for p in 0..d {
        tr += gamma.get(&[p, p]);
    }
    let n = (space.nup + space.ndn) as f64;
    let xhy = xt_h_y(&x, &h, &y);
    assert_close(
        tr,
        n * xhy,
        1e-12 * (n * xhy).abs().max(1.0),
        "tr gamma_H(x,H,y) != N·xt_h_y(x,h,y)",
    );
}

fn orbital_gradient_pairwise(h: &SparseHamiltonian<f64>, x: &TtVector<f64>) -> CoeffTensor<f64> {
    let space = x.space;
    let d = space.d;
    let pg = space.pg;
    let mut gamma = vec![0.0f64; d * d];
    for p in 0..d {
        for q in 0..d {
            if p == q || pg.irrep_of(p) != pg.irrep_of(q) {
                continue;
            }
            gamma[p * d + q] = xt_h_y(&e_pq(x, q, p, 1.0), h, x);
        }
    }
    let mut g = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in 0..d {
            if p == q {
                continue;
            }
            g.set(&[p, q], 2.0 * (gamma[p * d + q] - gamma[q * d + p]));
        }
    }
    g
}

fn check_gradient_swap(space: QnSpace, h: &SparseHamiltonian<f64>, seed: u64, tag: &str) {
    let d = space.d;
    let pg = space.pg;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    let g_sweep = orbital_gradient(h, &x);
    let g_pairwise = orbital_gradient_pairwise(h, &x);
    let (maxg, maxdiff) = maxg_maxdiff(&g_sweep, &g_pairwise, d);
    assert!(
        maxg > 1e-6,
        "[{tag}] anti-vacuity: orbital_gradient must be genuinely nonzero"
    );
    assert!(
        maxdiff <= 1e-12 * maxg.max(1.0),
        "[{tag}] orbital_gradient (sweep-backed) vs the per-entry twin: max|diff|={maxdiff:e}"
    );

    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert_eq!(
                    g_sweep.get(&[p, q]),
                    0.0,
                    "[{tag}] sweep-backed orbital_gradient must leave cross-irrep ({p},{q}) \
                     EXACTLY 0.0"
                );
            }
        }
    }
}

#[test]
fn tjr_gradient_swap() {
    let (space, h) = lih_fixture();
    check_gradient_swap(space, &h, 0x9A70_0001, "lih_sto3g");

    let load = load_h2o();
    let space = load.space;
    let h = load.hamiltonian(f64::EPSILON);
    check_gradient_swap(space, &h, 0x9A70_0002, "h2o_c2v");
}

#[test]
fn tjr_pg_sweep_zero_cells() {
    let load = load_h2o();
    let space = load.space;
    let d = space.d;
    let pg = space.pg;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB400_0001);
    let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let y: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    let g_sweep = transition_1rdm(&x, &y);
    let g_brute = transition_1rdm_bruteforce(&x, &y);

    let mut max_same_irrep = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert_eq!(
                    g_sweep.get(&[p, q]),
                    0.0,
                    "transition_1rdm must leave cross-irrep ({p},{q}) EXACTLY 0.0 under C2v"
                );
            } else {
                max_same_irrep = max_same_irrep.max(g_sweep.get(&[p, q]).abs());
            }
        }
    }
    assert!(
        max_same_irrep > 1e-6,
        "anti-vacuity: the same-irrep block must be genuinely nonzero"
    );
    let (maxg, maxdiff) = maxg_maxdiff(&g_sweep, &g_brute, d);
    assert!(
        maxdiff <= 1e-12 * maxg.max(1.0),
        "transition_1rdm vs bruteforce over h2o_c2v: max|diff|={maxdiff:e}"
    );

    let h = load.hamiltonian(f64::EPSILON);
    let hy = h.apply(&y);
    let g_dressed = transition_1rdm_h(&x, &h, &y);
    let g_dressed_brute = transition_1rdm_bruteforce(&x, &hy);
    let mut max_same_irrep_h = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                assert_eq!(
                    g_dressed.get(&[p, q]),
                    0.0,
                    "transition_1rdm_h must leave cross-irrep ({p},{q}) EXACTLY 0.0 under C2v"
                );
            } else {
                max_same_irrep_h = max_same_irrep_h.max(g_dressed.get(&[p, q]).abs());
            }
        }
    }
    assert!(
        max_same_irrep_h > 1e-6,
        "anti-vacuity: the H-dressed same-irrep block must be genuinely nonzero"
    );
    let (maxgh, maxdiffh) = maxg_maxdiff(&g_dressed, &g_dressed_brute, d);
    assert!(
        maxdiffh <= 1e-12 * maxgh.max(1.0),
        "transition_1rdm_h vs bruteforce(x,H·y) over h2o_c2v: max|diff|={maxdiffh:e}"
    );
}
