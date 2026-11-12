// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::hamiltonian::assembly::Construction;
use qntt::solvers::joint::JointScfOpts;
use qntt::{
    CoeffTensor, Integrals, QnSpace, RankGrid, TtVector, expm_antisym, max_ranks, rotate_integrals,
    tt_randn,
};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

pub const D2_KAPPA_MIX: [(usize, usize, f64); 7] = [
    (1, 0, 0.35),
    (2, 1, -0.25),
    (3, 2, 0.30),
    (4, 3, -0.20),
    (5, 4, 0.25),
    (2, 0, 0.15),
    (4, 1, -0.10),
];

pub const D2_ORB_GRAD: f64 = 1e-7;

pub struct D2Problem {
    pub t: CoeffTensor<f64>,
    pub v: CoeffTensor<f64>,
    pub ints: Integrals<f64>,
    pub space: QnSpace,
    pub x0: TtVector<f64>,
}

pub fn d2_problem() -> D2Problem {
    d2_problem_at(1, 1, 57)
}

pub fn d2_problem_at(lo: usize, hi: usize, seed: u64) -> D2Problem {
    let d = 6usize;
    let (t, v) = crate::generic_truncated_tv(d);

    let mut k = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q, val) in D2_KAPPA_MIX.iter() {
        k.set(&[p, q], val);
        k.set(&[q, p], -val);
    }
    let mut kflat = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            kflat[i * d + j] = k.get(&[i, j]);
        }
    }
    let u_mix = expm_antisym(&kflat, d);

    let ints0 = Integrals::<f64> {
        t: t.clone(),
        v: v.clone(),
        e0: 0.0,
        norb: d,
        nelec: 3,
        ms2: 1,
        orbsym: Vec::new(),
        isym: None,
    };
    let ints = rotate_integrals(&ints0, &u_mix);

    let space = QnSpace::new(d, 2, 1);
    let x0 = capped_randn(space, lo, hi, seed);
    D2Problem {
        t,
        v,
        ints,
        space,
        x0,
    }
}

pub fn d2_scf_opts() -> JointScfOpts {
    JointScfOpts {
        orb_max_iter: 40,
        orb_grad: D2_ORB_GRAD,
        tt_grad_tol: 1e-6,
        construction: Construction::Median,
        refill_h: false,
        ..JointScfOpts::default()
    }
}

fn capped_randn(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let d = space.d;
    let caps = max_ranks(&space);
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                let draw = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
                r[s] = draw.min(caps[b][s]);
            }
            r
        })
        .collect();
    tt_randn(space, &ranks, &mut rng)
}
