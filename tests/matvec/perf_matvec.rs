// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{CoeffTensor, QnSpace, SparseHamiltonian, TtVector};
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn mals_threaded_matches_serial_alt_seed() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9E37_79B9);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    x0.check();
    let opts = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        max_iter: 3,
        verbose: false,
        ..Default::default()
    };
    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| mals(&h, &x0, &opts).0)
    };
    let l1 = run(1);
    let l8 = run(8);
    assert!(
        (l1 - l8).abs() < 1e-10 * l1.abs().max(1.0),
        "MALS threaded vs serial energy mismatch: {l1} vs {l8}"
    );
}
