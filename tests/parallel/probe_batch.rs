// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::par::{set_par_min_fork_work, set_vector_par};
use qntt::qn::QnSpace;
use qntt::rayleigh_quotient;
use qntt::solvers::hess::{HessPart, PdVerdict, ProbeDavidson};
use qntt::solvers::riemannian::{certify_probe, normalize_mut};
use qntt_testkit::generic_truncated_h;
use qntt_testkit::{randomized_state, seeded};

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

fn verdict_bits(v: &PdVerdict) -> (u8, u64, u64) {
    match v {
        PdVerdict::WitnessNegative { lambda } => (0, lambda.to_bits(), 0),
        PdVerdict::LikelyPd {
            lambda_min,
            residual,
        } => (1, lambda_min.to_bits(), residual.to_bits()),
        PdVerdict::Inconclusive {
            lambda_min,
            residual,
        } => (2, lambda_min.to_bits(), residual.to_bits()),
    }
}

#[test]
fn tpb_probe_concurrent_batch_matches_serial() {
    let _lock = crate::lock();
    let _force = ForceParallel::new();

    let space = QnSpace::new(4, 2, 1);
    let ham = generic_truncated_h::<f64>(space);
    let mut rng = seeded(0xB0B1);
    let mut x = randomized_state::<f64, _>(space, 1, 2, &mut rng);
    normalize_mut(&mut x);
    let lambda = rayleigh_quotient(&ham, &x);
    let dav = ProbeDavidson {
        maxsubspace: 32,
        restart_keep: 8,
        nwarm: 4,
        block: 4,
    };

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("4-thread pool");

    for part in [HessPart::Rr, HessPart::Ii] {
        let (vs, ws, aps) = pool1.install(|| {
            assert_eq!(rayon::current_num_threads(), 1, "serial arm pool size");
            certify_probe(&ham, &x, lambda, part, 90, 0x5EED, &dav)
        });
        let (vp, wp, app) = pool4.install(|| {
            assert_eq!(rayon::current_num_threads(), 4, "parallel arm pool size");
            certify_probe(&ham, &x, lambda, part, 90, 0x5EED, &dav)
        });

        assert_eq!(aps, app, "{part:?}: applies {aps} (serial) vs {app} (fork)");
        assert_eq!(
            verdict_bits(&vs),
            verdict_bits(&vp),
            "{part:?}: verdict not bitwise equal ({vs:?} vs {vp:?})"
        );
        assert_eq!(
            ws.num_coeffs(),
            wp.num_coeffs(),
            "{part:?}: witness coefficient count"
        );
        for i in 0..ws.num_coeffs() {
            assert_eq!(
                ws.coeff(i).to_bits(),
                wp.coeff(i).to_bits(),
                "{part:?}: witness coefficient {i} not bitwise equal"
            );
        }
        println!(
            "tpb probe batch {part:?}: {vs:?} after {aps} applies, bit-identical 1 vs 4 threads"
        );
    }
}
