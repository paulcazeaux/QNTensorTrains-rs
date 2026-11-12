// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::SparseHamiltonian;
use qntt::core::tangent::TtTangent;
use qntt::par::{set_par_min_fork_work, set_vector_par};
use qntt::qn::QnSpace;
use qntt::solvers::InnerProductVector;
use qntt::solvers::hess::{
    TdaDavidsonOpts, TdaModes, TdaPrecond, tda_lowest_davidson, weingarten_operator,
};
use qntt::solvers::riemannian::normalize_mut;
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::Su2Space;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::Su2TtVector;
use qntt::tt_randn;
use qntt_testkit::generic_truncated_tv;
use qntt_testkit::{randomized_state, randomized_su2_state, seeded};

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

fn assert_bitwise<V: InnerProductVector<Scalar = f64>>(
    a: &TdaModes<V>,
    b: &TdaModes<V>,
    what: &str,
) {
    assert_eq!(a.info.applies, b.info.applies, "{what}: applies");
    assert_eq!(a.info.iterations, b.info.iterations, "{what}: iterations");
    assert_eq!(a.theta.len(), b.theta.len(), "{what}: root count");
    for (j, (p, q)) in a.theta.iter().zip(&b.theta).enumerate() {
        assert_eq!(p.to_bits(), q.to_bits(), "{what}: θ[{j}] {p} vs {q}");
    }
    for (j, (p, q)) in a.info.residuals.iter().zip(&b.info.residuals).enumerate() {
        assert_eq!(p.to_bits(), q.to_bits(), "{what}: residual[{j}]");
    }
    for (j, (u, w)) in a.vectors.iter().zip(&b.vectors).enumerate() {
        assert_eq!(u.num_coeffs(), w.num_coeffs(), "{what}: vector {j} shape");
        for i in 0..u.num_coeffs() {
            assert_eq!(
                u.coeff(i).to_bits(),
                w.coeff(i).to_bits(),
                "{what}: vector {j} coefficient {i}"
            );
        }
    }
}

#[test]
fn ttb_tda_concurrent_batch_matches_serial() {
    let _lock = crate::lock();
    let _force = ForceParallel::new();

    let space = QnSpace::new(4, 2, 1);
    let (t, v) = generic_truncated_tv::<f64>(space.d);
    let ham = SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xB0B1);
    let mut x = randomized_state::<f64, _>(space, 1, 2, &mut rng);
    normalize_mut(&mut x);
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let v0: Vec<TtTangent<f64>> = (0..6)
        .map(|_| TtTangent::project(bl, br, &tt_randn(bl.space, &bl.ranks, &mut rng)))
        .collect();

    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("4-thread pool");

    for precond in [TdaPrecond::Off, TdaPrecond::Diag { kappa: 10.0 }] {
        let opts = |max_concurrent| TdaDavidsonOpts {
            tol: 1e-10,
            max_iter: 1000,
            max_subspace: 8,
            restart_keep: 4,
            precond,
            max_concurrent,
            ..TdaDavidsonOpts::default()
        };
        let serial = pool1.install(|| {
            assert_eq!(rayon::current_num_threads(), 1, "serial arm pool size");
            tda_lowest_davidson(&ctx, &v0, 4, &opts(0))
        });
        let fork = pool4.install(|| {
            assert_eq!(rayon::current_num_threads(), 4, "parallel arm pool size");
            tda_lowest_davidson(&ctx, &v0, 4, &opts(0))
        });
        let chunked = pool4.install(|| tda_lowest_davidson(&ctx, &v0, 4, &opts(2)));
        assert!(
            serial.info.converged,
            "{precond:?}: the parity is read on a converged solve"
        );
        assert!(
            serial.info.applies > 8,
            "{precond:?}: the forced restarts ran"
        );
        assert_bitwise(&serial, &fork, &format!("{precond:?} fork"));
        assert_bitwise(&serial, &chunked, &format!("{precond:?} chunks of 2"));
        println!(
            "ttb tda batch {precond:?}: θ {:?} after {} applies, bit-identical 1 vs 4 threads vs chunks of 2",
            serial.theta, serial.info.applies
        );
    }
}

#[test]
fn ttb_su2_tda_concurrent_batch_matches_serial() {
    use qntt::su2::hess::{tda_lowest_davidson as su2_tda, weingarten_operator as su2_ctx};
    let _lock = crate::lock();
    let _force = ForceParallel::new();

    let space = Su2Space::new(4, 4, 0);
    let (t, v) = generic_truncated_tv::<f64>(space.d);
    let ham = Su2SparseHamiltonian::<f64>::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xB0B2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    let ctx = su2_ctx(&ham, &x);
    let (bl, br) = ctx.bases();
    let v0: Vec<Su2TtTangent<f64>> = (0..6)
        .map(|_| {
            let d: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
            Su2TtTangent::project(bl, br, &d)
        })
        .collect();
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let pool4 = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("4-thread pool");
    for precond in [TdaPrecond::Off, TdaPrecond::Diag { kappa: 10.0 }] {
        let opts = |max_concurrent| TdaDavidsonOpts {
            tol: 1e-10,
            max_iter: 2000,
            max_subspace: 8,
            restart_keep: 4,
            precond,
            max_concurrent,
            ..TdaDavidsonOpts::default()
        };
        let serial = pool1.install(|| su2_tda(&ctx, &v0, 4, &opts(0)));
        let fork = pool4.install(|| su2_tda(&ctx, &v0, 4, &opts(0)));
        let chunked = pool4.install(|| su2_tda(&ctx, &v0, 4, &opts(2)));
        assert!(serial.info.converged, "{precond:?}: converged");
        assert!(
            serial.info.applies > 8,
            "{precond:?}: the forced restarts ran"
        );
        assert_bitwise(&serial, &fork, &format!("su2 {precond:?} fork"));
        assert_bitwise(&serial, &chunked, &format!("su2 {precond:?} chunks of 2"));
        println!(
            "ttb su2 tda batch {precond:?}: θ {:?} after {} applies ({} it), bit-identical",
            serial.theta, serial.info.applies, serial.info.iterations
        );
    }
}
