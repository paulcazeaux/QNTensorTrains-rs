// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::TtVector;
use qntt::core::tangent::TtTangent;
use qntt::core::twosite::contract;
use qntt::par::{set_block_slice, set_par_min_coeffs, set_par_min_fork_work, set_vector_par};
use qntt::qn::QnSpace;
use qntt::solvers::InnerProductVector;
use qntt::su2::core::{Family as Su2Family, Su2Core};
use qntt::su2::qn::Su2Space;
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::twosite::contract as su2_contract;
use qntt::su2::vector::Su2TtVector;
use qntt_testkit::randomized_state;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::lock;

struct ForceParallel {
    coeffs: usize,
    fork: usize,
    on: bool,
}

impl ForceParallel {
    fn new() -> Self {
        ForceParallel {
            coeffs: set_par_min_coeffs(0),
            fork: set_par_min_fork_work(0),
            on: set_vector_par(true),
        }
    }
}

impl Drop for ForceParallel {
    fn drop(&mut self) {
        set_par_min_coeffs(self.coeffs);
        set_par_min_fork_work(self.fork);
        set_vector_par(self.on);
    }
}

pub(crate) fn serial_arm<R>(f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            set_vector_par(self.0);
        }
    }
    let _restore = Restore(set_vector_par(false));
    f()
}

pub(crate) fn parallel_arm<R: Send>(nt: usize, f: impl FnOnce() -> R + Send) -> R {
    let _force = ForceParallel::new();
    rayon::ThreadPoolBuilder::new()
        .num_threads(nt)
        .build()
        .unwrap()
        .install(f)
}

pub(crate) fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn close(a: f64, b: f64, scale: f64) -> bool {
    (a - b).abs() <= 64.0 * f64::EPSILON * scale.max(1.0)
}

pub(crate) fn coeffs<V: InnerProductVector<Scalar = f64>>(v: &V) -> Vec<f64> {
    (0..v.num_coeffs()).map(|i| v.coeff(i)).collect()
}

fn assert_bits(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}: coefficient {i}");
    }
}

pub(crate) fn verbs<V: InnerProductVector<Scalar = f64>>(
    u: &V,
    v: &V,
) -> (f64, f64, Vec<f64>, Vec<f64>) {
    let d = u.dot(v);
    let n = u.norm();
    let mut a = u.clone();
    a.axpy(-0.75, v);
    let mut s = u.clone();
    s.scale(-2.5);
    (d, n, coeffs(&a), coeffs(&s))
}

fn check_verbs<V: InnerProductVector<Scalar = f64> + Send + Sync>(u: &V, v: &V, what: &str) {
    assert!(
        u.num_coeffs() > 0,
        "{what}: fixture is empty — the test would be vacuous"
    );
    let (ds, ns, as_, ss) = serial_arm(|| verbs(u, v));
    let one = parallel_arm(1, || verbs(u, v));
    assert_eq!(
        one.0.to_bits(),
        ds.to_bits(),
        "{what}: a 1-thread pool must take the serial body (dot)"
    );
    assert_eq!(
        one.1.to_bits(),
        ns.to_bits(),
        "{what}: a 1-thread pool must take the serial body (norm)"
    );

    let mut many = Vec::new();
    for nt in [8usize, 16] {
        let (dp, np, ap, sp) = parallel_arm(nt, || verbs(u, v));
        assert!(
            close(ds, dp, ds.abs().max(ns * ns)),
            "{what}: dot at {nt} threads — serial {ds}, parallel {dp}"
        );
        assert!(
            close(ns, np, ns),
            "{what}: norm at {nt} threads — serial {ns}, parallel {np}"
        );
        assert_bits(&as_, &ap, &format!("{what}: axpy at {nt} threads"));
        assert_bits(&ss, &sp, &format!("{what}: scale at {nt} threads"));
        many.push((dp, np));
    }
    assert_eq!(
        many[0].0.to_bits(),
        many[1].0.to_bits(),
        "{what}: dot differs between 8 and 16 threads — the reassociation is schedule-dependent"
    );
    assert_eq!(
        many[0].1.to_bits(),
        many[1].1.to_bits(),
        "{what}: norm differs between 8 and 16 threads — the reassociation is schedule-dependent"
    );
}

#[test]
fn va1_sparsecore_verbs_match_serial() {
    let _g = lock();
    let space = QnSpace::new(6, 2, 2);
    let mut rng = seeded(0x5A11);
    let u: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    let v: TtVector<f64> = qntt::tt_randn(space, &u.ranks.clone(), &mut rng);
    for k in 0..space.d {
        check_verbs(u.core(k), v.core(k), &format!("SparseCore site {k}"));
    }
}

#[test]
fn va2_contracted_verbs_match_serial() {
    let _g = lock();
    let space = QnSpace::new(6, 2, 2);
    let mut rng = seeded(0x5A22);
    let u: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    let v: TtVector<f64> = qntt::tt_randn(space, &u.ranks.clone(), &mut rng);
    for k in 0..space.d - 1 {
        let cu = contract(u.core(k), u.core(k + 1));
        let cv = contract(v.core(k), v.core(k + 1));
        check_verbs(&cu, &cv, &format!("ContractedSparseCores pair {k}"));
    }
}

#[test]
fn va3_tangent_verbs_match_serial() {
    let _g = lock();
    let space = QnSpace::new(6, 2, 2);
    let mut rng = seeded(0x5A33);
    let x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    let d1: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    let d2: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    let u = TtTangent::project(&base_l, &base_r, &d1);
    let v = TtTangent::project(&base_l, &base_r, &d2);
    check_verbs(&u, &v, "TtTangent");

    let run = |par: bool| {
        let body = || {
            let mut a = u.clone();
            a.axpby(0.3, &v, -1.7);
            a
        };
        let a = if par {
            parallel_arm(16, body)
        } else {
            serial_arm(body)
        };
        coeffs(&a)
    };
    assert_bits(&run(false), &run(true), "TtTangent::axpby");
}

fn su2_dot_center_oracle(a: &Su2Core<f64>, b: &Su2Core<f64>, space: &Su2Space) -> f64 {
    let mut acc = 0.0f64;
    for f in Su2Family::ALL {
        for l in a.transitions(f).collect::<Vec<_>>() {
            let t = f.apply(l, a.space.pg.irrep_of(a.k)).expect("admissible");
            let w = (t.two_s + 1) as f64 / (space.two_s + 1) as f64;
            let (ab, bb) = (a.block(f, l).unwrap(), b.block(f, l).unwrap());
            let mut blk = 0.0f64;
            for j in 0..ab.ncols() {
                for i in 0..ab.nrows() {
                    blk += *ab.get(i, j) * *bb.get(i, j);
                }
            }
            acc += w * blk;
        }
    }
    acc
}

#[test]
fn va4_su2_core_centerweighted_verbs_match_serial() {
    let _g = lock();
    let space = Su2Space::new(6, 2, 2);
    let mut rng = seeded(0x5A44);
    let u: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let v: Su2TtVector<f64> = qntt::su2::vector::tt_randn(space, &u.ranks.clone(), &mut rng);
    for k in 0..space.d {
        check_verbs(u.core(k), v.core(k), &format!("Su2Core site {k}"));
        let oracle = su2_dot_center_oracle(u.core(k), v.core(k), &space);
        let par = parallel_arm(16, || InnerProductVector::dot(u.core(k), v.core(k)));
        assert!(
            close(oracle, par, oracle.abs()),
            "Su2Core site {k}: parallel centerweighted dot {par} vs oracle {oracle}"
        );
    }
}

#[test]
fn va5_su2_contracted_verbs_match_serial() {
    let _g = lock();
    let space = Su2Space::new(6, 2, 2);
    let mut rng = seeded(0x5A55);
    let u: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let v: Su2TtVector<f64> = qntt::su2::vector::tt_randn(space, &u.ranks.clone(), &mut rng);
    for k in 0..space.d - 1 {
        let cu = su2_contract(u.core(k), u.core(k + 1));
        let cv = su2_contract(v.core(k), v.core(k + 1));
        check_verbs(&cu, &cv, &format!("Su2ContractedCores pair {k}"));
    }
}

#[test]
fn va6_su2_tangent_verbs_match_serial() {
    let _g = lock();
    let space = Su2Space::new(6, 2, 2);
    let mut rng = seeded(0x5A66);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);
    let d1: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    let d2: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    let u = Su2TtTangent::project(&base_l, &base_r, &d1);
    let v = Su2TtTangent::project(&base_l, &base_r, &d2);
    check_verbs(&u, &v, "Su2TtTangent");

    let oracle: f64 = (0..space.d)
        .map(|k| su2_dot_center_oracle(u.component(k), v.component(k), &space))
        .sum();
    let par = parallel_arm(16, || u.dot(&v));
    assert!(
        close(oracle, par, oracle.abs()),
        "Su2TtTangent: parallel centerweighted dot {par} vs oracle {oracle}"
    );
}

#[test]
fn va7_tangent_basis_matches_serial() {
    let _g = lock();
    use qntt::core::tangent::tangent_basis;
    let space = QnSpace::new(4, 1, 1);
    let mut rng = seeded(0x5A77);
    let x: TtVector<f64> = randomized_state(space, 2, 3, &mut rng);
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);

    let ser = serial_arm(|| tangent_basis(&base_l, &base_r, true));
    assert!(!ser.is_empty(), "empty tangent basis — vacuous test");
    for nt in [1usize, 8, 16] {
        let par = parallel_arm(nt, || tangent_basis(&base_l, &base_r, true));
        assert_eq!(
            ser.len(),
            par.len(),
            "tangent_basis: dimension changed at {nt} threads"
        );
        for (j, (a, b)) in ser.iter().zip(par.iter()).enumerate() {
            let (ca, cb) = (coeffs(a), coeffs(b));
            for (i, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-11,
                    "tangent_basis vector {j} coefficient {i} at {nt} threads: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn va8_su2_tangent_basis_matches_serial() {
    let _g = lock();
    use qntt::su2::weingarten::tangent_basis as su2_tangent_basis;
    let space = Su2Space::new(4, 2, 0);
    let mut rng = seeded(0x5A88);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    let mut base_l = x.clone();
    base_l.left_orthogonalize(true);
    let mut base_r = x.clone();
    base_r.right_orthogonalize(true);

    let ser = serial_arm(|| su2_tangent_basis(&base_l, &base_r, true));
    assert!(!ser.is_empty(), "empty SU(2) tangent basis — vacuous test");
    for nt in [1usize, 8, 16] {
        let par = parallel_arm(nt, || su2_tangent_basis(&base_l, &base_r, true));
        assert_eq!(
            ser.len(),
            par.len(),
            "su2 tangent_basis: dimension changed at {nt} threads"
        );
        for (j, (a, b)) in ser.iter().zip(par.iter()).enumerate() {
            let (ca, cb) = (coeffs(a), coeffs(b));
            for (i, (x, y)) in ca.iter().zip(cb.iter()).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-11,
                    "su2 tangent_basis vector {j} coefficient {i} at {nt} threads: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn va9_block_slice_shapes_are_bit_identical() {
    let _g = lock();
    let space = QnSpace::new(6, 2, 2);
    let mut rng = seeded(0x5A99);
    let u: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
    let v: TtVector<f64> = qntt::tt_randn(space, &u.ranks.clone(), &mut rng);

    let run = |slice: bool| {
        let prev_par = set_vector_par(false);
        let prev_shape = set_block_slice(slice);
        let mut out: Vec<f64> = Vec::new();
        for k in 0..space.d {
            let (a, b) = (u.core(k), v.core(k));
            out.push(InnerProductVector::dot(a, b));
            out.push(InnerProductVector::norm(a));
            let mut ax = a.clone();
            ax.axpy(-0.75, b);
            out.extend(coeffs(&ax));
            let mut sc = a.clone();
            sc.scale(-2.5);
            out.extend(coeffs(&sc));
        }
        set_block_slice(prev_shape);
        set_vector_par(prev_par);
        out
    };
    let elem = run(false);
    let slice = run(true);
    assert!(!elem.is_empty(), "vacuous fixture");
    assert_bits(&elem, &slice, "BLOCK_SLICE shapes");

    let sspace = Su2Space::new(6, 2, 2);
    let mut srng = seeded(0x5A9A);
    let su: Su2TtVector<f64> = randomized_su2_state(sspace, 2, 4, &mut srng);
    let sv: Su2TtVector<f64> = qntt::su2::vector::tt_randn(sspace, &su.ranks.clone(), &mut srng);
    let srun = |slice: bool| {
        let prev_par = set_vector_par(false);
        let prev_shape = set_block_slice(slice);
        let mut out: Vec<f64> = Vec::new();
        for k in 0..sspace.d {
            out.push(InnerProductVector::dot(su.core(k), sv.core(k)));
            out.push(InnerProductVector::norm(su.core(k)));
        }
        set_block_slice(prev_shape);
        set_vector_par(prev_par);
        out
    };
    assert_bits(&srun(false), &srun(true), "SU(2) BLOCK_SLICE shapes");
}
