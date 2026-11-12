// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use super::{
    ModeRotSweepOpts, RotState, SplitView, mals_moderot_with_gauge_mode, moderot_back_sweep,
    moderot_forward_sweep, parse_su2_gauge, set_split_probe,
};
use crate::core::twosite::{ModeRotOpts, noise_rng};
use crate::integrals::{
    Integrals, IntegralsOptions, PointGroupPolicy, load_fcidump, load_fixture_dir,
};
use crate::ordering::{irrep_blocked_order, permute_integrals};
use crate::rotation::{compose_rotations, givens, rotate_integrals};
use crate::solvers::mals::{MalsOptions, NoiseType};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::matvec::Su2GaugeMode;
use crate::su2::hamiltonian::{
    Su2PatternMask, Su2SparseHamiltonian, Su2TermScreen, rayleigh_quotient,
};
use crate::su2::ops::dot;
use crate::su2::qn::Su2Space;
use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::su2::solvers::mals::{build_right_store, mals_with_history};
use crate::su2::vector::Su2TtVector;
use crate::test_util::{randomized_su2_state, seeded};

const TOL: f64 = 1e-10;
const ANTI: f64 = 1e-6;

fn testdata(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn h4() -> Integrals<f64> {
    load_fcidump::<f64>(
        &testdata("testdata/h4.FCIDUMP"),
        None,
        &IntegralsOptions::default(),
    )
    .expect("load h4.FCIDUMP")
    .ints
}

fn lih() -> Integrals<f64> {
    load_fixture_dir::<f64>(
        &testdata("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g")
    .ints
}

fn h2o_blocked(two_s: usize) -> (Integrals<f64>, Su2Space) {
    let o = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load =
        load_fixture_dir::<f64>(&testdata("testdata/h2o_sto3g"), &o).expect("load h2o_sto3g typed");
    let (pg0, d) = (load.pg.pg, load.ints.norb);
    assert!(!pg0.is_trivial(), "h2o_sto3g must resolve C2v");
    let perm = irrep_blocked_order(pg0, d);
    let ints = permute_integrals(&load.ints, &perm);
    let pgd = ints
        .point_group(Some(load.pg.kind))
        .expect("permuted dump resolves");
    let same: Vec<usize> = (0..d - 1)
        .filter(|&k| pgd.pg.irrep_of(k) == pgd.pg.irrep_of(k + 1))
        .collect();
    assert_eq!(
        same,
        vec![0, 1, 2, 4],
        "irrep-blocked h2o C2v same-irrep bonds"
    );
    let space = Su2Space::with_point_group(d, ints.nelec, two_s, pgd.pg, pgd.target);
    (ints, space)
}

fn sa_refillable(ints: &Integrals<f64>, space: Su2Space) -> Su2SparseHamiltonian<f64> {
    Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::SymmetryAllowed(space.pg),
    )
}

fn rq(h: &Su2SparseHamiltonian<f64>, x: &Su2TtVector<f64>) -> f64 {
    rayleigh_quotient(h, x) / dot(x, x)
}

fn rel_dev(a: &Su2Sketch<f64>, b: &Su2Sketch<f64>, what: &str) -> f64 {
    assert_eq!(a.site(), b.site(), "{what}: bonds");
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for s in b.sectors() {
        match (a.block(s), b.block(s)) {
            (None, None) => {}
            (Some(p), Some(q)) => {
                assert_eq!(
                    (p.nrows(), p.ncols()),
                    (q.nrows(), q.ncols()),
                    "{what}: bond {} sector {s:?} shape",
                    b.site()
                );
                for c in 0..q.ncols() {
                    for r in 0..q.nrows() {
                        num = num.max((p[(r, c)] - q[(r, c)]).abs());
                        den = den.max(q[(r, c)].abs());
                    }
                }
            }
            _ => panic!(
                "{what}: bond {} sector {s:?}: block presence differs",
                b.site()
            ),
        }
    }
    num / den.max(f64::MIN_POSITIVE)
}

fn t5_arm(name: &str, ints: &Integrals<f64>, space: Su2Space, seed: u64, mode: Su2GaugeMode) {
    let d = space.d;
    let mint = sa_refillable(ints, space);
    let mut h = sa_refillable(ints, space);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(seed));
    x.right_orthogonalize(true);
    let mut fr = build_right_store(&h, &x);
    let mut rot = RotState::with_gauge_mode(&h, ModeRotOpts::default(), mode);
    let mut rng = noise_rng(0);
    let ep = 1e-10 / ((d - 1) as f64).sqrt();
    let inner_tol = 1e-8;

    let (lf, mut fl, _) = moderot_forward_sweep(
        &mut h,
        ints,
        &mut x,
        &mut fr,
        Some(&mut rot),
        ep,
        0,
        inner_tol,
        0.0,
        NoiseType::Random,
        &mut rng,
    );
    let g_fwd = rot.n_gates();
    assert_eq!(rot.thetas().len(), d - 1, "{name}: forward angles");
    let g_fwd_chain = rot.thetas()[1..].iter().filter(|&&t| t != 0.0).count();
    let fresh = sa_refillable(&rotate_integrals(ints, &rot.u), space);
    let e_fwd = rq(&fresh, &x);
    let ofl = left_to_right_sketching(&fresh, &x);
    let stale = left_to_right_sketching(&mint, &x);
    let (mut dev_l, mut moved_l) = (0.0f64, 0.0f64);
    for b in 1..=d - 2 {
        let what = format!("{name}: forward Fᴸ bond {b}");
        dev_l = dev_l.max(rel_dev(fl.get(b), &ofl[b], &what));
        moved_l = moved_l.max(rel_dev(&stale[b], &ofl[b], &what));
    }

    let (lb, mut fr2, _) = moderot_back_sweep(
        &mut h,
        ints,
        &mut x,
        &mut fl,
        Some(&mut rot),
        ep,
        0,
        inner_tol,
        0.0,
        NoiseType::Random,
        &mut rng,
    );
    let g_back = rot.n_gates() - g_fwd;
    let back = &rot.thetas()[d - 1..];
    assert_eq!(back.len(), d - 1, "{name}: back angles");
    let g_back_chain = back[1..].iter().filter(|&&t| t != 0.0).count();
    let fresh = sa_refillable(&rotate_integrals(ints, &rot.u), space);
    let e_back = rq(&fresh, &x);
    let ofr = right_to_left_sketching(&fresh, &x);
    let stale = right_to_left_sketching(&mint, &x);
    let (mut dev_r, mut moved_r) = (0.0f64, 0.0f64);
    for s in 1..=d - 2 {
        let what = format!("{name}: back Fᴿ bond {}", s + 1);
        dev_r = dev_r.max(rel_dev(fr2.get(s), &ofr[s], &what));
        moved_r = moved_r.max(rel_dev(&stale[s], &ofr[s], &what));
    }

    let rel = |a: f64, e: f64| (a - e).abs() / e.abs();
    println!(
        "{name}: gates fwd {g_fwd} ({g_fwd_chain} on the observed chain) back {g_back} \
         ({g_back_chain}); λ_fwd {lf:.12} vs RQ {e_fwd:.12} ({:.1e}); λ_back {lb:.12} vs RQ \
         {e_back:.12} ({:.1e}); live vs fresh Fᴸ {dev_l:.2e} Fᴿ {dev_r:.2e}; fresh under U = I \
         off by Fᴸ {moved_l:.2e} Fᴿ {moved_r:.2e}",
        rel(lf, e_fwd),
        rel(lb, e_back)
    );
    assert!(
        g_fwd_chain >= 1 && g_back_chain >= 1,
        "{name}: vacuous — gates on the observed chain: forward {g_fwd_chain} of {g_fwd} (pairs \
         k ≥ 1), back {g_back_chain} of {g_back} (pairs k ≤ d−3)"
    );
    assert!(
        rel(lf, e_fwd) <= TOL,
        "{name}: forward λ {lf} vs the fresh-operator energy {e_fwd}"
    );
    assert!(
        rel(lb, e_back) <= TOL,
        "{name}: back λ {lb} vs the fresh-operator energy {e_back}"
    );
    assert!(
        dev_l <= TOL,
        "{name}: live Fᴸ after the forward half off the fresh stack by {dev_l:.3e}"
    );
    assert!(
        dev_r <= TOL,
        "{name}: live Fᴿ after the back half off the fresh stack by {dev_r:.3e}"
    );
    assert!(
        moved_l > ANTI && moved_r > ANTI,
        "{name}: anti-oracle — the U = I stacks are within Fᴸ {moved_l:.3e} / Fᴿ {moved_r:.3e} of \
         the rotated ones, so the pin cannot see a missing gauge"
    );
}

#[test]
fn su2_mr_t5_half_sweep_stacks_are_the_fresh_stacks() {
    let mode = Su2GaugeMode::Recontract;
    t5_arm("h4 (4,4,2)", &h4(), Su2Space::new(4, 4, 2), 0x9B_0501, mode);
    t5_arm(
        "lih (6,3,1)",
        &lih(),
        Su2Space::new(6, 3, 1),
        0x9B_0502,
        mode,
    );
    for two_s in [0usize, 2] {
        let (ints, space) = h2o_blocked(two_s);
        let name = format!("h2o C2v blocked (7,10,{two_s})");
        t5_arm(&name, &ints, space, 0x9B_0503 + two_s as u64, mode);
    }
}

struct SplitObs {
    forward: bool,
    k: usize,
    theta: f64,
    dev_l: f64,
    dev_r: f64,
    stale_l: f64,
    stale_r: f64,
}

struct ProbeGuard;

impl Drop for ProbeGuard {
    fn drop(&mut self) {
        set_split_probe(None);
    }
}

fn split_probe_arm(
    name: &str,
    ints: Integrals<f64>,
    space: Su2Space,
    seed: u64,
    mode: Su2GaugeMode,
) {
    let d = space.d;
    let ints = Rc::new(ints);
    let seen: Rc<RefCell<Vec<SplitObs>>> = Rc::default();
    let (ic, sink, tag) = (Rc::clone(&ints), Rc::clone(&seen), name.to_owned());
    let _guard = ProbeGuard;
    set_split_probe(Some(Box::new(move |v: &SplitView<'_>| {
        let what = format!(
            "{tag}: {} pair ({}, {})",
            if v.forward { "forward" } else { "back" },
            v.k,
            v.k + 1
        );
        let r = v.rot.expect("rotations on from iteration 0");
        let theta = *r.thetas().last().expect("this pair's angle");
        let ofl = left_to_right_sketching(v.h, v.x);
        let ofr = right_to_left_sketching(v.h, v.x);
        let dev_l = rel_dev(v.fl_k, &ofl[v.k], &what);
        let dev_r = rel_dev(v.fr_k1, &ofr[v.k + 1], &what);
        let (mut stale_l, mut stale_r) = (0.0, 0.0);
        if theta != 0.0 {
            let u_prev = compose_rotations(&r.u, &givens(d, v.k, -theta), d);
            let hp = sa_refillable(&rotate_integrals(&ic, &u_prev), space);
            stale_l = rel_dev(&left_to_right_sketching(&hp, v.x)[v.k], &ofl[v.k], &what);
            stale_r = rel_dev(
                &right_to_left_sketching(&hp, v.x)[v.k + 1],
                &ofr[v.k + 1],
                &what,
            );
        }
        sink.borrow_mut().push(SplitObs {
            forward: v.forward,
            k: v.k,
            theta,
            dev_l,
            dev_r,
            stale_l,
            stale_r,
        });
    })));

    let mut h = sa_refillable(&ints, space);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(seed));
    let opts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![4],
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Perturbative,
        max_iter: 3,
        ..MalsOptions::default()
    };
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let (hist, _u) = mals_moderot_with_gauge_mode(&mut h, &ints, &mut x, &opts, &sweep, mode);
    drop(_guard);

    let seen = seen.borrow();
    assert_eq!(
        seen.len(),
        hist.sweeps.len() * 2 * (d - 1),
        "{name}: the probe missed a pair"
    );
    let dev = seen
        .iter()
        .map(|o| o.dev_l.max(o.dev_r))
        .fold(0.0, f64::max);
    for half in [true, false] {
        let label = if half { "forward" } else { "back" };
        let mine = seen.iter().filter(|o| o.forward == half);
        let gated: Vec<&SplitObs> = mine.clone().filter(|o| o.theta != 0.0).collect();
        let both: Vec<&SplitObs> = gated
            .iter()
            .copied()
            .filter(|o| o.k >= 1 && o.k + 3 <= d)
            .collect();
        let gdev = gated
            .iter()
            .map(|o| o.dev_l.max(o.dev_r))
            .fold(0.0, f64::max);
        let (sl, sr) = both.iter().fold((0.0f64, 0.0f64), |(a, b), o| {
            (a.max(o.stale_l), b.max(o.stale_r))
        });
        println!(
            "{name} {label}: {} pairs, {} gated ({} with both read slots touched); live vs fresh \
             at the gated pairs {gdev:.2e}; pre-gate environments off by Fᴸ {sl:.2e} Fᴿ {sr:.2e}",
            mine.count(),
            gated.len(),
            both.len()
        );
        assert!(
            !both.is_empty(),
            "{name} {label}: vacuous — no gate at a pair whose two read slots the gauge touches"
        );
        assert!(
            sl > ANTI && sr > ANTI,
            "{name} {label}: anti-oracle — the pre-gate environments are within Fᴸ {sl:.3e} / Fᴿ \
             {sr:.3e} of the gauged ones, so a split before the gauge would pass"
        );
    }
    println!("{name}: live vs fresh over every split {dev:.2e}");
    assert!(
        dev <= TOL,
        "{name}: a split was handed an environment off the fresh stack by {dev:.3e}"
    );
}

#[test]
fn su2_mr_t5b_split_reads_the_gauged_environments() {
    let mode = Su2GaugeMode::Recontract;
    split_probe_arm("h4 (4,4,2)", h4(), Su2Space::new(4, 4, 2), 0x9B_0511, mode);
    split_probe_arm(
        "lih (6,3,1)",
        lih(),
        Su2Space::new(6, 3, 1),
        0x9B_0512,
        mode,
    );
    let (ints, space) = h2o_blocked(2);
    split_probe_arm("h2o C2v blocked (7,10,2)", ints, space, 0x9B_0513, mode);
}

#[test]
fn su2_mr_g2b_t8a_mix_half_sweep_stacks_are_the_fresh_stacks() {
    let mode = Su2GaugeMode::MixA;
    t5_arm("h4 (4,4,2)", &h4(), Su2Space::new(4, 4, 2), 0x9B_0501, mode);
    t5_arm(
        "lih (6,3,1)",
        &lih(),
        Su2Space::new(6, 3, 1),
        0x9B_0502,
        mode,
    );
    for two_s in [0usize, 2] {
        let (ints, space) = h2o_blocked(two_s);
        let name = format!("h2o C2v blocked (7,10,{two_s}) MixA");
        t5_arm(&name, &ints, space, 0x9B_0503 + two_s as u64, mode);
    }
}

#[test]
fn su2_mr_g2b_t8b_mix_split_reads_the_gauged_environments() {
    let mode = Su2GaugeMode::MixA;
    split_probe_arm(
        "h4 (4,4,2) MixA",
        h4(),
        Su2Space::new(4, 4, 2),
        0x9B_0511,
        mode,
    );
    split_probe_arm(
        "lih (6,3,1) MixA",
        lih(),
        Su2Space::new(6, 3, 1),
        0x9B_0512,
        mode,
    );
    let (ints, space) = h2o_blocked(2);
    split_probe_arm(
        "h2o C2v blocked (7,10,2) MixA",
        ints,
        space,
        0x9B_0513,
        mode,
    );
}

fn closed_loop_in(
    ints: &Integrals<f64>,
    space: Su2Space,
    x0: &Su2TtVector<f64>,
    mode: Su2GaugeMode,
) -> (super::ModeRotHistory, Vec<f64>, f64) {
    let opts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        noise: vec![0.0],
        noise_type: NoiseType::Random,
        max_iter: 6,
        ..MalsOptions::default()
    };
    let sweep = ModeRotSweepOpts {
        first_rot_iter: 0,
        ..ModeRotSweepOpts::default()
    };
    let mut h = sa_refillable(ints, space);
    let mut x = x0.clone();
    let (hist, u) = mals_moderot_with_gauge_mode(&mut h, ints, &mut x, &opts, &sweep, mode);
    let e = rq(&sa_refillable(&rotate_integrals(ints, &u), space), &x);
    (hist, u, e)
}

#[test]
fn su2_mr_g2b_t8c_mix_sweep_matches_the_recontract_sweep() {
    for (name, ints, space, seed) in [
        ("lih (6,3,1)", lih(), Su2Space::new(6, 3, 1), 0x9B_0301u64),
        ("h4 (4,4,2)", h4(), Su2Space::new(4, 4, 2), 0x9B_0302),
    ] {
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(seed));
        let (hr, ur, er) = closed_loop_in(&ints, space, &x0, Su2GaugeMode::Recontract);
        let (hm, um, em) = closed_loop_in(&ints, space, &x0, Su2GaugeMode::MixA);
        let gates = |h: &super::ModeRotHistory| h.sweeps.iter().map(|s| s.n_gates).sum::<usize>();
        let (gr, gm) = (gates(&hr), gates(&hm));
        let scale = hr.lambda.abs();
        let du = ur
            .iter()
            .zip(&um)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        let (loop_r, loop_m) = ((hr.lambda - er).abs(), (hm.lambda - em).abs());
        println!(
            "{name}: Recontract λ {:.12} ({} it, {gr} gates) RQ {er:.12}; MixA λ {:.12} ({} it, \
             {gm} gates) RQ {em:.12}; |Δλ| {:.2e}, |ΔRQ| {:.2e}, closed loop Recontract {:.2e} / \
             MixA {:.2e} (relative); max |Δu| {du:.2e}",
            hr.lambda,
            hr.sweeps.len(),
            hm.lambda,
            hm.sweeps.len(),
            (hm.lambda - hr.lambda).abs() / scale,
            (em - er).abs() / scale,
            loop_r / scale,
            loop_m / scale
        );
        assert!(
            gr > 0 && gm > 0,
            "{name}: vacuous — gates Recontract {gr}, MixA {gm}"
        );
        assert!(
            (hm.lambda - hr.lambda).abs() <= 1e-10 * scale,
            "{name}: MixA sweep energy {} vs Recontract {}",
            hm.lambda,
            hr.lambda
        );
        assert!(
            (em - er).abs() <= 1e-10 * scale,
            "{name}: MixA closed-loop energy {em} vs Recontract {er}"
        );
        assert!(
            loop_m <= loop_r + 1e-10 * scale,
            "{name}: MixA closed loop |λ − RQ_fresh| = {loop_m:.3e} vs Recontract's {loop_r:.3e}"
        );
    }
}

fn assert_bits_eq(a: &Su2TtVector<f64>, b: &Su2TtVector<f64>, what: &str) {
    assert_eq!(a.ranks, b.ranks, "{what}: rank grids differ");
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        assert_eq!(ca.num_coeffs(), cb.num_coeffs(), "{what}: core {k} size");
        for i in 0..ca.num_coeffs() {
            assert_eq!(
                ca.coeff(i).to_bits(),
                cb.coeff(i).to_bits(),
                "{what}: core {k} coefficient {i} ({} vs {}) not bitwise equal",
                ca.coeff(i),
                cb.coeff(i)
            );
        }
    }
}

fn mix_bitwise_arm(what: &str, o: &MalsOptions, sweep: &ModeRotSweepOpts, zero_angles: bool) {
    let ints = lih();
    let space = Su2Space::new(6, 3, 1);
    let d = space.d;
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(0x9B_0001));
    let h_ref = sa_refillable(&ints, space);
    let (hist_ref, x_ref) = mals_with_history(&h_ref, &x0, o);
    let mut h = sa_refillable(&ints, space);
    let mut x = x0;
    let (hist, u) =
        mals_moderot_with_gauge_mode(&mut h, &ints, &mut x, o, sweep, Su2GaugeMode::MixA);
    println!(
        "{what}: λ {:.12} in {} iterations (MALS {:.12} in {})",
        hist.lambda,
        hist.sweeps.len(),
        hist_ref.lambda,
        hist_ref.sweeps.len()
    );
    assert_eq!(
        hist.lambda.to_bits(),
        hist_ref.lambda.to_bits(),
        "{what}: λ {} vs the MALS driver's {}",
        hist.lambda,
        hist_ref.lambda
    );
    assert_eq!(
        hist.sweeps.len(),
        hist_ref.sweeps.len(),
        "{what}: iteration count"
    );
    for (it, (a, b)) in hist.sweeps.iter().zip(&hist_ref.sweeps).enumerate() {
        assert_eq!(
            (a.energy.to_bits(), a.discarded_weight.to_bits()),
            (b.energy.to_bits(), b.discarded_weight.to_bits()),
            "{what}: iteration {it} energy / discarded weight"
        );
        assert_eq!(
            (a.max_bond_dim, a.bond_dim, a.noise.to_bits()),
            (b.max_bond_dim, b.bond_dim, b.noise.to_bits()),
            "{what}: iteration {it} ranks / schedule"
        );
        assert_eq!(a.n_gates, 0, "{what}: iteration {it} took a gate");
    }
    assert_bits_eq(&x, &x_ref, what);
    assert!(
        (0..d * d).all(|i| u[i] == if i % (d + 1) == 0 { 1.0 } else { 0.0 }),
        "{what}: u is not exactly the identity"
    );
    assert_eq!(
        h.artifact_hash(),
        h_ref.artifact_hash(),
        "{what}: the operator was refilled"
    );
    for (it, th) in hist.thetas.iter().enumerate() {
        if zero_angles {
            assert_eq!(th.len(), 2 * (d - 1), "{what}: iteration {it} angle slots");
            assert!(
                th.iter().all(|t| t.to_bits() == 0),
                "{what}: iteration {it} recorded a non-(+0.0) angle: {th:?}"
            );
        } else {
            assert!(th.is_empty(), "{what}: iteration {it} recorded angles");
        }
    }
}

#[test]
fn su2_mr_g2b_t8d_mix_without_rotations_is_bitwise_the_mals_driver() {
    let plain = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![4],
        noise: vec![0.0],
        noise_type: NoiseType::Random,
        max_iter: 3,
        ..MalsOptions::default()
    };
    let perturbative = MalsOptions {
        noise: vec![1e-3, 1e-4, 0.0],
        noise_type: NoiseType::Perturbative,
        ..plain.clone()
    };
    let off = ModeRotSweepOpts {
        first_rot_iter: usize::MAX,
        ..ModeRotSweepOpts::default()
    };
    mix_bitwise_arm("T8d off, noise [0]", &plain, &off, false);
    mix_bitwise_arm("T8d off, perturbative", &perturbative, &off, false);
    let zero = ModeRotSweepOpts {
        rot: ModeRotOpts {
            improve_tol: f64::INFINITY,
            ..ModeRotOpts::default()
        },
        first_rot_iter: 0,
    };
    mix_bitwise_arm("T8d zero angles, perturbative", &perturbative, &zero, true);
}

#[test]
fn su2_mr_g2b_t8e_gauge_knob_parses_and_rot_state_reads_it() {
    assert_eq!(parse_su2_gauge(None), Su2GaugeMode::MixA);
    assert_eq!(parse_su2_gauge(Some("")), Su2GaugeMode::MixA);
    assert_eq!(
        parse_su2_gauge(Some("recontract")),
        Su2GaugeMode::Recontract
    );
    assert_eq!(parse_su2_gauge(Some("mix")), Su2GaugeMode::MixA);
    let h = sa_refillable(&h4(), Su2Space::new(4, 4, 2));
    let env = std::env::var("QNTT_SU2_GAUGE").ok();
    assert_eq!(
        RotState::new(&h, ModeRotOpts::default()).gauge_mode(),
        parse_su2_gauge(env.as_deref()),
        "RotState::new must take QNTT_SU2_GAUGE's mode (here {env:?})"
    );
    for mode in [Su2GaugeMode::Recontract, Su2GaugeMode::MixA] {
        assert_eq!(
            RotState::with_gauge_mode(&h, ModeRotOpts::default(), mode).gauge_mode(),
            mode
        );
    }
}

#[test]
#[should_panic(expected = "QNTT_SU2_GAUGE")]
fn su2_mr_g2b_t8f_gauge_knob_refuses_a_malformed_value() {
    let _ = parse_su2_gauge(Some("mixa"));
}

fn forward_half_gates(
    h: &mut Su2SparseHamiltonian<f64>,
    ints: &Integrals<f64>,
    seed: u64,
    mode: Su2GaugeMode,
) -> usize {
    let space = h.space;
    let d = space.d;
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(seed));
    x.right_orthogonalize(true);
    let mut fr = build_right_store(h, &x);
    let mut rot = RotState::with_gauge_mode(h, ModeRotOpts::default(), mode);
    let mut rng = noise_rng(0);
    let ep = 1e-10 / ((d - 1) as f64).sqrt();
    let _ = moderot_forward_sweep(
        h,
        ints,
        &mut x,
        &mut fr,
        Some(&mut rot),
        ep,
        0,
        1e-8,
        0.0,
        NoiseType::Random,
        &mut rng,
    );
    rot.n_gates()
}

#[test]
#[should_panic(expected = "MixA requires a SymmetryAllowed(space.pg) skeleton")]
fn su2_mr_g2b_t8g_the_driver_hands_its_mode_to_the_gauge() {
    let ints = h4();
    let space = Su2Space::new(4, 4, 2);
    let masked = || {
        let m = Su2PatternMask::from_magnitude(&ints.t, &ints.v, f64::EPSILON, space.pg);
        Su2SparseHamiltonian::new_refillable(
            space,
            &ints.t,
            &ints.v,
            None,
            Su2TermScreen::Mask(Arc::new(m)),
        )
    };
    let gates = forward_half_gates(&mut masked(), &ints, 0x9B_0501, Su2GaugeMode::Recontract);
    println!("T8g control: the Recontract half on the Mask build took {gates} gates");
    assert!(
        gates >= 1,
        "vacuous: the Recontract control half took no gate, so MixA's gauge would never run"
    );
    let _ = forward_half_gates(&mut masked(), &ints, 0x9B_0501, Su2GaugeMode::MixA);
}
