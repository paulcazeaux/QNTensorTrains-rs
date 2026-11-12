// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) two-site sweeps with LOCAL MODE ROTATIONS — the twin of [`crate::solvers::moderot`]:
//! the SU(2) [`crate::su2::solvers::mals`] driver with one extra step per bond. After the
//! local eigenproblem on sites `(k, k+1)`, rotate the two spatial orbitals by the angle that
//! minimizes the bond's entanglement functional, THEN truncate. The accumulated rotation `u` is
//! returned with the state; both live in the ROTATED basis (`h` is refilled to
//! `H(rotate_integrals(ints, u))` at every rotation, so the returned triple is consistent by
//! construction). The option and history types are the abelian driver's, shared
//! (scalar-only).

use std::time::Instant;

use crate::core::twosite::noise_rng;
use crate::hamiltonian::instrument;
use crate::integrals::Integrals;
use crate::rotation::{
    compose_rotations, givens, identity_matrix, orthogonality_defect, rotate_integrals,
};
use crate::scalar::Scalar;
use crate::solvers::envstore::{SpillMode, spill_mode};
use crate::solvers::mals::{MalsOptions, NoiseType, SweepTrunc};
use crate::solvers::moderot::{assert_irrep_blocked, assert_same_site_order};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::channel::Su2TouchIndex;
use crate::su2::hamiltonian::matvec::{
    Su2GaugeCache, Su2GaugeMode, gauge_after_pair_rotation, su2_onesite_workspace,
};
use crate::su2::hamiltonian::{Su2SparseHamiltonian, Su2TermScreen, rayleigh_quotient};
use crate::su2::solvers::als::{INNER_TOL_CAP, boundary_idsketch};
use crate::su2::solvers::envstore::Su2EnvStore;
use crate::su2::solvers::mals::{build_right_store, local_ground_2s, split_two_site};
use crate::su2::twosite::{Su2ContractedCores, apply_mode_rotation, contract, search_angle};
use crate::su2::vector::Su2TtVector;
use rand_xoshiro::Xoshiro256PlusPlus;

pub use crate::core::twosite::ModeRotOpts;
pub use crate::solvers::moderot::{ModeRotHistory, ModeRotSweep, ModeRotSweepOpts};

#[derive(Default)]
struct RotStats {
    n_gates: usize,
    max_theta: f64,
    functional_drop: f64,
    search_s: f64,
    refill_s: f64,
    thetas: Vec<f64>,
}

pub(crate) struct RotState {
    opts: ModeRotOpts,
    idx: Su2TouchIndex,
    cache: Su2GaugeCache,
    mode: Su2GaugeMode,
    pub(crate) u: Vec<f64>,
    stats: RotStats,
    gauge_s: f64,
    plan_s: f64,
}

pub(crate) fn parse_su2_gauge(raw: Option<&str>) -> Su2GaugeMode {
    match raw {
        None | Some("" | "mix") => Su2GaugeMode::MixA,
        Some("recontract") => Su2GaugeMode::Recontract,
        Some(v) => panic!("QNTT_SU2_GAUGE={v:?}: expected `mix` (the default) or `recontract`"),
    }
}

impl RotState {
    pub(crate) fn new<T: Scalar>(h: &Su2SparseHamiltonian<T>, opts: ModeRotOpts) -> Self {
        let raw = crate::knobs::env_lookup("QNTT_SU2_GAUGE");
        Self::with_gauge_mode(h, opts, parse_su2_gauge(raw.as_deref()))
    }

    pub(crate) fn with_gauge_mode<T: Scalar>(
        h: &Su2SparseHamiltonian<T>,
        opts: ModeRotOpts,
        mode: Su2GaugeMode,
    ) -> Self {
        let d = h.space.d;
        let t0 = Instant::now();
        let idx = Su2TouchIndex::new(h);
        let plan_s = t0.elapsed().as_secs_f64();
        RotState {
            opts,
            idx,
            cache: Su2GaugeCache::new(d),
            mode,
            u: identity_matrix(d),
            stats: RotStats::default(),
            gauge_s: 0.0,
            plan_s,
        }
    }

    #[cfg(test)]
    pub(crate) fn gauge_mode(&self) -> Su2GaugeMode {
        self.mode
    }

    #[cfg(test)]
    pub(crate) fn n_gates(&self) -> usize {
        self.stats.n_gates
    }

    #[cfg(test)]
    pub(crate) fn thetas(&self) -> &[f64] {
        &self.stats.thetas
    }
}

#[cfg(test)]
pub(crate) struct SplitView<'a> {
    pub(crate) h: &'a Su2SparseHamiltonian<f64>,
    pub(crate) x: &'a Su2TtVector<f64>,
    pub(crate) k: usize,
    pub(crate) fl_k: &'a Su2Sketch<f64>,
    pub(crate) fr_k1: &'a Su2Sketch<f64>,
    pub(crate) forward: bool,
    pub(crate) rot: Option<&'a RotState>,
}

#[cfg(test)]
pub(crate) type SplitProbe = Box<dyn FnMut(&SplitView<'_>)>;

#[cfg(test)]
thread_local! {
    static SPLIT_PROBE: std::cell::RefCell<Option<SplitProbe>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn set_split_probe(probe: Option<SplitProbe>) -> Option<SplitProbe> {
    SPLIT_PROBE.with(|p| p.replace(probe))
}

#[cfg(test)]
fn probe_split<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    k: usize,
    fl_k: &Su2Sketch<T>,
    fr_k1: &Su2Sketch<T>,
    forward: bool,
    rot: Option<&RotState>,
) {
    use std::any::Any;
    SPLIT_PROBE.with(|p| {
        let mut p = p.borrow_mut();
        let Some(probe) = p.as_mut() else { return };
        let (Some(h), Some(x), Some(fl_k), Some(fr_k1)) = (
            (h as &dyn Any).downcast_ref::<Su2SparseHamiltonian<f64>>(),
            (x as &dyn Any).downcast_ref::<Su2TtVector<f64>>(),
            (fl_k as &dyn Any).downcast_ref::<Su2Sketch<f64>>(),
            (fr_k1 as &dyn Any).downcast_ref::<Su2Sketch<f64>>(),
        ) else {
            return;
        };
        probe(&SplitView {
            h,
            x,
            k,
            fl_k,
            fr_k1,
            forward,
            rot,
        });
    });
}

fn gauge_stores<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    i: usize,
    theta: f64,
    rot: &mut RotState,
    fl: &mut Su2EnvStore<T>,
    fr: &mut Su2EnvStore<T>,
) {
    let d = x.space.d;
    let mut left: Vec<Su2Sketch<T>> = (0..=i)
        .map(|b| fl.take(b).expect("Fᴸ resident at the hook"))
        .collect();
    let mut right: Vec<Su2Sketch<T>> = (i + 1..d)
        .map(|s| fr.take(s).expect("Fᴿ resident at the hook"))
        .collect();
    gauge_after_pair_rotation(
        h,
        x,
        i,
        theta,
        &rot.idx,
        &mut left,
        &mut right,
        &mut rot.cache,
        rot.mode,
    );
    for (b, f) in left.into_iter().enumerate() {
        fl.put(b, f);
    }
    for (o, f) in right.into_iter().enumerate() {
        fr.put(i + 1 + o, f);
    }
}

#[allow(clippy::too_many_arguments)]
fn rotate_pair<T: Scalar>(
    h: &mut Su2SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &Su2TtVector<T>,
    fl: &mut Su2EnvStore<T>,
    fr: &mut Su2EnvStore<T>,
    rot: &mut RotState,
    cp: Su2ContractedCores<T>,
    k: usize,
) -> Su2ContractedCores<T> {
    let d = x.space.d;
    let pg = x.space.pg;
    if pg.irrep_of(k) != pg.irrep_of(k + 1) {
        rot.stats.thetas.push(0.0);
        return cp;
    }
    let t0 = Instant::now();
    let (theta, f0, f1) = search_angle(&cp, &rot.opts);
    rot.stats.search_s += t0.elapsed().as_secs_f64();
    rot.stats.thetas.push(theta);
    if theta == 0.0 {
        return cp;
    }
    let cp = apply_mode_rotation(&cp, theta);
    rot.u = compose_rotations(&rot.u, &givens(d, k, theta), d);
    let dev = orthogonality_defect(&rot.u, d);
    assert!(
        dev < 1e-12,
        "su2 mals_moderot: accumulated rotation lost orthogonality (max|uᵀu − I| = {dev:.3e})"
    );
    assert_irrep_blocked(&rot.u, pg, d, "su2 mals_moderot");
    let t0 = Instant::now();
    let rotated = rotate_integrals(ints, &rot.u);
    h.refill(&rotated.t, &rotated.v, None);
    rot.stats.refill_s += t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    gauge_stores(h, x, k, theta, rot, fl, fr);
    rot.gauge_s += t0.elapsed().as_secs_f64();
    rot.stats.n_gates += 1;
    rot.stats.max_theta = rot.stats.max_theta.max(theta.abs());
    rot.stats.functional_drop += f0 - f1;
    cp
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moderot_forward_sweep<T: Scalar>(
    h: &mut Su2SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &mut Su2TtVector<T>,
    fr: &mut Su2EnvStore<T>,
    mut rot: Option<&mut RotState>,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, Su2EnvStore<T>, SweepTrunc) {
    debug_assert_eq!(x.ortho, Some(0), "MALS forward sweep expects center 0");
    let d = x.space.d;
    let mut fl = Su2EnvStore::new(x.space, d, "L");
    fl.put(0, boundary_idsketch(x, 0, 1));
    fr.drop_slot(0);
    fr.prefetch(1);
    let mut lambda = 0.0f64;
    let mut trunc = SweepTrunc::default();

    for k in 0..(d - 1) {
        fr.prefetch(k + 2);
        let _t = instrument::tic();
        let cp0 = contract(x.core(k), x.core(k + 1));
        instrument::stage_end(instrument::stage::CONTRACT, _t);
        let _t = instrument::tic();
        let (theta, cp) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            local_ground_2s(h, fl_k, fr_k1, &cp0, inner_tol)
        };
        instrument::stage_end(instrument::stage::DAVIDSON, _t);
        lambda = theta;
        let cp = match rot.as_deref_mut() {
            Some(r) => rotate_pair(h, ints, x, &mut fl, fr, r, cp, k),
            None => cp,
        };
        let _t = instrument::tic();
        let (xk, s, mut xk1, info) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            #[cfg(test)]
            probe_split(h, x, k, fl_k, fr_k1, true, rot.as_deref());
            split_two_site(
                h, fl_k, fr_k1, &cp, ep, max_rank, noise, noise_type, true, rng,
            )
        };
        instrument::stage_end(instrument::stage::SPLIT, _t);
        trunc.absorb(info);
        fr.drop_slot(k + 1);
        xk1.fold_sketch_left_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k + 1);
        if k < d - 2 {
            let _t = instrument::tic();
            let ws = su2_onesite_workspace(h, x.core(k));
            let next = h.sketching_step_right(x, fl.get(k), &ws);
            fl.put(k + 1, next);
            fl.release(k);
            instrument::stage_end(instrument::stage::ENV, _t);
        }
    }
    (lambda, fl, trunc)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn moderot_back_sweep<T: Scalar>(
    h: &mut Su2SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &mut Su2TtVector<T>,
    fl: &mut Su2EnvStore<T>,
    mut rot: Option<&mut RotState>,
    ep: f64,
    max_rank: usize,
    inner_tol: f64,
    noise: f64,
    noise_type: NoiseType,
    rng: &mut Xoshiro256PlusPlus,
) -> (f64, Su2EnvStore<T>, SweepTrunc) {
    let d = x.space.d;
    if x.ortho != Some(d - 1) {
        x.move_core(d - 1, false);
    }
    let mut fr = Su2EnvStore::new(x.space, d, "R");
    fr.put(d - 1, boundary_idsketch(x, d, 1));
    fl.drop_slot(d - 1);
    if d >= 2 {
        fl.prefetch(d - 2);
    }
    let mut lambda = 0.0f64;
    let mut trunc = SweepTrunc::default();

    for k in (0..(d - 1)).rev() {
        if k >= 1 {
            fl.prefetch(k - 1);
        }
        let _t = instrument::tic();
        let cp0 = contract(x.core(k), x.core(k + 1));
        instrument::stage_end(instrument::stage::CONTRACT, _t);
        let _t = instrument::tic();
        let (theta, cp) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            local_ground_2s(h, fl_k, fr_k1, &cp0, inner_tol)
        };
        instrument::stage_end(instrument::stage::DAVIDSON, _t);
        lambda = theta;
        let cp = match rot.as_deref_mut() {
            Some(r) => rotate_pair(h, ints, x, fl, &mut fr, r, cp, k),
            None => cp,
        };
        let _t = instrument::tic();
        let (mut xk, s, xk1, info) = {
            let fl_k = fl.get(k);
            let fr_k1 = fr.get(k + 1);
            #[cfg(test)]
            probe_split(h, x, k, fl_k, fr_k1, false, rot.as_deref());
            split_two_site(
                h, fl_k, fr_k1, &cp, ep, max_rank, noise, noise_type, false, rng,
            )
        };
        instrument::stage_end(instrument::stage::SPLIT, _t);
        trunc.absorb(info);
        fl.drop_slot(k);
        xk.fold_sketch_right_mut(&s);
        x.set_cores(xk, xk1);
        x.ortho = Some(k);
        if k > 0 {
            let _t = instrument::tic();
            let ws = su2_onesite_workspace(h, x.core(k + 1));
            let next = h.sketching_step_left(x, fr.get(k + 1), &ws);
            fr.put(k, next);
            fr.release(k + 1);
            instrument::stage_end(instrument::stage::ENV, _t);
        }
    }
    (lambda, fr, trunc)
}

/// Rank-adaptive SU(2) two-site ground-state search with local mode rotations (module doc) — the
/// twin of [`crate::solvers::moderot::mals_moderot`]. `h` must be the refillable
/// `SymmetryAllowed(space.pg)` operator of `ints` at `u = I` on `x`'s space; on return it is the
/// operator of the returned rotation, `x` is the state in that basis, and the energy is the
/// ELECTRONIC one (`history.lambda`: the last back-sweep local Ritz value, pre-truncation). With
/// rotations off (`first_rot_iter` past the schedule) or every angle exactly `0`, the run is
/// [`crate::su2::solvers::mals::mals_mut`] bit for bit.
///
/// # Panics
///
/// On an operator that is not a refillable `Su2TermScreen::SymmetryAllowed(space.pg)` build (a
/// `Magnitude` or `Mask` screen, or a plain build), an operator of another space, integrals of
/// another `norb`, `d < 2`, empty `bond_dims` / `noise` schedules, and a negative or non-finite
/// noise amplitude. On a typed space whose `ints` carry an
/// ORBSYM, also when a site's irrep differs between `ints` and the space (the two are not in one
/// site order — e.g. only one of them went through `irrep_blocked_order`). On a malformed
/// `QNTT_SU2_GAUGE`.
pub fn mals_moderot<T: Scalar>(
    h: &mut Su2SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &mut Su2TtVector<T>,
    opts: &MalsOptions,
    sweep: &ModeRotSweepOpts,
) -> (ModeRotHistory, Vec<f64>) {
    moderot_run(h, ints, x, opts, sweep, None)
}

#[cfg(test)]
pub(crate) fn mals_moderot_with_gauge_mode<T: Scalar>(
    h: &mut Su2SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &mut Su2TtVector<T>,
    opts: &MalsOptions,
    sweep: &ModeRotSweepOpts,
    mode: Su2GaugeMode,
) -> (ModeRotHistory, Vec<f64>) {
    moderot_run(h, ints, x, opts, sweep, Some(mode))
}

fn moderot_run<T: Scalar>(
    h: &mut Su2SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &mut Su2TtVector<T>,
    opts: &MalsOptions,
    sweep: &ModeRotSweepOpts,
    mode: Option<Su2GaugeMode>,
) -> (ModeRotHistory, Vec<f64>) {
    let space = x.space;
    let d = space.d;
    assert!(
        h.term_screen() == Some(Su2TermScreen::SymmetryAllowed(space.pg)),
        "su2 mals_moderot: the operator must be a refillable build on \
         Su2TermScreen::SymmetryAllowed(space.pg) (`Su2SparseHamiltonian::new_refillable`) — a \
         Magnitude or Mask skeleton leaks at the first rotation"
    );
    assert!(
        h.space == space,
        "su2 mals_moderot: the operator and the state live in different spaces"
    );
    assert_eq!(
        ints.norb, d,
        "su2 mals_moderot: the integrals' norb must be the chain length"
    );
    assert_same_site_order(ints, space.pg, d, "su2 mals_moderot");
    assert!(d >= 2, "su2 mals_moderot needs at least two sites");
    assert!(
        !opts.bond_dims.is_empty(),
        "MalsOptions::bond_dims must be non-empty (use [0] for uncapped)"
    );
    assert!(
        !opts.noise.is_empty(),
        "MalsOptions::noise must be non-empty (use [0.0] for a noise-free run)"
    );
    assert!(
        opts.noise.iter().all(|a| a.is_finite() && *a >= 0.0),
        "MalsOptions::noise entries must be finite and non-negative"
    );
    assert!(
        matches!(spill_mode(), SpillMode::Mem),
        "su2 mals_moderot refuses a spilling environment store (QNTT_ENV_SPILL): the gauge \
         rewrites the cached stacks in place"
    );

    let mut rot = match mode {
        Some(mode) => RotState::with_gauge_mode(h, sweep.rot, mode),
        None => RotState::new(h, sweep.rot),
    };
    let gauge_plan_s = rot.plan_s;
    let gauge_plan_bytes = rot.idx.bytes();
    let mut rng = noise_rng(opts.seed);
    let mut sweeps: Vec<ModeRotSweep> = Vec::with_capacity(opts.max_iter);
    let mut all_thetas = Vec::with_capacity(opts.max_iter);
    if x.ortho != Some(0) {
        x.right_orthogonalize(true);
    }
    let mut fr = build_right_store(h, x);
    let mut lambda = rayleigh_quotient(h, x).re_s();
    let mut r = 1.0f64;

    for it in 0..opts.max_iter {
        let t_it = Instant::now();
        let bond_dim = opts.bond_dims[it.min(opts.bond_dims.len() - 1)];
        let noise = opts.noise[it.min(opts.noise.len() - 1)];
        let raw = 0.01 * r;
        let inner_tol = if raw > INNER_TOL_CAP {
            INNER_TOL_CAP
        } else if raw < opts.eps {
            opts.eps
        } else {
            raw
        };
        let ep = if d > 1 {
            opts.eps / ((d - 1) as f64).sqrt()
        } else {
            opts.eps
        };
        let rotate = it >= sweep.first_rot_iter;
        let gauge_before = rot.gauge_s;

        let (lf, mut fl, tf) = moderot_forward_sweep(
            h,
            ints,
            x,
            &mut fr,
            rotate.then_some(&mut rot),
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.noise_type,
            &mut rng,
        );
        if opts.verbose {
            println!("SU(2) MALS+moderot iter {it} forward sweep: energy = {lf}");
        }
        debug_assert!(
            fr.all_empty(),
            "L1: the forward sweep must consume the whole right stack"
        );
        let (lambda_n, fr_new, tb) = moderot_back_sweep(
            h,
            ints,
            x,
            &mut fl,
            rotate.then_some(&mut rot),
            ep,
            bond_dim,
            inner_tol,
            noise,
            opts.noise_type,
            &mut rng,
        );
        debug_assert!(
            fl.all_empty(),
            "L1: the back sweep must consume the whole left stack"
        );
        fr = fr_new;

        let trunc = tf.merge(tb);
        r = (lambda - lambda_n).abs() / lambda_n.abs();
        lambda = lambda_n;
        let stats = std::mem::take(&mut rot.stats);
        sweeps.push(ModeRotSweep {
            energy: lambda,
            discarded_weight: trunc.max_discarded,
            max_bond_dim: trunc.max_kept,
            bond_dim,
            noise,
            n_gates: stats.n_gates,
            max_theta: stats.max_theta,
            functional_drop: stats.functional_drop,
            search_s: stats.search_s,
            refill_s: stats.refill_s,
            gauge_s: rot.gauge_s - gauge_before,
            wall_s: t_it.elapsed().as_secs_f64(),
        });
        all_thetas.push(stats.thetas);
        if opts.verbose {
            let s = sweeps.last().expect("pushed above");
            println!(
                "SU(2) MALS+moderot iter {it}: energy = {lambda}, rel = {r:.3e}, gates = {}, \
                 max|θ| = {:.3e}, Δf = {:.3e}, M = {}, discarded = {:.3e}, noise = {noise:.1e} | \
                 search {:.2}s refill {:.2}s gauge {:.2}s",
                s.n_gates,
                s.max_theta,
                s.functional_drop,
                s.max_bond_dim,
                s.discarded_weight,
                s.search_s,
                s.refill_s,
                s.gauge_s
            );
        }
        if noise == 0.0 && r < opts.convergence_tol() && inner_tol <= opts.eps {
            break;
        }
    }
    let u_orth_dev = orthogonality_defect(&rot.u, d);
    (
        ModeRotHistory {
            lambda,
            sweeps,
            thetas: all_thetas,
            u_orth_dev,
            gauge_plan_s,
            gauge_plan_bytes,
        },
        rot.u,
    )
}

#[cfg(test)]
#[path = "moderot_tests.rs"]
mod tests;
