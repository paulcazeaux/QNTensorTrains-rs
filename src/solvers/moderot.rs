// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Two-site sweeps with LOCAL MODE ROTATIONS: the [`crate::solvers::mals`] driver with one extra
//! step per bond — after the local eigenproblem on sites `(k, k+1)`, rotate the two spatial
//! orbitals by the angle that minimizes the bond's entanglement functional, THEN truncate. The
//! orbital basis adapts to the state as the sweep goes; the accumulated rotation `u` is returned
//! with the state, and both live in the ROTATED basis (`h` is refilled to `H(rotate_integrals(ints,
//! u))` at every rotation, so the returned triple is consistent by construction).

use std::time::Instant;

use crate::core::twosite::{ModeRotOpts, apply_mode_rotation, contract, noise_rng, search_angle};
use crate::hamiltonian::matvec::onesite_workspace;
use crate::hamiltonian::{EnvGaugePlan, SparseHamiltonian, TermScreen, rayleigh_quotient};
use crate::integrals::Integrals;
use crate::rotation::{
    compose_rotations, givens, identity_matrix, orthogonality_defect, rotate_integrals,
};
use crate::scalar::Scalar;
use crate::solvers::als::{INNER_TOL_CAP, boundary_idsketch};
use crate::solvers::envstore::EnvStore;
use crate::solvers::interface::clamp_hi_first;
use crate::solvers::mals::{
    MalsOptions, SweepTrunc, build_right_store, local_ground_2s, split_two_site,
};
use crate::vector::TtVector;

/// Rotation policy of a [`mals_moderot`] run — shared by the SU(2) twin
/// [`crate::su2::solvers::moderot::mals_moderot`] (scalar-only), as are [`ModeRotSweep`]
/// and [`ModeRotHistory`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeRotSweepOpts {
    /// The per-bond angle search.
    pub rot: ModeRotOpts,
    /// First outer iteration that rotates (earlier ones are plain MALS iterations). Default `1`:
    /// one plain iteration gives the rotation a meaningful state to disentangle.
    pub first_rot_iter: usize,
}

impl Default for ModeRotSweepOpts {
    fn default() -> Self {
        ModeRotSweepOpts {
            rot: ModeRotOpts::default(),
            first_rot_iter: 1,
        }
    }
}

/// One outer iteration (forward + back sweep).
#[derive(Clone, Copy, Debug)]
pub struct ModeRotSweep {
    pub energy: f64,
    pub discarded_weight: f64,
    /// The largest kept bond dimension of the iteration: pooled states on the abelian layer,
    /// REDUCED multiplets on SU(2) (never comparable across layers).
    pub max_bond_dim: usize,
    /// The iteration's cap (`MalsOptions::bond_dims`), in the same unit as `max_bond_dim`.
    pub bond_dim: usize,
    pub noise: f64,
    /// Bonds at which a rotation was taken (`θ ≠ 0`).
    pub n_gates: usize,
    pub max_theta: f64,
    /// `Σ (f(0) − f(θ))` over the gates taken — the functional removed by this iteration, in the
    /// layer's own functional: of the bond's singular values on the abelian layer, of the
    /// multiplet weights `ŵ = √cw·σ` on SU(2) (`crate::su2::twosite::search_angle`).
    pub functional_drop: f64,
    /// Walls of the three rotation-specific stages (recorded, never asserted).
    pub search_s: f64,
    pub refill_s: f64,
    pub gauge_s: f64,
    /// The whole iteration's wall, forward + back sweep, the three stages above included
    /// (recorded, never asserted).
    pub wall_s: f64,
}

#[derive(Clone, Debug)]
pub struct ModeRotHistory {
    /// The last back-sweep local Ritz value (pre-truncation), electronic.
    pub lambda: f64,
    pub sweeps: Vec<ModeRotSweep>,
    /// Angles per outer iteration: the forward sweep's `d − 1` bonds, then the back sweep's (slot
    /// `p < d − 1` is forward bond `p`, slot `p ≥ d − 1` is back bond `2d − 3 − p`). A cross-irrep
    /// bond of a typed space records exactly `0.0`; it is never searched. Empty for an iteration
    /// that does not rotate.
    pub thetas: Vec<Vec<f64>>,
    /// `max |uᵀu − I|` of the returned rotation.
    pub u_orth_dev: f64,
    /// The environment-gauge plan, built once per call — the abelian `EnvGaugePlan`
    /// (`SparseHamiltonian::env_gauge_plan`) or the SU(2) `Su2TouchIndex` (the touched-band
    /// index of the gauge, with the class-A twin pairs and re-contract lists of its mixing mode,
    /// counted whichever mode runs): its wall and resident bytes.
    pub gauge_plan_s: f64,
    pub gauge_plan_bytes: usize,
}

struct Gauger {
    plan: EnvGaugePlan,
    log: Vec<(usize, [f64; 4])>,
    left_applied: Vec<usize>,
    right_applied: Vec<usize>,
    gauge_s: f64,
}

impl Gauger {
    fn new(d: usize, plan: EnvGaugePlan) -> Self {
        Gauger {
            plan,
            log: Vec::new(),
            left_applied: vec![0; d + 1],
            right_applied: vec![0; d + 1],
            gauge_s: 0.0,
        }
    }

    fn mark_left(&mut self, slot: usize) {
        self.left_applied[slot] = self.log.len();
    }

    fn mark_right(&mut self, slot: usize) {
        self.right_applied[slot] = self.log.len();
    }

    fn sync_left<T: Scalar>(
        &mut self,
        h: &SparseHamiltonian<T>,
        x: &TtVector<T>,
        fl: &mut EnvStore<T>,
        b: usize,
    ) {
        let from = self.left_applied[b];
        if from == self.log.len() {
            return;
        }
        if b >= 1 {
            let t0 = Instant::now();
            let mut f = fl.take(b).expect("left environment present");
            for &(i, g) in &self.log[from..] {
                assert!(
                    b <= i,
                    "pending rotation ({i}, {}) is not right of bond {b}",
                    i + 1
                );
                h.rotate_left_env_planned(&self.plan, x.core(b - 1), i, g, &mut f);
            }
            fl.put(b, f);
            self.gauge_s += t0.elapsed().as_secs_f64();
        }
        self.mark_left(b);
    }

    fn sync_right<T: Scalar>(
        &mut self,
        h: &SparseHamiltonian<T>,
        x: &TtVector<T>,
        fr: &mut EnvStore<T>,
        s: usize,
    ) {
        let from = self.right_applied[s];
        if from == self.log.len() {
            return;
        }
        let j = s + 1;
        if j < x.space.d {
            let t0 = Instant::now();
            let mut f = fr.take(s).expect("right environment present");
            for &(i, g) in &self.log[from..] {
                assert!(
                    j >= i + 2,
                    "pending rotation ({i}, {}) is not left of bond {j}",
                    i + 1
                );
                h.rotate_right_env_planned(&self.plan, x.core(j), i, g, &mut f);
            }
            fr.put(s, f);
            self.gauge_s += t0.elapsed().as_secs_f64();
        }
        self.mark_right(s);
    }
}

#[derive(Default)]
struct RotStats {
    n_gates: usize,
    max_theta: f64,
    functional_drop: f64,
    search_s: f64,
    refill_s: f64,
    thetas: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn rotate_pair<T: Scalar>(
    h: &mut SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &TtVector<T>,
    fl: &mut EnvStore<T>,
    fr: &mut EnvStore<T>,
    gauger: &mut Gauger,
    u: &mut Vec<f64>,
    cp: crate::core::twosite::ContractedSparseCores<T>,
    k: usize,
    rot: &ModeRotOpts,
    stats: &mut RotStats,
) -> crate::core::twosite::ContractedSparseCores<T> {
    let d = x.space.d;
    let pg = x.space.pg;
    if pg.irrep_of(k) != pg.irrep_of(k + 1) {
        stats.thetas.push(0.0);
        return cp;
    }
    let t0 = Instant::now();
    let (theta, f0, f1) = search_angle(&cp, rot);
    stats.search_s += t0.elapsed().as_secs_f64();
    stats.thetas.push(theta);
    if theta == 0.0 {
        return cp;
    }
    let cp = apply_mode_rotation(&cp, theta);
    let gk = givens(d, k, theta);
    let g = [
        gk[k * d + k],
        gk[k * d + k + 1],
        gk[(k + 1) * d + k],
        gk[(k + 1) * d + k + 1],
    ];
    *u = compose_rotations(u, &gk, d);
    let dev = orthogonality_defect(u, d);
    assert!(
        dev < 1e-12,
        "mals_moderot: accumulated rotation lost orthogonality (max|uᵀu − I| = {dev:.3e})"
    );
    assert_irrep_blocked(u, pg, d, "mals_moderot");
    let t0 = Instant::now();
    let rotated = rotate_integrals(ints, u);
    h.refill(&rotated.t, &rotated.v, None);
    stats.refill_s += t0.elapsed().as_secs_f64();
    gauger.log.push((k, g));
    gauger.sync_left(h, x, fl, k);
    gauger.sync_right(h, x, fr, k + 1);
    stats.n_gates += 1;
    stats.max_theta = stats.max_theta.max(theta.abs());
    stats.functional_drop += f0 - f1;
    cp
}

pub(crate) fn assert_irrep_blocked(
    u: &[f64],
    pg: crate::pointgroup::PointGroup,
    d: usize,
    who: &str,
) {
    if pg.is_trivial() {
        return;
    }
    for a in 0..d {
        for b in 0..d {
            if pg.irrep_of(a) != pg.irrep_of(b) {
                assert!(
                    u[a * d + b] == 0.0,
                    "{who}: the accumulated rotation mixes irreps: u[{a}][{b}] = {:e}",
                    u[a * d + b]
                );
            }
        }
    }
}

pub(crate) fn assert_same_site_order<T: Scalar>(
    ints: &Integrals<T>,
    pg: crate::pointgroup::PointGroup,
    d: usize,
    who: &str,
) {
    if pg.is_trivial() || ints.orbsym.is_empty() {
        return;
    }
    let ipg = ints.point_group(None).unwrap_or_else(|e| {
        panic!("{who}: the integrals' ORBSYM does not resolve to a point group: {e}")
    });
    for k in 0..d {
        assert!(
            ipg.pg.irrep_of(k) == pg.irrep_of(k),
            "{who}: site {k} carries irrep {:?} in the integrals but {:?} in the space — they are \
             not in the same site order (permute both with the same `perm`)",
            ipg.pg.irrep_of(k),
            pg.irrep_of(k)
        );
    }
}

/// Rank-adaptive two-site ground-state search with local mode rotations (module doc). `h` must
/// be the refillable median operator of `ints` at `u = I`; on return it is the operator of the
/// returned rotation, `x` is the state in that basis, and the energy is the ELECTRONIC one.
///
/// # Panics
///
/// On a non-refillable or non-`SymmetryAllowed` operator, or a skeleton whose same-irrep pairs
/// are not gauge-closed (the bipartite construction) — see the module doc §3. A typed space is
/// accepted: only its same-irrep adjacent pairs rotate. On a typed space whose `ints` carry an
/// ORBSYM, also when a site's irrep differs between `ints` and the space (the two are not in one
/// site order — e.g. only one of them went through `irrep_blocked_order`).
pub fn mals_moderot<T: Scalar>(
    h: &mut SparseHamiltonian<T>,
    ints: &Integrals<T>,
    x: &mut TtVector<T>,
    opts: &MalsOptions,
    sweep: &ModeRotSweepOpts,
) -> (ModeRotHistory, Vec<f64>) {
    let space = x.space;
    let d = space.d;
    assert!(
        h.term_screen() == Some(TermScreen::SymmetryAllowed(space.pg)),
        "mals_moderot: the operator must be a refillable build on TermScreen::SymmetryAllowed \
         (`SparseHamiltonian::new_refillable`)"
    );
    assert_same_site_order(ints, space.pg, d, "mals_moderot");
    let t_plan = Instant::now();
    let plan = h.env_gauge_plan().unwrap_or_else(|e| {
        panic!(
            "mals_moderot: the operator's channel sets are not closed under adjacent relabelling \
             ({e}); build it with Construction::Median"
        )
    });
    let gauge_plan_s = t_plan.elapsed().as_secs_f64();
    let gauge_plan_bytes = plan.bytes();
    assert!(
        !opts.bond_dims.is_empty(),
        "MalsOptions::bond_dims must be non-empty"
    );
    assert!(
        !opts.noise.is_empty(),
        "MalsOptions::noise must be non-empty"
    );
    assert!(d >= 2, "mals_moderot needs at least two sites");

    let mut rng = noise_rng(opts.seed);
    let mut u = identity_matrix(d);
    let mut gauger = Gauger::new(d, plan);
    let mut sweeps = Vec::with_capacity(opts.max_iter);
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
        let inner_tol = clamp_hi_first(0.01 * r, opts.eps, INNER_TOL_CAP);
        let ep = opts.eps / ((d - 1) as f64).sqrt();
        let rotate = it >= sweep.first_rot_iter;
        let lambda_start = lambda;
        let mut stats = RotStats::default();
        let gauge_before = gauger.gauge_s;
        let mut trunc = SweepTrunc::default();

        debug_assert_eq!(x.ortho, Some(0));
        let mut fl = EnvStore::new(space, d, "L");
        fl.put(0, boundary_idsketch(x, 0, 1));
        gauger.mark_left(0);
        fr.drop_slot(0);
        for k in 0..d - 1 {
            gauger.sync_right(h, x, &mut fr, k + 1);
            let cp0 = contract(x.core(k), x.core(k + 1));
            let (theta, cp) = local_ground_2s(h, fl.get(k), fr.get(k + 1), &cp0, inner_tol);
            lambda = theta;
            let cp = if rotate {
                rotate_pair(
                    h,
                    ints,
                    x,
                    &mut fl,
                    &mut fr,
                    &mut gauger,
                    &mut u,
                    cp,
                    k,
                    &sweep.rot,
                    &mut stats,
                )
            } else {
                cp
            };
            let (xk, s, mut xk1, info) = split_two_site(
                h,
                fl.get(k),
                fr.get(k + 1),
                &cp,
                ep,
                bond_dim,
                noise,
                opts.noise_type,
                true,
                &mut rng,
            );
            trunc.absorb(info);
            fr.drop_slot(k + 1);
            xk1.fold_sketch_left_mut(&s);
            x.set_cores(xk, xk1);
            x.ortho = Some(k + 1);
            if k < d - 2 {
                let ws = onesite_workspace(h, x.core(k));
                let next = h.sketching_step_right(x, fl.get(k), &ws);
                fl.put(k + 1, next);
                gauger.mark_left(k + 1);
                fl.release(k);
            }
        }

        if x.ortho != Some(d - 1) {
            x.move_core(d - 1, false);
        }
        let mut fr_new = EnvStore::new(space, d, "R");
        fr_new.put(d - 1, boundary_idsketch(x, d, 1));
        gauger.mark_right(d - 1);
        fl.drop_slot(d - 1);
        for k in (0..d - 1).rev() {
            gauger.sync_left(h, x, &mut fl, k);
            let cp0 = contract(x.core(k), x.core(k + 1));
            let (theta, cp) = local_ground_2s(h, fl.get(k), fr_new.get(k + 1), &cp0, inner_tol);
            lambda = theta;
            let cp = if rotate {
                rotate_pair(
                    h,
                    ints,
                    x,
                    &mut fl,
                    &mut fr_new,
                    &mut gauger,
                    &mut u,
                    cp,
                    k,
                    &sweep.rot,
                    &mut stats,
                )
            } else {
                cp
            };
            let (mut xk, s, xk1, info) = split_two_site(
                h,
                fl.get(k),
                fr_new.get(k + 1),
                &cp,
                ep,
                bond_dim,
                noise,
                opts.noise_type,
                false,
                &mut rng,
            );
            trunc.absorb(info);
            fl.drop_slot(k);
            xk.fold_sketch_right_mut(&s);
            x.set_cores(xk, xk1);
            x.ortho = Some(k);
            if k > 0 {
                let ws = onesite_workspace(h, x.core(k + 1));
                let next = h.sketching_step_left(x, fr_new.get(k + 1), &ws);
                fr_new.put(k, next);
                gauger.mark_right(k);
                fr_new.release(k + 1);
            }
        }
        fr = fr_new;

        r = (lambda_start - lambda).abs() / lambda.abs();
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
            gauge_s: gauger.gauge_s - gauge_before,
            wall_s: t_it.elapsed().as_secs_f64(),
        });
        all_thetas.push(stats.thetas);
        if opts.verbose {
            let s = sweeps.last().unwrap();
            println!(
                "MALS+moderot iter {it}: energy = {lambda}, rel = {r:.3e}, gates = {}, max|θ| = \
                 {:.3e}, Δf = {:.3e}, M = {}, discarded = {:.3e} | search {:.2}s refill {:.2}s \
                 gauge {:.2}s",
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
        if r < opts.convergence_tol() && inner_tol <= opts.eps && noise == 0.0 {
            break;
        }
    }
    let u_orth_dev = orthogonality_defect(&u, d);
    (
        ModeRotHistory {
            lambda,
            sweeps,
            thetas: all_thetas,
            u_orth_dev,
            gauge_plan_s,
            gauge_plan_bytes,
        },
        u,
    )
}
