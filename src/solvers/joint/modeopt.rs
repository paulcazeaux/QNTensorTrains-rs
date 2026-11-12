// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::hamiltonian::{Construction, SparseHamiltonian, TermScreen, rayleigh_quotient};
use crate::integrals::Integrals;
use crate::qn::QnSpace;
use crate::solvers::mals::MalsOptions;
use crate::solvers::moderot::{ModeRotHistory, ModeRotSweepOpts, mals_moderot};
use crate::solvers::riemannian::normalize_mut;
use crate::vector::TtVector;

/// Options of [`mode_optimize`] — shared by the SU(2) twin
/// [`crate::su2::joint::su2_mode_optimize`] (scalar-only), where `mals.bond_dims` count
/// REDUCED multiplets.
#[derive(Clone, Debug, Default)]
pub struct ModeOptOpts {
    /// The sweep schedule (bond dimensions, noise, tolerances) — a cheap rank is the point: the
    /// basis found at a small bond dimension is the warm start for the expensive solve.
    pub mals: MalsOptions,
    /// The rotation policy.
    pub sweep: ModeRotSweepOpts,
}

/// A consistent `(λ, x, u, h)` in the rotated basis `u`, plus the per-iteration record.
pub struct ModeOptOut {
    /// ELECTRONIC energy `⟨x, h x⟩` of the returned state — its Rayleigh quotient, `x` being
    /// normalized; `history.lambda` is the sweep's last local Ritz value, which precedes that
    /// bond's truncation.
    pub lambda: f64,
    /// The swept state, renormalized after the sweep (its last split keeps `‖x‖ < 1` at a capped
    /// bond dimension).
    pub x: TtVector<f64>,
    /// Row-major `d×d`, the [`crate::rotation::rotate_integrals`] convention.
    pub u: Vec<f64>,
    /// `H(rotate_integrals(ints, u))` — a refillable MEDIAN operator.
    pub h: SparseHamiltonian<f64>,
    pub history: ModeRotHistory,
}

/// Run the mode-optimizing sweep from `x0` on `ints` (module doc).
pub fn mode_optimize(
    ints: &Integrals<f64>,
    space: QnSpace,
    x0: &TtVector<f64>,
    opts: &ModeOptOpts,
) -> ModeOptOut {
    let mut h = SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        TermScreen::SymmetryAllowed(space.pg),
        Construction::Median,
    );
    let mut x = x0.clone();
    normalize_mut(&mut x);
    let (history, u) = mals_moderot(&mut h, ints, &mut x, &opts.mals, &opts.sweep);
    normalize_mut(&mut x);
    let lambda = rayleigh_quotient(&h, &x);
    ModeOptOut {
        lambda,
        x,
        u,
        h,
        history,
    }
}
