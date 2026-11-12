// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

pub use crate::solvers::joint::ModeOptOpts;

use crate::integrals::Integrals;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, Su2TermScreen, rayleigh_quotient};
use crate::su2::qn::Su2Space;
use crate::su2::riemannian::normalize_mut;
use crate::su2::solvers::moderot::{ModeRotHistory, mals_moderot};
use crate::su2::vector::Su2TtVector;

/// A consistent `(λ, x, u, h)` in the rotated basis `u`, plus the per-iteration record — the SU(2)
/// twin of [`crate::solvers::joint::ModeOptOut`].
pub struct Su2ModeOptOut {
    /// ELECTRONIC energy `⟨x, h x⟩` of the returned state — its Rayleigh quotient, `x` being
    /// normalized; `history.lambda` is the sweep's last local Ritz value, which precedes that
    /// bond's truncation.
    pub lambda: f64,
    /// The swept state, renormalized after the sweep (its last split keeps `‖x‖ < 1` at a capped
    /// bond dimension).
    pub x: Su2TtVector<f64>,
    /// Row-major `d×d`, the [`crate::rotation::rotate_integrals`] convention.
    pub u: Vec<f64>,
    /// `H(rotate_integrals(ints, u))` — a refillable `SymmetryAllowed(space.pg)` operator.
    pub h: Su2SparseHamiltonian<f64>,
    pub history: ModeRotHistory,
}

/// Run the SU(2) mode-optimizing sweep from `x0` on `ints` (module doc).
///
/// # Panics
///
/// On what [`mals_moderot`] refuses: `ints.norb != space.d`, `d < 2`, an empty `bond_dims` or
/// `noise` schedule, a negative or non-finite noise entry, the spilling environment store, and on
/// a typed `space` integrals in another site order.
pub fn su2_mode_optimize(
    ints: &Integrals<f64>,
    space: Su2Space,
    x0: &Su2TtVector<f64>,
    opts: &ModeOptOpts,
) -> Su2ModeOptOut {
    let mut h = Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::SymmetryAllowed(space.pg),
    );
    let mut x = x0.clone();
    normalize_mut(&mut x);
    let (history, u) = mals_moderot(&mut h, ints, &mut x, &opts.mals, &opts.sweep);
    normalize_mut(&mut x);
    let lambda = rayleigh_quotient(&h, &x);
    Su2ModeOptOut {
        lambda,
        x,
        u,
        h,
        history,
    }
}
