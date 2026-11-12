// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The excited-state ladder on the SU(2) spin-adapted layer — the abelian driver body
//! ([`crate::solvers::excited`], which carries the stage list, the penalized functional and the
//! completeness caveat) instantiated at this layer's types.

pub use crate::solvers::excited::{ExcitedInfo, SsStart};

use crate::scalar::Scalar;
use crate::solvers::excited::impl_excited_states;
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use crate::su2::solvers::multistate::MultiTt;
use crate::su2::vector::Su2TtVector;

impl_excited_states!(
    excited_states,
    Su2SparseHamiltonian,
    Su2TtVector,
    MultiTt,
    crate::su2::qn::Su2Space,
    crate::su2::riemannian::RiemannianOpts,
    crate::su2::solvers::mals::mals_mut,
    crate::su2::solvers::multistate::state_average_mals_mut,
    crate::su2::solvers::ssmals::state_specific_mals_with_history,
    crate::su2::riemannian::riemannian_cg_penalized,
    crate::su2::riemannian::penalized_exact_gradnorm,
    crate::su2::riemannian::tangent_bases,
    crate::su2::riemannian::normalize_mut,
    rayleigh_quotient,
    crate::su2::ops::dot,
    crate::su2::weingarten::min_schmidt,
    crate::su2::twosite::contract
);
