// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::scalar::{C64, Scalar};
use crate::solvers::hess::{
    HessPart, ModeOpts, ModeResult, NewtonPolishInfo, NewtonPolishOpts, PROBE_BLOCK_SEED,
    PdVerdict, ProbeDavidson, add_complement_shift, complement_shift, deflate_subspace,
    diag_inf_scale, impl_newton_polish, lowest_hessian_mode, pd_probe_kernel_with,
};
use crate::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use crate::su2::riemannian::tangent_bases;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use std::sync::Arc;

use super::Su2WeingartenCtx;

crate::solvers::hess::impl_hess_probes!(
    ctx: Su2WeingartenCtx,
    tangent: Su2TtTangent,
    vec: Su2TtVector,
    tangent_dim: crate::su2::weingarten::tangent_dim,
    tt_randn: crate::su2::vector::tt_randn,
    tag: "su2 ",
    hessian_pd_probe,
    hessian_pd_probe_witness,
    hessian_pd_probe_witness_with,
    hessian_pd_probe_c64,
    hessian_pd_probe_c64_witness,
);

impl_newton_polish!(
    newton_polish,
    Su2SparseHamiltonian<f64>,
    Su2TtVector<f64>,
    Su2TtTangent<f64>,
    Su2WeingartenCtx,
    rayleigh_quotient,
    tangent_bases,
    crate::su2::riemannian::normalize_mut,
    crate::su2::riemannian::linesearch
);
