// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::scalar::{C64, Scalar};
use crate::solvers::hess::{
    CurvatureReg, HessPart, PdVerdict, REFUSE_TAU, RpaMetric, SymplecticDavidsonOpts,
    build_symp_precond, omega_davidson_signed,
};
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use std::sync::Arc;

use super::{
    Su2WeingartenCtx, clean_tangent, deflate_radial, hessian_pd_probe, radial_mode,
    reproject_tangent,
};

crate::solvers::hess::impl_symplectic_drivers!(
    ctx: Su2WeingartenCtx,
    tangent: Su2TtTangent,
    vec: Su2TtVector,
    tangent_dim: crate::su2::weingarten::tangent_dim,
    paired: super::davidson_paired::paired_lowest_davidson,
    tag: "su2 ",
    su2_select_rpa_metric,
    symplectic_lowest_signed_davidson,
    symplectic_lowest_signed_davidson_targeted,
    symplectic_lowest_signed_davidson_c64,
);
