// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::solvers::hess::{RpaAmplitudes, RpaMetric, RpaReadout};
use crate::su2::tangent::Su2TtTangent;

use super::{Su2WeingartenCtx, deflate_radial, radial_mode, reproject_tangent};

/// [`crate::solvers::hess::rpa_mode_checks`] on the SU(2) layer: the same body
/// (`mode_checks_core`), the same numbers, over the reduced-multiplet tangent space. A context
/// regularized by [`Su2WeingartenCtx::with_curvature`] carries its `D·B·D` into every check, as
/// the solve did.
pub fn rpa_mode_checks(
    ctx: &Su2WeingartenCtx<'_, f64>,
    readout: &RpaReadout,
    modes: &[RpaAmplitudes<Su2TtTangent<f64>>],
    opts: &crate::solvers::hess::ModeCheckOpts,
) -> Vec<crate::solvers::hess::ModeCheck> {
    let _ = opts;
    use crate::solvers::hess::{ModeCheckOps, TanOp, mode_checks_core};
    let (base_l, base_r) = ctx.shared_bases();
    let radial = radial_mode(ctx);
    let hpp = |v: &Su2TtTangent<f64>| ctx.hpp_apply(v);
    let hmm = |v: &Su2TtTangent<f64>| ctx.hmm_apply(v);
    let (w, r, y_sign): (
        &TanOp<'_, Su2TtTangent<f64>>,
        &TanOp<'_, Su2TtTangent<f64>>,
        f64,
    ) = match readout.side() {
        RpaMetric::Rr => (&hpp, &hmm, -1.0),
        RpaMetric::Ii => (&hmm, &hpp, 1.0),
        RpaMetric::Auto => panic!("rpa_mode_checks: the solve's metric was never resolved"),
    };
    let a = |v: &Su2TtTangent<f64>| ctx.a_apply(v);
    let b = |v: &Su2TtTangent<f64>| ctx.b_apply(v);
    let reproject =
        |v: &mut Su2TtTangent<f64>| reproject_tangent(base_l, base_r, radial.as_ref(), v);
    let deflate = |v: &mut Su2TtTangent<f64>| deflate_radial(radial.as_ref(), v);
    let ops = ModeCheckOps {
        w,
        r,
        a: &a,
        b: &b,
        reproject: &reproject,
        deflate: &deflate,
        y_sign,
    };
    mode_checks_core(&readout.modes, modes, &ops)
}
