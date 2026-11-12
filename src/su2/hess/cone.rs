// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::factor::dense_self_adjoint_smallest;
use crate::qn::Pooled;
use crate::round::RoundMode;
use crate::solvers::hess::{
    ConeGuards, ConeOpts, ConeRefusal, ConeResponse, ConeStageStats, ISO_TOL, MatfreeOpts,
    MatfreeStats, SolveMode, drift_voids_record, mixed_lanczos, symplectic_from_grams,
};
use crate::su2::hamiltonian::{Su2SparseHamiltonian, xt_h_y};
use crate::su2::ops::dot;
use crate::su2::qn::Su2RankGrid;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;
use crate::su2::weingarten::tangent_basis;
use rayon::prelude::*;

use super::{Su2WeingartenCtx, hessian_grams, normal_residual, weingarten_operator};

crate::solvers::hess::impl_cone!(
    ham: Su2SparseHamiltonian<f64>,
    vec: Su2TtVector<f64>,
    tangent: Su2TtTangent<f64>,
    ctx: Su2WeingartenCtx,
    grid: Su2RankGrid,
    weingarten_operator: weingarten_operator,
    normal_residual: normal_residual,
    tangent_basis: tangent_basis,
    hessian_grams: hessian_grams,
    xt_h_y: xt_h_y,
    dot: dot,
    tt_randn: crate::su2::vector::tt_randn,
    tag: "su2 ",
    gram_note: " (including the centerweighted-vs-ambient metric chain)",

    cone_response,
    try_cone_response,
    cone_response_matfree,
    cone_response_matfree_with_stats,
    cone_response_with_stats,
);

/// [`cone_response_matfree_with_stats`] ALSO returning the accepted normals' cached mixed rows —
/// the study surface for assembling an augmented Williamson block against an EXTERNAL Gram cache.
///
/// # Panics
///
/// As [`cone_response_matfree`].
pub fn cone_response_matfree_with_rows(
    h: &Su2SparseHamiltonian<f64>,
    x: &Su2TtVector<f64>,
    opts: &ConeOpts,
    mf: &MatfreeOpts,
) -> (ConeResponse, MatfreeStats, Su2ConeMixedRows) {
    match cone_solve_with(h, x, opts, SolveMode::Matfree(mf)) {
        Ok(mut out) => {
            let rows = Su2ConeMixedRows {
                rows: out.gram.hrows,
                lam: out.lam,
                normals: out.gram.refs.split_off(out.gram.n_ref),
            };
            (out.response, out.matfree_stats, rows)
        }
        Err(refusal) => panic!("{}", refusal.message()),
    }
}

/// What [`cone_response_matfree_with_rows`] hands back beyond the response and stats: everything
/// an EXTERNAL augmented-Williamson assembly needs that the response itself does not carry.
pub struct Su2ConeMixedRows {
    /// Row `m` is `[⟨x, w_m⟩, ⟨ι(ξ_1), w_m⟩ … ⟨ι(ξ_{n_t}), w_m⟩, ⟨n_1, w_m⟩ … ⟨n_m, w_m⟩]` with
    /// `w_m = round(H·n_m)`, length `1 + n_t + m + 1`.
    pub rows: Vec<Vec<f64>>,
    /// The accepted normal directions, index-aligned with [`Self::rows`].
    pub normals: Vec<Su2TtVector<f64>>,
    /// The reference's Rayleigh quotient — the `λ` of the shift above. ELECTRONIC (add `ints.e0`
    /// for a total energy).
    pub lam: f64,
}
