// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::scalar::{C64, Scalar};
use crate::solvers::hess::{SymplecticSpectrum, symplectic_from_grams, symplectic_from_grams_c64};
use crate::su2::tangent::Su2TtTangent;

use super::Su2WeingartenCtx;

fn refuse_shifted_basis<T: Scalar>(basis: &[Su2TtTangent<T>], what: &str) {
    for (i, u) in basis.iter().enumerate() {
        u.refuse_if_shifted(&format!("{what} (basis vector {i})"));
    }
}

crate::solvers::hess::impl_hessian_grams!(
    ctx: Su2WeingartenCtx,
    tangent: Su2TtTangent,
    tag: "su2::hess::",
    hessian_grams,
    hessian_grams_owned,
    symplectic_eigenvalues,
    hessian_grams_c64,
    symplectic_eigenvalues_c64,
);
