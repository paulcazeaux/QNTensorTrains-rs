// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Consumption hooks for the SU(2) `Su2SparseHamiltonian`: H-extended row/col rank + range
//! tables, the materialized `H·x` (`apply`, THE cross-symmetry gate), the sketched-apply kernels,
//! and the ONE θ²-weight-bearing kernel in the file.

mod env_gauge;
mod noise;
mod ranks;
mod sketch;
mod sketching;
#[cfg(test)]
mod su2_noise_dm_tests;

pub(crate) use env_gauge::{Su2GaugeCache, Su2GaugeMode, gauge_after_pair_rotation};
pub use noise::{noise_density_matrix, noise_density_matrix_streamed};
pub use sketch::{h_sketch_mul, h_x_adjoint_r, sketch_h_mul, sketch_h_mul_keep};
pub use sketching::{
    Su2OneSiteWorkspace, Su2TwoSiteWorkspace, factor_qc_into, su2_onesite_workspace,
    su2_twosite_workspace,
};

pub(crate) const PAR_MIN_TASKS: usize = 8;
