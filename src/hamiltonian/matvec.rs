// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Sketched-apply Hamiltonian kernels: the streamed matvec/environment layer that powers
//! ALS/Davidson without materializing the full `H·x` tensor train, plus the `IdSketch` environments
//! and the streaming one-site kernels.

mod cell;
mod env_gauge;
pub(crate) use env_gauge::EnvGaugePlan;
mod noise;
mod onesite;
mod onesite_env;
mod scheduling;
mod twosite;

pub use cell::ColShard;
pub(crate) use cell::{
    BandKeep, BandMask, RowShard, cell_apply_left, cell_apply_right, cell_apply_right_shard,
};
pub use noise::{noise_density_matrix, noise_density_matrix_streamed};
pub use onesite::{OneSiteWorkspace, h_sketch_mul, onesite_workspace, sketch_h_mul};
pub(crate) use scheduling::BLOCK_SPLIT_TARGET;
#[allow(unused_imports)]
pub(crate) use scheduling::PAR_MIN_TASKS;
pub use scheduling::{
    block_fork, block_split, block_split_events, env_partials_bytes, local_matvec_applies,
    reset_block_split_events, reset_local_matvec_applies, set_block_fork, set_block_split,
    set_env_partials_bytes,
};
pub use twosite::{set_two_site_stream_bytes, two_site_stream_bytes};
