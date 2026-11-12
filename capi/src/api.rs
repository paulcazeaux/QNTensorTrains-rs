// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Generic bodies of every layer-parameterized entry point. `exports.rs` wraps each of these in
//! four `extern "C"` shims (abelian/SU(2) × f64/C64); all the logic — argument validation, buffer
//! marshalling, handle lifetime — lives here and is type-checked once.

mod hamiltonian;
mod multistate;
mod ranks;
mod solvers;
mod space;
mod threebody;
mod vector;

pub use hamiltonian::{
    ham_apply, ham_bond_dims, ham_expectation, ham_new, ham_rayleigh, ham_space, ham_xthy,
};
pub use multistate::{excited_states, state_average_mals, state_specific_mals};
pub use solvers::{als, lanczos_solve, mals, riemannian, tdvp};
pub use space::{space_bond_sectors, space_check, space_grid_extent, space_max_ranks};
pub use threebody::{w_new, w_nnz, w_set_coo};
pub use vector::{
    grad_norm, vec_add, vec_block_get, vec_block_set, vec_check, vec_clone, vec_dot,
    vec_dot_bilinear, vec_from_ranks, vec_left_orthonormalize, vec_load, vec_norm, vec_ortho,
    vec_randn, vec_ranks, vec_round, vec_round_mut, vec_round_sum, vec_save, vec_scaled,
    vec_schmidt, vec_sizes, vec_space, vec_zeros,
};
