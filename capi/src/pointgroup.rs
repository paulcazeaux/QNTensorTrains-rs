// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The point-group surface: the layer-independent group tables (`qntt_pg_*`), the **abelian**
//! `_pg` space queries, and the generic bodies behind every abelian `qntt_<scalar>_*_pg` shim in
//! [`crate::exports`]. The SU(2) mirror lives in [`crate::su2_pointgroup`] and shares this
//! module's group tables and [`pg_table`] verbatim — the GROUP is always abelian (C1..D2h); the
//! LAYER need not be.

#[allow(unused_imports)]
use qntt::{PointGroup, PointGroupKind, QnSpace};

mod space;
mod tables;
mod vector;

pub(crate) use space::pg_table;
pub use space::{
    qntt_space_bond_sectors_pg, qntt_space_check_pg, qntt_space_grid_extent_pg,
    qntt_space_max_ranks_pg, qntt_su2_space_bond_sectors_pg, qntt_su2_space_check_pg,
    qntt_su2_space_grid_extent_pg, qntt_su2_space_max_ranks_pg,
};
pub use tables::{
    qntt_pg_default_kind, qntt_pg_irrep_from_molpro, qntt_pg_irrep_name, qntt_pg_kind_from_name,
    qntt_pg_kind_name, qntt_pg_molpro_from_irrep, qntt_pg_orbsym_to_irreps, qntt_pg_order,
};
pub use vector::{
    ham_new_pg, ham_point_group, tt_ones_pg, tt_state_pg, vec_block_get_pg, vec_block_set_pg,
    vec_from_ranks_pg, vec_point_group, vec_randn_pg, vec_ranks_pg, vec_zeros_pg,
};
