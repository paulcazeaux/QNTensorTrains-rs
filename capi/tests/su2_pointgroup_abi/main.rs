// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ptr;

use qntt::su2::core::Family;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient};
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks as su2_max_ranks};
use qntt::su2::vector::{Su2TtVector, tt_csf_with_point_group};
use qntt::{CoeffTensor, Irrep, PointGroup, PointGroupKind};

use qntt_capi::err::{QNTT_INVALID_ARG, QNTT_OK, QNTT_PANIC};
use qntt_capi::exports::*;
use qntt_capi::fcidump::*;
use qntt_capi::qntt_last_error;
use qntt_capi::su2_pointgroup::*;

const PG_C1: i32 = 0;
const PG_C2V: i32 = 5;
const PG_AUTO: i32 = -1;

fn c2v_orbsym(d: usize) -> Vec<u8> {
    (0..d).map(|k| [2u8, 3, 4, 1][k % 4]).collect()
}
fn c2v_pg(d: usize) -> PointGroup {
    PointGroup::from_orbsym(PointGroupKind::C2v, &c2v_orbsym(d)).expect("valid orbsym table")
}

mod csf_and_hamiltonian;
mod fcidump_bondsectors_macrosync;
mod space_and_legacy;
