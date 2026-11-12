// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::hamiltonian::matvec::onesite_workspace;
use qntt::solvers::als::{
    AlsOptions, als, boundary_idsketch, left_to_right_sketching, right_to_left_sketching,
};
use qntt::{
    C64, CoeffTensor, Orbital, QnSpace, Scalar, Sector, SparseHamiltonian, Spin, TtVector, dot,
    rayleigh_quotient, tt_state, tt_zeros, xt_h_y,
};
use qntt_testkit::randomized_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

mod stage1;
mod stage3;

fn complex_hermitian_t() -> [(usize, usize, C64); 2] {
    [
        (1, 3, -C64::from_polar(1.0, 0.9)),
        (3, 1, -C64::from_polar(1.0, -0.9)),
    ]
}
