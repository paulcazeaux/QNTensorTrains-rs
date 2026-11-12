// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

mod item1_persector;
mod item2_3_4;
mod item5_6_9;
mod reconciliation;
mod util;

use qntt::fcidump::{TwoBodySymmetry, read_fcidump};
use qntt::hamiltonian::Construction;
use qntt::hamiltonian::assembly::{BlockStat, assemble_with, bipartite_probe};
use qntt::solvers::als::{AlsOptions, als};
use qntt::solvers::mals::{MalsOptions, mals};
use qntt::{
    CoeffTensor, Family, Orbital, QnSpace, Scalar, SparseHamiltonian, Spin, TtVector,
    rayleigh_quotient,
};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hash::{Hash, Hasher};
use std::path::Path;
