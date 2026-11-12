// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::core::{Family as Su2Family, Su2Core};
use qntt::su2::expand::expand;
use qntt::su2::ops::dot as su2_dot;
use qntt::su2::qn::{Su2Sector, Su2Space};
use qntt::su2::vector::{Su2TtVector, tt_csf};

use qntt::{
    Orbital, Scalar, Spin, TtVector, adag_i_a_j, adag_i_adag_k_a_l_a_j, dot as ab_dot, tt_state,
};
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

mod util;

mod arithmetic;
mod construction;
mod core_family;
mod dot_norm_and_spin;
mod expand_oracles;
