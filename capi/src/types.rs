// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Handle-type registration: one tag per exposed type per scalar.

use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::vector::Su2TtVector;
use qntt::{Sparse3Body, SparseHamiltonian, TtVector};

use crate::fcidump::CFcidump;
use crate::handle::{HandleType, ScalarTag};

macro_rules! tag {
    ($ty:ident $(:: $rest:ident)*, $id:literal, $name:literal) => {
        impl<T: ScalarTag> HandleType for $ty $(:: $rest)* <T> {
            const TAG: u64 = 0x716e_7474_0000_0000u64 | T::TAG_BITS | $id;
            const NAME: &'static str = $name;
        }
    };
}

tag!(TtVector, 0x01, "TtVector");
tag!(SparseHamiltonian, 0x02, "SparseHamiltonian");
tag!(Su2TtVector, 0x03, "Su2TtVector");
tag!(Su2SparseHamiltonian, 0x04, "Su2SparseHamiltonian");
tag!(Sparse3Body, 0x05, "Sparse3Body");
tag!(CFcidump, 0x06, "FcidumpData");
