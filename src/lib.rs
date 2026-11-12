// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! # qntt
//!
//! Quantum-number-preserving block-sparse Tensor Trains for fermionic many-body systems: the
//! abelian `Nup`/`Ndn`-conserving layer here, the SU(2) spin-adapted layer in `su2`.

#![allow(rustdoc::private_intra_doc_links)]

pub(crate) mod blockstore;
pub mod cis;
pub mod coeff;
pub mod core;
pub mod deas;
pub mod elementaryops;
pub mod entropy;
pub mod fcidump;
pub mod hamiltonian;
pub mod integrals;
/// Every environment variable the library reads (measurement instruments): see the module doc.
#[doc(hidden)]
pub mod knobs;
pub mod mem;
pub mod ops;
pub mod orbital;
pub mod orbopt;
pub mod ordering;
pub mod pairgram;
pub mod pairgram_tangent;
pub mod par;
pub mod pointgroup;
pub mod qn;
pub mod rdm;
pub mod rotation;
pub mod round;
pub mod scalar;
pub mod solvers;
pub mod storage_probe;
pub mod su2;
pub mod tt_io;

#[cfg(test)]
pub(crate) mod test_util;
pub mod vector;

pub use crate::coeff::{CoeffTensor, Sparse3Body, ThreeBody};
pub use crate::core::factor::Sketch;
pub use crate::core::{Family, SparseCore};
pub use crate::elementaryops::{
    adag_i_a_j, adag_i_adag_k_a_l_a_j, adag_i_adag_k_adag_m_a_n_a_l_a_j,
};
pub use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient, xt_h_y, xt_h_y_block};
pub use crate::integrals::{
    FixtureMeta, Integrals, IntegralsLoad, IntegralsOptions, PointGroupPolicy, PropertyOperator,
    ThreeBodyOptions,
};
pub use crate::mem::Footprint;
pub use crate::ops::{dot, dot_block, hadamard, irrep_weights};
pub use crate::orbital::{Orbital, Spin, fermionic_sign};
pub use crate::pointgroup::{Irrep, PointGroup, PointGroupError, PointGroupKind};
pub use crate::qn::{
    PathCountGrid, PerSector, Pooled, QnSpace, RankGrid, Sector, SectorGrid, max_ranks,
    max_ranks_open, path_counts,
};
pub use crate::rdm::transition_1rdm_bruteforce;
pub use crate::rotation::{
    PairMask, assert_antisym, e_pq, expm_antisym, expm_kappa_blocked, freeze_core, g_pq,
    kappa_apply, kappa_operator, orbital_gradient, orbital_pairs, rotate_integrals,
};
pub use crate::round::{RoundMode, dot_orth};
pub use crate::scalar::{C64, Scalar};
pub use crate::vector::{TtVector, tt_ones, tt_randn, tt_state, tt_state_in, tt_zeros};
