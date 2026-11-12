// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! FCIDUMP / TCDUMP readers.

use crate::pointgroup::{Irrep, PointGroup, PointGroupKind};

mod parser;
mod pointgroup_resolve;
mod readers;

pub use pointgroup_resolve::{
    SYMMETRY_FORBIDDEN_TOL, read_fcidump_with_point_group, screen_point_group_forbidden,
};
pub use readers::{parse_fcidump, read_fcidump, read_tcdump};

pub(crate) use pointgroup_resolve::check_nelec_ms2;

pub(crate) use parser::read_entries;

/// Two-body (`v`) symmetry-completion group for [`read_fcidump`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TwoBodySymmetry {
    /// 8-fold permutation symmetry of the chemists' `(ij|kl)` two-electron integral.
    Eightfold,
    /// Only the pair-swap image `v[k,l,i,j] = v[i,j,k,l]` (transcorrelated dumps: hermiticity is
    /// broken, but the pair-swap relation still holds).
    Pairswap,
    /// No completion — only explicit file entries are kept.
    None,
}

/// Three-body (`w`) symmetry-completion group for [`read_tcdump`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ThreeBodySymmetry {
    /// All 48 images: 6 pair permutations × 8 per-pair transposes.
    Full48,
    /// Only the 6 pair-permutation images (no per-pair transpose).
    Pairperm,
    /// No completion — only explicit file entries are kept.
    None,
}

/// Parse/IO/shape error.
#[derive(Debug, thiserror::Error)]
pub enum FcidumpError {
    #[error("fcidump io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("fcidump argument error: {0}")]
    ArgumentError(String),
    #[error("fcidump dimension mismatch: {0}")]
    DimensionMismatch(String),
}

/// [`read_fcidump`] return — the legacy name for [`crate::integrals::Integrals`].
pub use crate::integrals::Integrals as FcidumpData;

/// The point-group half of an FCIDUMP header, **resolved**: raw `ORBSYM`/`ISYM` validated,
/// interpreted in a [`PointGroupKind`] and converted Molpro → XOR (see
/// [`FcidumpData::point_group`], the single site where that happens).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PointGroupData {
    /// The group the ids were interpreted in: caller-supplied, or the per-order default guess
    /// (see `guessed`).
    pub kind: PointGroupKind,
    /// Packed per-site table, `PointGroup::from_orbsym(kind, orbsym)`.
    pub pg: PointGroup,
    /// Target (right-boundary) irrep as a **0-based XOR index**, from `ISYM`.
    /// [`Irrep::IDENTITY`] when `ISYM` is `0` or absent.
    pub target: Irrep,
    /// `true` when `kind` came from the per-order default guess rather than from the caller.
    pub guessed: bool,
}

impl PointGroupData {
    /// No point-group symmetry: what every `ORBSYM`-free or all-totally-symmetric dump resolves
    /// to. `QnSpace::with_point_group(d, ·, ·, TRIVIAL, IDENTITY)`
    /// is field-for-field `QnSpace::new(d, ·, ·)`, which is what makes the point-group-aware
    /// entry points bit-identical to the plain ones on those files.
    pub const TRIVIAL: PointGroupData = PointGroupData {
        kind: PointGroupKind::C1,
        pg: PointGroup::TRIVIAL,
        target: Irrep::IDENTITY,
        guessed: true,
    };
}
