// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The symmetric-FCIDUMP entry point for the SU(2) layer: read a dump, screen the integrals its
//! own `ORBSYM` forbids, and build the `(N, 2S, ISYM)` [`Su2Space`] it describes.

use std::path::Path;

use crate::fcidump::{
    FcidumpData, FcidumpError, PointGroupData, TwoBodySymmetry, read_fcidump_with_point_group,
    screen_point_group_forbidden,
};
use crate::pointgroup::PointGroupKind;
use crate::scalar::Scalar;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::{Su2Sector, Su2Space, max_ranks};

/// A symmetric dump resolved for the SU(2) layer: the integrals (**post-screening** — a no-op at
/// `|G| = 1`), the resolved point group, and the `(d, N, 2S, pg, target)` space they describe.
pub struct Su2FcidumpLoad<T: Scalar> {
    /// Parsed integrals. `t`/`v` have had every point-group-forbidden cell zeroed; `e0`, `norb`,
    /// `nelec`, `ms2`, `orbsym` and `isym` are exactly what the reader parsed.
    pub data: FcidumpData<T>,
    /// The dump's resolved point group (kind, packed per-site table, `ISYM` target, guess flag).
    pub pg: PointGroupData,
    /// `Su2Space::with_point_group(norb, nelec, 2S, pg.pg, pg.target)`.
    pub space: Su2Space,
}

impl<T: Scalar> std::fmt::Debug for Su2FcidumpLoad<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Su2FcidumpLoad")
            .field("norb", &self.data.norb)
            .field("nelec", &self.data.nelec)
            .field("ms2", &self.data.ms2)
            .field("orbsym", &self.data.orbsym)
            .field("isym", &self.data.isym)
            .field("pg", &self.pg)
            .field("space", &self.space)
            .finish_non_exhaustive()
    }
}

/// Read an FCIDUMP, resolve its point group, build the SU(2) space it describes, and project out
/// (or reject) the integrals that group forbids — the SU(2) twin of the abelian
/// [`crate::integrals::load_fcidump`] under `PointGroupPolicy::Resolve`.
pub fn read_screened<T: Scalar>(
    path: &Path,
    symmetry: TwoBodySymmetry,
    kind: Option<PointGroupKind>,
    two_s: Option<usize>,
) -> Result<Su2FcidumpLoad<T>, FcidumpError> {
    let (mut data, pg) = read_fcidump_with_point_group::<T>(path, symmetry, kind)?;
    let space = su2_space(&data, &pg, two_s)?;
    screen_point_group_forbidden(&mut data.t, &mut data.v, pg.pg)?;
    Ok(Su2FcidumpLoad { data, pg, space })
}

/// The point-group-aware [`Su2Space`] a parsed dump describes, with every
/// [`Su2Space::with_point_group`] `assert!` pre-empted as a [`FcidumpError::ArgumentError`].
pub fn su2_space<T: Scalar>(
    data: &FcidumpData<T>,
    pg: &PointGroupData,
    two_s: Option<usize>,
) -> Result<Su2Space, FcidumpError> {
    let (d, n) = (data.norb, data.nelec);
    let two_s = two_s.unwrap_or(data.ms2.unsigned_abs() as usize);
    if n > 2 * d {
        return Err(FcidumpError::ArgumentError(format!(
            "inconsistent header: NELEC={n} exceeds 2·NORB={}",
            2 * d
        )));
    }
    let cap = n.min(2 * d - n);
    if two_s > cap {
        return Err(FcidumpError::ArgumentError(format!(
            "2S={two_s} exceeds min(N, 2d-N)={cap} (N={n}, d={d})"
        )));
    }
    if !(n + two_s).is_multiple_of(2) {
        return Err(FcidumpError::ArgumentError(format!(
            "N - 2S must be even (N={n}, 2S={two_s})"
        )));
    }
    let ms2 = data.ms2.unsigned_abs() as usize;
    if two_s < ms2 {
        return Err(FcidumpError::ArgumentError(format!(
            "2S={two_s} is below the dump's |MS2|={ms2}: a state of total spin S cannot have \
             projection |Sz| > S"
        )));
    }
    let space = Su2Space::with_point_group(d, n, two_s, pg.pg, pg.target);
    let target_sector = Su2Sector::with_irrep(n, two_s, pg.target);
    if max_ranks(&space)[d][target_sector] == 0 {
        return Err(FcidumpError::ArgumentError(format!(
            "no CSF has N={n} electrons, 2S={two_s} and irrep {} in this space — the requested \
             target block is empty (check ISYM and the point-group kind)",
            pg.target.index()
        )));
    }
    Ok(space)
}

impl<T: Scalar> Su2SparseHamiltonian<T> {
    /// Load a (possibly symmetric) FCIDUMP and compile the SU(2) operator over the space it
    /// describes: [`read_screened`] followed by [`Su2SparseHamiltonian::new`]. At `|G| = 1` this
    /// is bit-identical to reading with [`crate::fcidump::read_fcidump`] and constructing over
    /// `Su2Space::new` — screening is a documented no-op there.
    pub fn from_fcidump_with_point_group(
        path: &Path,
        symmetry: TwoBodySymmetry,
        kind: Option<PointGroupKind>,
        two_s: Option<usize>,
        eps: f64,
    ) -> Result<(Self, Su2FcidumpLoad<T>), FcidumpError> {
        let load = read_screened::<T>(path, symmetry, kind, two_s)?;
        let h = Su2SparseHamiltonian::new(load.space, &load.data.t, &load.data.v, None, eps);
        Ok((h, load))
    }
}

#[cfg(test)]
mod spg6_tests {
    use super::su2_space;
    use crate::coeff::CoeffTensor;
    use crate::fcidump::{FcidumpData, FcidumpError, PointGroupData};
    use crate::pointgroup::{Irrep, PointGroup, PointGroupKind};

    fn zero_fcidump(d: usize, nelec: usize, ms2: i64) -> FcidumpData<f64> {
        FcidumpData {
            t: CoeffTensor::<f64>::zeros(d, 2),
            v: CoeffTensor::<f64>::zeros(d, 4),
            e0: 0.0,
            norb: d,
            nelec,
            ms2,
            orbsym: vec![],
            isym: None,
        }
    }

    #[test]
    fn spg6_l5_su2_space_prechecks_are_errors_not_panics() {
        let d = 4usize;

        let data = zero_fcidump(d, 2 * d + 1, 0);
        assert!(matches!(
            su2_space(&data, &PointGroupData::TRIVIAL, Some(0)),
            Err(FcidumpError::ArgumentError(_))
        ));

        let data = zero_fcidump(d, 4, 0);
        assert!(matches!(
            su2_space(&data, &PointGroupData::TRIVIAL, Some(6)),
            Err(FcidumpError::ArgumentError(_))
        ));

        let data = zero_fcidump(d, 4, 0);
        assert!(matches!(
            su2_space(&data, &PointGroupData::TRIVIAL, Some(1)),
            Err(FcidumpError::ArgumentError(_))
        ));

        let data = zero_fcidump(d, 4, 4);
        assert!(matches!(
            su2_space(&data, &PointGroupData::TRIVIAL, Some(2)),
            Err(FcidumpError::ArgumentError(_))
        ));

        let site_pg =
            PointGroup::from_orbsym(PointGroupKind::C2v, &[1u8]).expect("valid orbsym table");
        assert_eq!(site_pg.irrep_of(0), Irrep::IDENTITY);
        let pg_data = PointGroupData {
            kind: PointGroupKind::C2v,
            pg: site_pg,
            target: Irrep::new(1),
            guessed: false,
        };
        let data = zero_fcidump(1, 1, 1);
        assert!(matches!(
            su2_space(&data, &pg_data, Some(1)),
            Err(FcidumpError::ArgumentError(_))
        ));
    }
}
