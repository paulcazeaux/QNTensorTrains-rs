// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;

use crate::coeff::CoeffTensor;
use crate::pointgroup::{Irrep, PointGroup, PointGroupError, PointGroupKind};
use crate::qn::QnSpace;
use crate::scalar::Scalar;

use super::readers::read_fcidump;
use super::{FcidumpData, FcidumpError, PointGroupData, TwoBodySymmetry};

const MAX_POINT_GROUP_SITES: usize = 127;

fn resolve_point_group(
    norb: usize,
    orbsym: &[i64],
    isym: Option<i64>,
    kind: Option<PointGroupKind>,
) -> Result<PointGroupData, FcidumpError> {
    if !orbsym.is_empty() && orbsym.len() != norb {
        return Err(FcidumpError::DimensionMismatch(format!(
            "ORBSYM has {} entries but NORB is {norb}",
            orbsym.len()
        )));
    }
    for (k, &id) in orbsym.iter().enumerate() {
        if id < 1 {
            return Err(FcidumpError::ArgumentError(format!(
                "ORBSYM entry {id} at orbital {k} (1-based file position {}) is not a 1-based \
                 Molpro id (must be >= 1)",
                k + 1
            )));
        }
        if id > 8 {
            return Err(FcidumpError::ArgumentError(format!(
                "ORBSYM id {id} at orbital {k} exceeds 8: no abelian point group has more than 8 \
                 irreps"
            )));
        }
    }
    let max_id = orbsym.iter().copied().max().unwrap_or(1) as u8;
    let (kind, guessed) = match kind {
        Some(k) => (k, false),
        None => (
            PointGroupKind::default_for_order(max_id.next_power_of_two().max(1)),
            true,
        ),
    };
    if kind.order() > 1 && norb > MAX_POINT_GROUP_SITES {
        return Err(FcidumpError::ArgumentError(format!(
            "norb={norb} exceeds the {MAX_POINT_GROUP_SITES}-site point-group limit (the \
             automaton's byte-packed orbital key)"
        )));
    }
    if kind.order() == 1 && norb > PointGroup::MAX_SITES {
        return Err(FcidumpError::ArgumentError(format!(
            "norb={norb} exceeds the {}-site point-group table limit",
            PointGroup::MAX_SITES
        )));
    }
    let pg = if kind.order() == 1 {
        for (k, &id) in orbsym.iter().enumerate() {
            if id != 1 {
                return Err(FcidumpError::ArgumentError(
                    PointGroupError::MolproIdOutOfRange {
                        group: kind.name(),
                        id: id as u8,
                        site: Some(k),
                        order: 1,
                    }
                    .to_string(),
                ));
            }
        }
        PointGroup::TRIVIAL
    } else {
        let orbsym_u8: Vec<u8> = orbsym.iter().map(|&id| id as u8).collect();
        PointGroup::from_orbsym(kind, &orbsym_u8)
            .map_err(|e| FcidumpError::ArgumentError(e.to_string()))?
    };
    let order = i64::from(kind.order());
    let target_molpro = match isym {
        None | Some(0) => 1u8,
        Some(v) if (1..=order).contains(&v) => v as u8,
        Some(v) => {
            return Err(FcidumpError::ArgumentError(format!(
                "ISYM={v} is out of range 1..={order} for point group {}",
                kind.name()
            )));
        }
    };
    let target = kind
        .irrep_from_molpro(target_molpro)
        .map_err(|e| FcidumpError::ArgumentError(e.to_string()))?;
    Ok(PointGroupData {
        kind,
        pg,
        target,
        guessed,
    })
}

/// Magnitude above which a point-group-forbidden `t`/`v` entry is treated as a genuine symmetry
/// violation (an error) rather than floating-point noise (silently zeroed). Real symmetry-adapted
/// dumps routinely carry forbidden-entry noise several decades above `f64::EPSILON` — e.g. the
/// vendored `testdata/Be.ezfio.FCIDUMP` line 7's `-2.286208048345990E-015` — while a genuine
/// physical coupling the point group actually forbids would be many orders larger. `1e-8` sits
/// comfortably above the observed noise floor and comfortably below any plausible integral scale.
pub const SYMMETRY_FORBIDDEN_TOL: f64 = 1e-8;

/// Zero out `t`/`v` entries whose orbital indices combine (via `pg`'s XOR fusion) to a
/// non-identity irrep, i.e. entries the point group forbids by symmetry. A no-op at `|G| = 1`
/// (`pg.is_trivial()`), preserving the point-group entry points' bit-identity with the plain ones
/// on the vendored C1 fixtures.
pub fn screen_point_group_forbidden<T: Scalar>(
    t: &mut CoeffTensor<T>,
    v: &mut CoeffTensor<T>,
    pg: PointGroup,
) -> Result<(), FcidumpError> {
    if pg.is_trivial() {
        return Ok(());
    }
    let d = t.d;
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of_sites([i, j]) == Irrep::IDENTITY {
                continue;
            }
            let val = t.get(&[i, j]).abs_s();
            if val == 0.0 {
                continue;
            }
            if val > SYMMETRY_FORBIDDEN_TOL {
                return Err(FcidumpError::ArgumentError(format!(
                    "one-body integral t[{i},{j}] = {val:e} violates the point group's symmetry \
                     (orbitals {i} and {j} combine to a non-identity irrep) and exceeds the \
                     {SYMMETRY_FORBIDDEN_TOL:e} noise-floor tolerance: check ORBSYM/the chosen \
                     point-group kind"
                )));
            }
            t.set(&[i, j], T::zero());
        }
    }
    let dv = v.d;
    for i in 0..dv {
        for j in 0..dv {
            for k in 0..dv {
                for l in 0..dv {
                    if pg.irrep_of_sites([i, j, k, l]) == Irrep::IDENTITY {
                        continue;
                    }
                    let val = v.get(&[i, j, k, l]).abs_s();
                    if val == 0.0 {
                        continue;
                    }
                    if val > SYMMETRY_FORBIDDEN_TOL {
                        return Err(FcidumpError::ArgumentError(format!(
                            "two-body integral v[{i},{j},{k},{l}] = {val:e} violates the point \
                             group's symmetry and exceeds the {SYMMETRY_FORBIDDEN_TOL:e} \
                             noise-floor tolerance: check ORBSYM/the chosen point-group kind"
                        )));
                    }
                    v.set(&[i, j, k, l], T::zero());
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn check_nelec_ms2(nelec: usize, ms2: i64) -> Result<(), FcidumpError> {
    if ms2.unsigned_abs() as usize > nelec {
        return Err(FcidumpError::ArgumentError(format!(
            "inconsistent header: |MS2|={} exceeds NELEC={nelec}",
            ms2.unsigned_abs()
        )));
    }
    if (nelec as i64 + ms2) % 2 != 0 {
        return Err(FcidumpError::ArgumentError(format!(
            "inconsistent header: NELEC={nelec} and MS2={ms2} have mismatched parity"
        )));
    }
    Ok(())
}

impl<T: Scalar> FcidumpData<T> {
    /// Resolve this dump's raw [`orbsym`](Self::orbsym)/[`isym`](Self::isym) into a point group.
    /// `kind = None` guesses the conventional group of the implied order
    /// ([`PointGroupKind::default_for_order`] — the guess affects irrep NAMES only); `Some(k)`
    /// forces `k` and rejects any id outside `1..=k.order()`. Every id and `ISYM` is validated
    /// here, and this is the single site on this path where the Molpro `−1` is dropped.
    pub fn point_group(
        &self,
        kind: Option<PointGroupKind>,
    ) -> Result<PointGroupData, FcidumpError> {
        resolve_point_group(self.norb, &self.orbsym, self.isym, kind)
    }

    /// The point-group-aware [`QnSpace`] this dump describes:
    /// `QnSpace::from_n_two_sz_with_point_group(norb, nelec, ms2, pg, target)`. At `|G| = 1` the
    /// result is field-for-field `QnSpace::from_n_two_sz(norb, nelec, ms2)`.
    pub fn qn_space(&self, kind: Option<PointGroupKind>) -> Result<QnSpace, FcidumpError> {
        let pg = self.point_group(kind)?;
        check_nelec_ms2(self.nelec, self.ms2)?;
        Ok(QnSpace::from_n_two_sz_with_point_group(
            self.norb, self.nelec, self.ms2, pg.pg, pg.target,
        ))
    }
}

/// [`read_fcidump`] plus [`FcidumpData::point_group`] in one call: the orbsym-aware load path.
/// The `FcidumpData` half is **bit-identical** to `read_fcidump(path, symmetry)`'s — resolution
/// reads the header's point-group fields and touches nothing else.
pub fn read_fcidump_with_point_group<T: Scalar>(
    path: &Path,
    symmetry: TwoBodySymmetry,
    kind: Option<PointGroupKind>,
) -> Result<(FcidumpData<T>, PointGroupData), FcidumpError> {
    let data = read_fcidump::<T>(path, symmetry)?;
    let pg = data.point_group(kind)?;
    Ok((data, pg))
}
