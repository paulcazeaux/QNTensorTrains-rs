// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Abelian point-group symmetry: irreps as XOR indices, and the per-site orbsym table.

use std::str::FromStr;

const _: () = assert!(size_of::<PointGroup>() == 64);
const _: () = assert!(size_of::<Irrep>() == 1);

/// One irreducible representation of an abelian point group, as a 0-based **XOR index**: the
/// group is `Z2^k`, irreps are bit-vectors, and the direct product is bitwise XOR.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Irrep(pub u8);

impl Irrep {
    /// The totally symmetric irrep (XOR index 0) — the identity of [`Irrep::fuse`], and
    /// [`Irrep`]'s `Default`.
    pub const IDENTITY: Irrep = Irrep(0);

    /// Convenience constructor. Group-agnostic: no validation against any group's order.
    #[inline]
    pub const fn new(index: u8) -> Self {
        Irrep(index)
    }

    /// The raw 0-based XOR index.
    #[inline]
    pub const fn index(self) -> u8 {
        self.0
    }

    /// The character `χ_self(h) = (−1)^popcount(self & h)`, as `±1.0`.
    pub const fn character(self, h: Irrep) -> f64 {
        if (self.0 & h.0).count_ones().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        }
    }

    /// Direct product `a ⊗ b` = bitwise XOR (the group is `Z2^k`, all irreps 1-dimensional).
    /// Closed on `0..order` by construction, so no masking is performed or needed — and, being
    /// group-agnostic, this never validates: `Irrep(200).fuse(Irrep(3)) == Irrep(203)`.
    #[inline]
    pub const fn fuse(self, other: Irrep) -> Irrep {
        Irrep(self.0 ^ other.0)
    }

    /// Every element of `Z2^k` is its own inverse: returns `self`. Present so callers read
    /// symmetrically with quantum numbers whose inverse is not the identity map.
    #[inline]
    pub const fn inverse(self) -> Irrep {
        self
    }
}

/// Invalid point-group input (constructor validation and the FCIDUMP-boundary conversions).
#[derive(Clone, PartialEq, Eq, Debug, thiserror::Error)]
pub enum PointGroupError {
    /// A 1-based Molpro ORBSYM id outside `1..=order`. `site` is `Some(k)` when the id came
    /// from an ORBSYM slice, `None` for the scalar conversion.
    #[error(
        "point group {group}: Molpro ORBSYM id {id} (site {site:?}) is out of range 1..={order}"
    )]
    MolproIdOutOfRange {
        group: &'static str,
        id: u8,
        site: Option<usize>,
        order: u8,
    },
    /// A 0-based XOR index outside `0..order`. `site` is `Some(k)` when the irrep came from a
    /// per-site table, `None` for the scalar conversion.
    #[error("irrep index {irrep} (site {site:?}) is out of range 0..{order}")]
    IrrepOutOfRange {
        irrep: u8,
        site: Option<usize>,
        order: u8,
    },
    /// More sites than a bit-plane holds.
    #[error("too many sites for a point-group table: d={d} exceeds the {max}-site limit")]
    TooManySites { d: usize, max: usize },
    /// More irrep bits than the largest abelian point group has.
    #[error("nbits={nbits} exceeds {max} (the largest abelian point group is D2h, order 8)")]
    TooManyBits { nbits: u8, max: u8 },
    /// [`PointGroupKind::from_str`] got a name outside the eight supported groups.
    #[error(
        "unknown point-group name {name:?} (expected one of C1, Ci, C2, Cs, D2, C2v, C2h, D2h)"
    )]
    UnknownGroup { name: String },
}

/// The eight abelian point groups an FCIDUMP can carry, in the reference fixture's block order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum PointGroupKind {
    /// No symmetry; order 1.
    #[default]
    C1,
    /// Inversion only; order 2.
    Ci,
    /// One `C2` axis; order 2.
    C2,
    /// One mirror plane; order 2.
    Cs,
    /// Three `C2` axes; order 4.
    D2,
    /// `C2` + two mirror planes; order 4.
    C2v,
    /// `C2` + inversion; order 4.
    C2h,
    /// The full abelian group; order 8.
    D2h,
}

const MOLPRO_TO_XOR: [&[u8]; 8] = [
    &[0],
    &[0, 1],
    &[0, 1],
    &[0, 1],
    &[0, 3, 2, 1],
    &[0, 2, 3, 1],
    &[0, 2, 3, 1],
    &[0, 7, 6, 1, 5, 2, 3, 4],
];

const XOR_TO_MOLPRO: [&[u8]; 8] = [
    &[1],
    &[1, 2],
    &[1, 2],
    &[1, 2],
    &[1, 4, 3, 2],
    &[1, 4, 2, 3],
    &[1, 4, 2, 3],
    &[1, 4, 6, 7, 8, 5, 3, 2],
];

const NAMES_MOLPRO: [&[&str]; 8] = [
    &["A"],
    &["Ag", "Au"],
    &["A", "B"],
    &["A'", "A\""],
    &["A", "B3", "B2", "B1"],
    &["A1", "B1", "B2", "A2"],
    &["Ag", "Au", "Bu", "Bg"],
    &["Ag", "B3u", "B2u", "B1g", "B1u", "B2g", "B3g", "Au"],
];

const NAMES_XOR: [&[&str]; 8] = [
    &["A"],
    &["Ag", "Au"],
    &["A", "B"],
    &["A'", "A\""],
    &["A", "B1", "B2", "B3"],
    &["A1", "A2", "B1", "B2"],
    &["Ag", "Bg", "Au", "Bu"],
    &["Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"],
];

impl PointGroupKind {
    /// All eight groups, in the reference-fixture order (`testdata/pointgroups/tables.txt`).
    pub const ALL: [PointGroupKind; 8] = [
        PointGroupKind::C1,
        PointGroupKind::Ci,
        PointGroupKind::C2,
        PointGroupKind::Cs,
        PointGroupKind::D2,
        PointGroupKind::C2v,
        PointGroupKind::C2h,
        PointGroupKind::D2h,
    ];

    /// Schoenflies name, as it appears in an FCIDUMP header.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            PointGroupKind::C1 => "C1",
            PointGroupKind::Ci => "Ci",
            PointGroupKind::C2 => "C2",
            PointGroupKind::Cs => "Cs",
            PointGroupKind::D2 => "D2",
            PointGroupKind::C2v => "C2v",
            PointGroupKind::C2h => "C2h",
            PointGroupKind::D2h => "D2h",
        }
    }

    /// Number of irreps, `1 << nbits()`.
    #[inline]
    pub const fn order(self) -> u8 {
        1u8 << self.nbits()
    }

    /// Number of `Z2` generators (mirror planes / axes), `0..=3`.
    #[inline]
    pub const fn nbits(self) -> u8 {
        match self {
            PointGroupKind::C1 => 0,
            PointGroupKind::Ci | PointGroupKind::C2 | PointGroupKind::Cs => 1,
            PointGroupKind::D2 | PointGroupKind::C2v | PointGroupKind::C2h => 2,
            PointGroupKind::D2h => 3,
        }
    }

    /// The conventional group of a given order — what an FCIDUMP reader guesses when the header
    /// carries ORBSYM ids but names no group (`order = max(1, max_id.next_power_of_two())`).
    ///
    /// # Panics
    /// Unless `order ∈ {1, 2, 4, 8}` (crate style, cf. [`PointGroup::new`]). Data-driven callers
    /// validate first: the FCIDUMP resolver rejects `max(ORBSYM) > 8` before calling this.
    pub const fn default_for_order(order: u8) -> PointGroupKind {
        match order {
            1 => PointGroupKind::C1,
            2 => PointGroupKind::Cs,
            4 => PointGroupKind::C2v,
            8 => PointGroupKind::D2h,
            _ => panic!("point-group order must be 1, 2, 4 or 8"),
        }
    }

    /// Parse a group name, ASCII-case-insensitively (`"d2h"`, `"D2H"`, `"D2h"` all accepted —
    /// FCIDUMP producers are inconsistent). `None` if unrecognized; see also the [`FromStr`]
    /// impl, which reports [`PointGroupError::UnknownGroup`] instead.
    pub fn from_name(s: &str) -> Option<Self> {
        PointGroupKind::ALL
            .into_iter()
            .find(|k| k.name().eq_ignore_ascii_case(s))
    }

    /// Irrep names indexed by **XOR index** (`kind.irrep_names_xor()[ir.index() as usize]`).
    #[inline]
    pub const fn irrep_names_xor(self) -> &'static [&'static str] {
        NAMES_XOR[self as usize]
    }

    /// Irrep names indexed by **Molpro id − 1**, i.e. ORBSYM order.
    #[inline]
    pub const fn irrep_names_molpro(self) -> &'static [&'static str] {
        NAMES_MOLPRO[self as usize]
    }

    /// Name of an irrep given its XOR index.
    pub fn irrep_name(self, ir: Irrep) -> Result<&'static str, PointGroupError> {
        let order = self.order();
        if ir.0 >= order {
            return Err(PointGroupError::IrrepOutOfRange {
                irrep: ir.0,
                site: None,
                order,
            });
        }
        Ok(NAMES_XOR[self as usize][ir.0 as usize])
    }

    /// **FCIDUMP boundary, 1-based in.** Molpro ORBSYM id `id ∈ 1..=order()` → XOR index.
    /// This is the only place the `−1` of the Molpro convention is dropped.
    pub fn irrep_from_molpro(self, id: u8) -> Result<Irrep, PointGroupError> {
        self.irrep_from_molpro_at(id, None)
    }

    fn irrep_from_molpro_at(self, id: u8, site: Option<usize>) -> Result<Irrep, PointGroupError> {
        let order = self.order();
        if id == 0 || id > order {
            return Err(PointGroupError::MolproIdOutOfRange {
                group: self.name(),
                id,
                site,
                order,
            });
        }
        Ok(Irrep(MOLPRO_TO_XOR[self as usize][id as usize - 1]))
    }

    /// **FCIDUMP boundary, 1-based out.** XOR index → Molpro ORBSYM id in `1..=order()`.
    pub fn molpro_from_irrep(self, ir: Irrep) -> Result<u8, PointGroupError> {
        let order = self.order();
        if ir.0 >= order {
            return Err(PointGroupError::IrrepOutOfRange {
                irrep: ir.0,
                site: None,
                order,
            });
        }
        Ok(XOR_TO_MOLPRO[self as usize][ir.0 as usize])
    }
}

impl FromStr for PointGroupKind {
    type Err = PointGroupError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        PointGroupKind::from_name(s).ok_or_else(|| PointGroupError::UnknownGroup {
            name: s.to_string(),
        })
    }
}

/// The per-site irrep table of a molecule, packed as three `Z2` bit-planes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PointGroup {
    nbits: u8,
    planes: [u128; 3],
}

impl PointGroup {
    /// Maximum representable site count (the `u128` bit-plane width). See the module docs: the
    /// automaton's byte-packed orbital key caps at 127 sites, so this is never the binding
    /// constraint downstream.
    pub const MAX_SITES: usize = 128;
    /// Maximum irrep bit count (D2h, order 8).
    pub const MAX_BITS: u8 = 3;

    /// Every site totally symmetric: the `nbits = 0`, order-1 group (C1). Also `Default`.
    pub const TRIVIAL: PointGroup = PointGroup {
        nbits: 0,
        planes: [0; 3],
    };

    /// Pack a 0-based **XOR-index** per-site table. Panics on invalid input (crate style,
    /// cf. [`crate::qn::QnSpace::new`]); use [`try_new`](Self::try_new) on data-driven paths.
    pub fn new(nbits: u8, irreps: &[Irrep]) -> Self {
        Self::try_new(nbits, irreps).unwrap_or_else(|e| panic!("invalid point-group table: {e}"))
    }

    /// Checked twin of [`new`](Self::new).
    pub fn try_new(nbits: u8, irreps: &[Irrep]) -> Result<Self, PointGroupError> {
        if nbits > Self::MAX_BITS {
            return Err(PointGroupError::TooManyBits {
                nbits,
                max: Self::MAX_BITS,
            });
        }
        if irreps.len() > Self::MAX_SITES {
            return Err(PointGroupError::TooManySites {
                d: irreps.len(),
                max: Self::MAX_SITES,
            });
        }
        let order = 1u8 << nbits;
        let mut planes = [0u128; 3];
        for (k, ir) in irreps.iter().enumerate() {
            if ir.0 >= order {
                return Err(PointGroupError::IrrepOutOfRange {
                    irrep: ir.0,
                    site: Some(k),
                    order,
                });
            }
            for (b, plane) in planes.iter_mut().enumerate().take(nbits as usize) {
                *plane |= (((ir.0 >> b) & 1) as u128) << k;
            }
        }
        Ok(PointGroup { nbits, planes })
    }

    /// **FCIDUMP boundary, 1-based in.** Build from a Molpro ORBSYM slice: `orbsym[k]` is the
    /// 1-based ORBSYM id of site `k`, in `1..=kind.order()`.
    pub fn from_orbsym(kind: PointGroupKind, orbsym: &[u8]) -> Result<Self, PointGroupError> {
        if orbsym.len() > Self::MAX_SITES {
            return Err(PointGroupError::TooManySites {
                d: orbsym.len(),
                max: Self::MAX_SITES,
            });
        }
        let nbits = kind.nbits();
        let mut planes = [0u128; 3];
        for (k, &id) in orbsym.iter().enumerate() {
            let ir = kind.irrep_from_molpro_at(id, Some(k))?;
            for (b, plane) in planes.iter_mut().enumerate().take(nbits as usize) {
                *plane |= (((ir.0 >> b) & 1) as u128) << k;
            }
        }
        Ok(PointGroup { nbits, planes })
    }

    /// Number of `Z2` generators of the group this table was built for.
    #[inline]
    pub const fn nbits(self) -> u8 {
        self.nbits
    }

    /// Group order, `1 << nbits()`.
    #[inline]
    pub const fn order(self) -> u8 {
        1u8 << self.nbits
    }

    /// The raw bit-planes (`planes[b]` bit `k` = bit `b` of `irrep_of(k)`).
    #[inline]
    pub const fn planes(self) -> [u128; 3] {
        self.planes
    }

    /// Whether this is the order-1 (no-symmetry) table.
    #[inline]
    pub const fn is_trivial(self) -> bool {
        self.nbits == 0
    }

    /// Irrep of site `k` (0-based). Sites at or beyond the table this was built from —
    /// including `k >= 128` — return [`Irrep::IDENTITY`]: the packing is zero there.
    #[inline]
    pub fn irrep_of(self, k: usize) -> Irrep {
        if k >= Self::MAX_SITES {
            return Irrep::IDENTITY;
        }
        let b0 = ((self.planes[0] >> k) & 1) as u8;
        let b1 = ((self.planes[1] >> k) & 1) as u8;
        let b2 = ((self.planes[2] >> k) & 1) as u8;
        Irrep(b0 | (b1 << 1) | (b2 << 2))
    }

    /// Product of the irreps of a set of sites, e.g. a determinant's occupation pattern.
    pub fn irrep_of_sites(self, sites: impl IntoIterator<Item = usize>) -> Irrep {
        sites
            .into_iter()
            .fold(Irrep::IDENTITY, |acc, k| acc.fuse(self.irrep_of(k)))
    }
}

impl Default for PointGroup {
    fn default() -> Self {
        Self::TRIVIAL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::collections::{BTreeSet, HashSet};

    #[test]
    fn fuse_is_an_abelian_group() {
        for kind in PointGroupKind::ALL {
            let order = kind.order();
            for a in 0..order {
                let a = Irrep::new(a);
                assert_eq!(
                    a.fuse(Irrep::IDENTITY),
                    a,
                    "{}: identity a={a:?}",
                    kind.name()
                );
                assert_eq!(
                    a.fuse(a),
                    Irrep::IDENTITY,
                    "{}: self-inverse a={a:?}",
                    kind.name()
                );
                assert_eq!(a.inverse(), a, "{}: inverse == self, a={a:?}", kind.name());
                assert_eq!(
                    a.fuse(a.inverse()),
                    Irrep::IDENTITY,
                    "{}: a.fuse(a.inverse()) a={a:?}",
                    kind.name()
                );
                for b in 0..order {
                    let b = Irrep::new(b);
                    assert!(
                        a.fuse(b).index() < order,
                        "{}: closure a={a:?} b={b:?}",
                        kind.name()
                    );
                    assert_eq!(
                        a.fuse(b),
                        b.fuse(a),
                        "{}: commutativity a={a:?} b={b:?}",
                        kind.name()
                    );
                    for c in 0..order {
                        let c = Irrep::new(c);
                        assert_eq!(
                            a.fuse(b).fuse(c),
                            a.fuse(b.fuse(c)),
                            "{}: associativity a={a:?} b={b:?} c={c:?}",
                            kind.name()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn product_table_is_a_latin_square() {
        for kind in PointGroupKind::ALL {
            let order = kind.order();
            let full: BTreeSet<u8> = (0..order).collect();
            for a in 0..order {
                let a = Irrep::new(a);
                let row: BTreeSet<u8> = (0..order).map(|b| a.fuse(Irrep::new(b)).index()).collect();
                assert_eq!(
                    row,
                    full,
                    "{}: fuse row for a={a:?} is not a permutation of 0..{order}",
                    kind.name()
                );
            }
        }
    }

    #[test]
    fn molpro_round_trip_total() {
        for kind in PointGroupKind::ALL {
            let order = kind.order();

            let mut molpro_images = BTreeSet::new();
            for m in 1..=order {
                let ir = kind.irrep_from_molpro(m).unwrap();
                let back = kind.molpro_from_irrep(ir).unwrap();
                assert_eq!(
                    back,
                    m,
                    "{}: molpro_from_irrep(irrep_from_molpro({m})) round trip",
                    kind.name()
                );
                molpro_images.insert(ir.index());
            }
            assert_eq!(
                molpro_images.len(),
                order as usize,
                "{}: irrep_from_molpro is not injective on 1..=order",
                kind.name()
            );

            let mut xor_images = BTreeSet::new();
            for x in 0..order {
                let ir = Irrep::new(x);
                let m = kind.molpro_from_irrep(ir).unwrap();
                let back = kind.irrep_from_molpro(m).unwrap();
                assert_eq!(
                    back,
                    ir,
                    "{}: irrep_from_molpro(molpro_from_irrep({ir:?})) round trip",
                    kind.name()
                );
                xor_images.insert(m);
            }
            assert_eq!(
                xor_images.len(),
                order as usize,
                "{}: molpro_from_irrep is not injective on 0..order",
                kind.name()
            );
        }
    }

    #[test]
    fn irrep_of_round_trips() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x506f_696e_7447_5030);
        for kind in PointGroupKind::ALL {
            let order = kind.order();
            for &d in &[0usize, 1, 2, 7, 63, 64, 65, 127, 128] {
                let irreps: Vec<Irrep> = (0..d)
                    .map(|_| Irrep::new(rng.random_range(0..order)))
                    .collect();
                let pg = PointGroup::try_new(kind.nbits(), &irreps)
                    .unwrap_or_else(|e| panic!("{}: d={d}: {e}", kind.name()));

                for (k, &ir) in irreps.iter().enumerate() {
                    assert_eq!(pg.irrep_of(k), ir, "{}: d={d} k={k}", kind.name());
                }

                for &k in &[d, d + 1, 127usize, 128usize, 200usize, usize::MAX] {
                    if k >= d {
                        assert_eq!(
                            pg.irrep_of(k),
                            Irrep::IDENTITY,
                            "{}: d={d} k={k} beyond-table site",
                            kind.name()
                        );
                    }
                }

                assert_eq!(pg.order(), order, "{}: d={d} order", kind.name());
                assert_eq!(pg.nbits(), kind.nbits(), "{}: d={d} nbits", kind.name());
            }
        }
    }

    #[test]
    fn from_orbsym_matches_manual_conversion() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x506f_696e_7447_5031);
        for kind in PointGroupKind::ALL {
            let order = kind.order();
            for &d in &[0usize, 1, 2, 7, 63, 64, 65, 127, 128] {
                let ids: Vec<u8> = (0..d).map(|_| rng.random_range(1..=order)).collect();
                let pg = PointGroup::from_orbsym(kind, &ids)
                    .unwrap_or_else(|e| panic!("{}: d={d}: {e}", kind.name()));

                for (k, &id) in ids.iter().enumerate() {
                    let expected = kind.irrep_from_molpro(id).unwrap();
                    assert_eq!(pg.irrep_of(k), expected, "{}: d={d} k={k}", kind.name());
                    assert_eq!(
                        kind.molpro_from_irrep(pg.irrep_of(k)).unwrap(),
                        id,
                        "{}: d={d} k={k} reverse conversion",
                        kind.name()
                    );
                }
            }
        }
    }

    #[test]
    fn trivial_group() {
        assert_eq!(PointGroup::TRIVIAL, PointGroup::default());
        assert_eq!(PointGroup::TRIVIAL.nbits(), 0);
        assert_eq!(PointGroup::TRIVIAL.order(), 1);
        assert!(PointGroup::TRIVIAL.is_trivial());

        for k in (0..=128usize).chain(std::iter::once(usize::MAX)) {
            assert_eq!(PointGroup::TRIVIAL.irrep_of(k), Irrep::IDENTITY, "k={k}");
        }

        let orbsym = vec![1u8; 20];
        assert_eq!(
            PointGroup::from_orbsym(PointGroupKind::C1, &orbsym).unwrap(),
            PointGroup::TRIVIAL
        );

        let all_symmetric = vec![Irrep::IDENTITY; 128];
        assert_eq!(
            PointGroup::try_new(0, &all_symmetric).unwrap(),
            PointGroup::TRIVIAL
        );
    }

    #[test]
    fn eq_and_hash_are_content_based() {
        let kind = PointGroupKind::D2h;
        let irreps: Vec<Irrep> = (0..10u8).map(|i| Irrep::new(i % kind.order())).collect();
        let ids: Vec<u8> = irreps
            .iter()
            .map(|ir| kind.molpro_from_irrep(*ir).unwrap())
            .collect();

        let a = PointGroup::try_new(kind.nbits(), &irreps).unwrap();
        let b = PointGroup::from_orbsym(kind, &ids).unwrap();
        assert_eq!(
            a, b,
            "same irrep content via different constructors must be equal"
        );

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(
            set.len(),
            1,
            "equal PointGroups must hash into the same bucket"
        );

        let mut irreps2 = irreps.clone();
        irreps2[3] = Irrep::new((irreps2[3].index() + 1) % kind.order());
        let c = PointGroup::try_new(kind.nbits(), &irreps2).unwrap();
        assert_ne!(a, c, "flipping one site's irrep must change equality");
        set.insert(c);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn try_new_accepts_empty_and_boundary_sizes() {
        assert_eq!(PointGroup::try_new(0, &[]).unwrap(), PointGroup::TRIVIAL);

        let irreps = vec![Irrep::new(0); 128];
        assert!(
            PointGroup::try_new(3, &irreps).is_ok(),
            "d=128 must be accepted (the MAX_SITES boundary)"
        );
    }

    #[test]
    fn rejects_too_many_sites() {
        let irreps = vec![Irrep::new(0); 129];
        let err = PointGroup::try_new(3, &irreps).unwrap_err();
        assert!(
            matches!(err, PointGroupError::TooManySites { d: 129, max: 128 }),
            "expected TooManySites {{ d: 129, max: 128 }}, got {err:?}"
        );
    }

    #[test]
    fn rejects_too_many_bits() {
        let err = PointGroup::try_new(4, &[]).unwrap_err();
        assert!(
            matches!(err, PointGroupError::TooManyBits { nbits: 4, max: 3 }),
            "expected TooManyBits {{ nbits: 4, max: 3 }}, got {err:?}"
        );
    }

    #[test]
    fn rejects_irrep_out_of_range_in_try_new() {
        let err = PointGroup::try_new(2, &[Irrep::new(4)]).unwrap_err();
        assert!(
            matches!(
                err,
                PointGroupError::IrrepOutOfRange {
                    irrep: 4,
                    order: 4,
                    site: Some(0),
                    ..
                }
            ),
            "expected IrrepOutOfRange {{ irrep: 4, order: 4, site: Some(0) }}, got {err:?}"
        );
    }

    #[test]
    fn rejects_bad_molpro_id_in_from_orbsym() {
        let err = PointGroup::from_orbsym(PointGroupKind::D2h, &[1, 2, 0]).unwrap_err();
        assert!(
            matches!(
                err,
                PointGroupError::MolproIdOutOfRange {
                    id: 0,
                    site: Some(2),
                    ..
                }
            ),
            "expected MolproIdOutOfRange {{ id: 0, site: Some(2) }}, got {err:?}"
        );
    }

    #[test]
    fn rejects_molpro_id_out_of_range_scalar() {
        let low = PointGroupKind::D2h.irrep_from_molpro(0).unwrap_err();
        assert!(
            matches!(low, PointGroupError::MolproIdOutOfRange { id: 0, .. }),
            "expected MolproIdOutOfRange {{ id: 0, .. }}, got {low:?}"
        );

        let high = PointGroupKind::D2h.irrep_from_molpro(9).unwrap_err();
        assert!(
            matches!(high, PointGroupError::MolproIdOutOfRange { id: 9, .. }),
            "expected MolproIdOutOfRange {{ id: 9, .. }}, got {high:?}"
        );
    }

    #[test]
    fn rejects_irrep_out_of_range_scalar() {
        let err = PointGroupKind::C2v
            .molpro_from_irrep(Irrep::new(4))
            .unwrap_err();
        assert!(
            matches!(
                err,
                PointGroupError::IrrepOutOfRange {
                    irrep: 4,
                    order: 4,
                    ..
                }
            ),
            "expected IrrepOutOfRange {{ irrep: 4, order: 4, .. }}, got {err:?}"
        );
    }

    #[test]
    fn fuse_does_not_validate_out_of_range_irreps() {
        assert_eq!(Irrep::new(200).fuse(Irrep::new(3)), Irrep::new(203));
    }

    #[test]
    #[should_panic(expected = "129")]
    fn new_panics_on_too_many_sites() {
        let irreps = vec![Irrep::new(0); 129];
        let _ = PointGroup::new(3, &irreps);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn new_panics_on_irrep_out_of_range() {
        let _ = PointGroup::new(2, &[Irrep::new(4)]);
    }

    #[test]
    fn irrep_and_constant_basics() {
        assert_eq!(Irrep::new(5).index(), 5);
        assert_eq!(Irrep::default(), Irrep::IDENTITY);
        assert_eq!(Irrep::IDENTITY.index(), 0);

        assert_eq!(PointGroupKind::default(), PointGroupKind::C1);

        assert_eq!(PointGroup::MAX_SITES, 128);
        assert_eq!(PointGroup::MAX_BITS, 3);
    }

    #[test]
    fn irrep_of_sites_fuses_the_named_sites() {
        let kind = PointGroupKind::D2h;
        let mut irreps = vec![Irrep::IDENTITY; 10];
        irreps[2] = Irrep::new(3);
        irreps[5] = Irrep::new(5);
        irreps[7] = Irrep::new(6);
        let pg = PointGroup::try_new(kind.nbits(), &irreps).unwrap();

        assert_eq!(pg.irrep_of_sites(std::iter::empty()), Irrep::IDENTITY);

        for k in [0usize, 2, 5, 7, 9] {
            assert_eq!(pg.irrep_of_sites([k]), pg.irrep_of(k), "singleton k={k}");
        }

        assert_eq!(
            pg.irrep_of_sites([2, 5, 7]),
            Irrep::new(3 ^ 5 ^ 6),
            "expected explicit XOR of the three site irreps"
        );
        assert_eq!(pg.irrep_of_sites([2, 5, 7]), Irrep::IDENTITY);

        assert_eq!(pg.irrep_of_sites([2, 5]), Irrep::new(3 ^ 5));
        assert_ne!(pg.irrep_of_sites([2, 5]), Irrep::IDENTITY);

        assert_eq!(pg.irrep_of_sites([7, 2, 5]), pg.irrep_of_sites([2, 5, 7]));
        assert_eq!(pg.irrep_of_sites([2, 2]), Irrep::IDENTITY);

        let sites = [0usize, 2, 3, 5, 7, 9];
        let manual = sites
            .iter()
            .fold(Irrep::IDENTITY, |acc, &k| acc.fuse(pg.irrep_of(k)));
        assert_eq!(pg.irrep_of_sites(sites), manual);
    }

    #[test]
    fn from_orbsym_rejects_too_many_sites() {
        let ids = vec![1u8; 129];
        let err = PointGroup::from_orbsym(PointGroupKind::D2h, &ids).unwrap_err();
        assert!(
            matches!(err, PointGroupError::TooManySites { d: 129, max: 128 }),
            "expected TooManySites {{ d: 129, max: 128 }}, got {err:?}"
        );

        let ids_ok = vec![1u8; 128];
        assert!(
            PointGroup::from_orbsym(PointGroupKind::D2h, &ids_ok).is_ok(),
            "d=128 must still be accepted by from_orbsym (the MAX_SITES boundary)"
        );
    }

    #[test]
    fn irrep_name_rejects_out_of_range() {
        for kind in PointGroupKind::ALL {
            let order = kind.order();
            for bad in [order, order + 1, u8::MAX] {
                let err = kind.irrep_name(Irrep::new(bad)).unwrap_err();
                assert!(
                    matches!(
                        err,
                        PointGroupError::IrrepOutOfRange {
                            irrep,
                            order: o,
                            site: None,
                            ..
                        } if irrep == bad && o == order
                    ),
                    "{}: irrep_name({bad}) expected IrrepOutOfRange {{ irrep: {bad}, order: {order}, site: None }}, got {err:?}",
                    kind.name()
                );
            }
            for good in 0..order {
                assert!(kind.irrep_name(Irrep::new(good)).is_ok());
            }
        }
    }

    #[test]
    fn from_name_and_from_str_reject_unknown_names() {
        for bad in ["", "C3v", "D4h", "Td", "Oh", "d2hh", "C1 ", " C1", "XYZ"] {
            assert_eq!(
                PointGroupKind::from_name(bad),
                None,
                "from_name({bad:?}) should be None"
            );
            let parsed: Result<PointGroupKind, PointGroupError> = bad.parse();
            match parsed {
                Err(PointGroupError::UnknownGroup { name }) => {
                    assert_eq!(name, bad, "UnknownGroup should carry the rejected input");
                }
                other => panic!("{bad:?}: expected Err(UnknownGroup {{ .. }}), got {other:?}"),
            }
        }

        for kind in PointGroupKind::ALL {
            assert_eq!(kind.name().parse::<PointGroupKind>().unwrap(), kind);
        }
    }

    #[test]
    fn is_trivial_false_branch_and_planes_round_trip() {
        for kind in PointGroupKind::ALL {
            let order = kind.order();
            let irreps: Vec<Irrep> = (0..16u8).map(|i| Irrep::new(i % order)).collect();
            let pg = PointGroup::try_new(kind.nbits(), &irreps).unwrap();

            assert_eq!(
                pg.is_trivial(),
                kind == PointGroupKind::C1,
                "{}: is_trivial() must be true only for the order-1 group",
                kind.name()
            );
            if kind != PointGroupKind::C1 {
                assert!(
                    !pg.is_trivial(),
                    "{}: expected is_trivial() == false",
                    kind.name()
                );
            }

            let planes = pg.planes();
            for (k, &ir) in irreps.iter().enumerate() {
                let b0 = ((planes[0] >> k) & 1) as u8;
                let b1 = ((planes[1] >> k) & 1) as u8;
                let b2 = ((planes[2] >> k) & 1) as u8;
                let reconstructed = Irrep::new(b0 | (b1 << 1) | (b2 << 2));
                assert_eq!(
                    reconstructed,
                    ir,
                    "{}: planes()-reconstructed irrep at k={k}",
                    kind.name()
                );
                assert_eq!(reconstructed, pg.irrep_of(k), "{}: k={k}", kind.name());
            }
            for (b, &plane) in planes.iter().enumerate() {
                assert_eq!(
                    (plane >> 16) & 1,
                    0,
                    "{}: plane {b} bit 16 (beyond-table) must be unset",
                    kind.name()
                );
            }
            for (b, &plane) in planes.iter().enumerate().skip(kind.nbits() as usize) {
                assert_eq!(plane, 0, "{}: unused plane {b} must be zero", kind.name());
            }
        }

        assert_eq!(PointGroup::TRIVIAL.planes(), [0u128; 3]);
    }
}

#[cfg(test)]
mod pg5_rpa_tests {
    use super::{Irrep, PointGroupKind};

    #[test]
    fn identity_is_molpro_1_in_every_group() {
        for kind in PointGroupKind::ALL {
            assert_eq!(
                kind.irrep_from_molpro(1).unwrap(),
                Irrep::IDENTITY,
                "{kind:?}"
            );
        }
    }

    #[test]
    fn pgr_characters_are_an_orthogonal_table() {
        for order_bits in 1..=3u8 {
            let order = 1u8 << order_bits;
            for g in 0..order {
                let gamma = Irrep::new(g);
                assert_eq!(gamma.character(Irrep::IDENTITY), 1.0);
                let row: f64 = (0..order).map(|h| gamma.character(Irrep::new(h))).sum();
                let want = if g == 0 { order as f64 } else { 0.0 };
                assert!(
                    (row - want).abs() < 1e-12,
                    "Σ_h χ_{g}(h) = {row}, want {want}"
                );
                for h in 0..order {
                    assert_eq!(
                        gamma.character(Irrep::new(h)),
                        Irrep::new(h).character(gamma),
                        "character pairing is not symmetric"
                    );
                }
            }
        }
    }
}
