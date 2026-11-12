// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! On-disk persistence for TT states and SU(2) operator trains — a **local cache, not an archival
//! format**.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::core::{Family, SparseCore};
use crate::pointgroup::{Irrep, PointGroup, PointGroupError};
use crate::qn::{QnSpace, RankGrid};
use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::qn::{StateSym, Su2RankGrid, Su2Space, SymSpace, SymStep, Symmetry};
use crate::su2::vector::Su2TtVector;
use crate::vector::TtVector;

/// File magic — the first eight bytes of every state file.
pub const MAGIC: [u8; 8] = *b"QNTT-TT\0";

/// Format version this build reads and writes. See the module docs: a cache tag, not a
/// compatibility contract.
pub const FORMAT_VERSION: u32 = 1;

/// File magic of an SU(2) operator file (`QNTT-MPO`, module docs).
pub const MPO_MAGIC: [u8; 8] = *b"QNTT-MPO";

const LAYER_ABELIAN: u8 = 0;
const LAYER_SU2: u8 = 1;
const SCALAR_F64: u8 = 0;
const SCALAR_C64: u8 = 1;

fn layer_name(tag: u8) -> &'static str {
    match tag {
        LAYER_ABELIAN => "abelian",
        LAYER_SU2 => "SU(2)",
        _ => "unknown",
    }
}

fn scalar_name(tag: u8) -> &'static str {
    match tag {
        SCALAR_F64 => "f64",
        SCALAR_C64 => "C64",
        _ => "unknown",
    }
}

fn scalar_tag<T: Scalar>() -> u8 {
    if T::IS_COMPLEX {
        SCALAR_C64
    } else {
        SCALAR_F64
    }
}

/// Anything that can go wrong reading (or writing) a TT state file.
#[derive(Debug, thiserror::Error)]
pub enum TtIoError {
    /// Underlying stream failure.
    #[error("tt_io: {0}")]
    Io(#[from] std::io::Error),
    /// The first eight bytes are not the requested format's magic ([`MAGIC`] for a state,
    /// [`MPO_MAGIC`] for an operator) — not a file of the requested kind.
    #[error(
        "tt_io: not a qntt file of the requested kind (magic {found:02x?}; a TT state starts \
         with {:02x?}, an SU(2) operator with {:02x?})",
        MAGIC,
        MPO_MAGIC
    )]
    BadMagic { found: [u8; 8] },
    /// A version this build does not read. Caches are not migrated; delete and recompute.
    #[error(
        "tt_io: unsupported format version {found} (this build reads {supported}); a state cache \
         is not an archival format — delete the file and recompute"
    )]
    UnsupportedVersion { found: u32, supported: u32 },
    /// The file holds the other symmetry layer.
    #[error("tt_io: file holds a {} state, but a {} state was requested", layer_name(*found), layer_name(*expected))]
    LayerMismatch { found: u8, expected: u8 },
    /// The file holds the other scalar type. Widening `f64` → `C64` on read is deliberately not
    /// offered: silently changing a state's field is exactly the kind of thing a cache must not do.
    #[error("tt_io: file holds {} coefficients, but {} was requested", scalar_name(*found), scalar_name(*expected))]
    ScalarMismatch { found: u8, expected: u8 },
    /// The stream ended early.
    #[error("tt_io: truncated file: needed {want} more bytes for {context}, got {got}")]
    Truncated {
        context: &'static str,
        want: usize,
        got: usize,
    },
    /// Bytes remain after a complete state — the file is not what it claims to be.
    #[error("tt_io: {extra} trailing byte(s) after the end of the state")]
    TrailingBytes { extra: usize },
    /// The declared sector count at a bond disagrees with what the header's space implies.
    #[error(
        "tt_io: sector-count mismatch at bond {bond}: the space implies {expected} sectors, the \
         file declares {found}"
    )]
    SectorCountMismatch {
        bond: usize,
        expected: usize,
        found: u64,
    },
    /// The declared coefficient count at a site disagrees with what the rank grids imply.
    #[error(
        "tt_io: coefficient-count mismatch at site {site}: the rank grids imply {expected} \
         coefficients, the file declares {found}"
    )]
    CoeffCountMismatch {
        site: usize,
        expected: usize,
        found: u64,
    },
    /// A header field no constructor would accept (space parameters, `ortho`, the boundary law).
    #[error("tt_io: invalid header: {0}")]
    InvalidHeader(String),
    /// The per-site irrep table does not describe a point group.
    #[error("tt_io: invalid point-group table: {0}")]
    PointGroup(#[from] PointGroupError),
}

fn put_u8<W: Write>(w: &mut W, v: u8) -> Result<(), TtIoError> {
    w.write_all(&[v])?;
    Ok(())
}

fn put_u32<W: Write>(w: &mut W, v: u32) -> Result<(), TtIoError> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn put_u64<W: Write>(w: &mut W, v: u64) -> Result<(), TtIoError> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn put_usize<W: Write>(w: &mut W, v: usize) -> Result<(), TtIoError> {
    put_u64(w, v as u64)
}

fn put_i64<W: Write>(w: &mut W, v: i64) -> Result<(), TtIoError> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn fill<R: Read>(r: &mut R, buf: &mut [u8], context: &'static str) -> Result<(), TtIoError> {
    let want = buf.len();
    let mut got = 0usize;
    while got < want {
        match r.read(&mut buf[got..]) {
            Ok(0) => return Err(TtIoError::Truncated { context, want, got }),
            Ok(n) => got += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(TtIoError::Io(e)),
        }
    }
    Ok(())
}

fn get_u8<R: Read>(r: &mut R, context: &'static str) -> Result<u8, TtIoError> {
    let mut b = [0u8; 1];
    fill(r, &mut b, context)?;
    Ok(b[0])
}

fn get_u32<R: Read>(r: &mut R, context: &'static str) -> Result<u32, TtIoError> {
    let mut b = [0u8; 4];
    fill(r, &mut b, context)?;
    Ok(u32::from_le_bytes(b))
}

fn get_u64<R: Read>(r: &mut R, context: &'static str) -> Result<u64, TtIoError> {
    let mut b = [0u8; 8];
    fill(r, &mut b, context)?;
    Ok(u64::from_le_bytes(b))
}

fn get_i64<R: Read>(r: &mut R, context: &'static str) -> Result<i64, TtIoError> {
    let mut b = [0u8; 8];
    fill(r, &mut b, context)?;
    Ok(i64::from_le_bytes(b))
}

fn get_ortho<R: Read>(r: &mut R, d: usize) -> Result<Option<usize>, TtIoError> {
    match get_i64(r, "the orthogonality centre")? {
        -1 => Ok(None),
        k if k >= 0 && (k as usize) < d => Ok(Some(k as usize)),
        k => Err(TtIoError::InvalidHeader(format!(
            "orthogonality centre {k} is neither -1 (none) nor a site index in 0..{d}"
        ))),
    }
}

fn as_usize(v: u64, what: &str) -> Result<usize, TtIoError> {
    usize::try_from(v)
        .map_err(|_| TtIoError::InvalidHeader(format!("{what} = {v} does not fit in a usize")))
}

fn expect_eof<R: Read>(r: &mut R) -> Result<(), TtIoError> {
    let mut extra = [0u8; 1];
    match r.read(&mut extra) {
        Ok(0) => Ok(()),
        Ok(n) => Err(TtIoError::TrailingBytes { extra: n }),
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => Ok(()),
        Err(e) => Err(TtIoError::Io(e)),
    }
}

struct Header {
    d: usize,
    a: usize,
    b: usize,
    pg: PointGroup,
    target: Irrep,
    ortho: Option<usize>,
}

fn put_header<W: Write, T: Scalar>(w: &mut W, layer: u8, h: &Header) -> Result<(), TtIoError> {
    w.write_all(&MAGIC)?;
    put_u32(w, FORMAT_VERSION)?;
    put_u8(w, layer)?;
    put_u8(w, scalar_tag::<T>())?;
    put_usize(w, h.d)?;
    put_usize(w, h.a)?;
    put_usize(w, h.b)?;
    put_u8(w, h.pg.nbits())?;
    put_u8(w, h.target.index())?;
    for k in 0..h.d {
        put_u8(w, h.pg.irrep_of(k).index())?;
    }
    put_i64(w, h.ortho.map_or(-1, |k| k as i64))?;
    Ok(())
}

fn get_header<R: Read, T: Scalar>(r: &mut R, layer: u8) -> Result<Header, TtIoError> {
    let mut magic = [0u8; 8];
    fill(r, &mut magic, "the file magic")?;
    if magic != MAGIC {
        return Err(TtIoError::BadMagic { found: magic });
    }
    let version = get_u32(r, "the format version")?;
    if version != FORMAT_VERSION {
        return Err(TtIoError::UnsupportedVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    let found_layer = get_u8(r, "the layer tag")?;
    if found_layer != layer {
        return Err(TtIoError::LayerMismatch {
            found: found_layer,
            expected: layer,
        });
    }
    let found_scalar = get_u8(r, "the scalar tag")?;
    if found_scalar != scalar_tag::<T>() {
        return Err(TtIoError::ScalarMismatch {
            found: found_scalar,
            expected: scalar_tag::<T>(),
        });
    }
    let d = as_usize(get_u64(r, "the site count")?, "d")?;
    let a = as_usize(
        get_u64(r, "the first space parameter")?,
        "the first space parameter",
    )?;
    let b = as_usize(
        get_u64(r, "the second space parameter")?,
        "the second space parameter",
    )?;
    if d > PointGroup::MAX_SITES {
        return Err(TtIoError::InvalidHeader(format!(
            "d = {d} exceeds the {}-site point-group table limit",
            PointGroup::MAX_SITES
        )));
    }
    let nbits = get_u8(r, "the point-group generator count")?;
    let target = Irrep::new(get_u8(r, "the target irrep")?);
    let mut irreps = vec![Irrep::IDENTITY; d];
    for ir in irreps.iter_mut() {
        *ir = Irrep::new(get_u8(r, "the per-site irrep table")?);
    }
    let pg = PointGroup::try_new(nbits, &irreps)?;
    if target.index() >= pg.order() {
        return Err(TtIoError::InvalidHeader(format!(
            "target irrep {} is out of range 0..{}",
            target.index(),
            pg.order()
        )));
    }
    let ortho = get_ortho(r, d)?;
    Ok(Header {
        d,
        a,
        b,
        pg,
        target,
        ortho,
    })
}

fn put_coeffs<W: Write, T: Scalar>(
    w: &mut W,
    ncoeff: usize,
    each: impl FnOnce(&mut dyn FnMut(T)),
) -> Result<(), TtIoError> {
    put_usize(w, ncoeff)?;
    let width = if T::IS_COMPLEX { 16 } else { 8 };
    let mut bytes = Vec::with_capacity(ncoeff * width);
    let complex = T::IS_COMPLEX;
    each(&mut |c: T| {
        bytes.extend_from_slice(&c.re_s().to_le_bytes());
        if complex {
            bytes.extend_from_slice(&c.im_s().to_le_bytes());
        }
    });
    debug_assert_eq!(bytes.len(), ncoeff * width, "coefficient walk length");
    w.write_all(&bytes)?;
    Ok(())
}

fn get_coeffs<R: Read, T: Scalar>(
    r: &mut R,
    site: usize,
    expected: usize,
) -> Result<Vec<u8>, TtIoError> {
    let found = get_u64(r, "a coefficient count")?;
    if found != expected as u64 {
        return Err(TtIoError::CoeffCountMismatch {
            site,
            expected,
            found,
        });
    }
    let width = if T::IS_COMPLEX { 16 } else { 8 };
    let mut bytes = vec![0u8; expected * width];
    fill(r, &mut bytes, "a coefficient stream")?;
    Ok(bytes)
}

fn decode_coeffs<T: Scalar>(bytes: &[u8]) -> impl Iterator<Item = T> + '_ {
    let width = if T::IS_COMPLEX { 16 } else { 8 };
    bytes.chunks_exact(width).map(|chunk| {
        let re = f64::from_le_bytes(chunk[..8].try_into().expect("8 bytes"));
        let im = if T::IS_COMPLEX {
            f64::from_le_bytes(chunk[8..16].try_into().expect("8 bytes"))
        } else {
            0.0
        };
        T::from_complex_f64(re, im)
    })
}

const MAX_RANK: u64 = 1 << 20;

const MAX_CORE_COEFFS: u64 = 1 << 32;

fn get_rank<R: Read>(r: &mut R, bond: usize) -> Result<usize, TtIoError> {
    let raw = get_u64(r, "a sector rank")?;
    if raw > MAX_RANK {
        return Err(TtIoError::InvalidHeader(format!(
            "bond {bond}: sector rank {raw} exceeds the sanity cap of {MAX_RANK}"
        )));
    }
    as_usize(raw, "a sector rank")
}

/// Serialize an abelian TT state to a byte stream. See the module docs for the format.
pub fn write_tt<T: Scalar, W: Write>(w: &mut W, x: &TtVector<T>) -> Result<(), TtIoError> {
    let space = x.space;
    let d = space.d;
    put_header::<W, T>(
        w,
        LAYER_ABELIAN,
        &Header {
            d,
            a: space.nup,
            b: space.ndn,
            pg: space.pg,
            target: space.target,
            ortho: x.ortho,
        },
    )?;
    for b in 0..=d {
        let sectors = space.bond_sectors(b);
        put_usize(w, sectors.len())?;
        for s in sectors {
            put_usize(w, x.ranks[b][s])?;
        }
    }
    for k in 0..d {
        let core = &x.cores[k];
        put_coeffs::<W, T>(w, core.num_coeffs(), |sink| core.for_each_coeff(sink))?;
    }
    Ok(())
}

/// Deserialize an abelian TT state written by [`write_tt`].
pub fn read_tt<T: Scalar, R: Read>(r: &mut R) -> Result<TtVector<T>, TtIoError> {
    let h = get_header::<R, T>(r, LAYER_ABELIAN)?;
    if h.a > h.d || h.b > h.d {
        return Err(TtIoError::InvalidHeader(format!(
            "electron counts Nup = {}, Ndn = {} cannot exceed d = {}",
            h.a, h.b, h.d
        )));
    }
    let space = QnSpace::with_point_group(h.d, h.a, h.b, h.pg, h.target);
    let d = h.d;

    let mut ranks: Vec<RankGrid> = Vec::with_capacity(d + 1);
    for b in 0..=d {
        let sectors = space.bond_sectors(b);
        let found = get_u64(r, "a bond's sector count")?;
        if found != sectors.len() as u64 {
            return Err(TtIoError::SectorCountMismatch {
                bond: b,
                expected: sectors.len(),
                found,
            });
        }
        let mut g = RankGrid::zeros(&space);
        for s in sectors {
            g[s] = get_rank(r, b)?;
        }
        ranks.push(g);
    }
    let first = crate::qn::Sector::new(0, 0);
    let last = crate::qn::Sector::with_irrep(space.nup, space.ndn, space.target);
    let off0 = ranks[0]
        .iter()
        .filter(|&(s, &r)| r != 0 && s != first)
        .count();
    let offd = ranks[d]
        .iter()
        .filter(|&(s, &r)| r != 0 && s != last)
        .count();
    check_boundary(off0, 0)?;
    check_boundary(offd, d)?;

    let mut cores: Vec<SparseCore<T>> = Vec::with_capacity(d);
    for k in 0..d {
        checked_core_coeffs(space, k, &ranks[k], &ranks[k + 1])?;
        let mut core = SparseCore::<T>::zeros(space, k, &ranks[k], &ranks[k + 1]);
        let vals = get_coeffs::<R, T>(r, k, core.num_coeffs())?;
        let mut it = decode_coeffs::<T>(&vals);
        core.for_each_coeff_mut(|c| *c = it.next().expect("count validated above"));
        debug_assert!(it.next().is_none(), "coefficient walk length");
        cores.push(core);
    }
    expect_eof(r)?;

    let mut x = TtVector::new(space, cores);
    x.ortho = h.ortho;
    x.check();
    Ok(x)
}

fn checked_core_coeffs(
    space: QnSpace,
    k: usize,
    row_ranks: &RankGrid,
    col_ranks: &RankGrid,
) -> Result<(), TtIoError> {
    let g = space.pg.irrep_of(k);
    let mut total: u64 = 0;
    for f in Family::ALL {
        for s in space.bond_sectors(k) {
            let t = f.apply(s, g);
            if !space.contains(k + 1, t) {
                continue;
            }
            let block = (row_ranks[s] as u64)
                .checked_mul(col_ranks[t] as u64)
                .ok_or_else(|| {
                    TtIoError::InvalidHeader(format!(
                        "site {k}: a block's coefficient count overflows a u64"
                    ))
                })?;
            total = total.checked_add(block).ok_or_else(|| {
                TtIoError::InvalidHeader(format!(
                    "site {k}: the core's total coefficient count overflows a u64"
                ))
            })?;
            if total > MAX_CORE_COEFFS {
                return Err(TtIoError::InvalidHeader(format!(
                    "site {k}: the rank grids imply at least {total} coefficients, exceeding the \
                     sanity cap of {MAX_CORE_COEFFS}"
                )));
            }
        }
    }
    Ok(())
}

fn check_boundary(offenders: usize, bond: usize) -> Result<(), TtIoError> {
    if offenders != 0 {
        return Err(TtIoError::InvalidHeader(format!(
            "bond {bond}: {offenders} sector(s) carry positive rank away from the boundary \
             sector, which no state vector can"
        )));
    }
    Ok(())
}

/// Write an abelian TT state to `path` (truncating). Convenience wrapper over [`write_tt`].
pub fn save_tt<T: Scalar>(path: impl AsRef<Path>, x: &TtVector<T>) -> Result<(), TtIoError> {
    let mut w = BufWriter::new(File::create(path)?);
    write_tt(&mut w, x)?;
    w.flush()?;
    Ok(())
}

/// Read an abelian TT state from `path`. Convenience wrapper over [`read_tt`].
pub fn load_tt<T: Scalar>(path: impl AsRef<Path>) -> Result<TtVector<T>, TtIoError> {
    let mut r = BufReader::new(File::open(path)?);
    read_tt(&mut r)
}

/// Serialize an SU(2) TT state to a byte stream. See the module docs for the format.
pub fn write_su2_tt<T: Scalar, W: Write>(w: &mut W, x: &Su2TtVector<T>) -> Result<(), TtIoError> {
    let space = x.space;
    let d = space.d;
    put_header::<W, T>(
        w,
        LAYER_SU2,
        &Header {
            d,
            a: space.n,
            b: space.two_s,
            pg: space.pg,
            target: space.target,
            ortho: x.ortho,
        },
    )?;
    for b in 0..=d {
        let sectors = space.bond_sectors(b);
        put_usize(w, sectors.len())?;
        for s in sectors {
            put_usize(w, x.ranks[b][s])?;
        }
    }
    for k in 0..d {
        let core = &x.cores[k];
        put_coeffs::<W, T>(w, core.num_coeffs(), |sink| core.for_each_coeff(sink))?;
    }
    Ok(())
}

fn checked_su2_core_coeffs<S: Symmetry>(
    space: S::Space,
    k: usize,
    row_ranks: &S::Grid<usize>,
    col_ranks: &S::Grid<usize>,
) -> Result<(), TtIoError> {
    let g = space.site_ctx(k);
    let mut total: u64 = 0;
    for &f in S::STEPS {
        for s in space.bond_labels(k) {
            let Some(t) = f.step_apply(s, g) else {
                continue;
            };
            if !space.admissible(k + 1, t) {
                continue;
            }
            let block = (row_ranks[s] as u64)
                .checked_mul(col_ranks[t] as u64)
                .ok_or_else(|| {
                    TtIoError::InvalidHeader(format!(
                        "site {k}: a block's coefficient count overflows a u64"
                    ))
                })?;
            total = total.checked_add(block).ok_or_else(|| {
                TtIoError::InvalidHeader(format!(
                    "site {k}: the core's total coefficient count overflows a u64"
                ))
            })?;
            if total > MAX_CORE_COEFFS {
                return Err(TtIoError::InvalidHeader(format!(
                    "site {k}: the rank grids imply at least {total} coefficients, exceeding the \
                     sanity cap of {MAX_CORE_COEFFS}"
                )));
            }
        }
    }
    Ok(())
}

/// Deserialize an SU(2) TT state written by [`write_su2_tt`].
pub fn read_su2_tt<T: Scalar, R: Read>(r: &mut R) -> Result<Su2TtVector<T>, TtIoError> {
    let h = get_header::<R, T>(r, LAYER_SU2)?;
    let (n, two_s, d) = (h.a, h.b, h.d);
    if n > 2 * d {
        return Err(TtIoError::InvalidHeader(format!(
            "N = {n} cannot exceed 2d = {}",
            2 * d
        )));
    }
    let cap = n.min(2 * d - n);
    if two_s > cap {
        return Err(TtIoError::InvalidHeader(format!(
            "2S = {two_s} exceeds min(N, 2d-N) = {cap} (N = {n}, d = {d})"
        )));
    }
    if !(n + two_s).is_multiple_of(2) {
        return Err(TtIoError::InvalidHeader(format!(
            "N - 2S must be even (N = {n}, 2S = {two_s})"
        )));
    }
    let space = Su2Space::with_point_group(d, n, two_s, h.pg, h.target);

    let mut ranks: Vec<Su2RankGrid> = Vec::with_capacity(d + 1);
    for b in 0..=d {
        let sectors = space.bond_sectors(b);
        let found = get_u64(r, "a bond's sector count")?;
        if found != sectors.len() as u64 {
            return Err(TtIoError::SectorCountMismatch {
                bond: b,
                expected: sectors.len(),
                found,
            });
        }
        let mut g = Su2RankGrid::zeros(&space);
        for s in sectors {
            g[s] = get_rank(r, b)?;
        }
        ranks.push(g);
    }
    let first = crate::su2::qn::Su2Sector::new(0, 0);
    let last = crate::su2::qn::Su2Sector::with_irrep(space.n, space.two_s, space.target);
    let off0 = ranks[0]
        .iter()
        .filter(|&(s, &r)| r != 0 && s != first)
        .count();
    let offd = ranks[d]
        .iter()
        .filter(|&(s, &r)| r != 0 && s != last)
        .count();
    check_boundary(off0, 0)?;
    check_boundary(offd, d)?;

    let mut cores: Vec<Su2Core<T>> = Vec::with_capacity(d);
    for k in 0..d {
        checked_su2_core_coeffs::<StateSym>(space, k, &ranks[k], &ranks[k + 1])?;
        let mut core = Su2Core::<T>::zeros(space, k, &ranks[k], &ranks[k + 1]);
        let vals = get_coeffs::<R, T>(r, k, core.num_coeffs())?;
        let mut it = decode_coeffs::<T>(&vals);
        core.for_each_coeff_mut(|c| *c = it.next().expect("count validated above"));
        debug_assert!(it.next().is_none(), "coefficient walk length");
        cores.push(core);
    }
    expect_eof(r)?;

    let mut x = Su2TtVector::new(space, cores);
    x.ortho = h.ortho;
    x.check();
    Ok(x)
}

/// Write an SU(2) TT state to `path` (truncating). Convenience wrapper over [`write_su2_tt`].
pub fn save_su2_tt<T: Scalar>(path: impl AsRef<Path>, x: &Su2TtVector<T>) -> Result<(), TtIoError> {
    let mut w = BufWriter::new(File::create(path)?);
    write_su2_tt(&mut w, x)?;
    w.flush()?;
    Ok(())
}

/// Read an SU(2) TT state from `path`. Convenience wrapper over [`read_su2_tt`].
pub fn load_su2_tt<T: Scalar>(path: impl AsRef<Path>) -> Result<Su2TtVector<T>, TtIoError> {
    let mut r = BufReader::new(File::open(path)?);
    read_su2_tt(&mut r)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    use crate::integrals::read_meta;
    use crate::pointgroup::{Irrep, PointGroup, PointGroupKind};
    use crate::qn::{QnSpace, RankGrid, max_ranks};
    use crate::su2::qn::{Su2RankGrid, Su2Space, max_ranks as su2_max_ranks};
    use crate::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
    use crate::tt_io::{
        FORMAT_VERSION, MAGIC, TtIoError, load_su2_tt, load_tt, read_su2_tt, read_tt, save_su2_tt,
        save_tt, write_su2_tt, write_tt,
    };
    use crate::vector::{TtVector, tt_randn};
    use crate::{C64, Scalar};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn rng(seed: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(seed)
    }

    fn capped_grids(space: &QnSpace, m: usize) -> Vec<RankGrid> {
        max_ranks(space)
            .into_iter()
            .map(|g| {
                let mut out = RankGrid::zeros(space);
                for (s, &c) in g.iter() {
                    out[s] = c.min(m);
                }
                out
            })
            .collect()
    }

    fn su2_capped_grids(space: &Su2Space, m: usize) -> Vec<Su2RankGrid> {
        su2_max_ranks(space)
            .into_iter()
            .map(|g| {
                let mut out = Su2RankGrid::zeros(space);
                for (s, &c) in g.iter() {
                    out[s] = c.min(m);
                }
                out
            })
            .collect()
    }

    fn assert_nonvacuous<T: Scalar>(x: &TtVector<T>) {
        let d = x.space.d;
        let multi = (0..=d).any(|b| {
            x.space
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| x.ranks[b][s] > 0)
                .count()
                > 1
        });
        let maxr = (0..=d)
            .flat_map(|b| {
                x.space
                    .bond_sectors(b)
                    .into_iter()
                    .map(move |s| x.ranks[b][s])
            })
            .max()
            .unwrap_or(0);
        let ncoeff: usize = (0..d).map(|k| x.cores[k].num_coeffs()).sum();
        assert!(
            multi,
            "vacuous fixture: no bond carries more than one sector"
        );
        assert!(maxr > 1, "vacuous fixture: every sector rank is <= 1");
        assert!(ncoeff > 0, "vacuous fixture: no coefficients");
    }

    fn assert_nonvacuous_su2<T: Scalar>(x: &Su2TtVector<T>) {
        let d = x.space.d;
        let multi = (0..=d).any(|b| {
            x.space
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| x.ranks[b][s] > 0)
                .count()
                > 1
        });
        let ncoeff: usize = (0..d).map(|k| x.cores[k].num_coeffs()).sum();
        assert!(
            multi,
            "vacuous fixture: no bond carries more than one sector"
        );
        assert!(
            x.max_sector_rank() > 1,
            "vacuous fixture: every sector rank is <= 1"
        );
        assert!(ncoeff > 0, "vacuous fixture: no coefficients");
    }

    fn coeffs<T: Scalar>(c: &crate::core::SparseCore<T>) -> Vec<T> {
        let mut out = Vec::with_capacity(c.num_coeffs());
        c.for_each_coeff(|v| out.push(v));
        out
    }

    fn su2_coeffs<T: Scalar>(c: &crate::su2::core::Su2Core<T>) -> Vec<T> {
        let mut out = Vec::with_capacity(c.num_coeffs());
        c.for_each_coeff(|v| out.push(v));
        out
    }

    fn roundtrip<T: Scalar>(x: &TtVector<T>) -> TtVector<T> {
        assert_nonvacuous(x);
        let mut buf = Vec::new();
        write_tt(&mut buf, x).expect("write_tt");
        let y: TtVector<T> = read_tt(&mut Cursor::new(&buf)).expect("read_tt");

        assert_eq!(x.space, y.space, "space");
        assert_eq!(x.ranks.len(), y.ranks.len(), "bond count");
        for b in 0..=x.space.d {
            assert_eq!(x.ranks[b], y.ranks[b], "rank grid at bond {b}");
        }
        assert_eq!(x.ortho, y.ortho, "ortho centre");
        for k in 0..x.space.d {
            let (a, b) = (coeffs(&x.cores[k]), coeffs(&y.cores[k]));
            assert_eq!(a.len(), b.len(), "coefficient count at site {k}");
            for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    u.re_s().to_bits() == v.re_s().to_bits()
                        && u.im_s().to_bits() == v.im_s().to_bits(),
                    "coefficient {i} of site {k} changed: {:?} -> {:?}",
                    (u.re_s(), u.im_s()),
                    (v.re_s(), v.im_s())
                );
            }
        }
        assert!(y.check());
        y
    }

    fn roundtrip_su2<T: Scalar>(x: &Su2TtVector<T>) -> Su2TtVector<T> {
        assert_nonvacuous_su2(x);
        let mut buf = Vec::new();
        write_su2_tt(&mut buf, x).expect("write_su2_tt");
        let y: Su2TtVector<T> = read_su2_tt(&mut Cursor::new(&buf)).expect("read_su2_tt");

        assert_eq!(x.space, y.space, "space");
        assert_eq!(x.ranks.len(), y.ranks.len(), "bond count");
        for b in 0..=x.space.d {
            assert_eq!(x.ranks[b], y.ranks[b], "rank grid at bond {b}");
        }
        assert_eq!(x.ortho, y.ortho, "ortho centre");
        for k in 0..x.space.d {
            let (a, b) = (su2_coeffs(&x.cores[k]), su2_coeffs(&y.cores[k]));
            assert_eq!(a.len(), b.len(), "coefficient count at site {k}");
            for (i, (u, v)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    u.re_s().to_bits() == v.re_s().to_bits()
                        && u.im_s().to_bits() == v.im_s().to_bits(),
                    "coefficient {i} of site {k} changed: {:?} -> {:?}",
                    (u.re_s(), u.im_s()),
                    (v.re_s(), v.im_s())
                );
            }
        }
        assert!(y.check());
        y
    }

    #[test]
    fn abelian_f64_roundtrip() {
        let space = QnSpace::new(6, 3, 3);
        let grids = capped_grids(&space, 3);
        let mut x: TtVector<f64> = tt_randn(space, &grids, &mut rng(1));
        assert_eq!(x.ortho, None, "tt_randn is not canonical");
        roundtrip(&x);
        x.right_orthogonalize(true);
        assert_eq!(x.ortho, Some(0), "right_orthogonalize sets the centre");
        roundtrip(&x);
    }

    #[test]
    fn abelian_c64_roundtrip() {
        let space = QnSpace::new(6, 3, 3);
        let grids = capped_grids(&space, 3);
        let mut x: TtVector<C64> = tt_randn(space, &grids, &mut rng(2));
        let any_imag = (0..space.d).any(|k| coeffs(&x.cores[k]).iter().any(|c| c.im_s() != 0.0));
        assert!(any_imag, "C64 fixture has no imaginary part to round-trip");
        roundtrip(&x);
        x.left_orthogonalize(true);
        roundtrip(&x);
    }

    #[test]
    fn su2_f64_roundtrip() {
        let space = Su2Space::new(6, 6, 0);
        let grids = su2_capped_grids(&space, 3);
        let mut x: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng(3));
        assert_eq!(x.ortho, None);
        roundtrip_su2(&x);
        x.right_orthogonalize(true);
        assert_eq!(x.ortho, Some(0));
        roundtrip_su2(&x);
    }

    #[test]
    fn su2_c64_roundtrip() {
        let space = Su2Space::new(6, 6, 0);
        let grids = su2_capped_grids(&space, 3);
        let mut x: Su2TtVector<C64> = su2_tt_randn(space, &grids, &mut rng(4));
        let any_imag =
            (0..space.d).any(|k| su2_coeffs(&x.cores[k]).iter().any(|c| c.im_s() != 0.0));
        assert!(any_imag, "C64 fixture has no imaginary part to round-trip");
        roundtrip_su2(&x);
        x.left_orthogonalize(true);
        roundtrip_su2(&x);
    }

    #[test]
    fn su2_triplet_roundtrip() {
        let space = Su2Space::new(6, 6, 2);
        let grids = su2_capped_grids(&space, 3);
        let x: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng(5));
        let y = roundtrip_su2(&x);
        assert_eq!(y.two_s(), 2);
    }

    fn testdata_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(name)
    }

    fn parse_meta(path: &Path) -> HashMap<String, String> {
        read_meta(path)
            .unwrap_or_else(|e| panic!("missing point-group fixture {path:?}: {e}"))
            .keys
    }

    fn fixture_point_group(name: &str) -> (PointGroup, Irrep, usize, usize, i64) {
        let meta = parse_meta(&testdata_path(name).join("meta.txt"));
        let kind: PointGroupKind = meta["point_group"].parse().expect("point group name");
        let orbsym: Vec<u8> = meta["orbsym_molpro"]
            .split_whitespace()
            .map(|s| s.parse().expect("molpro id"))
            .collect();
        let pg = PointGroup::from_orbsym(kind, &orbsym).expect("orbsym table");
        let isym: u8 = meta["isym"].parse().expect("isym");
        let target = kind.irrep_from_molpro(isym).expect("isym in range");
        let norb: usize = meta["norb"].parse().unwrap();
        let nelec: usize = meta["nelec"].parse().unwrap();
        let ms2: i64 = meta["ms2"].parse().unwrap();
        assert_eq!(orbsym.len(), norb, "orbsym length vs norb");
        (pg, target, norb, nelec, ms2)
    }

    #[test]
    fn abelian_point_group_roundtrip_ch2_c2v() {
        let (pg, target, d, nelec, ms2) = fixture_point_group("ch2_triplet_c2v");
        assert!(!pg.is_trivial(), "fixture must carry a real point group");
        let nup = ((nelec as i64 + ms2) / 2) as usize;
        let ndn = ((nelec as i64 - ms2) / 2) as usize;
        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
        let grids = capped_grids(&space, 3);
        let mut x: TtVector<f64> = tt_randn(space, &grids, &mut rng(11));
        x.right_orthogonalize(true);
        let y = roundtrip(&x);
        assert_eq!(y.space.pg, pg, "point-group table survived");
        assert_eq!(y.space.target, target, "target irrep survived");
        assert_eq!(y.space.pg.order(), 4, "C2v has order 4");
    }

    #[test]
    fn su2_point_group_roundtrip_ch2_c2v() {
        let (pg, target, d, nelec, ms2) = fixture_point_group("ch2_triplet_c2v");
        let two_s = ms2.unsigned_abs() as usize;
        assert_eq!(two_s, 2, "the CH2 fixture is the triplet");
        let space = Su2Space::with_point_group(d, nelec, two_s, pg, target);
        let grids = su2_capped_grids(&space, 3);
        let mut x: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng(12));
        x.right_orthogonalize(true);
        let y = roundtrip_su2(&x);
        assert_eq!(y.space.pg, pg);
        assert_eq!(y.space.target, target);
        assert_eq!(y.two_s(), 2);
    }

    #[test]
    fn abelian_point_group_roundtrip_n2_d2h() {
        let (pg, target, d, nelec, _ms2) = fixture_point_group("n2_d2h");
        assert_eq!(pg.nbits(), 3, "D2h has three generators");
        assert_eq!(d, 28);
        let space = QnSpace::with_point_group(d, nelec / 2, nelec / 2, pg, target);
        let grids = capped_grids(&space, 2);
        let x: TtVector<f64> = tt_randn(space, &grids, &mut rng(13));
        let y = roundtrip(&x);
        for k in 0..d {
            assert_eq!(y.space.pg.irrep_of(k), pg.irrep_of(k), "site {k} irrep");
        }
    }

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("qntt_tt_io_{}_{name}.tt", std::process::id()))
    }

    #[test]
    fn save_and_load_round_trip_through_a_file() {
        let space = QnSpace::new(5, 2, 2);
        let grids = capped_grids(&space, 3);
        let mut x: TtVector<f64> = tt_randn(space, &grids, &mut rng(21));
        x.right_orthogonalize(true);
        assert_nonvacuous(&x);

        let path = scratch("abelian");
        save_tt(&path, &x).expect("save_tt");
        let y: TtVector<f64> = load_tt(&path).expect("load_tt");
        assert_eq!(x.space, y.space);
        assert_eq!(x.ortho, y.ortho);
        for k in 0..space.d {
            assert_eq!(coeffs(&x.cores[k]), coeffs(&y.cores[k]), "site {k}");
        }

        let su2_space = Su2Space::new(5, 4, 0);
        let su2_grids = su2_capped_grids(&su2_space, 3);
        let mut z: Su2TtVector<f64> = su2_tt_randn(su2_space, &su2_grids, &mut rng(22));
        z.right_orthogonalize(true);
        assert_nonvacuous_su2(&z);
        let su2_path = scratch("su2");
        save_su2_tt(&su2_path, &z).expect("save_su2_tt");
        let w: Su2TtVector<f64> = load_su2_tt(&su2_path).expect("load_su2_tt");
        assert_eq!(z.space, w.space);
        for k in 0..su2_space.d {
            assert_eq!(su2_coeffs(&z.cores[k]), su2_coeffs(&w.cores[k]), "site {k}");
        }

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&su2_path);
    }

    #[test]
    fn load_reports_a_missing_file_as_io() {
        let path = scratch("does_not_exist");
        let _ = std::fs::remove_file(&path);
        match load_tt::<f64>(&path) {
            Err(TtIoError::Io(_)) => {}
            other => panic!("expected Io, got {other:?}"),
        }
    }

    fn abelian_bytes() -> (TtVector<f64>, Vec<u8>) {
        let space = QnSpace::new(5, 2, 2);
        let grids = capped_grids(&space, 3);
        let mut x: TtVector<f64> = tt_randn(space, &grids, &mut rng(31));
        x.right_orthogonalize(true);
        assert_nonvacuous(&x);
        let mut buf = Vec::new();
        write_tt(&mut buf, &x).expect("write_tt");
        (x, buf)
    }

    fn su2_bytes() -> (Su2TtVector<f64>, Vec<u8>) {
        let space = Su2Space::new(5, 4, 0);
        let grids = su2_capped_grids(&space, 3);
        let mut x: Su2TtVector<f64> = su2_tt_randn(space, &grids, &mut rng(32));
        x.right_orthogonalize(true);
        assert_nonvacuous_su2(&x);
        let mut buf = Vec::new();
        write_su2_tt(&mut buf, &x).expect("write_su2_tt");
        (x, buf)
    }

    const OFF_VERSION: usize = 8;
    const OFF_LAYER: usize = 12;
    const OFF_SCALAR: usize = 13;
    const OFF_D: usize = 14;
    const OFF_A: usize = 22;
    const OFF_B: usize = 30;

    #[test]
    fn bad_magic_is_rejected() {
        let (_, mut buf) = abelian_bytes();
        buf[0] = b'X';
        match read_tt::<f64, _>(&mut Cursor::new(&buf)) {
            Err(TtIoError::BadMagic { found }) => assert_ne!(found, MAGIC),
            other => panic!("expected BadMagic, got {other:?}"),
        }
        match read_tt::<f64, _>(&mut Cursor::new(Vec::new())) {
            Err(TtIoError::Truncated { .. }) => {}
            other => panic!("expected Truncated on an empty stream, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_version_is_rejected() {
        let (_, mut buf) = abelian_bytes();
        buf[OFF_VERSION..OFF_VERSION + 4].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        match read_tt::<f64, _>(&mut Cursor::new(&buf)) {
            Err(TtIoError::UnsupportedVersion { found, supported }) => {
                assert_eq!(found, FORMAT_VERSION + 1);
                assert_eq!(supported, FORMAT_VERSION);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn layer_mismatch_is_rejected_both_ways() {
        let (_, abelian) = abelian_bytes();
        match read_su2_tt::<f64, _>(&mut Cursor::new(&abelian)) {
            Err(TtIoError::LayerMismatch { found, expected }) => {
                assert_eq!((found, expected), (0, 1));
            }
            other => panic!("expected LayerMismatch, got {other:?}"),
        }
        let (_, su2) = su2_bytes();
        match read_tt::<f64, _>(&mut Cursor::new(&su2)) {
            Err(TtIoError::LayerMismatch { found, expected }) => {
                assert_eq!((found, expected), (1, 0));
            }
            other => panic!("expected LayerMismatch, got {other:?}"),
        }
        assert_eq!(abelian[OFF_LAYER], 0);
        assert_eq!(su2[OFF_LAYER], 1);
    }

    #[test]
    fn scalar_mismatch_is_rejected_both_ways() {
        let (_, real) = abelian_bytes();
        assert_eq!(real[OFF_SCALAR], 0);
        match read_tt::<C64, _>(&mut Cursor::new(&real)) {
            Err(TtIoError::ScalarMismatch { found, expected }) => {
                assert_eq!((found, expected), (0, 1));
            }
            other => panic!("expected ScalarMismatch reading f64 as C64, got {other:?}"),
        }
        let space = QnSpace::new(5, 2, 2);
        let grids = capped_grids(&space, 3);
        let z: TtVector<C64> = tt_randn(space, &grids, &mut rng(33));
        let mut cbuf = Vec::new();
        write_tt(&mut cbuf, &z).expect("write_tt");
        assert_eq!(cbuf[OFF_SCALAR], 1);
        match read_tt::<f64, _>(&mut Cursor::new(&cbuf)) {
            Err(TtIoError::ScalarMismatch { found, expected }) => {
                assert_eq!((found, expected), (1, 0));
            }
            other => panic!("expected ScalarMismatch reading C64 as f64, got {other:?}"),
        }
    }

    #[test]
    fn truncation_is_rejected_at_every_stage() {
        let (_, buf) = abelian_bytes();
        let cuts = [10, OFF_B + 4, OFF_B + 8 + 3, buf.len() / 2, buf.len() - 1];
        for cut in cuts {
            assert!(cut < buf.len(), "cut {cut} is not a truncation");
            match read_tt::<f64, _>(&mut Cursor::new(&buf[..cut])) {
                Err(TtIoError::Truncated { .. }) => {}
                other => panic!("expected Truncated at cut {cut}, got {other:?}"),
            }
        }
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let (_, mut buf) = abelian_bytes();
        buf.push(0);
        match read_tt::<f64, _>(&mut Cursor::new(&buf)) {
            Err(TtIoError::TrailingBytes { extra }) => assert_eq!(extra, 1),
            other => panic!("expected TrailingBytes, got {other:?}"),
        }
    }

    #[test]
    fn sector_count_mismatch_is_rejected() {
        let (x, buf) = abelian_bytes();
        let off = OFF_B + 8 + 2 + x.space.d + 8;
        let mut corrupt = buf.clone();
        let nsec = u64::from_le_bytes(corrupt[off..off + 8].try_into().unwrap());
        assert_eq!(
            nsec as usize,
            x.space.bond_sectors(0).len(),
            "test's offset arithmetic tracks the format"
        );
        corrupt[off..off + 8].copy_from_slice(&(nsec + 1).to_le_bytes());
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::SectorCountMismatch {
                bond,
                expected,
                found,
            }) => {
                assert_eq!(bond, 0);
                assert_eq!(expected as u64, nsec);
                assert_eq!(found, nsec + 1);
            }
            other => panic!("expected SectorCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn coefficient_count_mismatch_is_rejected() {
        let (x, buf) = abelian_bytes();
        let mut off = OFF_B + 8 + 2 + x.space.d + 8;
        for b in 0..=x.space.d {
            let n = x.space.bond_sectors(b).len();
            off += 8 + 8 * n;
        }
        let mut corrupt = buf.clone();
        let ncoeff = u64::from_le_bytes(corrupt[off..off + 8].try_into().unwrap());
        assert_eq!(
            ncoeff as usize,
            x.cores[0].num_coeffs(),
            "test's offset arithmetic tracks the format"
        );
        corrupt[off..off + 8].copy_from_slice(&(ncoeff - 1).to_le_bytes());
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::CoeffCountMismatch {
                site,
                expected,
                found,
            }) => {
                assert_eq!(site, 0);
                assert_eq!(expected as u64, ncoeff);
                assert_eq!(found, ncoeff - 1);
            }
            other => panic!("expected CoeffCountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn abelian_rank_exceeding_cap_is_rejected() {
        let (x, buf) = abelian_bytes();
        let nsec_off = OFF_B + 8 + 2 + x.space.d + 8;
        assert_eq!(
            x.space.bond_sectors(0).len(),
            1,
            "bond 0 has one admissible sector"
        );
        let rank_off = nsec_off + 8;
        let mut corrupt = buf.clone();
        corrupt[rank_off..rank_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("sanity cap"), "message: {m}"),
            other => panic!("expected InvalidHeader for an absurd rank, got {other:?}"),
        }
    }

    #[test]
    fn su2_rank_exceeding_cap_is_rejected() {
        let (x, buf) = su2_bytes();
        let nsec_off = OFF_B + 8 + 2 + x.space.d + 8;
        assert_eq!(
            x.space.bond_sectors(0).len(),
            1,
            "bond 0 has one admissible sector"
        );
        let rank_off = nsec_off + 8;
        let mut corrupt = buf.clone();
        corrupt[rank_off..rank_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        match read_su2_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("sanity cap"), "message: {m}"),
            other => panic!("expected InvalidHeader for an absurd rank, got {other:?}"),
        }
    }

    #[test]
    fn invalid_space_parameters_are_errors_not_panics() {
        let (x, buf) = abelian_bytes();
        let mut corrupt = buf.clone();
        corrupt[OFF_A..OFF_A + 8].copy_from_slice(&((x.space.d + 1) as u64).to_le_bytes());
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("Nup"), "message: {m}"),
            other => panic!("expected InvalidHeader for Nup > d, got {other:?}"),
        }

        let (_, sbuf) = su2_bytes();
        let mut corrupt = sbuf.clone();
        corrupt[OFF_B..OFF_B + 8].copy_from_slice(&1u64.to_le_bytes());
        match read_su2_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("even"), "message: {m}"),
            other => panic!("expected InvalidHeader for odd N-2S, got {other:?}"),
        }

        let mut corrupt = sbuf.clone();
        corrupt[OFF_B..OFF_B + 8].copy_from_slice(&100u64.to_le_bytes());
        match read_su2_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("2S"), "message: {m}"),
            other => panic!("expected InvalidHeader for oversized 2S, got {other:?}"),
        }

        let mut corrupt = buf.clone();
        corrupt[OFF_D..OFF_D + 8].copy_from_slice(&200u64.to_le_bytes());
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("site"), "message: {m}"),
            other => panic!("expected InvalidHeader for d > MAX_SITES, got {other:?}"),
        }
    }

    #[test]
    fn invalid_point_group_is_rejected() {
        let (pg, target, d, nelec, ms2) = fixture_point_group("ch2_triplet_c2v");
        let nup = ((nelec as i64 + ms2) / 2) as usize;
        let ndn = ((nelec as i64 - ms2) / 2) as usize;
        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
        let grids = capped_grids(&space, 2);
        let x: TtVector<f64> = tt_randn(space, &grids, &mut rng(41));
        let mut buf = Vec::new();
        write_tt(&mut buf, &x).expect("write_tt");
        let off_nbits = OFF_B + 8;

        let mut corrupt = buf.clone();
        corrupt[off_nbits + 2] = 7;
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::PointGroup(_)) => {}
            other => {
                panic!("expected PointGroup error for an out-of-range site irrep, got {other:?}")
            }
        }

        let mut corrupt = buf.clone();
        corrupt[off_nbits + 1] = 9;
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("target"), "message: {m}"),
            other => panic!("expected InvalidHeader for an out-of-range target, got {other:?}"),
        }

        let mut corrupt = buf.clone();
        corrupt[off_nbits] = 5;
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::PointGroup(_)) => {}
            other => panic!("expected PointGroup error for nbits > 3, got {other:?}"),
        }
    }

    #[test]
    fn invalid_ortho_is_rejected() {
        let (x, buf) = abelian_bytes();
        let off = OFF_B + 8 + 2 + x.space.d;
        for bad in [x.space.d as i64, -2i64] {
            let mut corrupt = buf.clone();
            corrupt[off..off + 8].copy_from_slice(&bad.to_le_bytes());
            match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
                Err(TtIoError::InvalidHeader(m)) => {
                    assert!(m.contains("orthogonality"), "message: {m}")
                }
                other => panic!("expected InvalidHeader for ortho = {bad}, got {other:?}"),
            }
        }
    }

    #[test]
    fn boundary_law_violation_is_an_error() {
        let (pg, target, d, nelec, ms2) = fixture_point_group("ch2_triplet_c2v");
        let nup = ((nelec as i64 + ms2) / 2) as usize;
        let ndn = ((nelec as i64 - ms2) / 2) as usize;
        let space = QnSpace::with_point_group(d, nup, ndn, pg, target);
        let grids = capped_grids(&space, 2);
        let x: TtVector<f64> = tt_randn(space, &grids, &mut rng(51));
        let mut buf = Vec::new();
        write_tt(&mut buf, &x).expect("write_tt");

        let off = OFF_B + 8 + 2 + x.space.d + 8;
        let mut off_d = off;
        for b in 0..x.space.d {
            off_d += 8 + 8 * x.space.bond_sectors(b).len();
        }
        let sectors_d = x.space.bond_sectors(x.space.d);
        assert!(
            sectors_d.len() > 1,
            "fixture must have more than one admissible sector at bond d for this test to bite"
        );
        let last = *sectors_d.last().unwrap();
        assert_ne!(
            last.irrep, x.space.target,
            "the corrupted sector must not be the legal boundary one"
        );
        let mut corrupt = buf.clone();
        let at = off_d + 8 + 8 * (sectors_d.len() - 1);
        assert_eq!(
            u64::from_le_bytes(corrupt[at..at + 8].try_into().unwrap()),
            0,
            "the illegal boundary sector starts at rank 0"
        );
        corrupt[at..at + 8].copy_from_slice(&3u64.to_le_bytes());
        match read_tt::<f64, _>(&mut Cursor::new(&corrupt)) {
            Err(TtIoError::InvalidHeader(m)) => assert!(m.contains("boundary"), "message: {m}"),
            other => panic!("expected InvalidHeader for a boundary-law violation, got {other:?}"),
        }
    }
}
