// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Electronic-structure integrals: the one type every reader produces, and the one builder that
//! turns it into an operator.

use std::collections::HashMap;
use std::path::Path;

use crate::coeff::{CoeffTensor, Sparse3Body, ThreeBody};
use crate::fcidump::{
    FcidumpError, PointGroupData, ThreeBodySymmetry, TwoBodySymmetry, check_nelec_ms2,
    parse_fcidump, read_entries, read_tcdump, screen_point_group_forbidden,
};
use crate::hamiltonian::SparseHamiltonian;
use crate::pointgroup::{PointGroup, PointGroupKind};
use crate::qn::QnSpace;
use crate::scalar::Scalar;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::Su2Space;

/// Electronic-structure integrals plus the header/metadata their source file carried.
pub struct Integrals<T: Scalar> {
    /// One-body integrals, rank-2 `norb × norb`.
    pub t: CoeffTensor<T>,
    /// Two-body integrals, rank-4 `norb⁴`, chemists' `(ij|kl)` convention.
    pub v: CoeffTensor<T>,
    /// Core/nuclear-repulsion energy: the FCIDUMP `0 0 0 0` sentinel. **Never folded into the operator** — callers add it to computed energies.
    pub e0: T,
    pub norb: usize,
    pub nelec: usize,
    /// `2·Sz` (the FCIDUMP `MS2` header field). A spin *projection*, not a total spin: the SU(2)
    /// layer's `2S` is a separate parameter that no dump carries.
    pub ms2: i64,
    /// Raw 1-based Molpro `ORBSYM` ids exactly as parsed, in site order; EMPTY when the source
    /// carried none. **No validation and no conversion happen here** — see
    /// [`Integrals::point_group`], which is where every id is checked and the `−1` dropped.
    pub orbsym: Vec<i64>,
    /// Raw `ISYM` as parsed (`None` when absent). `Some(0)` is the "totally symmetric" spelling
    /// used by e.g. `testdata/Be.ezfio.FCIDUMP`; the resolver special-cases it.
    pub isym: Option<i64>,
}

/// A parsed keyed `meta.txt` (`key value…` lines with `#` comments).
#[derive(Clone, Debug)]
pub struct FixtureMeta {
    pub norb: usize,
    pub nelec: usize,
    pub ms2: i64,
    /// The keyed `e_nuc`.
    pub e_core: f64,
    /// The keyed `e_hf`.
    pub e_hf: Option<f64>,
    /// The first present of the keyed `e_fci`, `e_ccsd_t`, `e_ccsd`.
    pub e_ref: Option<f64>,
    /// Keyed `orbsym_molpro`, raw 1-based Molpro ids. Empty when absent.
    pub orbsym: Vec<i64>,
    /// Keyed `isym`. `None` when absent.
    pub isym: Option<i64>,
    /// Every `key value…` pair, verbatim, values joined by single spaces. This is what a caller
    /// reaches for when it needs a key this struct does not model —
    /// `dump_file`, `geometry`, `basis`, the forbidden-entry census.
    pub keys: HashMap<String, String>,
}

fn meta_arg_err(msg: String) -> FcidumpError {
    FcidumpError::ArgumentError(msg)
}

fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    matches!(c.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_')
        && c.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}

fn collect_keyed(text: &str) -> HashMap<String, String> {
    let mut keys: HashMap<String, String> = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else { continue };
        if !is_ident(key) {
            continue;
        }
        let rest: Vec<&str> = it.collect();
        keys.insert(key.to_string(), rest.join(" "));
    }
    keys
}

/// Parse a keyed `meta.txt`.
pub fn parse_meta(text: &str) -> Result<FixtureMeta, FcidumpError> {
    let first = text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .ok_or_else(|| meta_arg_err("meta.txt is empty (or all comments)".into()))?;

    let tok: Vec<&str> = first.split_whitespace().collect();
    if tok.len() < 2 || !is_ident(tok[0]) {
        return Err(meta_arg_err(format!(
            "meta.txt is not the keyed dialect (`key value…`; the positional one-line dialect \
             is no longer read); first content line was: {first:?}"
        )));
    }

    let keys = collect_keyed(text);

    let num = |k: &str| keys.get(k).and_then(|s| s.parse::<f64>().ok());
    let need_usize = |k: &str| -> Result<usize, FcidumpError> {
        keys.get(k)
            .and_then(|s| s.parse::<usize>().ok())
            .ok_or_else(|| meta_arg_err(format!("keyed meta.txt is missing the `{k}` key")))
    };

    let orbsym = keys
        .get("orbsym_molpro")
        .map(|s| {
            s.split_whitespace()
                .map(|w| w.parse::<i64>())
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|e| {
            meta_arg_err(format!(
                "keyed meta.txt has a malformed `orbsym_molpro`: {e}"
            ))
        })?
        .unwrap_or_default();

    Ok(FixtureMeta {
        norb: need_usize("norb")?,
        nelec: need_usize("nelec")?,
        ms2: keys
            .get("ms2")
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0),
        e_core: num("e_nuc")
            .ok_or_else(|| meta_arg_err("keyed meta.txt is missing the `e_nuc` key".into()))?,
        e_hf: num("e_hf"),
        e_ref: num("e_fci")
            .or_else(|| num("e_ccsd_t"))
            .or_else(|| num("e_ccsd")),
        orbsym,
        isym: keys.get("isym").and_then(|s| s.parse::<i64>().ok()),
        keys,
    })
}

/// [`parse_meta`] on a file.
pub fn read_meta(path: &Path) -> Result<FixtureMeta, FcidumpError> {
    parse_meta(&std::fs::read_to_string(path)?)
}

/// How the optional three-body (transcorrelated) channel is read.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ThreeBodyOptions {
    /// Completion group for the TCDUMP's `w` tensor.
    pub symmetry: ThreeBodySymmetry,
    /// Sign applied to the whole tensor after reading. The transcorrelated convention is
    /// `w = −L`, hence `-1.0`; a sign of `1.0` skips the scaling pass entirely.
    pub sign: f64,
}

impl Default for ThreeBodyOptions {
    fn default() -> Self {
        ThreeBodyOptions {
            symmetry: ThreeBodySymmetry::Full48,
            sign: -1.0,
        }
    }
}

/// What to do with a dump's point-group header.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PointGroupPolicy {
    /// Build over a `|G| = 1` space; any `ORBSYM`/`ISYM` present is ignored. The default, and what
    /// every plain-FCIDUMP call site in the repository does.
    Ignore,
    /// Resolve `ORBSYM`/`ISYM` into a point group. `None` guesses the conventional group of the
    /// implied order (which affects irrep *names* only, never a number); `Some(k)` forces `k` and
    /// rejects any id outside `1..=k.order()`.
    Resolve(Option<PointGroupKind>),
}

/// Reader and builder policy, shared by both load paths.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct IntegralsOptions {
    /// Two-body permutation group, **completed** from the packed dump so that `v` satisfies it.
    pub symmetry: TwoBodySymmetry,
    /// Read a TCDUMP alongside the FCIDUMP.
    pub three_body: Option<ThreeBodyOptions>,
    pub point_group: PointGroupPolicy,
    /// Project out (or reject) integrals the resolved point group forbids, before construction.
    pub screen: bool,
    /// Magnitude below which a term is dropped from the operator's enumeration
    /// (`SparseHamiltonian::new`'s screen).
    pub eps: f64,
}

impl Default for IntegralsOptions {
    fn default() -> Self {
        IntegralsOptions {
            symmetry: TwoBodySymmetry::Eightfold,
            three_body: None,
            point_group: PointGroupPolicy::Ignore,
            screen: true,
            eps: f64::EPSILON,
        }
    }
}

/// What a path-level loader returns: the integrals, the optional three-body channel, the fixture
/// metadata when the format carried any, and the already-resolved point group and space.
pub struct IntegralsLoad<T: Scalar> {
    pub ints: Integrals<T>,
    /// The TCDUMP channel, when one was requested and read.
    pub w: Option<Sparse3Body<T>>,
    /// `Some` from [`load_fixture_dir`]; `None` from a direct [`load_fcidump`], which has no `meta.txt`.
    pub meta: Option<FixtureMeta>,
    pub pg: PointGroupData,
    pub space: QnSpace,
}

/// Read a fixture directory: its keyed `meta.txt` plus the FCIDUMP that meta's
/// `dump_file` key names, resolved relative to the directory.
pub fn load_fixture_dir<T: Scalar>(
    dir: &Path,
    opts: &IntegralsOptions,
) -> Result<IntegralsLoad<T>, FcidumpError> {
    let meta_path = dir.join("meta.txt");
    let meta = read_meta(&meta_path)
        .map_err(|e| FcidumpError::ArgumentError(format!("{}: {e}", meta_path.display())))?;
    let dump = meta.keys.get("dump_file").ok_or_else(|| {
        FcidumpError::ArgumentError(format!(
            "{} has no `dump_file` key, so the FCIDUMP it describes cannot be located",
            meta_path.display()
        ))
    })?;
    let mut load = load_fcidump::<T>(&dir.join(dump), None, opts)?;
    if meta.norb != load.ints.norb || meta.nelec != load.ints.nelec {
        return Err(FcidumpError::DimensionMismatch(format!(
            "{}: meta.txt says norb={} nelec={} but {dump} has norb={} nelec={} — regenerate",
            dir.display(),
            meta.norb,
            meta.nelec,
            load.ints.norb,
            load.ints.nelec
        )));
    }
    load.meta = Some(meta);
    Ok(load)
}

/// Write `ints` as a packed FCIDUMP that [`load_fcidump`] reads back **exactly**.
///
/// # Errors
///
/// - `ArgumentError` if `T` is complex with any non-zero imaginary part (FCIDUMP is a real
///   format), or if `t`/`v` are NOT eightfold-symmetric to `sym_atol` (absolute): the writer emits
///   one representative per orbit, so an asymmetric tensor would be silently symmetrized on
///   reload — refuse instead. Pass a transcorrelated (`Pairswap`) set through the deprecated
///   readers' own conventions, not through here.
/// - `Io` on any filesystem failure.
pub fn write_fcidump<T: Scalar>(
    path: &Path,
    ints: &Integrals<T>,
    sym_atol: f64,
) -> Result<(), FcidumpError> {
    use std::io::Write;
    let d = ints.norb;
    if ints.t.d != d || ints.t.rank != 2 || ints.v.d != d || ints.v.rank != 4 {
        return Err(FcidumpError::DimensionMismatch(format!(
            "write_fcidump: t is rank {} over d = {}, v is rank {} over d = {}, norb = {d}",
            ints.t.rank, ints.t.d, ints.v.rank, ints.v.d
        )));
    }
    let re = |x: T, what: &str| -> Result<f64, FcidumpError> {
        if T::IS_COMPLEX && x.im_s() != 0.0 {
            return Err(FcidumpError::ArgumentError(format!(
                "write_fcidump: {what} has a non-zero imaginary part {:e}; FCIDUMP is a real format",
                x.im_s()
            )));
        }
        Ok(x.re_s())
    };
    let mut out = String::new();
    let orbsym: Vec<String> = if ints.orbsym.is_empty() {
        vec!["1".to_string(); d]
    } else {
        ints.orbsym.iter().map(|s| s.to_string()).collect()
    };
    out.push_str(&format!(
        " &FCI NORB= {d},NELEC= {},MS2={},\n  ORBSYM={}\n  ISYM={},\n &END\n",
        ints.nelec,
        ints.ms2,
        orbsym.join(","),
        ints.isym.unwrap_or(1)
    ));
    let pair = |i: usize, j: usize| i * (i + 1) / 2 + j;
    for i in 0..d {
        for j in 0..=i {
            for k in 0..d {
                for l in 0..=k {
                    if pair(i, j) < pair(k, l) {
                        continue;
                    }
                    let x = ints.v.get(&[i, j, k, l]);
                    for im in [
                        [j, i, k, l],
                        [i, j, l, k],
                        [j, i, l, k],
                        [k, l, i, j],
                        [l, k, i, j],
                        [k, l, j, i],
                        [l, k, j, i],
                    ] {
                        let y = ints.v.get(&im);
                        if (y - x).abs_s() > sym_atol {
                            return Err(FcidumpError::ArgumentError(format!(
                                "write_fcidump: v is not eightfold-symmetric: v[{i},{j},{k},{l}] = \
                                 {:e} but v{im:?} = {:e} (|Δ| > {sym_atol:e})",
                                x.re_s(),
                                y.re_s()
                            )));
                        }
                    }
                    let xr = re(x, &format!("v[{i},{j},{k},{l}]"))?;
                    if xr != 0.0 {
                        out.push_str(&format!("{xr:e} {} {} {} {}\n", i + 1, j + 1, k + 1, l + 1));
                    }
                }
            }
        }
    }
    for i in 0..d {
        for j in 0..=i {
            let x = ints.t.get(&[i, j]);
            let y = ints.t.get(&[j, i]);
            if (y - x).abs_s() > sym_atol {
                return Err(FcidumpError::ArgumentError(format!(
                    "write_fcidump: t is not symmetric: t[{i},{j}] = {:e} but t[{j},{i}] = {:e} \
                     (|Δ| > {sym_atol:e})",
                    x.re_s(),
                    y.re_s()
                )));
            }
            let xr = re(x, &format!("t[{i},{j}]"))?;
            if xr != 0.0 {
                out.push_str(&format!("{xr:e} {} {} 0 0\n", i + 1, j + 1));
            }
        }
    }
    out.push_str(&format!("{:e} 0 0 0 0\n", re(ints.e0, "e0")?));
    let mut f = std::fs::File::create(path)?;
    f.write_all(out.as_bytes())?;
    Ok(())
}

/// Write a complete fixture directory — `dump_name` via [`write_fcidump`] plus the keyed
/// `meta.txt` that [`load_fixture_dir`] needs (`norb`, `nelec`, `ms2`, `isym`, `e_nuc`,
/// `dump_file`, `dump_bytes`, `dump_lines`, `write_tol 0`), followed by `extra` keys verbatim.
pub fn write_fixture_dir<T: Scalar>(
    dir: &Path,
    dump_name: &str,
    ints: &Integrals<T>,
    sym_atol: f64,
    extra: &[(&str, &str)],
) -> Result<(), FcidumpError> {
    std::fs::create_dir_all(dir)?;
    let dump_path = dir.join(dump_name);
    write_fcidump(&dump_path, ints, sym_atol)?;
    let bytes = std::fs::metadata(&dump_path)?.len();
    let lines = std::fs::read_to_string(&dump_path)?.lines().count();
    let mut meta = String::new();
    meta.push_str(&format!(
        "# {}/meta.txt -- written by qntt::integrals::write_fixture_dir; keyed dialect.\n",
        dir.file_name()
            .map(|s| s.to_string_lossy())
            .unwrap_or_default()
    ));
    meta.push_str(&format!("norb            {}\n", ints.norb));
    meta.push_str(&format!("nelec           {}\n", ints.nelec));
    meta.push_str(&format!("ms2             {}\n", ints.ms2));
    meta.push_str(&format!("isym            {}\n", ints.isym.unwrap_or(1)));
    meta.push_str(&format!("e_nuc           {:e}\n", ints.e0.re_s()));
    meta.push_str("write_tol       0\n");
    meta.push_str(&format!("dump_file       {dump_name}\n"));
    meta.push_str(&format!("dump_bytes      {bytes}\n"));
    meta.push_str(&format!("dump_lines      {lines}\n"));
    for (k, v) in extra {
        if !is_ident(k) {
            return Err(FcidumpError::ArgumentError(format!(
                "write_fixture_dir: meta key `{k}` is not a bare identifier"
            )));
        }
        if v.contains('\n') {
            return Err(FcidumpError::ArgumentError(format!(
                "write_fixture_dir: meta value for `{k}` spans several lines"
            )));
        }
        meta.push_str(&format!("{k:<15} {v}\n"));
    }
    std::fs::write(dir.join("meta.txt"), meta)?;
    Ok(())
}

/// Read an FCIDUMP (and optionally a TCDUMP), apply `opts`, and resolve the space.
pub fn load_fcidump<T: Scalar>(
    fcidump: &Path,
    tcdump: Option<&Path>,
    opts: &IntegralsOptions,
) -> Result<IntegralsLoad<T>, FcidumpError> {
    let mut ints: Integrals<T> = parse_fcidump(&std::fs::read_to_string(fcidump)?, opts.symmetry)?;

    let w = match (tcdump, opts.three_body) {
        (Some(tc), tb) => {
            let tb = tb.unwrap_or_default();
            let mut wten = read_tcdump::<T>(tc, ints.norb, tb.symmetry)?;
            if wten.d != ints.norb {
                return Err(FcidumpError::DimensionMismatch(format!(
                    "TCDUMP orbital count {} ≠ FCIDUMP NORB {}",
                    wten.d, ints.norb
                )));
            }
            if tb.sign != 1.0 {
                wten.scale_all(T::from_f64(tb.sign));
            }
            Some(wten)
        }
        (None, _) => None,
    };

    let (space, pg) = ints.resolve(opts)?;
    Ok(IntegralsLoad {
        ints,
        w,
        meta: None,
        pg,
        space,
    })
}

impl<T: Scalar> Integrals<T> {
    /// The core energy. Alias for the [`e0`](Self::e0) field, spelled the way callers think of it.
    pub fn e_core(&self) -> T {
        self.e0
    }

    /// Apply `opts`' point-group policy: resolve the group, screen the forbidden entries **in
    /// place** when asked, and build the space.
    pub fn resolve(
        &mut self,
        opts: &IntegralsOptions,
    ) -> Result<(QnSpace, PointGroupData), FcidumpError> {
        check_nelec_ms2(self.nelec, self.ms2)?;
        match opts.point_group {
            PointGroupPolicy::Ignore => Ok((
                QnSpace::from_n_two_sz(self.norb, self.nelec, self.ms2),
                PointGroupData::TRIVIAL,
            )),
            PointGroupPolicy::Resolve(kind) => {
                let pg = self.point_group(kind)?;
                if opts.screen {
                    screen_point_group_forbidden(&mut self.t, &mut self.v, pg.pg)?;
                }
                let space = QnSpace::from_n_two_sz_with_point_group(
                    self.norb, self.nelec, self.ms2, pg.pg, pg.target,
                );
                Ok((space, pg))
            }
        }
    }

    /// [`Self::resolve`] followed by `SparseHamiltonian::new` — **the single builder both load
    /// paths go through**, and therefore the structural half of the equivalence claim: two files
    /// describing the same system cannot diverge in assembly, only in what they parsed.
    pub fn hamiltonian(
        &mut self,
        w: Option<&dyn ThreeBody<T>>,
        opts: &IntegralsOptions,
    ) -> Result<(SparseHamiltonian<T>, QnSpace, PointGroupData), FcidumpError> {
        let (space, pg) = self.resolve(opts)?;
        let h = SparseHamiltonian::new(space, &self.t, &self.v, w, opts.eps);
        Ok((h, space, pg))
    }

    /// The SU(2) twin of [`Self::hamiltonian`]. `two_s` is the total `2S`, which **no dump
    /// carries** (`MS2` is `2·Sz`, a projection); `None` defaults to `|MS2|`, the minimal total
    /// spin consistent with that projection. There is no three-body channel on this layer.
    pub fn su2_hamiltonian(
        &mut self,
        two_s: Option<usize>,
        opts: &IntegralsOptions,
    ) -> Result<(Su2SparseHamiltonian<T>, Su2Space, PointGroupData), FcidumpError> {
        check_nelec_ms2(self.nelec, self.ms2)?;
        let pg = match opts.point_group {
            PointGroupPolicy::Ignore => PointGroupData::TRIVIAL,
            PointGroupPolicy::Resolve(kind) => self.point_group(kind)?,
        };
        let space = crate::su2::fcidump::su2_space(self, &pg, two_s)?;
        if opts.screen && matches!(opts.point_group, PointGroupPolicy::Resolve(_)) {
            screen_point_group_forbidden(&mut self.t, &mut self.v, pg.pg)?;
        }
        let h = Su2SparseHamiltonian::new(space, &self.t, &self.v, None, opts.eps);
        Ok((h, space, pg))
    }
}

impl<T: Scalar> IntegralsLoad<T> {
    /// Assemble the operator this load describes, three-body channel included.
    pub fn hamiltonian(&self, eps: f64) -> SparseHamiltonian<T> {
        let w: Option<&dyn ThreeBody<T>> = self.w.as_ref().map(|w| w as &dyn ThreeBody<T>);
        SparseHamiltonian::new(self.space, &self.ints.t, &self.ints.v, w, eps)
    }
}

impl<T: Scalar> SparseHamiltonian<T> {
    /// Read an FCIDUMP (and optionally a TCDUMP) and compile the operator it describes:
    /// [`load_fcidump`] followed by [`IntegralsLoad::hamiltonian`].
    pub fn from_fcidump(
        fcidump: &Path,
        tcdump: Option<&Path>,
        opts: &IntegralsOptions,
    ) -> Result<(Self, IntegralsLoad<T>), FcidumpError> {
        let load = load_fcidump::<T>(fcidump, tcdump, opts)?;
        let h = load.hamiltonian(opts.eps);
        Ok((h, load))
    }

    /// Read a fixture directory and compile the operator it describes: [`load_fixture_dir`]
    /// followed by [`IntegralsLoad::hamiltonian`].
    pub fn from_fixture_dir(
        dir: &Path,
        opts: &IntegralsOptions,
    ) -> Result<(Self, IntegralsLoad<T>), FcidumpError> {
        let load = load_fixture_dir::<T>(dir, opts)?;
        let h = load.hamiltonian(opts.eps);
        Ok((h, load))
    }
}

const PROPERTY_HERM_ATOL: f64 = 100.0 * f64::EPSILON;

/// A one-body property (observable) matrix with the name it was read under.
#[derive(Clone, Debug)]
pub struct PropertyOperator<T: Scalar> {
    /// The component's name, as the file carried it (`dipole_z`, `mulliken_o`, …). Reproduced
    /// verbatim in every diagnostic this module raises, which is the whole reason it is stored:
    /// a response run threads several components, and a panic that only says "cross-irrep
    /// component" does not say *which* one.
    pub label: String,
    /// The one-body matrix, rank-2 `norb × norb`.
    pub m: CoeffTensor<T>,
}

impl<T: Scalar> PropertyOperator<T> {
    /// The validating constructor: shape-check `m` and run [`assert_property_hermitian`].
    pub fn new(label: impl Into<String>, m: CoeffTensor<T>) -> Self {
        let label = label.into();
        assert_eq!(m.rank, 2, "property {label:?} matrix must be rank-2");
        assert_property_hermitian(&m, &label);
        PropertyOperator { label, m }
    }
}

/// Reject `m` unless it is a full symmetric `d×d` matrix — the hermiticity twin of
/// `crate::rotation::assert_antisym_with_atol`, with the same two clauses inverted from
/// `κ = −κᵀ` to `m = mᵀ`, and **both are load-bearing**:
pub fn assert_property_hermitian<T: Scalar>(m: &CoeffTensor<T>, label: &str) {
    if let Some(msg) = property_hermiticity_error(m, label) {
        panic!("{msg}");
    }
}

/// [`assert_property_hermitian`]'s two clauses as a PREDICATE, for a boundary that must REFUSE
/// rather than panic — the C ABI, where a caller error must cross as `QNTT_INVALID_ARG` and never
/// as `QNTT_PANIC` (`qntt_f64_sector_alpha`, `qntt_f64_sector_transition_moments`).
pub fn property_hermiticity_error<T: Scalar>(m: &CoeffTensor<T>, label: &str) -> Option<String> {
    let d = m.d;

    let mut lo = false;
    let mut up = false;
    for p in 1..d {
        for q in 0..p {
            lo |= m.get(&[p, q]).abs_s() != 0.0;
            up |= m.get(&[q, p]).abs_s() != 0.0;
        }
    }
    if lo ^ up {
        return Some(format!(
            "property {label:?} has exactly one nonzero strict triangle — pass the FULL symmetric \
             matrix (m[q,p] = m[p,q]), not the p>q half"
        ));
    }

    let mut maxabs = 0.0f64;
    let mut maxasym = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            maxabs = maxabs.max(m.get(&[p, q]).abs_s());
        }
    }
    for p in 0..d {
        for q in p..d {
            maxasym = maxasym.max((m.get(&[p, q]) - m.get(&[q, p])).abs_s());
        }
    }
    let tol = PROPERTY_HERM_ATOL.max(100.0 * f64::EPSILON * maxabs);
    if maxasym > tol || maxasym.is_nan() {
        return Some(format!(
            "property {label:?} is not symmetric: max|m - mᵀ| = {maxasym:e} > tol {tol:e} \
             (max|m| = {maxabs:e})"
        ));
    }
    None
}

/// Parse a property-component text: keyed header lines followed by `value p q` records.
pub fn parse_property<T: Scalar>(
    text: &str,
    norb: usize,
) -> Result<PropertyOperator<T>, FcidumpError> {
    let keys = collect_keyed(text);
    let label = keys.get("label").cloned().ok_or_else(|| {
        meta_arg_err(
            "property text is missing the `label` key (a `label <name>` header line is required, \
             so every diagnostic can name the component)"
                .into(),
        )
    })?;

    let lines: Vec<&str> = text.lines().collect();
    let mut m = CoeffTensor::<T>::zeros(norb, 2);
    for &(x, [p, q]) in &read_entries::<2>(&lines) {
        let (p, q) = (p as usize, q as usize);
        if p == 0 || q == 0 {
            return Err(meta_arg_err(format!(
                "malformed property {label:?} record (zero orbital index): {x}  {p}  {q}"
            )));
        }
        if p > norb || q > norb {
            return Err(FcidumpError::DimensionMismatch(format!(
                "property {label:?} record {x}  {p}  {q} indexes past norb={norb}"
            )));
        }
        m.set(&[p - 1, q - 1], T::from_f64(x));
    }

    Ok(PropertyOperator::new(label, m))
}

/// [`parse_property`] on a file — `read_to_string` plus that call and nothing else.
pub fn load_property<T: Scalar>(
    path: &Path,
    norb: usize,
) -> Result<PropertyOperator<T>, FcidumpError> {
    parse_property(&std::fs::read_to_string(path)?, norb)
}

/// [`load_property`] plus the **phase-fingerprint pairing check** against the fixture's meta.
pub fn load_property_matching<T: Scalar>(
    path: &Path,
    norb: usize,
    meta: &FixtureMeta,
) -> Result<PropertyOperator<T>, FcidumpError> {
    let Some(want) = meta.keys.get("phase_fingerprint") else {
        return load_property(path, norb);
    };
    let text = std::fs::read_to_string(path)?;
    let keys = collect_keyed(&text);
    match keys.get("phase_fingerprint") {
        Some(got) if got == want => parse_property(&text, norb),
        Some(got) => Err(meta_arg_err(format!(
            "property file {} carries phase_fingerprint {got} but the fixture meta demands \
             {want}: the two were transformed by DIFFERENT MO matrices (column phases are not \
             reproducible across generator runs), and pairing them would be a silent O(1) basis \
             error — regenerate the fixture directory atomically",
            path.display()
        ))),
        None => Err(meta_arg_err(format!(
            "property file {} carries no phase_fingerprint but the fixture meta demands {want}: \
             a fingerprinted fixture pairs only with property files written by the same \
             generator process — regenerate the fixture directory atomically",
            path.display()
        ))),
    }
}

fn assert_property_representable<T: Scalar>(p: &PropertyOperator<T>, d: usize, pg: PointGroup) {
    if pg.is_trivial() {
        return;
    }
    for i in 0..d {
        for j in 0..d {
            if pg.irrep_of(i) != pg.irrep_of(j) && p.m.get(&[i, j]).abs_s() != 0.0 {
                panic!(
                    "property {:?} has a nonzero component m[{i},{j}] = {:e} across irreps \
                     {:?} != {:?}: the one-body term a†_{i} a_{j} carries the flux Γ_{i} ⊗ Γ_{j}, \
                     which maps the space's target irrep to target ⊗ Γ_{i} ⊗ Γ_{j} — a \
                     single-target space cannot represent it. A cross-irrep property needs the \
                     cross-target operator map (later phase), not a screened matrix.",
                    p.label,
                    p.m.get(&[i, j]).abs_s(),
                    pg.irrep_of(i),
                    pg.irrep_of(j),
                );
            }
        }
    }
}

fn check_property<T: Scalar>(p: &PropertyOperator<T>, d: usize, pg: PointGroup) {
    assert!(
        p.m.rank == 2 && p.m.d == d,
        "property {:?} matrix must be rank-2 over d={d} sites (got rank {}, d {})",
        p.label,
        p.m.rank,
        p.m.d
    );
    assert_property_hermitian(&p.m, &p.label);
    assert_property_representable(p, d, pg);
}

/// The one-body observable `M̂ = Σ_{p,q} m[p,q]·E_pq` assembled as an ordinary term automaton.
pub fn property_hamiltonian<T: Scalar>(
    space: QnSpace,
    p: &PropertyOperator<T>,
) -> SparseHamiltonian<T> {
    check_property(p, space.d, space.pg);
    SparseHamiltonian::new(space, &p.m, &CoeffTensor::<T>::zeros(space.d, 4), None, 0.0)
}

/// The SU(2) twin of [`property_hamiltonian`] — same reduction, same `eps = 0.0` reasoning, same
/// named cross-irrep refusal, over a spin-adapted space.
pub fn su2_property_hamiltonian<T: Scalar>(
    space: Su2Space,
    p: &PropertyOperator<T>,
) -> Su2SparseHamiltonian<T> {
    check_property(p, space.d, space.pg);
    Su2SparseHamiltonian::new(space, &p.m, &CoeffTensor::<T>::zeros(space.d, 4), None, 0.0)
}

#[cfg(test)]
mod tjq_tests {
    use super::{assert_property_hermitian, parse_property, property_hermiticity_error};
    use crate::CoeffTensor;
    use crate::fcidump::FcidumpError;

    #[test]
    fn tjq_property_zero_index_rejected() {
        let text = "label bad\n1.0 0 1\n";
        let err = parse_property::<f64>(text, 4).expect_err("a raw 0 index must be rejected");
        match err {
            FcidumpError::ArgumentError(msg) => {
                assert!(
                    !msg.is_empty(),
                    "ArgumentError should carry a non-empty message"
                );
            }
            other => panic!("expected FcidumpError::ArgumentError, got {other:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "symmetric")]
    fn tjq_hermiticity_rejected_asymmetric() {
        let text = "label bad_prop\n1.0 2 1\n2.0 2 2\n";
        let _ = parse_property::<f64>(text, 4);
    }

    #[test]
    #[should_panic(expected = "one nonzero")]
    fn tjq_hermiticity_rejected_half_matrix_any_magnitude() {
        let text = "label half\n1e-20 2 1\n";
        let _ = parse_property::<f64>(text, 4);
    }

    #[test]
    fn tjq_hermiticity_accepted_with_roundoff() {
        let text = "label ok\n1.0 2 1\n1.0000000000000002 1 2\n";
        let p = parse_property::<f64>(text, 4)
            .expect("symmetric-to-roundoff must be accepted (the absolute floor clause)");
        assert_eq!(
            p.m.get(&[1, 0]),
            1.0,
            "record `1.0 2 1` -> m[1][0], bitwise"
        );
        assert_eq!(
            p.m.get(&[0, 1]),
            1.0000000000000002,
            "record `1.0000000000000002 1 2` -> m[0][1], bitwise"
        );
    }

    fn m4(cells: &[(usize, usize, f64)]) -> CoeffTensor<f64> {
        let mut m = CoeffTensor::<f64>::zeros(4, 2);
        for &(p, q, v) in cells {
            m.set(&[p, q], v);
        }
        m
    }

    #[test]
    fn property_hermiticity_error_and_the_assert_agree() {
        let cases: [(&str, CoeffTensor<f64>); 4] = [
            ("symmetric", m4(&[(1, 2, 0.75), (2, 1, 0.75), (0, 0, -0.3)])),
            ("half_triangle", m4(&[(1, 0, 1e-20)])),
            ("two_sided", m4(&[(1, 0, 1.0), (0, 1, 0.5)])),
            (
                "roundoff",
                m4(&[(1, 0, 1.0), (0, 1, 1.000_000_000_000_000_2)]),
            ),
        ];
        for (label, m) in &cases {
            let predicate = property_hermiticity_error(m, label);
            let panicked = std::panic::catch_unwind(|| assert_property_hermitian(m, label))
                .err()
                .map(|e| {
                    if let Some(s) = e.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = e.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else {
                        panic!("{label}: an assert payload that is neither String nor &str")
                    }
                });
            assert_eq!(
                predicate, panicked,
                "{label}: the predicate and the assert must agree, message included"
            );
        }
        assert!(property_hermiticity_error(&cases[0].1, "symmetric").is_none());
        assert!(property_hermiticity_error(&cases[3].1, "roundoff").is_none());
        let c1 = property_hermiticity_error(&cases[1].1, "half_triangle")
            .expect("clause 1 must reject a lone strict triangle at ANY magnitude");
        assert!(c1.contains("one nonzero"), "{c1:?}");
        assert!(
            c1.contains("half_triangle"),
            "the message must name the component: {c1:?}"
        );
        let c2 = property_hermiticity_error(&cases[2].1, "two_sided")
            .expect("clause 2 must reject an O(1) two-sided asymmetry");
        assert!(c2.contains("is not symmetric"), "{c2:?}");
        assert!(
            c2.contains("two_sided"),
            "the message must name the component: {c2:?}"
        );
    }

    #[test]
    fn property_hermiticity_error_is_none_on_the_roundoff_fixture() {
        let m = m4(&[(1, 0, 1.0), (0, 1, 1.000_000_000_000_000_2)]);
        assert_eq!(property_hermiticity_error(&m, "ok"), None);
        let bad = m4(&[(1, 0, 1.0), (0, 1, 1.000_000_000_000_1)]);
        assert!(
            property_hermiticity_error(&bad, "too_asym").is_some(),
            "a 1e-13 asymmetry at max|m| = 1 is above the floor and must be refused"
        );
    }
}
