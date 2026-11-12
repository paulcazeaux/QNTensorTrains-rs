// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Reduced density matrices over tensor-train states.

use faer::Mat;

use crate::coeff::CoeffTensor;
use crate::core::SparseCore;
use crate::core::factor::{
    Sketch, adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
};
use crate::elementaryops::{SiteOp, State, apply_site_op};
use crate::hamiltonian::SparseHamiltonian;
use crate::hamiltonian::matvec::{h_sketch_mul, sketch_h_mul};
use crate::ops::dot;
use crate::orbital::{Orbital, Spin, fermionic_sign};
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, RankGrid, Sector, SectorGrid};
use crate::rotation::e_pq;
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// Per-entry reference implementation of the spin-summed transition 1-RDM:
/// `γ[p,q] = dot(x, E_pq·y)`, one exact one-body apply and one full inner product per entry.
pub fn transition_1rdm_bruteforce<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>) -> CoeffTensor<T> {
    let space = x.space;
    assert_eq!(
        space, y.space,
        "transition_1rdm: bra and ket must share a QnSpace"
    );
    let d = space.d;
    let pg = space.pg;
    let mut gamma = CoeffTensor::<T>::zeros(d, 2);
    for q in 0..d {
        for p in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                continue;
            }
            gamma.set(&[p, q], dot(x, &e_pq(y, p, q, T::one())));
        }
    }
    gamma
}

#[inline]
fn op_state(spin: Spin, f: i64, nl: i64, nr: i64, delta: Irrep) -> State {
    match spin {
        Spin::Up => State {
            fu: f,
            nlu: nl,
            nru: nr,
            fd: 0,
            nld: 0,
            nrd: 0,
            delta,
        },
        Spin::Dn => State {
            fu: 0,
            nlu: 0,
            nru: 0,
            fd: f,
            nld: nl,
            nrd: nr,
            delta,
        },
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn op_core<T: Scalar>(
    c: &SparseCore<T>,
    caps: Option<&[RankGrid]>,
    spin: Spin,
    op: SiteOp,
    f: i64,
    nl: i64,
    nr: i64,
    delta: Irrep,
) -> SparseCore<T> {
    let (up, dn) = match spin {
        Spin::Up => (op, SiteOp::Id),
        Spin::Dn => (SiteOp::Id, op),
    };
    apply_site_op(c, caps, up, dn, op_state(spin, f, nl, nr, delta)).0
}

#[inline]
pub(crate) fn pair_sign(p: usize, q: usize, spin: Spin) -> f64 {
    fermionic_sign(&[Orbital::new(p, spin), Orbital::new(q, spin)]) as f64
}

pub(crate) fn active_spins(space: &QnSpace) -> Vec<Spin> {
    let mut spins = Vec::with_capacity(2);
    if space.nup >= 1 {
        spins.push(Spin::Up);
    }
    if space.ndn >= 1 {
        spins.push(Spin::Dn);
    }
    spins
}

pub(crate) fn assert_unit_boundaries<T: Scalar>(v: &TtVector<T>, who: &str) {
    let space = v.space;
    assert_eq!(
        v.rank(0, Sector::new(0, 0)),
        1,
        "transition_1rdm needs unit rank at bond 0 ({who})"
    );
    assert_eq!(
        v.rank(
            space.d,
            Sector::with_irrep(space.nup, space.ndn, space.target)
        ),
        1,
        "transition_1rdm needs unit rank at bond d ({who})"
    );
}

pub(crate) fn restrict_cols<T: Scalar>(l: &Sketch<T>, rr: &RankGrid) -> Sketch<T> {
    let blocks = SectorGrid::from_fn(&l.space, |s| {
        l.block(s).map(|b| {
            if rr[s] == b.ncols() {
                b.to_owned()
            } else {
                debug_assert_eq!(
                    rr[s], 0,
                    "operator copy-or-zero invariant violated at sector {s:?}"
                );
                Mat::<T>::zeros(b.nrows(), 0)
            }
        })
    });
    Sketch::new(l.space, l.k, blocks)
}

pub(crate) fn restrict_rows<T: Scalar>(l: &Sketch<T>, rr: &RankGrid) -> Sketch<T> {
    let blocks = SectorGrid::from_fn(&l.space, |s| {
        l.block(s).map(|b| {
            if rr[s] == b.nrows() {
                b.to_owned()
            } else {
                debug_assert_eq!(
                    rr[s], 0,
                    "operator copy-or-zero invariant violated at sector {s:?}"
                );
                Mat::<T>::zeros(0, b.ncols())
            }
        })
    });
    Sketch::new(l.space, l.k, blocks)
}

pub(crate) fn close_trace<T: Scalar>(lc: &Sketch<T>, r: &Sketch<T>) -> T {
    debug_assert_eq!(lc.k, r.k, "close_trace: bond mismatch");
    let mut acc = T::zero();
    for s in lc.sectors() {
        let (Some(a), Some(b)) = (lc.block(s), r.block(s)) else {
            continue;
        };
        if a.nrows() == 0 || a.ncols() == 0 || b.nrows() == 0 || b.ncols() == 0 {
            continue;
        }
        debug_assert_eq!(a.ncols(), b.nrows(), "close_trace: shape mismatch at {s:?}");
        debug_assert_eq!(a.nrows(), b.ncols(), "close_trace: shape mismatch at {s:?}");
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                acc += *a.get(i, j) * *b.get(j, i);
            }
        }
    }
    acc
}

pub(crate) struct IdCache<T: Scalar> {
    cores: Vec<Option<SparseCore<T>>>,
}

impl<T: Scalar> IdCache<T> {
    pub(crate) fn new(order: usize) -> Self {
        IdCache {
            cores: (0..order).map(|_| None).collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get(
        &mut self,
        c: &SparseCore<T>,
        caps: Option<&[RankGrid]>,
        spin: Spin,
        f: i64,
        nl: i64,
        nr: i64,
        delta: Irrep,
    ) -> &SparseCore<T> {
        let i = usize::from(delta.index());
        if self.cores[i].is_none() {
            self.cores[i] = Some(op_core(c, caps, spin, SiteOp::Id, f, nl, nr, delta));
        }
        self.cores[i].as_ref().expect("just interned")
    }
}

/// Spin-summed transition 1-RDM `γ[p,q] = ⟨x| E_pq |y⟩` for **all** `(p,q)` in one left-to-right
/// pass — the production path behind [`transition_1rdm_bruteforce`]'s signature.
pub fn transition_1rdm<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>) -> CoeffTensor<T> {
    let space = x.space;
    assert_eq!(
        space, y.space,
        "transition_1rdm: bra and ket must share a QnSpace"
    );
    let d = space.d;
    let pg = space.pg;
    let mut gamma = CoeffTensor::<T>::zeros(d, 2);
    if d == 0 {
        return gamma;
    }
    assert_unit_boundaries(x, "x");
    assert_unit_boundaries(y, "y");

    let spins = active_spins(&space);
    if spins.is_empty() {
        return gamma;
    }

    let caps = (!pg.is_trivial()).then(|| crate::qn::max_ranks(&space));
    let caps = caps.as_deref();

    let mut l: Vec<Sketch<T>> = Vec::with_capacity(d);
    l.push(Sketch::identity(space, 0, 1));
    for k in 0..d - 1 {
        let t = mul_sketch_left(&l[k], y.core(k));
        l.push(adjoint_core_times_core(x.core(k), &t));
    }
    let mut r: Vec<Option<Sketch<T>>> = vec![None; d + 1];
    r[d] = Some(Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let t = mul_sketch_right(
            y.core(k),
            r[k + 1].as_ref().expect("right env built rightward"),
        );
        r[k] = Some(core_times_adjoint_core(&t, x.core(k)));
    }

    let mut per_spin: Vec<Vec<T>> = Vec::with_capacity(spins.len());
    for &spin in &spins {
        let mut gs = vec![T::zero(); d * d];
        let mut live_cre: Vec<(usize, Sketch<T>)> = Vec::new();
        let mut live_ann: Vec<(usize, Sketch<T>)> = Vec::new();

        for k in 0..d {
            let gk = pg.irrep_of(k);
            let xk = x.core(k);
            let yk = y.core(k);
            let rk1 = r[k + 1].as_ref().expect("right env present at bond k+1");

            if live_cre.iter().any(|&(a, _)| pg.irrep_of(a) == gk) {
                let ann_close = op_core(yk, caps, spin, SiteOp::A, 1, 1, 0, gk);
                for (a, dch) in live_cre.iter() {
                    if pg.irrep_of(*a) != gk {
                        continue;
                    }
                    let closed = adjoint_core_times_core(xk, &mul_sketch_left(dch, &ann_close));
                    let e = T::from_f64(pair_sign(*a, k, spin));
                    gs[*a * d + k] += e * close_trace(&closed, rk1);
                }
            }
            if live_ann.iter().any(|&(a, _)| pg.irrep_of(a) == gk) {
                let cre_close = op_core(yk, caps, spin, SiteOp::Adag, -1, 0, 1, gk);
                for (a, dch) in live_ann.iter() {
                    if pg.irrep_of(*a) != gk {
                        continue;
                    }
                    let closed = adjoint_core_times_core(xk, &mul_sketch_left(dch, &cre_close));
                    let e = T::from_f64(pair_sign(k, *a, spin));
                    gs[k * d + *a] += e * close_trace(&closed, rk1);
                }
            }

            let n_diag = op_core(yk, caps, spin, SiteOp::N, 0, 0, 1, Irrep::IDENTITY);
            let glk = restrict_cols(&l[k], &n_diag.row_ranks);
            let closed = adjoint_core_times_core(xk, &mul_sketch_left(&glk, &n_diag));
            let e = T::from_f64(pair_sign(k, k, spin));
            gs[k * d + k] += e * close_trace(&closed, rk1);

            if k + 1 >= d {
                continue;
            }

            let mut id_plus = IdCache::<T>::new(usize::from(pg.order()));
            for (a, dch) in live_cre.iter_mut() {
                let idc = id_plus.get(yk, caps, spin, 1, 1, 0, pg.irrep_of(*a));
                *dch = adjoint_core_times_core(xk, &mul_sketch_left(dch, idc));
            }
            let mut id_minus = IdCache::<T>::new(usize::from(pg.order()));
            for (a, dch) in live_ann.iter_mut() {
                let idc = id_minus.get(yk, caps, spin, -1, 0, 1, pg.irrep_of(*a));
                *dch = adjoint_core_times_core(xk, &mul_sketch_left(dch, idc));
            }

            let cre_start = op_core(yk, caps, spin, SiteOp::Adag, 0, 0, 1, Irrep::IDENTITY);
            live_cre.push((
                k,
                adjoint_core_times_core(xk, &mul_sketch_left(&glk, &cre_start)),
            ));
            let ann_start = op_core(yk, caps, spin, SiteOp::A, 0, 0, 1, Irrep::IDENTITY);
            live_ann.push((
                k,
                adjoint_core_times_core(xk, &mul_sketch_left(&glk, &ann_start)),
            ));
        }
        debug_assert_eq!(live_cre.len(), d - 1, "a chain opened at the last site");
        debug_assert_eq!(live_ann.len(), d - 1, "a chain opened at the last site");
        per_spin.push(gs);
    }

    for p in 0..d {
        for q in 0..d {
            let mut v = T::zero();
            for gs in &per_spin {
                v += gs[p * d + q];
            }
            gamma.set(&[p, q], v);
        }
    }
    gamma
}

/// `γ[p,q] = ⟨x| E_pq H |w⟩` in one sweep, with `H·w` **never materialized** — the production shape
/// for orbital gradients and joint covector rows.
pub fn transition_1rdm_h<T: Scalar>(
    x: &TtVector<T>,
    h: &SparseHamiltonian<T>,
    w: &TtVector<T>,
) -> CoeffTensor<T> {
    let space = x.space;
    assert_eq!(
        space, w.space,
        "transition_1rdm_h: bra and ket must share a QnSpace"
    );
    assert_eq!(
        space, h.space,
        "transition_1rdm_h: state and operator must share a QnSpace"
    );
    let d = space.d;
    let pg = space.pg;
    let mut gamma = CoeffTensor::<T>::zeros(d, 2);
    if d == 0 {
        return gamma;
    }
    assert_unit_boundaries(x, "x");
    assert_unit_boundaries(w, "w");

    let spins = active_spins(&space);
    if spins.is_empty() {
        return gamma;
    }

    let caps = (!pg.is_trivial()).then(|| crate::qn::max_ranks(&space));
    let caps = caps.as_deref();

    let mut l: Vec<Sketch<T>> = Vec::with_capacity(d);
    l.push(Sketch::identity(space, 0, 1));
    for k in 0..d - 1 {
        let t = sketch_h_mul(&l[k], h, w.core(k));
        l.push(adjoint_core_times_core(x.core(k), &t));
    }
    let mut r: Vec<Option<Sketch<T>>> = vec![None; d + 1];
    r[d] = Some(Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let t = h_sketch_mul(
            h,
            w.core(k),
            r[k + 1].as_ref().expect("right env built rightward"),
        );
        r[k] = Some(core_times_adjoint_core(&t, x.core(k)));
    }

    let mut per_spin: Vec<Vec<T>> = Vec::with_capacity(spins.len());
    for &spin in &spins {
        let mut gs = vec![T::zero(); d * d];
        let mut live_cre: Vec<(usize, Sketch<T>)> = Vec::new();
        let mut live_ann: Vec<(usize, Sketch<T>)> = Vec::new();

        for k in 0..d {
            let gk = pg.irrep_of(k);
            let xk = x.core(k);
            let wk = w.core(k);
            let rk1 = r[k + 1].as_ref().expect("right env present at bond k+1");
            let last = k + 1 >= d;

            let ann_close = live_cre
                .iter()
                .any(|&(a, _)| pg.irrep_of(a) == gk)
                .then(|| op_core(xk, caps, spin, SiteOp::A, 1, 1, 0, gk));
            let mut id_plus = IdCache::<T>::new(usize::from(pg.order()));
            for (a, dch) in live_cre.iter_mut() {
                if last && pg.irrep_of(*a) != gk {
                    continue;
                }
                let gmix = sketch_h_mul(dch, h, wk);
                if pg.irrep_of(*a) == gk {
                    let closed = adjoint_core_times_core(
                        ann_close.as_ref().expect("built when a chain can close"),
                        &gmix,
                    );
                    let e = T::from_f64(pair_sign(*a, k, spin));
                    gs[k * d + *a] += e * close_trace(&closed, rk1);
                }
                if !last {
                    let idc = id_plus.get(xk, caps, spin, 1, 1, 0, pg.irrep_of(*a));
                    *dch = adjoint_core_times_core(idc, &gmix);
                }
            }
            let cre_close = live_ann
                .iter()
                .any(|&(a, _)| pg.irrep_of(a) == gk)
                .then(|| op_core(xk, caps, spin, SiteOp::Adag, -1, 0, 1, gk));
            let mut id_minus = IdCache::<T>::new(usize::from(pg.order()));
            for (a, dch) in live_ann.iter_mut() {
                if last && pg.irrep_of(*a) != gk {
                    continue;
                }
                let gmix = sketch_h_mul(dch, h, wk);
                if pg.irrep_of(*a) == gk {
                    let closed = adjoint_core_times_core(
                        cre_close.as_ref().expect("built when a chain can close"),
                        &gmix,
                    );
                    let e = T::from_f64(pair_sign(k, *a, spin));
                    gs[*a * d + k] += e * close_trace(&closed, rk1);
                }
                if !last {
                    let idc = id_minus.get(xk, caps, spin, -1, 0, 1, pg.irrep_of(*a));
                    *dch = adjoint_core_times_core(idc, &gmix);
                }
            }

            let n_diag = op_core(xk, caps, spin, SiteOp::N, 0, 0, 1, Irrep::IDENTITY);
            let gl = sketch_h_mul(&restrict_rows(&l[k], &n_diag.row_ranks), h, wk);
            let closed = adjoint_core_times_core(&n_diag, &gl);
            let e = T::from_f64(pair_sign(k, k, spin));
            gs[k * d + k] += e * close_trace(&closed, rk1);

            if last {
                continue;
            }
            let cre_start = op_core(xk, caps, spin, SiteOp::Adag, 0, 0, 1, Irrep::IDENTITY);
            live_cre.push((k, adjoint_core_times_core(&cre_start, &gl)));
            let ann_start = op_core(xk, caps, spin, SiteOp::A, 0, 0, 1, Irrep::IDENTITY);
            live_ann.push((k, adjoint_core_times_core(&ann_start, &gl)));
        }
        debug_assert_eq!(live_cre.len(), d - 1, "a chain opened at the last site");
        debug_assert_eq!(live_ann.len(), d - 1, "a chain opened at the last site");
        per_spin.push(gs);
    }

    for p in 0..d {
        for q in 0..d {
            let mut v = T::zero();
            for gs in &per_spin {
                v += gs[p * d + q];
            }
            gamma.set(&[p, q], v);
        }
    }
    gamma
}

/// Every one-orbital RDM `ρᵢ` (4×4) and two-orbital RDM `ρᵢⱼ` (16×16, `i < j`) of one abelian state,
/// as computed by [`orbital_rdms`].
#[derive(Clone, Debug)]
pub struct OrbitalRdms<T: Scalar> {
    d: usize,
    rho1: Vec<[T; 16]>,
    rho2: Vec<[T; 256]>,
}

impl<T: Scalar> OrbitalRdms<T> {
    /// The number of orbitals.
    pub fn d(&self) -> usize {
        self.d
    }

    /// `ρᵢ`, 4×4 row-major: `rho1(i)[α·4 + α′] = ρᵢ[α, α′]`. Panics unless `i < d`.
    pub fn rho1(&self, i: usize) -> [T; 16] {
        assert!(
            i < self.d,
            "OrbitalRdms::rho1: orbital {i} is out of range 0..{}",
            self.d
        );
        self.rho1[i]
    }

    /// `ρᵢⱼ`, 16×16 row-major: `rho2(i, j)[(α + 4β)·16 + (α′ + 4β′)] = ρᵢⱼ[(α,β), (α′,β′)]`.
    pub fn rho2(&self, i: usize, j: usize) -> [T; 256] {
        assert!(
            i < j && j < self.d,
            "OrbitalRdms::rho2: needs i < j < d = {} (got ({i}, {j}))",
            self.d
        );
        self.rho2[pair_index(self.d, i, j)]
    }
}

#[inline]
fn pair_index(d: usize, i: usize, j: usize) -> usize {
    i * (2 * d - i - 1) / 2 + (j - i - 1)
}

#[inline]
fn creation_count(op: (SiteOp, SiteOp)) -> (i64, i64) {
    let c = |o: SiteOp| i64::from(matches!(o, SiteOp::Adag | SiteOp::N));
    (c(op.0), c(op.1))
}

struct ChainKind {
    open: (SiteOp, SiteOp),
    closers: &'static [(SiteOp, SiteOp)],
    slot0: usize,
    odd: bool,
}

const Z_CLOSERS: &[(SiteOp, SiteOp)] = &[
    (SiteOp::N, SiteOp::Id),
    (SiteOp::Id, SiteOp::N),
    (SiteOp::N, SiteOp::N),
];
const U_CLOSERS: &[(SiteOp, SiteOp)] = &[(SiteOp::A, SiteOp::Id), (SiteOp::A, SiteOp::N)];
const D_CLOSERS: &[(SiteOp, SiteOp)] = &[(SiteOp::Id, SiteOp::A), (SiteOp::N, SiteOp::A)];

const CHAIN_KINDS: [ChainKind; 9] = [
    ChainKind {
        open: (SiteOp::N, SiteOp::Id),
        closers: Z_CLOSERS,
        slot0: 0,
        odd: false,
    },
    ChainKind {
        open: (SiteOp::Id, SiteOp::N),
        closers: Z_CLOSERS,
        slot0: 3,
        odd: false,
    },
    ChainKind {
        open: (SiteOp::N, SiteOp::N),
        closers: Z_CLOSERS,
        slot0: 6,
        odd: false,
    },
    ChainKind {
        open: (SiteOp::Adag, SiteOp::Id),
        closers: U_CLOSERS,
        slot0: 9,
        odd: true,
    },
    ChainKind {
        open: (SiteOp::Adag, SiteOp::N),
        closers: U_CLOSERS,
        slot0: 11,
        odd: true,
    },
    ChainKind {
        open: (SiteOp::Id, SiteOp::Adag),
        closers: D_CLOSERS,
        slot0: 13,
        odd: true,
    },
    ChainKind {
        open: (SiteOp::N, SiteOp::Adag),
        closers: D_CLOSERS,
        slot0: 15,
        odd: true,
    },
    ChainKind {
        open: (SiteOp::Adag, SiteOp::Adag),
        closers: &[(SiteOp::A, SiteOp::A)],
        slot0: 17,
        odd: false,
    },
    ChainKind {
        open: (SiteOp::Adag, SiteOp::A),
        closers: &[(SiteOp::A, SiteOp::Adag)],
        slot0: 18,
        odd: false,
    },
];
const N_KINDS: usize = CHAIN_KINDS.len();
const N_ZKINDS: usize = 3;
const N_SLOTS: usize = 19;

const DIAG_WEIGHTS: [[f64; 4]; 4] = [
    [1.0, -1.0, -1.0, 1.0],
    [0.0, 1.0, 0.0, -1.0],
    [0.0, 0.0, 1.0, -1.0],
    [0.0, 0.0, 0.0, 1.0],
];

struct OffDiagElement {
    row: usize,
    col: usize,
    single_fermion: bool,
    terms: &'static [(usize, f64)],
}

const U00: usize = 9;
const U01: usize = 10;
const U10: usize = 11;
const U11: usize = 12;
const D00: usize = 13;
const D01: usize = 14;
const D10: usize = 15;
const D11: usize = 16;
const PAIR: usize = 17;
const FLIP: usize = 18;

const OFFDIAG_TABLE: [OffDiagElement; 10] = [
    OffDiagElement {
        row: 4,
        col: 1,
        single_fermion: true,
        terms: &[(U00, 1.0), (U01, -1.0), (U10, -1.0), (U11, 1.0)],
    },
    OffDiagElement {
        row: 12,
        col: 9,
        single_fermion: true,
        terms: &[(U01, 1.0), (U11, -1.0)],
    },
    OffDiagElement {
        row: 6,
        col: 3,
        single_fermion: true,
        terms: &[(U10, -1.0), (U11, 1.0)],
    },
    OffDiagElement {
        row: 14,
        col: 11,
        single_fermion: true,
        terms: &[(U11, -1.0)],
    },
    OffDiagElement {
        row: 8,
        col: 2,
        single_fermion: true,
        terms: &[(D00, 1.0), (D01, -1.0), (D10, -1.0), (D11, 1.0)],
    },
    OffDiagElement {
        row: 12,
        col: 6,
        single_fermion: true,
        terms: &[(D01, -1.0), (D11, 1.0)],
    },
    OffDiagElement {
        row: 9,
        col: 3,
        single_fermion: true,
        terms: &[(D10, 1.0), (D11, -1.0)],
    },
    OffDiagElement {
        row: 13,
        col: 7,
        single_fermion: true,
        terms: &[(D11, -1.0)],
    },
    OffDiagElement {
        row: 12,
        col: 3,
        single_fermion: false,
        terms: &[(PAIR, -1.0)],
    },
    OffDiagElement {
        row: 6,
        col: 9,
        single_fermion: false,
        terms: &[(FLIP, -1.0)],
    },
];

#[inline]
fn label_counts(idx: usize) -> (i64, i64) {
    let (a, b) = (idx % 4, idx / 4);
    (
        (a & 1) as i64 + (b & 1) as i64,
        (a >> 1) as i64 + (b >> 1) as i64,
    )
}

#[inline]
fn weighted_sum(w: &[f64; 4], g: &[f64; 4]) -> f64 {
    let mut acc = 0.0f64;
    for (&wa, &ga) in w.iter().zip(g) {
        if wa != 0.0 {
            acc += wa * ga;
        }
    }
    acc
}

fn restricted_env<'a, T: Scalar>(
    cache: &'a mut Vec<((i64, i64), Sketch<T>)>,
    l: &Sketch<T>,
    key: (i64, i64),
    rr: &RankGrid,
) -> &'a Sketch<T> {
    let pos = match cache.iter().position(|(k, _)| *k == key) {
        Some(p) => p,
        None => {
            cache.push((key, restrict_cols(l, rr)));
            cache.len() - 1
        }
    };
    &cache[pos].1
}

/// Every one-orbital RDM `ρᵢ` and two-orbital RDM `ρᵢⱼ` (`i < j`) of `x`, in ONE left-to-right
/// chain sweep over stored plain environments (the [`transition_1rdm`] structure with more chain
/// kinds).
pub fn orbital_rdms<T: Scalar>(x: &TtVector<T>) -> OrbitalRdms<T> {
    let space = x.space;
    let d = space.d;
    let pg = space.pg;
    if d == 0 {
        return OrbitalRdms {
            d: 0,
            rho1: Vec::new(),
            rho2: Vec::new(),
        };
    }
    assert_unit_boundaries(x, "orbital_rdms: x");
    let n = dot(x, x).re_s();
    assert!(
        n > 0.0 && n.is_finite(),
        "orbital_rdms: ⟨x,x⟩ = {n} must be positive and finite"
    );
    let (nup, ndn) = (space.nup as i64, space.ndn as i64);
    let fits = |c: (i64, i64)| c.0 <= nup && c.1 <= ndn;
    let feasible =
        |c: (i64, i64), outside: i64| fits(c) && nup - c.0 <= outside && ndn - c.1 <= outside;
    let d_i = d as i64;

    let closer_live: Vec<Vec<bool>> = CHAIN_KINDS
        .iter()
        .map(|kind| {
            let co = creation_count(kind.open);
            kind.closers
                .iter()
                .map(|&c| {
                    let cc = creation_count(c);
                    fits((co.0 + cc.0, co.1 + cc.1))
                })
                .collect()
        })
        .collect();
    let kind_open: Vec<bool> = closer_live.iter().map(|v| v.iter().any(|&b| b)).collect();
    let onsite_live: Vec<bool> = CHAIN_KINDS[..N_ZKINDS]
        .iter()
        .map(|kind| fits(creation_count(kind.open)))
        .collect();

    let open_state: Vec<State> = CHAIN_KINDS
        .iter()
        .map(|kind| {
            let co = creation_count(kind.open);
            let min_c = kind.closers.iter().fold((1, 1), |m, &c| {
                let cc = creation_count(c);
                (m.0.min(cc.0), m.1.min(cc.1))
            });
            State {
                fu: 0,
                nlu: 0,
                nru: co.0 + min_c.0,
                fd: 0,
                nld: 0,
                nrd: co.1 + min_c.1,
                delta: Irrep::IDENTITY,
            }
        })
        .collect();

    let caps = (!pg.is_trivial()).then(|| crate::qn::max_ranks(&space));
    let caps = caps.as_deref();

    let mut l: Vec<Sketch<T>> = Vec::with_capacity(d);
    l.push(Sketch::identity(space, 0, 1));
    for k in 0..d - 1 {
        let t = mul_sketch_left(&l[k], x.core(k));
        l.push(adjoint_core_times_core(x.core(k), &t));
    }
    let mut r: Vec<Option<Sketch<T>>> = vec![None; d + 1];
    r[d] = Some(Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let t = mul_sketch_right(
            x.core(k),
            r[k + 1].as_ref().expect("right env built rightward"),
        );
        r[k] = Some(core_times_adjoint_core(&t, x.core(k)));
    }

    let npairs = d * (d - 1) / 2;
    let mut mom: Vec<[T; N_SLOTS]> = vec![[T::zero(); N_SLOTS]; npairs];
    let mut onsite: Vec<[f64; N_ZKINDS]> = vec![[0.0; N_ZKINDS]; d];
    let mut live: Vec<Vec<(usize, Sketch<T>)>> = (0..N_KINDS).map(|_| Vec::new()).collect();
    let mut prop: [Option<State>; N_KINDS] = [None; N_KINDS];

    for k in 0..d {
        let gk = pg.irrep_of(k);
        let xk = x.core(k);
        let rk1 = r[k + 1].as_ref().expect("right env present at bond k+1");
        let last = k + 1 >= d;

        for (ki, kind) in CHAIN_KINDS.iter().enumerate() {
            let chains = &live[ki];
            if chains.is_empty() || (kind.odd && !chains.iter().any(|&(i, _)| pg.irrep_of(i) == gk))
            {
                continue;
            }
            let base = prop[ki].expect("a kind with live chains has been opened");
            let st = if kind.odd {
                State { delta: gk, ..base }
            } else {
                base
            };
            for (c, &(up, dn)) in kind.closers.iter().enumerate() {
                if !closer_live[ki][c] {
                    continue;
                }
                let core = apply_site_op(xk, caps, up, dn, st).0;
                for (i, p) in chains {
                    if kind.odd && pg.irrep_of(*i) != gk {
                        continue;
                    }
                    let closed = adjoint_core_times_core(xk, &mul_sketch_left(p, &core));
                    mom[pair_index(d, *i, k)][kind.slot0 + c] = close_trace(&closed, rk1);
                }
            }
        }

        let mut glk: Vec<((i64, i64), Sketch<T>)> = Vec::with_capacity(3);
        let mut start: Vec<Option<Sketch<T>>> = (0..N_KINDS).map(|_| None).collect();
        for (ki, kind) in CHAIN_KINDS.iter().enumerate() {
            let need_onsite = ki < N_ZKINDS && onsite_live[ki];
            let need_start = !last && kind_open[ki];
            if !need_onsite && !need_start {
                continue;
            }
            let st = open_state[ki];
            let (core, out) = apply_site_op(xk, caps, kind.open.0, kind.open.1, st);
            prop[ki].get_or_insert(out);
            let gl = restricted_env(&mut glk, &l[k], (st.nru, st.nrd), &core.row_ranks);
            let p = adjoint_core_times_core(xk, &mul_sketch_left(gl, &core));
            if need_onsite {
                onsite[k][ki] = close_trace(&p, rk1).re_s();
            }
            if need_start {
                start[ki] = Some(p);
            }
        }

        if last {
            continue;
        }

        for (ki, chains) in live.iter_mut().enumerate() {
            if chains.is_empty() {
                continue;
            }
            let base = prop[ki].expect("a kind with live chains has been opened");
            let mut idc: Vec<Option<SparseCore<T>>> =
                (0..usize::from(pg.order())).map(|_| None).collect();
            for (i, p) in chains.iter_mut() {
                let delta = if CHAIN_KINDS[ki].odd {
                    pg.irrep_of(*i)
                } else {
                    Irrep::IDENTITY
                };
                let slot = &mut idc[usize::from(delta.index())];
                let core = slot.get_or_insert_with(|| {
                    apply_site_op(xk, caps, SiteOp::Id, SiteOp::Id, State { delta, ..base }).0
                });
                *p = adjoint_core_times_core(xk, &mul_sketch_left(p, core));
            }
        }

        for (ki, s) in start.into_iter().enumerate() {
            if let Some(p) = s {
                live[ki].push((k, p));
            }
        }
    }
    debug_assert!(
        live.iter()
            .zip(&kind_open)
            .all(|(v, &o)| v.len() == if o { d - 1 } else { 0 }),
        "a chain opened at the last site, or a kind skipped a site"
    );

    let mut rho1 = vec![[T::zero(); 16]; d];
    for (k, r1) in rho1.iter_mut().enumerate() {
        let g = [n, onsite[k][0], onsite[k][1], onsite[k][2]];
        for (a, w) in DIAG_WEIGHTS.iter().enumerate() {
            if feasible(label_counts(a), d_i - 1) {
                r1[a * 4 + a] = T::from_f64(weighted_sum(w, &g) / n);
            }
        }
    }
    let inv_n = T::from_f64(1.0 / n);
    let mut rho2 = vec![[T::zero(); 256]; npairs];
    for i in 0..d {
        for j in (i + 1)..d {
            let pij = pair_index(d, i, j);
            let m = &mom[pij];
            let rho = &mut rho2[pij];
            let mut gm = [[0.0f64; 4]; 4];
            gm[0][0] = n;
            for a in 1..4 {
                gm[a][0] = onsite[i][a - 1];
                gm[0][a] = onsite[j][a - 1];
                for b in 1..4 {
                    gm[a][b] = m[3 * (a - 1) + (b - 1)].re_s();
                }
            }
            for (beta, wbeta) in DIAG_WEIGHTS.iter().enumerate() {
                for (alpha, walpha) in DIAG_WEIGHTS.iter().enumerate() {
                    let idx = alpha + 4 * beta;
                    if !feasible(label_counts(idx), d_i - 2) {
                        continue;
                    }
                    let mut acc = 0.0f64;
                    for (&wa, grow) in walpha.iter().zip(&gm) {
                        if wa != 0.0 {
                            acc += wa * weighted_sum(wbeta, grow);
                        }
                    }
                    rho[idx * 16 + idx] = T::from_f64(acc / n);
                }
            }
            let same_irrep = pg.irrep_of(i) == pg.irrep_of(j);
            for e in &OFFDIAG_TABLE {
                if (e.single_fermion && !same_irrep) || !feasible(label_counts(e.row), d_i - 2) {
                    continue;
                }
                let mut acc = T::zero();
                for &(slot, coeff) in e.terms {
                    debug_assert!(coeff == 1.0 || coeff == -1.0);
                    if coeff > 0.0 {
                        acc += m[slot];
                    } else {
                        acc -= m[slot];
                    }
                }
                let v = acc * inv_n;
                rho[e.row * 16 + e.col] = v;
                rho[e.col * 16 + e.row] = v.conj_s();
            }
        }
    }
    OrbitalRdms { d, rho1, rho2 }
}

#[cfg(test)]
mod tjr_tests {
    use super::{transition_1rdm, transition_1rdm_bruteforce, transition_1rdm_h};
    use crate::TtVector;
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::integrals::{IntegralsOptions, load_fcidump};
    use crate::qn::QnSpace;
    use crate::test_util::randomized_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::path::Path;

    fn maxg_maxdiff(a: &CoeffTensor<f64>, b: &CoeffTensor<f64>, d: usize) -> (f64, f64) {
        let mut maxg = 0.0f64;
        let mut maxdiff = 0.0f64;
        for p in 0..d {
            for q in 0..d {
                let av = a.get(&[p, q]);
                let bv = b.get(&[p, q]);
                maxg = maxg.max(av.abs()).max(bv.abs());
                maxdiff = maxdiff.max((av - bv).abs());
            }
        }
        (maxg, maxdiff)
    }

    fn h4_fixture() -> (QnSpace, SparseHamiltonian<f64>) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h4.FCIDUMP");
        let load = load_fcidump::<f64>(&path, None, &IntegralsOptions::default())
            .expect("h4 fcidump loads");
        let ham = load.hamiltonian(f64::EPSILON);
        (load.space, ham)
    }

    fn check_dressed_eq_materialized(
        space: QnSpace,
        h: &SparseHamiltonian<f64>,
        seed: u64,
        tag: &str,
    ) {
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let w: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);

        let hw = h.apply(&w);
        let g_dressed = transition_1rdm_h(&x, h, &w);
        let g_materialized = transition_1rdm_bruteforce(&x, &hw);
        let (maxg, maxdiff) = maxg_maxdiff(&g_dressed, &g_materialized, d);
        assert!(
            maxdiff <= 1e-12 * maxg.max(1.0),
            "[{tag}] transition_1rdm_h(x,h,w) vs bruteforce(x,H·w): max|diff|={maxdiff:e} \
             (scale={:e})",
            maxg.max(1.0)
        );

        let hx = h.apply(&x);
        let g_dressed_x = transition_1rdm_h(&x, h, &x);
        let g_materialized_x = transition_1rdm_bruteforce(&x, &hx);
        let (maxgx, maxdiffx) = maxg_maxdiff(&g_dressed_x, &g_materialized_x, d);
        assert!(
            maxdiffx <= 1e-12 * maxgx.max(1.0),
            "[{tag}] transition_1rdm_h(x,h,x) vs bruteforce(x,H·x): max|diff|={maxdiffx:e} \
             (scale={:e})",
            maxgx.max(1.0)
        );

        let w2 = h.apply(&w);
        let hw2 = h.apply(&w2);
        let g_dressed2 = transition_1rdm_h(&x, h, &w2);
        let g_materialized2 = transition_1rdm_bruteforce(&x, &hw2);
        let (maxg2, maxdiff2) = maxg_maxdiff(&g_dressed2, &g_materialized2, d);
        assert!(
            maxdiff2 <= 1e-12 * maxg2.max(1.0),
            "[{tag}] nested transition_1rdm_h(x,h,H·w) vs bruteforce(x,H·(H·w)): max|diff|={maxdiff2:e} \
             (scale={:e})",
            maxg2.max(1.0)
        );
    }

    #[test]
    fn tjr_sweep_eq_bruteforce() {
        let cases: [(usize, usize, i64); 5] =
            [(4, 3, 1), (6, 4, 0), (6, 3, 3), (10, 4, 0), (10, 5, 1)];
        for &(d, n, two_sz) in &cases {
            let space = QnSpace::from_n_two_sz(d, n, two_sz);
            for seed in [0x5EED_1111u64, 0x5EED_2222, 0x5EED_3333] {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
                let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
                let y: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);

                let g_sweep = transition_1rdm(&x, &y);
                let g_brute = transition_1rdm_bruteforce(&x, &y);
                let (maxg, maxdiff) = maxg_maxdiff(&g_sweep, &g_brute, d);
                assert!(
                    maxdiff <= 1e-13 * maxg.max(1.0),
                    "transition_1rdm vs bruteforce at d={d},N={n},2Sz={two_sz},seed={seed:#x}: \
                     max|diff|={maxdiff:e} (scale={:e})",
                    maxg.max(1.0)
                );
            }
        }
    }

    #[test]
    fn tjr_sweep_vacuum_is_exact_zero() {
        let d = 3;
        let space = QnSpace::from_n_two_sz(d, 0, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x0000_0001);
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let g = transition_1rdm(&x, &x);
        for p in 0..d {
            for q in 0..d {
                assert_eq!(
                    g.get(&[p, q]),
                    0.0,
                    "transition_1rdm at N=0 must be EXACTLY zero at ({p},{q})"
                );
            }
        }
    }

    #[test]
    fn tjr_dressed_eq_materialized_h4() {
        let (space, h) = h4_fixture();
        check_dressed_eq_materialized(space, &h, 0xD1E5_0001, "h4");
    }

    #[test]
    fn tjr_dressed_eq_materialized_c64() {
        use crate::C64;
        let d = 4usize;
        let space = QnSpace::from_n_two_sz(d, 2, 0);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for p in 0..d - 1 {
            t.set(&[p, p + 1], -1.0);
            t.set(&[p + 1, p], -1.0);
        }
        for p in 0..d {
            t.set(&[p, p], 0.15 * (p as f64 - 0.5 * d as f64));
        }
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for p in 0..d {
            for q in 0..d {
                if p != q {
                    v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
                }
            }
        }
        let theta: Vec<f64> = (0..d).map(|p| 0.37 + 0.83 * p as f64).collect();
        let ph = |a: f64| C64::new(a.cos(), a.sin());
        let mut tc = CoeffTensor::<C64>::zeros(d, 2);
        for p in 0..d {
            for q in 0..d {
                let c = t.get(&[p, q]);
                if c != 0.0 {
                    tc.set(&[p, q], ph(theta[p] - theta[q]) * C64::new(c, 0.0));
                }
            }
        }
        let mut vc = CoeffTensor::<C64>::zeros(d, 4);
        for p in 0..d {
            for q in 0..d {
                for r in 0..d {
                    for s in 0..d {
                        let c = v.get(&[p, q, r, s]);
                        if c != 0.0 {
                            vc.set(
                                &[p, q, r, s],
                                ph(theta[p] - theta[q] + theta[r] - theta[s]) * C64::new(c, 0.0),
                            );
                        }
                    }
                }
            }
        }
        let h = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD1E5_C640);
        let x: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
        let w: TtVector<C64> = randomized_state(space, 1, 4, &mut rng);

        let hw = h.apply(&w);
        let g_dressed = transition_1rdm_h(&x, &h, &w);
        let g_materialized = transition_1rdm_bruteforce(&x, &hw);
        let mut maxg = 0.0f64;
        let mut maxdiff = 0.0f64;
        let mut max_im = 0.0f64;
        for p in 0..d {
            for q in 0..d {
                let a = g_dressed.get(&[p, q]);
                let b = g_materialized.get(&[p, q]);
                maxg = maxg.max(a.norm()).max(b.norm());
                maxdiff = maxdiff.max((a - b).norm());
                max_im = max_im.max(a.im.abs());
            }
        }
        assert!(
            max_im > 1e-3 * maxg.max(1e-30),
            "anti-vacuity: the gauge-conjugated γ must carry O(1) imaginary mass, got \
             max|Im| = {max_im:e} against scale {maxg:e}"
        );
        assert!(
            maxdiff <= 1e-12 * maxg.max(1.0),
            "[c64] transition_1rdm_h(x,h,w) vs bruteforce(x,H·w): max|diff|={maxdiff:e} \
             (scale={:e})",
            maxg.max(1.0)
        );
    }

    #[test]
    fn tjr_transpose_identity() {
        let (space, h) = h4_fixture();
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7EA5_0001);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let v: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

        let hx = h.apply(&x);
        let lhs = transition_1rdm(&hx, &v);
        let rhs = transition_1rdm_h(&v, &h, &x);

        let mut maxg = 0.0f64;
        let mut maxdiff = 0.0f64;
        for p in 0..d {
            for q in 0..d {
                let a = lhs.get(&[p, q]);
                let b = rhs.get(&[q, p]);
                maxg = maxg.max(a.abs());
                maxdiff = maxdiff.max((a - b).abs());
            }
        }
        assert!(
            maxg > 1e-6,
            "anti-vacuity: gamma(Hx,v) must be genuinely nonzero"
        );
        assert!(
            maxdiff <= 1e-13 * maxg.max(1.0),
            "gamma(Hx,v) != transpose(transition_1rdm_h(v,h,x)): max|diff|={maxdiff:e}"
        );
    }

    #[test]
    fn tjr_op_role_ablation() {
        let d = 6;
        for &(p0, q0) in &[(0usize, 3usize), (4, 1), (2, 2)] {
            let space = QnSpace::from_n_two_sz(d, 4, 0);
            let mut t = CoeffTensor::<f64>::zeros(d, 2);
            t.set(&[p0, q0], 1.0);
            let v = CoeffTensor::<f64>::zeros(d, 4);
            let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7E5A_0000 + (p0 * d + q0) as u64);
            let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
            let y: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);

            let hy = h.apply(&y);
            let g_dressed = transition_1rdm_h(&x, &h, &y);
            let g_materialized = transition_1rdm_bruteforce(&x, &hy);
            let (maxg, maxdiff) = maxg_maxdiff(&g_dressed, &g_materialized, d);
            assert!(
                maxdiff <= 1e-12 * maxg.max(1.0),
                "T2 ablation ({p0},{q0}): transition_1rdm_h vs bruteforce: max|diff|={maxdiff:e}"
            );

            if p0 != q0 {
                let mut maxasym = 0.0f64;
                for p in 0..d {
                    for q in 0..d {
                        maxasym =
                            maxasym.max((g_dressed.get(&[p, q]) - g_dressed.get(&[q, p])).abs());
                    }
                }
                assert!(
                    maxasym > 1e-6 * maxg.max(1.0),
                    "T2 anti-vacuity ({p0},{q0}): gamma_H must be genuinely asymmetric, \
                     max|gamma-gamma^T|={maxasym:e} (a swapped op role or missing transpose would \
                     cancel here)"
                );
            }
        }

        let space = QnSpace::from_n_two_sz(d, 2, 2);
        let (p0, q0) = (5usize, 2usize);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[p0, q0], 1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7E5A_9999);
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let y: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let hy = h.apply(&y);
        let g_dressed = transition_1rdm_h(&x, &h, &y);
        let g_materialized = transition_1rdm_bruteforce(&x, &hy);
        let (maxg, maxdiff) = maxg_maxdiff(&g_dressed, &g_materialized, d);
        assert!(
            maxdiff <= 1e-12 * maxg.max(1.0),
            "T2 spin-polarized arm: transition_1rdm_h vs bruteforce: max|diff|={maxdiff:e}"
        );
    }
}
