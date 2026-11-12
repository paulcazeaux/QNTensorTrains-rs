// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Batched two-string contractions: every `⟨x| E_ij E_kl |x⟩` in one environment sweep — the
//! all-pairs orbital Gram `S_ab = ⟨Ĝ_a x, Ĝ_b x⟩` without ever forming a `Ĝ_a·x` train.

use crate::core::SparseCore;
use crate::core::factor::{
    Sketch, adjoint_core_times_core, core_times_adjoint_core, core_times_adjoint_core_seq,
    mul_sketch_left, mul_sketch_right,
};
use crate::elementaryops::SiteOp;
use crate::orbital::Spin;
use crate::pointgroup::Irrep;
use crate::qn::{RankGrid, SectorGrid};
use crate::rdm::{active_spins, assert_unit_boundaries, close_trace, op_core, pair_sign};
use crate::vector::TtVector;
use faer::Mat;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpenType {
    Cre,
    Ann,
}

pub(crate) const OPEN_TYPES: [OpenType; 2] = [OpenType::Cre, OpenType::Ann];

pub(crate) type StoredFamilies = Vec<Vec<Vec<Vec<Option<Sketch<f64>>>>>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Chain {
    t: OpenType,
    spin: Spin,
    site: usize,
}

impl Chain {
    fn pair(&self, close_site: usize) -> (usize, usize) {
        match self.t {
            OpenType::Cre => (self.site, close_site),
            OpenType::Ann => (close_site, self.site),
        }
    }

    fn key(&self) -> (usize, usize, usize) {
        (self.site, type_ix(self.t), spin_ix(self.spin))
    }
}

pub(crate) fn align_rows(l: &Sketch<f64>, rr: &RankGrid) -> Sketch<f64> {
    let blocks = SectorGrid::from_fn(&l.space, |s| {
        l.block(s).map(|b| {
            if rr[s] == b.nrows() {
                b.to_owned()
            } else if rr[s] == 0 || b.nrows() == 0 {
                Mat::<f64>::zeros(rr[s], b.ncols())
            } else {
                panic!(
                    "pairgram alignment: genuine row-rank mismatch at {s:?}: sketch {} vs \
                     factor {} — a flux argument is wrong",
                    b.nrows(),
                    rr[s]
                );
            }
        })
    });
    Sketch::new(l.space, l.k, blocks)
}

pub(crate) fn align_cols(l: &Sketch<f64>, rr: &RankGrid) -> Sketch<f64> {
    let blocks = SectorGrid::from_fn(&l.space, |s| {
        l.block(s).map(|b| {
            if rr[s] == b.ncols() {
                b.to_owned()
            } else if rr[s] == 0 || b.ncols() == 0 {
                Mat::<f64>::zeros(b.nrows(), rr[s])
            } else {
                panic!(
                    "pairgram alignment: genuine col-rank mismatch at {s:?}: sketch {} vs \
                     factor {} — a flux argument is wrong",
                    b.ncols(),
                    rr[s]
                );
            }
        })
    });
    Sketch::new(l.space, l.k, blocks)
}

fn align_rows_cow<'a>(l: &'a Sketch<f64>, rr: &RankGrid) -> Cow<'a, Sketch<f64>> {
    if l.blocks
        .iter()
        .all(|(s, b)| b.as_ref().is_none_or(|b| b.nrows() == rr[s]))
    {
        Cow::Borrowed(l)
    } else {
        Cow::Owned(align_rows(l, rr))
    }
}

fn align_cols_cow<'a>(l: &'a Sketch<f64>, rr: &RankGrid) -> Cow<'a, Sketch<f64>> {
    if l.blocks
        .iter()
        .all(|(s, b)| b.as_ref().is_none_or(|b| b.ncols() == rr[s]))
    {
        Cow::Borrowed(l)
    } else {
        Cow::Owned(align_cols(l, rr))
    }
}

pub(crate) fn fold_left(
    braop: &SparseCore<f64>,
    sk: &Sketch<f64>,
    ketop: &SparseCore<f64>,
) -> Sketch<f64> {
    let rows = align_rows_cow(sk, &braop.row_ranks);
    let both = align_cols_cow(&rows, &ketop.row_ranks);
    adjoint_core_times_core(braop, &mul_sketch_left(&both, ketop))
}

pub(crate) fn fold_right(
    ketop: &SparseCore<f64>,
    sk: &Sketch<f64>,
    braop: &SparseCore<f64>,
) -> Sketch<f64> {
    let rows = align_rows_cow(sk, &ketop.col_ranks);
    let both = align_cols_cow(&rows, &braop.col_ranks);
    core_times_adjoint_core(&mul_sketch_right(ketop, &both), braop)
}

const SEL_N: usize = 7;

fn sel_n(spin: Spin) -> usize {
    match spin {
        Spin::Up => 1,
        Spin::Dn => 2,
    }
}

fn sel_close(t: OpenType, spin: Spin) -> usize {
    match (t, spin) {
        (OpenType::Cre, Spin::Up) => 3,
        (OpenType::Cre, Spin::Dn) => 4,
        (OpenType::Ann, Spin::Up) => 5,
        (OpenType::Ann, Spin::Dn) => 6,
    }
}

fn spin_ix(s: Spin) -> usize {
    match s {
        Spin::Up => 0,
        Spin::Dn => 1,
    }
}

fn type_ix(t: OpenType) -> usize {
    match t {
        OpenType::Cre => 0,
        OpenType::Ann => 1,
    }
}

pub(crate) fn id_core(core: &SparseCore<f64>, t: OpenType, spin: Spin) -> SparseCore<f64> {
    match t {
        OpenType::Cre => op_core(core, None, spin, SiteOp::Id, 1, 1, 0, Irrep::IDENTITY),
        OpenType::Ann => op_core(core, None, spin, SiteOp::Id, -1, 0, 1, Irrep::IDENTITY),
    }
}

pub(crate) fn open_core(core: &SparseCore<f64>, t: OpenType, spin: Spin) -> SparseCore<f64> {
    let op = match t {
        OpenType::Cre => SiteOp::Adag,
        OpenType::Ann => SiteOp::A,
    };
    op_core(core, None, spin, op, 0, 0, 1, Irrep::IDENTITY)
}

pub(crate) fn close_core(core: &SparseCore<f64>, t: OpenType, spin: Spin) -> SparseCore<f64> {
    match t {
        OpenType::Cre => op_core(core, None, spin, SiteOp::A, 1, 1, 0, Irrep::IDENTITY),
        OpenType::Ann => op_core(core, None, spin, SiteOp::Adag, -1, 0, 1, Irrep::IDENTITY),
    }
}

fn n_core(core: &SparseCore<f64>, spin: Spin) -> SparseCore<f64> {
    op_core(core, None, spin, SiteOp::N, 0, 0, 1, Irrep::IDENTITY)
}

pub(crate) struct SiteCores<'a> {
    plain: &'a SparseCore<f64>,
    n: [Option<SparseCore<f64>>; 2],
    close: [[Option<SparseCore<f64>>; 2]; 2],
    open: [[Option<SparseCore<f64>>; 2]; 2],
    id: [[Option<SparseCore<f64>>; 2]; 2],
}

impl<'a> SiteCores<'a> {
    pub(crate) fn new(xm: &'a SparseCore<f64>, m: usize, d: usize, spins: &[Spin]) -> Self {
        let per_spin = |want: bool, f: &dyn Fn(Spin) -> SparseCore<f64>| {
            [Spin::Up, Spin::Dn].map(|s| (want && spins.contains(&s)).then(|| f(s)))
        };
        let (close_w, open_w) = (m >= 1, m + 1 < d);
        SiteCores {
            plain: xm,
            n: per_spin(true, &|s| n_core(xm, s)),
            close: OPEN_TYPES.map(|t| per_spin(close_w, &|s| close_core(xm, t, s))),
            open: OPEN_TYPES.map(|t| per_spin(open_w, &|s| open_core(xm, t, s))),
            id: OPEN_TYPES.map(|t| per_spin(close_w && open_w, &|s| id_core(xm, t, s))),
        }
    }

    fn n(&self, s: Spin) -> &SparseCore<f64> {
        self.n[spin_ix(s)]
            .as_ref()
            .expect("SiteCores: number core of an inactive spin")
    }

    pub(crate) fn close(&self, t: OpenType, s: Spin) -> &SparseCore<f64> {
        self.close[type_ix(t)][spin_ix(s)]
            .as_ref()
            .expect("SiteCores: close core read at m = 0 or for an inactive spin")
    }

    pub(crate) fn open(&self, t: OpenType, s: Spin) -> &SparseCore<f64> {
        self.open[type_ix(t)][spin_ix(s)]
            .as_ref()
            .expect("SiteCores: open core read at the last site or for an inactive spin")
    }

    pub(crate) fn id(&self, t: OpenType, s: Spin) -> &SparseCore<f64> {
        self.id[type_ix(t)][spin_ix(s)].as_ref().expect(
            "SiteCores: in-flight identity read at m = 0, the last site, or an inactive spin",
        )
    }

    fn sel(&self, sel: usize) -> &SparseCore<f64> {
        match sel {
            0 => self.plain,
            1 => self.n(Spin::Up),
            2 => self.n(Spin::Dn),
            3 => self.close(OpenType::Cre, Spin::Up),
            4 => self.close(OpenType::Cre, Spin::Dn),
            5 => self.close(OpenType::Ann, Spin::Up),
            6 => self.close(OpenType::Ann, Spin::Dn),
            _ => unreachable!("selection {sel} outside 0..{SEL_N}"),
        }
    }
}

struct RtpTable {
    m: usize,
    built: Vec<Option<Sketch<f64>>>,
}

impl RtpTable {
    fn plan(m: usize, spins: &[Spin], mirror: bool) -> Vec<(usize, usize)> {
        let n: Vec<usize> = spins.iter().map(|&s| sel_n(s)).collect();
        let pairs = |bs: &[usize], ks: &[usize], out: &mut Vec<(usize, usize)>| {
            for &b in bs {
                for &k in ks {
                    out.push((b, k));
                }
            }
        };
        let mut keys = Vec::new();
        pairs(&n, &n, &mut keys);
        if m == 0 {
            return keys;
        }
        let c: Vec<usize> = OPEN_TYPES
            .iter()
            .flat_map(|&t| spins.iter().map(move |&s| sel_close(t, s)))
            .collect();
        if !mirror {
            pairs(&c, &n, &mut keys);
            pairs(&n, &c, &mut keys);
            pairs(&[0], &n, &mut keys);
            pairs(&n, &[0], &mut keys);
            pairs(&c, &c, &mut keys);
            pairs(&[0], &c, &mut keys);
            pairs(&c, &[0], &mut keys);
            return keys;
        }
        pairs(&c, &n, &mut keys);
        if m >= 2 {
            pairs(&[0], &n, &mut keys);
        }
        pairs(&n, &[0], &mut keys);
        if m >= 2 {
            pairs(&c, &c, &mut keys);
        } else {
            for &b in &c {
                for &k in c.iter().filter(|&&k| b <= k) {
                    keys.push((b, k));
                }
            }
        }
        pairs(&c, &[0], &mut keys);
        keys
    }

    fn new(
        sc: &SiteCores<'_>,
        r_next: &Sketch<f64>,
        m: usize,
        spins: &[Spin],
        par: bool,
        mirror: bool,
    ) -> Self {
        let keys = Self::plan(m, spins, mirror);
        let sketches: Vec<Sketch<f64>> = if par {
            keys.par_iter()
                .map(|&(b, k)| {
                    crate::par::nested_scope(|| fold_right_seq(sc.sel(k), r_next, sc.sel(b)))
                })
                .collect()
        } else {
            keys.iter()
                .map(|&(b, k)| fold_right(sc.sel(k), r_next, sc.sel(b)))
                .collect()
        };
        let mut built: Vec<Option<Sketch<f64>>> = (0..SEL_N * SEL_N).map(|_| None).collect();
        for ((b, k), sk) in keys.into_iter().zip(sketches) {
            built[b * SEL_N + k] = Some(sk);
        }
        RtpTable { m, built }
    }

    fn get(&self, bsel: usize, ksel: usize) -> &Sketch<f64> {
        self.built[bsel * SEL_N + ksel].as_ref().unwrap_or_else(|| {
            panic!(
                "RtpTable: selection pair ({bsel}, {ksel}) is not planned at site {}",
                self.m
            )
        })
    }
}

pub(crate) fn fold_right_seq(
    ketop: &SparseCore<f64>,
    sk: &Sketch<f64>,
    braop: &SparseCore<f64>,
) -> Sketch<f64> {
    let rows = align_rows_cow(sk, &ketop.col_ranks);
    let both = align_cols_cow(&rows, &braop.col_ranks);
    core_times_adjoint_core_seq(&mul_sketch_right(ketop, &both), braop)
}

pub(crate) type FoldRight = fn(&SparseCore<f64>, &Sketch<f64>, &SparseCore<f64>) -> Sketch<f64>;

type OpenEntry = (Chain, Sketch<f64>);
type D1Entry = (usize, Spin, Sketch<f64>);
type D2Entry = ((usize, usize), Spin, Sketch<f64>);
type BoKoEntry = (Chain, Chain, Sketch<f64>);
type Bd1KoEntry = (usize, Spin, Chain, Sketch<f64>);
type BoKd1Entry = (Chain, usize, Spin, Sketch<f64>);

type Addend = (usize, f64);

struct EntryOut<B> {
    closes: Vec<Addend>,
    c3: Vec<Addend>,
    c3b: Vec<Addend>,
    births: B,
}

impl<B: Default> EntryOut<B> {
    fn new() -> Self {
        EntryOut {
            closes: Vec::new(),
            c3: Vec::new(),
            c3b: Vec::new(),
            births: B::default(),
        }
    }
}

type BoBirths = (Option<D2Entry>, Vec<BoKoEntry>, Vec<BoKd1Entry>);
type KoBirths = (Option<D2Entry>, Vec<BoKoEntry>, Vec<Bd1KoEntry>);

#[derive(Default)]
struct LenvBirths {
    bo: Vec<OpenEntry>,
    ko: Vec<OpenEntry>,
    bd1: Vec<D1Entry>,
    kd1: Vec<D1Entry>,
    boko: Vec<BoKoEntry>,
    bd1ko: Vec<Bd1KoEntry>,
    bokd1: Vec<BoKd1Entry>,
}

#[derive(Clone, Copy)]
enum LenvPlan {
    Bo(OpenType, Spin),
    Ko(OpenType, Spin),
    Bd1(Spin),
    Kd1(Spin),
    BoKo(OpenType, Spin, OpenType, Spin),
    Bd1Ko(Spin, OpenType, Spin),
    BoKd1(OpenType, Spin, Spin),
}

enum LenvBirth {
    Bo(OpenEntry),
    Ko(OpenEntry),
    Bd1(D1Entry),
    Kd1(D1Entry),
    BoKo(BoKoEntry),
    Bd1Ko(Bd1KoEntry),
    BoKd1(BoKd1Entry),
}

struct SiteCtx<'a> {
    m: usize,
    d: usize,
    last: bool,
    spins: &'a [Spin],
    xm: &'a SparseCore<f64>,
    sc: &'a SiteCores<'a>,
    rtp: &'a RtpTable,
    rk: &'a StoredFamilies,
    rb: &'a StoredFamilies,
    mirror: bool,
}

fn map_entries<E: Send, O: Send>(
    par: bool,
    fam: &mut [E],
    task: impl Fn(&mut E) -> O + Sync,
) -> Vec<O> {
    if par {
        fam.par_iter_mut()
            .map(|e| crate::par::nested_scope(|| task(e)))
            .collect()
    } else {
        fam.iter_mut().map(&task).collect()
    }
}

fn join2<A: Send, B: Send>(
    par: bool,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    if par { rayon::join(a, b) } else { (a(), b()) }
}

fn apply_addends(gamma: &mut [f64], adds: &[Addend]) {
    for &(i, a) in adds {
        gamma[i] += a;
    }
}

impl SiteCtx<'_> {
    fn idx(&self, i: usize, j: usize, k: usize, l: usize) -> usize {
        ((i * self.d + j) * self.d + k) * self.d + l
    }

    fn addend(&self, i: usize, j: usize, k: usize, l: usize, value: f64) -> Addend {
        let e = self.idx(i, j, k, l);
        if !self.mirror {
            return (e, value);
        }
        let me = self.idx(l, k, j, i);
        debug_assert_ne!(
            e,
            me,
            "mirror walk: a non-∅, non-BoKo-close addend landed on the self-mirror cell {:?}",
            (i, j, k, l)
        );
        (e.min(me), value)
    }

    fn chain(&self, t: OpenType, spin: Spin) -> Chain {
        Chain {
            t,
            spin,
            site: self.m,
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn c3_sweep(
        &self,
        fam: &StoredFamilies,
        t: OpenType,
        s: Spin,
        transient: &Sketch<f64>,
        out: &mut Vec<Addend>,
        cell: impl Fn(usize) -> ((usize, usize, usize, usize), f64),
    ) {
        for p in self.m + 1..self.d {
            let Some(stored) = fam[type_ix(t)][spin_ix(s)][p]
                .get(self.m + 1)
                .and_then(|o| o.as_ref())
            else {
                continue;
            };
            let v = close_trace(transient, stored);
            let ((i, j, k, l), sign) = cell(p);
            out.push(self.addend(i, j, k, l, sign * v));
        }
    }

    fn bo_task(&self, e: &mut OpenEntry) -> EntryOut<BoBirths> {
        let (m, ch) = (self.m, e.0);
        let mut out = EntryOut::new();
        let (cre, ann) = ch.pair(m);
        let sign = pair_sign(cre, ann, ch.spin);
        for &sk_spin in self.spins {
            let v = close_trace(&e.1, self.rtp.get(sel_close(ch.t, ch.spin), sel_n(sk_spin)));
            out.closes.push(self.addend(ann, cre, m, m, sign * v));
        }
        if self.last {
            return out;
        }
        let sk = &e.1;
        let close_b = self.sc.close(ch.t, ch.spin);
        for &tk in OPEN_TYPES.iter() {
            for &sk_spin in self.spins {
                let transient = fold_left(close_b, sk, self.sc.open(tk, sk_spin));
                let chk = self.chain(tk, sk_spin);
                self.c3_sweep(self.rk, tk, sk_spin, &transient, &mut out.c3, |p| {
                    let (kc, ka) = chk.pair(p);
                    ((ann, cre, kc, ka), sign * pair_sign(kc, ka, sk_spin))
                });
            }
        }
        let bd2 = (ch.pair(m), ch.spin, fold_left(close_b, sk, self.xm));
        let id_b = self.sc.id(ch.t, ch.spin);
        let mut boko = Vec::with_capacity(OPEN_TYPES.len() * self.spins.len());
        for &tk in OPEN_TYPES.iter() {
            for &sk_spin in self.spins {
                let opc = self.sc.open(tk, sk_spin);
                boko.push((ch, self.chain(tk, sk_spin), fold_left(id_b, sk, opc)));
            }
        }
        let bokd1 = self
            .spins
            .iter()
            .map(|&sk_spin| (ch, m, sk_spin, fold_left(id_b, sk, self.sc.n(sk_spin))))
            .collect();
        out.births = (Some(bd2), boko, bokd1);
        e.1 = fold_left(id_b, &e.1, self.xm);
        out
    }

    fn ko_task(&self, e: &mut OpenEntry) -> EntryOut<KoBirths> {
        let (m, ch) = (self.m, e.0);
        let mut out = EntryOut::new();
        let (cre, ann) = ch.pair(m);
        let sign = pair_sign(cre, ann, ch.spin);
        for &sb in self.spins {
            let v = close_trace(&e.1, self.rtp.get(sel_n(sb), sel_close(ch.t, ch.spin)));
            out.closes.push(self.addend(m, m, cre, ann, sign * v));
        }
        if self.last {
            return out;
        }
        let sk = &e.1;
        let close_k = self.sc.close(ch.t, ch.spin);
        for &tb in OPEN_TYPES.iter() {
            for &sb in self.spins {
                let transient = fold_left(self.sc.open(tb, sb), sk, close_k);
                let chb = self.chain(tb, sb);
                self.c3_sweep(self.rb, tb, sb, &transient, &mut out.c3, |p| {
                    let (bc, ba) = chb.pair(p);
                    ((ba, bc, cre, ann), sign * pair_sign(bc, ba, sb))
                });
            }
        }
        let kd2 = (ch.pair(m), ch.spin, fold_left(self.xm, sk, close_k));
        let id_k = self.sc.id(ch.t, ch.spin);
        let mut boko = Vec::with_capacity(OPEN_TYPES.len() * self.spins.len());
        for &tb in OPEN_TYPES.iter() {
            for &sb in self.spins {
                let opc = self.sc.open(tb, sb);
                boko.push((self.chain(tb, sb), ch, fold_left(opc, sk, id_k)));
            }
        }
        let bd1ko = self
            .spins
            .iter()
            .map(|&sb| (m, sb, ch, fold_left(self.sc.n(sb), sk, id_k)))
            .collect();
        out.births = (Some(kd2), boko, bd1ko);
        e.1 = fold_left(self.xm, &e.1, id_k);
        out
    }

    fn bd1_task(&self, e: &mut D1Entry) -> EntryOut<Vec<Bd1KoEntry>> {
        let (m, a, sb) = (self.m, e.0, e.1);
        let mut out = EntryOut::new();
        for &sk_spin in self.spins {
            let v = close_trace(&e.2, self.rtp.get(0, sel_n(sk_spin)));
            out.closes.push(self.addend(a, a, m, m, v));
        }
        if self.last {
            return out;
        }
        for &tk in OPEN_TYPES.iter() {
            for &sk_spin in self.spins {
                let opc = self.sc.open(tk, sk_spin);
                let born = fold_left(self.xm, &e.2, opc);
                out.births.push((a, sb, self.chain(tk, sk_spin), born));
            }
        }
        e.2 = fold_left(self.xm, &e.2, self.xm);
        out
    }

    fn kd1_task(&self, e: &mut D1Entry) -> EntryOut<Vec<BoKd1Entry>> {
        let (m, b, sk_spin) = (self.m, e.0, e.1);
        let mut out = EntryOut::new();
        for &sb in self.spins {
            let v = close_trace(&e.2, self.rtp.get(sel_n(sb), 0));
            out.closes.push(self.addend(m, m, b, b, v));
        }
        if self.last {
            return out;
        }
        for &tb in OPEN_TYPES.iter() {
            for &sb in self.spins {
                let opc = self.sc.open(tb, sb);
                let born = fold_left(opc, &e.2, self.xm);
                out.births.push((self.chain(tb, sb), b, sk_spin, born));
            }
        }
        e.2 = fold_left(self.xm, &e.2, self.xm);
        out
    }

    fn bd2_task(&self, e: &mut D2Entry) -> EntryOut<()> {
        let (m, (cre, ann)) = (self.m, e.0);
        let mut out = EntryOut::new();
        let sign = pair_sign(cre, ann, Spin::Up);
        for &sk_spin in self.spins {
            let v = close_trace(&e.2, self.rtp.get(0, sel_n(sk_spin)));
            out.closes.push(self.addend(ann, cre, m, m, sign * v));
        }
        if self.last {
            return out;
        }
        for &tk in OPEN_TYPES.iter() {
            for &sk_spin in self.spins {
                let transient = fold_left(self.xm, &e.2, self.sc.open(tk, sk_spin));
                let ch = self.chain(tk, sk_spin);
                self.c3_sweep(self.rk, tk, sk_spin, &transient, &mut out.c3, |p| {
                    let (kc, ka) = ch.pair(p);
                    ((ann, cre, kc, ka), sign * pair_sign(kc, ka, sk_spin))
                });
            }
        }
        e.2 = fold_left(self.xm, &e.2, self.xm);
        out
    }

    fn kd2_task(&self, e: &mut D2Entry) -> EntryOut<()> {
        let (m, (cre, ann)) = (self.m, e.0);
        let mut out = EntryOut::new();
        let sign = pair_sign(cre, ann, Spin::Up);
        for &sb in self.spins {
            let v = close_trace(&e.2, self.rtp.get(sel_n(sb), 0));
            out.closes.push(self.addend(m, m, cre, ann, sign * v));
        }
        if self.last {
            return out;
        }
        for &tb in OPEN_TYPES.iter() {
            for &sb in self.spins {
                let transient = fold_left(self.sc.open(tb, sb), &e.2, self.xm);
                let ch = self.chain(tb, sb);
                self.c3_sweep(self.rb, tb, sb, &transient, &mut out.c3, |p| {
                    let (bc, ba) = ch.pair(p);
                    ((ba, bc, cre, ann), sign * pair_sign(bc, ba, sb))
                });
            }
        }
        e.2 = fold_left(self.xm, &e.2, self.xm);
        out
    }

    fn boko_task(&self, e: &mut BoKoEntry) -> EntryOut<()> {
        let (m, chb, chk) = (self.m, e.0, e.1);
        debug_assert!(
            !self.mirror || chb.key() <= chk.key(),
            "mirror walk: a dropped-half BoKo element {chb:?} × {chk:?} exists"
        );
        let mut out = EntryOut::new();
        let (bc, ba) = chb.pair(m);
        let (kc, ka) = chk.pair(m);
        let sign = pair_sign(bc, ba, chb.spin) * pair_sign(kc, ka, chk.spin);
        let v = close_trace(
            &e.2,
            self.rtp
                .get(sel_close(chb.t, chb.spin), sel_close(chk.t, chk.spin)),
        );
        let cell = self.idx(ba, bc, kc, ka);
        if self.mirror {
            let me = self.idx(ka, kc, bc, ba);
            let w = if cell == me && chb != chk { 2.0 } else { 1.0 };
            out.closes.push((cell.min(me), w * (sign * v)));
        } else {
            out.closes.push((cell, sign * v));
        }
        if self.last {
            return out;
        }
        let sk = &e.2;
        let id_b = self.sc.id(chb.t, chb.spin);
        let id_k = self.sc.id(chk.t, chk.spin);
        let sign_b = pair_sign(bc, ba, chb.spin);
        let transient = fold_left(self.sc.close(chb.t, chb.spin), sk, id_k);
        self.c3_sweep(self.rk, chk.t, chk.spin, &transient, &mut out.c3, |p| {
            let (kc, ka) = chk.pair(p);
            ((ba, bc, kc, ka), sign_b * pair_sign(kc, ka, chk.spin))
        });
        if !self.mirror || chb != chk {
            let sign_k = pair_sign(kc, ka, chk.spin);
            let transient = fold_left(id_b, sk, self.sc.close(chk.t, chk.spin));
            self.c3_sweep(self.rb, chb.t, chb.spin, &transient, &mut out.c3b, |p| {
                let (bc, ba) = chb.pair(p);
                ((ba, bc, kc, ka), sign_k * pair_sign(bc, ba, chb.spin))
            });
        }
        e.2 = fold_left(id_b, &e.2, id_k);
        out
    }

    fn bd1ko_task(&self, e: &mut Bd1KoEntry) -> EntryOut<()> {
        let (m, a, chk) = (self.m, e.0, e.2);
        let mut out = EntryOut::new();
        let (kc, ka) = chk.pair(m);
        let sign = pair_sign(kc, ka, chk.spin);
        let v = close_trace(&e.3, self.rtp.get(0, sel_close(chk.t, chk.spin)));
        out.closes.push(self.addend(a, a, kc, ka, sign * v));
        if self.last {
            return out;
        }
        e.3 = fold_left(self.xm, &e.3, self.sc.id(chk.t, chk.spin));
        out
    }

    fn bokd1_task(&self, e: &mut BoKd1Entry) -> EntryOut<()> {
        let (m, chb, b) = (self.m, e.0, e.1);
        let mut out = EntryOut::new();
        let (bc, ba) = chb.pair(m);
        let sign = pair_sign(bc, ba, chb.spin);
        let v = close_trace(&e.3, self.rtp.get(sel_close(chb.t, chb.spin), 0));
        out.closes.push(self.addend(ba, bc, b, b, sign * v));
        if self.last {
            return out;
        }
        e.3 = fold_left(self.sc.id(chb.t, chb.spin), &e.3, self.xm);
        out
    }

    fn lenv_births(&self, lm: &Sketch<f64>, par: bool) -> LenvBirths {
        let (spins, full) = (self.spins, !self.mirror);
        let mut plan = Vec::new();
        for &t in OPEN_TYPES.iter() {
            for &s in spins {
                plan.push(LenvPlan::Bo(t, s));
                if full {
                    plan.push(LenvPlan::Ko(t, s));
                }
            }
        }
        for &s in spins {
            if full {
                plan.push(LenvPlan::Bd1(s));
            }
            plan.push(LenvPlan::Kd1(s));
        }
        for &tb in OPEN_TYPES.iter() {
            for &sb in spins {
                for &tk in OPEN_TYPES.iter() {
                    for &sk in spins {
                        if full || (type_ix(tb), spin_ix(sb)) <= (type_ix(tk), spin_ix(sk)) {
                            plan.push(LenvPlan::BoKo(tb, sb, tk, sk));
                        }
                    }
                }
            }
        }
        if full {
            for &sb in spins {
                for &tk in OPEN_TYPES.iter() {
                    for &sk in spins {
                        plan.push(LenvPlan::Bd1Ko(sb, tk, sk));
                    }
                }
            }
        }
        for &tb in OPEN_TYPES.iter() {
            for &sb in spins {
                for &sk in spins {
                    plan.push(LenvPlan::BoKd1(tb, sb, sk));
                }
            }
        }
        let (m, xm, sc) = (self.m, self.xm, self.sc);
        let build = |p: &mut LenvPlan| match *p {
            LenvPlan::Bo(t, s) => {
                LenvBirth::Bo((self.chain(t, s), fold_left(sc.open(t, s), lm, xm)))
            }
            LenvPlan::Ko(t, s) => {
                LenvBirth::Ko((self.chain(t, s), fold_left(xm, lm, sc.open(t, s))))
            }
            LenvPlan::Bd1(s) => LenvBirth::Bd1((m, s, fold_left(sc.n(s), lm, xm))),
            LenvPlan::Kd1(s) => LenvBirth::Kd1((m, s, fold_left(xm, lm, sc.n(s)))),
            LenvPlan::BoKo(tb, sb, tk, sk) => LenvBirth::BoKo((
                self.chain(tb, sb),
                self.chain(tk, sk),
                fold_left(sc.open(tb, sb), lm, sc.open(tk, sk)),
            )),
            LenvPlan::Bd1Ko(sb, tk, sk) => LenvBirth::Bd1Ko((
                m,
                sb,
                self.chain(tk, sk),
                fold_left(sc.n(sb), lm, sc.open(tk, sk)),
            )),
            LenvPlan::BoKd1(tb, sb, sk) => LenvBirth::BoKd1((
                self.chain(tb, sb),
                m,
                sk,
                fold_left(sc.open(tb, sb), lm, sc.n(sk)),
            )),
        };
        let mut out = LenvBirths::default();
        for b in map_entries(par, &mut plan, build) {
            match b {
                LenvBirth::Bo(e) => out.bo.push(e),
                LenvBirth::Ko(e) => out.ko.push(e),
                LenvBirth::Bd1(e) => out.bd1.push(e),
                LenvBirth::Kd1(e) => out.kd1.push(e),
                LenvBirth::BoKo(e) => out.boko.push(e),
                LenvBirth::Bd1Ko(e) => out.bd1ko.push(e),
                LenvBirth::BoKd1(e) => out.bokd1.push(e),
            }
        }
        out
    }
}

#[allow(clippy::too_many_arguments)]
fn stored_chain(
    x: &TtVector<f64>,
    sc_all: &[SiteCores<'_>],
    r_plain: &Sketch<f64>,
    ket: bool,
    t: OpenType,
    s: Spin,
    p: usize,
    fold: FoldRight,
) -> Vec<Option<Sketch<f64>>> {
    let close_p = sc_all[p].close(t, s);
    let mut by_bond: Vec<Option<Sketch<f64>>> = vec![None; p + 1];
    let mut cur = if ket {
        fold(close_p, r_plain, x.core(p))
    } else {
        fold(x.core(p), r_plain, close_p)
    };
    by_bond[p] = Some(cur.clone());
    for site in (1..p).rev() {
        let idc = sc_all[site].id(t, s);
        cur = if ket {
            fold(idc, &cur, x.core(site))
        } else {
            fold(x.core(site), &cur, idc)
        };
        by_bond[site] = Some(cur.clone());
    }
    by_bond
}

pub(crate) static GAMMA2_MIRROR: AtomicBool = AtomicBool::new(true);

/// Read the Γ₂ walk switch (see [`GAMMA2_MIRROR`]).
#[inline]
pub fn gamma2_mirror() -> bool {
    GAMMA2_MIRROR.load(Ordering::Relaxed)
}

/// Set the Γ₂ walk switch (see [`GAMMA2_MIRROR`]); returns the previous value.
pub fn set_gamma2_mirror(on: bool) -> bool {
    GAMMA2_MIRROR.swap(on, Ordering::Relaxed)
}

/// `Γ₂[i,j,k,l] = ⟨x| E_ij E_kl |x⟩` for all index quadruples, dense `d⁴` row-major
/// (`[((i·d + j)·d + k)·d + l]`). See the module header for the sweep; the composition oracle is
/// `transition_1rdm(x, e_pq(x, k, l, 1))[i, j]`, agreed with to roundoff, never bitwise.
///
/// # Panics
///
/// On a nontrivial point group (per-side identity-flux gating lands with the batched
/// tangent-projection kernel — §6), and on non-unit boundary ranks.
pub fn generalized_gamma2(x: &TtVector<f64>) -> Vec<f64> {
    let mirror = gamma2_mirror();
    let par = x.space.d >= 2 && crate::par::par_driver_fork(gamma2_work(x));
    gamma2_with(x, par, mirror)
}

fn gamma2_work(x: &TtVector<f64>) -> usize {
    let d = x.space.d;
    let s = active_spins(&x.space).len();
    let coeffs = (0..d).map(|m| x.core(m).coeff_hint()).max().unwrap_or(0);
    d.saturating_pow(3)
        .saturating_mul(s * s)
        .saturating_mul(coeffs)
}

fn gamma2_with(x: &TtVector<f64>, par: bool, mirror: bool) -> Vec<f64> {
    let space = x.space;
    let d = space.d;
    assert!(
        space.pg.is_trivial(),
        "generalized_gamma2: point-group gating is not implemented yet — under |G| > 1 only \
         per-side identity-flux cells survive, and computing the rest silently wrong would be \
         worse than this panic"
    );
    let mut gamma = vec![0.0f64; d * d * d * d];
    if d == 0 {
        return gamma;
    }
    assert_unit_boundaries(x, "x");
    let spins = active_spins(&space);
    if spins.is_empty() {
        return gamma;
    }
    let fold_r: FoldRight = if par { fold_right_seq } else { fold_right };

    let mut l_env: Vec<Sketch<f64>> = Vec::with_capacity(d);
    l_env.push(Sketch::identity(space, 0, 1));
    for m in 0..d - 1 {
        let next = fold_left(x.core(m), &l_env[m], x.core(m));
        l_env.push(next);
    }
    let mut r_env: Vec<Option<Sketch<f64>>> = vec![None; d + 1];
    r_env[d] = Some(Sketch::identity(space, d, 1));
    for m in (1..d).rev() {
        let prev = r_env[m + 1].as_ref().expect("right env built rightward");
        r_env[m] = Some(fold_right(x.core(m), prev, x.core(m)));
    }

    let site_idx: Vec<usize> = (0..d).collect();
    let sc_all: Vec<SiteCores<'_>> = if par {
        site_idx
            .par_iter()
            .map(|&m| crate::par::nested_scope(|| SiteCores::new(x.core(m), m, d, &spins)))
            .collect()
    } else {
        site_idx
            .iter()
            .map(|&m| SiteCores::new(x.core(m), m, d, &spins))
            .collect()
    };

    let mut chain_plan: Vec<(OpenType, Spin, usize, bool)> = Vec::new();
    for &t in OPEN_TYPES.iter() {
        for &s in spins.iter() {
            for p in 1..d {
                chain_plan.push((t, s, p, true));
                chain_plan.push((t, s, p, false));
            }
        }
    }
    let build_chain = |&(t, s, p, ket): &(OpenType, Spin, usize, bool)| {
        let r_plain = r_env[p + 1].as_ref().expect("plain right env");
        stored_chain(x, &sc_all, r_plain, ket, t, s, p, fold_r)
    };
    let chains: Vec<Vec<Option<Sketch<f64>>>> = if par {
        chain_plan
            .par_iter()
            .map(|c| crate::par::nested_scope(|| build_chain(c)))
            .collect()
    } else {
        chain_plan.iter().map(build_chain).collect()
    };
    let mut rk: StoredFamilies = vec![vec![vec![Vec::new(); d]; 2]; 2];
    let mut rb: StoredFamilies = vec![vec![vec![Vec::new(); d]; 2]; 2];
    for ((t, s, p, ket), by_bond) in chain_plan.into_iter().zip(chains) {
        let side = if ket { &mut rk } else { &mut rb };
        side[type_ix(t)][spin_ix(s)][p] = by_bond;
    }

    let mut bo: Vec<OpenEntry> = Vec::new();
    let mut ko: Vec<OpenEntry> = Vec::new();
    let mut bd1: Vec<D1Entry> = Vec::new();
    let mut kd1: Vec<D1Entry> = Vec::new();
    let mut bd2: Vec<D2Entry> = Vec::new();
    let mut kd2: Vec<D2Entry> = Vec::new();
    let mut boko: Vec<BoKoEntry> = Vec::new();
    let mut bd1ko: Vec<Bd1KoEntry> = Vec::new();
    let mut bokd1: Vec<BoKd1Entry> = Vec::new();

    for m in 0..d {
        let r_next = r_env[m + 1].as_ref().expect("plain right env at m+1");
        let sc = &sc_all[m];
        let rtp = RtpTable::new(sc, r_next, m, &spins, par, mirror);
        let ctx = SiteCtx {
            m,
            d,
            last: m + 1 >= d,
            spins: &spins,
            xm: x.core(m),
            sc,
            rtp: &rtp,
            rk: &rk,
            rb: &rb,
            mirror,
        };
        let c = &ctx;
        debug_assert!(
            !mirror || (ko.is_empty() && bd1.is_empty() && kd2.is_empty() && bd1ko.is_empty()),
            "mirror walk: a dropped family is populated at site {m}"
        );
        let lm = &l_env[m];

        let mut e0: Vec<Addend> = Vec::with_capacity(spins.len() * spins.len());
        for &sb in spins.iter() {
            for &sk_ in spins.iter() {
                let v = close_trace(lm, rtp.get(sel_n(sb), sel_n(sk_)));
                e0.push((c.idx(m, m, m, m), v));
            }
        }

        let (((o_bo, o_ko), (o_bd1, o_kd1)), ((o_bd2, o_kd2), ((o_boko, o_bd1ko), (o_bokd1, lb)))) =
            join2(
                par,
                || {
                    join2(
                        par,
                        || {
                            join2(
                                par,
                                || map_entries(par, &mut bo, |e| c.bo_task(e)),
                                || map_entries(par, &mut ko, |e| c.ko_task(e)),
                            )
                        },
                        || {
                            join2(
                                par,
                                || map_entries(par, &mut bd1, |e| c.bd1_task(e)),
                                || map_entries(par, &mut kd1, |e| c.kd1_task(e)),
                            )
                        },
                    )
                },
                || {
                    join2(
                        par,
                        || {
                            join2(
                                par,
                                || map_entries(par, &mut bd2, |e| c.bd2_task(e)),
                                || map_entries(par, &mut kd2, |e| c.kd2_task(e)),
                            )
                        },
                        || {
                            join2(
                                par,
                                || {
                                    join2(
                                        par,
                                        || map_entries(par, &mut boko, |e| c.boko_task(e)),
                                        || map_entries(par, &mut bd1ko, |e| c.bd1ko_task(e)),
                                    )
                                },
                                || {
                                    join2(
                                        par,
                                        || map_entries(par, &mut bokd1, |e| c.bokd1_task(e)),
                                        || (!c.last).then(|| c.lenv_births(lm, par)),
                                    )
                                },
                            )
                        },
                    )
                },
            );

        apply_addends(&mut gamma, &e0);
        let g = &mut gamma;
        o_bo.iter().for_each(|o| apply_addends(g, &o.closes));
        o_ko.iter().for_each(|o| apply_addends(g, &o.closes));
        o_bd1.iter().for_each(|o| apply_addends(g, &o.closes));
        o_kd1.iter().for_each(|o| apply_addends(g, &o.closes));
        o_bd2.iter().for_each(|o| apply_addends(g, &o.closes));
        o_kd2.iter().for_each(|o| apply_addends(g, &o.closes));
        o_boko.iter().for_each(|o| apply_addends(g, &o.closes));
        o_bd1ko.iter().for_each(|o| apply_addends(g, &o.closes));
        o_bokd1.iter().for_each(|o| apply_addends(g, &o.closes));

        let Some(lb) = lb else {
            continue;
        };
        o_bd2.iter().for_each(|o| apply_addends(g, &o.c3));
        o_kd2.iter().for_each(|o| apply_addends(g, &o.c3));
        o_bo.iter().for_each(|o| apply_addends(g, &o.c3));
        o_ko.iter().for_each(|o| apply_addends(g, &o.c3));
        o_boko.iter().for_each(|o| apply_addends(g, &o.c3));
        o_boko.iter().for_each(|o| apply_addends(g, &o.c3b));

        let (mut new_bd2, mut boko_bo, mut bokd1_bo) = (Vec::new(), Vec::new(), Vec::new());
        for o in o_bo {
            let (b2, bk, bkd) = o.births;
            new_bd2.extend(b2);
            boko_bo.extend(bk);
            bokd1_bo.extend(bkd);
        }
        let (mut new_kd2, mut boko_ko, mut bd1ko_ko) = (Vec::new(), Vec::new(), Vec::new());
        for o in o_ko {
            let (k2, bk, bdk) = o.births;
            new_kd2.extend(k2);
            boko_ko.extend(bk);
            bd1ko_ko.extend(bdk);
        }
        bo.extend(lb.bo);
        ko.extend(lb.ko);
        bd1.extend(lb.bd1);
        kd1.extend(lb.kd1);
        bd2.extend(new_bd2);
        kd2.extend(new_kd2);
        boko.extend(boko_bo);
        boko.extend(boko_ko);
        boko.extend(lb.boko);
        bd1ko.extend(o_bd1.into_iter().flat_map(|o| o.births));
        bd1ko.extend(bd1ko_ko);
        bd1ko.extend(lb.bd1ko);
        bokd1.extend(o_kd1.into_iter().flat_map(|o| o.births));
        bokd1.extend(bokd1_bo);
        bokd1.extend(lb.bokd1);
    }

    if mirror {
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    for l in 0..d {
                        let e = ((i * d + j) * d + k) * d + l;
                        let me = ((l * d + k) * d + j) * d + i;
                        if me < e {
                            debug_assert_eq!(
                                gamma[e].to_bits(),
                                0,
                                "mirror walk: an addend landed on the non-canonical cell {:?}",
                                (i, j, k, l)
                            );
                            gamma[e] = gamma[me];
                        }
                    }
                }
            }
        }
    }
    gamma
}

/// The all-pairs orbital Gram `S[a,b] = ⟨u_a, u_b⟩`, `u_a = −Ĝ_{p_a q_a}·x`, over `pairs`
/// (`p > q`, [`crate::rotation::orbital_pairs`] order), row-major `n_orb²` — assembled from
/// [`generalized_gamma2`] by the four-term identity of the module header (`u`-sign-free) and
/// symmetrized triangle-plus-mirror, so the output is exactly symmetric.
pub fn pair_gram(x: &TtVector<f64>, pairs: &[(usize, usize)]) -> Vec<f64> {
    let d = x.space.d;
    let g2 = generalized_gamma2(x);
    let idx = |i: usize, j: usize, k: usize, l: usize| ((i * d + j) * d + k) * d + l;
    let n = pairs.len();
    let mut s = vec![0.0f64; n * n];
    for a in 0..n {
        let (p, q) = pairs[a];
        for b in a..n {
            let (r, t) = pairs[b];
            let v = g2[idx(q, p, r, t)] - g2[idx(q, p, t, r)] - g2[idx(p, q, r, t)]
                + g2[idx(p, q, t, r)];
            s[a * n + b] = v;
            s[b * n + a] = v;
        }
    }
    s
}

#[cfg(test)]
mod frozen_walk;

#[cfg(test)]
mod mirror_tests;

#[cfg(test)]
mod align_tests {
    use super::{align_cols, align_rows};
    use crate::core::factor::Sketch;
    use crate::qn::{QnSpace, RankGrid};

    fn space() -> QnSpace {
        QnSpace::new(2, 1, 0)
    }

    fn grid(k: usize, r: usize) -> RankGrid {
        let sp = space();
        let mut g = RankGrid::zeros(&sp);
        for s in sp.bond_sectors(k) {
            g[s] = r;
        }
        g
    }

    #[test]
    fn align_kills_and_pads_zero_versus_n() {
        let sp = space();
        let k = 1usize;
        let sk = Sketch::<f64>::zeros(sp, k, &grid(k, 2), &grid(k, 3));

        let killed = align_rows(&sk, &grid(k, 0));
        let killed_cols = align_cols(&sk, &grid(k, 0));
        for s in sp.bond_sectors(k) {
            let b = killed.block(s).expect("block survives a kill as 0×n");
            assert_eq!((b.nrows(), b.ncols()), (0, 3), "kill rows at {s:?}");
            let c = killed_cols.block(s).expect("block survives a kill as n×0");
            assert_eq!((c.nrows(), c.ncols()), (2, 0), "kill cols at {s:?}");
        }
        let empty = Sketch::<f64>::zeros(sp, k, &grid(k, 0), &grid(k, 3));
        let padded = align_rows(&empty, &grid(k, 4));
        for s in sp.bond_sectors(k) {
            let b = padded.block(s).expect("block survives a pad");
            assert_eq!((b.nrows(), b.ncols()), (4, 3), "pad rows at {s:?}");
        }
    }

    #[test]
    #[should_panic(expected = "genuine row-rank mismatch")]
    fn align_rows_refuses_nonzero_mismatch() {
        let sp = space();
        let sk = Sketch::<f64>::zeros(sp, 1, &grid(1, 2), &grid(1, 3));
        let _ = align_rows(&sk, &grid(1, 1));
    }

    #[test]
    #[should_panic(expected = "genuine col-rank mismatch")]
    fn align_cols_refuses_nonzero_mismatch() {
        let sp = space();
        let sk = Sketch::<f64>::zeros(sp, 1, &grid(1, 2), &grid(1, 3));
        let _ = align_cols(&sk, &grid(1, 1));
    }
}

#[cfg(test)]
mod align_cow_tests {
    use super::{
        OPEN_TYPES, align_cols, align_cols_cow, align_rows, align_rows_cow, close_core, fold_left,
        fold_right, id_core, open_core,
    };
    use crate::core::SparseCore;
    use crate::core::factor::{
        Sketch, adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
    };
    use crate::qn::{QnSpace, RankGrid};
    use crate::rdm::active_spins;
    use crate::test_util::{randomized_state, seeded};
    use crate::vector::TtVector;
    use std::borrow::Cow;

    fn space() -> QnSpace {
        QnSpace::new(2, 1, 0)
    }

    fn grid(k: usize, r: usize) -> RankGrid {
        let sp = space();
        let mut g = RankGrid::zeros(&sp);
        for s in sp.bond_sectors(k) {
            g[s] = r;
        }
        g
    }

    fn filled(k: usize, rows: &RankGrid, cols: &RankGrid, seed: u64) -> Sketch<f64> {
        let mut sk = Sketch::<f64>::zeros(space(), k, rows, cols);
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for s in sk.sectors() {
            if let Some(mut b) = sk.block_mut(s) {
                for j in 0..b.ncols() {
                    for i in 0..b.nrows() {
                        b[(i, j)] = next();
                    }
                }
            }
        }
        sk
    }

    fn sketch_bits_eq(a: &Sketch<f64>, b: &Sketch<f64>, what: &str) {
        assert_eq!(a.space, b.space, "{what}: space");
        assert_eq!(a.k, b.k, "{what}: bond");
        assert_eq!(
            a.blocks.iter().count(),
            b.blocks.iter().count(),
            "{what}: grid extent"
        );
        for ((s, ba), (t, bb)) in a.blocks.iter().zip(b.blocks.iter()) {
            assert_eq!(s, t, "{what}: grid order");
            match (ba, bb) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(
                        (x.nrows(), x.ncols()),
                        (y.nrows(), y.ncols()),
                        "{what}: shape at {s:?}"
                    );
                    for j in 0..x.ncols() {
                        for i in 0..x.nrows() {
                            assert_eq!(
                                x[(i, j)].to_bits(),
                                y[(i, j)].to_bits(),
                                "{what}: entry ({i},{j}) at {s:?}"
                            );
                        }
                    }
                }
                _ => panic!("{what}: block presence differs at {s:?}"),
            }
        }
    }

    #[allow(clippy::ptr_arg)]
    fn is_borrowed(c: &Cow<'_, Sketch<f64>>) -> bool {
        matches!(c, Cow::Borrowed(_))
    }

    #[test]
    fn align_cow_borrows_iff_aligned() {
        let sp = space();
        let k = 1usize;
        let sk = filled(k, &grid(k, 2), &grid(k, 3), 0xA11C_0001);

        let r = align_rows_cow(&sk, &grid(k, 2));
        assert!(is_borrowed(&r), "matched rows must borrow");
        assert!(std::ptr::eq(&*r, &sk), "a borrow must be the input itself");
        let c = align_cols_cow(&sk, &grid(k, 3));
        assert!(is_borrowed(&c), "matched cols must borrow");
        assert!(std::ptr::eq(&*c, &sk), "a borrow must be the input itself");

        assert!(
            !is_borrowed(&align_rows_cow(&sk, &grid(k, 0))),
            "row kill must own"
        );
        assert!(
            !is_borrowed(&align_cols_cow(&sk, &grid(k, 0))),
            "col kill must own"
        );
        let empty = filled(k, &grid(k, 0), &grid(k, 3), 0xA11C_0002);
        assert!(
            !is_borrowed(&align_rows_cow(&empty, &grid(k, 4))),
            "row pad must own"
        );
        let empty_c = filled(k, &grid(k, 2), &grid(k, 0), 0xA11C_0003);
        assert!(
            !is_borrowed(&align_cols_cow(&empty_c, &grid(k, 5))),
            "col pad must own"
        );

        let secs = sp.bond_sectors(k);
        assert!(secs.len() >= 2, "fixture needs two sectors at bond {k}");
        let mut mixed_r = grid(k, 2);
        mixed_r[secs[0]] = 0;
        assert!(
            !is_borrowed(&align_rows_cow(&sk, &mixed_r)),
            "one killed row sector must own"
        );
        let mut mixed_c = grid(k, 3);
        mixed_c[secs[1]] = 0;
        assert!(
            !is_borrowed(&align_cols_cow(&sk, &mixed_c)),
            "one killed col sector must own"
        );

        let mut holey = sk.clone();
        holey.blocks[secs[0]] = None;
        let mut rr = grid(k, 2);
        rr[secs[0]] = 7;
        let mut cc = grid(k, 3);
        cc[secs[0]] = 7;
        assert!(
            is_borrowed(&align_rows_cow(&holey, &rr)),
            "a None block must not force a copy"
        );
        assert!(
            is_borrowed(&align_cols_cow(&holey, &cc)),
            "a None block must not force a copy"
        );
    }

    #[test]
    fn align_cow_owned_equals_owned_form_bitwise() {
        let sp = space();
        let k = 1usize;
        let secs = sp.bond_sectors(k);
        let sk = filled(k, &grid(k, 2), &grid(k, 3), 0xA11C_0011);
        let empty_r = filled(k, &grid(k, 0), &grid(k, 3), 0xA11C_0012);
        let empty_c = filled(k, &grid(k, 2), &grid(k, 0), 0xA11C_0013);
        let mut mixed_r = grid(k, 2);
        mixed_r[secs[0]] = 0;
        let mut mixed_c = grid(k, 3);
        mixed_c[secs[1]] = 0;
        let mut holey = sk.clone();
        holey.blocks[secs[0]] = None;

        let row_cases: [(&Sketch<f64>, RankGrid, &str); 5] = [
            (&sk, grid(k, 2), "rows matched"),
            (&sk, grid(k, 0), "rows killed"),
            (&empty_r, grid(k, 4), "rows padded"),
            (&sk, mixed_r, "rows mixed"),
            (&holey, grid(k, 2), "rows with a None block"),
        ];
        for (l, rr, what) in &row_cases {
            sketch_bits_eq(&align_rows_cow(l, rr), &align_rows(l, rr), what);
        }
        let col_cases: [(&Sketch<f64>, RankGrid, &str); 5] = [
            (&sk, grid(k, 3), "cols matched"),
            (&sk, grid(k, 0), "cols killed"),
            (&empty_c, grid(k, 5), "cols padded"),
            (&sk, mixed_c, "cols mixed"),
            (&holey, grid(k, 3), "cols with a None block"),
        ];
        for (l, rr, what) in &col_cases {
            sketch_bits_eq(&align_cols_cow(l, rr), &align_cols(l, rr), what);
        }
    }

    #[test]
    #[should_panic(expected = "genuine row-rank mismatch")]
    fn align_rows_cow_refuses_nonzero_mismatch() {
        let sk = filled(1, &grid(1, 2), &grid(1, 3), 0xA11C_0021);
        let _ = align_rows_cow(&sk, &grid(1, 1));
    }

    #[test]
    #[should_panic(expected = "genuine col-rank mismatch")]
    fn align_cols_cow_refuses_nonzero_mismatch() {
        let sk = filled(1, &grid(1, 2), &grid(1, 3), 0xA11C_0022);
        let _ = align_cols_cow(&sk, &grid(1, 1));
    }

    fn owned_left(b: &SparseCore<f64>, sk: &Sketch<f64>, k: &SparseCore<f64>) -> Sketch<f64> {
        let a = align_cols(&align_rows(sk, &b.row_ranks), &k.row_ranks);
        adjoint_core_times_core(b, &mul_sketch_left(&a, k))
    }

    fn owned_right(k: &SparseCore<f64>, sk: &Sketch<f64>, b: &SparseCore<f64>) -> Sketch<f64> {
        let a = align_cols(&align_rows(sk, &k.col_ranks), &b.col_ranks);
        core_times_adjoint_core(&mul_sketch_right(k, &a), b)
    }

    #[test]
    fn fold_cow_matches_owned_composition_bitwise() {
        let space = QnSpace::new(5, 3, 2);
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut seeded(0xF2C0_F01D));
        let d = space.d;
        let spins = active_spins(&space);
        let (mut borrowed_both, mut owned_some) = (0usize, 0usize);
        let mut tally_left = |b: &SparseCore<f64>, sk: &Sketch<f64>, k: &SparseCore<f64>| {
            let rows = align_rows_cow(sk, &b.row_ranks);
            let both = align_cols_cow(&rows, &k.row_ranks);
            if is_borrowed(&rows) && is_borrowed(&both) {
                borrowed_both += 1;
            } else {
                owned_some += 1;
            }
        };
        let mut check_left =
            |b: &SparseCore<f64>, sk: &Sketch<f64>, k: &SparseCore<f64>, what: &str| {
                tally_left(b, sk, k);
                let want = owned_left(b, sk, k);
                sketch_bits_eq(&fold_left(b, sk, k), &want, what);
                want
            };

        let mut l_env = vec![Sketch::identity(space, 0, 1)];
        for m in 0..d - 1 {
            let xm = x.core(m);
            let next = check_left(xm, &l_env[m], xm, &format!("plain left m={m}"));
            for &t in OPEN_TYPES.iter() {
                for &s in &spins {
                    let opc = open_core(xm, t, s);
                    let bo =
                        check_left(&opc, &l_env[m], xm, &format!("bo birth m={m} {t:?} {s:?}"));
                    let ko =
                        check_left(xm, &l_env[m], &opc, &format!("ko birth m={m} {t:?} {s:?}"));
                    let xn = x.core(m + 1);
                    let idn = id_core(xn, t, s);
                    let cln = close_core(xn, t, s);
                    check_left(&idn, &bo, xn, &format!("bo prop m={} {t:?} {s:?}", m + 1));
                    check_left(&cln, &bo, xn, &format!("bd2 birth m={} {t:?} {s:?}", m + 1));
                    check_left(xn, &ko, &idn, &format!("ko prop m={} {t:?} {s:?}", m + 1));
                    check_left(xn, &ko, &cln, &format!("kd2 birth m={} {t:?} {s:?}", m + 1));
                }
            }
            l_env.push(next);
        }

        let (mut rb_both, mut r_owned) = (0usize, 0usize);
        let mut check_right =
            |k: &SparseCore<f64>, sk: &Sketch<f64>, b: &SparseCore<f64>, what: &str| {
                let rows = align_rows_cow(sk, &k.col_ranks);
                let both = align_cols_cow(&rows, &b.col_ranks);
                if is_borrowed(&rows) && is_borrowed(&both) {
                    rb_both += 1;
                } else {
                    r_owned += 1;
                }
                let want = owned_right(k, sk, b);
                sketch_bits_eq(&fold_right(k, sk, b), &want, what);
                want
            };
        let mut r_env: Vec<Option<Sketch<f64>>> = vec![None; d + 1];
        r_env[d] = Some(Sketch::identity(space, d, 1));
        for m in (1..d).rev() {
            let prev = r_env[m + 1].take().expect("right env");
            let xm = x.core(m);
            r_env[m] = Some(check_right(xm, &prev, xm, &format!("plain right m={m}")));
            r_env[m + 1] = Some(prev);
        }
        for &t in OPEN_TYPES.iter() {
            for &s in &spins {
                for p in 1..d {
                    let r_plain = r_env[p + 1].as_ref().expect("plain right env");
                    let xp = x.core(p);
                    let cl = close_core(xp, t, s);
                    let mut cur =
                        check_right(&cl, r_plain, xp, &format!("rk close p={p} {t:?} {s:?}"));
                    let mut cur_b =
                        check_right(xp, r_plain, &cl, &format!("rb close p={p} {t:?} {s:?}"));
                    for site in (1..p).rev() {
                        let xs = x.core(site);
                        let idc = id_core(xs, t, s);
                        cur = check_right(&idc, &cur, xs, &format!("rk id p={p} site={site}"));
                        cur_b = check_right(xs, &cur_b, &idc, &format!("rb id p={p} site={site}"));
                    }
                }
            }
        }
        eprintln!(
            "fold_cow: left borrowed {borrowed_both} / owned {owned_some}; right borrowed \
             {rb_both} / owned {r_owned}"
        );
        assert!(
            borrowed_both > 0 && rb_both > 0,
            "vacuous: no left or no right fold took the borrowed path"
        );
        assert!(
            owned_some > 0 && r_owned > 0,
            "vacuous: no left or no right fold needed an owned alignment copy"
        );
    }
}

#[cfg(test)]
mod site_cores_tests {
    use super::{OPEN_TYPES, SiteCores, close_core, id_core, n_core, open_core, sel_close, sel_n};
    use crate::core::{Family, SparseCore};
    use crate::qn::QnSpace;
    use crate::rdm::active_spins;
    use crate::test_util::{randomized_state, seeded};
    use crate::vector::TtVector;

    fn core_bits_eq(a: &SparseCore<f64>, b: &SparseCore<f64>, what: &str) {
        assert_eq!(a.k, b.k, "{what}: site");
        assert_eq!(a.row_ranks, b.row_ranks, "{what}: row ranks");
        assert_eq!(a.col_ranks, b.col_ranks, "{what}: col ranks");
        assert_eq!(a.row_qn, b.row_qn, "{what}: row sectors");
        for f in Family::ALL {
            for &s in &a.row_qn {
                match (a.block(f, s), b.block(f, s)) {
                    (None, None) => {}
                    (Some(x), Some(y)) => {
                        assert_eq!(
                            (x.nrows(), x.ncols()),
                            (y.nrows(), y.ncols()),
                            "{what}: {f:?} shape at {s:?}"
                        );
                        for j in 0..x.ncols() {
                            for i in 0..x.nrows() {
                                assert_eq!(
                                    x[(i, j)].to_bits(),
                                    y[(i, j)].to_bits(),
                                    "{what}: {f:?} entry ({i},{j}) at {s:?}"
                                );
                            }
                        }
                    }
                    _ => panic!("{what}: {f:?} block presence differs at {s:?}"),
                }
            }
        }
    }

    #[test]
    fn site_cores_match_fresh_builds() {
        for (space, seed) in [
            (QnSpace::new(5, 3, 2), 0x51C0_0001u64),
            (QnSpace::new(4, 2, 0), 0x51C0_0002),
        ] {
            let x: TtVector<f64> = randomized_state(space, 1, 2, &mut seeded(seed));
            let d = space.d;
            let spins = active_spins(&space);
            for m in 0..d {
                let xm = x.core(m);
                let sc = SiteCores::new(xm, m, d, &spins);
                assert!(
                    std::ptr::eq(sc.sel(0), xm),
                    "sel(0) must be the plain core, m={m}"
                );
                for &s in &spins {
                    let w = format!("{space:?} m={m} {s:?}");
                    core_bits_eq(sc.n(s), &n_core(xm, s), &format!("n {w}"));
                    core_bits_eq(sc.sel(sel_n(s)), &n_core(xm, s), &format!("sel_n {w}"));
                    for &t in OPEN_TYPES.iter() {
                        if m >= 1 {
                            let fresh = close_core(xm, t, s);
                            core_bits_eq(sc.close(t, s), &fresh, &format!("close {t:?} {w}"));
                            core_bits_eq(
                                sc.sel(sel_close(t, s)),
                                &fresh,
                                &format!("sel_close {t:?} {w}"),
                            );
                        }
                        if m + 1 < d {
                            core_bits_eq(
                                sc.open(t, s),
                                &open_core(xm, t, s),
                                &format!("open {t:?} {w}"),
                            );
                        }
                        if m >= 1 && m + 1 < d {
                            core_bits_eq(sc.id(t, s), &id_core(xm, t, s), &format!("id {t:?} {w}"));
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod par_tests {
    use super::{
        OPEN_TYPES, RtpTable, SEL_N, SiteCores, fold_right, gamma2_with, gamma2_work,
        generalized_gamma2, sel_close, sel_n, set_gamma2_mirror,
    };
    use crate::core::factor::Sketch;
    use crate::qn::QnSpace;
    use crate::rdm::active_spins;
    use crate::test_util::{randomized_state, seeded};
    use crate::vector::TtVector;

    fn pool(n: usize) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon pool")
    }

    fn sketch_bits_eq(a: &Sketch<f64>, b: &Sketch<f64>, what: &str) {
        assert_eq!(a.k, b.k, "{what}: bond");
        for (s, x) in a.blocks.iter() {
            match (x.as_ref(), b.blocks[s].as_ref()) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(
                        (x.nrows(), x.ncols()),
                        (y.nrows(), y.ncols()),
                        "{what}: shape at {s:?}"
                    );
                    for j in 0..x.ncols() {
                        for i in 0..x.nrows() {
                            assert_eq!(
                                x[(i, j)].to_bits(),
                                y[(i, j)].to_bits(),
                                "{what}: entry ({i},{j}) at {s:?}"
                            );
                        }
                    }
                }
                _ => panic!("{what}: block presence differs at {s:?}"),
            }
        }
    }

    fn right_envs(x: &TtVector<f64>) -> Vec<Option<Sketch<f64>>> {
        let (space, d) = (x.space, x.space.d);
        let mut r_env: Vec<Option<Sketch<f64>>> = vec![None; d + 1];
        r_env[d] = Some(Sketch::identity(space, d, 1));
        for m in (1..d).rev() {
            let prev = r_env[m + 1].as_ref().expect("built rightward");
            r_env[m] = Some(fold_right(x.core(m), prev, x.core(m)));
        }
        r_env
    }

    fn expected_plan(
        m: usize,
        spins: &[crate::orbital::Spin],
        mirror: bool,
    ) -> Vec<(usize, usize)> {
        let n: Vec<usize> = spins.iter().map(|&s| sel_n(s)).collect();
        let c: Vec<usize> = OPEN_TYPES
            .iter()
            .flat_map(|&t| spins.iter().map(move |&s| sel_close(t, s)))
            .collect();
        let mut out = Vec::new();
        let mut pairs = |bs: &[usize], ks: &[usize], keep: &dyn Fn(usize, usize) -> bool| {
            for &b in bs {
                for &k in ks {
                    if keep(b, k) {
                        out.push((b, k));
                    }
                }
            }
        };
        let all = |_: usize, _: usize| true;
        pairs(&n, &n, &all);
        if m > 0 {
            pairs(&c, &n, &all);
            pairs(&n, &[0], &all);
            pairs(&c, &[0], &all);
            if !mirror {
                pairs(&n, &c, &all);
                pairs(&[0], &n, &all);
                pairs(&c, &c, &all);
                pairs(&[0], &c, &all);
            } else if m == 1 {
                pairs(&c, &c, &|b, k| b <= k);
            } else {
                pairs(&[0], &n, &all);
                pairs(&c, &c, &all);
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn rtp_table_is_exactly_the_planned_folds() {
        for mirror in [false, true] {
            for (space, seed) in [
                (QnSpace::new(5, 3, 2), 0xF3_7AB1u64),
                (QnSpace::new(4, 2, 0), 0xF3_7AB2),
                (QnSpace::new(4, 0, 2), 0xF3_7AB3),
            ] {
                let x: TtVector<f64> = randomized_state(space, 1, 2, &mut seeded(seed));
                let d = space.d;
                let spins = active_spins(&space);
                let s = spins.len();
                let r_env = right_envs(&x);
                for m in 0..d {
                    let what = format!("{space:?} m={m} mirror={mirror}");
                    let keys = RtpTable::plan(m, &spins, mirror);
                    let want = match (mirror, m) {
                        (_, 0) => s * s,
                        (false, _) => 9 * s * s + 6 * s,
                        (true, 1) => 5 * s * s + 4 * s,
                        (true, _) => 7 * s * s + 4 * s,
                    };
                    assert_eq!(keys.len(), want, "{what}: planned pair count");
                    let mut seen = keys.clone();
                    seen.sort_unstable();
                    seen.dedup();
                    assert_eq!(seen.len(), keys.len(), "{what}: duplicate planned pair");
                    assert_eq!(
                        seen,
                        expected_plan(m, &spins, mirror),
                        "{what}: planned pair SET"
                    );

                    let sc = SiteCores::new(x.core(m), m, d, &spins);
                    let r_next = r_env[m + 1].as_ref().expect("r_env[m+1]");
                    let serial =
                        pool(1).install(|| RtpTable::new(&sc, r_next, m, &spins, false, mirror));
                    let par =
                        pool(4).install(|| RtpTable::new(&sc, r_next, m, &spins, true, mirror));
                    for (b, k) in keys.iter().copied() {
                        let fresh = fold_right(sc.sel(k), r_next, sc.sel(b));
                        sketch_bits_eq(
                            serial.get(b, k),
                            &fresh,
                            &format!("{what} serial ({b},{k})"),
                        );
                        sketch_bits_eq(par.get(b, k), &fresh, &format!("{what} par ({b},{k})"));
                    }
                    for i in 0..SEL_N * SEL_N {
                        let planned = keys.contains(&(i / SEL_N, i % SEL_N));
                        assert_eq!(
                            serial.built[i].is_some(),
                            planned,
                            "{what}: serial slot {i}"
                        );
                        assert_eq!(par.built[i].is_some(), planned, "{what}: par slot {i}");
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "RtpTable: selection pair (0, 0) is not planned at site 1")]
    fn rtp_table_refuses_an_unplanned_pair() {
        let space = QnSpace::new(4, 2, 1);
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut seeded(0xF3_7AB4));
        let spins = active_spins(&space);
        let r_env = right_envs(&x);
        let sc = SiteCores::new(x.core(1), 1, space.d, &spins);
        let rtp = RtpTable::new(
            &sc,
            r_env[2].as_ref().expect("r_env[2]"),
            1,
            &spins,
            false,
            false,
        );
        let _ = rtp.get(0, 0);
    }

    #[test]
    #[should_panic(expected = "RtpTable: selection pair (1, 3) is not planned at site 2")]
    fn rtp_table_mirrored_refuses_a_full_only_pair() {
        let space = QnSpace::new(4, 2, 1);
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut seeded(0xF4_7AB5));
        let spins = active_spins(&space);
        let r_env = right_envs(&x);
        let sc = SiteCores::new(x.core(2), 2, space.d, &spins);
        let rtp = RtpTable::new(
            &sc,
            r_env[3].as_ref().expect("r_env[3]"),
            2,
            &spins,
            false,
            true,
        );
        let _ = rtp.get(1, 3);
    }

    #[test]
    #[should_panic(expected = "RtpTable: selection pair (4, 3) is not planned at site 1")]
    fn rtp_table_mirrored_site1_refuses_the_lower_boko_half() {
        let space = QnSpace::new(4, 2, 1);
        let x: TtVector<f64> = randomized_state(space, 1, 2, &mut seeded(0xF4_7AB6));
        let spins = active_spins(&space);
        let r_env = right_envs(&x);
        let sc = SiteCores::new(x.core(1), 1, space.d, &spins);
        let rtp = RtpTable::new(
            &sc,
            r_env[2].as_ref().expect("r_env[2]"),
            1,
            &spins,
            false,
            true,
        );
        let _ = rtp.get(4, 3);
    }

    #[test]
    fn parallel_walk_is_bitwise_the_serial_walk() {
        struct Restore {
            fork: usize,
            on: bool,
            mirror: bool,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                crate::par::set_par_min_fork_work(self.fork);
                crate::par::set_vector_par(self.on);
                set_gamma2_mirror(self.mirror);
            }
        }
        let bits_eq = |got: &[f64], want: &[f64], what: &str| {
            assert_eq!(got.len(), want.len(), "{what}: length");
            let bad = got
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(bad, 0, "{what}: {bad} cells differ from the serial walk");
        };

        let _lock = crate::par::lock_par_switches();
        let _restore = Restore {
            fork: crate::par::par_min_fork_work(),
            on: crate::par::set_vector_par(true),
            mirror: set_gamma2_mirror(true),
        };
        let mut modes_differ = 0usize;
        for (space, seed) in [
            (QnSpace::new(6, 3, 2), 0xF3_9A01u64),
            (QnSpace::new(5, 3, 2), 0xF3_9A02),
            (QnSpace::new(5, 3, 0), 0xF3_9A03),
        ] {
            let x: TtVector<f64> = randomized_state(space, 1, 2, &mut seeded(seed));
            let serial_full = pool(1).install(|| gamma2_with(&x, false, false));
            let serial_mir = pool(1).install(|| gamma2_with(&x, false, true));
            let differ = serial_full
                .iter()
                .zip(&serial_mir)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!("{space:?}: Full and Mirrored serial walks differ in {differ} cells");
            modes_differ += differ;
            for (mirror, want) in [(false, &serial_full), (true, &serial_mir)] {
                let tag = if mirror { "Mirrored" } else { "Full" };
                assert!(
                    want.iter().any(|v| *v != 0.0),
                    "{space:?} {tag}: all-zero reference (vacuous)"
                );
                for n in [1, 4, 16] {
                    let got = pool(n).install(|| gamma2_with(&x, true, mirror));
                    bits_eq(
                        &got,
                        want,
                        &format!("{space:?} {tag} gamma2_with(par) pool {n}"),
                    );
                }
                set_gamma2_mirror(mirror);
                crate::par::set_par_min_fork_work(_restore.fork);
                bits_eq(
                    &pool(1).install(|| generalized_gamma2(&x)),
                    want,
                    &format!("{space:?} {tag} public, pool 1"),
                );
                bits_eq(
                    &pool(4).install(|| generalized_gamma2(&x)),
                    want,
                    &format!("{space:?} {tag} public, pool 4, shipped floor"),
                );
                bits_eq(
                    &pool(4).install(|| crate::par::nested_scope(|| generalized_gamma2(&x))),
                    want,
                    &format!("{space:?} {tag} public, pool 4, nested"),
                );
                crate::par::set_par_min_fork_work(0);
                let got = pool(4).install(|| {
                    assert!(
                        crate::par::par_driver_fork(gamma2_work(&x)),
                        "{space:?}: the floor-0 arm would run the serial walk (vacuous)"
                    );
                    generalized_gamma2(&x)
                });
                bits_eq(
                    &got,
                    want,
                    &format!("{space:?} {tag} public, pool 4, floor 0"),
                );
            }
        }
        assert!(
            modes_differ > 0,
            "Full and Mirrored bitwise identical on every fixture — the public-entry arms \
             cannot tell whether the switch is routed"
        );
    }

    #[test]
    fn walk_is_bitwise_the_frozen_serial_walk() {
        use super::frozen_walk::frozen_generalized_gamma2;
        struct Restore {
            fork: usize,
            on: bool,
            mirror: bool,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                crate::par::set_par_min_fork_work(self.fork);
                crate::par::set_vector_par(self.on);
                set_gamma2_mirror(self.mirror);
            }
        }
        let bits_eq = |got: &[f64], want: &[f64], what: &str| {
            assert_eq!(got.len(), want.len(), "{what}: length");
            let bad = got
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                bad, 0,
                "{what}: {bad} cells differ from the frozen serial walk"
            );
        };
        let capped = |space: QnSpace, lo: usize, hi: usize, seed: u64| -> TtVector<f64> {
            use rand::RngExt;
            let mut rng = seeded(seed);
            let caps = crate::qn::max_ranks(&space);
            let d = space.d;
            let ranks: Vec<crate::qn::RankGrid> = (0..=d)
                .map(|b| {
                    let mut r = crate::qn::RankGrid::zeros(&space);
                    for s in space.bond_sectors(b) {
                        let draw = if b == 0 || b == d {
                            1
                        } else {
                            rng.random_range(lo..=hi)
                        };
                        r[s] = draw.min(caps[b][s]);
                    }
                    r
                })
                .collect();
            crate::vector::tt_randn(space, &ranks, &mut rng)
        };

        let _lock = crate::par::lock_par_switches();
        let _restore = Restore {
            fork: crate::par::par_min_fork_work(),
            on: crate::par::set_vector_par(true),
            mirror: set_gamma2_mirror(false),
        };
        let trunc = capped(QnSpace::new(6, 3, 2), 1, 2, 0xF3_F201);
        let caps6 = crate::qn::max_ranks(&trunc.space);
        assert!(
            (0..=6).any(|b| trunc
                .space
                .bond_sectors(b)
                .into_iter()
                .any(|s| trunc.ranks[b][s] < caps6[b][s])),
            "d=6 (3,2): must sit strictly below max_ranks somewhere"
        );
        let full = capped(QnSpace::new(5, 2, 2), 8, 8, 0xF3_F202);
        let caps5 = crate::qn::max_ranks(&full.space);
        assert_eq!(
            caps5.len(),
            full.ranks.len(),
            "d=5 (2,2): one cap grid per bond"
        );
        for (b, (have, cap)) in full.ranks.iter().zip(&caps5).enumerate() {
            for s in full.space.bond_sectors(b) {
                assert_eq!(have[s], cap[s], "d=5 (2,2): bond {b} {s:?} at cap");
            }
        }
        let spinless = capped(QnSpace::new(8, 4, 0), 1, 2, 0xF3_F203);
        let dn_only = capped(QnSpace::new(5, 0, 3), 1, 2, 0xF3_F204);

        for (x, label) in [
            (&trunc, "d=6 (3,2) truncated"),
            (&full, "d=5 (2,2) full rank"),
            (&spinless, "d=8 (4,0)"),
            (&dn_only, "d=5 (0,3)"),
        ] {
            let want = pool(1).install(|| frozen_generalized_gamma2(x));
            assert!(
                want.iter().any(|v| *v != 0.0),
                "{label}: all-zero reference (vacuous)"
            );
            bits_eq(
                &pool(1).install(|| gamma2_with(x, false, false)),
                &want,
                &format!("{label} gamma2_with(serial, Full)"),
            );
            for n in [1, 4, 16] {
                let got = pool(n).install(|| gamma2_with(x, true, false));
                bits_eq(&got, &want, &format!("{label} gamma2_with(par) pool {n}"));
            }
            crate::par::set_par_min_fork_work(_restore.fork);
            assert!(
                !super::GAMMA2_MIRROR.load(std::sync::atomic::Ordering::Relaxed),
                "{label}: the public arms must run with GAMMA2_MIRROR off (Full path)"
            );
            bits_eq(
                &pool(16).install(|| generalized_gamma2(x)),
                &want,
                &format!("{label} public (Full), pool 16, shipped floor"),
            );
            crate::par::set_par_min_fork_work(0);
            let got = pool(16).install(|| {
                assert!(
                    crate::par::par_driver_fork(gamma2_work(x)),
                    "{label}: the floor-0 arm would run the serial walk (vacuous)"
                );
                generalized_gamma2(x)
            });
            bits_eq(
                &got,
                &want,
                &format!("{label} public (Full), pool 16, floor 0"),
            );
        }
    }
}
