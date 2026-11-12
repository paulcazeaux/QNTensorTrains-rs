// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;

use crate::scalar::Scalar;

use super::automaton::EdgeKey;

pub(crate) const SRC_NONE: u32 = u32::MAX;

pub(crate) const SPILL: u32 = 1 << 31;

#[inline]
pub(crate) fn median_weight<T: Scalar>(w: T, cw: f64) -> T {
    w * T::from_f64(cw)
}

#[inline]
pub(crate) fn alpha<T: Scalar>(w_edge: T, neg: bool, fr: f64) -> T {
    w_edge * (if neg { T::from_f64(-1.0) } else { T::one() }) * T::from_f64(fr)
}

#[inline]
pub(crate) fn run_value<T: Scalar>(run: &[(u32, u32)], gamma: &[T], factors: &[Su2Factor]) -> T {
    let mut s = T::zero();
    for &(term, fid) in run {
        let f = &factors[fid as usize];
        s += if term == SRC_NONE {
            alpha(T::one(), f.neg, f.fr)
        } else {
            alpha(median_weight(gamma[term as usize], f.cw), f.neg, f.fr)
        };
    }
    s
}

#[inline]
fn references_changed(run: &[(u32, u32)], changed: &[u64]) -> bool {
    run.iter()
        .any(|&(term, _)| term != SRC_NONE && term_changed(changed, term))
}

pub(crate) fn frozen_run_cancels(run: &[(u32, u32)], factors: &[Su2Factor]) -> bool {
    for (i, &(term, _)) in run.iter().enumerate() {
        if run[..i].iter().any(|&(t, _)| t == term) {
            continue;
        }
        let (mut s, mut pmax) = (0.0f64, 0.0f64);
        for &(t, fid) in &run[i..] {
            if t != term {
                continue;
            }
            let f = &factors[fid as usize];
            debug_assert_eq!(
                t == SRC_NONE,
                f.transport,
                "a transport contribution carries a transport factor, a median one a median factor"
            );
            let jw = if f.neg { -1.0 } else { 1.0 };
            let p = if t == SRC_NONE {
                jw * f.fr
            } else {
                (f.cw * jw) * f.fr
            };
            s += p;
            pmax = pmax.max(p.abs());
        }
        if s.abs() > 16.0 * f64::EPSILON * pmax {
            return false;
        }
    }
    true
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Src {
    Median { term: u32, cw: f64 },
    Transport,
}

pub(crate) trait Recorder<T: Scalar> {
    const ON: bool;
    fn term(&mut self, w: T) -> u32;
    fn appended(&mut self, key: EdgeKey, src: Src);
    fn src(&self, key: EdgeKey, ci: usize) -> Src;
    fn contributions(&self, key: EdgeKey) -> usize;
    fn word(&mut self, src: Src, neg: bool, fr: f64) -> (u32, u32);
    fn frozen_factors(&self) -> Option<&[Su2Factor]>;
    fn site(&mut self, runs: Vec<Su2RunSrc>, debris_drops: usize);
}

impl<T: Scalar> Recorder<T> for () {
    const ON: bool = false;
    #[inline(always)]
    fn term(&mut self, _: T) -> u32 {
        0
    }
    #[inline(always)]
    fn appended(&mut self, _: EdgeKey, _: Src) {}
    #[inline(always)]
    fn src(&self, _: EdgeKey, _: usize) -> Src {
        Src::Transport
    }
    #[inline(always)]
    fn contributions(&self, _: EdgeKey) -> usize {
        0
    }
    #[inline(always)]
    fn word(&mut self, _: Src, _: bool, _: f64) -> (u32, u32) {
        (SRC_NONE, 0)
    }
    #[inline(always)]
    fn frozen_factors(&self) -> Option<&[Su2Factor]> {
        None
    }
    #[inline(always)]
    fn site(&mut self, _: Vec<Su2RunSrc>, _: usize) {}
}

pub(crate) struct BuildRecorder<T: Scalar> {
    gamma: Vec<T>,
    head: FxHashMap<EdgeKey, Src>,
    ovf: FxHashMap<EdgeKey, Vec<Src>>,
    factors: Vec<Su2Factor>,
    fid: FxHashMap<(Option<u64>, bool, u64), u32>,
    sites: Vec<Vec<Su2RunSrc>>,
    frozen: bool,
    debris_drops: usize,
}

impl<T: Scalar> BuildRecorder<T> {
    pub(crate) fn new() -> Self {
        BuildRecorder {
            gamma: Vec::new(),
            head: FxHashMap::default(),
            ovf: FxHashMap::default(),
            factors: Vec::new(),
            fid: FxHashMap::default(),
            sites: Vec::new(),
            frozen: false,
            debris_drops: 0,
        }
    }

    pub(crate) fn new_frozen() -> Self {
        BuildRecorder {
            frozen: true,
            ..Self::new()
        }
    }

    pub(crate) fn debris_drops(&self) -> usize {
        self.debris_drops
    }

    pub(crate) fn into_parts(self) -> (Vec<T>, Vec<Su2Factor>, Vec<Vec<Su2RunSrc>>) {
        (self.gamma, self.factors, self.sites)
    }
}

impl<T: Scalar> Recorder<T> for BuildRecorder<T> {
    const ON: bool = true;

    fn term(&mut self, w: T) -> u32 {
        let o = self.gamma.len();
        assert!(
            o < SRC_NONE as usize,
            "recording: the Phase-A term ordinal overflows its u32 word"
        );
        self.gamma.push(w);
        o as u32
    }

    fn appended(&mut self, key: EdgeKey, src: Src) {
        match self.head.entry(key) {
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(src);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                self.ovf.entry(key).or_default().push(src)
            }
        }
    }

    fn src(&self, key: EdgeKey, ci: usize) -> Src {
        if ci == 0 {
            self.head[&key]
        } else {
            self.ovf[&key][ci - 1]
        }
    }

    fn contributions(&self, key: EdgeKey) -> usize {
        usize::from(self.head.contains_key(&key)) + self.ovf.get(&key).map_or(0, Vec::len)
    }

    fn word(&mut self, src: Src, neg: bool, fr: f64) -> (u32, u32) {
        let (term, cw) = match src {
            Src::Median { term, cw } => (term, Some(cw)),
            Src::Transport => (SRC_NONE, None),
        };
        let next = self.factors.len();
        let fid = *self
            .fid
            .entry((cw.map(f64::to_bits), neg, fr.to_bits()))
            .or_insert_with(|| {
                assert!(
                    next < SPILL as usize,
                    "recording: the fixed-factor table overflows its 31-bit fid"
                );
                self.factors.push(Su2Factor {
                    cw: cw.unwrap_or(0.0),
                    fr,
                    neg,
                    transport: cw.is_none(),
                });
                next as u32
            });
        (term, fid)
    }

    fn frozen_factors(&self) -> Option<&[Su2Factor]> {
        self.frozen.then_some(self.factors.as_slice())
    }

    fn site(&mut self, runs: Vec<Su2RunSrc>, debris_drops: usize) {
        self.sites.push(runs);
        self.debris_drops += debris_drops;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Su2Factor {
    pub(crate) cw: f64,
    pub(crate) fr: f64,
    pub(crate) neg: bool,
    pub(crate) transport: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Su2RunSrc {
    inline: Vec<(u32, u32)>,
    spill: Vec<(u32, u32)>,
}

impl Su2RunSrc {
    pub(crate) fn push_run(&mut self, run: &[(u32, u32)]) {
        match run {
            [] => panic!("recording: a stored entry with no contribution"),
            [one] => self.inline.push(*one),
            _ => {
                let start = u32::try_from(self.spill.len()).expect("spill list overflows u32");
                let len = u32::try_from(run.len())
                    .ok()
                    .filter(|&l| l < SPILL)
                    .expect("a run overflows the 31-bit spill length");
                self.inline.push((start, SPILL | len));
                self.spill.extend_from_slice(run);
            }
        }
    }

    pub(crate) fn slots(&self) -> usize {
        self.inline.len()
    }

    pub(super) fn run(&self, p: usize) -> &[(u32, u32)] {
        let (a, b) = self.inline[p];
        if b & SPILL == 0 {
            std::slice::from_ref(&self.inline[p])
        } else {
            let a = a as usize;
            &self.spill[a..a + (b & !SPILL) as usize]
        }
    }
}

pub(crate) const BLOCK: usize = 64;

pub(crate) const LF_SPILLED: u8 = u8::MAX;

pub(crate) const LOCAL_FIDS: usize = 254;

const FID_FALLBACK: u8 = u8::MAX;

pub(crate) const FID_TRANSPORT: u32 = 1 << 31;

const NARROW_PERM_SLOTS: usize = 1 << 16;

#[inline]
fn term_changed(changed: &[u64], term: u32) -> bool {
    (changed[(term >> 6) as usize] >> (term & 63)) & 1 != 0
}

fn fid_word(term: u32, fid: u32, factors: &[Su2Factor]) -> u32 {
    assert!(
        fid < FID_TRANSPORT,
        "recording: a fid overflows its 31-bit word"
    );
    debug_assert_eq!(
        factors[fid as usize].transport,
        term == SRC_NONE,
        "a transport contribution carries a transport factor, a median one a median factor"
    );
    if term == SRC_NONE {
        fid | FID_TRANSPORT
    } else {
        fid
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Su2CellSrc {
    pub(super) lf: Box<[u8]>,
    pub(super) t16: Box<[u16]>,
    pub(super) perm16: Box<[u16]>,
    pub(super) aux: Box<[u32]>,
    pub(super) spill: Box<[(u32, u32)]>,
    pub(super) n_fids: u8,
    pub(super) spill_at: u32,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Su2LayoutCensus {
    pub(crate) blocks: usize,
    pub(crate) escape_blocks: usize,
    pub(crate) wide_perm: bool,
    pub(crate) fid_fallback: bool,
    pub(crate) local_fids: usize,
    pub(crate) spilled_runs: usize,
    pub(crate) transport_singles: usize,
}

type DecodedRun<'a> = (Option<(u32, u32)>, &'a [(u32, u32)]);

struct View<'a> {
    lf: &'a [u8],
    t16: &'a [u16],
    blocks: &'a [u32],
    bitmap: &'a [u32],
    fids: &'a [u32],
    fallback: bool,
    wide: bool,
    perm16: &'a [u16],
    perm32: &'a [u32],
    esc: &'a [u32],
    spill_slots: &'a [u32],
    spill_starts: &'a [u32],
    spill: &'a [(u32, u32)],
}

impl<'a> View<'a> {
    fn of(s: &'a Su2CellSrc) -> Self {
        let n = s.lf.len();
        let nb = n.div_ceil(BLOCK);
        let fallback = s.n_fids == FID_FALLBACK;
        let wide = n > NARROW_PERM_SLOTS;
        let (blocks, rest) = s.aux.split_at(nb);
        let (bitmap, rest) = rest.split_at(nb.div_ceil(32));
        let (fids, rest) = rest.split_at(if fallback { n } else { usize::from(s.n_fids) });
        let (perm32, rest) = rest.split_at(if wide { n } else { 0 });
        let (esc, spill_idx) = rest.split_at(s.spill_at as usize - (s.aux.len() - rest.len()));
        let (spill_slots, spill_starts) = spill_idx.split_at(spill_idx.len() / 2);
        View {
            lf: &s.lf,
            t16: &s.t16,
            blocks,
            bitmap,
            fids,
            fallback,
            wide,
            perm16: &s.perm16,
            perm32,
            esc,
            spill_slots,
            spill_starts,
            spill: &s.spill,
        }
    }

    #[inline]
    fn is_escape(&self, b: usize) -> bool {
        (self.bitmap[b >> 5] >> (b & 31)) & 1 != 0
    }

    #[inline]
    fn fid_word(&self, p: usize, l: u8) -> u32 {
        if self.fallback {
            self.fids[p]
        } else {
            self.fids[usize::from(l)]
        }
    }

    #[inline]
    fn term(&self, p: usize) -> u32 {
        let b = p / BLOCK;
        let w = self.blocks[b];
        if self.is_escape(b) {
            self.esc[w as usize + p % BLOCK]
        } else {
            w + u32::from(self.t16[p])
        }
    }

    #[inline]
    fn spill_rank(&self, p: usize) -> usize {
        self.spill_slots.partition_point(|&s| (s as usize) < p)
    }

    #[inline]
    fn spill_run(&self, k: usize) -> &'a [(u32, u32)] {
        let lo = self.spill_starts[k] as usize;
        let hi = self
            .spill_starts
            .get(k + 1)
            .map_or(self.spill.len(), |&s| s as usize);
        &self.spill[lo..hi]
    }

    fn run(&self, p: usize) -> DecodedRun<'a> {
        let l = self.lf[p];
        if l == LF_SPILLED {
            let k = self.spill_rank(p);
            debug_assert_eq!(self.spill_slots[k] as usize, p, "spill index out of step");
            (None, self.spill_run(k))
        } else {
            let g = self.fid_word(p, l);
            let one = if g & FID_TRANSPORT != 0 {
                (SRC_NONE, g & !FID_TRANSPORT)
            } else {
                (self.term(p), g)
            };
            (Some(one), &[])
        }
    }
}

impl Su2CellSrc {
    pub(crate) fn new(runs: Su2RunSrc, csc_perm: Vec<u32>, factors: &[Su2Factor]) -> Self {
        let n = runs.slots();
        assert_eq!(
            n,
            csc_perm.len(),
            "recording: run provenance and CSC permutation cover different slot counts"
        );
        let as_u32 = |x: usize| u32::try_from(x).expect("recording: a cell index overflows u32");
        assert!(
            u32::try_from(n).is_ok(),
            "recording: a cell's slot count overflows u32"
        );
        let nb = n.div_ceil(BLOCK);
        let mut lf = vec![0u8; n];
        let mut t16 = vec![0u16; n];
        let mut blocks = vec![0u32; nb];
        let mut bitmap = vec![0u32; nb.div_ceil(32)];
        let mut table: Vec<u32> = Vec::new();
        let mut fallback = false;
        let mut esc: Vec<u32> = Vec::new();
        let (mut spill_slots, mut spill_starts) = (Vec::new(), Vec::new());
        let mut spill: Vec<(u32, u32)> = Vec::new();
        let median = |p: usize| match *runs.run(p) {
            [(t, _)] if t != SRC_NONE => Some(t),
            _ => None,
        };
        for b in 0..nb {
            let (lo, hi) = (b * BLOCK, ((b + 1) * BLOCK).min(n));
            let mut span: Option<(u32, u32)> = None;
            for (l, p) in lf[lo..hi].iter_mut().zip(lo..hi) {
                match *runs.run(p) {
                    [(term, fid)] => {
                        let w = fid_word(term, fid, factors);
                        if !fallback {
                            match table.iter().position(|&x| x == w) {
                                Some(i) => *l = i as u8,
                                None if table.len() < LOCAL_FIDS => {
                                    *l = table.len() as u8;
                                    table.push(w);
                                }
                                None => fallback = true,
                            }
                        }
                        if term != SRC_NONE {
                            span = Some(
                                span.map_or((term, term), |(a, z)| (a.min(term), z.max(term))),
                            );
                        }
                    }
                    ref run => {
                        *l = LF_SPILLED;
                        spill_slots.push(as_u32(p));
                        spill_starts.push(as_u32(spill.len()));
                        spill.extend_from_slice(run);
                    }
                }
            }
            match span {
                None => {}
                Some((min, max)) if max - min <= u32::from(u16::MAX) => {
                    blocks[b] = min;
                    for (o, p) in t16[lo..hi].iter_mut().zip(lo..hi) {
                        if let Some(t) = median(p) {
                            *o = (t - min) as u16;
                        }
                    }
                }
                Some(_) => {
                    bitmap[b >> 5] |= 1 << (b & 31);
                    blocks[b] = as_u32(esc.len());
                    esc.extend((lo..hi).map(|p| median(p).unwrap_or(0)));
                }
            }
        }
        let fids = if fallback {
            for l in lf.iter_mut().filter(|l| **l != LF_SPILLED) {
                *l = 0;
            }
            (0..n)
                .map(|p| match *runs.run(p) {
                    [(term, fid)] => fid_word(term, fid, factors),
                    _ => 0,
                })
                .collect()
        } else {
            table
        };
        let n_fids = if fallback {
            FID_FALLBACK
        } else {
            fids.len() as u8
        };
        let (perm16, perm32): (Vec<u16>, Vec<u32>) = if n <= NARROW_PERM_SLOTS {
            let narrow =
                |&p: &u32| u16::try_from(p).expect("a narrow cell's CSR slot overflows u16");
            (csc_perm.iter().map(narrow).collect(), Vec::new())
        } else {
            (Vec::new(), csc_perm)
        };
        let mut aux = Vec::with_capacity(
            nb + bitmap.len()
                + fids.len()
                + perm32.len()
                + esc.len()
                + spill_slots.len()
                + spill_starts.len(),
        );
        aux.extend_from_slice(&blocks);
        aux.extend_from_slice(&bitmap);
        aux.extend_from_slice(&fids);
        aux.extend_from_slice(&perm32);
        aux.extend_from_slice(&esc);
        let spill_at = as_u32(aux.len());
        aux.extend_from_slice(&spill_slots);
        aux.extend_from_slice(&spill_starts);
        Su2CellSrc {
            lf: lf.into_boxed_slice(),
            t16: t16.into_boxed_slice(),
            perm16: perm16.into_boxed_slice(),
            aux: aux.into_boxed_slice(),
            spill: spill.into_boxed_slice(),
            n_fids,
            spill_at,
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn run(&self, p: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        let (one, words) = View::of(self).run(p);
        one.into_iter().chain(words.iter().copied())
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn csc_to_csr(&self, q: usize) -> usize {
        if self.slots() > NARROW_PERM_SLOTS {
            View::of(self).perm32[q] as usize
        } else {
            usize::from(self.perm16[q])
        }
    }

    pub(crate) fn slots(&self) -> usize {
        self.lf.len()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn payload_len_bytes(&self) -> usize {
        std::mem::size_of_val(&*self.lf)
            + std::mem::size_of_val(&*self.t16)
            + std::mem::size_of_val(&*self.perm16)
            + std::mem::size_of_val(&*self.aux)
            + std::mem::size_of_val(&*self.spill)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn layout_census(&self) -> Su2LayoutCensus {
        let v = View::of(self);
        let transport_singles = (0..self.slots())
            .filter(|&p| {
                let l = v.lf[p];
                l != LF_SPILLED && v.fid_word(p, l) & FID_TRANSPORT != 0
            })
            .count();
        Su2LayoutCensus {
            blocks: v.blocks.len(),
            escape_blocks: v.bitmap.iter().map(|w| w.count_ones() as usize).sum(),
            wide_perm: v.wide,
            fid_fallback: v.fallback,
            local_fids: if v.fallback { 0 } else { v.fids.len() },
            spilled_runs: v.spill_slots.len(),
            transport_singles,
        }
    }

    pub(crate) fn refill_csr<T: Scalar>(
        &self,
        vals: &mut [T],
        base: usize,
        gamma: &[T],
        factors: &[Su2Factor],
        changed: &[u64],
        written: &mut [u8],
    ) -> usize {
        assert_eq!(
            vals.len(),
            written.len(),
            "refill: the written-mask window is not the value window"
        );
        let v = View::of(self);
        let end = base + vals.len();
        let mut k = v.spill_rank(base);
        let mut n = 0;
        let mut lo = base;
        while lo < end {
            let b = lo / BLOCK;
            let hi = ((b + 1) * BLOCK).min(end);
            let (escape, w) = (v.is_escape(b), v.blocks[b]);
            for p in lo..hi {
                let l = v.lf[p];
                let x = if l == LF_SPILLED {
                    let run = v.spill_run(k);
                    k += 1;
                    references_changed(run, changed).then(|| run_value(run, gamma, factors))
                } else {
                    let g = v.fid_word(p, l);
                    if g & FID_TRANSPORT != 0 {
                        None
                    } else {
                        let t = if escape {
                            v.esc[w as usize + p % BLOCK]
                        } else {
                            w + u32::from(v.t16[p])
                        };
                        term_changed(changed, t).then(|| run_value(&[(t, g)], gamma, factors))
                    }
                };
                written[p - base] = u8::from(x.is_some());
                if let Some(x) = x {
                    vals[p - base] = x;
                    n += 1;
                }
            }
            lo = hi;
        }
        n
    }

    pub(crate) fn refill_csc<T: Scalar>(
        &self,
        vals: &mut [T],
        base: usize,
        csr: &[T],
        written: &[u8],
    ) -> usize {
        let v = View::of(self);
        let mut n = 0;
        let mut copy = |x: &mut T, p: usize| {
            if written[p] != 0 {
                *x = csr[p];
                n += 1;
            }
        };
        if v.wide {
            for (x, &p) in vals.iter_mut().zip(&v.perm32[base..]) {
                copy(x, p as usize);
            }
        } else {
            for (x, &p) in vals.iter_mut().zip(&v.perm16[base..]) {
                copy(x, usize::from(p));
            }
        }
        n
    }

    pub(crate) fn hash_into<H: std::hash::Hasher>(&self, h: &mut H) {
        let n = self.slots();
        h.write_u64(n as u64);
        for p in 0..n {
            let len = u32::try_from(self.run(p).count()).expect("a run overflows u32");
            h.write_u32(len);
            for (term, fid) in self.run(p) {
                h.write_u32(term);
                h.write_u32(fid);
            }
        }
        for q in 0..n {
            h.write_u32(u32::try_from(self.csc_to_csr(q)).expect("a CSR slot overflows u32"));
        }
    }
}

#[cfg(test)]
impl Su2CellSrc {
    pub(super) fn is_escape_block(&self, b: usize) -> bool {
        View::of(self).is_escape(b)
    }

    pub(super) fn block_word_at(&self, b: usize) -> usize {
        assert!(
            b < self.slots().div_ceil(BLOCK),
            "block {b} out of the cell"
        );
        b
    }

    pub(super) fn fid_entry_at(&self, i: usize) -> Option<usize> {
        let v = View::of(self);
        (!v.fallback && i < v.fids.len()).then(|| v.blocks.len() + v.bitmap.len() + i)
    }

    pub(super) fn escape_word_at(&self, p: usize) -> usize {
        let v = View::of(self);
        let b = p / BLOCK;
        assert!(v.is_escape(b), "slot {p} is not in an escape block");
        let esc_at = self.spill_at as usize - v.esc.len();
        esc_at + v.blocks[b] as usize + p % BLOCK
    }
}

impl crate::mem::Footprint for Su2CellSrc {
    fn bytes(&self) -> u64 {
        (self.payload_len_bytes() + std::mem::size_of::<Self>()) as u64
    }
}
