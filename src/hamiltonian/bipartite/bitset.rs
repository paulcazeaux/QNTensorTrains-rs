// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;
use std::rc::Rc;

use crate::elementaryops::shift_qn;
use crate::qn::{QnSpace, RankGrid, Sector};

use super::super::assembly::StateShift;

#[inline]
fn bits_words(n_slots: usize) -> usize {
    n_slots.div_ceil(64)
}

#[inline]
pub(super) fn plane_words(space: &QnSpace) -> usize {
    bits_words((space.nup + 1) * (space.ndn + 1))
}

#[inline]
fn plane_slot(space: &QnSpace, s: Sector) -> usize {
    s.nup * (space.ndn + 1) + s.ndn
}

#[inline]
pub(super) fn bit_slot(space: &QnSpace, s: Sector) -> usize {
    (s.irrep.index() as usize) * plane_words(space) * 64 + plane_slot(space, s)
}

#[inline]
pub(super) fn bits_set(w: &mut [u64], i: usize) {
    w[i >> 6] |= 1u64 << (i & 63);
}

#[inline]
pub(super) fn bits_or(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d |= *s;
    }
}

fn bits_shift_and_plane(
    dst: &mut [u64],
    src: &[u64],
    src_mask: &[u64],
    shift: usize,
    keep: &[u64],
) -> bool {
    let nw = dst.len();
    let word_shift = shift >> 6;
    let bit_shift = shift & 63;
    let mut any = false;
    for i in (0..nw).rev() {
        let mut v = 0u64;
        if i >= word_shift {
            let lo = i - word_shift;
            v = (src[lo] & src_mask[lo]) << bit_shift;
            if bit_shift != 0 && lo > 0 {
                v |= (src[lo - 1] & src_mask[lo - 1]) >> (64 - bit_shift);
            }
        }
        v &= keep[i];
        dst[i] = v;
        any |= v != 0;
    }
    any
}

pub(super) fn bits_shift_and(
    dst: &mut [u64],
    src: &[u64],
    src_mask: &[u64],
    shift: usize,
    keep: &[u64],
    pw: usize,
    dg: u8,
) -> bool {
    let nir = dst.len() / pw;
    debug_assert!((dg as usize) < nir, "irrep flux escapes the plane count");
    let mut any = false;
    for ir in 0..nir {
        let dir = ir ^ dg as usize;
        any |= bits_shift_and_plane(
            &mut dst[dir * pw..(dir + 1) * pw],
            &src[ir * pw..(ir + 1) * pw],
            src_mask,
            shift,
            &keep[dir * pw..(dir + 1) * pw],
        );
    }
    any
}

#[inline]
pub(super) fn bits_for_each(w: &[u64], mut f: impl FnMut(usize)) {
    for (wi, &word) in w.iter().enumerate() {
        let mut x = word;
        while x != 0 {
            let b = x.trailing_zeros() as usize;
            f((wi << 6) | b);
            x &= x - 1;
        }
    }
}

pub(super) struct AdmMask {
    pub(super) ord: Vec<bool>,
    pub(super) bits: Vec<u64>,
}

pub(super) fn adm_mask(
    cache: &mut FxHashMap<(usize, StateShift), Rc<AdmMask>>,
    space: &QnSpace,
    caps: Option<&[RankGrid]>,
    bond: usize,
    st: StateShift,
) -> Rc<AdmMask> {
    if let Some(m) = cache.get(&(bond, st)) {
        return Rc::clone(m);
    }
    let secs = space.bond_sectors(bond);
    let admissible = shift_qn(
        space,
        caps,
        bond,
        st.up.flux as i64,
        st.up.nl as i64,
        st.up.nr as i64,
        st.dn.flux as i64,
        st.dn.nl as i64,
        st.dn.nr as i64,
        st.delta,
    );
    let mut ord = vec![false; secs.len()];
    let mut bits = vec![0u64; space.pg.order() as usize * plane_words(space)];
    let mut it = admissible.iter().peekable();
    for (i, s) in secs.iter().enumerate() {
        if it.peek() == Some(&s) {
            ord[i] = true;
            bits_set(&mut bits, bit_slot(space, *s));
            it.next();
        }
    }
    debug_assert!(it.next().is_none());
    let mask = Rc::new(AdmMask { ord, bits });
    cache.insert((bond, st), Rc::clone(&mask));
    mask
}
