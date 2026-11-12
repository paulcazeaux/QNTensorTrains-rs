// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! DEAS start: a train whose VALUE is the Hartree–Fock determinant and whose RIGHT FRAMES are a
//! priority-selected set of determinants, the environment a first left→right sweep reads.

use crate::core::{Family, SparseCore};
use crate::ordering::{assert_permutation, inverse_permutation};
use crate::qn::{Pooled, QnSpace, RankGrid, Sector, max_ranks};
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// Orbital indices by single-orbital entropy `s1` DESCENDING ([`f64::total_cmp`]), ties by index
/// ascending: the active-space order DEAS extends along, most entangled orbital first.
pub fn active_space_order(s1: &[f64]) -> Vec<usize> {
    for (i, v) in s1.iter().enumerate() {
        assert!(
            v.is_finite(),
            "active_space_order: s1[{i}] = {v} is not finite"
        );
    }
    let mut idx: Vec<usize> = (0..s1.len()).collect();
    idx.sort_by(|&a, &b| s1[b].total_cmp(&s1[a]).then(a.cmp(&b)));
    idx
}

/// DEAS start for the HF determinant `(occ_up, occ_dn)` in the point-group-free space
/// `QnSpace::new(d, nup, ndn)` with the counts read off the flags — the [`deas_start_in`] twin of
/// [`crate::vector::tt_state`].
pub fn deas_start<T: Scalar>(
    occ_up: &[bool],
    occ_dn: &[bool],
    as_order: &[usize],
    chi: Pooled,
) -> TtVector<T> {
    let d = occ_up.len();
    assert_eq!(occ_dn.len(), d, "occupation vectors must share length d");
    let nup = occ_up.iter().filter(|&&o| o).count();
    let ndn = occ_dn.iter().filter(|&&o| o).count();
    deas_start_in(QnSpace::new(d, nup, ndn), occ_up, occ_dn, as_order, chi)
}

/// DEAS start in a given [`QnSpace`] (point group via `space.pg`) — the [`deas_start`] sibling,
/// as [`crate::vector::tt_state_in`] is to `tt_state`. See the module doc for the construction.
pub fn deas_start_in<T: Scalar>(
    space: QnSpace,
    occ_up: &[bool],
    occ_dn: &[bool],
    as_order: &[usize],
    chi: Pooled,
) -> TtVector<T> {
    let d = space.d;
    assert_eq!(
        occ_up.len(),
        d,
        "deas_start_in: occ_up has length {} but d = {d}",
        occ_up.len()
    );
    assert_eq!(
        occ_dn.len(),
        d,
        "deas_start_in: occ_dn has length {} but d = {d}",
        occ_dn.len()
    );
    assert!(d >= 1, "deas_start_in: d = 0 has no sites");
    assert_eq!(
        as_order.len(),
        d,
        "deas_start_in: as_order has length {} but d = {d}",
        as_order.len()
    );
    assert_permutation(as_order, d);
    let cap = if chi.is_none() { usize::MAX } else { chi.0 };
    let nup = occ_up.iter().filter(|&&o| o).count();
    let ndn = occ_dn.iter().filter(|&&o| o).count();
    assert_eq!(
        (nup, ndn),
        (space.nup, space.ndn),
        "deas_start_in: HF determinant occupies ({nup}, {ndn}) electrons, space wants ({}, {})",
        space.nup,
        space.ndn
    );
    let hf: Vec<Family> = (0..d)
        .map(|k| Family::from_occupation(occ_up[k], occ_dn[k]))
        .collect();
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    let s = hf.iter().enumerate().fold(Sector::new(0, 0), |s, (k, f)| {
        f.apply(s, space.pg.irrep_of(k))
    });
    assert_eq!(
        s, last,
        "deas_start_in: HF determinant's total irrep {:?} differs from the space target {:?}",
        s.irrep, space.target
    );

    let pos = inverse_permutation(as_order);
    let caps = max_ranks(&space);
    let frames = select_frames(&space, &hf, &pos, &caps, cap);

    let mut slots: Vec<Vec<usize>> = Vec::with_capacity(d + 1);
    let mut ranks: Vec<RankGrid> = Vec::with_capacity(d + 1);
    let mut r0 = RankGrid::zeros(&space);
    r0[Sector::new(0, 0)] = 1;
    ranks.push(r0);
    slots.push(Vec::new());
    for fb in &frames[1..] {
        let mut cnt = RankGrid::zeros(&space);
        let mut sl = Vec::with_capacity(fb.len());
        for c in fb {
            sl.push(cnt[c.sector]);
            cnt[c.sector] += 1;
        }
        ranks.push(cnt);
        slots.push(sl);
    }

    let mut cores = Vec::with_capacity(d);
    let s0 = Sector::new(0, 0);
    let mut core0 = SparseCore::zeros(space, 0, &ranks[0], &ranks[1]);
    debug_assert_eq!(hf[0].apply(s0, space.pg.irrep_of(0)), frames[1][0].sector);
    *core0
        .block_mut(hf[0], s0)
        .expect("deas_start_in: the HF block of core 0 is admissible")
        .get_mut(0, slots[1][0]) = T::one();
    cores.push(core0);
    for k in 1..d {
        let mut core = SparseCore::zeros(space, k, &ranks[k], &ranks[k + 1]);
        for (c, &slot) in frames[k].iter().zip(&slots[k]) {
            debug_assert_eq!(
                c.family.apply(c.sector, space.pg.irrep_of(k)),
                frames[k + 1][c.tail].sector
            );
            *core
                .block_mut(c.family, c.sector)
                .expect("deas_start_in: a frame's block is admissible")
                .get_mut(slot, slots[k + 1][c.tail]) = T::one();
        }
        cores.push(core);
    }

    let mut x = TtVector::new(space, cores);
    x.ortho = Some(0);
    debug_assert!(x.check());
    x
}

struct Frame {
    sector: Sector,
    family: Family,
    tail: usize,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    level: Option<usize>,
    n_dev: usize,
    tie: Vec<u8>,
}

fn select_frames(
    space: &QnSpace,
    hf: &[Family],
    pos: &[usize],
    caps: &[RankGrid],
    cap: usize,
) -> Vec<Vec<Frame>> {
    let d = space.d;
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    let mut frames: Vec<Vec<Frame>> = (0..=d).map(|_| Vec::new()).collect();
    frames[d].push(Frame {
        sector: last,
        family: Family::Empty,
        tail: 0,
    });
    let mut keys: Vec<Key> = vec![Key {
        level: None,
        n_dev: 0,
        tie: vec![0; d],
    }];
    for b in (1..d).rev() {
        let g = space.pg.irrep_of(b);
        let mut cand: Vec<(Key, Frame)> = Vec::with_capacity(4 * frames[b + 1].len());
        for (j, (c1, k1)) in frames[b + 1].iter().zip(&keys).enumerate() {
            for f in Family::ALL {
                let Some(s) = f.unapply(c1.sector, g) else {
                    continue;
                };
                if caps[b][s] == 0 {
                    continue;
                }
                let dev = f != hf[b];
                let mut tie = k1.tie.clone();
                tie[pos[b]] = f as u8;
                let key = Key {
                    level: if dev {
                        k1.level.max(Some(pos[b]))
                    } else {
                        k1.level
                    },
                    n_dev: k1.n_dev + usize::from(dev),
                    tie,
                };
                cand.push((
                    key,
                    Frame {
                        sector: s,
                        family: f,
                        tail: j,
                    },
                ));
            }
        }
        cand.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        debug_assert!(cand.windows(2).all(|w| w[0].0 < w[1].0));
        let mut cnt = RankGrid::zeros(space);
        let mut kept_f = Vec::new();
        let mut kept_k = Vec::new();
        for (k, c) in cand {
            if kept_f.len() == cap {
                break;
            }
            if cnt[c.sector] < caps[b][c.sector] {
                cnt[c.sector] += 1;
                kept_f.push(c);
                kept_k.push(k);
            }
        }
        debug_assert!(
            kept_k[0].level.is_none() && kept_f[0].family == hf[b] && kept_f[0].tail == 0
        );
        frames[b] = kept_f;
        keys = kept_k;
    }
    frames
}
