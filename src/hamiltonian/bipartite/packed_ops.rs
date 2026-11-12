// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::orbital::{Orbital, Spin};
use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;

use super::super::assembly::{Op, SpinShift, StateShift};

#[inline]
fn pack(o: Orbital) -> u8 {
    (((o.site as u8) + 1) << 1) | (o.spin == Spin::Dn) as u8
}

#[inline]
pub(super) fn unpack_site(b: u8) -> usize {
    ((b >> 1) - 1) as usize
}

#[inline]
fn unpack_spin(b: u8) -> Spin {
    if b & 1 == 0 { Spin::Up } else { Spin::Dn }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub(crate) struct PackedOps {
    pub(super) cre: [u8; 3],
    pub(super) ann: [u8; 3],
}

impl PackedOps {
    pub(crate) fn new(cre: &[Orbital], ann: &[Orbital]) -> Self {
        assert!(cre.len() <= 3 && ann.len() <= 3, "operator string too long");
        let mut p = PackedOps::default();
        for (slot, o) in p.cre.iter_mut().zip(cre) {
            *slot = pack(*o);
        }
        for (slot, o) in p.ann.iter_mut().zip(ann) {
            *slot = pack(*o);
        }
        p
    }

    pub(crate) fn orbitals(&self) -> (Vec<Orbital>, Vec<Orbital>) {
        let un = |list: &[u8; 3]| -> Vec<Orbital> {
            list.iter()
                .take_while(|&&b| b != 0)
                .map(|&b| Orbital::new(unpack_site(b), unpack_spin(b)))
                .collect()
        };
        (un(&self.cre), un(&self.ann))
    }

    pub(super) fn split_applied(&self, bond: usize) -> PackedOps {
        let keep = |list: [u8; 3]| {
            let mut out = [0u8; 3];
            let mut n = 0;
            for &b in &list {
                if b != 0 && unpack_site(b) < bond {
                    out[n] = b;
                    n += 1;
                }
            }
            out
        };
        PackedOps {
            cre: keep(self.cre),
            ann: keep(self.ann),
        }
    }

    pub(super) fn split_remainder(&self, bond: usize) -> PackedOps {
        let keep = |list: [u8; 3]| {
            let mut out = [0u8; 3];
            let mut n = 0;
            for &b in &list {
                if b != 0 && unpack_site(b) >= bond {
                    out[n] = b;
                    n += 1;
                }
            }
            out
        };
        PackedOps {
            cre: keep(self.cre),
            ann: keep(self.ann),
        }
    }

    fn is_canonical(&self) -> bool {
        let ordered = |list: [u8; 3], first: Spin| {
            let orbs: Vec<(Spin, usize)> = list
                .iter()
                .take_while(|&&b| b != 0)
                .map(|&b| (unpack_spin(b), unpack_site(b)))
                .collect();
            let split = orbs.iter().take_while(|&&(sp, _)| sp == first).count();
            orbs[split..].iter().all(|&(sp, _)| sp != first)
                && orbs[..split].windows(2).all(|w| w[0].1 < w[1].1)
                && orbs[split..].windows(2).all(|w| w[0].1 < w[1].1)
        };
        ordered(self.cre, Spin::Up) && ordered(self.ann, Spin::Dn)
    }
}

pub(crate) struct SopTerm<T> {
    pub ops: PackedOps,
    pub gamma: T,
}

impl<T: Scalar> SopTerm<T> {
    pub(crate) fn new(cre: &[Orbital], ann: &[Orbital], gamma: T) -> Self {
        let ops = PackedOps::new(cre, ann);
        debug_assert!(
            ops.is_canonical(),
            "term string not in canonical spin-block order: {cre:?} {ann:?}"
        );
        SopTerm { ops, gamma }
    }
}

pub(crate) fn sort_terms<T: Scalar>(terms: &mut [SopTerm<T>]) {
    terms.sort_unstable_by_key(|t| t.ops);
    debug_assert!(
        terms.windows(2).all(|w| w[0].ops != w[1].ops),
        "duplicate (cre, ann) term string in the Phase-A sink"
    );
}

pub(super) fn term_state(pg: &PointGroup, ops: &PackedOps, bond: usize) -> StateShift {
    let (mut fu, mut nlu, mut nru) = (0i8, 0i8, 0i8);
    let (mut fd, mut nld, mut nrd) = (0i8, 0i8, 0i8);
    let mut delta = Irrep::IDENTITY;
    for &b in &ops.cre {
        if b == 0 {
            continue;
        }
        let up = b & 1 == 0;
        if unpack_site(b) < bond {
            delta = delta.fuse(pg.irrep_of(unpack_site(b)));
            if up {
                fu += 1;
                nlu += 1;
            } else {
                fd += 1;
                nld += 1;
            }
        } else if up {
            nru += 1;
        } else {
            nrd += 1;
        }
    }
    for &b in &ops.ann {
        if b != 0 && unpack_site(b) < bond {
            delta = delta.fuse(pg.irrep_of(unpack_site(b)));
            if b & 1 == 0 {
                fu -= 1;
            } else {
                fd -= 1;
            }
        }
    }
    StateShift {
        up: SpinShift {
            flux: fu,
            nl: nlu,
            nr: nru,
        },
        dn: SpinShift {
            flux: fd,
            nl: nld,
            nr: nrd,
        },
        delta,
    }
}

pub(super) fn site_ops(ops: &PackedOps, site: usize) -> (Op, Op) {
    let one = |spin_bit: u8| {
        let hit = |list: [u8; 3]| {
            list.iter()
                .any(|&b| b != 0 && unpack_site(b) == site && (b & 1) == spin_bit)
        };
        match (hit(ops.cre), hit(ops.ann)) {
            (true, true) => Op::N,
            (true, false) => Op::Adag,
            (false, true) => Op::A,
            (false, false) => Op::Id,
        }
    };
    (one(0), one(1))
}

#[inline]
fn has_cre(op: Op) -> bool {
    matches!(op, Op::Adag | Op::N)
}

#[inline]
fn has_ann(op: Op) -> bool {
    matches!(op, Op::A | Op::N)
}

#[inline]
pub(super) fn z_pack(up: Op, dn: Op) -> u8 {
    let code = |op: Op| -> u8 {
        match op {
            Op::Id => 0,
            Op::A => 1,
            Op::Adag => 2,
            Op::N => 3,
        }
    };
    (code(up) << 2) | code(dn)
}

#[inline]
pub(super) fn z_unpack(z: u8) -> (Op, Op) {
    let op = |c: u8| -> Op {
        match c {
            0 => Op::Id,
            1 => Op::A,
            2 => Op::Adag,
            _ => Op::N,
        }
    };
    (op(z >> 2), op(z & 3))
}

pub(super) const Z_ID: u8 = 0;

pub(super) fn extend_applied(mut ops: PackedOps, site: usize, zu: Op, zdn: Op) -> PackedOps {
    let insert = |list: &mut [u8; 3], byte: u8, before_spin_bit: Option<u8>| {
        let pos = match before_spin_bit {
            Some(bit) => (0..3)
                .find(|&i| list[i] == 0 || (list[i] & 1) == bit)
                .expect("packed string capacity"),
            None => (0..3).find(|&i| list[i] == 0).expect("capacity"),
        };
        assert!(list[2] == 0, "applied-string capacity exceeded");
        for i in (pos + 1..3).rev() {
            list[i] = list[i - 1];
        }
        list[pos] = byte;
    };
    if has_cre(zu) {
        insert(&mut ops.cre, pack(Orbital::new(site, Spin::Up)), Some(1));
    }
    if has_cre(zdn) {
        insert(&mut ops.cre, pack(Orbital::new(site, Spin::Dn)), None);
    }
    if has_ann(zdn) {
        insert(&mut ops.ann, pack(Orbital::new(site, Spin::Dn)), Some(0));
    }
    if has_ann(zu) {
        insert(&mut ops.ann, pack(Orbital::new(site, Spin::Up)), None);
    }
    debug_assert!(ops.is_canonical());
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn term_state_delta_counts_annihilations_too() {
        let pg = crate::pointgroup::PointGroup::new(
            2,
            &[Irrep::new(1), Irrep::new(2), Irrep::new(3), Irrep::new(0)],
        );
        let ops = PackedOps::new(&[Orbital::new(0, Spin::Up)], &[Orbital::new(2, Spin::Up)]);
        let st = term_state(&pg, &ops, 3);
        assert_eq!((st.up.flux, st.dn.flux), (0, 0));
        assert_eq!((st.up.nl, st.up.nr), (1, 0));
        assert_eq!(
            st.delta,
            pg.irrep_of(0).fuse(pg.irrep_of(2)),
            "Δ must fuse the annihilation's site irrep as well as the creation's"
        );
        assert_ne!(st.delta, Irrep::IDENTITY, "vacuous fixture");
        assert_eq!(term_state(&pg, &ops, 1).delta, pg.irrep_of(0));
        assert_eq!(term_state(&pg, &ops, 0).delta, Irrep::IDENTITY);
        let triv = crate::pointgroup::PointGroup::default();
        for b in 0..=4 {
            assert_eq!(term_state(&triv, &ops, b).delta, Irrep::IDENTITY);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "duplicate (cre, ann) term string")]
    fn duplicate_term_assert_fires() {
        let o = Orbital::new(1, Spin::Up);
        let j = Orbital::new(2, Spin::Up);
        let mut terms = vec![
            SopTerm::new(&[o], &[j], 1.0f64),
            SopTerm::new(&[o], &[j], 2.0),
        ];
        sort_terms(&mut terms);
    }

    #[test]
    fn extend_applied_matches_subsequence() {
        let cre = [Orbital::new(1, Spin::Up), Orbital::new(3, Spin::Dn)];
        let ann = [Orbital::new(0, Spin::Dn), Orbital::new(4, Spin::Up)];
        let ops = PackedOps::new(&cre, &ann);
        assert!(ops.is_canonical());
        let mut applied = ops.split_applied(0);
        for k in 0..5 {
            let (zu, zdn) = site_ops(&ops, k);
            applied = extend_applied(applied, k, zu, zdn);
            assert_eq!(applied, ops.split_applied(k + 1), "bond {}", k + 1);
        }
    }
}
