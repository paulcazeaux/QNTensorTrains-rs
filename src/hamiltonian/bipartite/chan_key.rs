// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pointgroup::Irrep;
use crate::scalar::Scalar;

use super::super::assembly::Flux;
use super::packed_ops::PackedOps;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ChanKind {
    Normal = 0,
    Comp = 1,
}

pub(crate) fn chan_key(fu: i8, fd: i8, delta: Irrep, kind: ChanKind, ops: PackedOps) -> u64 {
    chan_key_refined(fu, fd, delta, kind, ops, (0, 0))
}

pub(super) fn chan_key_refined(
    fu: i8,
    fd: i8,
    delta: Irrep,
    kind: ChanKind,
    ops: PackedOps,
    refine: (i8, i8),
) -> u64 {
    debug_assert!((-3..=3).contains(&fu) && (-3..=3).contains(&fd));
    debug_assert!(
        delta.index() < 8,
        "irrep index exceeds chan_key's 3-bit Δ field"
    );
    debug_assert!((0..=3).contains(&refine.0) && (0..=3).contains(&refine.1));
    let mut low = 0u64;
    for &b in ops.cre.iter().chain(ops.ann.iter()) {
        low = (low << 8) | b as u64;
    }
    (delta.index() as u64) << 61
        | (refine.0 as u64) << 59
        | (refine.1 as u64) << 57
        | ((fu + 3) as u64) << 53
        | ((fd + 3) as u64) << 49
        | (kind as u64) << 48
        | low
}

pub(crate) fn key_flux(key: u64) -> Flux {
    Flux {
        up: ((key >> 53) & 0xF) as i8 - 3,
        dn: ((key >> 49) & 0xF) as i8 - 3,
        irrep: Irrep::new(((key >> 61) & 0x7) as u8),
    }
}

pub(crate) fn key_with_ops(key: u64, ops: PackedOps) -> u64 {
    let mut low = 0u64;
    for &b in ops.cre.iter().chain(ops.ann.iter()) {
        low = (low << 8) | b as u64;
    }
    (key & !((1u64 << 48) - 1)) | low
}

pub(crate) fn key_kind(key: u64) -> ChanKind {
    if (key >> 48) & 1 == 0 {
        ChanKind::Normal
    } else {
        ChanKind::Comp
    }
}

pub(crate) fn key_ops(key: u64) -> PackedOps {
    let mut bytes = [0u8; 6];
    for (i, slot) in bytes.iter_mut().enumerate() {
        *slot = ((key >> (8 * (5 - i))) & 0xFF) as u8;
    }
    PackedOps {
        cre: [bytes[0], bytes[1], bytes[2]],
        ann: [bytes[3], bytes[4], bytes[5]],
    }
}

pub(super) fn identity_key() -> u64 {
    chan_key(
        0,
        0,
        Irrep::IDENTITY,
        ChanKind::Normal,
        PackedOps::default(),
    )
}

pub(super) fn check_edge_collision<T: Scalar>(existing: T, incoming: T) {
    debug_assert!(
        (existing - incoming).abs_s() <= 1e-12,
        "edge-weight mismatch on collision: {} vs {}",
        existing.abs_s(),
        incoming.abs_s()
    );
    let _ = (existing, incoming);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbital::{Orbital, Spin};

    #[test]
    fn chan_key_total_order() {
        let i = Irrep::IDENTITY;
        let a = chan_key(0, 0, i, ChanKind::Comp, PackedOps::default());
        let b = chan_key(1, -1, i, ChanKind::Normal, PackedOps::default());
        assert!(a < b, "flux must dominate the kind bit");
        let c = chan_key(0, 0, i, ChanKind::Normal, PackedOps::default());
        assert!(c < a, "Normal sorts before Comp at equal flux");
        let ops = PackedOps::new(&[Orbital::new(2, Spin::Up)], &[]);
        let e = chan_key(0, 0, i, ChanKind::Normal, ops);
        assert!(c < e, "the empty string sorts first");
        assert_eq!(
            key_flux(b),
            Flux {
                up: 1,
                dn: -1,
                irrep: i
            }
        );
        assert_eq!(key_ops(e), ops);
        assert_eq!(key_kind(a), ChanKind::Comp);
        let g = chan_key(
            -3,
            -3,
            Irrep::new(1),
            ChanKind::Normal,
            PackedOps::default(),
        );
        assert!(b < g, "Δ must dominate the flux fields");
        assert_eq!(key_flux(g).irrep, Irrep::new(1));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "edge-weight mismatch")]
    fn edge_collision_assert_fires() {
        check_edge_collision(1.0f64, 2.0f64);
    }
}
