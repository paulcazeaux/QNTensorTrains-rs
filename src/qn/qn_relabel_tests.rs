// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::{QnSpace, RankGrid, Sector, SectorGrid, max_ranks};
use crate::pointgroup::{Irrep, PointGroup};

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
    )
}

fn fx_space(target: Irrep) -> QnSpace {
    QnSpace::with_point_group(4, 2, 1, fx_pg(), target)
}

fn shift(s: Sector, phi: Irrep) -> Sector {
    Sector::with_irrep(s.nup, s.ndn, s.irrep.fuse(phi))
}

fn distinct_grid(space: &QnSpace) -> RankGrid {
    let mut n = 0usize;
    SectorGrid::from_fn(space, |_| {
        n += 1;
        n
    })
}

#[test]
fn qnr_relabel_irrep_permutes_the_irrep_axis() {
    let space = fx_space(Irrep::IDENTITY);
    let g = distinct_grid(&space);
    let order = space.pg.order();
    assert!(order == 4, "fixture must exercise a genuine 4-element axis");
    let total0 = g.total();
    assert!(total0 > 0, "anti-vacuity: the fixture grid is empty");

    for p in 0..order {
        let phi = Irrep::new(p);
        let out = g.relabel_irrep(phi);
        for (s, v) in g.iter() {
            assert_eq!(
                out[s],
                g[shift(s, phi)],
                "relabel_irrep({p}): out[{s:?}] = {} but self[s+phi] = {} (self[s] = {v})",
                out[s],
                g[shift(s, phi)]
            );
        }
        assert_eq!(
            out.relabel_irrep(phi),
            g,
            "relabel_irrep({p}) is not an involution"
        );
        assert_eq!(
            out.total(),
            total0,
            "relabel_irrep({p}) changed the pooled total"
        );
        if p == 0 {
            assert_eq!(out, g, "relabel_irrep(IDENTITY) must be the identity map");
        } else {
            assert_ne!(
                out, g,
                "relabel_irrep({p}) left a per-sector-distinct grid unchanged — the permutation \
                 is not being applied"
            );
        }
    }
}

#[test]
fn qnr_relabel_irrep_moves_the_unreachable_zeros() {
    let space = fx_space(Irrep::IDENTITY);
    let caps = max_ranks(&space);
    let mut moved_somewhere = false;
    for (b, cap) in caps.iter().enumerate() {
        for p in 1..space.pg.order() {
            let phi = Irrep::new(p);
            let out = cap.relabel_irrep(phi);
            for (s, _) in cap.iter() {
                assert_eq!(
                    out[s],
                    cap[shift(s, phi)],
                    "bond {b}, phi {p}, sector {s:?}"
                );
            }
            assert_eq!(out.total(), cap.total(), "bond {b}, phi {p}: total changed");
            if out != *cap {
                moved_somewhere = true;
            }
        }
    }
    assert!(
        moved_somewhere,
        "anti-vacuity: no cap grid was moved by any non-identity relabel — the fixture has no \
         irrep structure at all"
    );
}

#[test]
fn qnr_with_target_and_same_lattice() {
    let base = fx_space(Irrep::IDENTITY);
    for p in 0..base.pg.order() {
        let phi = Irrep::new(p);
        let out = base.with_target(phi);
        assert_eq!(out.target, phi, "with_target({p}) did not set the target");
        assert_eq!((out.nup, out.ndn, out.d), (base.nup, base.ndn, base.d));
        assert_eq!(out.pg, base.pg, "with_target must not touch the site table");
        assert!(
            base.same_lattice(&out) && out.same_lattice(&base),
            "same_lattice must ignore the target (phi = {p})"
        );
        if p != 0 {
            assert_ne!(
                out, base,
                "QnSpace equality must still discriminate the target"
            );
        }
        for b in 0..=base.d {
            assert_eq!(out.bond_sectors(b), base.bond_sectors(b), "bond {b}");
        }
    }

    let pg = fx_pg();
    assert!(!base.same_lattice(&QnSpace::with_point_group(4, 1, 1, pg, Irrep::IDENTITY)));
    assert!(!base.same_lattice(&QnSpace::with_point_group(4, 2, 0, pg, Irrep::IDENTITY)));
    assert!(!base.same_lattice(&QnSpace::with_point_group(5, 2, 1, pg, Irrep::IDENTITY)));
    assert!(!base.same_lattice(&QnSpace::new(4, 2, 1)));
}

#[test]
#[should_panic]
fn qnr_with_target_rejects_an_out_of_range_irrep() {
    let base = fx_space(Irrep::IDENTITY);
    let _ = base.with_target(Irrep::new(base.pg.order()));
}
