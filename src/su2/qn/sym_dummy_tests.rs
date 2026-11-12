// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::fmt::Debug;

use crate::pointgroup::{Irrep, PointGroup};
use crate::su2::core::Family;
use crate::su2::qn::{StateSym, Su2Sector, Su2SectorGrid, Su2Space, Symmetry, max_ranks};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct DummySym;

impl Symmetry for DummySym {
    type Label = Su2Sector;
    type Ctx = Irrep;
    type Space = Su2Space;
    type Step = Family;
    type Grid<V>
        = Su2SectorGrid<V>
    where
        V: Clone + Debug + Send + Sync;
}

type D = DummySym;

fn shifted_pg_spaces() -> Vec<Su2Space> {
    let mut v = Vec::new();
    for (nbits, d, n, two_s) in [(1u8, 6usize, 4usize, 0usize), (2, 6, 4, 2), (2, 7, 5, 1)] {
        let order = 1u8 << nbits;
        let irreps: Vec<Irrep> = (0..d)
            .map(|k| Irrep::new(((k + 1) as u8) % order))
            .collect();
        let pg = PointGroup::new(nbits, &irreps);
        assert_ne!(pg.irrep_of(0), Irrep::IDENTITY);
        for ir in 0..pg.order() {
            let sp = Su2Space::with_point_group(d, n, two_s, pg, Irrep::new(ir));
            if max_ranks(&sp)[d][Su2Sector::with_irrep(n, two_s, Irrep::new(ir))] > 0 {
                v.push(sp);
            }
        }
    }
    v
}

#[test]
fn dummy_default_boundary_step_matches_state_override_on_reachable_labels() {
    let mut spaces = vec![
        Su2Space::new(6, 4, 2),
        Su2Space::new(7, 5, 3),
        Su2Space::new(5, 1, 1),
        Su2Space::new(8, 6, 0),
        Su2Space::new(4, 8, 0),
    ];
    let pg_spaces = shifted_pg_spaces();
    assert!(
        !pg_spaces.is_empty(),
        "anti-vacuity: no reachable shifted-irrep space"
    );
    spaces.extend(pg_spaces);
    for sp in spaces {
        let mr = max_ranks(&sp);
        let (mut checked, mut nontrivial_irrep) = (0usize, 0usize);
        for q in sp.bond_sectors(1) {
            if mr[1][q] == 0 {
                continue;
            }
            assert_eq!(
                <D as Symmetry>::boundary_step(&sp, q),
                <StateSym as Symmetry>::boundary_step(&sp, q),
                "{sp:?}: provided boundary_step({q:?}) drifted from the state override"
            );
            checked += 1;
            nontrivial_irrep += usize::from(q.irrep != Irrep::IDENTITY);
        }
        assert!(
            checked > 0,
            "{sp:?}: anti-vacuity — no reachable bond-1 label"
        );
        if !sp.pg.is_trivial() {
            assert!(
                nontrivial_irrep > 0,
                "{sp:?}: anti-vacuity — no reachable bond-1 label carries the site-0 irrep"
            );
        }
    }
}

#[test]
#[should_panic(expected = "is not a single-site transition")]
fn dummy_default_boundary_step_panics_on_contained_unreachable_label() {
    let sp = shifted_pg_spaces()
        .into_iter()
        .find(|sp| !sp.pg.is_trivial())
        .expect("a |G| > 1 fixture");
    let mr = max_ranks(&sp);
    let q = (0..sp.pg.order())
        .flat_map(|x| {
            [(0, 0), (1, 1), (2, 0)]
                .map(|(n, two_s)| Su2Sector::with_irrep(n, two_s, Irrep::new(x)))
        })
        .find(|&q| sp.contains(1, q) && mr[1][q] == 0)
        .expect("anti-vacuity: a contained-but-unreachable bond-1 label");
    assert_eq!(
        <StateSym as Symmetry>::boundary_step(&sp, q),
        <StateSym as Symmetry>::boundary_step(&sp, Su2Sector::new(q.n, q.two_s)),
        "the override is irrep-blind"
    );
    let _ = <D as Symmetry>::boundary_step(&sp, q);
}
