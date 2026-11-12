// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pointgroup::{Irrep, PointGroup};
use crate::su2::qn::{
    LabelTables, StateSym, Su2Sector, Su2Space, SymGrid, SymSpace, SymStep, Symmetry,
};

fn reference_slots<S: Symmetry>(space: &S::Space, k: usize) -> Vec<(S::Step, S::Label, S::Label)> {
    let g = space.site_ctx(k);
    let grid: Vec<S::Label> = <S::Grid<()> as SymGrid<()>>::filled(space, ())
        .iter()
        .map(|(l, _)| l)
        .collect();
    let mut out = Vec::new();
    for &f in S::STEPS {
        for &s in &grid {
            if let Some(t) = f.step_apply(s, g)
                && space.admissible(k, s)
                && space.admissible(k + 1, t)
            {
                out.push((f, s, t));
            }
        }
    }
    out
}

fn reference_caps<S: Symmetry>(space: &S::Space, rho: usize) -> Vec<Vec<usize>> {
    let (l, r) = S::pathcounts(space);
    (0..=space.sites())
        .map(|b| {
            <S::Grid<usize> as SymGrid<usize>>::from_fn(space, |s| {
                l[b][s].min(r[b][s].saturating_mul(rho as u64)) as usize
            })
            .values()
            .to_vec()
        })
        .collect()
}

fn check_space<S: Symmetry>(space: &S::Space, what: &str) -> usize {
    let d = space.sites();
    let tables = LabelTables::<S>::new(space);
    for b in 0..=d {
        assert_eq!(
            tables.labels(b),
            &space.bond_labels(b)[..],
            "{what}: labels({b}) ≠ bond_labels({b})"
        );
    }
    let (l_ref, r_ref) = S::pathcounts(space);
    let (l_got, r_got) = tables.pathcounts();
    assert_eq!(l_got.len(), l_ref.len(), "{what}: left pathcount bonds");
    assert_eq!(r_got.len(), r_ref.len(), "{what}: right pathcount bonds");
    for b in 0..l_ref.len() {
        assert_eq!(
            l_got[b].values(),
            l_ref[b].values(),
            "{what}: left pathcounts at bond {b}"
        );
        assert_eq!(
            r_got[b].values(),
            r_ref[b].values(),
            "{what}: right pathcounts at bond {b}"
        );
    }
    let caps = S::max_ranks(space);
    for rho in [1usize, 2, 3] {
        let want = reference_caps::<S>(space, rho);
        let open = tables.max_ranks_open(rho);
        let provided = S::max_ranks_open(space, rho);
        assert_eq!(open.len(), d + 1, "{what}: max_ranks_open({rho}) bonds");
        for b in 0..=d {
            let at = |g: &S::Grid<usize>| g.values().to_vec();
            assert_eq!(
                at(&open[b]),
                want[b],
                "{what}: tables.max_ranks_open({rho}), bond {b}"
            );
            assert_eq!(
                at(&provided[b]),
                want[b],
                "{what}: S::max_ranks_open({rho}), bond {b}"
            );
            if rho == 1 {
                assert_eq!(
                    at(&tables.max_ranks()[b]),
                    want[b],
                    "{what}: max_ranks(), bond {b}"
                );
                assert_eq!(at(&caps[b]), want[b], "{what}: S::max_ranks, bond {b}");
            }
        }
    }
    assert!(
        matches!(tables.max_ranks_open(1), std::borrow::Cow::Borrowed(_)),
        "{what}: max_ranks_open(1) borrows the memoized caps"
    );
    let len = <S::Grid<()> as SymGrid<()>>::storage_len(space);
    for b in 0..=d {
        assert_eq!(
            tables.storage_labels(b),
            &crate::su2::qn::storage_labels::<S>(space, b)[..],
            "{what}: storage_labels({b}) ≠ the grid scan"
        );
        let labels = tables.labels(b);
        for i in 0..len {
            let l = <S::Grid<()> as SymGrid<()>>::storage_label(space, i);
            let want = labels.iter().position(|&x| x == l);
            assert_eq!(tables.pos(b, l), want, "{what}: pos({b}, {l:?})");
        }
    }
    let mut slots = 0usize;
    for k in 0..d {
        let want = reference_slots::<S>(space, k);
        let got = tables.slots(k);
        assert_eq!(got.len(), want.len(), "{what}: slot count at site {k}");
        for (i, (x, y)) in got.iter().zip(&want).enumerate() {
            assert!(
                x.0.index() == y.0.index() && x.1 == y.1 && x.2 == y.2,
                "{what}: site {k} slot {i}: {x:?} ≠ the fill order's {y:?}"
            );
        }
        slots += want.len();
    }
    slots
}

#[test]
fn label_tables_equal_the_functions_on_state_spaces() {
    let mut slots = 0usize;
    for (l, n, two_s) in [
        (4usize, 3usize, 1usize),
        (4, 4, 0),
        (4, 4, 2),
        (5, 3, 3),
        (5, 4, 0),
        (5, 6, 2),
        (6, 5, 1),
        (6, 6, 0),
        (6, 7, 3),
        (7, 6, 2),
        (7, 7, 1),
        (8, 7, 1),
        (8, 8, 0),
        (8, 9, 5),
    ] {
        let space = Su2Space::new(l, n, two_s);
        slots += check_space::<StateSym>(&space, &format!("state ({l}, {n}, {two_s})"));
        let tables = LabelTables::<StateSym>::new(&space);
        for b in 0..=l {
            for off in [Su2Sector::new(n + 1, 0), Su2Sector::new(0, n + 1)] {
                assert_eq!(tables.pos(b, off), None, "off-grid {off:?} at bond {b}");
            }
        }
    }
    assert!(slots > 0, "anti-vacuity: no slot");
}

#[test]
fn label_tables_equal_the_functions_on_point_group_state_spaces() {
    let pg = |nbits: u8, d: usize| {
        let order = 1u8 << nbits;
        let irreps: Vec<Irrep> = (0..d)
            .map(|k| Irrep::new((k as u8 * 3 + 1) % order))
            .collect();
        PointGroup::new(nbits, &irreps)
    };
    let mut slots = 0usize;
    for (l, n, two_s, nbits, target) in [
        (6usize, 4usize, 2usize, 2u8, 1u8),
        (6, 4, 0, 1, 0),
        (5, 5, 1, 3, 3),
        (7, 6, 2, 2, 2),
    ] {
        let space = Su2Space::with_point_group(l, n, two_s, pg(nbits, l), Irrep::new(target));
        let what = format!("state ({l}, {n}, {two_s}) |G| = {}", 1 << nbits);
        slots += check_space::<StateSym>(&space, &what);
        let tables = LabelTables::<StateSym>::new(&space);
        let off = Su2Sector::with_irrep(0, 0, Irrep::new(1 << nbits));
        for b in 0..=l {
            assert_eq!(
                tables.pos(b, off),
                None,
                "{what}: off-grid irrep at bond {b}"
            );
        }
    }
    assert!(slots > 0, "anti-vacuity: no slot");
}
