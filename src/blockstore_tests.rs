// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeMap;

use faer::Mat;

use super::{BlockStore, DENSE_INDEX_MAX_KEYS};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn block(key: u32, rng: &mut Rng) -> Mat<f64> {
    let (r, c) = (1 + rng.below(4) as usize, 1 + rng.below(4) as usize);
    Mat::from_fn(r, c, |i, j| {
        key as f64 + 0.25 * i as f64 + 0.0625 * j as f64
    })
}

fn random_store(
    key_space: usize,
    fill: f64,
    rng: &mut Rng,
) -> (BlockStore<f64>, BTreeMap<u32, Mat<f64>>) {
    let mut model = BTreeMap::new();
    let want = ((key_space as f64 * fill) as usize).max(1);
    while model.len() < want {
        let k = rng.below(key_space as u64) as u32;
        let b = block(k, rng);
        model.entry(k).or_insert(b);
    }
    let store = BlockStore::from_sorted(key_space, model.iter().map(|(&k, m)| (k, m.clone())));
    (store, model)
}

fn same_bits(a: faer::MatRef<'_, f64>, b: faer::MatRef<'_, f64>) -> bool {
    a.nrows() == b.nrows()
        && a.ncols() == b.ncols()
        && (0..a.ncols())
            .all(|j| (0..a.nrows()).all(|i| a[(i, j)].to_bits() == b[(i, j)].to_bits()))
}

fn check_against_model(store: &BlockStore<f64>, model: &BTreeMap<u32, Mat<f64>>, rng: &mut Rng) {
    assert_eq!(store.len(), model.len());
    assert_eq!(
        store.keys(),
        model.keys().copied().collect::<Vec<_>>().as_slice()
    );
    for ((k, m), (mk, mm)) in store.iter().zip(model.iter()) {
        assert_eq!(k, *mk);
        assert!(same_bits(m.as_ref(), mm.as_ref()), "key {k}: bits");
    }
    let probes: Vec<u32> = if store.key_space() <= 1 << 14 {
        (0..store.key_space() as u32).collect()
    } else {
        (0..4096)
            .map(|_| rng.below(store.key_space() as u64) as u32)
            .chain(model.keys().copied())
            .collect()
    };
    for k in probes {
        match model.get(&k) {
            Some(m) => {
                let p = store.position(k).expect("stored key found");
                assert_eq!(store.keys()[p], k);
                assert!(same_bits(store.get(k).unwrap(), m.as_ref()));
            }
            None => {
                assert!(store.position(k).is_none(), "key {k}: phantom block");
                assert!(store.get(k).is_none());
            }
        }
    }
}

#[test]
fn matches_the_model_on_both_index_variants() {
    let mut rng = Rng(0x0C0F_FEE5_EED5_1234);
    for &(key_space, fill) in &[
        (37usize, 0.4),
        (DENSE_INDEX_MAX_KEYS, 0.05),
        (DENSE_INDEX_MAX_KEYS + 1, 0.05),
        (78_608, 0.01),
        (574_992, 0.002),
    ] {
        let (store, model) = random_store(key_space, fill, &mut rng);
        let dense = key_space <= DENSE_INDEX_MAX_KEYS;
        assert_eq!(
            store.dense_index_len(),
            if dense { key_space } else { 0 },
            "index variant at key space {key_space}"
        );
        check_against_model(&store, &model, &mut rng);
        check_against_model(&store.clone(), &model, &mut rng);
    }
}

#[test]
fn step_ranges_partition_the_store_by_step() {
    let mut rng = Rng(0x5EED_0000_0000_0001);
    for &(steps, grid_len) in &[(4usize, 64usize), (16, 4913), (4, 1)] {
        let (store, model) = random_store(steps * grid_len, 0.1, &mut rng);
        let mut seen = 0;
        for step in 0..steps {
            let r = store.step_range(step, grid_len);
            let want: Vec<u32> = model
                .keys()
                .copied()
                .filter(|&k| k as usize / grid_len == step)
                .collect();
            assert_eq!(&store.keys()[r.clone()], want.as_slice(), "step {step}");
            assert_eq!(r.start, seen, "ranges are contiguous and in step order");
            seen = r.end;
        }
        assert_eq!(seen, store.len(), "the steps cover the store");
    }
}

#[test]
fn in_place_mutation_reaches_the_lookup_and_the_slice() {
    let mut rng = Rng(0xABCD_EF01_2345_6789);
    for &key_space in &[200usize, 100_000] {
        let (mut store, mut model) = random_store(key_space, 0.05, &mut rng);
        let k = *model.keys().nth(model.len() / 2).unwrap();
        store.get_mut(k).unwrap()[(0, 0)] = -1.5;
        model.get_mut(&k).unwrap()[(0, 0)] = -1.5;
        for (m, (_, mm)) in store.blocks_mut().iter_mut().zip(model.iter_mut()) {
            m[(0, 0)] += 1.0;
            mm[(0, 0)] += 1.0;
        }
        check_against_model(&store, &model, &mut rng);
    }
}

#[test]
fn rekey_moves_blocks_and_keeps_key_order() {
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    let (steps, planes, per) = (4usize, 8usize, 16usize);
    let len = planes * per;
    let f = |k: u32| {
        let (s, g) = (k as usize / len, k as usize % len);
        let (p, i) = (g / per, g % per);
        (s * len + (p ^ 5) * per + i) as u32
    };
    let (mut store, model) = random_store(steps * len, 0.2, &mut rng);
    store.rekey(f);
    let moved: BTreeMap<u32, Mat<f64>> = model.iter().map(|(&k, m)| (f(k), m.clone())).collect();
    check_against_model(&store, &moved, &mut rng);
    store.rekey(f);
    check_against_model(&store, &model, &mut rng);
}

#[test]
fn from_unsorted_equals_from_sorted_and_same_keys_compares_key_sets() {
    let mut rng = Rng(0x2468_1357_ACE0_BDF1);
    let (store, model) = random_store(5_000, 0.03, &mut rng);
    let mut shuffled: Vec<(u32, Mat<f64>)> = model.iter().map(|(&k, m)| (k, m.clone())).collect();
    for i in (1..shuffled.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        shuffled.swap(i, j);
    }
    let other = BlockStore::from_unsorted(5_000, shuffled);
    assert!(store.same_keys(&other));
    check_against_model(&other, &model, &mut rng);
    let mut fewer = model.clone();
    fewer.pop_last();
    let smaller = BlockStore::from_sorted(5_000, fewer);
    assert!(!store.same_keys(&smaller));
    assert!(
        BlockStore::<f64>::from_sorted(5_000, std::iter::empty()).is_empty(),
        "an empty store"
    );
}

#[test]
#[should_panic(expected = "strictly increasing")]
fn refuses_keys_out_of_order() {
    let _ = BlockStore::from_sorted(10, [(3, Mat::<f64>::zeros(1, 1)), (2, Mat::zeros(1, 1))]);
}

#[test]
#[should_panic(expected = "duplicate key")]
fn refuses_a_duplicate_key() {
    let _ = BlockStore::from_unsorted(
        10,
        vec![(3, Mat::<f64>::zeros(1, 1)), (3, Mat::zeros(2, 1))],
    );
}

#[test]
#[should_panic(expected = "empty 0×3 block")]
fn refuses_an_empty_block() {
    let _ = BlockStore::from_sorted(10, [(3, Mat::<f64>::zeros(0, 3))]);
}

#[test]
#[should_panic(expected = "outside the key space")]
fn refuses_a_key_outside_the_key_space() {
    let _ = BlockStore::from_sorted(10, [(10, Mat::<f64>::zeros(1, 1))]);
}

#[test]
#[should_panic(expected = "does not fit u32 keys")]
fn refuses_a_key_space_beyond_u32() {
    let _ = BlockStore::<f64>::from_sorted(u32::MAX as usize, std::iter::empty());
}

#[test]
#[should_panic(expected = "duplicate key")]
fn rekey_refuses_a_collision() {
    let mut rng = Rng(7);
    let (mut store, _) = random_store(64, 0.5, &mut rng);
    store.rekey(|_| 1);
}

fn recording_store(key_space: usize) -> (BlockStore<f64>, Vec<(u32, usize, usize)>) {
    let mut rng = Rng(11);
    let entries: Vec<(u32, Mat<f64>)> = (0..key_space as u32)
        .filter(|k| k % 3 == 0)
        .map(|k| (k, block(k, &mut rng)))
        .collect();
    let empties: Vec<(u32, usize, usize)> = (0..key_space as u32)
        .filter(|k| k % 3 == 1)
        .map(|k| match (k / 3) % 3 {
            0 => (k, 0, 1 + (k as usize % 5)),
            1 => (k, 2 + (k as usize % 3), 0),
            _ => (k, 0, 0),
        })
        .collect();
    let store = BlockStore::from_sorted_recording_empty(key_space, entries, empties.clone());
    (store, empties)
}

#[test]
fn a_recording_store_classifies_every_key_and_rekey_moves_its_empty_slots() {
    use super::Lookup;
    for key_space in [37, DENSE_INDEX_MAX_KEYS + 1] {
        let (store, empties) = recording_store(key_space);
        assert!(store.records_empty());
        assert_eq!(store.recorded_empties(), empties, "key space {key_space}");
        for k in 0..key_space as u32 {
            let want = match k % 3 {
                0 => Lookup::Stored(k as usize / 3),
                1 => {
                    let &(_, r, c) = empties.iter().find(|e| e.0 == k).unwrap();
                    Lookup::Empty(r, c)
                }
                _ => Lookup::Absent,
            };
            assert_eq!(store.lookup(k), want, "key space {key_space}, key {k}");
            assert_eq!(store.position(k), (k % 3 == 0).then_some(k as usize / 3));
        }
        let plain = BlockStore::from_sorted(key_space, store.iter().map(|(k, m)| (k, m.clone())));
        assert!(!plain.records_empty() && plain.recorded_empties().is_empty());
        assert_eq!(plain.lookup(1), Lookup::Unrecorded);
        assert_eq!(plain.lookup(3), Lookup::Stored(1));
        let flip = |k: u32| key_space as u32 - 1 - k;
        let mut moved = store.clone();
        moved.rekey(flip);
        for k in 0..key_space as u32 {
            let (a, b) = (store.lookup(k), moved.lookup(flip(k)));
            match (a, b) {
                (Lookup::Stored(p), Lookup::Stored(q)) => {
                    assert!(same_bits(
                        store.blocks()[p].as_ref(),
                        moved.blocks()[q].as_ref()
                    ));
                }
                (x, y) => assert_eq!(x, y, "key {k} moved to {}", flip(k)),
            }
        }
        moved.rekey(flip);
        assert_eq!(moved.recorded_empties(), empties);
        assert_eq!(moved.keys(), store.keys());
    }
}

#[test]
#[should_panic(expected = "is 1×2")]
fn a_recording_store_refuses_a_nonempty_empty_slot() {
    let _ = BlockStore::<f64>::from_sorted_recording_empty(8, Vec::new(), vec![(3, 1, 2)]);
}

#[test]
#[should_panic(expected = "collides with another entry")]
fn a_recording_store_refuses_an_empty_slot_on_a_stored_key() {
    let mut rng = Rng(3);
    let entries = vec![(4u32, block(4, &mut rng))];
    let _ = BlockStore::<f64>::from_sorted_recording_empty(8, entries, vec![(4, 0, 2)]);
}

use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{QnSpace, SectorGrid};
use crate::su2::qn::{Su2SectorGrid, Su2Space, SymGrid};

fn pg(nbits: u8, d: usize) -> PointGroup {
    let order = 1u8 << nbits;
    let irreps: Vec<Irrep> = (0..d)
        .map(|k| Irrep::new((k as u8 * 3 + 1) % order))
        .collect();
    PointGroup::new(nbits, &irreps)
}

#[test]
fn su2_state_storage_index_inverts_the_grid_order() {
    for space in [
        Su2Space::new(6, 4, 0),
        Su2Space::new(5, 3, 1),
        Su2Space::with_point_group(6, 4, 2, pg(2, 6), Irrep::new(1)),
    ] {
        let g = Su2SectorGrid::<()>::filled(&space, ());
        let len = <Su2SectorGrid<()> as SymGrid<()>>::storage_len(&space);
        assert_eq!(len, g.values().len(), "{space:?}");
        for (i, (l, _)) in g.iter().enumerate() {
            assert_eq!(
                <Su2SectorGrid<()> as SymGrid<()>>::storage_index(&space, l),
                i
            );
            assert_eq!(
                <Su2SectorGrid<()> as SymGrid<()>>::storage_label(&space, i),
                l
            );
        }
    }
}

#[test]
fn abelian_storage_index_inverts_the_grid_order() {
    for space in [
        QnSpace::new(6, 3, 2),
        QnSpace::with_point_group(6, 2, 3, pg(2, 6), Irrep::new(3)),
    ] {
        let g = SectorGrid::<()>::from_fn(&space, |_| ());
        assert_eq!(SectorGrid::<()>::storage_len(&space), g.values().len());
        for (i, (s, _)) in g.iter().enumerate() {
            assert_eq!(SectorGrid::<()>::storage_index(&space, s), i);
            assert_eq!(SectorGrid::<()>::storage_label(&space, i), s);
        }
    }
}

use faer::linalg::matmul::matmul;
use faer::{Accum, MatRef, Par};

use crate::core::{Family, SparseCore};
use crate::qn::{RankGrid, Sector};
use crate::su2::core::{Family as Su2Family, Su2Core};
use crate::su2::qn::{Su2RankGrid, Su2Sector, SymSpace, SymStep};
use crate::test_util::{randomized_state, seeded};
use crate::vector::TtVector;

#[derive(Default)]
struct Views([usize; 4]);

impl Views {
    fn check(&mut self, v: MatRef<'_, f64>, (r, c): (usize, usize), what: &str) {
        assert_eq!((v.nrows(), v.ncols()), (r, c), "{what}: shape");
        if r * c > 0 {
            self.0[0] += 1;
            return;
        }
        self.0[match (r > 0, c > 0) {
            (true, false) => 1,
            (false, true) => 2,
            _ => 3,
        }] += 1;
        let z = Mat::<f64>::zeros(r, c);
        assert_eq!(v.row_stride(), 1, "{what}: row stride");
        assert_eq!(
            v.row_stride(),
            z.as_ref().row_stride(),
            "{what}: Mat::zeros' row stride"
        );
        assert!(v.try_as_col_major().is_some(), "{what}: not column-major");
        let w = Mat::<f64>::from_fn(c, 2, |i, j| 1.0 + i as f64 + 0.5 * j as f64);
        let mut out = Mat::<f64>::from_fn(r, 2, |i, j| 0.25 + i as f64 - j as f64);
        let before = out.clone();
        matmul(out.as_mut(), Accum::Add, v, w.as_ref(), 1.0, Par::Seq);
        assert!(
            same_bits(out.as_ref(), before.as_ref()),
            "{what}: not a zero factor"
        );
    }

    fn assert_all_kinds(&self, what: &str) {
        assert!(
            self.0.iter().all(|&n| n > 0),
            "{what}: anti-vacuity — views by kind (non-empty, r×0, 0×c, 0×0) {:?}",
            self.0
        );
    }
}

#[test]
fn abelian_block_views_follow_the_accessor_contract() {
    let mut views = Views::default();
    let g4 = pg(2, 6);
    let target = g4.irrep_of_sites(0..2).fuse(g4.irrep_of_sites(0..3));
    for space in [
        QnSpace::new(6, 3, 2),
        QnSpace::with_point_group(6, 2, 3, g4, target),
    ] {
        let mut rng = seeded(0x5704_AB5E ^ space.pg.order() as u64);
        let mut x: TtVector<f64> = randomized_state(space, 0, 3, &mut rng);
        let off = [
            Sector::new(space.nup + 1, 0),
            Sector::new(0, space.ndn + 1),
            Sector::with_irrep(0, 0, Irrep::new(space.pg.order())),
        ];
        let len = SectorGrid::<()>::storage_len(&space);
        for c in &mut x.cores {
            let (k, g) = (c.k, space.pg.irrep_of(c.k));
            for f in Family::ALL {
                for s in off {
                    assert!(c.block(f, s).is_none(), "{f:?} at off-grid {s:?}");
                    assert!(c.block_mut(f, s).is_none(), "{f:?} at off-grid {s:?}");
                }
                for i in 0..len {
                    let s = SectorGrid::<()>::storage_label(&space, i);
                    let t = f.apply(s, g);
                    let what = format!("site {k}, {f:?} at {s:?}");
                    if !(space.contains(k, s) && space.contains(k + 1, t)) {
                        assert!(c.block(f, s).is_none(), "{what}: inadmissible");
                        assert!(c.block_mut(f, s).is_none(), "{what}: inadmissible");
                        continue;
                    }
                    let shape = (c.row_ranks[s], c.col_ranks[t]);
                    views.check(c.block(f, s).expect("admissible"), shape, &what);
                    let m = c.block_mut(f, s).expect("admissible");
                    assert_eq!((m.nrows(), m.ncols()), shape, "{what}: mutable shape");
                    assert_eq!(m.row_stride(), 1, "{what}: mutable row stride");
                }
            }
        }
    }
    views.assert_all_kinds("abelian");
}

fn abelian_core_with_an_empty_row() -> SparseCore<f64> {
    let space = QnSpace::new(4, 1, 1);
    let (mut rr, mut cr) = (RankGrid::zeros(&space), RankGrid::zeros(&space));
    for s in space.bond_sectors(1) {
        rr[s] = 1;
    }
    rr[Sector::new(1, 0)] = 0;
    for s in space.bond_sectors(2) {
        cr[s] = 1;
    }
    let c = SparseCore::<f64>::zeros(space, 1, &rr, &cr);
    assert!(c.check_invariants());
    assert!(!c.store().is_empty() && c.block(Family::Empty, Sector::new(1, 0)).is_some());
    c
}

#[test]
#[should_panic(expected = "has wrong shape")]
fn abelian_invariants_refuse_a_stored_block_of_the_wrong_shape() {
    let mut c = abelian_core_with_an_empty_row();
    c.col_ranks[Sector::new(1, 1)] = 2;
    c.check_invariants();
}

#[test]
#[should_panic(expected = "has wrong shape")]
fn abelian_invariants_refuse_a_missing_nonempty_block() {
    let mut c = abelian_core_with_an_empty_row();
    c.row_ranks[Sector::new(1, 0)] = 1;
    c.check_invariants();
}

#[test]
fn abelian_accessor_answers_the_recorded_empty_shape() {
    let mut c = abelian_core_with_an_empty_row();
    let b = c
        .block(Family::Empty, Sector::new(1, 0))
        .expect("admissible");
    assert_eq!((b.nrows(), b.ncols()), (0, 1));
    c.row_ranks[Sector::new(1, 0)] = 1;
    let b = c
        .block(Family::Empty, Sector::new(1, 0))
        .expect("admissible");
    assert_eq!(
        (b.nrows(), b.ncols()),
        (0, 1),
        "the recorded shape, not the edited ranks'"
    );
    assert!(
        c.block(Family::Up, Sector::new(1, 0)).is_none(),
        "inadmissible"
    );
    assert!(
        c.block(Family::Empty, Sector::new(2, 0)).is_none(),
        "off the grid"
    );
}

#[test]
#[should_panic(expected = "has wrong shape")]
fn abelian_invariants_refuse_a_reshaped_empty_slot() {
    let space = QnSpace::new(4, 1, 1);
    let (mut rr, mut cr) = (RankGrid::zeros(&space), RankGrid::zeros(&space));
    for s in space.bond_sectors(1) {
        rr[s] = 1;
    }
    for s in space.bond_sectors(2) {
        cr[s] = 1;
    }
    cr[Sector::new(1, 0)] = 0;
    cr[Sector::new(1, 1)] = 0;
    let mut c = SparseCore::<f64>::zeros(space, 1, &rr, &cr);
    assert!(c.check_invariants());
    c.row_ranks[Sector::new(1, 0)] = 2;
    c.check_invariants();
}

#[test]
#[should_panic(expected = "spurious block")]
fn abelian_invariants_refuse_a_spurious_block() {
    let mut c = abelian_core_with_an_empty_row();
    let len = SectorGrid::<()>::storage_len(&c.space);
    let at = SectorGrid::<()>::storage_index(&c.space, Sector::new(1, 0));
    let mut entries: Vec<(u32, Mat<f64>)> = c
        .store()
        .keys()
        .iter()
        .copied()
        .zip(c.store().blocks().iter().cloned())
        .collect();
    entries.push(((Family::Up as usize * len + at) as u32, Mat::zeros(1, 1)));
    *c.store_mut() = BlockStore::from_unsorted(4 * len, entries);
    c.check_invariants();
}

fn su2_core_with_an_empty_row() -> Su2Core<f64> {
    let space = Su2Space::new(4, 2, 0);
    let (mut rr, mut cr) = (Su2RankGrid::zeros(&space), Su2RankGrid::zeros(&space));
    for l in space.bond_labels(1) {
        rr[l] = 1;
    }
    rr[Su2Sector::new(1, 1)] = 0;
    for l in space.bond_labels(2) {
        cr[l] = 1;
    }
    let c = Su2Core::<f64>::zeros(space, 1, &rr, &cr);
    assert!(c.check_invariants());
    assert!(c.block(Su2Family::Empty, Su2Sector::new(1, 1)).is_some());
    c
}

#[test]
#[should_panic(expected = "has wrong shape")]
fn su2_invariants_refuse_a_stored_block_of_the_wrong_shape() {
    let mut c = su2_core_with_an_empty_row();
    c.col_ranks[Su2Sector::new(1, 1)] = 2;
    c.check_invariants();
}

#[test]
#[should_panic(expected = "has wrong shape")]
fn su2_invariants_refuse_a_missing_nonempty_block() {
    let mut c = su2_core_with_an_empty_row();
    c.row_ranks[Su2Sector::new(1, 1)] = 1;
    c.check_invariants();
}

#[test]
#[should_panic(expected = "is not stored")]
fn su2_accessor_refuses_a_missing_nonempty_block() {
    let mut c = su2_core_with_an_empty_row();
    c.row_ranks[Su2Sector::new(1, 1)] = 1;
    let _ = c.block(Su2Family::Empty, Su2Sector::new(1, 1));
}

#[test]
#[should_panic(expected = "spurious block")]
fn su2_invariants_refuse_a_spurious_block() {
    let mut c = su2_core_with_an_empty_row();
    let len = <Su2SectorGrid<()> as SymGrid<()>>::storage_len(&c.space);
    let at = <Su2SectorGrid<()> as SymGrid<()>>::storage_index(&c.space, Su2Sector::new(0, 0));
    let mut entries: Vec<(u32, Mat<f64>)> = c
        .store()
        .keys()
        .iter()
        .copied()
        .zip(c.store().blocks().iter().cloned())
        .collect();
    entries.push(((Su2Family::Dn.index() * len + at) as u32, Mat::zeros(1, 1)));
    *c.store_mut() = BlockStore::from_unsorted(4 * len, entries);
    c.check_invariants();
}
