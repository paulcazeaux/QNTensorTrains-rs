// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use faer::{Mat, MatRef};
use rustc_hash::FxHashMap;

use crate::scalar::Scalar;

pub(crate) const DENSE_INDEX_MAX_KEYS: usize = 1 << 13;

const EMPTY_ROWS: u32 = 0b10 << 30;
const EMPTY_COLS: u32 = 0b11 << 30;
const LOW: u32 = (1 << 30) - 1;

#[derive(Clone, Debug)]
enum KeyIndex {
    Dense(Vec<u32>),
    Hash(FxHashMap<u32, u32>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lookup {
    Stored(usize),
    Empty(usize, usize),
    Absent,
    Unrecorded,
}

#[derive(Clone, Debug)]
pub(crate) struct BlockStore<T: Scalar> {
    keys: Vec<u32>,
    blocks: Vec<Mat<T>>,
    key_space: u32,
    index: KeyIndex,
    recording: bool,
}

impl<T: Scalar> BlockStore<T> {
    pub(crate) fn from_sorted(
        key_space: usize,
        entries: impl IntoIterator<Item = (u32, Mat<T>)>,
    ) -> Self {
        Self::build(key_space, entries, std::iter::empty(), false)
    }

    pub(crate) fn from_sorted_recording_empty(
        key_space: usize,
        entries: impl IntoIterator<Item = (u32, Mat<T>)>,
        empties: impl IntoIterator<Item = (u32, usize, usize)>,
    ) -> Self {
        Self::build(key_space, entries, empties, true)
    }

    fn build(
        key_space: usize,
        entries: impl IntoIterator<Item = (u32, Mat<T>)>,
        empties: impl IntoIterator<Item = (u32, usize, usize)>,
        recording: bool,
    ) -> Self {
        let key_space = checked_key_space(key_space);
        let mut keys = Vec::new();
        let mut blocks = Vec::new();
        for (key, m) in entries {
            assert!(
                key < key_space,
                "BlockStore: key {key} outside the key space 0..{key_space}"
            );
            if let Some(&last) = keys.last() {
                assert!(key != last, "BlockStore: duplicate key {key}");
                assert!(
                    key > last,
                    "BlockStore: key {key} after {last} — entries must be strictly increasing"
                );
            }
            assert!(
                m.nrows() * m.ncols() > 0,
                "BlockStore: key {key} holds an empty {}×{} block — only non-empty blocks are \
                 stored",
                m.nrows(),
                m.ncols()
            );
            keys.push(key);
            blocks.push(m);
        }
        assert!(
            keys.len() < EMPTY_ROWS as usize,
            "BlockStore: {} blocks exceed the index's position range",
            keys.len()
        );
        let empties: Vec<(u32, u32)> = empties
            .into_iter()
            .map(|(key, r, c)| {
                assert!(
                    key < key_space,
                    "BlockStore: empty key {key} outside the key space 0..{key_space}"
                );
                assert!(
                    r * c == 0 && r <= LOW as usize && c <= LOW as usize,
                    "BlockStore: the recorded empty slot at key {key} is {r}×{c}"
                );
                let entry = if r == 0 {
                    EMPTY_ROWS | c as u32
                } else {
                    EMPTY_COLS | r as u32
                };
                (key, entry)
            })
            .collect();
        let index = build_index(key_space, &keys, &empties);
        BlockStore {
            keys,
            blocks,
            key_space,
            index,
            recording,
        }
    }

    #[cfg(test)]
    pub(crate) fn from_unsorted(key_space: usize, mut entries: Vec<(u32, Mat<T>)>) -> Self {
        entries.sort_unstable_by_key(|&(k, _)| k);
        Self::from_sorted(key_space, entries)
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn key_space(&self) -> usize {
        self.key_space as usize
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.keys.len()
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    fn entry(&self, key: u32) -> u32 {
        match &self.index {
            KeyIndex::Dense(e) => e.get(key as usize).copied().unwrap_or(0),
            KeyIndex::Hash(map) => map.get(&key).copied().unwrap_or(0),
        }
    }

    #[inline]
    pub(crate) fn lookup(&self, key: u32) -> Lookup {
        match self.entry(key) {
            0 if self.recording => Lookup::Absent,
            0 => Lookup::Unrecorded,
            e if e < EMPTY_ROWS => Lookup::Stored(e as usize - 1),
            e if e & EMPTY_COLS == EMPTY_COLS => Lookup::Empty((e & LOW) as usize, 0),
            e => Lookup::Empty(0, (e & LOW) as usize),
        }
    }

    #[inline]
    pub(crate) fn position(&self, key: u32) -> Option<usize> {
        match self.entry(key) {
            e if e > 0 && e < EMPTY_ROWS => Some(e as usize - 1),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn records_empty(&self) -> bool {
        self.recording
    }

    pub(crate) fn recorded_empties(&self) -> Vec<(u32, usize, usize)> {
        let decode = |key: u32, e: u32| -> Option<(u32, usize, usize)> {
            if e < EMPTY_ROWS {
                None
            } else if e & EMPTY_COLS == EMPTY_COLS {
                Some((key, (e & LOW) as usize, 0))
            } else {
                Some((key, 0, (e & LOW) as usize))
            }
        };
        let mut out: Vec<(u32, usize, usize)> = match &self.index {
            KeyIndex::Dense(entries) => entries
                .iter()
                .enumerate()
                .filter_map(|(k, &e)| decode(k as u32, e))
                .collect(),
            KeyIndex::Hash(map) => map.iter().filter_map(|(&k, &e)| decode(k, e)).collect(),
        };
        out.sort_unstable_by_key(|&(k, _, _)| k);
        out
    }

    #[inline]
    pub(crate) fn get(&self, key: u32) -> Option<MatRef<'_, T>> {
        self.position(key).map(|p| self.blocks[p].as_ref())
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn get_mut(&mut self, key: u32) -> Option<faer::MatMut<'_, T>> {
        self.position(key).map(|p| self.blocks[p].as_mut())
    }

    #[inline]
    pub(crate) fn keys(&self) -> &[u32] {
        &self.keys
    }

    #[inline]
    pub(crate) fn blocks(&self) -> &[Mat<T>] {
        &self.blocks
    }

    #[inline]
    pub(crate) fn blocks_mut(&mut self) -> &mut [Mat<T>] {
        &mut self.blocks
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (u32, &Mat<T>)> + '_ {
        self.keys.iter().copied().zip(&self.blocks)
    }

    pub(crate) fn step_range(&self, step: usize, grid_len: usize) -> Range<usize> {
        let lo = (step * grid_len) as u64;
        let hi = ((step + 1) * grid_len) as u64;
        let a = self.keys.partition_point(|&k| (k as u64) < lo);
        let b = self.keys.partition_point(|&k| (k as u64) < hi);
        a..b
    }

    #[inline]
    pub(crate) fn same_keys(&self, other: &Self) -> bool {
        self.keys == other.keys
    }

    pub(crate) fn rekey(&mut self, mut f: impl FnMut(u32) -> u32) {
        let empties: Vec<(u32, usize, usize)> = self
            .recorded_empties()
            .into_iter()
            .map(|(k, r, c)| (f(k), r, c))
            .collect();
        let keys = std::mem::take(&mut self.keys);
        let blocks = std::mem::take(&mut self.blocks);
        let mut pairs: Vec<(u32, Mat<T>)> = keys.into_iter().map(&mut f).zip(blocks).collect();
        pairs.sort_unstable_by_key(|&(k, _)| k);
        *self = Self::build(self.key_space as usize, pairs, empties, self.recording);
    }

    pub(crate) fn dense_index_len(&self) -> usize {
        match &self.index {
            KeyIndex::Dense(pos) => pos.len(),
            KeyIndex::Hash(_) => 0,
        }
    }
}

fn checked_key_space(key_space: usize) -> u32 {
    u32::try_from(key_space)
        .ok()
        .filter(|&k| k < u32::MAX)
        .unwrap_or_else(|| panic!("BlockStore: key space {key_space} does not fit u32 keys"))
}

fn build_index(key_space: u32, keys: &[u32], empties: &[(u32, u32)]) -> KeyIndex {
    let collide = |key: u32| -> ! {
        panic!("BlockStore: the recorded empty slot at key {key} collides with another entry")
    };
    if key_space as usize <= DENSE_INDEX_MAX_KEYS {
        let mut entries = vec![0u32; key_space as usize];
        for (i, &k) in keys.iter().enumerate() {
            entries[k as usize] = i as u32 + 1;
        }
        for &(k, e) in empties {
            if entries[k as usize] != 0 {
                collide(k);
            }
            entries[k as usize] = e;
        }
        KeyIndex::Dense(entries)
    } else {
        let mut map =
            FxHashMap::with_capacity_and_hasher(keys.len() + empties.len(), Default::default());
        for (i, &k) in keys.iter().enumerate() {
            map.insert(k, i as u32 + 1);
        }
        for &(k, e) in empties {
            if map.insert(k, e).is_some() {
                collide(k);
            }
        }
        KeyIndex::Hash(map)
    }
}

#[cfg(test)]
#[path = "blockstore_tests.rs"]
mod tests;
