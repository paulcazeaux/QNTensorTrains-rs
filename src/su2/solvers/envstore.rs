// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) twin of [`crate::solvers::envstore`] over `Su2Sketch` / `Su2Sector` (a
//! per-layer duplicate whose mode, override and spill-byte witness are shared with the abelian
//! module).

use std::path::PathBuf;

use crate::scalar::Scalar;
use crate::solvers::envstore::{SpillMode, spill_mode};
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::Su2Space;

enum Slot<T: Scalar> {
    Empty,
    Resident(Su2Sketch<T>),
}

/// One environment stack: `d` slots (`Fᴸ[k]` at bond `k`; `Fᴿ[j]` at bond `j+1`, the sweeps'
/// convention), resident or spilled. See the module doc.
pub struct Su2EnvStore<T: Scalar> {
    space: Su2Space,
    slots: Vec<Slot<T>>,
    dir: Option<PathBuf>,
    tag: &'static str,
    written: u64,
    read: u64,
}

impl<T: Scalar> Su2EnvStore<T> {
    /// An empty store of `d` slots over `space` in the process's [`spill_mode`]. `tag` names the
    /// files (`L`/`R`).
    pub fn new(space: Su2Space, d: usize, tag: &'static str) -> Self {
        let dir = match spill_mode() {
            SpillMode::Mem => None,
        };
        Su2EnvStore {
            space,
            slots: (0..d).map(|_| Slot::Empty).collect(),
            dir,
            tag,
            written: 0,
            read: 0,
        }
    }

    /// The space every slot's environment lives in.
    pub fn space(&self) -> Su2Space {
        self.space
    }

    /// Is this store spilling to disk?
    pub fn is_spill(&self) -> bool {
        self.dir.is_some()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// `(bytes written, bytes read)` so far.
    pub fn io_bytes(&self) -> (u64, u64) {
        (self.written, self.read)
    }

    /// Every slot released (the L1 invariant after a sweep consumed its stack).
    pub fn all_empty(&self) -> bool {
        self.slots.iter().all(|s| matches!(s, Slot::Empty))
    }

    /// Slots holding an environment (resident, on disk or in flight).
    pub fn occupied(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| !matches!(s, Slot::Empty))
            .count()
    }

    /// Store the freshly built environment of slot `k` (resident).
    pub fn put(&mut self, k: usize, s: Su2Sketch<T>) {
        self.clear(k);
        self.slots[k] = Slot::Resident(s);
    }

    /// Spill the resident environment of slot `k` (no-op in memory mode, or when not resident).
    /// The write runs on a background thread that owns the `Sketch` until done.
    pub fn release(&mut self, k: usize) {
        {
            let _ = k;
        }
    }

    /// Start reading slot `k` back on a background thread (no-op unless it is on disk).
    pub fn prefetch(&mut self, k: usize) {
        {
            let _ = k;
        }
    }

    /// The environment of slot `k`, loading it if needed. Panics if the slot is empty (the same
    /// contract as the former `Option::unwrap`).
    pub fn get(&mut self, k: usize) -> &Su2Sketch<T> {
        self.make_resident(k);
        match &self.slots[k] {
            Slot::Resident(s) => s,
            _ => panic!("Su2EnvStore {}[{k}]: slot is empty", self.tag),
        }
    }

    /// Take the environment out of slot `k` (loading it if needed), leaving the slot empty.
    pub fn take(&mut self, k: usize) -> Option<Su2Sketch<T>> {
        self.make_resident(k);
        match std::mem::replace(&mut self.slots[k], Slot::Empty) {
            Slot::Resident(s) => {
                self.remove_file(k);
                Some(s)
            }
            _ => None,
        }
    }

    /// Empty slot `k` without loading it (its last reader is done).
    pub fn drop_slot(&mut self, k: usize) {
        self.clear(k);
    }

    fn make_resident(&mut self, k: usize) {
        match std::mem::replace(&mut self.slots[k], Slot::Empty) {
            Slot::Empty => {}
            Slot::Resident(s) => self.slots[k] = Slot::Resident(s),
        }
    }

    fn remove_file(&self, k: usize) {
        let _ = k;
    }

    fn clear(&mut self, k: usize) {
        match std::mem::replace(&mut self.slots[k], Slot::Empty) {
            Slot::Empty | Slot::Resident(_) => {}
        }
    }
}

impl<T: Scalar> Drop for Su2EnvStore<T> {
    fn drop(&mut self) {
        for k in 0..self.slots.len() {
            self.clear(k);
        }
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_dir(dir);
        }
    }
}
