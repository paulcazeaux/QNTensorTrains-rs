// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The environment-stack store behind the two-site MALS sweeps (memory campaign, lever L6,
//! opt-in, memory mode by default).

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::core::factor::Sketch;
use crate::qn::QnSpace;
use crate::scalar::Scalar;

/// Where a store keeps the bonds it is not using.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpillMode {
    /// Everything resident (the stable behaviour).
    Mem,
}

static OVERRIDE: Mutex<Option<SpillMode>> = Mutex::new(None);
static SPILL_WRITTEN: AtomicU64 = AtomicU64::new(0);

/// Process-wide bytes every store has spilled so far (an instrument/test witness).
pub fn spill_bytes_written() -> u64 {
    SPILL_WRITTEN.load(Ordering::Relaxed)
}

/// The mode a new store takes: the test/instrument override if set, else `QNTT_ENV_SPILL`.
pub fn spill_mode() -> SpillMode {
    if let Some(m) = OVERRIDE.lock().expect("envstore override poisoned").clone() {
        return m;
    }
    use std::sync::LazyLock;
    static FROM_ENV: LazyLock<SpillMode> = LazyLock::new(|| {
        let set = std::env::var_os("QNTT_ENV_SPILL").filter(|v| !v.is_empty());
        {
            if set.is_some() {
                eprintln!(
                    "qntt: QNTT_ENV_SPILL is set but this build has no spill backend — \
                     environment stacks stay in memory"
                );
            }
            SpillMode::Mem
        }
    });
    FROM_ENV.clone()
}

#[cfg(test)]
pub(crate) static OVERRIDE_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Force a mode for every store created afterwards (`None` = back to the env/default resolution).
/// Returns the previous override.
pub fn set_spill_override(mode: Option<SpillMode>) -> Option<SpillMode> {
    std::mem::replace(
        &mut *OVERRIDE.lock().expect("envstore override poisoned"),
        mode,
    )
}

enum Slot<T: Scalar> {
    Empty,
    Resident(Sketch<T>),
}

/// One environment stack: `d` slots (`Fᴸ[k]` at bond `k`; `Fᴿ[j]` at bond `j+1`, the sweeps'
/// convention), resident or spilled. See the module doc.
pub struct EnvStore<T: Scalar> {
    space: QnSpace,
    slots: Vec<Slot<T>>,
    dir: Option<PathBuf>,
    tag: &'static str,
    written: u64,
    read: u64,
}

impl<T: Scalar> EnvStore<T> {
    /// An empty store of `d` slots over `space` in the process's [`spill_mode`]. `tag` names the
    /// files (`L`/`R`).
    pub fn new(space: QnSpace, d: usize, tag: &'static str) -> Self {
        let dir = match spill_mode() {
            SpillMode::Mem => None,
        };
        EnvStore {
            space,
            slots: (0..d).map(|_| Slot::Empty).collect(),
            dir,
            tag,
            written: 0,
            read: 0,
        }
    }

    /// The space every slot's environment lives in.
    pub fn space(&self) -> QnSpace {
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
    pub fn put(&mut self, k: usize, s: Sketch<T>) {
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
    pub fn get(&mut self, k: usize) -> &Sketch<T> {
        self.make_resident(k);
        match &self.slots[k] {
            Slot::Resident(s) => s,
            _ => panic!("EnvStore {}[{k}]: slot is empty", self.tag),
        }
    }

    /// Take the environment out of slot `k` (loading it if needed), leaving the slot empty.
    pub fn take(&mut self, k: usize) -> Option<Sketch<T>> {
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

impl<T: Scalar> Drop for EnvStore<T> {
    fn drop(&mut self) {
        for k in 0..self.slots.len() {
            self.clear(k);
        }
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_dir(dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qn::QnSpace;

    #[test]
    fn mem_mode_is_a_plain_vector() {
        let _g = OVERRIDE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = set_spill_override(Some(SpillMode::Mem));
        let space = QnSpace::new(4, 1, 1);
        let mut st = EnvStore::<f64>::new(space, 4, "M");
        assert!(!st.is_spill());
        st.put(1, Sketch::identity(space, 1, 1));
        st.release(1);
        st.prefetch(1);
        assert_eq!(st.get(1).k, 1);
        assert!(!st.all_empty());
        assert!(st.take(1).is_some());
        assert!(st.all_empty());
        assert_eq!(st.io_bytes(), (0, 0));
        set_spill_override(prev);
    }
}
