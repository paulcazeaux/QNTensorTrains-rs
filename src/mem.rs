// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Memory accounting: the logical byte footprint of the crate's containers.

/// Logical coefficient payload, in bytes.
pub trait Footprint {
    /// Bytes of coefficient storage this value holds (see the module doc for what is excluded).
    fn bytes(&self) -> u64;
}

impl<F: Footprint> Footprint for [F] {
    fn bytes(&self) -> u64 {
        self.iter().map(Footprint::bytes).sum()
    }
}

impl<F: Footprint> Footprint for Vec<F> {
    fn bytes(&self) -> u64 {
        self.as_slice().bytes()
    }
}

impl<F: Footprint> Footprint for Option<F> {
    fn bytes(&self) -> u64 {
        self.as_ref().map_or(0, Footprint::bytes)
    }
}

#[inline]
pub(crate) fn block_bytes<T>(nrows: usize, ncols: usize) -> u64 {
    (nrows as u64) * (ncols as u64) * (std::mem::size_of::<T>() as u64)
}

/// Human-readable byte count (`1.23 GB`, `456.7 MB`, `12.3 kB`, `17 B`), base 1000.
pub fn fmt_bytes(b: u64) -> String {
    let f = b as f64;
    if f >= 1e9 {
        format!("{:.2} GB", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.1} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.1} kB", f / 1e3)
    } else {
        format!("{b} B")
    }
}
