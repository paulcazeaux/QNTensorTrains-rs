// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use crate::hamiltonian::matvec::block_split;
use crate::knobs;

pub(crate) const MAX_SHARDS: u64 = 16;

pub(crate) const SPLIT_MIN_WIDTH: usize = 8;

#[cfg(test)]
pub(crate) static FORCE_SPLIT: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Region {
    SketchHMul = 0,
    HSketchMul = 1,
    ZFill = 2,
    ZRegrow = 3,
    Resid = 4,
}

#[cfg(test)]
thread_local! {
    static SPLITS: std::cell::Cell<[usize; 5]> = const { std::cell::Cell::new([0; 5]) };
}

#[inline]
pub(crate) fn count_split(_r: Region) {
    #[cfg(test)]
    SPLITS.with(|c| {
        let mut v = c.get();
        v[_r as usize] += 1;
        c.set(v);
    });
}

#[cfg(test)]
pub(crate) fn take_split_counts() -> [usize; 5] {
    SPLITS.with(|c| c.replace([0; 5]))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Splitter {
    on: bool,
    target: u64,
    min_w: usize,
}

impl Splitter {
    pub(crate) fn new(total: u64) -> Self {
        let on = block_split() && rayon::current_num_threads() > 1;
        #[cfg(test)]
        if FORCE_SPLIT.load(std::sync::atomic::Ordering::Relaxed) {
            return Splitter {
                on,
                target: 1,
                min_w: 2,
            };
        }
        Splitter::with(
            on,
            total,
            knobs::SPLIT_TARGET.get(),
            knobs::SPLIT_MIN_WIDTH.get(),
        )
    }

    pub(crate) fn with(on: bool, total: u64, den: usize, min_w: usize) -> Self {
        let target = match den {
            0 => 1,
            d => (total / d as u64).max(1),
        };
        Splitter {
            on: on && den != knobs::NEVER,
            target,
            min_w: min_w.max(1),
        }
    }

    pub(crate) fn count(&self, cost: u64, width: usize) -> usize {
        if !self.on {
            return 1;
        }
        (cost.div_ceil(self.target).clamp(1, MAX_SHARDS) as usize)
            .min(width / self.min_w)
            .max(1)
    }
}

#[allow(dead_code)]
pub(crate) fn equal_cuts(width: usize, n: usize) -> Vec<Range<usize>> {
    (0..n).map(|s| width * s / n..width * (s + 1) / n).collect()
}

pub(crate) fn band_cuts(bands: &[Range<usize>], n: usize) -> Vec<(u32, u32)> {
    let len = bands.len();
    let total = bands.last().map_or(0, |r| r.end);
    let mut cuts = Vec::with_capacity(n);
    let mut o0 = 0usize;
    for sh in 1..=n {
        let want = total * sh / n;
        let mut o1 = o0;
        while o1 < len && bands[o1].end <= want {
            o1 += 1;
        }
        if sh == n {
            o1 = len;
        }
        if o1 > o0 {
            cuts.push((o0 as u32, o1 as u32));
            o0 = o1;
        }
    }
    cuts
}

#[cfg(test)]
mod gemm_slice_tests;
