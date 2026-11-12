// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Compressed row/column block storage for the Hamiltonian's per-site sparse matrices. `starts`
//! holds 0-based offsets into `index`/`vals` like every other index in the crate.
//! [`CsxBlock::span_len`]'s dense-over-span measure selects the matvec kernel branch and summation
//! order and must be preserved exactly.

use crate::scalar::Scalar;

/// Coordinate-format `(row, col, val)` triples for one (site, output-sector, input-sector)
/// block (0-based row/col ordinals into `row_ranges`/`col_ranges`).
#[derive(Clone, Debug)]
pub struct CooCell<T: Scalar> {
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    pub vals: Vec<T>,
}

impl<T: Scalar> CooCell<T> {
    /// An empty COO cell.
    pub fn empty() -> Self {
        CooCell {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }
}

impl<T: Scalar> Default for CooCell<T> {
    fn default() -> Self {
        Self::empty()
    }
}

/// Compressed sparse row/column block: `ids`/`starts` index distinct majors (rows for CSR, columns
/// for CSC, ascending); `index`/`vals` are the per-major minor-index/value runs,
/// `index[starts[k]..starts[k+1]]` giving the minors of major `ids[k]` (`starts` is 0-based:
/// `starts[0] == 0`, `starts[last] == nnz`).
#[derive(Clone, Debug)]
pub struct CsxBlock<T: Scalar> {
    pub ids: Vec<i32>,
    pub starts: Vec<i32>,
    pub index: Vec<i32>,
    pub vals: Vec<T>,
}

impl<T: Scalar> CsxBlock<T> {
    /// The empty block (`ids=[]`, `starts=[0]`, `index=[]`, `vals=[]`).
    pub fn empty() -> Self {
        CsxBlock {
            ids: Vec::new(),
            starts: vec![0],
            index: Vec::new(),
            vals: Vec::new(),
        }
    }

    /// Stored entry count (`nnz`). Equal to the entry count of the [`CooCell`] the block was built
    /// from — [`csx_pair`] permutes, it never coalesces or drops.
    pub fn nnz(&self) -> usize {
        self.vals.len()
    }

    /// The block's `(major, minor, val)` entries in stored order: major-ascending, and within a
    /// major, minor-ascending.
    pub fn entries(&self) -> impl Iterator<Item = (i32, i32, T)> + '_ {
        self.ids.iter().enumerate().flat_map(move |(k, &major)| {
            let lo = self.starts[k] as usize;
            let hi = self.starts[k + 1] as usize;
            (lo..hi).map(move |p| (major, self.index[p], self.vals[p]))
        })
    }

    /// Dense-over-span measure: `0` when empty, else `ids[last] - ids[0] + 2` — one MORE than the
    /// true span `ids[last] - ids[0] + 1`. That is the branch predicate's historical measure, kept
    /// numerically: it selects the matvec kernel branch and its summation order, so it is
    /// correctness-load-bearing, not merely a size.
    pub fn span_len(&self) -> usize {
        if self.ids.is_empty() {
            0
        } else {
            (self.ids[self.ids.len() - 1] - self.ids[0]) as usize + 2
        }
    }
}

/// Build the CSR/CSC pair `(csr, csc)` from a COO cell.
pub fn csx_pair<T: Scalar>(coo: &CooCell<T>) -> (CsxBlock<T>, CsxBlock<T>) {
    let (csr, csc, _, _) = csx_pair_perms(coo);
    (csr, csc)
}

/// Per-value provenance of one compressed cell — what a coefficient-only refill needs to rewrite
/// the value arrays without touching an index.
#[derive(Clone, Debug)]
pub struct CellSrc {
    pub csr_src: Vec<u32>,
    pub csc_src: Vec<u32>,
}

/// [`csx_pair`] plus the provenance permuted into CSR order and into CSC order.
/// The two blocks are built by the SAME code as [`csx_pair`], so they are bitwise the blocks a
/// provenance-free build produces.
pub fn csx_pair_with_src<T: Scalar>(
    coo: &CooCell<T>,
    src: &[u32],
) -> (CsxBlock<T>, CsxBlock<T>, CellSrc) {
    let nnz = coo.vals.len();
    assert_eq!(
        src.len(),
        nnz,
        "csx_pair_with_src: provenance length {} ≠ nnz {nnz}",
        src.len()
    );
    let (csr, csc, p_row, p_col) = csx_pair_perms(coo);
    let csr_src: Vec<u32> = p_row.iter().map(|&p| src[p]).collect();
    let csc_src: Vec<u32> = p_col.iter().map(|&p| src[p]).collect();
    (csr, csc, CellSrc { csr_src, csc_src })
}

fn csx_pair_perms<T: Scalar>(
    coo: &CooCell<T>,
) -> (CsxBlock<T>, CsxBlock<T>, Vec<usize>, Vec<usize>) {
    let nnz = coo.rows.len();
    assert!(
        nnz < i32::MAX as usize,
        "sparse block nnz {nnz} overflows the Int32 CSR/CSC starts"
    );

    let mut p_row: Vec<usize> = (0..nnz).collect();
    p_row.sort_by(|&a, &b| (coo.rows[a], coo.cols[a]).cmp(&(coo.rows[b], coo.cols[b])));
    let rows_sorted: Vec<i32> = p_row.iter().map(|&p| coo.rows[p]).collect();
    let (r_ids, r_starts) = csx_ids_starts_sorted(&rows_sorted);
    let csr = CsxBlock {
        ids: r_ids,
        starts: r_starts,
        index: p_row.iter().map(|&p| coo.cols[p]).collect(),
        vals: p_row.iter().map(|&p| coo.vals[p]).collect(),
    };

    let mut p_col: Vec<usize> = (0..nnz).collect();
    p_col.sort_by(|&a, &b| (coo.cols[a], coo.rows[a]).cmp(&(coo.cols[b], coo.rows[b])));
    let cols_sorted: Vec<i32> = p_col.iter().map(|&p| coo.cols[p]).collect();
    let (c_ids, c_starts) = csx_ids_starts_sorted(&cols_sorted);
    let csc = CsxBlock {
        ids: c_ids,
        starts: c_starts,
        index: p_col.iter().map(|&p| coo.rows[p]).collect(),
        vals: p_col.iter().map(|&p| coo.vals[p]).collect(),
    };

    (csr, csc, p_row, p_col)
}

/// Distinct sorted majors and 0-based start offsets (`starts.len() == ids.len()+1`, `starts[0] ==
/// 0`, `starts[last] == mi.len()`) from a major-index array. The input is sorted internally (sorting a copy makes the
/// standalone helper robust and idempotent); the `csx_pair` call sites, whose input already
/// arrives non-decreasing, walk it directly through [`csx_ids_starts_sorted`].
pub fn csx_ids_starts(mi: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let mut sorted = mi.to_vec();
    sorted.sort_unstable();
    csx_ids_starts_sorted(&sorted)
}

fn csx_ids_starts_sorted(sorted: &[i32]) -> (Vec<i32>, Vec<i32>) {
    debug_assert!(
        sorted.is_sorted(),
        "csx_ids_starts_sorted needs sorted majors"
    );
    let nnz = sorted.len();
    let mut ids = Vec::new();
    let mut starts = Vec::new();
    for i in 0..nnz {
        if i == 0 || sorted[i] > sorted[i - 1] {
            ids.push(sorted[i]);
            starts.push(i as i32);
        }
    }
    starts.push(nnz as i32);
    (ids, starts)
}

impl crate::mem::Footprint for CellSrc {
    fn bytes(&self) -> u64 {
        ((self.csr_src.capacity() + self.csc_src.capacity()) as u64) * 4
    }
}

impl<T: Scalar> crate::mem::Footprint for CsxBlock<T> {
    fn bytes(&self) -> u64 {
        let i32s = (self.ids.capacity() + self.starts.capacity() + self.index.capacity()) as u64;
        i32s * 4 + (self.vals.capacity() as u64) * (std::mem::size_of::<T>() as u64)
    }
}

#[cfg(test)]
mod phase4_tests {
    use super::{CooCell, CsxBlock, csx_ids_starts, csx_pair};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn t1_csx_pair_roundtrip() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5);
        let n = 12;
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        for _ in 0..n {
            rows.push(rng.random_range(0..5));
            cols.push(rng.random_range(0..4));
            vals.push(rng.random_range(-1.0..1.0));
        }
        let coo = CooCell {
            rows: rows.clone(),
            cols: cols.clone(),
            vals: vals.clone(),
        };
        let (csr, csc) = csx_pair(&coo);

        let mut expect: Vec<(i32, i32, u64)> = rows
            .iter()
            .zip(&cols)
            .zip(&vals)
            .map(|((&r, &c), &v)| (r, c, v.to_bits()))
            .collect();
        expect.sort();

        let mut from_csr: Vec<(i32, i32, u64)> = Vec::new();
        for (k, &id) in csr.ids.iter().enumerate() {
            let lo = csr.starts[k] as usize;
            let hi = csr.starts[k + 1] as usize;
            for p in lo..hi {
                from_csr.push((id, csr.index[p], csr.vals[p].to_bits()));
            }
        }
        from_csr.sort();
        assert_eq!(from_csr, expect, "CSR does not reconstruct the COO triples");

        let mut from_csc: Vec<(i32, i32, u64)> = Vec::new();
        for (k, &id) in csc.ids.iter().enumerate() {
            let lo = csc.starts[k] as usize;
            let hi = csc.starts[k + 1] as usize;
            for p in lo..hi {
                from_csc.push((csc.index[p], id, csc.vals[p].to_bits()));
            }
        }
        from_csc.sort();
        assert_eq!(from_csc, expect, "CSC does not reconstruct the COO triples");
    }

    #[test]
    fn t1_csx_ids_starts() {
        let mi = vec![3, 1, 1, 4, 1, 5, 3];
        let (ids, starts) = csx_ids_starts(&mi);
        let mut expect_ids = mi.clone();
        expect_ids.sort_unstable();
        expect_ids.dedup();
        assert_eq!(ids, expect_ids, "ids must be the distinct sorted majors");
        assert_eq!(
            starts.len(),
            ids.len() + 1,
            "starts has one more entry than ids"
        );
        assert_eq!(starts[0], 0, "starts is 0-based");
        assert_eq!(
            *starts.last().unwrap(),
            mi.len() as i32,
            "starts[last] = nnz"
        );
        for w in starts.windows(2) {
            assert!(w[0] <= w[1], "starts must be nondecreasing");
        }
    }

    #[test]
    fn t1_csx_spanlen() {
        let empty = CsxBlock::<f64>::empty();
        assert_eq!(empty.span_len(), 0, "empty block has span_len 0");

        let block = CsxBlock::<f64> {
            ids: vec![2, 5, 9],
            starts: vec![0, 1, 2, 3],
            index: vec![0, 1, 2],
            vals: vec![1.0, 2.0, 3.0],
        };
        assert_eq!(block.span_len(), (9 - 2) + 2);
    }
}
