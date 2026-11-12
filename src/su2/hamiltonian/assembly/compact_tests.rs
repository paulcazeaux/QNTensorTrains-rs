// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use rand::RngExt;

use super::{SRC_NONE, Su2CellSrc, Su2Factor, Su2RunSrc, alpha, median_weight};
use crate::mem::Footprint;
use crate::test_util::seeded;

const BLOCK: usize = 64;

const U16_RANGE: u32 = u16::MAX as u32;

const N_FACTORS: usize = 2048;

const TERM_MAX: u32 = SRC_NONE - 1;

const FID0: usize = 200;

fn factors() -> Vec<Su2Factor> {
    (0..N_FACTORS)
        .map(|i| {
            let transport = i % 4 == 3;
            Su2Factor {
                cw: if transport {
                    0.0
                } else {
                    0.5 + (i % 17) as f64 / 16.0
                },
                fr: 0.25 + (i % 29) as f64 / 8.0,
                neg: (i / 4) % 2 == 1,
                transport,
            }
        })
        .collect()
}

fn mfid(k: usize) -> u32 {
    let i = k + k / 3;
    assert!(i < N_FACTORS && i % 4 != 3, "mfid({k}) out of the table");
    i as u32
}

fn tfid(k: usize) -> u32 {
    let i = 4 * k + 3;
    assert!(i < N_FACTORS, "tfid({k}) out of the table");
    i as u32
}

fn med(term: u32, k: usize) -> (u32, u32) {
    (term, mfid(k))
}

fn tra(k: usize) -> (u32, u32) {
    (SRC_NONE, tfid(k))
}

struct Cell {
    runs: Vec<Vec<(u32, u32)>>,
    perm: Vec<u32>,
}

impl Cell {
    fn new(runs: Vec<Vec<(u32, u32)>>, seed: u64) -> Self {
        let perm = shuffled(runs.len(), seed);
        Cell { runs, perm }
    }

    fn staging(&self) -> Su2RunSrc {
        let mut s = Su2RunSrc::default();
        for r in &self.runs {
            s.push_run(r);
        }
        s
    }

    fn encode(&self, factors: &[Su2Factor]) -> Su2CellSrc {
        Su2CellSrc::new(self.staging(), self.perm.clone(), factors)
    }

    fn r1a_payload_bytes(&self) -> usize {
        let spill: usize = self.runs.iter().filter(|r| r.len() > 1).map(Vec::len).sum();
        (self.runs.len() + spill) * 8 + self.perm.len() * 4
    }
}

fn shuffled(n: usize, seed: u64) -> Vec<u32> {
    let mut rng = seeded(seed);
    let mut v: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        v.swap(i, rng.random_range(0..=i));
    }
    v
}

#[derive(Debug, PartialEq)]
struct Want {
    blocks: usize,
    escape_blocks: usize,
    wide_perm: bool,
    fid_fallback: bool,
    spilled_runs: usize,
    transport_singles: usize,
}

fn is_median_single(r: &[(u32, u32)]) -> bool {
    r.len() == 1 && r[0].0 != SRC_NONE
}

fn is_transport_single(r: &[(u32, u32)]) -> bool {
    r.len() == 1 && r[0].0 == SRC_NONE
}

fn want(cell: &Cell) -> Want {
    let n = cell.runs.len();
    let escape_blocks = cell
        .runs
        .chunks(BLOCK)
        .filter(|b| {
            let terms: Vec<u32> = b
                .iter()
                .filter(|r| is_median_single(r))
                .map(|r| r[0].0)
                .collect();
            match (terms.iter().min(), terms.iter().max()) {
                (Some(&lo), Some(&hi)) => hi - lo > U16_RANGE,
                _ => false,
            }
        })
        .count();
    Want {
        blocks: n.div_ceil(BLOCK),
        escape_blocks,
        wide_perm: n > 65_536,
        fid_fallback: distinct_single_fids(cell) > 254,
        spilled_runs: cell.runs.iter().filter(|r| r.len() > 1).count(),
        transport_singles: cell.runs.iter().filter(|r| is_transport_single(r)).count(),
    }
}

fn distinct_single_fids(cell: &Cell) -> usize {
    let mut fids: Vec<u32> = cell
        .runs
        .iter()
        .filter(|r| r.len() == 1)
        .map(|r| r[0].1)
        .collect();
    fids.sort_unstable();
    fids.dedup();
    fids.len()
}

fn no_median_blocks(cell: &Cell) -> usize {
    cell.runs
        .chunks(BLOCK)
        .filter(|b| !b.iter().any(|r| is_median_single(r)))
        .count()
}

fn assert_decodes(what: &str, src: &Su2CellSrc, cell: &Cell) {
    assert_eq!(src.slots(), cell.runs.len(), "{what}: slot count");
    for (p, staged) in cell.runs.iter().enumerate() {
        assert!(
            src.run(p).eq(staged.iter().copied()),
            "{what}: CSR slot {p}: decoded {:?} ≠ staged {staged:?}",
            src.run(p).collect::<Vec<_>>()
        );
    }
    for (q, &p) in cell.perm.iter().enumerate() {
        assert_eq!(
            src.csc_to_csr(q),
            p as usize,
            "{what}: CSC slot {q} → CSR slot"
        );
    }
    assert!(
        src.bytes() >= src.payload_len_bytes() as u64,
        "{what}: Footprint {} below the payload by length {}",
        src.bytes(),
        src.payload_len_bytes()
    );
}

fn assert_census(what: &str, src: &Su2CellSrc, cell: &Cell) -> Want {
    let c = src.layout_census();
    let got = Want {
        blocks: c.blocks,
        escape_blocks: c.escape_blocks,
        wide_perm: c.wide_perm,
        fid_fallback: c.fid_fallback,
        spilled_runs: c.spilled_runs,
        transport_singles: c.transport_singles,
    };
    let w = want(cell);
    assert_eq!(
        got, w,
        "{what}: layout census (left) ≠ the staged runs' (right)"
    );
    w
}

fn round_trip(what: &str, cell: &Cell, factors: &[Su2Factor]) -> Want {
    let src = cell.encode(factors);
    assert_decodes(what, &src, cell);
    assert_census(what, &src, cell)
}

struct Mix {
    n: usize,
    median_fids: usize,
    transport_fids: usize,
    max_base: u32,
}

fn draw_term<R: RngExt>(rng: &mut R, base: u32, span: u32) -> u32 {
    base + rng.random_range(0..span)
}

fn draw_spill<R: RngExt>(rng: &mut R, mix: &Mix, base: u32, span: u32) -> Vec<(u32, u32)> {
    let len = rng.random_range(2..=6usize);
    (0..len)
        .map(|_| {
            if rng.random_bool(0.25) {
                tra(rng.random_range(0..mix.transport_fids))
            } else {
                let t = draw_term(rng, base, span);
                med(t, FID0 + rng.random_range(0..mix.median_fids))
            }
        })
        .collect()
}

fn random_cell(mix: &Mix, seed: u64) -> Cell {
    let mut rng = seeded(seed);
    let mut runs = Vec::with_capacity(mix.n);
    for b in 0..mix.n.div_ceil(BLOCK) {
        let len = BLOCK.min(mix.n - b * BLOCK);
        let kind = match b {
            0..=2 => b,
            _ => match rng.random_range(0..8u32) {
                0 => 2,
                1 | 2 => 1,
                _ => 0,
            },
        };
        let base = rng.random_range(0..=mix.max_base);
        let span = if kind == 1 { 200_000 } else { 30_000 };
        for i in 0..len {
            let run = if kind == 1 && i < 2 {
                vec![med(base + 100_000 * i as u32, FID0 + i)]
            } else if kind == 2 {
                if rng.random_bool(0.5) {
                    vec![tra(rng.random_range(0..mix.transport_fids))]
                } else {
                    draw_spill(&mut rng, mix, base, span)
                }
            } else {
                match rng.random_range(0..100u32) {
                    0..78 => {
                        let t = draw_term(&mut rng, base, span);
                        vec![med(t, FID0 + rng.random_range(0..mix.median_fids))]
                    }
                    78..90 => vec![tra(rng.random_range(0..mix.transport_fids))],
                    _ => draw_spill(&mut rng, mix, base, span),
                }
            };
            runs.push(run);
        }
    }
    Cell::new(runs, seed ^ 0x9E37_79B9_7F4A_7C15)
}

#[test]
fn r1c1_codec_narrow_blocks_round_trip() {
    let f = factors();
    let hi = 4_000_000_000u32;
    let mut runs: Vec<Vec<(u32, u32)>> = (0..64u32).map(|p| vec![med(p, p as usize % 7)]).collect();
    runs.extend((0..64u32).map(|p| {
        let t = match p {
            0 => hi + U16_RANGE,
            63 => hi,
            _ => hi + 1 + (p * 977) % (U16_RANGE - 1),
        };
        vec![med(t, FID0 + p as usize % 5)]
    }));
    runs.extend((0..10).map(|_| vec![med(123_456, FID0 + 100)]));
    let w = round_trip("narrow", &Cell::new(runs, 0x51C1_0101), &f);
    assert_eq!((w.blocks, w.escape_blocks), (3, 0), "narrow: blocks");
    assert!(!w.wide_perm && !w.fid_fallback, "narrow: {w:?}");

    let w = round_trip(
        "one slot",
        &Cell::new(vec![vec![med(7, 0)]], 0x51C1_0102),
        &f,
    );
    assert_eq!((w.blocks, w.escape_blocks), (1, 0), "one slot: blocks");
    let full: Vec<_> = (0..64u32).map(|p| vec![med(1_000 + p, FID0)]).collect();
    let w = round_trip("one full block", &Cell::new(full, 0x51C1_0103), &f);
    assert_eq!(
        (w.blocks, w.escape_blocks),
        (1, 0),
        "one full block: blocks"
    );
}

#[test]
fn r1c1_codec_escape_blocks_round_trip() {
    let f = factors();
    let mut runs: Vec<Vec<(u32, u32)>> = Vec::new();
    runs.extend((0..64u32).map(|p| vec![med(10 + p, 1)]));
    runs.extend((0..64u32).map(|p| match p {
        5 => vec![med(0, 2)],
        40 => vec![med(70_000, 3)],
        10 => vec![tra(1)],
        20 => vec![med(35_000, 2), tra(2), med(69_999, 4)],
        _ => vec![med(35_000 + 13 * p, 2)],
    }));
    runs.extend((0..64u32).map(|p| vec![med(500_000 + 3 * p, 5)]));
    runs.extend((0..64u32).map(|p| match p {
        0 => vec![med(1_065_536, 6)],
        63 => vec![med(1_000_000, 6)],
        30 => vec![tra(3)],
        _ => vec![med(1_000_000 + 1_000 * p, 7)],
    }));
    runs.extend((0..64u32).map(|p| {
        if p.is_multiple_of(2) {
            vec![med(0, FID0 + 1)]
        } else {
            vec![med(TERM_MAX, FID0 + 2)]
        }
    }));
    runs.extend((0..20u32).map(|p| vec![med(2_000_000 + p, FID0 + 3)]));
    let w = round_trip("escape", &Cell::new(runs, 0x51C1_0201), &f);
    assert_eq!(w.blocks, 6, "escape: blocks");
    assert_eq!(w.escape_blocks, 3, "escape: escape blocks");
    assert!(
        w.escape_blocks >= 1,
        "escape: the escape branch did not fire"
    );

    for (range, escapes) in [(U16_RANGE, 0usize), (U16_RANGE + 1, 1)] {
        let runs: Vec<_> = (0..64u32)
            .map(|p| match p {
                0 => vec![med(7 + range, 0)],
                1 => vec![med(7, 0)],
                _ => vec![med(7 + p, 0)],
            })
            .collect();
        let what = format!("range {range}");
        let w = round_trip(&what, &Cell::new(runs, 0x51C1_0202), &f);
        assert_eq!(w.escape_blocks, escapes, "{what}: escape blocks");
    }
}

#[test]
fn r1c1_codec_blocks_without_a_median_single_round_trip() {
    let f = factors();
    let spill = |p: usize| -> Vec<(u32, u32)> {
        let t = 40_000 + p as u32;
        match p % 3 {
            0 => vec![med(t, 1), med(t + 1, 2)],
            1 => vec![tra(p % 5), med(t, 3), tra(1)],
            _ => vec![med(t, 4), med(t, 5), med(t + 7, 1), tra(0)],
        }
    };
    let mut runs: Vec<Vec<(u32, u32)>> = Vec::new();
    runs.extend((0..64).map(|p| vec![tra(p % 6)]));
    runs.extend((0..64).map(spill));
    runs.extend((0..64usize).map(|p| {
        if p.is_multiple_of(2) {
            vec![tra(p % 4)]
        } else {
            spill(p)
        }
    }));
    runs.extend((0..64u32).map(|p| vec![med(70_000 + p, p as usize % 3)]));
    runs.extend((0..5).map(|p| vec![tra(p)]));
    let cell = Cell::new(runs, 0x51C1_0301);
    let w = round_trip("no median", &cell, &f);
    assert_eq!(no_median_blocks(&cell), 4, "no median: oracle");
    assert_eq!(w.escape_blocks, 0, "no median: escape blocks");
    assert_eq!(
        (w.transport_singles, w.spilled_runs),
        (64 + 32 + 5, 64 + 32),
        "no median: transport singles, spilled runs"
    );

    let only_tra: Vec<_> = (0..3).map(|p| vec![tra(p)]).collect();
    let w = round_trip("transports only", &Cell::new(only_tra, 0x51C1_0302), &f);
    assert_eq!(
        (w.transport_singles, w.escape_blocks),
        (3, 0),
        "transports only"
    );
    let only_spill = vec![vec![med(9, 0), tra(0)], vec![tra(1), tra(2)]];
    let w = round_trip("spills only", &Cell::new(only_spill, 0x51C1_0303), &f);
    assert_eq!((w.spilled_runs, w.escape_blocks), (2, 0), "spills only");
}

#[test]
fn r1c1_codec_csc_perm_widens_above_65536_slots() {
    let f = factors();
    for (n, wide) in [(65_536usize, false), (65_537, true)] {
        let runs: Vec<_> = (0..n).map(|p| vec![med(p as u32 / 3, p % 11)]).collect();
        let what = format!("{n} slots");
        let w = round_trip(&what, &Cell::new(runs, 0x51C1_0400 + n as u64), &f);
        assert_eq!(w.wide_perm, wide, "{what}: wide_perm");
        assert_eq!(w.blocks, n.div_ceil(BLOCK), "{what}: blocks");
    }
}

#[test]
fn r1c1_codec_local_fid_table_falls_back_above_254_fids() {
    let f = factors();
    let edge = |n_med: usize| -> Vec<Vec<(u32, u32)>> {
        (0..n_med + 4 + 100)
            .map(|p| {
                let k = p % (n_med + 4);
                if k < n_med {
                    vec![med(3 * p as u32, FID0 + k)]
                } else {
                    vec![tra(k - n_med)]
                }
            })
            .collect()
    };
    for (n_med, fallback) in [(250usize, false), (251, true)] {
        let what = format!("{} distinct fids", n_med + 4);
        let cell = Cell::new(edge(n_med), 0x51C1_0501);
        assert_eq!(distinct_single_fids(&cell), n_med + 4, "{what}: oracle");
        let w = round_trip(&what, &cell, &f);
        assert_eq!(w.fid_fallback, fallback, "{what}: fid_fallback");
        assert_eq!(w.transport_singles, 4, "{what}: transport singles");
    }

    let mut runs: Vec<Vec<(u32, u32)>> = (0..700usize)
        .map(|p| match p % 10 {
            3 => vec![tra(p % 20)],
            7 => vec![med(5 * p as u32, FID0 + p % 400), tra(p % 20)],
            _ => vec![med(5 * p as u32, FID0 + p % 400)],
        })
        .collect();
    runs[128] = vec![med(0, FID0 + 1)];
    runs[129] = vec![med(1_000_000, FID0 + 2)];
    let w = round_trip("400 + 20 fids", &Cell::new(runs, 0x51C1_0502), &f);
    assert!(w.fid_fallback, "400 + 20 fids: the fallback did not fire");
    assert!(
        w.escape_blocks >= 1 && w.spilled_runs > 0 && w.transport_singles > 0,
        "400 + 20 fids: vacuous mix {w:?}"
    );
}

#[test]
fn r1c1_codec_spilled_runs_at_block_edges_round_trip() {
    let f = factors();
    let spilled = [0usize, 1, 62, 63, 64, 65, 127, 128, 191, 192, 199];
    let spill = |p: usize| -> Vec<(u32, u32)> {
        let t = 3_000 + p as u32;
        match p % 6 {
            0 => vec![med(t, 1), med(t + 9, 2)],
            1 => vec![tra(1), med(t, 3)],
            2 => vec![tra(2), tra(3), tra(2)],
            3 => vec![med(t, 4), med(t, 5), med(t, 4), med(t, 1)],
            4 => vec![med(0, 6), tra(4), med(TERM_MAX, 7), med(t, 1), med(t, 1)],
            _ => vec![
                med(t, 1),
                med(t + 1, 2),
                tra(5),
                med(t + 2, 3),
                med(t, 4),
                tra(0),
                med(t + 3, 5),
            ],
        }
    };
    let runs: Vec<_> = (0..200usize)
        .map(|p| {
            if spilled.contains(&p) {
                spill(p)
            } else if p % 9 == 4 {
                vec![tra(p % 3)]
            } else {
                vec![med(3_000 + p as u32, p % 4)]
            }
        })
        .collect();
    let cell = Cell::new(runs, 0x51C1_0601);
    let shapes: Vec<usize> = spilled.iter().map(|p| p % 6).collect();
    assert!(
        (0..6).all(|s| shapes.contains(&s)),
        "spill edges: a spill shape is missing"
    );
    let w = round_trip("spill edges", &cell, &f);
    assert_eq!(w.spilled_runs, spilled.len(), "spill edges: spilled runs");
    assert_eq!(
        w.escape_blocks, 0,
        "spill edges: a spilled run made a block escape"
    );

    let w = round_trip(
        "one spilled slot",
        &Cell::new(vec![spill(5)], 0x51C1_0602),
        &f,
    );
    assert_eq!((w.spilled_runs, w.blocks), (1, 1), "one spilled slot");
    let mut runs: Vec<_> = (0..64u32).map(|p| vec![med(50 + p, 0)]).collect();
    runs[63] = spill(4);
    let w = round_trip("spill at slot 63", &Cell::new(runs, 0x51C1_0603), &f);
    assert_eq!(
        (w.spilled_runs, w.escape_blocks),
        (1, 0),
        "spill at slot 63"
    );
}

#[test]
fn r1c1_codec_transport_singles_round_trip() {
    let f = factors();
    let at = [0usize, 1, 63, 64, 129];
    let runs: Vec<_> = (0..130usize)
        .map(|p| {
            if at.contains(&p) || p % 7 == 3 {
                vec![tra(p % 6)]
            } else {
                vec![med(4_000_000_000 + (p as u32 * 7) % 1_000, p % 5)]
            }
        })
        .collect();
    let cell = Cell::new(runs, 0x51C1_0701);
    let w = round_trip("transports", &cell, &f);
    assert!(w.transport_singles >= 20, "transports: {w:?}");
    assert_eq!(
        w.escape_blocks, 0,
        "transports: SRC_NONE entered a block's range"
    );
}

fn mixes(max_base: u32) -> [(&'static str, Mix, bool, bool); 3] {
    [
        (
            "random 19 fids",
            Mix {
                n: 2_577,
                median_fids: 15,
                transport_fids: 4,
                max_base,
            },
            false,
            false,
        ),
        (
            "random 340 fids",
            Mix {
                n: 2_577,
                median_fids: 320,
                transport_fids: 20,
                max_base,
            },
            true,
            false,
        ),
        (
            "random 65 601 slots",
            Mix {
                n: 65_601,
                median_fids: 15,
                transport_fids: 4,
                max_base,
            },
            false,
            true,
        ),
    ]
}

#[test]
fn r1c1_codec_random_cells_round_trip() {
    let f = factors();
    for (i, (what, mix, fallback, wide)) in mixes(4_000_000_000).into_iter().enumerate() {
        let cell = random_cell(&mix, 0x51C1_0800 + i as u64);
        let w = round_trip(what, &cell, &f);
        assert!(
            w.escape_blocks >= 1
                && no_median_blocks(&cell) >= 1
                && w.spilled_runs >= 1
                && w.transport_singles >= 1,
            "{what}: vacuous mix {w:?}"
        );
        assert_eq!(
            (w.fid_fallback, w.wide_perm),
            (fallback, wide),
            "{what}: fid_fallback, wide_perm"
        );
    }
}

#[test]
fn r1c1_codec_layout_is_a_function_of_the_logical_provenance() {
    let f = factors();
    for (i, (what, mix, _, _)) in mixes(4_000_000_000).into_iter().take(2).enumerate() {
        let cell = random_cell(&mix, 0x51C1_0900 + i as u64);
        let a = cell.encode(&f);
        let b = Su2CellSrc::new(cell.staging(), cell.perm.clone(), &f);
        let decoded = Cell {
            runs: (0..a.slots()).map(|p| a.run(p).collect()).collect(),
            perm: (0..a.slots()).map(|q| a.csc_to_csr(q) as u32).collect(),
        };
        assert_decodes(what, &a, &decoded);
        let c = decoded.encode(&f);
        for (twin, x) in [("re-encoded", &b), ("decoded and re-encoded", &c)] {
            assert_eq!(
                a.layout_census(),
                x.layout_census(),
                "{what}: {twin}: census"
            );
            assert_eq!(
                a.payload_len_bytes(),
                x.payload_len_bytes(),
                "{what}: {twin}: payload"
            );
            assert!(
                format!("{a:?}") == format!("{x:?}"),
                "{what}: {twin}: the layout words differ"
            );
        }
    }
}

#[test]
fn r1c1_codec_payload_is_below_the_r1a_words() {
    let f = factors();
    let single: Vec<_> = (0..1_024u32)
        .map(|p| vec![med(5_000 + (p * 37) % 2_000, p as usize % 12)])
        .collect();
    let mix: Vec<_> = (0..4_096usize)
        .map(|p| {
            if p.is_multiple_of(75) {
                vec![med(9_000 + p as u32, 1), tra(0)]
            } else if p % 15 == 7 {
                vec![tra(p % 3)]
            } else {
                vec![med(9_000 + p as u32, p % 12)]
            }
        })
        .collect();
    for (what, runs, half) in [
        ("1 024 single medians", single, true),
        ("4 096-slot mix", mix, false),
    ] {
        let cell = Cell::new(runs, 0x51C1_0A01);
        let src = cell.encode(&f);
        assert_decodes(what, &src, &cell);
        let (narrow, r1a) = (src.payload_len_bytes(), cell.r1a_payload_bytes());
        println!(
            "{what}: payload {narrow} B (narrow) vs {r1a} B (R1a) = {:.3}×, {:.2} B/slot",
            narrow as f64 / r1a as f64,
            narrow as f64 / cell.runs.len() as f64
        );
        assert!(
            narrow < r1a,
            "{what}: payload {narrow} B not below R1a's {r1a} B"
        );
        if half {
            assert!(
                2 * narrow <= r1a,
                "{what}: payload {narrow} B above half R1a's {r1a} B"
            );
        }
    }
}

fn references(run: &[(u32, u32)], changed: &[u64]) -> bool {
    run.iter()
        .any(|&(t, _)| t != SRC_NONE && (changed[(t >> 6) as usize] >> (t & 63)) & 1 != 0)
}

fn oracle_value(run: &[(u32, u32)], gamma: &[f64], f: &[Su2Factor]) -> f64 {
    let mut s = 0.0f64;
    for &(term, fid) in run {
        let x = &f[fid as usize];
        s += if term == SRC_NONE {
            alpha(1.0, x.neg, x.fr)
        } else {
            alpha(median_weight(gamma[term as usize], x.cw), x.neg, x.fr)
        };
    }
    s
}

fn csr_sentinel(p: usize) -> f64 {
    -(1.0e6 + p as f64)
}

fn csc_sentinel(q: usize) -> f64 {
    -(2.0e6 + q as f64)
}

fn refill_arm(what: &str, cell: &Cell, f: &[Su2Factor], seed: u64) {
    let n = cell.runs.len();
    let src = cell.encode(f);
    let n_terms = 1 + cell
        .runs
        .iter()
        .flatten()
        .filter(|&&(t, _)| t != SRC_NONE)
        .map(|&(t, _)| t as usize)
        .max()
        .expect("a median term");
    let mut rng = seeded(seed);
    let gamma: Vec<f64> = (0..n_terms)
        .map(|o| match (o % 17, o % 19) {
            (5, _) => -0.0,
            (_, 0) => 0.0,
            _ => rng.random_range(-1.0..1.0),
        })
        .collect();
    let mut changed = vec![0u64; n_terms.div_ceil(64)];
    for o in 0..n_terms {
        if rng.random_bool(0.3) {
            changed[o >> 6] |= 1 << (o & 63);
        }
    }
    let hit: Vec<bool> = cell.runs.iter().map(|r| references(r, &changed)).collect();
    let written = hit.iter().filter(|&&h| h).count();
    assert!(
        0 < written && written < n,
        "{what}: vacuous changed set ({written} of {n} slots)"
    );
    let late = cell
        .runs
        .iter()
        .filter(|r| !references(&r[..1], &changed) && references(r, &changed))
        .count();
    assert!(
        late > 0,
        "{what}: no run references a changed term past its head"
    );

    for chunk in [1, 3, 63, 64, 65, n] {
        let mut csr: Vec<f64> = (0..n).map(csr_sentinel).collect();
        let mut mask = vec![0xA5u8; n];
        let mut w_csr = 0;
        for b in (0..n).step_by(chunk) {
            let e = (b + chunk).min(n);
            w_csr += src.refill_csr(&mut csr[b..e], b, &gamma, f, &changed, &mut mask[b..e]);
        }
        let mut csc: Vec<f64> = (0..n).map(csc_sentinel).collect();
        let mut w_csc = 0;
        for b in (0..n).step_by(chunk) {
            let e = (b + chunk).min(n);
            w_csc += src.refill_csc(&mut csc[b..e], b, &csr, &mask);
        }
        assert_eq!(w_csr, written, "{what} chunk {chunk}: CSR slots written");
        assert_eq!(w_csc, written, "{what} chunk {chunk}: CSC slots written");
        for p in 0..n {
            let want = if hit[p] {
                oracle_value(&cell.runs[p], &gamma, f)
            } else {
                csr_sentinel(p)
            };
            assert_eq!(
                csr[p].to_bits(),
                want.to_bits(),
                "{what} chunk {chunk}: CSR slot {p} (run {:?}, hit {})",
                cell.runs[p],
                hit[p]
            );
        }
        for q in 0..n {
            let p = cell.perm[q] as usize;
            let want = if hit[p] { csr[p] } else { csc_sentinel(q) };
            assert_eq!(
                csc[q].to_bits(),
                want.to_bits(),
                "{what} chunk {chunk}: CSC slot {q} (CSR slot {p}, hit {})",
                hit[p]
            );
        }
    }

    let none = vec![0u64; changed.len()];
    let mut csr: Vec<f64> = (0..n).map(csr_sentinel).collect();
    let mut csc: Vec<f64> = (0..n).map(csc_sentinel).collect();
    let mut mask = vec![0xA5u8; n];
    assert_eq!(
        src.refill_csr(&mut csr, 0, &gamma, f, &none, &mut mask),
        0,
        "{what}: nothing changed, CSR"
    );
    assert_eq!(
        src.refill_csc(&mut csc, 0, &csr, &mask),
        0,
        "{what}: nothing changed, CSC"
    );
    assert!(
        (0..n).all(|p| csr[p].to_bits() == csr_sentinel(p).to_bits()
            && csc[p].to_bits() == csc_sentinel(p).to_bits()),
        "{what}: nothing changed, yet a value moved"
    );
}

#[test]
fn r1c1_codec_refill_passes_cover_every_branch() {
    let f = factors();
    for (i, (what, mix, fallback, wide)) in mixes(100_000).into_iter().enumerate() {
        let cell = random_cell(&mix, 0x51C1_0B00 + i as u64);
        let w = round_trip(what, &cell, &f);
        assert!(
            w.escape_blocks >= 1 && no_median_blocks(&cell) >= 1 && w.spilled_runs >= 1,
            "{what}: vacuous mix {w:?}"
        );
        assert_eq!(
            (w.fid_fallback, w.wide_perm),
            (fallback, wide),
            "{what}: fid_fallback, wide_perm"
        );
        refill_arm(what, &cell, &f, 0x51C1_0B10 + i as u64);
    }
}
