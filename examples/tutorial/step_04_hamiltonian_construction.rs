// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Inside the operator: the structure of the assembled Hamiltonian, counted bond by bond.
//!
//! # Introduction
//!
//! The Hamiltonian is assembled into a tensor-train operator whose shape depends only on the
//! integrals and the space, not on any state. This step reads a FCIDUMP, builds the operator, and
//! prints counts that describe its size. Apart from the construction wall time, every number is
//! deterministic:
//!
//! - **M1** — the operator's bond dimension at each bond (`automaton_bond_dims`), with its largest
//!   per-sector share. It is counted per sector and summed, so it is not the number an external
//!   MPO code reports as its bond dimension; `automaton_channel_counts` gives that one.
//! - **M2** — the number of nonzero operator entries, in total and per site; it tracks the cost of
//!   applying `H`, not the bond dimension.
//! - **M3** — for a state at uniform per-sector rank `R` (the step-03 convention), the bond rank
//!   of the untruncated product `H·x` at each bond.
//! - the construction wall time;
//! - a histogram of channel classes: at each bond and sector, the operator channels grouped by
//!   their flux (the change in `(Nup, Ndn)` they carry), with the largest classes listed.
//!
//! # Builds on
//!
//! step-03
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example step-04                       # LiH/STO-3G, bipartite layout
//! cargo run --release --example step-04 -- testdata/lih_sto3g/LIH.FCIDUMP
//! cargo run --release --example step-04 -- testdata/n2_ccpvdz/N2.FCIDUMP 8 median
//! cargo run --release --example step-04 -- --help
//! ```
//!
//! ```text
//! fcidump      <str>    a FCIDUMP path, or a testdata/ fixture name       [default lih_sto3g]
//! R            <usize>  reference per-sector x-rank for the M3 rank sums [default 4]
//! construction <str>    median|bipartite|persector|mediancover|bipmedian [default bipartite]
//! --tcdump     <path>   transcorrelated dump; turns the 3-body w channel on  [optional]
//! --pg [kind]           resolve the dump's point group; a bare --pg means `auto`
//! --format <text|tsv>   output format                                    [default text]
//! ```
//!
//! `construction` (case-insensitive) selects one of several assembly strategies for the same
//! Hamiltonian; they differ in the bond profile they produce. `bipartite` is the library default.
//! Every run prints `CONSTRUCTION=<arm>` under the header, so two runs are never confused.
//!
//! `--pg [KIND]` (optional, order-independent) resolves the dump's `ORBSYM`/`ISYM` and builds a
//! point-group-symmetric space; bare `--pg` guesses the conventional group of the implied order,
//! `--pg C2v` forces one (the guess only affects irrep *names* — see
//! `PointGroupKind::default_for_order`). Without the flag no point-group path is entered. The
//! header line names the resolved group and the target irrep when the flag is on.
//!
//! `--tcdump <path>` (optional, order-independent) loads a paired transcorrelated dump through
//! `integrals::load_fcidump` with `TwoBodySymmetry::Pairswap` (a transcorrelated Hamiltonian is
//! not Hermitian, so the 8-fold symmetry does not hold) and builds the Hamiltonian with the 3-body
//! `w` channel on. Without it the step reads a plain FCIDUMP (1+2-body only). Every run prints
//! `W_CHANNEL=on|off` under the header.
//!
//! # Results
//!
//! A run opens with a banner naming the dump, `norb`/`nelec`/`ms2`, the reference rank `R` and,
//! under `--pg`, the resolved group and target irrep; then `CONSTRUCTION=…`, `W_CHANNEL=on|off`,
//! and `M5_CONSTRUCTION_WALL` in seconds. The body is one row per bond `0..=d`: channels (M1 at
//! that bond), how many of the bond's sectors carry any channel, the largest per-sector channel
//! count, the number of `(sector, flux)` classes, the largest class, and the M3 rank sum. Three
//! summary lines follow with the full profiles — `M1_BOND_DIM` (max, mean, per-bond),
//! `M2_COO_NNZ` (total and per-site), `M3_RANKSUM_R<R>` — then the power-of-two histogram of class
//! sizes and the ten largest classes with their `(bond, sector, flux)` identity. The same dump and
//! the same `construction` reproduce every count exactly.
//!
//! # Possibilities for extensions
//!
//! - Run the same dump with `median` and `bipartite` and compare the `M1_BOND_DIM` and
//!   `M2_COO_NNZ` profiles.
//! - Walk up in size — `testdata/lih_sto3g/LIH.FCIDUMP`, `testdata/h2o_sto3g/H2O.FCIDUMP`,
//!   `testdata/n2_ccpvdz/N2.FCIDUMP` — and watch the bond profile and the nonzero count grow
//!   with `d`.
//! - Add `--pg` on a symmetric dump (`testdata/n2_d2h/N2.FCIDUMP`, `testdata/h2o_c2v/H2O.FCIDUMP`)
//!   and compare the bond profile to the same dump without the flag.
//! - Vary `R` (2, 4, 16) and read only the M3 column: it shows how large an untruncated `H·x`
//!   becomes as the state's rank grows.
//! - Wrap the run in `/usr/bin/time -l` (macOS) or `/usr/bin/time -v` (Linux) to add the peak
//!   memory of construction.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use qntt::coeff::ThreeBody;
use qntt::fcidump::{TwoBodySymmetry, read_fcidump};
use qntt::hamiltonian::Construction;
use qntt::hamiltonian::assembly::Flux;
use qntt::integrals::{IntegralsOptions, ThreeBodyOptions, load_fcidump};
use qntt::{QnSpace, Sector, SparseHamiltonian};

use qntt_exkit::prelude::*;

/// Build the harness's `QnSpace`: point-group-aware when `pgd` is `Some` (`--pg` was given),
/// plain `from_n_two_sz` otherwise. The returned label goes in the header line so a run
/// self-documents which group (and which target irrep NAME — the only thing the guess can get
/// wrong) produced the metrics that follow.
fn resolve_space(
    norb: usize,
    nelec: usize,
    ms2: i64,
    pgd: Option<qntt::fcidump::PointGroupData>,
) -> (QnSpace, String) {
    match pgd {
        Some(p) => (
            QnSpace::from_n_two_sz_with_point_group(norb, nelec, ms2, p.pg, p.target),
            format!(
                " | pg={} ({}) target={}",
                p.kind.name(),
                if p.guessed { "guessed" } else { "explicit" },
                p.kind.irrep_name(p.target).expect("target in range")
            ),
        ),
        None => (QnSpace::from_n_two_sz(norb, nelec, ms2), String::new()),
    }
}

fn parse_construction(s: &str) -> Construction {
    match s.to_ascii_lowercase().as_str() {
        "median" => Construction::Median,
        "bipartite" => Construction::Bipartite,
        "persector" | "bipartite_persector" => Construction::BipartitePerSector,
        // Two hybrids of the median and bipartite constructions.
        "mediancover" | "median_cover" => Construction::MedianCoverSwitch,
        "bipmedian" | "bipartite_median" => Construction::BipartiteMedianSwitch,
        other => panic!(
            "unknown construction {other:?}: expected \"median\", \"bipartite\", \"persector\", \
             or \"mediancover\""
        ),
    }
}

/// Reference per-sector x-rank at bond `b`, sector `s`: boundary bonds rank 1, interior
/// `min(R, max_rank)` — the step-03 convention.
fn x_rank(space: &QnSpace, b: usize, s: Sector, r: usize) -> usize {
    if b == 0 || b == space.d {
        1
    } else {
        r.min(space.max_rank(b, s) as usize)
    }
}

/// A FCIDUMP path, or a fixture name under `testdata/` whose directory holds exactly one
/// `*.FCIDUMP` — so the step runs bare on LiH and still accepts any dump on disk.
fn resolve_dump(arg: &str) -> std::path::PathBuf {
    let p = std::path::Path::new(arg);
    if p.is_file() {
        return p.to_path_buf();
    }
    let dir = qntt_exkit::repo!().fixture_dir(arg);
    let mut dumps: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{arg}: neither a file nor a fixture directory ({e})"))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "FCIDUMP"))
        .collect();
    match dumps.len() {
        1 => dumps.pop().unwrap(),
        n => panic!(
            "{}: expected exactly one *.FCIDUMP, found {n}",
            dir.display()
        ),
    }
}

fn main() {
    let m = Spec::new("step-04")
        .about("deterministic layout metrics for the automaton Hamiltonian construction")
        .pos(
            "fcidump",
            "a FCIDUMP path, or a testdata/ fixture name holding one",
            Val::Str,
            Def::Str("lih_sto3g"),
        )
        .pos(
            "R",
            "reference per-sector x-rank for the M3 rank sums",
            Val::Usize,
            Def::Usize(4),
        )
        .pos(
            "construction",
            "median|bipartite|persector|mediancover|bipmedian (case-insensitive)",
            Val::Str,
            Def::Str("bipartite"),
        )
        .opt(
            "tcdump",
            "paired transcorrelated dump; turns the 3-body w channel on",
            Val::Path,
            Def::Optional,
        )
        .pg()
        .parse();

    let dump = resolve_dump(m.str("fcidump"));
    let path = dump.to_str().expect("utf-8 path");
    let r: usize = m.usize("R");
    let construction = parse_construction(m.str("construction"));
    let tcdump: Option<String> = m.str_opt("tcdump").map(str::to_string);
    let pg_flag = match m.pg() {
        PgChoice::Unset => None,
        PgChoice::Auto => Some(None),
        PgChoice::Kind(k) => Some(Some(k)),
    };

    // Load either a plain FCIDUMP (w = None, 1+2-body) or a transcorrelated dump (w on, 3-body).
    // Both loaders yield the same downstream fields (t, v, norb, nelec, ms2); the w channel is the
    // only difference. Fields are moved out so the coefficient tensors need no clone.
    // With `--pg` the space carries the dump's ORBSYM table and ISYM target; without it no
    // point-group path is entered at all and `pg_label` is empty.
    let (t, v, w_opt, norb, nelec, ms2, space, pg_label) = match &tcdump {
        Some(tc) => {
            // `Pairswap`, not `Eightfold`: a transcorrelated dump breaks hermiticity, so only the
            // pair-swap relation survives. It is chosen here at the call site, never defaulted:
            // the same choice silently mis-reads an ordinary 8-fold-packed dump. The point group
            // is resolved below by `resolve_space`.
            let opts = IntegralsOptions {
                symmetry: TwoBodySymmetry::Pairswap,
                three_body: Some(ThreeBodyOptions::default()),
                ..IntegralsOptions::default()
            };
            let load = load_fcidump::<f64>(Path::new(path), Some(Path::new(tc)), &opts)
                .expect("load_fcidump");
            let (dat, w) = (load.ints, load.w);
            let pgd = pg_flag.map(|k| dat.point_group(k).expect("resolve ORBSYM/ISYM"));
            let (space, label) = resolve_space(dat.norb, dat.nelec, dat.ms2, pgd);
            (dat.t, dat.v, w, dat.norb, dat.nelec, dat.ms2, space, label)
        }
        None => {
            let dat = read_fcidump::<f64>(Path::new(path), TwoBodySymmetry::Eightfold)
                .expect("read_fcidump");
            let pgd = pg_flag.map(|k| dat.point_group(k).expect("resolve ORBSYM/ISYM"));
            let (space, label) = resolve_space(dat.norb, dat.nelec, dat.ms2, pgd);
            (
                dat.t, dat.v, None, dat.norb, dat.nelec, dat.ms2, space, label,
            )
        }
    };
    let w_dyn: Option<&dyn ThreeBody<f64>> = w_opt.as_ref().map(|w| w as &dyn ThreeBody<f64>);
    let d = space.d;
    banner(format!(
        "metrics_construction | {path} | norb={norb} nelec={nelec} ms2={ms2} | R={r}{pg_label}"
    ));
    println!(
        "CONSTRUCTION={}",
        match construction {
            Construction::Median => "Median",
            Construction::Bipartite => "Bipartite",
            Construction::BipartitePerSector => "BipartitePerSector",
            Construction::MedianCoverSwitch => "MedianCoverSwitch",
            Construction::BipartiteMedianSwitch => "BipartiteMedianSwitch",
        }
    );
    println!("W_CHANNEL={}", if w_dyn.is_some() { "on" } else { "off" });

    // Construction wall time (peak memory comes from a /usr/bin/time wrapper).
    let t0 = Instant::now();
    let h = SparseHamiltonian::new_with(space, &t, &v, w_dyn, f64::EPSILON, construction);
    let t_build = t0.elapsed();
    // Per-stage construction timings; compiled out in a default build, so nothing prints.
    if qntt::hamiltonian::construct_probe::ENABLED {
        print!(
            "{}",
            qntt::hamiltonian::construct_probe::snapshot().format()
        );
    }
    println!(
        "M5_CONSTRUCTION_WALL {:.3}s  (peak RSS: wrap with /usr/bin/time -l)",
        t_build.as_secs_f64()
    );

    // Per-bond scan: M1 channel counts, (sector, flux) classes, M3 rank sums, in the canonical
    // `bond_sectors` order.
    let abd = h.automaton_bond_dims();
    let mut m3_profile: Vec<usize> = Vec::with_capacity(d + 1);
    // Histogram of class sizes in power-of-two buckets: bucket i = sizes
    // [2^i, 2^(i+1)). Also track the largest blocks with their (bond, sector, flux) identity.
    let mut size_hist: BTreeMap<u32, usize> = BTreeMap::new();
    let mut top_blocks: Vec<(usize, usize, Sector, Flux)> = Vec::new(); // (size, b, s, flux)

    println!(
        "bond | channels | sectors | max/sector | flux-blocks | max-block | M3 ranksum (R={r})"
    );
    // Indexing `abd[b]` alongside per-bond scans — the range loop is the clearer shape here.
    #[allow(clippy::needless_range_loop)]
    for b in 0..=d {
        let sectors = space.bond_sectors(b);
        let mut nonzero_sectors = 0usize;
        let mut max_per_sector = 0usize;
        let mut nblocks = 0usize;
        let mut max_block = 0usize;
        let mut m3 = 0usize;
        for &s in &sectors {
            let fluxes = h.channel_fluxes(b, s);
            if !fluxes.is_empty() {
                nonzero_sectors += 1;
                max_per_sector = max_per_sector.max(fluxes.len());
            }
            // Channel classes by (sector, flux).
            let mut classes: BTreeMap<Flux, usize> = BTreeMap::new();
            for f in fluxes {
                *classes.entry(f).or_insert(0) += 1;
            }
            for (&flux, &size) in &classes {
                nblocks += 1;
                max_block = max_block.max(size);
                *size_hist.entry(size.ilog2()).or_insert(0) += 1;
                top_blocks.push((size, b, s, flux));
            }
            // M3: the rank of the untruncated H·x at (b, s) for the reference state ranks.
            m3 += h
                .state_qns_at(b, s)
                .iter()
                .map(|&m| x_rank(&space, b, m, r))
                .sum::<usize>();
        }
        m3_profile.push(m3);
        println!(
            "  {b:>2} | {:>8} | {nonzero_sectors:>7} | {max_per_sector:>10} | {nblocks:>11} | {max_block:>9} | {m3:>10}",
            abd[b]
        );
    }

    // M1 summary (same shape as step-03's AUTOMATON_BOND_DIM line).
    let abd_max = abd.iter().copied().max().unwrap_or(0);
    let abd_mean = abd.iter().sum::<usize>() as f64 / abd.len() as f64;
    println!("M1_BOND_DIM max={abd_max} mean={abd_mean:.1}  profile={abd:?}");

    // M2: per-site and total COO nnz.
    let nnz: Vec<usize> = (0..d).map(|k| h.coo_nnz(k)).collect();
    let nnz_total: usize = nnz.iter().sum();
    println!("M2_COO_NNZ total={nnz_total}  per_site={nnz:?}");

    // M3 summary: the largest untruncated H·x bond rank under the reference rank profile.
    let m3_max = m3_profile.iter().copied().max().unwrap_or(0);
    println!("M3_RANKSUM_R{r} max={m3_max}  profile={m3_profile:?}");

    // Distribution of per-(bond, sector, flux) class sizes.
    println!("FLUX_BLOCK_HISTOGRAM (class size 2^i..2^(i+1) -> #blocks over all bonds/sectors)");
    for (&log2, &count) in &size_hist {
        println!(
            "  {:>6}..{:<6} : {count}",
            1usize << log2,
            1usize << (log2 + 1)
        );
    }
    // Descending by size, ties broken by (bond, sector, flux) — deterministic top list.
    top_blocks
        .sort_unstable_by_key(|&(size, b, s, f)| (std::cmp::Reverse(size), b, s.nup, s.ndn, f));
    println!("TOP_FLUX_BLOCKS (size | bond | sector | flux) — the largest cover subproblems");
    for &(size, b, s, flux) in top_blocks.iter().take(10) {
        let (fu, fd) = (flux.up, flux.dn);
        println!(
            "  {size:>6} | b={b:>2} | ({},{}) | ({fu:+},{fd:+})",
            s.nup, s.ndn
        );
    }
}
