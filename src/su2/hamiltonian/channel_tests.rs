// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeMap;
use std::f64::consts::FRAC_PI_2;
use std::ops::Range;
use std::path::Path;

use rand::RngExt;

use super::*;
use crate::coeff::CoeffTensor;
use crate::hamiltonian::ChannelPhase;
use crate::integrals::{
    Integrals, IntegralsOptions, PointGroupPolicy, load_fcidump, load_fixture_dir,
};
use crate::ordering::{irrep_blocked_order, permute_integrals};
use crate::qn::Pooled;
use crate::rotation::{givens, rotate_integrals};
use crate::su2::core::Family;
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::assembly::SRC_NONE;
use crate::su2::hamiltonian::{StateT, Su2SparseHamiltonian, Su2TermScreen, Ts, rayleigh_quotient};
use crate::su2::qn::{Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::su2::twosite::{apply_mode_rotation, contract, factor_svd};
use crate::su2::vector::Su2TtVector;
use crate::test_util::{randomized_su2_state, seeded};

fn manifest(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn h4_ints() -> Integrals<f64> {
    let ints = load_fcidump::<f64>(
        &manifest("testdata/h4.FCIDUMP"),
        None,
        &IntegralsOptions::default(),
    )
    .expect("load h4.FCIDUMP")
    .ints;
    assert_eq!((ints.norb, ints.nelec), (4, 4));
    ints
}

fn lih_ints() -> Integrals<f64> {
    let ints = load_fixture_dir::<f64>(
        &manifest("testdata/lih_sto3g"),
        &IntegralsOptions::default(),
    )
    .expect("load lih_sto3g")
    .ints;
    assert_eq!((ints.norb, ints.nelec), (6, 4));
    ints
}

pub(crate) fn fixtures() -> Vec<(String, Integrals<f64>, Su2Space, u64)> {
    vec![
        (
            "h4 (4,4,0)".to_string(),
            h4_ints(),
            Su2Space::new(4, 4, 0),
            0x6000_0401,
        ),
        (
            "h4 (4,4,2)".to_string(),
            h4_ints(),
            Su2Space::new(4, 4, 2),
            0x6000_0402,
        ),
        (
            "lih (6,4,0)".to_string(),
            lih_ints(),
            Su2Space::new(6, 4, 0),
            0x6000_0601,
        ),
        (
            "lih (6,3,1)".to_string(),
            lih_ints(),
            Su2Space::new(6, 3, 1),
            0x6000_0602,
        ),
    ]
}

fn dense_ints(d: usize, seed: u64) -> Integrals<f64> {
    fn draw(rng: &mut rand_xoshiro::Xoshiro256PlusPlus) -> f64 {
        let w: f64 = rng.random_range(0.1..1.0);
        if rng.random_range(0..2) == 0 { w } else { -w }
    }
    let mut rng = seeded(seed);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in p..d {
            let w = draw(&mut rng);
            t.set(&[p, q], w);
            t.set(&[q, p], w);
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    let mut seen = std::collections::HashMap::new();
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let (a, b) = ((i.min(j), i.max(j)), (k.min(l), k.max(l)));
                    let key = (a.min(b), a.max(b));
                    let w = *seen.entry(key).or_insert_with(|| draw(&mut rng));
                    v.set(&[i, j, k, l], w);
                }
            }
        }
    }
    Integrals {
        t,
        v,
        e0: 0.0,
        norb: d,
        nelec: 0,
        ms2: 0,
        orbsym: Vec::new(),
        isym: None,
    }
}

pub(crate) fn dense_fixture() -> (String, Integrals<f64>, Su2Space, u64) {
    (
        "dense d=7 (7,5,1)".to_string(),
        dense_ints(7, 0x6000_0701),
        Su2Space::new(7, 5, 1),
        0x6000_0702,
    )
}

fn refillable(ints: &Integrals<f64>, space: Su2Space) -> Su2SparseHamiltonian<f64> {
    Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::SymmetryAllowed(space.pg),
    )
}

fn label(mask: u8, open: [u16; 2]) -> Ts {
    (
        StateT {
            mask,
            ..StateT::ZERO
        },
        open,
    )
}

fn outside(b: usize, i: usize) -> bool {
    b <= i || b >= i + 2
}

#[test]
fn g0_tagger_decodes_hand_made_labels() {
    use ChannelPhase::{Comp, Normal};
    assert_eq!(
        su2_channel_info(&label(0b1, [4, 0]), 4),
        (Normal, [Some(3), None])
    );
    assert_eq!(
        su2_channel_info(&label(0b1, [4, 0]), 3),
        (Comp, [Some(3), None])
    );
    assert_eq!(
        su2_channel_info(&label(0b11, [5, 2]), 5),
        (Normal, [Some(4), Some(1)])
    );
    assert_eq!(
        su2_channel_info(&label(0b11, [5, 2]), 1),
        (Comp, [Some(4), Some(1)])
    );
    assert_eq!(
        su2_channel_info(&label(0b1, [1, 0]), 1),
        (Normal, [Some(0), None])
    );
    for b in 1..6 {
        assert_eq!(
            su2_channel_info(&label(0, [0, 0]), b),
            (Normal, [None, None]),
            "empty open, mask 0, bond {b}"
        );
        assert_eq!(
            su2_channel_info(&label(0b101, [0, 0]), b),
            (Comp, [None, None]),
            "empty open, mask ≠ 0, bond {b}"
        );
    }

    let cases: [([u16; 2], Touch); 13] = [
        ([0, 0], Touch::None),
        ([2, 0], Touch::None),
        ([5, 0], Touch::None),
        ([2, 5], Touch::None),
        ([3, 0], Touch::A),
        ([4, 0], Touch::A),
        ([3, 6], Touch::A),
        ([1, 4], Touch::A),
        ([3, 4], Touch::B),
        ([4, 3], Touch::B),
        ([3, 3], Touch::B),
        ([4, 4], Touch::B),
        ([4, 2], Touch::A),
    ];
    for (open, want) in cases {
        for mask in [0u8, 0b1, 0b1111] {
            assert_eq!(
                touch(&label(mask, open), 2),
                want,
                "touch({open:?}, mask {mask:#b}, pair (2, 3))"
            );
        }
    }
}

#[test]
fn g0_every_interior_channel_is_tagged_and_the_index_is_its_recount() {
    for (name, ints, space, seed) in fixtures() {
        let d = space.d;
        let h = refillable(&ints, space);
        let idx = Su2TouchIndex::new(&h);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut seeded(seed));
        let i0 = (d - 2) / 2;
        println!(
            "{name}: census at pair ({i0}, {}) — bond: channels / touched / A / B | rank-weighted",
            i0 + 1
        );
        let mut tagged = 0usize;
        let (mut dom_n, mut dom_t, mut dom_b) = (0usize, 0usize, 0usize);
        let (mut dom_tw, mut dom_bw) = (0usize, 0usize);
        for b in 1..d {
            let ranks = &x.core(b).row_ranks;
            let hrows = h.row_ranks(x.core(b));
            for i in 0..d - 1 {
                let mut want: Vec<(Su2Sector, Vec<usize>)> = Vec::new();
                let (mut n, mut na, mut nb) = (0usize, 0usize, 0usize);
                let (mut nw, mut naw, mut nbw) = (0usize, 0usize, 0usize);
                let mut ext_rows = 0usize;
                for s in space.bond_sectors(b) {
                    let descs = h.channel_descs(b, s);
                    let qns = h.state_qns_at(b, s);
                    assert_eq!(
                        descs.len(),
                        qns.len(),
                        "{name}: bond {b} sector {s:?}: labels and x-sectors not index-aligned"
                    );
                    ext_rows += hrows[s];
                    let mut cs: Vec<usize> = Vec::new();
                    for (c, (ts, &m)) in descs.iter().zip(qns).enumerate() {
                        let (phase, sites) = su2_channel_info(ts, b);
                        for (k, (site, raw)) in sites.iter().zip(ts.1).enumerate() {
                            assert_eq!(
                                *site,
                                raw.checked_sub(1).map(usize::from),
                                "{name}: bond {b} sector {s:?} channel {c}: slot {k} decode"
                            );
                        }
                        assert!(
                            sites[0].is_some() || sites[1].is_none(),
                            "{name}: bond {b} sector {s:?} channel {c}: second slot without a first"
                        );
                        assert!(
                            sites.iter().flatten().all(|&p| p < d),
                            "{name}: bond {b} sector {s:?} channel {c}: site out of range {sites:?}"
                        );
                        let on = sites
                            .iter()
                            .flatten()
                            .filter(|&&p| p == i || p == i + 1)
                            .count();
                        let want_touch = match on {
                            0 => Touch::None,
                            1 => Touch::A,
                            _ => Touch::B,
                        };
                        assert_eq!(
                            touch(ts, i),
                            want_touch,
                            "{name}: bond {b} sector {s:?} channel {c} {sites:?}, pair ({i}, {})",
                            i + 1
                        );
                        if want_touch != Touch::None {
                            if b >= i + 2 {
                                assert_eq!(
                                    phase,
                                    ChannelPhase::Normal,
                                    "{name}: touched right-env channel bond {b} pair {i} {sites:?}"
                                );
                            } else if b <= i {
                                assert_eq!(
                                    phase,
                                    ChannelPhase::Comp,
                                    "{name}: touched left-env channel bond {b} pair {i} {sites:?}"
                                );
                            }
                        }
                        let w = ranks[m];
                        n += 1;
                        nw += w;
                        match want_touch {
                            Touch::None => {}
                            Touch::A => {
                                na += 1;
                                naw += w;
                                cs.push(c);
                            }
                            Touch::B => {
                                nb += 1;
                                nbw += w;
                                cs.push(c);
                            }
                        }
                        if i == 0 {
                            tagged += 1;
                        }
                    }
                    if !cs.is_empty() {
                        want.push((s, cs));
                    }
                }
                assert_eq!(
                    idx.touched(b, i),
                    want,
                    "{name}: touched({b}, {i}) is not the recount"
                );
                assert_eq!(
                    idx.census(b, i, None),
                    Su2TouchCensus {
                        channels: n,
                        touched: na + nb,
                        class_a: na,
                        class_b: nb,
                    },
                    "{name}: census({b}, {i}) counts"
                );
                assert_eq!(
                    idx.census(b, i, Some(ranks)),
                    Su2TouchCensus {
                        channels: nw,
                        touched: naw + nbw,
                        class_a: naw,
                        class_b: nbw,
                    },
                    "{name}: census({b}, {i}) rank-weighted"
                );
                assert_eq!(
                    nw, ext_rows,
                    "{name}: bond {b}: the rank-weighted channel count is not the operator's \
                     H-extended row count"
                );
                if outside(b, i) {
                    dom_n += n;
                    dom_t += na + nb;
                    dom_b += nb;
                    dom_tw += naw + nbw;
                    dom_bw += nbw;
                }
                if i == i0 {
                    println!(
                        "  bond {b}{}: {n} / {} / {na} / {nb} | {nw} / {} / {naw} / {nbw}",
                        if outside(b, i) { "" } else { " (inside)" },
                        na + nb,
                        naw + nbw
                    );
                }
            }
        }
        println!(
            "{name}: {tagged} interior channels tagged; gauge domain: {dom_t} of {dom_n} \
             (bond, pair, channel) instances touched, class-B fraction {:.3} (rank-weighted {:.3})",
            dom_b as f64 / dom_t.max(1) as f64,
            dom_bw as f64 / dom_tw.max(1) as f64
        );
        assert!(tagged > 0, "{name}: no interior channel");
        assert!(
            dom_b > 0,
            "{name}: class B never realized in the gauge domain"
        );
        assert!(
            dom_t > dom_b,
            "{name}: class A never realized in the gauge domain"
        );
        assert!(
            dom_n > dom_t,
            "{name}: every gauge-domain channel touched — pin (c) would be vacuous"
        );
    }
}

#[test]
fn g0_pin_a_every_interior_label_is_phase_pure() {
    for (name, ints, space, _) in fixtures() {
        let d = space.d;
        let h = refillable(&ints, space);
        let mut cases = [0usize; 4];
        for b in 1..d {
            for s in space.bond_sectors(b) {
                for (c, ts) in h.channel_descs(b, s).iter().enumerate() {
                    let (phase, sites) = su2_channel_info(ts, b);
                    let open: Vec<usize> = sites.iter().flatten().copied().collect();
                    let applied = open.iter().all(|&p| p < b);
                    let remaining = open.iter().all(|&p| p >= b);
                    assert!(
                        applied || remaining,
                        "{name}: bond {b} sector {s:?} channel {c}: open {open:?} straddles the bond"
                    );
                    let mask = ts.0.mask;
                    let want = match (open.is_empty(), applied, mask == 0) {
                        (true, _, true) => ChannelPhase::Normal,
                        (true, _, false) => ChannelPhase::Comp,
                        (false, true, _) => ChannelPhase::Normal,
                        (false, false, _) => ChannelPhase::Comp,
                    };
                    assert_eq!(
                        phase, want,
                        "{name}: bond {b} sector {s:?} channel {c}: open {open:?} mask {mask:#b}"
                    );
                    let napp = mask.count_ones() as usize;
                    match phase {
                        ChannelPhase::Normal => assert_eq!(
                            napp,
                            open.len(),
                            "{name}: bond {b} sector {s:?} channel {c}: a Normal label must list \
                             every applied token (open {open:?}, mask {mask:#b})"
                        ),
                        ChannelPhase::Comp => assert!(
                            napp >= open.len().max(1),
                            "{name}: bond {b} sector {s:?} channel {c}: a Comp label lists the \
                             smaller set (open {open:?}, mask {mask:#b})"
                        ),
                    }
                    let k = match (phase, open.is_empty()) {
                        (ChannelPhase::Normal, false) => 0,
                        (ChannelPhase::Normal, true) => 1,
                        (ChannelPhase::Comp, false) => 2,
                        (ChannelPhase::Comp, true) => 3,
                    };
                    cases[k] += 1;
                }
            }
        }
        println!(
            "{name}: pin (a) Normal open/empty {} / {}, Comp open/empty {} / {}",
            cases[0], cases[1], cases[2], cases[3]
        );
        assert!(
            cases.iter().all(|&n| n > 0),
            "{name}: a phase case is unrealized: {cases:?}"
        );
    }
}

#[test]
fn g0_pin_b_term_slots_cross_the_median_normal_to_comp() {
    for (name, ints, space, _) in fixtures() {
        let d = space.d;
        let h = refillable(&ints, space);
        assert!(h.is_refillable());
        let (mut term_rows, mut term_cols, mut term_both) = (0usize, 0usize, 0usize);
        let (mut transport_both, mut mixed_interior, mut slots) = (0usize, 0usize, 0usize);
        for k in 0..d {
            let g = space.pg.irrep_of(k);
            let nest = &h.nests[k];
            for y_occ in Family::ALL {
                for n1 in space.bond_sectors(k) {
                    let cells = nest.slot(y_occ, n1);
                    if cells.is_empty() {
                        continue;
                    }
                    let q = y_occ.apply(n1, g).unwrap_or_else(|| {
                        panic!("{name}: site {k} slot ({y_occ:?}, {n1:?}) has no column sector")
                    });
                    for cell in cells {
                        let src = cell
                            .src
                            .as_deref()
                            .expect("a refillable build carries provenance on every cell");
                        for (p, (r, c, _)) in cell.csr.entries().enumerate() {
                            slots += 1;
                            let (mut has_term, mut has_transport) = (false, false);
                            for (term, _) in src.run(p) {
                                if term == SRC_NONE {
                                    has_transport = true;
                                } else {
                                    has_term = true;
                                }
                            }
                            assert!(
                                has_term || has_transport,
                                "{name}: site {k}: empty run at slot {p}"
                            );
                            let row = (k >= 1).then(|| {
                                let descs = h.channel_descs(k, n1);
                                assert!(
                                    (r as usize) < descs.len(),
                                    "{name}: site {k}: row {r} outside {n1:?}'s channels"
                                );
                                su2_channel_info(&descs[r as usize], k)
                            });
                            let col = (k + 1 < d).then(|| {
                                let descs = h.channel_descs(k + 1, q);
                                assert!(
                                    (c as usize) < descs.len(),
                                    "{name}: site {k}: column {c} outside {q:?}'s channels"
                                );
                                su2_channel_info(&descs[c as usize], k + 1)
                            });
                            if has_term {
                                if let Some((ph, sites)) = row {
                                    assert_eq!(
                                        ph,
                                        ChannelPhase::Normal,
                                        "{name}: site {k} slot ({r}, {c}): a term deposit's row \
                                         must be Normal at bond {k} (open {sites:?})"
                                    );
                                    term_rows += 1;
                                }
                                if let Some((ph, sites)) = col {
                                    assert_eq!(
                                        ph,
                                        ChannelPhase::Comp,
                                        "{name}: site {k} slot ({r}, {c}): a term deposit's \
                                         column must be Comp at bond {} (open {sites:?})",
                                        k + 1
                                    );
                                    term_cols += 1;
                                }
                                if row.is_some() && col.is_some() {
                                    term_both += 1;
                                    mixed_interior += usize::from(has_transport);
                                }
                            } else if let (Some((pr, sr)), Some((pc, sc))) = (row, col) {
                                assert_eq!(
                                    pr, pc,
                                    "{name}: site {k} slot ({r}, {c}): a transport-only slot \
                                     changes phase ({sr:?} → {sc:?})"
                                );
                                transport_both += 1;
                            }
                        }
                    }
                }
            }
        }
        println!(
            "{name}: pin (b) {slots} slots; term slots checked row {term_rows}, column \
             {term_cols}, both {term_both} (of which transport+term {mixed_interior}); interior \
             transport-only {transport_both}"
        );
        assert!(
            term_rows > 0 && term_cols > 0 && term_both > 0 && transport_both > 0,
            "{name}: pin (b) is vacuous on a side"
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Axis {
    Rows,
    Cols,
}

#[derive(Clone, Copy, Debug, Default)]
struct BandDiff {
    untouched_max: f64,
    untouched_entries: usize,
    untouched_nonbitwise: usize,
    untouched_content: f64,
    touched_max: f64,
    scale: f64,
}

fn band_diff(
    h: &Su2SparseHamiltonian<f64>,
    b: usize,
    stale: &Su2Sketch<f64>,
    fresh: &Su2Sketch<f64>,
    bands: &Su2SectorGrid<Vec<Range<usize>>>,
    touched: &[(Su2Sector, Vec<usize>)],
    axis: Axis,
) -> BandDiff {
    let mut out = BandDiff::default();
    for s in stale.sectors() {
        let (a, f) = match (stale.block(s), fresh.block(s)) {
            (Some(a), Some(f)) => (a, f),
            (None, None) => continue,
            _ => panic!("bond {b} sector {s:?}: block presence differs stale vs fresh"),
        };
        assert_eq!(
            (a.nrows(), a.ncols()),
            (f.nrows(), f.ncols()),
            "bond {b} sector {s:?}: stale/fresh shapes differ"
        );
        let (len, other) = match axis {
            Axis::Rows => (a.nrows(), a.ncols()),
            Axis::Cols => (a.ncols(), a.nrows()),
        };
        let rgs = &bands[s];
        assert_eq!(
            rgs.last().map_or(0, |r| r.end),
            len,
            "bond {b} sector {s:?}: the bands do not tile the {axis:?} axis"
        );
        if !rgs.is_empty() {
            assert_eq!(
                rgs.len(),
                h.channel_descs(b, s).len(),
                "bond {b} sector {s:?}: one band per channel"
            );
        }
        let tset: &[usize] = touched
            .iter()
            .find(|(t, _)| *t == s)
            .map_or(&[][..], |(_, v)| v.as_slice());
        for (c, rg) in rgs.iter().enumerate() {
            let is_touched = tset.binary_search(&c).is_ok();
            for u in rg.clone() {
                for o in 0..other {
                    let (r, col) = match axis {
                        Axis::Rows => (u, o),
                        Axis::Cols => (o, u),
                    };
                    let (x0, x1) = (a[(r, col)], f[(r, col)]);
                    let dev = (x0 - x1).abs();
                    if is_touched {
                        out.touched_max = out.touched_max.max(dev);
                    } else {
                        out.untouched_max = out.untouched_max.max(dev);
                        out.untouched_entries += 1;
                        out.untouched_nonbitwise += usize::from(x0.to_bits() != x1.to_bits());
                        out.untouched_content = out.untouched_content.max(x1.abs());
                    }
                }
            }
        }
        for r in 0..f.nrows() {
            for col in 0..f.ncols() {
                out.scale = out.scale.max(f[(r, col)].abs());
            }
        }
    }
    out
}

#[test]
fn g0_pin_c_untouched_env_bands_survive_refill_and_pair_rotation() {
    for (name, ints, space, seed) in fixtures().into_iter().chain([dense_fixture()]) {
        let d = space.d;
        let mut h = refillable(&ints, space);
        let idx = Su2TouchIndex::new(&h);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut seeded(seed));
        let fl0 = left_to_right_sketching(&h, &x);
        let fr0 = right_to_left_sketching(&h, &x);
        let e0 = rayleigh_quotient(&h, &x);
        let (mut worst_rel, mut entries, mut nonbitwise, mut content) = (0.0f64, 0usize, 0, 0.0f64);
        let mut weakest_touched = f64::INFINITY;
        let (mut moving, mut symmetric) = (0usize, Vec::new());
        for i in 0..d - 1 {
            for theta in [0.37 + 0.1 * i as f64, FRAC_PI_2] {
                let rot = rotate_integrals(&ints, &givens(d, i, theta));
                let moved = rot
                    .t
                    .data
                    .iter()
                    .zip(&ints.t.data)
                    .chain(rot.v.data.iter().zip(&ints.v.data))
                    .fold(0.0f64, |m, (p, q)| m.max((p - q).abs()));
                h.refill(&rot.t, &rot.v, None);
                let tc = apply_mode_rotation(&contract(x.core(i), x.core(i + 1)), theta);
                let (a, sv, mut b1) = factor_svd(&tc, 0.0, Pooled::NONE, false);
                b1.fold_sketch_left_mut(&sv);
                let mut y = x.clone();
                y.set_cores(a, b1);
                let e1 = rayleigh_quotient(&h, &y);
                assert!(
                    (e1 - e0).abs() <= 1e-10 * e0.abs().max(1.0),
                    "{name}: pair {i} θ {theta}: ⟨y,H′y⟩ = {e1:.15e} vs ⟨x,Hx⟩ = {e0:.15e} — the \
                     state gate is not the integral rotation's partner"
                );
                let fl1 = left_to_right_sketching(&h, &y);
                let fr1 = right_to_left_sketching(&h, &y);
                let (mut pair_untouched, mut pair_entries, mut pair_touched) =
                    (0.0f64, 0usize, 0.0f64);
                let mut tally = |diff: BandDiff, stack: &str, bond: usize| {
                    let scale = diff.scale.max(f64::MIN_POSITIVE);
                    let rel = diff.untouched_max / scale;
                    assert!(
                        rel <= 1e-14,
                        "{name}: pair {i} θ {theta}: {stack} at bond {bond}: an UNTOUCHED band \
                         moved by {rel:.3e} (relative) — the index misses a touched channel"
                    );
                    pair_untouched = pair_untouched.max(rel);
                    pair_entries += diff.untouched_entries;
                    pair_touched = pair_touched.max(diff.touched_max / scale);
                    nonbitwise += diff.untouched_nonbitwise;
                    content = content.max(diff.untouched_content);
                };
                for b in 1..=i {
                    assert_eq!(fl0[b].site(), b);
                    assert_eq!(fl1[b].site(), b);
                    let bands = h.col_ranges(x.core(b - 1));
                    let diff = band_diff(
                        &h,
                        b,
                        &fl0[b],
                        &fl1[b],
                        &bands,
                        &idx.touched(b, i),
                        Axis::Cols,
                    );
                    tally(diff, "Fᴸ", b);
                }
                for j in i + 2..d {
                    assert_eq!(fr0[j - 1].site(), j);
                    assert_eq!(fr1[j - 1].site(), j);
                    let bands = h.row_ranges(x.core(j));
                    let diff = band_diff(
                        &h,
                        j,
                        &fr0[j - 1],
                        &fr1[j - 1],
                        &bands,
                        &idx.touched(j, i),
                        Axis::Rows,
                    );
                    tally(diff, "Fᴿ", j);
                }
                println!(
                    "{name}: pair ({i}, {}) θ {theta:.4}: untouched max rel {pair_untouched:.2e} \
                     over {pair_entries} entries; touched max rel {pair_touched:.2e}",
                    i + 1
                );
                assert!(
                    pair_entries > 0,
                    "{name}: pair {i} θ {theta}: no untouched band compared — pin (c) is vacuous"
                );
                if moved <= 1e-10 {
                    assert!(
                        pair_touched <= 1e-12,
                        "{name}: pair {i} θ {theta}: the integrals are invariant (max Δ \
                         {moved:.1e}) yet a touched band moved by {pair_touched:.3e}"
                    );
                    symmetric.push((i, theta, moved));
                } else {
                    assert!(
                        pair_touched > 1e-6,
                        "{name}: pair {i} θ {theta}: the integrals moved (max Δ {moved:.1e}) but \
                         no touched band did (max rel {pair_touched:.3e}) — the refill did not \
                         reach the stacks"
                    );
                    moving += 1;
                    weakest_touched = weakest_touched.min(pair_touched);
                }
                worst_rel = worst_rel.max(pair_untouched);
                entries += pair_entries;
            }
        }
        println!(
            "{name}: pin (c) max untouched deviation {worst_rel:.3e} (relative) over {entries} \
             entries; {nonbitwise} not bitwise → bitwise: {}; weakest touched move over the \
             {moving} moving (pair, θ) {weakest_touched:.3e}; symmetry (pair, θ, max Δ) \
             {symmetric:?}",
            nonbitwise == 0
        );
        assert!(
            content > 0.0,
            "{name}: every untouched band compared is identically zero — pin (c) is vacuous"
        );
        assert!(
            moving > 0,
            "{name}: no (pair, θ) moved the integrals — pin (c) is vacuous"
        );
    }
}

#[test]
fn g0_class_a_closure_holds_on_the_symmetry_allowed_fixtures() {
    let mut shared_label = 0usize;
    for (name, ints, space, _) in fixtures() {
        let d = space.d;
        let h = refillable(&ints, space);
        let idx = Su2TouchIndex::new(&h);
        let mut class_a = 0usize;
        for i in 0..d - 1 {
            for b in (1..d).filter(|&b| outside(b, i)) {
                class_a += idx.census(b, i, None).class_a;
            }
        }
        for b in 1..d {
            for s in space.bond_sectors(b) {
                let descs = h.channel_descs(b, s);
                shared_label += (1..descs.len())
                    .filter(|&c| descs[..c].contains(&descs[c]))
                    .count();
            }
        }
        let verdict = su2_env_gauge_closed(&h);
        println!(
            "{name}: class-A closure over {class_a} (bond, pair, channel) instances: {verdict:?}"
        );
        assert!(class_a > 0, "{name}: no class-A channel — closure vacuous");
        assert!(
            verdict.is_ok(),
            "{name}: class-A closure FAILS: {} — STOP and report the miss (spec §2)",
            verdict.unwrap_err()
        );
    }
    println!("channels sharing a label with an earlier channel (another m): {shared_label}");
    assert!(
        shared_label > 0,
        "no sector holds one label under two x-spins — the closure's same-m lookup is untested"
    );
}

fn hop_skeleton(hops: &[(usize, usize, f64)]) -> Su2SparseHamiltonian<f64> {
    let d = 4;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q, w) in hops {
        t.set(&[p, q], w);
        t.set(&[q, p], w);
    }
    Su2SparseHamiltonian::new_refillable(
        Su2Space::new(d, 2, 0),
        &t,
        &CoeffTensor::<f64>::zeros(d, 4),
        None,
        Su2TermScreen::Magnitude(f64::EPSILON),
    )
}

#[test]
fn g0_class_a_closure_reports_a_missing_relabelled_channel() {
    let lone = hop_skeleton(&[(0, 2, 1.0)]);
    let paired = hop_skeleton(&[(0, 2, 1.0), (0, 3, 0.5)]);
    let lone_census = Su2TouchIndex::new(&lone).census(2, 2, None);
    println!("lone hop: census(bond 2, pair (2, 3)) = {lone_census:?}");
    assert!(
        lone_census.class_a > 0,
        "the lone hop's bond-2 label must be class A for the pair (2, 3)"
    );
    let miss = su2_env_gauge_closed(&lone);
    println!("lone hop: {miss:?}");
    assert!(
        miss.is_err(),
        "the relabelled twin of the lone hop's bond-2 channel is absent — closure must fail"
    );
    let ok = su2_env_gauge_closed(&paired);
    println!("paired hops: {ok:?}");
    assert!(ok.is_ok(), "adding the relabelled hop must close: {ok:?}");
}

#[test]
fn g0_class_a_closure_reports_a_missing_downward_twin() {
    let lone = hop_skeleton(&[(0, 3, 1.0)]);
    let paired = hop_skeleton(&[(0, 2, 0.5), (0, 3, 1.0)]);
    let idx = Su2TouchIndex::new(&lone);
    let d = lone.space.d;
    let mut class_a = 0usize;
    for i in 0..d - 1 {
        for b in (1..d).filter(|&b| outside(b, i)) {
            class_a += idx.census(b, i, None).class_a;
        }
    }
    let here = idx.census(2, 2, None);
    println!(
        "lone hop t₀₃: census(bond 2, pair (2, 3)) = {here:?}; gauge-domain class A {class_a}"
    );
    assert!(
        here.class_a > 0,
        "the lone hop t₀₃'s bond-2 label must be class A for the pair (2, 3)"
    );
    assert_eq!(
        class_a, here.class_a,
        "the lone hop t₀₃ has a class-A channel outside (bond 2, pair (2, 3))"
    );
    for (s, cs) in idx.touched_ref(2, 2) {
        for &c in cs {
            let ts = &lone.channel_descs(2, *s)[c];
            let (phase, sites) = su2_channel_info(ts, 2);
            assert_eq!(
                (phase, sites),
                (ChannelPhase::Comp, [Some(3), None]),
                "bond 2 sector {s:?} channel {c}: expected the Comp label on the high site 3"
            );
        }
    }
    let miss = su2_env_gauge_closed(&lone);
    println!("lone hop t₀₃: {miss:?}");
    let msg = miss.expect_err(
        "the downward twin (site 2) of the lone hop's bond-2 channel is absent — closure must fail",
    );
    assert!(
        msg.contains("relabelled on pair (2, 3) to open [Some(2), None]"),
        "the miss must be the downward twin on the pair (2, 3): {msg}"
    );
    let ok = su2_env_gauge_closed(&paired);
    println!("hops t₀₂ + t₀₃: {ok:?}");
    assert!(ok.is_ok(), "adding the relabelled hop must close: {ok:?}");
}

#[test]
#[should_panic(expected = "refuses a non-refillable operator")]
fn g0_touch_index_refuses_a_plain_build() {
    let ints = h4_ints();
    let h = Su2SparseHamiltonian::new(Su2Space::new(4, 4, 0), &ints.t, &ints.v, None, f64::EPSILON);
    assert!(!h.is_refillable());
    let _ = Su2TouchIndex::new(&h);
}

#[test]
#[should_panic(expected = "refuses a non-refillable operator")]
fn g0_closure_refuses_a_plain_build() {
    let ints = h4_ints();
    let h = Su2SparseHamiltonian::new(Su2Space::new(4, 4, 0), &ints.t, &ints.v, None, f64::EPSILON);
    assert!(!h.is_refillable());
    let _ = su2_env_gauge_closed(&h);
}

#[test]
#[should_panic(expected = "not an interior bond")]
fn g0_touched_refuses_bond_0() {
    let space = Su2Space::new(4, 4, 0);
    let idx = Su2TouchIndex::new(&refillable(&h4_ints(), space));
    let _ = idx.touched(0, 1);
}

#[test]
#[should_panic(expected = "not an interior bond")]
fn g0_census_refuses_bond_d() {
    let space = Su2Space::new(4, 4, 0);
    let idx = Su2TouchIndex::new(&refillable(&h4_ints(), space));
    let _ = idx.census(4, 1, None);
}

fn typed_h2o_fixtures() -> Vec<(String, Integrals<f64>, Su2Space, u64)> {
    [(0usize, 0x6000_0711u64), (2, 0x6000_0712)]
        .into_iter()
        .map(|(two_s, seed)| {
            let o = IntegralsOptions {
                point_group: PointGroupPolicy::Resolve(None),
                screen: true,
                ..IntegralsOptions::default()
            };
            let load = load_fixture_dir::<f64>(&manifest("testdata/h2o_sto3g"), &o)
                .expect("load h2o_sto3g typed");
            let (pg0, d) = (load.pg.pg, load.ints.norb);
            assert!(!pg0.is_trivial(), "h2o_sto3g must resolve C2v");
            let perm = irrep_blocked_order(pg0, d);
            let ints = permute_integrals(&load.ints, &perm);
            let pgd = ints
                .point_group(Some(load.pg.kind))
                .expect("permuted dump resolves");
            let same: Vec<usize> = (0..d - 1)
                .filter(|&k| pgd.pg.irrep_of(k) == pgd.pg.irrep_of(k + 1))
                .collect();
            assert_eq!(
                same,
                vec![0, 1, 2, 4],
                "irrep-blocked h2o C2v same-irrep pairs"
            );
            let space = Su2Space::with_point_group(d, ints.nelec, two_s, pgd.pg, pgd.target);
            (format!("h2o C2v blocked (7,10,{two_s})"), ints, space, seed)
        })
        .collect()
}

fn in_b_window(d: usize, b: usize, i: usize) -> bool {
    let hd = (d - 1) / 2;
    (b >= i + 2 && b <= hd) || (b <= i && b > hd)
}

fn pair_slot(ts: &Ts, i: usize) -> (usize, bool) {
    let k =
        ts.1.iter()
            .position(|&r| usize::from(r) == i + 1 || usize::from(r) == i + 2)
            .expect("a class-A label has a slot on the pair");
    (k, usize::from(ts.1[k]) == i + 2)
}

fn assert_sector_lists<V>(lists: &[(Su2Sector, Vec<V>)], sectors: &[Su2Sector], what: &str) {
    let mut last: Option<usize> = None;
    for (s, v) in lists {
        let k = sectors
            .iter()
            .position(|t| t == s)
            .unwrap_or_else(|| panic!("{what}: sector {s:?} is not a sector of the bond"));
        assert!(
            last.is_none_or(|l| k > l),
            "{what}: sectors not strictly in bond order at {s:?}"
        );
        assert!(!v.is_empty(), "{what}: sector {s:?} listed with no entry");
        last = Some(k);
    }
}

#[test]
fn g2b_t1_plan_partitions_the_touched_set_into_twin_pairs_and_class_b() {
    let mut all = fixtures();
    all.push(dense_fixture());
    all.extend(typed_h2o_fixtures());
    for (name, ints, space, seed) in all {
        let (d, pg) = (space.d, space.pg);
        let h = refillable(&ints, space);
        let idx = Su2TouchIndex::new(&h);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut seeded(seed));
        let (mut pairs_left, mut pairs_right, mut class_b, mut b_cells) = (0usize, 0, 0, 0);
        let (mut widths, mut window_b) = (0usize, 0usize);
        for b in 1..d {
            let sectors = space.bond_sectors(b);
            let cols = h.col_ranges(x.core(b - 1));
            let rows = h.row_ranges(x.core(b));
            for i in 0..d - 1 {
                let what = format!("{name}: bond {b}, pair ({i}, {})", i + 1);
                let mix = idx.mix_ref(b, i);
                let rec = idx.recontract_ref(b, i);
                let touched = idx.touched_ref(b, i);
                if outside(b, i) {
                    for (s, cs) in touched {
                        let descs = h.channel_descs(b, *s);
                        for &c in cs {
                            if touch(&descs[c], i) == Touch::B {
                                assert!(
                                    in_b_window(d, b, i),
                                    "{what}: class-B channel {c} of sector {s:?} (open {:?}) \
                                     outside spec §2's window (hd = {})",
                                    su2_channel_info(&descs[c], b).1,
                                    (d - 1) / 2
                                );
                                window_b += 1;
                            }
                        }
                    }
                }
                let domain = outside(b, i) && pg.irrep_of(i) == pg.irrep_of(i + 1);
                if !domain {
                    assert!(
                        mix.is_empty() && rec.is_empty(),
                        "{what}: off the gauge domain (pair bond or cross-irrep pair) the plan \
                         must be empty: mix {mix:?}, recontract {rec:?}"
                    );
                    continue;
                }
                assert_sector_lists(mix, &sectors, &format!("{what}: mix_ref"));
                assert_sector_lists(rec, &sectors, &format!("{what}: recontract_ref"));
                let bands = if b <= i { &cols } else { &rows };
                let mut union: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
                let ord = |s: &Su2Sector| {
                    sectors
                        .iter()
                        .position(|t| t == s)
                        .expect("checked by assert_sector_lists")
                };
                for (s, ps) in mix {
                    let descs = h.channel_descs(b, *s);
                    let qns = h.state_qns_at(b, *s);
                    let rg = &bands[*s];
                    if !rg.is_empty() {
                        assert_eq!(
                            rg.len(),
                            descs.len(),
                            "{what}: sector {s:?}: one band per channel"
                        );
                    }
                    let mut prev: Option<usize> = None;
                    for &[lo, hi] in ps {
                        let (lo, hi) = (lo as usize, hi as usize);
                        assert!(
                            lo < descs.len() && hi < descs.len(),
                            "{what}: sector {s:?}: pair [{lo}, {hi}] outside its {} channels",
                            descs.len()
                        );
                        assert!(
                            prev.is_none_or(|p| lo > p),
                            "{what}: sector {s:?}: lo {lo} not strictly ascending"
                        );
                        prev = Some(lo);
                        let (tl, th) = (&descs[lo], &descs[hi]);
                        assert_eq!(touch(tl, i), Touch::A, "{what}: lo {lo} is not class A");
                        assert_eq!(touch(th, i), Touch::A, "{what}: hi {hi} is not class A");
                        let (kl, lo_on_hi) = pair_slot(tl, i);
                        let (kh, hi_on_hi) = pair_slot(th, i);
                        assert!(
                            !lo_on_hi && hi_on_hi,
                            "{what}: sector {s:?}: lo {lo} (open {:?}) must hold site {i} and hi \
                             {hi} (open {:?}) site {}",
                            su2_channel_info(tl, b).1,
                            su2_channel_info(th, b).1,
                            i + 1
                        );
                        assert_eq!(
                            kl, kh,
                            "{what}: sector {s:?}: lo {lo} / hi {hi} relabel different slots"
                        );
                        assert_eq!(
                            tl.0, th.0,
                            "{what}: sector {s:?}: lo {lo} / hi {hi} differ in StateT"
                        );
                        assert_eq!(
                            tl.1[1 - kl],
                            th.1[1 - kh],
                            "{what}: sector {s:?}: lo {lo} / hi {hi} differ in the other slot"
                        );
                        assert_eq!(
                            qns[lo], qns[hi],
                            "{what}: sector {s:?}: lo {lo} / hi {hi} differ in x-sector m"
                        );
                        if !rg.is_empty() {
                            assert_eq!(
                                rg[lo].len(),
                                rg[hi].len(),
                                "{what}: sector {s:?}: twin bands [{lo}, {hi}] of unequal widths"
                            );
                            widths += 1;
                        }
                        union.entry(ord(s)).or_default().extend([lo, hi]);
                    }
                    if b <= i {
                        pairs_left += ps.len();
                    } else {
                        pairs_right += ps.len();
                    }
                }
                for (s, cs) in rec {
                    union.entry(ord(s)).or_default().extend(cs.iter().copied());
                }
                let got: Vec<(Su2Sector, Vec<usize>)> = union
                    .into_iter()
                    .map(|(k, mut cs)| {
                        cs.sort_unstable();
                        let n = cs.len();
                        cs.dedup();
                        assert_eq!(
                            cs.len(),
                            n,
                            "{what}: sector {:?}: a channel is listed twice across mix and \
                             recontract",
                            sectors[k]
                        );
                        (sectors[k], cs)
                    })
                    .collect();
                assert_eq!(
                    got,
                    touched.to_vec(),
                    "{what}: mix ∪ recontract is not touched_ref"
                );
                let want_b: Vec<(Su2Sector, Vec<usize>)> = touched
                    .iter()
                    .filter_map(|(s, cs)| {
                        let descs = h.channel_descs(b, *s);
                        let bs: Vec<usize> = cs
                            .iter()
                            .copied()
                            .filter(|&c| touch(&descs[c], i) == Touch::B)
                            .collect();
                        (!bs.is_empty()).then_some((*s, bs))
                    })
                    .collect();
                assert_eq!(
                    rec.to_vec(),
                    want_b,
                    "{what}: recontract_ref is not the class-B channels of touched_ref"
                );
                let nb: usize = rec.iter().map(|(_, cs)| cs.len()).sum();
                class_b += nb;
                b_cells += usize::from(nb > 0);
            }
        }
        println!(
            "{name}: twin pairs left {pairs_left} / right {pairs_right}; class B {class_b} \
             channels in {b_cells} domain cells ({window_b} class-B instances outside the pair, \
             all in the §2 window); {widths} twin widths compared; unpaired {}; index {} B",
            idx.unpaired(),
            idx.bytes()
        );
        assert_eq!(
            idx.unpaired(),
            0,
            "{name}: under SymmetryAllowed every class-A channel has its twin (spec §1), yet \
             {} are re-contracted as unpaired",
            idx.unpaired()
        );
        assert!(
            pairs_left > 0 && pairs_right > 0,
            "{name}: twin pairs left {pairs_left} / right {pairs_right} — a chain's mix is vacuous"
        );
        assert!(
            class_b > 0,
            "{name}: no class-B channel in any domain cell — the re-contract path is vacuous"
        );
        assert!(widths > 0, "{name}: no twin width compared");
    }
}
