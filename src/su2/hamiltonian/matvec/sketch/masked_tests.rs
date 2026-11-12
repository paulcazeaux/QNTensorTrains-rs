// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::ops::Range;

use faer::prelude::ReborrowMut;
use rand::RngExt;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::compact_band_table;
use crate::coeff::CoeffTensor;
use crate::hamiltonian::csx::CsxBlock;
use crate::integrals::Integrals;
use crate::scalar::{C64, Scalar};
use crate::su2::core::{Family, Su2Core};
use crate::su2::coreops::core_times_adjoint_core;
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::channel::Su2TouchIndex;
use crate::su2::hamiltonian::channel::tests::{dense_fixture, fixtures};
use crate::su2::hamiltonian::matvec::{PAR_MIN_TASKS, su2_onesite_workspace};
use crate::su2::hamiltonian::{Su2SparseHamiltonian, Su2TermScreen};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::su2::vector::Su2TtVector;
use crate::test_util::{randomized_su2_state, seeded};

type Tab = Su2SectorGrid<Vec<Range<usize>>>;
type Keep = Vec<(Su2Sector, Vec<usize>)>;

const TOL: f64 = 1e-14;

const POISON: usize = usize::MAX / 2;

fn molecular() -> Vec<(String, Integrals<f64>, Su2Space, u64)> {
    fixtures()
}

fn phase<T: Scalar>(p: usize) -> f64 {
    if T::IS_COMPLEX {
        0.3 + 0.7 * p as f64
    } else {
        0.0
    }
}

fn lift<T: Scalar>(ints: &Integrals<f64>) -> (CoeffTensor<T>, CoeffTensor<T>) {
    let d = ints.norb;
    let z = |a: f64, ph: f64| T::from_complex_f64(a * ph.cos(), a * ph.sin());
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for p in 0..d {
        for q in 0..d {
            let a = ints.t.get(&[p, q]);
            if a != 0.0 {
                t.set(&[p, q], z(a, phase::<T>(p) - phase::<T>(q)));
            }
        }
    }
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for p in 0..d {
        for q in 0..d {
            for r in 0..d {
                for s in 0..d {
                    let a = ints.v.get(&[p, q, r, s]);
                    if a != 0.0 {
                        let ph = phase::<T>(p) - phase::<T>(q) + phase::<T>(r) - phase::<T>(s);
                        v.set(&[p, q, r, s], z(a, ph));
                    }
                }
            }
        }
    }
    (t, v)
}

fn refillable_t<T: Scalar>(ints: &Integrals<f64>, space: Su2Space) -> Su2SparseHamiltonian<T> {
    let (t, v) = lift::<T>(ints);
    Su2SparseHamiltonian::new_refillable(
        space,
        &t,
        &v,
        None,
        Su2TermScreen::SymmetryAllowed(space.pg),
    )
}

struct Setup<T: Scalar> {
    h: Su2SparseHamiltonian<T>,
    x: Su2TtVector<T>,
    fl: Vec<Su2Sketch<T>>,
    fr: Vec<Su2Sketch<T>>,
}

fn setup<T: Scalar>(ints: &Integrals<f64>, space: Su2Space, seed: u64) -> Setup<T> {
    let h = refillable_t::<T>(ints, space);
    let salt = if T::IS_COMPLEX { 0xC64 } else { 0 };
    let x: Su2TtVector<T> = randomized_su2_state(space, 1, 3, &mut seeded(seed ^ salt));
    let fl = left_to_right_sketching(&h, &x);
    let fr = right_to_left_sketching(&h, &x);
    Setup { h, x, fl, fr }
}

fn scalar_name<T: Scalar>() -> &'static str {
    if T::IS_COMPLEX { "C64" } else { "f64" }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Axis {
    Rows,
    Cols,
}

fn kept(keep: &[(Su2Sector, Vec<usize>)], s: Su2Sector) -> &[usize] {
    keep.iter()
        .find(|(t, _)| *t == s)
        .map_or(&[][..], |(_, v)| v.as_slice())
}

fn full_keep<T: Scalar>(h: &Su2SparseHamiltonian<T>, b: usize, full: &Tab) -> Keep {
    h.space
        .bond_sectors(b)
        .into_iter()
        .filter_map(|s| {
            let n = full[s].len();
            if n > 0 {
                assert_eq!(
                    n,
                    h.channel_descs(b, s).len(),
                    "bond {b} sector {s:?}: one band per channel"
                );
            }
            (n > 0).then(|| (s, (0..n).collect()))
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Empty,
    Strict,
    Full,
}

fn classify(space: &Su2Space, b: usize, full: &Tab, keep: &[(Su2Sector, Vec<usize>)]) -> Class {
    let total: usize = space.bond_sectors(b).iter().map(|&s| full[s].len()).sum();
    let n: usize = keep.iter().map(|(_, v)| v.len()).sum();
    if n == 0 {
        Class::Empty
    } else if n == total {
        Class::Full
    } else {
        Class::Strict
    }
}

fn random_strict(
    space: &Su2Space,
    b: usize,
    full: &Tab,
    rng: &mut Xoshiro256PlusPlus,
) -> Option<Keep> {
    let total: usize = space.bond_sectors(b).iter().map(|&s| full[s].len()).sum();
    if total < 2 {
        return None;
    }
    loop {
        let mut keep = Keep::new();
        for s in space.bond_sectors(b) {
            let cs: Vec<usize> = (0..full[s].len())
                .filter(|_| rng.random_range(0..2u32) == 1)
                .collect();
            if !cs.is_empty() {
                keep.push((s, cs));
            }
        }
        if classify(space, b, full, &keep) == Class::Strict {
            return Some(keep);
        }
    }
}

fn check_layout(
    what: &str,
    space: &Su2Space,
    b: usize,
    full: &Tab,
    keep: &[(Su2Sector, Vec<usize>)],
    ranks: &Su2RankGrid,
    tab: &Tab,
) {
    let order = space.bond_sectors(b);
    let mut last: Option<usize> = None;
    for (s, cs) in keep {
        let pos = order
            .iter()
            .position(|t| t == s)
            .unwrap_or_else(|| panic!("{what}: keep sector {s:?} is not a sector of bond {b}"));
        assert!(
            last.is_none_or(|l| l < pos),
            "{what}: keep sectors out of bond order"
        );
        last = Some(pos);
        assert!(
            cs.windows(2).all(|w| w[0] < w[1]),
            "{what}: sector {s:?}: channels not strictly ascending"
        );
        assert!(
            cs.last().is_none_or(|&c| c < full[*s].len()),
            "{what}: sector {s:?}: channel out of range"
        );
    }
    let mut total = 0usize;
    for &s in &order {
        assert_eq!(
            tab[s].len(),
            full[s].len(),
            "{what}: sector {s:?}: the compact table is not indexed by channel"
        );
        let mut off = 0usize;
        for &c in kept(keep, s) {
            let w = full[s][c].len();
            assert_eq!(
                tab[s][c],
                off..off + w,
                "{what}: sector {s:?} channel {c}: kept band not packed in channel order"
            );
            off += w;
        }
        assert_eq!(ranks[s], off, "{what}: sector {s:?}: compact rank");
        total += off;
    }
    assert_eq!(
        ranks.total(),
        total,
        "{what}: compact rank at a sector outside bond {b}"
    );
}

fn poisoned(
    space: &Su2Space,
    b: usize,
    full: &Tab,
    keep: &[(Su2Sector, Vec<usize>)],
    tab: &Tab,
) -> (Tab, usize) {
    let mut p = tab.clone();
    let mut n = 0usize;
    for s in space.bond_sectors(b) {
        let ks = kept(keep, s);
        for (c, (slot, rg)) in p[s].iter_mut().zip(&full[s]).enumerate() {
            if ks.binary_search(&c).is_err() {
                *slot = POISON..POISON + rg.len();
                n += 1;
            }
        }
    }
    (p, n)
}

fn bits<T: Scalar>(z: T) -> (u64, u64) {
    (z.re_s().to_bits(), z.im_s().to_bits())
}

fn sentinel<T: Scalar>(o: usize, r: usize, c: usize) -> T {
    T::from_complex_f64(
        -1.0e3 - o as f64 - 1.0e-2 * r as f64 - 1.0e-5 * c as f64,
        5.0e2 + 1.0e-3 * r as f64 + 1.0e-6 * c as f64,
    )
}

fn prefilled<T: Scalar>(like: &Su2Sketch<T>) -> Su2Sketch<T> {
    let mut f = like.clone();
    for (o, s) in like.sectors().into_iter().enumerate() {
        if let Some(mut b) = f.block_mut(s) {
            for c in 0..b.ncols() {
                for r in 0..b.nrows() {
                    *b.rb_mut().get_mut(r, c) = sentinel(o, r, c);
                }
            }
        }
    }
    f
}

fn assert_sketch_bits<T: Scalar>(a: &Su2Sketch<T>, b: &Su2Sketch<T>, what: &str) {
    assert_eq!(a.site(), b.site(), "{what}: bond");
    for s in a.sectors() {
        match (a.block(s), b.block(s)) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(
                    (x.nrows(), x.ncols()),
                    (y.nrows(), y.ncols()),
                    "{what}: sector {s:?}: shape"
                );
                for c in 0..x.ncols() {
                    for r in 0..x.nrows() {
                        assert_eq!(
                            bits(x[(r, c)]),
                            bits(y[(r, c)]),
                            "{what}: sector {s:?} entry ({r}, {c})"
                        );
                    }
                }
            }
            _ => panic!("{what}: sector {s:?}: block presence differs"),
        }
    }
}

fn assert_core_bits<T: Scalar>(a: &Su2Core<T>, b: &Su2Core<T>, what: &str) {
    assert_eq!(a.row_ranks, b.row_ranks, "{what}: row ranks");
    assert_eq!(a.col_ranks, b.col_ranks, "{what}: col ranks");
    for f in Family::ALL {
        let ta: Vec<Su2Sector> = a.transitions(f).collect();
        let tb: Vec<Su2Sector> = b.transitions(f).collect();
        assert_eq!(ta, tb, "{what}: {f:?} transitions");
        for s in ta {
            let (x, y) = (a.block(f, s).unwrap(), b.block(f, s).unwrap());
            assert_eq!(
                (x.nrows(), x.ncols()),
                (y.nrows(), y.ncols()),
                "{what}: ({f:?}, {s:?}) shape"
            );
            for c in 0..x.ncols() {
                for r in 0..x.nrows() {
                    assert_eq!(
                        bits(x[(r, c)]),
                        bits(y[(r, c)]),
                        "{what}: ({f:?}, {s:?}) entry ({r}, {c})"
                    );
                }
            }
        }
    }
}

fn sketch_entries<T: Scalar>(a: &Su2Sketch<T>) -> usize {
    a.sectors()
        .into_iter()
        .filter_map(|s| a.block(s))
        .map(|b| b.nrows() * b.ncols())
        .sum()
}

fn sketch_rel<T: Scalar>(a: &Su2Sketch<T>, b: &Su2Sketch<T>) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for s in b.sectors() {
        if let (Some(x), Some(y)) = (a.block(s), b.block(s)) {
            for c in 0..y.ncols() {
                for r in 0..y.nrows() {
                    num = num.max((x[(r, c)] - y[(r, c)]).abs_s());
                    den = den.max(y[(r, c)].abs_s());
                }
            }
        }
    }
    num / den.max(f64::MIN_POSITIVE)
}

#[derive(Clone, Copy, Debug, Default)]
struct EnvCheck {
    kept_max: f64,
    kept_entries: usize,
    kept_nonbitwise: usize,
    kept_content: f64,
    untouched_entries: usize,
    scale: f64,
}

fn check_env<T: Scalar>(
    what: &str,
    got: &Su2Sketch<T>,
    want: &Su2Sketch<T>,
    bands: &Tab,
    keep: &[(Su2Sector, Vec<usize>)],
    axis: Axis,
) -> EnvCheck {
    assert_eq!(got.site(), want.site(), "{what}: bond");
    let mut out = EnvCheck::default();
    for (o, s) in want.sectors().into_iter().enumerate() {
        let (g, w) = match (got.block(s), want.block(s)) {
            (Some(g), Some(w)) => (g, w),
            (None, None) => continue,
            _ => panic!("{what}: sector {s:?}: block presence differs"),
        };
        assert_eq!(
            (g.nrows(), g.ncols()),
            (w.nrows(), w.ncols()),
            "{what}: sector {s:?}: the destination's shape changed"
        );
        let len = match axis {
            Axis::Rows => w.nrows(),
            Axis::Cols => w.ncols(),
        };
        let rgs = &bands[s];
        assert_eq!(
            rgs.last().map_or(0, |r| r.end),
            len,
            "{what}: sector {s:?}: the bands do not tile the {axis:?} axis"
        );
        let mut in_kept = vec![false; len];
        for &c in kept(keep, s) {
            for u in rgs[c].clone() {
                in_kept[u] = true;
            }
        }
        for c in 0..w.ncols() {
            for r in 0..w.nrows() {
                let u = match axis {
                    Axis::Rows => r,
                    Axis::Cols => c,
                };
                let (gv, wv) = (g[(r, c)], w[(r, c)]);
                out.scale = out.scale.max(wv.abs_s());
                if in_kept[u] {
                    out.kept_max = out.kept_max.max((gv - wv).abs_s());
                    out.kept_entries += 1;
                    out.kept_nonbitwise += usize::from(bits(gv) != bits(wv));
                    out.kept_content = out.kept_content.max(wv.abs_s());
                } else {
                    assert_eq!(
                        bits(gv),
                        bits(sentinel::<T>(o, r, c)),
                        "{what}: sector {s:?} entry ({r}, {c}) lies outside every kept band but \
                         was written"
                    );
                    out.untouched_entries += 1;
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, Default)]
struct YCheck {
    max: f64,
    entries: usize,
    nonbitwise: usize,
    scale: f64,
}

fn check_y_bands<T: Scalar>(
    what: &str,
    y_full: &Su2Core<T>,
    y_c: &Su2Core<T>,
    full: &Tab,
    tab: &Tab,
    keep: &[(Su2Sector, Vec<usize>)],
    axis: Axis,
) -> YCheck {
    let g = y_full.space.pg.irrep_of(y_full.k);
    let mut out = YCheck::default();
    for f in Family::ALL {
        for n1 in y_full.transitions(f).collect::<Vec<_>>() {
            let s = match axis {
                Axis::Rows => n1,
                Axis::Cols => f.apply(n1, g).expect("an admissible transition"),
            };
            let ks = kept(keep, s);
            if ks.is_empty() {
                continue;
            }
            let a = y_full.block(f, n1).expect("full block");
            let b = y_c
                .block(f, n1)
                .unwrap_or_else(|| panic!("{what}: compact y lacks block ({f:?}, {n1:?})"));
            for &c in ks {
                let (fr, cr) = (full[s][c].clone(), tab[s][c].clone());
                assert_eq!(fr.len(), cr.len(), "{what}: band ({s:?}, {c}) width");
                let pairs: Vec<((usize, usize), (usize, usize))> = match axis {
                    Axis::Rows => (0..a.ncols())
                        .flat_map(|col| {
                            fr.clone()
                                .zip(cr.clone())
                                .map(move |(u, v)| ((u, col), (v, col)))
                        })
                        .collect(),
                    Axis::Cols => fr
                        .clone()
                        .zip(cr.clone())
                        .flat_map(|(u, v)| (0..a.nrows()).map(move |r| ((r, u), (r, v))))
                        .collect(),
                };
                for (pa, pb) in pairs {
                    let (av, bv) = (a[pa], b[pb]);
                    out.max = out.max.max((av - bv).abs_s());
                    out.scale = out.scale.max(av.abs_s());
                    out.entries += 1;
                    out.nonbitwise += usize::from(bits(av) != bits(bv));
                }
            }
        }
    }
    out
}

fn branch<T: Scalar>(csr: &CsxBlock<T>, csc: &CsxBlock<T>) -> usize {
    let (nnz, nrow, ncol) = (csr.nnz(), csr.span_len(), csc.span_len());
    if ncol < nnz.min(nrow) {
        0
    } else if nrow < nnz.min(ncol) {
        1
    } else {
        2
    }
}

fn mixed_cells<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    xk: &Su2Core<T>,
    keep: &[(Su2Sector, Vec<usize>)],
    axis: Axis,
) -> [usize; 3] {
    let g = h.space.pg.irrep_of(xk.k);
    let nest = &h.nests[xk.k];
    let mut out = [0usize; 3];
    for f in Family::ALL {
        for n1 in xk.space.bond_sectors(xk.k) {
            let s = match axis {
                Axis::Rows => n1,
                Axis::Cols => match f.apply(n1, g) {
                    Some(q) if xk.space.contains(xk.k + 1, q) => q,
                    _ => continue,
                },
            };
            let ks = kept(keep, s);
            if ks.is_empty() {
                continue;
            }
            for cell in nest.slot(f, n1) {
                let ids = match axis {
                    Axis::Rows => &cell.csr.ids,
                    Axis::Cols => &cell.csc.ids,
                };
                let n = ids
                    .iter()
                    .filter(|&&id| ks.binary_search(&(id as usize)).is_ok())
                    .count();
                if n > 0 && n < ids.len() {
                    out[branch(&cell.csr, &cell.csc)] += 1;
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy, Debug, Default)]
struct Tally {
    cases: usize,
    empty: usize,
    strict: usize,
    full: usize,
    env_worst: f64,
    y_worst: f64,
    kept_entries: usize,
    kept_content: f64,
    untouched_entries: usize,
    env_nonbitwise: usize,
    y_entries: usize,
    y_nonbitwise: usize,
    poisoned: usize,
    mixed: [usize; 3],
}

impl Tally {
    fn line(&self) -> String {
        format!(
            "{} cases (empty {} / strict {} / full {}); env max rel {:.2e}, {} kept entries ({} \
             not bitwise), {} untouched entries; y max rel {:.2e}, {} entries ({} not bitwise); \
             {} poisoned entries; mixed cells CSC/CSR/COO {:?}",
            self.cases,
            self.empty,
            self.strict,
            self.full,
            self.env_worst,
            self.kept_entries,
            self.env_nonbitwise,
            self.untouched_entries,
            self.y_worst,
            self.y_entries,
            self.y_nonbitwise,
            self.poisoned,
            self.mixed
        )
    }
}

struct CaseOut<T: Scalar> {
    class: Class,
    ranks: Su2RankGrid,
    y_full: Su2Core<T>,
    y_c: Su2Core<T>,
    want: Su2Sketch<T>,
}

fn right_case<T: Scalar>(
    su: &Setup<T>,
    j: usize,
    keep: &[(Su2Sector, Vec<usize>)],
    what: &str,
    t: &mut Tally,
) -> CaseOut<T> {
    let (h, space) = (&su.h, su.h.space);
    let xk = su.x.core(j);
    let fr_next = &su.fr[j];
    let ws = su2_onesite_workspace(h, xk);
    let class = classify(&space, j, &ws.i_tab, keep);

    let (ranks, tab) = compact_band_table(&space, &ws.i_tab, keep);
    check_layout(what, &space, j, &ws.i_tab, keep, &ranks, &tab);
    let (ptab, npois) = poisoned(&space, j, &ws.i_tab, keep, &tab);

    let mut y_full = Su2Core::zeros(space, j, &h.row_ranks(xk), &fr_next.col_ranks());
    h.h_sketch_mul_into(&mut y_full, xk, fr_next, |_| true, &ws.i_tab, &ws.j_tab);
    let mut y_c = Su2Core::zeros(space, j, &ranks, &fr_next.col_ranks());
    h.h_sketch_mul_masked_into(&mut y_c, xk, fr_next, keep, &ws.j_tab, &tab);
    let mut y_p = Su2Core::zeros(space, j, &ranks, &fr_next.col_ranks());
    h.h_sketch_mul_masked_into(&mut y_p, xk, fr_next, keep, &ws.j_tab, &ptab);
    assert_core_bits(
        &y_c,
        &y_p,
        &format!("{what}: compact y, poisoned vs clean table"),
    );
    let yc = check_y_bands(what, &y_full, &y_c, &ws.i_tab, &tab, keep, Axis::Rows);
    let y_rel = yc.max / yc.scale.max(f64::MIN_POSITIVE);
    assert!(
        y_rel <= TOL,
        "{what}: a kept row band of the masked product differs from the unmasked one by \
         {y_rel:.3e} (relative)"
    );

    let want = h.sketching_step_left_mixed(xk, xk, fr_next, &ws);
    let mut got = prefilled(&want);
    h.recontract_right_rows(xk, fr_next, keep, &ws, &mut got);
    let ec = check_env(what, &got, &want, &ws.i_tab, keep, Axis::Rows);
    let env_rel = ec.kept_max / ec.scale.max(f64::MIN_POSITIVE);
    assert!(
        env_rel <= TOL,
        "{what}: a kept row band of Fᴿ_{j} differs from the unmasked step by {env_rel:.3e} \
         (relative)"
    );

    absorb(t, class, &ec, env_rel, &yc, y_rel, npois);
    let m = mixed_cells(h, xk, keep, Axis::Rows);
    for (acc, add) in t.mixed.iter_mut().zip(m) {
        *acc += add;
    }
    CaseOut {
        class,
        ranks,
        y_full,
        y_c,
        want,
    }
}

fn left_case<T: Scalar>(
    su: &Setup<T>,
    b: usize,
    keep: &[(Su2Sector, Vec<usize>)],
    what: &str,
    t: &mut Tally,
) -> CaseOut<T> {
    let (h, space) = (&su.h, su.h.space);
    let xk = su.x.core(b - 1);
    let fl_prev = &su.fl[b - 1];
    let ws = su2_onesite_workspace(h, xk);
    let class = classify(&space, b, &ws.j_tab, keep);

    let (ranks, tab) = compact_band_table(&space, &ws.j_tab, keep);
    check_layout(what, &space, b, &ws.j_tab, keep, &ranks, &tab);
    let (ptab, npois) = poisoned(&space, b, &ws.j_tab, keep, &tab);

    let mut y_full = Su2Core::zeros(space, b - 1, &fl_prev.row_ranks(), &h.col_ranks(xk));
    h.sketch_h_mul_into(&mut y_full, fl_prev, xk, |_| true, &ws.i_tab, &ws.j_tab);
    let mut y_c = Su2Core::zeros(space, b - 1, &fl_prev.row_ranks(), &ranks);
    h.sketch_h_mul_masked_into(&mut y_c, fl_prev, xk, keep, &ws.i_tab, &tab);
    let mut y_p = Su2Core::zeros(space, b - 1, &fl_prev.row_ranks(), &ranks);
    h.sketch_h_mul_masked_into(&mut y_p, fl_prev, xk, keep, &ws.i_tab, &ptab);
    assert_core_bits(
        &y_c,
        &y_p,
        &format!("{what}: compact y, poisoned vs clean table"),
    );
    let yc = check_y_bands(what, &y_full, &y_c, &ws.j_tab, &tab, keep, Axis::Cols);
    let y_rel = yc.max / yc.scale.max(f64::MIN_POSITIVE);
    assert!(
        y_rel <= TOL,
        "{what}: a kept column band of the masked product differs from the unmasked one by \
         {y_rel:.3e} (relative)"
    );

    let want = h.sketching_step_right_mixed(xk, xk, fl_prev, &ws);
    let mut got = prefilled(&want);
    h.recontract_left_cols(xk, fl_prev, keep, &ws, &mut got);
    let ec = check_env(what, &got, &want, &ws.j_tab, keep, Axis::Cols);
    let env_rel = ec.kept_max / ec.scale.max(f64::MIN_POSITIVE);
    assert!(
        env_rel <= TOL,
        "{what}: a kept column band of Fᴸ_{b} differs from the unmasked step by {env_rel:.3e} \
         (relative)"
    );

    absorb(t, class, &ec, env_rel, &yc, y_rel, npois);
    let m = mixed_cells(h, xk, keep, Axis::Cols);
    for (acc, add) in t.mixed.iter_mut().zip(m) {
        *acc += add;
    }
    CaseOut {
        class,
        ranks,
        y_full,
        y_c,
        want,
    }
}

fn absorb(
    t: &mut Tally,
    class: Class,
    ec: &EnvCheck,
    env_rel: f64,
    yc: &YCheck,
    y_rel: f64,
    npois: usize,
) {
    t.cases += 1;
    match class {
        Class::Empty => t.empty += 1,
        Class::Strict => t.strict += 1,
        Class::Full => t.full += 1,
    }
    t.env_worst = t.env_worst.max(env_rel);
    t.y_worst = t.y_worst.max(y_rel);
    t.kept_entries += ec.kept_entries;
    t.kept_content = t.kept_content.max(ec.kept_content);
    t.untouched_entries += ec.untouched_entries;
    t.env_nonbitwise += ec.kept_nonbitwise;
    t.y_entries += yc.entries;
    t.y_nonbitwise += yc.nonbitwise;
    t.poisoned += npois;
}

fn full_mask_arm<T: Scalar>(name: &str, ints: &Integrals<f64>, space: Su2Space, seed: u64) {
    let su = setup::<T>(ints, space, seed);
    let d = space.d;
    let what0 = format!("{name} {}", scalar_name::<T>());
    let (mut tr, mut tl) = (Tally::default(), Tally::default());
    let mut theta_live = 0.0f64;
    for b in 1..d {
        let ws = su2_onesite_workspace(&su.h, su.x.core(b));
        let keep = full_keep(&su.h, b, &ws.i_tab);
        assert!(!keep.is_empty(), "{what0}: bond {b} has no row band");
        let what = format!("{what0}: full mask, Fᴿ at bond {b}");
        let out = right_case(&su, b, &keep, &what, &mut tr);
        assert_eq!(out.class, Class::Full, "{what}: the mask is not full");
        assert_eq!(
            out.ranks,
            su.h.row_ranks(su.x.core(b)),
            "{what}: a full mask's compact rows are not the H-extended rows"
        );
        assert_core_bits(&out.y_full, &out.y_c, &format!("{what}: y before the fold"));
        let plain = core_times_adjoint_core(&out.y_full, su.x.core(b));
        theta_live = theta_live.max(sketch_rel(&plain, &out.want));

        let ws = su2_onesite_workspace(&su.h, su.x.core(b - 1));
        let keep = full_keep(&su.h, b, &ws.j_tab);
        assert!(!keep.is_empty(), "{what0}: bond {b} has no column band");
        let what = format!("{what0}: full mask, Fᴸ at bond {b}");
        let out = left_case(&su, b, &keep, &what, &mut tl);
        assert_eq!(out.class, Class::Full, "{what}: the mask is not full");
        assert_eq!(
            out.ranks,
            su.h.col_ranks(su.x.core(b - 1)),
            "{what}: a full mask's compact columns are not the H-extended columns"
        );
        assert_core_bits(&out.y_full, &out.y_c, &format!("{what}: y before the fold"));
    }
    for (stack, t) in [("Fᴿ", &tr), ("Fᴸ", &tl)] {
        println!("{what0}: {stack}: {}", t.line());
        assert_eq!(
            t.untouched_entries, 0,
            "{what0}: {stack}: a full mask left entries"
        );
        assert!(
            t.kept_content > 0.0,
            "{what0}: {stack}: every compared entry is zero — test 1 is vacuous"
        );
    }
    println!("{what0}: θ² live — plain vs θ²-weighted right fold differ by {theta_live:.3e}");
    assert!(
        theta_live > 1e-3,
        "{what0}: the θ²-weighted and the plain right folds agree ({theta_live:.3e}) — the row \
         mask's commutation with θ² is not exercised on this fixture"
    );
}

#[test]
fn g1a_t1_full_mask_is_the_unmasked_step() {
    for (name, ints, space, seed) in molecular() {
        full_mask_arm::<f64>(&name, &ints, space, seed);
        full_mask_arm::<C64>(&name, &ints, space, seed);
    }
}

fn partial_mask_arm<T: Scalar>(name: &str, ints: &Integrals<f64>, space: Su2Space, seed: u64) {
    let su = setup::<T>(ints, space, seed);
    let d = space.d;
    let idx = Su2TouchIndex::new(&su.h);
    let what0 = format!("{name} {}", scalar_name::<T>());
    let (mut tr, mut tl) = (Tally::default(), Tally::default());
    for i in 0..d - 1 {
        for j in i + 2..d {
            let what = format!("{what0}: pair ({i}, {}), Fᴿ at bond {j}, T_{j}({i})", i + 1);
            right_case(&su, j, idx.touched_ref(j, i), &what, &mut tr);
        }
        for b in 1..=i {
            let what = format!("{what0}: pair ({i}, {}), Fᴸ at bond {b}, T_{b}({i})", i + 1);
            left_case(&su, b, idx.touched_ref(b, i), &what, &mut tl);
        }
    }
    for (stack, t) in [("Fᴿ", &tr), ("Fᴸ", &tl)] {
        println!("{what0}: touched sets, {stack}: {}", t.line());
        assert!(
            t.strict > 0,
            "{what0}: {stack}: no touched set is a strict nonempty subset of its bond — the \
             partial-mask clause is vacuous on this fixture"
        );
    }
    let mut rng = seeded(seed ^ 0x61A2);
    let (mut rr, mut rl) = (Tally::default(), Tally::default());
    for b in 1..d {
        let ws = su2_onesite_workspace(&su.h, su.x.core(b));
        if let Some(keep) = random_strict(&space, b, &ws.i_tab, &mut rng) {
            let what = format!("{what0}: random strict mask, Fᴿ at bond {b}");
            right_case(&su, b, &keep, &what, &mut rr);
        }
        let ws = su2_onesite_workspace(&su.h, su.x.core(b - 1));
        if let Some(keep) = random_strict(&space, b, &ws.j_tab, &mut rng) {
            let what = format!("{what0}: random strict mask, Fᴸ at bond {b}");
            left_case(&su, b, &keep, &what, &mut rl);
        }
    }
    for (stack, t) in [("Fᴿ", &rr), ("Fᴸ", &rl)] {
        println!("{what0}: random masks, {stack}: {}", t.line());
        assert!(t.strict > 0, "{what0}: {stack}: no random strict mask");
    }
    for (stack, a, b) in [("Fᴿ", &tr, &rr), ("Fᴸ", &tl, &rl)] {
        assert!(
            a.untouched_entries + b.untouched_entries > 0,
            "{what0}: {stack}: no entry outside a kept band — the sentinel clause is vacuous"
        );
        assert!(
            a.kept_content.max(b.kept_content) > 0.0,
            "{what0}: {stack}: every kept entry is zero — the band comparison is vacuous"
        );
        assert!(
            a.poisoned + b.poisoned > 0,
            "{what0}: {stack}: no untouched table entry was poisoned — the K1 clause is vacuous"
        );
        let mixed: usize = a.mixed.iter().chain(&b.mixed).sum();
        assert!(
            mixed > 0,
            "{what0}: {stack}: no cell has partly kept outputs — the per-entry keep test is never \
             exercised"
        );
    }
}

#[test]
fn g1a_t2_partial_mask_is_the_unmasked_step_on_kept_bands_and_leaves_the_rest() {
    for (name, ints, space, seed) in molecular() {
        partial_mask_arm::<f64>(&name, &ints, space, seed);
        partial_mask_arm::<C64>(&name, &ints, space, seed);
    }
}

#[test]
fn g1a_t6_partial_mask_dense_d7_second_slot_channels() {
    let (name, ints, space, seed) = dense_fixture();
    partial_mask_arm::<f64>(&name, &ints, space, seed);
    partial_mask_arm::<C64>(&name, &ints, space, seed);
}

fn empty_mask_arm<T: Scalar>(name: &str, ints: &Integrals<f64>, space: Su2Space, seed: u64) {
    let su = setup::<T>(ints, space, seed);
    let (h, d) = (&su.h, space.d);
    let what0 = format!("{name} {}", scalar_name::<T>());
    let mut compared = 0usize;
    for b in 1..d {
        let xk = su.x.core(b);
        let ws = su2_onesite_workspace(h, xk);
        let (ranks, _) = compact_band_table(&space, &ws.i_tab, &[]);
        assert_eq!(
            ranks.total(),
            0,
            "{what0}: bond {b}: an empty mask has compact rows"
        );
        let before = prefilled(&su.fr[b - 1]);
        let mut got = before.clone();
        h.recontract_right_rows(xk, &su.fr[b], &[], &ws, &mut got);
        assert_sketch_bits(
            &before,
            &got,
            &format!("{what0}: empty mask, Fᴿ at bond {b}"),
        );
        compared += sketch_entries(&before);

        let xk = su.x.core(b - 1);
        let ws = su2_onesite_workspace(h, xk);
        let (ranks, _) = compact_band_table(&space, &ws.j_tab, &[]);
        assert_eq!(
            ranks.total(),
            0,
            "{what0}: bond {b}: an empty mask has compact columns"
        );
        let before = prefilled(&su.fl[b]);
        let mut got = before.clone();
        h.recontract_left_cols(xk, &su.fl[b - 1], &[], &ws, &mut got);
        assert_sketch_bits(
            &before,
            &got,
            &format!("{what0}: empty mask, Fᴸ at bond {b}"),
        );
        compared += sketch_entries(&before);
    }
    assert!(compared > 0, "{what0}: no nonempty destination compared");
}

#[test]
fn g1a_t3_empty_mask_is_a_no_op() {
    for (name, ints, space, seed) in molecular() {
        empty_mask_arm::<f64>(&name, &ints, space, seed);
        empty_mask_arm::<C64>(&name, &ints, space, seed);
    }
}

fn kept_tasks<T: Scalar>(y_c: &Su2Core<T>, keep: &[(Su2Sector, Vec<usize>)], axis: Axis) -> usize {
    let g = y_c.space.pg.irrep_of(y_c.k);
    Family::ALL
        .into_iter()
        .map(|f| {
            y_c.transitions(f)
                .filter(|&n1| {
                    let s = match axis {
                        Axis::Rows => n1,
                        Axis::Cols => f.apply(n1, g).expect("an admissible transition"),
                    };
                    !kept(keep, s).is_empty()
                        && y_c.block_dims(f, n1).is_some_and(|(r, c)| r > 0 && c > 0)
                })
                .count()
        })
        .sum()
}

fn core_max_dim<T: Scalar>(c: &Su2Core<T>) -> usize {
    Family::ALL
        .into_iter()
        .flat_map(|f| c.transitions(f).filter_map(move |s| c.block_dims(f, s)))
        .map(|(r, k)| r.max(k))
        .max()
        .unwrap_or(0)
}

fn right_run<T: Scalar>(
    su: &Setup<T>,
    j: usize,
    keep: &[(Su2Sector, Vec<usize>)],
) -> (Su2Core<T>, Su2Sketch<T>) {
    let (h, space) = (&su.h, su.h.space);
    let xk = su.x.core(j);
    let ws = su2_onesite_workspace(h, xk);
    let (ranks, tab) = compact_band_table(&space, &ws.i_tab, keep);
    let mut y = Su2Core::zeros(space, j, &ranks, &su.fr[j].col_ranks());
    h.h_sketch_mul_masked_into(&mut y, xk, &su.fr[j], keep, &ws.j_tab, &tab);
    let mut env = prefilled(&su.fr[j - 1]);
    h.recontract_right_rows(xk, &su.fr[j], keep, &ws, &mut env);
    (y, env)
}

fn left_run<T: Scalar>(
    su: &Setup<T>,
    b: usize,
    keep: &[(Su2Sector, Vec<usize>)],
) -> (Su2Core<T>, Su2Sketch<T>) {
    let (h, space) = (&su.h, su.h.space);
    let xk = su.x.core(b - 1);
    let ws = su2_onesite_workspace(h, xk);
    let (ranks, tab) = compact_band_table(&space, &ws.j_tab, keep);
    let mut y = Su2Core::zeros(space, b - 1, &su.fl[b - 1].row_ranks(), &ranks);
    h.sketch_h_mul_masked_into(&mut y, &su.fl[b - 1], xk, keep, &ws.i_tab, &tab);
    let mut env = prefilled(&su.fl[b]);
    h.recontract_left_cols(xk, &su.fl[b - 1], keep, &ws, &mut env);
    (y, env)
}

fn serial_arm<T: Scalar>(
    name: &str,
    ints: &Integrals<f64>,
    space: Su2Space,
    seed: u64,
    pools: &[rayon::ThreadPool],
) -> ([usize; 2], usize, usize) {
    let su = setup::<T>(ints, space, seed);
    let d = space.d;
    let idx = Su2TouchIndex::new(&su.h);
    let what0 = format!("{name} {}", scalar_name::<T>());
    let mut tasks = [0usize; 2];
    let (mut r_max, mut h_max) = (0usize, 0usize);
    for k in 0..d {
        r_max = r_max.max(core_max_dim(su.x.core(k)));
    }
    for b in 1..d {
        let ws_r = su2_onesite_workspace(&su.h, su.x.core(b));
        let mut right: Vec<Keep> = vec![full_keep(&su.h, b, &ws_r.i_tab)];
        right.extend((0..b.saturating_sub(1)).map(|i| idx.touched(b, i)));
        let ws_l = su2_onesite_workspace(&su.h, su.x.core(b - 1));
        let mut left: Vec<Keep> = vec![full_keep(&su.h, b, &ws_l.j_tab)];
        left.extend((b..d - 1).map(|i| idx.touched(b, i)));
        for keep in &right {
            let runs: Vec<_> = pools
                .iter()
                .map(|p| p.install(|| right_run(&su, b, keep)))
                .collect();
            let (y0, e0) = &runs[0];
            tasks[0] = tasks[0].max(kept_tasks(y0, keep, Axis::Rows));
            h_max = h_max
                .max(core_max_dim(y0))
                .max(su.h.row_ranks(su.x.core(b)).total());
            for ((y, e), p) in runs.iter().zip(pools).skip(1) {
                let what = format!(
                    "{what0}: Fᴿ at bond {b}, 1 vs {} threads",
                    p.current_num_threads()
                );
                assert_core_bits(y0, y, &format!("{what}: compact y"));
                assert_sketch_bits(e0, e, &format!("{what}: environment"));
            }
        }
        for keep in &left {
            let runs: Vec<_> = pools
                .iter()
                .map(|p| p.install(|| left_run(&su, b, keep)))
                .collect();
            let (y0, e0) = &runs[0];
            tasks[1] = tasks[1].max(kept_tasks(y0, keep, Axis::Cols));
            h_max = h_max
                .max(core_max_dim(y0))
                .max(su.h.col_ranks(su.x.core(b - 1)).total());
            for ((y, e), p) in runs.iter().zip(pools).skip(1) {
                let what = format!(
                    "{what0}: Fᴸ at bond {b}, 1 vs {} threads",
                    p.current_num_threads()
                );
                assert_core_bits(y0, y, &format!("{what}: compact y"));
                assert_sketch_bits(e0, e, &format!("{what}: environment"));
            }
        }
    }
    println!(
        "{what0}: max kept tasks Fᴿ {} / Fᴸ {} (PAR_MIN_TASKS = {PAR_MIN_TASKS}); r_max {r_max}, \
         H-extended max {h_max}",
        tasks[0], tasks[1]
    );
    (tasks, r_max, h_max)
}

#[test]
fn g1a_t4_masked_steps_match_serial_bitwise() {
    let pools: Vec<rayon::ThreadPool> = [1usize, 8]
        .into_iter()
        .map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        })
        .collect();
    let all: Vec<_> = molecular().into_iter().chain([dense_fixture()]).collect();
    let mut best = [[0usize; 2]; 2];
    for (name, ints, space, seed) in &all {
        for (si, (tasks, r_max, h_max)) in [
            serial_arm::<f64>(name, ints, *space, *seed, &pools),
            serial_arm::<C64>(name, ints, *space, *seed, &pools),
        ]
        .into_iter()
        .enumerate()
        {
            assert!(
                r_max.pow(3) < 1 << 21 && r_max * r_max * h_max < 1 << 21,
                "{name}: r_max {r_max}, h_max {h_max}: a GEMM may cross the 2^21 gate \
                 — the bitwise claim no longer follows; shrink the fixture or relax to machine \
                 precision"
            );
            for (b, t) in best[si].iter_mut().zip(tasks) {
                *b = (*b).max(t);
            }
        }
    }
    for (si, scalar) in ["f64", "C64"].into_iter().enumerate() {
        for (stack, t) in ["Fᴿ", "Fᴸ"].into_iter().zip(best[si]) {
            assert!(
                t >= PAR_MIN_TASKS,
                "{scalar} {stack}: no call reached {PAR_MIN_TASKS} kept tasks (max {t}) — the \
                 threaded arm never runs and test 4 is vacuous"
            );
        }
    }
}
