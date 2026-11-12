// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use faer::prelude::ReborrowMut;

use super::{
    Su2GaugeCache, Su2GaugeFault, Su2GaugeMode, gauge_after_pair_rotation,
    gauge_after_pair_rotation_faulty,
};
use crate::coeff::CoeffTensor;
use crate::integrals::{Integrals, IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use crate::ordering::{irrep_blocked_order, permute_integrals};
use crate::qn::Pooled;
use crate::rotation::{
    compose_rotations, givens, identity_matrix, orthogonality_defect, rotate_integrals,
};
use crate::scalar::{C64, Scalar};
use crate::su2::core::{Family, Su2Core};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::channel::tests::{dense_fixture, fixtures};
use crate::su2::hamiltonian::channel::{Su2TouchIndex, Touch, touch};
use crate::su2::hamiltonian::matvec::su2_onesite_workspace;
use crate::su2::hamiltonian::{
    Su2PatternMask, Su2SparseHamiltonian, Su2TermScreen, Ts, rayleigh_quotient,
};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::su2::twosite::{apply_mode_rotation, contract, factor_svd};
use crate::su2::vector::Su2TtVector;
use crate::test_util::{randomized_su2_state, seeded};

type Fixture = (String, Integrals<f64>, Su2Space, u64);
type Keep = Vec<(Su2Sector, Vec<usize>)>;

const TOL: f64 = 1e-12;

const MOVE: f64 = 1e-6;

const INVARIANT: f64 = 1e-10;

fn all_fixtures() -> Vec<Fixture> {
    fixtures().into_iter().chain([dense_fixture()]).collect()
}

fn typed_fixtures() -> Vec<Fixture> {
    [(0usize, 0x6000_0711u64), (2, 0x6000_0712)]
        .into_iter()
        .map(|(two_s, seed)| {
            let o = IntegralsOptions {
                point_group: PointGroupPolicy::Resolve(None),
                screen: true,
                ..IntegralsOptions::default()
            };
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h2o_sto3g");
            let load = load_fixture_dir::<f64>(&dir, &o).expect("load h2o_sto3g typed");
            let (pg0, d) = (load.pg.pg, load.ints.norb);
            assert!(!pg0.is_trivial(), "h2o_sto3g must resolve C2v");
            let perm = irrep_blocked_order(pg0, d);
            let ints = permute_integrals(&load.ints, &perm);
            let pgd = ints
                .point_group(Some(load.pg.kind))
                .expect("permuted dump resolves");
            for (a, &old) in perm.iter().enumerate() {
                assert_eq!(pgd.pg.irrep_of(a), pg0.irrep_of(old), "irrep′({a})");
            }
            let same = (0..d - 1)
                .filter(|&k| pgd.pg.irrep_of(k) == pgd.pg.irrep_of(k + 1))
                .count();
            assert!(same >= 3, "vacuous: {same} same-irrep bonds");
            let space = Su2Space::with_point_group(d, ints.nelec, two_s, pgd.pg, pgd.target);
            (format!("h2o C2v blocked (7,10,{two_s})"), ints, space, seed)
        })
        .collect()
}

fn phase<T: Scalar>(p: usize) -> f64 {
    if T::IS_COMPLEX {
        0.3 + 0.7 * p as f64
    } else {
        0.0
    }
}

fn lift<T: Scalar>(ints: &Integrals<f64>) -> Integrals<T> {
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
    Integrals {
        t,
        v,
        e0: T::from_f64(ints.e0),
        norb: d,
        nelec: ints.nelec,
        ms2: ints.ms2,
        orbsym: ints.orbsym.clone(),
        isym: ints.isym,
    }
}

fn refillable_t<T: Scalar>(ints: &Integrals<T>, space: Su2Space) -> Su2SparseHamiltonian<T> {
    Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::SymmetryAllowed(space.pg),
    )
}

fn state<T: Scalar>(space: Su2Space, seed: u64) -> Su2TtVector<T> {
    let salt = if T::IS_COMPLEX { 0xC64 } else { 0 };
    randomized_su2_state(space, 1, 3, &mut seeded(seed ^ salt))
}

fn scalar_name<T: Scalar>() -> &'static str {
    if T::IS_COMPLEX { "C64" } else { "f64" }
}

fn rotate_pair_state<T: Scalar>(x: &Su2TtVector<T>, i: usize, theta: f64) -> Su2TtVector<T> {
    let tc = apply_mode_rotation(&contract(x.core(i), x.core(i + 1)), theta);
    let (a, sv, mut b1) = factor_svd(&tc, 0.0, Pooled::NONE, false);
    b1.fold_sketch_left_mut(&sv);
    let mut y = x.clone();
    y.set_cores(a, b1);
    y
}

fn integrals_delta<T: Scalar>(a: &Integrals<T>, b: &Integrals<T>) -> f64 {
    a.t.data
        .iter()
        .zip(&b.t.data)
        .chain(a.v.data.iter().zip(&b.v.data))
        .fold(0.0f64, |m, (p, q)| m.max((*p - *q).abs_s()))
}

fn bond_ranks<T: Scalar>(x: &Su2TtVector<T>, b: usize) -> Su2RankGrid {
    x.core(b).row_ranks.clone()
}

fn nonempty(keep: &[(Su2Sector, Vec<usize>)]) -> bool {
    keep.iter().any(|(_, cs)| !cs.is_empty())
}

fn left_bonds(i: usize) -> std::ops::RangeInclusive<usize> {
    1..=i
}

fn right_bonds(i: usize, d: usize) -> Range<usize> {
    i + 2..d
}

fn bits<T: Scalar>(z: T) -> (u64, u64) {
    (z.re_s().to_bits(), z.im_s().to_bits())
}

fn sketch_rel<T: Scalar>(a: &Su2Sketch<T>, b: &Su2Sketch<T>, what: &str) -> f64 {
    assert_eq!(a.site(), b.site(), "{what}: bond");
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for s in b.sectors() {
        match (a.block(s), b.block(s)) {
            (None, None) => {}
            (Some(x), Some(y)) => {
                assert_eq!(
                    (x.nrows(), x.ncols()),
                    (y.nrows(), y.ncols()),
                    "{what}: sector {s:?}: shape"
                );
                for c in 0..y.ncols() {
                    for r in 0..y.nrows() {
                        num = num.max((x[(r, c)] - y[(r, c)]).abs_s());
                        den = den.max(y[(r, c)].abs_s());
                    }
                }
            }
            _ => panic!("{what}: sector {s:?}: block presence differs"),
        }
    }
    num / den.max(f64::MIN_POSITIVE)
}

fn sketch_nonbitwise<T: Scalar>(a: &Su2Sketch<T>, b: &Su2Sketch<T>, what: &str) -> usize {
    assert_eq!(a.site(), b.site(), "{what}: bond");
    let mut n = 0;
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
                        n += usize::from(bits(x[(r, c)]) != bits(y[(r, c)]));
                    }
                }
            }
            _ => panic!("{what}: sector {s:?}: block presence differs"),
        }
    }
    n
}

#[derive(Debug, Default)]
struct DomainDev {
    left: Vec<(usize, f64)>,
    right: Vec<(usize, f64)>,
}

impl DomainDev {
    fn max_left(&self) -> f64 {
        self.left.iter().fold(0.0f64, |m, &(_, r)| m.max(r))
    }
    fn max_right(&self) -> f64 {
        self.right.iter().fold(0.0f64, |m, &(_, r)| m.max(r))
    }
    fn max(&self) -> f64 {
        self.max_left().max(self.max_right())
    }
    fn first_over(&self, tol: f64) -> Option<(&'static str, usize, f64)> {
        self.left
            .iter()
            .map(|&(b, r)| ("Fᴸ", b, r))
            .chain(self.right.iter().map(|&(j, r)| ("Fᴿ", j, r)))
            .find(|&(_, _, r)| r > tol)
    }
}

fn domain_dev<T: Scalar>(
    fl: &[Su2Sketch<T>],
    fr: &[Su2Sketch<T>],
    ofl: &[Su2Sketch<T>],
    ofr: &[Su2Sketch<T>],
    i: usize,
    d: usize,
    what: &str,
) -> DomainDev {
    DomainDev {
        left: left_bonds(i)
            .map(|b| {
                assert_eq!(ofl[b].site(), b, "{what}: fl slot {b} is bond {b}");
                (
                    b,
                    sketch_rel(&fl[b], &ofl[b], &format!("{what}: Fᴸ bond {b}")),
                )
            })
            .collect(),
        right: right_bonds(i, d)
            .map(|j| {
                assert_eq!(
                    ofr[j - 1].site(),
                    j,
                    "{what}: fr slot {} is bond {j}",
                    j - 1
                );
                (
                    j,
                    sketch_rel(&fr[j - 1], &ofr[j - 1], &format!("{what}: Fᴿ bond {j}")),
                )
            })
            .collect(),
    }
}

fn assert_outside_domain_bitwise<T: Scalar>(
    fl: &[Su2Sketch<T>],
    fr: &[Su2Sketch<T>],
    fl0: &[Su2Sketch<T>],
    fr0: &[Su2Sketch<T>],
    i: usize,
    d: usize,
    what: &str,
) {
    assert_eq!((fl.len(), fr.len()), (d, d), "{what}: stack lengths");
    let left_out = std::iter::once(0).chain(i + 1..d);
    let right_out = (0..=i).chain(std::iter::once(d - 1));
    for (stack, slots, got, was) in [
        ("fl", left_out.collect::<Vec<_>>(), fl, fl0),
        ("fr", right_out.collect::<Vec<_>>(), fr, fr0),
    ] {
        for k in slots {
            let n = sketch_nonbitwise(&got[k], &was[k], &format!("{what}: {stack}[{k}]"));
            assert_eq!(
                n,
                0,
                "{what}: {stack}[{k}] (bond {}) is outside the gauge's write domain for the pair \
                 ({i}, {}) but {n} entries changed",
                if stack == "fl" { k } else { k + 1 },
                i + 1
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_untouched_bands_bitwise<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    idx: &Su2TouchIndex,
    i: usize,
    fl: &[Su2Sketch<T>],
    fr: &[Su2Sketch<T>],
    fl0: &[Su2Sketch<T>],
    fr0: &[Su2Sketch<T>],
    what: &str,
) -> usize {
    let d = x.space.d;
    let mut compared = 0;
    let mut one = |b: usize,
                   got: &Su2Sketch<T>,
                   was: &Su2Sketch<T>,
                   bands: Su2SectorGrid<Vec<Range<usize>>>,
                   rows: bool| {
        let touched = idx.touched_ref(b, i);
        for s in was.sectors() {
            let (Some(g), Some(w)) = (got.block(s), was.block(s)) else {
                continue;
            };
            let tset: &[usize] = touched
                .iter()
                .find(|(t, _)| *t == s)
                .map_or(&[][..], |(_, v)| v.as_slice());
            for (c, rg) in bands[s].iter().enumerate() {
                if tset.binary_search(&c).is_ok() {
                    continue;
                }
                for u in rg.clone() {
                    let other = if rows { w.ncols() } else { w.nrows() };
                    for o in 0..other {
                        let (r, cc) = if rows { (u, o) } else { (o, u) };
                        assert_eq!(
                            bits(g[(r, cc)]),
                            bits(w[(r, cc)]),
                            "{what}: {} bond {b} sector {s:?} UNTOUCHED channel {c} entry \
                             ({r}, {cc}) was rewritten — the chain must overwrite T_b(i) only",
                            if rows { "Fᴿ" } else { "Fᴸ" }
                        );
                        compared += 1;
                    }
                }
            }
        }
    };
    for b in left_bonds(i) {
        one(b, &fl[b], &fl0[b], h.col_ranges(x.core(b - 1)), false);
    }
    for j in right_bonds(i, d) {
        one(j, &fr[j - 1], &fr0[j - 1], h.row_ranges(x.core(j)), true);
    }
    compared
}

type Band = (Su2Sector, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Variant {
    Intact,
    Drop,
    SkipLeft,
    SkipRight,
}

fn without(keep: &[(Su2Sector, Vec<usize>)], gone: Option<Band>) -> Keep {
    keep.iter()
        .filter_map(|(s, cs)| {
            let cs: Vec<usize> = cs
                .iter()
                .copied()
                .filter(|&c| gone != Some((*s, c)))
                .collect();
            (!cs.is_empty()).then_some((*s, cs))
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn local_chain<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    i: usize,
    idx: &Su2TouchIndex,
    fl: &mut [Su2Sketch<T>],
    fr: &mut [Su2Sketch<T>],
    variant: Variant,
    gone: &dyn Fn(usize) -> Option<Band>,
) {
    let d = x.space.d;
    let keep_at = |b: usize| -> Keep {
        let t = idx.touched_ref(b, i);
        if variant == Variant::Drop {
            without(t, gone(b))
        } else {
            t.to_vec()
        }
    };
    if variant != Variant::SkipLeft {
        for b in left_bonds(i) {
            let xk = x.core(b - 1);
            let ws = su2_onesite_workspace(h, xk);
            let (prev, rest) = fl.split_at_mut(b);
            h.recontract_left_cols(xk, &prev[b - 1], &keep_at(b), &ws, &mut rest[0]);
        }
    }
    if variant != Variant::SkipRight {
        for j in right_bonds(i, d).rev() {
            let xk = x.core(j);
            let ws = su2_onesite_workspace(h, xk);
            let (lo, hi) = fr.split_at_mut(j);
            h.recontract_right_rows(xk, &hi[0], &keep_at(j), &ws, &mut lo[j - 1]);
        }
    }
}

struct Case<'a, T: Scalar> {
    what: String,
    h: &'a Su2SparseHamiltonian<T>,
    idx: &'a Su2TouchIndex,
    y: &'a Su2TtVector<T>,
    d: usize,
    i: usize,
    theta: f64,
    moved: f64,
    fl0: &'a [Su2Sketch<T>],
    fr0: &'a [Su2Sketch<T>],
    ofl: &'a [Su2Sketch<T>],
    ofr: &'a [Su2Sketch<T>],
}

fn for_each_pair_rotation<T: Scalar>(fx: &Fixture, mut visit: impl FnMut(&Case<'_, T>)) {
    let (name, ints64, space, seed) = fx;
    let (space, d) = (*space, space.d);
    let ints = lift::<T>(ints64);
    let mut h = refillable_t(&ints, space);
    let idx = Su2TouchIndex::new(&h);
    let x = state::<T>(space, *seed);
    let fl0 = left_to_right_sketching(&h, &x);
    let fr0 = right_to_left_sketching(&h, &x);
    let e0 = rayleigh_quotient(&h, &x);
    for i in 0..d - 1 {
        if space.pg.irrep_of(i) != space.pg.irrep_of(i + 1) {
            continue;
        }
        for theta in [0.37 + 0.1 * i as f64, FRAC_PI_2] {
            let rot = rotate_integrals(&ints, &givens(d, i, theta));
            let moved = integrals_delta(&rot, &ints);
            h.refill(&rot.t, &rot.v, None);
            let y = rotate_pair_state(&x, i, theta);
            let what = format!(
                "{name} {}: pair ({i}, {}) θ {theta:.4}",
                scalar_name::<T>(),
                i + 1
            );
            let e1 = rayleigh_quotient(&h, &y);
            assert!(
                (e1 - e0).abs_s() <= 1e-10 * e0.abs_s().max(1.0),
                "{what}: ⟨y,H′y⟩ = {e1:?} vs ⟨x,Hx⟩ = {e0:?} — the state gate is not the \
                 integral rotation's partner"
            );
            let ofl = left_to_right_sketching(&h, &y);
            let ofr = right_to_left_sketching(&h, &y);
            visit(&Case {
                what,
                h: &h,
                idx: &idx,
                y: &y,
                d,
                i,
                theta,
                moved,
                fl0: &fl0,
                fr0: &fr0,
                ofl: &ofl,
                ofr: &ofr,
            });
        }
    }
}

fn gauged<T: Scalar>(c: &Case<'_, T>) -> (Vec<Su2Sketch<T>>, Vec<Su2Sketch<T>>) {
    let (mut fl, mut fr) = (c.fl0.to_vec(), c.fr0.to_vec());
    let mut cache = Su2GaugeCache::new(c.d);
    gauge_after_pair_rotation(
        c.h,
        c.y,
        c.i,
        c.theta,
        c.idx,
        &mut fl[..=c.i],
        &mut fr[c.i + 1..],
        &mut cache,
        Su2GaugeMode::Recontract,
    );
    (fl, fr)
}

fn go1_arm<T: Scalar>(fx: &Fixture) {
    let (mut worst, mut untouched, mut cases) = (0.0f64, 0usize, 0usize);
    let mut moving = 0usize;
    for_each_pair_rotation::<T>(fx, |c| {
        let stale = domain_dev(c.fl0, c.fr0, c.ofl, c.ofr, c.i, c.d, &c.what).max();
        moving += usize::from(stale > MOVE);
        let (fl, fr) = gauged(c);
        let dev = domain_dev(&fl, &fr, c.ofl, c.ofr, c.i, c.d, &c.what);
        if let Some((stack, b, rel)) = dev.first_over(TOL) {
            panic!(
                "{}: G-O1 fails — gauged {stack} at bond {b} is off the fresh environment by \
                 {rel:.3e} (relative) > {TOL:.0e}",
                c.what
            );
        }
        assert_outside_domain_bitwise(&fl, &fr, c.fl0, c.fr0, c.i, c.d, &c.what);
        untouched +=
            assert_untouched_bands_bitwise(c.h, c.y, c.idx, c.i, &fl, &fr, c.fl0, c.fr0, &c.what);
        worst = worst.max(dev.max());
        cases += 1;
    });
    let pg = fx.2.pg;
    let same = (0..fx.2.d - 1)
        .filter(|&i| pg.irrep_of(i) == pg.irrep_of(i + 1))
        .count();
    assert_eq!(
        cases,
        2 * same,
        "{} {}: two angles per same-irrep pair",
        fx.0,
        scalar_name::<T>()
    );
    println!(
        "{} {}: G-O1 max deviation {worst:.3e} (relative) over {cases} (pair, θ), {moving} of \
         them moving the stale stacks by > {MOVE:.0e}; {untouched} untouched entries bitwise kept",
        fx.0,
        scalar_name::<T>()
    );
    assert!(
        moving > 0,
        "{} {}: no rotation moves the stale stacks past {MOVE:.0e} — G-O1 is vacuous here",
        fx.0,
        scalar_name::<T>()
    );
    assert!(
        untouched > 0,
        "{} {}: no untouched band in any gauge domain — the bitwise-kept claim is vacuous",
        fx.0,
        scalar_name::<T>()
    );
}

#[test]
fn g1b_t1_gauged_stacks_are_the_fresh_stacks() {
    for fx in all_fixtures().iter().chain(&typed_fixtures()) {
        go1_arm::<f64>(fx);
        go1_arm::<C64>(fx);
    }
}

fn sweep_arm<T: Scalar>(fx: &Fixture) -> usize {
    let (name, ints64, space, seed) = fx;
    let (space, d) = (*space, space.d);
    let ints = lift::<T>(ints64);
    let mut h = refillable_t(&ints, space);
    let idx = Su2TouchIndex::new(&h);
    let mut x = state::<T>(space, *seed);
    let e0 = rayleigh_quotient(&h, &x);
    let mut fl = left_to_right_sketching(&h, &x);
    let mut fr = right_to_left_sketching(&h, &x);
    let mut u = identity_matrix(d);
    let mut cache = Su2GaugeCache::new(d);
    let mut last_key: Vec<Option<(Su2RankGrid, Su2RankGrid)>> = vec![None; d];
    let mut stale_reads = 0usize;
    let mut worst = 0.0f64;
    let schedule: Vec<(usize, f64, bool)> = (0..d - 1)
        .map(|i| (i, 0.37 + 0.1 * i as f64, true))
        .chain(
            (0..d - 1)
                .rev()
                .map(|i| (i, -(0.21 + 0.13 * i as f64), false)),
        )
        .collect();
    for (step, &(i, theta, forward)) in schedule.iter().enumerate() {
        let what = format!(
            "{name} {}: sweep step {step} ({}) pair ({i}, {}) θ {theta:.4}",
            scalar_name::<T>(),
            if forward { "→" } else { "←" },
            i + 1
        );
        u = compose_rotations(&u, &givens(d, i, theta), d);
        let defect = orthogonality_defect(&u, d);
        assert!(
            defect < 1e-12,
            "{what}: U_acc lost orthogonality ({defect:.3e})"
        );
        let rot = rotate_integrals(&ints, &u);
        h.refill(&rot.t, &rot.v, None);
        x = rotate_pair_state(&x, i, theta);
        let e1 = rayleigh_quotient(&h, &x);
        assert!(
            (e1 - e0).abs_s() <= 1e-10 * e0.abs_s().max(1.0),
            "{what}: ⟨x_acc, H(U_acc) x_acc⟩ = {e1:?} vs {e0:?} — the emulation lost the joint \
             rotation"
        );
        for k in (0..i).chain(i + 2..d) {
            let bond = if k < i { k + 1 } else { k };
            if !nonempty(idx.touched_ref(bond, i)) {
                continue;
            }
            let key = (x.core(k).row_ranks.clone(), x.core(k).col_ranks.clone());
            if last_key[k].as_ref().is_some_and(|prev| *prev != key) {
                stale_reads += 1;
            }
            last_key[k] = Some(key);
        }
        gauge_after_pair_rotation(
            &h,
            &x,
            i,
            theta,
            &idx,
            &mut fl[..=i],
            &mut fr[i + 1..],
            &mut cache,
            Su2GaugeMode::Recontract,
        );
        let ofl = left_to_right_sketching(&h, &x);
        let ofr = right_to_left_sketching(&h, &x);
        let dev = domain_dev(&fl, &fr, &ofl, &ofr, i, d, &what);
        if let Some((stack, b, rel)) = dev.first_over(TOL) {
            panic!(
                "{what}: after the gauge, {stack} at bond {b} is off the fresh environment by \
                 {rel:.3e} (relative) > {TOL:.0e} — the chains do not compose"
            );
        }
        worst = worst.max(dev.max());
        if forward {
            let ws = su2_onesite_workspace(&h, x.core(i));
            fl[i + 1] = h.sketching_step_right(&x, &fl[i], &ws);
            let rel = sketch_rel(&fl[i + 1], &ofl[i + 1], &what);
            assert!(
                rel <= TOL,
                "{what}: rebuilt Fᴸ at the pair bond off by {rel:.3e}"
            );
        } else {
            let ws = su2_onesite_workspace(&h, x.core(i + 1));
            fr[i] = h.sketching_step_left(&x, &fr[i + 1], &ws);
            let rel = sketch_rel(&fr[i], &ofr[i], &what);
            assert!(
                rel <= TOL,
                "{what}: rebuilt Fᴿ at the pair bond off by {rel:.3e}"
            );
        }
    }
    println!(
        "{name} {}: accumulated sweep of {} steps — max deviation {worst:.3e} (relative); {} \
         workspace builds; {stale_reads} chain reads of a core whose ranks changed since its \
         previous read",
        scalar_name::<T>(),
        schedule.len(),
        cache.builds()
    );
    stale_reads
}

#[test]
fn g1b_t2_accumulated_sweep_composes() {
    let mut stale_reads = 0;
    for fx in &all_fixtures() {
        stale_reads += sweep_arm::<f64>(fx);
        stale_reads += sweep_arm::<C64>(fx);
    }
    assert!(
        stale_reads > 0,
        "no chain ever read a core whose ranks changed since its previous read — the sweep \
         cannot tell a rank-keyed cache from a site-keyed one"
    );
}

fn touched_band_moves<T: Scalar>(c: &Case<'_, T>, b: usize) -> Vec<(Band, f64)> {
    let (stale, fresh, bands, rows) = if b <= c.i {
        (&c.fl0[b], &c.ofl[b], c.h.col_ranges(c.y.core(b - 1)), false)
    } else {
        (
            &c.fr0[b - 1],
            &c.ofr[b - 1],
            c.h.row_ranges(c.y.core(b)),
            true,
        )
    };
    let mut scale = 0.0f64;
    for s in fresh.sectors() {
        if let Some(f) = fresh.block(s) {
            for col in 0..f.ncols() {
                for r in 0..f.nrows() {
                    scale = scale.max(f[(r, col)].abs_s());
                }
            }
        }
    }
    let scale = scale.max(f64::MIN_POSITIVE);
    let mut out = Vec::new();
    for (s, cs) in c.idx.touched_ref(b, c.i) {
        let (a, f) = match (stale.block(*s), fresh.block(*s)) {
            (Some(a), Some(f)) => (a, f),
            _ => panic!("{}: bond {b}: a touched sector {s:?} has no block", c.what),
        };
        for &ch in cs {
            let mut m = 0.0f64;
            for u in bands[*s][ch].clone() {
                let other = if rows { f.ncols() } else { f.nrows() };
                for o in 0..other {
                    let (r, col) = if rows { (u, o) } else { (o, u) };
                    m = m.max((a[(r, col)] - f[(r, col)]).abs_s());
                }
            }
            out.push(((*s, ch), m / scale));
        }
    }
    out
}

#[derive(Default)]
struct AntiTally {
    moving: usize,
    invariant: Vec<(usize, f64, f64)>,
    blind_fails: usize,
    skip_cases: [usize; 2],
    skip_fails: [usize; 2],
    weakest: [f64; 4],
}

fn anti_arm<T: Scalar>(fx: &Fixture) -> AntiTally {
    let mut tally = AntiTally {
        weakest: [f64::INFINITY; 4],
        ..Default::default()
    };
    for_each_pair_rotation::<T>(fx, |c| {
        let (i, d) = (c.i, c.d);
        let run = |v: Variant, gone: &dyn Fn(usize) -> Option<Band>| {
            let (mut fl, mut fr) = (c.fl0.to_vec(), c.fr0.to_vec());
            local_chain(c.h, c.y, i, c.idx, &mut fl, &mut fr, v, gone);
            domain_dev(&fl, &fr, c.ofl, c.ofr, i, d, &c.what)
        };
        let keep_all = |_: usize| None;
        let stale = domain_dev(c.fl0, c.fr0, c.ofl, c.ofr, i, d, &c.what);
        let control = run(Variant::Intact, &keep_all);
        assert!(
            control.max() <= TOL,
            "{}: the intact test-local chain fails G-O1 ({:.3e}) — the anti-oracles below \
             would not isolate the part they remove",
            c.what,
            control.max()
        );
        if c.moved <= INVARIANT {
            assert!(
                stale.max() <= TOL,
                "{}: the integrals are invariant (max Δ {:.1e}) yet a stale environment is off \
                 the fresh one by {:.3e}",
                c.what,
                c.moved,
                stale.max()
            );
            tally.invariant.push((i, c.theta, c.moved));
            return;
        }
        tally.moving += 1;
        assert!(
            stale.max() > MOVE,
            "{}: the integrals moved (max Δ {:.1e}) but no stale environment did (max {:.3e}) — \
             G-O1 cannot fail here",
            c.what,
            c.moved,
            stale.max()
        );
        tally.weakest[0] = tally.weakest[0].min(stale.max());
        let moves: Vec<(usize, Vec<(Band, f64)>)> = left_bonds(i)
            .chain(right_bonds(i, d))
            .map(|b| (b, touched_band_moves(c, b)))
            .collect();
        let at = |b: usize| -> &[(Band, f64)] {
            moves
                .iter()
                .find(|(bb, _)| *bb == b)
                .map_or(&[][..], |(_, v)| v.as_slice())
        };
        let most_moved = |b: usize| -> Option<Band> {
            at(b)
                .iter()
                .fold(None::<(Band, f64)>, |best, &(band, m)| match best {
                    Some((_, bm)) if bm >= m => best,
                    _ => Some((band, m)),
                })
                .map(|(band, _)| band)
        };
        let dropped = run(Variant::Drop, &most_moved);
        assert!(
            dropped.max() > TOL,
            "{}: dropping the most-moved touched band per bond still passes G-O1 (max {:.3e}) — \
             the oracle cannot see an under-inclusive keep list",
            c.what,
            dropped.max()
        );
        tally.weakest[1] = tally.weakest[1].min(dropped.max());
        let last = |b: usize| -> Option<Band> {
            c.idx
                .touched_ref(b, i)
                .iter()
                .rev()
                .find_map(|(s, cs)| cs.last().map(|&ch| (*s, ch)))
        };
        let blind_move = left_bonds(i)
            .chain(right_bonds(i, d))
            .filter_map(|b| {
                let band = last(b)?;
                at(b).iter().find(|(x, _)| *x == band).map(|&(_, m)| m)
            })
            .fold(0.0f64, f64::max);
        let blind = run(Variant::Drop, &last);
        assert!(
            blind.max() >= blind_move,
            "{}: the oracle-blind drop deviates by {:.3e} < the largest move {blind_move:.3e} \
             of a band it left stale — a later step repaired a band it was never given",
            c.what,
            blind.max()
        );
        tally.blind_fails += usize::from(blind.max() > TOL);
        if i >= 1 {
            let no_left = run(Variant::SkipLeft, &keep_all).max_left();
            tally.skip_cases[0] += 1;
            if no_left > TOL {
                tally.skip_fails[0] += 1;
                tally.weakest[2] = tally.weakest[2].min(no_left);
            }
        }
        if i + 2 < d {
            let no_right = run(Variant::SkipRight, &keep_all).max_right();
            tally.skip_cases[1] += 1;
            if no_right > TOL {
                tally.skip_fails[1] += 1;
                tally.weakest[3] = tally.weakest[3].min(no_right);
            }
        }
    });
    println!(
        "{} {}: {} moving (pair, θ); weakest failure margins — stale {:.3e}, drop-most-moved \
         {:.3e}, skip-left {:.3e}, skip-right {:.3e}; oracle-blind drop fails {} of {}; \
         skip-left fails {} of {}, skip-right fails {} of {}; invariant (pair, θ, max Δ) {:?}",
        fx.0,
        scalar_name::<T>(),
        tally.moving,
        tally.weakest[0],
        tally.weakest[1],
        tally.weakest[2],
        tally.weakest[3],
        tally.blind_fails,
        tally.moving,
        tally.skip_fails[0],
        tally.skip_cases[0],
        tally.skip_fails[1],
        tally.skip_cases[1],
        tally.invariant
    );
    assert!(
        tally.moving > 0,
        "{} {}: no (pair, θ) moves the integrals — the anti-oracles are vacuous",
        fx.0,
        scalar_name::<T>()
    );
    tally
}

#[test]
fn g1b_t3_anti_oracles_fail_where_the_integrals_move() {
    for fx in &all_fixtures() {
        let real = anti_arm::<f64>(fx);
        if fx.0.starts_with("lih") {
            for theta in [0.37 + 0.1 * 3.0, FRAC_PI_2] {
                assert!(
                    real.invariant
                        .iter()
                        .any(|&(i, t, _)| i == 3 && t.to_bits() == theta.to_bits()),
                    "{}: lih's degenerate π pair (3, 4) at θ {theta:.4} was not recognised as a \
                     symmetry of H (invariant: {:?})",
                    fx.0,
                    real.invariant
                );
            }
        }
        let complex = anti_arm::<C64>(fx);
        let dense = fx.0.starts_with("dense");
        for (scalar, t) in [("f64", &real), ("C64", &complex)] {
            for (side, k) in [("left", 0), ("right", 1)] {
                let (fails, cases) = (t.skip_fails[k], t.skip_cases[k]);
                assert!(
                    cases > 0 && if dense { fails == cases } else { fails > 0 },
                    "{} {scalar}: skipping the {side} chain fails G-O1 on {fails} of {cases} \
                     moving cases with a {side} bond — expected {}",
                    fx.0,
                    if dense { "all" } else { "at least one" }
                );
            }
            if dense {
                assert_eq!(
                    t.blind_fails,
                    t.moving,
                    "{} {scalar}: an oracle-blind one-band-short keep list passed G-O1 on {} of \
                     {} moving cases although every channel of the dense set carries a \
                     coefficient",
                    fx.0,
                    t.moving - t.blind_fails,
                    t.moving
                );
            }
        }
    }
}

fn core_max_dim<T: Scalar>(c: &Su2Core<T>) -> usize {
    Family::ALL
        .into_iter()
        .flat_map(|f| c.transitions(f).filter_map(move |s| c.block_dims(f, s)))
        .map(|(r, k)| r.max(k))
        .max()
        .unwrap_or(0)
}

fn serial_arm<T: Scalar>(fx: &Fixture, pools: &[rayon::ThreadPool]) -> usize {
    let mut both = 0usize;
    for_each_pair_rotation::<T>(fx, |c| {
        let (i, d) = (c.i, c.d);
        let r_max = (0..d).map(|k| core_max_dim(c.y.core(k))).max().unwrap_or(0);
        let h_max = (0..d)
            .map(|k| {
                let xk = c.y.core(k);
                c.h.row_ranks(xk).total().max(c.h.col_ranks(xk).total())
            })
            .max()
            .unwrap_or(0);
        assert!(
            r_max.pow(3) < 1 << 21 && r_max * r_max * h_max < 1 << 21,
            "{}: r_max {r_max}, h_max {h_max}: a GEMM may cross the 2^21 gate — the \
             bitwise claim no longer follows; shrink the fixture or relax to machine precision",
            c.what
        );
        let runs: Vec<_> = pools.iter().map(|p| p.install(|| gauged(c))).collect();
        let (fl1, fr1) = &runs[0];
        for ((fl, fr), p) in runs.iter().zip(pools).skip(1) {
            let n = p.current_num_threads();
            for k in 0..d {
                let what = format!("{}: 1 vs {n} threads", c.what);
                let a = sketch_nonbitwise(&fl1[k], &fl[k], &format!("{what}: fl[{k}]"));
                let b = sketch_nonbitwise(&fr1[k], &fr[k], &format!("{what}: fr[{k}]"));
                assert_eq!(
                    (a, b),
                    (0, 0),
                    "{what}: slot {k}: {a} Fᴸ / {b} Fᴿ entries differ in some bit"
                );
            }
        }
        let has = |b: usize| nonempty(c.idx.touched_ref(b, i));
        if left_bonds(i).any(has) && right_bonds(i, d).any(has) {
            both += 1;
        }
    });
    both
}

#[test]
fn g1b_t4_gauge_matches_serial_bitwise() {
    let pools: Vec<rayon::ThreadPool> = [1usize, 8]
        .into_iter()
        .map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        })
        .collect();
    let mut both = 0;
    for fx in &all_fixtures() {
        both += serial_arm::<f64>(fx, &pools);
        both += serial_arm::<C64>(fx, &pools);
    }
    println!("matches-serial: {both} cases ran both chains with a nonempty keep");
    assert!(
        both > 0,
        "no case ran both chains — the rayon::join arm is untested"
    );
}

#[test]
#[should_panic(expected = "refuses a non-refillable operator")]
fn g1b_t5_gauge_refuses_a_non_refillable_operator() {
    let (_, ints, space, seed) = fixtures().swap_remove(0);
    assert_eq!(space.d, 4, "h4 (4,4,0)");
    let idx = Su2TouchIndex::new(&refillable_t::<f64>(&ints, space));
    let plain = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    assert!(!plain.is_refillable());
    let x = state::<f64>(space, seed);
    let mut fl = left_to_right_sketching(&plain, &x);
    let mut fr = right_to_left_sketching(&plain, &x);
    let mut cache = Su2GaugeCache::new(space.d);
    gauge_after_pair_rotation(
        &plain,
        &x,
        1,
        0.37,
        &idx,
        &mut fl[..=1],
        &mut fr[2..],
        &mut cache,
        Su2GaugeMode::Recontract,
    );
}

#[test]
fn g1b_t6_cost_census_report() {
    for fx in &all_fixtures() {
        let mut rows = Vec::new();
        let mut any = 0usize;
        for_each_pair_rotation::<f64>(fx, |c| {
            if c.theta == FRAC_PI_2 {
                return;
            }
            let (i, d) = (c.i, c.d);
            let frac = |bonds: &mut dyn Iterator<Item = usize>| {
                let (mut t, mut n) = (0usize, 0usize);
                for b in bonds {
                    let cs = c.idx.census(b, i, Some(&bond_ranks(c.y, b)));
                    t += cs.touched;
                    n += cs.channels;
                }
                (t, n)
            };
            let (lt, ln) = frac(&mut left_bonds(i));
            let (rt, rn) = frac(&mut right_bonds(i, d));
            assert!(lt <= ln && rt <= rn, "{}: touched exceeds channels", c.what);
            any += lt + rt;
            let t0 = Instant::now();
            let _ = gauged(c);
            let gauge_s = t0.elapsed().as_secs_f64();
            let t0 = Instant::now();
            let _ = (
                left_to_right_sketching(c.h, c.y),
                right_to_left_sketching(c.h, c.y),
            );
            let rebuild_s = t0.elapsed().as_secs_f64();
            let pct = |t: usize, n: usize| {
                if n == 0 {
                    "  —  ".to_string()
                } else {
                    format!("{:5.1}%", 100.0 * t as f64 / n as f64)
                }
            };
            rows.push(format!(
                "  pair ({i}, {}): left {} of {ln:>4} ({} bonds), right {} of {rn:>4} ({} bonds); \
                 gauge {:8.1} µs vs rebuild {:8.1} µs ({:.2}×)",
                i + 1,
                pct(lt, ln),
                left_bonds(i).count(),
                pct(rt, rn),
                right_bonds(i, d).count(),
                gauge_s * 1e6,
                rebuild_s * 1e6,
                gauge_s / rebuild_s.max(f64::MIN_POSITIVE)
            ));
        });
        println!(
            "{} f64 — rank-weighted touched fraction per chain, gauge wall vs full rebuild of \
             both stacks (indicative):\n{}",
            fx.0,
            rows.join("\n")
        );
        assert!(any > 0, "{}: no touched weight on any chain", fx.0);
    }
}

fn cache_arm<T: Scalar>(fx: &Fixture) {
    for_each_pair_rotation::<T>(fx, |c| {
        if c.theta == FRAC_PI_2 {
            return;
        }
        let (i, d) = (c.i, c.d);
        let steps = left_bonds(i).count() + right_bonds(i, d).count();
        let nonempty_cores = left_bonds(i)
            .chain(right_bonds(i, d))
            .filter(|&b| nonempty(c.idx.touched_ref(b, i)))
            .count();
        let (mut fl, mut fr) = (c.fl0.to_vec(), c.fr0.to_vec());
        let mut cache = Su2GaugeCache::new(d);
        gauge_after_pair_rotation(
            c.h,
            c.y,
            i,
            c.theta,
            c.idx,
            &mut fl[..=i],
            &mut fr[i + 1..],
            &mut cache,
            Su2GaugeMode::Recontract,
        );
        let first: usize = cache.builds();
        assert!(
            nonempty_cores <= first && first <= steps,
            "{}: a cold gauge built {first} workspaces; expected one per read core with a \
             nonempty keep ({nonempty_cores}) up to one per chain step ({steps})",
            c.what
        );
        let (fl1, fr1) = (fl.clone(), fr.clone());
        gauge_after_pair_rotation(
            c.h,
            c.y,
            i,
            c.theta,
            c.idx,
            &mut fl[..=i],
            &mut fr[i + 1..],
            &mut cache,
            Su2GaugeMode::Recontract,
        );
        assert_eq!(
            cache.builds(),
            first,
            "{}: a second gauge over unchanged cores rebuilt workspaces — the cache never hits",
            c.what
        );
        for k in 0..d {
            let a = sketch_nonbitwise(&fl[k], &fl1[k], &format!("{}: fl[{k}]", c.what));
            let b = sketch_nonbitwise(&fr[k], &fr1[k], &format!("{}: fr[{k}]", c.what));
            assert_eq!(
                (a, b),
                (0, 0),
                "{}: slot {k}: a second gauge changed {a} Fᴸ / {b} Fᴿ entries — not idempotent",
                c.what
            );
        }
    });
}

#[test]
fn g1b_t7_workspace_cache_hits_and_the_gauge_is_idempotent() {
    for fx in &all_fixtures() {
        cache_arm::<f64>(fx);
        cache_arm::<C64>(fx);
    }
}

fn gauged_as<T: Scalar>(
    c: &Case<'_, T>,
    theta: f64,
    mode: Su2GaugeMode,
) -> (Vec<Su2Sketch<T>>, Vec<Su2Sketch<T>>) {
    let (mut fl, mut fr) = (c.fl0.to_vec(), c.fr0.to_vec());
    let mut cache = Su2GaugeCache::new(c.d);
    gauge_after_pair_rotation(
        c.h,
        c.y,
        c.i,
        theta,
        c.idx,
        &mut fl[..=c.i],
        &mut fr[c.i + 1..],
        &mut cache,
        mode,
    );
    (fl, fr)
}

fn gauged_faulty<T: Scalar>(
    c: &Case<'_, T>,
    fault: &Su2GaugeFault,
) -> (Vec<Su2Sketch<T>>, Vec<Su2Sketch<T>>) {
    let (mut fl, mut fr) = (c.fl0.to_vec(), c.fr0.to_vec());
    let mut cache = Su2GaugeCache::new(c.d);
    gauge_after_pair_rotation_faulty(
        c.h,
        c.y,
        c.i,
        c.theta,
        c.idx,
        &mut fl[..=c.i],
        &mut fr[c.i + 1..],
        &mut cache,
        fault,
    );
    (fl, fr)
}

fn has_pairs(list: &[(Su2Sector, Vec<[u32; 2]>)]) -> bool {
    list.iter().any(|(_, ps)| !ps.is_empty())
}

fn assert_stacks_bitwise<T: Scalar>(
    fl_a: &[Su2Sketch<T>],
    fr_a: &[Su2Sketch<T>],
    fl_b: &[Su2Sketch<T>],
    fr_b: &[Su2Sketch<T>],
    what: &str,
) {
    assert_eq!(
        (fl_a.len(), fr_a.len()),
        (fl_b.len(), fr_b.len()),
        "{what}: stack lengths"
    );
    for k in 0..fl_a.len() {
        let l = sketch_nonbitwise(&fl_a[k], &fl_b[k], &format!("{what}: fl[{k}]"));
        let r = sketch_nonbitwise(&fr_a[k], &fr_b[k], &format!("{what}: fr[{k}]"));
        assert_eq!(
            (l, r),
            (0, 0),
            "{what}: slot {k}: {l} Fᴸ / {r} Fᴿ entries differ in some bit"
        );
    }
}

fn go1_mix_arm<T: Scalar>(fx: &Fixture) {
    let (mut worst, mut untouched, mut cases, mut moving, mut mixing) =
        (0.0f64, 0usize, 0usize, 0usize, 0usize);
    for_each_pair_rotation::<T>(fx, |c| {
        let stale = domain_dev(c.fl0, c.fr0, c.ofl, c.ofr, c.i, c.d, &c.what).max();
        moving += usize::from(stale > MOVE);
        let (fl, fr) = gauged_as(c, c.theta, Su2GaugeMode::MixA);
        let dev = domain_dev(&fl, &fr, c.ofl, c.ofr, c.i, c.d, &c.what);
        if let Some((stack, b, rel)) = dev.first_over(TOL) {
            panic!(
                "{}: MixA G-O1 fails — gauged {stack} at bond {b} is off the fresh environment by \
                 {rel:.3e} (relative) > {TOL:.0e}",
                c.what
            );
        }
        assert_outside_domain_bitwise(&fl, &fr, c.fl0, c.fr0, c.i, c.d, &c.what);
        untouched +=
            assert_untouched_bands_bitwise(c.h, c.y, c.idx, c.i, &fl, &fr, c.fl0, c.fr0, &c.what);
        mixing += usize::from(
            left_bonds(c.i)
                .chain(right_bonds(c.i, c.d))
                .any(|b| has_pairs(c.idx.mix_ref(b, c.i))),
        );
        worst = worst.max(dev.max());
        cases += 1;
    });
    let pg = fx.2.pg;
    let same = (0..fx.2.d - 1)
        .filter(|&i| pg.irrep_of(i) == pg.irrep_of(i + 1))
        .count();
    assert_eq!(
        cases,
        2 * same,
        "{} {}: two angles per same-irrep pair",
        fx.0,
        scalar_name::<T>()
    );
    println!(
        "{} {}: MixA G-O1 max deviation {worst:.3e} (relative) over {cases} (pair, θ), {moving} \
         of them moving the stale stacks by > {MOVE:.0e}, {mixing} with a twin pair to mix; \
         {untouched} untouched entries bitwise kept",
        fx.0,
        scalar_name::<T>()
    );
    assert!(
        moving > 0,
        "{} {}: no rotation moves the stale stacks past {MOVE:.0e} — MixA G-O1 is vacuous here",
        fx.0,
        scalar_name::<T>()
    );
    assert!(
        mixing > 0,
        "{} {}: no case has a twin pair in its domain — the mix is never exercised",
        fx.0,
        scalar_name::<T>()
    );
    assert!(
        untouched > 0,
        "{} {}: no untouched band in any gauge domain — the bitwise-kept claim is vacuous",
        fx.0,
        scalar_name::<T>()
    );
}

#[test]
fn g2b_t2_mix_gauged_stacks_are_the_fresh_stacks() {
    for fx in all_fixtures().iter().chain(&typed_fixtures()) {
        go1_mix_arm::<f64>(fx);
        go1_mix_arm::<C64>(fx);
    }
}

fn sweep_mix_arm<T: Scalar>(fx: &Fixture) {
    let (name, ints64, space, seed) = fx;
    let (space, d) = (*space, space.d);
    let ints = lift::<T>(ints64);
    let mut h = refillable_t(&ints, space);
    let idx = Su2TouchIndex::new(&h);
    let mut x = state::<T>(space, *seed);
    let e0 = rayleigh_quotient(&h, &x);
    let mut fl = left_to_right_sketching(&h, &x);
    let mut fr = right_to_left_sketching(&h, &x);
    let mut u = identity_matrix(d);
    let mut cache = Su2GaugeCache::new(d);
    let mut growth: Vec<f64> = Vec::new();
    let schedule: Vec<(usize, f64, bool)> = (0..d - 1)
        .map(|i| (i, 0.37 + 0.1 * i as f64, true))
        .chain(
            (0..d - 1)
                .rev()
                .map(|i| (i, -(0.21 + 0.13 * i as f64), false)),
        )
        .collect();
    for (step, &(i, theta, forward)) in schedule.iter().enumerate() {
        let what = format!(
            "{name} {}: MixA sweep step {step} ({}) pair ({i}, {}) θ {theta:.4}",
            scalar_name::<T>(),
            if forward { "→" } else { "←" },
            i + 1
        );
        u = compose_rotations(&u, &givens(d, i, theta), d);
        let defect = orthogonality_defect(&u, d);
        assert!(
            defect < 1e-12,
            "{what}: U_acc lost orthogonality ({defect:.3e})"
        );
        let rot = rotate_integrals(&ints, &u);
        h.refill(&rot.t, &rot.v, None);
        x = rotate_pair_state(&x, i, theta);
        let e1 = rayleigh_quotient(&h, &x);
        assert!(
            (e1 - e0).abs_s() <= 1e-10 * e0.abs_s().max(1.0),
            "{what}: ⟨x_acc, H(U_acc) x_acc⟩ = {e1:?} vs {e0:?} — the emulation lost the joint \
             rotation"
        );
        gauge_after_pair_rotation(
            &h,
            &x,
            i,
            theta,
            &idx,
            &mut fl[..=i],
            &mut fr[i + 1..],
            &mut cache,
            Su2GaugeMode::MixA,
        );
        let ofl = left_to_right_sketching(&h, &x);
        let ofr = right_to_left_sketching(&h, &x);
        let dev = domain_dev(&fl, &fr, &ofl, &ofr, i, d, &what);
        if let Some((stack, b, rel)) = dev.first_over(TOL) {
            panic!(
                "{what}: after the MixA gauge, {stack} at bond {b} is off the fresh environment \
                 by {rel:.3e} (relative) > {TOL:.0e} — the mixes do not compose"
            );
        }
        growth.push(dev.max());
        if forward {
            let ws = su2_onesite_workspace(&h, x.core(i));
            fl[i + 1] = h.sketching_step_right(&x, &fl[i], &ws);
            let rel = sketch_rel(&fl[i + 1], &ofl[i + 1], &what);
            assert!(
                rel <= TOL,
                "{what}: rebuilt Fᴸ at the pair bond off by {rel:.3e}"
            );
        } else {
            let ws = su2_onesite_workspace(&h, x.core(i + 1));
            fr[i] = h.sketching_step_left(&x, &fr[i + 1], &ws);
            let rel = sketch_rel(&fr[i], &ofr[i], &what);
            assert!(
                rel <= TOL,
                "{what}: rebuilt Fᴿ at the pair bond off by {rel:.3e}"
            );
        }
    }
    let worst = growth.iter().copied().fold(0.0f64, f64::max);
    println!(
        "{name} {}: MixA accumulated sweep of {} steps — max deviation {worst:.3e} (relative); \
         per step: {}; {} workspace builds",
        scalar_name::<T>(),
        schedule.len(),
        growth
            .iter()
            .map(|g| format!("{g:.1e}"))
            .collect::<Vec<_>>()
            .join(" "),
        cache.builds()
    );
}

#[test]
fn g2b_t3_mix_accumulated_sweep_composes() {
    for fx in &all_fixtures() {
        sweep_mix_arm::<f64>(fx);
        sweep_mix_arm::<C64>(fx);
    }
}

struct BondView<'c, T: Scalar> {
    b: usize,
    stale: &'c Su2Sketch<T>,
    fresh: &'c Su2Sketch<T>,
    bands: Su2SectorGrid<Vec<Range<usize>>>,
    rows: bool,
    scale: f64,
}

impl<T: Scalar> BondView<'_, T> {
    fn at(&self, u: usize, o: usize) -> (usize, usize) {
        if self.rows { (u, o) } else { (o, u) }
    }
}

fn bond_view<'c, T: Scalar>(c: &'c Case<'_, T>, b: usize) -> BondView<'c, T> {
    let (stale, fresh, bands, rows) = if b <= c.i {
        (&c.fl0[b], &c.ofl[b], c.h.col_ranges(c.y.core(b - 1)), false)
    } else {
        (
            &c.fr0[b - 1],
            &c.ofr[b - 1],
            c.h.row_ranges(c.y.core(b)),
            true,
        )
    };
    let mut scale = 0.0f64;
    for s in fresh.sectors() {
        if let Some(f) = fresh.block(s) {
            for col in 0..f.ncols() {
                for r in 0..f.nrows() {
                    scale = scale.max(f[(r, col)].abs_s());
                }
            }
        }
    }
    BondView {
        b,
        stale,
        fresh,
        bands,
        rows,
        scale: scale.max(f64::MIN_POSITIVE),
    }
}

struct PairStat {
    s: Su2Sector,
    lo: u32,
    moved: f64,
    gap: f64,
    content: f64,
}

fn pair_stats<T: Scalar>(c: &Case<'_, T>, v: &BondView<'_, T>) -> Vec<PairStat> {
    let mut out = Vec::new();
    for (s, ps) in c.idx.mix_ref(v.b, c.i) {
        let (a, f) = match (v.stale.block(*s), v.fresh.block(*s)) {
            (Some(a), Some(f)) => (a, f),
            _ => panic!(
                "{}: bond {}: a mixed sector {s:?} has no block",
                c.what, v.b
            ),
        };
        let other = if v.rows { f.ncols() } else { f.nrows() };
        for &[lo, hi] in ps {
            let (rl, rh) = (
                v.bands[*s][lo as usize].clone(),
                v.bands[*s][hi as usize].clone(),
            );
            assert_eq!(
                rl.len(),
                rh.len(),
                "{}: bond {} sector {s:?}: twin bands [{lo}, {hi}] of unequal widths",
                c.what,
                v.b
            );
            let (mut moved, mut gap, mut content) = (0.0f64, 0.0f64, 0.0f64);
            for (ul, uh) in rl.zip(rh) {
                for o in 0..other {
                    let (pl, ph) = (v.at(ul, o), v.at(uh, o));
                    moved = moved
                        .max((a[pl] - f[pl]).abs_s())
                        .max((a[ph] - f[ph]).abs_s());
                    gap = gap.max((f[pl] - f[ph]).abs_s());
                    content = content.max(a[pl].abs_s()).max(a[ph].abs_s());
                }
            }
            out.push(PairStat {
                s: *s,
                lo,
                moved: moved / v.scale,
                gap: gap / v.scale,
                content: content / v.scale,
            });
        }
    }
    out
}

fn max_over_pairs(stats: &[(usize, Vec<PairStat>)], f: impl Fn(&PairStat) -> f64) -> f64 {
    stats
        .iter()
        .flat_map(|(_, ps)| ps.iter().map(&f))
        .fold(0.0f64, f64::max)
}

fn class_b_moved<T: Scalar>(c: &Case<'_, T>, v: &BondView<'_, T>) -> f64 {
    let mut m = 0.0f64;
    for (s, cs) in c.idx.recontract_ref(v.b, c.i) {
        let (a, f) = match (v.stale.block(*s), v.fresh.block(*s)) {
            (Some(a), Some(f)) => (a, f),
            _ => panic!(
                "{}: bond {}: a re-contracted sector {s:?} has no block",
                c.what, v.b
            ),
        };
        let other = if v.rows { f.ncols() } else { f.nrows() };
        for &ch in cs {
            for u in v.bands[*s][ch].clone() {
                for o in 0..other {
                    let p = v.at(u, o);
                    m = m.max((a[p] - f[p]).abs_s());
                }
            }
        }
    }
    m / v.scale
}

fn mix_class_b_locally<T: Scalar>(
    c: &Case<'_, T>,
    fl: &mut [Su2Sketch<T>],
    fr: &mut [Su2Sketch<T>],
) -> usize {
    let (sin, cos) = c.theta.sin_cos();
    let m = [[cos, -sin], [sin, cos]];
    let i = c.i;
    let mut rewritten = 0;
    for b in left_bonds(i).chain(right_bonds(i, c.d)) {
        let v = bond_view(c, b);
        for (s, cs) in c.idx.recontract_ref(b, i) {
            let descs = c.h.channel_descs(b, *s);
            let qns = c.h.state_qns_at(b, *s);
            let lookup: HashMap<(Ts, Su2Sector), usize> = descs
                .iter()
                .zip(qns)
                .enumerate()
                .map(|(k, (&ts, &q))| ((ts, q), k))
                .collect();
            let a = v
                .stale
                .block(*s)
                .expect("a re-contracted sector has a block");
            let other = if v.rows { a.ncols() } else { a.nrows() };
            let mut writes: Vec<((usize, usize), T)> = Vec::new();
            for &ch in cs {
                let ts = descs[ch];
                assert_eq!(
                    touch(&ts, i),
                    Touch::B,
                    "{}: bond {b} sector {s:?}: re-contracted channel {ch} is not class B \
                     (unpaired must be 0 under SymmetryAllowed)",
                    c.what
                );
                let slot = |raw: u16| usize::from(usize::from(raw) == i + 2);
                let (a1, a2) = (slot(ts.1[0]), slot(ts.1[1]));
                let band = v.bands[*s][ch].clone();
                let mut acc = vec![T::zero(); band.len() * other];
                for q1 in 0..2 {
                    for q2 in 0..2 {
                        let mut twin = ts;
                        twin.1 = [(i + 1 + q1) as u16, (i + 1 + q2) as u16];
                        let Some(&tw) = lookup.get(&(twin, qns[ch])) else {
                            continue;
                        };
                        let tb = v.bands[*s][tw].clone();
                        assert_eq!(tb.len(), band.len(), "same-m twins share a width");
                        let w = T::from_f64(m[a1][q1] * m[a2][q2]);
                        for (k, ut) in tb.enumerate() {
                            for o in 0..other {
                                acc[k * other + o] += w * a[v.at(ut, o)];
                            }
                        }
                    }
                }
                for (k, ub) in band.enumerate() {
                    for o in 0..other {
                        writes.push((v.at(ub, o), acc[k * other + o]));
                    }
                }
                rewritten += 1;
            }
            let dst = if b <= i { &mut fl[b] } else { &mut fr[b - 1] };
            let mut blk = dst
                .block_mut(*s)
                .expect("a re-contracted sector has a block");
            for ((r, col), z) in writes {
                *blk.rb_mut().get_mut(r, col) = z;
            }
        }
    }
    rewritten
}

const MIX_ANTI: [&str; 6] = [
    "(a) Gᵀ (MixA at −θ)",
    "(b) twin outputs swapped",
    "(c) most-moved pair per bond unmixed",
    "(d) no mix",
    "(e) class B mixed by g⊗g",
    "(f) re-contract chains skipped",
];

#[derive(Default)]
struct MixAntiTally {
    moving: usize,
    live: [usize; 6],
    fails: [usize; 6],
    weakest: [f64; 6],
}

fn anti_mix_arm<T: Scalar>(fx: &Fixture) -> MixAntiTally {
    let mut t = MixAntiTally {
        weakest: [f64::INFINITY; 6],
        ..Default::default()
    };
    for_each_pair_rotation::<T>(fx, |c| {
        let (i, d) = (c.i, c.d);
        let dev_of = |s: &(Vec<Su2Sketch<T>>, Vec<Su2Sketch<T>>)| {
            domain_dev(&s.0, &s.1, c.ofl, c.ofr, i, d, &c.what).max()
        };
        let prod = gauged_as(c, c.theta, Su2GaugeMode::MixA);
        let control = dev_of(&prod);
        assert!(
            control <= TOL,
            "{}: the intact MixA gauge fails G-O1 ({control:.3e}) — the anti-oracles below would \
             not isolate the part they break",
            c.what
        );
        let intact = gauged_faulty(c, &Su2GaugeFault::Intact);
        assert_stacks_bitwise(
            &intact.0,
            &intact.1,
            &prod.0,
            &prod.1,
            &format!("{}: Su2GaugeFault::Intact vs MixA", c.what),
        );
        if c.moved <= INVARIANT {
            return;
        }
        t.moving += 1;
        let views: Vec<BondView<'_, T>> = left_bonds(i)
            .chain(right_bonds(i, d))
            .map(|b| bond_view(c, b))
            .collect();
        let stats: Vec<(usize, Vec<PairStat>)> =
            views.iter().map(|v| (v.b, pair_stats(c, v))).collect();
        let a_content = max_over_pairs(&stats, |p| p.content);
        let a_gap = max_over_pairs(&stats, |p| p.gap);
        let a_moved = max_over_pairs(&stats, |p| p.moved);
        let b_moved = views
            .iter()
            .map(|v| class_b_moved(c, v))
            .fold(0.0f64, f64::max);
        let mut record = |k: usize, live: bool, dev: f64| {
            if live {
                t.live[k] += 1;
                if dev > TOL {
                    t.fails[k] += 1;
                    t.weakest[k] = t.weakest[k].min(dev);
                }
            }
        };

        let dev = dev_of(&gauged_as(c, -c.theta, Su2GaugeMode::MixA));
        let live = a_content > MOVE;
        assert!(
            !live || dev > TOL,
            "{}: {} passes G-O1 ({dev:.3e}) although a twin band carries {a_content:.3e}",
            c.what,
            MIX_ANTI[0]
        );
        record(0, live, dev);

        let dev = dev_of(&gauged_faulty(c, &Su2GaugeFault::SwapOutputs));
        let live = a_gap > MOVE;
        assert!(
            !live || dev > TOL,
            "{}: {} passes G-O1 ({dev:.3e}) although fresh twins differ by {a_gap:.3e}",
            c.what,
            MIX_ANTI[1]
        );
        record(1, live, dev);

        let most: Vec<(usize, Su2Sector, u32)> = stats
            .iter()
            .filter_map(|(b, ps)| {
                ps.iter()
                    .max_by(|x, y| x.moved.total_cmp(&y.moved))
                    .map(|p| (*b, p.s, p.lo))
            })
            .collect();
        let dev = dev_of(&gauged_faulty(c, &Su2GaugeFault::Unmixed(most)));
        assert!(
            dev >= a_moved,
            "{}: {} deviates by {dev:.3e} < the largest move {a_moved:.3e} of a pair it left \
             stale — something repaired a twin band it was never given",
            c.what,
            MIX_ANTI[2]
        );
        record(2, a_moved > TOL, dev);

        let dev = dev_of(&gauged_faulty(c, &Su2GaugeFault::NoMix));
        assert!(
            dev >= a_moved,
            "{}: {} deviates by {dev:.3e} < the largest twin move {a_moved:.3e}",
            c.what,
            MIX_ANTI[3]
        );
        record(3, a_moved > TOL, dev);

        let mut e = prod.clone();
        let rewritten = mix_class_b_locally(c, &mut e.0, &mut e.1);
        let dev = dev_of(&e);
        record(4, rewritten > 0 && b_moved > MOVE, dev);

        let dev = dev_of(&gauged_faulty(c, &Su2GaugeFault::SkipRecontract));
        assert!(
            dev >= b_moved,
            "{}: {} deviates by {dev:.3e} < the largest class-B move {b_moved:.3e}",
            c.what,
            MIX_ANTI[5]
        );
        record(5, b_moved > TOL, dev);
    });
    println!(
        "{} {}: {} moving (pair, θ); per anti-oracle (fails / live, weakest failure): {}",
        fx.0,
        scalar_name::<T>(),
        t.moving,
        (0..6)
            .map(|k| format!(
                "{} {} / {} ({:.2e})",
                MIX_ANTI[k], t.fails[k], t.live[k], t.weakest[k]
            ))
            .collect::<Vec<_>>()
            .join("; ")
    );
    t
}

#[test]
fn g2b_t4_mix_anti_oracles_fail_where_their_part_moved() {
    for fx in &all_fixtures() {
        for (scalar, t) in [
            ("f64", anti_mix_arm::<f64>(fx)),
            ("C64", anti_mix_arm::<C64>(fx)),
        ] {
            assert!(
                t.moving > 0,
                "{} {scalar}: no (pair, θ) moves the integrals — the anti-oracles are vacuous",
                fx.0
            );
            for (k, label) in MIX_ANTI.iter().enumerate() {
                assert!(
                    t.live[k] > 0,
                    "{} {scalar}: anti-oracle {label} is live on no moving case — vacuous",
                    fx.0
                );
                if k == 4 {
                    assert!(
                        t.fails[k] > 0,
                        "{} {scalar}: {label} passes G-O1 on all {} live cases — class B would be \
                         a pure g⊗g orbit, contradicting spec §1: STOP and report",
                        fx.0,
                        t.live[k]
                    );
                } else {
                    assert_eq!(
                        t.fails[k], t.live[k],
                        "{} {scalar}: {label} fails on only {} of {} live cases",
                        fx.0, t.fails[k], t.live[k]
                    );
                }
            }
        }
    }
}

fn assert_gemm_gate_closed<T: Scalar>(c: &Case<'_, T>) {
    let d = c.d;
    let r_max = (0..d).map(|k| core_max_dim(c.y.core(k))).max().unwrap_or(0);
    let h_max = (0..d)
        .map(|k| {
            let xk = c.y.core(k);
            c.h.row_ranks(xk).total().max(c.h.col_ranks(xk).total())
        })
        .max()
        .unwrap_or(0);
    assert!(
        r_max.pow(3) < 1 << 21 && r_max * r_max * h_max < 1 << 21,
        "{}: r_max {r_max}, h_max {h_max}: a GEMM may cross the 2^21 gate — the bitwise \
         claim no longer follows; shrink the fixture or relax to machine precision",
        c.what
    );
}

fn serial_mix_arm<T: Scalar>(fx: &Fixture, pools: &[rayon::ThreadPool]) -> (usize, usize) {
    let (mut both, mut recon) = (0usize, 0usize);
    for_each_pair_rotation::<T>(fx, |c| {
        let (i, d) = (c.i, c.d);
        assert_gemm_gate_closed(c);
        let runs: Vec<_> = pools
            .iter()
            .map(|p| p.install(|| gauged_as(c, c.theta, Su2GaugeMode::MixA)))
            .collect();
        let (fl1, fr1) = &runs[0];
        for ((fl, fr), p) in runs.iter().zip(pools).skip(1) {
            let what = format!("{}: MixA 1 vs {} threads", c.what, p.current_num_threads());
            assert_stacks_bitwise(fl1, fr1, fl, fr, &what);
        }
        let works =
            |b: usize| has_pairs(c.idx.mix_ref(b, i)) || nonempty(c.idx.recontract_ref(b, i));
        both += usize::from(left_bonds(i).any(works) && right_bonds(i, d).any(works));
        recon += usize::from(
            left_bonds(i)
                .chain(right_bonds(i, d))
                .any(|b| nonempty(c.idx.recontract_ref(b, i))),
        );
    });
    (both, recon)
}

#[test]
fn g2b_t5_mix_matches_serial_bitwise() {
    let pools: Vec<rayon::ThreadPool> = [1usize, 8]
        .into_iter()
        .map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .unwrap()
        })
        .collect();
    let (mut both, mut recon) = (0, 0);
    for fx in all_fixtures().iter().chain(&typed_fixtures()) {
        for (b, r) in [
            serial_mix_arm::<f64>(fx, &pools),
            serial_mix_arm::<C64>(fx, &pools),
        ] {
            both += b;
            recon += r;
        }
    }
    println!(
        "MixA matches-serial: {both} cases with work on both chains, {recon} with a class-B step"
    );
    assert!(
        both > 0,
        "no case has work on both chains — the join arm is untested"
    );
    assert!(
        recon > 0,
        "no case runs a class-B chain step in MixA — the re-contract path is untested"
    );
}

fn theta_zero_arm<T: Scalar>(fx: &Fixture) -> usize {
    let (name, ints64, space, seed) = fx;
    let (space, d) = (*space, space.d);
    let ints = lift::<T>(ints64);
    let h = refillable_t(&ints, space);
    let idx = Su2TouchIndex::new(&h);
    let x = state::<T>(space, *seed);
    let fl0 = left_to_right_sketching(&h, &x);
    let fr0 = right_to_left_sketching(&h, &x);
    let mut live = 0usize;
    for i in (0..d - 1).filter(|&i| space.pg.irrep_of(i) == space.pg.irrep_of(i + 1)) {
        let what = format!("{name} {}: pair ({i}, {})", scalar_name::<T>(), i + 1);
        for theta in [0.0f64, -0.0] {
            let (mut fl, mut fr) = (fl0.clone(), fr0.clone());
            let mut cache = Su2GaugeCache::new(d);
            gauge_after_pair_rotation(
                &h,
                &x,
                i,
                theta,
                &idx,
                &mut fl[..=i],
                &mut fr[i + 1..],
                &mut cache,
                Su2GaugeMode::MixA,
            );
            assert_eq!(
                cache.builds(),
                0,
                "{what}: MixA at θ = {theta:?} built {} workspaces — the θ = 0 exit must come \
                 before any work",
                cache.builds()
            );
            assert_stacks_bitwise(
                &fl,
                &fr,
                &fl0,
                &fr0,
                &format!("{what}: MixA at θ = {theta:?}"),
            );
        }
        let recontracts = left_bonds(i)
            .chain(right_bonds(i, d))
            .any(|b| nonempty(idx.recontract_ref(b, i)));
        if recontracts {
            let (mut fl, mut fr) = (fl0.clone(), fr0.clone());
            let mut cache = Su2GaugeCache::new(d);
            gauge_after_pair_rotation(
                &h,
                &x,
                i,
                0.37,
                &idx,
                &mut fl[..=i],
                &mut fr[i + 1..],
                &mut cache,
                Su2GaugeMode::MixA,
            );
            assert!(
                cache.builds() > 0,
                "{what}: MixA at θ = 0.37 re-contracts class B yet built no workspace"
            );
            live += 1;
        }
    }
    live
}

#[test]
fn g2b_t6_mix_at_theta_zero_is_a_bitwise_no_op() {
    let mut live = 0;
    for fx in all_fixtures().iter().chain(&typed_fixtures()) {
        live += theta_zero_arm::<f64>(fx);
        live += theta_zero_arm::<C64>(fx);
    }
    assert!(
        live > 0,
        "no pair has a class-B band in its domain — the θ = 0 control is vacuous"
    );
}

fn mix_refusal_run(h: &Su2SparseHamiltonian<f64>, seed: u64, i: usize) {
    let space = h.space;
    let idx = Su2TouchIndex::new(h);
    let x = state::<f64>(space, seed);
    let mut fl = left_to_right_sketching(h, &x);
    let mut fr = right_to_left_sketching(h, &x);
    let mut cache = Su2GaugeCache::new(space.d);
    gauge_after_pair_rotation(
        h,
        &x,
        i,
        0.37,
        &idx,
        &mut fl[..=i],
        &mut fr[i + 1..],
        &mut cache,
        Su2GaugeMode::MixA,
    );
}

#[test]
#[should_panic(expected = "refuses a non-refillable operator")]
fn g2b_t7a_mix_refuses_a_non_refillable_operator_first() {
    let (_, ints, space, seed) = fixtures().swap_remove(0);
    assert_eq!(space.d, 4, "h4 (4,4,0)");
    let idx = Su2TouchIndex::new(&refillable_t::<f64>(&ints, space));
    let plain = Su2SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
    assert!(!plain.is_refillable() && plain.term_screen().is_none());
    let x = state::<f64>(space, seed);
    let mut fl = left_to_right_sketching(&plain, &x);
    let mut fr = right_to_left_sketching(&plain, &x);
    let mut cache = Su2GaugeCache::new(space.d);
    gauge_after_pair_rotation(
        &plain,
        &x,
        1,
        0.37,
        &idx,
        &mut fl[..=1],
        &mut fr[2..],
        &mut cache,
        Su2GaugeMode::MixA,
    );
}

#[test]
#[should_panic(expected = "MixA requires a SymmetryAllowed(space.pg) skeleton")]
fn g2b_t7b_mix_refuses_a_mask_screen() {
    let (_, ints, space, seed) = fixtures().swap_remove(0);
    let m = Su2PatternMask::from_magnitude(&ints.t, &ints.v, f64::EPSILON, space.pg);
    let h = Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::Mask(Arc::new(m)),
    );
    assert!(h.is_refillable());
    mix_refusal_run(&h, seed, 1);
}

#[test]
#[should_panic(expected = "MixA requires a SymmetryAllowed(space.pg) skeleton")]
fn g2b_t7c_mix_refuses_a_magnitude_screen() {
    let (_, ints, space, seed) = fixtures().swap_remove(0);
    let h = Su2SparseHamiltonian::new_refillable(
        space,
        &ints.t,
        &ints.v,
        None,
        Su2TermScreen::Magnitude(f64::EPSILON),
    );
    assert!(h.is_refillable());
    mix_refusal_run(&h, seed, 1);
}

#[test]
#[should_panic(expected = "MixA refuses the cross-irrep pair")]
fn g2b_t7d_mix_refuses_a_cross_irrep_pair() {
    let (_, ints, space, seed) = typed_fixtures().swap_remove(0);
    assert_ne!(
        space.pg.irrep_of(3),
        space.pg.irrep_of(4),
        "the typed fixture's pair (3, 4) must be cross-irrep"
    );
    let h = refillable_t::<f64>(&ints, space);
    mix_refusal_run(&h, seed, 3);
}
