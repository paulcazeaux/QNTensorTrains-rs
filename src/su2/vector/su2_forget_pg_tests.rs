// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::{Su2TtVector, apply_group_element, symmetry_defect, tt_randn};
use crate::C64;
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::QnSpace;
use crate::scalar::Scalar;
use crate::su2::core::Family;
use crate::su2::coreops::{adjoint_core_times_core, rproject};
use crate::su2::expand::expand;
use crate::su2::qn::{Su2Sector, Su2Space, max_ranks};
use crate::test_util::{seeded, su2_capped_grids, su2_coeffs};

const D: usize = 5;

const NS: [(usize, usize); 3] = [(4, 0), (3, 1), (4, 2)];

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[
            Irrep::new(0),
            Irrep::new(1),
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
        ],
    )
}

fn typed_space(n: usize, two_s: usize, target: Irrep) -> Su2Space {
    Su2Space::with_point_group(D, n, two_s, fx_pg(), target)
}

fn reachable_targets(n: usize, two_s: usize) -> Vec<Irrep> {
    let out: Vec<Irrep> = (0..fx_pg().order())
        .map(Irrep::new)
        .filter(|&g| {
            let sp = typed_space(n, two_s, g);
            max_ranks(&sp)[D][Su2Sector::with_irrep(n, two_s, g)] > 0
        })
        .collect();
    assert!(
        out.len() >= 2,
        "fixture: (N, 2S) = ({n}, {two_s}) reaches only {} target(s)",
        out.len()
    );
    out
}

fn typed_state<T: Scalar>(space: Su2Space, seed: u64) -> Su2TtVector<T> {
    tt_randn::<T, _>(space, &su2_capped_grids(&space, 2), &mut seeded(seed))
}

fn assert_multi_irrep_sector<T: Scalar>(x: &Su2TtVector<T>, label: &str) {
    let sp = x.space;
    let order = sp.pg.order();
    let untyped = Su2Space::new(sp.d, sp.n, sp.two_s);
    let hit = (0..=sp.d).any(|b| {
        untyped.bond_sectors(b).into_iter().any(|s| {
            s.two_s >= 1
                && (0..order)
                    .filter(|&ir| {
                        x.ranks[b][Su2Sector::with_irrep(s.n, s.two_s, Irrep::new(ir))] > 0
                    })
                    .count()
                    >= 2
        })
    });
    assert!(
        hit,
        "{label}: anti-vacuity: no bond sector (n, 2s ≥ 1) carries two irrep blocks, so a wrong \
         per-irrep offset would be invisible"
    );
}

fn members(two_s: usize) -> Vec<i64> {
    let t = two_s as i64;
    (0..=t).map(|i| -t + 2 * i).collect()
}

fn max_abs_diff<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs_s())
        .fold(0.0f64, f64::max)
}

fn max_abs<T: Scalar>(a: &[T]) -> f64 {
    a.iter().map(|&x| x.abs_s()).fold(0.0f64, f64::max)
}

fn assert_pooled<T: Scalar>(x: &Su2TtVector<T>, f: &Su2TtVector<T>, label: &str) {
    let sp = x.space;
    let untyped = Su2Space::new(sp.d, sp.n, sp.two_s);
    assert_eq!(f.space, untyped, "{label}: wrong output space");
    assert!(f.check(), "{label}: the embedding is not a legal train");
    assert_eq!(f.ortho, x.ortho, "{label}: `ortho` must be carried over");
    for b in 0..=sp.d {
        for s in untyped.bond_sectors(b) {
            let pooled: usize = (0..sp.pg.order())
                .map(|ir| x.ranks[b][Su2Sector::with_irrep(s.n, s.two_s, Irrep::new(ir))])
                .sum();
            assert_eq!(
                f.ranks[b][s], pooled,
                "{label}, bond {b}, sector {s:?}: rank {} != pooled {pooled}",
                f.ranks[b][s]
            );
        }
        assert_eq!(
            f.ranks[b].total(),
            x.ranks[b].total(),
            "{label}, bond {b}: rank total changed (a rank outside the pooled sectors)"
        );
    }
}

fn assert_members_identical<T: Scalar>(x: &Su2TtVector<T>, f: &Su2TtVector<T>, label: &str) {
    let sp = x.space;
    for m in members(sp.two_s) {
        let ex = expand(x, m);
        let ef = expand(f, m);
        let fe = ex.forget_point_group();
        let nup = ((sp.n as i64 + m) / 2) as usize;
        let ndn = ((sp.n as i64 - m) / 2) as usize;
        assert_eq!(
            ef.space,
            QnSpace::new(sp.d, nup, ndn),
            "{label}, 2Sz = {m}: expand(forget(x)) is not on the untyped abelian space"
        );
        assert_eq!(
            ef.ranks, fe.ranks,
            "{label}, 2Sz = {m}: expand(forget(x)) and forget(expand(x)) disagree on rank grids"
        );
        let (xd, fd, ad) = (ex.to_dense(), ef.to_dense(), fe.to_dense());
        let scale = max_abs(&xd);
        assert!(
            scale > 1e-3,
            "{label}, 2Sz = {m}: anti-vacuity: the typed member has no amplitude ({scale:e})"
        );
        let err_ab = max_abs_diff(&fd, &ad);
        let err_x = max_abs_diff(&fd, &xd);
        println!(
            "sfp {label}, 2Sz = {m}: max |expand(forget x) - forget_ab(expand x)| = {err_ab:e}, \
             max |expand(forget x) - expand(x)| = {err_x:e} (scale {scale:e})"
        );
        assert!(
            err_ab <= 1e-14 * scale,
            "{label}, 2Sz = {m}: expand(forget(x)) != forget_ab(expand(x)) (max diff {err_ab:e}, \
             scale {scale:e}); an offset inconsistent across a bond looks like this"
        );
        assert!(
            err_x <= 1e-14 * scale,
            "{label}, 2Sz = {m}: forget_point_group changed the coefficients (max diff {err_x:e}, \
             scale {scale:e})"
        );
    }
}

fn assert_canonical_layout<T: Scalar>(x: &Su2TtVector<T>, f: &Su2TtVector<T>, label: &str) {
    let sp = x.space;
    let order = sp.pg.order();
    let untyped = f.space;
    let off = |b: usize, n: usize, two_s: usize, ir: u8| -> usize {
        (0..ir)
            .map(|j| x.ranks[b][Su2Sector::with_irrep(n, two_s, Irrep::new(j))])
            .sum()
    };
    let mut placed = 0usize;
    let mut multi = 0usize;
    for k in 0..sp.d {
        let g = sp.pg.irrep_of(k);
        for fam in Family::ALL {
            for lu in untyped.bond_sectors(k) {
                let Some(got) = f.cores[k].block(fam, lu) else {
                    continue;
                };
                let (nr, nc) = (got.nrows(), got.ncols());
                let mut want = faer::Mat::<T>::zeros(nr, nc);
                let mut bands = 0usize;
                for ir in 0..order {
                    let l = Su2Sector::with_irrep(lu.n, lu.two_s, Irrep::new(ir));
                    let Some(t) = fam.apply(l, g) else {
                        continue;
                    };
                    if !sp.contains(k, l) || !sp.contains(k + 1, t) {
                        continue;
                    }
                    let Some(src) = x.cores[k].block(fam, l) else {
                        continue;
                    };
                    if src.nrows() == 0 || src.ncols() == 0 {
                        continue;
                    }
                    let i0 = off(k, l.n, l.two_s, ir);
                    let j0 = off(k + 1, t.n, t.two_s, t.irrep.index());
                    assert!(
                        i0 + src.nrows() <= nr && j0 + src.ncols() <= nc,
                        "{label}, core {k}, {fam:?}, row {l:?}: the typed block does not fit the \
                         pooled block at its canonical offset ({i0}, {j0})"
                    );
                    for j in 0..src.ncols() {
                        for i in 0..src.nrows() {
                            want[(i0 + i, j0 + j)] = src[(i, j)];
                        }
                    }
                    bands += 1;
                    placed += 1;
                }
                if bands >= 2 {
                    multi += 1;
                }
                for j in 0..nc {
                    for i in 0..nr {
                        assert!(
                            got[(i, j)] == want[(i, j)],
                            "{label}, core {k}, {fam:?}, row sector {lu:?}, entry ({i}, {j}): \
                             pooled block is not the canonical direct sum of the typed blocks \
                             (irrep bands at prefix-sum offsets over 0..pg.order(), zero elsewhere)"
                        );
                    }
                }
            }
        }
    }
    assert!(
        placed > 0,
        "{label}: anti-vacuity: no typed block was placed"
    );
    assert!(
        multi > 0,
        "{label}: anti-vacuity: no pooled block holds two or more irrep bands, so a permuted \
         band order would be invisible"
    );
}

#[test]
fn sfp_forget_is_member_identical_at_pooled_ranks() {
    let mut cases = 0usize;
    for (i, &(n, two_s)) in NS.iter().enumerate() {
        for target in reachable_targets(n, two_s) {
            let label = format!("(N, 2S) = ({n}, {two_s}), target {}", target.index());
            let space = typed_space(n, two_s, target);
            let x: Su2TtVector<f64> =
                typed_state(space, 0x5F10 + 16 * i as u64 + target.index() as u64);
            assert_multi_irrep_sector(&x, &label);
            let f = x.forget_point_group();
            assert_pooled(&x, &f, &label);
            assert_canonical_layout(&x, &f, &label);
            assert_members_identical(&x, &f, &label);
            cases += 1;
        }
    }
    assert_eq!(
        cases, 12,
        "anti-vacuity: expected 3 spaces x 4 reachable targets"
    );
}

#[test]
fn sfp_forget_keeps_the_reference_character() {
    let pg = fx_pg();
    let mut cases = 0usize;
    for (i, &(n, two_s)) in NS.iter().enumerate() {
        for target in reachable_targets(n, two_s) {
            if target == Irrep::IDENTITY {
                continue;
            }
            let label = format!("(N, 2S) = ({n}, {two_s}), target {}", target.index());
            let space = typed_space(n, two_s, target);
            let x: Su2TtVector<f64> =
                typed_state(space, 0x5F30 + 16 * i as u64 + target.index() as u64);
            assert_multi_irrep_sector(&x, &label);
            let f = x.forget_point_group();
            assert_eq!(
                f.space.pg,
                PointGroup::TRIVIAL,
                "{label}: the space kept its group"
            );

            let mut saw_minus = false;
            for h in 0..pg.order() {
                let hh = Irrep::new(h);
                let chi = target.character(hh);
                if chi < 0.0 {
                    saw_minus = true;
                }
                let uf = apply_group_element(&f, pg, hh);
                for m in members(two_s) {
                    let fd = expand(&f, m).to_dense();
                    let ud = expand(&uf, m).to_dense();
                    let scale = max_abs(&fd);
                    assert!(scale > 1e-3, "{label}, 2Sz = {m}: anti-vacuity ({scale:e})");
                    let want: Vec<f64> = fd.iter().map(|v| chi * v).collect();
                    let err = max_abs_diff(&ud, &want);
                    assert!(
                        err <= 1e-14 * scale,
                        "{label}, h = {h}, 2Sz = {m}: U_h forget(x) != chi({chi}) forget(x), max \
                         diff {err:e}"
                    );
                    let rel = want
                        .iter()
                        .zip(&ud)
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f64>()
                        .sqrt()
                        / fd.iter().map(|v| v * v).sum::<f64>().sqrt();
                    assert!(
                        rel <= 1e-14,
                        "{label}, h = {h}, 2Sz = {m}: dense symmetry defect {rel:e}"
                    );
                }
            }
            assert!(
                saw_minus,
                "{label}: anti-vacuity: every character was +1, so this arm discriminated nothing"
            );

            let (gamma, defects) = symmetry_defect(&f, pg);
            let worst = defects.iter().cloned().fold(0.0f64, f64::max);
            let (gx, dx) = symmetry_defect(&x, pg);
            let worst_x = dx.iter().cloned().fold(0.0f64, f64::max);
            println!(
                "sfp {label}: symmetry_defect(forget x) gamma = {gamma:?} worst = {worst:e}; \
                 typed input gamma = {gx:?} worst = {worst_x:e}"
            );
            assert_eq!(
                gx, target,
                "{label}: control: the typed input reads back the wrong irrep"
            );
            assert_eq!(
                gamma, target,
                "{label}: wrong irrep read back off forget(x)"
            );
            assert!(
                worst <= 1e-7,
                "{label}: defect {worst:e} above the readout floor"
            );
            cases += 1;
        }
    }
    assert_eq!(
        cases, 9,
        "anti-vacuity: expected 3 spaces x 3 reachable non-identity targets"
    );
}

#[test]
fn sfp_forget_on_a_trivial_group_is_a_clone() {
    let space = Su2Space::new(D, 4, 2);
    let x0: Su2TtVector<f64> = typed_state(space, 0x5F50);
    let mut xc = x0.clone();
    xc.move_core(1, false);
    assert_eq!(xc.ortho, Some(1));
    for (label, x) in [("gaussian", &x0), ("centred", &xc)] {
        let f = x.forget_point_group();
        assert_eq!(f.space, x.space, "{label}: space");
        assert_eq!(f.ranks, x.ranks, "{label}: ranks");
        assert_eq!(f.ortho, x.ortho, "{label}: ortho");
        let mut any = false;
        for k in 0..D {
            let (a, b) = (su2_coeffs(&x.cores[k]), su2_coeffs(&f.cores[k]));
            any |= a.iter().any(|v| v.abs() > 1e-3);
            assert_eq!(
                a, b,
                "{label}, core {k}: the trivial-group path must be an exact clone"
            );
        }
        assert!(any, "{label}: anti-vacuity: the fixture has no amplitude");
    }
}

fn identity_dev(b: faer::MatRef<'_, f64>, n: usize) -> f64 {
    assert_eq!((b.nrows(), b.ncols()), (n, n), "Gram block shape");
    let mut dev = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let want = if i == j { 1.0 } else { 0.0 };
            dev = dev.max((*b.get(i, j) - want).abs());
        }
    }
    dev
}

fn su2_ortho_devs(x: &Su2TtVector<f64>, c: usize) -> (f64, f64, usize, usize, usize) {
    let sp = x.space;
    let (mut ldev, mut rdev) = (0.0f64, 0.0f64);
    let (mut nl, mut nr, mut ntheta) = (0usize, 0usize, 0usize);
    for k in 0..c {
        let g = adjoint_core_times_core(&x.cores[k], &x.cores[k]);
        for t in sp.bond_sectors(k + 1) {
            let r = x.ranks[k + 1][t];
            if r == 0 {
                continue;
            }
            let b = g
                .block(t)
                .expect("left Gram: missing block at a positive-rank sector");
            ldev = ldev.max(identity_dev(b, r));
            nl += 1;
        }
    }
    for k in (c + 1)..sp.d {
        let g = rproject(&x.cores[k], &x.cores[k]);
        for l in sp.bond_sectors(k) {
            let r = x.ranks[k][l];
            if r == 0 {
                continue;
            }
            let b = g
                .block(l)
                .expect("right Gram: missing block at a positive-rank sector");
            rdev = rdev.max(identity_dev(b, r));
            nr += 1;
            let theta_live = l.two_s >= 1
                && [Family::Up, Family::Dn].into_iter().any(|fam| {
                    x.cores[k]
                        .block(fam, l)
                        .is_some_and(|m| m.nrows() > 0 && m.ncols() > 0)
                });
            if theta_live {
                ntheta += 1;
            }
        }
    }
    (ldev, rdev, nl, nr, ntheta)
}

#[test]
fn sfp_forget_preserves_orthogonality() {
    let mut theta_total = 0usize;
    for (i, &(n, two_s)) in NS.iter().enumerate() {
        let target = *reachable_targets(n, two_s)
            .iter()
            .find(|&&g| g != Irrep::IDENTITY)
            .expect("fixture: a non-identity target is reachable");
        let space = typed_space(n, two_s, target);
        let x0: Su2TtVector<f64> = typed_state(space, 0x5F70 + i as u64);
        for c in 0..D {
            let label = format!(
                "(N, 2S) = ({n}, {two_s}), target {}, centre {c}",
                target.index()
            );
            let mut x = x0.clone();
            x.move_core(c, false);
            assert_eq!(
                x.ortho,
                Some(c),
                "{label}: move_core did not flag the centre"
            );
            assert_multi_irrep_sector(&x, &label);

            let (lx, rx, _, _, _) = su2_ortho_devs(&x, c);
            assert!(
                lx <= 1e-12 && rx <= 1e-12,
                "{label}: control: the typed input fails the orthonormality oracle (left {lx:e}, \
                 right {rx:e})"
            );

            let f = x.forget_point_group();
            assert_eq!(f.ortho, Some(c), "{label}: `ortho` not carried over");
            assert!(f.check(), "{label}: the embedding is not a legal train");

            let nx = x.norm();
            assert!(nx > 1e-3, "{label}: anti-vacuity: zero-norm fixture");
            let flagged = f.norm();
            let mut z = f.clone();
            z.ortho = None;
            let gauge_free = z.norm();
            println!(
                "sfp {label}: ‖x‖ = {nx}, flagged ‖f‖ = {flagged}, gauge-free ‖f‖ = {gauge_free}"
            );
            assert!(
                (flagged - gauge_free).abs() <= 1e-12 * nx,
                "{label}: the inherited flag is false: centre-weighted centre-core norm {flagged} \
                 against full contraction {gauge_free}"
            );
            assert!(
                (flagged - nx).abs() <= 1e-12 * nx,
                "{label}: ‖forget(x)‖ = {flagged} vs ‖x‖ = {nx}"
            );

            let (lf, rf, nl, nr, nt) = su2_ortho_devs(&f, c);
            println!(
                "sfp {label}: left Gram dev {lf:e} ({nl} blocks), right θ²-Gram dev {rf:e} ({nr} \
                 blocks, {nt} with 2s ≥ 1 and Up/Dn transitions)"
            );
            assert!(
                lf <= 1e-12,
                "{label}: a left core of forget(x) is not orthonormal ({lf:e})"
            );
            assert!(
                rf <= 1e-12,
                "{label}: a right core of forget(x) is not θ-orthonormal ({rf:e})"
            );
            if c > 0 {
                assert!(nl > 0, "{label}: anti-vacuity: no left block checked");
            }
            if c + 1 < D {
                assert!(nr > 0, "{label}: anti-vacuity: no right block checked");
            }
            theta_total += nt;
        }
    }
    assert!(
        theta_total > 0,
        "anti-vacuity: no checked right-frame sector has 2s ≥ 1 with Up/Dn transitions, \
         so a θ-weighting error would be invisible"
    );
}

#[test]
fn sfp_forget_c64() {
    let (n, two_s) = (3, 1);
    let target = *reachable_targets(n, two_s)
        .iter()
        .find(|&&g| g != Irrep::IDENTITY)
        .expect("fixture: a non-identity target is reachable");
    let label = format!("C64 (N, 2S) = ({n}, {two_s}), target {}", target.index());
    let space = typed_space(n, two_s, target);
    let x: Su2TtVector<C64> = typed_state(space, 0x5F90);
    assert_multi_irrep_sector(&x, &label);
    let f = x.forget_point_group();
    assert_pooled(&x, &f, &label);
    assert_canonical_layout(&x, &f, &label);
    assert_members_identical(&x, &f, &label);
}
