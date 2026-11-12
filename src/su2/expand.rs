// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! `expand(x, two_sz)`: the SU(2) → abelian cross-symmetry oracle.

use std::ops::Range;

use faer::MatRef;
use faer::prelude::ReborrowMut;

use crate::core::{Family as AbFamily, SparseCore};
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, RankGrid, Sector};
use crate::scalar::Scalar;
use crate::su2::core::Family as Su2Family;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::vector::Su2TtVector;
use crate::vector::TtVector;

/// `⟨j m; ½ +½ | j+½ m+½⟩` — Condon-Shortley spin-½ Clebsch-Gordan coefficient. Args are the
/// "two-units" integers `two_s = 2j` (LEFT/row SU(2) spin) and `two_m = 2m` (LEFT/row abelian
/// `nup - ndn`); the `+`/`+2` inside the `sqrt` are CG algebra, NOT storage shifts — do
/// not drop them.
pub fn cg_up_raise(two_s: i64, two_m: i64) -> f64 {
    ((two_s + two_m + 2) as f64 / (2 * (two_s + 1)) as f64).sqrt()
}

/// `⟨j m; ½ −½ | j+½ m−½⟩`. See [`cg_up_raise`] for the argument convention.
pub fn cg_dn_raise(two_s: i64, two_m: i64) -> f64 {
    ((two_s - two_m + 2) as f64 / (2 * (two_s + 1)) as f64).sqrt()
}

/// `⟨j m; ½ +½ | j−½ m+½⟩` — the ONLY one of the four with a minus sign.
/// This sign is the anchor pinned by the tests (the two-electron singlet `[2,3]`).
pub fn cg_up_lower(two_s: i64, two_m: i64) -> f64 {
    -((two_s - two_m) as f64 / (2 * (two_s + 1)) as f64).sqrt()
}

/// `⟨j m; ½ −½ | j−½ m−½⟩`.
pub fn cg_dn_lower(two_s: i64, two_m: i64) -> f64 {
    ((two_s + two_m) as f64 / (2 * (two_s + 1)) as f64).sqrt()
}

/// Ascending-`two_s` sub-block layout of an abelian bond sector `(n, two_m)` within the SU(2)
/// multiplet list `qns` at the same bond. Collects every SU(2) spin `two_s` such
/// that `(n, two_s) ∈ qns` and
/// `two_s ≥ |two_m|` (spin `s` can host `Sz = two_m/2` iff `2s ≥ |2Sz|`), **sorted ascending in
/// `two_s`** — the ONE global ordering convention of the whole expansion; row-side (bond `k`)
/// and col-side (bond `k+1`) layouts must both use it or ranges misalign silently. Each `two_s`
/// gets a consecutive range of width `rank_su2(bond, (n, two_s, lambda))`. Returns `(spins,
/// ranges, total)`; `total == 0` means this abelian sector is empty at this bond.
pub fn ab_layout(
    su2_space: &Su2Space,
    k: usize,
    rank_su2: &Su2RankGrid,
    n: usize,
    two_m: i64,
    lambda: Irrep,
) -> (Vec<i64>, Vec<Range<usize>>, usize) {
    let mut spins: Vec<i64> = su2_space
        .bond_sectors(k)
        .into_iter()
        .filter(|s| s.n == n && s.irrep == lambda && (s.two_s as i64) >= two_m.abs())
        .map(|s| s.two_s as i64)
        .collect();
    spins.sort_unstable();

    let mut ranges = Vec::with_capacity(spins.len());
    let mut off = 0usize;
    for &sl in &spins {
        let width = rank_su2[Su2Sector::with_irrep(n, sl as usize, lambda)];
        ranges.push(off..off + width);
        off += width;
    }
    (spins, ranges, off)
}

fn write_subblock<T: Scalar>(
    dst: &mut SparseCore<T>,
    fam: AbFamily,
    sec: Sector,
    rrange: &Range<usize>,
    crange: &Range<usize>,
    src: MatRef<'_, T>,
    cg: f64,
) {
    let scalar = T::from_f64(cg);
    let mut blk = dst.block_mut(fam, sec).expect("abelian block present");
    let mut sub =
        blk.rb_mut()
            .submatrix_mut(rrange.start, crange.start, rrange.len(), crange.len());
    for j in 0..crange.len() {
        for i in 0..rrange.len() {
            *sub.rb_mut().get_mut(i, j) = *src.get(i, j) * scalar;
        }
    }
}

/// Contract the SU(2) train `x` down to the abelian `crate::vector::TtVector`
/// member with `Sz = two_sz / 2`.
/// **Panics** unless `|two_sz| ≤ x.two_s()` and `x.two_s() - two_sz` is
/// even. Builds the abelian `QnSpace` with `Nup = (N+twoSz)/2`,
/// `Ndn = (N-twoSz)/2`, then per bond derives the abelian rank grid from [`ab_layout`] and per
/// site fills the four abelian [`crate::core::Family`] blocks from the SU(2) blocks scaled by
/// the CG closed forms (`cg_up_raise`/`cg_dn_raise`/`cg_up_lower`/`cg_dn_lower`) — see the
/// module docs for which SU(2) branch feeds which abelian spin. `Empty`/`Occ` copy with CG
/// factor exactly 1 (the on-site-singlet-pair phase is PINNED at `+1`).
pub fn expand<T: Scalar>(x: &Su2TtVector<T>, two_sz: i64) -> TtVector<T> {
    let n = x.space.n as i64;
    let two_s = x.space.two_s as i64;
    let d = x.space.d;
    assert!(
        two_sz.abs() <= two_s && (two_s - two_sz).rem_euclid(2) == 0,
        "twoSz={two_sz} is not an Sz member of a 2S={two_s} multiplet"
    );
    let nup = ((n + two_sz) / 2) as usize;
    let ndn = ((n - two_sz) / 2) as usize;
    let ab_space = QnSpace::with_point_group(d, nup, ndn, x.space.pg, x.space.target);

    let abqns: Vec<Vec<Sector>> = (0..=d).map(|k| ab_space.bond_sectors(k)).collect();

    let mut r_ab: Vec<RankGrid> = (0..=d).map(|_| RankGrid::zeros(&ab_space)).collect();
    for k in 0..=d {
        for &s in &abqns[k] {
            let nn = s.nup + s.ndn;
            let twom = s.nup as i64 - s.ndn as i64;
            let (_, _, tot) = ab_layout(&x.space, k, &x.ranks[k], nn, twom, s.irrep);
            r_ab[k][s] = tot;
        }
    }

    let mut cores: Vec<SparseCore<T>> = Vec::with_capacity(d);
    for k in 0..d {
        let a = x.core(k);
        let g = ab_space.pg.irrep_of(k);
        let mut c = SparseCore::<T>::zeros(ab_space, k, &r_ab[k], &r_ab[k + 1]);
        for &s in &abqns[k] {
            let nn = s.nup + s.ndn;
            let twom = s.nup as i64 - s.ndn as i64;
            let (rsls, rranges, rtot) = ab_layout(&x.space, k, &x.ranks[k], nn, twom, s.irrep);
            if rtot == 0 {
                continue;
            }

            let empty_col = s;
            if ab_space.contains(k + 1, empty_col) {
                let (csls, cranges, _) =
                    ab_layout(&x.space, k + 1, &x.ranks[k + 1], nn, twom, empty_col.irrep);
                for (i, &sl) in rsls.iter().enumerate() {
                    if let Some(jp) = csls.iter().position(|&c| c == sl) {
                        let src = a
                            .block(
                                Su2Family::Empty,
                                Su2Sector::with_irrep(nn, sl as usize, s.irrep),
                            )
                            .expect("su2 ○○ block present");
                        write_subblock(
                            &mut c,
                            AbFamily::Empty,
                            s,
                            &rranges[i],
                            &cranges[jp],
                            src,
                            1.0,
                        );
                    }
                }
            }

            let up_col = AbFamily::Up.apply(s, g);
            if ab_space.contains(k + 1, up_col) {
                let (csls, cranges, _) = ab_layout(
                    &x.space,
                    k + 1,
                    &x.ranks[k + 1],
                    nn + 1,
                    twom + 1,
                    up_col.irrep,
                );
                for (i, &sl) in rsls.iter().enumerate() {
                    if let Some(jp) = csls.iter().position(|&c| c == sl + 1) {
                        let src = a
                            .block(
                                Su2Family::Up,
                                Su2Sector::with_irrep(nn, sl as usize, s.irrep),
                            )
                            .expect("su2 up block present");
                        write_subblock(
                            &mut c,
                            AbFamily::Up,
                            s,
                            &rranges[i],
                            &cranges[jp],
                            src,
                            cg_up_raise(sl, twom),
                        );
                    }
                    if let Some(jp) = csls.iter().position(|&c| c == sl - 1) {
                        let src = a
                            .block(
                                Su2Family::Dn,
                                Su2Sector::with_irrep(nn, sl as usize, s.irrep),
                            )
                            .expect("su2 dn block present");
                        write_subblock(
                            &mut c,
                            AbFamily::Up,
                            s,
                            &rranges[i],
                            &cranges[jp],
                            src,
                            cg_up_lower(sl, twom),
                        );
                    }
                }
            }

            let dn_col = AbFamily::Dn.apply(s, g);
            if ab_space.contains(k + 1, dn_col) {
                let (csls, cranges, _) = ab_layout(
                    &x.space,
                    k + 1,
                    &x.ranks[k + 1],
                    nn + 1,
                    twom - 1,
                    dn_col.irrep,
                );
                for (i, &sl) in rsls.iter().enumerate() {
                    if let Some(jp) = csls.iter().position(|&c| c == sl + 1) {
                        let src = a
                            .block(
                                Su2Family::Up,
                                Su2Sector::with_irrep(nn, sl as usize, s.irrep),
                            )
                            .expect("su2 up block present");
                        write_subblock(
                            &mut c,
                            AbFamily::Dn,
                            s,
                            &rranges[i],
                            &cranges[jp],
                            src,
                            cg_dn_raise(sl, twom),
                        );
                    }
                    if let Some(jp) = csls.iter().position(|&c| c == sl - 1) {
                        let src = a
                            .block(
                                Su2Family::Dn,
                                Su2Sector::with_irrep(nn, sl as usize, s.irrep),
                            )
                            .expect("su2 dn block present");
                        write_subblock(
                            &mut c,
                            AbFamily::Dn,
                            s,
                            &rranges[i],
                            &cranges[jp],
                            src,
                            cg_dn_lower(sl, twom),
                        );
                    }
                }
            }

            let occ_col = AbFamily::Occ.apply(s, g);
            if ab_space.contains(k + 1, occ_col) {
                let (csls, cranges, _) = ab_layout(
                    &x.space,
                    k + 1,
                    &x.ranks[k + 1],
                    nn + 2,
                    twom,
                    occ_col.irrep,
                );
                for (i, &sl) in rsls.iter().enumerate() {
                    if let Some(jp) = csls.iter().position(|&c| c == sl) {
                        let src = a
                            .block(
                                Su2Family::Occ,
                                Su2Sector::with_irrep(nn, sl as usize, s.irrep),
                            )
                            .expect("su2 ●● block present");
                        write_subblock(
                            &mut c,
                            AbFamily::Occ,
                            s,
                            &rranges[i],
                            &cranges[jp],
                            src,
                            1.0,
                        );
                    }
                }
            }
        }
        cores.push(c);
    }

    let tt = TtVector::new(ab_space, cores);
    debug_assert!(tt.check());
    tt
}

#[cfg(test)]
mod spg2_tests {
    use super::*;
    use crate::pointgroup::PointGroup;
    use crate::su2::vector::tt_csf;

    #[test]
    fn spg2_expand_space_matches_qnspace_new_at_trivial() {
        let codes = [Su2Family::Up, Su2Family::Dn, Su2Family::Up];
        let x: Su2TtVector<f64> = tt_csf(&codes);
        assert_eq!(x.space.pg, PointGroup::TRIVIAL);
        let y: TtVector<f64> = expand(&x, x.space.two_s as i64 % 2);
        let nup = ((x.space.n as i64 + (x.space.two_s as i64 % 2)) / 2) as usize;
        let ndn = ((x.space.n as i64 - (x.space.two_s as i64 % 2)) / 2) as usize;
        assert_eq!(y.space, QnSpace::new(x.space.d, nup, ndn));
    }
}

#[cfg(test)]
mod s1_tests {
    use super::{cg_dn_lower, cg_dn_raise, cg_up_lower, cg_up_raise, expand};
    use crate::C64;
    use crate::dot as ab_dot;
    use crate::su2::core::Family as Su2Family;
    use crate::su2::ops::dot as su2_dot;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::{Su2TtVector, tt_csf};
    use crate::test_util::randomized_su2_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn cg_closed_forms_match_known_spin_half_clebsch_gordan_values() {
        let s2 = 2.0_f64.sqrt();
        assert!((cg_up_raise(0, 0) - 1.0).abs() < 1e-14);
        assert!((cg_up_lower(0, 0) - 0.0).abs() < 1e-14);

        assert!((cg_up_raise(1, 1) - 1.0).abs() < 1e-14);
        assert!((cg_up_lower(1, 1) - 0.0).abs() < 1e-14);
        assert!((cg_dn_lower(1, 1) - 1.0 / s2).abs() < 1e-14);

        assert!((cg_dn_raise(1, -1) - 1.0).abs() < 1e-14);
        assert!((cg_up_raise(1, -1) - 1.0 / s2).abs() < 1e-14);
        assert!((cg_up_lower(1, -1) - (-1.0 / s2)).abs() < 1e-14);
        assert!((cg_dn_lower(1, -1) - 0.0).abs() < 1e-14);
    }

    #[test]
    fn cg_closed_forms_branch_normalization_identity() {
        for two_s in 0..8i64 {
            for two_m in -two_s..=two_s {
                if (two_s - two_m) % 2 != 0 {
                    continue;
                }
                let up = cg_up_raise(two_s, two_m).powi(2) + cg_up_lower(two_s, two_m).powi(2);
                let dn = cg_dn_raise(two_s, two_m).powi(2) + cg_dn_lower(two_s, two_m).powi(2);
                assert!(
                    (up - 1.0).abs() < 1e-12,
                    "two_s={two_s} two_m={two_m} up={up}"
                );
                assert!(
                    (dn - 1.0).abs() < 1e-12,
                    "two_s={two_s} two_m={two_m} dn={dn}"
                );
            }
        }
    }

    #[test]
    fn s1_17_three_electron_doublets_orthonormal_through_expansion() {
        let d1: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Up, Su2Family::Dn]);
        let d2: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Dn, Su2Family::Up]);
        assert!(d1.check());
        assert!(d2.check());
        assert!((su2_dot(&d1, &d1) - 1.0).abs() < 1e-14);
        assert!((su2_dot(&d2, &d2) - 1.0).abs() < 1e-14);
        assert!(su2_dot(&d1, &d2).abs() < 1e-14);

        let e1 = expand(&d1, 1);
        let e2 = expand(&d2, 1);
        assert!((e1.norm() - 1.0).abs() < 1e-14);
        assert!((e2.norm() - 1.0).abs() < 1e-14);
        assert!(ab_dot(&e1, &e2).abs() < 1e-14);
    }

    #[test]
    #[should_panic(expected = "Sz member")]
    fn s1_21_expand_rejects_invalid_two_sz_parity() {
        let x: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Up]);
        let _ = expand(&x, 1);
    }

    #[test]
    fn s1_22_dot_norm_agree_with_abelian_expansion() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1111_2222);
        for &(d, n, two_s) in &[(6usize, 4usize, 2usize), (7, 5, 1), (5, 4, 0)] {
            let space = Su2Space::new(d, n, two_s);
            let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 4, &mut rng);
            let y: Su2TtVector<f64> = randomized_su2_state(space, 1, 4, &mut rng);
            let ex = expand(&x, two_s as i64);
            let ey = expand(&y, two_s as i64);
            let scale = ex.norm().max(ey.norm()).max(1.0);
            assert!((su2_dot(&x, &y) - ab_dot(&ex, &ey)).abs() < 1e-12 * scale);
            assert!((x.norm() - ex.norm()).abs() < 1e-12 * scale);
        }
    }

    #[test]
    fn s1_23_two_sz_independence_of_expansion_inner_products() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x3333_4444);
        let (d, n, two_s) = (6, 4, 2);
        let space = Su2Space::new(d, n, two_s);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let y: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let s = su2_dot(&x, &y);
        let mut ts = two_s as i64;
        while ts >= -(two_s as i64) {
            let ex = expand(&x, ts);
            let ey = expand(&y, ts);
            assert!(
                (s - ab_dot(&ex, &ey)).abs() < 1e-12 * s.abs().max(1.0),
                "2Sz={ts}"
            );
            ts -= 2;
        }
    }

    #[test]
    fn dot_norm_agree_with_abelian_expansion_complex() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x6C64_6C64);
        let space = Su2Space::new(5, 3, 1);
        let x: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
        let y: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);
        let ex = expand(&x, 1);
        let ey = expand(&y, 1);
        let scale = ex.norm().max(ey.norm()).max(1.0);
        let dot_su2 = su2_dot(&x, &y);
        let dot_ab = ab_dot(&ex, &ey);
        assert!((dot_su2.re - dot_ab.re).abs() < 1e-10 * scale);
        assert!((dot_su2.im - dot_ab.im).abs() < 1e-10 * scale);
        assert!((x.norm() - ex.norm()).abs() < 1e-10 * scale);
    }
}
