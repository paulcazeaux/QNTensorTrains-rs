// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::core::tangent::tangent_dim as ab_tangent_dim;
use crate::su2::core::Family as Su2Family;
use crate::su2::expand::expand;
use crate::su2::flux::flux_ranks;
use crate::su2::qn::{Su2Space, max_ranks};
use crate::su2::riemannian::tangent_bases;
use crate::su2::vector::{tt_csf, tt_randn};
use crate::test_util::seeded;

const D: usize = 4;
const NEL: usize = 4;
const CHANNELS: [usize; 3] = [0, 2, 4];

fn fx_space() -> Su2Space {
    Su2Space::new(D, NEL, 0)
}

fn capped_grids(space: &Su2Space, cap: usize) -> Vec<Su2RankGrid> {
    let caps = max_ranks(space);
    (0..=space.d)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(space);
            for s in space.bond_sectors(b) {
                r[s] = if cap == 0 {
                    caps[b][s]
                } else {
                    caps[b][s].min(cap)
                };
            }
            r
        })
        .collect()
}

fn fx_state(
    cap: usize,
    seed: u64,
) -> (
    Su2Space,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
    Su2TtVector<f64>,
) {
    let space = fx_space();
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, cap), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let (bl, br) = tangent_bases(&x);
    (space, x, bl, br)
}

#[test]
fn csu2_shifted_dim_matches_the_basis_length_and_the_basis_is_orthonormal() {
    let space = fx_space();
    for cap in [0usize, 2, 1] {
        let (_, _x, bl, br) = fx_state(cap, 0xC3_00 + cap as u64);
        for two_s_flux in CHANNELS {
            let dim = tangent_dim_grids(&space, &bl.ranks, two_s_flux);
            let basis: Vec<Su2TtTangent<f64>> = tangent_basis_shifted(&bl, &br, two_s_flux);
            println!(
                "csu2 dim: cap {cap}, 2k = {two_s_flux}: dim {dim}, basis {}",
                basis.len()
            );
            assert!(
                basis.len() <= dim,
                "cap {cap}, 2k = {two_s_flux}: the enumerator accepted {} directions past the \
                 structural cap {dim} — the window-gate over-allocation symptom (C-spec §4 R3)",
                basis.len()
            );
            if cap == 0 {
                assert_eq!(
                    basis.len(),
                    dim,
                    "cap 0 (FULL rank), 2k = {two_s_flux}: the structural count and the physical \
                     span must coincide; a short basis here is the R1 symptom (a dressed frame \
                     carrying a parametrization kernel), not a truncation effect"
                );
            }
            for (i, a) in basis.iter().enumerate() {
                for (j, b) in basis.iter().enumerate() {
                    let want = if i == j { 1.0 } else { 0.0 };
                    let got = a.dot(b);
                    assert!(
                        (got - want).abs() <= 1e-13,
                        "cap {cap}, 2k = {two_s_flux}: the shifted basis must be orthonormal in \
                         the CENTERWEIGHTED tangent metric (H9-b): <b{i}, b{j}> = {got}"
                    );
                }
            }
        }
    }
}

#[test]
fn csu2_flux_zero_basis_is_bit_identical_to_tangent_basis() {
    for cap in [0usize, 2, 1] {
        let (space, _x, bl, br) = fx_state(cap, 0xC3_10 + cap as u64);
        assert_eq!(
            tangent_dim_grids(&space, &bl.ranks, 0),
            tangent_dim(&bl),
            "cap {cap}: tangent_dim_grids at flux 0 must BE tangent_dim"
        );
        let a: Vec<Su2TtTangent<f64>> = tangent_basis(&bl, &br, false);
        let b: Vec<Su2TtTangent<f64>> = tangent_basis_shifted(&bl, &br, 0);
        assert!(
            !a.is_empty(),
            "ANTI-VACUITY: empty unshifted basis at cap {cap}"
        );
        assert_eq!(
            a.len(),
            b.len(),
            "cap {cap}: the flux-0 enumerator accepted a different number of directions — the MGS \
             floors were forked"
        );
        for (k, (p, q)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(p.two_s_flux(), 0);
            assert_eq!(q.two_s_flux(), 0);
            assert_eq!(p.num_coeffs(), q.num_coeffs(), "cap {cap}, vector {k}");
            for i in 0..p.num_coeffs() {
                assert_eq!(
                    p.coeff(i),
                    q.coeff(i),
                    "cap {cap}: tangent_basis_shifted(.., 0) must be tangent_basis(.., false) BIT \
                     FOR BIT (vector {k}, coeff {i})"
                );
            }
        }
    }
}

#[test]
fn csu2_spin_channel_dims_sum_to_the_expanded_abelian_tangent_dim() {
    let space = fx_space();
    for &(cap, want, total) in &[
        (0usize, [20usize, 15, 1], 36usize),
        (2, [19, 15, 1], 35),
        (1, [14, 13, 1], 28),
    ] {
        let (_, x, bl, _br) = fx_state(cap, 0xC3_20 + cap as u64);
        let got: Vec<usize> = CHANNELS
            .iter()
            .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
            .collect();
        let x_ab = expand(&x, 0);
        let want_total = ab_tangent_dim(&x_ab);
        println!(
            "csu2 sum identity: cap {cap}: per-channel {got:?} (sum {}) vs tangent_dim(expand) {}",
            got.iter().sum::<usize>(),
            want_total
        );
        assert_eq!(
            got.as_slice(),
            want.as_slice(),
            "cap {cap}: the per-channel split is measured {want:?} (C-spec §1.5). A total that \
             matches with a wrong split is exactly the compensating error the fan can make"
        );
        assert_eq!(
            got.iter().sum::<usize>(),
            want_total,
            "cap {cap}: ★ Σ_S' dim T^S' must equal tangent_dim(expand(x, 0)) EXACTLY — it fails by \
             +2 here under the window gate (C-spec §1.2)"
        );
        assert_eq!(
            want_total, total,
            "cap {cap}: tangent_dim(expand(x, 0)) is measured {total}"
        );
    }
}

#[test]
fn csu2_sum_identity_holds_with_a_point_group() {
    use crate::pointgroup::{Irrep, PointGroup};
    let pg = PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(1), Irrep::new(0)],
    );
    let space = Su2Space::with_point_group(D, NEL, 0, pg, Irrep::IDENTITY);
    let caps = max_ranks(&space);
    let mut rng = seeded(0xC3_30);
    let mut x: Su2TtVector<f64> = tt_randn(space, &caps, &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let (bl, _br) = tangent_bases(&x);

    let mut multiplane = false;
    for b in 0..=D {
        let planes: std::collections::BTreeSet<usize> = space
            .bond_sectors(b)
            .into_iter()
            .filter(|&s| bl.ranks[b][s] > 0)
            .map(|s| s.irrep.index() as usize)
            .collect();
        if planes.len() > 1 {
            multiplane = true;
        }
    }
    assert!(
        multiplane,
        "ANTI-VACUITY: no bond carried rank in more than one irrep plane, so the per-plane rank \
         sums are untested and this is the trivial-group arm again"
    );

    let got: Vec<usize> = CHANNELS
        .iter()
        .map(|&k| tangent_dim_grids(&space, &bl.ranks, k))
        .collect();
    let want_total = ab_tangent_dim(&expand(&x, 0));
    println!("csu2 sum identity (pg [A,B,B,A]): per-channel {got:?} vs expand {want_total}");
    assert!(
        got[1] > 0,
        "ANTI-VACUITY: the spin flux does not move the irrep, so the triplet channel cannot be \
         irrep-empty — got dim T^{{S=1}} = 0, which means the fan lost the plane"
    );
    assert_eq!(
        got.iter().sum::<usize>(),
        want_total,
        "★ the sum identity must hold with a point group too (per-channel {got:?})"
    );
}

#[test]
fn csu2_triplet_channel_dim_is_the_sector_count_at_full_and_cap2_and_proper_at_cap1() {
    let space = fx_space();
    for &(cap, want) in &[(0usize, 15usize), (2, 15), (1, 13)] {
        let (_, _x, bl, _br) = fx_state(cap, 0xC3_40 + cap as u64);
        let dim = tangent_dim_grids(&space, &bl.ranks, 2);
        assert_eq!(
            dim, want,
            "cap {cap}: dim T^{{S=1}} is measured {want} (C-spec §1.5). 15 is the fixture's \
             triplet multiplet count, so `== 15` is the PRECONDITION of the exactness pin and \
             `13 < 15` at cap 1 is the borrowed-suffix caveat in numbers"
        );
    }
}

#[test]
fn csu2_structurally_empty_channel_is_a_reported_outcome() {
    let x: Su2TtVector<f64> = tt_csf(&[Su2Family::Up, Su2Family::Dn, Su2Family::Up, Su2Family::Dn]);
    let space = x.space;
    assert_eq!(
        (space.d, space.n, space.two_s),
        (D, NEL, 0),
        "the CSF must land on the fixture's own singlet space"
    );
    assert!(x.check());
    let (bl, br) = tangent_bases(&x);

    let out_space = space.with_two_s(4);
    assert_eq!(out_space.two_s, 4);
    assert!(
        !out_space.bond_sectors(D).is_empty(),
        "premise: the quintet SECTOR must be non-empty, or the arm tests an empty Hilbert space"
    );
    assert_eq!(
        flux_ranks(&bl.ranks[1], 4, space, out_space, 1).total(),
        0,
        "premise: the CSF's bond-1 label (1,1) must have no 2k = 4 feeder (tri(1,1,4) is false and \
         the CSF carries neither (1,3) nor (1,5))"
    );
    assert_eq!(
        flux_ranks(&bl.ranks[2], 4, space, out_space, 2).total(),
        0,
        "premise: nor may bond 2's label (2,0) (tri(2,0,4) is false: 4 > 2 + 0) — a zero at ONE \
         bond does not empty a channel, so BOTH are asserted (see this arm's doc for the \
         [Up,Up,Dn,Dn] counter-example, whose quintet channel is live with dim 1)"
    );

    assert_eq!(
        flux_ranks(&bl.ranks[D], 4, space, out_space, D).total(),
        1,
        "premise: the dressed bond-d grid must be rank 1 (at space_out's boundary sector), so a \
         zero-width bond-d seed is NOT what makes P vanish here"
    );

    let dim = tangent_dim_grids(&space, &bl.ranks, 4);
    assert_eq!(dim, 0, "the structurally empty channel must report dim 0");
    let basis: Vec<Su2TtTangent<f64>> = tangent_basis_shifted(&bl, &br, 4);
    assert!(basis.is_empty(), "an empty channel has an empty basis");

    let grids: Vec<Su2RankGrid> = (0..=D)
        .map(|b| {
            let mut r = Su2RankGrid::zeros(&out_space);
            for s in out_space.bond_sectors(b) {
                if out_space.contains(b, s) {
                    r[s] = 1;
                }
            }
            r
        })
        .collect();
    let mut rng = seeded(0xC3_51);
    let y: Su2TtVector<f64> = tt_randn(out_space, &grids, &mut rng);
    let p = Su2TtTangent::project_shifted(&bl, &br, 4, &y);
    assert_eq!(p.two_s_flux(), 4);
    assert_eq!(
        p.norm(),
        0.0,
        "P on a structurally empty channel must be EXACTLY zero — every component is zero-SIZED \
         (bonds 1 and 2 have empty dressed grids above, so no slot has any width; the bond-d seed \
         is NOT the reason: r'[d] = 1 at space_out's boundary sector on every feasible channel) — \
         not a small number"
    );
}

#[test]
fn csu2_d1_flux_zero_keeps_the_h10_special_case() {
    let space = Su2Space::new(1, 2, 0);
    assert_eq!(space.d, 1, "this arm must exercise the d == 1 branch");
    let caps = max_ranks(&space);
    let mut rng = seeded(0xC3_60);
    let mut x: Su2TtVector<f64> = tt_randn(space, &caps, &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    let (bl, br) = tangent_bases(&x);

    let xi = Su2TtTangent::project_shifted(&bl, &br, 0, &x);
    assert_eq!(xi.two_s_flux(), 0);
    let v = xi.to_vector();
    assert!(
        v.check(),
        "d == 1: to_vector must not double-write the single core (H10)"
    );
    assert_eq!(
        v.ranks, x.ranks,
        "d == 1: there is no interior bond to double, so the embedded grid IS x's"
    );
    let dim = tangent_dim_grids(&space, &bl.ranks, 0);
    let basis: Vec<Su2TtTangent<f64>> = tangent_basis_shifted(&bl, &br, 0);
    assert!(
        basis.len() <= dim,
        "d == 1: basis {} vs dim {dim}",
        basis.len()
    );
    assert!(dim > 0, "ANTI-VACUITY: the d == 1 tangent space is empty");
}

#[test]
#[should_panic(expected = "min(N, 2d-N)")]
fn csu2_infeasible_spin_channel_is_refused_at_the_shifted_entry() {
    let space = Su2Space::new(1, 2, 0);
    let caps = max_ranks(&space);
    let mut rng = seeded(0xC3_70);
    let x: Su2TtVector<f64> = tt_randn(space, &caps, &mut rng);
    let (bl, _br) = tangent_bases(&x);
    let _ = tangent_dim_grids(&space, &bl.ranks, 2);
}
