// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::round::RoundMode;
use crate::su2::flux::flux_ranks;
use crate::su2::ops::dot as su2_dot;
use crate::su2::qn::{Su2Sector, Su2Space, max_ranks};
use crate::su2::riemannian::tangent_bases;
use crate::su2::vector::tt_randn;
use crate::test_util::seeded;

const D: usize = 4;
const NEL: usize = 4;
const KAPPA2: usize = 2;

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

fn fx_state(cap: usize, seed: u64) -> (Su2Space, Su2TtVector<f64>, Su2TtVector<f64>) {
    let space = fx_space();
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, cap), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    let (bl, br) = tangent_bases(&x);
    (space, bl, br)
}

fn fx_dir(space: Su2Space, r: &[Su2RankGrid], two_s_flux: usize, seed: u64) -> Su2TtVector<f64> {
    let out_space = space.with_two_s(two_s_flux);
    let grids: Vec<Su2RankGrid> = (0..=space.d)
        .map(|b| {
            let row = flux_ranks(&r[b], 0, space, out_space, b);
            let col = flux_ranks(&r[b], two_s_flux as i64, space, out_space, b);
            let mut g = Su2RankGrid::zeros(&out_space);
            for s in out_space.bond_sectors(b) {
                g[s] = match b {
                    0 => usize::from(row[s] > 0),
                    _ if b == space.d => usize::from(col[s] > 0),
                    _ => row[s] + col[s],
                };
            }
            g
        })
        .collect();
    let mut rng = seeded(seed);
    let mut y: Su2TtVector<f64> = tt_randn(out_space, &grids, &mut rng);
    let n = y.norm_lr(false);
    assert!(n > 1e-8, "fixture: the shifted direction must be nonzero");
    y.scale_mut(1.0 / n);
    y
}

fn a_shifted_tangent() -> (Su2TtVector<f64>, Su2TtTangent<f64>) {
    let (space, bl, br) = fx_state(0, 0xC2_00);
    let y = fx_dir(space, &bl.ranks, KAPPA2, 0xC2_01);
    let xi = Su2TtTangent::project_shifted(&bl, &br, KAPPA2, &y);
    assert_eq!(xi.two_s_flux(), KAPPA2);
    assert!(
        xi.norm() > 1e-6,
        "fixture: the shifted direction must survive the projector, or the refusal arms could \
         pass on a zero object"
    );
    (bl, xi)
}

#[test]
fn csu2_flux_zero_is_bit_identical_to_the_shipped_project() {
    for cap in [0usize, 2, 1] {
        let (space, bl, br) = fx_state(cap, 0xC2_10 + cap as u64);
        let dir = fx_dir(space, &bl.ranks, 0, 0xC2_11 + cap as u64);
        assert_eq!(
            dir.space, space,
            "at flux 0 space_out IS space, so the direction lives in x's own space"
        );

        let a = Su2TtTangent::project(&bl, &br, &dir);
        let b = Su2TtTangent::project_shifted(&bl, &br, 0, &dir);
        assert_eq!(b.two_s_flux(), 0);
        assert_eq!(
            b.rank_col(),
            b.rank(),
            "at flux 0 the column grids must BE the row grids"
        );
        assert_eq!(
            a.num_coeffs(),
            b.num_coeffs(),
            "cap {cap}: coefficient count"
        );
        assert!(
            a.num_coeffs() > 0,
            "ANTI-VACUITY: empty tangent at cap {cap}"
        );
        for i in 0..a.num_coeffs() {
            assert_eq!(
                a.coeff(i),
                b.coeff(i),
                "cap {cap}: project_shifted(.., 0, ..) must be project() BIT FOR BIT (coeff {i}); \
                 a tolerance here would hide a reassociated fold in the path every SU(2) \
                 Riemannian solve runs"
            );
        }
        let (va, vb) = (a.to_vector(), b.to_vector());
        assert_eq!(va.space, vb.space);
        assert_eq!(va.ranks, vb.ranks, "cap {cap}: to_vector grids must agree");
        for k in 0..D {
            let (ca, cb) = (va.core(k), vb.core(k));
            assert_eq!(ca.num_coeffs(), cb.num_coeffs());
            for i in 0..ca.num_coeffs() {
                assert_eq!(
                    ca.coeff(i),
                    cb.coeff(i),
                    "cap {cap}: the three-band to_vector must reproduce the shipped doubling BIT \
                     FOR BIT at flux 0 (core {k}, coeff {i})"
                );
            }
        }
    }
}

#[test]
fn csu2_shifted_frames_are_legal_trains_in_space_out() {
    let (space, bl, br) = fx_state(0, 0xC2_20);
    let out_space = space.with_two_s(KAPPA2);
    let (u, v) = shifted_frames(&bl, &br, KAPPA2);

    for (name, f) in [("mask_restamp(U)", &u), ("V^(1)", &v)] {
        assert_eq!(f.space, out_space, "{name} must be stamped space_out");
        assert!(
            f.check(),
            "{name} must be a LEGAL Su2TtVector in space_out (C-spec §0 F2 — no illegal-frame \
             caveat on this layer)"
        );
    }
    for b in 0..=D {
        assert_eq!(
            u.ranks[b],
            flux_ranks(&bl.ranks[b], 0, space, out_space, b),
            "bond {b}: the left frame's grid is the MASKED copy of x's own"
        );
        assert_eq!(
            v.ranks[b],
            flux_ranks(&bl.ranks[b], KAPPA2 as i64, space, out_space, b),
            "bond {b}: the right frame's grid is the triangle-gated FAN"
        );
    }
    assert_eq!(
        u.ranks[D].total(),
        0,
        "the U-band must have zero width at bond d, or ι's last core is not `[V1 ; delta]`"
    );
    assert_eq!(
        v.ranks[0].total(),
        0,
        "the V-band must have zero width at bond 0, or ι's first core is not `[delta_0 | U_0]`"
    );
    assert_ne!(
        u.ranks, v.ranks,
        "ANTI-VACUITY: the two frames came out at the SAME grids, so the wire fanned nothing"
    );
}

#[test]
fn csu2_iota_is_a_legal_train_and_a_centerweighted_isometry() {
    let (space, bl, br) = fx_state(0, 0xC2_30);
    for two_s_flux in [0usize, 2, 4] {
        let out_space = space.with_two_s(two_s_flux);
        let dirs = [
            fx_dir(space, &bl.ranks, two_s_flux, 0xC2_31 + two_s_flux as u64),
            fx_dir(space, &bl.ranks, two_s_flux, 0xC2_41 + two_s_flux as u64),
        ];
        let xis: Vec<Su2TtTangent<f64>> = dirs
            .iter()
            .map(|y| Su2TtTangent::project_shifted(&bl, &br, two_s_flux, y))
            .collect();
        let nx = xis[0].norm();
        if nx <= 1e-10 {
            println!("csu2 iota: 2k = {two_s_flux} is empty at this base point, skipping");
            continue;
        }

        let w = xis[0].to_vector();
        assert!(
            w.check(),
            "2k = {two_s_flux}: iota(xi) must be a LEGAL train in space_out"
        );
        assert_eq!(w.space, out_space, "2k = {two_s_flux}: iota's space stamp");
        assert_eq!(
            w.space.two_s, two_s_flux,
            "2k = {two_s_flux}: space_out.two_s must BE the flux"
        );
        let boundary = Su2Sector::with_irrep(NEL, two_s_flux, out_space.target);
        assert_eq!(
            (w.ranks[D][boundary], w.ranks[D].total()),
            (1, 1),
            "2k = {two_s_flux}: bond d must be rank 1 at exactly {boundary:?}"
        );

        let np = w.norm_lr(false);
        assert!(
            (np - nx).abs() <= 1e-12 * nx.max(1.0),
            "2k = {two_s_flux}: iota must be an isometry from the CENTERWEIGHTED tangent metric to \
             the physical space_out dot: ||xi|| = {nx} vs ||iota(xi)|| = {np}. A factor \
             sqrt(2S_out+1) here means a frame was left stamped in x's singlet space (C-spec §4 R4)"
        );
        let td = xis[0].dot(&xis[1]);
        let vd = su2_dot(&xis[0].to_vector(), &xis[1].to_vector());
        assert!(
            (td - vd).abs() <= 1e-12 * td.abs().max(1.0),
            "2k = {two_s_flux}: the centerweighted tangent dot ({td}) must equal the physical dot \
             of the embeddings ({vd}) — a norm-only check cannot see a per-sector reweighting that \
             preserves magnitude"
        );
    }
}

#[test]
fn csu2_project_shifted_is_idempotent_and_self_adjoint() {
    for cap in [0usize, 2, 1] {
        let (space, bl, br) = fx_state(cap, 0xC2_50 + cap as u64);
        let y = fx_dir(space, &bl.ranks, KAPPA2, 0xC2_51 + cap as u64);
        let z = fx_dir(space, &bl.ranks, KAPPA2, 0xC2_61 + cap as u64);

        let py = Su2TtTangent::project_shifted(&bl, &br, KAPPA2, &y);
        let n = py.norm();
        assert!(
            n > 1e-6,
            "ANTI-VACUITY (cap {cap}): the projector annihilated the direction ({n:e}), so \
             idempotence and self-adjointness are statements about zero"
        );
        let ppy = Su2TtTangent::project_shifted(&bl, &br, KAPPA2, &py.to_vector());
        assert!(
            py.sub(&ppy).norm() <= 1e-11 * n,
            "cap {cap}: P must be idempotent: ||P(iota(P y)) - P y|| / ||P y|| = {:e}",
            py.sub(&ppy).norm() / n
        );

        let pz = Su2TtTangent::project_shifted(&bl, &br, KAPPA2, &z);
        let lhs = su2_dot(&py.to_vector(), &z);
        let rhs = su2_dot(&y, &pz.to_vector());
        let scale = lhs.abs().max(rhs.abs()).max(1e-3);
        assert!(
            (lhs - rhs).abs() <= 1e-11 * scale,
            "cap {cap}: P must be SELF-ADJOINT in the physical space_out dot: <P y, z> = {lhs} vs \
             <y, P z> = {rhs}. Idempotence alone cannot see the H9-a substitution — a projector \
             built with the plain adjoint instead of the θ²-weighted rproject is still idempotent"
        );
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn csu2_project_shifted_satisfies_the_rectangular_gauge_condition() {
    let (space, bl, br) = fx_state(0, 0xC2_70);
    let out_space = space.with_two_s(KAPPA2);
    let y = fx_dir(space, &bl.ranks, KAPPA2, 0xC2_71);
    let xi = Su2TtTangent::project_shifted(&bl, &br, KAPPA2, &y);
    assert!(xi.norm() > 1e-6, "ANTI-VACUITY: the projection vanished");
    let (u, _v) = shifted_frames(&bl, &br, KAPPA2);

    let mut rectangular = false;
    let mut checked = 0usize;
    for k in 0..D - 1 {
        let g = out_space.pg.irrep_of(k);
        for t in out_space.bond_sectors(k + 1) {
            let (rrow, rcol) = (u.ranks[k + 1][t], xi.rank_col()[k + 1][t]);
            if rrow == 0 || rcol == 0 {
                continue;
            }
            if rrow != rcol {
                rectangular = true;
            }
            let mut acc = vec![vec![0.0f64; rcol]; rrow];
            for l in out_space.bond_sectors(k) {
                for f in Family::ALL {
                    if f.apply(l, g) != Some(t) {
                        continue;
                    }
                    let (Some(a), Some(b)) = (u.core(k).block(f, l), xi.component(k).block(f, l))
                    else {
                        continue;
                    };
                    for i in 0..rrow {
                        for j in 0..rcol {
                            let mut s = 0.0f64;
                            for p in 0..a.nrows() {
                                s += *a.get(p, i) * *b.get(p, j);
                            }
                            acc[i][j] += s;
                        }
                    }
                }
            }
            let worst = acc
                .iter()
                .flat_map(|r| r.iter())
                .fold(0.0f64, |m, z| m.max(z.abs()));
            assert!(
                worst <= 1e-12,
                "gauge condition violated at bond {}, column label {t:?}: worst |U^H delta| = \
                 {worst:e} over a {rrow}x{rcol} block. Without it project∘iota is not a projector \
                 and the shifted MGS orthonormalizes the wrong objects",
                k + 1
            );
            checked += 1;
        }
    }
    assert!(
        checked > 0,
        "ANTI-VACUITY: no interior column label had both a row and a column width"
    );
    assert!(
        rectangular,
        "ANTI-VACUITY: every checked gauge block was SQUARE, so this arm is the unshifted \
         statement in disguise — the fan must make r_row != r_col somewhere"
    );
    println!("csu2 gauge: {checked} interior column labels checked, rectangular = {rectangular}");
}

#[test]
#[should_panic(expected = "undefined on a shifted tangent")]
fn csu2_moved_point_refuses_a_shifted_tangent() {
    let (_x, xi) = a_shifted_tangent();
    let _ = xi.moved_point(0.3);
}

#[test]
#[should_panic(expected = "undefined on a shifted tangent")]
fn csu2_retract_hosvd_refuses_a_shifted_tangent() {
    let (_x, xi) = a_shifted_tangent();
    let _ = xi.retract_hosvd(0.3);
}

#[test]
#[should_panic(expected = "undefined on a shifted tangent")]
fn csu2_retract_round_refuses_a_shifted_tangent() {
    let (_x, xi) = a_shifted_tangent();
    let _ = xi.retract_round(0.3, 1e-10, crate::qn::Pooled::NONE, RoundMode::Rel);
}

#[test]
#[should_panic(expected = "undefined on a shifted tangent")]
fn csu2_projective_transport_refuses_a_shifted_tangent() {
    let (x, xi) = a_shifted_tangent();
    let _ = projective_transport(&x, &xi);
}

#[test]
#[should_panic(expected = "singlet")]
fn csu2_shifted_project_refuses_a_non_singlet_reference() {
    let space = Su2Space::new(D, NEL, 2);
    assert_eq!(space.two_s, 2, "premise: the reference space is a TRIPLET");
    let mut rng = seeded(0xC2_F0);
    let mut x: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, 0), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check(), "premise: the triplet reference must be legal");
    let (bl, br) = tangent_bases(&x);
    let _ = Su2TtTangent::project_shifted(&bl, &br, KAPPA2, &x);
}
