// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::core::weingarten::Terms;
use crate::su2::flux::{Su2ShiftedFrames, flux_ranks};
use crate::su2::qn::{Su2Space, max_ranks};
use crate::su2::vector::tt_randn;
use crate::su2::weingarten::weingarten_assemble_cached;
use crate::test_util::generic_truncated_tv;
use crate::test_util::seeded;

const D: usize = 4;
const NEL: usize = 4;
const TRIPLET: usize = 2;

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

fn fixture(seed: u64) -> (Su2Space, Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
    let (t, v) = generic_truncated_tv(D);
    let space = Su2Space::new(D, NEL, 0);
    let ham = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, 2), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check());
    (space, ham, x)
}

fn shifted_dir(ctx: &Su2WeingartenCtx<'_, f64>, two_s_flux: usize, seed: u64) -> Su2TtTangent<f64> {
    let (bl, br) = ctx.bases();
    let space = bl.space;
    let out = shifted_out_space(space, two_s_flux);
    let grids: Vec<Su2RankGrid> = (0..=space.d)
        .map(|b| {
            let row = flux_ranks(&bl.ranks[b], 0, space, out, b);
            let col = flux_ranks(&bl.ranks[b], two_s_flux as i64, space, out, b);
            let mut g = Su2RankGrid::zeros(&out);
            for s in out.bond_sectors(b) {
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
    let y: Su2TtVector<f64> = tt_randn(out, &grids, &mut rng);
    let mut xi = Su2TtTangent::project_shifted(bl, br, two_s_flux, &y);
    let n = xi.norm();
    assert!(n > 1e-8, "shifted_dir: annihilated");
    xi.scale_mut(1.0 / n);
    xi
}

#[test]
fn csu2b_b_apply_any_at_flux_zero_is_b_apply() {
    let (space, ham, x) = fixture(0xCB_C0);
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let mut rng = seeded(0xCBC0_0001);
    let dir: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, 0), &mut rng);
    let xi = Su2TtTangent::project(bl, br, &dir);
    assert!(xi.norm() > 1e-8);
    let a = ctx.b_apply(&xi);
    let b = ctx.b_apply_any(&xi, None);
    assert!(a.norm() > 1e-8, "ANTI-VACUITY: B ξ = 0 at flux 0");
    assert_eq!(a.num_coeffs(), b.num_coeffs());
    for i in 0..a.num_coeffs() {
        assert_eq!(
            a.coeff(i),
            b.coeff(i),
            "coeff {i}: b_apply must be refusal + b_apply_any(xi, None)"
        );
    }
}

#[test]
fn csu2b_b_apply_any_split_z_matches_the_explicit_normal_residual_on_the_triplet() {
    let (_space, ham, x) = fixture(0xCB_C1);
    let ctx = weingarten_operator(&ham, &x);
    let (bl, br) = ctx.bases();
    let sf = Su2ShiftedFrames::new(bl, br, TRIPLET);
    let z = normal_residual(&ham, bl, br, &x, ctx.lambda);
    let rperp = z.norm_lr(false);
    assert!(
        rperp > 1e-6,
        "premise: a truncated reference has r⊥ ≠ 0 (got {rperp:e})"
    );
    for j in 0..3u64 {
        let xi = shifted_dir(&ctx, TRIPLET, 0xCBC1_0010 + j);
        let split = ctx.b_apply_any(&xi, Some(&sf));
        let explicit = weingarten_assemble_cached(bl, br, &z, &xi, ctx.bond_gauges(), Terms::all());
        assert_eq!(split.two_s_flux(), TRIPLET);
        let d = split.sub(&explicit).norm();
        println!(
            "csu2b b_apply_any dir {j}: ‖r⊥‖ = {rperp:e}, ‖B^(S=1) ξ‖ = {:e}, |split − explicit| = {d:e}",
            explicit.norm()
        );
        assert!(
            explicit.norm() > 1e-8,
            "dir {j}: ANTI-VACUITY — B^(S=1) ξ = 0 (the silent collapse of §1.6)"
        );
        assert!(
            d <= 1e-10 * explicit.norm(),
            "dir {j}: the implicit/explicit split-Z route and the single materialized normal \
             residual must agree on the shifted channel ({d:e} vs ‖Bξ‖ = {:e})",
            explicit.norm()
        );
    }
}

#[test]
#[should_panic(expected = "shifted tangent")]
fn csu2b_b_apply_still_refuses_a_shifted_tangent() {
    let (_space, ham, x) = fixture(0xCB_C2);
    let ctx = weingarten_operator(&ham, &x);
    let xi = shifted_dir(&ctx, TRIPLET, 0xCBC2_0001);
    let _ = ctx.b_apply(&xi);
}

#[test]
#[should_panic(expected = "shifted tangent")]
fn csu2b_b_apply_damped_keeps_its_refusal() {
    let (_space, ham, x) = fixture(0xCB_C3);
    let ctx = weingarten_operator(&ham, &x);
    let xi = shifted_dir(&ctx, TRIPLET, 0xCBC3_0001);
    let _ = ctx.b_apply_damped(&xi, 1e-2);
}
