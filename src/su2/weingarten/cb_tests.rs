// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::su2::flux::{
    CrossDir, Su2ShiftedFrames, cross_wire, flux_ranks, lift_bond_sketch, regrid_core,
    regrid_train, slot_sites_shifted,
};
use crate::su2::qn::max_ranks;
use crate::su2::riemannian::tangent_bases;
use crate::su2::vector::{tt_csf, tt_randn};
use crate::test_util::seeded;

const D: usize = 4;
const NEL: usize = 4;
const TRIPLET: usize = 2;
const QUINTET: usize = 4;

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

type Fixture = (
    Su2Space,
    Su2TtVector<f64>,
    std::sync::Arc<Su2TtVector<f64>>,
    std::sync::Arc<Su2TtVector<f64>>,
    Vec<Su2BondGauge<f64>>,
);

fn fx(cap: usize, seed: u64) -> Fixture {
    let space = fx_space();
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, cap), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    assert!(x.check(), "fixture: the reference must be a legal train");
    let (bl, br) = tangent_bases(&x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    (
        space,
        x,
        std::sync::Arc::new(bl),
        std::sync::Arc::new(br),
        gauges,
    )
}

fn normal_z(bl: &Su2TtVector<f64>, br: &Su2TtVector<f64>, seed: u64) -> Su2TtVector<f64> {
    let space = bl.space;
    let mut rng = seeded(seed);
    let w: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, 0), &mut rng);
    let z = &w - &Su2TtTangent::project(bl, br, &w).to_vector();
    let zn = z.norm_lr(false);
    assert!(
        zn > 1e-6,
        "normal_z: the normal part is degenerate ({zn:e})"
    );
    let tang = Su2TtTangent::project(bl, br, &z).norm();
    assert!(
        tang <= 1e-6 * zn,
        "normal_z: not normal (‖P(z)‖/‖z‖ = {:.3e})",
        tang / zn
    );
    z
}

fn shifted_probe(
    space: Su2Space,
    r: &[Su2RankGrid],
    two_s_flux: usize,
    seed: u64,
) -> Su2TtVector<f64> {
    let out = shifted_out_space(space, two_s_flux);
    let grids: Vec<Su2RankGrid> = (0..=space.d)
        .map(|b| {
            let row = flux_ranks(&r[b], 0, space, out, b);
            let col = flux_ranks(&r[b], two_s_flux as i64, space, out, b);
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
    let mut y: Su2TtVector<f64> = tt_randn(out, &grids, &mut rng);
    let n = y.norm_lr(false);
    assert!(n > 1e-8, "shifted_probe: zero draw");
    y.scale_mut(1.0 / n);
    y
}

fn shifted_dir(
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    two_s_flux: usize,
    seed: u64,
) -> Su2TtTangent<f64> {
    let y = shifted_probe(bl.space, &bl.ranks, two_s_flux, seed);
    let mut xi = Su2TtTangent::project_shifted(bl, br, two_s_flux, &y);
    let n = xi.norm();
    assert!(
        n > 1e-8,
        "shifted_dir: P^(S') annihilated the probe at 2κ = {two_s_flux}"
    );
    xi.scale_mut(1.0 / n);
    xi
}

fn assemble(
    bl: &Su2TtVector<f64>,
    br: &Su2TtVector<f64>,
    z: &Su2TtVector<f64>,
    xi: &Su2TtTangent<f64>,
    gauges: &[Su2BondGauge<f64>],
    terms: Terms,
) -> Su2TtTangent<f64> {
    weingarten_assemble_cached(bl, br, z, xi, gauges, terms)
}

#[test]
fn csu2b_flux_zero_assembly_is_construction_route_invariant() {
    for cap in [2usize, 1] {
        let (space, _x, bl, br, gauges) = fx(cap, 0xCB_01 + cap as u64);
        let z = normal_z(&bl, &br, 0xCB01_0001 + cap as u64);
        let mut rng = seeded(0xCB01_0002 + cap as u64);
        let dir: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, 0), &mut rng);
        let xi_p = Su2TtTangent::project(&bl, &br, &dir);
        let xi_s = Su2TtTangent::project_shifted(&bl, &br, 0, &dir);
        assert_eq!(xi_s.two_s_flux(), 0);
        assert!(xi_p.norm() > 1e-8, "ANTI-VACUITY: zero probe direction");

        let b_p = assemble(&bl, &br, &z, &xi_p, &gauges, Terms::all());
        let b_s = assemble(&bl, &br, &z, &xi_s, &gauges, Terms::all());
        println!(
            "csu2b B7 [cap {cap}]: σ_min = {:e}, ‖B ξ‖ = {:e}, |project − project_shifted(0)| = {:e}",
            min_schmidt(&bl, &br),
            b_p.norm(),
            b_p.sub(&b_s).norm()
        );
        assert!(
            b_p.norm() > 1e-8,
            "ANTI-VACUITY: B is zero on the flux-0 probe"
        );
        assert_eq!(
            b_p.sub(&b_s).norm(),
            0.0,
            "[cap {cap}] project and project_shifted(0) must be indistinguishable to the assembly \
             — at 2κ = 0 no Su2ShiftedFrames, regrid, lift or cross_wire may execute"
        );
        let b_entry = weingarten_assemble(
            &bl,
            &br,
            &z,
            &xi_p,
            FloorPolicy::default_hard(),
            Terms::all(),
        );
        assert_eq!(
            b_entry.sub(&b_p).norm(),
            0.0,
            "[cap {cap}] weingarten_assemble is bond_gauges + weingarten_assemble_cached by \
             definition"
        );
        let src = Su2ExplicitZ::new(&z);
        let b_dressed = weingarten_assemble_sources_dressed(
            &bl,
            &br,
            &[&src],
            &xi_p,
            &gauges,
            Terms::all(),
            None,
        );
        assert_eq!(
            b_dressed.sub(&b_p).norm(),
            0.0,
            "[cap {cap}] the _dressed route with None must BE the flux-0 body"
        );
    }
}

#[test]
fn csu2b_dressed_route_is_bit_identical_on_the_triplet_channel() {
    for cap in [2usize, 1] {
        let (_space, _x, bl, br, gauges) = fx(cap, 0xCB_02 + cap as u64);
        let z = normal_z(&bl, &br, 0xCB02_0001 + cap as u64);
        let xi = shifted_dir(&bl, &br, TRIPLET, 0xCB02_0002 + cap as u64);
        let sf = Su2ShiftedFrames::new(&bl, &br, TRIPLET);
        let src = Su2ExplicitZ::new(&z);
        let a = weingarten_assemble_sources(&bl, &br, &[&src], &xi, &gauges, Terms::all());
        let b = weingarten_assemble_sources_dressed(
            &bl,
            &br,
            &[&src],
            &xi,
            &gauges,
            Terms::all(),
            Some(&sf),
        );
        assert_eq!(a.two_s_flux(), TRIPLET);
        assert_eq!(b.two_s_flux(), TRIPLET);
        assert!(a.norm() > 1e-8, "[cap {cap}] ANTI-VACUITY: B^(S=1) ξ = 0");
        assert_eq!(a.num_coeffs(), b.num_coeffs());
        for i in 0..a.num_coeffs() {
            assert_eq!(
                a.coeff(i),
                b.coeff(i),
                "[cap {cap}] coeff {i}: the two routes must be bit-identical"
            );
        }
    }
}

#[test]
fn csu2b_shifted_assembly_is_linear_in_z() {
    for cap in [2usize, 1] {
        let (_space, _x, bl, br, gauges) = fx(cap, 0xCB_03 + cap as u64);
        let xi = shifted_dir(&bl, &br, TRIPLET, 0xCB03_0001 + cap as u64);
        let z1 = normal_z(&bl, &br, 0xCB03_0002 + cap as u64);
        let z2 = normal_z(&bl, &br, 0xCB03_0003 + cap as u64);
        let zs = &z1 + &z2;
        let b = |z: &Su2TtVector<f64>| assemble(&bl, &br, z, &xi, &gauges, Terms::all());
        let (b1, b2, bs) = (b(&z1), b(&z2), b(&zs));
        let scale = b1.norm().max(b2.norm());
        println!(
            "csu2b linearity [cap {cap}]: ‖B(z1)ξ‖ = {:e}, ‖B(z2)ξ‖ = {:e}, |additivity| = {:e}",
            b1.norm(),
            b2.norm(),
            bs.sub(&b1.add(&b2)).norm()
        );
        assert!(
            scale > 1e-8,
            "[cap {cap}] ANTI-VACUITY: B^(S=1) is identically zero on this direction — the \
             pZ/qZ-seeded-from-a-dressed-frame silent collapse (Cb-spec §1.6)"
        );
        assert!(
            b1.sub(&b2).norm() > 1e-3 * scale,
            "[cap {cap}] ANTI-VACUITY: the two z's give the same image"
        );
        assert!(
            bs.sub(&b1.add(&b2)).norm() <= 1e-12 * scale,
            "[cap {cap}] B(z1 + z2) ≠ B(z1) + B(z2) ({:e} vs scale {scale:e})",
            bs.sub(&b1.add(&b2)).norm()
        );
        for alpha in [-2.5f64, 0.5] {
            let ba = b(&(&z1 * alpha));
            let want = b1.scale(alpha);
            assert!(
                ba.sub(&want).norm() <= 1e-12 * (alpha.abs() * b1.norm()).max(1.0),
                "[cap {cap}] B(αz) ≠ αB(z) at α = {alpha}: {:e}",
                ba.sub(&want).norm()
            );
        }
    }
}

#[test]
fn csu2b_source_decomposition_matches_the_summed_normal_z() {
    for cap in [2usize, 1] {
        let (space, _x, bl, br, gauges) = fx(cap, 0xCB_04 + cap as u64);
        let xi = shifted_dir(&bl, &br, TRIPLET, 0xCB04_0001 + cap as u64);
        let z = normal_z(&bl, &br, 0xCB04_0002 + cap as u64);
        let mut rng = seeded(0xCB04_0003 + cap as u64);
        let z1: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, 0), &mut rng);
        let z2 = &z - &z1;
        let t1 = Su2TtTangent::project(&bl, &br, &z1).norm() / z1.norm_lr(false);
        let t2 = Su2TtTangent::project(&bl, &br, &z2).norm() / z2.norm_lr(false);
        assert!(
            t1 > 1e-2 && t2 > 1e-2,
            "premise: the pieces must carry tangential parts ({t1:e}, {t2:e})"
        );
        let (s1, s2) = (Su2ExplicitZ::new(&z1), Su2ExplicitZ::new(&z2));
        let split = weingarten_assemble_sources(&bl, &br, &[&s1, &s2], &xi, &gauges, Terms::all());
        let whole = assemble(&bl, &br, &z, &xi, &gauges, Terms::all());
        let d = split.sub(&whole).norm();
        println!(
            "csu2b 16d [cap {cap}]: ‖B ξ‖ = {:e}, |split − whole| = {d:e}",
            whole.norm()
        );
        assert!(
            whole.norm() > 1e-8,
            "[cap {cap}] ANTI-VACUITY: B^(S=1) ξ = 0"
        );
        assert!(
            d <= 1e-12 * whole.norm(),
            "[cap {cap}] the two-source assembly must equal the single normal-Z one ({d:e}); a \
             flux-dependent S2 would fail to cancel between the two non-normal pieces"
        );
    }
}

#[test]
fn csu2b_s2_is_identically_zero_at_every_flux() {
    let no_s2 = Terms {
        diag_s1: true,
        diag_s2: false,
        diag_s3: true,
    };
    for cap in [2usize, 1] {
        let (space, _x, bl, br, gauges) = fx(cap, 0xCB_05 + cap as u64);
        let z = normal_z(&bl, &br, 0xCB05_0001 + cap as u64);
        let mut curved = 0usize;
        for two_s_flux in [0usize, TRIPLET, QUINTET] {
            let xi = shifted_dir(&bl, &br, two_s_flux, 0xCB05_0010 + two_s_flux as u64);
            let full = assemble(&bl, &br, &z, &xi, &gauges, Terms::all());
            let ablated = assemble(&bl, &br, &z, &xi, &gauges, no_s2);
            let d = full.sub(&ablated).norm();
            let dim = tangent_dim_grids(&space, &bl.ranks, two_s_flux);
            println!(
                "csu2b S2 [cap {cap} 2κ = {two_s_flux}]: dim {dim}, ‖B ξ‖ = {:e}, |full − no_s2| = {d:e}",
                full.norm()
            );
            let scale = full.norm().max(z.norm_lr(false) * xi.norm());
            assert!(
                d <= 1e-12 * scale,
                "[cap {cap} 2κ = {two_s_flux}] turning S2 off moved the answer by {d:e} — its only \
                 Z-dependent factor is the overlap of a NORMAL Z with a bond variation, zero at every flux"
            );
            if two_s_flux != QUINTET {
                assert!(
                    full.norm() > 1e-8,
                    "[cap {cap} 2κ = {two_s_flux}] ANTI-VACUITY: B is zero"
                );
                curved += 1;
            }
        }
        assert!(
            curved >= 2,
            "the ablation must run on flux 0 and on the triplet"
        );
    }
}

#[test]
fn csu2b_empty_channel_returns_a_well_shaped_zero() {
    let x: Su2TtVector<f64> = tt_csf(&[Family::Up, Family::Dn, Family::Up, Family::Dn]);
    let space = x.space;
    assert_eq!((space.d, space.n, space.two_s), (D, NEL, 0));
    let (bl, br) = tangent_bases(&x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    let out = shifted_out_space(space, QUINTET);
    assert!(
        !out.bond_sectors(D).is_empty(),
        "premise: the quintet SECTOR exists"
    );
    assert_eq!(
        tangent_dim_grids(&space, &bl.ranks, QUINTET),
        0,
        "premise: the quintet CHANNEL is empty"
    );
    assert!(tangent_basis_shifted(&bl, &br, QUINTET).is_empty());
    let z = normal_z(&bl, &br, 0xCB06_0001);

    let rm = flux_grids(&bl.ranks, 0, space, out);
    let rd = flux_grids(&bl.ranks, QUINTET as i64, space, out);
    let out_grids: Vec<Su2RankGrid> = (0..=D)
        .map(|b| {
            let row = flux_ranks(&bl.ranks[b], 0, space, out, b);
            let col = flux_ranks(&bl.ranks[b], QUINTET as i64, space, out, b);
            let mut g = Su2RankGrid::zeros(&out);
            for s in out.bond_sectors(b) {
                g[s] = match b {
                    0 => usize::from(row[s] > 0),
                    _ if b == D => usize::from(col[s] > 0),
                    _ => row[s] + col[s],
                };
            }
            g
        })
        .collect();
    let y: Su2TtVector<f64> = tt_randn(out, &out_grids, &mut seeded(0xCB06_0002));
    assert_eq!(
        y.norm_lr(false),
        0.0,
        "premise: an empty channel's three-band grid is zero-width, so the probe IS the zero train"
    );
    let xi = Su2TtTangent::project_shifted(&bl, &br, QUINTET, &y);
    assert_eq!(xi.two_s_flux(), QUINTET);
    assert_eq!(
        xi.norm(),
        0.0,
        "P^(S=2) on the empty channel is exactly zero"
    );

    let out_t = assemble(&bl, &br, &z, &xi, &gauges, Terms::all());
    assert_eq!(out_t.two_s_flux(), QUINTET, "the flux is carried through");
    assert_eq!(out_t.norm(), 0.0, "an empty channel gives exactly 0");
    assert_eq!(out_t.rank(), &rm[..], "row grids are the masked restamp rm");
    assert_eq!(
        out_t.rank_col(),
        &rd[..],
        "column grids are the dressed fan rd"
    );
    for k in 0..D {
        let c = out_t.component(k);
        assert!(c.check_invariants(), "component {k}");
        assert_eq!(c.space, out, "component {k} stamped space_out");
        assert_eq!(c.row_ranks, rm[k], "component {k} rows");
        assert_eq!(
            c.col_ranks,
            rd[k + 1],
            "component {k}: a zero-COLUMN component (rather than a zero-VALUED one on rd) is what \
             breaks every shape-reading consumer downstream"
        );
    }
}

#[test]
fn csu2b_d1_flux_zero_returns_zero_at_the_declared_shapes() {
    let space = Su2Space::new(1, 2, 0);
    let caps = max_ranks(&space);
    let mut rng = seeded(0xCB_07);
    let mut x: Su2TtVector<f64> = tt_randn(space, &caps, &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));
    let (bl, br) = tangent_bases(&x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    assert!(gauges.is_empty(), "d = 1 has no interior bond gauges");
    assert_eq!(
        space.n.min(2 * space.d - space.n),
        0,
        "no shifted channel is feasible at d = 1, N = 2"
    );

    let mut rng = seeded(0xCB07_0001);
    let w: Su2TtVector<f64> = tt_randn(space, &caps, &mut rng);
    let z = &w - &Su2TtTangent::project(&bl, &br, &w).to_vector();
    assert!(
        z.norm_lr(false) <= 1e-12 * w.norm_lr(false).max(1.0),
        "at d = 1 the normal space is {{0}}: got ‖z‖ = {:e}",
        z.norm_lr(false)
    );
    let mut rng = seeded(0xCB07_0002);
    let dir: Su2TtVector<f64> = tt_randn(space, &caps, &mut rng);
    let xi = Su2TtTangent::project(&bl, &br, &dir);
    let out = assemble(&bl, &br, &z, &xi, &gauges, Terms::all());
    assert_eq!(out.two_s_flux(), 0);
    assert_eq!(out.norm(), 0.0, "d = 1: B is the zero map");
    let c = out.component(0);
    assert!(c.check_invariants());
    assert_eq!(c.space, space);
    assert_eq!(c.row_ranks, bl.ranks[0]);
    assert_eq!(c.col_ranks, bl.ranks[1]);
}

#[test]
fn csu2b_single_slot_quintet_channel_has_a_structurally_zero_b() {
    for cap in [2usize, 1] {
        let (space, _x, bl, br, gauges) = fx(cap, 0xCB_08 + cap as u64);
        let z = normal_z(&bl, &br, 0xCB08_0001 + cap as u64);
        let sites_q = slot_sites_shifted(&space, &bl.ranks, QUINTET);
        let sites_t = slot_sites_shifted(&space, &bl.ranks, TRIPLET);
        assert_eq!(
            sites_q.len(),
            1,
            "[cap {cap}] premise: the quintet channel is single-slot ({sites_q:?})"
        );
        assert!(
            sites_t.len() >= 2,
            "[cap {cap}] premise: the triplet channel is curved ({sites_t:?})"
        );
        let xi_q = shifted_dir(&bl, &br, QUINTET, 0xCB08_0002 + cap as u64);
        let xi_t = shifted_dir(&bl, &br, TRIPLET, 0xCB08_0003 + cap as u64);
        let b_q = assemble(&bl, &br, &z, &xi_q, &gauges, Terms::all());
        let b_t = assemble(&bl, &br, &z, &xi_t, &gauges, Terms::all());
        let scale = z.norm_lr(false);
        println!(
            "csu2b single-slot [cap {cap}]: ‖z‖ = {scale:e}, ‖B^(S=1) ξ‖ = {:e}, ‖B^(S=2) ξ‖ = {:e}",
            b_t.norm(),
            b_q.norm()
        );
        assert!(
            b_t.norm() > 1e-6 * scale,
            "[cap {cap}] ANTI-VACUITY: the triplet B must be nonzero at a generic normal z, or the \
             quintet zero below is indistinguishable from a collapsed chain"
        );
        assert!(
            b_q.norm() <= 1e-12 * scale,
            "[cap {cap}] the quintet channel varies ONE core slot, so the curve x + tι(ξ) is affine \
             and B^(S=2) ≡ 0 structurally; got {:e}",
            b_q.norm()
        );
        assert_eq!(b_q.two_s_flux(), QUINTET);
    }
}

#[test]
fn csu2b_zside_intermediates_live_on_rdz_and_land_in_component_shape() {
    let (space, _x, bl, br, gauges) = fx(2, 0xCB_09);
    let z = normal_z(&bl, &br, 0xCB09_0001);
    let xi = shifted_dir(&bl, &br, TRIPLET, 0xCB09_0002);
    let sf = Su2ShiftedFrames::new(&bl, &br, TRIPLET);
    let out = sf.out_space;
    let r = &bl.ranks;
    let d = D;
    let src = Su2ExplicitZ::new(&z);
    let zsrc: &dyn Su2ZSource<f64> = &src;
    let mut spin = crate::su2::spinalgebra::SpinAlgebra::new();

    let dropped: Vec<Su2Sector> = out
        .bond_sectors(3)
        .into_iter()
        .filter(|&s| sf.rd[3][s] > 0 && sf.rdz[3][s] == 0)
        .collect();
    let dead: Vec<Su2Sector> = space
        .bond_sectors(3)
        .into_iter()
        .filter(|&s| r[3][s] > 0 && !out.contains(3, s))
        .collect();
    println!("csu2b B6 real: bond 3 drops {dropped:?} from rd, and {dead:?} from space");
    assert!(
        !dropped.is_empty() && !dead.is_empty(),
        "ANTI-VACUITY: bond 3 must mask both ways"
    );

    let comps: Vec<Su2Core<f64>> = (0..d).map(|k| xi.component(k).clone()).collect();
    let comps_r = right_project_components(&sf.frames.0, &sf.frames.1, &xi.to_vector());
    let comps_scaled: Vec<Su2Core<f64>> = (0..d - 1)
        .map(|k| {
            let lifted = lift_bond_sketch(
                &gauges[k].r_inv,
                &r[k + 1],
                &sf.rd[k + 1],
                out,
                TRIPLET as i64,
            );
            mul_sketch_right(&comps[k], &lifted)
        })
        .collect();
    for (k, c) in comps_scaled.iter().enumerate() {
        assert_eq!(c.space, out);
        assert_eq!(c.row_ranks, sf.rm[k]);
        assert_eq!(c.col_ranks, sf.rd[k + 1], "comps_scaled[{k}] cols on rd");
    }
    let comps_r_scaled: Vec<Su2Core<f64>> = (1..d)
        .map(|b| {
            let masked = lift_bond_sketch(&gauges[b - 1].r_inv, &r[b], &sf.rm[b], out, 0);
            mul_sketch_left(&masked, &comps_r[b])
        })
        .collect();
    for (k, c) in comps_r_scaled.iter().enumerate() {
        assert_eq!(
            c.row_ranks,
            sf.rm[k + 1],
            "oracle: comps_r_scaled[{}] rows on rm",
            k + 1
        );
        assert_eq!(
            c.col_ranks,
            sf.rd[k + 2],
            "oracle: comps_r_scaled cols on rd"
        );
    }
    for (b, c) in comps_r.iter().enumerate().skip(1) {
        assert_eq!(c.space, out, "δ̂_{b} stamped in space_out");
        assert_eq!(c.row_ranks, sf.rm[b], "δ̂_{b} rows on rm");
        assert_eq!(c.col_ranks, sf.rd[b + 1], "δ̂_{b} cols on rd");
    }
    let rinv_h: Vec<Su2Sketch<f64>> = (1..d)
        .map(|b| {
            adjoint_blocks(&lift_bond_sketch(
                &gauges[b - 1].r_inv,
                &r[b],
                &sf.rdz[b],
                space,
                TRIPLET as i64,
            ))
        })
        .collect();
    for (i, rh) in rinv_h.iter().enumerate() {
        let b = i + 1;
        assert_eq!(rh.space, space, "R⁻ᴴ_{b} stamped in x's space");
        assert_eq!(rh.row_ranks(), sf.rdz[b], "R⁻ᴴ_{b} rows on rdz");
        assert_eq!(rh.col_ranks(), sf.rdz[b], "R⁻ᴴ_{b} cols on rdz");
    }
    assert!(
        rinv_h[2].norm() > 1e-10,
        "ANTI-VACUITY: the conversion R⁻ᴴ is zero at the masked bond 3"
    );
    let comps_scaled_z: Vec<Su2Core<f64>> = (0..d - 1)
        .map(|k| regrid_core(&comps_scaled[k], space, &r[k], &sf.rdz[k + 1]))
        .collect();
    let cross_a_to_b = |c: &Su2Core<f64>, b: usize, spin: &mut SpinAlgebra| {
        cross_wire(
            c,
            CrossDir::AtoB,
            TRIPLET as i64,
            (&sf.rm[b], &sf.rd[b + 1]),
            (&sf.rdz[b], &r[b + 1]),
            out,
            space,
            (&r[b], &r[b + 1]),
            spin,
        )
    };
    let comps_r_z: Vec<Su2Core<f64>> = (1..d)
        .map(|b| cross_a_to_b(&comps_r[b], b, &mut spin))
        .collect();
    let comps_r_scaled_z: Vec<Su2Core<f64>> = (1..d)
        .map(|b| cross_a_to_b(&comps_r_scaled[b - 1], b, &mut spin))
        .collect();
    for k in 0..d - 1 {
        assert_eq!(comps_scaled_z[k].space, space);
        assert_eq!(comps_scaled_z[k].row_ranks, r[k]);
        assert_eq!(comps_scaled_z[k].col_ranks, sf.rdz[k + 1]);
        assert_eq!(comps_r_z[k].space, space);
        assert_eq!(
            comps_r_z[k].row_ranks,
            sf.rdz[k + 1],
            "δ̂ crossed: rows on rdz"
        );
        assert_eq!(comps_r_z[k].col_ranks, r[k + 2], "δ̂ crossed: cols on r");
        assert_eq!(comps_r_scaled_z[k].space, space, "oracle");
        assert_eq!(comps_r_scaled_z[k].row_ranks, sf.rdz[k + 1], "oracle");
        assert_eq!(comps_r_scaled_z[k].col_ranks, r[k + 2], "oracle");
    }
    let pz = left_env_z(&bl, zsrc);
    let qz = right_env_z(&br, zsrc);
    let evl = var_left_env(&sf.zside, zsrc, &comps_scaled_z, &pz);
    let evr_tel = var_right_env(&sf.zside, zsrc, &comps_r_z, &qz);
    let mut evr = evr_tel.clone();
    for (e, rh) in evr[1..d].iter_mut().zip(&rinv_h) {
        e.rmul_sketch_mut(rh);
    }
    let vz = regrid_train(&sf.frames.1, space, &sf.rdz);
    let evr_oracle = var_right_env(&vz, zsrc, &comps_r_scaled_z, &qz);
    assert_eq!(evl.len(), d);
    assert_eq!(evr.len(), d + 1);
    assert_eq!(evr_oracle.len(), d + 1);
    let mut evl_mass = 0.0f64;
    for (b, e) in evl.iter().enumerate() {
        assert_eq!(e.space, space, "eVL[{b}] stamped in x's space");
        assert_eq!(e.row_ranks(), sf.rdz[b], "eVL[{b}] rows on rdz");
        assert_eq!(e.col_ranks(), z.ranks[b], "eVL[{b}] cols on rz");
        evl_mass += e.norm();
    }
    for (name, chain) in [
        ("eVR'", &evr_tel),
        ("eVR", &evr),
        ("oracle eVR", &evr_oracle),
    ] {
        for (b, e) in chain.iter().enumerate() {
            assert_eq!(e.space, space, "{name}[{b}] stamped in x's space");
            assert_eq!(e.row_ranks(), z.ranks[b], "{name}[{b}] rows on rz");
            assert_eq!(e.col_ranks(), sf.rdz[b], "{name}[{b}] cols on rdz");
        }
    }
    assert_eq!(evl[0].norm(), 0.0, "eVL[0] = 0 (rdz[0] = 0)");
    assert_eq!(evr[d].norm(), 0.0, "eVR[d] = 0 (rdz[d] = 0)");
    assert_eq!(evr_oracle[d].norm(), 0.0, "oracle eVR[d] = 0 (rdz[d] = 0)");
    assert!(
        evl[3].norm() > 1e-10,
        "ANTI-VACUITY: eVL[3] is zero at the masked bond"
    );
    assert!(evr[1].norm() > 1e-10, "ANTI-VACUITY: eVR[1] is zero");
    assert!(
        evr_oracle[1].norm() > 1e-10,
        "ANTI-VACUITY: the oracle's eVR[1] is zero"
    );
    println!(
        "csu2b B6 real: Σ‖eVL‖ = {evl_mass:e}, ‖eVL[3]‖ = {:e}, ‖eVR[1]‖ = {:e} (oracle {:e}), \
         ‖R⁻ᴴ_3‖ = {:e}",
        evl[3].norm(),
        evr[1].norm(),
        evr_oracle[1].norm(),
        rinv_h[2].norm()
    );

    for k in 0..d {
        if k == d - 1 {
            let boundary = zsrc.left_mul(&evl[k], k);
            assert_eq!(boundary.row_ranks, sf.rdz[k]);
            assert_eq!(boundary.col_ranks, z.ranks[d]);
            let crossed = cross_wire(
                &boundary,
                CrossDir::BtoA,
                TRIPLET as i64,
                (&sf.rdz[k], &z.ranks[d]),
                (&sf.rm[k], &sf.rd[d]),
                space,
                out,
                (&r[k], &r[d]),
                &mut spin,
            );
            assert_eq!(crossed.space, out);
            assert_eq!(crossed.row_ranks, sf.rm[k]);
            assert_eq!(
                crossed.col_ranks, sf.rd[d],
                "the crossing moves the boundary column onto space_out's target sector"
            );
            assert!(crossed.check_invariants());
        } else {
            let zq = zsrc.right_mul(k, &qz[k + 1]);
            let s1 = mul_sketch_left(&evl[k], &zq);
            assert_eq!(s1.row_ranks, sf.rdz[k], "S1_{k} rows on rdz");
            assert_eq!(s1.col_ranks, r[k + 1], "S1_{k} cols on x's grid");
            let s1a = cross_wire(
                &s1,
                CrossDir::BtoA,
                TRIPLET as i64,
                (&sf.rdz[k], &r[k + 1]),
                (&sf.rm[k], &sf.rd[k + 1]),
                space,
                out,
                (&r[k], &r[k + 1]),
                &mut spin,
            );
            assert_eq!(
                (s1a.space, &s1a.row_ranks, &s1a.col_ranks),
                (out, &sf.rm[k], &sf.rd[k + 1]),
                "S1_{k} crossed lands in component shape"
            );
            let s3a_of = |chain: &[Su2Sketch<f64>], name: &str| {
                let s3 = mul_sketch_left(&pz[k], &zsrc.right_mul(k, &chain[k + 1]));
                assert_eq!(s3.row_ranks, r[k], "{name} S3_{k} rows on x's grid");
                assert_eq!(s3.col_ranks, sf.rdz[k + 1], "{name} S3_{k} cols on rdz");
                let s3a = regrid_core(&s3, out, &sf.rm[k], &sf.rd[k + 1]);
                assert_eq!(
                    (s3a.space, &s3a.row_ranks, &s3a.col_ranks),
                    (out, &sf.rm[k], &sf.rd[k + 1]),
                    "{name} S3_{k} regridded lands in component shape"
                );
                s3a
            };
            let s3a = s3a_of(&evr, "telescoped");
            let s3a_oracle = s3a_of(&evr_oracle, "oracle");
            let mut acc = Su2Core::<f64>::zeros(out, k, &sf.rm[k], &sf.rd[k + 1]);
            acc.axpy(1.0, &s1a);
            acc.axpy(1.0, &s3a);
            let mut acc_oracle = Su2Core::<f64>::zeros(out, k, &sf.rm[k], &sf.rd[k + 1]);
            acc_oracle.axpy(1.0, &s1a);
            acc_oracle.axpy(1.0, &s3a_oracle);
            if k == 2 {
                assert!(acc.check_invariants());
                assert!(acc_oracle.check_invariants(), "oracle");
                println!(
                    "csu2b B6 real: ‖S1_2 + S3_2‖_center = {:e} (oracle chain {:e})",
                    acc.norm_center(),
                    acc_oracle.norm_center()
                );
            }
        }
    }
}
