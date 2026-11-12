// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::core::deflate::ColumnDeflation;
use crate::core::deflate::oracle::{asym, dense_projector};
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::tangent::{
    TtTangent, min_schmidt, tangent_basis, tangent_basis_shifted, tangent_dim_grids,
};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::QnSpace;
use crate::solvers::riemannian::tangent_bases;
use crate::test_util::{
    core_left_orthonormality_dev, core_right_orthonormality_dev, pg4_curved_state, pooled_schmidt,
    randomized_state, relabel_sensitive_floor,
};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const D: usize = 4;
const NUP: usize = 2;
const NDN: usize = 2;

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(3), Irrep::new(2), Irrep::new(1), Irrep::new(0)],
    )
}

type Fixture = (
    QnSpace,
    TtVector<f64>,
    TtVector<f64>,
    TtVector<f64>,
    Vec<BondGauge<f64>>,
);

fn fx(seed: u64) -> Fixture {
    let x = pg4_curved_state(seed);
    let space = x.space;
    assert_eq!(
        (space.d, space.nup, space.ndn, space.target),
        (D, NUP, NDN, Irrep::IDENTITY),
        "the hoisted pg4c draw must be this file's sector"
    );
    assert!(
        space.pg == fx_pg(),
        "the hoisted pg4c draw must be this file's table"
    );
    let (bl, br) = tangent_bases(&x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    (space, x, bl, br, gauges)
}

fn normal_z(bl: &TtVector<f64>, br: &TtVector<f64>, space: QnSpace, seed: u64) -> TtVector<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let w: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    let z = &w - &TtTangent::project(bl, br, &w).to_vector();
    let zn = z.norm_lr(false);
    assert!(
        zn > 1e-6,
        "normal_z: the normal part is degenerate ({zn:e})"
    );
    let tang = TtTangent::project(bl, br, &z).norm();
    assert!(
        tang <= 1e-6 * zn,
        "normal_z: not normal (||P(z)||/||z|| = {:.3e})",
        tang / zn
    );
    z
}

fn shifted_dir(
    bl: &TtVector<f64>,
    br: &TtVector<f64>,
    space: QnSpace,
    phi: Irrep,
    seed: u64,
) -> TtTangent<f64> {
    let space_out = space.with_target(space.target.fuse(phi));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    for _ in 0..16 {
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let mut xi = TtTangent::project_shifted(bl, br, phi, &y);
        let n = xi.norm();
        if n > 1e-8 {
            xi.scale_mut(1.0 / n);
            return xi;
        }
    }
    panic!("shifted_dir: 16 draws were all annihilated by P^phi at phi = {phi:?}, seed {seed:#x}");
}

fn live(space: &QnSpace, bl: &TtVector<f64>) -> Vec<Irrep> {
    (0..space.pg.order())
        .map(Irrep::new)
        .filter(|&phi| tangent_dim_grids(space, &bl.ranks, phi) > 0)
        .collect()
}

fn shifting_phi(space: &QnSpace, bl: &TtVector<f64>) -> Irrep {
    let l = live(space, bl);
    l.iter()
        .copied()
        .find(|&p| p != Irrep::IDENTITY)
        .unwrap_or_else(|| panic!("no live shifted channel (live = {l:?})"))
}

#[test]
fn twb_identity_path_is_unchanged() {
    let (space, _x, bl, br, gauges) = fx(0x7C01);
    let z = normal_z(&bl, &br, space, 0x7C01_0001);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7C01_0002);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    let xi_p = TtTangent::project(&bl, &br, &dir);
    let xi_s = TtTangent::project_shifted(&bl, &br, Irrep::IDENTITY, &dir);
    assert_eq!(xi_p.phi(), Irrep::IDENTITY);
    assert_eq!(xi_s.phi(), Irrep::IDENTITY);
    assert!(
        xi_p.norm() > 1e-8,
        "anti-vacuity: the IDENTITY probe direction is zero"
    );

    let b_p = weingarten_assemble_cached(&bl, &br, &z, &xi_p, &gauges, Terms::all());
    let b_s = weingarten_assemble_cached(&bl, &br, &z, &xi_s, &gauges, Terms::all());
    println!(
        "twb identity: sigma_min = {:e}, ||B xi|| = {:e}, ||z|| = {:e}, \
         |project - project_shifted(IDENTITY)| = {:e}",
        min_schmidt(&bl, &br),
        b_p.norm(),
        z.norm_lr(false),
        b_p.sub(&b_s).norm()
    );
    assert!(
        b_p.norm() > 1e-8,
        "ANTI-VACUITY: B is zero on this direction, so both pins below compare zero to zero"
    );
    assert_eq!(
        b_p.sub(&b_s).norm(),
        0.0,
        "project and project_shifted(IDENTITY) must be INDISTINGUISHABLE to the assembly \
         (||B xi|| = {:e}) — at phi = IDENTITY no relabel may execute",
        b_p.norm()
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
        "weingarten_assemble is bond_gauges + weingarten_assemble_cached by definition; with the \
         same floor the two must be bit-identical"
    );
}

#[test]
fn twb_shifted_assembly_is_linear_in_z() {
    let (space, _x, bl, br, gauges) = fx(0x7C02);
    let phi = shifting_phi(&space, &bl);
    let xi = shifted_dir(&bl, &br, space, phi, 0x7C02_0001);
    let z1 = normal_z(&bl, &br, space, 0x7C02_0002);
    let z2 = normal_z(&bl, &br, space, 0x7C02_0003);
    let zs = &z1 + &z2;

    let b = |z: &TtVector<f64>| weingarten_assemble_cached(&bl, &br, z, &xi, &gauges, Terms::all());
    let (b1, b2, bs) = (b(&z1), b(&z2), b(&zs));
    let scale = b1.norm().max(b2.norm());
    println!(
        "twb linearity: phi = {phi:?}, dim = {}, ||B(z1) xi|| = {:e}, ||B(z2) xi|| = {:e}, \
         ||B(z1+z2) xi|| = {:e}, |additivity| = {:e}",
        tangent_dim_grids(&space, &bl.ranks, phi),
        b1.norm(),
        b2.norm(),
        bs.norm(),
        bs.sub(&b1.add(&b2)).norm()
    );
    assert!(
        scale > 1e-8,
        "ANTI-VACUITY: B^phi is identically zero on this direction, and the zero map is linear. \
         This is the documented silent failure mode — pZ/qZ seeded from the SHIFTED frames"
    );
    assert!(
        b1.sub(&b2).norm() > 1e-3 * scale,
        "ANTI-VACUITY: the two normal Z's give the same image, so additivity is trivial here"
    );
    assert!(
        bs.sub(&b1.add(&b2)).norm() <= 1e-12 * scale,
        "B(z1 + z2) != B(z1) + B(z2) at phi = {phi:?} ({:e} vs scale {scale:e})",
        bs.sub(&b1.add(&b2)).norm()
    );

    for alpha in [-2.5f64, 0.5] {
        let za = &z1 * alpha;
        let ba = b(&za);
        let want = b1.scale(alpha);
        println!(
            "twb linearity: alpha = {alpha}, |B(alpha z) - alpha B(z)| = {:e}",
            ba.sub(&want).norm()
        );
        assert!(
            ba.sub(&want).norm() <= 1e-12 * (alpha.abs() * b1.norm()).max(1.0),
            "B(alpha*z) != alpha*B(z) at alpha = {alpha}"
        );
    }
}

#[test]
fn twb_s2_is_identically_zero_at_every_phi() {
    let (space, _x, bl, br, gauges) = fx(0x7C03);
    let z = normal_z(&bl, &br, space, 0x7C03_0001);
    let no_s2 = Terms {
        diag_s1: true,
        diag_s2: false,
        diag_s3: true,
    };
    let mut checked = 0usize;
    for phi in live(&space, &bl) {
        let xi = shifted_dir(&bl, &br, space, phi, 0x7C03_0100 + u64::from(phi.index()));
        let full = weingarten_assemble_cached(&bl, &br, &z, &xi, &gauges, Terms::all());
        let ablated = weingarten_assemble_cached(&bl, &br, &z, &xi, &gauges, no_s2);
        let d = full.sub(&ablated).norm();
        println!(
            "twb S2 phi={}: ||B xi|| = {:e}, |full - no_s2| = {d:e}",
            phi.index(),
            full.norm()
        );
        assert!(
            full.norm() > 1e-8,
            "phi {:?}: ANTI-VACUITY, B is zero so the ablation compares zero to zero",
            phi
        );
        assert!(
            d <= 1e-12 * full.norm(),
            "phi {:?}: turning S2 off moved the answer by {d:e} — S2's only Z-dependent factor \
             is the overlap of a NORMAL Z with a bond variation, which is identically zero at \
             every phi",
            phi
        );
        checked += 1;
    }
    assert!(
        checked >= 2,
        "anti-vacuity: the ablation ran on {checked} channels; it must cover IDENTITY and at \
         least one shifted channel for the phi-independence claim to have content"
    );
}

#[test]
fn twb_empty_channel_returns_a_well_shaped_zero() {
    let (space, _x, bl, br, gauges) = fx(0x7C04);
    let z = normal_z(&bl, &br, space, 0x7C04_0001);
    let empty: Vec<Irrep> = (0..space.pg.order())
        .map(Irrep::new)
        .filter(|&phi| tangent_dim_grids(&space, &bl.ranks, phi) == 0)
        .collect();
    println!(
        "twb empty: empty = {empty:?}, live = {:?}",
        live(&space, &bl)
    );
    assert!(
        !empty.is_empty(),
        "fixture premise: this grid must carry an empty channel (measured phi = 2, 3)"
    );

    for phi in empty {
        let space_out = space.with_target(space.target.fuse(phi));
        assert!(
            tangent_basis_shifted(&bl, &br, phi).is_empty(),
            "phi {phi:?}: an empty channel returned a basis"
        );
        let y: TtVector<f64> = crate::vector::tt_zeros(space_out);
        let xi = TtTangent::project_shifted(&bl, &br, phi, &y);
        assert_eq!(xi.phi(), phi);

        let out = weingarten_assemble_cached(&bl, &br, &z, &xi, &gauges, Terms::all());
        assert_eq!(
            out.phi(),
            phi,
            "phi {phi:?}: the flux was not carried through"
        );
        assert_eq!(out.norm(), 0.0, "phi {phi:?}: an empty channel must give 0");
        for k in 0..D {
            let c = out.component(k);
            c.check_invariants();
            assert_eq!(
                c.space, space_out,
                "phi {phi:?}: component {k} is not stamped space_out"
            );
            for s in space.bond_sectors(k) {
                assert_eq!(
                    c.row_ranks[s], bl.ranks[k][s],
                    "phi {phi:?}: component {k} row rank at {s:?}"
                );
            }
            for s in space.bond_sectors(k + 1) {
                assert_eq!(
                    c.col_ranks[s],
                    bl.ranks[k + 1][Sector::with_irrep(s.nup, s.ndn, s.irrep.fuse(phi))],
                    "phi {phi:?}: component {k} col rank at {s:?} must be r[k+1][s + phi] — a \
                     zero-COLUMN component (rather than a zero-VALUED one) is what breaks every \
                     shape-reading consumer downstream"
                );
            }
        }
    }
}

#[test]
fn twb_d1_returns_zero_at_the_declared_shapes() {
    let pg = PointGroup::new(2, &[Irrep::new(1)]);
    let space = QnSpace::with_point_group(1, 1, 0, pg, Irrep::new(1));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7C05);
    let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    x.scale_mut(1.0 / x.norm());
    let (bl, br) = tangent_bases(&x);
    let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
    assert!(gauges.is_empty(), "d = 1 has no interior bond gauges");

    for p in 0..space.pg.order() {
        let phi = Irrep::new(p);
        if phi == Irrep::IDENTITY {
            continue;
        }
        assert_eq!(
            tangent_dim_grids(&space, &bl.ranks, phi),
            0,
            "at d = 1 every phi != IDENTITY must be structurally empty; phi = {p} is not, so the \
             shifted d == 1 early return IS reachable and needs its own arm"
        );
    }

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7C05_0001);
    let w: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    let z = &w - &TtTangent::project(&bl, &br, &w).to_vector();
    assert!(
        z.norm_lr(false) <= 1e-12 * w.norm_lr(false).max(1.0),
        "at d = 1 every sector is one-dimensional, so the fixed-rank manifold IS the sector and \
         the normal space is {{0}} — got ||z|| = {:e}",
        z.norm_lr(false)
    );
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7C05_0002);
    let dir: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    let xi = TtTangent::project(&bl, &br, &dir);
    let out = weingarten_assemble_cached(&bl, &br, &z, &xi, &gauges, Terms::all());
    println!(
        "twb d=1: ||z|| = {:e}, ||xi|| = {:e}, ||B xi|| = {:e}",
        z.norm_lr(false),
        xi.norm(),
        out.norm()
    );
    assert_eq!(out.phi(), Irrep::IDENTITY);
    assert_eq!(
        out.norm(),
        0.0,
        "d = 1 has no interior bond, so B is the zero map"
    );
    let c = out.component(0);
    c.check_invariants();
    assert_eq!(c.space, space);
    for s in space.bond_sectors(0) {
        assert_eq!(c.row_ranks[s], bl.ranks[0][s], "d = 1: row rank at {s:?}");
    }
    for s in space.bond_sectors(1) {
        assert_eq!(c.col_ranks[s], bl.ranks[1][s], "d = 1: col rank at {s:?}");
    }
}

#[allow(clippy::type_complexity)]
fn relabel_fixture() -> (u64, Fixture, Irrep, f64, Vec<(usize, Sector)>) {
    for seed in [0x7C01u64, 0x7C02, 0x7C03, 0x7C04, 0x7C05] {
        let fixture = fx(seed);
        let (space, _x, bl, br, gauges) = &fixture;
        let phi = shifting_phi(space, bl);
        if let Some((f, blocks)) = relabel_sensitive_floor(bl, br, gauges, phi) {
            return (seed, fixture, phi, f, blocks);
        }
    }
    panic!(
        "fixture premise: no pg4c seed of the five carries a relabel-sensitive cut block on its \
         curved channel — every K1 arm below would be unable to tell t + phi from t"
    )
}

fn removed_by_grids(
    template: &TtTangent<f64>,
    bl: &TtVector<f64>,
    gauges: &[BondGauge<f64>],
    phi: Irrep,
    f: f64,
) -> usize {
    let space = bl.space;
    let mut removed = 0usize;
    for g in gauges {
        let b = g.k;
        for (t, sv) in g.svals() {
            let n_cut = sv.iter().filter(|&&s| s < f).count();
            if n_cut == 0 {
                continue;
            }
            let s = Sector::with_irrep(t.nup, t.ndn, t.irrep.fuse(phi));
            if !space.contains(b, s) {
                continue;
            }
            let rows = template.component(b - 1).vertical_unfolding(s).nrows();
            removed += rows.saturating_sub(bl.ranks[b][s]) * n_cut;
        }
    }
    removed
}

fn applied(p: &ColumnDeflation<f64>, xi: &TtTangent<f64>) -> TtTangent<f64> {
    let mut v = xi.clone();
    p.apply(&mut v);
    v
}

fn coeff_bits(v: &TtTangent<f64>) -> Vec<u64> {
    (0..v.num_coeffs()).map(|i| v.coeff(i).to_bits()).collect()
}

#[test]
fn cr_shifted_projector_is_orthogonal_counted_and_gauge_preserving() {
    let (seed, (space, _x, bl, br, gauges), phi, f, blocks) = relabel_fixture();
    let basis = tangent_basis_shifted(&bl, &br, phi);
    let n = basis.len();
    println!(
        "K1 projector: seed {seed:#x}, phi {phi:?}, n = {n}, floor {f:.6e}, premise blocks {blocks:?}, \
         sigma_min {:e}",
        min_schmidt(&bl, &br)
    );
    assert_eq!(
        n,
        tangent_dim_grids(&space, &bl.ranks, phi),
        "premise: the shifted basis spans T^phi, or a trace is not a rank"
    );
    let tik = |e: f64| move |s: f64| s * s / (s * s + e * e);
    for (label, p, hard) in [
        ("Hard", ColumnDeflation::from_gauges(&gauges, f), true),
        (
            "Tikhonov",
            ColumnDeflation::from_gauges_weighted(&gauges, tik(f)),
            false,
        ),
    ] {
        let m = dense_projector(&p, &basis);
        let sym = asym(&m, n);
        let ev = dense_self_adjoint_smallest::<f64>(&m, n, 0).0;
        let tr: f64 = (0..n).map(|i| m[i * n + i]).sum();
        let mut idem = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let mm: f64 = (0..n).map(|q| m[i * n + q] * m[q * n + j]).sum();
                idem = idem.max((mm - m[i * n + j]).abs());
            }
        }
        let (mut leak, mut moved) = (0.0f64, 0.0f64);
        for b in &basis {
            let v = applied(&p, b);
            let back = TtTangent::project_shifted(&bl, &br, phi, &v.to_vector());
            leak = leak.max(back.sub(&v).norm());
            moved = moved.max(v.sub(b).norm());
        }
        println!(
            "K1 projector [{label}]: trace {tr:.12}, sym {sym:.1e}, idem {idem:.1e}, spectrum \
             [{:.3e}, {:.3e}], gauge leak {leak:.1e}, max moved {moved:.3e}",
            ev[0],
            ev[n - 1]
        );
        assert!(
            sym <= 1e-14,
            "[{label}] the shifted projector is not symmetric ({sym:e})"
        );
        assert!(
            leak <= 1e-12,
            "[{label}] P xi left T^phi's gauge-fixed subspace ({leak:e}) — the operator acts on rows \
             or on the wrong label"
        );
        assert!(
            moved >= 1e-3,
            "[{label}] non-vacuity: the operator moved no basis vector ({moved:e})"
        );
        if hard {
            let removed = p.removed_dim(&basis[0]);
            let want = removed_by_grids(&basis[0], &bl, &gauges, phi, f);
            println!("K1 projector [Hard]: removed_dim {removed}, grid count {want}");
            assert!(idem <= 1e-12, "[Hard] not idempotent ({idem:e})");
            assert!(
                want > 0,
                "non-vacuity: the floor removes nothing from T^phi"
            );
            assert_eq!(
                removed, want,
                "removed_dim on a SHIFTED template must subtract the gauge rank r[k+1][t + phi] at \
                 the relabelled label (the P1 formula reads the column count r[k+1][t])"
            );
            assert!(
                (tr - (n - removed) as f64).abs() <= 1e-9,
                "tr P = {tr} but n - removed_dim = {}",
                n - removed
            );
        } else {
            assert!(
                ev[0] >= -1e-12 && ev[n - 1] <= 1.0 + 1e-12,
                "[Tikhonov] spectrum [{}, {}] not in [0, 1]",
                ev[0],
                ev[n - 1]
            );
        }
    }

    let smallest = pooled_schmidt(&gauges)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    for (label, id) in [
        (
            "Hard(smallest / 2)",
            ColumnDeflation::from_gauges(&gauges, 0.5 * smallest),
        ),
        (
            "Tikhonov(1e-12)",
            ColumnDeflation::from_gauges_weighted(&gauges, tik(1e-12)),
        ),
    ] {
        assert!(id.is_identity(), "{label}: must touch nothing");
        for b in &basis {
            assert_eq!(
                coeff_bits(&applied(&id, b)),
                coeff_bits(b),
                "{label}: the identity operator must leave a shifted tangent bitwise unchanged"
            );
        }
    }

    let p = ColumnDeflation::from_gauges(&gauges, f);
    let basis0 = tangent_basis(&bl, &br, false);
    for b in basis0.iter().take(4) {
        let mut u = b.clone();
        p.apply_unrelabeled(&mut u);
        assert_eq!(
            coeff_bits(&applied(&p, b)),
            coeff_bits(&u),
            "phi = IDENTITY: apply must be the unrelabelled (P1) operator bit for bit"
        );
    }
    assert_eq!(
        p.removed_dim(&basis0[0]),
        removed_by_grids(&basis0[0], &bl, &gauges, Irrep::IDENTITY, f),
        "phi = IDENTITY: removed_dim is the P1 grid count"
    );
}

#[test]
fn cr_untyped_projector_is_irrep_block_diagonal_with_the_relabelled_block() {
    let (seed, (space, _x, bl, br, gauges), phi, f, blocks) = relabel_fixture();
    let bl1 = bl.forget_point_group();
    let br1 = br.forget_point_group();
    let (dl, nl) = core_left_orthonormality_dev(&bl1);
    let (dr, nr) = core_right_orthonormality_dev(&br1);
    assert!(
        nl > 0 && nr > 0,
        "premise: the orthonormality checks saw no block"
    );
    assert!(
        dl <= 1e-14 && dr <= 1e-14,
        "premise: the forgotten frames are exactly orthonormal (left {dl:e}, right {dr:e})"
    );
    let gauges1 = bond_gauges(&bl1, &br1, FloorPolicy::default_hard());
    for (g1, g) in gauges1.iter().zip(&gauges) {
        assert_eq!(g1.k, g.k);
        for (s1, v1) in g1.svals() {
            let mut got = v1.clone();
            let mut want: Vec<f64> = g
                .svals()
                .iter()
                .filter(|(t, _)| t.nup == s1.nup && t.ndn == s1.ndn)
                .flat_map(|(_, v)| v.iter().copied())
                .collect();
            got.sort_by(f64::total_cmp);
            want.sort_by(f64::total_cmp);
            assert_eq!(
                got.len(),
                want.len(),
                "bond {} sector {s1:?}: Schmidt count",
                g.k
            );
            for (a, b) in got.iter().zip(&want) {
                assert!(
                    (a - b).abs() <= 1e-13 * b.max(1.0),
                    "premise: bond {} sector {s1:?}: untyped Schmidt value {a} vs typed {b} — R_b \
                     is not block-diagonal over irreps",
                    g.k
                );
            }
        }
    }
    let embed = |xi: &TtTangent<f64>, what: &str| -> TtTangent<f64> {
        let y = xi.to_vector().forget_point_group();
        let e = TtTangent::project(&bl1, &br1, &y);
        let (nx, ne) = (xi.norm(), e.norm());
        assert!(
            (ne - nx).abs() <= 1e-12 * nx.max(1.0),
            "{what}: iota is not isometric across the untyped embedding ({ne} vs {nx})"
        );
        let back = &e.to_vector() - &y;
        let defect = back.norm_lr(false);
        assert!(
            defect <= 1e-10 * y.norm_lr(false).max(1.0),
            "{what}: the embedded vector is not ALREADY an untyped tangent (moved by {defect:e})"
        );
        e
    };
    let tik = |e: f64| move |s: f64| s * s / (s * s + e * e);
    let xis: Vec<TtTangent<f64>> = (0..4)
        .map(|j| shifted_dir(&bl, &br, space, phi, 0x7C06_0000 + j))
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7C06_1000);
    let etas: Vec<TtTangent<f64>> = (0..3)
        .map(|_| {
            let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
            let mut eta = TtTangent::project(&bl, &br, &dir);
            eta.scale_mut(1.0 / eta.norm());
            eta
        })
        .collect();
    println!(
        "K1 untyped block: seed {seed:#x}, phi {phi:?}, floor {f:.6e}, premise blocks {blocks:?}"
    );
    for (label, p, p1) in [
        (
            "Hard",
            ColumnDeflation::from_gauges(&gauges, f),
            ColumnDeflation::from_gauges(&gauges1, f),
        ),
        (
            "Tikhonov",
            ColumnDeflation::from_gauges_weighted(&gauges, tik(f)),
            ColumnDeflation::from_gauges_weighted(&gauges1, tik(f)),
        ),
    ] {
        let (mut worst, mut moved, mut cross) = (0.0f64, 0.0f64, 0.0f64);
        for (j, xi) in xis.iter().enumerate() {
            let e = embed(xi, &format!("[{label}] xi {j}"));
            let lhs = embed(&applied(&p, xi), &format!("[{label}] P xi {j}"));
            let rhs = applied(&p1, &e);
            worst = worst.max(lhs.sub(&rhs).norm() / xi.norm());
            moved = moved.max(rhs.sub(&e).norm() / xi.norm());
            for (i, eta) in etas.iter().enumerate() {
                let e0 = embed(eta, &format!("[{label}] eta {i}"));
                cross = cross.max(e0.dot(&rhs).abs());
            }
        }
        println!(
            "K1 untyped block [{label}]: worst |embed(P xi) - P1 embed(xi)| {worst:.2e}, max moved \
             {moved:.3e}, max cross-block pairing {cross:.2e}"
        );
        assert!(
            moved >= 1e-3,
            "[{label}] non-vacuity: the untyped operator moved no embedded direction ({moved:e})"
        );
        assert!(
            worst <= 1e-12,
            "[{label}] the shifted column operator is not the phi block of the untyped one \
             ({worst:e}): it acts on the wrong label (t instead of t + phi) or the wrong side"
        );
        assert!(
            cross <= 1e-13,
            "[{label}] P1 mixed the phi = 0 and phi = {phi:?} blocks ({cross:e})"
        );
    }
    let p = ColumnDeflation::from_gauges(&gauges, f);
    let p1 = ColumnDeflation::from_gauges(&gauges1, f);
    let mut miss = 0.0f64;
    for (j, xi) in xis.iter().enumerate() {
        let mut v = xi.clone();
        p.apply_unrelabeled(&mut v);
        let lhs = embed(&v, &format!("unrelabelled P xi {j}"));
        let rhs = applied(&p1, &embed(xi, &format!("anti-oracle xi {j}")));
        miss = miss.max(lhs.sub(&rhs).norm() / xi.norm());
    }
    println!("K1 anti-oracle: the unrelabelled operator misses by {miss:.3e}");
    assert!(
        miss >= 1e-3,
        "premise: the unrelabelled operator agrees with the untyped block ({miss:e}), so this \
         fixture cannot tell t + phi from t — the premise blocks {blocks:?} carry no weight"
    );
}

fn col_norm(c: &Mat<f64>, v: &[f64]) -> f64 {
    (0..c.nrows())
        .map(|i| {
            (0..c.ncols())
                .map(|j| c[(i, j)] * v[j])
                .sum::<f64>()
                .powi(2)
        })
        .sum::<f64>()
        .sqrt()
}

#[test]
fn cr_shifted_projector_reads_the_right_schmidt_vectors_not_the_left() {
    let pg6 = PointGroup::new(
        2,
        &[
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
            Irrep::new(2),
            Irrep::new(1),
            Irrep::new(0),
        ],
    );
    let space = QnSpace::with_point_group(6, NUP, NDN, pg6, Irrep::IDENTITY);
    let mut found = None;
    'search: for seed in [0x7C11u64, 0x7C12, 0x7C13, 0x7C14] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.scale_mut(1.0 / x.norm());
        let (bl, br) = tangent_bases(&x);
        let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        for phi in live(&space, &bl) {
            if phi == Irrep::IDENTITY {
                continue;
            }
            let probe = shifted_dir(&bl, &br, space, phi, 0x7C1F_0000);
            for (gi, g) in gauges.iter().enumerate() {
                for (t, sv) in g.svals() {
                    let n = sv.len();
                    if n < 2 || sv[n - 1] < 1e-6 || sv[n - 2] <= sv[n - 1] * 1.01 {
                        continue;
                    }
                    let s = Sector::with_irrep(t.nup, t.ndn, t.irrep.fuse(phi));
                    if !space.contains(g.k, s) {
                        continue;
                    }
                    let c = probe.component(g.k - 1).vertical_unfolding(s);
                    if c.ncols() == n && c.nrows() > 0 && c.norm_l2() > 1e-6 {
                        found = Some((seed, bl.clone(), br.clone(), gauges.clone(), phi, gi, *t));
                        break 'search;
                    }
                }
            }
        }
    }
    let (seed, bl, br, mut gauges, phi, gi, t) =
        found.expect("fixture premise: a rank->=2 cut block with shifted room at some seed");
    let g = &mut gauges[gi];
    let b = g.k;
    let (_, sv, _) = svd_uvt_svals(g.r.block(t).expect("the planted block"));
    let n = sv.len();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x05EC_70C5);
    let mut gauss = |_: usize, _: usize| rand::RngExt::sample(&mut rng, rand_distr::StandardNormal);
    let g1 = Mat::<f64>::from_fn(n, n, &mut gauss);
    let g2 = Mat::<f64>::from_fn(n, n, &mut gauss);
    let (wq, _, _) = svd_uvt_svals(g1.as_ref());
    let (_, _, vqt) = svd_uvt_svals(g2.as_ref());
    let m = Mat::<f64>::from_fn(n, n, |i, j| {
        (0..n)
            .map(|q| wq[(i, q)] * sv[q] * vqt[(q, j)])
            .sum::<f64>()
    });
    g.r.block_mut(t)
        .expect("the planted block")
        .copy_from(m.as_ref());
    let w_n: Vec<f64> = (0..n).map(|i| wq[(i, n - 1)]).collect();
    let v_n: Vec<f64> = (0..n).map(|j| vqt[(n - 1, j)]).collect();
    let overlap = w_n.iter().zip(&v_n).map(|(a, c)| a * c).sum::<f64>().abs();
    let proj_gap = (2.0 * (1.0 - overlap * overlap)).max(0.0).sqrt();
    let f = (sv[n - 1] * sv[n - 2]).sqrt();
    let p = ColumnDeflation::from_gauges(&gauges, f);
    let s = Sector::with_irrep(t.nup, t.ndn, t.irrep.fuse(phi));
    let (mut worst_v, mut least_w) = (0.0f64, f64::INFINITY);
    for j in 0..3u64 {
        let xi = shifted_dir(&bl, &br, space, phi, 0x7C1E_0000 + j);
        let c0 = xi.component(b - 1).vertical_unfolding(s);
        let scale = c0.norm_l2();
        assert!(
            col_norm(&c0, &v_n) >= 1e-3 * scale,
            "premise: shifted seed {j} has weight along v_n"
        );
        let y = applied(&p, &xi);
        let c = y.component(b - 1).vertical_unfolding(s);
        worst_v = worst_v.max(col_norm(&c, &v_n) / scale);
        least_w = least_w.min(col_norm(&c, &w_n) / scale);
    }
    println!(
        "K1 T-b shifted: seed {seed:#x}, phi {phi:?}, bond {b}, sector {t:?} -> label {s:?}, n = \
         {n}, |<w_n, v_n>| = {overlap:.3}, ||v v^T - w w^T|| = {proj_gap:.3}, after apply max \
         ||c v_n||/||c|| = {worst_v:.2e}, min ||c w_n||/||c|| = {least_w:.2e}"
    );
    assert!(
        proj_gap >= 1e-3 && overlap <= 0.9,
        "premise: W and V differ (|<w_n, v_n>| = {overlap}, gap {proj_gap})"
    );
    assert!(
        worst_v <= 1e-13,
        "the shifted slot keeps weight along the RIGHT Schmidt vector ({worst_v:e})"
    );
    assert!(
        least_w >= 1e-3,
        "non-vacuity: the shifted slot lost its weight along the LEFT vector too ({least_w:e}) — \
         a W-built projector would look like this"
    );
}
