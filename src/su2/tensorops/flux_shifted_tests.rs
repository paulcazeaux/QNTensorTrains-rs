// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::su2::core::Family;
use crate::su2::coreops::{adjoint_core_times_core, rproject};
use crate::su2::flux::{dress_train, flux_ranks};
use crate::su2::qn::max_ranks;
use crate::su2::riemannian::tangent_bases;
use crate::su2::vector::tt_randn;
use crate::test_util::seeded;

const D: usize = 4;
const NEL: usize = 4;

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
    let grids = capped_grids(&space, cap);
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<f64> = tt_randn(space, &grids, &mut rng);
    let n = x.norm_lr(false);
    assert!(n > 1e-8, "fixture: the random state must be nonzero");
    x.scale_mut(1.0 / n);
    assert!(x.check(), "fixture: the reference must be a legal train");
    let (bl, br) = tangent_bases(&x);
    assert_eq!(
        bl.ranks, br.ranks,
        "fixture: the two frames must factorize the SAME point"
    );
    (space, bl, br)
}

fn tri(a: i64, b: i64, c: i64) -> bool {
    (a - b).abs() <= c && c <= a + b && (a + b + c).rem_euclid(2) == 0
}

fn window_contract(two_s_y: i64, twok: i64) -> Vec<i64> {
    let mut lo = two_s_y - twok;
    while lo < 0 {
        lo += 2;
    }
    let mut out = Vec::new();
    let mut j = lo;
    while j <= two_s_y + twok {
        out.push(j);
        j += 2;
    }
    out
}

fn fan(
    r: &Su2RankGrid,
    two_kappa: i64,
    space: Su2Space,
    out_space: Su2Space,
    bond: usize,
    gate: WireGate,
) -> Su2RankGrid {
    let mut out = Su2RankGrid::zeros(&out_space);
    for t in out_space.bond_sectors(bond) {
        let mut acc = 0usize;
        for two_j in window_contract(t.two_s as i64, two_kappa) {
            if matches!(gate, WireGate::Triangle) && !tri(t.two_s as i64, two_j, two_kappa) {
                continue;
            }
            let u = Su2Sector::with_irrep(t.n, two_j as usize, t.irrep);
            if space.contains(bond, u) {
                acc += r[u];
            }
        }
        out[t] = acc;
    }
    out
}

fn fan_in(space: Su2Space, r: &Su2RankGrid, bond: usize, t: Su2Sector) -> usize {
    [t.two_s as i64 - 2, t.two_s as i64, t.two_s as i64 + 2]
        .into_iter()
        .filter(|&u| {
            u >= 0
                && tri(t.two_s as i64, u, 2)
                && space.contains(bond, Su2Sector::with_irrep(t.n, u as usize, t.irrep))
                && r[Su2Sector::with_irrep(t.n, u as usize, t.irrep)] > 0
        })
        .count()
}

fn dim_over_fan(
    space: Su2Space,
    out_space: Su2Space,
    r: &[Su2RankGrid],
    two_kappa: i64,
    gate: WireGate,
) -> usize {
    let d = space.d;
    let rows: Vec<Su2RankGrid> = (0..=d)
        .map(|b| fan(&r[b], 0, space, out_space, b, gate))
        .collect();
    let cols: Vec<Su2RankGrid> = (0..=d)
        .map(|b| fan(&r[b], two_kappa, space, out_space, b, gate))
        .collect();
    let mut raw = 0usize;
    let mut gauge = 0usize;
    for k in 0..d {
        let g = out_space.pg.irrep_of(k);
        for l in out_space.bond_sectors(k) {
            for f in Family::ALL {
                let Some(t) = f.apply(l, g) else { continue };
                if out_space.contains(k + 1, t) {
                    raw += rows[k][l] * cols[k + 1][t];
                }
            }
        }
        if k < d - 1 {
            for t in out_space.bond_sectors(k + 1) {
                gauge += rows[k + 1][t] * cols[k + 1][t];
            }
        }
    }
    raw - gauge
}

#[test]
fn csu2_wire_gate_triangle_is_the_window_intersected_with_the_triangle() {
    let mut window_only = 0usize;
    for two_s_y in 0..=8i64 {
        for twok in 0..=6i64 {
            let w = jl_window_gated(two_s_y, twok, WireGate::Window);
            assert_eq!(
                w,
                window_contract(two_s_y, twok),
                "WireGate::Window must reproduce the documented window (parity-preserving clamp) \
                 at (2s_y, 2k) = ({two_s_y}, {twok}) — every Hamiltonian call site passes Window \
                 and its 2j enumeration must not move"
            );
            let t = jl_window_gated(two_s_y, twok, WireGate::Triangle);
            let want: Vec<i64> = w
                .iter()
                .copied()
                .filter(|&j| tri(two_s_y, j, twok))
                .collect();
            assert_eq!(
                t, want,
                "WireGate::Triangle must be the window intersected with tri(2s_y, 2j, 2k) at \
                 (2s_y, 2k) = ({two_s_y}, {twok}): got {t:?}, want {want:?}"
            );
            assert!(
                t.len() <= w.len(),
                "the triangle can only ever DROP channels ({two_s_y}, {twok})"
            );
            window_only += w.len() - t.len();
        }
    }
    assert!(
        window_only > 0,
        "ANTI-VACUITY: the two gates never differed over the whole sweep, so this arm would pass \
         with Triangle aliased to Window"
    );
    assert_eq!(
        jl_window_gated(0, 2, WireGate::Window),
        vec![0, 2],
        "the window at (2s_y, 2k) = (0, 2) is {{0, 2}}"
    );
    assert_eq!(
        jl_window_gated(0, 2, WireGate::Triangle),
        vec![2],
        "tri(0, 2, 0) is FALSE, so the triangle gate must drop 2j = 0 — the channel whose \
         recouple factor is identically zero and which a tangent parametrization cannot absorb"
    );
    println!("csu2 gate: {window_only} window-only channels over the (2s_y, 2k) sweep");
}

#[test]
fn csu2_window_gate_is_the_shipped_operator_paths_default() {
    let space = fx_space();
    let mut rng = seeded(0xC0_11);
    let mut x: Su2TtVector<f64> = tt_randn(space, &capped_grids(&space, 0), &mut rng);
    x.scale_mut(1.0 / x.norm_lr(false));

    let y = s_i(&x, 1, 2);
    assert!(y.check(), "s_i must return a legal train");
    assert_eq!(y.space.two_s, 2, "s_i(-, -, 2) must raise 2S to 2");
    let ny = y.norm_lr(false);
    assert!(
        ny > 1e-6,
        "ANTI-VACUITY: S_1 must not annihilate a random singlet (norm {ny:e})"
    );

    for b in 2..=D {
        let want = wire_ranks(
            &x.ranks[b],
            2,
            0,
            Irrep::IDENTITY,
            space,
            y.space,
            b,
            WireGate::Window,
        );
        assert_eq!(
            y.ranks[b], want,
            "bond {b}: the shipped rank-1 path's grid must be `wire_ranks(.., Window)`; a gate \
             default flipped to Triangle would shrink it and silently change the operator"
        );
    }
}

#[test]
fn csu2_window_gate_over_allocates_the_shifted_dimension() {
    let space = fx_space();
    let out_space = space.with_two_s(2);
    for &(cap, want_tri, want_excess) in &[(1usize, 13usize, 2usize), (2, 15, 2), (0, 15, 0)] {
        let grids = capped_grids(&space, cap);
        let tri_dim = dim_over_fan(space, out_space, &grids, 2, WireGate::Triangle);
        let win_dim = dim_over_fan(space, out_space, &grids, 2, WireGate::Window);
        println!(
            "csu2 gate dim: cap {cap} -> triangle {tri_dim}, window {win_dim} (excess {})",
            win_dim as i64 - tri_dim as i64
        );
        assert_eq!(
            tri_dim, want_tri,
            "cap {cap}: the triangle-gated dim T^{{S=1}} is measured {want_tri} (C-spec §1.2)"
        );
        assert_eq!(
            win_dim - tri_dim,
            want_excess,
            "cap {cap}: the window gate must over-allocate by exactly {want_excess} free \
             parameters spanning nothing (window {win_dim} vs triangle {tri_dim})"
        );
    }
}

#[test]
#[allow(clippy::needless_range_loop)]
fn csu2_flux_ranks_is_the_fan_and_obeys_the_boundary_laws() {
    let space = fx_space();
    let mut widest = 0usize;
    for two_kappa in [0i64, 2, 4] {
        let out_space = space.with_two_s(two_kappa as usize);
        for cap in [0usize, 1, 2] {
            let grids = capped_grids(&space, cap);
            for b in 0..=D {
                let got = flux_ranks(&grids[b], two_kappa, space, out_space, b);
                let want = fan(
                    &grids[b],
                    two_kappa,
                    space,
                    out_space,
                    b,
                    WireGate::Triangle,
                );
                assert_eq!(
                    got, want,
                    "flux_ranks must be the triangle-gated fan at (2k, cap, bond) = \
                     ({two_kappa}, {cap}, {b})"
                );
                if two_kappa == 2 {
                    for t in out_space.bond_sectors(b) {
                        widest = widest.max(fan_in(space, &grids[b], b, t));
                    }
                }
            }
        }
    }
    assert_eq!(
        widest, 2,
        "the (d, N) = (4, 4) fan is measured at most TWO-fold (see this arm's doc for the \
         admissibility derivation); got {widest}"
    );

    let big = Su2Space::new(8, 8, 0);
    let big_out = big.with_two_s(2);
    let big_caps = max_ranks(&big);
    let mut three_fold = 0usize;
    for b in 0..=big.d {
        let got = flux_ranks(&big_caps[b], 2, big, big_out, b);
        let want = fan(&big_caps[b], 2, big, big_out, b, WireGate::Triangle);
        assert_eq!(
            got, want,
            "flux_ranks must be the triangle-gated fan at the three-fold fixture too (bond {b})"
        );
        for t in big_out.bond_sectors(b) {
            if fan_in(big, &big_caps[b], b, t) == 3 {
                three_fold += 1;
            }
        }
    }
    assert!(
        three_fold > 0,
        "ANTI-VACUITY: no bond label anywhere borrowed THREE source multiplets, so the fan was \
         only ever a relabel and the recoupling structure is untested"
    );
    let witness = Su2Sector::new(4, 2);
    assert_eq!(
        fan_in(big, &big_caps[4], 4, witness),
        3,
        "the measured three-fold witness is bond 4, label {witness:?}, feeders 2s_u in {{0, 2, 4}}"
    );
    assert_eq!(
        flux_ranks(&big_caps[4], 2, big, big_out, 4)[witness],
        big_caps[4][Su2Sector::new(4, 0)]
            + big_caps[4][Su2Sector::new(4, 2)]
            + big_caps[4][Su2Sector::new(4, 4)],
        "and its dressed rank must be the SUM of all three source multiplets' ranks — the fan, not \
         a relabel"
    );
    println!(
        "csu2 fan: widest fan-in at (4,4) = {widest}; {three_fold} three-fold labels at (8,8)"
    );

    let out_space = space.with_two_s(2);
    let grids = capped_grids(&space, 0);
    let rp: Vec<Su2RankGrid> = (0..=D)
        .map(|b| flux_ranks(&grids[b], 2, space, out_space, b))
        .collect();
    let rows: Vec<Su2RankGrid> = (0..=D)
        .map(|b| flux_ranks(&grids[b], 0, space, out_space, b))
        .collect();
    assert_eq!(
        rp[0].total(),
        0,
        "the fan of (0,0) is (0,2), inadmissible at bond 0 by clause (2) — the V-band must have \
         ZERO width at bond 0, which is what makes ι's first core `[delta_0 | U_0]` legal"
    );
    let boundary = Su2Sector::with_irrep(NEL, 2, out_space.target);
    assert_eq!(
        rp[D][boundary], 1,
        "the dressed grid must be rank 1 at space_out's own boundary sector {boundary:?}"
    );
    assert_eq!(
        rp[D].total(),
        1,
        "and rank 0 everywhere else at bond d, or ι(xi) is not a single-target train"
    );
    assert_eq!(
        rows[D].total(),
        0,
        "(N, 0) is inadmissible at bond d in space_out by clause (4), so the masked U-band must \
         have ZERO width at bond d — this is what makes ι's last core `[V1 ; delta]` legal"
    );
    let left = Su2Sector::with_irrep(0, 0, Irrep::IDENTITY);
    assert_eq!(
        (rows[0][left], rows[0].total()),
        (1, 1),
        "the masked left frame must still be rank 1 at (0,0,I) — nothing is occupied left of \
         bond 0, so that end survives the 2S-aware mask"
    );
}

#[test]
#[allow(clippy::needless_range_loop)]
fn csu2_mask_lemma_is_non_vacuous_at_the_fixture() {
    let space = fx_space();
    let out_space = space.with_two_s(2);
    let grids = capped_grids(&space, 0);

    let mut masked_with_rank: Vec<(usize, Su2Sector)> = Vec::new();
    for k in 0..=D {
        for l in space.bond_sectors(k) {
            if grids[k][l] > 0 && !out_space.contains(k, l) {
                masked_with_rank.push((k, l));
            }
        }
    }
    println!("csu2 mask: dropped labels carrying rank = {masked_with_rank:?}");
    assert!(
        !masked_with_rank.is_empty(),
        "ANTI-VACUITY: the 2S-aware mask dropped nothing at this fixture, so the Mask Lemma has \
         no content here and the arm proves nothing"
    );
    for named in [Su2Sector::new(2, 0), Su2Sector::new(4, 0)] {
        assert!(
            masked_with_rank.contains(&(3, named)),
            "bond 3 must carry rank at {named:?} and the triplet space must NOT admit it \
             (C-spec §1.3's measured non-vacuity witness); dropped = {masked_with_rank:?}"
        );
    }
    assert_eq!(
        {
            let mut got: Vec<Su2Sector> = out_space
                .bond_sectors(3)
                .into_iter()
                .filter(|&s| out_space.contains(3, s))
                .collect();
            got.sort_by_key(|s| (s.n, s.two_s, s.irrep.index()));
            got
        },
        vec![
            Su2Sector::new(2, 2),
            Su2Sector::new(3, 1),
            Su2Sector::new(3, 3),
            Su2Sector::new(4, 2),
        ],
        "the triplet space's bond-3 labels are measured {{(3,1), (2,2), (4,2), (3,3)}}"
    );

    let mut checked = 0usize;
    for k in 0..D {
        let g = out_space.pg.irrep_of(k);
        for l in space.bond_sectors(k) {
            if out_space.contains(k, l) {
                continue;
            }
            for f in Family::ALL {
                let Some(t) = f.apply(l, g) else { continue };
                assert!(
                    !out_space.contains(k + 1, t),
                    "MASK LEMMA VIOLATED at bond {k}, label {l:?}, family {f:?}: {t:?} IS \
                     admissible in space_out, so the masked row is NOT structurally dead and \
                     dropping it changes the space"
                );
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "ANTI-VACUITY: no masked label had a single legal family transition to check"
    );
    println!("csu2 mask: {checked} family transitions out of masked labels, all dead");
}

#[test]
fn csu2_dressed_right_frame_is_theta_right_canonical() {
    let space = fx_space();
    for cap in [0usize, 2, 1] {
        let (_, bl, br) = fx_state(cap, 0xC1_A0 + cap as u64);
        for k in 1..D {
            let s = rproject(br.core(k), br.core(k));
            for sec in s.sectors() {
                let Some(b) = s.block(sec) else { continue };
                for i in 0..b.nrows() {
                    for j in 0..b.ncols() {
                        let want = if i == j { 1.0 } else { 0.0 };
                        let got = *b.get(i, j);
                        assert!(
                            (got - want).abs() <= 1e-12,
                            "PREMISE FAILED (cap {cap}): the UNDRESSED right frame is not \
                             θ-right-canonical at bond {k}, sector {sec:?}, entry ({i},{j}): \
                             {got} vs {want} — this is a fixture/gauge bug, not a dressing bug"
                        );
                    }
                }
            }
        }

        for two_kappa in [0i64, 2] {
            let out_space = space.with_two_s(two_kappa as usize);
            let v1 = dress_train(&br, out_space, two_kappa, &mut SpinAlgebra::new());
            let u0 = dress_train(&bl, out_space, 0, &mut SpinAlgebra::new());
            assert_eq!(
                v1.space, out_space,
                "the dressed right frame must be stamped in space_out"
            );
            assert_eq!(
                u0.space, out_space,
                "the masked restamp of U must be stamped in space_out"
            );

            let mut worst = (0usize, Su2Sector::new(0, 0), 0usize, 0usize, 0.0f64);
            let mut diag_seen = 0usize;
            for k in 1..D {
                let s = rproject(v1.core(k), v1.core(k));
                for sec in s.sectors() {
                    let Some(b) = s.block(sec) else { continue };
                    assert_eq!(
                        (b.nrows(), b.ncols()),
                        (v1.ranks[k][sec], v1.ranks[k][sec]),
                        "rproject must return a square block at the DRESSED rank (bond {k}, \
                         {sec:?})"
                    );
                    for i in 0..b.nrows() {
                        for j in 0..b.ncols() {
                            let want = if i == j { 1.0 } else { 0.0 };
                            let dev = (*b.get(i, j) - want).abs();
                            if i == j {
                                diag_seen += 1;
                            }
                            if dev > worst.4 {
                                worst = (k, sec, i, j, dev);
                            }
                        }
                    }
                }
            }
            println!(
                "csu2 R1 gate: cap {cap}, 2k = {two_kappa}: worst rproject(V1,V1) - I defect \
                 {:e} at bond {}, {:?}, entry ({},{})",
                worst.4, worst.0, worst.1, worst.2, worst.3
            );
            assert!(
                diag_seen > 0,
                "ANTI-VACUITY (cap {cap}, 2k {two_kappa}): the dressed frame had no interior \
                 diagonal entry to check, so the gate is empty"
            );
            assert!(
                worst.4 <= 1e-14 * 100.0,
                "★ R1 GATE FAILED (cap {cap}, 2k = {two_kappa}): the wire-dressed right frame is \
                 NOT θ-right-canonical — worst |rproject(V1,V1) - I| = {:e} at bond {}, sector \
                 {:?}, entry ({},{}). Do NOT relax this tolerance: project_shifted's right fold \
                 and right_gauge_complement_mut both require the identity, and a violation is a \
                 SILENT H9-class metric error (wrong omega, every structural arm still green). \
                 ESCALATE per C-spec §4 R1 (the fallback re-orthogonalizes the dressed frame and \
                 makes dim data-dependent).",
                worst.4,
                worst.0,
                worst.1,
                worst.2,
                worst.3
            );

            let mut lworst = (0usize, Su2Sector::new(0, 0), 0.0f64);
            for k in 0..D - 1 {
                let s = adjoint_core_times_core(u0.core(k), u0.core(k));
                for sec in s.sectors() {
                    let Some(b) = s.block(sec) else { continue };
                    for i in 0..b.nrows() {
                        for j in 0..b.ncols() {
                            let want = if i == j { 1.0 } else { 0.0 };
                            let dev = (*b.get(i, j) - want).abs();
                            if dev > lworst.2 {
                                lworst = (k, sec, dev);
                            }
                        }
                    }
                }
            }
            assert!(
                lworst.2 <= 1e-12,
                "the masked restamp of U must stay PLAIN left-orthonormal (H9-c: the left chain \
                 carries no θ and no centerweight) — worst defect {:e} at bond {}, {:?}",
                lworst.2,
                lworst.0,
                lworst.1
            );
        }
    }

    {
        let big = Su2Space::new(8, 8, 0);
        let big_out = big.with_two_s(2);
        let grids = capped_grids(&big, 1);
        let mut rng = seeded(0xC1_A9);
        let mut x: Su2TtVector<f64> = tt_randn(big, &grids, &mut rng);
        let n = x.norm_lr(false);
        assert!(n > 1e-8, "fixture: the (8,8) cap-1 state must be nonzero");
        x.scale_mut(1.0 / n);
        assert!(
            x.check(),
            "fixture: the (8,8) reference must be a legal train"
        );
        let (bl, br) = tangent_bases(&x);

        let witness = Su2Sector::new(4, 2);
        assert_eq!(
            fan_in(big, &bl.ranks[4], 4, witness),
            3,
            "PREMISE: at (8,8) cap 1 the bond-4 label {witness:?} must borrow THREE source \
             multiplets (2s_u in {{0, 2, 4}}), or this leg is the two-fold statement again"
        );
        for k in 1..big.d {
            let sk = rproject(br.core(k), br.core(k));
            for sec in sk.sectors() {
                let Some(b) = sk.block(sec) else { continue };
                for i in 0..b.nrows() {
                    for j in 0..b.ncols() {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (*b.get(i, j) - want).abs() <= 1e-12,
                            "PREMISE FAILED at (8,8) cap 1: the UNDRESSED right frame is not \
                             θ-right-canonical at bond {k}, sector {sec:?}, entry ({i},{j}) — a \
                             fixture/gauge bug, not a dressing bug"
                        );
                    }
                }
            }
        }

        let v1 = dress_train(&br, big_out, 2, &mut SpinAlgebra::new());
        assert_eq!(v1.space, big_out);
        let mut worst = (0usize, Su2Sector::new(0, 0), 0.0f64);
        let mut three_fold_rows = 0usize;
        for k in 1..big.d {
            let sk = rproject(v1.core(k), v1.core(k));
            for sec in sk.sectors() {
                let Some(b) = sk.block(sec) else { continue };
                assert_eq!(
                    (b.nrows(), b.ncols()),
                    (v1.ranks[k][sec], v1.ranks[k][sec]),
                    "rproject must return a square block at the DRESSED rank (bond {k}, {sec:?})"
                );
                if fan_in(big, &bl.ranks[k], k, sec) == 3 {
                    three_fold_rows += 1;
                }
                for i in 0..b.nrows() {
                    for j in 0..b.ncols() {
                        let want = if i == j { 1.0 } else { 0.0 };
                        let dev = (*b.get(i, j) - want).abs();
                        if dev > worst.2 {
                            worst = (k, sec, dev);
                        }
                    }
                }
            }
        }
        assert!(
            three_fold_rows > 0,
            "ANTI-VACUITY: the dressed frame carried no three-fold label with a live block, so this \
             leg is the two-fold statement in disguise"
        );
        println!(
            "csu2 R1 gate: (8,8) cap 1, 2k = 2: {three_fold_rows} three-fold dressed labels, worst \
             |rproject(V1,V1) - I| defect {:e} at bond {}, {:?}",
            worst.2, worst.0, worst.1
        );
        assert!(
            worst.2 <= 1e-14 * 100.0,
            "★ R1 GATE FAILED on the THREE-FOLD cell ((8,8) cap 1, 2k = 2): worst \
             |rproject(V1,V1) - I| = {:e} at bond {}, sector {:?}. Same escalation as the (4,4) \
             gate above — do NOT relax the tolerance; a three-fold-only failure says the recoupling \
             sum is orthogonal for two feeders but not for three, which is an argument-order or \
             normalization error that the (4,4) fixture cannot see.",
            worst.2,
            worst.0,
            worst.1
        );
    }
}

#[test]
fn csu2_dress_train_at_zero_wire_is_the_masked_restamp() {
    let space = fx_space();
    let out_space = space.with_two_s(2);
    let (_, bl, br) = fx_state(0, 0xC1_B0);

    let u0 = dress_train(&bl, out_space, 0, &mut SpinAlgebra::new());
    let mut compared = 0usize;
    for k in 0..D {
        let g = out_space.pg.irrep_of(k);
        for l in out_space.bond_sectors(k) {
            if !out_space.contains(k, l) {
                continue;
            }
            for f in Family::ALL {
                let Some(t) = f.apply(l, g) else { continue };
                if !out_space.contains(k + 1, t) {
                    continue;
                }
                let (Some(a), Some(b)) = (u0.core(k).block(f, l), bl.core(k).block(f, l)) else {
                    continue;
                };
                assert_eq!(
                    (a.nrows(), a.ncols()),
                    (b.nrows(), b.ncols()),
                    "kappa = 0 must not change block shapes (bond {k}, {f:?}, {l:?})"
                );
                for i in 0..a.nrows() {
                    for j in 0..a.ncols() {
                        assert_eq!(
                            *a.get(i, j),
                            *b.get(i, j),
                            "kappa = 0 must copy the source block EXACTLY (bond {k}, {f:?}, \
                             {l:?}, entry ({i},{j})) — a recoupling coefficient leaking into the \
                             no-op regime corrupts the shifted GAUGE condition, not the metric"
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(
        compared > 0,
        "ANTI-VACUITY: no surviving block was compared at kappa = 0"
    );

    let v1 = dress_train(&br, out_space, 2, &mut SpinAlgebra::new());
    let mut fanned = false;
    for b in 0..=D {
        let r0 = flux_ranks(&br.ranks[b], 0, space, out_space, b);
        if v1.ranks[b] != r0 {
            fanned = true;
        }
    }
    assert!(
        fanned,
        "ANTI-VACUITY: the kappa = 2 dressing produced the kappa = 0 grids at every bond, so the \
         wire fanned nothing and every downstream arm is a relabel test in disguise"
    );
}
