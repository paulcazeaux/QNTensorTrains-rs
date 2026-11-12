// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::core::weingarten::FloorPolicy;
use crate::su2::qn::{Su2Sector, max_ranks};
use crate::su2::riemannian::tangent_bases;
use crate::su2::tangent::shifted_frames;
use crate::su2::tensorops::src_offset;
use crate::su2::vector::tt_randn;
use crate::su2::weingarten::{bond_gauges, tangent_basis_shifted, tangent_dim_grids};
use crate::test_util::seeded;

const D: usize = 4;
const NEL: usize = 4;
const KAPPA2: i64 = 2;
const KAPPA4: i64 = 4;

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
    assert!(x.check(), "fixture: the reference must be a legal train");
    let (bl, br) = tangent_bases(&x);
    assert_eq!(bl.ranks, br.ranks, "fixture: one point, two frames");
    (space, bl, br)
}

fn tri(a: i64, b: i64, c: i64) -> bool {
    c >= (a - b).abs() && c <= a + b && (a + b + c) % 2 == 0
}

fn scheme_a(two_a: i64, two_sigma: i64, two_u: i64, two_kappa: i64) -> Vec<i64> {
    (0..=(two_a + two_sigma))
        .filter(|&t| tri(two_a, two_sigma, t) && tri(t, two_u, two_kappa))
        .collect()
}

fn scheme_b(two_a: i64, two_sigma: i64, two_u: i64, two_kappa: i64) -> Vec<i64> {
    (0..=(two_a + two_kappa))
        .filter(|&l| tri(two_a, two_kappa, l) && tri(l, two_sigma, two_u))
        .collect()
}

fn crossing_coeff(
    spin: &mut SpinAlgebra,
    two_a: i64,
    two_t: i64,
    two_sigma: i64,
    two_kappa: i64,
    two_l: i64,
    two_u: i64,
) -> f64 {
    spin.recouple(
        two_a as i32,
        two_t as i32,
        two_sigma as i32,
        two_kappa as i32,
        two_kappa as i32,
        0,
        two_l as i32,
        two_u as i32,
        two_sigma as i32,
    )
}

fn frob2(c: &Su2Core<f64>) -> f64 {
    let mut s = 0.0f64;
    for f in Family::ALL {
        for l in c.transitions(f).collect::<Vec<_>>() {
            if let Some(b) = c.block(f, l) {
                for j in 0..b.ncols() {
                    for i in 0..b.nrows() {
                        s += b.get(i, j) * b.get(i, j);
                    }
                }
            }
        }
    }
    s
}

fn frob_dot(a: &Su2Core<f64>, b: &Su2Core<f64>) -> f64 {
    assert_eq!(a.space, b.space, "frob_dot: stamps differ");
    assert_eq!(a.k, b.k);
    let mut s = 0.0f64;
    for f in Family::ALL {
        for l in a.transitions(f).collect::<Vec<_>>() {
            let (Some(x), Some(y)) = (a.block(f, l), b.block(f, l)) else {
                continue;
            };
            assert_eq!((x.nrows(), x.ncols()), (y.nrows(), y.ncols()));
            for j in 0..x.ncols() {
                for i in 0..x.nrows() {
                    s += x.get(i, j) * y.get(i, j);
                }
            }
        }
    }
    s
}

struct Grids {
    space: Su2Space,
    out: Su2Space,
    r: Vec<Su2RankGrid>,
    rm: Vec<Su2RankGrid>,
    rd: Vec<Su2RankGrid>,
    rdz: Vec<Su2RankGrid>,
}

fn grids(space: Su2Space, r: &[Su2RankGrid], two_kappa: i64) -> Grids {
    let out = shifted_out_space(space, two_kappa as usize);
    Grids {
        space,
        out,
        r: r.to_vec(),
        rm: flux_grids(r, 0, space, out),
        rd: flux_grids(r, two_kappa, space, out),
        rdz: flux_grids_zside(r, two_kappa, space, out),
    }
}

fn random_scheme_b(g: &Grids, k: usize, seed: u64) -> Su2Core<f64> {
    let mut c = Su2Core::<f64>::zeros(g.space, k, &g.rdz[k], &g.r[k + 1]);
    c.fill_randn(&mut seeded(seed));
    c
}

fn random_scheme_a(g: &Grids, k: usize, seed: u64) -> Su2Core<f64> {
    let mut c = Su2Core::<f64>::zeros(g.out, k, &g.rm[k], &g.rd[k + 1]);
    c.fill_randn(&mut seeded(seed));
    c
}

fn b_to_a(g: &Grids, k: usize, src: &Su2Core<f64>, spin: &mut SpinAlgebra) -> Su2Core<f64> {
    cross_wire(
        src,
        CrossDir::BtoA,
        (g.out.two_s) as i64,
        (&g.rdz[k], &g.r[k + 1]),
        (&g.rm[k], &g.rd[k + 1]),
        g.space,
        g.out,
        (&g.r[k], &g.r[k + 1]),
        spin,
    )
}

fn a_to_b(g: &Grids, k: usize, src: &Su2Core<f64>, spin: &mut SpinAlgebra) -> Su2Core<f64> {
    cross_wire(
        src,
        CrossDir::AtoB,
        (g.out.two_s) as i64,
        (&g.rm[k], &g.rd[k + 1]),
        (&g.rdz[k], &g.r[k + 1]),
        g.out,
        g.space,
        (&g.r[k], &g.r[k + 1]),
        spin,
    )
}

fn phi_channel(two_s_t: usize, two_s_u: usize) -> f64 {
    let phase = if ((two_s_t as i64 - two_s_u as i64) / 2).rem_euclid(2) == 0 {
        1.0
    } else {
        -1.0
    };
    phase * ((two_s_u as f64 + 1.0) / (two_s_t as f64 + 1.0)).sqrt()
}

fn unweight_scheme_a(g: &Grids, k: usize, c: &Su2Core<f64>, two_kappa: i64) -> Su2Core<f64> {
    use faer::prelude::ReborrowMut;
    let mut out = c.clone();
    let gk = g.out.pg.irrep_of(k);
    for f in Family::ALL {
        for a in g.out.bond_sectors(k) {
            let Some(t) = f.apply(a, gk) else { continue };
            if !g.out.contains(k + 1, t) {
                continue;
            }
            let Some(mut b) = out.block_mut(f, a) else {
                continue;
            };
            for u in g.space.bond_sectors(k + 1) {
                let ru = g.r[k + 1][u];
                if ru == 0
                    || u.n != t.n
                    || u.irrep != t.irrep
                    || !tri(t.two_s as i64, u.two_s as i64, two_kappa)
                {
                    continue;
                }
                let off = src_offset(
                    t.n as i64,
                    t.two_s as i64,
                    u.two_s as i64,
                    two_kappa,
                    0,
                    t.irrep,
                    &g.r[k + 1],
                    g.space.n,
                    WireGate::Triangle,
                );
                if off < 0 {
                    continue;
                }
                let (off, phi) = (off as usize, phi_channel(t.two_s, u.two_s));
                for j in 0..ru.min(b.ncols().saturating_sub(off)) {
                    for i in 0..b.nrows() {
                        *b.rb_mut().get_mut(i, off + j) /= phi;
                    }
                }
            }
        }
    }
    out
}

fn max_abs_diff(a: &Su2Core<f64>, b: &Su2Core<f64>) -> f64 {
    let mut m = 0.0f64;
    for f in Family::ALL {
        for l in a.transitions(f).collect::<Vec<_>>() {
            let (Some(x), Some(y)) = (a.block(f, l), b.block(f, l)) else {
                continue;
            };
            assert_eq!(
                (x.nrows(), x.ncols()),
                (y.nrows(), y.ncols()),
                "block {f:?} {l:?}"
            );
            for j in 0..x.ncols() {
                for i in 0..x.nrows() {
                    m = m.max((x.get(i, j) - y.get(i, j)).abs());
                }
            }
        }
    }
    m
}

#[test]
fn csu2b_crossing_coefficient_is_orthogonal_on_every_block() {
    let mut spin = SpinAlgebra::new();
    let mut blocks = 0usize;
    let mut worst = 0.0f64;
    let mut two_by_two = 0usize;
    for two_kappa in [KAPPA2, KAPPA4] {
        for two_sigma in [0i64, 1] {
            for two_a in 0..=6i64 {
                for two_u in 0..=6i64 {
                    let ts = scheme_a(two_a, two_sigma, two_u, two_kappa);
                    let ls = scheme_b(two_a, two_sigma, two_u, two_kappa);
                    assert_eq!(
                        ts.len(),
                        ls.len(),
                        "(2a, 2σ, 2u, 2κ) = ({two_a}, {two_sigma}, {two_u}, {two_kappa}): the two \
                         coupling schemes enumerate the SAME invariant space Hom(a⊗σ⊗u, κ), so \
                         their index sets must have equal size: A {ts:?} vs B {ls:?}"
                    );
                    let m = ts.len();
                    if m == 0 {
                        continue;
                    }
                    assert!(m <= 2, "m(a,σ,u;κ) ≤ 2 for a one-electron site: got {m}");
                    if m == 2 {
                        two_by_two += 1;
                    }
                    blocks += 1;
                    let c: Vec<Vec<f64>> = ts
                        .iter()
                        .map(|&t| {
                            ls.iter()
                                .map(|&l| {
                                    crossing_coeff(
                                        &mut spin, two_a, t, two_sigma, two_kappa, l, two_u,
                                    )
                                })
                                .collect()
                        })
                        .collect();
                    for i in 0..m {
                        for j in 0..m {
                            let dot: f64 = (0..m).map(|k| c[i][k] * c[j][k]).sum();
                            let want = if i == j { 1.0 } else { 0.0 };
                            worst = worst.max((dot - want).abs());
                            assert!(
                                (dot - want).abs() <= 1e-13,
                                "(2a, 2σ, 2u, 2κ) = ({two_a}, {two_sigma}, {two_u}, {two_kappa}): \
                                 (CCᵀ)_{i}{j} = {dot} — the crossed recouple must be ORTHOGONAL \
                                 (C = {c:?})"
                            );
                        }
                    }
                }
            }
        }
    }
    println!(
        "csu2b B1: {blocks} non-empty blocks ({two_by_two} of them 2×2), max |CCᵀ − I| = {worst:e}"
    );
    assert!(blocks > 0, "ANTI-VACUITY: the sweep produced no block");
    assert!(
        two_by_two > 0,
        "ANTI-VACUITY: no 2×2 block was swept, so the mixing (rather than a sign) is untested"
    );
}

#[test]
fn csu2b_crossing_sign_case_0112_is_minus_one_and_swaps_up_to_dn() {
    let mut spin = SpinAlgebra::new();
    let (two_a, two_sigma, two_u, two_kappa) = (0i64, 1i64, 1i64, 2i64);
    let ts = scheme_a(two_a, two_sigma, two_u, two_kappa);
    let ls = scheme_b(two_a, two_sigma, two_u, two_kappa);
    assert_eq!(ts, vec![1], "scheme A: only 2t = 1 is live");
    assert_eq!(ls, vec![2], "scheme B: only 2l = 2 is live");
    assert_eq!(ts[0] - two_a, 1, "scheme A's live family is Up (Δ2s = +1)");
    assert_eq!(two_u - ls[0], -1, "scheme B's live family is Dn (Δ2s = −1)");
    let c = crossing_coeff(&mut spin, two_a, ts[0], two_sigma, two_kappa, ls[0], two_u);
    println!(
        "csu2b B2: C(0,1,1,2) = {c:.17e}, |C + 1| = {:e}",
        (c + 1.0).abs()
    );
    assert!(
        (c + 1.0).abs() <= 2.0 * f64::EPSILON,
        "C(0,1,1,2) must be −1 (got {c}): the crossing carries a SIGN a copy would drop"
    );
}

#[test]
fn csu2b_crossing_is_plus_one_on_the_spin_zero_site_families() {
    let mut spin = SpinAlgebra::new();
    let mut blocks = 0usize;
    let mut worst = 0.0f64;
    for two_kappa in [KAPPA2, KAPPA4] {
        for two_a in 0..=6i64 {
            for two_u in 0..=6i64 {
                let ts = scheme_a(two_a, 0, two_u, two_kappa);
                let ls = scheme_b(two_a, 0, two_u, two_kappa);
                if ts.is_empty() {
                    assert!(ls.is_empty());
                    continue;
                }
                blocks += 1;
                assert_eq!(ts, vec![two_a], "2σ = 0: t = a");
                assert_eq!(ls, vec![two_u], "2σ = 0: l = u");
                let c = crossing_coeff(&mut spin, two_a, two_a, 0, two_kappa, two_u, two_u);
                worst = worst.max((c - 1.0).abs());
                assert!(
                    (c - 1.0).abs() <= 2.0 * f64::EPSILON,
                    "(2a, 2u, 2κ) = ({two_a}, {two_u}, {two_kappa}): C = {c}, must be +1"
                );
            }
        }
    }
    println!("csu2b B3: {blocks} spin-zero blocks, max |C − 1| = {worst:e}");
    assert_eq!(
        blocks, 35,
        "the spec's measured block count over this sweep"
    );
}

#[test]
fn csu2b_cross_wire_realizes_the_signed_family_swap_at_the_fixture() {
    for cap in [0usize, 2, 1] {
        let (space, bl, _br) = fx_state(cap, 0xCB_20 + cap as u64);
        let g = grids(space, &bl.ranks, KAPPA2);
        let k = 2usize;
        let a = Su2Sector::new(2, 0);
        let l = Su2Sector::new(2, 2);
        let u = Su2Sector::new(3, 1);
        let t = Su2Sector::new(3, 1);
        assert!(g.rdz[k][l] > 0 && g.r[k + 1][u] > 0 && g.rm[k][a] > 0 && g.rd[k + 1][t] > 0);
        let off_a = src_offset(
            l.n as i64,
            l.two_s as i64,
            a.two_s as i64,
            KAPPA2,
            0,
            Irrep::IDENTITY,
            &g.r[k],
            space.n,
            WireGate::Triangle,
        );
        let off_u = src_offset(
            t.n as i64,
            t.two_s as i64,
            u.two_s as i64,
            KAPPA2,
            0,
            Irrep::IDENTITY,
            &g.r[k + 1],
            space.n,
            WireGate::Triangle,
        );
        assert!(off_a >= 0 && off_u >= 0, "both channels must be allocated");
        let (off_a, off_u) = (off_a as usize, off_u as usize);

        let mut y = Su2Core::<f64>::zeros(g.space, k, &g.rdz[k], &g.r[k + 1]);
        {
            use faer::prelude::ReborrowMut;
            let mut blk = y
                .block_mut(Family::Dn, l)
                .expect("the (Dn, (2,2)) block exists");
            *blk.rb_mut().get_mut(off_a, 0) = 1.0;
        }
        let mut spin = SpinAlgebra::new();
        let out = b_to_a(&g, k, &y, &mut spin);
        assert_eq!(out.space, g.out, "BtoA lands in space_out");
        let got = out
            .block(Family::Up, a)
            .expect("the (Up, (2,0)) block exists")
            .get(0, off_u)
            .to_owned();
        println!(
            "csu2b B2 scatter [cap {cap}]: (Dn,(2,2))[{off_a},0] → (Up,(2,0))[0,{off_u}] = {got:.17e}"
        );
        assert!(
            (got + 1.0).abs() <= 2.0 * f64::EPSILON,
            "[cap {cap}] the crossed entry must be −1 (got {got}): a copy-shaped scatter gets the \
             family right and the sign wrong"
        );
        let total = frob2(&out);
        assert!(
            (total - 1.0).abs() <= 8.0 * f64::EPSILON,
            "[cap {cap}] the crossing is orthogonal, so one unit entry in must be one unit of \
             Frobenius mass out (got {total})"
        );
        let up_mass = {
            let b = out.block(Family::Up, a).unwrap();
            let mut s = 0.0;
            for j in 0..b.ncols() {
                for i in 0..b.nrows() {
                    s += b.get(i, j) * b.get(i, j);
                }
            }
            s
        };
        assert!(
            (up_mass - 1.0).abs() <= 8.0 * f64::EPSILON,
            "[cap {cap}] the whole mass must land in the Up block (got {up_mass} of {total}) — \
             the crossing MIXES site families, which no relabel can do"
        );
    }
}

#[test]
fn csu2b_cross_wire_round_trip_is_a_masked_partial_isometry() {
    let tol = 1e-14;
    for cap in [0usize, 2, 1] {
        let (space, bl, _br) = fx_state(cap, 0xCB_40 + cap as u64);
        let g = grids(space, &bl.ranks, KAPPA2);
        let mut spin = SpinAlgebra::new();
        let mut live_total = 0usize;
        let mut masked_total = 0usize;
        let mut moved_total = 0usize;
        for k in 1..D {
            let y0 = random_scheme_b(&g, k, 0xCB_41 + 16 * cap as u64 + k as u64);
            let a1 = b_to_a(&g, k, &y0, &mut spin);
            let y1 = a_to_b(&g, k, &a1, &mut spin);
            let a2 = b_to_a(&g, k, &y1, &mut spin);
            let y2 = a_to_b(&g, k, &a2, &mut spin);
            let n_a1 = frob2(&a1).sqrt();
            let n_y0 = frob2(&y0).sqrt();
            assert!(
                n_a1 > 1e-8,
                "[cap {cap} site {k}] ANTI-VACUITY: BtoA annihilated the core"
            );
            assert!(
                (frob2(&y1).sqrt() - n_y0).abs() <= tol * n_y0,
                "[cap {cap} site {k}] AtoB ∘ BtoA must be the IDENTITY on the live content: \
                 ‖AtoB(BtoA y)‖ = {} vs ‖y‖ = {n_y0} (‖BtoA y‖ = {n_a1})",
                frob2(&y1).sqrt()
            );
            assert!(
                max_abs_diff(&a2, &a1) <= tol * n_a1,
                "[cap {cap} site {k}] BtoA(AtoB(BtoA y)) ≠ BtoA y ({:e})",
                max_abs_diff(&a2, &a1)
            );
            assert!(
                max_abs_diff(&y2, &y1) <= tol * n_a1,
                "[cap {cap} site {k}] the composite AtoB∘BtoA must be idempotent ({:e})",
                max_abs_diff(&y2, &y1)
            );
            let (mut live, mut masked) = (0usize, 0usize);
            for f in Family::ALL {
                for l in y0.transitions(f).collect::<Vec<_>>() {
                    let (Some(p), Some(q)) = (y0.block(f, l), y1.block(f, l)) else {
                        continue;
                    };
                    for j in 0..p.ncols() {
                        for i in 0..p.nrows() {
                            let (v0, v1) = (p.get(i, j), q.get(i, j));
                            if (v1 - v0).abs() <= tol * v0.abs().max(1.0) {
                                live += 1;
                            } else if v1.abs() <= tol {
                                masked += 1;
                            } else {
                                panic!(
                                    "[cap {cap} site {k}] block ({f:?}, {l:?}) entry ({i},{j}): \
                                     AtoB∘BtoA changed {v0} into {v1} — a round trip must PRESERVE \
                                     or ZERO an entry, never mix it; the offsets or the family \
                                     keying disagree between the two directions"
                                );
                            }
                        }
                    }
                }
            }
            assert!(
                live > 0,
                "[cap {cap} site {k}] ANTI-VACUITY: no entry survived the round trip"
            );
            live_total += live;
            masked_total += masked;

            let c0 = random_scheme_a(&g, k, 0xCB_48 + 16 * cap as u64 + k as u64);
            let b1 = a_to_b(&g, k, &c0, &mut spin);
            let c1 = b_to_a(&g, k, &b1, &mut spin);
            let b2 = a_to_b(&g, k, &c1, &mut spin);
            let n_b1 = frob2(&b1).sqrt();
            assert!(
                n_b1 > 1e-8,
                "[cap {cap} site {k}] ANTI-VACUITY: AtoB annihilated the core"
            );
            assert!(
                max_abs_diff(&b2, &b1) <= tol * n_b1,
                "[cap {cap} site {k}] AtoB(BtoA(AtoB c)) ≠ AtoB c ({:e}) — the composite must be \
                 IDEMPOTENT on the scheme-B side",
                max_abs_diff(&b2, &b1)
            );
            let c2 = b_to_a(&g, k, &b2, &mut spin);
            assert!(
                max_abs_diff(&c2, &c1) <= tol * n_b1,
                "[cap {cap} site {k}] BtoA(AtoB(BtoA(AtoB c))) ≠ BtoA(AtoB c) ({:e}): the composite \
                 must be IDEMPOTENT on the scheme-A side too",
                max_abs_diff(&c2, &c1)
            );
            let (n_c0, n_c1) = (frob2(&c0).sqrt(), frob2(&c1).sqrt());
            assert!(
                n_c1 <= n_c0 * (1.0 + tol),
                "[cap {cap} site {k}] a projection cannot GROW the norm: {n_c1} vs {n_c0}"
            );
            let mut moved = 0usize;
            for f in Family::ALL {
                for l in c0.transitions(f).collect::<Vec<_>>() {
                    let (Some(p), Some(q)) = (c0.block(f, l), c1.block(f, l)) else {
                        continue;
                    };
                    for j in 0..p.ncols() {
                        for i in 0..p.nrows() {
                            let (v0, v1) = (p.get(i, j), q.get(i, j));
                            if (v1 - v0).abs() > tol * v0.abs().max(1.0) {
                                moved += 1;
                            }
                        }
                    }
                }
            }
            println!(
                "csu2b B4 [cap {cap} site {k}]: A→B→A moved {moved} entries (‖c0‖ = {n_c0:e} → \
                 ‖c1‖ = {n_c1:e}); the mask makes this an orthogonal, not a coordinate, projection"
            );
            moved_total += moved;
        }
        println!(
            "csu2b B4 [cap {cap}]: B→A→B round trip preserved {live_total} entries and masked \
             {masked_total} (the mask is what makes a bare '= id' the wrong statement)"
        );
        assert_eq!(
            masked_total, 0,
            "[cap {cap}] B→A→B zeroed {masked_total} of {live_total} entries — measured 0 at every \
             grid: a live scheme-B (l, u) always has its full scheme-A {{t}} set, so a nonzero count \
             here means the offsets or the family keying moved"
        );
        assert!(
            moved_total > 0,
            "[cap {cap}] ANTI-VACUITY: the A→B→A projection moved NO entry at any site, so the \
             masking rule was not exercised — Cb-spec §0 F6 says the two coupling schemes have \
             different representable supports at this fixture"
        );
    }
}

#[test]
fn csu2b_cross_wire_a_to_b_is_the_weighted_adjoint_of_b_to_a() {
    for cap in [0usize, 2, 1] {
        let (space, bl, _br) = fx_state(cap, 0xCB_80 + cap as u64);
        for two_kappa in [KAPPA2, KAPPA4] {
            let g = grids(space, &bl.ranks, two_kappa);
            let mut spin = SpinAlgebra::new();
            let mut nonzero = 0usize;
            for k in 1..D {
                let y = random_scheme_b(
                    &g,
                    k,
                    0xCB_81 + 32 * cap as u64 + 8 * two_kappa as u64 + k as u64,
                );
                let c = random_scheme_a(
                    &g,
                    k,
                    0xCB_85 + 32 * cap as u64 + 8 * two_kappa as u64 + k as u64,
                );
                let cw =
                    unweight_scheme_a(&g, k, &unweight_scheme_a(&g, k, &c, two_kappa), two_kappa);
                let lhs = frob_dot(&b_to_a(&g, k, &y, &mut spin), &cw);
                let rhs = frob_dot(&y, &a_to_b(&g, k, &c, &mut spin));
                let s = frob2(&y).sqrt() * frob2(&c).sqrt();
                if lhs.abs() > 1e-8 * s {
                    nonzero += 1;
                }
                assert!(
                    (lhs - rhs).abs() <= 1e-13 * s.max(1.0),
                    "[cap {cap} 2κ = {two_kappa} site {k}] ⟨BtoA(y), D_φ⁻²c⟩ = {lhs} vs \
                     ⟨y, AtoB(c)⟩ = {rhs}: the two scatter directions are not weighted adjoints of \
                     each other"
                );
            }
            assert!(
                nonzero > 0,
                "[cap {cap} 2κ = {two_kappa}] ANTI-VACUITY: every pairing vanished (measured ≥ 1                  non-trivial site at BOTH 2κ = 2 and 2κ = 4)"
            );
            println!(
                "csu2b B8 [cap {cap} 2κ = {two_kappa}]: {nonzero} of {} pairings non-trivial",
                D - 1
            );
        }
    }
}

#[test]
fn csu2b_lift_bond_sketch_is_the_channel_block_diagonal_copy() {
    for cap in [0usize, 2, 1] {
        let (space, bl, br) = fx_state(cap, 0xCB_50 + cap as u64);
        let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        assert_eq!(gauges.len(), D - 1);
        for two_kappa in [KAPPA2, KAPPA4] {
            let g = grids(space, &bl.ranks, two_kappa);
            let mut repeated = 0usize;
            for bg in &gauges {
                let b = bg.k;
                let lr = lift_bond_sketch(&bg.r, &g.r[b], &g.rd[b], g.out, two_kappa);
                let lri = lift_bond_sketch(&bg.r_inv, &g.r[b], &g.rd[b], g.out, two_kappa);
                assert_eq!(lr.space, g.out);
                assert_eq!(lr.k, b);
                assert_eq!(
                    lr.row_ranks(),
                    g.rd[b],
                    "bond {b}: lifted rows are the dressed grid"
                );
                assert_eq!(
                    lr.col_ranks(),
                    g.rd[b],
                    "bond {b}: lifted cols are the dressed grid"
                );
                for t in g.out.bond_sectors(b) {
                    let nt = g.rd[b][t];
                    if nt == 0 {
                        continue;
                    }
                    let blk = lr.block(t).expect("dressed label present");
                    let blki = lri.block(t).expect("dressed label present");
                    let mut covered = vec![false; nt];
                    let mut channels = 0usize;
                    for u in g.space.bond_sectors(b) {
                        let ru = g.r[b][u];
                        if ru == 0
                            || u.n != t.n
                            || u.irrep != t.irrep
                            || !tri(t.two_s as i64, u.two_s as i64, two_kappa)
                        {
                            continue;
                        }
                        let off = src_offset(
                            t.n as i64,
                            t.two_s as i64,
                            u.two_s as i64,
                            two_kappa,
                            0,
                            t.irrep,
                            &g.r[b],
                            space.n,
                            WireGate::Triangle,
                        );
                        assert!(
                            off >= 0,
                            "bond {b}: channel {u:?} → {t:?} must be allocated"
                        );
                        let off = off as usize;
                        channels += 1;
                        let src = bg.r.block(u).expect("source block");
                        let srci = bg.r_inv.block(u).expect("source inverse block");
                        for j in 0..ru {
                            for i in 0..ru {
                                assert_eq!(
                                    blk.get(off + i, off + j),
                                    src.get(i, j),
                                    "bond {b} label {t:?} channel {u:?} at offset {off}: lift(R) must \
                                     COPY R[u] bit for bit ({i},{j})"
                                );
                                assert_eq!(
                                    blki.get(off + i, off + j),
                                    srci.get(i, j),
                                    "bond {b} label {t:?} channel {u:?}: lift(R⁻¹) must COPY the \
                                     FLOORED inverse bit for bit — never re-SVD the stacked block"
                                );
                            }
                            covered[off + j] = true;
                        }
                    }
                    assert!(
                        covered.iter().all(|&c| c),
                        "bond {b} label {t:?}: the channels must tile the dressed rank {nt}"
                    );
                    if channels >= 2 {
                        repeated += 1;
                    }
                    let mut total = 0.0f64;
                    for j in 0..nt {
                        for i in 0..nt {
                            total += blk.get(i, j) * blk.get(i, j);
                        }
                    }
                    let mut diag = 0.0f64;
                    for u in g.space.bond_sectors(b) {
                        let ru = g.r[b][u];
                        if ru == 0
                            || u.n != t.n
                            || u.irrep != t.irrep
                            || !tri(t.two_s as i64, u.two_s as i64, two_kappa)
                        {
                            continue;
                        }
                        let src = bg.r.block(u).unwrap();
                        for j in 0..ru {
                            for i in 0..ru {
                                diag += src.get(i, j) * src.get(i, j);
                            }
                        }
                    }
                    assert_eq!(
                        total, diag,
                        "bond {b} label {t:?}: every cross-channel entry of the lift must be EXACTLY \
                         zero (R_b acts on the bond index alone; the wire is a spectator)"
                    );
                }
                let mut prod = lr.clone();
                prod.rmul_sketch_mut(&lri);
                for t in g.out.bond_sectors(b) {
                    let nt = g.rd[b][t];
                    if nt == 0 {
                        continue;
                    }
                    let p = prod.block(t).unwrap();
                    for j in 0..nt {
                        for i in 0..nt {
                            let want = if i == j { 1.0 } else { 0.0 };
                            assert!(
                                (p.get(i, j) - want).abs() <= 1e-12,
                                "bond {b} label {t:?}: lift(R)·lift(R⁻¹) ≠ I at ({i},{j}): {}",
                                p.get(i, j)
                            );
                        }
                    }
                }
            }
            println!(
                "csu2b B5 [cap {cap} 2κ = {two_kappa}]: {repeated} dressed labels borrow ≥ 2 channels \
                 (the same σ repeated down the diagonal)"
            );
            if two_kappa == KAPPA2 {
                assert!(
                    repeated > 0,
                    "[cap {cap}] ANTI-VACUITY: no label borrows two channels, so the repetition \
                     is untested"
                );
            }
        }
        let g0 = grids(space, &bl.ranks, 0);
        for bg in &gauges {
            let b = bg.k;
            let l0 = lift_bond_sketch(&bg.r, &g0.r[b], &g0.rm[b], g0.out, 0);
            assert_eq!(l0.row_ranks(), g0.rm[b]);
            for t in g0.out.bond_sectors(b) {
                if g0.rm[b][t] == 0 {
                    continue;
                }
                let (p, q) = (l0.block(t).unwrap(), bg.r.block(t).unwrap());
                for j in 0..q.ncols() {
                    for i in 0..q.nrows() {
                        assert_eq!(
                            p.get(i, j),
                            q.get(i, j),
                            "bond {b} label {t:?}: lift at 2κ = 0 is the identity"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn csu2b_zside_grid_is_the_double_gate_and_bond_3_masks_both_ways() {
    for cap in [0usize, 2, 1] {
        let (space, bl, _br) = fx_state(cap, 0xCB_60 + cap as u64);
        for two_kappa in [KAPPA2, KAPPA4] {
            let g = grids(space, &bl.ranks, two_kappa);
            for b in 0..=D {
                let full = flux_ranks(&bl.ranks[b], two_kappa, space, g.out, b);
                let z = flux_ranks_zside(&bl.ranks[b], two_kappa, space, g.out, b);
                assert_eq!(
                    z, g.rdz[b],
                    "flux_grids_zside must be flux_ranks_zside per bond"
                );
                for s in g.out.bond_sectors(b) {
                    let want = if space.contains(b, s) { full[s] } else { 0 };
                    assert_eq!(
                        z[s], want,
                        "[cap {cap} 2κ = {two_kappa}] bond {b} label {s:?}: rdz is rd masked to \
                         space.contains — whole labels, never a changed rank"
                    );
                }
                for s in space.bond_sectors(b) {
                    if !g.out.contains(b, s) {
                        assert_eq!(
                            z[s], 0,
                            "bond {b} label {s:?} ∈ space \\ space_out carries no dressed rank"
                        );
                    }
                }
            }
            assert_eq!(g.rdz[0].total(), 0, "rdz[0] = 0 (rd[0] = 0 already)");
            assert_eq!(
                g.rdz[D].total(),
                0,
                "rdz[d] = 0: rd[d] sits at (N, 2κ) ∉ space"
            );
            assert_eq!(
                g.rd[D].total(),
                1,
                "rd[d] = 1 at space_out's boundary sector"
            );
        }

        let g = grids(space, &bl.ranks, KAPPA2);
        let b = 3usize;
        let live = |grid: &Su2RankGrid, sp: &Su2Space| -> Vec<(usize, usize)> {
            let mut v: Vec<(usize, usize)> = sp
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| grid[s] > 0)
                .map(|s| (s.n, s.two_s))
                .collect();
            v.sort_unstable();
            v
        };
        let s_labels = live(&g.r[b], &space);
        let rd_labels = live(&g.rd[b], &g.out);
        let rdz_labels = live(&g.rdz[b], &g.out);
        println!(
            "csu2b B6 [cap {cap}] bond 3: space {s_labels:?}, rd {rd_labels:?}, rdz {rdz_labels:?}"
        );
        assert_eq!(
            s_labels,
            vec![(2, 0), (3, 1), (4, 0)],
            "space's bond-3 labels (Cb-spec §0 F6)"
        );
        assert_eq!(
            rd_labels,
            vec![(2, 2), (3, 1), (3, 3), (4, 2)],
            "space_out's dressed bond-3 labels"
        );
        assert_eq!(
            rdz_labels,
            vec![(3, 1)],
            "the double gate keeps exactly (3,1) at bond 3"
        );
        for s in g.out.bond_sectors(b) {
            if g.rd[b][s] > 0 && g.rdz[b][s] == 0 {
                assert!(
                    !space.contains(b, s),
                    "dropped label {s:?} must be inadmissible in space"
                );
                assert_eq!(g.r[b][s], 0, "dropped label {s:?} carries no rank in space");
            }
        }
    }
}

#[test]
fn csu2b_regrid_is_a_lossless_masked_copy_and_dress_core_at_zero_wire_is_regrid() {
    for cap in [0usize, 2, 1] {
        let (space, bl, _br) = fx_state(cap, 0xCB_70 + cap as u64);
        let g = grids(space, &bl.ranks, KAPPA2);
        let mut spin = SpinAlgebra::new();
        for k in 0..D {
            let dressed = dress_core(
                bl.core(k),
                (&g.r[k], &g.r[k + 1]),
                (&g.rm[k], &g.rm[k + 1]),
                space,
                g.out,
                0,
                &mut spin,
            );
            let regridded = regrid_core(bl.core(k), g.out, &g.rm[k], &g.rm[k + 1]);
            assert_eq!(dressed.space, regridded.space);
            assert_eq!(dressed.row_ranks, regridded.row_ranks);
            assert_eq!(dressed.col_ranks, regridded.col_ranks);
            assert_eq!(dressed.num_coeffs(), regridded.num_coeffs());
            for i in 0..dressed.num_coeffs() {
                assert_eq!(
                    dressed.coeff(i),
                    regridded.coeff(i),
                    "site {k} coeff {i}: dress_core at 2κ = 0 must BE regrid_core (flux-0 anchors \
                     are exact-0.0 pins)"
                );
            }
        }
        let (mut kept_total, mut dropped_total) = (0usize, 0usize);
        for k in 0..D {
            let c0 = random_scheme_a(&g, k, 0xCB_71 + 8 * cap as u64 + k as u64);
            let z = regrid_core(&c0, g.space, &g.r[k], &g.rdz[k + 1]);
            assert_eq!(z.space, g.space);
            assert_eq!(z.row_ranks, g.r[k]);
            assert_eq!(z.col_ranks, g.rdz[k + 1]);
            let back = regrid_core(&z, g.out, &g.rm[k], &g.rd[k + 1]);
            assert_eq!(back.row_ranks, g.rm[k]);
            assert_eq!(back.col_ranks, g.rd[k + 1]);
            let gk = g.out.pg.irrep_of(k);
            let (mut kept, mut dropped) = (0usize, 0usize);
            for f in Family::ALL {
                for l in g.out.bond_sectors(k) {
                    let Some(t) = f.apply(l, gk) else { continue };
                    let (Some(p), Some(q)) = (c0.block(f, l), back.block(f, l)) else {
                        continue;
                    };
                    let shared = space.contains(k, l) && space.contains(k + 1, t);
                    for j in 0..p.ncols() {
                        for i in 0..p.nrows() {
                            if shared {
                                assert_eq!(
                                    q.get(i, j),
                                    p.get(i, j),
                                    "site {k} ({f:?},{l:?}) ({i},{j}): shared labels copy exactly"
                                );
                                kept += 1;
                            } else {
                                assert_eq!(
                                    *q.get(i, j),
                                    0.0,
                                    "site {k} ({f:?},{l:?}): a label outside space comes back zero"
                                );
                                dropped += 1;
                            }
                        }
                    }
                }
            }
            println!("csu2b regrid [cap {cap} site {k}]: kept {kept}, masked {dropped}");
            kept_total += kept;
            dropped_total += dropped;
        }
        assert!(
            kept_total > 0,
            "[cap {cap}] ANTI-VACUITY: no label is shared at ANY site, so the copy half of the \
             round trip is untested"
        );
        assert!(
            dropped_total > 0,
            "[cap {cap}] ANTI-VACUITY: no label was masked at any site, so the DROP half of the \
             round trip is untested"
        );
        let train = regrid_train(&bl, g.out, &g.rm);
        assert_eq!(train.space, g.out);
        assert_eq!(train.ranks, g.rm);
        assert_eq!(
            train.ortho, bl.ortho,
            "the ortho flag is carried like dress_train carries it"
        );
        for k in 0..D {
            let want = regrid_core(bl.core(k), g.out, &g.rm[k], &g.rm[k + 1]);
            assert_eq!(train.core(k).num_coeffs(), want.num_coeffs());
            for i in 0..want.num_coeffs() {
                assert_eq!(
                    train.core(k).coeff(i),
                    want.coeff(i),
                    "regrid_train site {k} coeff {i}"
                );
            }
        }
    }
}

#[test]
fn csu2b_shifted_frames_bundle_is_the_spec_table() {
    for cap in [0usize, 2, 1] {
        let (space, bl, br) = fx_state(cap, 0xCB_90 + cap as u64);
        for two_kappa in [KAPPA2, KAPPA4] {
            let g = grids(space, &bl.ranks, two_kappa);
            let sf = Su2ShiftedFrames::new(&bl, &br, two_kappa as usize);
            assert_eq!(sf.two_s_flux, two_kappa as usize);
            assert_eq!(sf.out_space, g.out);
            assert_eq!(sf.rm, g.rm, "rm is flux_grids(r, 0)");
            assert_eq!(sf.rd, g.rd, "rd is flux_grids(r, 2κ)");
            assert_eq!(sf.rdz, g.rdz, "rdz is flux_grids_zside(r, 2κ)");
            let (u, v) = shifted_frames(&bl, &br, two_kappa as usize);
            for (name, a, b) in [("U^(0)", &sf.frames.0, &u), ("V^(1)", &sf.frames.1, &v)] {
                assert_eq!(a.space, b.space, "{name}");
                assert_eq!(a.ranks, b.ranks, "{name}");
                for k in 0..D {
                    assert_eq!(a.core(k).num_coeffs(), b.core(k).num_coeffs());
                    for i in 0..a.core(k).num_coeffs() {
                        assert_eq!(
                            a.core(k).coeff(i),
                            b.core(k).coeff(i),
                            "{name} site {k} coeff {i}: the bundle's frames must BE shifted_frames \
                             bit for bit"
                        );
                    }
                }
            }
            let mut spin = SpinAlgebra::new();
            let uz = regrid_train(
                &dress_train(&bl, g.out, two_kappa, &mut spin),
                space,
                &g.rdz,
            );
            let vz = regrid_train(
                &dress_train(&br, g.out, two_kappa, &mut spin),
                space,
                &g.rdz,
            );
            let vz_oracle = regrid_train(&sf.frames.1, space, &sf.rdz);
            for (name, a, b) in [("U^(1)_z", &sf.zside, &uz), ("V^(1)_z", &vz_oracle, &vz)] {
                assert_eq!(
                    a.space, space,
                    "{name} must be stamped in x's OWN space (F6)"
                );
                assert_eq!(a.ranks, g.rdz, "{name} lives on rdz");
                for k in 0..D {
                    assert!(a.core(k).check_invariants(), "{name} site {k}");
                    assert_eq!(a.core(k).num_coeffs(), b.core(k).num_coeffs());
                    for i in 0..a.core(k).num_coeffs() {
                        assert_eq!(
                            a.core(k).coeff(i),
                            b.core(k).coeff(i),
                            "{name} site {k} coeff {i}"
                        );
                    }
                }
            }
            let mass: f64 = (1..D).map(|k| frob2(vz_oracle.core(k))).sum();
            let zside_interior: usize = (1..D).map(|k| vz_oracle.core(k).num_coeffs()).sum();
            println!(
                "csu2b bundle [cap {cap} 2κ = {two_kappa}]: rdz totals {:?}, ‖V^(1)_z interior‖² = \
                 {mass:e} over {zside_interior} coefficients",
                g.rdz.iter().map(|z| z.total()).collect::<Vec<_>>()
            );
            if two_kappa == KAPPA2 {
                assert!(
                    mass > 1e-8,
                    "[cap {cap}] ANTI-VACUITY: V^(1)_z is zero on the CURVED triplet channel — the \
                     silent-collapse signature"
                );
            } else {
                assert_eq!(
                    zside_interior, 0,
                    "[cap {cap}] the quintet channel's Z-side dressed grid is empty at every \
                     interior bond, so V^(1)_z holds no coefficient at all"
                );
                assert_eq!(mass, 0.0);
            }
        }
    }
}

#[test]
fn csu2b_slot_sites_shifted_pins_the_gauge_aware_counts() {
    let per_site = |g: &Grids| -> Vec<usize> {
        (0..D)
            .map(|k| {
                let gk = g.out.pg.irrep_of(k);
                let mut raw = 0usize;
                for l in g.out.bond_sectors(k) {
                    for f in Family::ALL {
                        let Some(t) = f.apply(l, gk) else { continue };
                        if g.out.contains(k + 1, t) {
                            raw += g.rm[k][l] * g.rd[k + 1][t];
                        }
                    }
                }
                let gauge = if k < D - 1 {
                    g.out
                        .bond_sectors(k + 1)
                        .into_iter()
                        .map(|t| g.rm[k + 1][t] * g.rd[k + 1][t])
                        .sum::<usize>()
                } else {
                    0
                };
                assert!(raw >= gauge, "site {k}: raw {raw} < gauge {gauge}");
                raw - gauge
            })
            .collect()
    };
    for &(cap, want_t, dim_t) in &[
        (0usize, [0usize, 0, 14, 1], 15usize),
        (2, [0, 1, 13, 1], 15),
        (1, [0, 4, 8, 1], 13),
    ] {
        let (space, bl, br) = fx_state(cap, 0xCB_A0 + cap as u64);
        let gt = grids(space, &bl.ranks, KAPPA2);
        let counts = per_site(&gt);
        println!("csu2b B9 [cap {cap}] triplet per-site params {counts:?}");
        assert_eq!(
            counts.to_vec(),
            want_t.to_vec(),
            "[cap {cap}] the gauge-aware per-site counts (Cb-spec §0 F9)"
        );
        assert_eq!(
            counts.iter().sum::<usize>(),
            dim_t,
            "[cap {cap}] they sum to dim T^(S=1)"
        );
        assert_eq!(tangent_dim_grids(&space, &bl.ranks, 2), dim_t);
        let sites = slot_sites_shifted(&space, &bl.ranks, 2);
        let want_sites: Vec<usize> = (0..D).filter(|&k| counts[k] > 0).collect();
        assert_eq!(
            sites, want_sites,
            "[cap {cap}] slot_sites_shifted lists exactly the sites with parameters"
        );
        assert!(
            sites.len() >= 2,
            "[cap {cap}] the triplet channel must be CURVED (≥ 2 slot sites), or every B arm is vacuous"
        );
        let gq = grids(space, &bl.ranks, KAPPA4);
        let cq = per_site(&gq);
        assert_eq!(cq.iter().sum::<usize>(), 1, "[cap {cap}] dim T^(S=2) = 1");
        let sq = slot_sites_shifted(&space, &bl.ranks, 4);
        assert_eq!(
            sq.len(),
            1,
            "[cap {cap}] the quintet channel is SINGLE-SLOT: {sq:?} — B^(S=2) ≡ 0 structurally"
        );
        assert_eq!(tangent_basis_shifted(&bl, &br, 4).len(), 1);
        let s0 = slot_sites_shifted(&space, &bl.ranks, 0);
        assert!(
            s0.len() >= 2,
            "[cap {cap}] the unshifted space reaches ≥ 2 sites: {s0:?}"
        );
    }
}
