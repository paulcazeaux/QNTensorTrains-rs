// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg3::*;

#[test]
fn pg3_qr_lr_rl_boundary_factor() {
    let (pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_020);
    let ranks = capped_ranks(&space, 1, 2, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let xd = x.to_dense();

    let (y, r) = x.qr_lr();
    assert!(y.check());
    assert_eq!((r.nrows(), r.ncols()), (1, 1), "R is 1×1");
    let r00 = *r.get(0, 0);
    let recon: Vec<f64> = y.to_dense().iter().map(|&v| v * r00).collect();
    assert!(
        dense_diff_norm(&recon, &xd) <= 1e-10 * dense_norm(&xd),
        "‖y·R − x‖ ≤ 1e-10‖x‖"
    );
    assert_pure(&pg, &space, &y);

    let (y2, l) = x.qr_rl();
    assert!(y2.check());
    assert_eq!((l.nrows(), l.ncols()), (1, 1), "L is 1×1");
    let l00 = *l.get(0, 0);
    let recon2: Vec<f64> = y2.to_dense().iter().map(|&v| l00 * v).collect();
    assert!(
        dense_diff_norm(&recon2, &xd) <= 1e-10 * dense_norm(&xd),
        "‖L·y − x‖ ≤ 1e-10‖x‖"
    );
    assert_pure(&pg, &space, &y2);
}

#[test]
fn pg3_norm_sweeps_match_dot_norm() {
    let (_pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_021);
    let ranks = capped_ranks(&space, 1, 3, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let n0 = x.norm();

    assert!(
        (x.norm_rl(true) - n0).abs() <= 1e-10 * n0,
        "norm(:RL, keepRank)"
    );
    assert!(
        (x.norm_lr(true) - n0).abs() <= 1e-10 * n0,
        "norm(:LR, keepRank)"
    );
    assert!(
        (x.norm_lr(false) - n0).abs() <= 1e-10 * n0,
        "norm(:LR, revealing)"
    );
    assert!(
        (x.norm_rl(false) - n0).abs() <= 1e-10 * n0,
        "norm(:RL, revealing)"
    );
    assert!(
        (n0 - dense_norm(&x.to_dense())).abs() <= 1e-12 * n0,
        "TT norm ties to the dense oracle (ops.rs::dot's bond-0 seeding was the PG2 |G|>1 bug)"
    );
}

#[test]
fn pg3_svd_vertical_horizontal_reconstruct_blocks() {
    let (_pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_022);
    let ranks = capped_ranks(&space, 1, 3, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());

    for &k in &[2usize, 3usize] {
        let a = &x.cores[k];

        let (u, s, vt) = a.svd_vertical();
        let mut checked_v = false;
        for &cs in &a.col_qn {
            if a.col_ranks[cs] == 0 {
                continue;
            }
            let av = a.vertical_unfolding(cs);
            let Some(ub) = u[cs].as_ref() else { continue };
            let Some(sb) = s.block(cs) else { continue };
            let Some(vtb) = vt.block(cs) else { continue };
            let recon = mat_mul(mat_mul(ub.as_ref(), sb).as_ref(), vtb);
            assert!(
                frob_diff(recon.as_ref(), av.as_ref()) <= 1e-12 * frob(av.as_ref()).max(1.0),
                "core {k}: U·S·Vt reconstructs the vertical unfolding at {cs:?}"
            );
            checked_v = true;
        }
        assert!(
            checked_v,
            "core {k}: vacuous — no positive-rank column sector found"
        );

        let (u, s, vt) = a.svd_horizontal();
        let mut checked_h = false;
        for &rs in &a.row_qn {
            if a.row_ranks[rs] == 0 {
                continue;
            }
            let ah = a.horizontal_unfolding(rs);
            let Some(ub) = u.block(rs) else { continue };
            let Some(sb) = s.block(rs) else { continue };
            let Some(vtb) = vt[rs].as_ref() else { continue };
            let recon = mat_mul(mat_mul(ub, sb).as_ref(), vtb.as_ref());
            assert!(
                frob_diff(recon.as_ref(), ah.as_ref()) <= 1e-12 * frob(ah.as_ref()).max(1.0),
                "core {k}: U·S·Vt reconstructs the horizontal unfolding at {rs:?}"
            );
            checked_h = true;
        }
        assert!(
            checked_h,
            "core {k}: vacuous — no positive-rank row sector found"
        );
    }
}

#[test]
fn pg3_qc_cq_orthonormality() {
    let (_pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_023);
    let ranks = capped_ranks(&space, 1, 3, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());

    for k in [1usize, 2, 3, 4] {
        let (q_core, _c_sketch, new_col_ranks) = x.core(k).qc();
        let qtq = adjoint_core_times_core(&q_core, &q_core);
        let mut any_checked = false;
        for s in space.bond_sectors(k + 1) {
            let n = new_col_ranks[s];
            if n == 0 {
                continue;
            }
            let block = qtq
                .block(s)
                .unwrap_or_else(|| panic!("qc: missing block at {s:?}"));
            assert_eq!((block.nrows(), block.ncols()), (n, n));
            any_checked = true;
            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let got = *block.get(i, j);
                    assert!((got - expected).abs() < 1e-8, "qc: QᴴQ at {s:?} ({i},{j})");
                }
            }
        }
        assert!(
            any_checked,
            "core {k}: qc vacuous — no positive-rank column sector"
        );

        let (q2_core, _c2, new_row_ranks) = x.core(k).cq();
        let qqt = core_times_adjoint_core(&q2_core, &q2_core);
        let mut any_checked2 = false;
        for l in space.bond_sectors(k) {
            let n = new_row_ranks[l];
            if n == 0 {
                continue;
            }
            let block = qqt
                .block(l)
                .unwrap_or_else(|| panic!("cq: missing block at {l:?}"));
            assert_eq!((block.nrows(), block.ncols()), (n, n));
            any_checked2 = true;
            for i in 0..n {
                for j in 0..n {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let got = *block.get(i, j);
                    assert!((got - expected).abs() < 1e-8, "cq: Q·Qᴴ at {l:?} ({i},{j})");
                }
            }
        }
        assert!(
            any_checked2,
            "core {k}: cq vacuous — no positive-rank row sector"
        );
    }
}

#[test]
fn pg3_two_site_contract_factor_qc_round_trip() {
    let (_pg, space) = pg3_space_d6();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(303_024);
    let ranks = capped_ranks(&space, 1, 3, &mut rng);
    let x: TtVector<f64> = tt_randn(space, &ranks, &mut rng);
    assert!(x.check());
    let d = space.d;

    for k in 0..d - 1 {
        let tc = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&tc);
        let tc2 = contract(&xk, &xk1);

        let mut any_nonempty = false;
        for m in space.bond_sectors(k + 1) {
            match (tc.block(m), tc2.block(m)) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert_eq!(
                        (a.nrows(), a.ncols()),
                        (b.nrows(), b.ncols()),
                        "k={k} mid {m:?}"
                    );
                    if a.nrows() > 0 && a.ncols() > 0 {
                        any_nonempty = true;
                    }
                    for j in 0..a.ncols() {
                        for i in 0..a.nrows() {
                            let diff = (*a.get(i, j) - *b.get(i, j)).abs_s();
                            assert!(
                                diff < 1e-9,
                                "k={k} mid {m:?} ({i},{j}): round trip mismatch"
                            );
                        }
                    }
                }
                other => panic!("k={k} mid {m:?}: presence mismatch: {other:?}"),
            }
        }
        assert!(any_nonempty, "k={k}: vacuous — no nonempty mid block found");
    }
}

#[test]
fn pg3_dot_and_dot_orth_match_dense() {
    let (_pg, space) = pg3_space_d6();
    let x: TtVector<f64> = capped_randn(space, 1, 3, 303_030);
    let y: TtVector<f64> = capped_randn(space, 1, 3, 303_031);

    let want = dense_dot(&x.to_dense(), &y.to_dense());
    assert!(want.abs() > 0.0, "vacuous: dense dot is zero");

    let got = dot(&x, &y);
    assert!(
        (got - want).abs() <= 1e-8 * want.abs(),
        "dot {got} vs dense {want}"
    );

    let got_orth = dot_orth(&x, &y);
    assert!(
        (got_orth - want).abs() <= 1e-8 * want.abs(),
        "dot_orth {got_orth} vs dense {want}"
    );
}
