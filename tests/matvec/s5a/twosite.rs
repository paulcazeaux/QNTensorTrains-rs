// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5a_b0_contract_matches_expand() {
    let mut rng = seeded(0x5a10);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    for k in [0usize, 2, 4] {
        let tc = contract(x.core(k), x.core(k + 1));
        let (xa, xb) = factor_qc(&tc);
        let mut y = x.clone();
        y.set_cores(xa, xb);
        let ex = expand_dense(&x);
        let ey = expand_dense(&y);
        let nrm = dense_norm(&ex).max(1.0);
        assert!(
            dense_diff_norm(&ex, &ey) <= 1e-12 * nrm,
            "b0: contract/factor_qc roundtrip diverges from expand at k={k}"
        );
    }
}

#[test]
fn s5a_b0b_block_offset_pin() {
    let mut rng = seeded(0x5a10 + 1);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
    let k = 2usize;
    let tc = contract(x.core(k), x.core(k + 1));
    for &m in &tc.mid_qn {
        let Some(blk) = tc.block(m) else { continue };
        if blk.nrows() == 0 || blk.ncols() == 0 {
            continue;
        }
        let mut roff = 0usize;
        for fi in Su2Family::ALL {
            let (dn, dt) = fi.delta();
            if m.n < dn {
                continue;
            }
            let src_two_s = m.two_s as i64 - dt;
            if src_two_s < 0 {
                continue;
            }
            let l = Su2Sector::new(m.n - dn, src_two_s as usize);
            if !space.contains(k, l) {
                continue;
            }
            let r_l = x.core(k).row_ranks[l];
            let mut coff = 0usize;
            for fj in Su2Family::ALL {
                let Some(r) = fj.apply(m, space.pg.irrep_of(k + 1)) else {
                    continue;
                };
                if !space.contains(k + 2, r) {
                    continue;
                }
                let r_r = x.core(k + 1).col_ranks[r];
                if r_l > 0
                    && r_r > 0
                    && let (Some(a), Some(b)) = (x.core(k).block(fi, l), x.core(k + 1).block(fj, m))
                {
                    for jj in 0..r_r {
                        for ii in 0..r_l {
                            let mut want = 0.0f64;
                            for p in 0..a.ncols() {
                                want += a.get(ii, p) * b.get(p, jj);
                            }
                            let got = *blk.get(roff + ii, coff + jj);
                            assert!(
                                (want - got).abs() <= 1e-11 * (want.abs().max(1.0)),
                                "b0b: block offset mismatch at m={m:?} fi={fi:?} fj={fj:?}"
                            );
                        }
                    }
                }
                coff += r_r;
            }
            roff += r_l;
        }
    }
}

#[test]
fn s5a_b1_factor_svd_ep0_lossless() {
    let mut rng = seeded(0x5a11);
    let space = Su2Space::new(6, 2, 0);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    x.right_orthogonalize(true);
    let tc = contract(x.core(0), x.core(1));
    let (xa, s, xb) = factor_svd(&tc, 0.0, qntt::Pooled::NONE, false);
    let mut xb2 = xb;
    xb2.fold_sketch_left_mut(&s);
    let re = contract(&xa, &xb2);
    let mut compared = 0usize;
    for &m in &tc.mid_qn {
        let Some(want) = tc.block(m) else { continue };
        if want.nrows() == 0 || want.ncols() == 0 {
            continue;
        }
        assert_eq!(
            s.block(m).map_or(0, |b| b.nrows()),
            want.nrows().min(want.ncols()),
            "b1: ep=0 silently truncated at mid sector {m:?} — the lossless claim would be luck"
        );
        let got = re
            .block(m)
            .unwrap_or_else(|| panic!("b1: reconstruction dropped mid block {m:?}"));
        let (diff, refn) = block_diff_and_norm(want, got);
        assert!(
            diff <= 1e-12 * refn.max(1.0),
            "b1: ep=0 factor_svd reconstruction diverges at mid sector {m:?} (diff={diff}, ||ref||={refn})"
        );
        compared += 1;
    }
    assert!(compared > 0, "b1: vacuous — no mid block was compared");
}

#[test]
fn s5a_b2_factor_svd_ep0_lossless_weighted() {
    let mut rng = seeded(0x5a12);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    let tc = contract(x.core(0), x.core(1));
    let n_theta: usize = tc
        .mid_qn
        .iter()
        .map(|&m| non_vacuous_theta_count(&space, 0, m))
        .sum();
    assert!(
        n_theta >= 2,
        "b2: vacuous — theta is trivial everywhere at this size"
    );
    let (xa, s, xb) = factor_svd(&tc, 0.0, qntt::Pooled::NONE, false);
    let mut xb2 = xb;
    xb2.fold_sketch_left_mut(&s);
    let re = contract(&xa, &xb2);
    let mut compared = 0usize;
    for &m in &tc.mid_qn {
        let Some(want) = tc.block(m) else { continue };
        if want.nrows() == 0 || want.ncols() == 0 {
            continue;
        }
        assert_eq!(
            s.block(m).map_or(0, |b| b.nrows()),
            want.nrows().min(want.ncols()),
            "b2: ep=0 silently truncated at mid sector {m:?} — the lossless claim would be luck"
        );
        let got = re
            .block(m)
            .unwrap_or_else(|| panic!("b2: reconstruction dropped mid block {m:?}"));
        let (diff, refn) = block_diff_and_norm(want, got);
        assert!(
            diff <= 1e-12 * refn.max(1.0),
            "b2: weighted ep=0 factor_svd reconstruction diverges at mid sector {m:?} (diff={diff}, ||ref||={refn})"
        );
        compared += 1;
    }
    assert!(compared > 0, "b2: vacuous — no mid block was compared");
}

#[test]
fn s5a_b9_factor_svd_ep0_lossless_complex() {
    let mut rng = seeded(0x5a19);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<C64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    let tc = contract(x.core(0), x.core(1));
    let max_im = (0..tc.num_coeffs())
        .map(|i| tc.coeff(i).im.abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_im > 1e-6,
        "b9: vacuous — the contracted pair carries no imaginary part"
    );
    let n_theta: usize = tc
        .mid_qn
        .iter()
        .map(|&m| non_vacuous_theta_count(&space, 0, m))
        .sum();
    assert!(n_theta >= 2, "b9: vacuous — theta is trivial at this size");
    let (xa, s, xb) = factor_svd(&tc, 0.0, qntt::Pooled::NONE, false);
    let mut xb2 = xb;
    xb2.fold_sketch_left_mut(&s);
    let re = contract(&xa, &xb2);
    let mut compared = 0usize;
    for &m in &tc.mid_qn {
        let Some(want) = tc.block(m) else { continue };
        if want.nrows() == 0 || want.ncols() == 0 {
            continue;
        }
        let got = re
            .block(m)
            .unwrap_or_else(|| panic!("b9: reconstruction dropped mid block {m:?}"));
        let (diff, refn) = block_diff_and_norm(want, got);
        assert!(
            diff <= 1e-12 * refn.max(1.0),
            "b9: complex ep=0 factor_svd reconstruction diverges at mid sector {m:?} — a \
             transpose-for-adjoint slip in the Vt θ-unscale is the prime suspect (diff={diff})"
        );
        compared += 1;
    }
    assert!(compared > 0, "b9: vacuous — no mid block was compared");
}

#[test]
fn s5a_b3_truncation_round_bound() {
    let mut rng = seeded(0x5a13);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
    let k = 2usize;
    x.move_core(k, false);
    let tc = contract(x.core(k), x.core(k + 1));
    let ref_norm = tc.dot_cores(&tc).re_s().sqrt();

    let phys_norm = dense_norm(&expand_dense(&x));
    assert!(
        (ref_norm - phys_norm).abs() <= 1e-10 * phys_norm.max(1.0),
        "b3: the dot_cores metric norm {ref_norm} disagrees with the physical ||expand(x)|| \
         {phys_norm} — the centerweight argument (successor 2s vs mid 2s) is the prime suspect"
    );

    for eps in [1e-1, 1e-3] {
        let (xa, s, xb) = factor_svd(&tc, eps, qntt::Pooled::NONE, false);
        let kept: usize = tc
            .mid_qn
            .iter()
            .map(|&m| s.block(m).map_or(0, |b| b.nrows()))
            .sum();
        let full: usize = tc
            .mid_qn
            .iter()
            .map(|&m| tc.block(m).map_or(0, |b| b.nrows().min(b.ncols())))
            .sum();
        if eps >= 1e-1 {
            assert!(
                kept < full,
                "b3: vacuous at eps={eps} — no truncation fired (kept {kept} of {full}); the round \
                 bound below would then hold with err==0 regardless of the weights"
            );
        }
        let mut xb2 = xb;
        xb2.fold_sketch_left_mut(&s);
        let re = contract(&xa, &xb2);
        let mut diff = tc.clone();
        diff.axpy(-1.0, &re);
        let err = diff.dot_cores(&diff).re_s().sqrt();
        assert!(
            err <= eps * ref_norm.max(1e-12),
            "b3: weighted truncation error {err} exceeds the eps*||ref|| bound at eps={eps} \
             (bound={})",
            eps * ref_norm.max(1e-12)
        );
    }
}

#[test]
fn s5a_b5_factor_svd_normalize() {
    let mut rng = seeded(0x5a15);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    let tc = contract(x.core(0), x.core(1));
    let (_xa, s, _xb) = factor_svd(&tc, 0.0, qntt::Pooled::NONE, true);
    assert!(
        (s.norm() - 1.0).abs() <= 1e-13,
        "b5: normalize=true should give ||S||_centerweighted == 1, got {}",
        s.norm()
    );
}

#[test]
fn s5a_b6_factor_svd_max_rank_cap() {
    let mut rng = seeded(0x5a16);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
    x.right_orthogonalize(true);
    let k = 2usize;
    x.move_core(k, false);
    let tc = contract(x.core(k), x.core(k + 1));
    let (_ua, s_free, _ub) = factor_svd(&tc, 0.0, qntt::Pooled::NONE, false);
    let free_max = s_free
        .sectors()
        .into_iter()
        .filter_map(|m| s_free.block(m).map(|b| b.nrows()))
        .max()
        .unwrap_or(0);
    assert!(
        free_max > 2,
        "b6: vacuous — the uncapped mid ranks already max out at {free_max} <= 2, so max_rank=2 \
         never fires at this size"
    );
    let (_xa, s, _xb) = factor_svd(&tc, 0.0, qntt::Pooled(2), false);
    for m in s.sectors() {
        if let Some(b) = s.block(m) {
            assert!(
                b.nrows() <= 2,
                "b6: max_rank=2 violated at mid sector {m:?}: rank {}",
                b.nrows()
            );
        }
    }
}

#[test]
fn s5a_b7_factor_qc_lossless_and_unweighted_left_canonical() {
    let mut rng = seeded(0x5a17);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let tc = contract(x.core(0), x.core(1));
    let (xa, xb) = factor_qc(&tc);
    let re = contract(&xa, &xb);
    let mut compared = 0usize;
    for &m in &tc.mid_qn {
        let Some(want) = tc.block(m) else { continue };
        if want.nrows() == 0 || want.ncols() == 0 {
            continue;
        }
        let got = re
            .block(m)
            .unwrap_or_else(|| panic!("b7: reconstruction dropped mid block {m:?}"));
        let (diff, refn) = block_diff_and_norm(want, got);
        assert!(
            diff <= 1e-12 * refn.max(1.0),
            "b7: factor_qc/contract roundtrip diverges at {m:?} (diff={diff}, ||ref||={refn})"
        );
        compared += 1;
    }
    assert!(compared > 0, "b7: vacuous — no mid block was compared");
    let mut rank_sum = 0usize;
    for &m in &tc.mid_qn {
        let r = xa.col_ranks[m];
        if let Some(b) = tc.block(m) {
            assert!(
                r <= b.nrows().min(b.ncols()),
                "b7: revealed mid rank {r} exceeds min(rows,cols) at {m:?}"
            );
        }
        rank_sum += r;
    }
    assert!(
        rank_sum > 0,
        "b7: vacuous — factor_qc revealed rank 0 everywhere"
    );
    let mut checked_cols = 0usize;
    for r in space.bond_sectors(1) {
        let v = xa.vertical_unfolding(r);
        if v.ncols() == 0 {
            continue;
        }
        checked_cols += v.ncols();
        for i in 0..v.ncols() {
            for j in 0..v.ncols() {
                let mut acc = 0.0f64;
                for row in 0..v.nrows() {
                    acc += v.get(row, i) * v.get(row, j);
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - want).abs() <= 1e-11,
                    "b7: Xa not left-orthonormal (unweighted) at col sector {r:?}"
                );
            }
        }
    }
    assert!(
        checked_cols > 0,
        "b7: vacuous — no non-empty vertical unfolding to check left-canonicity on"
    );
}

#[test]
fn s5a_b8_dot_cores_metric_shape() {
    let mut rng = seeded(0x5a18);
    let space = Su2Space::new(6, 4, 2);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let tcx = contract(x.core(0), x.core(1));
    let mut tcy = Su2ContractedCores::<f64>::zeros(space, 0, &tcx.row_ranks, &tcx.col_ranks);
    for &m in tcy.mid_qn.clone().iter() {
        if let Some(mut b) = tcy.block_mut(m) {
            let (nr, nc) = (b.nrows(), b.ncols());
            for j in 0..nc {
                for i in 0..nr {
                    *b.as_mut().get_mut(i, j) = f64::sample_standard(&mut rng);
                }
            }
        }
    }
    let d_xy = tcx.dot_cores(&tcy);
    let d_yx = tcy.dot_cores(&tcx);
    assert!(
        (d_xy - d_yx).abs() <= 1e-12 * d_xy.abs().max(1.0),
        "b8: dot_cores not conjugate symmetric"
    );

    let n2 = tcx.dot_cores(&tcx).re_s();
    let nrm = tcx.norm();
    assert!(nrm > 0.0, "b8: norm(X) should be strictly positive");
    assert!(
        (nrm * nrm - n2).abs() <= 1e-13 * n2.abs().max(1.0),
        "b8: norm()² must equal Re dot_cores(X,X) ({} vs {n2})",
        nrm * nrm
    );

    let mut ref_dot = 0.0f64;
    let mut unweighted = 0.0f64;
    for &m in &tcx.mid_qn {
        let (Some(a), Some(b)) = (tcx.block(m), tcy.block(m)) else {
            continue;
        };
        let mut off = 0usize;
        for f in Su2Family::ALL {
            let Some(r) = f.apply(m, space.pg.irrep_of(1)) else {
                continue;
            };
            if !space.contains(2, r) {
                continue;
            }
            let w = tcx.col_ranks[r];
            let cw = qntt::su2::centerweight(r.two_s, space.two_s);
            for j in off..off + w {
                for i in 0..a.nrows() {
                    ref_dot += cw * a.get(i, j) * b.get(i, j);
                    unweighted += a.get(i, j) * b.get(i, j);
                }
            }
            off += w;
        }
    }
    assert!(
        (ref_dot - d_xy).abs() <= 1e-13 * d_xy.abs().max(1.0),
        "b8: dot_cores disagrees with the independent per-column-slice successor-weighted \
         reference ({d_xy} vs {ref_dot})"
    );
    assert!(
        (ref_dot - unweighted).abs() > 1e-3 * ref_dot.abs().max(1.0),
        "b8: vacuous — the successor-weighted metric coincides with the unweighted Frobenius \
         product at this size ({ref_dot} vs {unweighted})"
    );

    let mut mid_weighted = 0.0f64;
    for &m in &tcx.mid_qn {
        let (Some(a), Some(b)) = (tcx.block(m), tcy.block(m)) else {
            continue;
        };
        let cw = qntt::su2::centerweight(m.two_s, space.two_s);
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                mid_weighted += cw * a.get(i, j) * b.get(i, j);
            }
        }
    }
    assert!(
        (mid_weighted - d_xy).abs() > 1e-3 * d_xy.abs().max(1.0),
        "b8: the mid-weighted variant coincides with dot_cores — the successor-vs-mid centerweight \
         distinction is not observable here"
    );
}
