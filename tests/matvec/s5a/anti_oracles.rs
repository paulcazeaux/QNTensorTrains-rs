// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn s5a_h1_plain_adjoint_breaks_onesite_hermiticity() {
    let mut rng = seeded(0x5a20);
    let space = Su2Space::new(6, 2, 0);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (1, 2, -0.5)]);
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(6), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 3, &mut rng);
    x.right_orthogonalize(true);
    let fl = Su2Sketch::<f64>::identity(space, 0, 1);

    let good = right_to_left_chain(&h, &x);
    let bad = right_to_left_chain_plain(&h, &x);

    let (fr_good, fr_bad) = (&good[1], &bad[1]);
    let mut delta = 0.0f64;
    for s in fr_good.sectors() {
        if let (Some(a), Some(b)) = (fr_good.block(s), fr_bad.block(s)) {
            for j in 0..a.ncols() {
                for i in 0..a.nrows() {
                    delta = delta.max((*a.get(i, j) - *b.get(i, j)).abs());
                }
            }
        }
    }
    assert!(
        delta > 1e-6,
        "h1: non-vacuity — the θ²-weighted and plain-adjoint right frames must differ at this \
         size, else the anti-oracle gates nothing (max |Δ| = {delta})"
    );

    let ws = su2_onesite_workspace(&h, x.core(0));
    let lop_bad = |v: &Su2Core<f64>| h.sketched_hamiltonian(v, &fl, fr_bad, &ws);
    let mut big = Su2Core::<f64>::zeros(space, 0, &x.core(0).row_ranks, &x.core(0).col_ranks);
    big.fill_randn(&mut rng);
    let mut big2 = Su2Core::<f64>::zeros(space, 0, &x.core(0).row_ranks, &x.core(0).col_ranks);
    big2.fill_randn(&mut seeded(0x5a20 + 1));
    let lx = lop_bad(&big);
    let ly = lop_bad(&big2);
    assert!(
        InnerProductVector::norm(&ly) > 1e-6,
        "h1: non-vacuity — the sabotaged Lop(y) should not be the zero operator"
    );
    let asym = (big.dot_center(&ly) - lx.dot_center(&big2)).abs();
    let scale = InnerProductVector::norm(&big) * InnerProductVector::norm(&big2);
    assert!(
        asym > 1e-6 * scale,
        "h1: THE ANTI-ORACLE DID NOT FIRE — the plain-adjoint (`rproject_plain`) right frame still \
         yielded a self-adjoint one-site operator (asymmetry {asym} vs scale {scale}). Either the \
         θ² weights are not load-bearing at this size (non-vacuity (i) says otherwise) or \
         `sketching_step_left` is not actually routing through the weighted `rproject`."
    );
}

#[test]
fn s5a_h1b_plain_adjoint_breaks_twosite_hermiticity() {
    let mut rng = seeded(0x5a31);
    let space = Su2Space::new(6, 4, 2);
    let t = ha_tensor::<f64>(6, &[(0, 1, -1.0), (1, 2, -0.5)]);
    let h = Su2SparseHamiltonian::new(space, &t, &zero_v(6), None, 1e-12);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    x.right_orthogonalize(true);
    let fl = Su2Sketch::<f64>::identity(space, 0, 1);

    let good = right_to_left_chain(&h, &x);
    let bad = right_to_left_chain_plain(&h, &x);
    let (fr_good, fr_bad) = (&good[2], &bad[2]);

    let mut delta = 0.0f64;
    for s in fr_good.sectors() {
        if let (Some(a), Some(b)) = (fr_good.block(s), fr_bad.block(s)) {
            for j in 0..a.ncols() {
                for i in 0..a.nrows() {
                    delta = delta.max((*a.get(i, j) - *b.get(i, j)).abs());
                }
            }
        }
    }
    assert!(
        delta > 1e-6,
        "h1b: non-vacuity — the θ²-weighted and plain-adjoint right frames must differ at (6,4,2) \
         (max |Δ| = {delta})"
    );

    let tc = contract(x.core(0), x.core(1));
    let lop_bad = |v: &Su2ContractedCores<f64>| h.two_site_sketched_hamiltonian(v, &fl, fr_bad);
    let mut big = tc.zeros_like();
    big.for_each_coeff_mut(|v| *v = f64::sample_standard(&mut rng));
    let mut rng2 = seeded(0x5a31 + 1);
    let mut big2 = tc.zeros_like();
    big2.for_each_coeff_mut(|v| *v = f64::sample_standard(&mut rng2));
    let lx = lop_bad(&big);
    let ly = lop_bad(&big2);
    assert!(
        ly.norm() > 1e-6,
        "h1b: non-vacuity — the sabotaged Lop(y) should not be the zero operator"
    );
    let asym = (big.dot_cores(&ly) - lx.dot_cores(&big2)).abs();
    let scale = big.norm() * big2.norm();
    assert!(
        asym > 1e-6 * scale,
        "h1b: THE ANTI-ORACLE DID NOT FIRE — the plain-adjoint (`rproject_plain`) right frame still \
         yielded a self-adjoint TWO-SITE operator (asymmetry {asym} vs scale {scale}). Either the \
         θ² weights are not load-bearing at this size (non-vacuity (i) says otherwise) or \
         `sketching_step_left` is not routing through the weighted `rproject`."
    );
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Sabotage {
    None,
    NoThetaScale,
    NoThetaUnscale,
    UnweightedChop,
}

type SabSvdData = Vec<(Su2Sector, faer::Mat<f64>, Vec<f64>, faer::Mat<f64>)>;

fn factor_svd_sabotaged(
    tc: &Su2ContractedCores<f64>,
    eps: f64,
    sab: Sabotage,
) -> (Su2Core<f64>, Su2Sketch<f64>, Su2Core<f64>) {
    let space = tc.space;
    let k = tc.k;
    let shell = Su2Core::<f64>::zeros(space, k + 1, &Su2RankGrid::zeros(&space), &tc.col_ranks);

    let data: SabSvdData = tc
        .mid_qn
        .iter()
        .map(|&m| {
            let (u, sv, vt) = match tc.block(m) {
                Some(b) if b.nrows() > 0 && b.ncols() > 0 => {
                    let mut scaled = b.to_owned();
                    if sab != Sabotage::NoThetaScale {
                        shell.scale_horizontal(&mut scaled, m);
                    }
                    qntt::core::factor::svd_uvt_svals(scaled.as_ref())
                }
                _ => (
                    faer::Mat::<f64>::zeros(0, 0),
                    Vec::new(),
                    faer::Mat::<f64>::zeros(0, 0),
                ),
            };
            (m, u, sv, vt)
        })
        .collect();

    let s_blocks = Su2SectorGrid::from_fn(&space, |m| {
        space.contains(k + 1, m).then(|| {
            let sv = data.iter().find(|(s, ..)| *s == m).map(|(_, _, v, _)| v);
            let size = sv.map_or(0, |v| v.len());
            let mut mat = faer::Mat::<f64>::zeros(size, size);
            if let Some(v) = sv {
                for (i, &val) in v.iter().enumerate() {
                    *mat.get_mut(i, i) = val;
                }
            }
            mat
        })
    });
    let s_full = Su2Sketch::new(space, k + 1, s_blocks);
    let mid_rank = if sab == Sabotage::UnweightedChop {
        unweighted_chop_energy(&s_full, eps)
    } else {
        qntt::su2::chop::chop_energy(&s_full, eps, qntt::Pooled::NONE)
    };

    let mut xk = Su2Core::zeros(space, k, &tc.row_ranks, &mid_rank);
    let mut xk1 = Su2Core::zeros(space, k + 1, &mid_rank, &tc.col_ranks);
    let s_out_blocks = Su2SectorGrid::from_fn(&space, |m| {
        space.contains(k + 1, m).then(|| {
            let r = mid_rank[m];
            let sv = data.iter().find(|(s, ..)| *s == m).map(|(_, _, v, _)| v);
            let mut mat = faer::Mat::<f64>::zeros(r, r);
            if let Some(v) = sv {
                for (i, &val) in v.iter().enumerate().take(r) {
                    *mat.get_mut(i, i) = val;
                }
            }
            mat
        })
    });
    let s_out = Su2Sketch::new(space, k + 1, s_out_blocks);

    for (m, u, _sv, vt) in &data {
        let r = mid_rank[*m];
        if r > 0 {
            xk.scatter_vertical(*m, u.submatrix(0, 0, u.nrows(), r));
            let mut vt_r = vt.submatrix(0, 0, r, vt.ncols()).to_owned();
            if sab != Sabotage::NoThetaUnscale {
                xk1.unscale_horizontal(&mut vt_r, *m);
            }
            xk1.scatter_horizontal(*m, vt_r.as_ref());
        }
    }
    (xk, s_out, xk1)
}

fn unweighted_chop_energy(sv: &Su2Sketch<f64>, eps: f64) -> Su2RankGrid {
    let mut all: Vec<(f64, Su2Sector)> = Vec::new();
    for s in sv.sectors() {
        if let Some(b) = sv.block(s) {
            for i in 0..b.nrows() {
                all.push((*b.get(i, i), s));
            }
        }
    }
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut dropped_energy = 0.0f64;
    let mut cutoff = 0.0f64;
    for (val, _) in &all {
        if dropped_energy + val * val > eps * eps {
            break;
        }
        dropped_energy += val * val;
        cutoff = *val;
    }
    let mut ranks = Su2RankGrid::zeros(&sv.space);
    for s in sv.sectors() {
        if let Some(b) = sv.block(s) {
            ranks[s] = (0..b.nrows()).filter(|&i| *b.get(i, i) > cutoff).count();
        }
    }
    ranks
}

#[test]
fn s5a_h2a_factor_svd_weights_are_load_bearing() {
    let mut rng = seeded(0x5a32);
    let space = Su2Space::new(6, 4, 2);
    let mut x: Su2TtVector<f64> = randomized_su2_state(space, 3, 5, &mut rng);
    let k = 2usize;
    x.move_core(k, false);
    let tc = contract(x.core(k), x.core(k + 1));
    let ref_norm = tc.norm();
    let ep = 1e-1;

    let reconstruct = |sab: Sabotage| -> f64 {
        let (xa, s, xb) = factor_svd_sabotaged(&tc, ep, sab);
        let mut xb2 = xb;
        xb2.fold_sketch_left_mut(&s);
        let re = contract(&xa, &xb2);
        let mut diff = tc.clone();
        diff.axpy(-1.0, &re);
        diff.norm()
    };

    let (sa, ss, sb) = factor_svd(&tc, ep, qntt::Pooled::NONE, false);
    let mut sb2 = sb;
    sb2.fold_sketch_left_mut(&ss);
    let shipped = contract(&sa, &sb2);
    let (ca, cs, cb) = factor_svd_sabotaged(&tc, ep, Sabotage::None);
    let mut cb2 = cb;
    cb2.fold_sketch_left_mut(&cs);
    let control = contract(&ca, &cb2);
    for &m in &tc.mid_qn {
        if let (Some(a), Some(b)) = (shipped.block(m), control.block(m)) {
            let (d, r) = block_diff_and_norm(a, b);
            assert!(
                d <= 1e-12 * r.max(1.0),
                "h2a: the test-local control arm does not reproduce the shipped factor_svd at \
                 {m:?} — the sabotage results below would not be attributable"
            );
        }
    }
    let bound = ep * ref_norm;
    assert!(
        reconstruct(Sabotage::None) <= bound,
        "h2a: the control arm must satisfy the round bound"
    );

    for sab in [Sabotage::NoThetaScale, Sabotage::NoThetaUnscale] {
        let err = reconstruct(sab);
        assert!(
            err > bound,
            "h2a: THE ANTI-ORACLE DID NOT FIRE for {sab:?} — the sabotaged split still satisfied \
             the weighted round bound (err={err}, bound={bound}), so that convention is not \
             load-bearing at this size and B3 gates nothing about it"
        );
    }

    let ep_split = 3e-1;
    let (_wa, w_s, _wb) = factor_svd(&tc, ep_split, qntt::Pooled::NONE, false);
    let (_ua, u_s, _ub) = factor_svd_sabotaged(&tc, ep_split, Sabotage::UnweightedChop);
    let w_ranks: Vec<usize> = tc
        .mid_qn
        .iter()
        .map(|&m| w_s.block(m).map_or(0, |b| b.nrows()))
        .collect();
    let u_ranks: Vec<usize> = tc
        .mid_qn
        .iter()
        .map(|&m| u_s.block(m).map_or(0, |b| b.nrows()))
        .collect();
    assert_ne!(
        w_ranks, u_ranks,
        "h2a: THE ANTI-ORACLE DID NOT FIRE for UnweightedChop — the centerweighted and raw chops \
         selected the SAME mid ranks at ep={ep_split}, so the weighting does not decide which \
         multiplets survive at this size and the chop metric is untested"
    );
    assert!(
        w_ranks.iter().sum::<usize>() < u_ranks.iter().sum::<usize>(),
        "h2a: the centerweighted chop should retain fewer multiplets than the raw-σ chop at \
         ep={ep_split} ({w_ranks:?} vs {u_ranks:?})"
    );
}

#[test]
fn s5a_h3a_mid_vs_successor_centerweight() {
    let mut rng = seeded(0x5a34);
    let space = Su2Space::new(7, 5, 1);
    let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let tc = contract(x.core(1), x.core(2));
    let correct = tc.dot_cores(&tc).re_s();
    let mut wrong = 0.0f64;
    for &m in &tc.mid_qn {
        if let Some(b) = tc.block(m) {
            let cw = qntt::su2::centerweight(m.two_s, space.two_s);
            let mut blk = 0.0f64;
            for j in 0..b.ncols() {
                for i in 0..b.nrows() {
                    blk += b.get(i, j) * b.get(i, j);
                }
            }
            wrong += cw * blk;
        }
    }
    assert!(
        (correct - wrong).abs() > 1e-3 * correct.abs().max(1.0),
        "h3a: mid-weighted dot should differ from the correct successor-weighted dot_cores"
    );
}
