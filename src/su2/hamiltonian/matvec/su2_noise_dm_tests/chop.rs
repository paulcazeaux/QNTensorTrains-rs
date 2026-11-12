// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn forward_chop_matches_the_augmented_svd_route() {
    let (h, space) = h4_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5b_c0de);
    let x = randomized_state(space, 3, 5, &mut rng);
    let k = 1usize;
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = sketch_h_mul(&fl, &h, &xk);
    let noise_dm = noise_density_matrix(&h, &half, &xk, &xk1, true);

    let alpha = 3e-2;
    let (_xk_pert, _s_pert, xk1_pert, _info) = factor_svd_perturbative(
        &cp,
        &noise_dm,
        1e-10,
        crate::qn::Pooled::NONE,
        true,
        alpha,
        true,
    );

    let shell = Su2Core::<f64>::zeros(space, k + 1, &Su2RankGrid::zeros(&space), &cp.col_ranks);
    let mut nd_norm2: f64 = 0.0;
    for &m in &cp.mid_qn {
        let nd = &noise_dm[m];
        for j in 0..nd.ncols() {
            for i in 0..nd.nrows() {
                nd_norm2 += nd[(i, j)] * nd[(i, j)];
            }
        }
    }
    let alpha_eff = if nd_norm2 > 0.0 {
        alpha / nd_norm2.sqrt()
    } else {
        0.0
    };

    let mut sv_blocks: Su2SectorGrid<Option<Mat<f64>>> = Su2SectorGrid::from_fn(&space, |_| None);
    for &m in &cp.mid_qn {
        let b = match cp.block(m) {
            Some(b) if b.nrows() > 0 && b.ncols() > 0 => b,
            _ => continue,
        };
        let mut b_theta = b.to_owned();
        shell.scale_horizontal(&mut b_theta, m);
        let g = b_theta.nrows();
        let mut rho = Mat::<f64>::zeros(g, g);
        matmul(
            rho.as_mut(),
            Accum::Replace,
            b_theta.as_ref(),
            b_theta.adjoint(),
            1.0,
            Par::Seq,
        );
        let nd = &noise_dm[m];
        if nd.nrows() == g && nd.ncols() == g {
            for j in 0..g {
                for i in 0..g {
                    let v = rho[(i, j)] + alpha_eff * nd[(i, j)];
                    *rho.get_mut(i, j) = v;
                }
            }
        }
        let (evals, _u) = dense_self_adjoint_eigen(rho.as_ref());
        let lam_max = evals.last().copied().unwrap_or(0.0).max(0.0);
        let floor = lam_max * (evals.len() as f64) * f64::EPSILON;
        let mut eig_desc: Vec<f64> = evals
            .iter()
            .map(|&v| if v > floor { v.sqrt() } else { 0.0 })
            .collect();
        eig_desc.reverse();

        let e_theta = if nd.nrows() == g && nd.ncols() == g && g > 0 {
            let (nvals, nu) = dense_self_adjoint_eigen(nd.as_ref());
            let mut sq = Mat::<f64>::zeros(g, g);
            for c in 0..g {
                let lam = nvals[c].max(0.0).sqrt();
                for r in 0..g {
                    *sq.get_mut(r, c) = nu[(r, c)] * lam;
                }
            }
            sq
        } else {
            Mat::<f64>::zeros(g, 0)
        };
        let mut aug = Mat::<f64>::zeros(g, b_theta.ncols() + e_theta.ncols());
        for j in 0..b_theta.ncols() {
            for i in 0..g {
                *aug.get_mut(i, j) = b_theta[(i, j)];
            }
        }
        for j in 0..e_theta.ncols() {
            for i in 0..g {
                *aug.get_mut(i, b_theta.ncols() + j) = alpha_eff.sqrt() * e_theta[(i, j)];
            }
        }
        let (_u_aug, sv_aug, _vt_aug) = svd_uvt_svals(aug.as_ref());

        assert_eq!(
            eig_desc.len(),
            sv_aug.len(),
            "spectrum length mismatch at mid {m:?}"
        );
        for (a, bv) in eig_desc.iter().zip(sv_aug.iter()) {
            assert!(
                (a - bv).abs() <= 1e-8 * a.max(*bv).max(1.0),
                "forward ρ' spectrum via eigen ({a}) vs the augmented-SVD route ({bv}) \
                 disagree at mid {m:?} — factor_svd_perturbative's own ρ'/chop plumbing"
            );
        }
        let mut mat = Mat::<f64>::zeros(eig_desc.len(), eig_desc.len());
        for (i, &v) in eig_desc.iter().enumerate() {
            *mat.get_mut(i, i) = v;
        }
        sv_blocks[m] = Some(mat);
    }
    let s_full = Su2Sketch::new(space, k + 1, sv_blocks);
    let mid_rank_oracle = chop_energy(&s_full, 1e-10, crate::qn::Pooled::NONE);

    for &m in &cp.mid_qn {
        assert_eq!(
            mid_rank_oracle[m], xk1_pert.row_ranks[m],
            "kept rank mismatch at mid {m:?}: factor_svd_perturbative's chop diverges from \
             an independently recomputed chop_energy over the same (eigen-route-pinned) \
             augmented spectrum"
        );
    }
}

#[test]
fn back_chop_matches_the_augmented_svd_route() {
    let (h, space) = h4_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5b_c0de + 2);
    let x = randomized_state(space, 3, 5, &mut rng);
    let k = 1usize;
    let fr = right_to_left_sketching(&h, &x)
        .into_iter()
        .nth(k + 1)
        .unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = h_sketch_mul(&h, &xk1, &fr);
    let noise_dm = noise_density_matrix(&h, &half, &xk, &xk1, false);

    let mut any_applied = false;
    for &m in &cp.mid_qn {
        let nd = &noise_dm[m];
        if let Some(b) = cp.block(m)
            && nd.nrows() == b.ncols()
            && nd.ncols() == b.ncols()
            && nd.nrows() > 0
        {
            any_applied = true;
        }
    }
    assert!(
        any_applied,
        "no mid sector had a shape-matching ρ_noise at this interior back bond — \
         the guard in factor_svd_perturbative would silently drop all back noise here"
    );

    let alpha = 3e-2;
    let (xk_pert, _s_pert, _xk1_pert, _info) = factor_svd_perturbative(
        &cp,
        &noise_dm,
        1e-10,
        crate::qn::Pooled::NONE,
        true,
        alpha,
        false,
    );

    let shell = Su2Core::<f64>::zeros(space, k + 1, &Su2RankGrid::zeros(&space), &cp.col_ranks);
    let mut nd_norm2: f64 = 0.0;
    for &m in &cp.mid_qn {
        let nd = &noise_dm[m];
        for j in 0..nd.ncols() {
            for i in 0..nd.nrows() {
                nd_norm2 += nd[(i, j)] * nd[(i, j)];
            }
        }
    }
    let alpha_eff = if nd_norm2 > 0.0 {
        alpha / nd_norm2.sqrt()
    } else {
        0.0
    };

    let mut sv_blocks: Su2SectorGrid<Option<Mat<f64>>> = Su2SectorGrid::from_fn(&space, |_| None);
    for &m in &cp.mid_qn {
        let b = match cp.block(m) {
            Some(b) if b.nrows() > 0 && b.ncols() > 0 => b,
            _ => continue,
        };
        let mut b_theta = b.to_owned();
        shell.scale_horizontal(&mut b_theta, m);
        let g = b_theta.ncols();
        let mut rho = Mat::<f64>::zeros(g, g);
        matmul(
            rho.as_mut(),
            Accum::Replace,
            b_theta.adjoint(),
            b_theta.as_ref(),
            1.0,
            Par::Seq,
        );
        let nd = &noise_dm[m];
        if nd.nrows() == g && nd.ncols() == g {
            for j in 0..g {
                for i in 0..g {
                    let v = rho[(i, j)] + alpha_eff * nd[(i, j)];
                    *rho.get_mut(i, j) = v;
                }
            }
        }
        let (evals, _u) = dense_self_adjoint_eigen(rho.as_ref());
        let lam_max = evals.last().copied().unwrap_or(0.0).max(0.0);
        let floor = lam_max * (evals.len() as f64) * f64::EPSILON;
        let mut eig_desc: Vec<f64> = evals
            .iter()
            .map(|&v| if v > floor { v.sqrt() } else { 0.0 })
            .collect();
        eig_desc.reverse();

        let e_theta = if nd.nrows() == g && nd.ncols() == g && g > 0 {
            let (nvals, nu) = dense_self_adjoint_eigen(nd.as_ref());
            let mut sq = Mat::<f64>::zeros(g, g);
            for i in 0..g {
                let lam = nvals[i].max(0.0).sqrt();
                for j in 0..g {
                    *sq.get_mut(i, j) = nu[(j, i)] * lam;
                }
            }
            sq
        } else {
            Mat::<f64>::zeros(0, g)
        };
        let mut aug = Mat::<f64>::zeros(b_theta.nrows() + e_theta.nrows(), g);
        for j in 0..g {
            for i in 0..b_theta.nrows() {
                *aug.get_mut(i, j) = b_theta[(i, j)];
            }
            for i in 0..e_theta.nrows() {
                *aug.get_mut(b_theta.nrows() + i, j) = alpha_eff.sqrt() * e_theta[(i, j)];
            }
        }
        let (_u_aug, sv_aug, _vt_aug) = svd_uvt_svals(aug.as_ref());

        assert_eq!(
            eig_desc.len(),
            sv_aug.len(),
            "spectrum length mismatch at mid {m:?}"
        );
        for (a, bv) in eig_desc.iter().zip(sv_aug.iter()) {
            assert!(
                (a - bv).abs() <= 1e-8 * a.max(*bv).max(1.0),
                "back ρ' spectrum via eigen ({a}) vs the augmented-SVD route ({bv}) \
                 disagree at mid {m:?} — factor_svd_perturbative's own ρ'/chop plumbing"
            );
        }
        let mut mat = Mat::<f64>::zeros(eig_desc.len(), eig_desc.len());
        for (i, &v) in eig_desc.iter().enumerate() {
            *mat.get_mut(i, i) = v;
        }
        sv_blocks[m] = Some(mat);
    }
    let s_full = Su2Sketch::new(space, k + 1, sv_blocks);
    let mid_rank_oracle = chop_energy(&s_full, 1e-10, crate::qn::Pooled::NONE);

    for &m in &cp.mid_qn {
        assert_eq!(
            mid_rank_oracle[m], xk_pert.col_ranks[m],
            "kept rank mismatch at mid {m:?}: factor_svd_perturbative's back chop diverges \
             from an independently recomputed chop_energy over the same \
             (eigen-route-pinned) augmented spectrum"
        );
    }
}

#[test]
fn factor_svd_perturbative_zero_alpha_is_bit_identical_forward_and_back() {
    use crate::core::twosite::noise_rng;
    use crate::su2::twosite::factor_svd_noise;

    let (h, space) = h4_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5b_bee5);
    let x = randomized_state(space, 3, 5, &mut rng);
    let d = space.d;

    for k in 0..d - 1 {
        let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
        let fr = right_to_left_sketching(&h, &x)
            .into_iter()
            .nth(k + 1)
            .unwrap();
        let cp = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&cp);

        for forward in [true, false] {
            let half = if forward {
                sketch_h_mul(&fl, &h, &xk)
            } else {
                h_sketch_mul(&h, &xk1, &fr)
            };
            let noise_dm = noise_density_matrix(&h, &half, &xk, &xk1, forward);

            let (a0, s0, b0, info0) = factor_svd_perturbative(
                &cp,
                &noise_dm,
                1e-10,
                crate::qn::Pooled::NONE,
                true,
                0.0,
                forward,
            );
            let (a1, s1, b1, info1) = factor_svd_noise(
                &cp,
                1e-10,
                crate::qn::Pooled::NONE,
                true,
                0.0,
                &mut noise_rng(0),
            );

            assert_eq!(
                info0.kept, info1.kept,
                "k={k} forward={forward}: kept rank differs"
            );
            assert_eq!(
                info0.discarded_weight, info1.discarded_weight,
                "k={k} forward={forward}: discarded weight differs"
            );
            for m in space.bond_sectors(k) {
                for f in crate::su2::core::Family::ALL {
                    match (a0.block(f, m), a1.block(f, m)) {
                        (Some(x0), Some(x1)) => {
                            assert_eq!(x0.nrows(), x1.nrows());
                            assert_eq!(x0.ncols(), x1.ncols());
                            for j in 0..x0.ncols() {
                                for i in 0..x0.nrows() {
                                    assert_eq!(
                                        x0[(i, j)],
                                        x1[(i, j)],
                                        "k={k} forward={forward} left core {f:?}@{m:?}[{i},{j}]"
                                    );
                                }
                            }
                        }
                        (None, None) => {}
                        _ => panic!(
                            "k={k} forward={forward}: left core block presence differs at \
                             {f:?}@{m:?}"
                        ),
                    }
                }
            }
            for m in space.bond_sectors(k + 1) {
                match (s0.block(m), s1.block(m)) {
                    (Some(x0), Some(x1)) => {
                        for j in 0..x0.ncols() {
                            for i in 0..x0.nrows() {
                                assert_eq!(
                                    x0[(i, j)],
                                    x1[(i, j)],
                                    "k={k} forward={forward} sketch@{m:?}[{i},{j}]"
                                );
                            }
                        }
                    }
                    (None, None) => {}
                    _ => panic!("k={k} forward={forward}: sketch presence differs at {m:?}"),
                }
                for f in crate::su2::core::Family::ALL {
                    match (b0.block(f, m), b1.block(f, m)) {
                        (Some(x0), Some(x1)) => {
                            assert_eq!(x0.nrows(), x1.nrows());
                            assert_eq!(x0.ncols(), x1.ncols());
                            for j in 0..x0.ncols() {
                                for i in 0..x0.nrows() {
                                    assert_eq!(
                                        x0[(i, j)],
                                        x1[(i, j)],
                                        "k={k} forward={forward} right core {f:?}@{m:?}[{i},{j}]"
                                    );
                                }
                            }
                        }
                        (None, None) => {}
                        _ => panic!(
                            "k={k} forward={forward}: right core block presence differs at \
                             {f:?}@{m:?}"
                        ),
                    }
                }
            }
        }
    }
}

#[test]
fn each_channel_contributes_unit_trace() {
    let (h, space) = h4_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5b_7ace);
    let x = randomized_state(space, 3, 5, &mut rng);
    let k = 1usize;
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = sketch_h_mul(&fl, &h, &xk);
    let noise_dm = noise_density_matrix(&h, &half, &xk, &xk1, true);

    let mut total_trace = 0.0f64;
    for &m in &space.bond_sectors(k + 1) {
        let nd = &noise_dm[m];
        for i in 0..nd.nrows().min(nd.ncols()) {
            total_trace += nd[(i, i)];
        }
    }
    let mut groups: HashSet<Ts> = HashSet::default();
    for &s in &half.col_qn {
        for &desc in h.channel_descs(k + 1, s) {
            groups.insert(desc);
        }
    }
    let n_groups = groups.len();
    assert!(
        n_groups >= 2,
        "test is only meaningful with multiple channels; got {n_groups}"
    );
    let nearest = total_trace.round();
    assert!(
        (total_trace - nearest).abs() <= 1e-9 * total_trace.max(1.0),
        "Σ tr(ρ_noise) = {total_trace} is not an integer count of unit-trace channels"
    );
    assert!(
        (1.0..=n_groups as f64 + 1e-9).contains(&total_trace),
        "Σ tr(ρ_noise) = {total_trace} outside [1, {n_groups}] active-channel range"
    );
}

#[test]
fn same_ts_across_output_sectors_is_a_stable_nonvacuous_partition() {
    let (h, space) = h4_fixture();
    let k = 1usize;
    let sectors = space.bond_sectors(k + 1);

    let partition = |h: &Su2SparseHamiltonian<f64>| -> FxHashMap<Ts, HashSet<Su2Sector>> {
        let mut m: FxHashMap<Ts, HashSet<Su2Sector>> = FxHashMap::default();
        for &s in &sectors {
            for &desc in h.channel_descs(k + 1, s) {
                m.entry(desc).or_default().insert(s);
            }
        }
        m
    };
    let p1 = partition(&h);
    let p2 = partition(&h);
    assert_eq!(
        p1, p2,
        "channel_descs is not deterministic across re-scans — Ts grouping is not stable"
    );
    let multi_sector = p1.values().filter(|v| v.len() >= 2).count();
    assert!(
        multi_sector >= 1,
        "fixture too poor: no Ts value threads more than one output sector at bond {} — the \
         cross-sector grouping this certification (and noise_density_matrix's group_sq) \
         relies on is untested",
        k + 1
    );
}
