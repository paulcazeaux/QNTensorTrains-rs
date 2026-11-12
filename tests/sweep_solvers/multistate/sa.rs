// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::core::SparseCore;
use qntt::core::twosite::{TruncInfo, factor_svd_average};
use qntt::solvers::mals::{MalsOptions, NoiseType, mals, mals_with_history};
use qntt::solvers::multistate::{
    MultiTt, SaSweep, StateAverageOptions, sa_maxsubspace, sa_restart_keep, state_average_mals,
    state_average_mals_mut, state_average_mals_with_history,
};
use qntt::vector::perturbation_uniform;
use qntt::{C64, rayleigh_quotient, tt_state};
use qntt_testkit::randomized_state;

fn state_from_sites<T: Scalar>(d: usize, up_sites: &[usize], dn_sites: &[usize]) -> TtVector<T> {
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in up_sites {
        occ_up[s] = true;
    }
    for &s in dn_sites {
        occ_dn[s] = true;
    }
    tt_state(&occ_up, &occ_dn)
}

fn sa_opts(mals: MalsOptions) -> StateAverageOptions {
    StateAverageOptions {
        mals,
        ..Default::default()
    }
}

fn opts_exact() -> MalsOptions {
    MalsOptions {
        eps: 1e-10,
        conv_tol: Some(1e-12),
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 40,
        verbose: false,
        ..Default::default()
    }
}

fn max_total_bond_rank<T: Scalar>(x: &TtVector<T>) -> usize {
    x.ranks.iter().map(|g| g.total()).max().unwrap_or(0)
}

fn kept_ranks<T: Scalar>(shared: &SparseCore<T>, forward: bool) -> qntt::qn::RankGrid {
    if forward {
        shared.col_ranks.clone()
    } else {
        shared.row_ranks.clone()
    }
}

fn expected_full_kept<T: Scalar>(cp: &ContractedSparseCores<T>, forward: bool) -> usize {
    let mut total = 0usize;
    for &m in cp.mid_qn.iter() {
        if let Some(b) = cp.block(m)
            && b.nrows() > 0
            && b.ncols() > 0
        {
            total += if forward { b.nrows() } else { b.ncols() };
        }
    }
    total
}

fn core_gram(cs: &[SparseCore<f64>]) -> (f64, Vec<f64>) {
    let n = cs.len();
    let mut g = Mat::<f64>::zeros(n, n);
    let mut off = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let v = cs[i].dot_cores(&cs[j]);
            *g.get_mut(i, j) = v;
            if i != j {
                off = off.max(v.abs());
            }
        }
    }
    let (evals, _) = dense_self_adjoint_eigen(g.as_ref());
    (off, evals)
}

fn core_gram_dev<T: Scalar>(cs: &[SparseCore<T>]) -> f64 {
    let mut dev = 0.0f64;
    for i in 0..cs.len() {
        for j in 0..cs.len() {
            let want = if i == j { T::one() } else { T::zero() };
            dev = dev.max((cs[i].dot_cores(&cs[j]) - want).abs_s());
        }
    }
    dev
}

fn core_bits_eq<T: Scalar>(a: &SparseCore<T>, b: &SparseCore<T>, what: &str) {
    assert_eq!(a.num_coeffs(), b.num_coeffs(), "{what}: num_coeffs");
    for i in 0..a.num_coeffs() {
        assert!(bits_eq(a.coeff(i), b.coeff(i)), "{what}: coefficient {i}");
    }
}

fn trunc_bits_eq(a: &TruncInfo, b: &TruncInfo, what: &str) {
    assert_f64_bits(a.discarded_weight, b.discarded_weight, what);
    assert_eq!(a.kept, b.kept, "{what}: kept");
}

fn weighted(w: &[f64], l: &[f64]) -> f64 {
    w.iter().zip(l.iter()).map(|(a, b)| a * b).sum()
}

fn assert_weighted_ky_fan(sw: &SaSweep, w: &[f64], ev: &[f64], tag: &str) {
    let want: f64 = w.iter().zip(ev.iter()).map(|(a, b)| a * b).sum();
    let got = weighted(w, &sw.energies);
    assert!(
        got >= want - 1e-10,
        "{tag}: weighted state-average energy {got:.12e} below the exact weighted bound \
         {want:.12e} (Σwᵢλᵢ ≥ ΣwᵢEᵢ)"
    );
}

#[test]
fn tms_b1_state_average_n1_is_mals_bitwise() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0x7B01);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    assert!(x0.check());

    let noise_free = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 6,
        ..Default::default()
    };
    let ramp = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![4, 0],
        noise: vec![1e-3, 1e-4, 0.0, 0.0],
        max_iter: 6,
        ..Default::default()
    };

    for (arm, opts) in [("noise-free", noise_free), ("perturbative-ramp", ramp)] {
        let (mhist, my) = mals_with_history(&h, &x0, &opts);
        let o = sa_opts(opts.clone());
        let (shist, sys) = state_average_mals_with_history(&h, &x0, 1, &o);

        assert_eq!(sys.len(), 1, "{arm}: one state expected at nroots = 1");
        assert_eq!(
            shist.sweeps.len(),
            mhist.sweeps.len(),
            "{arm}: iteration count (the schedule and the stop test must be bitwise MALS's)"
        );
        assert_eq!(shist.lambdas.len(), 1, "{arm}: lambdas length");
        assert_f64_bits(shist.lambdas[0], mhist.lambda, &format!("{arm}: final λ"));

        for (it, (s, m)) in shist.sweeps.iter().zip(mhist.sweeps.iter()).enumerate() {
            let tag = |w: &str| format!("{arm}: sweep {it} {w}");
            assert_eq!(s.energies.len(), 1, "{}", tag("energies length"));
            assert_f64_bits(s.energies[0], m.energy, &tag("energies[0]"));
            assert_f64_bits(s.energy, m.energy, &tag("energy (Σwᵢλᵢ = 1·λ)"));
            assert_f64_bits(
                s.discarded_weight,
                m.discarded_weight,
                &tag("discarded_weight"),
            );
            assert_eq!(
                s.discarded_weights.len(),
                1,
                "{}",
                tag("discarded_weights length")
            );
            assert_f64_bits(
                s.discarded_weights[0],
                m.discarded_weight,
                &tag("discarded_weights[0]"),
            );
            assert_eq!(s.max_bond_dim, m.max_bond_dim, "{}", tag("max_bond_dim"));
            assert_eq!(s.bond_dim, m.bond_dim, "{}", tag("bond_dim"));
            assert_f64_bits(s.noise, m.noise, &tag("noise"));
            assert_eq!(
                s.overlap_dev,
                0.0,
                "{}",
                tag("overlap_dev must be 0 at n = 1")
            );
            assert!(
                s.numops > 0,
                "{}",
                tag("numops must count the local applies")
            );
        }

        coeffs_bits_eq_tt(&sys[0], &my, &format!("{arm}: returned state"));

        let (slam, _) = state_average_mals(&h, &x0, 1, &o);
        let (mlam, _) = mals(&h, &x0, &opts);
        assert_eq!(slam.len(), 1, "{arm}: state_average_mals λ length");
        assert_f64_bits(slam[0], mlam, &format!("{arm}: state_average_mals λ"));
    }
}

#[test]
fn tms_b1_sa_davidson_defaults_satisfy_the_block_preconditions() {
    assert_eq!(
        sa_maxsubspace(1),
        8,
        "sa_maxsubspace(1) is the n = 1 parity point"
    );
    assert_eq!(
        sa_restart_keep(1, sa_maxsubspace(1)),
        1,
        "sa_restart_keep(1, ·)"
    );
    for n in 2..=8usize {
        let ms = sa_maxsubspace(n);
        let rk = sa_restart_keep(n, ms);
        assert!(ms >= 2 * n, "n = {n}: maxsubspace {ms} < 2·nroots");
        assert!(
            n <= rk && rk < ms,
            "n = {n}: restart_keep {rk} outside [n, maxsubspace)"
        );
    }
}

#[test]
fn tms_b2_full_rank_average_reconstructs_each_block() {
    let space = QnSpace::new(6, 1, 1);
    let (x, cp0) = two_site_frame_abelian::<f64>(space, 2, 0xB2_0001);
    let mut rng = seeded(0xB2_0002);
    let cps = unit_blocks_abelian(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    assert!(
        (w.iter().sum::<f64>() - 1.0).abs() <= 1e-12,
        "fixture weights"
    );
    assert!(
        cp0.num_coeffs() >= 3,
        "fixture: local space cannot host 3 states"
    );
    let _ = x;

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };
        let (shared, s, cs, infos, pooled, lw) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled::NONE, forward, false);

        assert_eq!(cs.len(), 3, "{tag}: one centre per state");
        assert_eq!(infos.len(), 3, "{tag}: one TruncInfo per state");

        for (i, info) in infos.iter().enumerate() {
            assert!(
                info.discarded_weight <= 1e-12,
                "{tag}: state {i} discarded {:.3e} at full rank",
                info.discarded_weight
            );
        }
        assert!(
            pooled.discarded_weight <= 1e-12,
            "{tag}: pooled discarded {:.3e} at full rank",
            pooled.discarded_weight
        );
        assert_eq!(
            pooled.kept,
            expected_full_kept(&cp0, forward),
            "{tag}: pooled kept rank is not the full pooled ρ dimension"
        );

        let mr = kept_ranks(&shared, forward);
        for m in s.sectors() {
            let b = s
                .block(m)
                .unwrap_or_else(|| panic!("{tag}: missing gauge block at {m:?}"));
            let r = mr[m];
            assert_eq!(
                (b.nrows(), b.ncols()),
                (r, r),
                "{tag}: gauge shape at {m:?}"
            );
            for i in 0..r {
                for j in 0..r {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (*b.get(i, j) - want).abs() <= 0.0,
                        "{tag}: gauge at {m:?} entry ({i},{j}) = {} (must be exactly I)",
                        b.get(i, j)
                    );
                }
            }
        }

        for i in 0..3 {
            let r = if forward {
                contract(&shared, &cs[i])
            } else {
                contract(&cs[i], &shared)
            };
            assert_eq!(
                r.num_coeffs(),
                cps[i].num_coeffs(),
                "{tag}: state {i} coefficient count"
            );
            let mut worst = 0.0f64;
            for j in 0..r.num_coeffs() {
                worst = worst.max((r.coeff(j) - cps[i].coeff(j)).abs());
            }
            assert!(
                worst <= 1e-12,
                "{tag}: state {i} reconstruction error {worst:.3e} at full rank"
            );
        }

        let mut off = 0.0f64;
        let mut g = Mat::<f64>::zeros(3, 3);
        for i in 0..3 {
            for j in 0..3 {
                let val = cps[i].dot_cores(&cps[j]);
                *g.get_mut(i, j) = val;
                if i != j {
                    off = off.max(val.abs());
                }
            }
        }
        let (bevals, _) = dense_self_adjoint_eigen(g.as_ref());
        assert!(
            (lw.overlap_dev - off).abs() <= 1e-12,
            "{tag}: overlap_dev {} vs block Gram off-diagonal {off}",
            lw.overlap_dev
        );
        assert!(
            (lw.gram_min - bevals[0]).abs() <= 1e-12,
            "{tag}: gram_min {} vs {}",
            lw.gram_min,
            bevals[0]
        );
        assert!(
            (lw.gram_max - bevals[2]).abs() <= 1e-12,
            "{tag}: gram_max {} vs {}",
            lw.gram_max,
            bevals[2]
        );
    }
}

#[test]
#[should_panic(expected = "norm")]
fn tms_b2_average_refuses_a_non_unit_block() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 2, 0xB2_0011);
    let mut rng = seeded(0xB2_0012);
    let mut cps = unit_blocks_abelian(&cp0, 2, &mut rng);
    cps[1].scale_mut(1.1);
    let _ = factor_svd_average(
        &cps,
        &[0.5, 0.5],
        None,
        0.0,
        0.0,
        qntt::Pooled::NONE,
        true,
        false,
    );
}

#[test]
#[should_panic(expected = "non-increasing")]
fn tms_b2_average_refuses_increasing_weights() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 2, 0xB2_0021);
    let mut rng = seeded(0xB2_0022);
    let cps = unit_blocks_abelian(&cp0, 2, &mut rng);
    let _ = factor_svd_average(
        &cps,
        &[0.3, 0.7],
        None,
        0.0,
        0.0,
        qntt::Pooled::NONE,
        true,
        false,
    );
}

#[test]
fn tms_b3_weighted_ky_fan_and_trace_loss() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 2, 0xB3_0001);
    let mut rng = seeded(0xB3_0002);
    let cps = unit_blocks_abelian(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    let r = 3usize;

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };
        let (shared, _s, cs, infos, pooled, _lw) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), forward, false);

        let spec = pooled_rho_abelian(&cps, &w, forward);
        let mut pooled_vals: Vec<(f64, qntt::qn::Sector)> = Vec::new();
        for (m, evals, _) in &spec {
            for &l in evals {
                pooled_vals.push((l, *m));
            }
        }
        pooled_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let tr: f64 = pooled_vals.iter().map(|p| p.0).sum();
        assert!(
            (tr - 1.0).abs() <= 1e-12,
            "{tag}: tr ρ = {tr} — must be Σᵢwᵢ‖Bᵢ‖² = 1 (the unit-norm/Σw gate)"
        );
        let top: f64 = pooled_vals[..r].iter().map(|p| p.0).sum();

        let kept_w: f64 = (0..3).map(|i| w[i] * cs[i].norm().powi(2)).sum();
        assert!(
            (kept_w - top).abs() <= 1e-12,
            "{tag}: kept weight {kept_w:.15e} vs top-{r} eigenvalue sum {top:.15e}"
        );
        assert!(
            (pooled.discarded_weight - (tr - top)).abs() <= 1e-12,
            "{tag}: pooled discarded {:.15e} vs trace loss {:.15e}",
            pooled.discarded_weight,
            tr - top
        );
        let wsum: f64 = (0..3).map(|i| w[i] * infos[i].discarded_weight).sum();
        assert!(
            (wsum - pooled.discarded_weight).abs() <= 1e-12,
            "{tag}: Σwᵢdroppedᵢ = {wsum:.15e} vs pooled {:.15e}",
            pooled.discarded_weight
        );
        for i in 0..3 {
            let want = 1.0 - cs[i].norm().powi(2);
            assert!(
                (infos[i].discarded_weight - want).abs() <= 1e-12,
                "{tag}: state {i} discarded {:.15e} vs 1 − ‖cᵢ‖² = {want:.15e}",
                infos[i].discarded_weight
            );
        }
        assert_eq!(pooled.kept, r, "{tag}: pooled kept rank");

        let mr = kept_ranks(&shared, forward);
        for (m, _, _) in &spec {
            let want = pooled_vals[..r].iter().filter(|p| p.1 == *m).count();
            assert_eq!(
                mr[*m], want,
                "{tag}: sector {m:?} kept rank {} but the top-{r} pooled values place {want} \
                 there",
                mr[*m]
            );
        }

        let mut arng = seeded(0xB3_0003 + forward as u64);
        for trial in 0..8 {
            let mut alt = 0.0f64;
            for (m, evals, _) in &spec {
                let rr = mr[*m];
                if rr == 0 {
                    continue;
                }
                let q = random_orthonormal::<f64>(evals.len(), rr, &mut arng);
                for i in 0..3 {
                    let b = cps[i].block(*m).unwrap();
                    alt += w[i] * kept_weight(q.as_ref(), b, forward);
                }
            }
            assert!(
                alt <= top + 1e-12,
                "{tag}: trial {trial} random basis keeps {alt:.15e} > the Ky Fan optimum \
                 {top:.15e} — the shipped basis is not the leading eigenspace"
            );
        }
    }
}

#[test]
fn tms_b4_loewdin_gram_is_identity_after_writeback() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 2, 0xB4_0001);
    let mut rng = seeded(0xB4_0002);
    let cps = unit_blocks_abelian(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    let r = 3usize;

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };
        let (sa_shared, sa_s, plain, ia, pa, la) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), forward, false);
        let (sb_shared, sb_s, ortho, ib, pb, lb) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), forward, true);

        let (off, evals) = core_gram(&plain);
        for (name, lw) in [("reorth=false", &la), ("reorth=true", &lb)] {
            assert!(
                (lw.overlap_dev - off).abs() <= 1e-12,
                "{tag}/{name}: overlap_dev {} vs max off-diagonal {off}",
                lw.overlap_dev
            );
            assert!(
                (lw.gram_min - evals[0]).abs() <= 1e-12 && (lw.gram_max - evals[2]).abs() <= 1e-12,
                "{tag}/{name}: (gram_min, gram_max) = ({}, {}) vs ({}, {})",
                lw.gram_min,
                lw.gram_max,
                evals[0],
                evals[2]
            );
        }
        assert!(
            off > 1e-6,
            "{tag}: VACUOUS — the pre-Löwdin centres are already orthogonal to {off:.3e}; the \
             rank cap is not tight enough for this arm to see the Löwdin step"
        );

        let dev = core_gram_dev(&ortho);
        assert!(dev <= 1e-12, "{tag}: post-Löwdin Gram deviation {dev:.3e}");
        for (i, c) in ortho.iter().enumerate() {
            assert!(
                (c.norm() - 1.0).abs() <= 1e-12,
                "{tag}: centre {i} norm {}",
                c.norm()
            );
        }

        let mut basis: Vec<SparseCore<f64>> = Vec::new();
        for c in &plain {
            let mut q = c.clone();
            for b in &basis {
                let ip = b.dot_cores(&q);
                q.axpy(-ip, b);
            }
            let nq = q.norm();
            if nq > 1e-10 {
                q.scale_mut(1.0 / nq);
                basis.push(q);
            }
        }
        for (i, c) in ortho.iter().enumerate() {
            let mut resid = c.clone();
            for b in &basis {
                let ip = b.dot_cores(&resid);
                resid.axpy(-ip, b);
            }
            assert!(
                resid.norm() <= 1e-12,
                "{tag}: centre {i} left span{{PBⱼ}} by {:.3e}",
                resid.norm()
            );
        }

        core_bits_eq(&sa_shared, &sb_shared, &format!("{tag}: shared basis"));
        for m in sa_s.sectors() {
            let (a, b) = (sa_s.block(m), sb_s.block(m));
            match (a, b) {
                (Some(a), Some(b)) => {
                    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
                    for i in 0..a.nrows() {
                        for j in 0..a.ncols() {
                            assert!(
                                bits_eq(*a.get(i, j), *b.get(i, j)),
                                "{tag}: gauge {m:?} ({i},{j}) moved under Löwdin"
                            );
                        }
                    }
                }
                (None, None) => {}
                _ => panic!("{tag}: gauge sector {m:?} present in only one call"),
            }
        }
        trunc_bits_eq(&pa, &pb, &format!("{tag}: pooled TruncInfo"));
        for i in 0..3 {
            trunc_bits_eq(&ia[i], &ib[i], &format!("{tag}: state {i} TruncInfo"));
        }
    }
}

#[test]
#[should_panic(expected = "collapsed")]
fn tms_b4_loewdin_collapse_panics() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 2, 0xB4_0011);
    let mut rng = seeded(0xB4_0012);
    let cps = unit_blocks_abelian(&cp0, 1, &mut rng);
    let twins = vec![cps[0].clone(), cps[0].clone()];
    let _ = factor_svd_average(
        &twins,
        &[0.5, 0.5],
        None,
        0.0,
        0.0,
        qntt::Pooled::NONE,
        true,
        true,
    );
}

#[test]
fn tms_b6_full_rank_state_average_matches_the_dense_spectrum() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xB6_0000, 3, 0.05);
    assert_exclusion_gap(&fx.ev11, 3, 0.05, "tms_b6 (1,1)");
    let h = fx.abelian(space);
    let opts = opts_exact();

    for n in [1usize, 3] {
        let w = vec![1.0 / n as f64; n];
        let mut best: Option<(
            Vec<f64>,
            Vec<TtVector<f64>>,
            qntt::solvers::multistate::SaHistory,
        )> = None;
        for trial in 0..3u64 {
            let mut rng = seeded(0xB6_1000 + 0x100 * n as u64 + trial);
            let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
            assert!(x0.check());
            let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &sa_opts(opts.clone()));
            let lam = hist.lambdas.clone();
            let ok = (0..n).all(|i| (lam[i] - fx.ev11[i]).abs() <= 1e-8);
            best = Some((lam, ys, hist));
            if ok {
                break;
            }
        }
        let (lam, ys, hist) = best.unwrap();

        for i in 0..n {
            assert!(
                (lam[i] - fx.ev11[i]).abs() <= 1e-8,
                "n = {n}: root {i} λ = {:.12} vs dense E = {:.12}",
                lam[i],
                fx.ev11[i]
            );
            assert!(ys[i].check(), "n = {n}: root {i} check()");
            let nn = qntt::dot(&ys[i], &ys[i]).abs();
            assert!((nn - 1.0).abs() <= 1e-10, "n = {n}: root {i} ‖x‖² = {nn}");
            let rq = rayleigh_quotient(&h, &ys[i]);
            assert!(
                (rq - lam[i]).abs() <= 1e-10,
                "n = {n}: root {i} Rayleigh quotient {rq} vs reported λ {}",
                lam[i]
            );
            let resid = global_eigen_residual_abelian(&h, &ys[i], lam[i]);
            assert!(resid <= 1e-6, "n = {n}: root {i} residual {resid:.3e}");
            let proj = projection_on_dense(&ys[i], &fx.dets11, &fx.evec11[i]);
            assert!(
                proj >= 1.0 - 1e-8,
                "n = {n}: root {i} projection on the dense eigenvector {proj:.12}"
            );
        }
        let gd = gram_dev_tt(&ys);
        assert!(gd <= 1e-8, "n = {n}: returned Gram deviation {gd:.3e}");

        if n >= 2 {
            let mut mine = ys.clone();
            let before = mine[1].core(0).coeff(0);
            let bumped = mine[0].core(0).coeff(0) + 1.0;
            mine[0].cores[0].set_coeff(0, bumped);
            assert_eq!(
                mine[1].core(0).coeff(0),
                before,
                "n = {n}: states() aliased the shared cores"
            );
        }

        let last = hist.sweeps.last().expect("at least one sweep");
        assert_eq!(last.energies.len(), n, "n = {n}: history energies length");
        assert_eq!(
            last.discarded_weights.len(),
            n,
            "n = {n}: history discarded_weights length"
        );
        for i in 1..n {
            assert!(
                last.energies[i] >= last.energies[i - 1],
                "n = {n}: homing = false must label by ENERGY at every bond: {:?}",
                last.energies
            );
        }
        assert!(
            (last.energy - weighted(&w, &last.energies)).abs() <= 1e-14,
            "n = {n}: history energy {} vs Σwᵢλᵢ {}",
            last.energy,
            weighted(&w, &last.energies)
        );
        assert!(
            last.max_dev < 1e-10,
            "n = {n}: run stopped with max_dev {:.3e}",
            last.max_dev
        );
        for (it, sw) in hist.sweeps.iter().enumerate() {
            assert!(sw.numops > 0, "n = {n}: sweep {it} numops");
            assert!(
                sw.overlap_dev.is_finite() && sw.overlap_dev >= 0.0,
                "n = {n}: sweep {it} overlap_dev {}",
                sw.overlap_dev
            );
        }
    }
}

#[test]
fn tms_b7_truncated_state_average_ky_fan_and_host_n_states() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xB7_0000, 3, 0.05);
    let h = fx.abelian(space);
    let n = 3usize;
    let w = vec![0.5, 0.3, 0.2];
    let caps = vec![2usize, 3, 4];
    let opts = MalsOptions {
        eps: 1e-8,
        bond_dims: caps.clone(),
        noise: vec![0.0],
        max_iter: 8,
        ..Default::default()
    };
    let o = StateAverageOptions {
        mals: opts,
        weights: Some(w.clone()),
        ..Default::default()
    };

    let mut rng = seeded(0xB7_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &o);

    for (it, sw) in hist.sweeps.iter().enumerate() {
        assert_eq!(sw.noise, 0.0, "sweep {it}: the schedule is noise-free");
        assert_weighted_ky_fan(sw, &w, &fx.ev11, &format!("sweep {it}"));
        let nl = sw.energies.len();
        assert_eq!(
            sw.lowest_energies.len(),
            nl,
            "sweep {it}: lowest_energies length"
        );
        let (lo, en) = (
            weighted(&w, &sw.lowest_energies),
            weighted(&w, &sw.energies),
        );
        assert!(
            lo <= en + 1e-12,
            "sweep {it}: lowest {lo} above the state's {en}"
        );
        let ky = weighted(&w, &fx.ev11[..nl]);
        assert!(
            lo >= ky - 1e-10,
            "sweep {it}: lowest {lo} below the exact weighted levels {ky}"
        );
        let want_cap = caps[it.min(caps.len() - 1)];
        assert_eq!(sw.bond_dim, want_cap, "sweep {it}: clamp-to-last schedule");
        assert!(
            sw.max_bond_dim >= n,
            "sweep {it}: max_bond_dim {} below nroots {n} — host_n_states did not fire",
            sw.max_bond_dim
        );
        assert!(
            sw.max_bond_dim <= want_cap.max(n),
            "sweep {it}: max_bond_dim {} above max(bond_dim {want_cap}, n {n})",
            sw.max_bond_dim
        );
    }

    for i in 1..n {
        assert_eq!(
            ys[i].ranks, ys[0].ranks,
            "root {i}: returned ranks differ from root 0 (the basis is SHARED and the driver is \
             round-free)"
        );
    }
    for i in 0..n {
        assert!(ys[i].check(), "root {i}: check()");
        let rq = rayleigh_quotient(&h, &ys[i]);
        assert!(
            (rq - hist.lambdas[i]).abs() <= 1e-10,
            "root {i}: Rayleigh quotient {rq} vs reported λ {}",
            hist.lambdas[i]
        );
    }
    let gd = gram_dev_tt(&ys);
    assert!(
        gd <= 1e-10,
        "returned Gram deviation {gd:.3e} at a truncated rank"
    );
}

#[test]
fn tms_b8_degenerate_ground_space() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let span: [(usize, usize); 4] = [(0, 1), (2, 4), (1, 0), (4, 2)];
    let zs: Vec<TtVector<f64>> = span
        .iter()
        .map(|&(u, w)| state_from_sites(d, &[u], &[w]))
        .collect();
    let n = 2usize;
    let opts = opts_exact();

    let mut got: Option<(Vec<f64>, Vec<TtVector<f64>>)> = None;
    for trial in 0..4u64 {
        let mut rng = seeded(0xB8_0000 + trial);
        let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        let (lam, ys) = state_average_mals(&h, &x0, n, &sa_opts(opts.clone()));
        let ok = lam.iter().all(|l| (l + 0.5).abs() <= 1e-8);
        got = Some((lam, ys));
        if ok {
            break;
        }
    }
    let (lam, ys) = got.unwrap();
    for i in 0..n {
        assert!(
            (lam[i] + 0.5).abs() <= 1e-8,
            "root {i}: λ = {} (the doublet sits at −0.5)",
            lam[i]
        );
        let proj: f64 = zs
            .iter()
            .map(|z| {
                let ov = qntt::dot(&ys[i], z);
                ov.abs_s() * ov.abs_s()
            })
            .sum();
        assert!(
            (proj - 1.0).abs() <= 1e-8,
            "root {i}: analytic span projection {proj}"
        );
        assert!(ys[i].check(), "root {i}: check()");
    }
    let gd = gram_dev_tt(&ys);
    assert!(gd <= 1e-8, "degenerate doublet Gram deviation {gd:.3e}");
}

#[test]
fn tms_b9_homing_is_anchored_and_matches_energy_labels() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xB9_0000, 3, 0.05);
    let h = fx.abelian(space);
    let n = 3usize;
    let opts = opts_exact();

    for seed in 0xB9_1000u64..0xB9_1005 {
        let mut rng = seeded(seed);
        let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
        let homed = StateAverageOptions {
            mals: opts.clone(),
            homing: true,
            ..Default::default()
        };
        let plain = sa_opts(opts.clone());
        let (ha, _ya) = state_average_mals_with_history(&h, &x0, n, &homed);
        let (hb, _yb) = state_average_mals_with_history(&h, &x0, n, &plain);
        println!(
            "tms_b9 seed {seed:#x}: homed {} sweeps, energy-labelled {} sweeps",
            ha.sweeps.len(),
            hb.sweeps.len()
        );

        for i in 1..n {
            assert!(
                ha.lambdas[i] >= ha.lambdas[i - 1] - 1e-12,
                "seed {seed:#x}: homed labels are not energy-sorted at the end: {:?}",
                ha.lambdas
            );
        }
        for i in 0..n {
            assert!(
                (ha.lambdas[i] - hb.lambdas[i]).abs() <= 1e-10,
                "seed {seed:#x}: root {i} homed λ {} vs energy-labelled {}",
                ha.lambdas[i],
                hb.lambdas[i]
            );
            assert!(
                (ha.lambdas[i] - fx.ev11[i]).abs() <= 1e-8,
                "seed {seed:#x}: root {i} homed λ {} vs dense E {}",
                ha.lambdas[i],
                fx.ev11[i]
            );
        }
    }
}

#[test]
fn tms_b10_c64_state_average_matches_the_real_spectrum() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xBA_0000, 3, 0.05);
    let theta: Vec<f64> = (0..d).map(|p| 0.3 + 0.4 * p as f64).collect();
    let (tc, vc) = gauge_integrals(d, &fx.t, &fx.v, &theta);
    let h = SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
    let n = 3usize;
    let w = vec![1.0 / 3.0; 3];

    let noise_free = opts_exact();
    let noisy = MalsOptions {
        eps: 1e-10,
        conv_tol: Some(1e-12),
        bond_dims: vec![4, 6, 0],
        noise: vec![1e-3, 1e-4, 0.0, 0.0, 0.0],
        max_iter: 12,
        ..Default::default()
    };

    for (arm, opts, tol) in [
        ("noise-free", noise_free, 1e-8),
        ("perturbative-ramp", noisy, 1e-8),
    ] {
        let mut rng = seeded(0xBA_1000);
        let x0: TtVector<C64> = randomized_state(space, 5, 10, &mut rng);
        let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &sa_opts(opts));

        for i in 0..n {
            assert!(
                (hist.lambdas[i] - fx.ev11[i]).abs() <= tol,
                "{arm}: root {i} λ = {} vs the gauge-invariant dense E = {}",
                hist.lambdas[i],
                fx.ev11[i]
            );
            let resid = global_eigen_residual_abelian(&h, &ys[i], hist.lambdas[i]);
            assert!(resid <= 1e-6, "{arm}: root {i} residual {resid:.3e}");
            assert!(ys[i].check(), "{arm}: root {i} check()");
        }
        let gd = gram_dev_tt(&ys);
        assert!(gd <= 1e-8, "{arm}: returned Gram deviation {gd:.3e}");

        let rhos: Vec<f64> = ys.iter().map(rho_abelian).collect();
        assert!(
            rhos.iter().any(|r| *r < 0.99),
            "{arm}: VACUOUS — every root is real up to a global phase (ρ = {rhos:?})"
        );

        for (it, sw) in hist.sweeps.iter().enumerate() {
            assert_weighted_ky_fan(sw, &w, &fx.ev11, &format!("{arm}: sweep {it}"));
        }
    }
}

#[test]
fn tms_b11_perturbative_state_average_grows_rank_and_stays_variational() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xBB_0000, 3, 0.05);
    let h = fx.abelian(space);
    let n = 3usize;
    let w = vec![1.0 / 3.0; 3];

    let det: TtVector<f64> = state_from_sites(d, &[0], &[1]);
    assert_eq!(
        max_total_bond_rank(&det),
        1,
        "the start must be a determinant"
    );
    let mut rng = seeded(0xBB_0001);
    let x0 = perturbation_uniform(&det, 1, 0.2, &mut rng);
    assert!(x0.check());
    let r0 = max_total_bond_rank(&x0);

    let opts = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![1e-3, 1e-3, 1e-4, 0.0, 0.0, 0.0],
        max_iter: 12,
        ..Default::default()
    };
    let schedule = opts.noise.clone();
    let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &sa_opts(opts));

    for (it, sw) in hist.sweeps.iter().enumerate() {
        assert_eq!(
            sw.noise,
            schedule[it.min(schedule.len() - 1)],
            "sweep {it}: noise schedule clamp-to-last"
        );
        assert_weighted_ky_fan(sw, &w, &fx.ev11, &format!("sweep {it}"));
    }
    let final_rank = hist.sweeps.last().unwrap().max_bond_dim;
    println!("tms_b11: enriched start rank {r0} → final max_bond_dim {final_rank}");
    assert!(
        final_rank > r0,
        "the noisy ramp did not grow the rank: start {r0}, final {final_rank}"
    );
    assert!(
        final_rank >= n,
        "final max_bond_dim {final_rank} below nroots {n}"
    );

    for i in 0..n {
        assert!(
            (hist.lambdas[i] - fx.ev11[i]).abs() <= 1e-6,
            "root {i}: λ = {} vs dense E = {} after the ramp reached 0",
            hist.lambdas[i],
            fx.ev11[i]
        );
        assert!(ys[i].check(), "root {i}: check()");
    }
    let gd = gram_dev_tt(&ys);
    assert!(gd <= 1e-8, "returned Gram deviation {gd:.3e}");
}

#[test]
fn tms_b12_threaded_matches_serial_bitwise() {
    let d = 6usize;
    let space = QnSpace::new(d, 2, 2);
    let fx = gapped_tv(0xBC_0000, 3, 0.05);
    let h = fx.abelian(space);
    let n = 3usize;
    let opts = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 3,
        ..Default::default()
    };
    let o = sa_opts(opts);
    let mut rng = seeded(0xBC_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let run = |nt: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .build()
            .unwrap()
            .install(|| state_average_mals_with_history(&h, &x0, n, &o))
    };
    let (h1, y1) = run(1);
    let (h8, y8) = run(8);

    let widest = (0..d - 1)
        .map(|k| contract(y1[0].core(k), y1[0].core(k + 1)).mid_qn.len())
        .max()
        .unwrap_or(0);
    println!("tms_b12: widest visited bond carries {widest} mid sectors");
    assert!(
        widest >= 4,
        "VACUOUS — the widest bond carries {widest} mid sectors, below the kernel's parallel \
         fork threshold (4), so nothing here runs threaded"
    );

    let mut worst = 0.0f64;
    for (a, b) in h1.sweeps.iter().zip(h8.sweeps.iter()) {
        for i in 0..n {
            worst = worst.max((a.energies[i] - b.energies[i]).abs());
        }
        worst = worst.max((a.energy - b.energy).abs());
        worst = worst.max((a.discarded_weight - b.discarded_weight).abs());
    }
    println!("tms_b12: max |Δ| between the 1- and 8-thread trajectories = {worst:.3e}");

    assert_eq!(h1.sweeps.len(), h8.sweeps.len(), "sweep count");
    for (it, (a, b)) in h1.sweeps.iter().zip(h8.sweeps.iter()).enumerate() {
        for i in 0..n {
            assert_f64_bits(
                a.energies[i],
                b.energies[i],
                &format!("sweep {it} energies[{i}]"),
            );
            assert_f64_bits(
                a.discarded_weights[i],
                b.discarded_weights[i],
                &format!("sweep {it} discarded_weights[{i}]"),
            );
        }
        assert_f64_bits(a.energy, b.energy, &format!("sweep {it} energy"));
        assert_f64_bits(a.max_dev, b.max_dev, &format!("sweep {it} max_dev"));
        assert_f64_bits(
            a.discarded_weight,
            b.discarded_weight,
            &format!("sweep {it} discarded_weight"),
        );
        assert_f64_bits(
            a.overlap_dev,
            b.overlap_dev,
            &format!("sweep {it} overlap_dev"),
        );
        assert_eq!(a.max_bond_dim, b.max_bond_dim, "sweep {it} max_bond_dim");
        assert_eq!(a.numops, b.numops, "sweep {it} numops");
    }
    for i in 0..n {
        assert_f64_bits(h1.lambdas[i], h8.lambdas[i], &format!("final λ[{i}]"));
        coeffs_bits_eq_tt(&y1[i], &y8[i], &format!("root {i} state"));
    }
}

#[test]
fn tms_b13_per_state_discarded_weights_expose_the_light_root() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xBD_0000, 3, 0.05);
    let h = fx.abelian(space);
    let n = 3usize;
    let w = vec![0.6, 0.3, 0.1];
    let o = StateAverageOptions {
        mals: MalsOptions {
            eps: 1e-8,
            bond_dims: vec![2],
            noise: vec![0.0],
            max_iter: 4,
            ..Default::default()
        },
        weights: Some(w.clone()),
        ..Default::default()
    };
    let mut rng = seeded(0xBD_0001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let (hist, _ys) = state_average_mals_with_history(&h, &x0, n, &o);
    let s = hist.sweeps.last().expect("at least one sweep");

    assert_eq!(s.discarded_weights.len(), n, "per-state ledger length");
    for (i, dw) in s.discarded_weights.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(dw),
            "root {i}: discarded weight {dw} outside [0,1]"
        );
    }
    let dmax = s
        .discarded_weights
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "tms_b13: pooled {:.6e}, per-state {:?}, light-root ratio {:.3}",
        s.discarded_weight,
        s.discarded_weights,
        s.discarded_weights[2] / s.discarded_weight.max(1e-300)
    );
    assert!(
        dmax >= s.discarded_weight - 1e-15,
        "max per-state loss {dmax:.6e} below the pooled max {:.6e} (a convex combination can \
         never exceed its own maximum)",
        s.discarded_weight
    );
    assert!(
        dmax > s.discarded_weight,
        "VACUOUS — the per-state ledger carries no information the pooled number does not \
         (max {dmax:.6e} vs pooled {:.6e}); tighten the cap or lighten w₂",
        s.discarded_weight
    );
    assert!(
        s.discarded_weights[2] > s.discarded_weight,
        "the light root's loss {:.6e} does not exceed the pooled loss {:.6e} — this arm cannot \
         demonstrate what the per-state ledger is for",
        s.discarded_weights[2],
        s.discarded_weight
    );

    let (hist2, _) = state_average_mals_with_history(&h, &x0, n, &o);
    let s2 = hist2.sweeps.last().unwrap();
    assert_eq!(
        s.discarded_weights, s2.discarded_weights,
        "state average is not deterministic on a rerun"
    );
    assert_eq!(
        hist.lambdas, hist2.lambdas,
        "λ not deterministic on a rerun"
    );
}

#[test]
#[should_panic(expected = "the two-site local space at")]
fn tms_b14_multitt_new_refuses_a_determinant_start() {
    let d = 6usize;
    let det: TtVector<f64> = state_from_sites(d, &[0], &[1]);
    let _ = MultiTt::new(&det, 2, 0xB14_0001, 0.1);
}

#[test]
#[should_panic(expected = "state-average sweeps define only Perturbative noise")]
fn tms_b14_random_noise_is_refused() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xB14_0011);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    let o = sa_opts(MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![1e-4, 0.0],
        noise_type: NoiseType::Random,
        max_iter: 2,
        ..Default::default()
    });
    let _ = state_average_mals(&h, &x0, 2, &o);
}

#[test]
#[should_panic(expected = "non-increasing")]
fn tms_b14_increasing_weights_are_refused() {
    let (h, x0) = b14_fixture();
    let o = StateAverageOptions {
        mals: b14_opts(),
        weights: Some(vec![0.3, 0.7]),
        ..Default::default()
    };
    let _ = state_average_mals(&h, &x0, 2, &o);
}

#[test]
#[should_panic(expected = "sum")]
fn tms_b14_weights_must_sum_to_one() {
    let (h, x0) = b14_fixture();
    let o = StateAverageOptions {
        mals: b14_opts(),
        weights: Some(vec![0.5, 0.4]),
        ..Default::default()
    };
    let _ = state_average_mals(&h, &x0, 2, &o);
}

#[test]
#[should_panic(expected = "MalsOptions::bond_dims must be non-empty")]
fn tms_b14_empty_bond_dims_is_refused() {
    let (h, x0) = b14_fixture();
    let o = sa_opts(MalsOptions {
        bond_dims: vec![],
        ..b14_opts()
    });
    let _ = state_average_mals(&h, &x0, 2, &o);
}

#[test]
#[should_panic(expected = "core position 0")]
fn tms_b14_driver_refuses_a_container_off_core_position_zero() {
    let (h, x0) = b14_fixture();
    let mut m = MultiTt::new(&x0, 2, 0xB14_0021, 0.1);
    m.base.move_core(3, false);
    m.centers = vec![m.base.core(3).clone(), m.base.core(3).clone()];
    m.core_position = 3;
    let _ = state_average_mals_mut(&h, &mut m, &sa_opts(b14_opts()));
}

fn b14_fixture() -> (SparseHamiltonian<f64>, TtVector<f64>) {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (t, v) = degenerate_tv(d);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let mut rng = seeded(0xB14_0100);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);
    (h, x0)
}

fn b14_opts() -> MalsOptions {
    MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 2,
        ..Default::default()
    }
}

#[test]
fn tms_b15_check_accepts_a_fresh_container() {
    let (_, x0) = b14_fixture();
    let m = MultiTt::<f64>::new(&x0, 2, 0xB15_0001, 0.1);
    m.check();
    assert_eq!(m.nstates(), 2, "nstates");
    assert_eq!(
        m.core_position, 0,
        "a fresh container sits at core position 0"
    );
    assert_eq!(m.weights, vec![0.5, 0.5], "default weights are 1/n");
    assert_eq!(m.base.ortho, Some(0), "the shared train is orthogonal at 0");

    let xs = m.states();
    assert_eq!(xs.len(), 2, "states() length");
    let ov = qntt::dot(&xs[0], &xs[1]).abs_s() / (xs[0].norm_lr(false) * xs[1].norm_lr(false));
    println!("tms_b15: |⟨x₀, x₁⟩| / (‖x₀‖‖x₁‖) = {ov:.6}");
    assert!(
        ov < 1.0 - 1e-6,
        "the seeded states are collinear (overlap {ov}) — `scale` did not perturb them"
    );
    let mut mine = xs;
    let before = mine[1].core(0).coeff(0);
    let bumped = mine[0].core(0).coeff(0) + 1.0;
    mine[0].cores[0].set_coeff(0, bumped);
    assert_eq!(
        mine[1].core(0).coeff(0),
        before,
        "states() aliased its clones"
    );
    m.check();
}

#[test]
#[should_panic(expected = "must be orthogonal with ortho == Some")]
fn tms_b15_check_catches_an_orthogonalized_base() {
    let (_, x0) = b14_fixture();
    let mut m = MultiTt::<f64>::new(&x0, 2, 0xB15_0011, 0.1);
    m.check();
    m.base.left_orthogonalize(false);
    m.check();
}

#[test]
#[should_panic(expected = "is not centers[0]")]
fn tms_b15_check_catches_a_value_substituted_centre() {
    let (_, x0) = b14_fixture();
    let mut m = MultiTt::<f64>::new(&x0, 2, 0xB15_0021, 0.1);
    m.check();
    m.base.cores[0] = m.centers[1].clone();
    m.check();
}

#[test]
#[should_panic(expected = "does not have center 0's shape")]
fn tms_b15_check_catches_a_mis_shaped_centre() {
    let (_, x0) = b14_fixture();
    let mut m = MultiTt::<f64>::new(&x0, 2, 0xB15_0031, 0.1);
    m.check();
    m.centers[1] = m.base.core(1).clone();
    m.check();
}

#[test]
#[should_panic(expected = "weights must be non-increasing")]
fn tms_b15_check_catches_hand_mutated_weights() {
    let (_, x0) = b14_fixture();
    let mut m = MultiTt::<f64>::new(&x0, 2, 0xB15_0041, 0.1);
    m.check();
    m.weights = vec![0.2, 0.8];
    m.check();
}

#[test]
#[should_panic(expected = "core_position")]
fn tms_b15_check_catches_a_core_position_past_the_end() {
    let (_, x0) = b14_fixture();
    let mut m = MultiTt::<f64>::new(&x0, 2, 0xB15_0051, 0.1);
    m.check();
    m.core_position = x0.space.d;
    m.check();
}

#[test]
fn tms_b3_enriched_density_matrix_matches_the_dense_oracle() {
    let space = QnSpace::new(6, 1, 1);
    let (_, cp0) = two_site_frame_abelian::<f64>(space, 2, 0xB3_1001);
    let mut rng = seeded(0xB3_1002);
    let cps = unit_blocks_abelian(&cp0, 2, &mut rng);
    let w = [0.7, 0.3];
    let r = 3usize;
    let alpha = 0.25;

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };

        let mut nrng = seeded(0xB3_1003 + forward as u64);
        let mut placed: Option<qntt::qn::Sector> = None;
        let mut absent = 0usize;
        let nd = qntt::qn::SectorGrid::from_fn(&space, |m| {
            let Some(b) = cp0.block(m) else {
                return Mat::<f64>::zeros(0, 0);
            };
            let g = if forward { b.nrows() } else { b.ncols() };
            if g < 2 || placed.is_some() {
                if g > 0 {
                    absent += 1;
                }
                return Mat::<f64>::zeros(0, 0);
            }
            let q = random_orthonormal::<f64>(g, g, &mut nrng);
            let mut blk = Mat::<f64>::zeros(g, g);
            for i in 0..g {
                for j in 0..g {
                    let mut acc = 0.0;
                    for p in 0..g {
                        acc += q[(i, p)] * q[(j, p)] * 0.5f64.powi(p as i32);
                    }
                    *blk.get_mut(i, j) = acc;
                }
            }
            placed = Some(m);
            blk
        });
        let pm = placed.expect("fixture: no mid sector with basis room ≥ 2");
        assert!(
            absent > 0,
            "{tag}: fixture places noise on EVERY populated sector — the absent/`g×g` mix is not \
             exercised"
        );

        let (shared, _s, cs, infos, pooled, _lw) = factor_svd_average(
            &cps,
            &w,
            Some(&nd),
            alpha,
            0.0,
            qntt::Pooled(r),
            forward,
            false,
        );

        let nd_norm = frob_sq(nd[pm].as_ref()).sqrt();
        assert!(nd_norm > 0.0, "{tag}: the noise block is identically zero");
        let alpha_eff = alpha / nd_norm;
        let mut spec: Vec<(qntt::qn::Sector, Vec<f64>, Mat<f64>)> = Vec::new();
        for &m in cp0.mid_qn.iter() {
            let blocks: Vec<Mat<f64>> = cps
                .iter()
                .filter_map(|c| c.block(m).map(|b| b.to_owned()))
                .collect();
            if blocks.len() != cps.len() {
                continue;
            }
            let g = if forward {
                blocks[0].nrows()
            } else {
                blocks[0].ncols()
            };
            if g == 0 {
                continue;
            }
            let mut rho = rho_of_blocks(&blocks, &w, forward);
            if m == pm {
                for i in 0..g {
                    for j in 0..g {
                        *rho.get_mut(i, j) += alpha_eff * nd[m][(i, j)];
                    }
                }
            }
            let (evals, u) = dense_self_adjoint_eigen(rho.as_ref());
            spec.push((m, evals, u));
        }
        let mut pooled_vals: Vec<(f64, qntt::qn::Sector)> = Vec::new();
        for (m, evals, _) in &spec {
            for &l in evals {
                pooled_vals.push((l, *m));
            }
        }
        pooled_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mr = kept_ranks(&shared, forward);
        assert_eq!(pooled.kept, r, "{tag}: pooled kept rank");
        for (m, _, _) in &spec {
            let want = pooled_vals[..r].iter().filter(|p| p.1 == *m).count();
            assert_eq!(
                mr[*m], want,
                "{tag}: sector {m:?} kept rank {} but the top-{r} ENRICHED pooled values place \
                 {want} there",
                mr[*m]
            );
        }

        let mut want_kept = 0.0f64;
        for (m, evals, u) in &spec {
            let rr = mr[*m];
            if rr == 0 {
                continue;
            }
            let g = evals.len();
            let mut q = Mat::<f64>::zeros(g, rr);
            for c in 0..rr {
                for i in 0..g {
                    *q.get_mut(i, c) = u[(i, g - 1 - c)];
                }
            }
            for i in 0..cps.len() {
                let b = cps[i].block(*m).unwrap();
                want_kept += w[i] * kept_weight(q.as_ref(), b, forward);
            }
        }
        let kept: f64 = (0..cps.len()).map(|i| w[i] * cs[i].norm().powi(2)).sum();
        assert!(
            (kept - want_kept).abs() <= 1e-10,
            "{tag}: kept weight {kept:.15e} vs the enriched-eigenspace projection \
             {want_kept:.15e} — the pooled ρ the kernel chopped is not `Σwᵢρᵢ + α_eff·nd`"
        );
        for i in 0..cps.len() {
            let want = 1.0 - cs[i].norm().powi(2);
            assert!(
                (infos[i].discarded_weight - want).abs() <= 1e-12,
                "{tag}: state {i} discarded {:.15e} vs 1 − ‖cᵢ‖² = {want:.15e}",
                infos[i].discarded_weight
            );
        }

        let (_sh0, _s0, cs0, _in0, _p0, _lw0) = factor_svd_average(
            &cps,
            &w,
            Some(&nd),
            0.0,
            0.0,
            qntt::Pooled(r),
            forward,
            false,
        );
        let kept0: f64 = (0..cps.len()).map(|i| w[i] * cs0[i].norm().powi(2)).sum();
        println!("tms_b3 {tag}: kept α = {alpha} → {kept:.12e}, α = 0 → {kept0:.12e}");
        assert!(
            (kept - kept0).abs() > 1e-6,
            "{tag}: VACUOUS — enriching ρ with α = {alpha} kept exactly what α = 0 keeps \
             ({kept:.15e} vs {kept0:.15e}); the noise term is dead or the fixture is degenerate"
        );
        assert!(
            kept <= kept0 + 1e-12,
            "{tag}: the enriched basis keeps MORE state weight ({kept:.15e}) than the Ky Fan \
             optimum of Σwᵢρᵢ ({kept0:.15e})"
        );
    }
}

#[test]
fn tms_b4_driver_forwards_the_reorthonormalize_flag() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let fx = gapped_tv(0xB4_9000, 3, 0.05);
    let h = fx.abelian(space);
    let n = 3usize;
    let w = vec![0.5, 0.3, 0.2];
    let mals = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![3],
        noise: vec![0.0],
        max_iter: 3,
        ..Default::default()
    };
    let mut rng = seeded(0xB4_9001);
    let x0: TtVector<f64> = randomized_state(space, 5, 10, &mut rng);

    let mut devs = Vec::new();
    for reorth in [true, false] {
        let o = StateAverageOptions {
            mals: mals.clone(),
            weights: Some(w.clone()),
            reorthonormalize: reorth,
            ..Default::default()
        };
        let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &o);
        let gd = gram_dev_tt(&ys);
        println!("tms_b4 driver: reorthonormalize = {reorth} → returned Gram deviation {gd:.3e}");
        devs.push(gd);
        for (it, sw) in hist.sweeps.iter().enumerate() {
            assert_weighted_ky_fan(sw, &w, &fx.ev11, &format!("reorth = {reorth}: sweep {it}"));
        }
    }
    assert!(
        devs[0] <= 1e-8,
        "reorthonormalize = true returned a Gram {:.3e} from the identity",
        devs[0]
    );
    assert!(
        devs[1] > devs[0] + 1e-6,
        "VACUOUS — reorthonormalize = false returned a Gram deviation {:.3e} no larger than the \
         re-orthonormalized run's {:.3e}: the flag is not reaching the kernel, or this cap does \
         not truncate enough for the drift to be observable",
        devs[1],
        devs[0]
    );
}

#[test]
#[should_panic(expected = "sa_restart_keep: maxsubspace = 3 < 2·nroots = 4")]
fn tms_b11_sa_restart_keep_refuses_a_maxsubspace_below_two_nroots() {
    let _ = sa_restart_keep(2, 3);
}

#[test]
fn tms_b11_sa_restart_keep_defaults_are_in_range() {
    for n in 1..=6 {
        let ms = sa_maxsubspace(n);
        let keep = sa_restart_keep(n, ms);
        assert!(
            keep >= n.min(keep.max(1)) && keep < ms,
            "n = {n}: keep {keep}, maxsubspace {ms}"
        );
        if n == 1 {
            assert_eq!(
                keep, 1,
                "the n = 1 carve-out keeps the single-root restart rule"
            );
        } else {
            assert_eq!(keep, (2 * n).min(ms - 1));
        }
    }
}
