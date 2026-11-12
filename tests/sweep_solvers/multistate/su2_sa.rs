// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use qntt::solvers::mals::{MalsOptions, NoiseType};
use qntt::solvers::multistate::{SaSweep, StateAverageOptions};
use qntt::su2::expand::expand;
use qntt::su2::solvers::mals::{mals, mals_with_history};
use qntt::su2::solvers::multistate::{
    MultiTt as Su2MultiTt, state_average_mals, state_average_mals_mut,
    state_average_mals_with_history,
};
use qntt::su2::twosite::{factor_qc as su2_factor_qc, factor_svd_average};
use qntt_testkit::randomized_su2_state;

fn zero_t(d: usize) -> CoeffTensor<f64> {
    CoeffTensor::<f64>::zeros(d, 2)
}

fn hb_tensor(d: usize, entries: &[(usize, usize, usize, usize, f64)]) -> CoeffTensor<f64> {
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for &(i, j, k, l, val) in entries {
        v.set(&[i, j, k, l], val);
    }
    v
}

fn hb_su2(space: Su2Space) -> Su2SparseHamiltonian<f64> {
    let v = hb_tensor(space.d, &[(2, 0, 3, 1, -1.0), (1, 3, 0, 2, -1.0)]);
    Su2SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
}

fn hb_polarized_su2(
    space: Su2Space,
    a: usize,
    b: usize,
    c: usize,
    e: usize,
) -> Su2SparseHamiltonian<f64> {
    let v = hb_tensor(space.d, &[(b, a, c, e, -1.0), (a, b, e, c, -1.0)]);
    Su2SparseHamiltonian::new(space, &zero_t(space.d), &v, None, 1e-12)
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

fn max_total_bond_rank<T: Scalar>(x: &Su2TtVector<T>) -> usize {
    x.ranks.iter().map(|g| g.total()).max().unwrap_or(0)
}

fn kept_ranks<T: Scalar>(shared: &Su2Core<T>, forward: bool) -> Su2RankGrid {
    if forward {
        shared.col_ranks.clone()
    } else {
        shared.row_ranks.clone()
    }
}

fn core_gram_dev_su2<T: Scalar>(cs: &[Su2Core<T>]) -> f64 {
    let mut dev = 0.0f64;
    for i in 0..cs.len() {
        for j in 0..cs.len() {
            let want = if i == j { T::one() } else { T::zero() };
            dev = dev.max((cs[i].dot_center(&cs[j]) - want).abs_s());
        }
    }
    dev
}

fn core_gram_su2(cs: &[Su2Core<f64>]) -> (f64, Vec<f64>) {
    let n = cs.len();
    let mut g = Mat::<f64>::zeros(n, n);
    let mut off = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let v = cs[i].dot_center(&cs[j]);
            *g.get_mut(i, j) = v;
            if i != j {
                off = off.max(v.abs());
            }
        }
    }
    let (evals, _) = dense_self_adjoint_eigen(g.as_ref());
    (off, evals)
}

fn reapply_theta<T: Scalar>(core: &Su2Core<T>) -> Su2Core<T> {
    let mut out = core.clone();
    for l in core.row_qn.iter().copied() {
        if core.row_ranks[l] == 0 {
            continue;
        }
        let mut u = out.horizontal_unfolding(l);
        if u.ncols() == 0 {
            continue;
        }
        out.scale_horizontal(&mut u, l);
        out.scatter_horizontal(l, u.as_ref());
    }
    out
}

fn expand_norm_sq(x: &Su2TtVector<f64>, two_sz: i64) -> f64 {
    expand(x, two_sz).to_dense().iter().map(|c| c * c).sum()
}

#[test]
fn tms_b1su2_state_average_n1_is_mals_bitwise() {
    let d = 6usize;
    let cases: [(Su2Space, &str); 2] = [
        (Su2Space::new(d, 2, 0), "2S=0"),
        (Su2Space::new(d, 2, 2), "2S=2"),
    ];
    for (space, label) in cases {
        let h = if space.two_s == 0 {
            hb_su2(space)
        } else {
            hb_polarized_su2(space, 0, 1, 2, 3)
        };
        let mut rng = seeded(0x5A_1000 + space.two_s as u64);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
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

            assert_eq!(sys.len(), 1, "{label}/{arm}: one state at nroots = 1");
            assert_eq!(
                shist.sweeps.len(),
                mhist.sweeps.len(),
                "{label}/{arm}: iteration count"
            );
            assert_f64_bits(
                shist.lambdas[0],
                mhist.lambda,
                &format!("{label}/{arm}: final λ"),
            );

            for (it, (s, m)) in shist.sweeps.iter().zip(mhist.sweeps.iter()).enumerate() {
                let tag = |w: &str| format!("{label}/{arm}: sweep {it} {w}");
                assert_eq!(s.energies.len(), 1, "{}", tag("energies length"));
                assert_f64_bits(s.energies[0], m.energy, &tag("energies[0]"));
                assert_f64_bits(s.energy, m.energy, &tag("energy"));
                assert_f64_bits(
                    s.discarded_weight,
                    m.discarded_weight,
                    &tag("discarded_weight"),
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
                assert!(s.numops > 0, "{}", tag("numops"));
            }
            coeffs_bits_eq_su2(&sys[0], &my, &format!("{label}/{arm}: returned state"));

            let (slam, _) = state_average_mals(&h, &x0, 1, &o);
            let (mlam, _) = mals(&h, &x0, &opts);
            assert_f64_bits(
                slam[0],
                mlam,
                &format!("{label}/{arm}: state_average_mals λ"),
            );

            if space.two_s == 2 {
                assert_theta_cw_live(&sys[0], &format!("{label}/{arm}"));
            }
        }
    }
}

#[test]
fn tms_b2su2_full_rank_average_reconstructs_each_block() {
    let space = Su2Space::new(6, 2, 2);
    let k = 2usize;
    let (x, cp0) = two_site_frame_su2::<f64>(space, k, 0xB2_5001);
    assert_theta_cw_live(&x, "tms_b2su2 frame");
    let mut rng = seeded(0xB2_5002);
    let cps = unit_blocks_su2(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    assert!(
        cp0.num_coeffs() >= 3,
        "fixture: local space cannot host 3 states"
    );

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };
        let (shared, s, cs, infos, pooled, lw) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled::NONE, forward, false);

        for (i, info) in infos.iter().enumerate() {
            assert!(
                info.discarded_weight <= 1e-12,
                "{tag}: state {i} discarded {:.3e} at full rank",
                info.discarded_weight
            );
        }
        assert!(pooled.discarded_weight <= 1e-12, "{tag}: pooled discarded");

        let mr = kept_ranks(&shared, forward);
        for m in s.sectors() {
            let Some(b) = s.block(m) else { continue };
            let r = mr[m];
            assert_eq!(
                (b.nrows(), b.ncols()),
                (r, r),
                "{tag}: gauge shape at {m:?}"
            );
            for i in 0..r {
                for j in 0..r {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert_eq!(*b.get(i, j), want, "{tag}: gauge at {m:?} ({i},{j})");
                }
            }
        }

        for i in 0..3 {
            let r = if forward {
                su2_contract(&shared, &cs[i])
            } else {
                su2_contract(&cs[i], &shared)
            };
            assert_eq!(
                r.num_coeffs(),
                cps[i].num_coeffs(),
                "{tag}: state {i} coeff count"
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

        let mut g = Mat::<f64>::zeros(3, 3);
        let mut off = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let v = cps[i].dot_cores(&cps[j]);
                *g.get_mut(i, j) = v;
                if i != j {
                    off = off.max(v.abs());
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
            (lw.gram_min - bevals[0]).abs() <= 1e-12 && (lw.gram_max - bevals[2]).abs() <= 1e-12,
            "{tag}: (gram_min, gram_max) = ({}, {}) vs ({}, {})",
            lw.gram_min,
            lw.gram_max,
            bevals[0],
            bevals[2]
        );

        if !forward {
            assert!(
                count_core_theta(&shared) >= 2,
                "{tag}: VACUOUS — the shared Vᴴ carries fewer than 2 live θ ≠ 1 slices"
            );
            let mut psi = x.clone();
            psi.set_cores(cs[0].clone(), shared.clone());
            let dev = su2_right_canonicity_dev(&psi, (k + 1)..(k + 2));
            assert!(
                dev <= 1e-12,
                "{tag}: the written-back Vᴴ is not θ-right-canonical (Σ_r θ² A Aᴴ − I = \
                 {dev:.3e}); the `unscale_horizontal` on the horizontal write-back is missing"
            );
        }
    }
}

#[test]
fn tms_b3su2_weighted_ky_fan_and_trace_loss() {
    let space = Su2Space::new(6, 2, 2);
    let (x, cp0) = two_site_frame_su2::<f64>(space, 2, 0xB3_5001);
    assert_theta_cw_live(&x, "tms_b3su2 frame");
    let mut rng = seeded(0xB3_5002);
    let cps = unit_blocks_su2(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    let r = 3usize;

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };
        let (shared, _s, cs, infos, pooled, _lw) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), forward, false);

        let spec = pooled_rho_su2(&cps, &w, forward);
        let mut pooled_vals: Vec<(f64, Su2Sector)> = Vec::new();
        for (m, evals, _) in &spec {
            for &l in evals {
                pooled_vals.push((l, *m));
            }
        }
        pooled_vals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let tr: f64 = pooled_vals.iter().map(|p| p.0).sum();
        assert!(
            (tr - 1.0).abs() <= 1e-12,
            "{tag}: Σ_m cw(m)·tr ρ_θ(m) = {tr} — must be Σᵢwᵢ‖Bᵢ‖² = 1 in the package metric"
        );
        let top: f64 = pooled_vals[..r].iter().map(|p| p.0).sum();

        let kept_w: f64 = (0..3).map(|i| w[i] * cs[i].norm_center().powi(2)).sum();
        assert!(
            (kept_w - top).abs() <= 1e-12,
            "{tag}: kept weight {kept_w:.15e} vs top-{r} pooled cw·λ sum {top:.15e}"
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
            "{tag}: Σwᵢdroppedᵢ = {wsum:.15e} vs pooled {:.15e} — the per-state ledger is not \
             centerweighted (clause B8)",
            pooled.discarded_weight
        );
        for i in 0..3 {
            let want = 1.0 - cs[i].norm_center().powi(2);
            assert!(
                (infos[i].discarded_weight - want).abs() <= 1e-12,
                "{tag}: state {i} discarded {:.15e} vs 1 − ‖cᵢ‖²_center = {want:.15e}",
                infos[i].discarded_weight
            );
        }
        assert_eq!(pooled.kept, r, "{tag}: pooled kept rank");

        let mr = kept_ranks(&shared, forward);
        for (m, _, _) in &spec {
            let want = pooled_vals[..r].iter().filter(|p| p.1 == *m).count();
            assert_eq!(
                mr[*m], want,
                "{tag}: sector {m:?} kept rank {} but the top-{r} pooled values place {want} there",
                mr[*m]
            );
        }

        let mut arng = seeded(0xB3_5003 + forward as u64);
        for trial in 0..8 {
            let mut alt = 0.0f64;
            for (m, evals, _) in &spec {
                let rr = mr[*m];
                if rr == 0 {
                    continue;
                }
                let cw = qntt::su2::centerweight(m.two_s, space.two_s);
                let q = random_orthonormal::<f64>(evals.len(), rr, &mut arng);
                for i in 0..3 {
                    let bt = b_theta(&cps[i], *m).unwrap();
                    alt += w[i] * cw * kept_weight(q.as_ref(), bt.as_ref(), forward);
                }
            }
            assert!(
                alt <= top + 1e-12,
                "{tag}: trial {trial} random basis keeps {alt:.15e} > the Ky Fan optimum {top:.15e}"
            );
        }
    }
}

#[test]
fn tms_b4su2_loewdin_gram_is_identity_after_writeback() {
    let space = Su2Space::new(6, 2, 2);
    let k = 2usize;
    let (x, cp0) = two_site_frame_su2::<f64>(space, k, 0xB4_5001);
    assert_theta_cw_live(&x, "tms_b4su2 frame");
    let mut rng = seeded(0xB4_5002);
    let cps = unit_blocks_su2(&cp0, 3, &mut rng);
    let w = [0.5, 0.3, 0.2];
    let r = 3usize;

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };
        let (_sa, _ga, plain, _ia, _pa, la) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), forward, false);
        let (_sb, _gb, ortho, _ib, _pb, lb) =
            factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), forward, true);

        let (off, evals) = core_gram_su2(&plain);
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
            "{tag}: VACUOUS — the pre-Löwdin centres are already orthogonal to {off:.3e}"
        );

        let dev = core_gram_dev_su2(&ortho);
        assert!(
            dev <= 1e-12,
            "{tag}: post-Löwdin `dot_center` Gram deviation {dev:.3e}"
        );
        for (i, c) in ortho.iter().enumerate() {
            assert!(
                (c.norm_center() - 1.0).abs() <= 1e-12,
                "{tag}: centre {i} centerweighted norm {}",
                c.norm_center()
            );
        }

        if forward {
            let live: usize = ortho.iter().map(count_core_theta).sum();
            assert!(
                live >= 2,
                "{tag}: VACUOUS — the written-back centres carry fewer than 2 live θ ≠ 1 slices"
            );
            let scaled: Vec<Su2Core<f64>> = ortho.iter().map(reapply_theta).collect();
            let bad = core_gram_dev_su2(&scaled);
            println!("tms_b4su2 {tag}: Gram deviation without the unscale = {bad:.6e}");
            assert!(
                bad > 1e-3,
                "{tag}: ANTI-ORACLE FAILED — re-applying θ leaves the Gram at {bad:.3e}, so this \
                 fixture cannot tell the unscaled write-back from the raw one"
            );
        }
    }
}

#[test]
#[should_panic(expected = "collapsed")]
fn tms_b4su2_loewdin_collapse_panics() {
    let space = Su2Space::new(6, 2, 2);
    let (_, cp0) = two_site_frame_su2::<f64>(space, 2, 0xB4_5011);
    let mut rng = seeded(0xB4_5012);
    let cps = unit_blocks_su2(&cp0, 1, &mut rng);
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
fn tms_b5su2_theta_anti_oracles() {
    let space = Su2Space::new(6, 2, 2);
    let k = 2usize;
    let (x, cp0) = two_site_frame_su2::<f64>(space, k, 0xB5_0001);
    assert_theta_cw_live(&x, "tms_b5su2 frame");
    let mut rng = seeded(0xB5_0006);
    let cps = unit_blocks_su2(&cp0, 2, &mut rng);
    let w = [0.7, 0.3];
    let r = 3usize;

    let (shared, _s, cs, infos, pooled, _lw) =
        factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), true, false);
    assert!(
        count_core_theta(&cs[0]) >= 2,
        "VACUOUS — the written-back centre carries fewer than 2 live θ ≠ 1 slices"
    );

    let mut loss_shipped = [0.0f64; 2];
    let mut ref_norm = [0.0f64; 2];
    for i in 0..2 {
        let (a, b) = su2_factor_qc(&cps[i]);
        let mut xr = x.clone();
        xr.set_cores(a, b);
        xr.ortho = None;
        ref_norm[i] = expand_norm_sq(&xr, 2);
        assert!(ref_norm[i] > 1e-14, "state {i}: degenerate reference norm");
        let alt = expand_norm_sq(&xr, 0);
        assert!(
            alt.is_finite(),
            "state {i}: Sz = 0 member is not representable"
        );

        let mut psi = x.clone();
        psi.set_cores(shared.clone(), cs[i].clone());
        psi.ortho = None;
        loss_shipped[i] = 1.0 - expand_norm_sq(&psi, 2) / ref_norm[i];
        assert!(
            (infos[i].discarded_weight - loss_shipped[i]).abs() <= 1e-10,
            "state {i}: reported discarded weight {:.12e} vs the expand-measured projection loss \
             {:.12e} — the per-state ledger is not the loss in the package metric",
            infos[i].discarded_weight,
            loss_shipped[i]
        );
    }
    let pooled_measured: f64 = (0..2).map(|i| w[i] * loss_shipped[i]).sum();
    assert!(
        (pooled.discarded_weight - pooled_measured).abs() <= 1e-10,
        "pooled discarded {:.12e} vs Σwᵢ·(measured lossᵢ) {:.12e}",
        pooled.discarded_weight,
        pooled_measured
    );

    let mr = kept_ranks(&shared, true);
    let mut xk_raw = Su2Core::<f64>::zeros(space, k, &cp0.row_ranks, &mr);
    let mut centres_raw: Vec<Su2Core<f64>> = (0..2)
        .map(|_| Su2Core::<f64>::zeros(space, k + 1, &mr, &cp0.col_ranks))
        .collect();
    let mut dw_raw = [0.0f64; 2];
    let mut tot_raw = [0.0f64; 2];
    for &m in cp0.mid_qn.iter() {
        let Some(b0) = cps[0].block(m) else { continue };
        let (rows, cols) = (b0.nrows(), b0.ncols());
        if rows == 0 || cols == 0 {
            continue;
        }
        let rr = mr[m];
        let mut rho = Mat::<f64>::zeros(rows, rows);
        for (i, cp) in cps.iter().enumerate() {
            let b = cp.block(m).unwrap();
            for p in 0..rows {
                for q in 0..rows {
                    let mut acc = 0.0f64;
                    for c in 0..cols {
                        acc += *b.get(p, c) * *b.get(q, c);
                    }
                    *rho.get_mut(p, q) += w[i] * acc;
                }
            }
        }
        let (_evals, u) = dense_self_adjoint_eigen(rho.as_ref());
        let mut ub = Mat::<f64>::zeros(rows, rr);
        for c in 0..rr {
            for p in 0..rows {
                *ub.get_mut(p, c) = *u.get(p, rows - 1 - c);
            }
        }
        if rr > 0 {
            xk_raw.scatter_vertical(m, ub.as_ref());
        }
        for (i, cp) in cps.iter().enumerate() {
            let b = cp.block(m).unwrap();
            let mut cmat = Mat::<f64>::zeros(rr, cols);
            for p in 0..rr {
                for c in 0..cols {
                    let mut acc = 0.0f64;
                    for q in 0..rows {
                        acc += *ub.get(q, p) * *b.get(q, c);
                    }
                    *cmat.get_mut(p, c) = acc;
                }
            }
            tot_raw[i] += frob_sq(b);
            dw_raw[i] += frob_sq(b) - frob_sq(cmat.as_ref());
            if rr > 0 {
                centres_raw[i].scatter_horizontal(m, cmat.as_ref());
            }
        }
    }
    let mut loss_raw = [0.0f64; 2];
    for i in 0..2 {
        dw_raw[i] /= tot_raw[i].max(1e-300);
        let mut psi = x.clone();
        psi.set_cores(xk_raw.clone(), centres_raw[i].clone());
        psi.ortho = None;
        loss_raw[i] = 1.0 - expand_norm_sq(&psi, 2) / ref_norm[i];
    }
    println!(
        "tms_b5su2: shipped loss {loss_shipped:?}, raw-ρ bookkeeping {dw_raw:?}, raw-ρ measured \
         loss {loss_raw:?}"
    );
    let disagreement = (0..2)
        .map(|i| (dw_raw[i] - loss_raw[i]).abs())
        .fold(0.0f64, f64::max);
    assert!(
        disagreement > 1e-3,
        "ANTI-ORACLE FAILED: the raw (θ-blind) bookkeeping agrees with the true projection loss \
         to {disagreement:.3e} on this fixture, so it cannot distinguish the θ-scaled ρ from the \
         raw one — change the fixture, never the threshold"
    );
    let shipped_w: f64 = (0..2).map(|i| w[i] * loss_shipped[i]).sum();
    let raw_w: f64 = (0..2).map(|i| w[i] * loss_raw[i]).sum();
    assert!(
        shipped_w <= raw_w + 1e-12,
        "the shipped basis loses more WEIGHTED weight ({shipped_w:.12e}) than the raw-ρ basis \
         ({raw_w:.12e}) at the same per-sector ranks — the shipped ρ is not the θ-metric optimum"
    );

    let (shared_b, _sb, cs_b, _ib, _pb, _lb) =
        factor_svd_average(&cps, &w, None, 0.0, 0.0, qntt::Pooled(r), false, false);
    assert!(
        count_core_theta(&shared_b) >= 2,
        "VACUOUS — the backward shared Vᴴ carries fewer than 2 live θ ≠ 1 slices"
    );
    let mut psi = x.clone();
    psi.set_cores(cs_b[0].clone(), shared_b.clone());
    let good = su2_right_canonicity_dev(&psi, (k + 1)..(k + 2));
    assert!(
        good <= 1e-12,
        "the shipped Vᴴ is not θ-right-canonical (Σ_r θ² A Aᴴ − I = {good:.3e})"
    );
    let mut psi_raw = x.clone();
    psi_raw.set_cores(cs_b[0].clone(), reapply_theta(&shared_b));
    let bad = su2_right_canonicity_dev(&psi_raw, (k + 1)..(k + 2));
    println!("tms_b5su2: right-canonicity dev shipped {good:.3e}, θ-re-applied {bad:.3e}");
    assert!(
        bad > 1e-3,
        "ANTI-ORACLE FAILED: re-applying θ leaves right-canonicity at {bad:.3e}"
    );
    let n_good = expand_norm_sq(&psi, 2);
    let n_bad = expand_norm_sq(&psi_raw, 2);
    assert!(
        (n_good - n_bad).abs() > 1e-3 * n_good.max(1e-300),
        "ANTI-ORACLE FAILED: the θ-re-applied core represents the same state (‖·‖² {n_good:.6e} \
         vs {n_bad:.6e})"
    );
}

#[test]
fn tms_b6su2_full_rank_state_average_matches_the_dense_spectrum() {
    let d = 6usize;
    let fx = gapped_tv(0xB6_5000, 3, 0.05);
    let n = 3usize;
    let w = vec![1.0 / 3.0; 3];
    let opts = opts_exact();
    let slevels = fx.singlet_levels();
    assert_exclusion_gap(&slevels, n, 0.05, "tms_b6su2 singlets");
    assert_exclusion_gap(&fx.ev20, n, 0.05, "tms_b6su2 triplets");

    let arms: [(usize, i64, &str); 2] = [(0, 0, "2S=0"), (2, 2, "2S=2")];
    for (two_s, two_sz, label) in arms {
        let space = Su2Space::new(d, 2, two_s);
        let h = fx.su2(space);
        let (levels, dets, evecs): (Vec<f64>, &Vec<(Vec<bool>, Vec<bool>)>, Vec<Vec<f64>>) =
            if two_s == 0 {
                (
                    slevels.clone(),
                    &fx.dets11,
                    fx.singlets.iter().map(|&i| fx.evec11[i].clone()).collect(),
                )
            } else {
                (fx.ev20.clone(), &fx.dets20, fx.evec20.clone())
            };

        let mut best: Option<(qntt::solvers::multistate::SaHistory, Vec<Su2TtVector<f64>>)> = None;
        for trial in 0..3u64 {
            let mut rng = seeded(0xB6_6000 + 0x100 * two_s as u64 + trial);
            let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
            assert!(x0.check());
            let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &sa_opts(opts.clone()));
            let ok = (0..n).all(|i| (hist.lambdas[i] - levels[i]).abs() <= 1e-8);
            best = Some((hist, ys));
            if ok {
                break;
            }
        }
        let (hist, ys) = best.unwrap();

        for i in 0..n {
            assert!(
                (hist.lambdas[i] - levels[i]).abs() <= 1e-8,
                "{label}: root {i} λ = {:.12} vs the dense level {:.12}",
                hist.lambdas[i],
                levels[i]
            );
            assert!(ys[i].check(), "{label}: root {i} check()");
            let nn = qntt::su2::ops::dot(&ys[i], &ys[i]).abs();
            assert!((nn - 1.0).abs() <= 1e-10, "{label}: root {i} ‖x‖² = {nn}");
            let resid = global_eigen_residual_su2(&h, &ys[i], hist.lambdas[i]);
            assert!(resid <= 1e-6, "{label}: root {i} residual {resid:.3e}");
            let proj = projection_on_dense(&expand(&ys[i], two_sz), dets, &evecs[i]);
            assert!(
                proj >= 1.0 - 1e-8,
                "{label}: root {i} projection on the dense eigenvector {proj:.12}"
            );
            if two_s == 2 {
                assert_theta_cw_live(&ys[i], &format!("{label} root {i}"));
            }
        }
        let gd = gram_dev_su2(&ys);
        assert!(gd <= 1e-8, "{label}: returned Gram deviation {gd:.3e}");

        let last = hist.sweeps.last().unwrap();
        assert_eq!(last.energies.len(), n, "{label}: history energies length");
        assert_eq!(last.discarded_weights.len(), n, "{label}: ledger length");
        for i in 1..n {
            assert!(
                last.energies[i] >= last.energies[i - 1],
                "{label}: homing = false must label by ENERGY: {:?}",
                last.energies
            );
        }
        assert!(
            (last.energy - weighted(&w, &last.energies)).abs() <= 1e-14,
            "{label}: history energy vs Σwᵢλᵢ"
        );
        for (it, sw) in hist.sweeps.iter().enumerate() {
            assert!(sw.numops > 0, "{label}: sweep {it} numops");
        }
    }
}

#[test]
fn tms_b7su2_truncated_state_average_ky_fan_and_host_n_states() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xB7_5000, 3, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let n = 3usize;
    let w = vec![0.5, 0.3, 0.2];
    let caps = vec![2usize, 3, 4];
    let o = StateAverageOptions {
        mals: MalsOptions {
            eps: 1e-8,
            bond_dims: caps.clone(),
            noise: vec![0.0],
            max_iter: 8,
            ..Default::default()
        },
        weights: Some(w.clone()),
        ..Default::default()
    };
    let mut rng = seeded(0xB7_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &o);

    for (it, sw) in hist.sweeps.iter().enumerate() {
        assert_eq!(sw.noise, 0.0, "sweep {it}: noise-free schedule");
        assert_weighted_ky_fan(sw, &w, &levels, &format!("sweep {it}"));
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
        let ky = weighted(&w, &levels[..nl]);
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
            "root {i}: returned ranks differ from root 0 (shared basis, round-free driver)"
        );
    }
    for y in &ys {
        assert!(y.check());
    }
    let gd = gram_dev_su2(&ys);
    assert!(gd <= 1e-10, "returned Gram deviation {gd:.3e}");
}

#[test]
fn tms_b9su2_homing_is_anchored_and_matches_energy_labels() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xB9_5000, 3, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let n = 3usize;
    let opts = opts_exact();

    for seed in 0xB9_6000u64..0xB9_6005 {
        let mut rng = seeded(seed);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 6, 12, &mut rng);
        let homed = StateAverageOptions {
            mals: opts.clone(),
            homing: true,
            ..Default::default()
        };
        let (ha, _) = state_average_mals_with_history(&h, &x0, n, &homed);
        let (hb, _) = state_average_mals_with_history(&h, &x0, n, &sa_opts(opts.clone()));
        for i in 1..n {
            assert!(
                ha.lambdas[i] >= ha.lambdas[i - 1] - 1e-12,
                "seed {seed:#x}: homed labels not energy-sorted: {:?}",
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
                (ha.lambdas[i] - levels[i]).abs() <= 1e-8,
                "seed {seed:#x}: root {i} λ {} vs singlet level {}",
                ha.lambdas[i],
                levels[i]
            );
        }
    }
}

#[test]
fn tms_b11su2_perturbative_state_average_grows_rank_and_stays_variational() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 2);
    let fx = gapped_tv(0xBB_5000, 3, 0.05);
    let h = fx.su2(space);
    let n = 3usize;
    let w = vec![1.0 / 3.0; 3];

    let mut rng = seeded(0xBB_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut rng);
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
        assert_weighted_ky_fan(sw, &w, &fx.ev20, &format!("sweep {it}"));
    }
    let final_rank = hist.sweeps.last().unwrap().max_bond_dim;
    println!("tms_b11su2: start reduced rank {r0} → final max_bond_dim {final_rank}");
    assert!(
        final_rank >= n,
        "final max_bond_dim {final_rank} below nroots {n}"
    );
    for i in 0..n {
        assert!(
            (hist.lambdas[i] - fx.ev20[i]).abs() <= 1e-6,
            "root {i}: λ = {} vs the triplet level {} after the ramp reached 0",
            hist.lambdas[i],
            fx.ev20[i]
        );
        assert!(ys[i].check(), "root {i}: check()");
        assert_theta_cw_live(&ys[i], &format!("tms_b11su2 root {i}"));
    }
    let gd = gram_dev_su2(&ys);
    assert!(gd <= 1e-8, "returned Gram deviation {gd:.3e}");
}

#[test]
fn tms_b12su2_threaded_matches_serial_bitwise() {
    let d = 6usize;
    let space = Su2Space::new(d, 4, 0);
    let fx = gapped_tv(0xBC_5000, 3, 0.05);
    let h = fx.su2(space);
    let n = 3usize;
    let o = sa_opts(MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 3,
        ..Default::default()
    });
    let mut rng = seeded(0xBC_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

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
        .map(|k| su2_contract(y1[0].core(k), y1[0].core(k + 1)).mid_qn.len())
        .max()
        .unwrap_or(0);
    println!("tms_b12su2: widest visited bond carries {widest} mid sectors");
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
    }
    println!("tms_b12su2: max |Δ| between the 1- and 8-thread trajectories = {worst:.3e}");

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
        coeffs_bits_eq_su2(&y1[i], &y8[i], &format!("root {i} state"));
    }
}

fn b14su2_fixture() -> (Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
    let space = Su2Space::new(6, 2, 0);
    let h = hb_su2(space);
    let mut rng = seeded(0xB14_5100);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    (h, x0)
}

fn b14su2_opts() -> MalsOptions {
    MalsOptions {
        eps: 1e-8,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 2,
        ..Default::default()
    }
}

#[test]
#[should_panic(expected = "the two-site local space at")]
fn tms_b14su2_multitt_new_refuses_a_csf_start() {
    let codes = [
        Su2Family::Occ,
        Su2Family::Empty,
        Su2Family::Empty,
        Su2Family::Empty,
        Su2Family::Empty,
        Su2Family::Empty,
    ];
    let csf: Su2TtVector<f64> = qntt::su2::tt_csf(&codes);
    assert_eq!(csf.space, Su2Space::new(6, 2, 0), "CSF endpoint");
    assert_eq!(
        max_total_bond_rank(&csf),
        1,
        "the start must be a rank-1 CSF"
    );
    let _ = Su2MultiTt::new(&csf, 2, 0xB14_5001, 0.1);
}

#[test]
#[should_panic(expected = "state-average sweeps define only Perturbative noise")]
fn tms_b14su2_random_noise_is_refused() {
    let (h, x0) = b14su2_fixture();
    let o = sa_opts(MalsOptions {
        noise: vec![1e-4, 0.0],
        noise_type: NoiseType::Random,
        ..b14su2_opts()
    });
    let _ = state_average_mals(&h, &x0, 2, &o);
}

#[test]
#[should_panic(expected = "non-increasing")]
fn tms_b14su2_increasing_weights_are_refused() {
    let (h, x0) = b14su2_fixture();
    let o = StateAverageOptions {
        mals: b14su2_opts(),
        weights: Some(vec![0.3, 0.7]),
        ..Default::default()
    };
    let _ = state_average_mals(&h, &x0, 2, &o);
}

#[test]
#[should_panic(expected = "core position 0")]
fn tms_b14su2_driver_refuses_a_container_off_core_position_zero() {
    let (h, x0) = b14su2_fixture();
    let mut m = Su2MultiTt::new(&x0, 2, 0xB14_5011, 0.1);
    m.base.move_core(3, false);
    m.centers = vec![m.base.core(3).clone(), m.base.core(3).clone()];
    m.core_position = 3;
    let _ = state_average_mals_mut(&h, &mut m, &sa_opts(b14su2_opts()));
}

#[test]
fn tms_b15su2_check_accepts_a_fresh_container() {
    let (_, x0) = b14su2_fixture();
    let m = Su2MultiTt::<f64>::new(&x0, 2, 0xB15_5001, 0.1);
    m.check();
    assert_eq!(m.nstates(), 2, "nstates");
    assert_eq!(m.core_position, 0, "fresh container core position");
    assert_eq!(m.base.ortho, Some(0), "fresh container ortho flag");
    let xs = m.states();
    let ov =
        qntt::su2::ops::dot(&xs[0], &xs[1]).abs_s() / (xs[0].norm_lr(false) * xs[1].norm_lr(false));
    println!("tms_b15su2: |⟨x₀, x₁⟩| / (‖x₀‖‖x₁‖) = {ov:.6}");
    assert!(
        ov < 1.0 - 1e-6,
        "the seeded states are collinear (overlap {ov})"
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
fn tms_b15su2_check_catches_an_orthogonalized_base() {
    let (_, x0) = b14su2_fixture();
    let mut m = Su2MultiTt::<f64>::new(&x0, 2, 0xB15_5011, 0.1);
    m.check();
    m.base.left_orthogonalize(false);
    m.check();
}

#[test]
#[should_panic(expected = "is not centers[0]")]
fn tms_b15su2_check_catches_a_value_substituted_centre() {
    let (_, x0) = b14su2_fixture();
    let mut m = Su2MultiTt::<f64>::new(&x0, 2, 0xB15_5021, 0.1);
    m.check();
    m.base.cores[0] = m.centers[1].clone();
    m.check();
}

#[test]
#[should_panic(expected = "does not have center 0's shape")]
fn tms_b15su2_check_catches_a_mis_shaped_centre() {
    let (_, x0) = b14su2_fixture();
    let mut m = Su2MultiTt::<f64>::new(&x0, 2, 0xB15_5031, 0.1);
    m.check();
    m.centers[1] = m.base.core(1).clone();
    m.check();
}

#[test]
#[should_panic(expected = "weights must be non-increasing")]
fn tms_b15su2_check_catches_hand_mutated_weights() {
    let (_, x0) = b14su2_fixture();
    let mut m = Su2MultiTt::<f64>::new(&x0, 2, 0xB15_5041, 0.1);
    m.check();
    m.weights = vec![0.2, 0.8];
    m.check();
}

#[test]
#[should_panic(expected = "core_position")]
fn tms_b15su2_check_catches_a_core_position_past_the_end() {
    let (_, x0) = b14su2_fixture();
    let mut m = Su2MultiTt::<f64>::new(&x0, 2, 0xB15_5051, 0.1);
    m.check();
    m.core_position = x0.space.d;
    m.check();
}

#[test]
fn tms_b13su2_per_state_discarded_weights_expose_the_light_root() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xBD_5000, 3, 0.05);
    let h = fx.su2(space);
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
    let mut rng = seeded(0xBD_5001);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
    let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &o);
    for (i, y) in ys.iter().enumerate() {
        assert_theta_cw_live(y, &format!("tms_b13su2 root {i}"));
    }
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
        "tms_b13su2: pooled {:.6e}, per-state {:?}, light-root ratio {:.3}",
        s.discarded_weight,
        s.discarded_weights,
        s.discarded_weights[2] / s.discarded_weight.max(1e-300)
    );
    assert!(
        dmax >= s.discarded_weight - 1e-15,
        "max per-state loss {dmax:.6e} below the pooled max {:.6e}",
        s.discarded_weight
    );
    assert!(
        dmax > s.discarded_weight,
        "VACUOUS — the per-state ledger carries no information the pooled number does not \
         (max {dmax:.6e} vs pooled {:.6e})",
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
fn tms_b3su2_enriched_density_matrix_matches_the_dense_oracle() {
    let space = Su2Space::new(6, 2, 2);
    let (x, cp0) = two_site_frame_su2::<f64>(space, 2, 0xB3_5101);
    assert_theta_cw_live(&x, "tms_b3su2 enriched frame");
    let mut rng = seeded(0xB3_5102);
    let cps = unit_blocks_su2(&cp0, 2, &mut rng);
    let w = [0.7, 0.3];
    let r = 3usize;
    let alpha = 0.25;

    for forward in [true, false] {
        let tag = if forward { "fwd" } else { "back" };

        let mut nrng = seeded(0xB3_5103 + forward as u64);
        let mut placed: Option<Su2Sector> = None;
        let mut absent = 0usize;
        let nd = qntt::su2::qn::Su2SectorGrid::from_fn(&space, |m| {
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
        let mut spec: Vec<(Su2Sector, Vec<f64>, Mat<f64>)> = Vec::new();
        for &m in cp0.mid_qn.iter() {
            let blocks: Vec<Mat<f64>> = cps.iter().filter_map(|c| b_theta(c, m)).collect();
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
            let cw = qntt::su2::centerweight(m.two_s, space.two_s);
            spec.push((m, evals.into_iter().map(|l| cw * l).collect(), u));
        }
        let mut pooled_vals: Vec<(f64, Su2Sector)> = Vec::new();
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
                "{tag}: sector {m:?} kept rank {} but the top-{r} ENRICHED pooled cw·λ place \
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
            let cw = qntt::su2::centerweight(m.two_s, space.two_s);
            let mut q = Mat::<f64>::zeros(g, rr);
            for c in 0..rr {
                for i in 0..g {
                    *q.get_mut(i, c) = u[(i, g - 1 - c)];
                }
            }
            for i in 0..cps.len() {
                let bt = b_theta(&cps[i], *m).unwrap();
                want_kept += w[i] * cw * kept_weight(q.as_ref(), bt.as_ref(), forward);
            }
        }
        let kept: f64 = (0..cps.len())
            .map(|i| w[i] * cs[i].norm_center().powi(2))
            .sum();
        assert!(
            (kept - want_kept).abs() <= 1e-10,
            "{tag}: kept weight {kept:.15e} vs the enriched-eigenspace projection \
             {want_kept:.15e} — the pooled ρ_θ the kernel chopped is not `Σwᵢρ_θᵢ + α_eff·nd`"
        );
        for i in 0..cps.len() {
            let want = 1.0 - cs[i].norm_center().powi(2);
            assert!(
                (infos[i].discarded_weight - want).abs() <= 1e-12,
                "{tag}: state {i} discarded {:.15e} vs 1 − ‖cᵢ‖²_center = {want:.15e}",
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
        let kept0: f64 = (0..cps.len())
            .map(|i| w[i] * cs0[i].norm_center().powi(2))
            .sum();
        println!("tms_b3su2 {tag}: kept α = {alpha} → {kept:.12e}, α = 0 → {kept0:.12e}");
        assert!(
            (kept - kept0).abs() > 1e-6,
            "{tag}: VACUOUS — enriching ρ_θ with α = {alpha} kept exactly what α = 0 keeps \
             ({kept:.15e} vs {kept0:.15e}); the noise term is dead or the fixture is degenerate"
        );
        assert!(
            kept <= kept0 + 1e-12,
            "{tag}: the enriched basis keeps MORE state weight ({kept:.15e}) than the Ky Fan \
             optimum of Σwᵢρ_θᵢ ({kept0:.15e})"
        );
    }
}

#[test]
fn tms_b9su2_homing_binds_weights_and_centres_to_labels_across_a_crossing() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xB9_7000, 3, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let n = 3usize;
    let w = vec![0.5, 0.3, 0.2];
    let homed = StateAverageOptions {
        mals: opts_exact(),
        weights: Some(w.clone()),
        homing: true,
        ..Default::default()
    };

    let mut crossings = 0usize;
    for seed in 0xB9_7100u64..0xB9_7105 {
        let mut rng = seeded(seed);
        let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &homed);

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| hist.lambdas[a].partial_cmp(&hist.lambdas[b]).unwrap());
        let sorted = order.iter().enumerate().all(|(p, &i)| p == i);
        if !sorted {
            crossings += 1;
        }
        println!(
            "tms_b9su2 crossing: seed {seed:#x} λ = {:?}, energy-rank of each label = {order:?} \
             ({})",
            hist.lambdas,
            if sorted { "sorted" } else { "PERMUTED" }
        );

        let mut got = hist.lambdas.clone();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for i in 0..n {
            assert!(
                (got[i] - levels[i]).abs() <= 1e-8,
                "seed {seed:#x}: sorted λ {:?} is not the dense singlet spectrum {:?}",
                got,
                &levels[..n]
            );
        }

        let sw = hist.sweeps.last().expect("at least one sweep");
        let want: f64 = (0..n).map(|i| w[i] * sw.energies[i]).sum();
        assert!(
            (sw.energy - want).abs() <= 1e-12 * want.abs().max(1.0),
            "seed {seed:#x}: sweep energy {} vs Σᵢwᵢλᵢ over the label order {want}",
            sw.energy
        );
        let by_energy: f64 = {
            let mut e = sw.energies.clone();
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (0..n).map(|i| w[i] * e[i]).sum()
        };
        if !sorted {
            assert!(
                (sw.energy - by_energy).abs() > 1e-8,
                "seed {seed:#x}: VACUOUS — on a permuted run, weighting by LABEL and by ENERGY \
                 give the same number ({} vs {by_energy})",
                sw.energy
            );
        }

        for i in 0..n {
            let rq = qntt::su2::hamiltonian::rayleigh_quotient(&h, &ys[i]);
            assert!(
                (rq - hist.lambdas[i]).abs() <= 1e-8,
                "seed {seed:#x}: state {i}'s Rayleigh quotient {rq} is not its own label's λ {} \
                 — labels and centres are out of step",
                hist.lambdas[i]
            );
        }
    }
    println!("tms_b9su2 crossing: {crossings}/5 runs ended on a permuted label order");
    assert!(
        crossings >= 1,
        "VACUOUS — no run crossed at the (2,4) start, so homing was the identity throughout and \
         this arm discriminates nothing (measured on this fixture: 3/5 cross)"
    );
}

#[test]
fn tms_b4su2_driver_forwards_the_reorthonormalize_flag() {
    let d = 6usize;
    let space = Su2Space::new(d, 2, 0);
    let fx = gapped_tv(0xB4_9500, 3, 0.05);
    let h = fx.su2(space);
    let levels = fx.singlet_levels();
    let n = 3usize;
    let w = vec![0.5, 0.3, 0.2];
    let mals = MalsOptions {
        eps: 1e-8,
        bond_dims: vec![3],
        noise: vec![0.0],
        max_iter: 3,
        ..Default::default()
    };
    let mut rng = seeded(0xB4_9501);
    let x0: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);

    let mut devs = Vec::new();
    for reorth in [true, false] {
        let o = StateAverageOptions {
            mals: mals.clone(),
            weights: Some(w.clone()),
            reorthonormalize: reorth,
            ..Default::default()
        };
        let (hist, ys) = state_average_mals_with_history(&h, &x0, n, &o);
        let gd = gram_dev_su2(&ys);
        println!("tms_b4su2 driver: reorthonormalize = {reorth} → Gram deviation {gd:.3e}");
        devs.push(gd);
        for (it, sw) in hist.sweeps.iter().enumerate() {
            assert_weighted_ky_fan(sw, &w, &levels, &format!("reorth = {reorth}: sweep {it}"));
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
         re-orthonormalized run's {:.3e}",
        devs[1],
        devs[0]
    );
}
