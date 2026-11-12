// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::hamiltonian::SparseHamiltonian;
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fcidump, read_meta};
use qntt::pointgroup::Irrep;
use qntt::qn::QnSpace;
use qntt::solvers::joint::cas::{CasPartition, cas_reference_state};
use qntt::solvers::mals::MalsOptions;
use qntt::su2::core::Family;
use qntt::su2::expand::expand;
use qntt::su2::joint::cas::{su2_cas_rank_grid, su2_cas_reference_state};
use qntt::su2::qn::{Su2Sector, Su2Space};
use qntt::su2::riemannian::tangent_bases;
use qntt::su2::solvers::mals::mals;
use qntt::su2::vector::{Su2TtVector, tt_csf, tt_randn};
use qntt::su2::weingarten::{tangent_basis, tangent_dim};

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::su2_cas::util::*;

#[test]
fn tsc_a1_grid_legality_and_totals() {
    for (fx, want_totals) in [
        (lih_cas22(), vec![1, 1, 3, 1, 1, 1, 1]),
        (h2o_cas86(), vec![1, 1, 3, 10, 17, 10, 3, 1]),
    ] {
        let (space, part, d) = (fx.space, fx.part, fx.d());
        let grid = &fx.grid;
        assert_eq!(grid.len(), d + 1, "{}: grid length", fx.label);
        assert_eq!(
            grid_totals(grid),
            want_totals,
            "{}: per-bond totals (measured)",
            fx.label
        );

        assert_eq!(grid[0].total(), 1, "{}: bond 0 total", fx.label);
        assert_eq!(
            grid[0][Su2Sector::new(0, 0)],
            1,
            "{}: bond 0 must pin (0,0)",
            fx.label
        );
        let last = Su2Sector::with_irrep(space.n, space.two_s, space.target);
        assert_eq!(grid[d].total(), 1, "{}: bond d total", fx.label);
        assert_eq!(grid[d][last], 1, "{}: bond d must pin {last:?}", fx.label);

        for b in 0..=d {
            let pinned = b <= part.n_core || b >= part.n_core + part.n_active;
            let total = grid[b].total();
            if pinned {
                assert_eq!(
                    total, 1,
                    "{}: pinned bond {b} carries total {total}",
                    fx.label
                );
                if b <= part.n_core {
                    assert_eq!(
                        grid[b][Su2Sector::new(2 * b, 0)],
                        1,
                        "{}: core bond {b} must pin (2b, 0) = ({}, 0)",
                        fx.label,
                        2 * b
                    );
                }
            } else {
                assert!(total >= 1, "{}: active bond {b} is empty", fx.label);
            }
        }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA1 ^ (d as u64));
        let t: Su2TtVector<f64> = tt_randn(space, grid, &mut rng);
        assert!(t.check(), "{}: generic train on the pinned grid", fx.label);
        let diff = grid_diff(&space, &t.ranks, grid);
        assert!(
            diff.is_empty(),
            "{}: tt_randn did not realize the declared grid — {}",
            fx.label,
            fmt_grid_diff(&diff)
        );
    }
}

#[test]
fn tsc_a2_keystone_tangent_dim_is_the_weyl_paldus_count() {
    for (fx, want_su2, want_abelian) in [(lih_cas22(), 3usize, 4usize), (h2o_cas86(), 105, 225)] {
        let (space, part) = (fx.space, fx.part);
        let n_act = space.n - 2 * part.n_core;

        let predicted = weyl_paldus(part.n_active, n_act, space.two_s);
        assert_eq!(
            predicted, want_su2,
            "{}: Weyl-Paldus f({}, {n_act}, {}) disagrees with the pinned literal",
            fx.label, part.n_active, space.two_s
        );
        assert_eq!(
            weyl_paldus_sum(part.n_active, n_act),
            want_abelian,
            "{}: Σ_S f must reproduce the abelian pinned tangent_dim — the spin decomposition is \
             what makes the su2 manifold the spin-S slice of the abelian one",
            fx.label
        );

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2);
        let generic: Su2TtVector<f64> = tt_randn(space, &fx.grid, &mut rng);
        let (gl, gr) = tangent_bases(&generic);
        assert_eq!(
            tangent_dim(&gl),
            predicted,
            "{}: tangent_dim on the DECLARED grid",
            fx.label
        );
        assert_eq!(
            tangent_basis(&gl, &gr, true).len() + 1,
            predicted,
            "{}: the basis enumerator must agree with the raw−gauge count on the declared grid",
            fx.label
        );

        let (bl, br) = fx.frames();
        assert_eq!(
            tangent_dim(&bl),
            predicted,
            "{}: tangent_dim at the converged pinned state (‖grad‖ = {:.3e})",
            fx.label,
            fx.tt_gradnorm()
        );
        assert_eq!(
            tangent_basis(&bl, &br, true).len() + 1,
            predicted,
            "{}: the basis enumerator must agree at the converged state",
            fx.label
        );
    }
}

#[test]
fn tsc_a3_normal_classification_is_exact() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let ctx = fx
            .joint_ctx(false, 1e-8)
            .unwrap_or_else(|e| panic!("{}: joint context refused: {e:?}", fx.label));
        let (u, tau) = (ctx.orbital_directions(), ctx.orbital_projections());
        let iso = ctx.iso_defect();

        let (mut n_cross, mut n_aa, mut n_zero) = (0usize, 0usize, 0usize);
        for (a, &(p, q)) in ctx.pairs().iter().enumerate() {
            let un = qntt::su2::ops::dot(&u[a], &u[a]).sqrt();
            let class = fx.pair_class(p, q);
            if class == "CC" || class == "VV" {
                assert!(
                    un <= 1e-14,
                    "{}: {class} pair ({p},{q}) must be the exact-zero train, got ‖u_a‖ = {un:.3e} \
                     (iso_defect = {iso:.3e})",
                    fx.label
                );
                n_zero += 1;
                continue;
            }
            assert!(
                un > 1e-3,
                "{}: {class} pair ({p},{q}) is vacuous — ‖u_a‖ = {un:.3e}, so its classification \
                 says nothing (iso_defect = {iso:.3e})",
                fx.label
            );
            let ratio = tau[a].norm() / un;
            match class {
                "AA" => {
                    assert!(
                        (ratio - 1.0).abs() <= 1e-10,
                        "{}: AA pair ({p},{q}) must be entirely TANGENTIAL at full active rank, \
                         got ‖P(u_a)‖/‖u_a‖ = {ratio:.6e} (‖u_a‖ = {un:.3e}, iso_defect = \
                         {iso:.3e})",
                        fx.label
                    );
                    n_aa += 1;
                }
                _ => {
                    assert!(
                        ratio <= 1e-12,
                        "{}: {class} pair ({p},{q}) must be purely NORMAL, got \
                         ‖P(u_a)‖/‖u_a‖ = {ratio:.6e} (‖u_a‖ = {un:.3e}, iso_defect = {iso:.3e})",
                        fx.label
                    );
                    n_cross += 1;
                }
            }
        }
        assert_eq!(
            n_cross,
            expected_n_eff(&fx.part),
            "{}: the cross-window pairs classified must BE the non-redundant set",
            fx.label
        );
        assert!(
            n_aa >= 1,
            "{}: no active-active pair was classified — the tangential half is vacuous",
            fx.label
        );
        let _ = n_zero;

        let s = ctx.orbital_gram();
        let n_orb = ctx.pairs().len();
        let is_mask = |a: usize| {
            let (p, q) = ctx.pairs()[a];
            fx.window(p) != fx.window(q)
        };
        let cross = (0..n_orb)
            .flat_map(|a| (0..n_orb).map(move |b| (a, b)))
            .filter(|&(a, b)| is_mask(a) != is_mask(b))
            .map(|(a, b)| s[a * n_orb + b].abs())
            .fold(0.0f64, f64::max);
        assert!(
            cross <= 1e-12,
            "{}: max |S_ab| across the mask/non-mask split = {cross:.3e} — the classes must not \
             mix (iso_defect = {iso:.3e})",
            fx.label
        );
    }
}

#[test]
fn tsc_a4_neff_pins_and_floor_sweep() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let want = expected_n_eff(&fx.part);
        let g_kept = fx.orb_grad_norm(&fx.cas_mask_pairs());
        let g_orb = fx.orb_grad_norm(&fx.full_pairs());
        let z = fx.z_norm();

        assert!(
            z > 1e-2,
            "{}: ‖z‖ = {z:.3e} collapsed — the pinned point has become an exact eigenstate and \
             every B block below would be vacuous",
            fx.label
        );

        for floor in [1e-10, 1e-8, 1e-6] {
            for mask in [true, false] {
                let ctx = fx.joint_ctx(mask, floor).unwrap_or_else(|e| {
                    panic!(
                        "{}: context refused at floor {floor:.0e}, mask = {mask}: {e:?} \
                         (‖g_kept‖ = {g_kept:.3e}, ‖g_orb‖ = {g_orb:.3e})",
                        fx.label
                    )
                });
                assert_eq!(
                    ctx.n_eff(),
                    want,
                    "{}: n_eff at floor {floor:.0e}, mask = {mask} — analytic count is \
                     N_A·N_I + N_I·N_E + N_A·N_E = {want} (of {} pairs; ‖g_orb‖ = {g_orb:.3e}, \
                     iso = {:.3e}, sigma_min = {:.3e})",
                    fx.label,
                    ctx.pairs().len(),
                    ctx.iso_defect(),
                    ctx.sigma_min()
                );
                if mask {
                    assert_eq!(
                        ctx.pairs().len(),
                        want,
                        "{}: cas_pairs must keep exactly the non-redundant set",
                        fx.label
                    );
                }
                let n_dropped = ctx.pairs().len() - ctx.n_eff();
                let max_dropped = ctx
                    .orb_spectrum()
                    .iter()
                    .take(n_dropped)
                    .fold(0.0f64, |m, v| m.max(v.abs()));
                let min_kept = ctx.orb_kept().first().copied().unwrap_or(f64::NAN);
                assert!(
                    min_kept > 1e-4,
                    "{}: smallest kept μ = {min_kept:.3e} at floor {floor:.0e} — the quotient is \
                     keeping a near-null direction",
                    fx.label
                );
                assert!(
                    max_dropped < 1e-12,
                    "{}: largest dropped |μ| = {max_dropped:.3e} at floor {floor:.0e} — the \
                     quotient discarded a genuine direction (smallest kept = {min_kept:.3e})",
                    fx.label
                );
            }
        }
    }

    let fx = lih_cas22();
    let full = full_ranks(&fx.space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xF011);
    let mut y: Su2TtVector<f64> = tt_randn(fx.space, &full, &mut rng);
    qntt::su2::riemannian::normalize_mut(&mut y);
    let (lam, y) = qntt::su2::solvers::als::als(
        &fx.ham,
        &y,
        &qntt::su2::solvers::als::AlsOptions {
            eps: 1e-12,
            max_iter: 200,
            ..Default::default()
        },
    );
    let (mut y, _info) = qntt::su2::hess::newton_polish(
        &fx.ham,
        &y,
        &qntt::solvers::hess::NewtonPolishOpts {
            gradtol: 1e-11,
            ..Default::default()
        },
    );
    qntt::su2::riemannian::normalize_mut(&mut y);
    let opts = qntt::solvers::joint::JointOpts {
        tt_grad_tol: TT_GRAD_TOL,
        orb_floor: 1e-8,
        pairs: None,
        z_route: qntt::solvers::joint::ZRoute::Explicit,
        b_general: false,
    };
    let ctx = qntt::su2::joint::su2_joint_context(&fx.ham, &y, &opts)
        .unwrap_or_else(|e| panic!("unpinned full-rank context refused: {e:?} (λ = {lam:.9})"));
    assert_eq!(
        ctx.n_eff(),
        0,
        "at unpinned FULL TT rank every orbital direction is already tangential, so the quotient \
         must keep nothing — got n_eff = {} of {} (max |μ| = {:.3e})",
        ctx.n_eff(),
        ctx.pairs().len(),
        ctx.orb_spectrum()
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()))
    );
}

#[test]
fn tsc_a4b_truncated_rank_divergence() {
    let fx = h2o_cas86_m2();
    assert_eq!(fx.m_active, 2, "this arm is the truncated rung");
    let want = expected_n_eff(&fx.part);

    let c_mask = fx
        .joint_ctx(true, 1e-8)
        .unwrap_or_else(|e| panic!("{}: masked context refused: {e:?}", fx.label));
    let c_full = fx
        .joint_ctx(false, 1e-8)
        .unwrap_or_else(|e| panic!("{}: full-triangle context refused: {e:?}", fx.label));

    assert_eq!(
        c_mask.n_eff(),
        want,
        "{}: n_eff under the mask must be rank-INDEPENDENT (τ_a ≡ 0 for non-redundant pairs at \
         any active rank) — got {} of {}",
        fx.label,
        c_mask.n_eff(),
        c_mask.pairs().len()
    );
    assert!(
        c_full.n_eff() > want,
        "{}: at m_active = 2 the full triangle must keep MORE than the mask's {want}; got {} of \
         {} (iso = {:.3e}, sigma_min = {:.3e})",
        fx.label,
        c_full.n_eff(),
        c_full.pairs().len(),
        c_full.iso_defect(),
        c_full.sigma_min()
    );
    assert!(
        c_full.n_eff() <= c_full.pairs().len(),
        "{}: n_eff cannot exceed the pair count",
        fx.label
    );
    let min_kept = c_full.orb_kept().first().copied().unwrap_or(f64::NAN);
    assert!(
        min_kept > 1e-10,
        "{}: the extra kept directions are floor noise — smallest kept μ = {min_kept:.3e}",
        fx.label
    );
}

#[test]
fn tsc_a5_reference_state_contract() {
    for fx in [lih_cas22(), h2o_cas86(), h2o_cas86_m2(), lih_doublet()] {
        assert!(fx.x.check(), "{}: state is illegal", fx.label);
        assert!(
            grid_le(&fx.space, &fx.x.ranks, &fx.grid),
            "{}: the state left the declared grid from ABOVE — round_ranks_mut and the HOSVD \
             retraction are both '≤ target' operations, so only shrinkage is reachable ({})",
            fx.label,
            fmt_grid_diff(&grid_diff(&fx.space, &fx.x.ranks, &fx.grid))
        );
        let diff = grid_diff(&fx.space, &fx.x.ranks, &fx.grid);
        assert!(
            diff.is_empty(),
            "{}: the converged state is rank-deficient against its declared grid — {}",
            fx.label,
            fmt_grid_diff(&diff)
        );
        let g = fx.tt_gradnorm();
        assert!(
            g < 1e-9,
            "{}: reference state is not TT-critical, ‖P((H−λ)x)‖ = {g:.3e} (λ = {:.9}, \
             m_active = {})",
            fx.label,
            fx.lambda,
            fx.m_active
        );
    }

    let fx = lih_cas22();
    let (a, _xa) = su2_cas_reference_state(&fx.ham, &fx.space, &fx.part, 0, 0xDE7E);
    let (b, _xb) = su2_cas_reference_state(&fx.ham, &fx.space, &fx.part, 0, 0xDE7E);
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "same seed must give a bit-identical λ: {a:.17} vs {b:.17}"
    );
}

#[test]
fn tsc_a6_composition_randn_round_is_grid_exact() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA6);

        let r: Su2TtVector<f64> = tt_randn(fx.space, &fx.grid, &mut rng);
        let regrown = regrown_zero_sectors(&fx.space, &r.ranks, &fx.grid);
        assert!(
            regrown.is_empty(),
            "{}: tt_randn raised a pinned-ZERO sector — {} entries, first bond {} sector \
             (n={},2s={})",
            fx.label,
            regrown.len(),
            regrown[0].0,
            regrown[0].1.n,
            regrown[0].1.two_s
        );

        let s: Su2TtVector<f64> = tt_randn(fx.space, &fx.grid, &mut rng);
        let mut sum = &r + &s;
        assert!(
            grid_totals(&sum.ranks).iter().sum::<usize>()
                >= grid_totals(&fx.grid).iter().sum::<usize>(),
            "{}: the sum should not have SHRUNK below the grid before rounding",
            fx.label
        );
        sum.round_ranks_mut(&fx.grid);
        let diff = grid_diff(&fx.space, &sum.ranks, &fx.grid);
        assert!(
            diff.is_empty(),
            "{}: round_ranks_mut did not return the declared grid — {}",
            fx.label,
            fmt_grid_diff(&diff)
        );
        assert!(sum.check(), "{}: rounded sum is illegal", fx.label);

        let n_docc = fx.space.n / 2;
        let codes: Vec<Family> = (0..fx.d())
            .map(|k| {
                if k < n_docc {
                    Family::Occ
                } else {
                    Family::Empty
                }
            })
            .collect();
        let mut csf: Su2TtVector<f64> = tt_csf(&codes);
        qntt::su2::riemannian::normalize_mut(&mut csf);
        let ov = qntt::su2::ops::dot(&fx.x, &csf).abs();
        let e_corr = fx.lambda - qntt::su2::hamiltonian::rayleigh_quotient(&fx.ham, &csf);

        assert!(
            1.0 - ov > 1e-6,
            "{}: the converged reference state is still the bare CSF seed — |⟨x, csf⟩| = {ov:.12}, \
             1 − |⟨x, csf⟩| = {:.3e}, E_corr = {e_corr:.6e}. The enrichment did nothing, so every \
             arm reading this fixture is testing a determinant, not a CAS state",
            fx.label,
            1.0 - ov
        );
        assert!(
            e_corr < -1e-5,
            "{}: the pinned optimum does not improve on its own CSF seed — E_corr = {e_corr:.6e} \
             (λ = {:.9}, |⟨x, csf⟩| = {ov:.12})",
            fx.label,
            fx.lambda
        );
    }
}

#[test]
fn tsc_a7_mals_regrowth_antioracle() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let (lam_mals, x_mals) = mals(
            &fx.ham,
            &fx.x,
            &MalsOptions {
                eps: 1e-10,
                max_iter: 3,
                ..MalsOptions::default()
            },
        );
        let (bl_p, _) = fx.frames();
        let (bl_m, _) = tangent_bases(&x_mals);
        let (dim_pinned, dim_mals) = (tangent_dim(&bl_p), tangent_dim(&bl_m));

        let regrown = regrown_zero_sectors(&fx.space, &x_mals.ranks, &fx.grid);
        assert!(
            !regrown.is_empty(),
            "{}: MALS regrew NOTHING — the anti-oracle is vacuous, so it can no longer detect a \
             MALS wired into the pinned recipe (totals {:?} vs grid {:?})",
            fx.label,
            grid_totals(&x_mals.ranks),
            grid_totals(&fx.grid)
        );
        let witness = regrown
            .iter()
            .find(|(b, s, _)| *b == 1 && *s == Su2Sector::new(0, 0));
        assert!(
            witness.is_some(),
            "{}: the measured witness (bond 1, sector (0,0)) did not regrow; regrown set was {:?}",
            fx.label,
            regrown
                .iter()
                .take(6)
                .map(|(b, s, r)| format!("bond {b} (n={},2s={}) → {r}", s.n, s.two_s))
                .collect::<Vec<_>>()
        );
        assert!(
            dim_mals > dim_pinned,
            "{}: MALS must ENLARGE the tangent space, got {dim_mals} vs pinned {dim_pinned}",
            fx.label
        );
        assert!(
            fx.lambda - lam_mals > 1e-6,
            "{}: MALS must reach strictly below the pinned optimum (it leaves the CAS manifold), \
             margin was {:.3e} (λ_pinned = {:.9}, λ_mals = {lam_mals:.9})",
            fx.label,
            fx.lambda - lam_mals,
            fx.lambda
        );
    }
}

#[test]
fn tsc_a8_truncation_ladder_and_btt_teeth() {
    let full = h2o_cas86();
    let trunc = h2o_cas86_m2();

    let (tf, tt) = (grid_totals(&full.grid), grid_totals(&trunc.grid));
    assert_eq!(tf.len(), tt.len());
    assert!(
        tt.iter().zip(&tf).all(|(a, b)| a <= b) && tt.iter().sum::<usize>() < tf.iter().sum(),
        "m_active = 2 must give a strictly smaller grid: {tt:?} vs full {tf:?}"
    );
    assert!(
        grid_le(&trunc.space, &trunc.grid, &full.grid),
        "the truncated grid must sit inside the full one sector by sector"
    );
    assert!(trunc.x.check(), "the truncated state is illegal");
    assert!(
        tangent_dim(&trunc.frames().0) < tangent_dim(&full.frames().0),
        "the truncated tangent space must be strictly smaller"
    );

    let max_b = |fx: &Su2CasFixture| -> f64 {
        let wop = qntt::su2::hess::weingarten_operator(&fx.ham, &fx.x);
        let (bl, br) = wop.bases();
        tangent_basis(bl, br, true)
            .iter()
            .map(|xi| wop.b_apply(xi).norm())
            .fold(0.0f64, f64::max)
    };

    for (fx, floor) in [(h2o_cas86(), 1e-11), (lih_cas22(), 1e-14)] {
        let b = max_b(fx);
        assert!(
            b < floor,
            "{}: at FULL active rank the pinned manifold is a LINEAR slice, so B_tt must vanish as \
             an operator — got max‖B_tt ξ‖ = {b:.3e} against ‖z‖ = {:.3e}",
            fx.label,
            fx.z_norm()
        );
    }

    let b_trunc = max_b(trunc);
    assert!(
        b_trunc > 1e-2,
        "{}: the truncated rung must be genuinely CURVED or every curvature arm built on it is \
         vacuous — got max‖B_tt ξ‖ = {b_trunc:.3e} (‖z‖ = {:.3e})",
        trunc.label,
        trunc.z_norm()
    );
}

#[test]
fn tsc_a9_cross_layer_energy_and_state() {
    for (fx, seed) in [(lih_cas22(), 0xCA22u64), (h2o_cas86(), 0xCA86)] {
        let d = fx.d();
        let nup = (fx.ints.nelec + fx.ints.ms2.unsigned_abs() as usize) / 2;
        let ab_space = QnSpace::new(d, nup, fx.ints.nelec - nup);
        let ab_h = SparseHamiltonian::new(ab_space, &fx.ints.t, &fx.ints.v, None, f64::EPSILON);
        let (ab_lambda, ab_x) = cas_reference_state(&ab_h, &ab_space, &fx.part, 0, seed);

        let d_lam = (fx.lambda - ab_lambda).abs();
        assert!(
            d_lam <= 1e-10 * fx.lambda.abs().max(1.0),
            "{}: the singlet CAS ground state must be layer-independent — su2 λ = {:.12}, abelian \
             λ = {ab_lambda:.12}, Δ = {d_lam:.3e}",
            fx.label,
            fx.lambda
        );

        let ex = expand(&fx.x, 0);
        let (v_su2, v_ab) = (ex.to_dense(), ab_x.to_dense());
        assert_eq!(v_su2.len(), v_ab.len(), "{}: dense length", fx.label);
        let n_su2 = v_su2.iter().map(|a| a * a).sum::<f64>().sqrt();
        let n_ab = v_ab.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(
            (n_su2 - 1.0).abs() <= 1e-12 && (n_ab - 1.0).abs() <= 1e-12,
            "{}: dense norms ‖expand(x_su2,0)‖ = {n_su2:.15}, ‖x_ab‖ = {n_ab:.15}",
            fx.label
        );
        let ov: f64 = v_su2
            .iter()
            .zip(&v_ab)
            .map(|(a, b)| a * b)
            .sum::<f64>()
            .abs()
            / (n_su2 * n_ab);
        assert!(
            (ov - 1.0).abs() <= 1e-10,
            "{}: |⟨expand(x_su2, 0), x_ab⟩| = {ov:.15} — the two layers converged to DIFFERENT \
             states even though their energies agree to {d_lam:.3e}",
            fx.label
        );

        if let Some(gs) = fx.casscf_gs {
            let dev = fx.lambda + fx.ints.e0 - gs;
            assert!(
                dev.abs() <= 1e-9,
                "{}: λ + e0 = {:.12} vs the meta's casscf_gs = {gs:.12}, Δ = {dev:.3e}",
                fx.label,
                fx.lambda + fx.ints.e0
            );
        }
    }
}

#[test]
fn tsc_a10_typed_grid_and_unreachable_target() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("h2o_c2v");
    let meta = read_meta(&dir.join("meta.txt")).expect("h2o_c2v meta");
    let dump = meta.keys.get("dump_file").expect("dump_file key").clone();
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let load = load_fcidump::<f64>(&dir.join(dump), None, &opts).expect("h2o_c2v dump");
    let (d, n) = (load.ints.norb, load.ints.nelec);
    assert!(
        load.pg.pg.order() > 1,
        "h2o_c2v must resolve to a nontrivial point group, got order {}",
        load.pg.pg.order()
    );

    let part = CasPartition {
        n_core: 1,
        n_active: 6,
        n_virt: 0,
    };
    let untyped = Su2Space::new(d, n, 0);
    let typed = Su2Space::with_point_group(d, n, 0, load.pg.pg, load.pg.target);
    let g_un = su2_cas_rank_grid(&untyped, &part, qntt::PerSector::NONE);
    let g_ty = su2_cas_rank_grid(&typed, &part, qntt::PerSector::NONE);

    assert_eq!(g_ty[0].total(), 1, "typed bond 0 total");
    assert_eq!(g_ty[0][Su2Sector::new(0, 0)], 1, "typed bond 0 sector");
    assert_eq!(g_ty[d].total(), 1, "typed bond d total");
    assert_eq!(
        g_ty[d][Su2Sector::with_irrep(n, 0, load.pg.target)],
        1,
        "typed bond d must pin the target irrep {:?}",
        load.pg.target
    );

    assert!(
        (0..=d).all(|b| g_ty[b].total() <= g_un[b].total()),
        "the typed grid must be a refinement of the untyped one: typed {:?} vs untyped {:?}",
        grid_totals(&g_ty),
        grid_totals(&g_un)
    );
    assert!(
        grid_totals(&g_ty).iter().sum::<usize>() < grid_totals(&g_un).iter().sum::<usize>(),
        "the typed grid is not actually constrained — the arm would be vacuous (typed {:?})",
        grid_totals(&g_ty)
    );

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC2FE);
    let t: Su2TtVector<f64> = tt_randn(typed, &g_ty, &mut rng);
    assert!(t.check(), "typed pinned train is illegal");
    let diff = grid_diff(&typed, &t.ranks, &g_ty);
    assert!(
        diff.is_empty(),
        "tt_randn did not realize the typed grid — {}",
        fmt_grid_diff(&diff)
    );
    let (tl, tr) = tangent_bases(&t);
    let td = tangent_dim(&tl);
    assert_eq!(
        tangent_basis(&tl, &tr, true).len() + 1,
        td,
        "the basis enumerator must agree with raw−gauge on the typed grid"
    );
    assert!(
        td > 0 && td < weyl_paldus(part.n_active, n - 2 * part.n_core, 0),
        "the typed tangent_dim ({td}) must be a proper, nonempty subspace of the untyped singlet \
         CAS-CI space ({})",
        weyl_paldus(part.n_active, n - 2 * part.n_core, 0)
    );
}

#[test]
#[should_panic(expected = "unreachable from the active window")]
fn tsc_a10b_unreachable_typed_target_is_refused() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join("h2o_c2v");
    let meta = read_meta(&dir.join("meta.txt")).expect("h2o_c2v meta");
    let dump = meta.keys.get("dump_file").expect("dump_file key").clone();
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let load = load_fcidump::<f64>(&dir.join(dump), None, &opts).expect("h2o_c2v dump");
    let (d, n) = (load.ints.norb, load.ints.nelec);
    let part = CasPartition {
        n_core: 4,
        n_active: 1,
        n_virt: d - 5,
    };
    let unreachable = (1..load.pg.pg.order())
        .map(Irrep::new)
        .find(|&g| load.pg.pg.irrep_of(part.n_core) != g)
        .expect("C2v has a non-identity irrep");
    let space = Su2Space::with_point_group(d, n, 0, load.pg.pg, unreachable);
    let _ = su2_cas_rank_grid(&space, &part, qntt::PerSector::NONE);
}

#[test]
fn tsc_a11_degenerate_partition_reproduces_max_ranks() {
    for fx in [lih_cas22(), h2o_cas86()] {
        let part = CasPartition {
            n_core: 0,
            n_active: fx.d(),
            n_virt: 0,
        };
        let grid = su2_cas_rank_grid(&fx.space, &part, qntt::PerSector::NONE);
        let caps = full_ranks(&fx.space);
        let diff = grid_diff(&fx.space, &grid, &caps);
        assert!(
            diff.is_empty(),
            "{}: with no core and no virtual the pin must reproduce max_ranks exactly — {}",
            fx.label,
            fmt_grid_diff(&diff)
        );
        assert!(
            grid_totals(&grid).iter().sum::<usize>() > grid_totals(&fx.grid).iter().sum::<usize>(),
            "{}: the degenerate grid {:?} is not larger than the CAS grid {:?}",
            fx.label,
            grid_totals(&grid),
            grid_totals(&fx.grid)
        );
    }
}

#[test]
fn tsc_a12_doublet_seed_branch() {
    let fx = lih_doublet();
    assert_eq!((fx.space.n, fx.space.two_s), (3, 1), "the doublet sector");

    let predicted = weyl_paldus(
        fx.part.n_active,
        fx.space.n - 2 * fx.part.n_core,
        fx.space.two_s,
    );
    assert_eq!(predicted, 2, "Weyl-Paldus f(2, 1, 2S=1)");
    assert_eq!(
        grid_totals(&fx.grid),
        vec![1, 1, 2, 1, 1, 1, 1],
        "doublet pinned totals"
    );
    assert_eq!(
        fx.grid[1][Su2Sector::new(2, 0)],
        1,
        "the core prefix must pin the SPIN-0 sector even in a doublet"
    );
    assert_eq!(
        fx.grid[fx.d()][Su2Sector::new(3, 1)],
        1,
        "the virtual suffix must pin (N, 2S) = (3, 1)"
    );

    let (bl, br) = fx.frames();
    assert_eq!(
        tangent_dim(&bl),
        predicted,
        "doublet tangent_dim (‖grad‖ = {:.3e}, λ = {:.9})",
        fx.tt_gradnorm(),
        fx.lambda
    );
    assert_eq!(
        tangent_basis(&bl, &br, true).len() + 1,
        predicted,
        "the basis enumerator must agree in the doublet sector"
    );
    assert!(fx.x.check(), "doublet state is illegal");
    assert!(
        grid_diff(&fx.space, &fx.x.ranks, &fx.grid).is_empty(),
        "the doublet state left its declared grid — {}",
        fmt_grid_diff(&grid_diff(&fx.space, &fx.x.ranks, &fx.grid))
    );
}
