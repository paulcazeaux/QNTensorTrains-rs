// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::solvers::InnerProductVector;
use qntt::solvers::davidson::effective_diagonal_bruteforce;
use qntt::su2::frame::Su2Sketch;
use qntt::su2::hamiltonian::matvec::{h_sketch_mul, noise_density_matrix, sketch_h_mul};
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh_quotient};
use qntt::su2::qn::{Su2RankGrid, Su2Space, max_ranks};
use qntt::su2::twosite::{
    Su2ContractedCores, contract, factor_qc, factor_svd_noise, factor_svd_perturbative,
};
use qntt::su2::vector::Su2TtVector;
use qntt::{Irrep, Scalar};

use crate::spg4b::util::*;

#[test]
fn spg4b_x5_two_site_matvec_vs_cross_layer_dense_oracle() {
    let d = 2;
    let pg = c2v_table(d);
    let (n, two_s) = (2, 0);
    let target = Irrep::IDENTITY;
    assert!(pg.order() > 1, "fixture group must be nontrivial");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng = seeded(0xB5_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB5_0002);

    let fl = Su2Sketch::identity(space, 0, 1);
    let fr = Su2Sketch::identity(space, d, 1);
    let cp = contract(x.core(0), x.core(1));

    let y_cp = h.two_site_sketched_hamiltonian(&cp, &fl, &fr);
    let (yk, yk1) = factor_qc(&y_cp);
    let y = Su2TtVector::new(space, vec![yk, yk1]);
    let hx_norm: f64 = (0..y.space.d)
        .map(|k| {
            let c = y.core(k);
            (0..c.num_coeffs())
                .map(|i| c.coeff(i).abs_s() * c.coeff(i).abs_s())
                .sum::<f64>()
        })
        .sum::<f64>()
        .sqrt();
    assert!(hx_norm > 1e-8, "vacuous: two-site H·x is ~zero");

    let err = max_err_over_two_sz_pg(&x, &t, &v, &y, space);
    assert!(err <= 1e-10, "d=2 two-site matvec cross-symmetry err={err}");

    let mut ws = qntt::su2::hamiltonian::matvec::su2_twosite_workspace::<f64>();
    let y_ws1 = h.two_site_sketched_hamiltonian_ws(&mut ws, &cp, &fl, &fr);
    let y_ws2 = h.two_site_sketched_hamiltonian_ws(&mut ws, &cp, &fl, &fr);
    assert_su2contracted_exact_eq(
        &y_ws1,
        &y_ws2,
        "two_site_sketched_hamiltonian_ws repeat call",
    );
    assert_su2contracted_exact_eq(&y_cp, &y_ws1, "two_site_sketched_hamiltonian vs _ws");

    let d4 = 4;
    let pg4 = c2v_table(d4);
    let (n4, two_s4) = (2, 0);
    let target4 = Irrep::IDENTITY;
    let space4 = Su2Space::with_point_group(d4, n4, two_s4, pg4, target4);
    let mut rng4 = seeded(0xB5_0010);
    let (t4, v4) = c2v_coeffs(&pg4, d4, &mut rng4);
    let h4 = Su2SparseHamiltonian::new(space4, &t4, &v4, None, 1e-12);
    let mut x4 = build_pg_state::<f64>(d4, n4, two_s4, pg4, target4, 3, 0xB5_0011);

    let k = 1usize;
    x4.move_core(k, true);
    let lchain = left_to_right_chain(&h4, &x4);
    let rchain = right_to_left_chain(&h4, &x4);
    let fl4 = &lchain[k];
    let fr4 = &rchain[k + 2];
    let cp4 = contract(x4.core(k), x4.core(k + 1));
    let heff4 = h4.two_site_sketched_hamiltonian(&cp4, fl4, fr4);
    let local = InnerProductVector::dot(&cp4, &heff4).re_s();
    let global = su2_rayleigh_quotient(&h4, &x4);
    assert!(
        global.abs() > 1e-8,
        "d=4 variant vacuous: rayleigh_quotient ~0"
    );
    assert!(
        (local - global).abs() <= 1e-10 * global.abs().max(1.0),
        "local ⟨cp,H_eff cp⟩={local} vs global rayleigh_quotient(h,x)={global}"
    );
}

#[test]
fn spg4b_x6_noise_arms_inject_no_rank_at_cap_zero_sectors_and_stay_pure() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng_h = seeded(0xB6_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng_h);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB6_0002);
    let caps = max_ranks(&space);
    let lchain = left_to_right_chain(&h, &x);
    let rchain = right_to_left_chain(&h, &x);

    let mut rank_changed_somewhere = false;

    for &k in &[0usize, 2] {
        let fl = &lchain[k];
        let fr = &rchain[k + 2];
        let cp0 = contract(x.core(k), x.core(k + 1));

        let mut ranks_by_alpha: Vec<(f64, Su2RankGrid)> = Vec::new();
        for &alpha in &[0.0f64, 1e-2, 1.0] {
            let mut rng = seeded(0xB6_1000 + k as u64);
            let (xk, s, xk1, _info) =
                factor_svd_noise(&cp0, 0.0, qntt::Pooled(2), true, alpha, &mut rng);
            for m in space.bond_sectors(k + 1) {
                if caps[k + 1][m] == 0 {
                    assert_eq!(
                        s.row_ranks()[m],
                        0,
                        "k={k} alpha={alpha}: cap-0 sector {m:?} got nonzero noise rank"
                    );
                }
            }
            let mid_ranks = s.row_ranks();
            if alpha > 0.0 {
                let y = rebuild_two_site(&x, k, xk, &s, xk1);
                assert!(
                    y.check(),
                    "k={k} alpha={alpha}: factor_svd_noise failed check()"
                );
                assert_expand_purity_chain(&y, &pg);
            }
            ranks_by_alpha.push((alpha, mid_ranks));
        }
        let (base_alpha, base_ranks) = &ranks_by_alpha[0];
        assert_eq!(*base_alpha, 0.0);
        let (hi_alpha, hi_ranks) = &ranks_by_alpha[2];
        assert_eq!(*hi_alpha, 1.0);
        if base_ranks != hi_ranks {
            rank_changed_somewhere = true;
        }

        for &forward in &[true, false] {
            let (xk_qc, xk1_qc) = factor_qc(&cp0);
            let half = if forward {
                sketch_h_mul(fl, &h, &xk_qc)
            } else {
                h_sketch_mul(&h, &xk1_qc, fr)
            };
            let noise_dm = noise_density_matrix(&h, &half, &xk_qc, &xk1_qc, forward);
            let (xk, s, xk1, _info) =
                factor_svd_perturbative(&cp0, &noise_dm, 0.0, qntt::Pooled(2), true, 1.0, forward);
            for m in space.bond_sectors(k + 1) {
                if caps[k + 1][m] == 0 {
                    assert_eq!(
                        s.row_ranks()[m],
                        0,
                        "k={k} forward={forward}: cap-0 sector {m:?} got nonzero perturbative rank"
                    );
                }
            }
            let y = rebuild_two_site(&x, k, xk, &s, xk1);
            assert!(
                y.check(),
                "k={k} forward={forward}: factor_svd_perturbative failed check()"
            );
            assert_expand_purity_chain(&y, &pg);
        }
    }
    assert!(
        rank_changed_somewhere,
        "vacuous: alpha=1.0 never changed the kept-rank distribution vs alpha=0 at any bond"
    );
}

#[test]
fn spg4b_x7_noise_density_matrix_shape_and_hermiticity() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng_h = seeded(0xB7_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng_h);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB7_0002);
    let lchain = left_to_right_chain(&h, &x);
    let rchain = right_to_left_chain(&h, &x);

    let mut checked = 0usize;
    for &k in &[0usize, 2] {
        let fl = &lchain[k];
        let fr = &rchain[k + 2];
        let tc = contract(x.core(k), x.core(k + 1));
        let (xk, xk1) = factor_qc(&tc);
        for &forward in &[true, false] {
            let half = if forward {
                sketch_h_mul(fl, &h, &xk)
            } else {
                h_sketch_mul(&h, &xk1, fr)
            };
            let nd_grid = noise_density_matrix(&h, &half, &xk, &xk1, forward);
            for &m in &tc.mid_qn {
                let nd = &nd_grid[m];
                if nd.nrows() == 0 && nd.ncols() == 0 {
                    continue;
                }
                let expect_dim = tc
                    .block(m)
                    .map_or(0, |b| if forward { b.nrows() } else { b.ncols() });
                assert_eq!(
                    nd.nrows(),
                    nd.ncols(),
                    "k={k} forward={forward} m={m:?}: noise_dm not square"
                );
                assert_eq!(
                    nd.nrows(),
                    expect_dim,
                    "k={k} forward={forward} m={m:?}: noise_dm dim mismatch"
                );
                let g = nd.nrows();
                let mut max_herm = 0.0f64;
                for i in 0..g {
                    for j in 0..g {
                        let diff = nd[(i, j)] - nd[(j, i)].conj_s();
                        max_herm = max_herm.max(diff.abs_s());
                    }
                }
                assert!(
                    max_herm <= 1e-10,
                    "k={k} forward={forward} m={m:?}: noise_dm not Hermitian, max dev {max_herm}"
                );
                if g > 0 {
                    let (evals, _u) = dense_self_adjoint_eigen(nd.as_ref());
                    let lam_min = evals[0];
                    let lam_max = evals[g - 1].max(1.0);
                    assert!(
                        lam_min >= -1e-10 * lam_max,
                        "k={k} forward={forward} m={m:?}: noise_dm not PSD (lam_min={lam_min}, \
                         lam_max={lam_max})"
                    );
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "vacuous: no noise_dm block was checked");
}

#[test]
fn spg4b_x8_two_site_effective_diagonal_vs_bruteforce_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (3, 1);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: need a reachable nonidentity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let mut rng = seeded(0xB8_0001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    let h = Su2SparseHamiltonian::new(space, &t, &v, None, 1e-12);
    let mut x = build_pg_state::<f64>(d, n, two_s, pg, target, 3, 0xB8_0002);
    x.right_orthogonalize(true);

    let mut checked = 0usize;
    for k in 0..d - 1 {
        x.move_core(k, false);
        let lchain = left_to_right_chain(&h, &x);
        let rchain = right_to_left_chain(&h, &x);
        let fl = &lchain[k];
        let fr = &rchain[k + 2];
        let tc = contract(x.core(k), x.core(k + 1));
        let mut lop = |v: &Su2ContractedCores<f64>| h.two_site_sketched_hamiltonian(v, fl, fr);
        let d_diag = h.two_site_effective_diagonal(fl, fr, &tc);
        let o = effective_diagonal_bruteforce(&mut lop, &tc);
        let n_o = InnerProductVector::norm(&o);
        assert!(
            n_o > 1e-8,
            "x8: non-vacuous — the brute-force diagonal is ~0 at k={k}"
        );
        let mut diff = d_diag.clone();
        InnerProductVector::axpy(&mut diff, -1.0, &o);
        let dev = InnerProductVector::norm(&diff);
        assert!(
            dev <= 1e-10 * n_o.max(1.0),
            "x8: two_site_effective_diagonal disagrees with the brute-force oracle at k={k} \
             (dev={dev}, ||O||={n_o})"
        );
        checked += 1;
    }
    assert!(checked > 0, "vacuous: no bond was checked");
}
