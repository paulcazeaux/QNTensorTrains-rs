// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

#[test]
fn forward_density_matrix_matches_the_theta_reference_placement() {
    let (h, space) = h4_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5b_c0de);
    let x = randomized_state(space, 3, 5, &mut rng);
    let k = 1usize;
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = sketch_h_mul(&fl, &h, &xk);

    assert_non_vacuous(&h, k + 1, &half.col_qn);

    let production = noise_density_matrix(&h, &half, &xk, &xk1, true);
    let oracle = oracle_forward(&h, &half, &xk, &xk1);

    for &m in &space.bond_sectors(k + 1) {
        let (pd, od) = (&production[m], &oracle[m]);
        assert_eq!(
            pd.nrows(),
            od.nrows(),
            "ρ_noise row-shape mismatch at mid {m:?}"
        );
        assert_eq!(
            pd.ncols(),
            od.ncols(),
            "ρ_noise col-shape mismatch at mid {m:?}"
        );
        if pd.nrows() == 0 {
            continue;
        }
        let rel = block_rel_diff(pd.as_ref(), od.as_ref());
        assert!(
            rel <= 1e-9,
            "forward ρ_noise[{m:?}] disagrees with the reference-s θ-placement oracle \
             (rel={rel}) — check σ_sc=(m_c.two_s+1)/(s.two_s+1) and xk1.scale_horizontal(_,m_c)"
        );
    }
}

#[test]
fn back_density_matrix_matches_the_theta_reference_placement() {
    let (h, space) = h4_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5b_c0de + 1);
    let x = randomized_state(space, 3, 5, &mut rng);
    let d = space.d;
    let k = d - 2;
    let fr = right_to_left_sketching(&h, &x)
        .into_iter()
        .nth(k + 1)
        .unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = h_sketch_mul(&h, &xk1, &fr);

    assert_eq!(
        half.col_ranks, xk1.col_ranks,
        "fixture invariant broken: this test relies on the boundary col bond making \
         fr.col_ranks == tc.col_ranks trivially"
    );
    assert_non_vacuous(&h, k + 1, &half.row_qn);

    let production = noise_density_matrix(&h, &half, &xk, &xk1, false);
    let oracle = oracle_back(&h, &half, &xk, &xk1);

    for &m in &space.bond_sectors(k + 1) {
        let (pd, od) = (&production[m], &oracle[m]);
        assert_eq!(
            pd.nrows(),
            od.nrows(),
            "ρ_noise row-shape mismatch at mid {m:?}"
        );
        assert_eq!(
            pd.ncols(),
            od.ncols(),
            "ρ_noise col-shape mismatch at mid {m:?}"
        );
        if pd.nrows() == 0 {
            continue;
        }
        let rel = block_rel_diff(pd.as_ref(), od.as_ref());
        assert!(
            rel <= 1e-9,
            "back ρ_noise[{m:?}] disagrees with the reference-s θ-placement oracle (rel={rel}) \
             — check half.scale_horizontal(_,s)"
        );
    }
}

#[test]
fn forward_density_matrix_matches_the_theta_reference_placement_at_nontrivial_group() {
    let (h, space) = pg_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5bc0_de03);
    let x = randomized_state(space, 3, 5, &mut rng);
    let d = space.d;
    let k = d / 2 - 1;
    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = sketch_h_mul(&fl, &h, &xk);

    assert_non_vacuous(&h, k + 1, &half.col_qn);

    let production = noise_density_matrix(&h, &half, &xk, &xk1, true);
    let oracle = oracle_forward(&h, &half, &xk, &xk1);

    let mut checked = 0usize;
    for &m in &space.bond_sectors(k + 1) {
        let (pd, od) = (&production[m], &oracle[m]);
        assert_eq!(
            pd.nrows(),
            od.nrows(),
            "ρ_noise row-shape mismatch at mid {m:?}"
        );
        assert_eq!(
            pd.ncols(),
            od.ncols(),
            "ρ_noise col-shape mismatch at mid {m:?}"
        );
        if pd.nrows() == 0 {
            continue;
        }
        let rel = block_rel_diff(pd.as_ref(), od.as_ref());
        assert!(
            rel <= 1e-9,
            "forward ρ_noise[{m:?}] disagrees with the reference-s θ-placement oracle at \
             |G|>1 (rel={rel}) — check the pgram/state-Gram memo is keyed by the FULL \
             Su2Sector (irrep included), not just (n, two_s)"
        );
        checked += 1;
    }
    assert!(checked > 0, "vacuous: no noise_dm block was checked");
}

#[test]
fn back_density_matrix_matches_the_theta_reference_placement_at_nontrivial_group() {
    let (h, space) = pg_fixture();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5bc0_de03 + 1);
    let x = randomized_state(space, 3, 5, &mut rng);
    let d = space.d;
    let k = d - 2;
    let fr = right_to_left_sketching(&h, &x)
        .into_iter()
        .nth(k + 1)
        .unwrap();
    let cp = contract(x.core(k), x.core(k + 1));
    let (xk, xk1) = factor_qc(&cp);
    let half = h_sketch_mul(&h, &xk1, &fr);

    assert_eq!(
        half.col_ranks, xk1.col_ranks,
        "fixture invariant broken: this test relies on the boundary col bond making \
         fr.col_ranks == tc.col_ranks trivially"
    );
    assert_non_vacuous(&h, k + 1, &half.row_qn);

    let production = noise_density_matrix(&h, &half, &xk, &xk1, false);
    let oracle = oracle_back(&h, &half, &xk, &xk1);

    let mut checked = 0usize;
    for &m in &space.bond_sectors(k + 1) {
        let (pd, od) = (&production[m], &oracle[m]);
        assert_eq!(
            pd.nrows(),
            od.nrows(),
            "ρ_noise row-shape mismatch at mid {m:?}"
        );
        assert_eq!(
            pd.ncols(),
            od.ncols(),
            "ρ_noise col-shape mismatch at mid {m:?}"
        );
        if pd.nrows() == 0 {
            continue;
        }
        let rel = block_rel_diff(pd.as_ref(), od.as_ref());
        assert!(
            rel <= 1e-9,
            "back ρ_noise[{m:?}] disagrees with the reference-s θ-placement oracle at |G|>1 \
             (rel={rel}) — check the pgram/state-Gram memo is keyed by the FULL Su2Sector \
             (irrep included), not just (n, two_s)"
        );
        checked += 1;
    }
    assert!(checked > 0, "vacuous: no noise_dm block was checked");
}
