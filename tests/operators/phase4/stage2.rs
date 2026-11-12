// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::phase4::*;

#[test]
fn t2_elem_dispatcher_crosscheck() {
    let d = 6usize;
    let space = QnSpace::new(d, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x2C);
    let x0: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let x0d = x0.to_dense();

    let cases: [(Orbital, Orbital, Orbital, Orbital, &str); 12] = [
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(1, Spin::Up),
            Orbital::new(2, Spin::Dn),
            Orbital::new(3, Spin::Dn),
            "generic distinct",
        ),
        (
            Orbital::new(1, Spin::Up),
            Orbital::new(3, Spin::Up),
            Orbital::new(1, Spin::Dn),
            Orbital::new(4, Spin::Dn),
            "i.site == k.site",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(2, Spin::Up),
            Orbital::new(3, Spin::Dn),
            Orbital::new(2, Spin::Dn),
            "j.site == l.site",
        ),
        (
            Orbital::new(1, Spin::Up),
            Orbital::new(3, Spin::Up),
            Orbital::new(2, Spin::Dn),
            Orbital::new(1, Spin::Dn),
            "i.site == l.site",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(2, Spin::Up),
            Orbital::new(2, Spin::Dn),
            Orbital::new(4, Spin::Dn),
            "j.site == k.site",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(0, Spin::Up),
            Orbital::new(1, Spin::Dn),
            Orbital::new(1, Spin::Dn),
            "up-density(0) × dn-density(1) [(N,Id)+(Id,N)]",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(0, Spin::Up),
            Orbital::new(0, Spin::Dn),
            Orbital::new(0, Spin::Dn),
            "single-site NN (n↑·n↓) [(N,N)]",
        ),
        (
            Orbital::new(1, Spin::Up),
            Orbital::new(0, Spin::Up),
            Orbital::new(0, Spin::Dn),
            Orbital::new(0, Spin::Dn),
            "AN(Up): a↑·n↓ same site [(A,N)]",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(0, Spin::Up),
            Orbital::new(1, Spin::Dn),
            Orbital::new(0, Spin::Dn),
            "AN(Dn): n↑·a↓ same site [(N,A)]",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(1, Spin::Up),
            Orbital::new(0, Spin::Dn),
            Orbital::new(0, Spin::Dn),
            "AdagN(Up): a†↑·n↓ same site [(Adag,N)]",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(0, Spin::Up),
            Orbital::new(0, Spin::Dn),
            Orbital::new(1, Spin::Dn),
            "AdagN(Dn): n↑·a†↓ same site [(N,Adag)]",
        ),
        (
            Orbital::new(0, Spin::Up),
            Orbital::new(1, Spin::Up),
            Orbital::new(2, Spin::Up),
            Orbital::new(3, Spin::Up),
            "same-spin throughout",
        ),
    ];
    for (i, j, k, l, label) in cases {
        assert_ne!(i, k, "{label}: precondition i≠k");
        assert_ne!(j, l, "{label}: precondition j≠l");
        let coeff = 0.6_f64;
        let x = adag_i_adag_k_a_l_a_j(&x0, i, j, k, l, coeff);
        let refr: Vec<f64> =
            dense_apply_string(d, &[(i, true), (k, true), (l, false), (j, false)], &x0d)
                .into_iter()
                .map(|v| v * coeff)
                .collect();
        assert!(
            dense_norm(&refr) > 1e-9,
            "{label}: reference is zero — operator annihilated x0, test is vacuous"
        );
        let got = x.to_dense();
        let tol = 50.0 * d as f64 * f64::EPSILON;
        assert!(
            rel_err(&got, &refr) < tol,
            "{label}: coincidence-class dense mismatch"
        );
    }
}

#[test]
fn t2_elem_2body_dense_oracle() {
    let d = 4usize;
    let space = QnSpace::new(d, 2, 2);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x2E2);
    let x0: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
    let x0d = x0.to_dense();

    let i = Orbital::new(0, Spin::Up);
    let j = Orbital::new(1, Spin::Up);
    let k = Orbital::new(2, Spin::Dn);
    let l = Orbital::new(3, Spin::Dn);
    let coeff = 0.42_f64;
    let x = adag_i_adag_k_a_l_a_j(&x0, i, j, k, l, coeff);

    let refr: Vec<f64> =
        dense_apply_string(d, &[(i, true), (k, true), (l, false), (j, false)], &x0d)
            .into_iter()
            .map(|v| v * coeff)
            .collect();
    let got = x.to_dense();
    let tol = 50.0 * d as f64 * f64::EPSILON;
    assert!(rel_err(&got, &refr) < tol, "2-body dense oracle mismatch");
}

#[allow(clippy::too_many_arguments)]
fn occupancy_case(
    d: usize,
    up_a: &[usize],
    dn_a: &[usize],
    up_b: &[usize],
    dn_b: &[usize],
    i: Orbital,
    j: Orbital,
    k: Orbital,
    l: Orbital,
    expected: &[(usize, f64)],
    label: &str,
) {
    let x0a: TtVector<f64> = state_from_sites(d, up_a, dn_a);
    let x0b: TtVector<f64> = state_from_sites(d, up_b, dn_b);
    let x0 = &x0a + &x0b;
    let x = adag_i_adag_k_a_l_a_j(&x0, i, j, k, l, 1.0);
    x.check();
    let dxx = dot(&x, &x).re_s();
    assert!(dxx > 1e-10, "{label}: operator annihilated the input");

    let mut nx = vec![0.0f64; d];
    for (s, nx_s) in nx.iter_mut().enumerate() {
        let nup = adag_i_a_j(
            &x,
            Orbital::new(s, Spin::Up),
            Orbital::new(s, Spin::Up),
            1.0,
        );
        let ndn = adag_i_a_j(
            &x,
            Orbital::new(s, Spin::Dn),
            Orbital::new(s, Spin::Dn),
            1.0,
        );
        *nx_s = (dot(&x, &nup).re_s() + dot(&x, &ndn).re_s()) / dxx;
    }
    let mut nref = vec![0.0f64; d];
    for &(s, v) in expected {
        nref[s] = v;
    }
    let dev: f64 = nx
        .iter()
        .zip(&nref)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    assert!(
        dev < 1e-10,
        "{label}: occupancy mismatch, nx={nx:?} nref={nref:?}"
    );
}

#[test]
fn t2_elem_2body_occupancy() {
    let d = 8usize;

    occupancy_case(
        d,
        &[1, 2],
        &[6],
        &[3, 2],
        &[6],
        Orbital::new(0, Spin::Up),
        Orbital::new(2, Spin::Up),
        Orbital::new(4, Spin::Dn),
        Orbital::new(6, Spin::Dn),
        &[(0, 1.0), (1, 0.5), (3, 0.5), (4, 1.0)],
        "mixed-spin generic",
    );

    occupancy_case(
        d,
        &[1, 2, 6],
        &[],
        &[3, 2, 6],
        &[],
        Orbital::new(0, Spin::Up),
        Orbital::new(2, Spin::Up),
        Orbital::new(4, Spin::Up),
        Orbital::new(6, Spin::Up),
        &[(0, 1.0), (1, 0.5), (3, 0.5), (4, 1.0)],
        "same-spin (Up)",
    );

    occupancy_case(
        d,
        &[2, 4],
        &[7],
        &[3, 4],
        &[7],
        Orbital::new(1, Spin::Up),
        Orbital::new(4, Spin::Up),
        Orbital::new(4, Spin::Dn),
        Orbital::new(7, Spin::Dn),
        &[(1, 1.0), (2, 0.5), (3, 0.5), (4, 1.0)],
        "j.site == k.site coincidence",
    );
}

#[test]
fn t2_ham_2body_eigenpair() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let (a, b, c, e) = (0usize, 1usize, 2usize, 3usize);

    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[c, a, e, b], -1.0);
    v.set(&[b, e, a, c], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let basis: Vec<(usize, usize)> = (0..d).flat_map(|u| (0..d).map(move |w| (u, w))).collect();
    let n = basis.len();
    let mut hd = faer::Mat::<f64>::zeros(n, n);
    for (col, &(u, w)) in basis.iter().enumerate() {
        let x0: TtVector<f64> = state_from_sites(d, &[u], &[w]);
        let y = &h * &x0;
        let yd = y.to_dense();
        for (row, &(ur, wr)) in basis.iter().enumerate() {
            *hd.get_mut(row, col) = yd[det_index(d, ur, wr)];
        }
    }
    let eigs = hd
        .as_ref()
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .expect("self-adjoint eigendecomposition");
    let min_eig = eigs.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        (min_eig - (-0.5)).abs() < 1e-10,
        "2-body analytic ground energy mismatch: min_eig={min_eig}"
    );
}
