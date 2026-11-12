// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg4::util::*;
use qntt::hamiltonian::Construction;
use qntt::{SparseHamiltonian, TtVector, rayleigh_quotient};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn pg4_c2v_coeffs_fixture_is_nonvacuous() {
    let (pg, space) = g_a();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(50_001);
    let (t, _v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);
}

#[test]
fn pg4_h_apply_matches_dense_oracle_ga() {
    let (pg, space) = g_a();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(50_010);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);

    let x: TtVector<f64> = capped_randn(space, 1, 2, 50_011);
    let xd = x.to_dense();
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &xd);

    let y = h.apply(&x);
    let yd = y.to_dense();
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
    let want = dense_apply_hamiltonian_12(space.d, &t, &v, &xd);
    assert!(
        rel_diff(&yd, &want) < 1e-9,
        "H·x must match the JW dense oracle at C2v"
    );
    assert!(
        yd.iter().any(|&v| v.abs() > 1e-9),
        "vacuous: H·x is exactly zero — the operator-flux migration is silently dropping every \
         edge (spec 1 finding F4)"
    );
}

#[test]
fn pg4_h_apply_matches_dense_oracle_gb() {
    let (pg, space) = g_b();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(50_020);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);

    let x: TtVector<f64> = capped_randn(space, 1, 2, 50_021);
    let xd = x.to_dense();
    let y = h.apply(&x);
    let yd = y.to_dense();

    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
    let want = dense_apply_hamiltonian_12(space.d, &t, &v, &xd);
    assert!(rel_diff(&yd, &want) < 1e-9);
    assert!(yd.iter().any(|&v| v.abs() > 1e-9), "vacuous H·x");
}

#[test]
fn pg4_h_apply_matches_dense_oracle_gc() {
    let (pg, space) = g_c();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(50_030);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);

    let x: TtVector<f64> = capped_randn(space, 1, 2, 50_031);
    let xd = x.to_dense();
    let y = h.apply(&x);
    let yd = y.to_dense();

    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
    let want = dense_apply_hamiltonian_12(space.d, &t, &v, &xd);
    assert!(rel_diff(&yd, &want) < 1e-9);
    assert!(yd.iter().any(|&v| v.abs() > 1e-9), "vacuous H·x");
}

#[test]
fn pg4_median_bipartite_agree_at_nontrivial_group() {
    let (pg, space) = g_a();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(50_040);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);

    let h_bip: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    let h_med: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Median);

    let x: TtVector<f64> = capped_randn(space, 1, 2, 50_041);
    let y_bip = h_bip.apply(&x).to_dense();
    let y_med = h_med.apply(&x).to_dense();
    assert!(
        rel_diff(&y_bip, &y_med) < 1e-12,
        "Median and Bipartite H·x must agree at C2v"
    );
    assert!(
        y_bip.iter().any(|&v| v.abs() > 1e-9),
        "vacuous: both constructions agree on zero"
    );

    let rq_bip = rayleigh_quotient(&h_bip, &x);
    let rq_med = rayleigh_quotient(&h_med, &x);
    assert!(
        (rq_bip - rq_med).abs() <= 1e-10 * rq_bip.abs().max(1.0),
        "Median and Bipartite rayleigh_quotient must agree at C2v"
    );
}

#[test]
fn pg4_all_five_constructions_agree_at_nontrivial_group() {
    let (pg, space) = g_a();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(50_042);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);

    let constructions = [
        Construction::Median,
        Construction::Bipartite,
        Construction::BipartitePerSector,
        Construction::MedianCoverSwitch,
        Construction::BipartiteMedianSwitch,
    ];

    let x: TtVector<f64> = capped_randn(space, 1, 2, 50_043);
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(constructions.len());
    let mut rqs: Vec<f64> = Vec::with_capacity(constructions.len());
    for &c in &constructions {
        let h: SparseHamiltonian<f64> =
            SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, c);
        ys.push(h.apply(&x).to_dense());
        rqs.push(rayleigh_quotient(&h, &x));
    }

    assert!(
        ys[0].iter().any(|&v| v.abs() > 1e-9),
        "vacuous: reference construction's H·x is exactly zero"
    );
    for (i, &c) in constructions.iter().enumerate().skip(1) {
        assert!(
            rel_diff(&ys[i], &ys[0]) < 1e-12,
            "{c:?} H·x must agree with {:?} at C2v",
            constructions[0]
        );
        assert!(
            (rqs[i] - rqs[0]).abs() <= 1e-10 * rqs[0].abs().max(1.0),
            "{c:?} rayleigh_quotient must agree with {:?} at C2v",
            constructions[0]
        );
    }
}

#[test]
fn pg4_h_apply_matches_dense_oracle_c64() {
    use qntt::{C64, CoeffTensor, Scalar};
    let (pg, space) = g_a();
    let d = space.d;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC6_4001);
    let (tr, vr) = c2v_coeffs(&pg, d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, d, &tr);
    let mut t = CoeffTensor::<C64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            if j > i {
                continue;
            }
            let re = tr.get(&[i, j]);
            if i == j {
                t.set(&[i, i], C64::from_f64(re));
            } else if re != 0.0 {
                let phi = 0.25 + 0.13 * (i + 3 * j) as f64;
                let z = C64::new(re * phi.cos(), re * phi.sin());
                t.set(&[i, j], z);
                t.set(&[j, i], z.conj());
            }
        }
    }
    let mut v = CoeffTensor::<C64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    v.set(&[i, j, k, l], C64::from_f64(vr.get(&[i, j, k, l])));
                }
            }
        }
    }

    let h: SparseHamiltonian<C64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    let x: TtVector<C64> = capped_randn(space, 1, 2, 0xC6_4002);
    let xd = x.to_dense();
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &xd);

    let y = h.apply(&x);
    let yd = y.to_dense();
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
    let want = dense_apply_hamiltonian_12(space.d, &t, &v, &xd);
    let num = yd
        .iter()
        .zip(&want)
        .map(|(a, b)| (*a - *b).abs_s().powi(2))
        .sum::<f64>()
        .sqrt();
    let den = want.iter().map(|z| z.abs_s().powi(2)).sum::<f64>().sqrt();
    assert!(
        num <= 1e-9 * den.max(1.0),
        "C64 H·x must match the JW dense oracle at C2v: rel {:e}",
        num / den.max(1e-300)
    );
    assert!(
        yd.iter().any(|z| z.im.abs() > 1e-9),
        "vacuous: no imaginary mass reached H·x — the complex branch never ran"
    );
}

#[test]
fn pg4_h_apply_matches_dense_oracle_d2h_noncyclic() {
    let d = 7usize;
    let table = [0u8, 3, 5, 1, 6, 3, 0].map(qntt::Irrep);
    let pg = qntt::PointGroup::new(3, &table);
    let space = qntt::QnSpace::with_point_group(d, 2, 1, pg, qntt::Irrep::IDENTITY);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD2_8001);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert!(
        (0..d).any(|i| (0..d).any(|j| i != j && t.get(&[i, j]) != 0.0)),
        "fixture bug: no allowed off-diagonal one-body entry at this table"
    );

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    let x: TtVector<f64> = capped_randn(space, 1, 2, 0xD2_8002);
    let xd = x.to_dense();
    assert!(
        xd.iter().any(|&c| c.abs() > 1e-12),
        "fixture bug: the (2,1) identity-target sector is empty at this table"
    );
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &xd);

    let y = h.apply(&x);
    let yd = y.to_dense();
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
    let want = dense_apply_hamiltonian_12(space.d, &t, &v, &xd);
    assert!(
        rel_diff(&yd, &want) < 1e-9,
        "H·x must match the JW dense oracle at non-cyclic D2h"
    );
    assert!(
        yd.iter().any(|&c| c.abs() > 1e-9),
        "vacuous: H·x is exactly zero at the D2h table"
    );
}
