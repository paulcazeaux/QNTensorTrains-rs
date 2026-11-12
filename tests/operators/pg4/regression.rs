// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::pg4::util::*;
use qntt::coeff::ThreeBody;
use qntt::hamiltonian::Construction;
use qntt::{
    CoeffTensor, Irrep, Orbital, PointGroup, PointGroupKind, QnSpace, Scalar, SparseHamiltonian,
    Spin, TtVector, adag_i_a_j, adag_i_adag_k_a_l_a_j, tt_zeros,
};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;

fn find_same_irrep_pair(pg: &PointGroup, d: usize) -> (usize, usize) {
    for i in 0..d {
        for j in 0..d {
            if i != j && pg.irrep_of(i) == pg.irrep_of(j) {
                return (i, j);
            }
        }
    }
    panic!("fixture: no same-irrep pair found in 0..{d}");
}

fn find_different_irrep_pair(pg: &PointGroup, d: usize) -> (usize, usize) {
    for i in 0..d {
        for j in 0..d {
            if i != j && pg.irrep_of(i) != pg.irrep_of(j) {
                return (i, j);
            }
        }
    }
    panic!("fixture: no different-irrep pair found in 0..{d}");
}

fn find_symmetric_quadruple(pg: &PointGroup, d: usize) -> (usize, usize, usize, usize) {
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if i == k || j == l {
                        continue;
                    }
                    let tot = pg
                        .irrep_of(i)
                        .fuse(pg.irrep_of(j))
                        .fuse(pg.irrep_of(k))
                        .fuse(pg.irrep_of(l));
                    let any_nontrivial = [i, j, k, l]
                        .iter()
                        .any(|&s| pg.irrep_of(s) != Irrep::IDENTITY);
                    if tot == Irrep::IDENTITY && any_nontrivial {
                        return (i, j, k, l);
                    }
                }
            }
        }
    }
    panic!("fixture: no symmetry-respecting, non-vacuous quadruple found in 0..{d}");
}

#[test]
fn pg4_adag_i_a_j_matches_dense_oracle_at_nontrivial_group() {
    let (pg, space) = g_a();
    let (i_site, j_site) = find_same_irrep_pair(&pg, space.d);
    assert_ne!(
        pg.irrep_of(i_site),
        Irrep::IDENTITY,
        "vacuous: the matched same-irrep pair is trivially IDENTITY on both sites — pick a \
         different fixture"
    );
    let i = Orbital::new(i_site, Spin::Up);
    let j = Orbital::new(j_site, Spin::Up);

    let x: TtVector<f64> = capped_randn(space, 1, 2, 80_001);
    let xd = x.to_dense();
    let y = adag_i_a_j(&x, i, j, 1.0f64);
    assert!(y.check(), "adag_i_a_j output must pass check()");
    let yd = y.to_dense();

    let want = dense_apply_string(space.d, &[(i, true), (j, false)], &xd);
    assert!(
        rel_diff(&yd, &want) < 1e-9,
        "adag_i_a_j must match the JW dense oracle at |G|>1"
    );
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
    assert!(
        yd.iter().any(|&v| v.abs() > 1e-9),
        "vacuous: a_i a_j output is zero"
    );
}

#[test]
fn pg4_adag_2body_matches_dense_oracle_at_nontrivial_group() {
    let (pg, space) = g_a();
    let (i_site, j_site, k_site, l_site) = find_symmetric_quadruple(&pg, space.d);
    let i = Orbital::new(i_site, Spin::Up);
    let j = Orbital::new(j_site, Spin::Up);
    let k = Orbital::new(k_site, Spin::Up);
    let l = Orbital::new(l_site, Spin::Up);

    let x: TtVector<f64> = capped_randn(space, 1, 2, 80_011);
    let xd = x.to_dense();
    let y = adag_i_adag_k_a_l_a_j(&x, i, j, k, l, 1.0f64);
    assert!(y.check(), "adag_i_adag_k_a_l_a_j output must pass check()");
    let yd = y.to_dense();

    let want = dense_apply_string(
        space.d,
        &[(i, true), (k, true), (l, false), (j, false)],
        &xd,
    );
    assert!(
        rel_diff(&yd, &want) < 1e-9,
        "adag_i_adag_k_a_l_a_j must match the JW dense oracle at |G|>1"
    );
    assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
    assert!(
        yd.iter().any(|&v| v.abs() > 1e-9),
        "vacuous: two-body output is zero"
    );
}

#[test]
#[should_panic(expected = "total irrep flux")]
fn pg4_adag_i_a_j_panics_on_symmetry_violating_string() {
    let (pg, space) = g_a();
    let (i_site, j_site) = find_different_irrep_pair(&pg, space.d);
    let i = Orbital::new(i_site, Spin::Up);
    let j = Orbital::new(j_site, Spin::Up);
    let x: TtVector<f64> = capped_randn(space, 1, 2, 80_021);
    let _ = adag_i_a_j(&x, i, j, 1.0f64);
}

fn c2v_w(
    pg: &PointGroup,
    d: usize,
    count: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> CoeffTensor<f64> {
    let mut w = CoeffTensor::<f64>::zeros(d, 6);
    let mut placed = 0usize;
    let mut tries = 0usize;
    while placed < count && tries < count * 200 {
        tries += 1;
        let idx = [
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
            rng.random_range(0..d),
        ];
        let tot = idx
            .iter()
            .fold(Irrep::IDENTITY, |acc, &s| acc.fuse(pg.irrep_of(s)));
        if tot == Irrep::IDENTITY && w.get(&idx) == 0.0 {
            w.set(&idx, rng.sample::<f64, _>(StandardNormal));
            placed += 1;
        }
    }
    assert!(placed > 0, "vacuous: c2v_w placed zero entries");
    w
}

fn dense_apply_hamiltonian_123<T: Scalar>(
    d: usize,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: &CoeffTensor<T>,
    vec: &[T],
) -> Vec<T> {
    let mut out = dense_apply_hamiltonian_12(d, t, v, vec);
    let spins = [Spin::Up, Spin::Dn];
    let sixth = T::from_f64(1.0 / 6.0);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    for m in 0..d {
                        for n in 0..d {
                            let c = w.get(&[i, j, k, l, m, n]);
                            if c == T::zero() {
                                continue;
                            }
                            for &s1 in &spins {
                                for &s2 in &spins {
                                    for &s3 in &spins {
                                        let (oi, oj) = (Orbital::new(i, s1), Orbital::new(j, s1));
                                        let (ok, ol) = (Orbital::new(k, s2), Orbital::new(l, s2));
                                        let (om, on) = (Orbital::new(m, s3), Orbital::new(n, s3));
                                        let cd = oi != ok && oi != om && ok != om;
                                        let ad = oj != ol && oj != on && ol != on;
                                        if cd && ad {
                                            let term = dense_apply_string(
                                                d,
                                                &[
                                                    (oi, true),
                                                    (ok, true),
                                                    (om, true),
                                                    (on, false),
                                                    (ol, false),
                                                    (oj, false),
                                                ],
                                                vec,
                                            );
                                            let coeff = c * sixth;
                                            for (o, tm) in out.iter_mut().zip(term) {
                                                *o += tm * coeff;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

#[test]
fn pg4_h_apply_3body_matches_dense_oracle_at_nontrivial_group() {
    let (pg, space) = g_a();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(80_101);
    let (t, v) = c2v_coeffs(&pg, space.d, &mut rng);
    assert_c2v_coeffs_nonvacuous(&pg, space.d, &t);
    let w = c2v_w(&pg, space.d, 12, &mut rng);

    let x: TtVector<f64> = capped_randn(space, 1, 2, 80_102);
    let xd = x.to_dense();
    let want = dense_apply_hamiltonian_123(space.d, &t, &v, &w, &xd);
    assert!(
        want.iter().any(|&val| val.abs() > 1e-9),
        "vacuous: 3-body H·x oracle is zero"
    );

    for construction in [Construction::Bipartite, Construction::Median] {
        let h: SparseHamiltonian<f64> = SparseHamiltonian::new_with(
            space,
            &t,
            &v,
            Some(&w as &dyn ThreeBody<f64>),
            f64::EPSILON,
            construction,
        );
        let yd = h.apply(&x).to_dense();
        assert_purity(&pg, space.d, space.nup, space.ndn, space.target, &yd);
        assert!(
            rel_diff(&yd, &want) < 1e-9,
            "{construction:?}: 3-body H·x must match the dense oracle at |G|>1"
        );
    }
}

#[test]
fn pg4_zero_hamiltonian_at_nontrivial_group() {
    let (_pg, space) = g_a();
    let t = CoeffTensor::<f64>::zeros(space.d, 2);
    let v = CoeffTensor::<f64>::zeros(space.d, 4);
    let x: TtVector<f64> = capped_randn(space, 1, 2, 80_201);

    for construction in [Construction::Bipartite, Construction::Median] {
        let h: SparseHamiltonian<f64> =
            SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, construction);
        let y = h.apply(&x);
        assert!(
            y.check(),
            "{construction:?}: zero-Hamiltonian H·x must pass check()"
        );
        let yd = y.to_dense();
        assert!(
            yd.iter().all(|&val| val == 0.0),
            "{construction:?}: zero-Hamiltonian H·x must be the exact zero vector"
        );
    }
    let z: TtVector<f64> = tt_zeros(space);
    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    assert!(h.apply(&z).to_dense().iter().all(|&val| val == 0.0));
}

#[test]
fn pg4_h_apply_matches_dense_oracle_homogeneous_orbsym() {
    let d = 6;
    let orbsym = vec![2u8; d];
    let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &orbsym).expect("valid orbsym table");
    assert!(pg.order() > 1, "fixture group must be nontrivial");
    assert!(
        (1..d).all(|k| pg.irrep_of(k) == pg.irrep_of(0)),
        "fixture must be homogeneous across sites"
    );
    let space = QnSpace::with_point_group(d, 2, 2, pg, Irrep::IDENTITY);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(80_301);
    let (t, v) = c2v_coeffs(&pg, d, &mut rng);
    assert!(
        (0..d).any(|i| (0..d).any(|j| i != j && t.get(&[i, j]) != 0.0)),
        "vacuous: t has no allowed off-diagonal nonzero"
    );

    let h: SparseHamiltonian<f64> =
        SparseHamiltonian::new_with(space, &t, &v, None, f64::EPSILON, Construction::Bipartite);
    let x: TtVector<f64> = capped_randn(space, 1, 2, 80_302);
    let xd = x.to_dense();
    let yd = h.apply(&x).to_dense();

    assert_purity(&pg, d, space.nup, space.ndn, space.target, &yd);
    let want = dense_apply_hamiltonian_12(d, &t, &v, &xd);
    assert!(
        rel_diff(&yd, &want) < 1e-9,
        "homogeneous-orbsym H·x must match the dense oracle"
    );
    assert!(yd.iter().any(|&v| v.abs() > 1e-9), "vacuous: H·x is zero");
}
