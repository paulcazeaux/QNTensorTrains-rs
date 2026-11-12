// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::*;

use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{Sector, max_ranks};
use crate::round::RoundMode;
use crate::solvers::riemannian::tangent_bases;
use crate::test_util::{assert_purity, randomized_state};
use crate::vector::{tt_randn, tt_zeros};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const D: usize = 4;
const NUP: usize = 2;
const NDN: usize = 1;

fn fx_pg() -> PointGroup {
    PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
    )
}

fn fx_space(target: Irrep) -> QnSpace {
    QnSpace::with_point_group(D, NUP, NDN, fx_pg(), target)
}

fn shift(s: Sector, phi: Irrep) -> Sector {
    Sector::with_irrep(s.nup, s.ndn, s.irrep.fuse(phi))
}

fn fx_random(seed: u64) -> (QnSpace, TtVector<f64>, TtVector<f64>, TtVector<f64>) {
    let space = fx_space(Irrep::IDENTITY);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let (bl, br) = tangent_bases(&x);
    (space, x, bl, br)
}

fn fx_full_rank(seed: u64) -> (QnSpace, TtVector<f64>, TtVector<f64>, TtVector<f64>) {
    let space = fx_space(Irrep::IDENTITY);
    let grids = max_ranks(&space);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = tt_randn(space, &grids, &mut rng);
    x.scale_mut(1.0 / x.norm());
    let (bl, br) = tangent_bases(&x);
    (space, x, bl, br)
}

fn fx_truncated_22(seed: u64) -> (QnSpace, TtVector<f64>, TtVector<f64>, TtVector<f64>) {
    let space = QnSpace::with_point_group(D, 2, 2, fx_pg(), Irrep::IDENTITY);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut x: TtVector<f64> = randomized_state(space, 1, 1, &mut rng);
    x.round_mut(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let (bl, br) = tangent_bases(&x);
    (space, x, bl, br)
}

fn dense_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn dense_nrm(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn dense_axpy(y: &mut [f64], alpha: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}
fn dense_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn all_phi() -> Vec<Irrep> {
    (0..fx_pg().order()).map(Irrep::new).collect()
}

#[test]
fn tsh_shifted_frames_are_structurally_valid() {
    let (space, _x, bl, br) = fx_random(0x5E01);
    for phi in all_phi() {
        let space_out = space.with_target(space.target.fuse(phi));
        let (u, v) = shifted_frames(&bl, &br, phi);

        assert_eq!(u.space, space_out, "phi {phi:?}: left frame not restamped");
        assert_eq!(v.space, space_out, "phi {phi:?}: V' not stamped space_out");
        assert_eq!(u.ranks, bl.ranks, "phi {phi:?}: restamp moved a rank");
        for k in 0..D {
            u.core(k).check_invariants();
            v.core(k).check_invariants();
        }
        for b in 0..=D {
            for s in space_out.bond_sectors(b) {
                assert_eq!(
                    v.ranks[b][s],
                    br.ranks[b][shift(s, phi)],
                    "phi {phi:?}, bond {b}, sector {s:?}: V' rank law"
                );
            }
        }
        let last = Sector::with_irrep(NUP, NDN, space_out.target);
        for (s, &r) in v.ranks[D].iter() {
            assert!(
                r == 0 || s == last,
                "phi {phi:?}: V' has rank {r} at bond {D} sector {s:?}, outside {last:?}"
            );
        }
        assert_eq!(v.ranks[D][last], 1, "phi {phi:?}: V' bond-d rank must be 1");
        let first = Sector::with_irrep(0, 0, phi);
        for (s, &r) in v.ranks[0].iter() {
            assert!(
                r == 0 || s == first,
                "phi {phi:?}: V' has rank {r} at bond 0 sector {s:?}, outside {first:?}"
            );
        }
        assert_eq!(
            v.ranks[0][first], 1,
            "phi {phi:?}: V' bond-0 label must be (0,0,phi) — this is the law that makes V' \
             illegal as a TtVector"
        );
    }
}

#[test]
fn tsh_core_restamp_and_relabel() {
    let (space, _x, _bl, br) = fx_random(0x5E0E);

    let same = |a: Option<faer::MatRef<'_, f64>>, b: Option<faer::MatRef<'_, f64>>| -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => {
                if (x.nrows(), x.ncols()) != (y.nrows(), y.ncols()) {
                    return false;
                }
                (0..x.nrows()).all(|i| (0..x.ncols()).all(|j| *x.get(i, j) == *y.get(i, j)))
            }
            _ => false,
        }
    };

    for phi in all_phi() {
        let space_out = space.with_target(space.target.fuse(phi));
        for k in 0..D {
            let src = br.core(k);

            let re = src.restamp(space_out);
            assert_eq!(re.space, space_out, "restamp did not take the new space");
            assert_eq!(re.row_ranks, src.row_ranks, "restamp moved a row rank");
            assert_eq!(re.col_ranks, src.col_ranks, "restamp moved a column rank");
            re.check_invariants();
            for f in Family::ALL {
                for s in space.bond_sectors(k) {
                    assert!(
                        same(src.block(f, s), re.block(f, s)),
                        "restamp changed the data at site {k}, {f:?}, {s:?}"
                    );
                }
            }

            let out = src.clone().relabel_irrep_into(phi, space_out);
            out.check_invariants();
            assert_eq!(out.space, space_out);
            for s in space_out.bond_sectors(k) {
                assert_eq!(
                    out.row_ranks[s],
                    src.row_ranks[shift(s, phi)],
                    "phi {phi:?} site {k}: row rank law at {s:?}"
                );
            }
            for s in space_out.bond_sectors(k + 1) {
                assert_eq!(
                    out.col_ranks[s],
                    src.col_ranks[shift(s, phi)],
                    "phi {phi:?} site {k}: column rank law at {s:?}"
                );
            }
            for f in Family::ALL {
                for s in space_out.bond_sectors(k) {
                    assert!(
                        same(out.block(f, s), src.block(f, shift(s, phi))),
                        "phi {phi:?} site {k}: block law at {f:?}, {s:?}"
                    );
                }
            }

            let out2 = src.relabel_irrep(phi);
            assert_eq!(out2.row_ranks, out.row_ranks);
            assert_eq!(out2.col_ranks, out.col_ranks);
            for f in Family::ALL {
                for s in space.bond_sectors(k) {
                    assert!(
                        same(out2.block(f, s), out.block(f, s)),
                        "phi {phi:?} site {k}: relabel_irrep and relabel_irrep_into disagree at \
                         {f:?}, {s:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn tsh_project_shifted_is_idempotent() {
    let (space, _x, bl, br) = fx_random(0x5E02);
    let mut live = 0usize;
    for phi in all_phi() {
        let space_out = space.with_target(space.target.fuse(phi));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E02_0000 + phi.index() as u64);
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let p = TtTangent::project_shifted(&bl, &br, phi, &y);
        assert_eq!(p.phi(), phi, "phi {phi:?}: projection lost its flux label");
        let n = p.norm();
        if n <= 1e-12 {
            continue;
        }
        live += 1;
        let pp = TtTangent::project_shifted(&bl, &br, phi, &p.to_vector());
        let err = p.sub(&pp).norm() / n;
        println!("tsh idempotence phi={phi:?}: rel err {err:e} (|P y| = {n:e})");
        assert!(
            err <= 1e-12,
            "phi {phi:?}: P(iota(P y)) != P y, relative error {err:e}"
        );
    }
    assert!(
        live >= 2,
        "anti-vacuity: only {live} of the four channels carried a nonzero projection"
    );
}

#[test]
fn tsh_project_shifted_gauge_condition() {
    let (space, _x, bl, br) = fx_random(0x5E03);
    let mut nonzero_components = 0usize;
    for phi in all_phi() {
        let space_out = space.with_target(space.target.fuse(phi));
        let (u, _v) = shifted_frames(&bl, &br, phi);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E03_0000 + phi.index() as u64);
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let xi = TtTangent::project_shifted(&bl, &br, phi, &y);
        if xi.norm() <= 1e-12 {
            continue;
        }
        for k in 0..D - 1 {
            for t in space_out.bond_sectors(k + 1) {
                let l = u.core(k).vertical_unfolding(t);
                let c = xi.component(k).vertical_unfolding(t);
                if l.ncols() == 0 || c.ncols() == 0 || l.nrows() == 0 {
                    continue;
                }
                assert_eq!(
                    l.nrows(),
                    c.nrows(),
                    "phi {phi:?} k={k} t={t:?}: the two unfoldings must share the row stack"
                );
                let mut n2 = 0.0f64;
                let mut cn2 = 0.0f64;
                for j in 0..c.ncols() {
                    for i in 0..l.ncols() {
                        let mut acc = 0.0f64;
                        for kk in 0..l.nrows() {
                            acc += *l.get(kk, i) * *c.get(kk, j);
                        }
                        n2 += acc * acc;
                    }
                    for kk in 0..c.nrows() {
                        cn2 += *c.get(kk, j) * *c.get(kk, j);
                    }
                }
                if cn2.sqrt() > 1e-8 {
                    nonzero_components += 1;
                }
                assert!(
                    n2.sqrt() < 1e-12,
                    "phi {phi:?}: gauge condition at bond {}, sector {t:?}: {:e}",
                    k + 1,
                    n2.sqrt()
                );
            }
        }
    }
    assert!(
        nonzero_components > 0,
        "anti-vacuity: every gauged component block was zero, so the condition held trivially"
    );
}

#[test]
fn tsh_shifted_embedding_is_isometric() {
    let (space, _x, bl, br) = fx_random(0x5E04);
    let mut live = 0usize;
    for phi in all_phi() {
        let space_out = space.with_target(space.target.fuse(phi));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E04_0000 + phi.index() as u64);
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let xi = TtTangent::project_shifted(&bl, &br, phi, &y);
        let n = xi.norm();
        if n <= 1e-12 {
            continue;
        }
        live += 1;
        let vn = xi.to_vector().norm();
        let err = (vn - n).abs() / n;
        println!("tsh isometry phi={phi:?}: |iota(xi)| = {vn:e} vs |xi| = {n:e} (rel {err:e})");
        assert!(
            err <= 1e-12,
            "phi {phi:?}: iota is not an isometry, {vn:e} vs {n:e}"
        );
    }
    assert!(live >= 2, "anti-vacuity: only {live} live channels");
}

#[test]
fn tsh_identity_phi_reproduces_project_bit_for_bit() {
    let (space, _x, bl, br) = fx_random(0x5E05);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E05_0001);
    let dir: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);

    let a = TtTangent::project(&bl, &br, &dir);
    let b = TtTangent::project_shifted(&bl, &br, Irrep::IDENTITY, &dir);
    assert!(
        a.norm() > 1e-3,
        "anti-vacuity: the reference projection is 0"
    );

    assert_eq!(b.phi(), Irrep::IDENTITY);
    assert_eq!(
        b.rank_col(),
        a.rank(),
        "at IDENTITY the column grids ARE the row grids"
    );
    assert_eq!(
        a.sub(&b).norm(),
        0.0,
        "project_shifted(IDENTITY) must be bit-identical to project"
    );
    assert_eq!(
        a.to_vector().to_dense(),
        b.to_vector().to_dense(),
        "to_vector at IDENTITY must be bit-identical to today's three-band layout"
    );
    let s0 = tangent_basis_shifted(&bl, &br, Irrep::IDENTITY);
    let t0 = tangent_basis(&bl, &br, false);
    assert_eq!(
        s0.len(),
        t0.len(),
        "tangent_basis_shifted(IDENTITY) must be tangent_basis(.., exclude_radial = false)"
    );
    for (j, (p, q)) in s0.iter().zip(t0.iter()).enumerate() {
        assert_eq!(p.sub(q).norm(), 0.0, "basis vector {j} differs at IDENTITY");
    }
}

#[test]
fn tsh_project_shifted_dense_oracle() {
    let (space, _x, bl, br) = fx_random(0x5E06);
    let mut live = 0usize;
    for phi in all_phi() {
        let space_out = space.with_target(space.target.fuse(phi));
        let basis = tangent_basis_shifted(&bl, &br, phi);
        if basis.is_empty() {
            continue;
        }
        let bd: Vec<Vec<f64>> = basis.iter().map(|b| b.to_vector().to_dense()).collect();
        for i in 0..bd.len() {
            for j in 0..bd.len() {
                let g = dense_dot(&bd[i], &bd[j]);
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (g - want).abs() <= 1e-9,
                    "phi {phi:?}: basis Gram[{i}][{j}] = {g}, expected {want}"
                );
            }
        }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E06_0000 + phi.index() as u64);
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let yd = y.to_dense();
        let got = TtTangent::project_shifted(&bl, &br, phi, &y)
            .to_vector()
            .to_dense();
        let mut want = vec![0.0f64; yd.len()];
        for b in &bd {
            dense_axpy(&mut want, dense_dot(b, &yd), b);
        }
        let scale = dense_nrm(&want);
        if scale <= 1e-6 * dense_nrm(&yd) {
            continue;
        }
        live += 1;
        let err = dense_nrm(&dense_sub(&got, &want)) / scale;
        println!(
            "tsh dense oracle phi={phi:?}: dim {} rel err {err:e}",
            basis.len()
        );
        assert!(
            err <= 1e-9,
            "phi {phi:?}: P^phi is not the dense orthogonal projection onto span(iota(basis)): \
             rel err {err:e}"
        );
    }
    assert!(
        live >= 2,
        "anti-vacuity: only {live} channels carried a resolvable projection"
    );
}

#[test]
fn tsh_shifted_to_vector_is_a_pure_train() {
    let (space, _x, bl, br) = fx_random(0x5E07);
    let pg = fx_pg();
    let mut live = 0usize;
    for phi in all_phi() {
        let space_out = space.with_target(space.target.fuse(phi));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E07_0000 + phi.index() as u64);
        let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
        let xi = TtTangent::project_shifted(&bl, &br, phi, &y);
        if xi.norm() <= 1e-12 {
            continue;
        }
        live += 1;
        let v = xi.to_vector();
        assert_eq!(v.space, space_out, "phi {phi:?}: wrong output space");
        assert!(v.check(), "phi {phi:?}: iota(xi) is not a legal train");
        assert_purity(&pg, D, NUP, NDN, space_out.target, &v.to_dense());
    }
    assert!(live >= 2, "anti-vacuity: only {live} live channels");
}

fn a_shifted_tangent() -> (TtVector<f64>, TtTangent<f64>) {
    let (space, x, bl, br) = fx_random(0x5E08);
    let phi = Irrep::new(1);
    let space_out = space.with_target(space.target.fuse(phi));
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E08_0001);
    let y: TtVector<f64> = randomized_state(space_out, 1, 2, &mut rng);
    let xi = TtTangent::project_shifted(&bl, &br, phi, &y);
    assert!(
        xi.norm() > 1e-8,
        "fixture: the shifted direction must be nonzero for the refusal arms to mean anything"
    );
    (x, xi)
}

#[test]
#[should_panic]
fn tsh_moved_point_refuses_a_shifted_tangent() {
    let (_x, xi) = a_shifted_tangent();
    let _ = xi.moved_point(0.3);
}

#[test]
#[should_panic]
fn tsh_retract_hosvd_refuses_a_shifted_tangent() {
    let (_x, xi) = a_shifted_tangent();
    let _ = xi.retract_hosvd(0.3);
}

#[test]
#[should_panic]
fn tsh_retract_round_refuses_a_shifted_tangent() {
    let (_x, xi) = a_shifted_tangent();
    let _ = xi.retract_round(0.3, 1e-10, crate::qn::Pooled::NONE, RoundMode::Rel);
}

#[test]
#[should_panic]
fn tsh_projective_transport_refuses_a_shifted_tangent() {
    let (x, xi) = a_shifted_tangent();
    let _ = projective_transport(&x, &xi);
}

#[test]
fn tsh_dim_grids_matches_the_shifted_basis_length() {
    let (space, _x, bl, br) = fx_full_rank(0x5E09);
    assert_eq!(
        tangent_dim_grids(&space, &bl.ranks, Irrep::IDENTITY),
        tangent_dim(&bl),
        "tangent_dim must be the IDENTITY slice of tangent_dim_grids"
    );
    let mut total = 0usize;
    for phi in all_phi() {
        let dim = tangent_dim_grids(&space, &bl.ranks, phi);
        let basis = tangent_basis_shifted(&bl, &br, phi);
        println!("tsh dim phi={phi:?}: dim {dim}, basis {}", basis.len());
        assert_eq!(
            basis.len(),
            dim,
            "phi {phi:?}: full-rank basis length {} != structural dim {dim}",
            basis.len()
        );
        total += dim;
    }
    assert!(total > 0, "anti-vacuity: every channel came out empty");

    let (space_t, _xt, blt, brt) = fx_truncated_22(0x5E09 ^ 0xFF);
    for phi in all_phi() {
        let dim = tangent_dim_grids(&space_t, &blt.ranks, phi);
        let got = tangent_basis_shifted(&blt, &brt, phi).len();
        assert!(
            got <= dim,
            "phi {phi:?}: truncated-rank basis returned {got} vectors for a {dim}-dimensional \
             space (over-acceptance => rank-deficient Grams)"
        );
    }
}

#[test]
fn tsh_dim_grids_sum_to_the_untyped_tangent_dim() {
    for (label, (space, _x, bl, _br)) in [
        ("full-rank (2,1)", fx_full_rank(0x5E0A)),
        ("random (2,1)", fx_random(0x5E0B)),
        ("truncated (2,2)", fx_truncated_22(0x5E0D)),
    ] {
        let sum: usize = all_phi()
            .into_iter()
            .map(|phi| tangent_dim_grids(&space, &bl.ranks, phi))
            .sum();
        let untyped = tangent_dim(&bl.forget_point_group());
        println!("tsh sum ({label}): Sum_phi dim = {sum}, untyped tangent_dim = {untyped}");
        assert!(sum > 0, "{label}: anti-vacuity, the sum is zero");
        assert_eq!(
            sum, untyped,
            "{label}: Sum_phi dim T^phi != tangent_dim(untyped embedding)"
        );
    }
}

#[test]
fn tsh_unreachable_phi_is_empty_and_does_not_panic() {
    let pg = PointGroup::new(
        2,
        &[Irrep::new(0), Irrep::new(1), Irrep::new(1), Irrep::new(0)],
    );
    let space = QnSpace::with_point_group(D, NUP, NDN, pg, Irrep::IDENTITY);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5E0C);
    let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
    x.round_mut(1e-8, crate::qn::Pooled::NONE, RoundMode::Rel);
    x.scale_mut(1.0 / x.norm());
    let (bl, br) = tangent_bases(&x);

    for p in [2u8, 3] {
        let phi = Irrep::new(p);
        let space_out = space.with_target(phi);
        assert_eq!(
            tangent_dim_grids(&space, &bl.ranks, phi),
            0,
            "phi {p} is structurally unreachable and must have dimension 0"
        );
        assert!(
            tangent_basis_shifted(&bl, &br, phi).is_empty(),
            "phi {p}: an unreachable channel returned a basis"
        );
        let y: TtVector<f64> = tt_zeros(space_out);
        let xi = TtTangent::project_shifted(&bl, &br, phi, &y);
        assert_eq!(xi.phi(), phi);
        assert_eq!(xi.norm(), 0.0, "phi {p}: projection into an empty channel");
    }
    for p in [0u8, 1] {
        let phi = Irrep::new(p);
        assert!(
            tangent_dim_grids(&space, &bl.ranks, phi) > 0,
            "phi {p} must be reachable in this fixture"
        );
        assert!(!tangent_basis_shifted(&bl, &br, phi).is_empty());
    }
}
