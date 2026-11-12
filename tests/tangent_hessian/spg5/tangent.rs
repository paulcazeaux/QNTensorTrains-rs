// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::Irrep;
use qntt::su2::core::Family;
use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{normalize_mut, tangent_bases};
use qntt::su2::tangent::Su2TtTangent;
use qntt::su2::vector::{Su2TtVector, tt_randn as su2_tt_randn};
use qntt::su2::weingarten::{tangent_basis, tangent_dim};

use crate::spg5::util::*;

#[test]
fn spg5_y1_tangent_dim_equals_the_enumerated_basis_at_nontrivial_group() {
    let d = 4;
    let pg = c2v_table(d);
    let (n, two_s) = (3usize, 1usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: no nonidentity reachable target");
    assert_ne!(target, Irrep::IDENTITY);
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0xC5_0101);
    let (base_l, base_r) = tangent_bases(&x);
    let space = base_l.space;
    let r = &base_l.ranks;

    let dim = tangent_dim(&base_l);
    let basis = tangent_basis(&base_l, &base_r, true);
    assert_eq!(
        dim,
        basis.len() + 1,
        "tangent_dim vs enumerated basis + radial"
    );
    assert!(dim > 0, "vacuous tangent dimension");

    let mut raw = 0i64;
    let mut gauge = 0i64;
    for k in 0..d {
        let g = pg.irrep_of(k);
        for l in space.bond_sectors(k) {
            for f in Family::ALL {
                if let Some(t) = f.apply(l, g)
                    && space.contains(k + 1, t)
                {
                    raw += (r[k][l] * r[k + 1][t]) as i64;
                }
            }
        }
        if k < d - 1 {
            for s in space.bond_sectors(k + 1) {
                gauge += (r[k + 1][s] * r[k + 1][s]) as i64;
            }
        }
    }
    assert_eq!(dim as i64, raw - gauge, "independent oracle mismatch");

    let mut raw_swapped = 0i64;
    for k in 0..d {
        for l in space.bond_sectors(k) {
            for f in Family::ALL {
                if let Some(t) = f.apply(l, Irrep::IDENTITY)
                    && space.contains(k + 1, t)
                {
                    raw_swapped += (r[k][l] * r[k + 1][t]) as i64;
                }
            }
        }
    }
    assert_ne!(
        raw, raw_swapped,
        "swap detector: tangent_dim insensitive to irrep_of(k)"
    );

    let space_triv = Su2Space::new(d, n, two_s);
    let ranks_triv = su2_capped_ranks(&space_triv, 2);
    let mut rng_triv = seeded(0xC5_0104);
    let x_triv: Su2TtVector<f64> = su2_tt_randn(space_triv, &ranks_triv, &mut rng_triv);
    let (base_l_triv, _base_r_triv) = tangent_bases(&x_triv);
    let dim_triv = tangent_dim(&base_l_triv);
    assert!(
        dim < dim_triv,
        "pg-capped tangent_dim {dim} not strictly less than the irrep-blind {dim_triv}"
    );

    assert!(pg.order() > 1);
    assert!((0..d).any(|k| pg.irrep_of(k) != Irrep::IDENTITY));
}

#[test]
fn spg5_y2_tangent_projection_is_pure_idempotent_and_linear() {
    let d = 4;
    let pg = c2v_table(d);
    let (n, two_s) = (3usize, 1usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: no nonidentity reachable target");
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0xC5_0201);
    let (base_l, base_r) = tangent_bases(&x);

    let mut rng = seeded(0xC5_0202);
    let dir1: Su2TtVector<f64> = randomized_su2_state(x.space, 1, 2, &mut rng);
    let dir2: Su2TtVector<f64> = randomized_su2_state(x.space, 1, 2, &mut rng);
    let xi1 = Su2TtTangent::project(&base_l, &base_r, &dir1);
    let xi2 = Su2TtTangent::project(&base_l, &base_r, &dir2);

    let v1 = xi1.to_vector();
    assert!(v1.check());
    assert_expand_purity_chain(&v1, &pg);

    let xi1_re = Su2TtTangent::project(&base_l, &base_r, &v1);
    let diff = xi1_re.sub(&xi1);
    assert!(
        diff.norm() <= 1e-11 * xi1.norm().max(1.0),
        "project not idempotent: ‖diff‖={} ‖xi1‖={}",
        diff.norm(),
        xi1.norm()
    );

    let (a, b) = (0.7f64, -1.3f64);
    let d1 = &dir1 * a;
    let d2 = &dir2 * b;
    let comb_dir = &d1 + &d2;
    let xi_comb = Su2TtTangent::project(&base_l, &base_r, &comb_dir);
    let lin = xi1.scale(a).add(&xi2.scale(b));
    let diff2 = xi_comb.sub(&lin);
    assert!(
        diff2.norm() <= 1e-11 * lin.norm().max(1.0),
        "project not linear: ‖diff‖={} ‖lin‖={}",
        diff2.norm(),
        lin.norm()
    );

    assert!(xi1.norm() > 1e-6, "vacuous tangent vector");
    assert!(pg.order() > 1);
}

#[test]
fn spg5_y3_to_vector_and_retraction_roundtrip_at_nontrivial_group() {
    let d = 6;
    let pg = c2v_table(d);
    let (n, two_s) = (4usize, 2usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: no nonidentity reachable target");
    let mut x = build_pg_state::<f64>(d, n, two_s, pg, target, 1, 0xC5_0301);
    normalize_mut(&mut x);
    let (base_l, base_r) = tangent_bases(&x);

    let xi_x = Su2TtTangent::project(&base_l, &base_r, &x);
    let v = xi_x.to_vector();
    assert_expand_matches(&v, &x, 1e-12);

    let v0 = xi_x.retract_hosvd(0.0);
    assert_expand_matches(&v0, &x, 1e-12);

    let mut rng = seeded(0xC5_0302);
    let dir: Su2TtVector<f64> = randomized_su2_state(x.space, 1, 2, &mut rng);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);
    let xi = xi.scale(1.0 / xi.norm());
    let iota = xi.to_vector();
    let mut errs = Vec::new();
    for &alpha in &[1e-1_f64, 1e-2_f64] {
        let xa = xi.retract_hosvd(alpha);
        let lin = &x + &(&iota * alpha);
        let diff = &xa - &lin;
        errs.push(diff.norm());
    }
    assert!(
        errs[0] > 1e-9 && errs[1] > 1e-9,
        "vacuous O(α²) probe (still at the roundoff floor): errs={errs:?}"
    );
    let ratio = errs[0] / errs[1];
    assert!(
        (60.0..=160.0).contains(&ratio),
        "second-order retraction ratio {ratio} not ~100 (errs={errs:?})"
    );

    assert!(v.check());
    assert!(v0.check());
    assert_expand_purity_chain(&v, &pg);
    assert_expand_purity_chain(&v0, &pg);

    assert!(pg.order() > 1);
}

#[test]
fn spg5_y4_gauge_condition_and_tangent_metric_identity() {
    let d = 4;
    let pg = c2v_table(d);
    let (n, two_s) = (3usize, 1usize);
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture bug: no nonidentity reachable target");
    let x = build_pg_state::<f64>(d, n, two_s, pg, target, 2, 0xC5_0401);
    let (base_l, base_r) = tangent_bases(&x);

    let mut rng = seeded(0xC5_0402);
    let dir: Su2TtVector<f64> = randomized_su2_state(x.space, 1, 2, &mut rng);
    let xi = Su2TtTangent::project(&base_l, &base_r, &dir);

    for k in 0..d - 1 {
        for s in x.space.bond_sectors(k + 1) {
            let l = base_l.core(k).vertical_unfolding(s);
            let c = xi.component(k).vertical_unfolding(s);
            let n2 = gauge_residual_norm(l.as_ref(), c.as_ref());
            assert!(
                n2 < 1e-11,
                "gauge residual at bond {k} sector {s:?}: {n2:.3e}"
            );
        }
    }

    let a = xi.norm();
    let b = xi.to_vector().norm_lr(false);
    assert!(
        (a - b).abs() <= 1e-11 * a.max(1.0),
        "centerweighted tangent norm {a} vs embedded LR norm {b}"
    );

    assert!(pg.order() > 1);
    assert!(xi.norm() > 1e-6, "vacuous tangent direction");
}
