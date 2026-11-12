// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::*;
use crate::pointgroup::{Irrep, PointGroup};
use crate::su2::core::Family;

mod api {
    use crate::pointgroup::Irrep;
    use crate::su2::core::Family;
    use crate::su2::qn::{
        PathCountGrid, StateSym, Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space, SymGrid,
        SymLabel, SymSpace, SymStep, Symmetry, derived_pathcounts,
    };

    pub fn two_j(l: Su2Sector) -> usize {
        <Su2Sector as SymLabel>::two_j(l)
    }
    pub fn sharing_key(l: Su2Sector) -> usize {
        <Su2Sector as SymLabel>::sharing_key(l)
    }

    pub fn steps() -> &'static [Family] {
        <StateSym as Symmetry>::STEPS
    }
    pub fn step_all() -> &'static [Family] {
        <Family as SymStep>::ALL
    }
    pub fn index(f: Family) -> usize {
        <Family as SymStep>::index(f)
    }
    pub fn step_apply(f: Family, l: Su2Sector, ctx: Irrep) -> Option<Su2Sector> {
        <Family as SymStep>::step_apply(f, l, ctx)
    }
    pub fn step_unapply(f: Family, t: Su2Sector, ctx: Irrep) -> Option<Su2Sector> {
        <Family as SymStep>::step_unapply(f, t, ctx)
    }
    pub fn site_two_j(f: Family) -> usize {
        <Family as SymStep>::site_two_j(f)
    }
    pub fn odd(f: Family) -> bool {
        <Family as SymStep>::odd(f)
    }
    pub fn chain_factor(f: Family) -> usize {
        <Family as SymStep>::chain_factor(f)
    }

    pub fn sites(sp: &Su2Space) -> usize {
        <Su2Space as SymSpace>::sites(sp)
    }
    pub fn total_two_j(sp: &Su2Space) -> usize {
        <Su2Space as SymSpace>::total_two_j(sp)
    }
    pub fn site_ctx(sp: &Su2Space, k: usize) -> Irrep {
        <Su2Space as SymSpace>::site_ctx(sp, k)
    }
    pub fn admissible(sp: &Su2Space, b: usize, l: Su2Sector) -> bool {
        <Su2Space as SymSpace>::admissible(sp, b, l)
    }
    pub fn in_grid(sp: &Su2Space, l: Su2Sector) -> bool {
        <Su2Space as SymSpace>::in_grid(sp, l)
    }
    pub fn bond_labels(sp: &Su2Space, b: usize) -> Vec<Su2Sector> {
        <Su2Space as SymSpace>::bond_labels(sp, b)
    }
    pub fn root(sp: &Su2Space) -> Su2Sector {
        <Su2Space as SymSpace>::root(sp)
    }
    pub fn top(sp: &Su2Space) -> Su2Sector {
        <Su2Space as SymSpace>::top(sp)
    }
    pub fn chain_total(sp: &Su2Space) -> usize {
        <Su2Space as SymSpace>::chain_total(sp)
    }
    pub fn all_admissible_reachable(sp: &Su2Space) -> bool {
        <Su2Space as SymSpace>::all_admissible_reachable(sp)
    }
    pub fn same_lattice(a: &Su2Space, b: &Su2Space) -> bool {
        <Su2Space as SymSpace>::same_lattice(a, b)
    }

    pub fn grid_from_fn<V>(sp: &Su2Space, f: impl FnMut(Su2Sector) -> V) -> Su2SectorGrid<V> {
        <Su2SectorGrid<V> as SymGrid<V>>::from_fn(sp, f)
    }
    pub fn grid_filled<V: Clone>(sp: &Su2Space, v: V) -> Su2SectorGrid<V> {
        <Su2SectorGrid<V> as SymGrid<V>>::filled(sp, v)
    }
    pub fn grid_labels<V>(g: &Su2SectorGrid<V>) -> Vec<Su2Sector> {
        <Su2SectorGrid<V> as SymGrid<V>>::iter(g)
            .map(|(l, _)| l)
            .collect()
    }
    pub fn grid_iter_values(g: &Su2SectorGrid<usize>) -> Vec<usize> {
        <Su2SectorGrid<usize> as SymGrid<usize>>::iter(g)
            .map(|(_, &v)| v)
            .collect()
    }
    pub fn grid_iter_mut_labels(g: &mut Su2SectorGrid<usize>) -> Vec<Su2Sector> {
        <Su2SectorGrid<usize> as SymGrid<usize>>::iter_mut(g)
            .map(|(l, _)| l)
            .collect()
    }
    pub fn grid_values(g: &Su2SectorGrid<usize>) -> &[usize] {
        <Su2SectorGrid<usize> as SymGrid<usize>>::values(g)
    }
    pub fn grid_values_mut(g: &mut Su2SectorGrid<usize>) -> &mut [usize] {
        <Su2SectorGrid<usize> as SymGrid<usize>>::values_mut(g)
    }
    pub fn grid_label_at<V>(g: &Su2SectorGrid<V>, i: usize) -> Su2Sector {
        <Su2SectorGrid<V> as SymGrid<V>>::label_at(g, i)
    }
    pub fn grid_in_bounds<V>(g: &Su2SectorGrid<V>, l: Su2Sector) -> bool {
        <Su2SectorGrid<V> as SymGrid<V>>::in_bounds(g, l)
    }
    pub fn grid_hash<H: std::hash::Hasher>(g: &Su2RankGrid, h: &mut H) {
        <Su2RankGrid as SymGrid<usize>>::hash_grid(g, h)
    }
    pub fn grid_eq(a: &Su2RankGrid, b: &Su2RankGrid) -> bool {
        <Su2RankGrid as SymGrid<usize>>::eq_grid(a, b)
    }

    pub fn pathcounts(sp: &Su2Space) -> (Vec<PathCountGrid>, Vec<PathCountGrid>) {
        <StateSym as Symmetry>::pathcounts(sp)
    }
    pub fn derived(sp: &Su2Space) -> (Vec<PathCountGrid>, Vec<PathCountGrid>) {
        derived_pathcounts::<StateSym>(sp)
    }
    pub fn max_ranks(sp: &Su2Space) -> Vec<Su2RankGrid> {
        <StateSym as Symmetry>::max_ranks(sp)
    }
    pub fn max_ranks_open(sp: &Su2Space, rho: usize) -> Vec<Su2RankGrid> {
        <StateSym as Symmetry>::max_ranks_open(sp, rho)
    }
    pub fn theta_sq(l: Su2Sector, t: Su2Sector) -> f64 {
        <StateSym as Symmetry>::theta_sq(l, t)
    }
    pub fn boundary_step(sp: &Su2Space, q: Su2Sector) -> Family {
        <StateSym as Symmetry>::boundary_step(sp, q)
    }
}

fn cyclic_pg(nbits: u8, d: usize) -> PointGroup {
    let order = 1u8 << nbits;
    let irreps: Vec<Irrep> = (0..d).map(|k| Irrep::new((k as u8) % order)).collect();
    PointGroup::new(nbits, &irreps)
}

fn reachable_nontrivial_target(pg: PointGroup, d: usize, n: usize, two_s: usize) -> Irrep {
    (1..pg.order())
        .map(Irrep::new)
        .find(|&ir| {
            let sp = Su2Space::with_point_group(d, n, two_s, pg, ir);
            max_ranks(&sp)[d][Su2Sector::with_irrep(n, two_s, ir)] > 0
        })
        .expect("a reachable nontrivial target exists for this fixture")
}

fn spaces() -> Vec<Su2Space> {
    let mut v = vec![
        Su2Space::new(3, 2, 0),
        Su2Space::new(4, 3, 1),
        Su2Space::new(5, 3, 1),
        Su2Space::new(5, 4, 2),
        Su2Space::new(6, 4, 2),
        Su2Space::new(6, 6, 0),
        Su2Space::new(6, 5, 1),
        Su2Space::new(7, 5, 3),
        Su2Space::new(5, 8, 0),
    ];
    let pg2 = cyclic_pg(1, 5);
    v.push(Su2Space::with_point_group(
        5,
        3,
        1,
        pg2,
        reachable_nontrivial_target(pg2, 5, 3, 1),
    ));
    let pg4 = cyclic_pg(2, 6);
    v.push(Su2Space::with_point_group(
        6,
        4,
        0,
        pg4,
        reachable_nontrivial_target(pg4, 6, 4, 0),
    ));
    v.push(Su2Space::with_point_group(6, 4, 2, pg4, Irrep::IDENTITY));
    v
}

fn storage_nest(sp: &Su2Space) -> Vec<Su2Sector> {
    let mut out = Vec::new();
    for ir in 0..sp.pg.order() {
        for n in 0..=sp.n {
            for two_s in 0..=sp.n {
                out.push(Su2Sector::with_irrep(n, two_s, Irrep::new(ir)));
            }
        }
    }
    out
}

#[test]
fn sym_steps_are_family_all_in_physical_order() {
    assert_eq!(
        api::steps(),
        &Family::ALL[..],
        "StateSym::STEPS must be Family::ALL"
    );
    assert_eq!(
        api::step_all(),
        &Family::ALL[..],
        "<Family as SymStep>::ALL must be Family::ALL"
    );
    for (i, &f) in Family::ALL.iter().enumerate() {
        assert_eq!(api::index(f), i, "{f:?}: index must be the physical index");
        assert_eq!(
            api::index(f),
            f as usize,
            "{f:?}: index must be the discriminant"
        );
        assert_eq!(
            api::steps()[api::index(f)],
            f,
            "{f:?}: STEPS[index(f)] != f"
        );
        assert_eq!(
            api::chain_factor(f),
            i,
            "{f:?}: chain_factor must equal the step index"
        );
        assert!(
            api::chain_factor(f) < 4,
            "{f:?}: chain_factor must address one of ω₁..ω₄"
        );
    }
}

#[test]
fn sym_step_site_spin_and_parity_tables() {
    let site_two_j: Vec<usize> = Family::ALL.iter().map(|&f| api::site_two_j(f)).collect();
    assert_eq!(
        site_two_j,
        vec![0, 1, 1, 0],
        "site_two_j over Empty, Up, Dn, Occ"
    );
    let odd: Vec<bool> = Family::ALL.iter().map(|&f| api::odd(f)).collect();
    assert_eq!(
        odd,
        vec![false, true, true, false],
        "odd over Empty, Up, Dn, Occ"
    );
    for f in Family::ALL {
        assert_eq!(api::odd(f), f.delta().0 % 2 == 1, "{f:?}: odd != Δn odd");
        assert_eq!(
            api::site_two_j(f),
            usize::from(api::odd(f)),
            "{f:?}: site 2j vs parity"
        );
    }
}

#[test]
fn sym_step_apply_unapply_round_trip_every_admissible_label() {
    let (mut fwd, mut bwd, mut none_fwd, mut none_bwd) = (0usize, 0usize, 0usize, 0usize);
    for sp in spaces() {
        for k in 0..sp.d {
            let g = sp.pg.irrep_of(k);
            let ctx = api::site_ctx(&sp, k);
            assert_eq!(ctx, g, "{sp:?} site {k}: site_ctx must be pg.irrep_of(k)");
            for l in sp.bond_sectors(k) {
                for f in Family::ALL {
                    let t = api::step_apply(f, l, ctx);
                    assert_eq!(t, f.apply(l, g), "{sp:?} k={k} {f:?} {l:?}: step_apply");
                    match t {
                        None => none_fwd += 1,
                        Some(t) if sp.contains(k + 1, t) => {
                            assert_eq!(
                                api::step_unapply(f, t, ctx),
                                Some(l),
                                "{sp:?} k={k} {f:?}: unapply(apply({l:?})) != {l:?}"
                            );
                            fwd += 1;
                        }
                        Some(_) => {}
                    }
                }
            }
            for t in sp.bond_sectors(k + 1) {
                for f in Family::ALL {
                    let p = api::step_unapply(f, t, ctx);
                    assert_eq!(p, f.unapply(t, g), "{sp:?} k={k} {f:?} {t:?}: step_unapply");
                    match p {
                        None => none_bwd += 1,
                        Some(p) => {
                            assert_eq!(
                                api::step_apply(f, p, ctx),
                                Some(t),
                                "{sp:?} k={k} {f:?}: apply(unapply({t:?})) != {t:?}"
                            );
                            bwd += 1;
                        }
                    }
                }
            }
        }
    }
    println!("round trips: fwd {fwd}, bwd {bwd}; None: fwd {none_fwd}, bwd {none_bwd}");
    assert!(
        fwd > 0 && bwd > 0,
        "anti-vacuity: no round trip was exercised"
    );
    assert!(
        none_fwd > 0,
        "anti-vacuity: the Dn spin-underflow None was never hit"
    );
    assert!(none_bwd > 0, "anti-vacuity: no backward None was hit");
}

#[test]
fn sym_step_none_cases_and_the_off_grid_dn_adjoint() {
    let g = Irrep::IDENTITY;
    assert_eq!(api::step_apply(Family::Dn, Su2Sector::new(2, 0), g), None);
    assert_eq!(
        api::step_apply(Family::Dn, Su2Sector::new(2, 2), g),
        Some(Su2Sector::new(3, 1))
    );
    assert_eq!(api::step_unapply(Family::Up, Su2Sector::new(2, 0), g), None);
    assert_eq!(
        api::step_unapply(Family::Occ, Su2Sector::new(1, 1), g),
        None
    );
    assert_eq!(api::step_unapply(Family::Up, Su2Sector::new(0, 0), g), None);
    assert_eq!(api::step_unapply(Family::Dn, Su2Sector::new(0, 1), g), None);
    assert_eq!(
        api::step_unapply(Family::Dn, Su2Sector::new(3, 1), g),
        Some(Su2Sector::new(2, 2)),
        "the Dn adjoint must INCREMENT 2s"
    );
    let sp = Su2Space::new(6, 4, 2);
    let t = Su2Sector::new(4, 4);
    let p = api::step_unapply(Family::Dn, t, g).expect("arithmetic-only unapply");
    assert_eq!(p, Su2Sector::new(3, 5), "Dn adjoint of (4, 4)");
    assert!(
        p.two_s > sp.n,
        "fixture: the source must leave the (N+1)² grid"
    );
    let grid = api::grid_filled(&sp, 0usize);
    assert!(
        !api::grid_in_bounds(&grid, p),
        "in_bounds must reject the off-grid Dn source"
    );
    assert!(
        api::grid_in_bounds(&grid, t),
        "in_bounds must accept the on-grid target"
    );
    let x = Irrep::new(3);
    let l = Su2Sector::with_irrep(1, 1, Irrep::new(1));
    for f in Family::ALL {
        let t = api::step_apply(f, l, x).unwrap();
        let want = if api::odd(f) {
            Irrep::new(1).fuse(x)
        } else {
            Irrep::new(1)
        };
        assert_eq!(t.irrep, want, "{f:?}: forward irrep flux");
        assert_eq!(
            api::step_unapply(f, t, x),
            Some(l),
            "{f:?}: backward flux at |G| > 1"
        );
    }
}

#[test]
fn sym_space_and_label_scalars_match_the_concrete_fields() {
    for sp in spaces() {
        assert_eq!(api::sites(&sp), sp.d, "{sp:?}: sites");
        assert_eq!(api::total_two_j(&sp), sp.two_s, "{sp:?}: total_two_j");
        assert_eq!(
            api::chain_total(&sp),
            sp.n,
            "{sp:?}: chain_total (R = N, n_total)"
        );
        assert_eq!(api::root(&sp), Su2Sector::new(0, 0), "{sp:?}: root");
        assert_eq!(
            api::top(&sp),
            Su2Sector::with_irrep(sp.n, sp.two_s, sp.target),
            "{sp:?}: top"
        );
        assert_eq!(
            api::all_admissible_reachable(&sp),
            sp.pg.is_trivial(),
            "{sp:?}: the |G| = 1 DP-skip predicate"
        );
        for k in 0..sp.d {
            assert_eq!(
                api::site_ctx(&sp, k),
                sp.pg.irrep_of(k),
                "{sp:?}: site_ctx({k})"
            );
        }
        for l in storage_nest(&sp) {
            assert_eq!(api::two_j(l), l.two_s, "{l:?}: two_j");
            assert_eq!(
                api::sharing_key(l),
                l.n,
                "{l:?}: sharing_key is n (pre-Sharing::class)"
            );
        }
        assert!(api::same_lattice(&sp, &sp), "{sp:?}: reflexive");
        assert_eq!(
            api::same_lattice(&sp, &sp),
            sp.same_orbital_lattice(&sp),
            "{sp:?}: same_lattice"
        );
    }
    let a = Su2Space::new(6, 4, 0);
    assert!(api::same_lattice(&a, &a.with_two_s(2)), "two_s-blind");
    assert!(!api::same_lattice(&a, &Su2Space::new(6, 2, 0)), "n-exact");
    assert!(!api::same_lattice(&a, &Su2Space::new(5, 4, 0)), "d-exact");
    let pg = cyclic_pg(2, 6);
    let t1 = reachable_nontrivial_target(pg, 6, 4, 0);
    let b0 = Su2Space::with_point_group(6, 4, 0, pg, Irrep::IDENTITY);
    let b1 = Su2Space::with_point_group(6, 4, 0, pg, t1);
    assert!(api::same_lattice(&b0, &b1), "target-blind");
    assert!(!api::same_lattice(&a, &b0), "pg-exact");
}

#[test]
fn sym_bond_labels_equal_bond_sectors_in_order() {
    let mut total = 0usize;
    for sp in spaces() {
        for b in 0..=sp.d {
            let got = api::bond_labels(&sp, b);
            assert_eq!(
                got,
                sp.bond_sectors(b),
                "{sp:?} bond {b}: bond_labels order"
            );
            total += got.len();
        }
    }
    assert!(total > 0, "anti-vacuity");
    let sp = Su2Space::new(3, 2, 0);
    assert_eq!(
        api::bond_labels(&sp, 1),
        vec![
            Su2Sector::new(0, 0),
            Su2Sector::new(2, 0),
            Su2Sector::new(1, 1)
        ],
        "(3,2,0) bond 1: n fastest inside 2s"
    );
}

#[test]
fn sym_admissible_equals_contains_on_the_full_grid() {
    let (mut yes, mut no) = (0usize, 0usize);
    for sp in spaces() {
        for b in 0..=sp.d + 1 {
            for ir in 0..=sp.pg.order() {
                for n in 0..=sp.n + 1 {
                    for two_s in 0..=sp.n + 1 {
                        let l = Su2Sector::with_irrep(n, two_s, Irrep::new(ir));
                        let want = sp.contains(b, l);
                        assert_eq!(
                            api::admissible(&sp, b, l),
                            want,
                            "{sp:?} bond {b} {l:?}: admissible != contains"
                        );
                        if want {
                            yes += 1;
                        } else {
                            no += 1;
                        }
                    }
                }
            }
        }
        assert!(!api::admissible(
            &sp,
            sp.d + 1,
            Su2Sector::new(sp.n, sp.two_s)
        ));
        assert!(!api::admissible(
            &sp,
            0,
            Su2Sector::with_irrep(0, 0, Irrep::new(sp.pg.order()))
        ));
    }
    assert!(
        yes > 0 && no > 0,
        "anti-vacuity: {yes} admissible, {no} not"
    );
}

#[test]
fn sym_in_grid_is_the_in_bounds_of_the_spaces_grid() {
    let (mut on, mut off) = (0usize, 0usize);
    for sp in spaces() {
        let grid = api::grid_filled(&sp, 0usize);
        for ir in 0..=sp.pg.order() {
            for n in 0..=sp.n + 1 {
                for two_s in 0..=sp.n + 1 {
                    let l = Su2Sector::with_irrep(n, two_s, Irrep::new(ir));
                    let want = api::grid_in_bounds(&grid, l);
                    assert_eq!(
                        api::in_grid(&sp, l),
                        want,
                        "{sp:?} {l:?}: in_grid != in_bounds"
                    );
                    if want {
                        on += 1;
                    } else {
                        off += 1;
                    }
                }
            }
        }
    }
    assert!(
        on > 0 && off > 0,
        "anti-vacuity: {on} on the grid, {off} off"
    );
    let sp = Su2Space::new(6, 4, 2);
    let p = api::step_unapply(Family::Dn, Su2Sector::new(4, 4), Irrep::IDENTITY).unwrap();
    assert!(
        !api::in_grid(&sp, p),
        "the Dn adjoint of (4, 4) at N = 4 is off the grid"
    );
}

#[test]
fn sym_grid_storage_order_and_label_inverse() {
    for sp in spaces() {
        let nest = storage_nest(&sp);
        let mut visit = Vec::new();
        let mut i = 0usize;
        let mut g = api::grid_from_fn(&sp, |l| {
            visit.push(l);
            i += 1;
            i - 1
        });
        assert_eq!(visit, nest, "{sp:?}: from_fn visit order");
        let mut inherent = Vec::new();
        let _ = Su2SectorGrid::from_fn(&sp, |l| inherent.push(l));
        assert_eq!(visit, inherent, "{sp:?}: trait vs inherent from_fn order");
        assert_eq!(api::grid_labels(&g), nest, "{sp:?}: iter order");
        assert_eq!(
            api::grid_iter_values(&g),
            (0..nest.len()).collect::<Vec<_>>(),
            "{sp:?}: iter values"
        );
        assert_eq!(
            api::grid_values(&g),
            &(0..nest.len()).collect::<Vec<_>>()[..],
            "{sp:?}: values() is storage order"
        );
        for (idx, &l) in nest.iter().enumerate() {
            assert_eq!(api::grid_label_at(&g, idx), l, "{sp:?}: label_at({idx})");
            assert_eq!(g[l], idx, "{sp:?}: Index(label_at(i)) == i");
            assert_eq!(
                api::grid_label_at(&g, idx),
                g.sector_at(idx),
                "label_at == sector_at"
            );
            assert!(
                api::grid_in_bounds(&g, l),
                "{sp:?} {l:?}: in_bounds on the grid"
            );
        }
        assert_eq!(
            api::grid_iter_mut_labels(&mut g),
            nest,
            "{sp:?}: iter_mut order"
        );
        api::grid_values_mut(&mut g)[0] = usize::MAX;
        assert_eq!(g[nest[0]], usize::MAX, "values_mut writes storage slot 0");
        let top = sp.n + 1;
        assert!(!api::grid_in_bounds(&g, Su2Sector::new(top, 0)), "n = N+1");
        assert!(!api::grid_in_bounds(&g, Su2Sector::new(0, top)), "2s = N+1");
        assert!(
            !api::grid_in_bounds(&g, Su2Sector::with_irrep(0, 0, Irrep::new(sp.pg.order()))),
            "irrep = |G|"
        );
        let f = api::grid_filled(&sp, 7u64);
        assert_eq!(f, Su2SectorGrid::filled(&sp, 7u64), "{sp:?}: filled");
    }
}

#[test]
fn sym_grid_hash_and_eq_forward_to_the_derives() {
    let sp = Su2Space::new(6, 4, 2);
    let a = Su2SectorGrid::from_fn(&sp, |l| l.n * 10 + l.two_s);
    let mut b = a.clone();
    let (mut h1, mut h2) = (DefaultHasher::new(), DefaultHasher::new());
    api::grid_hash(&a, &mut h1);
    a.hash(&mut h2);
    assert_eq!(
        h1.finish(),
        h2.finish(),
        "hash_grid must be the derived Hash"
    );
    assert!(api::grid_eq(&a, &b) && a == b);
    b[Su2Sector::new(2, 2)] += 1;
    assert!(!api::grid_eq(&a, &b) && a != b);
    let c = api::grid_filled(&Su2Space::new(6, 2, 0), 0usize);
    let e = api::grid_filled(&Su2Space::new(6, 4, 0), 0usize);
    assert_eq!(api::grid_eq(&c, &e), c == e);
    assert!(!api::grid_eq(&c, &e));
}

#[test]
fn sym_pathcounts_delegate_and_derived_body_agree() {
    for sp in spaces() {
        let (l0, r0) = pathcounts(&sp);
        let (l1, r1) = api::pathcounts(&sp);
        assert_eq!(l1, l0, "{sp:?}: StateSym::pathcounts L");
        assert_eq!(r1, r0, "{sp:?}: StateSym::pathcounts R");
        let (l2, r2) = api::derived(&sp);
        assert_eq!(l2, l0, "{sp:?}: derived_pathcounts L");
        assert_eq!(r2, r0, "{sp:?}: derived_pathcounts R");
        assert_eq!(l0.len(), sp.d + 1);
        assert_eq!(
            l0[sp.d][api::top(&sp)],
            r0[0][api::root(&sp)],
            "{sp:?}: L_d(top) = R_0(root)"
        );
        assert!(
            r0[0][api::root(&sp)] > 0,
            "{sp:?}: anti-vacuity, the space has a path"
        );
    }
}

#[test]
fn sym_max_ranks_are_the_min_of_the_pathcounts() {
    for sp in spaces() {
        let (l, r) = pathcounts(&sp);
        let got = api::max_ranks(&sp);
        assert_eq!(got, max_ranks(&sp), "{sp:?}: max_ranks");
        for b in 0..=sp.d {
            for s in storage_nest(&sp) {
                assert_eq!(got[b][s] as u64, l[b][s].min(r[b][s]), "{sp:?} b={b} {s:?}");
            }
        }
        for rho in 1..=3 {
            let o = api::max_ranks_open(&sp, rho);
            assert_eq!(o, max_ranks_open(&sp, rho), "{sp:?}: max_ranks_open({rho})");
            for b in 0..=sp.d {
                for s in storage_nest(&sp) {
                    assert_eq!(
                        o[b][s] as u64,
                        l[b][s].min(r[b][s].saturating_mul(rho as u64)),
                        "{sp:?} rho={rho} b={b} {s:?}"
                    );
                }
            }
        }
        assert_eq!(
            api::max_ranks_open(&sp, 1),
            got,
            "{sp:?}: open(1) == closed"
        );
    }
}

#[test]
fn sym_theta_sq_is_bitwise_the_family_closed_form() {
    let mut n = 0usize;
    for two_s in 0..=12usize {
        for ir in 0..4u8 {
            let l = Su2Sector::with_irrep(two_s + 2, two_s, Irrep::new(ir));
            for f in Family::ALL {
                let Some(t) = f.apply(l, Irrep::new(1)) else {
                    continue;
                };
                let got = api::theta_sq(l, t);
                let want = crate::su2::coreops::theta_sq(f, l);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{f:?} 2s={two_s}: theta_sq {got:e} != closed form {want:e}"
                );
                n += 1;
            }
        }
    }
    assert!(n > 0);
}

#[test]
fn sym_boundary_step_matches_family_of_delta_irrep_blind() {
    let pg = cyclic_pg(2, 6);
    for sp in [
        Su2Space::new(6, 4, 2),
        Su2Space::with_point_group(6, 4, 0, pg, reachable_nontrivial_target(pg, 6, 4, 0)),
        Su2Space::with_point_group(6, 4, 2, pg, Irrep::IDENTITY),
    ] {
        let mut contained_unreachable = 0usize;
        for ir in 0..sp.pg.order() {
            let x = Irrep::new(ir);
            for (n, two_s, want) in [
                (0, 0, Family::Empty),
                (1, 1, Family::Up),
                (2, 0, Family::Occ),
            ] {
                let q = Su2Sector::with_irrep(n, two_s, x);
                assert_eq!(
                    api::boundary_step(&sp, q),
                    want,
                    "{sp:?}: boundary_step({q:?})"
                );
                if sp.contains(1, q) && max_ranks(&sp)[1][q] == 0 {
                    contained_unreachable += 1;
                }
            }
        }
        if !sp.pg.is_trivial() {
            assert!(
                contained_unreachable > 0,
                "{sp:?}: anti-vacuity — no contained-but-unreachable bond-1 label was exercised"
            );
        }
    }
}

#[test]
#[should_panic]
fn sym_boundary_step_panics_off_the_bond1_labels() {
    let sp = Su2Space::new(6, 4, 2);
    let _ = api::boundary_step(&sp, Su2Sector::new(3, 1));
}
