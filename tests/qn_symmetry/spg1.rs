// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::core::Family;
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks, pathcounts};
use qntt::{Irrep, PointGroup, PointGroupKind};
use std::collections::{BTreeSet, HashMap};

fn family_deltas() -> [(usize, i64, bool); 4] {
    let mut out = [(0usize, 0i64, false); 4];
    for (i, f) in Family::ALL.into_iter().enumerate() {
        let (dn, dt) = f.delta();
        out[i] = (dn, dt, dn % 2 == 1);
    }
    out
}

fn admissible_nt(n_total: usize, two_s_total: usize, d: usize) -> Vec<BTreeSet<(usize, usize)>> {
    let deltas: [(i64, i64); 4] = [(0, 0), (1, 1), (1, -1), (2, 0)];
    let mut f: Vec<BTreeSet<(usize, usize)>> = vec![BTreeSet::new(); d + 1];
    f[0].insert((0, 0));
    for k in 0..d {
        let cur: Vec<(usize, usize)> = f[k].iter().copied().collect();
        for (nn, tt) in cur {
            for (dn, dt) in deltas {
                let n2 = nn as i64 + dn;
                let t2 = tt as i64 + dt;
                if t2 < 0 || n2 > n_total as i64 || t2 > n2 {
                    continue;
                }
                f[k + 1].insert((n2 as usize, t2 as usize));
            }
        }
    }
    let mut b: Vec<BTreeSet<(usize, usize)>> = vec![BTreeSet::new(); d + 1];
    b[d].insert((n_total, two_s_total));
    for k in (0..d).rev() {
        let cur: Vec<(usize, usize)> = b[k + 1].iter().copied().collect();
        for (nn, tt) in cur {
            for (dn, dt) in deltas {
                let pn = nn as i64 - dn;
                let pt = tt as i64 - dt;
                if pn < 0 || pt < 0 || pt > pn {
                    continue;
                }
                b[k].insert((pn as usize, pt as usize));
            }
        }
    }
    (0..=d)
        .map(|k| f[k].intersection(&b[k]).copied().collect())
        .collect()
}

type IrGrid = HashMap<(usize, usize, u8), u64>;

fn brute_force_walks(
    pg: PointGroup,
    n: usize,
    two_s: usize,
    target: Irrep,
    d: usize,
) -> (Vec<IrGrid>, Vec<IrGrid>) {
    let admissible = admissible_nt(n, two_s, d);
    let deltas = family_deltas();

    let mut l: Vec<IrGrid> = vec![HashMap::new(); d + 1];
    l[0].insert((0, 0, 0u8), 1);
    for b in 0..d {
        let g = pg.irrep_of(b).index();
        let cur: Vec<((usize, usize, u8), u64)> = l[b].iter().map(|(&k, &c)| (k, c)).collect();
        for ((nn, tt, ir), c) in cur {
            for (dn, dt, odd) in deltas {
                let n2 = nn as i64 + dn as i64;
                let t2 = tt as i64 + dt;
                if t2 < 0 {
                    continue;
                }
                let key_nt = (n2 as usize, t2 as usize);
                if !admissible[b + 1].contains(&key_nt) {
                    continue;
                }
                let ir2 = if odd { ir ^ g } else { ir };
                *l[b + 1].entry((key_nt.0, key_nt.1, ir2)).or_insert(0) += c;
            }
        }
    }

    let mut r: Vec<IrGrid> = vec![HashMap::new(); d + 1];
    r[d].insert((n, two_s, target.index()), 1);
    for b in (0..d).rev() {
        let g = pg.irrep_of(b).index();
        let cur: Vec<((usize, usize, u8), u64)> = r[b + 1].iter().map(|(&k, &c)| (k, c)).collect();
        for ((nn, tt, ir), c) in cur {
            for (dn, dt, odd) in deltas {
                let pn = nn as i64 - dn as i64;
                let pt = tt as i64 - dt;
                if pn < 0 || pt < 0 {
                    continue;
                }
                let key_nt = (pn as usize, pt as usize);
                if !admissible[b].contains(&key_nt) {
                    continue;
                }
                let ir_parent = if odd { ir ^ g } else { ir };
                *r[b].entry((key_nt.0, key_nt.1, ir_parent)).or_insert(0) += c;
            }
        }
    }
    (l, r)
}

fn find_nonidentity_reachable_target(
    pg: PointGroup,
    n: usize,
    two_s: usize,
    d: usize,
) -> Option<Irrep> {
    let (l, _r) = brute_force_walks(pg, n, two_s, Irrep::IDENTITY, d);
    (1..pg.order())
        .map(Irrep::new)
        .find(|&ir| l[d].get(&(n, two_s, ir.index())).copied().unwrap_or(0) > 0)
}

fn check_pathcounts_case(pg: PointGroup, d: usize, n: usize, two_s: usize, target: Irrep) {
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let (l_dp, r_dp) = pathcounts(&space);
    let caps = max_ranks(&space);
    let (l_brute, r_brute) = brute_force_walks(pg, n, two_s, target, d);

    let mut any_reachable = false;
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            let key = (s.n, s.two_s, s.irrep.index());
            let lb = l_brute[b].get(&key).copied().unwrap_or(0);
            let rb = r_brute[b].get(&key).copied().unwrap_or(0);
            assert_eq!(
                l_dp[b][s], lb,
                "L mismatch d={d} n={n} two_s={two_s} target={target:?} b={b} s={s:?}"
            );
            assert_eq!(
                r_dp[b][s], rb,
                "R mismatch d={d} n={n} two_s={two_s} target={target:?} b={b} s={s:?}"
            );
            let expected_cap = lb.min(rb);
            assert_eq!(
                caps[b][s] as u64, expected_cap,
                "cap mismatch d={d} n={n} two_s={two_s} target={target:?} b={b} s={s:?}"
            );
            assert_eq!(
                caps[b][s] as u64,
                l_dp[b][s].min(r_dp[b][s]),
                "internal min(L,R) inconsistency b={b} s={s:?}"
            );
            if expected_cap > 0 {
                any_reachable = true;
            }
        }
    }
    assert!(
        any_reachable,
        "vacuous case: nothing reachable d={d} n={n} two_s={two_s} target={target:?}"
    );
}

#[test]
fn spg1_pathcounts_match_brute_force_genealogical_oracle() {
    let pg = PointGroup::from_orbsym(PointGroupKind::Cs, &[1, 2, 1]).expect("valid orbsym");
    let target =
        find_nonidentity_reachable_target(pg, 2, 0, 3).expect("Cs d=3 fixture must be non-vacuous");
    check_pathcounts_case(pg, 3, 2, 0, target);

    let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &[1, 2, 3, 4]).expect("valid orbsym");
    let target = find_nonidentity_reachable_target(pg, 2, 0, 4)
        .expect("C2v d=4 fixture must be non-vacuous");
    check_pathcounts_case(pg, 4, 2, 0, target);

    let pg2 = PointGroup::from_orbsym(PointGroupKind::C2v, &[1, 3, 2, 4]).expect("valid orbsym");
    let target2 = find_nonidentity_reachable_target(pg2, 3, 1, 4)
        .expect("C2v d=4 variant fixture must be non-vacuous");
    check_pathcounts_case(pg2, 4, 3, 1, target2);

    let d2 = PointGroup::new(
        2,
        &[
            Irrep::new(0),
            Irrep::new(1),
            Irrep::new(2),
            Irrep::new(3),
            Irrep::new(1),
        ],
    );
    let target3 =
        find_nonidentity_reachable_target(d2, 3, 1, 5).expect("D2 d=5 fixture must be non-vacuous");
    check_pathcounts_case(d2, 5, 3, 1, target3);
    let target4 = find_nonidentity_reachable_target(d2, 4, 2, 5)
        .expect("D2 d=5 high-spin fixture must be non-vacuous");
    check_pathcounts_case(d2, 5, 4, 2, target4);

    let ci =
        PointGroup::from_orbsym(PointGroupKind::Ci, &[1, 2, 1, 2, 1, 2]).expect("valid orbsym");
    let target5 =
        find_nonidentity_reachable_target(ci, 4, 0, 6).expect("Ci d=6 fixture must be non-vacuous");
    check_pathcounts_case(ci, 6, 4, 0, target5);

    let c2v6 =
        PointGroup::from_orbsym(PointGroupKind::C2v, &[1, 2, 3, 4, 2, 3]).expect("valid orbsym");
    let target6 = find_nonidentity_reachable_target(c2v6, 4, 2, 6)
        .expect("C2v d=6 fixture must be non-vacuous");
    check_pathcounts_case(c2v6, 6, 4, 2, target6);

    let all_identity = PointGroup::new(2, &[Irrep::IDENTITY; 4]);
    check_pathcounts_case(all_identity, 4, 2, 0, Irrep::IDENTITY);
}

#[test]
fn spg1_bondwise_invariant_with_point_group() {
    let d = 4;
    let (n, two_s) = (2, 0);
    let pg = PointGroup::from_orbsym(PointGroupKind::C2v, &[1, 2, 3, 4]).expect("valid orbsym");
    let target = find_nonidentity_reachable_target(pg, n, two_s, d)
        .expect("fixture must have a reachable non-identity target");
    let space = Su2Space::with_point_group(d, n, two_s, pg, target);
    let (l, r) = pathcounts(&space);
    let total = l[d][Su2Sector::with_irrep(n, two_s, target)];
    assert!(total > 0, "vacuous: target must be reachable");
    assert_eq!(r[0][Su2Sector::new(0, 0)], total);
    for k in 0..=d {
        let sum: u64 = l[k]
            .iter()
            .zip(r[k].iter())
            .map(|((_, &lv), (_, &rv))| lv * rv)
            .sum();
        assert_eq!(sum, total, "bond {k}");
    }
}
