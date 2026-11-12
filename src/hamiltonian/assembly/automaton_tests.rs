// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;

use crate::coeff::Sparse3Body;
use crate::orbital::{Orbital, Spin};
use crate::qn::QnSpace;
use rand::RngExt;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

use super::automaton::{Automaton, finalize, vertex, weighted_edge};
use super::enumerate::{enumerate_1body, enumerate_2body, enumerate_3body};
use super::{Assembled, Construction, assemble_with, collect_terms};

fn keep_applied_hd_rule(bond: usize, hd: i64, cre: &[Orbital], ann: &[Orbital]) -> bool {
    let b = bond as i64;
    let nhalf = cre.len();
    let napp = cre
        .iter()
        .chain(ann)
        .filter(|o| (o.site as i64) < b)
        .count();
    napp < nhalf || (napp == nhalf && b <= hd)
}

#[test]
fn median_switch_equals_weighted_edge() {
    let d = 6usize;
    let sites_of = |n: usize| -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut cur = vec![0usize; n];
        loop {
            out.push(cur.clone());
            let mut i = n;
            loop {
                if i == 0 {
                    return out;
                }
                i -= 1;
                cur[i] += 1;
                if cur[i] < d {
                    for j in i + 1..n {
                        cur[j] = cur[i];
                    }
                    break;
                }
            }
        }
    };
    let mut checked = 0usize;
    for n in 1..=3usize {
        let pats = sites_of(n);
        for cs in &pats {
            for as_ in &pats {
                for split in [0usize, 1] {
                    let cre: Vec<Orbital> = cs
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| Orbital::new(s, if i < split { Spin::Up } else { Spin::Dn }))
                        .collect();
                    let ann: Vec<Orbital> = as_
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| Orbital::new(s, if i < split { Spin::Dn } else { Spin::Up }))
                        .collect();
                    for hd in 0..d as i64 {
                        let bm = weighted_edge(hd, &cre, &ann);
                        for bond in 0..=d {
                            assert_eq!(
                                keep_applied_hd_rule(bond, hd, &cre, &ann),
                                bond <= bm,
                                "switch rule mismatch: n={n} cre={cs:?} ann={as_:?} hd={hd} \
                                 bond={bond} weighted_edge={bm}"
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(checked > 10_000, "vacuous sweep ({checked} checks)");
}

#[test]
fn weighted_edge_lies_in_the_balance_window() {
    let d = 6usize;
    for n in 1..=3usize {
        for c0 in 0..d {
            for c1 in c0..d {
                for a0 in 0..d {
                    for a1 in a0..d {
                        let mut cre: Vec<Orbital> = vec![Orbital::new(c0, Spin::Up)];
                        let mut ann: Vec<Orbital> = vec![Orbital::new(a0, Spin::Up)];
                        if n >= 2 {
                            cre.push(Orbital::new(c1, Spin::Dn));
                            ann.push(Orbital::new(a1, Spin::Dn));
                        }
                        if n == 3 {
                            cre.push(Orbital::new(c0, Spin::Dn));
                            ann.push(Orbital::new(a1, Spin::Up));
                        }
                        let mut sites: Vec<usize> =
                            cre.iter().chain(&ann).map(|o| o.site).collect();
                        sites.sort_unstable();
                        for hd in 0..d as i64 {
                            let bm = weighted_edge(hd, &cre, &ann);
                            assert!(
                                sites[n - 1] <= bm && bm <= sites[n],
                                "weighted edge {bm} outside the balance window \
                                 [{}, {}] (n={n})",
                                sites[n - 1],
                                sites[n]
                            );
                            for bond in 0..=d {
                                let (_, open) = vertex(
                                    &crate::pointgroup::PointGroup::default(),
                                    bond,
                                    bm,
                                    &cre,
                                    &ann,
                                );
                                assert!(
                                    open[3..].iter().all(|o| o.is_none()),
                                    "median-switch vertex used more than 3 open slots \
                                     (bond {bond}, switch {bm})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

fn dense_random_coeffs(
    d: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> (
    crate::coeff::CoeffTensor<f64>,
    crate::coeff::CoeffTensor<f64>,
) {
    let mut t = crate::coeff::CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], rng.sample::<f64, _>(StandardNormal));
        }
    }
    let mut v = crate::coeff::CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    if v.get(&[i, j, k, l]) != 0.0 {
                        continue;
                    }
                    let x: f64 = rng.sample(StandardNormal);
                    v.set(&[i, j, k, l], x);
                    v.set(&[k, l, i, j], x);
                }
            }
        }
    }
    (t, v)
}

fn assemble_median_policy(
    space: QnSpace,
    t: &crate::coeff::CoeffTensor<f64>,
    v: &crate::coeff::CoeffTensor<f64>,
    w: Option<&dyn crate::coeff::ThreeBody<f64>>,
    switches: Option<FxHashMap<super::super::bipartite::PackedOps, usize>>,
    early_break: bool,
) -> Assembled<f64> {
    let d = space.d;
    let mut auto = Automaton::<f64> {
        space,
        hd: (d as i64 - 1) / 2,
        vlabels: Vec::new(),
        fams: Vec::new(),
        fam_of: FxHashMap::default(),
        n_slots: space.pg.order() as usize * (space.nup + 1) * (space.ndn + 1),
        caps: (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space)),
        edge_keys: Vec::new(),
        edge_data: Vec::new(),
        edge_site: Vec::new(),
        fam_edge_seen: FxHashMap::default(),
        qn_cache: FxHashMap::default(),
        switches,
        early_break,
        cur_term: 0,
        record_src: false,
    };
    enumerate_1body(&mut auto, t, f64::EPSILON, d);
    enumerate_2body(&mut auto, v, f64::EPSILON, d);
    if let Some(w) = w {
        enumerate_3body(&mut auto, w, f64::EPSILON);
    }
    finalize(auto)
}

fn assert_same_artifact(space: QnSpace, a: &Assembled<f64>, b: &Assembled<f64>, ctx: &str) {
    let d = space.d;
    for bond in 0..=d {
        for &s in &space.bond_sectors(bond) {
            assert_eq!(
                a.states[bond][s].len(),
                b.states[bond][s].len(),
                "{ctx}: bond {bond} sector {s:?} channel count differs"
            );
            assert_eq!(a.states[bond][s], b.states[bond][s], "{ctx}: labels differ");
            assert_eq!(
                a.state_qns[bond][s], b.state_qns[bond][s],
                "{ctx}: qns differ"
            );
        }
    }
    for k in 0..d {
        let (ga, gb) = (a.coo[k].iter_grid_order(), b.coo[k].iter_grid_order());
        assert_eq!(
            ga.len(),
            gb.len(),
            "{ctx}: site {k} populated cell count differs"
        );
        for (ea, eb) in ga.iter().zip(&gb) {
            assert_eq!(
                (ea.0, ea.1, ea.2, ea.3),
                (eb.0, eb.1, eb.2, eb.3),
                "{ctx}: site {k} populated cell key differs"
            );
            assert_eq!(ea.4.rows, eb.4.rows, "{ctx}: site {k} COO rows differ");
            assert_eq!(ea.4.cols, eb.4.cols, "{ctx}: site {k} COO cols differ");
            assert_eq!(ea.4.vals, eb.4.vals, "{ctx}: site {k} COO vals differ");
        }
    }
}

#[test]
fn median_early_break_is_layout_neutral() {
    let space = QnSpace::from_n_two_sz(6, 4, 0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB4EA_1234);
    let (t, v) = dense_random_coeffs(space.d, &mut rng);
    let with_break = assemble_with(space, &t, &v, None, f64::EPSILON, Construction::Median);
    let no_break = assemble_median_policy(space, &t, &v, None, Some(FxHashMap::default()), false);
    assert_same_artifact(space, &with_break, &no_break, "median policy");
}

#[test]
fn cover_switch_early_break_is_layout_neutral() {
    let shapes = [
        QnSpace::from_n_two_sz(6, 4, 0),
        QnSpace::new(6, 2, 1),
        QnSpace::new(6, 3, 3),
    ];
    for (si, &space) in shapes.iter().enumerate() {
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xEB_0000 ^ si as u64);
        let (t, v) = dense_random_coeffs(d, &mut rng);
        let w = if si == 2 {
            let mut w = Sparse3Body::zeros(d);
            for _ in 0..40 {
                let idx: [usize; 6] = std::array::from_fn(|_| {
                    (rng.sample::<f64, _>(StandardNormal).abs() * d as f64) as usize % d
                });
                w.set(&idx, rng.sample::<f64, _>(StandardNormal));
            }
            Some(w)
        } else {
            None
        };
        let w_dyn: Option<&dyn crate::coeff::ThreeBody<f64>> =
            w.as_ref().map(|w| w as &dyn crate::coeff::ThreeBody<f64>);

        let mut terms = collect_terms(space, &t, &v, w_dyn, f64::EPSILON);
        let switches = super::super::bipartite::cover_switch_bonds(space, &mut terms, true);
        assert!(
            !switches.is_empty(),
            "shape {si}: no cover switches recorded"
        );

        let with_break = assemble_median_policy(space, &t, &v, w_dyn, Some(switches.clone()), true);
        let no_break = assemble_median_policy(space, &t, &v, w_dyn, Some(switches), false);
        assert_same_artifact(space, &with_break, &no_break, &format!("cover shape {si}"));
    }
}
