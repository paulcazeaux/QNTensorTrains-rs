// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::f64::consts::{FRAC_PI_4, LN_2};
use std::path::Path;

use qntt::core::factor::dense_self_adjoint_smallest;
use qntt::entropy::{OrbitalEntanglement, orbital_entanglement};
use qntt::integrals::{IntegralsOptions, PointGroupPolicy, load_fixture_dir};
use qntt::rdm::{OrbitalRdms, orbital_rdms};
use qntt::solvers::mals::{MalsOptions, mals_with_history};
use qntt::{
    C64, CoeffTensor, Irrep, Orbital, PointGroup, QnSpace, Scalar, SparseHamiltonian, Spin,
    TtVector, dot, max_ranks, tt_randn, tt_state,
};
use qntt_testkit::gad_mol::{coeff_vector, dense_h_in_sector, dense_sector_eigen, det_states};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::{
    dense_apply_string, dense_eigen, dense_hamiltonian_matrix, dets_for, randomized_state,
};

fn c_ops(k: usize, a: usize) -> Vec<(Orbital, bool)> {
    let up = (Orbital::new(k, Spin::Up), true);
    let dn = (Orbital::new(k, Spin::Dn), true);
    match a {
        0 => vec![],
        1 => vec![up],
        2 => vec![dn],
        3 => vec![up, dn],
        _ => unreachable!("local label {a} is not in 0..4"),
    }
}

fn adjoint(s: &[(Orbital, bool)]) -> Vec<(Orbital, bool)> {
    s.iter().rev().map(|&(o, c)| (o, !c)).collect()
}

fn dense_rho_entry<T: Scalar>(
    d: usize,
    xd: &[T],
    sites: &[usize],
    row: &[usize],
    col: &[usize],
) -> T {
    let cr: Vec<_> = sites
        .iter()
        .zip(row)
        .flat_map(|(&k, &a)| c_ops(k, a))
        .collect();
    let cc: Vec<_> = sites
        .iter()
        .zip(col)
        .flat_map(|(&k, &a)| c_ops(k, a))
        .collect();
    let mut w = dense_apply_string(d, &adjoint(&cr), xd);
    for (idx, wv) in w.iter_mut().enumerate() {
        if sites.iter().any(|&k| (idx >> (2 * k)) & 3 != 0) {
            *wv = T::zero();
        }
    }
    let u = dense_apply_string(d, &cc, &w);
    let num = xd
        .iter()
        .zip(&u)
        .fold(T::zero(), |acc, (&a, &b)| acc + a.conj_s() * b);
    let den: f64 = xd.iter().map(|a| a.abs_s() * a.abs_s()).sum();
    num * T::from_f64(1.0 / den)
}

fn occ(a: usize) -> (usize, usize) {
    (a & 1, a >> 1)
}

fn pair_counts(idx: usize) -> (usize, usize) {
    let (ua, da) = occ(idx % 4);
    let (ub, db) = occ(idx / 4);
    (ua + ub, da + db)
}

fn allowed_pair(r: usize, c: usize) -> bool {
    pair_counts(r) == pair_counts(c)
}

fn label_feasible(space: &QnSpace, (u, dn): (usize, usize), outside: usize) -> bool {
    u <= space.nup && dn <= space.ndn && space.nup - u <= outside && space.ndn - dn <= outside
}

fn site_i_flux(r: usize, c: usize) -> (i64, i64) {
    let (ua, da) = occ(r % 4);
    let (ub, db) = occ(c % 4);
    (ub as i64 - ua as i64, db as i64 - da as i64)
}

fn is_single_transfer(r: usize, c: usize) -> bool {
    let (u, dn) = site_i_flux(r, c);
    allowed_pair(r, c) && u.abs() + dn.abs() == 1
}

const PAIR_HOP: (usize, usize) = (12, 3);
const SPIN_FLIP: (usize, usize) = (6, 9);

struct PairStats {
    i: usize,
    j: usize,
    single: f64,
    pair_hop: f64,
    spin_flip: f64,
    imag: f64,
}

fn full_rank<T: Scalar>(space: QnSpace, seed: u64) -> TtVector<T> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    tt_randn(space, &max_ranks(&space), &mut rng)
}

fn check_against_dense<T: Scalar>(
    x: &TtVector<T>,
    label: &str,
) -> (OrbitalRdms<T>, Vec<PairStats>) {
    let d = x.space.d;
    let xd = x.to_dense();
    let rdms = orbital_rdms(x);
    assert_eq!(rdms.d(), d, "{label}: OrbitalRdms::d()");
    let mut worst = 0.0f64;
    let mut infeasible_checked = 0usize;
    for i in 0..d {
        let r1 = rdms.rho1(i);
        for a in 0..4 {
            for ap in 0..4 {
                let got = r1[a * 4 + ap];
                if !label_feasible(&x.space, occ(a), d - 1) {
                    infeasible_checked += 1;
                    assert!(
                        got == T::zero(),
                        "{label}: rho1({i})[{a}][{ap}] = ({:e}, {:e}) has a globally infeasible \
                         label and must be exactly 0 (R3)",
                        got.re_s(),
                        got.im_s()
                    );
                }
                if a != ap {
                    assert!(
                        got == T::zero(),
                        "{label}: rho1({i})[{a}][{ap}] = ({:e}, {:e}) must be exactly 0 (R3)",
                        got.re_s(),
                        got.im_s()
                    );
                }
                let want = dense_rho_entry(d, &xd, &[i], &[a], &[ap]);
                let err = (got - want).abs_s();
                worst = worst.max(err);
                assert!(
                    err <= 1e-12,
                    "{label}: rho1({i})[{a}][{ap}] = ({:e}, {:e}) vs R1 dense ({:e}, {:e}), |Δ| = {err:e}",
                    got.re_s(),
                    got.im_s(),
                    want.re_s(),
                    want.im_s()
                );
            }
        }
    }
    let mut stats = Vec::new();
    for i in 0..d {
        for j in (i + 1)..d {
            let r2 = rdms.rho2(i, j);
            let mut st = PairStats {
                i,
                j,
                single: 0.0,
                pair_hop: 0.0,
                spin_flip: 0.0,
                imag: 0.0,
            };
            for r in 0..16 {
                for c in 0..16 {
                    let got = r2[r * 16 + c];
                    if !label_feasible(&x.space, pair_counts(r), d - 2)
                        || !label_feasible(&x.space, pair_counts(c), d - 2)
                    {
                        infeasible_checked += 1;
                        assert!(
                            got == T::zero(),
                            "{label}: rho2({i},{j})[{r}][{c}] = ({:e}, {:e}) has a globally \
                             infeasible label and must be exactly 0 (R3)",
                            got.re_s(),
                            got.im_s()
                        );
                    }
                    if !allowed_pair(r, c) {
                        assert!(
                            got == T::zero(),
                            "{label}: rho2({i},{j})[{r}][{c}] = ({:e}, {:e}) is particle-number \
                             forbidden and must be exactly 0 (R3)",
                            got.re_s(),
                            got.im_s()
                        );
                    }
                    let want = dense_rho_entry(d, &xd, &[i, j], &[r % 4, r / 4], &[c % 4, c / 4]);
                    let err = (got - want).abs_s();
                    worst = worst.max(err);
                    assert!(
                        err <= 1e-12,
                        "{label}: rho2({i},{j})[(α,β)=({},{})][(α′,β′)=({},{})] = ({:e}, {:e}) vs R1 \
                         dense ({:e}, {:e}), |Δ| = {err:e}",
                        r % 4,
                        r / 4,
                        c % 4,
                        c / 4,
                        got.re_s(),
                        got.im_s(),
                        want.re_s(),
                        want.im_s()
                    );
                    if is_single_transfer(r, c) {
                        st.single = st.single.max(got.abs_s());
                    }
                    if r != c {
                        st.imag = st.imag.max(got.im_s().abs());
                    }
                }
            }
            st.pair_hop = r2[PAIR_HOP.0 * 16 + PAIR_HOP.1].abs_s();
            st.spin_flip = r2[SPIN_FLIP.0 * 16 + SPIN_FLIP.1].abs_s();
            stats.push(st);
        }
    }
    println!(
        "{label}: max |kernel − R1 dense| = {worst:e} over every ρᵢ and ρᵢⱼ entry; \
         {infeasible_checked} globally-infeasible-label entries checked exactly 0"
    );
    (rdms, stats)
}

fn a1_anti_vacuity<T: Scalar>(stats: &[PairStats], pair_classes: bool, label: &str) {
    let long = stats
        .iter()
        .filter(|s| s.j >= s.i + 2)
        .fold(0.0f64, |a, s| a.max(s.single));
    println!("{label}: max |single transfer| at j ≥ i+2 = {long:e}");
    assert!(
        long > 1e-6,
        "{label}: no single-transfer mass across an intermediate site ({long:e})"
    );
    if pair_classes {
        let p = stats.iter().fold(0.0f64, |a, s| a.max(s.pair_hop));
        let s = stats.iter().fold(0.0f64, |a, s| a.max(s.spin_flip));
        println!("{label}: max |pair hop| = {p:e}, max |spin flip| = {s:e}");
        assert!(
            p > 1e-6 && s > 1e-6,
            "{label}: pair-hop {p:e} / spin-flip {s:e} vacuous"
        );
    }
    if T::IS_COMPLEX {
        let im = stats.iter().fold(0.0f64, |a, s| a.max(s.imag));
        assert!(
            im > 1e-6,
            "{label}: C64 off-diagonals are real ({im:e}): transpose pin vacuous"
        );
    }
}

const A1_CELLS: [(usize, usize, usize); 6] = [
    (4, 2, 2),
    (4, 2, 1),
    (4, 3, 0),
    (4, 1, 0),
    (5, 2, 1),
    (4, 0, 2),
];

fn a1_seed(base: u64, (d, nu, nd): (usize, usize, usize)) -> u64 {
    base + (d * 100 + nu * 10 + nd) as u64
}

#[test]
fn tord_a1_dense_oracle_f64() {
    for &cell in &A1_CELLS {
        let (d, nu, nd) = cell;
        let space = QnSpace::new(d, nu, nd);
        let x: TtVector<f64> = full_rank(space, a1_seed(0xA1_0000, cell));
        let label = format!("tord_a1 f64 d={d} ({nu},{nd})");
        let (_, stats) = check_against_dense(&x, &label);
        a1_anti_vacuity::<f64>(&stats, nu > 0 && nd > 0, &label);
    }
}

#[test]
fn tord_a1_dense_oracle_c64() {
    for &cell in &A1_CELLS {
        let (d, nu, nd) = cell;
        let space = QnSpace::new(d, nu, nd);
        let x: TtVector<C64> = full_rank(space, a1_seed(0xA1_C000, cell));
        let label = format!("tord_a1 c64 d={d} ({nu},{nd})");
        let (_, stats) = check_against_dense(&x, &label);
        a1_anti_vacuity::<C64>(&stats, nu > 0 && nd > 0, &label);
    }
}

#[test]
fn tord_a1_zero_rank_interior_sectors() {
    let space = QnSpace::new(4, 2, 1);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA1_0201);
    let x: TtVector<f64> = randomized_state(space, 0, 2, &mut rng);
    let zero_sectors: usize = (1..space.d)
        .map(|b| {
            space
                .bond_sectors(b)
                .into_iter()
                .filter(|&s| x.ranks[b][s] == 0)
                .count()
        })
        .sum();
    let n = dot(&x, &x);
    println!("tord_a1 zero-rank: {zero_sectors} zero-rank interior sectors, <x,x> = {n:e}");
    assert!(
        zero_sectors > 0,
        "no zero-rank interior sector: arm is vacuous"
    );
    assert!(n > 1e-3, "state is (numerically) zero: <x,x> = {n:e}");
    let (_, stats) = check_against_dense(&x, "tord_a1 zero-rank sectors");
    a1_anti_vacuity::<f64>(&stats, false, "tord_a1 zero-rank sectors");
}

fn bits<T: Scalar>(z: T) -> (u64, u64) {
    (z.re_s().to_bits(), z.im_s().to_bits())
}

fn a2_invariants<T: Scalar>(x: &TtVector<T>, label: &str) {
    let d = x.space.d;
    let rdms = orbital_rdms(x);
    assert_eq!(rdms.d(), d);
    let rho1: Vec<[T; 16]> = (0..d).map(|i| rdms.rho1(i)).collect();
    let mut worst_tr = 0.0f64;
    let mut worst_pt = 0.0f64;
    for (i, r1) in rho1.iter().enumerate() {
        let mut tr = T::zero();
        for a in 0..4 {
            tr += r1[a * 4 + a];
            for b in 0..4 {
                assert!(
                    r1[a * 4 + b] == r1[b * 4 + a].conj_s(),
                    "{label}: rho1({i}) not bitwise Hermitian at ({a},{b})"
                );
                if a != b {
                    assert!(
                        r1[a * 4 + b] == T::zero(),
                        "{label}: rho1({i})[{a}][{b}] != 0"
                    );
                }
            }
        }
        worst_tr = worst_tr.max((tr - T::one()).abs_s());
    }
    for i in 0..d {
        for j in (i + 1)..d {
            let r2 = rdms.rho2(i, j);
            let mut tr = T::zero();
            for r in 0..16 {
                tr += r2[r * 16 + r];
                for c in 0..16 {
                    assert!(
                        r2[r * 16 + c] == r2[c * 16 + r].conj_s(),
                        "{label}: rho2({i},{j}) not bitwise Hermitian at ({r},{c})"
                    );
                }
            }
            worst_tr = worst_tr.max((tr - T::one()).abs_s());
            for a in 0..4 {
                for ap in 0..4 {
                    let mut ti = T::zero();
                    let mut tj = T::zero();
                    for b in 0..4 {
                        ti += r2[(a + 4 * b) * 16 + (ap + 4 * b)];
                        tj += r2[(b + 4 * a) * 16 + (b + 4 * ap)];
                    }
                    worst_pt = worst_pt.max((ti - rho1[i][a * 4 + ap]).abs_s());
                    worst_pt = worst_pt.max((tj - rho1[j][a * 4 + ap]).abs_s());
                }
            }
        }
    }
    println!("{label}: max |Tr ρ − 1| = {worst_tr:e}, max partial-trace defect = {worst_pt:e}");
    assert!(worst_tr <= 1e-12, "{label}: trace defect {worst_tr:e}");
    assert!(
        worst_pt <= 1e-12,
        "{label}: partial-trace defect {worst_pt:e}"
    );

    let ent = orbital_entanglement(&rdms);
    let mut want_min = 0.0f64;
    for i in 0..d {
        let (ev, _) = dense_self_adjoint_smallest::<T>(&rdms.rho1(i), 4, 0);
        want_min = ev.iter().fold(want_min, |m, &l| m.min(l));
        for j in (i + 1)..d {
            let (ev, _) = dense_self_adjoint_smallest::<T>(&rdms.rho2(i, j), 16, 0);
            want_min = ev.iter().fold(want_min, |m, &l| m.min(l));
        }
    }
    println!("{label}: recomputed min(0, min eig) = {want_min:e}");
    assert_eq!(
        ent.min_eigenvalue, want_min,
        "{label}: min_eigenvalue does not equal the minimum over every ρ (E3)"
    );
    println!(
        "{label}: min_eigenvalue = {:e}, I_tot = {}",
        ent.min_eigenvalue,
        ent.i_tot()
    );
    assert!(
        ent.min_eigenvalue <= 0.0 && ent.min_eigenvalue >= -1e-12,
        "{label}: min_eigenvalue = {:e}",
        ent.min_eigenvalue
    );
    assert_eq!(ent.d, d);
    assert_eq!(ent.s1.len(), d);
    assert_eq!(ent.s2.len(), d * d);
    assert_eq!(ent.mutual.len(), d * d);
    for i in 0..d {
        assert!(
            ent.s1[i] >= -1e-12 && ent.s1[i] <= 4f64.ln() + 1e-12,
            "{label}: s1[{i}] = {} outside [0, ln 4]",
            ent.s1[i]
        );
        assert_eq!(
            ent.s2[i * d + i],
            ent.s1[i],
            "{label}: s2[{i}][{i}] != s1[{i}]"
        );
        assert_eq!(ent.mutual[i * d + i], 0.0, "{label}: mutual[{i}][{i}] != 0");
        for j in 0..d {
            assert_eq!(
                ent.s2[i * d + j],
                ent.s2[j * d + i],
                "{label}: s2 not symmetric"
            );
            assert_eq!(
                ent.mutual[i * d + j],
                ent.mutual[j * d + i],
                "{label}: mutual not symmetric"
            );
            if i != j {
                let def = ent.s1[i] + ent.s1[j] - ent.s2[i * d + j];
                assert!(
                    (ent.mutual[i * d + j] - def).abs() <= 1e-14,
                    "{label}: mutual[{i}][{j}] = {} != s_i + s_j − s_ij = {def}",
                    ent.mutual[i * d + j]
                );
                assert!(
                    ent.mutual[i * d + j] >= -1e-12,
                    "{label}: I[{i}][{j}] = {:e} < 0",
                    ent.mutual[i * d + j]
                );
                assert!(
                    ent.s2[i * d + j] <= 16f64.ln() + 1e-12,
                    "{label}: s2 > ln 16"
                );
            }
        }
    }

    let again = orbital_rdms(x);
    for i in 0..d {
        let (a, b) = (rdms.rho1(i), again.rho1(i));
        for k in 0..16 {
            assert_eq!(
                bits(a[k]),
                bits(b[k]),
                "{label}: rho1({i}) not deterministic at {k}"
            );
        }
        for j in (i + 1)..d {
            let (a, b) = (rdms.rho2(i, j), again.rho2(i, j));
            for k in 0..256 {
                assert_eq!(
                    bits(a[k]),
                    bits(b[k]),
                    "{label}: rho2({i},{j}) not deterministic at {k}"
                );
            }
        }
    }

    let x2 = x * T::from_f64(2.0);
    let x3 = x * T::from_f64(3.0);
    let (r2x, r3x) = (orbital_rdms(&x2), orbital_rdms(&x3));
    let mut worst3 = 0.0f64;
    let mut cmp = |a: &[T], b: &[T], c: &[T], what: &str| {
        for k in 0..a.len() {
            assert!(
                a[k] == b[k],
                "{label}: {what} changed under x → 2x at {k} (R2)"
            );
            worst3 = worst3.max((a[k] - c[k]).abs_s());
        }
    };
    for i in 0..d {
        cmp(&rdms.rho1(i), &r2x.rho1(i), &r3x.rho1(i), "rho1");
        for j in (i + 1)..d {
            cmp(&rdms.rho2(i, j), &r2x.rho2(i, j), &r3x.rho2(i, j), "rho2");
        }
    }
    println!("{label}: max |ρ(3x) − ρ(x)| = {worst3:e}");
    assert!(
        worst3 <= 1e-13,
        "{label}: ρ(3x) differs from ρ(x) by {worst3:e}"
    );
    if T::IS_COMPLEX {
        let xp = x * T::from_complex_f64(0.6, 0.8);
        let rp = orbital_rdms(&xp);
        let mut worst = 0.0f64;
        for i in 0..d {
            let (a, b) = (rdms.rho1(i), rp.rho1(i));
            for k in 0..16 {
                worst = worst.max((a[k] - b[k]).abs_s());
            }
            for j in (i + 1)..d {
                let (a, b) = (rdms.rho2(i, j), rp.rho2(i, j));
                for k in 0..256 {
                    worst = worst.max((a[k] - b[k]).abs_s());
                }
            }
        }
        println!("{label}: max |ρ(e^{{iφ}}x) − ρ(x)| = {worst:e}");
        assert!(
            worst <= 1e-13,
            "{label}: global phase changed ρ by {worst:e}"
        );
    }
}

#[test]
fn tord_a2_invariants_f64() {
    for (k, &(nu, nd)) in [(3usize, 3usize), (4, 0)].iter().enumerate() {
        let space = QnSpace::new(8, nu, nd);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2_0000 + k as u64);
        let x: TtVector<f64> = randomized_state(space, 1, 8, &mut rng);
        a2_invariants(&x, &format!("tord_a2 f64 d=8 ({nu},{nd})"));
    }
}

#[test]
fn tord_a2_invariants_c64() {
    for (k, &(nu, nd)) in [(3usize, 3usize), (4, 0)].iter().enumerate() {
        let space = QnSpace::new(8, nu, nd);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2_C000 + k as u64);
        let x: TtVector<C64> = randomized_state(space, 1, 8, &mut rng);
        a2_invariants(&x, &format!("tord_a2 c64 d=8 ({nu},{nd})"));
    }
}

fn flags(d: usize, sites: &[usize]) -> Vec<bool> {
    let mut f = vec![false; d];
    for &s in sites {
        f[s] = true;
    }
    f
}

#[test]
fn tord_a3_determinants_have_zero_entanglement() {
    let cases: [(usize, &[usize], &[usize]); 3] = [
        (5, &[0, 2, 3], &[0, 1, 4]),
        (4, &[1], &[1, 3]),
        (4, &[0, 1, 2, 3], &[2]),
    ];
    for (d, up, dn) in cases {
        let (ou, od) = (flags(d, up), flags(d, dn));
        let x = tt_state::<f64>(&ou, &od);
        let rdms = orbital_rdms(&x);
        let label = format!("tord_a3 det d={d} up={up:?} dn={dn:?}");
        let code: Vec<usize> = (0..d)
            .map(|k| ou[k] as usize + 2 * od[k] as usize)
            .collect();
        for i in 0..d {
            let r1 = rdms.rho1(i);
            for k in 0..16 {
                let want = if k == code[i] * 4 + code[i] { 1.0 } else { 0.0 };
                assert!(
                    (r1[k] - want).abs() <= 1e-14,
                    "{label}: rho1({i})[{k}] = {}",
                    r1[k]
                );
            }
            for j in (i + 1)..d {
                let r2 = rdms.rho2(i, j);
                let lab = code[i] + 4 * code[j];
                for k in 0..256 {
                    let want = if k == lab * 16 + lab { 1.0 } else { 0.0 };
                    assert!(
                        (r2[k] - want).abs() <= 1e-14,
                        "{label}: rho2({i},{j})[{k}] = {}",
                        r2[k]
                    );
                }
            }
        }
        let ent = orbital_entanglement(&rdms);
        for (name, v) in [("s1", &ent.s1), ("s2", &ent.s2), ("mutual", &ent.mutual)] {
            for (k, &s) in v.iter().enumerate() {
                assert!(
                    s.is_finite() && s.abs() <= 1e-14,
                    "{label}: {name}[{k}] = {s:e}"
                );
            }
        }
    }
}

#[test]
fn tord_a3_vacuum_and_single_site() {
    let d = 3;
    let vac = tt_state::<f64>(&[false; 3], &[false; 3]);
    let rdms = orbital_rdms(&vac);
    assert_eq!(rdms.d(), d);
    for i in 0..d {
        let r1 = rdms.rho1(i);
        for k in 0..16 {
            assert_eq!(
                r1[k],
                if k == 0 { 1.0 } else { 0.0 },
                "vacuum rho1({i})[{k}]"
            );
        }
        for j in (i + 1)..d {
            let r2 = rdms.rho2(i, j);
            for k in 0..256 {
                assert_eq!(
                    r2[k],
                    if k == 0 { 1.0 } else { 0.0 },
                    "vacuum rho2({i},{j})[{k}]"
                );
            }
        }
    }
    let ent = orbital_entanglement(&rdms);
    assert!(
        ent.s1
            .iter()
            .chain(&ent.s2)
            .chain(&ent.mutual)
            .all(|s| s.abs() <= 1e-14)
    );
    assert!(ent.min_eigenvalue <= 0.0 && ent.min_eigenvalue >= -1e-12);

    let one = tt_state::<f64>(&[true], &[false]);
    let r = orbital_rdms(&one);
    assert_eq!(r.d(), 1);
    let r1 = r.rho1(0);
    for k in 0..16 {
        assert!(
            (r1[k] - if k == 5 { 1.0 } else { 0.0 }).abs() <= 1e-14,
            "d=1 rho1(0)[{k}]"
        );
    }
    let ent = orbital_entanglement(&r);
    assert_eq!(ent.d, 1);
    assert_eq!(ent.s2, vec![ent.s1[0]], "d=1: s2 must be [s1[0]]");
    assert_eq!(ent.mutual, vec![0.0], "d=1: mutual must be [0.0]");
    assert!(ent.s1[0].abs() <= 1e-14);
}

#[test]
fn tord_a3_spinless_electron_closed_forms() {
    let e0 = |d: usize| flags(d, &[0]);
    let e1 = |d: usize| flags(d, &[1]);
    let none = vec![false; 3];
    for theta in [0.3f64, FRAC_PI_4] {
        let (c, s) = (theta.cos(), theta.sin());
        let s0 = -(c * c) * (c * c).ln() - (s * s) * (s * s).ln();
        let a = tt_state::<f64>(&e0(3), &none);
        let b = tt_state::<f64>(&e1(3), &none);
        let x = &(&a * c) + &(&b * s);
        let xd = x.to_dense();
        for (k, &v) in xd.iter().enumerate() {
            let want = match k {
                1 => c,
                4 => s,
                _ => 0.0,
            };
            assert!(
                (v - want).abs() <= 1e-15,
                "θ={theta}: to_dense[{k}] = {v}, want {want}"
            );
        }
        let rdms = orbital_rdms(&x);
        let r01 = rdms.rho2(0, 1);
        assert!(
            (r01[16 + 4] - c * s).abs() <= 1e-12,
            "θ={theta}: ρ01[(1,0)][(0,1)] = {}",
            r01[16 + 4]
        );
        assert!((r01[4 * 16 + 1] - c * s).abs() <= 1e-12);
        let ent = orbital_entanglement(&rdms);
        check_spinless(&ent, s0, theta, "f64");

        let a = tt_state::<C64>(&e0(3), &none);
        let b = tt_state::<C64>(&e1(3), &none);
        let x = &(&a * C64::new(c, 0.0)) + &(&b * C64::new(0.0, s));
        let xd = x.to_dense();
        assert!(
            (xd[1] - C64::new(c, 0.0)).norm() <= 1e-15
                && (xd[4] - C64::new(0.0, s)).norm() <= 1e-15
        );
        let rdms = orbital_rdms(&x);
        let r01 = rdms.rho2(0, 1);
        assert!(
            (r01[16 + 4] - C64::new(0.0, -c * s)).norm() <= 1e-12,
            "θ={theta}: C64 ρ01[(1,0)][(0,1)] = {} (want −i·c·s)",
            r01[16 + 4]
        );
        assert!((r01[4 * 16 + 1] - C64::new(0.0, c * s)).norm() <= 1e-12);
        let ent = orbital_entanglement(&rdms);
        check_spinless(&ent, s0, theta, "c64");
    }
}

fn check_spinless(ent: &OrbitalEntanglement, s0: f64, theta: f64, ty: &str) {
    let d = 3;
    let tag = format!("tord_a3 spinless {ty} θ={theta}");
    println!("{tag}: s0 closed form {s0}, kernel s1 = {:?}", ent.s1);
    let close = |got: f64, want: f64, what: &str| {
        assert!(
            (got - want).abs() <= 1e-12,
            "{tag}: {what} = {got}, want {want}"
        );
    };
    close(ent.s1[0], s0, "s1[0]");
    close(ent.s1[1], s0, "s1[1]");
    close(ent.s1[2], 0.0, "s1[2]");
    close(ent.s2[1], 0.0, "s2[0][1]");
    close(ent.mutual[1], 2.0 * s0, "mutual[0][1]");
    close(ent.mutual[2], 0.0, "mutual[0][2]");
    close(ent.mutual[d + 2], 0.0, "mutual[1][2]");
    close(ent.i_tot(), 2.0 * s0, "i_tot");
    close(ent.i_dist(2), 4.0 * s0, "i_dist(2)");
    for i in 0..d {
        assert_eq!(ent.s2[i * d + i], ent.s1[i], "{tag}: s2 diagonal");
        assert_eq!(ent.mutual[i * d + i], 0.0, "{tag}: mutual diagonal");
        for j in 0..d {
            assert_eq!(
                ent.mutual[i * d + j],
                ent.mutual[j * d + i],
                "{tag}: mutual symmetry"
            );
        }
    }
}

#[test]
fn tord_a3_d2_pure_pair_state() {
    fn arm<T: Scalar>(seed: u64, ty: &str) {
        let space = QnSpace::new(2, 1, 1);
        let mut x: TtVector<T> = full_rank(space, seed);
        let nrm = x.norm();
        x.scale_mut(T::from_f64(1.0 / nrm));
        let xd = x.to_dense();
        let mut total = 0.0;
        for (k, v) in xd.iter().enumerate() {
            if ![3, 6, 9, 12].contains(&k) {
                assert!(
                    *v == T::zero(),
                    "{ty}: to_dense[{k}] must be 0 in the (1,1) sector"
                );
            }
            total += v.abs_s() * v.abs_s();
        }
        assert!((total - 1.0).abs() <= 1e-14, "{ty}: Σ|x|² = {total}");
        let p = |k: usize| xd[k].abs_s() * xd[k].abs_s();
        let d0 = [p(12), p(9), p(6), p(3)];
        let d1 = [p(3), p(6), p(9), p(12)];
        let shannon = -d0
            .iter()
            .filter(|&&q| q > 0.0)
            .map(|&q| q * q.ln())
            .sum::<f64>();
        let rdms = orbital_rdms(&x);
        let (r0, r1) = (rdms.rho1(0), rdms.rho1(1));
        for a in 0..4 {
            assert!(
                (r0[a * 4 + a] - T::from_f64(d0[a])).abs_s() <= 1e-12,
                "{ty}: rho1(0)[{a}]"
            );
            assert!(
                (r1[a * 4 + a] - T::from_f64(d1[a])).abs_s() <= 1e-12,
                "{ty}: rho1(1)[{a}]"
            );
        }
        let ent = orbital_entanglement(&rdms);
        println!(
            "tord_a3 d=2 {ty}: shannon {shannon}, s1 {:?}, s01 {:e}",
            ent.s1, ent.s2[1]
        );
        assert!(
            ent.s2[1].abs() <= 1e-12,
            "{ty}: s01 = {:e} (pure pair state)",
            ent.s2[1]
        );
        assert!(
            (ent.s1[0] - shannon).abs() <= 1e-12,
            "{ty}: s0 = {} vs {shannon}",
            ent.s1[0]
        );
        assert!(
            (ent.s1[1] - shannon).abs() <= 1e-12,
            "{ty}: s1 = {} vs {shannon}",
            ent.s1[1]
        );
        assert!((ent.mutual[1] - 2.0 * shannon).abs() <= 1e-12, "{ty}: I01");
        assert!(
            shannon > 0.1,
            "{ty}: random state is nearly a product ({shannon})"
        );
    }
    arm::<f64>(0xA3_0002, "f64");
    arm::<C64>(0xA3_C002, "c64");
}

#[test]
fn tord_a3_i_tot_and_i_dist_hand_built() {
    let (a, b, c) = (0.25, 0.125, 0.5);
    let ent = OrbitalEntanglement {
        d: 3,
        s1: vec![0.5, 0.25, 0.75],
        s2: vec![99.0; 9],
        mutual: vec![1000.0, a, b, a, 1000.0, c, b, c, 1000.0],
        min_eigenvalue: 0.0,
    };
    assert_eq!(ent.i_tot(), 1.5);
    assert_eq!(ent.i_dist(0), 2.0 * (a + b + c));
    assert_eq!(ent.i_dist(1), 2.0 * (a + 2.0 * b + c));
    assert_eq!(ent.i_dist(2), 2.0 * (a + 4.0 * b + c));
    assert_eq!(ent.i_dist(3), 2.0 * (a + 8.0 * b + c));
}

fn a4_arm<T: Scalar>(pg: PointGroup, target: Irrep, seed: u64, spin_flip_floor: bool, label: &str) {
    let space = QnSpace::with_point_group(4, 2, 1, pg, target);
    let x: TtVector<T> = full_rank(space, seed);
    assert!(x.check(), "{label}: tt_randn state fails check()");
    let (rdms, stats) = check_against_dense(&x, label);
    let mut same_single = 0.0f64;
    let (mut cross_p, mut cross_s) = (0.0f64, 0.0f64);
    let mut n_cross = 0;
    for st in &stats {
        let cross = pg.irrep_of(st.i) != pg.irrep_of(st.j);
        if cross {
            n_cross += 1;
            let r2 = rdms.rho2(st.i, st.j);
            for r in 0..16 {
                for c in 0..16 {
                    if is_single_transfer(r, c) {
                        assert!(
                            r2[r * 16 + c] == T::zero(),
                            "{label}: cross-irrep pair ({},{}) single transfer [{r}][{c}] must be \
                             exactly 0 (R3)",
                            st.i,
                            st.j
                        );
                    }
                }
            }
            cross_p = cross_p.max(st.pair_hop);
            cross_s = cross_s.max(st.spin_flip);
        } else {
            same_single = same_single.max(st.single);
        }
    }
    println!(
        "{label}: {n_cross} cross-irrep pairs; same-irrep max single {same_single:e}; cross max \
         pair-hop {cross_p:e}, spin-flip {cross_s:e}"
    );
    assert!(n_cross > 0, "{label}: no cross-irrep pair");
    assert!(
        same_single > 1e-6,
        "{label}: same-irrep single transfers vacuous"
    );
    assert!(
        cross_p > 1e-6,
        "{label}: cross-irrep pair hop vacuous (over-gated?)"
    );
    if spin_flip_floor {
        assert!(
            cross_s > 1e-6,
            "{label}: cross-irrep spin flip vacuous (over-gated?)"
        );
    }
}

#[test]
fn tord_a4_point_group_gates_single_transfers() {
    let tables: [(u8, [u8; 4], bool); 2] = [(1, [0, 1, 1, 0], true), (2, [0, 1, 2, 1], false)];
    for (nbits, tab, sf) in tables {
        let irreps: Vec<Irrep> = tab.iter().map(|&g| Irrep::new(g)).collect();
        let pg = PointGroup::new(nbits, &irreps);
        for tg in [0u8, 1] {
            let target = Irrep::new(tg);
            let off = 16 * nbits as u64 + tg as u64;
            a4_arm::<f64>(
                pg,
                target,
                0xA4_0000 + off,
                sf,
                &format!("tord_a4 f64 {tab:?} target {tg}"),
            );
            a4_arm::<C64>(
                pg,
                target,
                0xA4_C000 + off,
                sf,
                &format!("tord_a4 c64 {tab:?} target {tg}"),
            );
        }
    }
}

#[test]
fn tord_a4_h2o_c2v_typed_fixture() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/h2o_c2v");
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        screen: true,
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(&dir, &opts).expect("h2o_c2v fixture loads");
    let space = load.space;
    assert!(space.pg.order() > 1, "h2o_c2v must resolve to |G| > 1");
    let x: TtVector<f64> = full_rank(space, 0xA4_C2F0);
    let (rdms, stats) = check_against_dense(&x, "tord_a4 h2o_c2v");
    let mut same_single = 0.0f64;
    let mut cross_p = 0.0f64;
    for st in &stats {
        if space.pg.irrep_of(st.i) != space.pg.irrep_of(st.j) {
            let r2 = rdms.rho2(st.i, st.j);
            for r in 0..16 {
                for c in 0..16 {
                    if is_single_transfer(r, c) {
                        assert!(
                            r2[r * 16 + c] == 0.0,
                            "h2o_c2v cross pair ({},{}) [{r}][{c}]",
                            st.i,
                            st.j
                        );
                    }
                }
            }
            cross_p = cross_p.max(st.pair_hop);
        } else {
            same_single = same_single.max(st.single);
        }
    }
    println!(
        "tord_a4 h2o_c2v: same-irrep max single {same_single:e}, cross max pair-hop {cross_p:e}"
    );
    assert!(same_single > 1e-6 && cross_p > 1e-6);
}

#[test]
fn tord_a5_tv_ring_translation_invariance() {
    let d = 6;
    let space = QnSpace::new(d, 3, 0);
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        let j = (i + 1) % d;
        t.set(&[i, j], 1.0);
        t.set(&[j, i], 1.0);
        v.set(&[i, i, j, j], 1.0);
        v.set(&[j, j, i, i], 1.0);
    }
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
    let dets = det_states::<f64>(d, 3, 0);
    let n = dets.len();
    let mat = dense_h_in_sector(&dets, &h);
    let (ev, _) = dense_sector_eigen(&mat, n);
    let gap = ev[1] - ev[0];
    let (ev_ind, _) = dense_eigen(&dense_hamiltonian_matrix(d, &t, &v, &dets_for(d, 3, 0)));
    println!(
        "tord_a5: E0 = {:.12}, E1 = {:.12}, gap = {gap:.6}",
        ev[0], ev[1]
    );
    assert!(
        (ev[0] - ev_ind[0]).abs() <= 1e-12,
        "E0 {} vs independent {}",
        ev[0],
        ev_ind[0]
    );
    assert!(gap > 1e-3, "(3,0) ground state is degenerate: gap {gap:e}");

    let x0: TtVector<f64> = full_rank(space, 0xA5_0001);
    let opts = MalsOptions {
        eps: 1e-10,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 20,
        ..Default::default()
    };
    let (hist, x) = mals_with_history(&h, &x0, &opts);
    println!(
        "tord_a5: MALS {} sweeps (budget {}), λ = {:.14}",
        hist.sweeps.len(),
        opts.max_iter,
        hist.lambda
    );
    assert!(
        hist.sweeps.len() < opts.max_iter,
        "MALS used its whole budget"
    );
    assert!(
        (hist.lambda - ev[0]).abs() <= 1e-10 * ev[0].abs().max(1.0),
        "MALS λ = {} vs dense E0 = {}",
        hist.lambda,
        ev[0]
    );
    let c = coeff_vector(&dets, &x);
    let nrm = c.iter().map(|a| a * a).sum::<f64>().sqrt();
    let c: Vec<f64> = c.iter().map(|a| a / nrm).collect();
    let hc: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| mat[i * n + j] * c[j]).sum())
        .collect();
    let rq: f64 = (0..n).map(|i| c[i] * hc[i]).sum();
    let res = (0..n)
        .map(|i| (hc[i] - rq * c[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("tord_a5: residual certificate ‖r‖/gap = {:e}", res / gap);
    assert!(
        res / gap <= 1e-9,
        "state not converged: ‖r‖/gap = {:e}",
        res / gap
    );

    let rdms = orbital_rdms(&x);
    for i in 0..d {
        let r1 = rdms.rho1(i);
        for (k, want) in [(0, 0.5), (5, 0.5), (10, 0.0), (15, 0.0)] {
            assert!(
                (r1[k] - want).abs() <= 1e-8,
                "ρ{i}[{k}] = {} (half filling)",
                r1[k]
            );
        }
    }
    let ent = orbital_entanglement(&rdms);
    println!(
        "tord_a5: I(k=1,2,3) = {}, {}, {}",
        ent.mutual[1], ent.mutual[2], ent.mutual[3]
    );
    for i in 0..d {
        assert!(
            (ent.s1[i] - LN_2).abs() <= 1e-8,
            "s{i} = {} vs ln 2",
            ent.s1[i]
        );
        for k in 1..=3 {
            let j = (i + k) % d;
            let got = ent.mutual[i * d + j];
            assert!(
                (got - ent.mutual[k]).abs() <= 1e-8,
                "I[{i}][{j}] = {got} vs I[0][{k}] = {} (translation invariance)",
                ent.mutual[k]
            );
        }
        for j in 0..d {
            assert_eq!(ent.mutual[i * d + j], ent.mutual[j * d + i]);
        }
        assert_eq!(ent.mutual[i * d + i], 0.0);
    }
    assert!(
        ent.mutual[1] > 0.1,
        "nearest-neighbour I = {} is vacuous",
        ent.mutual[1]
    );
}

fn refusal_rdms() -> OrbitalRdms<f64> {
    orbital_rdms(&tt_state::<f64>(
        &[true, false, false],
        &[false, true, false],
    ))
}

#[test]
#[should_panic(expected = "needs i < j < d = 3 (got (1, 0))")]
fn tord_a3_rho2_refuses_swapped_pair() {
    let _ = refusal_rdms().rho2(1, 0);
}

#[test]
#[should_panic(expected = "needs i < j < d = 3 (got (1, 1))")]
fn tord_a3_rho2_refuses_equal_pair() {
    let _ = refusal_rdms().rho2(1, 1);
}

#[test]
#[should_panic(expected = "needs i < j < d = 3 (got (1, 3))")]
fn tord_a3_rho2_refuses_out_of_range() {
    let _ = refusal_rdms().rho2(1, 3);
}

#[test]
#[should_panic(expected = "orbital 3 is out of range 0..3")]
fn tord_a3_rho1_refuses_out_of_range() {
    let _ = refusal_rdms().rho1(3);
}

#[test]
#[should_panic(expected = "must be positive and finite")]
fn tord_a3_zero_norm_state_refused() {
    let x = tt_state::<f64>(&[true, false, false], &[false, true, false]);
    let _ = orbital_rdms(&(&x * 0.0));
}
