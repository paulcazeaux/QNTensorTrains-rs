// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::collections::{HashMap, HashSet};

use qntt::deas::{active_space_order, deas_start, deas_start_in};
use qntt::ordering::inverse_permutation;
use qntt::solvers::mals::{MalsOptions, mals_with_history};
use qntt::{
    C64, CoeffTensor, Family, Irrep, PointGroup, Pooled, QnSpace, RankGrid, Scalar, Sector,
    SparseHamiltonian, TtVector, dot, max_ranks, path_counts, rayleigh_quotient, tt_randn,
    tt_state, tt_state_in,
};
use qntt_testkit::gad_mol::{coeff_vector, dense_h_in_sector, dense_sector_eigen, det_states};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::panic_message;
use crate::util::{
    basis_index, dense_eigen, dense_hamiltonian_matrix, dets_for, random_sym_t, random_sym_v,
};

const P5: [usize; 5] = [3, 0, 4, 1, 2];
const P6: [usize; 6] = [4, 2, 0, 5, 1, 3];

fn occ(d: usize, sites: &[usize]) -> Vec<bool> {
    let mut o = vec![false; d];
    for &s in sites {
        o[s] = true;
    }
    o
}

#[derive(Clone)]
struct Fixture {
    label: &'static str,
    space: QnSpace,
    up: Vec<bool>,
    dn: Vec<bool>,
}

impl Fixture {
    fn plain(label: &'static str, d: usize, up: &[usize], dn: &[usize]) -> Self {
        let (up, dn) = (occ(d, up), occ(d, dn));
        let nup = up.iter().filter(|&&o| o).count();
        let ndn = dn.iter().filter(|&&o| o).count();
        Fixture {
            label,
            space: QnSpace::new(d, nup, ndn),
            up,
            dn,
        }
    }
}

fn c1_fixtures() -> Vec<Fixture> {
    vec![
        Fixture::plain("(3,3) d=6 closed", 6, &[0, 1, 2], &[0, 1, 2]),
        Fixture::plain("(3,3) d=6 open", 6, &[0, 2, 4], &[1, 2, 5]),
        Fixture::plain("(2,1) d=5", 5, &[1, 3], &[0]),
        Fixture::plain("(3,0) d=6 CDW", 6, &[0, 2, 4], &[]),
    ]
}

fn as_orders(d: usize) -> Vec<Vec<usize>> {
    let fixed: Vec<usize> = match d {
        5 => P5.to_vec(),
        6 => P6.to_vec(),
        4 => vec![2, 0, 3, 1],
        _ => unreachable!(),
    };
    vec![(0..d).collect(), (0..d).rev().collect(), fixed]
}

fn full_cap(d: usize) -> usize {
    1usize << (2 * d)
}

#[inline]
fn digit(cfg: usize, b: usize, k: usize) -> usize {
    (cfg >> (2 * (k - b))) & 3
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Key {
    level: i64,
    n_dev: usize,
    tie: Vec<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Decided {
    Level,
    NDev,
    Tie,
}

fn decided_by(lo: &Key, hi: &Key) -> Decided {
    assert!(lo < hi, "decided_by: {lo:?} !< {hi:?}");
    if lo.level != hi.level {
        Decided::Level
    } else if lo.n_dev != hi.n_dev {
        Decided::NDev
    } else {
        Decided::Tie
    }
}

struct Ctx {
    space: QnSpace,
    d: usize,
    hf: Vec<usize>,
    as_order: Vec<usize>,
    pos: Vec<usize>,
    maxr: Vec<RankGrid>,
}

impl Ctx {
    fn new(space: QnSpace, up: &[bool], dn: &[bool], as_order: &[usize]) -> Self {
        let d = space.d;
        Ctx {
            space,
            d,
            hf: (0..d)
                .map(|k| up[k] as usize + 2 * dn[k] as usize)
                .collect(),
            as_order: as_order.to_vec(),
            pos: inverse_permutation(as_order),
            maxr: max_ranks(&space),
        }
    }

    fn last(&self) -> Sector {
        Sector::with_irrep(self.space.nup, self.space.ndn, self.space.target)
    }

    fn hf_tail(&self, b: usize) -> usize {
        (b..self.d).map(|k| self.hf[k] << (2 * (k - b))).sum()
    }

    fn try_sector_of(&self, cfg: usize, b: usize) -> Option<Sector> {
        let mut t = self.last();
        for k in (b..self.d).rev() {
            t = Family::from_physical_index(digit(cfg, b, k))
                .unapply(t, self.space.pg.irrep_of(k))?;
        }
        Some(t)
    }

    fn sector_of(&self, cfg: usize, b: usize) -> Sector {
        let s = self
            .try_sector_of(cfg, b)
            .unwrap_or_else(|| panic!("bond {b}: configuration {cfg} overfills the space"));
        let (mut nu, mut nd, mut g) = (0usize, 0usize, Irrep::IDENTITY);
        for k in b..self.d {
            let dg = digit(cfg, b, k);
            nu += dg & 1;
            nd += dg >> 1;
            if dg == 1 || dg == 2 {
                g = g.fuse(self.space.pg.irrep_of(k));
            }
        }
        let hand = Sector::with_irrep(
            self.space.nup - nu,
            self.space.ndn - nd,
            self.space.target.fuse(g),
        );
        assert_eq!(
            s, hand,
            "unapply chain vs hand sector at bond {b}, cfg {cfg}"
        );
        s
    }

    fn key(&self, cfg: usize, b: usize) -> Key {
        let mut level = -1i64;
        let mut n_dev = 0usize;
        for k in b..self.d {
            if digit(cfg, b, k) != self.hf[k] {
                n_dev += 1;
                level = level.max(self.pos[k] as i64);
            }
        }
        let tie = self
            .as_order
            .iter()
            .filter(|&&o| o >= b)
            .map(|&o| digit(cfg, b, o))
            .collect();
        Key { level, n_dev, tie }
    }
}

fn dense_frames<T: Scalar>(x: &TtVector<T>) -> Vec<HashMap<Sector, Vec<Vec<T>>>> {
    let d = x.space.d;
    let last = Sector::with_irrep(x.space.nup, x.space.ndn, x.space.target);
    assert_eq!(x.ranks[d][last], 1, "dense_frames: unit right boundary");
    let mut fr: Vec<HashMap<Sector, Vec<Vec<T>>>> = (0..=d).map(|_| HashMap::new()).collect();
    fr[d].insert(last, vec![vec![T::one()]]);
    for b in (0..d).rev() {
        let g = x.space.pg.irrep_of(b);
        let len = 1usize << (2 * (d - b));
        let mut cur = HashMap::new();
        for (s, &r) in x.ranks[b].iter() {
            if r == 0 {
                continue;
            }
            let mut rows = vec![vec![T::zero(); len]; r];
            for f in Family::ALL {
                let Some((rr, cc)) = x.cores[b].block_dims(f, s) else {
                    continue;
                };
                if cc == 0 {
                    continue;
                }
                assert_eq!(
                    rr, r,
                    "core {b} block ({f:?}, {s:?}) has {rr} rows, rank {r}"
                );
                let t = f.apply(s, g);
                let next = fr[b + 1]
                    .get(&t)
                    .unwrap_or_else(|| panic!("bond {}: no frames in {t:?}", b + 1));
                assert_eq!(next.len(), cc, "core {b} block ({f:?}, {s:?}) width");
                let mut blk = vec![T::zero(); rr * cc];
                x.cores[b].block_copy_out(f, s, &mut blk);
                for j in 0..cc {
                    for i in 0..rr {
                        let a = blk[j * rr + i];
                        if a == T::zero() {
                            continue;
                        }
                        for (idx, &v) in next[j].iter().enumerate() {
                            rows[i][f as usize + 4 * idx] += a * v;
                        }
                    }
                }
            }
            cur.insert(s, rows);
        }
        fr[b] = cur;
    }
    fr
}

fn frame_config<T: Scalar>(v: &[T]) -> Option<usize> {
    let mut hit = None;
    for (i, &a) in v.iter().enumerate() {
        if a != T::zero() {
            if hit.is_some() || a != T::one() {
                return None;
            }
            hit = Some(i);
        }
    }
    hit
}

type Configs = Vec<Vec<(usize, Sector)>>;

fn frame_configs<T: Scalar>(
    x: &TtVector<T>,
    fr: &[HashMap<Sector, Vec<Vec<T>>>],
    ctx: &Ctx,
    label: &str,
) -> Configs {
    let d = x.space.d;
    let mut out: Configs = vec![Vec::new(); d + 1];
    for b in 1..=d {
        for (s, &r) in x.ranks[b].iter() {
            if r == 0 {
                continue;
            }
            let rows = &fr[b][&s];
            assert_eq!(rows.len(), r, "{label}: bond {b} {s:?}");
            for i in 0..r {
                for j in 0..r {
                    let mut g = T::zero();
                    for (a, c) in rows[i].iter().zip(&rows[j]) {
                        g += a.conj_s() * *c;
                    }
                    let want = if i == j { T::one() } else { T::zero() };
                    assert!(
                        g == want,
                        "{label}: bond {b} {s:?}: frame Gram [{i}][{j}] = {g:?}, want {want:?} \
                         (D2: frames are orthonormal)"
                    );
                }
                let cfg = frame_config(&rows[i]).unwrap_or_else(|| {
                    panic!(
                        "{label}: bond {b} {s:?} frame {i} is not a single basis vector with \
                         coefficient 1 (D2: frames are determinants)"
                    )
                });
                assert_eq!(
                    ctx.sector_of(cfg, b),
                    s,
                    "{label}: bond {b} frame cfg {cfg} sits in the wrong sector"
                );
                out[b].push((cfg, s));
            }
        }
    }
    out
}

fn b_min(space: &QnSpace) -> usize {
    let (l, r) = path_counts(space);
    let d = space.d;
    let ok = |b: usize| l[b].iter().all(|(s, &lv)| lv == 0 || r[b][s] <= lv);
    let mut b = d;
    while b >= 1 && ok(b) {
        b -= 1;
    }
    b + 1
}

struct C1Stats {
    max_total: usize,
    dead: usize,
}

fn c1_arm<T: Scalar>(
    fx: &Fixture,
    as_order: &[usize],
    cap: usize,
) -> (TtVector<T>, Configs, C1Stats) {
    let space = fx.space;
    let d = space.d;
    let label = format!(
        "{} as={as_order:?} cap={cap} {}",
        fx.label,
        if T::IS_COMPLEX { "c64" } else { "f64" }
    );
    let ctx = Ctx::new(space, &fx.up, &fx.dn, as_order);
    let x: TtVector<T> = deas_start_in(space, &fx.up, &fx.dn, as_order, Pooled(cap));
    assert!(x.check(), "{label}: check()");
    assert_eq!(x.space, space, "{label}: space");

    assert_eq!(x.ortho, Some(0), "{label}: D1 ortho == Some(0)");
    let dense = x.to_dense();
    let want = tt_state_in::<T>(space, &fx.up, &fx.dn).to_dense();
    let hf_idx = basis_index(&fx.up, &fx.dn);
    assert!(want[hf_idx] == T::one(), "{label}: tt_state_in oracle");
    for (i, (a, w)) in dense.iter().zip(&want).enumerate() {
        assert!(
            a == w,
            "{label}: D1 to_dense[{i}] = {a:?}, HF determinant has {w:?}"
        );
    }
    let mut y = x.clone();
    y.ortho = None;
    assert!(
        dot(&y, &y) == T::one(),
        "{label}: ⟨x, x⟩ = {:?}",
        dot(&y, &y)
    );
    assert!(x.norm() == 1.0, "{label}: ‖x‖ = {}", x.norm());

    let fr = dense_frames(&x);
    let whole = &fr[0][&Sector::new(0, 0)];
    assert_eq!(whole.len(), 1);
    assert!(
        whole[0] == dense,
        "{label}: dense_frames' bond-0 contraction disagrees with to_dense"
    );
    let cfgs = frame_configs(&x, &fr, &ctx, &label);

    let s0 = Sector::new(0, 0);
    let hf0 = Family::from_physical_index(ctx.hf[0]);
    let hf1 = ctx.hf_tail(1);
    let mut ones = 0;
    for f in Family::ALL {
        let Some((rr, cc)) = x.cores[0].block_dims(f, s0) else {
            continue;
        };
        assert!(
            rr == 1 || cc == 0,
            "{label}: core 0 block ({f:?}) has {rr} rows"
        );
        let mut blk = vec![T::zero(); rr * cc];
        x.cores[0].block_copy_out(f, s0, &mut blk);
        let t = f.apply(s0, space.pg.irrep_of(0));
        for j in 0..cc {
            let is_hf = f == hf0 && frame_config(&fr[1][&t][j]).is_some_and(|c| c == hf1);
            if is_hf {
                assert!(
                    blk[j] == T::one(),
                    "{label}: core 0 HF amplitude {:?}",
                    blk[j]
                );
                ones += 1;
            } else {
                assert!(
                    blk[j] == T::zero(),
                    "{label}: core 0 ({f:?}, col {j}) = {:?}, expected 0 (D2)",
                    blk[j]
                );
            }
        }
    }
    assert_eq!(ones, 1, "{label}: core 0 must carry exactly one amplitude");

    for b in 0..=d {
        if (1..d).contains(&b) {
            assert!(
                x.ranks[b].total() <= cap,
                "{label}: bond {b} total rank {} > cap {cap}",
                x.ranks[b].total()
            );
        }
        for (s, &r) in x.ranks[b].iter() {
            assert!(
                r <= ctx.maxr[b][s],
                "{label}: bond {b} {s:?} rank {r} > max_ranks {}",
                ctx.maxr[b][s]
            );
        }
    }
    assert_eq!(x.ranks[d][ctx.last()], 1);
    assert_eq!(x.ranks[d].total(), 1);

    let sets: Vec<HashSet<(usize, Sector)>> =
        cfgs.iter().map(|v| v.iter().copied().collect()).collect();
    for b in 1..d {
        let g = space.pg.irrep_of(b);
        for &(c, s) in &cfgs[b] {
            let f = Family::from_physical_index(digit(c, b, b));
            let tail = (c >> 2, f.apply(s, g));
            assert!(
                sets[b + 1].contains(&tail),
                "{label}: bond {b} frame {c} has tail {tail:?} outside R_{} (D3 nested)",
                b + 1
            );
        }
    }
    let mut dead = 0;
    for b in 1..d {
        let g = space.pg.irrep_of(b);
        let tails: HashSet<usize> = cfgs[b].iter().map(|&(c, _)| c >> 2).collect();
        for &(c1, t) in &cfgs[b + 1] {
            let extends = Family::ALL
                .iter()
                .any(|f| f.unapply(t, g).is_some_and(|s| ctx.maxr[b][s] > 0));
            assert!(
                extends,
                "{label}: bond {} frame {c1} ({t:?}) extends to no reachable bond-{b} sector",
                b + 1
            );
            if !tails.contains(&c1) {
                dead += 1;
            }
        }
    }
    for b in 1..=d {
        let h = ctx.hf_tail(b);
        assert!(
            sets[b].contains(&(h, ctx.sector_of(h, b))),
            "{label}: HF tail {h} missing from R_{b} (D3)"
        );
    }

    let x2: TtVector<T> = deas_start_in(space, &fx.up, &fx.dn, as_order, Pooled(cap));
    assert_eq!(x2.ranks, x.ranks, "{label}: second build's ranks");
    assert_eq!(x2.ortho, x.ortho);
    for k in 0..d {
        for (s, _) in x.ranks[k].iter() {
            for f in Family::ALL {
                let dims = x.cores[k].block_dims(f, s);
                assert_eq!(dims, x2.cores[k].block_dims(f, s), "{label}: core {k} dims");
                if let Some((rr, cc)) = dims {
                    let (mut a, mut b2) = (vec![T::zero(); rr * cc], vec![T::zero(); rr * cc]);
                    x.cores[k].block_copy_out(f, s, &mut a);
                    x2.cores[k].block_copy_out(f, s, &mut b2);
                    assert!(
                        a == b2,
                        "{label}: core {k} ({f:?}, {s:?}) differs between builds"
                    );
                }
            }
        }
    }

    let max_total = x.ranks.iter().map(|g| g.total()).max().unwrap();
    (x, cfgs, C1Stats { max_total, dead })
}

fn assert_full_cap_law<T: Scalar>(
    fx: &Fixture,
    x: &TtVector<T>,
    cfgs: &Configs,
    as_order: &[usize],
) {
    let space = fx.space;
    let d = space.d;
    let ctx = Ctx::new(space, &fx.up, &fx.dn, as_order);
    let label = format!("{} as={as_order:?} full cap", fx.label);
    for b in 0..=d {
        assert!(
            x.ranks[b] == ctx.maxr[b],
            "{label}: bond {b} ranks do not saturate max_ranks (D5 full cap; K1/K2)"
        );
    }
    let bm = b_min(&space);
    assert!(bm < d, "{label}: b_min = {bm} makes the set law vacuous");
    for b in bm..=d {
        let mut all: HashSet<(usize, Sector)> = HashSet::new();
        for c in 0..(1usize << (2 * (d - b))) {
            if let Some(s) = ctx.try_sector_of(c, b)
                && space.contains(b, s)
                && ctx.maxr[b][s] > 0
            {
                all.insert((c, s));
            }
        }
        let got: HashSet<(usize, Sector)> = cfgs[b].iter().copied().collect();
        assert!(
            got == all,
            "{label}: bond {b} ≥ b_min = {bm}: frames are not every reachable configuration \
             ({} kept, {} reachable)",
            got.len(),
            all.len()
        );
    }
}

fn c1_fixture<T: Scalar>(fx: &Fixture) -> usize {
    let d = fx.space.d;
    let mut max_full = 0;
    for aso in as_orders(d) {
        let ctx = Ctx::new(fx.space, &fx.up, &fx.dn, &aso);
        for cap in [1, 2, 3, 5, 8, full_cap(d)] {
            let (x, cfgs, st) = c1_arm::<T>(fx, &aso, cap);
            let label = format!("{} as={aso:?} cap={cap}", fx.label);
            check_priority(&x, &ctx, cap, &label);
            if cap == 1 {
                let hf = tt_state_in::<T>(fx.space, &fx.up, &fx.dn);
                assert_eq!(
                    x.ranks, hf.ranks,
                    "{}: cap 1 must be the determinant's rank grid",
                    fx.label
                );
            }
            if cap == full_cap(d) {
                assert_full_cap_law(fx, &x, &cfgs, &aso);
                max_full = max_full.max(st.max_total);
            }
            println!(
                "tord_c1 {} {} as={aso:?} cap={cap}: ranks {:?}, literal-D3 dead frames {} (K1)",
                fx.label,
                if T::IS_COMPLEX { "c64" } else { "f64" },
                x.ranks.iter().map(|g| g.total()).collect::<Vec<_>>(),
                st.dead
            );
        }
    }
    max_full
}

#[test]
fn tord_c1_value_and_frames_f64() {
    for fx in c1_fixtures() {
        println!("tord_c1 {}: b_min = {}", fx.label, b_min(&fx.space));
        let m = c1_fixture::<f64>(&fx);
        println!(
            "tord_c1 {}: max total bond rank at full cap = {m}",
            fx.label
        );
        assert!(
            m > 4,
            "{}: full-cap start has max bond rank {m} ≤ 4 (K1)",
            fx.label
        );
    }
}

#[test]
fn tord_c1_value_and_frames_c64() {
    for fx in c1_fixtures() {
        let m = c1_fixture::<C64>(&fx);
        assert!(
            m > 4,
            "{}: full-cap start has max bond rank {m} ≤ 4 (K1)",
            fx.label
        );
    }
}

#[test]
fn tord_c1_deas_start_delegates() {
    for fx in c1_fixtures() {
        let d = fx.space.d;
        for aso in as_orders(d) {
            for cap in [3, full_cap(d)] {
                let a: TtVector<f64> = deas_start(&fx.up, &fx.dn, &aso, Pooled(cap));
                let b: TtVector<f64> = deas_start_in(fx.space, &fx.up, &fx.dn, &aso, Pooled(cap));
                assert_eq!(a.space, b.space, "{}", fx.label);
                assert_eq!(a.ranks, b.ranks, "{}", fx.label);
                assert_eq!(a.ortho, Some(0), "{}", fx.label);
                assert!(a.to_dense() == b.to_dense(), "{}", fx.label);
                for k in 0..d {
                    for (s, _) in a.ranks[k].iter() {
                        for f in Family::ALL {
                            let dims = a.cores[k].block_dims(f, s);
                            assert_eq!(dims, b.cores[k].block_dims(f, s));
                            if let Some((r, c)) = dims {
                                let (mut u, mut v) = (vec![0.0; r * c], vec![0.0; r * c]);
                                a.cores[k].block_copy_out(f, s, &mut u);
                                b.cores[k].block_copy_out(f, s, &mut v);
                                assert!(u == v, "{}: core {k} ({f:?}, {s:?})", fx.label);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn tord_c1_frame_oracle_self_check() {
    fn arm<T: Scalar>(space: QnSpace, seed: u64) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut x: TtVector<T> = tt_randn(space, &max_ranks(&space), &mut rng);
        let dense = x.to_dense();
        let scale = dense.iter().map(|a| a.abs_s()).fold(0.0, f64::max);
        let fr = dense_frames(&x);
        let err = fr[0][&Sector::new(0, 0)][0]
            .iter()
            .zip(&dense)
            .map(|(a, b)| (*a - *b).abs_s())
            .fold(0.0, f64::max);
        assert!(
            err <= 1e-13 * scale,
            "bond-0 contraction vs to_dense: {err:e} (scale {scale:e})"
        );
        x.right_orthogonalize(true);
        let fr = dense_frames(&x);
        let mut non_unit = 0;
        for b in 1..=space.d {
            for (s, &r) in x.ranks[b].iter() {
                if r == 0 {
                    continue;
                }
                let rows = &fr[b][&s];
                for i in 0..r {
                    for j in 0..r {
                        let mut g = T::zero();
                        for (a, c) in rows[i].iter().zip(&rows[j]) {
                            g += a.conj_s() * *c;
                        }
                        let want = if i == j { T::one() } else { T::zero() };
                        assert!(
                            (g - want).abs_s() <= 1e-13,
                            "Gram b={b} {s:?} [{i}][{j}] = {g:?}"
                        );
                    }
                    if frame_config(&rows[i]).is_none() {
                        non_unit += 1;
                    }
                }
            }
        }
        assert!(
            non_unit > 0,
            "random right-orthonormal frames all passed the unit-vector test"
        );
    }
    arm::<f64>(QnSpace::new(5, 2, 1), 0xC1_0001);
    arm::<C64>(QnSpace::new(5, 2, 1), 0xC1_0002);
    arm::<f64>(QnSpace::new(6, 3, 0), 0xC1_0003);
}

#[derive(Default, Debug)]
struct PriorityStats {
    excluded: usize,
    pooled: usize,
    sat_below_max: usize,
    by_level: usize,
    by_ndev: usize,
    by_tie: usize,
}

impl PriorityStats {
    fn add(&mut self, o: &PriorityStats) {
        self.excluded += o.excluded;
        self.pooled += o.pooled;
        self.sat_below_max += o.sat_below_max;
        self.by_level += o.by_level;
        self.by_ndev += o.by_ndev;
        self.by_tie += o.by_tie;
    }
    fn count(&mut self, dcd: Decided) {
        match dcd {
            Decided::Level => self.by_level += 1,
            Decided::NDev => self.by_ndev += 1,
            Decided::Tie => self.by_tie += 1,
        }
    }
}

fn check_priority<T: Scalar>(x: &TtVector<T>, ctx: &Ctx, cap: usize, label: &str) -> PriorityStats {
    let d = ctx.d;
    let fr = dense_frames(x);
    let cfgs = frame_configs(x, &fr, ctx, label);
    let mut st = PriorityStats::default();
    for b in 1..d {
        let g = ctx.space.pg.irrep_of(b);
        let kept: HashSet<usize> = cfgs[b].iter().map(|&(c, _)| c).collect();
        let mut cands: Vec<(usize, Sector)> = Vec::new();
        for &(c1, t) in &cfgs[b + 1] {
            for f in Family::ALL {
                if let Some(s) = f.unapply(t, g)
                    && ctx.maxr[b][s] > 0
                {
                    cands.push((f as usize + 4 * c1, s));
                }
            }
        }
        let cand_set: HashSet<usize> = cands.iter().map(|&(c, _)| c).collect();
        for c in &kept {
            assert!(
                cand_set.contains(c),
                "{label}: bond {b} kept frame {c} is not a candidate"
            );
        }
        let total = kept.len();
        assert!(total <= cap, "{label}: bond {b} keeps {total} > cap {cap}");
        let kept_keys: Vec<(Key, Sector)> =
            cfgs[b].iter().map(|&(c, s)| (ctx.key(c, b), s)).collect();
        let n_in = |s: Sector| kept_keys.iter().filter(|(_, ks)| *ks == s).count();
        let max_kept = kept_keys.iter().map(|(k, _)| k.clone()).max();

        let mut pooled_min: Option<Key> = None;
        let mut sat_min: HashMap<Sector, Key> = HashMap::new();
        for &(e, se) in &cands {
            if kept.contains(&e) {
                continue;
            }
            st.excluded += 1;
            let ke = ctx.key(e, b);
            let pe = (ke.level, ke.n_dev);
            let m = ctx.maxr[b][se];
            let in_s = n_in(se);
            assert!(
                in_s <= m,
                "{label}: bond {b} {se:?} keeps {in_s} > max_ranks {m}"
            );
            let sat = in_s == m
                && kept_keys
                    .iter()
                    .filter(|(_, s)| *s == se)
                    .all(|(k, _)| *k < ke);
            let pool = total == cap && kept_keys.iter().all(|(k, _)| *k < ke);
            assert!(
                sat || pool,
                "{label}: bond {b}: excluded candidate {e} ({se:?}, key {ke:?}) is neither \
                 behind a saturated sector ({in_s}/{m}) nor behind the pooled cap ({total}/{cap}) \
                 (D4)"
            );
            let sat_p = in_s == m
                && kept_keys
                    .iter()
                    .filter(|(_, s)| *s == se)
                    .all(|(k, _)| (k.level, k.n_dev) <= pe);
            let pool_p = total == cap && kept_keys.iter().all(|(k, _)| (k.level, k.n_dev) <= pe);
            assert!(
                sat_p || pool_p,
                "{label}: bond {b}: excluded {e} violates the partial-key order (level, n_dev) \
                 = {pe:?} (D4, alphabet-free)"
            );
            if in_s < m {
                st.pooled += 1;
                if pooled_min.as_ref().is_none_or(|k| ke < *k) {
                    pooled_min = Some(ke.clone());
                }
            } else {
                if max_kept.as_ref().is_some_and(|k| ke < *k) {
                    st.sat_below_max += 1;
                }
                let slot = sat_min.entry(se).or_insert_with(|| ke.clone());
                if ke < *slot {
                    *slot = ke.clone();
                }
            }
        }
        if let (Some(lo), Some(hi)) = (max_kept.as_ref(), pooled_min.as_ref()) {
            st.count(decided_by(lo, hi));
        }
        for (s, hi) in &sat_min {
            let lo = kept_keys
                .iter()
                .filter(|(_, ks)| ks == s)
                .map(|(k, _)| k.clone())
                .max()
                .expect("a saturated sector holds frames");
            st.count(decided_by(&lo, hi));
        }
    }
    st
}

#[test]
fn tord_c2_priority_is_the_deas_key() {
    let a = Fixture::plain("(3,3) d=6 closed", 6, &[0, 1, 2], &[0, 1, 2]);
    let b = Fixture::plain("(2,1) d=5", 5, &[1, 3], &[0]);
    let runs: Vec<(&Fixture, Vec<usize>, Vec<usize>)> = vec![
        (&a, (0..6).collect(), vec![3, 6, 10]),
        (&a, P6.to_vec(), vec![3, 6, 10]),
        (&b, P5.to_vec(), vec![2, 5, full_cap(5)]),
    ];
    let mut tot = PriorityStats::default();
    for (fx, aso, caps) in runs {
        let ctx = Ctx::new(fx.space, &fx.up, &fx.dn, &aso);
        for cap in caps {
            let label = format!("{} as={aso:?} cap={cap}", fx.label);
            let x: TtVector<f64> = deas_start_in(fx.space, &fx.up, &fx.dn, &aso, Pooled(cap));
            let st = check_priority(&x, &ctx, cap, &label);
            println!("tord_c2 {label}: {st:?}");
            tot.add(&st);
        }
    }
    println!("tord_c2 total: {tot:?}");
    assert!(tot.pooled > 0, "no bond where the pooled cap binds");
    assert!(
        tot.sat_below_max > 0,
        "no saturated-sector exclusion below a kept key (K3 witness)"
    );
    assert!(tot.by_level > 0, "no boundary decided by level");
    assert!(tot.by_ndev > 0, "no boundary decided by n_dev");
    assert!(tot.by_tie > 0, "no boundary decided by the tie");
}

#[test]
fn tord_c2_as_order_locality() {
    let fx = Fixture::plain("(3,3) d=6 closed", 6, &[0, 1, 2], &[0, 1, 2]);
    let a: Vec<usize> = (0..6).collect();
    let a2: Vec<usize> = vec![0, 1, 2, 4, 3, 5];
    let lo_orbital = 3;
    let mut differs_somewhere = false;
    for cap in [4, 6, 10] {
        let ca = Ctx::new(fx.space, &fx.up, &fx.dn, &a);
        let ca2 = Ctx::new(fx.space, &fx.up, &fx.dn, &a2);
        let xa: TtVector<f64> = deas_start_in(fx.space, &fx.up, &fx.dn, &a, Pooled(cap));
        let xa2: TtVector<f64> = deas_start_in(fx.space, &fx.up, &fx.dn, &a2, Pooled(cap));
        let fa = frame_configs(&xa, &dense_frames(&xa), &ca, &format!("A cap={cap}"));
        let fa2 = frame_configs(&xa2, &dense_frames(&xa2), &ca2, &format!("A′ cap={cap}"));
        let mut pattern = Vec::new();
        for b in 1..=6 {
            let sa: HashSet<(usize, Sector)> = fa[b].iter().copied().collect();
            let sa2: HashSet<(usize, Sector)> = fa2[b].iter().copied().collect();
            let same = sa == sa2;
            pattern.push(same);
            if b > lo_orbital {
                assert!(
                    same,
                    "cap {cap}: bond {b} frame sets differ under an AS swap of orbitals 3, 4, \
                     which bond {b}'s block does not both contain"
                );
            } else if !same {
                differs_somewhere = true;
            }
        }
        println!("tord_c2 locality cap={cap}: bond 1..=6 equal = {pattern:?}");
    }
    assert!(
        differs_somewhere,
        "the AS swap changed no frame set at bonds 1..=3 for any cap"
    );
}

struct C3Fixture {
    label: &'static str,
    d: usize,
    nup: usize,
    ndn: usize,
    up: Vec<bool>,
    dn: Vec<bool>,
    t: CoeffTensor<f64>,
    v: CoeffTensor<f64>,
    as_alt: Vec<usize>,
}

fn c3_ring() -> C3Fixture {
    let d = 6;
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for i in 0..d {
        let j = (i + 1) % d;
        t.set(&[i, j], 1.0);
        t.set(&[j, i], 1.0);
        v.set(&[i, i, j, j], 1.0);
        v.set(&[j, j, i, i], 1.0);
    }
    C3Fixture {
        label: "t-V ring d=6 (3,0)",
        d,
        nup: 3,
        ndn: 0,
        up: occ(d, &[0, 2, 4]),
        dn: occ(d, &[]),
        t,
        v,
        as_alt: P6.to_vec(),
    }
}

fn c3_random() -> C3Fixture {
    let d = 5;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEA5_C003);
    let t = random_sym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    C3Fixture {
        label: "random d=5 (2,2)",
        d,
        nup: 2,
        ndn: 2,
        up: occ(d, &[0, 1]),
        dn: occ(d, &[0, 1]),
        t,
        v,
        as_alt: P5.to_vec(),
    }
}

fn one_iteration() -> MalsOptions {
    MalsOptions {
        eps: 1e-12,
        bond_dims: vec![0],
        noise: vec![0.0],
        max_iter: 1,
        conv_tol: Some(0.0),
        ..Default::default()
    }
}

#[test]
fn tord_c3_one_iteration_reaches_the_ground_state() {
    for fx in [c3_ring(), c3_random()] {
        let d = fx.d;
        let space = QnSpace::new(d, fx.nup, fx.ndn);
        let h = SparseHamiltonian::new(space, &fx.t, &fx.v, None, f64::EPSILON);
        let dets = det_states::<f64>(d, fx.nup, fx.ndn);
        let n = dets.len();
        let mat = dense_h_in_sector(&dets, &h);
        let (ev, _) = dense_sector_eigen(&mat, n);
        let (e0, gap) = (ev[0], ev[1] - ev[0]);
        let (ev_ind, _) = dense_eigen(&dense_hamiltonian_matrix(
            d,
            &fx.t,
            &fx.v,
            &dets_for(d, fx.nup, fx.ndn),
        ));
        println!(
            "tord_c3 {}: dim {n}, E0 = {e0:.12}, gap = {gap:.6}",
            fx.label
        );
        assert!(
            (e0 - ev_ind[0]).abs() <= 1e-12 * e0.abs().max(1.0),
            "{}: E0 {e0} vs independent JW {}",
            fx.label,
            ev_ind[0]
        );
        assert!(
            gap > 1e-3,
            "{}: ground state degenerate, gap {gap:e}",
            fx.label
        );
        let e_hf = rayleigh_quotient(&h, &tt_state::<f64>(&fx.up, &fx.dn));
        let tol = 1e-7 * e0.abs().max(1.0);

        let mut c2_miss = false;
        for aso in [(0..d).collect::<Vec<usize>>(), fx.as_alt.clone()] {
            let label = format!("{} as={aso:?}", fx.label);
            let x0: TtVector<f64> = deas_start(&fx.up, &fx.dn, &aso, Pooled(full_cap(d)));
            assert_eq!(x0.ranks, max_ranks(&space), "{label}: full-cap start ranks");
            let e_start = rayleigh_quotient(&h, &x0);
            assert!(
                (e_start - e_hf).abs() <= 1e-13 * e_hf.abs().max(1.0),
                "{label}: start energy {e_start} vs E_HF {e_hf}"
            );
            let (_, x_same) = mals_with_history(
                &h,
                &x0,
                &MalsOptions {
                    max_iter: 0,
                    ..one_iteration()
                },
            );
            assert_eq!(
                x_same.ranks, x0.ranks,
                "{label}: zero-iteration run changed ranks"
            );
            assert!(
                x_same.to_dense() == x0.to_dense(),
                "{label}: zero-iteration run changed the start"
            );

            let (hist, x) = mals_with_history(&h, &x0, &one_iteration());
            assert_eq!(hist.sweeps.len(), 1, "{label}: exactly one iteration");
            let err = hist.lambda - e0;
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

            let (h_tt, _) =
                mals_with_history(&h, &tt_state::<f64>(&fx.up, &fx.dn), &one_iteration());
            let x2: TtVector<f64> = deas_start(&fx.up, &fx.dn, &aso, Pooled(2));
            let (h_c2, _) = mals_with_history(&h, &x2, &one_iteration());
            println!(
                "tord_c3 {label}: E_HF = {e_hf:.12}; DEAS(full) 1 it: λ − E0 = {err:.3e} (margin \
                 {:.1}× under {tol:.0e}), ‖r‖/gap = {:.3e}, ranks {:?}; tt_state 1 it: λ − E0 = \
                 {:.3e}; DEAS(cap 2) 1 it: λ − E0 = {:.3e}",
                tol / err.abs().max(f64::MIN_POSITIVE),
                res / gap,
                x.ranks.iter().map(|g| g.total()).collect::<Vec<_>>(),
                h_tt.lambda - e0,
                h_c2.lambda - e0,
            );
            assert!(
                err >= -1e-12 * e0.abs().max(1.0),
                "{label}: λ below E0 by {err:e}"
            );
            assert!(
                err.abs() <= tol,
                "{label}: one MALS iteration from the full-cap DEAS start misses E0 by {err:e} \
                 (> {tol:e})"
            );
            assert!(
                h_tt.lambda - e0 > 100.0 * tol,
                "{label}: one iteration from the rank-1 tt_state misses E0 by only {:e} \
                 (≤ 100·tol = {:e}) — the oracle would not need the DEAS frames",
                h_tt.lambda - e0,
                100.0 * tol
            );
            c2_miss |= h_c2.lambda - e0 > tol;
        }
        assert!(
            c2_miss,
            "{}: DEAS at cap 2 reaches E0 on every AS order — the cap contrast is vacuous",
            fx.label
        );
    }
}

#[test]
fn tord_c4_point_group() {
    let pg = PointGroup::new(1, &[0u8, 1, 1, 0].map(Irrep::new));
    let d = 4;
    let mk = |label, nup, ndn, tg: u8, up: &[usize], dn: &[usize]| Fixture {
        label,
        space: QnSpace::with_point_group(d, nup, ndn, pg, Irrep::new(tg)),
        up: occ(d, up),
        dn: occ(d, dn),
    };
    let fixtures = [
        mk("(2,1) G=2 target 1", 2, 1, 1, &[0, 1], &[0]),
        mk("(2,1) G=2 target 0", 2, 1, 0, &[0, 3], &[0]),
        mk("(2,2) G=2 target 1", 2, 2, 1, &[0, 1], &[0, 3]),
    ];
    let mut unreachable = 0;
    let mut both_irreps = false;
    let mut prio = PriorityStats::default();
    for fx in &fixtures {
        let maxr = max_ranks(&fx.space);
        for b in 0..=d {
            for s in fx.space.bond_sectors(b) {
                if fx.space.contains(b, s) && maxr[b][s] == 0 {
                    unreachable += 1;
                }
            }
        }
        for aso in as_orders(d) {
            for cap in [1, 3, full_cap(d)] {
                let (x, cfgs, _) = c1_arm::<f64>(fx, &aso, cap);
                if cap == 1 {
                    let hf = tt_state_in::<f64>(fx.space, &fx.up, &fx.dn);
                    assert_eq!(x.ranks, hf.ranks, "{}: cap 1", fx.label);
                }
                let ctx = Ctx::new(fx.space, &fx.up, &fx.dn, &aso);
                let label = format!("{} as={aso:?} cap={cap}", fx.label);
                let st = check_priority(&x, &ctx, cap, &label);
                if cap == 3 {
                    prio.add(&st);
                }
                if cap == full_cap(d) {
                    assert_full_cap_law(fx, &x, &cfgs, &aso);
                    for b in 1..d {
                        let irreps: HashSet<Irrep> = x.ranks[b]
                            .iter()
                            .filter(|(_, r)| **r > 0)
                            .map(|(s, _)| s.irrep)
                            .collect();
                        both_irreps |= irreps.len() == 2;
                    }
                    let (xc, cfc, _) = c1_arm::<C64>(fx, &aso, cap);
                    assert_full_cap_law(fx, &xc, &cfc, &aso);
                }
                println!(
                    "tord_c4 {} as={aso:?} cap={cap}: ranks {:?}",
                    fx.label,
                    x.ranks.iter().map(|g| g.total()).collect::<Vec<_>>()
                );
            }
        }
    }
    println!("tord_c4: {unreachable} admissible-but-unreachable sectors; cap-3 priority {prio:?}");
    assert!(
        unreachable > 0,
        "no admissible-but-unreachable irrep sector: the irrep mask is untested"
    );
    assert!(both_irreps, "no bond carries rank on both irreps");
    assert!(prio.excluded > 0, "cap 3 excluded nothing at |G| = 2");
}

fn assert_panics_with(f: impl FnOnce(), phrases: &[&str]) {
    let msg = panic_message(f);
    println!("refusal: {msg}");
    for p in phrases {
        assert!(msg.contains(p), "panic message lacks {p:?}: {msg}");
    }
}

#[test]
fn tord_c5_active_space_order() {
    assert_eq!(
        active_space_order(&[0.3, 0.7, 0.3, 0.7, 0.1]),
        vec![1, 3, 0, 2, 4]
    );
    assert_eq!(active_space_order(&[0.5; 4]), vec![0, 1, 2, 3]);
    assert_eq!(active_space_order(&[0.0, -2e-16]), vec![0, 1]);
    assert_eq!(active_space_order(&[-2e-16, 0.0]), vec![1, 0]);
    let s1 = [0.11, 0.52, 0.03, 0.77, 0.29, 0.64];
    let aso = active_space_order(&s1);
    assert_eq!(aso, vec![3, 5, 1, 4, 0, 2]);
    let inv = inverse_permutation(&P6);
    let s1p: Vec<f64> = P6.iter().map(|&p| s1[p]).collect();
    let want: Vec<usize> = aso.iter().map(|&o| inv[o]).collect();
    assert_eq!(active_space_order(&s1p), want);
}

#[test]
fn tord_c5_active_space_order_rejects_non_finite() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_panics_with(
            || {
                let _ = active_space_order(&[0.1, 0.2, bad, 0.3]);
            },
            &["s1[2]", "not finite"],
        );
    }
}

fn d4_fixture() -> (QnSpace, Vec<bool>, Vec<bool>) {
    (QnSpace::new(4, 2, 1), occ(4, &[0, 1]), occ(4, &[0]))
}

#[test]
#[should_panic(expected = "occupation vectors must share length d")]
fn tord_c5_deas_start_rejects_unequal_occupations() {
    let _: TtVector<f64> = deas_start(&occ(4, &[0, 1]), &occ(3, &[0]), &[0, 1, 2, 3], Pooled(4));
}

#[test]
fn tord_c5_deas_start_in_rejects_occupation_length() {
    let (space, up, dn) = d4_fixture();
    assert_panics_with(
        || {
            let _: TtVector<f64> = deas_start_in(space, &up[..3], &dn, &[0, 1, 2, 3], Pooled(4));
        },
        &["occ_up", "d = 4"],
    );
    assert_panics_with(
        || {
            let _: TtVector<f64> = deas_start_in(space, &up, &dn[..3], &[0, 1, 2, 3], Pooled(4));
        },
        &["occ_dn", "d = 4"],
    );
}

#[test]
#[should_panic(expected = "has length 3")]
fn tord_c5_rejects_short_as_order() {
    let (space, up, dn) = d4_fixture();
    let _: TtVector<f64> = deas_start_in(space, &up, &dn, &[0, 1, 2], Pooled(4));
}

#[test]
#[should_panic(expected = "[1] = 4 is out of range")]
fn tord_c5_rejects_out_of_range_as_order() {
    let (space, up, dn) = d4_fixture();
    let _: TtVector<f64> = deas_start_in(space, &up, &dn, &[0, 4, 1, 2], Pooled(4));
}

#[test]
#[should_panic(expected = "[2] = 1 duplicates")]
fn tord_c5_rejects_duplicate_as_order() {
    let (space, up, dn) = d4_fixture();
    let _: TtVector<f64> = deas_start_in(space, &up, &dn, &[0, 1, 1, 2], Pooled(4));
}

#[test]
fn tord_c5_no_cap_is_full_environment() {
    let (space, up, dn) = d4_fixture();
    let aso = [0, 1, 2, 3];
    let none: TtVector<f64> = deas_start_in(space, &up, &dn, &aso, Pooled::NONE);
    let full: TtVector<f64> = deas_start_in(space, &up, &dn, &aso, Pooled(full_cap(space.d)));
    assert_eq!(
        none.ranks,
        max_ranks(&space),
        "Pooled::NONE must saturate max_ranks"
    );
    assert_eq!(none.ranks, full.ranks);
    assert_eq!(none.ortho, Some(0));
    assert!(none.to_dense() == full.to_dense());
    for k in 0..space.d {
        for (s, _) in none.ranks[k].iter() {
            for f in Family::ALL {
                let dims = none.cores[k].block_dims(f, s);
                assert_eq!(dims, full.cores[k].block_dims(f, s), "core {k} block dims");
                if let Some((r, c)) = dims {
                    let (mut u, mut v) = (vec![0.0; r * c], vec![0.0; r * c]);
                    none.cores[k].block_copy_out(f, s, &mut u);
                    full.cores[k].block_copy_out(f, s, &mut v);
                    assert!(u == v, "core {k} block ({f:?}, {s:?}) differs");
                }
            }
        }
    }
}

#[test]
#[should_panic(expected = "occupies (2, 2) electrons, space wants (2, 1)")]
fn tord_c5_rejects_wrong_electron_count() {
    let (space, up, _) = d4_fixture();
    let _: TtVector<f64> = deas_start_in(space, &up, &occ(4, &[0, 2]), &[0, 1, 2, 3], Pooled(4));
}

#[test]
#[should_panic(expected = "differs from the space target")]
fn tord_c5_rejects_wrong_irrep() {
    let pg = PointGroup::new(1, &[0u8, 1, 1, 0].map(Irrep::new));
    let space = QnSpace::with_point_group(4, 2, 1, pg, Irrep::new(0));
    let _: TtVector<f64> = deas_start_in(
        space,
        &occ(4, &[0, 1]),
        &occ(4, &[0]),
        &[0, 1, 2, 3],
        Pooled(4),
    );
}
