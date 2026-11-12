// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::path::Path;
use std::sync::OnceLock;

use qntt::CoeffTensor;
use qntt::core::weingarten::williamson_reduced;
use qntt::integrals::{Integrals, IntegralsOptions, load_fixture_dir};
use qntt::rotation::{expm_antisym, freeze_core, rotate_integrals};
use qntt::solvers::joint::cas::{CasPartition, cas_pairs};
use rand::{RngExt, SeedableRng};
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256PlusPlus;
use rustc_hash::FxHashMap;

use crate::joint::util::*;

#[inline]
fn so_of(site: usize, dn: bool) -> usize {
    2 * site + dn as usize
}

#[inline]
fn so_orb(p: usize) -> usize {
    p >> 1
}

#[inline]
fn so_dn(p: usize) -> bool {
    p & 1 == 1
}

type DetMask = u64;

fn det_phase_apply(mask: DetMask, ops: &[(usize, bool)]) -> Option<(DetMask, f64)> {
    let mut m = mask;
    let mut sign = 1.0f64;
    for &(p, creation) in ops.iter().rev() {
        let bit = 1u64 << p;
        let occupied = m & bit != 0;
        if creation == occupied {
            return None;
        }
        if (m & (bit - 1)).count_ones() % 2 == 1 {
            sign = -sign;
        }
        m ^= bit;
    }
    Some((m, sign))
}

fn pack_det(occ_up: &[bool], occ_dn: &[bool]) -> DetMask {
    let mut m = 0u64;
    for (k, (&u, &v)) in occ_up.iter().zip(occ_dn).enumerate() {
        if u {
            m |= 1u64 << so_of(k, false);
        }
        if v {
            m |= 1u64 << so_of(k, true);
        }
    }
    m
}

pub(crate) struct DetSpace {
    pub d: usize,
    pub nup: usize,
    pub ndn: usize,
    pub dets: Vec<(Vec<bool>, Vec<bool>)>,
    #[allow(dead_code)]
    pub masks: Vec<(u32, u32)>,
    so: Vec<DetMask>,
    index: FxHashMap<DetMask, usize>,
}

impl DetSpace {
    pub(crate) fn new(d: usize, nup: usize, ndn: usize) -> Self {
        assert!(
            d <= 31,
            "DetSpace: d = {d} exceeds the u64 spin-orbital mask"
        );
        let dets = dets_for(d, nup, ndn);
        let masks: Vec<(u32, u32)> = dets
            .iter()
            .map(|(ou, od)| {
                let bits = |o: &[bool]| {
                    o.iter()
                        .enumerate()
                        .filter(|&(_, &b)| b)
                        .fold(0u32, |acc, (k, _)| acc | (1u32 << k))
                };
                (bits(ou), bits(od))
            })
            .collect();
        let so: Vec<DetMask> = dets.iter().map(|(ou, od)| pack_det(ou, od)).collect();
        let mut index = FxHashMap::default();
        for (i, &m) in so.iter().enumerate() {
            let prev = index.insert(m, i);
            assert!(prev.is_none(), "DetSpace: duplicate determinant at {i}");
        }
        DetSpace {
            d,
            nup,
            ndn,
            dets,
            masks,
            so,
            index,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.dets.len()
    }

    #[allow(dead_code)]
    pub(crate) fn position(&self, mask: (u32, u32)) -> Option<usize> {
        let mut m = 0u64;
        for k in 0..self.d {
            if mask.0 >> k & 1 == 1 {
                m |= 1u64 << so_of(k, false);
            }
            if mask.1 >> k & 1 == 1 {
                m |= 1u64 << so_of(k, true);
            }
        }
        self.index.get(&m).copied()
    }

    pub(crate) fn cas_subset(&self, part: &CasPartition) -> (Vec<usize>, DetSpace) {
        assert_eq!(
            part.d(),
            self.d,
            "cas_subset: partition covers {} orbitals, space has {}",
            part.d(),
            self.d
        );
        let ni = part.n_core;
        let act = DetSpace::new(part.n_active, self.nup - ni, self.ndn - ni);
        let full: Vec<usize> = act
            .dets
            .iter()
            .map(|(ou, od)| {
                let mut up = vec![false; self.d];
                let mut dn = vec![false; self.d];
                for (u, v) in up.iter_mut().zip(dn.iter_mut()).take(ni) {
                    *u = true;
                    *v = true;
                }
                for (j, &b) in ou.iter().enumerate() {
                    up[ni + j] = b;
                }
                for (j, &b) in od.iter().enumerate() {
                    dn[ni + j] = b;
                }
                self.index
                    .get(&pack_det(&up, &dn))
                    .copied()
                    .expect("a CAS determinant must live in the full particle sector")
            })
            .collect();
        let mut seen: Vec<usize> = full.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            full.len(),
            "cas_subset: the map is not injective"
        );
        (full, act)
    }

    pub(crate) fn embed(&self, full_positions: &[usize], c_act: &[f64]) -> Vec<f64> {
        assert_eq!(full_positions.len(), c_act.len());
        let mut out = vec![0.0f64; self.len()];
        for (&p, &c) in full_positions.iter().zip(c_act) {
            out[p] = c;
        }
        out
    }
}

#[inline]
fn eri(v: &CoeffTensor<f64>, p: usize, q: usize, r: usize, s: usize) -> f64 {
    if so_dn(p) != so_dn(r) || so_dn(q) != so_dn(s) {
        return 0.0;
    }
    v.get(&[so_orb(p), so_orb(r), so_orb(q), so_orb(s)])
}

#[inline]
fn eri_as(v: &CoeffTensor<f64>, p: usize, q: usize, r: usize, s: usize) -> f64 {
    eri(v, p, q, r, s) - eri(v, p, q, s, r)
}

#[inline]
fn h1(t: &CoeffTensor<f64>, a: usize, b: usize) -> f64 {
    if so_dn(a) != so_dn(b) {
        return 0.0;
    }
    t.get(&[so_orb(a), so_orb(b)])
}

fn occ_virt(mask: DetMask, d: usize) -> (Vec<usize>, Vec<usize>) {
    let mut occ = Vec::new();
    let mut virt = Vec::new();
    for p in 0..2 * d {
        if mask >> p & 1 == 1 {
            occ.push(p);
        } else {
            virt.push(p);
        }
    }
    (occ, virt)
}

fn sc_apply_det(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    mask: DetMask,
    out: &mut impl FnMut(DetMask, f64),
) {
    let (occ, virt) = occ_virt(mask, d);

    let mut diag = 0.0f64;
    for &p in &occ {
        diag += h1(t, p, p);
    }
    for (ip, &p) in occ.iter().enumerate() {
        for &q in &occ[ip + 1..] {
            diag += eri_as(v, p, q, p, q);
        }
    }
    out(mask, diag);

    for &p in &occ {
        for &r in &virt {
            let mut el = h1(t, r, p);
            for &q in &occ {
                if q != p {
                    el += eri_as(v, r, q, p, q);
                }
            }
            if el != 0.0
                && let Some((m, ph)) = det_phase_apply(mask, &[(r, true), (p, false)])
            {
                out(m, ph * el);
            }
        }
    }

    for (ip, &p) in occ.iter().enumerate() {
        for &q in &occ[ip + 1..] {
            for (ir, &r) in virt.iter().enumerate() {
                for &s in &virt[ir + 1..] {
                    let el = eri_as(v, r, s, p, q);
                    if el != 0.0
                        && let Some((m, ph)) =
                            det_phase_apply(mask, &[(r, true), (s, true), (q, false), (p, false)])
                    {
                        out(m, ph * el);
                    }
                }
            }
        }
    }
}

pub(crate) fn sc_hamiltonian_matrix(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    sp: &DetSpace,
) -> Vec<Vec<f64>> {
    let n = sp.len();
    let mut m = vec![vec![0.0f64; n]; n];
    #[allow(clippy::needless_range_loop)]
    for col in 0..n {
        let src = sp.so[col];
        sc_apply_det(d, t, v, src, &mut |mask, c| {
            let row = *sp
                .index
                .get(&mask)
                .expect("Slater–Condon excitation left the particle sector");
            m[row][col] += c;
        });
    }
    m
}

pub(crate) fn apply_epq(d: usize, p: usize, q: usize, sp: &DetSpace, x: &[f64]) -> Vec<f64> {
    let _ = d;
    assert_eq!(x.len(), sp.len(), "apply_epq: vector length");
    let mut out = vec![0.0f64; sp.len()];
    for (j, &xj) in x.iter().enumerate() {
        if xj == 0.0 {
            continue;
        }
        for dn in [false, true] {
            if let Some((m, ph)) =
                det_phase_apply(sp.so[j], &[(so_of(p, dn), true), (so_of(q, dn), false)])
                && let Some(&i) = sp.index.get(&m)
            {
                out[i] += ph * xj;
            }
        }
    }
    out
}

pub(crate) fn apply_gpq(d: usize, p: usize, q: usize, sp: &DetSpace, x: &[f64]) -> Vec<f64> {
    let a = apply_epq(d, p, q, sp, x);
    let b = apply_epq(d, q, p, sp, x);
    a.iter().zip(&b).map(|(&x, &y)| x - y).collect()
}

#[derive(Clone, Default)]
struct SpVec {
    m: FxHashMap<DetMask, f64>,
}

impl SpVec {
    fn with_capacity(n: usize) -> Self {
        SpVec {
            m: FxHashMap::with_capacity_and_hasher(n, Default::default()),
        }
    }

    #[inline]
    fn add(&mut self, k: DetMask, c: f64) {
        if c != 0.0 {
            *self.m.entry(k).or_insert(0.0) += c;
        }
    }

    #[inline]
    fn get(&self, k: DetMask) -> f64 {
        self.m.get(&k).copied().unwrap_or(0.0)
    }

    fn len(&self) -> usize {
        self.m.len()
    }

    fn keys_sorted(&self) -> Vec<DetMask> {
        let mut k: Vec<DetMask> = self.m.keys().copied().collect();
        k.sort_unstable();
        k
    }

    fn dot(&self, other: &SpVec) -> f64 {
        let (a, b) = if self.len() <= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        a.keys_sorted().iter().map(|&k| a.get(k) * b.get(k)).sum()
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    fn axpy(&mut self, a: f64, other: &SpVec) {
        for k in other.keys_sorted() {
            self.add(k, a * other.get(k));
        }
    }
}

fn sparse_epq(p: usize, q: usize, x: &SpVec) -> SpVec {
    let mut out = SpVec::with_capacity(2 * x.len());
    for k in x.keys_sorted() {
        let c = x.get(k);
        for dn in [false, true] {
            if let Some((m, ph)) =
                det_phase_apply(k, &[(so_of(p, dn), true), (so_of(q, dn), false)])
            {
                out.add(m, ph * c);
            }
        }
    }
    out
}

fn sparse_gpq(p: usize, q: usize, x: &SpVec) -> SpVec {
    let mut out = sparse_epq(p, q, x);
    out.axpy(-1.0, &sparse_epq(q, p, x));
    out
}

fn sparse_h(d: usize, t: &CoeffTensor<f64>, v: &CoeffTensor<f64>, x: &SpVec) -> SpVec {
    let mut out = SpVec::with_capacity(8 * x.len());
    for k in x.keys_sorted() {
        let c = x.get(k);
        if c == 0.0 {
            continue;
        }
        sc_apply_det(d, t, v, k, &mut |mask, el| out.add(mask, el * c));
    }
    out
}

pub(crate) fn slice_leading(ints: &Integrals<f64>, m: usize) -> Integrals<f64> {
    assert!(
        m <= ints.norb,
        "slice_leading: m = {m} exceeds norb = {}",
        ints.norb
    );
    let mut t = CoeffTensor::<f64>::zeros(m, 2);
    for a in 0..m {
        for b in 0..m {
            t.set(&[a, b], ints.t.get(&[a, b]));
        }
    }
    let mut v = CoeffTensor::<f64>::zeros(m, 4);
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                for e in 0..m {
                    v.set(&[a, b, c, e], ints.v.get(&[a, b, c, e]));
                }
            }
        }
    }
    Integrals {
        t,
        v,
        e0: 0.0,
        norb: m,
        nelec: ints.nelec,
        ms2: ints.ms2,
        orbsym: ints.orbsym.iter().copied().take(m).collect(),
        isym: ints.isym,
    }
}

pub(crate) fn kappa_flat(d: usize, pairs: &[(usize, usize)], coords: &[f64]) -> Vec<f64> {
    assert_eq!(pairs.len(), coords.len(), "kappa_flat: coordinate count");
    let mut k = vec![0.0f64; d * d];
    for (&(p, q), &c) in pairs.iter().zip(coords) {
        debug_assert!(p > q, "kappa_flat: pairs must be strict-lower-triangle");
        k[p * d + q] = c;
        k[q * d + p] = -c;
    }
    k
}

pub(crate) fn cas_active_integrals(
    ints: &Integrals<f64>,
    part: &CasPartition,
    kappa_full: &[f64],
) -> (Integrals<f64>, f64) {
    let d = ints.norb;
    assert_eq!(kappa_full.len(), d * d, "cas_active_integrals: κ shape");
    let u = expm_antisym(&scale_flat(kappa_full, -1.0), d);
    let rot = rotate_integrals(ints, &u);
    let frozen = freeze_core(&rot, part.n_core);
    let act = slice_leading(&frozen, part.n_active);
    (act, frozen.e0)
}

fn sign_fix(v: &mut [f64]) {
    let mut best = 0usize;
    for i in 1..v.len() {
        if v[i].abs() > v[best].abs() {
            best = i;
        }
    }
    if v[best] < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

pub(crate) fn cas_ci(act: &Integrals<f64>, sp_act: &DetSpace) -> (Vec<f64>, Vec<Vec<f64>>) {
    let h = sc_hamiltonian_matrix(act.norb, &act.t, &act.v, sp_act);
    let (roots, mut vecs) = dense_eigen(&h);
    for v in vecs.iter_mut() {
        sign_fix(v);
    }
    (roots, vecs)
}

pub(crate) fn e_of_c_kappa(
    ints: &Integrals<f64>,
    part: &CasPartition,
    sp_act: &DetSpace,
    pairs: &[(usize, usize)],
    c_act: &[f64],
    kappa_coords: &[f64],
) -> f64 {
    let k = kappa_flat(ints.norb, pairs, kappa_coords);
    let (act, e0_tot) = cas_active_integrals(ints, part, &k);
    let h = sc_hamiltonian_matrix(act.norb, &act.t, &act.v, sp_act);
    let hc = mat_vec(&h, c_act);
    dot_vec(c_act, &hc) / dot_vec(c_act, c_act) + e0_tot
}

pub(crate) fn one_index_transform(ints: &Integrals<f64>, kappa_full: &[f64]) -> Integrals<f64> {
    let d = ints.norb;
    assert_eq!(kappa_full.len(), d * d, "one_index_transform: κ shape");
    let nz: Vec<(usize, usize, f64)> = (0..d)
        .flat_map(|i| (0..d).map(move |a| (i, a)))
        .filter_map(|(i, a)| {
            let k = kappa_full[i * d + a];
            (k != 0.0).then_some((i, a, k))
        })
        .collect();

    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for &(i, a, k) in &nz {
        for b in 0..d {
            t.data[a * d + b] -= k * ints.t.data[i * d + b];
            t.data[b * d + a] -= k * ints.t.data[b * d + i];
        }
    }

    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    let d2 = d * d;
    let d3 = d2 * d;
    for &(i, a, k) in &nz {
        for x in 0..d {
            for y in 0..d {
                for z in 0..d {
                    v.data[a * d3 + x * d2 + y * d + z] -=
                        k * ints.v.data[i * d3 + x * d2 + y * d + z];
                    v.data[x * d3 + a * d2 + y * d + z] -=
                        k * ints.v.data[x * d3 + i * d2 + y * d + z];
                    v.data[x * d3 + y * d2 + a * d + z] -=
                        k * ints.v.data[x * d3 + y * d2 + i * d + z];
                    v.data[x * d3 + y * d2 + z * d + a] -=
                        k * ints.v.data[x * d3 + y * d2 + z * d + i];
                }
            }
        }
    }

    Integrals {
        t,
        v,
        e0: 0.0,
        norb: d,
        nelec: ints.nelec,
        ms2: ints.ms2,
        orbsym: ints.orbsym.clone(),
        isym: ints.isym,
    }
}

pub(crate) fn dense_fixture(
    dirname: &str,
    nelecas: usize,
    ncas: usize,
) -> (Integrals<f64>, CasPartition) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(dirname);
    let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .unwrap_or_else(|e| panic!("{dirname}: {e:?}"));
    let ints = load.ints;
    assert!(
        ints.nelec >= nelecas && (ints.nelec - nelecas).is_multiple_of(2),
        "{dirname}: nelec = {} cannot host a CAS({nelecas},{ncas}) with a doubly-occupied core",
        ints.nelec
    );
    let n_core = (ints.nelec - nelecas) / 2;
    assert!(
        n_core + ncas <= ints.norb,
        "{dirname}: n_core {n_core} + ncas {ncas} exceeds norb {}",
        ints.norb
    );
    let part = CasPartition {
        n_core,
        n_active: ncas,
        n_virt: ints.norb - n_core - ncas,
    };
    (ints, part)
}

const DENSE_PSD_TOL: f64 = 1e-10;

pub(crate) fn williamson_dense(hrr: &[Vec<f64>], hii: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    let n = hrr.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let sym = |m: &[Vec<f64>]| -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| 0.5 * (m[i][j] + m[j][i])).collect())
            .collect()
    };
    let r = sym(hrr);
    let ii = sym(hii);
    let (lam, vecs) = dense_eigen(&r);
    let max_abs_eig = lam.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let min_eig = lam.iter().cloned().fold(f64::INFINITY, f64::min);
    if min_eig <= DENSE_PSD_TOL * max_abs_eig {
        return Err(format!(
            "H^rr is not positive definite: min_eig = {min_eig:.6e} against max|eig| = \
             {max_abs_eig:.6e}"
        ));
    }
    let mut half = vec![vec![0.0f64; n]; n];
    for (c, w) in vecs.iter().enumerate() {
        let sq = lam[c].sqrt();
        for i in 0..n {
            let wi = w[i] * sq;
            for j in 0..n {
                half[i][j] += wi * w[j];
            }
        }
    }
    let m = sym(&mat_mul(&mat_mul(&half, &ii), &half));
    let (mu, _) = dense_eigen(&m);
    let mu_max = mu.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    let mu_min = mu.iter().cloned().fold(f64::INFINITY, f64::min);
    if mu_min < -DENSE_PSD_TOL * mu_max {
        return Err(format!("Ω̃² has a negative eigenvalue {mu_min:.6e}"));
    }
    let mut omega: Vec<f64> = mu.iter().map(|&x| x.max(0.0).sqrt()).collect();
    omega.sort_by(|a, b| a.partial_cmp(b).expect("finite ω"));
    Ok(omega)
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DenseLrOpts {
    pub drop_b_orb: bool,
    pub flip_b_orb_sign: bool,
    pub halve_b_cross: bool,
    pub metric_identity: bool,
    pub plus_g: bool,
    pub swap_ab: bool,
    pub ci_irreps: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DenseLrDiag {
    pub mu_min: f64,
    pub mu_max: f64,
    pub max_tau: f64,
    pub z_norm: f64,
    pub max_b_orb: f64,
    pub max_b_cross: f64,
    pub max_b_tt: f64,
    pub ci_gap_min: f64,
    pub core_fold_dev: f64,
    pub sigma_antisym: f64,
    pub sigma_vs_s: f64,
    pub max_eqp_c: f64,
    pub hrr_min_eig: f64,
}

pub(crate) struct DenseLrFull {
    pub pairs: Vec<(usize, usize)>,
    pub c: Vec<f64>,
    pub c_act: Vec<f64>,
    pub zeta: Vec<Vec<f64>>,
    pub roots: Vec<f64>,
    pub e0_tot: f64,
    pub lambda: f64,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<Vec<f64>>,
    pub s: Vec<Vec<f64>>,
    pub sigma: Vec<Vec<f64>>,
    pub dim_ci: usize,
    pub dim_orb: usize,
    pub diag: DenseLrDiag,
    zeta_a: Vec<Vec<f64>>,
    s_inv_half: Vec<Vec<f64>>,
}

impl DenseLrFull {
    pub(crate) fn dim(&self) -> usize {
        self.dim_ci + self.dim_orb
    }

    pub(crate) fn hrr_raw(&self) -> Vec<Vec<f64>> {
        mat_add(&self.a, &self.b)
    }

    pub(crate) fn hii_raw(&self) -> Vec<Vec<f64>> {
        mat_add(&self.a, &mat_scale(&self.b, -1.0))
    }

    fn congruence(&self, m: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.dim();
        let mut w = mat_identity(n);
        for i in 0..self.dim_orb {
            for j in 0..self.dim_orb {
                w[self.dim_ci + i][self.dim_ci + j] = self.s_inv_half[i][j];
            }
        }
        mat_mul(&mat_mul(&w, m), &w)
    }

    pub(crate) fn hrr(&self) -> Vec<Vec<f64>> {
        self.congruence(&self.hrr_raw())
    }

    pub(crate) fn hii(&self) -> Vec<Vec<f64>> {
        self.congruence(&self.hii_raw())
    }

    pub(crate) fn omega(&self) -> Vec<f64> {
        williamson_dense(&self.hrr(), &self.hii()).unwrap_or_else(|e| {
            panic!(
                "dense Williamson readout refused: {e} (min_eig(H^rr) = {:.6e}, μ(S) ∈ \
                 [{:.3e}, {:.3e}], ‖z‖ = {:.3e})",
                self.diag.hrr_min_eig, self.diag.mu_min, self.diag.mu_max, self.diag.z_norm
            )
        })
    }

    pub(crate) fn zeta_act(&self, i: usize) -> Vec<f64> {
        self.zeta_a[i].clone()
    }
}

fn select_ci_directions(vecs: &[Vec<f64>], irreps: Option<&[usize]>) -> (Vec<Vec<f64>>, f64) {
    let Some(labels) = irreps else {
        return (vecs[1..].to_vec(), 1.0);
    };
    assert_eq!(
        labels.len(),
        vecs[0].len(),
        "ci_irreps must carry one label per CAS determinant"
    );
    let n_irrep = labels.iter().copied().max().unwrap_or(0) + 1;
    let weights = |v: &[f64]| -> Vec<f64> {
        let mut w = vec![0.0f64; n_irrep];
        for (i, &c) in v.iter().enumerate() {
            w[labels[i]] += c * c;
        }
        w
    };
    let argmax =
        |w: &[f64]| -> usize { (0..w.len()).fold(0, |b, i| if w[i] > w[b] { i } else { b }) };
    let w0 = weights(&vecs[0]);
    let g0 = argmax(&w0);
    let mut purity = w0[g0] / w0.iter().sum::<f64>();
    let mut kept = Vec::new();
    for v in &vecs[1..] {
        let w = weights(v);
        let g = argmax(&w);
        let p = w[g] / w.iter().sum::<f64>();
        if g == g0 {
            purity = purity.min(p);
            kept.push(v.clone());
        }
    }
    assert!(
        purity > 0.99,
        "ci_irreps: a kept CAS-CI eigenvector is not irrep-pure (worst weight fraction \
         {purity:.6}) — the Γ₀ restriction is then not a well-defined block and the anchor would \
         compare two different charts"
    );
    (kept, purity)
}

fn min_eig_of(m: &[Vec<f64>]) -> f64 {
    if m.is_empty() {
        return f64::INFINITY;
    }
    dense_eigen(m).0[0]
}

fn inv_sqrt_spd(s: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64, f64) {
    let n = s.len();
    if n == 0 {
        return (Vec::new(), f64::INFINITY, 0.0);
    }
    let (mu, vecs) = dense_eigen(s);
    let mu_min = mu[0];
    let mu_max = mu[n - 1];
    assert!(
        mu_min > 1e-12 * mu_max.max(1.0),
        "the orbital Gram S is numerically singular (μ_min = {mu_min:.3e}, μ_max = \
         {mu_max:.3e}): S^{{−½}} would amplify roundoff by 1/μ_min. This is the dense twin of the \
         joint stack's `orth_error` refusal — report the numbers, do not invert."
    );
    let mut out = vec![vec![0.0f64; n]; n];
    for (c, w) in vecs.iter().enumerate() {
        let f = 1.0 / mu[c].sqrt();
        for i in 0..n {
            let wi = w[i] * f;
            for j in 0..n {
                out[i][j] += wi * w[j];
            }
        }
    }
    (out, mu_min, mu_max)
}

#[allow(clippy::too_many_arguments)]
fn assemble_blocks(
    n_ci: usize,
    n_orb: usize,
    a_blocks: &[Vec<f64>],
    b_orb: &[Vec<f64>],
    b_cross: &[Vec<f64>],
    o: &DenseLrOpts,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, f64, f64) {
    let dim = n_ci + n_orb;
    let mut b = vec![vec![0.0f64; dim]; dim];
    let orb_scale = if o.drop_b_orb {
        0.0
    } else if o.flip_b_orb_sign {
        -1.0
    } else {
        1.0
    };
    let cross_scale = if o.halve_b_cross { 0.5 } else { 1.0 };
    for i in 0..n_ci {
        for x in 0..n_orb {
            let e = cross_scale * b_cross[i][x];
            b[i][n_ci + x] = e;
            b[n_ci + x][i] = e;
        }
    }
    for x in 0..n_orb {
        for y in 0..n_orb {
            b[n_ci + x][n_ci + y] = orb_scale * b_orb[x][y];
        }
    }
    if o.swap_ab {
        b = mat_scale(&b, -1.0);
    }
    let max_b_orb = (0..n_orb)
        .flat_map(|x| (0..n_orb).map(move |y| (x, y)))
        .fold(0.0f64, |acc, (x, y)| acc.max(b[n_ci + x][n_ci + y].abs()));
    let max_b_cross = (0..n_ci)
        .flat_map(|i| (0..n_orb).map(move |x| (i, x)))
        .fold(0.0f64, |acc, (i, x)| acc.max(b[i][n_ci + x].abs()));
    let a: Vec<Vec<f64>> = (0..dim)
        .map(|i| {
            (0..dim)
                .map(|j| 0.5 * (a_blocks[i][j] + a_blocks[j][i]))
                .collect()
        })
        .collect();
    (a, b, max_b_orb, max_b_cross)
}

pub(crate) fn dense_lr_full(ints: &Integrals<f64>, part: &CasPartition) -> DenseLrFull {
    dense_lr_full_with(ints, part, &DenseLrOpts::default())
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn dense_lr_full_with(
    ints: &Integrals<f64>,
    part: &CasPartition,
    o: &DenseLrOpts,
) -> DenseLrFull {
    let d = ints.norb;
    let (nup, ndn) = electrons(ints);
    let sp = DetSpace::new(d, nup, ndn);
    let (full_pos, sp_act) = sp.cas_subset(part);

    let (act, e0_tot) = cas_active_integrals(ints, part, &vec![0.0f64; d * d]);
    let (roots, vecs) = cas_ci(&act, &sp_act);
    let c_act = vecs[0].clone();
    let (zeta_a, _purity) = select_ci_directions(&vecs, o.ci_irreps.as_deref());

    let c = sp.embed(&full_pos, &c_act);
    let zeta: Vec<Vec<f64>> = zeta_a.iter().map(|z| sp.embed(&full_pos, z)).collect();

    let hfull = sc_hamiltonian_matrix(d, &ints.t, &ints.v, &sp);
    let hc = mat_vec(&hfull, &c);
    let lambda = dot_vec(&c, &hc);
    let z: Vec<f64> = hc.iter().zip(&c).map(|(&h, &x)| h - lambda * x).collect();
    let core_fold_dev = (lambda - (roots[0] + e0_tot - ints.e0)).abs();

    let pairs = cas_pairs(part, d).keep;
    let n_orb = pairs.len();
    let n_ci = zeta_a.len();
    let dim = n_ci + n_orb;

    let sigma_sign = if o.plus_g { -1.0 } else { 1.0 };
    let gc: Vec<Vec<f64>> = pairs
        .iter()
        .map(|&(p, q)| apply_gpq(d, p, q, &sp, &c))
        .collect();
    let u: Vec<Vec<f64>> = gc
        .iter()
        .map(|g| g.iter().map(|&x| -sigma_sign * x).collect())
        .collect();

    let mut max_eqp_c = 0.0f64;
    let mut max_tau = 0.0f64;
    for (a, &(p, q)) in pairs.iter().enumerate() {
        let dex = apply_epq(d, q, p, &sp, &c);
        max_eqp_c = max_eqp_c.max(dot_vec(&dex, &dex).sqrt());
        let tau2: f64 = full_pos.iter().map(|&i| u[a][i] * u[a][i]).sum();
        max_tau = max_tau.max(tau2.sqrt());
    }

    let mut s = vec![vec![0.0f64; n_orb]; n_orb];
    let mut sigma = vec![vec![0.0f64; n_orb]; n_orb];
    for a in 0..n_orb {
        for b in 0..n_orb {
            s[a][b] = dot_vec(&gc[a], &gc[b]);
            let (pa, qa) = pairs[a];
            let (pb, qb) = pairs[b];
            let t1 = dot_vec(
                &c,
                &apply_epq(d, qa, pa, &sp, &apply_epq(d, pb, qb, &sp, &c)),
            );
            let t2 = dot_vec(
                &c,
                &apply_epq(d, pb, qb, &sp, &apply_epq(d, qa, pa, &sp, &c)),
            );
            sigma[a][b] = t1 - t2;
        }
    }
    let mut sigma_vs_s = 0.0f64;
    let mut sigma_antisym = 0.0f64;
    for a in 0..n_orb {
        for b in 0..n_orb {
            sigma_vs_s = sigma_vs_s.max((sigma[a][b] - s[a][b]).abs());
            sigma_antisym = sigma_antisym.max((sigma[a][b] - sigma[b][a]).abs());
        }
    }

    let dirs: Vec<&Vec<f64>> = zeta.iter().chain(u.iter()).collect();
    let hv: Vec<Vec<f64>> = dirs
        .iter()
        .map(|v| {
            let hvv = mat_vec(&hfull, v);
            hvv.iter()
                .zip(v.iter())
                .map(|(&h, &x)| h - lambda * x)
                .collect()
        })
        .collect();
    let a_blocks: Vec<Vec<f64>> = (0..dim)
        .map(|k| (0..dim).map(|l| dot_vec(dirs[k], &hv[l])).collect())
        .collect();

    let mut b_orb = vec![vec![0.0f64; n_orb]; n_orb];
    for a in 0..n_orb {
        let (pa, qa) = pairs[a];
        for b in 0..n_orb {
            let gagb = apply_gpq(d, pa, qa, &sp, &gc[b]);
            let (pb, qb) = pairs[b];
            let gbga = apply_gpq(d, pb, qb, &sp, &gc[a]);
            let anti: Vec<f64> = gagb
                .iter()
                .zip(&gbga)
                .map(|(&x, &y)| 0.5 * (x + y))
                .collect();
            b_orb[a][b] = dot_vec(&z, &anti);
        }
    }
    let mut b_cross = vec![vec![0.0f64; n_orb]; n_ci];
    for (a, &(p, q)) in pairs.iter().enumerate() {
        let gz = apply_gpq(d, p, q, &sp, &z);
        for i in 0..n_ci {
            b_cross[i][a] = sigma_sign * dot_vec(&gz, &zeta[i]);
        }
    }

    let (a_mat, b_mat, max_b_orb, max_b_cross) =
        assemble_blocks(n_ci, n_orb, &a_blocks, &b_orb, &b_cross, o);

    let (s_inv_half, mu_min, mu_max) = if o.metric_identity {
        (mat_identity(n_orb), f64::NAN, f64::NAN)
    } else {
        inv_sqrt_spd(&s)
    };
    let (mu_min, mu_max) = if o.metric_identity {
        let (mu, _) = dense_eigen(&s);
        (mu[0], mu[n_orb - 1])
    } else {
        (mu_min, mu_max)
    };

    let ci_gap_min = roots
        .windows(2)
        .fold(f64::INFINITY, |acc, w| acc.min(w[1] - w[0]));
    let z_norm = dot_vec(&z, &z).sqrt();

    let mut lr = DenseLrFull {
        pairs,
        c,
        c_act,
        zeta,
        roots,
        e0_tot,
        lambda,
        a: a_mat,
        b: b_mat,
        s,
        sigma,
        dim_ci: n_ci,
        dim_orb: n_orb,
        diag: DenseLrDiag {
            mu_min,
            mu_max,
            max_tau,
            z_norm,
            max_b_orb,
            max_b_cross,
            max_b_tt: 0.0,
            ci_gap_min,
            core_fold_dev,
            sigma_antisym,
            sigma_vs_s,
            max_eqp_c,
            hrr_min_eig: f64::NAN,
        },
        zeta_a,
        s_inv_half,
    };
    lr.diag.hrr_min_eig = min_eig_of(&lr.hrr());
    lr
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn dense_lr_cas(ints: &Integrals<f64>, part: &CasPartition) -> DenseLrFull {
    let d = ints.norb;
    let (nup, ndn) = electrons(ints);
    let ni = part.n_core;
    let sp_act = DetSpace::new(part.n_active, nup - ni, ndn - ni);

    let (act, e0_tot) = cas_active_integrals(ints, part, &vec![0.0f64; d * d]);
    let (roots, vecs) = cas_ci(&act, &sp_act);
    let c_act = vecs[0].clone();
    let zeta_a: Vec<Vec<f64>> = vecs[1..].to_vec();

    let cas_masks: Vec<DetMask> = sp_act
        .dets
        .iter()
        .map(|(ou, od)| {
            let mut up = vec![false; d];
            let mut dn = vec![false; d];
            for (u, v) in up.iter_mut().zip(dn.iter_mut()).take(ni) {
                *u = true;
                *v = true;
            }
            for (j, &b) in ou.iter().enumerate() {
                up[ni + j] = b;
            }
            for (j, &b) in od.iter().enumerate() {
                dn[ni + j] = b;
            }
            pack_det(&up, &dn)
        })
        .collect();
    let to_sparse = |v: &[f64]| -> SpVec {
        let mut s = SpVec::with_capacity(v.len());
        for (i, &x) in v.iter().enumerate() {
            s.add(cas_masks[i], x);
        }
        s
    };
    let csp = to_sparse(&c_act);
    let zsp: Vec<SpVec> = zeta_a.iter().map(|z| to_sparse(z)).collect();

    let lambda = roots[0] + e0_tot - ints.e0;
    let mut z = sparse_h(d, &ints.t, &ints.v, &csp);
    z.axpy(-lambda, &csp);

    let pairs = cas_pairs(part, d).keep;
    let n_orb = pairs.len();
    let n_ci = zeta_a.len();
    let dim = n_ci + n_orb;

    let gc: Vec<SpVec> = pairs.iter().map(|&(p, q)| sparse_gpq(p, q, &csp)).collect();
    let mut max_eqp_c = 0.0f64;
    let mut max_tau = 0.0f64;
    for (a, &(p, q)) in pairs.iter().enumerate() {
        max_eqp_c = max_eqp_c.max(sparse_epq(q, p, &csp).norm());
        let tau2: f64 = cas_masks.iter().map(|&m| gc[a].get(m).powi(2)).sum();
        max_tau = max_tau.max(tau2.sqrt());
    }

    let mut s = vec![vec![0.0f64; n_orb]; n_orb];
    let mut sigma = vec![vec![0.0f64; n_orb]; n_orb];
    for a in 0..n_orb {
        for b in 0..n_orb {
            s[a][b] = gc[a].dot(&gc[b]);
            let (pa, qa) = pairs[a];
            let (pb, qb) = pairs[b];
            let t1 = csp.dot(&sparse_epq(qa, pa, &sparse_epq(pb, qb, &csp)));
            let t2 = csp.dot(&sparse_epq(pb, qb, &sparse_epq(qa, pa, &csp)));
            sigma[a][b] = t1 - t2;
        }
    }
    let mut sigma_vs_s = 0.0f64;
    let mut sigma_antisym = 0.0f64;
    for a in 0..n_orb {
        for b in 0..n_orb {
            sigma_vs_s = sigma_vs_s.max((sigma[a][b] - s[a][b]).abs());
            sigma_antisym = sigma_antisym.max((sigma[a][b] - sigma[b][a]).abs());
        }
    }

    let mut a_blocks = vec![vec![0.0f64; dim]; dim];
    for i in 0..n_ci {
        a_blocks[i][i] = roots[i + 1] - roots[0];
    }
    let mut w: Vec<SpVec> = Vec::with_capacity(n_orb);
    for a in 0..n_orb {
        let mut hg = sparse_h(d, &ints.t, &ints.v, &gc[a]);
        hg.axpy(-lambda, &gc[a]);
        w.push(hg);
    }
    for a in 0..n_orb {
        for b in 0..n_orb {
            a_blocks[n_ci + a][n_ci + b] = gc[a].dot(&w[b]);
        }
        for i in 0..n_ci {
            let e = -zsp[i].dot(&w[a]);
            a_blocks[i][n_ci + a] = e;
            a_blocks[n_ci + a][i] = e;
        }
    }

    let mut b_orb = vec![vec![0.0f64; n_orb]; n_orb];
    for a in 0..n_orb {
        let (pa, qa) = pairs[a];
        for b in 0..n_orb {
            let (pb, qb) = pairs[b];
            let mut anti = sparse_gpq(pa, qa, &gc[b]);
            anti.axpy(1.0, &sparse_gpq(pb, qb, &gc[a]));
            b_orb[a][b] = 0.5 * z.dot(&anti);
        }
    }
    let mut b_cross = vec![vec![0.0f64; n_orb]; n_ci];
    for (a, &(p, q)) in pairs.iter().enumerate() {
        for i in 0..n_ci {
            b_cross[i][a] = -z.dot(&sparse_gpq(p, q, &zsp[i]));
        }
    }

    let o = DenseLrOpts::default();
    let (a_mat, b_mat, max_b_orb, max_b_cross) =
        assemble_blocks(n_ci, n_orb, &a_blocks, &b_orb, &b_cross, &o);
    let (s_inv_half, mu_min, mu_max) = inv_sqrt_spd(&s);
    let ci_gap_min = roots
        .windows(2)
        .fold(f64::INFINITY, |acc, x| acc.min(x[1] - x[0]));

    let mut lr = DenseLrFull {
        pairs,
        c: Vec::new(),
        c_act,
        zeta: Vec::new(),
        roots,
        e0_tot,
        lambda,
        a: a_mat,
        b: b_mat,
        s,
        sigma,
        dim_ci: n_ci,
        dim_orb: n_orb,
        diag: DenseLrDiag {
            mu_min,
            mu_max,
            max_tau,
            z_norm: z.norm(),
            max_b_orb,
            max_b_cross,
            max_b_tt: 0.0,
            ci_gap_min,
            core_fold_dev: 0.0,
            sigma_antisym,
            sigma_vs_s,
            max_eqp_c,
            hrr_min_eig: f64::NAN,
        },
        zeta_a,
        s_inv_half,
    };
    lr.diag.hrr_min_eig = min_eig_of(&lr.hrr());
    lr
}

pub(crate) fn dense_orb_gradient(
    ints: &Integrals<f64>,
    part: &CasPartition,
    pairs: &[(usize, usize)],
    c_act: &[f64],
    sp_act: &DetSpace,
) -> Vec<f64> {
    let d = ints.norb;
    let nrm = dot_vec(c_act, c_act);
    (0..pairs.len())
        .map(|a| {
            let mut coords = vec![0.0f64; pairs.len()];
            coords[a] = 1.0;
            let k = kappa_flat(d, pairs, &coords);
            let tr = one_index_transform(ints, &k);
            let frozen = freeze_core(&tr, part.n_core);
            let e_core_d = frozen.e0;
            let act = slice_leading(&frozen, part.n_active);
            let h = sc_hamiltonian_matrix(act.norb, &act.t, &act.v, sp_act);
            e_core_d + dot_vec(c_act, &mat_vec(&h, c_act)) / nrm
        })
        .collect()
}

pub(crate) struct DenseCasscfOpts {
    pub g_tol: f64,
    pub max_iter: usize,
    pub trust0: f64,
    pub shift_floor: f64,
    pub verbose: bool,
}

pub(crate) struct DenseCasscfOut {
    pub e_total: f64,
    pub u: Vec<f64>,
    #[allow(dead_code)]
    pub c_act: Vec<f64>,
    pub roots: Vec<f64>,
    pub e0_tot: f64,
    pub g_norm: f64,
    pub iters: usize,
    pub stalled_at_floor: bool,
    pub converged: bool,
}

const NOISE_FLOOR_C: f64 = 8.0;

const GRAD_SANITY: f64 = 1e-5;

fn flat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                c[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    c
}

fn flat_identity(n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

pub(crate) fn dense_casscf(
    ints: &Integrals<f64>,
    part: &CasPartition,
    o: &DenseCasscfOpts,
) -> DenseCasscfOut {
    let d = ints.norb;
    let (nup, ndn) = electrons(ints);
    let sp_act = DetSpace::new(part.n_active, nup - part.n_core, ndn - part.n_core);
    let pairs = cas_pairs(part, d).keep;
    let n_o = pairs.len();

    let mut u_acc = flat_identity(d);
    let mut iters = 0usize;
    let mut converged = false;
    let mut stalled_at_floor = false;
    let mut e_total: f64;
    let mut g_norm: f64;
    let mut c_act: Vec<f64>;
    let mut roots: Vec<f64>;
    let mut e0_tot: f64;

    loop {
        let cur = rotate_integrals(ints, &u_acc);
        let lr = dense_lr_cas(&cur, part);
        c_act = lr.c_act.clone();
        roots = lr.roots.clone();
        e0_tot = lr.e0_tot;
        e_total = roots[0] + e0_tot;

        let g = dense_orb_gradient(&cur, part, &pairs, &c_act, &sp_act);
        g_norm = g.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
        if o.verbose {
            eprintln!(
                "dense_casscf iter {iters}: E = {e_total:.12}, ‖g‖_inf = {g_norm:.3e}, \
                 n_ci = {}, n_o = {n_o}",
                lr.dim_ci
            );
        }
        if g_norm < o.g_tol {
            converged = true;
            break;
        }
        if iters >= o.max_iter {
            break;
        }

        let hrr = lr.hrr_raw();
        let n_ci = lr.dim_ci;
        let mut model = vec![vec![0.0f64; n_o]; n_o];
        for a in 0..n_o {
            for b in 0..n_o {
                model[a][b] = 2.0 * hrr[n_ci + a][n_ci + b];
            }
        }
        for (i, hrr_i) in hrr.iter().enumerate().take(n_ci) {
            let de = hrr_i[i];
            if de <= 1e-8 {
                continue;
            }
            for a in 0..n_o {
                for b in 0..n_o {
                    model[a][b] -= 2.0 * hrr_i[n_ci + a] * hrr_i[n_ci + b] / de;
                }
            }
        }
        let (ev, evec) = dense_eigen(&model);
        let shift = if ev[0] < o.shift_floor {
            o.shift_floor - ev[0]
        } else {
            0.0
        };
        let decrement: f64 = 0.5
            * evec
                .iter()
                .enumerate()
                .map(|(k, w)| {
                    let gw = dot_vec(w, &g);
                    gw * gw / (ev[k] + shift)
                })
                .sum::<f64>();
        let mut delta = vec![0.0f64; n_o];
        for (k, w) in evec.iter().enumerate() {
            let coef = -dot_vec(w, &g) / (ev[k] + shift);
            for (dx, &wi) in delta.iter_mut().zip(w) {
                *dx += coef * wi;
            }
        }
        let dn_norm = dot_vec(&delta, &delta).sqrt();
        if dn_norm > o.trust0 {
            delta = delta.iter().map(|x| x * o.trust0 / dn_norm).collect();
        }

        let energy_at = |coords: &[f64]| -> f64 {
            let k = kappa_flat(d, &pairs, coords);
            let (act, e0t) = cas_active_integrals(&cur, part, &k);
            let h = sc_hamiltonian_matrix(act.norb, &act.t, &act.v, &sp_act);
            dense_eigen(&h).0[0] + e0t
        };

        let mut accepted: Option<Vec<f64>> = None;
        for dir in [delta.clone(), {
            let gn = dot_vec(&g, &g).sqrt().max(f64::MIN_POSITIVE);
            g.iter().map(|x| -x * o.trust0 / gn).collect::<Vec<f64>>()
        }] {
            let slope = dot_vec(&g, &dir);
            if slope >= 0.0 {
                continue;
            }
            let mut tau = 1.0f64;
            for _ in 0..30 {
                let trial: Vec<f64> = dir.iter().map(|x| tau * x).collect();
                if energy_at(&trial) <= e_total + 1e-4 * tau * slope {
                    accepted = Some(trial);
                    break;
                }
                tau *= 0.5;
            }
            if accepted.is_some() {
                break;
            }
        }
        let Some(step) = accepted else {
            let noise = NOISE_FLOOR_C * f64::EPSILON * e_total.abs();
            stalled_at_floor = decrement <= noise && g_norm <= GRAD_SANITY;
            converged = stalled_at_floor;
            if o.verbose {
                eprintln!(
                    "dense_casscf: both line searches refused at ‖g‖ = {g_norm:.3e}, \
                     decrement = {decrement:.3e} vs noise floor {noise:.3e} ⇒ converged = {converged}"
                );
            }
            break;
        };

        let k = kappa_flat(d, &pairs, &step);
        u_acc = flat_mul(&u_acc, &expm_antisym(&scale_flat(&k, -1.0), d), d);
        iters += 1;
    }

    DenseCasscfOut {
        e_total,
        u: u_acc,
        c_act,
        roots,
        e0_tot,
        g_norm,
        iters,
        stalled_at_floor,
        converged,
    }
}

pub(crate) fn lih_dense() -> &'static (Integrals<f64>, CasPartition) {
    static CELL: OnceLock<(Integrals<f64>, CasPartition)> = OnceLock::new();
    CELL.get_or_init(|| dense_fixture("lih_sto3g", 2, 2))
}

pub(crate) fn h2o_dense() -> &'static (Integrals<f64>, CasPartition) {
    static CELL: OnceLock<(Integrals<f64>, CasPartition)> = OnceLock::new();
    CELL.get_or_init(|| dense_fixture("h2o_sto3g", 8, 6))
}

pub(crate) fn lih_lr() -> &'static DenseLrFull {
    static CELL: OnceLock<DenseLrFull> = OnceLock::new();
    CELL.get_or_init(|| {
        let (ints, part) = lih_dense();
        dense_lr_full(ints, part)
    })
}

pub(crate) fn h2o_lr() -> &'static DenseLrFull {
    static CELL: OnceLock<DenseLrFull> = OnceLock::new();
    CELL.get_or_init(|| {
        let (ints, part) = h2o_dense();
        dense_lr_full(ints, part)
    })
}

pub(crate) fn lih_opt() -> &'static (Integrals<f64>, CasPartition) {
    static CELL: OnceLock<(Integrals<f64>, CasPartition)> = OnceLock::new();
    CELL.get_or_init(|| dense_opt_fixture("lih", lih_dense()))
}

pub(crate) fn h2o_opt() -> &'static (Integrals<f64>, CasPartition) {
    static CELL: OnceLock<(Integrals<f64>, CasPartition)> = OnceLock::new();
    CELL.get_or_init(|| dense_opt_fixture("h2o", h2o_dense()))
}

pub(crate) fn lih_lr_opt() -> &'static DenseLrFull {
    static CELL: OnceLock<DenseLrFull> = OnceLock::new();
    CELL.get_or_init(|| {
        let (ints, part) = lih_opt();
        dense_lr_full(ints, part)
    })
}

pub(crate) fn h2o_lr_opt() -> &'static DenseLrFull {
    static CELL: OnceLock<DenseLrFull> = OnceLock::new();
    CELL.get_or_init(|| {
        let (ints, part) = h2o_opt();
        dense_lr_full(ints, part)
    })
}

fn dense_opt_fixture(
    label: &str,
    base: &'static (Integrals<f64>, CasPartition),
) -> (Integrals<f64>, CasPartition) {
    let out = dense_casscf(
        &base.0,
        &base.1,
        &DenseCasscfOpts {
            g_tol: 1e-9,
            max_iter: 200,
            trust0: 0.5,
            shift_floor: 1e-6,
            verbose: false,
        },
    );
    assert!(
        out.converged,
        "{label}: the dense (c,κ) optimizer did not converge (‖g‖ = {:.3e} after {} iters); ω at \
         a near-optimum carries a first-order-in-‖g‖ bias behind a converged-looking energy",
        out.g_norm, out.iters
    );
    assert!(
        !out.stalled_at_floor,
        "{label}: the dense optimizer stopped at the √ε gradient floor (‖g‖ = {:.3e} after {} \
         iters) rather than at g_tol — an ω read at this point carries a first-order-in-‖g‖ bias",
        out.g_norm, out.iters
    );
    eprintln!(
        "{label} dense CASSCF optimum: E = {:.12}, ‖g‖_inf = {:.3e}, iters = {}",
        out.e_total, out.g_norm, out.iters
    );
    (rotate_integrals(&base.0, &out.u), base.1)
}

pub(crate) fn paper_json() -> &'static Json {
    static CELL: OnceLock<Json> = OnceLock::new();
    CELL.get_or_init(|| {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/casscf_paper_references.json");
        let text = std::fs::read_to_string(&p).expect("casscf_paper_references.json");
        json_parse(&text).expect("casscf_paper_references.json parses")
    })
}

fn electrons(ints: &Integrals<f64>) -> (usize, usize) {
    let nup = (ints.nelec + ints.ms2.unsigned_abs() as usize) / 2;
    (nup, ints.nelec - nup)
}

fn random_tv(d: usize, seed: u64) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let t = random_sym_t(d, &mut rng);
    let v = random_sym_v(d, &mut rng);
    (t, v)
}

fn max_abs_vec(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |a, &x| a.max(x.abs()))
}

#[test]
fn tjcd_b0_json_reader_self_test() {
    let v = paper_json();

    assert_eq!(
        json_num(v, "h2o/bases/sto-3g/casscf_gs"),
        Some(-75.01981141824061),
        "the reader must round-trip the generator's Python-repr floats exactly"
    );
    assert_eq!(
        json_num(v, "h2o/bases/6-31g/casscf_gs"),
        Some(-76.03955711741688)
    );
    assert_eq!(
        json_num(v, "hcho/bases/6-31g/casscf_gs"),
        Some(-113.8494245534876)
    );
    assert_eq!(
        json_num(v, "c2h4/bases/6-31g/casscf_gs"),
        Some(-78.03492072956435)
    );

    assert_eq!(json_num(v, "h2o/bases/sto-3g/norb"), Some(7.0));
    assert_eq!(json_num(v, "h2o/bases/sto-3g/nelec"), Some(10.0));

    assert_eq!(
        json_nums(v, "hcho/bases/6-31g/fci_roots"),
        None,
        "hcho/6-31g carries `fci_roots: null`"
    );
    let roots = json_nums(v, "h2o/bases/sto-3g/fci_roots").expect("h2o sto-3g fci_roots");
    assert_eq!(roots.len(), 3);

    assert!(
        json_get(v, "lih").is_none(),
        "lih must be absent from casscf_paper_references.json; if it has been added, the LiH \
         pins in tjcd_b12 / cas_anchor.rs should be revisited to use it"
    );

    let ex = json_nums(v, "h2o/paper/ex/0").expect("h2o paper ex[0]");
    assert_eq!(ex.len(), 3);
    assert_eq!(ex[2], 0.313274, "index 2 is the LR column");

    let (ints, _part) = h2o_dense();
    let e_nuc_json = json_num(v, "h2o/bases/sto-3g/e_nuc").unwrap();
    eprintln!(
        "tjcd_b0: h2o sto-3g e_nuc fixture {:.15} vs JSON {:.15} (Δ = {:.3e})",
        ints.e0,
        e_nuc_json,
        (ints.e0 - e_nuc_json).abs()
    );
    assert!(
        (ints.e0 - e_nuc_json).abs() <= 1e-10,
        "h2o sto-3g: fixture e_nuc {:.15} vs JSON {e_nuc_json:.15} — the FCIDUMP and the JSON are \
         not the same geometry",
        ints.e0
    );
}

#[test]
fn tjcd_b1_sc_hamiltonian_matches_jw_oracle_small_d() {
    let (t4, v4) = random_tv(4, 0xB100);
    let sp4 = DetSpace::new(4, 2, 1);
    assert_eq!(
        sp4.dets,
        dets_for(4, 2, 1),
        "DetSpace must reproduce util::dets_for's order EXACTLY (up outer, dn inner, \
         lexicographic): three independent consumers index rows by position in that list"
    );
    assert_eq!(sp4.len(), 24, "C(4,2)·C(4,1) = 24");
    let want4 = dense_hamiltonian_matrix(4, &t4, &v4, &sp4.dets);
    let got4 = sc_hamiltonian_matrix(4, &t4, &v4, &sp4);
    let dev4 = max_abs_diff(&got4, &want4);
    let sym4 = max_abs_diff(&got4, &mat_transpose(&got4));
    eprintln!(
        "tjcd_b1 A5 (d=4, {} dets): max|SC − JW| = {dev4:.3e}, max|H − Hᵀ| = {sym4:.3e}, \
         max|H| = {:.3e}",
        sp4.len(),
        max_abs(&want4)
    );
    assert!(
        dev4 <= 1e-13,
        "d=4: the Slater–Condon builder disagrees with the independent JW string oracle \
         entrywise by {dev4:.3e} — a phase or an index convention is wrong"
    );
    assert!(
        sym4 <= 1e-13,
        "d=4: H must be symmetric to roundoff, got {sym4:.3e}"
    );

    let (d5, nup5, ndn5) = (5usize, 3usize, 2usize);
    let (t5, v5) = random_tv(d5, 0xB105);
    let sp5 = DetSpace::new(d5, nup5, ndn5);
    assert_eq!(sp5.len(), 10 * 10, "C(5,3)·C(5,2) = 100");
    let want5 = dense_hamiltonian_matrix(d5, &t5, &v5, &sp5.dets);
    let got5 = sc_hamiltonian_matrix(d5, &t5, &v5, &sp5);
    let dev5 = max_abs_diff(&got5, &want5);
    eprintln!(
        "tjcd_b1 random (d=5, 100 dets): max|SC − JW| = {dev5:.3e}, max|H| = {:.3e}",
        max_abs(&want5)
    );
    assert!(
        dev5 <= 1e-12 * max_abs(&want5).max(1.0),
        "d=5 random (t,v): SC vs JW deviation {dev5:.3e}"
    );
}

#[test]
fn tjcd_b2_full_space_fci_matches_meta_e_fci() {
    for (dirname, e_fci) in [
        ("lih_sto3g", -7.882401932290224f64),
        ("h2o_sto3g", -75.01985479624324f64),
    ] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("testdata")
            .join(dirname);
        let load = load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
            .unwrap_or_else(|e| panic!("{dirname}: {e:?}"));
        let ints = load.ints;
        let (nup, ndn) = electrons(&ints);
        let sp = DetSpace::new(ints.norb, nup, ndn);
        let h = sc_hamiltonian_matrix(ints.norb, &ints.t, &ints.v, &sp);
        let (evals, _vecs) = dense_eigen(&h);
        let got = evals[0] + ints.e0;
        eprintln!(
            "tjcd_b2 {dirname}: {} dets, FCI = {got:.14} vs meta e_fci = {e_fci:.14} \
             (Δ = {:.3e}); roots[0..3] = {:?}",
            sp.len(),
            (got - e_fci).abs(),
            &evals[..3.min(evals.len())]
        );
        assert!(
            (got - e_fci).abs() <= 1e-9,
            "{dirname}: full-determinant-space ground root {got:.14} vs pyscf FCI {e_fci:.14} \
             — the Slater–Condon builder is wrong at production d, and every block built on it is \
             worthless. MEASURED-then-assert: tighten to the measured Δ after the first green run."
        );
    }

    let v = paper_json();
    let want = json_nums(v, "h2o/bases/sto-3g/fci_roots").expect("h2o sto-3g fci_roots");
    let (ints, _part) = h2o_dense();
    let (nup, ndn) = electrons(ints);
    let sp = DetSpace::new(ints.norb, nup, ndn);
    let h = sc_hamiltonian_matrix(ints.norb, &ints.t, &ints.v, &sp);
    let (evals, _vecs) = dense_eigen(&h);
    for (k, &w) in want.iter().enumerate() {
        let got = evals[k] + ints.e0;
        eprintln!(
            "tjcd_b2 h2o root {k}: {got:.14} vs JSON {w:.14} (Δ = {:.3e})",
            (got - w).abs()
        );
        assert!(
            (got - w).abs() <= 1e-8,
            "h2o sto-3g FCI root {k}: {got:.14} vs {w:.14}"
        );
    }
}

#[test]
fn tjcd_b3_apply_gpq_matches_dense_gpq() {
    let sp = DetSpace::new(4, 2, 1);
    let mut worst = 0.0f64;
    for p in 0..4 {
        for q in 0..4 {
            if p == q {
                continue;
            }
            let want = dense_gpq(4, p, q, &sp.dets);
            for col in 0..sp.len() {
                let mut e = vec![0.0f64; sp.len()];
                e[col] = 1.0;
                let got = apply_gpq(4, p, q, &sp, &e);
                for row in 0..sp.len() {
                    worst = worst.max((got[row] - want[row][col]).abs());
                }
            }
        }
    }
    eprintln!("tjcd_b3 d=4: max|apply_gpq − dense_gpq| = {worst:.3e} over all 12 ordered pairs");
    assert!(
        worst <= 1e-13,
        "d=4: apply_gpq vs dense_gpq deviation {worst:.3e}"
    );

    let (ints, part) = lih_dense();
    let (nup, ndn) = electrons(ints);
    let sp6 = DetSpace::new(ints.norb, nup, ndn);
    let pairs = cas_pairs(part, ints.norb).keep;
    let mut worst6 = 0.0f64;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB306);
    for &(p, q) in &pairs {
        let want = dense_gpq(ints.norb, p, q, &sp6.dets);
        for _ in 0..3 {
            let x: Vec<f64> = (0..sp6.len())
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect();
            let got = apply_gpq(ints.norb, p, q, &sp6, &x);
            let ref_ = mat_vec(&want, &x);
            worst6 = worst6.max(max_abs_diff(&[got], &[ref_]));
        }
    }
    eprintln!(
        "tjcd_b3 d=6 (lih, {} cas pairs, {} dets): max|apply_gpq·x − dense_gpq·x| = {worst6:.3e}",
        pairs.len(),
        sp6.len()
    );
    assert!(
        worst6 <= 1e-12,
        "d=6: apply_gpq vs dense_gpq deviation {worst6:.3e}"
    );
}

#[test]
fn tjcd_b4_one_index_transform_matches_rotate_integrals_fd() {
    let cases: Vec<(&str, Integrals<f64>)> = vec![
        ("random d=4", {
            let d = 4;
            let (t, v) = random_tv(d, 0xB404);
            Integrals {
                t,
                v,
                e0: 0.0,
                norb: d,
                nelec: 4,
                ms2: 0,
                orbsym: Vec::new(),
                isym: None,
            }
        }),
        ("lih_sto3g d=6", {
            let ints = &lih_dense().0;
            Integrals {
                t: ints.t.clone(),
                v: ints.v.clone(),
                e0: ints.e0,
                norb: ints.norb,
                nelec: ints.nelec,
                ms2: ints.ms2,
                orbsym: ints.orbsym.clone(),
                isym: ints.isym,
            }
        }),
    ];

    for (label, ints) in cases {
        let d = ints.norb;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB4_0000 + d as u64);
        let kappa = random_antisym_flat(d, &mut rng);
        let analytic = one_index_transform(&ints, &kappa);

        let mut prev: Option<(f64, f64)> = None;
        for &h in &[1e-2f64, 1e-3, 1e-4] {
            let up = rotate_integrals(&ints, &expm_antisym(&scale_flat(&kappa, -h), d));
            let dn = rotate_integrals(&ints, &expm_antisym(&scale_flat(&kappa, h), d));
            let mut dev_t = 0.0f64;
            for i in 0..d * d {
                let fd = (up.t.data[i] - dn.t.data[i]) / (2.0 * h);
                dev_t = dev_t.max((fd - analytic.t.data[i]).abs());
            }
            let mut dev_v = 0.0f64;
            for i in 0..d.pow(4) {
                let fd = (up.v.data[i] - dn.v.data[i]) / (2.0 * h);
                dev_v = dev_v.max((fd - analytic.v.data[i]).abs());
            }
            let dev = dev_t.max(dev_v);
            eprintln!(
                "tjcd_b4 {label}: h = {h:.1e}  max|FD − T(κ)| = {dev:.3e} (t {dev_t:.3e}, v {dev_v:.3e})"
            );
            if let Some((hp, dp)) = prev {
                let ratio = dp / dev.max(f64::MIN_POSITIVE);
                let order = ratio.ln() / (hp / h).ln();
                eprintln!(
                    "tjcd_b4 {label}: observed order between h = {hp:.1e} and {h:.1e} is {order:.2}"
                );
                assert!(
                    order > 1.5,
                    "{label}: T(κ) must be the O(h²) limit of rotate_integrals' own central \
                     difference; observed order {order:.2} means the analytic transform is not \
                     the derivative at all (a slot, a sign, or the exp(−κ) vs exp(+κ) choice)"
                );
            }
            prev = Some((h, dev));
        }
        assert!(
            prev.unwrap().1 <= 1e-6 * max_abs_vec(&analytic.v.data).max(1.0),
            "{label}: T(κ) vs FD did not reach the roundoff floor"
        );
    }
}

#[test]
fn tjcd_b5_core_fold_and_structural_invariants() {
    for (label, lr) in [("lih", lih_lr()), ("h2o", h2o_lr())] {
        let d = lr.diag;
        eprintln!(
            "tjcd_b5 {label}: dim = {} ({} CI + {} orb), λ_elec = {:.12}, e0_tot = {:.12}, \
             core_fold_dev = {:.3e}, max‖E_qp c‖ = {:.3e}, max|τ| = {:.3e}, ‖z‖ = {:.3e}, \
             μ(S) ∈ [{:.3e}, {:.3e}] (cond {:.3e}), max|B_orb| = {:.3e}, max|B_cross| = {:.3e}, \
             max|B_tt| = {:.3e}",
            lr.dim(),
            lr.dim_ci,
            lr.dim_orb,
            lr.lambda,
            lr.e0_tot,
            d.core_fold_dev,
            d.max_eqp_c,
            d.max_tau,
            d.z_norm,
            d.mu_min,
            d.mu_max,
            d.mu_max / d.mu_min,
            d.max_b_orb,
            d.max_b_cross,
            d.max_b_tt,
        );
        assert!(
            d.core_fold_dev <= 1e-10,
            "{label}: λ_full ≠ roots[0] + E_core(0) (dev {:.3e}) — the freeze_core fold, the \
             slice_leading window or the e0 ownership is wrong",
            d.core_fold_dev
        );
        assert!(
            d.max_eqp_c <= 1e-13,
            "{label}: E_{{q_a p_a}}·c must vanish identically on the non-redundant set \
             (got {:.3e}) — if it does not, `cas_pairs` is not the {{IA,IE,AE}} set it claims",
            d.max_eqp_c
        );
        assert!(
            d.max_tau <= 1e-12,
            "{label}: τ_a = P_CAS(Ĝ_a c) must vanish (got {:.3e}); without it G ≠ diag(I,S) and \
             the anchor compares two different metrics",
            d.max_tau
        );
    }

    let lih = lih_lr();
    assert_eq!(
        (lih.dim_ci, lih.dim_orb, lih.dim()),
        (3, 11, 14),
        "lih chart dimension"
    );
    let h2o = h2o_lr();
    assert_eq!(
        (h2o.dim_ci, h2o.dim_orb, h2o.dim()),
        (224, 6, 230),
        "h2o chart dimension"
    );
    assert_eq!(
        lih.pairs,
        cas_pairs(&lih_dense().1, lih_dense().0.norb).keep,
        "the orbital coordinate order MUST be cas_pairs' order verbatim — it indexes every row of \
         S, B_orb, B_cross and must match JointContext::pairs() element for element"
    );
}

#[test]
fn tjcd_b5b_cas_ci_gap_census() {
    for (label, lr) in [("lih", lih_lr()), ("h2o", h2o_lr())] {
        let gaps: Vec<f64> = lr.roots.windows(2).map(|w| w[1] - w[0]).take(8).collect();
        eprintln!(
            "tjcd_b5b {label}: CAS-CI roots[0..min(6)] = {:?}; first gaps = {:?}; min gap over the \
             whole spectrum = {:.3e}",
            &lr.roots[..6.min(lr.roots.len())],
            gaps,
            lr.diag.ci_gap_min
        );
        assert!(
            lr.diag.ci_gap_min > 1e-8,
            "{label}: a degenerate CAS-CI pair (min gap {:.3e}) makes the excited eigenvectors \
             defined only up to rotation. ω stays invariant; the entrywise block comparisons in \
             tjcd_b6/b8/b11 and cas_anchor.rs do NOT. Report it and restate those arms.",
            lr.diag.ci_gap_min
        );
        assert!(
            lr.roots[0] < lr.roots[1],
            "{label}: the CAS-CI ground root must be non-degenerate"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn q_fd2(
    ints: &Integrals<f64>,
    part: &CasPartition,
    sp_act: &DetSpace,
    pairs: &[(usize, usize)],
    c: &[f64],
    zeta: &[f64],
    kap: &[f64],
    e0: f64,
    h: f64,
) -> f64 {
    let cp: Vec<f64> = c.iter().zip(zeta).map(|(&a, &b)| a + h * b).collect();
    let cm: Vec<f64> = c.iter().zip(zeta).map(|(&a, &b)| a - h * b).collect();
    let kp: Vec<f64> = kap.iter().map(|&k| h * k).collect();
    let km: Vec<f64> = kap.iter().map(|&k| -h * k).collect();
    let ep = e_of_c_kappa(ints, part, sp_act, pairs, &cp, &kp);
    let em = e_of_c_kappa(ints, part, sp_act, pairs, &cm, &km);
    (ep - 2.0 * e0 + em) / (2.0 * h * h)
}

#[allow(clippy::too_many_arguments)]
fn q_fd4(
    ints: &Integrals<f64>,
    part: &CasPartition,
    sp_act: &DetSpace,
    pairs: &[(usize, usize)],
    c: &[f64],
    zeta: &[f64],
    kap: &[f64],
    e0: f64,
    h: f64,
) -> f64 {
    let ev = |s: f64| {
        let cs: Vec<f64> = c.iter().zip(zeta).map(|(&a, &b)| a + s * b).collect();
        let ks: Vec<f64> = kap.iter().map(|&k| s * k).collect();
        e_of_c_kappa(ints, part, sp_act, pairs, &cs, &ks)
    };
    (-ev(2.0 * h) + 16.0 * ev(h) - 30.0 * e0 + 16.0 * ev(-h) - ev(-2.0 * h)) / (24.0 * h * h)
}

#[test]
fn tjcd_b6_fd_of_e_c_kappa_certifies_a_plus_b() {
    for (label, ints, part, lr) in [
        ("lih", &lih_dense().0, &lih_dense().1, lih_lr()),
        ("h2o", &h2o_dense().0, &h2o_dense().1, h2o_lr()),
    ] {
        let d = ints.norb;
        let (nup, ndn) = electrons(ints);
        let sp = DetSpace::new(d, nup, ndn);
        let (full_pos, sp_act) = sp.cas_subset(part);
        let _ = full_pos;
        let pairs = &lr.pairs;
        let n_ci = lr.dim_ci;
        let n_orb = lr.dim_orb;

        let e0 = e_of_c_kappa(ints, part, &sp_act, pairs, &lr.c_act, &vec![0.0; n_orb]);
        eprintln!(
            "tjcd_b6 {label}: E(c,0) = {e0:.14}, λ + e0_tot = {:.14}",
            lr.lambda + lr.e0_tot
        );

        let hrr = lr.hrr_raw();
        let (ci_idx, orb_idx) = (0..n_ci)
            .flat_map(|i| (0..n_orb).map(move |a| (i, a)))
            .max_by(|&(i, a), &(j, b)| {
                hrr[i][n_ci + a]
                    .abs()
                    .partial_cmp(&hrr[j][n_ci + b].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .expect("at least one CI and one orbital direction");
        eprintln!(
            "tjcd_b6 {label}: probing ζ_{ci_idx} (‖ζ‖ = 1) and κ_{orb_idx} = pair {:?} \
             (S_aa = {:.3e}, cross entry {:.3e} — the largest of the {} candidates)",
            pairs[orb_idx],
            lr.s[orb_idx][orb_idx],
            hrr[ci_idx][n_ci + orb_idx],
            n_ci * n_orb
        );

        let zeta_act = lr.zeta_act(ci_idx);
        let zero_zeta = vec![0.0f64; lr.c_act.len()];
        let mut kap = vec![0.0f64; n_orb];
        kap[orb_idx] = 1.0;
        let zero_kap = vec![0.0f64; n_orb];

        let want_cc = hrr[ci_idx][ci_idx];
        let want_oo = hrr[n_ci + orb_idx][n_ci + orb_idx];
        let want_co = hrr[ci_idx][n_ci + orb_idx];

        let floor_at = |h: f64| f64::EPSILON * e0.abs() / (h * h);
        const FLOOR_MARGIN: f64 = 4.0;
        let mut checks = 0usize;
        let mut prev: Option<(f64, f64)> = None;
        for &h in &[3e-2f64, 1e-2, 3e-3, 1e-3, 3e-4] {
            let got = q_fd2(
                ints, part, &sp_act, pairs, &lr.c_act, &zeta_act, &zero_kap, e0, h,
            );
            let err = (got - want_cc).abs();
            let fl = floor_at(h);
            eprintln!(
                "tjcd_b6 {label} CI-CI: h = {h:.1e}  q_FD2 = {got:.10}  err = {err:.3e}  \
                 floor = {fl:.3e}"
            );
            if let Some((hp, ep)) = prev {
                let order = (ep / err.max(f64::MIN_POSITIVE)).ln() / (hp / h).ln();
                let clean = err > FLOOR_MARGIN * fl && ep > FLOOR_MARGIN * floor_at(hp);
                eprintln!(
                    "tjcd_b6 {label} CI-CI: observed order {order:.2} between {hp:.1e} and {h:.1e}\
                     {}",
                    if clean {
                        ""
                    } else {
                        " — NOT ASSERTED, inside the FD roundoff floor"
                    }
                );
                if clean {
                    checks += 1;
                    assert!(
                        order > 1.4,
                        "{label}: q_FD's 2nd-order stencil must converge at O(h²) toward (A+B); \
                         observed {order:.2} over a floor-clean interval ({hp:.1e} → {h:.1e}, \
                         errors {ep:.3e}/{err:.3e} against floors {:.3e}/{fl:.3e}). A flat bias \
                         means a WRONG (A+B), not a stencil problem.",
                        floor_at(hp)
                    );
                }
            }
            prev = Some((h, err));
        }
        assert!(
            checks >= 2,
            "{label}: the slope arm asserted only {checks} interval(s) — the floor-clean window \
             collapsed, so the O(h²) claim is untested. Extend the scan upward, do not relax it."
        );

        let h_acc = 1e-3;
        let fd_cc = q_fd4(
            ints, part, &sp_act, pairs, &lr.c_act, &zeta_act, &zero_kap, e0, h_acc,
        );
        let fd_oo = q_fd4(
            ints, part, &sp_act, pairs, &lr.c_act, &zero_zeta, &kap, e0, h_acc,
        );
        let fd_sum = q_fd4(
            ints, part, &sp_act, pairs, &lr.c_act, &zeta_act, &kap, e0, h_acc,
        );
        let fd_co = 0.5 * (fd_sum - fd_cc - fd_oo);

        let floor = f64::EPSILON * e0.abs() / (h_acc * h_acc);
        eprintln!(
            "tjcd_b6 {label}: FD floor ≈ {floor:.3e} at h = {h_acc:.1e}\n  CI-CI    FD {fd_cc:.10} \
             vs A+B {want_cc:.10}  (Δ = {:.3e})\n  orb-orb  FD {fd_oo:.10} vs A+B {want_oo:.10}  \
             (Δ = {:.3e})\n  cross    FD {fd_co:.10} vs A+B {want_co:.10}  (Δ = {:.3e})",
            (fd_cc - want_cc).abs(),
            (fd_oo - want_oo).abs(),
            (fd_co - want_co).abs(),
        );
        for (what, want) in [("CI-CI", want_cc), ("orb-orb", want_oo), ("cross", want_co)] {
            assert!(
                want.abs() >= 1e3 * floor,
                "{label} {what}: the probed entry {want:.3e} is within 3 decades of the FD \
                 roundoff floor {floor:.3e} — this arm would be green by vacuity. Re-polarize \
                 along an O(1) direction (and, at the C2v h2o point, note that the three \
                 cross-irrep pairs carry EXACT zeros by symmetry)."
            );
        }
        let tol = (1e4 * floor).max(1e-7);
        assert!(
            (fd_cc - want_cc).abs() <= tol,
            "{label} CI-CI: FD {fd_cc:.12} vs A+B {want_cc:.12}"
        );
        assert!(
            (fd_oo - want_oo).abs() <= tol,
            "{label} orb-orb: FD {fd_oo:.12} vs A+B {want_oo:.12}"
        );
        assert!(
            (fd_co - want_co).abs() <= tol,
            "{label} cross: FD {fd_co:.12} vs A+B {want_co:.12}"
        );
    }
}

#[test]
fn tjcd_b7_independent_dense_twin_of_the_chart() {
    let (ints, part) = lih_dense();
    let lr = lih_lr();
    let d = ints.norb;
    let (nup, ndn) = electrons(ints);
    let sp = DetSpace::new(d, nup, ndn);
    let (_full_pos, sp_act) = sp.cas_subset(part);
    let hfull = sc_hamiltonian_matrix(d, &ints.t, &ints.v, &sp);
    let e_exact = dot_vec(&lr.c, &mat_vec(&hfull, &lr.c));

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xB707);
    let mut worst = 0.0f64;
    for trial in 0..4 {
        let coords: Vec<f64> = (0..lr.dim_orb)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let ci_coeffs: Vec<f64> = (0..lr.dim_ci)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();

        let zeta_act: Vec<f64> = {
            let mut acc = vec![0.0f64; lr.c_act.len()];
            for (i, &w) in ci_coeffs.iter().enumerate() {
                let z = lr.zeta_act(i);
                for (a, &zi) in acc.iter_mut().zip(&z) {
                    *a += w * zi;
                }
            }
            acc
        };
        let e0 = e_of_c_kappa(
            ints,
            part,
            &sp_act,
            &lr.pairs,
            &lr.c_act,
            &vec![0.0; lr.dim_orb],
        );
        let q_chart = q_fd4(
            ints, part, &sp_act, &lr.pairs, &lr.c_act, &zeta_act, &coords, e0, 1e-3,
        );

        let kflat = kappa_flat(d, &lr.pairs, &coords);
        let kd = dense_kappa_matrix(d, &flat_to_coeff(d, &kflat), &sp.dets);
        let zeta_full: Vec<f64> = {
            let mut acc = vec![0.0f64; sp.len()];
            for (i, &w) in ci_coeffs.iter().enumerate() {
                for (a, &zi) in acc.iter_mut().zip(&lr.zeta[i]) {
                    *a += w * zi;
                }
            }
            acc
        };
        let q_twin = q2_dense(&hfull, &lr.c, &kd, &zeta_full, e_exact, 1e-3);

        let dev = (q_chart - q_twin).abs();
        worst = worst.max(dev);
        eprintln!(
            "tjcd_b7 lih trial {trial}: q(rotate_integrals chart) = {q_chart:.10}, \
             q(dense e^{{−sK̂}} twin) = {q_twin:.10}, Δ = {dev:.3e}"
        );
    }
    assert!(
        worst <= 1e-6,
        "the integral-rotation chart and the dense e^{{−sK̂}} operator chart disagree by \
         {worst:.3e}. They are two independent constructions of the SAME curve ψ(s) = \
         e^{{−sK̂}}(c+sζ); a disagreement is a sign/side error in rotate_integrals' use, not noise."
    );
}

#[test]
fn tjcd_b8_negative_arms() {
    for (label, ints, part, lr) in [
        ("lih", &lih_opt().0, &lih_opt().1, lih_lr_opt()),
        ("h2o", &h2o_opt().0, &h2o_opt().1, h2o_lr_opt()),
    ] {
        let good = lr.omega();
        let scale = good.iter().fold(0.0f64, |a, &x| a.max(x)).max(1.0);
        eprintln!(
            "tjcd_b8 {label}: reference ω[0..4] = {:?} (max ω = {scale:.6}), ‖z‖ = {:.3e}, \
             max|B_orb| = {:.3e}, max|B_cross| = {:.3e}",
            &good[..4.min(good.len())],
            lr.diag.z_norm,
            lr.diag.max_b_orb,
            lr.diag.max_b_cross
        );

        let arms: [(&str, DenseLrOpts); 3] = [
            (
                "drop B_orb",
                DenseLrOpts {
                    drop_b_orb: true,
                    ..DenseLrOpts::default()
                },
            ),
            (
                "flip sign of B_orb",
                DenseLrOpts {
                    flip_b_orb_sign: true,
                    ..DenseLrOpts::default()
                },
            ),
            (
                "halve the cross term",
                DenseLrOpts {
                    halve_b_cross: true,
                    ..DenseLrOpts::default()
                },
            ),
        ];
        for (name, o) in arms {
            let bad = dense_lr_full_with(ints, part, &o);
            let bad_omega = bad.omega();
            assert_eq!(
                bad_omega.len(),
                good.len(),
                "{label} [{name}]: dimension changed"
            );
            let dev = good
                .iter()
                .zip(&bad_omega)
                .fold(0.0f64, |a, (&g, &b)| a.max((g - b).abs()));
            eprintln!(
                "tjcd_b8 {label} [{name}]: max|Δω| = {dev:.3e} (relative {:.3e}); bad ω[0..3] = {:?}",
                dev / scale,
                &bad_omega[..3.min(bad_omega.len())]
            );
            assert!(
                dev >= 1e-9,
                "{label} [{name}]: this corruption must move ω well above the readout's own noise \
                 floor (~1e-12); measured only {dev:.3e} (relative to max ω = {scale:.6}: \
                 {:.3e}). ★ If the margin is genuinely small, that is a FINDING about the anchor's \
                 RPA character (‖z‖ = {:.3e}, max|B_orb| = {:.3e}, max|B_cross| = {:.3e}) — report \
                 the number rather than replacing the arm.",
                dev / scale,
                lr.diag.z_norm,
                lr.diag.max_b_orb,
                lr.diag.max_b_cross
            );
        }

        let flipped = dense_lr_full_with(
            ints,
            part,
            &DenseLrOpts {
                plus_g: true,
                ..DenseLrOpts::default()
            },
        );
        let om_flip = flipped.omega();
        let dev_om = good
            .iter()
            .zip(&om_flip)
            .fold(0.0f64, |a, (&g, &b)| a.max((g - b).abs()));
        let (hg, hf) = (lr.hrr_raw(), flipped.hrr_raw());
        let mut cross_dev = 0.0f64;
        let mut cross_scale = 0.0f64;
        for i in 0..lr.dim_ci {
            for a in 0..lr.dim_orb {
                let g = hg[i][lr.dim_ci + a];
                let f = hf[i][lr.dim_ci + a];
                cross_dev = cross_dev.max((f + g).abs());
                cross_scale = cross_scale.max(g.abs());
            }
        }
        eprintln!(
            "tjcd_b8 {label} [+Ĝ instead of −Ĝ]: max|Δω| = {dev_om:.3e} (EXPECTED ≈ 0 — the arm is \
             ω-invisible), max|H_cross(+) + H_cross(−)| = {cross_dev:.3e}, max|H_cross| = \
             {cross_scale:.3e}"
        );
        assert!(
            dev_om <= 1e-9 * scale,
            "{label}: the +Ĝ flip is a DHD congruence with a D-invariant metric and MUST leave ω \
             unchanged; got {dev_om:.3e}. If it moves ω, the metric is not D-invariant and the \
             assembly has a deeper problem."
        );
        assert!(
            cross_scale >= 1e-3,
            "{label}: the cross block is vacuous ({cross_scale:.3e}) — this negative arm has \
             nothing to bite on"
        );
        assert!(
            cross_dev <= 1e-10 * cross_scale.max(1.0),
            "{label}: every cross entry must flip sign under u_a = +Ĝ_a c; residual {cross_dev:.3e}"
        );

        let swapped = dense_lr_full_with(
            ints,
            part,
            &DenseLrOpts {
                swap_ab: true,
                ..DenseLrOpts::default()
            },
        );
        let dev_swap = good
            .iter()
            .zip(&swapped.omega())
            .fold(0.0f64, |a, (&g, &b)| a.max((g - b).abs()));
        let hrr_swap_dev = max_abs_diff(&lr.hrr(), &swapped.hrr());
        eprintln!(
            "tjcd_b8 {label} [A↔B swap]: max|Δω| = {dev_swap:.3e} (EXPECTED ≈ 0 — ω-invisible); \
             min_eig(H^rr) good {:.6e} vs swapped {:.6e} (Δ = {:.3e}); max|ΔH^rr| = \
             {hrr_swap_dev:.3e}",
            lr.diag.hrr_min_eig,
            swapped.diag.hrr_min_eig,
            (lr.diag.hrr_min_eig - swapped.diag.hrr_min_eig).abs()
        );
        assert!(
            dev_swap <= 1e-9 * scale,
            "{label}: the A↔B swap is ω-invariant by eig(X^½YX^½) = eig(XY); got {dev_swap:.3e}"
        );
        assert!(
            hrr_swap_dev >= 1e-3,
            "{label}: the A↔B swap changes H^rr by only {hrr_swap_dev:.3e} — B is numerically \
             negligible and the whole RPA character of the anchor is in doubt (min_eig(H^rr) good \
             {:.3e} vs swapped {:.3e})",
            lr.diag.hrr_min_eig,
            swapped.diag.hrr_min_eig
        );

        let unit_metric = dense_lr_full_with(
            ints,
            part,
            &DenseLrOpts {
                metric_identity: true,
                ..DenseLrOpts::default()
            },
        );
        let dev_metric = good
            .iter()
            .zip(&unit_metric.omega())
            .fold(0.0f64, |a, (&g, &b)| a.max((g - b).abs()));
        eprintln!(
            "tjcd_b8 {label} [G = I]: max|Δω| = {dev_metric:.3e}; μ(S) ∈ [{:.3e}, {:.3e}]",
            lr.diag.mu_min, lr.diag.mu_max
        );
        assert!(
            dev_metric >= 1e-9,
            "{label}: dropping the S^{{−½}} congruence must change ω (μ(S) spans \
             [{:.3e}, {:.3e}]); measured {dev_metric:.3e}",
            lr.diag.mu_min,
            lr.diag.mu_max
        );
    }
}

#[test]
fn tjcd_b9_williamson_dense_matches_shipped_kernel() {
    for (label, lr) in [("lih", lih_lr_opt()), ("h2o", h2o_lr_opt())] {
        let hrr = lr.hrr();
        let hii = lr.hii();
        let n = lr.dim();
        let mine = williamson_dense(&hrr, &hii).unwrap_or_else(|e| {
            panic!(
                "{label}: williamson_dense refused: {e} (min_eig(H^rr) = {:.3e})",
                lr.diag.hrr_min_eig
            )
        });
        let flat_r: Vec<f64> = hrr.iter().flat_map(|r| r.iter().copied()).collect();
        let flat_i: Vec<f64> = hii.iter().flat_map(|r| r.iter().copied()).collect();
        let shipped = williamson_reduced(&flat_r, &flat_i, n, 1e-10)
            .unwrap_or_else(|e| panic!("{label}: shipped williamson_reduced refused: {e:?}"));
        assert_eq!(mine.len(), shipped.len(), "{label}: ω count");
        let dev = mine
            .iter()
            .zip(&shipped)
            .fold(0.0f64, |a, (&x, &y)| a.max((x - y).abs()));
        let scale = shipped.iter().fold(0.0f64, |a, &x| a.max(x)).max(1.0);
        eprintln!(
            "tjcd_b9 {label}: n = {n}, max|ω_mine − ω_shipped| = {dev:.3e} (rel {:.3e}); \
             ω_mine[0..4] = {:?}; min_eig(H^rr) = {:.6e}",
            dev / scale,
            &mine[..4.min(mine.len())],
            lr.diag.hrr_min_eig
        );
        assert!(
            dev <= 1e-12 * scale,
            "{label}: the in-test Williamson readout must reproduce the shipped kernel to 1e-12 \
             relative; got {dev:.3e}"
        );
        for w in mine.windows(2) {
            assert!(w[0] <= w[1] + 1e-14, "{label}: ω must come back ascending");
        }
    }
}

#[test]
fn tjcd_b10_commutator_metric_congruence() {
    for (label, lr) in [("lih", lih_lr()), ("h2o", h2o_lr())] {
        let n = lr.dim_orb;
        let mut dev = 0.0f64;
        let mut antisym = 0.0f64;
        let mut scale = 0.0f64;
        for a in 0..n {
            for b in 0..n {
                dev = dev.max((lr.sigma[a][b] - lr.s[a][b]).abs());
                antisym = antisym.max((lr.sigma[a][b] - lr.sigma[b][a]).abs());
                scale = scale.max(lr.s[a][b].abs());
            }
        }
        eprintln!(
            "tjcd_b10 {label}: max|Σ − S| = {dev:.3e}, max|Σ − Σᵀ| = {antisym:.3e}, max|S| = \
             {scale:.3e} (diag reports sigma_vs_s = {:.3e}, sigma_antisym = {:.3e})",
            lr.diag.sigma_vs_s, lr.diag.sigma_antisym
        );
        assert!(
            scale >= 1e-3,
            "{label}: the orbital Gram is vacuous (max|S| = {scale:.3e})"
        );
        assert!(
            dev <= 1e-12 * scale.max(1.0),
            "{label}: the paper's commutator metric Σ and our normal Gram S must COINCIDE on the \
             non-redundant set (max deviation {dev:.3e}). If they do not, the two conventions are \
             genuinely different objects and every ω comparison against the paper carries that \
             difference — report it as a finding with this number, do NOT loosen the tolerance."
        );
        assert!(
            antisym <= 1e-12 * scale.max(1.0),
            "{label}: Σ must be symmetric for a REAL state; max|Σ − Σᵀ| = {antisym:.3e}"
        );
    }
}

#[test]
fn tjcd_b10b_nested_commutator_spot_check() {
    let (ints, part) = lih_dense();
    let lr = lih_lr();
    let d = ints.norb;
    let (nup, ndn) = electrons(ints);
    let sp = DetSpace::new(d, nup, ndn);
    let hfull = sc_hamiltonian_matrix(d, &ints.t, &ints.v, &sp);
    let lam = lr.lambda;
    let hrr = lr.hrr_raw();
    let hii = lr.hii_raw();
    let _ = part;

    let hx = |x: &[f64]| mat_vec(&hfull, x);
    let g = |a: usize, x: &[f64]| {
        let (p, q) = lr.pairs[a];
        apply_gpq(d, p, q, &sp, x)
    };

    let z: Vec<f64> = {
        let hc = hx(&lr.c);
        hc.iter().zip(&lr.c).map(|(&h, &c)| h - lam * c).collect()
    };
    let mut worst_rr = 0.0f64;
    let mut worst_ii = 0.0f64;
    let mut scale = 0.0f64;
    for a in 0..lr.dim_orb.min(4) {
        for b in 0..lr.dim_orb.min(4) {
            let gb_c = g(b, &lr.c);
            let h_gb_c = hx(&gb_c);
            let g_hc = g(b, &hx(&lr.c));
            let comm_hgb: Vec<f64> = h_gb_c.iter().zip(&g_hc).map(|(&x, &y)| x - y).collect();
            let ga_comm = g(a, &comm_hgb);
            let _ = &ga_comm;
            let ga_c = g(a, &lr.c);
            let lit_rr = 0.5
                * (dot_vec(&ga_c, &comm_hgb) + {
                    let ga_cc = g(a, &lr.c);
                    let h_ga_c = hx(&ga_cc);
                    let g_hc_a = g(a, &hx(&lr.c));
                    let comm_a: Vec<f64> =
                        h_ga_c.iter().zip(&g_hc_a).map(|(&x, &y)| x - y).collect();
                    dot_vec(&gb_c, &comm_a)
                });
            let want_rr = hrr[lr.dim_ci + a][lr.dim_ci + b];
            let want_ii = hii[lr.dim_ci + a][lr.dim_ci + b];
            let gagbc = g(a, &g(b, &lr.c));
            let gbgac = g(b, &g(a, &lr.c));
            let anti: Vec<f64> = gagbc
                .iter()
                .zip(&gbgac)
                .map(|(&x, &y)| 0.5 * (x + y))
                .collect();
            let b_lit = dot_vec(&z, &anti);
            let a_lit = lit_rr - b_lit;
            worst_rr = worst_rr.max((lit_rr - want_rr).abs());
            worst_ii = worst_ii.max((a_lit - b_lit - want_ii).abs());
            scale = scale.max(want_rr.abs());
        }
    }
    eprintln!(
        "tjcd_b10b lih: literal nested-commutator spot check over the first 4×4 orbital entries: \
         max|Δ H^rr| = {worst_rr:.3e}, max|Δ H^ii| = {worst_ii:.3e}, max|H^rr| = {scale:.3e}"
    );
    assert!(
        scale >= 1e-3,
        "lih: the orb-orb block is vacuous ({scale:.3e})"
    );
    assert!(
        worst_rr <= 1e-10 * scale.max(1.0),
        "lih: the literal double-commutator evaluation and the chart-native (A+B) must agree \
         entrywise; got {worst_rr:.3e}"
    );
    assert!(
        worst_ii <= 1e-10 * scale.max(1.0),
        "lih: the A/B SPLIT must survive the reconciliation too (H^ii = A − B); got {worst_ii:.3e}"
    );
}

#[test]
fn tjcd_b11_tier2_matches_tier1_entrywise() {
    for (label, ints, part, t1) in [
        ("lih", &lih_opt().0, &lih_opt().1, lih_lr_opt()),
        ("h2o", &h2o_opt().0, &h2o_opt().1, h2o_lr_opt()),
    ] {
        let t2 = dense_lr_cas(ints, part);
        assert_eq!(
            (t2.dim_ci, t2.dim_orb),
            (t1.dim_ci, t1.dim_orb),
            "{label}: the two tiers must build the same chart"
        );
        assert!(
            (t2.lambda - t1.lambda).abs() <= 1e-10 * t1.lambda.abs().max(1.0),
            "{label}: λ differs between tiers ({:.14} vs {:.14})",
            t2.lambda,
            t1.lambda
        );
        let dev_a = max_abs_diff(&t2.a, &t1.a);
        let dev_b = max_abs_diff(&t2.b, &t1.b);
        let dev_s = max_abs_diff(&t2.s, &t1.s);
        let scale = max_abs(&t1.a).max(max_abs(&t1.b)).max(1.0);
        let dev_om = t1
            .omega()
            .iter()
            .zip(&t2.omega())
            .fold(0.0f64, |acc, (&x, &y)| acc.max((x - y).abs()));
        eprintln!(
            "tjcd_b11 {label}: max|ΔA| = {dev_a:.3e}, max|ΔB| = {dev_b:.3e}, max|ΔS| = {dev_s:.3e}, \
             max|Δω| = {dev_om:.3e}, block scale = {scale:.3e}"
        );
        assert!(
            dev_a <= 1e-10 * scale,
            "{label}: Tier-2 A block differs by {dev_a:.3e}"
        );
        assert!(
            dev_b <= 1e-10 * scale,
            "{label}: Tier-2 B block differs by {dev_b:.3e} — ★ this is the block that carries the \
             E_pq/E_qp split, the one thing FD cannot see. A failure here means the 6-31G paper pin \
             would be certified against a wrong H^ii."
        );
        assert!(
            dev_s <= 1e-10 * scale,
            "{label}: Tier-2 metric differs by {dev_s:.3e}"
        );
        assert!(dev_om <= 1e-9, "{label}: Tier-2 ω differs by {dev_om:.3e}");
    }
}

#[test]
fn tjcd_b12_dense_casscf_optimizer_pins() {
    let (ints, part) = h2o_dense();
    let d = ints.norb;
    let (nup, ndn) = electrons(ints);
    let sp = DetSpace::new(d, nup, ndn);
    let (_full_pos, sp_act) = sp.cas_subset(part);
    let pairs = cas_pairs(part, d).keep;

    let lr = h2o_lr();
    let g = dense_orb_gradient(ints, part, &pairs, &lr.c_act, &sp_act);
    assert_eq!(
        g.len(),
        pairs.len(),
        "gradient length must be the pair count"
    );
    let e0 = e_of_c_kappa(
        ints,
        part,
        &sp_act,
        &pairs,
        &lr.c_act,
        &vec![0.0; pairs.len()],
    );
    let a_probe = (0..pairs.len())
        .max_by(|&i, &j| {
            g[i].abs()
                .partial_cmp(&g[j].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    eprintln!(
        "tjcd_b12 h2o: ‖g‖_inf = {:.3e} at pair {:?} (index {a_probe}); g = {:?}",
        max_abs_vec(&g),
        pairs[a_probe],
        g
    );
    assert!(
        g[a_probe].abs() >= 1e-4,
        "h2o at HF MOs: the orbital gradient must be non-vacuous ({:.3e}) or the FD certification \
         below has nothing to certify",
        g[a_probe].abs()
    );
    let mut prev: Option<(f64, f64)> = None;
    for &h in &[1e-3f64, 1e-4, 1e-5] {
        let mut kp = vec![0.0f64; pairs.len()];
        kp[a_probe] = h;
        let mut km = vec![0.0f64; pairs.len()];
        km[a_probe] = -h;
        let fd = (e_of_c_kappa(ints, part, &sp_act, &pairs, &lr.c_act, &kp)
            - e_of_c_kappa(ints, part, &sp_act, &pairs, &lr.c_act, &km))
            / (2.0 * h);
        let err = (fd - g[a_probe]).abs();
        eprintln!(
            "tjcd_b12 h2o: h = {h:.1e}  dE/dκ_a FD = {fd:.12}  analytic = {:.12}  err = {err:.3e}",
            g[a_probe]
        );
        if let Some((_hp, ep)) = prev {
            assert!(
                err <= ep * 1.5 || err <= 1e-9,
                "the analytic orbital gradient must be the FD limit; the error is not decreasing \
                 ({ep:.3e} -> {err:.3e}). ★ A CONSTANT offset (e.g. a stray factor of 2 from the \
                 strict-lower-triangle pairing convention) is exactly what this sweep exists to \
                 catch — FD is the decider, never the formula's typography."
            );
        }
        prev = Some((h, err));
    }
    let _ = e0;

    let out = dense_casscf(
        ints,
        part,
        &DenseCasscfOpts {
            g_tol: 1e-7,
            max_iter: 200,
            trust0: 0.5,
            shift_floor: 1e-6,
            verbose: false,
        },
    );
    let want = json_num(paper_json(), "h2o/bases/sto-3g/casscf_gs").unwrap();
    eprintln!(
        "tjcd_b12 h2o sto-3g: dense CASSCF E = {:.14} vs pyscf casscf_gs {want:.14} (Δ = {:.3e}); \
         iters = {}, ‖g‖ = {:.3e}, converged = {}",
        out.e_total,
        (out.e_total - want).abs(),
        out.iters,
        out.g_norm,
        out.converged
    );
    assert!(
        out.converged,
        "h2o sto-3g: the dense (c,κ) optimizer did not converge (‖g‖ = {:.3e} after {} iters). \
         ★ A NEAR-miss is worse than a failure: ω at a slightly-off optimum carries a \
         first-order-in-‖g‖ bias behind a converged-looking energy. Per the pre-committed \
         fallback, a system that stalls has its paper pin DEFERRED to Phase 5 and said so in the \
         report — never a loosened tolerance.",
        out.g_norm, out.iters
    );
    assert!(
        (out.e_total - want).abs() <= 1e-7,
        "h2o sto-3g: dense CASSCF {:.14} vs pyscf {want:.14}",
        out.e_total
    );

    let roots = json_nums(paper_json(), "h2o/bases/sto-3g/casci_roots").unwrap();
    for (k, &w) in roots.iter().enumerate() {
        let got = out.roots[k] + out.e0_tot;
        eprintln!(
            "tjcd_b12 h2o casci root {k}: {got:.12} vs JSON {w:.12} (Δ = {:.3e})",
            (got - w).abs()
        );
        assert!(
            (got - w).abs() <= 1e-7,
            "h2o sto-3g CAS-CI root {k} at the dense optimum: {got:.12} vs {w:.12}"
        );
    }
}
