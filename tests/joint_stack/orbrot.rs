// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop)]

use std::f64::consts::FRAC_PI_2;
use std::path::Path;

use qntt::CoeffTensor;
use qntt::integrals::{
    Integrals, IntegralsLoad, IntegralsOptions, PointGroupPolicy, load_fixture_dir,
};
use qntt::rdm::transition_1rdm;
use qntt::rotation::{
    GivensNetwork, RotateInfo, RotateOpts, apply_orbital_rotation, apply_orbital_signs,
    compose_rotations, expm_antisym, givens, givens_compose, givens_decompose, identity_matrix,
    kappa_apply, orthogonality_defect, rotate_integrals,
};
use qntt::su2::expand;
use qntt::su2::hamiltonian::{Su2SparseHamiltonian, xt_h_y as su2_xt_h_y};
use qntt::su2::ops::irrep_weights as su2_irrep_weights;
use qntt::su2::orbital::{
    apply_orbital_rotation as su2_apply_orbital_rotation,
    apply_orbital_signs as su2_apply_orbital_signs, kappa_apply as su2_kappa_apply,
};
use qntt::su2::qn::{Su2Sector, Su2Space, max_ranks as su2_max_ranks};
use qntt::su2::transition1rdm::su2_transition_1rdm;
use qntt::su2::vector::Su2TtVector;
use qntt::{
    C64, Irrep, PointGroup, Pooled, QnSpace, Sector, SparseHamiltonian, TtVector, irrep_weights,
    xt_h_y,
};
use qntt_testkit::pg::su2_capped_ranks;
use qntt_testkit::{randomized_su2_state, seeded};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::util::{
    basis_index, capped_randn, coeff2_flat, dense_gpq, dense_kappa_matrix, dets_for, flat_to_coeff,
    mat_expm, mat_scale, mat_vec, neg_flat, panic_message, random_antisym_flat, randomized_state,
    scale_flat, transpose_flat,
};

type Dets = Vec<(Vec<bool>, Vec<bool>)>;

fn full() -> RotateOpts {
    RotateOpts {
        eps: 0.0,
        chi: Pooled::NONE,
        threads: None,
    }
}

fn load_untyped(name: &str) -> Integrals<f64> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name);
    load_fixture_dir::<f64>(&dir, &IntegralsOptions::default())
        .expect(name)
        .ints
}

fn random_orthogonal(d: usize, det_minus: bool, rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
    let mut u = expm_antisym(&random_antisym_flat(d, rng), d);
    if det_minus {
        let c = d / 2;
        for i in 0..d {
            u[i * d + c] = -u[i * d + c];
        }
    }
    assert!(orthogonality_defect(&u, d) <= 1e-13);
    let net = givens_decompose(&u, d);
    let neg = net.signs.iter().filter(|&&s| s == -1.0).count();
    assert_eq!(neg % 2 == 1, det_minus, "det sign not in the network signs");
    assert!(
        net.gates.len() >= d,
        "a generic U should need ≥ d gates, got {}",
        net.gates.len()
    );
    u
}

fn diag(signs: &[f64]) -> Vec<f64> {
    let d = signs.len();
    let mut u = vec![0.0f64; d * d];
    for p in 0..d {
        u[p * d + p] = signs[p];
    }
    u
}

fn vmax(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn vdiff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn vnorm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vdist(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

fn mdiff(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max(vdiff(x, y)))
}

fn mmax(a: &[Vec<f64>]) -> f64 {
    a.iter().fold(0.0f64, |m, x| m.max(vmax(x)))
}

fn bits(a: &[Vec<f64>]) -> Vec<Vec<u64>> {
    a.iter()
        .map(|v| v.iter().map(|x| x.to_bits()).collect())
        .collect()
}

fn coeffs(dense: &[f64], dets: &Dets) -> Vec<f64> {
    dets.iter()
        .map(|(ou, od)| dense[basis_index(ou, od)])
        .collect()
}

fn sign_phase(signs: &[f64], dets: &Dets) -> Vec<f64> {
    dets.iter()
        .map(|(ou, od)| {
            let mut ph = 1.0;
            for (p, &s) in signs.iter().enumerate() {
                if s == -1.0 && (ou[p] as usize + od[p] as usize) % 2 == 1 {
                    ph = -ph;
                }
            }
            ph
        })
        .collect()
}

fn congruence(u: &[f64], g: &[f64], d: usize, transpose_left: bool) -> Vec<f64> {
    let ut = transpose_flat(u, d);
    if transpose_left {
        compose_rotations(&compose_rotations(&ut, g, d), u, d)
    } else {
        compose_rotations(&compose_rotations(u, g, d), &ut, d)
    }
}

trait Layer: Clone {
    fn rot(&self, u: &[f64], opts: &RotateOpts) -> (Self, RotateInfo);
    fn members(&self) -> Vec<Vec<f64>>;
    fn member_sectors(&self) -> Vec<(usize, usize)>;
    fn nrm(&self) -> f64;
    fn gauge_free_nrm(&self) -> f64;
    fn sites(&self) -> usize;
    fn bond_max(&self) -> usize;
    fn same_ranks(&self, other: &Self) -> bool;
    fn same_ranks_at(&self, other: &Self, b: usize) -> bool;
    fn ortho_flag(&self) -> Option<usize>;
    fn centred(&self, c: usize) -> Self;
    fn scaled(&self, a: f64) -> Self;
    fn signed(&self, signs: &[f64]) -> Self;
    fn kappa(&self, k: &CoeffTensor<f64>) -> Self;
    fn gamma(&self) -> Vec<f64>;
    fn xhy(x: &Self, ints: &Integrals<f64>, y: &Self) -> f64;
    fn valid(&self) -> bool;
    fn irreps(&self, pg: PointGroup) -> Vec<f64>;
}

impl Layer for TtVector<f64> {
    fn rot(&self, u: &[f64], opts: &RotateOpts) -> (Self, RotateInfo) {
        apply_orbital_rotation(self, u, opts)
    }
    fn members(&self) -> Vec<Vec<f64>> {
        vec![self.to_dense()]
    }
    fn member_sectors(&self) -> Vec<(usize, usize)> {
        vec![(self.space.nup, self.space.ndn)]
    }
    fn nrm(&self) -> f64 {
        self.norm()
    }
    fn gauge_free_nrm(&self) -> f64 {
        let mut z = self.clone();
        z.ortho = None;
        z.norm()
    }
    fn sites(&self) -> usize {
        self.space.d
    }
    fn bond_max(&self) -> usize {
        self.max_total_bond_rank()
    }
    fn same_ranks(&self, other: &Self) -> bool {
        self.ranks == other.ranks
    }
    fn same_ranks_at(&self, other: &Self, b: usize) -> bool {
        self.ranks[b] == other.ranks[b]
    }
    fn ortho_flag(&self) -> Option<usize> {
        self.ortho
    }
    fn centred(&self, c: usize) -> Self {
        let mut z = self.clone();
        z.move_core(c, true);
        z
    }
    fn scaled(&self, a: f64) -> Self {
        let mut z = self.clone();
        z.scale_mut(a);
        z
    }
    fn signed(&self, signs: &[f64]) -> Self {
        let mut z = self.clone();
        apply_orbital_signs(&mut z, signs);
        z
    }
    fn kappa(&self, k: &CoeffTensor<f64>) -> Self {
        kappa_apply(k, self)
    }
    fn gamma(&self) -> Vec<f64> {
        coeff2_flat(&transition_1rdm(self, self), self.space.d)
    }
    fn xhy(x: &Self, ints: &Integrals<f64>, y: &Self) -> f64 {
        let h = SparseHamiltonian::new(x.space, &ints.t, &ints.v, None, f64::EPSILON);
        xt_h_y(x, &h, y)
    }
    fn valid(&self) -> bool {
        self.check()
    }
    fn irreps(&self, pg: PointGroup) -> Vec<f64> {
        irrep_weights(self, pg)
    }
}

impl Layer for Su2TtVector<f64> {
    fn rot(&self, u: &[f64], opts: &RotateOpts) -> (Self, RotateInfo) {
        su2_apply_orbital_rotation(self, u, opts)
    }
    fn members(&self) -> Vec<Vec<f64>> {
        su2_members(&self.space)
            .into_iter()
            .map(|tz| expand(self, tz).to_dense())
            .collect()
    }
    fn member_sectors(&self) -> Vec<(usize, usize)> {
        let n = self.space.n as i64;
        su2_members(&self.space)
            .into_iter()
            .map(|tz| (((n + tz) / 2) as usize, ((n - tz) / 2) as usize))
            .collect()
    }
    fn nrm(&self) -> f64 {
        self.norm()
    }
    fn gauge_free_nrm(&self) -> f64 {
        let mut z = self.clone();
        z.ortho = None;
        z.norm()
    }
    fn sites(&self) -> usize {
        self.space.d
    }
    fn bond_max(&self) -> usize {
        self.max_total_bond_rank()
    }
    fn same_ranks(&self, other: &Self) -> bool {
        self.ranks == other.ranks
    }
    fn same_ranks_at(&self, other: &Self, b: usize) -> bool {
        self.ranks[b] == other.ranks[b]
    }
    fn ortho_flag(&self) -> Option<usize> {
        self.ortho
    }
    fn centred(&self, c: usize) -> Self {
        let mut z = self.clone();
        z.move_core(c, true);
        z
    }
    fn scaled(&self, a: f64) -> Self {
        let mut z = self.clone();
        z.scale_mut(a);
        z
    }
    fn signed(&self, signs: &[f64]) -> Self {
        let mut z = self.clone();
        su2_apply_orbital_signs(&mut z, signs);
        z
    }
    fn kappa(&self, k: &CoeffTensor<f64>) -> Self {
        su2_kappa_apply(k, self)
    }
    fn gamma(&self) -> Vec<f64> {
        coeff2_flat(&su2_transition_1rdm(self, self), self.space.d)
    }
    fn xhy(x: &Self, ints: &Integrals<f64>, y: &Self) -> f64 {
        let h = Su2SparseHamiltonian::new(x.space, &ints.t, &ints.v, None, f64::EPSILON);
        su2_xt_h_y(x, &h, y)
    }
    fn valid(&self) -> bool {
        self.check()
    }
    fn irreps(&self, pg: PointGroup) -> Vec<f64> {
        su2_irrep_weights(self, pg)
    }
}

fn su2_members(space: &Su2Space) -> Vec<i64> {
    let two_s = space.two_s as i64;
    (0..=two_s).map(|i| -two_s + 2 * i).collect()
}

fn ab_state(space: QnSpace, lo: usize, hi: usize, seed: u64) -> TtVector<f64> {
    randomized_state::<f64, _>(space, lo, hi, &mut seeded(seed))
}

fn su2_state(space: Su2Space, lo: usize, hi: usize, seed: u64) -> Su2TtVector<f64> {
    randomized_su2_state::<f64, _>(space, lo, hi, &mut seeded(seed))
}

fn unit<L: Layer>(x: &L) -> L {
    x.scaled(1.0 / x.nrm())
}

fn check_full_rank_info<L: Layer>(label: &str, x: &L, u: &[f64], r: &L, info: &RotateInfo) {
    let d = x.sites();
    let net = givens_decompose(u, d);
    let nx = x.nrm();
    assert_eq!(info.gates, net.gates.len(), "{label}: gates");
    assert_eq!(
        info.discarded.len(),
        info.gates,
        "{label}: one discard per gate"
    );
    for (g, &w) in info.discarded.iter().enumerate() {
        assert_eq!(w, 0.0, "{label}: full-rank discarded[{g}] = {w:e}");
    }
    assert_eq!(
        info.error_bound, 0.0,
        "{label}: full-rank error_bound {:e}",
        info.error_bound
    );
    assert!(
        (info.norm_out - 1.0).abs() <= 1e-12,
        "{label}: full-rank norm_out {}",
        info.norm_out
    );
    assert!(
        (r.nrm() - nx).abs() <= 1e-12 * nx,
        "{label}: ‖Ûx‖ = {} vs ‖x‖ = {nx}",
        r.nrm()
    );
    assert!(
        r.ortho_flag().is_some(),
        "{label}: result carries no centre"
    );
    assert!(
        (r.nrm() - r.gauge_free_nrm()).abs() <= 1e-12 * nx,
        "{label}: orthogonality flag lies: flagged norm {} vs gauge-free {}",
        r.nrm(),
        r.gauge_free_nrm()
    );
    assert!(r.valid(), "{label}: result fails check()");
    assert!(
        info.max_rank >= r.bond_max(),
        "{label}: max_rank {} < result bond {}",
        info.max_rank,
        r.bond_max()
    );
}

#[test]
fn trot_opts_default() {
    let o = RotateOpts::default();
    assert_eq!(o.eps, 0.0);
    assert!(o.chi.is_none(), "default chi must be Pooled::NONE");
    assert_eq!(o.threads, None);
}

fn t5_case<L: Layer>(label: &str, ints: &Integrals<f64>, x: &L, y: &L, seed: u64) {
    let d = x.sites();
    let reference = L::xhy(x, ints, y);
    let scale = reference.abs().max(1.0);
    for det_minus in [false, true] {
        let tag = format!("{label} det {}", if det_minus { "-1" } else { "+1" });
        let u = random_orthogonal(d, det_minus, &mut seeded(seed + det_minus as u64));
        let net = givens_decompose(&u, d);
        let rints = rotate_integrals(ints, &u);
        let (ux, ix) = x.rot(&u, &full());
        let (uy, iy) = y.rot(&u, &full());
        check_full_rank_info(&format!("{tag} x"), x, &u, &ux, &ix);
        check_full_rank_info(&format!("{tag} y"), y, &u, &uy, &iy);
        let got = L::xhy(&ux, &rints, &uy);
        let err = (got - reference).abs();

        let rev = givens_compose(&GivensNetwork {
            d,
            signs: net.signs.clone(),
            gates: net.gates.iter().rev().copied().collect(),
        });
        let (rx, _) = x.rot(&rev, &full());
        let (ry, _) = y.rot(&rev, &full());
        let miss_rev = (L::xhy(&rx, &rints, &ry) - reference).abs();
        let ut = transpose_flat(&u, d);
        let (tx, _) = x.rot(&ut, &full());
        let (ty, _) = y.rot(&ut, &full());
        let miss_t = (L::xhy(&tx, &rints, &ty) - reference).abs();
        println!(
            "trot_t5 {tag}: ⟨x,Hy⟩ = {reference:.6}, |Δ| = {err:e}; reversed order misses by \
             {miss_rev:e}, Uᵀ misses by {miss_t:e} ({} gates)",
            net.gates.len()
        );
        assert!(
            err <= 1e-12 * scale,
            "{tag}: ⟨Ûx, H(rot(ints,U)) Ûy⟩ − ⟨x,Hy⟩ = {err:e}"
        );
        assert!(
            miss_rev > 1e-6 * scale,
            "{tag}: reversed gate order not discriminated ({miss_rev:e})"
        );
        assert!(
            miss_t > 1e-6 * scale,
            "{tag}: Uᵀ not discriminated ({miss_t:e})"
        );
    }
}

#[test]
fn trot_t5_convention_abelian() {
    let h4 = load_untyped("h4_chain_stretch");
    let lih = load_untyped("lih_sto3g");
    for (label, ints, space, seed) in [
        ("h4 (4,2,2)", &h4, QnSpace::new(4, 2, 2), 0x7_5A01u64),
        ("lih (6,2,2)", &lih, QnSpace::new(6, 2, 2), 0x7_5A11),
    ] {
        let x = ab_state(space, 1, 2, seed);
        let y = ab_state(space, 1, 2, seed + 100);
        t5_case(&format!("abelian {label}"), ints, &x, &y, seed + 200);
    }
}

#[test]
fn trot_t5_convention_su2() {
    let h4 = load_untyped("h4_chain_stretch");
    let lih = load_untyped("lih_sto3g");
    for (label, ints, space, seed) in [
        ("h4 (4,4,0)", &h4, Su2Space::new(4, 4, 0), 0x7_5B01u64),
        ("h4 (4,4,2)", &h4, Su2Space::new(4, 4, 2), 0x7_5B11),
        ("lih (6,4,0)", &lih, Su2Space::new(6, 4, 0), 0x7_5B21),
        ("lih (6,3,1)", &lih, Su2Space::new(6, 3, 1), 0x7_5B31),
    ] {
        let x = su2_state(space, 1, 2, seed);
        let y = su2_state(space, 1, 2, seed + 100);
        t5_case(&format!("su2 {label}"), ints, &x, &y, seed + 200);
    }
}

fn t6_case<L: Layer>(label: &str, x: &L, seed: u64) {
    let d = x.sites();
    let x = unit(x);
    let k = scale_flat(&random_antisym_flat(d, &mut seeded(seed)), 0.5);
    let kc = flat_to_coeff(d, &k);
    let h = 1e-5;
    let (a, _) = x.rot(&expm_antisym(&scale_flat(&k, -h), d), &full());
    let (b, _) = x.rot(&expm_antisym(&scale_flat(&k, h), d), &full());
    let (am, bm, want) = (a.members(), b.members(), x.kappa(&kc).members());
    for (i, w) in want.iter().enumerate() {
        let fd: Vec<f64> = am[i]
            .iter()
            .zip(&bm[i])
            .map(|(p, m)| (p - m) / (2.0 * h))
            .collect();
        let wn = vnorm(w);
        let rel = vdist(&fd, w) / wn;
        let neg: Vec<f64> = w.iter().map(|v| -v).collect();
        let flip = vdist(&fd, &neg) / wn;
        println!(
            "trot_t6 {label} member {i}: ‖κ̂x‖ = {wn:.4}, FD relative error {rel:e}, sign-flipped \
             generator misses by {flip:.3}"
        );
        assert!(wn > 1e-3, "{label}: vacuous generator ‖κ̂x‖ = {wn:e}");
        assert!(rel <= 1e-6, "{label} member {i}: FD vs κ̂x {rel:e}");
        assert!(
            flip > 1.0,
            "{label} member {i}: −κ̂x not discriminated ({flip:e})"
        );
    }
}

#[test]
fn trot_t6_generator() {
    t6_case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_6A01),
        0x7_6A02,
    );
    t6_case(
        "su2 (6,4,0)",
        &su2_state(Su2Space::new(6, 4, 0), 1, 2, 0x7_6B01),
        0x7_6B02,
    );
    t6_case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_6B11),
        0x7_6B12,
    );
}

fn t7_case<L: Layer>(label: &str, x: &L, seed: u64) {
    let d = x.sites();
    let x = unit(x);
    let g0 = x.gamma();
    let scale = vmax(&g0).max(1.0);
    for det_minus in [false, true] {
        let tag = format!("{label} det {}", if det_minus { "-1" } else { "+1" });
        let u = random_orthogonal(d, det_minus, &mut seeded(seed + det_minus as u64));
        let (ux, _) = x.rot(&u, &full());
        let g1 = ux.gamma();
        let e_ut = vdiff(&g1, &congruence(&u, &g0, d, true));
        let e_u = vdiff(&g1, &congruence(&u, &g0, d, false));
        println!("trot_t7 {tag}: |γ(Ûx) − UᵀγU| = {e_ut:e}, |γ(Ûx) − UγUᵀ| = {e_u:e}");
        assert!(e_ut <= 1e-12 * scale, "{tag}: γ(Ûx) ≠ Uᵀ γ(x) U ({e_ut:e})");
        assert!(
            e_u > 1e-6 * scale,
            "{tag}: R = U not discriminated ({e_u:e})"
        );
    }
}

#[test]
fn trot_t7_one_rdm_transform() {
    t7_case(
        "abelian (4,2,2)",
        &ab_state(QnSpace::new(4, 2, 2), 1, 2, 0x7_7A01),
        0x7_7A02,
    );
    t7_case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_7A11),
        0x7_7A12,
    );
    t7_case(
        "su2 (4,4,2)",
        &su2_state(Su2Space::new(4, 4, 2), 1, 2, 0x7_7B01),
        0x7_7B02,
    );
    t7_case(
        "su2 (6,4,0)",
        &su2_state(Su2Space::new(6, 4, 0), 1, 2, 0x7_7B11),
        0x7_7B12,
    );
    t7_case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_7B21),
        0x7_7B22,
    );
}

fn t8_case<L: Layer>(label: &str, x: &L, seed: u64) {
    let d = x.sites();
    let x = unit(x);
    let k = random_antisym_flat(d, &mut seeded(seed));
    let kc = flat_to_coeff(d, &k);
    let e = expm_antisym(&neg_flat(&k), d);
    let mut signs = vec![1.0f64; d];
    signs[1] = -1.0;
    let dm = diag(&signs);
    let (ye, ie) = x.rot(&e, &full());
    let (yde, _) = x.rot(&compose_rotations(&dm, &e, d), &full());
    let (yed, _) = x.rot(&compose_rotations(&e, &dm, d), &full());
    assert!(ie.gates > 0);
    let (xm, yem, ydem, yedm) = (x.members(), ye.members(), yde.members(), yed.members());
    for (i, &(nup, ndn)) in x.member_sectors().iter().enumerate() {
        let dets = dets_for(d, nup, ndn);
        let w = mat_expm(&dense_kappa_matrix(d, &kc, &dets));
        let c = coeffs(&xm[i], &dets);
        assert!(
            (vnorm(&c) - vnorm(&xm[i])).abs() <= 1e-12,
            "{label}: member {i} not in sector ({nup},{ndn})"
        );
        let phase = sign_phase(&signs, &dets);
        let pc: Vec<f64> = c.iter().zip(&phase).map(|(a, p)| a * p).collect();
        let wc = mat_vec(&w, &c);
        let want_e = wc.clone();
        let want_de = mat_vec(&w, &pc);
        let want_ed: Vec<f64> = wc.iter().zip(&phase).map(|(a, p)| a * p).collect();
        let got_e = coeffs(&yem[i], &dets);
        let got_de = coeffs(&ydem[i], &dets);
        let got_ed = coeffs(&yedm[i], &dets);
        let (err_e, err_de, err_ed) = (
            vdiff(&got_e, &want_e),
            vdiff(&got_de, &want_de),
            vdiff(&got_ed, &want_ed),
        );
        let swapped = vdiff(&got_de, &want_ed);
        let phase_move = vdist(&pc, &c);
        let moved = vdiff(&want_e, &c);
        println!(
            "trot_t8 {label} member {i} ({nup},{ndn}), {} dets: exp(κ̂) {err_e:e}, D·E {err_de:e}, \
             E·D {err_ed:e}; swapped oracle misses by {swapped:e}; ‖φ⊙c − c‖ = {phase_move:.3e}, \
             max|Wc − c| = {moved:.3e}",
            dets.len()
        );
        assert!(
            err_e <= 1e-12,
            "{label} member {i}: Û(exp(−κ)) vs exp(κ̂) {err_e:e}"
        );
        assert!(
            err_de <= 1e-12,
            "{label} member {i}: Û(D·E) vs W(φ⊙c) {err_de:e}"
        );
        assert!(
            err_ed <= 1e-12,
            "{label} member {i}: Û(E·D) vs φ⊙(Wc) {err_ed:e}"
        );
        assert!(
            swapped > 1e-6,
            "{label} member {i}: D·E/E·D not discriminated"
        );
        assert!(phase_move > 1e-3, "{label} member {i}: vacuous sign phase");
        assert!(moved > 1e-3, "{label} member {i}: vacuous rotation");
    }
}

#[test]
fn trot_t8_dense_oracle_abelian() {
    t8_case(
        "abelian (4,2,2)",
        &ab_state(QnSpace::new(4, 2, 2), 1, 2, 0x7_8A01),
        0x7_8A02,
    );
    t8_case(
        "abelian (6,2,1)",
        &ab_state(QnSpace::new(6, 2, 1), 1, 2, 0x7_8A11),
        0x7_8A12,
    );
}

#[test]
fn trot_t8_dense_oracle_su2() {
    t8_case(
        "su2 (4,4,0)",
        &su2_state(Su2Space::new(4, 4, 0), 1, 2, 0x7_8B01),
        0x7_8B02,
    );
    t8_case(
        "su2 (4,4,2)",
        &su2_state(Su2Space::new(4, 4, 2), 1, 2, 0x7_8B11),
        0x7_8B12,
    );
    t8_case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_8B21),
        0x7_8B22,
    );
}

fn t9_case<L: Layer>(label: &str, x: &L, seed: u64) {
    let d = x.sites();
    let x = unit(x);
    let u1 = random_orthogonal(d, false, &mut seeded(seed));
    let u2 = random_orthogonal(d, true, &mut seeded(seed + 1));
    let u12 = compose_rotations(&u1, &u2, d);
    let comm = vdiff(&u12, &compose_rotations(&u2, &u1, d));
    assert!(comm > 1e-2, "{label}: U₁, U₂ commute ({comm:e})");
    let xm = x.members();
    for (name, u) in [("U1", &u1), ("U2", &u2)] {
        let (a, _) = x.rot(u, &full());
        let (b, _) = a.rot(&transpose_flat(u, d), &full());
        let err = mdiff(&b.members(), &xm);
        let moved = mdiff(&a.members(), &xm);
        println!("trot_t9 {label} inverse {name}: {err:e} (Û moved x by {moved:e})");
        assert!(err <= 1e-12, "{label}: Û({name}ᵀ)Û({name})x ≠ x ({err:e})");
        assert!(moved > 1e-3, "{label}: vacuous {name}");
    }
    let (a1, _) = x.rot(&u1, &full());
    let (a12, _) = a1.rot(&u2, &full());
    let (a2, _) = x.rot(&u2, &full());
    let (a21, _) = a2.rot(&u1, &full());
    let (p, _) = x.rot(&u12, &full());
    let pm = p.members();
    let err = mdiff(&a12.members(), &pm);
    let miss = mdiff(&a21.members(), &pm);
    println!("trot_t9 {label}: |Û(U₂)Û(U₁)x − Û(U₁U₂)x| = {err:e}, other order misses by {miss:e}");
    assert!(err <= 1e-12, "{label}: Û(U₁U₂) ≠ Û(U₂)∘Û(U₁) ({err:e})");
    assert!(
        miss > 1e-6,
        "{label}: composition order not discriminated ({miss:e})"
    );
}

#[test]
fn trot_t9_inverse_and_composition() {
    t9_case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_9A01),
        0x7_9A02,
    );
    t9_case(
        "su2 (6,4,0)",
        &su2_state(Su2Space::new(6, 4, 0), 1, 2, 0x7_9B01),
        0x7_9B02,
    );
    t9_case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_9B11),
        0x7_9B12,
    );
}

fn t10_ledger<L: Layer>(
    label: &str,
    x: &L,
    u: &[f64],
    opts: &RotateOpts,
) -> (L, RotateInfo, f64, f64) {
    let d = x.sites();
    let nx = x.nrm();
    let (exact, _) = x.rot(u, &full());
    let (r, info) = x.rot(u, opts);
    assert_eq!(
        info.gates,
        givens_decompose(u, d).gates.len(),
        "{label}: gates"
    );
    assert_eq!(
        info.discarded.len(),
        info.gates,
        "{label}: one discard per gate"
    );
    for (g, &w) in info.discarded.iter().enumerate() {
        assert!(w >= -1e-14, "{label}: discarded[{g}] = {w:e} < 0");
    }
    let sum: f64 = info.discarded.iter().sum();
    let ledger = (1.0f64 - info.norm_out * info.norm_out - sum).abs();
    assert!(
        ledger <= 1e-12,
        "{label}: 1 − norm_out² = {} vs Σ discarded = {sum} (|Δ| = {ledger:e})",
        1.0 - info.norm_out * info.norm_out
    );
    let eb: f64 = info.discarded.iter().map(|w| w.max(0.0).sqrt()).sum();
    assert!(
        (info.error_bound - eb).abs() <= 1e-14 * eb.max(1.0),
        "{label}: error_bound {} vs Σ√discarded {eb}",
        info.error_bound
    );
    assert!(
        (r.nrm() - info.norm_out * nx).abs() <= 1e-12 * nx,
        "{label}: ‖result‖ = {} vs norm_out·‖x‖ = {}",
        r.nrm(),
        info.norm_out * nx
    );
    assert!(
        (r.nrm() - r.gauge_free_nrm()).abs() <= 1e-12 * nx,
        "{label}: orthogonality flag lies"
    );
    assert!(r.valid(), "{label}: result fails check()");
    assert!(
        info.max_rank >= r.bond_max(),
        "{label}: max_rank below a result bond"
    );
    let mut worst = 0.0f64;
    for (i, (a, b)) in r.members().iter().zip(exact.members().iter()).enumerate() {
        let dist = vdist(a, b);
        assert!(
            dist <= info.error_bound * nx + 1e-12 * nx,
            "{label} member {i}: ‖Ûx − result‖ = {dist:e} > error_bound·‖x‖ = {:e}",
            info.error_bound * nx
        );
        worst = worst.max(dist / nx);
    }
    println!(
        "trot_t10 {label}: gates {}, Σ discarded {sum:e}, error_bound {:e}, actual ‖Δ‖/‖x‖ \
         {worst:e}, norm_out {:.12}, max_rank {}, result max bond {}",
        info.gates,
        info.error_bound,
        info.norm_out,
        info.max_rank,
        r.bond_max()
    );
    (r, info, sum, worst)
}

fn t10_case<L: Layer>(label: &str, x: &L, cap: usize, eps: f64, seed: u64) {
    let d = x.sites();
    assert!(
        x.bond_max() <= cap,
        "{label}: input max bond {} exceeds the cap {cap} (the max_rank ≤ cap check needs it)",
        x.bond_max()
    );
    let u = random_orthogonal(d, true, &mut seeded(seed));
    let opts = RotateOpts {
        eps: 0.0,
        chi: Pooled(cap),
        threads: None,
    };
    let (r, info, sum, worst) = t10_ledger(&format!("{label} chi={cap}"), x, &u, &opts);
    assert!(
        sum > 1e-6,
        "{label} chi={cap}: the cap did not bind (Σ discarded {sum:e})"
    );
    assert!(
        worst > 1e-8,
        "{label} chi={cap}: result indistinguishable from exact"
    );
    assert!(
        info.error_bound < 1.0,
        "{label} chi={cap}: bound is trivial ({:e})",
        info.error_bound
    );
    assert!(
        info.max_rank <= cap,
        "{label}: max_rank {} > cap {cap}",
        info.max_rank
    );
    assert!(
        r.bond_max() <= cap,
        "{label}: result bond {} > cap {cap}",
        r.bond_max()
    );
    let opts = RotateOpts {
        eps,
        chi: Pooled::NONE,
        threads: None,
    };
    let (_, info, sum, _) = t10_ledger(&format!("{label} eps={eps:e}"), x, &u, &opts);
    assert!(
        sum > 1e-10,
        "{label} eps={eps:e}: nothing dropped (Σ discarded {sum:e})"
    );
    for (g, &w) in info.discarded.iter().enumerate() {
        assert!(
            w <= eps * eps + 1e-14,
            "{label} eps={eps:e}: discarded[{g}] = {w:e} > eps² = {:e}",
            eps * eps
        );
    }
}

#[test]
fn trot_t10_truncation_bound() {
    let xa = ab_state(QnSpace::new(6, 2, 2), 1, 1, 0x7_AA01).scaled(2.5);
    t10_case("abelian (6,2,2)", &xa, 18, 1e-2, 0x7_AA02);
    let xs = su2_state(Su2Space::new(6, 4, 0), 1, 1, 0x7_AB01).scaled(2.5);
    t10_case("su2 (6,4,0)", &xs, 10, 1e-2, 0x7_AB02);
    let xs = su2_state(Su2Space::new(6, 3, 1), 1, 1, 0x7_AB11).scaled(0.4);
    t10_case("su2 (6,3,1)", &xs, 8, 1e-2, 0x7_AB12);
}

#[test]
fn trot_t10_scale_covariance() {
    fn case<L: Layer>(label: &str, x: &L, cap: usize, seed: u64) {
        let u = random_orthogonal(x.sites(), false, &mut seeded(seed));
        let opts = RotateOpts {
            eps: 0.0,
            chi: Pooled(cap),
            threads: None,
        };
        let (r1, i1) = x.rot(&u, &opts);
        let (r3, i3) = x.scaled(3.0).rot(&u, &opts);
        let want: Vec<Vec<f64>> = r1
            .members()
            .iter()
            .map(|m| m.iter().map(|v| 3.0 * v).collect())
            .collect();
        let err = mdiff(&r3.members(), &want) / mmax(&want);
        let sum: f64 = i1.discarded.iter().sum();
        println!("trot_t10 scale {label}: relative {err:e}, Σ discarded {sum:e}");
        assert!(sum > 1e-6, "{label}: cap {cap} did not bind");
        assert!(err <= 1e-13, "{label}: Û(3x) ≠ 3Û(x) ({err:e})");
        assert!(
            (i1.norm_out - i3.norm_out).abs() <= 1e-14,
            "{label}: norm_out"
        );
        assert_eq!(i1.discarded.len(), i3.discarded.len());
        for (g, (a, b)) in i1.discarded.iter().zip(&i3.discarded).enumerate() {
            assert!(
                (a - b).abs() <= 1e-14,
                "{label}: discarded[{g}] {a:e} vs {b:e}"
            );
        }
    }
    case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 1, 0x7_AC01),
        12,
        0x7_AC02,
    );
    case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 1, 0x7_AC11),
        6,
        0x7_AC12,
    );
}

#[test]
fn trot_cross_layer() {
    for (label, space, seed) in [
        ("(6,4,0)", Su2Space::new(6, 4, 0), 0x7_C001u64),
        ("(6,3,1)", Su2Space::new(6, 3, 1), 0x7_C011),
        ("(4,4,2)", Su2Space::new(4, 4, 2), 0x7_C021),
    ] {
        let x = unit(&su2_state(space, 1, 2, seed));
        let d = space.d;
        let mut us = vec![
            ("det +1", random_orthogonal(d, false, &mut seeded(seed + 1))),
            ("det -1", random_orthogonal(d, true, &mut seeded(seed + 2))),
        ];
        let (perm, sgn) = signed_permutation(d);
        us.push(("signed permutation", signed_permutation_matrix(&perm, &sgn)));
        for (name, u) in &us {
            let (r, _) = x.rot(u, &full());
            for tz in su2_members(&space) {
                let xa = expand(&x, tz);
                let (ra, _) = xa.rot(u, &full());
                let (got, want) = (expand(&r, tz).to_dense(), ra.to_dense());
                let err = vdiff(&got, &want);
                let moved = vdiff(&want, &xa.to_dense());
                println!("trot_cross {label} {name} tz={tz}: {err:e} (moved {moved:e})");
                assert!(
                    err <= 1e-12,
                    "{label} {name} tz={tz}: su2 vs abelian {err:e}"
                );
                assert!(moved > 1e-3, "{label} {name} tz={tz}: vacuous");
            }
        }
    }
}

#[test]
fn trot_identity_and_sign_diagonal() {
    fn case<L: Layer>(label: &str, x: &L) {
        let d = x.sites();
        let mut s = vec![1.0f64; d];
        s[0] = -1.0;
        s[d / 2] = -1.0;
        s[d - 1] = -1.0;
        let tight = RotateOpts {
            eps: 0.5,
            chi: Pooled(1),
            threads: None,
        };
        for opts in [full(), tight] {
            for (name, u, want) in [
                ("identity", identity_matrix(d), x.clone()),
                ("sign diagonal", diag(&s), x.signed(&s)),
            ] {
                let tag = format!("{label} {name} eps={}", opts.eps);
                let (r, info) = x.rot(&u, &opts);
                assert_eq!(info.gates, 0, "{tag}: gates");
                assert!(info.discarded.is_empty(), "{tag}: discarded");
                assert_eq!(info.error_bound, 0.0, "{tag}: error_bound");
                assert_eq!(info.norm_out, 1.0, "{tag}: norm_out");
                assert_eq!(
                    info.max_rank,
                    x.bond_max(),
                    "{tag}: zero-gate max_rank is the input's largest bond"
                );
                assert_eq!(
                    bits(&r.members()),
                    bits(&want.members()),
                    "{tag}: not bitwise"
                );
                assert!(r.same_ranks(x), "{tag}: ranks changed");
                assert_eq!(r.ortho_flag(), x.ortho_flag(), "{tag}: ortho changed");
            }
        }
        assert!(
            mdiff(&x.signed(&s).members(), &x.members()) > 1e-3,
            "{label}: vacuous signs"
        );
    }
    case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_D001),
    );
    let mut xc = ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_D002);
    xc.move_core(2, true);
    case("abelian (6,2,2) centred", &xc);
    case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_D011),
    );
}

fn signed_permutation(d: usize) -> (Vec<usize>, Vec<f64>) {
    assert_eq!(d % 2, 0);
    let perm: Vec<usize> = if d == 6 {
        vec![3, 0, 5, 1, 4, 2]
    } else {
        vec![2, 0, 3, 1]
    };
    let sgn: Vec<f64> = (0..d)
        .map(|i| if i % 3 == 1 { -1.0 } else { 1.0 })
        .collect();
    (perm, sgn)
}

fn signed_permutation_matrix(perm: &[usize], sgn: &[f64]) -> Vec<f64> {
    let d = perm.len();
    let mut u = vec![0.0f64; d * d];
    for new in 0..d {
        u[perm[new] * d + new] = sgn[new];
    }
    assert_eq!(orthogonality_defect(&u, d), 0.0);
    u
}

#[test]
fn trot_signed_permutation_abelian() {
    let d = 6;
    let space = QnSpace::new(d, 2, 1);
    let x = unit(&ab_state(space, 1, 2, 0x7_E001));
    let (perm, sgn) = signed_permutation(d);
    let u = signed_permutation_matrix(&perm, &sgn);
    let net = givens_decompose(&u, d);
    assert!(!net.gates.is_empty());
    for &(k, th) in &net.gates {
        assert!(
            th == FRAC_PI_2 || th == -FRAC_PI_2,
            "gate ({k}, {th}) is not an exact swap"
        );
    }
    let dets = dets_for(d, 2, 1);
    let c0 = coeffs(&x.to_dense(), &dets);
    let mut c: Vec<f64> = c0
        .iter()
        .zip(sign_phase(&net.signs, &dets))
        .map(|(a, p)| a * p)
        .collect();
    for &(k, th) in &net.gates {
        let w = mat_expm(&mat_scale(&dense_gpq(d, k + 1, k, &dets), th));
        c = mat_vec(&w, &c);
    }
    let (r, info) = x.rot(&u, &full());
    let got = coeffs(&r.to_dense(), &dets);
    let err = vdiff(&got, &c);
    let mut a: Vec<f64> = got.iter().map(|v| v.abs()).collect();
    let mut b: Vec<f64> = c0.iter().map(|v| v.abs()).collect();
    a.sort_by(f64::total_cmp);
    b.sort_by(f64::total_cmp);
    let perm_err = vdiff(&a, &b);
    let moved = vdiff(&got, &c0);
    println!(
        "trot_signed_permutation: {} gates, dense network {err:e}, |amplitude| multiset \
         {perm_err:e}, moved {moved:e}",
        info.gates
    );
    assert!(
        err <= 1e-12,
        "network order vs dense gate-by-gate oracle: {err:e}"
    );
    assert!(
        perm_err <= 1e-13,
        "amplitude multiset not preserved: {perm_err:e}"
    );
    assert!(moved > 1e-3, "vacuous permutation");
}

#[test]
fn trot_typed_space_refused() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(&dir, &opts).expect("lih_sto3g typed");
    assert!(
        !load.space.pg.is_trivial(),
        "lih_sto3g did not resolve a point group"
    );
    let d = load.space.d;
    let u = random_orthogonal(d, false, &mut seeded(0x7_F001));
    let us = [("mixing U", u.clone()), ("identity", identity_matrix(d))];

    let x = capped_randn(load.space, 1, 2, 0x7_F002);
    for (name, uu) in &us {
        let msg = panic_message(|| {
            let _ = apply_orbital_rotation(&x, uu, &full());
        });
        assert!(
            msg.contains("forget_point_group"),
            "abelian {name}: message does not name forget_point_group: {msg}"
        );
    }
    let xf = x.forget_point_group();
    let (r, info) = apply_orbital_rotation(&xf, &u, &full());
    check_full_rank_info("abelian forgotten", &xf, &u, &r, &info);

    let space = Su2Space::with_point_group(d, 4, 0, load.pg.pg, load.pg.target);
    let xs: Su2TtVector<f64> =
        qntt::su2::vector::tt_randn(space, &su2_capped_ranks(&space, 2), &mut seeded(0x7_F003));
    for (name, uu) in &us {
        let msg = panic_message(|| {
            let _ = su2_apply_orbital_rotation(&xs, uu, &full());
        });
        assert!(
            msg.contains("forget_point_group"),
            "su2 {name}: message does not name forget_point_group: {msg}"
        );
    }
}

#[test]
fn trot_non_orthogonal_refused() {
    let mut u = identity_matrix(6);
    u[0] = 1.0 + 1e-9;
    let xa = ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_F101);
    let msg = panic_message(|| {
        let _ = apply_orbital_rotation(&xa, &u, &full());
    });
    assert!(msg.contains("not orthogonal"), "abelian: {msg}");
    let xs = su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_F102);
    let msg = panic_message(|| {
        let _ = su2_apply_orbital_rotation(&xs, &u, &full());
    });
    assert!(msg.contains("not orthogonal"), "su2: {msg}");
}

#[test]
fn trot_threads_match_serial() {
    fn case<L: Layer>(label: &str, x: &L, cap: usize, seed: u64) {
        let u = random_orthogonal(x.sites(), true, &mut seeded(seed));
        for chi in [Pooled::NONE, Pooled(cap)] {
            let o = |threads| RotateOpts {
                eps: 0.0,
                chi,
                threads,
            };
            let (r1, i1) = x.rot(&u, &o(Some(1)));
            let m1 = r1.members();
            for threads in [Some(4), None] {
                let (rn, inn) = x.rot(&u, &o(threads));
                let mn = rn.members();
                let err = mdiff(&mn, &m1) / mmax(&m1);
                let bitwise = bits(&mn) == bits(&m1);
                println!(
                    "trot_threads {label} chi={} {threads:?} vs Some(1): {err:e}, \
                     bitwise = {bitwise}",
                    chi.0
                );
                assert!(rn.same_ranks(&r1), "{label} {threads:?}: rank grids differ");
                assert_eq!(inn.gates, i1.gates);
                assert_eq!(inn.max_rank, i1.max_rank, "{label} {threads:?}: max_rank");
                assert!(err <= 1e-14, "{label} chi={} {threads:?}: {err:e}", chi.0);
                assert_eq!(inn.discarded.len(), i1.discarded.len());
                for (g, (a, b)) in inn.discarded.iter().zip(&i1.discarded).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-14,
                        "{label}: discarded[{g}] {a:e} vs {b:e}"
                    );
                }
            }
        }
    }
    case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_F201),
        12,
        0x7_F202,
    );
    case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_F211),
        6,
        0x7_F212,
    );
}

#[test]
fn trot_untouched_bonds_keep_ranks() {
    fn case<L: Layer>(label: &str, x: &L) {
        let d = x.sites();
        let u = compose_rotations(&givens(d, 0, 0.7), &givens(d, 1, -0.4), d);
        let net = givens_decompose(&u, d);
        assert!(!net.gates.is_empty());
        for &(k, _) in &net.gates {
            assert!(k <= 1, "{label}: gate ({k}, {}) leaves the block", k + 1);
        }
        let (r, info) = x.rot(&u, &full());
        check_full_rank_info(label, x, &u, &r, &info);
        for b in [0, 3, 4, 5, 6].into_iter().filter(|&b| b <= d) {
            assert!(
                r.same_ranks_at(x, b),
                "{label}: untouched bond {b} changed rank grid"
            );
        }
    }
    let xa = ab_state(QnSpace::new(6, 2, 2), 2, 3, 0x7_F301);
    case("abelian unflagged", &xa);
    let mut xr = xa.clone();
    xr.move_core(5, true);
    case("abelian centred at 5", &xr);
    let xs = su2_state(Su2Space::new(6, 3, 1), 2, 3, 0x7_F311);
    case("su2 unflagged", &xs);
    let mut xsr = xs.clone();
    xsr.move_core(5, true);
    case("su2 centred at 5", &xsr);
}

#[test]
fn trot_left_centred_input() {
    fn case<L: Layer>(label: &str, x: &L, cap: usize, seed: u64) {
        let d = x.sites();
        let xc = x.centred(0);
        assert_eq!(xc.ortho_flag(), Some(0), "{label}: not centred at 0");
        let u = random_orthogonal(d, true, &mut seeded(seed));
        let k0 = givens_decompose(&u, d).gates[0].0;
        assert!(
            k0 > 1,
            "{label}: first gate {k0} does not need a right transport"
        );
        let (r, info) = xc.rot(&u, &full());
        check_full_rank_info(&format!("{label} centred 0"), &xc, &u, &r, &info);
        t8_case(&format!("{label} centred 0"), &xc, seed + 1);
        t10_case(&format!("{label} centred 0"), &xc, cap, 1e-2, seed + 2);
    }
    let xa = ab_state(QnSpace::new(6, 2, 2), 1, 1, 0x7_AA01).scaled(2.5);
    case("abelian (6,2,2)", &xa, 18, 0x7_C001);
    let xs = su2_state(Su2Space::new(6, 3, 1), 1, 1, 0x7_AB11).scaled(0.4);
    case("su2 (6,3,1)", &xs, 8, 0x7_C011);
}

#[test]
fn trot_max_rank_running_peak() {
    fn case<L: Layer>(label: &str, x: &L) {
        let d = x.sites();
        let k = random_antisym_flat(d, &mut seeded(0x7_E000));
        let u = expm_antisym(&scale_flat(&k, -1e-3), d);
        let opts = RotateOpts {
            eps: 1e-3,
            chi: Pooled::NONE,
            threads: None,
        };
        let (r, info, _, _) = t10_ledger(&format!("{label} near-identity"), x, &u, &opts);
        assert!(info.gates > 0, "{label}: no gates");
        println!(
            "trot_max_rank {label}: max_rank {}, input {}, result {}",
            info.max_rank,
            x.bond_max(),
            r.bond_max()
        );
        assert!(
            info.max_rank > x.bond_max().max(r.bond_max()),
            "{label}: max_rank {} is not the running peak (input {}, result {})",
            info.max_rank,
            x.bond_max(),
            r.bond_max()
        );
    }
    case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 1, 0x7_E101),
    );
    case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 1, 0x7_E111),
    );
}

#[test]
fn trot_bad_opts_and_zero_norm_refused() {
    fn case<L: Layer>(label: &str, x: &L) {
        let d = x.sites();
        let u = random_orthogonal(d, false, &mut seeded(0x7_F201));
        assert!(!givens_decompose(&u, d).gates.is_empty());
        for eps in [-1e-3, f64::NAN, f64::INFINITY] {
            let o = RotateOpts { eps, ..full() };
            let msg = panic_message(|| {
                let _ = x.rot(&u, &o);
            });
            assert!(msg.contains("eps"), "{label} eps={eps}: {msg}");
        }
        let o = RotateOpts {
            threads: Some(0),
            ..full()
        };
        let msg = panic_message(|| {
            let _ = x.rot(&u, &o);
        });
        assert!(msg.contains("Some(0)"), "{label} threads: {msg}");
        let z = x.scaled(0.0);
        let msg = panic_message(|| {
            let _ = z.rot(&u, &full());
        });
        assert!(msg.contains("input norm"), "{label} zero norm: {msg}");
    }
    case(
        "abelian (6,2,2)",
        &ab_state(QnSpace::new(6, 2, 2), 1, 2, 0x7_F202),
    );
    case(
        "su2 (6,3,1)",
        &su2_state(Su2Space::new(6, 3, 1), 1, 2, 0x7_F203),
    );
}

#[test]
fn trot_c64_dense_oracle() {
    type Mem = Vec<Vec<C64>>;
    type Rot<'a> = &'a dyn Fn(&[f64], &RotateOpts) -> (Mem, RotateInfo, f64);
    fn cdist(a: &[C64], b: &[C64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p.re - q.re).powi(2) + (p.im - q.im).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    fn case(
        label: &str,
        d: usize,
        xm: &Mem,
        sectors: &[(usize, usize)],
        nx: f64,
        cap: usize,
        rot: Rot<'_>,
    ) {
        let k = random_antisym_flat(d, &mut seeded(0x7_C640));
        let kc = flat_to_coeff(d, &k);
        let e = expm_antisym(&neg_flat(&k), d);
        let mut signs = vec![1.0f64; d];
        signs[1] = -1.0;
        let u = compose_rotations(&diag(&signs), &e, d);
        let (ym, info, ny) = rot(&u, &full());
        assert!(info.gates > 0);
        assert!(
            info.discarded.iter().all(|&w| w == 0.0),
            "{label}: full-rank discard"
        );
        assert_eq!(info.error_bound, 0.0, "{label}: full-rank bound");
        assert!((ny - nx).abs() <= 1e-12 * nx, "{label}: norm {ny} vs {nx}");
        for (i, &(nup, ndn)) in sectors.iter().enumerate() {
            let dets = dets_for(d, nup, ndn);
            let w = mat_expm(&dense_kappa_matrix(d, &kc, &dets));
            let phase = sign_phase(&signs, &dets);
            let part = |m: &[C64], im: bool| -> Vec<f64> {
                let v: Vec<f64> = m.iter().map(|z| if im { z.im } else { z.re }).collect();
                coeffs(&v, &dets)
            };
            let mut err = 0.0f64;
            let mut imag = 0.0f64;
            for im in [false, true] {
                let c: Vec<f64> = part(&xm[i], im)
                    .iter()
                    .zip(&phase)
                    .map(|(a, p)| a * p)
                    .collect();
                imag = imag.max(if im { vnorm(&c) } else { 0.0 });
                err = err.max(vdiff(&part(&ym[i], im), &mat_vec(&w, &c)));
            }
            println!(
                "trot_c64 {label} member {i} ({nup},{ndn}): max|Δ| = {err:e}, ‖Im x‖ = {imag:.3e}"
            );
            assert!(imag > 1e-3, "{label} member {i}: imaginary part is empty");
            assert!(
                err <= 1e-12 * nx,
                "{label} member {i}: dense mismatch {err:e}"
            );
        }
        let opts = RotateOpts {
            chi: Pooled(cap),
            ..full()
        };
        let (tm, info, nt) = rot(&u, &opts);
        let sum: f64 = info.discarded.iter().sum();
        assert!(
            info.discarded.iter().all(|&w| w >= -1e-14),
            "{label}: negative discard"
        );
        let ledger = (1.0 - info.norm_out * info.norm_out - sum).abs();
        println!(
            "trot_c64 {label} chi={cap}: Σ discarded {sum:e}, ledger {ledger:e}, error_bound {:e}",
            info.error_bound
        );
        assert!(sum > 1e-6, "{label}: the cap did not bind ({sum:e})");
        assert!(ledger <= 1e-12, "{label}: ledger {ledger:e}");
        assert!(
            (nt - info.norm_out * nx).abs() <= 1e-12 * nx,
            "{label}: ‖result‖"
        );
        for (i, (a, b)) in tm.iter().zip(&ym).enumerate() {
            let dist = cdist(a, b);
            assert!(
                dist <= info.error_bound * nx + 1e-12 * nx,
                "{label} member {i}: {dist:e} > bound {:e}",
                info.error_bound * nx
            );
        }
    }
    let space = QnSpace::new(6, 2, 2);
    let x: TtVector<C64> = randomized_state(space, 1, 1, &mut seeded(0x7_C641));
    let xm = vec![x.to_dense()];
    case(
        "abelian (6,2,2)",
        6,
        &xm,
        &[(2, 2)],
        x.norm(),
        18,
        &|u, o| {
            let (y, info) = apply_orbital_rotation(&x, u, o);
            (vec![y.to_dense()], info, y.norm())
        },
    );
    let space = Su2Space::new(6, 3, 1);
    let xs: Su2TtVector<C64> = randomized_su2_state(space, 1, 1, &mut seeded(0x7_C642));
    let tzs = su2_members(&space);
    let mem =
        |z: &Su2TtVector<C64>| -> Mem { tzs.iter().map(|&tz| expand(z, tz).to_dense()).collect() };
    let sectors: Vec<(usize, usize)> = tzs
        .iter()
        .map(|&tz| (((3 + tz) / 2) as usize, ((3 - tz) / 2) as usize))
        .collect();
    case(
        "su2 (6,3,1)",
        6,
        &mem(&xs),
        &sectors,
        xs.norm(),
        8,
        &|u, o| {
            let (y, info) = su2_apply_orbital_rotation(&xs, u, o);
            (mem(&y), info, y.norm())
        },
    );
}

fn load_typed(name: &str) -> IntegralsLoad<f64> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("testdata")
        .join(name);
    let opts = IntegralsOptions {
        point_group: PointGroupPolicy::Resolve(None),
        ..IntegralsOptions::default()
    };
    let load = load_fixture_dir::<f64>(&dir, &opts).expect(name);
    assert!(
        !load.space.pg.is_trivial(),
        "{name}: did not resolve a point group"
    );
    load
}

fn su2_has_multi_irrep_sector(x: &Su2TtVector<f64>) -> bool {
    let sp = x.space;
    let untyped = Su2Space::new(sp.d, sp.n, sp.two_s);
    (0..=sp.d).any(|b| {
        untyped.bond_sectors(b).into_iter().any(|s| {
            s.two_s >= 1
                && (0..sp.pg.order())
                    .filter(|&ir| {
                        x.ranks[b][Su2Sector::with_irrep(s.n, s.two_s, Irrep::new(ir))] > 0
                    })
                    .count()
                    >= 2
        })
    })
}

fn ab_has_multi_irrep_sector(x: &TtVector<f64>) -> bool {
    let sp = x.space;
    let untyped = QnSpace::new(sp.d, sp.nup, sp.ndn);
    (0..=sp.d).any(|b| {
        untyped.bond_sectors(b).into_iter().any(|s| {
            (0..sp.pg.order())
                .filter(|&ir| x.ranks[b][Sector::with_irrep(s.nup, s.ndn, Irrep::new(ir))] > 0)
                .count()
                >= 2
        })
    })
}

fn irrep_mixing(u: &[f64], d: usize, pg: PointGroup) -> f64 {
    let mut m = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                m = m.max(u[p * d + q].abs());
            }
        }
    }
    m
}

fn t11_bridge(label: &str, pairs: [(f64, f64); 2], norms: (f64, f64)) {
    for (what, (reference, got)) in ["⟨x,Hy⟩", "⟨x,Hx⟩"].into_iter().zip(pairs) {
        let scale = reference.abs().max(1.0);
        let err = (got - reference).abs();
        println!("trot_t11 {label} {what}: typed {reference:.12}, embedded |Δ| = {err:e}");
        assert!(
            reference.abs() > 1e-6,
            "{label} {what}: anti-vacuity: the typed matrix element is {reference:e}"
        );
        assert!(
            err <= 1e-12 * scale,
            "{label} {what}: embedded {got} vs typed {reference} (|Δ| = {err:e})"
        );
    }
    let (nx, nf) = norms;
    assert!(
        (nf - nx).abs() <= 1e-12 * nx,
        "{label}: ‖forget(x)‖ = {nf} vs ‖x‖ = {nx}"
    );
}

#[test]
fn trot_t11_forget_then_rotate_su2() {
    for (label, name, seed) in [
        ("h2o_sto3g", "h2o_sto3g", 0x7_B101u64),
        ("ch2_triplet_c2v", "ch2_triplet_c2v", 0x7_B111),
    ] {
        let load = load_typed(name);
        let d = load.space.d;
        let (pg, target) = (load.pg.pg, load.pg.target);
        let n = load.space.nup + load.space.ndn;
        let two_s = load.space.nup - load.space.ndn;
        let label = format!("su2 {label} (N, 2S) = ({n}, {two_s})");
        let space = Su2Space::with_point_group(d, n, two_s, pg, target);
        assert!(
            su2_max_ranks(&space)[d][Su2Sector::with_irrep(n, two_s, target)] > 0,
            "{label}: the target irrep {target:?} is unreachable"
        );
        let ranks = su2_capped_ranks(&space, 2);
        let x: Su2TtVector<f64> = qntt::su2::vector::tt_randn(space, &ranks, &mut seeded(seed));
        let y: Su2TtVector<f64> = qntt::su2::vector::tt_randn(space, &ranks, &mut seeded(seed + 1));
        assert!(
            su2_has_multi_irrep_sector(&x),
            "{label}: anti-vacuity: no (n, 2s ≥ 1) bond sector carries two irrep blocks"
        );
        let u0 = random_orthogonal(d, false, &mut seeded(seed + 200));
        let mix = irrep_mixing(&u0, d, pg);
        assert!(
            mix > 1e-2,
            "{label}: anti-vacuity: U does not mix irreps ({mix:e})"
        );

        let untyped = Su2Space::new(d, n, two_s);
        let xf = x.forget_point_group();
        let yf = y.forget_point_group();
        for (what, a, af) in [("x", &x, &xf), ("y", &y, &yf)] {
            assert_eq!(af.space, untyped, "{label} {what}: wrong output space");
            assert!(
                af.check(),
                "{label} {what}: the embedding is not a legal train"
            );
            for b in 0..=d {
                assert_eq!(
                    af.ranks[b].total(),
                    a.ranks[b].total(),
                    "{label} {what}, bond {b}: pooled rank total"
                );
            }
        }

        let ht = Su2SparseHamiltonian::new(space, &load.ints.t, &load.ints.v, None, f64::EPSILON);
        let hu = Su2SparseHamiltonian::new(untyped, &load.ints.t, &load.ints.v, None, f64::EPSILON);
        t11_bridge(
            &label,
            [
                (su2_xt_h_y(&x, &ht, &y), su2_xt_h_y(&xf, &hu, &yf)),
                (su2_xt_h_y(&x, &ht, &x), su2_xt_h_y(&xf, &hu, &xf)),
            ],
            (x.norm(), xf.norm()),
        );
        t5_case(
            &format!("{label} forgotten"),
            &load.ints,
            &xf,
            &yf,
            seed + 200,
        );
    }
}

#[test]
fn trot_t11_forget_then_rotate_abelian() {
    for (label, name, seed) in [
        ("h2o_sto3g", "h2o_sto3g", 0x7_B201u64),
        ("ch2_triplet_c2v", "ch2_triplet_c2v", 0x7_B211),
    ] {
        let load = load_typed(name);
        let space = load.space;
        let d = space.d;
        let label = format!(
            "abelian {label} (Nup, Ndn) = ({}, {})",
            space.nup, space.ndn
        );
        let x = capped_randn(space, 1, 2, seed);
        let y = capped_randn(space, 1, 2, seed + 1);
        assert!(
            ab_has_multi_irrep_sector(&x),
            "{label}: anti-vacuity: no bond sector carries two irrep blocks"
        );
        let u0 = random_orthogonal(d, false, &mut seeded(seed + 200));
        let mix = irrep_mixing(&u0, d, space.pg);
        assert!(
            mix > 1e-2,
            "{label}: anti-vacuity: U does not mix irreps ({mix:e})"
        );

        let untyped = QnSpace::new(d, space.nup, space.ndn);
        let xf = x.forget_point_group();
        let yf = y.forget_point_group();
        for (what, a, af) in [("x", &x, &xf), ("y", &y, &yf)] {
            assert_eq!(af.space, untyped, "{label} {what}: wrong output space");
            assert!(
                af.check(),
                "{label} {what}: the embedding is not a legal train"
            );
            for b in 0..=d {
                assert_eq!(
                    af.ranks[b].total(),
                    a.ranks[b].total(),
                    "{label} {what}, bond {b}: pooled rank total"
                );
            }
        }

        let ht = SparseHamiltonian::new(space, &load.ints.t, &load.ints.v, None, f64::EPSILON);
        let hu = SparseHamiltonian::new(untyped, &load.ints.t, &load.ints.v, None, f64::EPSILON);
        t11_bridge(
            &label,
            [
                (xt_h_y(&x, &ht, &y), xt_h_y(&xf, &hu, &yf)),
                (xt_h_y(&x, &ht, &x), xt_h_y(&xf, &hu, &xf)),
            ],
            (x.norm(), xf.norm()),
        );
        t5_case(
            &format!("{label} forgotten"),
            &load.ints,
            &xf,
            &yf,
            seed + 200,
        );
    }
}

fn characters_of(w: &[f64]) -> Vec<f64> {
    (0..w.len())
        .map(|h| {
            w.iter()
                .enumerate()
                .map(|(g, &wg)| Irrep::new(g as u8).character(Irrep::new(h as u8)) * wg)
                .sum()
        })
        .collect()
}

fn pure_readout(target: Irrep, order: u8) -> (Vec<f64>, Vec<f64>) {
    let chars = (0..order)
        .map(|h| target.character(Irrep::new(h)))
        .collect();
    let weights = (0..order)
        .map(|g| if g == target.index() { 1.0 } else { 0.0 })
        .collect();
    (chars, weights)
}

fn t12_case<L: Layer>(label: &str, xf: &L, target: Irrep, pg: PointGroup, cap: usize, seed: u64) {
    let d = xf.sites();
    let order = pg.order();
    let (want_c, want_w) = pure_readout(target, order);
    let u = random_orthogonal(d, false, &mut seeded(seed));
    let mix = irrep_mixing(&u, d, pg);
    assert!(
        mix > 1e-2,
        "{label}: anti-vacuity: U does not mix irreps ({mix:e})"
    );
    let ut = transpose_flat(&u, d);

    let (x_loc, info_f) = xf.rot(&u, &full());
    assert_eq!(
        info_f.error_bound, 0.0,
        "{label}: forward rotation truncated"
    );
    let wl = x_loc.irreps(pg);
    let wmax = wl.iter().cloned().fold(0.0f64, f64::max);
    assert!(
        wmax < 1.0 - 1e-3,
        "{label}: anti-vacuity: the localized state reads as pure ({wl:?})"
    );

    let (y, info_b) = x_loc.rot(&ut, &full());
    assert_eq!(info_b.error_bound, 0.0, "{label}: back rotation truncated");
    let w = y.irreps(pg);
    let chars = characters_of(&w);
    let (ec, ew) = (vdiff(&chars, &want_c), vdiff(&w, &want_w));
    let sum: f64 = w.iter().sum();
    assert!(
        ec <= 1e-10 && ew <= 1e-10,
        "{label}: rotated-back readout {w:?} / {chars:?} is not target {}",
        target.index()
    );
    assert!((sum - 1.0).abs() <= 1e-12, "{label}: weights sum to {sum}");

    let xm = &x_loc.members()[0];
    let n2 = xm.iter().map(|v| v * v).sum::<f64>();
    let mut ecomp = 0.0f64;
    for h in 0..order {
        let dh: Vec<f64> = (0..d)
            .map(|j| pg.irrep_of(j).character(Irrep::new(h)))
            .collect();
        let rh = compose_rotations(&compose_rotations(&ut, &diag(&dh), d), &u, d);
        let (z, _) = x_loc.rot(&rh, &full());
        let zm = &z.members()[0];
        let c = xm.iter().zip(zm).map(|(a, b)| a * b).sum::<f64>() / n2;
        ecomp = ecomp.max((c - chars[h as usize]).abs());
    }
    assert!(
        ecomp <= 1e-10,
        "{label}: characters differ from ⟨x_loc, Û(R_h) x_loc⟩ by {ecomp:e}"
    );

    let opts = RotateOpts {
        eps: 0.0,
        chi: Pooled(cap),
        threads: None,
    };
    let (yt, info_t) = x_loc.rot(&ut, &opts);
    let disc: f64 = info_t.discarded.iter().sum();
    let eb = info_t.error_bound;
    assert!(
        disc > 1e-6,
        "{label} chi={cap}: the cap did not bind (Σ discarded {disc:e})"
    );
    assert!(
        4.0 * eb < 1.0,
        "{label} chi={cap}: the bound is not informative (error_bound {eb:e})"
    );
    let wt = yt.irreps(pg);
    let (dw, dc) = (vdiff(&wt, &w), vdiff(&characters_of(&wt), &chars));
    println!(
        "trot_t12 {label}: mixing {mix:.3}, localized max weight {wmax:.4}; back |Δw| = {ew:e}, \
         |Δχ| = {ec:e}, composition |Δ| = {ecomp:e}; chi={cap}: Σ discarded {disc:.2e}, \
         error_bound {eb:.3e}, |Δw| = {dw:.2e} (≤ 2ε = {:.2e}), |Δχ| = {dc:.2e} (≤ 4ε = {:.2e})",
        2.0 * eb,
        4.0 * eb
    );
    assert!(
        dw <= 2.0 * eb + 1e-12,
        "{label} chi={cap}: weights moved by {dw:e} > 2·error_bound = {:e}",
        2.0 * eb
    );
    assert!(
        dc <= 4.0 * eb + 1e-12,
        "{label} chi={cap}: characters moved by {dc:e} > 4·error_bound = {:e}",
        4.0 * eb
    );
}

#[test]
fn trot_t12_irrep_readout_abelian() {
    let load = load_typed("h2o_sto3g");
    let sp = load.space;
    let (d, pg) = (sp.d, load.pg.pg);
    let mut exercised = 0;
    for g in 0..pg.order() {
        let target = Irrep::new(g);
        let space = QnSpace::with_point_group(d, sp.nup, sp.ndn, pg, target);
        if qntt::qn::max_ranks(&space)[d][Sector::with_irrep(sp.nup, sp.ndn, target)] == 0 {
            continue;
        }
        exercised += 1;
        let xf = capped_randn(space, 1, 2, 0x7_C001 + u64::from(g)).forget_point_group();
        t12_case(
            &format!("abelian h2o target {g}"),
            &xf,
            target,
            pg,
            28,
            0x7_C101 + u64::from(g),
        );
    }
    assert!(exercised >= 3, "only {exercised} target irreps reachable");
}

#[test]
fn trot_t12_irrep_readout_su2() {
    let load = load_typed("h2o_sto3g");
    let sp = load.space;
    let (d, pg) = (sp.d, load.pg.pg);
    let (n, two_s) = (sp.nup + sp.ndn, sp.nup - sp.ndn);
    let mut exercised = 0;
    for g in 0..pg.order() {
        let target = Irrep::new(g);
        let space = Su2Space::with_point_group(d, n, two_s, pg, target);
        if su2_max_ranks(&space)[d][Su2Sector::with_irrep(n, two_s, target)] == 0 {
            continue;
        }
        exercised += 1;
        let x: Su2TtVector<f64> = qntt::su2::vector::tt_randn(
            space,
            &su2_capped_ranks(&space, 2),
            &mut seeded(0x7_C201 + u64::from(g)),
        );
        t12_case(
            &format!("su2 h2o target {g}"),
            &x.forget_point_group(),
            target,
            pg,
            16,
            0x7_C101 + u64::from(g),
        );
    }
    assert!(exercised >= 3, "only {exercised} target irreps reachable");
}
