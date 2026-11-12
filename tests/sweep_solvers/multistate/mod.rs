// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use faer::Mat;
use qntt::coeff::CoeffTensor;
use qntt::core::factor::dense_self_adjoint_eigen;
use qntt::core::twosite::{ContractedSparseCores, contract};
use qntt::hamiltonian::SparseHamiltonian;
use qntt::qn::QnSpace;
use qntt::solvers::ssmals::{SsHistory, penalty_weight};
use qntt::solvers::{TtHamiltonian, TtState};
use qntt::su2::core::{Family as Su2Family, Su2Core};
use qntt::su2::hamiltonian::Su2SparseHamiltonian;
use qntt::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use qntt::su2::twosite::{Su2ContractedCores, contract as su2_contract};
use qntt::su2::vector::Su2TtVector;
use qntt::{Scalar, TtVector};
use qntt_testkit::dense::{basis_index, dense_eigen, dense_hamiltonian_matrix, dets_for};
use qntt_testkit::gad_mol::singlet_indices;
use rand_xoshiro::Xoshiro256PlusPlus;

pub(crate) use qntt_testkit::seeded;

mod excited;
mod overlap;
mod sa;
mod ss;
mod su2_excited;
mod su2_overlap;
mod su2_sa;
mod su2_ss;

fn sparse_sym_v(d: usize, rng: &mut Xoshiro256PlusPlus) -> CoeffTensor<f64> {
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    for p in 0..d {
        for q in (p + 1)..d {
            let dd = f64::sample_standard(rng);
            v.set(&[p, p, q, q], dd);
            v.set(&[q, q, p, p], dd);
            let ex = f64::sample_standard(rng);
            v.set(&[p, q, q, p], ex);
            v.set(&[q, p, p, q], ex);
        }
    }
    v
}

fn sym_t(d: usize, rng: &mut Xoshiro256PlusPlus) -> CoeffTensor<f64> {
    let mut a = vec![vec![0.0f64; d]; d];
    for row in a.iter_mut() {
        for x in row.iter_mut() {
            *x = f64::sample_standard(rng);
        }
    }
    let mut t = CoeffTensor::<f64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], 0.5 * (a[i][j] + a[j][i]));
        }
    }
    t
}

pub(crate) struct GappedFixture {
    #[allow(dead_code)]
    pub seed: u64,
    pub t: CoeffTensor<f64>,
    pub v: CoeffTensor<f64>,
    pub dets11: Vec<(Vec<bool>, Vec<bool>)>,
    pub ev11: Vec<f64>,
    pub evec11: Vec<Vec<f64>>,
    pub dets20: Vec<(Vec<bool>, Vec<bool>)>,
    pub ev20: Vec<f64>,
    pub evec20: Vec<Vec<f64>>,
    pub singlets: Vec<usize>,
}

impl GappedFixture {
    pub fn singlet_levels(&self) -> Vec<f64> {
        self.singlets.iter().map(|&i| self.ev11[i]).collect()
    }

    pub fn abelian(&self, space: QnSpace) -> SparseHamiltonian<f64> {
        SparseHamiltonian::new(space, &self.t, &self.v, None, f64::EPSILON)
    }

    pub fn su2(&self, space: Su2Space) -> Su2SparseHamiltonian<f64> {
        Su2SparseHamiltonian::new(space, &self.t, &self.v, None, 1e-12)
    }
}

fn gaps_ok(ev: &[f64], nroots: usize, floor: f64) -> bool {
    if ev.len() <= nroots {
        return false;
    }
    (0..nroots).all(|i| ev[i + 1] - ev[i] >= floor)
}

pub(crate) fn dense_sector(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    nup: usize,
    ndn: usize,
) -> (Vec<(Vec<bool>, Vec<bool>)>, Vec<f64>, Vec<Vec<f64>>) {
    let dets = dets_for(d, nup, ndn);
    let mat = dense_hamiltonian_matrix(d, t, v, &dets);
    let (ev, evecs) = dense_eigen(&mat);
    (dets, ev, evecs)
}

pub(crate) fn gapped_tv(seed0: u64, nroots: usize, floor: f64) -> GappedFixture {
    let d = 6usize;
    for attempt in 0..64u64 {
        let seed = seed0 + attempt;
        let mut rng = seeded(seed);
        let t = sym_t(d, &mut rng);
        let v = sparse_sym_v(d, &mut rng);
        let (dets11, ev11, evec11) = dense_sector(d, &t, &v, 1, 1);
        let (dets20, ev20, evec20) = dense_sector(d, &t, &v, 2, 0);
        let singlets = singlet_indices(&ev11, &ev20);
        if singlets.len() != ev11.len() - ev20.len() {
            continue;
        }
        let slevels: Vec<f64> = singlets.iter().map(|&i| ev11[i]).collect();
        if gaps_ok(&ev11, nroots, floor)
            && gaps_ok(&slevels, nroots, floor)
            && gaps_ok(&ev20, nroots, floor)
        {
            println!(
                "gapped_tv: seed {seed:#x} (attempt {attempt}), nroots {nroots}, floor {floor}\n  \
                 ev11[..={nroots}]     = {:?}\n  singlets ({}) [..={nroots}] = {:?}\n  \
                 ev20[..={nroots}]     = {:?}",
                &ev11[..=nroots],
                singlets.len(),
                &slevels[..=nroots],
                &ev20[..=nroots],
            );
            return GappedFixture {
                seed,
                t,
                v,
                dets11,
                ev11,
                evec11,
                dets20,
                ev20,
                evec20,
                singlets,
            };
        }
    }
    panic!(
        "gapped_tv: no generic (t,v) with {nroots} resolved roots at gap floor {floor} in 64 \
         seeds from {seed0:#x} — this is a FIXTURE finding; change the generator, never the floor"
    );
}

pub(crate) fn degenerate_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 4, 1], -1.0);
    v.set(&[1, 4, 0, 2], -1.0);
    (t, v)
}

pub(crate) fn gauge_integrals(
    d: usize,
    t: &CoeffTensor<f64>,
    v: &CoeffTensor<f64>,
    theta: &[f64],
) -> (CoeffTensor<qntt::C64>, CoeffTensor<qntt::C64>) {
    use qntt::C64;
    let ph = |a: f64| C64::new(a.cos(), a.sin());
    let mut tc = CoeffTensor::<C64>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            let c = t.get(&[i, j]);
            if c != 0.0 {
                tc.set(&[i, j], ph(theta[i] - theta[j]) * C64::new(c, 0.0));
            }
        }
    }
    let mut vc = CoeffTensor::<C64>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    let c = v.get(&[i, j, k, l]);
                    if c != 0.0 {
                        vc.set(
                            &[i, j, k, l],
                            ph(theta[i] - theta[j] + theta[k] - theta[l]) * C64::new(c, 0.0),
                        );
                    }
                }
            }
        }
    }
    (tc, vc)
}

pub(crate) fn rho_abelian(x: &TtVector<qntt::C64>) -> f64 {
    let mut xb = x.clone();
    for k in 0..x.space.d {
        let n = xb.cores[k].num_coeffs();
        for i in 0..n {
            let c = xb.cores[k].coeff(i);
            xb.cores[k].set_coeff(i, c.conj_s());
        }
    }
    qntt::dot(&xb, x).abs_s() / qntt::dot(x, x).re_s().max(1e-300)
}

pub(crate) fn global_eigen_residual_abelian<T: Scalar>(
    h: &SparseHamiltonian<T>,
    x: &TtVector<T>,
    lambda: f64,
) -> f64 {
    let hx = h * x;
    let r = &hx - &(x * T::from_f64(lambda));
    r.norm_lr(false) / x.norm_lr(false).max(1e-30)
}

pub(crate) fn global_eigen_residual_su2<T: Scalar>(
    h: &Su2SparseHamiltonian<T>,
    x: &Su2TtVector<T>,
    lambda: f64,
) -> f64 {
    let hx = TtHamiltonian::matvec(h, x);
    let r = TtState::sub(&hx, &TtState::scaled(x, T::from_f64(lambda)));
    TtState::norm(&r) / TtState::norm(x).max(1e-30)
}

pub(crate) fn gram_dev_tt<T: Scalar>(xs: &[TtVector<T>]) -> f64 {
    let mut dev = 0.0f64;
    for (i, xi) in xs.iter().enumerate() {
        for (j, xj) in xs.iter().enumerate() {
            let want = if i == j { T::one() } else { T::zero() };
            dev = dev.max((qntt::dot(xi, xj) - want).abs_s());
        }
    }
    dev
}

pub(crate) fn gram_dev_su2<T: Scalar>(xs: &[Su2TtVector<T>]) -> f64 {
    let mut dev = 0.0f64;
    for (i, xi) in xs.iter().enumerate() {
        for (j, xj) in xs.iter().enumerate() {
            let want = if i == j { T::one() } else { T::zero() };
            dev = dev.max((qntt::su2::ops::dot(xi, xj) - want).abs_s());
        }
    }
    dev
}

pub(crate) fn projection_on_dense<T: Scalar>(
    x: &TtVector<T>,
    dets: &[(Vec<bool>, Vec<bool>)],
    evec: &[f64],
) -> f64 {
    let xd = x.to_dense();
    let mut acc = T::zero();
    let mut ne = 0.0f64;
    for (pos, (ou, od)) in dets.iter().enumerate() {
        acc += T::from_f64(evec[pos]) * xd[basis_index(ou, od)];
        ne += evec[pos] * evec[pos];
    }
    let nx: f64 = xd.iter().map(|c| c.abs_s().powi(2)).sum::<f64>().sqrt();
    acc.abs_s() / (nx * ne.sqrt()).max(1e-300)
}

pub(crate) fn bits_eq<T: Scalar>(a: T, b: T) -> bool {
    a.re_s().to_bits() == b.re_s().to_bits() && a.im_s().to_bits() == b.im_s().to_bits()
}

pub(crate) fn assert_f64_bits(got: f64, want: f64, what: &str) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{what}: {got:.17e} vs {want:.17e} (bit-identity required — the fix is in the driver's \
         operation order, never in this test)"
    );
}

pub(crate) fn coeffs_bits_eq_tt<T: Scalar>(a: &TtVector<T>, b: &TtVector<T>, what: &str) {
    assert_eq!(a.space, b.space, "{what}: space");
    assert_eq!(a.ranks.len(), b.ranks.len(), "{what}: rank grid count");
    for (k, (ra, rb)) in a.ranks.iter().zip(b.ranks.iter()).enumerate() {
        assert_eq!(ra, rb, "{what}: rank grid at bond {k}");
    }
    assert_eq!(a.ortho, b.ortho, "{what}: ortho flag");
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        assert_eq!(
            ca.num_coeffs(),
            cb.num_coeffs(),
            "{what}: num_coeffs core {k}"
        );
        for i in 0..ca.num_coeffs() {
            assert!(
                bits_eq(ca.coeff(i), cb.coeff(i)),
                "{what}: core {k} coefficient {i}: {:?} vs {:?}",
                ca.coeff(i),
                cb.coeff(i)
            );
        }
    }
}

pub(crate) fn coeffs_bits_eq_su2<T: Scalar>(a: &Su2TtVector<T>, b: &Su2TtVector<T>, what: &str) {
    assert_eq!(a.space, b.space, "{what}: space");
    assert_eq!(a.ranks.len(), b.ranks.len(), "{what}: rank grid count");
    for (k, (ra, rb)) in a.ranks.iter().zip(b.ranks.iter()).enumerate() {
        assert_eq!(ra, rb, "{what}: rank grid at bond {k}");
    }
    assert_eq!(a.ortho, b.ortho, "{what}: ortho flag");
    for k in 0..a.space.d {
        let (ca, cb) = (a.core(k), b.core(k));
        assert_eq!(
            ca.num_coeffs(),
            cb.num_coeffs(),
            "{what}: num_coeffs core {k}"
        );
        for i in 0..ca.num_coeffs() {
            assert!(
                bits_eq(ca.coeff(i), cb.coeff(i)),
                "{what}: core {k} coefficient {i}"
            );
        }
    }
}

pub(crate) fn two_site_frame_abelian<T: Scalar>(
    space: QnSpace,
    k: usize,
    seed: u64,
) -> (TtVector<T>, ContractedSparseCores<T>) {
    let mut rng = seeded(seed);
    let mut x: TtVector<T> = qntt::tt_randn(space, &qntt::max_ranks(&space), &mut rng);
    x.right_orthogonalize(false);
    x.move_core(k, false);
    assert!(x.check());
    let cp0 = contract(x.core(k), x.core(k + 1));
    (x, cp0)
}

pub(crate) fn two_site_frame_su2<T: Scalar>(
    space: Su2Space,
    k: usize,
    seed: u64,
) -> (Su2TtVector<T>, Su2ContractedCores<T>) {
    let mut rng = seeded(seed);
    let mut x: Su2TtVector<T> =
        qntt::su2::vector::tt_randn(space, &qntt::su2::qn::max_ranks(&space), &mut rng);
    x.right_orthogonalize(false);
    x.move_core(k, false);
    assert!(x.check());
    let cp0 = su2_contract(x.core(k), x.core(k + 1));
    (x, cp0)
}

pub(crate) fn unit_blocks_abelian<T: Scalar>(
    template: &ContractedSparseCores<T>,
    n: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<ContractedSparseCores<T>> {
    (0..n)
        .map(|i| {
            let mut b = template.clone();
            if i > 0 {
                b.for_each_coeff_mut(|c| *c = T::sample_standard(rng));
            }
            let nb = b.norm();
            assert!(nb > 0.0, "unit_blocks: state {i} has zero norm");
            b.scale_mut(T::from_f64(1.0 / nb));
            b
        })
        .collect()
}

pub(crate) fn unit_blocks_su2<T: Scalar>(
    template: &Su2ContractedCores<T>,
    n: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Vec<Su2ContractedCores<T>> {
    (0..n)
        .map(|i| {
            let mut b = template.clone();
            if i > 0 {
                b.for_each_coeff_mut(|c| *c = T::sample_standard(rng));
            }
            let nb = b.norm();
            assert!(nb > 0.0, "unit_blocks: state {i} has zero norm");
            b.scale_mut(T::from_f64(1.0 / nb));
            b
        })
        .collect()
}

fn rho_of_blocks<T: Scalar>(blocks: &[Mat<T>], w: &[f64], forward: bool) -> Mat<T> {
    let b0 = &blocks[0];
    let g = if forward { b0.nrows() } else { b0.ncols() };
    let mut rho = Mat::<T>::zeros(g, g);
    for (b, &wi) in blocks.iter().zip(w.iter()) {
        let wt = T::from_f64(wi);
        for i in 0..g {
            for j in 0..g {
                let mut acc = T::zero();
                if forward {
                    for c in 0..b.ncols() {
                        acc += *b.get(i, c) * b.get(j, c).conj_s();
                    }
                } else {
                    for r in 0..b.nrows() {
                        acc += b.get(r, i).conj_s() * *b.get(r, j);
                    }
                }
                *rho.get_mut(i, j) += acc * wt;
            }
        }
    }
    rho
}

pub(crate) fn pooled_rho_abelian<T: Scalar>(
    cps: &[ContractedSparseCores<T>],
    w: &[f64],
    forward: bool,
) -> Vec<(qntt::qn::Sector, Vec<f64>, Mat<T>)> {
    let mut out = Vec::new();
    for &m in cps[0].mid_qn.iter() {
        let blocks: Vec<Mat<T>> = cps
            .iter()
            .filter_map(|c| c.block(m).map(|b| b.to_owned()))
            .collect();
        if blocks.len() != cps.len() {
            continue;
        }
        let g = if forward {
            blocks[0].nrows()
        } else {
            blocks[0].ncols()
        };
        if g == 0 {
            continue;
        }
        let rho = rho_of_blocks(&blocks, w, forward);
        let (evals, u) = dense_self_adjoint_eigen(rho.as_ref());
        out.push((m, evals, u));
    }
    out
}

pub(crate) fn pooled_rho_su2<T: Scalar>(
    cps: &[Su2ContractedCores<T>],
    w: &[f64],
    forward: bool,
) -> Vec<(Su2Sector, Vec<f64>, Mat<T>)> {
    let mut out = Vec::new();
    for &m in cps[0].mid_qn.iter() {
        let blocks: Vec<Mat<T>> = cps.iter().filter_map(|c| b_theta(c, m)).collect();
        if blocks.len() != cps.len() {
            continue;
        }
        let g = if forward {
            blocks[0].nrows()
        } else {
            blocks[0].ncols()
        };
        if g == 0 {
            continue;
        }
        let rho = rho_of_blocks(&blocks, w, forward);
        let (evals, u) = dense_self_adjoint_eigen(rho.as_ref());
        let cw = qntt::su2::centerweight(m.two_s, cps[0].space.two_s);
        out.push((m, evals.into_iter().map(|l| cw * l).collect(), u));
    }
    out
}

pub(crate) fn b_theta<T: Scalar>(tc: &Su2ContractedCores<T>, m: Su2Sector) -> Option<Mat<T>> {
    let shell = Su2Core::<T>::zeros(
        tc.space,
        tc.k + 1,
        &Su2RankGrid::zeros(&tc.space),
        &tc.col_ranks,
    );
    let b = tc.block(m)?;
    let mut sc = b.to_owned();
    shell.scale_horizontal(&mut sc, m);
    Some(sc)
}

pub(crate) fn frob_sq<T: Scalar>(a: faer::MatRef<'_, T>) -> f64 {
    let mut acc = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            acc += a.get(i, j).abs_s().powi(2);
        }
    }
    acc
}

pub(crate) fn random_orthonormal<T: Scalar>(
    g: usize,
    r: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Mat<T> {
    let mut cols: Vec<Vec<T>> = (0..r)
        .map(|_| (0..g).map(|_| T::sample_standard(rng)).collect())
        .collect();
    for j in 0..r {
        for p in 0..j {
            let mut ip = T::zero();
            for i in 0..g {
                ip += cols[p][i].conj_s() * cols[j][i];
            }
            for i in 0..g {
                let sub = ip * cols[p][i];
                cols[j][i] -= sub;
            }
        }
        let nrm: f64 = cols[j]
            .iter()
            .map(|c| c.abs_s().powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            nrm > 1e-8,
            "random_orthonormal: degenerate draw at column {j}"
        );
        let inv = T::from_f64(1.0 / nrm);
        for i in 0..g {
            cols[j][i] *= inv;
        }
    }
    let mut q = Mat::<T>::zeros(g, r);
    for j in 0..r {
        for i in 0..g {
            *q.get_mut(i, j) = cols[j][i];
        }
    }
    q
}

pub(crate) fn kept_weight<T: Scalar>(
    q: faer::MatRef<'_, T>,
    b: faer::MatRef<'_, T>,
    forward: bool,
) -> f64 {
    let r = q.ncols();
    let mut acc = 0.0f64;
    if forward {
        for c in 0..b.ncols() {
            for p in 0..r {
                let mut s = T::zero();
                for i in 0..b.nrows() {
                    s += q.get(i, p).conj_s() * *b.get(i, c);
                }
                acc += s.abs_s().powi(2);
            }
        }
    } else {
        for row in 0..b.nrows() {
            for p in 0..r {
                let mut s = T::zero();
                for c in 0..b.ncols() {
                    s += *b.get(row, c) * *q.get(c, p);
                }
                acc += s.abs_s().powi(2);
            }
        }
    }
    acc
}

fn accumulate_weighted_bbh<T: Scalar>(g: &mut Mat<T>, b: faer::MatRef<'_, T>, weight: f64) {
    let w = T::from_f64(weight);
    for i in 0..b.nrows() {
        for j in 0..b.nrows() {
            let mut acc = T::zero();
            for c in 0..b.ncols() {
                acc += *b.get(i, c) * b.get(j, c).conj_s();
            }
            *g.get_mut(i, j) += acc * w;
        }
    }
}

fn max_dev_identity<T: Scalar>(g: faer::MatRef<'_, T>) -> f64 {
    let n = g.nrows();
    let mut worst = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { T::one() } else { T::zero() };
            worst = worst.max((*g.get(i, j) - target).abs_s());
        }
    }
    worst
}

pub(crate) fn su2_right_canonicity_dev<T: Scalar>(
    x: &Su2TtVector<T>,
    range: std::ops::Range<usize>,
) -> f64 {
    let mut maxdev = 0.0f64;
    for k in range {
        let core = x.core(k);
        for l in core.row_qn.iter().copied() {
            let r = core.row_ranks[l];
            if r == 0 {
                continue;
            }
            let mut g = Mat::<T>::zeros(r, r);
            let weights: [(Su2Family, f64); 4] = [
                (Su2Family::Empty, 1.0),
                (
                    Su2Family::Up,
                    (l.two_s as f64 + 2.0) / (l.two_s as f64 + 1.0),
                ),
                (Su2Family::Dn, l.two_s as f64 / (l.two_s as f64 + 1.0)),
                (Su2Family::Occ, 1.0),
            ];
            for (f, w) in weights {
                if let Some(b) = core.block(f, l) {
                    accumulate_weighted_bbh(&mut g, b, w);
                }
            }
            maxdev = maxdev.max(max_dev_identity(g.as_ref()));
        }
    }
    maxdev
}

pub(crate) fn count_theta_cw<T: Scalar>(x: &Su2TtVector<T>) -> (usize, usize) {
    let space = x.space;
    let d = space.d;
    let mut n_theta = 0usize;
    let mut n_cw = 0usize;
    for b in 0..d {
        for m in space.bond_sectors(b) {
            if x.rank(b, m) == 0 {
                continue;
            }
            for succ in space.bond_sectors(b + 1) {
                if x.rank(b + 1, succ) == 0 {
                    continue;
                }
                if (qntt::su2::theta(m.two_s, succ.two_s) - 1.0).abs() > 1e-12 {
                    n_theta += 1;
                }
            }
        }
    }
    for b in 0..=d {
        for s in space.bond_sectors(b) {
            if x.rank(b, s) == 0 {
                continue;
            }
            if (qntt::su2::centerweight(s.two_s, space.two_s) - 1.0).abs() > 1e-12 {
                n_cw += 1;
            }
        }
    }
    (n_theta, n_cw)
}

pub(crate) fn assert_theta_cw_live<T: Scalar>(x: &Su2TtVector<T>, tag: &str) {
    let (n_theta, n_cw) = count_theta_cw(x);
    assert!(
        n_theta >= 2 && n_cw >= 2,
        "{tag}: VACUOUS — theta/centerweight trivial at this size (n_theta={n_theta}, \
         n_cw={n_cw}); the weighted SU(2) path is not exercised"
    );
}

pub(crate) fn count_core_theta<T: Scalar>(core: &Su2Core<T>) -> usize {
    let g = core.space.pg.irrep_of(core.k);
    let mut n = 0usize;
    for f in Su2Family::ALL {
        for l in core.transitions(f) {
            if core.row_ranks[l] == 0 {
                continue;
            }
            let Some(t) = f.apply(l, g) else { continue };
            if !core.space.contains(core.k + 1, t) || core.col_ranks[t] == 0 {
                continue;
            }
            if (qntt::su2::theta(l.two_s, t.two_s) - 1.0).abs() > 1e-12 {
                n += 1;
            }
        }
    }
    n
}

pub(crate) fn assert_exclusion_gap(ev: &[f64], nroots: usize, floor: f64, tag: &str) {
    assert!(
        ev.len() > nroots,
        "{tag}: spectrum of length {} has no level above root {}",
        ev.len(),
        nroots - 1
    );
    let gap = ev[nroots] - ev[nroots - 1];
    assert!(
        gap >= floor,
        "{tag}: exclusion gap E[{nroots}] − E[{}] = {gap:.4e} below the fixture floor \
         {floor:.4e} — change the fixture, never the floor",
        nroots - 1
    );
}

pub(crate) fn assert_rule_replay(hist: &SsHistory, e_lower: &[f64], tag: &str) {
    for it in 1..hist.sweeps.len() {
        let prev = &hist.sweeps[it - 1];
        let want = penalty_weight(prev.energy, &prev.overlaps, e_lower, prev.penalty);
        assert_f64_bits(
            hist.sweeps[it].penalty,
            want,
            &format!("{tag}: sweep {it} penalty replayed from sweep {}", it - 1),
        );
    }
}
