// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Vector-level algebra on `TtVector`: the Hermitian inner product `dot`, `norm`, `+`/`-`/unary
//! `-`, scalar multiply/divide, constant offset `add_scalar`, conjugation, and the `⊗`-based
//! `hadamard`.

use std::ops::{Add, Div, Mul, Neg, Sub};

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use crate::core::{Family, SparseCore};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{Sector, SectorGrid};
use crate::scalar::{C64, Scalar};
use crate::vector::TtVector;

/// Hermitian inner product `⟨x, y⟩ = conj(x)·y`. Panics on space mismatch. Requires unit boundary
/// ranks; see [`dot_block`] for open blocks.
pub fn dot<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>) -> T {
    dot_phased(x, y, None)
}

type FamilyPhase<'a> = Option<&'a dyn Fn(usize, Family) -> f64>;

fn dot_phased<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>, phase: FamilyPhase<'_>) -> T {
    assert_eq!(x.space, y.space, "dot: QnSpace mismatch");
    let space = x.space;
    let d = space.d;

    let s0 = Sector::new(0, 0);
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    assert_eq!(
        x.rank(0, s0),
        1,
        "dot fast path needs unit rank at bond 0 (x)"
    );
    assert_eq!(
        y.rank(0, s0),
        1,
        "dot fast path needs unit rank at bond 0 (y)"
    );
    assert_eq!(
        x.rank(d, last),
        1,
        "dot fast path needs unit rank at bond d (x)"
    );
    assert_eq!(
        y.rank(d, last),
        1,
        "dot fast path needs unit rank at bond d (y)"
    );

    let mut env = seed_env(x, y);
    for k in 0..d {
        env = fold_site(x, y, k, x.core(k), y.core(k), &env, phase);
    }

    *env[last]
        .as_ref()
        .expect("final env block present")
        .get(0, 0)
}

type Env<T> = SectorGrid<Option<Mat<T>>>;

fn seed_env<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>) -> Env<T> {
    let space = x.space;
    let s0 = Sector::new(0, 0);
    SectorGrid::from_fn(&space, |s| {
        space.contains(0, s).then(|| {
            let mut m = Mat::<T>::zeros(x.ranks[0][s], y.ranks[0][s]);
            if s == s0 {
                *m.get_mut(0, 0) = T::one();
            }
            m
        })
    })
}

fn fold_site<T: Scalar>(
    x: &TtVector<T>,
    y: &TtVector<T>,
    k: usize,
    cx: &SparseCore<T>,
    cy: &SparseCore<T>,
    env: &Env<T>,
    phase: FamilyPhase<'_>,
) -> Env<T> {
    let space = x.space;
    let mut temp = SparseCore::<T>::zeros(space, k, &x.ranks[k], &y.ranks[k + 1]);
    for f in Family::ALL {
        let secs: Vec<Sector> = temp.transitions(f).collect();
        for s in secs {
            let e = env[s]
                .as_ref()
                .expect("env block present at admissible sector");
            let yb = cy.block(f, s).expect("y transition present");
            let tb = temp.block_mut(f, s).unwrap();
            matmul(tb, Accum::Replace, e.as_ref(), yb, T::one(), Par::Seq);
        }
    }

    let mut next: Env<T> = SectorGrid::from_fn(&space, |t| {
        space
            .contains(k + 1, t)
            .then(|| Mat::<T>::zeros(x.ranks[k + 1][t], y.ranks[k + 1][t]))
    });
    let g = space.pg.irrep_of(k);
    for t in space.bond_sectors(k + 1) {
        for f in Family::ALL {
            let Some(s) = f.unapply(t, g) else {
                continue;
            };
            if !space.contains(k, s) {
                continue;
            }
            let xb = cx.block(f, s).expect("x transition present");
            let tb = temp.block(f, s).expect("temp transition present");
            let dst = next[t].as_mut().unwrap();
            let alpha = match phase {
                None => T::one(),
                Some(p) => T::from_f64(p(k, f)),
            };
            matmul(dst, Accum::Add, xb.adjoint(), tb, alpha, Par::Seq);
        }
    }
    next
}

pub(crate) type OpenSite<'a, T> = Option<(usize, &'a [SparseCore<T>])>;

pub(crate) fn gram_fold<T: Scalar>(
    x: &TtVector<T>,
    xc: OpenSite<'_, T>,
    y: &TtVector<T>,
    yc: OpenSite<'_, T>,
) -> Vec<Mat<T>> {
    assert_eq!(x.space, y.space, "dot_block: QnSpace mismatch");
    let space = x.space;
    let d = space.d;
    let s0 = Sector::new(0, 0);
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    assert_eq!(x.rank(0, s0), 1, "dot_block needs unit rank at bond 0 (x)");
    assert_eq!(y.rank(0, s0), 1, "dot_block needs unit rank at bond 0 (y)");

    let (mut nx, mut ny) = (1usize, 1usize);
    let mut envs: Vec<Env<T>> = vec![seed_env(x, y)];
    for k in 0..d {
        let xs: Vec<&SparseCore<T>> = match xc {
            Some((p, cs)) if p == k => cs.iter().collect(),
            _ => vec![x.core(k)],
        };
        let ys: Vec<&SparseCore<T>> = match yc {
            Some((p, cs)) if p == k => cs.iter().collect(),
            _ => vec![y.core(k)],
        };
        let (nx2, ny2) = (nx.max(xs.len()), ny.max(ys.len()));
        let mut next = Vec::with_capacity(nx2 * ny2);
        for i in 0..nx2 {
            for j in 0..ny2 {
                let src = &envs[(i % nx) * ny + (j % ny)];
                let cx = xs[i % xs.len()];
                let cy = ys[j % ys.len()];
                next.push(fold_site(x, y, k, cx, cy, src, None));
            }
        }
        (nx, ny, envs) = (nx2, ny2, next);
    }
    envs.into_iter()
        .map(|mut e| e[last].take().expect("final env block present"))
        .collect()
}

/// Boundary Gram of two OPEN BLOCKS: `G[i, j] = ⟨x_i, y_j⟩` (conjugation on `x`), of shape
/// `x.boundary_width() × y.boundary_width()`, where column `j` of a block is the train obtained
/// by fixing its open bond-`d` index ([`TtVector::column`]). Bond 0 must carry unit rank.
pub fn dot_block<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>) -> Mat<T> {
    gram_fold(x, None, y, None)
        .pop()
        .expect("one state pair without open sites")
}

/// Irrep weights of a state under an orbital point group: `w_Γ = |G|⁻¹ Σ_h χ_Γ(h) ⟨x, h·x⟩`,
/// indexed by the irrep's XOR index (`w[0]` is the totally-symmetric weight), normalized by
/// `⟨x, x⟩`. The point group is the ORBITAL table (which orbital carries which irrep), passed
/// separately so the readout works on a state in an untyped (trivial-group) space — the case it
/// exists for: an untyped polish that lowers the energy by leaving the symmetric sector shows
/// `w[0] < 1` (the eps-1e-3 untyped water state). Cost: `|G| − 1` inner products
/// ([`dot_phased`], no clone) — a trivial group returns `[1.0]` without any.
pub fn irrep_weights<T: Scalar>(x: &TtVector<T>, pg: PointGroup) -> Vec<f64> {
    let order = pg.order() as usize;
    if order <= 1 {
        return vec![1.0];
    }
    let d = x.space.d;
    let nrm2 = dot(x, x).re_s();
    let mut ov = vec![0.0f64; order];
    ov[0] = 1.0;
    for (hi, o) in ov.iter_mut().enumerate().skip(1) {
        let h = Irrep::new(hi as u8);
        let odd: Vec<bool> = (0..d).map(|k| pg.irrep_of(k).character(h) < 0.0).collect();
        let phase = |k: usize, f: Family| {
            if odd[k] && matches!(f, Family::Up | Family::Dn) {
                -1.0
            } else {
                1.0
            }
        };
        *o = dot_phased(x, x, Some(&phase)).re_s() / nrm2;
    }
    (0..order)
        .map(|g| {
            let gi = Irrep::new(g as u8);
            (0..order)
                .map(|hi| gi.character(Irrep::new(hi as u8)) * ov[hi])
                .sum::<f64>()
                / order as f64
        })
        .collect()
}

/// Elementwise (Hadamard) product of two trains via the per-sector Kronecker core kernel. Ranks
/// multiply per sector.
pub fn hadamard<T: Scalar>(x: &TtVector<T>, y: &TtVector<T>) -> TtVector<T> {
    assert_eq!(x.space, y.space, "hadamard: QnSpace mismatch");
    let cores = (0..x.space.d)
        .map(|k| x.cores[k].kron(&y.cores[k]))
        .collect();
    TtVector::new(x.space, cores)
}

impl<T: Scalar> TtVector<T> {
    /// Width `ρ_d` of the open right boundary: the rank at bond `d`'s `(Nup, Ndn, target)`
    /// sector. `1` for a state vector; an open block carries `ρ_d` of them in one train.
    pub fn boundary_width(&self) -> usize {
        let sp = &self.space;
        self.ranks[sp.d][Sector::with_irrep(sp.nup, sp.ndn, sp.target)]
    }

    /// Column `j` of an open block as a unit-boundary train: every core shared, the last one
    /// restricted to its `j`-th bond-`d` column. `ortho` survives only where it still holds (a
    /// centre at the last site; elsewhere the restricted last core is no longer an isometry).
    pub fn column(&self, j: usize) -> Self {
        let space = self.space;
        let d = space.d;
        let last = Sector::with_irrep(space.nup, space.ndn, space.target);
        let rho = self.boundary_width();
        assert!(j < rho, "column {j} of a width-{rho} block");
        let mut col_ranks = self.ranks[d].clone();
        col_ranks[last] = 1;
        let src = self.core(d - 1);
        let mut core = SparseCore::zeros(space, d - 1, &self.ranks[d - 1], &col_ranks);
        for f in Family::ALL {
            let secs: Vec<Sector> = src.transitions(f).collect();
            for s in secs {
                let b = src.block(f, s).expect("transition present");
                if b.ncols() > 0 {
                    core.copy_block_into(f, s, 0, 0, b.subcols(j, 1));
                }
            }
        }
        let mut cores = self.cores.clone();
        cores[d - 1] = core;
        let mut out = TtVector::new(space, cores);
        out.ortho = self.ortho.filter(|&c| c == d - 1);
        out
    }

    /// Euclidean norm. Uses the orthogonality shortcut when `ortho` is set (Frobenius norm of the
    /// center core), else `sqrt(abs(re(dot(self, self))))`.
    pub fn norm(&self) -> f64 {
        if let Some(k) = self.ortho {
            self.cores[k].norm()
        } else if self.boundary_width() == 1 {
            dot(self, self).re_s().abs().sqrt()
        } else {
            let g = dot_block(self, self);
            let mut tr = 0.0;
            for i in 0..g.nrows() {
                tr += g.get(i, i).re_s();
            }
            tr.abs().sqrt()
        }
    }

    /// Scale in place: scales the ortho center core if orthogonal, else
    /// core 0. Preserves `ortho`.
    pub fn scale_mut(&mut self, a: T) {
        let idx = self.ortho.unwrap_or(0);
        self.cores[idx].scale_mut(a);
    }

    /// Complex conjugate, leaving ranks/structure and flags untouched.
    /// Identity for real `T`.
    pub fn conj(&self) -> Self {
        let mut out = self.clone();
        out.conj_mut();
        out
    }

    /// In-place complex conjugation.
    pub fn conj_mut(&mut self) {
        for core in &mut self.cores {
            core.conj_mut();
        }
    }

    /// Constant offset `self + c·𝟙`: the constant enters once, via the per-core constant-chain
    /// augmentation. Resets orthogonality.
    pub fn add_scalar(&self, c: T) -> Self {
        let cores = self.cores.iter().map(|core| core.oplus_scalar(c)).collect();
        TtVector::new(self.space, cores)
    }
}

impl<T: Scalar> Add for &TtVector<T> {
    type Output = TtVector<T>;
    fn add(self, rhs: &TtVector<T>) -> TtVector<T> {
        assert_eq!(self.space, rhs.space, "add: QnSpace mismatch");
        let cores = (0..self.space.d)
            .map(|k| self.cores[k].oplus(&rhs.cores[k]))
            .collect();
        TtVector::new(self.space, cores)
    }
}

impl<T: Scalar> Sub for &TtVector<T> {
    type Output = TtVector<T>;
    fn sub(self, rhs: &TtVector<T>) -> TtVector<T> {
        assert_eq!(self.space, rhs.space, "sub: QnSpace mismatch");
        let cores = (0..self.space.d)
            .map(|k| {
                if k == 0 {
                    self.cores[0].oplus(&rhs.cores[0].neg())
                } else {
                    self.cores[k].oplus(&rhs.cores[k])
                }
            })
            .collect();
        TtVector::new(self.space, cores)
    }
}

impl<T: Scalar> Neg for &TtVector<T> {
    type Output = TtVector<T>;
    fn neg(self) -> TtVector<T> {
        let mut out = self.clone();
        let idx = out.ortho.unwrap_or(0);
        out.cores[idx].scale_mut(T::from_f64(-1.0));
        out
    }
}

impl<T: Scalar> Mul<T> for &TtVector<T> {
    type Output = TtVector<T>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, b: T) -> TtVector<T> {
        let mut out = self.clone();
        let idx = out.ortho.unwrap_or(out.space.d - 1);
        out.cores[idx].scale_mut(b);
        out
    }
}

impl<T: Scalar> Div<T> for &TtVector<T> {
    type Output = TtVector<T>;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, b: T) -> TtVector<T> {
        self * b.recip()
    }
}

impl Mul<&TtVector<f64>> for f64 {
    type Output = TtVector<f64>;
    fn mul(self, x: &TtVector<f64>) -> TtVector<f64> {
        let mut out = x.clone();
        let idx = out.ortho.unwrap_or(0);
        out.cores[idx].scale_mut(self);
        out
    }
}

impl Mul<&TtVector<C64>> for C64 {
    type Output = TtVector<C64>;
    fn mul(self, x: &TtVector<C64>) -> TtVector<C64> {
        let mut out = x.clone();
        let idx = out.ortho.unwrap_or(0);
        out.cores[idx].scale_mut(self);
        out
    }
}

#[cfg(test)]
mod pg3_tests {
    use crate::scalar::Scalar;
    use crate::test_util::{naive_ranks, pg3_space_d6, rel_lr_c};
    use crate::{C64, TtVector, dot, tt_randn};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn pg3_complex_algebra_and_conj() {
        let (_pg, space) = pg3_space_d6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(2_303_032);
        let ranks = naive_ranks(&space, 0, 3, &mut rng);
        let x: TtVector<C64> = tt_randn(space, &ranks, &mut rng);
        assert!(x.check());
        let ranks2 = naive_ranks(&space, 0, 3, &mut rng);
        let y: TtVector<C64> = tt_randn(space, &ranks2, &mut rng);
        assert!(y.check());

        let dxy = dot(&x, &y);
        let dyx = dot(&y, &x);
        assert!((dxy - dyx.conj_s()).abs_s() <= 1e-9 * dxy.abs_s().max(1e-300));

        let a = C64::new(0.4, -1.3);
        let base = dot(&x, &y);
        let ax = a * &x;
        assert!(
            (dot(&ax, &y) - a.conj_s() * base).abs_s()
                <= 1e-9 * (a.conj_s() * base).abs_s().max(1e-300)
        );
        let ay = a * &y;
        assert!((dot(&x, &ay) - a * base).abs_s() <= 1e-9 * (a * base).abs_s().max(1e-300));

        let cc = x.conj().conj();
        assert!(rel_lr_c(&cc, &x) < 1e-12);

        let xd = x.to_dense();
        let xc = x.conj();
        assert!(xc.check());
        for (got, &orig) in xc.to_dense().iter().zip(&xd) {
            assert!((*got - orig.conj_s()).abs_s() < 1e-13);
        }
    }
}

#[cfg(test)]
mod phase2_tests {
    use crate::qn::QnSpace;
    use crate::scalar::Scalar;
    use crate::test_util::{dense_dot, randomized_state};
    use crate::{C64, TtVector, dot, tt_state, tt_zeros};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn dot_of_determinants_is_kronecker_delta() {
        let a = tt_state::<f64>(&[true, true, false, false], &[true, false, false, false]);
        let b = tt_state::<f64>(&[true, false, true, false], &[false, true, false, false]);
        let c = tt_state::<f64>(&[false, false, true, true], &[false, false, false, true]);
        for s in [&a, &b, &c] {
            s.check();
            assert_eq!(s.space, a.space);
        }
        assert!((dot(&a, &a) - 1.0).abs() < 1e-12);
        assert!((dot(&b, &b) - 1.0).abs() < 1e-12);
        assert!((dot(&c, &c) - 1.0).abs() < 1e-12);
        assert!(dot(&a, &b).abs() < 1e-12);
        assert!(dot(&a, &c).abs() < 1e-12);
        assert!(dot(&b, &c).abs() < 1e-12);
        assert!((dot(&a, &b) - dot(&b, &a)).abs() < 1e-12);
    }

    #[test]
    fn dot_is_bilinear_real() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(21);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let y: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let z: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let (alpha, beta) = (0.37, -1.8);

        let ax = alpha * &x;
        let by = beta * &y;
        let lhs = &ax + &by;
        let got = dot(&lhs, &z);
        let want = alpha * dot(&x, &z) + beta * dot(&y, &z);
        assert!(
            (got - want).abs() <= 1e-10 * want.abs().max(1.0),
            "left linearity"
        );

        let az = alpha * &z;
        let bx = beta * &x;
        let rhs = &az + &bx;
        let got2 = dot(&y, &rhs);
        let want2 = alpha * dot(&y, &z) + beta * dot(&y, &x);
        assert!(
            (got2 - want2).abs() <= 1e-10 * want2.abs().max(1.0),
            "right linearity"
        );
    }

    #[test]
    fn norm_of_zero_is_zero() {
        let x: TtVector<f64> = tt_zeros(QnSpace::from_n_two_sz(6, 4, 0));
        assert!(x.norm() < 1e-14);
    }

    #[test]
    fn add_scalar_enters_once() {
        let d = 5;
        let space = QnSpace::from_n_two_sz(d, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(700);
        let x: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);
        let c = 0.7;

        let ones: Vec<f64> = crate::tt_ones::<f64>(space).to_dense();
        let xd = x.to_dense();

        let x_plus_c = x.add_scalar(c);
        x_plus_c.check();
        let want_plus: Vec<f64> = xd.iter().zip(&ones).map(|(&a, &o)| a + c * o).collect();
        for (got, want) in x_plus_c.to_dense().iter().zip(&want_plus) {
            assert!((got - want).abs() < 1e-12, "x+c: {got} vs {want}");
        }

        let c_plus_x = x.add_scalar(c);
        for (got, want) in c_plus_x.to_dense().iter().zip(&want_plus) {
            assert!((got - want).abs() < 1e-12, "c+x: {got} vs {want}");
        }

        let c_minus_x = (-&x).add_scalar(c);
        c_minus_x.check();
        let want_minus: Vec<f64> = xd.iter().zip(&ones).map(|(&a, &o)| c * o - a).collect();
        for (got, want) in c_minus_x.to_dense().iter().zip(&want_minus) {
            assert!((got - want).abs() < 1e-12, "c-x: {got} vs {want}");
        }
    }

    #[test]
    fn x_minus_x_has_zero_norm() {
        let space = QnSpace::from_n_two_sz(5, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(900);
        let x: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);
        let neg = (-1.0) * &x;
        let z = &x + &neg;
        z.check();
        assert!(z.norm() < 1e-10, "norm(x + (-1)x) = {}", z.norm());
    }

    #[test]
    fn dot_conjugate_symmetry_complex() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1104);
        let x: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
        let y: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);

        let dxy = dot(&x, &y);
        let dyx = dot(&y, &x);
        assert!(
            (dxy - dyx.conj_s()).abs_s() <= 1e-9 * dxy.abs_s(),
            "conj symmetry"
        );

        let d_dense = dense_dot(&x.to_dense(), &y.to_dense());
        assert!(
            (dxy - d_dense).abs_s() <= 1e-8 * d_dense.abs_s(),
            "dot vs dense complex"
        );
    }

    #[test]
    fn dot_sesquilinearity_side_convention() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1105);
        let x: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
        let y: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
        let a = C64::new(0.4, -1.3);

        let base = dot(&x, &y);

        let ax = a * &x;
        let lhs_first = dot(&ax, &y);
        let rhs_first = a.conj_s() * base;
        assert!(
            (lhs_first - rhs_first).abs_s() <= 1e-9 * rhs_first.abs_s(),
            "first-arg: conj(a) expected"
        );

        let ay = a * &y;
        let lhs_second = dot(&x, &ay);
        let rhs_second = a * base;
        assert!(
            (lhs_second - rhs_second).abs_s() <= 1e-9 * rhs_second.abs_s(),
            "second-arg: a expected"
        );
    }

    #[test]
    fn conj_matches_elementwise_dense() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1106);
        let x: TtVector<C64> = randomized_state(space, 1, 3, &mut rng);
        let xd = x.to_dense();

        let xc = x.conj();
        xc.check();
        for (got, &orig) in xc.to_dense().iter().zip(&xd) {
            assert!((*got - orig.conj_s()).abs_s() < 1e-13, "conj mismatch");
        }

        let mut xm = x.clone();
        xm.conj_mut();
        for (a, b) in xm.to_dense().iter().zip(&xc.to_dense()) {
            assert!((*a - *b).abs_s() < 1e-14);
        }
    }

    #[test]
    fn fresh_vector_is_non_orthogonal_and_norm_is_sqrt_re_dot() {
        let space = QnSpace::from_n_two_sz(7, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1200);
        let x: TtVector<f64> = randomized_state(space, 1, 4, &mut rng);
        assert!(x.ortho.is_none(), "fresh vector must not be orthogonal");

        let n = x.norm();
        let n_ref = dot(&x, &x).re_s().sqrt();
        assert!(
            (n - n_ref).abs() <= 1e-14 * n_ref.max(1.0),
            "norm {n} vs {n_ref}"
        );
    }
}

#[cfg(test)]
mod irrep_weight_tests {
    use super::irrep_weights;
    use crate::pointgroup::{Irrep, PointGroup};
    use crate::vector::{TtVector, tt_state};

    #[test]
    fn irrep_weights_of_determinants_and_a_superposition() {
        let pg = PointGroup::new(
            2,
            &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(3)],
        );
        let a1: TtVector<f64> =
            tt_state(&[false, true, false, false], &[false, true, false, false]);
        let b: TtVector<f64> = tt_state(&[true, false, false, false], &[false, true, false, false]);
        let c: TtVector<f64> = tt_state(&[false, false, true, false], &[false, true, false, false]);
        let w = irrep_weights(&a1, pg);
        assert!(
            (w[0] - 1.0).abs() < 1e-14 && w[1..].iter().all(|v| v.abs() < 1e-14),
            "{w:?}"
        );
        let w = irrep_weights(&b, pg);
        assert!((w[1] - 1.0).abs() < 1e-14 && w[0].abs() < 1e-14, "{w:?}");
        let w = irrep_weights(&c, pg);
        assert!((w[3] - 1.0).abs() < 1e-14, "irrep 2 ⊕ 1 = 3: {w:?}");
        let x = &(&a1 * 0.6) + &(&b * 0.8);
        let w = irrep_weights(&x, pg);
        assert!(
            (w[0] - 0.36).abs() < 1e-13 && (w[1] - 0.64).abs() < 1e-13,
            "{w:?}"
        );
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-13);
        assert_eq!(irrep_weights(&x, PointGroup::TRIVIAL), vec![1.0]);
    }
}

#[cfg(test)]
pub(crate) mod block_tests;

#[cfg(test)]
mod ops_irrep_weights_tests;
