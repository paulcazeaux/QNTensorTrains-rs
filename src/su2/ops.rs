// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2) inner product and norm. **Both are UNWEIGHTED** — identical shape to the abelian
//! `crate::ops::dot`, just with SU(2) sector labels and the SU(2) 4-family connectivity
//! (`p * core` and `adjoint(core) * (...)` are plain per-block `matmul`, no `θ`, no
//! `centerweight`). This is the crux invariant: `dot(x,y) == dot(expand(x,twoSz),
//! expand(y,twoSz))` for every admissible `twoSz`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::qn::{StateSym, SymGrid, SymSpace, SymStep, Symmetry};
use crate::su2::vector::{Su2TtVector, apply_group_element};

/// Hermitian inner product `⟨x, y⟩ = conj(x)·y` (`dot`). Fast path only
/// (unit rank-1 boundaries at bond 0 and bond `d`); an open block goes through [`dot_block`], which
/// returns the boundary Gram from the same fold. UNWEIGHTED left-to-right
/// sweep: per site, `temp = env · core(y, k)` then `env' = adjoint(core(x,k)) · temp`, accumulated
/// over the four SU(2) families in `Empty, Up, Dn, Occ` order.
pub fn dot<T: Scalar, S: Symmetry>(x: &Su2TtVector<T, S>, y: &Su2TtVector<T, S>) -> T {
    assert_eq!(x.space, y.space, "dot: Su2Space mismatch");
    let space = x.space;
    let d = space.sites();

    let s0 = space.root();
    let last = space.top();
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
        env = fold_site(x, y, k, x.core(k), y.core(k), &env);
    }

    *env[last]
        .as_ref()
        .expect("final env block present")
        .get(0, 0)
}

type Env<T, S> = <S as Symmetry>::Grid<Option<Mat<T>>>;

fn seed_env<T: Scalar, S: Symmetry>(x: &Su2TtVector<T, S>, y: &Su2TtVector<T, S>) -> Env<T, S> {
    let space = x.space;
    let s0 = space.root();
    <Env<T, S>>::from_fn(&space, |s| {
        space.admissible(0, s).then(|| {
            let mut m = Mat::<T>::zeros(x.ranks[0][s], y.ranks[0][s]);
            if s == s0 {
                *m.get_mut(0, 0) = T::one();
            }
            m
        })
    })
}

fn fold_site<T: Scalar, S: Symmetry>(
    x: &Su2TtVector<T, S>,
    y: &Su2TtVector<T, S>,
    k: usize,
    cx: &Su2Core<T, S>,
    cy: &Su2Core<T, S>,
    env: &Env<T, S>,
) -> Env<T, S> {
    let space = x.space;
    let mut temp = Su2Core::<T, S>::zeros_sym(space, k, &x.ranks[k], &y.ranks[k + 1]);
    for &f in S::STEPS {
        let secs: Vec<S::Label> = temp.transitions(f).collect();
        for s in secs {
            let e = env[s]
                .as_ref()
                .expect("env block present at admissible sector");
            let yb = cy.block(f, s).expect("y transition present");
            let tb = temp.block_mut(f, s).unwrap();
            matmul(tb, Accum::Replace, e.as_ref(), yb, T::one(), Par::Seq);
        }
    }

    let mut next: Env<T, S> = <Env<T, S>>::from_fn(&space, |t| {
        space
            .admissible(k + 1, t)
            .then(|| Mat::<T>::zeros(x.ranks[k + 1][t], y.ranks[k + 1][t]))
    });
    let g = space.site_ctx(k);
    for t in space.bond_labels(k + 1) {
        for &f in S::STEPS {
            let Some(s) = f.step_unapply(t, g) else {
                continue;
            };
            if !space.admissible(k, s) {
                continue;
            }
            let xb = cx.block(f, s).expect("x transition present");
            let tb = temp.block(f, s).expect("temp transition present");
            let dst = next[t].as_mut().unwrap();
            matmul(dst, Accum::Add, xb.adjoint(), tb, T::one(), Par::Seq);
        }
    }
    next
}

pub(crate) type OpenSite<'a, T, S = StateSym> = Option<(usize, &'a [Su2Core<T, S>])>;

pub(crate) fn gram_fold<T: Scalar, S: Symmetry>(
    x: &Su2TtVector<T, S>,
    xc: OpenSite<'_, T, S>,
    y: &Su2TtVector<T, S>,
    yc: OpenSite<'_, T, S>,
) -> Vec<Mat<T>> {
    assert_eq!(x.space, y.space, "dot_block: Su2Space mismatch");
    let space = x.space;
    let d = space.sites();
    let s0 = space.root();
    let last = space.top();
    assert_eq!(x.rank(0, s0), 1, "dot_block needs unit rank at bond 0 (x)");
    assert_eq!(y.rank(0, s0), 1, "dot_block needs unit rank at bond 0 (y)");

    let (mut nx, mut ny) = (1usize, 1usize);
    let mut envs: Vec<Env<T, S>> = vec![seed_env(x, y)];
    for k in 0..d {
        let xs: Vec<&Su2Core<T, S>> = match xc {
            Some((p, cs)) if p == k => cs.iter().collect(),
            _ => vec![x.core(k)],
        };
        let ys: Vec<&Su2Core<T, S>> = match yc {
            Some((p, cs)) if p == k => cs.iter().collect(),
            _ => vec![y.core(k)],
        };
        let (nx2, ny2) = (nx.max(xs.len()), ny.max(ys.len()));
        let mut next = Vec::with_capacity(nx2 * ny2);
        for i in 0..nx2 {
            for j in 0..ny2 {
                let src = &envs[(i % nx) * ny + (j % ny)];
                next.push(fold_site(x, y, k, xs[i % xs.len()], ys[j % ys.len()], src));
            }
        }
        (nx, ny, envs) = (nx2, ny2, next);
    }
    envs.into_iter()
        .map(|mut e| e[last].take().expect("final env block present"))
        .collect()
}

/// Boundary Gram of two OPEN BLOCKS: `G[i, j] = ⟨x_i, y_j⟩` (conjugation on `x`), of shape
/// `x.boundary_width() × y.boundary_width()`. UNWEIGHTED, like [`dot`]: it equals the dense Gram
/// of every fixed-`Sz` member. On a unit-boundary pair it is the `1×1` matrix holding [`dot`].
pub fn dot_block<T: Scalar, S: Symmetry>(x: &Su2TtVector<T, S>, y: &Su2TtVector<T, S>) -> Mat<T> {
    gram_fold(x, None, y, None)
        .pop()
        .expect("one state pair without open sites")
}

/// Irrep weights of an SU(2) state under an orbital point group: `w_Γ = |G|⁻¹ Σ_h χ_Γ(h)
/// ⟨x, h·x⟩`, indexed by the irrep's XOR index (`w[0]` is the totally-symmetric weight),
/// normalized by `⟨x, x⟩` in the layer's own (fixed-`Sz`-member) inner product. The SU(2) twin of
/// [`crate::ops::irrep_weights`]: same signature shape, index convention, normalization and
/// trivial-group behaviour. The point group is the ORBITAL table, passed separately so the readout
/// works on a state in an untyped (trivial-group) space. Cost: `|G| − 1` sign flips
/// ([`apply_group_element`], one clone each) and inner products ([`dot`]) — a trivial group
/// returns `[1.0]` without any.
pub fn irrep_weights<T: Scalar>(x: &Su2TtVector<T>, pg: PointGroup) -> Vec<f64> {
    let order = pg.order() as usize;
    if order <= 1 {
        return vec![1.0];
    }
    let nrm2 = dot(x, x).re_s();
    let mut ov = vec![0.0f64; order];
    ov[0] = 1.0;
    for (hi, o) in ov.iter_mut().enumerate().skip(1) {
        let hx = apply_group_element(x, pg, Irrep::new(hi as u8));
        *o = dot(x, &hx).re_s() / nrm2;
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

impl<T: Scalar, S: Symmetry> Su2TtVector<T, S> {
    /// Euclidean norm. **The orthogonal center-core shortcut**: when the train carries an
    /// orthogonality center `k`, `‖x‖` equals the
    /// CENTERWEIGHTED norm of the center core alone ([`Su2Core::norm_center`], the fixed-Sz-member
    /// metric) — the surrounding cores are isometries in their respective (unweighted-left /
    /// θ-weighted-right) metrics. Non-orthogonal trains fall back to the unweighted
    /// `sqrt(|Re dot(x,x)|)` path, which is exact for any unit-boundary train regardless of
    /// canonical form (so the shortcut is a fidelity + perf change, not a correctness fix; the two
    /// paths agree to roundoff).
    pub fn norm(&self) -> f64 {
        if let Some(center) = self.ortho {
            return self.cores[center].norm_center();
        }
        if self.boundary_width() == 1 {
            return dot(self, self).re_s().abs().sqrt();
        }
        let g = dot_block(self, self);
        let mut tr = 0.0;
        for i in 0..g.nrows() {
            tr += g.get(i, i).re_s();
        }
        tr.abs().sqrt()
    }

    /// Width `ρ_d` of the open right boundary: the reduced rank at bond `d`'s `(N, 2S, target)`
    /// sector. `1` for a state; an open block carries `ρ_d` states of the same `(N, S)`.
    pub fn boundary_width(&self) -> usize {
        let sp = &self.space;
        self.ranks[sp.sites()][sp.top()]
    }

    /// Column `j` of an open block as a unit-boundary train (SU(2) twin of
    /// [`crate::TtVector::column`]). `ortho` survives only for a centre at the last site.
    pub fn column(&self, j: usize) -> Self {
        let space = self.space;
        let d = space.sites();
        let last = space.top();
        let rho = self.boundary_width();
        assert!(j < rho, "column {j} of a width-{rho} block");
        let mut col_ranks = self.ranks[d].clone();
        col_ranks[last] = 1;
        let src = self.core(d - 1);
        let mut core = Su2Core::<T, S>::zeros_sym(space, d - 1, &self.ranks[d - 1], &col_ranks);
        for &f in S::STEPS {
            let secs: Vec<S::Label> = src.transitions(f).collect();
            for s in secs {
                let b = src.block(f, s).expect("transition present");
                if b.ncols() == 0 {
                    continue;
                }
                let mut dst = core.block_mut(f, s).expect("same transitions");
                for i in 0..b.nrows() {
                    *dst.as_mut().get_mut(i, 0) = *b.get(i, j);
                }
            }
        }
        let mut cores = self.cores.clone();
        cores[d - 1] = core;
        let mut out = Self::new(space, cores);
        out.ortho = self.ortho.filter(|&c| c == d - 1);
        out
    }
}

impl<T: Scalar> Su2TtVector<T> {
    /// Complex conjugate — SU(2) twin of [`crate::TtVector::conj`], which the spin-adapted layer
    /// was missing entirely. Identity for real `T`.
    pub fn conj(&self) -> Self {
        let mut out = self.clone();
        out.conj_mut();
        out
    }

    /// In-place complex conjugation. Leaves ranks, structure and `ortho` untouched — conjugation is
    /// an isometry of the Hermitian metric, so a canonical train stays canonical at the same
    /// centre. Linear in the coefficient count: one [`Su2Core::for_each_coeff_mut`] walk,
    /// not the quadratic `coeff`/`set_coeff` pair.
    pub fn conj_mut(&mut self) {
        for core in &mut self.cores {
            core.for_each_coeff_mut(|c| *c = c.conj_s());
        }
    }
}

#[cfg(test)]
mod cast_tests {
    use super::dot;
    use crate::C64;
    use crate::scalar::Scalar;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::{Su2TtVector, tt_randn};
    use crate::test_util::{seeded, su2_capped_grids, su2_coeffs};

    #[test]
    fn su2_conj_gives_the_bilinear_form_and_hence_rho() {
        let space = Su2Space::new(6, 6, 0);
        let a = tt_randn::<f64, _>(space, &su2_capped_grids(&space, 4), &mut seeded(1)).to_c64();
        let b = tt_randn::<f64, _>(space, &su2_capped_grids(&space, 4), &mut seeded(2)).to_c64();

        let rho = |x: &Su2TtVector<C64>| {
            let herm = dot(x, x).re_s();
            dot(&x.conj(), x).abs_s() / herm
        };

        let aa = a.conj().conj();
        assert_eq!(aa.ortho, a.ortho);
        for k in 0..space.d {
            assert_eq!(su2_coeffs(&aa.cores[k]), su2_coeffs(&a.cores[k]));
        }

        let r = rho(&a);
        assert!((r - 1.0).abs() < 1e-12, "real-embedded state read ρ = {r}");

        let phased = &a * C64::new(0.6, 0.8);
        let rp = rho(&phased);
        assert!((rp - 1.0).abs() < 1e-12, "ρ is not phase-invariant: {rp}");

        let mixed = &a + &(&b * C64::new(0.0, 1.0));
        let rm = rho(&mixed);
        assert!(rm < 1.0 - 1e-6, "a + i·b read ρ = {rm}, must be < 1");
    }
}

#[cfg(test)]
mod s1_tests {
    use super::dot as su2_dot;
    use crate::su2::expand::expand;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::randomized_su2_state;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn sum_scalar_scaling_unary_negation_and_dot_linearity() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x9999_AAAA);
        let space = Su2Space::new(6, 4, 2);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let nx = x.norm();

        let scaled = &x * 2.5;
        assert!((scaled.norm() - 2.5 * nx).abs() < 1e-12 * nx.max(1.0));
        let scaled2 = 2.5 * &x;
        assert!((scaled2.norm() - 2.5 * nx).abs() < 1e-12 * nx.max(1.0));
        let neg_scaled = &x * (-3.0);
        assert!((neg_scaled.norm() - 3.0 * nx).abs() < 1e-12 * nx.max(1.0));
        let divided = &x / 4.0;
        assert!((divided.norm() - nx / 4.0).abs() < 1e-12 * nx.max(1.0));

        let plus_minus = &(-(&x)) + &x;
        assert!(plus_minus.norm() < 1e-10 * nx.max(1.0));

        let dneg = su2_dot(&(-(&x)), &x);
        let dpos = su2_dot(&x, &x);
        assert!((dneg + dpos).abs() < 1e-10 * dpos.abs().max(1.0));
    }

    #[test]
    fn s1_29_gaussian_mode_norm_via_dot_and_expansion() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5555_6666);
        let space = Su2Space::new(6, 5, 1);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let via_dot = su2_dot(&x, &x).abs().sqrt();
        assert!((x.norm() - via_dot).abs() < 1e-12 * via_dot.max(1.0));
        let e = expand(&x, x.two_s() as i64);
        assert!((x.norm() - e.norm()).abs() < 1e-12 * e.norm().max(1.0));
    }
}

#[cfg(test)]
pub(crate) mod block_tests;

#[cfg(test)]
mod su2_irrep_weights_tests;
