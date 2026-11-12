// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Orbital-rotation primitives on the SU(2) (spin-adapted) layer: the spin-summed singlet
//! excitation `E_pq`, its antisymmetric generator `G_pq`, the one-body rotation generator `κ̂` as an
//! ordinary [`Su2SparseHamiltonian`], and the orbital gradient of a fixed state's energy.

use crate::coeff::CoeffTensor;
use crate::rotation::{
    GivensNetwork, RotateInfo, RotateOpts, assert_antisym, givens_cs, givens_decompose,
    orbital_pairs, pair_target,
};
use crate::scalar::Scalar;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::qn::Su2Space;
use crate::su2::tensorops::adag_i_a_j;
use crate::su2::transition1rdm::su2_transition_1rdm_h;
use crate::su2::vector::Su2TtVector;

/// `t · E_pq · x = t · Σ_σ a†_{pσ} a_{qσ} x`, **exact and unrounded**.
pub fn e_pq<T: Scalar>(x: &Su2TtVector<T>, p: usize, q: usize, t: T) -> Su2TtVector<T> {
    let d = x.space.d;
    assert!(
        p < d && q < d,
        "e_pq: orbital index out of range (p={p}, q={q}, d={d})"
    );
    &adag_i_a_j(x, p, q) * t
}

/// `G_pq · x = (E_pq − E_qp) · x`, exact and unrounded (two [`e_pq`] applies joined by the exact
/// [`Su2TtVector`] subtraction).
pub fn g_pq<T: Scalar>(x: &Su2TtVector<T>, p: usize, q: usize) -> Su2TtVector<T> {
    assert_ne!(p, q, "G_pp ≡ 0; g_pq requires p != q (got p = q = {p})");
    let a = e_pq(x, p, q, T::one());
    let b = e_pq(x, q, p, T::one());
    &a - &b
}

/// The one-body generator `κ̂ = Σ_{p,q} κ[p,q]·E_pq` assembled as an ordinary SU(2) term automaton.
pub fn kappa_operator<T: Scalar>(
    space: Su2Space,
    kappa: &CoeffTensor<T>,
) -> Su2SparseHamiltonian<T> {
    let d = space.d;
    assert!(
        kappa.rank == 2 && kappa.d == d,
        "kappa must be rank-2 over d={d} sites (got rank {}, d {})",
        kappa.rank,
        kappa.d
    );
    assert_antisym(kappa);
    let pg = space.pg;
    if !pg.is_trivial() {
        for p in 0..d {
            for q in 0..d {
                if pg.irrep_of(p) != pg.irrep_of(q) && kappa.get(&[p, q]).abs_s() != 0.0 {
                    panic!(
                        "kappa has a nonzero entry k[{p},{q}] = {:e} across irreps {:?} != {:?}: \
                         the one-body term a†_{p} a_{q} carries the flux Γ_{p} ⊗ Γ_{q}, which maps \
                         the space's target irrep to target ⊗ Γ_{p} ⊗ Γ_{q} — a single-target \
                         Su2Space cannot represent it. Enumerate generators with \
                         `crate::rotation::orbital_pairs`, which filters exactly these out.",
                        kappa.get(&[p, q]).abs_s(),
                        pg.irrep_of(p),
                        pg.irrep_of(q),
                    );
                }
            }
        }
    }
    Su2SparseHamiltonian::new(space, kappa, &CoeffTensor::<T>::zeros(d, 4), None, 0.0)
}

/// `κ̂ · x`. A pure alias for [`kappa_operator`] followed by the matvec, with no special-casing, so
/// the two entry points are bit-identical on every κ — including `κ = 0`, where both give the exact
/// zero train.
pub fn kappa_apply<T: Scalar>(kappa: &CoeffTensor<T>, x: &Su2TtVector<T>) -> Su2TtVector<T> {
    &kappa_operator(x.space, kappa) * x
}

/// The SU(2) twin of [`crate::rotation::apply_orbital_signs`]: orbital `p` with `signs[p] = −1` is
/// `a_p → −a_p` (spin-free), the phase `(−1)^{n_p}`, so the one-electron families of site `p` —
/// `Up` and `Dn`, the two S±½ coupling branches — flip sign and `Empty`/`Occ` stay. Exact and
/// site-local; the orthogonality flag is kept. Every entry of `signs` must be `±1`.
pub fn apply_orbital_signs<T: Scalar>(x: &mut Su2TtVector<T>, signs: &[f64]) {
    use crate::su2::core::Family;
    assert_eq!(
        signs.len(),
        x.space.d,
        "apply_orbital_signs: one sign per orbital"
    );
    let minus = T::from_f64(-1.0);
    for (p, &sg) in signs.iter().enumerate() {
        assert!(
            sg == 1.0 || sg == -1.0,
            "apply_orbital_signs: signs[{p}] = {sg} is not ±1"
        );
        if sg == -1.0 {
            let core = &mut x.cores[p];
            for f in [Family::Up, Family::Dn] {
                let secs: Vec<_> = core.transitions(f).collect();
                for s in secs {
                    let blk = core.block_mut(f, s).expect("transition present");
                    crate::core::algebra::block_scale(blk, minus);
                }
            }
        }
    }
}

/// The SU(2) twin of [`crate::rotation::apply_orbital_rotation`] (a per-layer duplicate
/// driver): the real orthogonal orbital rotation `U ∈ O(d)` (row-major `u[old·d + new]`)
/// applied to a spin-adapted state, `(Û x, info)`. Options and report are the shared
/// [`RotateOpts`] / [`RotateInfo`]; `chi` and `max_rank` count REDUCED multiplets.
pub fn apply_orbital_rotation<T: Scalar>(
    x: &Su2TtVector<T>,
    u: &[f64],
    opts: &RotateOpts,
) -> (Su2TtVector<T>, RotateInfo) {
    const WHO: &str = "su2::orbital::apply_orbital_rotation";
    assert!(
        x.space.pg.is_trivial(),
        "{WHO}: the state lives on a typed (point-group) space; a general orbital rotation mixes \
         irreps, so embed the state with `forget_point_group` first"
    );
    opts.validate(WHO);
    let net = givens_decompose(u, x.space.d);
    if net.gates.is_empty() {
        let mut y = x.clone();
        if net.signs.contains(&-1.0) {
            apply_orbital_signs(&mut y, &net.signs);
        }
        let info = RotateInfo {
            gates: 0,
            discarded: Vec::new(),
            error_bound: 0.0,
            max_rank: x.max_total_bond_rank(),
            norm_out: 1.0,
        };
        return (y, info);
    }
    opts.install(|| rotation_walk(x, &net, opts))
}

fn rotation_walk<T: Scalar>(
    x: &Su2TtVector<T>,
    net: &GivensNetwork,
    opts: &RotateOpts,
) -> (Su2TtVector<T>, RotateInfo) {
    use crate::core::twosite::noise_rng;
    use crate::su2::twosite::{apply_mode_rotation_cs, contract, factor_svd_noise};
    let mut y = x.clone();
    if let Some(p) = pair_target(y.ortho, net.gates[0].0) {
        y.move_core(p, true);
    }
    let n_in = y.norm();
    assert!(
        n_in.is_finite() && n_in > 0.0,
        "su2::orbital::apply_orbital_rotation: input norm {n_in} must be finite and > 0"
    );
    y.scale_mut(T::from_f64(1.0 / n_in));
    apply_orbital_signs(&mut y, &net.signs);
    let mut discarded = Vec::with_capacity(net.gates.len());
    let mut max_rank = y.max_total_bond_rank();
    for &(k, theta) in &net.gates {
        if let Some(p) = pair_target(y.ortho, k) {
            y.move_core(p, true);
        }
        let (c, s) = givens_cs(theta);
        let tc = apply_mode_rotation_cs(&contract(y.core(k), y.core(k + 1)), c, s);
        let (xk, sk, mut xk1, trunc) =
            factor_svd_noise(&tc, opts.eps, opts.chi, false, 0.0, &mut noise_rng(0));
        xk1.fold_sketch_left_mut(&sk);
        discarded.push(if trunc.discarded_weight == 0.0 {
            0.0
        } else {
            let mut r = tc;
            r.axpy(T::from_f64(-1.0), &contract(&xk, &xk1));
            r.dot_cores(&r).re_s()
        });
        max_rank = max_rank.max(trunc.kept);
        y.set_cores(xk, xk1);
        y.ortho = Some(k + 1);
    }
    let norm_out = y.norm();
    y.scale_mut(T::from_f64(n_in));
    let error_bound = discarded.iter().map(|&w: &f64| w.max(0.0).sqrt()).sum();
    let info = RotateInfo {
        gates: net.gates.len(),
        discarded,
        error_bound,
        max_rank,
        norm_out,
    };
    (y, info)
}

/// Gradient of the rotated-operator energy `E(κ) = ⟨x| e^{κ̂} H e^{−κ̂} |x⟩` at `κ = 0`, with the
/// state `x` held **fixed**: the antisymmetric `d×d` matrix `g[p,q] = ⟨x|[G_pq, H]|x⟩`.
pub fn orbital_gradient(h: &Su2SparseHamiltonian<f64>, x: &Su2TtVector<f64>) -> CoeffTensor<f64> {
    let space = h.space;
    let d = space.d;
    assert_eq!(
        x.space, space,
        "orbital_gradient: state and operator must share a Su2Space"
    );

    let gamma = su2_transition_1rdm_h(x, h, x);

    let mut g = CoeffTensor::<f64>::zeros(d, 2);
    for &(p, q) in &orbital_pairs(space.pg, d) {
        let val = 2.0 * (gamma.get(&[p, q]) - gamma.get(&[q, p]));
        g.set(&[p, q], val);
        g.set(&[q, p], -val);
    }
    g
}
