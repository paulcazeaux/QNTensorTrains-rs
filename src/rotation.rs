// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Orbital-rotation primitives: the spin-summed singlet excitation `E_pq`, its antisymmetric
//! generator `G_pq`, the one-body rotation generator `κ̂` as an ordinary [`SparseHamiltonian`], the
//! orbital gradient of a fixed state's energy, the integral transform `H ↦ e^{κ̂} H e^{−κ̂}` in the
//! coefficient picture, the exact exponential of a real antisymmetric matrix (whole-matrix and
//! irrep-blocked), the closed-shell frozen-core fold, and the canonical `p > q` pair list.

use crate::coeff::CoeffTensor;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::elementaryops::adag_i_a_j;
use crate::hamiltonian::SparseHamiltonian;
use crate::integrals::Integrals;
use crate::orbital::{Orbital, Spin};
use crate::pointgroup::{Irrep, PointGroup};
use crate::qn::{Pooled, QnSpace};
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// `t · E_pq · x = t · Σ_σ a†_{pσ} a_{qσ} x`, **exact and unrounded**.
pub fn e_pq<T: Scalar>(x: &TtVector<T>, p: usize, q: usize, t: T) -> TtVector<T> {
    let d = x.space.d;
    assert!(
        p < d && q < d,
        "e_pq: orbital index out of range (p={p}, q={q}, d={d})"
    );
    let up = adag_i_a_j(x, Orbital::new(p, Spin::Up), Orbital::new(q, Spin::Up), t);
    let dn = adag_i_a_j(x, Orbital::new(p, Spin::Dn), Orbital::new(q, Spin::Dn), t);
    &up + &dn
}

/// `G_pq · x = (E_pq − E_qp) · x`, exact and unrounded (four [`adag_i_a_j`] applies; ranks at most
/// four times the input's).
pub fn g_pq<T: Scalar>(x: &TtVector<T>, p: usize, q: usize) -> TtVector<T> {
    assert_ne!(p, q, "G_pp ≡ 0; g_pq requires p != q (got p = q = {p})");
    let a = e_pq(x, p, q, T::one());
    let b = e_pq(x, q, p, T::one());
    &a - &b
}

const ANTISYM_ATOL: f64 = 100.0 * f64::EPSILON;

/// Reject `kappa` unless it is a full antisymmetric `d×d` matrix, with the default tolerance floor
/// [`ANTISYM_ATOL`]. See [`assert_antisym_with_atol`] for the two clauses and why both are needed.
pub fn assert_antisym<T: Scalar>(kappa: &CoeffTensor<T>) {
    assert_antisym_with_atol(kappa, ANTISYM_ATOL);
}

/// [`assert_antisym`] with an explicit absolute floor, for callers who know the scale at which
/// `kappa` was assembled and want a tighter predicate.
pub fn assert_antisym_with_atol<T: Scalar>(kappa: &CoeffTensor<T>, atol: f64) {
    assert_eq!(kappa.rank, 2, "kappa must be a rank-2 coefficient tensor");
    let d = kappa.d;
    assert_antisym_predicate(
        d,
        |p, q| kappa.get(&[p, q]).abs_s(),
        |p, q| (kappa.get(&[p, q]) + kappa.get(&[q, p])).abs_s(),
        atol,
        "kappa",
    );
}

fn assert_antisym_predicate(
    d: usize,
    abs_at: impl Fn(usize, usize) -> f64,
    abs_sym_at: impl Fn(usize, usize) -> f64,
    atol: f64,
    what: &str,
) {
    let mut lo = false;
    let mut up = false;
    for p in 1..d {
        for q in 0..p {
            lo |= abs_at(p, q) != 0.0;
            up |= abs_at(q, p) != 0.0;
        }
    }
    assert!(
        !(lo ^ up),
        "{what} has exactly one nonzero strict triangle — pass the FULL antisymmetric matrix \
         (kappa[q,p] = -kappa[p,q]), not the p>q half"
    );

    let mut maxabs = 0.0f64;
    let mut maxsym = 0.0f64;
    for p in 0..d {
        for q in 0..d {
            maxabs = maxabs.max(abs_at(p, q));
        }
    }
    for p in 0..d {
        for q in p..d {
            maxsym = maxsym.max(abs_sym_at(p, q));
        }
    }
    let tol = atol.max(100.0 * f64::EPSILON * maxabs);
    assert!(
        maxsym <= tol,
        "{what} is not antisymmetric: max|k + kᵀ| = {maxsym:e} > tol {tol:e} (max|k| = {maxabs:e})"
    );
}

/// The one-body generator `κ̂ = Σ_{p,q} κ[p,q]·E_pq` assembled as an ordinary term automaton.
pub fn kappa_operator<T: Scalar>(space: QnSpace, kappa: &CoeffTensor<T>) -> SparseHamiltonian<T> {
    assert!(
        kappa.rank == 2 && kappa.d == space.d,
        "kappa must be rank-2 over d={} sites (got rank {}, d {})",
        space.d,
        kappa.rank,
        kappa.d
    );
    assert_antisym(kappa);
    SparseHamiltonian::new(
        space,
        kappa,
        &CoeffTensor::<T>::zeros(space.d, 4),
        None,
        0.0,
    )
}

/// `κ̂ · x`. A pure alias for [`kappa_operator`] followed by the matvec, with no special-casing, so
/// the two entry points are bit-identical on every κ — including `κ = 0`, where both give the exact
/// zero train.
pub fn kappa_apply<T: Scalar>(kappa: &CoeffTensor<T>, x: &TtVector<T>) -> TtVector<T> {
    &kappa_operator(x.space, kappa) * x
}

/// Gradient of the rotated-operator energy `E(κ) = ⟨x| e^{κ̂} H e^{−κ̂} |x⟩` at `κ = 0`, with the
/// state `x` held **fixed**: the antisymmetric `d×d` matrix `g[p,q] = ⟨x|[G_pq, H]|x⟩`.
pub fn orbital_gradient(h: &SparseHamiltonian<f64>, x: &TtVector<f64>) -> CoeffTensor<f64> {
    let space = h.space;
    let d = space.d;
    assert_eq!(
        x.space, space,
        "orbital_gradient: state and operator must share a QnSpace"
    );
    let pg = space.pg;

    let gamma = crate::rdm::transition_1rdm_h(x, h, x);

    let mut g = CoeffTensor::<f64>::zeros(d, 2);
    for p in 0..d {
        for q in 0..d {
            if p == q || pg.irrep_of(p) != pg.irrep_of(q) {
                continue;
            }
            g.set(&[p, q], 2.0 * (gamma.get(&[p, q]) - gamma.get(&[q, p])));
        }
    }
    g
}

/// The row-major `d×d` identity.
pub fn identity_matrix(d: usize) -> Vec<f64> {
    let mut u = vec![0.0f64; d * d];
    for i in 0..d {
        u[i * d + i] = 1.0;
    }
    u
}

/// `A·B` for row-major `d×d` flats — THE composition of accumulated rotations. The order is
/// load-bearing: `U ← U_old · U_step` (first rotation leftmost) is what makes
/// `rotate_integrals(rotate_integrals(ints, U₁), U₂) == rotate_integrals(ints, U₁·U₂)` hold, so a
/// caller chaining stages composes `compose_rotations(&u_first, &u_second, d)`.
pub fn compose_rotations(a: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    assert!(
        a.len() == d * d && b.len() == d * d,
        "compose_rotations: d×d flats"
    );
    let mut out = vec![0.0f64; d * d];
    for i in 0..d {
        for k in 0..d {
            let aik = a[i * d + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..d {
                out[i * d + j] += aik * b[k * d + j];
            }
        }
    }
    out
}

/// `max |UᵀU − I|` of a row-major `d×d` matrix — the drift gauge of an accumulated rotation.
pub fn orthogonality_defect(u: &[f64], d: usize) -> f64 {
    assert_eq!(u.len(), d * d, "orthogonality_defect: d×d flat");
    let mut worst = 0.0f64;
    for a in 0..d {
        for b in 0..d {
            let mut acc = 0.0;
            for i in 0..d {
                acc += u[i * d + a] * u[i * d + b];
            }
            let want = if a == b { 1.0 } else { 0.0 };
            worst = worst.max((acc - want).abs());
        }
    }
    worst
}

/// The row-major `d×d` Givens rotation of adjacent orbitals `(i, i+1)` by `theta`: the identity
/// except for the block `[[c, s], [−s, c]]` on rows/cols `(i, i+1)`. In [`rotate_integrals`]'s
/// convention (`u[old][new]`) the new orbitals are `b†_i = c·a†_i − s·a†_{i+1}`,
/// `b†_{i+1} = s·a†_i + c·a†_{i+1}`; `givens(d, i, 0.0)` is bitwise the identity.
pub fn givens(d: usize, i: usize, theta: f64) -> Vec<f64> {
    assert!(i + 1 < d, "givens: pair ({i}, {}) outside 0..{d}", i + 1);
    let mut u = vec![0.0f64; d * d];
    for p in 0..d {
        u[p * d + p] = 1.0;
    }
    let (s, c) = theta.sin_cos();
    u[i * d + i] = c;
    u[i * d + i + 1] = s;
    u[(i + 1) * d + i] = -s;
    u[(i + 1) * d + i + 1] = c;
    u
}

pub(crate) fn givens_cs(theta: f64) -> (f64, f64) {
    if theta == std::f64::consts::FRAC_PI_2 {
        (0.0, 1.0)
    } else if theta == -std::f64::consts::FRAC_PI_2 {
        (0.0, -1.0)
    } else {
        let (s, c) = theta.sin_cos();
        (c, s)
    }
}

/// An orthogonal `d×d` matrix as a sign diagonal followed by adjacent Givens gates:
#[derive(Clone, Debug, PartialEq)]
pub struct GivensNetwork {
    /// Number of orbitals.
    pub d: usize,
    /// The diagonal `±1` factor, one entry per orbital.
    pub signs: Vec<f64>,
    /// `(k, θ)` per gate, in composition order.
    pub gates: Vec<(usize, f64)>,
}

/// Decompose a real orthogonal row-major `d×d` matrix `u` into a [`GivensNetwork`].
pub fn givens_decompose(u: &[f64], d: usize) -> GivensNetwork {
    assert_eq!(u.len(), d * d, "givens_decompose: d×d flat");
    let defect = orthogonality_defect(u, d);
    assert!(
        defect <= 1e-12,
        "givens_decompose: u is not orthogonal (max |UᵀU − I| = {defect:.3e} > 1e-12)"
    );
    let mut w = u.to_vec();
    let mut elim: Vec<(usize, f64)> = Vec::with_capacity(d * d.saturating_sub(1) / 2);
    for r in 0..d.saturating_sub(1) {
        for j in (r..d - 1).rev() {
            let (x, y) = (w[r * d + j], w[r * d + j + 1]);
            if y == 0.0 {
                continue;
            }
            let theta = y.atan2(x);
            let (c, s) = givens_cs(theta);
            for i in 0..d {
                let (a, b) = (w[i * d + j], w[i * d + j + 1]);
                w[i * d + j] = a * c + b * s;
                w[i * d + j + 1] = b * c - a * s;
            }
            w[r * d + j + 1] = 0.0;
            elim.push((j, theta));
        }
    }
    let signs: Vec<f64> = (0..d)
        .map(|r| if w[r * d + r] < 0.0 { -1.0 } else { 1.0 })
        .collect();
    let mut resid = 0.0f64;
    for a in 0..d {
        for b in 0..d {
            let want = if a == b { signs[a] } else { 0.0 };
            resid = resid.max((w[a * d + b] - want).abs());
        }
    }
    assert!(
        resid <= 1e-10,
        "givens_decompose: eliminated residue {resid:.3e} is not a sign diagonal"
    );
    elim.reverse();
    GivensNetwork {
        d,
        signs,
        gates: elim,
    }
}

/// The exact inverse map of [`givens_decompose`]: `diag(signs)` right-multiplied by each gate in
/// list order, `O(d)` per gate. The oracle for the decomposition and the integral-side partner of
/// a network applied to a state.
pub fn givens_compose(net: &GivensNetwork) -> Vec<f64> {
    let d = net.d;
    assert_eq!(net.signs.len(), d, "givens_compose: one sign per orbital");
    let mut a = vec![0.0f64; d * d];
    for (i, &sg) in net.signs.iter().enumerate() {
        a[i * d + i] = sg;
    }
    for &(k, theta) in &net.gates {
        assert!(
            k + 1 < d,
            "givens_compose: gate ({k}, {}) outside 0..{d}",
            k + 1
        );
        let (c, s) = givens_cs(theta);
        for i in 0..d {
            let (x, y) = (a[i * d + k], a[i * d + k + 1]);
            a[i * d + k] = x * c - y * s;
            a[i * d + k + 1] = x * s + y * c;
        }
    }
    a
}

/// Apply the sign diagonal of an orbital rotation to a state in place: orbital `p` with
/// `signs[p] = −1` is `a_p → −a_p` (both spins), the phase `(−1)^{n_p}`, so the singly occupied
/// families `Up`/`Dn` of site `p` flip sign and `Empty`/`Occ` stay. Exact and site-local; the
/// orthogonality flag is kept (a `±1` family scale is an orthogonal map of the core). Every entry
/// of `signs` must be `±1`.
pub fn apply_orbital_signs<T: Scalar>(x: &mut TtVector<T>, signs: &[f64]) {
    use crate::core::Family;
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
            x.cores[p].scale_family_mut(Family::Up, minus);
            x.cores[p].scale_family_mut(Family::Dn, minus);
        }
    }
}

/// Options of [`apply_orbital_rotation`] and its SU(2) twin
/// [`crate::su2::orbital::apply_orbital_rotation`] (one scalar-only type for both layers).
/// The `Default` is the exact, uncapped walk on the ambient rayon pool.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RotateOpts {
    /// Per-split truncation tolerance, passed unchanged to the two-site `factor_svd` of every gate:
    /// the smallest pooled singular values (on SU(2) the centerweighted `√cw·σ`) are dropped while
    /// their cumulative `Σσ² ≤ eps²` — an ABSOLUTE norm rule. The walk runs on a unit-normalized
    /// copy, so with `chi = Pooled::NONE` every gate discards at most `eps²` of `‖x‖²` and
    /// `error_bound ≤ gates·eps`. A binding `chi` drops past that budget (the cap wins over
    /// `eps`), so neither a-priori bound holds then and [`RotateInfo::error_bound`] is the only
    /// bound. `0.0` drops nothing beyond what `chi` forces: an uncapped split keeps
    /// `min(rows, cols)` values per mid sector, zero singular values included. Must be finite and
    /// `≥ 0`.
    pub eps: f64,
    /// Pooled cap on each split bond's total dimension (reduced multiplets on SU(2)).
    /// [`Pooled::NONE`] = uncapped. At `eps = 0` the cap is strict; at `eps > 0` every copy of a
    /// singular value tied exactly at the cut is kept, so a split can exceed it on exact ties.
    /// Bonds no gate splits keep their input rank, which may exceed the cap.
    pub chi: Pooled,
    /// `Some(n)` (`n ≥ 1`) runs the walk inside a dedicated `n`-thread rayon pool; `None` uses
    /// the ambient pool. The result does not depend on the thread count.
    pub threads: Option<usize>,
}

/// What one [`apply_orbital_rotation`] (either layer) did and dropped. Every norm is in the layer's
/// own inner product (on SU(2) the fixed-`Sz`-member norm, which [`fn@crate::su2::expand`]
/// preserves) and is relative to `‖x‖`.
#[derive(Clone, Debug, PartialEq)]
pub struct RotateInfo {
    /// Number of Givens gates applied (`givens_decompose(u, d).gates.len()`; `0` for a sign
    /// diagonal).
    pub gates: usize,
    /// Per gate, in application order: the exact norm² removed by that gate's split, on the
    /// unit-normalized walk — `‖P‖² − ‖P′‖² = ‖P − P′‖²` for the rotated pair block `P` and its
    /// truncation `P′` (an orthogonal projection, so the two forms are equal; the residual form is
    /// the one evaluated, free of the cancellation of a difference of two `O(1)` norms). Exactly
    /// `0.0` for a split that dropped nothing but zero singular values.
    pub discarded: Vec<f64>,
    /// `Σ_g √discarded[g]`: `‖Û x − result‖ ≤ error_bound·‖x‖` up to roundoff (every gate is
    /// orthogonal and every truncation an orthogonal projection, and nothing renormalizes in
    /// between). `0.0` exactly when no split dropped anything.
    pub error_bound: f64,
    /// The largest pooled bond dimension (reduced multiplets on SU(2)) over every intermediate
    /// state of the walk, the input's bonds included (a bond changes only at the gate that splits
    /// it, so this is `max(input, every split's kept total)`). At `eps = 0` uncapped a split keeps
    /// `min(rows, cols)` per mid sector, zero singular values included.
    pub max_rank: usize,
    /// `‖result‖ / ‖x‖`. With no renormalization in the walk, `1 − norm_out² = Σ discarded` to
    /// roundoff; `1.0` exactly for a gate-free rotation.
    pub norm_out: f64,
}

impl RotateOpts {
    pub(crate) fn validate(&self, who: &str) {
        assert!(
            self.eps.is_finite() && self.eps >= 0.0,
            "{who}: eps = {} must be finite and ≥ 0",
            self.eps
        );
        assert!(
            self.threads != Some(0),
            "{who}: threads = Some(0); use None for the ambient pool or Some(n ≥ 1)"
        );
    }

    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match self.threads {
            None => f(),
            Some(n) => rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("apply_orbital_rotation: rayon pool")
                .install(f),
        }
    }
}

/// The real orthogonal orbital rotation `U ∈ O(d)` (row-major `u[old·d + new]`, the storage of
/// [`rotate_integrals`]) applied to a state: `(Û x, info)`.
///
/// # Panics
///
/// On a typed space (`!space.pg.is_trivial()`: a general `U` mixes irreps; embed the state with
/// [`TtVector::forget_point_group`] first), a non-orthogonal `u` (`givens_decompose`'s
/// `max |UᵀU − I| ≤ 1e-12` entry check), a negative or non-finite `eps`, `threads = Some(0)`, or —
/// when there is at least one gate — an input of zero or non-finite norm.
pub fn apply_orbital_rotation<T: Scalar>(
    x: &TtVector<T>,
    u: &[f64],
    opts: &RotateOpts,
) -> (TtVector<T>, RotateInfo) {
    const WHO: &str = "apply_orbital_rotation";
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

pub(crate) fn pair_target(ortho: Option<usize>, k: usize) -> Option<usize> {
    match ortho {
        Some(c) if c == k || c == k + 1 => None,
        Some(c) if c > k + 1 => Some(k + 1),
        _ => Some(k),
    }
}

fn rotation_walk<T: Scalar>(
    x: &TtVector<T>,
    net: &GivensNetwork,
    opts: &RotateOpts,
) -> (TtVector<T>, RotateInfo) {
    use crate::core::twosite::{apply_mode_rotation_cs, contract, factor_svd_noise, noise_rng};
    let mut y = x.clone();
    if let Some(p) = pair_target(y.ortho, net.gates[0].0) {
        y.move_core(p, true);
    }
    let n_in = y.norm();
    assert!(
        n_in.is_finite() && n_in > 0.0,
        "apply_orbital_rotation: input norm {n_in} must be finite and > 0"
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

/// Orbital rotation in the coefficient picture: the integrals of `e^{κ̂} H e^{−κ̂}` for
/// `U = exp(−κ)`, row-major `d×d`.
pub fn rotate_integrals<T: Scalar>(ints: &Integrals<T>, u: &[f64]) -> Integrals<T> {
    let d = ints.norb;
    assert_eq!(
        u.len(),
        d * d,
        "rotate_integrals: u must be a row-major {d}x{d} matrix"
    );
    assert!(
        ints.t.rank == 2 && ints.t.d == d,
        "rotate_integrals: t must be rank-2 over d={d}"
    );
    assert!(
        ints.v.rank == 4 && ints.v.d == d,
        "rotate_integrals: v must be rank-4 over d={d}"
    );

    let mut tw = contract_first_axis(&ints.t.data, u, d, d);
    tw = contract_first_axis(&tw, u, d, d);
    let tp = CoeffTensor {
        d,
        rank: 2,
        data: tw,
    };

    let rest = d * d * d;
    let mut vw = contract_first_axis(&ints.v.data, u, d, rest);
    for _ in 0..3 {
        vw = contract_first_axis(&vw, u, d, rest);
    }
    let vp = CoeffTensor {
        d,
        rank: 4,
        data: vw,
    };

    Integrals {
        t: tp,
        v: vp,
        e0: ints.e0,
        norb: ints.norb,
        nelec: ints.nelec,
        ms2: ints.ms2,
        orbsym: ints.orbsym.clone(),
        isym: ints.isym,
    }
}

fn contract_first_axis<T: Scalar>(x: &[T], u: &[f64], d: usize, rest: usize) -> Vec<T> {
    contract_first_axis_blocked(x, u, d, rest, ROTATE_ROW_BLOCK)
}

const ROTATE_ROW_BLOCK: usize = 256;

fn contract_first_axis_blocked<T: Scalar>(
    x: &[T],
    u: &[f64],
    d: usize,
    rest: usize,
    rows: usize,
) -> Vec<T> {
    use rayon::prelude::*;
    debug_assert_eq!(x.len(), d * rest);
    let mut y = vec![T::zero(); rest * d];
    y.par_chunks_mut(rows * d)
        .enumerate()
        .for_each(|(blk, yblk)| {
            let r0 = blk * rows;
            for i in 0..d {
                let urow = &u[i * d..(i + 1) * d];
                let xrow = &x[i * rest + r0..i * rest + r0 + yblk.len() / d];
                for (yrow, &xv) in yblk.chunks_exact_mut(d).zip(xrow) {
                    if xv == T::zero() {
                        continue;
                    }
                    for (yv, &uv) in yrow.iter_mut().zip(urow) {
                        *yv += T::from_f64(uv) * xv;
                    }
                }
            }
        });
    y
}

/// Exact matrix exponential of a real **antisymmetric** `K` (row-major `n×n` in and out).
pub fn expm_antisym(k: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(
        k.len(),
        n * n,
        "expm_antisym: k must be a row-major {n}x{n} matrix"
    );
    if n == 0 || k.iter().all(|&x| x == 0.0) {
        let mut w = vec![0.0f64; n * n];
        for p in 0..n {
            w[p * n + p] = 1.0;
        }
        return w;
    }
    assert_antisym_predicate(
        n,
        |p, q| k[p * n + q].abs(),
        |p, q| (k[p * n + q] + k[q * n + p]).abs(),
        ANTISYM_ATOL,
        "expm_antisym argument",
    );

    let mut s = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f64;
            for m in 0..n {
                acc += k[i * n + m] * k[j * n + m];
            }
            s[i * n + j] = acc;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let m = 0.5 * (s[i * n + j] + s[j * n + i]);
            s[i * n + j] = m;
            s[j * n + i] = m;
        }
    }

    let (lam, vecs) = dense_self_adjoint_smallest::<f64>(&s, n, n);

    let mut c = vec![0.0f64; n * n];
    let mut z = vec![0.0f64; n * n];
    for (col, vec_c) in vecs.iter().enumerate() {
        let theta = lam[col].max(0.0).sqrt();
        let (ct, st) = (theta.cos(), sinc(theta));
        for i in 0..n {
            let vi = vec_c[i];
            if vi == 0.0 {
                continue;
            }
            for j in 0..n {
                let vij = vi * vec_c[j];
                c[i * n + j] += ct * vij;
                z[i * n + j] += st * vij;
            }
        }
    }

    let mut out = c;
    for i in 0..n {
        for m in 0..n {
            let kim = k[i * n + m];
            if kim == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += kim * z[m * n + j];
            }
        }
    }
    out
}

#[inline]
fn sinc(theta: f64) -> f64 {
    if theta < 1e-8 {
        1.0 - theta * theta / 6.0
    } else {
        theta.sin() / theta
    }
}

/// `exp(κ)` computed **one irrep block at a time**, for a κ that is irrep-block-diagonal.
pub fn expm_kappa_blocked(kappa: &CoeffTensor<f64>, pg: PointGroup) -> Vec<f64> {
    assert_eq!(kappa.rank, 2, "expm_kappa_blocked: kappa must be rank-2");
    let d = kappa.d;
    assert_antisym(kappa);

    for p in 0..d {
        for q in 0..d {
            assert!(
                pg.irrep_of(p) == pg.irrep_of(q) || kappa.get(&[p, q]) == 0.0,
                "expm_kappa_blocked: kappa[{p},{q}] is nonzero across irreps {:?} != {:?}; \
                 build kappa on orbital_pairs(pg, d)",
                pg.irrep_of(p),
                pg.irrep_of(q)
            );
        }
    }

    let mut u = vec![0.0f64; d * d];
    for p in 0..d {
        u[p * d + p] = 1.0;
    }

    for ir in 0..pg.order() {
        let irrep = Irrep::new(ir);
        let sites: Vec<usize> = (0..d).filter(|&p| pg.irrep_of(p) == irrep).collect();
        if sites.is_empty() {
            continue;
        }
        let m = sites.len();
        let mut b = vec![0.0f64; m * m];
        for (a, &pa) in sites.iter().enumerate() {
            for (bb, &qb) in sites.iter().enumerate() {
                b[a * m + bb] = kappa.get(&[pa, qb]);
            }
        }
        let e = expm_antisym(&b, m);
        for (a, &pa) in sites.iter().enumerate() {
            for (bb, &qb) in sites.iter().enumerate() {
                u[pa * d + qb] = e[a * m + bb];
            }
        }
    }
    u
}

/// Fold the **first `n_core` orbitals** into a closed-shell frozen core, returning the active-space
/// integrals.
pub fn freeze_core<T: Scalar>(ints: &Integrals<T>, n_core: usize) -> Integrals<T> {
    let d = ints.norb;
    assert!(n_core <= d, "freeze_core: n_core={n_core} exceeds norb={d}");
    assert!(
        ints.nelec >= 2 * n_core,
        "freeze_core: nelec={} cannot accommodate {n_core} doubly-occupied core orbitals",
        ints.nelec
    );
    assert!(
        ints.t.rank == 2 && ints.t.d == d && ints.v.rank == 4 && ints.v.d == d,
        "freeze_core: integral tensors must be rank-2/rank-4 over d={d}"
    );
    assert!(
        ints.orbsym.is_empty() || ints.orbsym.len() == d,
        "freeze_core: orbsym must be empty or have one entry per orbital (got {}, norb={d})",
        ints.orbsym.len()
    );

    let two = T::from_f64(2.0);
    let d_act = d - n_core;

    let mut t = CoeffTensor::<T>::zeros(d_act, 2);
    for p in n_core..d {
        for q in n_core..d {
            let mut acc = ints.t.get(&[p, q]);
            for i in 0..n_core {
                acc += two * ints.v.get(&[p, q, i, i]) - ints.v.get(&[p, i, i, q]);
            }
            t.set(&[p - n_core, q - n_core], acc);
        }
    }

    let mut v = CoeffTensor::<T>::zeros(d_act, 4);
    for a in 0..d_act {
        for b in 0..d_act {
            for c in 0..d_act {
                for e in 0..d_act {
                    v.set(
                        &[a, b, c, e],
                        ints.v
                            .get(&[a + n_core, b + n_core, c + n_core, e + n_core]),
                    );
                }
            }
        }
    }

    let mut e_core = T::zero();
    for i in 0..n_core {
        e_core += two * ints.t.get(&[i, i]);
    }
    for i in 0..n_core {
        for j in 0..n_core {
            e_core += two * ints.v.get(&[i, i, j, j]) - ints.v.get(&[i, j, j, i]);
        }
    }

    Integrals {
        t,
        v,
        e0: ints.e0 + e_core,
        norb: d_act,
        nelec: ints.nelec - 2 * n_core,
        ms2: ints.ms2,
        orbsym: ints.orbsym[n_core.min(ints.orbsym.len())..].to_vec(),
        isym: ints.isym,
    }
}

/// A list of orbital pairs `(p, q)` with `p > q`, over **active** orbitals (i.e. after any
/// [`freeze_core`] reindexing), in the module's canonical order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairMask {
    /// The kept pairs, each with `p > q`, in canonical order.
    pub keep: Vec<(usize, usize)>,
}

impl PairMask {
    /// The unfiltered strict lower triangle over `d` orbitals, in canonical order. `d = 0` and
    /// `d = 1` both give the empty list.
    pub fn all(d: usize) -> Self {
        Self::with_predicate(d, |_, _| true)
    }

    /// The strict lower triangle filtered to the pairs `f(p, q)` accepts, order preserved. `f` is
    /// always called with `p > q`.
    pub fn with_predicate(d: usize, f: impl Fn(usize, usize) -> bool) -> Self {
        let mut keep = Vec::new();
        for p in 1..d {
            for q in 0..p {
                if f(p, q) {
                    keep.push((p, q));
                }
            }
        }
        PairMask { keep }
    }
}

/// The representable orbital-rotation pairs under a point group: [`PairMask::all`] filtered to
/// `Γ_p == Γ_q`.
pub fn orbital_pairs(pg: PointGroup, d: usize) -> Vec<(usize, usize)> {
    PairMask::with_predicate(d, |p, q| pg.irrep_of(p) == pg.irrep_of(q)).keep
}

#[cfg(test)]
mod tja_tests {
    use super::{freeze_core, kappa_apply, kappa_operator};
    use crate::coeff::CoeffTensor;
    use crate::core::tangent::{TtTangent, project_irrep};
    use crate::core::weingarten::{FloorPolicy, bond_gauges};
    use crate::integrals::{Integrals, IntegralsOptions, load_fixture_dir};
    use crate::pointgroup::{Irrep, PointGroup};
    use crate::qn::QnSpace;
    use crate::rotation::{PairMask, g_pq, orbital_pairs};
    use crate::solvers::riemannian::{normalize_mut, tangent_bases};
    use crate::test_util::randomized_state;
    use crate::vector::{TtVector, apply_group_element, symmetry_defect};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::panic::AssertUnwindSafe;
    use std::path::Path;
    use std::sync::OnceLock;

    struct PgFixture {
        d: usize,
        pg: PointGroup,
        space_c1: QnSpace,
        gamma0: Irrep,
    }

    fn pg_fixture() -> &'static PgFixture {
        static CELL: OnceLock<PgFixture> = OnceLock::new();
        CELL.get_or_init(|| {
            let d = 4usize;
            let pg = PointGroup::new(
                2,
                &[Irrep::new(0), Irrep::new(1), Irrep::new(2), Irrep::new(1)],
            );
            let space_c1 = QnSpace::new(d, 2, 1);
            PgFixture {
                d,
                pg,
                space_c1,
                gamma0: Irrep::IDENTITY,
            }
        })
    }

    fn pure_reference(x: &TtVector<f64>, pg: PointGroup, gamma0: Irrep) -> TtVector<f64> {
        let mut acc: Option<TtVector<f64>> = None;
        for h in 0..pg.order() {
            let h = Irrep::new(h);
            let term = &apply_group_element(x, pg, h) * gamma0.character(h);
            acc = Some(match acc {
                None => term,
                Some(a) => &a + &term,
            });
        }
        let mut out = acc.expect("pg.order() >= 1");
        normalize_mut(&mut out);
        out
    }

    fn sigma_min_at(base_l: &TtVector<f64>, base_r: &TtVector<f64>) -> f64 {
        bond_gauges(base_l, base_r, FloorPolicy::default_hard())
            .iter()
            .flat_map(|g| g.svals().iter().flat_map(|(_, v)| v.iter().copied()))
            .fold(f64::INFINITY, f64::min)
    }

    fn lih_sto3g_ints() -> Integrals<f64> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/lih_sto3g");
        let load =
            load_fixture_dir::<f64>(&dir, &IntegralsOptions::default()).expect("lih_sto3g loads");
        load.ints
    }

    #[test]
    fn tji_orbital_in_gamma0() {
        let fx = pg_fixture();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xE105);
        let x_rand: TtVector<f64> = randomized_state(fx.space_c1, 1, 2, &mut rng);
        let x = pure_reference(&x_rand, fx.pg, fx.gamma0);
        let (base_l, base_r) = tangent_bases(&x);
        let sigma_min = sigma_min_at(&base_l, &base_r);

        let pairs = orbital_pairs(fx.pg, fx.d);
        assert!(
            !pairs.is_empty(),
            "fixture must have at least one same-irrep pair"
        );

        for &(p, q) in &pairs {
            let u_a = &g_pq(&x, p, q) * (-1.0f64);
            let (gamma_ua, defects) = symmetry_defect(&u_a, fx.pg);
            let worst = defects.iter().cloned().fold(0.0f64, f64::max);
            println!(
                "tji_orbital_in_gamma0: pair ({p},{q}) gamma={gamma_ua:?} defect={worst:e} sigma_min={sigma_min:e}"
            );
            assert_eq!(
                gamma_ua, fx.gamma0,
                "u_a for a same-irrep pair must carry Gamma0"
            );
            assert!(worst <= 1e-8, "u_a defect too large: {worst:e}");

            let tau_a = TtTangent::project(&base_l, &base_r, &u_a);
            let tn = tau_a.norm();
            assert!(
                tn > 1e-10,
                "degenerate orbital direction for pair ({p},{q})"
            );
            for h in 0..fx.pg.order() {
                let gamma = Irrep::new(h);
                if gamma == fx.gamma0 {
                    continue;
                }
                let (proj, _leak) = project_irrep(&tau_a, &base_l, &base_r, fx.pg, gamma);
                let frac = proj.norm() / tn;
                assert!(
                    frac <= 1e-6,
                    "orbital tangent leg for pair ({p},{q}) leaked {frac:e} into Gamma={gamma:?} \
                     (sigma_min={sigma_min:e})"
                );
            }
        }
    }

    #[test]
    fn tja_a2_kappa_zero_is_termfree_and_composes() {
        let d = 5;
        let space = QnSpace::from_n_two_sz(d, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xA2A2);
        let x0: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let zero = CoeffTensor::<f64>::zeros(d, 2);

        let op0 = kappa_operator(space, &zero);
        let y = &op0 * &x0;
        assert_eq!(
            y.norm(),
            0.0,
            "term-free automaton apply must be EXACTLY zero, not merely small"
        );
        assert_eq!(
            kappa_apply(&zero, &x0).norm(),
            0.0,
            "kappa_apply(0, x) must be EXACTLY zero, not merely small"
        );

        let composed = &y + &x0;
        let diff = (composed.norm() - x0.norm()).abs();
        assert!(
            diff <= 1e-12 * x0.norm().max(1e-8),
            "composing the zero train with x0 downstream must not perturb its norm: diff={diff:e}"
        );
    }

    #[test]
    fn tja_fc_preconditions_panic() {
        let ints = lih_sto3g_ints();
        let norb = ints.norb;
        let r1 = std::panic::catch_unwind(AssertUnwindSafe(|| freeze_core(&ints, norb + 1)));
        assert!(r1.is_err(), "freeze_core must panic when n_core > norb");
        let r2 = std::panic::catch_unwind(AssertUnwindSafe(|| freeze_core(&ints, 3)));
        assert!(r2.is_err(), "freeze_core must panic when nelec < 2*n_core");
    }

    #[test]
    fn tja_pairs_with_predicate_matches_manual_filter_and_preserves_order() {
        let d = 7;
        let predicate = |p: usize, q: usize| (p + q).is_multiple_of(2);
        let got = PairMask::with_predicate(d, predicate).keep;
        let want: Vec<(usize, usize)> = (1..d)
            .flat_map(|p| (0..p).map(move |q| (p, q)))
            .filter(|&(p, q)| predicate(p, q))
            .collect();
        assert_eq!(
            got, want,
            "PairMask::with_predicate must filter the canonical triangle in order"
        );

        let all_true = PairMask::with_predicate(d, |_, _| true).keep;
        assert_eq!(
            all_true,
            PairMask::all(d).keep,
            "an always-true predicate must reproduce PairMask::all"
        );
    }

    #[test]
    fn tja_pairs_trivial_group_is_the_full_triangle() {
        for d in [0usize, 1, 4, 6] {
            let want = PairMask::all(d).keep;
            let got = orbital_pairs(PointGroup::TRIVIAL, d);
            assert_eq!(
                got, want,
                "orbital_pairs(TRIVIAL, {d}) must equal the unfiltered strict lower triangle"
            );
            let mut expected = Vec::new();
            for p in 1..d {
                for q in 0..p {
                    expected.push((p, q));
                }
            }
            assert_eq!(
                want, expected,
                "PairMask::all order must be p-outer, q-inner ascending"
            );
        }
    }
}

#[cfg(test)]
mod compose_tests {
    use super::{compose_rotations, givens, identity_matrix, orthogonality_defect};

    #[test]
    fn contract_first_axis_blocked_is_bitwise_the_serial_loop() {
        fn serial(x: &[f64], u: &[f64], d: usize, rest: usize) -> Vec<f64> {
            let mut y = vec![0.0f64; rest * d];
            for i in 0..d {
                let urow = &u[i * d..(i + 1) * d];
                for r in 0..rest {
                    let xv = x[i * rest + r];
                    if xv == 0.0 {
                        continue;
                    }
                    let yrow = &mut y[r * d..(r + 1) * d];
                    for (yv, &uv) in yrow.iter_mut().zip(urow) {
                        *yv += uv * xv;
                    }
                }
            }
            y
        }
        let d = 7;
        let rest = d * d * d;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let u: Vec<f64> = (0..d * d).map(|_| next()).collect();
        let x: Vec<f64> = (0..d * rest)
            .map(|k| if k % 5 == 0 { 0.0 } else { next() })
            .collect();
        let want = serial(&x, &u, d, rest);
        for rows in [super::ROTATE_ROW_BLOCK, 1, 5] {
            let got = super::contract_first_axis_blocked(&x, &u, d, rest, rows);
            assert!(
                got.iter()
                    .zip(&want)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "rows = {rows}: not bitwise the serial loop"
            );
        }
    }

    #[test]
    fn compose_is_a_times_b_and_keeps_structure() {
        let d = 4;
        let (a, b) = (givens(d, 0, 0.3), givens(d, 1, -0.5));
        let ab = compose_rotations(&a, &b, d);
        let ba = compose_rotations(&b, &a, d);
        let mut want = vec![0.0f64; d * d];
        for i in 0..d {
            for j in 0..d {
                for k in 0..d {
                    want[i * d + j] += a[i * d + k] * b[k * d + j];
                }
            }
        }
        let diff = |x: &[f64], y: &[f64]| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0, f64::max)
        };
        assert!(diff(&ab, &want) < 1e-16);
        assert!(
            diff(&ab, &ba) > 1e-2,
            "the two pairs share orbital 1: they must not commute"
        );
        assert!(orthogonality_defect(&ab, d) < 1e-15);
        assert_eq!(givens(d, 2, 0.0), identity_matrix(d));
        for k in 0..3 {
            assert_eq!(ab[3 * d + k].to_bits(), 0.0f64.to_bits());
            assert_eq!(ab[k * d + 3].to_bits(), 0.0f64.to_bits());
        }
        assert_eq!(ab[3 * d + 3], 1.0);
    }
}

#[cfg(test)]
mod network_tests {
    use super::{
        GivensNetwork, apply_orbital_signs, expm_antisym, givens_compose, givens_decompose,
        identity_matrix,
    };
    use crate::qn::QnSpace;
    use crate::su2::qn::Su2Space;
    use crate::test_util::{randomized_state, randomized_su2_state, seeded};
    use rand::RngExt;
    use std::f64::consts::FRAC_PI_2;

    fn random_orthogonal(d: usize, flip: bool, seed: u64) -> Vec<f64> {
        let mut rng = seeded(seed);
        let mut k = vec![0.0f64; d * d];
        for i in 0..d {
            for j in 0..i {
                let v: f64 = rng.random_range(-2.0..2.0);
                k[i * d + j] = v;
                k[j * d + i] = -v;
            }
        }
        let mut u = expm_antisym(&k, d);
        if flip {
            for i in 0..d {
                u[i * d + d / 2] = -u[i * d + d / 2];
            }
        }
        u
    }

    fn max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn t1_round_trip_random_orthogonal_both_determinants() {
        for d in [1usize, 2, 3, 5, 8, 13] {
            for flip in [false, true] {
                if d == 1 && flip {
                    continue;
                }
                let u = random_orthogonal(d, flip, 17 + d as u64);
                let net = givens_decompose(&u, d);
                assert!(
                    net.gates.len() <= d * (d - 1) / 2,
                    "d = {d}: too many gates"
                );
                assert!(net.gates.iter().all(|&(k, th)| k + 1 < d && th != 0.0));
                let det: f64 = net.signs.iter().product();
                assert_eq!(det, if flip { -1.0 } else { 1.0 }, "d = {d}: det sign");
                let err = max_diff(&givens_compose(&net), &u);
                assert!(
                    err <= 1e-14 * d as f64,
                    "d = {d}, flip = {flip}: err {err:.3e}"
                );
            }
        }
    }

    #[test]
    fn t1_identity_and_signed_permutations_are_exact() {
        let d = 6;
        let id = identity_matrix(d);
        let net = givens_decompose(&id, d);
        assert!(net.gates.is_empty());
        assert!(net.signs.iter().all(|&s| s == 1.0));
        let back = givens_compose(&net);
        assert!(
            back.iter()
                .zip(&id)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );

        for k in 0..d - 1 {
            let sw = crate::ordering::swap_matrix(d, k);
            let net = givens_decompose(&sw, d);
            assert_eq!(net.gates, vec![(k, FRAC_PI_2)], "swap_matrix({k})");
            assert!(net.signs.iter().all(|&s| s == 1.0));
            assert_eq!(givens_compose(&net), sw);
        }

        let perm = [3usize, 0, 5, 1, 4, 2];
        let sgn = [1.0, -1.0, -1.0, 1.0, -1.0, 1.0];
        let mut p = vec![0.0f64; d * d];
        for r in 0..d {
            p[r * d + perm[r]] = sgn[r];
        }
        let net = givens_decompose(&p, d);
        assert!(
            net.gates
                .iter()
                .all(|&(_, th)| th == FRAC_PI_2 || th == -FRAC_PI_2)
        );
        assert_eq!(givens_compose(&net), p, "signed permutation not exact");
    }

    #[test]
    fn t1_block_diagonal_gates_stay_inside_blocks() {
        let d = 7;
        let (b1, b2) = (
            random_orthogonal(3, true, 5),
            random_orthogonal(4, false, 6),
        );
        let mut u = vec![0.0f64; d * d];
        for i in 0..3 {
            for j in 0..3 {
                u[i * d + j] = b1[i * 3 + j];
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                u[(i + 3) * d + j + 3] = b2[i * 4 + j];
            }
        }
        let net = givens_decompose(&u, d);
        assert!(!net.gates.is_empty());
        assert!(
            net.gates.iter().all(|&(k, _)| k != 2),
            "a gate straddles the 3|4 block boundary: {:?}",
            net.gates
        );
        assert!(max_diff(&givens_compose(&net), &u) <= 1e-14 * d as f64);
    }

    #[test]
    #[should_panic(expected = "not orthogonal")]
    fn t1_non_orthogonal_input_panics() {
        let mut u = identity_matrix(3);
        u[1] = 1e-9;
        let _ = givens_decompose(&u, 3);
    }

    #[test]
    fn t1_compose_is_the_left_to_right_product() {
        let d = 4;
        let net = GivensNetwork {
            d,
            signs: vec![1.0, -1.0, 1.0, 1.0],
            gates: vec![(0, 0.3), (2, -1.1), (1, 0.7)],
        };
        let mut want = identity_matrix(d);
        want[d + 1] = -1.0;
        for &(k, th) in &net.gates {
            want = super::compose_rotations(&want, &super::givens(d, k, th), d);
        }
        assert!(max_diff(&givens_compose(&net), &want) <= 1e-15);
    }

    #[test]
    fn signs_are_the_occupation_parity_both_layers() {
        let mut rng = seeded(41);
        let space = QnSpace::new(4, 2, 1);
        let mut x = randomized_state::<f64, _>(space, 1, 3, &mut rng);
        x.left_orthogonalize(true);
        let ortho = x.ortho;
        let dense = x.to_dense();
        let signs = [1.0, -1.0, 1.0, -1.0];
        let mut y = x.clone();
        apply_orbital_signs(&mut y, &signs);
        assert_eq!(y.ortho, ortho);
        let got = y.to_dense();
        for (idx, (&a, &b)) in dense.iter().zip(&got).enumerate() {
            let mut phase = 1.0;
            for (p, &sg) in signs.iter().enumerate() {
                let loc = (idx >> (2 * p)) & 3;
                if sg == -1.0 && (loc == 1 || loc == 2) {
                    phase = -phase;
                }
            }
            assert_eq!(b, phase * a, "dense index {idx}");
        }
        apply_orbital_signs(&mut y, &signs);
        assert_eq!(y.to_dense(), dense, "not an involution");

        for two_s in [0usize, 2] {
            let sp = Su2Space::new(4, 4, two_s);
            let xs = randomized_su2_state::<f64, _>(sp, 1, 2, &mut rng);
            let mut ys = xs.clone();
            crate::su2::orbital::apply_orbital_signs(&mut ys, &signs);
            let lhs = crate::su2::expand(&ys, two_s as i64).to_dense();
            let mut ab = crate::su2::expand(&xs, two_s as i64);
            apply_orbital_signs(&mut ab, &signs);
            let rhs = ab.to_dense();
            assert!(max_diff(&lhs, &rhs) <= 1e-14, "2S = {two_s}");
            assert!(lhs.iter().any(|v| v.abs() > 1e-3), "2S = {two_s}: vacuous");
        }
    }
}
