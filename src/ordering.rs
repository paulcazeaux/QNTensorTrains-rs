// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Orbital orderings: the graph-Laplacian (Fiedler) ordering of an orbital affinity matrix, the
//! exchange-integral affinity, and the permutation plumbing that relabels integrals.

use std::collections::VecDeque;

use crate::coeff::CoeffTensor;
use crate::core::factor::dense_self_adjoint_smallest;
use crate::core::twosite::{apply_mode_swap, contract, factor_svd_noise, noise_rng};
use crate::integrals::Integrals;
use crate::pointgroup::PointGroup;
use crate::qn::Pooled;
use crate::rotation::{compose_rotations, identity_matrix};
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// One connected component of size `≥ 2` ordered by [`fiedler_order`], in concatenation order.
#[derive(Clone, Debug)]
pub struct FiedlerComponent {
    /// The number of orbitals in the component.
    pub size: usize,
    /// The Fiedler value `λ₂` of the component's Laplacian (its algebraic connectivity).
    pub lambda2: f64,
    /// `λ₃`, or NaN when `size < 3`.
    pub lambda3: f64,
    /// `λ₃ − λ₂ ≤ 1e-8·λ_max`: the Fiedler vector is not unique and the order is roundoff-dependent.
    /// Always `false` when `size < 3`.
    pub degenerate: bool,
}

/// The result of [`fiedler_order`].
#[derive(Clone, Debug)]
pub struct FiedlerOrder {
    /// `perm[new] = old`, a permutation of `0..d`.
    pub perm: Vec<usize>,
    /// Every component of size `≥ 2`, in the order its block appears in `perm`. Singletons are not
    /// listed.
    pub components: Vec<FiedlerComponent>,
}

/// The Fiedler ordering of the orbital graph with row-major `d×d` affinity `w`.
pub fn fiedler_order(w: &[f64], d: usize) -> FiedlerOrder {
    let ws = validated_affinity(w, d);
    let comps = connected_components(&ws, d);
    let mut perm = Vec::with_capacity(d);
    let mut components = Vec::new();
    for mem in comps {
        let m = mem.len();
        if m == 1 {
            perm.push(mem[0]);
            continue;
        }
        let mut lap = vec![0.0f64; m * m];
        for a in 0..m {
            let mut deg = 0.0f64;
            for b in 0..m {
                if b != a {
                    let wab = ws[mem[a] * d + mem[b]];
                    lap[a * m + b] = -wab;
                    deg += wab;
                }
            }
            lap[a * m + a] = deg;
        }
        let (evals, evecs) = dense_self_adjoint_smallest::<f64>(&lap, m, m.min(3));
        let lambda2 = evals[1];
        let (lambda3, degenerate) = if m >= 3 {
            (evals[2], evals[2] - lambda2 <= 1e-8 * evals[m - 1])
        } else {
            (f64::NAN, false)
        };
        perm.extend(orient_and_sort(&mem, &evecs[1]));
        components.push(FiedlerComponent {
            size: m,
            lambda2,
            lambda3,
            degenerate,
        });
    }
    assert_permutation(&perm, d);
    FiedlerOrder { perm, components }
}

fn validated_affinity(w: &[f64], d: usize) -> Vec<f64> {
    assert!(
        w.len() == d * d,
        "fiedler_order: w must be a row-major {d}x{d} matrix (got {} entries)",
        w.len()
    );
    let mut wmax = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            if i == j {
                continue;
            }
            let x = w[i * d + j];
            assert!(
                x.is_finite(),
                "fiedler_order: w[{i},{j}] = {x} is not finite"
            );
            wmax = wmax.max(x.abs());
        }
    }
    let tol = 1e-12 * wmax;
    for i in 0..d {
        for j in 0..d {
            let x = w[i * d + j];
            assert!(
                i == j || x >= -tol,
                "fiedler_order: w[{i},{j}] = {x} is negative (below the roundoff band \
                 −1e-12·max|w| = {:e})",
                -tol
            );
        }
    }
    let mut ws = vec![0.0f64; d * d];
    for i in 0..d {
        for j in (i + 1)..d {
            let (a, b) = (w[i * d + j], w[j * d + i]);
            let delta = (a - b).abs();
            assert!(
                delta <= tol,
                "fiedler_order: w is not symmetric at ({i}, {j}): w[{i},{j}] = {a}, \
                 w[{j},{i}] = {b}, |Δ| = {delta:e} > 1e-12·max|w| = {tol:e}"
            );
            let (a, b) = (a.max(0.0), b.max(0.0));
            let s = a + 0.5 * (b - a);
            ws[i * d + j] = s;
            ws[j * d + i] = s;
        }
    }
    ws
}

fn connected_components(ws: &[f64], d: usize) -> Vec<Vec<usize>> {
    let mut seen = vec![false; d];
    let mut comps = Vec::new();
    for s in 0..d {
        if seen[s] {
            continue;
        }
        seen[s] = true;
        let mut queue = VecDeque::from([s]);
        let mut mem = Vec::new();
        while let Some(v) = queue.pop_front() {
            mem.push(v);
            for u in 0..d {
                if u != v && !seen[u] && ws[v * d + u] > 0.0 {
                    seen[u] = true;
                    queue.push_back(u);
                }
            }
        }
        mem.sort_unstable();
        comps.push(mem);
    }
    comps
}

fn orient_and_sort(members: &[usize], f: &[f64]) -> Vec<usize> {
    assert_eq!(
        members.len(),
        f.len(),
        "orient_and_sort: one Fiedler entry per member"
    );
    let mut kmax = 0;
    for (k, x) in f.iter().enumerate().skip(1) {
        if x.abs() > f[kmax].abs() {
            kmax = k;
        }
    }
    let flip = f[kmax] < 0.0;
    let g: Vec<f64> = f.iter().map(|&x| if flip { -x } else { x }).collect();
    let mut order: Vec<usize> = (0..f.len()).collect();
    order.sort_unstable_by(|&a, &b| g[a].total_cmp(&g[b]).then(members[a].cmp(&members[b])));
    order.into_iter().map(|k| members[k]).collect()
}

/// The exchange-integral affinity `w[i·d + j] = |(ij|ji)| = |v[i,j,j,i]|` for `i ≠ j`, `0.0` on
/// the diagonal; `d = ints.norb`, row-major `d×d`.
pub fn exchange_affinity<T: Scalar>(ints: &Integrals<T>) -> Vec<f64> {
    let d = ints.norb;
    assert!(
        ints.v.rank == 4 && ints.v.d == d,
        "exchange_affinity: v must be rank-4 over d={d}"
    );
    let mut w = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..d {
            if i != j {
                w[i * d + j] = ints.v.get(&[i, j, j, i]).abs_s();
            }
        }
    }
    w
}

/// Refuse anything that is not a permutation of `0..d`: a wrong length, an entry `≥ d`, or a
/// duplicate. One ascending scan, so the panic names the first offending POSITION `k`
/// (`"perm[k] = p is out of range 0..d"`, `"perm[k] = p duplicates perm[k0]"`).
pub fn assert_permutation(perm: &[usize], d: usize) {
    assert!(
        perm.len() == d,
        "assert_permutation: perm has length {}, expected d = {d}",
        perm.len()
    );
    let mut first: Vec<Option<usize>> = vec![None; d];
    for (k, &p) in perm.iter().enumerate() {
        assert!(
            p < d,
            "assert_permutation: perm[{k}] = {p} is out of range 0..{d}"
        );
        if let Some(k0) = first[p] {
            panic!("assert_permutation: perm[{k}] = {p} duplicates perm[{k0}]");
        }
        first[p] = Some(k);
    }
}

/// Relabel the orbitals of `ints` by `perm` (`perm[new] = old`, `d = ints.norb`): a pure value
/// copy, bitwise.
pub fn permute_integrals<T: Scalar>(ints: &Integrals<T>, perm: &[usize]) -> Integrals<T> {
    let d = ints.norb;
    assert_permutation(perm, d);
    assert!(
        ints.t.rank == 2 && ints.t.d == d,
        "permute_integrals: t must be rank-2 over d={d}"
    );
    assert!(
        ints.v.rank == 4 && ints.v.d == d,
        "permute_integrals: v must be rank-4 over d={d}"
    );
    assert!(
        ints.orbsym.is_empty() || ints.orbsym.len() == d,
        "permute_integrals: orbsym has {} entries, expected 0 or d = {d}",
        ints.orbsym.len()
    );
    let mut t = Vec::with_capacity(d * d);
    for &pa in perm {
        for &pb in perm {
            t.push(ints.t.data[pa * d + pb]);
        }
    }
    let mut v = Vec::with_capacity(d * d * d * d);
    for &pa in perm {
        for &pb in perm {
            for &pc in perm {
                for &pe in perm {
                    v.push(ints.v.data[((pa * d + pb) * d + pc) * d + pe]);
                }
            }
        }
    }
    let orbsym = if ints.orbsym.is_empty() {
        Vec::new()
    } else {
        perm.iter().map(|&p| ints.orbsym[p]).collect()
    };
    Integrals {
        t: CoeffTensor {
            d,
            rank: 2,
            data: t,
        },
        v: CoeffTensor {
            d,
            rank: 4,
            data: v,
        },
        e0: ints.e0,
        norb: ints.norb,
        nelec: ints.nelec,
        ms2: ints.ms2,
        orbsym,
        isym: ints.isym,
    }
}

/// The IRREP-BLOCKED site order of the table `pg` over `d` sites: a STABLE sort of `0..d` by irrep
/// XOR index ascending, so every irrep's sites are contiguous and keep their incoming relative
/// order. `perm[new] = old` (module convention).
pub fn irrep_blocked_order(pg: PointGroup, d: usize) -> Vec<usize> {
    assert!(
        d <= PointGroup::MAX_SITES,
        "irrep_blocked_order: d = {d} exceeds the table width {}",
        PointGroup::MAX_SITES
    );
    let mut perm: Vec<usize> = (0..d).collect();
    perm.sort_by_key(|&k| pg.irrep_of(k).index());
    perm
}

/// The permutation as a row-major `d×d` orbital rotation in the `u[old·d + new]` convention of
/// `rotate_integrals` (`d = perm.len()`): `u[old·d + new] = 1.0` iff `old == perm[new]`, else
/// `0.0`. At `|G| = 1`, `rotate_integrals(ints, &permutation_matrix(perm))` equals
/// [`permute_integrals`]`(ints, perm)` cell-wise under f64 `==` (but it leaves `orbsym` stale).
pub fn permutation_matrix(perm: &[usize]) -> Vec<f64> {
    let d = perm.len();
    assert_permutation(perm, d);
    let mut u = vec![0.0f64; d * d];
    for (new, &old) in perm.iter().enumerate() {
        u[old * d + new] = 1.0;
    }
    u
}

/// The inverse permutation: `inverse_permutation(perm)[perm[k]] = k`, so relabelling by `perm`
/// and then by its inverse is the identity.
pub fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    assert_permutation(perm, perm.len());
    let mut inv = vec![0usize; perm.len()];
    for (k, &p) in perm.iter().enumerate() {
        inv[p] = k;
    }
    inv
}

/// The pairs of layer `layer` of the odd–even transposition network on `d` sites, as their LEFT
/// sites `k` (the pair `(k, k+1)`, split at bond `k + 1`), in ascending order.
pub fn swap_network_layer(d: usize, layer: usize) -> Vec<usize> {
    assert!(
        d.is_multiple_of(2),
        "swap_network_layer: d = {d} is odd (the odd–even network is defined for even d)"
    );
    assert!(d >= 2, "swap_network_layer: d = {d} < 2 has no pair");
    (layer % 2..d - 1).step_by(2).collect()
}

/// The exact swap of orbitals `(k, k+1)` as a row-major `d×d` rotation in the `u[old·d + new]`
/// convention of [`crate::rotation::rotate_integrals`]: `u[k][k+1] = +1`, `u[k+1][k] = −1`, zeros
/// at `u[k][k]` and `u[k+1][k+1]`, the identity elsewhere.
pub fn swap_matrix(d: usize, k: usize) -> Vec<f64> {
    assert!(
        k + 1 < d,
        "swap_matrix: pair ({k}, {}) is out of range for d = {d}",
        k + 1
    );
    let mut u = identity_matrix(d);
    u[k * d + k] = 0.0;
    u[k * d + k + 1] = 1.0;
    u[(k + 1) * d + k] = -1.0;
    u[(k + 1) * d + k + 1] = 0.0;
    u
}

/// What one [`apply_swap_layer`] call truncated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwapLayerInfo {
    /// `1 − Π_j (1 − w_j)` over the layer's splits, where `w_j = Σ_dropped σ² / Σ σ²` is split
    /// `j`'s relative discarded weight. For splits taken left to right this is the exact fidelity
    /// loss of the layer, `1 − |⟨ŷ_exact, ŷ⟩|²`. It is `0.0` exactly when nothing was dropped.
    pub discarded: f64,
    /// The largest pooled dimension kept at any of the layer's split bonds (`0` for an empty
    /// layer).
    pub max_bond_dim: usize,
}

/// Apply one layer of exact adjacent swaps to `x` in place, and return the layer's orbital
/// rotation `u` (`d×d`, `u[old·d + new]`) together with what the truncation dropped.
pub fn apply_swap_layer<T: Scalar>(
    x: &mut TtVector<T>,
    bonds: &[usize],
    chi: Pooled,
    eps: f64,
) -> (Vec<f64>, SwapLayerInfo) {
    let d = x.space.d;
    for (j, &k) in bonds.iter().enumerate() {
        if j > 0 {
            let prev = bonds[j - 1];
            assert!(
                k >= prev + 2,
                "apply_swap_layer: bonds[{j}] = {k} must be ≥ bonds[{}] + 2 = {} (pairs strictly \
                 ascending and disjoint)",
                j - 1,
                prev + 2
            );
        }
        assert!(
            k + 1 < d,
            "apply_swap_layer: bonds[{j}] = {k}: pair ({k}, {}) is out of range for d = {d}",
            k + 1
        );
    }
    assert!(
        eps.is_finite() && eps >= 0.0,
        "apply_swap_layer: eps = {eps} must be finite and ≥ 0"
    );
    let empty = SwapLayerInfo {
        discarded: 0.0,
        max_bond_dim: 0,
    };
    let Some(&first) = bonds.first() else {
        return (identity_matrix(d), empty);
    };
    x.move_core(first, true);
    let n_in = x.norm();
    assert!(
        n_in.is_finite() && n_in > 0.0,
        "apply_swap_layer: input norm {n_in} must be finite and > 0"
    );
    let mut u = identity_matrix(d);
    let mut info = empty;
    for &k in bonds {
        x.move_core(k, true);
        let cp = apply_mode_swap(&contract(x.core(k), x.core(k + 1)));
        let (xk, s, mut xk1, trunc) =
            factor_svd_noise(&cp, eps, chi, false, 0.0, &mut noise_rng(0));
        xk1.fold_sketch_left_mut(&s);
        let n_kept = xk1.norm();
        xk1.scale_mut(T::from_f64(n_in / n_kept));
        x.set_cores(xk, xk1);
        x.ortho = Some(k + 1);
        info.discarded += trunc.discarded_weight * (1.0 - info.discarded);
        info.max_bond_dim = info.max_bond_dim.max(trunc.kept);
        u = compose_rotations(&u, &swap_matrix(d, k), d);
    }
    (u, info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orient_and_sort_is_sign_invariant_with_pinned_ties() {
        let neg = |f: &[f64]| f.iter().map(|x| -x).collect::<Vec<f64>>();
        let cases: [(&[usize], &[f64], &[usize]); 6] = [
            (&[0, 1, 2], &[0.5, -0.5, 0.1], &[1, 2, 0]),
            (&[0, 1, 2], &[0.3, 0.3, -1.0], &[0, 1, 2]),
            (&[0, 1, 2, 3], &[0.2, -0.7, 0.4, 0.7], &[3, 2, 0, 1]),
            (&[1, 4, 6], &[0.3, 0.3, -1.0], &[1, 4, 6]),
            (&[2, 5, 9], &[-0.6, 0.1, 0.2], &[9, 5, 2]),
            (&[0, 1, 2], &[0.0, -0.0, 1.0], &[1, 0, 2]),
        ];
        for (members, f, want) in cases {
            let got = orient_and_sort(members, f);
            assert_eq!(got, want.to_vec(), "orient_and_sort({members:?}, {f:?})");
            assert_eq!(
                orient_and_sort(members, &neg(f)),
                got,
                "f → −f changed the order for {f:?}"
            );
        }
        let f = [0.1, -0.2, 0.9, -0.4, 0.3];
        let got = orient_and_sort(&[0, 1, 2, 3, 4], &f);
        assert_eq!(*got.last().unwrap(), 2);
    }

    use crate::pointgroup::{Irrep, PointGroup, PointGroupKind};

    fn table(nbits: u8, xor: &[u8]) -> PointGroup {
        let irs: Vec<Irrep> = xor.iter().map(|&i| Irrep::new(i)).collect();
        PointGroup::new(nbits, &irs)
    }

    fn keys(pg: PointGroup, perm: &[usize]) -> Vec<u8> {
        perm.iter().map(|&o| pg.irrep_of(o).index()).collect()
    }

    fn filtered(pg: PointGroup, seq: &[usize], r: u8) -> Vec<usize> {
        seq.iter()
            .copied()
            .filter(|&o| pg.irrep_of(o).index() == r)
            .collect()
    }

    #[test]
    fn irrep_blocked_order_is_a_stable_xor_sort() {
        let pg = table(2, &[3, 1, 0, 2, 0, 3, 1]);
        let p = irrep_blocked_order(pg, 7);
        assert_permutation(&p, 7);
        assert_eq!(p, vec![2, 4, 1, 6, 3, 0, 5], "stable XOR-ascending sort");
        let k = keys(pg, &p);
        assert!(k.windows(2).all(|w| w[0] <= w[1]), "keys {k:?} not sorted");
        for r in 0..4u8 {
            let blk = filtered(pg, &p, r);
            assert!(
                blk.windows(2).all(|w| w[0] < w[1]),
                "irrep {r}: incoming order not kept: {blk:?}"
            );
        }

        let c2v = PointGroup::from_orbsym(PointGroupKind::C2v, &[2, 4, 1, 4, 2, 1]).unwrap();
        assert_eq!(
            keys(c2v, &[0, 1, 2, 3, 4, 5]),
            vec![2, 1, 0, 1, 2, 0],
            "C2v Molpro → XOR is [A1, B1, B2, A2] → [0, 2, 3, 1]"
        );
        let q = irrep_blocked_order(c2v, 6);
        assert_eq!(q, vec![2, 5, 1, 3, 0, 4], "A1, A2, B1 — XOR order");
        assert_ne!(
            q,
            vec![2, 5, 0, 4, 1, 3],
            "a Molpro-id sort (A1, B1, A2) is the wrong key"
        );
    }

    #[test]
    fn irrep_blocked_order_is_the_identity_on_trivial_and_blocked_tables() {
        for d in [0usize, 1, 5] {
            assert_eq!(
                irrep_blocked_order(PointGroup::TRIVIAL, d),
                (0..d).collect::<Vec<_>>(),
                "trivial group, d = {d}"
            );
        }
        let blocked = table(2, &[0, 0, 1, 2, 2, 3]);
        assert_eq!(irrep_blocked_order(blocked, 6), (0..6).collect::<Vec<_>>());

        let pg = table(3, &[5, 0, 7, 0, 5, 2, 7, 1]);
        let p = irrep_blocked_order(pg, 8);
        let relabelled: Vec<u8> = keys(pg, &p);
        let pg2 = table(3, &relabelled);
        assert_eq!(
            irrep_blocked_order(pg2, 8),
            (0..8).collect::<Vec<_>>(),
            "blocking the blocked table must be the identity"
        );
    }

    #[test]
    fn irrep_blocked_order_composes_after_an_incoming_order() {
        let pg = table(2, &[0, 1, 0, 1, 0, 1, 2, 0]);
        let d = 8;
        let mut w = vec![0.0f64; d * d];
        for a in 0..d - 1 {
            let x = 1.0 + 0.37 * a as f64 + 0.05 * (a * a) as f64;
            w[a * d + a + 1] = x;
            w[(a + 1) * d + a] = x;
        }
        let fo = fiedler_order(&w, d).perm;
        let scrambled = vec![6usize, 4, 0, 7, 1, 2, 5, 3];
        assert_ne!(
            filtered(pg, &scrambled, 0),
            {
                let mut s = filtered(pg, &scrambled, 0);
                s.sort_unstable();
                s
            },
            "the scrambled arm must be non-monotone within irrep 0"
        );
        for (name, p0) in [("scrambled", scrambled), ("fiedler", fo)] {
            assert_permutation(&p0, d);
            let pg0 = table(2, &keys(pg, &p0));
            let pb = irrep_blocked_order(pg0, d);
            assert_permutation(&pb, d);
            let tot: Vec<usize> = (0..d).map(|a| p0[pb[a]]).collect();
            let k = keys(pg, &tot);
            assert!(
                k.windows(2).all(|w| w[0] <= w[1]),
                "{name}: blocks not contiguous: {k:?}"
            );
            for r in 0..4u8 {
                assert_eq!(
                    filtered(pg, &tot, r),
                    filtered(pg, &p0, r),
                    "{name}: irrep {r} lost the incoming order"
                );
            }
        }
    }
}
