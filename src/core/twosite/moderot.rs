// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;

use faer::Mat;
use rayon::prelude::*;

use super::ContractedSparseCores;
use crate::core::Family;
use crate::core::factor::svals_qr;
use crate::entropy::{BondFunctional, bond_functional};
use crate::qn::Sector;
use crate::scalar::Scalar;

#[inline]
pub(crate) fn chan(fi: Family, fj: Family) -> usize {
    4 * (fi as usize) + fj as usize
}

/// The 16×16 action of `exp(θ·Ĝ_{k+1,k})` on the two-site occupation basis, `g[out][in]`, indexed
/// by [`chan`]. `gate16(0.0)` is bitwise the identity (`+0.0` off the diagonal), and every entry
/// outside the nine `(N↑, N↓)` groups is exactly `0.0`. The table itself is [`gate16_cs`] at
/// `(c, s) = (cos θ, sin θ)`.
pub fn gate16(theta: f64) -> [[f64; 16]; 16] {
    if theta == 0.0 {
        return identity16();
    }
    let (s, c) = theta.sin_cos();
    gate16_cs(c, s)
}

fn identity16() -> [[f64; 16]; 16] {
    let mut g = [[0.0f64; 16]; 16];
    for (i, row) in g.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    g
}

pub(crate) fn gate16_cs(c: f64, s: f64) -> [[f64; 16]; 16] {
    debug_assert!(
        (c * c + s * s - 1.0).abs() <= 4.0 * f64::EPSILON,
        "gate16_cs: c² + s² = {} ≠ 1",
        c * c + s * s
    );
    let mut g = identity16();
    let ch = |a: Family, b: Family| chan(a, b);
    use Family::{Dn, Empty, Occ, Up};
    for sp in [Up, Dn] {
        let (a, b) = (ch(sp, Empty), ch(Empty, sp));
        g[a][a] = c;
        g[a][b] = -s;
        g[b][a] = s;
        g[b][b] = c;
    }
    for sp in [Up, Dn] {
        let (a, b) = (ch(Occ, sp), ch(sp, Occ));
        g[a][a] = c;
        g[a][b] = s;
        g[b][a] = -s;
        g[b][b] = c;
    }
    let idx = [ch(Occ, Empty), ch(Up, Dn), ch(Dn, Up), ch(Empty, Occ)];
    let (cc, ss, cs) = (c * c, s * s, c * s);
    let m = [
        [cc, -cs, cs, ss],
        [cs, cc, ss, -cs],
        [-cs, ss, cc, cs],
        [ss, cs, -cs, cc],
    ];
    for (a, &ia) in idx.iter().enumerate() {
        for (b, &ib) in idx.iter().enumerate() {
            g[ia][ib] = m[a][b];
        }
    }
    g
}

#[derive(Clone, Copy)]
struct Tile {
    fam: Family,
    sec: Sector,
    off: usize,
    len: usize,
}

fn tilings<T: Scalar>(tc: &ContractedSparseCores<T>) -> HashMap<Sector, (Vec<Tile>, Vec<Tile>)> {
    let space = tc.space;
    let k = tc.k;
    let gl = space.pg.irrep_of(k);
    let gr = space.pg.irrep_of(k + 1);
    let mut out = HashMap::new();
    for &m in &tc.mid_qn {
        if tc.block(m).is_none() {
            continue;
        }
        let mut rows = Vec::with_capacity(4);
        let mut roff = 0usize;
        for fi in Family::ALL {
            let Some(l) = fi.unapply(m, gl) else { continue };
            if !space.contains(k, l) {
                continue;
            }
            let len = tc.row_ranks[l];
            rows.push(Tile {
                fam: fi,
                sec: l,
                off: roff,
                len,
            });
            roff += len;
        }
        let mut cols = Vec::with_capacity(4);
        let mut coff = 0usize;
        for fj in Family::ALL {
            let r = fj.apply(m, gr);
            if !space.contains(k + 2, r) {
                continue;
            }
            let len = tc.col_ranks[r];
            cols.push(Tile {
                fam: fj,
                sec: r,
                off: coff,
                len,
            });
            coff += len;
        }
        out.insert(m, (rows, cols));
    }
    out
}

fn rotated_block<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    g: &[[f64; 16]; 16],
    tiles: &HashMap<Sector, (Vec<Tile>, Vec<Tile>)>,
    m: Sector,
) -> Option<Mat<T>> {
    let src = tc.block(m)?;
    let (rows, cols) = tiles.get(&m)?;
    let gl = tc.space.pg.irrep_of(tc.k);
    let mut out = Mat::<T>::zeros(src.nrows(), src.ncols());
    for rt in rows {
        if rt.len == 0 {
            continue;
        }
        for ct in cols {
            if ct.len == 0 {
                continue;
            }
            let o = chan(rt.fam, ct.fam);
            for fi2 in Family::ALL {
                for fj2 in Family::ALL {
                    let coef = g[o][chan(fi2, fj2)];
                    if coef == 0.0 {
                        continue;
                    }
                    let m2 = fi2.apply(rt.sec, gl);
                    let (Some(sb), Some((rows2, cols2))) = (tc.block(m2), tiles.get(&m2)) else {
                        continue;
                    };
                    let (Some(r2), Some(c2)) = (
                        rows2.iter().find(|t| t.fam == fi2 && t.sec == rt.sec),
                        cols2.iter().find(|t| t.fam == fj2 && t.sec == ct.sec),
                    ) else {
                        continue;
                    };
                    debug_assert_eq!((r2.len, c2.len), (rt.len, ct.len));
                    let a = T::from_f64(coef);
                    for cc in 0..ct.len {
                        for rr in 0..rt.len {
                            let v = out[(rt.off + rr, ct.off + cc)]
                                + a * sb[(r2.off + rr, c2.off + cc)];
                            out[(rt.off + rr, ct.off + cc)] = v;
                        }
                    }
                }
            }
        }
    }
    Some(out)
}

fn assert_representable<T: Scalar>(tc: &ContractedSparseCores<T>) {
    let pg = tc.space.pg;
    assert!(
        pg.irrep_of(tc.k) == pg.irrep_of(tc.k + 1),
        "mode rotation of orbitals ({}, {}): different irreps — a cross-irrep rotation is not \
         representable over this space",
        tc.k,
        tc.k + 1
    );
}

/// `G(θ)` applied to a two-site block (out of place; the container shape is θ-invariant).
/// `theta == 0.0` returns an exact copy. Threaded over output mid sectors (disjoint writes).
pub fn apply_mode_rotation<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    theta: f64,
) -> ContractedSparseCores<T> {
    apply_impl(tc, theta, true)
}

/// Serial twin of [`apply_mode_rotation`] (the matches-serial reference).
pub fn apply_mode_rotation_serial<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    theta: f64,
) -> ContractedSparseCores<T> {
    apply_impl(tc, theta, false)
}

/// The exact adjacent-orbital SWAP of a two-site block: the pair rotation at `(c, s) = (0, 1)`,
/// i.e. `S = exp((π/2)·Ĝ_{k+1,k})` with no `cos(π/2)` roundoff (out of place, threaded over output
/// mid sectors with disjoint writes).
pub fn apply_mode_swap<T: Scalar>(tc: &ContractedSparseCores<T>) -> ContractedSparseCores<T> {
    apply_mode_rotation_cs(tc, 0.0, 1.0)
}

pub(crate) fn apply_mode_rotation_cs<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    c: f64,
    s: f64,
) -> ContractedSparseCores<T> {
    apply_gate(tc, &gate16_cs(c, s), true)
}

fn apply_impl<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    theta: f64,
    par: bool,
) -> ContractedSparseCores<T> {
    if theta == 0.0 {
        return tc.clone();
    }
    apply_gate(tc, &gate16(theta), par)
}

fn apply_gate<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    g: &[[f64; 16]; 16],
    par: bool,
) -> ContractedSparseCores<T> {
    let mut out = tc.clone();
    assert_representable(tc);
    let tiles = tilings(tc);
    let blocks: Vec<(Sector, Option<Mat<T>>)> = if par {
        tc.mid_qn
            .par_iter()
            .map(|&m| (m, rotated_block(tc, g, &tiles, m)))
            .collect()
    } else {
        tc.mid_qn
            .iter()
            .map(|&m| (m, rotated_block(tc, g, &tiles, m)))
            .collect()
    };
    for (m, b) in blocks {
        if let Some(b) = b {
            out.blocks[m] = Some(b);
        }
    }
    out
}

/// The pooled bond-`(k+1)` singular values of `G(θ)·tc` (values only, canonical mid-sector
/// order). Meaningful as a Schmidt spectrum when the train's orthogonality centre is on the pair.
pub fn rotated_bond_svals<T: Scalar>(tc: &ContractedSparseCores<T>, theta: f64) -> Vec<f64> {
    let rot = apply_mode_rotation(tc, theta);
    let per: Vec<Vec<f64>> = rot
        .mid_qn
        .par_iter()
        .map(|&m| rot.block(m).map(svals_qr).unwrap_or_default())
        .collect();
    per.into_iter().flatten().collect()
}

fn rotated_bond_svals_tiled<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    tiles: &HashMap<Sector, (Vec<Tile>, Vec<Tile>)>,
    theta: f64,
) -> Vec<f64> {
    let g = (theta != 0.0).then(|| gate16(theta));
    let per: Vec<Vec<f64>> = tc
        .mid_qn
        .par_iter()
        .map(
            |&m| match g.as_ref().and_then(|g| rotated_block(tc, g, tiles, m)) {
                Some(b) => svals_qr(b.as_ref()),
                None => tc.block(m).map(svals_qr).unwrap_or_default(),
            },
        )
        .collect();
    per.into_iter().flatten().collect()
}

/// Options of the angle search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModeRotOpts {
    /// What is minimized over θ.
    pub functional: BondFunctional,
    /// Coarse grid points on `(−π/2, π/2]` (even ⇒ `θ = 0` is on the grid). Default 8.
    pub n_grid: usize,
    /// Golden-section refinements around the best grid point. Default 8.
    pub n_refine: usize,
    /// A rotation is taken only if it lowers the functional by more than
    /// `improve_tol · max(|f(0)|, 1)`; otherwise θ is EXACTLY `0.0`. Default `1e-10`.
    pub improve_tol: f64,
    /// A grid whose spread is within `flat_tol · max(|f(0)|, 1)` is flat: θ = 0. Default `1e-12`.
    pub flat_tol: f64,
}

impl Default for ModeRotOpts {
    fn default() -> Self {
        ModeRotOpts {
            functional: BondFunctional::default(),
            n_grid: 8,
            n_refine: 8,
            improve_tol: 1e-10,
            flat_tol: 1e-12,
        }
    }
}

/// Minimize `opts.functional` of the rotated block's bond spectrum over `θ ∈ (−π/2, π/2]` (the
/// spectrum is π-periodic: `G(θ+π)` differs from `G(θ)` by row/column signs). Derivative-free and
/// fixed-budget — an even coarse grid, then golden section around the best point (not Brent: the
/// objective is only piecewise smooth at singular-value crossings). Returns
/// `(theta, f(0), f(theta))`; `theta` is exactly `0.0` unless it improves by `improve_tol`.
/// The minimizer itself is the crate-private `minimize_angle`, shared with the SU(2) twin.
pub fn search_angle<T: Scalar>(
    tc: &ContractedSparseCores<T>,
    opts: &ModeRotOpts,
) -> (f64, f64, f64) {
    assert_representable(tc);
    let tiles = tilings(tc);
    minimize_angle(
        |theta: f64| {
            bond_functional(
                &rotated_bond_svals_tiled(tc, &tiles, theta),
                opts.functional,
            )
        },
        opts,
    )
}

pub(crate) fn minimize_angle(eval: impl Fn(f64) -> f64, opts: &ModeRotOpts) -> (f64, f64, f64) {
    assert!(
        opts.n_grid >= 2 && opts.n_grid.is_multiple_of(2),
        "search_angle: n_grid must be even and ≥ 2 so that θ = 0 is a grid point"
    );
    let f0 = eval(0.0);
    let scale = f0.abs().max(1.0);
    let n = opts.n_grid;
    let step = std::f64::consts::PI / n as f64;
    let mut best = (0.0f64, f0);
    let (mut lo, mut hi) = (f0, f0);
    for j in 0..n {
        let theta = if j + 1 == n / 2 {
            0.0
        } else {
            -std::f64::consts::FRAC_PI_2 + (j + 1) as f64 * step
        };
        let f = if theta == 0.0 { f0 } else { eval(theta) };
        lo = lo.min(f);
        hi = hi.max(f);
        if f < best.1 {
            best = (theta, f);
        }
    }
    if hi - lo <= opts.flat_tol * scale {
        return (0.0, f0, f0);
    }
    let inv_phi = 0.5 * (5.0f64.sqrt() - 1.0);
    let (mut a, mut b) = (best.0 - step, best.0 + step);
    let mut x1 = b - inv_phi * (b - a);
    let mut x2 = a + inv_phi * (b - a);
    let (mut f1, mut f2) = (eval(x1), eval(x2));
    for _ in 0..opts.n_refine {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - inv_phi * (b - a);
            f1 = eval(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = eval(x2);
        }
        if f1 < best.1 {
            best = (x1, f1);
        }
        if f2 < best.1 {
            best = (x2, f2);
        }
    }
    if f0 - best.1 > opts.improve_tol * scale {
        (best.0, f0, best.1)
    } else {
        (0.0, f0, f0)
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::coeff::CoeffTensor;
    use crate::core::twosite::{contract, factor_qc};
    use crate::hamiltonian::{SparseHamiltonian, rayleigh_quotient};
    use crate::integrals::Integrals;
    use crate::qn::QnSpace;
    use crate::rotation::{g_pq, givens, rotate_integrals};
    use crate::test_util::randomized_state;
    use crate::vector::TtVector;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn centre_on(x: &mut TtVector<f64>, k: usize) {
        x.right_orthogonalize(true);
        for j in 0..k {
            let cp = contract(x.core(j), x.core(j + 1));
            let (a, b) = factor_qc(&cp);
            x.set_cores(a, b);
        }
    }

    fn with_gate(x: &TtVector<f64>, k: usize, theta: f64) -> TtVector<f64> {
        let mut y = x.clone();
        let cp = apply_mode_rotation(&contract(y.core(k), y.core(k + 1)), theta);
        let (a, b) = factor_qc(&cp);
        y.set_cores(a, b);
        y
    }

    fn dense(x: &TtVector<f64>) -> Vec<f64> {
        x.to_dense()
    }

    #[test]
    fn mr_gate16_orthogonal_group_law_and_structure() {
        let id = gate16(0.0);
        for (i, row) in id.iter().enumerate() {
            for (j, &v) in row.iter().enumerate() {
                assert_eq!(v.to_bits(), (if i == j { 1.0f64 } else { 0.0 }).to_bits());
            }
        }
        let mul = |a: &[[f64; 16]; 16], b: &[[f64; 16]; 16]| {
            let mut c = [[0.0f64; 16]; 16];
            for i in 0..16 {
                for j in 0..16 {
                    for l in 0..16 {
                        c[i][j] += a[i][l] * b[l][j];
                    }
                }
            }
            c
        };
        for &th in &[0.1, 0.7, -1.3, std::f64::consts::FRAC_PI_2] {
            let g = gate16(th);
            let mut gt = [[0.0f64; 16]; 16];
            for i in 0..16 {
                for j in 0..16 {
                    gt[i][j] = g[j][i];
                }
            }
            let p = mul(&gt, &g);
            for (i, row) in p.iter().enumerate() {
                for (j, &v) in row.iter().enumerate() {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((v - want).abs() < 1e-15, "θ={th}: GᵀG[{i}][{j}] = {v}");
                }
            }
            let count = |c: usize| {
                let (fi, fj) = (Family::ALL[c / 4], Family::ALL[c % 4]);
                let (a, b) = (fi.delta(), fj.delta());
                (a.0 + b.0, a.1 + b.1)
            };
            for i in 0..16 {
                for j in 0..16 {
                    if count(i) != count(j) {
                        assert_eq!(g[i][j], 0.0, "θ={th}: leak {i}→{j}");
                    }
                }
            }
        }
        let (a, b) = (0.31, -0.47);
        let lhs = mul(&gate16(a), &gate16(b));
        let rhs = gate16(a + b);
        for i in 0..16 {
            for j in 0..16 {
                assert!((lhs[i][j] - rhs[i][j]).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn mr_generator_matches_g_pq() {
        let space = QnSpace::from_n_two_sz(5, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_7A7E);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let h = 1e-4;
        for k in 0..4 {
            let plus = dense(&with_gate(&x, k, h));
            let minus = dense(&with_gate(&x, k, -h));
            let want = dense(&g_pq(&x, k + 1, k));
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for ((p, m), w) in plus.iter().zip(&minus).zip(&want) {
                let fd = (p - m) / (2.0 * h);
                num += (fd - w) * (fd - w);
                den += w * w;
            }
            let rel = (num / den).sqrt();
            assert!(
                rel < 1e-6,
                "pair ({k}, {}): generator mismatch rel {rel:.3e}",
                k + 1
            );
        }
    }

    #[test]
    fn mr_energy_invariance_under_rotated_integrals() {
        let d = 4;
        let space = QnSpace::from_n_two_sz(d, 2, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_E1A7);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for xx in t.data.iter_mut().chain(v.data.iter_mut()) {
            *xx = rng.random_range(-1.0..1.0);
        }
        let ints = Integrals {
            t,
            v,
            e0: 0.0,
            norb: d,
            nelec: 2,
            ms2: 0,
            orbsym: Vec::new(),
            isym: None,
        };
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let h0 = SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
        let e0 = rayleigh_quotient(&h0, &x);
        for k in 0..d - 1 {
            for &theta in &[0.4, -1.1] {
                let y = with_gate(&x, k, theta);
                let rot = rotate_integrals(&ints, &givens(d, k, theta));
                let h1 = SparseHamiltonian::new(space, &rot.t, &rot.v, None, f64::EPSILON);
                let e1 = rayleigh_quotient(&h1, &y);
                assert!(
                    (e1 - e0).abs() <= 1e-10 * e0.abs().max(1.0),
                    "pair {k}, θ={theta}: E {e0} → {e1}"
                );
            }
        }
    }

    #[test]
    fn mr_typed_same_irrep_gate_energy_invariance_and_dense_oracle() {
        use crate::pointgroup::{Irrep, PointGroup};
        let d = 6;
        let irs: Vec<Irrep> = [0u8, 0, 0, 2, 2, 3]
            .iter()
            .map(|&i| Irrep::new(i))
            .collect();
        let pg = PointGroup::new(2, &irs);
        let space = QnSpace::with_point_group(d, 2, 2, pg, Irrep::IDENTITY);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_7E9D);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for xx in t.data.iter_mut().chain(v.data.iter_mut()) {
            *xx = rng.random_range(-1.0..1.0);
        }
        for i in 0..d {
            for j in 0..d {
                if pg.irrep_of_sites([i, j]) != Irrep::IDENTITY {
                    t.set(&[i, j], 0.0);
                }
                for k in 0..d {
                    for l in 0..d {
                        if pg.irrep_of_sites([i, j, k, l]) != Irrep::IDENTITY {
                            v.set(&[i, j, k, l], 0.0);
                        }
                    }
                }
            }
        }
        let ints = Integrals {
            t,
            v,
            e0: 0.0,
            norb: d,
            nelec: 4,
            ms2: 0,
            orbsym: Vec::new(),
            isym: None,
        };
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let h0 = SparseHamiltonian::new(space, &ints.t, &ints.v, None, f64::EPSILON);
        let e0 = rayleigh_quotient(&h0, &x);
        let xf = x.forget_point_group();
        let mut n_pairs = 0;
        for k in 0..d - 1 {
            if pg.irrep_of(k) != pg.irrep_of(k + 1) {
                continue;
            }
            n_pairs += 1;
            for &theta in &[0.4, -1.1] {
                let y = with_gate(&x, k, theta);
                let rot = rotate_integrals(&ints, &givens(d, k, theta));
                let h1 = SparseHamiltonian::new(space, &rot.t, &rot.v, None, f64::EPSILON);
                let e1 = rayleigh_quotient(&h1, &y);
                assert!(
                    (e1 - e0).abs() <= 1e-10 * e0.abs().max(1.0),
                    "pair {k}, θ={theta}: typed E {e0} → {e1}"
                );
                let a = dense(&y.forget_point_group());
                let b = dense(&with_gate(&xf, k, theta));
                let (mut num, mut den, mut moved) = (0.0f64, 0.0f64, 0.0f64);
                let x0 = dense(&xf);
                for ((p, q), r) in a.iter().zip(&b).zip(&x0) {
                    num += (p - q) * (p - q);
                    den += q * q;
                    moved += (q - r) * (q - r);
                }
                let rel = (num / den).sqrt();
                assert!(
                    rel < 1e-12,
                    "pair {k}, θ={theta}: forget∘gate ≠ gate∘forget ({rel:.3e})"
                );
                assert!(
                    (moved / den).sqrt() > 1e-3,
                    "pair {k}, θ={theta}: the gate did not move the state"
                );
            }
        }
        assert_eq!(n_pairs, 3);
    }

    #[test]
    fn mr_zero_angle_serial_parity_and_norm() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_5E71);
        let x: TtVector<f64> = randomized_state(space, 2, 4, &mut rng);
        let cp = contract(x.core(2), x.core(3));
        let same = apply_mode_rotation(&cp, 0.0);
        let norm2 = |c: &ContractedSparseCores<f64>| -> f64 {
            c.mid_qn
                .iter()
                .filter_map(|&m| c.block(m))
                .map(|b| {
                    let mut s = 0.0;
                    for j in 0..b.ncols() {
                        for i in 0..b.nrows() {
                            s += b[(i, j)] * b[(i, j)];
                        }
                    }
                    s
                })
                .sum()
        };
        for &m in &cp.mid_qn {
            if let (Some(a), Some(b)) = (cp.block(m), same.block(m)) {
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits());
                    }
                }
            }
        }
        let n0 = norm2(&cp);
        for &th in &[0.3, -0.9, 1.4] {
            let p = apply_mode_rotation(&cp, th);
            let s = apply_mode_rotation_serial(&cp, th);
            for &m in &cp.mid_qn {
                if let (Some(a), Some(b)) = (p.block(m), s.block(m)) {
                    for j in 0..a.ncols() {
                        for i in 0..a.nrows() {
                            assert_eq!(a[(i, j)].to_bits(), b[(i, j)].to_bits());
                        }
                    }
                }
            }
            assert!((norm2(&p) - n0).abs() <= 1e-13 * n0, "θ={th}");
        }
    }

    #[test]
    fn mr_search_beats_the_planted_inverse() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_5EA2);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let k = 2;
        centre_on(&mut x, k);
        let opts = ModeRotOpts::default();
        let cp0 = contract(x.core(k), x.core(k + 1));
        let f_origin = bond_functional(&rotated_bond_svals(&cp0, 0.0), opts.functional);
        let phi = 0.37;
        let cp = apply_mode_rotation(&cp0, phi);
        let (theta, f0, ft) = search_angle(&cp, &opts);
        assert!(ft <= f0, "search returned a worse point: {ft} > {f0}");
        assert!(
            ft <= f_origin + 1e-3 * f_origin.abs().max(1.0),
            "search (θ={theta}, f={ft}) did worse than undoing the plant (f={f_origin})"
        );
        let a = bond_functional(&rotated_bond_svals(&cp, 0.2), opts.functional);
        let b = bond_functional(
            &rotated_bond_svals(&cp, 0.2 + std::f64::consts::PI),
            opts.functional,
        );
        assert!((a - b).abs() < 1e-12, "π-periodicity: {a} vs {b}");
    }

    #[test]
    fn mr_tiled_scorer_is_bitwise_rotated_bond_svals() {
        let space = QnSpace::from_n_two_sz(6, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x7_11ED);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        let k = 2;
        centre_on(&mut x, k);
        let cp = contract(x.core(k), x.core(k + 1));
        let tiles = tilings(&cp);
        for theta in [0.0, 0.37, -0.81, 0.2, std::f64::consts::FRAC_PI_2, -1e-9] {
            let a = rotated_bond_svals(&cp, theta);
            let b = rotated_bond_svals_tiled(&cp, &tiles, theta);
            assert_eq!(a.len(), b.len(), "θ = {theta}: spectrum length");
            for (i, (u, v)) in a.iter().zip(&b).enumerate() {
                assert_eq!(u.to_bits(), v.to_bits(), "θ = {theta}, σ[{i}]: {u} vs {v}");
            }
        }
    }

    #[test]
    fn mr_c64_gate_acts_on_real_and_imaginary_parts_separately() {
        use crate::scalar::C64;
        let space = QnSpace::from_n_two_sz(5, 3, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_C64);
        let xr: TtVector<f64> = randomized_state(space, 2, 2, &mut rng);
        let cr = contract(xr.core(1), xr.core(2));
        let mut ci = cr.clone();
        for &m in &cr.mid_qn {
            if let Some(mut b) = ci.block_mut(m) {
                for j in 0..b.ncols() {
                    for i in 0..b.nrows() {
                        b[(i, j)] = rng.random_range(-1.0..1.0);
                    }
                }
            }
        }
        let mut cz = ContractedSparseCores::<C64>::zeros(space, 1, &cr.row_ranks, &cr.col_ranks);
        for &m in &cr.mid_qn {
            if let (Some(a), Some(b), Some(mut z)) = (cr.block(m), ci.block(m), cz.block_mut(m)) {
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        z[(i, j)] = C64::new(a[(i, j)], b[(i, j)]);
                    }
                }
            }
        }
        let th = 0.83;
        let (gr, gi, gz) = (
            apply_mode_rotation(&cr, th),
            apply_mode_rotation(&ci, th),
            apply_mode_rotation(&cz, th),
        );
        for &m in &cr.mid_qn {
            if let (Some(a), Some(b), Some(z)) = (gr.block(m), gi.block(m), gz.block(m)) {
                for j in 0..a.ncols() {
                    for i in 0..a.nrows() {
                        assert!((z[(i, j)].re - a[(i, j)]).abs() < 1e-15);
                        assert!((z[(i, j)].im - b[(i, j)]).abs() < 1e-15);
                    }
                }
            }
        }
    }

    #[test]
    fn mr_svals_qr_matches_thin_svd_qr() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_5A15);
        for &(m, n) in &[(7usize, 4usize), (3, 9), (5, 5)] {
            let a = Mat::<f64>::from_fn(m, n, |_, _| rng.random_range(-1.0..1.0));
            let full = crate::core::factor::thin_svd_qr(a.as_ref()).s;
            let vals = svals_qr(a.as_ref());
            assert_eq!(full.len(), vals.len());
            for (x, y) in full.iter().zip(&vals) {
                assert!((x - y).abs() < 1e-13, "{m}×{n}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn mr_gate16_cs_is_the_exact_signed_swap() {
        use std::f64::consts::FRAC_PI_2;
        let n = |f: Family| {
            let (a, b) = f.delta();
            a + b
        };
        let g = gate16_cs(0.0, 1.0);
        let h = gate16(FRAC_PI_2);
        for i in 0..16 {
            assert_eq!(g[i].iter().filter(|v| **v != 0.0).count(), 1, "row {i}");
            assert_eq!((0..16).filter(|&r| g[r][i] != 0.0).count(), 1, "col {i}");
            for j in 0..16 {
                assert!(
                    g[i][j] == 0.0 || g[i][j].abs() == 1.0,
                    "entry [{i}][{j}] = {}",
                    g[i][j]
                );
            }
        }
        for fi in Family::ALL {
            for fj in Family::ALL {
                let sign = if n(fj) % 2 == 1 && n(fi) % 2 == 0 {
                    -1.0
                } else {
                    1.0
                };
                assert!(
                    g[chan(fj, fi)][chan(fi, fj)] == sign,
                    "({fi:?},{fj:?}) → ({fj:?},{fi:?}): {} vs {sign}",
                    g[chan(fj, fi)][chan(fi, fj)]
                );
            }
        }
        let mut worst = 0.0f64;
        for i in 0..16 {
            for j in 0..16 {
                assert!(
                    g[i][j] == h[i][j].round(),
                    "[{i}][{j}]: {} vs round({})",
                    g[i][j],
                    h[i][j]
                );
                worst = worst.max((g[i][j] - h[i][j]).abs());
            }
        }
        println!("mr_gate16_cs: max|gate16_cs(0,1) − gate16(π/2)| = {worst:e}");
        assert!(worst <= 1e-16, "near-swap gap {worst:e}");
        let mul = |a: &[[f64; 16]; 16], b: &[[f64; 16]; 16]| {
            let mut c = [[0.0f64; 16]; 16];
            for i in 0..16 {
                for j in 0..16 {
                    for l in 0..16 {
                        c[i][j] += a[i][l] * b[l][j];
                    }
                }
            }
            c
        };
        let g2 = mul(&g, &g);
        let g4 = mul(&g2, &g2);
        for i in 0..16 {
            let parity = if (n(Family::ALL[i / 4]) + n(Family::ALL[i % 4])) % 2 == 1 {
                -1.0
            } else {
                1.0
            };
            for j in 0..16 {
                let want2 = if i == j { parity } else { 0.0 };
                let want4 = if i == j { 1.0 } else { 0.0 };
                assert!(g2[i][j] == want2, "S²[{i}][{j}] = {}", g2[i][j]);
                assert!(g4[i][j] == want4, "S⁴[{i}][{j}] = {}", g4[i][j]);
            }
        }
        for &th in &[0.1, 0.37, 0.7, -0.83, -1.3, FRAC_PI_2, 3.0] {
            let (s, c) = f64::sin_cos(th);
            let (a, b) = (gate16(th), gate16_cs(c, s));
            for i in 0..16 {
                for j in 0..16 {
                    assert_eq!(a[i][j].to_bits(), b[i][j].to_bits(), "θ={th} [{i}][{j}]");
                }
            }
        }
        let one = gate16_cs(1.0, 0.0);
        let id = gate16(0.0);
        let mut neg_zero = 0usize;
        for i in 0..16 {
            for j in 0..16 {
                assert!(one[i][j] == id[i][j], "gate16_cs(1,0) [{i}][{j}]");
                neg_zero += (one[i][j].to_bits() == (-0.0f64).to_bits()) as usize;
            }
        }
        println!("mr_gate16_cs: gate16_cs(1, 0) carries {neg_zero} −0.0 entries");
    }

    #[test]
    fn mr_apply_mode_swap_period_parity_and_serial() {
        let space = QnSpace::from_n_two_sz(5, 4, 0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_5A4B);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        let n = |f: Family| {
            let (a, b) = f.delta();
            a + b
        };
        let entries = |c: &ContractedSparseCores<f64>| -> Vec<f64> {
            let mut v = Vec::new();
            for &m in &c.mid_qn {
                if let Some(b) = c.block(m) {
                    for j in 0..b.ncols() {
                        for i in 0..b.nrows() {
                            v.push(b[(i, j)]);
                        }
                    }
                }
            }
            v
        };
        for k in 0..4 {
            let tc = contract(x.core(k), x.core(k + 1));
            let s1 = apply_mode_swap(&tc);
            let serial = apply_gate(&tc, &gate16_cs(0.0, 1.0), false);
            let (e0, e1, es) = (entries(&tc), entries(&s1), entries(&serial));
            assert!(
                e1.iter().zip(&es).all(|(a, b)| a.to_bits() == b.to_bits()),
                "k={k}: par vs serial"
            );
            assert!(e1 != e0, "k={k}: anti-vacuity: the swap did nothing");
            let mut a0: Vec<u64> = e0.iter().map(|v| v.abs().to_bits()).collect();
            let mut a1: Vec<u64> = e1.iter().map(|v| v.abs().to_bits()).collect();
            a0.sort_unstable();
            a1.sort_unstable();
            assert_eq!(a0, a1, "k={k}: |coefficient| multiset");
            let s2 = apply_mode_swap(&s1);
            let s4 = apply_mode_swap(&apply_mode_swap(&s2));
            assert_eq!(entries(&s4), e0, "k={k}: swap⁴");
            let tiles = tilings(&tc);
            for &m in &tc.mid_qn {
                let (Some(a), Some(b)) = (tc.block(m), s2.block(m)) else {
                    continue;
                };
                let (rows, cols) = &tiles[&m];
                for rt in rows {
                    for ct in cols {
                        let sign = if (n(rt.fam) + n(ct.fam)) % 2 == 1 {
                            -1.0
                        } else {
                            1.0
                        };
                        for cc in 0..ct.len {
                            for rr in 0..rt.len {
                                let (i, j) = (rt.off + rr, ct.off + cc);
                                assert!(
                                    b[(i, j)] == sign * a[(i, j)],
                                    "k={k}: swap² tile ({:?},{:?})",
                                    rt.fam,
                                    ct.fam
                                );
                            }
                        }
                    }
                }
            }
            let near = entries(&apply_mode_rotation(&tc, std::f64::consts::FRAC_PI_2));
            let scale = e0.iter().fold(1.0f64, |m, v| m.max(v.abs()));
            let gap = e1
                .iter()
                .zip(&near)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            println!("mr_apply_mode_swap k={k}: max|swap − G(π/2)| = {gap:e}");
            assert!(gap <= 1e-15 * scale, "k={k}: exact vs near swap {gap:e}");
        }
    }

    #[test]
    #[should_panic(expected = "different irreps")]
    fn mr_apply_mode_swap_refuses_cross_irrep() {
        let (_, space) = crate::test_util::pg3_space_d6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x60_C2F0);
        let x: TtVector<f64> =
            crate::vector::tt_randn(space, &crate::qn::max_ranks(&space), &mut rng);
        let _ = apply_mode_swap(&contract(x.core(0), x.core(1)));
    }
}
