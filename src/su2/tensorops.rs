// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Elementary (irreducible-tensor) operators on an SU(2) TT: number/hopping (rank-0, `ΔN=0`, routed
//! through [`Su2SparseHamiltonian`]) and the spin-shifting family `a†`/`a`/`S`/`Pair` (built
//! directly via the per-core Wigner–Eckart engine [`apply_site_tensor`]). Elementary SU(2) tensor
//! operators.

use faer::prelude::ReborrowMut;
use faer::{unzip, zip};

use crate::coeff::CoeffTensor;
use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::assembly::{WireGate, blockclass, jl_window_gated};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::spinalgebra::{SpinAlgebra, TensorOp, tensor_a, tensor_cdag, tensor_identity};
use crate::su2::vector::{Su2TtVector, tt_zeros};

/// Number operator `n̂_i` — the `i==j` special case of
/// [`adag_i_a_j`], routed through [`Su2SparseHamiltonian`] with a single-entry one-body tensor.
pub fn n_i<T: Scalar>(x: &Su2TtVector<T>, i: usize) -> Su2TtVector<T> {
    adag_i_a_j(x, i, i)
}

/// Spin-summed hopping/density `a†_iσ a_jσ` summed over σ (`AdagᵢAⱼ`). Preserves `twoS`. Builds a `d×d` one-body [`CoeffTensor`] with
/// a single nonzero entry `t[i,j] = 1` and a zero two-body tensor, and applies the resulting
/// [`Su2SparseHamiltonian`] to `x`.
pub fn adag_i_a_j<T: Scalar>(x: &Su2TtVector<T>, i: usize, j: usize) -> Su2TtVector<T> {
    assert_eq!(
        x.space.pg.irrep_of(i),
        x.space.pg.irrep_of(j),
        "adag_i_a_j: sites {i}/{j} carry different irreps; a†_i a_j maps the target irrep to \
         target ⊗ g_i ⊗ g_j, which a single-target Su2Space cannot represent"
    );
    let d = x.d();
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    t.set(&[i, j], T::one());
    let v = CoeffTensor::<T>::zeros(d, 4);
    let h = Su2SparseHamiltonian::new(x.space, &t, &v, None, f64::EPSILON);
    h.apply(x)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn src_offset(
    n_y: i64,
    s_y: i64,
    jl: i64,
    wire: i64,
    flux: i64,
    x_ir: Irrep,
    rxk: &Su2RankGrid,
    big_n: usize,
    gate: WireGate,
) -> i64 {
    let mlx = n_y - flux;
    if mlx < 0 || mlx > big_n as i64 {
        return -1;
    }
    if wire == 0 {
        return if jl == s_y { 0 } else { -1 };
    }
    let mut off: i64 = 0;
    let mut jlp = s_y - wire;
    while jlp <= s_y + wire {
        if jlp >= 0 && (gate == WireGate::Window || wire <= jlp + s_y) {
            if jlp == jl {
                return off;
            }
            if jlp <= big_n as i64 {
                off += rxk[Su2Sector::with_irrep(mlx as usize, jlp as usize, x_ir)] as i64;
            }
        }
        jlp += 2;
    }
    -1
}

#[derive(Clone, Copy)]
pub(crate) struct WireSite<'a> {
    pub red: &'a [[f64; 3]; 3],
    pub op_odd: bool,
    pub op_two_k: i64,
    pub wire_l: i64,
    pub wire_r: i64,
    pub flux_l: i64,
    pub flux_r: i64,
    pub irr_l: Irrep,
    pub irr_r: Irrep,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn wire_ranks(
    rx: &Su2RankGrid,
    wire: i64,
    flux: i64,
    irr: Irrep,
    src_space: Su2Space,
    out_space: Su2Space,
    bond: usize,
    gate: WireGate,
) -> Su2RankGrid {
    let mut ry = Su2RankGrid::zeros(&out_space);
    for s_x in src_space.bond_sectors(bond) {
        let r = rx[s_x];
        if r == 0 {
            continue;
        }
        let n_o = s_x.n as i64 + flux;
        let mut sly = s_x.two_s as i64 - wire;
        while sly <= s_x.two_s as i64 + wire {
            if sly >= 0 && n_o >= 0 && (gate == WireGate::Window || wire <= sly + s_x.two_s as i64)
            {
                let sy = Su2Sector::with_irrep(n_o as usize, sly as usize, s_x.irrep.fuse(irr));
                if out_space.contains(bond, sy) {
                    ry[sy] += r;
                }
            }
            sly += 2;
        }
    }
    ry
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn wire_core<T: Scalar>(
    src: &Su2Core<T>,
    src_ranks: (&Su2RankGrid, &Su2RankGrid),
    src_space: Su2Space,
    out_space: Su2Space,
    out_ranks: (&Su2RankGrid, &Su2RankGrid),
    site: &WireSite<'_>,
    spin: &mut SpinAlgebra,
    gate: WireGate,
) -> Su2Core<T> {
    let k = src.k;
    let big_n = src_space.n;
    let (rx_l, rx_r) = src_ranks;
    let (ry_l, ry_r) = out_ranks;
    let WireSite {
        red,
        op_odd,
        op_two_k,
        wire_l,
        wire_r,
        flux_l,
        flux_r,
        irr_l,
        irr_r,
    } = *site;

    let mut yk = Su2Core::zeros(out_space, k, ry_l, ry_r);
    let g_k = out_space.pg.irrep_of(k);
    let row_qn = out_space.bond_sectors(k);
    let col_qn = out_space.bond_sectors(k + 1);

    for &s_row in &row_qn {
        for &s_col in &col_qn {
            let (occ_n, occ_y, twosy) = match blockclass(
                s_col.n as i64 - s_row.n as i64,
                s_col.two_s as i64 - s_row.two_s as i64,
            ) {
                Some(v) => v,
                None => continue,
            };
            if s_col.irrep != s_row.irrep.fuse(occ_n.flux(g_k)) {
                continue;
            }
            let ml_m = s_row.n as i64 - flux_l;
            let ml_p = s_col.n as i64 - flux_r;
            let x_ir_m = s_row.irrep.fuse(irr_l);
            let x_ir_p = s_col.irrep.fuse(irr_r);
            for two_j_m in jl_window_gated(s_row.two_s as i64, wire_l, gate) {
                let ro = src_offset(
                    s_row.n as i64,
                    s_row.two_s as i64,
                    two_j_m,
                    wire_l,
                    flux_l,
                    x_ir_m,
                    rx_l,
                    big_n,
                    gate,
                );
                if ro < 0 {
                    continue;
                }
                if ml_m < 0
                    || !src_space.contains(
                        k,
                        Su2Sector::with_irrep(ml_m as usize, two_j_m as usize, x_ir_m),
                    )
                {
                    continue;
                }
                for two_j_p in jl_window_gated(s_col.two_s as i64, wire_r, gate) {
                    let co = src_offset(
                        s_col.n as i64,
                        s_col.two_s as i64,
                        two_j_p,
                        wire_r,
                        flux_r,
                        x_ir_p,
                        rx_r,
                        big_n,
                        gate,
                    );
                    if co < 0 {
                        continue;
                    }
                    if ml_p < 0
                        || !src_space.contains(
                            k + 1,
                            Su2Sector::with_irrep(ml_p as usize, two_j_p as usize, x_ir_p),
                        )
                    {
                        continue;
                    }
                    let (occ_m, occ_x, twosx) = match blockclass(ml_p - ml_m, two_j_p - two_j_m) {
                        Some(v) => v,
                        None => continue,
                    };
                    if red[occ_x][occ_y] == 0.0 {
                        continue;
                    }
                    debug_assert_eq!(
                        x_ir_p,
                        x_ir_m.fuse(occ_m.flux(g_k)),
                        "tensorops: x-side irrep transition disagrees with the family flux at site {k}"
                    );
                    let xblk = match src.block(
                        occ_m,
                        Su2Sector::with_irrep(ml_m as usize, two_j_m as usize, x_ir_m),
                    ) {
                        Some(b) if b.nrows() > 0 && b.ncols() > 0 => b,
                        _ => continue,
                    };
                    let jw: T = if op_odd && ml_m.rem_euclid(2) == 1 {
                        T::from_f64(-1.0)
                    } else {
                        T::one()
                    };
                    let f = spin.recouple(
                        two_j_m as i32,
                        two_j_p as i32,
                        twosx as i32,
                        wire_l as i32,
                        wire_r as i32,
                        op_two_k as i32,
                        s_row.two_s as i32,
                        s_col.two_s as i32,
                        twosy as i32,
                    );
                    let alpha = jw * T::from_f64(f * red[occ_x][occ_y]);
                    if alpha.abs_s() == 0.0 {
                        continue;
                    }
                    let (nr, nc) = (xblk.nrows(), xblk.ncols());
                    let mut yblk = yk.block_mut(occ_n, s_row).expect("output block present");
                    let dst = yblk
                        .rb_mut()
                        .submatrix_mut(ro as usize, co as usize, nr, nc);
                    zip!(dst, xblk).for_each(|unzip!(dd, ss)| *dd += alpha * *ss);
                }
            }
        }
    }

    yk
}

/// The per-core Wigner–Eckart engine: apply irreducible-rank site tensor `op` at site `i`,
/// producing an output train of total doubled spin `two_s_out`. Admissibility (`Nout = N + op.flux`,
/// `|two_s_out - twoS| <= op.twok`, parity) gates a `tt_zeros` short-circuit. Builds output bond
/// ranks via the `wire`-fusion accumulation, then per-core scatters the reduced
/// `recouple`-weighted contraction — the SAME `recouple`/`blockclass`/`jl_window`
/// machinery the Hamiltonian fill uses. Finalizes with a uniform scalar `weight` (`weight · y`).
pub fn apply_site_tensor<T: Scalar>(
    x: &Su2TtVector<T>,
    i: usize,
    op: &TensorOp,
    two_s_out: i64,
    weight: T,
) -> Su2TtVector<T> {
    let d = x.d();
    let big_n = x.space.n as i64;
    let two_s_in = x.space.two_s as i64;
    let two_kappa = op.two_k as i64;
    let n_out = big_n + op.flux as i64;

    let gamma = if op.odd {
        x.space.pg.irrep_of(i)
    } else {
        Irrep::IDENTITY
    };
    let out_target = x.space.target.fuse(gamma);

    let ok = (0..=2 * d as i64).contains(&n_out)
        && two_s_out >= 0
        && two_s_out <= n_out.min(2 * d as i64 - n_out)
        && (n_out - two_s_out).rem_euclid(2) == 0
        && (two_s_out - two_s_in).abs() <= two_kappa;
    if !ok {
        let clamped_n = n_out.clamp(0, 2 * d as i64) as usize;
        let clamped_s = two_s_out.max(0) as usize;
        return tt_zeros(Su2Space {
            pg: x.space.pg,
            target: out_target,
            ..Su2Space::new(d, clamped_n, clamped_s)
        });
    }

    let out_space = Su2Space {
        pg: x.space.pg,
        target: out_target,
        ..Su2Space::new(d, n_out as usize, two_s_out as usize)
    };

    let mut spin = SpinAlgebra::new();
    let opred = spin.reduce_siteop(op);
    let id_red = spin.reduce_siteop(&tensor_identity());

    let ry: Vec<Su2RankGrid> = (0..=d)
        .map(|k| {
            let right = k > i;
            wire_ranks(
                &x.ranks[k],
                if right { two_kappa } else { 0 },
                if right { op.flux as i64 } else { 0 },
                if right { gamma } else { Irrep::IDENTITY },
                x.space,
                out_space,
                k,
                WireGate::Window,
            )
        })
        .collect();
    if ry[d].total() == 0 {
        return tt_zeros(out_space);
    }

    let mut cores: Vec<Su2Core<T>> = Vec::with_capacity(d);
    for k in 0..d {
        let site = WireSite {
            red: if k == i { &opred } else { &id_red },
            op_odd: if k == i { op.odd } else { false },
            op_two_k: if k == i { two_kappa } else { 0 },
            wire_l: if k > i { two_kappa } else { 0 },
            wire_r: if k >= i { two_kappa } else { 0 },
            flux_l: if k > i { op.flux as i64 } else { 0 },
            flux_r: if k >= i { op.flux as i64 } else { 0 },
            irr_l: if k > i { gamma } else { Irrep::IDENTITY },
            irr_r: if k >= i { gamma } else { Irrep::IDENTITY },
        };
        cores.push(wire_core(
            x.core(k),
            (&x.ranks[k], &x.ranks[k + 1]),
            x.space,
            out_space,
            (&ry[k], &ry[k + 1]),
            &site,
            &mut spin,
            WireGate::Window,
        ));
    }

    let y = Su2TtVector::new(out_space, cores);
    if weight != T::one() { &y * weight } else { y }
}

/// `a†_i`, rank-½ creation, `ΔN=+1` (`Adagᵢ`). Admissible `two_s_out
/// ∈ {twoS-1, twoS+1}` (clamped `>= 0`).
pub fn adag_i<T: Scalar>(x: &Su2TtVector<T>, i: usize, two_s_out: i64) -> Su2TtVector<T> {
    apply_site_tensor(x, i, &tensor_cdag(), two_s_out, T::one())
}

/// `a_i` (tilde-annihilation), rank-½, `ΔN=-1` (`Aᵢ`). Admissible
/// `two_s_out ∈ {twoS-1, twoS+1}` (clamped `>= 0`).
pub fn a_i<T: Scalar>(x: &Su2TtVector<T>, i: usize, two_s_out: i64) -> Su2TtVector<T> {
    apply_site_tensor(x, i, &tensor_a(), two_s_out, T::one())
}

/// On-site spin `S_i`, rank-1, `ΔN=0` (`Sᵢ`): `[a† ⊗ ã]²` (`couple`
/// at `two_kappa=2`, i.e. spherical rank 1). Admissible `two_s_out ∈ {twoS-2, twoS, twoS+2}`
/// (clamped `>= 0`).
pub fn s_i<T: Scalar>(x: &Su2TtVector<T>, i: usize, two_s_out: i64) -> Su2TtVector<T> {
    let mut spin = SpinAlgebra::new();
    let op = spin.couple(&tensor_cdag(), &tensor_a(), 2);
    apply_site_tensor(x, i, &op, two_s_out, T::one())
}

/// On-site singlet pair `a†_i↑ a†_i↓`, rank-0, `ΔN=+2` (`Pairᵢ`):
/// `[a† ⊗ a†]⁰` (`couple` at `two_kappa=0`). Preserves `twoS`
/// (the output space's `two_s_out` is `x`'s own `twoS`), but is NOT number-conserving
/// (`ΔN=+2`), so it routes through [`apply_site_tensor`] rather than
/// [`Su2SparseHamiltonian`].
pub fn pair_i<T: Scalar>(x: &Su2TtVector<T>, i: usize) -> Su2TtVector<T> {
    let mut spin = SpinAlgebra::new();
    let op = spin.couple(&tensor_cdag(), &tensor_cdag(), 0);
    let two_s_out = x.two_s() as i64;
    apply_site_tensor(x, i, &op, two_s_out, T::one())
}

#[cfg(test)]
mod tests {
    use super::{a_i, adag_i, n_i, pair_i, s_i};
    use crate::C64;
    use crate::scalar::Scalar;
    use crate::su2::ops::dot as su2_dot;
    use crate::su2::qn::Su2Space;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randomized_su2_state, seeded};

    #[test]
    fn s5_c2_number_operator_totals_n() {
        let d = 7;
        let n = 3;
        let two_s = 1;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(734);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 2, 4, &mut rng);
        let mut tot = 0.0f64;
        let denom = su2_dot(&x, &x).re_s();
        for i in 0..d {
            tot += su2_dot(&x, &n_i(&x, i)).re_s();
        }
        tot /= denom;
        assert!(
            (tot - n as f64).abs() <= 1e-8,
            "C2 total={tot} expected={n}"
        );
    }

    #[test]
    fn s5_c3_a_adag_scalar_channel() {
        let d = 5;
        let n = 2;
        let two_s = 0;
        let space = Su2Space::new(d, n, two_s);
        let mut rng = seeded(735);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        for i in 0..d {
            let raised = adag_i(&x, i, 1);
            let z = a_i(&raised, i, 0);
            let two_x = &x + &x;
            let ref_state = &two_x - &n_i(&x, i);
            let c = su2_dot(&ref_state, &z).re_s() / su2_dot(&ref_state, &ref_state).re_s();
            assert!(
                (c - std::f64::consts::FRAC_1_SQRT_2).abs() <= 1e-6,
                "C3 site {i}: c={c}"
            );
            let residual = (&z - &(&ref_state * c)).norm();
            assert!(
                residual / z.norm() <= 1e-7,
                "C3 site {i}: collinearity residual={}",
                residual / z.norm()
            );
        }
    }

    #[test]
    fn s5_c4_adag_a_adjoint_up_to_sqrt2() {
        let d = 5;
        let space_x = Su2Space::new(d, 2, 0);
        let space_z = Su2Space::new(d, 3, 1);
        let mut rng = seeded(736);
        let x: Su2TtVector<f64> = randomized_su2_state(space_x, 1, 3, &mut rng);
        let z: Su2TtVector<f64> = randomized_su2_state(space_z, 1, 3, &mut rng);
        for i in 0..d {
            let lhs = su2_dot(&z, &adag_i(&x, i, 1));
            let rhs = std::f64::consts::FRAC_1_SQRT_2 * su2_dot(&a_i(&z, i, 0), &x);
            assert!(
                (lhs - rhs).abs() <= 1e-7,
                "C4 site {i}: lhs={lhs} rhs={rhs}"
            );
        }
    }

    #[test]
    fn s5_c5_s_total_annihilates_singlet() {
        let d = 5;
        let space = Su2Space::new(d, 2, 0);
        let mut rng = seeded(737);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let mut stot = s_i(&x, 0, 2);
        for i in 1..d {
            stot = &stot + &s_i(&x, i, 2);
        }
        let norm_x = su2_dot(&x, &x).re_s().sqrt();
        let norm_stot = stot.norm();
        assert!(
            norm_stot / norm_x <= 1e-7,
            "C5 total ratio={}",
            norm_stot / norm_x
        );
        let single = s_i(&x, 2, 2);
        let norm_single = su2_dot(&single, &single).re_s().sqrt();
        assert!(norm_single / norm_x > 0.1);
    }

    #[test]
    fn s5_c6_pair_adds_two_electrons() {
        let d = 5;
        let space = Su2Space::new(d, 2, 0);
        let mut rng = seeded(738);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        for i in 0..d {
            let p = pair_i(&x, i);
            let occ = su2_dot(&p, &n_i(&p, i)).re_s() / su2_dot(&p, &p).re_s();
            assert!((occ - 2.0).abs() <= 1e-7, "C6 site {i}: occ={occ}");
        }
    }

    #[test]
    fn s5_tensorops_branch_coverage() {
        let d = 6;
        let space = Su2Space::new(d, 3, 1);
        let mut rng = seeded(20918);
        let x: Su2TtVector<f64> = randomized_su2_state(space, 1, 3, &mut rng);
        let y_lower = adag_i(&x, 2, 0);
        assert!(y_lower.check());
        assert_eq!(y_lower.space.n, 4);
        assert_eq!(y_lower.space.two_s, 0);
        let y_raise = a_i(&x, 2, 2);
        assert!(y_raise.check());
        assert_eq!(y_raise.space.n, 2);
        assert_eq!(y_raise.space.two_s, 2);
        let y_zero = adag_i(&x, 2, 4);
        assert!(y_zero.check());
        assert!(
            y_zero.norm() <= 1e-30,
            "inadmissible target not annihilated"
        );

        let space2 = Su2Space::new(d, 4, 2);
        let x2: Su2TtVector<f64> = randomized_su2_state(space2, 1, 3, &mut rng);
        for &tso in &[0i64, 2] {
            let ys = s_i(&x2, 2, tso);
            assert!(ys.check());
            assert_eq!(ys.space.n, 4);
            assert_eq!(ys.space.two_s, tso as usize);
        }

        let space_c = Su2Space::new(d, 3, 1);
        let xc: Su2TtVector<C64> = randomized_su2_state(space_c, 1, 3, &mut rng);
        let yc = adag_i(&xc, 2, 2);
        assert!(yc.check());
        assert_eq!(yc.space.n, 4);
        assert_eq!(yc.space.two_s, 2);
    }
}

#[cfg(test)]
mod flux_shifted_tests;
