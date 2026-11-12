// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The batched tangent-projection matrix `Θ`: every column `P(u_a)`, `u_a = −Ĝ_{pq}·x`, as raw
//! canonical coefficients — no `u_a` train and no per-pair `TtTangent::project` call.

use crate::core::SparseCore;
use crate::core::factor::{Sketch, mul_sketch_left, mul_sketch_right};
use crate::core::tangent::gauge_complement;
use crate::pairgram::{
    FoldRight, OPEN_TYPES, OpenType, SiteCores, StoredFamilies, align_cols, align_rows, fold_left,
    fold_right, fold_right_seq,
};
use crate::rdm::{active_spins, assert_unit_boundaries};
use crate::vector::TtVector;
use rayon::prelude::*;

fn component(
    left: &Sketch<f64>,
    ket_core: &SparseCore<f64>,
    right: &Sketch<f64>,
) -> SparseCore<f64> {
    let r = align_rows(right, &ket_core.col_ranks);
    let dq = mul_sketch_right(ket_core, &r);
    let l = align_cols(left, &ket_core.row_ranks);
    mul_sketch_left(&l, &dq)
}

/// Every tangent projection `P(u_a)`, `u_a = −Ĝ_{p_a q_a}·x`, as one column-major matrix:
/// `num_coeffs × pairs.len()`, column `a` in the canonical coefficient order of
/// [`TtTangent`](crate::core::tangent::TtTangent) at the frames `(base_l, base_r)` (left- and
/// right-orthogonal factorizations of the same state, `TtTangent::project`'s own contract;
/// `x` is the state the strings act on — pass the SAME state the frames factorize).
///
/// # Panics
///
/// On a nontrivial point group (§3), non-unit boundary ranks, or mismatched frame ranks.
pub fn orbital_tangent_matrix(
    base_l: &TtVector<f64>,
    base_r: &TtVector<f64>,
    x: &TtVector<f64>,
    pairs: &[(usize, usize)],
) -> Vec<f64> {
    let par = crate::par::par_driver_fork(theta_work(x, pairs.len()));
    orbital_tangent_matrix_with(base_l, base_r, x, pairs, par)
}

fn theta_work(x: &TtVector<f64>, npairs: usize) -> usize {
    let d = x.space.d;
    let s = active_spins(&x.space).len();
    let coeffs = (0..d).map(|m| x.core(m).coeff_hint()).max().unwrap_or(0);
    npairs
        .saturating_mul(d)
        .saturating_mul(s)
        .saturating_mul(coeffs)
}

fn orbital_tangent_matrix_with(
    base_l: &TtVector<f64>,
    base_r: &TtVector<f64>,
    x: &TtVector<f64>,
    pairs: &[(usize, usize)],
    par: bool,
) -> Vec<f64> {
    let space = x.space;
    let d = space.d;
    assert!(
        space.pg.is_trivial(),
        "orbital_tangent_matrix: point-group gating is not implemented yet (same door as \
         generalized_gamma2)"
    );
    assert_eq!(
        base_l.ranks, base_r.ranks,
        "orbital_tangent_matrix: frame ranks must agree"
    );
    assert_unit_boundaries(x, "x");
    let mut theta: Vec<f64> = Vec::new();
    if pairs.is_empty() || d == 0 {
        return theta;
    }
    for &(p, q) in pairs {
        assert!(
            p > q && p < d,
            "orbital_tangent_matrix: pairs must be strict lower-triangle (p > q), got ({p},{q})"
        );
    }
    let spins = active_spins(&space);
    let fold_r: FoldRight = if par { fold_right_seq } else { fold_right };

    let sc_all: Vec<SiteCores<'_>> = (0..d)
        .map(|m| SiteCores::new(x.core(m), m, d, &spins))
        .collect();

    let mut p_plain: Vec<Sketch<f64>> = Vec::with_capacity(d);
    p_plain.push(Sketch::identity(space, 0, 1));
    for b in 0..d - 1 {
        let next = fold_left(base_l.core(b), &p_plain[b], x.core(b));
        p_plain.push(next);
    }
    let mut q_plain: Vec<Option<Sketch<f64>>> = vec![None; d + 1];
    q_plain[d] = Some(Sketch::identity(space, d, 1));
    for b in (1..d).rev() {
        let prev = q_plain[b + 1].as_ref().expect("built rightward");
        q_plain[b] = Some(fold_right(x.core(b), prev, base_r.core(b)));
    }

    let spin_idx = |s: crate::orbital::Spin| match s {
        crate::orbital::Spin::Up => 0usize,
        crate::orbital::Spin::Dn => 1usize,
    };
    let type_idx = |t: OpenType| match t {
        OpenType::Cre => 0usize,
        OpenType::Ann => 1usize,
    };

    let mut ls: StoredFamilies = vec![vec![vec![Vec::new(); d]; 2]; 2];
    let mut rs: StoredFamilies = vec![vec![vec![Vec::new(); d]; 2]; 2];
    for &t in OPEN_TYPES.iter() {
        for &s in spins.iter() {
            for o in 0..d - 1 {
                let mut by_bond: Vec<Option<Sketch<f64>>> = vec![None; d];
                let mut cur = fold_left(base_l.core(o), &p_plain[o], sc_all[o].open(t, s));
                if o + 1 < d {
                    by_bond[o + 1] = Some(cur.clone());
                }
                for b in o + 1..d - 1 {
                    cur = fold_left(base_l.core(b), &cur, sc_all[b].id(t, s));
                    by_bond[b + 1] = Some(cur.clone());
                }
                ls[type_idx(t)][spin_idx(s)][o] = by_bond;
            }
            for c in 1..d {
                let mut by_bond: Vec<Option<Sketch<f64>>> = vec![None; c + 1];
                let mut cur = fold_right(
                    sc_all[c].close(t, s),
                    q_plain[c + 1].as_ref().expect("plain right partial"),
                    base_r.core(c),
                );
                by_bond[c] = Some(cur.clone());
                for b in (1..c).rev() {
                    cur = fold_right(sc_all[b].id(t, s), &cur, base_r.core(b));
                    by_bond[b] = Some(cur.clone());
                }
                rs[type_idx(t)][spin_idx(s)][c] = by_bond;
            }
        }
    }

    let column = |pp: usize, qq: usize, out: &mut Vec<f64>| {
        let mut rd: Vec<Vec<Vec<Option<Sketch<f64>>>>> = vec![vec![vec![None; qq + 1]; 2]; 2];
        let mut ld: Vec<Vec<Vec<Option<Sketch<f64>>>>> = vec![vec![vec![None; d + 1]; 2]; 2];
        for &t in OPEN_TYPES.iter() {
            for &s in spins.iter() {
                let (ti, si) = (type_idx(t), spin_idx(s));
                if qq > 0 {
                    let mut cur = fold_r(
                        sc_all[qq].open(t, s),
                        rs[ti][si][pp][qq + 1]
                            .as_ref()
                            .expect("stored right family"),
                        base_r.core(qq),
                    );
                    rd[ti][si][qq] = Some(cur.clone());
                    for b in (1..qq).rev() {
                        cur = fold_r(x.core(b), &cur, base_r.core(b));
                        rd[ti][si][b] = Some(cur.clone());
                    }
                }
                if pp + 1 < d {
                    let mut cur = fold_left(
                        base_l.core(pp),
                        ls[ti][si][qq][pp].as_ref().expect("stored left family"),
                        sc_all[pp].close(t, s),
                    );
                    ld[ti][si][pp + 1] = Some(cur.clone());
                    for b in pp + 1..d - 1 {
                        cur = fold_left(base_l.core(b), &cur, x.core(b));
                        ld[ti][si][b + 1] = Some(cur.clone());
                    }
                }
            }
        }

        for kc in 0..d {
            let mut comp: Option<SparseCore<f64>> = None;
            for &t in OPEN_TYPES.iter() {
                for &s in spins.iter() {
                    let (ti, si) = (type_idx(t), spin_idx(s));
                    let contrib = if kc < qq {
                        component(
                            &p_plain[kc],
                            x.core(kc),
                            rd[ti][si][kc + 1].as_ref().expect("pair right partial"),
                        )
                    } else if kc == qq {
                        component(
                            &p_plain[kc],
                            sc_all[kc].open(t, s),
                            rs[ti][si][pp][kc + 1]
                                .as_ref()
                                .expect("stored right family"),
                        )
                    } else if kc < pp {
                        component(
                            ls[ti][si][qq][kc].as_ref().expect("stored left family"),
                            sc_all[kc].id(t, s),
                            rs[ti][si][pp][kc + 1]
                                .as_ref()
                                .expect("stored right family"),
                        )
                    } else if kc == pp {
                        component(
                            ls[ti][si][qq][kc].as_ref().expect("stored left family"),
                            sc_all[kc].close(t, s),
                            q_plain[kc + 1].as_ref().expect("plain right partial"),
                        )
                    } else {
                        component(
                            ld[ti][si][kc].as_ref().expect("pair left partial"),
                            x.core(kc),
                            q_plain[kc + 1].as_ref().expect("plain right partial"),
                        )
                    };
                    match comp.as_mut() {
                        None => comp = Some(contrib),
                        Some(acc) => acc.axpy(1.0, &contrib),
                    }
                }
            }
            let mut comp = comp.expect("at least one active spin");
            if kc < d - 1 {
                gauge_complement(&mut comp, base_l.core(kc), space, kc + 1);
            }
            comp.for_each_coeff(|v| out.push(v));
        }
    };

    let check_len = |num_coeffs: &mut Option<usize>, n: usize| match *num_coeffs {
        None => *num_coeffs = Some(n),
        Some(expect) => assert_eq!(
            n, expect,
            "orbital_tangent_matrix: inconsistent column length — a component's block \
             shape drifted between pairs"
        ),
    };
    let mut num_coeffs: Option<usize> = None;
    if par {
        let cols: Vec<Vec<f64>> = pairs
            .par_iter()
            .map(|&(pp, qq)| {
                crate::par::nested_scope(|| {
                    let mut col = Vec::new();
                    column(pp, qq, &mut col);
                    col
                })
            })
            .collect();
        for col in &cols {
            check_len(&mut num_coeffs, col.len());
        }
        theta.reserve_exact(cols.iter().map(Vec::len).sum());
        for col in cols {
            theta.extend_from_slice(&col);
        }
    } else {
        for &(pp, qq) in pairs {
            let col_start = theta.len();
            column(pp, qq, &mut theta);
            let n = theta.len() - col_start;
            if num_coeffs.is_none() {
                theta.reserve_exact(n * (pairs.len() - 1));
            }
            check_len(&mut num_coeffs, n);
        }
    }
    theta
}

/// `T = ΘᵀΘ` — the tangent Gram of every projected direction in one GEMM (the tangent dot IS
/// the Euclidean dot of the canonical coefficients, so this is exact, not an approximation).
/// `theta` is column-major `ncoef × n`; the result is row-major `n × n`, exactly symmetric
/// (upper triangle mirrored). The GEMM is `scalar`'s (faer stays confined there) under the
/// shape gate: `ncoef·n²` multiply-adds clear its floor at any joint-context size, so
/// it threads, and matches the `Par::Seq` product to machine precision rather than bitwise.
pub fn tangent_coeff_gram(theta: &[f64], ncoef: usize, n: usize) -> Vec<f64> {
    assert_eq!(theta.len(), ncoef * n, "tangent_coeff_gram: shape mismatch");
    let par = crate::scalar::gemm_par(n, n, ncoef);
    crate::scalar::gram_colmajor(theta, ncoef, n, par)
}

/// `Θ·V` — the kept-coordinate tangent columns in one GEMM: `theta` column-major `ncoef × n`,
/// `v` row-major `n × n_eff` (the keep-basis layout the joint context stores); result
/// column-major `ncoef × n_eff`, written by the GEMM in place. Same shape gate as
/// [`tangent_coeff_gram`].
pub fn theta_times(theta: &[f64], ncoef: usize, n: usize, v: &[f64], n_eff: usize) -> Vec<f64> {
    assert_eq!(theta.len(), ncoef * n, "theta_times: theta shape mismatch");
    assert_eq!(v.len(), n * n_eff, "theta_times: v shape mismatch");
    let par = crate::scalar::gemm_par(ncoef, n_eff, n);
    crate::scalar::matmul_colmajor_rowmajor(theta, ncoef, n, v, n_eff, par)
}

#[cfg(test)]
mod tests {
    use super::{tangent_coeff_gram, theta_times};

    #[test]
    fn tangent_coeff_gram_matches_hand_3x2() {
        let theta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = tangent_coeff_gram(&theta, 3, 2);
        assert_eq!(t, vec![14.0, 32.0, 32.0, 77.0]);
    }

    #[test]
    fn theta_times_matches_hand_3x2_asymmetric_v() {
        let theta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = [1.0, 10.0, 100.0, 1000.0];
        let out = theta_times(&theta, 3, 2, &v, 2);
        assert_eq!(out, vec![401.0, 502.0, 603.0, 4010.0, 5020.0, 6030.0]);
    }

    #[test]
    fn theta_gemms_threaded_match_serial() {
        use crate::scalar::gemm_par_hook::{Hook, pool};
        use crate::scalar::{gram_colmajor, matmul_colmajor_rowmajor};
        use crate::test_util::seeded;
        use rand::RngExt;
        let (ncoef, n, n_eff) = (2053usize, 41usize, 29usize);
        let mut rng = seeded(0x7E7A_6E44);
        let theta: Vec<f64> = (0..ncoef * n).map(|_| rng.random::<f64>() - 0.5).collect();
        let v: Vec<f64> = (0..n * n_eff).map(|_| rng.random::<f64>() - 0.5).collect();
        let hook = Hook::observed();
        let p4 = pool(4, &hook);
        let (t_par, tv_par) = p4.install(|| {
            (
                tangent_coeff_gram(&theta, ncoef, n),
                theta_times(&theta, ncoef, n, &v, n_eff),
            )
        });
        assert_eq!(
            hook.taken(),
            2,
            "both GEMMs must clear the shape gate (vacuous)"
        );
        let t_seq = gram_colmajor(&theta, ncoef, n, faer::Par::Seq);
        let tv_seq = matmul_colmajor_rowmajor(&theta, ncoef, n, &v, n_eff, faer::Par::Seq);
        for (label, got, want) in [("gram", &t_par, &t_seq), ("theta_times", &tv_par, &tv_seq)] {
            assert_eq!(got.len(), want.len(), "{label}: length");
            let y = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
            let d = got
                .iter()
                .zip(want.iter())
                .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
            let ulps = d / (y * f64::EPSILON);
            println!("{label}: max|Δ| = {d:.3e} = {ulps:.2} ULP of max|y| = {y:.3e}");
            assert!(ulps <= 4.0, "{label}: {ulps:.2} ULP > 4 against Par::Seq");
        }
        for i in 0..n {
            for j in 0..i {
                assert_eq!(
                    t_par[i * n + j].to_bits(),
                    t_par[j * n + i].to_bits(),
                    "Gram symmetry"
                );
            }
        }
    }

    #[test]
    fn parallel_pairs_are_bitwise_the_serial_loop() {
        use super::{orbital_tangent_matrix, orbital_tangent_matrix_with, theta_work};
        use crate::qn::QnSpace;
        use crate::solvers::riemannian::tangent_bases;
        use crate::test_util::{randomized_state, seeded};
        use crate::vector::TtVector;
        struct Restore {
            fork: usize,
            on: bool,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                crate::par::set_par_min_fork_work(self.fork);
                crate::par::set_vector_par(self.on);
            }
        }
        let pool = |n: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("rayon pool")
        };
        let bits_eq = |got: &[f64], want: &[f64], what: &str| {
            assert_eq!(got.len(), want.len(), "{what}: length");
            let bad = got
                .iter()
                .zip(want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(bad, 0, "{what}: {bad} entries differ from the serial loop");
        };
        let _lock = crate::par::lock_par_switches();
        let restore = Restore {
            fork: crate::par::par_min_fork_work(),
            on: crate::par::set_vector_par(true),
        };
        for (space, seed) in [
            (QnSpace::new(6, 3, 2), 0x7E7A_0001u64),
            (QnSpace::new(5, 3, 0), 0x7E7A_0002),
        ] {
            let x: TtVector<f64> = randomized_state(space, 1, 3, &mut seeded(seed));
            let (bl, br) = tangent_bases(&x);
            let d = space.d;
            let pairs: Vec<(usize, usize)> =
                (0..d).flat_map(|p| (0..p).map(move |q| (p, q))).collect();
            let want = pool(1).install(|| orbital_tangent_matrix_with(&bl, &br, &x, &pairs, false));
            assert!(
                want.iter().any(|v| *v != 0.0),
                "{space:?}: all-zero reference"
            );
            for n in [1, 4, 16] {
                let got =
                    pool(n).install(|| orbital_tangent_matrix_with(&bl, &br, &x, &pairs, true));
                bits_eq(&got, &want, &format!("{space:?} par pool {n}"));
            }
            crate::par::set_par_min_fork_work(0);
            let got = pool(4).install(|| {
                assert!(
                    crate::par::par_driver_fork(theta_work(&x, pairs.len())),
                    "{space:?}: the floor-0 arm would run the serial loop (vacuous)"
                );
                orbital_tangent_matrix(&bl, &br, &x, &pairs)
            });
            bits_eq(&got, &want, &format!("{space:?} public, pool 4, floor 0"));
            crate::par::set_par_min_fork_work(restore.fork);
        }
    }

    #[test]
    #[should_panic(expected = "tangent_coeff_gram: shape mismatch")]
    fn tangent_coeff_gram_rejects_wrong_shape() {
        let _ = tangent_coeff_gram(&[1.0; 5], 3, 2);
    }

    #[test]
    #[should_panic(expected = "theta_times: v shape mismatch")]
    fn theta_times_rejects_wrong_v_shape() {
        let _ = theta_times(&[1.0; 6], 3, 2, &[1.0; 3], 2);
    }
}
