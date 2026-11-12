// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, Par};

use crate::core::factor::svd_uvt_svals;
use crate::core::tangent::TtTangent;
use crate::core::weingarten::BondGauge;
use crate::pointgroup::Irrep;
use crate::qn::Sector;
use crate::scalar::Scalar;

#[derive(Clone)]
struct NullBlock<T: Scalar> {
    k: usize,
    t: Sector,
    vn: Mat<T>,
    cut: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BondDeflation {
    pub(crate) bond: usize,
    pub(crate) total: usize,
    pub(crate) deflated: usize,
    pub(crate) smallest_kept: f64,
    pub(crate) largest_deflated: f64,
}

#[derive(Clone)]
pub(crate) struct ColumnDeflation<T: Scalar> {
    null: Vec<NullBlock<T>>,
    bonds: Vec<BondDeflation>,
}

impl<T: Scalar> ColumnDeflation<T> {
    pub(crate) fn from_gauges(gauges: &[BondGauge<T>], floor: f64) -> Self {
        let mut null = Vec::new();
        let mut bonds = Vec::with_capacity(gauges.len());
        for g in gauges {
            let mut rep = BondDeflation {
                bond: g.k,
                total: 0,
                deflated: 0,
                smallest_kept: f64::INFINITY,
                largest_deflated: 0.0,
            };
            for t in g.r.sectors() {
                let Some(blk) = g.r.block(t) else { continue };
                if blk.nrows() == 0 || blk.ncols() == 0 {
                    continue;
                }
                let (_w, sv, vt) = svd_uvt_svals(blk);
                rep.total += sv.len();
                let idx: Vec<usize> = (0..sv.len()).filter(|&i| sv[i] < floor).collect();
                for (i, &s) in sv.iter().enumerate() {
                    if idx.contains(&i) {
                        rep.largest_deflated = rep.largest_deflated.max(s);
                    } else {
                        rep.smallest_kept = rep.smallest_kept.min(s);
                    }
                }
                if idx.is_empty() {
                    continue;
                }
                rep.deflated += idx.len();
                let vn = Mat::from_fn(vt.ncols(), idx.len(), |r, c| vt.get(idx[c], r).conj_s());
                null.push(NullBlock {
                    k: g.k - 1,
                    t,
                    vn,
                    cut: vec![1.0; idx.len()],
                });
            }
            bonds.push(rep);
        }
        ColumnDeflation { null, bonds }
    }

    pub(crate) fn from_gauges_weighted(gauges: &[BondGauge<T>], w: impl Fn(f64) -> f64) -> Self {
        let mut null = Vec::new();
        let mut bonds = Vec::with_capacity(gauges.len());
        for g in gauges {
            let mut rep = BondDeflation {
                bond: g.k,
                total: 0,
                deflated: 0,
                smallest_kept: f64::INFINITY,
                largest_deflated: 0.0,
            };
            for t in g.r.sectors() {
                let Some(blk) = g.r.block(t) else { continue };
                if blk.nrows() == 0 || blk.ncols() == 0 {
                    continue;
                }
                let (_w, sv, vt) = svd_uvt_svals(blk);
                rep.total += sv.len();
                let idx: Vec<usize> = (0..sv.len()).filter(|&i| 1.0 - w(sv[i]) > 1e-15).collect();
                if idx.is_empty() {
                    continue;
                }
                rep.deflated += idx.len();
                let vn = Mat::from_fn(vt.ncols(), idx.len(), |r, c| vt.get(idx[c], r).conj_s());
                let cut = idx.iter().map(|&i| 1.0 - w(sv[i])).collect();
                null.push(NullBlock {
                    k: g.k - 1,
                    t,
                    vn,
                    cut,
                });
            }
            bonds.push(rep);
        }
        ColumnDeflation { null, bonds }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.null.is_empty()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn bonds(&self) -> &[BondDeflation] {
        &self.bonds
    }

    pub(crate) fn apply(&self, xi: &mut TtTangent<T>) {
        let phi = xi.phi();
        self.apply_at(xi, phi);
    }

    #[cfg(test)]
    pub(crate) fn apply_unrelabeled(&self, xi: &mut TtTangent<T>) {
        self.apply_at(xi, Irrep::IDENTITY);
    }

    #[inline]
    fn label(t: Sector, label_phi: Irrep) -> Sector {
        Sector::with_irrep(t.nup, t.ndn, t.irrep.fuse(label_phi))
    }

    fn apply_at(&self, xi: &mut TtTangent<T>, label_phi: Irrep) {
        for nb in &self.null {
            let s = Self::label(nb.t, label_phi);
            let comp = &mut xi.components[nb.k];
            let mut c = comp.vertical_unfolding(s);
            if c.nrows() == 0 || c.ncols() == 0 {
                continue;
            }
            debug_assert_eq!(
                c.ncols(),
                nb.vn.nrows(),
                "ColumnDeflation: component {} at column label {s:?} carries {} columns, but the \
                 null block (x-sector {:?}) has {} right Schmidt coordinates — the tangent was \
                 built at another base point, or the label map is wrong",
                nb.k,
                c.ncols(),
                nb.t,
                nb.vn.nrows()
            );
            let mut cv = Mat::<T>::zeros(c.nrows(), nb.vn.ncols());
            matmul(
                cv.as_mut(),
                Accum::Replace,
                c.as_ref(),
                nb.vn.as_ref(),
                T::one(),
                Par::Seq,
            );
            if nb.cut.iter().any(|&f| f != 1.0) {
                for (j, &f) in nb.cut.iter().enumerate() {
                    for i in 0..cv.nrows() {
                        let x = cv[(i, j)];
                        cv[(i, j)] = x * T::from_f64(f);
                    }
                }
            }
            matmul(
                c.as_mut(),
                Accum::Add,
                cv.as_ref(),
                nb.vn.as_ref().adjoint(),
                T::from_f64(-1.0),
                Par::Seq,
            );
            comp.scatter_vertical(s, c.as_ref());
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn removed_dim(&self, template: &TtTangent<T>) -> usize {
        let phi = template.phi();
        self.null
            .iter()
            .map(|nb| {
                let s = Self::label(nb.t, phi);
                let rows = template.components[nb.k].vertical_unfolding(s).nrows();
                rows.saturating_sub(template.r[nb.k + 1][s]) * nb.vn.ncols()
            })
            .sum()
    }

    #[allow(dead_code)]
    pub(crate) fn null_weights(&self, xi: &TtTangent<T>) -> Vec<f64> {
        let mut w = vec![0.0f64; self.bonds.len()];
        let phi = xi.phi();
        for nb in &self.null {
            let c = xi.components[nb.k].vertical_unfolding(Self::label(nb.t, phi));
            if c.nrows() == 0 || c.ncols() == 0 {
                continue;
            }
            let mut cv = Mat::<T>::zeros(c.nrows(), nb.vn.ncols());
            matmul(
                cv.as_mut(),
                Accum::Replace,
                c.as_ref(),
                nb.vn.as_ref(),
                T::one(),
                Par::Seq,
            );
            let mut acc = 0.0f64;
            for j in 0..cv.ncols() {
                for i in 0..cv.nrows() {
                    acc += cv.get(i, j).abs_s().powi(2);
                }
            }
            w[nb.k] += acc;
        }
        let n2 = xi.norm().powi(2).max(f64::MIN_POSITIVE);
        w.iter_mut().for_each(|x| *x /= n2);
        w
    }
}

#[cfg(test)]
pub(crate) mod oracle {
    use super::ColumnDeflation;
    use crate::core::factor::{dense_eigen_general, dense_self_adjoint_smallest};
    use crate::core::tangent::TtTangent;

    pub(crate) fn dense_projector(p: &ColumnDeflation<f64>, basis: &[TtTangent<f64>]) -> Vec<f64> {
        let n = basis.len();
        let pb: Vec<TtTangent<f64>> = basis
            .iter()
            .map(|b| {
                let mut v = b.clone();
                p.apply(&mut v);
                v
            })
            .collect();
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                m[i * n + j] = basis[i].dot(&pb[j]);
            }
        }
        m
    }

    #[allow(dead_code)]
    pub(crate) fn min_eig(m: &[f64], n: usize) -> f64 {
        dense_self_adjoint_smallest::<f64>(m, n, 0).0[0]
    }

    #[allow(dead_code)]
    pub(crate) fn compress(m: &[f64], n: usize, q: &[Vec<f64>]) -> Vec<f64> {
        let g = q.len();
        let mut out = vec![0.0; g * g];
        for a in 0..g {
            let mq: Vec<f64> = (0..n)
                .map(|i| (0..n).map(|j| m[i * n + j] * q[a][j]).sum())
                .collect();
            for b in 0..g {
                out[b * g + a] = (0..n).map(|i| q[b][i] * mq[i]).sum();
            }
        }
        out
    }

    pub(crate) fn product_spectrum(
        r: &[f64],
        w: &[f64],
        dim: usize,
    ) -> (Vec<f64>, Vec<(f64, f64)>) {
        let mut pm = vec![0.0; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                pm[i * dim + j] = (0..dim).map(|l| r[i * dim + l] * w[l * dim + j]).sum();
            }
        }
        let (z, _) = dense_eigen_general::<f64>(&pm, dim);
        let scale = z.iter().fold(0.0f64, |acc, c| acc.max(c.norm()));
        let mut real: Vec<f64> = z
            .iter()
            .filter(|c| c.im.abs() <= 1e-10 * scale)
            .map(|c| c.re)
            .collect();
        real.sort_by(f64::total_cmp);
        let mut cplx: Vec<(f64, f64)> = z
            .iter()
            .filter(|c| c.im > 1e-10 * scale)
            .map(|c| (c.re, c.im))
            .collect();
        cplx.sort_by(|u, v| u.0.total_cmp(&v.0));
        (real, cplx)
    }

    pub(crate) fn radius(re: &[f64], cx: &[(f64, f64)]) -> f64 {
        re.iter()
            .map(|w| w.abs())
            .chain(cx.iter().map(|&(u, v)| u.hypot(v)))
            .fold(0.0f64, f64::max)
    }

    pub(crate) fn congruence(d: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mul = |x: &[f64], y: &[f64]| {
            let mut z = vec![0.0; n * n];
            for i in 0..n {
                for l in 0..n {
                    let xil = x[i * n + l];
                    if xil != 0.0 {
                        for j in 0..n {
                            z[i * n + j] += xil * y[l * n + j];
                        }
                    }
                }
            }
            z
        };
        mul(&mul(d, b), d)
    }

    pub(crate) fn max_abs(a: &[f64]) -> f64 {
        a.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    }

    pub(crate) fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "max_abs_diff: shapes");
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
    }

    pub(crate) fn asym(a: &[f64], n: usize) -> f64 {
        let mut m = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                m = m.max((a[i * n + j] - a[j * n + i]).abs());
            }
        }
        m
    }
}

#[cfg(test)]
mod reg_tests {
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    use super::oracle::{asym, dense_projector};
    use super::*;
    use crate::core::border_fixture::{BOND, base_state, sigma_min, with_small_schmidt};
    use crate::core::factor::dense_self_adjoint_smallest;
    use crate::core::tangent::{tangent_basis, tangent_dim};
    use crate::core::weingarten::{FloorPolicy, bond_gauges};
    use crate::solvers::riemannian::tangent_bases;
    use crate::vector::tt_randn;

    fn deflated(p: &ColumnDeflation<f64>) -> usize {
        p.bonds().iter().map(|r| r.deflated).sum()
    }

    #[test]
    fn cr_weighted_projector_is_symmetric_psd_and_counted() {
        let x0 = base_state();
        let s0 = sigma_min(&x0);
        let x = with_small_schmidt(&x0, BOND, 1e-4);
        let (bl, br) = tangent_bases(&x);
        let gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        let e = s0;
        let w = |s: f64| s * s / (s * s + e * e);
        let d = ColumnDeflation::from_gauges_weighted(&gauges, w);
        let basis = tangent_basis(&bl, &br, false);
        let n = basis.len();
        assert_eq!(n, tangent_dim(&bl));
        let m = dense_projector(&d, &basis);
        let sym = asym(&m, n);
        let mut leak = 0.0f64;
        for (j, b) in basis.iter().enumerate() {
            let mut v = b.clone();
            d.apply(&mut v);
            let mut r = v.clone();
            for (i, bi) in basis.iter().enumerate() {
                r.axpy(-m[i * n + j], bi);
            }
            leak = leak.max(r.norm());
        }
        let ev = dense_self_adjoint_smallest::<f64>(&m, n, 0).0;
        let mut cut_count = 0.0f64;
        for g in &gauges {
            for t in g.r.sectors() {
                let Some(blk) = g.r.block(t) else { continue };
                if blk.nrows() == 0 || blk.ncols() == 0 {
                    continue;
                }
                let (_w, sv, _vt) = svd_uvt_svals(blk);
                let c = basis[0].components[g.k - 1].vertical_unfolding(t);
                let room = c.nrows().saturating_sub(c.ncols());
                let cut: f64 = sv.iter().map(|&s| 1.0 - w(s)).filter(|&c| c > 1e-15).sum();
                cut_count += room as f64 * cut;
            }
        }
        let tr: f64 = (0..n).map(|i| m[i * n + i]).sum();
        let rad0 = TtTangent::project(&bl, &br, &bl);
        let mut rad = rad0.clone();
        d.apply(&mut rad);
        rad.axpy(-1.0, &rad0);
        let radial = rad.norm() / rad0.norm();
        println!(
            "T1: n = {n}, ε = {e:.3e}, trace {tr:.12} vs grid count {:.12}, spectrum [{:.3e}, \
             {:.3e}], sym {sym:.1e}, leak {leak:.1e}, radial moved {radial:.1e}",
            n as f64 - cut_count,
            ev[0],
            ev[n - 1]
        );
        assert!(sym <= 1e-12 && leak <= 1e-12, "sym {sym:e}, leak {leak:e}");
        assert!(
            ev[0] >= -1e-12 && ev[n - 1] <= 1.0 + 1e-12,
            "D's spectrum must lie in [0, 1]: [{}, {}]",
            ev[0],
            ev[n - 1]
        );
        assert!(
            (tr - (n as f64 - cut_count)).abs() <= 1e-9,
            "trace {tr} vs the grid count {}",
            n as f64 - cut_count
        );
        assert!(radial <= 1e-14, "the radial mode moved by {radial:e}");
        assert!(
            cut_count > 1.0,
            "the Tikhonov weights removed only {cut_count}"
        );

        let f = (s0 * 1e-4).sqrt();
        let hard = ColumnDeflation::from_gauges(&gauges, f);
        let ind = ColumnDeflation::from_gauges_weighted(&gauges, |s| if s < f { 0.0 } else { 1.0 });
        assert!(!hard.is_identity() && !ind.is_identity());
        assert_eq!(
            deflated(&hard),
            1,
            "the fixture floor cuts the planted direction only"
        );
        assert_eq!(deflated(&ind), deflated(&hard));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC1A7);
        let mut moved = 0.0f64;
        for _ in 0..3 {
            let xi = TtTangent::project(&bl, &br, &tt_randn(x.space, &x.ranks, &mut rng));
            let (mut u, mut v) = (xi.clone(), xi.clone());
            hard.apply(&mut u);
            ind.apply(&mut v);
            for i in 0..u.num_coeffs() {
                assert_eq!(
                    u.coeff(i).to_bits(),
                    v.coeff(i).to_bits(),
                    "coefficient {i}: the indicator weight is not bitwise the hard projector"
                );
            }
            let mut dlt = u.clone();
            dlt.axpy(-1.0, &xi);
            moved = moved.max(dlt.norm() / xi.norm());
        }
        assert!(
            moved > 1e-6,
            "non-vacuity: the projector moved nothing ({moved:e})"
        );
    }

    fn col_norm(c: &Mat<f64>, v: &[f64]) -> f64 {
        (0..c.nrows())
            .map(|i| {
                (0..c.ncols())
                    .map(|j| c[(i, j)] * v[j])
                    .sum::<f64>()
                    .powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    #[test]
    fn cr_projector_reads_the_right_schmidt_vectors_not_the_left() {
        let x0 = base_state();
        let f = (sigma_min(&x0) * 1e-4).sqrt();
        let x = with_small_schmidt(&x0, BOND, 1e-4);
        let (bl, br) = tangent_bases(&x);
        let mut gauges = bond_gauges(&bl, &br, FloorPolicy::default_hard());
        let g = &mut gauges[BOND - 1];
        let t = g
            .sigma
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .min_by(|a, b| {
                let ma = a.1.iter().copied().fold(f64::INFINITY, f64::min);
                let mb = b.1.iter().copied().fold(f64::INFINITY, f64::min);
                ma.total_cmp(&mb)
            })
            .expect("fixture: a planted sector")
            .0;
        let (_, sv, _) = svd_uvt_svals(g.r.block(t).expect("fixture: the planted block"));
        let n = sv.len();
        assert!(n >= 2, "fixture: the planted sector has rank ≥ 2");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x05EC_70B5);
        let mut gauss =
            |_: usize, _: usize| rand::RngExt::sample(&mut rng, rand_distr::StandardNormal);
        let g1 = Mat::<f64>::from_fn(n, n, &mut gauss);
        let g2 = Mat::<f64>::from_fn(n, n, &mut gauss);
        let (wq, _, _) = svd_uvt_svals(g1.as_ref());
        let (_, _, vqt) = svd_uvt_svals(g2.as_ref());
        let m = Mat::<f64>::from_fn(n, n, |i, j| {
            (0..n)
                .map(|q| wq[(i, q)] * sv[q] * vqt[(q, j)])
                .sum::<f64>()
        });
        g.r.block_mut(t)
            .expect("the planted block")
            .copy_from(m.as_ref());
        let w_n: Vec<f64> = (0..n).map(|i| wq[(i, n - 1)]).collect();
        let v_n: Vec<f64> = (0..n).map(|j| vqt[(n - 1, j)]).collect();
        let overlap = w_n.iter().zip(&v_n).map(|(a, b)| a * b).sum::<f64>().abs();
        let p = ColumnDeflation::from_gauges(&gauges, f);
        assert_eq!(
            deflated(&p),
            1,
            "the floor still cuts exactly the planted direction"
        );
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x05EC_70B6);
        let (mut worst_v, mut least_w) = (0.0f64, f64::INFINITY);
        for _ in 0..3 {
            let xi = TtTangent::project(&bl, &br, &tt_randn(x.space, &x.ranks, &mut rng));
            let c0 = xi.components[BOND - 1].vertical_unfolding(t);
            let scale = c0.norm_l2();
            assert!(
                col_norm(&c0, &v_n) >= 1e-3 * scale,
                "premise: the seed has weight along v_n"
            );
            let mut y = xi.clone();
            p.apply(&mut y);
            let c = y.components[BOND - 1].vertical_unfolding(t);
            worst_v = worst_v.max(col_norm(&c, &v_n) / scale);
            least_w = least_w.min(col_norm(&c, &w_n) / scale);
        }
        println!(
            "T-b: n = {n}, |⟨w_n, v_n⟩| = {overlap:.3}, after apply max ‖c·v_n‖/‖c‖ = {worst_v:.2e}, \
             min ‖c·w_n‖/‖c‖ = {least_w:.2e}"
        );
        assert!(
            overlap <= 0.9,
            "premise: W and V differ (|⟨w_n, v_n⟩| = {overlap})"
        );
        assert!(
            worst_v <= 1e-13,
            "the touched slot keeps weight along the RIGHT Schmidt vector ({worst_v:e})"
        );
        assert!(
            least_w >= 1e-3,
            "non-vacuity: the slot lost its weight along the LEFT vector too ({least_w:e}) — a \
             W-built projector would look like this"
        );
    }
}
