// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::solvers::InnerProductVector;
use crate::su2::ops::dot;
use crate::su2::tangent::Su2TtTangent;
use crate::su2::vector::Su2TtVector;

use super::{Su2JointContext, Su2JointVector, chol_back, chol_flat, chol_forward};

#[derive(Clone)]
pub(crate) struct Su2JointWhite {
    pub(crate) xi: Su2TtTangent<f64>,
    pub(crate) c: Vec<f64>,
}

impl InnerProductVector for Su2JointWhite {
    type Scalar = f64;

    fn dot(&self, other: &Self) -> f64 {
        debug_assert_eq!(self.c.len(), other.c.len());
        self.xi.dot(&other.xi) + self.c.iter().zip(&other.c).map(|(a, b)| a * b).sum::<f64>()
    }

    fn norm(&self) -> f64 {
        self.dot(self).max(0.0).sqrt()
    }

    fn axpy(&mut self, a: f64, other: &Self) {
        debug_assert_eq!(self.c.len(), other.c.len());
        self.xi.axpy(a, &other.xi);
        for (s, &o) in self.c.iter_mut().zip(other.c.iter()) {
            *s += a * o;
        }
    }

    fn scale(&mut self, a: f64) {
        self.xi.scale_mut(a);
        for s in self.c.iter_mut() {
            *s *= a;
        }
    }

    fn for_each_coeff_mut<F: FnMut(&mut f64)>(&mut self, mut f: F) {
        self.xi.for_each_coeff_mut(&mut f);
        for s in self.c.iter_mut() {
            f(s);
        }
    }

    fn zip_coeffs_mut<F: FnMut(&mut f64, f64)>(&mut self, other: &Self, mut f: F) {
        debug_assert_eq!(self.c.len(), other.c.len());
        self.xi.zip_coeffs_mut(&other.xi, &mut f);
        for (s, &o) in self.c.iter_mut().zip(other.c.iter()) {
            f(s, o);
        }
    }

    fn coeff(&self, idx: usize) -> f64 {
        let nt = self.xi.num_coeffs();
        if idx < nt {
            self.xi.coeff(idx)
        } else {
            self.c[idx - nt]
        }
    }

    fn set_coeff(&mut self, idx: usize, val: f64) {
        let nt = self.xi.num_coeffs();
        if idx < nt {
            self.xi.set_coeff(idx, val);
        } else {
            self.c[idx - nt] = val;
        }
    }

    fn num_coeffs(&self) -> usize {
        self.xi.num_coeffs() + self.c.len()
    }
}

impl Su2JointContext<'_> {
    pub(crate) fn to_white(&self, v: &Su2JointVector) -> Su2JointWhite {
        self.to_white_with(&self.nv_chol, v)
    }

    pub(crate) fn to_white_with(&self, l: &[f64], v: &Su2JointVector) -> Su2JointWhite {
        let n = self.n_eff();
        debug_assert_eq!(v.c.len(), n);
        let c: Vec<f64> = (0..n)
            .map(|i| (i..n).map(|k| l[k * n + i] * v.c[k]).sum::<f64>())
            .collect();
        let mut xi = v.xi.clone();
        for (&cj, tv) in v.c.iter().zip(self.spec.tau_v.iter()) {
            xi.axpy(cj, tv);
        }
        Su2JointWhite { xi, c }
    }

    pub(crate) fn to_joint(&self, y: &Su2JointWhite) -> Su2JointVector {
        self.to_joint_with(&self.nv_chol, y)
    }

    pub(crate) fn to_joint_with(&self, l: &[f64], y: &Su2JointWhite) -> Su2JointVector {
        let n = self.n_eff();
        debug_assert_eq!(y.c.len(), n);
        let c = chol_back(l, &y.c, n);
        let mut xi = y.xi.clone();
        for (&cj, tv) in c.iter().zip(self.spec.tau_v.iter()) {
            xi.axpy(-cj, tv);
        }
        Su2JointVector {
            xi,
            c,
            spec: self.spec.clone(),
        }
    }

    pub(crate) fn white_covector(&self, f_xi: Su2TtTangent<f64>, f_c: Vec<f64>) -> Su2JointWhite {
        self.white_covector_with(&self.nv_chol, f_xi, f_c)
    }

    pub(crate) fn white_covector_with(
        &self,
        l: &[f64],
        f_xi: Su2TtTangent<f64>,
        f_c: Vec<f64>,
    ) -> Su2JointWhite {
        let n = self.n_eff();
        assert_eq!(
            f_c.len(),
            n,
            "white_covector: F_c must be in kept coordinates"
        );
        let rhs: Vec<f64> = (0..n)
            .map(|j| f_c[j] - self.spec.tau_v[j].dot(&f_xi))
            .collect();
        Su2JointWhite {
            xi: f_xi,
            c: chol_forward(l, &rhs, n),
        }
    }

    fn ab_white(&self, y: &Su2JointWhite) -> (Su2JointWhite, Su2JointWhite) {
        let ((fa_xi, fa_c), (fb_xi, fb_c)) = self.ab_covectors(&self.to_joint(y));
        (
            self.white_covector(fa_xi, fa_c),
            self.white_covector(fb_xi, fb_c),
        )
    }

    pub(crate) fn hpp_white_direct(&self, y: &Su2JointWhite) -> Su2JointWhite {
        let (mut a, b) = self.ab_white(y);
        a.axpy(1.0, &b);
        a
    }

    pub(crate) fn hmm_white_direct(&self, y: &Su2JointWhite) -> Su2JointWhite {
        let (mut a, b) = self.ab_white(y);
        a.axpy(-1.0, &b);
        a
    }

    pub(crate) fn accurate_metric_chol(&self) -> Option<(Vec<f64>, f64)> {
        let n = self.n_eff();
        let parts: Vec<Su2TtVector<f64>> = (0..n)
            .map(|j| {
                let mut e = vec![0.0; n];
                e[j] = 1.0;
                let im = self.kappa_images(&self.v_times(&e))?;
                let mut p = &im.kx + &self.spec.tau_v[j].to_vector();
                p.left_orthogonalize(true);
                Some(p)
            })
            .collect::<Option<_>>()?;
        let mut m = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let d = dot(&parts[i], &parts[j]);
                m[i * n + j] = d;
                m[j * n + i] = d;
            }
        }
        let l = chol_flat(&m, n)?;
        Some((l, metric_deviation(&m, &self.nv_chol, &self.orb_kept, n)))
    }
}

pub(crate) struct Su2CondensedWhite {
    r_plus: Vec<Su2TtTangent<f64>>,
    r_minus: Vec<Su2TtTangent<f64>>,
    c_plus: Vec<f64>,
    c_minus: Vec<f64>,
    pub(crate) asym_a: f64,
    pub(crate) asym_b: f64,
    pub(crate) n_pre: usize,
    pub(crate) l: Vec<f64>,
}

#[allow(clippy::type_complexity)]
fn precompute(
    ctx: &Su2JointContext<'_>,
    l: &[f64],
) -> (
    Vec<Su2TtTangent<f64>>,
    Vec<Su2TtTangent<f64>>,
    Vec<f64>,
    Vec<f64>,
    f64,
    f64,
) {
    let n = ctx.n_eff();
    let (mut ra, mut rb) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut ca, mut cb) = (vec![0.0f64; n * n], vec![0.0f64; n * n]);
    for j in 0..n {
        let mut c = vec![0.0; n];
        c[j] = 1.0;
        let v = ctx.to_joint_with(
            l,
            &Su2JointWhite {
                xi: ctx.zero_tangent(),
                c,
            },
        );
        let ((fa_xi, fa_c), (fb_xi, fb_c)) = ctx.ab_covectors(&v);
        let wa = ctx.white_covector_with(l, fa_xi, fa_c);
        let wb = ctx.white_covector_with(l, fb_xi, fb_c);
        for i in 0..n {
            ca[i * n + j] = wa.c[i];
            cb[i * n + j] = wb.c[i];
        }
        ra.push(wa.xi);
        rb.push(wb.xi);
    }
    let (asym_a, sa) = symmetrized(&ca, n);
    let (asym_b, sb) = symmetrized(&cb, n);
    (ra, rb, sa, sb, asym_a, asym_b)
}

impl Su2CondensedWhite {
    pub(crate) fn build(ctx: &Su2JointContext<'_>, l: &[f64]) -> Self {
        let n = ctx.n_eff();
        let (ra, rb, sa, sb, asym_a, asym_b) = precompute(ctx, l);
        let combine = |sign: f64| -> Vec<Su2TtTangent<f64>> {
            ra.iter()
                .zip(&rb)
                .map(|(a, b)| {
                    let mut t = a.clone();
                    t.axpy(sign, b);
                    t
                })
                .collect()
        };
        Su2CondensedWhite {
            r_plus: combine(1.0),
            r_minus: combine(-1.0),
            c_plus: sa.iter().zip(&sb).map(|(a, b)| a + b).collect(),
            c_minus: sa.iter().zip(&sb).map(|(a, b)| a - b).collect(),
            asym_a,
            asym_b,
            n_pre: n,
            l: l.to_vec(),
        }
    }

    pub(crate) fn to_white(&self, ctx: &Su2JointContext<'_>, v: &Su2JointVector) -> Su2JointWhite {
        ctx.to_white_with(&self.l, v)
    }

    pub(crate) fn apply(
        &self,
        ctx: &Su2JointContext<'_>,
        y: &Su2JointWhite,
        sign: f64,
    ) -> Su2JointWhite {
        let (r, cm) = if sign > 0.0 {
            (&self.r_plus, &self.c_plus)
        } else {
            (&self.r_minus, &self.c_minus)
        };
        let n = r.len();
        debug_assert_eq!(y.c.len(), n);
        let mut xi = ctx.ctx.a_apply_fused(&y.xi);
        xi.axpy(sign, &ctx.ctx.b_apply(&y.xi));
        for (rj, &cj) in r.iter().zip(&y.c) {
            xi.axpy(cj, rj);
        }
        let c = (0..n)
            .map(|i| {
                r[i].dot(&y.xi)
                    + cm[i * n..(i + 1) * n]
                        .iter()
                        .zip(&y.c)
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
            })
            .collect();
        Su2JointWhite { xi, c }
    }
}

fn metric_deviation(m: &[f64], l: &[f64], mu: &[f64], n: usize) -> f64 {
    let mut dev = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let llt: f64 = (0..=i.min(j)).map(|k| l[i * n + k] * l[j * n + k]).sum();
            dev = dev.max((m[i * n + j] - llt).abs() / (mu[i] * mu[j]).sqrt());
        }
    }
    dev
}

fn symmetrized(m: &[f64], n: usize) -> (f64, Vec<f64>) {
    let mut s = vec![0.0f64; n * n];
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (m[i * n + j], m[j * n + i]);
            s[i * n + j] = 0.5 * (a + b);
            num += (a - b) * (a - b);
            den += a * a;
        }
    }
    (num.sqrt() / den.sqrt().max(f64::MIN_POSITIVE), s)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::solvers::joint::{JointOpts, ZRoute};
    use crate::su2::hamiltonian::Su2SparseHamiltonian;
    use crate::su2::hess::deflate_radial;
    use crate::su2::joint::su2_joint_context;
    use crate::su2::qn::Su2Space;
    use crate::su2::riemannian::normalize_mut;
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{generic_truncated_tv, randomized_su2_state, seeded};
    use rand::RngExt;

    fn point(seed: u64) -> (Su2SparseHamiltonian<f64>, Su2TtVector<f64>) {
        let d = 6;
        let space = Su2Space::new(d, 3, 1);
        let (t, v) = generic_truncated_tv::<f64>(d);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let mut x: Su2TtVector<f64> = randomized_su2_state(space, 1, 2, &mut seeded(seed));
        normalize_mut(&mut x);
        (h, x)
    }

    fn opts() -> JointOpts {
        JointOpts {
            tt_grad_tol: f64::INFINITY,
            orb_floor: 1e-8,
            pairs: None,
            z_route: ZRoute::Split,
            b_general: false,
        }
    }

    fn random_joint(ctx: &Su2JointContext<'_>, seed: u64) -> Su2JointVector {
        let mut rng = seeded(seed);
        let (bl, br) = ctx.bases();
        let mut d: Su2TtVector<f64> = randomized_su2_state(bl.space, 1, 2, &mut rng);
        normalize_mut(&mut d);
        let xi = Su2TtTangent::project(bl, br, &d);
        let mut v = Su2JointVector::from_tangent(ctx.spec(), xi);
        for c in v.c.iter_mut() {
            *c = rng.random_range(-1.0..1.0);
        }
        v
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / a.abs().max(b.abs()).max(1e-300)
    }

    fn joint_dist(a: &Su2JointVector, b: &Su2JointVector) -> f64 {
        let mut d = a.clone();
        d.axpy(-1.0, b);
        let dx = d.xi.norm();
        let dc = d.c.iter().map(|x| x * x).sum::<f64>().sqrt();
        let s = a.xi.norm() + a.c.iter().map(|x| x * x).sum::<f64>().sqrt();
        (dx + dc) / s
    }

    fn kappa(ctx: &Su2JointContext<'_>) -> f64 {
        let k = ctx.orb_kept();
        k.last().unwrap() / k.first().unwrap()
    }

    const SEEDS: [u64; 2] = [0x5A1E, 0x5A0D];

    #[test]
    fn su2_accurate_metric_matches_the_dense_normal_gram() {
        let ddot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        for seed in SEEDS {
            let (h, x) = point(seed);
            let ctx = su2_joint_context(&h, &x, &opts()).expect("context");
            let n = ctx.n_eff();
            let two_sz = ctx.bases().0.two_s() as i64;
            let dense = |v: &Su2TtVector<f64>| crate::su2::expand::expand(v, two_sz).to_dense();
            let ud: Vec<Vec<f64>> = ctx.u.iter().map(dense).collect();
            let n_orb = ud.len();
            let (tr_d, tr_s) = (0..n_orb).fold((0.0, 0.0), |(p, q), a| {
                (p + ddot(&ud[a], &ud[a]), q + ctx.s[a * n_orb + a])
            });
            assert!(
                (tr_d / tr_s - 1.0).abs() <= 1e-12,
                "seed {seed:#x}: dense/ctx S trace ratio {}",
                tr_d / tr_s
            );
            let parts: Vec<Vec<f64>> = (0..n)
                .map(|j| {
                    let mut p: Vec<f64> = dense(&ctx.spec.tau_v[j].to_vector())
                        .iter()
                        .map(|t| -t)
                        .collect();
                    for (a, ua) in ud.iter().enumerate() {
                        for (pi, ui) in p.iter_mut().zip(ua) {
                            *pi += ctx.v[a * n + j] * ui;
                        }
                    }
                    p
                })
                .collect();
            let md: Vec<f64> = (0..n * n)
                .map(|k| ddot(&parts[k / n], &parts[k % n]))
                .collect();
            let (l2, dev) = ctx.accurate_metric_chol().expect("M′ positive definite");
            let c2 = metric_deviation(&md, &l2, &ctx.orb_kept, n);
            let prod = metric_deviation(&md, &ctx.nv_chol, &ctx.orb_kept, n);
            eprintln!(
                "su2 accurate metric vs dense [seed {seed:#x}, κ {:.1e}] (entry/√μμ): C2 {c2:.2e} | \
                 production {prod:.2e} | C2 vs production {dev:.2e}",
                kappa(&ctx)
            );
            assert!(
                c2 <= 2e-12,
                "seed {seed:#x}: accurate metric off the dense Gram by {c2:.3e}"
            );
        }
    }

    #[test]
    fn su2_white_map_identities() {
        for seed in SEEDS {
            let (h, x) = point(seed);
            let ctx = su2_joint_context(&h, &x, &opts()).expect("context");
            let n = ctx.n_eff();
            assert!(
                n >= 2,
                "seed {seed:#x}: fixture must carry an orbital block (n_eff = {n})"
            );
            let kap = kappa(&ctx);
            let budget = 1e2 * f64::EPSILON * kap.max(1.0);
            let (v1, v2) = (random_joint(&ctx, 1), random_joint(&ctx, 2));
            let (y1, y2) = (ctx.to_white(&v1), ctx.to_white(&v2));
            let (g, e) = (v1.dot(&v2), y1.dot(&y2));
            let back = ctx.to_joint(&y1);
            eprintln!(
                "su2 white map [{seed:#x}]: n_eff {n} κ {kap:.2e} | G-dot vs white dot {:.2e} | \
                 J·J⁻¹ {:.2e}",
                rel(g, e),
                joint_dist(&v1, &back)
            );
            assert!(rel(g, e) <= budget, "seed {seed:#x}: JᵀGJ ≠ I");
            assert!(joint_dist(&v1, &back) <= 1e-13, "seed {seed:#x}: J·J⁻¹ ≠ I");
            let mut dy = ctx.to_white(&ctx.to_joint(&y2));
            dy.axpy(-1.0, &y2);
            assert!(dy.norm() <= 1e-13 * y2.norm(), "seed {seed:#x}: J⁻¹·J ≠ I");

            let ((fa_xi, fa_c), _) = ctx.ab_covectors(&v1);
            let raised = ctx.joint_raise(fa_xi.clone(), fa_c.clone());
            let via_white = ctx.to_joint(&ctx.white_covector(fa_xi, fa_c));
            assert_eq!(
                joint_dist(&raised, &via_white),
                0.0,
                "seed {seed:#x}: J·JᵀF must be the raise operation for operation"
            );
            let hp = ctx.to_white(&ctx.hpp_apply(&v1));
            let mut dp = ctx.hpp_white_direct(&y1);
            dp.axpy(-1.0, &hp);
            let hm = ctx.to_white(&ctx.hmm_apply(&v1));
            let mut dm = ctx.hmm_white_direct(&y1);
            dm.axpy(-1.0, &hm);
            assert!(dp.norm() <= budget * hp.norm(), "seed {seed:#x}: hpp_white");
            assert!(dm.norm() <= budget * hm.norm(), "seed {seed:#x}: hmm_white");

            let radial = ctx.radial().expect("a radial mode at this point");
            let mut a = ctx.to_white(&v2);
            deflate_radial(Some(radial), &mut a.xi);
            let mut b = v2.clone();
            deflate_radial(Some(radial), &mut b.xi);
            assert!(
                joint_dist(&b, &ctx.to_joint(&a)) <= 1e-12,
                "seed {seed:#x}: deflate∘J ≠ J∘deflate"
            );
        }
    }

    #[test]
    fn su2_white_hessian_gram_is_symmetric() {
        for seed in SEEDS {
            let (h, x) = point(seed);
            let ctx = su2_joint_context(&h, &x, &opts()).expect("context");
            let vs: Vec<Su2JointVector> = (10..14).map(|s| random_joint(&ctx, s)).collect();
            let ys: Vec<Su2JointWhite> = vs.iter().map(|v| ctx.to_white(v)).collect();
            let wimg: Vec<Su2JointWhite> = ys.iter().map(|y| ctx.hpp_white_direct(y)).collect();
            let rimg: Vec<Su2JointVector> = vs.iter().map(|v| ctx.hpp_apply(v)).collect();
            let (mut w_asym, mut r_asym, mut scale) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..vs.len() {
                for j in 0..vs.len() {
                    let a = ys[i].dot(&wimg[j]);
                    w_asym = w_asym.max((a - ys[j].dot(&wimg[i])).abs());
                    r_asym = r_asym.max((vs[i].dot(&rimg[j]) - vs[j].dot(&rimg[i])).abs());
                    scale = scale.max(a.abs());
                }
            }
            eprintln!(
                "su2 Gram asymmetry [{seed:#x}] (κ {:.2e}): white {:.2e}, raised {:.2e} of scale \
                 {scale:.2e}",
                kappa(&ctx),
                w_asym / scale,
                r_asym / scale
            );
            assert!(
                w_asym <= 1e-12 * scale,
                "seed {seed:#x}: white Gram asymmetry"
            );
            assert!(
                w_asym <= r_asym.max(1e-14 * scale),
                "seed {seed:#x}: white less symmetric than raised"
            );
        }
    }

    #[test]
    fn su2_condensed_white_equals_direct_and_is_symmetric() {
        for seed in SEEDS {
            let (h, x) = point(seed);
            let ctx = su2_joint_context(&h, &x, &opts()).expect("context");
            let cw = Su2CondensedWhite::build(&ctx, &ctx.nv_chol);
            let max_s = ctx.s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let floor = f64::EPSILON * max_s.max(1.0) / ctx.orb_kept[0];
            let mut worst = 0.0f64;
            for s in 20..24 {
                let y = ctx.to_white(&random_joint(&ctx, s));
                for (sign, direct) in [
                    (1.0, ctx.hpp_white_direct(&y)),
                    (-1.0, ctx.hmm_white_direct(&y)),
                ] {
                    let mut d = cw.apply(&ctx, &y, sign);
                    d.axpy(-1.0, &direct);
                    worst = worst.max(d.norm() / direct.norm());
                }
            }
            let n = ctx.n_eff();
            let mut ys: Vec<Su2JointWhite> = (0..n)
                .map(|j| {
                    let mut c = vec![0.0; n];
                    c[j] = 1.0;
                    Su2JointWhite {
                        xi: ctx.zero_tangent(),
                        c,
                    }
                })
                .collect();
            ys.extend((30..33).map(|s| ctx.to_white(&random_joint(&ctx, s))));
            let (mut ca, mut da, mut scale) = (0.0f64, 0.0f64, 0.0f64);
            for sign in [1.0, -1.0] {
                let ci: Vec<Su2JointWhite> = ys.iter().map(|y| cw.apply(&ctx, y, sign)).collect();
                let di: Vec<Su2JointWhite> = ys
                    .iter()
                    .map(|y| {
                        if sign > 0.0 {
                            ctx.hpp_white_direct(y)
                        } else {
                            ctx.hmm_white_direct(y)
                        }
                    })
                    .collect();
                for i in 0..ys.len() {
                    for j in 0..ys.len() {
                        let a = ys[i].dot(&ci[j]);
                        ca = ca.max((a - ys[j].dot(&ci[i])).abs());
                        da = da.max((ys[i].dot(&di[j]) - ys[j].dot(&di[i])).abs());
                        scale = scale.max(a.abs());
                    }
                }
            }
            eprintln!(
                "su2 condensed [{seed:#x}]: vs direct {worst:.2e} (ε·max|S|/μ_min {floor:.1e}) | Gram \
                 asym condensed {:.2e} direct {:.2e} | C asym A {:.1e} B {:.1e}",
                ca / scale,
                da / scale,
                cw.asym_a,
                cw.asym_b
            );
            assert!(
                worst <= 1e2 * floor.max(f64::EPSILON),
                "seed {seed:#x}: condensed vs direct"
            );
            assert!(
                ca <= 1e-15 * scale,
                "seed {seed:#x}: condensed Gram asymmetry {ca:.3e}"
            );
            assert!(
                cw.asym_a <= 1e-6 && cw.asym_b <= 1e-6,
                "seed {seed:#x}: C asymmetry tripwire"
            );
        }
    }
}
