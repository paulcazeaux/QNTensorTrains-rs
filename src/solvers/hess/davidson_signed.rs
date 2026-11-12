// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::factor::{SignedRitzPath, dense_self_adjoint_smallest, signed_ritz};
use crate::scalar::{C64, Scalar};
use crate::solvers::InnerProductVector;

use super::davidson::{
    PushOutcome, SymplecticDavidsonOpts, push_core, recombine, resolve_max_subspace,
    survival_guard_trips, vector_budget, w_restart_compact,
};
use super::lanczos::{
    ComplexRitz, IndefiniteReport, MetricEvent, NegativeWitness, SympDiagnostics,
    SymplecticLowestInfo,
};
use super::lobpcg::{gram_matrix_symmetrized, rr_matrix_symmetrized};

pub(crate) struct ComplexRitzOut<V> {
    pub(crate) theta: C64,
    pub(crate) residual: f64,
    pub(crate) x_re: V,
    pub(crate) x_im: V,
    pub(crate) w_re: V,
    pub(crate) w_im: V,
}

pub(crate) struct SignedDavidsonOut<V> {
    pub(crate) omega2: Vec<f64>,
    pub(crate) vectors: Vec<V>,
    pub(crate) w_vectors: Vec<V>,
    pub(crate) complex: Vec<ComplexRitzOut<V>>,
    pub(crate) info: SymplecticLowestInfo,
}

#[cfg(test)]
impl<V> SignedDavidsonOut<V> {
    fn metric_indefinite(&self) -> bool {
        self.info.indefinite.is_some()
    }
    fn signature(&self) -> &[i8] {
        &self.info.signature
    }
    fn events(&self) -> &[MetricEvent] {
        self.info.indefinite.as_ref().map_or(&[], |r| &r.events)
    }
    fn general_path(&self) -> bool {
        self.info
            .indefinite
            .as_ref()
            .is_some_and(|r| r.general_path)
    }
    fn n_negative_basis(&self) -> usize {
        self.info
            .indefinite
            .as_ref()
            .map_or(0, |r| r.n_negative_basis)
    }
    fn fresh_w(&self) -> Vec<f64> {
        self.info.negative.iter().map(|w| w.fresh_w).collect()
    }
}

struct RitzStep {
    theta: Vec<f64>,
    y: Vec<Vec<f64>>,
    sign: Vec<i8>,
    complex: Vec<(C64, Vec<f64>, Vec<f64>)>,
    general: bool,
}

#[allow(clippy::type_complexity)]
fn g_orthonormalize<V: InnerProductVector>(
    vs: Vec<V>,
    ws: Vec<V>,
    ms: Vec<V>,
) -> (Vec<V>, Vec<V>, Vec<V>, Vec<Vec<f64>>) {
    let n = vs.len();
    let (mut ov, mut ow, mut om): (Vec<V>, Vec<V>, Vec<V>) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(n);
    for ((mut v, mut w), mut m) in vs.into_iter().zip(ws).zip(ms) {
        let n0 = v.norm();
        let mut col = vec![0.0; n];
        for _ in 0..2 {
            for i in 0..ov.len() {
                let c = ov[i].dot(&v).re_s();
                let sc = V::Scalar::from_f64(-c);
                v.axpy(sc, &ov[i]);
                w.axpy(sc, &ow[i]);
                m.axpy(sc, &om[i]);
                col[i] += c;
            }
        }
        let nr = v.norm();
        if nr > 0.0 && nr > 1e-10 * n0 {
            let sc = V::Scalar::from_f64(1.0 / nr);
            v.scale(sc);
            w.scale(sc);
            m.scale(sc);
            col[ov.len()] = nr;
            ov.push(v);
            ow.push(w);
            om.push(m);
        }
        cols.push(col);
    }
    let r = ov.len();
    for c in cols.iter_mut() {
        c.truncate(r);
    }
    (ov, ow, om, cols)
}

fn w_labels<V: InnerProductVector>(vs: &[V], ws: &[V]) -> Vec<i8> {
    vs.iter()
        .zip(ws)
        .map(|(v, w)| if v.dot(w).re_s() < 0.0 { -1 } else { 1 })
        .collect()
}

pub(crate) fn pencil_eta(r: f64, q2: f64, w2: f64, theta_abs: f64) -> f64 {
    r / (w2 + theta_abs * q2).sqrt()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn omega_davidson_signed<V, R, W, D, Rp, P>(
    mut r_apply: R,
    mut w_apply: W,
    mut deflate: D,
    mut reproject: Rp,
    mut precond: Option<P>,
    v0s: &[V],
    k: usize,
    space_dim: usize,
    opts: &SymplecticDavidsonOpts,
    tau: f64,
) -> SignedDavidsonOut<V>
where
    V: InnerProductVector + Send + Sync,
    R: FnMut(&V) -> V,
    W: FnMut(&V) -> V,
    D: FnMut(&mut V),
    Rp: FnMut(&mut V),
    P: FnMut(&mut V, f64),
{
    assert!(k >= 1, "omega_davidson_signed: k must be ≥ 1");
    assert!(
        space_dim >= 1,
        "omega_davidson_signed: empty iteration space"
    );
    assert!(!v0s.is_empty(), "omega_davidson_signed: empty start block");
    let b = v0s.len();
    let max_subspace = resolve_max_subspace(opts, k, b, space_dim);
    let max_basis = vector_budget(opts).max(k);
    let mut applies = 0usize;
    let mut total_built = 0usize;
    let mut diag = SympDiagnostics::default();
    let omega_trace = crate::knobs::OMEGA_TRACE.get();

    let mut vs: Vec<V> = Vec::new();
    let mut ws: Vec<V> = Vec::new();
    let mut ms: Vec<V> = Vec::new();
    let mut signs: Vec<i8> = Vec::new();
    let mut events: Vec<MetricEvent> = Vec::new();
    let mut n_negative_basis = 0usize;
    let mut general_path = false;
    let mut gbasis = false;
    let mut last_scale_pos: f64;

    let record =
        |events: &mut Vec<MetricEvent>, basis_len: usize, rep: super::davidson::PushReport| {
            if rep.wnorm2 < 0.0 {
                events.push(MetricEvent {
                    basis_len,
                    gn_before: rep.gn_before,
                    gn_after: rep.gn_after,
                    wnorm2: rep.wnorm2,
                    accepted: rep.outcome.accepted(),
                });
            }
        };

    for v0 in v0s {
        if vs.len() == space_dim {
            break;
        }
        let basis_len = vs.len();
        let rep = push_core(
            v0.clone(),
            &mut vs,
            &mut ws,
            Some(&mut signs),
            gbasis,
            &mut w_apply,
            &mut deflate,
            &mut reproject,
            0.0,
            tau,
            &mut applies,
            &mut diag,
            omega_trace,
        )
        .unwrap_or_else(|_| unreachable!("the signed push never refuses"));
        record(&mut events, basis_len, rep);
        if rep.outcome.accepted() {
            total_built += 1;
        }
    }
    assert!(
        !vs.is_empty(),
        "omega_davidson_signed: every seed collapsed under deflation — caller bug"
    );
    while ms.len() < vs.len() {
        let mut m_img = r_apply(&ws[ms.len()]);
        applies += 1;
        reproject(&mut m_img);
        ms.push(m_img);
    }
    n_negative_basis = n_negative_basis.max(signs.iter().filter(|&&s| s < 0).count());
    if signs.iter().any(|&s| s < 0) {
        let (v2, w2, m2, _) = g_orthonormalize(
            std::mem::take(&mut vs),
            std::mem::take(&mut ws),
            std::mem::take(&mut ms),
        );
        (vs, ws, ms) = (v2, w2, m2);
        signs = w_labels(&vs, &ws);
        gbasis = true;
    }

    let mut wanted_theta: Vec<f64>;
    let mut wanted_y: Vec<Vec<f64>>;
    let mut wanted_sign: Vec<i8>;
    let mut wanted_cplx: Vec<(C64, Vec<f64>, Vec<f64>)>;
    let mut residuals: Vec<f64>;
    let mut cplx_res: Vec<f64>;
    let converged;
    let mut outer = 0usize;

    loop {
        let m = vs.len();
        diag.outer_iterations += 1;
        if opts.instrument {
            let g = gram_matrix_symmetrized(&ws, &vs);
            let (gev, _) = dense_self_adjoint_smallest::<f64>(&g, m, 0);
            diag.gram_min_eig.push(gev[0]);
        }
        let t = rr_matrix_symmetrized(&ws, &ms);
        let any_neg = gbasis;
        let keep_target = (k + b).min(m);
        let step = if !any_neg {
            let want = k.min(m).max(keep_target);
            let (ev, yv) = dense_self_adjoint_smallest::<f64>(&t, m, want);
            let q = ev.iter().filter(|&&e| e < 0.0).count();
            let (ev, yv) = if q > 0 && (q + k + b).min(m) > want {
                dense_self_adjoint_smallest::<f64>(&t, m, (q + k + b).min(m))
            } else {
                (ev, yv)
            };
            let ncols = yv.len();
            RitzStep {
                theta: ev,
                y: yv,
                sign: vec![1; ncols],
                complex: Vec::new(),
                general: false,
            }
        } else {
            let g = gram_matrix_symmetrized(&ws, &vs);
            let sr = signed_ritz(&t, &g, m);
            RitzStep {
                general: sr.path == SignedRitzPath::General,
                theta: sr.theta,
                y: sr.y,
                sign: sr.sign,
                complex: sr.complex,
            }
        };
        general_path |= step.general;
        let ny = step.y.len();
        let neg_idx: Vec<usize> = (0..ny).filter(|&i| step.theta[i] < 0.0).collect();
        let pos_all: Vec<usize> = (0..ny).filter(|&i| step.theta[i] >= 0.0).collect();
        let pos_idx: Vec<usize> = pos_all.iter().copied().take(k).collect();
        let buf_idx: Vec<usize> = pos_all.iter().copied().skip(k).take(b).collect();
        let window = if pos_idx.len() == k {
            step.theta[pos_idx[k - 1]]
        } else {
            f64::INFINITY
        };
        wanted_cplx = step
            .complex
            .iter()
            .filter(|c| c.0.re <= window)
            .cloned()
            .collect();
        let wanted: Vec<usize> = neg_idx.iter().chain(pos_idx.iter()).copied().collect();
        wanted_theta = wanted.iter().map(|&i| step.theta[i]).collect();
        wanted_y = wanted.iter().map(|&i| step.y[i].clone()).collect();
        wanted_sign = wanted.iter().map(|&i| step.sign[i]).collect();
        let nw = wanted.len();
        let scale_pos = if pos_idx.is_empty() {
            wanted_theta.iter().fold(0.0f64, |a, &e| a.max(e.abs()))
        } else {
            pos_idx
                .iter()
                .fold(0.0f64, |a, &i| a.max(step.theta[i].abs()))
        }
        .max(f64::MIN_POSITIVE);
        last_scale_pos = scale_pos;

        let mut ys_all = wanted_y.clone();
        for c in &wanted_cplx {
            ys_all.push(c.1.clone());
            ys_all.push(c.2.clone());
        }
        let n_all = ys_all.len();
        let rs = recombine(&ms, &ys_all, n_all);
        let xs = recombine(&vs, &ys_all, n_all);
        let wqs = recombine(&ws, &ys_all, n_all);
        let mut res_vecs: Vec<V> = Vec::with_capacity(nw);
        residuals = Vec::with_capacity(nw);
        let mut rs_iter = rs.into_iter();
        for j in 0..nw {
            let mut r = rs_iter.next().expect("one image per wanted vector");
            r.axpy(V::Scalar::from_f64(-wanted_theta[j]), &xs[j]);
            residuals.push(r.norm());
            res_vecs.push(r);
        }
        let mut cplx_vecs: Vec<(V, V)> = Vec::with_capacity(wanted_cplx.len());
        cplx_res = Vec::with_capacity(wanted_cplx.len());
        for (c, cp) in wanted_cplx.iter().enumerate() {
            let (a, bb) = (cp.0.re, cp.0.im);
            let mut r_re = rs_iter.next().expect("re image");
            let mut r_im = rs_iter.next().expect("im image");
            let (x_re, x_im) = (&xs[nw + 2 * c], &xs[nw + 2 * c + 1]);
            r_re.axpy(V::Scalar::from_f64(-a), x_re);
            r_re.axpy(V::Scalar::from_f64(bb), x_im);
            r_im.axpy(V::Scalar::from_f64(-a), x_im);
            r_im.axpy(V::Scalar::from_f64(-bb), x_re);
            cplx_res.push((r_re.norm().powi(2) + r_im.norm().powi(2)).sqrt());
            cplx_vecs.push((r_re, r_im));
        }
        let omega_max = scale_pos.sqrt();
        let eta_r: Vec<f64> = (0..nw)
            .map(|j| {
                pencil_eta(
                    residuals[j],
                    xs[j].norm().powi(2),
                    wqs[j].norm().powi(2),
                    wanted_theta[j].abs(),
                )
            })
            .collect();
        let eta_c: Vec<f64> = (0..wanted_cplx.len())
            .map(|c| {
                let (i_re, i_im) = (nw + 2 * c, nw + 2 * c + 1);
                pencil_eta(
                    cplx_res[c],
                    xs[i_re].norm().powi(2) + xs[i_im].norm().powi(2),
                    wqs[i_re].norm().powi(2) + wqs[i_im].norm().powi(2),
                    wanted_cplx[c].0.norm(),
                )
            })
            .collect();
        let conv_real = |j: usize| -> bool {
            let bound = if wanted_theta[j] >= 0.0 {
                opts.tol * omega_max
            } else {
                opts.tol * wanted_theta[j].abs().sqrt()
            };
            eta_r[j] <= bound
        };
        let conv_cplx =
            |c: usize| -> bool { eta_c[c] <= opts.tol * wanted_cplx[c].0.norm().sqrt() };
        if omega_trace {
            let rmax = residuals.iter().fold(0.0f64, |a, &r| a.max(r));
            eprintln!(
                "  signed davidson outer={outer} m={m} built={total_built} neg_basis={} \
                 path={} θ<0: {} θ0+={:.6e} maxres={rmax:.3e} complex={}",
                signs.iter().filter(|&&s| s < 0).count(),
                if step.general {
                    "general"
                } else if any_neg {
                    "definite"
                } else {
                    "J=I"
                },
                neg_idx.len(),
                pos_idx.first().map_or(f64::NAN, |&i| step.theta[i]),
                wanted_cplx.len()
            );
        }
        let conv_r: Vec<bool> = (0..nw).map(conv_real).collect();
        let conv_c: Vec<bool> = (0..wanted_cplx.len()).map(conv_cplx).collect();
        if omega_trace {
            let list = |f: &dyn Fn(usize) -> String| (0..nw).map(f).collect::<Vec<_>>().join(",");
            let mut line = format!(
                "  signed modes outer={outer} thr={:.3e} θ=[{}] res=[{}] conv=[{}] s=[{}]",
                opts.tol * omega_max,
                list(&|j| format!("{:.12e}", wanted_theta[j])),
                list(&|j| format!("{:.3e}", residuals[j])),
                list(&|j| u8::from(conv_r[j]).to_string()),
                list(&|j| format!("{:+}", wanted_sign[j])),
            );
            if !wanted_cplx.is_empty() {
                let cx: Vec<String> = wanted_cplx
                    .iter()
                    .zip(&cplx_res)
                    .zip(&conv_c)
                    .map(|((cp, r), &c)| {
                        format!("{:.12e}{:+.12e}i:{r:.3e}:{}", cp.0.re, cp.0.im, u8::from(c))
                    })
                    .collect();
                line.push_str(&format!(" complex=[{}]", cx.join(",")));
            }
            eprintln!("{line}");
        }
        let all_conv = conv_r.iter().all(|&c| c) && conv_c.iter().all(|&c| c);
        if pos_idx.len() >= k.min(m) && m >= k && all_conv {
            converged = true;
            break;
        }
        if vs.len() == space_dim {
            converged = if gbasis { all_conv } else { true };
            break;
        }
        if total_built >= max_basis {
            converged = false;
            break;
        }

        let verbatim = !any_neg && neg_idx.is_empty() && wanted_cplx.is_empty();
        let n_kept = (nw + buf_idx.len() + 2 * wanted_cplx.len())
            .min(max_subspace.saturating_sub(1))
            .max(nw);
        let did_restart = if verbatim {
            vs.len() >= max_subspace && vs.len() > keep_target
        } else {
            vs.len() >= max_subspace && vs.len() > n_kept
        };
        let mut g_remap: Option<Vec<Vec<f64>>> = None;
        if did_restart {
            diag.restarts += 1;
            if verbatim {
                let keep = keep_target.min(step.y.len());
                (vs, ws, ms) = w_restart_compact(&vs, &ws, &ms, &step.y, keep);
                signs = vec![1; vs.len()];
            } else if gbasis {
                let mut kept_y: Vec<Vec<f64>> = wanted_y.clone();
                for c in &wanted_cplx {
                    kept_y.push(c.1.clone());
                    kept_y.push(c.2.clone());
                }
                for &i in &buf_idx {
                    kept_y.push(step.y[i].clone());
                }
                kept_y.truncate(n_kept);
                let keep = kept_y.len();
                let (v2, w2, m2, cols) = g_orthonormalize(
                    recombine(&vs, &kept_y, keep),
                    recombine(&ws, &kept_y, keep),
                    recombine(&ms, &kept_y, keep),
                );
                (vs, ws, ms) = (v2, w2, m2);
                signs = w_labels(&vs, &ws);
                g_remap = Some(cols);
            } else {
                let mut kept_y: Vec<Vec<f64>> = wanted_y.clone();
                let mut kept_hint: Vec<i8> = wanted_sign.clone();
                for c in &wanted_cplx {
                    kept_y.push(c.1.clone());
                    kept_hint.push(0);
                    kept_y.push(c.2.clone());
                    kept_hint.push(0);
                }
                for &i in &buf_idx {
                    kept_y.push(step.y[i].clone());
                    kept_hint.push(step.sign[i]);
                }
                kept_y.truncate(n_kept);
                kept_hint.truncate(n_kept);
                let keep = kept_y.len();
                let mut new_vs = recombine(&vs, &kept_y, keep);
                let mut new_ws = recombine(&ws, &kept_y, keep);
                let mut new_ms = recombine(&ms, &kept_y, keep);
                let mut new_signs: Vec<i8> = Vec::with_capacity(keep);
                for (j, &hint) in kept_hint.iter().enumerate().take(keep) {
                    let (v_head, v_tail) = new_vs.split_at_mut(j);
                    let (w_head, w_tail) = new_ws.split_at_mut(j);
                    let (m_head, m_tail) = new_ms.split_at_mut(j);
                    let (vj, wj, mj) = (&mut v_tail[0], &mut w_tail[0], &mut m_tail[0]);
                    for _ in 0..2 {
                        for i in 0..j {
                            let d = w_head[i].dot(vj).re_s();
                            let c = if new_signs[i] < 0 { -d } else { d };
                            let sc = V::Scalar::from_f64(-c);
                            vj.axpy(sc, &v_head[i]);
                            wj.axpy(sc, &w_head[i]);
                            mj.axpy(sc, &m_head[i]);
                        }
                    }
                    let beta2 = vj.dot(wj).re_s();
                    let sgn = if beta2 < 0.0 || (beta2 == 0.0 && hint < 0) {
                        -1
                    } else {
                        1
                    };
                    if beta2 != 0.0 {
                        let s = V::Scalar::from_f64(1.0 / beta2.abs().sqrt());
                        vj.scale(s);
                        wj.scale(s);
                        mj.scale(s);
                    }
                    new_signs.push(sgn);
                }
                vs = new_vs;
                ws = new_ws;
                ms = new_ms;
                signs = new_signs;
            }
            let mlen = vs.len();
            let unit = |j: usize| -> Vec<f64> {
                let mut e = vec![0.0; mlen];
                if j < mlen {
                    e[j] = 1.0;
                }
                e
            };
            if let Some(cols) = g_remap.take() {
                let col = |j: usize| -> Vec<f64> {
                    cols.get(j).cloned().unwrap_or_else(|| vec![0.0; mlen])
                };
                wanted_y = (0..nw).map(col).collect();
                for (c, cp) in wanted_cplx.iter_mut().enumerate() {
                    cp.1 = col(nw + 2 * c);
                    cp.2 = col(nw + 2 * c + 1);
                }
            } else {
                wanted_y = (0..nw).map(unit).collect();
                if !verbatim {
                    for (c, cp) in wanted_cplx.iter_mut().enumerate() {
                        cp.1 = unit(nw + 2 * c);
                        cp.2 = unit(nw + 2 * c + 1);
                    }
                }
            }
        }

        let use_precond = outer >= opts.nwarm;
        let mut accepted_this_round = 0usize;
        let mut corrections: Vec<(V, f64, usize)> = Vec::new();
        for (j, r) in res_vecs.into_iter().enumerate() {
            if !conv_r[j] {
                corrections.push((r, wanted_theta[j], j));
            }
        }
        for (c, (r_re, r_im)) in cplx_vecs.into_iter().enumerate() {
            if !conv_c[c] {
                let a = wanted_cplx[c].0.re;
                corrections.push((r_re, a, nw + 2 * c));
                corrections.push((r_im, a, nw + 2 * c + 1));
            }
        }
        for (mut r, theta_j, xi) in corrections {
            if vs.len() >= max_subspace.min(space_dim) || total_built >= max_basis {
                break;
            }
            if use_precond
                && precond.is_some()
                && survival_guard_trips(&diag.survival, b, opts.guard_survival)
            {
                precond = None;
                diag.guard_tripped = Some(diag.survival.len());
            }
            if use_precond && let Some(p) = precond.as_mut() {
                p(&mut r, theta_j);
            }
            let den = r.norm() * xs[xi].norm();
            diag.collapse_angle.push(if den > 0.0 {
                r.dot(&xs[xi]).abs_s() / den
            } else {
                f64::NAN
            });
            let basis_len = vs.len();
            let rep = push_core(
                r,
                &mut vs,
                &mut ws,
                Some(&mut signs),
                gbasis,
                &mut w_apply,
                &mut deflate,
                &mut reproject,
                scale_pos,
                tau,
                &mut applies,
                &mut diag,
                omega_trace,
            )
            .unwrap_or_else(|_| unreachable!("the signed push never refuses"));
            record(&mut events, basis_len, rep);
            if let PushOutcome::Accepted { .. } = rep.outcome {
                total_built += 1;
                accepted_this_round += 1;
                let mut m_img = r_apply(&ws[ws.len() - 1]);
                applies += 1;
                reproject(&mut m_img);
                ms.push(m_img);
                if !gbasis && rep.outcome == (PushOutcome::Accepted { sign: -1 }) {
                    n_negative_basis = n_negative_basis.max(1);
                    let (v2, w2, m2, _) = g_orthonormalize(
                        std::mem::take(&mut vs),
                        std::mem::take(&mut ws),
                        std::mem::take(&mut ms),
                    );
                    (vs, ws, ms) = (v2, w2, m2);
                    signs = w_labels(&vs, &ws);
                    gbasis = true;
                }
            }
        }
        n_negative_basis = n_negative_basis.max(signs.iter().filter(|&&s| s < 0).count());
        if accepted_this_round == 0 {
            converged = all_conv && pos_idx.len() >= k.min(m) && vs.len() >= k;
            break;
        }
        outer += 1;
    }

    let nw = wanted_theta.len();
    let (vectors, mut w_vectors) = if nw == 0 {
        (Vec::new(), Vec::new())
    } else {
        (recombine(&vs, &wanted_y, nw), recombine(&ws, &wanted_y, nw))
    };
    let mut complex: Vec<ComplexRitzOut<V>> = wanted_cplx
        .iter()
        .zip(&cplx_res)
        .map(|(c, &res)| {
            let mut v = recombine(&vs, &[c.1.clone(), c.2.clone()], 2);
            let x_im = v.pop().expect("im part");
            let x_re = v.pop().expect("re part");
            let mut w = recombine(&ws, &[c.1.clone(), c.2.clone()], 2);
            let w_im = w.pop().expect("im image");
            let w_re = w.pop().expect("re image");
            ComplexRitzOut {
                theta: c.0,
                residual: res,
                x_re,
                x_im,
                w_re,
                w_im,
            }
        })
        .collect();
    let mut fresh_ok = true;
    if gbasis {
        let mut fresh_m = |x: &V, applies: &mut usize| -> (V, V, V) {
            let mut v = x.clone();
            reproject(&mut v);
            let mut u = w_apply(&v);
            deflate(&mut u);
            let mut m = r_apply(&u);
            reproject(&mut m);
            *applies += 2;
            (v, m, u)
        };
        for (j, x) in vectors.iter().enumerate() {
            let (v, mut m, u) = fresh_m(x, &mut applies);
            let w2 = u.norm().powi(2);
            w_vectors[j] = u;
            m.axpy(V::Scalar::from_f64(-wanted_theta[j]), &v);
            let r = m.norm();
            let th = wanted_theta[j];
            let eta = pencil_eta(r, v.norm().powi(2), w2, th.abs());
            fresh_ok &= if th >= 0.0 {
                eta <= opts.tol * last_scale_pos.sqrt()
            } else {
                eta <= opts.tol * th.abs().sqrt()
            };
            residuals[j] = r;
        }
        for c in complex.iter_mut() {
            let (a, bb) = (c.theta.re, c.theta.im);
            let (vr, mut mr, ur) = fresh_m(&c.x_re, &mut applies);
            let (vi, mut mi, ui) = fresh_m(&c.x_im, &mut applies);
            let w2 = ur.norm().powi(2) + ui.norm().powi(2);
            c.w_re = ur;
            c.w_im = ui;
            mr.axpy(V::Scalar::from_f64(-a), &vr);
            mr.axpy(V::Scalar::from_f64(bb), &vi);
            mi.axpy(V::Scalar::from_f64(-a), &vi);
            mi.axpy(V::Scalar::from_f64(-bb), &vr);
            let r = (mr.norm().powi(2) + mi.norm().powi(2)).sqrt();
            let q2 = vr.norm().powi(2) + vi.norm().powi(2);
            let eta = pencil_eta(r, q2, w2, c.theta.norm());
            fresh_ok &= eta <= opts.tol * c.theta.norm().sqrt();
            c.residual = r;
        }
    }
    let converged = converged && fresh_ok;
    let scale_all = wanted_theta
        .iter()
        .fold(0.0f64, |a, &e| a.max(e.abs()))
        .max(f64::MIN_POSITIVE);
    let mut negative = Vec::new();
    for (index, (&theta, x)) in wanted_theta.iter().zip(&vectors).enumerate() {
        if theta >= -1e-8 * scale_all {
            continue;
        }
        let mut v = x.clone();
        reproject(&mut v);
        let mut u = w_apply(&v);
        applies += 1;
        deflate(&mut u);
        let wn2 = v.dot(&u).re_s();
        let vn2 = v.norm().powi(2).max(f64::MIN_POSITIVE);
        let fresh = if wn2 == 0.0 {
            f64::NAN
        } else {
            let ru = r_apply(&u);
            applies += 1;
            u.dot(&ru).re_s() / wn2
        };
        negative.push(NegativeWitness {
            index,
            theta,
            fresh,
            fresh_w: wn2 / vn2,
        });
    }
    let indefinite = (!events.is_empty()).then(|| IndefiniteReport {
        events,
        n_negative_basis,
        general_path,
        complex: complex
            .iter()
            .map(|c| ComplexRitz {
                re: c.theta.re,
                im: c.theta.im,
                residual: c.residual,
            })
            .collect(),
    });
    SignedDavidsonOut {
        omega2: wanted_theta,
        vectors,
        w_vectors,
        complex,
        info: SymplecticLowestInfo {
            converged,
            iterations: total_built,
            applies,
            residuals,
            negative,
            diag: Some(diag),
            signature: wanted_sign,
            indefinite,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::omega_davidson_signed;
    use crate::core::factor::dense_eigen_general;
    use crate::solvers::hess::davidson::{
        SympPrecond, SymplecticDavidsonOpts, omega_davidson_vectors,
    };
    use crate::solvers::hess::drivers::build_symp_precond;
    use crate::solvers::steihaug::testvec::{DVec, mat_apply};
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn orthogonal(n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<Vec<f64>> {
        let mut q: Vec<Vec<f64>> = Vec::with_capacity(n);
        while q.len() < n {
            let mut v: Vec<f64> = (0..n).map(|_| rng.random_range(-1.0..1.0)).collect();
            for _ in 0..2 {
                for u in &q {
                    let c: f64 = u.iter().zip(&v).map(|(a, b)| a * b).sum();
                    v.iter_mut().zip(u).for_each(|(x, y)| *x -= c * y);
                }
            }
            let nv = v.iter().map(|a| a * a).sum::<f64>().sqrt();
            if nv > 1e-8 {
                q.push(v.iter().map(|a| a / nv).collect());
            }
        }
        q
    }

    fn sym(evals: &[f64], rng: &mut Xoshiro256PlusPlus) -> Vec<f64> {
        let n = evals.len();
        let q = orthogonal(n, rng);
        let mut a = vec![0.0; n * n];
        for (c, &l) in evals.iter().enumerate() {
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] += l * q[c][i] * q[c][j];
                }
            }
        }
        a
    }

    fn spread(n: usize, lo: f64, hi: f64) -> Vec<f64> {
        (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n.max(2) - 1) as f64)
            .collect()
    }

    fn oracle(r: &[f64], w: &[f64], n: usize) -> (Vec<f64>, Vec<crate::scalar::C64>) {
        let mut p = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                p[i * n + j] = (0..n).map(|l| r[i * n + l] * w[l * n + j]).sum();
            }
        }
        let (ev, _) = dense_eigen_general::<f64>(&p, n);
        let scale = ev.iter().fold(0.0f64, |a, z| a.max(z.norm()));
        let mut real: Vec<f64> = ev
            .iter()
            .filter(|z| z.im.abs() <= 1e-10 * scale)
            .map(|z| z.re)
            .collect();
        real.sort_by(f64::total_cmp);
        let mut cplx: Vec<crate::scalar::C64> = ev
            .iter()
            .filter(|z| z.im > 1e-10 * scale)
            .copied()
            .collect();
        cplx.sort_by(|a, b| a.re.total_cmp(&b.re));
        (real, cplx)
    }

    fn seeds(n: usize, b: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<DVec> {
        (0..b)
            .map(|_| DVec((0..n).map(|_| rng.random_range(-1.0..1.0)).collect()))
            .collect()
    }

    #[test]
    fn pd_problem_is_bitwise_the_refuse_kernel() {
        let n = 48;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC1_C1);
        let r = sym(&spread(n, 0.5, 5.0), &mut rng);
        let w = sym(&spread(n, 0.3, 4.0), &mut rng);
        let v0s = seeds(n, 4, &mut rng);
        let dg: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        let opts = SymplecticDavidsonOpts {
            tol: 0.0,
            max_basis: 800,
            max_subspace: 16,
            nwarm: 2,
            ..SymplecticDavidsonOpts::default()
        };
        let k = 3;
        let pre = |t: &mut DVec, _th: f64| t.0.iter_mut().zip(&dg).for_each(|(x, d)| *x /= d);
        let (om_r, vec_r, info_r) = omega_davidson_vectors(
            |v: &DVec| mat_apply(&r, n, v),
            |v: &DVec| mat_apply(&w, n, v),
            |_: &mut DVec| {},
            |_: &mut DVec| {},
            Some(pre),
            &v0s,
            k,
            n,
            &opts,
        )
        .expect("PD metric: the Refuse kernel cannot refuse");
        let out = omega_davidson_signed(
            |v: &DVec| mat_apply(&r, n, v),
            |v: &DVec| mat_apply(&w, n, v),
            |_: &mut DVec| {},
            |_: &mut DVec| {},
            Some(pre),
            &v0s,
            k,
            n,
            &opts,
            super::super::davidson::REFUSE_TAU,
        );
        let dr = info_r.diag.as_ref().unwrap();
        let ds = out.info.diag.as_ref().unwrap();
        println!(
            "C1: converged {} built {} applies {} restarts {}",
            info_r.converged, info_r.iterations, info_r.applies, dr.restarts
        );
        assert!(dr.restarts >= 1, "the fixture must exercise the restart");
        assert_eq!(om_r.len(), out.omega2.len());
        for (a, b) in om_r.iter().zip(&out.omega2) {
            assert_eq!(a.to_bits(), b.to_bits(), "ω²: {a} vs {b}");
        }
        for (a, b) in info_r.residuals.iter().zip(&out.info.residuals) {
            assert_eq!(a.to_bits(), b.to_bits(), "residual: {a} vs {b}");
        }
        assert_eq!(info_r.iterations, out.info.iterations);
        assert_eq!(info_r.applies, out.info.applies);
        assert_eq!(info_r.converged, out.info.converged);
        assert_eq!(dr.restarts, ds.restarts);
        assert_eq!(dr.outer_iterations, ds.outer_iterations);
        for (x, y) in vec_r.iter().zip(&out.vectors) {
            for (a, b) in x.0.iter().zip(&y.0) {
                assert_eq!(a.to_bits(), b.to_bits(), "Ritz vector coefficient");
            }
        }
        assert!(out.signature().iter().all(|&s| s == 1));
        assert!(out.events().is_empty() && !out.general_path() && out.n_negative_basis() == 0);
    }

    #[test]
    fn stagnation_right_after_a_restart_returns_the_kept_ritz_vectors() {
        let n = 40;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x57A6_5167);
        let r = sym(&spread(n, 0.5, 5.0), &mut rng);
        let mut wev = spread(n, 0.3, 4.0);
        wev[0] = -0.4;
        let w = sym(&wev, &mut rng);
        let v0s = seeds(n, 3, &mut rng);
        let opts = SymplecticDavidsonOpts {
            tol: 1e-13,
            max_basis: 800,
            max_subspace: 12,
            nwarm: 0,
            ..SymplecticDavidsonOpts::default()
        };
        let mut restarted_runs = 0usize;
        for n_switch in 0..60usize {
            let calls = std::cell::Cell::new(0usize);
            let pre = |t: &mut DVec, _th: f64| {
                if calls.get() >= n_switch {
                    t.0.iter_mut().for_each(|x| *x = 0.0);
                }
                calls.set(calls.get() + 1);
            };
            let out = omega_davidson_signed(
                |v: &DVec| mat_apply(&r, n, v),
                |v: &DVec| mat_apply(&w, n, v),
                |_: &mut DVec| {},
                |_: &mut DVec| {},
                Some(pre),
                &v0s,
                2,
                n,
                &opts,
                super::super::davidson::REFUSE_TAU,
            );
            let d = out.info.diag.as_ref().unwrap();
            if d.restarts > 0 {
                restarted_runs += 1;
            }
            for (j, (x, &th)) in out.vectors.iter().zip(&out.omega2).enumerate() {
                let wx = mat_apply(&w, n, x);
                let mx = mat_apply(&r, n, &wx);
                let res =
                    mx.0.iter()
                        .zip(&x.0)
                        .map(|(a, b)| (a - th * b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                let wn: f64 = x.0.iter().zip(&wx.0).map(|(a, b)| a * b).sum();
                let rep = out.info.residuals[j];
                assert!(
                    (res - rep).abs() <= 1e-8 * (1.0 + rep),
                    "n_switch {n_switch}, pair {j}: dense residual {res:.3e} vs reported {rep:.3e}"
                );
                assert!(
                    (wn - f64::from(out.signature()[j])).abs() <= 1e-8,
                    "n_switch {n_switch}, pair {j}: W-norm² {wn} vs signature {}",
                    out.signature()[j]
                );
            }
        }
        assert!(
            restarted_runs >= 10,
            "only {restarted_runs} runs restarted at all"
        );
    }

    #[test]
    fn live_tangent_space_matches_the_dense_grams() {
        use crate::core::tangent::{TtTangent, tangent_basis};
        use crate::qn::QnSpace;
        use crate::round::RoundMode;
        use crate::solvers::hess::{
            clean_tangent, deflate_radial, hessian_grams, radial_mode, weingarten_operator,
        };
        use crate::test_util::{generic_truncated_h, randomized_state};
        use crate::vector::TtVector;
        let space = QnSpace::new(4, 2, 1);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x81E1);
        let mut x: TtVector<f64> = randomized_state(space, 1, 2, &mut rng);
        x.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        x.scale_mut(1.0 / x.norm());
        let h = generic_truncated_h::<f64>(space);
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.bases();
        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let (a, b) = hessian_grams(&ctx, &basis);
        let hrr: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        let hii: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p - q).collect();
        let neg = |m: &[f64]| {
            let (ev, _) = crate::core::factor::dense_self_adjoint_smallest::<f64>(m, n, 0);
            ev.iter().filter(|&&e| e < -1e-10).count()
        };
        let (nrr, nii) = (neg(&hrr), neg(&hii));
        let (real, cplx) = oracle(&hrr, &hii, n);
        println!(
            "C7: n = {n}, Morse index H^rr {nrr}, H^ii {nii}; dense real {real:?} complex {cplx:?}"
        );
        assert!(
            nii >= 1,
            "fixture precondition: H^ii indefinite at this non-critical point"
        );
        let radial = radial_mode(&ctx);
        let k = 3;
        let dir = |rng: &mut Xoshiro256PlusPlus| {
            let d: TtVector<f64> = randomized_state(space, 1, 2, rng);
            TtTangent::project(bl, br, &d)
        };
        let v0s: Vec<TtTangent<f64>> = (0..k + 2).map(|_| dir(&mut rng)).collect();
        let opts = SymplecticDavidsonOpts {
            tol: 3.6e-12,
            max_basis: 800,
            max_subspace: n,
            ..SymplecticDavidsonOpts::default()
        };
        let out = omega_davidson_signed(
            |w: &TtTangent<f64>| ctx.hpp_apply(w),
            |v: &TtTangent<f64>| ctx.hmm_apply(v),
            |v: &mut TtTangent<f64>| deflate_radial(radial.as_ref(), v),
            |v: &mut TtTangent<f64>| clean_tangent(radial.as_ref(), v),
            None::<fn(&mut TtTangent<f64>, f64)>,
            &v0s,
            k,
            n,
            &opts,
            super::super::davidson::REFUSE_TAU,
        );
        println!(
            "C7: converged {} built {} general {} events {} ω² {:?} complex {:?}",
            out.info.converged,
            out.info.iterations,
            out.general_path(),
            out.events().len(),
            out.omega2,
            out.complex.iter().map(|c| c.theta).collect::<Vec<_>>()
        );
        assert!(
            out.metric_indefinite(),
            "the Ii lane meets the negative W directions"
        );
        let negs: Vec<f64> = real.iter().copied().filter(|&x| x < 0.0).collect();
        let pos: Vec<f64> = real.iter().copied().filter(|&x| x >= 0.0).take(k).collect();
        let want: Vec<f64> = negs.iter().chain(&pos).copied().collect();
        let scale = real
            .iter()
            .map(|x| x.abs())
            .chain(cplx.iter().map(|z| z.norm()))
            .fold(0.0f64, f64::max);
        assert_eq!(
            out.omega2.len(),
            want.len(),
            "wanted set: every θ < 0 plus k ≥ 0"
        );
        for (j, (&got, &w)) in out.omega2.iter().zip(&want).enumerate() {
            assert!(
                (got - w).abs() <= 1e-8 * scale,
                "ω²[{j}] {got} vs dense {w}"
            );
        }
        let window = pos.last().copied().unwrap_or(f64::INFINITY);
        let dense_in: Vec<_> = cplx.iter().filter(|z| z.re <= window).collect();
        assert_eq!(
            out.complex.len(),
            dense_in.len(),
            "complex pairs inside the window"
        );
        for (c, z) in out.complex.iter().zip(&dense_in) {
            assert!(
                (c.theta - **z).norm() <= 1e-8 * scale,
                "complex {} vs dense {z}",
                c.theta
            );
        }
    }

    #[test]
    fn exhaustion_holding_negative_signatures_is_judged_by_the_residuals() {
        use crate::core::border_fixture::{BOND, base_state, with_small_schmidt};
        use crate::core::tangent::{TtTangent, tangent_basis};
        use crate::solvers::hess::{
            RpaMetric, a_diagonal_squared, hessian_grams, weingarten_operator,
        };
        use crate::test_util::generic_truncated_h;
        let x = with_small_schmidt(&base_state(), BOND, 1e-6);
        let h = generic_truncated_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.bases();
        let basis = tangent_basis(bl, br, true);
        let n = basis.len();
        let (a, b) = hessian_grams(&ctx, &basis);
        let hrr: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p + q).collect();
        let hii: Vec<f64> = a.iter().zip(&b).map(|(p, q)| p - q).collect();
        let (real, _) = oracle(&hrr, &hii, n);
        let k = 3;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEF5);
        let v0s: Vec<TtTangent<f64>> = (0..k + 2)
            .map(|_| {
                TtTangent::project(
                    bl,
                    br,
                    &crate::vector::tt_randn(x.space, &x.ranks, &mut rng),
                )
            })
            .collect();
        let tol = 1e-10;
        let opts = SymplecticDavidsonOpts {
            tol,
            max_basis: 8 * n,
            max_subspace: n,
            ..SymplecticDavidsonOpts::default()
        };
        let negs: Vec<f64> = real.iter().copied().filter(|&w| w < 0.0).collect();
        let pos: Vec<f64> = real.iter().copied().filter(|&w| w >= 0.0).take(k).collect();
        let want: Vec<f64> = negs.iter().chain(&pos).copied().collect();
        use super::super::newton_polish::{clean_tangent, deflate_radial, radial_mode};
        let radial = radial_mode(&ctx);
        let (sbl, sbr) = ctx.shared_bases();
        let seeds = super::super::drivers::transport_seeds(sbl, sbr, radial.as_ref(), &v0s);
        let space_dim = crate::core::tangent::tangent_dim(sbl) - radial.as_ref().map_or(0, |_| 1);
        for lane in [RpaMetric::Ii, RpaMetric::Rr] {
            let precond =
                build_symp_precond(opts.precond.resolve_tt(), || a_diagonal_squared(&ctx));
            let rr = lane == RpaMetric::Rr;
            let r_op = |v: &TtTangent<f64>| {
                if rr {
                    ctx.hmm_apply(v)
                } else {
                    ctx.hpp_apply(v)
                }
            };
            let w_op = |v: &TtTangent<f64>| {
                if rr {
                    ctx.hpp_apply(v)
                } else {
                    ctx.hmm_apply(v)
                }
            };
            let so = omega_davidson_signed(
                r_op,
                w_op,
                |v: &mut TtTangent<f64>| deflate_radial(radial.as_ref(), v),
                |v: &mut TtTangent<f64>| clean_tangent(radial.as_ref(), v),
                precond,
                &seeds,
                k,
                space_dim,
                &opts,
                super::super::davidson::REFUSE_TAU,
            );
            let err = so
                .omega2
                .iter()
                .zip(&want)
                .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
            let rmax = so.info.residuals.iter().fold(0.0f64, |m, &r| m.max(r));
            let scale_pos = so
                .omega2
                .iter()
                .filter(|&&w| w >= 0.0)
                .fold(f64::MIN_POSITIVE, |m, &w| m.max(w.abs()));
            println!(
                "{lane:?}: n {n} built {} converged {} neg_basis {:?} max err {err:.3e} \
                 max residual {rmax:.3e} (tol·scale {:.1e})",
                so.info.iterations,
                so.info.converged,
                so.info.indefinite.as_ref().map(|r| r.n_negative_basis),
                tol * scale_pos
            );
            assert_eq!(so.info.iterations, n, "the run exhausts the space");
            assert!(
                so.info.indefinite.is_some(),
                "precondition: signature −1 vectors are held"
            );
            assert!(
                rmax > 1e3 * tol * scale_pos,
                "precondition: the exhausted solve is inexact here (residual {rmax:e})"
            );
            let w_max = scale_pos.sqrt();
            let mut headroom = 0.0f64;
            for (j, &th) in so.omega2.iter().enumerate() {
                let (q, wq) = (&so.vectors[j], &so.w_vectors[j]);
                let eta = super::pencil_eta(
                    so.info.residuals[j],
                    q.norm().powi(2),
                    wq.norm().powi(2),
                    th.abs(),
                );
                let bound = tol * if th >= 0.0 { w_max } else { th.abs().sqrt() };
                headroom = headroom.max(eta / bound);
            }
            for c in &so.complex {
                let eta = super::pencil_eta(
                    c.residual,
                    c.x_re.norm().powi(2) + c.x_im.norm().powi(2),
                    c.w_re.norm().powi(2) + c.w_im.norm().powi(2),
                    c.theta.norm(),
                );
                headroom = headroom.max(eta / (tol * c.theta.norm().sqrt()));
            }
            println!("{lane:?}: max η/(tol·bound) = {headroom:.3e}");
            assert!(
                headroom >= 1e3,
                "precondition: the worst pencil residual is only {headroom:.1e}× its bound"
            );
            assert!(
                !so.info.converged,
                "{lane:?}: an exhausted hyperbolic basis with residual {rmax:e} was called converged"
            );
        }
    }

    #[test]
    fn off_the_identity_path_the_reported_residuals_are_fresh() {
        use crate::core::border_fixture::{BOND, base_state, with_small_schmidt};
        use crate::core::tangent::TtTangent;
        use crate::solvers::hess::{
            ModeCheckOpts, RpaKernel, RpaMetric, rpa_mode_checks,
            symplectic_lowest_signed_davidson, weingarten_operator,
        };
        use crate::test_util::generic_truncated_h;
        let x = with_small_schmidt(&base_state(), BOND, 1e-6);
        let h = generic_truncated_h::<f64>(x.space);
        let ctx = weingarten_operator(&h, &x);
        let (bl, br) = ctx.bases();
        let n = crate::core::tangent::tangent_dim(bl);
        let k = 3;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xDEF5);
        let v0s: Vec<TtTangent<f64>> = (0..k + 2)
            .map(|_| {
                TtTangent::project(
                    bl,
                    br,
                    &crate::vector::tt_randn(x.space, &x.ranks, &mut rng),
                )
            })
            .collect();
        let opts = SymplecticDavidsonOpts {
            tol: 2.5e-11,
            max_basis: 8 * n,
            max_subspace: n,
            ..SymplecticDavidsonOpts::default()
        };
        for lane in [RpaMetric::Ii, RpaMetric::Rr] {
            let opts = SymplecticDavidsonOpts {
                kernel: RpaKernel::signed(lane),
                ..opts
            };
            let (so, vecs) = symplectic_lowest_signed_davidson(&ctx, &v0s, k, &opts);
            assert!(
                so.metric_events() > 0,
                "precondition: signature −1 vectors are held"
            );
            let checks = rpa_mode_checks(&ctx, &so, &vecs, &ModeCheckOpts::default());
            let mut compared = 0usize;
            for c in checks.iter().filter(|c| c.omega2 > 0.0) {
                let reported = so.modes[c.index].residual;
                let ratio = reported / c.residual_fresh;
                println!(
                    "{lane:?} mode {}: reported {reported:.3e} fresh {:.3e} ratio {ratio:.3}",
                    c.index, c.residual_fresh
                );
                assert!(
                    (0.5..=2.0).contains(&ratio),
                    "{lane:?} mode {}: reported residual {reported:e} vs fresh {:e}",
                    c.index,
                    c.residual_fresh
                );
                compared += 1;
            }
            assert!(compared >= 1, "{lane:?}: no positive mode to compare");
        }
    }

    #[test]
    fn case_ii_returns_the_dense_spectrum_where_refuse_refuses() {
        let n = 48;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC3_C3);
        let r = sym(&spread(n, 0.5, 5.0), &mut rng);
        let mut wev = spread(n, 0.25, 4.0);
        wev[0] = -0.6;
        wev[1] = -0.35;
        wev[2] = -0.2;
        let w = sym(&wev, &mut rng);
        let v0s = seeds(n, 6, &mut rng);
        let opts = SymplecticDavidsonOpts {
            tol: 6.5e-10,
            max_basis: 2400,
            max_subspace: 40,
            ..SymplecticDavidsonOpts::default()
        };
        let k = 4;
        let refuse = omega_davidson_vectors(
            |v: &DVec| mat_apply(&r, n, v),
            |v: &DVec| mat_apply(&w, n, v),
            |_: &mut DVec| {},
            |_: &mut DVec| {},
            None::<fn(&mut DVec, f64)>,
            &v0s,
            k,
            n,
            &SymplecticDavidsonOpts { tol: 1e-9, ..opts },
        );
        assert!(
            refuse.is_err(),
            "nonvacuity: an indefinite metric must refuse today"
        );
        let out = omega_davidson_signed(
            |v: &DVec| mat_apply(&r, n, v),
            |v: &DVec| mat_apply(&w, n, v),
            |_: &mut DVec| {},
            |_: &mut DVec| {},
            None::<fn(&mut DVec, f64)>,
            &v0s,
            k,
            n,
            &opts,
            super::super::davidson::REFUSE_TAU,
        );
        let (real, cplx) = oracle(&r, &w, n);
        assert!(cplx.is_empty(), "case II has a real product spectrum");
        let negs: Vec<f64> = real.iter().copied().filter(|&x| x < 0.0).collect();
        assert_eq!(negs.len(), 3, "Sylvester: three negative ω²");
        let pos: Vec<f64> = real.iter().copied().filter(|&x| x >= 0.0).take(k).collect();
        println!(
            "C3: converged {} built {} applies {} events {} neg_basis {} ω² {:?}",
            out.info.converged,
            out.info.iterations,
            out.info.applies,
            out.events().len(),
            out.n_negative_basis(),
            out.omega2
        );
        assert!(out.info.converged, "the signed kernel converges case II");
        assert!(
            !out.general_path(),
            "R ≻ 0 ⇒ T ≻ 0 ⇒ the definite path throughout"
        );
        assert!(out.metric_indefinite() && out.n_negative_basis() >= 3);
        let want: Vec<f64> = negs.iter().chain(&pos).copied().collect();
        assert_eq!(out.omega2.len(), want.len());
        let scale = want.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        for (j, (&a, &b)) in out.omega2.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-7 * scale, "ω²[{j}] {a} vs dense {b}");
            assert_eq!(
                out.signature()[j],
                if b < 0.0 { -1 } else { 1 },
                "signature at {j}"
            );
        }
        assert_eq!(out.info.negative.len(), 3);
        for (wn, &fw) in out.info.negative.iter().zip(&out.fresh_w()) {
            assert!(wn.certified(), "certificate {wn:?}");
            assert!(
                fw < 0.0,
                "the Ritz vector itself witnesses W's indefiniteness: {fw}"
            );
        }
    }

    #[test]
    fn case_iii_negative_omega2_on_the_identity_gram() {
        let n = 40;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC4_C4);
        let mut rev = spread(n, 0.4, 5.0);
        rev[0] = -0.5;
        rev[1] = -0.25;
        let r = sym(&rev, &mut rng);
        let w = sym(&spread(n, 0.3, 3.0), &mut rng);
        let v0s = seeds(n, 5, &mut rng);
        let opts = SymplecticDavidsonOpts {
            tol: 4.7e-10,
            max_basis: 2400,
            max_subspace: 36,
            ..SymplecticDavidsonOpts::default()
        };
        let k = 3;
        let out = omega_davidson_signed(
            |v: &DVec| mat_apply(&r, n, v),
            |v: &DVec| mat_apply(&w, n, v),
            |_: &mut DVec| {},
            |_: &mut DVec| {},
            None::<fn(&mut DVec, f64)>,
            &v0s,
            k,
            n,
            &opts,
            super::super::davidson::REFUSE_TAU,
        );
        let (real, cplx) = oracle(&r, &w, n);
        assert!(cplx.is_empty());
        let negs: Vec<f64> = real.iter().copied().filter(|&x| x < 0.0).collect();
        assert_eq!(negs.len(), 2);
        let pos: Vec<f64> = real.iter().copied().filter(|&x| x >= 0.0).take(k).collect();
        let want: Vec<f64> = negs.iter().chain(&pos).copied().collect();
        println!(
            "C4: converged {} built {} ω² {:?}",
            out.info.converged, out.info.iterations, out.omega2
        );
        assert!(out.info.converged);
        assert!(!out.metric_indefinite(), "W ≻ 0: no metric event");
        assert_eq!(out.omega2.len(), want.len());
        let scale = want.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        for (j, (&a, &b)) in out.omega2.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-7 * scale, "ω²[{j}] {a} vs dense {b}");
        }
        assert!(
            out.signature().iter().all(|&s| s == 1),
            "a PD metric has signature +1 only"
        );
        assert!(out.info.negative.iter().all(|w| w.certified()));
    }

    #[test]
    fn signed_stops_on_the_pencil_residual_in_omega_units() {
        use crate::solvers::hess::{RpaMetric, RpaModeKind, readout_from_signed};
        type Fixture = (Vec<f64>, Vec<f64>, usize, Vec<DVec>, usize, usize);
        let c1 = || -> Fixture {
            let n = 48;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC1_C1);
            let r = sym(&spread(n, 0.5, 5.0), &mut rng);
            let w = sym(&spread(n, 0.3, 4.0), &mut rng);
            (r, w, n, seeds(n, 4, &mut rng), 3, 16)
        };
        let c4 = || -> Fixture {
            let n = 40;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC4_C4);
            let mut rev = spread(n, 0.4, 5.0);
            rev[0] = -0.5;
            rev[1] = -0.25;
            let r = sym(&rev, &mut rng);
            let w = sym(&spread(n, 0.3, 3.0), &mut rng);
            (r, w, n, seeds(n, 5, &mut rng), 3, 36)
        };
        let c9 = || -> Fixture {
            let n = 64;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC9_C9);
            let mut rev = spread(n, 0.5, 4.0);
            rev[0] = -0.4;
            let mut wev = spread(n, 0.3, 3.0);
            wev[0] = -0.5;
            wev[1] = -0.3;
            let r = sym(&rev, &mut rng);
            let w = sym(&wev, &mut rng);
            (r, w, n, seeds(n, 6, &mut rng), 3, 32)
        };
        let solve = |fx: &Fixture, tol: f64, max_basis: usize| {
            let (r, w, n, v0s, k, ms) = fx;
            let opts = SymplecticDavidsonOpts {
                tol,
                max_basis,
                max_subspace: *ms,
                ..SymplecticDavidsonOpts::default()
            };
            let out = omega_davidson_signed(
                |v: &DVec| mat_apply(r, *n, v),
                |v: &DVec| mat_apply(w, *n, v),
                |_: &mut DVec| {},
                |_: &mut DVec| {},
                None::<fn(&mut DVec, f64)>,
                v0s,
                *k,
                *n,
                &opts,
                super::super::davidson::REFUSE_TAU,
            );
            let (gbasis, built) = (out.info.indefinite.is_some(), out.info.iterations);
            let applies = out.info.applies;
            let (ro, _) = readout_from_signed(out, RpaMetric::Ii, applies);
            (ro, gbasis, built)
        };
        let worst = |ro: &crate::solvers::hess::RpaReadout| -> f64 {
            let w_max = ro
                .modes
                .iter()
                .filter(|m| m.kind == RpaModeKind::Real)
                .fold(0.0f64, |a, m| a.max(m.omega.norm()));
            ro.modes
                .iter()
                .map(|m| {
                    let scale = if m.kind == RpaModeKind::Real {
                        w_max
                    } else {
                        m.omega.norm()
                    };
                    m.residual / scale
                })
                .fold(0.0f64, f64::max)
        };
        let tol = 1e-9;
        for (label, fx, kinds) in [
            ("C1", c1(), [RpaModeKind::Real].as_slice()),
            ("C4", c4(), &[RpaModeKind::Imaginary, RpaModeKind::Real]),
            ("IV", c9(), &[RpaModeKind::Complex]),
        ] {
            let (ro, gbasis, built) = solve(&fx, tol, 2 * 4000);
            let ratio = worst(&ro);
            println!(
                "{label}: converged {} built {built} g-basis {gbasis} worst η/(ω-scale) {ratio:.3e} \
                 (tol {tol:e})",
                ro.converged
            );
            assert!(
                ro.converged,
                "{label}: the fixture converges at tol {tol:e}"
            );
            for kind in kinds {
                assert!(
                    ro.modes.iter().any(|m| m.kind == *kind),
                    "{label}: non-vacuity, a {kind:?} mode is reported"
                );
            }
            assert!(
                ratio <= tol,
                "{label}: converged with a pencil residual {ratio:e}·ω-scale above tol {tol:e}"
            );
            assert_eq!(gbasis, label == "IV", "{label}: the exit path");
            let tight = 0.5 * ratio;
            let (ro2, _, built2) = solve(&fx, tight, 2 * built);
            println!(
                "{label}: tol {tight:.3e}, budget {built} vectors: converged {} built {built2}",
                ro2.converged
            );
            assert!(
                !ro2.converged,
                "{label}: converged at tol {tight:e}, half its own worst ratio {ratio:e}"
            );
        }
        for eta in [
            super::pencil_eta(0.0, 0.0, 0.0, 1.0),
            super::pencil_eta(f64::NAN, 1.0, 1.0, 1.0),
            super::pencil_eta(1.0, f64::NAN, 1.0, 1.0),
        ] {
            assert!(eta.is_nan(), "η {eta} is a number: a bound could pass it");
            assert_eq!(eta.partial_cmp(&f64::MAX), None, "η {eta} compares");
        }
    }

    #[test]
    fn max_basis_counts_pair_equivalents() {
        let n = 48;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC1_C1);
        let r = sym(&spread(n, 0.5, 5.0), &mut rng);
        let w = sym(&spread(n, 0.3, 4.0), &mut rng);
        let v0s = seeds(n, 2, &mut rng);
        for (max_basis, vectors) in [(6, 3), (7, 4), (8, 4)] {
            let opts = SymplecticDavidsonOpts {
                tol: 0.0,
                max_basis,
                precond: SympPrecond::Off,
                ..SymplecticDavidsonOpts::default()
            };
            let signed = omega_davidson_signed(
                |v: &DVec| mat_apply(&r, n, v),
                |v: &DVec| mat_apply(&w, n, v),
                |_: &mut DVec| {},
                |_: &mut DVec| {},
                None::<fn(&mut DVec, f64)>,
                &v0s,
                1,
                n,
                &opts,
                super::super::davidson::REFUSE_TAU,
            );
            let (_, _, unsigned) = omega_davidson_vectors(
                |v: &DVec| mat_apply(&r, n, v),
                |v: &DVec| mat_apply(&w, n, v),
                |_: &mut DVec| {},
                |_: &mut DVec| {},
                None::<fn(&mut DVec, f64)>,
                &v0s,
                1,
                n,
                &opts,
            )
            .expect("PD metric");
            assert_eq!(
                signed.info.iterations, vectors,
                "signed, max_basis {max_basis}"
            );
            assert_eq!(
                unsigned.iterations, vectors,
                "unsigned, max_basis {max_basis}"
            );
            assert!(!signed.info.converged && !unsigned.converged);
            let (_, popts) = crate::solvers::hess::paired_route(&opts).expect("a paired route");
            assert_eq!(popts.max_basis, max_basis, "the paired route's budget");
        }
    }

    #[test]
    fn both_indefinite_reports_the_complex_pair_at_exhaustion() {
        let n = 16;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC5_C5);
        let mut rev = spread(n, 0.5, 4.0);
        rev[0] = -0.4;
        let mut wev = spread(n, 0.3, 3.0);
        wev[0] = -0.5;
        wev[1] = -0.3;
        let r = sym(&rev, &mut rng);
        let w = sym(&wev, &mut rng);
        let (real, cplx) = oracle(&r, &w, n);
        assert!(
            !cplx.is_empty(),
            "fixture precondition: a complex quartet exists"
        );
        let v0s = seeds(n, 4, &mut rng);
        let opts = SymplecticDavidsonOpts {
            tol: 1e-12,
            max_basis: 800,
            max_subspace: n,
            ..SymplecticDavidsonOpts::default()
        };
        let k = 2;
        let out = omega_davidson_signed(
            |v: &DVec| mat_apply(&r, n, v),
            |v: &DVec| mat_apply(&w, n, v),
            |_: &mut DVec| {},
            |_: &mut DVec| {},
            None::<fn(&mut DVec, f64)>,
            &v0s,
            k,
            n,
            &opts,
            super::super::davidson::REFUSE_TAU,
        );
        println!(
            "C5: converged {} built {} general {} real {:?} complex {:?} | dense real {:?} complex {:?}",
            out.info.converged,
            out.info.iterations,
            out.general_path(),
            out.omega2,
            out.complex.iter().map(|c| c.theta).collect::<Vec<_>>(),
            real,
            cplx
        );
        assert!(out.general_path(), "both blocks indefinite ⇒ T is not PD");
        let window = real
            .iter()
            .copied()
            .filter(|&x| x >= 0.0)
            .nth(k - 1)
            .unwrap_or(f64::INFINITY);
        let dense_in_window: Vec<_> = cplx.iter().filter(|z| z.re <= window).collect();
        assert_eq!(
            out.complex.len(),
            dense_in_window.len(),
            "complex pairs inside the window"
        );
        let scale = real
            .iter()
            .map(|x| x.abs())
            .chain(cplx.iter().map(|z| z.norm()))
            .fold(0.0f64, f64::max);
        for (c, z) in out.complex.iter().zip(&dense_in_window) {
            assert!(
                (c.theta - **z).norm() <= 1e-8 * scale,
                "complex θ {} vs dense {z}",
                c.theta
            );
            let mre = mat_apply(&r, n, &mat_apply(&w, n, &c.x_re));
            let mim = mat_apply(&r, n, &mat_apply(&w, n, &c.x_im));
            let (a, b) = (c.theta.re, c.theta.im);
            let mut res2 = 0.0;
            for i in 0..n {
                let rr = mre.0[i] - (a * c.x_re.0[i] - b * c.x_im.0[i]);
                let ri = mim.0[i] - (a * c.x_im.0[i] + b * c.x_re.0[i]);
                res2 += rr * rr + ri * ri;
            }
            let xn = (c.x_re.0.iter().chain(&c.x_im.0).map(|v| v * v).sum::<f64>()).sqrt();
            assert!(
                res2.sqrt() <= 1e-8 * scale * xn,
                "dense residual {:.2e}",
                res2.sqrt()
            );
            assert!(
                c.residual <= 1e-8 * scale * xn,
                "reported residual {:.2e}",
                c.residual
            );
        }
    }

    fn d0(r: &[f64], w: &[f64], n: usize) -> Option<impl FnMut(&mut DVec, f64) + use<>> {
        let dsq: Vec<f64> = (0..n)
            .map(|i| (0.5 * (r[i * n + i] + w[i * n + i])).powi(2))
            .collect();
        build_symp_precond(SympPrecond::DiagUnshifted { kappa: 0.0 }, || DVec(dsq))
    }

    #[allow(clippy::type_complexity)]
    fn precond_arms(
        r: &[f64],
        w: &[f64],
        n: usize,
    ) -> Vec<(&'static str, Option<Box<dyn FnMut(&mut DVec, f64)>>)> {
        let dg: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        vec![
            ("off", None),
            (
                "d0",
                d0(r, w, n).map(|p| Box::new(p) as Box<dyn FnMut(&mut DVec, f64)>),
            ),
            (
                "spd",
                Some(Box::new(move |t: &mut DVec, _th: f64| {
                    t.0.iter_mut().zip(&dg).for_each(|(x, d)| *x /= d)
                })),
            ),
        ]
    }

    fn assert_dense_spectrum(
        out: &super::SignedDavidsonOut<DVec>,
        (r, w, n, k): (&[f64], &[f64], usize, usize),
        case_ii: bool,
        tol: f64,
        arm: &str,
    ) {
        let (real, cplx) = oracle(r, w, n);
        let negs: Vec<f64> = real.iter().copied().filter(|&x| x < 0.0).collect();
        let pos: Vec<f64> = real.iter().copied().filter(|&x| x >= 0.0).take(k).collect();
        let want: Vec<f64> = negs.iter().chain(&pos).copied().collect();
        let scale = real
            .iter()
            .map(|x| x.abs())
            .chain(cplx.iter().map(|z| z.norm()))
            .fold(0.0f64, f64::max);
        assert!(out.info.converged, "{arm}: not converged");
        assert_eq!(out.omega2.len(), want.len(), "{arm}: {:?}", out.omega2);
        for (j, (&a, &b)) in out.omega2.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= tol * scale,
                "{arm}: ω²[{j}] {a} vs dense {b}"
            );
            if case_ii {
                assert_eq!(
                    out.signature()[j],
                    if b < 0.0 { -1 } else { 1 },
                    "{arm}: sign {j}"
                );
            }
        }
        assert_eq!(out.info.negative.len(), negs.len(), "{arm}: certificates");
        assert!(
            out.info.negative.iter().all(|c| c.certified()),
            "{arm}: {:?}",
            out.info.negative
        );
        let window = pos.last().copied().unwrap_or(f64::INFINITY);
        let dense_in: Vec<_> = cplx.iter().filter(|z| z.re <= window).collect();
        assert_eq!(out.complex.len(), dense_in.len(), "{arm}: complex pairs");
        for (c, z) in out.complex.iter().zip(&dense_in) {
            assert!(
                (c.theta - **z).norm() <= tol * scale,
                "{arm}: complex {} vs {z}",
                c.theta
            );
        }
    }

    #[test]
    fn d0_preconditioner_returns_the_dense_spectrum_on_both_indefinite_paths() {
        let n = 48;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC3_C3);
        let r = sym(&spread(n, 0.5, 5.0), &mut rng);
        let mut wev = spread(n, 0.25, 4.0);
        wev[0] = -0.6;
        wev[1] = -0.35;
        wev[2] = -0.2;
        let w = sym(&wev, &mut rng);
        let v0s = seeds(n, 6, &mut rng);
        let opts = SymplecticDavidsonOpts {
            tol: 6.5e-10,
            max_basis: 2400,
            max_subspace: 40,
            ..SymplecticDavidsonOpts::default()
        };
        let k = 4;
        for (arm, p) in precond_arms(&r, &w, n) {
            let arm = format!("II {arm}");
            let arm = arm.as_str();
            let out = omega_davidson_signed(
                |v: &DVec| mat_apply(&r, n, v),
                |v: &DVec| mat_apply(&w, n, v),
                |_: &mut DVec| {},
                |_: &mut DVec| {},
                p,
                &v0s,
                k,
                n,
                &opts,
                super::super::davidson::REFUSE_TAU,
            );
            println!(
                "{arm}: built {} applies {} neg_basis {} general {}",
                out.info.iterations,
                out.info.applies,
                out.n_negative_basis(),
                out.general_path()
            );
            assert!(
                out.metric_indefinite() && !out.general_path(),
                "{arm}: definite path"
            );
            assert_dense_spectrum(&out, (&r, &w, n, k), true, 1e-7, arm);
        }

        let n = 64;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC9_C9);
        let mut rev = spread(n, 0.5, 4.0);
        rev[0] = -0.4;
        let mut wev = spread(n, 0.3, 3.0);
        wev[0] = -0.5;
        wev[1] = -0.3;
        let r = sym(&rev, &mut rng);
        let w = sym(&wev, &mut rng);
        let (_, cplx) = oracle(&r, &w, n);
        let v0s = seeds(n, 6, &mut rng);
        let opts = SymplecticDavidsonOpts {
            tol: 5.8e-10,
            max_basis: 4000,
            max_subspace: 32,
            ..SymplecticDavidsonOpts::default()
        };
        let k = 3;
        println!("IV fixture: dense complex {cplx:?}");
        for (arm, p) in precond_arms(&r, &w, n) {
            let arm = format!("IV {arm}");
            let arm = arm.as_str();
            let out = omega_davidson_signed(
                |v: &DVec| mat_apply(&r, n, v),
                |v: &DVec| mat_apply(&w, n, v),
                |_: &mut DVec| {},
                |_: &mut DVec| {},
                p,
                &v0s,
                k,
                n,
                &opts,
                super::super::davidson::REFUSE_TAU,
            );
            let restarts = out.info.diag.as_ref().map_or(0, |d| d.restarts);
            println!(
                "{arm}: built {} applies {} restarts {restarts} neg_basis {} general {} conv {} ω² {:?} complex {:?}",
                out.info.iterations,
                out.info.applies,
                out.n_negative_basis(),
                out.general_path(),
                out.info.converged,
                out.omega2,
                out.complex.iter().map(|c| c.theta).collect::<Vec<_>>()
            );
            assert!(
                out.general_path(),
                "{arm}: both blocks indefinite ⇒ the general path"
            );
            assert!(
                restarts >= 1,
                "{arm}: the fixture must restart, not exhaust"
            );
            assert_dense_spectrum(&out, (&r, &w, n, k), false, 1e-7, arm);
        }
    }
}
