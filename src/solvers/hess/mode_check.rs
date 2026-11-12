// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::tangent::TtTangent;
use crate::solvers::InnerProductVector;

use super::newton_polish::{deflate_radial, radial_mode, reproject_tangent};
use super::{RpaAmplitudes, RpaMetric, RpaMode, RpaModeKind, RpaReadout, WeingartenCtx};

pub(crate) type TanOp<'a, V> = dyn Fn(&V) -> V + 'a;
pub(crate) type TanHook<'a, V> = dyn Fn(&mut V) + 'a;

pub(crate) struct ModeCheckOps<'a, V> {
    pub(crate) w: &'a TanOp<'a, V>,
    pub(crate) r: &'a TanOp<'a, V>,
    pub(crate) a: &'a TanOp<'a, V>,
    pub(crate) b: &'a TanOp<'a, V>,
    pub(crate) reproject: &'a TanHook<'a, V>,
    pub(crate) deflate: &'a TanHook<'a, V>,
    pub(crate) y_sign: f64,
}

/// Knobs of [`rpa_mode_checks`].
#[derive(Clone, Copy, Debug)]
pub struct ModeCheckOpts {}

#[allow(clippy::derivable_impls)]
impl Default for ModeCheckOpts {
    fn default() -> Self {
        ModeCheckOpts {}
    }
}

/// One mode's acceptance numbers ([module docs](self)); aligned with `RpaReadout::modes`.
#[derive(Clone, Copy, Debug)]
pub struct ModeCheck {
    /// Index into `RpaReadout::modes` / the amplitudes.
    pub index: usize,
    /// The mode's `ω²`; `NaN` for a complex mode.
    pub omega2: f64,
    /// `√ω²`; `NaN` on a non-positive entry.
    pub omega: f64,
    /// Krein signature from the fresh `⟨q, Wq⟩` (`0` when it vanished or the entry has no pair).
    pub signature: i8,
    /// Pencil residual `‖R p − ω q‖ / (ω ‖(p, q)‖)`.
    pub eta: f64,
    /// The pencil residual `‖Hz − ωΣz‖/‖z‖` recomputed with fresh applies (`= ω·η`): the mode's
    /// reported [`RpaMode::residual`].
    pub residual_fresh: f64,
    /// `‖X‖² + ‖Y‖²` under `⟨X,X⟩ − ⟨Y,Y⟩ = 1`.
    pub kappa: f64,
    /// `κ ω η`.
    pub delta_omega_est: f64,
    /// `E_ξ − λ = ⟨X̂, A X̂⟩` (electronic).
    pub e_state: f64,
    /// `⟨X̂, A X̂⟩ − ω`.
    pub delta_e: f64,
    /// `−⟨X, B Y⟩ / ‖X‖²`, the exact-pair value of `delta_e`.
    pub delta_e_pair: f64,
    /// `‖A X̂ − ⟨X̂, A X̂⟩ X̂‖`.
    pub sigma_t: f64,
    /// `|⟨q, Bq⟩| / |⟨q, Aq⟩|` (`+∞` when `⟨q, Aq⟩ = 0`).
    pub f_w: f64,
    /// `‖X − P ι X‖ / ‖X‖` of the assembled `X` (its reprojection is what the state uses).
    pub gauge_defect: f64,
}

impl ModeCheck {
    fn without_pair(index: usize, omega2: f64, f_w: f64) -> Self {
        ModeCheck {
            index,
            omega2,
            omega: f64::NAN,
            signature: 0,
            eta: f64::NAN,
            residual_fresh: f64::NAN,
            kappa: f64::NAN,
            delta_omega_est: f64::NAN,
            e_state: f64::NAN,
            delta_e: f64::NAN,
            delta_e_pair: f64::NAN,
            sigma_t: f64::NAN,
            f_w,
            gauge_defect: f64::NAN,
        }
    }
}

/// The per-mode acceptance numbers of a signed solve ([module docs](self)), one per entry of
/// `so.omega2`, from the amplitudes the `_vectors` entry point returned with it (same order).
/// Six Hessian-class applies per positive mode (`W q`, `R p`, `A X̂`, `B Y`, `A q`, `B q`), two
/// per other entry (`f_W`).
pub fn rpa_mode_checks(
    ctx: &WeingartenCtx<'_, f64>,
    readout: &RpaReadout,
    modes: &[RpaAmplitudes<TtTangent<f64>>],
    opts: &ModeCheckOpts,
) -> Vec<ModeCheck> {
    let _ = opts;
    let (base_l, base_r) = ctx.shared_bases();
    let radial = radial_mode(ctx);
    let hpp = |v: &TtTangent<f64>| ctx.hpp_apply(v);
    let hmm = |v: &TtTangent<f64>| ctx.hmm_apply(v);
    let (w, r, y_sign): (&TanOp<'_, TtTangent<f64>>, &TanOp<'_, TtTangent<f64>>, f64) =
        match readout.side() {
            RpaMetric::Rr => (&hpp, &hmm, -1.0),
            RpaMetric::Ii => (&hmm, &hpp, 1.0),
            RpaMetric::Auto => panic!("rpa_mode_checks: the solve's metric was never resolved"),
        };
    let a = |v: &TtTangent<f64>| ctx.a_apply(v);
    let b = |v: &TtTangent<f64>| ctx.b_apply(v);
    let reproject = |v: &mut TtTangent<f64>| {
        reproject_tangent(base_l, base_r, radial.as_ref(), v);
    };
    let deflate = |v: &mut TtTangent<f64>| {
        deflate_radial(radial.as_ref(), v);
    };
    let ops = ModeCheckOps {
        w,
        r,
        a: &a,
        b: &b,
        reproject: &reproject,
        deflate: &deflate,
        y_sign,
    };
    mode_checks_core(&readout.modes, modes, &ops)
}

pub(crate) fn mode_checks_core<V: InnerProductVector<Scalar = f64>>(
    readout_modes: &[RpaMode],
    modes: &[RpaAmplitudes<V>],
    ops: &ModeCheckOps<'_, V>,
) -> Vec<ModeCheck> {
    assert_eq!(
        modes.len(),
        readout_modes.len(),
        "rpa_mode_checks: one amplitude pair per mode"
    );
    let metric = if ops.y_sign > 0.0 {
        RpaMetric::Ii
    } else {
        RpaMetric::Rr
    };
    let scaled = |v: &V, c: f64| {
        let mut u = v.clone();
        u.scale(c);
        u
    };
    let fw_of = |v: &V| {
        let qa = v.dot(&(ops.a)(v));
        let qb = v.dot(&(ops.b)(v));
        if qa == 0.0 {
            f64::INFINITY
        } else {
            qb.abs() / qa.abs()
        }
    };
    let mut out = Vec::with_capacity(modes.len());
    for (index, (rm, mode)) in readout_modes.iter().zip(modes).enumerate() {
        let theta = if rm.kind == RpaModeKind::Complex {
            f64::NAN
        } else {
            rm.omega2.re
        };
        let mut q0 = mode.metric_side(metric);
        (ops.reproject)(&mut q0);
        let f_w = fw_of(&q0);
        if rm.kind != RpaModeKind::Real || !(theta.is_finite() && theta > 0.0) {
            out.push(ModeCheck::without_pair(index, theta, f_w));
            continue;
        }
        let omega = theta.sqrt();
        let mut p0 = mode.x.clone();
        p0.axpy(ops.y_sign, &mode.y);
        (ops.reproject)(&mut p0);
        let kn = p0.dot(&q0);
        if !(kn.is_finite() && kn != 0.0) {
            out.push(ModeCheck::without_pair(index, theta, f_w));
            continue;
        }
        let c = 1.0 / kn.abs().sqrt();
        let q = scaled(&q0, c);
        let p = scaled(&p0, c);
        let mut wq = (ops.w)(&q);
        (ops.deflate)(&mut wq);
        let wn = q.dot(&wq);
        if !(wn.is_finite() && wn != 0.0) {
            out.push(ModeCheck::without_pair(index, theta, f_w));
            continue;
        }
        let signature: i8 = if wn > 0.0 { 1 } else { -1 };
        let mut rp = (ops.r)(&p);
        (ops.reproject)(&mut rp);
        let mut res_r = rp;
        res_r.axpy(-omega, &q);
        let mut res_w = wq;
        res_w.axpy(-omega, &p);
        let pq = (p.norm().powi(2) + q.norm().powi(2)).sqrt();
        let pencil = (res_r.norm().powi(2) + res_w.norm().powi(2)).sqrt();
        let eta = pencil / (omega * pq);
        let residual_fresh = pencil / pq;
        let mut x = p.clone();
        x.axpy(1.0, &q);
        x.scale(0.5);
        let mut y = p.clone();
        y.axpy(-1.0, &q);
        y.scale(0.5 * ops.y_sign);
        if signature < 0 {
            std::mem::swap(&mut x, &mut y);
        }
        let kappa = x.norm().powi(2) + y.norm().powi(2);
        let mut xr = x.clone();
        (ops.reproject)(&mut xr);
        let mut dx = xr.clone();
        dx.axpy(-1.0, &x);
        let gauge_defect = dx.norm() / x.norm();
        let xn = xr.norm();
        let xh = scaled(&xr, 1.0 / xn);
        let mut ax = (ops.a)(&xh);
        (ops.deflate)(&mut ax);
        let e_state = xh.dot(&ax);
        let mut st = ax.clone();
        st.axpy(-e_state, &xh);
        let sigma_t = st.norm();
        let by = (ops.b)(&y);
        let delta_e_pair = -xr.dot(&by) / (xn * xn);
        out.push(ModeCheck {
            index,
            omega2: theta,
            omega,
            signature,
            eta,
            residual_fresh,
            kappa,
            delta_omega_est: kappa * omega * eta,
            e_state,
            delta_e: e_state - omega,
            delta_e_pair,
            sigma_t,
            f_w,
            gauge_defect,
        });
    }
    out
}
