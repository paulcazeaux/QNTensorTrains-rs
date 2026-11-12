// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Spin-summed transition one-particle RDMs over SU(2) (spin-adapted) tensor trains — the SU(2)
//! twin of [`crate::rdm`].

use crate::coeff::CoeffTensor;
use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::su2::core::Su2Core;
use crate::su2::coreops::{
    adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::Su2SparseHamiltonian;
use crate::su2::hamiltonian::assembly::WireGate;
use crate::su2::hamiltonian::matvec::{h_sketch_mul, sketch_h_mul};
use crate::su2::ops::dot;
use crate::su2::orbital::e_pq;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::spinalgebra::{SiteOp, SpinAlgebra, tensor_identity};
use crate::su2::tensorops::{WireSite, wire_core, wire_ranks};
use crate::su2::vector::Su2TtVector;

/// Per-entry reference implementation: `γ[p,q] = dot(x, E_pq·y)`, one exact one-body apply and one
/// full inner product per entry.
pub fn su2_transition_1rdm_bruteforce<T: Scalar>(
    x: &Su2TtVector<T>,
    y: &Su2TtVector<T>,
) -> CoeffTensor<T> {
    let space = x.space;
    assert_eq!(
        space, y.space,
        "su2_transition_1rdm: bra and ket must share a Su2Space"
    );
    let d = space.d;
    let pg = space.pg;
    let mut gamma = CoeffTensor::<T>::zeros(d, 2);
    for q in 0..d {
        for p in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                continue;
            }
            gamma.set(&[p, q], dot(x, &e_pq(y, p, q, T::one())));
        }
    }
    gamma
}

#[doc(hidden)]
pub fn su2_transition_1rdm_per_entry<T: Scalar>(
    x: &Su2TtVector<T>,
    y: &Su2TtVector<T>,
) -> CoeffTensor<T> {
    let space = x.space;
    assert_eq!(
        space, y.space,
        "su2_transition_1rdm: bra and ket must share a Su2Space"
    );
    let d = space.d;
    let pg = space.pg;
    let mut gamma = CoeffTensor::<T>::zeros(d, 2);
    for q in 0..d {
        for p in 0..d {
            if pg.irrep_of(p) != pg.irrep_of(q) {
                continue;
            }
            gamma.set(&[p, q], dot(&e_pq(x, q, p, T::one()), y));
        }
    }
    gamma
}

pub(crate) struct PairChannels {
    pub(crate) cre: SiteOp,
    pub(crate) ann: SiteOp,
    diag: SiteOp,
    id_red: [[f64; 3]; 3],
    pub(crate) wire: i64,
    pub(crate) w_lt: f64,
    pub(crate) w_gt: f64,
    w_eq: f64,
}

impl PairChannels {
    pub(crate) fn resolve(spin: &mut SpinAlgebra) -> Self {
        let lt = spin.term_channels(&[(0, true), (1, false)]);
        assert_eq!(lt.len(), 1, "su2 transition-1RDM: expected ONE p<q channel");
        let lt = lt[0].clone();
        let gt = spin.term_channels(&[(1, true), (0, false)]);
        assert_eq!(gt.len(), 1, "su2 transition-1RDM: expected ONE p>q channel");
        let gt = gt[0].clone();
        let eq = spin.term_channels(&[(0, true), (0, false)]);
        assert_eq!(
            eq.len(),
            1,
            "su2 transition-1RDM: expected ONE p==q channel"
        );
        let eq = eq[0].clone();

        assert_eq!(
            lt.ranks,
            vec![0, 1, 0],
            "su2 transition-1RDM: p<q rank path"
        );
        assert_eq!(
            gt.ranks,
            vec![0, 1, 0],
            "su2 transition-1RDM: p>q rank path"
        );
        assert_eq!(eq.ranks, vec![0, 0], "su2 transition-1RDM: p==q rank path");
        assert_eq!(
            (gt.siteops[0], gt.siteops[1]),
            (lt.siteops[1], lt.siteops[0]),
            "su2 transition-1RDM: the p>q pattern must reuse the p<q site ops in swapped roles"
        );

        let cre = spin.siteop(lt.siteops[0]).clone();
        let ann = spin.siteop(lt.siteops[1]).clone();
        let diag = spin.siteop(eq.siteops[0]).clone();
        let id_red = spin.reduce_siteop(&tensor_identity());
        PairChannels {
            cre,
            ann,
            diag,
            id_red,
            wire: lt.ranks[1] as i64,
            w_lt: lt.weight,
            w_gt: gt.weight,
            w_eq: eq.weight,
        }
    }

    pub(crate) fn site<'a>(
        op: &'a SiteOp,
        wire_l: i64,
        wire_r: i64,
        flux_l: i64,
        flux_r: i64,
        irr_l: Irrep,
        irr_r: Irrep,
    ) -> WireSite<'a> {
        WireSite {
            red: &op.red,
            op_odd: op.odd,
            op_two_k: op.two_k as i64,
            wire_l,
            wire_r,
            flux_l,
            flux_r,
            irr_l,
            irr_r,
        }
    }

    fn transport(&self, flux: i64, irr: Irrep) -> WireSite<'_> {
        WireSite {
            red: &self.id_red,
            op_odd: false,
            op_two_k: 0,
            wire_l: self.wire,
            wire_r: self.wire,
            flux_l: flux,
            flux_r: flux,
            irr_l: irr,
            irr_r: irr,
        }
    }
}

#[doc(hidden)]
pub fn su2_close_trace<T: Scalar>(lc: &Su2Sketch<T>, r: &Su2Sketch<T>) -> T {
    debug_assert_eq!(lc.k, r.k, "su2_close_trace: bond mismatch");
    let mut acc = T::zero();
    for s in lc.sectors() {
        let (Some(a), Some(b)) = (lc.block(s), r.block(s)) else {
            continue;
        };
        if a.nrows() == 0 || a.ncols() == 0 || b.nrows() == 0 || b.ncols() == 0 {
            continue;
        }
        debug_assert_eq!(
            a.ncols(),
            b.nrows(),
            "su2_close_trace: shape mismatch at {s:?}"
        );
        debug_assert_eq!(
            a.nrows(),
            b.ncols(),
            "su2_close_trace: shape mismatch at {s:?}"
        );
        for j in 0..a.ncols() {
            for i in 0..a.nrows() {
                acc += *a.get(i, j) * *b.get(j, i);
            }
        }
    }
    acc
}

struct WireGrids<'r> {
    grids: Vec<Option<Vec<Su2RankGrid>>>,
    src: &'r [Su2RankGrid],
    space: Su2Space,
    flux: i64,
    wire: i64,
}

impl<'r> WireGrids<'r> {
    fn new(src: &'r [Su2RankGrid], space: Su2Space, order: usize, wire: i64, flux: i64) -> Self {
        WireGrids {
            grids: vec![None; order],
            src,
            space,
            flux,
            wire,
        }
    }

    fn get(&mut self, irr: Irrep) -> &[Su2RankGrid] {
        let i = usize::from(irr.index());
        if self.grids[i].is_none() {
            let g = (0..self.src.len())
                .map(|b| {
                    wire_ranks(
                        &self.src[b],
                        self.wire,
                        self.flux,
                        irr,
                        self.space,
                        self.space,
                        b,
                        WireGate::Window,
                    )
                })
                .collect();
            self.grids[i] = Some(g);
        }
        self.grids[i].as_deref().expect("grid built above")
    }
}

struct TransportCache<T: Scalar> {
    cores: Vec<Option<Su2Core<T>>>,
}

impl<T: Scalar> TransportCache<T> {
    fn new(order: usize) -> Self {
        TransportCache {
            cores: (0..order).map(|_| None).collect(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn get(
        &mut self,
        ch: &PairChannels,
        src: &Su2Core<T>,
        src_ranks: (&Su2RankGrid, &Su2RankGrid),
        space: Su2Space,
        grids: &[Su2RankGrid],
        flux: i64,
        irr: Irrep,
        spin: &mut SpinAlgebra,
    ) -> &Su2Core<T> {
        let i = usize::from(irr.index());
        if self.cores[i].is_none() {
            let k = src.k;
            let site = ch.transport(flux, irr);
            self.cores[i] = Some(wire_core(
                src,
                src_ranks,
                space,
                space,
                (&grids[k], &grids[k + 1]),
                &site,
                spin,
                WireGate::Window,
            ));
        }
        self.cores[i].as_ref().expect("core built above")
    }
}

fn assert_unit_boundaries<T: Scalar>(v: &Su2TtVector<T>, who: &str) {
    let space = v.space;
    assert_eq!(
        v.rank(0, Su2Sector::new(0, 0)),
        1,
        "su2 transition-1RDM needs unit rank at bond 0 ({who})"
    );
    assert_eq!(
        v.rank(
            space.d,
            Su2Sector::with_irrep(space.n, space.two_s, space.target)
        ),
        1,
        "su2 transition-1RDM needs unit rank at bond d ({who})"
    );
}

/// `γ[p,q] = ⟨x| E_pq |y⟩` — all `d²` entries in ONE left-to-right pass.
pub fn su2_transition_1rdm<T: Scalar>(x: &Su2TtVector<T>, y: &Su2TtVector<T>) -> CoeffTensor<T> {
    let space = x.space;
    assert_eq!(
        space, y.space,
        "su2_transition_1rdm: bra and ket must share a Su2Space"
    );
    let d = space.d;
    let pg = space.pg;
    let mut gamma = CoeffTensor::<T>::zeros(d, 2);
    if d == 0 {
        return gamma;
    }
    assert_unit_boundaries(x, "x");
    assert_unit_boundaries(y, "y");

    let mut spin = SpinAlgebra::new();
    let ch = PairChannels::resolve(&mut spin);
    let order = usize::from(pg.order());

    let mut l: Vec<Su2Sketch<T>> = Vec::with_capacity(d);
    l.push(Su2Sketch::identity(space, 0, 1));
    for k in 0..d - 1 {
        let t = mul_sketch_left(&l[k], y.core(k));
        l.push(adjoint_core_times_core(x.core(k), &t));
    }
    let mut r: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    r[d] = Some(Su2Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let t = mul_sketch_right(
            y.core(k),
            r[k + 1].as_ref().expect("right env built rightward"),
        );
        r[k] = Some(core_times_adjoint_core(&t, x.core(k)));
    }

    let mut rp = WireGrids::new(&y.ranks, space, order, ch.wire, 1);
    let mut rm = WireGrids::new(&y.ranks, space, order, ch.wire, -1);

    let mut g = vec![T::zero(); d * d];
    let mut live_cre: Vec<(usize, Su2Sketch<T>)> = Vec::new();
    let mut live_ann: Vec<(usize, Su2Sketch<T>)> = Vec::new();

    for k in 0..d {
        let g_k = pg.irrep_of(k);
        let xk = x.core(k);
        let yk = y.core(k);
        let sr = (&y.ranks[k], &y.ranks[k + 1]);
        let rk1 = r[k + 1].as_ref().expect("right env present at bond k+1");

        if live_cre.iter().any(|&(a, _)| pg.irrep_of(a) == g_k) {
            let grid = rp.get(g_k);
            let site = PairChannels::site(&ch.ann, ch.wire, 0, 1, 0, g_k, Irrep::IDENTITY);
            let ann_close = wire_core(
                yk,
                sr,
                space,
                space,
                (&grid[k], &y.ranks[k + 1]),
                &site,
                &mut spin,
                WireGate::Window,
            );
            for (a, dch) in live_cre.iter() {
                if pg.irrep_of(*a) != g_k {
                    continue;
                }
                let closed = adjoint_core_times_core(xk, &mul_sketch_left(dch, &ann_close));
                g[*a * d + k] += T::from_f64(ch.w_lt) * su2_close_trace(&closed, rk1);
            }
        }
        if live_ann.iter().any(|&(a, _)| pg.irrep_of(a) == g_k) {
            let grid = rm.get(g_k);
            let site = PairChannels::site(&ch.cre, ch.wire, 0, -1, 0, g_k, Irrep::IDENTITY);
            let cre_close = wire_core(
                yk,
                sr,
                space,
                space,
                (&grid[k], &y.ranks[k + 1]),
                &site,
                &mut spin,
                WireGate::Window,
            );
            for (a, dch) in live_ann.iter() {
                if pg.irrep_of(*a) != g_k {
                    continue;
                }
                let closed = adjoint_core_times_core(xk, &mul_sketch_left(dch, &cre_close));
                g[k * d + *a] += T::from_f64(ch.w_gt) * su2_close_trace(&closed, rk1);
            }
        }

        {
            let site = PairChannels::site(&ch.diag, 0, 0, 0, 0, Irrep::IDENTITY, Irrep::IDENTITY);
            let n_diag = wire_core(yk, sr, space, space, sr, &site, &mut spin, WireGate::Window);
            let closed = adjoint_core_times_core(xk, &mul_sketch_left(&l[k], &n_diag));
            g[k * d + k] += T::from_f64(ch.w_eq) * su2_close_trace(&closed, rk1);
        }

        if k + 1 >= d {
            continue;
        }

        let mut id_plus = TransportCache::<T>::new(order);
        for (a, dch) in live_cre.iter_mut() {
            let irr = pg.irrep_of(*a);
            let grid = rp.get(irr);
            let idc = id_plus.get(&ch, yk, sr, space, grid, 1, irr, &mut spin);
            *dch = adjoint_core_times_core(xk, &mul_sketch_left(dch, idc));
        }
        let mut id_minus = TransportCache::<T>::new(order);
        for (a, dch) in live_ann.iter_mut() {
            let irr = pg.irrep_of(*a);
            let grid = rm.get(irr);
            let idc = id_minus.get(&ch, yk, sr, space, grid, -1, irr, &mut spin);
            *dch = adjoint_core_times_core(xk, &mul_sketch_left(dch, idc));
        }

        {
            let grid = rp.get(g_k);
            let site = PairChannels::site(&ch.cre, 0, ch.wire, 0, 1, Irrep::IDENTITY, g_k);
            let cre_start = wire_core(
                yk,
                sr,
                space,
                space,
                (&y.ranks[k], &grid[k + 1]),
                &site,
                &mut spin,
                WireGate::Window,
            );
            live_cre.push((
                k,
                adjoint_core_times_core(xk, &mul_sketch_left(&l[k], &cre_start)),
            ));
        }
        {
            let grid = rm.get(g_k);
            let site = PairChannels::site(&ch.ann, 0, ch.wire, 0, -1, Irrep::IDENTITY, g_k);
            let ann_start = wire_core(
                yk,
                sr,
                space,
                space,
                (&y.ranks[k], &grid[k + 1]),
                &site,
                &mut spin,
                WireGate::Window,
            );
            live_ann.push((
                k,
                adjoint_core_times_core(xk, &mul_sketch_left(&l[k], &ann_start)),
            ));
        }
    }
    debug_assert_eq!(live_cre.len(), d - 1, "a chain opened at the last site");
    debug_assert_eq!(live_ann.len(), d - 1, "a chain opened at the last site");

    for p in 0..d {
        for q in 0..d {
            gamma.set(&[p, q], g[p * d + q]);
        }
    }
    gamma
}

/// `γ[p,q] = ⟨x| E_pq H |w⟩` — the shape [`crate::su2::orbital::orbital_gradient`] and the joint
/// covector rows consume, computed by environment contraction with `H·w` **NEVER materialized**.
/// One left-to-right sweep, `O(d²)` mixed-`H` steps.
pub fn su2_transition_1rdm_h<T: Scalar>(
    x: &Su2TtVector<T>,
    h: &Su2SparseHamiltonian<T>,
    w: &Su2TtVector<T>,
) -> CoeffTensor<T> {
    let space = x.space;
    assert_eq!(
        space, w.space,
        "su2_transition_1rdm_h: bra and ket must share a Su2Space"
    );
    assert_eq!(
        space, h.space,
        "su2_transition_1rdm_h: state and operator must share a Su2Space"
    );
    let d = space.d;
    let pg = space.pg;
    let mut gamma = CoeffTensor::<T>::zeros(d, 2);
    if d == 0 {
        return gamma;
    }
    assert_unit_boundaries(x, "x");
    assert_unit_boundaries(w, "w");

    let mut spin = SpinAlgebra::new();
    let ch = PairChannels::resolve(&mut spin);
    let order = usize::from(pg.order());

    let mut l: Vec<Su2Sketch<T>> = Vec::with_capacity(d);
    l.push(Su2Sketch::identity(space, 0, 1));
    for k in 0..d - 1 {
        let t = sketch_h_mul(&l[k], h, w.core(k));
        l.push(adjoint_core_times_core(x.core(k), &t));
    }
    let mut r: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    r[d] = Some(Su2Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let t = h_sketch_mul(
            h,
            w.core(k),
            r[k + 1].as_ref().expect("right env built rightward"),
        );
        r[k] = Some(core_times_adjoint_core(&t, x.core(k)));
    }

    let mut rp = WireGrids::new(&x.ranks, space, order, ch.wire, 1);
    let mut rm = WireGrids::new(&x.ranks, space, order, ch.wire, -1);

    let mut g = vec![T::zero(); d * d];
    let mut live_cre: Vec<(usize, Su2Sketch<T>)> = Vec::new();
    let mut live_ann: Vec<(usize, Su2Sketch<T>)> = Vec::new();

    for k in 0..d {
        let g_k = pg.irrep_of(k);
        let xk = x.core(k);
        let wk = w.core(k);
        let sr = (&x.ranks[k], &x.ranks[k + 1]);
        let rk1 = r[k + 1].as_ref().expect("right env present at bond k+1");
        let last = k + 1 >= d;
        let i_tab = h.row_ranges(wk);
        let j_tab = h.col_ranges(wk);
        let h_cols = h.col_ranks(wk);
        let mix = |fl: &Su2Sketch<T>| -> Su2Core<T> {
            let mut y = Su2Core::zeros(space, k, &fl.row_ranks(), &h_cols);
            h.sketch_h_mul_into(&mut y, fl, wk, |_| true, &i_tab, &j_tab);
            y
        };

        let mut id_plus = TransportCache::<T>::new(order);
        let mut id_minus = TransportCache::<T>::new(order);

        let ann_close = live_cre
            .iter()
            .any(|&(a, _)| pg.irrep_of(a) == g_k)
            .then(|| {
                let grid = rp.get(g_k);
                let site = PairChannels::site(&ch.ann, ch.wire, 0, 1, 0, g_k, Irrep::IDENTITY);
                wire_core(
                    xk,
                    sr,
                    space,
                    space,
                    (&grid[k], &x.ranks[k + 1]),
                    &site,
                    &mut spin,
                    WireGate::Window,
                )
            });
        for (a, dch) in live_cre.iter_mut() {
            let irr = pg.irrep_of(*a);
            if last && irr != g_k {
                continue;
            }
            let gmix = mix(dch);
            if irr == g_k {
                let close = ann_close.as_ref().expect("built when a chain can close");
                let closed = adjoint_core_times_core(close, &gmix);
                g[k * d + *a] += T::from_f64(ch.w_lt) * su2_close_trace(&closed, rk1);
            }
            if !last {
                let grid = rp.get(irr);
                let idc = id_plus.get(&ch, xk, sr, space, grid, 1, irr, &mut spin);
                *dch = adjoint_core_times_core(idc, &gmix);
            }
        }

        let cre_close = live_ann
            .iter()
            .any(|&(a, _)| pg.irrep_of(a) == g_k)
            .then(|| {
                let grid = rm.get(g_k);
                let site = PairChannels::site(&ch.cre, ch.wire, 0, -1, 0, g_k, Irrep::IDENTITY);
                wire_core(
                    xk,
                    sr,
                    space,
                    space,
                    (&grid[k], &x.ranks[k + 1]),
                    &site,
                    &mut spin,
                    WireGate::Window,
                )
            });
        for (a, dch) in live_ann.iter_mut() {
            let irr = pg.irrep_of(*a);
            if last && irr != g_k {
                continue;
            }
            let gmix = mix(dch);
            if irr == g_k {
                let close = cre_close.as_ref().expect("built when a chain can close");
                let closed = adjoint_core_times_core(close, &gmix);
                g[*a * d + k] += T::from_f64(ch.w_gt) * su2_close_trace(&closed, rk1);
            }
            if !last {
                let grid = rm.get(irr);
                let idc = id_minus.get(&ch, xk, sr, space, grid, -1, irr, &mut spin);
                *dch = adjoint_core_times_core(idc, &gmix);
            }
        }

        let gl = mix(&l[k]);

        {
            let site = PairChannels::site(&ch.diag, 0, 0, 0, 0, Irrep::IDENTITY, Irrep::IDENTITY);
            let n_diag = wire_core(xk, sr, space, space, sr, &site, &mut spin, WireGate::Window);
            let closed = adjoint_core_times_core(&n_diag, &gl);
            g[k * d + k] += T::from_f64(ch.w_eq) * su2_close_trace(&closed, rk1);
        }

        if last {
            continue;
        }

        {
            let grid = rp.get(g_k);
            let site = PairChannels::site(&ch.cre, 0, ch.wire, 0, 1, Irrep::IDENTITY, g_k);
            let cre_start = wire_core(
                xk,
                sr,
                space,
                space,
                (&x.ranks[k], &grid[k + 1]),
                &site,
                &mut spin,
                WireGate::Window,
            );
            live_cre.push((k, adjoint_core_times_core(&cre_start, &gl)));
        }
        {
            let grid = rm.get(g_k);
            let site = PairChannels::site(&ch.ann, 0, ch.wire, 0, -1, Irrep::IDENTITY, g_k);
            let ann_start = wire_core(
                xk,
                sr,
                space,
                space,
                (&x.ranks[k], &grid[k + 1]),
                &site,
                &mut spin,
                WireGate::Window,
            );
            live_ann.push((k, adjoint_core_times_core(&ann_start, &gl)));
        }
    }
    debug_assert_eq!(live_cre.len(), d - 1, "a chain opened at the last site");
    debug_assert_eq!(live_ann.len(), d - 1, "a chain opened at the last site");

    for p in 0..d {
        for q in 0..d {
            gamma.set(&[p, q], g[p * d + q]);
        }
    }
    gamma
}

#[cfg(test)]
mod tests {
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::rdm::{transition_1rdm, transition_1rdm_h};
    use crate::round::RoundMode;
    use crate::su2::coreops::{
        adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
        rproject,
    };
    use crate::su2::expand::expand;
    use crate::su2::frame::Su2Sketch;
    use crate::su2::hamiltonian::{Su2SparseHamiltonian, xt_h_y};
    use crate::su2::ops::dot as su2_dot;
    use crate::su2::qn::Su2Space;
    use crate::su2::tensorops::n_i;
    use crate::su2::theta;
    use crate::su2::transition1rdm::{
        su2_close_trace, su2_transition_1rdm, su2_transition_1rdm_bruteforce,
        su2_transition_1rdm_h, su2_transition_1rdm_per_entry,
    };
    use crate::su2::vector::Su2TtVector;
    use crate::test_util::{randomized_su2_state, seeded};
    use crate::vector::TtVector;

    const ST_FIX: [(usize, usize, usize); 3] = [(6, 3, 1), (5, 2, 0), (4, 1, 1)];

    fn st_state(space: Su2Space, lo: usize, hi: usize, seed: u64) -> Su2TtVector<f64> {
        let mut rng = seeded(seed);
        let mut v: Su2TtVector<f64> = randomized_su2_state(space, lo, hi, &mut rng);
        v.round_mut(1e-12, crate::qn::Pooled::NONE, RoundMode::Rel);
        v
    }

    fn spin_free_tv(d: usize) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        for p in 0..d - 1 {
            t.set(&[p, p + 1], -1.0);
            t.set(&[p + 1, p], -1.0);
        }
        for p in 0..d {
            t.set(&[p, p], 0.15 * (p as f64 - 0.5 * d as f64));
        }
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for p in 0..d {
            for q in 0..d {
                if p != q {
                    v.set(&[p, p, q, q], 0.5 / (1.0 + (p as f64 - q as f64).abs()));
                }
            }
        }
        (t, v)
    }

    fn h_materialized(
        x: &Su2TtVector<f64>,
        h: &Su2SparseHamiltonian<f64>,
        w: &Su2TtVector<f64>,
    ) -> CoeffTensor<f64> {
        su2_transition_1rdm(x, &h.apply(w))
    }

    fn maxg_maxdiff(g: &CoeffTensor<f64>, r: &CoeffTensor<f64>, d: usize) -> (f64, f64) {
        let mut maxg = 0.0f64;
        let mut maxd = 0.0f64;
        for p in 0..d {
            for q in 0..d {
                maxg = maxg.max(r.get(&[p, q]).abs());
                maxd = maxd.max((g.get(&[p, q]) - r.get(&[p, q])).abs());
            }
        }
        (maxg, maxd)
    }

    fn scale_of(x: &Su2TtVector<f64>, y: &Su2TtVector<f64>) -> f64 {
        (x.norm() * y.norm()).max(1.0)
    }

    fn members(two_s: usize) -> Vec<i64> {
        let s = two_s as i64;
        (0..=two_s).map(|i| -s + 2 * i as i64).collect()
    }

    fn st_partials(
        x: &Su2TtVector<f64>,
        y: &Su2TtVector<f64>,
        dual: bool,
    ) -> (Vec<Su2Sketch<f64>>, Vec<Su2Sketch<f64>>) {
        let space = x.space;
        let d = space.d;
        let mut l: Vec<Su2Sketch<f64>> = Vec::with_capacity(d + 1);
        l.push(Su2Sketch::identity(space, 0, 1));
        for k in 0..d {
            let t = mul_sketch_left(&l[k], y.core(k));
            l.push(adjoint_core_times_core(x.core(k), &t));
        }
        let mut r: Vec<Option<Su2Sketch<f64>>> = vec![None; d + 1];
        r[d] = Some(Su2Sketch::identity(space, d, 1));
        for k in (0..d).rev() {
            let a = mul_sketch_right(y.core(k), r[k + 1].as_ref().expect("built rightward"));
            r[k] = Some(if dual {
                rproject(&a, x.core(k))
            } else {
                core_times_adjoint_core(&a, x.core(k))
            });
        }
        (l, r.into_iter().map(|s| s.expect("all built")).collect())
    }

    fn has_spinful_interior_rank(x: &Su2TtVector<f64>) -> bool {
        let space = x.space;
        (1..space.d).any(|b| {
            space
                .bond_sectors(b)
                .iter()
                .any(|&s| s.two_s > 0 && x.rank(b, s) > 0)
        })
    }

    #[test]
    fn tst0_closure_identity_and_theta_anti_oracle() {
        for (fi, &(d, n, two_s)) in ST_FIX.iter().enumerate() {
            let space = Su2Space::new(d, n, two_s);
            let x = st_state(space, 1, 2, 0x5700 + fi as u64);
            let y = st_state(space, 1, 3, 0x5780 + fi as u64);
            let refv = su2_dot(&x, &y);
            assert!(
                refv.abs() > 1e-6,
                "vacuous fixture (d={d}, N={n}, 2S={two_s}): dot(x,y) = {refv}"
            );
            let tol = 1e-11 * refv.abs().max(1.0);

            let (l, r) = st_partials(&x, &y, false);
            for k in 0..=d {
                let c = su2_close_trace(&l[k], &r[k]);
                assert!(
                    (c - refv).abs() <= tol,
                    "R2 closure identity broken at bond {k} (d={d}, N={n}, 2S={two_s}): \
                     Σ tr(L·R) = {c}, dot(x,y) = {refv}"
                );
            }

            if has_spinful_interior_rank(&x) {
                let (ld, rd) = st_partials(&x, &y, true);
                let broke = (1..d).any(|k| {
                    (su2_close_trace(&ld[k], &rd[k]) - refv).abs() > 1e-6 * refv.abs().max(1.0)
                });
                assert!(
                    broke,
                    "the θ²-weighted (rproject) right fold satisfied the closure identity at every \
                     interior bond (d={d}, N={n}, 2S={two_s}) — the anti-oracle has gone vacuous, so \
                     ST0 is no longer detecting an R1/R2 confusion"
                );
            } else {
                assert_eq!(
                    two_s, 0,
                    "a 2S ≥ 1 fixture with no spinful interior rank makes the anti-oracle vacuous"
                );
            }
        }
    }

    #[test]
    fn tst1_sweep_matches_both_per_entry_oracles() {
        for (fi, &(d, n, two_s)) in ST_FIX.iter().enumerate() {
            let space = Su2Space::new(d, n, two_s);
            let x = st_state(space, 1, 2, 0x5800 + fi as u64);
            let y = st_state(space, 1, 3, 0x5880 + fi as u64);
            let sc = scale_of(&x, &y);

            let gs = su2_transition_1rdm(&x, &y);
            let gb = su2_transition_1rdm_bruteforce(&x, &y);
            let gp = su2_transition_1rdm_per_entry(&x, &y);

            let (maxg, db) = maxg_maxdiff(&gs, &gb, d);
            assert!(maxg > 1e-6, "vacuous γ (d={d}, N={n}, 2S={two_s})");
            assert!(
                db <= 1e-13 * sc,
                "sweep vs bruteforce: maxdiff {db} > {} (d={d}, N={n}, 2S={two_s})",
                1e-13 * sc
            );
            let (_, dp) = maxg_maxdiff(&gs, &gp, d);
            assert!(
                dp <= 1e-13 * sc,
                "sweep vs per-entry adjoint route: maxdiff {dp} > {} (d={d}, N={n}, 2S={two_s})",
                1e-13 * sc
            );
        }
    }

    #[test]
    fn tst1b_sweep_matches_the_abelian_gamma_through_expand() {
        for (fi, &(d, n, two_s)) in ST_FIX.iter().enumerate() {
            let space = Su2Space::new(d, n, two_s);
            let x = st_state(space, 1, 2, 0x5900 + fi as u64);
            let y = st_state(space, 1, 3, 0x5980 + fi as u64);
            let sc = scale_of(&x, &y);
            let gs = su2_transition_1rdm(&x, &y);

            for two_sz in members(two_s) {
                let ax: TtVector<f64> = expand(&x, two_sz);
                let ay: TtVector<f64> = expand(&y, two_sz);
                let ga = transition_1rdm(&ax, &ay);
                let (maxg, diff) = maxg_maxdiff(&gs, &ga, d);
                assert!(maxg > 1e-6, "vacuous abelian γ at 2Sz={two_sz}");
                assert!(
                    diff <= 1e-11 * sc,
                    "su2 sweep vs abelian γ at 2Sz={two_sz}: maxdiff {diff} > {} \
                     (d={d}, N={n}, 2S={two_s})",
                    1e-11 * sc
                );
            }
        }
    }

    #[test]
    fn tst2_h_dressed_sweep_matches_the_materialized_identity() {
        for (fi, &(d, n, two_s)) in ST_FIX.iter().enumerate() {
            let space = Su2Space::new(d, n, two_s);
            let (t, v) = spin_free_tv(d);
            let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
            let x = st_state(space, 1, 2, 0x5A00 + fi as u64);
            let y = st_state(space, 1, 3, 0x5A80 + fi as u64);
            let sc = scale_of(&x, &y);

            let gh = h_materialized(&x, &h, &y);
            let gs = su2_transition_1rdm_h(&x, &h, &y);
            let hy = h.apply(&y);
            let hsc = (x.norm() * hy.norm()).max(1.0);
            let (maxg, diff) = maxg_maxdiff(&gs, &gh, d);
            assert!(maxg > 1e-6, "vacuous γ_H (d={d}, N={n}, 2S={two_s})");
            assert!(
                diff <= 1e-12 * hsc,
                "H-streamed sweep vs the materialized route: maxdiff {diff} > {} \
                 (d={d}, N={n}, 2S={two_s})",
                1e-12 * hsc
            );

            let want = n as f64 * xt_h_y(&x, &h, &y);
            for (name, g) in [("materialized", &gh), ("streamed", &gs)] {
                let tr: f64 = (0..d).map(|p| g.get(&[p, p])).sum();
                assert!(
                    (tr - want).abs() <= 1e-10 * (want.abs()).max(sc),
                    "{name}: tr γ_H = {tr}, expected N·xᵀHy = {want} (d={d}, N={n}, 2S={two_s})"
                );
            }
        }
    }

    #[test]
    fn tst2c_h_dressed_sweep_matches_materialized_c64() {
        use crate::C64;
        let (d, n, two_s) = (5usize, 2usize, 0usize);
        let (d2, n2, two_s2) = (4usize, 1usize, 1usize);
        for (fi, (d, n, two_s)) in [(d, n, two_s), (d2, n2, two_s2)].into_iter().enumerate() {
            let space = Su2Space::new(d, n, two_s);
            let (t, v) = spin_free_tv(d);
            let theta: Vec<f64> = (0..d).map(|p| 0.29 + 0.91 * p as f64).collect();
            let ph = |a: f64| C64::new(a.cos(), a.sin());
            let mut tc = CoeffTensor::<C64>::zeros(d, 2);
            for p in 0..d {
                for q in 0..d {
                    let c = t.get(&[p, q]);
                    if c != 0.0 {
                        tc.set(&[p, q], ph(theta[p] - theta[q]) * C64::new(c, 0.0));
                    }
                }
            }
            let mut vc = CoeffTensor::<C64>::zeros(d, 4);
            for p in 0..d {
                for q in 0..d {
                    for r in 0..d {
                        for s in 0..d {
                            let c = v.get(&[p, q, r, s]);
                            if c != 0.0 {
                                vc.set(
                                    &[p, q, r, s],
                                    ph(theta[p] - theta[q] + theta[r] - theta[s])
                                        * C64::new(c, 0.0),
                                );
                            }
                        }
                    }
                }
            }
            let h = Su2SparseHamiltonian::<C64>::new(space, &tc, &vc, None, f64::EPSILON);
            let mut rng = seeded(0x57C6_4000 + fi as u64);
            let x: Su2TtVector<C64> = randomized_su2_state(space, 1, 2, &mut rng);
            let y: Su2TtVector<C64> = randomized_su2_state(space, 1, 3, &mut rng);

            let hy = h.apply(&y);
            let gh = su2_transition_1rdm(&x, &hy);
            let gs = su2_transition_1rdm_h(&x, &h, &y);
            let hsc = (x.norm() * hy.norm()).max(1.0);
            let mut maxg = 0.0f64;
            let mut maxd = 0.0f64;
            let mut max_im = 0.0f64;
            for p in 0..d {
                for q in 0..d {
                    let a = gs.get(&[p, q]);
                    let b = gh.get(&[p, q]);
                    maxg = maxg.max(a.norm()).max(b.norm());
                    maxd = maxd.max((a - b).norm());
                    max_im = max_im.max(a.im.abs());
                }
            }
            assert!(maxg > 1e-6, "vacuous C64 γ_H (d={d}, N={n}, 2S={two_s})");
            assert!(
                max_im > 1e-3 * maxg,
                "anti-vacuity: gauge-conjugated γ_H must carry O(1) imaginary mass, got \
                 max|Im| = {max_im:e} against {maxg:e} (d={d}, N={n}, 2S={two_s})"
            );
            assert!(
                maxd <= 1e-12 * hsc,
                "[c64] H-streamed sweep vs materialized: maxdiff {maxd:e} > {:e} \
                 (d={d}, N={n}, 2S={two_s})",
                1e-12 * hsc
            );
        }
    }

    #[test]
    fn tst2b_h_dressed_sweep_matches_the_abelian_value_through_expand() {
        for (fi, &(d, n, two_s)) in ST_FIX.iter().enumerate() {
            let space = Su2Space::new(d, n, two_s);
            let (t, v) = spin_free_tv(d);
            let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
            let x = st_state(space, 1, 2, 0x5B00 + fi as u64);
            let y = st_state(space, 1, 3, 0x5B80 + fi as u64);
            let gh = h_materialized(&x, &h, &y);
            let gs = su2_transition_1rdm_h(&x, &h, &y);

            for two_sz in members(two_s) {
                let ax: TtVector<f64> = expand(&x, two_sz);
                let ay: TtVector<f64> = expand(&y, two_sz);
                let ha = SparseHamiltonian::new(ax.space, &t, &v, None, f64::EPSILON);
                let ga = transition_1rdm_h(&ax, &ha, &ay);
                let sc = (ax.norm() * ha.apply(&ay).norm()).max(1.0);
                let (maxg, _) = maxg_maxdiff(&gh, &ga, d);
                assert!(maxg > 1e-6, "vacuous abelian γ_H at 2Sz={two_sz}");
                for (name, g) in [("materialized", &gh), ("streamed", &gs)] {
                    let (_, diff) = maxg_maxdiff(g, &ga, d);
                    assert!(
                        diff <= 1e-10 * sc,
                        "su2 γ_H ({name}) vs abelian γ_H at 2Sz={two_sz}: maxdiff {diff} > {} \
                         (d={d}, N={n}, 2S={two_s})",
                        1e-10 * sc
                    );
                }
            }
        }
    }

    #[test]
    fn tst3_consumer_identities() {
        let (d, n, two_s) = (6usize, 3usize, 1usize);
        let space = Su2Space::new(d, n, two_s);
        let (t, v) = spin_free_tv(d);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let x = st_state(space, 1, 2, 0x5C01);
        let w = st_state(space, 1, 3, 0x5C02);
        let lambda = 0.7f64;

        let gh = su2_transition_1rdm_h(&x, &h, &w);
        let g0 = su2_transition_1rdm(&x, &w);
        let hw = h.apply(&w);
        let rhs_ket = &hw - &(&w * lambda);
        let rhs = su2_transition_1rdm(&x, &rhs_ket);
        let sc = (x.norm() * rhs_ket.norm()).max(1.0);
        for p in 0..d {
            for q in 0..d {
                let lhs = gh.get(&[p, q]) - lambda * g0.get(&[p, q]);
                assert!(
                    (lhs - rhs.get(&[p, q])).abs() <= 1e-10 * sc,
                    "linearity in the ket failed at [{p},{q}]: {lhs} vs {}",
                    rhs.get(&[p, q])
                );
            }
        }

        let v1 = st_state(space, 1, 2, 0x5C03);
        let hx = h.apply(&x);
        let lhs = su2_transition_1rdm(&hx, &v1);
        let rhs = su2_transition_1rdm_h(&v1, &h, &x);
        let sc = (hx.norm() * v1.norm()).max(1.0);
        let mut maxg = 0.0f64;
        for p in 0..d {
            for q in 0..d {
                maxg = maxg.max(lhs.get(&[p, q]).abs());
                assert!(
                    (lhs.get(&[p, q]) - rhs.get(&[q, p])).abs() <= 1e-10 * sc,
                    "transpose identity failed at [{p},{q}]"
                );
            }
        }
        assert!(maxg > 1e-6, "vacuous transpose-identity arm");

        let gxx = su2_transition_1rdm(&x, &x);
        let sxx = (x.norm() * x.norm()).max(1.0);
        for p in 0..d {
            for q in 0..d {
                assert!(
                    (gxx.get(&[p, q]) - gxx.get(&[q, p])).abs() <= 1e-11 * sxx,
                    "γ(x,x) not symmetric at [{p},{q}]"
                );
            }
        }
        let tr: f64 = (0..d).map(|p| gxx.get(&[p, p])).sum();
        assert!(
            (tr - n as f64 * su2_dot(&x, &x)).abs() <= 1e-11 * sxx,
            "tr γ(x,x) = {tr}, expected {}",
            n as f64 * su2_dot(&x, &x)
        );
        let gxw = su2_transition_1rdm(&x, &w);
        let gwx = su2_transition_1rdm(&w, &x);
        let sc = scale_of(&x, &w);
        for p in 0..d {
            for q in 0..d {
                assert!(
                    (gxw.get(&[p, q]) - gwx.get(&[q, p])).abs() <= 1e-11 * sc,
                    "γ(x,y) ≠ γ(y,x)ᵀ at [{p},{q}]"
                );
            }
        }
    }

    #[test]
    fn tst4_diagonal_matches_the_number_operator() {
        for (fi, &(d, n, two_s)) in ST_FIX.iter().enumerate() {
            let space = Su2Space::new(d, n, two_s);
            let x = st_state(space, 1, 2, 0x5D00 + fi as u64);
            let y = st_state(space, 1, 3, 0x5D80 + fi as u64);
            let sc = scale_of(&x, &y);
            let g = su2_transition_1rdm(&x, &y);
            let maxdiag = (0..d).fold(0.0f64, |m, p| m.max(g.get(&[p, p]).abs()));
            assert!(
                maxdiag > 1e-6,
                "vacuous diagonal (d={d}, N={n}, 2S={two_s}): max|γ[p,p]| = {maxdiag}"
            );
            for p in 0..d {
                let want = su2_dot(&x, &n_i(&y, p));
                assert!(
                    (g.get(&[p, p]) - want).abs() <= 1e-11 * sc,
                    "γ[{p},{p}] = {} vs dot(x, n_p y) = {want} (d={d}, N={n}, 2S={two_s})",
                    g.get(&[p, p])
                );
            }
            let tr: f64 = (0..d).map(|p| g.get(&[p, p])).sum();
            assert!(
                (tr - n as f64 * su2_dot(&x, &y)).abs() <= 1e-11 * sc,
                "tr γ = {tr}, expected N·dot(x,y) = {}",
                n as f64 * su2_dot(&x, &y)
            );
        }
    }

    #[test]
    fn tst5_operator_role_ablation() {
        let (d, n, two_s) = (6usize, 3usize, 1usize);
        let space = Su2Space::new(d, n, two_s);
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        t.set(&[1, 4], 1.0);
        let v = CoeffTensor::<f64>::zeros(d, 4);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let x = st_state(space, 1, 3, 0x5E01);

        let hx = h.apply(&x);
        let g = su2_transition_1rdm(&x, &hx);
        let (mut maxg, mut maxasym) = (0.0f64, 0.0f64);
        for p in 0..d {
            for q in 0..d {
                maxg = maxg.max(g.get(&[p, q]).abs());
                maxasym = maxasym.max((g.get(&[p, q]) - g.get(&[q, p])).abs());
            }
        }
        assert!(maxg > 1e-6, "vacuous ablation fixture");
        assert!(
            maxasym > 1e-6 * maxg,
            "the ablation fixture is not asymmetric (maxasym {maxasym}, maxg {maxg}) — it cannot \
             detect a swapped creation/annihilation role"
        );

        let gb = su2_transition_1rdm_bruteforce(&x, &hx);
        let sc = (x.norm() * hx.norm()).max(1.0);
        let (_, diff) = maxg_maxdiff(&g, &gb, d);
        assert!(
            diff <= 1e-12 * sc,
            "sweep vs bruteforce on the ablation fixture: maxdiff {diff}"
        );

        for (name, g) in [
            ("materialized", h_materialized(&x, &h, &x)),
            ("streamed", su2_transition_1rdm_h(&x, &h, &x)),
        ] {
            let (_, dh) = maxg_maxdiff(&g, &gb, d);
            assert!(
                dh <= 1e-12 * sc,
                "H-dressed sweep ({name}) vs bruteforce on the ablation fixture: maxdiff {dh} — a \
                 global transpose in the bra-dressed writes would land here"
            );
        }
    }

    #[test]
    fn tst7_sweeps_are_bit_identical_across_runs() {
        let (d, n, two_s) = (6usize, 3usize, 1usize);
        let space = Su2Space::new(d, n, two_s);
        let (t, v) = spin_free_tv(d);
        let h = Su2SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);
        let x = st_state(space, 1, 2, 0x6001);
        let y = st_state(space, 1, 3, 0x6002);

        let a = su2_transition_1rdm(&x, &y);
        let b = su2_transition_1rdm(&x, &y);
        let ah = su2_transition_1rdm_h(&x, &h, &y);
        let bh = su2_transition_1rdm_h(&x, &h, &y);
        for p in 0..d {
            for q in 0..d {
                assert_eq!(
                    a.get(&[p, q]).to_bits(),
                    b.get(&[p, q]).to_bits(),
                    "plain sweep not bit-identical at [{p},{q}]"
                );
                assert_eq!(
                    ah.get(&[p, q]).to_bits(),
                    bh.get(&[p, q]).to_bits(),
                    "H-dressed sweep not bit-identical at [{p},{q}]"
                );
            }
        }
        let spinful = (1..d).any(|b| {
            space.bond_sectors(b).iter().any(|&s| {
                s.two_s > 0 && x.rank(b, s) > 0 && (theta(s.two_s, s.two_s + 1) - 1.0).abs() > 1e-12
            })
        });
        assert!(spinful, "fixture carries no θ ≠ 1 interior transition");
    }
}
