// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;

use crate::pointgroup::{Irrep, PointGroup};
use crate::scalar::Scalar;
use crate::su2::core::Family;
use crate::su2::hamiltonian::{StateT, Su2CooNest, Ts};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2SectorGrid, Su2Space};
use crate::su2::spinalgebra::{SiteOp, SpinAlgebra};

use super::{Recorder, Src, median_weight};

/// Result bundle returned by [`super::assemble`] (mirrors the abelian `assembly::Assembled`;
/// the pipeline steps 1-5 folded into one call).
pub struct Assembled<T: Scalar> {
    pub states: Vec<Su2SectorGrid<Vec<Ts>>>,
    pub state_qns: Vec<Su2SectorGrid<Vec<Su2Sector>>>,
    pub coo: Vec<Su2CooNest<T>>,
}

/// Interned vertex id.
pub type VertexId = usize;

/// One interned vertex label: the bond, y-sector, reduced [`StateT`], and role-ordered open-index
/// pair it was interned from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VertexLabel {
    pub bond: usize,
    pub sector: Su2Sector,
    pub state: StateT,
    pub open: [u16; 2],
}

/// Packed edge key `(id_minus << 32) | id_plus` (the abelian automaton's packing).
pub type EdgeKey = u64;

type ChanKey = (Vec<i32>, Vec<(i32, i32, bool, [[u64; 3]; 3])>);

/// Per-construction interner + edge/channel-id bookkeeping. No global mutable state — owned and
/// threaded through [`super::assemble`].
pub struct Automaton<T: Scalar> {
    /// [`Self::getid`]'s vertex-label -> id map. Keyed on the [`VertexLabel`] struct itself (see
    /// the parent module's shape notes — no `u128` packing).
    pub vid: FxHashMap<VertexLabel, VertexId>,
    /// Interned vertex labels, index == [`VertexId`].
    pub vlabels: Vec<VertexLabel>,
    /// `chanid` content key -> small interned ordinal (`≥ 1`), assigned in term-enumeration order.
    pub chan_intern: FxHashMap<ChanKey, u64>,
    /// Median-carrying edge accumulator: the INLINE HEAD contribution per edge key. 99.7% of
    /// edges carry exactly
    /// one contribution.
    pub edges: FxHashMap<EdgeKey, (SiteOp, T)>,
    /// Overflow contributions beyond the inline head, in ARRIVAL order — load-bearing for the
    /// freeze summation order.
    pub edges_ovf: FxHashMap<EdgeKey, Vec<(SiteOp, T)>>,
}

impl<T: Scalar> Automaton<T> {
    /// Fresh, empty interner.
    pub fn new() -> Self {
        Automaton {
            vid: FxHashMap::default(),
            vlabels: Vec::new(),
            chan_intern: FxHashMap::default(),
            edges: FxHashMap::default(),
            edges_ovf: FxHashMap::default(),
        }
    }

    /// Intern vertex `v`, returning `(id, is_new)`.
    pub fn getid(&mut self, v: VertexLabel) -> (VertexId, bool) {
        if let Some(&id) = self.vid.get(&v) {
            return (id, false);
        }
        let id = self.vlabels.len();
        self.vlabels.push(v);
        self.vid.insert(v, id);
        (id, true)
    }

    /// Intern a 3-body channel's content `(ranks, [(twok,flux,odd,red)])` to a small ordinal,
    /// assigned in first-encounter order. Returns `0` for terms under 6 tokens (the ≥6-token gate)
    /// and a stable `≥ 1` ordinal otherwise.
    pub fn chanid(&mut self, n_tokens: usize, ranks: &[i32], siteops: &[SiteOp]) -> u64 {
        if n_tokens < 6 {
            return 0;
        }
        let key: ChanKey = (
            ranks.to_vec(),
            siteops
                .iter()
                .map(|s| (s.two_k, s.flux, s.odd, red_bits(&s.red)))
                .collect(),
        );
        let next = self.chan_intern.len() as u64 + 1;
        *self.chan_intern.entry(key).or_insert(next)
    }

    /// Accumulate `(op, w)` onto edge `(id_minus, id_plus)`. Median contributions (`is_median=true`)
    /// accumulate UNCONDITIONALLY; transport contributions dedup by content equality against
    /// the edge's existing contribution list ("inline head first, then overflow in arrival
    /// order" — load-bearing for the freeze summation order).
    pub fn accumulate_edge(
        &mut self,
        id_minus: VertexId,
        id_plus: VertexId,
        op: SiteOp,
        w: T,
        is_median: bool,
    ) {
        self.push_contribution(id_minus, id_plus, op, w, is_median);
    }

    #[inline]
    pub(crate) fn push_contribution(
        &mut self,
        id_minus: VertexId,
        id_plus: VertexId,
        op: SiteOp,
        w: T,
        is_median: bool,
    ) -> Option<EdgeKey> {
        let key = ((id_minus as u64) << 32) | id_plus as u64;
        if let std::collections::hash_map::Entry::Vacant(e) = self.edges.entry(key) {
            e.insert((op, w));
            return Some(key);
        }
        if is_median {
            self.edges_ovf.entry(key).or_default().push((op, w));
            return Some(key);
        }
        if self.edges[&key].0 == op {
            return None;
        }
        let ovf = self.edges_ovf.entry(key).or_default();
        if !ovf.iter().any(|(o, _)| *o == op) {
            ovf.push((op, w));
            return Some(key);
        }
        None
    }
}

impl<T: Scalar> Default for Automaton<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn id_siteop() -> SiteOp {
    SiteOp {
        two_k: 0,
        flux: 0,
        odd: false,
        red: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
}

#[inline]
pub(super) fn op_irrep_flux(op: &SiteOp, g: Irrep) -> Irrep {
    debug_assert_eq!(
        op.odd,
        op.flux.rem_euclid(2) == 1,
        "SiteOp parity/flux disagree (two_k={}, flux={}, odd={})",
        op.two_k,
        op.flux,
        op.odd
    );
    if op.odd { g } else { Irrep::IDENTITY }
}

fn red_bits(red: &[[f64; 3]; 3]) -> [[u64; 3]; 3] {
    let mut o = [[0u64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            o[i][j] = red[i][j].to_bits();
        }
    }
    o
}

/// The occupation class of a single-site block step. `(Δn, Δs)` are in the SAME units as
/// [`Su2Sector`] fields. Returns `(Family, reduced-occ-idx, site-2s)`, or `None` for a
/// non-admissible step:
pub fn blockclass(dn: i64, ds: i64) -> Option<(Family, usize, i64)> {
    match (dn, ds) {
        (0, 0) => Some((Family::Empty, 0, 0)),
        (1, 1) => Some((Family::Up, 1, 1)),
        (1, -1) => Some((Family::Dn, 1, 1)),
        (2, 0) => Some((Family::Occ, 2, 0)),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum WireGate {
    Window,
    Triangle,
}

/// The admissible x-spin values (doubled `2j`) for a y-sector doubled spin `two_s_y` and
/// operator rank `twok` — the triangle window `two_s_y - twok ..= two_s_y + twok` stepping by
/// 2. **KEEP the
/// comment verbatim**: the lower clamp MUST preserve the window's parity (steps of 2) — a plain
/// `max(0, lo)` clamp silently shifts `lo` by 1, flipping the parity and enumerating the WRONG
/// `jl` lattice. Bump `lo += 2` until `lo >= 0`.
pub fn jl_window(two_s_y: i64, twok: i64) -> Vec<i64> {
    jl_window_gated(two_s_y, twok, WireGate::Window)
}

pub(crate) fn jl_window_gated(two_s_y: i64, twok: i64, gate: WireGate) -> Vec<i64> {
    let mut lo = two_s_y - twok;
    while lo < 0 {
        lo += 2;
    }
    let mut out = Vec::new();
    let mut j = lo;
    while j <= two_s_y + twok {
        if gate == WireGate::Window || twok <= j + two_s_y {
            out.push(j);
        }
        j += 2;
    }
    out
}

/// One bond's reduced vertex label for one channel (`chain_state`). `sites =
/// sort(unique(tokens.map(site)))` is hoisted by the caller (`add_term`). Computes `flux`/`mask`,
/// `twok = ranks[nleft]` (0-based), and
/// the role-ordered (NEVER sorted) open-index pair (sorting would collide role-swapped terms
/// like `(l,j)` vs `(j,l)` on a shared median edge). The bond convention is 0-based throughout: `s
/// < κ` means site `s` lies strictly left of bond `κ`.
pub fn chain_state(
    pg: &PointGroup,
    tokens: &[(usize, bool)],
    ranks: &[i32],
    chanid: u64,
    kappa: usize,
    sites: &[usize],
    hd: usize,
) -> Ts {
    let mut flux: i64 = 0;
    let mut mask: u8 = 0;
    let mut napp: usize = 0;
    let mut delta = Irrep::IDENTITY;
    for (p, &(s, cdag)) in tokens.iter().enumerate() {
        if s < kappa {
            flux += if cdag { 1 } else { -1 };
            delta = delta.fuse(pg.irrep_of(s));
            mask |= 1u8 << p;
            napp += 1;
        }
    }
    let nleft = sites.iter().filter(|&&s| s < kappa).count();
    let twok = ranks[nleft] as u8;

    let nrem = tokens.len() - napp;
    let useapplied = napp < nrem || (napp == nrem && kappa <= hd);
    let mut i1: u16 = 0;
    let mut i2: u16 = 0;
    for &(s, _) in tokens {
        if (s < kappa) == useapplied {
            if i1 == 0 {
                i1 = (s as u16) + 1;
            } else if i2 == 0 {
                i2 = (s as u16) + 1;
                break;
            }
        }
    }

    (
        StateT {
            flux: flux as i8,
            mask,
            twok,
            delta,
            chanid,
        },
        [i1, i2],
    )
}

fn ysectors(
    space: &Su2Space,
    caps: Option<&[Su2RankGrid]>,
    st: &StateT,
    kappa: usize,
) -> Vec<Su2Sector> {
    let mut out = Vec::new();
    for s in space.bond_sectors(kappa) {
        if caps.is_some_and(|c| c[kappa][s] == 0) {
            continue;
        }
        let ml = s.n as i64 - st.flux as i64;
        if ml < 0 || ml > space.n as i64 {
            continue;
        }
        let x_ir = s.irrep.fuse(st.delta.inverse());
        let ok = jl_window(s.two_s as i64, st.twok as i64)
            .into_iter()
            .any(|two_j| {
                space.contains(
                    kappa,
                    Su2Sector::with_irrep(ml as usize, two_j as usize, x_ir),
                )
            });
        if ok {
            out.push(s);
        }
    }
    out
}

/// Set the four `(Δn,Δs)` y-sector transition edges out of one bond for one channel. `w = None` is
/// a transport contribution (weight 1, content-deduped against the edge's existing list); `w =
/// Some(_)` is the median contribution (accumulates unconditionally, carries `term_weight *
/// chan.weight`). `op = None` selects the transport [`id_siteop`]. Returns `continue_path` (did any
/// endpoint become a new vertex).
#[allow(clippy::too_many_arguments)]
pub fn set_edges<T: Scalar>(
    automaton: &mut Automaton<T>,
    space: &Su2Space,
    caps: Option<&[Su2RankGrid]>,
    kappa: usize,
    op: Option<&SiteOp>,
    w: Option<T>,
    st_minus: Ts,
    st_plus: Ts,
) -> bool {
    set_edges_rec(
        automaton,
        &mut (),
        space,
        caps,
        kappa,
        op,
        w,
        Src::Transport,
        st_minus,
        st_plus,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn set_edges_rec<T: Scalar, R: Recorder<T>>(
    automaton: &mut Automaton<T>,
    rec: &mut R,
    space: &Su2Space,
    caps: Option<&[Su2RankGrid]>,
    kappa: usize,
    op: Option<&SiteOp>,
    w: Option<T>,
    src: Src,
    st_minus: Ts,
    st_plus: Ts,
) -> bool {
    debug_assert!(
        !R::ON || w.is_some() == matches!(src, Src::Median { .. }),
        "a median contribution carries a weight and a term, a transport neither"
    );
    let (stm, idxm) = st_minus;
    let (stp, idxp) = st_plus;
    let op_flux: i64 = op.map(|o| o.flux as i64).unwrap_or(0);
    let g = space.pg.irrep_of(kappa);
    let op_ir = op.map_or(Irrep::IDENTITY, |o| op_irrep_flux(o, g));
    debug_assert_eq!(
        stp.delta.fuse(stm.delta.inverse()),
        op_ir,
        "chain_state Δ increment disagrees with the site-{kappa} operator irrep flux (a Δ that \
         skips annihilations produces exactly this)"
    );
    let y_plus = ysectors(space, caps, &stp, kappa + 1);

    let mut cont = false;
    for sm in ysectors(space, caps, &stm, kappa) {
        let nm = sm.n as i64;
        for y_occ in Family::ALL {
            let Some(plus_sec) = y_occ.apply(sm, g) else {
                continue;
            };
            let np = plus_sec.n as i64;
            if !y_plus.contains(&plus_sec) {
                continue;
            }
            let dm = (np - nm) - op_flux;
            if !(0..=2).contains(&dm) {
                continue;
            }
            debug_assert_eq!(
                plus_sec
                    .irrep
                    .fuse(stp.delta.inverse())
                    .fuse(sm.irrep.fuse(stm.delta.inverse()).inverse()),
                y_occ.flux(g).fuse(op_ir),
                "operator/state irrep bookkeeping disagree at site {kappa}"
            );
            let vm = VertexLabel {
                bond: kappa,
                sector: sm,
                state: stm,
                open: idxm,
            };
            let vp = VertexLabel {
                bond: kappa + 1,
                sector: plus_sec,
                state: stp,
                open: idxp,
            };
            let (id_m, new_m) = automaton.getid(vm);
            let (id_p, new_p) = automaton.getid(vp);
            cont = new_m | new_p || cont;
            let op_use = match op {
                Some(o) => o.clone(),
                None => id_siteop(),
            };
            if let Some(key) = automaton.push_contribution(
                id_m,
                id_p,
                op_use,
                w.unwrap_or_else(T::one),
                w.is_some(),
            ) {
                rec.appended(key, src);
            }
        }
    }
    cont
}

/// Enumerate one term's channels and sweep the median + transport edges. Median bond `κ_m` = the
/// positional median of `[hd, sitelist...]` WITH multiplicity (the abelian `weighted_edge`
/// precedent, re-derived for the SU(2) token-count formula — pinned by the cross-symmetry dense
/// oracle). Endpoint chain states are recomputed per bond (carrying them forward is a perf
/// refinement, not a correctness requirement).
pub fn add_term<T: Scalar>(
    automaton: &mut Automaton<T>,
    spin: &mut SpinAlgebra,
    space: &Su2Space,
    caps: Option<&[Su2RankGrid]>,
    tokens: &[(usize, bool)],
    w: T,
    hd: usize,
) {
    add_term_rec(automaton, &mut (), spin, space, caps, tokens, w, hd, 0);
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn add_term_rec<T: Scalar, R: Recorder<T>>(
    automaton: &mut Automaton<T>,
    rec: &mut R,
    spin: &mut SpinAlgebra,
    space: &Su2Space,
    caps: Option<&[Su2RankGrid]>,
    tokens: &[(usize, bool)],
    w: T,
    hd: usize,
    term: u32,
) {
    let d = space.d;
    let mut sites: Vec<usize> = tokens.iter().map(|t| t.0).collect();
    sites.sort_unstable();
    sites.dedup();

    debug_assert_eq!(
        tokens
            .iter()
            .fold(Irrep::IDENTITY, |a, &(s, _)| a.fuse(space.pg.irrep_of(s))),
        Irrep::IDENTITY,
        "Hamiltonian term on sites {:?} is not totally symmetric",
        tokens.iter().map(|t| t.0).collect::<Vec<_>>()
    );

    let mut arr: Vec<usize> = Vec::with_capacity(1 + tokens.len());
    arr.push(hd);
    for &(s, _) in tokens {
        arr.push(s);
    }
    arr.sort_unstable();
    let km = arr[(tokens.len() + 2) / 2 - 1];

    let chans = spin.term_channels(tokens);
    for chan in chans.iter() {
        let resolved: Vec<SiteOp> = chan
            .siteops
            .iter()
            .map(|&id| spin.siteop(id).clone())
            .collect();
        let cid = automaton.chanid(tokens.len(), &chan.ranks, &resolved);

        let op_at = |k: usize| -> Option<&SiteOp> {
            sites.iter().position(|&s| s == k).map(|p| &resolved[p])
        };

        {
            let st_m = chain_state(&space.pg, tokens, &chan.ranks, cid, km, &sites, hd);
            let st_p = chain_state(&space.pg, tokens, &chan.ranks, cid, km + 1, &sites, hd);
            set_edges_rec(
                automaton,
                rec,
                space,
                caps,
                km,
                op_at(km),
                Some(median_weight(w, chan.weight)),
                Src::Median {
                    term,
                    cw: chan.weight,
                },
                st_m,
                st_p,
            );
        }
        for k in km + 1..d {
            let st_m = chain_state(&space.pg, tokens, &chan.ranks, cid, k, &sites, hd);
            let st_p = chain_state(&space.pg, tokens, &chan.ranks, cid, k + 1, &sites, hd);
            if !set_edges_rec(
                automaton,
                rec,
                space,
                caps,
                k,
                op_at(k),
                None,
                Src::Transport,
                st_m,
                st_p,
            ) {
                break;
            }
        }
        for k in (0..km).rev() {
            let st_m = chain_state(&space.pg, tokens, &chan.ranks, cid, k, &sites, hd);
            let st_p = chain_state(&space.pg, tokens, &chan.ranks, cid, k + 1, &sites, hd);
            if !set_edges_rec(
                automaton,
                rec,
                space,
                caps,
                k,
                op_at(k),
                None,
                Src::Transport,
                st_m,
                st_p,
            ) {
                break;
            }
        }
    }
}
