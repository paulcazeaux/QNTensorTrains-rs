// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rustc_hash::FxHashMap;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use crate::coeff::{CoeffTensor, ThreeBody};
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, Sector};
use crate::scalar::Scalar;

use super::super::assembly::{Flux, StateShift};
use super::bitset::{AdmMask, adm_mask};
use super::cover_switch_bonds;
use super::matching::hopcroft_karp;
use super::packed_ops::{PackedOps, sort_terms, term_state, unpack_site};
use super::sweep::{CompMap, Sweep};

/// Rider statistics of one Comp channel within one `(bond, sector, flux)` cover block: how many
/// terms ride it there and their distinct per-spin applied-creation counts `nl` — the
/// non-vacuity witness for the engineered heterogeneous-`nl` Comp-merge certification.
#[derive(Clone, Debug)]
pub struct CompChannelStat {
    pub num_terms: usize,
    pub distinct_nl_up: Vec<i8>,
    pub distinct_nl_dn: Vec<i8>,
}

/// One `(bond, sector, flux)` cover block of the term-level bipartite graph: rows (`U`) are the
/// distinct applied-left strings admissible at the sector, columns (`V`) the distinct
/// right-remainder strings, `coeff_rowmajor[p·ncols + q]` the folded term coefficient (0 when
/// absent). `matching_size` is the Hopcroft–Karp maximum matching of that graph — equal, by
/// König's theorem plus genericity, to the block's operator Schmidt rank (`dense_rank` oracle
/// in the `operators` binary). `comp_channels` reports the sweep's Comp channels instantiated at
/// the block (see [`CompChannelStat`]).
#[derive(Clone, Debug)]
pub struct BlockStat<T: Scalar> {
    pub bond: usize,
    pub sector: Sector,
    pub flux_up: i8,
    pub flux_dn: i8,
    /// The channel block's accumulated irrep flux Δ (always `IDENTITY` at `|G| = 1`).
    pub flux_irrep: Irrep,
    pub matching_size: usize,
    pub coeff_nrows: usize,
    pub coeff_ncols: usize,
    pub coeff_rowmajor: Vec<T>,
    pub comp_channels: Vec<CompChannelStat>,
}

/// One term's two weighted-edge answers, for the median-vs-bipartite reconciliation diagnostic
/// ([`switch_profile`]; `docs/median-bipartite-reconciliation.md`).
#[derive(Clone, Copy, Debug)]
pub struct TermSwitch {
    /// Body order (1, 2 or 3) — the number of creation operators.
    pub nbody: usize,
    /// The median construction's weighted edge: `median(hd, s_{n-1}, s_n)` over the term's
    /// sorted site multiset (`assembly::weighted_edge`).
    pub median_bond: usize,
    /// The bipartite cover's weighted edge: the bond at which the sweep deposits γ and the term
    /// switches Normal → Comp. `None` if the sweep dropped the ride (the string cannot continue
    /// in this symmetry space, so the term contributes no edge on either construction).
    pub cover_bond: Option<usize>,
    /// The same, from the cover run on the **median's** channel identity (channels and
    /// complementary operators refined by the creation count on the untracked side) — the switch
    /// that minimizes the *median* automaton's own channel count, which is what
    /// `Construction::MedianCoverSwitch` consumes.
    pub median_cover_bond: Option<usize>,
    /// The balance window `[s_{n-1}, s_n]`: the bonds at which the applied and remaining sides
    /// have equal size. The median switch always lies inside it; the cover's need not.
    pub window: (usize, usize),
}

/// Per-term comparison of the two constructions' weighted edges — the empirical half of the
/// reconciliation (`docs/median-bipartite-reconciliation.md` §2). The median rule places the
/// switch positionally (clamped median of the site multiset against the half-chain bond `hd`);
/// the cover places it by per-bond minimum vertex cover. This surfaces both for every screened
/// term so their agreement, and the cover's excursions outside the balance window, can be
/// measured on real integrals rather than argued.
pub fn switch_profile<T: Scalar>(
    space: QnSpace,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn ThreeBody<T>>,
    eps: f64,
) -> Vec<TermSwitch> {
    let mut terms = crate::hamiltonian::assembly::collect_terms(space, t, v, w, eps);
    let switches = cover_switch_bonds(space, &mut terms, false);
    let med_switches = cover_switch_bonds(space, &mut terms, true);
    let hd = (space.d as i64 - 1) / 2;
    terms
        .iter()
        .map(|term| {
            let nbody = term.ops.cre.iter().filter(|&&b| b != 0).count();
            let mut sites: Vec<usize> = term
                .ops
                .cre
                .iter()
                .chain(term.ops.ann.iter())
                .filter(|&&b| b != 0)
                .map(|&b| unpack_site(b))
                .collect();
            sites.sort_unstable();
            let mut arr: Vec<i64> = sites.iter().map(|&s| s as i64).collect();
            arr.push(hd);
            arr.sort_unstable();
            TermSwitch {
                nbody,
                median_bond: arr[nbody] as usize,
                cover_bond: switches.get(&term.ops).copied(),
                median_cover_bond: med_switches.get(&term.ops).copied(),
                window: (sites[nbody - 1], sites[nbody]),
            }
        })
        .collect()
}

/// Metric hook (design §2 M1): run the bipartite sweep for
/// its Comp-channel rider statistics, and independently rebuild every interior-bond
/// `(sector, flux)` block of the *term-level* bipartite graph straight from the sorted term
/// list — rows/columns/coefficient matrix as documented on [`BlockStat`].
pub fn bipartite_probe<T: Scalar>(
    space: QnSpace,
    t: &CoeffTensor<T>,
    v: &CoeffTensor<T>,
    w: Option<&dyn ThreeBody<T>>,
    eps: f64,
) -> Vec<BlockStat<T>> {
    let d = space.d;
    let mut terms = crate::hamiltonian::assembly::collect_terms(space, t, v, w, eps);
    sort_terms(&mut terms);

    let mut comp: CompMap = CompMap::new();
    if !terms.is_empty() {
        let mut sweep = Sweep::new(space, &terms, false);
        for k in 0..d {
            if k >= 1 {
                sweep.record_comp_stats(k, &mut comp);
            }
            sweep.site(k);
        }
    }

    let mut adm_cache: FxHashMap<(usize, StateShift), Rc<AdmMask>> = FxHashMap::default();
    let caps = (!space.pg.is_trivial()).then(|| crate::qn::max_ranks(&space));
    type TermBlock<T> = Vec<(PackedOps, PackedOps, T)>;
    type TermBlockMap<T> = BTreeMap<(usize, (usize, usize, u8), Flux), TermBlock<T>>;
    let mut blocks: TermBlockMap<T> = TermBlockMap::new();
    for b in 0..=d {
        let secs = space.bond_sectors(b);
        for term in &terms {
            let st = term_state(&space.pg, &term.ops, b);
            let flux = Flux {
                up: st.up.flux,
                dn: st.dn.flux,
                irrep: st.delta,
            };
            let left = term.ops.split_applied(b);
            let right = term.ops.split_remainder(b);
            let adm = adm_mask(&mut adm_cache, &space, caps.as_deref(), b, st);
            for (i, &s) in secs.iter().enumerate() {
                if !adm.ord[i] {
                    continue;
                }
                blocks
                    .entry((b, (s.nup, s.ndn, s.irrep.index()), flux))
                    .or_default()
                    .push((left, right, term.gamma));
            }
        }
    }

    let mut out = Vec::with_capacity(blocks.len());
    for ((bond, (nup, ndn, ir), flux), entries) in blocks {
        let rows: Vec<PackedOps> = entries
            .iter()
            .map(|&(l, _, _)| l)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let cols: Vec<PackedOps> = entries
            .iter()
            .map(|&(_, r, _)| r)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let (nrows, ncols) = (rows.len(), cols.len());
        let mut coeff = vec![T::zero(); nrows * ncols];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); nrows];
        for &(l, r, g) in &entries {
            let p = rows.binary_search(&l).expect("row present");
            let q = cols.binary_search(&r).expect("col present");
            coeff[p * ncols + q] = g;
            adj[p].push(q);
        }
        for a in &mut adj {
            a.sort_unstable();
            a.dedup();
        }
        let (matching, _, _) = hopcroft_karp(nrows, ncols, &adj);
        let comp_channels = comp
            .remove(&(bond, (nup, ndn, ir), flux))
            .map(|m| {
                m.into_values()
                    .map(|agg| CompChannelStat {
                        num_terms: agg.terms,
                        distinct_nl_up: agg.nl_up.into_iter().collect(),
                        distinct_nl_dn: agg.nl_dn.into_iter().collect(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.push(BlockStat {
            bond,
            sector: Sector::with_irrep(nup, ndn, Irrep::new(ir)),
            flux_up: flux.up,
            flux_dn: flux.dn,
            flux_irrep: flux.irrep,
            matching_size: matching,
            coeff_nrows: nrows,
            coeff_ncols: ncols,
            coeff_rowmajor: coeff,
            comp_channels,
        });
    }
    out
}
