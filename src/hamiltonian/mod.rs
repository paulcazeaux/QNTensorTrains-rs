// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The sparse Hamiltonian: term-enumeration constructor, `H·x` matvec, and the Rayleigh-quotient /
//! `xᵀHy` boundary contractions. The term-enumeration automaton lives in [`assembly`]; the full
//! `H·x` [`SparseHamiltonian::apply`] walks each cell's triples, while the sketched-apply
//! kernels in [`matvec`] select among per-triple/CSR/CSC traversals of the same blocks.
//! [`rayleigh_quotient`]/[`xt_h_y`] use the fused `IdSketch` env sweep.

pub mod assembly;
mod bipartite;
pub mod channel;
pub mod construct_probe;
pub mod csx;
#[doc(hidden)]
pub mod instrument;
pub mod matvec;

pub use assembly::{Construction, PatternMask, TermScreen};
pub use channel::{ChannelPhase, OpenOps, OrbitViolation, switch_orbits_homogeneous};
pub(crate) use matvec::EnvGaugePlan;

use std::ops::{Mul, Range};

use faer::prelude::ReborrowMut;
use faer::{MatMut, MatRef, unzip, zip};
use rayon::prelude::*;

use crate::coeff::{CoeffTensor, ThreeBody};
use crate::core::factor::{Sketch, adjoint_core_times_core};
use crate::core::{Family, SparseCore};
use crate::qn::{QnSpace, RankGrid, Sector, SectorGrid};
use crate::scalar::Scalar;
use crate::vector::TtVector;

use assembly::{Assembled, ChannelDesc, CooNest, SRC_NEG, SRC_NONE, jw_apply};
use csx::{CellSrc, CsxBlock};

pub(crate) struct HCell<T: Scalar> {
    pub x_occ: u8,
    pub m_ord: u32,
    pub csr: CsxBlock<T>,
    pub csc: CsxBlock<T>,
    pub src: Option<CellSrc>,
}

impl<T: Scalar> HCell<T> {
    #[inline]
    pub(crate) fn triples(&self) -> impl Iterator<Item = (i32, i32, T)> + '_ {
        self.csr.entries()
    }

    fn refill(&mut self, gamma: &[T], changed: &[u64], chunk: usize) -> usize {
        let HCell { csr, csc, src, .. } = self;
        let s = src
            .as_ref()
            .expect("HNest::refill on a cell built without provenance (not a refillable build)");
        if csr.vals.len() <= chunk {
            fill_changed(&mut csr.vals, &s.csr_src, gamma, changed)
                + fill_changed(&mut csc.vals, &s.csc_src, gamma, changed)
        } else {
            let split = |vals: &mut [T], words: &[u32]| -> usize {
                vals.par_chunks_mut(chunk)
                    .zip(words.par_chunks(chunk))
                    .map(|(v, w)| fill_changed(v, w, gamma, changed))
                    .sum()
            };
            let (a, b) = rayon::join(
                || split(&mut csr.vals, &s.csr_src),
                || split(&mut csc.vals, &s.csc_src),
            );
            a + b
        }
    }
}

#[inline]
fn fill_changed<T: Scalar>(vals: &mut [T], words: &[u32], gamma: &[T], changed: &[u64]) -> usize {
    let mut n = 0;
    for (v, &word) in vals.iter_mut().zip(words) {
        if word == SRC_NONE {
            continue;
        }
        let o = (word & !SRC_NEG) as usize;
        if (changed[o >> 6] >> (o & 63)) & 1 != 0 {
            *v = jw_apply(gamma[o], word & SRC_NEG != 0);
            n += 1;
        }
    }
    n
}

pub(crate) const REFILL_CHUNK: usize = 1 << 16;

pub(crate) struct HNest<T: Scalar> {
    offsets: Vec<u32>,
    diag: Vec<u32>,
    cells: Vec<HCell<T>>,
    sectors: Vec<Sector>,
    nup1: usize,
    ndn1: usize,
    nsec: usize,
}

impl<T: Scalar> HNest<T> {
    #[inline]
    fn oslot(&self, y_occ: Family, n1: Sector) -> usize {
        debug_assert!(
            n1.nup < self.nup1 && n1.ndn < self.ndn1,
            "sector {n1:?} out of grid"
        );
        let g = (n1.irrep.index() as usize) * self.nup1 * self.ndn1 + n1.nup * self.ndn1 + n1.ndn;
        (y_occ as usize) * self.nsec + g
    }

    #[inline]
    pub(crate) fn slot(&self, y_occ: Family, n1: Sector) -> &[HCell<T>] {
        let o = self.oslot(y_occ, n1);
        &self.cells[self.offsets[o] as usize..self.offsets[o + 1] as usize]
    }

    #[inline]
    pub(crate) fn sector_of(&self, c: &HCell<T>) -> Sector {
        self.sectors[c.m_ord as usize]
    }

    #[inline]
    fn diag_cell(&self, y_occ: Family, n1: Sector) -> Option<&HCell<T>> {
        let i = self.diag[self.oslot(y_occ, n1)];
        (i != u32::MAX).then(|| &self.cells[i as usize])
    }

    fn from_coo(coo: CooNest<T>, space: &QnSpace) -> Self {
        let (nup1, ndn1) = (space.nup + 1, space.ndn + 1);
        let nsec = space.pg.order() as usize * nup1 * ndn1;
        let sectors = coo.sectors().to_vec();
        let offsets = coo.offsets().to_vec();
        let cells: Vec<HCell<T>> = coo
            .into_cells()
            .map(|e| {
                let (csr, csc, src) = if e.src.is_empty() {
                    let (csr, csc) = csx::csx_pair(&e.cell);
                    (csr, csc, None)
                } else {
                    let (csr, csc, src) = csx::csx_pair_with_src(&e.cell, &e.src);
                    (csr, csc, Some(src))
                };
                HCell {
                    x_occ: e.x_occ,
                    m_ord: e.m_ord,
                    csr,
                    csc,
                    src,
                }
            })
            .collect();
        let mut diag = vec![u32::MAX; 4 * nsec];
        for y_occ in 0..4usize {
            for n1g in 0..nsec {
                let o = y_occ * nsec + n1g;
                let (lo, hi) = (offsets[o] as usize, offsets[o + 1] as usize);
                let plane = nup1 * ndn1;
                let (ir, p) = (n1g / plane, n1g % plane);
                let n1 =
                    Sector::with_irrep(p / ndn1, p % ndn1, crate::pointgroup::Irrep::new(ir as u8));
                for (i, c) in cells[lo..hi].iter().enumerate() {
                    if c.x_occ as usize == y_occ && sectors[c.m_ord as usize] == n1 {
                        diag[o] = (lo + i) as u32;
                        break;
                    }
                }
            }
        }
        HNest {
            offsets,
            diag,
            cells,
            sectors,
            nup1,
            ndn1,
            nsec,
        }
    }

    fn refill(&mut self, gamma: &[T], changed: &[u64], chunk: usize) -> usize {
        self.cells
            .par_iter_mut()
            .map(|c| c.refill(gamma, changed, chunk))
            .sum()
    }
}

#[derive(Clone, Debug)]
struct Skeleton {
    screen: TermScreen,
    construction: Construction,
    n_terms: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RefillStats {
    changed_terms: usize,
    slots_written: usize,
}

/// The assembled operator-action automaton for a screened one-/two-/three-body Hamiltonian.
pub struct SparseHamiltonian<T: Scalar> {
    /// Symmetry descriptor shared with the trains it acts on.
    pub space: QnSpace,
    states: Vec<SectorGrid<Vec<ChannelDesc>>>,
    state_qns: Vec<SectorGrid<Vec<Sector>>>,
    nests: Vec<HNest<T>>,
    empty_csx: CsxBlock<T>,
    skeleton: Option<Skeleton>,
    refill_gamma: Option<Vec<T>>,
}

impl<T: Scalar> SparseHamiltonian<T> {
    /// Screen and assemble the one-/two-/(optional) three-body term automaton. `w = None` omits the
    /// three-body channel entirely.
    pub fn new(
        space: QnSpace,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        eps: f64,
    ) -> Self {
        Self::new_with(space, t, v, w, eps, assembly::DEFAULT_CONSTRUCTION)
    }

    /// [`Self::new`] with an explicit [`Construction`] selection (`docs/bipartite-design.md` §0:
    /// both constructions are first-class; they build the same exact-integer operator with
    /// different channel layouts, tied together by the cross-oracle in the `operators` binary).
    pub fn new_with(
        space: QnSpace,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        eps: f64,
        construction: Construction,
    ) -> Self {
        Self::build(
            space,
            t,
            v,
            w,
            TermScreen::Magnitude(eps),
            construction,
            false,
        )
    }

    /// A **refillable** build: the automaton skeleton — term set, vertex/channel
    /// ordinals, the bipartite cover, every CSR/CSC index array — is a function of `space`,
    /// `screen` and `construction` ONLY, and every stored value remembers which term's `γ` it
    /// holds (or that it is a transport `±1`). [`Self::refill`] then rewrites the values from new
    /// integrals over the same space without rebuilding anything.
    ///
    /// # Panics
    ///
    /// On a three-body `w` with `SymmetryAllowed` (that channel's candidate screen is
    /// value-driven), and on a `SymmetryAllowed` point group that is not `space.pg`.
    pub fn new_refillable(
        space: QnSpace,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        screen: TermScreen,
        construction: Construction,
    ) -> Self {
        match &screen {
            TermScreen::SymmetryAllowed(pg) => {
                assert!(
                    *pg == space.pg,
                    "new_refillable: the SymmetryAllowed screen's point group must be the space's"
                );
                assert!(
                    w.is_none(),
                    "new_refillable: a three-body channel is not pattern-frozen under a pattern \
                     screen — build it with `new_with` instead"
                );
            }
            TermScreen::Mask(_) => assert!(
                w.is_none(),
                "new_refillable: a three-body channel is not pattern-frozen under a pattern \
                 screen — build it with `new_with` instead"
            ),
            TermScreen::Magnitude(_) => {}
        }
        Self::build(space, t, v, w, screen, construction, true)
    }

    /// A 64-bit hash of the WHOLE artifact: per-bond channel lists and coupled sectors, every
    /// nest's offsets/diagonal index/sector list, and every cell's `x_occ`/`m_ord`, CSR/CSC index
    /// arrays AND values (by IEEE bit pattern, real and imaginary parts). Two operators with equal
    /// hashes are bitwise the same operator — the determinism canary and the [`Self::refill`]
    /// oracle.
    pub fn artifact_hash(&self) -> u64 {
        self.hash_artifact(true)
    }

    /// [`Self::artifact_hash`] without the values: the SKELETON a refill must leave untouched.
    pub fn skeleton_hash(&self) -> u64 {
        self.hash_artifact(false)
    }

    fn hash_artifact(&self, with_vals: bool) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = rustc_hash::FxHasher::default();
        for b in 0..=self.space.d {
            for s in self.space.bond_sectors(b) {
                self.states[b][s].hash(&mut h);
                self.state_qns[b][s].hash(&mut h);
            }
        }
        for n in &self.nests {
            n.offsets.hash(&mut h);
            n.diag.hash(&mut h);
            n.sectors.hash(&mut h);
            for c in &n.cells {
                c.x_occ.hash(&mut h);
                c.m_ord.hash(&mut h);
                for blk in [&c.csr, &c.csc] {
                    blk.ids.hash(&mut h);
                    blk.starts.hash(&mut h);
                    blk.index.hash(&mut h);
                    if with_vals {
                        for &v in &blk.vals {
                            v.re_s().to_bits().hash(&mut h);
                            v.im_s().to_bits().hash(&mut h);
                        }
                    }
                }
            }
        }
        h.finish()
    }

    /// Was this operator built by [`Self::new_refillable`]?
    pub fn is_refillable(&self) -> bool {
        self.skeleton.is_some()
    }

    /// The screen a refillable build was minted with (`None` on a plain build).
    pub fn term_screen(&self) -> Option<TermScreen> {
        self.skeleton.as_ref().map(|s| s.screen.clone())
    }

    /// Classes of `(t, v)` that a fresh `new_with(.., eps, ..)` would carry and this skeleton
    /// cannot: `0` means a [`Self::refill`] with `(t, v)` yields exactly that operator up to
    /// classes whose `|W| ≤ eps` are stored explicitly; a positive count means the pattern has
    /// genuinely grown and the skeleton must be re-minted. Always `0` for a `SymmetryAllowed`
    /// skeleton.
    ///
    /// # Panics
    ///
    /// On a plain build, or on a `Magnitude` skeleton (whose only guard is `refill`'s term-count
    /// assert).
    pub fn pattern_leak(&self, t: &CoeffTensor<T>, v: &CoeffTensor<T>, eps: f64) -> usize {
        let sk = self
            .skeleton
            .as_ref()
            .expect("SparseHamiltonian::pattern_leak: not a refillable build");
        match &sk.screen {
            TermScreen::Mask(m) => m.leak(t, v, eps, self.space.pg),
            TermScreen::SymmetryAllowed(_) => 0,
            TermScreen::Magnitude(_) => panic!(
                "SparseHamiltonian::pattern_leak: a Magnitude skeleton has no frozen class set — \
                 mint with TermScreen::Mask(PatternMask::from_magnitude(..)) instead"
            ),
        }
    }

    /// Rewrite the coefficients from new integrals over the same space, keeping the skeleton —
    /// `states`, `state_qns`, and every nest's offsets, diagonal index and CSR/CSC index arrays —
    /// bitwise unchanged. Bit-identical to `new_refillable(space, t, v, w, screen, construction)`
    /// on the same arguments (same Phase-A code in the same order, exact sign flips, no sums).
    ///
    /// # Panics
    ///
    /// If the operator is not refillable, or if the Phase-A term count differs from the one the
    /// skeleton was minted with (the pattern is not frozen — a different space, or a `Magnitude`
    /// screen whose surviving set moved with the values).
    pub fn refill(&mut self, t: &CoeffTensor<T>, v: &CoeffTensor<T>, w: Option<&dyn ThreeBody<T>>) {
        self.refill_chunked(t, v, w, REFILL_CHUNK);
    }

    fn refill_chunked(
        &mut self,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        chunk: usize,
    ) -> RefillStats {
        let sk = self
            .skeleton
            .as_ref()
            .expect("SparseHamiltonian::refill: not a refillable build (use `new_refillable`)");
        Self::check_shapes(self.space, t, v, w);
        let gamma = assembly::gamma_stream(self.space, t, v, w, &sk.screen, sk.construction);
        assert_eq!(
            gamma.len(),
            sk.n_terms,
            "SparseHamiltonian::refill: the Phase-A term count changed ({} vs {} at build) — the \
             skeleton is not frozen for these integrals (a `TermScreen::Magnitude` screen moves with \
             the values; use `SymmetryAllowed`)",
            gamma.len(),
            sk.n_terms
        );
        let old = self
            .refill_gamma
            .as_mut()
            .expect("a refillable build carries its coefficient stream");
        let changed: Vec<u64> = gamma
            .par_chunks(64)
            .zip(old.par_chunks(64))
            .map(|(new, old)| {
                let mut w = 0u64;
                for (b, (x, y)) in new.iter().zip(old).enumerate() {
                    let same = x.re_s().to_bits() == y.re_s().to_bits()
                        && x.im_s().to_bits() == y.im_s().to_bits();
                    w |= u64::from(!same) << b;
                }
                w
            })
            .collect();
        let mut stats = RefillStats {
            changed_terms: changed.iter().map(|w| w.count_ones() as usize).sum(),
            ..RefillStats::default()
        };
        if stats.changed_terms > 0 {
            stats.slots_written = self
                .nests
                .par_iter_mut()
                .map(|n| n.refill(&gamma, &changed, chunk))
                .sum();
        }
        *old = gamma;
        stats
    }

    fn check_shapes(
        space: QnSpace,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
    ) {
        assert!(
            t.rank == 2 && t.d == space.d,
            "one-body tensor must be rank-2 over d={} sites (got rank {}, d {})",
            space.d,
            t.rank,
            t.d
        );
        assert!(
            v.rank == 4 && v.d == space.d,
            "two-body tensor must be rank-4 over d={} sites (got rank {}, d {})",
            space.d,
            v.rank,
            v.d
        );
        if let Some(w) = w {
            assert!(
                w.norb() == space.d,
                "three-body tensor orbital count must equal d={} sites (got {})",
                space.d,
                w.norb()
            );
        }
    }

    fn build(
        space: QnSpace,
        t: &CoeffTensor<T>,
        v: &CoeffTensor<T>,
        w: Option<&dyn ThreeBody<T>>,
        screen: TermScreen,
        construction: Construction,
        record_src: bool,
    ) -> Self {
        Self::check_shapes(space, t, v, w);
        let probe_t0 = std::time::Instant::now();
        let Assembled {
            states,
            state_qns,
            coo,
            n_terms,
        } = assembly::assemble_screened(space, t, v, w, screen.clone(), construction, record_src);
        let nests: Vec<HNest<T>> = {
            let _t = construct_probe::timer(construct_probe::Stage::NestCompress);
            coo.into_iter()
                .map(|c| HNest::from_coo(c, &space))
                .collect()
        };
        construct_probe::set_total(probe_t0.elapsed().as_nanos() as u64);
        let refill_gamma = record_src.then(|| {
            let gamma = assembly::gamma_stream(space, t, v, w, &screen, construction);
            assert_eq!(
                gamma.len(),
                n_terms,
                "new_refillable: the Phase-A stream disagrees with the build's term count"
            );
            gamma
        });
        SparseHamiltonian {
            space,
            states,
            state_qns,
            nests,
            empty_csx: CsxBlock::empty(),
            skeleton: record_src.then_some(Skeleton {
                screen,
                construction,
                n_terms,
            }),
            refill_gamma,
        }
    }

    pub(crate) fn self_cell(&self, k: usize, occ: Family, l: Sector) -> &CsxBlock<T> {
        match self.nests[k].diag_cell(occ, l) {
            Some(c) => &c.csr,
            None => &self.empty_csx,
        }
    }

    #[inline]
    pub(crate) fn nest_slot(&self, k: usize, y_occ: Family, n1: Sector) -> &[HCell<T>] {
        self.nests[k].slot(y_occ, n1)
    }

    /// Per-bond effective bond dimension of the operator-action automaton: the number of
    /// operator-states (automaton vertices) threading each cut, summed over QN sectors — bonds
    /// `0..=d`. This is the automaton's analogue of an explicit MPO's bond dimension, i.e. how
    /// compressed its Hamiltonian representation is; comparing it to a DMRG code's compressed-MPO
    /// bond dimension is the mechanistic side of the automaton-vs-MPO efficiency question.
    pub fn automaton_bond_dims(&self) -> Vec<usize> {
        (0..=self.space.d)
            .map(|b| {
                self.space
                    .bond_sectors(b)
                    .iter()
                    .map(|&s| self.states[b][s].len())
                    .sum()
            })
            .collect()
    }

    /// Per-bond count of *distinct* operator channels, ignoring how many sectors each is
    /// instantiated at — the counting convention external MPO codes (block2) report their bond
    /// dimension in. [`Self::automaton_bond_dims`] is the sector-summed instantiation count
    /// (each channel counted once per admissible sector): the automaton's own storage/rank
    /// convention, structurally larger whenever a channel threads several sectors.
    /// `bipartite_probe`'s Hopcroft-Karp == dense-rank check certifies only its own independently
    /// rebuilt term-level graph, never the real sweep's cover, so it is NOT a lower bound the
    /// sweep is held to.
    pub fn automaton_channel_counts(&self) -> Vec<usize> {
        (0..=self.space.d)
            .map(|b| {
                let mut distinct = std::collections::BTreeSet::new();
                for s in self.space.bond_sectors(b) {
                    for c in &self.states[b][s] {
                        distinct.insert(*c);
                    }
                }
                distinct.len()
            })
            .collect()
    }

    /// Metric surface M1 + the cover-subproblem histogram (`docs/bipartite-design.md` §2): the
    /// per-spin flux `(up, dn)` of every operator channel threading bond `b` at sector `s`, in the
    /// bucket-sorted channel order. Grouping the result by value yields the per-`(bond, sector,
    /// flux)` class sizes — exactly the cover-blocking subproblems of the bipartite sweep (§1
    /// step 2). The median construction reads the flux off `StateShift`; the bipartite one reads
    /// it off its `ChannelDesc` — same signature either way, so the permanent gate harness
    /// (`examples/tutorial/step_04_hamiltonian_construction.rs`) survives the construction swap unchanged.
    pub fn channel_fluxes(&self, b: usize, s: Sector) -> Vec<assembly::Flux> {
        self.states[b][s].iter().map(|c| c.flux()).collect()
    }

    pub(crate) fn channel_descs(&self, b: usize, s: Sector) -> &[ChannelDesc] {
        &self.states[b][s]
    }

    /// Metric surface M2 (`docs/bipartite-design.md` §2): operator nonzeros at site `k`, summed
    /// over every `[y_occ][n1][x_occ][m]` nest cell. Each triple is one `axpy_block` dispatch in
    /// [`Self::apply`]'s `build_core`, making Σ_k `coo_nnz(k)` the one-site matvec wall predictor
    /// — the §0 merge-gate metric (NOT bond dim).
    pub fn coo_nnz(&self, k: usize) -> usize {
        self.nests[k].cells.iter().map(|c| c.csr.nnz()).sum()
    }

    /// Populated nest cells at site `k` — the item-C storage census surface.
    pub fn nest_cells(&self, k: usize) -> usize {
        self.nests[k].cells.len()
    }

    /// Metric surface M3 driver (`docs/bipartite-design.md` §2): the coupled x-sector list
    /// `state_qns[b][s]` — the operator-band expansion multiset whose Σ-of-x-ranks IS the
    /// [`Self::row_ranks`]/[`Self::col_ranks`] entry at `(b, s)`.
    pub fn state_qns_at(&self, b: usize, s: Sector) -> &[Sector] {
        &self.state_qns[b][s]
    }

    /// `H·x`: a per-core per-triple axpy over the full bond-sector grid (empty cells do not exist
    /// in the nest), rather than a `±3`-banded window. Rayon
    /// parallelizes over
    /// cores (coarser than a per-output-block spawn); cores are built
    /// independently (disjoint writes), so the result is bit-identical to serial regardless of
    /// thread count.
    pub fn apply(&self, x: &TtVector<T>) -> TtVector<T> {
        assert_eq!(self.space, x.space, "apply: QnSpace mismatch");
        let space = self.space;
        let d = space.d;
        let cores: Vec<SparseCore<T>> = (0..d)
            .into_par_iter()
            .map(|k| self.build_core(x.core(k)))
            .collect();
        TtVector::new(space, cores)
    }

    pub(crate) fn build_core(&self, x: &SparseCore<T>) -> SparseCore<T> {
        let space = self.space;
        let k = x.k;
        let rr = self.row_ranks(x);
        let cr = self.col_ranks(x);
        let mut y = SparseCore::zeros(space, k, &rr, &cr);
        let i_tab = self.row_ranges(x);
        let j_tab = self.col_ranges(x);
        let nest = &self.nests[k];

        for occ_n in Family::ALL {
            let out_secs: Vec<Sector> = y.transitions(occ_n).collect();
            for n1 in out_secs {
                let n2 = occ_n.apply(n1, space.pg.irrep_of(k));
                let i_ranges = &i_tab[n1];
                let j_ranges = &j_tab[n2];
                let mut yb = match y.block_mut(occ_n, n1) {
                    Some(b) => b,
                    None => continue,
                };
                if yb.nrows() == 0 || yb.ncols() == 0 {
                    continue;
                }
                for rec in nest.slot(occ_n, n1) {
                    let occ_m = Family::from_physical_index(rec.x_occ as usize);
                    let m = nest.sector_of(rec);
                    let xb = match x.block(occ_m, m) {
                        Some(b) => b,
                        None => continue,
                    };
                    for (row, col, alpha) in rec.triples() {
                        let ir = &i_ranges[row as usize];
                        let jr = &j_ranges[col as usize];
                        axpy_block(
                            yb.rb_mut(),
                            ir.start,
                            jr.start,
                            ir.len(),
                            jr.len(),
                            alpha,
                            xb,
                        );
                    }
                }
            }
        }
        y
    }

    /// Operator-band-expanded row ranks of the output core replacing `x`'s core at site `k`.
    pub fn row_ranks(&self, x: &SparseCore<T>) -> RankGrid {
        let space = self.space;
        let k = x.k;
        if k == 0 {
            return x.row_ranks.clone();
        }
        let sq = &self.state_qns[k];
        let mut ranks = RankGrid::zeros(&space);
        for &s in &x.row_qn {
            let total: usize = sq[s].iter().map(|&m| x.row_ranks[m]).sum();
            ranks[s] = total;
        }
        ranks
    }

    /// Operator-band-expanded column ranks.
    pub fn col_ranks(&self, x: &SparseCore<T>) -> RankGrid {
        let space = self.space;
        let k = x.k;
        if k == space.d - 1 {
            return x.col_ranks.clone();
        }
        let sq = &self.state_qns[k + 1];
        let mut ranks = RankGrid::zeros(&space);
        for &s in &x.col_qn {
            let total: usize = sq[s].iter().map(|&m| x.col_ranks[m]).sum();
            ranks[s] = total;
        }
        ranks
    }

    /// Per-sector row index ranges into the dense-over-span output block.
    pub fn row_ranges(&self, x: &SparseCore<T>) -> SectorGrid<Vec<Range<usize>>> {
        let space = self.space;
        let k = x.k;
        let mut out = SectorGrid::filled(&space, Vec::new());
        if k == 0 {
            let s0 = Sector::new(0, 0);
            #[allow(clippy::single_range_in_vec_init)]
            {
                out[s0] = vec![0..x.row_ranks[s0]];
            }
        } else {
            let sq = &self.state_qns[k];
            for &s in &x.row_qn {
                out[s] = cumulative_ranges(sq[s].iter().map(|&m| x.row_ranks[m]));
            }
        }
        out
    }

    /// Per-sector column index ranges.
    pub fn col_ranges(&self, x: &SparseCore<T>) -> SectorGrid<Vec<Range<usize>>> {
        let space = self.space;
        let k = x.k;
        let mut out = SectorGrid::filled(&space, Vec::new());
        if k == space.d - 1 {
            let last = Sector::with_irrep(space.nup, space.ndn, space.target);
            #[allow(clippy::single_range_in_vec_init)]
            {
                out[last] = vec![0..x.col_ranks[last]];
            }
        } else {
            let sq = &self.state_qns[k + 1];
            for &s in &x.col_qn {
                out[s] = cumulative_ranges(sq[s].iter().map(|&m| x.col_ranks[m]));
            }
        }
        out
    }
}

fn axpy_block<T: Scalar>(
    mut dst: MatMut<'_, T>,
    i0: usize,
    j0: usize,
    nrows: usize,
    ncols: usize,
    alpha: T,
    src: MatRef<'_, T>,
) {
    let d_sub = dst.rb_mut().submatrix_mut(i0, j0, nrows, ncols);
    let s_sub = src.submatrix(0, 0, nrows, ncols);
    zip!(d_sub, s_sub).for_each(|unzip!(d, s)| *d += alpha * *s);
}

fn cumulative_ranges(widths: impl Iterator<Item = usize>) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for w in widths {
        ranges.push(start..start + w);
        start += w;
    }
    ranges
}

impl<T: Scalar> Mul<&TtVector<T>> for &SparseHamiltonian<T> {
    type Output = TtVector<T>;
    fn mul(self, x: &TtVector<T>) -> TtVector<T> {
        self.apply(x)
    }
}

/// `⟨x, Hx⟩` via the fused rank-1 `IdSketch` environment sweep. NOT divided by `⟨x,x⟩` despite the
/// name; callers pass unit-norm `x`.
pub fn rayleigh_quotient<T: Scalar>(h: &SparseHamiltonian<T>, x: &TtVector<T>) -> T {
    xt_h_y(x, h, x)
}

/// `⟨x, Hy⟩`, the bilinear twin of [`rayleigh_quotient`].
pub fn xt_h_y<T: Scalar>(x: &TtVector<T>, h: &SparseHamiltonian<T>, y: &TtVector<T>) -> T {
    assert!(
        x.boundary_width() == 1 && y.boundary_width() == 1,
        "xt_h_y is the scalar ⟨x, Hy⟩ of two unit-boundary trains (widths {} and {}); an \
         open block goes through xt_h_y_block",
        x.boundary_width(),
        y.boundary_width()
    );
    *fold_x_h_y(x, h, y).get(0, 0)
}

/// Operator Gram of two OPEN BLOCKS: `G[i, j] = ⟨X_i, H·Y_j⟩` (conjugation on `x`), of shape
/// `x.boundary_width() × y.boundary_width()` — the sketched-Gram kernel of block trace
/// estimation. It is [`xt_h_y`]'s own LEFT fold (unweighted on both layers; `H·Y` is never
/// materialized) with the whole bond-`d` block read out instead of one entry, so the cost does
/// not grow with the widths beyond the last core's. On a unit-boundary pair it is the `1×1`
/// matrix holding [`xt_h_y`], bit for bit.
pub fn xt_h_y_block<T: Scalar>(
    x: &TtVector<T>,
    h: &SparseHamiltonian<T>,
    y: &TtVector<T>,
) -> faer::Mat<T> {
    fold_x_h_y(x, h, y)
}

fn fold_x_h_y<T: Scalar>(
    x: &TtVector<T>,
    h: &SparseHamiltonian<T>,
    y: &TtVector<T>,
) -> faer::Mat<T> {
    let space = x.space;
    let d = space.d;
    let mut p = Sketch::identity(space, 0, 1);
    for k in 0..d {
        let l = matvec::sketch_h_mul(&p, h, y.core(k));
        p = adjoint_core_times_core(x.core(k), &l);
    }
    let last = Sector::with_irrep(space.nup, space.ndn, space.target);
    p.block(last)
        .expect("final bond has the (nup,ndn) sector")
        .to_owned()
}

#[cfg(test)]
mod refill_tests {
    use super::*;
    use crate::coeff::ThreeBody;
    use crate::scalar::C64;
    use crate::test_util::randomized_state;
    use rand::{RngExt, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    const ARMS: [Construction; 2] = [Construction::Median, Construction::Bipartite];

    fn dense_random_f64(
        d: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> (CoeffTensor<f64>, CoeffTensor<f64>) {
        let mut t = CoeffTensor::<f64>::zeros(d, 2);
        let mut v = CoeffTensor::<f64>::zeros(d, 4);
        for x in t.data.iter_mut().chain(v.data.iter_mut()) {
            *x = rng.random_range(-1.0..1.0);
        }
        (t, v)
    }

    fn dense_random_c64(
        d: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> (CoeffTensor<C64>, CoeffTensor<C64>) {
        let mut t = CoeffTensor::<C64>::zeros(d, 2);
        let mut v = CoeffTensor::<C64>::zeros(d, 4);
        for x in t.data.iter_mut().chain(v.data.iter_mut()) {
            *x = C64::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
        }
        (t, v)
    }

    fn space6() -> QnSpace {
        QnSpace::from_n_two_sz(6, 4, 0)
    }

    #[test]
    fn t8_refill_is_bit_identical_to_a_fresh_build() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_11);
        let (t0, v0) = dense_random_f64(space.d, &mut rng);
        let (t1, v1) = dense_random_f64(space.d, &mut rng);
        let x: TtVector<f64> = randomized_state(space, 1, 3, &mut rng);
        for c in ARMS {
            let screen = TermScreen::SymmetryAllowed(space.pg);
            let mut h = SparseHamiltonian::new_refillable(space, &t0, &v0, None, screen.clone(), c);
            assert!(h.is_refillable());
            let before = h.artifact_hash();
            h.refill(&t1, &v1, None);
            let fresh = SparseHamiltonian::new_refillable(space, &t1, &v1, None, screen.clone(), c);
            assert_ne!(h.artifact_hash(), before, "{c:?}: refill changed nothing");
            assert_eq!(
                h.artifact_hash(),
                fresh.artifact_hash(),
                "{c:?}: refill ≠ fresh build"
            );
            let plain = SparseHamiltonian::new_with(space, &t1, &v1, None, f64::EPSILON, c);
            assert_eq!(
                h.artifact_hash(),
                plain.artifact_hash(),
                "{c:?}: on dense values the symmetry-allowed skeleton must equal the magnitude one"
            );
            let (q, qf) = (rayleigh_quotient(&h, &x), rayleigh_quotient(&fresh, &x));
            assert!(
                (q - qf).abs() <= 1e-14 * q.abs().max(1.0),
                "{c:?}: RQ {q} vs {qf}"
            );
        }
    }

    #[test]
    fn t8_refill_c64_is_bit_identical_to_a_fresh_build() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_C6);
        let (t0, v0) = dense_random_c64(space.d, &mut rng);
        let (t1, v1) = dense_random_c64(space.d, &mut rng);
        for c in ARMS {
            let screen = TermScreen::SymmetryAllowed(space.pg);
            let mut h = SparseHamiltonian::new_refillable(space, &t0, &v0, None, screen.clone(), c);
            h.refill(&t1, &v1, None);
            let fresh = SparseHamiltonian::new_refillable(space, &t1, &v1, None, screen.clone(), c);
            assert_eq!(
                h.artifact_hash(),
                fresh.artifact_hash(),
                "{c:?}: c64 refill ≠ fresh build"
            );
            let plain = SparseHamiltonian::new_with(space, &t1, &v1, None, f64::EPSILON, c);
            assert_eq!(
                h.artifact_hash(),
                plain.artifact_hash(),
                "{c:?}: c64 refill ≠ plain build"
            );
        }
    }

    #[test]
    fn t8_refill_structure_frozen_across_successive_refills() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_55);
        let (t0, v0) = dense_random_f64(space.d, &mut rng);
        for c in ARMS {
            let screen = TermScreen::SymmetryAllowed(space.pg);
            let mut h = SparseHamiltonian::new_refillable(space, &t0, &v0, None, screen.clone(), c);
            let skel = h.skeleton_hash();
            assert_eq!(h.term_screen(), Some(screen.clone()));
            for step in 0..5 {
                let (t, v) = dense_random_f64(space.d, &mut rng);
                h.refill(&t, &v, None);
                assert_eq!(
                    h.skeleton_hash(),
                    skel,
                    "{c:?}: skeleton moved at refill {step}"
                );
                let fresh =
                    SparseHamiltonian::new_refillable(space, &t, &v, None, screen.clone(), c);
                assert_eq!(
                    h.artifact_hash(),
                    fresh.artifact_hash(),
                    "{c:?}: values at refill {step}"
                );
            }
            let minted =
                SparseHamiltonian::new_refillable(space, &t0, &v0, None, screen.clone(), c);
            h.refill(&t0, &v0, None);
            assert_eq!(
                h.artifact_hash(),
                minted.artifact_hash(),
                "{c:?}: round trip"
            );
        }
    }

    #[test]
    fn t8_refill_split_into_chunks_is_bit_identical_to_a_fresh_build() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_C4);
        let (t0, v0) = dense_random_f64(space.d, &mut rng);
        let (t1, v1) = dense_random_f64(space.d, &mut rng);
        let chunk = 3;
        for c in ARMS {
            let screen = TermScreen::SymmetryAllowed(space.pg);
            let mut h = SparseHamiltonian::new_refillable(space, &t0, &v0, None, screen.clone(), c);
            let split_cells = h
                .nests
                .iter()
                .flat_map(|n| &n.cells)
                .filter(|cell| cell.csr.vals.len() > chunk)
                .count();
            assert!(split_cells > 0, "{c:?}: no cell exercises the split path");
            h.refill_chunked(&t1, &v1, None, chunk);
            let fresh = SparseHamiltonian::new_refillable(space, &t1, &v1, None, screen, c);
            assert_eq!(
                h.artifact_hash(),
                fresh.artifact_hash(),
                "{c:?}: chunked refill ({split_cells} split cells)"
            );
        }
    }

    #[test]
    fn t8_refill_rewrites_only_changed_coefficients_bitwise() {
        let space = space6();
        let d = space.d;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_D7);
        let (t0, v0) = dense_random_f64(d, &mut rng);
        let ints = crate::integrals::Integrals {
            t: t0.clone(),
            v: v0.clone(),
            e0: 0.0,
            norb: d,
            nelec: 0,
            ms2: 0,
            orbsym: Vec::new(),
            isym: None,
        };
        for c in ARMS {
            let screen = TermScreen::SymmetryAllowed(space.pg);
            let mut h = SparseHamiltonian::new_refillable(space, &t0, &v0, None, screen.clone(), c);
            let slots: usize = h
                .nests
                .iter()
                .flat_map(|n| &n.cells)
                .map(|cell| cell.csr.vals.len() + cell.csc.vals.len())
                .sum();
            let mut u = crate::rotation::identity_matrix(d);
            let mut written = 0usize;
            for (step, &i) in [0usize, 1, 2, 3, 4, 3, 2, 1, 0, 2].iter().enumerate() {
                let g = crate::rotation::givens(d, i, 0.3 + 0.07 * step as f64);
                u = crate::rotation::compose_rotations(&u, &g, d);
                let rot = crate::rotation::rotate_integrals(&ints, &u);
                let st = h.refill_chunked(&rot.t, &rot.v, None, REFILL_CHUNK);
                let fresh = SparseHamiltonian::new_refillable(
                    space,
                    &rot.t,
                    &rot.v,
                    None,
                    screen.clone(),
                    c,
                );
                assert_eq!(
                    h.artifact_hash(),
                    fresh.artifact_hash(),
                    "{c:?}: step {step} (pair {i}) {st:?}"
                );
                assert!(st.changed_terms > 0, "{c:?}: step {step} moved nothing");
                assert!(
                    st.slots_written < slots,
                    "{c:?}: step {step}: a one-pair rotation rewrote all {slots} slots"
                );
                written += st.slots_written;
            }
            println!(
                "{c:?}: 10 one-pair refills wrote {written} of {} slots",
                10 * slots
            );
            let rot = crate::rotation::rotate_integrals(&ints, &u);
            let st = h.refill_chunked(&rot.t, &rot.v, None, REFILL_CHUNK);
            assert_eq!(
                st,
                RefillStats::default(),
                "{c:?}: the same integrals moved something"
            );
            let (t1, v1) = dense_random_f64(d, &mut rng);
            let st = h.refill_chunked(&t1, &v1, None, REFILL_CHUNK);
            let gamma_slots: usize = h
                .nests
                .iter()
                .flat_map(|n| &n.cells)
                .map(|cell| {
                    let src = cell.src.as_ref().expect("provenance");
                    src.csr_src
                        .iter()
                        .chain(&src.csc_src)
                        .filter(|&&w| w != SRC_NONE)
                        .count()
                })
                .sum();
            let n_terms = h.skeleton.as_ref().expect("refillable").n_terms;
            assert_eq!(
                (st.changed_terms, st.slots_written),
                (n_terms, gamma_slots),
                "{c:?}: dense values must move every term and rewrite every γ slot"
            );
            let fresh = SparseHamiltonian::new_refillable(space, &t1, &v1, None, screen, c);
            assert_eq!(
                h.artifact_hash(),
                fresh.artifact_hash(),
                "{c:?}: dense refill"
            );
        }
    }

    #[test]
    fn t8_mask_skeleton_is_the_magnitude_build_and_leak_sees_growth() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_AA);
        let (t0, v0) = dense_random_f64(space.d, &mut rng);
        let (t1, v1) = dense_random_f64(space.d, &mut rng);
        let eps = f64::EPSILON;
        for c in ARMS {
            let mask = PatternMask::from_magnitude(&t0, &v0, eps, space.pg);
            let screen = TermScreen::Mask(std::sync::Arc::new(mask));
            let mut h = SparseHamiltonian::new_refillable(space, &t0, &v0, None, screen.clone(), c);
            let plain0 = SparseHamiltonian::new_with(space, &t0, &v0, None, eps, c);
            assert_eq!(
                h.artifact_hash(),
                plain0.artifact_hash(),
                "{c:?}: mask mint ≠ plain"
            );
            let sym = SparseHamiltonian::new_refillable(
                space,
                &t0,
                &v0,
                None,
                TermScreen::SymmetryAllowed(space.pg),
                c,
            );
            assert_eq!(
                h.artifact_hash(),
                sym.artifact_hash(),
                "{c:?}: dense mask ≠ symmetry mask"
            );
            assert_eq!(
                h.pattern_leak(&t1, &v1, eps),
                0,
                "{c:?}: dense→dense must not leak"
            );
            h.refill(&t1, &v1, None);
            let plain1 = SparseHamiltonian::new_with(space, &t1, &v1, None, eps, c);
            assert_eq!(
                h.artifact_hash(),
                plain1.artifact_hash(),
                "{c:?}: mask refill ≠ plain"
            );
        }
        let mut ts = CoeffTensor::<f64>::zeros(space.d, 2);
        ts.set(&[0, 1], 0.5);
        ts.set(&[1, 0], 0.5);
        let vs = CoeffTensor::<f64>::zeros(space.d, 4);
        let mask_s = PatternMask::from_magnitude(&ts, &vs, eps, space.pg);
        assert_eq!(mask_s.n_classes(), 2);
        let hs = SparseHamiltonian::new_refillable(
            space,
            &ts,
            &vs,
            None,
            TermScreen::Mask(std::sync::Arc::new(mask_s)),
            Construction::Bipartite,
        );
        let leak = hs.pattern_leak(&t1, &v1, eps);
        assert!(
            leak > 1000,
            "sparse skeleton must report the dense classes it lacks: {leak}"
        );
        assert_eq!(hs.pattern_leak(&ts, &vs, eps), 0);
    }

    #[test]
    #[should_panic(expected = "term count changed")]
    fn t8_refill_pattern_guard_fires_on_a_magnitude_skeleton() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_66);
        let mut t0 = CoeffTensor::<f64>::zeros(space.d, 2);
        t0.set(&[0, 1], 0.5);
        t0.set(&[1, 0], 0.5);
        let v0 = CoeffTensor::<f64>::zeros(space.d, 4);
        let mut h = SparseHamiltonian::new_refillable(
            space,
            &t0,
            &v0,
            None,
            TermScreen::Magnitude(f64::EPSILON),
            Construction::Bipartite,
        );
        let (t1, v1) = dense_random_f64(space.d, &mut rng);
        h.refill(&t1, &v1, None);
    }

    #[test]
    #[should_panic(expected = "not a refillable build")]
    fn t8_refill_refuses_a_plain_build() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_77);
        let (t0, v0) = dense_random_f64(space.d, &mut rng);
        let mut h = SparseHamiltonian::new(space, &t0, &v0, None, f64::EPSILON);
        assert!(!h.is_refillable());
        h.refill(&t0, &v0, None);
    }

    #[test]
    #[should_panic(expected = "three-body")]
    fn t8_refillable_refuses_three_body_under_symmetry_allowed() {
        let space = space6();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8E_F1_33);
        let (t0, v0) = dense_random_f64(space.d, &mut rng);
        let w = CoeffTensor::<f64>::zeros(space.d, 6);
        let _ = SparseHamiltonian::new_refillable(
            space,
            &t0,
            &v0,
            Some(&w as &dyn ThreeBody<f64>),
            TermScreen::SymmetryAllowed(space.pg),
            Construction::Bipartite,
        );
    }
}

impl<T: Scalar> crate::mem::Footprint for SparseHamiltonian<T> {
    fn bytes(&self) -> u64 {
        let blocks: u64 = self
            .nests
            .iter()
            .flat_map(|n| n.cells.iter())
            .map(|c| c.csr.bytes() + c.csc.bytes() + c.src.as_ref().map_or(0, |s| s.bytes()))
            .sum();
        let states: u64 = self
            .states
            .iter()
            .flat_map(|g| g.iter())
            .map(|(_, v)| (v.capacity() * std::mem::size_of::<ChannelDesc>()) as u64)
            .sum();
        let state_qns: u64 = self
            .state_qns
            .iter()
            .flat_map(|g| g.iter())
            .map(|(_, v)| (v.capacity() * std::mem::size_of::<Sector>()) as u64)
            .sum();
        let refill = self
            .refill_gamma
            .as_ref()
            .map_or(0, |g| (g.capacity() * std::mem::size_of::<T>()) as u64);
        blocks + states + state_qns + refill
    }
}

impl<T: Scalar> SparseHamiltonian<T> {
    /// Predicted logical bytes of the environment `Sketch` at every bond `0..=d` for a train with
    /// the given `d+1` bond rank grids — computed from the rank profile and the automaton's
    /// `state_qns` tables alone, so a rank ladder can be sized before anything is built.
    pub fn predicted_env_bytes(&self, ranks: &[RankGrid]) -> Vec<u64> {
        let d = self.space.d;
        assert_eq!(
            ranks.len(),
            d + 1,
            "predicted_env_bytes: need d+1 rank grids"
        );
        (0..=d)
            .map(|b| {
                self.space
                    .bond_sectors(b)
                    .iter()
                    .map(|&s| {
                        let rows = ranks[b][s] as u64;
                        let cols: u64 = if b == 0 || b == d {
                            rows
                        } else {
                            self.state_qns[b][s]
                                .iter()
                                .map(|&m| ranks[b][m] as u64)
                                .sum()
                        };
                        rows * cols * std::mem::size_of::<T>() as u64
                    })
                    .sum()
            })
            .collect()
    }

    /// Predicted logical bytes of the two wide halves `(Fᴸ·H·Xₖ, H·Xₖ₊₁·Fᴿ)` the two-site apply
    /// at pair `(k, k+1)` materializes, for a train with the given rank grids. The QC mid rank at
    /// mid sector `m` (bond `k+1`) is bounded by
    /// `min(Σ_fi ranks[k][fi⁻¹m], Σ_fj ranks[k+2][fj m])` — the rows and columns of the two-site
    /// block at `m` — so this is an upper bound on what `factor_qc` produces (it is also capped by
    /// the reachable-path count, which is ignored here). Both halves are then sized exactly as
    /// [`Self::col_ranks`] / [`Self::row_ranks`] size a core whose mid bond carries that rank.
    pub fn predicted_two_site_half_bytes(&self, ranks: &[RankGrid], k: usize) -> (u64, u64) {
        let space = self.space;
        let d = space.d;
        assert!(
            k + 2 <= d,
            "predicted_two_site_half_bytes: pair ({k},{}) needs bond k+2 ≤ d",
            k + 1
        );
        assert_eq!(ranks.len(), d + 1, "need d+1 rank grids");
        let gl = space.pg.irrep_of(k);
        let gr = space.pg.irrep_of(k + 1);
        use crate::core::Family;
        let mut r_mid = RankGrid::zeros(&space);
        for &m in &space.bond_sectors(k + 1) {
            let rows: usize = Family::ALL
                .into_iter()
                .filter_map(|fi| fi.unapply(m, gl))
                .filter(|&l| space.contains(k, l))
                .map(|l| ranks[k][l])
                .sum();
            let cols: usize = Family::ALL
                .into_iter()
                .map(|fj| fj.apply(m, gr))
                .filter(|&r| space.contains(k + 2, r))
                .map(|r| ranks[k + 2][r])
                .sum();
            r_mid[m] = rows.min(cols);
        }
        let mut mid_exp = RankGrid::zeros(&space);
        for &s in &space.bond_sectors(k + 1) {
            mid_exp[s] = self.state_qns[k + 1][s].iter().map(|&m| r_mid[m]).sum();
        }
        let sz = std::mem::size_of::<T>() as u64;
        let mut l = 0u64;
        for fi in Family::ALL {
            for &lsec in &space.bond_sectors(k) {
                let s = fi.apply(lsec, gl);
                if space.contains(k + 1, s) {
                    l += (ranks[k][lsec] as u64) * (mid_exp[s] as u64) * sz;
                }
            }
        }
        let mut r = 0u64;
        for fj in Family::ALL {
            for &s in &space.bond_sectors(k + 1) {
                let t = fj.apply(s, gr);
                if space.contains(k + 2, t) {
                    r += (mid_exp[s] as u64) * (ranks[k + 2][t] as u64) * sz;
                }
            }
        }
        (l, r)
    }
}

#[cfg(test)]
mod phase4_tests {
    use crate::coeff::CoeffTensor;
    use crate::hamiltonian::SparseHamiltonian;
    use crate::qn::QnSpace;
    use crate::round::RoundMode;
    use crate::scalar::Scalar;
    use crate::test_util::randomized_state;
    use crate::{C64, TtVector, rayleigh_quotient, xt_h_y};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn complex_state(
        space: QnSpace,
        lo: usize,
        hi: usize,
        rng: &mut Xoshiro256PlusPlus,
    ) -> TtVector<C64> {
        let mut x = randomized_state::<C64, _>(space, lo, hi, rng);
        x.round_mut(1e-14, crate::qn::Pooled::NONE, RoundMode::Abs);
        x
    }

    fn complex_hermitian_t() -> [(usize, usize, C64); 2] {
        [
            (1, 3, -C64::from_polar(1.0, 0.9)),
            (3, 1, -C64::from_polar(1.0, -0.9)),
        ]
    }

    #[test]
    fn t3_ham_complex_hermitian_rq() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<C64>::zeros(d, 2);
        for &(i, j, c) in &complex_hermitian_t() {
            t.set(&[i, j], c);
        }
        let v = CoeffTensor::<C64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC88);
        let x = complex_state(space, 2, 4, &mut rng);
        let rq = rayleigh_quotient(&h, &x);
        assert!(
            rq.im_s().abs() < 1e-10,
            "Hermitian H must give a real Rayleigh quotient, imag={}",
            rq.im_s()
        );
    }

    #[test]
    fn t3_ham_hermiticity() {
        let d = 6usize;
        let space = QnSpace::new(d, 1, 1);
        let mut t = CoeffTensor::<C64>::zeros(d, 2);
        for &(i, j, c) in &complex_hermitian_t() {
            t.set(&[i, j], c);
        }
        let v = CoeffTensor::<C64>::zeros(d, 4);
        let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC03);
        let x = complex_state(space, 2, 4, &mut rng);
        let y = complex_state(space, 2, 4, &mut rng);

        let lhs = xt_h_y(&y, &h, &x);
        let rhs = xt_h_y(&x, &h, &y).conj_s();
        let diff = (lhs - rhs).abs_s();
        let scale = lhs.abs_s().max(rhs.abs_s()).max(1e-300);
        assert!(
            diff <= 1e-10 * scale,
            "H is not Hermitian: ⟨y,Hx⟩ ≠ conj⟨x,Hy⟩"
        );
    }
}

#[cfg(test)]
mod mem_tests {
    use crate::mem::Footprint;
    use crate::qn::QnSpace;
    use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
    use crate::test_util::{build_h_matvec, randomized_state, seeded};

    #[test]
    fn predicted_env_bytes_equals_built_stacks() {
        let space = QnSpace::new(6, 2, 2);
        let h = build_h_matvec(space);
        let mut rng = seeded(0x3E31_0001);
        let x = randomized_state::<f64, _>(space, 1, 3, &mut rng);
        let pred = h.predicted_env_bytes(&x.ranks);
        let fl = left_to_right_sketching(&h, &x);
        let fr = right_to_left_sketching(&h, &x);
        let d = space.d;
        assert_eq!(pred.len(), d + 1);
        for b in 0..d {
            assert_eq!(pred[b], fl[b].bytes(), "Fᴸ at bond {b}");
        }
        for b in 1..=d {
            assert_eq!(pred[b], fr[b - 1].bytes(), "Fᴿ at bond {b}");
        }
        assert!(
            pred[1..d].iter().any(|&v| v > 8 * 8),
            "fixture too small to discriminate"
        );
    }

    #[test]
    fn predicted_two_site_halves_bound_the_apply() {
        use crate::core::twosite::{contract, factor_qc};
        use crate::hamiltonian::matvec::{h_sketch_mul, sketch_h_mul};
        let space = QnSpace::new(6, 2, 2);
        let h = build_h_matvec(space);
        let mut rng = seeded(0x3E31_0002);
        let mut x = randomized_state::<f64, _>(space, 1, 3, &mut rng);
        x.right_orthogonalize(true);
        let fl = left_to_right_sketching(&h, &x);
        let fr = right_to_left_sketching(&h, &x);
        for k in 0..space.d - 1 {
            let (pl, pr) = h.predicted_two_site_half_bytes(&x.ranks, k);
            let cp = contract(x.core(k), x.core(k + 1));
            let (xk, xk1) = factor_qc(&cp);
            let l = sketch_h_mul(&fl[k], &h, &xk);
            let r = h_sketch_mul(&h, &xk1, &fr[k + 1]);
            assert!(
                l.bytes() <= pl,
                "pair {k}: l {} > predicted {pl}",
                l.bytes()
            );
            assert!(
                r.bytes() <= pr,
                "pair {k}: r {} > predicted {pr}",
                r.bytes()
            );
        }
    }
}

#[cfg(test)]
mod block_tests;
