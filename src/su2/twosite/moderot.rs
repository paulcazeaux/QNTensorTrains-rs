// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::HashMap;

use faer::Mat;
use rayon::prelude::*;

use super::noise::ThetaMetric;
use super::{PAR_MIN_MID_SECTORS, Su2ContractedCores, contract, mpred, msucc};
use crate::core::factor::svals_qr;
use crate::core::twosite::{ModeRotOpts, minimize_angle};
use crate::entropy::bond_functional;
use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::su2::chop::pool_weight;
use crate::su2::core::{Family, Su2Core};
use crate::su2::hamiltonian::assembly::WireGate;
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::spinalgebra::{SiteOp, SpinAlgebra};
use crate::su2::tensorops::{wire_core, wire_ranks};
use crate::su2::transition1rdm::PairChannels;

const STRUCTURE_TOL: f64 = 1e-12;

pub(crate) struct ClassGenerator {
    pub(crate) l: Su2Sector,
    pub(crate) r: Su2Sector,
    pub(crate) mids: Vec<Su2Sector>,
    pub(crate) generator: Vec<f64>,
}

fn assert_representable(space: &Su2Space, k: usize) {
    let pg = space.pg;
    assert!(
        pg.irrep_of(k) == pg.irrep_of(k + 1),
        "mode rotation of orbitals ({}, {}): different irreps — a cross-irrep rotation is not \
         representable over this space",
        k,
        k + 1
    );
}

type ClassKey = (Su2Sector, Su2Sector);
type ClassPaths = (Su2Sector, Su2Sector, Vec<Su2Sector>);

fn classes_of(space: &Su2Space, k: usize) -> (Vec<ClassPaths>, HashMap<ClassKey, usize>) {
    let mut list: Vec<ClassPaths> = Vec::new();
    let mut index: HashMap<ClassKey, usize> = HashMap::new();
    for m in space.bond_sectors(k + 1) {
        if !space.contains(k + 1, m) {
            continue;
        }
        for (_, l) in mpred(space, k, m) {
            for (_, r) in msucc(space, k, m) {
                let ci = *index.entry((l, r)).or_insert_with(|| {
                    list.push((l, r, Vec::new()));
                    list.len() - 1
                });
                list[ci].2.push(m);
            }
        }
    }
    (list, index)
}

fn window_offset(win: &[(Family, Su2Sector)], ranks: &Su2RankGrid, s: Su2Sector) -> (usize, usize) {
    let mut off = 0usize;
    for &(_, t) in win {
        if t == s {
            return (off, ranks[s]);
        }
        off += ranks[t];
    }
    unreachable!("sector {s:?} not in the window")
}

pub(crate) fn class_generators(space: &Su2Space, k: usize) -> Vec<ClassGenerator> {
    assert_representable(space, k);
    let space = *space;
    let g = space.pg.irrep_of(k);
    let (list, index) = classes_of(&space, k);

    let mut rl = Su2RankGrid::zeros(&space);
    for l in space.bond_sectors(k) {
        if space.contains(k, l) {
            rl[l] = 1;
        }
    }
    let mut rm = Su2RankGrid::zeros(&space);
    for m in space.bond_sectors(k + 1) {
        if space.contains(k + 1, m) {
            rm[m] = mpred(&space, k, m).iter().map(|&(_, l)| rl[l]).sum();
        }
    }
    let mut rr = Su2RankGrid::zeros(&space);
    for (_, r, mids) in &list {
        assert!(
            mids.len() <= 4,
            "su2 mode rotation: class with {} paths (at most 4 coupling paths cross two sites)",
            mids.len()
        );
        rr[*r] = rr[*r].max(mids.len());
    }

    let mut xk = Su2Core::<f64>::zeros(space, k, &rl, &rm);
    let mut xk1 = Su2Core::<f64>::zeros(space, k + 1, &rm, &rr);
    for m in space.bond_sectors(k + 1) {
        if !space.contains(k + 1, m) {
            continue;
        }
        for (p, (f, l)) in mpred(&space, k, m).into_iter().enumerate() {
            let mut b = xk.block_mut(f, l).expect("probe: site-k block present");
            b[(0, p)] = 1.0;
            for (f1, r) in msucc(&space, k, m) {
                let cl = &list[index[&(l, r)]];
                let tag =
                    cl.2.iter()
                        .position(|&x| x == m)
                        .expect("m is on its class");
                let mut b = xk1
                    .block_mut(f1, m)
                    .expect("probe: site-(k+1) block present");
                b[(p, tag)] = 1.0;
            }
        }
    }

    let mut spin = SpinAlgebra::new();
    let ch = PairChannels::resolve(&mut spin);
    let dressed = |xk: &Su2Core<f64>,
                   xk1: &Su2Core<f64>,
                   flux: i64,
                   open: &SiteOp,
                   close: &SiteOp,
                   spin: &mut SpinAlgebra| {
        let gm = wire_ranks(&rm, ch.wire, flux, g, space, space, k + 1, WireGate::Window);
        let open_site = PairChannels::site(open, 0, ch.wire, 0, flux, Irrep::IDENTITY, g);
        let close_site = PairChannels::site(close, ch.wire, 0, flux, 0, g, Irrep::IDENTITY);
        let dk = wire_core(
            xk,
            (&rl, &rm),
            space,
            space,
            (&rl, &gm),
            &open_site,
            spin,
            WireGate::Window,
        );
        let dk1 = wire_core(
            xk1,
            (&rm, &rr),
            space,
            space,
            (&gm, &rr),
            &close_site,
            spin,
            WireGate::Window,
        );
        contract(&dk, &dk1)
    };
    let e_down = dressed(&xk, &xk1, -1, &ch.ann, &ch.cre, &mut spin);
    let e_up = dressed(&xk, &xk1, 1, &ch.cre, &ch.ann, &mut spin);
    let mut kz = &e_down * ch.w_gt;
    kz.axpy(-ch.w_lt, &e_up);

    list.into_iter()
        .map(|(l, r, mids)| {
            let n = mids.len();
            let mut generator = vec![0.0f64; n * n];
            for (i, &mi) in mids.iter().enumerate() {
                let blk = kz.block(mi).expect("probe: mid block present");
                let (row, _) = window_offset(&mpred(&space, k, mi), &rl, l);
                let (col, width) = window_offset(&msucc(&space, k, mi), &rr, r);
                for j in 0..n {
                    generator[i * n + j] = blk[(row, col + j)];
                }
                debug_assert!(
                    (n..width).all(|j| blk[(row, col + j)] == 0.0),
                    "probe: Ĝ leaked out of class ({l:?}, {r:?})"
                );
            }
            let scale = generator.iter().fold(1.0f64, |a, v| a.max(v.abs()));
            for i in 0..n {
                for j in 0..n {
                    let a = (generator[i * n + j] + generator[j * n + i]).abs();
                    assert!(
                        a <= STRUCTURE_TOL * scale,
                        "su2 mode rotation: class ({l:?}, {r:?}) generator not antisymmetric \
                         (|K+Kᵀ|[{i}][{j}] = {a:e}) — a coupling-convention defect, do not \
                         symmetrize"
                    );
                }
            }
            ClassGenerator {
                l,
                r,
                mids,
                generator,
            }
        })
        .collect()
}

fn matmul_rm(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for l in 0..n {
            let ail = a[i * n + l];
            for j in 0..n {
                c[i * n + j] += ail * b[l * n + j];
            }
        }
    }
    c
}

pub(crate) fn class_gate_cs(generator: &[f64], n: usize, c: f64, s: f64) -> Vec<f64> {
    debug_assert!(
        (c * c + s * s - 1.0).abs() <= 4.0 * f64::EPSILON,
        "class_gate_cs: c² + s² = {} ≠ 1",
        c * c + s * s
    );
    assert_eq!(generator.len(), n * n, "class_gate_cs: generator size");
    let mut g = vec![0.0f64; n * n];
    for i in 0..n {
        g[i * n + i] = 1.0;
    }
    if generator.iter().all(|&v| v == 0.0) {
        return g;
    }
    let kmax = generator.iter().fold(1.0f64, |a, v| a.max(v.abs()));
    let tol = STRUCTURE_TOL * kmax * kmax * kmax;
    let k2 = matmul_rm(generator, generator, n);
    let res1 = (0..n * n).fold(0.0f64, |a, ij| {
        let e = if ij / n == ij % n { 1.0 } else { 0.0 };
        a.max((k2[ij] + e).abs())
    });
    if res1 <= tol {
        for ij in 0..n * n {
            g[ij] = s * generator[ij];
        }
        for i in 0..n {
            g[i * n + i] += c;
        }
        return g;
    }
    let k3 = matmul_rm(&k2, generator, n);
    let res2 = (0..n * n).fold(0.0f64, |a, ij| a.max((k3[ij] + 4.0 * generator[ij]).abs()));
    assert!(
        res2 <= tol,
        "su2 mode rotation: class generator spectrum outside {{0, ±i, ±2i}} (|K²+I| = {res1:e}, \
         |K³+4K| = {res2:e}) — a coupling-convention defect"
    );
    let (a1, a2) = (c * s, 0.5 * s * s);
    for ij in 0..n * n {
        g[ij] += a1 * generator[ij] + a2 * k2[ij];
    }
    g
}

#[derive(Clone, Copy)]
struct Tile {
    sec: Su2Sector,
    off: usize,
    len: usize,
}

fn tilings<T: Scalar>(tc: &Su2ContractedCores<T>) -> HashMap<Su2Sector, (Vec<Tile>, Vec<Tile>)> {
    let (space, k) = (&tc.space, tc.k);
    let mut out = HashMap::new();
    for &m in &tc.mid_qn {
        if tc.block(m).is_none() {
            continue;
        }
        let tile = |win: Vec<(Family, Su2Sector)>, ranks: &Su2RankGrid| {
            let mut off = 0usize;
            win.into_iter()
                .map(|(_, sec)| {
                    let len = ranks[sec];
                    let t = Tile { sec, off, len };
                    off += len;
                    t
                })
                .collect::<Vec<_>>()
        };
        out.insert(
            m,
            (
                tile(mpred(space, k, m), &tc.row_ranks),
                tile(msucc(space, k, m), &tc.col_ranks),
            ),
        );
    }
    out
}

fn rotated_block<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    classes: &[ClassGenerator],
    gates: &[Vec<f64>],
    index: &HashMap<ClassKey, usize>,
    tiles: &HashMap<Su2Sector, (Vec<Tile>, Vec<Tile>)>,
    m: Su2Sector,
) -> Option<Mat<T>> {
    let src = tc.block(m)?;
    let (rows, cols) = tiles.get(&m)?;
    let mut out = Mat::<T>::zeros(src.nrows(), src.ncols());
    for rt in rows {
        if rt.len == 0 {
            continue;
        }
        for ct in cols {
            if ct.len == 0 {
                continue;
            }
            let ci = index[&(rt.sec, ct.sec)];
            let (cl, g) = (&classes[ci], &gates[ci]);
            let n = cl.mids.len();
            let i = cl
                .mids
                .iter()
                .position(|&x| x == m)
                .expect("m is on its class");
            for (j, &mj) in cl.mids.iter().enumerate() {
                let coef = g[i * n + j];
                if coef == 0.0 {
                    continue;
                }
                let sb = tc.block(mj).expect("class path block present");
                let (rows2, cols2) = &tiles[&mj];
                let r2 = rows2.iter().find(|t| t.sec == rt.sec).expect("row tile");
                let c2 = cols2.iter().find(|t| t.sec == ct.sec).expect("column tile");
                debug_assert_eq!((r2.len, c2.len), (rt.len, ct.len));
                let a = T::from_f64(coef);
                for cc in 0..ct.len {
                    for rr in 0..rt.len {
                        let v =
                            out[(rt.off + rr, ct.off + cc)] + a * sb[(r2.off + rr, c2.off + cc)];
                        out[(rt.off + rr, ct.off + cc)] = v;
                    }
                }
            }
        }
    }
    Some(out)
}

struct GateTables {
    classes: Vec<ClassGenerator>,
    index: HashMap<ClassKey, usize>,
    tiles: HashMap<Su2Sector, (Vec<Tile>, Vec<Tile>)>,
}

fn gate_tables<T: Scalar>(tc: &Su2ContractedCores<T>) -> GateTables {
    assert_representable(&tc.space, tc.k);
    let classes = class_generators(&tc.space, tc.k);
    let index: HashMap<ClassKey, usize> = classes
        .iter()
        .enumerate()
        .map(|(i, cl)| ((cl.l, cl.r), i))
        .collect();
    let tiles = tilings(tc);
    GateTables {
        classes,
        index,
        tiles,
    }
}

fn gates_at(classes: &[ClassGenerator], c: f64, s: f64) -> Vec<Vec<f64>> {
    classes
        .iter()
        .map(|cl| class_gate_cs(&cl.generator, cl.mids.len(), c, s))
        .collect()
}

fn apply_gate<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    c: f64,
    s: f64,
    par: bool,
) -> Su2ContractedCores<T> {
    let tab = gate_tables(tc);
    let gates = gates_at(&tab.classes, c, s);
    let rotate = |m: Su2Sector| {
        let b = rotated_block(tc, &tab.classes, &gates, &tab.index, &tab.tiles, m);
        (m, b)
    };
    let blocks: Vec<(Su2Sector, Option<Mat<T>>)> = if par && tc.mid_qn.len() >= PAR_MIN_MID_SECTORS
    {
        tc.mid_qn.par_iter().map(|&m| rotate(m)).collect()
    } else {
        tc.mid_qn.iter().map(|&m| rotate(m)).collect()
    };
    let mut out = tc.clone();
    for (m, b) in blocks {
        if let Some(b) = b {
            out.blocks[m] = Some(b);
        }
    }
    out
}

fn apply_impl<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    theta: f64,
    par: bool,
) -> Su2ContractedCores<T> {
    if theta == 0.0 {
        return tc.clone();
    }
    let (s, c) = theta.sin_cos();
    apply_gate(tc, c, s, par)
}

/// `G(θ) = exp(θ·Ĝ_{k+1,k})` applied to an SU(2) two-site block (out of place; the container
/// shape is θ-invariant). `theta == 0.0` returns an exact copy. Same operator and convention as
/// [`crate::core::twosite::apply_mode_rotation`]: `expand` of the result equals the
/// abelian gate on `expand` of the input, every `Sz` member. Threaded over output mid sectors
/// (disjoint writes) from `PAR_MIN_MID_SECTORS` up. Panics on a pair of different irreps (not
/// representable over a typed space). No truncation: split with [`super::factor_svd`] et al.
pub fn apply_mode_rotation<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    theta: f64,
) -> Su2ContractedCores<T> {
    apply_impl(tc, theta, true)
}

/// Serial twin of [`apply_mode_rotation`] (the matches-serial reference).
pub fn apply_mode_rotation_serial<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    theta: f64,
) -> Su2ContractedCores<T> {
    apply_impl(tc, theta, false)
}

pub(crate) fn apply_mode_rotation_cs<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    c: f64,
    s: f64,
) -> Su2ContractedCores<T> {
    apply_gate(tc, c, s, true)
}

/// The adjacent-orbital SWAP of an SU(2) two-site block: the pair rotation at the exact
/// `(c, s) = (0, 1)`, `S = exp((π/2)·Ĝ_{k+1,k})` with no `cos(π/2)` roundoff; its integral-side
/// partner is `rotate_integrals(·, crate::ordering::swap_matrix(d, k))`.
pub fn apply_mode_swap<T: Scalar>(tc: &Su2ContractedCores<T>) -> Su2ContractedCores<T> {
    apply_mode_rotation_cs(tc, 0.0, 1.0)
}

fn block_wvals<T: Scalar>(
    space: &Su2Space,
    met: &ThetaMetric<T>,
    m: Su2Sector,
    mut b: Mat<T>,
) -> Vec<f64> {
    if b.nrows() == 0 || b.ncols() == 0 {
        return Vec::new();
    }
    met.scale_in_place(&mut b, m);
    let w = pool_weight(space, m);
    svals_qr(b.as_ref()).into_iter().map(|s| w * s).collect()
}

fn pooled_over_mids(mid_qn: &[Su2Sector], per: impl Fn(Su2Sector) -> Vec<f64> + Sync) -> Vec<f64> {
    let all: Vec<Vec<f64>> = if mid_qn.len() >= PAR_MIN_MID_SECTORS {
        mid_qn.par_iter().map(|&m| per(m)).collect()
    } else {
        mid_qn.iter().map(|&m| per(m)).collect()
    };
    all.into_iter().flatten().collect()
}

/// The bond-`(k+1)` spectrum the SU(2) angle search scores, of `G(θ)·tc`: per mid sector `m` (in
/// `mid_qn` order, `min(rows, cols)` values per present non-empty block), the singular values `σ`
/// of the block with each successor column slice `(f, r)` scaled by `θ(2s_m, 2s_r)`, times
/// `√cw(2s_m, 2S)` — `ŵ = √cw·σ`, the values the two-site split's pooled chop truncates, up to
/// roundoff (`svals_qr` here, the split's `svd_uvt_svals` there). The name is the abelian twin's
/// ([`crate::core::twosite::rotated_bond_svals`]); unlike it these are WEIGHTED values.
pub fn rotated_bond_svals<T: Scalar>(tc: &Su2ContractedCores<T>, theta: f64) -> Vec<f64> {
    let rot = apply_mode_rotation(tc, theta);
    let met = ThetaMetric::new(&rot);
    pooled_over_mids(&rot.mid_qn, |m| {
        rot.block(m)
            .map(|b| block_wvals(&rot.space, &met, m, b.to_owned()))
            .unwrap_or_default()
    })
}

fn rotated_bond_wvals_tiled<T: Scalar>(
    tc: &Su2ContractedCores<T>,
    tab: &GateTables,
    met: &ThetaMetric<T>,
    theta: f64,
) -> Vec<f64> {
    let gates = (theta != 0.0).then(|| {
        let (s, c) = theta.sin_cos();
        gates_at(&tab.classes, c, s)
    });
    pooled_over_mids(&tc.mid_qn, |m| {
        let b = match &gates {
            Some(g) => rotated_block(tc, &tab.classes, g, &tab.index, &tab.tiles, m),
            None => tc.block(m).map(|b| b.to_owned()),
        };
        b.map(|b| block_wvals(&tc.space, met, m, b))
            .unwrap_or_default()
    })
}

/// Minimize `opts.functional` of the gated pair's weighted bond spectrum ([`rotated_bond_svals`]'s
/// `ŵ`) over `θ ∈ (−π/2, π/2]` — the SU(2) twin of [`crate::core::twosite::search_angle`], with the
/// same minimizer and options (even grid, golden section, `θ` EXACTLY `0.0` unless the functional
/// improves by `improve_tol`; 18 scorings at the defaults). Returns `(theta, f(0), f(theta))`. The
/// θ-free generator tables and the θ metric are built once per call; the scorer reads the
/// noise-free block. Panics on a pair of different irreps, and unless `opts.n_grid` is even and
/// `≥ 2`.
pub fn search_angle<T: Scalar>(tc: &Su2ContractedCores<T>, opts: &ModeRotOpts) -> (f64, f64, f64) {
    let (tab, met) = (gate_tables(tc), ThetaMetric::new(tc));
    minimize_angle(
        |theta: f64| {
            bond_functional(
                &rotated_bond_wvals_tiled(tc, &tab, &met, theta),
                opts.functional,
            )
        },
        opts,
    )
}

#[cfg(test)]
#[path = "moderot_tests.rs"]
mod tests;
