// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use faer::prelude::ReborrowMut;

use crate::pointgroup::Irrep;
use crate::scalar::Scalar;
use crate::su2::core::{Family, Su2Core};
use crate::su2::frame::Su2Sketch;
use crate::su2::hamiltonian::assembly::{WireGate, blockclass};
use crate::su2::qn::{Su2RankGrid, Su2Sector, Su2Space};
use crate::su2::spinalgebra::{SpinAlgebra, tensor_identity};
use crate::su2::tensorops::{WireSite, src_offset, wire_core, wire_ranks};
use crate::su2::vector::Su2TtVector;
use std::sync::Arc;

pub(crate) fn assert_feasible_channel(space: &Su2Space, two_s_flux: usize) {
    let bound = space.n.min(2 * space.d - space.n);
    assert!(
        two_s_flux <= bound,
        "infeasible spin channel: the requested 2S' = {two_s_flux} exceeds min(N, 2d-N) = {bound} \
         (N = {}, d = {}), so the target sector does not exist",
        space.n,
        space.d
    );
    assert!(
        (space.n + two_s_flux).is_multiple_of(2),
        "infeasible spin channel: the requested 2S' = {two_s_flux} has the wrong parity against \
         N = {} (N - 2S' must be even); the bound is min(N, 2d-N) = {bound}",
        space.n
    );
}

pub(crate) fn shifted_out_space(space: Su2Space, two_s_flux: usize) -> Su2Space {
    if two_s_flux == 0 {
        return space;
    }
    assert!(
        two_s_flux == 0 || space.two_s == 0,
        "shifted_out_space: a spin flux 2S_flux = {two_s_flux} is defined only at a singlet \
         reference (2S = 0), got 2S = {}. The dressed frame's theta-canonicality (C-spec §4 R1) \
         rests on the suffix Gram being m-INDEPENDENT, which holds only at 2S = 0; for 2S > 0 the \
         suffix frame is an intertwiner and the borrowing needs a second, genuinely three-index \
         recoupling that does not exist yet — the missing generalization, not a policy",
        space.two_s
    );
    assert_feasible_channel(&space, two_s_flux);
    space.with_two_s(two_s_flux)
}

pub(crate) fn flux_ranks(
    r: &Su2RankGrid,
    two_kappa: i64,
    space: Su2Space,
    out_space: Su2Space,
    bond: usize,
) -> Su2RankGrid {
    wire_ranks(
        r,
        two_kappa,
        0,
        Irrep::IDENTITY,
        space,
        out_space,
        bond,
        WireGate::Triangle,
    )
}

pub(crate) fn flux_grids(
    r: &[Su2RankGrid],
    two_kappa: i64,
    space: Su2Space,
    out_space: Su2Space,
) -> Vec<Su2RankGrid> {
    (0..=space.d)
        .map(|b| flux_ranks(&r[b], two_kappa, space, out_space, b))
        .collect()
}

pub(crate) fn dress_core<T: Scalar>(
    src: &Su2Core<T>,
    r: (&Su2RankGrid, &Su2RankGrid),
    rp: (&Su2RankGrid, &Su2RankGrid),
    space: Su2Space,
    out_space: Su2Space,
    two_kappa: i64,
    spin: &mut SpinAlgebra,
) -> Su2Core<T> {
    if two_kappa == 0 {
        return regrid_core(src, out_space, rp.0, rp.1);
    }
    let id_red = spin.reduce_siteop(&tensor_identity());
    let site = WireSite {
        red: &id_red,
        op_odd: false,
        op_two_k: 0,
        wire_l: two_kappa,
        wire_r: two_kappa,
        flux_l: 0,
        flux_r: 0,
        irr_l: Irrep::IDENTITY,
        irr_r: Irrep::IDENTITY,
    };
    wire_core(
        src,
        r,
        space,
        out_space,
        rp,
        &site,
        spin,
        WireGate::Triangle,
    )
}

pub(crate) fn dress_train<T: Scalar>(
    x: &Su2TtVector<T>,
    out_space: Su2Space,
    two_kappa: i64,
    spin: &mut SpinAlgebra,
) -> Su2TtVector<T> {
    let space = x.space;
    assert!(
        out_space.same_orbital_lattice(&space),
        "dress_train: the output space must share the orbital lattice of the source ({space:?} vs \
         {out_space:?})"
    );
    let d = space.d;
    let rp = flux_grids(&x.ranks, two_kappa, space, out_space);
    let cores: Vec<Su2Core<T>> = (0..d)
        .map(|k| {
            dress_core(
                x.core(k),
                (&x.ranks[k], &x.ranks[k + 1]),
                (&rp[k], &rp[k + 1]),
                space,
                out_space,
                two_kappa,
                spin,
            )
        })
        .collect();
    let mut out = Su2TtVector::new(out_space, cores);
    out.ortho = x.ortho;
    out
}

#[inline]
fn tri2(a: i64, b: i64, c: i64) -> bool {
    c >= (a - b).abs() && c <= a + b && (a + b + c) % 2 == 0
}

pub(crate) fn flux_ranks_zside(
    r: &Su2RankGrid,
    two_kappa: i64,
    space: Su2Space,
    out_space: Su2Space,
    bond: usize,
) -> Su2RankGrid {
    let full = flux_ranks(r, two_kappa, space, out_space, bond);
    let mut z = Su2RankGrid::zeros(&out_space);
    for s in out_space.bond_sectors(bond) {
        if space.contains(bond, s) {
            z[s] = full[s];
        }
    }
    z
}

pub(crate) fn flux_grids_zside(
    r: &[Su2RankGrid],
    two_kappa: i64,
    space: Su2Space,
    out_space: Su2Space,
) -> Vec<Su2RankGrid> {
    (0..=space.d)
        .map(|b| flux_ranks_zside(&r[b], two_kappa, space, out_space, b))
        .collect()
}

pub(crate) fn regrid_core<T: Scalar>(
    src: &Su2Core<T>,
    out_space: Su2Space,
    rows: &Su2RankGrid,
    cols: &Su2RankGrid,
) -> Su2Core<T> {
    let k = src.k;
    let mut out = Su2Core::zeros(out_space, k, rows, cols);
    let g = out_space.pg.irrep_of(k);
    for l in out_space.bond_sectors(k) {
        for f in Family::ALL {
            let Some(t) = f.apply(l, g) else { continue };
            if !out_space.contains(k + 1, t) {
                continue;
            }
            let Some(sb) = src.block(f, l) else { continue };
            let Some(mut db) = out.block_mut(f, l) else {
                continue;
            };
            for j in 0..db.ncols().min(sb.ncols()) {
                for i in 0..db.nrows().min(sb.nrows()) {
                    *db.rb_mut().get_mut(i, j) = *sb.get(i, j);
                }
            }
        }
    }
    out
}

pub(crate) fn regrid_train<T: Scalar>(
    src: &Su2TtVector<T>,
    out_space: Su2Space,
    grids: &[Su2RankGrid],
) -> Su2TtVector<T> {
    let d = src.space.d;
    let cores: Vec<Su2Core<T>> = (0..d)
        .map(|k| regrid_core(src.core(k), out_space, &grids[k], &grids[k + 1]))
        .collect();
    let mut out = Su2TtVector::new(out_space, cores);
    out.ortho = src.ortho;
    out
}

pub(crate) fn lift_bond_sketch<T: Scalar>(
    src: &Su2Sketch<T>,
    r: &Su2RankGrid,
    rp: &Su2RankGrid,
    out_space: Su2Space,
    two_kappa: i64,
) -> Su2Sketch<T> {
    let b = src.k;
    let big_n = out_space.n;
    let mut out = Su2Sketch::<T>::zeros(out_space, b, rp, rp);
    for t in out_space.bond_sectors(b) {
        if rp[t] == 0 {
            continue;
        }
        let Some(mut db) = out.block_mut(t) else {
            continue;
        };
        for u in src.space.bond_sectors(b) {
            let ru = r[u];
            let Some(off) = channel_offset(t, u, r, two_kappa, big_n) else {
                continue;
            };
            let Some(sb) = src.block(u) else { continue };
            for j in 0..ru.min(sb.ncols()) {
                for i in 0..ru.min(sb.nrows()) {
                    *db.rb_mut().get_mut(off + i, off + j) = *sb.get(i, j);
                }
            }
        }
    }
    out
}

pub(crate) fn channel_offset(
    t: Su2Sector,
    u: Su2Sector,
    r: &Su2RankGrid,
    two_kappa: i64,
    big_n: usize,
) -> Option<usize> {
    if r[u] == 0
        || u.n != t.n
        || u.irrep != t.irrep
        || !tri2(t.two_s as i64, u.two_s as i64, two_kappa)
    {
        return None;
    }
    let off = src_offset(
        t.n as i64,
        t.two_s as i64,
        u.two_s as i64,
        two_kappa,
        0,
        t.irrep,
        r,
        big_n,
        WireGate::Triangle,
    );
    (off >= 0).then_some(off as usize)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum CrossDir {
    BtoA,
    AtoB,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cross_wire<T: Scalar>(
    src: &Su2Core<T>,
    dir: CrossDir,
    two_kappa: i64,
    grids_in: (&Su2RankGrid, &Su2RankGrid),
    grids_out: (&Su2RankGrid, &Su2RankGrid),
    src_space: Su2Space,
    out_space: Su2Space,
    r: (&Su2RankGrid, &Su2RankGrid),
    spin: &mut SpinAlgebra,
) -> Su2Core<T> {
    let k = src.k;
    let mut dst = Su2Core::<T>::zeros(out_space, k, grids_out.0, grids_out.1);
    let (space_b, space_a, rdz, rz_col, rm, rd) = match dir {
        CrossDir::BtoA => (
            src_space,
            out_space,
            grids_in.0,
            grids_in.1,
            grids_out.0,
            grids_out.1,
        ),
        CrossDir::AtoB => (
            out_space,
            src_space,
            grids_out.0,
            grids_out.1,
            grids_in.0,
            grids_in.1,
        ),
    };
    let big_n = space_b.n;
    let g_k = space_b.pg.irrep_of(k);
    for a in space_b.bond_sectors(k) {
        let nr = r.0[a];
        if nr == 0 || rm[a] == 0 {
            continue;
        }
        for l in space_b.bond_sectors(k) {
            if rdz[l] == 0
                || l.n != a.n
                || l.irrep != a.irrep
                || !tri2(l.two_s as i64, a.two_s as i64, two_kappa)
            {
                continue;
            }
            let off_a = src_offset(
                l.n as i64,
                l.two_s as i64,
                a.two_s as i64,
                two_kappa,
                0,
                l.irrep,
                r.0,
                big_n,
                WireGate::Triangle,
            );
            if off_a < 0 {
                continue;
            }
            let off_a = off_a as usize;
            for f_a in Family::ALL {
                let Some(t) = f_a.apply(a, g_k) else { continue };
                if !space_a.contains(k + 1, t) || rd[t] == 0 {
                    continue;
                }
                let (dn_a, ds_a) = f_a.delta();
                let Some((_, occ_a, two_sigma)) = blockclass(dn_a as i64, ds_a) else {
                    continue;
                };
                for f_b in Family::ALL {
                    let (dn_b, ds_b) = f_b.delta();
                    let Some((_, occ_b, _)) = blockclass(dn_b as i64, ds_b) else {
                        continue;
                    };
                    if occ_a != occ_b {
                        continue;
                    }
                    let Some(u) = f_b.apply(l, g_k) else { continue };
                    if !space_b.contains(k + 1, u) || rz_col[u] == 0 || r.1[u] == 0 {
                        continue;
                    }
                    if !tri2(t.two_s as i64, u.two_s as i64, two_kappa) {
                        continue;
                    }
                    let off_u = src_offset(
                        t.n as i64,
                        t.two_s as i64,
                        u.two_s as i64,
                        two_kappa,
                        0,
                        t.irrep,
                        r.1,
                        big_n,
                        WireGate::Triangle,
                    );
                    if off_u < 0 {
                        continue;
                    }
                    let off_u = off_u as usize;
                    let c = spin.recouple(
                        a.two_s as i32,
                        t.two_s as i32,
                        two_sigma as i32,
                        two_kappa as i32,
                        two_kappa as i32,
                        0,
                        l.two_s as i32,
                        u.two_s as i32,
                        two_sigma as i32,
                    );
                    let phase = if ((t.two_s as i64 - u.two_s as i64) / 2).rem_euclid(2) == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    let phi = phase * ((u.two_s as f64 + 1.0) / (t.two_s as f64 + 1.0)).sqrt();
                    let c = match dir {
                        CrossDir::BtoA => c * phi,
                        CrossDir::AtoB => c / phi,
                    };
                    if c == 0.0 {
                        continue;
                    }
                    let alpha = T::from_f64(c);
                    let nc = rz_col[u];
                    debug_assert_eq!(
                        nc, r.1[u],
                        "cross_wire: the scheme-B column width at {u:?} must be x's own channel \
                         width, or the stacked offsets do not address it"
                    );
                    match dir {
                        CrossDir::BtoA => {
                            let Some(sb) = src.block(f_b, l) else {
                                continue;
                            };
                            let Some(mut db) = dst.block_mut(f_a, a) else {
                                continue;
                            };
                            for j in 0..nc {
                                for i in 0..nr {
                                    *db.rb_mut().get_mut(i, off_u + j) +=
                                        alpha * *sb.get(off_a + i, j);
                                }
                            }
                        }
                        CrossDir::AtoB => {
                            let Some(sb) = src.block(f_a, a) else {
                                continue;
                            };
                            let Some(mut db) = dst.block_mut(f_b, l) else {
                                continue;
                            };
                            for j in 0..nc {
                                for i in 0..nr {
                                    *db.rb_mut().get_mut(off_a + i, j) +=
                                        alpha * *sb.get(i, off_u + j);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    dst
}

pub(crate) struct Su2ShiftedFrames<T: Scalar> {
    pub(crate) two_s_flux: usize,
    pub(crate) out_space: Su2Space,
    pub(crate) frames: (Arc<Su2TtVector<T>>, Arc<Su2TtVector<T>>),
    pub(crate) zside: Su2TtVector<T>,
    pub(crate) rm: Vec<Su2RankGrid>,
    pub(crate) rd: Vec<Su2RankGrid>,
    pub(crate) rdz: Vec<Su2RankGrid>,
}

impl<T: Scalar> Su2ShiftedFrames<T> {
    pub(crate) fn new(base_l: &Su2TtVector<T>, base_r: &Su2TtVector<T>, two_s_flux: usize) -> Self {
        debug_assert_eq!(
            base_l.ranks, base_r.ranks,
            "Su2ShiftedFrames: the two frames must factorize the SAME state"
        );
        let space = base_l.space;
        let out_space = shifted_out_space(space, two_s_flux);
        let two_kappa = two_s_flux as i64;
        let r = &base_l.ranks;
        let rm = flux_grids(r, 0, space, out_space);
        let rd = flux_grids(r, two_kappa, space, out_space);
        let rdz = flux_grids_zside(r, two_kappa, space, out_space);
        let mut spin = SpinAlgebra::new();
        let frames = if two_s_flux == 0 {
            (Arc::new(base_l.clone()), Arc::new(base_r.clone()))
        } else {
            let u = dress_train(base_l, out_space, 0, &mut spin);
            let v = dress_train(base_r, out_space, two_kappa, &mut spin);
            (Arc::new(u), Arc::new(v))
        };
        let zside = regrid_train(
            &dress_train(base_l, out_space, two_kappa, &mut spin),
            space,
            &rdz,
        );
        Su2ShiftedFrames {
            two_s_flux,
            out_space,
            frames,
            zside,
            rm,
            rd,
            rdz,
        }
    }
}

#[allow(dead_code)]
pub(crate) fn slot_sites_shifted(
    space: &Su2Space,
    r: &[Su2RankGrid],
    two_s_flux: usize,
) -> Vec<usize> {
    let space = *space;
    let out_space = shifted_out_space(space, two_s_flux);
    let d = space.d;
    let rows = flux_grids(r, 0, space, out_space);
    let cols = flux_grids(r, two_s_flux as i64, space, out_space);
    (0..d)
        .filter(|&k| {
            let g = out_space.pg.irrep_of(k);
            let mut raw = 0usize;
            for l in out_space.bond_sectors(k) {
                for f in Family::ALL {
                    let Some(t) = f.apply(l, g) else { continue };
                    if out_space.contains(k + 1, t) {
                        raw += rows[k][l] * cols[k + 1][t];
                    }
                }
            }
            let gauge = if k < d - 1 {
                out_space
                    .bond_sectors(k + 1)
                    .into_iter()
                    .map(|t| rows[k + 1][t] * cols[k + 1][t])
                    .sum::<usize>()
            } else {
                0
            };
            raw > gauge
        })
        .collect()
}

#[cfg(test)]
mod cb_tests;
