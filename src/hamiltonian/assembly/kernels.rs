// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::core::Family;
use crate::pointgroup::Irrep;
use crate::qn::{QnSpace, Sector};
use crate::scalar::Scalar;

use super::super::csx::CooCell;
use super::types::{CooEntry, CooNest, Flux, GridGeom, Op, SRC_NEG, SRC_NONE};

#[inline]
pub(crate) fn src_tag(src: u32, neg: bool) -> u32 {
    if src == SRC_NONE {
        SRC_NONE
    } else if neg {
        src | SRC_NEG
    } else {
        src
    }
}

pub(crate) fn occupation(l: Sector, r: Sector) -> Family {
    Family::from_occupation(r.nup > l.nup, r.ndn > l.ndn)
}

#[inline]
pub(crate) fn op_delta_cands(op: Op) -> &'static [usize] {
    match op {
        Op::A => &[0],
        Op::Id => &[0, 1],
        _ => &[1],
    }
}

#[inline]
pub(crate) fn op_irrep_flux(zu: Op, zdn: Op, g: Irrep) -> Irrep {
    let odd = |o: Op| matches!(o, Op::A | Op::Adag);
    if odd(zu) ^ odd(zdn) {
        g
    } else {
        Irrep::IDENTITY
    }
}

#[inline]
pub(crate) fn lr_sectors(
    n_minus: Sector,
    n_plus: Sector,
    flux_in: Flux,
    flux_out: Flux,
) -> (Sector, Sector) {
    let l = Sector::with_irrep(
        (n_minus.nup as i64 - flux_in.up as i64) as usize,
        (n_minus.ndn as i64 - flux_in.dn as i64) as usize,
        n_minus.irrep.fuse(flux_in.irrep.inverse()),
    );
    let r = Sector::with_irrep(
        (n_plus.nup as i64 - flux_out.up as i64) as usize,
        (n_plus.ndn as i64 - flux_out.dn as i64) as usize,
        n_plus.irrep.fuse(flux_out.irrep.inverse()),
    );
    (l, r)
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn push_coo_cell<T: Scalar>(
    coo: &mut NestBuilder<T>,
    n_minus: Sector,
    n_plus: Sector,
    x_occ: Family,
    l: Sector,
    row_col: (i32, i32),
    val: T,
    src: u32,
) {
    let y_occ = occupation(n_minus, n_plus);
    let (row, col) = row_col;
    super::super::construct_probe::bump(super::super::construct_probe::Counter::CooTriples, 1);
    let record_src = coo.record_src;
    let entry = coo.cell_mut(y_occ, n_minus, x_occ, l);
    entry.cell.rows.push(row);
    entry.cell.cols.push(col);
    entry.cell.vals.push(val);
    if record_src {
        entry.src.push(src);
    }
}

pub(crate) struct NestBuilder<T: Scalar> {
    geom: GridGeom,
    sectors: Vec<Sector>,
    index: Vec<u32>,
    cells: Vec<CooEntry<T>>,
    record_src: bool,
}

impl<T: Scalar> NestBuilder<T> {
    pub(crate) fn new(space: &QnSpace, k: usize) -> Self {
        Self::new_with_src(space, k, false)
    }

    pub(crate) fn new_with_src(space: &QnSpace, k: usize, record_src: bool) -> Self {
        let geom = GridGeom::of(space);
        let nsec = geom.nsec();
        NestBuilder {
            geom,
            sectors: space.bond_sectors(k),
            index: vec![u32::MAX; 16 * nsec * nsec],
            cells: Vec::new(),
            record_src,
        }
    }

    #[inline]
    fn ckey(&self, y_occ: Family, n1: Sector, x_occ: Family, m: Sector) -> usize {
        let nsec = self.geom.nsec();
        let oslot = (y_occ as usize) * nsec + self.geom.idx(n1);
        (oslot * 4 + x_occ as usize) * nsec + self.geom.idx(m)
    }

    #[inline]
    fn cell_mut(
        &mut self,
        y_occ: Family,
        n1: Sector,
        x_occ: Family,
        m: Sector,
    ) -> &mut CooEntry<T> {
        let ck = self.ckey(y_occ, n1, x_occ, m);
        let mut i = self.index[ck];
        if i == u32::MAX {
            let m_ord = self
                .sectors
                .iter()
                .position(|&s| s == m)
                .expect("COO x-row sector is admissible at this bond");
            i = self.cells.len() as u32;
            assert!(i != u32::MAX, "populated COO cells overflow the u32 index");
            self.index[ck] = i;
            self.cells.push(CooEntry {
                x_occ: x_occ as u8,
                m_ord: m_ord as u32,
                cell: CooCell::empty(),
                src: Vec::new(),
            });
        }
        &mut self.cells[i as usize]
    }

    pub(crate) fn freeze(mut self) -> CooNest<T> {
        let nsec = self.geom.nsec();
        let staged = self.cells.len();
        let mut offsets = Vec::with_capacity(4 * nsec + 1);
        let mut cells: Vec<CooEntry<T>> = Vec::with_capacity(staged);
        let mgrid: Vec<usize> = self.sectors.iter().map(|&s| self.geom.idx(s)).collect();
        for y_occ in 0..4usize {
            for n1g in 0..nsec {
                offsets.push(cells.len() as u32);
                let oslot = y_occ * nsec + n1g;
                for x_occ in 0..4usize {
                    let base = (oslot * 4 + x_occ) * nsec;
                    for &mg in &mgrid {
                        let i = self.index[base + mg];
                        if i == u32::MAX {
                            continue;
                        }
                        let e = &mut self.cells[i as usize];
                        cells.push(CooEntry {
                            x_occ: e.x_occ,
                            m_ord: e.m_ord,
                            cell: std::mem::take(&mut e.cell),
                            src: std::mem::take(&mut e.src),
                        });
                    }
                }
            }
        }
        offsets.push(cells.len() as u32);
        assert_eq!(
            cells.len(),
            staged,
            "COO freeze lost {} staged cell(s) — an x-sector outside bond_sectors(k)",
            staged - cells.len()
        );
        CooNest {
            offsets,
            cells,
            geom: self.geom,
            sectors: self.sectors,
        }
    }
}

pub(crate) fn jordanwigner_sign(l: Sector, r: Sector, op: (Op, Op)) -> (Family, bool) {
    let lup = l.nup;
    let ldn = l.ndn;
    let up_fermi = matches!(op.0, Op::A | Op::Adag);
    let dn_fermi = matches!(op.1, Op::A | Op::Adag);
    let jw_up_neg = up_fermi && (lup + ldn) % 2 == 1;
    let occ = occupation(l, r);
    let carry = match occ {
        Family::Up | Family::Occ => 1,
        _ => 0,
    };
    let dn_neg = dn_fermi && (lup + ldn + carry) % 2 == 1;
    (occ, jw_up_neg ^ dn_neg)
}

#[inline]
pub(crate) fn jw_apply<T: Scalar>(w: T, neg: bool) -> T {
    if neg { T::from_f64(-1.0) * w } else { w }
}

pub(crate) fn empty_nest<T: Scalar>(space: &QnSpace, k: usize) -> CooNest<T> {
    NestBuilder::new(space, k).freeze()
}
