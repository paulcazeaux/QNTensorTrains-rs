// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use crate::err::Error;
use crate::layer::Layer;

pub(crate) fn refuse_point_group<L: Layer>(space: &L::Space, twin: &str) -> Result<(), Error> {
    if L::space_nirrep(space) > 1 {
        return Err(Error::invalid(format!(
            "this train carries a non-trivial point group ({} irreps); use the {twin} symbol",
            L::space_nirrep(space)
        )));
    }
    Ok(())
}

pub(crate) fn ranks_to_flat<L: Layer>(space: &L::Space, grids: &[L::Grid]) -> Vec<usize> {
    let (rows, cols) = L::grid_extent(space);
    let mut out = Vec::with_capacity(grids.len() * rows * cols);
    for g in grids {
        for i in 0..rows {
            for j in 0..cols {
                out.push(L::grid_get(g, L::sector_new(i, j)));
            }
        }
    }
    out
}

pub(crate) fn ranks_from_flat<L: Layer>(
    space: &L::Space,
    flat: &[usize],
) -> Result<Vec<L::Grid>, Error> {
    let (rows, cols) = L::grid_extent(space);
    let (d, _, _) = L::space_parts(space);
    let stride = rows * cols;
    if flat.len() != (d + 1) * stride {
        return Err(Error::invalid(format!(
            "rank grid buffer has {} entries, need (d+1)·rows·cols = {}",
            flat.len(),
            (d + 1) * stride
        )));
    }
    Ok((0..=d)
        .map(|b| {
            let mut g = L::grid_zeros(space);
            for i in 0..rows {
                for j in 0..cols {
                    let r = flat[b * stride + i * cols + j];
                    if r != 0 {
                        L::grid_set(&mut g, L::sector_new(i, j), r);
                    }
                }
            }
            g
        })
        .collect())
}
