// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::{MultiTt, validate_weights};
use crate::core::{Family, SparseCore};
use crate::qn::Sector;
use crate::scalar::Scalar;
use crate::vector::TtVector;

impl<T: Scalar> MultiTt<T> {
    /// The `R` columns of an open-boundary block (boundary width `R =`
    /// [`omega.boundary_width()`](TtVector::boundary_width)) as a container at core position
    /// `d − 1`: a clone of the block is left-orthogonalized, cores `0..d−2` become the shared
    /// base, and column `j` of the last core becomes centre `j` — so
    /// [`state(j)`](Self::state) equals `omega.column(j)` to roundoff. The weights are the
    /// caller's, in column (label) order, validated by the state-average rule (finite, strictly
    /// positive, non-increasing, summing to 1).
    ///
    /// # Panics
    /// When the block's bond-`d` rank sits in more than one sector (possible only with a point
    /// group, where every irrep is admissible at bond `d`), when its width is 0, on weights that
    /// break the rule above, on a malformed train (`TtVector::check`), and when bond 0 does not
    /// carry unit rank.
    pub fn from_open_block(omega: &TtVector<T>, weights: Vec<f64>) -> Self {
        let space = omega.space;
        let d = space.d;
        let last = Sector::with_irrep(space.nup, space.ndn, space.target);
        for (s, &r) in omega.ranks[d].iter() {
            assert!(
                r == 0 || s == last,
                "MultiTt::from_open_block: the block's bond-{d} rank sits in {s:?} (rank {r}) as \
                 well as the boundary sector {last:?} — an open block carries its R columns in ONE \
                 sector"
            );
        }
        let r = omega.boundary_width();
        assert!(
            r >= 1,
            "MultiTt::from_open_block: an open block has width ≥ 1 (boundary_width() = 0 at \
             {last:?})"
        );
        validate_weights(&weights, r, "MultiTt::from_open_block");
        omega.check();
        assert_eq!(
            omega.rank(0, Sector::new(0, 0)),
            1,
            "MultiTt::from_open_block: the block needs unit rank at bond 0"
        );

        let mut ol = omega.clone();
        ol.left_orthogonalize(false);
        debug_assert_eq!(ol.ortho, Some(d - 1));
        let centers: Vec<SparseCore<T>> =
            (0..r).map(|j| ol.column(j).cores[d - 1].clone()).collect();
        let base = ol.column(0);
        let m = MultiTt {
            base,
            centers,
            weights,
            core_position: d - 1,
        };
        m.check();
        m
    }

    /// Inverse of [`from_open_block`](Self::from_open_block): the container as ONE open-boundary
    /// block of width `n = nstates()` whose column `j` is [`state(j)`](Self::state) — the `n`
    /// centres stacked, in label order, along the bond-`d` boundary sector
    /// (`dst(f, s)[i, j] = centers[j](f, s)[i, 0]`). Every other core is the base's, unchanged.
    ///
    /// # Panics
    /// When the container breaks its invariant ([`check`](Self::check)), is not at core position
    /// `d − 1` (the block form; the shared cores are never re-gauged here), or its base lacks a
    /// unit bond-`d` boundary.
    pub fn to_open_block(&self) -> TtVector<T> {
        self.check();
        let space = self.base.space;
        let d = space.d;
        let n = self.nstates();
        assert_eq!(
            self.core_position,
            d - 1,
            "MultiTt::to_open_block: the container must sit at core position {} (the block form), \
             found {} — shared cores are never re-gauged here",
            d - 1,
            self.core_position
        );
        let last = Sector::with_irrep(space.nup, space.ndn, space.target);
        assert_eq!(
            self.base.boundary_width(),
            1,
            "MultiTt::to_open_block: the shared train must have a unit bond-{d} boundary"
        );
        let mut col_ranks = self.base.ranks[d].clone();
        col_ranks[last] = n;
        let mut core = SparseCore::zeros(space, d - 1, &self.base.ranks[d - 1], &col_ranks);
        for f in Family::ALL {
            let secs: Vec<Sector> = core.transitions(f).collect();
            for s in secs {
                let Some((rows, cols)) = core.block_dims(f, s) else {
                    continue;
                };
                if cols == 0 {
                    continue;
                }
                debug_assert_eq!(cols, n, "the boundary sector is the only open one");
                let mut buf = vec![T::zero(); rows * n];
                for (j, c) in self.centers.iter().enumerate() {
                    c.block_copy_out(f, s, &mut buf[j * rows..(j + 1) * rows])
                        .expect("a centre has every transition of the block's last core");
                }
                core.block_copy_in(f, s, &buf);
            }
        }
        let mut cores = self.base.cores.clone();
        cores[d - 1] = core;
        let mut out = TtVector::new(space, cores);
        out.ortho = Some(d - 1);
        out
    }
}
