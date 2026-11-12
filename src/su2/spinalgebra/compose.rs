// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::siteops::{OCC_SPIN, OCC_STATES};
use super::{SpinAlgebra, TensorOp};

const EXT: usize = 3;

#[inline]
fn tri(a: i32, b: i32, c: i32) -> bool {
    (a - b).abs() <= c && c <= a + b && (a + b + c) % 2 == 0
}

fn unit_tensor(sp: &mut SpinAlgebra, occ_in: usize, occ_out: usize, two_r: i32) -> TensorOp {
    let (sx, sy) = (OCC_SPIN[occ_in], OCC_SPIN[occ_out]);
    let mut comps = vec![[[0.0f64; 4]; 4]; (two_r + 1) as usize];
    for (imu, mu) in (0..).zip((-two_r..=two_r).rev().step_by(2)) {
        for (ix, mx) in (0..).zip((-sx..=sx).rev().step_by(2)) {
            for (iy, my) in (0..).zip((-sy..=sy).rev().step_by(2)) {
                comps[imu][OCC_STATES[occ_out][iy]][OCC_STATES[occ_in][ix]] =
                    sp.clebschgordan(sx, mx, two_r, mu, sy, my);
            }
        }
    }
    TensorOp {
        two_k: two_r,
        flux: occ_out as i32 - occ_in as i32,
        odd: (occ_in + occ_out) % 2 == 1,
        components: comps,
    }
}

#[inline]
fn site_op_exists(occ_in: usize, occ_out: usize, two_r: i32) -> bool {
    two_r <= 2 && tri(OCC_SPIN[occ_in], two_r, OCC_SPIN[occ_out])
}

/// The composed reduced site elements `S` of the product `A·B` (module docs), keyed by
/// `(occ_in(B), occ_mid, occ_out(A), 2r_B, 2r_A, 2κ)`: plain data (729 `f64`s), built once by
/// [`ComposeTable::new`] with a local [`SpinAlgebra`] that never leaves the constructor, so the
/// table is `Send + Sync` and is shared by reference.
#[derive(Clone, Debug, PartialEq)]
pub struct ComposeTable {
    s: [f64; EXT * EXT * EXT * EXT * EXT * EXT],
}

impl Default for ComposeTable {
    fn default() -> Self {
        Self::new()
    }
}

impl ComposeTable {
    /// Build the table from [`SpinAlgebra::couple`] and [`SpinAlgebra::reduce_siteop`] on unit
    /// site tensors (Clebsch–Gordan values exact-rational up to one rounding each), with the
    /// coupling-order phase `(−1)^{(2r_A + 2r_B − 2κ)/2}` that turns `couple`'s A-first element
    /// into the B-first one (module docs). About 0.1 ms per build (measured 115 µs); the
    /// [`SpinAlgebra`] is local.
    pub fn new() -> Self {
        let mut sp = SpinAlgebra::new();
        let mut s = [0.0f64; EXT * EXT * EXT * EXT * EXT * EXT];
        for occ_in_b in 0..EXT {
            for occ_mid in 0..EXT {
                for occ_out_a in 0..EXT {
                    for two_r_b in 0..=2i32 {
                        if !site_op_exists(occ_in_b, occ_mid, two_r_b) {
                            continue;
                        }
                        let tb = unit_tensor(&mut sp, occ_in_b, occ_mid, two_r_b);
                        for two_r_a in 0..=2i32 {
                            if !site_op_exists(occ_mid, occ_out_a, two_r_a) {
                                continue;
                            }
                            let ta = unit_tensor(&mut sp, occ_mid, occ_out_a, two_r_a);
                            for two_kappa in (two_r_a - two_r_b).abs()..=(two_r_a + two_r_b).min(2)
                            {
                                if !tri(two_r_a, two_r_b, two_kappa)
                                    || !site_op_exists(occ_in_b, occ_out_a, two_kappa)
                                {
                                    continue;
                                }
                                let coupled = sp.couple(&ta, &tb, two_kappa);
                                let red = sp.reduce_siteop(&coupled);
                                let phase = if ((two_r_a + two_r_b - two_kappa) / 2) % 2 == 0 {
                                    1.0
                                } else {
                                    -1.0
                                };
                                s[Self::index(
                                    occ_in_b,
                                    occ_mid,
                                    occ_out_a,
                                    two_r_b as usize,
                                    two_r_a as usize,
                                    two_kappa as usize,
                                )] = phase * red[occ_in_b][occ_out_a];
                            }
                        }
                    }
                }
            }
        }
        ComposeTable { s }
    }

    #[inline]
    fn index(
        occ_in_b: usize,
        occ_mid: usize,
        occ_out_a: usize,
        two_r_b: usize,
        two_r_a: usize,
        two_kappa: usize,
    ) -> usize {
        ((((occ_in_b * EXT + occ_mid) * EXT + occ_out_a) * EXT + two_r_b) * EXT + two_r_a) * EXT
            + two_kappa
    }

    /// The B-first composed element `S` of `A·B` on one site (module docs): `B` takes
    /// `occ_in_b → occ_mid` at doubled rank `two_r_b`, then `A` takes `occ_mid → occ_out_a` at
    /// `two_r_a`, recoupled onto `two_kappa`. `0.0` when no such composition exists (any key out
    /// of range included).
    #[inline]
    pub fn element(
        &self,
        occ_in_b: usize,
        occ_mid: usize,
        occ_out_a: usize,
        two_r_b: usize,
        two_r_a: usize,
        two_kappa: usize,
    ) -> f64 {
        if occ_in_b >= EXT
            || occ_mid >= EXT
            || occ_out_a >= EXT
            || two_r_b >= EXT
            || two_r_a >= EXT
            || two_kappa >= EXT
        {
            return 0.0;
        }
        self.s[Self::index(occ_in_b, occ_mid, occ_out_a, two_r_b, two_r_a, two_kappa)]
    }
}
