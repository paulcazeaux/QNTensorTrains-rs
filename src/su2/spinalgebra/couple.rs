// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::helpers::{descending, mat4_mul};
use super::siteops::{OCC_SPIN, OCC_STATES};
use super::{SpinAlgebra, TensorOp};

#[allow(dead_code)]
impl SpinAlgebra {
    /// Coupled product `[T1 ⊗ T2]^κ` (`couple`). `T2` acts first (matrix product
    /// `T1.components[i1] *
    /// T2.components[i2]`, operator-order composition). Domain guard `|k1-k2| ≤ κ ≤ k1+k2`
    /// else `panic!`.
    pub fn couple(&mut self, t1: &TensorOp, t2: &TensorOp, two_kappa: i32) -> TensorOp {
        assert!(
            (t1.two_k - t2.two_k).abs() <= two_kappa && two_kappa <= t1.two_k + t2.two_k,
            "rank {two_kappa} out of range"
        );
        let mut comps = vec![[[0.0f64; 4]; 4]; (two_kappa + 1) as usize];
        for (i1, twomu1) in descending(t1.two_k) {
            for (i2, twomu2) in descending(t2.two_k) {
                let twomu = twomu1 + twomu2;
                if twomu.abs() > two_kappa {
                    continue;
                }
                let c = self.clebschgordan(t1.two_k, twomu1, t2.two_k, twomu2, two_kappa, twomu);
                if c == 0.0 {
                    continue;
                }
                let i = ((two_kappa - twomu) / 2) as usize;
                let prod = mat4_mul(&t1.components[i1], &t2.components[i2]);
                for a in 0..4 {
                    for b in 0..4 {
                        comps[i][a][b] += c * prod[a][b];
                    }
                }
            }
        }
        TensorOp {
            two_k: two_kappa,
            flux: t1.flux + t2.flux,
            odd: t1.odd ^ t2.odd,
            components: comps,
        }
    }

    /// Wigner–Eckart projection to per-`(occ_x -> occ_y)` reduced elements. Returns `0.0` (not
    /// `NaN`) where no transition exists. Both
    /// convention-tripwire asserts (Wigner–Eckart violation + cross-component inconsistency) are
    /// release `assert!`s, not `debug_assert!`.
    pub fn reduce_siteop(&mut self, op: &TensorOp) -> [[f64; 3]; 3] {
        let atol = 1e-13;
        let mut red = [[0.0f64; 3]; 3];
        for ox in 0..3 {
            for oy in 0..3 {
                let sx = OCC_SPIN[ox];
                let sy = OCC_SPIN[oy];
                let mut vals: Vec<f64> = Vec::new();
                let mut allzero = true;
                for (imu, twomu) in descending(op.two_k) {
                    for (ix, twomux) in descending(sx) {
                        for (iy, twomuy) in descending(sy) {
                            let el = op.components[imu][OCC_STATES[oy][iy]][OCC_STATES[ox][ix]];
                            let cg = self.clebschgordan(sx, twomux, op.two_k, twomu, sy, twomuy);
                            if cg == 0.0 {
                                assert!(
                                    el.abs() <= atol,
                                    "Wigner-Eckart violation: nonzero element where CG vanishes"
                                );
                            } else {
                                if el.abs() > atol {
                                    allzero = false;
                                }
                                vals.push(el / cg);
                            }
                        }
                    }
                }
                if !allzero {
                    let mut r = vals[0];
                    let mut rabs = vals[0].abs();
                    for &v in &vals {
                        if v.abs() > rabs {
                            rabs = v.abs();
                            r = v;
                        }
                    }
                    assert!(
                        vals.iter()
                            .all(|v| (v - r).abs() <= 1e-11 * f64::max(1.0, r.abs())),
                        "Wigner-Eckart inconsistency across components: {vals:?}"
                    );
                    red[ox][oy] = vals.iter().sum::<f64>() / vals.len() as f64;
                }
            }
        }
        red
    }
}
