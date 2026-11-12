// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use super::SpinAlgebra;

#[allow(dead_code)]
impl SpinAlgebra {
    /// Brute-force 6-CG recoupling factor. All nine args are two-unit angular momenta
    /// (`twoj,twojp,twosx, twok,twokp,twokop, twoJ,twoJp,twosy`) — PHYSICAL, no storage `+1`.
    /// Memoized on the 9-tuple key; the nested-loop summation order and the fixed 6-factor product
    /// order are carried VERBATIM (determinism).
    #[allow(clippy::too_many_arguments)]
    pub fn recouple(
        &mut self,
        two_j: i32,
        two_jp: i32,
        two_sx: i32,
        two_k: i32,
        two_kp: i32,
        two_kop: i32,
        two_big_j: i32,
        two_big_jp: i32,
        two_sy: i32,
    ) -> f64 {
        let key = [
            two_j, two_jp, two_sx, two_k, two_kp, two_kop, two_big_j, two_big_jp, two_sy,
        ];
        if let Some(&v) = self.recouple_cache.get(&key) {
            return v;
        }
        let mut f = 0.0f64;
        let mut twom = -two_j;
        while twom <= two_j {
            let mut twoq = -two_k;
            while twoq <= two_k {
                let mut twomux = -two_sx;
                while twomux <= two_sx {
                    let mut twomu = -two_kop;
                    while twomu <= two_kop {
                        let twomp = twom + twomux;
                        let twoqp = twoq + twomu;
                        let two_big_m = twom + twoq;
                        let twomuy = twomux + twomu;
                        let two_big_mp = two_big_m + twomuy;
                        if !(twomp.abs() <= two_jp && twoqp.abs() <= two_kp) {
                            twomu += 2;
                            continue;
                        }
                        if !(two_big_m.abs() <= two_big_j
                            && twomuy.abs() <= two_sy
                            && two_big_mp.abs() <= two_big_jp)
                        {
                            twomu += 2;
                            continue;
                        }
                        f += self.clebschgordan(two_j, twom, two_k, twoq, two_big_j, two_big_m)
                            * self.clebschgordan(
                                two_jp, twomp, two_kp, twoqp, two_big_jp, two_big_mp,
                            )
                            * self.clebschgordan(two_j, twom, two_sx, twomux, two_jp, twomp)
                            * self.clebschgordan(two_k, twoq, two_kop, twomu, two_kp, twoqp)
                            * self.clebschgordan(two_sx, twomux, two_kop, twomu, two_sy, twomuy)
                            * self.clebschgordan(
                                two_big_j, two_big_m, two_sy, twomuy, two_big_jp, two_big_mp,
                            );
                        twomu += 2;
                    }
                    twomux += 2;
                }
                twoq += 2;
            }
            twom += 2;
        }
        let v = f / (two_big_jp + 1) as f64;
        self.recouple_cache.insert(key, v);
        v
    }
}
