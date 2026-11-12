// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive};

use super::SpinAlgebra;

#[allow(dead_code)]
impl SpinAlgebra {
    /// Memoized Clebsch–Gordan coefficient `⟨j1 m1; j2 m2 | J M⟩`. Delegates to [`Self::cg_exact`] on a cache miss.
    pub fn clebschgordan(
        &mut self,
        two_j1: i32,
        two_m1: i32,
        two_j2: i32,
        two_m2: i32,
        two_j: i32,
        two_m: i32,
    ) -> f64 {
        let key = [two_j1, two_m1, two_j2, two_m2, two_j, two_m];
        if let Some(&v) = self.cg_cache.get(&key) {
            return v;
        }
        let v = self.cg_exact(two_j1, two_m1, two_j2, two_m2, two_j, two_m);
        self.cg_cache.insert(key, v);
        v
    }

    fn cg_exact(
        &mut self,
        two_j1: i32,
        two_m1: i32,
        two_j2: i32,
        two_m2: i32,
        two_j: i32,
        two_m: i32,
    ) -> f64 {
        if two_m1 + two_m2 != two_m {
            return 0.0;
        }
        if !((two_j1 - two_j2).abs() <= two_j && two_j <= two_j1 + two_j2) {
            return 0.0;
        }
        if (two_j1 + two_j2 + two_j) & 1 != 0 {
            return 0.0;
        }
        if !(two_m1.abs() <= two_j1 && two_m2.abs() <= two_j2 && two_m.abs() <= two_j) {
            return 0.0;
        }
        if (two_j1 + two_m1) & 1 != 0 || (two_j2 + two_m2) & 1 != 0 || (two_j + two_m) & 1 != 0 {
            return 0.0;
        }

        let h = |x: i32| x / 2;

        let d1 = self.fact(h(two_j1 + two_j2 - two_j) as u32);
        let d2 = self.fact(h(two_j1 - two_j2 + two_j) as u32);
        let d3 = self.fact(h(-two_j1 + two_j2 + two_j) as u32);
        let d4 = self.fact((h(two_j1 + two_j2 + two_j) + 1) as u32);
        let delta2 = BigRational::new(d1 * d2 * d3, d4);

        let g1 = self.fact(h(two_j1 + two_m1) as u32);
        let g2 = self.fact(h(two_j1 - two_m1) as u32);
        let g3 = self.fact(h(two_j2 + two_m2) as u32);
        let g4 = self.fact(h(two_j2 - two_m2) as u32);
        let g5 = self.fact(h(two_j + two_m) as u32);
        let g6 = self.fact(h(two_j - two_m) as u32);
        let intpart: BigInt = BigInt::from(two_j + 1) * g1 * g2 * g3 * g4 * g5 * g6;
        let pre2 = delta2 * BigRational::from(intpart);

        let tmin = 0
            .max(h(two_j2 - two_j - two_m1))
            .max(h(two_j1 + two_m2 - two_j));
        let tmax = h(two_j1 + two_j2 - two_j)
            .min(h(two_j1 - two_m1))
            .min(h(two_j2 + two_m2));
        let mut s = BigRational::from(BigInt::from(0));
        let mut t = tmin;
        while t <= tmax {
            let e1 = self.fact(t as u32);
            let e2 = self.fact((h(two_j1 + two_j2 - two_j) - t) as u32);
            let e3 = self.fact((h(two_j1 - two_m1) - t) as u32);
            let e4 = self.fact((h(two_j2 + two_m2) - t) as u32);
            let e5 = self.fact((h(two_j - two_j2 + two_m1) + t) as u32);
            let e6 = self.fact((h(two_j - two_j1 - two_m2) + t) as u32);
            let den: BigInt = e1 * e2 * e3 * e4 * e5 * e6;
            let sign = if t & 1 == 1 { -1i32 } else { 1i32 };
            s += BigRational::new(BigInt::from(sign), den);
            t += 1;
        }

        let cg2 = &pre2 * (&s * &s);
        let mag = cg2.to_f64().expect("CG² to f64");
        debug_assert!(mag.is_finite());
        let sign = if s.is_negative() { -1.0 } else { 1.0 };
        sign * mag.sqrt()
    }

    #[allow(dead_code)]
    fn fact(&mut self, n: u32) -> BigInt {
        let n = n as usize;
        if self.fact_cache.is_empty() {
            self.fact_cache.push(BigInt::from(1u32));
        }
        while self.fact_cache.len() <= n {
            let k = self.fact_cache.len() as u32;
            let next = self.fact_cache.last().unwrap() * BigInt::from(k);
            self.fact_cache.push(next);
        }
        self.fact_cache[n].clone()
    }
}
