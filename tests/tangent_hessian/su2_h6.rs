// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::qn::Su2Space;
use qntt::su2::riemannian::{normalize_mut, tangent_bases};
use qntt::su2::vector::Su2TtVector;
use qntt::su2::weingarten::tangent_basis;
use qntt_testkit::randomized_su2_state;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

#[test]
fn h6_tangent_basis_span_is_scale_free() {
    use qntt::su2::weingarten::tangent_dim;
    for (d, nelec, two_s, eps, exact) in [
        (4usize, 3usize, 1usize, 1e-4f64, true),
        (4, 3, 1, 1e-6, true),
        (4, 3, 1, 1e-10, false),
    ] {
        let space = Su2Space::new(d, nelec, two_s);
        let count_at = |scale: f64| -> (usize, usize) {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD3AD_BEEF);
            let x1: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
            let mut x2: Su2TtVector<f64> = randomized_su2_state(space, 1, 1, &mut rng);
            x2.scale_mut(scale);
            let mut x = &x1 + &x2;
            normalize_mut(&mut x);
            let (base_l, base_r) = tangent_bases(&x);
            (
                tangent_basis(&base_l, &base_r, true).len(),
                tangent_dim(&base_l),
            )
        };
        let (n_ref, _) = count_at(1.0);
        let (n_eps, dim) = count_at(eps);
        assert!(
            n_eps < dim,
            "SU(2) d={d} eps={eps:.0e}: enumerator returned {} vectors (radial included), \
             tangent_dim is {dim}",
            n_eps + 1
        );
        assert!(n_eps > 1, "SU(2) d={d} eps={eps:.0e}: span collapsed");
        if exact {
            assert_eq!(
                n_eps, n_ref,
                "SU(2) d={d} eps={eps:.0e}: accepted count {n_eps} vs O(1)-weight reference \
                 {n_ref} — a resolvable direction was culled (span not scale-free)"
            );
        }
    }
}
