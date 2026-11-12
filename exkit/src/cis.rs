// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

pub use qntt::cis::*;

#[cfg(test)]
mod tests {
    use super::*;
    use qntt::{Orbital, Spin, adag_i_a_j, dot};

    #[test]
    fn cis_matrix_is_h_on_the_spin_adapted_singles() {
        let repo = crate::repo::Repo::from_manifest_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
        let fx = repo.load("lih_sto3g");
        let (d, nocc) = (fx.d(), fx.nelec() / 2);
        let (nv, h, hf) = (d - nocc, fx.ham(), fx.hf());
        let e_hf = dot(&hf, &(&h * &hf));
        for triplet in [false, true] {
            let sign = if triplet { -1.0 } else { 1.0 };
            let singles: Vec<_> = (0..nocc)
                .flat_map(|i| (nocc..d).map(move |a| (i, a)))
                .map(|(i, a)| {
                    let up = adag_i_a_j(
                        &hf,
                        Orbital::new(a, Spin::Up),
                        Orbital::new(i, Spin::Up),
                        1.0,
                    );
                    let dn = adag_i_a_j(
                        &hf,
                        Orbital::new(a, Spin::Dn),
                        Orbital::new(i, Spin::Dn),
                        sign,
                    );
                    &(&up + &dn) * std::f64::consts::FRAC_1_SQRT_2
                })
                .collect();
            let n = singles.len();
            assert_eq!(n, nocc * nv);
            let a = cis_matrix(fx.ints(), nocc, triplet);
            let mut worst = 0.0f64;
            for (p, sp) in singles.iter().enumerate() {
                let hsp = &h * sp;
                for (q, sq) in singles.iter().enumerate() {
                    let m = dot(sq, &hsp) - e_hf * dot(sq, sp);
                    worst = worst.max((m - a[q * n + p]).abs());
                }
            }
            println!(
                "lih_sto3g CIS {}: max |A − ⟨ia|H−E_HF|jb⟩| = {worst:.2e}",
                if triplet { "triplet" } else { "singlet" }
            );
            assert!(worst <= 1e-10, "CIS matrix vs TT singles: {worst:e}");
        }
    }
}
