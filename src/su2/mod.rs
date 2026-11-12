// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! SU(2)-symmetric ("spin-adapted") tensor trains.

#[cfg(test)]
pub(crate) mod border_fixture;
pub mod chop;
pub mod core;
pub mod coreops;
pub(crate) mod deflate;
#[cfg(test)]
mod deflate_tests;
pub mod expand;
pub mod fcidump;
pub(crate) mod flux;
pub mod frame;
pub mod hamiltonian;
pub mod hess;
pub mod joint;
#[doc(hidden)]
pub use crate::knobs;
pub mod layer;
pub mod ops;
pub mod orbital;
pub mod qn;
pub mod riemannian;
pub mod round;
pub(crate) mod shard;
pub mod solvers;
pub mod spinalgebra;
pub mod tangent;
pub mod tensorops;
pub mod transition1rdm;
pub mod twosite;
pub mod vector;
pub mod weingarten;

pub use crate::su2::expand::expand;
pub use crate::su2::vector::tt_csf;

/// `θ(2j_row, 2j_col) = √((2j_col+1)/(2j_row+1))` — the right-canonical horizontal-unfolding
/// weight. Used only by `scale_horizontal`/`unscale_horizontal` (the θ-weighted factorization
/// unfoldings) and the θ²-weighted `rproject` dual pairing. Do NOT use this to "symmetrize" the
/// UNWEIGHTED `dot`/`expand` chain — that is a silent O(1) error.
pub fn theta(two_j_row: usize, two_j_col: usize) -> f64 {
    ((two_j_col + 1) as f64 / (two_j_row + 1) as f64).sqrt()
}

/// `centerweight(2j_col, 2J_total) = (2j_col+1)/(2J_total+1)` — the column-sector weight of a
/// center matrix/core in the fixed-Sz-member metric — the weight the orthogonal-center-core norm
/// shortcut uses. Do NOT use this to weight the UNWEIGHTED `dot`/`expand` chain.
pub fn centerweight(two_j_col: usize, two_j_total: usize) -> f64 {
    (two_j_col + 1) as f64 / (two_j_total + 1) as f64
}

#[cfg(test)]
mod spg3_tests {
    use super::{centerweight, theta};

    #[test]
    fn spg3_h4_theta_centerweight_identity_sweep_documentation() {
        for total in 0..=6usize {
            for a in 0..=total {
                for b in 0..=total {
                    let lhs = centerweight(a, total) * theta(a, b).powi(2);
                    let rhs = centerweight(b, total);
                    assert!(
                        (lhs - rhs).abs() < 1e-13,
                        "total={total} a={a} b={b}: centerweight(a)*theta(a,b)^2 != \
                         centerweight(b)"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod s1_tests {
    use super::{centerweight, theta};

    #[test]
    fn theta_and_centerweight_numeric_spot_values() {
        assert!((theta(3, 3) - 1.0).abs() < 1e-15);
        assert!((theta(1, 3) - 2.0_f64.sqrt()).abs() < 1e-14);
        assert!((centerweight(0, 0) - 1.0).abs() < 1e-15);
        assert!((centerweight(2, 4) - 0.6).abs() < 1e-14);
    }
}
