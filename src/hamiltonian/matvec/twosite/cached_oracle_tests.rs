// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use super::SparseHamiltonian;
use crate::core::twosite::{ContractedSparseCores, contract, factor_qc};
use crate::qn::{QnSpace, RankGrid};
use crate::solvers::als::{left_to_right_sketching, right_to_left_sketching};
use crate::vector::TtVector;
use crate::{CoeffTensor, Scalar};

fn randomized_state<T: Scalar>(
    space: QnSpace,
    lo: usize,
    hi: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> TtVector<T> {
    let d = space.d;
    let ranks: Vec<RankGrid> = (0..=d)
        .map(|b| {
            let mut r = RankGrid::zeros(&space);
            for s in space.bond_sectors(b) {
                r[s] = if b == 0 || b == d {
                    1
                } else {
                    rng.random_range(lo..=hi)
                };
            }
            r
        })
        .collect();
    crate::tt_randn(space, &ranks, rng)
}

fn assert_blocks_exact(a: &ContractedSparseCores<f64>, b: &ContractedSparseCores<f64>, ctx: &str) {
    assert_eq!(a.mid_qn, b.mid_qn, "{ctx}: mid_qn sector list mismatch");
    for &m in &a.mid_qn {
        match (a.block(m), b.block(m)) {
            (Some(pa), Some(pb)) => {
                assert_eq!(
                    (pa.nrows(), pa.ncols()),
                    (pb.nrows(), pb.ncols()),
                    "{ctx}: block shape mismatch at mid {m:?}"
                );
                for j in 0..pa.ncols() {
                    for i in 0..pa.nrows() {
                        assert_eq!(
                            *pa.get(i, j),
                            *pb.get(i, j),
                            "{ctx}: cached vs uncached differ at mid {m:?} ({i},{j})"
                        );
                    }
                }
            }
            (None, None) => {}
            _ => panic!("{ctx}: block presence mismatch at mid {m:?}"),
        }
    }
}

#[test]
fn two_site_cached_matches_uncached_exact() {
    let d = 6usize;
    let space = QnSpace::new(d, 1, 1);
    let t = CoeffTensor::<f64>::zeros(d, 2);
    let mut v = CoeffTensor::<f64>::zeros(d, 4);
    v.set(&[2, 0, 3, 1], -1.0);
    v.set(&[1, 3, 0, 2], -1.0);
    let h = SparseHamiltonian::new(space, &t, &v, None, f64::EPSILON);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x8B_AD_F0_0D);
    let mut x: TtVector<f64> = randomized_state(space, 3, 5, &mut rng);
    x.right_orthogonalize(true);
    let k = 2usize;
    x.move_core(k, true);
    x.check();

    let fl = left_to_right_sketching(&h, &x).into_iter().nth(k).unwrap();
    let fr_all = right_to_left_sketching(&h, &x);
    let fr = &fr_all[k + 1];
    let cp0 = contract(x.core(k), x.core(k + 1));

    let (xk0, xk10) = factor_qc(&cp0);
    let il = h.row_ranges(&xk0);
    let jr = h.col_ranges(&xk10);

    let a0 = h.two_site_sketched_hamiltonian(&cp0, &fl, fr);
    let b0 = h.two_site_sketched_hamiltonian_cached(&cp0, &fl, fr, &il, &jr);
    assert_blocks_exact(&a0, &b0, "iterate cp0");

    let a1 = h.two_site_sketched_hamiltonian(&a0, &fl, fr);
    let b1 = h.two_site_sketched_hamiltonian_cached(&a0, &fl, fr, &il, &jr);
    assert_blocks_exact(&a1, &b1, "iterate H*cp0");
}
