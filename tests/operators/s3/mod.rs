// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::su2::expand;
use qntt::su2::hamiltonian::{
    Su2SparseHamiltonian, rayleigh_quotient as su2_rayleigh_quotient, xt_h_y as su2_xt_h_y,
};
use qntt::su2::qn::Su2Space;
use qntt::su2::spinalgebra::{SpinAlgebra, dense_term};
use qntt::su2::tensorops::adag_i_a_j as su2_adag_i_a_j;
use qntt::su2::vector::Su2TtVector;

use qntt::{
    C64, CoeffTensor, Orbital, QnSpace, Scalar, SparseHamiltonian, Spin, TtVector,
    adag_i_a_j as ab_adag_i_a_j, dot as ab_dot,
};

use qntt_testkit::randomized_su2_state;
use rand::{Rng, RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn seeded(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn randn_matrix<T: Scalar, R: Rng + ?Sized>(d: usize, rng: &mut R) -> CoeffTensor<T> {
    let mut t = CoeffTensor::<T>::zeros(d, 2);
    for i in 0..d {
        for j in 0..d {
            t.set(&[i, j], T::from_f64(rng.sample(rand_distr::StandardNormal)));
        }
    }
    t
}

fn randn_4tensor<T: Scalar, R: Rng + ?Sized>(d: usize, rng: &mut R) -> CoeffTensor<T> {
    let mut v = CoeffTensor::<T>::zeros(d, 4);
    for i in 0..d {
        for j in 0..d {
            for k in 0..d {
                for l in 0..d {
                    v.set(
                        &[i, j, k, l],
                        T::from_f64(rng.sample(rand_distr::StandardNormal)),
                    );
                }
            }
        }
    }
    v
}

fn zeros_matrix<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 2)
}

fn zeros_4tensor<T: Scalar>(d: usize) -> CoeffTensor<T> {
    CoeffTensor::<T>::zeros(d, 4)
}

fn dense_term_norm(q: usize, tokens: &[(usize, bool)]) -> f64 {
    let d = dense_term(q, tokens);
    d.values().map(|v| v * v).sum::<f64>().sqrt()
}

mod one_body;
mod spinalgebra;
mod tensorops;
