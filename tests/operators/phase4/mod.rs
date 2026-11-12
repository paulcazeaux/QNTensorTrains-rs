// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use qntt::{
    C64, CoeffTensor, Family, Orbital, QnSpace, RoundMode, Scalar, SparseHamiltonian, Spin,
    TtVector, adag_i_a_j, adag_i_adag_k_a_l_a_j, adag_i_adag_k_adag_m_a_n_a_l_a_j, dot,
    rayleigh_quotient, tt_state,
};
use qntt_testkit::dense_apply_string;
use qntt_testkit::randomized_state;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

mod stage1;
mod stage2;
mod stage3;

fn state_from_sites<T: Scalar>(d: usize, up_sites: &[usize], dn_sites: &[usize]) -> TtVector<T> {
    let mut occ_up = vec![false; d];
    let mut occ_dn = vec![false; d];
    for &s in up_sites {
        occ_up[s] = true;
    }
    for &s in dn_sites {
        occ_dn[s] = true;
    }
    tt_state(&occ_up, &occ_dn)
}

fn dense_norm<T: Scalar>(x: &[T]) -> f64 {
    x.iter().map(|v| v.abs_s() * v.abs_s()).sum::<f64>().sqrt()
}

fn dense_diff_norm<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x - *y;
            d.abs_s() * d.abs_s()
        })
        .sum::<f64>()
        .sqrt()
}

fn rel_err<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    dense_diff_norm(a, b) / dense_norm(b)
}

fn det_index(d: usize, up_site: usize, dn_site: usize) -> usize {
    let mut idx = 0usize;
    for k in (0..d).rev() {
        let fam = ((k == up_site) as usize) | (((k == dn_site) as usize) << 1);
        idx = (idx << 2) | fam;
    }
    idx
}
