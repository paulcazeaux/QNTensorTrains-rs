// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

use std::collections::BTreeMap;

use super::helpers::{mat4_mul, mat4_neg, pow4};
use super::{Coo, Mat4, TensorOp};

const CDAG_UP: Mat4 = [
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];
const CDAG_DN: Mat4 = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
];
const C_UP: Mat4 = [
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0],
];
const C_DN: Mat4 = [
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, -1.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];
const PARITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];
const ID4: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

pub(super) const OCC_SPIN: [i32; 3] = [0, 1, 0];
pub(super) const OCC_STATES: [&[usize]; 3] = [&[0], &[1, 2], &[3]];

/// `TENSOR_CDAG` — the elementary creation tensor (rank-½, `flux=+1`, `odd=true`, `components =
/// [_CDAG_UP, _CDAG_DN]` μ-descending).
pub fn tensor_cdag() -> TensorOp {
    TensorOp {
        two_k: 1,
        flux: 1,
        odd: true,
        components: vec![CDAG_UP, CDAG_DN],
    }
}

/// `TENSOR_A` — the elementary tilde-annihilation tensor (rank-½, `flux=-1`, `odd=true`,
/// `components = [-_C_DN, _C_UP]`). **Tilde convention** (keep the comment verbatim):
/// `ã_q = (-1)^(1/2+q) a_{-q}`, i.e. `(ã_{+1/2}, ã_{-1/2}) = (-a_dn,
/// a_up)`.
pub fn tensor_a() -> TensorOp {
    TensorOp {
        two_k: 1,
        flux: -1,
        odd: true,
        components: vec![mat4_neg(&C_DN), C_UP],
    }
}

/// `elementary_tensor(cdag)` — dispatch to [`tensor_cdag`]/[`tensor_a`].
pub fn elementary_tensor(cdag: bool) -> TensorOp {
    if cdag { tensor_cdag() } else { tensor_a() }
}

/// The rank-0 identity site tensor `TensorOp(0, 0, false, [_ID4])` — the operand of the `_ID_RED =
/// reduce_siteop(TensorOp(0,0,false,[_ID4]))`. Used by the elementary-operator engine as the
/// transport site op on every bond off the operator site.
pub fn tensor_identity() -> TensorOp {
    TensorOp {
        two_k: 0,
        flux: 0,
        odd: false,
        components: vec![ID4],
    }
}

/// Spin-summed term on `q` sites (`dense_term`). `tokens = (site ∈ 0..q,
/// cdag)` in operator order; the SPIN-PAIRING BY ROLE rule (2/4/6-token role pairing) determines
/// which token indices share a spin-summation variable. `panic!` on any other token count.
pub fn dense_term(q: usize, tokens: &[(usize, bool)]) -> Coo {
    let mut d: Coo = BTreeMap::new();
    match tokens.len() {
        2 => {
            for &sigma in &[true, false] {
                accumulate_term(&mut d, q, tokens, &[sigma, sigma]);
            }
        }
        4 => {
            for &sigma in &[true, false] {
                for &sigmap in &[true, false] {
                    accumulate_term(&mut d, q, tokens, &[sigma, sigmap, sigmap, sigma]);
                }
            }
        }
        6 => {
            for &sigma in &[true, false] {
                for &sigmap in &[true, false] {
                    for &sigmapp in &[true, false] {
                        accumulate_term(
                            &mut d,
                            q,
                            tokens,
                            &[sigma, sigmap, sigmapp, sigmapp, sigmap, sigma],
                        );
                    }
                }
            }
        }
        _ => panic!("tokens must have length 2, 4, or 6"),
    }
    d
}

/// The single-mode operator matrix for a role's `(cdag, up)` assignment: `cdag ? (up ? _CDAG_UP : _CDAG_DN) : (up ? _C_UP : _C_DN)`.
pub fn site_op_matrix(cdag: bool, up: bool) -> Mat4 {
    if cdag {
        if up { CDAG_UP } else { CDAG_DN }
    } else if up {
        C_UP
    } else {
        C_DN
    }
}

/// Accumulate one spin assignment's JW-factorized-over-sites contribution into the dense reference
/// `D`. Builds the per-site operator-product `F` (JW `PARITY` strictly left of each token's own
/// site, identity right), then dispatches to [`kron_accum`].
pub fn accumulate_term(d: &mut Coo, q: usize, tokens: &[(usize, bool)], spins: &[bool]) {
    let mut f = vec![ID4; q];
    for (t, &(p, cdag)) in tokens.iter().enumerate() {
        let m = site_op_matrix(cdag, spins[t]);
        #[allow(clippy::needless_range_loop)]
        for s in 0..=p {
            let factor = if s == p { m } else { PARITY };
            f[s] = mat4_mul(&f[s], &factor);
        }
    }
    kron_accum(d, &f, q, 0, 0, 0, 1.0);
}

/// Recursive Kronecker-accumulate: `amp·(⊗_{s=q-1..0} F_s)`
/// into `D`, site `0` the FASTEST determinant digit (matches TT/determinant ordering). Digit weight
/// `4^p` (0-based `p`).
pub fn kron_accum(d: &mut Coo, f: &[Mat4], q: usize, p: usize, dout: usize, din: usize, amp: f64) {
    if p == q {
        *d.entry((dout, din)).or_insert(0.0) += amp;
        return;
    }
    let w4 = pow4(p);
    for s_in in 0..4 {
        for s_out in 0..4 {
            let c = f[p][s_out][s_in];
            if c == 0.0 {
                continue;
            }
            kron_accum(d, f, q, p + 1, dout + s_out * w4, din + s_in * w4, amp * c);
        }
    }
}

/// Electron count of a single 4-dim determinant digit: `[0,1,1,2][s]`, `s ∈ 0..4`.
#[inline]
pub fn occ_count(s: usize) -> usize {
    const TABLE: [usize; 4] = [0, 1, 1, 2];
    TABLE[s]
}
