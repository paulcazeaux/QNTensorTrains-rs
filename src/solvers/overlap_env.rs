// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Overlap environments between two trains, and the two-site vector that represents a fixed state
//! inside a local eigenproblem.

use crate::core::factor::{
    Sketch, adjoint_core_times_core, core_times_adjoint_core, mul_sketch_left, mul_sketch_right,
};
use crate::core::twosite::{ContractedSparseCores, contract};
use crate::qn::Sector;
use crate::scalar::Scalar;
use crate::vector::TtVector;

/// The two overlap stacks between a swept train and a fixed one, indexed by BOND (`0..=d`).
#[derive(Clone)]
pub struct OverlapEnvs<T: Scalar> {
    /// `l[b]`: blocks `rank_x(b) × rank_psi(b)`, folded from sites `0..b`.
    pub l: Vec<Option<Sketch<T>>>,
    /// `r[b]`: blocks `rank_psi(b) × rank_x(b)`, folded from sites `b..d`.
    pub r: Vec<Option<Sketch<T>>>,
}

/// Both stacks in their driver-entry state: `l[0]` and `r[d]` the boundary identities, `r[1..d]`
/// folded right to left, `l[1..]` and `r[0]` left `None`.
///
/// # Panics
/// On a space mismatch, or when either train lacks unit boundary ranks (the closure would then not
/// be a scalar).
pub fn overlap_envs<T: Scalar>(x: &TtVector<T>, psi: &TtVector<T>) -> OverlapEnvs<T> {
    assert_eq!(
        x.space, psi.space,
        "overlap_envs: the swept train and the fixed one must share a QnSpace"
    );
    unit_boundaries(x, "x");
    unit_boundaries(psi, "psi");
    let space = x.space;
    let d = space.d;
    let mut l: Vec<Option<Sketch<T>>> = vec![None; d + 1];
    let mut r: Vec<Option<Sketch<T>>> = vec![None; d + 1];
    l[0] = Some(Sketch::identity(space, 0, 1));
    r[d] = Some(Sketch::identity(space, d, 1));
    for k in (1..d).rev() {
        let next = r[k + 1].as_ref().expect("r is built right to left");
        r[k] = Some(overlap_step_left(x, psi, next, k));
    }
    OverlapEnvs { l, r }
}

/// One step rightward: fold site `k` into the left stack, producing `l[k+1]` from `l[k]`.
///
/// # Panics
/// (Debug) when `l_k` does not live on bond `k`.
pub fn overlap_step_right<T: Scalar>(
    x: &TtVector<T>,
    psi: &TtVector<T>,
    l_k: &Sketch<T>,
    k: usize,
) -> Sketch<T> {
    debug_assert_eq!(l_k.k, k, "overlap_step_right: l_k must live on bond k");
    let t = mul_sketch_left(l_k, psi.core(k));
    adjoint_core_times_core(x.core(k), &t)
}

/// One step leftward: fold site `k` into the right stack, producing `r[k]` from `r[k+1]`.
///
/// # Panics
/// (Debug) when `r_k1` does not live on bond `k+1`.
pub fn overlap_step_left<T: Scalar>(
    x: &TtVector<T>,
    psi: &TtVector<T>,
    r_k1: &Sketch<T>,
    k: usize,
) -> Sketch<T> {
    debug_assert_eq!(
        r_k1.k,
        k + 1,
        "overlap_step_left: r_k1 must live on bond k+1"
    );
    let t = mul_sketch_right(psi.core(k), r_k1);
    core_times_adjoint_core(&t, x.core(k))
}

/// `psi`'s image in the two-site local space of the pair `(k, k+1)`:
/// `contract(l[k] · psi.core(k), psi.core(k+1) · r[k+2])`.
///
/// # Panics
/// When `l[k]` or `r[k+2]` has not been built.
pub fn local_overlap_vector<T: Scalar>(
    env: &OverlapEnvs<T>,
    psi: &TtVector<T>,
    k: usize,
) -> ContractedSparseCores<T> {
    let lk = env.l[k]
        .as_ref()
        .unwrap_or_else(|| panic!("local_overlap_vector: l[{k}] has not been built"));
    let rk2 = env.r[k + 2]
        .as_ref()
        .unwrap_or_else(|| panic!("local_overlap_vector: r[{}] has not been built", k + 2));
    let left = mul_sketch_left(lk, psi.core(k));
    let right = mul_sketch_right(psi.core(k + 1), rk2);
    contract(&left, &right)
}

fn unit_boundaries<T: Scalar>(v: &TtVector<T>, who: &str) {
    let space = v.space;
    assert_eq!(
        v.rank(0, Sector::new(0, 0)),
        1,
        "overlap_envs needs unit rank at bond 0 ({who})"
    );
    assert_eq!(
        v.rank(
            space.d,
            Sector::with_irrep(space.nup, space.ndn, space.target)
        ),
        1,
        "overlap_envs needs unit rank at bond d ({who})"
    );
}
