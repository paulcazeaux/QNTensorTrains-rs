// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! Spin-adapted overlap environments — the twin of [`crate::solvers::overlap_env`], mirrored step
//! for step, with ONE deliberate difference: the weight of the left step (R1 below).

use crate::scalar::Scalar;
use crate::su2::coreops::{adjoint_core_times_core, mul_sketch_left, mul_sketch_right, rproject};
use crate::su2::frame::Su2Sketch;
use crate::su2::qn::Su2Sector;
use crate::su2::twosite::{Su2ContractedCores, contract};
use crate::su2::vector::Su2TtVector;

/// The two overlap stacks between a swept train and a fixed one, indexed by BOND (`0..=d`).
#[derive(Clone)]
pub struct OverlapEnvs<T: Scalar> {
    /// `l[b]`: blocks `rank_x(b) × rank_psi(b)`, folded from sites `0..b`.
    pub l: Vec<Option<Su2Sketch<T>>>,
    /// `r[b]`: blocks `rank_psi(b) × rank_x(b)`, folded from sites `b..d`.
    pub r: Vec<Option<Su2Sketch<T>>>,
}

/// Both stacks in their driver-entry state: `l[0]` and `r[d]` the boundary identities, `r[1..d]`
/// folded right to left, `l[1..]` and `r[0]` left `None`.
///
/// # Panics
/// On a space mismatch, or when either train lacks unit boundary ranks.
pub fn overlap_envs<T: Scalar>(x: &Su2TtVector<T>, psi: &Su2TtVector<T>) -> OverlapEnvs<T> {
    assert_eq!(
        x.space, psi.space,
        "overlap_envs: the swept train and the fixed one must share a Su2Space"
    );
    unit_boundaries(x, "x");
    unit_boundaries(psi, "psi");
    let space = x.space;
    let d = space.d;
    let mut l: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    let mut r: Vec<Option<Su2Sketch<T>>> = vec![None; d + 1];
    l[0] = Some(Su2Sketch::identity(space, 0, 1));
    r[d] = Some(Su2Sketch::identity(space, d, 1));
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
    x: &Su2TtVector<T>,
    psi: &Su2TtVector<T>,
    l_k: &Su2Sketch<T>,
    k: usize,
) -> Su2Sketch<T> {
    debug_assert_eq!(l_k.k, k, "overlap_step_right: l_k must live on bond k");
    let t = mul_sketch_left(l_k, psi.core(k));
    adjoint_core_times_core(x.core(k), &t)
}

/// One step leftward: fold site `k` into the right stack, producing `r[k]` from `r[k+1]`.
///
/// # Panics
/// (Debug) when `r_k1` does not live on bond `k+1`.
pub fn overlap_step_left<T: Scalar>(
    x: &Su2TtVector<T>,
    psi: &Su2TtVector<T>,
    r_k1: &Su2Sketch<T>,
    k: usize,
) -> Su2Sketch<T> {
    debug_assert_eq!(
        r_k1.k,
        k + 1,
        "overlap_step_left: r_k1 must live on bond k+1"
    );
    let t = mul_sketch_right(psi.core(k), r_k1);
    rproject(&t, x.core(k))
}

/// `psi`'s image in the two-site local space of the pair `(k, k+1)`:
/// `contract(l[k] · psi.core(k), psi.core(k+1) · r[k+2])`.
///
/// # Panics
/// When `l[k]` or `r[k+2]` has not been built.
pub fn local_overlap_vector<T: Scalar>(
    env: &OverlapEnvs<T>,
    psi: &Su2TtVector<T>,
    k: usize,
) -> Su2ContractedCores<T> {
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

fn unit_boundaries<T: Scalar>(v: &Su2TtVector<T>, who: &str) {
    let space = v.space;
    assert_eq!(
        v.rank(0, Su2Sector::new(0, 0)),
        1,
        "overlap_envs needs unit rank at bond 0 ({who})"
    );
    assert_eq!(
        v.rank(
            space.d,
            Su2Sector::with_irrep(space.n, space.two_s, space.target)
        ),
        1,
        "overlap_envs needs unit rank at bond d ({who})"
    );
}
