// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! The SU(2) sweep drivers — ALS (one-site), MALS (two-site, rank-growing), MALS with local mode
//! rotations ([`moderot`]) and TDVP (imaginary + real time), with the multi-root and block-solve
//! drivers beside them. Duplicated per layer, like `su2::riemannian`: the algorithm is
//! generic in the two layer traits, but its kernel surface —
//! `Su2OneSiteWorkspace`/`Su2TwoSiteWorkspace`, the `sketching_step_*` pair, `factor_svd`'s
//! θ/centerweight-live truncation — is layer-local in Rust and would need a 4+-associated-type
//! trait to genericize. The abelian `src/solvers/{als,mals,moderot,tdvp}.rs` are mirrored
//! line-for-line where the two agree.

pub mod als;
pub mod excited;

pub mod envstore;
pub mod mals;
pub mod moderot;
pub mod multistate;
pub mod overlap_env;
pub mod ssmals;
pub mod tdvp;
