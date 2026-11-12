// Copyright (C) 2023-2026 Paul Cazeaux
//
// This file is part of qntt. qntt is free software: you can
// redistribute it and/or modify it under the terms of the GNU
// General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your
// option) any later version. See LICENSE for details.

//! `int32_t` ↔ core-crate enum mappings. The numeric values are part of the ABI: they are
//! mirrored in `include/qntt.h` and in `julia/src/options.jl`, and must never be renumbered.

use qntt::PointGroupKind;
use qntt::fcidump::{ThreeBodySymmetry, TwoBodySymmetry};
use qntt::hamiltonian::Construction;
use qntt::round::RoundMode;
use qntt::solvers::excited::SsStart;
use qntt::solvers::hess::{HessPart, PdVerdict};
use qntt::solvers::mals::NoiseType;
use qntt::solvers::riemannian::CgVariant;
use qntt::solvers::ssmals::{PenaltyRule, SsMode};

use crate::err::Error;

/// `RoundMode`: 0 = absolute error budget, 1 = relative.
pub fn round_mode(v: i32) -> Result<RoundMode, Error> {
    match v {
        0 => Ok(RoundMode::Abs),
        1 => Ok(RoundMode::Rel),
        _ => Err(Error::invalid(format!(
            "round mode {v} (want 0=abs, 1=rel)"
        ))),
    }
}

/// Operator construction: 0 = layer default, 1 = median path-threading, 2 = bipartite (the
/// abelian default), 3 = bipartite per-sector. `0` maps to `None`, i.e. "call the default
/// constructor" — the only value the SU(2) layer accepts.
pub fn construction(v: i32) -> Result<Option<Construction>, Error> {
    match v {
        0 => Ok(None),
        1 => Ok(Some(Construction::Median)),
        2 => Ok(Some(Construction::Bipartite)),
        3 => Ok(Some(Construction::BipartitePerSector)),
        _ => Err(Error::invalid(format!(
            "construction {v} (want 0=default, 1=median, 2=bipartite, 3=bipartite-per-sector)"
        ))),
    }
}

/// FCIDUMP two-body completion group: 0 = none, 1 = pair-swap, 2 = eightfold.
pub fn two_body_symmetry(v: i32) -> Result<TwoBodySymmetry, Error> {
    match v {
        0 => Ok(TwoBodySymmetry::None),
        1 => Ok(TwoBodySymmetry::Pairswap),
        2 => Ok(TwoBodySymmetry::Eightfold),
        _ => Err(Error::invalid(format!(
            "two-body symmetry {v} (want 0=none, 1=pairswap, 2=eightfold)"
        ))),
    }
}

/// TCDUMP three-body completion group: 0 = none, 1 = pair-permutation, 2 = full 48-fold.
pub fn three_body_symmetry(v: i32) -> Result<ThreeBodySymmetry, Error> {
    match v {
        0 => Ok(ThreeBodySymmetry::None),
        1 => Ok(ThreeBodySymmetry::Pairperm),
        2 => Ok(ThreeBodySymmetry::Full48),
        _ => Err(Error::invalid(format!(
            "three-body symmetry {v} (want 0=none, 1=pairperm, 2=full48)"
        ))),
    }
}

/// Conjugate-gradient variant: 0 = Fletcher–Reeves, 1 = Polak–Ribière (the default).
pub fn cg_variant(v: i32) -> Result<CgVariant, Error> {
    match v {
        0 => Ok(CgVariant::Fr),
        1 => Ok(CgVariant::Pr),
        _ => Err(Error::invalid(format!(
            "cg variant {v} (want 0=fletcher-reeves, 1=polak-ribiere)"
        ))),
    }
}

/// MALS noise flavour: 0 = random, 1 = perturbative (White density-matrix noise, the default).
pub fn noise_type(v: i32) -> Result<NoiseType, Error> {
    match v {
        0 => Ok(NoiseType::Random),
        1 => Ok(NoiseType::Perturbative),
        _ => Err(Error::invalid(format!(
            "noise type {v} (want 0=random, 1=perturbative)"
        ))),
    }
}

/// Wire code for a [`NoiseType`] — the inverse of [`noise_type`], so the ABI default derives from
/// the single Rust source of truth ([`qntt::solvers::mals::MalsOptions`]'s `Default`) rather
/// than hardcoding a code.
pub fn noise_type_code(nt: NoiseType) -> i32 {
    match nt {
        NoiseType::Random => 0,
        NoiseType::Perturbative => 1,
    }
}

/// Abelian point-group kind: `0 = C1, 1 = Ci, 2 = C2, 3 = Cs, 4 = D2, 5 = C2v, 6 = C2h, 7 = D2h`.
pub fn pg_kind(v: i32) -> Result<PointGroupKind, Error> {
    match v {
        0..=7 => Ok(PointGroupKind::ALL[v as usize]),
        _ => Err(Error::invalid(format!(
            "point group kind {v} (want 0=C1, 1=Ci, 2=C2, 3=Cs, 4=D2, 5=C2v, 6=C2h, 7=D2h)"
        ))),
    }
}

/// [`pg_kind`] widened by the reader-only sentinel `-1` (`QNTT_PG_AUTO`) = "guess the kind from
/// `max(ORBSYM)`", which maps to the core reader's `kind: Option<PointGroupKind>` being `None`.
pub fn pg_kind_opt(v: i32) -> Result<Option<PointGroupKind>, Error> {
    match v {
        -1 => Ok(None),
        _ => pg_kind(v).map(Some),
    }
}

/// Wire code for a [`PointGroupKind`] — the inverse of [`pg_kind`], so a resolved kind crosses
/// back out through the single source of truth rather than a hardcoded table.
pub fn pg_kind_code(k: PointGroupKind) -> i32 {
    PointGroupKind::ALL
        .iter()
        .position(|&x| x == k)
        .expect("PointGroupKind::ALL is exhaustive") as i32
}

/// Which Hessian part a PD probe targets: 0 = `Rr` (`A + B`, the real-layer energy Hessian),
/// 1 = `Ii` (`A − B`, curvature along the imaginary directions).
pub fn hess_part(v: i32) -> Result<HessPart, Error> {
    match v {
        0 => Ok(HessPart::Rr),
        1 => Ok(HessPart::Ii),
        _ => Err(Error::invalid(format!(
            "Hessian part {v} (want 0=rr (A+B), 1=ii (A-B))"
        ))),
    }
}

/// Wire code for a [`PdVerdict`]: 0 = witness-negative (PROVEN indefinite), 1 = likely-PD
/// (converged, no negative direction), 2 = inconclusive (budget exhausted — **not** a
/// positive-definiteness claim).
pub fn pd_verdict_code(v: &PdVerdict) -> i32 {
    match v {
        PdVerdict::WitnessNegative { .. } => 0,
        PdVerdict::LikelyPd { .. } => 1,
        PdVerdict::Inconclusive { .. } => 2,
    }
}

/// [`PdVerdict::is_pd`]'s three-way answer as a wire code: `1` PD-certified, `0` proven
/// indefinite, `-1` inconclusive.
pub fn pd_is_pd_code(v: &PdVerdict) -> i32 {
    match v.is_pd() {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    }
}

/// Block family index: 0 = empty, 1 = up, 2 = down, 3 = doubly occupied. On the SU(2) layer 1/2
/// are the `S ± ½` genealogical coupling branches, NOT electron spins (see `src/su2/mod.rs`).
pub fn family(v: i32) -> Result<qntt::core::Family, Error> {
    match v {
        0..=3 => Ok(qntt::core::Family::ALL[v as usize]),
        _ => Err(Error::invalid(format!("family {v} (want 0..=3)"))),
    }
}

/// State-specific route: 0 = penalty (the default), 1 = project.
pub fn ss_mode(v: i32) -> Result<SsMode, Error> {
    match v {
        0 => Ok(SsMode::Penalty),
        1 => Ok(SsMode::Project),
        _ => Err(Error::invalid(format!(
            "ss mode {v} (want 0=penalty, 1=project)"
        ))),
    }
}

/// Wire code for an [`SsMode`] — the inverse of [`ss_mode`], so the ABI default derives from the
/// core crate's own `Default` rather than a hardcoded code.
pub fn ss_mode_code(m: SsMode) -> i32 {
    match m {
        SsMode::Penalty => 0,
        SsMode::Project => 1,
    }
}

/// [`PenaltyRule`] from its `(kind, value)` wire pair: `0` = auto (the default; `value` is a
/// placeholder and is IGNORED, not validated), `1` = fixed at `value`.
pub fn penalty_rule(kind: i32, value: f64) -> Result<PenaltyRule, Error> {
    match kind {
        0 => Ok(PenaltyRule::Auto),
        1 => {
            if !value.is_finite() || value < 0.0 {
                return Err(Error::invalid(format!(
                    "state_specific_mals: PenaltyRule::Fixed({value}) must be finite and \
                     non-negative"
                )));
            }
            Ok(PenaltyRule::Fixed(value))
        }
        _ => Err(Error::invalid(format!(
            "penalty rule {kind} (want 0=auto, 1=fixed)"
        ))),
    }
}

/// Wire pair for a [`PenaltyRule`] — the inverse of [`penalty_rule`]. `Auto` carries the
/// placeholder `0.0`, which the forward direction ignores.
pub fn penalty_rule_code(r: PenaltyRule) -> (i32, f64) {
    match r {
        PenaltyRule::Auto => (0, 0.0),
        PenaltyRule::Fixed(w) => (1, w),
    }
}

/// Where each state-specific sweep of the ladder starts: 0 = that root's own state-average state
/// (the default). Code 1 is reserved and refused by builds that do not provide it.
pub fn ss_start(v: i32) -> Result<SsStart, Error> {
    match v {
        0 => Ok(SsStart::Sa),
        1 => Err(Error::invalid(
            "ss start 1 is not available in this build".to_string(),
        )),
        _ => Err(Error::invalid(format!(
            "ss start {v} (want 0 = state-average state)"
        ))),
    }
}

/// Wire code for an [`SsStart`] — the inverse of [`ss_start`].
pub fn ss_start_code(s: SsStart) -> i32 {
    match s {
        SsStart::Sa => 0,
    }
}

/// Wire code for a [`CgVariant`] — the inverse of [`cg_variant`], so a derived default (the
/// ladder's polish CG) crosses out through the single source of truth.
pub fn cg_variant_code(v: CgVariant) -> i32 {
    match v {
        CgVariant::Fr => 0,
        CgVariant::Pr => 1,
    }
}

/// [`cg_variant`] at the SU(2) layer, whose CG-variant enum is a DIFFERENT type with the same two
/// spellings and the same wire codes.
pub fn su2_cg_variant(v: i32) -> Result<qntt::su2::riemannian::CgVariant, Error> {
    match v {
        0 => Ok(qntt::su2::riemannian::CgVariant::Fr),
        1 => Ok(qntt::su2::riemannian::CgVariant::Pr),
        _ => Err(Error::invalid(format!(
            "cg variant {v} (want 0=fletcher-reeves, 1=polak-ribiere)"
        ))),
    }
}
